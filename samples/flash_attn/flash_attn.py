# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Channel-first software-pipeline Flash Attention kernel.

Supports mask_mode==0 (full) and mask_mode==3 (right-context causal).
3-stage pipeline (QK -> softmax -> PV -> update), raw-mode Vector.
"""

from cannbotdsl.buffer import Buffer

import torch
from cannbotdsl import dtypes, select
from cannbotdsl.channel import Channel
from cannbotdsl.constexpr import const_expr
from cannbotdsl.delay_line import DelayLineGroup
from cannbotdsl.arch import get_subblock_id, get_block_idx, get_block_num

from cannbotdsl.jit_runner import jit
from cannbotdsl.kernel_launcher import kernel
from cannbotdsl.math import (
    matmul,
    cast,
)
from cannbotdsl.runtime import from_torch_npu
from cannbotdsl.typing.types import ChannelKind, MemLoc, Tensor
from cannbotdsl.tensor import (
    ceil_div,
    rebase_view,
    idx2crd,
    partition_view,
    local_slice,
    tile_view,
    make_copy_engine,
    make_layout,
    make_partition_tiler,
    mem_copy,
    _layout_op_wrapper,
)
from cannbotdsl.vf import vf
from cannbotdsl import raw_reg as rr

# Vector register width: 2048 bits / 32 bits per fp32 element = 64 elements per vector.
VL_T = 2048 // 32

# Mask threshold for detecting fully-masked rows (additive mask path).
MASK_NEG_THRESHOLD = -1e29

# Pipeline depth: 3-stage (QK -> softmax -> PV -> update).
PIPELINE_DEPTH = 3


def _clamp_nonneg(value: int) -> int:
    """max(0, value) for dynamic Int64. Prevents negative mask counts from
    wrapping to large unsigned values in update_mask."""
    return select(value < 0, 0, value)


def _clamp_range(value: int, hi: int) -> int:
    """clamp(value, 0, hi) for dynamic Int64. Used for per-row causal
    valid-column count: clamped into [0, actual_n]."""
    v = select(value < 0, 0, value)
    return select(v > hi, hi, v)


def _bsnd_to_bnsd(t: Tensor) -> Tensor:
    """BSND [B,S,N,D] -> logical BNSD [B,N,S,D] view (axis 1<->2 swap, pure metadata)."""

    def _swap_1_2(layout):
        shp, st = layout.shape, layout.stride
        return make_layout((shp[0], shp[2], shp[1], shp[3]), stride=(st[0], st[2], st[1], st[3]))

    return _layout_op_wrapper(_swap_1_2, t)


# ---- cube side: matmul + L0C->UB store, all Channels (channel-first) ----
class Matmul:
    def __init__(self, tile_cube_m, tile_n, tile_d, dtype_16):
        self.tile_cube_m = tile_cube_m
        self.tile_n = tile_n
        self.tile_d = tile_d

        self.nd2nz = make_copy_engine(format_transform="nd2nz", dtype=dtype_16, pad_value=0.0)
        self.fixpipe = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=1)

        tmp_n = max(tile_n, tile_d)
        self.q_l1 = Channel(MemLoc.L1, shape=(tile_cube_m, tile_d), dtype=dtype_16, depth=2)
        self.k_l1 = Channel(MemLoc.L1, shape=(tile_n, tile_d), dtype=dtype_16, depth=2)
        self.v_l1 = Channel(MemLoc.L1, shape=(tile_n, tile_d), dtype=dtype_16, depth=2)
        self.l0a = Channel(MemLoc.L0A, shape=(tile_cube_m, tmp_n), dtype=dtype_16, depth=2)
        self.l0b = Channel(MemLoc.L0B, shape=(tile_d, tile_n), dtype=dtype_16, depth=2)
        self.l0c = Channel(MemLoc.L0C, shape=(tile_cube_m, tmp_n), dtype=dtypes.float32, depth=2)

    def load_q(self, gm_tensor):
        """GM -> L1 via nd2nz."""
        mem_copy(self.q_l1, gm_tensor, engine=self.nd2nz)

    def load_k(self, gm_tensor):
        """GM -> L1 via nd2nz."""
        mem_copy(self.k_l1, gm_tensor, engine=self.nd2nz)

    def load_v(self, gm_tensor):
        """GM -> L1 via nd2nz."""
        mem_copy(self.v_l1, gm_tensor, engine=self.nd2nz)

    def compute_qk(self):
        """S = Q @ K^T -> L0C."""
        mem_copy(self.l0a, self.q_l1)        # Q L1->L0A
        mem_copy(self.l0b, self.k_l1)        # K L1->L0B
        matmul(self.l0c, self.l0a, self.l0b, init=True)

    def compute_pv(self, p_l1_ch):
        """O = P @ V -> L0C."""
        mem_copy(self.l0b, self.v_l1, transpose=True)   # V L1->L0B^T
        mem_copy(self.l0a, p_l1_ch)                    # P L1->L0A
        matmul(self.l0c, self.l0a, self.l0b, init=True)

    def store_s(self, ub_ch, partition):
        """L0C -> UB via FIXPIPE (split-M). Cross-core produce."""
        mem_copy(ub_ch, self.l0c, engine=self.fixpipe, partition=partition)

    def store_o(self, ub_ch, partition):
        """L0C -> UB via FIXPIPE (split-M). Cross-core produce."""
        mem_copy(ub_ch, self.l0c, engine=self.fixpipe, partition=partition)


# ---- vec side: raw-mode softmax + O accumulation, all channel-first ----
class Vector:
    def __init__(self, tile_vec_m, tile_n, tile_d, subblock_idx, mask_mode=0, dtype_16=dtypes.float16, preload_num=PIPELINE_DEPTH):
        self.tile_vec_m = tile_vec_m
        self.tile_m = tile_vec_m * 2
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.subblock_idx = subblock_idx
        self.mask_mode = mask_mode
        self.dtype_16 = dtype_16

        self.sm_max_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(preload_num)]
        self.sm_sum_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(preload_num)]
        self.sm_exp_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(preload_num)]

        self.tmp_new_max = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.tmp_sum = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.res_o = Buffer(MemLoc.UB, (tile_vec_m, tile_d), dtypes.float32)

        p_n1_pad = 32 // 2  # NZ n1 alignment: 16 elements (half of 32-element n0)
        self.p_ub = Channel(
            MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=self.dtype_16, depth=2,
            data_format="nz", n1_pad=p_n1_pad,
        )
        self.o_ub = Channel(MemLoc.UB, shape=(tile_vec_m, tile_d), dtype=self.dtype_16, depth=1)
        if const_expr(mask_mode == 3):
            self.mask_ch = Channel(
                MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=dtypes.float32, depth=1,
            )

    def _nz_params(self):
        """Derive NZ fractal addressing from p_ub physical layout stride."""
        s = self.p_ub.physical_stride
        s_n1, s_m1, s_m0, s_n0 = s[0], s[1], s[2], s[3]
        m0 = s_m1 // s_m0
        n0 = s_m0 // s_n0
        return m0, s_m1, s_m0, s_n1 // n0

    def _softmax_fold_row(self, qk_ch, base, nz_off, max_brc_buf, row,
                          ve_mask, vo_mask, b16, b16_full, full, block_stride):
        """Per-row fold: deinterleave-load qk, exp(sub max), cast to fp16,
        merge even/odd halves, store into p_ub NZ slot. Returns exp halves
        for reduce_sum."""
        mx = rr.vload_brc(max_brc_buf, row)
        ve, vo = rr.vload_deinterleave(qk_ch, base, width="b32")
        ve = rr.vexp_sub(ve, mx, mask=ve_mask)
        vo = rr.vexp_sub(vo, mx, mask=vo_mask)
        he = rr.vcast(ve, self.dtype_16, mask=ve_mask, reg_layout=rr.RegLayout.ZERO)
        ho = rr.vcast(vo, self.dtype_16, mask=vo_mask, reg_layout=rr.RegLayout.ONE)
        merged = rr.vor(he, ho, mask=b16)
        rr.vstore_strided(self.p_ub, nz_off, merged, b16_full, block_stride=block_stride, repeat_stride=0)
        return ve, vo

    def _pass_a_row(self, qk_ch, scale, sm_max_dst, row, row_stride, VL_T,
                    half0_mask, half1_mask, full_mask):
        """Scale qk row in place, compute rowmax -> sm_max_dst[row]."""
        base = row * row_stride
        v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
        v1 = rr.vmuls(rr.vload(qk_ch, base + VL_T), scale, mask=half1_mask)
        rr.vstore(qk_ch, base, v0, half0_mask)
        rr.vstore(qk_ch, base + VL_T, v1, half1_mask)
        rmax = rr.vreduce_max(rr.vmax(v0, v1, mask=full_mask), mask=full_mask)
        rr.vstore_first(sm_max_dst, row, rmax)

    def _load_mask(self, attn_mask, base_valid):
        """Load additive causal mask tile from 2048x2048 GM template.
        Shifts rows (base>=0) or columns (base<0, S1>S2) to align."""
        base = base_valid - 1
        is_nonneg = base >= 0
        row_off = select(is_nonneg, base, 0)
        neg_base = 0 - base
        col_off = select(is_nonneg, 0, neg_base)
        attn_mask_offset = rebase_view(attn_mask, (row_off, col_off))
        tile_mask = tile_view(
            attn_mask_offset, (self.tile_vec_m, self.tile_n), (0, 0)
        )
        mem_copy(self.mask_ch, tile_mask)

    def _pass_a_row_masked(self, qk_ch, scale, sm_max_dst, row, row_stride, VL_T,
                           half0_mask, half1_mask, full_mask):
        """Pass-A row with additive causal mask."""
        base = row * row_stride
        v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
        v1 = rr.vmuls(rr.vload(qk_ch, base + VL_T), scale, mask=half1_mask)
        v0 = rr.vadd(v0, rr.vload(self.mask_ch, base), mask=half0_mask)
        v1 = rr.vadd(v1, rr.vload(self.mask_ch, base + VL_T), mask=half1_mask)
        rr.vstore(qk_ch, base, v0, half0_mask)
        rr.vstore(qk_ch, base + VL_T, v1, half1_mask)
        rmax = rr.vreduce_max(rr.vmax(v0, v1, mask=full_mask), mask=full_mask)
        is_all_masked = rr.vcmp_le_scalar(rmax, MASK_NEG_THRESHOLD, mask=full_mask)
        zeros = rr.vdup_scalar(0.0, dtypes.float32, mask=full_mask)
        rmax = rr.vselect(zeros, rmax, cond_mask=is_all_masked)
        rr.vstore_first(sm_max_dst, row, rmax)

    @jit
    def _softmax_fold_loop(self, qk_ch, max_buf, sum_dst, actual_n, rows,
                           src_row_stride, N, m0, s_m1, s_m0, block_stride):
        """Per-row fold loop: exp-sub-max into p_ub, reduce_sum into sum_dst.
        Shared by all softmax variants."""
        for row in range(rows):
            full, _ = rr.update_mask(VL_T, elem_bits=32)
            b16_full, _ = rr.update_mask(N, elem_bits=16)
            ve_mask, _ = rr.update_mask((actual_n + 1) // 2, elem_bits=32)
            vo_mask, _ = rr.update_mask(actual_n // 2, elem_bits=32)
            b16, _ = rr.update_mask(actual_n, elem_bits=16)
            base = row * src_row_stride
            nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
            ve, vo = self._softmax_fold_row(
                qk_ch, base, nz_off, max_buf, row,
                ve_mask, vo_mask, b16, b16_full, full, block_stride
            )
            rsum = rr.vreduce_sum(rr.vadd(ve, vo, mask=full), mask=ve_mask)
            rr.vstore_first(sum_dst, row, rsum)
        rr.vmem_bar("vst_vld")

    @jit
    def softmax_first(self, qk_ch, scale, m_axis_triple: int):
        """First n-tile of a new m-tile: P = softmax(S); init running max/sum."""
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]

        N = self.tile_n
        m0, s_m1, s_m0, block_stride = self._nz_params()
        with vf(mode="raw"):
            rows, actual_n = qk_ch.shape
            src_row_stride = qk_ch.physical_stride[0]
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                half1_mask, _ = rr.update_mask(_clamp_nonneg(actual_n - VL_T), elem_bits=32)
                self._pass_a_row(qk_ch, scale, sm_max, row, src_row_stride, VL_T,
                                 half0_mask, half1_mask, full)
            rr.vmem_bar("vst_vld")
            self._softmax_fold_loop(qk_ch, sm_max, sm_sum, actual_n, rows,
                                    src_row_stride, N, m0, s_m1, s_m0, block_stride)

    def _softmax_rest_tail(self, sm_max, sm_sum, sm_exp, rowmask):
        """Online-softmax running-state update."""
        old_max = rr.vload(sm_max, 0)
        new_max = rr.vload(self.tmp_new_max, 0)
        se = rr.vexp_sub(old_max, new_max, mask=rowmask)  # exp(old-new)
        rr.vstore(sm_exp, 0, se, rowmask)
        rr.vstore(sm_max, 0, new_max, rowmask)  # max = new_max
        old_sum = rr.vload(sm_sum, 0)
        new_sum = rr.vload(self.tmp_sum, 0)
        ss = rr.vmadd(old_sum, se, new_sum, mask=rowmask)
        rr.vstore(sm_sum, 0, ss, rowmask)

    @jit
    def softmax_rest(self, qk_ch, scale, m_axis_triple: int, tile_triple: int):
        """Non-first n-tile: rescale running max/sum, P = exp(S - new_max)."""
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        sm_exp = self.sm_exp_tb[tile_triple]

        N = self.tile_n
        m0, s_m1, s_m0, block_stride = self._nz_params()
        with vf(mode="raw"):
            rows, actual_n = qk_ch.shape
            src_row_stride = qk_ch.physical_stride[0]
            rowmask, _ = rr.update_mask(rows, elem_bits=32)
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                half1_mask, _ = rr.update_mask(_clamp_nonneg(actual_n - VL_T), elem_bits=32)
                self._pass_a_row(qk_ch, scale, self.tmp_new_max, row, src_row_stride, VL_T,
                                 half0_mask, half1_mask, full)
            rr.vmem_bar("vst_vld")
            nm = rr.vmax(rr.vload(sm_max, 0), rr.vload(self.tmp_new_max, 0), mask=rowmask)
            rr.vstore(self.tmp_new_max, 0, nm, rowmask)
            rr.vmem_bar("vst_vld")
            self._softmax_fold_loop(qk_ch, self.tmp_new_max, self.tmp_sum, actual_n, rows,
                                    src_row_stride, N, m0, s_m1, s_m0, block_stride)
            self._softmax_rest_tail(sm_max, sm_sum, sm_exp, rowmask)

    @jit
    def softmax_first_masked(self, qk_ch, attn_mask, scale,
                             m_axis_triple: int, base_valid: int):
        """First n-tile with additive causal mask."""
        self._load_mask(attn_mask, base_valid)
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]

        N = self.tile_n
        m0, s_m1, s_m0, block_stride = self._nz_params()
        with vf(mode="raw"):
            rows, actual_n = qk_ch.shape
            src_row_stride = qk_ch.physical_stride[0]
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                half1_mask, _ = rr.update_mask(_clamp_nonneg(actual_n - VL_T), elem_bits=32)
                self._pass_a_row_masked(qk_ch, scale, sm_max, row, src_row_stride, VL_T,
                                        half0_mask, half1_mask, full)
            rr.vmem_bar("vst_vld")
            self._softmax_fold_loop(qk_ch, sm_max, sm_sum, actual_n, rows,
                                    src_row_stride, N, m0, s_m1, s_m0, block_stride)

    @jit
    def softmax_rest_masked(self, qk_ch, attn_mask, scale,
                            m_axis_triple: int, tile_triple: int,
                            base_valid: int):
        """Non-first n-tile with additive causal mask."""
        self._load_mask(attn_mask, base_valid)
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        sm_exp = self.sm_exp_tb[tile_triple]

        N = self.tile_n
        m0, s_m1, s_m0, block_stride = self._nz_params()
        with vf(mode="raw"):
            rows, actual_n = qk_ch.shape
            src_row_stride = qk_ch.physical_stride[0]
            rowmask, _ = rr.update_mask(rows, elem_bits=32)
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                half1_mask, _ = rr.update_mask(_clamp_nonneg(actual_n - VL_T), elem_bits=32)
                self._pass_a_row_masked(qk_ch, scale, self.tmp_new_max, row, src_row_stride, VL_T,
                                        half0_mask, half1_mask, full)
            rr.vmem_bar("vst_vld")
            nm = rr.vmax(rr.vload(sm_max, 0), rr.vload(self.tmp_new_max, 0), mask=rowmask)
            rr.vstore(self.tmp_new_max, 0, nm, rowmask)
            rr.vmem_bar("vst_vld")
            self._softmax_fold_loop(qk_ch, self.tmp_new_max, self.tmp_sum, actual_n, rows,
                                    src_row_stride, N, m0, s_m1, s_m0, block_stride)
            self._softmax_rest_tail(sm_max, sm_sum, sm_exp, rowmask)

    def store_p(self, p_l1_ch, partition):
        """p_ub -> the explicit per-AIV M tile of the shared L1 slot."""
        piece = partition_view(p_l1_ch, partition, self.subblock_idx)
        mem_copy(piece, self.p_ub)

    def init_o(self, pv_ch):
        """First PV tile: res_o = P0 @ V0."""
        mem_copy(self.res_o, pv_ch)

    @jit
    def update_o(self, pv_ch, sm_exp_buf):
        """Non-first PV tile: res_o = res_o * exp(old_max-new_max) + P_i @ V_i."""

        with vf(mode="raw"):
            for row in range(self.tile_vec_m):
                exp_b = rr.vload_brc(sm_exp_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    mask, _ = rr.update_mask(self.tile_d - col, elem_bits=32)
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    o = rr.vmadd(pre, exp_b, cur, mask=mask)
                    rr.vstore(self.res_o, off, o, mask)

    @jit
    def update_o_last(self, pv_ch, sm_exp_buf, sm_sum_buf):
        """Last PV tile: res_o = (res_o * exp + pv) / sum. Fuses division.
        Fully-masked rows (sum==0) output 0."""

        with vf(mode="raw"):
            for row in range(self.tile_vec_m):
                exp_b = rr.vload_brc(sm_exp_buf, row)
                sum_b = rr.vload_brc(sm_sum_buf, row)
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                is_zero = rr.vcmp_eq_scalar(sum_b, 0.0, mask=full)
                one_b = rr.vdup_scalar(1.0, dtypes.float32, mask=full)
                safe_sum = rr.vselect(one_b, sum_b, cond_mask=is_zero)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    mask, _ = rr.update_mask(self.tile_d - col, elem_bits=32)
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    o = rr.vmadd(pre, exp_b, cur, mask=mask)
                    o = rr.vdiv(o, safe_sum, mask=mask)
                    zero_b = rr.vdup_scalar(0.0, dtypes.float32, mask=full)
                    o = rr.vselect(zero_b, o, cond_mask=is_zero)
                    rr.vstore(self.res_o, off, o, mask)

    @jit
    def init_o_last(self, pv_ch, sm_sum_buf):
        """Single-tile case: res_o = pv / sum. Fully-masked rows output 0."""

        with vf(mode="raw"):
            for row in range(self.tile_vec_m):
                sum_b = rr.vload_brc(sm_sum_buf, row)
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                is_zero = rr.vcmp_eq_scalar(sum_b, 0.0, mask=full)
                one_b = rr.vdup_scalar(1.0, dtypes.float32, mask=full)
                safe_sum = rr.vselect(one_b, sum_b, cond_mask=is_zero)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    mask, _ = rr.update_mask(self.tile_d - col, elem_bits=32)
                    cur = rr.vload(pv_ch, off)
                    val = rr.vdiv(cur, safe_sum, mask=mask)
                    zero_b = rr.vdup_scalar(0.0, dtypes.float32, mask=full)
                    val = rr.vselect(zero_b, val, cond_mask=is_zero)
                    rr.vstore(self.res_o, off, val, mask)

    @jit
    def _finalize_div_vf(self, sm_sum_buf, actual_vec_m):
        """res_o /= sm_sum. Fully-masked rows (sum==0) output 0."""

        with vf(mode="raw"):
            for row in range(actual_vec_m):
                sum_b = rr.vload_brc(sm_sum_buf, row)
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                is_zero = rr.vcmp_eq_scalar(sum_b, 0.0, mask=full)
                one_b = rr.vdup_scalar(1.0, dtypes.float32, mask=full)
                safe_sum = rr.vselect(one_b, sum_b, cond_mask=is_zero)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    mask, _ = rr.update_mask(self.tile_d - col, elem_bits=32)
                    val = rr.vdiv(rr.vload(self.res_o, off), safe_sum, mask=mask)
                    zero_b = rr.vdup_scalar(0.0, dtypes.float32, mask=full)
                    val = rr.vselect(zero_b, val, cond_mask=is_zero)
                    rr.vstore(self.res_o, off, val, mask)

    def finalize_o(self, o_tile_gm, sm_sum_buf, div_done=False):
        """Cast res_o f32->f16, store to GM. Divides by sm_sum if not div_done."""
        split_m = make_partition_tiler(
            o_tile_gm.shape,
            (self.tile_m, max(self.tile_n, self.tile_d)),
        )
        half = partition_view(o_tile_gm, split_m, self.subblock_idx)
        actual_vec_m = half.shape[0]

        if not div_done:
            self._finalize_div_vf(sm_sum_buf, actual_vec_m)

        o_slot = self.o_ub.acquire()
        o_full = local_slice(o_slot, (self.tile_vec_m, self.tile_d), stride=(self.tile_d, 1))
        cast(o_full, self.res_o)
        self.o_ub.commit(o_slot)

        o_slot_r = self.o_ub.wait()
        o_view_r = local_slice(o_slot_r, (actual_vec_m, self.tile_d), stride=(self.tile_d, 1))
        mem_copy(half, o_view_r)
        self.o_ub.release(o_slot_r)


@kernel
class flash_attn_kernel:
    def __init__(self, tile_cube_m, tile_vec_m, tile_n, tile_d, return_softmax_lse,
                 mask_mode=0, dtype_16=dtypes.float16):
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.return_softmax_lse = return_softmax_lse
        self.mask_mode = mask_mode
        self.preload_num = PIPELINE_DEPTH  # 3-stage pipeline depth

        self.qk_ub = Channel(MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=dtypes.float32, depth=2, kind=ChannelKind.CrossCore)
        self.pv_ub = Channel(MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=dtypes.float32, depth=2, kind=ChannelKind.CrossCore)
        self.p_l1 = Channel(MemLoc.L1, shape=(tile_cube_m, tile_n), dtype=dtype_16,
                            depth=3, kind=ChannelKind.CrossCore)

        self.block_idx = get_block_idx()
        self.subblock_idx = get_subblock_id()

        self.matmul = Matmul(tile_cube_m, tile_n, tile_d, dtype_16)
        self.vector = Vector(tile_vec_m, tile_n, tile_d, self.subblock_idx, mask_mode, dtype_16)

    @jit
    def _n_tile_max(self, seqlen_q, seqlen_k, m_idx):
        """Number of KV tiles for this m-tile. mask_mode==3 clamps to 0
        for tiles fully above the causal diagonal (S1>S2)."""
        n_block_max = ceil_div(seqlen_k, self.tile_n)
        if const_expr(self.mask_mode == 3):
            last_key = (m_idx + 1) * self.tile_cube_m + (seqlen_k - seqlen_q)
            return _clamp_nonneg(min(n_block_max, ceil_div(last_key, self.tile_n)))
        return n_block_max

    @jit
    def _stage_qk(self, tile_idx: int, n_idx: int, batch_idx: int,
                  kv_head: int, q_tile_gm: Tensor):
        """S = Q @ K^T -> qk_ub."""
        key_slice = self._key[batch_idx, kv_head, None, None]
        k_tile_gm = tile_view(key_slice, (self.tile_n, self.tile_d), (n_idx, 0))
        self.matmul.load_k(k_tile_gm)
        self.matmul.compute_qk()
        split_m = make_partition_tiler(
            (q_tile_gm.shape[0], k_tile_gm.shape[0]),
            (self.tile_cube_m, max(self.tile_n, self.tile_d)),
        )
        self.matmul.store_s(self.qk_ub, split_m)

    @jit
    def _stage_softmax(self, tick: int, tile_idx: int, n_idx: int,
                       m_seq: int):
        """softmax(S) -> p_l1 (V2C split-M)."""
        batch_idx, head_idx, m_idx, _ = idx2crd(
            tile_idx, [self._batch_size, self._head_num_q, self._seq_tile_num, 1]
        )

        kv_head = head_idx // self._head_group_num
        query_slice = self._query[batch_idx, head_idx, None, None]
        q_tile_gm = tile_view(
            query_slice, (self.tile_cube_m, self.tile_d), (m_idx, 0)
        )
        key_slice = self._key[batch_idx, kv_head, None, None]
        k_tile_gm = tile_view(
            key_slice, (self.tile_n, self.tile_d), (n_idx, 0)
        )
        split_m = make_partition_tiler(
            (q_tile_gm.shape[0], k_tile_gm.shape[0]),
            (self.tile_cube_m, max(self.tile_n, self.tile_d)),
        )
        q_half_gm = partition_view(q_tile_gm, split_m, self.subblock_idx)
        actual_vec_m = q_half_gm.shape[0]

        is_first = 1 if n_idx == 0 else 0
        tile_triple = (tick - 1) % 3
        m_axis_triple = m_seq % self.preload_num

        if const_expr(self.mask_mode == 3):
            # Per-row valid column count for local row 0 of this subblock.
            # vn(row) = clamp(base_valid + row, 0, actual_n) computed inside
            # softmax_first/softmax_rest via _clamp_range.
            #
            # local_start is the subblock's offset WITHIN the m-tile. The balanced
            # partition tiler does NOT split at a fixed tile_vec_m: for an M-tail
            # tile of e.g. 122 rows it splits 61/61, not 64/58. subblock 1 is
            # always the TRAILING chunk, so its start = m_tile_rows - actual_vec_m;
            # subblock 0 starts at 0. (Hardcoding subblock_idx*tile_vec_m is only
            # correct for full 128-row tiles and silently shifts vn on M-tail.)
            offset = self._seqlen_k - self._seqlen_q
            m_tile_rows = q_tile_gm.shape[0]
            local_start = select(
                self.subblock_idx == 0,
                0,
                m_tile_rows - actual_vec_m,
            )
            q_row0 = m_idx * self.tile_cube_m + local_start
            k_start = n_idx * self.tile_n
            base_valid = q_row0 + offset - k_start + 1

            if base_valid >= self.tile_n:
                qk_view = local_slice(
                    self.qk_ub, (actual_vec_m, k_tile_gm.shape[0]),
                    stride=(self.tile_n, 1),
                )
                if is_first:
                    self.vector.softmax_first(qk_view, self._scale, m_axis_triple)
                else:
                    self.vector.softmax_rest(qk_view, self._scale, m_axis_triple, tile_triple)
            else:
                qk_view = local_slice(
                    self.qk_ub, (actual_vec_m, k_tile_gm.shape[0]),
                    stride=(self.tile_n, 1),
                )
                if is_first:
                    self.vector.softmax_first_masked(
                        qk_view, self._attn_mask, self._scale, m_axis_triple, base_valid
                    )
                else:
                    self.vector.softmax_rest_masked(
                        qk_view, self._attn_mask, self._scale,
                        m_axis_triple, tile_triple, base_valid
                    )
        else:
            qk_view = local_slice(
                self.qk_ub, (actual_vec_m, k_tile_gm.shape[0]),
                stride=(self.tile_n, 1),
            )
            if is_first:
                self.vector.softmax_first(qk_view, self._scale, m_axis_triple)
            else:
                self.vector.softmax_rest(qk_view, self._scale, m_axis_triple, tile_triple)
        self.vector.store_p(self.p_l1, split_m)

    @jit
    def _stage_pv(self, tile_idx: int, n_idx: int):
        """P @ V -> pv_ub."""
        batch_idx, head_idx, m_idx, _ = idx2crd(
            tile_idx, [self._batch_size, self._head_num_q, self._seq_tile_num, 1]
        )

        kv_head = head_idx // self._head_group_num

        value_slice = self._value[batch_idx, kv_head, None, None]
        v_tile_gm = tile_view(value_slice, (self.tile_n, self.tile_d), (n_idx, 0))

        self.matmul.load_v(v_tile_gm)
        p_l1_view = local_slice(
            self.p_l1, (self.tile_cube_m, v_tile_gm.shape[0])
        )
        self.matmul.compute_pv(p_l1_view)
        out_slice = self._attn_out[batch_idx, head_idx, None, None]
        o_tile_gm = tile_view(
            out_slice, (self.tile_cube_m, self.tile_d), (m_idx, 0)
        )
        split_m = make_partition_tiler(
            o_tile_gm.shape,
            (self.tile_cube_m, max(self.tile_n, self.tile_d)),
        )
        self.matmul.store_o(self.pv_ub, split_m)

    @jit
    def _stage_update(self, tick: int, tile_idx: int, n_idx: int,
                      m_seq: int):
        """Update res_o with P@V; finalize on last kv tile."""
        batch_idx, head_idx, m_idx, _ = idx2crd(
            tile_idx, [self._batch_size, self._head_num_q, self._seq_tile_num, 1]
        )

        is_first_kv_tile = 1 if n_idx == 0 else 0
        n_end = self._n_tile_max(self._seqlen_q, self._seqlen_k, m_idx)
        is_last_kv_tile = 1 if n_idx == n_end - 1 else 0

        m_axis_triple = m_seq % self.preload_num
        tile_triple = (tick - 3) % 3

        if is_first_kv_tile and is_last_kv_tile:
            sm_sum_buf = self.vector.sm_sum_tb[m_axis_triple]
            self.vector.init_o_last(self.pv_ub, sm_sum_buf)
            out_slice = self._attn_out[batch_idx, head_idx, None, None]
            o_tile_gm = tile_view(
                out_slice, (self.tile_cube_m, self.tile_d), (m_idx, 0)
            )
            self.vector.finalize_o(o_tile_gm, sm_sum_buf, div_done=True)
        elif is_first_kv_tile:
            self.vector.init_o(self.pv_ub)
        elif is_last_kv_tile:
            sm_exp_buf = self.vector.sm_exp_tb[tile_triple]
            sm_sum_buf = self.vector.sm_sum_tb[m_axis_triple]
            self.vector.update_o_last(self.pv_ub, sm_exp_buf, sm_sum_buf)
            out_slice = self._attn_out[batch_idx, head_idx, None, None]
            o_tile_gm = tile_view(
                out_slice, (self.tile_cube_m, self.tile_d), (m_idx, 0)
            )
            self.vector.finalize_o(o_tile_gm, sm_sum_buf, div_done=True)
        else:
            sm_exp_buf = self.vector.sm_exp_tb[tile_triple]
            self.vector.update_o(self.pv_ub, sm_exp_buf)

    def __call__(
        self,
        attn_out: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: float,
        block_table: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_kv: Tensor,
        seqused_q: Tensor,
        seqused_kv: Tensor,
        sinks: Tensor,
        win_left: int,
        win_right: int,
        softmax_lse_gm: Tensor = None,
        attn_mask: Tensor = None,
        metadata: Tensor = None,
    ):
        """Merged FA kernel, channel-first, typed channels."""
        delay_slots = self.preload_num + 1  # pipeline depth + 1 for drain

        batch_size, head_num_q, seqlen_q = query.shape[:3]
        head_num_kv, seqlen_k = key.shape[1], key.shape[2]
        head_group_num = head_num_q // head_num_kv
        seq_tile_num = ceil_div(seqlen_q, self.tile_cube_m)

        start_tile_idx = metadata[0, self.block_idx]
        tiles_per_block = metadata[1, self.block_idx]

        dl = DelayLineGroup(delay_slots, 'tile', 'n', 'm_seq')
        tick = 0
        issued = 0
        m_seq = 0

        self._batch_size = batch_size
        self._head_num_q = head_num_q
        self._seq_tile_num = seq_tile_num
        self._head_group_num = head_group_num
        self._seqlen_q = seqlen_q
        self._seqlen_k = seqlen_k
        self._scale = scale
        self._query = query
        self._key = key
        self._value = value
        self._attn_out = attn_out
        self._attn_mask = attn_mask

        if tiles_per_block > 0:
            for tile_idx in range(start_tile_idx, start_tile_idx + tiles_per_block):
                batch_idx, head_idx, m_idx, _ = idx2crd(
                    tile_idx, [batch_size, head_num_q, seq_tile_num, 1]
                )

                seqlen_k_curr = seqlen_k
                n_end = self._n_tile_max(seqlen_q, seqlen_k_curr, m_idx)

                if n_end > 0:
                    kv_head = head_idx // head_group_num
                    m_seq = m_seq + 1

                    query_slice = query[batch_idx, head_idx, None, None]
                    q_tile_gm = tile_view(query_slice, (self.tile_cube_m, self.tile_d), (m_idx, 0))
                    self.matmul.load_q(q_tile_gm)

                    for n_idx in range(n_end):
                        dl.push(tile=tile_idx, n=n_idx, m_seq=m_seq)
                        self._stage_qk(tile_idx, n_idx, batch_idx, kv_head, q_tile_gm)
                        issued = issued + 1

                        if tick >= 1 and tick - 1 < issued:
                            self._stage_softmax(tick, dl.tile.tap(1), dl.n.tap(1), dl.m_seq.tap(1))
                        if tick >= 2 and tick - 2 < issued:
                            self._stage_pv(dl.tile.tap(2), dl.n.tap(2))
                        if tick >= 3 and tick - 3 < issued:
                            self._stage_update(tick, dl.tile.tap(3), dl.n.tap(3), dl.m_seq.tap(3))

                        dl.advance()
                        tick += 1

            for _ in range(self.preload_num):
                if tick >= 1 and tick - 1 < issued:
                    self._stage_softmax(tick, dl.tile.tap(1), dl.n.tap(1), dl.m_seq.tap(1))
                if tick >= 2 and tick - 2 < issued:
                    self._stage_pv(dl.tile.tap(2), dl.n.tap(2))
                if tick >= 3 and tick - 3 < issued:
                    self._stage_update(tick, dl.tile.tap(3), dl.n.tap(3), dl.m_seq.tap(3))

                dl.advance()
                tick += 1


def get_tile_config(D):
    """Return (tile_cube_m, tile_vec_m, tile_n, tile_d) for the given head dim."""
    if const_expr(D == 128):
        return 128, 64, 128, 128
    raise ValueError(f"Unsupported head dim D={D}, only D=128 is supported")


# ============================================================================
# 3. Host-side load balancing + JIT launch
# ============================================================================

class FlashAttnLauncher:
    """JIT-compiled kernel launch with layout transforms."""

    def __init__(self, layout_q, layout_kv, layout_out, mask_mode, dtype, block_dim):
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.layout_out = layout_out
        self.mask_mode = mask_mode
        self.dtype = dtype
        self.block_dim = block_dim

    @jit
    def launch(
        self,
        attn_out: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: float,
        block_table: Tensor | None = None,
        seqused_kv: Tensor | None = None,
        cu_seqlens_q: Tensor | None = None,
        cu_seqlens_kv: Tensor | None = None,
        seqused_q: Tensor | None = None,
        sinks: Tensor | None = None,
        win_left: int = 0,
        win_right: int = 0,
        softmax_lse_gm: Tensor | None = None,
        attn_mask: Tensor | None = None,
        metadata: Tensor | None = None,
    ):
        """JIT-compiled kernel launch with layout transforms."""
        if const_expr(self.layout_q == "BSND"):
            query = _bsnd_to_bnsd(query)
        if const_expr(self.layout_kv == "BSND"):
            key = _bsnd_to_bnsd(key)
            value = _bsnd_to_bnsd(value)
        if const_expr(self.layout_out == "BSND"):
            attn_out = _bsnd_to_bnsd(attn_out)

        D = query.shape[3]
        tile_cube_m, tile_vec_m, tile_n, tile_d = get_tile_config(D)
        op = flash_attn_kernel(
            tile_cube_m=tile_cube_m,
            tile_vec_m=tile_vec_m,
            tile_n=tile_n,
            tile_d=tile_d,
            return_softmax_lse=False,
            mask_mode=self.mask_mode,
            dtype_16=self.dtype,
        )
        op[self.block_dim](attn_out, query, key,
                           value, scale, block_table, cu_seqlens_q,
                           cu_seqlens_kv, seqused_q, seqused_kv, sinks,
                           win_left, win_right, softmax_lse_gm,
                           attn_mask, metadata)

def _compute_load_balance(q_shape, n_kv_head, seqlen_k, mask_mode, block_dim):
    """Compute per-core m-tile ranges based on causal cost weighting.
    Returns metadata as [2, block_dim] int64 NPU tensor where
    metadata[0] = m_tile_starts, metadata[1] = m_tile_counts."""
    B, N1, S1, D = q_shape
    S2 = seqlen_k

    tile_cube_m, _, tile_n, _ = get_tile_config(D)
    seq_tile_num = (S1 + tile_cube_m - 1) // tile_cube_m
    total_m_tiles = B * N1 * seq_tile_num

    n_block_max = (S2 + tile_n - 1) // tile_n
    offset = S2 - S1

    costs = []
    for i in range(total_m_tiles):
        m_idx = i % seq_tile_num
        if mask_mode == 3:
            last_key = (m_idx + 1) * tile_cube_m + offset
            cost = max(0, min(n_block_max, (last_key + tile_n - 1) // tile_n))
        else:
            cost = n_block_max
        costs.append(cost)

    total_cost = sum(costs)
    target = total_cost / block_dim

    starts = [total_m_tiles] * block_dim
    counts = [0] * block_dim

    cumulative = 0
    core = 0
    for i in range(total_m_tiles):
        if core < block_dim - 1 and cumulative >= (core + 1) * target:
            core += 1
        if counts[core] == 0:
            starts[core] = i
        counts[core] += 1
        cumulative += costs[i]

    metadata = torch.tensor([starts, counts], dtype=torch.int64).npu()
    return from_torch_npu(metadata)


# ============================================================================
# 4. Torch Interface
# ============================================================================

def flash_attn(query, key, value, scale, *, mask_mode=0, attn_mask=None,
               layout_q="BNSD", layout_kv="BNSD", layout_out="BNSD",
               dtype=dtypes.float16):
    """Torch-facing wrapper for the Flash Attention kernel.

    Computes O = softmax(Q @ K^T * scale) @ V with optional causal masking.

    Args:
        query: [B, N1, S1, D] (BNSD) or [B, S1, N1, D] (BSND) NPU tensor.
        key:   [B, N2, S2, D] (BNSD) or [B, S2, N2, D] (BSND) NPU tensor.
        value: same layout as key.
        scale: float, softmax scaling factor (typically 1/sqrt(D)).
        mask_mode: 0=full attention, 3=right-context causal.
        attn_mask: optional causal mask tensor (2048x2048, float32).
        layout_q/layout_kv/layout_out: "BNSD" or "BSND".
        dtype: canonical canndsl descriptor (dtypes.float16 or dtypes.bfloat16).

    Returns:
        Output tensor with same shape/layout as query.
    """
    import torch
    block_dim = torch.npu.get_device_properties(0).cube_core_num
    if layout_q == "BSND":
        B, N1, S1, D = query.shape[0], query.shape[2], query.shape[1], query.shape[3]
    else:
        B, N1, S1, D = query.shape[:4]
    if layout_kv == "BSND":
        N2, S2 = key.shape[2], key.shape[1]
    else:
        N2, S2 = key.shape[1], key.shape[2]

    if layout_out == "BSND":
        out = torch.zeros(B, S1, N1, D, dtype=query.dtype, device=query.device)
    else:
        out = torch.zeros(B, N1, S1, D, dtype=query.dtype, device=query.device)

    metadata = _compute_load_balance(
        (B, N1, S1, D), N2, S2, mask_mode, block_dim,
    )

    launcher = FlashAttnLauncher(layout_q, layout_kv, layout_out, mask_mode, dtype, block_dim)
    launcher.launch(
        from_torch_npu(out), from_torch_npu(query),
        from_torch_npu(key), from_torch_npu(value),
        scale,
        None, None, None, None, None, None, 0, 0, None,
        from_torch_npu(attn_mask) if attn_mask is not None else None,
        metadata,
    )
    return out
