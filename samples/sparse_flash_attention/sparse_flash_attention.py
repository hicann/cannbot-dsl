# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Sparse Flash Attention for MLA-absorb.

The kernel gathers sparse KV rows, computes tiled QK attention with online
softmax, and accumulates the absorbed Value output. It supports FP16/BF16,
BSND/TND queries, and BSND/TND/PA_BSND KV storage.
"""

from __future__ import annotations
from cannbotdsl.lang.host import host
import cannbotdsl

import torch

from sparse_flash_attention_validation import (
    DIM_NOPE,
    DIM_ROPE,
    INT64_MAX,
    SPARSE_MODE_DENSE,
    SPARSE_MODE_RIGHT_DOWN_CAUSAL,
    SUPPORTED_ATTENTION_MODE,
    SUPPORTED_SPARSE_BLOCK_SIZE,
    validate_sparse_flash_attention_args,
)

from cannbotdsl import (
    ChannelKind,
    MemLoc,
    PIPE,
    Tensor,
    dtypes,
    make_copy_engine,
    matmul,
    mem_copy,
)
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.sync import (
    cube_sync_block_wait,
    vec_sync_block_arrive,
)
from cannbotdsl.tensor import (
    make_tiler,
    reinterpret,
    tile_slice,
)
from cannbotdsl.types.delay_line import DelayLineGroup


INT32_MAX = 2_147_483_647

INVALID_SPARSE_INDEX = -1
EMPTY_QUERY_ID = -1

TILE_M = 64
TILE_S2 = 128
CUBE_K_TILE = 128
GATHER_CHUNK_ROWS = 16
AIV_ROW_CAPACITY = 32
AIV_COUNT = 2
SPARSE_INDEX_PAIR_SIZE = 2
DIM_QK = DIM_NOPE + DIM_ROPE

PIPELINE_SLOT_COUNT = 3
PIPELINE_STAGE_COUNT = 4
PIPELINE_DRAIN_TICKS = PIPELINE_STAGE_COUNT - 1

BYTES_PER_16BIT_ELEMENT = 2
NZ_BLOCK_BYTES = 32
VECTOR_REGISTER_BITS = 2048
FP32_BITS = 32
VL_T = VECTOR_REGISTER_BITS // FP32_BITS
SOFTMAX_NEG_INF = -1.0e30
EMPTY_SOFTMAX_THRESHOLD = -1.0e29


@jit
def _copy_kv_token(stage, key_token, rope_token, row):
    """Copy one sparse KV row with NoPE followed by RoPE in UB."""
    dst = tile_slice(stage, (1, DIM_QK), (row, 0))
    mem_copy(tile_slice(dst[None, :DIM_NOPE], (1, DIM_NOPE), (0, 0)), key_token)
    mem_copy(tile_slice(dst[None, DIM_NOPE:DIM_QK], (1, DIM_ROPE), (0, 0)), rope_token)


@jit
def _copy_kv_token_no_rope(stage, key_token, row):
    """Copy one sparse K row when RoPE is disabled."""
    dst = tile_slice(stage, (1, DIM_NOPE), (row, 0))
    mem_copy(dst, key_token)


@jit
def _copy_kv_pair(stage, key0, key1, rope0, rope1, pair_index):
    """Copy two sparse KV rows together, separately for NoPE and RoPE."""
    key_dst = tile_slice(
        stage[None, :DIM_NOPE], (SPARSE_INDEX_PAIR_SIZE, DIM_NOPE),
        (pair_index, 0),
    )
    rope_dst = tile_slice(
        stage[None, DIM_NOPE:DIM_QK], (SPARSE_INDEX_PAIR_SIZE, DIM_ROPE),
        (pair_index, 0),
    )
    mem_copy(key_dst, (key0, key1), axis=0)
    mem_copy(rope_dst, (rope0, rope1), axis=0)


@jit
def _copy_kv_pair_no_rope(stage, key0, key1, pair_index):
    """Copy two sparse K rows when RoPE is disabled."""
    dst = tile_slice(
        stage, (SPARSE_INDEX_PAIR_SIZE, DIM_NOPE), (pair_index, 0)
    )
    mem_copy(dst, (key0, key1), axis=0)


@jit
def _clamp_nonneg(value: int) -> int:
    """Clamp a dynamic mask length before converting it to an unsigned value."""
    return 0 if value < 0 else value


def _nz_params(p_ub):
    """Derive NZ fractal addressing from p_ub physical layout stride."""
    s = p_ub.physical_stride
    s_n1, s_m1, s_m0, s_n0 = s[0], s[1], s[2], s[3]
    m0 = s_m1 // s_m0
    n0 = s_m0 // s_n0
    return m0, s_m1, s_m0, s_n1 // n0


class _StableOnlineSoftmaxVector:
    def __init__(
        self,
        tile_vec_m,
        tile_n,
        tile_d,
        subblock_idx,
        dtype_16=dtypes.float16,
        preload_num=PIPELINE_SLOT_COUNT,
    ):
        self.tile_vec_m = tile_vec_m
        self.tile_m = tile_vec_m * 2
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.subblock_idx = subblock_idx
        self.dtype_16 = dtype_16

        self.sm_max_tb = [
            Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(preload_num)
        ]
        self.sm_sum_tb = [
            Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(preload_num)
        ]
        self.sm_exp_tb = [
            Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
            for _ in range(preload_num)
        ]

        self.tmp_new_max = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.tmp_sum = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.res_o = Buffer(MemLoc.UB, (tile_vec_m, tile_d), dtypes.float32)

        # Address NZ data with its padded physical stride.
        p_n1_pad = NZ_BLOCK_BYTES // BYTES_PER_16BIT_ELEMENT
        self.p_ub = Channel(
            MemLoc.UB,
            shape=(tile_vec_m, tile_n),
            dtype=self.dtype_16,
            depth=2,
            data_format="nz",
            n1_pad=p_n1_pad,
        )
        self.o_ub = Channel(
            MemLoc.UB, shape=(tile_vec_m, tile_d), dtype=self.dtype_16, depth=1
        ).produce()

    @jit
    def softmax_first(self, qk_ch, actual_n, scale, m_axis_triple: int):
        """Initialize running max/sum and unnormalized weights for the first tile."""
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]

        p_ub = self.p_ub.produce()
        tile_n = self.tile_n
        m0, s_m1, s_m0, block_stride = _nz_params(p_ub)
        with vf(mode="simd"):
            # The score buffer keeps a fixed TILE_S2 row pitch for dynamic N.
            rows = self.tile_vec_m
            src_row_stride = self.tile_n
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                half1_mask, _ = rr.update_mask(
                    _clamp_nonneg(actual_n - VL_T), elem_bits=32
                )
                self._pass_a_row(
                    qk_ch,
                    scale,
                    sm_max,
                    row,
                    src_row_stride,
                    VL_T,
                    half0_mask,
                    half1_mask,
                    full,
                )
            rr.vmem_bar("vst_vld")
            self._softmax_fold_loop(
                qk_ch,
                p_ub,
                sm_max,
                sm_sum,
                actual_n,
                rows,
                src_row_stride,
                tile_n,
                m0,
                s_m1,
                s_m0,
                block_stride,
            )

    @jit
    def softmax_rest(
        self, qk_ch, actual_n, scale, m_axis_triple: int, tile_triple: int
    ):
        """Non-first n-tile: rescale running max/sum, P = exp(S - new_max)."""
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        sm_exp = self.sm_exp_tb[tile_triple]

        p_ub = self.p_ub.produce()
        tile_n = self.tile_n
        m0, s_m1, s_m0, block_stride = _nz_params(p_ub)
        with vf(mode="simd"):
            # The score buffer keeps a fixed TILE_S2 row pitch for dynamic N.
            rows = self.tile_vec_m
            src_row_stride = self.tile_n
            rowmask, _ = rr.update_mask(rows, elem_bits=32)
            for row in range(rows):
                full, _ = rr.update_mask(VL_T, elem_bits=32)
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                half1_mask, _ = rr.update_mask(
                    _clamp_nonneg(actual_n - VL_T), elem_bits=32
                )
                self._pass_a_row(
                    qk_ch,
                    scale,
                    self.tmp_new_max,
                    row,
                    src_row_stride,
                    VL_T,
                    half0_mask,
                    half1_mask,
                    full,
                )
            rr.vmem_bar("vst_vld")
            nm = rr.vmax(
                rr.vload(sm_max, 0), rr.vload(self.tmp_new_max, 0), mask=rowmask
            )
            rr.vstore(self.tmp_new_max, 0, nm, rowmask)
            rr.vmem_bar("vst_vld")
            self._softmax_fold_loop(
                qk_ch,
                p_ub,
                self.tmp_new_max,
                self.tmp_sum,
                actual_n,
                rows,
                src_row_stride,
                tile_n,
                m0,
                s_m1,
                s_m0,
                block_stride,
            )
            self._softmax_rest_tail(sm_max, sm_sum, sm_exp, rowmask)

    def store_p(self, p_l1_ch):
        """p_ub -> the explicit per-AIV M tile of the shared L1 slot."""
        mem_copy(p_l1_ch, self.p_ub.consume(),
                 engine=make_copy_engine(split_axis=0), part_id=self.subblock_idx)

    def init_o(self, pv_ch):
        """First PV tile: initialize the FP32 output accumulator directly."""
        mem_copy(reinterpret(self.res_o, shape=(self.tile_vec_m, self.tile_d)), pv_ch)

    @jit
    def update_o(self, pv_ch, sm_exp_buf):
        """Non-first PV tile: res_o = res_o * exp(old_max-new_max) + P_i @ V_i."""
        with vf(mode="simd"):
            for row in range(self.tile_vec_m):
                exp_b = rr.vload_broadcast(sm_exp_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, VL_T)):
                    off = base + col
                    mask, _ = rr.update_mask(self.tile_d - col, elem_bits=32)
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    o = rr.vmadd(pre, exp_b, cur, mask=mask)
                    rr.vstore(self.res_o, off, o, mask)

    @jit
    def _softmax_fold_loop(
        self,
        qk_ch,
        p_ub,
        max_buf,
        sum_dst,
        actual_n,
        rows,
        src_row_stride,
        tile_n,
        m0,
        s_m1,
        s_m0,
        block_stride,
    ):
        """Write unnormalized weight rows and their exponential sums."""
        for row in range(rows):
            full, _ = rr.update_mask(VL_T, elem_bits=32)
            b16_full, _ = rr.update_mask(tile_n, elem_bits=16)
            ve_mask, _ = rr.update_mask((actual_n + 1) // 2, elem_bits=32)
            vo_mask, _ = rr.update_mask(actual_n // 2, elem_bits=32)
            b16, _ = rr.update_mask(actual_n, elem_bits=16)
            base = row * src_row_stride
            nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
            ve, vo = self._softmax_fold_row(
                qk_ch,
                p_ub,
                base,
                nz_off,
                max_buf,
                row,
                ve_mask,
                vo_mask,
                b16,
                b16_full,
                block_stride,
            )
            rsum = rr.vreduce_sum(rr.vadd(ve, vo, mask=full), mask=ve_mask)
            rr.vstore_first(sum_dst, row, rsum)
        rr.vmem_bar("vst_vld")

    def _softmax_fold_row(
        self,
        qk_ch,
        p_ub,
        base,
        nz_off,
        max_brc_buf,
        row,
        ve_mask,
        vo_mask,
        b16,
        b16_full,
        block_stride,
    ):
        """Store unnormalized NZ weights and return the FP32 exponential halves."""
        mx = rr.vload_broadcast(max_brc_buf, row)
        ve, vo = rr.vload_deinterleave(qk_ch, base, width="b32")
        ve = rr.vexp_sub(ve, mx, mask=ve_mask)
        vo = rr.vexp_sub(vo, mx, mask=vo_mask)
        he = rr.vcast(ve, self.dtype_16, mask=ve_mask, reg_layout=rr.RegLayout.ZERO)
        ho = rr.vcast(vo, self.dtype_16, mask=vo_mask, reg_layout=rr.RegLayout.ONE)
        merged = rr.vbitwise_or(he, ho, mask=b16)
        rr.vstore_strided(
            p_ub,
            nz_off,
            merged,
            b16_full,
            block_stride=block_stride,
            repeat_stride=0,
        )
        return ve, vo

    def _softmax_rest_tail(self, sm_max, sm_sum, sm_exp, rowmask):
        """Online-softmax running-state update."""
        old_max = rr.vload(sm_max, 0)
        new_max = rr.vload(self.tmp_new_max, 0)
        se = rr.vexp_sub(old_max, new_max, mask=rowmask)  # exp(old-new)
        rr.vstore(sm_exp, 0, se, rowmask)
        rr.vstore(sm_max, 0, new_max, rowmask)
        old_sum = rr.vload(sm_sum, 0)
        new_sum = rr.vload(self.tmp_sum, 0)
        ss = rr.vmadd(old_sum, se, new_sum, mask=rowmask)
        rr.vstore(sm_sum, 0, ss, rowmask)


class _SFAOnlineSoftmaxVector(_StableOnlineSoftmaxVector):
    """Online softmax with tail-safe maxima for signed scores."""

    @jit
    def finish_o_cast(self, pv_ch, sm_sum_buf, output_heads, output_head, sm_exp_buf=None):
        """Normalize the final PV result and cast it to the output type."""
        with vf(mode="simd"):
            full, _ = rr.update_mask(64, elem_bits=32)
            zero = rr.vdups(0.0, dtypes.float32, mask=full)
            one = rr.vdups(1.0, dtypes.float32, mask=full)
            for row in range(self.tile_vec_m):
                sum_b = rr.vload_broadcast(sm_sum_buf, row)
                is_zero = rr.veqs(sum_b, 0.0, mask=full)
                safe_sum = rr.vselect(one, sum_b, cond_mask=is_zero)
                if const_expr(sm_exp_buf is not None):
                    exp_b = rr.vload_broadcast(sm_exp_buf, row)
                for col in tuple(range(0, self.tile_d, 64)):
                    off = row * self.tile_d + col
                    cur = rr.vload(pv_ch, off)
                    if const_expr(sm_exp_buf is not None):
                        pre = rr.vload(self.res_o, off)
                        cur = rr.vmadd(pre, exp_b, cur, mask=full)
                    val = rr.vdiv(cur, safe_sum, mask=full)
                    val = rr.vselect(zero, val, cond_mask=is_zero)
                    packed = rr.vcast(val, self.dtype_16, mask=full)
                    rr.vstore_pack(self.o_ub, off, packed, full,
                                   pack_mode="b32_to_b16")

        # Keep the UB producer and GM consumer in the same JIT region.
        for part in tuple(range((output_heads.shape[0] + TILE_M - 1) // TILE_M)):
            for sub in (0, 1):
                rows = min(
                    AIV_ROW_CAPACITY,
                    max(0, output_heads.shape[0] - part * TILE_M - sub * AIV_ROW_CAPACITY),
                )
                if const_expr(rows > 0):
                    row_start = part * TILE_M + sub * AIV_ROW_CAPACITY
                    if output_head == part * TILE_M and self.subblock_idx == sub:
                        dst = tile_slice(output_heads[row_start:row_start + rows, None],
                                        (rows, self.tile_d), (0, 0))
                        mem_copy(dst, reinterpret(self.o_ub, shape=(rows, self.tile_d)))

    def _pass_a_row(
        self,
        qk_ch,
        scale,
        sm_max_dst,
        row,
        row_stride,
        vector_lanes,
        half0_mask,
        half1_mask,
        full_mask,
    ):
        base = row * row_stride
        v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
        v1 = rr.vmuls(rr.vload(qk_ch, base + vector_lanes), scale, mask=half1_mask)
        rr.vstore(qk_ch, base, v0, half0_mask)
        rr.vstore(qk_ch, base + vector_lanes, v1, half1_mask)
        neg_inf = rr.vdups(SOFTMAX_NEG_INF, dtypes.float32, mask=full_mask)
        max0 = rr.vselect(v0, neg_inf, cond_mask=half0_mask)
        max1 = rr.vselect(v1, neg_inf, cond_mask=half1_mask)
        rmax = rr.vreduce_max(rr.vmax(max0, max1, mask=full_mask), mask=full_mask)
        # Initialize an empty first tile without replacing later running maxima.
        if sm_max_dst is not self.tmp_new_max:
            zeros = rr.vdups(0.0, dtypes.float32, mask=full_mask)
            all_masked = rr.vles(rmax, EMPTY_SOFTMAX_THRESHOLD, mask=full_mask)
            rmax = rr.vselect(zeros, rmax, cond_mask=all_masked)
        rr.vstore_first(sm_max_dst, row, rmax)


def _native_head_tile(gm, layout, query_id, head_offset, rows):
    """Fix one physical token in native BSND/TND storage; tile the G,D axes."""
    if const_expr(layout == "BSND"):
        span = gm[query_id // gm.shape[1], query_id % gm.shape[1], None, None]
    else:
        span = gm[query_id, None, None]
    base = tile_slice(span, (gm.shape[-2], gm.shape[-1]), (0, 0))
    return tile_slice(base[head_offset:, None],
                     (rows, gm.shape[-1]), (0, 0))


def _align16_py(value: int) -> int:
    return (value + 15) // 16 * 16


def _bounded_local(buffer, shape, capacity, offset=0, stride=None):
    # Preserve the allocation pitch while narrowing the runtime extent.
    root = reinterpret(buffer, shape=capacity, offset=offset, stride=stride)
    return reinterpret(root, shape=make_tiler(shape, alignment=(1, 1)), stride=stride)


class _SparseMatmul:
    """Cube matmul stages and their L1/L0 buffers."""

    def __init__(self, group_size: int, dtype_16, has_rope: bool):
        self.group_size = group_size
        self.has_rope = has_rope

        self.nd2nz = make_copy_engine(format_transform="nd2nz")
        self.fixpipe = make_copy_engine(split_axis=0)

        # QK produces each K slot; PV consumes it after the final L1 read.
        self.right_root = Channel(
            MemLoc.L1, (TILE_S2, DIM_NOPE), dtype_16,
            depth=PIPELINE_SLOT_COUNT,
        )
        self.k_rope_l1 = None
        if has_rope:
            self.k_rope_l1 = Channel(
                MemLoc.L1, (TILE_S2, DIM_ROPE), dtype_16,
                depth=1,
            ).produce()
        self.q_nope_l1 = Channel(
            MemLoc.L1, (group_size, DIM_NOPE), dtype_16,
            depth=1,
        ).produce()
        self.q_rope_l1 = None
        if has_rope:
            self.q_rope_l1 = Channel(
                MemLoc.L1, (group_size, DIM_ROPE), dtype_16,
                depth=1,
            ).produce()

        # L0A/L0B/L0C double buffers: 32 KiB / 64 KiB / 256 KiB total.
        self.l0a = Channel(
            MemLoc.L0A, (TILE_M, TILE_S2), dtype_16, depth=2
        )
        self.l0b = Channel(
            MemLoc.L0B, (TILE_S2, TILE_S2), dtype_16, depth=2
        )
        self.l0c = Channel(
            MemLoc.L0C, (TILE_M, DIM_NOPE), dtypes.float32, depth=2
        )

    def load_q(self, query_token, query_rope_token):
        mem_copy(self.q_nope_l1, query_token, engine=self.nd2nz)
        if const_expr(self.has_rope):
            mem_copy(self.q_rope_l1, query_rope_token, engine=self.nd2nz)

    def load_sparse_kv(self, workspace_tile, valid_n):
        engine = make_copy_engine(format_transform="nd2nz")
        nope = tile_slice(workspace_tile[0:valid_n, 0:DIM_NOPE],
                         (TILE_S2, DIM_NOPE), (0, 0))
        slot = self.right_root.produce()
        mem_copy(slot, nope, engine=engine)
        if const_expr(self.has_rope):
            rope = tile_slice(workspace_tile[0:valid_n, DIM_NOPE:DIM_QK],
                             (TILE_S2, DIM_ROPE), (0, 0))
            mem_copy(self.k_rope_l1, rope, engine=engine)
        return slot

    def compute_qk(self, right_slot, valid_n):
        """Accumulate four NoPE chunks and an optional 64-feature RoPE tail."""
        m = self.group_size
        score = _bounded_local(
            self.l0c.produce(), (m, valid_n), (m, TILE_S2)
        )
        # L1 retains the TILE_S2 pitch when valid_n is smaller.
        for feature in (0, 128, 256, 384):
            width = 128
            q = reinterpret(
                self.q_nope_l1, shape=(m, width),
                offset=_align16_py(m) * feature * BYTES_PER_16BIT_ELEMENT,
            )
            k_root = reinterpret(
                right_slot, shape=(TILE_S2, width),
                offset=TILE_S2 * feature * BYTES_PER_16BIT_ELEMENT,
            )
            k = _bounded_local(k_root, (valid_n, width), (TILE_S2, width))
            a = reinterpret(self.l0a.produce(), shape=(m, width))
            # QK uses L0B as (N, K); PV loads its L0B operand transposed.
            b = _bounded_local(
                self.l0b.produce(), (valid_n, width), (TILE_S2, width)
            )
            mem_copy(a, q)
            mem_copy(b, k)
            matmul(score, a, b, init=(feature == 0))
        if const_expr(self.has_rope):
            q = reinterpret(self.q_rope_l1, shape=(m, DIM_ROPE))
            k_root = reinterpret(self.k_rope_l1, shape=(TILE_S2, DIM_ROPE))
            k = _bounded_local(
                k_root, (valid_n, DIM_ROPE), (TILE_S2, DIM_ROPE)
            )
            a = reinterpret(self.l0a.produce(), shape=(m, DIM_ROPE))
            b = _bounded_local(
                self.l0b.produce(), (valid_n, DIM_ROPE),
                (TILE_S2, DIM_ROPE)
            )
            mem_copy(a, q)
            mem_copy(b, k)
            matmul(score, a, b, init=False)

    def compute_pv(self, right_slot, p_l1, valid_n):
        """[G,valid_n] @ [valid_n,512], retaining static NZ storage pitch."""
        m = self.group_size
        p = _bounded_local(p_l1, (m, valid_n), (m, TILE_S2))
        a = _bounded_local(
            self.l0a.produce(), (m, valid_n), (m, TILE_S2)
        )
        mem_copy(a, p)
        output = self.l0c.produce()
        for feature in (0, 128, 256, 384):
            v_root = reinterpret(
                right_slot, shape=(TILE_S2, TILE_S2),
                offset=TILE_S2 * feature * BYTES_PER_16BIT_ELEMENT,
            )
            v = _bounded_local(
                v_root, (valid_n, TILE_S2), (TILE_S2, TILE_S2)
            )
            b = _bounded_local(
                self.l0b.produce(), (valid_n, TILE_S2),
                (TILE_S2, TILE_S2),
            )
            mem_copy(b, v, transpose=True)
            dst = reinterpret(output, shape=(m, TILE_S2), offset=TILE_M * feature * 4)
            matmul(dst, a, b, init=True)

    def store_score(self, qk_ub, valid_n):
        """Split L0C score rows across AIVs while preserving TILE_S2 pitch."""
        mem_copy(_bounded_local(qk_ub.produce(), (self.group_size // 2, valid_n),
                                (self.group_size // 2, TILE_S2), stride=(TILE_S2, 1)),
                 _bounded_local(self.l0c.consume(), (self.group_size, valid_n),
                                      (self.group_size, TILE_S2)),
                 engine=self.fixpipe,
                 actual=(self.group_size, valid_n))

    def store_pv(self, pv_ub):
        """L0C PV result -> the cross-core pv_ub Channel, all G rows."""
        mem_copy(pv_ub, self.l0c.consume(), engine=self.fixpipe,
                 actual=(self.group_size, DIM_NOPE))


@kernel(profile=cannbotdsl.ProfileSpec(
    name="SparseFlashAttention_DSL", op_type="SparseFlashAttention_DSL",
    inputs=("query",), outputs=("output",),
))
class _SparseFlashAttentionKernel:
    """Process the query rows assigned to one AIC head group."""

    def __init__(
        self,
        group_size: int,
        sparse_count: int,
        head_chunks: int,
        queries_per_group: int,
        dtype_16,
        use_sinks: bool,
        return_lse: bool,
        layout_query: str = "BSND",
        head_count: int = 64,
        total_queries: int = 1,
        layout_kv: str = "BSND",
        batch_count: int = 1,
        sparse_mode: int = 0,
        has_rope: bool = True,
    ):
        self.group_size = group_size
        self.vec_m = (group_size + 1) // 2
        self.sparse_count = sparse_count
        self.head_chunks = head_chunks
        self.queries_per_group = queries_per_group
        self.dtype_16 = dtype_16
        self.use_sinks = use_sinks
        self.return_lse = return_lse
        self.layout_query = layout_query
        self.layout_kv = layout_kv
        self.batch_count = batch_count
        self.sparse_mode = sparse_mode
        self.has_rope = has_rope
        self.qk_dim = DIM_QK if has_rope else DIM_NOPE
        self.head_count = head_count
        self.total_queries = total_queries

        self.block_idx = get_block_idx()
        self.subblock_idx = get_subblock_id()

        self.matmul = _SparseMatmul(
            group_size,
            dtype_16,
            has_rope,
        )
        self.vector = _SFAOnlineSoftmaxVector(
            self.vec_m,
            TILE_S2,
            DIM_NOPE,
            self.subblock_idx,
            dtype_16=dtype_16,
            preload_num=PIPELINE_SLOT_COUNT,
        )

        self.qk_ub = Channel(
            MemLoc.UB,
            (self.vec_m, TILE_S2),
            dtypes.float32,
            depth=2,
            kind=ChannelKind.CrossCore,
        )
        self.pv_ub = Channel(
            MemLoc.UB,
            (self.vec_m, DIM_NOPE),
            dtypes.float32,
            depth=1,
            kind=ChannelKind.CrossCore,
        )
        # Both AIV partitions form one P tile for the AIC.
        self.p_channel = Channel(
            MemLoc.L1, (group_size, TILE_S2), dtype_16,
            depth=1, kind=ChannelKind.CrossCore,
        )

        # Ping-pong sparse loads with workspace stores on Vec0.
        self.stage0 = Channel(
            MemLoc.UB, (GATHER_CHUNK_ROWS, self.qk_dim), dtype_16, depth=2
        )

        # Channels order GM transfers around Vec1 softmax state updates.
        self.sink_state = Channel(
            MemLoc.UB, (self.vec_m, 1), dtypes.float32, depth=1
        ).produce()
        self.lse_max_state = Channel(
            MemLoc.UB, (self.vec_m, 1), dtypes.float32, depth=1
        ).produce()
        self.lse_sum_state = Channel(
            MemLoc.UB, (self.vec_m, 1), dtypes.float32, depth=1
        ).produce()

    def __call__(
        self,
        output: Tensor,
        query: Tensor,
        key: Tensor,
        indices: Tensor,
        query_rope: Tensor,
        key_rope: Tensor,
        sinks: Tensor,
        maxima: Tensor,
        sums: Tensor,
        workspace: Tensor,
        table: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        used_q: Tensor,
        used_kv: Tensor,
        scale: float,
    ):
        # Map this core to a query group and a head tile.
        core_group = self.block_idx // self.head_chunks
        head_offset = self.block_idx % self.head_chunks * TILE_M

        # Determine the query range assigned to this core.
        first_query = core_group * self.queries_per_group
        query_end = min(first_query + self.queries_per_group, self.total_queries)
        current_query = first_query
        query_count = query_end - first_query

        # Initialize sparse-block traversal and sequence-boundary state.
        sparse_idx = dtypes.int64(0)
        is_first_block = dtypes.int64(1)
        batch = dtypes.int64(0)
        sparse_end = dtypes.int64(0)
        is_query_active = dtypes.int64(0)

        # Nonempty tasks cover at most one KV tile; EMPTY_QUERY_ID marks a bubble.
        task_pipeline = DelayLineGroup(
            PIPELINE_STAGE_COUNT,
            "query_id",
            "valid_kv_count",
            "is_first_block",
            "is_last_block",
            "is_query_active",
        )
        # Each tick gathers a new KV tile while older tiles move through
        # QK, softmax/PV, and output. The final three ticks drain the pipeline.
        max_gather_tasks = query_count * (
            (self.sparse_count + TILE_S2 - 1) // TILE_S2
        )
        for tick in range(max_gather_tasks + PIPELINE_DRAIN_TICKS):
            # Stage 0: gather a KV tile and register its task.
            if current_query < query_end:
                # A new Query needs its batch and valid sparse range.
                if is_first_block != 0:
                    sparse_idx = 0
                    is_query_active = 0
                    sparse_end = 0
                    (
                        batch,
                        unused_kv_length,
                        unused_kv_end,
                        planned_sparse_end,
                        is_query_active,
                    ) = self._prepare_sparse_query(
                        query, key, table, cu_q, cu_kv,
                        used_q, used_kv, current_query,
                    )
                    sparse_end = self._find_valid_end(
                        indices, current_query, planned_sparse_end,
                    )

                # Gather at most one TILE_S2-sized block.
                valid_kv_count = dtypes.int64(0)
                if sparse_idx < sparse_end:
                    valid_kv_count, sparse_idx = self._stage_gather(
                        tick % PIPELINE_SLOT_COUNT,
                        current_query,
                        sparse_idx,
                        sparse_end,
                        batch,
                        key,
                        key_rope,
                        indices,
                        table,
                        cu_kv,
                        workspace,
                    )

                # The last block lets the gather stage move to the next Query.
                is_last_block = 1 if sparse_idx >= sparse_end else 0
                task_pipeline.push(
                    query_id=current_query,
                    valid_kv_count=valid_kv_count,
                    is_first_block=is_first_block,
                    is_last_block=is_last_block,
                    is_query_active=is_query_active,
                )
                is_first_block = 0
                if is_last_block != 0:
                    current_query += 1
                    is_first_block = 1
            else:
                # Fill unused ticks so older tasks can finish the pipeline.
                task_pipeline.push(
                    query_id=EMPTY_QUERY_ID,
                    valid_kv_count=0,
                    is_first_block=0,
                    is_last_block=0,
                    is_query_active=0,
                )

            # Stage 1: compute QK for the tile gathered one tick ago.
            if tick >= 1:
                qk_task = task_pipeline.tap(1)
                if qk_task.query_id >= 0 and qk_task.valid_kv_count > 0:
                    if qk_task.is_first_block != 0:
                        query_rope_tile = None
                        if const_expr(self.has_rope):
                            query_rope_tile = self._query_tile(
                                query_rope, qk_task.query_id, head_offset,
                            )
                        self.matmul.load_q(
                            self._query_tile(
                                query, qk_task.query_id, head_offset,
                            ),
                            query_rope_tile,
                        )
                    self._stage_qk(
                        (tick - 1) % PIPELINE_SLOT_COUNT,
                        workspace,
                        qk_task.valid_kv_count,
                    )

            # Stage 2: update online softmax and compute PV for the tile
            # gathered two ticks ago.
            if tick >= 2:
                compute_task = task_pipeline.tap(2)
                if compute_task.query_id >= 0:
                    state_slot = compute_task.query_id % PIPELINE_SLOT_COUNT
                    if compute_task.is_first_block != 0:
                        self._init_query_state(
                            sinks,
                            head_offset,
                            (
                                compute_task.valid_kv_count
                                if compute_task.is_query_active != 0
                                else -1
                            ),
                            state_slot,
                        )
                    if compute_task.valid_kv_count > 0:
                        self._stage_softmax(
                            compute_task.is_first_block,
                            compute_task.valid_kv_count,
                            (tick - 2) % PIPELINE_SLOT_COUNT,
                            state_slot,
                            scale,
                        )
                        self._stage_pv(compute_task.valid_kv_count)

            # Stage 3: accumulate or finalize the tile gathered three ticks ago.
            if tick >= 3:
                output_task = task_pipeline.tap(3)
                if output_task.query_id >= 0:
                    self._stage_output(
                        output_task.is_first_block,
                        output_task.is_last_block,
                        output_task.valid_kv_count,
                        (tick - 3) % PIPELINE_SLOT_COUNT,
                        output_task.query_id,
                        head_offset,
                        output,
                        maxima,
                        sums,
                    )

            task_pipeline.advance()

    @jit
    def _fill_softmax_sum_one(self, state_slot=0):
        with vf(mode="simd"):
            mask, _ = rr.update_mask(self.vec_m, elem_bits=32)
            ones = rr.vdups(1.0, dtypes.float32, mask=mask)
            rr.vstore(self.vector.sm_sum_tb[state_slot], 0, ones, mask)

    @jit
    def _store_softmax_state(self, max_token, sum_token, state_slot=0):
        mem_copy(reinterpret(self.lse_max_state, shape=(self.vec_m, 1)),
                 self.vector.sm_max_tb[state_slot])
        mem_copy(max_token, reinterpret(self.lse_max_state, shape=(self.vec_m, 1)),
                 engine=make_copy_engine(split_axis=0), part_id=self.subblock_idx)
        mem_copy(reinterpret(self.lse_sum_state, shape=(self.vec_m, 1)),
                 self.vector.sm_sum_tb[state_slot])
        mem_copy(sum_token, reinterpret(self.lse_sum_state, shape=(self.vec_m, 1)),
                 engine=make_copy_engine(split_axis=0), part_id=self.subblock_idx)

    @jit
    def _kv_token(self, gm, token):
        """Select one physical KV row without flattening paged strides."""
        if const_expr(self.layout_kv == "TND"):
            span = gm[token, None, None]
        else:
            span = gm[token // gm.shape[1], token % gm.shape[1], None, None]
        return tile_slice(span, (1, gm.shape[-1]), (0, 0))

    @jit
    def _workspace_tile(self, workspace, task_slot):
        """Return one slot from this block's GM workspace ring."""
        return tile_slice(workspace, (TILE_S2, self.qk_dim),
            (self.block_idx * PIPELINE_SLOT_COUNT + task_slot, 0))

    @jit
    def _find_valid_end(self, indices, query_id, planned_end):
        """Find the first padding entry in a valid-prefix sparse row."""
        lo = dtypes.int64(0)
        hi = planned_end
        while lo < hi:
            mid = (lo + hi) // 2
            if dtypes.int64(indices[query_id, mid]) == INVALID_SPARSE_INDEX:
                hi = mid
            else:
                lo = mid + 1
        return lo

    @jit
    def _stage_gather(
        self, slot, query_id, sparse_idx, sparse_end, batch,
        key, key_rope, indices, table, cu_kv, workspace,
    ):
        """Gather one sparse tile using disjoint, pair-aligned AIV ranges."""
        workspace_tile = self._workspace_tile(workspace, slot)
        tile_rows = min(TILE_S2, sparse_end - sparse_idx)
        rows_per_aiv_round = AIV_COUNT * SPARSE_INDEX_PAIR_SIZE
        pair_loops = (
            tile_rows + rows_per_aiv_round - 1
        ) // rows_per_aiv_round
        split = min(pair_loops * SPARSE_INDEX_PAIR_SIZE, tile_rows)
        local_start = 0 if self.subblock_idx == 0 else split
        local_end = split if self.subblock_idx == 0 else tile_rows
        local_cursor = local_start
        while local_cursor < local_end:
            chunk_rows = min(GATHER_CHUNK_ROWS, local_end - local_cursor)
            stage = self.stage0.produce()
            row = dtypes.int64(0)
            while row < chunk_rows:
                index_pos = sparse_idx + local_cursor + row
                kv_idx0 = dtypes.int64(indices[query_id, index_pos])
                token0 = self._kv_address(
                    key, table, cu_kv, batch, kv_idx0
                )
                if row + SPARSE_INDEX_PAIR_SIZE <= chunk_rows:
                    kv_idx1 = dtypes.int64(indices[query_id, index_pos + 1])
                    token1 = self._kv_address(
                        key, table, cu_kv, batch, kv_idx1
                    )
                    token_gap = (
                        token1 - token0 if token0 < token1 else token0 - token1
                    )
                    source_gap_bytes = (
                        token_gap - 1
                    ) * DIM_NOPE * BYTES_PER_16BIT_ELEMENT
                    if source_gap_bytes >= 0 and source_gap_bytes < INT32_MAX:
                        first_token = token0 if token0 < token1 else token1
                        second_token = token1 if token0 < token1 else token0
                        pair_index = row // SPARSE_INDEX_PAIR_SIZE
                        if const_expr(self.has_rope):
                            _copy_kv_pair(
                                stage,
                                self._kv_token(key, first_token),
                                self._kv_token(key, second_token),
                                self._kv_token(key_rope, first_token),
                                self._kv_token(key_rope, second_token),
                                pair_index,
                            )
                        else:
                            _copy_kv_pair_no_rope(
                                stage,
                                self._kv_token(key, first_token),
                                self._kv_token(key, second_token),
                                pair_index,
                            )
                    else:
                        if const_expr(self.has_rope):
                            _copy_kv_token(
                                stage, self._kv_token(key, token0),
                                self._kv_token(key_rope, token0), row,
                            )
                            _copy_kv_token(
                                stage, self._kv_token(key, token1),
                                self._kv_token(key_rope, token1), row + 1,
                            )
                        else:
                            _copy_kv_token_no_rope(
                                stage, self._kv_token(key, token0), row
                            )
                            _copy_kv_token_no_rope(
                                stage, self._kv_token(key, token1), row + 1
                            )
                    row += SPARSE_INDEX_PAIR_SIZE
                else:
                    if const_expr(self.has_rope):
                        _copy_kv_token(
                            stage, self._kv_token(key, token0),
                            self._kv_token(key_rope, token0), row,
                        )
                    else:
                        _copy_kv_token_no_rope(
                            stage, self._kv_token(key, token0), row
                        )
                    row += 1
            ready = self.stage0.consume()
            dst = tile_slice(
                workspace_tile[local_cursor:local_cursor + chunk_rows, None],
                (GATHER_CHUNK_ROWS, self.qk_dim), (0, 0),
            )
            mem_copy(
                dst,
                _bounded_local(ready, (chunk_rows, self.qk_dim),
                               (GATHER_CHUNK_ROWS, self.qk_dim),
                               stride=(self.qk_dim, 1)),
            )
            local_cursor += chunk_rows
        if tile_rows > 0:
            # Notify AIC after the AIV workspace write completes.
            vec_sync_block_arrive(PIPE.MTE3, slot, mode=2)
        return tile_rows, sparse_idx + tile_rows

    @jit
    def _stage_qk(self, slot, workspace, valid_n):
        """Compute QK scores from the gathered KV tile."""
        # Wait for the AIV workspace write before reading it on AIC.
        cube_sync_block_wait(PIPE.S, slot, mode=2)
        workspace_tile = self._workspace_tile(workspace, slot)
        shared_k = self.matmul.load_sparse_kv(workspace_tile, valid_n)
        self.matmul.compute_qk(shared_k, valid_n)
        self.matmul.store_score(self.qk_ub, valid_n)

    @jit
    def _store_p(self):
        self.vector.store_p(self.p_channel.produce())

    @jit
    def _stage_pv(self, valid_n):
        shared_k = self.matmul.right_root.consume()
        self.matmul.compute_pv(shared_k, self.p_channel.consume(), valid_n)
        self.matmul.store_pv(self.pv_ub.produce())

    @jit
    def _cross_init_state(self, state_slot, valid_count):
        """Reset running softmax max and sum for one state slot."""
        with vf(mode="simd"):
            rows, _ = rr.update_mask(self.vec_m, elem_bits=32)
            zero = rr.vdups(0.0, dtypes.float32, mask=rows)
            initial = rr.vdups(
                SOFTMAX_NEG_INF if valid_count > 0 else 0.0,
                dtypes.float32,
                mask=rows,
            )
            rr.vstore(self.vector.sm_max_tb[state_slot], 0, initial, rows)
            rr.vstore(self.vector.sm_sum_tb[state_slot], 0, zero, rows)

    @jit
    def _query_tile(self, gm, query_id, head_offset):
        return _native_head_tile(gm, self.layout_query, query_id, head_offset, TILE_M)

    @jit
    def _output_tile(self, gm, query_id, head_offset, rows):
        return _native_head_tile(gm, self.layout_query, query_id, head_offset, rows)

    @jit
    def _sequence_length(self, cu, used, batch, capacity, packed):
        """Prefer the used length; otherwise use the packed length or capacity."""
        length = capacity
        if const_expr(packed):
            length = dtypes.int64(cu[batch + 1]) - dtypes.int64(cu[batch])
        if const_expr(used is not None):
            length = dtypes.int64(used[batch])
        return length

    @jit
    def _q_length(self, query, cu, used, batch):
        capacity = query.shape[1] if self.layout_query == "BSND" else query.shape[0]
        return self._sequence_length(cu, used, batch, capacity, self.layout_query == "TND")

    @jit
    def _kv_capacity(self, key, table):
        """Return the logical KV capacity for the active storage layout."""
        if const_expr(self.layout_kv == "BSND"):
            capacity = key.shape[1]
        elif const_expr(self.layout_kv == "PA_BSND"):
            capacity = table.shape[1] * key.shape[1]
        else:
            capacity = key.shape[0]
        return capacity

    @jit
    def _kv_length(self, key, table, cu, used, batch):
        capacity = self._kv_capacity(key, table)
        return self._sequence_length(cu, used, batch, capacity, self.layout_kv == "TND")

    @jit
    def _query_coords(self, query, cu, query_id):
        """Map a flattened Query ID to its batch and position."""
        batch = dtypes.int64(0)
        position = query_id
        if const_expr(self.layout_query == "BSND"):
            batch = query_id // query.shape[1]
            position = query_id % query.shape[1]
        else:
            # Repeated boundaries represent empty packed sequences.
            for b in range(self.batch_count):
                if query_id >= dtypes.int64(cu[b]) and query_id < dtypes.int64(cu[b + 1]):
                    batch = b
                    position = query_id - dtypes.int64(cu[b])
        return batch, position

    @jit
    def _kv_address(self, key, table, cu, batch, kv_idx):
        """Map one logical KV position to a physical row."""
        token = kv_idx
        if const_expr(self.layout_kv == "BSND"):
            token = batch * key.shape[1] + kv_idx
        elif const_expr(self.layout_kv == "TND"):
            token = dtypes.int64(cu[batch]) + kv_idx
        else:
            page = dtypes.int64(table[batch, kv_idx // key.shape[1]])
            token = page * key.shape[1] + kv_idx % key.shape[1]
        return token

    @jit
    def _prepare_sparse_query(self, query, key, table, cu_q, cu_kv,
                              used_q, used_kv, query_id):
        """Resolve this Query's batch, visible KV limit, and sparse scan bound."""
        batch, position = self._query_coords(query, cu_q, query_id)
        q_length = self._q_length(query, cu_q, used_q, batch)
        kv_length = self._kv_length(key, table, cu_kv, used_kv, batch)
        kv_end = kv_length
        if const_expr(self.sparse_mode == SPARSE_MODE_RIGHT_DOWN_CAUSAL):
            kv_end = kv_length - q_length + position + 1
        is_query_active = 1 if position < q_length and kv_end >= 0 else 0
        if const_expr(self.sparse_mode == SPARSE_MODE_RIGHT_DOWN_CAUSAL):
            if kv_end == 0:
                is_query_active = 0
        sparse_end = (
            min(self.sparse_count, kv_end) if is_query_active != 0 else 0
        )
        return batch, kv_length, kv_end, sparse_end, is_query_active

    @jit
    def _init_query_state(self, sinks, head_offset, count, state_slot):
        """Initialize a Query's softmax state and optional sink."""
        self._cross_init_state(state_slot, count)
        if const_expr(self.use_sinks):
            # A negative count marks an inactive Query: do not load its sink.
            if count >= 0:
                # Limit sink loads to valid rows in a partial head group.
                for part in tuple(range(self.head_chunks)):
                    for sub in (0, 1):
                        rows = min(
                            AIV_ROW_CAPACITY,
                            max(0, self.head_count - part * TILE_M - sub * AIV_ROW_CAPACITY),
                        )
                        if const_expr(rows > 0):
                            row_start = part * TILE_M + sub * AIV_ROW_CAPACITY
                            if head_offset == part * TILE_M and self.subblock_idx == sub:
                                source = tile_slice(
                                    sinks[row_start:row_start + rows, None],
                                    (rows, 1),
                                    (0, 0),
                                )
                                mem_copy(
                                    reinterpret(self.sink_state, shape=(rows, 1)),
                                    source,
                                )
                                mem_copy(
                                    reinterpret(
                                        self.vector.sm_max_tb[state_slot],
                                        shape=(rows, 1),
                                    ),
                                    reinterpret(self.sink_state, shape=(rows, 1)),
                                )
                self._fill_softmax_sum_one(state_slot)

    @jit
    def _emit_stats(self, maxima, sums, query_id, head_offset, state_slot):
        if const_expr(self.return_lse):
            mb = tile_slice(maxima[query_id, None, None], (maxima.shape[1], 1), (0, 0))
            sb = tile_slice(sums[query_id, None, None], (sums.shape[1], 1), (0, 0))
            self._store_softmax_state(
                tile_slice(mb, (TILE_M, 1), (head_offset // TILE_M, 0)),
                tile_slice(sb, (TILE_M, 1), (head_offset // TILE_M, 0)), state_slot)

    @jit
    def _write_empty_query(self, output, query_id, head_offset):
        zero_row = Channel(MemLoc.UB, (1, DIM_NOPE), self.dtype_16, depth=1).produce()
        for row in range(AIV_ROW_CAPACITY):
            head = head_offset + self.subblock_idx * AIV_ROW_CAPACITY + row
            if head < self.head_count:
                with vf(mode="simd"):
                    mask, _ = rr.update_mask(128, elem_bits=16)
                    bits = rr.vdups(0, dtypes.uint16, mask=mask)
                    value = rr.vreinterpret(bits, self.dtype_16)
                    for col in tuple(range(0, DIM_NOPE, 128)):
                        rr.vstore(zero_row, col, value, mask)
                mem_copy(self._output_tile(output, query_id, head, 1), zero_row)

    @jit
    def _stage_softmax(self, first, valid_n, slot, state_slot, scale):
        """Update online softmax and stage unnormalized weights for PV."""
        qk = self.qk_ub.consume()
        if const_expr(not self.use_sinks):
            if first != 0:
                self.vector.softmax_first(
                    qk, valid_n, scale, state_slot
                )
            else:
                self.vector.softmax_rest(
                    qk, valid_n, scale, state_slot, slot
                )
        else:
            self.vector.softmax_rest(
                qk, valid_n, scale, state_slot, slot
            )
        self._store_p()

    @jit
    def _stage_output(self, first, last, valid_n, slot, query_id,
                       head_offset, output, maxima, sums):
        """Accumulate or finalize one PV tile, then close the Query if needed."""
        state_slot = query_id % PIPELINE_SLOT_COUNT
        dst = self._output_tile(output, query_id, 0, self.head_count)
        if valid_n > 0:
            pv = self.pv_ub.consume()
            if last != 0:
                if first != 0:
                    self.vector.finish_o_cast(
                        pv, self.vector.sm_sum_tb[state_slot],
                        dst, head_offset,
                    )
                else:
                    self.vector.finish_o_cast(
                        pv, self.vector.sm_sum_tb[state_slot],
                        dst, head_offset, self.vector.sm_exp_tb[slot],
                    )
            elif first != 0:
                self.vector.init_o(pv)
            else:
                self.vector.update_o(pv, self.vector.sm_exp_tb[slot])
        if last != 0:
            if valid_n == 0:
                if first != 0:
                    self._write_empty_query(output, query_id, head_offset)
                else:
                    # An empty final task can carry only the list terminator.
                    self.vector.finish_o_cast(
                        self.vector.res_o, self.vector.sm_sum_tb[state_slot],
                        dst, head_offset,
                    )
            self._emit_stats(maxima, sums, query_id, head_offset, state_slot)


class _SparseFlashAttentionLauncher:
    """Hold JIT launch configuration for native-layout tensors."""

    def __init__(
        self,
        heads,
        sparse_count,
        total_queries,
        head_chunks,
        queries_per_group,
        dtype,
        block_dim,
        use_sinks,
        return_lse,
        layout_query,
        layout_kv,
        batch,
        sparse_mode,
        has_rope=True,
    ):
        self.heads = heads
        self.sparse_count = sparse_count
        self.total_queries = total_queries
        self.head_chunks = head_chunks
        self.queries_per_group = queries_per_group
        self.dtype = dtype
        self.block_dim = block_dim
        self.use_sinks = use_sinks
        self.return_lse = return_lse
        self.layout_query = layout_query
        self.layout_kv = layout_kv
        self.batch = batch
        self.sparse_mode = sparse_mode
        self.has_rope = has_rope

    # Keep this argument order aligned with the kernel and launcher call sites.
    @host
    def launch(
        self,
        output: Tensor,
        query: Tensor,
        key: Tensor,
        indices: Tensor,
        query_rope: Tensor,
        key_rope: Tensor,
        sinks: Tensor,
        maxima: Tensor,
        sums: Tensor,
        workspace: Tensor,
        table: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        used_q: Tensor,
        used_kv: Tensor,
        scale: float,
    ):
        op = _SparseFlashAttentionKernel(
            TILE_M,
            self.sparse_count,
            self.head_chunks,
            self.queries_per_group,
            self.dtype,
            self.use_sinks,
            self.return_lse,
            self.layout_query,
            self.heads,
            self.total_queries,
            self.layout_kv,
            self.batch,
            self.sparse_mode,
            self.has_rope,
        )
        op[self.block_dim](
            output,
            query,
            key,
            indices,
            query_rope,
            key_rope,
            sinks,
            maxima,
            sums,
            workspace,
            table,
            cu_q,
            cu_kv,
            used_q,
            used_kv,
            scale,
        )


def sparse_flash_attention(
    query,
    key,
    value,
    sparse_indices,
    block_table=None,
    query_rope=None,
    key_rope=None,
    sinks=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    scale_value=1.0,
    sparse_block_size=SUPPORTED_SPARSE_BLOCK_SIZE,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=SPARSE_MODE_DENSE,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    attention_mode=SUPPORTED_ATTENTION_MODE,
    return_softmax_lse=False,
):
    """Run sparse MLA attention for valid-prefix, minus-one-padded indices."""
    validate_sparse_flash_attention_args(
        query=query,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        block_table=block_table,
        query_rope=query_rope,
        key_rope=key_rope,
        sinks=sinks,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        sparse_block_size=sparse_block_size,
        layout_query=layout_query,
        layout_kv=layout_kv,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        attention_mode=attention_mode,
        return_softmax_lse=return_softmax_lse,
    )

    return _native_sparse_launch(
        query,
        key,
        query_rope,
        key_rope,
        sparse_indices,
        block_table,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        scale_value,
        return_softmax_lse,
        layout_query,
        layout_kv,
        sparse_mode,
    )


def _calculate_core_partition(total, heads, core_num):
    """Return head-chunk count, queries per group, and launch block count."""
    head_chunks = (heads + TILE_M - 1) // TILE_M
    core_groups = core_num // head_chunks
    if core_groups == 0:
        raise ValueError(
            f"Head tiles ({head_chunks}) exceed available cube cores ({core_num})"
        )
    queries_per_group = (total + core_groups - 1) // core_groups
    used_core_groups = (total + queries_per_group - 1) // queries_per_group
    block_dim = used_core_groups * head_chunks
    return head_chunks, queries_per_group, block_dim


def _native_sparse_launch(
    query,
    key,
    query_rope,
    key_rope,
    sparse_indices,
    table,
    cu_q,
    cu_kv,
    used_q,
    used_kv,
    sinks,
    scale,
    return_lse,
    layout,
    layout_kv,
    sparse_mode,
):
    """Allocate outputs and workspace, then launch the configured kernel."""
    has_rope = query_rope is not None
    batch = query.shape[0] if layout == "BSND" else cu_q.numel() - 1
    total = query.shape[0] * query.shape[1] if layout == "BSND" else query.shape[0]
    heads = query.shape[-2]
    index_capacity = sparse_indices.shape[-1]
    indices = sparse_indices.view(total, index_capacity)
    core_num = torch.npu.get_device_properties(query.device).cube_core_num
    head_chunks, queries_per_group, block_dim = _calculate_core_partition(
        total, heads, core_num
    )

    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)

    # Reserve padded max/sum output buffers for all head tiles.
    state_shape = (total, head_chunks * TILE_M, 1)
    max_out = torch.empty(state_shape, dtype=torch.float32, device=query.device)
    sum_out = torch.empty(state_shape, dtype=torch.float32, device=query.device)

    sink_view = None if sinks is None else sinks.view(heads, 1)
    # Each core reuses its three workspace slots across pipeline ticks.
    qk_dim = DIM_QK if has_rope else DIM_NOPE
    workspace = torch.empty(
        (block_dim * PIPELINE_SLOT_COUNT * TILE_S2, qk_dim),
        dtype=query.dtype,
        device=query.device,
    )
    launcher = _SparseFlashAttentionLauncher(
        heads,
        index_capacity,
        total,
        head_chunks,
        queries_per_group,
        dtypes.bfloat16 if query.dtype == torch.bfloat16 else dtypes.float16,
        block_dim,
        sinks is not None,
        return_lse,
        layout,
        layout_kv,
        batch,
        sparse_mode,
        has_rope,
    )
    launcher.launch(
        output,
        query,
        key,
        indices,
        query_rope,
        key_rope,
        sink_view,
        max_out,
        sum_out,
        workspace,
        table,
        cu_q,
        cu_kv,
        used_q,
        used_kv,
        scale,
    )

    if return_lse:
        max_out = max_out[:, :heads, 0]
        sum_out = sum_out[:, :heads, 0]
        if layout == "BSND":
            # Restore [B, 1, S1, H] stats from the flattened query axis.
            max_out = max_out.view(batch, query.shape[1], heads).unsqueeze(1)
            sum_out = sum_out.view(batch, query.shape[1], heads).unsqueeze(1)
        else:
            max_out = max_out.unsqueeze(0)
            sum_out = sum_out.unsqueeze(0)
    else:
        max_out = torch.empty((0,), dtype=torch.float32, device=query.device)
        sum_out = torch.empty((0,), dtype=torch.float32, device=query.device)
    return output, max_out, sum_out
