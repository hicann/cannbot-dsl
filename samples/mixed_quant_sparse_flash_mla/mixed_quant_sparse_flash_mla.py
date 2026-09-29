# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details on how to use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.

import math
import os
import threading
from itertools import product
import cannbotdsl
import torch
from cannbotdsl.aot import export

from cannbotdsl.channel import Channel
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.types.delay_line import DelayLineGroup
from cannbotdsl.ops.arch import get_subblock_id, get_block_idx, get_subblock_dim
from cannbotdsl.types._integer import Int32, Int64
from cannbotdsl import dtypes
from cannbotdsl import Dim, ProfileSpec, TensorSpec, host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl import ChannelKind, MemLoc, PIPE, Tensor
from cannbotdsl.tensor import (
    ceil_div,
    tile_slice,
    reinterpret,
)
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.sync import (
    cube_sync_block_arrive,
    cube_sync_intra_wait,
    global_sync_all,
    vec_sync_block_wait,
    vec_sync_all,
    vec_busy_wait,
    vec_sync_intra_arrive,
    vec_sync_notify,
    vec_sync_wait,
)
from cannbotdsl.ops.sync import channel_rewind
from cannbotdsl.ops.scalar import vec_store_bypass
from cannbotdsl.ops import reg as rr

# Metadata constants and public host entry are re-exported from the metadata module.
if __package__:
    from .mixed_quant_sparse_flash_mla_metadata import (
        AIC_CORE_MAX_NUM, MQSMLA_METADATA_TOTAL_SIZE, FA_METADATA_SIZE,
        FA_BN2_START_INDEX, FA_M_START_INDEX, FA_S2_START_INDEX,
        FA_BN2_END_INDEX, FA_M_END_INDEX, FA_S2_END_INDEX,
        FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, FD_METADATA_BASE, FD_METADATA_SIZE,
        FD_CORE_ENABLE_INDEX, FD_BN2_IDX_INDEX, FD_M_IDX_INDEX, FD_WORKSPACE_IDX_INDEX,
        FD_WORKSPACE_NUM_INDEX, FD_M_START_INDEX, FD_USED_VEC_NUM_WORD,
        _get_cube_core_num, mixed_quant_sparse_flash_mla_metadata,
    )
else:
    from mixed_quant_sparse_flash_mla_metadata import (
        AIC_CORE_MAX_NUM, MQSMLA_METADATA_TOTAL_SIZE, FA_METADATA_SIZE,
        FA_BN2_START_INDEX, FA_M_START_INDEX, FA_S2_START_INDEX,
        FA_BN2_END_INDEX, FA_M_END_INDEX, FA_S2_END_INDEX,
        FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, FD_METADATA_BASE, FD_METADATA_SIZE,
        FD_CORE_ENABLE_INDEX, FD_BN2_IDX_INDEX, FD_M_IDX_INDEX, FD_WORKSPACE_IDX_INDEX,
        FD_WORKSPACE_NUM_INDEX, FD_M_START_INDEX, FD_USED_VEC_NUM_WORD,
        _get_cube_core_num, mixed_quant_sparse_flash_mla_metadata,
    )

# Address precomputation partitions one UB arena by runtime K/page-table widths.
# Dense power-of-two pages use vector addressing; other layouts use scalar addressing.
_ADDR_VEC_OVERRIDE = None


# NZ face stride in BF16 elements; UB padding is skipped by copy-out.
_NZ_ROW = 16
_NZ_PAD_ROWS = 17
_NZ_CHUNK = _NZ_PAD_ROWS * _NZ_ROW
# Output HardEvent IDs are separate from intra-core and block-sync IDs.
# Keep both directions until removal is validated for every output alias path.
_OUT_V_TO_MTE3_EVENT_ID = 0
_OUT_MTE3_TO_V_EVENT_ID = 1
_FD_READY_BASE = FD_USED_VEC_NUM_WORD + 1
_FD_ASYNC_MODE_WORD = FD_USED_VEC_NUM_WORD + 50

# Address-prepass buffers are reclaimed after the block handshake.
# Main-loop and FD reduction buffers reuse UB at separate phases.
# Ascend950DT_9582 / dav-3510: ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE reports
# 216 KiB usable per AIV. Do not use the DSL allocator's larger 256 KiB ceiling.
_ADDR_UB_BYTES = 216 * 1024
# Separate small-pool owners retain MTE2/MTE3 overlap between prepass rows.
_ADDR_COMPACT_BT = 2048
_ADDR_COMPACT_K = 1024

# Compile-time shape contract.
D = 512  # Logical head dimension: nope[448] followed by rope[64].
N1 = 64                  # Q heads
N2 = 1  # KV heads; the query-to-KV head ratio is 64.
TILE_N = 128  # KV tokens per S2 task.

# Per-token physical layout: nope[448], rope[64], then inline BF16 scales.
# FP8 E4M3 uses one scale per 32 values: 512 data bytes + 32 scale bytes.
# FP4 E2M1 uses one scale per 16 values: 256 packed bytes + 64 scale bytes.
# The even FP4 value occupies the low nibble; inputs use uint8 byte views.
KV_ROW_BYTES_ORI = 544
KV_ROW_BYTES_CMP = 320
ROW_BF16_ORI = KV_ROW_BYTES_ORI // 2  # ORI row width in BF16 elements.
SCALE_BF16_OFF_ORI = 256  # ORI scale offset in BF16 elements.
SCALE_BF16_OFF_CMP = 128  # CMP scale offset in BF16 elements.

# FD workspace: normalized O slots followed by per-slot max/sum.
# Each core stages at most two partial rows: a leading continuation and a trailing split.
FD_SLOTS_PER_CORE = 2
FD_O_ELEMS = N1 * D  # FP32 output elements per slot.
FD_MS_ELEMS = 2 * N1  # FP32 max[64] and sum[64] per slot.
FD_SLOT_ELEMS = FD_O_ELEMS + FD_MS_ELEMS  # 32896 fp32 = 131584 B
FD_BYTES_PER_CORE = FD_SLOTS_PER_CORE * FD_SLOT_ELEMS * 4  # 263168
WS_BYTES_PER_CORE = FD_BYTES_PER_CORE                    # 263168

# FD is selected solely by runtime metadata. Partial O is written directly
# from res_o; reduction storage reuses UB after the global barrier.
_FD_NEG_INF = -1.0e38
_FD_MP_SLOTS = AIC_CORE_MAX_NUM  # One row may span every launched Cube core.

# Dequantized KV is written directly from UB to the shared L1 ring.

# Default ORI offsets; _vec0_body selects the offsets for each pool.
_ROW_BF16 = ROW_BF16_ORI  # Both input pools use the ORI 544-byte UB row stride.
_SCALE_BF16_OFF = SCALE_BF16_OFF_ORI  # ORI scale offset in the shared input layout.

_VL = 128  # BF16 vector width for dequantization.
_TILE_ROWS = 16  # Rows per dequantization subtile and per NZ face.
_ADDR_HALF = 64  # Row indices processed by each address vector.
# Sentinel beyond every valid query index; prevents reads past the final cu_q span.
_ADDR_M_INF = 1 << 40


TILE = 128
D_TILES = D // TILE  # Four D chunks: QK reduction axis and PV output axis.

# L1 per AIC: KV 384 KiB + Q 96 KiB + P 32 KiB = 512 KiB.
# KV slots remain live through QK and PV; Q half-slots remain live for a query.
_KV_RING = 3  # KV slots shared by QK and PV.

_Q_RING = 3  # Three BF16 [64,256] Q half-slots, totaling 96 KiB.

_P_L1_DEPTH = 2  # Cross-core P slots from AIV store_p to AIC PV.

# Eight cross-core Channel slots occupy sync ids 0..7. The address-table
# publication barrier uses id 9; AIV1's intra-core token is offset by 16.
_AIV1_ID_OFFSET = 16

_KV_NUM_PER_LOOP = 128
_TILE_SIZE = 64
_COMBINE_DIM = _ROW_BF16 * 2

# Rank contracts and hardware limits (not tensor data values).
_TENSOR_RANK_1D = 1
_TENSOR_RANK_2D = 2
_TENSOR_RANK_3D = 3
_TENSOR_RANK_4D = 4
_MAX_PA_BLOCK_SIZE = 1024
# UB/GM paired-copy source stride is encoded in an unsigned 39-bit field.
_DMA_SOURCE_STRIDE_MAX_BYTES = (1 << 39) - 1

_COMPILED_KERNELS = {}
_COMPILED_KERNEL_LOCK = threading.Lock()


def _addr_vec_mode():
    return _ADDR_VEC_OVERRIDE or "pre"


def _addr_vec_eligible(ori_kv, ori_bt, ori_idx, cmp_kv, cmp_bt, cmp_idx):
    """Select vector addressing only when each pool's aligned staging fits UB.

    ORI/CMP run sequentially and reuse the same arena. Main-loop UB is allocated
    after the prepass publication barrier and rewind, so it is not additive.
    """
    def pool_supports_vector_addressing(kv, bt, idx, row_bytes):
        block_size, table_width, index_width = kv.shape[1], bt.shape[1], idx.shape[2]
        if not all(isinstance(n, int) for n in (block_size, table_width, index_width)):
            return False
        return (block_size >= 16 and block_size & (block_size - 1) == 0
                and kv.stride(0) == block_size * row_bytes
                and _addr_ub_bytes(table_width, index_width) <= _ADDR_UB_BYTES)

    return (pool_supports_vector_addressing(ori_kv, ori_bt, ori_idx, KV_ROW_BYTES_ORI)
            and (cmp_kv is None or pool_supports_vector_addressing(
                cmp_kv, cmp_bt, cmp_idx, KV_ROW_BYTES_CMP)))


def _addr_ub_bytes(table_width, index_width):
    """Include DMA alignment and the output's padded 16-column consumer tail."""
    return 4 * (_addr_tab_w(table_width) + 2 * _addr_tab_w(index_width))


def _addr_tab_w(k):
    """Round K up to 16 columns; the prepass clamps padded columns to the last valid index."""
    return ((k + _TILE_ROWS - 1) // _TILE_ROWS) * _TILE_ROWS


def _copy_addr_to_gm(gm, ch, elem_off, n_elems):
    mem_copy(gm.view(gm.shape[0] * gm.shape[1])[elem_off:elem_off + n_elems, ].view(1, n_elems),
             reinterpret(ch, shape=(1, n_elems)))


def _copy_addr_from_gm(ch, gm, elem_off, burst_bytes):
    count = burst_bytes // 4
    mem_copy(reinterpret(ch, shape=(1, count)),
             gm.view(gm.shape[0] * gm.shape[1])[elem_off:elem_off + count, ].view(1, count))


def _i64(gm, *idx):
    raw = gm[idx] if len(idx) > 1 else gm[idx[0]]
    # Extend int32 GM indices to int64 before physical-address multiplication.
    return dtypes.int64(raw)


class Kvcache:
    """Resolve PA_BBND block-table indices to physical pages and row offsets.

    Block-table values are global physical page indices; batch selects the table.
    Runtime page sizes need not be powers of two.
    """

    @jit
    def pa_blk_off(self, gm_block_table, s2_idx, bo_idx, bs):
        # Common page size avoids runtime division; all other sizes remain valid.
        blk = dtypes.int64(0)
        off = dtypes.int64(0)
        if bs == 128:
            blk = s2_idx // 128
            off = s2_idx % 128
        else:
            blk = s2_idx // bs
            off = s2_idx % bs
        return _i64(gm_block_table, bo_idx, blk), off


class Matmul:
    """Load Q/KV into L1, compute QK and PV, and send results through FIXPIPE.

    Three Q [64,256] BF16 half-slots occupy 96 KiB and remain live through
    the last QK read of each query. Three KV [128,512] slots occupy 384 KiB
    and remain live through both QK and PV. Channel ownership prevents early
    reuse. Two shared [64,512] FP32 L0C slots hold QK/PV; QK uses 128 columns.
    """

    def __init__(self, tile_cube_m, tile_vec_m, tile_n):
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = D

        self.nd2nz = make_copy_engine(format_transform="nd2nz")
        self.fixpipe = make_copy_engine(split_axis=0)

        # Two half-slots remain live through the query's final QK tile.
        self.q_ring = Channel(MemLoc.L1, (tile_cube_m, D // 2),
                              dtypes.bfloat16, depth=_Q_RING)
        self.kv_ring = Channel(MemLoc.L1, (tile_n, D), dtypes.bfloat16,
                               depth=_KV_RING, kind=ChannelKind.CrossCore)
        self.l0a = Channel(MemLoc.L0A, shape=(tile_cube_m, TILE), dtype=dtypes.bfloat16, depth=2)
        self.l0a_p = Channel(MemLoc.L0A, shape=(tile_cube_m, tile_n), dtype=dtypes.bfloat16, depth=2)
        self.l0b = Channel(MemLoc.L0B, shape=(TILE, TILE), dtype=dtypes.bfloat16, depth=2)
        # Shared two-slot L0C ring; matmul produces and FIXPIPE consumes.
        self.l0c = Channel(MemLoc.L0C, shape=(tile_cube_m, D), dtype=dtypes.float32,
                           depth=2, kind=ChannelKind.SameCore)

    @jit
    def load_q(self, q_tile_gm, m_seq):
        for half in range_constexpr(2):
            mem_copy(self.q_ring.produce(),
                     tile_slice(q_tile_gm, (self.tile_cube_m, D // 2), (0, half)),
                     engine=self.nd2nz)

    @jit
    def load_qk(self, q_tile_gm, m_seq, tick):
        # Load Q for the first nonempty row; subsequent rows are prefetched by bmm1.
        # KV is already in L1 and remains live through QK and PV.
        if tick == dtypes.int64(0):
            self.load_q(q_tile_gm, m_seq)

    def bmm1_fanout_q(self, kv_slot, q_half0, q_half1):
        qk_dst = tile_slice(self.l0c.produce(), (self.tile_cube_m, TILE), (0, 0))
        for j in range_constexpr(D_TILES):
            q_half = q_half0 if j < 2 else q_half1
            self._bmm1_chunk_fanout(qk_dst, kv_slot, q_half, j, init=(j == 0))

    def store_s(self, qk_ub_ch):
        # FIXPIPE sends S from L0C to the paired AIV qk_ub slots.
        # Split-M assigns heads [0:32] to AIV0 and [32:64] to AIV1.
        # store_p and output finalization use the same head partition.
        mem_copy(qk_ub_ch.produce(), reinterpret(self.l0c.consume(), shape=(self.tile_cube_m, TILE)),
                 engine=self.fixpipe)

    def compute_pv_fanout(self, p_l1_ch, pv_ub_ch, kv_slot, actual_n):
        # PV[64,512] = P[64,128] @ KV[128,512] at pipeline lag 3.
        # P contains exp(score - running_max), before denominator normalization.
        # Load P once into L0A; transpose the shared KV L1 slot through L0B.
        # Initialize each of four output chunks, then send the full PV through FIXPIPE.
        # Channels manage the cross-core ownership transitions.
        p_slot = p_l1_ch.consume()
        a_slot = self.l0a_p.produce()
        mem_copy(a_slot, p_slot)

        # Only reduce initialized 16-row subblocks; masked P=0 cannot suppress NaN V.
        pv_k = ceil_div(actual_n, _TILE_ROWS) * _TILE_ROWS
        a_view = reinterpret(a_slot, shape=(self.tile_cube_m, pv_k),
                             stride=(self.tile_n, 1))
        acc = self.l0c.produce()
        for n in range(D_TILES):
            b_slot = self.l0b.produce()
            mem_copy(b_slot, tile_slice(kv_slot, (self.tile_n, TILE), (0, n)),
                     transpose=True)
            b_view = reinterpret(b_slot, shape=(pv_k, TILE), stride=(TILE, 1))
            pv_output_chunk = tile_slice(acc, (self.tile_cube_m, TILE), (0, n))
            matmul(pv_output_chunk, a_view, b_view, init=True)
        mem_copy(pv_ub_ch.produce(), self.l0c.consume(), engine=self.fixpipe)

    def _bmm1_chunk_fanout(self, qk_slot, kv_slot, q_half, j, init):
        a_slot = self.l0a.produce()
        b_slot = self.l0b.produce()
        mem_copy(a_slot, tile_slice(q_half, (self.tile_cube_m, TILE), (0, j % 2)))
        mem_copy(b_slot, tile_slice(kv_slot, (self.tile_n, TILE), (0, j)))
        matmul(qk_slot, a_slot, b_slot, init=init)


class Vector:
    """Compute online softmax and output accumulation for 32 heads per AIV.

    For old state (m,l,o) and scaled scores s:
      m_new = max(m, max(s)); alpha = exp(m-m_new); p = exp(s-m_new).
      l_new = alpha*l + sum(p); o_new = alpha*o + BF16(p) @ KV.
    Only valid columns contribute to exponent/P; masked zeros can raise max.
    Sinks seed (m=sink,l=1,o=0) without a value contribution. Final output is
    BF16(o/l), with LSE=log(l)+m. A zero sink still contributes exp(0)=1.
    Double-buffer max/sum by nonempty query and alpha by tick. One res_o
    buffer serves consecutive queries, with copy completion before reuse.
    """

    def __init__(self, tile_vec_m, tile_n, tile_d, subblock_idx, sinks_span):
        self.sinks_span = sinks_span
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = tile_d
        self.subblock_idx = subblock_idx

        # Double-buffer max/sum by query and exp by task tick.
        self.sm_max_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(2)]
        self.sm_sum_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(2)]
        self.sm_exp_tb = [Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32) for _ in range(2)]
        # Partial weights are needed only after the main-loop rewind.
        self.fd_mp_tb = []

        self.tmp_new_max = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.tmp_sum = Buffer(MemLoc.UB, (tile_vec_m, 1), dtypes.float32)
        self.lse_ub = Buffer(MemLoc.UB, shape=(tile_vec_m, 1), dtype=dtypes.float32)
        self.res_o = Buffer(MemLoc.UB, (tile_vec_m, tile_d), dtypes.float32)
        # BF16 output aliases the first 32 KiB of the 64 KiB FP32 res_o buffer.
        self.out_view = reinterpret(self.res_o, dtypes.bfloat16,
                                    (tile_vec_m, tile_d))

        p_n1_pad = 32 // 2
        self.p_ub = Channel(
            MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=dtypes.bfloat16, depth=2,
            data_format="nz", n1_pad=p_n1_pad,
        )
        self.sinks_ub = Buffer(MemLoc.UB, shape=(1, self.sinks_span), dtype=dtypes.float32)

    @jit
    def softmax_rest(self, qk_ch, scale, m_axis_triple: int, tile_triple: int, actual_n):
        # Merge the current tile into the running state, including a seeded first tile.
        # ORI and CMP share one max/sum slot and one continuous tile sequence.
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        sm_exp = self.sm_exp_tb[tile_triple]
        vector_length = 2048 // 32
        tile_n = self.tile_n
        p_slot = self.p_ub.produce()
        m0, s_m1, s_m0, block_stride = self._nz_params(p_slot)
        with vf(mode="simd"):
            rows = qk_ch.shape[0]
            src_row_stride = qk_ch.stride[0]
            rowmask, _ = rr.update_mask(rows, elem_bits=32)
            if actual_n == tile_n:
                self._softmax_tile_max(qk_ch, scale, tile_n, True)
            else:
                self._softmax_tile_max(qk_ch, scale, actual_n, False)
            rr.vmem_bar("vst_vld")
            nm = rr.vmax(rr.vload(sm_max, 0), rr.vload(self.tmp_new_max, 0), mask=rowmask)
            rr.vstore(self.tmp_new_max, 0, nm, rowmask)
            rr.vmem_bar("vst_vld")
            for row in range(rows):
                full, _ = rr.update_mask(vector_length, elem_bits=32)
                b16_full, _ = rr.update_mask(tile_n, elem_bits=16)
                ve_mask, _ = rr.update_mask((actual_n + 1) // 2, elem_bits=32)
                vo_mask, _ = rr.update_mask(actual_n // 2, elem_bits=32)
                b16, _ = rr.update_mask(actual_n, elem_bits=16)
                base = row * src_row_stride
                nz_off = (row // m0) * s_m1 + (row % m0) * s_m0
                ve, vo = self._softmax_fold_row(
                    p_slot, qk_ch, base, nz_off, self.tmp_new_max, row,
                    ve_mask, vo_mask, b16, b16_full, block_stride
                )
                rsum = rr.vreduce_sum(rr.vadd(ve, vo, mask=full), mask=ve_mask)
                rr.vstore_first(self.tmp_sum, row, rsum)
            rr.vmem_bar("vst_vld")
            self._softmax_rest_tail(sm_max, sm_sum, sm_exp, rowmask)

    @jit
    def seed_from_sinks(self, sinks_view, m_axis_triple: int):
        # Seed the first part with max=sink and sum=1.
        # The sink contributes denominator mass without a value contribution.
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        span = self.sinks_span
        with vf(mode="simd"):
            full, _ = rr.update_mask(2048 // 32, elem_bits=32)
            one_reg = rr.vdups(1.0, dtypes.float32, mask=full)
            for row in tuple(range(self.tile_vec_m)):
                rr.vstore_first(sm_max, row, rr.vload_broadcast(sinks_view, row % span))
                rr.vstore_first(sm_sum, row, one_reg)
            rr.vmem_bar("vst_vld")

    @jit
    def seed_empty(self, m_axis_triple: int):
        # Seed FD continuation parts with max=FP32 lower bound and sum=0.
        # Only the first part includes sinks; alpha underflows to zero on continuation
        # so its first tile starts an independent online-softmax accumulation.
        sm_max = self.sm_max_tb[m_axis_triple]
        sm_sum = self.sm_sum_tb[m_axis_triple]
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            neg = rr.vdups(_FD_NEG_INF, dtypes.float32, mask=rowmask)
            zero = rr.vdups(0.0, dtypes.float32, mask=rowmask)
            rr.vstore(sm_max, 0, neg, rowmask)
            rr.vstore(sm_sum, 0, zero, rowmask)
            rr.vmem_bar("vst_vld")

    def store_p(self, p_l1_ch):
        # Each AIV sends its 32-head P half to L1; two full slots occupy 32 KiB.
        piece = tile_slice(p_l1_ch.produce(), (self.tile_vec_m, self.tile_n), (self.subblock_idx, 0))
        mem_copy(piece, self.p_ub.consume())

    @jit
    def init_o(self, pv_ch):
        # The first tile has no prior value contribution; initialize res_o from PV.
        mem_copy(reinterpret(self.res_o, shape=(pv_ch.shape[0], self.tile_d)), pv_ch)

    @jit
    def update_o(self, pv_ch, exp_idx: int):
        # Accumulate the FP32 numerator: new_o = alpha * old_o + PV.
        # The same tick supplies alpha and PV; P is BF16 and matmul accumulates FP32.
        # Reuse the full mask across eight 64-lane D chunks.
        sm_exp_buf = self.sm_exp_tb[exp_idx]
        vector_length = 2048 // 32
        with vf(mode="simd"):
            full = rr.full_mask()
            for row in range(self.tile_vec_m):
                exp_b = rr.vload_broadcast(sm_exp_buf, row)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, vector_length)):
                    off = base + col
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    o = rr.vmadd(pre, exp_b, cur, mask=full)
                    rr.vstore(self.res_o, off, o, full)

    @jit
    def update_o_last(self, pv_ch, exp_idx: int, sum_idx: int):
        # Normalize the final FP32 accumulation before converting it to BF16.
        sm_exp_buf = self.sm_exp_tb[exp_idx]
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        vector_length = 2048 // 32
        with vf(mode="simd"):
            full = rr.full_mask()
            one_reg = rr.vdups(1.0, dtypes.float32)
            for row in range(self.tile_vec_m):
                exp_b = rr.vload_broadcast(sm_exp_buf, row)
                sum_b = rr.vload_broadcast(sm_sum_buf, row)
                inv_sum = rr.vdiv(one_reg, sum_b, mask=full)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, vector_length)):
                    off = base + col
                    pre = rr.vload(self.res_o, off)
                    cur = rr.vload(pv_ch, off)
                    o = rr.vmadd(pre, exp_b, cur, mask=full)
                    o = rr.vmul(o, inv_sum, mask=full)
                    rr.vstore(self.res_o, off, o, full)

    @jit
    def init_o_last(self, pv_ch, sum_idx: int):
        # For a single-tile query, initialize O directly from PV / final_sum.
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        vector_length = 2048 // 32
        with vf(mode="simd"):
            full = rr.full_mask()
            one_reg = rr.vdups(1.0, dtypes.float32)
            for row in range(self.tile_vec_m):
                sum_b = rr.vload_broadcast(sm_sum_buf, row)
                inv_sum = rr.vdiv(one_reg, sum_b, mask=full)
                base = row * self.tile_d
                for col in tuple(range(0, self.tile_d, vector_length)):
                    off = base + col
                    cur = rr.vload(pv_ch, off)
                    rr.vstore(self.res_o, off, rr.vmul(cur, inv_sum, mask=full), full)

    def finalize_o(self, o_tile_gm, sum_idx: int,
                   lse_gm=None, lse_base=None):
        # Output aliases res_o: wait for this MTE3 copy before the next query overwrites res_o.
        # LSE and output copies run in order after the cast completes.
        half = tile_slice(o_tile_gm, (self.tile_vec_m, self.tile_d), (self.subblock_idx, 0))

        if const_expr(lse_gm is not None):
            self._finalize_lse_vf(lse_gm, lse_base, sum_idx)

        self._cast_output()
        vec_sync_notify(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        mem_copy(half, self.out_view)
        # Publish completion immediately after this output copy, before later L1 stores.
        # A final unused event belongs to this launch only.
        vec_sync_notify(PIPE.MTE3, PIPE.V, _OUT_MTE3_TO_V_EVENT_ID)

    @jit
    def wait_prev_out_copy(self):
        # Wait only when a preceding task published the output-copy completion event.
        vec_sync_wait(PIPE.MTE3, PIPE.V, _OUT_MTE3_TO_V_EVENT_ID)

    @jit
    def finalize_empty(self, o_tile_gm, sinks_gm, lse_gm=None, lse_base=None):
        # Empty rows produce O=0 and LSE=sink. Drain the previous output copy
        # before overwriting the shared output buffer.
        vec_sync_notify(PIPE.MTE3, PIPE.V, _OUT_MTE3_TO_V_EVENT_ID)
        vec_sync_wait(PIPE.MTE3, PIPE.V, _OUT_MTE3_TO_V_EVENT_ID)
        half = tile_slice(o_tile_gm, (self.tile_vec_m, self.tile_d), (self.subblock_idx, 0))
        with vf(mode="simd"):
            full, _ = rr.update_mask(128, elem_bits=16)
            zero = rr.vdups(0.0, dtypes.bfloat16, mask=full)
            for offset in range(0, self.tile_vec_m * self.tile_d, 128):
                rr.vstore(self.out_view, offset, zero, full)
        vec_sync_notify(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        mem_copy(half, self.out_view)
        if const_expr(lse_gm is not None):
            # Keep lse_ub's producer on V, as in _finalize_lse_vf; using
            # MTE2 here would change the shared buffer's event protocol.
            mem_copy(self.sinks_ub, tile_slice(sinks_gm, (1, self.sinks_span),
                                            (0, self.subblock_idx)))
            with vf(mode="simd"):
                mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
                rr.vstore(self.lse_ub, 0, rr.vload(self.sinks_ub, 0), mask)
            mem_copy(tile_slice(lse_gm, (self.tile_vec_m,), (lse_base,)),
                     reinterpret(self.lse_ub, shape=(self.tile_vec_m,)))

    # Keep each FD VF in a separate JIT method: register/mask locals must not
    # become loop-carried scalar values in a surrounding runtime loop.

    @jit
    def stage_partial(self, fd_o, fd_ms, slot, sum_idx: int, skip_o: int = 0):
        # Stage the normalized FP32 partial O and its max/sum for this AIV.
        # The caller normalizes O using init_o_last/update_o_last before staging.
        # O is one 64 KiB V→MTE3 copy; max/sum reuse the 32-FP32 lse buffer.
        # Workspace layout: O[slots*64,512], followed by max/sum[slots*128].
        # Each statistics slot contains max[0:64] followed by sum[0:64].
        sub = self.subblock_idx
        vec_sync_notify(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        if skip_o == 0:
            mem_copy(tile_slice(fd_o, (self.tile_vec_m, self.tile_d),
                                (slot * (N1 // self.tile_vec_m) + sub, 0)),
                     self.res_o)
        self._fd_ms_to_ub(0, sum_idx)
        mem_copy(tile_slice(fd_ms, (self.tile_vec_m,),
                           (slot * (FD_MS_ELEMS // self.tile_vec_m) + sub,)),
                 reinterpret(self.lse_ub, shape=(self.tile_vec_m,)))
        self._fd_ms_to_ub(1, sum_idx)
        mem_copy(tile_slice(fd_ms, (self.tile_vec_m,),
                           (slot * (FD_MS_ELEMS // self.tile_vec_m) + (N1 // self.tile_vec_m) + sub,)),
                 reinterpret(self.lse_ub, shape=(self.tile_vec_m,)))
        vec_sync_notify(PIPE.MTE3, PIPE.V, _OUT_MTE3_TO_V_EVENT_ID)

    @jit
    def fd_reduce(self, fd_o, fd_ms, fd_ub, out_gm, lse_gm,
                  row, slot0, k, h0):
        # Reduce k normalized partials for the 32 heads beginning at h0.
        # M = max(m_p); t_p = exp(m_p-M)*s_p; G = sum(t_p).
        # O = sum((t_p/G)*O_p); LSE = M + log(G).
        # Accumulate in metadata slot order for deterministic reassociation.
        # After the barrier, reuse main-loop UB for max/sum and the output accumulator.
        # weights remain in registers; res_o accumulates, fd_ub receives O parts.
        sub = h0 // self.tile_vec_m
        self._fd_init_acc()
        ms_offset = slot0 * FD_MS_ELEMS
        ms_count = k * FD_MS_ELEMS
        ms_source = fd_ms[ms_offset:ms_offset + ms_count, ].view(k, FD_MS_ELEMS)
        mem_copy(self.fd_ms_ub.produce(),
                 tile_slice(ms_source, (_FD_MP_SLOTS, FD_MS_ELEMS), (0, 0)))
        ms_buf = self.fd_ms_ub.consume()
        for p in range(k):
            self._fd_acc_max_save(p, ms_buf)
        for p in range(k):
            self._fd_acc_sum(p, ms_buf)
        # Optionally store LSE = M + log(G).
        if const_expr(lse_gm is not None):
            self._fd_lse_to_ub(self.lse_ub.produce())
            mem_copy(tile_slice(lse_gm, (self.tile_vec_m,), (row * 2 + sub,)),
                     reinterpret(self.lse_ub.consume(), shape=(self.tile_vec_m,)))
        # Accumulate normalized partials using cached weights t_p/G.
        mem_copy(fd_ub.produce(),
                 tile_slice(fd_o, (self.tile_vec_m, self.tile_d),
                            (slot0 * (N1 // self.tile_vec_m) + sub, 0)))
        self._fd_acc_full(fd_ub.consume(), 0, True)
        for p in range(1, k):
            mem_copy(fd_ub.produce(),
                     tile_slice(fd_o, (self.tile_vec_m, self.tile_d),
                                ((slot0 + p) * (N1 // self.tile_vec_m) + sub, 0)))
            self._fd_acc_full(fd_ub.consume(), p, False)
        # Cast the normalized result in place and write its 32-head tile.
        # The pre-reduction barrier drained old MTE3 operations; the following
        # V-to-MTE3 dependency still protects this newly computed output.
        self._cast_output()
        vec_sync_notify(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        mem_copy(tile_slice(out_gm, (self.tile_vec_m, self.tile_d),
                           (row * (N1 // self.tile_vec_m) + sub, 0)),
                 self.out_view)

    @jit
    def fd_reduce_local_first(self, fd_o, fd_ms, fd_ub, out_gm, lse_gm,
                              row, slot0, h0):
        # The first normalized partial remains in this AIV's res_o. Only the
        # second partial crosses GM; readiness is polled before this call.
        sub = h0 // self.tile_vec_m
        self._fd_init_acc()
        ms_offset = slot0 * FD_MS_ELEMS
        ms_source = fd_ms[ms_offset:ms_offset + 2 * FD_MS_ELEMS, ].view(2, FD_MS_ELEMS)
        mem_copy(self.fd_ms_ub.produce(),
                 tile_slice(ms_source, (_FD_MP_SLOTS, FD_MS_ELEMS), (0, 0)))
        ms_buf = self.fd_ms_ub.consume()
        for p in range_constexpr(2):
            self._fd_acc_max_save(p, ms_buf)
        for p in range_constexpr(2):
            self._fd_acc_sum(p, ms_buf)
        if const_expr(lse_gm is not None):
            self._fd_lse_to_ub(self.lse_ub.produce())
            mem_copy(tile_slice(lse_gm, (self.tile_vec_m,), (row * 2 + sub,)),
                     reinterpret(self.lse_ub.consume(), shape=(self.tile_vec_m,)))
        self._fd_scale_local_first()
        mem_copy(fd_ub.produce(),
                 tile_slice(fd_o, (self.tile_vec_m, self.tile_d),
                            ((slot0 + 1) * (N1 // self.tile_vec_m) + sub, 0)))
        self._fd_acc_full(fd_ub.consume(), 1, False)
        self._cast_output()
        # Retain this output-alias dependency until its removal passes precision
        # and performance regression; the pre-reduction barrier precedes this cast.
        vec_sync_notify(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        vec_sync_wait(PIPE.V, PIPE.MTE3, _OUT_V_TO_MTE3_EVENT_ID)
        mem_copy(tile_slice(out_gm, (self.tile_vec_m, self.tile_d),
                           (row * (N1 // self.tile_vec_m) + sub, 0)),
                 self.out_view)

    @staticmethod
    def _half_only(actual_n, vector_length):
        return isinstance(actual_n, int) and actual_n <= vector_length

    def _nz_params(self, p_slot):
        # Derive P offsets from its physical stride, including Channel NZ padding.
        # This layout is independent of the dequantization NZ17 layout.
        s = p_slot.physical_stride
        s_n1, s_m1, s_m0, s_n0 = s[0], s[1], s[2], s[3]
        m0 = s_m1 // s_m0
        n0 = s_m0 // s_n0
        return m0, s_m1, s_m0, s_n1 // n0

    def _softmax_fold_row(self, p_slot, qk_ch, base, nz_off, max_brc_buf, row,
                          ve_mask, vo_mask, b16, b16_full, block_stride):
        # Second softmax pass: exponentiate and cast even/odd lanes to BF16.
        # ZERO/ONE register layouts pack the lanes; bitwise OR forms contiguous P.
        # Store P in NZ order and return both FP32 exponent vectors for row sums.
        # Separate even/odd masks handle odd tails; write all 128 BF16 columns
        # including zero padding so reused slots cannot retain stale P values.
        mx = rr.vload_broadcast(max_brc_buf, row)
        ve, vo = rr.vload_deinterleave(qk_ch, base, width="b32")
        ve = rr.vexp_sub(ve, mx, mask=ve_mask)
        vo = rr.vexp_sub(vo, mx, mask=vo_mask)
        he = rr.vcast(ve, dtypes.bfloat16, mask=ve_mask, reg_layout=rr.RegLayout.ZERO)
        ho = rr.vcast(vo, dtypes.bfloat16, mask=vo_mask, reg_layout=rr.RegLayout.ONE)
        merged = rr.vbitwise_or(he, ho, mask=b16)
        rr.vstore_strided(p_slot, nz_off, merged, b16_full, block_stride=block_stride, repeat_stride=0)
        return ve, vo

    def _pass_a_row(self, qk_ch, scale, sm_max_dst, row, row_stride, vector_length,
                    half0_mask, half1_mask, full_mask, half_only=False, full_tile=False):
        # First pass: scale valid scores in place and compute the row maximum.
        # Only valid score lanes may determine the maximum. Padding zeros would
        # change BF16 probability rounding for negative-score tail tiles.
        # Only a compile-time width <= 64 removes the second half.
        base = row * row_stride
        if const_expr(half_only):
            v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
            rr.vstore(qk_ch, base, v0, half0_mask)
            rmax = rr.vreduce_max(v0, mask=half0_mask)
        else:
            v0 = rr.vmuls(rr.vload(qk_ch, base), scale, mask=half0_mask)
            v1 = rr.vmuls(rr.vload(qk_ch, base + vector_length), scale, mask=half1_mask)
            rr.vstore(qk_ch, base, v0, half0_mask)
            rr.vstore(qk_ch, base + vector_length, v1, half1_mask)
            if const_expr(full_tile):
                rmax = rr.vreduce_max(rr.vmax(v0, v1, mask=full_mask), mask=full_mask)
            else:
                # The second half's valid mask is a subset of the first half's.
                # Reuse v0 outside that subset, then reduce only valid first-half lanes.
                v1_valid = rr.vselect(v1, v0, cond_mask=half1_mask)
                rmax = rr.vreduce_max(rr.vmax(v0, v1_valid, mask=full_mask), mask=half0_mask)
        rr.vstore_first(sm_max_dst, row, rmax)

    @jit
    def _softmax_tile_max(self, qk_ch, scale, actual_n, full_tile):
        # Keep VF temporaries local to this helper. Full tiles need no tail
        # selection; partial tiles must exclude invalid lanes from the maximum.
        vl = 2048 // 32
        rows = qk_ch.shape[0]
        row_stride = qk_ch.stride[0]
        half_only = self._half_only(actual_n, vl)
        for row in range(rows):
            full, _ = rr.update_mask(vl, elem_bits=32)
            if const_expr(full_tile):
                half0_mask, half1_mask = full, full
            else:
                half0_mask, _ = rr.update_mask(actual_n, elem_bits=32)
                if const_expr(half_only):
                    half1_mask = None
                else:
                    half1_mask, _ = rr.update_mask(max(0, actual_n - vl), elem_bits=32)
            self._pass_a_row(qk_ch, scale, self.tmp_new_max, row, row_stride, vl,
                             half0_mask, half1_mask, full,
                             half_only=half_only, full_tile=full_tile)

    def _softmax_rest_tail(self, sm_max, sm_sum, sm_exp, rowmask):
        # Merge denominators with alpha = exp(old_max - new_max).
        # Store alpha in this task slot so lag-3 output accumulation uses the same base.
        old_max = rr.vload(sm_max, 0)
        new_max = rr.vload(self.tmp_new_max, 0)
        se = rr.vexp_sub(old_max, new_max, mask=rowmask)
        rr.vstore(sm_exp, 0, se, rowmask)
        rr.vstore(sm_max, 0, new_max, rowmask)
        old_sum = rr.vload(sm_sum, 0)
        new_sum = rr.vload(self.tmp_sum, 0)
        ss = rr.vmadd(old_sum, se, new_sum, mask=rowmask)
        rr.vstore(sm_sum, 0, ss, rowmask)

    @jit
    def _finalize_lse_vf(self, lse_gm, lse_base, sum_idx: int):
        # LSE = log(sum) + max; the framework tracks Buffer V-to-MTE3 accesses.
        sm_max_buf = self.sm_max_tb[sum_idx]
        sm_sum_buf = self.sm_sum_tb[sum_idx]
        with vf(mode="simd"):
            mask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            total = rr.vload(sm_sum_buf, 0)
            maximum = rr.vload(sm_max_buf, 0)
            rr.vstore(self.lse_ub, 0, rr.vadd(rr.vlog(total, mask=mask), maximum, mask=mask), mask)
        mem_copy(tile_slice(lse_gm, (self.tile_vec_m,), (lse_base,)),
                 reinterpret(self.lse_ub, shape=(self.tile_vec_m,)))

    @jit
    def _cast_output(self):
        # Cast FP32 res_o in place to the first 32 KiB BF16 output view.
        # Each 64-element read precedes its packed write; that write ends before
        # the next unread FP32 chunk, preserving forward traversal safety.
        with vf(mode="simd"):
            full = rr.full_mask()
            for offset in range(0, self.tile_vec_m * self.tile_d, 64):
                value = rr.vload(self.res_o, offset)
                result = rr.vcast(value, dtypes.bfloat16, mask=full)
                rr.vstore_pack(self.out_view, offset, result, full, pack_mode=rr.PackMode.B32_TO_B16)

    @jit
    def _fd_ms_to_ub(self, which: int, sum_idx: int):
        # Copy max (which=0) or sum (which=1) to the LSE staging buffer.
        if const_expr(which == 0):
            buf = self.sm_max_tb[sum_idx]
        else:
            buf = self.sm_sum_tb[sum_idx]
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            rr.vstore(self.lse_ub, 0, rr.vload(buf, 0), rowmask)

    @jit
    def _fd_init_acc(self):
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            rr.vstore(self.sm_max_tb[0], 0,
                      rr.vdups(_FD_NEG_INF, dtypes.float32, mask=rowmask), rowmask)
            rr.vstore(self.sm_sum_tb[0], 0,
                      rr.vdups(0.0, dtypes.float32, mask=rowmask), rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_acc_max_save(self, p: int, lse_buf):
        # Cache m_p and accumulate the global maximum; the next pass reuses
        # the same weight slot for t_p = s_p * exp(m_p - M).
        mp_buf = tile_slice(self.fd_mp_tb, (self.tile_vec_m, 1), (p, 0))
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            mp = rr.vload(lse_buf, p * FD_MS_ELEMS + self.subblock_idx * self.tile_vec_m + 0)
            nm = rr.vmax(rr.vload(self.sm_max_tb[0], 0), mp, mask=rowmask)
            rr.vstore(self.sm_max_tb[0], 0, nm, rowmask)
            rr.vstore(mp_buf, 0, mp, rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_acc_sum(self, p: int, lse_buf):
        # Bulk max/sum landing: t_p = s_p·exp(m_p−M); G += t_p. Cache t_p.
        # Reuse cached m_p for t_p, then form t_p/G during output accumulation.
        mp_buf = tile_slice(self.fd_mp_tb, (self.tile_vec_m, 1), (p, 0))
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            e = rr.vexp_sub(rr.vload(mp_buf, 0),
                            rr.vload(self.sm_max_tb[0], 0), mask=rowmask)
            t = rr.vmul(rr.vload(lse_buf, p * FD_MS_ELEMS + self.subblock_idx * self.tile_vec_m + 64), e, mask=rowmask)
            acc = rr.vadd(rr.vload(self.sm_sum_tb[0], 0), t, mask=rowmask)
            rr.vstore(self.sm_sum_tb[0], 0, acc, rowmask)
            rr.vstore(mp_buf, 0, t, rowmask)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_lse_to_ub(self, lse_buf):
        # lse_ub ← M + log(G)
        with vf(mode="simd"):
            rowmask, _ = rr.update_mask(self.tile_vec_m, elem_bits=32)
            lse = rr.vadd(rr.vlog(rr.vload(self.sm_sum_tb[0], 0), mask=rowmask),
                          rr.vload(self.sm_max_tb[0], 0), mask=rowmask)
            rr.vstore(lse_buf, 0, lse, rowmask)

    @jit
    def _fd_acc_full(self, fd_ub, p: int, first: cannbotdsl.Constexpr[bool]):
        # Compute each head weight in registers while consuming partial O.
        # Part zero initializes the accumulator; later parts retain slot order.
        mp_buf = tile_slice(self.fd_mp_tb, (self.tile_vec_m, 1), (p, 0))
        with vf(mode="simd"):
            full = rr.full_mask()
            for r in range(self.tile_vec_m):
                wb = rr.vdiv(rr.vload_broadcast(mp_buf, r),
                             rr.vload_broadcast(self.sm_sum_tb[0], r), mask=full)
                for col in range(0, self.tile_d, 2048 // 32):
                    off = r * self.tile_d + col
                    cur = rr.vload(fd_ub, off)
                    if const_expr(first):
                        value = rr.vmul(cur, wb, mask=full)
                    else:
                        pre = rr.vload(self.res_o, off)
                        value = rr.vmadd(cur, wb, pre, mask=full)
                    rr.vstore(self.res_o, off, value, full)
            rr.vmem_bar("vst_vld")

    @jit
    def _fd_scale_local_first(self):
        # The first partial O is still in res_o: weight it in place.
        mp_buf = tile_slice(self.fd_mp_tb, (self.tile_vec_m, 1), (0, 0))
        with vf(mode="simd"):
            full = rr.full_mask()
            for r in range(self.tile_vec_m):
                wb = rr.vdiv(rr.vload_broadcast(mp_buf, r),
                             rr.vload_broadcast(self.sm_sum_tb[0], r), mask=full)
                for col in range(0, self.tile_d, 2048 // 32):
                    off = r * self.tile_d + col
                    cur = rr.vload(self.res_o, off)
                    rr.vstore(self.res_o, off, rr.vmul(cur, wb, mask=full), full)
            rr.vmem_bar("vst_vld")

    def _setup_fd_reduce_buffers(self):
        # After the global barrier, reuse main-loop UB for one full partial O,
        # an FP32 accumulator, all max/sum rows and up to 36 partial weights.
        # Main-loop allocations and reduction allocations are not simultaneous.
        self.fd_ms_ub = Channel(MemLoc.UB, shape=(_FD_MP_SLOTS, FD_MS_ELEMS),
                                dtype=dtypes.float32, depth=1)
        self.fd_ub = Channel(MemLoc.UB, shape=(self.tile_vec_m, self.tile_d),
                             dtype=dtypes.float32, depth=1)
        # Keep the main-loop res_o alive across rewind for local-first FD.
        # All other main-loop users are drained before this setup is called.
        self.out_view = reinterpret(self.res_o, dtypes.bfloat16,
                                    (self.tile_vec_m, self.tile_d))
        self.lse_ub = Channel(MemLoc.UB, shape=(self.tile_vec_m, 1),
                              dtype=dtypes.float32, depth=1)
        self.sm_max_tb[0] = Buffer(MemLoc.UB, (self.tile_vec_m, 1), dtypes.float32)
        self.sm_sum_tb[0] = Buffer(MemLoc.UB, (self.tile_vec_m, 1), dtypes.float32)
        self.tmp_new_max = Buffer(MemLoc.UB, (self.tile_vec_m, 1), dtypes.float32)
        self.fd_mp_tb = Buffer(MemLoc.UB, (_FD_MP_SLOTS * self.tile_vec_m, 1),
                               dtypes.float32)


@kernel(profile=ProfileSpec(name="mixed_quant_sparse_flash_mla"))
class MqsmlaKernel:
    """Device attention pipeline; tensor views and launch live in _run_mqsmla."""

    def __init__(self, tile_cube_m, tile_vec_m, tile_n,
                 has_cmp=False):
        self.tile_cube_m = tile_cube_m
        self.tile_vec_m = tile_vec_m
        self.tile_n = tile_n
        self.tile_d = D
        self.has_cmp = has_cmp
        self.p_l1 = Channel(MemLoc.L1, shape=(tile_cube_m, tile_n), dtype=dtypes.bfloat16,
                            depth=_P_L1_DEPTH, kind=ChannelKind.CrossCore)

        # Cross-core chains: QK (AIC to AIV, depth 2), P (AIV to AIC, depth 2),
        # and PV (AIC to AIV, depth 1). UB shapes cover one AIV; P covers both halves.

        self.block_idx = get_block_idx()
        self.subblock_idx = get_subblock_id()

        self.sinks_span = tile_vec_m

        self.matmul = Matmul(tile_cube_m, tile_vec_m, tile_n)
        self.kvcache = Kvcache()
        self.n_sub = tile_n // _TILE_ROWS

    def __call__(self, out_gm: Tensor, q_gm: Tensor,
                 sinks_gm: Tensor,
                 pa_phys_gm: Tensor = None, pa_bt_gm: Tensor = None, ws_gm: Tensor = None,
                 pa_phys_cmp_gm: Tensor = None, pa_bt_cmp_gm: Tensor = None,
                 cmp_idx_gm: Tensor = None,
                 ori_idx_gm: Tensor = None,
                 cu_seqlens_q: Tensor = None,
                 softmax_lse: Tensor = None,
                 ori_topk_length: Tensor = None, cmp_topk_length: Tensor = None,
                 metadata_gm: Tensor = None, softmax_scale: dtypes.float32 = 1.0,
                 batch_consistency: dtypes.int32 = 0,
                 addr_ws_gm: Tensor = None):
        # Workspace capacity determines the launch grid and FD slot count.
        self._b_rt = cu_seqlens_q.shape[1] - 1
        self._scale = softmax_scale
        self._batch_consistency = batch_consistency
        self._ori_page_stride = pa_phys_gm.stride[0]
        self._pa_bs_rt = pa_phys_gm.shape[1]
        if const_expr(self.has_cmp):
            self._cmp_page_stride = pa_phys_cmp_gm.stride[0]
            self._pa_cmp_bs_rt = pa_phys_cmp_gm.shape[1]
        self._ws = ws_gm
        self._q = q_gm
        self._sinks = sinks_gm
        self._out = out_gm
        self._lse = softmax_lse
        self._ori_topk_length = ori_topk_length
        self._cmp_topk_length = cmp_topk_length

        self._cu_q = cu_seqlens_q

        self._pa_phys, self._pa_bt = pa_phys_gm, pa_bt_gm
        self._ori_idx = ori_idx_gm
        if const_expr(self.has_cmp):
            self._pa_phys_cmp, self._pa_bt_cmp = pa_phys_cmp_gm, pa_bt_cmp_gm
            self._cmp_idx = cmp_idx_gm
        # Address-prepass storage exists only in the vector-addressing template.
        _avec = _addr_vec_mode()
        if const_expr(_avec != "off"):
            self._avec_bs_ori = pa_phys_gm.shape[1]
            self._avec_k1 = ori_idx_gm.shape[1]
            self._avec_k1_tab = _addr_tab_w(self._avec_k1)
            self._avec_k2_tab = 0
            if const_expr(self.has_cmp):
                self._avec_bs_cmp = pa_phys_cmp_gm.shape[1]
                self._avec_k2 = cmp_idx_gm.shape[1]
                self._avec_k2_tab = _addr_tab_w(self._avec_k2)
            if const_expr(_addr_vec_mode() == "arena"):
                self._addr_ub = Buffer(MemLoc.UB, shape=(1, _ADDR_UB_BYTES // 4),
                                       dtype=dtypes.int32)
            else:
                # Distinct owners allow the next index load to overlap the
                # preceding address store. No large-pool capacity is lost:
                # the host dispatch selects the arena kernel outside this range.
                self._ch_addr_bt = Buffer(MemLoc.UB, shape=(1, _ADDR_COMPACT_BT),
                                           dtype=dtypes.int32)
                self._ch_addr_sp = Buffer(MemLoc.UB, shape=(1, _ADDR_COMPACT_K),
                                           dtype=dtypes.int32)
                self._ch_addr_out = Buffer(MemLoc.UB, shape=(1, _ADDR_COMPACT_K),
                                            dtype=dtypes.int32)
            tab_cols = self._avec_k1_tab + (self._avec_k2_tab
                                            if self.has_cmp else 0)
            self._addr_tab = addr_ws_gm.view(dtypes.int32).view(
                addr_ws_gm.shape[0] // 4 // tab_cols, tab_cols)
            self._addr_bar_flag = 9
            # Both AIVs share an intra-sync ID; AIV1 uses the fixed ID offset on the AIC side.
            self._addr_bar_sid = 9

        # FA metadata follows the AscendC field layout.
        # A core owns rows [m_start,m_end], excluding m_end when s2_end is zero.
        # Boundary rows use the recorded S2 tile limits; zero denotes a full boundary.
        # Consecutive cores partition one contiguous (row,tile) interval.
        # Disabled cores have zero rows and only participate in required barriers.
        fa = self.block_idx * FA_METADATA_SIZE
        bn2_start = _i64(metadata_gm, fa + FA_BN2_START_INDEX)
        m_start_g = _i64(metadata_gm, fa + FA_M_START_INDEX)
        bn2_end = _i64(metadata_gm, fa + FA_BN2_END_INDEX)
        m_end_g = _i64(metadata_gm, fa + FA_M_END_INDEX)
        self._m_start = _i64(self._cu_q, 0, bn2_start) + m_start_g
        self._s2_start = _i64(metadata_gm, fa + FA_S2_START_INDEX)
        m_end = _i64(self._cu_q, 0, bn2_end) + m_end_g
        s2_end = _i64(metadata_gm, fa + FA_S2_END_INDEX)
        self._metadata_gm = metadata_gm
        self._first_fd = _i64(metadata_gm, fa + FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX)
        self._fd_async = _i64(metadata_gm, _FD_ASYNC_MODE_WORD)
        row_count = m_end - self._m_start + (1 if s2_end > 0 else 0)
        self._first_split_row = (dtypes.int64(0) if self._s2_start > 0 else dtypes.int64(-1))
        self._last_split_row = (row_count - 1 if s2_end > 0 else dtypes.int64(-1))
        # Zero s2 boundaries encode whole-row execution. Compute tile bounds
        # directly; the global FD gate is read only after the main loop.
        # DelayLineGroup carries coordinates, not tensors; depth 4 supports lag 3.
        # g is the pipeline clock; each stage selects slots with the carried task tick.
        # Keep row, tile, first/last and statistics-slot coordinates across queries.
        # Recover split state from metadata at each part boundary.
        lag_load = 1
        lag_b = 2
        lag_c = lag_b + 1
        depth = 4
        drain = lag_c
        dl = DelayLineGroup(depth, 'm', 'n', 't', 'last', 'first', 'sum_slot')
        g = 0
        active_row = dtypes.int64(0)

        if const_expr(_addr_vec_mode() != "off"):
            # Publish the address table before the main loop consumes it.
            self._addr_precompute(row_count)
            # AIC waits for both AIV writes before releasing the block.
            # Use MTE1 so the wait does not block Q prefetch on MTE2.
            cube_sync_intra_wait(PIPE.MTE1, self._addr_bar_sid)
            cube_sync_intra_wait(PIPE.MTE1,
                                 self._addr_bar_sid + _AIV1_ID_OFFSET)
            # Wait and release must use the same pipe to preserve ordering.
            cube_sync_block_arrive(PIPE.MTE1, self._addr_bar_flag, mode=2)

            # _addr_precompute's MTE3 arrive follows the last workspace store;
            # its S wait observes both AIV completions through the AIC relay.
            # All address Buffer reads/writes are therefore complete before any
            # main-loop UB use. Rewind reclaims storage, not synchronization.
            channel_rewind(reset_sync_id=False)

        self._setup_main_ub()

        # Warmup and steady state share a continuous per-core clock.
        # Only stages with g >= lag have a task; Channel dependencies govern hardware
        # overlap, so Python call order does not serialize all engines.
        # First/last flags initialize and finalize each part; split parts stage FD data.
        # Cores with no rows do not enqueue tasks.
        kv_qk = self.matmul.kv_ring.consume()
        kv_pv = kv_qk
        q_half0 = self.matmul.q_ring.consume()
        q_half1 = self.matmul.q_ring.consume()
        for m_seq in range(0, row_count):
            m_idx = self._m_idx(m_seq)
            tiles = self._n_end_rt(m_idx)
            t0 = (self._s2_start if m_seq == 0 else dtypes.int64(0))
            t1_tail = (s2_end if s2_end > 0 else tiles)
            t1 = (t1_tail if m_seq == row_count - 1 else tiles)
            for n in range(t0, t1):
                is_last = 1 if n == t1 - 1 else 0
                is_first = 1 if n == t0 else 0
                dl.push(m=m_seq, n=n, t=g, last=is_last, first=is_first,
                        sum_slot=active_row % 2)

                next_pv = kv_pv
                self._stage_vec0(m_seq, n)
                if g >= lag_load:
                    s0 = dl.tap(lag_load)
                    self._stage_loadqk(s0.t, s0.m)
                if g >= lag_b:
                    s1 = dl.tap(lag_b)
                    self._stage_qk_softmax(kv_qk, q_half0, q_half1, s1.t, s1.m, s1.n, s1.last, s1.first,
                                           row_count, s1.sum_slot)
                    next_pv = kv_qk
                    kv_qk = self.matmul.kv_ring.consume()
                    if s1.last == dtypes.int64(1):
                        q_half0 = self.matmul.q_ring.consume()
                        q_half1 = self.matmul.q_ring.consume()
                    self._prefetch_q(s1.m, s1.last, row_count)

                if g >= lag_c:
                    s2 = dl.tap(lag_c)
                    self._stage_pv_update(kv_pv, s2.t, s2.m, s2.n, s2.last, s2.first,
                                          s2.sum_slot)

                kv_pv = next_pv
                dl.advance()
                g = g + 1
            active_row = active_row + (dtypes.int64(1) if t1 > t0 else dtypes.int64(0))
        # Drain three steps without pushing new tasks. Keep a dynamic loop so DSL
        # 0.7 tracks the retained KV version from QK through its final PV read.
        # Stage guards exclude empty taps and exhausted stages, including one-task cores.
        for r in range(drain):
            next_pv = kv_pv
            if r < lag_load:
                if g >= lag_load:
                    s0 = dl.tap(lag_load)
                    self._stage_loadqk(s0.t, s0.m)
            if r < lag_b:
                if g >= lag_b:
                    s1 = dl.tap(lag_b)
                    self._stage_qk_softmax(kv_qk, q_half0, q_half1, s1.t, s1.m, s1.n, s1.last, s1.first,
                                           row_count, s1.sum_slot)
                    next_pv = kv_qk
                    kv_qk = self.matmul.kv_ring.consume()
                    if s1.last == dtypes.int64(1):
                        q_half0 = self.matmul.q_ring.consume()
                        q_half1 = self.matmul.q_ring.consume()
                    self._prefetch_q(s1.m, s1.last, row_count)
            if r < lag_c:
                if g >= lag_c:
                    s2 = dl.tap(lag_c)
                    self._stage_pv_update(kv_pv, s2.t, s2.m, s2.n, s2.last, s2.first,
                                          s2.sum_slot)
            kv_pv = next_pv
            dl.advance()
            g = g + 1

        # Empty queries never enter the delay line. Write their outputs after
        # draining so the shared output buffer cannot disturb pending PVs.
        for m_seq in range(0, row_count):
            m_idx = self._m_idx(m_seq)
            if self._n_end_rt(m_idx) == 0:
                o_tile = tile_slice(self._out, (self.tile_cube_m, self.tile_d),
                                   (m_idx, 0))
                lse_base = (m_idx * 2 + self.subblock_idx
                            if const_expr(self._lse is not None) else None)
                self.vector.finalize_empty(o_tile, self._sinks, self._lse, lse_base)

        # The uniform fd_any field gates synchronization and reduction on every core.
        # Without split rows, skip both. Enabled AIVs reduce their assigned partials.
        fd_any = _i64(metadata_gm, FD_USED_VEC_NUM_WORD)
        if fd_any != 0:
            # Async FD drains local UB users here; per-slot ready polling below
            # provides remote publication ordering. Synchronous FD instead needs
            # the all-core barrier before reading any remote partials. Neither
            # Channel rewind nor new local slots establish these cross-core dependencies.
            if self._fd_async == 1:
                vec_sync_all()
            else:
                global_sync_all()
            # After the barrier, reclaim main-loop UB for the full (32,512) FP32 reduction landing.
            channel_rewind(reset_sync_id=False)
            self.vector._setup_fd_reduce_buffers()
            aiv_idx = self.block_idx * get_subblock_dim() + self.subblock_idx
            fd_metadata_base = FD_METADATA_BASE + aiv_idx * FD_METADATA_SIZE
            fd_enable = _i64(metadata_gm, fd_metadata_base + FD_CORE_ENABLE_INDEX)
            if fd_enable != 0:
                fd_bn2 = _i64(metadata_gm, fd_metadata_base + FD_BN2_IDX_INDEX)
                fd_m = _i64(metadata_gm, fd_metadata_base + FD_M_IDX_INDEX)
                fd_row = _i64(self._cu_q, 0, fd_bn2) + fd_m
                fd_ws = _i64(metadata_gm, fd_metadata_base + FD_WORKSPACE_IDX_INDEX)
                fd_k = _i64(metadata_gm, fd_metadata_base + FD_WORKSPACE_NUM_INDEX)
                fd_h0 = _i64(metadata_gm, fd_metadata_base + FD_M_START_INDEX)
                fd_o, fd_ms = self._fd_workspace_views()
                if self._fd_async == 1:
                    ready_word = _FD_READY_BASE + (fd_ws // 2) * 2 + self.subblock_idx
                    vec_busy_wait(metadata_gm[ready_word], Int32(1))
                    self.vector.fd_reduce_local_first(
                        fd_o, fd_ms, self.vector.fd_ub, self._out, self._lse,
                        fd_row, fd_ws, fd_h0)
                else:
                    self.vector.fd_reduce(fd_o, fd_ms, self.vector.fd_ub,
                                          self._out, self._lse,
                                          fd_row, fd_ws, fd_k, fd_h0)

    @staticmethod
    def _scale_idx(grp_lane, base: int, m16):
        # Scale gather uses uint16 indices, so its arithmetic requires a 16-bit mask.
        return rr.vadd(grp_lane, rr.vdups(base, dtypes.uint16), mask=m16)

    @jit
    def _setup_main_ub(self):
        # Allocate only after address-prepass storage is dead. L1/L0 and their
        # cross-core protocols remain live across rewind; do not reset IDs.
        tile_vec_m, tile_n = self.tile_vec_m, self.tile_n
        self.qk_ub = Channel(MemLoc.UB, shape=(tile_vec_m, tile_n), dtype=dtypes.float32,
                             depth=2, kind=ChannelKind.CrossCore)
        # FIXPIPE-to-V communication requires CrossCore and a full-PV transaction.
        self.pv_ub = Channel(MemLoc.UB, shape=(tile_vec_m, self.tile_d), dtype=dtypes.float32,
                             depth=1, kind=ChannelKind.CrossCore)
        self.vector = Vector(tile_vec_m, tile_n, self.tile_d, self.subblock_idx,
                             sinks_span=self.sinks_span)

        # The existing scalar PA path accepts runtime page sizes and does not
        # allocate an on-chip buffer proportional to the block-table capacity.
        ch_rows = _TILE_ROWS
        # FP8/FP4 share four input slots with automatic MTE2-to-V reuse ordering.
        self.ch_batch_bytes = Channel(MemLoc.UB,
                                      shape=(ch_rows, _COMBINE_DIM),
                                      dtype=dtypes.uint8, depth=4, kind=ChannelKind.SameCore)
        # DMA and VF share one byte Channel; interpret scale bits in registers.
        # Do not create a second typed Channel for the same slot.
        self._v0_out = Channel(MemLoc.UB, shape=(D // 16, _NZ_PAD_ROWS, 16),
                               dtype=dtypes.bfloat16, depth=2)

    # m_seq is the core-local query index; m_idx is the global query index.
    # n spans ORI then CMP tiles; tick increases continuously across queries.
    # Each pool uses its valid prefix for both gather and softmax masking.
    # CMP tile coordinates subtract the ORI tile count.
    # S2 start values are sparse-index column offsets, not token values.

    def _m_idx(self, m_seq: int):
        # Map the local query index with the preloaded metadata start.
        return self._m_start + m_seq

    @jit
    def _batch_of(self, m_idx: int):
        # Dynamic upper-bound search over the B query spans.
        lo, hi = 0, self._b_rt
        while lo < hi:
            mid = (lo + hi) // 2
            take = _i64(self._cu_q, 0, mid + 1) <= m_idx
            lo = (mid + 1 if take else lo)
            hi = (hi if take else mid)
        return lo

    @jit
    def _bound_after(self, batch: int, b_last: int):
        # Read the next batch boundary once. At the final span, return a sentinel
        # instead of reading beyond cu_q; this matches _batch_of clamping.
        nxt = (batch + 1 if batch < b_last else b_last)
        v = _i64(self._cu_q, 0, nxt)
        return (v if batch < b_last else dtypes.int64(_ADDR_M_INF))

    def _cmp_tiles_rt(self, m_idx: int):
        w = self._csa_valid_w(m_idx)
        return (w + self.tile_n - 1) // self.tile_n

    @jit
    def _ori_valid_w(self, m_idx: int):
        # ORI width is min(K1, topk_length), or K1 when lengths are absent.
        # Promote it to runtime Int64 even without lengths so FD tile-bound
        # predicates remain SSA values rather than Python booleans.
        if const_expr(self._ori_topk_length is not None):
            return min(self._ori_idx.shape[1], _i64(self._ori_topk_length, m_idx))
        return Int64(self._ori_idx.shape[1])

    @jit
    def _cmp_topk_len(self, m_idx: int):
        if const_expr(self._cmp_topk_length is not None):
            return min(self._cmp_idx.shape[1], _i64(self._cmp_topk_length, m_idx))
        return Int64(self._cmp_idx.shape[1])

    def _ori_tiles_rt(self, m_idx: int):
        w = self._ori_valid_w(m_idx)
        return (w + self.tile_n - 1) // self.tile_n

    def _n_end_rt(self, m_idx: int):
        # Runtime query tile count: ORI tiles plus CMP tiles.
        ori_tiles = self._ori_tiles_rt(m_idx)
        if const_expr(not self.has_cmp):
            return ori_tiles
        return ori_tiles + self._cmp_tiles_rt(m_idx)

    @jit
    def _actual_n(self, m_idx: int, n):
        # Return the selected pool tile width for exponent and P masks.
        # The unselected conditional width does not describe an actual task.
        ori_w = self._ori_width(m_idx, n)
        if const_expr(not self.has_cmp):
            return ori_w
        ori_tiles = self._ori_tiles_rt(m_idx)
        cmp_tile_index = n - ori_tiles
        cmp_w = self._cmp_valid(m_idx, cmp_tile_index)
        return (cmp_w if n >= ori_tiles else ori_w)

    @jit
    def _ori_width(self, m_idx: int, n):
        w = self._ori_valid_w(m_idx)
        return min(self.tile_n, w - n * self.tile_n)

    def _csa_valid_w(self, m_idx: int):
        return self._cmp_topk_len(m_idx)

    @jit
    def _cmp_valid(self, m_idx: int, c):
        return min(self.tile_n, self._csa_valid_w(m_idx) - c * self.tile_n)

    @jit
    def _antiquant_vf_fp8_g32(self, kv_fp8, out):
        # ORI byte order is nope[448], rope[64], then 16 BF16 scales.
        # Dequantize four 128-value chunks through FP32 to BF16 and compact even lanes.
        # Each chunk uses four group-32 scales and writes matching NZ columns.
        subtile_rows = _TILE_ROWS
        with vf(mode="simd"):
            m32, _ = rr.update_mask(_TILE_SIZE, elem_bits=32)
            m16, _ = rr.update_mask(_VL, elem_bits=16)
            m8, _ = rr.update_mask(256, elem_bits=8)
            grp_lane = rr.vshr(rr.varange(0, dtypes.uint16), 5, mask=m16)
            row_bf16, row_fp8 = _ROW_BF16, _COMBINE_DIM

            # Iterate over four D chunks and the token rows in this subtile.
            for j in tuple(range(D // _KV_NUM_PER_LOOP)):
                src_j = j * _KV_NUM_PER_LOOP
                dst_j = j * _KV_NUM_PER_LOOP
                idx_j = self._scale_idx(grp_lane, 4 * j, m16)
                for i in range(subtile_rows):
                    s = i * row_fp8 + src_j
                    u0 = rr.vload_unpack(kv_fp8, s, unpack_mode="b8_to_b32")
                    u1 = rr.vload_unpack(kv_fp8, s + _TILE_SIZE,
                                         unpack_mode="b8_to_b32")
                    e0 = rr.vreinterpret_lanes(u0, dtypes.float8_e4m3fn)
                    e1 = rr.vreinterpret_lanes(u1, dtypes.float8_e4m3fn)
                    f0 = rr.vcast(e0, dtypes.float32, mask=m8, reg_layout=rr.RegLayout.ZERO)
                    f1 = rr.vcast(e1, dtypes.float32, mask=m8, reg_layout=rr.RegLayout.ZERO)
                    b0 = rr.vcast(f0, dtypes.bfloat16, mask=m32)
                    b1 = rr.vcast(f1, dtypes.bfloat16, mask=m32)
                    raw, _ = rr.vdeinterleave(b0, b1)
                    srow = rr.vreinterpret_lanes(
                        rr.vload(kv_fp8, 2 * (_SCALE_BF16_OFF + i * row_bf16)), dtypes.bfloat16)
                    svec = rr.vgather_reg(srow, idx_j)
                    packed = rr.vmul(raw, svec, mask=m16)
                    self._v0_store(out, i, dst_j, packed, m16)

    @jit
    def _antiquant_vf_fp4_g16(self, kv_fp8, out):
        # CMP layout: 224 nope bytes, 32 rope bytes, and 64 scale bytes.
        # UNPACK4 + packed FP4 reinterpret + quarter-0 cast.
        # Packed FP4 consumes an 8-bit mask; each byte expands to two BF16 values.
        # Scales follow groups of 16 features, and _v0_store writes NZ17 faces.
        subtile_rows = _TILE_ROWS
        with vf(mode="simd"):
            m16, _ = rr.update_mask(_VL, elem_bits=16)
            row_fp8 = _COMBINE_DIM
            scale_off = SCALE_BF16_OFF_CMP

            # Gather each FP4 group scale from the byte buffer and interpret it as BF16.
            grp_lane = rr.vshr(rr.varange(0, dtypes.uint16), 4, mask=m16)
            for p in tuple(range(D // _KV_NUM_PER_LOOP)):
                src_p = p * _TILE_SIZE
                dst_p = p * _KV_NUM_PER_LOOP
                idx_p = self._scale_idx(grp_lane, 8 * p, m16)
                for i in range(subtile_rows):
                    u = rr.vload_unpack(kv_fp8, src_p + i * row_fp8,
                                        unpack_mode=rr.UnpackMode.UNPACK4)
                    f4 = rr.vreinterpret_lanes(u, dtypes.fp4x2_e2m1)
                    fm = rr.full_mask()
                    fm.elem_bits = 8
                    vals = rr.vcast(f4, dtypes.bfloat16, mask=fm,
                                    reg_layout=rr.RegLayout.ZERO)
                    srow = rr.vreinterpret_lanes(
                        rr.vload(kv_fp8, 2 * (scale_off + i * _ROW_BF16)),
                        dtypes.bfloat16)
                    packed = rr.vmul(vals, rr.vgather_reg(srow, idx_p),
                                     mask=m16)
                    self._v0_store(out, i, dst_p, packed, m16)

    def _v0_store(self, out, i, col, val, mask):
        # i is the token row and col the D column; offsets use BF16 elements.
        # NZ17 address = (col//16)*(17*16) + i*16 + col%16.
        # Each 16-column store advances by 17 blocks of 32 bytes; row 17 is padding.
        rr.vstore_strided(out, (col // 16) * _NZ_CHUNK + i * _NZ_ROW,
                          val, mask, block_stride=_NZ_PAD_ROWS,
                          repeat_stride=0)

    @jit
    def _vec0_body(self, kv_dsts, kv_pool, blk_table, blk_size, s2_start: int,
                   n_real, idx_tab=None, idx_row=None, batch=None,
                   is_cmp=False):
        # Each AIV handles four 16-token subblocks of a 128-token tile.
        # Pair DMA sorts KV and inline scales together; otherwise copy rows individually.
        # Tail rows repeat the last valid index. Uninitialized subblocks are excluded from PV.
        # Both pools use 544-byte UB rows; cmp reads only its 320-byte payload.
        subtile_rows = _TILE_ROWS
        if const_expr(is_cmp):
            row_bytes = KV_ROW_BYTES_CMP
            page_stride = self._cmp_page_stride
        else:
            row_bytes = KV_ROW_BYTES_ORI
            page_stride = self._ori_page_stride
        n_rt = n_real
        # Cover physical storage including page gaps; do not flatten logical tokens.
        # mem_copy preserves the Channel producer identity.
        storage_bytes = ((kv_pool.shape[0] - 1) * page_stride + blk_size * row_bytes)
        kv_bytes = kv_pool.view(storage_bytes)

        part = self.n_sub // 2
        t0 = self.subblock_idx * part
        _mode = _addr_vec_mode()
        _consumer_vec = _mode != "off"
        tab = None
        col_base = 0
        if const_expr(_mode != "off"):
            # The prepass fills GM row indices; col_base selects the pool columns.
            tab = self._addr_tab
            col_base = self._avec_k1_tab if const_expr(is_cmp) else 0
        # Every enqueued tile has n_real > 0. Write all subtiles, including
        # padding, so both AIVs publish one complete KV slot. QK masks padding;
        # PV uses zero probabilities there. Never publish an empty alias of KV.
        for lt in range_constexpr(part):
            sub_b = t0 + lt
            sub_row = sub_b * subtile_rows
            v0_out = self._v0_out.produce()
            buf = self.ch_batch_bytes.produce()
            bo_sp = 0 if batch is None else batch
            # Clamp padded rows to the final token in the valid prefix before addressing.
            last_off = n_rt - 1 - sub_row
            addresses = []
            pair_first = []
            pair_gap = []
            all_pairs_ok = Int32(1)
            if const_expr(_consumer_vec):
                # Dense page strides make row_number * row_bytes the physical offset.
                # The prepass pads to 16 columns and clamps each address. Reuse
                # its last subtile for padding beyond the valid tile prefix.
                index_base = col_base + s2_start + min(sub_row, ((n_rt - 1) // subtile_rows) * subtile_rows)
                for pair in range_constexpr(subtile_rows // 2):
                    pair_row_offset = pair * 2
                    first_index_col = index_base + pair_row_offset
                    second_index_col = first_index_col + 1
                    first_row_index = _i64(tab, idx_row, first_index_col)
                    second_row_index = _i64(tab, idx_row, second_index_col)
                    addr0 = first_row_index * row_bytes
                    addr1 = second_row_index * row_bytes
                    first = min(addr0, addr1)
                    gap = max(addr0, addr1) - first
                    addresses.append(addr0)
                    addresses.append(addr1)
                    pair_first.append(first)
                    pair_gap.append(gap)
                    all_pairs_ok = (
                        (all_pairs_ok
                         if gap - row_bytes <= _DMA_SOURCE_STRIDE_MAX_BYTES
                         else Int32(0))
                        if gap >= row_bytes else Int32(0))
            else:
                for pair in range_constexpr(subtile_rows // 2):
                    pair_row_offset = pair * 2
                    pos0 = s2_start + sub_row + min(pair_row_offset, last_off)
                    pos1 = s2_start + sub_row + min(pair_row_offset + 1, last_off)
                    tok0 = _i64(idx_tab, idx_row, pos0)
                    tok1 = _i64(idx_tab, idx_row, pos1)
                    blk0, off0 = self.kvcache.pa_blk_off(
                        blk_table, tok0, bo_sp, blk_size)
                    blk1, off1 = self.kvcache.pa_blk_off(
                        blk_table, tok1, bo_sp, blk_size)
                    addr0 = blk0 * page_stride + off0 * row_bytes
                    addr1 = blk1 * page_stride + off1 * row_bytes
                    first = min(addr0, addr1)
                    gap = max(addr0, addr1) - first
                    addresses.append(addr0)
                    addresses.append(addr1)
                    pair_first.append(first)
                    pair_gap.append(gap)
                    all_pairs_ok = (
                        (all_pairs_ok
                         if gap - row_bytes <= _DMA_SOURCE_STRIDE_MAX_BYTES
                         else Int32(0))
                        if gap >= row_bytes else Int32(0))
            # Deterministic level 3 requires sparse-index order, so disable pair sorting.
            all_pairs_ok = (Int32(0) if self._batch_consistency != 0 else all_pairs_ok)
            # Fill all 16 rows before publishing the slot. Reorder KV with inline scales;
            # QK/PV share that order. Duplicate tail addresses use per-row copies.
            if all_pairs_ok != 0:
                for pair in range_constexpr(subtile_rows // 2):
                    first = pair_first[pair]
                    second = first + pair_gap[pair]
                    dst = tile_slice(buf, (2, row_bytes), (pair, 0))
                    mem_copy(dst, (kv_bytes[first:first + row_bytes, ],
                                   kv_bytes[second:second + row_bytes, ]))
            else:
                for pair_row_offset in range_constexpr(subtile_rows):
                    addr = addresses[pair_row_offset]
                    row = kv_bytes[addr:addr + row_bytes, ].view(1, row_bytes)
                    mem_copy(tile_slice(buf, (1, row_bytes), (pair_row_offset, 0)), row)
            # Consume the byte slot once; scale reinterpretation adds no Channel.
            rslot_bytes = self.ch_batch_bytes.consume()

            if const_expr(is_cmp):
                self._antiquant_vf_fp4_g16(rslot_bytes, v0_out)
            else:
                self._antiquant_vf_fp8_g32(rslot_bytes, v0_out)
            # NZ17 faces become one regular strided UB-to-L1 copy.
            src = reinterpret(v0_out, shape=(D // 16, subtile_rows * 16),
                              stride=(_NZ_CHUNK, 1))
            dst = kv_dsts[lt]
            mem_copy(dst, src)

    @jit
    def _addr_vector_chunk(self, chunk, ktab, w, bs, inv_bs):
        lanes = min(_ADDR_HALF, ktab - chunk * _ADDR_HALF)
        mk, _ = rr.update_mask(lanes, elem_bits=32)
        lane = rr.varange(0, dtypes.int32)
        col = rr.vadds(lane, chunk * _ADDR_HALF, mask=mk)
        col = rr.vmins(col, w - Int64(1), mask=mk)
        idx = rr.vgather(self._ch_addr_sp,
                         rr.vreinterpret(col, dtypes.uint32), mask=mk)
        if const_expr(isinstance(bs, int)):
            blk = rr.vshr(idx, bs.bit_length() - 1, mask=mk)
        else:
            # Drop four irrelevant offset bits before FP32 conversion. With
            # BT bounded by this UB arena and BS<=1024, idx>>4 is below 2**22,
            # hence exact even when the original logical index exceeds 2**24.
            if const_expr(_addr_vec_mode() == "arena"):
                idx_f = rr.vcast(rr.vshr(idx, 4, mask=mk), dtypes.float32, mask=mk)
            else:
                idx_f = rr.vcast(idx, dtypes.float32, mask=mk)
            blk = rr.vcast(rr.vmuls(idx_f, inv_bs, mask=mk),
                           dtypes.int32, mask=mk, rounding="rd")
        off = rr.vsub(idx, rr.vmuls(blk, bs, mask=mk), mask=mk)
        p = rr.vgather(self._ch_addr_bt,
                       rr.vreinterpret(blk, dtypes.uint32), mask=mk)
        row = rr.vadd(rr.vmuls(p, bs, mask=mk), off, mask=mk)
        rr.vstore(self._ch_addr_out, chunk * _ADDR_HALF, row, mk)

    @jit
    def _addr_precompute(self, m_count):
        """Precompute clamped physical row numbers for ori/cmp, then publish them within the block."""
        tab_cols = self._avec_k1_tab + (self._avec_k2_tab if self.has_cmp else 0)
        # Query rows are consecutive: locate the first batch once, then advance at cu_q boundaries.
        b_last = dtypes.int64(self._b_rt)
        batch0 = self._batch_of(self._m_start)
        bound0 = self._bound_after(batch0, b_last)
        for side in range_constexpr(2 if self.has_cmp else 1):
            if const_expr(side == 0):
                idx_tab, bt_gm = self._ori_idx, self._pa_bt
                bs, kwidth, col0 = self._avec_bs_ori, self._avec_k1, 0
                ktab = self._avec_k1_tab

                def w_fn(mi):
                    return self._ori_valid_w(mi)
            else:
                idx_tab, bt_gm = self._cmp_idx, self._pa_bt_cmp
                bs, kwidth, col0 = (self._avec_bs_cmp, self._avec_k2,
                                    self._avec_k1_tab)
                ktab = self._avec_k2_tab

                def w_fn(mi):
                    return self._cmp_topk_len(mi)
            btp = bt_gm.shape[1]
            if const_expr(_addr_vec_mode() == "arena"):
                # Disjoint runtime windows share an owner; sides reuse the arena.
                bt_capacity = _addr_tab_w(btp)
                self._ch_addr_bt = reinterpret(self._addr_ub, shape=(1, bt_capacity))
                self._ch_addr_sp = reinterpret(self._addr_ub, shape=(1, ktab),
                                                offset=bt_capacity * 4)
                self._ch_addr_out = reinterpret(self._addr_ub, shape=(1, ktab),
                                                 offset=(bt_capacity + ktab) * 4)
            numerator = 16.0 if _addr_vec_mode() == "arena" else 1.0
            if const_expr(isinstance(bs, int)):
                inv_bs = numerator / bs
            else:
                inv_bs = dtypes.float32(numerator) / dtypes.float32(bs)
            # Round up to cover every 16-column consumer subblock; mask the final vector chunk.
            chunks = (ktab + _ADDR_HALF - 1) // _ADDR_HALF
            total_valid = m_count
            per = total_valid // 2
            tail = total_valid % 2
            core = dtypes.int64(self.subblock_idx)
            # For an odd row count, give the extra ori row to AIV0 and the extra cmp row to AIV1.
            if const_expr(side == 0):
                start = per * core + min(core, tail)
                count = per + (Int64(1) if core < tail else Int64(0))
            else:
                start = per * core
                count = per + (Int64(0) if core < Int64(1) else tail)
            counter = Int64(0)
            bt_staged = Int64(-1)
            # Initialize loop-carried batch state outside the loop; the inner while may execute zero times.
            batch = batch0
            bound = bound0
            for m_seq in range(0, m_count):
                m_idx = self._m_idx(m_seq)
                while m_idx >= bound:
                    batch = batch + 1
                    bound = self._bound_after(batch, b_last)
                w = w_fn(m_idx)
                active = (Int64(1) if w > Int64(0) else Int64(0)) * \
                    (Int64(1) if counter >= start else Int64(0)) * \
                    (Int64(1) if counter < start + count else Int64(0))
                if active == Int64(1):
                    _copy_addr_from_gm(self._ch_addr_sp, idx_tab,
                               m_idx * kwidth, kwidth * 4)
                    key = batch * 2 + side
                    if key != bt_staged:
                        _copy_addr_from_gm(self._ch_addr_bt, bt_gm,
                                   batch * btp, btp * 4)
                    with vf(mode="simd"):
                        for chunk in range(chunks):
                            self._addr_vector_chunk(chunk, ktab, w, bs, inv_bs)
                    dst = m_idx * tab_cols + col0
                    # Write the padded ktab columns, clamped by w-1. Multiples of 16 columns
                    # keep each UB-to-GM transfer aligned to 32 bytes without padding support.
                    _copy_addr_to_gm(self._addr_tab, self._ch_addr_out, dst, ktab)
                counter = counter + Int64(1)
                # An active entry already guarantees a nonempty prefix.
                bt_staged = (batch * 2 + side if active == Int64(1) else bt_staged)
        # Publish both AIVs' address writes through the AIC relay before either AIV reads the table.
        vec_sync_intra_arrive(PIPE.MTE3, self._addr_bar_sid)
        # Both AIV GM writes must be visible before either reads the shared table.
        # Local Channel locks cannot replace this AIC-relayed publication.
        vec_sync_block_wait(PIPE.S, self._addr_bar_flag, mode=2)

    @jit
    def _stage_vec0(self, m_seq: int, n):
        # Select the ORI/CMP pool and valid width for the shared L1 task ring.
        # CMP uses n - ori_tiles and the address-table column region after ORI.
        # Both AIVs fill their complete half before the L1 slot is consumed.
        # The same publication stays live through QK and the delayed PV read.
        m_idx = self._m_idx(m_seq)
        # Only scalar addressing needs a per-tile batch lookup.
        batch = None
        if const_expr(_addr_vec_mode() == "off"):
            batch = self._batch_of(m_idx)

        # Select slots and views outside branches to expose the full publish boundary.
        kv_slot = self.matmul.kv_ring.produce()
        l1_nd = reinterpret(kv_slot, shape=(D // 16, self.tile_n * 16), data_format="nd")
        kv_dsts = [tile_slice(l1_nd, (D // 16, _TILE_ROWS * 16),
                              (0, self.subblock_idx * (self.n_sub // 2) + i))
                   for i in range(self.n_sub // 2)]

        if const_expr(not self.has_cmp):
            actual_n = self._ori_width(m_idx, n)
            self._vec0_body(kv_dsts, self._pa_phys, self._pa_bt, self._pa_bs_rt,
                            n * self.tile_n, actual_n,
                            idx_tab=self._ori_idx, idx_row=m_idx, batch=batch,
                            is_cmp=False)
        else:
            ori_tiles = self._ori_tiles_rt(m_idx)
            ori_bound = ori_tiles
            if n >= ori_bound:
                cmp_tile_index = n - ori_bound
                self._vec0_body(kv_dsts, self._pa_phys_cmp, self._pa_bt_cmp, self._pa_cmp_bs_rt,
                                cmp_tile_index * self.tile_n,
                                self._cmp_valid(m_idx, cmp_tile_index),
                                idx_tab=self._cmp_idx, idx_row=m_idx, batch=batch,
                                is_cmp=True)
            else:
                self._vec0_body(kv_dsts, self._pa_phys, self._pa_bt, self._pa_bs_rt,
                                n * self.tile_n, self._ori_width(m_idx, n),
                                idx_tab=self._ori_idx, idx_row=m_idx, batch=batch,
                                is_cmp=False)

    @jit
    def _prefetch_q(self, m_seq: int, is_last: int, row_count):
        # Skip empty queries: they never consume Q Channel slots.
        if is_last == dtypes.int64(1):
            # Keep both scan states in the loop condition for DSL while lowering.
            j = m_seq + dtypes.int64(1)
            stop = row_count
            while j < stop:
                found = self._n_end_rt(self._m_idx(j)) != 0
                stop = (j if found else stop)
                j = (j if found else j + dtypes.int64(1))
            if j < row_count:
                q_queries = self._q.view(self._q.shape[0] // self.tile_cube_m,
                                         self.tile_cube_m, self.tile_d)
                next_q = q_queries[self._m_idx(j), None, None]
                self.matmul.load_q(next_q, j)

    @jit
    def _stage_softmax(self, tick: int, m_seq: int, n, is_first: int,
                       sum_slot: int):
        # At lag 2, consume QK through its Channel and compute online softmax.
        # Seed first parts from sinks and FD continuations from an empty state,
        # then send P to L1.
        # max/sum use the nonempty-row slot; exp uses tick%2.
        m_idx = self._m_idx(m_seq)
        actual_n = self._actual_n(m_idx, n)
        actual_vec_m = self.tile_vec_m

        m_axis_triple = sum_slot
        tile_triple = tick % 2

        qk_slot = self.qk_ub.consume()
        qk_view = reinterpret(qk_slot, shape=(actual_vec_m, self.tile_n), stride=(self.tile_n, 1))
        # A leading continuation must not include sinks a second time.
        cont_part = ((Int32(1) if self._s2_start > 0 else Int32(0))
                     if m_seq == 0 else Int32(0))
        if is_first == dtypes.int64(1):
            if cont_part == Int32(1):
                self.vector.seed_empty(m_axis_triple)
            else:
                sinks_coord = self.subblock_idx
                sk = self.vector.sinks_ub
                mem_copy(sk, tile_slice(self._sinks, (1, self.sinks_span),
                                       (0, sinks_coord)))
                skr = self.vector.sinks_ub
                self.vector.seed_from_sinks(skr, m_axis_triple)
        self.vector.softmax_rest(qk_view, self._scale, m_axis_triple, tile_triple, actual_n)

        self.vector.store_p(self.p_l1)

    @jit
    def _stage_pv_fanout(self, kv_slot, m_seq: int, n):
        # At lag 3, compute P @ KV using the same L1 slot, then FIXPIPE to PV UB.
        self.matmul.compute_pv_fanout(self.p_l1, self.pv_ub,
                                      kv_slot, self._actual_n(self._m_idx(m_seq), n))
        # This is the final read of the KV slot; Channel ownership then permits reuse.

    @jit
    def _stage_loadqk(self, tick: int, m_seq: int):
        q_queries = self._q.view(self._q.shape[0] // self.tile_cube_m,
                                 self.tile_cube_m, self.tile_d)
        q_tile_gm = q_queries[self._m_idx(m_seq), None, None]
        self.matmul.load_qk(q_tile_gm, m_seq, tick)

    @jit
    def _stage_qk_softmax(self, kv_slot, q_half0, q_half1, tick: int, m_seq: int, n, is_last: int,
                          is_first: int, row_count, sum_slot: int):
        # The Channel waits for both AIV MTE3 publications before the AIC reads L1.

        self.matmul.bmm1_fanout_q(kv_slot, q_half0, q_half1)
        self.matmul.store_s(self.qk_ub)
        # DSL 0.7 retains this KV version across QK and the delayed PV read.
        # PV is the final consumer; only then may the producer reuse its slot.
        # A synthetic re-publication would split that retained lifetime.
        self._stage_softmax(tick, m_seq, n, is_first, sum_slot)

    @jit
    def _stage_pv_update(self, kv_slot, tick: int, m_seq: int, n, is_last: int,
                         is_first: int, sum_slot: int):
        self._stage_pv_fanout(kv_slot, m_seq, n)
        self._stage_update(tick, m_seq, is_last, is_first,
                           sum_slot)

    @jit
    def _stage_update(self, tick: int, m_seq: int, is_last: int,
                      is_first: int, sum_slot: int):
        # At a part boundary, normalize O and either stage the split part or write final output.
        # Empty rows bypass the pipeline and are finalized after draining.
        exp_idx = tick % 2
        sum_idx = sum_slot

        m_idx = self._m_idx(m_seq)
        o_row_tile = m_idx
        lse_tile_base = (
            o_row_tile * (self.tile_cube_m // self.tile_vec_m)
            + self.subblock_idx) if const_expr(self._lse is not None) else None
        # LSE coordinates count 32-element tiles: flat offset m_idx*64 + aiv_id*32.
        if is_last == 1:
            if is_first == 1:
                if tick > dtypes.int64(0):
                    self.vector.wait_prev_out_copy()
                self.vector.init_o_last(self.pv_ub.consume(), sum_idx)
            else:
                self.vector.update_o_last(self.pv_ub.consume(), exp_idx, sum_idx)
            split_count = (dtypes.int64(1) if m_seq == self._first_split_row else dtypes.int64(0))
            split_count += (dtypes.int64(1) if m_seq == self._last_split_row else dtypes.int64(0))
            if split_count > 0:
                # A leading continuation uses slot 0. A trailing split uses slot 1 only
                # when this core also starts with a continuation.
                part_idx = (dtypes.int64(0) if m_seq == 0 else
                            (dtypes.int64(1) if self._s2_start > 0 else dtypes.int64(0)))
                slot = self._first_fd + part_idx
                fd_o, fd_ms = self._fd_workspace_views()
                local_first = ((Int32(1) if slot % 2 == 0 else Int32(0))
                               if self._fd_async == 1 else Int32(0))
                self.vector.stage_partial(fd_o, fd_ms, slot, sum_idx,
                                          skip_o=local_first)
                if self._fd_async == 1 and slot % 2 == 1:
                    # The second part publishes readiness only after all
                    # three MTE3 staging writes have completed.
                    # Drain MTE3 before publishing ready to the remote reducer.
                    # This is a local completion fence, not an all-core barrier.
                    vec_sync_all()
                    ready_word = _FD_READY_BASE + (slot // 2) * 2 + self.subblock_idx
                    vec_store_bypass(self._metadata_gm.ptr((ready_word,)), Int32(1))
            else:
                o_tile_gm = tile_slice(self._out, (self.tile_cube_m, self.tile_d),
                                      (o_row_tile, 0))
                self.vector.finalize_o(o_tile_gm, sum_idx,
                                       lse_gm=self._lse, lse_base=lse_tile_base)
        else:
            if is_first == 1:
                if tick > dtypes.int64(0):
                    self.vector.wait_prev_out_copy()
                self.vector.init_o(self.pv_ub.consume())
            else:
                self.vector.update_o(self.pv_ub.consume(), exp_idx)

    @jit
    def _fd_workspace_views(self):
        # FD views are needed only by split-row staging and final reduction.
        blocks = self._ws.shape[0] // WS_BYTES_PER_CORE
        slots = blocks * FD_SLOTS_PER_CORE
        fd_o0 = dtypes.int64(0)
        ws_f32 = self._ws.view(dtypes.float32)
        fd_o = ws_f32[fd_o0:fd_o0 + slots * FD_O_ELEMS, ].view(slots * N1, D)
        fd_ms0 = fd_o0 + slots * FD_O_ELEMS
        fd_ms = ws_f32[fd_ms0:fd_ms0 + slots * FD_MS_ELEMS, ].view(slots * FD_MS_ELEMS)
        return fd_o, fd_ms


def _batch_consistency_enabled():
    """Enable batch consistency at torch_npu deterministic level 3.

    This selects per-row sparse KV copies instead of paired-row DMA.
    """
    try:
        import torch_npu
        return int(torch_npu.npu._get_deterministic_level()) == 3
    except (AttributeError, ImportError):
        return False


def _validate_inputs(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
):
    """Validate tensor metadata only; never read tensor values on the host.

    Callers supply valid prefix sums, sequence lengths and active indices.
    seqused_q, when supplied, must equal the cu_seqlens_q spans (no padding).
    """
    if ori_kv is None:
        raise ValueError("Invalid input: ori_kv is required")
    if str(layout_q).upper() != "TND" or str(layout_kv).upper() != "PA_BBND":
        raise ValueError("Invalid input: mixed_quant_sparse_flash_mla supports only "
                         "layout_q='TND' and layout_kv='PA_BBND'")
    if int(quant_mode) != 1:
        raise ValueError("Invalid input: mixed_quant_sparse_flash_mla supports only "
                         "quant_mode=1")

    if not isinstance(q, torch.Tensor):
        raise TypeError("Invalid input: q must be a torch.Tensor")
    if q.dtype != torch.bfloat16 or q.dim() != _TENSOR_RANK_3D or int(q.shape[-1]) != D:
        raise ValueError("Invalid input: q must be a BF16 TND tensor with shape [T1, N1, 512]")
    rows, n1 = int(q.shape[0]), int(q.shape[1])
    if n1 != N1:
        raise ValueError(f"N1 must be {N1} (the only supported query-head "
                         f"count), got {n1}")
    if rows <= 0:
        raise ValueError("Invalid input: q must contain at least one token")

    if cu_seqlens_q is None:
        raise ValueError(
            "Invalid input: cu_seqlens_q is required for TND varlen and is never synthesized: "
            "pass an int32 [B+1] prefix-sum tensor ending at q.shape[0]")
    if (not isinstance(cu_seqlens_q, torch.Tensor)
            or cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.dim() != _TENSOR_RANK_1D):
        raise ValueError("Invalid input: cu_seqlens_q must be an int32 tensor of shape (B+1,)")
    if cu_seqlens_q.numel() < 2:
        raise ValueError("Invalid input: cu_seqlens_q must describe at least one batch")
    b = cu_seqlens_q.numel() - 1

    if seqused_q is not None:
        if not isinstance(seqused_q, torch.Tensor):
            raise TypeError("Invalid input: seqused_q must be a torch.Tensor when provided")
        if seqused_q.dtype != torch.int32 or seqused_q.dim() != _TENSOR_RANK_1D:
            raise ValueError("Invalid input: seqused_q must be a 1-D int32 tensor of shape (B,)")
        if seqused_q.numel() != b:
            raise ValueError("Invalid input: seqused_q must have one length per cu_seqlens_q batch")

    for name, seqused in (("seqused_ori_kv", seqused_ori_kv),
                          ("seqused_cmp_kv", seqused_cmp_kv)):
        if seqused is None:
            continue
        if not isinstance(seqused, torch.Tensor):
            raise TypeError(f"Invalid input: {name} must be a torch.Tensor when provided")
        if seqused.dtype != torch.int32 or seqused.dim() != _TENSOR_RANK_1D:
            raise ValueError(f"Invalid input: {name} must be a 1-D int32 tensor of shape (B,)")
        if int(seqused.numel()) != b:
            raise ValueError(f"Invalid input: {name} must have one entry per batch element (B={b})")

    has_cmp = (cmp_kv is not None or cmp_sparse_indices is not None
               or cmp_block_table is not None)
    if has_cmp:
        if (cmp_kv is None or cmp_sparse_indices is None
                or cmp_block_table is None):
            raise ValueError("ORI_CMP_SPARSE requires cmp_kv, cmp_sparse_indices, "
                             "cmp_block_table together")
    elif cmp_topk_length is not None:
        raise ValueError("Invalid input: cmp inputs are not valid for ORI_SPARSE")

    _validate_pa_side(ori_kv, ori_block_table, name="ori_kv",
                      row_bytes=KV_ROW_BYTES_ORI)
    _validate_indices(ori_sparse_indices, name="ori_sparse_indices", rows=rows)
    _validate_lengths(ori_topk_length, name="ori_topk_length", rows=rows)

    if has_cmp:
        _validate_pa_side(cmp_kv, cmp_block_table, name="cmp_kv",
                          row_bytes=KV_ROW_BYTES_CMP)
        _validate_indices(cmp_sparse_indices, name="cmp_sparse_indices", rows=rows)
        _validate_lengths(cmp_topk_length, name="cmp_topk_length", rows=rows)

    if sinks is None:
        raise ValueError("Invalid input: sinks is required: pass an FP32 [N1] tensor shared by all batches")
    if (not isinstance(sinks, torch.Tensor) or sinks.dtype != torch.float32
            or tuple(sinks.shape) != (n1,)):
        raise ValueError("Invalid input: sinks must be an FP32 tensor with shape [N1], shared by all batches")

    if metadata is None:
        raise ValueError(
            "Invalid input: metadata is required and is never synthesized: generate it with "
            "mixed_quant_sparse_flash_mla_metadata(ori_topk_length, cmp_topk_length, "
            "num_heads_q=64, num_heads_kv=1, head_dim=512, quant_mode=1, has_cmp_kv=...) and pass "
            "the result in (two-stage call)")
    if not isinstance(metadata, torch.Tensor):
        raise TypeError("Invalid input: metadata must be a torch.Tensor")
    if (metadata.dtype != torch.int32 or metadata.dim() != _TENSOR_RANK_1D
            or metadata.numel() != MQSMLA_METADATA_TOTAL_SIZE):
        raise ValueError(f"Invalid input: metadata must be a 1-D int32 tensor with exactly "
                         f"{MQSMLA_METADATA_TOTAL_SIZE} elements (fixed shape)")


def _validate_pa_side(kv, block_table, *, name, row_bytes):
    if not isinstance(kv, torch.Tensor) or not isinstance(block_table, torch.Tensor):
        raise TypeError(f"Invalid input: {name} and {name}_block_table must be torch.Tensor")
    if (kv.dtype != torch.uint8 or kv.dim() != _TENSOR_RANK_4D
            or int(kv.shape[2]) != N2 or int(kv.shape[-1]) != row_bytes):
        side = "FP8 (ori)" if name != "cmp_kv" else "FP4 (cmp)"
        raise ValueError(
            f"Invalid input: {name} must be a uint8 [blocknum, blocksize, 1, {row_bytes}] "
            f"tensor ({side} byte view); got shape {tuple(kv.shape)}, "
            f"dtype {kv.dtype}")
    if int(kv.shape[0]) <= 0 or int(kv.shape[1]) <= 0:
        raise ValueError(f"Invalid input: {name} must be non-empty")
    if (block_table.dtype != torch.int32 or block_table.dim() != _TENSOR_RANK_2D
            or int(block_table.shape[0]) <= 0 or int(block_table.shape[1]) <= 0):
        raise ValueError(f"Invalid input: {name}_block_table must be a non-empty int32 "
                         f"[batch, num_blocks] tensor")
    bs = int(kv.shape[1])
    strides = tuple(kv.stride())
    page_stride = int(kv.stride(0))  # The uint8 page stride is measured in bytes, independently of page length.
    if strides[1:] != (N2 * row_bytes, row_bytes, 1):
        raise ValueError(f"Invalid input: {name} only supports axis-0 non-contiguity; "
                         f"page dimensions must be contiguous, got strides {strides}")
    if page_stride < bs * N2 * row_bytes:
        raise ValueError(f"Invalid input: {name} page stride must be at least {bs * N2 * row_bytes} "
                         "bytes (positive, non-overlapping pages)")
    if bs > _MAX_PA_BLOCK_SIZE:
        raise ValueError("PA block size must be in 1..1024")


def _validate_indices(idx, *, name, rows):
    if not isinstance(idx, torch.Tensor) or idx.dtype != torch.int32:
        raise ValueError(f"Invalid input: {name} must be an int32 torch.Tensor")
    if (idx.dim() != _TENSOR_RANK_3D or int(idx.shape[0]) != rows
            or int(idx.shape[1]) != N2):
        raise ValueError(f"Invalid input: {name} must be int32 with shape [T1, 1, K] "
                         f"(one row per query token, {rows} rows)")
    k = int(idx.shape[2])
    if k <= 0:
        raise ValueError(f"Invalid input: {name} must have at least one index column")


def _validate_lengths(x, *, name, rows):
    if x is None:
        return
    if not isinstance(x, torch.Tensor) or x.dtype != torch.int32:
        raise ValueError(f"Invalid input: {name} must be an int32 torch.Tensor")
    if x.dim() != _TENSOR_RANK_2D or int(x.shape[0]) != rows or int(x.shape[1]) != N2:
        raise ValueError(f"Invalid input: {name} must be int32 with shape [T1, 1]")


def construct_workspace(metadata, addr_bytes=0):
    # Query the same device core count as metadata; workspace capacity determines the launch grid.
    blocks = _get_cube_core_num(metadata.device)
    if not 0 < blocks <= AIC_CORE_MAX_NUM:
        raise ValueError(f"Invalid input: core count must be in 1..{AIC_CORE_MAX_NUM}, got {blocks}")
    ws_bytes = blocks * WS_BYTES_PER_CORE
    # Allocation only: avoid a zeros device op during ACLGraph capture.
    # FD slots are fully written by their staging core before publication.
    # Append the prepass row-index table to the same device allocation.
    # The kernel writes and reads the table without host staging or H2D copies.
    return torch.empty(ws_bytes + addr_bytes, dtype=torch.uint8,
                       device=metadata.device)


@host
def _run_mqsmla(q_gm: Tensor, ori_kv_gm: Tensor, ori_idx_gm: Tensor,
                ori_bt_gm: Tensor, ori_len_gm: Tensor, out_gm: Tensor,
                sinks_gm: Tensor, cu_seqlens_q: Tensor,
                metadata_gm: Tensor, workspace_gm: Tensor,
                cmp_kv_gm: Tensor = None, cmp_idx_gm: Tensor = None,
                cmp_bt_gm: Tensor = None, cmp_len_gm: Tensor = None,
                lse_gm: Tensor = None, softmax_scale: dtypes.float32 = 1.0,
                batch_consistency: dtypes.int32 = 0,
                addr_ws_gm: Tensor = None):
    """Adapt dynamic tensor views and launch the device kernel from a JIT entry."""
    # Adapt DSL views during tracing without torch data conversions.
    # Flatten Q/O to [T1*64,512], indices to [T1,K], lengths to [T1],
    # cu_q to [1,B+1], sinks to [1,64], and LSE to [T1*64].
    # Keep KV byte-addressed so physical page padding cannot become logical tokens.
    m_rows = q_gm.shape[0] * q_gm.shape[1]
    q_flat = q_gm.view(m_rows, D)
    out_flat = out_gm.view(m_rows, D)
    ori_pool = ori_kv_gm
    ori_idx_flat = ori_idx_gm.view(
        ori_idx_gm.shape[0] * ori_idx_gm.shape[1], ori_idx_gm.shape[2])
    ori_len_flat = None
    if const_expr(ori_len_gm is not None):
        ori_len_flat = ori_len_gm.view(
            ori_len_gm.shape[0] * ori_len_gm.shape[1])
    cu_q = cu_seqlens_q.view(1, cu_seqlens_q.shape[0])
    sinks_k = sinks_gm.view(1, q_gm.shape[1])
    lse_flat = None
    if const_expr(lse_gm is not None):
        lse_flat = lse_gm.view(lse_gm.shape[1] * lse_gm.shape[2])
    cmp_pool = None
    cmp_idx_flat = None
    cmp_len_flat = None
    if const_expr(cmp_kv_gm is not None):
        cmp_pool = cmp_kv_gm
        cmp_idx_flat = cmp_idx_gm.view(
            cmp_idx_gm.shape[0] * cmp_idx_gm.shape[1], cmp_idx_gm.shape[2])
        if const_expr(cmp_len_gm is not None):
            cmp_len_flat = cmp_len_gm.view(
                cmp_len_gm.shape[0] * cmp_len_gm.shape[1])
    op = MqsmlaKernel(
        tile_cube_m=N1, tile_vec_m=N1 // 2, tile_n=TILE_N,
        has_cmp=(cmp_kv_gm is not None))
    # Each core has two FD staging slots; workspace capacity determines the dynamic launch grid.
    blocks = workspace_gm.shape[0] // WS_BYTES_PER_CORE
    op[blocks](out_flat, q_flat, sinks_k,
               ori_pool, ori_bt_gm, workspace_gm,
               cmp_pool, cmp_bt_gm, cmp_idx_flat,
               ori_idx_flat, cu_q, lse_flat, ori_len_flat,
               cmp_len_flat, metadata_gm,
               softmax_scale, batch_consistency, addr_ws_gm)


def clear_caches():
    """Close cached executables."""
    with _COMPILED_KERNEL_LOCK:
        for compiled in _COMPILED_KERNELS.values():
            compiled.close()
        _COMPILED_KERNELS.clear()


def _get_compiled_kernel(
    ori_kv, ori_sparse_indices, ori_block_table, ori_topk_length,
    cmp_kv=None, cmp_sparse_indices=None, cmp_block_table=None,
    cmp_topk_length=None, lse=None,
):
    """Cache the dynamic executable by optional-input structure.

    The host entry validates the fixed tensor contract before calling here.
    S2, K1/K2 and page-table widths remain runtime values in every template.
    Eligible pages share one symbolic vector template; remaining layouts
    share scalar AOT.
    """
    has_cmp = cmp_kv is not None
    # Select algorithms by layout/capacity only, never by individual S2/K values.
    eligible = _addr_vec_eligible(
        ori_kv, ori_block_table, ori_sparse_indices,
        cmp_kv, cmp_block_table, cmp_sparse_indices)
    eff_mode = "off"
    if eligible:
        compact = (ori_block_table.shape[1] <= _ADDR_COMPACT_BT
                   and ori_sparse_indices.shape[2] <= _ADDR_COMPACT_K)
        if has_cmp:
            compact = (compact and cmp_block_table.shape[1] <= _ADDR_COMPACT_BT
                       and cmp_sparse_indices.shape[2] <= _ADDR_COMPACT_K)
        eff_mode = "pre" if compact else "arena"
    return _get_compiled_variant(
        has_cmp, ori_topk_length is not None, cmp_topk_length is not None,
        lse is not None, eff_mode,
    )


def _get_compiled_variant(
    has_cmp, has_ori_length, has_cmp_length, has_lse, addr_mode,
):
    """Share compilation, caching and address-mode lifetime with AOT export."""
    key = (has_cmp, has_ori_length, has_cmp_length, has_lse, addr_mode)
    with _COMPILED_KERNEL_LOCK:
        compiled = _COMPILED_KERNELS.get(key)
        if compiled is not None:
            return compiled
        global _ADDR_VEC_OVERRIDE
        previous_mode = _ADDR_VEC_OVERRIDE
        _ADDR_VEC_OVERRIDE = addr_mode
        try:
            # CANNBotDSL resolves packaged resources through its compile hook.
            # The same signature is observed during collection and runtime.
            compiled = _compile_mqsmla(*key)
            # Publish only after compilation succeeds; failures can be retried.
            _COMPILED_KERNELS[key] = compiled
            return compiled
        finally:
            _ADDR_VEC_OVERRIDE = previous_mode


def _compile_mqsmla(
    has_cmp, has_ori_length, has_cmp_length, has_lse, addr_mode,
):
    """Declare the dynamic signature; the caller holds the compilation lock."""
    rows = Dim("T1", min=1)
    batch = Dim("B", min=1)
    ori_block_size = Dim("ORI_BS", min=1, max=_MAX_PA_BLOCK_SIZE)
    cmp_block_size = Dim("CMP_BS", min=1, max=_MAX_PA_BLOCK_SIZE)
    ori_table_width = Dim("ORI_BT", min=1)
    cmp_table_width = Dim("CMP_BT", min=1)
    ori_slots = Dim("K1", min=1)
    cmp_slots = Dim("K2", min=1)
    ori_stride = Dim("ORI_STRIDE", min=1)
    cmp_stride = Dim("CMP_STRIDE", min=1)
    # Divisibility lets the kernel recover its runtime launch grid.
    workspace_bytes = Dim("WORKSPACE_BYTES", min=WS_BYTES_PER_CORE,
                          multiple_of=WS_BYTES_PER_CORE)
    i32 = dtypes.int32
    # Argument order matches _run_mqsmla, including absent optional tensors.
    specs = (
        TensorSpec((rows, N1, D), dtypes.bfloat16),
        TensorSpec((Dim("ORI_PAGES", min=1), ori_block_size, N2, KV_ROW_BYTES_ORI),
                   dtypes.uint8, stride=(ori_stride, KV_ROW_BYTES_ORI, KV_ROW_BYTES_ORI, 1)),
        TensorSpec((rows, N2, ori_slots), i32),
        TensorSpec((batch, ori_table_width), i32),
        TensorSpec((rows, N2), i32) if has_ori_length else None,
        TensorSpec((rows, N1, D), dtypes.bfloat16),
        TensorSpec((N1,), dtypes.float32),
        TensorSpec((batch + 1,), i32),
        TensorSpec((MQSMLA_METADATA_TOTAL_SIZE,), i32),
        TensorSpec((workspace_bytes,), dtypes.uint8),
        TensorSpec((Dim("CMP_PAGES", min=1), cmp_block_size, N2, KV_ROW_BYTES_CMP),
                   dtypes.uint8, stride=(cmp_stride, KV_ROW_BYTES_CMP, KV_ROW_BYTES_CMP, 1))
        if has_cmp else None,
        TensorSpec((rows, N2, cmp_slots), i32) if has_cmp else None,
        TensorSpec((batch, cmp_table_width), i32) if has_cmp else None,
        TensorSpec((rows, N2), i32) if has_cmp_length else None,
        TensorSpec((N2, rows, N1), dtypes.float32) if has_lse else None,
        dtypes.float32,
        i32,
        TensorSpec((Dim("ADDR_WS_BYTES", min=1),), dtypes.uint8)
        if addr_mode != "off" else None,
    )
    return cannbotdsl.compile(_run_mqsmla, *specs)


def mixed_quant_sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
    out,
    lse,
):
    _validate_inputs(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        quant_mode=quant_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
    )
    for name, tensor, shape, dtype in (
        ("out", out, tuple(q.shape), torch.bfloat16),
        ("lse", lse, (N2, q.shape[0], q.shape[1]), torch.float32),
    ):
        if name == "lse" and not return_softmax_lse:
            shape = (0,)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Invalid input: {name} must be a caller-allocated tensor")
        if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                or not tensor.is_contiguous()):
            raise ValueError(f"Invalid input: {name} must be contiguous {dtype} {shape}")
    # An absent length tensor activates all sparse-index columns; the kernel
    # reads K from the index shape without allocating a replacement tensor.
    addr_ws = None
    addr_bytes = 0
    if _addr_vec_eligible(ori_kv, ori_block_table, ori_sparse_indices,
                          cmp_kv, cmp_block_table, cmp_sparse_indices):
        columns = _addr_tab_w(ori_sparse_indices.shape[2])
        if cmp_kv is not None:
            columns += _addr_tab_w(cmp_sparse_indices.shape[2])
        addr_bytes = q.shape[0] * columns * 4
    # Allocate FD staging and the address table together.
    workspace_full = construct_workspace(metadata, addr_bytes)
    if addr_bytes:
        workspace = workspace_full[:-addr_bytes]
        addr_ws = workspace_full[-addr_bytes:]
    else:
        workspace = workspace_full
    tensors = (
        q, ori_kv, ori_sparse_indices, ori_block_table, ori_topk_length,
        out, sinks, cu_seqlens_q, metadata, workspace,
        cmp_kv, cmp_sparse_indices, cmp_block_table, cmp_topk_length,
        lse if return_softmax_lse else None,
    )
    compiled = _get_compiled_kernel(
        ori_kv, ori_sparse_indices, ori_block_table, ori_topk_length,
        cmp_kv, cmp_sparse_indices, cmp_block_table, cmp_topk_length,
        lse if return_softmax_lse else None,
    )
    # AOT fixes absent Tensor parameters at compile time and removes them from
    # the runtime signature. Remaining Tensor arguments keep their order.
    scale = (float(softmax_scale) if softmax_scale is not None
             else 1.0 / math.sqrt(q.shape[-1]))
    batch_consistency = 1 if _batch_consistency_enabled() else 0
    compiled(*(tensor for tensor in tensors if tensor is not None), scale,
             batch_consistency,
             *([addr_ws] if addr_ws is not None else []))


@export("mqsmla")
def export_mqsmla():
    """Collect all 36 optional-input/addressing variants with dynamic shapes."""
    clear_caches()
    try:
        for has_cmp in (True, False):
            cmp_lengths = (True, False) if has_cmp else (False,)
            for has_ori_length, has_cmp_length, has_lse, addr_mode in product(
                (True, False), cmp_lengths, (True, False), ("pre", "off", "arena"),
            ):
                _get_compiled_variant(
                    has_cmp, has_ori_length, has_cmp_length, has_lse, addr_mode,
                )
    finally:
        clear_caches()
