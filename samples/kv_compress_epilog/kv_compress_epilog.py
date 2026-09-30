# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""KvCompressEpilog —— KV Cache 压缩更新的 CANNBot-DSL 实现（simd 寄存器模式 + channel-first）。

原地更新算子：将 bfloat16 激活值 x 量化压缩后，按 slotMapping 散写到 cache 的对应行。
slotMapping 值为 -1 的 token 跳过。输出行布局：

    [rope bf16 128B][nope fp8 (d-64)B][scale][pad]

约束：headDim >= kvCacheCol

x/y 用 Channel（depth=2，显式选槽），scale 用 Buffer（纯 V→V scratch）。
rowFactor 由 UB 预算计算，多行批量加载/量化/散写。
"""
import math

import torch

from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.channel import Channel
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.control_flow import range as dsl_range
from cannbotdsl.buffer import Buffer
from cannbotdsl.types.dtypes import bfloat16 as BF16, float32 as F32, uint8 as U8, uint32 as U32
from cannbotdsl.tensor import tile_slice
from cannbotdsl import MemLoc, RegLayout, Tensor, dtypes
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    PackMode, UnpackMode, full_mask, update_mask,
    vabs, vadd, vcast, vbitwise_and, vne, vdiv, vmaxs, vmins, vmuls,
    vload_unpack, vreduce_max,
    vreinterpret, vselect, vshl, vshr,
    vstore_first, vstore_pack, vdups, vdup,
    vsqueeze_and_storeunalign_init, vstore_unalign, vstore_unalign_begin, vstore_unalign_post,
)

VL = 64
DEFAULT_BLOCK_NUM = 64
ROPE_COLS = 64
ROPE_BYTES = ROPE_COLS * 2
FP8_E4M3FN_MAX = 448.0
FP8_E4M3FN_MIN = -448.0
GROUP_SIZE = 64
BLOCK_BYTES = 32
FP32_SHIFT_BITS = 23
FP32_MANTISSA_MASK = 0x7FFFFF
UB_SIZE = 248 * 1024
DOUBLE_BUFFER = 2
MAX_ROW_FACTOR = 4


def _ceil_div(a, b):
    return (a + b - 1) // b


def _round_up(a, b):
    return _ceil_div(a, b) * b


def _calc_kv_cache_col(d, quant_mode, quant_group_size=64):
    d_nope = d - ROPE_COLS
    num_groups = _ceil_div(d_nope, quant_group_size)
    if quant_mode == 0:
        return _round_up(d_nope + ROPE_BYTES + num_groups * 2, BLOCK_BYTES)
    return d_nope + ROPE_BYTES + num_groups * 1


def _calc_row_factor(d, kv_cache_col):
    x_align = _round_up(d, 16)
    y_align = _round_up(kv_cache_col, 32)
    rf = 1
    while True:
        x_size = rf * x_align * 2 * DOUBLE_BUFFER
        y_size = rf * y_align * 1 * DOUBLE_BUFFER
        tmp_size = _round_up(rf, 8) * 4
        total = x_size + y_size + tmp_size
        if total > UB_SIZE:
            return max(1, rf - 1)
        rf += 1


def _get_row_factor(d, kv_cache_col, bs, block_num):
    ub_rf = _calc_row_factor(d, kv_cache_col)
    rpc = _ceil_div(bs, block_num)
    return min(ub_rf, rpc, MAX_ROW_FACTOR) if rpc > 0 else 1


@kernel
class KvCompressEpilogKernel:
    """``round_scale`` True/False 统一 kernel。"""

    def __init__(self, d: int, kv_cache_col: int, quant_mode: int = 1,
                 round_scale: bool = True, row_factor: int = 1):
        self.d = int(d)
        self.kv_cache_col = int(kv_cache_col)
        self.quant_mode = int(quant_mode)
        self.round_scale = bool(round_scale)
        self.row_factor = int(row_factor)

    def __call__(self, gm_x: Tensor, gm_slot: Tensor, gm_cache: Tensor):
        d = self.d
        d_nope = d - ROPE_COLS
        num_groups = d_nope // VL
        kv_cache_col = self.kv_cache_col
        quant_mode = self.quant_mode
        round_scale = self.round_scale
        scale_bytes = 2 if quant_mode == 0 else 1
        concat_col = d_nope + ROPE_BYTES + num_groups * scale_bytes
        pad_col = kv_cache_col - concat_col
        x_w = _round_up(d, VL)
        y_w = _round_up(kv_cache_col, VL)
        row_factor = self.row_factor

        block_idx = get_block_idx()
        block_num = get_block_num()
        num_row = gm_x.shape[0]
        total_tiles = (num_row + row_factor - 1) // row_factor
        tiles_per_block = (total_tiles + block_num - 1) // block_num
        tile_idx = block_idx * tiles_per_block
        row_start = tile_idx * row_factor
        row_end = row_start + tiles_per_block * row_factor
        if row_end > num_row:
            row_end = num_row

        x_ch = Channel(MemLoc.UB, shape=(row_factor, x_w), dtype=BF16, depth=2)
        y_ch = Channel(MemLoc.UB, shape=(row_factor, y_w), dtype=U8, depth=2)
        ub_scale = Buffer(MemLoc.UB, (VL,), F32)

        if row_start < num_row:
            total = row_end - row_start
            outer = _ceil_div(total, row_factor)

            mem_copy(x_ch.produce(), tile_slice(gm_x, (row_factor, d), (tile_idx, 0)))
            for i in range(outer):
                cur_start = row_start + i * row_factor
                cur_rf = row_factor
                if i == outer - 1:
                    cur_rf = total - i * row_factor

                if cur_rf > 0:
                    nxt_tile = tile_idx + 1
                    nxt_start = cur_start + row_factor
                    if nxt_start < row_end:
                        mem_copy(x_ch.produce(), tile_slice(gm_x, (row_factor, d), (nxt_tile, 0)))

                    x = x_ch.consume()
                    y = y_ch.produce()
                    self._quant_rows(x, y, ub_scale, d_nope, num_groups, quant_mode,
                                     concat_col, pad_col, x_w, y_w, cur_rf, round_scale)

                    for k in range(cur_rf):
                        slot = gm_slot[cur_start + k]
                        if slot != -1:
                            mem_copy(tile_slice(gm_cache, (1, concat_col), (slot, 0)),
                                     tile_slice(y, (1, concat_col), (k, 0)))
                    tile_idx = nxt_tile

    @jit
    def _quant_rows(self, x_ch, y_ch, ub_scale, d_nope, num_groups, quant_mode,
                    concat_col, pad_col, x_w, y_w, row_count, round_scale):
        with vf(mode="simd"):
            mask32 = full_mask()
            coeff = 1.0 / FP8_E4M3FN_MAX

            if const_expr(round_scale):
                mant_mask = vdups(FP32_MANTISSA_MASK, U32, mask=mask32)
                zero_u = vdups(0, U32, mask=mask32)
                one_u = vdups(1, U32, mask=mask32)

            for r in dsl_range(0, row_count, 1):
                x_base = r * x_w
                y_base = r * y_w

                rope_val = vload_unpack(x_ch, x_base + d_nope, unpack_mode=UnpackMode.B16_TO_B32)
                vstore_pack(y_ch, y_base, rope_val, mask32, pack_mode=PackMode.B32_TO_B16)

                for g in range(num_groups):
                    off = g * VL
                    xu = vload_unpack(x_ch, x_base + off, unpack_mode=UnpackMode.B16_TO_B32)
                    xf = vcast(xu, F32, mask=mask32)
                    xa = vabs(xf, mask=mask32)
                    amax = vreduce_max(xa, mask=mask32)
                    safe_max = vmaxs(amax, 1e-4, mask=mask32)
                    scale = vmuls(safe_max, coeff, mask=mask32)

                    if const_expr(round_scale):
                        sb = vreinterpret(scale, U32)
                        eb = vshr(sb, FP32_SHIFT_BITS, mask=mask32)
                        mn = vbitwise_and(sb, mant_mask, mask=mask32)
                        hm = vne(mn, zero_u, mask=mask32)
                        inc = vselect(one_u, zero_u, cond_mask=hm)
                        er = vadd(eb, inc, mask=mask32)
                        sbr = vshl(er, FP32_SHIFT_BITS, mask=mask32)
                        scale = vreinterpret(sbr, F32)
                        ds = vdup(scale, mask=mask32)
                        q = vdiv(xf, ds, mask=mask32)
                        q = vmaxs(q, FP8_E4M3FN_MIN, mask=mask32)
                        q = vmins(q, FP8_E4M3FN_MAX, mask=mask32)
                        qf = vcast(q, dtypes.float8_e4m3fn, mask=mask32, reg_layout=RegLayout.ZERO)
                        vstore_pack(y_ch, y_base + ROPE_BYTES + off, qf, mask32, pack_mode=PackMode.B32_TO_B8)

                        if quant_mode == 1:
                            er_u8 = vcast(er, U8, mask=mask32, reg_layout=RegLayout.ZERO)
                            vstore_first(y_ch, y_base + ROPE_BYTES + d_nope + g, er_u8)
                        else:
                            scale_bf16 = vcast(scale, BF16, mask=mask32)
                            vstore_first(y_ch, y_base + ROPE_BYTES + d_nope + g * 2, scale_bf16)
                    else:
                        ds = vdup(scale, mask=mask32)
                        q = vdiv(xf, ds, mask=mask32)
                        q = vmaxs(q, FP8_E4M3FN_MIN, mask=mask32)
                        q = vmins(q, FP8_E4M3FN_MAX, mask=mask32)
                        qf = vcast(q, dtypes.float8_e4m3fn, mask=mask32, reg_layout=RegLayout.ZERO)
                        vstore_pack(y_ch, y_base + ROPE_BYTES + off, qf, mask32, pack_mode=PackMode.B32_TO_B8)

                        if quant_mode == 1:
                            sb = vreinterpret(scale, U32)
                            eb = vshr(sb, FP32_SHIFT_BITS, mask=mask32)
                            eb_u8 = vcast(eb, U8, mask=mask32, reg_layout=RegLayout.ZERO)
                            vstore_first(y_ch, y_base + ROPE_BYTES + d_nope + g, eb_u8)
                        else:
                            scale_bf16 = vcast(scale, BF16, mask=mask32)
                            vstore_first(y_ch, y_base + ROPE_BYTES + d_nope + g * 2, scale_bf16)

                pad_mask = update_mask(pad_col, 8)[0]
                zero_b = vdups(0, U8, mask=pad_mask)
                ureg = vstore_unalign_begin(y_ch)
                sqz = vsqueeze_and_storeunalign_init(zero_b, mask=pad_mask)
                vstore_unalign(y_ch, y_base + concat_col, sqz, ureg)
                vstore_unalign_post(y_ch, y_base + concat_col, ureg)


class KvCompressEpilog:
    def __init__(self, d, kv_cache_col, quant_mode=1, round_scale=True, row_factor=1, block_num=None):
        self.d = int(d)
        self.kv_cache_col = int(kv_cache_col)
        self.quant_mode = int(quant_mode)
        self.round_scale = bool(round_scale)
        self.row_factor = int(row_factor)
        self.block_num = block_num

    @host
    def run(self, gm_x, gm_slot, gm_cache):
        num_row = gm_x.shape[0]
        block_num = self.block_num if self.block_num is not None else _device_block_num()
        block_factor = math.ceil(num_row / block_num)
        block_dim = math.ceil(num_row / block_factor)
        op = KvCompressEpilogKernel(
            self.d, self.kv_cache_col, self.quant_mode,
            self.round_scale, self.row_factor)
        op[block_dim](gm_x, gm_slot, gm_cache)


def _device_block_num() -> int:
    """Return a launch size for the selected NPU (AIV core count)."""
    npu = getattr(torch, "npu", None)
    if npu is None:
        return DEFAULT_BLOCK_NUM

    try:
        device_index = npu.current_device()
    except RuntimeError:
        return DEFAULT_BLOCK_NUM

    properties = npu.get_device_properties(device_index)
    try:
        vector_core_num = int(properties.vector_core_num)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"NPU {device_index} does not expose a valid vector_core_num"
        ) from exc
    if vector_core_num <= 0:
        raise RuntimeError(
            f"NPU {device_index} reports invalid vector_core_num={vector_core_num}"
        )
    return min(vector_core_num, DEFAULT_BLOCK_NUM)


def kv_compress_epilog(
    cache, x, slot_mapping, *,
    quant_group_size=64, quant_mode=1, round_scale=True, x_scale=1.0,
):
    """KvCompressEpilog 前向（对外公开入口）。"""
    if x.dim() != 2 or x.dtype != torch.bfloat16:
        raise ValueError("x must be a 2-D bfloat16 tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")

    bs, d = x.shape[0], x.shape[1]
    if d % 64 != 0 or not 64 < d <= 8192:
        raise ValueError(f"headDim must be a multiple of 64 in (64, 8192], got {d}")
    if slot_mapping.shape[0] != bs:
        raise ValueError(
            f"slot_mapping length ({slot_mapping.shape[0]}) must equal batch ({bs})"
        )
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError("slot_mapping must be int32 or int64")
    if cache.dim() != 4 or cache.dtype != torch.uint8 or cache.shape[2] != 1:
        raise ValueError("cache must be a 4-D uint8 tensor with shape[2] == 1")

    bn, bs_c, _, hd = cache.shape
    ns = bn * bs_c
    kvcc = _calc_kv_cache_col(d, quant_mode, quant_group_size)
    if hd < kvcc:
        raise ValueError(f"headDim({hd}) must be >= kvCacheCol({kvcc})")

    cache_flat = cache.view(ns, hd).contiguous()
    if slot_mapping.dtype == torch.int32:
        slot_mapping = slot_mapping.to(torch.int64)

    row_factor = _get_row_factor(d, kvcc, bs, DEFAULT_BLOCK_NUM)
    op = KvCompressEpilog(d, kvcc, quant_mode, round_scale, row_factor)
    op.run(x, slot_mapping, cache_flat)
    torch.npu.synchronize()

    return cache
