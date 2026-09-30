# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math

import torch

import cannbotdsl
from cannbotdsl.lang.host import host
from cannbotdsl import dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.channel import Channel
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.tensor import tile_slice
from cannbotdsl.tensor import MemLoc
from cannbotdsl.tensor import Tensor
from cannbotdsl.ops.reg.cast import RoundingMode
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import (
    PackMode,
    UnpackMode,
    full_mask,
    update_mask,
    mask_counter,
    vadd,
    vadds,
    vcast,
    vgts,
    vdiv,
    vdups,
    vload,
    vload_broadcast,
    vload_unpack,
    vmaxs,
    vmem_bar,
    vmul,
    vmuls,
    vreduce_sum,
    vselect,
    vsqrt,
    vstore,
    vstore_first,
    vstore_pack,
)

VL = 64
DEFAULT_BLOCK_NUM = 64
FULL_LOAD_R_MAX = 16384
DICHOTOMY_ADD_COEFF = 2
RETAINED_SIZE_1K = 2 * 1024
DOUBLE_BUFFER_NUM = 2
X_REDUCE_TMP_NUM = 1
MULTI_FACTOR_2 = 2
FLOAT_BYTE_SIZE = 4
UB_SIZE = 248 * 1024
_F32_MAX = float.fromhex("0x1.fffffep+127")

_TORCH_TO_DSL = {
    torch.bfloat16: dtypes.bfloat16,
    torch.float16: dtypes.float16,
    torch.float32: dtypes.float32,
}


@kernel
class RowFullLoadKernel:
    def __init__(self, num_col, dtype=dtypes.bfloat16):
        self._num_col = int(num_col)
        self._dtype = dtype
        self._x_is_16bit = dtype in (dtypes.bfloat16, dtypes.float16)
        self._num_seg = math.ceil(self._num_col / VL)
        self._w = self._num_seg * VL
        self._nca = self._w
        self._fp = 1 << ((self._nca - 1).bit_length() - 1)
        fold_loops = math.ceil(self._fp / VL)
        self._tmp_stride = math.ceil(fold_loops / 8) * 8
        self._row_stride = max(self._tmp_stride, VL)
        elem_bytes = 2 if self._x_is_16bit else 4
        gamma_bytes = elem_bytes if self._x_is_16bit else 4
        nca = self._nca
        ub_factor = (UB_SIZE - RETAINED_SIZE_1K - nca * gamma_bytes) // (
            nca * elem_bytes * MULTI_FACTOR_2 * DOUBLE_BUFFER_NUM
            + FLOAT_BYTE_SIZE * (DOUBLE_BUFFER_NUM + X_REDUCE_TMP_NUM)
            + self._row_stride * FLOAT_BYTE_SIZE
            + FLOAT_BYTE_SIZE * 2
        )
        self._row_factor = max(int(ub_factor), 1)
        buf_dtype = dtype if self._x_is_16bit else dtypes.float32
        self._reduce_branch = 0
        if self._num_col <= VL:
            self._reduce_branch = 1
        elif self._num_col <= VL * 2:
            self._reduce_branch = 2
        elif self._nca <= VL * VL * DICHOTOMY_ADD_COEFF:
            self._reduce_branch = 3
        else:
            self._reduce_branch = 4
        self._y_aligned = num_col % VL == 0
        if self._reduce_branch <= 2:
            self._tmp = Buffer(
                MemLoc.UB, (1, max(self._tmp_stride, VL)), dtypes.float32
            )
        else:
            rf = self._row_factor
            self._tmp = Buffer(MemLoc.UB, (rf, self._row_stride), dtypes.float32)
        rf = self._row_factor
        self._ssq_buf = Buffer(MemLoc.UB, (rf, 1), dtypes.float32)
        rf = self._row_factor
        w = self._w
        gamma_dtype = dtype if self._x_is_16bit else dtypes.float32
        self._gamma_ch = Channel(MemLoc.UB, shape=(1, w), dtype=gamma_dtype, depth=1).produce()
        self._y_ch = Channel(MemLoc.UB, shape=(rf, w), dtype=buf_dtype, depth=2)
        self._rstd_ch = Channel(MemLoc.UB, shape=(rf, 1), dtype=dtypes.float32, depth=2)
        self._x_ch = Channel(MemLoc.UB, shape=(rf, w), dtype=buf_dtype, depth=2)

    def __call__(
        self,
        gm_x: Tensor,
        gm_gamma: Tensor,
        gm_y: Tensor,
        gm_rstd: Tensor,
        epsilon: dtypes.float32,
    ):
        num_col = self._num_col
        avg = 1.0 / num_col
        rf = self._row_factor
        w = self._w
        bi = get_block_idx()
        bn = get_block_num()

        nr = gm_x.shape[0]
        total_tiles = (nr + rf - 1) // rf
        tiles_per_block = (total_tiles + bn - 1) // bn
        tile_idx = bi * tiles_per_block
        row_start = tile_idx * rf
        row_end = row_start + tiles_per_block * rf
        if row_end > nr:
            row_end = nr

        if row_start < nr:
            mem_copy(self._gamma_ch, tile_slice(gm_gamma, (1, num_col), (0, 0)))
            gamma = self._gamma_ch
            total = row_end - row_start
            outer = (total + rf - 1) // rf

            for i in range(outer):
                cur_start = row_start + i * rf
                cur_rf = rf
                if i == outer - 1:
                    cur_rf = total - i * rf

                if cur_rf > 0:
                    mem_copy(self._x_ch.produce(), tile_slice(gm_x, (rf, num_col), (tile_idx, 0)))
                    x = self._x_ch.consume()
                    self._compute_x_squared_sum(
                        x,
                        self._tmp,
                        self._ssq_buf,
                        w,
                        cur_rf,
                    )
                    self._compute_rstd(
                        self._ssq_buf,
                        self._tmp,
                        self._rstd_ch.produce(),
                        avg,
                        epsilon,
                        cur_rf,
                    )
                    mem_copy(
                        tile_slice(gm_rstd, (rf, 1), (cur_start // rf, 0)),
                        self._rstd_ch.consume(),
                    )
                    self._compute_y(
                        x,
                        gamma,
                        self._y_ch.produce(),
                        self._tmp,
                        w,
                        cur_rf,
                    )

                    mem_copy(tile_slice(gm_y, (rf, num_col), (tile_idx, 0)), self._y_ch.consume())
                    tile_idx = tile_idx + 1

    @jit
    def _compute_x_squared_sum(self, x_ch, tmp_buf, ssq_buf, w, cur_rf):
        num_col = self._num_col
        is_16bit = self._x_is_16bit
        with vf(mode="simd"):
            full = full_mask()
            zero = vdups(0.0, dtypes.float32, mask=full)

            if const_expr(self._reduce_branch <= 2):
                for k in cannbotdsl.range(cur_rf):
                    x_base = k * w
                    if const_expr(self._reduce_branch == 1):
                        preg = update_mask(num_col, elem_bits=32)[0]
                        if const_expr(is_16bit):
                            xu = vload_unpack(x_ch, x_base, unpack_mode=UnpackMode.B16_TO_B32)
                            x0 = vcast(xu, dtypes.float32, mask=preg)
                        else:
                            x0 = vload(x_ch, x_base)
                            x0 = vselect(x0, zero, cond_mask=preg)
                        x0 = vmul(x0, x0, mask=preg)
                        total = vreduce_sum(x0, mask=preg)
                    else:
                        preg_full = update_mask(num_col - VL, elem_bits=32)[0]
                        if const_expr(is_16bit):
                            xu0 = vload_unpack(x_ch, x_base, unpack_mode=UnpackMode.B16_TO_B32)
                            xu1 = vload_unpack(
                                x_ch, x_base + VL, unpack_mode=UnpackMode.B16_TO_B32
                            )
                            x0 = vcast(xu0, dtypes.float32, mask=full)
                            x1 = vcast(xu1, dtypes.float32, mask=preg_full)
                        else:
                            x0 = vload(x_ch, x_base)
                            x1 = vload(x_ch, x_base + VL)
                            x1 = vselect(x1, zero, cond_mask=preg_full)
                        x0 = vmul(x0, x0, mask=full)
                        x1 = vmul(x1, x1, mask=preg_full)
                        s = vadd(x0, x1, mask=full)
                        total = vreduce_sum(s, mask=full)
                    vstore_first(ssq_buf, k, total)
            else:
                fp = self._fp
                tail = num_col - fp if num_col > fp else 0
                tail_full = tail // VL
                tail_ceil = math.ceil(tail / VL)
                fold_loops = math.ceil(fp / VL)
                last_num = fp // VL
                row_stride = self._row_stride

                for k in cannbotdsl.range(cur_rf):
                    x_base = k * w
                    for r in range(tail_full):
                        off = r * VL
                        if const_expr(is_16bit):
                            xu0 = vload_unpack(
                                x_ch, x_base + off, unpack_mode=UnpackMode.B16_TO_B32
                            )
                            xu1 = vload_unpack(
                                x_ch, x_base + off + fp, unpack_mode=UnpackMode.B16_TO_B32
                            )
                            x0 = vcast(xu0, dtypes.float32, mask=full)
                            x1 = vcast(xu1, dtypes.float32, mask=full)
                        else:
                            x0 = vload(x_ch, x_base + off)
                            x1 = vload(x_ch, x_base + off + fp)
                        x0 = vmul(x0, x0, mask=full)
                        x1 = vmul(x1, x1, mask=full)
                        s = vadd(x0, x1, mask=full)
                        vstore_first(
                            tmp_buf, k * row_stride + r, vreduce_sum(s, mask=full)
                        )
                    tail_remain = tail - tail_full * VL
                    if tail_remain != 0:
                        preg_t = update_mask(tail_remain, elem_bits=32)[0]
                        off = tail_full * VL
                        if const_expr(is_16bit):
                            xu0 = vload_unpack(
                                x_ch, x_base + off, unpack_mode=UnpackMode.B16_TO_B32
                            )
                            xu1 = vload_unpack(
                                x_ch, x_base + off + fp, unpack_mode=UnpackMode.B16_TO_B32
                            )
                            x0 = vcast(xu0, dtypes.float32, mask=full)
                            x1 = vcast(xu1, dtypes.float32, mask=preg_t)
                        else:
                            x0 = vload(x_ch, x_base + off)
                            x1 = vload(x_ch, x_base + off + fp)
                            x1 = vselect(x1, zero, cond_mask=preg_t)
                        x0 = vmul(x0, x0, mask=full)
                        x1 = vmul(x1, x1, mask=preg_t)
                        s = vadd(x0, x1, mask=full)
                        vstore_first(
                            tmp_buf,
                            k * row_stride + tail_full,
                            vreduce_sum(s, mask=full),
                        )
                    for r in range(tail_ceil, fold_loops):
                        off = r * VL
                        preg_r = update_mask(num_col - off, elem_bits=32)[0]
                        if const_expr(is_16bit):
                            xu = vload_unpack(
                                x_ch, x_base + off, unpack_mode=UnpackMode.B16_TO_B32
                            )
                            x0 = vcast(xu, dtypes.float32, mask=full)
                        else:
                            x0 = vload(x_ch, x_base + off)
                        x0 = vselect(x0, zero, cond_mask=preg_r)
                        x0 = vmul(x0, x0, mask=full)
                        vstore_first(
                            tmp_buf, k * row_stride + r, vreduce_sum(x0, mask=full)
                        )

                vmem_bar(mode="vst_vld")

                for k in cannbotdsl.range(cur_rf):
                    if const_expr(self._reduce_branch == 3):
                        preg_last = update_mask(max(last_num, 1), elem_bits=32)[0]
                        x0 = vload(tmp_buf, k * row_stride)
                        total = vreduce_sum(x0, mask=preg_last)
                    else:
                        preg_last = update_mask(last_num - VL, elem_bits=32)[0]
                        x0 = vload(tmp_buf, k * row_stride)
                        x1 = vload(tmp_buf, k * row_stride + VL)
                        x1 = vselect(
                            x1,
                            vdups(0.0, dtypes.float32, mask=full),
                            cond_mask=preg_last,
                        )
                        s2 = vadd(x0, x1, mask=full)
                        total = vreduce_sum(s2, mask=full)
                    vstore_first(ssq_buf, k, total)

    @jit
    def _compute_rstd(self, ssq_buf, tmp_buf, rstd_ch, avg, epsilon, cur_rf):
        with vf(mode="simd"):
            full = full_mask()
            newton_loops = (self._row_factor + VL - 1) // VL
            remaining_rf = mask_counter(cur_rf)
            newton_one = vdups(1.0, dtypes.float32, mask=full)
            newton_half = vdups(0.5, dtypes.float32, mask=full)
            newton_one_half = vdups(1.5, dtypes.float32, mask=full)
            newton_zero = vdups(0.0, dtypes.float32, mask=full)
            newton_inf = vdups(_F32_MAX, dtypes.float32, mask=full)
            for ni in range(newton_loops):
                preg_n, remaining_rf = update_mask(remaining_rf, elem_bits=32)
                var = vload(ssq_buf, ni * VL)
                var = vmuls(var, avg, mask=preg_n)
                var = vadds(var, epsilon, mask=preg_n)
                var = vmaxs(var, -99.99, mask=preg_n)
                r = vdiv(newton_one, var, mask=preg_n)
                ys = vsqrt(r, mask=preg_n)
                t = vmuls(var, -0.5, mask=preg_n)
                t = vmul(t, ys, mask=preg_n)
                t1 = vadd(
                    newton_one_half,
                    vmul(t, ys, mask=preg_n),
                    mask=preg_n,
                )
                rstd = vmul(ys, t1, mask=preg_n)
                t3 = vmuls(var, -1.0, mask=preg_n)
                s = vadd(newton_one, vmul(t3, r, mask=preg_n), mask=preg_n)
                t4 = vmuls(rstd, -1.0, mask=preg_n)
                r = vadd(r, vmul(t4, rstd, mask=preg_n), mask=preg_n)
                s = vadd(s, vmul(var, r, mask=preg_n), mask=preg_n)
                s = vmul(s, rstd, mask=preg_n)
                rstd = vadd(
                    rstd,
                    vmul(s, newton_half, mask=preg_n),
                    mask=preg_n,
                )
                cmp_inf = vgts(var, _F32_MAX * 0.999, mask=preg_n)
                rstd = vselect(
                    newton_zero,
                    rstd,
                    cond_mask=cmp_inf,
                )
                cmp_pos = vgts(var, 0.0, mask=preg_n)
                rstd = vselect(
                    rstd,
                    newton_inf,
                    cond_mask=cmp_pos,
                )
                vstore(tmp_buf, ni * VL, rstd, preg_n)
                vstore(rstd_ch, ni * VL, rstd, preg_n)

    @jit
    def _compute_y(self, x_ch, gamma_buf, y_buf, rstd_buf, w, cur_rf):
        nca = self._nca
        num_col = self._num_col
        is_16bit = self._x_is_16bit
        out_dtype = self._dtype
        with vf(mode="simd"):
            full = full_mask()
            col_loops = math.ceil(nca / VL)
            for k in cannbotdsl.range(cur_rf):
                x_base = k * w
                rstd_brc = vload_broadcast(rstd_buf, k)
                if const_expr(self._y_aligned):
                    for j in range(col_loops):
                        off = j * VL
                        if const_expr(is_16bit):
                            xu = vload_unpack(
                                x_ch, x_base + off, unpack_mode=UnpackMode.B16_TO_B32
                            )
                            gu = vload_unpack(
                                gamma_buf, off, unpack_mode=UnpackMode.B16_TO_B32
                            )
                            x = vcast(xu, dtypes.float32, mask=full)
                            g = vcast(gu, dtypes.float32, mask=full)
                        else:
                            x = vload(x_ch, x_base + off)
                            g = vload(gamma_buf, off)
                        mul1 = vmul(x, rstd_brc, mask=full)
                        yval = vmul(mul1, g, mask=full)
                        if const_expr(is_16bit):
                            vstore_pack(
                                y_buf,
                                x_base + off,
                                vcast(
                                    yval, out_dtype, mask=full, rounding=RoundingMode.RN
                                ),
                                full,
                                pack_mode=PackMode.B32_TO_B16,
                            )
                        else:
                            vstore(y_buf, x_base + off, yval, full)
                else:
                    remaining = mask_counter(num_col)
                    for j in range(col_loops):
                        preg, remaining = update_mask(remaining, elem_bits=32)
                        off = j * VL
                        if const_expr(is_16bit):
                            xu = vload_unpack(
                                x_ch, x_base + off, unpack_mode=UnpackMode.B16_TO_B32
                            )
                            gu = vload_unpack(
                                gamma_buf, off, unpack_mode=UnpackMode.B16_TO_B32
                            )
                            x = vcast(xu, dtypes.float32, mask=full)
                            g = vcast(gu, dtypes.float32, mask=full)
                        else:
                            x = vload(x_ch, x_base + off)
                            g = vload(gamma_buf, off)
                        mul1 = vmul(x, rstd_brc, mask=preg)
                        yval = vmul(mul1, g, mask=preg)
                        if const_expr(is_16bit):
                            vstore_pack(
                                y_buf,
                                x_base + off,
                                vcast(
                                    yval, out_dtype, mask=full, rounding=RoundingMode.RN
                                ),
                                preg,
                                pack_mode=PackMode.B32_TO_B16,
                            )
                        else:
                            vstore(y_buf, x_base + off, yval, preg)


@kernel
class ColSplitKernel:
    def __init__(self, num_col, dtype=dtypes.bfloat16):
        self._num_col = int(num_col)
        self._dtype = dtype
        self._x_is_16bit = dtype in (dtypes.bfloat16, dtypes.float16)
        elem_bytes = 2 if self._x_is_16bit else 4
        gamma_bytes = elem_bytes

        depth, ub_factor, _, _ = self._find_best_colsplit_tiling(
            self._num_col, elem_bytes, gamma_bytes
        )
        self._col_tile = ub_factor
        self._num_tiles = math.ceil(self._num_col / ub_factor)
        self._num_seg = math.ceil(ub_factor / VL)
        self._w = self._num_seg * VL
        self._last_tile_n = (
            self._num_col - (self._num_tiles - 1) * ub_factor
            if self._num_tiles > 1
            else self._num_col
        )
        self._aligned = (self._num_col % ub_factor == 0) and (ub_factor % VL == 0)
        buf_dtype = dtype if self._x_is_16bit else dtypes.float32
        self._buf_dtype = buf_dtype

        self._gamma_16_ch = Channel(
            MemLoc.UB, shape=(1, self._w), dtype=buf_dtype, depth=2
        )
        self._x_db_ch = Channel(
            MemLoc.UB, shape=(1, self._w), dtype=buf_dtype, depth=depth
        )
        self._y_ch = Channel(
            MemLoc.UB, shape=(1, self._w), dtype=buf_dtype, depth=depth
        )
        self._rstd_gm_ch = Channel(
            MemLoc.UB, shape=(1, 1), dtype=dtypes.float32, depth=1
        ).produce()
        self._tile_ss_buf = Buffer(
            MemLoc.UB, (1, math.ceil(self._num_tiles / VL) * VL), dtypes.float32
        )
        self._row_rstd_buf = Buffer(MemLoc.UB, (1, 16), dtypes.float32)

    def __call__(
        self,
        gm_x: Tensor,
        gm_gamma: Tensor,
        gm_y: Tensor,
        gm_rstd: Tensor,
        epsilon: dtypes.float32,
    ):
        num_col = self._num_col
        avg = 1.0 / num_col
        bi = get_block_idx()
        bn = get_block_num()
        ct = self._col_tile
        nt = self._num_tiles

        nr = gm_x.shape[0]
        lcn = bn if bn < nr else nr
        rpc = (nr + lcn - 1) // lcn
        rs = bi * rpc
        re = min(rs + rpc, nr)

        if bi < lcn:
            cur_rows = re - rs

            for row in range(cur_rows):
                gm_row = rs + row
                mem_copy(self._x_db_ch.produce(), tile_slice(gm_x, (1, ct), (gm_row, 0)))
                for t in range(nt):
                    nxt = t + 1
                    tile_n = self._last_tile_n if t == nt - 1 else ct
                    if nxt < nt:
                        mem_copy(self._x_db_ch.produce(), tile_slice(gm_x, (1, ct), (gm_row, nxt)))
                    x = self._x_db_ch.consume()
                    if t == nt - 1 and not const_expr(self._aligned):
                        self._compute_x_squared_sum_tail(
                            x, self._tile_ss_buf, t, tile_n
                        )
                    else:
                        self._compute_x_squared_sum(x, self._tile_ss_buf, t)

                self._compute_rstd(
                    self._tile_ss_buf,
                    self._rstd_gm_ch,
                    self._row_rstd_buf,
                    row,
                    avg,
                    epsilon,
                    nt,
                )
                mem_copy(
                    tile_slice(gm_rstd, (1, 1), (gm_row, 0)),
                    tile_slice(self._rstd_gm_ch, (1, 1), (0, 0)),
                )

            for t in range(nt):
                tile_n = self._last_tile_n if t == nt - 1 else ct
                mem_copy(self._gamma_16_ch.produce(), tile_slice(gm_gamma, (1, ct), (0, t)))
                gamma = self._gamma_16_ch.consume()
                for row in range(cur_rows):
                    mem_copy(self._x_db_ch.produce(), tile_slice(gm_x, (1, ct), (rs + row, t)))
                    self._compute_y(
                        self._x_db_ch.consume(),
                        gamma,
                        self._y_ch.produce(),
                        self._row_rstd_buf,
                        row,
                        ct,
                    )
                    mem_copy(tile_slice(gm_y, (1, ct), (rs + row, t)), self._y_ch.consume())

    @jit
    def _compute_x_squared_sum(self, x_buf, ss_buf, t):
        with vf(mode="simd"):
            full = full_mask()
            zero = vdups(0.0, dtypes.float32, mask=full)
            acc = vdups(0.0, dtypes.float32, mask=full)
            pair_loops = self._num_seg // 2
            if const_expr(self._aligned):
                for seg in range(pair_loops):
                    off0 = (seg * 2) * VL
                    off1 = (seg * 2 + 1) * VL
                    if const_expr(self._x_is_16bit):
                        xu0 = vload_unpack(x_buf, off0, unpack_mode=UnpackMode.B16_TO_B32)
                        xu1 = vload_unpack(x_buf, off1, unpack_mode=UnpackMode.B16_TO_B32)
                        x0 = vcast(xu0, dtypes.float32, mask=full)
                        x1 = vcast(xu1, dtypes.float32, mask=full)
                    else:
                        x0 = vload(x_buf, off0)
                        x1 = vload(x_buf, off1)
                    x0 = vmul(x0, x0, mask=full)
                    x1 = vmul(x1, x1, mask=full)
                    s = vadd(x0, x1, mask=full)
                    acc = vadd(acc, vreduce_sum(s, mask=full), mask=full)
            else:
                remaining = mask_counter(self._col_tile)
                for seg in range(pair_loops):
                    off0 = (seg * 2) * VL
                    off1 = (seg * 2 + 1) * VL
                    m0, remaining = update_mask(remaining, elem_bits=32)
                    m1, remaining = update_mask(remaining, elem_bits=32)
                    if const_expr(self._x_is_16bit):
                        xu0 = vload_unpack(x_buf, off0, unpack_mode=UnpackMode.B16_TO_B32)
                        xu1 = vload_unpack(x_buf, off1, unpack_mode=UnpackMode.B16_TO_B32)
                        x0 = vcast(xu0, dtypes.float32, mask=full)
                        x1 = vcast(xu1, dtypes.float32, mask=full)
                        x0 = vselect(x0, zero, cond_mask=m0)
                        x1 = vselect(x1, zero, cond_mask=m1)
                    else:
                        x0 = vload(x_buf, off0)
                        x1 = vload(x_buf, off1)
                        x0 = vselect(x0, zero, cond_mask=m0)
                        x1 = vselect(x1, zero, cond_mask=m1)
                    x0 = vmul(x0, x0, mask=full)
                    x1 = vmul(x1, x1, mask=full)
                    s = vadd(x0, x1, mask=full)
                    acc = vadd(acc, vreduce_sum(s, mask=full), mask=full)
            if self._num_seg % 2 != 0:
                off = (pair_loops * 2) * VL
                if const_expr(self._aligned):
                    if const_expr(self._x_is_16bit):
                        xu = vload_unpack(x_buf, off, unpack_mode=UnpackMode.B16_TO_B32)
                        x0 = vcast(xu, dtypes.float32, mask=full)
                    else:
                        x0 = vload(x_buf, off)
                    x0 = vmul(x0, x0, mask=full)
                    acc = vadd(acc, vreduce_sum(x0, mask=full), mask=full)
                else:
                    m_last, _ = update_mask(remaining, elem_bits=32)
                    if const_expr(self._x_is_16bit):
                        xu = vload_unpack(x_buf, off, unpack_mode=UnpackMode.B16_TO_B32)
                        x0 = vcast(xu, dtypes.float32, mask=full)
                        x0 = vselect(x0, zero, cond_mask=m_last)
                    else:
                        x0 = vload(x_buf, off)
                        x0 = vselect(x0, zero, cond_mask=m_last)
                    x0 = vmul(x0, x0, mask=full)
                    acc = vadd(acc, vreduce_sum(x0, mask=full), mask=full)
            vstore_first(ss_buf, t, acc)

    @jit
    def _compute_x_squared_sum_tail(self, x_buf, ss_buf, t, tile_n):
        with vf(mode="simd"):
            full = full_mask()
            zero = vdups(0.0, dtypes.float32, mask=full)
            acc = vdups(0.0, dtypes.float32, mask=full)
            pair_loops = self._num_seg // 2
            remaining = mask_counter(tile_n)
            for seg in range(pair_loops):
                off0 = (seg * 2) * VL
                off1 = (seg * 2 + 1) * VL
                m0, remaining = update_mask(remaining, elem_bits=32)
                m1, remaining = update_mask(remaining, elem_bits=32)
                if const_expr(self._x_is_16bit):
                    xu0 = vload_unpack(x_buf, off0, unpack_mode=UnpackMode.B16_TO_B32)
                    xu1 = vload_unpack(x_buf, off1, unpack_mode=UnpackMode.B16_TO_B32)
                    x0 = vcast(xu0, dtypes.float32, mask=full)
                    x1 = vcast(xu1, dtypes.float32, mask=full)
                    x0 = vselect(x0, zero, cond_mask=m0)
                    x1 = vselect(x1, zero, cond_mask=m1)
                else:
                    x0 = vload(x_buf, off0)
                    x1 = vload(x_buf, off1)
                    x0 = vselect(x0, zero, cond_mask=m0)
                    x1 = vselect(x1, zero, cond_mask=m1)
                x0 = vmul(x0, x0, mask=full)
                x1 = vmul(x1, x1, mask=full)
                s = vadd(x0, x1, mask=full)
                acc = vadd(acc, vreduce_sum(s, mask=full), mask=full)
            vstore_first(ss_buf, t, acc)

    @jit
    def _compute_rstd(self, ss_buf, rstd_gm, row_rstd_buf, idx, avg, epsilon, nt):
        with vf(mode="simd"):
            full = full_mask()
            total = vdups(0.0, dtypes.float32, mask=full)
            for i in range(math.ceil(nt / VL)):
                pm, _ = update_mask(nt - i * VL, elem_bits=32)
                pv = vload(ss_buf, i * VL)
                total = vadd(total, vreduce_sum(pv, mask=pm), mask=full)
            var = vmuls(total, avg, mask=full)
            var = vadds(var, epsilon, mask=full)
            var = vmaxs(var, -99.99, mask=full)
            one = vdups(1.0, dtypes.float32, mask=full)
            r = vdiv(one, var, mask=full)
            y = vsqrt(r, mask=full)
            t = vmuls(var, -0.5, mask=full)
            t = vmul(t, y, mask=full)
            t1 = vadd(
                vdups(1.5, dtypes.float32, mask=full),
                vmul(t, y, mask=full),
                mask=full,
            )
            rstd = vmul(y, t1, mask=full)
            t3 = vmuls(var, -1.0, mask=full)
            s = vadd(one, vmul(t3, r, mask=full), mask=full)
            t4 = vmuls(rstd, -1.0, mask=full)
            r = vadd(r, vmul(t4, rstd, mask=full), mask=full)
            s = vadd(s, vmul(var, r, mask=full), mask=full)
            s = vmul(s, rstd, mask=full)
            rstd = vadd(
                rstd,
                vmul(s, vdups(0.5, dtypes.float32, mask=full), mask=full),
                mask=full,
            )
            cmp_inf = vgts(var, _F32_MAX * 0.999, mask=full)
            rstd = vselect(
                vdups(0.0, dtypes.float32, mask=full), rstd, cond_mask=cmp_inf
            )
            cmp_pos = vgts(var, 0.0, mask=full)
            rstd = vselect(
                rstd,
                vdups(_F32_MAX, dtypes.float32, mask=full),
                cond_mask=cmp_pos,
            )
            vstore_first(rstd_gm, 0, rstd)
            vstore_first(row_rstd_buf, idx, rstd)

    @jit
    def _compute_y(self, x_buf, gamma_buf, y_buf, rstd_buf, row_idx, tile_n):
        with vf(mode="simd"):
            full = full_mask()
            rstd_brc = vload_broadcast(rstd_buf, row_idx)
            pair_loops = self._num_seg // 2
            for seg in range(pair_loops):
                off0 = (seg * 2) * VL
                off1 = (seg * 2 + 1) * VL
                if const_expr(self._x_is_16bit):
                    xu0 = vload_unpack(x_buf, off0, unpack_mode=UnpackMode.B16_TO_B32)
                    xu1 = vload_unpack(x_buf, off1, unpack_mode=UnpackMode.B16_TO_B32)
                    gu0 = vload_unpack(gamma_buf, off0, unpack_mode=UnpackMode.B16_TO_B32)
                    gu1 = vload_unpack(gamma_buf, off1, unpack_mode=UnpackMode.B16_TO_B32)
                    x0 = vcast(xu0, dtypes.float32, mask=full)
                    x1 = vcast(xu1, dtypes.float32, mask=full)
                    g0 = vcast(gu0, dtypes.float32, mask=full)
                    g1 = vcast(gu1, dtypes.float32, mask=full)
                else:
                    x0 = vload(x_buf, off0)
                    x1 = vload(x_buf, off1)
                    g0 = vload(gamma_buf, off0)
                    g1 = vload(gamma_buf, off1)
                y0 = vmul(vmul(x0, rstd_brc, mask=full), g0, mask=full)
                y1 = vmul(vmul(x1, rstd_brc, mask=full), g1, mask=full)
                if const_expr(self._x_is_16bit):
                    vstore_pack(
                        y_buf,
                        off0,
                        vcast(y0, self._dtype, mask=full),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                    vstore_pack(
                        y_buf,
                        off1,
                        vcast(y1, self._dtype, mask=full),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                else:
                    vstore(y_buf, off0, y0, full)
                    vstore(y_buf, off1, y1, full)
            if self._num_seg % 2 != 0:
                off = (pair_loops * 2) * VL
                if const_expr(self._x_is_16bit):
                    xu = vload_unpack(x_buf, off, unpack_mode=UnpackMode.B16_TO_B32)
                    gu = vload_unpack(gamma_buf, off, unpack_mode=UnpackMode.B16_TO_B32)
                    x0 = vcast(xu, dtypes.float32, mask=full)
                    g0 = vcast(gu, dtypes.float32, mask=full)
                else:
                    x0 = vload(x_buf, off)
                    g0 = vload(gamma_buf, off)
                y0 = vmul(vmul(x0, rstd_brc, mask=full), g0, mask=full)
                if const_expr(self._x_is_16bit):
                    vstore_pack(
                        y_buf,
                        off,
                        vcast(y0, self._dtype, mask=full),
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                else:
                    vstore(y_buf, off, y0, full)

    @staticmethod
    def _find_best_colsplit_tiling(num_col, elem_bytes, gamma_bytes):
        candidates = []
        for d in (4, 2):
            uf = num_col
            while uf > 0:
                tw = math.ceil(uf / VL) * VL
                tnt = math.ceil(num_col / uf)
                gamma_ch_bytes = tw * gamma_bytes * 2
                ub_need = (
                    RETAINED_SIZE_1K
                    + tw * elem_bytes * d
                    + tw * elem_bytes * d
                    + gamma_ch_bytes
                    + VL * FLOAT_BYTE_SIZE
                    + math.ceil(tnt / VL) * VL * FLOAT_BYTE_SIZE
                    + 16 * FLOAT_BYTE_SIZE
                )
                if ub_need <= UB_SIZE:
                    candidates.append((d, uf, tw, tnt))
                    break
                uf //= 2
        if len(candidates) == 2:
            d4, d2 = candidates[0], candidates[1]
            if d2[3] <= 1 or d4[3] > d2[3] * 2:
                return d2
            return d4
        return candidates[0]


#  Host
class RmsNorm:
    def __init__(self, dtype=dtypes.bfloat16):
        self.dtype = dtype

    @staticmethod
    def _is_row_full_load(num_col, dtype=dtypes.float32):
        nca = math.ceil(num_col / VL) * VL
        is_16bit = dtype in (dtypes.bfloat16, dtypes.float16)
        elem_bytes = 2 if is_16bit else 4
        gamma_bytes = nca * elem_bytes
        if nca > FULL_LOAD_R_MAX:
            return False
        ub_factor = (UB_SIZE - RETAINED_SIZE_1K - gamma_bytes) / (
            nca * elem_bytes * MULTI_FACTOR_2 * DOUBLE_BUFFER_NUM
            + FLOAT_BYTE_SIZE * (DOUBLE_BUFFER_NUM + X_REDUCE_TMP_NUM)
        )
        return ub_factor >= 1

    @host
    def run(self, gm_x, gm_gamma, gm_y, gm_rstd, eps: dtypes.float32):
        num_col = gm_x.shape[1]
        num_row = gm_x.shape[0]
        block_factor = math.ceil(num_row / DEFAULT_BLOCK_NUM)
        block_dim = math.ceil(num_row / block_factor)
        if const_expr(self._is_row_full_load(num_col, self.dtype)):
            op = RowFullLoadKernel(num_col, self.dtype)
        else:
            op = ColSplitKernel(num_col, self.dtype)
        op[block_dim](gm_x, gm_gamma, gm_y, gm_rstd, eps)


def rms_norm(x, gamma, *, epsilon=1e-6):
    """RMSNorm with an fp32 rstd side output.

    Only the operator spec (ranks, shapes, dtypes, layout) is validated.
    """
    if x.dim() < 1:
        raise ValueError(f"x must have at least one dimension, got {x.dim()}")
    norm_rank = gamma.dim()
    if not 1 <= norm_rank <= x.dim():
        raise ValueError(
            f"gamma rank must be in [1, {x.dim()}], got {norm_rank}"
        )
    if tuple(x.shape[-norm_rank:]) != tuple(gamma.shape):
        raise ValueError(
            f"gamma shape {tuple(gamma.shape)} must match the trailing "
            f"{norm_rank} dimension(s) of x {tuple(x.shape)}"
        )
    if x.dtype not in _TORCH_TO_DSL:
        raise TypeError("x dtype must be float16, bfloat16 or float32")
    if gamma.dtype != x.dtype:
        raise TypeError(
            f"gamma dtype ({gamma.dtype}) must match x dtype ({x.dtype})"
        )
    if not x.is_contiguous() or not gamma.is_contiguous():
        raise ValueError("x and gamma must be contiguous")

    num_col = 1
    for d in gamma.shape:
        num_col *= int(d)
    num_row = 1
    for d in x.shape[:-norm_rank]:
        num_row *= int(d)
    device = x.device
    rstd_shape = (*x.shape[:-norm_rank], *([1] * norm_rank))

    out = torch.empty(x.shape, dtype=x.dtype, device=device)
    rstd = torch.empty(rstd_shape, dtype=torch.float32, device=device)

    x2d = x.reshape(num_row, num_col)
    y2d = out.reshape(num_row, num_col)
    gamma2d = gamma.reshape(1, num_col)
    rstd2d = rstd.reshape(num_row, 1)

    op = RmsNorm(dtype=_TORCH_TO_DSL[x.dtype])
    op.run(x2d, gamma2d, y2d, rstd2d, float(epsilon))
    return out, rstd
