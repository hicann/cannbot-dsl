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
from cannbotdsl import dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.jit_function import jit
from cannbotdsl.kernel_launcher import kernel
from cannbotdsl.arch import get_block_idx, get_block_num
from cannbotdsl.channel import Channel
from cannbotdsl.constexpr import const_expr
from cannbotdsl.tensor import local_slice, tile_view, mem_copy
from cannbotdsl.types import MemLoc, Tensor
from cannbotdsl.vf import vf
from cannbotdsl.raw_reg import (
    PackMode, UnpackMode, full_mask, update_mask,
    vadd, vadds, vcast, vcmp_gt_scalar, vdiv, vdup_scalar,
    vload, vload_brc, vload_unpack, vmaxs, vmem_bar,
    vmul, vmuls, vreduce_sum, vselect, vsqrt,
    vstore, vstore_first, vstore_pack, vdup_lane0,
)

VL = 64
DEFAULT_BLOCK_NUM = 64
FULL_LOAD_R_MAX = 16384
COL_TILE_CAP = 4096
COL_TILE_CAP_LARGE = 8192
DICHOTOMY_ADD_COEFF = 2
RETAINED_SIZE_1K = 2 * 1024
DOUBLE_BUFFER_NUM = 2
X_REDUCE_TMP_NUM = 1
MULTI_FACTOR_2 = 2
FLOAT_BYTE_SIZE = 4
UB_SIZE = 192 * 1024
_F32_MAX = float.fromhex('0x1.fffffep+127')

_TORCH_TO_DSL = {torch.bfloat16: dtypes.bfloat16, torch.float16: dtypes.float16, torch.float32: dtypes.float32}


def _find_power_two(n):
    if n <= 2:
        return 1
    return 2 ** int(math.floor(math.log2(n - 1)))

@jit
def _cast_b16_to_fp32(b16_buf, fp32_buf, num_seg, col_tile):
    with vf(mode="raw"):
        for seg in range(0, num_seg):
            mask, _ = update_mask(col_tile - seg * VL, elem_bits=32)
            off = seg * VL
            val = vload_unpack(b16_buf, off, mode=UnpackMode.B16_TO_B32)
            vstore(fp32_buf, off, vcast(val, dtypes.float32, mask=mask), mask)

@kernel
class RowFullLoadKernel:
    def __init__(self, num_col, dtype=dtypes.bfloat16):
        self._num_col = int(num_col)
        self._dtype = dtype
        self._x_is_16bit = dtype in (dtypes.bfloat16, dtypes.float16)
        self._num_seg = math.ceil(self._num_col / VL)
        self._w = self._num_seg * VL
        self._nca = self._w
        self._fp = _find_power_two(self._nca)
        fold_loops = math.ceil(self._fp / VL)
        self._tmp_stride = math.ceil(fold_loops / 8) * 8
        elem_bytes = 2 if self._x_is_16bit else 4
        gamma_bytes = self._w * (elem_bytes + FLOAT_BYTE_SIZE) if self._x_is_16bit else self._w * FLOAT_BYTE_SIZE
        self._depth = DOUBLE_BUFFER_NUM
        for d in (DOUBLE_BUFFER_NUM, 1):
            if RETAINED_SIZE_1K + gamma_bytes + self._w * elem_bytes * MULTI_FACTOR_2 * d + FLOAT_BYTE_SIZE * (d + X_REDUCE_TMP_NUM) + max(self._tmp_stride, VL) * FLOAT_BYTE_SIZE <= UB_SIZE:
                self._depth = d
                break
        buf_dtype = dtype if self._x_is_16bit else dtypes.float32
        if self._x_is_16bit:
            self._gamma_16_ch = Channel(MemLoc.UB, shape=(1, self._w), dtype=dtype, depth=1)
            self._gamma_fp32 = Buffer(MemLoc.UB, (1, self._w), dtypes.float32)
        else:
            self._gamma_fp32 = Channel(
                MemLoc.UB, shape=(1, self._w), dtype=dtypes.float32, depth=1,
            )
        self._x_ch = Channel(MemLoc.UB, shape=(1, self._w), dtype=buf_dtype, depth=self._depth)
        self._y_ch = Channel(MemLoc.UB, shape=(1, self._w), dtype=buf_dtype, depth=self._depth)
        self._rstd_ch = Channel(MemLoc.UB, shape=(1, VL), dtype=dtypes.float32, depth=self._depth)
        self._tmp = Buffer(MemLoc.UB, (1, max(self._tmp_stride, VL)), dtypes.float32)
        self._reduce = Buffer(MemLoc.UB, (1, VL), dtypes.float32)

    def __call__(self, gm_x: Tensor, gm_gamma: Tensor, gm_y: Tensor, gm_rstd: Tensor,
                 epsilon: dtypes.float32):
        num_col = self._num_col
        avg = 1.0 / num_col
        bi = get_block_idx()
        bn = get_block_num()

        nr = gm_x.shape[0]
        lcn = bn if bn < nr else nr
        rpc = (nr + lcn - 1) // lcn
        rs = bi * rpc
        re = rs + rpc
        if re > nr:
            re = nr

        if bi < lcn:
            gamma_view = self._load_gamma(gm_gamma)
            if self._depth == 2:
                mem_copy(self._x_ch, tile_view(gm_x, (1, num_col), (rs, 0)))
            for row in range(rs, re):
                if self._depth == 1:
                    mem_copy(self._x_ch, tile_view(gm_x, (1, num_col), (row, 0)))
                else:
                    nxt = row + 1
                    if nxt < re:
                        mem_copy(self._x_ch, tile_view(gm_x, (1, num_col), (nxt, 0)))
                self._compute(
                    gamma_view, avg, epsilon,
                )
                mem_copy(tile_view(gm_y, (1, num_col), (row, 0)),
                         self._y_ch)
                mem_copy(tile_view(gm_rstd, (1, 1), (row, 0)),
                         local_slice(self._rstd_ch, (1, 1), offset=0))

    @jit
    def _compute(self, gamma_fp32, avg, epsilon):
        x_buf = self._x_ch
        y_buf = self._y_ch
        rstd_ub = self._rstd_ch
        tmp_buf = self._tmp
        reduce_tmp = self._reduce
        nca = self._nca
        fp = self._fp
        num_col = self._num_col
        is_16bit = self._x_is_16bit
        out_dtype = self._dtype
        with vf(mode="raw"):
            full = full_mask(32)
            zero = vdup_scalar(0.0, dtypes.float32, mask=full)
            fold_loops = math.ceil(fp / VL)
            last_num = fp // VL
            tail = nca - fp if nca > fp else 0
            tail_full = tail // VL
            tail_ceil = math.ceil(tail / VL)

            # --- Phase 1: 二分折叠求 x 平方和 ---
            for r in range(tail_full):
                off = r * VL
                if const_expr(is_16bit):
                    xu0 = vload_unpack(x_buf, off, mode=UnpackMode.B16_TO_B32)
                    xu1 = vload_unpack(x_buf, off + fp, mode=UnpackMode.B16_TO_B32)
                    x0 = vcast(xu0, dtypes.float32, mask=full)
                    x1 = vcast(xu1, dtypes.float32, mask=full)
                    x0 = vselect(x0, zero, cond_mask=update_mask(num_col - off, elem_bits=32)[0])
                    x1 = vselect(x1, zero, cond_mask=update_mask(num_col - off - fp, elem_bits=32)[0])
                else:
                    x0 = vload(x_buf, off)
                    x0 = vselect(x0, zero, cond_mask=update_mask(num_col - off, elem_bits=32)[0])
                    x1 = vload(x_buf, off + fp)
                    x1 = vselect(x1, zero, cond_mask=update_mask(num_col - off - fp, elem_bits=32)[0])
                x0 = vmul(x0, x0, mask=full)
                x1 = vmul(x1, x1, mask=full)
                s = vadd(x0, x1, mask=full)
                vstore_first(tmp_buf, r, vreduce_sum(s, mask=full))
            tail_remain = tail - tail_full * VL
            if tail_remain != 0:
                preg_t = update_mask(tail_remain, elem_bits=32)[0]
                off = tail_full * VL
                if const_expr(is_16bit):
                    xu0 = vload_unpack(x_buf, off, mode=UnpackMode.B16_TO_B32)
                    xu1 = vload_unpack(x_buf, off + fp, mode=UnpackMode.B16_TO_B32)
                    x0 = vcast(xu0, dtypes.float32, mask=full)
                    x1 = vcast(xu1, dtypes.float32, mask=preg_t)
                    x0 = vselect(x0, zero, cond_mask=update_mask(num_col - off, elem_bits=32)[0])
                    x1 = vselect(x1, zero, cond_mask=update_mask(num_col - off - fp, elem_bits=32)[0])
                else:
                    x0 = vload(x_buf, off)
                    x0 = vselect(x0, zero, cond_mask=update_mask(num_col - off, elem_bits=32)[0])
                    x1 = vload(x_buf, off + fp)
                    x1 = vselect(x1, zero, cond_mask=update_mask(num_col - off - fp, elem_bits=32)[0])
                x0 = vmul(x0, x0, mask=full)
                x1 = vmul(x1, x1, mask=preg_t)
                x1 = vselect(x1, vdup_scalar(0.0, dtypes.float32, mask=full), cond_mask=preg_t)
                s = vadd(x0, x1, mask=full)
                vstore_first(tmp_buf, tail_full, vreduce_sum(s, mask=full))
            for r in range(tail_ceil, fold_loops):
                off = r * VL
                if const_expr(is_16bit):
                    xu = vload_unpack(x_buf, off, mode=UnpackMode.B16_TO_B32)
                    x0 = vcast(xu, dtypes.float32, mask=full)
                    x0 = vselect(x0, zero, cond_mask=update_mask(num_col - off, elem_bits=32)[0])
                else:
                    x0 = vload(x_buf, off)
                    x0 = vselect(x0, zero, cond_mask=update_mask(num_col - off, elem_bits=32)[0])
                x0 = vmul(x0, x0, mask=full)
                vstore_first(tmp_buf, r, vreduce_sum(x0, mask=full))
            vmem_bar(mode="vst_vld")
            if const_expr(nca <= VL * VL * DICHOTOMY_ADD_COEFF):
                preg_last = update_mask(last_num, elem_bits=32)[0]
                x0 = vload(tmp_buf, 0)
                total = vreduce_sum(x0, mask=preg_last)
            else:
                preg_last = update_mask(last_num - VL, elem_bits=32)[0]
                x0 = vload(tmp_buf, 0)
                x1 = vload(tmp_buf, VL)
                x1 = vselect(x1, vdup_scalar(0.0, dtypes.float32, mask=full), cond_mask=preg_last)
                s2 = vadd(x0, x1, mask=full)
                total = vreduce_sum(s2, mask=full)

            # --- Phase 2: 牛顿迭代求 rstd = 1/sqrt(var + eps) ---
            var = vmuls(total, avg, mask=full)
            var = vadds(var, epsilon, mask=full)
            var = vmaxs(var, -99.99, mask=full)
            one = vdup_scalar(1.0, dtypes.float32, mask=full)
            r = vdiv(one, var, mask=full)
            y = vsqrt(r, mask=full)
            t = vmuls(var, -0.5, mask=full)
            t = vmul(t, y, mask=full)
            t1 = vadd(vdup_scalar(1.5, dtypes.float32, mask=full), vmul(t, y, mask=full), mask=full)
            rstd = vmul(y, t1, mask=full)
            t3 = vmuls(var, -1.0, mask=full)
            s = vadd(one, vmul(t3, r, mask=full), mask=full)
            t4 = vmuls(rstd, -1.0, mask=full)
            r = vadd(r, vmul(t4, rstd, mask=full), mask=full)
            s = vadd(s, vmul(var, r, mask=full), mask=full)
            s = vmul(s, rstd, mask=full)
            rstd = vadd(rstd, vmul(s, vdup_scalar(0.5, dtypes.float32, mask=full), mask=full), mask=full)
            cmp_inf = vcmp_gt_scalar(var, _F32_MAX * 0.999, mask=full)
            rstd = vselect(vdup_scalar(0.0, dtypes.float32, mask=full), rstd, cond_mask=cmp_inf)
            cmp_pos = vcmp_gt_scalar(var, 0.0, mask=full)
            rstd = vselect(rstd, vdup_scalar(_F32_MAX, dtypes.float32, mask=full), cond_mask=cmp_pos)
            vstore(rstd_ub, 0, rstd, full)
            rstd_brc = vdup_lane0(rstd, mask=full)

            # --- Phase 3: y = x * rstd * gamma ---
            col_loops = math.ceil(nca / VL)
            for j in range(col_loops):
                preg = update_mask(num_col - j * VL, elem_bits=32)[0]
                off = j * VL
                if const_expr(is_16bit):
                    xu = vload_unpack(x_buf, off, mode=UnpackMode.B16_TO_B32)
                    x = vcast(xu, dtypes.float32, mask=preg)
                else:
                    x = vload(x_buf, off)
                    x = vselect(x, zero, cond_mask=preg)
                mul1 = vmul(x, rstd_brc, mask=preg)
                g = vload(gamma_fp32, off)
                g = vselect(g, zero, cond_mask=preg)
                yval = vmul(mul1, g, mask=preg)
                if const_expr(is_16bit):
                    vstore_pack(y_buf, off, vcast(yval, out_dtype, mask=preg), preg, mode=PackMode.B32_TO_B16)
                else:
                    vstore(y_buf, off, yval, preg)

    @jit
    def _load_gamma(self, gm_gamma):
        ct = self._num_col
        if const_expr(self._x_is_16bit):
            mem_copy(self._gamma_16_ch, tile_view(gm_gamma, (1, ct), (0, 0)))
            _cast_b16_to_fp32(
                self._gamma_16_ch, self._gamma_fp32, self._num_seg, ct,
            )
        else:
            mem_copy(
                self._gamma_fp32,
                tile_view(gm_gamma, (1, ct), (0, 0)),
            )
        return self._gamma_fp32

@kernel
class ColSplitKernel:
    def __init__(self, num_col, dtype=dtypes.bfloat16):
        self._num_col = int(num_col)
        self._dtype = dtype
        self._x_is_16bit = dtype in (dtypes.bfloat16, dtypes.float16)
        num_col_int = self._num_col
        elem_bytes = 2 if self._x_is_16bit else 4
        for cap in (COL_TILE_CAP_LARGE, COL_TILE_CAP):
            if num_col_int <= cap:
                self._col_tile = num_col_int
            else:
                tile = 1
                for d in range(cap, 0, -1):
                    if num_col_int % d == 0:
                        tile = d
                        break
                self._col_tile = tile
            self._num_tiles = self._num_col // self._col_tile
            self._num_seg = math.ceil(self._col_tile / VL)
            self._w = self._num_seg * VL
            w = self._w
            nt = self._num_tiles
            ub_need = RETAINED_SIZE_1K + w * elem_bytes * 4 + w * elem_bytes + w * 4 + VL * 4 + math.ceil(nt / VL) * VL * 4
            if ub_need <= UB_SIZE:
                break
        self._tile_squared_sum_width = math.ceil(self._num_tiles / VL) * VL
        buf_dtype = dtype if self._x_is_16bit else dtypes.float32
        if self._x_is_16bit:
            self._gamma_16_ch = Channel(
                MemLoc.UB, shape=(1, self._w), dtype=dtype, depth=1,
            )
            self._gamma_fp32 = Buffer(MemLoc.UB, (1, self._w), dtypes.float32)
        else:
            self._gamma_fp32 = Channel(
                MemLoc.UB, shape=(1, self._w), dtype=dtypes.float32, depth=1,
            )
        self._rstd_ch = Channel(MemLoc.UB, shape=(1, VL), dtype=dtypes.float32, depth=1)
        self._rstd_gm_ch = Channel(MemLoc.UB, shape=(1, 1), dtype=dtypes.float32, depth=1)
        self._tile_squared_sum_buffer = Buffer(MemLoc.UB, (1, self._tile_squared_sum_width), dtypes.float32)
        self._x_db_ch = Channel(MemLoc.UB, shape=(1, self._w), dtype=buf_dtype, depth=2)
        self._x_ch = Channel(MemLoc.UB, shape=(1, self._w), dtype=buf_dtype, depth=2)
        self._y_ch = Channel(MemLoc.UB, shape=(1, self._w), dtype=buf_dtype, depth=1)

    def __call__(self, gm_x: Tensor, gm_gamma: Tensor, gm_y: Tensor, gm_rstd: Tensor,
                 epsilon: dtypes.float32):
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
        re = rs + rpc
        if re > nr:
            re = nr

        if bi < lcn:
            for row in range(rs, re):
                mem_copy(self._x_db_ch, tile_view(gm_x, (1, ct), (row, 0)))

                for t in range(0, nt):
                    nxt = t + 1
                    if nxt < nt:
                        mem_copy(self._x_db_ch, tile_view(gm_x, (1, ct), (row, nxt)))
                    self._compute_x_squared_sum(
                        self._x_db_ch, self._tile_squared_sum_buffer, t,
                    )

                self._compute_rstd(
                    self._tile_squared_sum_buffer, self._rstd_ch, self._rstd_gm_ch, avg, epsilon,
                )
                mem_copy(tile_view(gm_rstd, (1, 1), (row, 0)),
                         local_slice(self._rstd_gm_ch, (1, 1), offset=0))

                for t in range(0, nt):
                    gamma_view = self._load_gamma(gm_gamma, t)
                    mem_copy(self._x_ch, tile_view(gm_x, (1, ct), (row, t)))
                    self._compute_y(
                        self._x_ch, gamma_view, self._y_ch,
                        self._rstd_ch,
                    )
                    mem_copy(tile_view(gm_y, (1, ct), (row, t)),
                             self._y_ch)

    @jit
    def _load_gamma(self, gm_gamma, t):
        ct = self._col_tile
        if const_expr(self._x_is_16bit):
            mem_copy(self._gamma_16_ch, tile_view(gm_gamma, (1, ct), (0, t)))
            _cast_b16_to_fp32(
                self._gamma_16_ch, self._gamma_fp32, self._num_seg, ct,
            )
        else:
            mem_copy(
                self._gamma_fp32,
                tile_view(gm_gamma, (1, ct), (0, t)),
            )
        return self._gamma_fp32

    @jit
    def _compute_y(self, x_buf, gamma_fp32, y_buf, rstd_ub):
        with vf(mode="raw"):
            full = full_mask(32)
            zero = vdup_scalar(0.0, dtypes.float32, mask=full)
            rstd_brc = vload_brc(rstd_ub, 0)
            for seg in cannbotdsl.range(self._num_seg, unroll=2):
                mask, _ = update_mask(self._col_tile - seg * VL, elem_bits=32)
                off = seg * VL
                if const_expr(self._x_is_16bit):
                    xu = vload_unpack(x_buf, off, mode=UnpackMode.B16_TO_B32)
                    xf = vcast(xu, dtypes.float32, mask=mask)
                else:
                    xf = vload(x_buf, off)
                    xf = vselect(xf, zero, cond_mask=mask)
                gf = vload(gamma_fp32, off)
                gf = vselect(gf, zero, cond_mask=mask)
                yr = vmul(xf, rstd_brc, mask=mask)
                yval = vmul(yr, gf, mask=mask)
                if const_expr(self._x_is_16bit):
                    vstore_pack(y_buf, off, vcast(yval, self._dtype, mask=mask), mask, mode=PackMode.B32_TO_B16)
                else:
                    vstore(y_buf, off, yval, mask)

    @jit
    def _compute_x_squared_sum(self, x_buf, tile_squared_sum_buffer, t):
        with vf(mode="raw"):
            full = full_mask(32)
            zero = vdup_scalar(0.0, dtypes.float32, mask=full)
            ct = self._col_tile
            acc = vdup_scalar(0.0, dtypes.float32, mask=full)
            for seg in range(0, self._num_seg):
                mask, _ = update_mask(ct - seg * VL, elem_bits=32)
                off = seg * VL
                if const_expr(self._x_is_16bit):
                    xu = vload_unpack(x_buf, off, mode=UnpackMode.B16_TO_B32)
                    xf = vcast(xu, dtypes.float32, mask=mask)
                else:
                    xf = vload(x_buf, off)
                    xf = vselect(xf, zero, cond_mask=mask)
                xf = vmul(xf, xf, mask=mask)
                acc = vadd(acc, vreduce_sum(xf, mask=mask), mask=full)
            vstore_first(tile_squared_sum_buffer, t, acc)

    @jit
    def _compute_rstd(self, tile_squared_sum_buffer, rstd_ub, rstd_gm, avg, epsilon):
        with vf(mode="raw"):
            full = full_mask(32)
            total = vdup_scalar(0.0, dtypes.float32, mask=full)
            num_tiles = self._num_tiles
            for i in range(math.ceil(num_tiles / VL)):
                pm, _ = update_mask(num_tiles, elem_bits=32)
                pv = vload(tile_squared_sum_buffer, i * VL)
                total = vadd(total, vreduce_sum(pv, mask=pm), mask=full)
            var = vmuls(total, avg, mask=full)
            var = vadds(var, epsilon, mask=full)
            var = vmaxs(var, -99.99, mask=full)
            one = vdup_scalar(1.0, dtypes.float32, mask=full)
            r = vdiv(one, var, mask=full)
            y = vsqrt(r, mask=full)
            t = vmuls(var, -0.5, mask=full)
            t = vmul(t, y, mask=full)
            t1 = vadd(vdup_scalar(1.5, dtypes.float32, mask=full), vmul(t, y, mask=full), mask=full)
            rstd = vmul(y, t1, mask=full)
            t3 = vmuls(var, -1.0, mask=full)
            s = vadd(one, vmul(t3, r, mask=full), mask=full)
            t4 = vmuls(rstd, -1.0, mask=full)
            r = vadd(r, vmul(t4, rstd, mask=full), mask=full)
            s = vadd(s, vmul(var, r, mask=full), mask=full)
            s = vmul(s, rstd, mask=full)
            rstd = vadd(rstd, vmul(s, vdup_scalar(0.5, dtypes.float32, mask=full), mask=full), mask=full)
            cmp_inf = vcmp_gt_scalar(var, _F32_MAX * 0.999, mask=full)
            rstd = vselect(vdup_scalar(0.0, dtypes.float32, mask=full), rstd, cond_mask=cmp_inf)
            cmp_pos = vcmp_gt_scalar(var, 0.0, mask=full)
            rstd = vselect(rstd, vdup_scalar(_F32_MAX, dtypes.float32, mask=full), cond_mask=cmp_pos)
            vstore(rstd_ub, 0, rstd, full)
            vstore(rstd_gm, 0, rstd, full)

#  Host
class RmsNorm:
    def __init__(self, dtype=dtypes.bfloat16):
        self.dtype = dtype

    @staticmethod
    def _is_row_full_load(num_col, dtype=dtypes.float32):
        nca = math.ceil(num_col / VL) * VL
        is_16bit = dtype in (dtypes.bfloat16, dtypes.float16)
        elem_bytes = 2 if is_16bit else 4
        gamma_bytes = nca * (elem_bytes + FLOAT_BYTE_SIZE) if is_16bit else nca * FLOAT_BYTE_SIZE
        if nca > FULL_LOAD_R_MAX:
            return False
        for depth in (DOUBLE_BUFFER_NUM, 1):
            ub_factor = (UB_SIZE - RETAINED_SIZE_1K - gamma_bytes) / \
                        (nca * elem_bytes * MULTI_FACTOR_2 * depth +
                         FLOAT_BYTE_SIZE * (depth + X_REDUCE_TMP_NUM))
            if ub_factor >= 1:
                return True
        return False

    @jit
    def run(self, gm_x, gm_gamma, gm_y, gm_rstd, eps: dtypes.float32):
        num_col = gm_x.shape[1]
        if const_expr(self._is_row_full_load(num_col, self.dtype)):
            op = RowFullLoadKernel(num_col, self.dtype)
        else:
            op = ColSplitKernel(num_col, self.dtype)
        op[DEFAULT_BLOCK_NUM](gm_x, gm_gamma, gm_y, gm_rstd, eps)

def rms_norm(x, gamma, *, epsilon=1e-6):
    assert x.dim() >= 1
    norm_rank = gamma.dim()
    assert 1 <= norm_rank <= x.dim()
    assert tuple(x.shape[-norm_rank:]) == tuple(gamma.shape)
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert gamma.dtype == x.dtype
    assert x.is_contiguous() and gamma.is_contiguous()
    assert x.device == gamma.device

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