# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""KvCompressEpilog 精度测试。

输出行布局：[rope bf16 128B][nope fp8 (d-64)B][scale][pad]
对比 cache 的前 kvCacheCol 字节。
"""

import logging
import math
import os
import struct
import sys
from collections import namedtuple

import numpy as np
import pytest
import torch

os.environ.setdefault("CANNBOTDSL_AUTO_BUFID_SYNC", "0")
os.environ.setdefault("CANNBOTDSL_AUTO_INTRABLOCKSYNC", "0")

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "samples", "kv_compress_epilog"
    ),
)

from kv_compress_epilog import kv_compress_epilog, _calc_kv_cache_col  # noqa: E402

LOG = logging.getLogger(__name__)


FP8_E4M3FN_MAX = 448.0
FP8_E4M3FN_MIN = -448.0
ROPE_COLS = 64
ROPE_BYTES = ROPE_COLS * 2
GROUP_SIZE = 64
BLOCK_BYTES = 32


def _ceil_div(a, b):
    return (a + b - 1) // b


def _round_up(a, b):
    return _ceil_div(a, b) * b


def _fp32_to_bf16_bits(val):
    bits = struct.unpack("<I", struct.pack("<f", float(val)))[0]
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return (rounded >> 16) & 0xFFFF


def _fp32_to_fp8_e4m3fn(val):
    if np.isnan(val):
        return 0x7F
    if val >= FP8_E4M3FN_MAX:
        return 126
    if val <= FP8_E4M3FN_MIN:
        return 254
    if val == 0.0:
        return 0
    sign = 0
    abs_val = abs(val)
    if val < 0:
        sign = 0x80
    if abs_val < 2**-9:
        return sign
    exp = int(np.floor(np.log2(abs_val)))
    if exp < -9:
        return sign
    if exp > 7:
        return sign | 0x7E
    mantissa_bits = 3
    if exp < -6:
        sub = abs_val / (2**-9)
        m = int(np.round(sub))
        if m == 0:
            return sign
        return sign | (m & 0x07)
    else:
        e = exp + 7
        frac = abs_val / (2.0**exp) - 1.0
        m = int(np.round(frac * (2**mantissa_bits)))
        if m >= (2**mantissa_bits):
            e += 1
            m = 0
        if e > 15:
            return sign | 0x7E
        return sign | ((e & 0x0F) << 3) | (m & 0x07)


def _fp32_scale_to_e8m0(scale_fp32):
    bits = struct.unpack("<I", struct.pack("<f", float(scale_fp32)))[0]
    exp_bits = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if mantissa != 0:
        exp_bits = (exp_bits + 1) & 0xFF
    return exp_bits


# ---------------------------------------------------------------------------
# Golden kit
# ---------------------------------------------------------------------------


def _make_inputs(bs, d, block_num, block_size, head_dim, *, seed=42, skip_ratio=0.0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(bs, d, generator=gen, dtype=torch.float32) * 3.0
    x = x.to(torch.bfloat16)

    slot_mapping = torch.arange(bs, dtype=torch.int32)
    if skip_ratio > 0:
        num_skip = int(bs * skip_ratio)
        skip_idx = torch.randperm(bs, generator=gen)[:num_skip]
        slot_mapping[skip_idx] = -1
    valid_mask = slot_mapping >= 0
    num_valid = int(valid_mask.sum())
    if num_valid > block_num * block_size:
        num_valid = block_num * block_size
    slot_mapping[valid_mask] = torch.randperm(block_num * block_size, generator=gen)[
        :num_valid
    ].to(torch.int32)

    cache = torch.zeros(block_num * block_size * head_dim, dtype=torch.uint8)
    cache = cache.view(block_num, block_size, 1, head_dim)
    return cache, x, slot_mapping


_GoldenCtx = namedtuple(
    "_GoldenCtx",
    [
        "d_nope",
        "num_groups",
        "quant_group_size",
        "scale_bytes",
        "quant_mode",
        "round_scale",
        "kv_cache_col",
    ],
)


def _golden_group(ctx, x_bf16_f32, g, row_out):
    qs = ctx.quant_group_size
    g_start = g * qs
    g_end = g_start + qs
    group = x_bf16_f32[g_start:g_end]
    amax = np.max(np.abs(group))
    safe_amax = max(amax, 1e-4)
    scale = safe_amax / FP8_E4M3FN_MAX

    if ctx.round_scale:
        scale = 2.0 ** math.ceil(math.log2(scale)) if scale > 0 else 0.0

    if scale > 0:
        q_vals = x_bf16_f32[g_start:g_end] / scale
    else:
        q_vals = np.zeros(qs)
    q_tensor = torch.from_numpy(q_vals.astype(np.float32)).to(torch.float8_e4m3fn)
    fp8_bytes = q_tensor.view(torch.int8).numpy().astype(np.uint8)
    nope_off = ROPE_BYTES + g * qs
    nope_end = nope_off + qs
    row_out[nope_off:nope_end] = fp8_bytes

    scale_off = ROPE_BYTES + ctx.d_nope + g * ctx.scale_bytes
    if ctx.quant_mode == 0:
        scale_bf16 = torch.tensor([scale], dtype=torch.bfloat16)
        scale_end = scale_off + 2
        row_out[scale_off:scale_end] = scale_bf16.view(torch.uint8).numpy()
    else:
        e8m0 = _fp32_scale_to_e8m0(scale) if scale > 0 else 0
        row_out[scale_off] = e8m0


def _golden_row(ctx, x_bf16_f32):
    row_out = np.zeros(ctx.kv_cache_col, dtype=np.uint8)
    d_nope = ctx.d_nope
    rope_data = x_bf16_f32[d_nope:]
    rope_bf16 = np.zeros(ROPE_COLS, dtype=np.uint16)
    for j in range(ROPE_COLS):
        rope_bf16[j] = _fp32_to_bf16_bits(rope_data[j])
    row_out[:ROPE_BYTES] = rope_bf16.view(np.uint8)
    for g in range(ctx.num_groups):
        _golden_group(ctx, x_bf16_f32, g, row_out)
    return row_out


def _kv_compress_epilog_golden(
    cache, x, slot_mapping, *, quant_group_size=64, quant_mode=1, round_scale=True
):
    cache = cache.clone()
    x_np = x.to(torch.float32).cpu().numpy()
    slot_np = slot_mapping.cpu().numpy().astype(np.int64)

    bs, d = x_np.shape
    d_nope = d - ROPE_COLS
    num_groups = d_nope // quant_group_size
    head_dim = cache.shape[3]
    kv_cache_col = _calc_kv_cache_col(d, quant_mode, quant_group_size)
    scale_bytes = 2 if quant_mode == 0 else 1
    ctx = _GoldenCtx(
        d_nope=d_nope,
        num_groups=num_groups,
        quant_group_size=quant_group_size,
        scale_bytes=scale_bytes,
        quant_mode=quant_mode,
        round_scale=round_scale,
        kv_cache_col=kv_cache_col,
    )

    cache_np = cache.view(-1, head_dim).numpy()

    for i in range(bs):
        slot = int(slot_np[i])
        if slot == -1:
            continue
        x_bf16_f32 = (
            torch.from_numpy(x_np[i]).to(torch.bfloat16).to(torch.float32).numpy()
        )
        cache_np[slot, :kv_cache_col] = _golden_row(ctx, x_bf16_f32)

    return torch.from_numpy(cache_np).view(cache.shape[0], cache.shape[1], 1, head_dim)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


# 8 个用例：quant_mode(2) × round_scale(2) × 2 种 shape，覆盖各功能组合
_Case = namedtuple(
    "_Case",
    [
        "bs",
        "d",
        "block_num",
        "block_size",
        "head_dim",
        "skip_ratio",
        "round_scale",
        "quant_mode",
        "seed",
    ],
)

_CASES = [
    pytest.param(_Case(4, 128, 1, 8, 256, 0.0, True, 1, 42), id="e8m0_rs_bs4_d128"),
    pytest.param(_Case(4, 128, 1, 8, 256, 0.0, False, 1, 42), id="e8m0_nors_bs4_d128"),
    pytest.param(_Case(4, 128, 1, 8, 256, 0.0, True, 0, 42), id="bf16_rs_bs4_d128"),
    pytest.param(_Case(4, 128, 1, 8, 256, 0.0, False, 0, 42), id="bf16_nors_bs4_d128"),
    pytest.param(
        _Case(256, 512, 1, 256, 608, 0.0, True, 1, 42), id="e8m0_rs_bs256_d512"
    ),
    pytest.param(
        _Case(256, 512, 1, 256, 608, 0.0, False, 1, 42), id="e8m0_nors_bs256_d512"
    ),
    pytest.param(
        _Case(256, 512, 1, 256, 608, 0.0, True, 0, 42), id="bf16_rs_bs256_d512"
    ),
    pytest.param(
        _Case(256, 512, 1, 256, 608, 0.0, False, 0, 42), id="bf16_nors_bs256_d512"
    ),
]


# ---------------------------------------------------------------------------
# NPU accuracy tests
# ---------------------------------------------------------------------------


def _assert_rows_match(case, out_np, golden_np, slot_np, kv_cache_col):
    max_diff = 0
    mismatch_count = 0
    total_checked = 0
    for i in range(case.bs):
        slot = int(slot_np[i])
        if slot == -1:
            continue
        out_row = out_np[slot, :kv_cache_col]
        golden_row = golden_np[slot, :kv_cache_col]
        diff = np.abs(out_row.astype(np.int32) - golden_row.astype(np.int32))
        if diff.size:
            max_diff = max(max_diff, int(diff.max()))
        if case.round_scale:
            # round_scale=True：与 golden 逐字节完全一致
            mismatch_count += np.sum(diff > 0)
        else:
            # round_scale=False：接受 ±1 字节容差（FP8 舍入方向差异）
            mismatch_count += np.sum(diff > 1)
        total_checked += len(out_row)

    assert mismatch_count == 0, (
        f"mismatch={mismatch_count}/{total_checked} max_diff={max_diff} "
        f"for bs={case.bs}, d={case.d}, quant_mode={case.quant_mode}, "
        f"round_scale={case.round_scale}"
    )
    LOG.info(
        "kv_compress_epilog (bs=%s, d=%s, qm=%s, rs=%s) "
        "mismatch=%s/%s max_diff=%s finished!",
        case.bs,
        case.d,
        case.quant_mode,
        case.round_scale,
        mismatch_count,
        total_checked,
        max_diff,
    )


@pytest.mark.npu
@pytest.mark.parametrize("case", _CASES)
def test_kv_compress_epilog_npu(case):
    pytest.importorskip("torch_npu")

    cache, x, slot_mapping = _make_inputs(
        bs=case.bs,
        d=case.d,
        block_num=case.block_num,
        block_size=case.block_size,
        head_dim=case.head_dim,
        skip_ratio=case.skip_ratio,
        seed=case.seed,
    )

    golden = _kv_compress_epilog_golden(
        cache,
        x,
        slot_mapping,
        quant_group_size=64,
        quant_mode=case.quant_mode,
        round_scale=case.round_scale,
    )

    cache_out = kv_compress_epilog(
        cache.npu(),
        x.npu(),
        slot_mapping.npu(),
        quant_group_size=64,
        quant_mode=case.quant_mode,
        round_scale=case.round_scale,
    )
    torch.npu.synchronize()

    kv_cache_col = _calc_kv_cache_col(case.d, case.quant_mode)
    out_np = cache_out.cpu().view(-1, case.head_dim).numpy()
    golden_np = golden.view(-1, case.head_dim).numpy()
    slot_np = slot_mapping.cpu().numpy()
    _assert_rows_match(case, out_np, golden_np, slot_np, kv_cache_col)
