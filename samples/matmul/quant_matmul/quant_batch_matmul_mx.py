# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend950 MX ``npu_quant_matmul`` sample.

Structure:
  1. Host-side ASWT tiling
  2. MX kernel with GM->L1->L0->L0C->GM data flow
  3. Torch-facing ``npu_quant_matmul`` wrapper and issue reproduction helpers

Formula: Y[M,N] = dequant(X1[M,K]) @ dequant(X2[K,N])
MX scale ABI: x1Scale uses ScaleAND[M,G,2], x2Scale uses ScaleBND[G,N,2],
where G=ceil(K/64). Transpose is represented by stride metadata.
"""

__all__ = ["npu_quant_matmul"]

import functools
from typing import NamedTuple

import cannbotdsl
import torch

from cannbotdsl import dtypes, TensorSpec, get_mem_size, get_platform_info
from cannbotdsl.ops.arch import get_block_idx
from cannbotdsl.channel import Channel
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.tensor import MemLoc
from cannbotdsl.tensor import Tensor
from cannbotdsl.tensor import tile_slice
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy


L1_CAPACITY_BYTES = get_mem_size("l1")
L0A_CAPACITY_BYTES = get_mem_size("l0a")
L0B_CAPACITY_BYTES = get_mem_size("l0b")
L0C_CAPACITY_BYTES = get_mem_size("l0c")
MX_GROUP_SIZE = 32  # One E8M0 scale is shared by every 32 K elements.
MX_K_ALIGN = 64  # Every 64 K elements map to one paired-scale group.
MX_SCALE_PAIR = 2
WINDOW_LEN = 4  # Number of output M tiles in one ASW window.
ESTIMATED_SCALE_K = 4096  # Initial scaleKL1 used by the mainline L1 model.
MX_L0C_PINGPONG_SCALE_K_L1_TARGET = 2048
MX_L0C_PINGPONG_OUTPUT_SIZE_LIMIT = 128 * 1024 * 1024
L2_CACHE_THRESHOLD_BYTES = 128 * 1024 * 1024 * 80 // 100
MMAD_BLOCK_SIZE = 256
CUBE_BLOCK = 16
AIC_NUM = get_platform_info().cube_core_num
BASIC_BLOCK_SIZE_128 = 128
LOAD_BALANCE_BASE_N_128_ALIGN_K_THRESHOLD = 2560
LOAD_BALANCE_THRESHOLD = 1792
MTE2_ADDRESS_ALIGN_SIZE = 128
# Floating-point tolerance used by tiling score comparisons.
SCORE_COMPARE_EPS = 1e-12
BASE_K_LIMIT = 4095


class _TilingProfile(NamedTuple):
    """Per-dtype tiling knobs shared by every host-side sizing helper.

    ``element_bytes`` scales all L1/L0 byte models; ``l1_align`` and
    ``l2_align`` mirror the mainline inner-axis alignment for the dtype
    (32/128 elements for FP8, 64/256 for packed FP4 where one element is
    half a byte); ``cube_throughput``/``hbm_bandwidth`` feed the
    cube-vs-transfer bound estimates.
    """

    element_bytes: float
    l1_align_size: int
    l2_align_size: int
    cube_throughput: float
    hbm_bandwidth: float
    l2_bandwidth: float


FP8_TILING_PROFILE = _TilingProfile(
    element_bytes=1.0,
    l1_align_size=32,
    l2_align_size=128,
    cube_throughput=864.0,
    hbm_bandwidth=1.4,
    l2_bandwidth=5.2,
)
FP4_TILING_PROFILE = _TilingProfile(
    element_bytes=0.5,  # fp4x2 packs two E2M1 elements per byte.
    l1_align_size=64,
    l2_align_size=256,
    cube_throughput=1730.0,
    hbm_bandwidth=1.6,
    l2_bandwidth=5.2,
)

# Legacy FP8 aliases kept for backwards compatibility.
HBM_BANDWIDTH = FP8_TILING_PROFILE.hbm_bandwidth
L2_BANDWIDTH = FP8_TILING_PROFILE.l2_bandwidth
MX_CUBE_THROUGHPUT = FP8_TILING_PROFILE.cube_throughput
L1_ALIGN_SIZE = FP8_TILING_PROFILE.l1_align_size
L2_ALIGN_SIZE = FP8_TILING_PROFILE.l2_align_size

# ============================================================================
# 1. Host-side Tiling
# ============================================================================


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def ceil_align(value, divisor):
    return ceil_div(value, divisor) * divisor


def floor_align(value, divisor):
    return value // divisor * divisor


def get_scale_k_len(k):
    """Return the contiguous L1 scale length for a data K extent."""
    return ceil_div(k, MX_K_ALIGN) * MX_SCALE_PAIR


def _optimize_base_block_for_load_balance(
    m,
    n,
    k,
    base_m,
    base_n,
    base_m_align=CUBE_BLOCK,
    base_n_align=CUBE_BLOCK,
):
    """Search ``baseM/baseN`` for better load balance in two or three rounds.

    The search may create more output tiles, but it never increases the round
    count or reduces the number of active cores in the final round. Return the
    selected ``(baseM, baseN)``.
    """
    origin_base_m, origin_base_n = base_m, base_n
    round_limit = ceil_div(ceil_div(m, base_m) * ceil_div(n, base_n), AIC_NUM)
    if round_limit == 1 or round_limit > 3:
        return base_m, base_n

    origin_blocks = ceil_div(m, base_m) * ceil_div(n, base_n)
    origin_tail_blocks = origin_blocks % AIC_NUM or AIC_NUM
    origin_tail_used_cores = origin_tail_blocks * (AIC_NUM // origin_tail_blocks)
    origin_memory_score = (base_m + base_n) / (base_m * base_n)

    best_m, best_n = base_m, base_n
    best_balance = 0.0
    best_memory_score = float("inf")
    found = False
    for candidate_m in range(ceil_align(base_m, base_m_align), 0, -base_m_align):
        for candidate_n in range(ceil_align(base_n, base_n_align), 0, -base_n_align):
            blocks = ceil_div(m, candidate_m) * ceil_div(n, candidate_n)
            rounds = ceil_div(blocks, AIC_NUM)
            if rounds > round_limit:
                break
            if candidate_m == base_m and candidate_n == base_n:
                continue
            tail_blocks = blocks % AIC_NUM or AIC_NUM
            memory_score = (candidate_m + candidate_n) / (candidate_m * candidate_n)
            if tail_blocks < origin_tail_used_cores:
                continue
            if (
                tail_blocks == origin_tail_used_cores
                and memory_score > origin_memory_score + SCORE_COMPARE_EPS
            ):
                continue
            # Fraction of scheduled tile capacity that contains valid output.
            balance = m * n / AIC_NUM / (rounds * candidate_m * candidate_n)
            tie_break = (
                abs(balance - best_balance) <= SCORE_COMPARE_EPS
                and memory_score < best_memory_score
            )
            if not found or balance > best_balance + SCORE_COMPARE_EPS or tie_break:
                best_m, best_n = candidate_m, candidate_n
                best_balance = balance
                best_memory_score = memory_score
                found = True
    if not found:
        return base_m, base_n
    if (
        best_n % BASIC_BLOCK_SIZE_128 != 0
        and k < LOAD_BALANCE_BASE_N_128_ALIGN_K_THRESHOLD
    ):
        return origin_base_m, origin_base_n
    if best_m > m:
        best_m = ceil_align(m, base_m_align)
    if best_n > n:
        best_n = ceil_align(n, base_n_align)
    return best_m, best_n


def _try_swap_base_mn_for_mx_false_true(m, n, base_m, base_n):
    """Try swapping ``baseM/baseN`` for a false/true single-round shape.

    Swap only when baseN is not 128-aligned, the swapped shape remains a
    single-round launch, core usage does not decrease, and the core grid is
    better balanced. Return the original pair when these conditions fail.
    """
    if base_n % BASIC_BLOCK_SIZE_128 == 0:
        return base_m, base_n
    swap_base_m, swap_base_n = base_n, base_m
    swap_round = ceil_div(ceil_div(m, swap_base_m) * ceil_div(n, swap_base_n), AIC_NUM)
    if swap_round != 1:
        return base_m, base_n

    cur_m_core = ceil_div(m, base_m)
    cur_n_core = ceil_div(n, base_n)
    swap_m_core = ceil_div(m, swap_base_m)
    swap_n_core = ceil_div(n, swap_base_n)
    cur_used_core = cur_m_core * cur_n_core
    swap_used_core = swap_m_core * swap_n_core
    # A core-grid aspect ratio closer to one is better balanced.
    cur_core_shape_score = max(cur_m_core, cur_n_core) / min(cur_m_core, cur_n_core)
    swap_core_shape_score = max(swap_m_core, swap_n_core) / min(
        swap_m_core, swap_n_core
    )
    swap_core_shape_better = (
        swap_core_shape_score < cur_core_shape_score - SCORE_COMPARE_EPS
    )
    swap_base_n_is_friendly = (
        abs(swap_core_shape_score - cur_core_shape_score) <= SCORE_COMPARE_EPS
        and swap_base_n % BASIC_BLOCK_SIZE_128 == 0
    )
    if swap_used_core >= cur_used_core and (
        swap_core_shape_better or swap_base_n_is_friendly
    ):
        return swap_base_m, swap_base_n
    return base_m, base_n


def _get_base_block(
    m, n, k, transpose_a=False, transpose_b=True, profile=FP8_TILING_PROFILE
):
    """Select the per-AIC ``baseM/baseN/baseK`` compute block.

    Start from at most 256x256x128 and use transpose-specific alignment. For a
    multi-round launch, optimize final-round balance. For a single-round
    launch, redistribute the core grid and improve the tile aspect ratio.
    Finally, limit baseK by the double-buffered L0A/L0B capacity.
    """
    base_m_align = profile.l1_align_size if transpose_a else CUBE_BLOCK
    base_n_align = CUBE_BLOCK if transpose_b else profile.l1_align_size
    base_k_align = (
        MX_K_ALIGN if (transpose_a or not transpose_b) else profile.l2_align_size
    )
    adjust_base_m_align = profile.l2_align_size if transpose_a else CUBE_BLOCK
    adjust_base_n_align = CUBE_BLOCK if transpose_b else profile.l2_align_size
    # Start from at most 256x256x128; once all AICs are used, tune only the final round.
    base_m = ceil_align(min(m, 256), base_m_align)
    base_n = ceil_align(min(n, 256), base_n_align)
    base_k = ceil_align(min(k, 128), MX_K_ALIGN)
    if ceil_div(m, base_m) * ceil_div(n, base_n) >= AIC_NUM:
        base_m, base_n = _optimize_base_block_for_load_balance(
            m,
            n,
            k,
            base_m,
            base_n,
            adjust_base_m_align,
            adjust_base_n_align,
        )
        return base_m, base_n, base_k

    # Single-round case: keep the less divisible axis and assign the remaining
    # cores to the other axis so that as many available AICs run as possible.
    m_max_tile = ceil_div(m, adjust_base_m_align)
    n_max_tile = ceil_div(n, adjust_base_n_align)
    m_core = ceil_div(m, base_m)
    n_core = ceil_div(n, base_n)
    if m_max_tile < n_max_tile or (
        m_max_tile == n_max_tile and adjust_base_n_align == CUBE_BLOCK
    ):
        base_m = ceil_align(ceil_div(m, m_core), adjust_base_m_align)
        m_core = ceil_div(m, base_m)
        n_core = AIC_NUM // m_core
        base_n = ceil_align(ceil_div(n, n_core), adjust_base_n_align)
    else:
        base_n = ceil_align(ceil_div(n, n_core), adjust_base_n_align)
        n_core = ceil_div(n, base_n)
        m_core = AIC_NUM // n_core
        base_m = ceil_align(ceil_div(m, m_core), adjust_base_m_align)

    # Move baseM/baseN toward a square tile. This lowers the A/B transfer ratio
    # and may allow a larger baseK under the double-buffered L0 capacity limit.
    # When baseN >= 2*baseM, double n_core and derive m_core from BLOCK_DIM.
    while (
        base_n >= base_m * 2 and n_core < AIC_NUM // 2 and base_n != adjust_base_n_align
    ):
        n_core *= 2
        m_core = AIC_NUM // n_core
        base_m = ceil_align(ceil_div(m, m_core), adjust_base_m_align)
        base_n = ceil_align(ceil_div(n, n_core), adjust_base_n_align)
        m_core = ceil_div(m, base_m)
        n_core = ceil_div(n, base_n)
    # Apply the symmetric adjustment when the M direction is too long.
    while (
        base_m >= base_n * 2 and m_core < AIC_NUM // 2 and base_m != adjust_base_m_align
    ):
        m_core *= 2
        n_core = AIC_NUM // m_core
        base_m = ceil_align(ceil_div(m, m_core), adjust_base_m_align)
        base_n = ceil_align(ceil_div(n, n_core), adjust_base_n_align)
        m_core = ceil_div(m, base_m)
        n_core = ceil_div(n, base_n)

    if not transpose_a and transpose_b:
        base_m, base_n = _try_swap_base_mn_for_mx_false_true(m, n, base_m, base_n)
    # L0A and L0B each hold two baseK tiles. Derive the baseK limit from their
    # 64 KiB capacity, align it down, and keep it within K and BASE_K_LIMIT.
    # Packed FP4 transpose lowering doubles the outer extent in L0A/L0B.
    elem = profile.element_bytes
    l0a_m = base_m * (2 if (elem != 1.0 and transpose_a) else 1)
    l0b_n = base_n * (2 if (elem != 1.0 and not transpose_b) else 1)
    max_base_k = int(
        min(
            L0A_CAPACITY_BYTES // 2 / l0a_m / elem,
            L0B_CAPACITY_BYTES // 2 / l0b_n / elem,
        )
    )
    max_base_k = floor_align(max_base_k, base_k_align)
    if max_base_k >= base_k_align:
        base_k = min(ceil_align(k, base_k_align), max_base_k)
        if base_k > BASE_K_LIMIT:
            base_k = ceil_align(base_k // 2, base_k_align)
    base_m, base_n = _optimize_base_block_for_load_balance(
        m,
        n,
        k,
        base_m,
        base_n,
        adjust_base_m_align,
        adjust_base_n_align,
    )
    return base_m, base_n, base_k


def _get_l1_used_bytes(
    base_m,
    base_n,
    k,
    k_l1,
    scale_k_l1,
    buffers,
    full_load,
    has_bias=False,
    profile=FP8_TILING_PROFILE,
):
    """Return the L1 bytes used by A/B data, their scales, and optional bias."""
    elem = profile.element_bytes
    a_k_extent = ceil_align(k, MX_K_ALIGN) if full_load else k_l1
    a_scale_k_extent = k if full_load else scale_k_l1
    a_buffers = 1 if full_load else buffers
    a_scale_buffers = 1 if full_load else 2
    a_bytes = base_m * a_k_extent * a_buffers * elem
    a_scale_bytes = base_m * get_scale_k_len(a_scale_k_extent) * a_scale_buffers
    b_bytes = base_n * k_l1 * buffers * elem
    b_scale_bytes = base_n * get_scale_k_len(scale_k_l1) * 2
    bias_bytes = base_n * 4 * 2 if has_bias else 0
    return a_bytes + b_bytes + a_scale_bytes + b_scale_bytes + bias_bytes


def _cal_normal_l1_tiling(
    base_m, base_n, k, base_k, has_bias=False, profile=FP8_TILING_PROFILE
):
    """Calculate ``kL1`` and ``scaleKL1`` for NORMAL_MODE."""
    elem = profile.element_bytes
    bias_l1 = base_n * 4 * 2 if has_bias else 0
    available_l1_size = L1_CAPACITY_BYTES - bias_l1
    base_ab_size = (base_m + base_n) * base_k * elem
    scale_base_size = base_m + base_n
    scale_k_l1 = min(k, ESTIMATED_SCALE_K)
    scale_size = scale_base_size * ceil_div(scale_k_l1, MX_K_ALIGN) * MX_SCALE_PAIR * 2

    # Estimate the largest stepK that fits double-buffered data and scales.
    depth = 1
    while depth * base_ab_size + scale_size <= available_l1_size:
        depth *= 2
        k_l1 = depth // 2 * base_k
        if k_l1 > scale_k_l1:
            scale_k_l1 = k_l1
            scale_size = (
                scale_base_size * ceil_div(scale_k_l1, MX_K_ALIGN) * MX_SCALE_PAIR * 2
            )
    depth = depth if depth == 1 else depth // 2

    # kL1 = stepK * baseK; NORMAL_MODE limits stepK to four.
    step_k = min(max(1, depth // 2), ceil_div(k, base_k), 4)
    k_l1 = step_k * base_k
    # Use the remaining L1 for scales and keep scaleKL1 a multiple of kL1.
    used_data_size = 2 * k_l1 * (base_m + base_n) * elem
    left_l1_size = max(0, available_l1_size - used_data_size)
    scale_group_size = (base_m + base_n) * MX_SCALE_PAIR * 2
    max_scale_k_l1 = left_l1_size // scale_group_size * MX_K_ALIGN
    scale_k_l1 = min(max_scale_k_l1, ceil_align(k, k_l1))
    scale_k_l1 = scale_k_l1 // k_l1 * k_l1
    return int(k_l1), int(max(k_l1, scale_k_l1))


def _cal_a_full_l1_tiling(
    base_m, base_n, k, base_k, has_bias=False, profile=FP8_TILING_PROFILE
):
    """Calculate K windows for AL1_FULL_LOAD with resident A and ScaleA."""
    elem = profile.element_bytes
    bias_l1 = base_n * 4 * 2 if has_bias else 0
    # Reserve resident A/ScaleA storage; B and ScaleB use the remaining L1.
    a_full_size = base_m * ceil_align(k, MX_K_ALIGN) * elem
    a_scale_size = base_m * get_scale_k_len(k)
    left_l1_size = max(
        0,
        L1_CAPACITY_BYTES - a_full_size - a_scale_size - bias_l1,
    )

    # Meet the 128-byte B transfer granularity before expanding stepK.
    step_k_base = max(1, ceil_div(128, base_k))
    base_b_with_scale = base_n * (base_k * elem + get_scale_k_len(base_k)) * step_k_base
    if left_l1_size >= 64 * 1024:
        step_k_scale = ceil_div(32 * 1024, base_b_with_scale)
    else:
        step_k_scale = ceil_div(left_l1_size // 2, base_b_with_scale)
    step_k = step_k_base * max(1, int(step_k_scale))
    # Merge at least two baseK blocks when another K window remains and fits.
    if step_k == 1 and k > base_k and left_l1_size > base_b_with_scale * 2:
        step_k = 2
    k_l1 = step_k * base_k
    # Use the space left by double-buffered B to increase ScaleB reuse.
    b_data_size = 2 * base_n * k_l1 * elem
    left_scale_size = max(0, left_l1_size - b_data_size)
    base_scale_b_size = base_n * get_scale_k_len(base_k)
    max_factor_by_k = max(1, min(127, k // k_l1))
    max_factor_by_l1 = min(64 * 1024, left_scale_size) // max(
        1, base_scale_b_size * step_k * 2
    )

    # Group ScaleB transfers to 128 bytes; do not reuse across kL1 if it cannot fit.
    scale_factor_base = max(
        1,
        ceil_div(128, get_scale_k_len(base_k)),
    )
    if scale_factor_base <= max_factor_by_k and max_factor_by_l1 >= scale_factor_base:
        scale_factor = min(
            max_factor_by_l1 // scale_factor_base * scale_factor_base,
            max_factor_by_k,
        )
    else:
        scale_factor = 1
    return int(k_l1), int(scale_factor * k_l1)


def _can_use_four_buffer(
    base_m,
    base_n,
    k,
    k_l1,
    scale_k_l1,
    full_load,
    has_bias=False,
    profile=FP8_TILING_PROFILE,
):
    """Return whether four A/B buffers and double-buffered scales fit in L1."""
    return (
        _get_l1_used_bytes(
            base_m,
            base_n,
            k,
            k_l1,
            scale_k_l1,
            4,
            full_load,
            has_bias,
            profile=profile,
        )
        <= L1_CAPACITY_BYTES
    )


def _can_use_three_buffer(
    base_m,
    base_n,
    k,
    k_l1,
    scale_k_l1,
    full_load,
    has_bias=False,
    profile=FP8_TILING_PROFILE,
):
    """Return whether three A/B buffers and double-buffered scales fit in L1."""
    return (
        _get_l1_used_bytes(
            base_m,
            base_n,
            k,
            k_l1,
            scale_k_l1,
            3,
            full_load,
            has_bias,
            profile=profile,
        )
        <= L1_CAPACITY_BYTES
    )


# Bandwidth/frequency constants mirrored from the mainline tiling
# (qmmv3_tiling_const / MTE2_BW_UTILIZATION) for the MTE2-bound estimate.
_MAX_HBM_BW_TBPS = 4.0
_MAX_L2_BW_TBPS = 5.2
_MTE2_BW_UTILIZATION = 0.9
_BYTES_PER_US_PER_TBPS = 1_000_000.0
_MTE2_ADDRESS_ALIGN_BYTES = 128
_CUBE_BLOCK = 16
_MXFP_DIVISOR_SIZE = 64
_MX_CUBE_MACS_PER_CYCLE = 16 * 32 * 16
_CUBE_FREQ_MHZ = 1650.0


def _mx_full_k_load_bytes(outer_size, k, dtype_bytes):
    """Bytes of one full-K GM load for one outer row (data + E8M0 scales).

    Mirrors CalcMxFullKLoadSize: outer * ceilAlign(k, 64) data bytes plus
    outer * ceilDiv(k, 64) * 2 paired-scale bytes.
    """
    k_aligned = ceil_align(k, _MXFP_DIVISOR_SIZE)
    scale_k = ceil_div(k, _MXFP_DIVISOR_SIZE) * MX_SCALE_PAIR
    return outer_size * k_aligned * dtype_bytes + outer_size * scale_k


def _is_mx_mte2_bound(
    m,
    n,
    k,
    base_m,
    base_n,
    transpose_a,
    transpose_b,
    a_dtype_bytes,
    b_dtype_bytes,
    has_bias,
    used_core_num,
):
    """Estimate whether GM->L1 traffic (MTE2) dominates the cube time.

    Mirrors IsMxMte2Bound: unaligned inner axes force the bound, otherwise
    compare the estimated MTE2 time against the estimated cube time
    (batch=1, non-full-load GM/L2 traffic split).
    """
    if used_core_num == 0 or base_m == 0 or base_n == 0:
        return False
    a_inner = m if transpose_a else k
    b_inner = k if transpose_b else n
    inner_aligned = (
        a_inner * a_dtype_bytes % _MTE2_ADDRESS_ALIGN_BYTES == 0
        and b_inner * b_dtype_bytes % _MTE2_ADDRESS_ALIGN_BYTES == 0
    )
    if not inner_aligned:
        return True

    a_bytes = _mx_full_k_load_bytes(m, k, a_dtype_bytes)
    b_bytes = _mx_full_k_load_bytes(n, k, b_dtype_bytes)
    bias_bytes = n * 4 if has_bias else 0
    m_blocks = ceil_div(m, base_m)
    n_blocks = ceil_div(n, base_n)
    a_load_count = n_blocks
    b_load_count = m_blocks
    bias_load_count = m_blocks if has_bias else 0
    # One GM pass per operand, the remaining rounds are expected to hit L2.
    a_gm_count = min(a_load_count, 1)
    b_gm_count = min(b_load_count, 1)
    bias_gm_count = min(bias_load_count, 1)
    gm_bytes = a_bytes * a_gm_count + b_bytes * b_gm_count + bias_bytes * bias_gm_count
    l2_bytes = (
        a_bytes * (a_load_count - a_gm_count)
        + b_bytes * (b_load_count - b_gm_count)
        + bias_bytes * (bias_load_count - bias_gm_count)
    )
    mte2_us = gm_bytes / (
        _MAX_HBM_BW_TBPS * _MTE2_BW_UTILIZATION * _BYTES_PER_US_PER_TBPS
    ) + l2_bytes / (_MAX_L2_BW_TBPS * _MTE2_BW_UTILIZATION * _BYTES_PER_US_PER_TBPS)

    aligned_m = ceil_align(m, _CUBE_BLOCK)
    aligned_n = ceil_align(n, _CUBE_BLOCK)
    aligned_k = ceil_align(k, _MXFP_DIVISOR_SIZE)
    cube_us = (
        aligned_m
        * aligned_n
        * aligned_k
        / used_core_num
        / _MX_CUBE_MACS_PER_CYCLE
        / _CUBE_FREQ_MHZ
    )
    return mte2_us > cube_us


def _adjust_scale_k_l1_for_four_buffer(k, k_l1, scale_k_l1):
    """Shrink scaleKL1 for four buffers without adding a scale copy round."""
    if scale_k_l1 % ESTIMATED_SCALE_K == 0:
        return scale_k_l1
    half_k = ceil_div(k, 2)
    if half_k < scale_k_l1 < k:
        return min(scale_k_l1, ceil_align(half_k, k_l1))
    return scale_k_l1


def _get_full_cover_scale_k_l1_if_possible(
    fit_args,
    k_l1,
    scale_k_l1,
    full_load,
    has_bias=False,
    profile=FP8_TILING_PROFILE,
):
    """Expand scaleKL1 to full K when four buffers still fit in L1."""
    full_cover_scale_k_l1 = ceil_align(fit_args[2], k_l1)
    if full_cover_scale_k_l1 <= scale_k_l1:
        return scale_k_l1
    if _can_use_four_buffer(
        *fit_args,
        k_l1,
        full_cover_scale_k_l1,
        full_load,
        has_bias,
        profile=profile,
    ):
        return full_cover_scale_k_l1
    return scale_k_l1


def _cal_l1_tiling(
    m,
    n,
    k,
    base_m,
    base_n,
    base_k,
    full_load,
    transpose_a=False,
    transpose_b=True,
    has_bias=False,
    used_core_num=None,
    enable_l0c_pingpong=False,
    profile=FP8_TILING_PROFILE,
):
    """Return ``(kL1, scaleKL1, l1BufferNum)`` for the selected L1 mode.

    Mirrors the mainline ApplyL1Tiling decision chain: calculate the
    NORMAL_MODE or AL1_FULL_LOAD K windows, try four A/B buffers, then three
    when the double-buffer K coverage falls short and MTE2 is expected to
    dominate, and finally fall back to two.
    """
    cal_k_l1 = _cal_a_full_l1_tiling if full_load else _cal_normal_l1_tiling
    k_l1, scale_k_l1 = cal_k_l1(base_m, base_n, k, base_k, has_bias, profile=profile)
    if enable_l0c_pingpong:
        scale_k_l1 = _adjust_scale_k_l1_for_l0c_pingpong(m, n, k_l1, scale_k_l1)
    scale_k_l1 = _adjust_scale_k_l1_for_four_buffer(k, k_l1, scale_k_l1)
    fit_args = (base_m, base_n, k)
    if _can_use_four_buffer(
        *fit_args, k_l1, scale_k_l1, full_load, has_bias, profile=profile
    ):
        scale_k_l1 = _get_full_cover_scale_k_l1_if_possible(
            fit_args, k_l1, scale_k_l1, full_load, has_bias, profile=profile
        )
        return k_l1, scale_k_l1, 4

    # If stepK 3/4 cannot use four buffers, try aligned stepK 2 for multi-round K.
    step_k = k_l1 // base_k
    step_k_two_k_l1 = 2 * base_k
    step_two_aligned = k % 128 == 0 and step_k_two_k_l1 % 256 == 0
    can_reduce_step_k = step_k in (3, 4) and k_l1 * 2 < k and step_two_aligned
    step_k_two_scale_k_l1 = _adjust_scale_k_l1_for_four_buffer(
        k, step_k_two_k_l1, scale_k_l1
    )
    if can_reduce_step_k and _can_use_four_buffer(
        *fit_args,
        step_k_two_k_l1,
        step_k_two_scale_k_l1,
        full_load,
        has_bias,
        profile=profile,
    ):
        step_k_two_scale_k_l1 = _get_full_cover_scale_k_l1_if_possible(
            fit_args,
            step_k_two_k_l1,
            step_k_two_scale_k_l1,
            full_load,
            has_bias,
            profile=profile,
        )
        return step_k_two_k_l1, step_k_two_scale_k_l1, 4

    # A full-load uses the third buffer only for the B-side pipeline.
    # Otherwise, keep the current stepK and enable triple-buffer only when
    # the current double buffer cannot cover K and MTE2 is expected to
    # dominate (mirrors the mainline canUseCurrentThreeBuffer).
    can_use_three_buffer = full_load or (
        k_l1 * 2 < k
        and _is_mx_mte2_bound(
            m,
            n,
            k,
            base_m,
            base_n,
            transpose_a,
            transpose_b,
            profile.element_bytes,
            profile.element_bytes,
            has_bias,
            used_core_num,
        )
    )
    if can_use_three_buffer and _can_use_three_buffer(
        *fit_args, k_l1, scale_k_l1, full_load, has_bias, profile=profile
    ):
        return k_l1, scale_k_l1, 3
    if (
        can_reduce_step_k
        and can_use_three_buffer
        and _can_use_three_buffer(
            *fit_args,
            step_k_two_k_l1,
            step_k_two_scale_k_l1,
            full_load,
            has_bias,
            profile=profile,
        )
    ):
        return step_k_two_k_l1, step_k_two_scale_k_l1, 3

    return k_l1, scale_k_l1, 2


def _adjust_scale_k_l1_for_l0c_pingpong(
    m,
    n,
    k_l1,
    scale_k_l1,
):
    """Limit scaleKL1 to about 2048 K elements for L0C ping-pong.

    Keep the original window when the output is too large, L0C is not
    double-buffered, or scaleKL1 is already within the target.
    """
    output_size = m * n * 2
    if (
        output_size > MX_L0C_PINGPONG_OUTPUT_SIZE_LIMIT
        or scale_k_l1 <= MX_L0C_PINGPONG_SCALE_K_L1_TARGET
    ):
        return scale_k_l1
    scale_factor = max(
        1,
        MX_L0C_PINGPONG_SCALE_K_L1_TARGET // k_l1,
    )
    return max(k_l1, min(scale_k_l1, scale_factor * k_l1))


def _is_cube_bound(m, n, k, output_element_bytes, profile=FP8_TILING_PROFILE):
    """Estimate whether Cube compute time exceeds data transfer time."""
    elem = profile.element_bytes
    a_input_cost = m * k * elem
    b_input_cost = n * k * elem
    a_scale_cost = ceil_div(a_input_cost, MX_GROUP_SIZE)
    b_scale_cost = ceil_div(b_input_cost, MX_GROUP_SIZE)
    copy_out_bytes = m * n * output_element_bytes
    hbm_bytes = a_input_cost + a_scale_cost + b_input_cost + b_scale_cost
    a_l2_cost = m * k * elem * (ceil_div(n, MMAD_BLOCK_SIZE) - 1)
    b_l2_cost = n * k * elem * (ceil_div(m, MMAD_BLOCK_SIZE) - 1)
    l2_bytes = (
        a_l2_cost
        + ceil_div(a_l2_cost, MX_GROUP_SIZE)
        + b_l2_cost
        + ceil_div(b_l2_cost, MX_GROUP_SIZE)
    )
    l2_footprint = hbm_bytes + copy_out_bytes
    copy_out_bandwidth = (
        profile.l2_bandwidth
        if l2_footprint < L2_CACHE_THRESHOLD_BYTES
        else profile.hbm_bandwidth
    )
    transfer_cost = (
        hbm_bytes / profile.hbm_bandwidth
        + l2_bytes / profile.l2_bandwidth
        + copy_out_bytes / copy_out_bandwidth
    )
    compute_cost = 2.0 * m * n * k / profile.cube_throughput
    return compute_cost > transfer_cost


def _can_use_a_full_load(
    m,
    n,
    k,
    base_m,
    base_n,
    base_k,
    transpose_a=False,
    transpose_b=True,
    has_bias=False,
    profile=FP8_TILING_PROFILE,
):
    """Return whether the shape should use ``AL1_FULL_LOAD``.

    A and ScaleA must fit in half of L1, use fewer than four M tiles, and be
    reusable across N tiles. Enable full load directly with four buffers;
    otherwise require two buffers in NORMAL_MODE or more than 20 percent
    repeated A traffic.
    """
    elem = profile.element_bytes
    m_tiles = ceil_div(m, base_m)
    n_tiles = ceil_div(n, base_n)
    candidate = (
        base_m * ceil_align(k, MX_K_ALIGN) * elem <= L1_CAPACITY_BYTES // 2
        and m_tiles < WINDOW_LEN
        and AIC_NUM % m_tiles == 0
        and m_tiles * n_tiles > AIC_NUM
    )
    if not candidate:
        return False

    used_core_num = min(AIC_NUM, m_tiles * n_tiles)
    _, _, full_buffers = _cal_l1_tiling(
        m,
        n,
        k,
        base_m,
        base_n,
        base_k,
        True,
        transpose_a,
        transpose_b,
        has_bias,
        used_core_num,
        profile=profile,
    )
    if full_buffers > 2:
        return True

    _, _, normal_buffers = _cal_l1_tiling(
        m,
        n,
        k,
        base_m,
        base_n,
        base_k,
        False,
        transpose_a,
        transpose_b,
        has_bias,
        used_core_num,
        profile=profile,
    )
    repeated_a = (
        m * (ceil_align(k, MX_K_ALIGN) * elem + get_scale_k_len(k)) * (n_tiles - 1)
    )
    normal_bytes = (
        m * (ceil_align(k, MX_K_ALIGN) * elem + get_scale_k_len(k)) * n_tiles
        + n * (ceil_align(k, MX_K_ALIGN) * elem + get_scale_k_len(k)) * m_tiles
    )
    return normal_buffers == 2 or repeated_a / normal_bytes > 0.20


def _optimize_edge_basic_block(
    m,
    n,
    k,
    base_m,
    base_n,
    transpose_a=False,
    transpose_b=True,
):
    """Rebalance the final M/N blocks across adjacent outer-axis blocks."""
    m_tiles = ceil_div(m, base_m)
    n_tiles = ceil_div(n, base_n)
    m_base_tail_split_count = n_base_tail_split_count = 1
    m_tail_main = n_tail_main = 0
    if m_tiles == 1 or n_tiles == 1:
        return 1, 1, 0, 0

    inner_axis_aligned = k % MTE2_ADDRESS_ALIGN_SIZE == 0
    m_tail = m % base_m
    if (
        m_tail
        and not transpose_a
        and (inner_axis_aligned or m >= LOAD_BALANCE_THRESHOLD)
    ):
        merge_count_max = min((base_m - m_tail) // CUBE_BLOCK, m_tiles)
        window_size = min(WINDOW_LEN, m_tiles)
        main_window_count = m_tiles // window_size - 1
        tail_window_size = m_tiles - main_window_count * window_size
        best_cost = (main_window_count + 1) * base_m
        merged_window_count = 1
        for merge_len in range(tail_window_size - 1, merge_count_max, window_size):
            candidate = ceil_align(
                ceil_div(merge_len * base_m + m_tail, merge_len + 1),
                CUBE_BLOCK,
            )
            cost = (
                main_window_count + 1 - merged_window_count
            ) * base_m + merged_window_count * candidate
            if cost <= best_cost:
                best_cost = cost
                m_tail_main = candidate
                m_base_tail_split_count = merge_len + 1
            merged_window_count += 1

    n_tail = n % base_n
    balance_after_fixpipe = k < 1024 or (k == 1024 and n_tiles >= 8)
    is_n_inner_balanced = inner_axis_aligned or n >= LOAD_BALANCE_THRESHOLD
    if n_tail and transpose_b and not balance_after_fixpipe and is_n_inner_balanced:
        total_blocks = m_tiles * n_tiles
        total_rounds = ceil_div(total_blocks, AIC_NUM)
        main_rounds = ceil_div(
            (n_tiles - 1) * m_tiles + m_tiles % AIC_NUM,
            AIC_NUM,
        )
        is_aligned_window = m_tiles % AIC_NUM == 0 and (
            n_tiles % WINDOW_LEN == 0 or WINDOW_LEN % n_tiles == 0
        )
        if total_blocks <= AIC_NUM or is_aligned_window:
            main_rounds = total_rounds
        best_cost = main_rounds * base_n + (total_rounds - main_rounds) * n_tail
        merge_count_max = min((base_n - n_tail) // CUBE_BLOCK, n_tiles)
        for merge_len in range(1, merge_count_max):
            candidate = ceil_align(
                ceil_div(merge_len * base_n + n_tail, merge_len + 1),
                CUBE_BLOCK,
            )
            last = merge_len * (base_n - candidate) + n_tail
            new_main_rounds = 0
            new_tail_rounds = 0
            if merge_len < n_tiles - 1:
                new_main_rounds = ceil_div(
                    (n_tiles - 1 - merge_len) * m_tiles
                    + (merge_len + 1) * m_tiles % AIC_NUM,
                    AIC_NUM,
                )
            if merge_len:
                new_tail_rounds = min(
                    ceil_div(
                        merge_len * m_tiles + m_tiles % AIC_NUM,
                        AIC_NUM,
                    ),
                    total_rounds - new_main_rounds,
                )
            cost = (
                new_main_rounds * base_n
                + new_tail_rounds * candidate
                + (total_rounds - new_main_rounds - new_tail_rounds) * last
            )
            if cost < best_cost:
                best_cost = cost
                n_tail_main = candidate
                n_base_tail_split_count = merge_len + 1

    return (
        m_base_tail_split_count,
        n_base_tail_split_count,
        m_tail_main,
        n_tail_main,
    )


def _calc_tail_basic_block_split(
    m,
    n,
    base_m,
    base_n,
    aic_num,
    edge_tiling,
    transpose_a=False,
    transpose_b=True,
    profile=FP8_TILING_PROFILE,
):
    """Split each last-round block so idle AICs process its M/N sub-blocks."""
    m_tiles = ceil_div(m, base_m)
    n_tiles = ceil_div(n, base_n)
    tail_round = m_tiles * n_tiles % aic_num
    if tail_round == 0:
        return 1, 1

    m_tail = m - (m_tiles - 1) * base_m
    n_tail = n - (n_tiles - 1) * base_n
    m_base_tail_split_count, _, m_tail_main, _ = edge_tiling
    split_first_m = m_tail >= n_tail
    tile_max = aic_num // tail_round
    m_align = profile.l1_align_size if transpose_a else CUBE_BLOCK
    n_align = CUBE_BLOCK if transpose_b else profile.l1_align_size
    tail_base_m = m_tail_main if m_base_tail_split_count != 1 else base_m
    m_split_max = min(tile_max, ceil_div(min(m, tail_base_m), m_align))
    n_split_max = min(tile_max, ceil_div(min(n, base_n), n_align))
    first_max, second_max = (
        (m_split_max, n_split_max) if split_first_m else (n_split_max, m_split_max)
    )
    first_size, second_size = (
        (min(m, tail_base_m), min(n, base_n))
        if split_first_m
        else (min(n, base_n), min(m, tail_base_m))
    )
    first_inner_axis, second_inner_axis = (
        (transpose_a, not transpose_b)
        if split_first_m
        else (not transpose_b, transpose_a)
    )
    first_tail_size, second_tail_size = (
        (m_tail, n_tail) if split_first_m else (n_tail, m_tail)
    )
    first_packed_axis = profile.element_bytes == 0.5 and first_inner_axis
    second_packed_axis = profile.element_bytes == 0.5 and second_inner_axis
    single_window = m_tiles * n_tiles <= aic_num
    first_split = second_split = 1
    first_valid = second_valid = 1
    first_aligned = second_aligned = 1
    while (
        first_split < first_max
        and (first_split + 1) * second_split * tail_round <= aic_num
    ) or (
        second_split < second_max
        and first_split * (second_split + 1) * tail_round <= aic_num
    ):
        if (
            first_split < first_max
            and (first_split + 1) * second_split * tail_round <= aic_num
        ):
            first_split += 1
            first_byte_aligned = not first_packed_axis or (
                ceil_div(first_size, first_split) % 2 == 0
                and ceil_div(first_tail_size, first_split) % 2 == 0
            )
            if first_byte_aligned:
                first_valid = first_split
                if (
                    first_inner_axis
                    and single_window
                    and ceil_div(first_size, first_split) % profile.l1_align_size == 0
                ):
                    first_aligned = first_split
        if (
            second_split < second_max
            and first_split * (second_split + 1) * tail_round <= aic_num
        ):
            second_split += 1
            second_byte_aligned = not second_packed_axis or (
                ceil_div(second_size, second_split) % 2 == 0
                and ceil_div(second_tail_size, second_split) % 2 == 0
            )
            if second_byte_aligned:
                second_valid = second_split
                if (
                    second_inner_axis
                    and single_window
                    and ceil_div(second_size, second_split) % profile.l1_align_size == 0
                ):
                    second_aligned = second_split
    first_split = first_aligned if first_aligned != 1 else first_valid
    second_split = second_aligned if second_aligned != 1 else second_valid
    return (first_split, second_split) if split_first_m else (second_split, first_split)


class QbmmMxTiling(NamedTuple):
    """Host tiling data consumed by the block scheduler and kernel."""

    base_m: int
    base_n: int
    base_k: int
    k_l1: int
    scale_k_l1: int
    l1_buffers: int
    l0c_buffers: int
    used_core_num: int
    full_load: bool
    m_tail_tile: int
    n_tail_tile: int
    m_base_tail_split_count: int
    n_base_tail_split_count: int
    m_tail_main: int
    n_tail_main: int


def get_qbmm_mx_tile_config(
    m,
    n,
    k,
    transpose_a=False,
    transpose_b=True,
    has_bias=False,
    output_dtype=None,
    profile=FP8_TILING_PROFILE,
):
    """Generate the complete MX host tiling for one shape."""
    if profile.element_bytes != 1.0:
        # Packed FP4: whole-tile GM staging requires tile-aligned K, so the
        # ping-pong guard only checks the shared K extent.
        pingpong_aligned = k % 128 == 0
    else:
        # FP8: both GM-contiguous inner extents must stay 128-aligned or the
        # ping-pong scaleKL1 window adjustment hits tail blocks.
        pingpong_aligned = (m if transpose_a else k) % 128 == 0 and (
            k if transpose_b else n
        ) % 128 == 0
    # L0C ping-pong doubles the accumulator footprint, so it needs 16-bit
    # FixPipe output, aligned tiles, and a cube-bound shape where overlapping
    # the next tile's MMAD with FixPipe actually hides latency.
    enable_l0c_pingpong = (
        output_dtype in (torch.float16, torch.bfloat16)
        and pingpong_aligned
        and _is_cube_bound(m, n, k, 2, profile=profile)
    )
    base_m, base_n, base_k = _get_base_block(
        m, n, k, transpose_a, transpose_b, profile=profile
    )
    if profile.element_bytes != 1.0:
        pass
    full_load = _can_use_a_full_load(
        m,
        n,
        k,
        base_m,
        base_n,
        base_k,
        transpose_a,
        transpose_b,
        has_bias,
        profile=profile,
    )
    l0c_buffers = 2 if base_m * base_n * 4 * 2 <= L0C_CAPACITY_BYTES else 1
    k_l1, scale_k_l1, l1_buffers = _cal_l1_tiling(
        m,
        n,
        k,
        base_m,
        base_n,
        base_k,
        full_load,
        transpose_a,
        transpose_b,
        has_bias,
        min(
            AIC_NUM,
            ceil_div(m, base_m) * ceil_div(n, base_n),
        ),
        enable_l0c_pingpong,
        profile=profile,
    )
    total_tiles = ceil_div(m, base_m) * ceil_div(n, base_n)
    if profile.element_bytes != 1.0:
        # Packed FP4: whole-tile GM staging requires tile-aligned outer
        # blocks, so keep the plain grid (no edge-basic-block merging).
        optimized_edge_tiling = (1, 1, 0, 0)
    else:
        optimized_edge_tiling = _optimize_edge_basic_block(
            m,
            n,
            k,
            base_m,
            base_n,
            transpose_a,
            transpose_b,
        )
    m_tail_tile, n_tail_tile = _calc_tail_basic_block_split(
        m,
        n,
        base_m,
        base_n,
        AIC_NUM,
        optimized_edge_tiling,
        transpose_a,
        transpose_b,
        profile=profile,
    )
    edge_tiling = (1, 1, 0, 0) if full_load else optimized_edge_tiling
    if full_load:
        used_core_num = min(AIC_NUM, total_tiles)
    elif total_tiles > AIC_NUM or total_tiles % AIC_NUM == 0:
        used_core_num = AIC_NUM
    else:
        used_core_num = total_tiles * m_tail_tile * n_tail_tile
    return QbmmMxTiling(
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        k_l1=k_l1,
        scale_k_l1=scale_k_l1,
        l1_buffers=l1_buffers,
        l0c_buffers=l0c_buffers,
        used_core_num=used_core_num,
        full_load=full_load,
        m_tail_tile=m_tail_tile,
        n_tail_tile=n_tail_tile,
        m_base_tail_split_count=edge_tiling[0],
        n_base_tail_split_count=edge_tiling[1],
        m_tail_main=edge_tiling[2],
        n_tail_main=edge_tiling[3],
    )


# ============================================================================
# 2. Block Scheduler and Kernel
# ============================================================================


class QbmmMxBlockScheduler:
    """Map each AIC round to one output block and its valid M/N region."""

    def __init__(self, m, n, tiling):
        base_m, base_n = tiling.base_m, tiling.base_n
        self.base_m, self.base_n = base_m, base_n
        self.used_core_num = tiling.used_core_num
        self.m_tiles = ceil_div(m, base_m)
        self.n_tiles = ceil_div(n, base_n)
        self.total_tiles = self.m_tiles * self.n_tiles

        self.m_base_tail_split_count = tiling.m_base_tail_split_count
        self.n_base_tail_split_count = tiling.n_base_tail_split_count
        self.m_base_normal_count = self.m_tiles - tiling.m_base_tail_split_count
        self.n_base_normal_count = self.n_tiles - tiling.n_base_tail_split_count
        merged_m = m - self.m_base_normal_count * base_m
        merged_n = n - self.n_base_normal_count * base_n
        self.m_base_tail_main = (
            merged_m if tiling.m_base_tail_split_count == 1 else tiling.m_tail_main
        )
        self.n_base_tail_main = (
            merged_n if tiling.n_base_tail_split_count == 1 else tiling.n_tail_main
        )
        self.m_base_tail_last = (
            merged_m - (tiling.m_base_tail_split_count - 1) * self.m_base_tail_main
        )
        self.n_base_tail_last = (
            merged_n - (tiling.n_base_tail_split_count - 1) * self.n_base_tail_main
        )

        self.main_window = min(WINDOW_LEN, self.m_tiles)
        self.main_row = self.m_tiles // self.main_window - 1
        self.tail_window = self.m_tiles - self.main_window * self.main_row
        self.total_main_tiles = self.main_row * self.main_window * self.n_tiles

        self.end_block_idx = (self.total_tiles - 1) % tiling.used_core_num
        total_tail_tile = tiling.m_tail_tile * tiling.n_tail_tile
        self.use_tail_split = (
            total_tail_tile > 1
            and (self.end_block_idx + 1) * total_tail_tile <= tiling.used_core_num
        )
        self.m_tail_tile = tiling.m_tail_tile if self.use_tail_split else 1
        self.n_tail_tile = tiling.n_tail_tile if self.use_tail_split else 1
        self.total_tail_tile = self.m_tail_tile * self.n_tail_tile
        self.tail_win_block_cnt = min(self.total_tiles, self.end_block_idx + 1)
        self.new_end_block_idx = self.end_block_idx + self.tail_win_block_cnt * (
            self.total_tail_tile - 1
        )

    @jit
    def get_round_count(self, block_idx):
        """Return the number of output-block rounds assigned to this AIC."""
        round_count = ceil_div(self.total_tiles, self.used_core_num)
        if block_idx > self.end_block_idx:
            round_count -= 1
        if (
            const_expr(self.use_tail_split)
            and self.end_block_idx < block_idx <= self.new_end_block_idx
        ):
            round_count += 1
        return round_count

    @jit
    def get_tile_idx(self, block_idx, round_idx, round_count):
        """Return the ASW M/N index and final-round split index for this AIC."""
        tail_split = False
        logical_idx = round_idx * self.used_core_num + block_idx
        sub_m_idx = 0
        sub_n_idx = 0
        if const_expr(self.use_tail_split):
            tail_split = (
                round_idx == round_count - 1 and block_idx <= self.new_end_block_idx
            )
            if tail_split:
                logical_idx = (
                    round_idx * self.used_core_num + block_idx // self.total_tail_tile
                )
                split_idx = block_idx % self.total_tail_tile
                sub_m_idx = split_idx % self.m_tail_tile
                sub_n_idx = split_idx // self.m_tail_tile

        m_idx = 0
        n_forward = 0
        row_idx = 0
        if logical_idx < self.total_main_tiles:
            row_idx = logical_idx // (self.n_tiles * self.main_window)
            m_idx = row_idx * self.main_window + logical_idx % self.main_window
            n_forward = (logical_idx // self.main_window) % self.n_tiles
        else:
            tail_idx = logical_idx - self.total_main_tiles
            row_idx = self.main_row
            m_idx = self.main_row * self.main_window + tail_idx % self.tail_window
            n_forward = (tail_idx // self.tail_window) % self.n_tiles
        reverse = row_idx % 2
        n_idx = n_forward + reverse * ((self.n_tiles - 1) - 2 * n_forward)
        return m_idx, n_idx, tail_split, sub_m_idx, sub_n_idx

    @jit
    def get_block(self, m_idx, n_idx, tail_split, sub_m_idx, sub_n_idx):
        """Return the GM start and valid M/N shape of one scheduled block."""
        current_m = self.base_m
        if m_idx >= self.m_base_normal_count:
            current_m = (
                self.m_base_tail_last
                if m_idx == self.m_tiles - 1
                else self.m_base_tail_main
            )
        current_n = self.base_n
        if n_idx >= self.n_base_normal_count:
            current_n = (
                self.n_base_tail_last
                if n_idx == self.n_tiles - 1
                else self.n_base_tail_main
            )

        m_offset = 0
        n_offset = 0
        if tail_split:
            sub_m = ceil_div(current_m, self.m_tail_tile)
            sub_n = ceil_div(current_n, self.n_tail_tile)
            m_offset = sub_m_idx * sub_m
            n_offset = sub_n_idx * sub_n
            current_m = min(current_m - m_offset, sub_m)
            current_n = min(current_n - n_offset, sub_n)
        m_start = m_idx * self.base_m
        if m_idx > self.m_base_normal_count:
            m_start -= (m_idx - self.m_base_normal_count) * (
                self.base_m - self.m_base_tail_main
            )
        n_start = n_idx * self.base_n
        if n_idx > self.n_base_normal_count:
            n_start -= (n_idx - self.n_base_normal_count) * (
                self.base_n - self.n_base_tail_main
            )
        return (
            m_start + m_offset,
            n_start + n_offset,
            current_m,
            current_n,
        )


class QbmmMxKernel:
    def __init__(
        self,
        m,
        n,
        k,
        tiling,
        a_dtype=dtypes.float8_e4m3fn,
        b_dtype=dtypes.float8_e4m3fn,
        transpose_a=False,
        transpose_b=True,
        has_bias=False,
    ):
        """Create channels and copy engines from the host tiling."""
        self.k = k
        self.base_m, self.base_n = tiling.base_m, tiling.base_n
        self.base_k, self.k_l1 = tiling.base_k, tiling.k_l1
        self.l1_buffers = tiling.l1_buffers
        self.l0c_buffers = tiling.l0c_buffers
        self.a_dtype, self.b_dtype = a_dtype, b_dtype
        self.full_load = tiling.full_load
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.used_core_num = tiling.used_core_num
        self.has_bias = has_bias
        self.block_scheduler = QbmmMxBlockScheduler(m, n, tiling)
        self.k_l1_tiles = ceil_div(k, tiling.k_l1)
        self.scale_l1_tiles = ceil_div(k, tiling.scale_k_l1)
        self.scale_k_l1_factor = tiling.scale_k_l1 // tiling.k_l1
        self.step_k = tiling.k_l1 // tiling.base_k
        self.k_l0_tiles = ceil_div(k, tiling.base_k)
        # Contiguous L1 scale lengths for baseK, scaleKL1, and full K.
        self.scale_k_l0_len = self.base_k // MX_GROUP_SIZE
        self.scale_k_l1_len = get_scale_k_len(tiling.scale_k_l1)
        self.scale_k_len = get_scale_k_len(k)
        # L2 cache policy mirroring the mainline SetBL2Cache semantics
        # (blaze kernel_qbmm_mx_without_batch.h): the B panel keeps the
        # normal cache mode whenever multiple M tiles reuse it
        # (bMustHitL2 = baseM < m), and bypasses L2 only when the whole M
        # fits one window and the N tile is cache-line aligned. A never
        # carries an explicit hint (hardware default NORMAL).
        # cannbotdsl mapping: 0 = NORMAL, 4 = DISABLE.
        # 128-byte cache-line alignment for FP8 B streaming (mainline 0x7f).
        b_cache_line_aligned = transpose_b or tiling.base_n % 128 == 0
        b_must_hit_l2 = tiling.base_m < m
        disable_weight_l2 = not b_must_hit_l2 and b_cache_line_aligned
        self.l2_cache_ctl_a = 0
        self.l2_cache_ctl_b = 0 if not disable_weight_l2 else 4
        if dtypes.fp4x2_e2m1 in (a_dtype, b_dtype):
            # Packed FP4 workaround: the fp4x2 nd2nz lowering keeps sub-byte
            # tails correct only when the packed+transposed-loaded operand
            # carries L2 hint 1 (the historical MXFP4 value) while the
            # others keep NORMAL(0). A cache hint must not affect numerics
            # (framework defect, tracked with the nd2nz tail issue).
            self.l2_cache_ctl_a = 1 if transpose_a else 0
            self.l2_cache_ctl_b = 0 if transpose_b else 1

    @jit
    def _compute_output_tile(
        self,
        out_gm,
        a_gm,
        b_gm,
        scale_a_gm,
        scale_b_gm,
        l1_a,
        l1_b,
        l1_scale_a,
        l1_scale_b,
        l0a,
        l0b,
        l0c,
        l1_bias,
        l1_bt,
        copy_a_gm_to_l1,
        copy_b_gm_to_l1,
        copy_scale_a_gm_to_l1,
        copy_scale_b_gm_to_l1,
        fp,
        bias_nd_engine,
        m_idx,
        n_idx,
        bias_gm,
        tail_split=False,
        sub_m_idx=0,
        sub_n_idx=0,
    ):
        """Compute one output tile with the shared L0C and FixPipe path."""
        m_start, n_start, current_m, current_n = self.block_scheduler.get_block(
            m_idx,
            n_idx,
            tail_split,
            sub_m_idx,
            sub_n_idx,
        )
        m_end = m_start + current_m
        n_end = n_start + current_n
        out_tile = out_gm[m_start:m_end, n_start:n_end]
        # One L0C slot per output tile: the whole K-accumulation and the
        # FixPipe read below share it; the produce cursor advances once per
        # tile so depth>1 rotates slots across tiles (FixPipe overlap).
        l0c_slot = l0c.produce()

        bias = None
        if const_expr(self.has_bias):
            bias_tile = bias_gm[n_start:n_end,]
            mem_copy(l1_bias.produce(), bias_tile, engine=bias_nd_engine)
            mem_copy(l1_bt.produce(), l1_bias.consume())
            bias = l1_bt.consume()

        # Each scaleKL1 window covers scale_k_l1_factor data kL1 windows.
        scale_group_count = self.scale_k_l1_len // MX_SCALE_PAIR
        for scale_l1_idx in range(self.scale_l1_tiles):
            scale_group_start = scale_l1_idx * scale_group_count
            scale_group_end = min(
                scale_group_start + scale_group_count,
                ceil_div(self.k, MX_K_ALIGN),
            )
            # Copy ScaleA from GM to L1 unless it is resident in full-load mode.
            if const_expr(not self.full_load):
                if const_expr(self.transpose_a):
                    scale_a_gm_tile = scale_a_gm[
                        scale_group_start:scale_group_end,
                        m_start:m_end,
                        0:MX_SCALE_PAIR,
                    ]
                else:
                    scale_a_gm_tile = scale_a_gm[
                        m_start:m_end,
                        scale_group_start:scale_group_end,
                        0:MX_SCALE_PAIR,
                    ]
                mem_copy(
                    l1_scale_a.produce(),
                    scale_a_gm_tile,
                    engine=copy_scale_a_gm_to_l1,
                    l2_cache_ctl=1,
                )
            # Scale window is written once above and read by every kL0 step
            # below: select the read slot once per scale window so the
            # consume cursor tracks the produce cursor (depth rotation).
            scale_a_l1_rd = l1_scale_a.consume()
            # ScaleB GM→L1
            if const_expr(self.transpose_b):
                scale_b_gm_tile = scale_b_gm[
                    n_start:n_end,
                    scale_group_start:scale_group_end,
                    0:MX_SCALE_PAIR,
                ]
            else:
                scale_b_gm_tile = scale_b_gm[
                    scale_group_start:scale_group_end,
                    n_start:n_end,
                    0:MX_SCALE_PAIR,
                ]
            mem_copy(
                l1_scale_b.produce(),
                scale_b_gm_tile,
                engine=copy_scale_b_gm_to_l1,
                l2_cache_ctl=1,
            )
            scale_b_l1_rd = l1_scale_b.consume()

            # Process only valid data windows in the final scaleKL1 window.
            current_scale_k_l1_factor = min(
                self.scale_k_l1_factor,
                self.k_l1_tiles - scale_l1_idx * self.scale_k_l1_factor,
            )
            for k_l1_idx_in_scale in range(current_scale_k_l1_factor):
                global_k_l1_idx = (
                    scale_l1_idx * self.scale_k_l1_factor + k_l1_idx_in_scale
                )
                k_start = global_k_l1_idx * self.k_l1
                k_end = min(k_start + self.k_l1, self.k)
                # Copy A/B from GM to L1; full-load mode copies only B here.
                if const_expr(not self.full_load):
                    if const_expr(self.transpose_a):
                        a_gm_tile = a_gm[k_start:k_end, m_start:m_end]
                    else:
                        a_gm_tile = a_gm[m_start:m_end, k_start:k_end]
                    mem_copy(
                        l1_a.produce(),
                        a_gm_tile,
                        engine=copy_a_gm_to_l1,
                        l2_cache_ctl=self.l2_cache_ctl_a,
                    )
                if const_expr(self.transpose_b):
                    b_gm_tile = b_gm[n_start:n_end, k_start:k_end]
                else:
                    b_gm_tile = b_gm[k_start:k_end, n_start:n_end]
                mem_copy(
                    l1_b.produce(),
                    b_gm_tile,
                    engine=copy_b_gm_to_l1,
                    l2_cache_ctl=self.l2_cache_ctl_b,
                )
                # Same kL1 window for the L1→L0A/L0B reads below: select
                # once per window (full_load keeps A resident in slot 0).
                a_l1_rd = l1_a.consume()
                b_l1_rd = l1_b.consume()

                # The final kL1 window contains only its valid baseK blocks.
                current_k_l1 = min(self.k_l1, self.k - k_start)
                current_step_k = ceil_div(current_k_l1, self.base_k)
                # Move each baseK block to L0A/L0B and accumulate with MX MMAD.
                for k_l0_idx in range(current_step_k):
                    global_k_l0_idx = global_k_l1_idx * self.step_k + k_l0_idx
                    scale_l0_idx_in_l1 = k_l1_idx_in_scale * self.step_k + k_l0_idx
                    # Full-load A uses a global K index; streamed A uses a local one.
                    a_k_l0_idx = (
                        global_k_l0_idx if const_expr(self.full_load) else k_l0_idx
                    )
                    scale_a_k_l0_idx = (
                        global_k_l0_idx
                        if const_expr(self.full_load)
                        else scale_l0_idx_in_l1
                    )
                    # L1→L0A + MX ScaleA
                    a_l1_tile = tile_slice(
                        a_l1_rd,
                        (self.base_k, self.base_m)
                        if const_expr(self.transpose_a)
                        else (self.base_m, self.base_k),
                        (a_k_l0_idx, 0)
                        if const_expr(self.transpose_a)
                        else (0, a_k_l0_idx),
                    )
                    scale_a_l1_tile = tile_slice(
                        scale_a_l1_rd,
                        (self.base_m, self.scale_k_l0_len),
                        (0, scale_a_k_l0_idx),
                    )
                    mem_copy(
                        l0a.produce(),
                        a_l1_tile,
                        mx_scale=scale_a_l1_tile,
                        transpose=self.transpose_a,
                    )
                    # L1→L0B + MX ScaleB
                    b_l1_tile = tile_slice(
                        b_l1_rd,
                        (self.base_n, self.base_k)
                        if const_expr(self.transpose_b)
                        else (self.base_k, self.base_n),
                        (0, k_l0_idx)
                        if const_expr(self.transpose_b)
                        else (k_l0_idx, 0),
                    )
                    scale_b_l1_tile = tile_slice(
                        scale_b_l1_rd,
                        (self.scale_k_l0_len, self.base_n),
                        (scale_l0_idx_in_l1, 0),
                    )
                    mem_copy(
                        l0b.produce(),
                        b_l1_tile,
                        mx_scale=scale_b_l1_tile,
                        transpose=not self.transpose_b,
                    )
                    # MX MMAD
                    is_final_acc = global_k_l0_idx + 1 == self.k_l0_tiles
                    matmul(
                        l0c_slot,
                        l0a.consume(),
                        l0b.consume(),
                        init=(global_k_l0_idx == 0),
                        bias=bias,
                        unit_flag=3 if is_final_acc else 2,
                    )

        # FixPipe: L0C→GM
        mem_copy(out_tile, l0c_slot, engine=fp, unit_flag=3, l2_cache_ctl=1)

    # Device kernel entry. It maps output tiles to AIC cores, stages A/B/Scale
    # from GM to L1 and then L0A/L0B, performs K-sliced MX MMAD accumulation
    # in L0C, and writes the completed tile back to GM via FixPipe.
    @kernel
    def qbmm_kernel(
        self,
        out_gm: Tensor,
        a_gm: Tensor,
        b_gm: Tensor,
        scale_a_gm: Tensor,
        scale_b_gm: Tensor,
        bias_gm: Tensor,
    ):
        """Assign output tiles to AICs with AL1_FULL_LOAD or ASW snake order."""
        full_load = self.full_load
        transpose_a, transpose_b = self.transpose_a, self.transpose_b
        base_m, base_n, base_k = self.base_m, self.base_n, self.base_k
        k_l1, l1_buffers, l0c_buffers = self.k_l1, self.l1_buffers, self.l0c_buffers
        a_dtype, b_dtype = self.a_dtype, self.b_dtype

        # L1/L0 Channels and copy engines — created inside kernel context and
        # kept as locals: SSA values must not ride the `self` receiver across
        # the @jit helper boundary (ST101), so they are passed as explicit
        # arguments to _compute_output_tile instead.
        a_k = ceil_align(self.k, MX_K_ALIGN) if full_load else k_l1
        a_depth = 1 if full_load else l1_buffers
        scale_a_l1_len = self.scale_k_len if full_load else self.scale_k_l1_len
        a_scale_depth = 1 if full_load else 2
        l1_a = Channel(
            MemLoc.L1,
            (a_k, base_m) if transpose_a else (base_m, a_k),
            a_dtype,
            depth=a_depth,
            data_format="nz",
        )
        l1_b = Channel(
            MemLoc.L1,
            (base_n, k_l1) if transpose_b else (k_l1, base_n),
            b_dtype,
            depth=l1_buffers,
            data_format="nz",
        )
        l1_scale_a = Channel(
            MemLoc.L1,
            (base_m, scale_a_l1_len),
            dtypes.float8_e8m0,
            depth=a_scale_depth,
            data_format="zn",
        )
        l1_scale_b = Channel(
            MemLoc.L1,
            (self.scale_k_l1_len, base_n),
            dtypes.float8_e8m0,
            depth=2,
            data_format="nz",
        )
        # Packed FP4: transpose lowering doubles the packed outer extent
        # (two nibbles per byte) and swaps the fractal orientation.
        l0a = Channel(
            MemLoc.L0A,
            (base_m * 2, base_k)
            if const_expr(a_dtype == dtypes.fp4x2_e2m1 and transpose_a)
            else (base_m, base_k),
            a_dtype,
            depth=2,
            data_format="zn"
            if const_expr(a_dtype == dtypes.fp4x2_e2m1 and transpose_a)
            else "nz",
        )
        l0b = Channel(
            MemLoc.L0B,
            (base_n, base_k)
            if const_expr(b_dtype == dtypes.fp4x2_e2m1 and transpose_b)
            else (base_n * 2, base_k)
            if const_expr(b_dtype == dtypes.fp4x2_e2m1)
            else (base_n, base_k),
            b_dtype,
            depth=2,
            data_format="nz"
            if const_expr(b_dtype == dtypes.fp4x2_e2m1 and transpose_b)
            else "zn",
        )
        l0c = Channel(MemLoc.L0C, (base_m, base_n), dtypes.float32, depth=l0c_buffers)
        l1_bias = None
        l1_bt = None
        bias_nd_engine = None
        if const_expr(self.has_bias):
            l1_bias = Channel(
                MemLoc.L1, (base_n,), dtypes.float32, depth=2, data_format="nd"
            )
            l1_bt = Channel(
                MemLoc.BIAS, (base_n,), dtypes.float32, depth=2, data_format="nd"
            )
            bias_nd_engine = make_copy_engine(format_transform="identity")
        copy_a_gm_to_l1 = make_copy_engine(format_transform="nd2nz")
        copy_b_gm_to_l1 = make_copy_engine(format_transform="nd2nz")
        copy_scale_a_gm_to_l1 = make_copy_engine(
            format_transform="mx_scale_adn" if transpose_a else "mx_scale_and",
        )
        copy_scale_b_gm_to_l1 = make_copy_engine(
            format_transform="mx_scale_bdn" if transpose_b else "mx_scale_bnd",
        )
        fp = make_copy_engine()

        # Shared tile context (GM tensors, channels, engines): SSA values
        # must not ride `self` across the @jit boundary (ST101), so they are
        # bundled here and passed to _compute_output_tile explicitly.
        tile_ctx = (
            out_gm,
            a_gm,
            b_gm,
            scale_a_gm,
            scale_b_gm,
            l1_a,
            l1_b,
            l1_scale_a,
            l1_scale_b,
            l0a,
            l0b,
            l0c,
            l1_bias,
            l1_bt,
            copy_a_gm_to_l1,
            copy_b_gm_to_l1,
            copy_scale_a_gm_to_l1,
            copy_scale_b_gm_to_l1,
            fp,
            bias_nd_engine,
        )

        # Multi-core scheduling: full_load assigns fixed M tiles across cores;
        # normal mode uses ASW snake ordering with 4-row sliding window.
        block_idx = get_block_idx()

        if const_expr(self.full_load):
            # Keep A/ScaleA for one M tile resident and reuse them across N tiles.
            full_m_idx = block_idx % self.block_scheduler.m_tiles
            n_lane = block_idx // self.block_scheduler.m_tiles
            n_stride = AIC_NUM // self.block_scheduler.m_tiles
            a_k_len = ceil_align(self.k, MX_K_ALIGN)
            a_tile = tile_slice(
                a_gm,
                (a_k_len, self.base_m)
                if const_expr(self.transpose_a)
                else (self.base_m, a_k_len),
                (0, full_m_idx) if const_expr(self.transpose_a) else (full_m_idx, 0),
            )
            mem_copy(
                l1_a.produce(),
                a_tile,
                engine=copy_a_gm_to_l1,
                l2_cache_ctl=self.l2_cache_ctl_a,
            )
            scale_group_len = self.scale_k_len // MX_SCALE_PAIR
            scale_a_tile = tile_slice(
                scale_a_gm,
                (scale_group_len, self.base_m, MX_SCALE_PAIR)
                if const_expr(self.transpose_a)
                else (self.base_m, scale_group_len, MX_SCALE_PAIR),
                (0, full_m_idx, 0)
                if const_expr(self.transpose_a)
                else (full_m_idx, 0, 0),
            )
            mem_copy(
                l1_scale_a.produce(),
                scale_a_tile,
                engine=copy_scale_a_gm_to_l1,
                l2_cache_ctl=1,
            )
            for full_n_idx in range(n_lane, self.block_scheduler.n_tiles, n_stride):
                self._compute_output_tile(
                    *tile_ctx,
                    full_m_idx,
                    full_n_idx,
                    bias_gm,
                )
        else:
            round_count = self.block_scheduler.get_round_count(block_idx)
            for round_idx in range(round_count):
                m_idx, n_idx, tail_split, sub_m_idx, sub_n_idx = (
                    self.block_scheduler.get_tile_idx(block_idx, round_idx, round_count)
                )
                self._compute_output_tile(
                    *tile_ctx,
                    m_idx,
                    n_idx,
                    bias_gm,
                    tail_split,
                    sub_m_idx,
                    sub_n_idx,
                )

    @host
    def run(
        self,
        out_gm: Tensor,
        a_gm: Tensor,
        b_gm: Tensor,
        a_scale_gm: Tensor,
        b_scale_gm: Tensor,
        bias_gm: Tensor,
    ):
        self.qbmm_kernel[self.used_core_num](
            out_gm,
            a_gm,
            b_gm,
            a_scale_gm,
            b_scale_gm,
            bias_gm,
        )


# ============================================================================
# 3. Torch Interface
# ============================================================================

_TORCH_FP8_INPUT_TO_DSL = {
    torch.float8_e4m3fn: dtypes.float8_e4m3fn,
    torch.float8_e5m2: dtypes.float8_e5m2,
}
_TORCH_FP4_INPUT = torch.float4_e2m1fn_x2
_TORCH_OUTPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_TORCH_OUTPUT_TO_DSL = {
    torch.float32: dtypes.float32,
    torch.float16: dtypes.float16,
    torch.bfloat16: dtypes.bfloat16,
}


def _infer_qbmm_mx_layout(a, b, scale_a, scale_b, is_mxfp4=False):
    """Infer M/N/K and transpose flags from X1/X2/Scale view shapes.

    Do not rewrite the input tensors. Return
    ``(M, N, K, transposeX1, transposeX2)``. MXFP4 stores two logical
    elements in each byte of the physical innermost axis.
    """
    if scale_a.stride(-1) != 1 or scale_b.stride(-1) != 1:
        raise ValueError("scale paired lane must have unit stride")

    matches = []
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)
    scale_a_shape = tuple(scale_a.shape)
    scale_b_shape = tuple(scale_b.shape)
    packed = 2 if is_mxfp4 else 1  # fp4x2 stores two nibbles per byte
    for transpose_a in (False, True):
        m = a_shape[1] * packed if transpose_a else a_shape[0]
        k = a_shape[0] if transpose_a else a_shape[1] * packed
        group_count = ceil_div(k, MX_K_ALIGN)
        expected_scale_a = (
            (group_count, m, MX_SCALE_PAIR)
            if transpose_a
            else (m, group_count, MX_SCALE_PAIR)
        )
        if scale_a_shape != expected_scale_a:
            continue
        for transpose_b in (False, True):
            # Packing follows the innermost dimension: transposeB packs the
            # (N, K) view along K; !transposeB packs the (K, N) view along
            # N, so K stays logical and N doubles.
            if transpose_b:
                n = b_shape[0]
                b_k = b_shape[1] * packed
            else:
                b_k = b_shape[0]
                n = b_shape[1] * packed
            if b_k != k:
                continue
            expected_scale_b = (
                (n, group_count, MX_SCALE_PAIR)
                if transpose_b
                else (group_count, n, MX_SCALE_PAIR)
            )
            if scale_b_shape == expected_scale_b:
                matches.append((m, n, k, transpose_a, transpose_b))

    if len(matches) != 1:
        if not matches:
            raise ValueError(
                "cannot infer MX layout from view shape: "
                f"a={a_shape}, b={b_shape}, "
                f"pertoken_scale={scale_a_shape}, scale={scale_b_shape}"
            )
        raise ValueError(
            "ambiguous MX view shape; please use non-ambiguous "
            f"a/b/scale shapes, candidates={matches}"
        )
    m, n, k, transpose_a, transpose_b = matches[0]
    if is_mxfp4:
        # Sub-byte packing follows the innermost dimension, so the packed
        # extent must be even: M for transposeA, N for !transposeB, K when
        # either operand packs along K.
        if transpose_a and m % 2 != 0:
            raise ValueError(
                f"transposeA=True packs 2 fp4 elements per byte along M; M={m} must be even"
            )
        if not transpose_b and n % 2 != 0:
            raise ValueError(
                f"transposeB=False packs 2 fp4 elements per byte along N; N={n} must be even"
            )
        if (not transpose_a or transpose_b) and k % 2 != 0:
            raise ValueError(
                f"K={k} must be even when either operand is packed along K"
            )
    return m, n, k, transpose_a, transpose_b


# AOT compilation is expensive (tiling + kernel build + run.compile), so the
# builder is memoized per (shape config, dtypes, profile) with lru_cache.
@functools.cache
def _build_mx_aot_callable(
    m,
    n,
    k,
    has_bias,
    output_dtype,
    transpose_a,
    transpose_b,
    a_dtype,
    b_dtype,
    profile,
):
    tiling = get_qbmm_mx_tile_config(
        m,
        n,
        k,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        has_bias=has_bias,
        output_dtype=output_dtype,
        profile=profile,
    )
    kernel_obj = QbmmMxKernel(
        m,
        n,
        k,
        tiling,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        has_bias=has_bias,
    )
    group_count = ceil_div(k, MX_K_ALIGN)
    a_shape = (k, m) if transpose_a else (m, k)
    b_shape = (k, n) if not transpose_b else (n, k)
    scale_a_shape = (
        (group_count, m, MX_SCALE_PAIR)
        if transpose_a
        else (m, group_count, MX_SCALE_PAIR)
    )
    scale_b_shape = (
        (n, group_count, MX_SCALE_PAIR)
        if transpose_b
        else (group_count, n, MX_SCALE_PAIR)
    )
    specs = (
        TensorSpec(shape=(m, n), dtype=_TORCH_OUTPUT_TO_DSL[output_dtype]),
        TensorSpec(shape=a_shape, dtype=a_dtype),
        TensorSpec(shape=b_shape, dtype=b_dtype),
        TensorSpec(shape=scale_a_shape, dtype=dtypes.float8_e8m0),
        TensorSpec(shape=scale_b_shape, dtype=dtypes.float8_e8m0),
        TensorSpec(shape=(n if has_bias else 0,), dtype=dtypes.float32),
    )
    return cannbotdsl.compile(kernel_obj.run, *specs)


def _validate_mx_quant_matmul_input(a, b, scale_a, scale_b, bias, output_dtype):
    """Validate npu_quant_matmul inputs; return (is_mxfp4, has_bias)."""
    is_mxfp4 = a.dtype == _TORCH_FP4_INPUT and b.dtype == _TORCH_FP4_INPUT
    if a.dtype not in _TORCH_FP8_INPUT_TO_DSL and not is_mxfp4:
        raise TypeError(
            f"a/b must be FP8 or packed FP4 tensors, got {a.dtype} and {b.dtype}"
        )
    if is_mxfp4 != (b.dtype == _TORCH_FP4_INPUT):
        raise TypeError("a and b must use the same MX data dtype")
    has_bias = bias is not None
    if output_dtype not in _TORCH_OUTPUT_DTYPES:
        raise ValueError(
            f"MX quant matmul requires a supported output_dtype, got {output_dtype}"
        )
    if a.dim() != 2 or b.dim() != 2:
        raise NotImplementedError(
            f"MX batch dims not implemented, got rank {a.dim()} and {b.dim()}"
        )
    if scale_a.dim() != 3 or scale_b.dim() != 3:
        raise ValueError("MX scales must use rank-3 paired layouts")
    if any(t.device != a.device for t in (a, b, scale_a, scale_b)):
        raise ValueError("a, b and both MX scales must be on the same device")
    if scale_a.dtype != torch.float8_e8m0fnu or scale_b.dtype != torch.float8_e8m0fnu:
        raise TypeError("scale_a and scale_b must be FLOAT8_E8M0 tensors")
    if has_bias and bias.dtype != torch.float32:
        raise TypeError(f"bias must be float32, got {bias.dtype}")
    return is_mxfp4, has_bias


def _prepare_bias_arg(bias, n, device):
    """Normalize bias for the kernel call: validate the length, or return a
    length-0 placeholder when no bias is used (six-tensor ABI contract)."""
    if bias is None:
        return torch.empty(0, dtype=torch.float32, device=device)
    if bias.shape[0] != n:
        raise ValueError(f"bias length {bias.shape[0]} != N={n}")
    return bias


def npu_quant_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    bias: torch.Tensor = None,
    output_dtype=None,
) -> torch.Tensor:
    """Execute a two-dimensional, fully quantized MX matrix multiply.

    Supports both MXFP8 (``float8_e4m3fn``/``float8_e5m2`` data) and packed
    MXFP4 (``float4_e2m1fn_x2`` data) with E8M0 block scales: ``scale_a``
    is the paired ScaleA of ``a`` and ``scale_b`` is the paired ScaleB of
    ``b``.
    """
    is_mxfp4, has_bias = _validate_mx_quant_matmul_input(
        a,
        b,
        scale_a,
        scale_b,
        bias,
        output_dtype,
    )

    m, n, k, transpose_a, transpose_b = _infer_qbmm_mx_layout(
        a, b, scale_a, scale_b, is_mxfp4=is_mxfp4
    )
    bias = _prepare_bias_arg(bias, n, a.device)
    output = torch.empty((m, n), dtype=output_dtype, device=a.device)

    if is_mxfp4:
        a_dtype = b_dtype = dtypes.fp4x2_e2m1
        profile = FP4_TILING_PROFILE
        a_input, b_input = a.view(torch.int8), b.view(torch.int8)
    else:
        a_dtype = _TORCH_FP8_INPUT_TO_DSL[a.dtype]
        b_dtype = _TORCH_FP8_INPUT_TO_DSL[b.dtype]
        profile = FP8_TILING_PROFILE
        a_input, b_input = a, b
    compiled_kernel = _build_mx_aot_callable(
        m,
        n,
        k,
        has_bias,
        output_dtype,
        transpose_a,
        transpose_b,
        a_dtype,
        b_dtype,
        profile,
    )
    compiled_kernel(output, a_input, b_input, scale_a, scale_b, bias)
    return output
