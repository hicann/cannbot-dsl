# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Ascend950 MXFP8 ``npu_quant_matmul`` sample.

Structure:
  1. Host-side ASWT tiling
  2. MXFP8 kernel with GM->L1->L0->L0C->GM data flow
  3. Torch-facing ``npu_quant_matmul`` wrapper and issue reproduction helpers

Formula: Y[M,N] = dequant(X1[M,K]) @ dequant(X2[K,N])
MX scale ABI: x1Scale uses ScaleAND[M,G,2], x2Scale uses ScaleBND[G,N,2],
where G=ceil(K/64). Transpose is represented by stride metadata.
"""

__all__ = ["npu_quant_matmul"]

import os

import torch

from cannbotdsl import dtypes, get_mem_size, get_platform_info
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.channel import Channel
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.types import Float8E4M3FN, Float8E5M2, Float8E8M0
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.types import MemLoc, Tensor
from cannbotdsl.tensor import tile_view
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy


L1_CAPACITY_BYTES = get_mem_size("l1")
L0A_CAPACITY_BYTES = get_mem_size("l0a")
L0B_CAPACITY_BYTES = get_mem_size("l0b")
L0C_CAPACITY_BYTES = get_mem_size("l0c")
L2_CAPACITY_BYTES = int(  # Used to decide whether an input should bypass L2.
    os.environ.get("QBMM_MX_L2_SIZE_BYTES", str(128 * 1024 * 1024))
)
MX_GROUP_SIZE = 32  # One E8M0 scale is shared by every 32 K elements.
MX_K_ALIGN = 64  # Every 64 K elements map to one paired-scale group.
MX_SCALE_PAIR = 2
WINDOW_LEN = 4  # Number of output M tiles in one ASW window.
ESTIMATED_SCALE_K = 4096  # Initial scaleKL1 used by the mainline L1 model.
MX_L0C_PINGPONG_SCALE_K_L1_TARGET = 2048
MX_L0C_PINGPONG_OUTPUT_SIZE_LIMIT = 128 * 1024 * 1024
HBM_BANDWIDTH = 1.4
L2_BANDWIDTH = 5.2
MXFP8_CUBE_THROUGHPUT = 864.0
L2_CACHE_THRESHOLD_BYTES = 128 * 1024 * 1024 * 80 // 100
MMAD_BLOCK_SIZE = 256
CUBE_BLOCK = 16
L1_ALIGN_SIZE = 32
L2_ALIGN_SIZE = 128
BASIC_BLOCK_SIZE_128 = 128
LOAD_BALANCE_BASE_N_128_ALIGN_K_THRESHOLD = 2560
# Floating-point tolerance used by tiling score comparisons.
SCORE_COMPARE_EPS = 1e-12
BASE_K_LIMIT = 4095


def _get_block_dim():
    return get_platform_info().cube_core_num


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
    round_limit = ceil_div(ceil_div(m, base_m) * ceil_div(n, base_n), _get_block_dim())
    if round_limit == 1 or round_limit > 3:
        return base_m, base_n

    origin_blocks = ceil_div(m, base_m) * ceil_div(n, base_n)
    origin_tail_blocks = origin_blocks % _get_block_dim() or _get_block_dim()
    origin_tail_used_cores = origin_tail_blocks * (
        _get_block_dim() // origin_tail_blocks
    )
    origin_memory_score = (base_m + base_n) / (base_m * base_n)

    best_m, best_n = base_m, base_n
    best_balance = 0.0
    best_memory_score = float("inf")
    found = False
    for candidate_m in range(ceil_align(base_m, base_m_align), 0, -base_m_align):
        for candidate_n in range(ceil_align(base_n, base_n_align), 0, -base_n_align):
            blocks = ceil_div(m, candidate_m) * ceil_div(n, candidate_n)
            rounds = ceil_div(blocks, _get_block_dim())
            if rounds > round_limit:
                break
            if candidate_m == base_m and candidate_n == base_n:
                continue
            tail_blocks = blocks % _get_block_dim() or _get_block_dim()
            memory_score = (candidate_m + candidate_n) / (candidate_m * candidate_n)
            if tail_blocks < origin_tail_used_cores:
                continue
            if (
                tail_blocks == origin_tail_used_cores
                and memory_score > origin_memory_score + SCORE_COMPARE_EPS
            ):
                continue
            # Fraction of scheduled tile capacity that contains valid output.
            balance = m * n / _get_block_dim() / (rounds * candidate_m * candidate_n)
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
    swap_round = ceil_div(
        ceil_div(m, swap_base_m) * ceil_div(n, swap_base_n), _get_block_dim()
    )
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


def _get_base_block(m, n, k, transpose_a=False, transpose_b=True):
    """Select the per-AIC ``baseM/baseN/baseK`` compute block.

    Start from at most 256x256x128 and use transpose-specific alignment. For a
    multi-round launch, optimize final-round balance. For a single-round
    launch, redistribute the core grid and improve the tile aspect ratio.
    Finally, limit baseK by the double-buffered L0A/L0B capacity.
    """
    base_m_align = L1_ALIGN_SIZE if transpose_a else CUBE_BLOCK
    base_n_align = CUBE_BLOCK if transpose_b else L1_ALIGN_SIZE
    base_k_align = MX_K_ALIGN if (transpose_a or not transpose_b) else L2_ALIGN_SIZE
    # Start from at most 256x256x128; once all AICs are used, tune only the final round.
    base_m = ceil_align(min(m, 256), base_m_align)
    base_n = ceil_align(min(n, 256), base_n_align)
    base_k = ceil_align(min(k, 128), MX_K_ALIGN)
    if ceil_div(m, base_m) * ceil_div(n, base_n) >= _get_block_dim():
        base_m, base_n = _optimize_base_block_for_load_balance(
            m,
            n,
            k,
            base_m,
            base_n,
            base_m_align,
            base_n_align,
        )
        return base_m, base_n, base_k

    # Single-round case: keep the less divisible axis and assign the remaining
    # cores to the other axis so that as many available AICs run as possible.
    m_max_tile = ceil_div(m, base_m_align)
    n_max_tile = ceil_div(n, base_n_align)
    m_core = ceil_div(m, base_m)
    n_core = ceil_div(n, base_n)
    if m_max_tile <= n_max_tile:
        base_m = ceil_align(ceil_div(m, m_core), base_m_align)
        m_core = ceil_div(m, base_m)
        n_core = _get_block_dim() // m_core
        base_n = ceil_align(ceil_div(n, n_core), base_n_align)
    else:
        base_n = ceil_align(ceil_div(n, n_core), base_n_align)
        n_core = ceil_div(n, base_n)
        m_core = _get_block_dim() // n_core
        base_m = ceil_align(ceil_div(m, m_core), base_m_align)

    # Move baseM/baseN toward a square tile. This lowers the A/B transfer ratio
    # and may allow a larger baseK under the double-buffered L0 capacity limit.
    # When baseN >= 2*baseM, double n_core and derive m_core from BLOCK_DIM.
    while (
        base_n >= base_m * 2
        and n_core < _get_block_dim() // 2
        and base_n != base_n_align
    ):
        n_core *= 2
        m_core = _get_block_dim() // n_core
        base_m = ceil_align(ceil_div(m, m_core), base_m_align)
        base_n = ceil_align(ceil_div(n, n_core), base_n_align)
        m_core = ceil_div(m, base_m)
        n_core = ceil_div(n, base_n)
    # Apply the symmetric adjustment when the M direction is too long.
    while (
        base_m >= base_n * 2
        and m_core < _get_block_dim() // 2
        and base_m != base_m_align
    ):
        m_core *= 2
        n_core = _get_block_dim() // m_core
        base_m = ceil_align(ceil_div(m, m_core), base_m_align)
        base_n = ceil_align(ceil_div(n, n_core), base_n_align)
        m_core = ceil_div(m, base_m)
        n_core = ceil_div(n, base_n)

    if not transpose_a and transpose_b:
        base_m, base_n = _try_swap_base_mn_for_mx_false_true(m, n, base_m, base_n)
    # L0A and L0B each hold two baseK tiles. Derive the baseK limit from their
    # 64 KiB capacity, align it down, and keep it within K and BASE_K_LIMIT.
    max_base_k = min(L0A_CAPACITY_BYTES, L0B_CAPACITY_BYTES) // 2 // max(base_m, base_n)
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
        base_m_align,
        base_n_align,
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
):
    """Return the L1 bytes used by A/B data, their scales, and optional bias."""
    b_bytes = base_n * k_l1 * buffers
    b_scale_bytes = base_n * get_scale_k_len(scale_k_l1) * 2
    if full_load:
        a_bytes = base_m * ceil_align(k, MX_K_ALIGN)
        a_scale_bytes = base_m * get_scale_k_len(k)
    else:
        a_bytes = base_m * k_l1 * buffers
        a_scale_bytes = base_m * get_scale_k_len(scale_k_l1) * 2
    bias_bytes = base_n * 4 * 2 if has_bias else 0
    return a_bytes + b_bytes + a_scale_bytes + b_scale_bytes + bias_bytes


def _cal_normal_l1_tiling(base_m, base_n, k, base_k, has_bias=False):
    """Calculate ``kL1`` and ``scaleKL1`` for NORMAL_MODE."""
    bias_l1 = base_n * 4 * 2 if has_bias else 0
    available_l1_size = L1_CAPACITY_BYTES - bias_l1
    base_ab_size = (base_m + base_n) * base_k
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
    used_data_size = 2 * k_l1 * (base_m + base_n)
    left_l1_size = max(0, available_l1_size - used_data_size)
    scale_group_size = (base_m + base_n) * MX_SCALE_PAIR * 2
    max_scale_k_l1 = left_l1_size // scale_group_size * MX_K_ALIGN
    scale_k_l1 = min(max_scale_k_l1, ceil_align(k, k_l1))
    scale_k_l1 = scale_k_l1 // k_l1 * k_l1
    return k_l1, max(k_l1, scale_k_l1)


def _cal_a_full_l1_tiling(base_m, base_n, k, base_k, has_bias=False):
    """Calculate K windows for AL1_FULL_LOAD with resident A and ScaleA."""
    bias_l1 = base_n * 4 * 2 if has_bias else 0
    # Reserve resident A/ScaleA storage; B and ScaleB use the remaining L1.
    a_full_size = base_m * ceil_align(k, MX_K_ALIGN)
    a_scale_size = base_m * get_scale_k_len(k)
    left_l1_size = max(
        0,
        L1_CAPACITY_BYTES - a_full_size - a_scale_size - bias_l1,
    )

    # Meet the 128-byte B transfer granularity before expanding stepK.
    step_k_base = max(1, ceil_div(128, base_k))
    base_b_with_scale = base_n * (base_k + get_scale_k_len(base_k)) * step_k_base
    if left_l1_size >= 64 * 1024:
        step_k_scale = ceil_div(32 * 1024, base_b_with_scale)
    else:
        step_k_scale = ceil_div(left_l1_size // 2, base_b_with_scale)
    step_k = step_k_base * max(1, step_k_scale)
    # Merge at least two baseK blocks when another K window remains and fits.
    if step_k == 1 and k > base_k and left_l1_size > base_b_with_scale * 2:
        step_k = 2
    k_l1 = step_k * base_k
    # Use the space left by double-buffered B to increase ScaleB reuse.
    b_data_size = 2 * base_n * k_l1
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
    return k_l1, scale_factor * k_l1


def _can_use_four_buffer(
    base_m,
    base_n,
    k,
    k_l1,
    scale_k_l1,
    full_load,
    has_bias=False,
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
        )
        <= L1_CAPACITY_BYTES
    )


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
):
    """Expand scaleKL1 to full K when four buffers still fit in L1."""
    full_cover_scale_k_l1 = ceil_align(fit_args[2], k_l1)
    if _can_use_four_buffer(
        *fit_args, k_l1, full_cover_scale_k_l1, full_load, has_bias
    ):
        return full_cover_scale_k_l1
    return scale_k_l1


def _cal_l1_tiling(base_m, base_n, k, base_k, full_load, has_bias=False):
    """Return ``(kL1, scaleKL1, l1BufferNum)`` for the selected L1 mode.

    Calculate the NORMAL_MODE or AL1_FULL_LOAD K windows, try four A/B
    buffers, and fall back to two when the larger pipeline does not fit.
    """
    cal_k_l1 = _cal_a_full_l1_tiling if full_load else _cal_normal_l1_tiling
    k_l1, scale_k_l1 = cal_k_l1(base_m, base_n, k, base_k, has_bias)
    scale_k_l1 = _adjust_scale_k_l1_for_four_buffer(k, k_l1, scale_k_l1)
    fit_args = (base_m, base_n, k)
    if _can_use_four_buffer(*fit_args, k_l1, scale_k_l1, full_load, has_bias):
        scale_k_l1 = _get_full_cover_scale_k_l1_if_possible(
            fit_args, k_l1, scale_k_l1, full_load, has_bias
        )
        return k_l1, scale_k_l1, 4

    # If stepK 3/4 cannot use four buffers, try aligned stepK 2 for multi-round K.
    step_k = k_l1 // base_k
    step_k_two_k_l1 = 2 * base_k
    step_two_aligned = k % 128 == 0 and step_k_two_k_l1 % 256 == 0
    if step_k in (3, 4) and k_l1 * 2 < k and step_two_aligned:
        step_k_two_scale_k_l1 = _adjust_scale_k_l1_for_four_buffer(
            k, step_k_two_k_l1, scale_k_l1
        )
        if _can_use_four_buffer(
            *fit_args, step_k_two_k_l1, step_k_two_scale_k_l1, full_load, has_bias
        ):
            step_k_two_scale_k_l1 = _get_full_cover_scale_k_l1_if_possible(
                fit_args, step_k_two_k_l1, step_k_two_scale_k_l1, full_load, has_bias
            )
            return step_k_two_k_l1, step_k_two_scale_k_l1, 4

    return k_l1, scale_k_l1, 2


def _adjust_scale_k_l1_for_l0c_pingpong(
    m,
    n,
    k_l1,
    scale_k_l1,
    l0c_buffers,
):
    """Limit scaleKL1 to about 2048 K elements for L0C ping-pong.

    Keep the original window when the output is too large, L0C is not
    double-buffered, or scaleKL1 is already within the target.
    """
    output_size = m * n * 4
    if (
        l0c_buffers != 2
        or output_size > MX_L0C_PINGPONG_OUTPUT_SIZE_LIMIT
        or scale_k_l1 <= MX_L0C_PINGPONG_SCALE_K_L1_TARGET
    ):
        return scale_k_l1
    scale_factor = max(
        1,
        MX_L0C_PINGPONG_SCALE_K_L1_TARGET // k_l1,
    )
    return max(k_l1, min(scale_k_l1, scale_factor * k_l1))


def _is_cube_bound(m, n, k, output_element_bytes):
    """Estimate whether Cube compute time exceeds data transfer time."""
    a_input_cost = m * k
    b_input_cost = n * k
    a_scale_cost = ceil_div(a_input_cost, MX_GROUP_SIZE)
    b_scale_cost = ceil_div(b_input_cost, MX_GROUP_SIZE)
    copy_out_bytes = m * n * output_element_bytes
    hbm_bytes = a_input_cost + a_scale_cost + b_input_cost + b_scale_cost
    a_l2_cost = m * k * (ceil_div(n, MMAD_BLOCK_SIZE) - 1)
    b_l2_cost = n * k * (ceil_div(m, MMAD_BLOCK_SIZE) - 1)
    l2_bytes = (
        a_l2_cost
        + ceil_div(a_l2_cost, MX_GROUP_SIZE)
        + b_l2_cost
        + ceil_div(b_l2_cost, MX_GROUP_SIZE)
    )
    l2_footprint = hbm_bytes + copy_out_bytes
    copy_out_bandwidth = (
        L2_BANDWIDTH if l2_footprint < L2_CACHE_THRESHOLD_BYTES else HBM_BANDWIDTH
    )
    transfer_cost = (
        hbm_bytes / HBM_BANDWIDTH
        + l2_bytes / L2_BANDWIDTH
        + copy_out_bytes / copy_out_bandwidth
    )
    compute_cost = 2.0 * m * n * k / MXFP8_CUBE_THROUGHPUT
    return compute_cost > transfer_cost


def _can_use_a_full_load(m, n, k, base_m, base_n, base_k, has_bias=False):
    """Return whether the shape should use ``AL1_FULL_LOAD``.

    A and ScaleA must fit in half of L1, use fewer than four M tiles, and be
    reusable across N tiles. Enable full load directly with four buffers;
    otherwise require two buffers in NORMAL_MODE or more than 20 percent
    repeated A traffic.
    """
    m_tiles = ceil_div(m, base_m)
    n_tiles = ceil_div(n, base_n)
    candidate = (
        base_m * ceil_align(k, MX_K_ALIGN) <= L1_CAPACITY_BYTES // 2
        and m_tiles < WINDOW_LEN
        and _get_block_dim() % m_tiles == 0
        and m_tiles * n_tiles > _get_block_dim()
    )
    if not candidate:
        return False

    _, _, full_buffers = _cal_l1_tiling(
        base_m,
        base_n,
        k,
        base_k,
        True,
        has_bias,
    )
    if full_buffers > 2:
        return True

    _, _, normal_buffers = _cal_l1_tiling(
        base_m,
        base_n,
        k,
        base_k,
        False,
        has_bias,
    )
    repeated_a = m * (ceil_align(k, MX_K_ALIGN) + get_scale_k_len(k)) * (n_tiles - 1)
    normal_bytes = (
        m * (ceil_align(k, MX_K_ALIGN) + get_scale_k_len(k)) * n_tiles
        + n * (ceil_align(k, MX_K_ALIGN) + get_scale_k_len(k)) * m_tiles
    )
    return normal_buffers == 2 or repeated_a / normal_bytes > 0.20


def get_qbmm_mxfp8_tile_config(
    m,
    n,
    k,
    enable_l0c_pingpong=False,
    transpose_a=False,
    transpose_b=True,
    has_bias=False,
):
    """Generate the complete MXFP8 host tiling for one shape.

    Return ``(baseM, baseN, baseK, kL1, scaleKL1, l1BufferNum, l0cBufferNum,
    usedCoreNum, isAFullLoad)``. The first three values define the per-AIC
    compute block; kL1 and scaleKL1 define the GM-to-L1 K windows.
    """
    base_m, base_n, base_k = _get_base_block(m, n, k, transpose_a, transpose_b)
    full_load = _can_use_a_full_load(
        m,
        n,
        k,
        base_m,
        base_n,
        base_k,
        has_bias,
    )
    k_l1, scale_k_l1, l1_buffers = _cal_l1_tiling(
        base_m,
        base_n,
        k,
        base_k,
        full_load,
        has_bias,
    )
    l0c_buffers = 2 if base_m * base_n * 4 * 2 <= L0C_CAPACITY_BYTES else 1
    if enable_l0c_pingpong:
        scale_k_l1 = _adjust_scale_k_l1_for_l0c_pingpong(
            m,
            n,
            k_l1,
            scale_k_l1,
            l0c_buffers,
        )
    l1_buffers = (
        4
        if _can_use_four_buffer(
            base_m,
            base_n,
            k,
            k_l1,
            scale_k_l1,
            full_load,
            has_bias,
        )
        else 2
    )
    used_core_num = min(_get_block_dim(), ceil_div(m, base_m) * ceil_div(n, base_n))
    return (
        base_m,
        base_n,
        base_k,
        k_l1,
        scale_k_l1,
        l1_buffers,
        l0c_buffers,
        used_core_num,
        full_load,
    )


# ============================================================================
# 2. Kernel
# ============================================================================


class QbmmMxfp8Kernel:
    def __init__(
        self,
        m,
        n,
        k,
        base_m,
        base_n,
        base_k,
        k_l1,
        scale_k_l1,
        l1_buffers=2,
        l0c_buffers=1,
        a_dtype=Float8E4M3FN,
        b_dtype=Float8E4M3FN,
        full_load=False,
        transpose_a=False,
        transpose_b=True,
        used_core_num=None,
        has_bias=False,
    ):
        """Create channels and copy engines from the host tiling."""
        self.k = k
        self.base_m, self.base_n = base_m, base_n
        self.base_k, self.k_l1 = base_k, k_l1
        self.scale_k_l1 = scale_k_l1
        self.l1_buffers = l1_buffers
        self.l0c_buffers = l0c_buffers
        self.a_dtype, self.b_dtype = a_dtype, b_dtype
        self.full_load = full_load
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.block_dim = _get_block_dim()
        self.used_core_num = self.block_dim if used_core_num is None else used_core_num
        self.has_bias = has_bias
        self.m_tiles = ceil_div(m, base_m)
        self.n_tiles = ceil_div(n, base_n)
        self.main_window = min(WINDOW_LEN, self.m_tiles)
        self.main_row = self.m_tiles // self.main_window - 1
        self.tail_window = self.m_tiles - self.main_window * self.main_row
        self.total_main_tiles = self.main_row * self.main_window * self.n_tiles
        self.k_l1_tiles = ceil_div(k, k_l1)
        self.scale_l1_tiles = ceil_div(k, scale_k_l1)
        self.scale_k_l1_factor = scale_k_l1 // k_l1
        self.step_k = k_l1 // base_k
        self.k_l0_tiles = ceil_div(k, base_k)
        # Contiguous L1 scale lengths for baseK, scaleKL1, and full K.
        self.scale_k_l0_len = self.base_k // MX_GROUP_SIZE
        self.scale_k_l1_len = get_scale_k_len(scale_k_l1)
        self.scale_k_len = get_scale_k_len(k)
        # Bypass L2 for a single window with 128-byte-aligned K/L1 partitions.
        total_size = m * n * 4 + m * k + n * k
        self.l2_cache_ctl_a = (
            0
            if (
                not transpose_a
                and total_size >= L2_CAPACITY_BYTES
                and base_n >= n
                and k % 128 == 0
                and k_l1 % 128 == 0
            )
            else 1
        )
        self.l2_cache_ctl_b = (
            0
            if (transpose_b and base_m >= m and k % 128 == 0 and k_l1 % 128 == 0)
            else 1
        )
        # Channels and copy engines; created inside the @kernel body.
        self.l1_a = None
        self.l1_b = None
        self.l1_scale_a = None
        self.l1_scale_b = None
        self.l0a = None
        self.l0b = None
        self.l0c = None
        self.l1_bias = None
        self.l1_bt = None
        self.bias_nd_engine = None
        self.copy_a_gm_to_l1 = None
        self.copy_b_gm_to_l1 = None
        self.copy_scale_a_gm_to_l1 = None
        self.copy_scale_b_gm_to_l1 = None
        self.fp = None

    @jit
    def _compute_output_tile(
        self,
        out_gm,
        a_gm,
        b_gm,
        scale_a_gm,
        scale_b_gm,
        m_idx,
        n_idx,
        bias_gm,
    ):
        """Compute one output tile with the shared L0C and FixPipe path."""
        out_tile = tile_view(out_gm, (self.base_m, self.base_n), (m_idx, n_idx))

        if const_expr(self.has_bias):
            bias_tile = tile_view(bias_gm, (self.base_n,), (n_idx,))
            mem_copy(self.l1_bias, bias_tile, engine=self.bias_nd_engine)
            mem_copy(self.l1_bt, self.l1_bias)

        # Each scaleKL1 window covers scale_k_l1_factor data kL1 windows.
        for scale_l1_idx in range(self.scale_l1_tiles):
            # Copy ScaleA from GM to L1 unless it is resident in full-load mode.
            if const_expr(not self.full_load):
                if const_expr(self.transpose_a):
                    scale_a_gm_tile = tile_view(
                        scale_a_gm,
                        (
                            self.scale_k_l1_len // MX_SCALE_PAIR,
                            self.base_m,
                            MX_SCALE_PAIR,
                        ),
                        (scale_l1_idx, m_idx, 0),
                    )
                else:
                    scale_a_gm_tile = tile_view(
                        scale_a_gm,
                        (
                            self.base_m,
                            self.scale_k_l1_len // MX_SCALE_PAIR,
                            MX_SCALE_PAIR,
                        ),
                        (m_idx, scale_l1_idx, 0),
                    )
                mem_copy(
                    self.l1_scale_a,
                    scale_a_gm_tile,
                    engine=self.copy_scale_a_gm_to_l1,
                    l2_cache_ctl=1,
                )
            # ScaleB GM→L1
            scale_group_count = self.scale_k_l1_len // MX_SCALE_PAIR
            if const_expr(self.transpose_b):
                scale_b_gm_tile = tile_view(
                    scale_b_gm,
                    (self.base_n, scale_group_count, MX_SCALE_PAIR),
                    (n_idx, scale_l1_idx, 0),
                )
            else:
                scale_b_gm_tile = tile_view(
                    scale_b_gm,
                    (scale_group_count, self.base_n, MX_SCALE_PAIR),
                    (scale_l1_idx, n_idx, 0),
                )
            mem_copy(
                self.l1_scale_b,
                scale_b_gm_tile,
                engine=self.copy_scale_b_gm_to_l1,
                l2_cache_ctl=1,
            )

            # Process only valid data windows in the final scaleKL1 window.
            current_scale_k_l1_factor = min(
                self.scale_k_l1_factor,
                self.k_l1_tiles - scale_l1_idx * self.scale_k_l1_factor,
            )
            for k_l1_idx_in_scale in range(current_scale_k_l1_factor):
                global_k_l1_idx = (
                    scale_l1_idx * self.scale_k_l1_factor + k_l1_idx_in_scale
                )
                # Copy A/B from GM to L1; full-load mode copies only B here.
                if const_expr(not self.full_load):
                    if const_expr(self.transpose_a):
                        a_gm_tile = tile_view(
                            a_gm, (self.k_l1, self.base_m), (global_k_l1_idx, m_idx)
                        )
                    else:
                        a_gm_tile = tile_view(
                            a_gm, (self.base_m, self.k_l1), (m_idx, global_k_l1_idx)
                        )
                    mem_copy(
                        self.l1_a,
                        a_gm_tile,
                        engine=self.copy_a_gm_to_l1,
                        l2_cache_ctl=self.l2_cache_ctl_a,
                    )
                if const_expr(self.transpose_b):
                    b_gm_tile = tile_view(
                        b_gm, (self.base_n, self.k_l1), (n_idx, global_k_l1_idx)
                    )
                else:
                    b_gm_tile = tile_view(
                        b_gm, (self.k_l1, self.base_n), (global_k_l1_idx, n_idx)
                    )
                mem_copy(
                    self.l1_b,
                    b_gm_tile,
                    engine=self.copy_b_gm_to_l1,
                    l2_cache_ctl=self.l2_cache_ctl_b,
                )

                # The final kL1 window contains only its valid baseK blocks.
                current_k_l1 = min(
                    self.k_l1,
                    self.k - global_k_l1_idx * self.k_l1,
                )
                current_step_k = ceil_div(current_k_l1, self.base_k)
                # Move each baseK block to L0A/L0B and accumulate with MX MMAD.
                for k_l0_idx in range(current_step_k):
                    global_k_l0_idx = global_k_l1_idx * self.step_k + k_l0_idx
                    scale_l0_idx_in_l1 = k_l1_idx_in_scale * self.step_k + k_l0_idx
                    # Full-load A uses a global K index; streamed A uses a local one.
                    if const_expr(self.full_load):
                        a_k_l0_idx = global_k_l0_idx
                        scale_a_k_l0_idx = global_k_l0_idx
                    else:
                        a_k_l0_idx = k_l0_idx
                        scale_a_k_l0_idx = scale_l0_idx_in_l1
                    # L1→L0A + MX ScaleA
                    if const_expr(self.transpose_a):
                        a_l1_tile = tile_view(
                            self.l1_a, (self.base_k, self.base_m), (a_k_l0_idx, 0)
                        )
                    else:
                        a_l1_tile = tile_view(
                            self.l1_a, (self.base_m, self.base_k), (0, a_k_l0_idx)
                        )
                    scale_a_l1_tile = tile_view(
                        self.l1_scale_a,
                        (self.base_m, self.scale_k_l0_len),
                        (0, scale_a_k_l0_idx),
                    )
                    mem_copy(self.l0a, a_l1_tile, mx_scale=scale_a_l1_tile)
                    # L1→L0B + MX ScaleB
                    if const_expr(self.transpose_b):
                        b_l1_tile = tile_view(
                            self.l1_b, (self.base_n, self.base_k), (0, k_l0_idx)
                        )
                    else:
                        b_l1_tile = tile_view(
                            self.l1_b, (self.base_k, self.base_n), (k_l0_idx, 0)
                        )
                    scale_b_l1_tile = tile_view(
                        self.l1_scale_b,
                        (self.scale_k_l0_len, self.base_n),
                        (scale_l0_idx_in_l1, 0),
                    )
                    mem_copy(self.l0b, b_l1_tile, mx_scale=scale_b_l1_tile)
                    # MX MMAD
                    is_final_acc = global_k_l0_idx + 1 == self.k_l0_tiles
                    bias = self.l1_bt if const_expr(self.has_bias) else None
                    matmul(
                        self.l0c,
                        self.l0a,
                        self.l0b,
                        init=(global_k_l0_idx == 0),
                        bias=bias,
                        unit_flag=3 if is_final_acc else 2,
                    )

        # FixPipe: L0C→GM
        mem_copy(out_tile, self.l0c, engine=self.fp, l2_cache_ctl=1)

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

        # L1/L0 Channels and copy engines — created inside kernel context
        a_k = ceil_align(self.k, MX_K_ALIGN) if full_load else k_l1
        a_depth = 1 if full_load else l1_buffers
        scale_a_l1_len = self.scale_k_len if full_load else self.scale_k_l1_len
        a_scale_depth = 1 if full_load else 2
        self.l1_a = Channel(
            MemLoc.L1,
            (a_k, base_m) if transpose_a else (base_m, a_k),
            a_dtype,
            depth=a_depth,
            data_format="zn" if transpose_a else "nz",
        )
        self.l1_b = Channel(
            MemLoc.L1,
            (base_n, k_l1) if transpose_b else (k_l1, base_n),
            b_dtype,
            depth=l1_buffers,
            data_format="nz" if transpose_b else "zn",
        )
        self.l1_scale_a = Channel(
            MemLoc.L1,
            (base_m, scale_a_l1_len),
            Float8E8M0,
            depth=a_scale_depth,
            data_format="zn",
        )
        self.l1_scale_b = Channel(
            MemLoc.L1,
            (self.scale_k_l1_len, base_n),
            Float8E8M0,
            depth=2,
            data_format="nz",
        )
        self.l0a = Channel(MemLoc.L0A, (base_m, base_k), a_dtype, depth=2)
        self.l0b = Channel(MemLoc.L0B, (base_n, base_k), b_dtype, depth=2)
        self.l0c = Channel(
            MemLoc.L0C, (base_m, base_n), dtypes.float32, depth=l0c_buffers
        )
        if const_expr(self.has_bias):
            self.l1_bias = Channel(
                MemLoc.L1, (base_n,), dtypes.float32, depth=2, data_format="nd"
            )
            self.l1_bt = Channel(
                MemLoc.BIAS, (base_n,), dtypes.float32, depth=2, data_format="nd"
            )
            self.bias_nd_engine = make_copy_engine(
                format_transform="identity", dtype=dtypes.float32, pad_value=0.0
            )
        self.copy_a_gm_to_l1 = make_copy_engine(
            format_transform="dn2nz" if transpose_a else "nd2nz",
            dtype=a_dtype,
            pad_value=0.0,
        )
        self.copy_b_gm_to_l1 = make_copy_engine(
            format_transform="nd2nz" if transpose_b else "dn2nz",
            dtype=b_dtype,
            pad_value=0.0,
        )
        self.copy_scale_a_gm_to_l1 = make_copy_engine(
            format_transform="mx_scale_adn" if transpose_a else "mx_scale_and",
            dtype=Float8E8M0,
            pad_value=0.0,
        )
        self.copy_scale_b_gm_to_l1 = make_copy_engine(
            format_transform="mx_scale_bdn" if transpose_b else "mx_scale_bnd",
            dtype=Float8E8M0,
            pad_value=0.0,
        )
        self.fp = make_copy_engine(dtype=dtypes.float32, unit_flag_mode=3)

        # Multi-core scheduling: full_load assigns fixed M tiles across cores;
        # normal mode uses ASW snake ordering with 4-row sliding window.
        block_idx, block_num = get_block_idx(), get_block_num()
        m_tile_count = self.m_tiles
        n_tile_count = self.n_tiles

        if const_expr(self.full_load):
            # Keep A/ScaleA for one M tile resident and reuse them across N tiles.
            full_m_idx = block_idx % self.m_tiles
            n_lane = block_idx // self.m_tiles
            n_stride = self.block_dim // self.m_tiles
            if const_expr(self.transpose_a):
                a_tile = tile_view(
                    a_gm, (ceil_align(self.k, MX_K_ALIGN), self.base_m), (0, full_m_idx)
                )
            else:
                a_tile = tile_view(
                    a_gm, (self.base_m, ceil_align(self.k, MX_K_ALIGN)), (full_m_idx, 0)
                )
            mem_copy(
                self.l1_a,
                a_tile,
                engine=self.copy_a_gm_to_l1,
                l2_cache_ctl=self.l2_cache_ctl_a,
            )
            if const_expr(self.transpose_a):
                scale_a_tile = tile_view(
                    scale_a_gm,
                    (self.scale_k_len // MX_SCALE_PAIR, self.base_m, MX_SCALE_PAIR),
                    (0, full_m_idx, 0),
                )
            else:
                scale_a_tile = tile_view(
                    scale_a_gm,
                    (self.base_m, self.scale_k_len // MX_SCALE_PAIR, MX_SCALE_PAIR),
                    (full_m_idx, 0, 0),
                )
            mem_copy(
                self.l1_scale_a,
                scale_a_tile,
                engine=self.copy_scale_a_gm_to_l1,
                l2_cache_ctl=1,
            )
            for full_n_idx in range(n_lane, n_tile_count, n_stride):
                self._compute_output_tile(
                    out_gm,
                    a_gm,
                    b_gm,
                    scale_a_gm,
                    scale_b_gm,
                    full_m_idx,
                    full_n_idx,
                    bias_gm,
                )
        else:
            for logical_idx in range(block_idx, m_tile_count * n_tile_count, block_num):
                # Each ASW window contains up to four M tiles; handle the tail separately.
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
                    m_idx = (
                        self.main_row * self.main_window + tail_idx % self.tail_window
                    )
                    n_forward = (tail_idx // self.tail_window) % self.n_tiles
                reverse = row_idx % 2
                n_idx = n_forward + reverse * ((self.n_tiles - 1) - 2 * n_forward)
                self._compute_output_tile(
                    out_gm,
                    a_gm,
                    b_gm,
                    scale_a_gm,
                    scale_b_gm,
                    m_idx,
                    n_idx,
                    bias_gm,
                )

    @jit
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

_TORCH_INPUT_TO_ASL = {torch.float8_e4m3fn: Float8E4M3FN, torch.float8_e5m2: Float8E5M2}
_TORCH_OUTPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _infer_qbmm_mxfp8_layout(x1, x2, scale_a, scale_b):
    """Infer M/N/K and transpose flags from X1/X2/Scale view shapes.

    Do not rewrite the input tensors. Return
    ``(M, N, K, transposeX1, transposeX2)``.
    """
    if scale_a.stride(-1) != 1:
        raise ValueError("pertoken_scale paired lane must have unit stride")
    if scale_b.stride(-1) != 1:
        raise ValueError("scale paired lane must have unit stride")

    matches = []
    x1_shape = tuple(x1.shape)
    x2_shape = tuple(x2.shape)
    scale_a_shape = tuple(scale_a.shape)
    scale_b_shape = tuple(scale_b.shape)
    for transpose_a in (False, True):
        m = x1_shape[1] if transpose_a else x1_shape[0]
        k = x1_shape[0] if transpose_a else x1_shape[1]
        group_count = ceil_div(k, MX_K_ALIGN)
        expected_scale_a = (
            (group_count, m, MX_SCALE_PAIR)
            if transpose_a
            else (m, group_count, MX_SCALE_PAIR)
        )
        if scale_a_shape != expected_scale_a:
            continue
        for transpose_b in (False, True):
            x2_k = x2_shape[1] if transpose_b else x2_shape[0]
            n = x2_shape[0] if transpose_b else x2_shape[1]
            if x2_k != k:
                continue
            expected_scale_b = (
                (n, group_count, MX_SCALE_PAIR)
                if transpose_b
                else (group_count, n, MX_SCALE_PAIR)
            )
            if scale_b_shape == expected_scale_b:
                matches.append((m, n, k, transpose_a, transpose_b))

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "cannot infer MXFP8 layout from view shape: "
            f"x1={x1_shape}, x2={x2_shape}, "
            f"pertoken_scale={scale_a_shape}, scale={scale_b_shape}"
        )
    raise ValueError(
        "ambiguous MXFP8 view shape; please use non-ambiguous "
        f"x1/x2/scale shapes, candidates={matches}"
    )


def npu_quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    *,
    offset: torch.Tensor = None,
    pertoken_scale: torch.Tensor = None,
    bias: torch.Tensor = None,
    output_dtype=None,
    x1_dtype=None,
    x2_dtype=None,
    pertoken_scale_dtype=None,
    scale_dtype=None,
    y_scale: torch.Tensor = None,
) -> torch.Tensor:
    """Execute a two-dimensional, fully quantized MXFP8 matrix multiply.

    The argument order matches ``torch_npu.npu_quant_matmul``: ``scale`` is
    X2Scale and ``pertoken_scale`` is X1Scale.
    """
    if x1.dtype not in _TORCH_INPUT_TO_ASL or x2.dtype not in _TORCH_INPUT_TO_ASL:
        raise TypeError(f"x1/x2 must be FP8 tensors, got {x1.dtype} and {x2.dtype}")
    if x1_dtype is not None or x2_dtype is not None:
        raise NotImplementedError("x1_dtype/x2_dtype overrides select non-MXFP8 modes")
    if offset is not None or y_scale is not None:
        raise NotImplementedError("MXFP8 requires offset/y_scale=None")
    has_bias = bias is not None
    if pertoken_scale is None:
        raise ValueError("MXFP8 requires pertoken_scale as x1Scale")
    if output_dtype not in _TORCH_OUTPUT_DTYPES:
        raise ValueError(f"MXFP8 requires a supported output_dtype, got {output_dtype}")
    if x1.dim() != 2 or x2.dim() != 2:
        raise NotImplementedError(
            f"MXFP8 batch dims not implemented, got rank {x1.dim()} and {x2.dim()}"
        )
    if pertoken_scale.dim() != 3 or scale.dim() != 3:
        raise ValueError("MXFP8 ScaleA/ScaleB must use rank-3 paired layouts")
    if any(t.device != x1.device for t in (x1, x2, pertoken_scale, scale)):
        raise ValueError("x1, x2 and both MX scales must be on the same device")
    if pertoken_scale_dtype is not None or scale_dtype is not None:
        raise NotImplementedError("MXFP8 only accepts native FLOAT8_E8M0 scales")
    if (
        pertoken_scale.dtype != torch.float8_e8m0fnu
        or scale.dtype != torch.float8_e8m0fnu
    ):
        raise TypeError("pertoken_scale and scale must be FLOAT8_E8M0 tensors")

    m, n, k, transpose_a, transpose_b = _infer_qbmm_mxfp8_layout(
        x1, x2, pertoken_scale, scale
    )
    if has_bias:
        if bias.dtype != torch.float32:
            raise TypeError(f"bias must be float32, got {bias.dtype}")
        if bias.shape[0] != n:
            raise ValueError(f"bias length {bias.shape[0]} != N={n}")
    output = torch.empty((m, n), dtype=output_dtype, device=x1.device)

    a_runtime = x1.view(torch.int8)
    b_runtime = x2.view(torch.int8)
    scale_a_runtime = pertoken_scale.view(torch.int8)
    scale_b_runtime = scale.view(torch.int8)
    enable_l0c_pingpong = (
        output_dtype in (torch.float16, torch.bfloat16)
        and k % 128 == 0
        and _is_cube_bound(m, n, k, 2)
    )
    (
        base_m,
        base_n,
        base_k,
        k_l1,
        scale_k_l1,
        l1_buf,
        l0c_buf,
        used_core,
        full_load,
    ) = get_qbmm_mxfp8_tile_config(
        m,
        n,
        k,
        enable_l0c_pingpong=enable_l0c_pingpong,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        has_bias=has_bias,
    )
    bias_kernel = (
        bias if has_bias else torch.empty(0, dtype=torch.float32, device=x1.device)
    )
    kernel_obj = QbmmMxfp8Kernel(
        m,
        n,
        k,
        base_m,
        base_n,
        base_k,
        k_l1,
        scale_k_l1,
        l1_buffers=l1_buf,
        l0c_buffers=l0c_buf,
        a_dtype=_TORCH_INPUT_TO_ASL[x1.dtype],
        b_dtype=_TORCH_INPUT_TO_ASL[x2.dtype],
        full_load=full_load,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        used_core_num=used_core,
        has_bias=has_bias,
    )
    kernel_obj.run(
        output,
        a_runtime,
        b_runtime,
        scale_a_runtime,
        scale_b_runtime,
        bias_kernel,
    )
    return output
