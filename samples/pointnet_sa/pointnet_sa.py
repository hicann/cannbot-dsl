# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PointNet Set Abstraction (SA) layer.

Implements the core computation of PointNet++ (Qi et al., 2017):
  1. Shared MLP on each point within grouped local regions (1x1 conv)
  2. Max-pooling across points to produce per-region features

Structure:
  1. SATiling       - host-side tiling (maps shared MLP to matmul)
  2. SAKernel       - @kernel with multi-core slide window scheduling
  3. pointnet_sa()  - torch interface

Formula:
  For each of K sampled centroids, group neighboring points (up to N_per_group):
    feat[K, D_out] = max_over_points( MLP( points[K, N_per_group, D_in] ) )

  The MLP is a 1x1 convolution (weights shared across all points),
  implemented as batched matmul:
    reshaped_points[D_in, K*N_per_group] @ weight[D_out, D_in]^T -> [D_out, K*N_per_group]
  followed by max-pool along N_per_group dimension.
"""

import math
from cannbotdsl.arch import get_block_idx, get_block_num
from cannbotdsl.channel import Channel
from cannbotdsl.dtypes import float16, float32, bfloat16
from cannbotdsl.integer import Int64
from cannbotdsl.jit_runner import jit
from cannbotdsl.kernel_launcher import kernel
from cannbotdsl.math import matmul
from cannbotdsl.tensor import tile_view, make_copy_engine, mem_copy
from cannbotdsl.typing.types import MemLoc, Tensor


L2_CACHE_DEFAULT = 0
A_L2_CACHE_DISABLE = 1
B_L2_CACHE_DISABLE = 2
ALL_L2_CACHE_DISABLE = 3


class SATiling:
    """Host-side tiling for the shared MLP (1x1 conv) in Set Abstraction.

    Maps: M=D_out, N=K*num_points_per_group, K_dim=D_in
    Tiling logic mirrors Conv2dTiling / MatmulTiling.
    """

    L1_SIZE = 512 * 1024
    L0A_SIZE = 64 * 1024
    L0C_SIZE = 256 * 1024
    L2_SIZE = 128 * 1024 * 1024
    AIC_NUM = 32

    BASIC_BLOCK_16 = 16
    BASIC_BLOCK_128 = 128
    BASIC_BLOCK_256 = 256
    DB_SIZE = 2
    BASIC_L1_BUFFER_NUM = 4
    DATA_SIZE_FP32 = 4
    BASIC_BLOCK_K_128B = 128
    BASIC_BLOCK_K_256B = 256
    BASIC_BLOCK_K_512B = 512
    L1_SINGLE_SIZE_LIMIT = 48 * 1024
    MIN_TAIL_BLOCK_SIZE = 4096
    CUBE_BOUND_RATIO = 0.85
    EPSILON = 1e-9
    DEFAULT_CORE_FREQ = 1.65
    DEFAULT_DDR_RATE = 31
    DEFAULT_L2_RATE = 100
    ALIGN_128 = 128
    _DTYPE_PROPS = {
        "bfloat16": (bfloat16, 2),
        "float16": (float16, 2),
    }

    def __init__(self, num_centroids, num_points, d_in, d_out, dtype=float16):
        self.num_centroids = num_centroids
        self.num_points = num_points
        self.d_in = d_in
        self.d_out = d_out

        self.m = d_out
        self.n = num_centroids * num_points
        self.k = d_in

        dtype_name = getattr(dtype, "__name__", str(dtype).rsplit(".", 1)[-1]).lower()
        if dtype_name not in self._DTYPE_PROPS:
            raise TypeError(f"Unsupported dtype: {dtype}")
        self.a_dtype, self.a_dtype_size = self._DTYPE_PROPS[dtype_name]
        self.b_dtype = self.a_dtype
        self.b_dtype_size = self.a_dtype_size
        self.c_dtype = self.a_dtype
        self.c_dtype_size = self.a_dtype_size

        self.transpose_a = False
        self.transpose_b = True
        self._compute()

    def _compute(self):
        self._reset_base()
        self._rebalance_block()
        self._cal_l1_tiling()
        self._finalize()

    def _reset_base(self):
        self.baseM = min(self.BASIC_BLOCK_256, self._ceil_align(self.m, self.BASIC_BLOCK_16))
        self.baseN = min(self.BASIC_BLOCK_256, self._ceil_align(self.n, self.BASIC_BLOCK_16))
        self.baseK = self.BASIC_BLOCK_K_128B // self.a_dtype_size

    def _rebalance_block(self):
        hbm_bw = self._get_hbm_bw()
        l2_bw = self._get_l2_bw()
        single_core_compute_power = self.DEFAULT_CORE_FREQ * 8
        compute_power = single_core_compute_power * self.AIC_NUM

        base_mn_buffer_limit = self.L0C_SIZE
        l2_cache_usage = max(
            (self.m + self.n) * self.k * self.a_dtype_size / self.L2_SIZE, 1.0
        )
        cmr = (self.m + self.n) / (self.m * self.n)
        self.cubeBoundEdge = (
            (l2_bw / compute_power)
            + l2_cache_usage * (1 - l2_bw / hbm_bw) * cmr
            - (1 + l2_bw / hbm_bw) / self.k
        )

        base_m_best = min(self._ceil_align(self.m, self.BASIC_BLOCK_16), self.BASIC_BLOCK_256)
        base_n_best = max(
            self.BASIC_BLOCK_16,
            min(
                self._ceil_align(self.n, self.BASIC_BLOCK_16),
                self._floor_align(base_mn_buffer_limit // self.DATA_SIZE_FP32 // base_m_best, self.BASIC_BLOCK_16),
            ),
        )
        cube_bound_param_best = (1.0 / base_m_best) + (1.0 / base_n_best)
        is_memory_bound = cube_bound_param_best > self.cubeBoundEdge
        inner_align_unit = self.BASIC_BLOCK_128 if is_memory_bound else self.BASIC_BLOCK_K_128B // self.a_dtype_size

        fixp_bound_edge = (self.m * self.n * hbm_bw) / ((self.m + self.n) * l2_bw)
        base_m_align_unit = self.BASIC_BLOCK_16
        base_n_align_unit = (
            self.BASIC_BLOCK_K_256B // self.b_dtype_size
            if self.k < fixp_bound_edge
            else self.BASIC_BLOCK_16
        )

        max_base_m = self._get_max_base_with_limit(
            base_mn_buffer_limit, base_m_align_unit, is_right_matrix=False, is_memory_bound=is_memory_bound
        )
        max_base_n = self._get_max_base_with_limit(
            base_mn_buffer_limit, base_n_align_unit, is_right_matrix=True, is_memory_bound=is_memory_bound
        )

        self.baseM = max(self.BASIC_BLOCK_16, min(max_base_m, self.BASIC_BLOCK_256))
        self.baseN = max(
            self.BASIC_BLOCK_16,
            min(
                max_base_n,
                self._floor_align(base_mn_buffer_limit // self.DATA_SIZE_FP32 // self.baseM, base_n_align_unit),
            ),
        )
        self.cubeBoundParam = (1.0 / self.baseM) + (1.0 / self.baseN)
        self.cubeBoundEdge *= self.CUBE_BOUND_RATIO
        balance_rate = self._get_balance_rate_with_tail(self.baseM, self.baseN)

        cur_base_m = max_base_m
        while cur_base_m >= 1:
            cur_max_base_n = min(
                max_base_n,
                self._floor_align(base_mn_buffer_limit // self.DATA_SIZE_FP32 // cur_base_m, base_n_align_unit),
            )
            cur_base_n = cur_max_base_n
            while cur_base_n >= 1:
                cur_cube_bound_param = (1.0 / cur_base_m) + (1.0 / cur_base_n)
                cur_balance_rate = self._get_balance_rate_with_tail(cur_base_m, cur_base_n)
                skip_cond = (
                    balance_rate >= 0.9
                    and cur_cube_bound_param > self.cubeBoundParam
                    and cur_cube_bound_param > self.cubeBoundEdge
                    and self.cubeBoundEdge > 0
                )
                if not skip_cond:
                    cube_bound_cond = cur_cube_bound_param <= self.cubeBoundEdge and cur_balance_rate > balance_rate
                    balance_cond = (
                        (cur_cube_bound_param / cur_balance_rate) < (self.cubeBoundParam / balance_rate)
                        or (
                            abs(cur_cube_bound_param / cur_balance_rate - self.cubeBoundParam / balance_rate)
                            < self.EPSILON
                            and cur_balance_rate > balance_rate
                        )
                    )
                    if cube_bound_cond or balance_cond:
                        if cube_bound_cond:
                            self.cubeBoundEdge = cur_cube_bound_param
                        self.baseM = cur_base_m
                        self.baseN = cur_base_n
                        self.cubeBoundParam = cur_cube_bound_param
                        balance_rate = cur_balance_rate
                cur_base_n -= base_n_align_unit
            cur_base_m -= base_m_align_unit

        self.baseM = min(self._ceil_align(self.m, self.BASIC_BLOCK_16), self.baseM)
        self.baseN = min(self._ceil_align(self.n, self.BASIC_BLOCK_16), self.baseN)
        self._get_base_k()

        m_core = self._ceil_div(self.m, self.baseM)
        n_core = self._ceil_div(self.n, self.baseN)
        self.mn_core = m_core * n_core
        self.usedCoreNum = min(self.mn_core, self.AIC_NUM)
        self.l0cDB = (
            self.DB_SIZE
            if self.baseM * self.baseN * self.DATA_SIZE_FP32 * self.DB_SIZE <= self.L0C_SIZE
            else 1
        )

    def _get_base_k(self):
        k_align = self._ceil_align(self.k, self.BASIC_BLOCK_16)
        max_base_k = self.L0A_SIZE // self.DB_SIZE // self.a_dtype_size // max(self.baseM, self.baseN)
        if k_align <= max_base_k:
            self.baseK = k_align
        elif max_base_k * self.a_dtype_size >= self.BASIC_BLOCK_K_256B:
            k_256_elems = self.BASIC_BLOCK_K_256B // self.a_dtype_size
            self.baseK = self._floor_align(max_base_k, k_256_elems)
        else:
            self.baseK = self.BASIC_BLOCK_16
            for cand in [128, 64, 32, 16]:
                if max_base_k >= cand:
                    self.baseK = cand
                    break

    def _cal_l1_tiling(self):
        total_l1 = self.L1_SIZE
        max_step_k = min(self._ceil_div(self.k, self.baseK), 8)

        k_256_elems = self.BASIC_BLOCK_K_256B // self.a_dtype_size
        k_align_unit = self.BASIC_BLOCK_K_512B // self.a_dtype_size
        res_kl1 = 0
        single_mte_size = 0
        for step_k in range(1, max_step_k + 1):
            cur_kl1 = self.baseK * step_k
            a_l1 = self.baseM * cur_kl1 * self.a_dtype_size
            b_l1 = self.baseN * cur_kl1 * self.b_dtype_size
            if (a_l1 + b_l1) * self.DB_SIZE > total_l1:
                break
            if max(a_l1, b_l1) * self.DB_SIZE * 2 > self.L1_SIZE:
                break
            cond_no_res = res_kl1 == 0
            cond_k_align_256b = cur_kl1 % k_256_elems == 0
            cond_k_align = (
                res_kl1 % k_align_unit != 0
                and (cond_k_align_256b or (not cond_k_align_256b and single_mte_size < self.L1_SINGLE_SIZE_LIMIT))
            )
            cond_mte_size = (
                res_kl1 % k_align_unit == 0
                and cur_kl1 % k_align_unit == 0
                and single_mte_size < self.L1_SINGLE_SIZE_LIMIT
            )
            if cond_no_res or cond_k_align or cond_mte_size:
                res_kl1 = cur_kl1
                single_mte_size = max(a_l1, b_l1)

        self.stepKa = max(1, res_kl1 // self.baseK)
        self.stepKb = max(1, res_kl1 // self.baseK)

    def _set_disable_l2cache(self, mL1, kaL1, kbL1, nL1):
        total_size = (
            self.m * self.n * self.c_dtype_size
            + self.m * self.k * self.a_dtype_size
            + self.k * self.n * self.b_dtype_size
        )
        if total_size < self.L2_SIZE:
            return L2_CACHE_DEFAULT

        inner_a = self.k
        inner_b = self.k
        flag_a = kaL1 * self.a_dtype_size % self.ALIGN_128 == 0
        flag_b = kbL1 * self.b_dtype_size % self.ALIGN_128 == 0

        m_cnt = self._ceil_div(self.m, self.baseM)
        n_cnt = self._ceil_div(self.n, self.baseN)

        left_not_l2 = self.baseN >= self.n and n_cnt <= 1 and inner_a * self.a_dtype_size % self.ALIGN_128 == 0 and flag_a
        right_not_l2 = self.baseM >= self.m and m_cnt <= 1 and inner_b * self.b_dtype_size % self.ALIGN_128 == 0 and flag_b

        if left_not_l2 and right_not_l2:
            return ALL_L2_CACHE_DISABLE
        elif left_not_l2:
            return A_L2_CACHE_DISABLE
        elif right_not_l2:
            return B_L2_CACHE_DISABLE
        return L2_CACHE_DEFAULT

    def _finalize(self):
        step_m = 1
        step_n = 1
        self.mL1 = min(self._ceil_align(self.m, self.BASIC_BLOCK_16), self.baseM * step_m)
        self.nL1 = min(self._ceil_align(self.n, self.BASIC_BLOCK_16), self.baseN * step_n)
        step_ka = min(self.stepKa, self.stepKb, self.BASIC_L1_BUFFER_NUM)
        self.kL1 = self.baseK * step_ka

        m_core = self._ceil_div(self.m, self.baseM)
        n_core = self._ceil_div(self.n, self.baseN)
        self.mn_core = m_core * n_core

        a_l1_4buf = self.kL1 * self.baseM * self.a_dtype_size * self.BASIC_L1_BUFFER_NUM
        b_l1_4buf = self.kL1 * self.baseN * self.b_dtype_size * self.BASIC_L1_BUFFER_NUM
        self.l1BufferNum = self.DB_SIZE if (a_l1_4buf + b_l1_4buf) > self.L1_SIZE else self.BASIC_L1_BUFFER_NUM

        self.l0cDB = self.DB_SIZE if self.baseM * self.baseN * self.DATA_SIZE_FP32 * self.DB_SIZE <= self.L0C_SIZE else 1
        self.l2CacheDisable = self._set_disable_l2cache(self.mL1, self.kL1, self.kL1, self.nL1)

    def _get_max_base_with_limit(self, base_mn_buffer_limit, base_align_unit, is_right_matrix, is_memory_bound):
        shape_value = self.n if is_right_matrix else self.m
        k_align_value = self._ceil_align(self.k, self.BASIC_BLOCK_16)
        k_limit_value = self.BASIC_BLOCK_16 if is_memory_bound else self.BASIC_BLOCK_K_128B // self.a_dtype_size
        min_kl0 = min(k_limit_value, k_align_value) * self.a_dtype_size

        max_base_mn_with_buffer = base_mn_buffer_limit // self.DATA_SIZE_FP32 // self.BASIC_BLOCK_16
        max_base_block = min(self.L0A_SIZE // self.DB_SIZE // min_kl0, max_base_mn_with_buffer)

        k_align_unit = (self.BASIC_BLOCK_K_256B if is_memory_bound else self.BASIC_BLOCK_K_512B) // self.a_dtype_size
        max_base_mn_with_k_inner = self.L1_SIZE // (2 * self.DB_SIZE * self.a_dtype_size * min(k_align_unit, k_align_value))
        max_base_block = min(max_base_block, max_base_mn_with_k_inner)
        max_base_block = min(self._ceil_align(shape_value, base_align_unit), self._floor_align(max_base_block, base_align_unit))
        if shape_value < base_align_unit:
            max_base_block = min(max_base_block, self._ceil_align(shape_value, self.BASIC_BLOCK_16))
        return max_base_block

    def _get_balance_rate_with_tail(self, base_m, base_n):
        total_round = self._ceil_div(self.m, base_m) * self._ceil_div(self.n, base_n)
        main_round = self._ceil_div(total_round, self.AIC_NUM) - 1
        tail_round_blocks = total_round - self.AIC_NUM * main_round
        total_tail_split = self.AIC_NUM // tail_round_blocks if tail_round_blocks else 1

        eff_base_m = self.m if self.m <= self.BASIC_BLOCK_16 else base_m
        eff_base_n = self.n if self.n <= self.BASIC_BLOCK_16 else base_n
        if main_round == 0 or (eff_base_m * eff_base_n) // total_tail_split < self.MIN_TAIL_BLOCK_SIZE:
            return (self.m * self.n / self.AIC_NUM) / ((main_round + 1) * eff_base_m * eff_base_n)

        tail_split_sqrt = int(math.sqrt(total_tail_split))
        offset = (total_tail_split - tail_split_sqrt * tail_split_sqrt) // tail_split_sqrt + 1
        tail_round = 1.0 / (tail_split_sqrt * (tail_split_sqrt + offset - 1))
        return (self.m * self.n / self.AIC_NUM) / ((main_round + tail_round) * eff_base_m * eff_base_n)

    def _get_hbm_bw(self):
        return self.DEFAULT_CORE_FREQ * self.AIC_NUM * self.DEFAULT_DDR_RATE / 1024

    def _get_l2_bw(self):
        return self.DEFAULT_CORE_FREQ * self.AIC_NUM * self.DEFAULT_L2_RATE / 1024

    @staticmethod
    def _ceil_div(a, b):
        return (a + b - 1) // b

    @staticmethod
    def _ceil_align(a, b):
        return (a + b - 1) // b * b

    @staticmethod
    def _floor_align(a, b):
        return a // b * b


WINDOW_LEN = 4


class SAKernel:
    """Set Abstraction shared MLP kernel: maps to 2D matmul + max-pool.

    weight_2d[D_out, D_in], points_2d[D_in, K*N_per_group]
      -> mlp_out[D_out, K*N_per_group] via matmul
      -> reshape to [D_out, K, N_per_group] -> max over N_per_group -> [D_out, K]

    Data flow: GM -> L1 (MTE2, nd2nz) -> L0A/L0B (MTE1) -> MMAD (M) -> L0C -> GM (FIXPIPE)
    """

    def __init__(self, tiling: SATiling):
        self.t = tiling
        self.m_tiles = math.ceil(tiling.m / tiling.mL1)
        self.n_tiles = math.ceil(tiling.n / tiling.nL1)
        self.k_l1_tiles = math.ceil(tiling.k / tiling.kL1)
        self.k_l0_per_l1 = math.ceil(tiling.kL1 / tiling.baseK)
        self.main_window = min(WINDOW_LEN, self.m_tiles)
        self.main_row = self.m_tiles // self.main_window - 1
        self.tail_window = self.m_tiles - self.main_row * self.main_window
        self.total_tiles = self.m_tiles * self.n_tiles

        disable_a = tiling.l2CacheDisable in (A_L2_CACHE_DISABLE, ALL_L2_CACHE_DISABLE)
        disable_b = tiling.l2CacheDisable in (B_L2_CACHE_DISABLE, ALL_L2_CACHE_DISABLE)
        self.l2_cache_ctl_a = 0 if disable_a else 1
        self.l2_cache_ctl_b = 0 if disable_b else 1

        self.used_core_num = min(self.total_tiles, tiling.AIC_NUM)

    @kernel
    def sa_kernel(self, gm_points: Tensor, gm_weight: Tensor, gm_mlp_out: Tensor):
        t = self.t
        l1_a = Channel(MemLoc.L1, shape=(t.baseM, t.kL1), dtype=t.a_dtype, depth=t.l1BufferNum)
        l1_b = Channel(MemLoc.L1, shape=(t.baseN, t.kL1), dtype=t.b_dtype, depth=t.l1BufferNum)
        l0a = Channel(MemLoc.L0A, shape=(t.baseM, t.baseK), dtype=t.a_dtype, depth=2)
        l0b = Channel(MemLoc.L0B, shape=(t.baseN, t.baseK), dtype=t.b_dtype, depth=2)
        l0c = Channel(MemLoc.L0C, shape=(t.baseM, t.baseN), dtype=float32, depth=t.l0cDB)

        nd2nz_engine_a = make_copy_engine(format_transform="nd2nz", dtype=t.a_dtype, pad_value=0.0)
        nd2nz_engine_b = make_copy_engine(format_transform="nd2nz", dtype=t.b_dtype, pad_value=0.0)

        n_tiles = self.n_tiles
        main_window = self.main_window
        main_row = self.main_row
        tail_window = self.tail_window
        total_tiles = self.total_tiles
        k_l1_tiles = self.k_l1_tiles
        k_l0_per_l1 = self.k_l0_per_l1

        block_idx = get_block_idx()
        block_num = get_block_num()

        for tile_idx in range(block_idx, total_tiles, block_num):
            m_idx = Int64(0)
            n_idx = Int64(0)
            row_idx = tile_idx // n_tiles // main_window
            if row_idx < main_row:
                m_idx = row_idx * main_window + tile_idx % main_window
                n_idx = (tile_idx // main_window) % n_tiles
            else:
                row_idx = Int64(main_row)
                tail_index = tile_idx - main_row * main_window * n_tiles
                m_idx = main_row * main_window + tail_index % tail_window
                n_idx = (tail_index // tail_window) % n_tiles

            n_idx = (n_tiles - 1 - n_idx) if (row_idx % 2 != 0) else n_idx

            for k_l1_idx in range(k_l1_tiles):
                gm_a_tile = tile_view(gm_weight, (t.baseM, t.kL1), (m_idx, k_l1_idx))
                gm_b_tile = tile_view(gm_points, (t.baseN, t.kL1), (n_idx, k_l1_idx))
                mem_copy(l1_a, gm_a_tile, engine=nd2nz_engine_a, l2_cache_ctl=self.l2_cache_ctl_a)
                mem_copy(l1_b, gm_b_tile, engine=nd2nz_engine_b, l2_cache_ctl=self.l2_cache_ctl_b)

                for k_l0_idx in range(k_l0_per_l1):
                    l1_a_slice = tile_view(l1_a, (t.baseM, t.baseK), (0, k_l0_idx))
                    l1_b_slice = tile_view(l1_b, (t.baseN, t.baseK), (0, k_l0_idx))
                    mem_copy(l0a, l1_a_slice)
                    mem_copy(l0b, l1_b_slice)
                    global_k = k_l1_idx * k_l0_per_l1 + k_l0_idx
                    matmul(l0c, l0a, l0b, init=(global_k == 0))

            gm_c_tile = tile_view(gm_mlp_out, (t.baseM, t.baseN), (m_idx, n_idx))
            mem_copy(gm_c_tile, l0c)

    @jit
    def run(self, gm_points: Tensor, gm_weight: Tensor, gm_mlp_out: Tensor):
        self.sa_kernel[self.used_core_num](gm_points, gm_weight, gm_mlp_out)


def pointnet_sa(points, weight):
    """Torch-facing wrapper for PointNet Set Abstraction shared MLP + max-pool.

    Args:
      points: (K, N_per_group, D_in) FP16/BF16 NPU tensor
        K = number of sampled centroids
        N_per_group = max points per group (zero-padded)
        D_in = input feature dimension
      weight: (D_out, D_in) FP16/BF16 NPU tensor (shared MLP weights)

    Returns:
      output: (K, D_out) same dtype as input
        Per-centroid aggregated features after shared MLP + max-pool.

    The shared MLP is applied to each point independently (1x1 conv),
    then max-pooling aggregates points within each group.
    """
    import torch
    from cannbotdsl.runtime import from_torch_npu

    K, N_per_group, D_in = points.shape
    D_out = weight.shape[0]
    assert weight.shape[1] == D_in, f"channel mismatch: D_in={D_in} vs weight D_in={weight.shape[1]}"

    points_2d = points.permute(2, 0, 1).contiguous().view(D_in, K * N_per_group).t().contiguous()
    if points_2d.stride()[-1] != 1:
        points_2d = points_2d.view(-1).reshape_as(points_2d)

    dtype = points.dtype
    dsl_dtype = float16 if dtype == torch.float16 else bfloat16

    tiling = SATiling(K, N_per_group, D_in, D_out, dtype=dsl_dtype)
    mlp_out_2d = torch.zeros(D_out, K * N_per_group, dtype=dtype, device=points.device)
    op = SAKernel(tiling)

    gm_points = from_torch_npu(points_2d)
    gm_weight = from_torch_npu(weight)
    gm_mlp_out = from_torch_npu(mlp_out_2d)

    op.run(gm_points, gm_weight, gm_mlp_out)
    torch.npu.synchronize()

    mlp_out = mlp_out_2d.view(D_out, K, N_per_group).permute(1, 2, 0).contiguous()
    output = mlp_out.max(dim=1).values

    return output
