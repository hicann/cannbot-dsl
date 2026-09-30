# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for the PointNet Set Abstraction kernel.

Formula:
  feat[K, D_out] = max_over_points( MLP( points[K, N_per_group, D_in] ) )
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_samples_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "samples", "pointnet_sa"
)
sys.path.insert(0, _samples_dir)

import cannbotdsl
from cannbotdsl import dtypes

from pointnet_sa import SATiling, SAKernel, pointnet_sa

_CASES = [
    pytest.param(512, 64, 64, 128, torch.float16, id="sa_512x64x64_128"),
    pytest.param(512, 64, 64, 128, torch.bfloat16, id="sa_512x64x64_128_bf16"),
    pytest.param(333, 32, 64, 64, torch.float16, id="odd_centroid_count"),
    pytest.param(64, 32, 128, 256, torch.float16, id="wide_d_out"),
    pytest.param(1024, 64, 32, 48, torch.float16, id="many_n_tiles"),
    pytest.param(16, 16, 96, 96, torch.bfloat16, id="single_tile"),
]

_COMPILE_CASES = [
    pytest.param(512, 64, 64, 128, dtypes.float16, id="fp16_512x64x64_128"),
    pytest.param(333, 32, 64, 64, dtypes.bfloat16, id="bf16_333x32x64_64"),
    pytest.param(1024, 64, 32, 48, dtypes.float16, id="fp16_1024x64x32_48"),
]


def _torch_dtype_of(dsl_dtype):
    return torch.float16 if dsl_dtype is dtypes.float16 else torch.bfloat16


@pytest.mark.cannir_install
@pytest.mark.parametrize(
    "num_centroids,num_points,d_in,d_out,dsl_dtype", _COMPILE_CASES
)
def test_pointnet_sa_compile_matrix(num_centroids, num_points, d_in, d_out, dsl_dtype):
    tiling = SATiling(num_centroids, num_points, d_in, d_out, dtype=dsl_dtype)
    op = SAKernel(tiling)
    cannbotdsl.compile(
        op.run,
        cannbotdsl.TensorSpec((d_in, num_centroids * num_points), dsl_dtype),
        cannbotdsl.TensorSpec((d_out, d_in), dsl_dtype),
        cannbotdsl.TensorSpec((d_out, num_centroids * num_points), dsl_dtype),
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "num_centroids,num_points,d_in,d_out,dtype", _CASES
)
def test_pointnet_sa_accuracy(num_centroids, num_points, d_in, d_out, dtype):
    pytest.importorskip("torch_npu")
    torch.manual_seed(42)

    points = torch.randn(num_centroids, num_points, d_in, dtype=dtype) * 0.1
    weight = torch.randn(d_out, d_in, dtype=dtype) * 0.1

    output = pointnet_sa(points.npu(), weight.npu())
    torch.npu.synchronize()

    mlp_out = points.float() @ weight.float().t()
    expected = mlp_out.max(dim=1).values

    torch.testing.assert_close(output.cpu().float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.npu
def test_pointnet_sa_matches_torch_reference():
    """Cross-check the tiled matmul path against a per-group torch reference."""
    pytest.importorskip("torch_npu")
    torch.manual_seed(0)

    num_centroids, num_points, d_in, d_out = 128, 48, 64, 128
    points = torch.randn(num_centroids, num_points, d_in, dtype=torch.float16) * 0.2
    weight = torch.randn(d_out, d_in, dtype=torch.float16) * 0.2

    output = pointnet_sa(points.npu(), weight.npu())
    torch.npu.synchronize()

    expected = torch.empty(num_centroids, d_out, dtype=torch.float32)
    for i in range(num_centroids):
        expected[i] = (points[i].float() @ weight.float().t()).max(dim=0).values

    torch.testing.assert_close(
        output.cpu().float(), expected, atol=2e-2, rtol=2e-2
    )


def test_pointnet_sa_rejects_channel_mismatch():
    points = torch.empty(8, 4, 64, dtype=torch.float16)
    weight = torch.empty(32, 128, dtype=torch.float16)
    with pytest.raises(ValueError, match="channel mismatch"):
        pointnet_sa(points, weight)


def test_pointnet_sa_rejects_bad_rank():
    points = torch.empty(8, 64, dtype=torch.float16)
    weight = torch.empty(32, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="must be 3-D"):
        pointnet_sa(points, weight)


def test_pointnet_sa_rejects_dtype_mismatch():
    points = torch.empty(8, 4, 64, dtype=torch.float16)
    weight = torch.empty(32, 64, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="must match"):
        pointnet_sa(points, weight)
