# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for the PointNet Set Abstraction kernel.

Formula:
  feat[K, D_out] = max_over_points( MLP( points[K, N_per_group, D_in] ) )

Coverage:
  * fp16 typical PointNet++ SA layer shape
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "samples", "pointnet_sa"))

from pointnet_sa import pointnet_sa


@pytest.mark.npu
def test_pointnet_sa_accuracy():
    pytest.importorskip("torch_npu")

    torch.manual_seed(42)
    K = 512
    N_per_group = 64
    D_in = 64
    D_out = 128

    dtype = torch.float16
    points = torch.randn(K, N_per_group, D_in, dtype=dtype) * 0.1
    weight = torch.randn(D_out, D_in, dtype=dtype) * 0.1

    output = pointnet_sa(points.npu(), weight.npu())
    torch.npu.synchronize()

    mlp_out = points.float() @ weight.float().t()
    expected = mlp_out.max(dim=1).values

    torch.testing.assert_close(
        output.cpu().float(), expected, atol=2e-2, rtol=2e-2
    )
