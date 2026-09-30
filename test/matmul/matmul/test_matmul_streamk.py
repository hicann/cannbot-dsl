# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for the matmul_streamk kernel (DPSK, transpose_a=False, transpose_b=False).

Formula:  C[M,N] = A[M,K] @ B[K,N]   (fp16/bf16 inputs, fp32 accumulator)

Constraints honoured by the shape list:
  * Inputs are 2-D contiguous matrices.
  * Current kernel path only implements transpose_a=False, transpose_b=False.
"""

from __future__ import annotations

import os

import pytest

import torch

from _samples_path import load_sample

_sample = load_sample("matmul/matmul/matmul_streamk.py")
MatmulStreamK = _sample.MatmulStreamK
matmul_streamk = _sample.matmul_streamk


_TRANSPOSE_A = False
_TRANSPOSE_B = False


# 5 representative shapes (full 41-shape regression list kept locally):
# DP-dominant large, SK-dominant small MN, degenerate 1x1, mixed
# DP+SK with big K, and a tall-aspect grid.
_TEST_SHAPES = [
    pytest.param(1024, 1024, 8192, id="M1024-N1024-K8192"),
    pytest.param(37, 37, 8192, id="M37-N37-K8192"),
    pytest.param(1, 1, 8192, id="M1-N1-K8192"),
    pytest.param(300, 300, 16384, id="M300-N300-K16384"),
    pytest.param(2048, 300, 8192, id="M2048-N300-K8192"),
]


@pytest.mark.npu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("m,n,k", _TEST_SHAPES)
def test_matmul_streamk_precision(m: int, n: int, k: int, dtype: torch.dtype) -> None:
    pytest.importorskip("torch_npu")
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("ASCEND_HOME_PATH not set")
    torch.npu.empty_cache()
    torch.npu.synchronize()

    op = MatmulStreamK(m, n, k, dtype="fp16" if dtype == torch.float16 else "bf16")
    op.print_config()
    a = torch.randn(m, k, dtype=dtype).npu()
    b = torch.randn(k, n, dtype=dtype).npu()
    out = torch.zeros(m, n, dtype=dtype).npu()
    ws = torch.zeros(op.ws_rows, op.ws_cols, dtype=torch.float32).npu()
    op.run(out, ws, a, b)
    torch.npu.synchronize()

    ref = a.cpu().float() @ b.cpu().float()
    err = (out.cpu().float() - ref).abs().max().item()
    ref_max = ref.abs().max().item()
    assert err < 1e-2 * max(abs(ref_max), 1.0), (
        f"DPSK {m}x{n}x{k}: err={err:.4e}, ref_max={ref_max:.4e}"
    )
