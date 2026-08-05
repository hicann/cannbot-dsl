# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""RmsNorm 编译与精度测试。"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "samples", "rms_norm"))

import cannbotdsl
from rms_norm import RmsNorm, rms_norm

_CASES = [
    pytest.param((20, 300, 4096), (4096,), torch.float16, torch.float16, None, None,
                 id="fp16_20x300x4096"),
    pytest.param((4, 180, 8192), (8192,), torch.float16, torch.float16, None, None,
                 id="fp16_4x180x8192"),
    pytest.param((4, 1, 8192), (8192,), torch.float16, torch.float16, None, None,
                 id="fp16_4x1x8192"),
    pytest.param((1024, 1, 8192), (8192,), torch.float16, torch.float16, None, None,
                 id="fp16_1024x1x8192"),
    pytest.param((512, 1, 8192), (8192,), torch.float16, torch.float16, None, None,
                 id="fp16_512x1x8192"),
    pytest.param((4, 1, 110000), (110000,), torch.float16, torch.float16, (-0.01, -0.001), (-0.01, -0.001),
                 id="fp16_4x1x110000"),
    pytest.param((1024, 1, 12288), (12288,), torch.float32, torch.float32, (-0.01, 0.01), (-0.01, 0.01),
                 id="fp32_1024x1x12288"),
    pytest.param((768, 1, 12288), (12288,), torch.float32, torch.float32, None, None,
                 id="fp32_768x1x12288"),
    pytest.param((1024, 1, 16384), (16384,), torch.float32, torch.float32, None, None,
                 id="fp32_1024x1x16384"),
    pytest.param((768, 18432), (18432,), torch.float32, torch.float32, None, None,
                 id="fp32_768x18432"),
]

_COMPILE_CASES = [
    pytest.param(4096, cannbotdsl.dtypes.float16, "perf", id="perf_fp16"),
    pytest.param(4096, cannbotdsl.dtypes.float32, "perf", id="perf_fp32"),
    pytest.param(12288, cannbotdsl.dtypes.float16, "split", id="split_fp16"),
    pytest.param(12288, cannbotdsl.dtypes.float32, "split", id="split_fp32"),
]

ATOL = 1e-3
RTOL = 1e-3


def _make_inputs(x_shape, norm_shape, dtype, gamma_dtype, x_range, gamma_range, seed=42):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if x_range is not None:
        x = torch.empty(x_shape, dtype=torch.float32).uniform_(x_range[0], x_range[1], generator=gen)
    else:
        x = torch.randn(x_shape, generator=gen, dtype=torch.float32)
    if gamma_range is not None:
        gamma = torch.empty(norm_shape, dtype=torch.float32).uniform_(gamma_range[0], gamma_range[1], generator=gen)
    else:
        gamma = torch.randn(norm_shape, generator=gen, dtype=torch.float32) * 0.2 + 1.0
    return x.to(dtype), gamma.to(gamma_dtype)


def _rms_norm_golden(x, gamma, epsilon):
    input_dtype = x.dtype
    norm_rank = gamma.dim()
    x_fp32 = x.to(torch.float32)
    gamma_fp32 = gamma.to(torch.float32)
    dims = tuple(range(x.dim() - norm_rank, x.dim()))
    var = x_fp32.pow(2).mean(dim=dims, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + epsilon)
    y = (x_fp32 * rstd) * gamma_fp32
    y = y.to(input_dtype)
    return y.to(torch.float32), rstd.to(torch.float32)


@pytest.mark.cannir_install
@pytest.mark.parametrize("num_col,dsl_dtype,route", _COMPILE_CASES)
def test_rms_norm_compile_matrix(num_col, dsl_dtype, route):
    assert RmsNorm._is_row_full_load(num_col) == (route == "perf")
    RmsNorm(dtype=dsl_dtype).run.compile(
        cannbotdsl.TensorSpec((2, num_col), dsl_dtype),
        cannbotdsl.TensorSpec((1, num_col), dsl_dtype),
        cannbotdsl.TensorSpec((2, num_col), dsl_dtype),
        cannbotdsl.TensorSpec((2, 1), cannbotdsl.dtypes.float32),
        cannbotdsl.dtypes.float32,
    )


@pytest.mark.npu
@pytest.mark.parametrize("x_shape,gamma_shape,x_dtype,gamma_dtype,x_range,gamma_range", _CASES)
def test_rms_norm_npu_precision(x_shape, gamma_shape, x_dtype, gamma_dtype, x_range, gamma_range):
    pytest.importorskip("torch_npu")

    x_cpu, gamma_cpu = _make_inputs(x_shape, gamma_shape, x_dtype, gamma_dtype, x_range, gamma_range)
    y_golden, rstd_golden = _rms_norm_golden(x_cpu, gamma_cpu, 1e-6)

    y, rstd = rms_norm(x_cpu.npu(), gamma_cpu.npu(), epsilon=1e-6)
    torch.npu.synchronize()

    y_max_err = (y.cpu().float() - y_golden).abs().max().item()
    rstd_max_err = (rstd.cpu().float() - rstd_golden).abs().max().item()
    assert torch.allclose(y.cpu().float(), y_golden, atol=ATOL, rtol=RTOL), (
        f"y mismatch: x_shape={x_shape}, dtype={x_dtype}, max|err|={y_max_err:.4e}"
    )
    assert torch.allclose(rstd.cpu().float(), rstd_golden, atol=ATOL, rtol=RTOL), (
        f"rstd mismatch: x_shape={x_shape}, dtype={x_dtype}, max|err|={rstd_max_err:.4e}"
    )
