# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Precision tests for the MXFP8 QBMM kernel.

Formula: Y[M,N] = dequant(X1)[M,K] @ dequant(X2)[K,N]

The golden path decodes the public rank-3 E8M0 paired-scale tensors on CPU.
It does not reuse the kernel's GM-to-L1 scale transformation.
"""

from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "samples",
        "matmul",
        "quant_matmul",
    ),
)

from quant_batch_matmul_mxfp8 import (  # noqa: E402
    MX_GROUP_SIZE,
    MX_SCALE_PAIR,
    ceil_div,
    get_scale_k_len,
    npu_quant_matmul,
)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


_E4 = torch.float8_e4m3fn
_E5 = torch.float8_e5m2

_TEST_CASES = [
    pytest.param(
        256, 256, 256, _E4, _E4, torch.float32, False, True, False, id="aligned-ft"
    ),
    pytest.param(
        130, 130, 70, _E4, _E4, torch.float32, False, True, False, id="mnk-tail-ft"
    ),
    pytest.param(
        15, 256, 192, _E4, _E4, torch.float32, False, True, False, id="m-tail-ft"
    ),
    pytest.param(
        1536, 31, 384, _E4, _E4, torch.float32, False, True, False, id="n-tail-ft"
    ),
    pytest.param(
        15, 256, 192, _E4, _E4, torch.float32, False, False, False, id="m-tail-ff"
    ),
    pytest.param(
        1536, 31, 384, _E4, _E4, torch.float32, False, False, False, id="n-tail-ff"
    ),
    pytest.param(
        31, 64, 320, _E4, _E4, torch.float32, False, False, False, id="narrow-m-tail-ff"
    ),
    pytest.param(
        64, 16384, 256, _E4, _E4, torch.float32, False, True, False, id="a-full-load-ft"
    ),
    pytest.param(128, 128, 128, _E4, _E4, torch.float32, False, False, False, id="ff"),
    pytest.param(128, 128, 128, _E4, _E4, torch.float32, True, False, False, id="tf"),
    pytest.param(128, 128, 128, _E4, _E4, torch.float32, True, True, False, id="tt"),
    pytest.param(
        256, 256, 256, _E4, _E5, torch.bfloat16, False, True, False, id="e4-e5-bf16"
    ),
    pytest.param(
        256, 256, 256, _E5, _E4, torch.float16, False, True, False, id="e5-e4-fp16"
    ),
    # K non-aligned to MX_K_ALIGN(64): group tail + intra-group tail
    pytest.param(
        128,
        128,
        96,
        _E4,
        _E4,
        torch.float32,
        False,
        False,
        False,
        id="k-96-group-tail-ff",
    ),
    pytest.param(
        256,
        256,
        70,
        _E4,
        _E4,
        torch.float32,
        False,
        True,
        False,
        id="k-70-intra-group-tail-ft",
    ),
    # Large K non-aligned: kL1=256 with remainder, scaleKL1 reuse across kL1 windows
    pytest.param(
        1024,
        1024,
        1056,
        _E4,
        _E4,
        torch.float32,
        False,
        False,
        False,
        id="kl1-remainer-ff",
    ),
    pytest.param(
        1024,
        1024,
        2048,
        _E4,
        _E4,
        torch.float32,
        False,
        False,
        False,
        id="scalekl1-full-cover-ff",
    ),
    # Large K + M/N half-tail: exercises kL1 + M/N tail simultaneously
    pytest.param(
        960,
        1088,
        1024,
        _E4,
        _E4,
        torch.float32,
        False,
        False,
        False,
        id="kl1-mn-tail-ff",
    ),
    pytest.param(
        1088,
        960,
        1024,
        _E4,
        _E4,
        torch.float32,
        False,
        True,
        False,
        id="kl1-mn-tail-ft",
    ),
    # Unaligned K shape that makes kL1 fall back to two K steps
    pytest.param(
        512,
        512,
        320,
        _E4,
        _E4,
        torch.float32,
        False,
        False,
        False,
        id="k-320-stepk2-ff",
    ),
    # K=4096: scaleKL1 > kL1 (scale cache reuse factor > 1)
    pytest.param(
        256,
        256,
        4096,
        _E4,
        _E4,
        torch.float32,
        False,
        False,
        False,
        id="scalekl1-cache-ff",
    ),
    # Transpose + large non-aligned
    pytest.param(
        256, 15, 1024, _E4, _E4, torch.float32, True, True, False, id="tt-n-tail-kl1"
    ),
    pytest.param(
        15, 256, 1024, _E4, _E4, torch.float32, True, False, False, id="tf-m-tail-kl1"
    ),
    # Bias smoke cases
    pytest.param(
        256, 256, 256, _E4, _E4, torch.float32, False, True, True, id="bias-ft"
    ),
    pytest.param(
        128, 130, 128, _E4, _E4, torch.float32, False, True, True, id="bias-n-tail-ft"
    ),
    pytest.param(
        64,
        16384,
        256,
        _E4,
        _E4,
        torch.float32,
        False,
        True,
        True,
        id="bias-a-full-load-ft",
    ),
    pytest.param(
        256, 256, 256, _E4, _E5, torch.bfloat16, False, True, True, id="bias-e4-e5-bf16"
    ),
    pytest.param(
        256, 256, 256, _E5, _E4, torch.float16, False, True, True, id="bias-e5-e4-fp16"
    ),
]

# ---------------------------------------------------------------------------
# Golden kit
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class QbmmMxfp8Inputs:
    x1: torch.Tensor
    x2: torch.Tensor
    scale_a: torch.Tensor
    scale_b: torch.Tensor
    transpose_a: bool
    transpose_b: bool
    bias: torch.Tensor | None = None


def _make_paired_scale(
    outer_size: int,
    scale_k_len: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Build public E8M0 storage with shape [outer,G,2]."""
    exponents = torch.randint(
        -2,
        3,
        (outer_size, scale_k_len),
        dtype=torch.int16,
        generator=generator,
    )
    return (
        (exponents + 127)
        .to(torch.uint8)
        .view(torch.int8)
        .reshape(outer_size, scale_k_len // MX_SCALE_PAIR, MX_SCALE_PAIR)
        .contiguous()
    )


def make_inputs(
    m: int,
    k: int,
    n: int,
    *,
    transpose_a: bool = False,
    transpose_b: bool = True,
    a_dtype: torch.dtype = _E4,
    b_dtype: torch.dtype | None = None,
    seed: int = 42,
    has_bias: bool = False,
) -> QbmmMxfp8Inputs:
    """Build deterministic X1/X2 and public rank-3 paired-scale tensors."""
    b_dtype = a_dtype if b_dtype is None else b_dtype
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a_logical = (torch.randn((m, k), generator=generator) * 0.5).to(a_dtype)
    b_logical = (torch.randn((n, k), generator=generator) * 0.5).to(b_dtype)
    scale_k_len = get_scale_k_len(k)
    scale_a_and = _make_paired_scale(m, scale_k_len, generator=generator)
    scale_b_bdn = _make_paired_scale(n, scale_k_len, generator=generator)
    bias = (
        (torch.randn((n,), generator=generator) * 0.5).to(torch.float32)
        if has_bias
        else None
    )

    return QbmmMxfp8Inputs(
        x1=a_logical.T.contiguous() if transpose_a else a_logical,
        x2=b_logical if transpose_b else b_logical.T.contiguous(),
        scale_a=(
            scale_a_and.permute(1, 0, 2).contiguous() if transpose_a else scale_a_and
        ),
        scale_b=(
            scale_b_bdn if transpose_b else scale_b_bdn.permute(1, 0, 2).contiguous()
        ),
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        bias=bias,
    )


def _decode_e8m0_scale(scale: torch.Tensor, outer_size: int, k: int) -> torch.Tensor:
    """Decode [outer,G,2] E8M0 bytes to one FP32 value per 32 K elements."""
    valid_scale_k_len = ceil_div(k, MX_GROUP_SIZE)
    exponent_bytes = scale.contiguous().view(torch.uint8).reshape(outer_size, -1)
    exponents = exponent_bytes[:, :valid_scale_k_len].to(torch.int16) - 127
    return torch.pow(2.0, exponents.float())


def qbmm_mxfp8_golden(inputs: QbmmMxfp8Inputs) -> torch.Tensor:
    """CPU golden that decodes paired Scale and performs FP32 matmul."""
    a = inputs.x1.T.contiguous() if inputs.transpose_a else inputs.x1
    b = inputs.x2 if inputs.transpose_b else inputs.x2.T.contiguous()
    m, k = a.shape
    n = b.shape[0]

    scale_a = (
        inputs.scale_a.permute(1, 0, 2).contiguous()
        if inputs.transpose_a
        else inputs.scale_a
    )
    scale_b = (
        inputs.scale_b
        if inputs.transpose_b
        else inputs.scale_b.permute(1, 0, 2).contiguous()
    )
    scale_a = _decode_e8m0_scale(scale_a, m, k)
    scale_b = _decode_e8m0_scale(scale_b, n, k)
    scale_a = scale_a.repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    scale_b = scale_b.repeat_interleave(MX_GROUP_SIZE, dim=1)[:, :k]
    result = (a.float() * scale_a) @ (b.float() * scale_b).T
    if inputs.bias is not None:
        result = result + inputs.bias.to(torch.float32)
    return result


def assert_isclose(
    actual: torch.Tensor,
    golden: torch.Tensor,
) -> None:
    """Compare the NPU result with the independently computed CPU Golden.

    Uses np.isclose(rtol=1e-3, atol=1e-3) element-wise, then requires the
    mismatch ratio to be at most 0.1%.
    """
    atol, rtol = 1e-3, 1e-3
    np_actual = actual.float().cpu().numpy()
    np_golden = golden.float().cpu().numpy()
    match = np.isclose(np_actual, np_golden, rtol=rtol, atol=atol, equal_nan=True)
    mismatch_ratio = 1.0 - float(match.sum() / match.size)
    if mismatch_ratio <= 1e-3:
        return
    max_error = float((actual.float() - golden.float()).abs().max())
    raise AssertionError(
        f"QBMM MXFP8 mismatch: max|err|={max_error:.4e}, "
        f"mismatch={mismatch_ratio:.4%}, "
        f"atol={atol}, rtol={rtol}"
    )


# ---------------------------------------------------------------------------
# NPU accuracy tests
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.parametrize(
    "m,n,k,a_dtype,b_dtype,output_dtype,transpose_a,transpose_b,has_bias",
    _TEST_CASES,
)
def test_quant_batch_matmul_mxfp8(
    m: int,
    n: int,
    k: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    output_dtype: torch.dtype,
    transpose_a: bool,
    transpose_b: bool,
    has_bias: bool,
) -> None:
    """Verify MXFP8 QBMM precision across supported shapes and dtypes."""
    pytest.importorskip("torch_npu")
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("ASCEND_HOME_PATH not set")
    if torch.npu.device_count() == 0:
        pytest.skip("no NPU device is available")
    if "Ascend950" not in torch.npu.get_device_name(0):
        pytest.skip("MXFP8 MMAD requires Ascend950")

    inputs = make_inputs(
        m,
        k,
        n,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        has_bias=has_bias,
    )
    golden = qbmm_mxfp8_golden(inputs).to(output_dtype)
    x1, x2 = inputs.x1.npu(), inputs.x2.npu()
    scale_a = inputs.scale_a.npu().view(torch.float8_e8m0fnu)
    scale_b = inputs.scale_b.npu().view(torch.float8_e8m0fnu)
    bias = inputs.bias.npu() if inputs.bias is not None else None
    actual = npu_quant_matmul(
        x1,
        x2,
        scale_b,
        pertoken_scale=scale_a,
        bias=bias,
        output_dtype=output_dtype,
    )
    torch.npu.synchronize()
    assert_isclose(actual.cpu(), golden)
