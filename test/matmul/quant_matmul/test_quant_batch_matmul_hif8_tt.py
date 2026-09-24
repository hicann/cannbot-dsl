# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Precision tests for per-tensor quantized matmul.

Formula: C[M,N] = (scaleA * scaleB) * (A[M,K] @ B[K,N])
"""

from __future__ import annotations

import os
import sys

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

from quant_batch_matmul_hif8_tt import npu_quant_matmul  # noqa: E402


_INT8 = "int8"
_FP8 = "fp8"
_HIF8 = "hif8"
_E4 = torch.float8_e4m3fn
_E5 = torch.float8_e5m2


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


# Each row covers a distinct dtype, output, layout, shape, or scale condition.
_TEST_CASES = [
    # M, N, K, path, A dtype, B dtype, output, transposeA, transposeB, scaleA, scaleB
    pytest.param(
        1,
        17,
        64,
        _INT8,
        torch.int8,
        torch.int8,
        torch.float16,
        False,
        False,
        0.01,
        0.02,
        id="int8-ff-singleton",
    ),
    pytest.param(
        33,
        257,
        128,
        _INT8,
        torch.int8,
        torch.int8,
        torch.bfloat16,
        False,
        True,
        0.01,
        0.02,
        id="int8-ft-mn-tail",
    ),
    pytest.param(
        130,
        200,
        70,
        _INT8,
        torch.int8,
        torch.int8,
        torch.float16,
        True,
        False,
        0.01,
        0.02,
        id="int8-tf-mnk-tail",
    ),
    pytest.param(
        192,
        384,
        640,
        _INT8,
        torch.int8,
        torch.int8,
        torch.bfloat16,
        True,
        True,
        0.01,
        0.02,
        id="int8-tt-multi-kl1",
    ),
    pytest.param(
        31,
        47,
        64,
        _FP8,
        _E4,
        _E4,
        torch.float32,
        False,
        False,
        0.01,
        0.02,
        id="fp8-ff-e4-single-tile",
    ),
    pytest.param(
        65,
        129,
        128,
        _FP8,
        _E5,
        _E5,
        torch.float16,
        False,
        True,
        0.01,
        0.02,
        id="fp8-ft-e5-mn-tail",
    ),
    pytest.param(
        130,
        258,
        192,
        _FP8,
        _E4,
        _E5,
        torch.bfloat16,
        True,
        False,
        0.01,
        0.02,
        id="fp8-tf-e4e5",
    ),
    pytest.param(
        259,
        131,
        320,
        _FP8,
        _E5,
        _E4,
        torch.float32,
        True,
        True,
        0.01,
        0.02,
        id="fp8-tt-e5e4",
    ),
    pytest.param(
        15,
        256,
        192,
        _HIF8,
        None,
        None,
        torch.float32,
        False,
        False,
        0.01,
        0.02,
        id="hif8-ff-m-tail",
    ),
    pytest.param(
        513,
        129,
        320,
        _HIF8,
        None,
        None,
        torch.float16,
        False,
        True,
        0.01,
        0.02,
        id="hif8-ft-rect",
    ),
    pytest.param(
        200,
        300,
        640,
        _HIF8,
        None,
        None,
        torch.bfloat16,
        True,
        False,
        0.01,
        0.02,
        id="hif8-tf-multi-kl1",
    ),
    pytest.param(
        127,
        511,
        129,
        _HIF8,
        None,
        None,
        torch.float32,
        True,
        True,
        0.01,
        0.02,
        id="hif8-tt-k-tail",
    ),
    pytest.param(
        768,
        1024,
        1024,
        _HIF8,
        None,
        None,
        torch.float16,
        False,
        True,
        0.01,
        0.02,
        id="hif8-ft-multi-round",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        100,
        100,
        130,
        _HIF8,
        None,
        None,
        torch.bfloat16,
        False,
        False,
        0.1,
        0.1,
        id="hif8-ff-k-tail-scale-1e-2",
    ),
    pytest.param(
        256,
        256,
        256,
        _HIF8,
        None,
        None,
        torch.float32,
        True,
        False,
        100.0,
        100.0,
        id="hif8-tf-scale-1e4",
    ),
    pytest.param(
        257,
        65,
        250,
        _HIF8,
        None,
        None,
        torch.bfloat16,
        True,
        True,
        0.01,
        0.01,
        id="hif8-tt-k-tail-scale-1e-4",
    ),
]


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------


def _transpose_view(tensor, transpose):
    """Return a logical A[M,K]/B[K,N] view backed by transposed storage."""
    return tensor.t().contiguous().t() if transpose else tensor


def make_inputs(m, n, k, path, a_dtype, b_dtype, transpose_a, transpose_b):
    """Create NPU inputs and decoded FP32 operands for the reference path."""
    generator_a = torch.Generator(device="cpu").manual_seed(42)
    generator_b = torch.Generator(device="cpu").manual_seed(43)

    if path == _HIF8:
        import torch_npu

        converter = torch_npu._C._cd
        a_float = (torch.randn((m, k), generator=generator_a) * 0.5).npu()
        b_float = (torch.randn((k, n), generator=generator_b) * 0.5).npu()
        a_bytes = torch.empty(m * k, dtype=torch.uint8, device="npu")
        b_bytes = torch.empty(k * n, dtype=torch.uint8, device="npu")
        converter.cast_to_fp8_noalloc(
            a_float.view(1, -1), a_bytes.view(1, -1), converter.DType.hifloat8
        )
        converter.cast_to_fp8_noalloc(
            b_float.view(1, -1), b_bytes.view(1, -1), converter.DType.hifloat8
        )
        a_reference = converter.cast_from_fp8(
            a_bytes.view(1, -1), converter.DType.hifloat8, converter.DType.float32
        ).view(m, k)
        b_reference = converter.cast_from_fp8(
            b_bytes.view(1, -1), converter.DType.hifloat8, converter.DType.float32
        ).view(k, n)
        a = a_bytes.view(m, k).view(torch.int8)
        b = b_bytes.view(k, n).view(torch.int8)
    elif path == _INT8:
        a_reference = torch.randint(
            -127, 128, (m, k), generator=generator_a, dtype=torch.int8
        )
        b_reference = torch.randint(
            -127, 128, (k, n), generator=generator_b, dtype=torch.int8
        )
        a, b = a_reference.npu(), b_reference.npu()
    else:
        a_reference = (torch.randn((m, k), generator=generator_a) * 0.5).to(a_dtype)
        b_reference = (torch.randn((k, n), generator=generator_b) * 0.5).to(b_dtype)
        a, b = a_reference.npu(), b_reference.npu()

    return (
        _transpose_view(a, transpose_a),
        _transpose_view(b, transpose_b),
        a_reference.float(),
        b_reference.float(),
    )


# ---------------------------------------------------------------------------
# Golden calculation
# ---------------------------------------------------------------------------


def quant_batch_matmul_golden(
    a_reference,
    b_reference,
    scale_a,
    scale_b,
    output_dtype,
):
    """Compute C=(A*scaleA)@(B*scaleB) and cast to the output dtype."""
    return ((a_reference * scale_a) @ (b_reference * scale_b)).to(output_dtype)


# ---------------------------------------------------------------------------
# Precision standard: (absolute tolerance, relative tolerance, mismatch ratio)
# ---------------------------------------------------------------------------


_PRECISION_STANDARD = {
    (_INT8, torch.float16): (1e-3, 1e-2, 1e-3),
    (_INT8, torch.bfloat16): (2e-2, 1e-2, 1e-3),
    (_FP8, torch.float16): (5e-2, 1e-2, 1e-3),
    (_FP8, torch.bfloat16): (5e-2, 1e-2, 1e-3),
    (_FP8, torch.float32): (1e-2, 1e-2, 1e-3),
    (_HIF8, torch.float16): (5e-2, 1e-2, 1e-3),
    (_HIF8, torch.bfloat16): (5e-2, 1e-2, 1e-3),
    (_HIF8, torch.float32): (1e-2, 1e-2, 1e-3),
}


def get_precision_standard(path, output_dtype, scale_a, scale_b):
    """Return atol, rtol, and the allowed mismatch ratio for one case."""
    atol, rtol, max_mismatch = _PRECISION_STANDARD[(path, output_dtype)]
    if path != _INT8:
        atol *= max(1.0, abs(scale_a * scale_b))
    return atol, rtol, max_mismatch


def assert_precision(actual, golden, *, atol, rtol, max_mismatch):
    """Check element tolerances and the allowed 0.1% mismatch ratio."""
    actual = actual.float().cpu()
    golden = golden.float().cpu()
    mismatch = ~torch.isclose(actual, golden, rtol=rtol, atol=atol, equal_nan=True)
    mismatch_ratio = float(mismatch.float().mean())
    if mismatch_ratio > max_mismatch:
        max_error = float((actual - golden).abs().max())
        raise AssertionError(
            f"quant matmul mismatch: max|err|={max_error:.4e}, "
            f"mismatch={mismatch_ratio:.4%}, atol={atol}, rtol={rtol}"
        )


# ---------------------------------------------------------------------------
# NPU precision test
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.parametrize(
    "m,n,k,path,a_dtype,b_dtype,out_dtype,transpose_a,transpose_b,scale_a,scale_b",
    _TEST_CASES,
)
def test_quant_batch_matmul_hif8_tt_precision(
    m,
    n,
    k,
    path,
    a_dtype,
    b_dtype,
    out_dtype,
    transpose_a,
    transpose_b,
    scale_a,
    scale_b,
):
    """Verify dtype, output, transpose, tail, pipeline, and scale coverage."""
    torch_npu = pytest.importorskip("torch_npu")
    if not os.environ.get("ASCEND_HOME_PATH"):
        pytest.skip("ASCEND_HOME_PATH not set")
    if torch.npu.device_count() == 0:
        pytest.skip("no NPU device is available")
    if "Ascend950" not in torch.npu.get_device_name(0):
        pytest.skip("quant matmul requires Ascend950")

    # 1. Construct NPU inputs and independent decoded reference operands.
    a, b, a_reference, b_reference = make_inputs(
        m, n, k, path, a_dtype, b_dtype, transpose_a, transpose_b
    )
    scale_a_tensor = torch.tensor([scale_a], dtype=torch.float32, device="npu")
    scale_b_tensor = torch.tensor([scale_b], dtype=torch.float32, device="npu")
    golden = quant_batch_matmul_golden(
        a_reference, b_reference, scale_a, scale_b, out_dtype
    )

    # 2. Execute the DSL kernel.
    dtype_options = {}
    if path == _HIF8:
        dtype_options = {
            "a_dtype": torch_npu.hifloat8,
            "b_dtype": torch_npu.hifloat8,
        }

    actual = npu_quant_matmul(
        a,
        b,
        scale_a=scale_a_tensor,
        scale_b=scale_b_tensor,
        output_dtype=out_dtype,
        **dtype_options,
    )
    torch.npu.synchronize()

    # 3. Compare the result using the explicit precision standard above.
    atol, rtol, max_mismatch = get_precision_standard(path, out_dtype, scale_a, scale_b)
    assert_precision(actual, golden, atol=atol, rtol=rtol, max_mismatch=max_mismatch)
