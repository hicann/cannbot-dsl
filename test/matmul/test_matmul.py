# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for the matmul kernel (transpose_a=False, transpose_b=True).

Formula:  C[M,N] = A[M,K] @ B[N,K]^T   (fp16/bf16 inputs, fp32 accumulator)

Constraints honoured by the shape list:
  * Inputs are 2-D contiguous matrices.
  * Current kernel path only implements transpose_a=False, transpose_b=True.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys

import numpy as np
import pytest
import torch
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "samples", "matmul"))

from matmul import matmul


_TRANSPOSE_A = False
_TRANSPOSE_B = True


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


_TEST_SHAPES = [
    pytest.param(4096, 3840, 384, id="M4096-K3840-N384"),
    pytest.param(2047, 6145, 895, id="M2047-K6145-N895"),
    pytest.param(8192, 1793, 1280, id="M8192-K1793-N1280"),
    pytest.param(5120, 2049, 2048, id="M5120-K2049-N2048"),
    pytest.param(7168, 1025, 4096, id="M7168-K1025-N4096"),
    pytest.param(16384, 769, 2304, id="M16384-K769-N2304"),
    pytest.param(6145, 4097, 833, id="M6145-K4097-N833"),
    pytest.param(6144, 2048, 640, id="M6144-K2048-N640"),
    pytest.param(6144, 8193, 640, id="M6144-K8193-N640"),
    pytest.param(7168, 2048, 1535, id="M7168-K2048-N1535"),
]


# ---------------------------------------------------------------------------
# Golden kit
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MatmulInputs:
    a: torch.Tensor
    b: torch.Tensor
    transpose_a: bool
    transpose_b: bool


@dataclasses.dataclass
class MatmulGolden:
    result: np.ndarray


def make_inputs(
    m: int,
    k: int,
    n: int,
    *,
    transpose_a: bool = False,
    transpose_b: bool = True,
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> MatmulInputs:
    """Build deterministic A/B for C[M,N] = A[M,K] @ B[N,K]^T."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a_shape = (k, m) if transpose_a else (m, k)
    b_shape = (n, k) if transpose_b else (k, n)
    a = torch.randn(a_shape, generator=gen, dtype=dtype)
    b = torch.randn(b_shape, generator=gen, dtype=dtype)
    return MatmulInputs(a=a, b=b, transpose_a=transpose_a, transpose_b=transpose_b)


def maybe_to_npu(inputs: MatmulInputs) -> MatmulInputs:
    """Move the input tensors to the NPU device, preserving the transpose flags."""
    return MatmulInputs(
        a=inputs.a.npu(),
        b=inputs.b.npu(),
        transpose_a=inputs.transpose_a,
        transpose_b=inputs.transpose_b,
    )


def matmul_golden(inputs: MatmulInputs) -> MatmulGolden:
    """CPU golden: A/B -> compute dtype -> np.matmul (fp32 accumulator) -> cast back.

    Numerical path matches the kernel: low-precision inputs are up-cast to fp32
    before the matmul and the result is cast back to the output dtype at the end.
    """
    output_dtype = inputs.a.dtype
    a_ref = inputs.a.transpose(-2, -1) if inputs.transpose_a else inputs.a
    b_ref = inputs.b.transpose(-2, -1) if inputs.transpose_b else inputs.b

    a_np = torch_tensor_to_numpy(a_ref)
    b_np = torch_tensor_to_numpy(b_ref)

    if output_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = np.float32
    else:
        compute_dtype = np.float64

    a_np = a_np.astype(compute_dtype)
    b_np = b_np.astype(compute_dtype)
    c_np = np.matmul(a_np, b_np)

    return MatmulGolden(result=_cast_output_dtype(c_np, output_dtype))


def assert_isclose(actual: np.ndarray, golden: np.ndarray, dtype: torch.dtype) -> None:
    """Precision-tolerant comparison: allows a small fraction (ptol) of mismatches."""
    actual = actual.reshape(-1)
    golden = golden.reshape(-1)
    if actual.size != golden.size:
        raise AssertionError(f"Output size mismatch: actual={actual.size}, golden={golden.size}")

    rtol, ptol = _dtype_tolerances(dtype)
    atol = _DEFAULT_ATOL
    actual = _normalize_compare_dtype(actual)
    golden = _normalize_compare_dtype(golden)

    if rtol == 0 and atol == 0:
        diff_results = (actual == golden) | (np.isnan(actual) & np.isnan(golden))
    else:
        diff_results = np.isclose(actual, golden, rtol=rtol, atol=atol, equal_nan=True)
    diff_indices = np.where(~diff_results)[0]
    precision = (golden.size - diff_indices.size) / golden.size
    if (1 - precision) <= ptol:
        return

    first = int(diff_indices[0]) if diff_indices.size else 0
    raise AssertionError(
        f"matmul mismatch: "
        f"precision={precision * 100:.6f}%, "
        f"failed_ratio={1 - precision:.6e}, ptol={ptol:.6e}, "
        f"rtol={rtol:.6e}, atol={atol:.6e}, first_index={first}, "
        f"actual={actual[first]}, golden={golden[first]}"
    )


def torch_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (possibly bfloat16) torch tensor to a numpy array.

    numpy has no native bfloat16, so the raw int16 storage is reinterpreted
    through ``ml_dtypes.bfloat16``.
    """
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(bfloat16)
    return tensor.numpy()


def _cast_output_dtype(array: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    if dtype == torch.float16:
        return array.astype(np.float16)
    if dtype == torch.bfloat16:
        return array.astype(bfloat16)
    return array


def _normalize_compare_dtype(array: np.ndarray) -> np.ndarray:
    """Promote bfloat16 to float32 so numpy comparison primitives accept it."""
    if hasattr(array, "dtype") and hasattr(array.dtype, "name") and array.dtype.name == "bfloat16":
        return array.astype("float32", copy=False)
    return array


_DEFAULT_ATOL = 1e-8
_DTYPE_TOLERANCES = {
    torch.float16:  (0.001, 0.001),
    torch.bfloat16: (0.001, 0.001),
}
_DEFAULT_TOLERANCES = (0.0001, 0.0001)


def _dtype_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """Return (rtol, ptol) for the given dtype."""
    return _DTYPE_TOLERANCES.get(dtype, _DEFAULT_TOLERANCES)


# ---------------------------------------------------------------------------
# NPU accuracy tests — matmul DSL kernel
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.parametrize("m,k,n", _TEST_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_matmul_precision(m: int, k: int, n: int, dtype: torch.dtype) -> None:
    pytest.importorskip("torch_npu")

    inputs = make_inputs(
        m, k, n,
        transpose_a=_TRANSPOSE_A,
        transpose_b=_TRANSPOSE_B,
        dtype=dtype,
    )
    npu_inputs = maybe_to_npu(inputs)

    c_npu = matmul(
        npu_inputs.a, npu_inputs.b,
        transpose_a=_TRANSPOSE_A,
        transpose_b=_TRANSPOSE_B,
    )
    torch.npu.synchronize()

    result_npu = torch_tensor_to_numpy(c_npu.cpu())
    golden = matmul_golden(inputs)

    assert_isclose(result_npu, golden.result, dtype)

    max_err = float(np.abs(result_npu.astype(np.float32) - golden.result.astype(np.float32)).max())
    logging.info(
        "matmul (M=%s,K=%s,N=%s,dtype=%s) max|err|=%.4e finished!",
        m, k, n, dtype, max_err,
    )
