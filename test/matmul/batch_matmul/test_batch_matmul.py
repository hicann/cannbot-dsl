# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Precision tests for the batch_matmul wrapper (stride-derived layouts).

Formula:  C[c_batch, M, N] = A[a_batch, M, K] @ B[b_batch, N, K]^T (+ bias),
          c_batch = broadcast(a_batch, b_batch), K always last, ranks 2..6.

Cases cover:
  * dtypes fp16 / bf16 / fp32 exact / fp32 hf32 fast mode
  * bias epilogue: shared [N] and per-batch [*c_batch, N], incl. tail-N pad
  * K-major storages handed as canonical transpose(-1,-2) views
  * multi-dim numpy-style batch broadcast up to rank 6 (either side
    broadcast, mixed ranks), unaligned tails, K at the k_l1 segment
    boundary, 2-D, and non-broadcastable batch rejection
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

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "samples", "matmul", "batch_matmul"
    ),
)

from batch_matmul import batch_matmul


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Case:
    id: str
    batch_shape_a: tuple           # leading batch dims of a, () = rank 2
    m: int
    k: int
    n: int
    dtype: torch.dtype = torch.float16
    batch_shape_b: tuple | None = None  # None -> same as batch_shape_a
    a_k_major: bool = False        # logical a [*a,M,K] as a view of (K,M) storage
    b_k_major: bool = False        # logical b [*b,N,K] as a view of (K,N) storage
    bias: str | None = None        # None | "shared" [N] | "per_batch" [*c,N]
    bias_input_dtype: bool = False
    hf32: bool = False
    two_d: bool = False            # squeeze both inputs to 2-D before the call


_CASES = [
    _Case("std-bs4-1024cubed-fp16", (4,), 1024, 1024, 1024),
    _Case("std-bs8-256-1024-512-bf16", (8,), 256, 1024, 512,
          dtype=torch.bfloat16),
    _Case("fp32-bs2-2048-512-1024", (2,), 2048, 512, 1024,
          dtype=torch.float32),
    _Case("hf32-bs2-1024cubed", (2,), 1024, 1024, 1024,
          dtype=torch.float32, hf32=True),
    _Case("tail-bs3-100-33-17-bf16", (3,), 100, 33, 17, dtype=torch.bfloat16),
    _Case("kL1-boundary-bs2-128-513-128", (2,), 128, 513, 128),
    _Case("bias-shared-bs2-64-128-96", (2,), 64, 128, 96, bias="shared"),
    _Case("bias-input-dtype-bs2-64-128-96", (2,), 64, 128, 96, bias="shared",
          bias_input_dtype=True),
    _Case("bias-perbatch-tailN-bs2-64-33-17", (2,), 64, 33, 17,
          dtype=torch.bfloat16, bias="per_batch"),
    _Case("kmajor-views-tail-bs2-100-128-96", (2,), 100, 128, 96,
          a_k_major=True, b_k_major=True),
    _Case("bcast-a-side-bs1x4-64-64-96", (1,), 64, 64, 96,
          batch_shape_b=(4,), bias="shared"),
    _Case("2d-128-256-64", (1,), 128, 256, 64, two_d=True),
    # ---- multi-dim numpy-style batch broadcast (the wrapper's core
    # addition over plain bmm; ranks 4-6, either side broadcast) ----
    _Case("bcast4d-2x1-1x3-64-64-96", (2, 1), 64, 64, 96,
          batch_shape_b=(1, 3)),
    _Case("bcast-a-side-1x4x2-64-64-96", (1, 4, 2), 64, 64, 96,
          batch_shape_b=(2, 4, 2)),
    _Case("bcast-b-side-perbatch-2x4-64-64-96", (2, 4), 64, 64, 96,
          batch_shape_b=(2, 1), bias="per_batch"),
    _Case("bcast-rank2x5-64-64-96", (), 64, 64, 96,
          batch_shape_b=(2, 3, 1)),
    _Case("bcast6d-views-64-64-96", (2, 1, 2, 1), 64, 64, 96,
          batch_shape_b=(1, 3, 1, 2), a_k_major=True, b_k_major=True),
]


# ---------------------------------------------------------------------------
# Golden kit (CPU, independent of the NPU implementation)
# ---------------------------------------------------------------------------


def _batch_shape_b(case: _Case) -> tuple:
    return case.batch_shape_b if case.batch_shape_b is not None else case.batch_shape_a


def make_inputs(case: _Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Deterministic logical a [*a,M,K] / b [*b,N,K] (+ bias)."""
    gen = torch.Generator(device="cpu").manual_seed(42)
    shape_b = _batch_shape_b(case)
    if case.a_k_major:
        a = torch.randn((*case.batch_shape_a, case.k, case.m), generator=gen,
                        dtype=case.dtype).transpose(-1, -2)
    else:
        a = torch.randn((*case.batch_shape_a, case.m, case.k), generator=gen,
                        dtype=case.dtype)
    if case.b_k_major:
        b = torch.randn((*shape_b, case.k, case.n), generator=gen,
                        dtype=case.dtype).transpose(-1, -2)
    else:
        b = torch.randn((*shape_b, case.n, case.k), generator=gen,
                        dtype=case.dtype)
    bias = None
    if case.bias == "shared":
        bdtype = case.dtype if case.bias_input_dtype else torch.float32
        bias = torch.randn(case.n, generator=gen, dtype=bdtype) * 2.0
    elif case.bias == "per_batch":
        c_batch = torch.broadcast_shapes(case.batch_shape_a, shape_b)
        bias = torch.randn(*c_batch, case.n, generator=gen,
                           dtype=case.dtype) * 2.0
    return a, b, bias


def torch_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (possibly bfloat16) torch tensor to numpy (logical order)."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(bfloat16)
    return tensor.numpy()


def bmm_golden(case: _Case, a: torch.Tensor, b: torch.Tensor,
               bias: torch.Tensor | None) -> np.ndarray:
    """CPU golden: fp32 upcast -> np.matmul (batch broadcast) -> cast back."""
    a_np = torch_tensor_to_numpy(a).astype(np.float32)
    b_np = torch_tensor_to_numpy(b).astype(np.float32)
    g = np.matmul(a_np, np.swapaxes(b_np, -1, -2))
    if bias is not None:
        bias_np = torch_tensor_to_numpy(bias).astype(np.float32)
        g = g + bias_np.reshape(
            *bias_np.shape[:-1], *((1,) * (g.ndim - bias_np.ndim)), -1
        )
    if case.dtype == torch.float16:
        return g.astype(np.float16)
    if case.dtype == torch.bfloat16:
        return g.astype(bfloat16)
    return g


_DEFAULT_ATOL = 1e-8
# dtype -> (rtol, ptol, atol).  fp32 rtol covers NPU-vs-BLAS accumulation
# reassociation noise (~K * eps); the compute itself is exact fp32.
_DTYPE_TOLERANCES = {
    torch.float16: (0.001, 0.001, 1e-8),
    torch.bfloat16: (0.001, 0.001, 1e-8),
    torch.float32: (3e-4, 3e-3, 1e-6),
}
_DEFAULT_TOLERANCES = (0.0001, 0.0001, 1e-8)
# HF32 fast mode: TF32-tier operand truncation (~2^-11 per operand).
# Scale-normalized error stays ~3e-4; element-wise rel can reach ~1e-2 on
# long-K cancellation-heavy sums.
_HF32_TOLERANCES = (1e-2, 5e-2, 1e-3)


def _normalize_compare_dtype(array: np.ndarray) -> np.ndarray:
    """Promote bfloat16 to float32 so numpy comparison primitives accept it."""
    if (
        hasattr(array, "dtype")
        and hasattr(array.dtype, "name")
        and array.dtype.name == "bfloat16"
    ):
        return array.astype("float32", copy=False)
    return array


def assert_isclose(actual: np.ndarray, golden: np.ndarray,
                   tolerances: tuple) -> None:
    """Precision-tolerant comparison: allows a small fraction (ptol) of mismatches."""
    actual = actual.reshape(-1)
    golden = golden.reshape(-1)
    if actual.size != golden.size:
        raise AssertionError(
            f"Output size mismatch: actual={actual.size}, golden={golden.size}"
        )

    rtol, ptol, atol = tolerances
    actual = _normalize_compare_dtype(actual)
    golden = _normalize_compare_dtype(golden)

    diff_results = np.isclose(actual, golden, rtol=rtol, atol=atol, equal_nan=True)
    diff_indices = np.where(~diff_results)[0]
    precision = (golden.size - diff_indices.size) / golden.size
    if (1 - precision) <= ptol:
        return

    first = int(diff_indices[0]) if diff_indices.size else 0
    raise AssertionError(
        f"batch_matmul mismatch: "
        f"precision={precision * 100:.6f}%, "
        f"failed_ratio={1 - precision:.6e}, ptol={ptol:.6e}, "
        f"rtol={rtol:.6e}, atol={atol:.6e}, first_index={first}, "
        f"actual={actual[first]}, golden={golden[first]}"
    )


# ---------------------------------------------------------------------------
# NPU precision tests
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.parametrize("case", [pytest.param(c, id=c.id) for c in _CASES])
def test_batch_matmul(case: _Case) -> None:
    pytest.importorskip("torch_npu")

    a, b, bias = make_inputs(case)
    if case.two_d:
        a, b = a[0], b[0]

    c_npu = batch_matmul(a.npu(), b.npu(),
                         bias=(bias.npu() if bias is not None else None),
                         hf32=case.hf32)
    torch.npu.synchronize()

    expect_shape = torch.broadcast_shapes(
        case.batch_shape_a, _batch_shape_b(case)
    ) + (case.m, case.n)
    if case.two_d:
        expect_shape = expect_shape[1:]
    assert tuple(c_npu.shape) == expect_shape, (
        f"{case.id}: shape {tuple(c_npu.shape)} != {expect_shape}"
    )

    golden = bmm_golden(case, a, b, bias)
    tolerances = _HF32_TOLERANCES if case.hf32 else _DTYPE_TOLERANCES.get(
        case.dtype, _DEFAULT_TOLERANCES
    )
    try:
        assert_isclose(torch_tensor_to_numpy(c_npu.cpu()), golden, tolerances)
    except AssertionError as e:
        raise AssertionError(f"{case.id}: {e}") from None

    max_err = float(
        np.abs(
            _normalize_compare_dtype(torch_tensor_to_numpy(c_npu)).astype(np.float32)
            - _normalize_compare_dtype(golden).astype(np.float32)
        ).max()
    )
    logging.info(
        "batch_matmul (%s, dtype=%s, bias=%s, hf32=%s) max|err|=%.4e finished!",
        case.id, case.dtype, case.bias, case.hf32, max_err,
    )


@pytest.mark.npu
def test_batch_matmul_batch_not_broadcastable() -> None:
    pytest.importorskip("torch_npu")

    a = torch.randn(2, 64, 128, dtype=torch.float16).npu()
    b = torch.randn(3, 96, 128, dtype=torch.float16).npu()
    with pytest.raises(ValueError, match="broadcastable"):
        batch_matmul(a, b)
