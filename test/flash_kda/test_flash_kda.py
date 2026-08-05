# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FlashKDA NPU precision comparison against a CPU chunk golden."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


CHUNK_SIZE = 64
HEAD_DIM = 128
REFERENCE_BLOCK_SIZE = 16
DEFAULT_CASE = (1, 32, 2048, HEAD_DIM)
DEFAULT_LOWER_BOUND = -5.0
_TOLERANCE = 5e-3


@dataclasses.dataclass(frozen=True)
class KdaInputs:
    k: torch.Tensor
    v: torch.Tensor
    q: torch.Tensor
    beta: torch.Tensor
    g: torch.Tensor
    initial_state: torch.Tensor
    scale: float
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    lower_bound: float


def _make_inputs(*, seed: int) -> KdaInputs:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch, value_heads, sequence_length, dim = DEFAULT_CASE
    key_heads = value_heads
    qk_shape = (batch, key_heads, sequence_length, dim)
    value_shape = (batch, value_heads, sequence_length, dim)
    return KdaInputs(
        k=F.normalize(torch.randn(qk_shape, generator=generator), p=2, dim=-1).bfloat16(),
        v=torch.randn(value_shape, generator=generator).bfloat16(),
        q=F.normalize(torch.randn(qk_shape, generator=generator), p=2, dim=-1).bfloat16(),
        beta=torch.randn((batch, value_heads, sequence_length, 1), generator=generator).bfloat16(),
        g=torch.randn(value_shape, generator=generator).bfloat16(),
        initial_state=torch.rand(
            (batch, value_heads, dim, dim), generator=generator, dtype=torch.float32
        ) * 0.1,
        scale=dim ** -0.5,
        a_log=torch.linspace(-1.0, 0.5, value_heads, dtype=torch.float32),
        dt_bias=torch.randn((value_heads, dim), generator=generator, dtype=torch.float32),
        lower_bound=DEFAULT_LOWER_BOUND,
    )


def _activate(inputs: KdaInputs) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = torch.exp(inputs.a_log).view(1, -1, 1, 1)
    bias = inputs.dt_bias.view(1, -1, 1, HEAD_DIM)
    gate = inputs.lower_bound * torch.sigmoid(alpha * (inputs.g.float() + bias))
    return gate, torch.sigmoid(inputs.beta.float())


def _expand_gqa(inputs: KdaInputs) -> tuple[torch.Tensor, torch.Tensor]:
    repeats = inputs.v.shape[1] // inputs.k.shape[1]
    return (
        inputs.k.float().repeat_interleave(repeats, dim=1),
        inputs.q.float().repeat_interleave(repeats, dim=1),
    )


def _chunk_major(values: torch.Tensor) -> torch.Tensor:
    batch, heads, sequence_length, dim = values.shape
    return values.reshape(
        batch, heads, sequence_length // CHUNK_SIZE, CHUNK_SIZE, dim
    ).contiguous()


def _reference_lower_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    cumulative_decay: torch.Tensor,
    *,
    diagonal: int,
) -> torch.Tensor:
    """Compute causal tiles with CANNDSL's 16-token relative-decay references."""
    result = torch.zeros(
        *left.shape[:-1], right.shape[-2], dtype=torch.float32, device=left.device
    )
    for row_start in range(0, CHUNK_SIZE, REFERENCE_BLOCK_SIZE):
        row_end = row_start + REFERENCE_BLOCK_SIZE
        reference = cumulative_decay[..., row_start : row_start + 1, :]
        left_rows = left[..., row_start:row_end, :] * (
            cumulative_decay[..., row_start:row_end, :] - reference
        ).exp()
        for col_start in range(0, row_end, REFERENCE_BLOCK_SIZE):
            col_end = col_start + REFERENCE_BLOCK_SIZE
            right_cols = right[..., col_start:col_end, :] * (
                reference - cumulative_decay[..., col_start:col_end, :]
            ).exp()
            result[..., row_start:row_end, col_start:col_end] = (
                left_rows @ right_cols.transpose(-1, -2)
            )
    return result * torch.tril(
        torch.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.float32, device=left.device),
        diagonal=diagonal,
    )


def _cpu_chunk(inputs: KdaInputs) -> tuple[torch.Tensor, torch.Tensor]:
    gate, beta = _activate(inputs)
    k, q = _expand_gqa(inputs)
    k = _chunk_major(k)
    q = _chunk_major(q)
    v = _chunk_major(inputs.v.float())
    gate = _chunk_major(gate)
    beta = _chunk_major(beta)
    cumulative_decay = gate.cumsum(dim=-2)
    gamma = cumulative_decay.exp()
    k_decayed = gamma * k
    q_decayed = gamma * q * inputs.scale
    transition = _reference_lower_matmul(beta * k, k, cumulative_decay, diagonal=-1)
    inverse_beta = torch.linalg.solve_triangular(
        torch.eye(CHUNK_SIZE, dtype=torch.float32, device=k.device) + transition,
        torch.diag_embed(beta.squeeze(-1)),
        upper=False,
        unitriangular=True,
    )
    mqk = _reference_lower_matmul(q * inputs.scale, k, cumulative_decay, diagonal=0)
    k_restored = k * (cumulative_decay[..., -1:, :] - cumulative_decay).exp()
    gamma_c = gamma[..., -1, :, None].contiguous()
    u_pre = inverse_beta @ v
    w = inverse_beta @ k_decayed

    state = inputs.initial_state.float().clone()
    output = torch.empty_like(q_decayed)
    for chunk_index in range(u_pre.shape[2]):
        u = u_pre[:, :, chunk_index] - w[:, :, chunk_index] @ state
        o_state = q_decayed[:, :, chunk_index] @ state
        state = (
            gamma_c[:, :, chunk_index] * state
            + k_restored[:, :, chunk_index].transpose(-1, -2) @ u
        )
        output[:, :, chunk_index] = o_state + mqk[:, :, chunk_index] @ u
    output = output.reshape(inputs.v.shape).contiguous()
    return output, state.transpose(-1, -2).contiguous()


@pytest.mark.npu
def test_flash_kda_npu_precision_matches_cpu_chunk_golden() -> None:
    pytest.importorskip("torch_npu")
    sample_dir = Path(__file__).resolve().parents[2] / "samples" / "flash_kda"
    sys.path.insert(0, str(sample_dir))
    from flash_kda import flash_kda

    inputs = _make_inputs(seed=7)
    golden_output, golden_state = _cpu_chunk(inputs)
    initial_state = inputs.initial_state.transpose(-1, -2).contiguous().npu()
    initial_state_before = initial_state.clone()

    output, final_state = flash_kda(
        inputs.q.npu(),
        inputs.k.npu(),
        inputs.v.npu(),
        inputs.g.npu(),
        inputs.beta.squeeze(-1).npu(),
        inputs.scale,
        initial_state,
        inputs.a_log.npu(),
        inputs.dt_bias.npu(),
        inputs.lower_bound,
        "BNSD",
    )
    torch.npu.synchronize()

    torch.testing.assert_close(initial_state.cpu(), initial_state_before.cpu(), atol=0, rtol=0)
    torch.testing.assert_close(
        output.cpu().float(), golden_output, atol=_TOLERANCE, rtol=_TOLERANCE
    )
    torch.testing.assert_close(
        final_state.cpu().float(), golden_state, atol=_TOLERANCE, rtol=_TOLERANCE
    )
