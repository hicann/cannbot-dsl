# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
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
    batch, value_heads, key_heads, sequence_length = 1, 2, 1, CHUNK_SIZE
    qk_shape = (batch, key_heads, sequence_length, HEAD_DIM)
    value_shape = (batch, value_heads, sequence_length, HEAD_DIM)
    return KdaInputs(
        k=F.normalize(torch.randn(qk_shape, generator=generator), p=2, dim=-1).bfloat16(),
        v=torch.randn(value_shape, generator=generator).bfloat16(),
        q=F.normalize(torch.randn(qk_shape, generator=generator), p=2, dim=-1).bfloat16(),
        beta=torch.randn((batch, value_heads, sequence_length, 1), generator=generator).bfloat16(),
        g=torch.randn(value_shape, generator=generator).bfloat16(),
        initial_state=torch.rand(
            (batch, value_heads, HEAD_DIM, HEAD_DIM), generator=generator, dtype=torch.float32
        ),
        scale=HEAD_DIM ** -0.5,
        a_log=torch.linspace(-1.0, 0.5, value_heads, dtype=torch.float32),
        dt_bias=torch.randn((value_heads, HEAD_DIM), generator=generator, dtype=torch.float32),
        lower_bound=-1.0,
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


@dataclasses.dataclass(frozen=True)
class _StageOne:
    q_decayed: torch.Tensor
    mqk: torch.Tensor
    k_restored: torch.Tensor
    gamma_c: torch.Tensor
    u_pre: torch.Tensor
    w: torch.Tensor


def _lower_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    cumulative_decay: torch.Tensor,
    *,
    diagonal: int,
) -> torch.Tensor:
    result = torch.empty(
        (*left.shape[:-1], CHUNK_SIZE), dtype=torch.float32, device=left.device
    )
    for row in range(CHUNK_SIZE):
        relative_decay = (cumulative_decay[:, :, row : row + 1] - cumulative_decay).exp()
        result[:, :, row] = (
            (left[:, :, row : row + 1] * relative_decay[:, :, row : row + 1])
            @ (right * relative_decay).transpose(-1, -2)
        ).squeeze(-2)
    return result * torch.tril(
        torch.ones((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.float32), diagonal=diagonal
    )


def _stage_one(inputs: KdaInputs) -> _StageOne:
    gate, beta = _activate(inputs)
    k, q = _expand_gqa(inputs)
    cumulative_decay = gate.cumsum(dim=-2)
    gamma = cumulative_decay.exp()
    k_decayed = gamma * k
    q_decayed = gamma * q * inputs.scale
    transition = _lower_matmul(beta * k, k, cumulative_decay, diagonal=-1)
    inverse_beta = torch.linalg.solve_triangular(
        torch.eye(CHUNK_SIZE).expand_as(transition) + transition,
        torch.diag_embed(beta.squeeze(-1)),
        upper=False,
        unitriangular=True,
    )
    return _StageOne(
        q_decayed=q_decayed,
        mqk=_lower_matmul(q * inputs.scale, k, cumulative_decay, diagonal=0),
        k_restored=k * (cumulative_decay[:, :, -1:] - cumulative_decay).exp(),
        gamma_c=gamma[:, :, -1].unsqueeze(-1),
        u_pre=inverse_beta @ inputs.v.float(),
        w=inverse_beta @ k_decayed,
    )


def _cpu_chunk(inputs: KdaInputs) -> tuple[torch.Tensor, torch.Tensor]:
    stage_one = _stage_one(inputs)
    state = inputs.initial_state.float().clone()
    u = stage_one.u_pre - stage_one.w @ state
    output = stage_one.q_decayed @ state + stage_one.mqk @ u
    state = stage_one.gamma_c * state + stage_one.k_restored.transpose(-1, -2) @ u
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
