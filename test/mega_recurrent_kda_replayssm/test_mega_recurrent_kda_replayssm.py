# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent precision tests for Mega KDA ReplaySSM."""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch

from _samples_path import load_sample


SAMPLE = "mega_recurrent_kda_replayssm/mega_recurrent_kda_replayssm.py"
COMMIT_SAMPLE = "commit_recurrent_kda_replayssm/commit_recurrent_kda_replayssm.py"
BASELINE_SAMPLE = "mega_recurrent_kda/mega_recurrent_kda.py"
NZ_WEIGHT_ARGUMENTS = (1, 2, 3, 4, 5, 7)


class _CpuArguments(NamedTuple):
    hidden: torch.Tensor
    qkv_weight: torch.Tensor
    decay_a_weight: torch.Tensor
    decay_b_weight: torch.Tensor
    beta_weight: torch.Tensor
    gate_weight: torch.Tensor
    norm_weight: torch.Tensor
    output_weight: torch.Tensor
    convolution_weight: torch.Tensor
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    conv_state_indices: torch.Tensor
    ssm_state_indices: torch.Tensor
    conv_num_accepted_tokens: torch.Tensor
    num_accepted_tokens: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor


def _module():
    return load_sample(SAMPLE)


def _make_cpu_arguments(seed=20260921):
    generator = torch.Generator().manual_seed(seed)
    batch, sequence, hidden, heads, dim = 16, 8, 7168, 6, 128
    projection = heads * dim

    def randn(shape, dtype=torch.bfloat16, scale=0.02):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    conv_indices = torch.randperm(batch, generator=generator).int() + 1
    conv_indices[::7] = 0
    state_indices = (torch.arange(batch, dtype=torch.int32) + 1).repeat_interleave(
        sequence
    )
    conv_accepted = torch.arange(batch, dtype=torch.int32) % (sequence + 1)
    return _CpuArguments(
        randn((batch, sequence, hidden)),
        randn((3 * projection, hidden)),
        randn((dim, hidden)),
        randn((projection, dim)),
        randn((heads, hidden)),
        randn((projection, hidden)),
        randn((dim,), torch.float32, scale=0.2),
        randn((hidden, projection)),
        randn((4, 3 * projection)),
        randn((batch + 1, sequence + 2, 3 * projection)),
        randn((batch * sequence + 1, heads, dim, dim), torch.float32, scale=0.002),
        conv_indices,
        state_indices,
        conv_accepted,
        torch.zeros(batch, dtype=torch.int32),
        randn((heads,), torch.float32, scale=0.1),
        randn((heads, dim), torch.float32, scale=0.1),
    )


def _to_npu_arguments(arguments):
    torch_npu = pytest.importorskip("torch_npu")
    actual = [value.to("npu") for value in arguments]
    for index in NZ_WEIGHT_ARGUMENTS:
        actual[index] = torch_npu.npu_format_cast(
            actual[index], torch_npu.Format.FRACTAL_NZ
        )
    return tuple(actual)


def _allocate_replay(batch_size, *, fill_nan=False):
    factory = torch.full if fill_nan else torch.empty
    extra = (float("nan"),) if fill_nan else ()
    replay_u = factory(
        (batch_size, 8, 6, 128), *extra, dtype=torch.float16, device="npu"
    )
    replay_k = factory(
        (batch_size, 8, 6, 128), *extra, dtype=torch.float16, device="npu"
    )
    replay_decay = factory(
        (batch_size, 8, 6, 128), *extra, dtype=torch.float32, device="npu"
    )
    return replay_u, replay_k, replay_decay


def _run_verify(module, arguments, batch_size, state, block_num):
    replay_u, replay_k, replay_decay = _allocate_replay(batch_size, fill_nan=True)
    output = module.mega_recurrent_kda_replayssm(
        arguments[0][:batch_size].contiguous(),
        *arguments[1:9],
        arguments[9].clone(),
        state,
        replay_u,
        replay_k,
        replay_decay,
        arguments[11][:batch_size].contiguous(),
        arguments[13][:batch_size].contiguous(),
        arguments[15],
        arguments[16],
        128**-0.5,
        -5.0,
        1e-5,
        block_num=block_num,
    )
    return output, replay_u, replay_k, replay_decay


@pytest.mark.npu
def test_npu_precision_batch_generalization_and_core_control(monkeypatch):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    if int(torch.npu.get_device_properties().cube_core_num) < 32:
        pytest.skip("requires 32 AIC cores")

    module = _module()
    baseline = load_sample(BASELINE_SAMPLE)
    arguments = _to_npu_arguments(_make_cpu_arguments())
    initial_state = arguments[10][1:17].contiguous().clone()
    monkeypatch.setattr(baseline, "_device_block_num", lambda _ref: 32)
    baseline_arguments = list(arguments)
    baseline_arguments[9] = arguments[9].clone()
    expected_output = baseline.mega_recurrent_kda(
        *baseline_arguments, 128**-0.5, -5.0, 1e-5
    )

    state = initial_state.clone()
    output, replay_u, replay_k, replay_decay = _run_verify(
        module, arguments, 16, state, 32
    )
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu().float(),
        expected_output.cpu().float(),
        atol=0.005,
        rtol=0.005,
    )
    torch.testing.assert_close(state.cpu(), initial_state.cpu(), atol=0, rtol=0)
    for records in (replay_u, replay_k, replay_decay):
        assert torch.isfinite(records).all()

    # Every supported grid must execute the same full verify range.
    for cores in (8, 24, 28):
        core_state = initial_state.clone()
        core_output, core_u, core_k, core_decay = _run_verify(
            module, arguments, 16, core_state, cores
        )
        torch.npu.synchronize()
        torch.testing.assert_close(
            core_output.cpu().float(), output.cpu().float(), atol=0.005, rtol=0.005
        )
        for records in (core_u, core_k, core_decay):
            assert torch.isfinite(records).all()
        torch.testing.assert_close(
            core_state.cpu(), initial_state.cpu(), atol=0, rtol=0
        )


@pytest.mark.npu
def test_npu_verify_records_commit_to_old_mega_snapshots(monkeypatch):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires Ascend NPU")
    if int(torch.npu.get_device_properties().cube_core_num) < 32:
        pytest.skip("requires 32 AIC cores")

    verify = _module()
    commit = load_sample(COMMIT_SAMPLE)
    baseline = load_sample(BASELINE_SAMPLE)
    cpu_arguments = list(_make_cpu_arguments(seed=20260922))
    cpu_arguments[12] = torch.arange(1, 16 * 8 + 1, dtype=torch.int32)
    arguments = _to_npu_arguments(tuple(cpu_arguments))
    initial_state = torch.stack(
        [arguments[10][1 + batch_index * 8] for batch_index in range(16)]
    ).contiguous()

    monkeypatch.setattr(baseline, "_device_block_num", lambda _ref: 32)
    baseline_arguments = list(arguments)
    baseline_arguments[9] = arguments[9].clone()
    baseline_arguments[10] = arguments[10].clone()
    expected_output = baseline.mega_recurrent_kda(
        *baseline_arguments, 128**-0.5, -5.0, 1e-5
    )

    state = initial_state.clone()
    output, replay_u, replay_k, replay_decay = _run_verify(
        verify, arguments, 16, state, 32
    )
    accepted_cpu = torch.arange(16, dtype=torch.int32) % 9
    accepted = accepted_cpu.npu()
    commit.commit_recurrent_kda_replayssm(
        state,
        replay_u,
        replay_k,
        replay_decay,
        accepted,
        block_num=32,
    )
    torch.npu.synchronize()

    expected_state = torch.stack(
        [
            initial_state[batch_index]
            if int(accepted_cpu[batch_index]) == 0
            else baseline_arguments[10][
                1 + batch_index * 8 + int(accepted_cpu[batch_index]) - 1
            ]
            for batch_index in range(16)
        ]
    ).contiguous()
    torch.testing.assert_close(
        output.cpu().float(), expected_output.cpu().float(), atol=0.005, rtol=0.005
    )
    torch.testing.assert_close(
        state.cpu(), expected_state.cpu(), atol=0.005, rtol=0.005
    )
