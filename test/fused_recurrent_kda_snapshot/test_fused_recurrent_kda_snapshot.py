# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NPU precision coverage for the Snapshot kernel."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from _samples_path import load_sample


L2_NORM_EPSILON = 1e-6


@dataclasses.dataclass
class FusedKdaInputs:
    K: torch.Tensor
    V: torch.Tensor
    Q: torch.Tensor
    beta: torch.Tensor
    g: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    lower_bound: float
    state_pool: torch.Tensor
    ssm_state_indices: torch.Tensor
    scale_value: float


@dataclasses.dataclass
class DecodeGolden:
    O: torch.Tensor
    S: torch.Tensor


def _bf16_to_fp32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.bfloat16).to(torch.float32)


def normalize_qk(raw_qk: torch.Tensor) -> torch.Tensor:
    """Normalize each raw Q/K row in fp32 for the fused recurrence."""
    values = raw_qk.to(torch.float32)
    inverse_norm = torch.rsqrt((values * values).sum(dim=-1, keepdim=True) + L2_NORM_EPSILON)
    return values * inverse_norm


def activate_gate(raw_g: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, lower_bound: float) -> torch.Tensor:
    """Activate raw gate logits with V-head parameters in fp32."""
    alpha = torch.exp(A_log.to(torch.float32)).view(1, -1, 1, 1)
    bias = dt_bias.to(torch.float32).view(1, -1, 1, dt_bias.shape[-1])
    return float(lower_bound) * torch.sigmoid(alpha * (raw_g.to(torch.float32) + bias))


def activate_beta(raw_beta: torch.Tensor) -> torch.Tensor:
    """Activate raw beta logits in fp32."""
    return torch.sigmoid(raw_beta.to(torch.float32))


def _validate_inputs(inputs: FusedKdaInputs) -> tuple[int, int, int, int, int]:
    assert inputs.K.dim() == 4 and inputs.Q.shape == inputs.K.shape
    batch, num_kv_heads, seq_len, dim = inputs.K.shape
    assert inputs.V.shape[0] == batch and tuple(inputs.V.shape[2:]) == (seq_len, dim)
    num_value_heads = inputs.V.shape[1]
    assert num_value_heads % num_kv_heads == 0
    assert inputs.beta.shape == (batch, num_value_heads, seq_len, 1)
    assert inputs.g.shape == inputs.V.shape
    assert inputs.A_log.shape == (num_value_heads,)
    assert inputs.dt_bias.shape == (num_value_heads, dim)
    assert -5.0 <= inputs.lower_bound <= 0.0
    assert inputs.state_pool.dim() == 4 and tuple(inputs.state_pool.shape[1:]) == (num_value_heads, dim, dim)
    assert inputs.state_pool.shape[0] >= batch * seq_len
    assert inputs.ssm_state_indices.shape == (batch * seq_len,)
    assert dim == 128
    return batch, num_value_heads, num_kv_heads, seq_len, dim


def make_inputs(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    *,
    nk: int | None = None,
    block_num: int | None = None,
    seed: int = 42,
    ssm_perm: bool = False,
    state_dtype: torch.dtype = torch.bfloat16,
    lower_bound: float = -3.0,
) -> FusedKdaInputs:
    """Create raw bf16 fused-decode inputs and a paged state pool."""
    assert dim == 128
    nk = heads if nk is None else nk
    if heads <= 0 or nk <= 0 or heads % nk:
        raise ValueError(f"GQA requires positive Nv % Nk == 0, got Nv={heads}, Nk={nk}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    qk_shape = (batch, nk, seq_len, dim)
    value_shape = (batch, heads, seq_len, dim)
    pool_slots = batch * seq_len if block_num is None else int(block_num)
    if pool_slots < batch * seq_len:
        raise ValueError(f"block_num must be at least B*S={batch * seq_len}, got {pool_slots}")
    ssm = torch.arange(batch * seq_len, dtype=torch.int32)
    if ssm_perm and seq_len > 1:
        for batch_index in range(batch):
            start = batch_index * seq_len
            ssm[start:start + seq_len] = ssm[start:start + seq_len].roll(1)

    inputs = FusedKdaInputs(
        K=torch.randn(qk_shape, generator=generator).to(torch.bfloat16),
        V=((torch.rand(value_shape, generator=generator) * 2.0 - 1.0) * 0.05).to(torch.bfloat16),
        Q=torch.randn(qk_shape, generator=generator).to(torch.bfloat16),
        beta=((torch.rand((batch, heads, seq_len, 1), generator=generator) * 6.0) - 3.0).to(torch.bfloat16),
        g=((torch.rand(value_shape, generator=generator) * 6.0) - 3.0).to(torch.bfloat16),
        A_log=torch.linspace(-1.0, 0.5, heads, dtype=torch.float32),
        dt_bias=(torch.rand((heads, dim), generator=generator) - 0.5).to(torch.float32),
        lower_bound=float(lower_bound),
        state_pool=(((torch.rand((pool_slots, heads, dim, dim), generator=generator) * 2.0) - 1.0) * 0.01).to(state_dtype),
        ssm_state_indices=ssm,
        scale_value=dim ** -0.5,
    )
    _validate_inputs(inputs)
    return inputs


def pack_mixedqkv_bsnd(inputs: FusedKdaInputs) -> torch.Tensor:
    """Pack golden BNSD Q/K/V into one contiguous BSND ``[B, S, 3*P]`` tensor.

    The packed kernels address, after its internal BSND->BNSD view swap, head
    slots ``[0, N) = Q``, ``[N, 2N) = K``, ``[2N, 3N) = V``.
    """
    query = inputs.Q.permute(0, 2, 1, 3)
    key = inputs.K.permute(0, 2, 1, 3)
    value = inputs.V.permute(0, 2, 1, 3)
    packed = torch.cat((query, key, value), dim=2).contiguous()
    batch, seq_len, width, dim = packed.shape
    return packed.view(batch, seq_len, width * dim)


def to_bsnd(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a ``[B, N, S, D]`` tensor to a contiguous ``[B, S, N, D]`` one."""
    return tensor.permute(0, 2, 1, 3).contiguous()


def pack_layout_inputs(inputs: FusedKdaInputs, layout: str, lengths: tuple[int, ...]):
    """Pack CPU inputs, excluding padded tokens from ragged TND sequences."""
    mixed = pack_mixedqkv_bsnd(inputs)
    gate, beta = to_bsnd(inputs.g), to_bsnd(inputs.beta)
    if layout == "TND":
        tensors = tuple(torch.cat([tensor[b, :length] for b, length in enumerate(lengths)])
                        for tensor in (mixed, gate, beta))
        cu = torch.tensor((0, *lengths), dtype=torch.int32).cumsum(0).to(torch.int32)
        return (*tensors, cu)
    assert all(length == inputs.K.shape[2] for length in lengths)
    if layout == "BNSD":
        mixed = mixed.view(*mixed.shape[:2], -1, 128).permute(0, 2, 1, 3).contiguous()
        gate, beta = inputs.g.contiguous(), inputs.beta.contiguous()
    else:
        assert layout == "BSND"
    return mixed, gate, beta, None


def output_to_packed(output: torch.Tensor, layout: str) -> torch.Tensor:
    """View a layout-specific output as contiguous token/head/value rows."""
    if layout == "BNSD":
        output = to_bsnd(output)
    return output.reshape(-1, output.shape[-2], output.shape[-1])


def decode_pool_golden(inputs: FusedKdaInputs, num_accepted: torch.Tensor | None = None) -> DecodeGolden:
    """Run the raw-input recurrent reference with the production pool semantics."""
    batch, num_value_heads, num_kv_heads, seq_len, _ = _validate_inputs(inputs)
    accepted = num_accepted.tolist() if num_accepted is not None else None
    if accepted is not None:
        assert len(accepted) == batch and all(1 <= value <= seq_len for value in accepted)

    K = normalize_qk(inputs.K)
    Q = normalize_qk(inputs.Q)
    V = _bf16_to_fp32(inputs.V)
    beta = activate_beta(inputs.beta)
    gate = activate_gate(inputs.g, inputs.A_log, inputs.dt_bias, inputs.lower_bound)
    state_is_bf16 = inputs.state_pool.dtype == torch.bfloat16
    pool = (_bf16_to_fp32(inputs.state_pool) if state_is_bf16 else inputs.state_pool.to(torch.float32)).clone()
    output = torch.zeros(batch, num_value_heads, seq_len, 128, dtype=torch.float32)
    ssm = inputs.ssm_state_indices.tolist()
    gqa_group = num_value_heads // num_kv_heads

    for batch_index in range(batch):
        initial_offset = accepted[batch_index] - 1 if accepted is not None else 0
        for value_head in range(num_value_heads):
            state = pool[ssm[batch_index * seq_len + initial_offset], value_head].clone()
            kv_head = value_head // gqa_group
            for token_index in range(seq_len):
                state = state * torch.exp(gate[batch_index, value_head, token_index])[None, :]
                state_key = (state * K[batch_index, kv_head, token_index][None, :]).sum(dim=-1)
                delta = (V[batch_index, value_head, token_index] - state_key) * beta[batch_index, value_head, token_index, 0]
                state = state + delta[:, None] * K[batch_index, kv_head, token_index][None, :]
                output[batch_index, value_head, token_index] = (
                    state * Q[batch_index, kv_head, token_index][None, :]
                ).sum(dim=-1) * float(inputs.scale_value)
                pool[ssm[batch_index * seq_len + token_index], value_head] = state

    return DecodeGolden(
        O=_bf16_to_fp32(output),
        S=_bf16_to_fp32(pool) if state_is_bf16 else pool,
    )


def _require_npu(torch):
    try:
        probe = torch.empty(1).npu()
        torch.npu.synchronize()
        del probe
    except RuntimeError as exc:
        pytest.skip(f"NPU runtime unavailable: {exc}")


@pytest.mark.npu
@pytest.mark.parametrize("batch,heads,seq_len,layout,accepted", [
    pytest.param(16, 6, 8, "BSND", 4, id="bsnd-b16-n6-s8-accepted4"),
    pytest.param(4, 3, 4, "BNSD", 0, id="bnsd-b4-n3-s4-accepted0"),
    pytest.param(4, 3, 8, "TND", 8, id="tnd-b4-n3-s8-accepted8"),
])
def test_fused_recurrent_kda_snapshot(batch, heads, seq_len, layout, accepted):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    _require_npu(torch)
    snapshot = load_sample("fused_recurrent_kda_snapshot/fused_recurrent_kda_snapshot.py")

    accepted = min(accepted, seq_len)
    inputs = make_inputs(batch, heads, seq_len, 128, state_dtype=torch.float32,
                         lower_bound=-5.0, seed=43, ssm_perm=True)
    mixed = pack_mixedqkv_bsnd(inputs)
    gate = to_bsnd(inputs.g)
    raw_beta = to_bsnd(inputs.beta)
    kwargs = {}
    if layout == "BNSD":
        mixed = mixed.view(batch, seq_len, 3 * heads, 128)
        mixed = mixed.permute(0, 2, 1, 3).contiguous()
        gate = inputs.g
        raw_beta = inputs.beta
    elif layout == "TND":
        mixed = mixed.view(batch * seq_len, 3 * heads * 128)
        gate = gate.view(batch * seq_len, heads, 128)
        raw_beta = raw_beta.view(batch * seq_len, heads, 1)
        kwargs["cu_seqlens"] = torch.arange(
            0, (batch + 1) * seq_len, seq_len, dtype=torch.int32).npu()
    state = inputs.state_pool.npu()
    _, output = snapshot.fused_recurrent_kda(
        mixed.npu(), state, raw_beta.npu(), gate.npu(), inputs.scale_value,
        inputs.A_log.npu(), inputs.dt_bias.npu(), -5.0, layout,
        ssm_state_indices=inputs.ssm_state_indices.npu(),
        num_accepted_tokens=torch.full(
            (batch,), accepted, dtype=torch.int32).npu(), **kwargs)
    torch.npu.synchronize()
    golden = decode_pool_golden(
        inputs, num_accepted=torch.full(
            (batch,), max(accepted, 1), dtype=torch.int32))
    actual = output.cpu().float()
    if layout == "BSND":
        actual = actual.permute(0, 2, 1, 3)
    elif layout == "TND":
        actual = actual.view(batch, seq_len, heads, 128).permute(0, 2, 1, 3)
    torch.testing.assert_close(actual, golden.O, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(state.cpu(), golden.S, atol=5e-3, rtol=5e-3)


@pytest.mark.npu
@pytest.mark.parametrize("layout,heads,lengths,accepted", [
    pytest.param("BSND", 3, (1, 1, 1), (0, 1, 0), id="bsnd-b3-n3-s1"),
    pytest.param("BNSD", 5, (3, 3, 3), (0, 1, 2), id="bnsd-b3-n5-s3"),
    pytest.param("BSND", 7, (5, 5, 5), (0, 2, 4), id="bsnd-b3-n7-s5"),
    pytest.param("BNSD", 9, (7, 7, 7, 7, 7), (0, 1, 3, 5, 6), id="bnsd-b5-n9-s7"),
    pytest.param("TND", 9, (1, 3, 5, 7, 3), (0, 2, 3, 6, 1), id="tnd-ragged-b5-n9"),
    pytest.param("BSND", 33, (1,), (1,), id="bsnd-b1-n33-full-head"),
    pytest.param("BNSD", 65, (3,), (2,), id="bnsd-b1-n65-half-tail"),
    pytest.param("TND", 32, (1, 3, 5), (0, 2, 4), id="tnd-ragged-b3-n32-balanced"),
])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_fused_recurrent_kda_snapshot_whitebox(layout, heads, lengths, accepted, state_dtype):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    _require_npu(torch)
    snapshot = load_sample("fused_recurrent_kda_snapshot/fused_recurrent_kda_snapshot.py")

    batch, sequence = len(lengths), max(lengths)
    inputs = make_inputs(batch, heads, sequence, 128, seed=1201,
                         state_dtype=state_dtype, lower_bound=-5.0,
                         block_num=batch * sequence + 5)
    mixed, gate, beta, cu = pack_layout_inputs(inputs, layout, lengths)
    token_count = sum(lengths)
    # Unique, shuffled slots cross sequence boundaries; leave both edge slots guarded.
    generator = torch.Generator().manual_seed(1202)
    slots = (torch.randperm(inputs.state_pool.shape[0] - 2, generator=generator)
             [:token_count] + 1).to(torch.int32)
    before = inputs.state_pool.clone()
    state = before.npu()
    kwargs = {"cu_seqlens": cu.npu()} if cu is not None else {}
    returned_state, output = snapshot.fused_recurrent_kda(
        mixed.npu(), state, beta.npu(), gate.npu(), inputs.scale_value,
        inputs.A_log.npu(), inputs.dt_bias.npu(), -5.0, layout,
        ssm_state_indices=slots.npu(),
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32).npu(), **kwargs)
    torch.npu.synchronize()
    assert returned_state is state
    actual_pool = state.cpu()
    actual_output = output_to_packed(output.cpu(), layout).float()
    key, query = normalize_qk(inputs.K), normalize_qk(inputs.Q)
    decay = activate_gate(inputs.g, inputs.A_log, inputs.dt_bias, -5.0).exp()
    beta_act = activate_beta(inputs.beta)
    packed_offset = 0
    for b, length in enumerate(lengths):
        assert 0 <= accepted[b] <= length
        initial_slot = int(slots[packed_offset + max(accepted[b], 1) - 1])
        current = before[initial_slot].float().clone()
        for token in range(length):
            current = current * decay[b, :, token, None, :]
            state_key = (current * key[b, :, token, None, :]).sum(-1)
            update = (inputs.V[b, :, token].float() - state_key) * beta_act[b, :, token]
            current = current + update[..., None] * key[b, :, token, None, :]
            expected = ((current * query[b, :, token, None, :]).sum(-1)
                        * inputs.scale_value).bfloat16().float()
            row = packed_offset + token
            torch.testing.assert_close(actual_output[row], expected, atol=5e-3, rtol=5e-3,
                                       msg=f"output batch={b}, token={token}")
            torch.testing.assert_close(actual_pool[int(slots[row])], current.to(state_dtype),
                                       atol=5e-3, rtol=5e-3,
                                       msg=f"snapshot batch={b}, token={token}, slot={int(slots[row])}")
        packed_offset += length
    untouched = torch.ones(before.shape[0], dtype=torch.bool)
    untouched[slots.long()] = False
    torch.testing.assert_close(actual_pool[untouched], before[untouched], atol=0, rtol=0)
