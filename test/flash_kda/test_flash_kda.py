# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Three-way FlashKDA precision: CPU FP32, NPU-rounding simulation, and DSL."""

import dataclasses
import os

import pytest
import torch
import torch.nn.functional as F
from _samples_path import load_sample

CHUNK_SIZE = 64
REFERENCE_BLOCK_SIZE = 16
LOW_DTYPE = torch.bfloat16
SUPPORTED_HEAD_DIM = 128

# Independent PyTorch references from the original FlashKDA example:
# cpu_chunk uses FP32; npu_chunk models current BF16/FP16 handoffs on CPU,
# including FP32 recurrent-state and output accumulators.
@dataclasses.dataclass
class KdaInputs:
    K: torch.Tensor              # (B, Nk, S, D) bf16
    V: torch.Tensor              # (B, Nv, S, D) bf16
    Q: torch.Tensor              # (B, Nk, S, D) bf16
    beta_brc: torch.Tensor       # (B, Nv, S, 1) raw BF16 logits
    g: torch.Tensor              # (B, Nv, S, D) raw BF16 logits
    initial_state: torch.Tensor  # (B, Nv, Dk, Dv) fp32
    scale_value: float
    A_log: torch.Tensor           # (Nv,) fp32
    dt_bias: torch.Tensor         # (Nv, D) fp32
    lower_bound: float


def _chunk_major(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, dim = x.shape
    chunks = seq_len // CHUNK_SIZE
    return x.reshape(batch, heads, chunks, CHUNK_SIZE, dim).contiguous()


def make_inputs_generalized(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    *,
    nk: int | None = None,
    seed: int = 0,
    lower_bound: float = -1.0,
    v_scale: float = 1.0,
    b_spread: float = 1.0,
    init_scale: float = 0.1,
) -> KdaInputs:
    """Build BF16 Q/K/V and raw gate/beta logits with a seeded CPU generator."""
    if dim != SUPPORTED_HEAD_DIM:
        raise ValueError(f"flash_kda 当前只支持 D={SUPPORTED_HEAD_DIM}，收到 D={dim}")
    nk = heads if nk is None else nk
    if heads % nk != 0:
        raise ValueError(f"GQA requires Nv % Nk == 0，收到 Nv={heads}, Nk={nk}")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    K = F.normalize(torch.randn(batch, nk, seq_len, dim, generator=gen), p=2, dim=-1)
    Q = F.normalize(torch.randn(batch, nk, seq_len, dim, generator=gen), p=2, dim=-1)
    V = torch.randn(batch, heads, seq_len, dim, generator=gen) * float(v_scale)
    g = torch.randn(batch, heads, seq_len, dim, generator=gen).to(LOW_DTYPE)
    beta = (torch.randn(batch, heads, seq_len, 1, generator=gen) * float(b_spread)).to(LOW_DTYPE)
    initial_state = torch.rand(batch, heads, dim, dim, generator=gen) * float(init_scale)

    return KdaInputs(
        K=K.to(LOW_DTYPE), V=V.to(LOW_DTYPE), Q=Q.to(LOW_DTYPE),
        beta_brc=beta, g=g, initial_state=initial_state, scale_value=dim ** -0.5,
        A_log=torch.linspace(-1.0, 0.5, heads, dtype=torch.float32),
        dt_bias=torch.randn(heads, dim, generator=gen, dtype=torch.float32),
        lower_bound=float(lower_bound),
    )


@dataclasses.dataclass
class _StageOneGolden:
    Gamma: torch.Tensor
    K_decayed_beta: torch.Tensor
    K_decayed: torch.Tensor
    Q_decayed: torch.Tensor
    T: torch.Tensor
    U_pre: torch.Tensor
    W: torch.Tensor
    Mqk: torch.Tensor
    K_restored: torch.Tensor
    gamma_C: torch.Tensor


@dataclasses.dataclass
class _StageTwoGolden:
    U: torch.Tensor
    O_state: torch.Tensor
    S: torch.Tensor


@dataclasses.dataclass
class _StageThreeGolden:
    O: torch.Tensor


@dataclasses.dataclass
class _ChainGolden:
    stage1: _StageOneGolden
    stage2: _StageTwoGolden
    stage3: _StageThreeGolden


def _to_fp16_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """近似 Stage1 FP16 packed/NZ L1 交接。"""
    return x.to(torch.float16).to(torch.float32)


def _to_bf16_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """近似 BF16 GM 读写精度损失。"""
    return x.to(LOW_DTYPE).to(torch.float32)


def _reference_lower_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    cumulative_decay: torch.Tensor,
    *,
    diagonal: int,
    cast_operand=None,
) -> torch.Tensor:
    """Compute causal QK/KKT tiles with 16-token relative-decay references."""
    result = torch.zeros(
        *left.shape[:-1], right.shape[-2], dtype=torch.float32, device=left.device,
    )
    cast = (lambda x: x) if cast_operand is None else cast_operand
    for row_start in range(0, CHUNK_SIZE, REFERENCE_BLOCK_SIZE):
        row_end = row_start + REFERENCE_BLOCK_SIZE
        reference = cumulative_decay[..., row_start:row_start + 1, :]
        left_rows = cast(
            left[..., row_start:row_end, :] * (cumulative_decay[..., row_start:row_end, :] - reference).exp()
        )
        for col_start in range(0, row_end, REFERENCE_BLOCK_SIZE):
            col_end = col_start + REFERENCE_BLOCK_SIZE
            right_cols = cast(
                right[..., col_start:col_end, :] * (reference - cumulative_decay[..., col_start:col_end, :]).exp()
            )
            result[..., row_start:row_end, col_start:col_end] = left_rows @ right_cols.transpose(-1, -2)

    return result * torch.tril(
        torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32, device=left.device), diagonal=diagonal,
    )


def _npu_neumann_inverse_from_negative_t(negative_t: torch.Tensor) -> torch.Tensor:
    """从 NPU ``-T`` 交接重建驻留逆矩阵（packed64：16→32→64 块下三角组合）。"""
    inverse = torch.zeros_like(negative_t, dtype=torch.float32)
    for block_idx in range(0, CHUNK_SIZE, 16):
        block = _to_fp16_to_fp32(negative_t[block_idx:block_idx + 16, block_idx:block_idx + 16])
        diag_inv = torch.eye(16, dtype=torch.float32, device=negative_t.device) + block
        diag_power2 = block @ block
        diag_inv = diag_inv + _to_fp16_to_fp32(diag_inv) @ _to_fp16_to_fp32(diag_power2)
        diag_power4 = _to_fp16_to_fp32(diag_power2) @ _to_fp16_to_fp32(diag_power2)
        diag_inv = diag_inv + _to_fp16_to_fp32(diag_inv) @ _to_fp16_to_fp32(diag_power4)
        diag_power8 = _to_fp16_to_fp32(diag_power4) @ _to_fp16_to_fp32(diag_power4)
        diag_inv = diag_inv + _to_fp16_to_fp32(diag_inv) @ _to_fp16_to_fp32(diag_power8)
        inverse[block_idx:block_idx + 16, block_idx:block_idx + 16] = diag_inv

    for block_idx in range(0, CHUNK_SIZE, 32):
        even = slice(block_idx, block_idx + 16)
        odd = slice(block_idx + 16, block_idx + 32)
        odd_lower_even = _to_fp16_to_fp32(negative_t[odd, even])
        lower_tmp = odd_lower_even @ _to_fp16_to_fp32(inverse[even, even])
        inverse[odd, even] = _to_fp16_to_fp32(inverse[odd, odd]) @ _to_fp16_to_fp32(lower_tmp)

    even32 = slice(0, 32)
    odd32 = slice(32, 64)
    odd32_lower_even32 = _to_fp16_to_fp32(negative_t[odd32, even32])
    lower32_tmp = odd32_lower_even32 @ _to_fp16_to_fp32(inverse[even32, even32])
    inverse[odd32, even32] = _to_fp16_to_fp32(inverse[odd32, odd32]) @ _to_fp16_to_fp32(lower32_tmp)
    return _to_bf16_to_fp32(inverse)


def _stage1_cpu(K, V, Q, beta_brc, g, *, scale_value) -> _StageOneGolden:
    """Stage1 CPU 公式（全 fp32）：生成不依赖状态的 chunk 工作区。"""
    if V.shape[1] != K.shape[1]:                            # GQA repeat 到 Nv
        repeat = V.shape[1] // K.shape[1]
        K = K.repeat_interleave(repeat, dim=1)
        Q = Q.repeat_interleave(repeat, dim=1)
    K_i = _chunk_major(K).to(torch.float32)
    V_i = _chunk_major(V).to(torch.float32)
    Q_i = _chunk_major(Q).to(torch.float32)
    beta = _chunk_major(beta_brc).to(torch.float32)
    g_i = _chunk_major(g).to(torch.float32)

    cumulative_decay = torch.cumsum(g_i, dim=-2)
    gamma = torch.exp(cumulative_decay)
    K_decayed = gamma * K_i
    K_decayed_beta = beta * K_decayed
    Q_decayed = gamma * Q_i * scale_value

    T = _reference_lower_matmul(beta * K_i, K_i, cumulative_decay, diagonal=-1)

    identity = torch.eye(CHUNK_SIZE, dtype=torch.float32, device=K.device)
    diag_beta = torch.diag_embed(beta.squeeze(-1))
    M = torch.linalg.solve_triangular(identity + T, diag_beta, upper=False, unitriangular=True)
    U_pre = M @ V_i
    W = M @ K_decayed
    Mqk = _reference_lower_matmul(Q_i * scale_value, K_i, cumulative_decay, diagonal=0)
    gamma_C = gamma[..., -1, :, None].contiguous()
    K_restored = K_i * (cumulative_decay[..., -1:, :] - cumulative_decay).exp()
    return _StageOneGolden(gamma, K_decayed_beta, K_decayed, Q_decayed, T,
                           U_pre, W, Mqk, K_restored, gamma_C)


def _stage1_npu(K, V, Q, beta_brc, g, *, scale_value) -> _StageOneGolden:
    """Stage1 NPU-oriented golden：显式 batch/head/chunk 循环 + bf16/fp16 交接。"""
    if V.shape[1] != K.shape[1]:
        rep = V.shape[1] // K.shape[1]
        K = K.repeat_interleave(rep, dim=1)
        Q = Q.repeat_interleave(rep, dim=1)
    K_i = _to_bf16_to_fp32(_chunk_major(K))
    V_i = _chunk_major(V).to(torch.float32)
    Q_i = _to_bf16_to_fp32(_chunk_major(Q))
    beta = _chunk_major(beta_brc).to(torch.float32)
    g_i = _chunk_major(g).to(torch.float32)

    batch, heads, chunks, _, dim = K_i.shape
    gamma = torch.empty_like(K_i)
    K_decayed_beta = torch.empty_like(K_i)
    K_decayed = torch.empty_like(K_i)
    Q_decayed = torch.empty_like(K_i)
    T_out = torch.empty(batch, heads, chunks, CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32, device=K.device)
    U_pre = torch.empty_like(K_i)
    W = torch.empty_like(K_i)
    Mqk = torch.empty_like(T_out)
    K_restored = torch.empty_like(K_i)
    gamma_C = torch.empty(batch, heads, chunks, dim, 1, dtype=torch.float32, device=K.device)
    for b in range(batch):
        for h in range(heads):
            for c in range(chunks):
                K_c = K_i[b, h, c]; V_c = V_i[b, h, c]; Q_c = Q_i[b, h, c]
                beta_c = beta[b, h, c]; g_c = g_i[b, h, c]

                g_cumsum = torch.cumsum(g_c, dim=0)         # AIV 全程 fp32
                gamma_c = torch.exp(g_cumsum)
                K_decayed_prepare = gamma_c * K_c
                K_decayed_c = _to_bf16_to_fp32(K_decayed_prepare)
                Q_decayed_c = _to_bf16_to_fp32(gamma_c * Q_c * scale_value)

                T = _reference_lower_matmul(
                    K_c, K_c, g_cumsum, diagonal=-1, cast_operand=_to_bf16_to_fp32,
                ) * beta_c
                Mqk_c = _to_bf16_to_fp32(_reference_lower_matmul(
                    Q_c * scale_value, K_c, g_cumsum, diagonal=0, cast_operand=_to_bf16_to_fp32,
                ))

                negative_t = _to_fp16_to_fp32(-T)
                resident_inv = _npu_neumann_inverse_from_negative_t(negative_t)

                K_decayed_beta_c = _to_bf16_to_fp32(beta_c * K_decayed_prepare)
                V_beta = _to_bf16_to_fp32(beta_c * V_c)
                U_pre_c = _to_bf16_to_fp32(_to_bf16_to_fp32(resident_inv) @ _to_bf16_to_fp32(V_beta))
                workspace_w = _to_bf16_to_fp32(_to_bf16_to_fp32(resident_inv) @ _to_bf16_to_fp32(-K_decayed_beta_c))
                gamma_c_last = gamma_c[-1].reshape(K_c.shape[-1], 1)
                K_restored_c = _to_bf16_to_fp32(
                    K_c * (g_cumsum[-1:] - g_cumsum).exp()
                )

                gamma[b, h, c] = gamma_c
                K_decayed[b, h, c] = K_decayed_c
                K_decayed_beta[b, h, c] = K_decayed_beta_c
                Q_decayed[b, h, c] = Q_decayed_c
                T_out[b, h, c] = T
                U_pre[b, h, c] = U_pre_c
                W[b, h, c] = -workspace_w
                Mqk[b, h, c] = Mqk_c
                gamma_C[b, h, c] = gamma_c_last
                K_restored[b, h, c] = K_restored_c
    return _StageOneGolden(gamma, K_decayed_beta, K_decayed, Q_decayed, T_out,
                           U_pre, W, Mqk, K_restored, gamma_C)


def _stage2_cpu(stage1: _StageOneGolden, initial_state) -> _StageTwoGolden:
    """Stage2 CPU 公式（全 fp32）：用 stage1 工作区 + 初始 state 递推。"""
    _, _, chunks, _, _ = stage1.U_pre.shape
    U = torch.zeros_like(stage1.U_pre)
    O_state = torch.zeros_like(stage1.U_pre)
    state = initial_state.to(torch.float32).clone()
    U_pre = stage1.U_pre.to(torch.float32)
    W = stage1.W.to(torch.float32)
    Q_decayed = stage1.Q_decayed.to(torch.float32)
    K_restored = stage1.K_restored.to(torch.float32)
    gamma_C = stage1.gamma_C.to(torch.float32)
    for c in range(chunks):
        U_c = U_pre[:, :, c] - W[:, :, c] @ state
        O_c = Q_decayed[:, :, c] @ state
        state = gamma_C[:, :, c] * state + K_restored[:, :, c].transpose(-1, -2) @ U_c
        U[:, :, c] = U_c
        O_state[:, :, c] = O_c
    return _StageTwoGolden(U=U, O_state=O_state, S=state)


def _stage2_npu(stage1: _StageOneGolden, initial_state) -> _StageTwoGolden:
    """Stage2 NPU-oriented golden：显式 chunk 递推 + bf16 交接。"""
    batch, heads, chunks, _, _ = stage1.U_pre.shape
    U = torch.zeros_like(stage1.U_pre, dtype=torch.float32)
    O_state = torch.zeros_like(stage1.U_pre, dtype=torch.float32)
    U_pre = _to_bf16_to_fp32(stage1.U_pre)
    W_neg = _to_bf16_to_fp32(-stage1.W)                     # workspace 存负 W
    Q_decayed = _to_bf16_to_fp32(stage1.Q_decayed)
    K_restored = _to_bf16_to_fp32(stage1.K_restored)
    gamma_C = stage1.gamma_C.to(torch.float32)
    state = initial_state.to(torch.float32).clone()
    for b in range(batch):
        for h in range(heads):
            state_bh = state[b, h]
            for c in range(chunks):
                w_s = _to_bf16_to_fp32(W_neg[b, h, c]) @ _to_bf16_to_fp32(state_bh)
                U_c = _to_bf16_to_fp32(U_pre[b, h, c] + w_s)
                O_c = _to_bf16_to_fp32(Q_decayed[b, h, c]) @ _to_bf16_to_fp32(state_bh)
                kr_u = _to_bf16_to_fp32(K_restored[b, h, c].transpose(-1, -2)) @ _to_bf16_to_fp32(U_c)
                # Current DSL retains the recurrent accumulator in FP32.
                scaled_state = gamma_C[b, h, c] * state_bh
                state_bh = scaled_state + kr_u
                U[b, h, c] = U_c
                O_state[b, h, c] = O_c
            state[b, h] = state_bh
    return _StageTwoGolden(U=U, O_state=O_state, S=state)


def _stage3_cpu(stage1: _StageOneGolden, stage2: _StageTwoGolden) -> _StageThreeGolden:
    """Stage3 CPU 公式：O = O_state + Mqk @ U。"""
    O_i = stage2.O_state.to(torch.float32) + stage1.Mqk.to(torch.float32) @ stage2.U.to(torch.float32)
    batch, heads, chunks, chunk, dim = O_i.shape
    return _StageThreeGolden(O=O_i.reshape(batch, heads, chunks * chunk, dim).contiguous())


def _stage3_npu(stage1: _StageOneGolden, stage2: _StageTwoGolden) -> _StageThreeGolden:
    """Stage3 NPU-oriented golden：显式 chunk 合成 + bf16 交接。"""
    batch, heads, chunks, chunk, dim = stage2.O_state.shape
    O_i = torch.empty_like(stage2.O_state, dtype=torch.float32)
    # Q@state and Mqk@U accumulate into one FP32 L0C tile.
    O_state = stage2.O_state.float()
    Mqk = _to_bf16_to_fp32(stage1.Mqk)
    U = _to_bf16_to_fp32(stage2.U)
    for b in range(batch):
        for h in range(heads):
            for c in range(chunks):
                mqk_u = _to_bf16_to_fp32(Mqk[b, h, c]) @ _to_bf16_to_fp32(U[b, h, c])
                O_i[b, h, c] = _to_bf16_to_fp32(O_state[b, h, c] + mqk_u)
    return _StageThreeGolden(O=O_i.reshape(batch, heads, chunks * chunk, dim).contiguous())


def _chain_cpu(inputs: KdaInputs) -> _ChainGolden:
    s1 = _stage1_cpu(inputs.K, inputs.V, inputs.Q, inputs.beta_brc, inputs.g, scale_value=inputs.scale_value)
    s2 = _stage2_cpu(s1, inputs.initial_state)
    s3 = _stage3_cpu(s1, s2)
    return _ChainGolden(stage1=s1, stage2=s2, stage3=s3)


def _chain_npu(inputs: KdaInputs) -> _ChainGolden:
    s1 = _stage1_npu(inputs.K, inputs.V, inputs.Q, inputs.beta_brc, inputs.g, scale_value=inputs.scale_value)
    s2 = _stage2_npu(s1, inputs.initial_state)
    s3 = _stage3_npu(s1, s2)
    return _ChainGolden(stage1=s1, stage2=s2, stage3=s3)


def compute_golden(name: str, inputs: KdaInputs):
    """Apply activation/normalization, then pad with an identity recurrence tail."""
    tokens = inputs.Q.shape[-2]
    alpha = torch.exp(inputs.A_log).view(1, -1, 1, 1)
    bias = inputs.dt_bias.view(1, -1, 1, inputs.dt_bias.shape[-1])
    activated = dataclasses.replace(
        inputs,
        Q=inputs.Q.float() / torch.sqrt(inputs.Q.float().square().sum(-1, keepdim=True) + 1e-6),
        K=inputs.K.float() / torch.sqrt(inputs.K.float().square().sum(-1, keepdim=True) + 1e-6),
        g=inputs.lower_bound * torch.sigmoid(alpha * (inputs.g.float() + bias)),
        beta_brc=torch.sigmoid(inputs.beta_brc.float()),
    )
    # Zero activated decay means exp(g)=1; beta=0 means no state update.
    padding = (-tokens) % CHUNK_SIZE
    if padding:
        activated = dataclasses.replace(activated, **{
            name: F.pad(getattr(activated, name), (0, 0, 0, padding))
            for name in ("Q", "K", "V", "g", "beta_brc")
        })
    chain = _chain_cpu(activated) if name == "cpu_chunk" else _chain_npu(activated)
    return chain.stage3.O[:, :, :tokens].contiguous(), chain.stage2.S.transpose(-1, -2).contiguous()


pytestmark = pytest.mark.npu
_LOWER_BOUND = float(os.environ.get("KDA_LOWER_BOUND") or -5.0)
_TOL = float(os.environ.get("KDA_TOL", "5e-3"))
_DYNAMIC_KDA = load_sample("flash_kda/flash_kda.py")
_DYNAMIC_METADATA = load_sample("flash_kda_metadata/flash_kda_metadata.py")


def require_npu():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")


@dataclasses.dataclass(frozen=True)
class _DynamicPrecisionCase:
    layout: str
    has_cu: bool
    lengths: tuple[int, ...]
    nv: int
    nk: int
    seed: int

    @property
    def case_id(self):
        lengths = "-".join(str(length) for length in self.lengths)
        suffix = "-cu" if self.has_cu else ""
        return f"{self.layout.lower()}{suffix}_n{self.nv}k{self.nk}_s{lengths}_seed{self.seed}"


_DYNAMIC_FAST_CASES = (
    _DynamicPrecisionCase("BNSD", False, (513,), 1, 1, 101),
    _DynamicPrecisionCase("BSND", False, (512,), 2, 1, 102),
    _DynamicPrecisionCase("TND", True, (1, 512, 513), 3, 1, 105),
)

# Each representative ragged case covers a distinct internal tile boundary.
_WHITEBOX_TAIL_CASES = (
    _DynamicPrecisionCase("BNSD", True, (1, 15, 16, 17), 3, 1, 4100),
    _DynamicPrecisionCase("BSND", True, (31, 32, 33, 47, 48, 49), 5, 1, 4105),
    _DynamicPrecisionCase("TND", True, (63, 64, 65), 3, 1, 4110),
    _DynamicPrecisionCase("TND", True, (127, 128, 129), 3, 1, 4111),
)
_WHITEBOX_MULTIROUND_CASES = (
    # 96 * (4 + 3 + 2) = 864 chunk tasks cross the 861-slot boundary.
    _DynamicPrecisionCase("TND", True, (193, 129, 65), 96, 3, 4201),
)


def _make_dynamic_case(case):
    inputs = make_inputs_generalized(
        len(case.lengths), case.nv, max(case.lengths), 128, nk=case.nk, seed=case.seed, lower_bound=_LOWER_BOUND,
    )
    outputs = {name: [] for name in ("cpu_chunk", "npu_chunk")}
    states = {name: [] for name in outputs}
    for batch_idx, length in enumerate(case.lengths):
        one = dataclasses.replace(
            inputs,
            K=inputs.K[batch_idx:batch_idx + 1, :, :length],
            V=inputs.V[batch_idx:batch_idx + 1, :, :length],
            Q=inputs.Q[batch_idx:batch_idx + 1, :, :length],
            beta_brc=inputs.beta_brc[batch_idx:batch_idx + 1, :, :length],
            g=inputs.g[batch_idx:batch_idx + 1, :, :length],
            initial_state=inputs.initial_state[batch_idx:batch_idx + 1],
        )
        for name in outputs:
            output, state = compute_golden(name, one)
            outputs[name].append(output)
            states[name].append(state)
    return inputs, outputs, {name: torch.cat(values, dim=0) for name, values in states.items()}


def _dynamic_layout_inputs(inputs, case):
    beta = inputs.beta_brc.squeeze(-1)
    if case.layout == "BNSD":
        return inputs.Q, inputs.K, inputs.V, inputs.g, beta
    if case.layout == "BSND":
        return tuple(tensor.transpose(1, 2).contiguous() for tensor in (inputs.Q, inputs.K, inputs.V, inputs.g, beta))

    def packed(tensor):
        return torch.cat(
            [tensor[b, :, :length].transpose(0, 1) for b, length in enumerate(case.lengths)],
            dim=0,
        ).contiguous()

    return tuple(packed(tensor) for tensor in (inputs.Q, inputs.K, inputs.V, inputs.g, beta))


def _dynamic_valid_outputs(output, case):
    if case.layout == "TND":
        result = []
        start = 0
        for length in case.lengths:
            result.append(output[start:start + length].transpose(0, 1).unsqueeze(0))
            start += length
        return result
    if case.layout == "BSND":
        output = output.transpose(1, 2)
    return [output[b:b + 1, :, :length] for b, length in enumerate(case.lengths)]


def _assert_schedule_covers_logical_chunks(metadata, case):
    schedule = _DYNAMIC_METADATA.decode_metadata(metadata.cpu().tolist())
    assert schedule["status"] == _DYNAMIC_METADATA.STATUS_OK
    assert schedule["batch"] == len(case.lengths)
    assert schedule["value_heads"] == case.nv
    chunk_cursor = [0] * len(case.lengths)
    starts = [sum(case.lengths[:batch]) for batch in range(len(case.lengths))]
    for index, record in enumerate(schedule["stage12_rounds"]):
        assert record["stage12_round_idx"] == index
        for active, batch in enumerate(record["batch_idx"]):
            assert record["valid_seq_len"][active] == case.lengths[batch]
            assert record["storage_batch_idx"][active] == (0 if case.layout == "TND" else batch)
            assert record["token_start"][active] == (starts[batch] if case.layout == "TND" else 0)
            assert record["chunk_start_per_group"][active] == chunk_cursor[batch]
            count = record["chunk_num_per_group"][active]
            assert 0 < count <= (case.lengths[batch] + 63) // 64 - chunk_cursor[batch]
            assert record["stage1_task_prefix"][active + 1] - record["stage1_task_prefix"][active] == case.nv * count
            chunk_cursor[batch] += count
        assert record["stage1_task_prefix"][-1] <= _DYNAMIC_METADATA.WORKSPACE_SLOTS
    assert chunk_cursor == [(length + 63) // 64 for length in case.lengths]
    return schedule


def _assert_dynamic_precision_case(case, *, whitebox=False):
    require_npu()
    inputs, golden_outputs, golden_state = _make_dynamic_case(case)
    if whitebox and case.has_cu and case.layout != "TND":
        # Invalid rows must never affect valid outputs or the final state.
        # Padding output is undefined, so only poison inputs, not expected output.
        for batch, length in enumerate(case.lengths):
            for tensor in (inputs.Q, inputs.K, inputs.V, inputs.g, inputs.beta_brc):
                tensor[batch, :, length:] = float("nan")
    q, k, v, g, beta = (tensor.npu() for tensor in _dynamic_layout_inputs(inputs, case))
    cu_seqlens = None
    if case.has_cu:
        cu_seqlens = torch.tensor((0, *torch.tensor(case.lengths).cumsum(0).tolist()), dtype=torch.int32).npu()
    initial_state = inputs.initial_state.transpose(-1, -2).contiguous().npu()
    initial_state_before = initial_state.clone()
    metadata = _DYNAMIC_METADATA.flash_kda_metadata(q, v, initial_state, case.layout, cu_seqlens)
    if whitebox:
        schedule = _assert_schedule_covers_logical_chunks(metadata, case)
        if sum((length + 63) // 64 for length in case.lengths) * case.nv > _DYNAMIC_METADATA.WORKSPACE_SLOTS:
            assert schedule["stage12_round_num"] > 1
    output, state = _DYNAMIC_KDA.flash_kda(
        q, k, v, g, beta, inputs.scale_value, initial_state, inputs.A_log.npu(), inputs.dt_bias.npu(),
        inputs.lower_bound, case.layout, metadata,
    )
    torch.npu.synchronize()

    actual_state = state.cpu().float()
    assert torch.isfinite(actual_state).all(), f"{case.case_id}: final_state contains inf/nan"
    torch.testing.assert_close(initial_state.cpu().float(), initial_state_before.cpu().float(), atol=0.0, rtol=0.0)
    actual_outputs = [actual.cpu().float() for actual in _dynamic_valid_outputs(output, case)]
    for name in ("cpu_chunk", "npu_chunk"):
        expected_state = golden_state[name].float()
        assert torch.isfinite(expected_state).all(), f"{case.case_id}: {name} state is not finite"
        torch.testing.assert_close(actual_state, expected_state, atol=_TOL, rtol=_TOL,
                                   msg=lambda msg: f"DSL vs {name} state: {msg}")
        output_max = 0.0
        for actual, expected in zip(actual_outputs, golden_outputs[name]):
            assert torch.isfinite(actual).all(), f"{case.case_id}: DSL output is not finite"
            assert torch.isfinite(expected).all(), f"{case.case_id}: {name} output is not finite"
            torch.testing.assert_close(actual, expected.float(), atol=_TOL, rtol=_TOL,
                                       msg=lambda msg: f"DSL vs {name} output: {msg}")
            output_max = max(output_max, (actual - expected).abs().amax().item())
        state_max = (actual_state - expected_state).abs().amax().item()
        print(f"{case.case_id} vs {name}: O max_abs={output_max:.3e} state max_abs={state_max:.3e}")
    torch.testing.assert_close(golden_state["npu_chunk"], golden_state["cpu_chunk"], atol=_TOL, rtol=_TOL,
                               msg=lambda msg: f"npu_chunk vs cpu_chunk state: {msg}")
    for simulated, fp32 in zip(golden_outputs["npu_chunk"], golden_outputs["cpu_chunk"]):
        torch.testing.assert_close(simulated, fp32, atol=_TOL, rtol=_TOL,
                                   msg=lambda msg: f"npu_chunk vs cpu_chunk output: {msg}")


@pytest.mark.parametrize("case", _DYNAMIC_FAST_CASES, ids=lambda case: case.case_id)
def test_flash_kda_dynamic_precision(case):
    _assert_dynamic_precision_case(case)


@pytest.mark.parametrize("case", _WHITEBOX_TAIL_CASES + _WHITEBOX_MULTIROUND_CASES,
                         ids=lambda case: case.case_id)
def test_flash_kda_non_aligned_whitebox(case):
    _assert_dynamic_precision_case(case, whitebox=True)


def teardown_module():
    _DYNAMIC_KDA.clear_caches()
    _DYNAMIC_METADATA.clear_metadata_caches()
