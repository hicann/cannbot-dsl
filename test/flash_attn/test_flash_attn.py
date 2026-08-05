# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for the Flash Attention kernel (FlashAttn).

Formula:
    O = softmax(Q @ K^T * scale) @ V

mask_mode:
  * 0 -- full (no-mask) attention.
  * 3 -- right-context causal.  Query position *i* attends to key position *j*
    iff *j <= i + offset* where *offset = S2 - S1*.  The causal diagonal is
    derived internally from *mask_mode + shapes*; no external mask tensor is
    passed (each causal diagonal tile is treated as a per-row N-tail).

Constraints honoured by the shape list:
  * D == 128 (kernel ``get_tile_config`` branch).
  * N1 % N2 == 0 (GQA group size is an integer).

Coverage of the 12 test cases:
  * dtype      : float16, bfloat16
  * layout     : BNSD [B,N,S,D], BSND [B,S,N,D]
  * mask_mode  : 0 (full), 3 (causal)
  * group size : g=1 (MHA), g=2, g=8, g=10, g=16
  * scenario   : prefill, decode (S1=1), MTP (1<S1<=4), causal
  * scale      : small (~25us) to large (~24000us)
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "samples", "flash_attn"))

from cannbotdsl import dtypes
from cannbotdsl.runtime import from_torch_npu

from flash_attn import flash_attn


_TOLERANCE = {
    torch.float16:  (1e-3, 1e-3),
    torch.bfloat16: (2e-2, 2e-2),
}

_TORCH_TO_DSL = {
    torch.float16: dtypes.float16,
    torch.bfloat16: dtypes.bfloat16,
}


# ---------------------------------------------------------------------------
# Test cases
#   Each tuple: (B, N1, N2, S1, S2, D, mask_mode, layout, dtype)
# ---------------------------------------------------------------------------

_TEST_CASES = [
    # 1. decode g=10, small S2
    pytest.param(1, 10, 1, 1, 4096, 128, 0, "BSND", torch.bfloat16,
                 id="BF16_L0_1_10_1_1_4096_BSND_decode_g10"),
    # 2. decode g=8, nonaligned S2
    pytest.param(1, 32, 4, 1, 4160, 128, 0, "BSND", torch.bfloat16,
                 id="BF16_L0_1_32_4_1_4160_BSND_decode_g8"),
    # 3. decode g=8, large S2
    pytest.param(1, 32, 4, 1, 32768, 128, 0, "BSND", torch.bfloat16,
                 id="BF16_L0_1_32_4_1_32768_BSND_decode_g8"),
    # 4. MTP g=1, S1=2
    pytest.param(1, 24, 24, 2, 16384, 128, 0, "BSND", torch.bfloat16,
                 id="BF16_L0_1_24_24_2_16384_BSND_mtp"),
    # 5. MTP g=1, S1=4
    pytest.param(1, 24, 24, 4, 32768, 128, 0, "BSND", torch.bfloat16,
                 id="BF16_L0_1_24_24_4_32768_BSND_mtp"),
    # 6. prefill g=1, medium
    pytest.param(1, 3, 3, 4608, 4608, 128, 0, "BSND", torch.bfloat16,
                 id="BF16_L0_1_3_3_4608_4608_BSND"),
    # 7. prefill g=1, large, fp16
    pytest.param(8, 8, 8, 8000, 8000, 128, 0, "BNSD", torch.float16,
                 id="FP16_L0_8_8_8_8000_8000_BNSD"),
    # 8. prefill g=1, huge
    pytest.param(32, 6, 6, 9216, 9216, 128, 0, "BNSD", torch.bfloat16,
                 id="BF16_L0_32_6_6_9216_9216_BNSD"),
    # 9. causal g=16, medium
    pytest.param(1, 64, 4, 2048, 2048, 128, 3, "BSND", torch.bfloat16,
                 id="BF16_L3_1_64_4_2048_2048_BSND"),
    # 10. causal g=16, large
    pytest.param(1, 16, 1, 16384, 16384, 128, 3, "BSND", torch.bfloat16,
                 id="BF16_L3_1_16_1_16384_16384_BSND"),
    # 11. prefill g=1, medium
    pytest.param(32, 8, 8, 1280, 1280, 128, 0, "BNSD", torch.bfloat16,
                 id="BF16_L0_32_8_8_1280_1280_BNSD"),
    # 12. prefill g=2, fp16
    pytest.param(32, 12, 6, 1280, 1280, 128, 0, "BSND", torch.float16,
                 id="FP16_L0_32_12_6_1280_1280_BSND_g2"),
]


# ---------------------------------------------------------------------------
# Golden kit
# ---------------------------------------------------------------------------


def make_inputs(B, N1, N2, S1, S2, D, layout, dtype, seed=42):
    """Build deterministic Q/K/V on NPU and CPU golden views.

    Returns ``(q_npu, k_npu, v_npu, q_bnsd, k_bnsd, v_bnsd)`` where
    the ``*_npu`` tensors live on the NPU device and the ``*_bnsd`` tensors
    are CPU views in logical [B,N,S,D] layout for golden computation.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if layout == "BNSD":
        q_cpu = torch.randn(B, N1, S1, D, generator=gen, dtype=dtype)
        k_cpu = torch.randn(B, N2, S2, D, generator=gen, dtype=dtype)
        v_cpu = torch.randn(B, N2, S2, D, generator=gen, dtype=dtype)
        q_bnsd, k_bnsd, v_bnsd = q_cpu, k_cpu, v_cpu
    else:
        q_cpu = torch.randn(B, S1, N1, D, generator=gen, dtype=dtype)
        k_cpu = torch.randn(B, S2, N2, D, generator=gen, dtype=dtype)
        v_cpu = torch.randn(B, S2, N2, D, generator=gen, dtype=dtype)
        q_bnsd = q_cpu.transpose(1, 2)
        k_bnsd = k_cpu.transpose(1, 2)
        v_bnsd = v_cpu.transpose(1, 2)
    return q_cpu.npu(), k_cpu.npu(), v_cpu.npu(), q_bnsd, k_bnsd, v_bnsd


def make_attn_mask(mask_mode):
    """Create the compressed 2048x2048 causal mask for mask_mode==3."""
    if mask_mode != 3:
        return None
    mask_size = 2048
    attn_mask_cpu = torch.zeros(mask_size, mask_size, dtype=torch.float32)
    causal = torch.ones(mask_size, mask_size, dtype=torch.bool).tril()
    attn_mask_cpu = attn_mask_cpu.masked_fill(~causal, -1e30)
    return attn_mask_cpu.npu()


def flash_attn_golden(q_bnsd, k_bnsd, v_bnsd, scale, group, mask_mode, S1, S2, dtype):
    """CPU golden for Flash Attention.

    For mask_mode==0: full (no-mask) attention.
    For mask_mode==3: right-context causal attention where query position *i*
    attends to key position *j* iff *j <= i + (S2 - S1)*.
    """
    k_ref = k_bnsd.repeat_interleave(group, dim=1).float()
    v_ref = v_bnsd.repeat_interleave(group, dim=1).float()
    s_cpu = torch.matmul(q_bnsd.float(), k_ref.transpose(-2, -1)) * scale
    if mask_mode == 3:
        offset = S2 - S1
        causal = torch.ones(S1, S2, dtype=torch.bool).tril(diagonal=offset)
        s_cpu = s_cpu.masked_fill(~causal, float("-inf"))
    p_cpu = torch.softmax(s_cpu, dim=-1)
    if mask_mode == 3:
        p_cpu = torch.nan_to_num(p_cpu, nan=0.0)
    return torch.matmul(p_cpu, v_ref).to(dtype)


# ---------------------------------------------------------------------------
# NPU accuracy tests — FlashAttn DSL kernel
# ---------------------------------------------------------------------------


@pytest.mark.npu
@pytest.mark.parametrize("B,N1,N2,S1,S2,D,mask_mode,layout,dtype", _TEST_CASES)
def test_flash_attn_precision(B, N1, N2, S1, S2, D, mask_mode, layout, dtype):
    pytest.importorskip("torch_npu")

    q_npu, k_npu, v_npu, q_bnsd, k_bnsd, v_bnsd = make_inputs(
        B, N1, N2, S1, S2, D, layout, dtype,
    )

    scale = 1.0 / math.sqrt(D)
    attn_mask = make_attn_mask(mask_mode)

    result = flash_attn(
        q_npu, k_npu, v_npu, scale,
        mask_mode=mask_mode, attn_mask=attn_mask,
        layout_q=layout, layout_kv=layout, layout_out=layout,
        dtype=_TORCH_TO_DSL[dtype],
    )
    torch.npu.synchronize()

    if layout == "BNSD":
        result_npu = result[:, :, :S1, :].cpu()
    else:
        result_npu = result[:, :S1, :, :].cpu()

    group = N1 // N2
    golden = flash_attn_golden(q_bnsd, k_bnsd, v_bnsd, scale, group, mask_mode, S1, S2, dtype)
    if layout == "BSND":
        golden = golden.transpose(1, 2).contiguous()

    atol, rtol = _TOLERANCE[dtype]
    max_err = (result_npu.float() - golden.float()).abs().max().item()
    assert torch.allclose(result_npu, golden, atol=atol, rtol=rtol), (
        f"flash_attn mismatch: "
        f"B={B}, N1={N1}, N2={N2}, S1={S1}, S2={S2}, D={D}, "
        f"mask_mode={mask_mode}, layout={layout}, dtype={dtype}, "
        f"max|err|={max_err:.4e}"
    )
    print(
        f"flash_attn "
        f"(B={B},N1={N1},N2={N2},S1={S1},S2={S2},D={D},"
        f"mask_mode={mask_mode},layout={layout},dtype={dtype}) "
        f"max|err|={max_err:.4e} finished!"
    )
