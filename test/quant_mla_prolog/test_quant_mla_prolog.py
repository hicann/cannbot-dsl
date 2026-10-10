# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public-ABI and four-network-shape tests for MXFP8 MLA Prolog."""

import importlib.util
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

# Make the test runnable through either pytest or an absolute Python path from
# any working directory. Pytest normally adds test/ through conftest.py, while
# direct execution does not load conftest before this module is imported.
TEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TEST_ROOT.parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from _samples_path import load_sample  # noqa: E402


public_module = load_sample("quant_mla_prolog/quant_mla_prolog.py")
sys.modules.setdefault("quant_mla_prolog", public_module)
LOGGER = logging.getLogger(__name__)


def _load_golden():
    path = Path(__file__).with_name("golden.py")
    spec = importlib.util.spec_from_file_location("quant_mla_prolog_golden", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden = _load_golden()
NETWORK_CASE_IDS = (
    "case_013_aw1_c1",  # B=1, S=8, T=8
    golden.SINGLE_MIX_O11_CASE_ID,  # B=4, S=8, T=32
    "case_016_aw1_c1",  # B=16, S=8, T=128
)
EXPECTED_TOKENS = (8, 32, 128)
pytestmark = pytest.mark.npu


def _assert_all_finite(name, value):
    """Reject NaN/Inf before numerical error metrics can hide them."""
    finite = torch.isfinite(value.float())
    if not bool(finite.all().item()):
        invalid = int((~finite).sum().item())
        raise AssertionError(f"{name} contains {invalid} NaN/Inf values")


def _capture_c1_kernel(public_call):
    """Keep the actual call-private buffers for a poison/recovery regression."""
    getter = getattr(public_module, "_get_native_program")
    captured = []

    def capture_program(*args):
        program = getter(*args)

        def launch(*kernel_args):
            captured.append((program, list(kernel_args)))
            return program(*kernel_args)

        return launch

    with patch.object(public_module, "_get_native_program", capture_program):
        outputs = public_call()
    assert len(captured) == 1
    return outputs, captured[0]


def _assert_c1_workspace_recovery(captured, outputs, reference, physical_cache, tokens):
    """Stale workspace contents must not affect the next finite invocation.

    This reuses exactly the buffers captured from the public torch call. NaN
    input is intentional test pollution; no diagnostic path enters the kernel.
    Three repetitions guard against a previous invocation leaking into QB/WKB
    consumers. Controlled producer-delay testing is run separately.
    """
    program, kernel_args = captured
    clean_x = kernel_args[2]
    poison_x = (
        torch.full(tuple(clean_x.shape), float("nan"), dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .npu()
        .view(torch.int8)
    )
    poison_args = list(kernel_args)
    poison_args[2] = poison_x
    for iteration in range(3):
        program(*poison_args)
        # Keep both device invocations consecutive on the current stream.
        program(*kernel_args)
        torch.npu.synchronize()
        q = outputs["q"].cpu()
        scale = outputs["descale_q"].cpu()
        cache = physical_cache.cpu()
        _assert_all_finite(f"recovery[{iteration}].q", q)
        _assert_all_finite(f"recovery[{iteration}].descale_q", scale)
        torch.testing.assert_close(
            q.view(torch.uint8), reference["q"].view(torch.uint8), atol=0, rtol=0
        )
        torch.testing.assert_close(scale, reference["descale_q"], atol=0, rtol=0)
        torch.testing.assert_close(
            cache.view(torch.uint8),
            reference["kv_cache_out"].view(torch.uint8),
            atol=0,
            rtol=0,
        )
        # Inspect the first known stale-data boundary, not only the epilog.
        _assert_all_finite("recovery.q_head", kernel_args[18][:tokens].cpu())
        _assert_all_finite("recovery.q_latent_h", kernel_args[19][:, :tokens].cpu())


@pytest.mark.parametrize(
    ("case_id", "tokens"),
    tuple(zip(NETWORK_CASE_IDS, EXPECTED_TOKENS)),
    ids=("b1_s8", "b4_s8", "b16_s8"),
)
def test_network_mxfp8_precision(case_id, tokens):
    pytest.importorskip("torch_npu")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU unavailable")

    case = golden.load_case(case_id)
    inputs, expected = golden.run_case(case)
    assert case["T"] == tokens

    public_scales = (
        golden.logical_descale(inputs["descale_wqa"], 1536, 7168).npu(),
        golden.logical_descale(inputs["descale_wqb"], 96 * 192, 1536).npu(),
        golden.logical_descale(inputs["descale_wkva"], 576, 7168).npu(),
    )
    physical_cache = inputs["kv_cache"].npu()
    blocks, _, block_size, _ = physical_cache.shape
    logical_cache = physical_cache.view(blocks, block_size, 1, 576)

    actual, captured = _capture_c1_kernel(
        lambda: public_module.quant_mla_prolog(
            inputs["x"].npu(),
            inputs["wqa"].npu(),
            inputs["wqb"].npu(),
            inputs["wkva"].npu(),
            inputs["wkb"].npu(),
            inputs["descale_x"].npu(),
            *public_scales,
            inputs["norm_weight_qa"].npu(),
            inputs["norm_weight_kva"].npu(),
            logical_cache,
            inputs["cache_index"].npu(),
            inputs["qscale_kv"].npu(),
            norm_eps=case["norm_eps"],
            quant_mode_aw=1,
            quant_mode_c=1,
        )
    )
    torch.npu.synchronize()

    assert list(actual) == ["q", "kv_cache_out", "descale_q"]
    assert actual["kv_cache_out"] is logical_cache
    assert actual["kv_cache_out"].data_ptr() == logical_cache.data_ptr()
    assert tuple(actual["q"].shape) == (tokens, 96, 576)
    assert actual["q"].dtype == torch.float8_e4m3fn
    assert tuple(actual["descale_q"].shape) == (tokens, 96)
    assert actual["descale_q"].dtype == torch.float32

    actual_cpu = {
        "q": actual["q"].cpu(),
        "descale_q": actual["descale_q"].cpu(),
        "kv_cache_out": actual["kv_cache_out"].view_as(physical_cache).cpu(),
    }
    report = golden.compare_outputs(
        actual_cpu,
        expected,
        inputs["cache_index"],
        quant_mode_c=1,
        qscale_kv=inputs["qscale_kv"],
    )
    _assert_all_finite("q", actual_cpu["q"])
    _assert_all_finite("descale_q", actual_cpu["descale_q"])
    q_actual = actual_cpu["q"].float() * actual_cpu["descale_q"][..., None]
    q_expected = expected["q"].float() * expected["descale_q"][..., None]
    gather_cache = getattr(golden, "_gather_cache")
    relative_l2 = getattr(golden, "_relative_l2")
    kv_actual = gather_cache(actual_cpu["kv_cache_out"], inputs["cache_index"]).float()
    kv_expected = gather_cache(expected["kv_cache_out"], inputs["cache_index"]).float()
    _assert_all_finite("kv_cache_written", kv_actual)
    q_relative_l2 = relative_l2(q_actual, q_expected)
    descale_relative_l2 = relative_l2(actual_cpu["descale_q"], expected["descale_q"])
    kv_relative_l2 = relative_l2(
        kv_actual * inputs["qscale_kv"],
        kv_expected * inputs["qscale_kv"],
    )
    metrics = {
        "case_id": case_id,
        "tokens": tokens,
        "q_relative_l2": q_relative_l2,
        "descale_relative_l2": descale_relative_l2,
        "kv_relative_l2": kv_relative_l2,
        "q_cosine": report["q_dequantized"]["cosine_similarity"],
    }
    LOGGER.info("precision metrics:\n%s", json.dumps(metrics, indent=2))

    # Match the standalone network script's accuracy contract. Encoded FP8
    # mismatch ratios remain diagnostics; reconstructed values determine the
    # model-facing pass/fail result.
    assert report["passed"]
    assert q_relative_l2 < golden.NETWORK_RELATIVE_L2_LIMIT
    assert descale_relative_l2 < golden.NETWORK_RELATIVE_L2_LIMIT
    assert kv_relative_l2 < golden.NETWORK_RELATIVE_L2_LIMIT
    assert report["q_dequantized"]["cosine_similarity"] > golden.NETWORK_Q_COSINE_MIN
    assert report["kv_cache_untouched"]["bitwise_equal"]
    _assert_c1_workspace_recovery(captured, actual, actual_cpu, physical_cache, tokens)
