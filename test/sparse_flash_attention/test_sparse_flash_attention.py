# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from pathlib import Path
import sys
import logging
import torch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "samples" / "sparse_flash_attention"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]-%(filename)s:%(lineno)04d - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


def cal_relative_diff_np_isclose(real_data, expect_data, type_str="fp16"):
    diff = abs(float(real_data) - float(expect_data))
    result = diff / (np.abs(expect_data) + 10e-10)
    return result


def display_output_np_isclose(
    real_data, expect_data, start, end, expect_fp32_data=None
):
    def display_inner(idx):
        j = idx + start
        diff_rate = cal_relative_diff_np_isclose(real_data[j], expect_data[j])

        is_special = "inf" in str(expect_data[j]) or "nan" in str(expect_data[j])
        if is_special:
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
        else:
            diff_abs = abs(np.float64(expect_data[j]) - np.float64(real_data[j]))

        fields = (expect_data[j], real_data[j], diff_abs, diff_rate)
        if expect_fp32_data is not None:
            fields = (expect_fp32_data[j], *fields)
        value_format = "%-7s" if is_special else "%0.7f"
        line_format = "%08d \t " + " \t ".join([value_format] * len(fields))
        print_log(line_format % (start + idx + 1, *fields))

    print_log(
        "---------------------------------------------------------------------------------------"
    )
    if expect_fp32_data is not None:
        print_log(
            "Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff"
        )
    else:
        print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log("...   \t   ...   \t   ...   \t   ...    \t   ...")
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)


def print_log(data=None, level="INFO"):
    logger.log(getattr(logging, level.upper()), "%s", data, stacklevel=2)


def _contains_nan_or_inf(data):
    text = str(data)
    return "nan" in text or "inf" in text


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log(
        "Error Line-----------------------------------------------------------------------------"
    )
    print_log("Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    i,
                    expect_data[i],
                    real_data[i],
                    abs(np.float64(expect_data[i]) - np.float64(real_data[i])),
                    relative_diff[count - 1],
                )
            )
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = "..."
            print_log(
                "%08s \t %07s \t %07s \t %07s \t %07s"
                % (dot_3, dot_3, dot_3, dot_3, dot_3)
            )
        elif count > 100:
            break

    print_log(
        "Max-RE line:---------------------------------------------------------------------------"
    )
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log(
                "%08d \t %.7f \t %.7f \t %.7f \t %.7f"
                % (
                    m_idx,
                    expect_data[m_idx],
                    real_data[m_idx],
                    abs(np.float64(expect_data[m_idx]) - np.float64(real_data[m_idx])),
                    max_error,
                )
            )
        else:
            break
    print_log(
        "---------------------------------------------------------------------------------------"
    )


def check_result(expect, npu_result):
    diff_thd = 0.005
    pct_thd = 0.005
    max_diff_hd = 10
    rtol = 0.005
    atol = 0.000025
    max_error_idx = 10000000

    real_data = npu_result.cpu().to(torch.float).numpy()
    data_compe = expect.cpu().to(torch.float).numpy()
    real_data = real_data.flatten()
    data_compe = data_compe.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size:
        print_log(
            'The npu_output is [],and it is same as bm_output, the result of data_compare is "Pass"'
        )
        return "Pass", 100.0, 0
    start = 0
    end = real_data.size - 1
    if end < start:
        end = start
    max_error = 0
    result = "Failed"

    if real_data.size != data_compe.size:
        print_log(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal."
            % (real_data.size, data_compe.size)
        )
        return result, 0.0, max_error
    overflows_count = (
        data_compe[np.isinf(data_compe)].size + data_compe[np.isnan(data_compe)].size
    )

    if overflows_count > 0:
        print_log(
            "Overflow,size:%s,benchmark_output:%s, %s"
            % (
                overflows_count,
                data_compe[np.isinf(data_compe)][0:10],
                data_compe[np.isnan(data_compe)][0:10],
            )
        )

    split_count = int(end - start + 1) if end != start else 1
    print_log("split_count:%s; max_diff_hd:%s;" % (float(split_count), max_diff_hd))

    has_nan_inf = (
        _contains_nan_or_inf(real_data) or _contains_nan_or_inf(data_compe)
    )

    if npu_result.dtype == torch.bfloat16:
        rtol = 0.0078125
        atol = 0.0001
        diff_result = np.isclose(
            real_data.astype(np.float32),
            data_compe.astype(np.float32),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    elif npu_result.dtype == "float8_e4m3fn":
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        arr_string = real_data.tobytes()
        real_data = np.frombuffer(arr_string, dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        arr_string = data_compe.tobytes()
        data_compe = np.frombuffer(arr_string, dtype="uint8")
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    elif npu_result.dtype == "float8_e5m2":
        nan_mask = np.isnan(real_data)
        real_data[nan_mask] = 0
        nan_pos_inf = np.isposinf(real_data)
        real_data[nan_pos_inf] = 57344
        nan_neg_inf = np.isneginf(real_data)
        real_data[nan_neg_inf] = -57344

        arr_string = real_data.tobytes()
        real_data = np.frombuffer(arr_string, dtype="uint8")
        nan_mask = np.isnan(data_compe)
        data_compe[nan_mask] = 0
        nan_pos_inf = np.isposinf(data_compe)
        data_compe[nan_pos_inf] = 57344
        nan_neg_inf = np.isneginf(data_compe)
        data_compe[nan_neg_inf] = -57344

        arr_string = data_compe.tobytes()
        data_compe = np.frombuffer(arr_string, dtype="uint8")
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    else:
        diff_result = np.isclose(
            real_data, data_compe, rtol=rtol, atol=atol, equal_nan=True
        )
    err_idx = np.where(diff_result != np.array((True,)))[0]

    if str(data_compe.dtype) == "bool":
        data_compe = data_compe.astype(np.int8)
        real_data = real_data.astype(np.int8)
    diff_abs = abs(data_compe - real_data)
    b1 = np.maximum(np.abs(real_data), (np.abs(data_compe)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    eps = 10e-10
    err_diff = diff_abs / (b + eps)
    err_diff = err_diff[err_idx]

    fulfill_percent = float(split_count - err_idx.size) / float(split_count) * 100.0

    display_output_np_isclose(real_data, data_compe, start, end)
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error >= max_diff_hd:
            result = "Failed"
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log("Rtol   \t Atol   \t PctThd   \t PctRlt   \t Result")
    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log(
        "%.4f    \t %.6f  \t %.2f%%   \t %.6f%%   \t %s"
        % (rtol, atol, pct_thd, fulfill_percent, result)
    )
    if len(err_diff) > 0:
        print_log(
            "Max-RelativeError is: %s. Threshold is: %s." % (max_error, max_diff_hd)
        )
    if result == "Failed":
        display_error_output(real_data, data_compe, err_idx, err_diff[0:max_error_idx])
    return result, fulfill_percent


def golden(query, key, value, sparse_indices, block_table=None,
           query_rope=None, key_rope=None, sinks=None, *, scale_value=1.0,
           cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None,
           sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
           sparse_mode=0, return_softmax_lse=False, quantize_exp=True,
           attention_mode=2, pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1):
    assert sparse_block_size == 1
    for tensor in (query, key, value, sparse_indices, query_rope, key_rope):
        if tensor is None:
            continue
        assert tensor.device.type == "cpu"

    def lengths(cu, used, layout, default):
        if layout == "TND":
            boundaries = [int(v) for v in cu]
            default = [end - start for start, end in zip(boundaries, boundaries[1:])]
        return default if used is None else [int(v) for v in used]

    qlens = lengths(cu_seqlens_q, seqused_q, layout_query,
                    [query.shape[1]] * query.shape[0] if layout_query == "BSND" else [])
    klens = lengths(cu_seqlens_kv, seqused_kv, layout_kv,
                    [key.shape[1]] * len(qlens))
    heads = query.shape[-2]
    out = torch.zeros_like(query, dtype=torch.float32)
    auxshape = ((len(qlens), 1, query.shape[1], heads) if layout_query == "BSND"
                else (1, query.shape[0], heads))
    rowmax, rowsum = torch.zeros(auxshape), torch.zeros(auxshape)
    for batch, (qlen, klen) in enumerate(zip(qlens, klens)):
        qstart = int(cu_seqlens_q[batch]) if layout_query == "TND" else 0
        kstart = int(cu_seqlens_kv[batch]) if layout_kv == "TND" else 0
        for row in range(qlen):
            qi = (batch, row) if layout_query == "BSND" else (qstart + row,)
            ai = (batch, 0, row) if layout_query == "BSND" else (0, qstart + row)
            threshold = klen if sparse_mode == 0 else klen - qlen + row + 1
            if sparse_mode == 3 and threshold <= 0:  # early causal exit precedes sinks
                continue
            ids = []
            for token in sparse_indices[qi][0, :min(sparse_indices.shape[-1], threshold)]:
                token = int(token)
                if token == -1:
                    break
                if 0 <= token < threshold:
                    ids.append(token)
            if not ids:
                if sinks is not None:
                    rowmax[ai], rowsum[ai] = sinks.float(), 1.0
                continue
            if layout_kv == "PA_BSND":
                physical = [int(block_table[batch, i // key.shape[1]]) for i in ids]
                offset = [i % key.shape[1] for i in ids]
                k = key[physical, offset, 0].float()
                if key_rope is not None:
                    kr = key_rope[physical, offset, 0].float()
            elif layout_kv == "TND":
                k = key[[kstart + i for i in ids], 0].float()
                if key_rope is not None:
                    kr = key_rope[[kstart + i for i in ids], 0].float()
            else:
                k = key[batch, ids, 0].float()
                if key_rope is not None:
                    kr = key_rope[batch, ids, 0].float()
            if query_rope is None:
                q = query[qi].float()
                joint = k
            else:
                # Concatenated QK matches upstream FP32 matmul reduction.
                q = torch.cat((query[qi].float(), query_rope[qi].float()), dim=-1)
                joint = torch.cat((k, kr), dim=-1)
            scores = (q @ joint.T) * scale_value
            maxima = scores.max(dim=-1, keepdim=True).values
            if sinks is not None:
                maxima = torch.maximum(maxima, sinks.float()[:, None])
            exp = torch.exp(scores - maxima)
            sums = exp.sum(dim=-1, keepdim=True)
            if sinks is not None:
                sums += torch.exp(sinks.float()[:, None] - maxima)
            p = exp.to(query.dtype).float() if quantize_exp else exp
            # MLA absorb: upstream derives V from K[..., :512].
            out[qi] = (p @ k) / sums
            rowmax[ai], rowsum[ai] = maxima[:, 0], sums[:, 0]
    if not return_softmax_lse:
        rowmax, rowsum = torch.empty(0), torch.empty(0)
    return out.to(query.dtype), rowmax, rowsum


def sequence_kwargs(qlens, klens, layout_query, layout_kv):
    """Construct new-interface metadata directly from per-sequence lengths."""
    result = {}
    for side, values, layout in (("q", qlens, layout_query), ("kv", klens, layout_kv)):
        tensor = None if values is None else torch.tensor(values, dtype=torch.int32)
        if layout == "TND":
            result[f"cu_seqlens_{side}"] = (None if tensor is None else
                torch.cat((torch.zeros(1, dtype=torch.int32), tensor.cumsum(0).int())))
        else:
            result[f"seqused_{side}"] = tensor
    return result


def make_case(dtype=torch.float16, heads=8, count=65, layout_query="BSND",
              layout_kv="BSND", mode=0, sinks=False, empty=False,
              qlens=(3, 2), klens=(81, 69), scale=None, aux=True,
              use_rope=True, page_size=16,
              seed=20260905):
    generator = torch.Generator().manual_seed(seed)
    batch, sq, sk = len(qlens), max(qlens), max(klens)

    def rand(shape):
        return (torch.randn(shape, generator=generator) * 0.3).to(dtype)

    q, qr = rand((batch, sq, heads, 512)), rand((batch, sq, heads, 64))
    k, kr = rand((batch, sk, 1, 512)), rand((batch, sk, 1, 64))
    indices = torch.full((batch, sq, 1, count), -1, dtype=torch.int32)
    for b, (ql, kl) in enumerate(zip(qlens, klens)):
        for row in range(ql):
            limit = kl if mode == 0 else kl - ql + row + 1
            n = min(max(limit, 0), count)
            if n and not empty:
                indices[b, row, 0, :n] = torch.randperm(limit, generator=generator)[:n].int()
    table = None
    if layout_query == "TND":
        q, qr, indices = [torch.cat([t[b, :ql] for b, ql in enumerate(qlens)])
                           for t in (q, qr, indices)]
    if layout_kv == "TND":
        k, kr = [torch.cat([t[b, :kl] for b, kl in enumerate(klens)]) for t in (k, kr)]
    if layout_kv == "PA_BSND":
        blocksize = page_size
        perbatch = (sk + blocksize - 1) // blocksize
        # Reverse page order so the test always exercises block-table remapping.
        table = torch.arange(batch * perbatch - 1, -1, -1, dtype=torch.int32).view(batch, perbatch)
        caches = []
        for source in (k, kr):
            cache = torch.zeros(batch * perbatch, blocksize, 1, source.shape[-1], dtype=dtype)
            for b in range(batch):
                for row in range(klens[b]):
                    cache[table[b, row // blocksize], row % blocksize] = source[b, row]
            caches.append(cache)
        k, kr = caches
        aux = False
    if not use_rope:
        qr = None
        kr = None
    if scale is None:
        scale = (512 + (64 if use_rope else 0)) ** -0.5
    return dict(query=q, key=k, value=rand(k.shape), sparse_indices=indices,
                query_rope=qr, key_rope=kr, block_table=table,
                **sequence_kwargs(qlens, klens, layout_query, layout_kv),
                sinks=torch.linspace(-1, 1, heads) if sinks else None,
                layout_query=layout_query, layout_kv=layout_kv,
                sparse_mode=mode, scale_value=scale, return_softmax_lse=aux)


def assert_outputs(expected, actual):
    assert isinstance(actual, tuple) and len(actual) == 3
    for ref, got in zip(expected, actual):
        got = got.cpu()
        assert got.shape == ref.shape and got.dtype == ref.dtype
        assert torch.isfinite(got).all()
        assert check_result(ref, got)[0] == "Pass"
        # An aggregate percentage must not hide incorrect empty rows.
        zeros = (ref == 0).all(dim=-1) if ref.ndim > 1 else None
        if zeros is not None and zeros.any():
            assert (got[zeros] == 0).all()


def _run_case(case, repeats=1, *, aclgraph=False, validator=None):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU runtime unavailable")
    from sparse_flash_attention import sparse_flash_attention
    expected = golden(**case) if validator is None else None
    device_case = {k: v.npu() if isinstance(v, torch.Tensor) else v for k, v in case.items()}

    if aclgraph:
        # Compile and warm up outside capture; keep the same input tensors alive.
        sparse_flash_attention(**device_case)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual = sparse_flash_attention(**device_case)
        torch.npu.synchronize()

    first = None
    for _ in range(max(repeats, 2) if aclgraph else repeats):
        if aclgraph:
            graph.replay()
        else:
            actual = sparse_flash_attention(**device_case)
        torch.npu.synchronize()
        if validator is None:
            assert_outputs(expected, actual)
        else:
            validator(case, actual)
        current = tuple(x.cpu() for x in actual)
        if first is not None:
            assert all(torch.equal(a, b) for a, b in zip(first, current))
        first = current


@pytest.fixture
def run_case(request):
    """Select eager or ACL Graph mode from pytest's built-in -o option."""
    mode = "eager"
    for option in request.config.getoption("override_ini") or ():
        name, separator, value = option.partition("=")
        if separator and name == "mode":
            mode = value
    if mode not in ("eager", "aclgraph"):
        pytest.fail("mode must be eager or aclgraph", pytrace=False)

    def invoke(case, repeats=1, *, validator=None):
        _run_case(case, repeats=repeats, aclgraph=(mode == "aclgraph"), validator=validator)

    return invoke

# Functional cases cover layouts, dtypes, optional metadata, head and sparse
# tails, empty rows, and repeated execution.


@pytest.mark.npu
def test_f01_bsnd_fp16_baseline(run_case):
    """基线：BSND/BSND，mode 0，无 sinks，fp16。"""
    run_case(make_case(heads=2, count=67))


@pytest.mark.npu
def test_f02_full_length_without_rope_or_lengths(run_case):
    """BF16 full-length BSND input without RoPE or length tensors."""
    case = make_case(dtype=torch.bfloat16, heads=2, count=67,
                     qlens=(3, 3), klens=(81, 81), use_rope=False)
    case["seqused_q"] = None
    case["seqused_kv"] = None
    run_case(case)


@pytest.mark.npu
def test_f03_tnd_causal_sinks(run_case):
    """TND 打包布局 + 因果(mode 3) + sink 状态。"""
    run_case(make_case(heads=8, count=67,
                       layout_query="TND", layout_kv="TND",
                       mode=3, sinks=True))


@pytest.mark.npu
def test_f04_paged_kv(run_case):
    """Remapped 128-token PA pages with sparse indices spanning two pages."""
    run_case(make_case(heads=8, count=129, layout_kv="PA_BSND",
                       klens=(145, 133), page_size=128))


@pytest.mark.npu
def test_f05_tnd_paged_bf16(run_case):
    """TND query + PA KV 组合，bf16。"""
    run_case(make_case(dtype=torch.bfloat16, heads=8, count=67,
                       layout_query="TND", layout_kv="PA_BSND",
                       mode=3, sinks=True))


@pytest.mark.npu
def test_f06_head_group_split_over_64(run_case):
    """G=128 > TILE_M：group_chunks=2，一个 token 的 head 拆给两个 AIC。"""
    run_case(make_case(heads=128, count=67))


@pytest.mark.npu
def test_f07_nonpow2_heads_multitile(run_case):
    """G=33 非 2 幂 + K=129 跨 128 tile 边界（延迟线跑满多任务）。"""
    run_case(make_case(dtype=torch.bfloat16, heads=33, count=129,
                       qlens=(4, 3), klens=(263, 259)))


@pytest.mark.npu
def test_f08_min_sparse_count(run_case):
    """K=1 最小候选数。"""
    run_case(make_case(heads=64, count=1, mode=3))


@pytest.mark.npu
def test_f09_empty_rows(run_case):
    """空行（索引全 -1）：输出必须为 0，不能被聚合百分比掩盖。"""
    run_case(make_case(mode=3, sinks=True, empty=True,
                       qlens=(5, 3), klens=(2, 0)))


@pytest.mark.npu
def test_f10_multitile_determinism(run_case):
    """Repeated nonzero-scale multi-tile execution is bitwise stable."""
    run_case(make_case(heads=16, count=129, qlens=(4, 3),
                       klens=(263, 259)), repeats=2)


# Performance configurations shown in README: accuracy checks only, no timing,
# profiler, mainline package or SFA_RUN_PERF switch is required.
# Within each group the order matches the chart's DSL mean kernel duration.
PERFORMANCE_PRECISION_CASES = [
    pytest.param(1, 1024, 8192, 128, 64, "BSND", id="prefill_topk_128"),
    pytest.param(1, 1024, 8192, 512, 64, "BSND", id="prefill_topk_512"),
    pytest.param(1, 1024, 8192, 2048, 64, "BSND", id="prefill_base"),
    pytest.param(1, 4096, 8192, 2048, 64, "BSND", id="prefill_s1_4096"),
    pytest.param(1, 8192, 2048, 2048, 64, "BSND", id="prefill_s1_8192_s2_2048"),
    pytest.param(48, 1, 8192, 128, 64, "BSND", id="decode_topk_128"),
    pytest.param(8, 1, 8192, 2048, 64, "BSND", id="decode_batch_8"),
    pytest.param(48, 1, 8192, 2048, 64, "BSND", id="decode_base"),
    pytest.param(128, 1, 8192, 2048, 64, "BSND", id="decode_batch_128"),
    pytest.param(48, 4, 8192, 2048, 64, "BSND", id="decode_s1_4"),
    pytest.param(1, 1, 20480, 2048, 128, "PA_BSND", id="network_case"),
    pytest.param(1, 4, 20480, 2048, 128, "PA_BSND", id="network_case_s1_4"),
    pytest.param(1, 8, 20480, 2048, 128, "PA_BSND", id="network_case_s1_8"),
    pytest.param(1, 1024, 20480, 2048, 128, "PA_BSND", id="pa_prefill_s1_1024"),
    pytest.param(1, 4096, 20480, 2048, 128, "PA_BSND", id="pa_prefill_s1_4096"),
]


def make_performance_case(batch, s1, s2, topk, heads, layout_kv):
    """Reproduce performance shapes, BF16/RoPE, mode 3 and causal sparse indices."""
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU runtime unavailable")
    torch.manual_seed(20260905)

    def rand(shape):
        return torch.empty(shape, dtype=torch.bfloat16, device="npu").normal_(0, 0.3)

    query = rand((batch, s1, heads, 512))
    query_rope = rand((batch, s1, heads, 64))
    page_size = 128
    paged = layout_kv == "PA_BSND"
    kv_shape = (batch * (s2 // page_size), page_size, 1) if paged else (batch, s2, 1)
    key = rand((*kv_shape, 512))
    key_rope = rand((*kv_shape, 64))
    indices = torch.full((batch, s1, 1, topk), -1, dtype=torch.int32)
    for row in range(s1):
        visible = max(0, s2 - s1 + row + 1)
        count = min(topk, visible)
        if count:
            ids = (torch.arange(count, dtype=torch.int64) * (visible - 1)
                   // max(count - 1, 1)).int()
            if count == 1:
                ids[0] = visible - 1
            indices[:, row, 0, :count] = ids
    block_table = None
    if paged:
        block_table = torch.arange(kv_shape[0] - 1, -1, -1, dtype=torch.int32).view(batch, -1).npu()
    return dict(query=query, key=key, value=key, query_rope=query_rope,
                key_rope=key_rope, sparse_indices=indices.npu(), block_table=block_table,
                **sequence_kwargs((s1,) * batch, (s2,) * batch, "BSND", layout_kv),
                scale_value=576 ** -0.5, sparse_block_size=1, sparse_mode=3,
                attention_mode=2, layout_query="BSND", layout_kv=layout_kv,
                sinks=None, return_softmax_lse=False)


def assert_performance_precision(inputs, actual):
    """Check all short-query rows; sample large prefill rows and all their heads."""
    assert isinstance(actual, tuple) and len(actual) == 3
    out, rowmax, rowsum = actual
    query = inputs["query"]
    assert out.shape == query.shape and out.dtype == query.dtype
    assert out.device == query.device and torch.isfinite(out).all()
    for stats in (rowmax, rowsum):
        assert stats.numel() == 0 and stats.dtype == torch.float32
    batch_count, s1 = query.shape[:2]
    s2 = int(inputs["seqused_kv"][0])
    rows = (list(range(s1)) if s1 <= 8 else
            sorted({0, 1, 127, 128, s1 // 2, s1 - 1,
                    max(0, s1 - s2 - 1), max(0, s1 - s2)}))
    for batch in range(batch_count):
        for row in rows:
            ids = inputs["sparse_indices"][batch, row, 0].cpu().long()
            ids = ids[ids >= 0]
            got = out[batch:batch + 1, row:row + 1].cpu()
            if not ids.numel():
                assert (got == 0).all(), (batch, row)
                continue
            if inputs["layout_kv"] == "PA_BSND":
                page_size = inputs["key"].shape[1]
                pages = inputs["block_table"][batch].cpu().long()
                physical = pages[ids // page_size] * page_size + ids % page_size
                key = inputs["key"].reshape(-1, 1, 512)[physical.npu()].unsqueeze(0).cpu()
                rope = inputs["key_rope"].reshape(-1, 1, 64)[physical.npu()].unsqueeze(0).cpu()
            else:
                key = inputs["key"][batch, ids.npu()].unsqueeze(0).cpu()
                rope = inputs["key_rope"][batch, ids.npu()].unsqueeze(0).cpu()
            # The original mode-3 indices have already been masked. The CPU
            # reference computes attention on the selected tokens independently.
            expected = golden(
                query=query[batch:batch + 1, row:row + 1].cpu(),
                query_rope=inputs["query_rope"][batch:batch + 1, row:row + 1].cpu(),
                key=key, value=key, key_rope=rope,
                sparse_indices=torch.arange(ids.numel(), dtype=torch.int32).view(1, 1, 1, -1),
                scale_value=inputs["scale_value"], sparse_mode=0, return_softmax_lse=False)
            assert_outputs(expected, (got, rowmax.cpu(), rowsum.cpu()))
