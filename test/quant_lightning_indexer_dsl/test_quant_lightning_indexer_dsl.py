# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent reference and precision tests for quant_lightning_indexer_dsl."""

import os
from itertools import accumulate

import pytest
import torch

from _samples_path import load_sample

HEADS = 32
BLOCK_SIZE = 8
CANDIDATE_CAPACITY = 2048
TOPK = 512
SCORE_RTOL = 1 / 128
SCORE_ATOL = 2.5e-5


def _prefix(lengths):
    return [0, *accumulate(lengths)]


def _decode_mxfp4(packed, scales):
    """Low nibble precedes high nibble; one E8M0 scale covers 32 values."""
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    values = magnitudes[codes & 7] * torch.where(codes < 8, 1.0, -1.0)
    exponents = scales.reshape(*packed.shape[:-1], -1).int() - 127
    return values * torch.exp2(exponents.float()).repeat_interleave(32, dim=-1)


def _scores_reference(query, keys, weights, query_scale, key_scale):
    query_float = _decode_mxfp4(query, query_scale)
    key_float = _decode_mxfp4(keys, key_scale).squeeze(1)
    # Cube QK, Vector1 weights and each fused multiply-add result are BF16.
    qk = (query_float @ key_float.T).to(torch.bfloat16).float()
    bf16_weights = weights.to(torch.bfloat16).float()
    scores = torch.zeros((query.shape[0], keys.shape[0]), dtype=torch.float32)
    for head in range(HEADS):
        scores = (
            (scores + qk[:, head].clamp_min(0) * bf16_weights[:, head, None])
            .to(torch.bfloat16)
            .float()
        )
    return scores


def _visible_tokens(key_length, used_query, row, mask, ratio, residual):
    if row >= used_query:
        return 0
    if mask == 0:
        return key_length
    return min(
        key_length,
        max(0, (key_length * ratio + residual - used_query + row + 1) // ratio),
    )


def _check_topk(indices, values, scores, allowed, offset=0):
    """Check values, completeness and the TopK boundary without ordering ties."""
    valid = indices != -1
    if values is not None:
        assert torch.isneginf(values[~valid]).all(), "Invalid TopK values must be -inf"
    selected = indices[valid].long() - offset
    assert selected.numel() == min(indices.numel(), int(allowed.sum()))
    assert selected.unique().numel() == selected.numel()
    assert ((selected >= 0) & (selected < scores.numel())).all()
    assert allowed[selected].all()
    if selected.numel() == 0:
        return
    selected_scores = scores[selected]
    if values is not None:
        torch.testing.assert_close(
            values[valid].float(), selected_scores, rtol=SCORE_RTOL, atol=SCORE_ATOL
        )
    boundary = scores[allowed].topk(selected.numel()).values[-1]
    if torch.isposinf(boundary):
        assert torch.isposinf(selected_scores).all()
    else:
        assert (
            selected_scores.min() >= boundary - SCORE_ATOL - SCORE_RTOL * boundary.abs()
        )


def _npu_device():
    pytest.importorskip("torch_npu")
    device = torch.device("npu", int(os.environ.get("NPU_DEVICE_ID", "0")))
    torch.npu.set_device(device)
    return device


def _load_operators():
    kernel = load_sample("quant_lightning_indexer_dsl/quant_lightning_indexer_dsl.py")
    metadata = load_sample(
        "quant_lightning_indexer_metadata_dsl/quant_lightning_indexer_metadata_dsl.py"
    )
    return kernel.quant_lightning_indexer, metadata.quant_lightning_indexer_metadata


def _make_inputs(
    layout,
    page_size,
    mask,
    *,
    sparse,
    candidates=False,
    candidate_capacity=CANDIDATE_CAPACITY,
    key_lengths=(513, 777),
    topk=TOPK,
    query_lengths=None,
    signed_weights=False,
    uniform_scale=False,
):
    generator = torch.Generator().manual_seed(20260929)
    typical = query_lengths is not None
    query_lengths = list(query_lengths) if typical else [3, 4]
    used_queries = query_lengths if typical else [3, 3]
    assert len(query_lengths) == len(key_lengths)
    total_queries = sum(query_lengths)
    ratio = 2 if mask == 3 else 1
    residual = [0] * len(query_lengths) if typical else [1, 0]

    def random_bytes(shape, low=0, high=256):
        return torch.randint(low, high, shape, generator=generator, dtype=torch.uint8)

    q = random_bytes((total_queries, HEADS, 64))
    qs = (
        torch.full((total_queries, HEADS, 2, 2), 127, dtype=torch.uint8)
        if uniform_scale
        else random_bytes((total_queries, HEADS, 2, 2), 124, 128)
    )
    weights = torch.rand((total_queries, HEADS), generator=generator)
    weights = weights * 2 - 1 if signed_weights else weights + 0.1
    logical_keys = [random_bytes((length, 1, 64)) for length in key_lengths]
    logical_scales = [
        torch.full((length, 1, 2, 2), 127, dtype=torch.uint8)
        if uniform_scale
        else random_bytes((length, 1, 2, 2), 124, 128)
        for length in key_lengths
    ]
    table = None
    if layout == "TND":
        k, ks = torch.cat(logical_keys), torch.cat(logical_scales)
    else:
        max_pages = (max(key_lengths) + page_size - 1) // page_size
        pages = len(key_lengths) * max_pages
        table = (
            torch.randperm(pages, generator=generator)
            .int()
            .reshape(len(key_lengths), max_pages)
        )
        k = torch.zeros((pages, page_size, 1, 64), dtype=torch.uint8)
        ks = torch.full((pages, page_size, 1, 2, 2), 127, dtype=torch.uint8)
        for batch, length in enumerate(key_lengths):
            tokens = torch.arange(length)
            physical_rows = (
                table[batch, tokens // page_size].long() * page_size
                + tokens % page_size
            )
            k.view(-1, 1, 64).index_copy_(0, physical_rows, logical_keys[batch])
            ks.view(-1, 1, 2, 2).index_copy_(0, physical_rows, logical_scales[batch])
        if sparse:
            k = torch.cat(
                (
                    k.reshape(pages, page_size // 8, 512),
                    ks.reshape(pages, page_size // 8, 32),
                ),
                -1,
            )
            ks = None
    candidate_ids = torch.full(
        (total_queries, 1, CANDIDATE_CAPACITY), -1, dtype=torch.int32
    )
    candidate_lengths = torch.zeros((total_queries, 1), dtype=torch.int32)
    references, permitted = [], []
    for batch, (start, length) in enumerate(zip(_prefix(query_lengths), query_lengths)):
        reference = _scores_reference(
            q[start : start + length],
            logical_keys[batch],
            weights[start : start + length],
            qs[start : start + length],
            logical_scales[batch],
        )
        for row in range(length):
            visible = _visible_tokens(
                key_lengths[batch],
                used_queries[batch],
                row,
                mask,
                ratio,
                residual[batch],
            )
            allowed = torch.arange(key_lengths[batch]) < visible
            count = (key_lengths[batch] + 7) // 8
            chosen = torch.randperm(count, generator=generator)[
                : min(CANDIDATE_CAPACITY, max(1, count - 5))
            ].int()
            if sparse and not typical and start + row == 1:
                chosen = chosen[:0]
            candidate_ids[start + row, 0, : chosen.numel()] = chosen
            candidate_lengths[start + row] = chosen.numel()
            if sparse:
                allowed &= torch.isin(torch.arange(key_lengths[batch]) // 8, chosen)
            references.append(reference[row])
            permitted.append(allowed)
    device = _npu_device()

    def to_device(tensor):
        return tensor.to(device) if tensor is not None else None

    sequence = dict(
        cu_seqlens_q=torch.tensor(
            _prefix(query_lengths), dtype=torch.int32, device=device
        ),
        seqused_q=torch.tensor(used_queries, dtype=torch.int32, device=device),
        seqused_k=torch.tensor(key_lengths, dtype=torch.int32, device=device),
    )
    if layout == "TND":
        sequence["cu_seqlens_k"] = torch.tensor(
            _prefix(key_lengths), dtype=torch.int32, device=device
        )
    if mask == 3:
        sequence["cmp_residual_k"] = torch.tensor(
            residual, dtype=torch.int32, device=device
        )
    attributes = dict(
        topk=topk,
        max_seqlen_q=max(query_lengths),
        layout_q="TND",
        layout_k=layout,
        mask_mode=mask,
        cmp_ratio=ratio,
    )
    metadata_attributes = dict(
        attributes,
        max_seqlen_k=max(key_lengths),
        num_heads_q=HEADS,
        num_heads_k=1,
        head_dim=128,
    )
    if not sparse:
        attributes.update(
            candidate_topk_blocks=candidate_capacity if candidates else -1,
            candidate_block_size=8 if candidates else -1,
        )
        metadata_attributes.update(
            candidate_topk_blocks=attributes["candidate_topk_blocks"],
            candidate_block_size=attributes["candidate_block_size"],
        )
    else:
        attributes["candidate_block_size"] = 8
        metadata_attributes.update(candidate_block_size=8, quant_mode=1)
    return dict(
        q=to_device(q),
        k=to_device(k),
        w=to_device(weights),
        descale_q=to_device(qs),
        descale_k=to_device(ks),
        block_table=to_device(table),
        sequence=sequence,
        attributes=attributes,
        metadata_attributes=metadata_attributes,
        candidate_ids=to_device(candidate_ids),
        candidate_lengths=to_device(candidate_lengths),
        reference=references,
        allowed=permitted,
        offsets=to_device(
            torch.arange(total_queries, dtype=torch.int32).reshape(-1, 1) * 10000
        ),
    )


def _check_kernel(
    layout,
    page_size,
    mask,
    *,
    sparse,
    candidates=False,
    candidate_capacity=CANDIDATE_CAPACITY,
    key_lengths=(513, 777),
    topk=TOPK,
    empty=None,
    query_lengths=None,
    signed_weights=False,
    uniform_scale=False,
):
    data = _make_inputs(
        layout,
        page_size,
        mask,
        sparse=sparse,
        candidates=candidates,
        candidate_capacity=candidate_capacity,
        key_lengths=key_lengths,
        topk=topk,
        query_lengths=query_lengths,
        signed_weights=signed_weights,
        uniform_scale=uniform_scale,
    )
    if empty is not None:
        if empty == "query":
            data["sequence"]["seqused_q"].zero_()
        elif empty == "key":
            data["sequence"]["seqused_k"].zero_()
        elif empty == "candidate" and sparse:
            data["candidate_lengths"].zero_()
        else:
            raise ValueError(f"Unsupported empty input: {empty}")
        for allowed in data["allowed"]:
            allowed.zero_()
    kernel, make_metadata = _load_operators()
    metadata_args = (data["candidate_lengths"],) if sparse else ()
    metadata = make_metadata(
        *metadata_args, **data["sequence"], **data["metadata_attributes"]
    )
    args = [data[key] for key in ("q", "k", "w", "descale_q")]
    if sparse:
        args.extend((data["candidate_ids"], data["candidate_lengths"]))
    else:
        args.append(data["descale_k"])
    kwargs = dict(
        data["sequence"],
        **data["attributes"],
        quant_mode=1,
        block_table=data["block_table"],
        output_idx_offset=data["offsets"],
        metadata=metadata,
    )
    if sparse:
        kwargs["descale_k"] = data["descale_k"]
    for return_value in (True, False):
        outputs = kernel(*args, **kwargs, return_value=return_value)
        torch.npu.synchronize()
        indices, values = (tensor.cpu() for tensor in outputs[:2])
        assert indices.dtype == torch.int32
        assert indices.shape == (len(data["reference"]), 1, topk)
        assert values.dtype == torch.bfloat16
        assert values.shape == indices.shape if return_value else values.numel() == 0
        for row, (scores, allowed) in enumerate(
            zip(data["reference"], data["allowed"])
        ):
            _check_topk(
                indices[row, 0],
                values[row, 0] if return_value else None,
                scores,
                allowed,
                row * 10000,
            )
        if not sparse:
            candidate_ids, candidate_lengths = (tensor.cpu() for tensor in outputs[2:])
            if not candidates:
                assert candidate_ids.numel() == candidate_lengths.numel() == 0
                continue
            assert candidate_ids.shape == (indices.shape[0], 1, candidate_capacity)
            for row, (scores, allowed) in enumerate(
                zip(data["reference"], data["allowed"])
            ):
                padded = torch.full((((scores.numel() + 7) // 8) * 8,), -float("inf"))
                padded[: scores.numel()] = scores.masked_fill(~allowed, -float("inf"))
                block_scores = padded.reshape(-1, 8).amax(-1)
                valid_blocks = torch.isfinite(block_scores)
                visible = int(allowed.sum())
                # The public candidate contract always retains the partial block.
                if visible % BLOCK_SIZE:
                    block_scores[visible // BLOCK_SIZE] = float("inf")
                count = min(candidate_capacity, int(valid_blocks.sum()))
                assert candidate_lengths[row, 0] == count
                _check_topk(
                    candidate_ids[row, 0, :count], None, block_scores, valid_blocks
                )


def test_mxfp4_decode():
    packed = torch.tensor([[0x21, 0xF8] * 32], dtype=torch.uint8)
    scales = torch.tensor([[[127, 128], [126, 127]]], dtype=torch.uint8)
    expected = torch.tensor([0.5, 1.0, 0.0, -6.0] * 32)
    expected *= torch.tensor([1.0, 2.0, 0.5, 1.0]).repeat_interleave(32)
    torch.testing.assert_close(_decode_mxfp4(packed, scales)[0], expected)


def test_topk_reference_rejects_non_topk():
    with pytest.raises(AssertionError):
        _check_topk(
            torch.tensor([0, 1]),
            None,
            torch.tensor([1.0, 2.0, 9.0]),
            torch.ones(3, dtype=torch.bool),
        )


def test_topk_reference_rejects_zero_padding():
    with pytest.raises(AssertionError, match="Invalid TopK values"):
        _check_topk(
            torch.tensor([-1]),
            torch.tensor([0.0]),
            torch.tensor([1.0]),
            torch.tensor([False]),
        )


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,page_size,mask,candidates",
    [
        pytest.param("PA_BBND", 128, 0, False, id="pa128_ragged"),
        pytest.param("PA_BBND", 64, 3, True, id="pa64_causal_candidates"),
        pytest.param("TND", 128, 3, False, id="tnd_causal"),
    ],
)
def test_quant_lightning_indexer(layout, page_size, mask, candidates):
    _check_kernel(layout, page_size, mask, sparse=False, candidates=candidates)


@pytest.mark.npu
def test_candidate16_ld_tail():
    _check_kernel(
        "PA_BBND", 64, 3, sparse=False, candidates=True, candidate_capacity=16
    )


@pytest.mark.npu
@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
def test_invalid_values_are_negative_infinity(layout):
    _check_kernel(layout, 128, 0, sparse=False, key_lengths=(19, 73))


@pytest.mark.npu
@pytest.mark.parametrize("layout", ["PA_BBND", "TND"])
@pytest.mark.parametrize("empty", ["query", "key"])
def test_all_rows_empty(layout, empty):
    _check_kernel(layout, 128, 0, sparse=False, key_lengths=(19, 73), empty=empty)


@pytest.mark.npu
@pytest.mark.slow
@pytest.mark.parametrize(
    "batch_size,query_length,key_length",
    [
        pytest.param(12, 6, 65536, id="b12_s1_6_n32_s2_64k"),
        pytest.param(12, 6, 131072, id="b12_s1_6_n32_s2_128k"),
    ],
)
def test_typical_case(batch_size, query_length, key_length):
    _check_kernel(
        "PA_BBND",
        128,
        3,
        sparse=False,
        query_lengths=(query_length,) * batch_size,
        key_lengths=(key_length,) * batch_size,
        signed_weights=True,
        uniform_scale=True,
    )
