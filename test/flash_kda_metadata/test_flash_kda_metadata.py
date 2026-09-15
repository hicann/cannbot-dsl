# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""NPU precision and schedule coverage for the standalone FlashKDA metadata API."""

import pytest
import torch
from _samples_path import load_sample

metadata_module = load_sample("flash_kda_metadata/flash_kda_metadata.py")
STATUS_OK = metadata_module.STATUS_OK
WORKSPACE_SLOTS = metadata_module.WORKSPACE_SLOTS


def _cu(lengths):
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    return prefix


def _assert_partition(ranges, total):
    cursor = 0
    unused = False
    for start, end in ranges:
        if (start, end) == (0, 0):
            unused = True
            continue
        assert not unused
        assert start == cursor
        assert start < end <= total
        cursor = end
    assert cursor == total


def _assert_complete_chunk_schedule(decoded, lengths, nv, packed):
    """Check logical coverage independently of producer scheduling decisions."""
    assert decoded["status"] == STATUS_OK
    assert decoded["batch"] == len(lengths)
    assert decoded["value_heads"] == nv
    expected_chunks = [(length + 63) // 64 for length in lengths]
    chunk_coverage = [[] for _ in lengths]
    token_starts = _cu(lengths)[:-1] if packed else [0] * len(lengths)
    for round_index, record in enumerate(decoded["stage12_rounds"]):
        assert record["stage12_round_idx"] == round_index
        active_batches = [
            batch for batch, chunks in enumerate(expected_chunks)
            if len(chunk_coverage[batch]) < chunks
        ]
        assert record["batch_idx"] == active_batches
        assert record["active_batch_num"] == len(active_batches)
        stage1_prefix = [0]
        stage2_prefix = [0]
        for entry, batch in enumerate(active_batches):
            start = record["chunk_start_per_group"][entry]
            count = record["chunk_num_per_group"][entry]
            assert start == len(chunk_coverage[batch])
            assert 0 < count <= record["group_num"]
            assert start + count <= expected_chunks[batch]
            assert record["valid_seq_len"][entry] == lengths[batch]
            assert record["token_start"][entry] == token_starts[batch]
            assert record["storage_batch_idx"][entry] == (0 if packed else batch)
            for chunk in range(start, start + count):
                assert 0 <= chunk * 64 < lengths[batch]
                assert 1 <= min(64, lengths[batch] - chunk * 64) <= 64
                chunk_coverage[batch].append(chunk)
            splits = record["dv_splits_num"][entry]
            assert splits in (1, 2, 4, 8)
            stage1_prefix.append(stage1_prefix[-1] + nv * count)
            stage2_prefix.append(stage2_prefix[-1] + nv * splits)
        assert record["stage1_task_prefix"] == stage1_prefix
        assert record["stage2_task_prefix"] == stage2_prefix
        assert stage1_prefix[-1] <= WORKSPACE_SLOTS
        for stage, prefix in ((1, stage1_prefix), (2, stage2_prefix)):
            ranges = record[f"stage{stage}_core_ranges"]
            _assert_partition(ranges, prefix[-1])
            assert all(pair == (0, 0) for pair in ranges[decoded["core_num"]:])
    assert chunk_coverage == [list(range(chunks)) for chunks in expected_chunks]



_BOUNDARY_LENGTHS = (15, 16, 17, 63, 64, 65, 127, 128, 129)
_CASES = [
    pytest.param("BNSD", False, (129, 129), 2, [12], [2], id="bnsd-dense"),
    pytest.param("BSND", False, (129, 129), 2, [12], [2], id="bsnd-dense"),
    pytest.param("TND", True, _BOUNDARY_LENGTHS, 3, [42], [9], id="packed-tails"),
    pytest.param("BNSD", True, _BOUNDARY_LENGTHS, 5, [70], [9], id="padded-tails"),
    pytest.param("TND", True, (1, 63, 284 * 64), 3, [858], [3], id="below-861"),
    pytest.param("TND", True, (1, 63, 285 * 64), 3, [861], [3], id="exact-861"),
    pytest.param("TND", True, (1, 63, 285 * 64 + 1), 3, [861, 3], [3, 1], id="above-861"),
    pytest.param("BSND", True, (*_BOUNDARY_LENGTHS, 65 * 64 + 1, 107 * 64 + 17),
                 5, [860, 80], [11, 1], id="ragged-multiround-remainder"),
]


@pytest.mark.npu
@pytest.mark.parametrize("layout,has_cu,lengths,nv,round_slots,active_counts", _CASES)
def test_flash_kda_metadata(layout, has_cu, lengths, nv, round_slots, active_counts):
    pytest.importorskip("torch_npu")
    batch, nk, dim = len(lengths), 1, 128
    packed = layout == "TND"
    storage = sum(lengths) if packed else max(lengths)
    if packed:
        q_shape, v_shape = (storage, nk, dim), (storage, nv, dim)
    elif layout == "BNSD":
        q_shape, v_shape = (batch, nk, storage, dim), (batch, nv, storage, dim)
    else:
        q_shape, v_shape = (batch, storage, nk, dim), (batch, storage, nv, dim)
    q = torch.empty(q_shape, dtype=torch.bfloat16, device="npu")
    v = torch.empty(v_shape, dtype=torch.bfloat16, device="npu")
    state = torch.empty((batch, nv, dim, dim), dtype=torch.float32, device="npu")
    cu = torch.tensor(_cu(lengths), dtype=torch.int32, device="npu") if has_cu else None
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        actual = metadata_module.flash_kda_metadata(q, v, state, layout, cu)
    stream.synchronize()

    assert actual.dtype == torch.int32 and actual.is_contiguous()
    assert actual.device == q.device
    decoded = metadata_module.decode_metadata(actual.cpu().tolist())
    assert 0 < decoded["used_words"] <= actual.numel()
    _assert_complete_chunk_schedule(decoded, lengths, nv, packed)
    records = decoded["stage12_rounds"]
    assert decoded["stage12_round_num"] == len(round_slots)
    assert [record["stage1_task_prefix"][-1] for record in records] == round_slots
    assert [record["active_batch_num"] for record in records] == active_counts
    if len(round_slots) > 1:
        assert WORKSPACE_SLOTS - round_slots[0] == WORKSPACE_SLOTS % nv
        assert active_counts[-1] < active_counts[0]


def teardown_module():
    metadata_module.clear_metadata_caches()
