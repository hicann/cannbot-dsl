# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Host-side argument validation for sparse flash attention."""

from __future__ import annotations

import torch


INT64_MAX = 9_223_372_036_854_775_807

SUPPORTED_ATTENTION_MODE = 2
SUPPORTED_SPARSE_BLOCK_SIZE = 1
SPARSE_MODE_DENSE = 0
SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3
SUPPORTED_SPARSE_MODES = (SPARSE_MODE_DENSE, SPARSE_MODE_RIGHT_DOWN_CAUSAL)

MAX_HEAD_COUNT = 128
DIM_NOPE = 512
DIM_ROPE = 64
KV_HEAD_COUNT = 1
PA_BLOCK_ALIGNMENT = 16
PA_MAX_BLOCK_SIZE = 1024


def _resolve_rope_mode(query_rope, key_rope):
    """Validate the RoPE pairing and return whether RoPE is enabled."""
    if (query_rope is None) != (key_rope is None):
        raise ValueError("Query_rope and key_rope must be both present or both None")
    return query_rope is not None


def _validate_operator_params(
    attention_mode,
    sparse_block_size,
    sparse_mode,
    pre_tokens,
    next_tokens,
    layout_query,
    layout_kv,
    block_table,
    return_softmax_lse,
):
    """Validate operator attributes and layout-dependent options."""
    if attention_mode != SUPPORTED_ATTENTION_MODE:
        raise ValueError(f"Attention_mode must be {SUPPORTED_ATTENTION_MODE}")
    if sparse_block_size != SUPPORTED_SPARSE_BLOCK_SIZE:
        raise ValueError(f"Sparse_block_size must be {SUPPORTED_SPARSE_BLOCK_SIZE}")
    if sparse_mode not in SUPPORTED_SPARSE_MODES:
        raise ValueError(
            f"Sparse_mode must be {SPARSE_MODE_DENSE} or "
            f"{SPARSE_MODE_RIGHT_DOWN_CAUSAL}"
        )
    if pre_tokens != INT64_MAX or next_tokens != INT64_MAX:
        raise ValueError("Pre_tokens and next_tokens must keep their default values")

    supported_layouts = {
        ("BSND", "BSND"),
        ("BSND", "PA_BSND"),
        ("TND", "TND"),
        ("TND", "PA_BSND"),
    }
    if (layout_query, layout_kv) not in supported_layouts:
        raise ValueError("Unsupported query/KV layout combination")
    if layout_kv != "PA_BSND" and block_table is not None:
        raise ValueError("Block_table must be null for non-PA layout")
    if return_softmax_lse and layout_kv == "PA_BSND":
        raise ValueError("PA_BSND does not support return_softmax_lse=true")


def _validate_data_tensor_specs(
    query,
    key,
    value,
    sparse_indices,
    query_rope,
    key_rope,
    has_rope,
    sinks,
    block_table,
    layout_query,
    layout_kv,
):
    """Validate data tensor types, ranks, dtypes, storage, and dimensions."""
    if not isinstance(value, torch.Tensor):
        raise ValueError("Value must be a tensor")
    required_tensors = {
        "query": query,
        "key": key,
        "value": value,
        "sparse_indices": sparse_indices,
    }
    if has_rope:
        required_tensors["query_rope"] = query_rope
        required_tensors["key_rope"] = key_rope
    if any(not isinstance(tensor, torch.Tensor) for tensor in required_tensors.values()):
        raise ValueError("Q/K/V/RoPE/sparse_indices must be tensors")

    query_rank = 4 if layout_query == "BSND" else 3
    key_rank = 3 if layout_kv == "TND" else 4
    if query.ndim != query_rank or (has_rope and query_rope.ndim != query_rank):
        raise ValueError("Query and query_rope ranks must match layout_query")
    kv_tensors = (key, value)
    if has_rope:
        kv_tensors += (key_rope,)
    if any(tensor.ndim != key_rank for tensor in kv_tensors):
        raise ValueError("Key, value, and key_rope ranks must match layout_kv")
    if sparse_indices.ndim != query_rank:
        raise ValueError("Sparse_indices rank must match layout_query")
    if query.numel() == 0:
        raise ValueError("Query must be nonempty")

    for tensor_name, tensor in required_tensors.items():
        # PA blocks may have gaps between them, but their inner axes stay contiguous.
        if tensor_name in ("key", "value", "key_rope") and layout_kv == "PA_BSND":
            inner_span = 1
            for axis in range(tensor.ndim - 1, 0, -1):
                if tensor.shape[axis] > 1 and tensor.stride(axis) != inner_span:
                    raise ValueError(
                        f"{tensor_name.capitalize()} PA inner axes must be contiguous"
                    )
                inner_span *= tensor.shape[axis]
            if tensor.shape[0] > 1 and tensor.stride(0) < inner_span:
                raise ValueError(
                    f"{tensor_name.capitalize()} PA physical blocks must not overlap"
                )
        elif not tensor.is_contiguous():
            raise ValueError(f"{tensor_name.capitalize()} must be contiguous")

    for tensor_name, tensor in (("block_table", block_table), ("sinks", sinks)):
        if tensor is not None:
            if not isinstance(tensor, torch.Tensor) or not tensor.is_contiguous():
                raise ValueError(
                    f"{tensor_name.capitalize()} must be a contiguous tensor"
                )

    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Query must use FP16 or BF16")
    same_dtype_tensors = (key, value)
    if has_rope:
        same_dtype_tensors += (query_rope, key_rope)
    if any(tensor.dtype != query.dtype for tensor in same_dtype_tensors):
        raise ValueError("Q/K/V/RoPE tensors must use one common dtype")
    if sparse_indices.dtype != torch.int32:
        raise ValueError("Sparse_indices must be INT32")
    if sparse_indices.shape[-1] <= 0:
        raise ValueError("Sparse_indices K must be positive")

    if query.shape[-1] != DIM_NOPE or key.shape[-1] != DIM_NOPE:
        raise ValueError(f"Query and key head dimensions must be {DIM_NOPE}")
    if has_rope and (query_rope.shape[-1] != DIM_ROPE
                     or key_rope.shape[-1] != DIM_ROPE):
        raise ValueError(f"RoPE head dimensions must be {DIM_ROPE}")
    if key.shape[-2] != KV_HEAD_COUNT:
        raise ValueError(f"Key requires N2={KV_HEAD_COUNT}")


def _validate_sequence_length_tensors(
    cumulative_query_lengths,
    cumulative_kv_lengths,
    used_query_lengths,
    used_kv_lengths,
):
    """Validate sequence length tensor type, rank, dtype, and contiguity."""
    for tensor_name, tensor in (
        ("cu_seqlens_q", cumulative_query_lengths),
        ("cu_seqlens_kv", cumulative_kv_lengths),
        ("seqused_q", used_query_lengths),
        ("seqused_kv", used_kv_lengths),
    ):
        if tensor is None:
            continue
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 1
                or not tensor.is_contiguous()):
            raise ValueError(
                f"{tensor_name.capitalize()} must be a contiguous rank-1 tensor"
            )
        if tensor.dtype != torch.int32:
            raise ValueError(f"{tensor_name.capitalize()} must be INT32")


def _resolve_batch_head(query, cumulative_query_lengths, layout_query):
    """Derive the logical batch and head counts from the query layout."""
    if layout_query == "BSND":
        batch_size, _, head_count, _ = query.shape
        return batch_size, head_count
    if (not isinstance(cumulative_query_lengths, torch.Tensor)
            or cumulative_query_lengths.numel() < 2):
        raise ValueError("Cu_seqlens_q is required for TND and must contain B+1 values")
    _, head_count, _ = query.shape
    return cumulative_query_lengths.numel() - 1, head_count


def _validate_query_shapes(
    query,
    sparse_indices,
    query_rope,
    has_rope,
    sinks,
    head_count,
    layout_query,
):
    """Validate Query-side shape relations and head-dependent tensors."""
    if has_rope and query_rope.shape[:-1] != query.shape[:-1]:
        raise ValueError("Query_rope prefix dimensions must match query")
    if layout_query == "BSND":
        if sparse_indices.shape[:-1] != (query.shape[0], query.shape[1], KV_HEAD_COUNT):
            raise ValueError(
                "BSND sparse_indices must match query B/S dimensions and use N2=1"
            )
    else:
        if sparse_indices.shape[:-1] != (query.shape[0], KV_HEAD_COUNT):
            raise ValueError("TND sparse_indices must match query T and use N2=1")

    if not 1 <= head_count <= MAX_HEAD_COUNT:
        raise ValueError(f"Query head count must be in [1, {MAX_HEAD_COUNT}]")
    if sinks is not None and (
        sinks.dtype != torch.float32 or tuple(sinks.shape) != (head_count,)
    ):
        raise ValueError("Sinks must be FP32 with one value per query head")


def _validate_kv_shapes(
    key,
    key_rope,
    has_rope,
    block_table,
    batch_size,
    layout_kv,
):
    """Validate KV-side shape relations for the active layout."""
    if has_rope and key_rope.shape[:-1] != key.shape[:-1]:
        raise ValueError("Key_rope prefix dimensions must match key")
    if layout_kv == "BSND":
        if key.shape[0] != batch_size:
            raise ValueError("BSND key batch dimension must match query batch size")
        if key.shape[1] <= 0:
            raise ValueError("BSND key sequence length must be positive")
    elif layout_kv == "PA_BSND":
        block_size = key.shape[1]
        if (block_size < PA_BLOCK_ALIGNMENT
                or block_size > PA_MAX_BLOCK_SIZE
                or block_size % PA_BLOCK_ALIGNMENT != 0):
            raise ValueError(
                f"PA block size must be a multiple of {PA_BLOCK_ALIGNMENT} "
                f"in [{PA_BLOCK_ALIGNMENT}, {PA_MAX_BLOCK_SIZE}]"
            )
        if block_table is None or block_table.dtype != torch.int32:
            raise ValueError("PA_BSND requires an INT32 block_table")
        if (block_table.ndim != 2
                or block_table.shape[0] != batch_size
                or block_table.shape[1] <= 0):
            raise ValueError(
                "Block_table must be rank 2 with B rows and at least one column"
            )


def _validate_shape_consistency(
    query,
    key,
    value,
    sparse_indices,
    query_rope,
    key_rope,
    has_rope,
    sinks,
    block_table,
    batch_size,
    head_count,
    layout_query,
    layout_kv,
):
    """Validate Value, Query, then KV shape relations."""
    if value.shape != key.shape:
        raise ValueError("Value shape must match key shape")
    _validate_query_shapes(
        query, sparse_indices, query_rope, has_rope, sinks, head_count,
        layout_query,
    )
    _validate_kv_shapes(
        key, key_rope, has_rope, block_table, batch_size, layout_kv,
    )


def _validate_sequence_length_args(
    layout_query,
    layout_kv,
    cumulative_query_lengths,
    cumulative_kv_lengths,
    used_query_lengths,
    used_kv_lengths,
    batch_size,
):
    """Validate required sequence lengths and their B/B+1 shapes."""
    if layout_query == "TND":
        if cumulative_query_lengths is None:
            raise ValueError("Cu_seqlens_q is required for TND query layout")
        if cumulative_query_lengths.numel() != batch_size + 1:
            raise ValueError("Cu_seqlens_q length must equal B+1")
    elif cumulative_query_lengths is not None:
        raise ValueError("Cu_seqlens_q is only valid for TND query layout")

    if layout_kv == "TND":
        if cumulative_kv_lengths is None:
            raise ValueError("Cu_seqlens_kv is required for TND KV layout")
        if cumulative_kv_lengths.numel() != batch_size + 1:
            raise ValueError("Cu_seqlens_kv length must equal B+1")
    elif cumulative_kv_lengths is not None:
        raise ValueError("Cu_seqlens_kv is only valid for TND KV layout")

    if used_query_lengths is not None and used_query_lengths.numel() != batch_size:
        raise ValueError("Seqused_q length must equal B")
    if used_kv_lengths is not None and used_kv_lengths.numel() != batch_size:
        raise ValueError("Seqused_kv length must equal B")
    if layout_kv == "PA_BSND" and used_kv_lengths is None:
        raise ValueError("PA_BSND requires seqused_kv")


def validate_sparse_flash_attention_args(
    *,
    query,
    key,
    value,
    sparse_indices,
    block_table,
    query_rope,
    key_rope,
    sinks,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    sparse_block_size,
    layout_query,
    layout_kv,
    sparse_mode,
    pre_tokens,
    next_tokens,
    attention_mode,
    return_softmax_lse,
):
    """Validate public operator arguments before allocating launch resources."""
    has_rope = _resolve_rope_mode(query_rope, key_rope)

    _validate_operator_params(
        attention_mode=attention_mode,
        sparse_block_size=sparse_block_size,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        layout_query=layout_query,
        layout_kv=layout_kv,
        block_table=block_table,
        return_softmax_lse=return_softmax_lse,
    )
    _validate_data_tensor_specs(
        query=query,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        query_rope=query_rope,
        key_rope=key_rope,
        has_rope=has_rope,
        sinks=sinks,
        block_table=block_table,
        layout_query=layout_query,
        layout_kv=layout_kv,
    )
    _validate_sequence_length_tensors(
        cumulative_query_lengths=cu_seqlens_q,
        cumulative_kv_lengths=cu_seqlens_kv,
        used_query_lengths=seqused_q,
        used_kv_lengths=seqused_kv,
    )

    batch_size, head_count = _resolve_batch_head(
        query, cu_seqlens_q, layout_query
    )
    _validate_shape_consistency(
        query=query,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        query_rope=query_rope,
        key_rope=key_rope,
        has_rope=has_rope,
        sinks=sinks,
        block_table=block_table,
        batch_size=batch_size,
        head_count=head_count,
        layout_query=layout_query,
        layout_kv=layout_kv,
    )
    _validate_sequence_length_args(
        layout_query=layout_query,
        layout_kv=layout_kv,
        cumulative_query_lengths=cu_seqlens_q,
        cumulative_kv_lengths=cu_seqlens_kv,
        used_query_lengths=seqused_q,
        used_kv_lengths=seqused_kv,
        batch_size=batch_size,
    )
