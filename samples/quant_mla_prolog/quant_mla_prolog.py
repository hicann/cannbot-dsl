# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Single-launch AW1 MLA prolog with runtime token and core counts.

The kernel consumes physical FP8 NZ weights directly.  Its MXFP8 matrix
multiplications use declarative depth-2 L1/L0 channels, so CANNBotDSL derives
the double-buffer schedule without manual channel transactions.  C0 produces
BF16 Q/KV; C1 produces per-token-head FP8 Q plus ``descale_q`` and per-tensor
FP8 KV.  Each AOT specialization completes in one MixKernel and writes KV
directly to the PA_NZ cache.
"""

from __future__ import annotations

__all__ = ["quant_mla_prolog"]

import math
from functools import lru_cache
from importlib import import_module
from typing import NamedTuple

import cannbotdsl
import torch
import torch._dynamo as torch_dynamo

from cannbotdsl import MemLoc, Tensor, dtypes
from cannbotdsl.aot import export
from cannbotdsl.buffer import Buffer
from cannbotdsl.channel import Channel

try:
    # CANNBotDSL >= 0.5 moved the IR transport boundary under core.
    from cannbotdsl.core.ir_transport import unwrap_operand
except ImportError:  # Compatibility with the validated 0.3 toolchain.
    from cannbotdsl.ir_transport import unwrap_operand
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl._mlir.dialects import cannir
from cannbotdsl.ops.arch import get_block_idx, get_block_num, get_subblock_id
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import make_copy_engine, mem_copy
from cannbotdsl.ops.sync import (
    channel_rewind,
    cube_sync_all,
    cube_sync_block_arrive,
    cube_sync_notify,
    cube_sync_wait,
    global_sync_all,
    vec_sync_all,
    vec_sync_block_arrive,
    vec_sync_block_wait,
)
from cannbotdsl.ops import sync as _sync_ops
from cannbotdsl.ops.reg import (
    PackMode,
    RoundingMode,
    UnpackMode,
    full_mask,
    update_mask,
    vabs,
    vadd,
    vadds,
    vbitwise_and,
    vcast,
    vdiv,
    vdup,
    vdups,
    vgather,
    vgather_reg,
    vgt,
    vload,
    vload_broadcast,
    vload_unpack,
    vmax,
    vmaxs,
    vmins,
    vmul,
    vmuls,
    vreduce_max,
    vreduce_sum,
    vreinterpret,
    vselect,
    vshl,
    vshr,
    vsqrt,
    vstore_first,
    vstore_pack,
    vsub,
    varange,
    vmem_bar,
)
from cannbotdsl.tensor import make_layout, make_tensor, make_tiler, tile_slice

try:
    from cannbotdsl.types import Float8E4M3FN, Float8E8M0, RegLayout
except ImportError:  # CANNBotDSL >= 0.5 uses public dtype descriptors.
    from cannbotdsl import RegLayout

    Float8E4M3FN = dtypes.float8_e4m3fn
    Float8E8M0 = dtypes.float8_e8m0

# The C0/C1, decode/prefill, and dispatcher variants are deliberately explicit.
# Extracting their large or repeated bodies would change DSL statement/channel,
# synchronization, or rounding order. Their long signatures are fixed JIT/Torch
# schemas; runtime-created channel attributes must retain their creation order.
# The one cross-file clone is kept local because that peer is outside this change.

DIM, Q_LORA, KV_LORA = 7168, 1536, 512
N_HEADS, QK_NOPE, ROPE = 96, 128, 64
HEAD_DIM, FEATURES, FP8_MAX = QK_NOPE + ROPE, KV_LORA + ROPE, 448.0
HIF8_MAX = 32768.0
PA_BLOCK_SIZE = 128
MIN_AIC_CORE_COUNT, MAX_AIC_CORE_COUNT = 1, 32
# These shapes remain the performance regression set.  They are not an
# operator ABI restriction: T is the runtime token count B*S.
PROFILED_TOKEN_COUNTS = (8, 32, 64, 128, 256, 512, 1024)
# Tiling intervals, not exact token counts; core count and cache pages are dynamic.
NATIVE_TILE_RANGES = ((1, 8), (9, 32), (33, 64), (65, 128), (129, None))


def _ceil_align(value, divisor):
    return (value + divisor - 1) // divisor * divisor


def _workspace_token_capacity(tokens):
    """Physical GM workspace rows; public tensors keep the logical T."""
    if tokens <= 16:
        return 16
    if tokens <= 128:
        return _ceil_align(tokens, 32)
    return _ceil_align(tokens, 128)


def _native_tile_m(tokens):
    if tokens <= 8:
        return 8
    if tokens <= 32:
        return 32
    if tokens <= 64:
        return 64
    return 128


def _output_token_capacity(tokens):
    return _ceil_align(tokens, _native_tile_m(tokens))


@lru_cache(maxsize=None)
def _device_aic_core_num(device_index):
    """Return the hardware AIC limit; stream quotas remain runtime values."""
    props = torch.npu.get_device_properties(device_index)
    core_num = min(int(getattr(props, "cube_core_num", 0)), MAX_AIC_CORE_COUNT)
    if not MIN_AIC_CORE_COUNT <= core_num <= MAX_AIC_CORE_COUNT:
        raise RuntimeError(
            "quant_mla_prolog requires an available AIC count in "
            f"[{MIN_AIC_CORE_COUNT}, {MAX_AIC_CORE_COUNT}], got {core_num}"
        )
    return core_num


def _current_aic_core_num(device_index):
    """Resolve this stream's effective Mix launch width without caching quotas."""
    core_num = _device_aic_core_num(device_index)
    get_limit = getattr(torch.npu, "get_stream_limit", None)
    if get_limit is not None:
        limits = get_limit(torch.npu.current_stream(device_index))
        cube_num = int(limits["cube_core_num"])
        vector_num = int(limits["vector_core_num"])
        # A Mix block uses one AIC and two AIV subcores.  Device properties
        # alone do not describe a stream whose resources have been limited.
        core_num = min(core_num, cube_num, vector_num // 2)
    if not MIN_AIC_CORE_COUNT <= core_num <= MAX_AIC_CORE_COUNT:
        raise RuntimeError(
            "quant_mla_prolog needs at least one AIC and two AIVs on "
            f"the current stream; resolved Mix block count is {core_num}"
        )
    return core_num


def _tensor_aic_core_num(value):
    device_index = value.device.index
    if device_index is None:
        device_index = torch.npu.current_device()
    return _current_aic_core_num(int(device_index))


def _fp8_nz_transposed_view(gm_weight, output_size, input_size):
    """Describe packed ``[O/32,K,32]`` storage as logical ``W.T[K,O]``.

    Splitting the physical K axis into ``K/16 x 16`` gives the canonical
    FP8 NZ descriptor without moving any byte in GM.  The resulting view can
    be copied directly into an NZ L1B tile; only the required L1B-to-L0B
    transpose remains.
    """
    typed_weight = gm_weight.view(dtype=dtypes.int8)
    pointer = cannir.extract_pointer(unwrap_operand(typed_weight))
    return make_tensor(
        pointer,
        make_layout((input_size, output_size)),
        physical_layout=make_layout((output_size // 32, input_size // 16, 16, 32)),
        data_format="nz",
        storage_size_bytes=output_size * input_size,
    )


_QBMM_MODULE = None


def _require_npu(name, value, device):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device.type not in {"npu", "privateuseone"}:
        raise RuntimeError(f"{name} must be on NPU")
    if value.device != device or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous on the common NPU device")


def _logical_scale(value, outer, inner):
    groups = (inner + 63) // 64
    if value.dim() == 3:
        if tuple(value.shape) != (outer, groups, 2):
            raise ValueError(
                f"scale shape {tuple(value.shape)} != {(outer, groups, 2)}"
            )
        return value
    if value.dim() == 4:
        if (
            value.shape[0] * value.shape[2] < outer
            or value.shape[1] < groups
            or value.shape[3] != 2
        ):
            raise ValueError("invalid physical NZ scale storage")
        raw = value.view(torch.uint8).permute(0, 2, 1, 3).contiguous()
        logical = raw.reshape(-1, value.shape[1], 2)[:outer, :groups].contiguous()
        return logical.view(torch.float8_e8m0fnu)
    raise ValueError("MX scale must be logical rank-3 or physical NZ rank-4")


class _MlaKvEpilogMxfp8Unit:
    """One-task KV RMSNorm, combined quantization and PA_NZ scatter.

    KVA remains FP32 ND in workspace. The kernel reads each row once, applies
    the AscendC BF16 gamma/rope boundaries, quantizes the combined 512+64 row,
    and writes its 18 physical C0=32 cache blocks directly. No cache or weight
    NZ tensor is materialized as ND.
    """

    def __init__(self, page_size, norm_eps, output_depth=1):
        self.page_size = page_size
        self.norm_eps = norm_eps
        self.x_channel = Channel(MemLoc.UB, (1, FEATURES), dtypes.float32, depth=2)
        self.gamma_channel = Channel(MemLoc.UB, (1, KV_LORA), dtypes.float32, depth=1)
        self.output_channel = Channel(
            MemLoc.UB, (FEATURES // 32, 32), Float8E4M3FN, depth=output_depth
        )
        self.scale_ub = Buffer(MemLoc.UB, (1, 8), dtypes.float32)
        self.scalar_ub = Buffer(MemLoc.UB, (1, 8), dtypes.float32)

    @jit
    def __call__(
        self,
        kva: Tensor,
        gamma: Tensor,
        qscale: Tensor,
        cache: Tensor,
        cache_index: Tensor,
    ):
        # get_subblock_id cannot be used as a second logical row owner here:
        # on MIX_AIC_1_2 its physical owner numbering differs from AIC block
        # numbering. Subblock 0 with block-strided rows is deterministic and
        # covers the AIV cores needed by all supported T values.
        if get_subblock_id() == 0:
            start = get_block_idx()
            stride = get_block_num()
            if start < cache_index.shape[0]:
                gamma_slot = self.gamma_channel.produce()
                mem_copy(gamma_slot, gamma)
                mem_copy(tile_slice(self.scale_ub, (1, 1), (0, 0)), qscale)
                gamma_ub = self.gamma_channel.consume()
                for token in range(start, cache_index.shape[0], stride):
                    self._process_row(kva, gamma_ub, cache, cache_index, token)

    @jit
    def _process_row(self, kva, gamma, cache, cache_index, token):
        x_row = self.x_channel.produce()
        mem_copy(x_row, tile_slice(kva, (1, FEATURES), (token, 0)))
        x_row = self.x_channel.consume()
        encoded = self.output_channel.produce()
        with vf(mode="simd"):
            vmem_bar(mode="vst_vld")
            full = full_mask()
            scalar = update_mask(1, elem_bits=32)[0]
            total = vdups(0.0, dtypes.float32, mask=scalar)
            for segment in range_constexpr(KV_LORA // 64):
                value = vload(x_row, segment * 64)
                total = vadd(
                    total,
                    vreduce_sum(vmul(value, value, mask=full), mask=full),
                    mask=scalar,
                )
            root = vsqrt(
                vadds(
                    vmuls(total, 1.0 / KV_LORA, mask=scalar), self.norm_eps, mask=scalar
                ),
                mask=scalar,
            )
            inv_rms = vdiv(vdups(1.0, dtypes.float32, mask=scalar), root, mask=scalar)
            vstore_first(self.scalar_ub, 0, inv_rms)
            vmem_bar(mode="vst_vld")
            inv_rms_vector = vload_broadcast(self.scalar_ub, 0)
            kv_scale = vload_broadcast(self.scale_ub, 0)
            for segment in range_constexpr(KV_LORA // 64):
                value = vload(x_row, segment * 64)
                gamma_value = vload(gamma, segment * 64)
                gamma_value = vcast(
                    vcast(
                        gamma_value,
                        dtypes.bfloat16,
                        mask=full,
                        rounding=RoundingMode.RN,
                    ),
                    dtypes.float32,
                    mask=full,
                )
                normalized = vmul(
                    vmul(value, inv_rms_vector, mask=full), gamma_value, mask=full
                )
                quantized = vdiv(normalized, kv_scale, mask=full)
                quantized = vmaxs(quantized, -FP8_MAX, mask=full)
                quantized = vmins(quantized, FP8_MAX, mask=full)
                fp8_value = vcast(
                    quantized,
                    dtypes.float8_e4m3fn,
                    mask=full,
                    reg_layout=RegLayout.ZERO,
                )
                vstore_pack(
                    encoded, segment * 64, fp8_value, full, pack_mode=PackMode.B32_TO_B8
                )
            rope_value = vload(x_row, KV_LORA)
            rope_value = vcast(
                vcast(rope_value, dtypes.bfloat16, mask=full, rounding=RoundingMode.RN),
                dtypes.float32,
                mask=full,
            )
            quantized = vdiv(rope_value, kv_scale, mask=full)
            quantized = vmaxs(quantized, -FP8_MAX, mask=full)
            quantized = vmins(quantized, FP8_MAX, mask=full)
            fp8_value = vcast(
                quantized, dtypes.float8_e4m3fn, mask=full, reg_layout=RegLayout.ZERO
            )
            vstore_pack(encoded, KV_LORA, fp8_value, full, pack_mode=PackMode.B32_TO_B8)

        output_row = self.output_channel.consume()
        slot = cache_index[token]
        if slot >= 0:
            page = slot // self.page_size
            row = slot % self.page_size
            cache_blocks = FEATURES // 32
            cache_flat = cache.view(
                (cache.shape[0] * cache_blocks * self.page_size, 32)
            )
            for feature_block in range(cache_blocks):
                offset = (page * cache_blocks + feature_block) * self.page_size + row
                mem_copy(
                    tile_slice(cache_flat, (1, 32), (offset, 0)),
                    tile_slice(output_row, (1, 32), (0, feature_block)),
                )

    @jit
    def process_assigned_rows(
        self,
        kva: Tensor,
        gamma: Tensor,
        qscale: Tensor,
        cache: Tensor,
        cache_index: Tensor,
        start,
        stride,
    ):
        # get_subblock_id cannot be used as a second logical row owner here:
        # on MIX_AIC_1_2 its physical owner numbering differs from AIC block
        # numbering. Subblock 0 with block-strided rows is deterministic and
        # covers the AIV cores needed by all supported T values.
        if get_subblock_id() == 0:
            if start < cache_index.shape[0]:
                gamma_slot = self.gamma_channel.produce()
                mem_copy(gamma_slot, gamma)
                mem_copy(tile_slice(self.scale_ub, (1, 1), (0, 0)), qscale)
                gamma_ub = self.gamma_channel.consume()
                for token in range(start, cache_index.shape[0], stride):
                    self._process_row(kva, gamma_ub, cache, cache_index, token)


class _MlaKvEpilogBf16Unit:
    """Normalize the 512-wide latent, append 64 BF16 channels and scatter."""

    def __init__(self, page_size, norm_eps, output_depth=1):
        self.page_size = page_size
        self.norm_eps = norm_eps
        self.x_channel = Channel(MemLoc.UB, (1, FEATURES), dtypes.float32, depth=2)
        self.gamma_channel = Channel(MemLoc.UB, (1, KV_LORA), dtypes.float32, depth=1)
        self.output_channel = Channel(
            MemLoc.UB,
            (1, FEATURES),
            dtypes.bfloat16,
            depth=output_depth,
        )
        self.scalar_ub = Buffer(MemLoc.UB, (1, 8), dtypes.float32)

    @jit
    def _process_row(self, kva, gamma, cache, cache_index, token):
        x_row = self.x_channel.produce()
        mem_copy(
            x_row,
            tile_slice(kva, (1, FEATURES), (token, 0)),
        )
        x_row = self.x_channel.consume()
        output_row = self.output_channel.produce()
        with vf(mode="simd"):
            vmem_bar(mode="vst_vld")
            full = full_mask()
            scalar = update_mask(1, elem_bits=32)[0]
            total = vdups(0.0, dtypes.float32, mask=scalar)
            for segment in range_constexpr(KV_LORA // 64):
                value = vload(x_row, segment * 64)
                total = vadd(
                    total,
                    vreduce_sum(vmul(value, value, mask=full), mask=full),
                    mask=scalar,
                )
            root = vsqrt(
                vadds(
                    vmuls(total, 1.0 / KV_LORA, mask=scalar),
                    self.norm_eps,
                    mask=scalar,
                ),
                mask=scalar,
            )
            inv_rms = vdiv(vdups(1.0, dtypes.float32, mask=scalar), root, mask=scalar)
            vstore_first(self.scalar_ub, 0, inv_rms)
            vmem_bar(mode="vst_vld")
            inv_rms_vector = vload_broadcast(self.scalar_ub, 0)
            for segment in range_constexpr(KV_LORA // 64):
                value = vload(x_row, segment * 64)
                gamma_value = vload(gamma, segment * 64)
                gamma_value = vcast(
                    vcast(
                        gamma_value,
                        dtypes.bfloat16,
                        mask=full,
                        rounding=RoundingMode.RN,
                    ),
                    dtypes.float32,
                    mask=full,
                )
                normalized = vmul(
                    vmul(value, inv_rms_vector, mask=full),
                    gamma_value,
                    mask=full,
                )
                rounded = vcast(
                    normalized,
                    dtypes.bfloat16,
                    mask=full,
                    rounding=RoundingMode.RN,
                )
                vstore_pack(
                    output_row,
                    segment * 64,
                    rounded,
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
            rope_value = vload(x_row, KV_LORA)
            rope_value = vcast(
                rope_value,
                dtypes.bfloat16,
                mask=full,
                rounding=RoundingMode.RN,
            )
            vstore_pack(
                output_row,
                KV_LORA,
                rope_value,
                full,
                pack_mode=PackMode.B32_TO_B16,
            )

        encoded = self.output_channel.consume()
        slot = cache_index[token]
        if slot >= 0:
            page = slot // self.page_size
            row = slot % self.page_size
            cache_blocks = FEATURES // 16
            cache_flat = cache.view(
                (cache.shape[0] * cache_blocks * self.page_size, 16)
            )
            for feature_block in range(cache_blocks):
                offset = (page * cache_blocks + feature_block) * self.page_size + row
                mem_copy(
                    tile_slice(cache_flat, (1, 16), (offset, 0)),
                    tile_slice(encoded, (1, 16), (0, feature_block)),
                )

    @jit
    def process_assigned_rows(
        self,
        kva: Tensor,
        gamma: Tensor,
        cache: Tensor,
        cache_index: Tensor,
        start,
        stride,
    ):
        if get_subblock_id() == 0:
            if start < cache_index.shape[0]:
                gamma_slot = self.gamma_channel.produce()
                mem_copy(gamma_slot, gamma)
                gamma_ub = self.gamma_channel.consume()
                for token in range(start, cache_index.shape[0], stride):
                    self._process_row(kva, gamma_ub, cache, cache_index, token)


class _QaRmsMixUnit:
    def __init__(self, eps: float):
        self.eps = eps
        self.x = Channel(MemLoc.UB, (1, Q_LORA), dtypes.float32, depth=2)
        self.gamma = Channel(MemLoc.UB, (1, Q_LORA), dtypes.float32, depth=1)
        self.bf16 = Buffer(MemLoc.UB, (1, Q_LORA), dtypes.bfloat16)
        self.out = Channel(MemLoc.UB, (1, Q_LORA), Float8E4M3FN, depth=2)
        # Packed stores are 32-byte transactions even when only two lanes are
        # active, so keep one aligned tail beyond the 48 logical scale bytes.
        self.scale8 = Buffer(MemLoc.UB, (1, 64), dtypes.int8)
        # vload reads one full 64-lane B32 vector.
        self.scale32 = Buffer(MemLoc.UB, ((Q_LORA // 64), 64), dtypes.uint32)
        self.scalar = Buffer(MemLoc.UB, (1, 8), dtypes.float32)

    @jit
    def process_one(self, qa_raw, gamma, qa_mx, qa_scale_raw, token):
        x_slot = self.x.produce()
        mem_copy(x_slot, tile_slice(qa_raw, (1, Q_LORA), (token, 0)))
        x = self.x.consume()
        y = self.out.produce()
        with vf(mode="simd"):
            full = full_mask()
            mask32 = update_mask(32, elem_bits=32)[0]
            mask48 = update_mask((Q_LORA // 32), elem_bits=32)[0]
            mask1 = update_mask(1, elem_bits=32)[0]

            # FP32 source-order RMS sum and sqrt->division, matching the
            # current golden expression rather than an rsqrt rewrite.
            total = vdups(0.0, dtypes.float32, mask=mask1)
            for seg in range_constexpr((Q_LORA // 64)):
                value = vload(x, seg * 64)
                total = vadd(
                    total,
                    vreduce_sum(vmul(value, value, mask=full), mask=full),
                    mask=mask1,
                )
            mean = vmuls(total, 1.0 / Q_LORA, mask=mask1)
            root = vsqrt(vadds(mean, self.eps, mask=mask1), mask=mask1)
            inv = vdiv(vdups(1.0, dtypes.float32, mask=mask1), root, mask=mask1)
            vstore_first(self.scalar, 0, inv)
            vmem_bar(mode="vst_vld")
            inv_vec = vload_broadcast(self.scalar, 0)

            # Public gamma is FP32, while the AscendC ABI consumes BF16.
            # The RMS output then crosses CAST_ROUND (ties away) to BF16.
            for seg in range_constexpr((Q_LORA // 64)):
                value = vload(x, seg * 64)
                gamma_value = vload(gamma, seg * 64)
                gamma_bf16 = vcast(
                    gamma_value,
                    dtypes.bfloat16,
                    mask=full,
                    rounding=RoundingMode.RN,
                )
                gamma_fp32 = vcast(gamma_bf16, dtypes.float32, mask=full)
                normalized = vmul(
                    vmul(value, inv_vec, mask=full), gamma_fp32, mask=full
                )
                rounded = vcast(
                    normalized,
                    dtypes.bfloat16,
                    mask=full,
                    rounding=RoundingMode.RNA,
                )
                vstore_pack(
                    self.bf16,
                    seg * 64,
                    rounded,
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )

            vmem_bar(mode="vst_vld")
            ids = varange(0, dtypes.uint32)
            second_ids = vadds(
                vbitwise_and(ids, vdups(31, dtypes.uint32, mask=full), mask=full),
                32,
                mask=full,
            )
            zero_u32 = vdups(0, dtypes.uint32, mask=full)
            eight_u32 = vdups(8, dtypes.uint32, mask=full)
            inverse_bias_u32 = vdups(254, dtypes.uint32, mask=full)
            for seg in range_constexpr((Q_LORA // 64)):
                packed = vload_unpack(
                    self.bf16, seg * 64, unpack_mode=UnpackMode.B16_TO_B32
                )
                value = vcast(packed, dtypes.float32, mask=full)
                upper = vreinterpret(
                    vgather_reg(vreinterpret(value, dtypes.uint32), second_ids),
                    dtypes.float32,
                )
                max0 = vreduce_max(vabs(value, mask=mask32), mask=mask32)
                max1 = vreduce_max(vabs(upper, mask=mask32), mask=mask32)

                bits0 = vreinterpret(max0, dtypes.uint32)
                bits1 = vreinterpret(max1, dtypes.uint32)
                exp0 = vbitwise_and(
                    vshr(bits0, 23, mask=mask1),
                    vdups(255, dtypes.uint32, mask=full),
                    mask=mask1,
                )
                exp1 = vbitwise_and(
                    vshr(bits1, 23, mask=mask1),
                    vdups(255, dtypes.uint32, mask=full),
                    mask=mask1,
                )
                raw0 = vselect(
                    vsub(exp0, eight_u32, mask=mask1),
                    zero_u32,
                    cond_mask=vgt(exp0, eight_u32, mask=mask1),
                )
                raw1 = vselect(
                    vsub(exp1, eight_u32, mask=mask1),
                    zero_u32,
                    cond_mask=vgt(exp1, eight_u32, mask=mask1),
                )
                scale32_row = tile_slice(self.scale32, (1, 64), (seg, 0))
                vstore_first(scale32_row, 0, raw0)
                vstore_first(scale32_row, 1, raw1)

                inv_bits0 = vshl(
                    vsub(inverse_bias_u32, raw0, mask=mask1), 23, mask=mask1
                )
                inv_bits1 = vshl(
                    vsub(inverse_bias_u32, raw1, mask=mask1), 23, mask=mask1
                )
                inv_bits0 = vselect(
                    inv_bits0, zero_u32, cond_mask=vgt(exp0, zero_u32, mask=mask1)
                )
                inv_bits1 = vselect(
                    inv_bits1, zero_u32, cond_mask=vgt(exp1, zero_u32, mask=mask1)
                )
                inv0 = vreinterpret(inv_bits0, dtypes.float32)
                inv1 = vreinterpret(inv_bits1, dtypes.float32)
                inv_scale = vselect(
                    vdup(inv0, mask=full),
                    vdup(inv1, mask=full),
                    cond_mask=vgt(vdups(32, dtypes.uint32, mask=full), ids, mask=full),
                )
                # DynamicQuantPerBlock multiplies in BF16 before E4M3 RNE.
                scaled_bf16 = vcast(
                    vmul(value, inv_scale, mask=full),
                    dtypes.bfloat16,
                    mask=full,
                    rounding=RoundingMode.RN,
                )
                scaled = vcast(scaled_bf16, dtypes.float32, mask=full)
                scaled = vmaxs(scaled, -FP8_MAX, mask=full)
                scaled = vmins(scaled, FP8_MAX, mask=full)
                fp8 = vcast(
                    scaled,
                    dtypes.float8_e4m3fn,
                    mask=full,
                    rounding=RoundingMode.RN,
                    reg_layout=RegLayout.ZERO,
                )
                vstore_pack(y, seg * 64, fp8, full, pack_mode=PackMode.B32_TO_B8)

            # Gather [row0.lane0,row0.lane1,...,row23.lane1] into one
            # register, then perform one aligned 48-byte packed store.
            vmem_bar(mode="vst_vld")
            gather_idx = vadd(
                vshl(vshr(ids, 1, mask=full), 6, mask=full),
                vbitwise_and(ids, vdups(1, dtypes.uint32, mask=full), mask=full),
                mask=full,
            )
            scales = vgather(self.scale32, gather_idx, mask=mask48)
            vstore_pack(self.scale8, 0, scales, mask48, pack_mode=PackMode.B32_TO_B8)

        result = self.out.consume()
        mem_copy(tile_slice(qa_mx, (1, Q_LORA), (token, 0)), result)
        scale_flat = qa_scale_raw.view((qa_scale_raw.shape[0], (Q_LORA // 32)))
        mem_copy(
            tile_slice(scale_flat, (1, (Q_LORA // 32)), (token, 0)),
            tile_slice(self.scale8, (1, (Q_LORA // 32)), (0, 0)),
        )


class _MlaQHeadTokenEpilogMxfp8Unit:
    """Eight-row combined Q per-token-head E4M3 quantization."""

    def __init__(self, workers, padded_t, logical_t, token_major_rope=False):
        self.workers = workers
        self.padded_t = padded_t
        self.logical_t = logical_t
        self.token_major_rope = token_major_rope
        self.rows = 8
        self.nope_channel = Channel(MemLoc.UB, (8, KV_LORA), dtypes.bfloat16, depth=2)
        self.rope_channel = Channel(MemLoc.UB, (8, ROPE), dtypes.bfloat16, depth=2)
        self.output_channel = Channel(MemLoc.UB, (8, FEATURES), Float8E4M3FN, depth=2)
        self.scale_channel = Channel(MemLoc.UB, (8, 1), dtypes.float32, depth=2)
        self.scalar_ub = Buffer(MemLoc.UB, (8, 8), dtypes.float32)

    @jit
    def __call__(
        self, q_nope: Tensor, q_combined: Tensor, q_out: Tensor, descale_q: Tensor
    ):
        nope = q_nope.view((N_HEADS * self.padded_t, KV_LORA))
        combined = q_combined.view((N_HEADS * self.padded_t, HEAD_DIM))
        output = q_out.view((self.logical_t, N_HEADS * FEATURES))
        scales = descale_q.view((self.logical_t, N_HEADS))
        total_batches = (N_HEADS * self.logical_t) // 8
        chunk = (total_batches + self.workers - 1) // self.workers
        start = get_block_idx() * chunk
        end = start + chunk
        if end > total_batches:
            end = total_batches
        for batch in range(start, end):
            self.process_batch(nope, combined, output, scales, batch)

    @jit
    def process_batch(self, q_nope, q_rope, q_out, descale_q, batch):
        nope_slot = self.nope_channel.produce()
        rope_slot = self.rope_channel.produce()
        head = batch // (self.logical_t // 8)
        token_batch = batch % (self.logical_t // 8)
        source_batch = head * (self.padded_t // 8) + token_batch
        mem_copy(nope_slot, tile_slice(q_nope, (8, KV_LORA), (source_batch, 0)))
        if self.token_major_rope:
            token_start = token_batch * 8
            rope_col_start = head * HEAD_DIM + QK_NOPE
            # fmt: off
            mem_copy(
                rope_slot,
                q_rope[
                    token_start:token_start + 8,
                    rope_col_start:rope_col_start + ROPE,
                ],
            )
            # fmt: on
        else:
            mem_copy(
                rope_slot,
                tile_slice(
                    q_rope,
                    (8, ROPE),
                    (source_batch, QK_NOPE // ROPE),
                ),
            )
        q_nope_rows = self.nope_channel.consume()
        q_rope_rows = self.rope_channel.consume()
        encoded = self.output_channel.produce()
        scales = self.scale_channel.produce()
        with vf(mode="simd"):
            vmem_bar(mode="vst_vld")
            full = full_mask()
            scalar = update_mask(1, elem_bits=32)[0]
            for row in range_constexpr(8):
                absmax = vdups(1.0e-8, dtypes.float32, mask=scalar)
                for segment in range_constexpr(KV_LORA // 64):
                    unpacked = vload_unpack(
                        q_nope_rows,
                        row * KV_LORA + segment * 64,
                        unpack_mode=UnpackMode.B16_TO_B32,
                    )
                    value = vcast(unpacked, dtypes.float32, mask=full)
                    absmax = vmax(
                        absmax,
                        vreduce_max(vabs(value, mask=full), mask=full),
                        mask=scalar,
                    )
                unpacked = vload_unpack(
                    q_rope_rows,
                    row * ROPE,
                    unpack_mode=UnpackMode.B16_TO_B32,
                )
                value = vcast(unpacked, dtypes.float32, mask=full)
                absmax = vmax(
                    absmax,
                    vreduce_max(vabs(value, mask=full), mask=full),
                    mask=scalar,
                )
                scale = vmuls(absmax, 1.0 / FP8_MAX, mask=scalar)
                vstore_first(self.scalar_ub, row * 8, scale)
                vstore_first(scales, row, scale)
                vmem_bar(mode="vst_vld")
                scale_vector = vload_broadcast(self.scalar_ub, row * 8)
                for segment in range_constexpr(KV_LORA // 64):
                    unpacked = vload_unpack(
                        q_nope_rows,
                        row * KV_LORA + segment * 64,
                        unpack_mode=UnpackMode.B16_TO_B32,
                    )
                    value = vcast(unpacked, dtypes.float32, mask=full)
                    quantized = vdiv(value, scale_vector, mask=full)
                    fp8_value = vcast(
                        quantized,
                        dtypes.float8_e4m3fn,
                        mask=full,
                        reg_layout=RegLayout.ZERO,
                    )
                    vstore_pack(
                        encoded,
                        row * FEATURES + segment * 64,
                        fp8_value,
                        full,
                        pack_mode=PackMode.B32_TO_B8,
                    )
                unpacked = vload_unpack(
                    q_rope_rows,
                    row * ROPE,
                    unpack_mode=UnpackMode.B16_TO_B32,
                )
                value = vcast(unpacked, dtypes.float32, mask=full)
                quantized = vdiv(value, scale_vector, mask=full)
                fp8_value = vcast(
                    quantized,
                    dtypes.float8_e4m3fn,
                    mask=full,
                    reg_layout=RegLayout.ZERO,
                )
                vstore_pack(
                    encoded,
                    row * FEATURES + KV_LORA,
                    fp8_value,
                    full,
                    pack_mode=PackMode.B32_TO_B8,
                )
        output_rows = self.output_channel.consume()
        output_scales = self.scale_channel.consume()
        mem_copy(tile_slice(q_out, (8, FEATURES), (token_batch, head)), output_rows)
        mem_copy(tile_slice(descale_q, (8, 1), (token_batch, head)), output_scales)


class _MlaQHeadTokenEpilogBf16Unit:
    """Merge one head's BF16 NoPE and RoPE slices without quantization."""

    def __init__(self, rows, padded_t, logical_t):
        self.rows = rows
        self.padded_t = padded_t
        self.logical_t = logical_t
        self.nope_channel = Channel(
            MemLoc.UB, (rows, KV_LORA), dtypes.bfloat16, depth=2
        )
        self.rope_channel = Channel(MemLoc.UB, (rows, ROPE), dtypes.bfloat16, depth=2)
        self.output_channel = Channel(
            MemLoc.UB, (rows, FEATURES), dtypes.bfloat16, depth=2
        )

    @jit
    def process_batch(self, q_nope, q_combined, q_out, head, token_batch):
        nope = q_nope.view((N_HEADS * self.padded_t, KV_LORA))
        combined = q_combined.view((self.padded_t, N_HEADS * HEAD_DIM))
        output = q_out.view((self.logical_t, N_HEADS * FEATURES))
        source_batch = head * (self.padded_t // self.rows) + token_batch
        token_start = token_batch * self.rows
        nope_slot = self.nope_channel.produce()
        mem_copy(
            nope_slot,
            tile_slice(nope, (self.rows, KV_LORA), (source_batch, 0)),
        )
        rope_slot = self.rope_channel.produce()
        # fmt: off
        mem_copy(
            rope_slot,
            combined[
                token_start:token_start + self.rows,
                head * HEAD_DIM + QK_NOPE:(head + 1) * HEAD_DIM,
            ],
        )
        # fmt: on
        q_nope_rows = self.nope_channel.consume()
        q_rope_rows = self.rope_channel.consume()
        merged = self.output_channel.produce()
        with vf(mode="simd"):
            vmem_bar(mode="vst_vld")
            full = full_mask()
            for row in range_constexpr(self.rows):
                for segment in range_constexpr(KV_LORA // 64):
                    value = vload_unpack(
                        q_nope_rows,
                        row * KV_LORA + segment * 64,
                        unpack_mode=UnpackMode.B16_TO_B32,
                    )
                    vstore_pack(
                        merged,
                        row * FEATURES + segment * 64,
                        value,
                        full,
                        pack_mode=PackMode.B32_TO_B16,
                    )
                value = vload_unpack(
                    q_rope_rows,
                    row * ROPE,
                    unpack_mode=UnpackMode.B16_TO_B32,
                )
                vstore_pack(
                    merged,
                    row * FEATURES + KV_LORA,
                    value,
                    full,
                    pack_mode=PackMode.B32_TO_B16,
                )
        mem_copy(
            tile_slice(output, (self.rows, FEATURES), (token_batch, head)),
            self.output_channel.consume(),
        )


def _o11_front_sync_aic_to_aiv(drain_vec=True):
    """Three-phase Mix barrier without the redundant AIV all-core phase."""
    # Drain every producer before publishing readiness.  AIVs funnel through
    # their paired AIC, the 32 AICs meet once, and each AIC wakes its two
    # local AIVs.  The omitted AIV-to-AIV meeting in global_sync_all() cannot
    # add ordering here because all AIVs are already held by the funnel.
    cube_sync_all()
    if drain_vec:
        vec_sync_all()
    _sync_ops.vec_sync_block_arrive(_sync_ops.PIPE.MTE3, 0, mode=2)
    _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, 0, mode=2)
    cube_sync_block_arrive(_sync_ops.PIPE.FIXPIPE, 1, mode=0)
    _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, 1, mode=0)
    cube_sync_block_arrive(_sync_ops.PIPE.MTE3, 2, mode=2)
    vec_sync_block_wait(_sync_ops.PIPE.S, 2, mode=2)


class _O11MlaRef17FrontUnit:
    """N64/K256 front over the public WQA/WKVA descriptors."""

    def __init__(self, m):
        self.m = m
        # AscendC/David front tiling uses M128/N64/K256 as the per-core tile
        # ceiling.  The valid M remains the actual T: GM copies and FixPipe
        # only transfer T rows; format alignment handles inactive tail lanes
        # inside the tile without changing the logical M dimension.
        self.base_m = 128
        self.base_n, self.base_k, self.k_l1 = 64, 256, 1024
        self.scale_groups = DIM // 64
        self.scale_l1_len = self.scale_groups * 2
        self.scale_l0_len = self.base_k // 32
        _bm, bn, bk, kl1 = (
            self.base_m,
            self.base_n,
            self.base_k,
            self.k_l1,
        )
        self.l1_a = Channel(
            MemLoc.L1,
            (self.m, kl1),
            dtypes.int8,
            depth=2,
            data_format="nz",
        )
        self.l1_b = Channel(
            MemLoc.L1,
            (kl1, bn),
            dtypes.int8,
            depth=2,
            data_format="nz",
        )
        self.l1_scale_a = Channel(
            MemLoc.L1,
            (self.m, self.scale_l1_len),
            Float8E8M0,
            depth=1,
            data_format="zn",
        )
        self.l1_scale_b = Channel(
            MemLoc.L1,
            (self.scale_l1_len, bn),
            Float8E8M0,
            depth=1,
            data_format="nz",
        )
        self.l0a = Channel(MemLoc.L0A, (self.m, bk), Float8E4M3FN, depth=2)
        self.l0b = Channel(MemLoc.L0B, (bk, bn), Float8E4M3FN, depth=2)
        self.l0c = Channel(MemLoc.L0C, (self.m, bn), dtypes.float32, depth=2)
        self.copy_a = make_copy_engine(format_transform="nd2nz")
        self.copy_scale_a = make_copy_engine(
            format_transform="mx_scale_and",
        )
        self.copy_scale_b = make_copy_engine(
            format_transform="mx_scale_bdn",
        )
        self.fixpipe = make_copy_engine()

    @jit
    def project_qa_owned_tile(
        self,
        qa_gm,
        x_gm,
        wqa_gm,
        sx_gm,
        swqa_gm,
        block,
        m_idx,
    ):
        """Run one complete N64 tile using twenty-eight K256 accumulations."""
        _bm, bn, bk, kl1 = (
            self.base_m,
            self.base_n,
            self.base_k,
            self.k_l1,
        )
        weight = _fp8_nz_transposed_view(wqa_gm, Q_LORA, DIM)
        scale_a_write = self.l1_scale_a.produce()
        mem_copy(
            scale_a_write,
            tile_slice(sx_gm, (self.m, self.scale_groups, 2), (m_idx, 0, 0)),
            engine=self.copy_scale_a,
        )
        scale_b_write = self.l1_scale_b.produce()
        mem_copy(
            scale_b_write,
            tile_slice(
                swqa_gm,
                (bn, self.scale_groups, 2),
                (block, 0, 0),
            ),
            engine=self.copy_scale_b,
        )
        scale_a = self.l1_scale_a.consume().reinterpret(
            shape=(self.m, self.scale_l1_len)
        )
        scale_b = self.l1_scale_b.consume()
        l0c_write = self.l0c.produce()
        for k1 in range_constexpr(DIM // kl1):
            # A/B are one logical transaction per K1024 window.  Channel
            # depth=2 lets the compiler pipeline consecutive iterations.
            l1_a_write = self.l1_a.produce()
            mem_copy(
                l1_a_write,
                tile_slice(x_gm, (self.m, kl1), (m_idx, k1)),
                engine=self.copy_a,
            )
            l1_b_write = self.l1_b.produce()
            mem_copy(
                l1_b_write,
                tile_slice(
                    weight,
                    (kl1, bn),
                    (k1, block),
                ),
            )
            l1_a_read = self.l1_a.consume()
            l1_b_read = self.l1_b.consume()
            # Keep the declared pitch when interpreting the int8 carrier as
            # FP8, then bound the logical M window for the K subtiles.
            a_fp8 = l1_a_read.reinterpret(dtype=Float8E4M3FN)
            a_window = a_fp8.reinterpret(shape=(self.m, kl1))
            b_fp8 = l1_b_read.reinterpret(dtype=Float8E4M3FN)
            for k0 in range_constexpr(kl1 // bk):
                global_k0 = k1 * (kl1 // bk) + k0
                l0a_write = self.l0a.produce()
                mem_copy(
                    l0a_write,
                    tile_slice(
                        a_window,
                        (self.m, bk),
                        (0, k0),
                    ),
                    mx_scale=tile_slice(
                        scale_a,
                        (self.m, self.scale_l0_len),
                        (0, global_k0),
                    ),
                )
                l0b_write = self.l0b.produce()
                mem_copy(
                    l0b_write,
                    tile_slice(
                        b_fp8,
                        (bk, bn),
                        (k0, 0),
                    ),
                    transpose=True,
                    mx_scale=tile_slice(
                        scale_b,
                        (self.scale_l0_len, bn),
                        (global_k0, 0),
                    ),
                )
                final = global_k0 + 1 == DIM // bk
                matmul(
                    l0c_write,
                    self.l0a.consume(),
                    self.l0b.consume(),
                    init=(global_k0 == 0),
                    unit_flag=3 if final else 2,
                )
        mem_copy(
            tile_slice(qa_gm, (self.m, bn), (m_idx, block)),
            self.l0c.consume(),
            engine=self.fixpipe,
            unit_flag=3,
        )
        # The standalone ref17 kernel drains every owner after its only M tile.
        cube_sync_all()

    @jit
    def project_kva_tile(self, kva_gm, x_gm, wkva_gm, sx_gm, swkva_gm, block, m_idx):
        """Run one complete N64 tile using regular K1024 FIFO windows."""
        _bm, bn, bk, kl1 = (
            self.base_m,
            self.base_n,
            self.base_k,
            self.k_l1,
        )
        weight = _fp8_nz_transposed_view(wkva_gm, FEATURES, DIM)
        scale_a_write = self.l1_scale_a.produce()
        mem_copy(
            scale_a_write,
            tile_slice(sx_gm, (self.m, self.scale_groups, 2), (m_idx, 0, 0)),
            engine=self.copy_scale_a,
        )
        scale_b_write = self.l1_scale_b.produce()
        mem_copy(
            scale_b_write,
            tile_slice(
                swkva_gm,
                (bn, self.scale_groups, 2),
                (block, 0, 0),
            ),
            engine=self.copy_scale_b,
        )
        scale_a = self.l1_scale_a.consume().reinterpret(
            shape=(self.m, self.scale_l1_len)
        )
        scale_b = self.l1_scale_b.consume()
        l0c_write = self.l0c.produce()
        for k1 in range_constexpr(DIM // kl1):
            l1_a_write = self.l1_a.produce()
            mem_copy(
                l1_a_write,
                tile_slice(x_gm, (self.m, kl1), (m_idx, k1)),
                engine=self.copy_a,
            )
            l1_b_write = self.l1_b.produce()
            mem_copy(
                l1_b_write,
                tile_slice(
                    weight,
                    (kl1, bn),
                    (k1, block),
                ),
            )
            l1_a_read = self.l1_a.consume()
            l1_b_read = self.l1_b.consume()
            a_fp8 = l1_a_read.reinterpret(dtype=Float8E4M3FN)
            a_window = a_fp8.reinterpret(shape=(self.m, kl1))
            b_fp8 = l1_b_read.reinterpret(dtype=Float8E4M3FN)
            for k0 in range_constexpr(kl1 // bk):
                global_k0 = k1 * (kl1 // bk) + k0
                l0a_write = self.l0a.produce()
                mem_copy(
                    l0a_write,
                    tile_slice(
                        a_window,
                        (self.m, bk),
                        (0, k0),
                    ),
                    mx_scale=tile_slice(
                        scale_a,
                        (self.m, self.scale_l0_len),
                        (0, global_k0),
                    ),
                )
                l0b_write = self.l0b.produce()
                mem_copy(
                    l0b_write,
                    tile_slice(
                        b_fp8,
                        (bk, bn),
                        (k0, 0),
                    ),
                    transpose=True,
                    mx_scale=tile_slice(
                        scale_b,
                        (self.scale_l0_len, bn),
                        (global_k0, 0),
                    ),
                )
                final = global_k0 + 1 == DIM // bk
                matmul(
                    l0c_write,
                    self.l0a.consume(),
                    self.l0b.consume(),
                    init=(global_k0 == 0),
                    unit_flag=3 if final else 2,
                )
        mem_copy(
            tile_slice(kva_gm, (self.m, bn), (m_idx, block)),
            self.l0c.consume(),
            engine=self.fixpipe,
            unit_flag=3,
        )
        cube_sync_all()


class _O11QbHeadMixNoAlias:
    """David MXFP8 QcQr with resident A and BN128/BK128 tiles."""

    def __init__(self, m):
        self.m = m
        # Type-only declarations keep resource allocation inside the JIT body.
        self.l1_a: object
        self.l1_b: object
        self.l1_scale_a: object
        self.l1_scale_b: object
        self.padded_m = _workspace_token_capacity(m)
        # Match the reviewed QcQr split: M<=64 keeps a 64-row resident A;
        # T=128 uses one 128-row resident A.  Both variants remain inside the
        # same MixKernel and retain the BN128/BK128 K pipeline.
        self.base_m = 64 if m <= 64 else 128
        self.base_n, self.base_k = 128, 128
        self.l0a: object
        self.l0b: object
        self.l0c: object
        self.k_l1 = 512
        self.n_tiles = (N_HEADS * HEAD_DIM) // self.base_n
        self.scale_groups = Q_LORA // 64
        self.scale_l1_len = self.scale_groups * 2
        self.scale_l0_len = self.base_k // 32
        self.copy_a: object
        self.copy_scale_a: object
        self.copy_scale_b: object
        self.fixpipe: object

    @jit
    def projection_kernel(
        self,
        out_gm: Tensor,
        qa_gm: Tensor,
        wqb_gm: Tensor,
        sqa_gm: Tensor,
        swqb_gm: Tensor,
    ):
        _bm, bn, bk, kl1 = (self.base_m, self.base_n, self.base_k, self.k_l1)
        wqb = _fp8_nz_transposed_view(wqb_gm, N_HEADS * HEAD_DIM, Q_LORA)
        self.l1_a = Channel(
            MemLoc.L1,
            (self.m, Q_LORA),
            dtypes.int8,
            depth=1,
            data_format="nz",
        )
        self.l1_b = Channel(
            MemLoc.L1,
            (kl1, bn),
            dtypes.int8,
            depth=2,
            data_format="nz",
        )
        self.l1_scale_a = Channel(
            MemLoc.L1,
            (self.m, self.scale_l1_len),
            Float8E8M0,
            depth=1,
            data_format="zn",
        )
        self.l1_scale_b = Channel(
            MemLoc.L1,
            (self.scale_l1_len, bn),
            Float8E8M0,
            depth=2,
            data_format="nz",
        )
        self.l0a = Channel(MemLoc.L0A, (self.m, bk), Float8E4M3FN, depth=2)
        self.l0b = Channel(MemLoc.L0B, (bn, bk), Float8E4M3FN, depth=2)
        self.l0c = Channel(MemLoc.L0C, (self.m, bn), dtypes.float32, depth=2)
        self.copy_a = make_copy_engine(
            format_transform="nd2nz",
        )
        self.copy_scale_a = make_copy_engine(
            format_transform="mx_scale_and",
        )
        self.copy_scale_b = make_copy_engine(
            format_transform="mx_scale_bdn",
        )
        self.fixpipe = make_copy_engine()

        block = get_block_idx()
        out_token_major = out_gm.view((self.padded_m, N_HEADS * HEAD_DIM))

        # M<=64 path: A [64,1536] and its full-K scale remain resident in L1
        # while this core consumes all assigned N128 tiles.
        l1_a_write = self.l1_a.produce()
        mem_copy(
            l1_a_write,
            tile_slice(
                qa_gm.view(dtype=dtypes.int8),
                (self.m, Q_LORA),
                (0, 0),
            ),
            engine=self.copy_a,
        )
        scale_a_write = self.l1_scale_a.produce()
        mem_copy(
            scale_a_write,
            tile_slice(sqa_gm, (self.m, self.scale_groups, 2), (0, 0, 0)),
            engine=self.copy_scale_a,
        )
        l1_a_read = self.l1_a.consume()
        scale_a_read = self.l1_scale_a.consume()
        a_fp8 = l1_a_read.reinterpret(dtype=Float8E4M3FN)
        a_window = a_fp8.reinterpret(shape=(self.m, Q_LORA))
        scale_a = scale_a_read.reinterpret(shape=(self.m, self.scale_l1_len))
        # Assign complete N128 tiles round-robin.  This mapping is independent
        # of runtime branches and covers every output column exactly once.
        for n_idx in range(block, self.n_tiles, get_block_num()):
            scale_b_write = self.l1_scale_b.produce()
            mem_copy(
                scale_b_write,
                tile_slice(
                    swqb_gm,
                    (bn, self.scale_groups, 2),
                    (n_idx, 0, 0),
                ),
                engine=self.copy_scale_b,
            )
            scale_b_read = self.l1_scale_b.consume()
            l0c_write = self.l0c.produce()
            for k1 in range_constexpr(Q_LORA // kl1):
                l1_b_write = self.l1_b.produce()
                mem_copy(
                    l1_b_write,
                    tile_slice(
                        wqb,
                        (kl1, bn),
                        (k1, n_idx),
                    ),
                )
                l1_b_read = self.l1_b.consume()
                b_fp8 = l1_b_read.reinterpret(dtype=Float8E4M3FN)
                for k0 in range_constexpr(kl1 // bk):
                    global_k0 = k1 * (kl1 // bk) + k0
                    l0a_write = self.l0a.produce()
                    mem_copy(
                        l0a_write,
                        tile_slice(
                            a_window,
                            (self.m, bk),
                            (0, global_k0),
                        ),
                        mx_scale=tile_slice(
                            scale_a,
                            (self.m, self.scale_l0_len),
                            (0, global_k0),
                        ),
                    )
                    l0b_write = self.l0b.produce()
                    mem_copy(
                        l0b_write,
                        tile_slice(
                            b_fp8,
                            (bk, bn),
                            (k0, 0),
                        ),
                        transpose=True,
                        mx_scale=tile_slice(
                            scale_b_read,
                            (self.scale_l0_len, bn),
                            (global_k0, 0),
                        ),
                    )
                    final = global_k0 + 1 == Q_LORA // bk
                    matmul(
                        l0c_write,
                        self.l0a.consume(),
                        self.l0b.consume(),
                        init=(global_k0 == 0),
                        unit_flag=3 if final else 2,
                    )

            mem_copy(
                tile_slice(out_token_major, (self.m, bn), (0, n_idx)),
                self.l0c.consume(),
                engine=self.fixpipe,
                unit_flag=3,
            )


class _O11QbHeadOwnedMixT128:
    """T128 QcQr split by complete heads for same-core WKB handoff."""

    def __init__(self, m, core_num):
        self.m = m
        # Type-only declarations keep resource allocation inside the JIT body.
        self.l0a: object
        self.l0b: object
        self.l0c: object
        self.core_num = core_num
        self.l1_a: object
        self.l1_b: object
        self.l1_scale_a: object
        self.l1_scale_b: object
        self.max_heads = (N_HEADS + core_num - 1) // core_num
        self.copy_a: object
        self.copy_scale_a: object
        self.copy_scale_b: object
        self.fixpipe: object
        self.padded_m = _workspace_token_capacity(m)
        self.base_m = 128
        self.base_n, self.base_k = HEAD_DIM, 128
        self.k_l1 = 512
        self.scale_groups = Q_LORA // 64
        self.scale_l1_len = self.scale_groups * 2
        self.scale_l0_len = self.base_k // 32

    @jit
    def projection_kernel(
        self,
        out_gm: Tensor,
        qa_gm: Tensor,
        wqb_gm: Tensor,
        sqa_gm: Tensor,
        swqb_gm: Tensor,
    ):
        bm, bn, bk, kl1 = (self.base_m, self.base_n, self.base_k, self.k_l1)
        wqb = _fp8_nz_transposed_view(wqb_gm, N_HEADS * HEAD_DIM, Q_LORA)
        self.l1_a = Channel(
            MemLoc.L1,
            (bm, Q_LORA),
            dtypes.int8,
            depth=1,
            data_format="nz",
        )
        self.l1_b = Channel(
            MemLoc.L1,
            (kl1, bn),
            dtypes.int8,
            depth=2,
            data_format="nz",
        )
        self.l1_scale_a = Channel(
            MemLoc.L1,
            (bm, self.scale_l1_len),
            Float8E8M0,
            depth=1,
            data_format="zn",
        )
        self.l1_scale_b = Channel(
            MemLoc.L1,
            (self.scale_l1_len, bn),
            Float8E8M0,
            depth=2,
            data_format="nz",
        )
        self.l0a = Channel(MemLoc.L0A, (bm, bk), Float8E4M3FN, depth=2)
        self.l0b = Channel(
            # The transpose load keeps logical [K, N] axes in its descriptor.
            MemLoc.L0B,
            (bk, bn),
            Float8E4M3FN,
            depth=2,
        )
        self.l0c = Channel(MemLoc.L0C, (bm, bn), dtypes.float32, depth=2)
        self.copy_a = make_copy_engine(
            format_transform="nd2nz",
        )
        self.copy_scale_a = make_copy_engine(
            format_transform="mx_scale_and",
        )
        self.copy_scale_b = make_copy_engine(
            format_transform="mx_scale_bdn",
        )
        self.fixpipe = make_copy_engine()

        block = get_block_idx()
        out_token_major = out_gm.view((self.padded_m, N_HEADS * HEAD_DIM))

        # T128 keeps A [128,1536] and its full-K scale resident while this
        # AIC produces the three complete heads consumed by its own WKB path.
        l1_a_write = self.l1_a.produce()
        mem_copy(
            l1_a_write,
            tile_slice(
                qa_gm.view(dtype=dtypes.int8),
                (self.m, Q_LORA),
                (0, 0),
            ),
            engine=self.copy_a,
        )
        scale_a_write = self.l1_scale_a.produce()
        mem_copy(
            scale_a_write,
            tile_slice(sqa_gm, (self.m, self.scale_groups, 2), (0, 0, 0)),
            engine=self.copy_scale_a,
        )
        l1_a_read = self.l1_a.consume()
        scale_a_read = self.l1_scale_a.consume()
        a_fp8 = l1_a_read.reinterpret(dtype=Float8E4M3FN)
        a_window = a_fp8.reinterpret(shape=(self.m, Q_LORA))
        scale_a = scale_a_read.reinterpret(shape=(self.m, self.scale_l1_len))
        # Core b owns complete heads b + i*core_num.  The core count is a
        # launcher specialization, so the hot loop has no runtime division.
        for local_head in range_constexpr(self.max_heads):
            n_idx = block + local_head * self.core_num
            # Keep the 8/16/24/32-core specializations branch-free.  Only
            # 28 cores has a partial final head group; heads inside the even
            # quotient are statically in range and run unguarded, so only
            # that final group takes a runtime bounds guard.  Guarding every
            # unrolled head instead carries channel slot rotation state
            # across the scf.if boundary, which defeats the L0B (K, N) view
            # inference on the head-owned MXFP8 matmuls and also trips the
            # CANN 9.2 Bisheng frontend on the large body.
            if n_idx < N_HEADS:
                scale_b_write = self.l1_scale_b.produce()
                mem_copy(
                    scale_b_write,
                    tile_slice(
                        swqb_gm,
                        (bn, self.scale_groups, 2),
                        (n_idx, 0, 0),
                    ),
                    engine=self.copy_scale_b,
                )
                scale_b_read = self.l1_scale_b.consume()
                l0c_write = self.l0c.produce()
                for k1 in range_constexpr(Q_LORA // kl1):
                    l1_b_write = self.l1_b.produce()
                    mem_copy(
                        l1_b_write,
                        tile_slice(
                            wqb,
                            (kl1, bn),
                            (k1, n_idx),
                        ),
                    )
                    l1_b_read = self.l1_b.consume()
                    b_fp8 = l1_b_read.reinterpret(dtype=Float8E4M3FN)
                    for k0 in range_constexpr(kl1 // bk):
                        global_k0 = k1 * (kl1 // bk) + k0
                        l0a_write = self.l0a.produce()
                        mem_copy(
                            l0a_write,
                            tile_slice(
                                a_window,
                                (self.m, bk),
                                (0, global_k0),
                            ),
                            mx_scale=tile_slice(
                                scale_a,
                                (self.m, self.scale_l0_len),
                                (0, global_k0),
                            ),
                        )
                        l0b_write = self.l0b.produce()
                        mem_copy(
                            l0b_write,
                            tile_slice(
                                b_fp8,
                                (bk, bn),
                                (k0, 0),
                            ),
                            transpose=True,
                            mx_scale=tile_slice(
                                scale_b_read,
                                (self.scale_l0_len, bn),
                                (global_k0, 0),
                            ),
                        )
                        final = global_k0 + 1 == Q_LORA // bk
                        matmul(
                            l0c_write.reinterpret(shape=(bm, bn)),
                            self.l0a.consume().reinterpret(shape=(bm, bk)),
                            self.l0b.consume().reinterpret(shape=(bk, bn)),
                            init=(global_k0 == 0),
                            unit_flag=3 if final else 2,
                        )

                mem_copy(
                    tile_slice(out_token_major, (self.m, bn), (0, n_idx)),
                    self.l0c.consume(),
                    engine=self.fixpipe,
                    unit_flag=3,
                )


class _O11MlaQHeadLocalEpilogUnit:
    """Quantize one T8 head as one four-token half per AIV subblock."""

    def __init__(self, padded_t, logical_t):
        self.padded_t = padded_t
        self.logical_t = logical_t
        self.nope_channel = Channel(MemLoc.UB, (4, KV_LORA), dtypes.bfloat16, depth=2)
        self.rope_channel = Channel(MemLoc.UB, (4, ROPE), dtypes.bfloat16, depth=2)
        self.output_channel = Channel(MemLoc.UB, (4, FEATURES), Float8E4M3FN, depth=2)
        self.scale_channel = Channel(MemLoc.UB, (4, 1), dtypes.float32, depth=2)
        self.scalar_ub = Buffer(MemLoc.UB, (4, 8), dtypes.float32)

    @jit
    def process_batch(self, q_nope, q_combined, q_out, descale_q, head, token_batch):
        nope = q_nope.view((N_HEADS * self.padded_t, KV_LORA))
        combined = q_combined.view((self.padded_t, N_HEADS * HEAD_DIM))
        output = q_out.view((self.logical_t, N_HEADS * FEATURES))
        scales_out = descale_q.view((self.logical_t, N_HEADS))
        source_batch = head * (self.padded_t // 4) + token_batch
        nope_slot = self.nope_channel.produce()
        mem_copy(
            nope_slot,
            tile_slice(nope, (4, KV_LORA), (source_batch, 0)),
        )
        rope_slot = self.rope_channel.produce()
        # fmt: off
        mem_copy(
            rope_slot,
            combined[
                token_batch * 4:token_batch * 4 + 4,
                head * HEAD_DIM + QK_NOPE:(head + 1) * HEAD_DIM,
            ],
        )
        # fmt: on
        q_nope_rows = self.nope_channel.consume()
        q_rope_rows = self.rope_channel.consume()
        encoded = self.output_channel.produce()
        scales = self.scale_channel.produce()
        with vf(mode="simd"):
            vmem_bar(mode="vst_vld")
            full = full_mask()
            scalar = update_mask(1, elem_bits=32)[0]
            for row in range_constexpr(4):
                absmax = vdups(1.0e-8, dtypes.float32, mask=scalar)
                for segment in range_constexpr(KV_LORA // 64):
                    unpacked = vload_unpack(
                        q_nope_rows,
                        row * KV_LORA + segment * 64,
                        unpack_mode=UnpackMode.B16_TO_B32,
                    )
                    value = vcast(unpacked, dtypes.float32, mask=full)
                    absmax = vmax(
                        absmax,
                        vreduce_max(vabs(value, mask=full), mask=full),
                        mask=scalar,
                    )
                unpacked = vload_unpack(
                    q_rope_rows,
                    row * ROPE,
                    unpack_mode=UnpackMode.B16_TO_B32,
                )
                value = vcast(unpacked, dtypes.float32, mask=full)
                absmax = vmax(
                    absmax,
                    vreduce_max(vabs(value, mask=full), mask=full),
                    mask=scalar,
                )
                scale = vmuls(absmax, 1.0 / FP8_MAX, mask=scalar)
                vstore_first(self.scalar_ub, row * 8, scale)
                vstore_first(scales, row, scale)
                vmem_bar(mode="vst_vld")
                scale_vector = vload_broadcast(self.scalar_ub, row * 8)
                for segment in range_constexpr(KV_LORA // 64):
                    unpacked = vload_unpack(
                        q_nope_rows,
                        row * KV_LORA + segment * 64,
                        unpack_mode=UnpackMode.B16_TO_B32,
                    )
                    value = vcast(unpacked, dtypes.float32, mask=full)
                    quantized = vdiv(value, scale_vector, mask=full)
                    quantized = vmaxs(quantized, -FP8_MAX, mask=full)
                    quantized = vmins(quantized, FP8_MAX, mask=full)
                    fp8_value = vcast(
                        quantized,
                        dtypes.float8_e4m3fn,
                        mask=full,
                        reg_layout=RegLayout.ZERO,
                    )
                    vstore_pack(
                        encoded,
                        row * FEATURES + segment * 64,
                        fp8_value,
                        full,
                        pack_mode=PackMode.B32_TO_B8,
                    )
                unpacked = vload_unpack(
                    q_rope_rows,
                    row * ROPE,
                    unpack_mode=UnpackMode.B16_TO_B32,
                )
                value = vcast(unpacked, dtypes.float32, mask=full)
                quantized = vdiv(value, scale_vector, mask=full)
                quantized = vmaxs(quantized, -FP8_MAX, mask=full)
                quantized = vmins(quantized, FP8_MAX, mask=full)
                fp8_value = vcast(
                    quantized,
                    dtypes.float8_e4m3fn,
                    mask=full,
                    reg_layout=RegLayout.ZERO,
                )
                vstore_pack(
                    encoded,
                    row * FEATURES + KV_LORA,
                    fp8_value,
                    full,
                    pack_mode=PackMode.B32_TO_B8,
                )
        output_rows = self.output_channel.consume()
        output_scales = self.scale_channel.consume()
        mem_copy(
            tile_slice(output, (4, FEATURES), (token_batch, head)),
            output_rows,
        )
        mem_copy(
            tile_slice(scales_out, (4, 1), (token_batch, head)),
            output_scales,
        )


class _O11MlaWkbTokenMajorUnit:
    """Read one head's NOPE prefix from token-major QcQr workspace."""

    def __init__(self, padded_m, cdepth=1, bdepth=1):
        self.cdepth = cdepth
        self.padded_m = padded_m
        self.bm = 16 if padded_m == 16 else 32
        self.m_tiles = padded_m // self.bm
        self.a_l1 = Channel(
            MemLoc.L1, (self.bm, QK_NOPE), dtypes.bfloat16, depth=2, data_format="nz"
        )
        self.b_l1 = Channel(
            MemLoc.L1,
            (QK_NOPE, KV_LORA),
            dtypes.bfloat16,
            depth=1 if padded_m == 128 else 2,
            data_format="nz",
        )
        self.a_l0 = Channel(MemLoc.L0A, (self.bm, QK_NOPE), dtypes.bfloat16, depth=2)
        self.b_l0 = Channel(MemLoc.L0B, (QK_NOPE, 128), dtypes.bfloat16, depth=bdepth)
        self.c_l0 = Channel(
            MemLoc.L0C, (self.bm, 128), dtypes.float32, depth=self.cdepth
        )
        self.nd2nz = make_copy_engine(format_transform="nd2nz")
        self.fixpipe = make_copy_engine()

    @jit
    def process_head(self, q_combined, wkb, q_latent, head):
        q_token_major = q_combined.view((self.padded_m, N_HEADS * HEAD_DIM))
        q_col_start = head * HEAD_DIM
        w_flat = wkb.view((N_HEADS * QK_NOPE, KV_LORA))
        out_flat = q_latent.view((N_HEADS * self.padded_m, KV_LORA))
        out_head = tile_slice(out_flat, (self.padded_m, KV_LORA), (head, 0))
        # For T<=64, two L1B slots let the next head's ND->NZ weight transfer
        # run while the current head consumes its four N128 tiles on AIC.
        # T=128 keeps one slot: the extra 128 KiB slot makes its resident
        # M128 path slower through L1 pressure.
        b_l1_write = self.b_l1.produce()
        mem_copy(
            b_l1_write,
            tile_slice(w_flat, (QK_NOPE, KV_LORA), (head, 0)),
            engine=self.nd2nz,
        )
        b_l1_read = self.b_l1.consume()
        for m_tile in range_constexpr(self.m_tiles):
            m_start = m_tile * self.bm
            a_l1_write = self.a_l1.produce()
            # fmt: off
            mem_copy(
                a_l1_write,
                q_token_major[
                    m_start:m_start + self.bm,
                    q_col_start:q_col_start + QK_NOPE,
                ],
                engine=self.nd2nz,
            )
            # fmt: on
            # A is invariant across the four 128-column WKB tiles.  Keep one
            # L0A slot live until every tile has consumed it instead of
            # issuing the same L1 -> L0A transfer four times.
            a_l0_write = self.a_l0.produce()
            mem_copy(a_l0_write, self.a_l1.consume())
            a_l0_read = self.a_l0.consume()
            for n_tile in range_constexpr(4):
                b_l0_write = self.b_l0.produce()
                mem_copy(
                    b_l0_write,
                    tile_slice(b_l1_read, (QK_NOPE, 128), (0, n_tile)),
                    transpose=True,
                )
                c_l0_write = self.c_l0.produce()
                matmul(c_l0_write, a_l0_read, self.b_l0.consume(), init=True)
                mem_copy(
                    tile_slice(out_head, (self.bm, 128), (m_tile, n_tile)),
                    self.c_l0.consume(),
                    engine=self.fixpipe,
                )


class _MlaInputTailUnit:
    def __init__(self):
        self.x = Buffer(MemLoc.UB, (1, DIM), dtypes.int8)
        self.scale = Buffer(MemLoc.UB, (1, DIM // 32), dtypes.int8)

    @jit
    def pad(self, x, sx, x_pad, sx_pad, valid, tile_m):
        sx_rows = sx.view((sx.shape[0], DIM // 32))
        dst_scales = sx_pad.view((tile_m, DIM // 32))
        for row in range(
            get_block_idx() * 2 + get_subblock_id(), tile_m, get_block_num() * 2
        ):
            if row < valid:
                mem_copy(self.x, tile_slice(x, (1, DIM), (row, 0)))
                mem_copy(self.scale, tile_slice(sx_rows, (1, DIM // 32), (row, 0)))
            else:
                with vf(mode="simd"):
                    full = full_mask()
                    zero = vdups(0, dtypes.int32, mask=full)
                    unity_e8m0 = vdups(127, dtypes.int32, mask=full)
                    for segment in range_constexpr(DIM // 64):
                        vstore_pack(
                            self.x,
                            segment * 64,
                            zero,
                            full,
                            pack_mode=PackMode.B32_TO_B8,
                        )
                    for segment in range_constexpr((DIM // 32 + 63) // 64):
                        mask = update_mask(
                            min(64, DIM // 32 - segment * 64), elem_bits=32
                        )[0]
                        vstore_pack(
                            self.scale,
                            segment * 64,
                            unity_e8m0,
                            mask,
                            pack_mode=PackMode.B32_TO_B8,
                        )
            mem_copy(tile_slice(x_pad, (1, DIM), (row, 0)), self.x)
            mem_copy(tile_slice(dst_scales, (1, DIM // 32), (row, 0)), self.scale)


class _MlaRuntimeTileUnit:
    """C0/C1 output specialization; each invocation is one MixKernel."""

    def __init__(self, m, page_size, norm_eps, quant_mode_c):
        self.m = m
        self.padded_m = _workspace_token_capacity(m)
        self.output_m = _output_token_capacity(m)
        self.page_size = page_size
        self.norm_eps = norm_eps
        self.quant_mode_c = quant_mode_c
        self.qa = _QaRmsMixUnit(norm_eps)
        # The public CANNBotDSL API uses deterministic FFTS flag IDs.
        self.front_ready = 0
        self.qa_ready = 1
        self.tail_done = 2
        self.head_ready = 3

    @jit
    def run(
        self,
        qa_raw: Tensor,
        kva_raw: Tensor,
        x: Tensor,
        wqa: Tensor,
        wkva: Tensor,
        sx: Tensor,
        swqa: Tensor,
        swkva: Tensor,
        gamma_qa: Tensor,
        wqb: Tensor,
        swqb: Tensor,
        wkb: Tensor,
        gamma_kv: Tensor,
        qscale_kv: Tensor,
        cache: Tensor,
        cache_index: Tensor,
        qa_mx: Tensor,
        qa_scale: Tensor,
        q_head: Tensor,
        q_latent_h: Tensor,
        q_out: Tensor,
        descale_q: Tensor,
    ):
        block = get_block_idx()
        core_num = get_block_num()
        sub = get_subblock_id()

        # Probe path: preserve the validated QA -> KVA phase ordering while
        # distributing every N64 tile by the actual launch block count.
        qa_tiles = Q_LORA // 64
        if core_num >= qa_tiles:
            # At 24/28/32 cores each active AIC owns exactly one N64 tile.
            # Express this as a guard rather than a single-iteration runtime
            # loop: the CANN 9.2 Bisheng frontend crashes while lowering the
            # large unrolled MXFP8 body when it is nested inside that loop.
            if block < qa_tiles:
                front = _O11MlaRef17FrontUnit(self.m)
                front.project_qa_owned_tile(
                    qa_raw,
                    x,
                    wqa,
                    sx,
                    swqa,
                    block,
                    0,
                )
        else:
            front = _O11MlaRef17FrontUnit(self.m)
            for qa_tile in range(block, qa_tiles, core_num):
                front.project_qa_owned_tile(
                    qa_raw,
                    x,
                    wqa,
                    sx,
                    swqa,
                    qa_tile,
                    0,
                )
        # No AIV work precedes the first front handoff.  T<=64 can publish
        # AIC readiness without an empty vector-pipeline drain.  T=128 keeps
        # the drain as a useful pacing point for its bandwidth-heavy path.
        _o11_front_sync_aic_to_aiv(self.m == 128)
        # QA has fully drained its L1/L0 channel epoch.  KVA reuses the same
        # physical arena with the same regular K1024 FIFO structure.
        channel_rewind(reset_sync_id=True)
        kva_tiles = FEATURES // 64
        if core_num >= kva_tiles:
            if block < kva_tiles:
                front = _O11MlaRef17FrontUnit(self.m)
                front.project_kva_tile(
                    kva_raw,
                    x,
                    wkva,
                    sx,
                    swkva,
                    block,
                    0,
                )
        else:
            front = _O11MlaRef17FrontUnit(self.m)
            for kva_tile in range(block, kva_tiles, core_num):
                front.project_kva_tile(
                    kva_raw,
                    x,
                    wkva,
                    sx,
                    swkva,
                    kva_tile,
                    0,
                )

        # Phase Q: both AIV subblocks share contiguous token ownership.
        qa = self.qa
        qa_token = block * 2 + sub
        if qa_token < x.shape[0]:
            gamma_slot = qa.gamma.produce()
            mem_copy(gamma_slot, gamma_qa)
            gamma = qa.gamma.consume()
            for token in range(qa_token, x.shape[0], 2 * core_num):
                qa.process_one(qa_raw, gamma, qa_mx, qa_scale, token)

        _o11_front_sync_aic_to_aiv()
        # Front AIC and QA AIV have fully drained.  Reuse their L1/channel
        # arena for QB instead of statically retaining both phase footprints.
        channel_rewind(reset_sync_id=True)

        # Phase H1: QB and KV start together.  WKB only consumes q_head, so
        # it can follow QB while the independent KV vector tail is still
        # normalizing, quantizing and scattering cache rows.
        qb = _O11QbHeadMixNoAlias(self.m)
        if const_expr(self.quant_mode_c == 1):
            kv = _MlaKvEpilogMxfp8Unit(self.page_size, self.norm_eps, output_depth=2)
            if sub == 0:
                if block < x.shape[0]:
                    kv.process_assigned_rows(
                        kva_raw,
                        gamma_kv,
                        qscale_kv,
                        cache,
                        cache_index,
                        block,
                        core_num,
                    )
        else:
            kv_bf16 = _MlaKvEpilogBf16Unit(
                self.page_size, self.norm_eps, output_depth=2
            )
            if sub == 0:
                if block < x.shape[0]:
                    kv_bf16.process_assigned_rows(
                        kva_raw,
                        gamma_kv,
                        cache,
                        cache_index,
                        block,
                        core_num,
                    )
        qb.projection_kernel(q_head, qa_mx, wqb, qa_scale, swqb)
        # QB assigns N128 tiles whereas WKB assigns full N192 heads.
        # A consumer head can span tiles written by other AICs.  sync_all()
        # only drains this AIC, so publish completion across all AICs before
        # WKB (and the paired Q epilog) reads the shared q_head workspace.
        # This does not join the independent KV vector tail.
        cube_sync_all()
        cube_sync_block_arrive(_sync_ops.PIPE.FIXPIPE, 1, mode=0)
        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, 1, mode=0)
        if const_expr(self.m == 128):
            cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
            cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
        # Q channels reuse the KV UB arena. Source order and AIC head-ready
        # do not complete this AIV's asynchronous VECTOR/MTE3 accesses.
        # Drain this AIV before reusing its UB; AIC WKB remains independent.
        vec_sync_all()
        channel_rewind(reset_sync_id=True)

        if core_num < 8:
            vec_sync_block_wait(_sync_ops.PIPE.S, self.head_ready, mode=2)

        if const_expr(self.quant_mode_c == 1):
            if self.m == 8:
                q_token4 = _O11MlaQHeadLocalEpilogUnit(self.padded_m, self.m)
                for head in range(block, N_HEADS, core_num):
                    if core_num >= 8:
                        vec_sync_block_wait(
                            _sync_ops.PIPE.S,
                            self.head_ready + head // core_num,
                            mode=2,
                        )
                    q_token4.process_batch(
                        q_latent_h, q_head, q_out, descale_q, head, sub
                    )
            else:
                q_token = _MlaQHeadTokenEpilogMxfp8Unit(
                    64, self.padded_m, self.output_m, token_major_rope=True
                )
                nope = q_latent_h.view((N_HEADS * self.padded_m, KV_LORA))
                combined = q_head.view((self.padded_m, N_HEADS * HEAD_DIM))
                output = q_out.view((self.output_m, N_HEADS * FEATURES))
                for head in range(block, N_HEADS, core_num):
                    if core_num >= 8:
                        vec_sync_block_wait(
                            _sync_ops.PIPE.S,
                            self.head_ready + head // core_num,
                            mode=2,
                        )
                    batch_count = self.output_m // 8
                    if const_expr(batch_count % 2 == 0):
                        batches_per_sub = batch_count // 2
                        for local_batch in range_constexpr(batches_per_sub):
                            token_batch = sub * batches_per_sub + local_batch
                            batch = head * batch_count + token_batch
                            q_token.process_batch(
                                nope,
                                combined,
                                output,
                                descale_q,
                                batch,
                            )
                    else:
                        batches_per_sub = (batch_count + 1) // 2
                        for local_batch in range_constexpr(batches_per_sub):
                            token_batch = sub * batches_per_sub + local_batch
                            if token_batch < batch_count:
                                batch = head * batch_count + token_batch
                                q_token.process_batch(
                                    nope,
                                    combined,
                                    output,
                                    descale_q,
                                    batch,
                                )
        else:
            q_bf16 = _MlaQHeadTokenEpilogBf16Unit(
                4 if self.m == 8 else 8,
                self.padded_m,
                self.output_m,
            )
            for head in range(block, N_HEADS, core_num):
                if core_num >= 8:
                    vec_sync_block_wait(
                        _sync_ops.PIPE.S,
                        self.head_ready + head // core_num,
                        mode=2,
                    )
                if const_expr(self.m == 8):
                    q_bf16.process_batch(q_latent_h, q_head, q_out, head, sub)
                else:
                    batch_count = self.output_m // 8
                    if const_expr(batch_count % 2 == 0):
                        batches_per_sub = batch_count // 2
                        for local_batch in range_constexpr(batches_per_sub):
                            token_batch = sub * batches_per_sub + local_batch
                            q_bf16.process_batch(
                                q_latent_h,
                                q_head,
                                q_out,
                                head,
                                token_batch,
                            )
                    else:
                        batches_per_sub = (batch_count + 1) // 2
                        for local_batch in range_constexpr(batches_per_sub):
                            token_batch = sub * batches_per_sub + local_batch
                            if token_batch < batch_count:
                                q_bf16.process_batch(
                                    q_latent_h,
                                    q_head,
                                    q_out,
                                    head,
                                    token_batch,
                                )
        vec_sync_all()
        vec_sync_block_arrive(_sync_ops.PIPE.MTE3, self.tail_done, mode=2)
        wkb_unit = _O11MlaWkbTokenMajorUnit(self.padded_m, cdepth=2, bdepth=2)
        for head in range(block, N_HEADS, core_num):
            wkb_unit.process_head(q_head, wkb, q_latent_h, head)
            if core_num >= 8:
                cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                cube_sync_block_arrive(
                    _sync_ops.PIPE.MTE2,
                    self.head_ready + head // core_num,
                    mode=2,
                )
        if core_num < 8:
            cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
            cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
            cube_sync_block_arrive(_sync_ops.PIPE.MTE2, self.head_ready, mode=2)
        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, self.tail_done, mode=2)
        global_sync_all()


@kernel
class _MlaRuntimeKernel:
    """One launch, runtime token count/core count, bounded on-chip tiles."""

    def __init__(self, tile_m, norm_eps, quant_mode_c):
        self.tile_m = tile_m
        self.quant_mode_c = quant_mode_c
        self.tail = _MlaInputTailUnit()
        self.unit = _MlaRuntimeTileUnit(tile_m, PA_BLOCK_SIZE, norm_eps, quant_mode_c)

    def __call__(
        self,
        qa_raw: Tensor,
        kva_raw: Tensor,
        x: Tensor,
        wqa: Tensor,
        wkva: Tensor,
        sx: Tensor,
        swqa: Tensor,
        swkva: Tensor,
        gamma_qa: Tensor,
        wqb: Tensor,
        swqb: Tensor,
        wkb: Tensor,
        gamma_kv: Tensor,
        qscale_kv: Tensor,
        cache: Tensor,
        cache_index: Tensor,
        qa_mx: Tensor,
        qa_scale: Tensor,
        q_head: Tensor,
        q_latent_h: Tensor,
        q_out: Tensor,
        descale_q: Tensor,
        x_pad: Tensor,
        sx_pad: Tensor,
    ):
        valid = x.shape[0]
        x_tile = x
        sx_tile = sx
        x_block = x_pad
        sx_block = sx_pad
        if valid < self.tile_m:
            self.tail.pad(x_tile, sx_tile, x_pad, sx_pad, valid, self.tile_m)
            global_sync_all()
            x_block = x_pad
            sx_block = sx_pad
        else:
            x_block = x_tile.view((self.tile_m, DIM))
            sx_block = sx_tile.view((self.tile_m, DIM // 64, 2))
        self.unit.run(
            qa_raw,
            kva_raw,
            x_block,
            wqa,
            wkva,
            sx_block,
            swqa,
            swkva,
            gamma_qa,
            wqb,
            swqb,
            wkb,
            gamma_kv,
            qscale_kv,
            cache,
            cache_index,
            qa_mx,
            qa_scale,
            q_head,
            q_latent_h,
            q_out,
            descale_q,
        )


class _MlaRuntimeLauncher:
    def __init__(self, tile_m, norm_eps, quant_mode_c):
        self.tile_m, self.norm_eps, self.quant_mode_c = tile_m, norm_eps, quant_mode_c

    @host
    def run(
        self,
        qa_raw,
        kva_raw,
        x,
        wqa,
        wkva,
        sx,
        swqa,
        swkva,
        gamma_qa,
        wqb,
        swqb,
        wkb,
        gamma_kv,
        qscale_kv,
        cache,
        cache_index,
        qa_mx,
        qa_scale,
        q_head,
        q_latent_h,
        q_out,
        descale_q,
        x_pad,
        sx_pad,
        core_num: int,
    ):
        _MlaRuntimeKernel(self.tile_m, self.norm_eps, self.quant_mode_c)[core_num](
            qa_raw,
            kva_raw,
            x,
            wqa,
            wkva,
            sx,
            swqa,
            swkva,
            gamma_qa,
            wqb,
            swqb,
            wkb,
            gamma_kv,
            qscale_kv,
            cache,
            cache_index,
            qa_mx,
            qa_scale,
            q_head,
            q_latent_h,
            q_out,
            descale_q,
            x_pad,
            sx_pad,
        )


class _O11MlaWkbTokenMajorLargeUnit:
    """Read one head's NOPE prefix from token-major QcQr workspace."""

    def __init__(self, cdepth=1, bdepth=1):
        self.cdepth = cdepth
        self.bm = 32
        self.a_l1 = Channel(
            MemLoc.L1, (self.bm, QK_NOPE), dtypes.bfloat16, depth=2, data_format="nz"
        )
        self.b_l1 = Channel(
            MemLoc.L1, (QK_NOPE, KV_LORA), dtypes.bfloat16, depth=1, data_format="nz"
        )
        self.a_l0 = Channel(MemLoc.L0A, (self.bm, QK_NOPE), dtypes.bfloat16, depth=2)
        self.b_l0 = Channel(MemLoc.L0B, (QK_NOPE, 128), dtypes.bfloat16, depth=bdepth)
        self.c_l0 = Channel(
            MemLoc.L0C, (self.bm, 128), dtypes.float32, depth=self.cdepth
        )
        self.nd2nz = make_copy_engine(format_transform="nd2nz")
        self.fixpipe = make_copy_engine()

    @jit
    def process_head(self, q_combined, wkb, q_latent, head):
        padded_m = q_combined.shape[0]
        m_tiles = padded_m // self.bm
        q_token_major = q_combined.view((padded_m, N_HEADS * HEAD_DIM))
        q_col_start = head * HEAD_DIM
        w_flat = wkb.view((N_HEADS * QK_NOPE, KV_LORA))
        out_flat = q_latent.view((N_HEADS * padded_m, KV_LORA))
        out_head = tile_slice(
            out_flat, make_tiler((padded_m, KV_LORA), alignment=(128, 1)), (head, 0)
        )
        # For T<=64, two L1B slots let the next head's ND->NZ weight transfer
        # run while the current head consumes its four N128 tiles on AIC.
        # T=128 keeps one slot: the extra 128 KiB slot makes its resident
        # M128 path slower through L1 pressure.
        b_l1_write = self.b_l1.produce()
        mem_copy(
            b_l1_write,
            tile_slice(w_flat, (QK_NOPE, KV_LORA), (head, 0)),
            engine=self.nd2nz,
        )
        b_l1_read = self.b_l1.consume()
        for m_tile in range(m_tiles):
            m_start = m_tile * self.bm
            a_l1_write = self.a_l1.produce()
            # fmt: off
            mem_copy(
                a_l1_write,
                q_token_major[
                    m_start:m_start + self.bm,
                    q_col_start:q_col_start + QK_NOPE,
                ],
                engine=self.nd2nz,
            )
            # fmt: on
            a_l1_read = self.a_l1.consume()
            # A is invariant across the four 128-column WKB tiles.  Keep one
            # L0A slot live until every tile has consumed it instead of
            # issuing the same L1 -> L0A transfer four times.
            a_l0_write = self.a_l0.produce()
            mem_copy(a_l0_write, a_l1_read)
            a_l0_read = self.a_l0.consume()
            for n_tile in range_constexpr(4):
                b_l0_write = self.b_l0.produce()
                mem_copy(
                    b_l0_write,
                    tile_slice(b_l1_read, (QK_NOPE, 128), (0, n_tile)),
                    transpose=True,
                )
                c_l0_write = self.c_l0.produce()
                matmul(c_l0_write, a_l0_read, self.b_l0.consume(), init=True)
                mem_copy(
                    tile_slice(out_head, (self.bm, 128), (m_tile, n_tile)),
                    self.c_l0.consume(),
                    engine=self.fixpipe,
                )


class _MlaPrefillQbUnit:
    """David MXFP8 QcQr with resident A and BN128/BK128 tiles."""

    def __init__(self, m):
        self.m = m
        # Type-only declarations keep resource allocation inside the JIT body.
        self.copy_a: object
        self.copy_scale_a: object
        self.copy_scale_b: object
        self.fixpipe: object
        self.padded_m = _workspace_token_capacity(m)
        self.l1_a: object
        self.l1_b: object
        self.l1_scale_a: object
        self.l1_scale_b: object
        # Match the reviewed QcQr split: M<=64 keeps a 64-row resident A;
        # T=128 uses one 128-row resident A.  Both variants remain inside the
        # same MixKernel and retain the BN128/BK128 K pipeline.
        self.base_m = 64 if m <= 64 else 128
        self.l0a: object
        self.l0b: object
        self.l0c: object
        self.base_n, self.base_k = 128, 128
        self.k_l1 = 512
        self.n_tiles = (N_HEADS * HEAD_DIM) // self.base_n
        self.scale_groups = Q_LORA // 64
        self.scale_l1_len = self.scale_groups * 2
        self.scale_l0_len = self.base_k // 32

    @jit
    def projection_kernel(
        self,
        out_gm: Tensor,
        qa_gm: Tensor,
        wqb_gm: Tensor,
        sqa_gm: Tensor,
        swqb_gm: Tensor,
    ):
        _bm, bn, bk, kl1 = (self.base_m, self.base_n, self.base_k, self.k_l1)
        wqb = _fp8_nz_transposed_view(wqb_gm, N_HEADS * HEAD_DIM, Q_LORA)
        self.l1_a = Channel(
            MemLoc.L1,
            (self.m, Q_LORA),
            dtypes.int8,
            depth=1,
            data_format="nz",
        )
        self.l1_b = Channel(
            MemLoc.L1,
            (kl1, bn),
            dtypes.int8,
            depth=2,
            data_format="nz",
        )
        self.l1_scale_a = Channel(
            MemLoc.L1,
            (self.m, self.scale_l1_len),
            Float8E8M0,
            depth=1,
            data_format="zn",
        )
        self.l1_scale_b = Channel(
            MemLoc.L1,
            (self.scale_l1_len, bn),
            Float8E8M0,
            depth=2,
            data_format="nz",
        )
        self.l0a = Channel(MemLoc.L0A, (self.m, bk), Float8E4M3FN, depth=2)
        self.l0b = Channel(MemLoc.L0B, (bk, bn), Float8E4M3FN, depth=2)
        self.l0c = Channel(MemLoc.L0C, (self.m, bn), dtypes.float32, depth=2)
        self.copy_a = make_copy_engine(
            format_transform="nd2nz",
        )
        self.copy_scale_a = make_copy_engine(
            format_transform="mx_scale_and",
        )
        self.copy_scale_b = make_copy_engine(
            format_transform="mx_scale_bdn",
        )
        self.fixpipe = make_copy_engine()

        block = get_block_idx()
        out_token_major = out_gm.view((self.padded_m, N_HEADS * HEAD_DIM))

        # M<=64 path: A [64,1536] and its full-K scale remain resident in L1
        # while this core consumes all assigned N128 tiles.
        l1_a_write = self.l1_a.produce()
        mem_copy(
            l1_a_write,
            tile_slice(
                qa_gm,
                (self.m, Q_LORA),
                (0, 0),
            ),
            engine=self.copy_a,
        )
        scale_a_write = self.l1_scale_a.produce()
        mem_copy(
            scale_a_write,
            tile_slice(sqa_gm, (self.m, self.scale_groups, 2), (0, 0, 0)),
            engine=self.copy_scale_a,
        )
        l1_a_read = self.l1_a.consume()
        scale_a_read = self.l1_scale_a.consume()
        a_fp8 = l1_a_read.reinterpret(dtype=Float8E4M3FN)
        # Assign complete N128 tiles round-robin.  This mapping is independent
        # of runtime branches and covers every output column exactly once.
        for n_idx in range(block, self.n_tiles, get_block_num()):
            scale_b_write = self.l1_scale_b.produce()
            mem_copy(
                scale_b_write,
                tile_slice(
                    swqb_gm,
                    (bn, self.scale_groups, 2),
                    (n_idx, 0, 0),
                ),
                engine=self.copy_scale_b,
            )
            scale_b_read = self.l1_scale_b.consume()
            l0c_write = self.l0c.produce()
            for k1 in range_constexpr(Q_LORA // kl1):
                l1_b_write = self.l1_b.produce()
                mem_copy(
                    l1_b_write,
                    tile_slice(
                        wqb,
                        (kl1, bn),
                        (k1, n_idx),
                    ),
                )
                l1_b_read = self.l1_b.consume()
                b_fp8 = l1_b_read.reinterpret(dtype=Float8E4M3FN)
                for k0 in range_constexpr(kl1 // bk):
                    global_k0 = k1 * (kl1 // bk) + k0
                    l0a_write = self.l0a.produce()
                    mem_copy(
                        l0a_write,
                        tile_slice(
                            a_fp8,
                            (self.m, bk),
                            (0, global_k0),
                        ),
                        mx_scale=tile_slice(
                            scale_a_read,
                            (self.m, self.scale_l0_len),
                            (0, global_k0),
                        ),
                    )
                    l0b_write = self.l0b.produce()
                    mem_copy(
                        l0b_write,
                        tile_slice(
                            b_fp8,
                            (bk, bn),
                            (k0, 0),
                        ),
                        transpose=True,
                        mx_scale=tile_slice(
                            scale_b_read,
                            (self.scale_l0_len, bn),
                            (global_k0, 0),
                        ),
                    )
                    final = global_k0 + 1 == Q_LORA // bk
                    matmul(
                        l0c_write,
                        self.l0a.consume(),
                        self.l0b.consume(),
                        init=(global_k0 == 0),
                        unit_flag=3 if final else 2,
                    )

            mem_copy(
                tile_slice(out_token_major, (self.m, bn), (0, n_idx)),
                self.l0c.consume(),
                engine=self.fixpipe,
                unit_flag=3,
            )


@kernel
class _MlaRuntimePrefillKernel:
    """Dynamic prefill keeps all token blocks live through each pipeline phase."""

    def __init__(self, norm_eps, quant_mode_c):
        self.norm_eps, self.quant_mode_c = norm_eps, quant_mode_c
        self.tail = _MlaInputTailUnit()
        self.qa = _QaRmsMixUnit(norm_eps)

    def __call__(
        self,
        qa_raw: Tensor,
        kva_raw: Tensor,
        x: Tensor,
        wqa: Tensor,
        wkva: Tensor,
        sx: Tensor,
        swqa: Tensor,
        swkva: Tensor,
        gamma_qa: Tensor,
        wqb: Tensor,
        swqb: Tensor,
        wkb: Tensor,
        gamma_kv: Tensor,
        qscale_kv: Tensor,
        cache: Tensor,
        cache_index: Tensor,
        qa_mx: Tensor,
        qa_scale: Tensor,
        q_head: Tensor,
        q_latent_h: Tensor,
        q_out: Tensor,
        descale_q: Tensor,
        x_pad: Tensor,
        sx_pad: Tensor,
    ):
        block, cores, sub = get_block_idx(), get_block_num(), get_subblock_id()
        valid, capacity = x.shape[0], x_pad.shape[0]
        x_block, sx_block = x_pad, sx_pad
        self.tail.pad(x, sx, x_pad, sx_pad, valid, capacity)
        global_sync_all()
        front = _O11MlaRef17FrontUnit(128)
        for n in range(block, Q_LORA // 64, cores):
            for m_idx in range(capacity // 128):
                front.project_qa_owned_tile(
                    qa_raw, x_block, wqa, sx_block, swqa, n, m_idx
                )
        _o11_front_sync_aic_to_aiv()
        channel_rewind(reset_sync_id=True)
        for n in range(block, FEATURES // 64, cores):
            for m_idx in range(capacity // 128):
                front.project_kva_tile(
                    kva_raw, x_block, wkva, sx_block, swkva, n, m_idx
                )
        qa_token = block * 2 + sub
        if qa_token < capacity:
            gamma_write = self.qa.gamma.produce()
            mem_copy(gamma_write, gamma_qa)
            gamma = self.qa.gamma.consume()
            for token in range(qa_token, capacity, 2 * cores):
                self.qa.process_one(qa_raw, gamma, qa_mx, qa_scale, token)
        _o11_front_sync_aic_to_aiv()
        channel_rewind(reset_sync_id=True)
        if const_expr(self.quant_mode_c == 1):
            kv = _MlaKvEpilogMxfp8Unit(PA_BLOCK_SIZE, self.norm_eps, output_depth=2)
            kv.process_assigned_rows(
                kva_raw, gamma_kv, qscale_kv, cache, cache_index, block, cores
            )
        else:
            kv = _MlaKvEpilogBf16Unit(PA_BLOCK_SIZE, self.norm_eps, output_depth=2)
            kv.process_assigned_rows(
                kva_raw, gamma_kv, cache, cache_index, block, cores
            )
        for m_idx in range(capacity // 128):
            qb = _MlaPrefillQbUnit(128)
            qb.projection_kernel(
                tile_slice(q_head, (128, N_HEADS, HEAD_DIM), (m_idx, 0, 0)),
                tile_slice(qa_mx.view(dtype=dtypes.int8), (128, Q_LORA), (m_idx, 0)),
                wqb,
                tile_slice(qa_scale, (128, Q_LORA // 64, 2), (m_idx, 0, 0)),
                swqb,
            )
        cube_sync_all()
        cube_sync_block_arrive(_sync_ops.PIPE.FIXPIPE, 1, mode=0)
        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, 1, mode=0)
        cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
        cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
        # Finish KV VECTOR/MTE3 accesses before Q reuses the same UB arena.
        vec_sync_all()
        channel_rewind(reset_sync_id=True)
        if cores < 8:
            vec_sync_block_wait(_sync_ops.PIPE.S, 3, mode=2)
        if const_expr(self.quant_mode_c == 1):
            epilog = _MlaQHeadTokenEpilogMxfp8Unit(
                64, capacity, capacity, token_major_rope=True
            )
            nope = q_latent_h.view((N_HEADS * capacity, KV_LORA))
            rope = q_head.view((capacity, N_HEADS * HEAD_DIM))
            out = q_out.view((capacity, N_HEADS * FEATURES))
            for head in range(block, N_HEADS, cores):
                if cores >= 8:
                    vec_sync_block_wait(_sync_ops.PIPE.S, 3 + head // cores, mode=2)
                for token_batch in range(sub, capacity // 8, 2):
                    epilog.process_batch(
                        nope, rope, out, descale_q, head * (capacity // 8) + token_batch
                    )
        else:
            epilog_bf16 = _MlaQHeadTokenEpilogBf16Unit(8, capacity, capacity)
            for head in range(block, N_HEADS, cores):
                if cores >= 8:
                    vec_sync_block_wait(_sync_ops.PIPE.S, 3 + head // cores, mode=2)
                for token_batch in range(sub, capacity // 8, 2):
                    epilog_bf16.process_batch(
                        q_latent_h, q_head, q_out, head, token_batch
                    )
        vec_sync_all()
        vec_sync_block_arrive(_sync_ops.PIPE.MTE3, 2, mode=2)
        wkb_unit = _O11MlaWkbTokenMajorLargeUnit(cdepth=2, bdepth=2)
        for head in range(block, N_HEADS, cores):
            wkb_unit.process_head(q_head, wkb, q_latent_h, head)
            if cores >= 8:
                cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
                cube_sync_block_arrive(_sync_ops.PIPE.MTE2, 3 + head // cores, mode=2)
        if cores < 8:
            cube_sync_notify(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
            cube_sync_wait(_sync_ops.PIPE.FIXPIPE, _sync_ops.PIPE.MTE2, 0)
            cube_sync_block_arrive(_sync_ops.PIPE.MTE2, 3, mode=2)
        _sync_ops.cube_sync_block_wait(_sync_ops.PIPE.S, 2, mode=2)
        global_sync_all()


class _MlaRuntimePrefillLauncher:
    def __init__(self, norm_eps, quant_mode_c):
        self.norm_eps, self.quant_mode_c = norm_eps, quant_mode_c

    @host
    def run(
        self,
        qa_raw,
        kva_raw,
        x,
        wqa,
        wkva,
        sx,
        swqa,
        swkva,
        gamma_qa,
        wqb,
        swqb,
        wkb,
        gamma_kv,
        qscale_kv,
        cache,
        cache_index,
        qa_mx,
        qa_scale,
        q_head,
        q_latent_h,
        q_out,
        descale_q,
        x_pad,
        sx_pad,
        core_num: int,
    ):
        _MlaRuntimePrefillKernel(self.norm_eps, self.quant_mode_c)[core_num](
            qa_raw,
            kva_raw,
            x,
            wqa,
            wkva,
            sx,
            swqa,
            swkva,
            gamma_qa,
            wqb,
            swqb,
            wkb,
            gamma_kv,
            qscale_kv,
            cache,
            cache_index,
            qa_mx,
            qa_scale,
            q_head,
            q_latent_h,
            q_out,
            descale_q,
            x_pad,
            sx_pad,
        )


class _O11Workspaces(NamedTuple):
    x_pad: torch.Tensor
    sx_pad: torch.Tensor
    qa_raw: torch.Tensor
    kva_raw: torch.Tensor
    qa_mx: torch.Tensor
    qa_scale: torch.Tensor
    q_head: torch.Tensor
    q_latent_h: torch.Tensor


def _allocate_o11_workspaces(x, qa_capacity, padded_m):
    x_pad = torch.empty((qa_capacity, DIM), dtype=torch.int8, device=x.device)
    sx_pad = torch.empty((qa_capacity, DIM // 64, 2), dtype=torch.int8, device=x.device)
    qa_raw = torch.empty((qa_capacity, Q_LORA), dtype=torch.float32, device=x.device)
    kva_raw = torch.empty((qa_capacity, FEATURES), dtype=torch.float32, device=x.device)
    qa_mx = torch.empty(
        (qa_capacity, Q_LORA), dtype=torch.float8_e4m3fn, device=x.device
    )
    qa_scale = torch.empty(
        (qa_capacity, Q_LORA // 64, 2), dtype=torch.int8, device=x.device
    )
    q_head = torch.empty(
        (padded_m, N_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=x.device,
    )
    q_latent_h = torch.empty(
        (N_HEADS, padded_m, KV_LORA),
        dtype=torch.bfloat16,
        device=x.device,
    )
    return _O11Workspaces(
        x_pad,
        sx_pad,
        qa_raw,
        kva_raw,
        qa_mx,
        qa_scale,
        q_head,
        q_latent_h,
    )


def _prepare_o11_outputs(x, gamma_kv, output_m, quant_mode_c, output_tensors):
    if output_tensors is None:
        output_dtype = torch.float8_e4m3fn if quant_mode_c == 1 else torch.bfloat16
        q_out = torch.empty(
            (output_m, N_HEADS, FEATURES),
            dtype=output_dtype,
            device=x.device,
        )
        descale_q = (
            torch.empty((output_m, N_HEADS), dtype=torch.float32, device=x.device)
            if quant_mode_c == 1
            else gamma_kv[:1]
        )
        return q_out, descale_q

    q_out, descale_q = output_tensors
    expected_dtype = torch.float8_e4m3fn if quant_mode_c == 1 else torch.bfloat16
    if q_out.dtype != expected_dtype or tuple(q_out.shape) != (
        output_m,
        N_HEADS,
        FEATURES,
    ):
        raise ValueError("output Q tensor does not match the C0/C1 ABI")
    if quant_mode_c == 1 and (
        descale_q.dtype != torch.float32
        or tuple(descale_q.shape) != (output_m, N_HEADS)
    ):
        raise ValueError("output descale_q does not match the C1 ABI")
    if quant_mode_c == 0:
        descale_q = gamma_kv[:1]
    return q_out, descale_q


def _run_o11_single_launch(
    x,
    sx,
    wqa,
    wkva,
    swqa,
    swkva,
    gamma_qa,
    wqb,
    swqb,
    wkb,
    gamma_kv,
    qscale_kv,
    cache,
    cache_index,
    norm_eps,
    quant_mode_c,
    output_tensors=None,
):
    """Allocate the reviewed call-private GM workspaces."""
    m = x.shape[0]
    if m <= 0:
        raise ValueError("single MixKernel requires a positive runtime T")
    tile_m = _native_tile_m(m)
    profile = 0 if m > 128 else tile_m
    padded_m = _workspace_token_capacity(m if profile == 0 else tile_m)
    output_m = _output_token_capacity(m)
    qa_capacity = padded_m if profile == 0 else tile_m
    workspaces = _allocate_o11_workspaces(x, qa_capacity, padded_m)
    x_pad, sx_pad, qa_raw, kva_raw, qa_mx, qa_scale, q_head, q_latent_h = workspaces
    q_out, descale_q = _prepare_o11_outputs(
        x, gamma_kv, output_m, quant_mode_c, output_tensors
    )
    core_num = _tensor_aic_core_num(x)
    program = _get_native_program(profile, quant_mode_c, float(norm_eps))
    program(
        qa_raw,
        kva_raw,
        x.view(torch.int8),
        wqa,
        wkva,
        sx.view(torch.int8),
        swqa.view(torch.int8),
        swkva.view(torch.int8),
        gamma_qa,
        wqb,
        swqb.view(torch.int8),
        wkb,
        gamma_kv,
        qscale_kv,
        cache,
        cache_index,
        qa_mx,
        qa_scale,
        q_head,
        q_latent_h,
        q_out,
        descale_q,
        x_pad,
        sx_pad,
        core_num,
    )
    logical_q = q_out[:m]
    logical_descale = descale_q[:m] if quant_mode_c == 1 else None
    return logical_q, logical_descale


def _same_storage(left, right):
    left_storage = left.untyped_storage()
    right_storage = right.untyped_storage()
    return left_storage.data_ptr() == right_storage.data_ptr() and getattr(
        left_storage, "_cdata", None
    ) == getattr(right_storage, "_cdata", None)


def _validate_public_scales(
    x,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
):
    expected = (
        ("descale_x", descale_x, (x.shape[0], DIM // 64, 2)),
        ("descale_wqa", descale_wqa, (Q_LORA, DIM // 64, 2)),
        ("descale_wqb", descale_wqb, (N_HEADS * HEAD_DIM, Q_LORA // 64, 2)),
        ("descale_wkva", descale_wkva, (FEATURES, DIM // 64, 2)),
    )
    for name, value, shape in expected:
        if value.dtype != torch.float8_e8m0fnu or tuple(value.shape) != shape:
            raise ValueError(f"{name} must be FP8 E8M0 with exact rank-3 shape {shape}")


def _logical_cache_as_pa_nz(kv_cache, quant_mode_c=1):
    if quant_mode_c not in (0, 1):
        raise ValueError("quant_mode_c must be 0 or 1")
    if kv_cache.dim() != 4:
        raise ValueError("kv_cache must be rank-4")
    blocks, block_size, heads, features = kv_cache.shape
    if blocks <= 0 or block_size != PA_BLOCK_SIZE:
        raise ValueError(
            "kv_cache must be [blocks,block_size,1,576] with block_size=128"
        )
    if heads != 1 or features != FEATURES:
        raise ValueError(
            "kv_cache must be [blocks,block_size,1,576] with block_size=128"
        )
    expected_dtype = torch.float8_e4m3fn if quant_mode_c == 1 else torch.bfloat16
    c0 = 32 if quant_mode_c == 1 else 16
    if kv_cache.dtype != expected_dtype or not kv_cache.is_contiguous():
        mode = "C1 FP8 E4M3" if quant_mode_c == 1 else "C0 BF16"
        raise ValueError(f"{mode} kv_cache must be contiguous")
    physical = kv_cache.view(blocks, FEATURES // c0, block_size, c0)
    if not _same_storage(physical, kv_cache):
        raise RuntimeError("logical-to-PA_NZ cache view copied")
    return physical


def _validate_cache_capacity(kv_cache, token_count):
    blocks, block_size, _, _ = kv_cache.shape
    if blocks * block_size < token_count:
        raise ValueError(
            f"kv_cache capacity {blocks}*{block_size} is smaller than "
            f"token count {token_count}"
        )
    return blocks


def _validate_o11_shapes(x, wkb, norm_weight_qa, norm_weight_kva, cache_index):
    if x.dtype != torch.float8_e4m3fn or x.dim() != 2:
        raise ValueError("candidate x must be FP8 E4M3 [T,7168] with T > 0")
    if x.shape[0] <= 0 or x.shape[1] != DIM:
        raise ValueError("candidate x must be FP8 E4M3 [T,7168] with T > 0")
    if wkb.dtype != torch.bfloat16 or tuple(wkb.shape) != (N_HEADS, QK_NOPE, KV_LORA):
        raise ValueError("wkb must be BF16 [96,128,512]")
    if norm_weight_qa.dtype != torch.float32 or tuple(norm_weight_qa.shape) != (
        Q_LORA,
    ):
        raise ValueError("norm_weight_qa must be FP32 [1536]")
    if norm_weight_kva.dtype != torch.float32 or tuple(norm_weight_kva.shape) != (
        KV_LORA,
    ):
        raise ValueError("norm_weight_kva must be FP32 [512]")
    if cache_index.dtype != torch.int64 or tuple(cache_index.shape) != (x.shape[0],):
        raise ValueError("cache_index must be INT64 [T]")


def _validate_o11_qscale(qscale_kv, quant_mode_c):
    if quant_mode_c == 1:
        if (
            qscale_kv is None
            or qscale_kv.dtype != torch.float32
            or tuple(qscale_kv.shape) != (1,)
        ):
            raise ValueError("qscale_kv must be FP32 with exact shape (1,)")
    if quant_mode_c == 0 and qscale_kv is not None:
        if qscale_kv.dtype != torch.float32 or qscale_kv.numel() < 1:
            raise ValueError("internal C0 qscale placeholder must be FP32")


def _validate_o11_weights(wqa, wqb, wkva):
    for name, weight, output_size, input_size in (
        ("wqa", wqa, Q_LORA, DIM),
        ("wqb", wqb, N_HEADS * HEAD_DIM, Q_LORA),
        ("wkva", wkva, FEATURES, DIM),
    ):
        expected = (output_size // 32, input_size, 32)
        if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != expected:
            raise ValueError(f"{name} must be FP8 E4M3 NZ {expected}")


def _validate_o11_fast_path_inputs(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    qscale_kv,
    norm_eps,
    quant_mode_c,
):
    if not math.isfinite(float(norm_eps)) or norm_eps <= 0:
        raise ValueError("norm_eps must be finite and positive")
    tensors = {
        "x": x,
        "wqa": wqa,
        "wqb": wqb,
        "wkva": wkva,
        "wkb": wkb,
        "descale_x": descale_x,
        "descale_wqa": descale_wqa,
        "descale_wqb": descale_wqb,
        "descale_wkva": descale_wkva,
        "norm_weight_qa": norm_weight_qa,
        "norm_weight_kva": norm_weight_kva,
        "kv_cache": kv_cache,
        "cache_index": cache_index,
    }
    if qscale_kv is not None:
        tensors["qscale_kv"] = qscale_kv
    for name, value in tensors.items():
        _require_npu(name, value, x.device)
    _validate_o11_shapes(x, wkb, norm_weight_qa, norm_weight_kva, cache_index)
    _validate_o11_qscale(qscale_kv, quant_mode_c)
    _validate_o11_weights(wqa, wqb, wkva)
    _validate_public_scales(x, descale_x, descale_wqa, descale_wqb, descale_wkva)


def _validate_quant_mla_modes(x, qscale_kv, quant_mode_aw, quant_mode_c):
    fast = (
        isinstance(x, torch.Tensor)
        and x.dim() == 2
        and x.shape[0] > 0
        and quant_mode_aw == 1
        and quant_mode_c in (0, 1)
    )
    if not fast:
        raise ValueError(
            "strict single-MixKernel path requires "
            "T > 0, "
            "quant_mode_aw=1 and quant_mode_c in {0,1}"
        )
    if quant_mode_c == 1 and qscale_kv is None:
        raise ValueError("qscale_kv is required when quant_mode_c=1")
    if quant_mode_c == 0 and qscale_kv is not None:
        raise ValueError("qscale_kv must be None when quant_mode_c=0")


def _build_quant_mla_outputs(q_out, descale_q, kv_cache, physical_cache):
    if physical_cache.data_ptr() != kv_cache.data_ptr() or not _same_storage(
        physical_cache, kv_cache
    ):
        raise RuntimeError("candidate cache update was not in place")
    outputs = {
        "q": q_out,
        "kv_cache_out": kv_cache,
    }
    if descale_q is not None:
        outputs["descale_q"] = descale_q
    return outputs


def quant_mla_prolog(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    qscale_kv=None,
    *,
    norm_eps,
    quant_mode_aw=1,
    quant_mode_c=0,
):
    """Dispatch runtime-T AW1 C0/C1 inputs to one MixKernel."""
    _validate_quant_mla_modes(x, qscale_kv, quant_mode_aw, quant_mode_c)
    _validate_o11_fast_path_inputs(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        qscale_kv,
        norm_eps,
        quant_mode_c,
    )
    physical_cache = _logical_cache_as_pa_nz(kv_cache, quant_mode_c)
    _validate_cache_capacity(kv_cache, x.shape[0])
    sx = _logical_scale(descale_x, x.shape[0], DIM)
    swqa = _logical_scale(descale_wqa, Q_LORA, DIM)
    swqb = _logical_scale(descale_wqb, N_HEADS * HEAD_DIM, Q_LORA)
    swkva = _logical_scale(descale_wkva, FEATURES, DIM)
    q_out, descale_q = _run_o11_single_launch(
        x,
        sx,
        wqa,
        wkva,
        swqa,
        swkva,
        norm_weight_qa,
        wqb,
        swqb,
        wkb,
        norm_weight_kva,
        qscale_kv if qscale_kv is not None else norm_weight_kva[:1],
        physical_cache,
        cache_index,
        norm_eps,
        quant_mode_c,
    )
    return _build_quant_mla_outputs(q_out, descale_q, kv_cache, physical_cache)


# Graph execution uses one byte-packed output because the dispatcher schema
# cannot return tensors with two dtypes through a single preallocated output.
# The mutable cache remains an explicit alias in the schema and is updated by
# the same single MixKernel used by eager execution.
_GRAPH_LIBRARY = torch.library.Library("cannbotdsl_quant_mla_prolog", "DEF")
_GRAPH_LIBRARY.define(
    "quant_mla_prolog("
    "Tensor x, Tensor wqa, Tensor wqb, Tensor wkva, Tensor wkb, "
    "Tensor descale_x, Tensor descale_wqa, Tensor descale_wqb, "
    "Tensor descale_wkva, Tensor norm_weight_qa, "
    "Tensor norm_weight_kva, Tensor(a!) kv_cache, "
    "Tensor cache_index, Tensor qscale_kv, float norm_eps, "
    "int quant_mode_aw=1, int quant_mode_c=1"
    ") -> (Tensor, Tensor(a!))"
)
_GRAPH_LIBRARY.define(
    "quant_mla_prolog_functional("
    "Tensor x, Tensor wqa, Tensor wqb, Tensor wkva, Tensor wkb, "
    "Tensor descale_x, Tensor descale_wqa, Tensor descale_wqb, "
    "Tensor descale_wkva, Tensor norm_weight_qa, "
    "Tensor norm_weight_kva, Tensor kv_cache, "
    "Tensor cache_index, Tensor qscale_kv, float norm_eps, "
    "int quant_mode_aw=1, int quant_mode_c=1"
    ") -> (Tensor, Tensor)"
)

_GRAPH_FP8_Q_BYTES_PER_TOKEN = N_HEADS * FEATURES
_GRAPH_BF16_Q_BYTES_PER_TOKEN = N_HEADS * FEATURES * 2
_GRAPH_SCALE_BYTES_PER_TOKEN = N_HEADS * 4
_GRAPH_C1_BYTES_PER_TOKEN = _GRAPH_FP8_Q_BYTES_PER_TOKEN + _GRAPH_SCALE_BYTES_PER_TOKEN


def _graph_output_views(packed, tokens, quant_mode_c):
    capacity = _output_token_capacity(tokens)
    if quant_mode_c == 0:
        q = packed.view(torch.bfloat16).view(capacity, N_HEADS, FEATURES)[:tokens]
        return q, None
    if quant_mode_c != 1:
        raise ValueError("quant_mode_c must be 0 or 1")
    q_bytes = capacity * _GRAPH_FP8_Q_BYTES_PER_TOKEN
    q = (
        packed.narrow(0, 0, q_bytes)
        .view(torch.float8_e4m3fn)
        .view(capacity, N_HEADS, FEATURES)[:tokens]
    )
    descale_q = (
        packed.narrow(0, q_bytes, capacity * _GRAPH_SCALE_BYTES_PER_TOKEN)
        .view(torch.float32)
        .view(capacity, N_HEADS)[:tokens]
    )
    return q, descale_q


def _allocate_graph_output(x, quant_mode_c):
    bytes_per_token = (
        _GRAPH_C1_BYTES_PER_TOKEN
        if quant_mode_c == 1
        else _GRAPH_BF16_Q_BYTES_PER_TOKEN
    )
    packed = torch.empty(
        (_output_token_capacity(x.shape[0]) * bytes_per_token,),
        dtype=torch.uint8,
        device=x.device,
    )
    q_out, descale_q = _graph_output_views(packed, x.shape[0], quant_mode_c)
    return packed, q_out, descale_q


def _graph_launch_output_tensors(q_out, descale_q, token_count):
    capacity = _output_token_capacity(token_count)
    q_storage = q_out.as_strided(
        (capacity, N_HEADS, FEATURES),
        q_out.stride(),
    )
    descale_storage = (
        descale_q.as_strided((capacity, N_HEADS), descale_q.stride())
        if descale_q is not None
        else descale_q
    )
    return q_storage, descale_storage


@torch.library.impl(_GRAPH_LIBRARY, "quant_mla_prolog", "Meta")
def _quant_mla_prolog_meta(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    qscale_kv,
    norm_eps,
    quant_mode_aw=1,
    quant_mode_c=1,
):
    del (
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        cache_index,
        qscale_kv,
        norm_eps,
        quant_mode_aw,
    )
    bytes_per_token = (
        _GRAPH_C1_BYTES_PER_TOKEN
        if quant_mode_c == 1
        else _GRAPH_BF16_Q_BYTES_PER_TOKEN
    )
    packed = torch.empty(
        (_output_token_capacity(x.shape[0]) * bytes_per_token,),
        dtype=torch.uint8,
        device="meta",
    )
    return packed, kv_cache


@torch.library.impl(_GRAPH_LIBRARY, "quant_mla_prolog_functional", "Meta")
def _quant_mla_prolog_functional_meta(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    qscale_kv,
    norm_eps,
    quant_mode_aw=1,
    quant_mode_c=1,
):
    packed, _ = _quant_mla_prolog_meta(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        qscale_kv,
        norm_eps,
        quant_mode_aw,
        quant_mode_c,
    )
    return packed, torch.empty_like(kv_cache, device="meta")


@torch.library.impl(_GRAPH_LIBRARY, "quant_mla_prolog", "PrivateUse1")
def _quant_mla_prolog_privateuse1(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    qscale_kv,
    norm_eps,
    quant_mode_aw=1,
    quant_mode_c=1,
):
    """ACLGraph adapter backed by the eager path's single MixKernel."""
    _validate_o11_fast_path_inputs(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        qscale_kv,
        norm_eps,
        quant_mode_c,
    )
    if quant_mode_aw != 1:
        raise ValueError("quant_mode_aw must be 1")
    physical_cache = _logical_cache_as_pa_nz(kv_cache, quant_mode_c)
    _validate_cache_capacity(kv_cache, x.shape[0])
    packed, q_out, descale_q = _allocate_graph_output(x, quant_mode_c)
    output_tensors = _graph_launch_output_tensors(q_out, descale_q, x.shape[0])
    outputs = _run_o11_single_launch(
        x,
        _logical_scale(descale_x, x.shape[0], DIM),
        wqa,
        wkva,
        _logical_scale(descale_wqa, Q_LORA, DIM),
        _logical_scale(descale_wkva, FEATURES, DIM),
        norm_weight_qa,
        wqb,
        _logical_scale(descale_wqb, N_HEADS * HEAD_DIM, Q_LORA),
        wkb,
        norm_weight_kva,
        qscale_kv,
        physical_cache,
        cache_index,
        norm_eps,
        quant_mode_c,
        output_tensors=output_tensors,
    )
    if outputs[0].data_ptr() != q_out.data_ptr():
        raise RuntimeError("graph Q output did not use packed storage")
    if quant_mode_c == 1 and (outputs[1].data_ptr() != descale_q.data_ptr()):
        raise RuntimeError("graph descale_q did not use packed storage")
    return packed, kv_cache


@torch.library.impl(_GRAPH_LIBRARY, "quant_mla_prolog_functional", "PrivateUse1")
def _quant_mla_prolog_functional_privateuse1(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    qscale_kv,
    norm_eps,
    quant_mode_aw=1,
    quant_mode_c=1,
):
    cache_out = kv_cache.clone()
    packed, _ = _quant_mla_prolog_privateuse1(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        cache_out,
        cache_index,
        qscale_kv,
        norm_eps,
        quant_mode_aw,
        quant_mode_c,
    )
    return packed, cache_out


_quant_mla_prolog_mutable_op = (
    torch.ops.cannbotdsl_quant_mla_prolog.quant_mla_prolog.default
)
_quant_mla_prolog_functional_op = (
    torch.ops.cannbotdsl_quant_mla_prolog.quant_mla_prolog_functional.default
)


@_quant_mla_prolog_mutable_op.py_functionalize_impl
def _quant_mla_prolog_functionalize(
    ctx,
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    qscale_kv,
    norm_eps,
    quant_mode_aw=1,
    quant_mode_c=1,
):
    args = ctx.unwrap_tensors(
        (
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            descale_x,
            descale_wqa,
            descale_wqb,
            descale_wkva,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            qscale_kv,
        )
    )
    with ctx.redispatch_to_next():
        packed, cache_out = _quant_mla_prolog_functional_op(
            *args,
            float(norm_eps),
            int(quant_mode_aw),
            int(quant_mode_c),
        )
    ctx.replace(kv_cache, cache_out)
    ctx.commit_update(kv_cache)
    ctx.sync(kv_cache)
    return ctx.wrap_tensors((packed, cache_out))


@torch.library.impl(_GRAPH_LIBRARY, "quant_mla_prolog", "Functionalize")
def _quant_mla_prolog_functionalize_dispatch(*args, **kwargs):
    from torch._subclasses.functional_tensor import CppFunctionalizeAPI

    return _quant_mla_prolog_functionalize(CppFunctionalizeAPI(), *args, **kwargs)


def _register_quant_mla_prolog_inplaceable():
    modules = (
        "npugraph_ex._acl_concrete_graph.graph_pass",
        "torch_npu.dynamo.torchair._acl_concrete_graph.graph_pass",
        "torch_npu.dynamo.npugraph_ex._acl_concrete_graph.graph_pass",
        "torchair._acl_concrete_graph.graph_pass",
    )
    for module_name in modules:
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        extra_check = getattr(
            module,
            "check_multi_stream_for_multi_reinplace",
            lambda node: True,
        )
        module.inplaceable_npu_ops[_quant_mla_prolog_functional_op] = (
            module.InplaceableNpuOp(
                inplace_op=_quant_mla_prolog_mutable_op,
                mutated_arg=[11],
                extra_check=extra_check,
            )
        )


_register_quant_mla_prolog_inplaceable()


def quant_mla_prolog_packed_op(*args, **kwargs):
    packed, _ = _quant_mla_prolog_mutable_op(*args, **kwargs)
    return packed


unpack_quant_mla_prolog_packed = _graph_output_views


def quant_mla_prolog_op(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    qscale_kv=None,
    *,
    norm_eps,
    quant_mode_aw=1,
    quant_mode_c=0,
):
    """Graph entry with the public Prolog argument contract.

    The graph path accepts the same explicit mode attributes as the network.
    Dispatcher defaults only protect functionalization paths that omit the two
    trailing scalar values; normal model calls still pass both values.
    """
    if quant_mode_aw != 1:
        raise ValueError("quant_mla_prolog_op requires quant_mode_aw=1")
    if quant_mode_c not in (0, 1):
        raise ValueError("quant_mode_c must be 0 or 1")
    if quant_mode_c == 1 and qscale_kv is None:
        raise ValueError("qscale_kv is required when quant_mode_c=1")
    if quant_mode_c == 0 and qscale_kv is not None:
        raise ValueError("qscale_kv must be None when quant_mode_c=0")
    graph_qscale = qscale_kv if qscale_kv is not None else norm_weight_kva[:1]
    packed, _ = _quant_mla_prolog_mutable_op(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        graph_qscale,
        float(norm_eps),
        int(quant_mode_aw),
        int(quant_mode_c),
    )
    return _graph_output_views(packed, x.shape[0], quant_mode_c)


def _quant_mla_prolog_dynamo(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    qscale_kv=None,
    *,
    norm_eps,
    quant_mode_aw=1,
    quant_mode_c=0,
):
    """Traceable replacement preserving the public dictionary contract."""
    if quant_mode_aw != 1 or quant_mode_c not in (0, 1):
        raise ValueError(
            "graph execution requires quant_mode_aw=1 and quant_mode_c in {0,1}"
        )
    if quant_mode_c == 1 and qscale_kv is None:
        raise ValueError("graph C1 execution requires qscale_kv")
    if quant_mode_c == 0 and qscale_kv is not None:
        raise ValueError("graph C0 execution requires qscale_kv=None")
    graph_qscale = qscale_kv if qscale_kv is not None else norm_weight_kva[:1]
    packed, cache_out = _quant_mla_prolog_mutable_op(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        graph_qscale,
        float(norm_eps),
        int(quant_mode_aw),
        int(quant_mode_c),
    )
    q, descale_q = _graph_output_views(packed, x.shape[0], quant_mode_c)
    outputs = {
        "q": q,
        "kv_cache_out": cache_out,
    }
    if descale_q is not None:
        outputs["descale_q"] = descale_q
    return outputs


# Existing network code calls the public eager function directly.  Dynamo
# substitutes only while tracing, preserving that function's dictionary return
# while recording the registered mutable dispatcher op in the graph.
torch_dynamo.substitute_in_graph(quant_mla_prolog, can_constant_fold_through=False)(
    _quant_mla_prolog_dynamo
)


def _native_token_bounds(tile_m):
    bounds = {8: (1, 8), 32: (9, 32), 64: (33, 64), 128: (65, 128), 0: (129, None)}
    selected_bounds = bounds.get(tile_m)
    if selected_bounds is None:
        raise ValueError(f"unsupported tile_m: {tile_m}")
    return selected_bounds


def _compile_o11_native_variant(tile_m, quant_mode_c, norm_eps=1.0e-6):
    """Compile a T interval; hardware/stream core count is a runtime scalar."""

    lower, upper = _native_token_bounds(tile_m)
    kwargs = {"min": lower}
    if upper is not None:
        kwargs["max"] = upper
    tokens = cannbotdsl.Dim("tokens", **kwargs)
    cache_blocks = cannbotdsl.Dim("cache_blocks", min=1)
    spec = cannbotdsl.TensorSpec
    capacity = ((tokens + 127) // 128) * 128 if tile_m == 0 else tile_m
    padded = capacity if tile_m == 0 else _workspace_token_capacity(tile_m)
    output_rows = capacity
    fp8 = dtypes.float8_e4m3fn
    output_dtype = fp8 if quant_mode_c == 1 else dtypes.bfloat16
    cache_dtype = output_dtype
    cache_c0 = 32 if quant_mode_c == 1 else 16
    descale_shape = (output_rows, N_HEADS) if quant_mode_c == 1 else (1,)
    launcher = (
        _MlaRuntimePrefillLauncher(float(norm_eps), quant_mode_c)
        if tile_m == 0
        else _MlaRuntimeLauncher(tile_m, float(norm_eps), quant_mode_c)
    )
    return cannbotdsl.compile(
        launcher.run,
        spec((capacity, Q_LORA), dtypes.float32),
        spec((capacity, FEATURES), dtypes.float32),
        spec((tokens, DIM), dtypes.int8),
        spec((Q_LORA // 32, DIM, 32), fp8),
        spec((FEATURES // 32, DIM, 32), fp8),
        spec((tokens, DIM // 64, 2), dtypes.int8),
        spec((Q_LORA, DIM // 64, 2), dtypes.int8),
        spec((FEATURES, DIM // 64, 2), dtypes.int8),
        spec((Q_LORA,), dtypes.float32),
        spec((N_HEADS * HEAD_DIM // 32, Q_LORA, 32), fp8),
        spec((N_HEADS * HEAD_DIM, Q_LORA // 64, 2), dtypes.int8),
        spec((N_HEADS, QK_NOPE, KV_LORA), dtypes.bfloat16),
        spec((KV_LORA,), dtypes.float32),
        spec((1,), dtypes.float32),
        spec(
            (cache_blocks, FEATURES // cache_c0, PA_BLOCK_SIZE, cache_c0), cache_dtype
        ),
        spec((tokens,), dtypes.int64),
        spec((capacity, Q_LORA), fp8),
        spec((capacity, Q_LORA // 64, 2), dtypes.int8),
        spec((padded, N_HEADS, HEAD_DIM), dtypes.bfloat16),
        spec((N_HEADS, padded, KV_LORA), dtypes.bfloat16),
        spec((output_rows, N_HEADS, FEATURES), output_dtype),
        spec(descale_shape, dtypes.float32),
        spec((capacity, DIM), dtypes.int8),
        spec((capacity, DIM // 64, 2), dtypes.int8),
        dtypes.int64,
    )


@lru_cache(maxsize=16)
def _get_native_program(tile_m, quant_mode_c, norm_eps):
    return _compile_o11_native_variant(tile_m, quant_mode_c, norm_eps)


@export("quant_mla_prolog")
def export_quant_mla_prolog():
    """Collect the same runtime-core/T providers used by the public operator."""
    handles = []
    try:
        for tile_m in (8, 32, 64, 128, 0):
            for quant_mode_c in (0, 1):
                handles.append(_get_native_program(tile_m, quant_mode_c, 1.0e-6))
    finally:
        for handle in handles:
            handle.close()
        _get_native_program.cache_clear()
