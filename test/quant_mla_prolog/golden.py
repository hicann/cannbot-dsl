# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden, metadata cases and NPU precision tests for quant_mla_prolog.

qa/qb/kva all use block-32 E4M3/E8M0 operands. RMS reductions are FP32; the
public FP32 gamma is rounded to the current AscendC BF16 boundary. qa RMS is
followed by BF16 CAST_ROUND (ties away) and source-compatible MX quantization.
qscale_kv is a dequantization scale: encoded KV = KV / qscale_kv.
Q/KV output quantization operates once on each combined 512+64 row. MXFP8
mode stores E4M3 output bytes; HIF8 mode stores raw HiFloat8 uint8 bytes.
The golden itself performs CPU work only; pytest also launches the NPU kernel.
External descale_x is ND; descale_wqa/descale_wqb/descale_wkva are NZ only.
Generic ND scale helpers remain available for internally generated MX scales.
"""

import importlib
import json
import math
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
torch = importlib.import_module("torch")

SEED = 20260908
NORM_EPS = 1.0e-6
BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
FP8_MAX = 448.0
HIF8_MAX = 32768.0
MIN_ACCURACY = 0.995
NETWORK_RELATIVE_L2_LIMIT = 3.0e-2
NETWORK_Q_COSINE_MIN = 0.999
FIXED_KV_LORA = 512
FIXED_FEATURES = 576
COMMON_INPUTS = (
    "x",
    "wqa",
    "wqb",
    "wkva",
    "wkb",
    "norm_weight_qa",
    "norm_weight_kva",
    "kv_cache",
    "cache_index",
)
WEIGHTS = ("wqa", "wqb", "wkva")
DESCALES = ("descale_x", "descale_wqa", "descale_wqb", "descale_wkva")
TENSOR_INPUTS = COMMON_INPUTS + DESCALES + ("qscale_kv",)
INPUT_RANKS = (
    ("x", 2),
    ("wqa", 3),
    ("wqb", 3),
    ("wkva", 3),
    ("wkb", 3),
    ("norm_weight_qa", 1),
    ("norm_weight_kva", 1),
    ("kv_cache", 4),
    ("cache_index", 1),
)
DTYPES = {
    "bf16": BF16,
    "fp8_e4m3": FP8,
    "fp8_e8m0": E8M0,
    "hif8": torch.uint8,
    "float32": torch.float32,
    "int64": torch.int64,
}


def _check_modes(quant_mode_aw, quant_mode_c):
    if type(quant_mode_aw) is not int or quant_mode_aw != 1:
        raise ValueError("This quant_mla_prolog golden requires quant_mode_aw=1")
    if type(quant_mode_c) is not int or quant_mode_c not in (0, 1):
        raise ValueError("quant_mode_c must be 0 or 1")


def _check_eps(norm_eps):
    if (
        type(norm_eps) not in (int, float)
        or not math.isfinite(norm_eps)
        or norm_eps <= 0
    ):
        raise ValueError("norm_eps must be positive and finite")


def _model_dims(values):
    t, he, qr, kr, n, d, dn, dr, blocks, page = values
    return {
        "T": t,
        "He": he,
        "q_lora": qr,
        "kv_lora": kr,
        "N": n,
        "D": d,
        "Dn": dn,
        "Dr": dr,
        "block_number": blocks,
        "block_size": page,
    }


def _finite(tensor, name):
    if not bool(torch.isfinite(tensor.float()).all()):
        raise ValueError(f"{name}: numerical inputs/intermediates must be finite")


def _raw(tensor):
    if tensor.dtype == BF16:
        return tensor.view(torch.int16)
    if tensor.dtype in (FP8, E8M0, torch.uint8):
        return tensor.view(torch.uint8)
    raise ValueError("Expected BF16 or an 8-bit float storage tensor")


def bitwise_equal(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    )


def pack_nz(weight):
    if weight.dtype != FP8 or weight.device.type != "cpu" or weight.ndim != 2:
        raise ValueError("Weight NZ expects an E4M3 CPU matrix W[O,K]")
    o, k = weight.shape
    if o <= 0 or k <= 0 or o % 32:
        raise ValueError("NZ output channels must be divisible by 32")
    return weight.reshape(o // 32, 32, k).permute(0, 2, 1).contiguous()


def unpack_nz(weight):
    if weight.dtype != FP8 or weight.device.type != "cpu":
        raise ValueError("Expected E4M3 NZ weight storage [O/32,K,32]")
    if weight.ndim != 3 or weight.shape[-1] != 32:
        raise ValueError("Expected E4M3 NZ weight storage [O/32,K,32]")
    return (
        weight.permute(0, 2, 1)
        .reshape(weight.shape[0] * 32, weight.shape[1])
        .contiguous()
    )


def pack_scale_pair_nz(logical, *, group_alignment):
    """Explicit layout utility: caller must select tight (1) or G-padded (16).

    [O,G,2] -> [ceil(O/16),ceil(G/alignment)*alignment,16,2].
    Pair bytes stay adjacent; there is no numeric cast through BF16.
    """
    if logical.dtype != E8M0 or logical.device.type != "cpu":
        raise ValueError("Scale input must be E8M0 [O,G,2] on CPU")
    if logical.ndim != 3 or logical.shape[-1] != 2:
        raise ValueError("Scale input must be E8M0 [O,G,2] on CPU")
    if group_alignment not in (1, 16):
        raise ValueError("Explicit group_alignment must be 1 or 16")
    o, g, _ = logical.shape
    gp = ((g + group_alignment - 1) // group_alignment) * group_alignment
    storage = torch.full(((o + 15) // 16, gp, 16, 2), 127, dtype=torch.uint8)
    padded = storage.permute(0, 2, 1, 3).contiguous().reshape(-1, gp, 2)
    padded[:o, :g] = logical.contiguous().view(torch.uint8)
    return padded.reshape(-1, 16, gp, 2).permute(0, 2, 1, 3).contiguous().view(E8M0)


def unpack_scale_pair_nz(storage, outer, groups, *, group_alignment):
    if (
        storage.dtype != E8M0
        or storage.device.type != "cpu"
        or group_alignment not in (1, 16)
    ):
        raise ValueError("Expected E8M0 CPU storage and explicit alignment 1/16")
    gp = ((groups + group_alignment - 1) // group_alignment) * group_alignment
    if tuple(storage.shape) != ((outer + 15) // 16, gp, 16, 2):
        raise ValueError(
            "Scale NZ storage shape does not match the explicitly selected mapping"
        )
    data = storage.view(torch.uint8).permute(0, 2, 1, 3).contiguous().reshape(-1, gp, 2)
    return data[:outer, :groups].contiguous().view(E8M0)


def logical_descale(storage, outer, k):
    groups = (k + 63) // 64
    if storage.dtype != E8M0 or storage.device.type != "cpu":
        raise ValueError("descale must have dtype E8M0 on CPU")
    if tuple(storage.shape) == (outer, groups, 2):
        return storage
    if (
        storage.ndim != 4
        or storage.shape[0] != (outer + 15) // 16
        or tuple(storage.shape[2:]) != (16, 2)
    ):
        raise ValueError("Invalid E8M0 NZ storage shape")
    gp = storage.shape[1]
    if gp not in (groups, (groups + 15) // 16 * 16):
        raise ValueError("Scale NZ groups must be G or ceil(G/16)*16")
    return unpack_scale_pair_nz(
        storage, outer, groups, group_alignment=1 if gp == groups else 16
    )


def decode_e8m0(storage, outer, k):
    raw = logical_descale(storage, outer, k).contiguous().view(torch.uint8)
    # 0xFF is NaN, not an ordinary power-of-two scale. Reject even unused tails
    # in numerical input descriptors; pure layout pack/unpack accepts every byte.
    if bool((raw == 255).any()):
        raise ValueError("E8M0 raw 255 is NaN and is not accepted as numerical input")
    values = raw.reshape(outer, -1)[:, : (k + 31) // 32].float() - 127
    return torch.pow(2.0, values).repeat_interleave(32, dim=1)[:, :k]


def cast_bf16_round_away(value):
    """FP32 -> BF16 CAST_ROUND: midpoint rounds away from zero, including sign."""
    value = value.float().contiguous()
    _finite(value, "CAST_ROUND source")
    bits = value.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    upper = (bits >> 16) + ((bits & 0xFFFF) >= 0x8000).to(torch.int64)
    result = (upper & 0xFFFF).to(torch.int16).view(BF16)
    _finite(result, "CAST_ROUND result")
    return result


def dynamic_mx_quant_bf16(value):
    """Source-compatible BF16 block-32 MX quantization, including E=0 handling.

    E=max BF16 biased exponent per block; raw=max(E-8,0). A block with E=0
    (all zero/subnormal) uses multiplier 0. BF16 multiply precedes E4M3 RNE.
    Internal scale shape is [T,ceil(K/64),2]; an unused pair lane is raw 0.
    """
    if value.dtype != BF16 or value.device.type != "cpu" or value.ndim != 2:
        raise ValueError("Internal MX quantizer requires a BF16 CPU [T,K] tensor")
    _finite(value, "Internal MX source")
    t, k = value.shape
    if k <= 0 or k % 32:
        raise ValueError("Internal MX K must be divisible by 32")
    exponent = (value.contiguous().view(torch.int16).to(torch.int32) & 0x7F80) >> 7
    maximum = exponent.reshape(t, k // 32, 32).amax(-1)
    raw = (maximum - 8).clamp_min(0)
    inverse_bits = (0x7F00 - (raw << 7)).to(torch.int16)
    inverse_bits = torch.where(
        maximum == 0, torch.zeros_like(inverse_bits), inverse_bits
    )
    inverse = inverse_bits.contiguous().view(BF16).repeat_interleave(32, dim=1)
    normalized = (value.float() * inverse.float()).to(BF16).float()
    encoded = normalized.clamp(-FP8_MAX, FP8_MAX).to(FP8)
    scale = torch.zeros((t, (k + 63) // 64 * 2), dtype=torch.uint8)
    scale[:, : k // 32] = raw.to(torch.uint8)
    return encoded, scale.reshape(t, (k + 63) // 64, 2).view(E8M0)


def cache_c0(dtype):
    if dtype == BF16:
        return 16
    if dtype in (FP8, torch.uint8):
        return 32
    raise ValueError("Cache must be BF16, E4M3 or HiFloat8 byte storage")


def pa_nz_storage_shape(logical_shape, dtype):
    if len(logical_shape) != 4 or any(
        type(v) is not int or v <= 0 for v in logical_shape
    ):
        raise ValueError("Logical cache must have four positive dimensions")
    blocks, page_size, heads, f = logical_shape
    c0 = cache_c0(dtype)
    if heads != 1 or f % c0:
        raise ValueError("Cache requires one KV head and aligned feature count")
    return (blocks, f // c0, page_size, c0)


def pack_pa_nz(logical):
    blocks, f1, page, c0 = pa_nz_storage_shape(tuple(logical.shape), logical.dtype)
    return logical.reshape(blocks, page, f1, c0).permute(0, 2, 1, 3).contiguous()


def unpack_pa_nz(storage):
    if storage.ndim != 4 or storage.shape[-1] != cache_c0(storage.dtype):
        raise ValueError("Invalid physical PA_NZ shape")
    blocks, f1, page, c0 = storage.shape
    return storage.permute(0, 2, 1, 3).contiguous().reshape(blocks, page, 1, f1 * c0)


def _check_indices(indices, capacity, tokens):
    if indices.dtype != torch.int64 or tuple(indices.shape) != (tokens,):
        raise ValueError("cache_index must be INT64 [T]")
    if tokens and (indices.min() < 0 or indices.max() >= capacity):
        raise ValueError("cache_index exceeds cache slot capacity")
    if indices.unique().numel() != tokens:
        raise ValueError("Duplicate cache writes have undefined ordering")


def _cache_offsets(storage, indices):
    _, f1, page, c0 = storage.shape
    f = f1 * c0
    slots = indices.unsqueeze(1)
    d = torch.arange(f, dtype=torch.int64).unsqueeze(0)
    return (
        (slots // page) * f * page
        + (d // c0) * page * c0
        + (slots % page) * c0
        + d % c0
    )


def scatter_pa_nz(storage, indices, kv):
    if any(value.device.type != "cpu" for value in (storage, indices, kv)):
        raise ValueError("CPU tensors required")
    if (
        storage.ndim != 4
        or storage.shape[-1] != cache_c0(storage.dtype)
        or not storage.is_contiguous()
    ):
        raise ValueError("Invalid PA_NZ storage")
    blocks, f1, page, c0 = storage.shape
    if min(storage.shape) <= 0 or kv.dtype != storage.dtype:
        raise ValueError("KV/cache shape or dtype mismatch")
    if kv.ndim != 2 or kv.shape[-1] != f1 * c0:
        raise ValueError("KV/cache shape or dtype mismatch")
    _check_indices(indices, blocks * page, kv.shape[0])
    result = storage.clone()
    flat = _raw(result).reshape(-1)
    data = _raw(kv.contiguous())
    for start in range(0, indices.numel(), 128):
        stop = start + 128
        flat[_cache_offsets(storage, indices[start:stop])] = data[start:stop]
    return result


def rms_norm_fp32(value, weight, eps):
    x = value.float()
    # Match the source's sqrt -> division -> gamma order, rather than rsqrt*X.
    denominator = torch.sqrt(
        x.square().sum(-1, keepdim=True) * (1.0 / x.shape[-1]) + eps
    )
    result = (x / denominator) * weight
    _finite(result, "RMSNorm output")
    return result


def npu_divide_fp32_cpu(numerator, denominator):
    """Model the observed Ascend/NPU FP32 divide result on CPU.

    At exact E4M3 midpoints the device quotient is one FP32 ULP toward zero
    relative to torch CPU division. That one ULP selects the neighboring FP8
    bin, so preserve it explicitly at output-quantization boundaries.
    """
    quotient = numerator.float() / denominator.float()
    return torch.nextafter(quotient, torch.zeros_like(quotient))


def quantize_e4m3_npu_div_cpu(value, scale):
    normalized = npu_divide_fp32_cpu(value, scale)
    return normalized.clamp(-FP8_MAX, FP8_MAX).to(FP8)


def _hif8_positive_lut():
    """Finite positive HiFloat8 values and their storage codes."""
    values = [0.0]
    values += [2.0 ** (i - 23) for i in range(1, 8)]
    values += [1.0 + i / 8.0 for i in range(8)]
    values += [2.0 + i / 4.0 for i in range(8)]
    values += [0.5 + i / 16.0 for i in range(8)]
    values += [4.0 + i / 2.0 for i in range(8)]
    values += [8.0 + i for i in range(8)]
    values += [0.25 + i / 32.0 for i in range(8)]
    values += [0.125 + i / 64.0 for i in range(8)]
    values += [16.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0, 56.0]
    values += [64.0, 80.0, 96.0, 112.0, 128.0, 160.0, 192.0, 224.0]
    values += [0.0625 + i / 64.0 for i in range(4)]
    values += [0.03125 + i / 128.0 for i in range(4)]
    values += [0.015625 + i / 256.0 for i in range(4)]
    values += [0.0078125 + i / 512.0 for i in range(4)]
    values += [
        256.0,
        384.0,
        512.0,
        768.0,
        1024.0,
        1536.0,
        2048.0,
        3072.0,
        4096.0,
        6144.0,
        8192.0,
        12288.0,
        16384.0,
        24576.0,
        32768.0,
    ]
    # Code 111 is +inf. Codes 112..127 alternate 1.0/1.5 times powers.
    tail = []
    for exponent in range(-8, -16, -1):
        tail.extend((2.0**exponent, 1.5 * 2.0**exponent))
    values += tail
    codes = list(range(111)) + list(range(112, 128))
    pairs = sorted(zip(values, codes))
    return (
        torch.tensor([p[0] for p in pairs], dtype=torch.float32),
        torch.tensor([p[1] for p in pairs], dtype=torch.uint8),
    )


HIF8_POSITIVE_VALUES, HIF8_POSITIVE_CODES = _hif8_positive_lut()


def quantize_hif8_cpu(value):
    """Nearest finite HiFloat8 encoding; ties select the larger magnitude."""
    source = value.float().clamp(-HIF8_MAX, HIF8_MAX)
    magnitude = source.abs().reshape(-1)
    upper = torch.bucketize(magnitude, HIF8_POSITIVE_VALUES).clamp_max(
        HIF8_POSITIVE_VALUES.numel() - 1
    )
    lower = (upper - 1).clamp_min(0)
    choose_upper = (HIF8_POSITIVE_VALUES[upper] - magnitude) <= (
        magnitude - HIF8_POSITIVE_VALUES[lower]
    )
    index = torch.where(choose_upper, upper, lower)
    code = HIF8_POSITIVE_CODES[index]
    code = torch.where(source.reshape(-1) < 0, code | 0x80, code)
    return code.reshape(source.shape).contiguous()


def dequantize_hif8_cpu(value):
    if value.dtype != torch.uint8:
        raise ValueError("HiFloat8 storage must be uint8")
    code = value.to(torch.int64)
    magnitude_code = code & 0x7F
    lut = torch.empty(128, dtype=torch.float32)
    lut[HIF8_POSITIVE_CODES.long()] = HIF8_POSITIVE_VALUES
    lut[111] = float("inf")
    magnitude = lut[magnitude_code]
    return torch.where((code & 0x80) != 0, -magnitude, magnitude)


def cpu_golden(
    inputs, *, norm_eps=NORM_EPS, quant_mode_aw=1, quant_mode_c=0, quant_dtype_c="mxfp8"
):
    dims = validate_inputs(
        inputs,
        norm_eps=norm_eps,
        quant_mode_aw=quant_mode_aw,
        quant_mode_c=quant_mode_c,
        quant_dtype_c=quant_dtype_c,
    )
    x = inputs["x"].float() * decode_e8m0(inputs["descale_x"], dims["T"], dims["He"])
    weights = []
    for name in WEIGHTS:
        value = unpack_nz(inputs[name]).float()
        value *= decode_e8m0(inputs["descale_" + name], *value.shape)
        weights.append(value)
    wqa, wqb, wkva = weights
    # Kimi/Transformer RMSNorm math remains FP32.  The current AscendC MXFP8
    # ABI reads gamma as BF16, so model that boundary while accepting the
    # public DSL FP32 input.
    gamma_qa = inputs["norm_weight_qa"].to(BF16).float()
    gamma_kva = inputs["norm_weight_kva"].to(BF16).float()
    # FP8 operands and E8M0 descales are dyadic.  Use a stable, high-precision
    # reference dot product, then round the projection to the public FP32
    # RMSNorm boundary.  CPU BLAS FP32 accumulation can move values across
    # BF16/MXFP8 midpoints and must not define the mathematical oracle.
    qa_raw = (x.double() @ wqa.double().T).float()
    q_a_fp32 = rms_norm_fp32(qa_raw, gamma_qa, norm_eps)
    q_a_bf16 = cast_bf16_round_away(q_a_fp32)
    q_a_mx, q_a_scale = dynamic_mx_quant_bf16(q_a_bf16)
    q_a = q_a_mx.float() * decode_e8m0(q_a_scale, dims["T"], dims["q_lora"])
    q_states = (q_a @ wqb.T).reshape(dims["T"], dims["N"], dims["D"])
    nope_width = dims["Dn"]
    q_pass = q_states[..., :nope_width].to(BF16)
    q_rot = q_states[..., nope_width:].to(BF16)
    _finite(q_pass, "q_pass")
    _finite(q_rot, "q_rot")
    q_latent = torch.einsum("tnd,ndk->tnk", q_pass, inputs["wkb"])
    q = torch.cat((q_latent, q_rot), -1)
    kv_raw = x @ wkva.T
    kv_lora_width = dims["kv_lora"]
    kv_norm = rms_norm_fp32(kv_raw[..., :kv_lora_width], gamma_kva, norm_eps)
    k_rot = kv_raw[..., kv_lora_width:].to(BF16)
    _finite(q, "q")
    _finite(k_rot, "k_rot")
    if quant_mode_c == 0:
        kv = torch.cat((kv_norm.to(BF16), k_rot), -1)
    else:
        # The 512-wide q_nope and 64-wide q_rope keep distinct producers, then
        # share one per-token-head scale across the combined 576-wide row.
        q_float = q.float()
        quant_max = FP8_MAX if quant_dtype_c == "mxfp8" else HIF8_MAX
        descale_q = q_float.abs().amax(-1).clamp_min(1e-8) / quant_max
        if quant_dtype_c == "mxfp8":
            q = quantize_e4m3_npu_div_cpu(q_float, descale_q.unsqueeze(-1))
        else:
            q = quantize_hif8_cpu(q_float / descale_q.unsqueeze(-1))
        # Target combined-cache contract: latent remains FP32; Dr retains BF16 rounding.
        kv_float = torch.cat((kv_norm, k_rot.float()), -1)
        # The KV cache path uses the ordinary CPU-equivalent division/cast
        # semantics in the tested DSL kernel; the one-ULP NPU quotient model
        # above is specific to the split-Q output quantization boundary.
        if quant_dtype_c == "mxfp8":
            kv = (kv_float / inputs["qscale_kv"]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
        else:
            kv = quantize_hif8_cpu(kv_float / inputs["qscale_kv"])
    outputs = {
        "q": q,
        "kv_cache_out": scatter_pa_nz(inputs["kv_cache"], inputs["cache_index"], kv),
    }
    if quant_mode_c:
        outputs["descale_q"] = descale_q
    return outputs


def validate_inputs(
    inputs, *, norm_eps=NORM_EPS, quant_mode_aw=1, quant_mode_c=0, quant_dtype_c="mxfp8"
):
    _check_modes(quant_mode_aw, quant_mode_c)
    if quant_dtype_c not in ("mxfp8", "hif8"):
        raise ValueError("quant_dtype_c must be mxfp8 or hif8")
    _check_eps(norm_eps)
    required = set(COMMON_INPUTS + DESCALES) | (
        {"qscale_kv"} if quant_mode_c else set()
    )
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise ValueError(f"Expected exactly inputs {sorted(required)}")
    for name, value in inputs.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or not value.is_contiguous()
        ):
            raise ValueError(f"{name}: contiguous CPU tensor required")
        cache_dtype = torch.uint8 if quant_mode_c and quant_dtype_c == "hif8" else FP8
        expected_dtype = (
            E8M0
            if name in DESCALES
            else torch.float32
            if name.startswith("norm_weight_") or name == "qscale_kv"
            else torch.int64
            if name == "cache_index"
            else BF16
            if name == "wkb" or (name == "kv_cache" and quant_mode_c == 0)
            else cache_dtype
            if name == "kv_cache"
            else FP8
        )
        if value.dtype != expected_dtype:
            raise ValueError(f"{name}: expected dtype {expected_dtype}")
        if any(v <= 0 for v in value.shape):
            raise ValueError(f"{name}: dimensions must be positive")
    for name, rank in INPUT_RANKS:
        if inputs[name].ndim != rank:
            raise ValueError(f"{name}: expected rank {rank}")
    t, he = inputs["x"].shape
    qr, kr = inputs["norm_weight_qa"].numel(), inputs["norm_weight_kva"].numel()
    n, dn, _ = inputs["wkb"].shape
    blocks, f1, page, c0 = inputs["kv_cache"].shape
    if c0 != cache_c0(inputs["kv_cache"].dtype):
        raise ValueError("kv_cache: wrong physical inner axis")
    f, dr = f1 * c0, f1 * c0 - kr
    d = dn + dr
    if dr <= 0 or any(v % 32 for v in (qr, n * d, f)):
        raise ValueError("Invalid projection dimensions or NZ block-32 alignment")
    for name, shape in (
        ("wqa", (qr // 32, he, 32)),
        ("wqb", (n * d // 32, qr, 32)),
        ("wkva", (f // 32, he, 32)),
        ("wkb", (n, dn, kr)),
    ):
        if tuple(inputs[name].shape) != shape:
            raise ValueError(f"{name}: expected shape {shape}")
    for name, outer, k in (
        ("descale_x", t, he),
        ("descale_wqa", qr, he),
        ("descale_wqb", n * d, qr),
        ("descale_wkva", f, he),
    ):
        if name == "descale_x" and inputs[name].ndim != 3:
            raise ValueError("descale_x: external input must use 3D ND storage")
        if name != "descale_x" and inputs[name].ndim != 4:
            raise ValueError(f"{name}: external weight descale must use 4D NZ storage")
        logical = logical_descale(inputs[name], outer, k)
        if bool((logical.view(torch.uint8) == 255).any()):
            raise ValueError(f"{name}: logical E8M0 raw 255 is NaN")
    _check_indices(inputs["cache_index"], blocks * page, t)
    for name in ("x",) + WEIGHTS + ("wkb", "norm_weight_qa", "norm_weight_kva"):
        _finite(inputs[name], name)
    if quant_mode_c:
        if tuple(inputs["qscale_kv"].shape) != (1,):
            raise ValueError("qscale_kv must be FP32 [1]")
        scale = inputs["qscale_kv"].item()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("qscale_kv must be positive and finite")
    return _model_dims((t, he, qr, kr, n, d, dn, dr, blocks, page))


def parse_tensor(name, text):
    if not text:
        return None
    try:
        spec = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{name}: invalid tensor JSON") from exc
    required = {"shape", "dtype", "format"}
    allowed = required | {"storage_shape"}
    if name in ("cache_index", "qscale_kv"):
        allowed.add("values")
    if name == "kv_cache":
        allowed.update(("init", "occupied_ranges"))
    if not isinstance(spec, dict) or not required <= set(spec) or set(spec) - allowed:
        raise ValueError(f"{name}: expected keys from {sorted(allowed)}")
    if (
        not isinstance(spec["shape"], list)
        or not spec["shape"]
        or any(type(v) is not int or v <= 0 for v in spec["shape"])
    ):
        raise ValueError(f"{name}: shape must contain positive integers")
    if (
        not isinstance(spec["dtype"], str)
        or spec["dtype"] not in DTYPES
        or not isinstance(spec["format"], str)
    ):
        raise ValueError(f"{name}: invalid dtype or format")
    return spec


def _scale_storage_shape(spec):
    o, g, lane = spec["shape"]
    fmt = spec["format"].upper()
    if lane != 2:
        raise ValueError("MX scales require paired lanes")
    if fmt == "ND":
        expected = tuple(spec["shape"])
        if "storage_shape" in spec and tuple(spec["storage_shape"]) != expected:
            raise ValueError("ND scale storage_shape must equal logical shape")
        return expected
    if fmt != "NZ" or "storage_shape" not in spec:
        raise ValueError(
            "NZ descale requires an explicit storage_shape; no ABI is inferred from its label"
        )
    shape = spec["storage_shape"]
    choices = {((o + 15) // 16, g, 16, 2), ((o + 15) // 16, (g + 15) // 16 * 16, 16, 2)}
    if (
        not isinstance(shape, list)
        or any(type(v) is not int for v in shape)
        or tuple(shape) not in choices
    ):
        raise ValueError(f"NZ descale storage_shape must be one of {sorted(choices)}")
    return tuple(shape)


def validate_case(case):
    _check_modes(case["quant_mode_aw"], case["quant_mode_c"])
    _check_eps(case["norm_eps"])
    specs = case["specs"]
    if set(specs) != set(TENSOR_INPUTS):
        raise ValueError(
            "Case must include all declared input descriptors, optional qscale may be empty"
        )
    for name in COMMON_INPUTS + DESCALES:
        if specs[name] is None:
            raise ValueError(f"{name}: required input missing")
    for name, spec in specs.items():
        if spec is not None:
            parse_tensor(name, json.dumps(spec))
    for name, rank in INPUT_RANKS:
        if len(specs[name]["shape"]) != rank:
            raise ValueError(f"{name}: expected rank {rank}")
    t, he = specs["x"]["shape"]
    qr, kr = specs["norm_weight_qa"]["shape"][0], specs["norm_weight_kva"]["shape"][0]
    n, dn, _ = specs["wkb"]["shape"]
    blocks, page, heads, f = specs["kv_cache"]["shape"]
    dr, d = f - kr, dn + f - kr
    if heads != 1 or dr <= 0 or any(v % 32 for v in (qr, n * d, f)):
        raise ValueError("Invalid model dimensions/NZ alignment")
    quant_dtype_c = case.get("quant_dtype_c", "mxfp8")
    if quant_dtype_c not in ("mxfp8", "hif8"):
        raise ValueError("quant_dtype_c must be mxfp8 or hif8")
    if not case["quant_mode_c"] and quant_dtype_c != "mxfp8":
        raise ValueError("hif8 output storage requires quant_mode_c=1")
    cache_dtype = (
        ("hif8" if quant_dtype_c == "hif8" else "fp8_e4m3")
        if case["quant_mode_c"]
        else "bf16"
    )
    expected = {
        "x": ([t, he], "fp8_e4m3", "ND"),
        "wqa": ([qr // 32, he, 32], "fp8_e4m3", "NZ"),
        "wqb": ([n * d // 32, qr, 32], "fp8_e4m3", "NZ"),
        "wkva": ([f // 32, he, 32], "fp8_e4m3", "NZ"),
        "wkb": ([n, dn, kr], "bf16", "ND"),
        "norm_weight_qa": ([qr], "float32", "ND"),
        "norm_weight_kva": ([kr], "float32", "ND"),
        "kv_cache": ([blocks, page, 1, f], cache_dtype, "PA_NZ"),
        "cache_index": ([t], "int64", "ND"),
    }
    if case["quant_mode_c"]:
        expected["qscale_kv"] = ([1], "float32", "ND")
    elif specs["qscale_kv"] is not None:
        raise ValueError("qscale_kv must be empty for quant_mode_c=0")
    for name, (shape, dtype, fmt) in expected.items():
        spec = specs[name]
        if spec is None or (spec["shape"], spec["dtype"], spec["format"].upper()) != (
            shape,
            dtype,
            fmt,
        ):
            raise ValueError(f"{name}: expected {shape}, {dtype}, {fmt}")
        physical = (
            pa_nz_storage_shape(shape, DTYPES[dtype])
            if name == "kv_cache"
            else tuple(shape)
        )
        if "storage_shape" in spec:
            storage_shape = spec["storage_shape"]
            if not isinstance(storage_shape, list) or any(
                type(value) is not int for value in storage_shape
            ):
                raise ValueError(f"{name}: wrong storage_shape, expected {physical}")
            if tuple(storage_shape) != physical:
                raise ValueError(f"{name}: wrong storage_shape, expected {physical}")
    for name, o, k in (
        ("descale_x", t, he),
        ("descale_wqa", qr, he),
        ("descale_wqb", n * d, qr),
        ("descale_wkva", f, he),
    ):
        spec = specs[name]
        if spec["shape"] != [o, (k + 63) // 64, 2] or spec["dtype"] != "fp8_e8m0":
            raise ValueError(f"{name}: incorrect logical MX scale shape/dtype")
        required_format = "ND" if name == "descale_x" else "NZ"
        if spec["format"].upper() != required_format:
            raise ValueError(f"{name}: external input format must be {required_format}")
        _scale_storage_shape(spec)
    indices = specs["cache_index"].get("values")
    if not isinstance(indices, list) or len(indices) != t:
        raise ValueError(
            "cache_index.values must contain T unique, in-bounds INT64 slots"
        )
    if any(type(value) is not int for value in indices) or any(
        not 0 <= value < blocks * page for value in indices
    ):
        raise ValueError(
            "cache_index.values must contain T unique, in-bounds INT64 slots"
        )
    if len(set(indices)) != t:
        raise ValueError(
            "cache_index.values must contain T unique, in-bounds INT64 slots"
        )
    cache = specs["kv_cache"]
    init, ranges = cache.get("init", "zeros"), cache.get("occupied_ranges", [])
    if init not in ("zeros", "history_pattern") or not isinstance(ranges, list):
        raise ValueError("Invalid cache initialization")
    if init == "zeros" and ranges:
        raise ValueError("Invalid cache initialization")
    end_previous = 0
    for span in ranges:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(type(v) is not int for v in span)
        ):
            raise ValueError("occupied_ranges must contain [start,end) integer pairs")
        start, end = span
        if not 0 <= start < end <= blocks * page or start < end_previous:
            raise ValueError("occupied_ranges must be sorted, disjoint and in bounds")
        end_previous = end
    if case["quant_mode_c"]:
        values = specs["qscale_kv"].get("values")
        if (
            not isinstance(values, list)
            or len(values) != 1
            or type(values[0]) not in (float, int)
        ):
            raise ValueError("qscale_kv.values must contain one numeric scalar")
        scale = torch.tensor(values[0], dtype=torch.float32).item()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("qscale_kv must be positive finite FP32")
    case.update(_model_dims((t, he, qr, kr, n, d, dn, dr, blocks, page)))
    return case


SINGLE_MIX_O11_CASE_ID = "case_014_aw1_c1"

NETWORK_CASE_IDS = (
    "case_013_aw1_c1",
    SINGLE_MIX_O11_CASE_ID,
    "gen_t0064_c1",
    "case_016_aw1_c1",
)

_NETWORK_CASE_CONFIG = {
    "case_013_aw1_c1": {
        "tokens": 8,
        "seed": 20260908,
        "blocks": 72,
        "indices": [5375, 768, 6657, 3711, 1536, 5505, 3583, 2432],
        "occupied_ranges": [[0, 128]],
        "scale_groups": 24,
        "qscale": 0.5,
        "notes": "network B=1, S=8; fragmented PA_NZ slots; tight WQB scale NZ",
    },
    "case_014_aw1_c1": {
        "tokens": 32,
        "seed": 20260908,
        "blocks": 72,
        "indices": [
            511,
            1535,
            1791,
            8831,
            5887,
            8703,
            2431,
            8447,
            4223,
            767,
            1663,
            383,
            8063,
            5759,
            8319,
            3199,
            2303,
            6271,
            127,
            2687,
            5119,
            9215,
            4735,
            5503,
            4991,
            7295,
            4095,
            6655,
            7807,
            6015,
            5247,
            2559,
        ],
        "occupied_ranges": [[0, 64]],
        "scale_groups": 32,
        "qscale": 0.5,
        "notes": "network B=4, S=8; fragmented PA_NZ slots; padded WQB scale NZ",
    },
    "gen_t0064_c1": {
        "tokens": 64,
        "seed": 2032,
        "blocks": 12,
        "indices": list(range(384, 448)),
        "occupied_ranges": [[0, 384]],
        "scale_groups": 32,
        "qscale": 1.0,
        "notes": "network B=8, S=8; page-contiguous PA_NZ slots",
    },
    "case_016_aw1_c1": {
        "tokens": 128,
        "seed": 20260908,
        "blocks": 21,
        "indices": list(range(128)),
        "occupied_ranges": [[128, 384], [512, 1280], [1408, 1920], [2176, 2688]],
        "scale_groups": 32,
        "qscale": 0.5,
        "notes": "network B=16, S=8; chunked-prefill PA_NZ slots",
    },
}


def _network_case(case_id):
    try:
        config = _NETWORK_CASE_CONFIG[case_id]
    except KeyError as exc:
        raise ValueError(f"Case not found: {case_id}") from exc
    tokens = config["tokens"]
    specs = {
        "x": {"shape": [tokens, 7168], "dtype": "fp8_e4m3", "format": "ND"},
        "wqa": {"shape": [48, 7168, 32], "dtype": "fp8_e4m3", "format": "NZ"},
        "wqb": {"shape": [576, 1536, 32], "dtype": "fp8_e4m3", "format": "NZ"},
        "wkva": {"shape": [18, 7168, 32], "dtype": "fp8_e4m3", "format": "NZ"},
        "wkb": {"shape": [96, 128, 512], "dtype": "bf16", "format": "ND"},
        "descale_x": {"shape": [tokens, 112, 2], "dtype": "fp8_e8m0", "format": "ND"},
        "descale_wqa": {
            "shape": [1536, 112, 2],
            "dtype": "fp8_e8m0",
            "format": "NZ",
            "storage_shape": [96, 112, 16, 2],
        },
        "descale_wqb": {
            "shape": [18432, 24, 2],
            "dtype": "fp8_e8m0",
            "format": "NZ",
            "storage_shape": [1152, config["scale_groups"], 16, 2],
        },
        "descale_wkva": {
            "shape": [576, 112, 2],
            "dtype": "fp8_e8m0",
            "format": "NZ",
            "storage_shape": [36, 112, 16, 2],
        },
        "norm_weight_qa": {"shape": [1536], "dtype": "float32", "format": "ND"},
        "norm_weight_kva": {"shape": [512], "dtype": "float32", "format": "ND"},
        "kv_cache": {
            "shape": [config["blocks"], 128, 1, 576],
            "dtype": "fp8_e4m3",
            "format": "PA_NZ",
            "init": "history_pattern",
            "occupied_ranges": config["occupied_ranges"],
        },
        "cache_index": {
            "shape": [tokens],
            "dtype": "int64",
            "format": "ND",
            "values": config["indices"],
        },
        "qscale_kv": {
            "shape": [1],
            "dtype": "float32",
            "format": "ND",
            "values": [config["qscale"]],
        },
    }
    return validate_case(
        {
            "case_id": case_id,
            "notes": config["notes"],
            "norm_eps": NORM_EPS,
            "quant_mode_aw": 1,
            "quant_mode_c": 1,
            "quant_dtype_c": "mxfp8",
            "seed": config["seed"],
            "specs": specs,
        }
    )


def load_case(case_id, quant_mode_c=1):
    """Return one network case in BF16 C0 or MXFP8 C1 output mode."""
    if not isinstance(case_id, str):
        raise TypeError("case_id must be a string")
    _check_modes(1, quant_mode_c)
    case = _network_case(case_id.strip())
    if quant_mode_c == 0:
        case["case_id"] = case["case_id"].replace("_c1", "_c0")
        case["quant_mode_c"] = 0
        case["specs"]["kv_cache"]["dtype"] = "bf16"
        case["specs"]["qscale_kv"] = None
        validate_case(case)
    return case


def make_inputs(case):
    validate_case(case)
    generator = torch.Generator(device="cpu").manual_seed(case.get("seed", SEED))
    result = {}
    for name in TENSOR_INPUTS:
        spec = case["specs"][name]
        if spec is None:
            continue
        shape, dtype = spec["shape"], DTYPES[spec["dtype"]]
        if name in ("cache_index", "qscale_kv"):
            value = torch.tensor(spec["values"], dtype=dtype).reshape(shape)
        elif name in DESCALES:
            o, g, _ = shape
            # Deterministic, nonunit, distinct channel/group/lane scales catch layout errors.
            raw = (
                125
                + (
                    3 * torch.arange(o)[:, None, None]
                    + 2 * torch.arange(g)[None, :, None]
                    + torch.arange(2)[None, None, :]
                )
                % 5
            ).to(torch.uint8)
            value = raw.contiguous().view(E8M0)
            if spec["format"].upper() == "NZ":
                physical = _scale_storage_shape(spec)
                value = pack_scale_pair_nz(
                    value, group_alignment=1 if physical[1] == g else 16
                )
        elif name == "kv_cache":
            logical = torch.zeros(shape, dtype=dtype)
            flat = _raw(logical).reshape(-1, shape[-1])
            features = torch.arange(shape[-1], dtype=torch.int64).unsqueeze(0)
            for start, end in spec.get("occupied_ranges", []):
                for offset in range(start, end, 512):
                    slots = torch.arange(offset, min(offset + 512, end))
                    values = (((slots[:, None] + features) % 15 - 7).float() / 8).to(
                        dtype
                    )
                    flat[slots] = _raw(values)
            value = pack_pa_nz(logical)
        elif name in WEIGHTS:
            o1, k, _ = shape
            value = pack_nz(
                (torch.randn(o1 * 32, k, generator=generator) * 0.5).to(FP8)
            )
        else:
            value = (
                torch.randn(shape, generator=generator) * (0.5 if name == "x" else 1.0)
            ).to(dtype)
        result[name] = value
    return result


def run_case(case):
    inputs = make_inputs(case)
    return inputs, cpu_golden(
        inputs,
        norm_eps=case["norm_eps"],
        quant_mode_aw=case["quant_mode_aw"],
        quant_mode_c=case["quant_mode_c"],
        quant_dtype_c=case.get("quant_dtype_c", "mxfp8"),
    )


def _metrics(chunks, *, atol, rtol=1e-3, max_mismatch_ratio=1e-3):
    count = bad_count = 0
    largest = total = dot = aa = ee = delta_sq = 0.0
    finite = True
    for actual, expected in chunks:
        a, e = actual.reshape(-1).double(), expected.reshape(-1).double()
        valid = torch.isfinite(a) & torch.isfinite(e)
        good = bool(valid.all())
        finite = finite and good
        diff = (a - e).abs()
        count += a.numel()
        bad_count += int((~valid | (diff > atol + rtol * e.abs())).sum())
        if not good:
            largest = total = math.inf
        elif a.numel():
            largest = max(largest, float(diff.max()))
            total += float(diff.sum())
            delta_sq += float(torch.dot(a - e, a - e))
            dot += float(torch.dot(a, e))
            aa += float(torch.dot(a, a))
            ee += float(torch.dot(e, e))
    ratio = bad_count / count if count else 0.0
    cosine = (
        (
            1.0
            if aa == 0 and ee == 0
            else 0.0
            if aa == 0 or ee == 0
            else max(-1.0, min(1.0, dot / math.sqrt(aa * ee)))
        )
        if finite
        else 0.0
    )
    relative_l2 = (
        math.sqrt(delta_sq) / max(math.sqrt(ee), 1e-12) if finite else math.inf
    )
    return dict(
        passed=finite and ratio <= max_mismatch_ratio,
        elements=count,
        max_abs_err=largest,
        mean_abs_err=total / count if count else 0.0,
        cosine_similarity=cosine,
        mismatch_ratio=ratio,
        mismatch_elements=bad_count,
        relative_l2=relative_l2,
        all_finite=finite,
        atol=atol,
        rtol=rtol,
        max_mismatch_ratio=max_mismatch_ratio,
    )


def _flat_chunks(a, e):
    a, e = a.reshape(-1), e.reshape(-1)
    for start in range(0, a.numel(), 262144):
        stop = start + 262144
        yield a[start:stop], e[start:stop]


def _relative_l2(actual, expected):
    actual, expected = actual.float(), expected.float()
    return float(
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1e-12)
    )


def _gather_cache(cache, index):
    return (
        _raw(cache)
        .reshape(-1)[_cache_offsets(cache, index)]
        .contiguous()
        .view(cache.dtype)
    )


def _gather_cache_chunks(actual_cache, expected_cache, index, chunk_size=128):
    for start in range(0, index.numel(), chunk_size):
        stop = start + chunk_size
        chunk_index = index[start:stop]
        yield (
            _gather_cache(actual_cache, chunk_index),
            _gather_cache(expected_cache, chunk_index),
        )


def compare_outputs(actual, expected, cache_index, *, quant_mode_c=0, qscale_kv=None):
    """Same-process CPU comparison, including encoded+dequantized FP8 outputs.

    c=1 requires each result's own descale_q and the original FP32 qscale_kv
    input. Encoded mismatch counts remain diagnostic because neighboring FP8
    codes and one-head scale boundary changes do not directly represent model
    accuracy. The final verdict follows the network test: reconstructed Q/KV
    relative L2 must be below 3%, Q cosine must exceed 0.999, every value must
    be finite, and untouched cache bytes must remain exactly unchanged.
    """
    _check_modes(1, quant_mode_c)
    keys = {"q", "kv_cache_out"} | ({"descale_q"} if quant_mode_c else set())
    dtype = FP8 if quant_mode_c else BF16
    for label, values in (("actual", actual), ("expected", expected)):
        if not isinstance(values, dict) or set(values) != keys:
            raise ValueError(f"{label}: expected output keys {sorted(keys)}")
        for name, value in values.items():
            if not isinstance(value, torch.Tensor) or value.dtype != (
                torch.float32 if name == "descale_q" else dtype
            ):
                raise ValueError(f"{label}.{name}: wrong tensor dtype")
    if any(actual[name].shape != expected[name].shape for name in keys):
        raise ValueError("Actual/expected output shapes differ")
    if expected["q"].ndim != 3 or min(expected["q"].shape) <= 0:
        raise ValueError("q must have positive shape [T,N,F]")
    t, n, f = expected["q"].shape
    cache = expected["kv_cache_out"]
    if cache.ndim != 4 or cache.shape[-1] != cache_c0(dtype):
        raise ValueError("Invalid physical PA_NZ output shape")
    if min(cache.shape) <= 0 or cache.shape[1] * cache.shape[3] != f:
        raise ValueError("Invalid physical PA_NZ output shape")
    a = {name: value.detach().cpu().contiguous() for name, value in actual.items()}
    e = {name: value.detach().cpu().contiguous() for name, value in expected.items()}
    if not isinstance(cache_index, torch.Tensor):
        raise ValueError("cache_index must be an INT64 tensor")
    index = cache_index.detach().cpu()
    _check_indices(index, cache.shape[0] * cache.shape[2], t)
    scalar = None
    if quant_mode_c:
        if tuple(e["descale_q"].shape) != (t, n):
            raise ValueError("descale_q must have shape [T,N]")
        for values in (a, e):
            _finite(values["descale_q"], "descale_q")
            if not bool((values["descale_q"] > 0).all()):
                raise ValueError("descale_q must be positive")
        if (
            not isinstance(qscale_kv, torch.Tensor)
            or qscale_kv.dtype != torch.float32
            or tuple(qscale_kv.shape) != (1,)
        ):
            raise ValueError(
                "c=1 comparison requires the original FP32 qscale_kv[1] input"
            )
        scalar = qscale_kv.detach().cpu()
        _finite(scalar, "qscale_kv")
        if scalar.item() <= 0:
            raise ValueError("qscale_kv must be positive")
    elif qscale_kv is not None:
        raise ValueError("c=0 does not use qscale_kv")
    atol = 1e-3 if quant_mode_c else 1e-8
    q_metrics = _metrics(_flat_chunks(a["q"], e["q"]), atol=atol)
    written = _metrics(
        _gather_cache_chunks(a["kv_cache_out"], e["kv_cache_out"], index), atol=atol
    )
    touched = {}
    for slot in index.tolist():
        touched.setdefault(slot // cache.shape[2], set()).add(slot % cache.shape[2])
    unchanged, elements = True, 0
    ab, eb = _raw(a["kv_cache_out"]), _raw(e["kv_cache_out"])
    for page in range(cache.shape[0]):
        rows = [r for r in range(cache.shape[2]) if r not in touched.get(page, set())]
        for start in range(0, len(rows), 16):
            stop = start + 16
            selected = rows[start:stop]
            elements += len(selected) * f
            if not torch.equal(ab[page, :, selected, :], eb[page, :, selected, :]):
                unchanged = False
    report = {
        "q_encoded" if quant_mode_c else "q": q_metrics,
        "kv_cache_written_encoded" if quant_mode_c else "kv_cache_written": written,
        "kv_cache_untouched": {"bitwise_equal": unchanged, "elements": elements},
    }
    passed = (
        q_metrics["all_finite"]
        and written["all_finite"]
        and q_metrics["relative_l2"] < NETWORK_RELATIVE_L2_LIMIT
        and written["relative_l2"] < NETWORK_RELATIVE_L2_LIMIT
        and q_metrics["cosine_similarity"] > NETWORK_Q_COSINE_MIN
        and unchanged
    )
    if quant_mode_c:
        scale_metrics = _metrics(
            _flat_chunks(a["descale_q"], e["descale_q"]), atol=1e-8
        )

        def q_dequant_chunks():
            for start in range(0, t, 4):
                stop = start + 4
                yield (
                    a["q"][start:stop].float() * a["descale_q"][start:stop, :, None],
                    e["q"][start:stop].float() * e["descale_q"][start:stop, :, None],
                )

        def cache_dequant_chunks():
            for actual_chunk, expected_chunk in _gather_cache_chunks(
                a["kv_cache_out"], e["kv_cache_out"], index
            ):
                yield (
                    actual_chunk.float() * scalar,
                    expected_chunk.float() * scalar,
                )

        q_dequant = _metrics(q_dequant_chunks(), atol=1e-3)
        cache_dequant = _metrics(cache_dequant_chunks(), atol=1e-3)
        report.update(
            descale_q=scale_metrics,
            q_dequantized=q_dequant,
            kv_cache_written_dequantized=cache_dequant,
        )
        passed = (
            scale_metrics["all_finite"]
            and q_dequant["all_finite"]
            and cache_dequant["all_finite"]
            and scale_metrics["relative_l2"] < NETWORK_RELATIVE_L2_LIMIT
            and q_dequant["relative_l2"] < NETWORK_RELATIVE_L2_LIMIT
            and cache_dequant["relative_l2"] < NETWORK_RELATIVE_L2_LIMIT
            and q_dequant["cosine_similarity"] > NETWORK_Q_COSINE_MIN
            and unchanged
        )
    report["acceptance"] = {
        "relative_l2_limit": NETWORK_RELATIVE_L2_LIMIT,
        "q_cosine_min": NETWORK_Q_COSINE_MIN,
        "cache_untouched_bitwise": True,
    }
    report["passed"] = passed
    return report


# ---------------------------------------------------------------------------
