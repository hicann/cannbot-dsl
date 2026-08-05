# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tests for the VoxelConv kernel.

Formula:  C[N,Co,Ho,Wo] = Conv2D(x[N,Ci,Hi,Wi], filter[Co,CiG,Kh,Kw])

Coverage:
  * fp16 / bf16
  * 3x3 pad=1 baseline
  * stride/dilation asymmetric + asymmetric padding
  * all-padding M tiles (fully padded input)
  * groups=2
  * AscendC compile-only (dump + assert fill_l1_sync path)
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "samples", "voxel_conv"))

import cannbotdsl
from cannbotdsl import dtypes
from cannbotdsl.jit_runner import jit
from cannbotdsl.tensor import make_layout, make_pointer, make_tensor
from cannbotdsl.typing.types import MemLoc

from voxel_conv import VoxelConvKernel, voxel_conv


_INPUT_SHAPE = (1, 20, 5, 7)
_FILTER_SHAPE = (18, 20, 3, 3)
_TILE_SHAPE = (16, 16, 16)


def _make_gm(dtype, shape, address):
    return make_tensor(
        make_pointer(dtype, address, MemLoc.GM),
        make_layout(shape),
    )


@pytest.mark.ascendc_toolchain
def test_voxel_conv_full_pipeline_compiles(tmp_path, monkeypatch):
    monkeypatch.setenv("CANNBOTDSL_PIPE_STAGE", "compile")
    monkeypatch.setenv("CANNBOTDSL_DUMP_ASCENDC", "1")
    monkeypatch.setenv("CANNBOTDSL_DUMP_DIR", str(tmp_path))
    cannbotdsl.clear_compile_cache()
    conv = VoxelConvKernel(
        input_shape=_INPUT_SHAPE,
        filter_shape=_FILTER_SHAPE,
        padding=(1, 1, 1, 1),
    )

    @jit
    def compile_voxel_conv():
        gm_x = _make_gm(dtypes.float16, _INPUT_SHAPE, 0)
        gm_filter = _make_gm(dtypes.float16, _FILTER_SHAPE, 4096)
        gm_y = _make_gm(dtypes.float16, conv.spec.output_shape, 12288)
        conv.voxel_conv_kernel[1](gm_x, gm_filter, gm_y)

    try:
        compile_voxel_conv()
        ascendc_path = tmp_path / "compile_voxel_conv.asc"
        assert ascendc_path.exists()
        ascendc = ascendc_path.read_text()
        assert "asc_fill_l1_sync(" in ascendc
        assert "AscendC::Fill(" not in ascendc
        assert "asc_fill_l0a(" not in ascendc
    finally:
        cannbotdsl.clear_compile_cache()


def _assert_voxel_conv_accuracy(
    *,
    input_shape,
    filter_shape,
    stride=(1, 1),
    padding=(0, 0, 0, 0),
    dilation=(1, 1),
    groups=1,
    dtype=dtypes.float16,
):
    pytest.importorskip("torch_npu")
    import torch.nn.functional as torch_functional

    torch.manual_seed(0)
    torch_dtype = torch.float16 if dtype is dtypes.float16 else torch.bfloat16
    x_cpu = torch.randn(input_shape, dtype=torch_dtype) * 0.1
    filter_cpu = torch.randn(filter_shape, dtype=torch_dtype) * 0.1

    y_npu = voxel_conv(
        x_cpu.npu(),
        filter_cpu.npu(),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    torch.npu.synchronize()

    pad_top, pad_bottom, pad_left, pad_right = padding
    padded = torch_functional.pad(
        x_cpu.float(), (pad_left, pad_right, pad_top, pad_bottom)
    )
    expected = torch_functional.conv2d(
        padded,
        filter_cpu.float(),
        stride=stride,
        dilation=dilation,
        groups=groups,
    )
    torch.testing.assert_close(
        y_npu.cpu().float(), expected, atol=2e-2, rtol=2e-2
    )


@pytest.mark.npu
def test_voxel_conv_tail_accuracy():
    _assert_voxel_conv_accuracy(
        input_shape=_INPUT_SHAPE,
        filter_shape=_FILTER_SHAPE,
        padding=(1, 1, 1, 1),
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    ("input_shape", "filter_shape", "stride", "padding", "dilation", "groups"),
    [
        pytest.param(
            (1, 16, 7, 8),
            (16, 16, 2, 3),
            (2, 1),
            (1, 0, 2, 1),
            (2, 1),
            1,
            id="stride-dilation-asymmetric-pad",
        ),
        pytest.param(
            (1, 16, 1, 1),
            (16, 16, 1, 1),
            (1, 1),
            (20, 20, 0, 0),
            (1, 1),
            1,
            id="all-padding-m-tiles",
        ),
        pytest.param(
            (1, 32, 4, 4),
            (32, 16, 1, 1),
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
            2,
            id="groups-2",
        ),
    ],
)
def test_voxel_conv_boundary_accuracy(
    input_shape, filter_shape, stride, padding, dilation, groups
):
    _assert_voxel_conv_accuracy(
        input_shape=input_shape,
        filter_shape=filter_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


@pytest.mark.npu
def test_voxel_conv_bfloat16_accuracy():
    _assert_voxel_conv_accuracy(
        input_shape=(1, 16, 4, 4),
        filter_shape=(16, 16, 3, 3),
        padding=(1, 1, 1, 1),
        dtype=dtypes.bfloat16,
    )


@pytest.mark.npu
def test_voxel_conv_1x1_baseline():
    _assert_voxel_conv_accuracy(
        input_shape=(1, 32, 8, 8),
        filter_shape=(32, 32, 1, 1),
        stride=(1, 1),
        padding=(0, 0, 0, 0),
    )


@pytest.mark.npu
def test_voxel_conv_stride2_downsample():
    _assert_voxel_conv_accuracy(
        input_shape=(1, 16, 32, 32),
        filter_shape=(16, 16, 3, 3),
        stride=(2, 2),
        padding=(1, 1, 1, 1),
    )


@pytest.mark.npu
def test_voxel_conv_dilation2():
    _assert_voxel_conv_accuracy(
        input_shape=(1, 16, 16, 16),
        filter_shape=(16, 16, 3, 3),
        stride=(1, 1),
        padding=(2, 2, 2, 2),
        dilation=(2, 2),
    )


@pytest.mark.npu
def test_voxel_conv_large_feature_map():
    _assert_voxel_conv_accuracy(
        input_shape=(1, 64, 128, 128),
        filter_shape=(64, 64, 3, 3),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
    )


@pytest.mark.npu
def test_voxel_conv_multi_batch():
    _assert_voxel_conv_accuracy(
        input_shape=(17, 16, 16, 16),
        filter_shape=(16, 16, 3, 3),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
    )


@pytest.mark.npu
def test_voxel_conv_depthwise_groups():
    _assert_voxel_conv_accuracy(
        input_shape=(1, 16, 8, 8),
        filter_shape=(16, 1, 3, 3),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        groups=16,
    )


@pytest.mark.npu
def test_voxel_conv_asymmetric_kernel():
    _assert_voxel_conv_accuracy(
        input_shape=(1, 16, 16, 16),
        filter_shape=(16, 16, 1, 3),
        stride=(1, 1),
        padding=(0, 1, 0, 1),
    )


@pytest.mark.npu
def test_voxel_conv_5x5_kernel():
    _assert_voxel_conv_accuracy(
        input_shape=(1, 16, 16, 16),
        filter_shape=(16, 16, 5, 5),
        stride=(1, 1),
        padding=(2, 2, 2, 2),
    )


@pytest.mark.npu
def test_voxel_conv_non_aligned_channels():
    _assert_voxel_conv_accuracy(
        input_shape=(1, 20, 16, 16),
        filter_shape=(18, 20, 3, 3),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
    )
