# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""VoxelConv pipeline using movement ops plus ``matmul``.

Structure:
  1. VoxelConvKernel  - @kernel with multi-core flat binding + Cin reduction
  2. voxel_conv()     - torch interface

Formula:  C[N,Co,Ho,Wo] = Conv2D(x[N,Ci,Hi,Wi], filter[Co,CiG,Kh,Kw])

Multi-core: all (batch, group, M_tile, N_tile) work items are linearized
into a flat index and round-robin distributed across AIC cores. Each core
independently completes the full Cin reduction for its assigned tiles.
"""

import torch
from torch import as_tensor as from_torch_npu

from cannbotdsl import dtypes
from cannbotdsl.arch import get_block_idx, get_block_num
from cannbotdsl.channel import Channel
from cannbotdsl.conv import (
    conv2d_load_filter,
    conv2d_load_fmap,
    conv2d_load_im2col,
    conv2d_store_output,
    make_conv2d_spec,
)
from cannbotdsl.integer import Int64
from cannbotdsl.jit_runner import jit
from cannbotdsl.kernel_launcher import kernel
from cannbotdsl.math import matmul
from cannbotdsl.tensor import mem_copy
from cannbotdsl.typing.types import MemLoc, Tensor

__all__ = ["voxel_conv"]

_DEFAULT_TILE_SHAPE = (16, 16, 16)


class VoxelConvKernel:
    """Conv2D kernel with multi-core flat binding.

    tiling flow:
      make_conv2d_spec -> tile_count -> block_num = min(total_work, 32)

    Hardware constants for arch35:
      L1 = 512 KB, L0A = 64 KB, L0B = 64 KB, L0C = 256 KB
      AIC = 32 cores
    """

    def __init__(
        self,
        *,
        dtype=dtypes.float16,
        input_shape,
        filter_shape,
        tile_shape=_DEFAULT_TILE_SHAPE,
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        dilation=(1, 1),
        groups=1,
        block_num=32,
    ):
        self.dtype = dtype
        self.tile_shape = tile_shape
        self.spec = make_conv2d_spec(
            dtype=dtype,
            input_shape=input_shape,
            filter_shape=filter_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        m_tiles, n_tiles, ci_tiles = self.spec.tile_count(tile_shape)
        self.m_tiles = m_tiles
        self.n_tiles = n_tiles
        self.ci_tiles = ci_tiles
        total_work = self.spec.batch_size * groups * m_tiles * n_tiles
        self.block_num = min(total_work, block_num)

    @kernel
    def voxel_conv_kernel(self, gm_x: Tensor, gm_filter: Tensor, gm_y: Tensor):
        bm, bn, bc = self.tile_shape
        khkw = self.spec.kernel_height * self.spec.kernel_width

        fmap_l1 = Channel(
            MemLoc.L1,
            shape=self.spec.fmap_l1_shape(tile=(bm, bc)),
            dtype=self.dtype,
            depth=2,
            data_format="nd",
        )
        filter_l1 = Channel(
            MemLoc.L1,
            shape=(khkw * bc, bn),
            dtype=self.dtype,
            depth=2,
            data_format="zn",
        )
        l0a = Channel(
            MemLoc.L0A,
            shape=(bm, khkw * bc),
            dtype=self.dtype,
            depth=2,
            data_format="nz",
        )
        l0b = Channel(
            MemLoc.L0B,
            shape=(khkw * bc, bn),
            dtype=self.dtype,
            depth=2,
            data_format="zn",
        )
        l0c = Channel(
            MemLoc.L0C,
            shape=(bm, bn),
            dtype=dtypes.float32,
            depth=2,
            data_format="nz",
        )

        block_idx = get_block_idx()
        block_num = get_block_num()

        m_tiles = self.m_tiles
        n_tiles = self.n_tiles
        ci_tiles = self.ci_tiles
        batch_size = self.spec.batch_size
        n_groups = self.spec.groups

        mn = m_tiles * n_tiles
        g_mn = n_groups * mn
        total_work = batch_size * g_mn

        for work_idx in range(block_idx, total_work, block_num):
            batch = work_idx // g_mn
            rem1 = work_idx % g_mn
            group = rem1 // mn
            rem2 = rem1 % mn
            m_idx = rem2 // n_tiles
            n_idx = rem2 % n_tiles

            for ci_idx in range(Int64(ci_tiles)):
                tile = self.spec.tile(
                    tile=self.tile_shape,
                    coord=(m_idx, n_idx, ci_idx),
                    batch=batch,
                    group=group,
                )

                conv2d_load_fmap(fmap_l1, gm_x, tile)
                conv2d_load_filter(filter_l1, gm_filter, tile)

                conv2d_load_im2col(tile.l0a_view(l0a), fmap_l1, tile)
                mem_copy(tile.l0b_view(l0b), tile.l1b_view(filter_l1))

                matmul(
                    tile.l0c_view(l0c),
                    tile.l0a_view(l0a),
                    tile.l0b_view(l0b),
                    init=(ci_idx == 0),
                )

            output_tile = self.spec.tile(
                tile=self.tile_shape,
                coord=(m_idx, n_idx, 0),
                batch=batch,
                group=group,
            )
            conv2d_store_output(
                gm_y,
                output_tile.l0c_view(l0c),
                output_tile,
            )

    @jit
    def run(self, gm_x: Tensor, gm_filter: Tensor, gm_y: Tensor):
        self.voxel_conv_kernel[self.block_num](gm_x, gm_filter, gm_y)


def voxel_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    stride=(1, 1),
    padding=(1, 1, 1, 1),
    dilation=(1, 1),
    groups=1,
    tile_shape=_DEFAULT_TILE_SHAPE,
    block_num=32,
) -> torch.Tensor:
    """Torch-facing wrapper for the VoxelConv kernel.

    The kernel computes standard 2D convolution with NCHW input/output
    and OIHW filter layout. This wrapper delegates geometry to
    Conv2dSpec, converts torch NPU tensors to cannbotdsl tensors, and
    synchronously launches the JIT-compiled multi-core kernel.
    """
    dtype = dtypes.float16 if x.dtype == torch.float16 else dtypes.bfloat16
    op = VoxelConvKernel(
        dtype=dtype,
        input_shape=tuple(x.shape),
        filter_shape=tuple(weight.shape),
        tile_shape=tile_shape,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        block_num=block_num,
    )
    y = torch.empty(
        op.spec.output_shape,
        dtype=x.dtype,
        device=x.device,
    )
    op.run(
        from_torch_npu(x),
        from_torch_npu(weight),
        from_torch_npu(y),
    )
    torch.npu.synchronize()
    return y
