---
title: enable_hf32_trans
api_name: enable_hf32_trans
category: cube_compute
api_group: kernel
layer: cube
call_context: device
execution_unit: cube
status: experimental
since: 待追溯
---

# `enable_hf32_trans`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

本接口用于设置HF32模式舍入方式，使用该接口前需要先使用`enable_hf32`开启HF32模式。

## 函数原型

```python
def enable_hf32_trans(mode: int | ScalarValue) -> None: ...
```

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `mode` | 输入 | HF32舍入模式控制参数，支持如下2种取值：<br>&nbsp;&nbsp;&bull; `0`：FP32将以向最接近的值舍入，平局时向偶数舍入的方式舍入为HF32（NEAREST_EVEN）。<br>&nbsp;&nbsp;&bull; `1`：FP32将以向最接近的值舍入，平局时远离零的方式舍入为HF32（NEAREST_AWAY）。<br>除整数常量外，也可传入取值等价的`UInt32`标量。 |

## 返回值说明

- 无返回值。
## 流水类型

`PIPE_S`

## 约束说明

- 本接口需在矩阵乘加指令（`matmul`）执行前调用，以此来确保模式配置在矩阵乘加计算过程中生效。
- 本接口配置的舍入模式仅在HF32模式开启期间生效，需先调用`enable_hf32`开启HF32模式再调用本接口，否则会报错。
- 本接口与[`set_hf32_round_mode`](./set-hf32-round-mode.md)均用于配置HF32模式的舍入方式：`set_hf32_round_mode`使用编译期枚举常量，本接口可传入运行期标量。
- 舍入模式配置后会持续生效，HF32模式关闭后再次开启仍将沿用上次的舍入模式配置，如需切换舍入模式，请重新调用本接口。

## 调用示例

将代码保存为`enable_hf32_trans.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import cannbotdsl as cb
import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.


@cb.kernel
def enable_hf32_trans_kernel(a, b, c):
    l1a = cb.Channel(cb.MemLoc.L1, shape=(16, 16), dtype=a.dtype, depth=1)
    l1b = cb.Channel(cb.MemLoc.L1, shape=(16, 16), dtype=b.dtype, depth=1)
    l0a = cb.Channel(cb.MemLoc.L0A, shape=(16, 16), dtype=a.dtype, depth=1)
    l0b = cb.Channel(cb.MemLoc.L0B, shape=(16, 16), dtype=b.dtype, depth=1)
    l0c = cb.Channel(cb.MemLoc.L0C, shape=(16, 16), dtype=cb.dtypes.float32, depth=1)

    nd2nz = cb.make_copy_engine(format_transform="nd2nz")
    cb.mem_copy(l1a.produce(), a, engine=nd2nz)
    cb.mem_copy(l1b.produce(), b, engine=nd2nz)

    # 舍入方式配置接口不产生数据结果，其作用体现在后续matmul的计算过程中。
    cb.cube.enable_hf32()
    cb.cube.enable_hf32_trans(0)

    cb.mem_copy(l0a.produce(), l1a.consume())
    cb.mem_copy(l0b.produce(), l1b.consume())
    cb.matmul(l0c.produce(), l0a.consume(), l0b.consume(), init=True)
    cb.mem_copy(c, l0c.consume())


@cb.jit
def run(a, b, c):
    enable_hf32_trans_kernel[1](a, b, c)


a = torch.arange(256, dtype=torch.float16, device="npu:0").reshape(16, 16) / 16.0
b = torch.arange(256, dtype=torch.float16, device="npu:0").reshape(16, 16) / 16.0
c = torch.empty((16, 16), dtype=torch.float32, device="npu:0")

run(a, b, c)
torch.npu.synchronize()

torch.testing.assert_close(c.cpu(), a.cpu().float() @ b.cpu().float().T, rtol=1e-2, atol=1e-2)
print("enable_hf32_trans example passed")
```

### 预期结果

```text
enable_hf32_trans example passed
```
