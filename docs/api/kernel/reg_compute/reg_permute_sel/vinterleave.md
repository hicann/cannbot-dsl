---
title: vinterleave
api_name: vinterleave
category: reg_compute
api_group: kernel
layer: register
call_context: device
execution_unit: vector
status: experimental
since: 待追溯
---

# `vinterleave`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

将两个源操作数中的数据交织，结果作为返回值返回。根据操作数类型，本接口提供两种功能模式：

- **操作数为矢量数据寄存器：** 以元素为单位进行交织。`src0`和`src1`前半部分的元素依次交替写入`dst0`，后半部分的元素依次交替写入`dst1`。以`dtypes.int8`类型为例，交织过程如下图所示。
- **操作数为掩码寄存器：** 按掩码设置接口设置的掩码元素位宽确定交织粒度：元素位宽为8bit时以1bit为一组，16bit时以2bit为一组，32bit时以4bit为一组。

**图 1** dtypes.int8类型交织过程

![](../../figures/vinterleave.png)

本接口仅在AIV上生效。

## 函数原型

```python
def vinterleave(src0, src1) -> tuple[RawVReg, RawVReg]: ...
```

**支持的数据类型：**

`dtypes.int8`、`dtypes.uint8`、`dtypes.hifloat8`、`dtypes.float8_e8m0`、`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`、`dtypes.int16`、`dtypes.uint16`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.float32`。

## 参数说明

**表** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| `src0` | 输入 | 源操作数（矢量数据寄存器或掩码寄存器）。 |
| `src1` | 输入 | 源操作数（矢量数据寄存器或掩码寄存器）。数据类型须与`src0`一致。 |

## 返回值说明

- 返回`(dst0, dst1)`，为交织得到的两个结果寄存器：`src0`和`src1`前半部分的元素依次交替写入`dst0`，后半部分的元素依次交替写入`dst1`。

## 约束说明

### 通用约束

- 源操作数和目的操作数为矢量数据寄存器或掩码寄存器。

### 寄存器约束

- 本接口需在`cb.vf()`作用域内调用。
- `src0`和`src1`可以为同一个矢量数据寄存器或掩码寄存器。
- `dst0`和`dst1`不能为同一个矢量数据寄存器或掩码寄存器。
- 源操作数和目的操作数可以使用同一个矢量数据寄存器或掩码寄存器。

## 调用示例

将代码保存为`vinterleave.py`后，可通过`python`命令运行。

以下调用示例代码仅Ascend 950PR&950DT系列产品支持。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import cannbotdsl as cb
import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.


@cb.kernel
def vinterleave_kernel(src0, src1, dst):
    buf0 = cb.Channel(cb.MemLoc.UB, (64,), src0.dtype, depth=1)
    buf1 = cb.Channel(cb.MemLoc.UB, (64,), src1.dtype, depth=1)
    out = cb.Channel(cb.MemLoc.UB, (128,), dst.dtype, depth=1)

    cb.mem_copy(buf0.produce(), src0)
    cb.mem_copy(buf1.produce(), src1)

    in0 = buf0.consume()
    in1 = buf1.consume()
    res = out.produce()
    with cb.vf(mode="simd"):
        mask = cb.reg.full_mask()
        low, high = cb.reg.vinterleave(cb.reg.vload(in0, 0), cb.reg.vload(in1, 0))
        cb.reg.vstore(res, 0, low, mask)
        cb.reg.vstore(res, 64, high, mask)

    cb.mem_copy(dst, out.consume())


@cb.jit
def run(src0, src1, dst):
    vinterleave_kernel[1](src0, src1, dst)


src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
src1 = src0 + 100.0
dst = torch.empty((128,), dtype=torch.float32, device="npu:0")

run(src0, src1, dst)
torch.npu.synchronize()

a, b = src0.cpu(), src1.cpu()
expected = torch.cat([torch.stack([a[:32], b[:32]], 1).flatten(),
                      torch.stack([a[32:], b[32:]], 1).flatten()])
torch.testing.assert_close(dst.cpu(), expected)
print("vinterleave example passed")
print(f"first={float(dst.cpu()[0]):.1f}, last={float(dst.cpu()[-1]):.1f}")
```

### 预期结果

```text
vinterleave example passed
first=0.0, last=163.0
```
