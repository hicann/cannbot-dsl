---
title: get_squeeze_status
api_name: get_squeeze_status
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_squeeze_status`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

读取 `vsqueeze_and_storeunalign()` 操作后保存在 AR 特殊寄存器中的有效数据长度，即已连续搬出的有效数据字节数。

## 函数原型

```python
def get_squeeze_status() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

返回 squeeze 有效数据长度，单位为字节。

## 约束说明

- 必须在兼容的压缩/非对齐存储序列产生有效数据长度之后读取；独立调用时返回值没有确定含义。

## 调用示例

以下示例在统一缓冲区（Unified Buffer，UB）中压缩存储 4 个 `uint32` 元素，再读取并打印有效数据长度。

```python
import cannbotdsl as cb
import torch
import torch_npu  # noqa: F401

@cb.jit
def compact_store(ub_output):
    with cb.vf(mode="raw"):
        mask = cb.reg.create_mask(pattern="vl4", elem_bits=32)
        source = cb.reg.vdups(7, cb.dtypes.uint32, mask=mask)
        cursor = cb.reg.vstore_unalign_begin(ub_output)
        squeezed = cb.reg.vsqueeze_and_storeunalign_init(source, mask=mask)
        cb.reg.vsqueeze_and_storeunalign(ub_output, 0, squeezed, cursor)
        cb.reg.vsqueeze_and_storeunalign_finalize(ub_output, 0, cursor)

@cb.kernel
def kernel(status):
    ub_space = cb.UB.view(262144)
    ub_output = cb.make_buffer(
        ub_space[slice(0, 256),], dtype=cb.dtypes.uint32
    )
    compact_store(ub_output)
    status[0] = cb.get_squeeze_status()

@cb.jit
def run(status):
    kernel[1](status)

status = torch.empty(1, dtype=torch.int64, device="npu")
run(status)
torch.npu.synchronize()
valid_bytes = int(status.cpu()[0])
assert valid_bytes == 16
print(f"squeeze_valid_bytes: {valid_bytes}")
print("get_squeeze_status example passed")
```

### 预期结果

```text
squeeze_valid_bytes: 16
get_squeeze_status example passed
```
