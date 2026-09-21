---
title: get_vf_len
api_name: get_vf_len
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_vf_len`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

获取当前向量函数（Vector Function）一次处理的向量长度（Vector Length，VL）。

## 函数原型

```python
def get_vf_len() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

返回向量长度。Ascend 950PR/950DT 当前返回 `256`。

## 约束说明

无。

## 调用示例

```python
import cannbotdsl as cb
import torch
import torch_npu  # noqa: F401

@cb.kernel
def kernel(output):
    output[0] = cb.get_vf_len()

@cb.jit
def run(output):
    kernel[1](output)

output = torch.empty(1, dtype=torch.int64, device="npu")
run(output)
torch.npu.synchronize()
vf_len = int(output.cpu()[0])
assert vf_len == 256
print(f"vf_len: {vf_len}")
print("get_vf_len example passed")
```

### 预期结果

```text
vf_len: 256
get_vf_len example passed
```
