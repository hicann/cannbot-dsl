---
title: get_block_num
api_name: get_block_num
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_block_num`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

获取本次 Kernel 启动配置的任务块（block）数量。

## 函数原型

```python
def get_block_num() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

返回本次启动 Kernel 时指定的 block 数量。

## 约束说明

无。

## 调用示例

```python
import cannbotdsl as cb
import torch
import torch_npu  # noqa: F401

@cb.kernel
def kernel(output):
    output[cb.get_block_idx()] = cb.get_block_num()

@cb.jit
def run(output):
    kernel[4](output)

output = torch.empty(4, dtype=torch.int64, device="npu")
run(output)
torch.npu.synchronize()
block_nums = output.cpu().tolist()
assert block_nums == [4, 4, 4, 4]
print(f"block_num: {block_nums[0]}")
print("get_block_num example passed")
```

### 预期结果

```text
block_num: 4
get_block_num example passed
```
