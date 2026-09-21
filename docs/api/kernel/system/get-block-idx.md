---
title: get_block_idx
api_name: get_block_idx
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_block_idx`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

获取当前任务块（block）在本次 Kernel 启动任务中的索引。常用于多核任务的数据切分和输出定位。

在 Mix 1:2 模式下，两个 Vector Core 的 block 索引相同；如需区分，可使用 `get_block_idx() * get_subblock_dim() + get_subblock_id()` 获取 Vector Core 的逻辑索引。

## 函数原型

```python
def get_block_idx() -> Int64: ...
```

## 参数说明

无。

## 返回值说明

返回 `[0, get_block_num())` 范围内的 block 索引。

## 约束说明

无。

## 调用示例

```python
import cannbotdsl as cb
import torch
import torch_npu  # noqa: F401

@cb.kernel
def kernel(output):
    index = cb.get_block_idx()
    output[index] = index

@cb.jit
def run(output):
    kernel[4](output)

output = torch.full((4,), -1, dtype=torch.int64, device="npu")
run(output)
torch.npu.synchronize()
block_indices = output.cpu().tolist()
assert block_indices == [0, 1, 2, 3]
print(f"block_indices: {block_indices}")
print("get_block_idx example passed")
```

### 预期结果

```text
block_indices: [0, 1, 2, 3]
get_block_idx example passed
```
