---
title: get_core_id
api_name: get_core_id
category: system
api_group: kernel
layer: system
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# `get_core_id`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

`get_core_id` 获取当前执行单元所在的物理 AI Core ID，适用于按物理核标识进行诊断或资源映射。

## 函数原型

```python
def get_core_id() -> Int64: ...
```

## 参数说明

无参数。

## 返回值说明

返回当前物理 AI Core ID，类型为有符号 64 位整数。

## 约束说明

- `get_core_id()` 获取当前执行物理核的编号；`get_block_idx()` 获取当前 Kernel 任务中的逻辑核索引，用于多核任务切分。逻辑核索引由 Kernel 启动配置和执行模式决定，与物理核编号不一定一一对应。

## 调用示例

以下示例获取当前物理 AI Core ID，将其写入输出 Tensor，并在 Host 侧校验结果为非负整数。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import cannbotdsl as cb
import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.


@cb.kernel
def core_id_kernel(output):
    output[0] = cb.get_core_id()


class CoreIdOperator:
    @cb.jit
    def run(self, output):
        core_id_kernel[1](output)


def main():
    output = torch.empty((1,), dtype=torch.int64, device="npu:0")

    CoreIdOperator().run(output)
    torch.npu.synchronize()

    core_id = int(output.cpu()[0])
    assert core_id >= 0

    print("get_core_id example passed")
    print(f"core_id: {core_id}")


if __name__ == "__main__":
    main()
```

### 预期结果

```text
get_core_id example passed
core_id: <非负整数>
```
