# CANNBot-DSL

> 基于 CANNBot-DSL 的 Ascend NPU 复杂算子示例集合。

[📖 概述](#概述) · [📦 算子列表](#算子列表) · [📂 目录结构](#目录结构) · [📜 许可证](#许可证)

---

## 概述

当前仓库中的 samples 使用 CANNBot 基于 CANNBot-DSL 开发，涵盖 Flash Attention、Conv2D、Kimi Delta Attention 等复杂算子。本次仅开源样例代码，自定义开发、测试等功能将于近期发布，敬请期待。

项目面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），详见 [算子列表](#算子列表)。

## 算子列表

| 算子 | 公式 | 文档 |
| :--- | :--- | :--- |
| conv2d | $C[N, Co, Ho, Wo] = \text{Conv2D}(x, filter)$ | [samples/conv2d](samples/conv2d) |
| flash_attn | $O = softmax(QK^T \cdot scale) V$ | [samples/flash_attn](samples/flash_attn) |
| flash_kda | Kimi Delta Attention prefill 融合算子 | [samples/flash_kda](samples/flash_kda) |
| matmul | $C[M,N] = A[M,K] @ B[N,K]^T$ | [samples/matmul](samples/matmul) |
| rms_norm | $y = x \cdot rstd \cdot \gamma$ | [samples/rms_norm](samples/rms_norm) |

> **支持架构**：NPU ARCH 3510（Ascend 950PR / Ascend 950DT）

## 目录结构

```text
├── samples/            # 算子实现与使用说明
│   ├── conv2d/         # Conv2D 卷积
│   ├── flash_attn/     # Flash Attention
│   ├── flash_kda/      # Kimi Delta Attention
│   ├── matmul/         # 非量化矩阵乘
│   └── rms_norm/       # RmsNorm 归一化
├── test/               # 测试
│   ├── conv2d/
│   ├── flash_attn/
│   ├── flash_kda/
│   ├── matmul/
│   └── rms_norm/
├── media/
└── README.md
```

## 相关信息

- [许可证](LICENSE)
- 源文件头部包含完整的版权与许可声明，遵循 CANN Open Software License Agreement Version 2.0
