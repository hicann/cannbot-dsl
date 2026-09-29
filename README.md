# CANNBot-DSL

> 基于 CANNBot-DSL 的 Ascend NPU 复杂算子示例集合。

[📖 概述](#概述) · [📦 算子列表](#算子列表) · [📂 目录结构](#目录结构) · [🔥 更新日志](CHANGELOG.md) · [📜 许可证](#许可证)

---

## 概述

CANNBot 是 [CANN](https://hiascend.com/software/cann) 社区的 Infra 智能体层，用 Agent 完成 AscendC/PyPTO/TileLang/Triton 等各类语言的算子开发、模型迁移与推理优化，并延伸至图模式、Runtime 等更多 CANN 开发场景。

本仓（cannbot-dsl）是其 DSL 仓，提供 Agent 亲和的编程范式；仓群还包括 [cannbot](https://gitcode.com/cann/cannbot)、[cannbot-skills](https://gitcode.com/cann/cannbot-skills)、[cannbot-knowledge](https://gitcode.com/cann/cannbot-knowledge)、[cann-bench](https://gitcode.com/cann/cann-bench)、[cannbot-sentry](https://gitcode.com/cann/cannbot-sentry) 等仓库，结构如下。

![CANNBot 仓群结构](docs/figures/cannbot-repo-map.png)

当前仓库中的 samples 使用 CANNBot 基于 CANNBot-DSL 生成，涵盖 VoxelConv、PointNet Set Abstraction、Flash Attention、Kimi Delta Attention 等复杂算子。本次开源样例代码，自定义开发、测试等功能将于近期发布，敬请期待。

项目面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），详见 [算子列表](#算子列表)。

## 算子列表

| 算子 | 公式 | 文档 |
| :--- | :--- | :--- |
| voxel_conv | $C[N, Co, Ho, Wo] = \text{VoxelConv}(x, filter)$ | [samples/voxel_conv](samples/voxel_conv) |
| flash_attn | $O = softmax(QK^T \cdot scale) V$ | [samples/flash_attn](samples/flash_attn) |
| qwen_sparse_attn | `block_size=128` 的分块稀疏注意力 | [samples/qwen_sparse_attn](samples/qwen_sparse_attn) |
| flash_attn_fp8_fullquant | FP8 全量化 Attention，支持 GQA 与分页 KV Cache | [samples/flash_attn_fp8_fullquant](samples/flash_attn_fp8_fullquant) |
| stem_indexer | 稀疏 Attention 选块，包含 AICPU metadata | [samples/stem_indexer](samples/stem_indexer) |
| qsa_indexer | 压缩 Key 稀疏索引，包含 AICPU metadata | [samples/qsa_indexer](samples/qsa_indexer) |
| flash_kda | Kimi Delta Attention prefill 融合算子 | [samples/flash_kda](samples/flash_kda) |
| flash_kda_metadata | FlashKDA 调度 metadata 生成（AICPU） | [samples/flash_kda_metadata](samples/flash_kda_metadata) |
| fused_recurrent_kda_snapshot | Kimi Delta Attention decode（1～8 token，状态快照） | [samples/fused_recurrent_kda_snapshot](samples/fused_recurrent_kda_snapshot) |
| matmul | $C[M,N] = A[M,K] @ B[N,K]^T$ | [samples/matmul/matmul](samples/matmul/matmul) |
| batch_matmul | $C[c\_batch, M, N] = A[a\_batch, M, K] @ B[b\_batch, N, K]^T$ | [samples/matmul/batch_matmul](samples/matmul/batch_matmul) |
| quant_matmul | $C[M,N] = Dequant(A)[M,K] @ Dequant(B)[N,K]^T$ | [samples/matmul/quant_matmul](samples/matmul/quant_matmul) |
| grouped_matmul | $y_i[m_i,n_i] = x_i[m_i,k_i] \times weight_i[k_i,n_i]$ | [samples/grouped_matmul](samples/grouped_matmul) |
| pointnet_sa | $\text{feat}[K, D_{out}] = \max_{j} \text{MLP}(\text{points}[K, j, D_{in}])$ | [samples/pointnet_sa](samples/pointnet_sa) |
| rms_norm | $y = x \cdot rstd \cdot \gamma$ | [samples/rms_norm](samples/rms_norm) |
| kv_compress_epilog | KV Cache 压缩、量化与按槽位原地更新 | [samples/kv_compress_epilog](samples/kv_compress_epilog) |

## 目录结构

```text
├── samples/            # 算子实现与使用说明
│   ├── voxel_conv/     # VoxelConv 卷积
│   ├── flash_attn/     # Flash Attention
│   ├── qwen_sparse_attn/ # Qwen 稀疏注意力（block_size=128）
│   ├── flash_attn_fp8_fullquant/ # FP8 全量化 Attention
│   ├── stem_indexer/   # Stem Indexer 与 metadata
│   ├── qsa_indexer/    # QSA Indexer 与 metadata
│   ├── flash_kda/      # Kimi Delta Attention
│   ├── flash_kda_metadata/ # FlashKDA 调度 metadata（AICPU）
│   ├── fused_recurrent_kda_snapshot/ # KDA decode 状态快照
│   ├── matmul/         # 矩阵乘
│   │   ├── matmul/         # 非量化矩阵乘
│   │   ├── batch_matmul/   # 非量化批量矩阵乘（batch 维广播）
│   │   └── quant_matmul/   # MXFP8/MXFP4 全量化矩阵乘
│   ├── grouped_matmul/ # 非量化分组矩阵乘
│   ├── pointnet_sa/    # PointNet Set Abstraction
│   ├── rms_norm/       # RmsNorm 归一化
│   └── kv_compress_epilog/ # KV Cache 压缩更新
├── test/               # 测试
│   ├── voxel_conv/
│   ├── flash_attn/
│   ├── flash_attn_fp8_fullquant/
│   ├── stem_indexer/
│   ├── qsa_indexer/
│   ├── flash_kda/
│   ├── flash_kda_metadata/
│   ├── fused_recurrent_kda_snapshot/
│   ├── matmul/
│   │   ├── matmul/
│   │   ├── batch_matmul/
│   │   └── quant_matmul/
│   ├── grouped_matmul/
│   ├── pointnet_sa/
│   ├── rms_norm/
│   └── kv_compress_epilog/
├── figures/
└── README.md
```

## 🔥 最新动态

最新更新与历史记录详见 [CHANGELOG.md](CHANGELOG.md)。

## 相关信息

- [许可证](LICENSE)
- 源文件头部包含完整的版权与许可声明，遵循 CANN Open Software License Agreement Version 2.0
