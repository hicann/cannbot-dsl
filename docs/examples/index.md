# 样例导航

当前仓库包含以下算子样例。每个链接会进入仓库中的对应目录说明。

| 场景 | 样例 | 说明 |
| --- | --- | --- |
| 矩阵计算 | [MatMul](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/matmul/matmul) | 非量化矩阵乘 |
| 矩阵计算 | [quant_batch_matmul_mxfp8](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/matmul/quant_matmul) | MXFP8 全量化批量矩阵乘 |
| 矩阵计算 | [grouped_matmul](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/grouped_matmul) | 非量化分组矩阵乘 |
| 归一化 | [RMSNorm](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/rms_norm) | 均方根归一化 |
| 注意力 | [Flash Attention](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/flash_attn) | 注意力计算样例 |
| 注意力 | [Flash KDA](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/flash_kda) | Kimi Delta Attention 样例 |
| KV Cache | [kv_compress_epilog](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/kv_compress_epilog) | KV Cache 压缩、量化与按槽位更新 |
| 点云 | [PointNet SA](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/pointnet_sa) | PointNet Set Abstraction |
| 卷积 | [VoxelConv](https://gitcode.com/cann/cannbot-dsl/tree/master/samples/voxel_conv) | 体素卷积样例 |

## 阅读样例

建议依次查看目录 README、实现文件和对应测试。README 描述样例用途和约束，实现文件展示主要逻辑，测试文件给出输入构造与结果验证方式。
