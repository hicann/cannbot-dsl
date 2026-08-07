# PointNet Set Abstraction

基于 CANNBot-DSL 实现的 PointNet++ Set Abstraction 层中的 shared MLP + max-pool 模块，支持 float16/bfloat16 数据类型，面向 Ascend NPU。

## 背景

PointNet++（Qi et al., 2017）是点云层次化特征学习的代表性网络。其核心模块 Set Abstraction (SA) 通过 Farthest Point Sampling 采样质心点、Ball Query 分组邻居点、shared MLP 提取逐点特征、max-pooling 聚合为质心特征，实现从不规则点云到结构化特征的提取。

SA 层中的 shared MLP 对每个点独立施加相同的全连接变换（权重共享），等价于 1x1 卷积。变换后的特征沿点维度做 max-pooling，生成每个质心点的单一特征向量。这一"逐点变换 + 对称聚合"的设计使网络对点云的排列顺序不变。

## 算子介绍

计算公式：

$$
\text{feat}[K, D_{out}] = \max_{j \in \text{group}} \; \text{MLP}(\text{points}[K, j, D_{in}])
$$

其中 MLP 为 shared weight 全连接：$\text{MLP}(x) = W[D_{out}, D_{in}] \cdot x$，等价于 1x1 卷积。

| 特性与约束 | 说明 |
| :--------- | :------------------------------------------------ |
| 数据类型   | float16、bfloat16 |
| 输入格式   | points: (K, N_per_group, D_in), weight: (D_out, D_in) |
| 输出格式   | (K, D_out) |
| MLP 实现   | 1x1 卷积，映射为 batched matmul (M=D_out, N=K*N_per_group, K=Ci) |
| 聚合方式   | max-pooling (torch.max，NPU 原生算子) |
| 多核并行   | slide window 多核调度，与 matmul_basic 一致 |
| 支持架构   | NPU ARCH 3510（Ascend 950DT / Ascend 950PR） |

shared MLP 的 matmul 部分使用 CANNBot-DSL 的 Channel + matmul 实现，数据流为 GM -> L1（MTE2, nd2nz）-> L0A/L0B（MTE1）-> L0C（M, MMAD）-> GM（FIXPIPE）。max-pooling 聚合由 torch NPU 原生算子完成。

实现详见 `pointnet_sa.py`。

## 快速开始

```python
import torch
import torch_npu
from pointnet_sa import pointnet_sa

K, N_per_group, D_in, D_out = 512, 64, 64, 128
points = torch.randn(K, N_per_group, D_in, dtype=torch.float16).npu()
weight = torch.randn(D_out, D_in, dtype=torch.float16).npu()

feat = pointnet_sa(points, weight)
```

`pointnet_sa()` 参数说明：

| 参数 | shape | dtype | 说明 |
| :--- | :----: | :---: | :--- |
| `points` | (K, N_per_group, D_in) | float16/bfloat16 | 分组后的局部点集 |
| `weight` | (D_out, D_in) | float16/bfloat16 | shared MLP 权重 |
| 返回值 `feat` | (K, D_out) | 同输入 | 每个质心的聚合特征 |

## 精度测试

测试代码位于 `test/pointnet_sa/test_pointnet_sa.py`，使用 pytest 驱动，运行命令如下：

```bash
pytest test/pointnet_sa/test_pointnet_sa.py -v
```
