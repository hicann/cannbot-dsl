# Conv2D

基于 cannbotdsl 实现的 Conv2D 算子，支持 float16/bfloat16 数据类型，面向 Ascend NPU。

## 算子介绍

计算公式：

$$
C[N, Co, Ho, Wo] = \text{Conv2D}(x[N, Ci, Hi, Wi],\ filter[Co, CiG, Kh, Kw])
$$

| 特性与约束 | 说明 |
| :--------- | :------------------------------------------------ |
| 数据类型   | float16、bfloat16（输入/权重一致） |
| 数据格式   | NCHW 输入/输出，OIHW 权重 |
| stride    | 支持 H/W 独立 stride，范围 [1, 63] |
| padding   | 支持四侧独立 padding，范围 [0, 255] |
| dilation  | 支持 H/W 独立 dilation，范围 [1, 255] |
| groups    | 支持 groups > 1 的分组卷积 |
| tile 形状 | BM=16，BN 为 16 的倍数，BC 为 16 或 32 |
| 多核并行   | 扁平分核：所有 (batch, group, M, N) tile 线性化后 round-robin 分配 |
| 支持架构   | NPU ARCH 3510（Ascend 950DT / Ascend 950PR） |

算子使用 `Conv2dSpec` 进行 host 侧卷积几何推导，device kernel 采用多核扁平分核调度：

- **GM → L1**：fmap 通过 `conv2d_load_fmap`（DN2NZ）加载到 L1，filter 通过 `conv2d_load_filter`（DN2NZ）打包到 L1。
- **L1 → L0A**：通过 `conv2d_load_im2col`（Load3D）将 L1 fmap 窗口展开为 L0A 矩阵 tile。
- **L1 → L0B**：通过 `mem_copy` 将 L1 filter 搬到 L0B。
- **L0A × L0B → L0C**：使用 `matmul` 进行矩阵乘累加，沿 Cin 方向 reduction。
- **L0C → GM**：通过 `conv2d_store_output`（NZ2DN）将 L0C 结果写回 GM 输出。

数据流为 GM → L1（MTE2, DN2NZ）→ L0A（MTE1, Load3D）/ L0B（MTE1）→ L0C（M, MMAD）→ GM（FIX, NZ2DN）。

实现详见 `conv2d.py`。

## 快速开始

```python
import torch
import torch_npu
from conv2d import conv2d

x = torch.randn(1, 64, 128, 128, dtype=torch.float16).npu()
weight = torch.randn(64, 64, 3, 3, dtype=torch.float16).npu()

y = conv2d(x, weight, stride=(1, 1), padding=(1, 1, 1, 1))
```

`conv2d()` 参数说明：

| 参数 | shape | dtype | 说明 |
| :--- | :----: | :---: | :--- |
| `x` | (N, Ci, Hi, Wi) | float16/bfloat16 | 输入特征图 |
| `weight` | (Co, CiG, Kh, Kw) | float16/bfloat16 | 卷积权重 |
| `stride` | - | (int, int) | H/W 方向步长 |
| `padding` | - | (int, int, int, int) | top/bottom/left/right padding |
| `dilation` | - | (int, int) | H/W 方向空洞 |
| `groups` | - | int | 分组数 |
| `tile_shape` | - | (int, int, int) | (BM, BN, BC) tile 形状，默认 (16, 16, 16) |
| `block_num` | - | int | 最大核数，默认 32 |
| 返回值 `y` | (N, Co, Ho, Wo) | 同输入 | 卷积输出 |

## 精度测试

测试代码位于 `test/conv2d/test_conv2d.py`，使用 pytest 驱动，运行命令如下：

```bash
pytest test/conv2d/test_conv2d.py -v
```
