# CANNBot-DSL

> 基于 CANNBotDSL 的 Ascend NPU 算子示例集合，提供完整实现、精度测试与性能对比数据。

[📖 概述](#概述) · [🛠️ 环境部署](#环境部署) · [⚡️ 快速入门](#快速入门) · [📦 算子列表](#算子列表) · [🧪 测试](#测试) · [📂 目录结构](#目录结构) · [📜 许可证](#许可证)

---

## 概述

CANNBot-DSL 收录了一批基于 CANNBotDSL 实现的 Ascend NPU 算子示例。每个算子均包含完整的 host 侧 tiling 推导、device 侧 kernel 实现、面向 CPU 的精度 golden 与面向 NPU 的精度比对测试，以及与 CANN 内置算子的性能对比数据。

项目面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），旨在为开发者提供从算子定义、切分调度到精度性能验证的端到端参考。

## 环境部署

### 前置依赖

| 依赖 | 版本要求 |
| :--- | :--- |
| OS | Linux x86_64 |
| NPU | Ascend 950 |
| Python | 3.12 |
| CANN Toolkit | 9.1.0 |
| torch / torch_npu | 与 CANN 匹配 |
| pytest / pytest-xdist | 9.1.1 |

### 安装 CANN Toolkit

当前仓库验证通过的社区版 CANN Toolkit 如下：

| CANN 版本 | 验证结果 | 下载链接（x86_64） |
| :--- | :--- | :--- |
| `9.1.0` | ✅ PASS | [Ascend-cann_9.1.0_linux-x86_64.run](https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann_9.1.0_linux-x86_64.run) |

安装命令：

```bash
chmod +x Ascend-cann_9.1.0_linux-x86_64.run
./Ascend-cann_9.1.0_linux-x86_64.run --install --force --install-path=${install_path}
```

`${install_path}` 为安装路径，默认 `/usr/local/Ascend`。

### 配置环境变量

安装完成后执行：

```bash
source ${install_path}/ascend-toolkit/set_env.sh
```

请将 `${install_path}` 替换为实际安装目录，例如 `/usr/local/Ascend` 或 `${HOME}/Ascend`。

### 安装 CANNBotDSL

```bash
python -m pip install /absolute/path/to/cannbotdsl-*-cp312-cp312-manylinux_*_x86_64.whl
```

验证安装：

```bash
python -c 'import cannbotdsl; print(cannbotdsl.__file__)'
```

## 快速入门

以 `matmul_basic` 为例，计算 $C[M,N] = A[M,K] @ B[N,K]^T$：

```python
import torch
import torch_npu
from matmul_basic import matmul_basic

M, K, N = 1024, 1024, 1024
dtype = torch.float16

a = torch.randn(M, K, dtype=dtype).npu()   # (M, K)
b = torch.randn(N, K, dtype=dtype).npu()   # (N, K)

c = matmul_basic(a, b, transpose_a=False, transpose_b=True)
```

更多算子的接口参数与约束详见 [算子列表](#算子列表) 中各子目录的 README。

## 算子列表

| 算子 | 公式 | 文档 |
| :--- | :--- | :--- |
| flash_attn | $O = softmax(QK^T \cdot scale) V$ | [samples/flash_attn](samples/flash_attn) |
| matmul_basic | $C[M,N] = A[M,K] @ B[N,K]^T$ | [samples/matmul_basic](samples/matmul_basic) |
| rms_norm | $y = x \cdot rstd \cdot \gamma$ | [samples/rms_norm](samples/rms_norm) |

> **支持架构**：NPU ARCH 3510（Ascend 950PR / Ascend 950DT）

## 测试

测试使用 pytest 驱动，需在 NPU 环境下运行。运行单个算子的精度测试：

```bash
pytest test/flash_attn/test_flash_attn.py -v
pytest test/matmul_basic/test_matmul_basic.py -v
pytest test/rms_norm/test_rms_norm.py -v
```

## 目录结构

```text
├── samples/            # 算子实现与使用说明
│   ├── flash_attn/     # Flash Attention
│   ├── matmul_basic/   # 非量化矩阵乘
│   └── rms_norm/       # RmsNorm 归一化
├── test/               # pytest 精度测试
│   ├── flash_attn/
│   ├── matmul_basic/
│   └── rms_norm/
├── media/              # 性能对比图
└── README.md
```

## 相关信息

- [许可证](LICENSE)
- 源文件头部包含完整的版权与许可声明，遵循 CANN Open Software License Agreement Version 2.0
