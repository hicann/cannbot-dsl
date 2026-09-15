# flash_kda_metadata

## 产品支持情况

- Ascend 950：支持。

## 功能说明

`flash_kda_metadata` 是 FlashKDA 配套的独立 AICPU 调度算子。它根据输入 shape、布局、有效序列长度和设备 AIC 核数生成调度 metadata，供 [`flash_kda`](../flash_kda/README.md) 消费。实现位于 [`flash_kda_metadata.py`](flash_kda_metadata.py)。

该算子不计算 attention 输出，也不读取 `q`、`v` 或 `initial_state` 的数值。它将每条有效序列按 64 个 token 划分为 chunk，并生成以下信息：

- 每轮参与调度的 batch、chunk 起点和 chunk 数量。
- Stage 1、Stage 2 的任务前缀和。
- 各 AIC 核负责的连续任务区间。
- 有效序列长度、token 起点和物理 batch 下标。

## 函数原型

```python
flash_kda_metadata.flash_kda_metadata.flash_kda_metadata(
    q: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    layout_qkv: str,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor
```

模块路径为 `flash_kda_metadata.flash_kda_metadata`。

## 参数说明

`B` 为逻辑 batch 数，`Nqk` 为 Query/Key 头数，`Nv` 为 Value 头数，`S` 为每个 batch 的存储序列长度，`T` 为 packed 序列的总 token 数，`D = 128`。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `q` | Tensor | 必选 | 用于确定 Query/Key 头数、存储序列长度和布局；不读取数值。 | bfloat16 | BNSD：`[B, Nqk, S, D]`；其他布局见下表 |
| `v` | Tensor | 必选 | 用于确定 Value 头数和布局；不读取数值。 | bfloat16 | BNSD：`[B, Nv, S, D]`；其他布局见下表 |
| `initial_state` | Tensor | 必选 | 用于确定逻辑 batch 数和状态维度；不读取数值。 | float32 | `[B, Nv, D, D]` |
| `layout_qkv` | str | 必选 | 输入布局，取值为 `"BNSD"`、`"BSND"` 或 `"TND"`。 | - | - |
| `cu_seqlens` | Tensor | 可选 | 各序列边界的前缀和；TND 必传，BNSD/BSND 可选。 | int32 | `[B + 1]` |

各布局对应的输入 shape：

| `layout_qkv` | `q` | `v` |
|:---|:---|:---|
| `BNSD` | `[B, Nqk, S, D]` | `[B, Nv, S, D]` |
| `BSND` | `[B, S, Nqk, D]` | `[B, S, Nv, D]` |
| `TND` | `[T, Nqk, D]` | `[T, Nv, D]` |

## 返回值说明

返回当前设备上的一维连续 int32 metadata 张量。算子在当前 stream 上生成 metadata；调用 `flash_kda` 时应直接传入该张量。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `metadata` | Tensor | 必选 | FlashKDA 的调度信息；只有 header、offset 表和活动轮次记录有定义。 | int32 | `[M]`，容量 `M` 由输入 shape 决定 |

header 中的 `status` 等于 `STATUS_OK` 时，调度信息有效。`decode_metadata(metadata.cpu().tolist())` 可用于测试和诊断，返回 header、各轮任务及核间分区；将 metadata 复制到 CPU 会同步设备。

## 约束说明

- 要求 `D = 128`、`B >= 1`、`S > 0` 或 `T > 0`。
- 支持 GQA，要求 `1 <= Nqk <= Nv`、`Nv % Nqk == 0`，且 `Nv <= WORKSPACE_SLOTS`；当前 `WORKSPACE_SLOTS` 为 861。
- `q`、`v` 和 `initial_state` 必须连续并位于同一 NPU，且满足参数表中的 dtype。传入 `cu_seqlens` 时，它也必须连续并位于同一 NPU。
- BNSD/BSND 的物理 batch 数必须等于 `initial_state` 的 `B`。不传 `cu_seqlens` 时，每条序列的有效长度均为 `S`。
- BNSD/BSND 可传 `cu_seqlens` 表示各 batch 的有效长度。它必须从 0 开始，相邻差值位于 `[1, S]`。
- TND 必须传 `cu_seqlens`。它必须从 0 开始并严格递增，最后一项等于 `T`。
- 算子最多使用 32 个 AIC。设备 AIC 核数、workspace slots 或调度算法变化后，已有 metadata 不再适用。

### metadata 复用

整网可在 forward 开始时生成一次 metadata，并在调度配置相同的层间复用。

当 `B`、物理 batch 数、`S/T`、`Nv`、布局、`cu_seqlens` 内容、设备 AIC 核数、workspace slots 或内置调度 cost 公式版本变化时，必须重新生成 metadata。`Nqk` 和 Q/K/V/g/beta/state 的数值不影响 metadata；每次调用仍须满足当前输入的 shape 和 GQA 约束。

## 调用示例

```python
import torch
import torch_npu

from flash_kda.flash_kda import flash_kda
from flash_kda_metadata.flash_kda_metadata import flash_kda_metadata

B, Nqk, Nv, S, D = 1, 3, 3, 8192, 128
q = torch.randn(B, Nqk, S, D).bfloat16().npu()
k = torch.randn(B, Nqk, S, D).bfloat16().npu()
v = torch.randn(B, Nv, S, D).bfloat16().npu()
g = torch.randn(B, Nv, S, D).bfloat16().npu()
beta = torch.randn(B, Nv, S).bfloat16().npu()
initial_state = torch.zeros(B, Nv, D, D, dtype=torch.float32).npu()
A_log = torch.zeros(Nv, dtype=torch.float32).npu()
dt_bias = torch.zeros(Nv, D, dtype=torch.float32).npu()

metadata = flash_kda_metadata(q, v, initial_state, "BNSD", None)
out, final_state = flash_kda(q, k, v, g, beta, D ** -0.5, initial_state, A_log, dt_bias, -5.0, "BNSD", metadata)
```

## 精度测试

在仓库根目录执行独立测试。测试只调用公开的 `flash_kda_metadata` 接口，不加载 FlashKDA 计算算子：

```bash
python -m pytest -q test/flash_kda_metadata/test_flash_kda_metadata.py
```

共 8 个 NPU 用例，覆盖 BNSD/BSND dense、TND packed 变长尾块、BNSD padded 变长尾块、858/861/864 个 chunk task 的容量边界，以及 `Nv=5` 时的 860+80 多轮调度。

测试按输入长度独立检查每个逻辑 chunk 恰好出现一次，并核对 token 起点、物理 batch、有效长度、任务前缀和及连续核区间。尾块覆盖 15/16/17/63/64/65/127/128/129，多轮场景覆盖活跃 batch 减少和容量余数。当前测试不包含 host/x86 执行、编译、缓存或 ABI 合约测试。
