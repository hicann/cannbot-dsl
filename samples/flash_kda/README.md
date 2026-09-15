# flash_kda

## 产品支持情况

- Ascend 950：支持。

## 功能说明

`flash_kda` 基于 CANNBot-DSL 实现 Kimi Delta Attention 的 prefill 前向计算，返回序列输出和最终递归状态。实现位于 [`flash_kda.py`](flash_kda.py)。

调用时先生成调度 metadata，再执行计算：

1. [`flash_kda_metadata`](../flash_kda_metadata/README.md)：独立的 AICPU 算子，根据输入 shape、布局、有效序列长度和设备核数生成调度信息。
2. `flash_kda`：消费 metadata，启动 AICore 算子，以 group chunk 为粒度分阶段循环计算。

### 计算过程

先对输入进行归一化与门控激活：

$$
Q=\operatorname{L2Norm}(Q_{\mathrm{raw}}),\qquad
K=\operatorname{L2Norm}(K_{\mathrm{raw}}),
$$

$$
g=\mathrm{lower\_bound}\cdot
\operatorname{sigmoid}\left(\exp(\mathrm{A\_log})\cdot(g_{\mathrm{raw}}+\mathrm{dt\_bias})\right),
\qquad
\beta=\operatorname{sigmoid}(\beta_{\mathrm{raw}}).
$$

下标 raw 表示接口传入的原始张量。`L2Norm` 沿最后一维归一化，epsilon 为 `1e-6`；后续公式中的 $Q$、$K$、$g$、$\beta$ 均为上述处理后的结果。

令 $\gamma_t=\exp(\sum_{i\le t}g_i)$，并定义：

$$
\tilde K=\gamma\odot K,\qquad
\bar K=\gamma^{-1}\odot K,\qquad
\tilde Q=\operatorname{scale}\cdot(\gamma\odot Q).
$$

序列按 $C=64$ 切分。每个 chunk 内先计算严格下三角矩阵及其逆：

$$
T=\operatorname{stril}\!\left(\operatorname{diag}(\beta)\tilde K\bar K^{\top}\right),
\qquad A^{-1}=(I+T)^{-1},
$$

$$
U=A^{-1}\operatorname{diag}(\beta)V,
\qquad W=A^{-1}\operatorname{diag}(\beta)\tilde K,
\qquad M_{qk}=\operatorname{tril}\!\left(\tilde Q\bar K^{\top}\right).
$$

以 $S_0=$ `initial_state`，第 $c$ 个 chunk 的状态和输出为：

$$
S_c=\operatorname{diag}(\gamma_C)S_{c-1}+(K_c^r)^{\top}(U-WS_{c-1}),
$$

$$
O_c=\tilde Q_cS_{c-1}+M_{qk,c}(U-WS_{c-1}).
$$

## 函数原型

先调用 `flash_kda_metadata(q, v, initial_state, layout_qkv, cu_seqlens=None)`，再将返回值作为 `metadata` 传入下列接口。metadata 的参数说明见[独立文档](../flash_kda_metadata/README.md)。

```python
flash_kda.flash_kda.flash_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    layout_qkv: str,
    metadata: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]
```

模块路径为 `flash_kda.flash_kda`，以下参数均为必选，没有默认值。

## 参数说明

`B` 为逻辑 batch 数，`Nqk` 为 Query/Key 头数，`Nv` 为 Value 头数，`S` 为每个 batch 的存储序列长度，`T` 为 packed 序列的总 token 数，`Dk = Dv = D = 128`。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `q` | Tensor | 必选 | Query；内核按行归一化。 | bfloat16 | BNSD：`[B, Nqk, S, D]`；其他布局见下表 |
| `k` | Tensor | 必选 | Key；shape 和布局与 `q` 相同，内核按行归一化。 | bfloat16 | 与 `q` 相同 |
| `v` | Tensor | 必选 | Value。 | bfloat16 | BNSD：`[B, Nv, S, D]`；其他布局见下表 |
| `g` | Tensor | 必选 | 原始门控输入，激活公式见功能说明。 | bfloat16 | 与 `v` 相同 |
| `beta` | Tensor | 必选 | 原始写入 logits，内核执行 sigmoid。 | bfloat16 | BNSD：`[B, Nv, S]`；其他布局见下表 |
| `scale` | float | 必选 | Query-key 缩放系数，常用值为 `D ** -0.5`。 | float32 | - |
| `initial_state` | Tensor | 必选 | 初始递归状态，最后两维依次为 `Dv`、`Dk`；无历史状态时传全零张量。 | float32 | `[B, Nv, Dv, Dk]` |
| `A_log` | Tensor | 必选 | 每个 Value 头的 log 时间尺度。 | float32 | `[Nv]` |
| `dt_bias` | Tensor | 必选 | 每个 Value 头、每个 Key 通道的门控偏置。 | float32 | `[Nv, Dk]` |
| `lower_bound` | float | 必选 | 门控下界，取值范围为 `[-5, 0]`。 | float32 | - |
| `layout_qkv` | str | 必选 | 输入布局，取值为 `"BNSD"`、`"BSND"` 或 `"TND"`。 | - | - |
| `metadata` | Tensor | 必选 | 配套 `flash_kda_metadata` 返回的调度张量。 | int32 | `[M]`，容量 `M` 由 metadata 算子确定 |

各布局对应的输入 shape：

| `layout_qkv` | `q`、`k` | `v`、`g` | `beta` |
|:---|:---|:---|:---|
| `BNSD` | `[B, Nqk, S, D]` | `[B, Nv, S, D]` | `[B, Nv, S]` |
| `BSND` | `[B, S, Nqk, D]` | `[B, S, Nv, D]` | `[B, S, Nv]` |
| `TND` | `[T, Nqk, D]` | `[T, Nv, D]` | `[T, Nv]` |

## 返回值说明

返回 `(out, final_state)`，两个张量均位于输入所在设备。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `out` | Tensor | 必选 | 序列输出，布局与 `v` 相同；metadata 标记的有效序列范围之外，输出值未定义。 | bfloat16 | 与 `v` 相同 |
| `final_state` | Tensor | 必选 | 每条序列的最终递归状态；不原地修改 `initial_state`。 | float32 | `[B, Nv, Dv, Dk]` |

内部输出存储按 64 个 token 对齐，返回时裁剪为逻辑 shape；非对齐场景下 `out` 可能是不连续的视图。

## 约束说明

- 当前接口用于 prefill 前向计算，要求 `Dk = Dv = 128`，`B >= 1`，`S > 0` 或 `T > 0`。
- 支持 GQA，要求 `1 <= Nqk <= Nv`、`Nv % Nqk == 0`；metadata 调度要求 `Nv` 不超过 `WORKSPACE_SLOTS`（当前为 861）。
- 所有输入张量（包括 `metadata`）必须连续，位于同一 NPU，且满足参数表中的 dtype。
- BNSD/BSND 的物理 batch 数必须等于 `initial_state` 的 `B`。不传 `cu_seqlens` 时，metadata 按每条序列有效长度均为 `S` 调度。
- BNSD/BSND 可在生成 metadata 时传 `cu_seqlens` 表示各 batch 的有效长度。它是从 0 开始、形状为 `[B + 1]` 的 int32 前缀和，相邻差值必须在 `[1, S]` 内。
- TND 必须在生成 metadata 时传 `cu_seqlens`：形状为 `[B + 1]`、dtype 为 int32、从 0 开始、严格递增，最后一项为 `T`。它必须连续且位于输入所在设备。
- 输入序列长度无需按 64 对齐，尾块由内核处理。只读取和比较 metadata 指定有效区的输出。
- 图捕获兼容性由调用网络验证；当前接口不查询或拒绝 capture 状态。

### metadata 复用

`flash_kda` 不生成 metadata，也不启动 AICPU。整网可在 forward 开始时生成一次 metadata，并在调度配置相同的层间复用。

当 `B`、物理 batch 数、`S/T`、`Nv`、布局、`cu_seqlens` 内容、设备 AIC 核数、workspace slots 或内置调度 cost 公式版本变化时，必须重新生成 metadata。`Nqk` 和 Q/K/V/g/beta/state 的数值不影响 metadata；每次调用仍须满足当前输入的 shape 和 GQA 约束。

## 确定性计算

当前样例未提供逐位确定性保证，现有精度测试按数值容差校验。

## 调用示例

```python
import math
import torch
import torch_npu

from flash_kda.flash_kda import flash_kda
from flash_kda_metadata.flash_kda_metadata import flash_kda_metadata

B, Nk, Nv, S, D = 1, 3, 3, 8192, 128
q = torch.randn(B, Nk, S, D).bfloat16().npu()
k = torch.randn(B, Nk, S, D).bfloat16().npu()
v = torch.randn(B, Nv, S, D).bfloat16().npu()
g = torch.randn(B, Nv, S, D).bfloat16().npu()
beta = torch.randn(B, Nv, S).bfloat16().npu()
initial_state = torch.zeros(B, Nv, D, D, dtype=torch.float32).npu()
A_log = torch.zeros(Nv, dtype=torch.float32).npu()
dt_bias = torch.zeros(Nv, D, dtype=torch.float32).npu()

metadata = flash_kda_metadata(q, v, initial_state, "BNSD", None)
out, final_state = flash_kda(
    q, k, v, g, beta, 1 / math.sqrt(D), initial_state,
    A_log, dt_bias, -5.0, "BNSD", metadata
)
```

## 精度测试

在仓库根目录执行单文件测试。两个 PyTorch golden 内联于
`test/flash_kda/test_flash_kda.py`：`cpu_chunk` 用 FP32 分块递推，`npu_chunk`
在 CPU 上模拟当前算子的 BF16/FP16 交接（state 累加保持 FP32）。
两者与 DSL 做输出及最终 state 的三方对照，不依赖其他 golden 文件：

```bash
python -m pytest -q test/flash_kda/test_flash_kda.py
```

共 12 个 NPU 用例：3 个布局代表精度场景、4 个非对齐尾块、1 个跨轮调度，
以及 4 个归一化 K 快照补零回归。容差为 `atol=rtol=5e-3`。
尾块覆盖 16、32、48、64、128 边界，含 packed 非对齐起点和奇数头数；
跨轮场景有 864 个 chunk/head 任务，超过 861 个 workspace slots。
逐轮检查逻辑 chunk 无遗漏、无重叠，padded 无效输入填 NaN；
快照回归检查有效行和尾部严格补零。全部用例默认启用。

## 性能

以下结果使用 Release 构建和 msprof 采集。
形状为 `B=1, NV=NK=32, S=8192, D=128`，输入为零。
AICPU metadata 在采样循环前生成一次，`Task Duration` 只统计 AICore consumer。

每种布局采样三轮，每轮去掉 2 次 warmup，保留 10 个有效任务。
中位数取三轮中位数的中位数，最小值取该布局的 30 个有效任务；
管线指标同样取三轮中位数的中位数，并非最短任务所在行的指标。

| Layout | Task Duration 中位数 (us) | Task Duration 最小值 (us) | aiv_mte2_ratio | aiv_vec_ratio | aic_mac_ratio | aiv_vec_time (us) | aic_mac_time (us) |
|:---|--:|--:|--:|--:|--:|--:|--:|
| BNSD | 808.552 | 797.867 | 0.3830 | 0.6200 | 0.4010 | 499.468 | 324.350 |
| BSND | 846.577 | 827.835 | 0.4460 | 0.5970 | 0.3830 | 503.887 | 324.383 |
| TND | 841.117 | 828.706 | 0.4420 | 0.6015 | 0.3860 | 503.493 | 324.418 |
