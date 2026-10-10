# Quant MLA Prolog AW1 BF16 / MXFP8

## 当前框架适配：gcbe9dba

当前适配目标为 CANNBot-DSL + OpKit。
本次核验现有 K3 group 的源码兼容性；MegaMoE 不纳入本轮。
已兼容新包的算子不修改源码，测试结果以完成后的日志为准。
沿用 `simd` 向量 API、单 MixKernel 与现有 Torch 图注册。
2026-09-24 本算子自身测试在新包上完成 `57 passed`，包括 C0/C1 的四条网络 case；
同轮 `commit_recurrent_kda_replayssm` 测试完成 `113 passed`。其余 K3 的整组验证
按要求停止，不能据此声明全组已通过。
本轮不重新编包，下面 Native/入图/性能数据均注明历史框架版本，不能视为 gcbe9dba 的实测结果。

## 历史验证：gff410b6

当前目标为 CANNBot-DSL + OpKit。
新框架的寄存器向量区间使用 `vf(mode="simd")`，不再接受 `raw/raw_reg`。
本次补齐输入补齐路径 `_MlaInputPadUnit.pad` 的残留 `raw`；全部 7 处 VF 区间
统一使用 `simd`。量化公式、单 MixKernel、动态 T/core/cache 策略及 Torch/图注册
保持原有语义。

2026-09-24 在该新包上重新执行：

```bash
python -m pytest -q test/quant_mla_prolog/test_quant_mla_prolog.py -k "not bf16"
```

结果：`49 passed, 8 deselected`。四条 C1 网络 case（T=8/32/64/128）的精度与
固定工作区污染恢复测试通过。重新导出的 Native wheel 在 `require` 模式完成
以下验收，全部命中包内二进制：

- 四条网络 case × 1～32 AIC × eager/ACLGraph/npugraph_ex × 10 次，共 3,840 次检查通过；AIC:AIV=1:2。
- `blocks=261/804/6657` 在 7/32 AIC 的三种模式通过，cache 页数不作为精确 Native 模板键。
- 原始网络脚本仅移除 FA 的 Prolog 片段，eager、ACLGraph、npugraph_ex 均通过；更换输入和 cache slot 后，回放结果与 eager 严格一致。
- 三份故障 PT 各 200 轮固定工作区污染恢复无 NaN/Inf；按既定网络相对 L2/余弦标准通过。rank18 的严格逐元素 Q 诊断仍有 0.110994% 超差，反量化相对 L2 为 0.289306%，与旧基线一致，非 CPU 逐位一致。
- 本轮选取的 15 项 Native 验证全部通过，不代表完整 nightly、C0、大 T 或含 FA 整网的重新验收。

msprof 单 MixKernel 耗时：32 AIC / 64 AIV，C1，npugraph_ex，预热后各采样
20 次。以下是设备 kernel 耗时，不是冷启动或 Python 端到端耗时。

| B，S=8 | T | 平均耗时（μs） | 中位数（μs） |
| --- | --- | --- | --- |
| 1 | 8 | 34.966 | 34.870 |
| 4 | 32 | 41.806 | 41.778 |
| 8 | 64 | 50.687 | 50.642 |
| 16 | 128 | 66.962 | 66.987 |

采样区间无整块 KV cache TensorMove。默认图后端仍有 index ForeachCopy，
T64/128 仍有输入 x 的 TensorMove，不能描述为整张图完全无拷贝。
切换框架后需要重新导出 Native 二进制；新 wheel 内仍为五个 T 区间 × C0/C1
共十个变体，实际 T、core 数和 cache 页数保持运行时传入。

## 历史验证：g1120db9


> **上一轮适配框架**：CANNBot-DSL 与 OpKit。
> 下文标注 `92af3529` 的结果属于旧框架的历史验证，不代表新框架的验收结果。
> Native collector 使用 `cannbotdsl.aot.export`；旧的注册接口不适用于
> 该拆包版本。实现使用 CANNBot-DSL 的 `Channel.produce()/consume()`、公开
> `channel_rewind`、固定 FFTS flag 与 `tile_slice`，不依赖私有内部接口。
> 切换框架后须重新导出 Native 二进制，不能把旧框架 wheel 的命中或精度结果沿用到新包。

本轮提交前执行 `pytest -q test/quant_mla_prolog/test_quant_mla_prolog.py -k "not bf16"`：
`49 passed, 8 deselected`，未重跑 BF16 用例。

本轮新框架的源码 JIT 验证：C1 的 T=8/32/64/128 共四条网络 case 全部通过，
Q 相对 L2 分别为 0.113425% / 0.158928% / 0.137906% / 0.141803%，
KV 相对 L2 均为 0，与旧框架基线的误差指标一致。

新导出 wheel 的 C1 独立验证（2026-09-24）：四条网络 case 分别覆盖 1～32
个 AIC、AIC:AIV=1:2、eager/ACLGraph/npugraph_ex 三种模式各 10 次，
共 3,840 次检查通过。原始网络脚本的 Prolog-only 片段在三种模式下通过；
三份故障 PT 各 200 轮固定工作区污染恢复未出现 NaN/Inf。本轮选取的
12 项 Native require 验证全部通过，不代表完整 nightly、C0 或大 T 的重验。
PT 对照采用既定网络相对 L2/余弦标准；rank18 的 Q 严格逐元素诊断有
0.110994% 超差、反量化相对 L2 为 0.289306%，与旧基线一致，非 CPU 逐位一致。

新框架 msprof 单 MixKernel 耗时（32 AIC / 64 AIV，C1，npugraph_ex，
预热后每条采样 20 次，单位 μs）：

| B，S=8 | T | 平均耗时 | 中位数 |
| --- | --- | --- | --- |
| 1 | 8 | 35.770 | 35.709 |
| 4 | 32 | 42.188 | 42.149 |
| 8 | 64 | 51.415 | 51.507 |
| 16 | 128 | 67.843 | 67.880 |

这些数值是 kernel 耗时，不含 Python 端到端开销。采样区间无整块 KV cache
TensorMove；默认图后端保留 index 的 ForeachCopy，T64/128 还保留输入
x 的 TensorMove，不能将其描述为整张图完全无拷贝。

基于 CANNBot-DSL 实现的 AW1 MLA Prolog，支持 BF16 非量化输出（C0）和
MXFP8 全量化输出（C1），支持由当前设备与流配额选择 1～32 个 AIC。两个模式都在一次
MixKernel launch 中完成 WQA/WKVA 投影、QA/KVA RMSNorm、WQB/WKB 投影、
Q 合并和 PA_NZ KV cache 原位更新；C1 额外完成 Q/KV 输出量化。

C0/C1 接收任意正整数 T，不要求 8 的倍数。Decode 使用 8/32/64/128 行
切分区间；T>128 使用运行时 M128 分块的 prefill 路径。尾块在同一个 MixKernel
内处理，公开输出 shape 始终为实际 T，每次调用只发射一个 MixKernel。

Q 与 KV cache 都使用合并后的 576 维布局：前 512 维为 NoPE，后 64 维为
RoPE 通道。当前算子不接收 sin/cos，不执行旋转位置编码；这里只合并投影得到的
64 维通道，不能把该操作视为已完成 RoPE。

## 跨核依赖与精度看护

QB 按 N128 tile 轮转分配，WKB 按完整 head 分配。一个 head 的 NoPE128
可能来自其他 AIC 的 QB tile，因此本核 `cube_sync_all()` 不能代替跨 AIC
生产完成通知。在 QB 的 FIXPIPE 写回之后，所有实际启动的 AIC 都执行
`cube_sync_block_arrive(PIPE.FIXPIPE, 1, mode=0)` 和相应等待，再进入 WKB。
无 QB 任务的核也参与该屏障；flag 1 的前阶段握手已结束后才复用。

WKB 写完一个 head 后先完成本核 FIXPIPE→MTE2 交接，再通知 AIV 量化；
1～7 核使用完成所有自有 head 后的一次通知，避免耗尽硬件同步 flag。
Q/scale 的 MTE3 写回及最终 AIC/AIV 汇合保持不变。修复不改变量化公式、
cache 地址或图注册，每次调用仍为一个 MixKernel。

四个 C1 网络 case 的单元测试除完整 golden 比较外，增加固定工作区污染恢复：
先用 NaN 输入污染中间结果，紧接着执行有限输入，每个 case 重复三轮。
恢复后 Q、descale_q、完整 cache 必须逐位等于清洁参考，且 Q、scale、
q_head 与 q_latent_h 必须有限。NaN 输入仅用于测试；生产 kernel 不包含
延迟、工作区 hook、NaN 清零或放宽精度阈值的逻辑。

人工延迟 QB 生产核、真实 PT 压力、Native require 及三种入图模式属于独立
验证，不在普通单元测试中枚举 Native 编译配置。自然回放未复现异常不等于
证明不存在竞态，修复还需结合生成代码的生产/消费顺序和受控延迟对照验收。

## API

```python
outputs = quant_mla_prolog(
    x, wqa, wqb, wkva, wkb,
    descale_x, descale_wqa, descale_wqb, descale_wkva,
    norm_weight_qa, norm_weight_kva,
    kv_cache, cache_index, qscale_kv,
    norm_eps=1.0e-6,
    quant_mode_aw=1,
    quant_mode_c=1,
)
```

公开模块只导出 `quant_mla_prolog`。返回值为有序字典：

- `quant_mode_c=0`：`{"q", "kv_cache_out"}`
- `quant_mode_c=1`：`{"q", "kv_cache_out", "descale_q"}`

`kv_cache` 是 in/out（输入输出）；`kv_cache_out is kv_cache`，数据指针和底层 storage 均不变。

### 参数表

| 分组 | 名称 | 类型 | Shape | 默认值 | Format | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| INPUT | `x` | FP8_E4M3 | `(T, dim)` | N/A | ND | 必填 |
| INPUT | `wqa` | FP8_E4M3 | `(q_lora/32, dim, 32)` | N/A | NZ | 必填 |
| INPUT | `wqb` | FP8_E4M3 | `(N*D/32, q_lora, 32)` | N/A | NZ | 必填 |
| INPUT | `wkva` | FP8_E4M3 | `((kv_lora+Dr)/32, dim, 32)` | N/A | NZ | 必填 |
| INPUT | `wkb` | BF16 | `(N, Dn, kv_lora)` | N/A | ND | 必填 |
| INPUT | `descale_x` | FP8_E8M0 | `(T, ceil(dim/64), 2)` | N/A | ND | 必填 |
| INPUT | `descale_wqa` | FP8_E8M0 | `(q_lora, ceil(dim/64), 2)` | N/A | NZ | 必填 |
| INPUT | `descale_wqb` | FP8_E8M0 | `(N*D, ceil(q_lora/64), 2)` | N/A | NZ | 必填 |
| INPUT | `descale_wkva` | FP8_E8M0 | `(kv_lora+Dr, ceil(dim/64), 2)` | N/A | NZ | 必填 |
| INPUT | `norm_weight_qa` | FP32 | `(q_lora,)` | N/A | ND | 必填 |
| INPUT | `norm_weight_kva` | FP32 | `(kv_lora,)` | N/A | ND | 必填 |
| INPUT | `kv_cache` | BF16 / FP8_E4M3 | `(block_number, block_size, 1, kv_lora+Dr)` | N/A | PA_NZ | 原位更新 |
| INPUT | `cache_index` | INT64 | `(T,)` | N/A | ND | cache slot |
| INPUT 可选 | `qscale_kv` | FP32（接口表 DT_BF32） | `(1,)` | `None` | ND | `quant_mode_c=1` 时必填 |
| ATTR | `norm_eps` | FLOAT | 标量 | 无默认值 | N/A | 有限正数 |
| ATTR | `quant_mode_aw` | INT | `1` | `1` | N/A | QA/QB/KVA 全量 MXFP8 |
| ATTR | `quant_mode_c` | INT | `0/1` | `0` | N/A | 0：不量化；1：Q per-token-head FP8，KV per-tensor FP8 |
| OUTPUT | `q` | BF16 / FP8_E4M3 | `(T, N, kv_lora+Dr)` | N/A | ND | NoPE 512 + RoPE 64 |
| OUTPUT | `kv_cache_out` | BF16 / FP8_E4M3 | 同 `kv_cache` | N/A | PA_NZ | 输入 cache 的原地输出 |
| OUTPUT 可选 | `descale_q` | FP32 | `(T, N)` | N/A | ND | 仅 `quant_mode_c=1` |

固定优化规格：

```text
dim=7168, q_lora=1536, kv_lora=512
N=96, D=192, Dn=128, Dr=64
block_size=128, profiled_block_dim=32
supported_aic_core_counts=1..32（运行时有效核数）
supported_t_c0_c1=任意正整数（受设备可用内存限制）
```

`block_number` 与 BF16 接口定义一致：它严格取自 `kv_cache.shape[0]`，表示调用方
已经分配的 PA page 总数，包含历史上下文驻留页，不能从本次调用的 `T` 或
`ceil(T/128)` 推导。`block_size` 固定为 128，并要求
`block_number * 128 >= T`；Native 的 T 与 `block_number` 为动态维，有效 AIC 核数为运行时标量。

公开 E8M0 scale 使用 rank-3 逻辑描述；权重 scale 的底层字节保留 NZ 排列。
`kv_cache` 的公开 shape 是逻辑 shape，连续字节按 PA_NZ 排列，接口通过零拷贝 view
交给 kernel。

## 外部调用示例

下面示例与公开 API 一致，先展示 `quant_mode_c=1` 的完整调用。示例中的单位
E8M0 scale 和随机权重只用于说明接口；整网应直接传入模型量化阶段生成的
E4M3 权重与 E8M0 scale。

```python
import torch
import torch_npu

from quant_mla_prolog import quant_mla_prolog


FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu


def pack_fp8_nz(weight):
    """Pack logical FP8 weight W[O,K] as public NZ [O/32,K,32]."""
    if weight.dtype != FP8 or weight.ndim != 2 or weight.shape[0] % 32:
        raise ValueError("weight must be FP8 [O,K] with O divisible by 32")
    output_size, input_size = weight.shape
    return weight.reshape(output_size // 32, 32, input_size).permute(
        0, 2, 1
    ).contiguous()


def e8m0_ones(shape):
    """Create E8M0 value 1.0; raw exponent byte 127 represents 2**0."""
    return torch.full(shape, 127, dtype=torch.uint8).view(E8M0)


torch.npu.set_device(0)
device = "npu:0"
T = 8
block_number, block_size = 12, 128
generator = torch.Generator(device="cpu").manual_seed(20260916)


def random_fp8(shape):
    return (torch.randn(shape, generator=generator) * 0.25).to(FP8)


x = random_fp8((T, 7168)).to(device)
wqa = pack_fp8_nz(random_fp8((1536, 7168))).to(device)
wqb = pack_fp8_nz(random_fp8((96 * 192, 1536))).to(device)
wkva = pack_fp8_nz(random_fp8((512 + 64, 7168))).to(device)
wkb = torch.randn(
    96, 128, 512, generator=generator, dtype=torch.bfloat16
).to(device)

descale_x = e8m0_ones((T, 112, 2)).to(device)
descale_wqa = e8m0_ones((1536, 112, 2)).to(device)
descale_wqb = e8m0_ones((96 * 192, 24, 2)).to(device)
descale_wkva = e8m0_ones((512 + 64, 112, 2)).to(device)
norm_weight_qa = torch.ones(1536, dtype=torch.float32, device=device)
norm_weight_kva = torch.ones(512, dtype=torch.float32, device=device)

# 全零 cache 的 ND 与 PA_NZ 字节相同，可以直接创建。若 cache 已有有效
# ND 数据，必须在模型/cache 初始化阶段预先转换为 PA_NZ 物理排列。
kv_cache = torch.zeros(
    block_number, block_size, 1, 576, dtype=FP8, device=device
)
cache_index = torch.arange(T, dtype=torch.int64, device=device)
qscale_kv = torch.tensor([0.5], dtype=torch.float32, device=device)

outputs = quant_mla_prolog(
    x, wqa, wqb, wkva, wkb,
    descale_x, descale_wqa, descale_wqb, descale_wkva,
    norm_weight_qa, norm_weight_kva,
    kv_cache, cache_index, qscale_kv,
    norm_eps=1.0e-6,
    quant_mode_aw=1,
    quant_mode_c=1,
)
torch.npu.synchronize()

q = outputs["q"]
descale_q = outputs["descale_q"]
assert q.shape == (T, 96, 576) and q.dtype == FP8
assert descale_q.shape == (T, 96) and descale_q.dtype == torch.float32
assert outputs["kv_cache_out"] is kv_cache
assert outputs["kv_cache_out"].data_ptr() == kv_cache.data_ptr()
```

已经按 `(O/32,K,32)` 保存的权重不能再次调用 `pack_fp8_nz`。静态权重和
weight scale 应在模型加载阶段准备一次并跨调用复用；算子直接消费分别传入的 WQA/WKVA 权重，不需要向公开 API 额外传 prepared handle。

`quant_mode_c=0` 时，`kv_cache` 改用 BF16，`qscale_kv` 传 `None`，返回字典中
不包含 `descale_q`；`q` 和 `kv_cache_out` 均为 BF16。

```python
kv_cache_bf16 = torch.zeros(
    block_number, block_size, 1, 576,
    dtype=torch.bfloat16,
    device=device,
)

outputs_bf16 = quant_mla_prolog(
    x, wqa, wqb, wkva, wkb,
    descale_x, descale_wqa, descale_wqb, descale_wkva,
    norm_weight_qa, norm_weight_kva,
    kv_cache_bf16, cache_index, None,
    norm_eps=1.0e-6,
    quant_mode_aw=1,
    quant_mode_c=0,
)
assert list(outputs_bf16) == ["q", "kv_cache_out"]
assert outputs_bf16["q"].dtype == torch.bfloat16
assert outputs_bf16["kv_cache_out"] is kv_cache_bf16
```

## Torch Native wheel 调用

仓库同时提供与 Native 算子包一致的 Torch 入口。相关文件如下：

| 文件 | 用途 |
| :--- | :--- |
| `net/native_package/run-build.sh` | OpKit Native 编译与 wheel 打包入口 |
| `net/native_package/operator_groups.toml` | `quant_mla_prolog` 源文件映射及 `k3` 分组 |
| `net/verification/aclgraph/aclgraph_quant_mla_prolog.py` | ACLGraph capture/replay 与可选 `npugraph_ex` 探针 |
| `test/quant_mla_prolog/golden.py` | 与算子实现独立的 CPU golden |

Native 构建器直接从 `samples/` 收集源文件（按 `operator_groups.toml` 的文件名
映射），无需复制到 `net/ops`。构建需要 CANNBot-DSL 与 OpKit 两个 wheel：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate cannbot
cd net/native_package

# T 仅按以下切分区间导出，设备/流的有效核数在调用时获取：
# 5 个 T 切分区间 × C0/C1；不枚举具体 T、核数或 cache 页数。
CANN_ENV=<cann-toolkit>/set_env.sh \
  CANNBOTDSL_WHEEL=<path>/cannbotdsl-*.whl \
  OPKIT_WHEEL=<path>/opkit-0.2.0-*.whl \
  ./run-build.sh --operator quant_mla_prolog

python -m pip install --no-deps \
  output/operators/quant_mla_prolog/wheels/cannbot_arena_net-*.whl \
  <path>/opkit-0.2.0-*.whl
export CANNBOTDSL_NATIVE_BINARY_MODE=require
export CANNBOTDSL_NATIVE_BINARY_REPORT=/tmp/quant_mla_native_report.json
```

`require` 模式禁止 Native miss 后回退 source-JIT。运行验证后应检查 report：
`installed_hit > 0` 且 `installed_miss == 0`。只有满足这两个条件，才能证明实际执行的是
安装 wheel 中的 Provider，而不是源码回退路径。

wheel 的普通 Torch 调用方式与上面的公开 API 完全一致，只需把导入改为：

```python
from ops.quant_mla_prolog import quant_mla_prolog

outputs = quant_mla_prolog(
    x, wqa, wqb, wkva, wkb,
    descale_x, descale_wqa, descale_wqb, descale_wkva,
    norm_weight_qa, norm_weight_kva,
    kv_cache, cache_index, qscale_kv,
    norm_eps=1.0e-6,
    quant_mode_aw=1,
    quant_mode_c=1,
)
q = outputs["q"]
descale_q = outputs["descale_q"]
assert outputs["kv_cache_out"] is kv_cache
```

ACLGraph 使用包内的 `torch.library` 适配入口。该入口与公开 API 使用相同参数；
C1 返回 `q, descale_q`，C0 返回 `q, None`。
`kv_cache` 在 schema 中标记为 `Tensor(a!)` 并原位更新，因此不作为别名输出返回：

```python
from ops.quant_mla_prolog import quant_mla_prolog_op

q, descale_q = quant_mla_prolog_op(
    x, wqa, wqb, wkva, wkb,
    descale_x, descale_wqa, descale_wqb, descale_wkva,
    norm_weight_qa, norm_weight_kva,
    kv_cache, cache_index, qscale_kv,
    norm_eps=1.0e-6,
    quant_mode_aw=1,
    quant_mode_c=1,
)
# q: FP8_E4M3 [T,96,576]
# descale_q: FP32 [T,96]
# kv_cache: 已按 PA_NZ 字节布局原位更新
```

不要绕过包装函数直接调用下面的 Dispatcher 名称：

```python
torch.ops.cannbotdsl_quant_mla_prolog.quant_mla_prolog
```

该底层入口为兼容 AOT functionalization，只返回一块内部 packed storage；
`quant_mla_prolog_op` 会对其建立零拷贝的 FP8 `q` 和 FP32 `descale_q` view。
`kv_cache` 是 schema 中唯一的可变输入，仍由 kernel 原位更新。

直接 ACLGraph capture/replay 的调用方式如下；C0 只需换成 BF16 cache、
`qscale_kv=None` 和 `quant_mode_c=0`：

```python
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    captured_q, captured_descale = quant_mla_prolog_op(
        x, wqa, wqb, wkva, wkb,
        descale_x, descale_wqa, descale_wqb, descale_wkva,
        norm_weight_qa, norm_weight_kva,
        kv_cache, cache_index, qscale_kv,
        norm_eps=1.0e-6,
        quant_mode_aw=1,
        quant_mode_c=1,
    )
graph.replay()
torch.npu.synchronize()
```

模型代码可以直接编译调用 `quant_mla_prolog_op`；适配层会在图内保留 packed
storage，并在 Python 边界建立对应的 BF16 或 FP8/FP32 零拷贝 view：

```python
import torch
from ops.quant_mla_prolog import quant_mla_prolog_op


class QuantMlaPrologGraph(torch.nn.Module):
    def forward(self, *args):
        return quant_mla_prolog_op(
            *args,
            norm_eps=1.0e-6,
            quant_mode_aw=1,
            quant_mode_c=1,
        )


compiled = torch.compile(
    QuantMlaPrologGraph(),
    backend="npugraph_ex",
    options={"force_eager": False},
)

q, descale_q = compiled(
    x, wqa, wqb, wkva, wkb,
    descale_x, descale_wqa, descale_wqb, descale_wkva,
    norm_weight_qa, norm_weight_kva,
    kv_cache, cache_index, qscale_kv,
)

# q: FP8_E4M3 [T,96,576]
# descale_q: FP32 [T,96]
# kv_cache: 由编译图按 PA_NZ 字节布局原位更新
```

Native 包导出 10 个变体：`T∈[1,8]/[9,32]/[33,64]/[65,128]/[129,+∞)` × C0/C1。
每个区间内 T 为动态值，核数为运行时标量；`block_number` 从实际 cache shape 取得，
不构造固定页数的内部描述。模型维度与 `block_size=128` 是该优化实现的固定规格，
预编译 `norm_eps=1e-6`；其他 epsilon 需要重新导出匹配变体。验证命令：

```bash
pytest -q test/quant_mla_prolog/test_quant_mla_prolog.py -s
```

`npugraph_ex` 通过配套的 functional 算子和 ACLGraph reinplace 规则处理 cache 原位更新。
图捕获前按上面的调用示例预热；capture 阶段仅记录图，首次 replay 后才检查输出值。
独立入图验证交替更换 x 和 cache_index，并检查 Q/scale/完整 cache 与对应 eager 结果
在 `atol=0, rtol=0` 下逐位一致；NaN/Inf 检查独立于误差门限。

## 实现要点

- WQA/WKVA 分阶段执行，使用各自的 NZ 权重；QA 向量阶段与 KVA 投影重叠。
- MXFP8 数据与 E8M0 scale 通道配套双缓冲。
- 调用方按 T 切分区间与 `quant_mode_c` 选择一个 AOT device 特化，
  每次调用只发射一个 MixKernel，不存在两个 kernel 串行执行。
- T=8/32 的 C1 路径使用 producer-local WKB→Q ready handoff，让下一 head 的
  Cube 工作与上一 head 的 AIV Q 量化重叠。
- T>128 的 C0/C1 prefill 路径按 M128 分块执行 QA/KVA/QB/WKB，保持
  单次 MixKernel launch；大 T 不加入常规网络单元测试参数表。
- 大 M 路径通过 `produce()/consume()` 选定 Tensor：L1 panel 在对应 K0 循环内复用，
  L0C 在整次 K 累加中复用，WKB 的 L0A 在四个 N tile 间复用。整块和尾块共用
  buffer 声明的物理 pitch；动态 prefill 先在 GM 将尾块补齐到 M128，再复用整块计算路径。
- KVA RMSNorm 与 cache scatter 和 WQB/WKB 流水重叠；C1 在同一尾阶段完成 KV 量化。
- C0 将 BF16 NoPE/RoPE 直接在 UB 合并；C1 对合并后的 576 维 Q 做 per-token-head 量化。
- 静态权重与 scale 由调用方准备并复用，不扩展公开 API。

## 精度测试

CPU golden 与 DSL kernel 源码独立：

- [golden.py](../../test/quant_mla_prolog/golden.py)：E4M3/E8M0 解码、FP32 累加、
  BF16 边界、RMSNorm、Q/KV 量化和 PA_NZ scatter。
- [test_quant_mla_prolog.py](../../test/quant_mla_prolog/test_quant_mla_prolog.py)：
  C0/C1 各四条网络 shape 的 pytest，覆盖 T=8/32/64/128。
- [test_large_m.py](../../test/quant_mla_prolog/test_large_m.py)：
  C0/C1 的完整编译覆盖动态 T 与 T=136/256/512，上板精度覆盖 T=136/256，分别检查
  M128 尾块和整块，并沿用网络 case 的精度与未写 cache 检查。

四组网络输入直接定义在 `golden.py` 中，并分别派生 C0/C1，不依赖外部 CSV fixture。

```bash
pytest test/quant_mla_prolog/test_quant_mla_prolog.py -v
pytest test/quant_mla_prolog/test_large_m.py -v
```

八条网络 case 使用 `1e-2` 相对 L2 门限，并额外检查：

- Q 反量化余弦相似度大于 0.999；
- `descale_q` 与 KV 写入精度；
- 未写 cache slot 逐 bit 不变；
- cache 输出保持输入对象和数据指针。

## 本次核数与同步修复（基于 PR #255）

公开接口与图注册保持原样，每次调用仍发射一个 MixKernel。

- Host 在调用时读取当前流配额，取 `min(device_AIC, stream_AIC, stream_AIV // 2, 32)`，
  不缓存流配额；不足一个 AIC/两个 AIV 时直接报错。
- Launcher 和 head ownership 使用同一个有效核数。1～7 核在该核所有 WKB head
  完成后使用一个 ready flag；8～32 核保留逐 head 通知，避免低核数下 flag 数量越界。
- QB 按 N128 tile 分核，WKB 按 192 维完整 head 消费。QB 完成后增加显式
  AIC-wide FIXPIPE arrive/wait，不能把本核 `cube_sync_all()` 当作跨核交接。
- 小 T 在 WKB 写回之后通过本核 FIXPIPE→MTE2 notify/wait 再发布 head-ready；
  AIV 读取前等待通知，末尾增加 MTE3 完成确认，避免 workspace/通知过早复用。
- T8 scale 的生成代码为 4 次 4 字节写回，GM 起点间隔 384 字节、UB 间隔 4 字节；
  两个 AIV 分别写 token 0～3、4～7。此检查针对本次框架生成的代码，不是旧 wheel 的反汇编结论。

### 验证范围

环境：py312、CANN 9.2 beta、CANNBot-DSL、OpKit。
在 Ascend950DT_9582 的 32 AIC 设备上验证不同 launch 核数与实际流配额；
这不等同于逐个硬件 SKU 的验证。

- 源码 pytest：57 passed，含四条网络 case 的 C0/C1、host 核数策略与 NaN/Inf 检查。
- 源码 C1 动态 T：1/7/8/9/17/31/32/33/63/64/65/127/128/129/255/256/257/511/512/513/1024，全通过。
- Native require：T8 在 1～32 每个整数核数通过；32 核四条网络 case C0/C1 通过，共 39 项。
- Native require 泛化：7 核 C1 的 T=1/9/17/33/65/129/257/1024；31 核 C0 的 T=9/33/129/257，全通过。
- Native require Cache：261、6657 页均通过，检查负索引跳过、跨页、末页、未写区域逐位保持、原位指针不变。
- 三份整网故障 PT：新 wheel 各回放 100 次，无新 NaN/Inf、无非确定性输出、只读输入不变、未写 cache 保持。
  Dump 自带的 NaN 输出不是正确性 oracle；这些回放不等同于完整 FA 整网验证。

- Native 连续调用/入图共 540 次：T8 的 1/32 核 eager、双流、ACLGraph、npugraph_ex，
  实际 1/7 核配额双流与双流图，以及 T128/T129 的 ACLGraph/npugraph_ex。
  循环中不做 host 同步，交替输入与 slots，Q/scale/完整 cache 与 eager 逐位一致，全部有限。
- 原始网络片段（已有仅去除 FA 版本、未修改）B4S8/261 页：eager 两步、ACLGraph 两次 replay、
  npugraph_ex 两次 replay 全部通过，`atol=rtol=0`。本环境没有 FA 包，未验证完整 attention。

Native `require` 命中预编译 provider，不允许 JIT fallback。这不表示同一个已捕获图可以任意改变 T：
T/缓冲区 shape 变化需要由图后端重新捕获，Native 二进制仍可复用。
模型 `register_buffer` 静态权重方式的 npugraph_ex 稳态采集中，四条网络 case
均未观察到整块 Cache 或权重拷贝。图外输入仍可能发生 staging：T8/T32 有
`ForeachCopy`，T64/T128 另有输入 x `[T,7168]` 的 `TensorMove`。
如果把静态权重作为每次调用的普通图输入，后端也可能搬运权重；算子不承诺消除图边界上的搬运。
修改流核数配额后需要重新准备并捕获图，不能改变已捕获 kernel 的 launch width。

## 本次性能采集

2026-09-23 重新采集，算子源码基线为 `bf09a4c`。Ascend950DT_9582，
32 AIC/64 AIV，C1；上述框架与本轮重新构建的 Native wheel，`native require`。
`msprof --task-time=l0 --aicore-shape=on`，每条 case 预热 10 次，取后 30 次
MixKernel 的 `Task Duration(us)`。下表不含图输入 staging、输入构造、首次编译、
Python 调用或 FA 开销，不从流水图估算。每次 Prolog 调用仍为一个 MixKernel。
12 组采集前后均通过 CPU golden 与 Q/descale_q 有限值检查，Native 全部命中，
没有 JIT fallback。共享设备上的单次采集不作为前后版本无退化的证明。

| Case | T | eager 均值（μs） | ACLGraph 均值（μs） | npugraph_ex 均值（μs） |
| :--- | ---: | ---: | ---: | ---: |
| B1S8 | 8 | 34.717 | 32.510 | 34.832 |
| B4S8 | 32 | 40.293 | 42.376 | 40.373 |
| B8S8 | 64 | 49.498 | 49.268 | 51.272 |
| B16S8 | 128 | 67.220 | 66.516 | 67.626 |

### 图输入搬运的独立定位

本轮 torch_npu 的 `npugraph_ex` 默认 `clone_input=True`。其 `capture` 为普通
图输入建立固定缓冲区，`process_input` 在 replay 前执行输入复制。测量模型将
静态权重注册为 buffers，将 x、descale_x、cache、index 作为调用参数。

T64 对照仅在独立测量脚本中设 `clone_input=False` 并保持输入地址固定：同一
wheel、同一算子代码下，30 次采样窗口内不再出现 TensorMove 或 ForeachCopy，
精度和有限值检查通过。这确认了该次输入 x 的 TensorMove 来自图边界 staging。
T128 观察到相同形状的输入搬运，尚未执行关闭该开关的对照。
此实验没有修改算子 Torch 接口、schema、Functionalize 或 reinplace 注册，
也没有把该选项设为算子的默认值；不能据此要求整网直接关闭输入复制。
