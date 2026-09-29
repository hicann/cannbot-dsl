# Mixed Quant Sparse Flash MLA Metadata

通过 NPU 的 AICPU 执行混合量化稀疏 MLA 分核，生成主算子使用的
`int32[1024]` metadata。配套 AICore 实现位于
[`mixed_quant_sparse_flash_mla`](../mixed_quant_sparse_flash_mla)。

## 算子介绍

根据 Query 前缀和、原始/压缩 TopK 有效长度及当前设备 AIC 核数，生成整行或
FlashDecode 分片计划。metadata 使用 batch 相对坐标，主 Kernel 按槽位处理任务，
有分片时由归约核写入最终输出。

- 默认使用整行分核；满足条件时启用选择性 72 行 FD 计划。
- 通用 cost-based FD 由实现中的 `SUPPORT_FD` 控制，默认关闭。
- 在当前 NPU 流启动，不将输入 Tensor 的值复制到 CPU。
- 首次调用编译 AICPU 产物，后续调用复用进程内缓存。

## 接口

```python
def mixed_quant_sparse_flash_mla_metadata(
    ori_topk_length,
    cmp_topk_length,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_ori_kv=None,
    max_seqlen_cmp_kv=None,
    num_heads_q,
    num_heads_kv,
    head_dim,
    quant_mode,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=True,
):
    ...
```

### 输入要求

| 参数                                             | 说明                                                                 |
| :----------------------------------------------- | :------------------------------------------------------------------- |
| `ori_topk_length/cmp_topk_length`              | 必填连续 INT32`[T1,1]`，同设备、同 shape；无压缩 KV 时压缩长度填 0 |
| `cu_seqlens_q`                                 | 连续 INT32`[B+1]` Query 前缀和；省略时按单 batch `[0,T1]` 处理   |
| `num_heads_q/num_heads_kv/head_dim/quant_mode` | 固定为 64 / 1 / 512 / 1                                              |
| `layout_q/layout_kv`                           | 固定为 TND / PA_BBND                                                 |
| `has_ori_kv`                                   | 必须为 True                                                          |
| `has_cmp_kv`                                   | 与主算子是否提供压缩 KV 保持一致                                     |
| `seqused_*`、`batch_size`、`max_seqlen_*`  | 兼容参数；当前规划不读取这些参数，实际划分由前缀和与 TopK 长度确定   |

输入应位于当前 NPU。实际 AIC 核数从设备属性读取，不能超过 ABI 的 36 核容量。
有效 TopK 长度应与主算子使用的稀疏列数一致，并由调用方保证合法。

### 输出

返回当前 NPU 上的 INT32 `[1024]` metadata，用于负载均衡分核。
主算子根据 metadata 将计算任务分配到各核；调用方将其传入
`mixed_quant_sparse_flash_mla(..., metadata=metadata)` 即可。

## 快速开始

先按照 [主算子快速开始](../mixed_quant_sparse_flash_mla/README.md#快速开始)
加载 CANN 环境，并将两个 sample 目录加入 `PYTHONPATH`。下面演示单 Query、原始 KV
全零的最小调用；KV 使用全零字节编码，输出也应为零：

```python
import torch
import torch_npu

from mixed_quant_sparse_flash_mla import mixed_quant_sparse_flash_mla
from mixed_quant_sparse_flash_mla_metadata import mixed_quant_sparse_flash_mla_metadata

torch.npu.set_device(0)
q = torch.zeros((1, 64, 512), dtype=torch.bfloat16, device="npu")
ori_kv = torch.zeros((1, 128, 1, 544), dtype=torch.uint8, device="npu")
block_table = torch.zeros((1, 1), dtype=torch.int32, device="npu")
indices = torch.arange(128, dtype=torch.int32, device="npu").reshape(1, 1, 128)
cu_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")
ori_length = torch.full((1, 1), 128, dtype=torch.int32, device="npu")
cmp_length = torch.zeros((1, 1), dtype=torch.int32, device="npu")
sinks = torch.zeros(64, dtype=torch.float32, device="npu")

metadata = mixed_quant_sparse_flash_mla_metadata(
    ori_length, cmp_length,
    cu_seqlens_q=cu_q,
    num_heads_q=64, num_heads_kv=1, head_dim=512, quant_mode=1,
    has_cmp_kv=False,
)
out = torch.empty_like(q)
lse = torch.empty(0, dtype=torch.float32, device="npu")
mixed_quant_sparse_flash_mla(
    q, ori_kv=ori_kv, ori_sparse_indices=indices,
    ori_block_table=block_table, cu_seqlens_q=cu_q,
    ori_topk_length=ori_length, sinks=sinks, metadata=metadata,
    quant_mode=1, out=out, lse=lse,
)
torch.npu.synchronize()
assert torch.count_nonzero(out).item() == 0
```

两个算子在同一流上依次提交；跨流由调用方建立依赖。启用 LSE 时设置
`return_softmax_lse=True` 并分配 FP32 `[1,T1,64]` 的 `lse`。

## 测试

测试参照 `sparse_flash_mla_metadata`，直接加载本仓库
metadata 源码，现场构造输入，无需外部测试工程或保存的 `.pt` 文件。

[metadata 测试文件](../../test/mixed_quant_sparse_flash_mla_metadata/test_mixed_quant_sparse_flash_mla_metadata.py)
包含 8 个用例：

- 3 个 NPU 结构用例：覆盖有/无压缩 KV、空行、空 batch 和多核划分，检查输出 shape/dtype/device、
  FA 区间连续且完整覆盖所有 Query 行、未启用 FA 槽清零，以及整行模式下 FD 区和使用计数为零。
  启用核数按运行设备属性检查，不将 ABI 容量视为设备核数，也不要求尾部保留区全零。
- 5 个非法参数用例：检查不支持的 head 配置、量化模式、layout，以及长度 Tensor 的 dtype 和 shape。
  这些检查在设备查询和 AICPU 启动之前执行。

按 [主算子快速开始](../mixed_quant_sparse_flash_mla/README.md#快速开始) 配置自己的 Python、
CANN 环境和设备后，在仓库根目录单独运行 metadata 测试，无需先构建、安装算子 wheel。
测试使用当前可见的逻辑设备 0：

```bash
"$PYTHON" -m pytest test/mixed_quant_sparse_flash_mla_metadata/test_mixed_quant_sparse_flash_mla_metadata.py -v
```

端到端精度由主算子测试负责：本地生成输入和独立 CPU golden，依次调用 metadata 与
AICore，检查 BF16 输出、LSE 和空行结果。72 行选择性 FD 场景由完整 attention 参数集覆盖。
联合运行时，直接将两个规范测试目录交给 pytest：

```bash
export CANNBOTDSL_NATIVE_BINARY_MODE=off
# 默认：4 个 attention 用例 + 全部 8 个 metadata 用例
"$PYTHON" -m pytest test/mixed_quant_sparse_flash_mla test/mixed_quant_sparse_flash_mla_metadata -v
# 完整：13 个 attention 用例 + 全部 8 个 metadata 用例
MQSMLA_CASES=all "$PYTHON" -m pytest test/mixed_quant_sparse_flash_mla test/mixed_quant_sparse_flash_mla_metadata -v
```

`MQSMLA_CASES` 仅选择 attention 用例，不影响 metadata 用例。
测试直接加载当前仓库的 sample 源码；用例选择和 CPU golden
说明见 [主算子测试说明](../mixed_quant_sparse_flash_mla/README.md#精度测试)。
