# fused_recurrent_kda_snapshot

## 产品支持情况

- Ascend 950：支持。

## 功能说明

`fused_recurrent_kda_snapshot` 计算 1～8 token 的 KDA decode，并把每个 token 对应的递归状态写入指定 state-pool 槽位。它适用于为候选分支保存完整状态的 Snapshot 协议。实现位于 [`fused_recurrent_kda_snapshot.py`](fused_recurrent_kda_snapshot.py)。

Q/K 归一化、门控激活和逐 token 递推公式与 ReplaySSM 相同。对第 `t` 个候选，算子把 `S_t` 写到 `state[ssm_state_indices[token_offset+t]]`。每个 batch 的输入初态由第一个 token 对应槽位读取；`num_accepted_tokens` 用于选择前一轮已经接受的状态。

## 函数原型

```python
fused_recurrent_kda(
    mixedqkv: torch.Tensor,
    state: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float | None,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = -5.0,
    layout_qkv: str = "BSND",
    *,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

`mixedqkv`、`g`、`beta` 为 BF16；`A_log`、`dt_bias` 为 FP32；`state` 支持 BF16 或 FP32。支持 BSND、BNSD 和 packed TND，`D=128`，`1<=N<=96`，每条序列长度不超过 8。`state` 的 shape 为 `[pool_slots,N,128,128]`，pool 至少包含 `B*S` 个槽；`ssm_state_indices` 为 `[B*S]` int32。返回 `(state,out)`，state 原地更新，out 为 BF16 且布局与 `g` 相同。

## 测试

```bash
python -m pytest -q test/fused_recurrent_kda_snapshot/test_fused_recurrent_kda_snapshot.py
python -m pytest -q test/fused_recurrent_kda_snapshot/test_fused_recurrent_kda_snapshot.py --whitebox
```

默认回归保留 BSND、BNSD 和 TND 三个代表场景。`--whitebox` 追加 BF16/FP32 state、非连续 state 索引、不同接受数和多轮状态更新，并与 FP32/BF16 存储路径的 PyTorch golden 对照。

## 性能

以下结果于 2026-09-14 使用 CANN 9.2、Release 构建和 msprof `Task Duration` 采集。固定形状为 `B=16,N=6,S=8,D=128`，采用 BSND、FP32 state 和 `accepted=4`；先运行 10 次 warmup，再保留 100 个 Snapshot 任务。

| Task Duration 中位数 (us) | 最小值 (us) | 平均值 (us) | P90 (us) | 最大值 (us) |
|--:|--:|--:|--:|--:|
| 18.0585 | 16.758 | 17.9963 | 18.702 | 19.258 |
