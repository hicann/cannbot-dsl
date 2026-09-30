# EngramGate

`engram_gate` 使用 CANNBot-DSL 融合 Engram residual gate 的双路 RMS、加权点积、
signed-sqrt sigmoid 和 residual update。

## 接口

```python
y = engram_gate(
    x,
    key,
    value,
    weight,
    image_mask=None,
    eps=1e-6,
    clamp_value=1e-6,
)
```

| 参数 | dtype | shape |
| :--- | :--- | :--- |
| `x`, `key` | BF16 | `[..., hc_mult, dim]`，ndim ≥ 2；前导维折叠为 token，2-D 表示 `hc_mult=1` |
| `value` | BF16 | `[..., dim]`（与 `x` 前导维一致） |
| `weight` | FP32 | `[hc_mult, dim]` |
| `image_mask` | BOOL，可选 | `[...]`（与 `x` 前导维一致），`True` 的 token 原样返回 `x` |
| `y` | BF16 | 与 `x` 同 shape |

`dim` 支持任意正整数：不足 64 的尾部按掩码处理；超过单行驻留 UB 预算
（`ceil(dim/64)*64 > 8448`）时自动切换为列分块流式路径（两遍扫描、串行行
处理），无 dim 上限。模型默认配置为 `dim=5120, hc_mult=4`。

计算公式：

```text
rstd = rsqrt(mean(x²) + eps) * rsqrt(mean(key²) + eps)
dot  = sum(x * weight * key) * rstd / sqrt(dim)
gate = sigmoid(copysign(sqrt(max(abs(dot), clamp_value)), dot))
y    = bf16(x + gate * value)
```

运行测试：

```bash
pytest -q test/engram_gate/test_engram_gate.py
```
