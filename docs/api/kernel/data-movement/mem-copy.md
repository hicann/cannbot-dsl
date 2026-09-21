---
title: mem_copy
api_name: mem_copy
category: data-movement
api_group: kernel
layer: tensor
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# `mem_copy`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持


## 功能说明

`mem_copy` 是 Kernel 内统一的数据搬运接口。它根据 `src`、`dst` 的存储位置，实现 MTE2、MTE3、MTE1、FIXPIPE 或 UB 内部搬运；除普通复制外，
还可以表达 ND/NZ 格式转换、L1 到 L0 的转置加载、FIXPIPE 量化/反量化、MX scale
绑定、L2 Cache 策略、原子累加以及 Cube/Vector 分核搬运。

搬运范围由传入 Tensor/view 的逻辑 shape、stride 和运行时 actual shape 决定；物理
layout、容量和对齐必须能覆盖该范围。`dst` 和 `src` 也可以是兼容的 `Buffer` 或
`Channel`。直接把 `Channel` 传给 `mem_copy` 时，把生产者/消费者访问
解析为对应的 `acquire`/`commit` 或 `wait`/`release` 事务。

未指定 `engine` 时，普通 Identity 搬运直接按内存方向选择实现；GM→L1 且源为
Identity layout 时，还可以根据目标 layout 自动推导 `nd2nz` 或 `dn2nz`。NDDMA
不会被自动选择；其他显式格式转换、padding 和 FIXPIPE 配置通过
`make_copy_engine(...)` 传入。

### 支持的搬运方向

| `src` → `dst` | 支持的格式变换 | 主要执行单元 |
| --- | --- | --- |
| GM → UB | `identity`；可显式选择 `kind="nddma"` | MTE2 |
| UB → GM | `identity` | MTE3 |
| GM → L1 | `identity`、`nd2nz`、`dn2nz`、MX scale A/B 系列变换 | MTE2 |
| UB → L1 | `identity`、`nd2nz` | MTE3 |
| UB → UB | `identity`、`nd2nz`、`nz2nd` | Vector/UB 搬运 |
| L1 → L0A/L0B | `identity`、`transpose` | MTE1 |
| L1 → BIAS | `identity` | MTE1 |
| L0C → GM | `identity`、`nz2nd` | FIXPIPE |
| L0C → UB | `identity`、`nz2nd` | FIXPIPE |
| L0C → L1 | `identity` | FIXPIPE |

## 函数原型

```python
def mem_copy(
    dst: Tensor,
    src: Tensor,
    engine: CopyEngine | None = None,
    transpose: bool = False,
    deq_scale_buf: Tensor | None = None,
    deq_scale_val: int | float | Float32 | None = None,
    mx_scale: Tensor | None = None,
    dual_param: DualParam | None = None,
    l2_cache_ctl: int = 0,
    atomic_add: bool | Boolean = False,
) -> None: ...
```

## 参数说明

| 参数 | 输入/输出 | 类型 | 必选 | 默认值 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `dst` | 输出；启用原子累加时为读写 | `Tensor`；兼容的 `Buffer`/`Channel` | 是 | 无 | 目标逻辑区域。必须具有受支持的目标 `MemLoc`、layout 和足够的物理容量。 |
| `src` | 输入 | `Tensor`；兼容的 `Buffer`/`Channel` | 是 | 无 | 源逻辑区域。其 shape/stride/view 决定待搬运区域。 |
| `engine` | 输入 | `CopyEngine` 或 `None` | 否 | `None` | 由 `make_copy_engine` 创建，携带格式、padding、步长、FIXPIPE 等静态配置。为 `None` 时仅执行默认搬运或可唯一推导的 GM→L1 变换。 |
| `transpose` | 输入 | `bool` | 否 | `False` | 对 L1→L0A/L0B 加载启用转置；其他方向不支持。该参数优先于 `engine.format_transform`。 |
| `deq_scale_buf` | 输入 | `Tensor` 或 `None` | 否 | `None` | FIXPIPE 逐 N 通道反量化表，支持 L0C→GM/UB。 |
| `deq_scale_val` | 输入 | `int`、`float`、`Float32` 或 `None` | 否 | `None` | FIXPIPE 标量反量化系数；Python 数值会提升为 `float32` 常量，`Float32` 可以是运行时 kernel 参数。 |
| `mx_scale` | 输入 | `Tensor`/兼容的 `Channel` 或 `None` | 否 | `None` | L1→L0A/L0B MX 数据加载所绑定的 E8M0 scale。它不是 GM→L1 的 MX scale 格式变换参数。 |
| `dual_param` | 输入 | `DualParam` 或 `None` | 否 | `None` | 两个 Vector 子核的静态分区方案；支持 GM→UB、UB→GM、UB→L1 和 L0C→UB。 |
| `l2_cache_ctl` | 输入 | `int` | 否 | `0` | 编译期 L2 Cache 策略。不同搬运 C API 对同一数值的枚举含义不同，见[约束说明](#约束说明)。 |
| `atomic_add` | 输入 | `bool` 或运行时 `Boolean`/i1 | 否 | `False` | 将 `src` 原子累加到 GM `dst`；运行时布尔值会生成条件分支。 |

## 返回值说明

无返回值。搬运结果写入 `dst`；启用原子累加时，结果累加到 `dst` 的原有数据中。

## 约束说明

### 通用约束

- `dst`、`src` 必须能解析为带类型的 Tensor 或 Channel；搬运方向必须在上表中。
- 格式转换必须同时满足方向和物理格式要求：`nd2nz` 的目标为 NZ；`dn2nz`
  和 ScaleA 系列变换的目标为 ZN；ScaleB 系列变换的目标为 NZ；`nz2nd`
  要求源为 NZ/ZN/ZZ 等 fractal 格式、目标为 ND；`transpose` 要求源和目标均为
  fractal 格式。
- `make_copy_engine(copy_type=...)` 可用 `"identity"`、`"format"`、`"fixpipe"`
  限定一组合法配置。指定 `copy_type` 后，不能再传入其他组的非默认参数；省略它
  仅用于兼容已有的参数组合，并不会放宽底层方向和类型校验。
- 普通非 FIXPIPE 搬运的 `engine.dtype` 应与源 dtype 一致；FP8 以整数 Tensor
  作为等宽字节载体时允许相同存储位宽。FIXPIPE 转换中，`engine.dtype` 必须匹配
  源或目标 dtype。
- 使用带 `alignment` 的 `tile_view` 执行 `nd2nz`/`dn2nz` 时，末两维对齐必须
  分别满足硬件 M0/C0 粒度；元素位宽必须整除 32 字节。

### NDDMA、尾块与 padding

- `kind="nddma"` 只能显式用于 Identity GM→UB；`mem_copy` 不会自动选择它。
  源和目标 dtype 必须一致，支持 8/16/32 位整数、`float16`、`bfloat16`、
  `float32`，逻辑 rank 为 1～5 且源/目标 rank 相同。
- NDDMA 的 `left_padding`/`right_padding` 按 DSL 轴顺序给出，长度必须等于搬运
  rank，每个值为 0～255；静态目标维度必须等于“源维度 + 左 padding + 右
  padding”。`padding_mode` 只能为 `"constant"` 或 `"nearest"`。每个静态源
  维度小于 `2^20`，源 stride 在 `[0, 2^40)`，目标 stride 在 `[0, 2^20)`。
- NDDMA 不能与格式/FIXPIPE/strided engine 选项、`mx_scale`、反量化、
  `dual_param`、原子累加、转置或非零 `l2_cache_ctl` 组合。
- `tail_alignment="ub"` 仅适用于默认 Identity GM→UB。它把运行时尾行按 32
  字节对齐紧凑写入 UB，并把该 pitch 传播给后续读取；默认 `"tiler"` 保留目标
  tiler 声明的行 stride。该选项不会初始化 32 字节对齐尾端与更大 tiler pitch
  之间的残余区域。
- 普通 Identity GM→UB 的最后一个不满 DataBlock 固定补零；`pad_value` 用于
  NDDMA 显式 padding 和 raw GM→L1 的部分 DataBlock padding。
- GM→L1 Identity 对齐搬运可使用 `left_padding_count`/
  `right_padding_count`，单位是元素，仅支持 8/16/32 位类型；每侧上限依次为
  32/16/8 个元素。`smallc0_en=True` 只适用于 `dn2nz`，调用者还必须保证 D
  维不大于 4。
- `nd_num`、`src_nd_stride`、`dst_nd_stride` 只适用于 `nd2nz` 或 `dn2nz`；
  `nd_num >= 1`，所有 ND/C0 stride 和 `unit_flag_mode` 都必须为非负整数。

### 转置与 MX scale

- `transpose=True` 仅支持 L1→L0A/L0B；其他方向会报错。它可以与合法的
  `mx_scale` 绑定组合使用。
- `mx_scale` 仅用于 L1→L0A/L0B 的 MX 数据加载。scale 必须位于 L1、dtype
  为 `fp8_e8m0`，数据和 scale 都是逻辑 rank 2；L0 目标数据类型必须为
  `fp8_e4m3fn`、`fp8_e5m2` 或 `fp4x2_e2m1`。ScaleA 的物理格式必须为 ZN，
  ScaleB 必须为 NZ；scale 外轴与数据外轴相等，scale 的 K 轴长度为
  `2 * ceil(K / 64)`。
- GM→L1 搬运 MX scale 数据时，应在 `engine` 中选择
  `mx_scale_and`/`mx_scale_adn`（ScaleA）或
  `mx_scale_bnd`/`mx_scale_bdn`（ScaleB）。其中成对 E8M0 数据的最后一维为
  2，`G = ceil(K / 64)`；`mx_scale_a`/`mx_scale_b` 仅保留给旧的二维平铺布局。

### FIXPIPE 量化/反量化

- `deq_scale_buf` 与 `deq_scale_val` 不能同时传入。
- `deq_scale_buf` 支持 int32 L0C→GM/UB，输出仅支持 `float16`/`bfloat16`；
  必须提供携带输出 dtype 的 `engine`。scale 表位于 L1 或 FBUF，使用
  `int64[N]` 或字节等价的 `int32[2N]`，每个 64 位条目的低 32 位保存 fp32
  scale 位模式。
- 非单位标量 `deq_scale_val`/`engine.deq_scale` 仅用于 L0C→GM/UB/L1。
  支持的转换为：int32→float16/bfloat16/int8/uint8，以及
  float32→float32/float16/bfloat16/int8/uint8/fp8_e4m3fn/hifloat8。
- L0C→GM/UB/L1 还允许同 dtype 直通。除此之外的 FIXPIPE dtype 组合会被拒绝，
  不会静默忽略 scale。

### L1→BIAS 特殊约束

- 浮点路径支持 `float16`/`bfloat16`/`float32` L1 源到 `float32` BIAS；整数
  路径仅支持 `int32`→`int32`。源和目标都必须是相同正长度的逻辑 rank-1。
- 两侧物理布局必须是连续 rank-1。BIAS 容量按 16 个元素向上对齐，物理存储按
  64 字节对齐且总量不超过 4096 字节；L1 源还必须覆盖搬运 C API 的 DataBlock
  向上对齐读取范围。

### 双子核分区

- `DualParam(axis, alignment, part_id, actual)` 的 `axis` 只能是 0 或 1，
  `alignment` 必须为正整数；`axis=1` 时 `alignment` 必须为 1。分区搬运要求
  rank-2 操作数，`actual` 若提供也必须是 rank-2，且每一维不能超过静态
  `full_shape`/存储容量。
- GM→UB、UB→GM、UB→L1 必须提供 `part_id`，通常使用
  `get_subblock_id()`；每个子核只搬运自己分区的运行时有效范围。
- L0C→UB 是硬件 dual-destination FIXPIPE 模式，必须省略 `part_id`，同时提供
  `dual_dst_ctl=1`（按 M/axis 0 切分）或 `dual_dst_ctl=2`（按 N/axis 1
  切分）的 FIXPIPE `engine`。该模式仅允许源、目标和 engine 同 dtype 的
  NoQuant 直通，不能与反量化或窄化转换组合。
- L0C→UB 按 N 切分时，还要求源为 NZ、目标为 ND，只允许 `identity` 或
  `nz2nd`，目标 UB 行 stride 必须覆盖最大分区跨度并按 32 字节对齐；硬件编程
  N 上限为 4064，M 上限为 65535。
- `layout_transform="nz_fold_n_to_m"` 只支持 UB→L1，且必须与
  `DualParam(axis=1, alignment=1, part_id=...)` 配合，源/目标/engine dtype
  一致；不能再组合 MX scale、反量化、转置、padding、FIXPIPE 或 strided-copy
  配置。

### L2 Cache 与原子累加

- `l2_cache_ctl` 只能取 0、1、2、4，且非零值只支持 GM→UB、UB→GM、
  GM→L1、L0C→GM。显式 NDDMA 只允许 0。
- 对 GM↔UB 的 load/store L2 枚举，0/1/2 分别表示
  `NORMAL_FIRST_VICTIM`/`NORMAL_LAST_VICTIM`/`NORMAL_PERSISTENT`；4 在
  load 侧表示 `NOTALLOC_KEEP`，在 store 侧表示 `NOTALLOC_CLEAN`。
- 对 GM→L1 和 L0C→GM 的 `uint8_t l2_cache_ctl`，0/1/2/4 分别表示
  `DISABLE`/`NORMAL`/`LAST`/`PERSISTENT`。因此不能把 0～4 在所有搬运方向
  上解释成同一套策略。
- `atomic_add` 仅支持 UB→GM 和 L0C→GM。GM 目标 dtype 必须为 signless
  `int8`/`int16`/`int32`、`bfloat16`、`float16` 或 `float32`；参数只能是
  Python `bool` 或运行时 i1/`Boolean`，普通整数 0/1 不会被当作 bool 接受。

### 流水同步

- 不同硬件流水访问同一或重叠存储区域，且至少一个操作写入时，必须建立可证明的
  先后关系。优先使用 `Channel` 表达生产/消费事务；使用裸 `Buffer`/Tensor 时，
  需要根据流水显式调用同步接口。`mem_copy` 本身不等价于全流水 barrier。

## 调用示例

以下示例将 32 个 `float32` 元素从 GM 搬运到 UB，再从 UB 写回 GM，并在 Host 侧校验输出数据与输入数据一致。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import cannbotdsl as cb
import torch
import torch_npu  # noqa: F401  # Register the Ascend NPU backend with PyTorch.


@cb.kernel
def copy_kernel(src, dst):
    """Copy 32 float32 elements from GM to UB and then back to GM."""
    tmp = cb.Channel(
        cb.MemLoc.UB,
        shape=(32,),
        dtype=src.dtype,
        depth=1,
    )

    cb.mem_copy(tmp, src)
    cb.mem_copy(dst, tmp)


class CopyOperator:
    @cb.jit
    def run(self, src, dst):
        copy_kernel[1](src, dst)


def main():
    src = torch.arange(32, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src)

    CopyOperator().run(src, dst)
    torch.npu.synchronize()

    actual = dst.cpu()
    expected = torch.arange(32, dtype=torch.float32)
    torch.testing.assert_close(actual, expected)

    print("mem_copy example passed")
    print(
        f"output verified: shape={tuple(actual.shape)}, "
        f"first={actual[0].item()}, last={actual[-1].item()}"
    )


if __name__ == "__main__":
    main()
```

### 预期结果

```text
mem_copy example passed
output verified: shape=(32,), first=0.0, last=31.0
```
