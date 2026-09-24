# Reg矢量计算

Reg矢量计算接口在 `simd` 模式下操作矢量数据寄存器与掩码寄存器，均须在 `with cb.vf(mode="simd"):` 作用域内调用，仅在 AIV 上生效。

- **[概述](/api/kernel/reg_compute/overview)**
- **[关键特性说明](/api/kernel/reg_compute/key-features)**

## 子分类

- **[Reg数据搬入](/api/kernel/reg_compute/load/)**
- **[基础算术](/api/kernel/reg_compute/reg_arith/)**
- **[广播操作](/api/kernel/reg_compute/reg_broadcast/)**
- **[比较计算](/api/kernel/reg_compute/reg_compare/)**
- **[类型转换](/api/kernel/reg_compute/reg_convert/)**
- **[复合计算](/api/kernel/reg_compute/reg_fused/)**
- **[聚合操作](/api/kernel/reg_compute/reg_gather/)**
- **[直方图](/api/kernel/reg_compute/reg_histogram/)**
- **[索引操作](/api/kernel/reg_compute/reg_index/)**
- **[逻辑计算](/api/kernel/reg_compute/reg_logic/)**
- **[掩码寄存器操作](/api/kernel/reg_compute/reg_mask/)**
- **[排布变换](/api/kernel/reg_compute/reg_permute_sel/)**
- **[归约计算](/api/kernel/reg_compute/reg_reduce/)**
- **[同步控制](/api/kernel/reg_compute/reg_sync/)**
- **[Reg离散搬出](/api/kernel/reg_compute/scatter/)**
- **[Reg数据搬出](/api/kernel/reg_compute/store/)**
- **[Reg聚合搬入](/api/kernel/reg_compute/ub_gather/)**

