# 矩阵计算

矩阵计算接口完成 Mmad 矩阵乘加及其计算模式配置，仅在 AIC 上生效。

- **[概述](/api/kernel/cube_compute/overview)**
- **[矩阵计算单元](/api/kernel/cube_compute/unit)**
- **[计算流程](/api/kernel/cube_compute/flow)**
- **[背景与核心概念](/api/kernel/cube_compute/fractal-intro)**
- **[关键分形格式](/api/kernel/cube_compute/fractal-formats)**
- **[矩阵计算关键特性说明](/api/kernel/cube_compute/key-features)**
  - **[GEMV](/api/kernel/cube_compute/key-features-gemv)**
  - **[HF32](/api/kernel/cube_compute/key-features-hf32)**
  - **[UnitFlag](/api/kernel/cube_compute/key-features-unit-flag)**

## 接口

- **[`enable_fp8`](/api/kernel/cube_compute/enable-fp8)**
- **[`enable_hf32`](/api/kernel/cube_compute/enable-hf32)**
- **[`enable_hif8`](/api/kernel/cube_compute/enable-hif8)**
- **[`matmul`](/api/kernel/cube_compute/matmul)**
- **[`set_fp32_mode`](/api/kernel/cube_compute/set-fp32-mode)**
- **[`set_hf32_round_mode`](/api/kernel/cube_compute/set-hf32-round-mode)**
- **[`set_mmad_direction`](/api/kernel/cube_compute/set-mmad-direction)**
