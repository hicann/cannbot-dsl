# Kernel API

Kernel API 面向 AI Core，在 `@kernel` 函数体或其展开的设备侧辅助函数中使用，用于描述设备上的数据访问、计算、同步和控制逻辑。

## 接口分类

| 分类 | 主要内容 | 接口 |
| --- | --- | --- |
| 基本数据类型与操作 | dtype、Tensor、shape、layout、坐标、构造和视图 | `DType`、`Tensor`、`Shape`、`Layout`、`make_tensor`、`tile_view` |
| 数据搬运 | 存储层级间搬运和搬运引擎配置 | [`mem_copy`](/api/kernel/data-movement/mem-copy)、`make_copy_engine` |
| 向量/寄存器（Vector/Reg）运算 | Mask、访存、算术、比较、归约、转换和重排 | `reg.vload`、`reg.vadd`、`reg.vreduce_sum`、`reg.vcast` 等 |
| 标量（Scalar）运算 | 原子、位运算、标量访存和类型转换 | `scalar.atomic_add`、`scalar.popc`、`scalar.cast` 等 |
| [同步与缓存控制](/api/kernel/synchronization-cache/) | Cube、Vector、全局同步和缓存维护 | [`dcci_single`](/api/kernel/synchronization-cache/dcci-single)、[`dcci_entire_out`](/api/kernel/synchronization-cache/dcci-entire-out)、[`dcci_entire_atomic`](/api/kernel/synchronization-cache/dcci-entire-atomic)、[`dci`](/api/kernel/synchronization-cache/dci) |
| [系统变量访问](/api/kernel/system/) | block、物理核、子核、周期和状态查询 | [`get_block_idx`](/api/kernel/system/get-block-idx)、[`get_block_num`](/api/kernel/system/get-block-num)、[`get_core_id`](/api/kernel/system/get-core-id) 等 |
| 资源管理 | 统一缓冲区（UB）、Buffer、Channel 和延迟线 | `UB`、`make_buffer`、`make_channel`、`DelayLine` |
| 分布式通信 | 通信上下文、通信引擎和设备地址 | `distributed.get_comm_context`、`distributed.CommEngine` 等 |
| 调试接口 | 设备侧标量、Tensor 和寄存器调试 | `print_scalar`、`print_tensor`、`dump_reg`、`dump_tensor` |
| 作用域与控制流 | 设备控制流、编译期表达式和执行作用域 | `range`、`select`、`const_expr`、`range_constexpr`、`cube`、`vf` |
