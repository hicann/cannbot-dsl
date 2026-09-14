# Host API

Host API 在普通 Python 代码或 Host 编排函数中使用，负责定义 kernel、准备编译参数、编译和加载产物、启动设备任务以及查询平台信息。

## 接口分类

| 分类 | 主要内容 | 接口 |
| --- | --- | --- |
| 装饰器与程序定义 | 定义 JIT 函数、设备 Kernel 和调用入口 | `jit`、`kernel`、`JitFunction`、`KernelLauncher` |
| 编译与运行 | 编译、加载、调用和复用编译产物 | `JitFunction.compile`、`load`、`ProviderCallable`、`clear_compile_cache` |
| 参数与数据描述 | 描述编译期常量、动态维度和 Kernel 参数结构 | `Constexpr`、`Dim`、`TensorSpec`、`TensorListSpec`、`StructSpec` |
| 平台与性能分析 | 查询设备及存储能力并配置 profiling | `PlatformInfo`、`get_platform_info`、`get_mem_size`、`ProfileSpec`、`profiler.report_tensor_info` |
| 异常与诊断 | 表达 Host 构图、编译和运行阶段错误 | `CANNBotError`、`DiagnosticCode` 等 |
