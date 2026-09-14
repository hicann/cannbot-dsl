---
pageClass: api-index-page
---

# API 文档

CANNBot-DSL 提供 Host、Kernel 和 AI CPU 三类公共 Python 接口。文档按照接口的调用位置与执行模型分类，并在各分类下按照功能模块组织。

## API 分类

- [Host API](/api/host/)：用于在 Host 侧声明、编译、加载和启动 Kernel，并提供平台信息查询与性能分析能力。
- [Kernel API](/api/kernel/)：用于在 AI Core Kernel 中描述数据组织、数据搬运、计算、同步、控制流、系统信息查询和调试操作。
- [AI CPU API](/api/aicpu/)：用于声明 AI CPU Kernel 入口及参数类型，并提供 Host 侧参数准备、任务启动和运行诊断能力。
