# CANNBot-DSL 装饰器

CANNBot-DSL 与 OpKit 提供八个 Python 装饰器，用于声明 Host 入口、DSL 辅助函数、设备 Kernel、结构化数据、AI CPU Kernel，以及配置算子的编译、发布和 ACLNN 接口。

本文介绍各装饰器的用途和基本用法。

## 装饰器概览

| 装饰器 | 主要用途 | 典型使用位置 |
| --- | --- | --- |
| [`@host`](#host) | 定义可从 Python 调用的 Host 编排入口 | 算子入口、Kernel 启动 |
| [`@jit`](#jit) | 定义 DSL 内部的内联辅助函数 | 可复用的 Host 或设备逻辑 |
| [`@kernel`](#kernel) | 定义在 NPU AI Core 上执行的 Kernel | 算子的设备计算代码 |
| [`@datastruct`](#datastruct) | 定义可传入 DSL 程序的结构化数据 | Kernel 配置、复合参数 |
| [`@aicpu_kernel`](#aicpu-kernel) | 定义在 AI CPU 上执行的 Kernel | 元数据、调度或轻量控制计算 |
| [`@compile_cache`](#compile-cache) | 按配置编译并复用算子程序 | dtype、shape 或算法配置对应的编译入口 |
| [`@export`](#export) | 声明需要发布的算子配置 | Native 算子包的构建入口 |
| [`@aclnn`](#aclnn) | 声明 ACLNN 算子接口 | 算子的 Host 入口与输出定义 |

## `@host` {#host}

`@host` 用于定义可从普通 Python 调用的 Host 编排函数。函数体负责组织 Kernel 调用，调用时根据参数完成编译、缓存复用和执行，也可以装饰类的实例方法。

```python
import cannbotdsl as cb


@cb.kernel
def add_kernel(x, y, out):
    # 使用 CANNBot-DSL 编写设备计算
    ...


class Add:
    @cb.host
    def run(self, x, y, out):
        add_kernel[8](x, y, out)
```

作为算子入口时，可以直接传入运行时参数调用；输出 Tensor 由调用方准备，通过参数传入：

```python
Add().run(x, y, out)
```

也可以根据 `TensorSpec` 等参数规格，使用 `cannbotdsl.compile(...)` 提前编译：

```python
spec = cb.TensorSpec((1024,), cb.dtypes.float16)
program = cb.compile(Add().run, spec, spec, spec)
program(x, y, out)
```

`@host` 函数应返回 `None`，只能从 DSL 外部的普通 Python 调用。提前编译使用 `cb.compile(host_fn, ...)`。

## `@jit` {#jit}

`@jit` 用于定义 DSL 内部的内联辅助函数，可由 `@host`、`@kernel` 或其他 `@jit` 函数调用，用于拆分和复用 DSL 逻辑。

```python
import cannbotdsl as cb


@cb.jit
def tile_offset(block_idx, tile_size):
    return block_idx * tile_size


@cb.kernel
def add_kernel(x, y, out):
    offset = tile_offset(cb.get_block_idx(), 128)
    # 从 offset 对应的位置搬运数据并完成计算
    ...
```

辅助函数沿用调用方的 Host 或设备执行环境，可以返回计算结果。`@jit` 不能直接从普通 Python 调用，也不能作为 `cb.compile(...)` 的编译入口；Kernel 启动应直接写在 `@host` 函数体中。

## `@kernel` {#kernel}

`@kernel` 用于定义在 NPU AI Core 上执行的设备函数。Kernel 负责张量数据搬运和计算，由 `@host` 函数直接发起，返回值应为 `None`。

```python
import cannbotdsl as cb


@cb.kernel
def add_kernel(x, y, out):
    # 使用 CANNBot-DSL 编写设备计算
    ...


@cb.host
def add(x, y, out):
    add_kernel[8](x, y, out)
```

方括号中的值表示启动的 block 数量，省略时默认为 1。`@kernel` 函数不能直接从普通 Python 代码调用，也不能由 `@jit` 辅助函数间接启动；启动语句应直接位于 `@host` 函数体中。

`@kernel` 也可以装饰类，用于组织多个相关的 Kernel 方法和共享配置：

```python
import cannbotdsl as cb


@cb.kernel
class AddKernel:
    def __init__(self, tile_size):
        self.tile_size = tile_size

    def __call__(self, x, y, out):
        ...


@cb.host
def add(x, y, out):
    AddKernel(128)[8](x, y, out)
```

类装饰器记录构造参数，并在设备函数构建时执行 `__init__`。构造参数用于传递编译期配置，运行时参数应通过 Kernel 入口方法传入。

## `@datastruct` {#datastruct}

`@datastruct` 用于声明一组具有名称和类型的字段。此类对象可以在普通 Python 中创建，也可以作为 DSL 程序或 Kernel 的参数。

```python
import cannbotdsl as cb


@cb.datastruct
class TileConfig:
    rows: cb.dtypes.int32
    columns: cb.dtypes.int32


config = TileConfig(rows=64, columns=128)
```

该装饰器适用于表示 tile 大小、循环次数或其他需要成组传递的配置。类体仅用于字段声明，不应定义默认值、继承关系或业务方法。

## `@aicpu_kernel` {#aicpu-kernel}

`@aicpu_kernel` 用于定义在 AI CPU 上执行的函数，通常用于生成元数据、计算调度信息或完成轻量控制任务。

```python
from cannbotdsl.aicpu import GmIn, GmOut, U32, aicpu_kernel


class MetadataArgs:
    source: GmIn(U32)
    result: GmOut(U32)
    count: U32


@aicpu_kernel
def build_metadata(a: MetadataArgs):
    for i in range(0, a.count):
        a.result[i] = a.source[i]
    return 0
```

参数类用于说明输入、输出和标量参数。装饰后的对象可以通过 `run_host(...)` 运行 Host 参考计算，也可以通过 `compile(...)` 编译 AI CPU Kernel。

## `@compile_cache` {#compile-cache}

`@compile_cache` 用于定义带缓存的算子编译函数。函数接收 dtype、shape 或算法模式等编译配置，并返回一次 `cb.compile(...)` 得到的 `ProviderCallable`。

```python
import cannbotdsl as cb
from opkit import compile_cache


@compile_cache
def get_add_program(dtype, length):
    spec = cb.TensorSpec((length,), dtype)
    return cb.compile(Add().run, spec, spec, spec)
```

使用相同配置再次调用时，可以直接复用对应程序：

```python
program = get_add_program(cb.dtypes.float16, 1024)
program(x, y, out)
```

编译函数应使用明确的参数列表，不支持 `*args`、`**kwargs` 或嵌套调用其他 `@compile_cache` 编译函数。每次执行必须恰好发起一次编译请求，并返回该请求产生的、尚未关闭的程序。

配置参数支持 DSL 或 PyTorch dtype、`bool`、`int`、有限浮点数和字符串。整数须落在有符号 64 位范围内；shape 应拆成各个整数维度传入。参数会按函数签名和默认值归一化，PyTorch dtype 会转换为对应的 DSL dtype。

需要释放编译函数缓存的程序句柄时，可以调用 `get_add_program.cache_clear()`；已有句柄会被关闭，再次使用时应重新调用编译函数获取程序。

## `@export` {#export}

`@export("name")` 用于声明算子发布入口。被装饰的函数不接收参数，函数体中列出需要包含在 Native 算子包中的编译配置。

```python
import cannbotdsl as cb
from opkit import export


@export("add")
def export_add():
    for dtype in (cb.dtypes.float16, cb.dtypes.float32):
        get_add_program(dtype, 1024)
```

装饰器参数是算子的发布名称，以字母或数字开头，可包含字母、数字、下划线、点和连字符。发布入口必须是无参数的同步函数。构建时会执行发布入口中列出的配置；仅导入包含该函数的模块不会触发编译。

## `@aclnn` {#aclnn}

`@aclnn("Name")` 用于声明 ACLNN 算子的 Host 入口。函数参数需要使用类型注解，函数体负责检查输入条件、准备输出，并调用已发布的算子程序。

```python
import torch
from opkit import aclnn


@aclnn("CannBotAdd")
def add(x: torch.Tensor, y: torch.Tensor):
    assert x.ndim == 1
    assert y.ndim == 1
    assert x.shape == y.shape
    assert x.dtype == y.dtype

    out = torch.empty_like(x)
    program = get_add_program(x.dtype, x.shape[0])
    program(x, y, out)
    return out
```

装饰器参数是生成接口时使用的算子名称。函数的参数注解定义输入类型，返回的 Tensor 或 Tensor 元组定义输出。普通 Python 调用会按照函数体执行；构建 ACLNN 接口时，使用同一函数声明生成相应接口。

输入注解还支持可选 Tensor、Tensor 列表、整数序列、`bool`、`int`、`float` 和字符串等类型。函数需要显式返回输出 Tensor，各返回分支的输出数量应一致；不支持直接返回输入参数或重复返回同一个输出。构建时会分析函数源码，函数体需使用 OpKit Host 编译器支持的语法。
