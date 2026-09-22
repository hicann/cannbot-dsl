<!-- changelog-cursor: ba3d3b5e58ebcc072f14d32e00dc6d403eb1c8cb -->

## 🔥 更新日志

### 【2026-09-21】
#### 文档 Documentation
- 【API 文档】扩充设备侧接口文档并重组 Kernel API 分类：Host API 新增平台信息（`get_platform_info`、`get_mem_size`）；Kernel API 划分为数据搬运、系统变量访问、同步与缓存控制三类，新增系统变量访问 9 个与同步与缓存控制 4 个接口页，原 `api/operations/` 下的接口页迁至 `api/kernel/`，侧边栏同步更新。

### 【2026-09-15】
#### 新特性 New Features
- 【flash_kda_metadata】新增 FlashKDA 配套的独立 AICPU 调度算子 `flash_kda_metadata()`：按 64 个 token 划分 chunk，生成各轮 batch、任务前缀和、核间任务区间与有效序列长度等调度信息，供 `flash_kda` 消费；不读取 Q/V/state 数值，也不计算 attention 输出，返回一维 int32 张量。支持 BNSD / BSND / TND 布局（TND 必传 `cu_seqlens`），约束 D = 128、Nv % Nqk == 0、Nv <= WORKSPACE_SLOTS（当前 861）；调度配置不变时 metadata 可在层间复用。共 8 个 NPU 用例，覆盖三种布局、变长尾块与 858/861/864 容量边界。

- 【fused_recurrent_kda_snapshot】新增 KDA decode 算子 `fused_recurrent_kda()`：计算 1～8 token 的 KDA decode，并把每个 token 的递归状态写入 state-pool 指定槽位，用于为候选分支保存完整状态的 Snapshot 协议。支持 BSND / BNSD / packed TND，D = 128、N <= 96、单序列长度不超过 8；`state` 为 `[pool_slots, N, 128, 128]`（BF16 或 FP32），返回 `(state, out)`，**state 原地更新**。测试覆盖三种布局与 BF16/FP32 state，并与 PyTorch golden 对照。

#### 特性增强 Feature Enhancement
- 【flash_kda】接口升级为「调度与计算解耦」：新增必选参数 `metadata` 与布局 `TND`，`flash_kda()` 不再生成调度信息、也不启动 AICPU，改为消费配套 `flash_kda_metadata` 的输出，可在调度配置相同的层间复用；序列长度不再要求 64 对齐，尾块由内核处理。**注意：`metadata` 为必选参数，属破坏性接口变更，旧调用方式不再可用。**

#### 测试框架 Test Framework
- 【测试】新增 `test/_samples_path.py` 共享加载器，为每个样例分配私有包前缀按路径加载，避免与环境中已安装的同名外部模块冲突；flash_kda 测试重写为 12 个 NPU 用例（布局精度、非对齐尾块、跨轮调度与 K 快照补零回归，容差 atol=rtol=5e-3），并内联 PyTorch golden 做输出与最终 state 的三方对照。

### 【2026-09-14】
#### 文档 Documentation
- 【文档站点】文档站从框架推进到可对外发布：新增 5 个内容分区（开始使用 / 样例 / API 文档 / 参与贡献 / 关于），覆盖项目定位与支持范围（NPU ARCH 3510 / Ascend 950PR / 950DT）、样例导航、测试与贡献流程与文档分批发布方式；首页改为 Hero + 特性卡片，落地亮/暗两套品牌主题与中文全文搜索。

- 【API 文档】确立公共接口文档的分类与编写模板：接口按调用位置与执行模型分为 Host API、Kernel API、AI CPU API 三类，新增通用接口文档模板 `docs/api/api-template.md`（要求以公开接口和验证结果为准，未经验证的产品不得标记为「支持」），并按模板落地首批接口页——数据搬运 `api/operations/data-movement/mem-copy` 与系统变量访问 `api/operations/system/get-core-id`。

### 【2026-09-11】
#### 文档 Documentation
- 【文档站点】搭建基于 VitePress 的文档站点框架（`docs/`），配置站点标题、描述、base 路径、`cleanUrls`、`lastUpdated`、页脚与 sitemap，并声明 `package.json` / `package-lock.json` 依赖。

### 【2026-09-10】
#### 新特性 New Features
- 【grouped_matmul】新增非量化分组矩阵乘算子 `group_matmul()`：$y_i[m_i,n_i] = x_i[m_i,k_i] \times weight_i[k_i,n_i]$，覆盖 S1–S6 六种场景，支持 group_type=-1（不分组）/ 0（M 轴分组 SPLIT_M）/ 2（K 轴分组 SPLIT_K），group_list_type 支持 0（cumsum 累积和）与 1（count 计数），group_list 为 None 时按各组张量首维隐式分组，并支持零大小组；x/weight/y 支持 float16、bfloat16，返回值恒为 `List[Tensor]`。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），性能对比 9.1.0 CANN 包内置 GroupedMatmul。

- 【quant_batch_matmul_mxfp8】新增 MXFP8 全量化矩阵乘算子 `npu_quant_matmul()`（QBMM MXFP8）：$C[M,N] = Dequant(A)[M,K] \times Dequant(B)[N,K]^T$，MXFP8 沿 K 轴每 32 个数据使用一个 E8M0 Scale，公开 Scale 接口将相邻两个 Scale 组成一个 pair（分组轴长 `ceil(K/64)`、末维 2）；a/b 支持 float8_e4m3fn 与 float8_e5m2，输出支持 float16、bfloat16、float32；四种 transposeA/transposeB 组合由 A/B 与 Scale 的 view shape 推导。面向 NPU ARCH 3510，性能对比 CANN 包内置 `npu_quant_matmul`。

#### 架构重构 Architecture Refactoring
- 【matmul】将 `samples/matmul/` 拆分为 `matmul/`（非量化 A16W16）与 `quant_matmul/`（MXFP8 全量化）两个子目录，测试同步迁移至 `test/matmul/matmul/` 与 `test/matmul/quant_matmul/`。

### 【2026-09-09】
#### 新特性 New Features
- 【kv_compress_epilog】新增 KV Cache 压缩更新算子 `kv_compress_epilog()`：将 bfloat16 激活值量化压缩后按 slotMapping 散写到 cache 对应行并**原地更新**，输出行布局为 `[rope bf16 128B][nope fp8 (d-64)B][scale][pad]`；nope 段每 64 个元素独立计算 scale 并量化为 FP8(e4m3)，rope 段保留 bfloat16；scale 段支持 bf16（quant_mode=0）与 e8m0（quant_mode=1）；约束 64 < d ≤ 8192 且 d % 64 == 0。面向 NPU ARCH 3510，测试覆盖 quant_mode × round_scale × 2 种 shape 共 8 个用例并逐字节对比 golden。

### 【2026-09-04】
#### 缺陷修复 Bug Fixes
- 【兼容性】适配 cannbotdsl 0.0.3：flash_attn、flash_kda、matmul、pointnet_sa、rms_norm、voxel_conv 各样例的 `from_torch_npu` 改由 `from torch import as_tensor` 提供，并同步更新相关测试用例。

### 【2026-08-07】
#### 特性增强 Feature Enhancement
- 【rms_norm】UB 缓冲深度由固定双缓冲改为按 UB 容量自适应计算（在 `DOUBLE_BUFFER_NUM` 与单缓冲之间择优），并同步更新算例图与测试用例。

### 【2026-08-05】
#### 新特性 New Features
- 【flash_attn】新增 Flash Attention 算子 `flash_attn()`：$O = softmax(QK^T \cdot scale)V$，D=128，Q/K/V/O 支持 float16、bfloat16；支持 mask_mode 0（全注意力）与 3（right-context causal），支持 GQA（N1 须为 N2 整数倍），layout_q / layout_kv / layout_out 可分别指定为 BNSD / BSND。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），测试覆盖 prefill、decode（S1=1）、MTP（1<S1≤4）与 causal 场景共 12 个代表性用例，并与 CANN 内置 FIA 做 msprof 性能对比。

- 【matmul】新增非量化矩阵乘算子 `matmul()`（A16W16）：$C[M,N] = A[M,K] \times B[N,K]^T$，A/B/C 支持 float16、bfloat16，输入为二维 contiguous 矩阵；L2 cache 按矩阵复用情况自适应开关；核内仅支持 transpose_a=False、transpose_b=True。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），性能对比 CANN 包内置 matmul 算子基础模板。

- 【rms_norm】新增 RmsNorm 归一化算子 `rms_norm()`：$rstd = 1/\sqrt{mean(x^2)+\epsilon}$，$y = x \cdot rstd \cdot \gamma$；x/gamma/y 支持 bfloat16、float16、float32，rstd 恒为 float32，归一化轴为 x 的最后 K 维且须与 gamma 形状一致，epsilon 默认 1e-6，返回 `(y, rstd)`；核内按 UB 容量与数据类型自适应选择 UB 全载或沿归一化轴列切分。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），性能对比 CANN BuiltIn 实现。

- 【flash_kda】新增 Kimi Delta Attention prefill 融合算子 `flash_kda()`：融合原始 gate 与 beta 的激活、Q/K 的 L2 normalize 以及完整的 Chunk KDA 计算，按 64-token chunk 切分并在 chunk 间递推状态，返回 `(out, final_state)`；q/k/v/g 为 BF16，initial_state / A_log / dt_bias 为 FP32，Dk = Dv = 128，S 为 64 的整数倍，支持 GQA（Nv % Nk == 0）与 BNSD/BSND layout。面向 Ascend 950，性能对比 H800 上的 FlashKDA 代码。

- 【voxel_conv】新增 VoxelConv 卷积算子 `voxel_conv()`：$C[N,Co,Ho,Wo] = \text{Conv2D}(x[N,Ci,Hi,Wi], filter[Co,CiG,Kh,Kw])$，来自 VoxelNet 点云 3D 目标检测的 Convolutional Middle Layers；数据格式为 NCHW 输入/输出、OIHW 权重，x 与 weight 支持 float16、bfloat16，支持 groups>1 分组卷积，stride ∈ [1,63]、padding 四侧独立 ∈ [0,255]、dilation ∈ [1,255] 且 H/W 独立。面向 NPU ARCH 3510（Ascend 950DT / Ascend 950PR），Conv1D 与 Conv3D 为后续扩展范围。

- 【pointnet_sa】新增 PointNet Set Abstraction 算子 `pointnet_sa()`：$feat[K, D_{out}] = \max_{j} \text{MLP}(points[K,j,D_{in}])$，对应 PointNet++ 点云层次化特征学习 SA 层中的 shared MLP + max-pool 模块（不含 Farthest Point Sampling 与 Ball Query）；输入 points (K,N_per_group,D_in) 与 weight (D_out,D_in)，输出 feat (K,D_out)，支持 float16、bfloat16。面向 NPU ARCH 3510（Ascend 950DT / Ascend 950PR）。

#### 架构重构 Architecture Refactoring
- 【voxel_conv】算子由 `conv2d` 更名为 `voxel_conv`，样例、测试与文档引用路径同步重命名。
