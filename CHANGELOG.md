## 🔥 更新日志

### 【2026-09-11】
#### 文档 Documentation
- 【文档站点】搭建基于 VitePress 的文档站点框架（`docs/`），配置站点标题、描述、base 路径、`cleanUrls`、`lastUpdated`、页脚与 sitemap，并声明 `package.json` / `package-lock.json` 依赖。

### 【2026-09-10】
#### 新特性 New Features
- 【grouped_matmul】新增非量化分组矩阵乘算子 `group_matmul()`：$y_i[m_i,n_i] = x_i[m_i,k_i] \times weight_i[k_i,n_i]$，覆盖 S1–S6 六种场景，支持 group_type=-1（不分组）/ 0（M 轴分组 SPLIT_M）/ 2（K 轴分组 SPLIT_K），group_list_type 支持 0（cumsum 累积和）与 1（count 计数），group_list 为 None 时按各组张量首维隐式分组，并支持零大小组；采用无 padding 设计（核内按组真实边界切片读写，无需 host 侧 padding 或对齐约束）与零物化（输入 tensorlist 原样传入 kernel，不做 cat/stack/contiguous），自适应滑动窗口多核调度配合偶数行 N 反转、count 跨组负载均衡与逐组 L2 cache 自适应开关；x/weight/y 支持 float16、bfloat16，返回值恒为 `List[Tensor]`。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），性能对比 9.1.0 CANN 包内置 GroupedMatmul。

- 【matmul/quant_matmul】新增 MXFP8 全量化矩阵乘算子 `npu_quant_matmul()`（QBMM MXFP8）：$C[M,N] = Dequant(A)[M,K] \times Dequant(B)[N,K]^T$，MXFP8 沿 K 轴每 32 个数据使用一个 E8M0 Scale，公开 Scale 接口将相邻两个 Scale 组成一个 pair（Scale 分组轴长度为 `ceil(K/64)`、最后一维为 2，对应 ScaleBDN / ScaleBND / ScaleAND / ScaleADN）；a/b 支持 float8_e4m3fn 与 float8_e5m2，输出支持 float16、bfloat16、float32；采用 `AL1_FULL_LOAD` 使 A 与 ScaleA 常驻 L1 并跨多个 N Tile 复用，B 与 ScaleB 沿 K 流式搬运，按 L1 容量选择 2/4 Buffer，L0A/L0B 双缓冲，L0C 容量足够时双缓冲，Cube-bound 的 FP16/BF16 场景同步调整 Scale 窗口并开启 UnitFlag；通过 A/B 与 Scale 的 view shape 推导四种 transposeA/transposeB 组合。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），性能对比 CANN 包内置 `npu_quant_matmul`。

#### 架构重构 Architecture Refactoring
- 【matmul】将 `samples/matmul/` 拆分为 `matmul/`（非量化 A16W16）与 `quant_matmul/`（MXFP8 全量化）两个子目录，测试同步迁移至 `test/matmul/matmul/` 与 `test/matmul/quant_matmul/`。

### 【2026-09-09】
#### 新特性 New Features
- 【kv_compress_epilog】新增 KV Cache 压缩更新算子 `kv_compress_epilog()`：将 bfloat16 激活值量化压缩后按 slotMapping 散写到 cache 对应行并**原地更新**，输出行布局为 `[rope bf16 128B][nope fp8 (d-64)B][scale][pad]`；nope 段每 64 个元素为一组独立计算 scale 并量化为 FP8(e4m3)，rope 段保留原始 bfloat16；scale 段支持 bf16（quant_mode=0）与 e8m0（quant_mode=1）两种格式，`round_scale=True` 时 scale 向上取整为 2 的幂、False 时不取整；slot_mapping 中 -1 表示跳过；约束 64 < d ≤ 8192 且 d % 64 == 0，headDim ≥ kvCacheCol。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），测试覆盖 quant_mode × round_scale × 2 种 shape 共 8 个用例并逐字节对比 golden（`round_scale=False` 接受 ±1 容差，`round_scale=True` 与 golden 完全一致），性能对比 CANN Built-in。

### 【2026-09-04】
#### 缺陷修复 Bug Fixes
- 【兼容性】适配 cannbotdsl 0.0.3：flash_attn、flash_kda、matmul、pointnet_sa、rms_norm、voxel_conv 各样例的 `from_torch_npu` 改由 `from torch import as_tensor` 提供，flash_kda 调整环境变量与常量定义位置，并同步更新相关测试用例。

### 【2026-08-07】
#### 特性增强 Feature Enhancement
- 【rms_norm】UB 缓冲深度由固定双缓冲改为按 UB 容量自适应计算（在 `DOUBLE_BUFFER_NUM` 与单缓冲之间择优），新增 `COL_TILE_CAP_LARGE`，并同步更新算例图与测试用例。

### 【2026-08-05】
#### 新特性 New Features
- 【flash_attn】新增 Flash Attention 算子 `flash_attn()`：$O = softmax(QK^T \cdot scale)V$，D=128，Q/K/V/O 支持 float16、bfloat16；支持 mask_mode 0（全注意力）与 3（right-context causal，causal mask 2048×2048 float32），支持 GQA（N1 须为 N2 整数倍），BNSD/BSND 三类 layout 可分别指定 layout_q / layout_kv / layout_out；测试覆盖 prefill、decode（S1=1）、MTP（1<S1≤4）与 causal 场景共 12 个代表性用例（group size g=1/2/8/10/16），并与 CANN 内置 FIA 做 msprof 性能对比。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT）。

- 【matmul】新增非量化矩阵乘算子 `matmul()`（A16W16）：$C[M,N] = A[M,K] \times B[N,K]^T$，A/B/C 支持 float16、bfloat16，输入为二维 contiguous 矩阵；基础实现采用自适应滑动窗口多核调度与 L1/L0 ping-pong 流水线，另提供 Stream-K 实现 `matmul_streamk()`（DP + SK 混合调度 DPSK），适合小 MN、大 K 场景；host 侧推导 baseM/baseN/baseK 与 L1/L0 切分及 double buffer，L2 cache 按矩阵复用情况自适应开关；核内仅支持 transpose_a=False、transpose_b=True。数据流 GM → L1(MTE2) → L0A/L0B(MTE1) → MMAD → L0C → GM(FIXPIPE)。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），性能对比 CANN 包内置 matmul 算子基础模板。

- 【rms_norm】新增 RmsNorm 归一化算子 `rms_norm()`：$rstd = 1/\sqrt{mean(x^2)+\epsilon}$，$y = x \cdot rstd \cdot \gamma$；x/gamma/y 支持 bfloat16、float16、float32，rstd 恒为 float32，归一化轴为 x 的最后 K 维且须与 gamma 形状一致，epsilon 默认 1e-6，返回 `(y, rstd)`；host 侧通过 `const_expr` 编译期分支选择 UB 全载或列切分 kernel 模板，核内按 UB 容量与数据类型自适应选择全载（整行驻留 UB）或列切分（沿归一化轴分 tile），device kernel 采用二分折叠归约 + 牛顿迭代 + Channel 流水；纯 Vector 算子，无 Cube。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），性能对比 CANN BuiltIn 实现。

- 【flash_kda】新增 Kimi Delta Attention prefill 融合算子 `flash_kda()`：融合原始 gate 与 beta 的激活、Q/K 的 L2 normalize 以及完整的 Chunk KDA 计算，按 64-token chunk 切分并在 chunk 间递推状态 $S_c = \text{diag}(\gamma_C)S_{c-1} + (K_c^r)^T(U - WS_{c-1})$，输出 $O_c = \tilde{Q}_cS_{c-1} + \text{tril}(\tilde{Q}_c\bar{K}_c^T)(U - WS_{c-1})$，返回 `(out, final_state)`；q/k/v/g 为 BF16（核内做行级 L2 normalize 与 sigmoid(beta)），initial_state / A_log / dt_bias 为 FP32，Dk = Dv = 128，S 为 64 的整数倍，支持 GQA（Nv % Nk == 0）与 BNSD/BSND layout。面向 Ascend 950，性能对比 H800 上的 FlashKDA 代码。

- 【voxel_conv】新增 VoxelConv 卷积算子 `voxel_conv()`：$C[N,Co,Ho,Wo] = \text{Conv2D}(x[N,Ci,Hi,Wi], filter[Co,CiG,Kh,Kw])$，来自 VoxelNet 点云 3D 目标检测的 Convolutional Middle Layers；数据格式为 NCHW 输入/输出、OIHW 权重，x 与 weight 支持 float16、bfloat16，支持 groups>1 分组卷积，stride ∈ [1,63]、padding 四侧独立 ∈ [0,255]、dilation ∈ [1,255] 且 H/W 独立，默认 tile_shape (16,16,16)、block_num 32；采用 GM → L1(DN2NZ) → L0A(Load3D im2col)/L0B(MTE1) → L0C(MMAD) → GM(NZ2DN) 数据流，并以扁平分核将所有 (batch, group, M, N) tile 线性化后 round-robin 分配。面向 NPU ARCH 3510（Ascend 950DT / Ascend 950PR），Conv1D 与 Conv3D 为后续扩展范围。

- 【pointnet_sa】新增 PointNet Set Abstraction 算子 `pointnet_sa()`：$feat[K, D_{out}] = \max_{j} \text{MLP}(points[K,j,D_{in}])$，对应 PointNet++ 点云层次化特征学习 SA 层中的 shared MLP + max-pool 模块（不含 Farthest Point Sampling 与 Ball Query）；MLP 等价于 1x1 卷积并映射为 batched matmul（M=D_out, N=K\*N_per_group, K=Ci），max-pooling 由 torch NPU 原生算子完成，多核采用与 matmul_basic 一致的 slide window 调度；输入 points (K,N_per_group,D_in) 与 weight (D_out,D_in)，输出 feat (K,D_out)，支持 float16、bfloat16。面向 NPU ARCH 3510（Ascend 950DT / Ascend 950PR）。

#### 架构重构 Architecture Refactoring
- 【voxel_conv】算子由 `conv2d` 更名为 `voxel_conv`，样例、测试与文档引用路径同步重命名。

#### 测试框架 Test Framework
- 【测试】新增 flash_attn、matmul、rms_norm、flash_kda、voxel_conv、pointnet_sa 的 kernel 正确性测试用例。

#### 文档 Documentation
- 【文档】新增仓库 README 与 LICENSE，确立「复杂算子示例集合」定位；精简 README 结构（移除环境部署 / 快速入门 / 测试章节），更新算子列表与目录结构。
