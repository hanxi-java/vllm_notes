# MegaKernel 学习笔记：GPU SM 与 Ascend NPU 计算架构

> 参考文章：[Weave：面向计算通信重叠的 MoE MegaKernel 细粒度动态 SM 调度，较 DeepEP 1.95 倍层加速，1.12 倍端到端提速](https://mp.weixin.qq.com/s/ShGX7mdhwbNY0MoYY00L1Q)

## 目录

- [问题一：GPU 的流多处理器（SM）是什么](#问题一gpu-的流多处理器sm是什么)
- [回答一：GPU SM 的结构与调度](#回答一gpu-sm-的结构与调度)
- [问题二：Ascend 950 是否有类似结构，与 GPU 有何区别](#问题二ascend-950-是否有类似结构与-gpu-有何区别)
- [回答二：Ascend 950 与 GPU 的结构对比](#回答二ascend-950-与-gpu-的结构对比)
- [参考资料](#参考资料)

## 问题一：GPU 的流多处理器（SM）是什么

[这篇微信文章](https://mp.weixin.qq.com/s/ShGX7mdhwbNY0MoYY00L1Q)中提到的 GPU 的流多处理器（SM）是什么？

## 回答一：GPU SM 的结构与调度

### 1. SM 是什么

这里的 **SM** 是 NVIDIA GPU 中的 **Streaming Multiprocessor**，通常译为“流式多处理器”或“流多处理器”。它是 GPU 中负责执行 CUDA 线程的核心硬件计算单元。

可以把一块 NVIDIA GPU 粗略理解为由许多 SM 并排组成：

```text
一块 GPU
├── SM 0
├── SM 1
├── SM 2
├── ...
└── SM N-1
```

CPU 通常依靠少量功能复杂、擅长低延迟串行执行的 CPU Core；GPU 则依靠较多的 SM 同时处理大量并行任务。每个 SM 又能够让大量 CUDA 线程驻留，并以 warp 为单位进行调度。

因此，SM 不是一个单独的“CUDA Core”，而更像是一个能够独立管理和执行大量线程的小型并行处理器。

### 2. 一个 SM 内部包含什么

一个 SM 通常包含以下几类资源：

- **CUDA Core**：执行 FP32、INT32 等常规标量算术指令；
- **Tensor Core**：执行矩阵乘加，主要用于深度学习中的 GEMM；
- **Warp Scheduler 和 Dispatch Unit**：选择可执行的 warp，并向执行单元发射指令；
- **Register File**：保存当前驻留线程的寄存器状态；
- **Shared Memory / L1 Cache**：供同一 thread block 内的线程快速交换和复用数据；
- **Load/Store Unit**：执行全局内存、共享内存等数据加载和存储；
- **Special Function Unit（SFU）**：处理指数、倒数、三角函数等特殊运算。

其逻辑关系可简化表示为：

```text
SM
├── Warp Scheduler / Dispatch
├── CUDA Core
├── Tensor Core
├── Load/Store Unit
├── Special Function Unit
├── Register File
└── Shared Memory / L1 Cache
```

不同代际 GPU 的 SM 结构并不完全相同。例如，Turing、Ampere、Hopper 和 Blackwell 中每个 SM 所包含的 CUDA Core、Tensor Core、调度器和缓存配置都有差异。因此不能脱离具体架构，把“一个 SM”固定等同于某个不变数量的 CUDA Core。

### 3. CUDA 程序如何映射到 SM

CUDA 的执行层次可以简化为：

```text
Kernel
└── Grid
    ├── Thread Block 0 ──→ 某个 SM
    ├── Thread Block 1 ──→ 某个 SM
    ├── Thread Block 2 ──→ 某个 SM
    └── ...
```

一个 thread block 又包含多个线程，这些线程会被组织成 warp：

```text
Thread Block
└── 多个 Warp
    └── 每个 Warp 通常包含 32 个线程
```

主要执行过程如下：

1. CPU 启动一个 CUDA kernel，kernel 产生一个由许多 thread block 组成的 grid。
2. GPU 的硬件调度器把不同的 thread block 分派到有可用资源的 SM。
3. 一个 block 被分派到某个 SM 后，通常会在该 SM 上驻留并执行到结束，不会在执行一半时迁移到另一个 SM。
4. 如果寄存器、共享内存和线程槽位足够，一个 SM 可以同时驻留多个 block。
5. SM 内部的 warp scheduler 从已经就绪的 warp 中选择指令，发给 CUDA Core、Tensor Core或访存单元执行。
6. 当某个 warp 等待内存访问或同步时，调度器可以执行其他已经就绪的 warp，以隐藏等待延迟。

因此，GPU 的大规模并行不是“每个 CUDA Core 独立运行一个完整程序”，而是：

> 多个 SM 并行执行多个 thread block；每个 SM 再以 warp 为单位调度大量线程，最后由 SM 内部的不同执行单元完成具体指令。

### 4. Weave 文章中的 SM 是什么含义

文章讨论的 Weave 面向专家并行 MoE 推理。在专家并行场景中，一个 MoE 层主要包含两类工作：

- **通信工作**：把 token dispatch 到持有目标专家的其他 GPU，并在专家计算完成后执行 combine；
- **计算工作**：执行专家网络的 GEMM、激活函数以及后续 GEMM。

为了重叠通信和计算，已有方案通常把一部分 SM 留给通信 kernel，把剩余 SM 用于专家计算。假设参与 MegaKernel 的 SM 总数为 `N`，其中 `c` 个 SM 被分配给通信任务，那么可以表示为：

```text
GPU 上参与执行的 N 个 SM

SM 0 ... SM c-1       → 通信 worker
SM c ... SM N-1       → 计算 worker
```

也就是：

```text
通信 SM 数量 = c
计算 SM 数量 = N - c
```

文章中的 **Communication SMs**，是正在运行 dispatch、combine 或相关通信推进代码的 SM；**Compute SMs**，是正在运行专家 GEMM 和激活函数等计算代码的 SM。

这里调度的粒度是 SM，而不是单独的 CUDA Core。Weave 通过 persistent MegaKernel 保持一组长期驻留的 thread block，并把这些 block 对应的 SM 组织成不同 worker 组，从而在同一个 MegaKernel 内协调通信与计算任务。

### 5. 为什么需要动态分配 SM

MoE 的路由结果是动态的。不同层、不同推理批次以及同一时刻不同 GPU 接收到的 token 数量都可能不同，因此每块 GPU 的通信量和专家计算量也会变化。

如果始终采用固定划分，例如固定让 20 个 SM 做通信，就可能出现两类浪费：

1. **通信任务较少**：通信 SM 很快完成任务后空闲，但计算 SM 仍在执行大量 GEMM。
2. **通信任务较多**：通信 SM 数量不足，NVLink/NVSwitch 带宽没有被充分利用，计算 SM 可能等待输入数据。

Weave 在路由完成后已经知道每块 GPU 需要发送、接收和计算多少 token，于是可以结合硬件测得的“SM 数量—通信带宽”和“SM 数量—计算吞吐”曲线，在运行时选择更合适的 `c`。

论文中的调度可以分成两个层面：

- **空间调度（Spatial Scheduling）**：决定多少个 SM 做通信、多少个 SM 做计算；
- **时间调度（Temporal Scheduling）**：决定通信和计算任务在什么时间执行，以及怎样利用暂时空闲的 SM。

例如：

```text
路由结果显示通信量较大
→ 增加通信 SM 数量 c
→ 更快推进 dispatch/combine

路由结果显示专家计算量较大
→ 减少通信 SM 数量 c
→ 把更多 SM 留给 GEMM
```

Weave 还会把 token 划分为多个 chunk，使某个 chunk 的专家计算完成后，可以立即开始该 chunk 的 combine，而不必等待所有 token 全部计算完毕。此外，在 dispatch 结束到 combine 开始之间，如果通信 worker 暂时没有通信任务，它们还可以执行一部分 GEMM tile。论文把这种做法称为 **bubble stealing**，目的是填补 SM 的空闲气泡。

### 6. “通信 SM”到底在做什么

“通信 SM”这个说法容易让人误以为 SM 本身变成了网卡，或者数据完全由 SM 在 NVLink 上搬运。实际情况不是这样。

更准确地说：

> 通信 SM 运行负责发起、协调或推进 GPU 间通信的 CUDA 线程；真正的数据传输还会经过 GPU 内存子系统、NVLink/NVSwitch、网络接口以及相应通信硬件。

因此，SM 的角色仍然是执行程序。所谓“把 SM 分给通信”，表示让这些 SM 执行通信相关的 thread block，而不是在物理上改变 SM 的硬件类型。

同一个 SM 在不同时间可以承担不同类型的任务。例如，一个通信 worker 完成 dispatch 后，可以临时执行 GEMM tile；当 combine 条件满足时，再回到通信任务。Weave 的重点正是利用 persistent MegaKernel 在应用内部细粒度地协调这种角色和任务切换。

### 7. 容易混淆的概念

| 概念 | 含义 |
| --- | --- |
| SM | GPU 内部完整的多线程计算和调度单元 |
| CUDA Core | SM 内部执行普通标量算术指令的执行通道 |
| Tensor Core | SM 内部执行矩阵乘加的专用执行单元 |
| Warp | NVIDIA GPU 的线程调度单位，通常包含 32 个线程 |
| Thread Block | 一组协作线程，整体被分派到某一个 SM 上执行 |
| Kernel | 在 GPU 上执行的函数，一次启动会生成由许多 block 组成的 grid |
| CUDA Stream | 软件层面的异步任务队列，不是硬件 SM |
| MegaKernel | 把原本多个 kernel 的工作融合到一个大型、通常长期驻留的 kernel 中 |

尤其需要注意：**Streaming Multiprocessor 中的“Streaming”和 CUDA Stream 不是同一个概念。**

- SM 是硬件计算单元；
- CUDA Stream 是软件提交和排序 kernel、memcpy 等操作的队列。

### 8. 一句话总结

> SM 可以理解为 GPU 上可独立调度大量线程的“并行计算工人组”。在 Weave 中，系统根据每层 MoE 的实际路由结果，动态决定多少个 SM 运行通信任务、多少个 SM 运行专家计算，并让暂时空闲的通信 SM 帮助执行计算，以提高计算—通信重叠程度并减少空闲气泡。

## 问题二：Ascend 950 是否有类似结构，与 GPU 有何区别

上面提到了 GPU 并行计算的单元和调度，如 SM、CUDA Core、Tensor Core、Warp Scheduler 等。那么，以 Ascend 950 为例，NPU 上是不是也有类似的结构？它和 GPU 有什么区别？

## 回答二：Ascend 950 与 GPU 的结构对比

有类似结构，而且 **Ascend 950 的 Vector Core 已经引入了明显类似 GPU 的 SIMT、Thread、Warp 和 Shared Memory 能力**。但 Ascend 950 并不是简单地把 NVIDIA SM 换了一个名字。两者最核心的设计思路不同：

> NVIDIA GPU 以 **SM、Warp 和 SIMT 线程调度**为中心；Ascend 950 以 **AI Core、Cube/Vector 异构计算单元和显式数据搬运流水线**为中心，并在 Vector Core 中引入 SIMT，作为处理不规则逻辑的补充。

下文所说的 Ascend 950，具体对应当前 CANN 文档公开支持的 **Ascend 950PR / Ascend 950DT**。

### 1. 两种处理器的总体结构

NVIDIA GPU 可以简化为：

```text
GPU
└── 多个 SM
    ├── Warp Scheduler
    ├── CUDA Core：FP/INT 标量计算
    ├── Tensor Core：矩阵乘加
    ├── Load/Store Unit
    ├── Register File
    └── Shared Memory / L1
```

Ascend 950PR/950DT 采用分离模式。一个逻辑 AI Core 由不同类型的物理计算核组合而成：

```text
Ascend NPU
└── 多个逻辑 AI Core
    ├── AI Cube Core，简称 AIC
    │   ├── Scalar 调度单元
    │   ├── Cube 矩阵计算单元
    │   ├── MTE 数据搬运单元
    │   ├── L1
    │   ├── L0A / L0B / L0C
    │   └── FixPipe / BT Buffer 等
    │
    └── 一个或多个 AI Vector Core，简称 AIV
        ├── Scalar 调度单元
        ├── Vector SIMD 计算单元
        ├── SIMT 线程执行能力
        ├── MTE 数据搬运单元
        ├── Unified Buffer，UB
        └── Vector Register
```

需要区分三个术语：

- **AI Core**：由 Cube Core 和 Vector Core 按一定比例组成的逻辑计算单元；
- **AIC**：AI Cube Core，主要负责矩阵计算；
- **AIV**：AI Vector Core，主要负责向量、控制以及 Ascend 950 新增的 SIMT 计算。

因此，在 profiling 数据中看到的 `AIC`、`AIV` 通常对应后两类物理核心，不能把二者都简单理解成完整 AI Core。

### 2. GPU 与 Ascend 950 的近似映射

下面的关系只能用来帮助理解，不能视为严格的硬件等价关系。

| NVIDIA GPU | Ascend 950 中较接近的结构 | 主要作用 | 是否严格等价 |
| --- | --- | --- | --- |
| SM | 逻辑 AI Core；SIMT 语境下更接近 AIV | 承载并行 kernel 工作 | 否 |
| CUDA Core | AIV 内的 Vector/SIMT 执行资源 | 普通算术、向量和线程计算 | 否 |
| Tensor Core | AIC 内的 Cube Unit | 矩阵乘加 | 功能接近，组织方式不同 |
| Warp | Ascend 950 SIMT Warp | 一组共同执行指令的线程 | 很接近，通常为 32 个线程 |
| Warp Scheduler | AIV 内部 SIMT 调度机制；传统路径还依赖 Scalar Unit | 调度和发射指令 | 官方抽象不同，不能直接等同 |
| Shared Memory | AIV 的 UB 中用于 SIMT 共享的空间 | 线程间共享数据 | 接近 |
| Register File | AIV 可编程向量寄存器及 SIMT 寄存器 | 保存线程和向量数据 | 接近但组织不同 |
| Load/Store Unit | MTE、DMA 和相关访存流水线 | 搬运数据 | 思路不同 |
| CUDA Thread Block | Ascend SIMT Thread Block；传统 SIMD 中还有 AI Core 级 Block | 工作划分和多核并行 | 需要区分编程模式 |

如果讨论传统 Ascend SIMD/Cube 算子，最接近 GPU SM 的是“逻辑 AI Core”；如果讨论 Ascend 950 新增的 thread、warp、shared memory 和 SIMT 执行，则 AIV/Vector Core 更像 GPU SM。

### 3. Cube Unit 类似 Tensor Core，但所处层级不同

两者最直观的功能映射是：

```text
GPU Tensor Core  ≈  Ascend Cube Unit
```

它们都为 GEMM、卷积、Attention 矩阵运算和 MoE 专家矩阵乘提供高密度矩阵计算能力。以 FP16 为例，Ascend Cube 的典型基础计算块是 `16 × 16` 矩阵。

但其硬件组织方式不同：

```text
NVIDIA GPU：
SM
├── CUDA Core
├── Tensor Core
└── 其他执行单元

Ascend 950：
逻辑 AI Core
├── AIC：包含 Cube 的矩阵计算核
└── AIV：Vector/SIMT 计算核
```

因此，不能说“AIC 就是一个 Tensor Core”。更准确的说法是：

> AIC 是一个包含 Scalar、Cube、MTE 和本地存储的矩阵计算核心；其中的 Cube Unit 才在功能上最接近 GPU Tensor Core。

### 4. Ascend 950 已经支持 Thread 和 Warp

传统 Ascend 算子主要采用两层并行模型：

```text
外层：多个 AI Core 执行 SPMD
内层：Vector/Cube 执行 SIMD
```

Ascend 950PR/950DT 在 AIV 上新增了公开的 SIMT 编程能力，包括：

- Grid；
- Thread Block；
- Thread；
- Warp；
- Lane ID；
- Warp shuffle；
- Shared Memory；
- 线程级寄存器和离散访存；
- 类似 `<<<...>>>` 的 SIMT 启动配置。

CANN 文档将 Warp 定义为执行相同指令的一组线程，相关接口以 32 个线程为一个 Warp。Runtime 还公开了以下硬件属性：

- `ACL_DEV_ATTR_WARP_SIZE`；
- `ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE`；
- `ACL_DEV_ATTR_MAX_GRID_DIM_X`；
- 每个 Vector Core 可用的 UB 大小。

因此，在 Ascend 950 AIV 的 SIMT 路径上，可以建立更直接的映射：

```text
CUDA Grid          ↔ Ascend SIMT Grid
CUDA Thread Block  ↔ Ascend SIMT Thread Block
CUDA Warp          ↔ Ascend SIMT Warp
CUDA Thread        ↔ Ascend SIMT Thread
Shared Memory      ↔ AIV UB 中的 Shared Memory
```

但对于 Warp Scheduler 需要谨慎。公开文档确认了 Warp、线程驻留和 Warp 级指令，却没有把某个独立硬件模块统一描述为 NVIDIA 式的 `Warp Scheduler`。稳妥的表述是：

> AIV 内部存在支持 Warp 执行所需的 SIMT 调度机制，但在缺少更底层公开资料时，不能把它的微架构直接画成 NVIDIA Warp Scheduler 的复制品。

### 5. 最根本的差异：SIMT 是主体还是补充

NVIDIA GPU 的核心执行逻辑是：

```text
大量线程
→ 组成 Warp
→ Warp Scheduler 选择就绪 Warp
→ 发给 CUDA Core、Tensor Core或访存单元
```

当一个 Warp 等待显存访问时，硬件可以切换到其他就绪 Warp，通过大量并发驻留线程隐藏延迟。即使运行 Tensor Core 指令，Tensor Core 仍处于 SM 和 Warp 的整体执行模型中。

Ascend 950 的主要吞吐仍来自：

- Cube 矩阵计算；
- Vector SIMD 计算；
- MTE 数据搬运；
- 各条流水线的重叠。

其混合模型可以表示为：

```text
Ascend 950
├── 规则、连续、计算密集
│   ├── Cube
│   └── Vector SIMD
│
└── 不规则、复杂控制、离散访存
    └── Vector SIMT
```

根据 CANN 文档，Cube 和 Vector SIMD 共同提供超过 90% 的主要计算能力；SIMT 主要用于分支、索引映射、scatter/gather 和离散访存等不规则场景。SIMT 并不是用来替换 Cube/SIMD 主路径，而是补充其灵活性。

### 6. 两者的调度与延迟隐藏方式不同

GPU 更侧重线程和 Warp 调度：

```text
Grid
├── Block 0 → SM
├── Block 1 → SM
├── Block 2 → SM
└── ...
```

进入 SM 后，Warp Scheduler 会在多个驻留 Warp 之间动态选择可运行任务。GPU 优化通常重点关注：

- occupancy；
- active warps；
- block 数量；
- register pressure；
- shared memory 占用；
- warp divergence；
- Tensor Core utilization。

Ascend 的传统 SIMD/Cube 算子更侧重“数据切块 + 异构流水线调度”：

```text
1. Tiling
2. CopyIn：GM → L1/UB
3. Compute：Cube 或 Vector
4. CopyOut：L1/UB → GM
```

AI Core 内部存在多条可以异步并行的流水线：

```text
PIPE_S       Scalar
PIPE_V       Vector
PIPE_M       Cube
PIPE_MTE1    L1 → L0A/L0B
PIPE_MTE2    GM → L1/UB
PIPE_MTE3    UB → GM
PIPE_FIX     L0C → GM/L1
```

Scalar 负责地址、循环和指令发射；Cube、Vector、MTE 从各自的指令队列异步执行。当流水线之间存在数据依赖时，需要通过 Event、`SetFlag/WaitFlag`，或 Ascend 950 支持的 `Lock/Unlock` 建立同步。

两者隐藏延迟的主要方式可以对比为：

```text
GPU：
一个 Warp 等待访存
→ 切换到另一个就绪 Warp

Ascend SIMD/Cube：
MTE 搬运下一块数据
+ Cube/Vector 计算当前块
+ MTE 写回上一块结果
→ 多流水线并行
```

Ascend 950 新增 SIMT 后，也可以在 AIV 内利用 Thread/Warp 并行，但其高性能主路径仍非常强调 Tiling、Local Memory 和流水线重叠。

### 7. 两者的存储层次和数据搬运方式不同

GPU 常见的存储层次可以简化为：

```text
HBM / Global Memory
        ↓
       L2
        ↓
SM: L1 / Shared Memory
        ↓
     Registers
```

Ascend 950 的 Cube 矩阵计算路径更接近：

```text
GM
 ↓ MTE2
L1
 ↓ MTE1
L0A / L0B
 ↓
Cube
 ↓
L0C
 ↓ FixPipe
GM 或 L1
```

传统 Vector SIMD 路径是：

```text
GM → UB → Vector → UB → GM
```

Ascend 950 新增寄存器级 Vector 路径：

```text
GM → UB → Register → Vector → Register → UB → GM
```

Ascend C 开发者通常需要更明确地控制数据 Tiling、GM 到 L1/UB 的搬运、L0A/L0B/L0C 的组织、Cube/Vector 协同、双缓冲以及流水线同步。因此，Ascend 算子的常见性能问题包括：

- MTE 搬运瓶颈；
- Cube 等待数据；
- Vector 与 Cube 负载不平衡；
- UB/L1 容量或复用不合理；
- 同步指令过多；
- AIC/AIV 比例不合适；
- Tiling 导致尾块或核间负载不均。

### 8. 一个直观类比

可以把 NVIDIA SM 想成一个综合车间：

```text
一个 SM 车间
├── 普通算术单元：CUDA Core
├── 矩阵机器：Tensor Core
├── 调度员：Warp Scheduler
└── 仓库：Register + Shared Memory
```

Ascend 950 更像把矩阵车间和向量车间拆开：

```text
一个逻辑 AI Core
├── AIC 矩阵车间
│   ├── Cube
│   ├── 自己的 Scalar
│   └── L1/L0/MTE
│
└── AIV 向量车间
    ├── Vector SIMD
    ├── SIMT Thread/Warp
    ├── 自己的 Scalar
    └── UB/Register/MTE
```

GPU 倾向于在一个 SM 内让 Warp 使用多种执行单元；Ascend 950 更强调把矩阵和向量工作分给不同类型的核心，再通过编译器、Runtime 和 kernel 内同步进行协同。

### 9. 对 MegaKernel 的影响

Weave 在 GPU 上可以把近似同构的 SM 分为：

```text
部分 SM → 通信
部分 SM → GEMM
```

在 Ascend 950 上，不能机械地改成“部分 AI Core 做通信、部分 AI Core 做 GEMM”，因为主要资源本身是异构的：

```text
AIC      → 更适合专家 GEMM/Cube
AIV      → 更适合路由、激活、索引和数据重排
MTE      → 片内及片外数据搬运
通信资源 → HCCL、互联链路及相应通信引擎
```

所以 Ascend 版本 MegaKernel 更可能需要联合考虑：

- 启动多少 AIC 和 AIV；
- AIC:AIV 采用 `1:0`、`0:1`、`1:1` 还是 `1:2`；
- AIV 运行 SIMD 还是局部 SIMT；
- Cube、Vector、MTE 与通信资源怎样组成流水线；
- AIC/AIV 之间的数据是否需要经 GM 传递；
- 是否能让暂时空闲的 AIV 执行其他 Vector/SIMT 工作。

Ascend C 的 `__mix__(cube, vec)` 体现了这种异构资源组合。例如：

```cpp
__mix__(1, 2)
```

表示融合算子以一个 Cube Core 配两个 Vector Core 的方式启动。它有一点像资源分区，但不是从 GPU 的同构 SM 池中取出 `c` 个 SM 做通信、其余做计算。

### 10. 总结

可以用下面的关系帮助记忆：

```text
SM                 ≈ 逻辑 AI Core；SIMT 语境下更接近 AIV
Tensor Core        ≈ Cube Unit
CUDA Core          ≈ Vector/SIMT 执行资源
Warp               ≈ Ascend 950 SIMT Warp
Shared Memory      ≈ AIV 的 UB Shared Memory
Warp Scheduler     ≈ AIV 内部 SIMT 调度机制，但不能严格等同
Load/Store Unit    ≈ MTE/DMA 流水线
```

一句话概括：

> GPU 是“以 Warp/SIMT 为中心，在 SM 内选择不同执行单元”；Ascend 950 是“以 Cube、Vector、MTE 异构流水线为中心，并在 AIV 中加入 Warp/SIMT 作为不规则计算的补充”。两者的能力正在靠近，但硬件组织、调度粒度和性能优化方法仍有明显区别。

## 参考资料

1. [微信文章：Weave——面向计算通信重叠的 MoE MegaKernel 细粒度动态 SM 调度](https://mp.weixin.qq.com/s/ShGX7mdhwbNY0MoYY00L1Q)
2. [Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap](https://arxiv.org/abs/2609.21483)
3. [NVIDIA CUDA C++ Programming Guide：Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/04-special-topics/hardware-implementation.html)
4. [NVIDIA Turing GPU Architecture：Streaming Multiprocessor Architecture](https://developer.nvidia.com/blog/nvidia-turing-architecture-in-depth/)
5. [Ascend AI Core 基本架构：分离模式、AIC 与 AIV](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/advanced_programming/hardware_implementation/basic_architecture.md)
6. [Ascend 950 AI Core SIMD 编程模型与硬件组件](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/programming_model/ai_core_simd_programming/overview.md)
7. [Ascend 950 SIMD 与 SIMT 混合编程](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/programming_guide/advanced_programming/advanced_ai_core_programming_model/simd_simt_hybrid_programming/overview.md)
8. [Ascend 950 SIMT Warp Shuffle API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMT-API/Warp_functions/Warp_shfl_functions/asc_shfl_up.md)
9. [Ascend AI Core 流水线与核内同步](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/intra_core_sync/intra_core_synchronization_capability_overview.md)
