# vLLM 并行策略学习笔记（DP / TP / PP / EP）

> 基于 vLLM V1 代码（`C:\Code\vllm`）阅读整理。
> 示例参数：`vllm serve --data-parallel-size 2 --tensor-parallel-size 2 --pipeline-parallel-size 2 --enable-expert-parallel`

---

## 第一部分：EngineCore / Scheduler / Executor / Worker 的创建关系

### 1.1 五层树形结构

```
API Server 前端 (1)
   └── EngineCore 进程        ← 数量由 DP 决定
         ├── Scheduler        ← 每个 EngineCore 内 1 个
         └── Executor         ← 每个 EngineCore 内 1 个
               └── Worker 进程 ← 数量由 TP×PP 决定
```

| 对象 | 创建代码 | 数量规则 |
|---|---|---|
| EngineCore | `vllm/v1/engine/utils.py: launch_core_engines()` → `CoreEngineProcManager` 为每个 DP rank fork 一个子进程 | = **DP size** |
| — DP 下的具体类 | `vllm/v1/engine/core.py:1288`：DP>1 **且 MoE 模型** → `DPEngineCoreProc`（需跨 DP 锁步）；非 MoE → 普通 `EngineCoreProc`，互相视为 DP=1 | — |
| Scheduler | `vllm/v1/engine/core.py:153`，在 `EngineCore.__init__` 中创建 | **每个 EngineCore 恰好 1 个** |
| Executor | `vllm/v1/engine/core.py:125` `executor_class(vllm_config)`；backend 默认 `"mp"`（`vllm/config/parallel.py:889`）→ `MultiprocExecutor` | **每个 EngineCore 恰好 1 个** |
| Worker | `vllm/v1/executor/multiproc_executor.py:118`：`world_size = TP × PP × PCP`，逐 rank `WorkerProc.make_worker_process()` 起子进程 | 每个 Executor 管 **TP×PP** 个 |
| EP | `vllm/distributed/parallel_state.py:1890`：只在已有 Worker 上建通信组，大小 = **DP × PCP × TP**，按 PP stage 分层 | **不创建任何新进程/对象** |

### 1.2 代入参数（DP=2, TP=2, PP=2, EP 开启）后的对象清单

前提：`--enable-expert-parallel` 说明是 MoE 模型，所以 EngineCore 用 `DPEngineCoreProc`，且 DP rank0 所在节点会多起一个 **DPCoordinator 进程**（`utils.py:1110`，MoE+DP>1 时做 wave 同步）。

- 每个 DP rank 内 world_size = TP×PP = **4**
- 总 GPU 进程 = 2 × 4 = **8**

| 对象 | 数量 |
|---|---|
| API Server 前端（AsyncLLM + MPEngineCoreClient） | 1 |
| DPCoordinator 进程 | 1 |
| **EngineCore（DPEngineCoreProc）** | **2** |
| **Scheduler** | **2** |
| **Executor（MultiprocExecutor）** | **2** |
| **Worker（WorkerProc）** | **8** |

### 1.3 结构图

```
                ┌────────────────────────────────────────┐
                │   API Server 进程 (vllm serve 主进程)    │
                │   AsyncLLM + MPEngineCoreClient (ZMQ)   │
                └──────┬───────────────────────┬─────────┘
                       │ 请求分发(负载均衡)      │ wave 同步/统计
             ┌─────────▼──────────┐            │
             │ DPCoordinator 进程  │◄───────────┘
             │ (仅 DP rank0 启动)  │
             └─────────┬──────────┘
        ┌──────────────┴───────────────┐
        ▼                              ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│ EngineCore_DP0 进程      │  │ EngineCore_DP1 进程      │
│ (DPEngineCoreProc)      │  │ (DPEngineCoreProc)      │
│ ┌─────────────────────┐ │  │ ┌─────────────────────┐ │
│ │ Scheduler ×1        │ │  │ │ Scheduler ×1        │ │
│ │ 调度请求/KV块管理    │ │  │ │ 调度请求/KV块管理    │ │
│ └─────────┬───────────┘ │  │ └─────────┬───────────┘ │
│ ┌─────────▼───────────┐ │  │ ┌─────────▼───────────┐ │
│ │ MultiprocExecutor×1 │ │  │ │ MultiprocExecutor×1 │ │
│ │ RPC广播/收集Worker结果│ │  │ │ RPC广播/收集Worker结果│ │
│ └─┬─────┬─────┬─────┬─┘ │  │ └─┬─────┬─────┬─────┬─┘ │
└───┼─────┼─────┼─────┼───┘  └───┼─────┼─────┼─────┼───┘
    ▼     ▼     ▼     ▼          ▼     ▼     ▼     ▼
   W0    W1    W2    W3         W4    W5    W6    W7
  GPU0  GPU1  GPU2  GPU3       GPU4  GPU5  GPU6  GPU7
  PP0   PP0   PP1   PP1        PP0   PP0   PP1   PP1
  TP0   TP1   TP0   TP1        TP0   TP1   TP0   TP1

  (rank 编排: TP 最内层 — rank0~1 = PP stage0, rank2~3 = PP stage1)
```

### 1.4 四类对象的职能

**EngineCore（DPEngineCoreProc）—— 每个 DP rank 一台"发动机"**
- 拥有并驱动 Scheduler + Executor，跑请求处理主循环（`step()`；PP 下用 `step_with_batch_queue` 异步调度消除流水线气泡）
- 通过 ZMQ 与前端握手、收请求、回输出
- DP 特有：与 DPCoordinator 做 wave 锁步，没有真实请求时也要发 dummy batch 陪跑，保证跨 DP 的 EP 通信对齐

**Scheduler —— 每个引擎内唯一的"调度大脑"**
- 管理 waiting/running 队列、抢占、chunked prefill、prefix caching
- 分配本 DP rank 全部 GPU 的 KV cache block
- 每步产出 `SchedulerOutput` 交给 Executor
- 2 个 Scheduler **完全独立**，互不感知对方的请求；负载均衡在请求入口由前端/Coordinator 完成

**Executor（MultiprocExecutor）—— Scheduler 与 Worker 之间的"传令官"**
- 启动并管理本 DP rank 的 4 个 Worker 子进程
- 把调度结果通过共享内存消息队列**广播 RPC**（`collective_rpc`）给所有 Worker
- 只从 output_rank（TP rank0 且 PP 最后一级，即 W2/W6，`multiproc_executor.py:509`）收集最终输出
- 初始化阶段汇总各 Worker 的 KV cache spec / 显存 profiling 结果

**Worker —— 每张 GPU 上的"执行单元"**
- 初始化设备和通信组，按 TP/PP rank 加载权重分片，按 EP rank 只加载 1/4 的专家
- 分配本卡 KV cache 显存
- `execute_model` 执行前向：参与 TP all-reduce、PP p2p、EP all-to-all
- 驱动 ModelRunner 完成采样

**一句话总结**：DP 决定 EngineCore / Scheduler / Executor 的数量（各 2 个），TP×PP 决定每个 Executor 下 Worker 的数量（各 4 个，共 8 个），EP 不产生新对象，只在 Worker 上叠加一个跨 DP 边界的 MoE 通信组。

---

## 第二部分：Worker 之间的通信组详解（TP / PP / DP / EP 组）

### 2.1 根源：所有分组都来自同一个 5 维张量

`vllm/distributed/parallel_state.py:1793`：

```python
# the layout order is: ExternalDP x DP x PP x PCP x TP
all_ranks = torch.arange(world_size).reshape(
    -1,                # ExternalDP (本例=1)
    data_parallel_size,                    # 2
    pipeline_model_parallel_size,          # 2
    prefill_context_model_parallel_size,   # 1 (PCP 未开启)
    tensor_model_parallel_size,            # 2
)
```

关键点：**8 个 Worker 的全局 rank 不是随便编的，而是按 `DP × PP × TP` 多维坐标展开的一维序号，TP 是最内层（变化最快）**。对号入座：

```
全局 rank = dp_rank × (PP×TP) + pp_rank × TP + tp_rank
          = dp_rank × 4      + pp_rank × 2 + tp_rank
```

| 全局 rank | Worker | dp | pp | tp | 所在引擎 |
|---|---|---|---|---|---|
| 0 | W0 | 0 | 0 | 0 | EngineCore_DP0 |
| 1 | W1 | 0 | 0 | **1** | EngineCore_DP0 |
| 2 | W2 | 0 | **1** | 0 | EngineCore_DP0 |
| 3 | W3 | 0 | **1** | **1** | EngineCore_DP0 |
| 4 | W4 | **1** | 0 | 0 | EngineCore_DP1 |
| 5 | W5 | **1** | 0 | **1** | EngineCore_DP1 |
| 6 | W6 | **1** | **1** | 0 | EngineCore_DP1 |
| 7 | W7 | **1** | **1** | **1** | EngineCore_DP1 |

分组的生成手法统一是：**把该维度 transpose 到最后一维 → reshape 成 2D → 每行就是一个组**（代码注释 1791-1792 行）。

### 2.2 TP 组：{W0,W1} {W2,W3} {W4,W5} {W6,W7}

生成代码（1804 行）：直接 `view(-1, tp_size)` —— 取**连续的两个 rank**，即固定 (dp, pp)、只让 tp 变化的组合。

**语义：同一层网络的权重被横向切开，两个 GPU 各算一半，再合并。**

以 PP stage0 的某一层为例，W0 和 W1 上跑的是**同一层、同一批 token**：

```
输入 hidden states (同一份, 两个rank都有)
        │
   ┌────┴────┐
   ▼         ▼
 W0: QKV 权重的前半    W1: QKV 权重的后半
 (attention 头 0~h/2)  (attention 头 h/2~h)
 (MLP 中间维前半)      (MLP 中间维后半)
   │         │
   └────┬────┘
        ▼
   ALL-REDUCE (两组部分结果相加)   ← 这就是 TP 组的通信
        │
   完整的层输出
```

特点：
- **通信极其频繁**：每个 decoder layer 至少 2 次 all-reduce，延迟敏感，所以 TP 组要放在**互联最快的卡之间**（这就是把 TP 排在 rank 最内层的原因——连续 rank 通常对应同机 NVLink 相邻的卡）
- 4 个 TP 组之间**完全独立**，各算各的层/各批数据

### 2.3 PP 组：{W0,W2} {W1,W3} {W4,W6} {W5,W7}

生成代码（1859 行）：`transpose(2, 4)` 把 PP 维换到最后 —— 即固定 (dp, tp)、只让 pp 变化的组合。

**语义：模型的 L 层被纵向切成 2 段，每个 rank 只持有其中一段。**

```
EngineCore_DP0 内:
W0 (pp0,tp0): 第 0 ~ L/2 层的 tp0 分片 ──┐
                                        │ p2p send/recv
W2 (pp1,tp0): 第 L/2 ~ L 层的 tp0 分片 ◄─┘   (只传 hidden states 激活值)
```

一次前向：token 在 W0 上算完前一半层 → 把激活值**点对点**发给 W2 → W2 算完后一半层并采样出 token。

特点：
- 组内两个 rank 持有**完全不同的层**（不像 TP 是同层切分）
- 通信是 **p2p**（send/recv），不是集合通信，且只发生在 stage 边界，一次前向只传 1 次，对带宽要求低
- 为什么配对的 PP 组是 4 条"流水线"？因为每条流水线需要完整的 tp0 列和 tp1 列：{W0,W2} 负责 tp0 列，{W1,W3} 负责 tp1 列
- 最终输出只由 PP 最后一级的 TP rank0（W2、W6）返回，这就是前面说的 output_rank

### 2.4 DP 组：{W0,W4} {W1,W5} {W2,W6} {W3,W7}

生成代码（1875 行）：`transpose(1, 4)` 把 DP 维换到最后 —— 固定 (pp, tp)、只让 dp 变化，即**两个引擎中坐标位置完全相同**的 rank 配对。

**语义：这是跨 EngineCore 进程边界的组，作用是"锁步"。**

代码 1784-1790 行的注释：

> *all the ranks in the same DP group should generate simultaneously, i.e. the `generate` call in the same DP group should be called together, otherwise it will cause deadlock.*

对 dense 模型，两个 DP 引擎各跑各的请求，DP 组几乎不通信（甚至 `run_engine_core` 里非 MoE 时直接把 dp_size 改成 1）。但 **MoE + EP 时它变成关键组**——原因见下面的 EP。

### 2.5 EP 组：{W0,W1,W4,W5} 和 {W2,W3,W6,W7}

生成代码（1894-1903 行）：

```python
group_ranks = all_ranks.transpose(1, 2).reshape(
    -1, data_parallel_size * prefill_context_model_parallel_size
        * tensor_model_parallel_size   # = 2×1×2 = 4
).unbind(0)
```

`transpose(1, 2)` 把 DP 维和 PP 维交换，维度顺序变成 `(extDP, PP, DP, PCP, TP)`，再按最后三维（DP×PCP×TP=4）切组。效果就是：**固定 pp 坐标，把该 stage 内所有 (dp, tp) 组合的 rank 收进一个组**：

```
PP stage0: dp0的{tp0,tp1} + dp1的{tp0,tp1} = {W0, W1, W4, W5}
PP stage1: dp0的{tp0,tp1} + dp1的{tp0,tp1} = {W2, W3, W6, W7}
```

**语义：MoE 层的专家不按 TP 切，而是按"专家编号"分散到这 4 个 rank 上。**

假设模型有 8 个专家，PP stage0 的某个 MoE 层：

```
W0: 专家 0,1     W1: 专家 2,3     W4: 专家 4,5     W5: 专家 6,7
```

前向时，每个 rank 上的 token 经过 router 算出该去哪些专家，然后：

```
Step 1: all-to-all DISPATCH
  W0 上的 token A 被路由到专家5 → 发给 W4
  W4 上的 token B 被路由到专家1 → 发给 W0
  ... (4个rank两两互相交换token)
Step 2: 各 rank 用自己持有的专家计算收到的 token
Step 3: all-to-all COMBINE，把结果送回原 rank
Step 4: 各 rank 继续自己后续的层
```

三个设计要点：

1. **为什么 EP = DP × TP？** 开启 EP 后，MoE 层不再做 TP 切分（每个专家完整地放在某一个 rank 上），而是用"数据并行的 batch + TP 并行的 batch"合在一起喂给 4 份专家分片。相当于把 TP 组的 2 路并行和 DP 的 2 路并行"借用"过来组成 4 路专家并行，组 size = 2×2 = 4。

2. **为什么 EP 组要跨 EngineCore 边界（W0 和 W4 在不同进程）？** 这是 vLLM 的刻意设计：专家总数往往很多，EP 范围越大每个 rank 持有的专家越少、越省显存。代价就是 all-to-all 的参与者散布在两个引擎里。

3. **这正解释了 DP 组和 DPCoordinator 存在的必要性**：all-to-all 是集合通信，4 个参与者必须**同一步**都进入这个 MoE 层，缺一个就死锁。但两个 EngineCore 各有各的 Scheduler、各收各的请求，节奏天然不同步——所以由 DPCoordinator 做 wave 同步：每个引擎每步前上报"我这步有没有活"，Coordinator 统一发令"第 N 波开始"，没活的引擎也要跑 dummy batch 陪跑，保证跨引擎的 EP all-to-all 永远齐整。

4. **为什么 EP 不跨 PP？** 因为不同 PP stage 持有不同的层——stage0 的 MoE 层在 W2/W3 上根本不存在，把它们拉进组没有意义。

### 2.6 总结图（以 W0 的视角）

```
W0 同时是 4 个组的成员：

  TP 组 {W0,W1}        → 同一层，切权重，每层 all-reduce（最频繁，要 NVLink）
  PP 组 {W0,W2}        → 不同层，传激活，p2p（每 token 每方向 1 次）
  DP 组 {W0,W4}        → 同坐标跨引擎，锁步对齐（为 EP 服务）
  EP 组 {W0,W1,W4,W5}  → 同一 MoE 层，切专家，all-to-all（每 MoE 层 2 次）

一次前向经过 stage0 的 MoE 层时，W0 实际发生的事：
  ① 和 W1 做 TP all-reduce（attention 部分）
  ② 和 W1,W4,W5 做 EP all-to-all dispatch → 算专家0,1 → combine（MoE 部分）
  ③ 算完本 stage 所有层后，把激活 p2p 发给 W2（PP）
  ④ 全程与 W4 保持 wave 锁步（DP），否则 ② 会挂死
```

**简记口诀**：TP 切权重（同层）、PP 切层（不同层）、DP 切请求（不同 batch）、EP 切专家（同 MoE 层、跨 DP）；四种切法正交叠加，就是 rank 张量 `(DP, PP, TP)` 的四个维度。

---

## 第三部分：问答

### Q1：提及的 PCP（prefill_context_model_parallel_size）是什么作用？

**一句话：PCP 是把"序列长度"这个维度切开做并行——prefill 阶段把一个长 prompt 的 token 序列切成几段，分给多个 rank 同时算。**

配置定义（`vllm/config/parallel.py:124`）：

```python
prefill_context_parallel_size: int = Field(default=1, ge=1)
"""Number of ranks that split prefill sequence computation. PCP expands
the process world size but does not increase the KV-cache shard count."""
```

#### 为什么需要它

前面的四种并行，没有一个能加速**单条长请求的 prefill**：

- TP 切的是权重维度，一个 128K 的 prompt 来了，TP 组内每个 rank 都要对**全部 128K token** 做 attention，只是各算一半的头——序列长度带来的计算量一点没少
- DP 切的是"不同请求"，管不了单条请求
- PP 切层，token 还是得逐层流过整个序列

于是对超长上下文（128K、1M token），TTFT（首 token 延迟）会很长。PCP 就是为此加的第五种切法：

```
一个 128K token 的 prompt，PCP=2：

PCP rank0: 负责 token 0 ~ 64K 的 Q 计算
PCP rank1: 负责 token 64K ~ 128K 的 Q 计算
           │
           └─ 但 attention 要求每个 Q 看到完整的 K/V，
              所以两个 rank 之间要通过 PCP 通信组交换/聚合 KV
              （ring attention 或 KV all-gather，
               代码里可见 get_pcp_group().all_gather(...)，
               如 v1/attention/backends/mla/indexer.py:815）
```

效果：prefill 计算量近似减半，TTFT 下降；每个 rank 的激活显存也减半。

#### 在 rank 布局中的位置

5 维 rank 张量 `(ExternalDP, DP, PP, PCP, TP)` 里，PCP 位于 PP 和 TP 之间。如果例子中再加 `--prefill-context-parallel-size 2`：

- 每个 EngineCore 的 world_size = TP×PCP×PP = 2×2×2 = **8 个 Worker**（`multiproc_executor.py:118` 的断言 `world_size == tp*pp*pcp`）
- EP 组大小也相应变成 DP×PCP×TP = 8

#### 几个重要约束（都来自 parallel.py 的校验逻辑）

- **PCP 暂不支持与 DP 同开**（`parallel.py:508`：`"PCP does not support data parallelism yet"`）
- **"不增加 KV cache 分片数"**：KV cache 的头仍然只按 TP 切分，PCP rank 只是各自存自己那段序列的 KV，不会多复制一份
- 它有个 decode 阶段的"兄弟" **DCP**（decode context parallel，`parallel.py:340`）：decode 时每步只有 1 个新 token，没法切 token，DCP 改为**切历史 KV cache**——每个 rank 只对 KV 的一段做 attention，再合并部分结果（类似 flash-decoding 的 split-K 思想）。且 **PCP 不开时 DCP 直接复用 TP 的 rank**（`parallel.py:511`：`"DCP reuses the TP ranks when PCP is disabled"`），不额外占进程

### Q2：讲解 TP 时提到的"MLP 中间维前半"是什么意思？

#### 先看一个 MLP 层的完整结构

Transformer 每个 decoder layer 里除了 attention 还有一个 MLP（以 Llama 系的 SwiGLU 为例），涉及两个维度：

- **hidden_size H**（如 4096）：token 向量的"宽度"，层与层之间传递的维度
- **intermediate_size I**（如 14336）：MLP 内部先把向量**放大**到的中间维度，算完再缩回 H

计算分三步：

```
x (shape: [tokens, H])
  │
  ├─ gate_proj: H → I   ┐
  ├─ up_proj:   H → I   ┘ 通常合并成一个 gate_up_proj: H → 2I
  ▼
h = SiLU(gate(x)) * up(x)        ← shape: [tokens, I]，这就是"中间维"
  │
  └─ down_proj: I → H
  ▼
y (shape: [tokens, H])
```

#### TP=2 时怎么切

切的就是这个 **I 维度**，每个 rank 拿一半（`intermediate_size_per_partition = intermediate_size // tp_size`，见 `layers/activation.py:764`）：

```
                 x (两个 rank 都有完整副本)
                ┌┴┐
                ▼ ▼
   W0: gate_up 权重的【前半】     W1: gate_up 权重的【后半】
       输出中间维的 0 ~ I/2            输出中间维的 I/2 ~ I
                │                        │
   h0 = SiLU(gate₀(x)) * up₀(x)   h1 = SiLU(gate₁(x)) * up₁(x)
       shape [tokens, I/2]             shape [tokens, I/2]
                │                        │
   W0: down 权重的【前 I/2 列】    W1: down 权重的【后 I/2 列】
   y0 = down₀(h0)                  y1 = down₁(h1)
                └────────┬────────┘
                         ▼
                   ALL-REDUCE: y = y0 + y1
```

#### 为什么这样切是数学上严格正确的

关键在于 down_proj 是一个 `H × I` 的矩阵，它的**每一列对应一个中间维神经元**。W0 拿前 I/2 列、W1 拿后 I/2 列，则：

```
y = down · h = down₀·h₀ + down₁·h₁ = y0 + y1
```

矩阵乘法按输入维度分块后就是部分和相加，所以最后 all-reduce 一次即得精确结果。

#### 这个设计的精妙之处（Megatron 式切分）

- gate_up_proj 按**输出维**切（列并行），down_proj 按**输入维**切（行并行），两者在"中间维 I"上正好衔接
- 中间结果 `h`（shape I/2）**完全不需要通信**——每个 rank 本地做激活、本地乘 down 分片
- 整个 MLP 只有 down_proj 之后**一次 all-reduce**。attention 部分同理（QKV 切头 + O_proj 行并行）也是一次 all-reduce，所以"每个 decoder layer 至少 2 次 all-reduce"就是这么来的：attention 1 次 + MLP 1 次

对比记忆：**"中间维前半" = MLP 内部放大后的 I 维度，TP 把它对半劈开，两个 rank 各持有一半中间神经元的计算。**
