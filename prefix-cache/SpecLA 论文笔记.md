# SpecLA：面向线性注意力模型的高效投机解码

> 论文：*SpecLA: Efficient Speculative Decoding for Linear-Attention Models*  
> 作者：Zhibin Wang 等  
> 版本：arXiv:2607.16673v1，2026-07-18  
> 原始 PDF：**D:\files\15.kv cache\SpecLA Efficient Speculative Decoding for Linear-Attention Models.pdf**

## 目录

- [1. 核心结论](#1-核心结论)
- [2. 背景：线性注意力为何仍需要投机解码](#2-背景线性注意力为何仍需要投机解码)
- [3. 为什么 Transformer 方案不能直接迁移](#3-为什么-transformer-方案不能直接迁移)
- [4. SpecLA 整体架构](#4-specla-整体架构)
- [5. 拓扑感知验证](#5-拓扑感知验证)
- [6. 投机状态管理](#6-投机状态管理)
- [7. Draft Tree 构造与剪枝](#7-draft-tree-构造与剪枝)
- [8. 一轮推理的完整示例](#8-一轮推理的完整示例)
- [9. 实现与实验](#9-实现与实验)
- [10. 贡献、局限与结论](#10-贡献局限与结论)

---

## 1. 核心结论

SpecLA 要解决的问题可以概括为：

> 如何把依赖 Transformer KV Cache 后缀语义的投机解码，改造成适用于 Gated DeltaNet 等有状态线性注意力模型的“状态轨迹验证、选择与提交”。

Transformer 与线性注意力模型对历史的表达方式不同：

| 模型 | 历史表示 | 新 token 到来时 | 拒绝 draft token 时 |
|---|---|---|---|
| Transformer | 按 token 保存的 KV Cache | 追加 K/V 槽位 | 截断 speculative KV 后缀 |
| Linear Attention / GDN | 固定大小的递归状态 | 原地推进同一个稠密状态 | 不能通过删除槽位回滚 |

SpecLA 设计了三个相互配合的机制：

1. **Topology-aware verification**：根据候选 chain/tree 的拓扑关系调度验证 kernel；
2. **Accepted-factor state management**：保存验证阶段产生的紧凑 factors，而不是完整状态快照；
3. **Target-aligned drafting**：使用目标模型的递归特征训练 EAGLE 风格 drafter，并在验证前剪掉低价值候选。

论文在单张 NVIDIA H100 和公开 GDN-1.3B target 上报告：

| 数据集 | SpecLA-Tree 相对自回归解码的加速 |
|---|---:|
| Mixed suite | 1.42x |
| GSM8K | 1.70x |
| HumanEval | 1.06x |

这说明 SpecLA 的收益不是恒定的。只有当 drafter 能提供足够长的 accepted prefix 时，一次目标验证所摊薄的状态访存和 kernel launch 成本才会超过 draft、验证与提交的额外开销。

---

## 2. 背景：线性注意力为何仍需要投机解码

### 2.1 KV Cache 与递归状态

Transformer 在 decode 时为每个历史 token 保存 Key 和 Value，因此 KV Cache 随上下文长度 $T$ 增长：

$$
\mathrm{KVBytes} \propto T
$$

线性注意力不再保存每个 token 的独立 K/V，而是把前缀压缩到固定大小的递归状态。论文第 3 页用下面的简化更新式说明这种状态语义：

$$
S_t = S_{t-1} + v_t k_t^{\top}
\tag{1}
$$

其中：

- $S_{t-1}$：处理当前 token 前的历史状态；
- $k_t$：当前 token 的 key factor；
- $v_t$：当前 token 的 value factor；
- $v_t k_t^{\top}$：当前 token 形成的秩一状态增量；
- $S_t$：加入当前 token 后的新状态。

公式 (1) 只是用于解释状态语义的最简形式。真实 GDN 还包含 gate、decay、Delta rule、卷积状态等结构，不能把完整 GDN 实现等同于公式 (1)。

### 2.2 固定大小状态仍然是 memory-bound

假设一层 GDN 的主递归状态为：

$$
S \in \mathbb{R}^{h_v \times d_k \times d_v}
$$

若每个元素占 $w$ 字节，则状态大小为：

$$
\mathrm{StateBytes} = h_v d_k d_v w
$$

论文实现采用：

$$
h_v = 8, \qquad d_k = d_v = 256, \qquad w = 4
$$

因此：

$$
\begin{aligned}
\mathrm{StateBytes}
&= 8 \times 256 \times 256 \times 4 \\
&= 2{,}097{,}152\ \mathrm{bytes} \\
&= 2\ \mathrm{MiB}.
\end{aligned}
$$

一个普通 decode step 至少需要读取一次旧状态并写回一次新状态，所以每 token、每层约有：

$$
2\ \mathrm{MiB\ read}
+
2\ \mathrm{MiB\ write}
=
4\ \mathrm{MiB}.
$$

线性注意力消除了随上下文增长的 KV Cache，却没有消除每个 token 对固定稠密状态的读写。论文引用的配置在 batch size 为 1 时算术强度低于 1 FLOP/B，因此 decode 仍然是 memory-bound。

### 2.3 投机解码能够摊薄什么

普通自回归解码是：

~~~text
Target forward -> 1 token
Target forward -> 1 token
Target forward -> 1 token
~~~

投机解码先由较快的 drafter 提出多个未来 token，再让 target 一次验证这些候选。如果一次验证最终接受多个 token，就可以共同分摊：

- recurrent state 的读取和写回；
- kernel launch；
- 同步与固定调度开销；
- target model 的部分固定成本。

对线性注意力而言，理想执行形态是：

~~~text
读取一次状态 tile
        |
连续验证多个已知候选 token
        |
写回一次状态 tile
~~~

---

## 3. 为什么 Transformer 方案不能直接迁移

### 3.1 Transformer 的后缀截断

Transformer 验证候选后，临时 KV Cache 可以表示为：

~~~text
Committed KV | draft 1 | draft 2 | draft 3 | draft 4
~~~

如果只接受前两个候选，只需截断为：

~~~text
Committed KV | draft 1 | draft 2
~~~

被拒绝 token 的 K/V 位于独立槽位中，删除 speculative suffix 不需要重建已接受部分。

### 3.2 递归状态没有可截断后缀

对于候选链：

~~~text
x1 -> x2 -> x3 -> x4
~~~

状态按下面的顺序演化：

$$
\begin{aligned}
S_1 &= F(S_0,x_1), \\
S_2 &= F(S_1,x_2), \\
S_3 &= F(S_2,x_3), \\
S_4 &= F(S_3,x_4).
\end{aligned}
$$

如果只接受 $x_1,x_2$，持久状态必须停在 $S_2$。如果系统已经把 $x_3,x_4$ 的影响写入同一个状态对象，就不能通过删除两个 token 槽位恢复 $S_2$。

### 3.3 逐 token 重放 decode kernel

第一种朴素方案是对每个候选调用一次单 token decode kernel：

~~~text
load S0 -> verify x1 -> store S1
load S1 -> verify x2 -> store S2
load S2 -> verify x3 -> store S3
~~~

数值语义正确，但每个候选仍然进行一次完整状态的 HBM round trip。投机解码希望摊薄的主要开销被完整保留下来。

对于树形候选，不同分支还需要不同的中间状态。系统只能保存完整状态快照、重放共享前缀，或把分支串行化，三种做法都会增加开销。

### 3.4 把 draft 当成短 prefill

长序列 GDN prefill 通常包含：

- layout conversion；
- chunk/block summary；
- scan 或 UT transform；
- lower-triangular solve；
- 中间张量读写和同步。

这些固定成本适合由几百或几千个 token 摊薄，而 speculative window 通常只有数个到十几个 token。

树形候选还不满足普通 causal sequence 的依赖。普通 causal mask 允许一个位置读取此前所有位置；draft tree 中，一个节点只能读取自己的祖先，不能读取兄弟分支。

因此 SpecLA 面临三个问题：

1. 怎样在短 chain/tree 上复用片上状态；
2. 怎样只提交 accepted state trajectory；
3. 怎样避免 target 验证大量低价值候选。

---

## 4. SpecLA 整体架构

论文第 5 页图 2 的整体流程可以整理为：

~~~text
Target-aligned EAGLE-style drafter
        |
        | candidate tokens + scores
        v
Confidence-guided pruning
        |
        | parent-pointer topology
        v
Topology-aware target verification
  - state-resident serial
  - tree-masked parallel
  - chain-decomposed hybrid
        |
        | target logits + factor records
        v
Posterior acceptance
        |
        | accepted-path factors only
        v
Pending accepted-factor buffer
        |
        v
Next round: fused update-and-verify
~~~

外层仍然是标准的：

~~~text
Draft -> Verify -> Accept -> Next iteration
~~~

内部变化包括：

- draft 不再盲目扩展候选，而是考虑 target verification 成本；
- verifier 不把候选视为普通序列，而是读取其拓扑；
- acceptance 不再截断 KV Cache，而是收集 accepted path 的 factors；
- commit 不立即单独执行，而是延迟到下一轮验证时融合完成。

---

## 5. 拓扑感知验证

### 5.1 State-resident serial verification

位置：论文第 5 至 6 页，Section 4.1，图 3 和图 4。

#### 5.1.1 从 token-major 改为 layer-major

普通自回归执行顺序是 token-major：

~~~text
token 0: layer 0 -> layer 1 -> layer 2 -> layer 3
token 1: layer 0 -> layer 1 -> layer 2 -> layer 3
token 2: layer 0 -> layer 1 -> layer 2 -> layer 3
~~~

相邻 token 在同一层的状态更新之间隔着整个模型其余层的执行，该层状态通常需要先写回 HBM。

在 speculative verification 中，draft tokens 已经全部已知，所以可以改为 layer-major：

~~~text
layer 0: token 0 -> token 1 -> token 2
layer 1: token 0 -> token 1 -> token 2
layer 2: token 0 -> token 1 -> token 2
layer 3: token 0 -> token 1 -> token 2
~~~

这样，同一层的状态 tile 被加载后，可以连续处理多个候选，再统一写回。

#### 5.1.2 为什么沿 V 维切片

一层完整 GDN 状态约为 2 MiB，而论文指出 H100 每个 SM 的片上 SRAM 约为 228 KiB，因此完整状态无法常驻单个 SM。

简化输出为：

$$
o_t = q_t^{\top} S_t.
$$

如果沿 $K$ 维切片，$q_t^{\top}S_t$ 的同一输出片段会分散到多个 block，需要跨 block reduction。

SpecLA 沿 $V$ 维切片：

$$
S[:,0:B_V],\quad
S[:,B_V:2B_V],\quad
\ldots
$$

每个 block：

1. 保留完整 $K$ 维；
2. 更新一个 $K \times B_V$ 状态 tile；
3. 独立生成对应的输出切片；
4. 在该 tile 上连续执行多个候选 token。

这样避免了跨 block reduction，同时保留链内递归顺序。

### 5.2 Tree-masked parallel verification

位置：论文第 6 至 7 页，Section 4.2，图 5。

对于包含 $L$ 个候选节点的 draft tree，定义：

$$
M_{\mathrm{tree}} \in \{0,1\}^{L \times L}.
$$

若节点 $j$ 是节点 $i$ 的合法祖先依赖，则：

$$
M_{\mathrm{tree}}[i,j] = 1.
$$

例如：

~~~text
        1
       / \
      2   3
     /     \
    4       5
~~~

节点 4 可以接收节点 1、2 的更新，但不能接收节点 3、5 的更新。树 mask 用来阻止 sibling leakage。

对于简化线性注意力，树并行计算可表示为：

$$
O
=
Q S_0^{\top}
+
\left(
\left(QK^{\top}\right)
\odot
M_{\mathrm{tree}}
\right)V.
\tag{2}
$$

其中：

- $S_0$ 是本轮验证开始前的 committed state；
- $Q S_0^{\top}$ 是所有候选读取共享历史得到的贡献；
- $QK^{\top}$ 表示候选窗口内 token 之间的相互作用；
- 与 $M_{\mathrm{tree}}$ 做逐元素乘法后，只保留祖先依赖；
- 最后乘 $V$ 聚合合法候选更新。

#### 5.2.1 GDN 的树形 UT factorization

GDN 的 corrected update 会递归依赖此前的 corrected updates，因此不能只在最终 attention 矩阵上乘 mask。SpecLA 把树 mask 放进 UT factorization：

$$
A_{\mathrm{tree}}
=
M_{\mathrm{tree}}
\odot
\left(K_b K_a^{\top}\right),
\tag{3}
$$

$$
T_{\mathrm{tree}}
=
\operatorname{solve\_tril}
\left(I-A_{\mathrm{tree}}\right).
\tag{4}
$$

然后从 $T_{\mathrm{tree}}$ 派生 $W$、$U$ 和 $\mathrm{Attn}$，计算：

$$
O_{\mathrm{GDN}}
=
\widehat{Q} S_0^{\top}
-
\mathrm{Attn}
\left(W S_0^{\top}\right)
+
\mathrm{Attn}U.
\tag{5}
$$

三项可以理解为：

1. $\widehat{Q}S_0^{\top}$：从 committed state 读取共享历史贡献；
2. $-\mathrm{Attn}(WS_0^{\top})$：按照 Delta rule 修正旧状态贡献；
3. $+\mathrm{Attn}U$：聚合当前候选窗口内的 corrected updates。

由于 $T_{\mathrm{tree}}$ 由树 mask 构造，信息只沿祖先路径传播。

这条路径能够并行验证所有树节点，但需要构造和搬运 $A$、$T$、$W$、$U$、$\mathrm{Attn}$ 等中间张量。对于短 speculative window，这些开销常常无法摊薄。

### 5.3 Chain-decomposed hybrid verification

位置：论文第 7 页，Section 4.3，图 6。

SpecLA 使用 heavy-light decomposition 把树拆成若干互不重叠的链。论文图 6 的例子是：

~~~text
Chain A: [1, 2, 4, 7]
Chain B: [3, 6, 9]
Chain C: [5, 8]
~~~

压缩后的依赖为：

~~~text
             Chain A
             /     \
        Chain B   Chain C
~~~

执行规则：

- 链内使用 state-resident serial kernel；
- 父链到达分支边界时，物化供子链使用的 boundary state；
- 多条 ready 子链可以并行执行；
- 跨模型层仍保持正常的层间依赖。

因此：

- 非边界节点的中间状态无需写回 HBM；
- 共享前缀只计算一次；
- 不需要完整 tree-masked UT 路径的全部中间张量；
- 分支之间仍然保留一定并行度。

三种路径的取舍如下：

| 验证路径 | 优点 | 缺点 |
|---|---|---|
| State-resident serial | 设置开销低，状态驻留充分 | token/branch 并行度有限 |
| Tree-masked parallel | 所有树节点可并行 | UT、solve 和中间张量成本高 |
| Chain-decomposed hybrid | 链内驻留、链间并行 | 需要边界状态与链调度 |

论文的重要结论不是“并行越多越好”，而是：短 speculative tree 更需要控制 setup、reduction 和中间张量开销。

---

## 6. 投机状态管理

### 6.1 三种 accepted-state recovery 方案

位置：论文第 7 至 9 页，Section 5.1，图 7。

#### 6.1.1 Full-state snapshotting

为每个候选保存完整 endpoint state。若单层状态是 2 MiB，验证 $L$ 个候选需要：

$$
\mathrm{SnapshotBytes}
=
2L\ \mathrm{MiB\ per\ layer}.
$$

当 $L=8$ 时，每层需要 16 MiB，远超单个 SM 的片上容量，只能放入 HBM。

优点是 posterior selection 后可以直接取得 accepted endpoint；缺点是每个候选都物化完整稠密状态，内存占用和状态流量过大。

#### 6.1.2 Token replay

只保存候选 token，选择 accepted path 后重新执行 target：

$$
S_A
=
S_0
+
\sum_{t=1}^{A}\Delta S_t,
\tag{6}
$$

$$
\Delta S_t
=
\operatorname{DeltaConstruct}
\left(
\operatorname{FactorGenerate}(x_t)
\right).
\tag{7}
$$

这种方法存储开销小，但 $\operatorname{FactorGenerate}$ 已经在验证阶段计算过。接受后重新运行 target 会重复执行 projection、feed-forward 和相关前向计算。

#### 6.1.3 Factor buffering

SpecLA 保存验证过程中已经生成的紧凑 factor record：

$$
f_t = \operatorname{FactorGenerate}(x_t).
$$

GDN 的记录主要包括：

$$
f_t
=
\left(k_t,v_t,\alpha_t,\beta_t\right).
$$

posterior selection 后只收集 accepted path 上的 factors：

$$
S_A
=
S_0
+
\sum_{t=1}^{A}
\operatorname{DeltaConstruct}(f_t).
\tag{8}
$$

论文图 7 给出的每层存储规模为：

| 方案 | 每层存储量 | 提交阶段主要代价 |
|---|---:|---|
| Full-state snapshot | 约 $2L$ MiB | 拷贝 accepted endpoint |
| Factor buffer | 约 $16L$ KiB | 执行较轻的 DeltaConstruct |
| Token replay | 约 $4L$ B | 重跑 FactorGenerate 和 DeltaConstruct |

因此 factor buffer 相比 full-state snapshot 约缩小：

$$
\frac{2\ \mathrm{MiB}}
{16\ \mathrm{KiB}}
=
128.
$$

Factor buffering 的本质是在“保存完整结果”和“只保存输入并重新计算”之间选取一个中间点：保存足以恢复 accepted update、且验证阶段已经产生的最小记录。

不同状态层需要保存不同 factor schema：

- GLA：key、value、gate/decay，不需要 Delta-rule $\beta$；
- DeltaNet：key、value、$\beta$，不需要 GDN decay $\alpha$；
- Mamba 类 SSM：保存相应的 selective state-space 参数。

所以 factor buffering 是通用思想，但不是无需适配的统一张量格式。

### 6.2 Delayed state update

位置：论文第 9 页，Section 5.2，图 7(b)。

如果 posterior selection 后立刻更新 durable state，会产生：

~~~text
Iteration i:
  load state -> apply accepted factors -> store state

Iteration i+1:
  load state again -> verify new draft -> store state
~~~

中间多了一次完整状态读写。

SpecLA 在第 $i$ 轮结束时只保存 accepted-factor buffer $B_i$，不立即推进 durable state。第 $i+1$ 轮验证开始时：

$$
\widetilde{S}_i
=
\operatorname{ApplyFactors}
\left(S_{i-1},B_i\right).
\tag{9}
$$

随后直接从 $\widetilde{S}_i$ 验证新 draft。状态 tile 在应用 $B_i$ 后继续留在 SRAM，无需先写回再重新读取：

~~~text
load durable state
        |
apply pending accepted factors
        |
keep updated tile in SRAM
        |
verify current draft
        |
store state
~~~

该机制减少的是 HBM round trip，而不是省略必要的状态数学更新。

---

## 7. Draft Tree 构造与剪枝

### 7.1 Confidence-guided pruning

位置：论文第 9 至 10 页，Section 6.1，图 8。

对候选节点 $v$，定义从根到该节点的累计 log probability：

$$
q(v)
=
\sum_{i=1}^{\operatorname{depth}(v)}
\log
p_{\mathrm{draft}}
\left(
t_i
\mid
\mathrm{prefix},t_{<i}
\right).
\tag{10}
$$

令：

$$
q^{\star} = \max_v q(v).
$$

给定 margin $\tau_{\mathrm{tree}}$，只保留：

$$
q(v)
\ge
q^{\star}
-
\tau_{\mathrm{tree}}.
\tag{11}
$$

因为每一步的 log probability 通常不大于 0，$q(v)$ 沿根到叶路径单调不增。所以：

- 如果子节点通过阈值，其祖先也会通过；
- 一个阈值能够同时限制树的宽度和深度；
- 深而低概率的路径会自然离开分数窗口。

系统还设置 token budget $T$。如果通过阈值的节点超过预算，就按分数选择候选并补齐其祖先，保证最终 topology 仍然 ancestor-closed。

剪枝只改变提交给 target 的 proposal set，不直接决定最终输出。最终 acceptance 仍然由 target probabilities 完成。

### 7.2 Target-aligned EAGLE-style drafter

位置：论文第 10 页，Section 6.2。

直接使用通用 Transformer drafter 时，其 hidden-state dynamics 可能与 GDN target 的 recurrent dynamics 不一致，进而导致：

- proposal 不准确；
- candidate score 不能可靠预测 target acceptance；
- confidence pruning 删除正确路径或保留低价值路径。

SpecLA 使用目标模型递归执行过程产生的 feature 训练 EAGLE-style draft layer，使 proposal score 与 target dynamics 对齐。

drafter 本身仍使用 Transformer-style attention。论文认为 draft window 很短：以 $L=8$、$h=16$、$d=64$ 为例，KV Cache 约为 32 KiB，$O(L^2)$ attention 成本较小。若把 drafter 也改成 recurrent model，还需要额外管理 drafter 的 speculative state，却难以在短窗口上发挥线性注意力的渐近优势。

---

## 8. 一轮推理的完整示例

假设当前 durable state 是 $S_0$，drafter 生成：

~~~text
        a
       / \
      b   c
     /     \
    d       e
~~~

### 步骤 1：候选打分和剪枝

假设累计 log probability 为：

~~~text
a       -0.1
a->b    -0.4
a->b->d -0.8
a->c    -1.5
a->c->e -3.2
~~~

若 margin 只覆盖到 -2.0，则节点 $e$ 被剪掉。

### 步骤 2：打包拓扑

可用 parent array 表示：

~~~text
node:    a  b  c  d
parent: -1  0  0  1
~~~

### 步骤 3：链分解

一种可能的分解是：

~~~text
Chain A: a -> b -> d
Chain B: c
~~~

Chain A 使用 state-resident serial kernel。到达节点 $a$ 后，因为 Chain B 需要以 $S_a$ 为边界状态，所以 $S_a$ 必须物化。随后：

- Chain A 继续验证 $b,d$；
- Chain B 从 $S_a$ 开始验证 $c$；
- 两条 ready chain 可以并行执行。

### 步骤 4：记录 factors

验证产生：

$$
f_a,\qquad f_b,\qquad f_c,\qquad f_d.
$$

但系统不把所有候选 endpoint state 都写入 durable state。

### 步骤 5：Posterior acceptance

假设 target 接受路径 $a \rightarrow b$，并在 $d$ 处拒绝，则只保留：

$$
B_{\mathrm{pending}}
=
[f_a,f_b].
$$

$f_c$ 和 $f_d$ 被释放。

### 步骤 6：下一轮融合更新

下一轮 verifier 加载 $S_0$，应用 pending factors：

$$
S_2
=
\operatorname{ApplyFactors}
\left(
S_0,[f_a,f_b]
\right).
$$

然后不写回中间的 $S_2$，而是直接从 $S_2$ 开始验证下一批候选。

---

## 9. 实现与实验

### 9.1 实现细节

位置：论文第 10 页，Section 7。

论文使用 PyTorch 和 Triton 实现，并集成到 EAGLE-style generation runtime。目标 GDN 层配置为：

$$
h_v = 8,\qquad d_k=d_v=256.
$$

同时包含：

- kernel size 为 4 的 causal depthwise convolution；
- SiLU gating；
- FP32 recurrent state；
- BF16 projected activations。

论文实现了四类 Triton kernel：

1. serial tree-state verifier；
2. parallel factorized tree-state verifier；
3. 针对链式 parent 关系 $p_j=j-1$ 的 parallel chain verifier；
4. fused commit-and-verify kernel。

主要 kernel 配置包括：

- head-dimension tile size：32；
- key block size：$B_K=32$；
- factorized kernel 将 speculative length pad 到 16 的倍数，以控制 register pressure。

运行时主要改动：

1. cache holder 从 KV tensors 改为每层固定大小的 FP32 recurrent states $(S,C)$；
2. target layer forward 返回 candidate logits 和 factor records；
3. proposal converter 把 draft tree mask 转换为 parent array；
4. generation loop 不再截断 KV Cache，而是收集 accepted factor records。

论文没有给出公开源码的逐文件位置，因此这里是对论文接口的总结，不是仓库源码追踪。

### 9.2 实验设置

位置：论文第 10 至 11 页，Section 8.1。

- 硬件：单张 NVIDIA H100；
- target：公开 GDN-1.3B checkpoint；
- drafter：EAGLE-style GDN draft layer；
- decode：greedy；
- 每个 prompt 最多生成 128 个新 token；
- chain route：8-token draft budget；
- tree route：最多 16 个 draft nodes，top-$k=4$；
- Mixed：480 prompts；
- GSM8K：1,319 prompts；
- HumanEval：164 prompts。

所有 speculative route 都与 autoregressive decoding 做 exact output hash 检查；Triton kernel 也与 sequential 或 standalone GDN reference 对比。

### 9.3 端到端性能

位置：论文第 11 页，图 9。

| 数据集 | SpecLA-Chain | SpecLA-Tree |
|---|---:|---:|
| Mixed | 1.26x | 1.42x |
| GSM8K | 1.54x | 1.70x |
| HumanEval | 1.03x | 1.06x |

GSM8K 上 Tree 相对 Chain：

- average accepted length：3.47 增至 5.00；
- first-token match：0.68 增至 0.81。

Mixed 上：

- average accepted length：3.08 增至 3.92；
- first-token match：0.59 增至 0.73。

HumanEval 的 Tree average accepted length 只有 2.34，因此加速仅为 1.06x。

可用下面的近似关系理解性能趋势：

$$
\mathrm{Speedup}
\approx
\frac{\mathrm{committed\ tokens\ per\ round}}
{\mathrm{draft\ cost}
+\mathrm{verify\ cost}
+\mathrm{commit\ cost}}.
$$

这不是论文给出的严格性能公式，只用于说明 accepted progress 必须足够大，才能摊薄 speculative overhead。

### 9.4 Tree verification 微基准

位置：论文第 11 至 12 页，图 10 和表 1。

各路径相对相同 proposal tree 的 root-to-leaf replay：

| Top-$k$ | Config depth | Nodes | Serial | Parallel | Hybrid |
|---:|---:|---:|---:|---:|---:|
| 2 | 12 | 16 | 1.31x | 0.58x | 2.20x |
| 2 | 24 | 32 | 1.36x | 1.02x | 2.55x |
| 4 | 4 | 16 | 3.73x | 1.70x | 5.14x |
| 4 | 4 | 32 | 4.74x | 3.47x | 7.11x |
| 4 | 12 | 16 | 1.19x | 0.52x | 1.80x |
| 4 | 24 | 32 | 1.30x | 0.95x | 2.19x |

Hybrid 在所有分组中至少达到 1.80x，最高达到 7.11x。Parallel 路径多次低于 1x，说明完整 factorized parallel verification 在短窗口下可能连 root-to-leaf replay 都不如。

### 9.5 Factor buffer 与 delayed update

位置：论文第 12 页，图 11。

在 24 层、$h_v=8$、$d_k=d_v=256$ 的状态提交测试中：

- token replay：19.70 ms 至 35.96 ms；
- factor buffer：5.15 ms 至 13.10 ms；
- factor reuse 延迟降低：2.74x 至 4.28x。

FP32 correctness check：

- recurrent-state 最大差异不超过 $9.54 \times 10^{-7}$；
- convolution-state difference 为 0。

在 verify length 为 4、accepted length 为 1 至 8 的 delayed update 测试中：

- separate verify-then-commit：0.102 ms 至 0.153 ms；
- fused update-and-verify：0.071 ms 至 0.133 ms；
- 延迟降低：1.15x 至 1.44x。

### 9.6 接受率与盈利区间

位置：论文第 12 至 13 页，图 12。

在实测 1.3B target/runtime 上：

- $p=0.5$ 或 $p=0.6$：所有测试 draft length 都低于 1x；
- $p\approx0.7$：短 draft 可能略高于 1x，长 draft 仍可能亏损；
- $p\ge0.8$：所有测试 draft length 都有收益；
- oracle acceptance、$L=12$：最高 3.71x。

论文还给出 9B-scale projection，oracle acceptance 时达到 4.61x。但它不是实测 9B SpecLA：只替换了 autoregressive target-only latency，speculative-side overhead 仍由 1.3B GDN runtime 校准。因此它只能说明趋势，不能作为 9B 端到端实测结果。

---

## 10. 贡献、局限与结论

### 10.1 主要贡献

#### 贡献一：识别了正确的运行时抽象

~~~text
Transformer rollback
= 删除 token-indexed KV suffix

Recurrent-state rollback
= 恢复并提交正确的 state trajectory
~~~

这不是替换缓存数据结构就能解决的问题，而是 verification、acceptance 和 commitment 语义都发生了变化。

#### 贡献二：联合设计 kernel 与状态生命周期

- topology-aware kernel 决定怎样验证；
- factor buffer 决定验证产物怎样保存；
- delayed update 决定 accepted state 怎样提交；
- confidence pruning 决定哪些候选值得验证。

这些机制形成完整闭环，而不是孤立的 micro-kernel 优化。

#### 贡献三：证明完全并行不一定更快

Parallel 路径在多个真实形状上低于 serial，甚至低于 root-to-leaf baseline。Hybrid 的优势说明，短窗口下减少 setup、reduction 和中间张量流量，往往比追求最大 token parallelism 更重要。

#### 贡献四：Factor buffer 具有迁移价值

保存“足以重建 accepted update 的最小验证记录”这一思想，不只适用于 GDN，也能推广到 GLA、DeltaNet 和 Mamba 类模型，只是具体 factor schema 不同。

### 10.2 主要局限

#### 局限一：模型和硬件覆盖有限

端到端实验只覆盖单张 H100、一个公开 GDN-1.3B、纯 Gated DeltaNet target 和 greedy decoding。尚未证明能直接扩展到大型 hybrid attention/GDN、MoE、tensor parallel、continuous batching 或 Ascend NPU。

#### 局限二：9B 结果只是 projection

图 12(b) 不是完整 9B speculative runtime 实测，不能把 4.61x 表述成 9B SpecLA 的真实端到端加速。

#### 局限三：缺少随机采样实验

论文沿用标准 speculative acceptance 的分布保持思想，但端到端正确性主要通过 greedy exact hash 验证。没有展示 temperature、top-$p$ 或 top-$k$ sampling 下的分布一致性实验。

#### 局限四：端到端计时重复不足

论文说明 full-test 中每个 prompt-route pair 只计时一次，置信区间来自 prompt 维度 bootstrap。这能反映 workload 差异，却不能充分分离 GPU 时钟、运行噪声和单次测量误差。kernel microbenchmark 的 warmup 加 median 方法更稳健。

#### 局限五：Hybrid 调度细节不完整

论文没有完整给出：

- heavy edge 的精确选择规则；
- 是否使用 cost model；
- boundary state 的实际布局；
- ready chains 的 stream/launch 调度；
- serial、parallel、hybrid 的动态选择策略。

#### 局限六：Delayed update 的边界流程较少

论文没有详细展开：

- 生成结束时最后一批 pending factors 如何 flush；
- EOS 出现在 accepted path 中时怎样提交；
- 下一轮 drafter feature 如何与 pending target state 对齐；
- batch 内不同序列 accepted length 不同时怎样融合；
- convolution state $C$ 的完整 speculative 生命周期。

这些是把论文原型集成到真实 serving runtime 时必须解决的问题。

### 10.3 最终结论

SpecLA 的核心不是提出新的 speculative acceptance 规则，而是把已有投机解码重新映射到 stateful linear-attention 的执行语义上。

最值得记住的三点：

1. **线性注意力不是没有缓存。** 它把不断增长的 KV 历史变成固定大小、但每个 token 都要读写的稠密递归状态。
2. **树验证不应只追求最大并行度。** SpecLA 通过链内状态驻留和链间并行，减少真正昂贵的状态物化与 HBM 流量。
3. **Factor buffer 是关键抽象。** 它既不保存完整状态，也不在接受后重跑 target，而是保存验证阶段已经生成、足以恢复 accepted update 的紧凑记录。

论文在 GDN-1.3B 和 H100 上证明了该方向可以获得最高 1.70x 的端到端收益，但收益强烈依赖 drafter 的接受率。未来要判断其真实工程价值，还需要在更大模型、hybrid 架构、并行 serving、随机采样和不同硬件平台上进一步验证。

