# MegaMoe 哨兵 token（dummy tokens）机制笔记

## 目录

1. [问题一：哨兵 token（dummy tokens）的工作原理](#问题一哨兵-tokendummy-tokens的工作原理)
2. [问题二：dummy 行数与 global_dummy_capacity 的疑问](#问题二dummy-行数与-global_dummy_capacity-的疑问)
3. [问题三：all-to-all 对称性的含义](#问题三all-to-all-对称性的含义)

---

## 问题一：哨兵 token（dummy tokens）的工作原理

**原问题：**

> 请你结合本目录下的vllm和vllm-ascend文件夹，以及C:\Code\cann\ops-transformer文件夹的代码内容，回答以下问题：哨兵 token"（dummy tokens）的工作原理需要结合代码讲透，最好是有图。它的实现是在_append_cann_megamoe_dummy_tokens方法中吗？

**回答：**

是的，核心实现就是 `_append_cann_megamoe_dummy_tokens`（位于 `vllm_ascend/ops/fused_moe/moe_comm_method.py:60-103`）。但它只是"运行时拼接"这一半——完整的哨兵 token 机制由**三个函数协同**完成，分工如下：

| 函数 | 位置 | 职责 |
|---|---|---|
| `get_cann_megamoe_dummy_token_capacity` | `utils.py:1034` | 算"全局需要多少行 dummy 才能覆盖所有 expert" |
| `get_cann_megamoe_buffer_params` | `utils.py:1046` | 把 dummy 容量**折算进**对称 buffer 的 `num_max_tokens_per_rank` |
| `_append_cann_megamoe_dummy_tokens` | `moe_comm_method.py:60` | 运行时真正把 dummy 拼到 tensor 里 |

下面按"为什么需要 → 怎么构造 → 怎么丢弃"讲透。

### 一、为什么需要哨兵 token

这要回到 MegaMoe 算子的性质：它是一个 **all-to-all 集合通信 + FFN 计算融合**的算子。在 EP（专家并行）下，每个 rank 只持有 `num_experts / ep_world_size` 个 expert，dispatch 阶段每个 rank 要把本地 token 按 `topk_ids` 路由发给"持有对应 expert"的那个 rank。

问题在于：**真实推理里 token 分布是不均的**。某一步可能出现某个 expert 一个 token 都没被路由到，对应那个 rank 在 combine 阶段就空转。对于 HCCL 集合通信来说，收发量必须对称（这跟上一个问题的"对称 buffer 形状必须一致"是同一件事）——一旦某个 rank 空转 / 收发不匹配，就可能挂死（SUSPECT REMOTE ERROR 507057 那一类）。

```
EP world = 4, num_experts = 8, 每个 rank 持有 2 个 expert

  rank0: experts {0,1}      rank1: experts {2,3}
  rank2: experts {4,5}      rank3: experts {6,7}

某一步真实 token 只命中了 expert 0/2/4:
  rank0 有活(e0)   rank1 有活(e2)   rank2 有活(e4)
  rank3 一个 token 都没有  →  all-to-all 收发不对称 → HCCL 挂死风险
```

哨兵 token 的作用就是**保证每个 expert 至少被路由到一次**，并且**每个 rank 追加的 dummy 行数完全相同**，从而让 all-to-all 的对称性成立。

### 二、逐步拆解 `_append_cann_megamoe_dummy_tokens`

用 `num_experts=8, num_topk=2, ep_world_size=4` 做例子（和单测 `test_a2_megamoe.py` 一致），逐行看：

#### 第 1 步：算容量

```python
global_dummy_capacity = get_cann_megamoe_dummy_token_capacity(num_experts, num_topk)  # ceil(8/2) = 4
local_dummy_capacity  = math.ceil(global_dummy_capacity / ep_world_size)              # ceil(4/4) = 1
```

- `global_dummy_capacity` = 全局最少需要的 dummy 行数 = `ceil(num_experts / num_topk)`。因为一行 dummy 有 `num_topk` 个路由槽，要让 `num_experts` 个 expert 都被覆盖，至少需要 `ceil(num_experts / num_topk)` 行。
- `local_dummy_capacity` = 每个 rank 分到的行数，用 **ceil** 保证：即使不能整除，每个 rank 的行数也**完全一致**（这是对称性的关键）。

#### 第 2 步：给每个 rank 分配一段互不重叠的全局行号

```python
global_dummy_rows = ep_rank_id * local_dummy_capacity + torch.arange(local_dummy_capacity, ...)
```

每个 rank 拿到连续一段行号：`[ep_rank_id * local, (ep_rank_id+1) * local)`。这样 4 个 rank 的行号区间互不重叠，拼起来就是 `0 .. global_dummy_capacity-1` 的连续空间。

#### 第 3 步：把行号映射成 topk 路由，再取模覆盖所有 expert

```python
dummy_topk_ids = (global_dummy_rows[:, None] * num_topk + torch.arange(num_topk, ...)).remainder(num_experts)
```

这是最精妙的一步。第 `r` 行 dummy 的 `num_topk` 个路由槽被映射成连续 id，再对 `num_experts` 取模：

```
  rank0 → row 0 → topk_ids = (0*2 + {0,1}) % 8 = {0, 1}
  rank1 → row 1 → topk_ids = (1*2 + {0,1}) % 8 = {2, 3}
  rank2 → row 2 → topk_ids = (2*2 + {0,1}) % 8 = {4, 5}
  rank3 → row 3 → topk_ids = (3*2 + {0,1}) % 8 = {6, 7}

  合并后 = {0,1,2,3,4,5,6,7}  ← 8 个 expert 全部覆盖 ✓
```

#### 第 4 步：构造 dummy 的数值（设计成"无害的垃圾"）

```python
dummy_hidden_states = torch.ones((local_dummy_capacity, hidden), ...)   # 全 1
dummy_topk_weights  = torch.full((local_dummy_capacity, num_topk), 1.0/num_topk, ...)  # 均匀权重
```

- hidden 全 `1`、权重均匀 `1/topk`：保证计算不会产生数值异常（NaN/Inf）。
- `x_active_mask` 里 dummy 位置置 `1`，标记为有效 token。

#### 第 5 步：拼接到真实 tensor 后面

```python
hidden_states = torch.cat((hidden_states, dummy_hidden_states), dim=0)   # [N+local, hidden]
topk_ids      = torch.cat((topk_ids, dummy_topk_ids), dim=0)
topk_weights  = torch.cat((topk_weights, dummy_topk_weights), dim=0)
x_active_mask = torch.cat((x_active_mask, dummy_mask), dim=0)
```

拼接后整批送进 `mega_moe(...)`。**关键点：算子对 dummy 完全无感知**——它看到的只是"一批多了几行的普通 token"，按 `topk_ids` 正常做 all-to-all dispatch。

### 三、数据流全景图

```
                    ┌──────────────────────────────────────────────┐
                    │  真实 token: hidden_states[0 .. N-1]  (N 行)   │
                    └──────────────────────────────────────────────┘
                                        │
                   _append_cann_megamoe_dummy_tokens()
                                        │  torch.cat 追加
                                        ▼
        ┌───────────────────────────────────────────────────────────┐
        │ hidden_states[0 .. N+local-1]  ← 尾部 local 行是全 1 的 dummy │
        └───────────────────────────────────────────────────────────┘
                                        │
                         mega_moe(x, topk_ids, topk_weights, ...)
                         （算子内部 all-to-all dispatch → FFN → combine）
                                        │
                                        ▼
        ┌───────────────────────────────────────────────────────────┐
        │  out[0 .. N+local-1]  ← 真实输出 + 局部垃圾输出              │
        └───────────────────────────────────────────────────────────┘
                                        │
                        return out[:original_num_tokens]   ← 裁掉 dummy 行
                                        │
                                        ▼
                           只把真实 token 的输出交给上层
```

对应代码在 `moe_comm_method.py:481-502`：

```python
out, expert_tokens = self.mega_moe(...)
# ...
return out[:original_num_tokens], expert_tokens
```

`original_num_tokens` 就是在拼接**之前**记录下来的真实行数（`moe_comm_method.py:95`），最后用切片把 dummy 的垃圾输出丢掉。

### 四、触发条件（什么时候才走这条路）

dummy 注入不是所有 MegaMoe 场景都启用，只在 **A2（910B）专用模式**下：

```python
# ascend_forward_context.py:51
def _is_a2_megamoe_enabled(ascend_config):
    return get_ascend_device_type() == AscendDeviceType.A2 and ascend_config.enable_fused_mc2 == 2
```

在 `_apply_cann_mega_moe` 里（`moe_comm_method.py:468-479`）：

```python
if _is_a2_megamoe_enabled(get_ascend_config()):
    hidden_states, topk_ids, topk_weights, x_active_mask, original_num_tokens = (
        _append_cann_megamoe_dummy_tokens(...)
    )
```

也就是说：只有 **A2 芯片 + `enable_fused_mc2 == 2`（走 CANN MegaMoe 融合算子）** 时，才注入哨兵 token。其他芯片（A3/A5）或 `enable_fused_mc2 == 1` 走的是不同路径，不需要这个机制。

### 五、小结

- **是的**，运行时主实现就是 `_append_cann_megamoe_dummy_tokens`。
- 它的本质是：**在算子外部、纯 Python 层**，向每个 rank 追加**等量**的 dummy 行，并让这些行的路由（通过 `行号 × topk + 槽位` 取模）全局覆盖所有 expert，从而（1）保证 all-to-all 收发对称不挂死，（2）保证每个 rank 的 tensor 形状一致满足对称 buffer 要求。
- dummy 的数值被设计成无害（全 1 hidden + 均匀权重），计算后被 `out[:original_num_tokens]` 裁掉丢弃。

---

## 问题二：dummy 行数与 global_dummy_capacity 的疑问

**原问题：**

> 对于以上回答，我有疑问：
> dummy 行数是不是序列的长度？比如上个例子中，填满8个专家，topk=2时，最少也得有4行序列。local_dummy_capacity实际上是每个rank上每个专家分到的行数，如果 global_dummy_capacity表示全局最少需要的 dummy 行数，难道它不是一直等于1的吗？

**回答：**

你的三个点里，第一点对，第二、三点需要纠正。核心混淆在"行数"到底指什么、以及"全局 vs 每 rank"这两个层次。我逐个讲。

### 先纠正：local_dummy_capacity 不是"每个专家分到的行数"

`local_dummy_capacity` 是**每个 rank 追加的 dummy token 行数（即序列长度增量）**，它按 **rank** 划分，不按专家划分。dummy 行和专家之间没有"一行对应一个专家"的关系——一行 token 有 `topk` 个路由槽，一行能覆盖 `topk` 个专家。

### 核心：为什么 global_dummy_capacity 不恒等于 1

你的直觉卡在一个地方：**1 行 dummy 到底能覆盖几个专家？** 答案不是 1 个，而是 `topk` 个。

因为 dummy 也是一行 token，它有 `topk` 个路由槽（每个 token 选中的专家数）。所以：

```
覆盖全部 num_experts 个专家，最少需要的行数 = ceil(num_experts / num_topk)

num_experts=8, num_topk=2 → ceil(8/2) = 4 行
```

1 行只有 2 个槽，只能命中 2 个专家，填不满 8 个。所以 `global_dummy_capacity` 是 **4**，不是 1。

**为什么上个例子 `local_dummy_capacity` 恰好 = 1？** 因为 global=4 正好被 ep=4 个 rank 平分。这是巧合，不是规律。换个配置就不是 1 了。

### 用反例打破"恒等于 1"

| 配置 | global = ceil(N/topk) | local = ceil(global/E) |
|---|---|---|
| N=8, topk=2, E=4（上个例子） | 4 | **1**（恰好） |
| N=8, topk=2, E=2 | 4 | **2** |
| N=256, topk=8, E=32 | 32 | **1** |
| N=256, topk=8, E=4 | 32 | **8** |

看第二行 `N=8, topk=2, E=2`：每个 rank 持有 `8/2 = 4` 个专家，但一行 token 只有 2 个槽，所以**每个 rank 必须追加 2 行**才能覆盖自己本地的 4 个专家。local=2，不是 1。

### 用图看清楚"行 → 专家"的映射

`N=8, topk=2, E=2` 这个例子（特意让 local ≠ 1）：

```
每个 rank 持有 4 个专家：
  rank0: {0,1,2,3}        rank1: {4,5,6,7}

global_dummy_capacity = ceil(8/2) = 4   ← 所有 rank 加起来共 4 行
local_dummy_capacity  = ceil(4/2) = 2   ← 每个 rank 追加 2 行（不是 1!）

rank0 追加 2 行 dummy：
  row 0 → topk_ids = (0*2 + {0,1}) % 8 = {0, 1}   ┐
  row 1 → topk_ids = (1*2 + {0,1}) % 8 = {2, 3}   ┘ 覆盖 {0,1,2,3} ← rank0 本地全部专家

rank1 追加 2 行 dummy：
  row 2 → topk_ids = (2*2 + {0,1}) % 8 = {4, 5}   ┐
  row 3 → topk_ids = (3*2 + {0,1}) % 8 = {6, 7}   ┘ 覆盖 {4,5,6,7} ← rank1 本地全部专家
```

关键结论：

- **1 行 = 1 个 token，有 topk 个路由槽 → 1 行覆盖 topk 个专家。**
- rank0 有 4 个专家要填，1 行（2 个槽）填不满，**必须 2 行**。
- 所以"全局最少行数"是 `ceil(总专家数 / topk)`，只有恰好被 ep 整除时才可能让 local 落到 1。

### 三个层次的关系，一句话总结

```
num_experts          → 全局专家总数
global_dummy_capacity = ceil(num_experts / num_topk)   → 覆盖全部专家需要的最少行数（所有 rank 之和）
local_dummy_capacity  = ceil(global_dummy_capacity / ep_world_size)  → 每个 rank 分摊到的行数
```

`global` 是"总量"，`local` 是"每个 rank 的份额"。你之前把 `local` 当成"每个专家分到的行数"，又把 `global` 想成"每个 rank 都只需要 1 行"，于是两层都偏了——实际上它们是"总量 / 份额"的关系，而且总量由 `num_experts / num_topk` 决定，不是 1。

---

## 问题三：all-to-all 对称性的含义

**原问题：**

> 上面哨兵token的设计目的是给hidden_states和weight拼接了一个定长的dummy张量，作用是给每个 rank 追加的 dummy 行数完全相同,从而让 all-to-all 的对称性成立。这里的all-to-all 的对称性是什么意思？

**回答：**

"对称性"这个词在 MegaMoe 这个上下文里，指的是一件事的两个层面：**所有 EP rank 参与这次 all-to-all 集合通信时，（1）声明的 buffer 形状完全一致，（2）每个 rank 都有 token 可发、可收，不存在空 rank。** 我把它讲透。

### 一、先看 EP 下 all-to-all 通信本身

在专家并行（EP）下，一个 token 的 `topk` 个专家可能分布在**不同的 rank** 上，所以需要两次 all-to-all：

```
EP=4, 每 rank 持有 2 个专家, topk=2

  rank0 持有 {e0,e1}    rank1 持有 {e2,e3}
  rank2 持有 {e4,e5}    rank3 持有 {e6,e7}

① dispatch（发送）：每个 rank 把本地 token 按 topk 路由，发给"持有目标专家"的 rank
   例：token A 的 topk = {e0, e4} → 发 1 份给 rank0（算 e0）、1 份给 rank2（算 e4）

② 各 rank 用本地专家算 FFN

③ combine（回传）：每个 rank 把算完的 token 送回它"原来所在"的 rank
```

关键点：**每个 rank 在 dispatch 阶段发给谁、发多少，完全由路由结果动态决定，天然不均匀**。某一步可能 rank3 持有的 e6/e7 一个 token 都没被命中。

### 二、"对称性"指的两个层面

#### 层面 1：形状对称（buffer shape symmetry）

MegaMoe 是**融合算子**，它不是"先统计 token 数、再动态分配不等长 buffer"的普通 all-to-all，而是**预先分配一个对称 buffer 做集合通信**。这个 buffer 的形状由 `num_max_tokens_per_rank` 等参数决定，而这个参数**必须在所有 rank 上是一个相同的常数**：

```
num_max_tokens_per_rank 一致（✓ 对称）：
  rank0 = 100   rank1 = 100   rank2 = 100   rank3 = 100
  → buffer 形状相同 → HCCL 正常

num_max_tokens_per_rank 不一致（✗ 不对称）：
  rank0 = 100   rank3 = 50   ← 某 rank 比别的少
  → buffer 形状不匹配 → HCCL 挂死（SUSPECT REMOTE ERROR 507057）
```

这就是代码里那条注释的本意（`moe_comm_method.py:367`）："**Symmetric buffer dimensions must be identical across the EP group**"——`get_symm_buffer_for_mega_moe` 里的 "symm" 就是 "symmetric" 的缩写。

#### 层面 2：参与对称（participation symmetry）

即使形状一致了，还有一层更隐蔽的不对称：如果某个 rank 这一 step **没有任何 token 命中它持有的专家**，它在 all-to-all 的接收侧就是空的：

```
没有 dummy 时的不对称场景：
  某一步所有真实 token 只命中 e0~e5（rank0/1/2 的专家）
  rank3 持有的 e6/e7 完全没人路由到

  dispatch 阶段：
    rank0/1/2：有 token 要发、要收 → 正常推进
    rank3    ：没有 token 要发，也没有 token 要收 → 空转

  但 all-to-all 是"全体 rank 一起"的集合操作：
    rank3 在接收侧死等数据（永远等不到）
    发送方在等 rank3 的参与/完成信号（rank3 却觉得自己没事可做）
    → 互相等待 → HCCL 挂死
```

### 三、为什么"不对称"会挂死（本质）

HCCL 集合通信是**全体一致的同步操作**：一次 all-to-all 里，每个 rank 都按同一个协议、同一个 buffer 布局收发，谁都不能缺席或早退。一旦出现"某个 rank 的形状和别人不同"或"某个 rank 无数据可收"，就会出现**死等（deadlock）**——一方在等对方的数据，对方却在等它的信号，谁都动不了。这就是注释里 "HCCL aborts / 挂死" 的根源。

### 四、dummy token 分别满足这两层对称

回到哨兵 token 的设计，它同时解决了上面两层：

| 对称层面 | dummy 做了什么 | 对应代码 |
|---|---|---|
| **形状对称** | 每个 rank 追加**等量**的 dummy 行（`local_dummy_capacity` 对所有 rank 相同），且 `num_max_tokens_per_rank = base + dummy_token_capacity` 中的两个加数都是 rank 无关的，故总和 rank-invariant | `_append_cann_megamoe_dummy_tokens` + `get_cann_megamoe_buffer_params` |
| **参与对称** | dummy 的 topk 路由通过取模**全局覆盖所有 expert**，保证每个 rank 持有的专家至少被 1 个 token 命中，于是每个 rank 在接收侧都不为空 | `dummy_topk_ids` 的构造 + `cover all experts globally` |

所以一句话总结：**"all-to-all 的对称性" = 所有 rank 用同一个形状的 buffer、并且都有 token 参与这次集合通信；任何 rank 在形状或参与度上和别人不一致，都会让集合通信死等而挂死。** dummy token 通过"等量追加 + 全局覆盖专家"同时保住了这两点。
