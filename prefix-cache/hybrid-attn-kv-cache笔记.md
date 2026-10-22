# 混合注意力与 KV Cache 加载逻辑

## 目录

- [问题](#问题)
- [源码范围](#源码范围)
- [一、结论概览](#一结论概览)
- [二、模型层面的混合注意力](#二模型层面的混合注意力)
- [三、为什么需要多个 KV Cache Group](#三为什么需要多个-kv-cache-group)
- [四、本地 KV Cache 的分配和绑定](#四本地-kv-cache-的分配和绑定)
- [五、本地 Prefix Cache 命中并不是一次数据加载](#五本地-prefix-cache-命中并不是一次数据加载)
- [六、多个 Group 如何协调 Prefix Cache 命中](#六多个-group-如何协调-prefix-cache-命中)
- [七、外部 KV Cache 的加载主流程](#七外部-kv-cache-的加载主流程)
- [八、AscendStore 非 Layerwise 加载](#八ascendstore-非-layerwise-加载)
- [九、Ascend Mooncake Layerwise Hybrid 加载](#九ascend-mooncake-layerwise-hybrid-加载)
- [十、Mamba/GDN 加载后的 State Copy](#十mambagdn-加载后的-state-copy)
- [十一、完整示例](#十一完整示例)
- [十二、最终理解](#十二最终理解)
- [十三、验证边界](#十三验证边界)
- [十四、Sliding Window Attention 在哪些模型中使用](#十四sliding-window-attention-在哪些模型中使用)

## 问题

请结合 `D:\Code\vllm-ascend\vllm-ascend` 和 `D:\Code\vllm-github` 的内容，分析什么是混合注意力，以及它的 KV Cache 加载逻辑是怎样的。

## 源码范围

本文基于以下工作区进行静态源码分析：

- vLLM-Ascend：分支 `l1_922`，提交 `cec9d64be060b583e7ec53689ffd360c70ea83fe`
- 上游 vLLM：分支 `main`，提交 `0f2a15c9277f34c9afe141cf578d1f02b3bebdfe`

两个工作区都存在未跟踪文件，分析过程中未修改上述仓库。Ascend 的 Mooncake layerwise hybrid 实现文档注明最初基于上游 `be427041bf63a620e4a637b60f2656e87dcdf8f6`，因此当前两个仓库并不是严格一一对应的相同基线。以下结论以当前工作区中的实际代码为准。

## 一、结论概览

“混合注意力”在这两个仓库中至少有三层含义：

1. **模型结构层面的混合注意力**

   同一个模型的不同层使用不同的上下文建模机制，例如 Full Attention、Sliding Window Attention，以及 Linear Attention、GDN 或 Mamba。

2. **vLLM KV Cache Manager 层面的 hybrid cache**

   不同 attention 类型对应不同的 `KVCacheSpec`，因而需要不同的 block table、命中判断和生命周期管理器。vLLM 将模型层划分为多个 `kv_cache_groups`，并求出所有必要 group 都能恢复的共同前缀。

3. **vLLM-Ascend Mooncake layerwise hybrid transfer**

   多个 KV cache group 分别存入 Mooncake，并按物理层逐层加载。不同 group 可以有不同的 block size、层集合、byte range 和提交边界。

最重要的原则是：

> 混合模型的 KV cache 命中长度不能由某一个 Full Attention group 单独决定，而必须是所有参与恢复的 cache group 共同可达的最长前缀。只要某个 group 的必要状态缺失，该段前缀就不能安全地跳过计算。

## 二、模型层面的混合注意力

### 2.1 Qwen3.5/Qwen3-Next：Full Attention 与 GDN Linear Attention 混合

以 Qwen3.5 为例，每层类型由 HF 配置中的 `config.layer_types[layer_idx]` 决定：

- `D:\Code\vllm-github\vllm\model_executor\models\qwen3_5.py:250-255`

构造具体层时：

- `linear_attention` 创建 `QwenGatedDeltaNetAttention`；
- `full_attention` 创建 `Qwen3NextAttention`。

对应代码位置：

- `D:\Code\vllm-github\vllm\model_executor\models\qwen3_5.py:120-162`
- `D:\Code\vllm-github\vllm\model_executor\models\qwen3_next.py:463-509`

前向计算同样按层类型分流：

- linear 层调用 `self.linear_attn(...)`；
- full 层调用 `self.self_attn(...)`。

对应代码：

- `D:\Code\vllm-github\vllm\model_executor\models\qwen3_next.py:549-576`

vLLM 在统计模型层数时，也明确将：

- `full_attention` 计为 `attention`；
- `linear_attention` 计为 `linear_attention`。

参见：

- `D:\Code\vllm-github\vllm\config\model.py:1713-1725`

因此，Qwen3.5 所谓混合注意力通常不是“一层内部同时执行两种 attention”，而是不同 transformer layer 交替使用不同机制，例如：

```text
Layer 0   Linear Attention / GDN
Layer 1   Linear Attention / GDN
Layer 2   Linear Attention / GDN
Layer 3   Full Attention
Layer 4   Linear Attention / GDN
...
```

具体排列和比例由模型的 `layer_types` 配置决定。

### 2.2 Full Attention 的 Cache

Full Attention 对历史 token `j` 保存：

```text
K_j, V_j
```

新 token `t` 的注意力需要读取历史 K/V：

```text
O_t = softmax(Q_t K_<=t^T / sqrt(d)) V_<=t
```

因此，其 cache 容量通常随上下文长度线性增长。

普通 attention 层的 `get_kv_cache_spec()` 会：

- 当设置了 `sliding_window` 时返回 `SlidingWindowSpec`；
- 否则返回 `FullAttentionSpec`。

代码位置：

- `D:\Code\vllm-github\vllm\model_executor\layers\attention\attention.py:602-668`

### 2.3 Linear Attention、GDN、Mamba 的 Cache

Linear Attention/GDN/Mamba 通常不保存完整的历史 K/V，而是递归更新有限状态，可以抽象为：

```text
S_t = f(S_(t-1), K_t, V_t)
O_t = g(Q_t, S_t)
```

这里保存的是：

- convolution state；
- recurrent/SSM state；
- GDN 的矩阵状态；
- speculative decoding 需要的附加状态。

vLLM 为这些层生成的不是 `FullAttentionSpec`，而是 `MambaSpec`。构造入口为：

- `D:\Code\vllm-github\vllm\model_executor\layers\mamba\abstract.py:65-84`

其中记录了状态张量 shape、dtype、Mamba block size、cache mode、speculative blocks，以及状态是否在 TP rank 间复制。

因此，“KV cache”在 vLLM 接口中是广义名称：

> 对 Full Attention，它是真正的 K/V；对 GDN/Mamba，它实际是恢复递归计算所需的 state cache。

## 三、为什么需要多个 KV Cache Group

### 3.1 KVCacheSpec 描述单层 Cache 语义

`KVCacheSpec` 描述一个模型层的 cache 格式。其核心字段是 `block_size`，即一个逻辑 block 覆盖多少 token，同时通过 `page_size_bytes` 描述一个物理页的字节数。

相关定义：

- `KVCacheSpec`：`D:\Code\vllm-github\vllm\v1\kv_cache_interface.py:156-260`
- `FullAttentionSpec`：`D:\Code\vllm-github\vllm\v1\kv_cache_interface.py:536-615`
- `SlidingWindowSpec`：`D:\Code\vllm-github\vllm\v1\kv_cache_interface.py:811-866`
- `MambaSpec`：`D:\Code\vllm-github\vllm\v1\kv_cache_interface.py:1014-1085`

`KVCacheGroupSpec` 表示一组共享同一个 block table、由同一个 cache manager 统一管理的模型层：

- `D:\Code\vllm-github\vllm\v1\kv_cache_interface.py:1410-1427`

需要注意：共享 block table 不等于共享相同的 K/V 内容。各层仍然有独立的物理 cache 区域，只是同一请求在这些层使用相同的逻辑 block 编号映射。

### 3.2 分组不是简单地“每种 Attention 一个 Group”

vLLM 会根据模型的重复模式分组。例如模型包含：

```text
10 个 Full Attention 层
20 个 Sliding Window 层
重复模式：Full, SW, SW
```

可以形成三个 group：

```text
group 0: full.0, full.1, ..., full.9
group 1: sw.0,   sw.2,   ..., sw.18
group 2: sw.1,   sw.3,   ..., sw.19
```

每个重复模式位置维护一张 block table，然后将该 block table 应用于相应的十个物理层。

详细设计说明位于：

- `D:\Code\vllm-github\vllm\v1\core\kv_cache_utils.py:1438-1495`

代码先按可兼容的 spec 聚类，再拆成相同层数的 group；必要时添加 padding layer：

- `D:\Code\vllm-github\vllm\v1\core\kv_cache_utils.py:1503-1575`

总入口 `get_kv_cache_groups()` 位于：

- `D:\Code\vllm-github\vllm\v1\core\kv_cache_utils.py:2288-2384`

其主要决策过程为：

```text
attention-free
    -> 不创建 group
所有层 spec 完全一致
    -> 一个 group
所有层属于同一种统一 cache 类型
    -> 一个统一 group
模型专用布局
    -> 模型专用分组
可打包的异构 cache
    -> packed groups
其他普通混合模型
    -> 统一 page size，再生成多个 group
```

如果指定 `--disable-hybrid-kv-cache-manager`，vLLM 会尝试在分配层面将 Sliding Window 等 cache 退化为 Full Attention cache，但 attention kernel 仍然可以执行 sliding-window 计算。相关说明：

- `D:\Code\vllm-github\vllm\v1\kv_cache_interface.py:536-543`

## 四、本地 KV Cache 的分配和绑定

### 4.1 分配统一 Backing Buffer

上游 `init_kv_cache()` 创建一块大的 `int8` backing allocation，然后根据每个 `KVCacheTensor` 的：

- `offset`；
- `layer_stride`；
- `block_stride`；
- group spec；
- block 数量；

切出各层 cache view。

实现位置：

- `D:\Code\vllm-github\vllm\v1\worker\utils.py:390-455`

物理地址关系可以抽象为：

```text
addr(layer, block)
  = base
  + offset
  + layer * layer_stride
  + block * block_stride
```

其中 `layer` 是 group 内层索引，`block` 是物理 block ID。

### 4.2 绑定到模型层

`bind_kv_cache()` 完成两件事：

1. 将 cache 放入 ModelRunner 的 cache 列表；
2. 根据 layer name 绑定到 forward context 中相应的 Attention/Mamba 模块。

代码位置：

- `D:\Code\vllm-github\vllm\v1\worker\utils.py:591-641`

Mamba/GDN 层绑定之后，还会将一块原始分配拆成需要的 conv/SSM 状态 view：

- `D:\Code\vllm-github\vllm\v1\worker\utils.py:644-656`

Ascend V2 ModelRunner 基本复用上游初始化，只额外处理 graph manager、PCP 和 KVPPRuntime：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\worker\v2\model_runner.py:265-300`

## 五、本地 Prefix Cache 命中并不是一次数据加载

如果 KV 已在当前 worker 的 NPU block pool 中，命中本地 prefix cache 通常不需要复制 KV 数据。调度器主要进行：

1. 根据 token hash 找到已有物理 block；
2. 增加 block 引用；
3. 将物理 block ID 写入请求对应 group 的 block table；
4. attention kernel 通过 block table 直接读取原 cache。

即：

```text
本地 prefix cache hit
    ≈ 重用 block ID / 更新 block table
    ≠ 从磁盘或远端重新加载 KV
```

上游 `BlockTables` 为每个 KV cache group 创建独立 block table：

- `D:\Code\vllm-github\vllm\v1\worker\gpu\block_table.py:17-113`

物理 slot 换算关系近似为：

```text
slot = block_id * kernel_block_size + offset_in_block
```

对应 kernel：

- `D:\Code\vllm-github\vllm\v1\worker\gpu\block_table.py:282-351`

## 六、多个 Group 如何协调 Prefix Cache 命中

### 6.1 不能使用某一个 Group 的最大命中长度

假设各 group 的命中情况为：

| Group | 命中长度 |
| --- | ---: |
| Full Attention KV | 1024 token |
| Sliding Window KV | 896 token |
| GDN/Mamba state | 768 token |

请求不能直接从 token 1024 继续，而只能从所有必要状态一致的边界恢复：

```text
L_usable = 768
```

原因是：从 768 到 1024 的 Full Attention KV 虽然存在，但 GDN 的递归状态缺失；直接跳过这段计算会产生错误结果。

### 6.2 Fixed-Point 命中协调算法

`HybridKVCacheCoordinator.find_longest_cache_hit()` 使用单调收缩的 fixed-point 算法：

- `D:\Code\vllm-github\vllm\v1\core\kv_cache_coordinator.py:833-968`

可以概括为：

```text
candidate = 最大可能命中长度

repeat:
    old_candidate = candidate

    对每种 cache spec：
        询问该 group 在 candidate 边界是否可恢复
        如果只能恢复更短前缀：
            candidate = 更短长度

until candidate 不再缩短
```

数学形式为：

```text
L_(n+1) = min_i F_i(L_n)
```

其中 `F_i` 表示第 `i` 类 cache manager 在当前候选边界下能够恢复的最长长度。候选长度只会减小，且下界为 0，因此算法必然收敛。

Full Attention 被放在最前面，因为 Full Attention 的命中具有向下封闭性：如果 1024 token 的完整 KV 存在，那么更短的 block-aligned 前缀同样存在。相关逻辑：

- group 排序：`D:\Code\vllm-github\vllm\v1\core\kv_cache_coordinator.py:728-779`
- fixed-point 循环：`D:\Code\vllm-github\vllm\v1\core\kv_cache_coordinator.py:874-944`
- 最终裁剪 Full Attention block：`D:\Code\vllm-github\vllm\v1\core\kv_cache_coordinator.py:946-968`

## 七、外部 KV Cache 的加载主流程

外部加载指从 Mooncake、AscendStore、CPU/SSD offload 或其他 connector，将数据真正写入当前 worker 的本地 cache block。

总体流程为：

```text
模型层生成 KVCacheSpec
        ↓
Engine 合并并生成 kv_cache_groups
        ↓
Scheduler 查本地 prefix cache
        ↓
Connector 查外部 cache
        ↓
所有 group 协调出共同命中长度
        ↓
KVCacheManager 为外部命中分配本地物理 block
        ↓
Connector metadata 携带：
  请求、hash、每组本地 block IDs、load range
        ↓
Worker 将远端 bytes 写入相应 NPU block 地址
        ↓
更新/写入各 group block table
        ↓
Attention 或 GDN 从本地 cache/state 继续 forward
```

### 7.1 Scheduler 先查本地，再查外部

请求第一次调度时，Scheduler：

1. 调用 `_get_local_prefix_cache_hit()` 查询本地；
2. 调用 `connector.get_num_new_matched_tokens()` 查询外部；
3. 合并本地和外部命中；
4. 处理本地 partial block 与更长远端命中的冲突。

代码位置：

- `D:\Code\vllm-github\vllm\v1\core\sched\scheduler.py:918-1016`

如果本地命中存在 sub-block tail：

- 只有当远端命中严格超过完整本地命中时，才丢弃本地 tail，让远端加载覆盖；
- 否则保留本地 tail，并放弃外部加载。

相关代码：

- `D:\Code\vllm-github\vllm\v1\core\sched\scheduler.py:964-984`

### 7.2 为外部 KV 分配本地落点

远端 KV 不能悬空，必须先为每个 group 分配本地 block ID。`allocate_slots()` 将请求布局划分为：

```text
| 已计算 | 新本地命中 | 外部命中 | 要计算的新 token | lookahead |
```

随后执行：

1. 清理 Sliding Window 已不需要的旧 block；
2. 为外部命中的 token 分配 block；
3. 为本轮计算和 speculative lookahead 分配 block；
4. 异步加载时，延迟将这些 block 标记为已缓存。

代码位置：

- `D:\Code\vllm-github\vllm\v1\core\kv_cache_manager.py:371-608`

外部 block 被纳入请求 block table 的关键位置为：

- `D:\Code\vllm-github\vllm\v1\core\kv_cache_manager.py:572-585`

分配完成后，Scheduler 调用 connector 的 `update_state_after_alloc()`，将本地 block IDs 交给 connector：

- `D:\Code\vllm-github\vllm\v1\core\sched\scheduler.py:1224-1233`

### 7.3 同步加载与异步加载

上游 worker connector 的控制逻辑位于：

- `D:\Code\vllm-github\vllm\v1\worker\gpu\kv_connector.py:53-133`

其行为为：

- `has_sync_kv_loads=True`：在 forward 之前调用 `start_load_kv()`；
- 异步加载：先将请求置为 `WAITING_FOR_REMOTE_KVS`，本轮不执行真实 forward；
- forward 后提交异步 load，避免提交操作占用 forward 临界路径；
- load 完成后 worker 返回 `finished_recving`，Scheduler 下一步重新调度请求。

异步请求状态切换：

- `D:\Code\vllm-github\vllm\v1\core\sched\scheduler.py:1251-1260`

完成通知处理：

- `D:\Code\vllm-github\vllm\v1\core\sched\scheduler.py:3050-3075`

## 八、AscendStore 非 Layerwise 加载

AscendStore worker 注册 cache 时，会提取每个 group 的：

- cache base address；
- 每 block 的实际字节数；
- block stride；
- 每个物理层包含的 cache entry 数量；
- group 内的物理层顺序。

代码位置：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\pool_worker.py:915-1013`

普通非 layerwise 加载位于：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\pool_worker.py:1037-1202`

主要步骤为：

1. 遍历请求需要加载的 group；
2. 根据 group block size 将 token hash 聚合为 group block hash；
3. 使用目标 block ID 计算目标 NPU 地址；
4. 构造 `key_list`、`addr_list`、`size_list`；
5. 调用 `self.m_store.get(key_list, addr_list, size_list)`。

地址及 key 构造代码：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\pool_worker.py:1103-1158`

实际 backend get 调用：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\pool_worker.py:1159-1202`

这条路径属于整段加载：一次准备所有 group 的目标地址，在 forward 前或异步阶段将外部数据直接写入本地 cache block。

## 九、Ascend Mooncake Layerwise Hybrid 加载

### 9.1 每个 Group 使用独立对象和 Session

多 group key 使用如下命名空间：

```text
model@mooncake_hybrid_v1:<layout-digest>@group:<id>@block:<size>@<hash>@<head>
```

相关实现和说明：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\backend\mooncake_layerwise.py:92-118`
- `D:\Code\vllm-ascend\vllm-ascend\docs\source\user_guide\feature_guide\mooncake_hybrid_attention.md:34-54`

layout digest 包含：

- TP size；
- group 顺序；
- group 成员层；
- 各 group page-size signature。

这样可以防止物理布局不兼容的模型实例错误地读取同一个远端对象。

### 9.2 为每个 Group 计算 Reachability Mask

Sliding Window、compressor state 一类 cache 不一定需要保存完整前缀。AscendStore coordinator 为每个 group 计算：

- `store_mask`：哪些逻辑 block 应持久化；
- `lookup_mask`：查池时允许查询哪些 block；
- `load_mask`：恢复共同前缀时实际需要加载哪些 block。

代码位置：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\coordinator.py:177-227`

外部命中的计算仍使用与上游相同的共同前缀思想：

1. 对各 group 只查询 mask 允许的 block；
2. 将命中结果放入虚拟 `ExternalCachedBlockPool`；
3. 运行 hybrid longest-hit 协调；
4. 得到所有 group 共同可恢复的 token 数。

代码位置：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\coordinator.py:229-271`

这里不能简单计算“各组连续 block 数的最小值”，因为 Sliding Window/state group 本来就可能稀疏保存，必须依据各自的 reachability mask 判断。

### 9.3 一个物理层可能同时属于多个 Group

worker 建立如下映射：

```text
physical_layer -> [(group_id, layer_idx_in_group), ...]
```

定义和构造位置：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\pool_worker.py:422-511`

例如：

| Group | 物理层 |
| --- | --- |
| Full KV | 0, 2 |
| Compressed KV | 1, 2, 3 |
| Window/state | 0, 3 |

则映射为：

```text
physical layer 0 -> Full + Window
physical layer 1 -> Compressed
physical layer 2 -> Full + Compressed
physical layer 3 -> Compressed + Window
```

所以同一个物理层入口可能需要提交或等待多个 group 的 transfer task。

`process_layer_data()` 会根据该映射先为所有物理层构造 save tasks，再构造 load tasks：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\pool_worker.py:2319-2367`

### 9.4 Group 内层索引与物理层索引的作用不同

一条 layerwise range 需要两个索引：

- `layer_idx_in_group`：计算远端对象内部该 group 的 byte offset；
- `physical_layer`：决定在哪个真实 transformer 层等待完成、记录事件。

因此不能简单地使用物理层号乘固定 stride。不同 group 的层集合可能不同。

具体 `LayerTransferTask` 构造：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\pool_worker.py:1302-1412`

### 9.5 加载与 Attention 逐层流水重叠

进入某个 attention 层之前，代码会调用：

```python
connector.wait_for_layer_load(layer_name)
```

入口位置：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\utils.py:445-456`

AscendStore connector 再调用 worker 的 `wait_for_layer_load()`：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\ascend_store_connector.py:277-290`

worker 的行为为：

1. 提交当前层和未来若干层的 load task；
2. 如果当前层需要恢复 cache，则等待当前层 event；
3. 当前 attention 开始后，利用其计算窗口提交未来层预取和当前层保存；
4. 下一层再等待对应层自己的 event。

核心代码：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\pool_worker.py:2487-2539`

时序可表示为：

```text
上一层计算 / collective
        ↓
记录 cache-ready NPU event
        ↓
提交当前层 PUT + 未来层 GET
        ↓
当前层 attention kernel
        ↓
output projection / MoE
        ↓
下一层 wait_for_layer_load
```

初次没有预取到的 demand load 会同步此前的 NPU 工作，以避免远端写入与仍在使用同一 cache buffer 的计算发生竞争：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\pool_worker.py:2499-2504`

### 9.6 加载失败采用 Fail-Closed

在 hybrid cache 中，一个逻辑 block 可能对应 SWA KV、compressed KV、compressor state 或其他辅助状态。如果只恢复一部分就继续运行 attention，可能产生静默错误。

因此 layerwise hybrid 检测到 load error 后会直接中止 forward，而不是让残缺状态进入 kernel：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\pool_worker.py:2369-2375`

## 十、Mamba/GDN 加载后的 State Copy

对普通 attention，只要远端数据写入正确的 KV block，kernel 就可以通过 block table 访问。

Mamba/GDN 运行时通常还维护：

- 当前请求使用哪个 state slot；
- 当前活动的 conv state；
- 当前活动的 SSM/GDN state。

因此 layerwise 加载完成后，可能还需要把恢复出的边界状态复制到当前执行 slot。正确顺序为：

```text
完成该层远端 load
        ↓
执行该层 Mamba state copy
        ↓
允许该层计算
```

否则 state copy 可能读到只写入一部分的状态。

相关代码：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\ascend_store_connector.py:277-313`

需要特别区分：当前 `mooncake_hybrid_attention.md` 描述的是 DeepSeek-V4 风格的多 attention-cache-group layerwise range 实现，并明确拒绝 recurrent Mamba state：

- `D:\Code\vllm-ascend\vllm-ascend\docs\source\user_guide\feature_guide\mooncake_hybrid_attention.md:28-32`

所以：

- vLLM 通用 hybrid KV manager 支持 Full Attention 与 Mamba/GDN 的组合；
- AscendStore 的其他 hybrid/Mamba 路径存在 Mamba state copy 支持；
- 这份特定 Mooncake layerwise multi-group range 实现面向 DeepSeek-V4 类 attention cache，不能仅据此认定其支持 Qwen3.5 recurrent GDN state。

## 十一、完整示例

假设模型有三个 group：

```text
G0: Full Attention，block_size=16
G1: Sliding Window，block_size=16
G2: Compressor State，block_size=32
```

请求 prompt 长度为 160 token，外部池查询结果为：

```text
G0: 0..9 block 都存在              -> 可恢复到 160
G1: 按 reachability mask 所需块存在  -> 可恢复到 144
G2: 0..3 存在，第 4 块不存在         -> 可恢复到 128
```

共同恢复边界为：

```text
L = 128
```

随后执行：

1. Scheduler 将前 128 token 视为外部已计算；
2. 为 G0、G1、G2 分别分配覆盖 128 token 的本地 block；
3. connector metadata 记录各 group 的目标 block IDs；
4. worker 按 group 计算本地地址；
5. layerwise 模式按物理层生成 range task；
6. 进入 Layer 0 前，等待 Layer 0 所需的 G0/G1 range；
7. Layer 0 attention 启动后，预取 Layer 1/2；
8. 某个 group 到达自身最后一个 group layer 时，可以独立提交和关闭；
9. token 128 之后的内容正常重新计算并生成新 cache。

系统不是“把 160 token 中找到的所有数据尽量装入本地”，而是：

> 只采用能够组成一致模型状态的 128-token 快照；更深但不完整的单组命中不能直接使用。

## 十二、最终理解

可以将整个系统分为三层：

| 层次 | 决定内容 |
| --- | --- |
| 模型 `layer_types` | 当前层执行 Full、SWA 还是 GDN/Mamba |
| `KVCacheSpec / kv_cache_groups` | 状态如何分页、如何查询命中、如何回收 block |
| Connector / AscendStore | 外部对象如何映射到本地 group、物理层和 NPU 地址 |

因此，混合注意力的 KV cache 加载不是一个单独的 `load_kv()` 可以完整描述的过程，而是：

```text
模型语义分层
→ cache spec 分组
→ 多组共同命中协调
→ 本地 block 分配
→ 外部字节加载
→ 逐层同步
→ block table / recurrent state 恢复
→ forward
```

## 十三、验证边界

本文结论来自当前两个工作区的静态源码分析，没有运行真实 NPU/Mooncake 服务。因此能够确认：

- Python 控制流；
- KV cache 数据结构；
- group/block table 的组织方式；
- Connector metadata 和地址映射意图；
- layerwise wait/prefetch 的代码顺序。

但不能仅凭静态分析确认：

- 当前分支已在某个具体混合模型上运行通过；
- 所有 NPU stream/event 时序都与设计一致；
- 真实 Mooncake 服务上的 range/session 接口完全兼容；
- layerwise pipeline 一定带来性能收益。

这些结论仍需通过真实模型、NPU trace、Mooncake 日志、冷/热 cache 输出一致性和性能基准进一步验证。

---

## 十四、Sliding Window Attention 在哪些模型中使用

### 14.1 原问题

> 请你结合 `D:\Code\vllm-ascend\vllm-ascend` 和 `D:\Code\vllm-github` 的代码内容，继续回答以下问题。你在回答问题的时候可以辅助进行一些网络知识的搜索，请尽量详细回答，并配合适当举例和图表。我的问题如下：  
> 基于上面的回答，我有以下问题：1. Sliding Window Attention 在哪些模型中使用？

### 14.2 结论

Sliding Window Attention（SWA，滑动窗口注意力）并不只用于 Mistral。结合当前两个仓库，典型模型包括：

- Mistral 7B、Mistral Instruct、Ministral；
- Gemma 2、Gemma 3、Gemma 4；
- OpenAI gpt-oss-20b、gpt-oss-120b；
- Cohere Command R/Cohere2 的部分版本；
- EXAONE 4 的部分规格；
- GraniteSWA、GraniteMoeSWA；
- Apertus、Jais2、PLaMo3、MiniMax-M2/M3 等较新的模型；
- Ascend 中还有 DeepSeek-V4/V4.1 的 DSA 窗口化路径。

但“配置或代码中出现 `sliding_window`”并不自动等价于“这个 checkpoint 正在使用 SWA”。必须至少结合以下三层判断：

1. **模型配置层**：具体 checkpoint 的 `sliding_window` 是否非空，哪些层的 `layer_types` 是 `sliding_attention`；
2. **vLLM 模型实现层**：模型是否真的把窗口传入 `Attention(per_layer_sliding_window=...)`；
3. **Cache/后端层**：vLLM 是否为该层生成 `SlidingWindowSpec`，后端是否支持并实际执行窗口化读取。

本节分析对应的本地源码版本为：

- 上游 vLLM：`main@027b6f3a22f40b057474f053278e6323d81462b9`；
- vLLM-Ascend：`l2_928@d007e4205d93fd746d4bc61ed19ed3761044a73f`。

以下属于静态源码分析，不代表已经对每个模型 checkpoint 做过运行验证。

### 14.3 SWA 的基本语义

对于位置为 `t` 的 token，Full Attention 可以直接读取：

```text
[0, t]
```

窗口大小为 `W` 的因果 Sliding Window Attention 只能直接读取：

```text
[max(0, t-W+1), t]
```

例如 `W=4`：

| 当前 token | Full Attention 可见 token | SWA 可见 token |
| ---: | --- | --- |
| 2 | 0、1、2 | 0、1、2 |
| 5 | 0～5 | 2、3、4、5 |
| 9 | 0～9 | 6、7、8、9 |

其直接效果是：

- Full Attention 需要访问从序列开头到当前 token 的历史 K/V；
- SWA 执行当前层时，主要访问最近 `W` 个 token 的 K/V；
- 多层 SWA 的有效感受野仍会逐层扩大，因为上一层的 token 表示已经融合过它自己的历史窗口。

所以“窗口为 4096”并不代表深层输出完全不含 4096 token 以前的信息；它表示单层注意力的直接 K/V 访问范围受限。

### 14.4 vLLM 如何判定某一层是 SWA

vLLM 中的主要链路如下：

```text
checkpoint config.sliding_window / config.layer_types
                    │
                    ▼
模型层创建 Attention(
    per_layer_sliding_window=...
)
                    │
                    ▼
Attention.get_kv_cache_spec()
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 window 非空            window 为空
SlidingWindowSpec       FullAttentionSpec
```

关键代码位置：

- `D:\Code\vllm-github\vllm\model_executor\layers\attention\attention.py:252`：`Attention` 接收 `per_layer_sliding_window`；
- `D:\Code\vllm-github\vllm\model_executor\layers\attention\attention.py:265-272`：选择逐层窗口或模型级窗口；
- `D:\Code\vllm-github\vllm\model_executor\layers\attention\attention.py:339`：保存为 `self.sliding_window`；
- `D:\Code\vllm-github\vllm\model_executor\layers\attention\attention.py:606-668`：`get_kv_cache_spec()` 根据窗口生成 `SlidingWindowSpec` 或 `FullAttentionSpec`。

因此，判断某个具体 checkpoint 是否真正使用 SWA，至少需要确认：

1. 配置中存在非空的 `sliding_window`；
2. 当前层在 `layer_types` 中标为 `sliding_attention`，或者架构约定所有层都使用 SWA；
3. 模型实现把窗口传给 `Attention`；
4. 最终为这些层创建了 `SlidingWindowSpec`。

### 14.5 Mistral 7B、Mistral Instruct 和 Ministral

Mistral 7B 是最典型的 SWA 模型之一。原始 Mistral 7B 使用约 4096 token 的滑动窗口，所有 Transformer 层都按窗口注意力运行。官方说明参见：

- <https://mistral.ai/news/announcing-mistral-7b>

当前 vLLM 会对 Mistral 配置做规范化：

- 如果 `sliding_window` 是整数且没有显式 `layer_types`，则所有层被设为 `sliding_attention`；
- 如果窗口配置是逐层列表，则可按层决定 Full Attention 或 SWA。

相关代码：

- `D:\Code\vllm-github\vllm\transformers_utils\configs\mistral.py:176-194`；
- `D:\Code\vllm-github\vllm\model_executor\models\mistral.py:77`：MistralAttention 继承 LlamaAttention；
- `D:\Code\vllm-github\vllm\model_executor\models\llama.py:185-219`：根据层类型传递 `per_layer_sliding_window`。

典型结构：

```text
Mistral 7B

Layer 0   SWA, W=4096
Layer 1   SWA, W=4096
Layer 2   SWA, W=4096
...
Layer 31  SWA, W=4096
```

需要注意，当前仓库的 Mixtral 实现在 Attention 构造处没有明确传入 `per_layer_sliding_window`。因此不能只因 Mixtral 属于 Mistral 系，或者配置可能含有同名字段，就断言当前实现必然生成 `SlidingWindowSpec`。

### 14.6 Gemma 2

Gemma 2 使用交替的局部滑动窗口层和全局注意力层。Google 的架构说明参见：

- <https://developers.googleblog.com/en/gemma-explained-new-in-gemma-2/>

结构可简化为：

```text
Layer 0   Sliding Window
Layer 1   Full Attention
Layer 2   Sliding Window
Layer 3   Full Attention
...
```

vLLM 的混合 KV Cache 设计文档把 Gemma 2 概括为约 `1 SW : 1 Full`：

- `D:\Code\vllm-github\docs\design\hybrid_kv_cache_manager.md:77-80`。

模型实现会检查当前层类型，只对滑动窗口层传入窗口：

- `D:\Code\vllm-github\vllm\model_executor\models\gemma2.py:154-166`。

因此 Gemma 2 是很典型的 Hybrid Attention 模型：

```text
同一个模型
├── 一部分层：SlidingWindowSpec
└── 一部分层：FullAttentionSpec
```

其 KV Cache 不能被当作“整个模型只有一种 Cache 类型”。

### 14.7 Gemma 3

Gemma 3 进一步提高了局部窗口层的比例。Google 官方介绍给出的典型周期是 5 个 Local SWA 层后接 1 个 Global Full Attention 层，局部窗口通常为 1024 token：

- <https://developers.googleblog.com/en/gemma-explained-whats-new-in-gemma-3/>

```text
Gemma 3 一个周期

L0  SWA, W=1024 ─┐
L1  SWA, W=1024  │
L2  SWA, W=1024  ├── 局部建模
L3  SWA, W=1024  │
L4  SWA, W=1024 ─┘
L5  Full Attention ── 全局信息汇聚
```

vLLM 对应实现：

- `D:\Code\vllm-github\vllm\model_executor\models\gemma3.py:152-198`。

在 vLLM 的混合 KV Cache 示例中，Gemma 3 27B 被描述为：

- 52 个 SWA 层；
- 10 个 Full Attention 层。

位置：

- `D:\Code\vllm-github\docs\design\hybrid_kv_cache_manager.md:113-121`。

Gemma 1 主要使用全局注意力，不能因为名字相近就归为 SWA 模型。

### 14.8 Gemma 4

当前 vLLM 已包含 Gemma 4 的逐层 SWA 支持：

- `D:\Code\vllm-github\vllm\model_executor\models\gemma4.py:449-518`。

代码会读取每层类型、决定窗口，并把窗口传入 Attention。多模态场景中还会对图像前缀相关窗口做额外处理。

从代码能够确认的是“架构实现支持逐层 Full/SWA”；某个具体 Gemma 4 checkpoint 有多少 SWA 层，仍应以它自己的 `config.json` 为准。

### 14.9 OpenAI gpt-oss-20b/120b

gpt-oss 采用交替的全局注意力和局部带状稀疏注意力。官方资料：

- <https://openai.com/index/introducing-gpt-oss/>
- <https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf>

vLLM 把其 locally banded sparse attention 映射成 SWA：偶数层设置 `config.sliding_window`，奇数层使用 Full Attention。

- `D:\Code\vllm-github\vllm\model_executor\models\gpt_oss.py:170-182`。

结构大致为：

```text
L0  SWA, W=128
L1  Full
L2  SWA, W=128
L3  Full
...
```

它和 Gemma 2 都是 Full+SWA 混合模型，但窗口大小和层比例由各自架构决定。

### 14.10 Cohere Command R / Cohere2

当前 vLLM 的 Command R 实现区分：

- v1 模型不采用这里的交错 SWA 路径；
- v2 模型根据 `layer_types` 设置交错 SWA。

代码位置：

- `D:\Code\vllm-github\vllm\model_executor\models\commandr.py:233-253`。

Cohere2 MoE 也有逐层窗口支持：

- `D:\Code\vllm-github\vllm\model_executor\models\cohere2_moe.py:207-233`。

所以准确说法是：Cohere/Command R 的部分新版本支持或使用交错 SWA，不能把所有 Command R checkpoint 一概视为 SWA 模型。

### 14.11 EXAONE 4

EXAONE 4 按 `layer_types` 决定当前层使用局部还是全局注意力：

- `D:\Code\vllm-github\vllm\model_executor\models\exaone4.py:171-192`。

例如官方 EXAONE 4.0 32B 配置包含 `sliding_window=4096`，并通过局部层和全局层组成重复模式：

- <https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B/blob/main/config.json>
- <https://huggingface.co/docs/transformers/en/model_doc/exaone4>

但部分较小 checkpoint 可能把 `sliding_window` 设为空。因此：

```text
架构代码支持 SWA
≠
该架构的每一个 checkpoint 都启用了 SWA
```

### 14.12 Granite SWA 系列

vLLM 注册了名称明确的两个 SWA 架构：

- `GraniteSWAForCausalLM`；
- `GraniteMoeSWAForCausalLM`。

注册位置：

- `D:\Code\vllm-github\vllm\model_executor\models\registry.py:127-132`。

Granite 实现会解析 `layer_types` 并逐层传入窗口：

- `D:\Code\vllm-github\vllm\model_executor\models\granite.py:77-104`；
- `D:\Code\vllm-github\vllm\model_executor\models\granite.py:200-226`。

仓库测试还包含 `full_attention、sliding_attention、sliding_attention` 这样的配置：

- `D:\Code\vllm-github\tests\models\language\generation\test_granite.py:67-73`。

需要区分普通 Granite 与名称中明确带 SWA 的 Granite 变体。

### 14.13 当前 vLLM 中其他显式 SWA 实现

按“模型实现明确向 Attention 传递 `per_layer_sliding_window`”这一标准，当前仓库还包括：

| 分类 | 模型或实现 |
| --- | --- |
| 主流混合注意力 | Gemma 2/3/3n/4、gpt-oss、Command R/Cohere2 |
| Mistral 路径 | Mistral、Ministral、部分复用 LlamaAttention 的窗口化模型 |
| IBM/企业模型 | GraniteSWA、GraniteMoeSWA |
| 新模型 | Apertus、Jais2、Laguna、Mellum、MiMo-V2、MiniMax-M2 |
| MoE/新架构 | AFMoE、EXAONE MoE、Cohere Compass、MuseGlimmer |
| 其他 | Nemotron-H、OpenPangu、PLaMo3、RNJ1、Step3.5 |

代表性位置：

- Apertus：`D:\Code\vllm-github\vllm\model_executor\models\apertus.py:180-199`；
- Jais2：`D:\Code\vllm-github\vllm\model_executor\models\jais2.py:188`；
- PLaMo3：`D:\Code\vllm-github\vllm\model_executor\models\plamo3.py:205`；
- MiniMax-M2：`D:\Code\vllm-github\vllm\model_executor\models\minimax_m2.py:209`；
- Nemotron-H：`D:\Code\vllm-github\vllm\model_executor\models\nemotron_h.py:489`；
- OpenPangu：`D:\Code\vllm-github\vllm\model_executor\models\openpangu.py:485`；
- Step3.5：`D:\Code\vllm-github\vllm\model_executor\models\step3p5.py:247`。

这张表表示当前代码有显式 SWA 实现，不表示每个架构的所有公开 checkpoint 都默认启用 SWA。

### 14.14 vLLM-Ascend 如何执行 SWA

Ascend 支持矩阵中明确列出的典型 SWA 模型包括：

- Gemma 2；
- Gemma 3；
- Mistral；
- 多模态 Mistral 3/Gemma 3。

位置：

- `D:\Code\vllm-ascend\vllm-ascend\docs\source\user_guide\support_matrix\supported_models.md:78-81`；
- `D:\Code\vllm-ascend\vllm-ascend\docs\source\user_guide\support_matrix\supported_models.md:173-175`。

upstream vLLM 负责识别模型层类型并生成 `SlidingWindowSpec`，Ascend 后端再把窗口语义落实为 NPU Attention 参数：

```text
vLLM SlidingWindowSpec
          │
          ▼
Ascend Attention metadata
          │
          ├── sparse_mode = 4
          └── pre_tokens = sliding_window
          │
          ▼
NPU Attention 只读取窗口范围内的历史 KV
```

关键位置：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\attention_v1.py:605-607`。

当前 Ascend FA3 后端不支持 SWA，需要 SWA 的模型应使用默认 FIA 路径：

- `D:\Code\vllm-ascend\vllm-ascend\docs\source\user_guide\feature_guide\flash_attention.md:24-43`；
- `D:\Code\vllm-ascend\vllm-ascend\docs\source\user_guide\feature_guide\flash_attention.md:139-143`。

### 14.15 DeepSeek-V4/V4.1 的特殊窗口化路径

Ascend 仓库中的 DeepSeek-V4/V4.1 也使用窗口概念：

- DeepSeek-V4 读取 `config.sliding_window`：`D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v4\model.py:487`；
- DeepSeek-V4.1：`D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:377`；
- DSA 设置窗口 mask：`D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v1.py:950-955`；
- V4.1 DSA：`D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:824-853`。

但它不能被简单等同于 Gemma 2 的“部分 `SlidingWindowSpec` + 部分 `FullAttentionSpec`”。DeepSeek-V4/V4.1 的窗口化发生在 DSA、压缩 KV、索引选择等专用架构逻辑中，KV 表达和加载路径可能不同，应单独分析。

### 14.16 容易被误认为 SWA、但实际上不是的情况

#### 14.16.1 Qwen3.5/Qwen3-Next 的 GDN

Qwen3.5/Qwen3-Next 主要是：

```text
GDN/Linear Attention 层
+
Full Attention 层
```

而不是 Sliding Window Attention。其层类型是 `linear_attention` 与 `full_attention`：

- `D:\Code\vllm-github\vllm\model_executor\models\qwen3_next.py:491-506`；
- `D:\Code\vllm-github\vllm\model_executor\models\qwen3_next.py:568-574`；
- `D:\Code\vllm-github\vllm\model_executor\models\qwen3_5.py:146-160`。

GDN 保存递归状态，不是“只保存或读取最近 `W` 个普通 K/V”。

#### 14.16.2 Llama 4 Chunked Local Attention

Llama 4 的局部注意力在 vLLM 中有独立类型 `CHUNKED_LOCAL_ATTENTION`：

- `D:\Code\vllm-github\vllm\v1\kv_cache_interface.py:139-144`。

它与普通 causal SWA 的 mask、block 边界和 KV Cache 语义不完全相同。vLLM 设计文档也将 Llama 4 的 Chunked Local Attention 与 SWA 分开分类：

- `D:\Code\vllm-github\docs\design\hybrid_kv_cache_manager.md:8-12`。

#### 14.16.3 Jamba/Bamba 的 Mamba 层

Jamba/Bamba 属于 Mamba/SSM 层与 Full Attention 层的混合。Mamba state 不是窗口化 K/V，不能归入 SWA。

#### 14.16.4 Speculative Decoding 的 draft KV sliding window

Ascend 还存在推测解码 Draft KV Cache 滑窗功能：

- `D:\Code\vllm-ascend\vllm-ascend\docs\source\developer_guide\Design_Documents\speculative_kv_sliding_window.md:1-30`。

它支持 EAGLE3、DFlash、DSpark 等 draft model，但这是推理优化：

```text
Target 模型 KV Cache：不变
Draft 模型 KV Cache：Attention 只读取最近一段 KV
```

这不表示目标模型原生采用 SWA。文档还说明只有 FIA 的读取窗口被限制，`slot_mapping` 仍保持正常写入：

- `D:\Code\vllm-ascend\vllm-ascend\docs\source\developer_guide\Design_Documents\speculative_kv_sliding_window.md:61-67`。

#### 14.16.5 视觉模型的 Window Attention

视觉编码器也常出现 `window_size`，但它限制的是图像 patch 的二维空间邻域，不是自回归文本 token 的历史 KV Cache，因此不能与 causal SWA 混为一谈。

### 14.17 SWA 与混合 KV Cache 的对应关系

| 模型类型 | 层结构示例 | KV Cache 类型 |
| --- | --- | --- |
| 纯 Full | 标准 Llama 2/3 | 全部 `FullAttentionSpec` |
| 纯 SWA | Mistral 7B | 全部 `SlidingWindowSpec` |
| Full+SWA 混合 | Gemma 2/3、gpt-oss | 多个 Full/SWA KV Cache Group |
| SSM/GDN+Full | Jamba、Qwen3.5 | SSM/GDN state + Full KV |
| Chunked Local+Full | Llama 4 | ChunkedLocal Spec + Full Spec |

vLLM 的混合 KV Cache 设计文档明确区分：

- SWA+Full：gpt-oss、Gemma 2/3、Ministral、Cohere；
- Mamba+Full：另一类混合 Cache；
- Chunked Local+Full：Llama 4。

位置：

- `D:\Code\vllm-github\docs\design\hybrid_kv_cache_manager.md:8-21`。

SWA 的 prefix cache 搜索也与 Full Attention 不同。由于窗口层主要关心尾部一段 block，vLLM 会从请求右端向左检查可复用 block：

- `D:\Code\vllm-github\vllm\v1\core\single_type_kv_cache_manager.py:946-1037`。

但对于 Full+SWA 混合模型，最终能跳过计算的 prefix 仍然必须是所有 KV Cache Group 都能接受的共同前缀。不能因为 SWA 层只需要最近窗口，就忽略 Full Attention 层对完整连续前缀的要求。

### 14.18 如何检查一个具体 checkpoint

可以直接读取模型配置：

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "模型目录或模型名称",
    trust_remote_code=True,
)

text_config = getattr(config, "text_config", config)

print("sliding_window =", getattr(text_config, "sliding_window", None))
print("layer_types =", getattr(text_config, "layer_types", None))
```

判断方式：

```text
sliding_window = None
    → 通常没有启用原生 SWA

sliding_window = 4096
layer_types 全部是 sliding_attention
    → 所有层使用 SWA

sliding_window = 1024
layer_types 同时有 sliding_attention/full_attention
    → Hybrid Full+SWA

配置中有 sliding_window，
但模型构造 Attention 时没有传入窗口
    → 不能证明当前 vLLM 实现真正使用 SWA
```

最终，针对“Sliding Window Attention 在哪些模型中使用”，更准确的回答是：

> Mistral/Ministral、Gemma 2/3/4、gpt-oss、部分 Cohere/Command R、EXAONE 4、GraniteSWA，以及若干新模型架构明确支持或使用 SWA；但是否真正启用、窗口多大、哪些层启用，最终应以具体 checkpoint 的 `sliding_window`、`layer_types`、vLLM 模型构造逻辑和后端能力为准。
