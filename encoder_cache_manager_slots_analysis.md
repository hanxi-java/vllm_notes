# EncoderCacheManager 中 num_free_slots 与 num_freeable_slots 分析

## 一、字段含义

| 字段 | 含义 |
|------|------|
| `cache_size` | 缓存总容量，以 **encoder embeddings 数量** 为单位 |
| `num_free_slots` | **当前可用槽位**，= `cache_size - 已分配且仍被引用的大小` |
| `num_freeable_slots` | **可回收槽位**，= `num_free_slots + 已引用数为 0 但仍占着内存的大小` |

关键区别：

- `num_free_slots`：现在立刻就能用，无需驱逐。
- `num_freeable_slots`：现在可用 + 驱逐掉无人引用的条目后可用。

它们的关系恒有：

```
num_freeable_slots >= num_free_slots
```

## 二、状态变化场景

用一个 `cache_size = 100` 的例子来演示。

### 场景 1：初始状态

```
cache_size = 100
num_free_slots = 100
num_freeable_slots = 100

cached = {}        # mm_hash → {request_ids}
freeable = {}      # mm_hash → num_encoder_embeds (LRU，无人引用)
freed = []         # 被驱逐的 mm_hash，通知 worker
```

画图：

```
┌─────────────────────────────────────────┐
│  Cache Capacity: 100 embeddings         │
│  ┌─────────────────────────────────┐    │
│  │ Free slots: 100                 │    │
│  │                                 │    │
│  │                                 │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│ num_free_slots = 100                    │
│ num_freeable_slots = 100                │
└─────────────────────────────────────────┘
```

---

### 场景 2：Request A 缓存了一个 image（30 embeds）

调用路径：`can_allocate()` → `allocate()`

```
can_allocate(num_embeds=30):
  30 <= num_free_slots(100) → True

allocate(num_embeds=30):
  num_free_slots = 100 - 30 = 70
  num_freeable_slots = 100 - 30 = 70
```

`num_freeable_slots` 为什么也减 30？

因为“可回收”=“真正可用 + 没人要的缓存”。新分配的数据现在有人引用，所以既不可用也不可回收。

画图：

```
┌─────────────────────────────────────────┐
│  Cache Capacity: 100 embeddings         │
│  ┌─────────────────────────────────┐    │
│  │ Request A: image#1 (30)         │ ← 被引用，不能驱逐 │
│  │                                 │    │
│  │ Free slots: 70                  │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│ num_free_slots = 70                     │
│ num_freeable_slots = 70                 │
└─────────────────────────────────────────┘
```

---

### 场景 3：Request A 结束，调用 `free()`

```
free_encoder_input(input_id=0):
  cached[image#1] 从 {A} 变成 {}
  把 image#1 加入 freeable: 30
  num_freeable_slots = 70 + 30 = 100
  num_free_slots 不变，还是 70
```

注意：**物理内存没释放**，只是变成“可驱逐”。

画图：

```
┌─────────────────────────────────────────┐
│  Cache Capacity: 100 embeddings         │
│  ┌─────────────────────────────────┐    │
│  │ image#1 (30)                    │ ← 无人引用，可驱逐 │
│  │     [in freeable LRU]           │    │
│  │                                 │    │
│  │ Free slots: 70                  │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
│ num_free_slots = 70                     │
│ num_freeable_slots = 100                │
│  ↑ 因为 70 + 30(可驱逐) = 100           │
└─────────────────────────────────────────┘
```

---

### 场景 4：新 Request B 想用 50 embeds，不触发驱逐

```
can_allocate(num_embeds=50):
  50 <= num_free_slots(70)? 是
  → 返回 True，无需驱逐

allocate(num_embeds=50):
  num_free_slots = 70 - 50 = 20
  num_freeable_slots = 100 - 50 = 50
```

注意：`num_freeable_slots` 直接减 50，而不是先把 freeable 的 30 挪走再减。因为 image#1 仍然无人引用，还在 `freeable` 里；新分配的 50 是被引用的。

画图：

```
┌─────────────────────────────────────────┐
│  Cache Capacity: 100 embeddings         │
│  ┌─────────────────────────────────┐    │
│  │ Request B: data#2 (50)          │ ← 被引用，不能驱逐 │
│  │ image#1 (30)                    │ ← 无人引用，可驱逐 │
│  │     [in freeable LRU]           │    │
│  │ Free slots: 20                  │    │
│  └─────────────────────────────────┘    │
│                                         │
│ num_free_slots = 20                     │
│ num_freeable_slots = 50                 │
│  ↑ 20 + 30 = 50                         │
└─────────────────────────────────────────┘
```

---

### 场景 5：新 Request C 想用 80 embeds，触发驱逐

```
can_allocate(num_embeds=80):
  80 > num_free_slots(20)? 是
  80 <= num_freeable_slots(50)? 否
  → 返回 False，无法分配
```

那如果 Request C 想用 **40 embeds**：

```
can_allocate(num_embeds=40):
  40 > num_free_slots(20)? 是
  40 <= num_freeable_slots(50)? 是
  → 驱逐 freeable 中最老的条目 image#1 (30)
     num_free_slots = 20 + 30 = 50
     freed.append(image#1)
  → 返回 True

allocate(num_embeds=40):
  num_free_slots = 50 - 40 = 10
  num_freeable_slots = 50 - 40 = 10
```

画图：

```
Before can_allocate:
┌─────────────────────────────────────────┐
│  data#2 (50, 被引用) + image#1 (30, 可驱逐) + Free 20 │
│  num_free=20, num_freeable=50           │
└─────────────────────────────────────────┘

After eviction + allocate:
┌─────────────────────────────────────────┐
│  Request B: data#2 (50)                 │ ← 被引用 │
│  Request C: data#3 (40)                 │ ← 被引用 │
│  Free slots: 10                         │         │
│                                         │         │
│  image#1 → freed[] (通知 worker 释放)   │         │
│                                         │         │
│  num_free_slots = 10                    │         │
│  num_freeable_slots = 10                │         │
└─────────────────────────────────────────┘
```

---

### 场景 6：缓存命中 `check_and_update_cache()`

假设 Request D 也来了，它的 image#1 和 A 相同（hash 命中），但此时 image#1 还没被驱逐，仍在 `freeable` 中。

```
check_and_update_cache(image#1):
  image#1 在 cached 中，但 cached[image#1] == {}
  从 freeable 弹出 image#1
  num_freeable_slots = 50 - 30 = 20
  cached[image#1].add(D)
  request_cached_ids[D].add(0)
  return True
```

注意：`num_free_slots` 不变，因为这条数据本来就占着物理空间，现在只是重新被引用。

画图：

```
Before:
┌─────────────────────────────────────────┐
│  data#2 (50) + data#3 (40) + image#1 (30, 可驱逐) + Free 10 │
│  num_free=10, num_freeable=40           │
│  实际总占用：50+40+30 = 120 > cache_size? 不可能，说明此状态不成立 │
└─────────────────────────────────────────┘
```

> 说明：上面的状态不成立，因为 cache_size=100。更合理的场景是：image#1 之前被 Request A 使用，A 结束后进入 `freeable`，此时 cache 中可能还有其他请求的数据。

重新举例：假设 Request A 只用了 image#1 (30)，A 结束后 cache 里只有 image#1 在 `freeable` 中。

```
状态：
num_free_slots = 70
num_freeable_slots = 100
freeable = {image#1: 30}
```

Request D 的 image#1 命中：

```
check_and_update_cache(image#1):
  从 freeable 弹出 image#1
  num_freeable_slots = 100 - 30 = 70
  cached[image#1].add(D)
  return True
```

画图：

```
Before:
┌─────────────────────────────────────────┐
│  image#1 (30, 可驱逐)                   │
│  Free: 70                               │
│  num_free=70, num_freeable=100          │
└─────────────────────────────────────────┘

After cache hit:
┌─────────────────────────────────────────┐
│  Request D: image#1 (30)                │ ← 重新被引用 │
│  Free: 70                               │              │
│  num_free=70, num_freeable=70           │              │
│  ↑ 可驱逐部分回到被引用状态              │              │
└─────────────────────────────────────────┘
```

---

## 三、变化规则总结

| 操作 | `num_free_slots` | `num_freeable_slots` | 说明 |
|------|------------------|----------------------|------|
| 初始化 | = cache_size | = cache_size | 全部空闲 |
| `can_allocate()` 足够 | 不变 | 不变 | 只检查 |
| `can_allocate()` 不够，触发驱逐 | +被驱逐大小 | 不变 | 被驱逐条目从 `freeable` 移入 `freed` |
| `allocate()` | -num_embeds | -num_embeds | 新数据被引用，既不可用也不可回收 |
| `free_encoder_input()` 取消引用 | 不变 | +num_embeds | 数据加入 `freeable` |
| `check_and_update_cache()` 命中无人引用条目 | 不变 | -num_embeds | 从 `freeable` 取出，重新被引用 |
| `reset()` | = cache_size | = cache_size | 清空一切 |

---

## 四、核心不变式

```
num_free_slots <= num_freeable_slots <= cache_size
```

以及：

```
num_freeable_slots = num_free_slots + sum(freeable.values())
```

也就是说，`num_free_slots` 反映的是**物理上完全空闲的容量**，而 `num_freeable_slots` 反映的是**逻辑上还能再塞多少**（包括可以驱逐的）。这个设计让 vLLM 能够延迟释放物理内存，直到真正需要空间时才通知 worker 驱逐。

## 五、源码对应

相关代码位于：

- `vllm/v1/core/encoder_cache_manager.py:68-80` — 初始化
- `vllm/v1/core/encoder_cache_manager.py:95-122` — `check_and_update_cache`
- `vllm/v1/core/encoder_cache_manager.py:124-183` — `can_allocate`（含驱逐逻辑）
- `vllm/v1/core/encoder_cache_manager.py:185-211` — `allocate`
- `vllm/v1/core/encoder_cache_manager.py:217-242` — `free_encoder_input`
- `vllm/v1/core/encoder_cache_manager.py:244-254` — `free`
- `vllm/v1/core/encoder_cache_manager.py:82-93` — `reset`
