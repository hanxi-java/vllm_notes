# ScoreEncoderCacheManager 详解

`ScoreEncoderCacheManager` 位于 `vllm/v1/core/encoder_cache_manager.py`，是 `EncoderCacheManager` 的扩展版。它引入了 **GPU + CPU 两级缓存**，并通过 **score（访问频率 + 时钟 + 计算代价）** 决定哪些多模态 encoder embedding 放在 GPU、哪些放在 CPU。

---

## 1. 与基础 `EncoderCacheManager` 的区别

| 特性 | `EncoderCacheManager` | `ScoreEncoderCacheManager` |
|---|---|---|
| 缓存层级 | 单级 | GPU + CPU 两级 |
| 新 embedding 去向 | 直接放入缓存 | 先放入 CPU cache |
| 命中后行为 | 直接复用 | CPU 命中可能触发 promote 到 GPU |
| 淘汰策略 | FIFO / LRU | 基于 score 淘汰 |
| 冷热识别 | 无 | clock 衰减 + freq + cal_cost |
| 物理存储 | manager 只记账 | 实际 tensor 由 worker/gpu_model_runner 管理 |

---

## 2. 配置来源

配置通过 `vllm_config.additional_config["score_encoder_cache_config"]` 传入，由 `vllm/config/score_encoder_cache.py` 解析。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `enabled` | `False` | 是否启用 |
| `cpu_cache_slots` | ~10GB | CPU cache 容量（以 encoder embeddings 数量计） |
| `max_clock` | 15 | 时钟最大值 |
| `clock_decay_every` | 64 | 每多少次请求衰减一次 clock |
| `watermark` | 0.2 | 淘汰后水位线 |
| `promote_percentile` | 0.2 | 晋升到 GPU 的 score 百分位阈值。0.2表示前20%得分的才能够晋升 |

启用方式：

```python
{"score_encoder_cache_config": {"enabled": True, "cpu_cache_slots": 1000000}}
```

Scheduler 中根据 `enabled` 决定是否替换 manager：

```python
# vllm/v1/core/sched/scheduler.py:218-220
if get_score_encoder_cache_config(vllm_config).enabled:
    self.encoder_cache_manager = ScoreEncoderCacheManager(
        cache_size=encoder_cache_size, vllm_config=self.vllm_config
    )
```

---

## 3. 两级缓存结构

```python
# GPU cache
self.cache_size = cache_size                    # GPU 总容量
self.gpu_num_free_slots = cache_size            # GPU 空闲槽位
self.gpu_num_freeable_slots = cache_size        # GPU 可回收槽位（空闲 + 无引用）
self.gpu_cache: Dict[str, CacheEntry] = {}      # GPU 中的条目
self.gpu_freeable: Dict[str, CacheEntry] = {}   # GPU 中可被回收的条目

# CPU cache
self.cpu_cache_size = score_encoder_cache_config.cpu_cache_slots
self.cpu_num_free_slots = self.cpu_cache_size
self.cpu_num_freeable_slots = self.cpu_cache_size
self.cpu_cache: Dict[str, CacheEntry] = {}      # CPU 中的条目
self.cpu_freeable: OrderedDict[str, CacheEntry] = OrderedDict()  # CPU 可回收，FIFO
```

`CacheEntry` 是元数据：

```python
@dataclass
class CacheEntry:
    mm_hash: str
    freq: int           # 访问次数
    clock: int          # 时钟值，最近访问时重置为 max_clock
    num_embeds: int     # 占多少 slots
    cal_cost: int       # 理论重计算代价
```

实际的 embedding tensor 由 worker 端维护，manager 只负责**调度决策**。

---

## 4. 核心流程

### 4.1 新请求到达：`check_and_update_cache()`

位置：`vllm/v1/core/encoder_cache_manager.py:522-571`

逻辑：

1. 如果 `mm_hash` 从未缓存 → **miss**，返回 `False`，需要计算。
2. 如果缓存过但无请求引用 → 先从 `freeable` 中取出。
3. 如果已经在 GPU cache → 直接命中。
4. 如果在 CPU cache：
   - 调用 `should_promote()`，满足条件则**晋升到 GPU**。
   - 否则留在 CPU，后续需要从 CPU 加载 embedding。
5. 更新 `freq` 和 `clock`。

```python
def check_and_update_cache(self, request: Request, input_id: int) -> bool:
    mm_hash = request.mm_features[input_id].identifier

    if mm_hash not in self.cached:
        self.on_request()
        return False
    # 有缓存，但引用列表为空
    if not self.cached[mm_hash]:
        if mm_hash in self.cpu_freeable:
            ent = self.cpu_freeable.pop(mm_hash)
            self.cpu_num_freeable_slots -= ent.num_embeds
        if mm_hash in self.gpu_freeable:
            ent = self.gpu_freeable.pop(mm_hash)
            self.gpu_num_freeable_slots -= ent.num_embeds
    # 对于未缓存过的新请求
    if request.request_id not in self.cached[mm_hash]:
        self.cached[mm_hash].add(request.request_id)
        # 已经在GPU上，直接命中
        if mm_hash in self.gpu_cache:
            ent = self.gpu_cache[mm_hash]
        else:
            if self.should_promote(mm_hash):
                ent = self.cpu_cache[mm_hash]
                self.gpu_cache[mm_hash] = ent
                self.gpu_num_free_slots -= ent.num_embeds
                self.gpu_num_freeable_slots -= ent.num_embeds
                # 标记mm_hash到待晋升列表
                self.promoting.append(mm_hash)
            else:
                # 当前 step 需要，但不够热，用完可能释放。放入tmp_encoder_cache
                self.cpu_get_encoder_mm_hashes.append(mm_hash)
                ent = self.cpu_cache[mm_hash]

        self.on_request()
        ent.freq += 1
        ent.clock = self.max_clock

    return True
```

---

### 4.2 晋升判断：`should_promote()`

位置：`vllm/v1/core/encoder_cache_manager.py:477-520`

逻辑：

- GPU 总可回收空间不够 → 无法晋升。
- GPU 有直接空间 → 直接晋升。
- 否则与 `gpu_freeable` 中低分条目比较：
  - 当前 entry score 高于 `promote_percentile` （晋升到GPU的百分位阈值）对应的得分 → 淘汰 GPU 中 score 最低的若干条目，然后晋升。

```python
def should_promote(self, mm_hash: str) -> bool:
    ent = self.cpu_cache[mm_hash]

    if ent.num_embeds > self.gpu_num_freeable_slots:
        return False

    if ent.num_embeds <= self.gpu_num_free_slots:
        return True

    ent_value = self.score(ent)
    scored = [(self.score(cur_ent), cur_hash, cur_ent)
              for cur_hash, cur_ent in self.gpu_freeable.items()]
    scored.sort(key=lambda x: x[0])
    # 取前promote_percentile比例的索引idx, 并得到其对应得分threshold
    idx = max(0, min(len(scored) - 1, int(len(scored) * self.promote_percentile)))
    threshold = scored[idx][0]

    # 低于晋升最低分数线，不能晋升了
    if ent_value < threshold:
        return False

    # 两个条件同时满足：1.保证淘汰后的水位线；2.至少能够腾出一个新emb cache的空间
    free_slots = max(
        self.cache_size * self.watermark - self.gpu_num_free_slots,
        ent.num_embeds - self.gpu_num_free_slots
    )
    i = 0
    while free_slots > 0:
        min_hash = scored[i][1]
        evict_ent = self.gpu_freeable.pop(min_hash)
        self.evict_from_gpu(evict_ent)
        i += 1
        free_slots -= evict_ent.num_embeds

    return True
```

---

### 4.3 Score 计算

位置：`vllm/v1/core/encoder_cache_manager.py:465-467`

```python
def score(self, ent: CacheEntry) -> float:
    # ent.clock从最开始的最大，每次on_request都会减1
    return (ent.freq + ent.clock) * ent.cal_cost
```

- `freq`：访问频率。
- `clock`：时钟值，最近访问重置为 `max_clock`，定期衰减。
- `cal_cost`：理论重计算代价，越大越值得缓存。

`cal_cost` 通过 vision config 估算：

```python
self.attn_heads = vllm_config.model_config.hf_config.vision_config.num_heads
self.hidden_size = vllm_config.model_config.hf_config.vision_config.hidden_size
self.feedforward = vllm_config.model_config.hf_config.vision_config.intermediate_size

self.alpha = 4 * self.hidden_size + 5 * self.attn_heads
self.beta = self.hidden_size * (8 * self.hidden_size + 6 * self.feedforward + 14)

def cal_theory_cost_storage_cost(self, seq_len: int) -> float:
    cost = 32 * (self.alpha * seq_len + self.beta)
    return cost / self.hardware_flops
```

---

### 4.4 CPU空间分配：`can_allocate()`

位置：`vllm/v1/core/encoder_cache_manager.py:583-620`

```python
def can_allocate(self, request, input_id, encoder_compute_budget, num_embeds_to_schedule):
    num_embeds = request.get_num_encoder_embeds(input_id)

    if num_embeds > encoder_compute_budget:
        return False

    num_embeds += num_embeds_to_schedule

    if num_embeds > self.cpu_num_freeable_slots:
        return False

    while num_embeds > self.cpu_num_free_slots:
        mm_hash, ent = self.cpu_freeable.popitem(last=False)
        del self.cached[mm_hash]
        del self.cpu_cache[mm_hash]
        self.freed.append(mm_hash)
        self.cpu_num_free_slots += ent.num_embeds

    return True
```

注意：**新计算的 embedding 总是先放入 CPU cache**。

---

### 4.5 实际分配：`allocate()`

位置：`vllm/v1/core/encoder_cache_manager.py:639-671`

```python
def allocate(self, request: Request, input_id: int) -> None:
    mm_hash = request.mm_features[input_id].identifier

    if mm_hash not in self.cached:
        self.cached[mm_hash] = set()

    num_encoder_embeds = request.get_num_encoder_embeds(input_id)
    cache_entry = CacheEntry(
        mm_hash=mm_hash,
        freq=1,
        clock=self.max_clock,
        num_embeds=num_encoder_embeds,
        cal_cost=self.cal_theory_cost_storage_cost(num_encoder_embeds),
    )

    self.cpu_num_free_slots -= num_encoder_embeds
    self.cpu_num_freeable_slots -= num_encoder_embeds

    self.cpu_cache[mm_hash] = cache_entry
    self.cached[mm_hash].add(request.request_id)
```

---

### 4.6 时钟衰减：`on_request()`

位置：`vllm/v1/core/encoder_cache_manager.py:573-581`

```python
def on_request(self):
    self.req_cnt += 1
    if self.req_cnt % self.clock_decay_every == 0:
        for ent in self.gpu_cache.values():
            ent.clock = max(0, ent.clock - 1)

    if self.req_cnt % 1000 == 0:
        self._check_invariant()
```

每隔 `clock_decay_every` 个请求，GPU cache 中所有 entry 的 clock 减 1，实现 aging。

---

### 4.7 释放引用：`free_encoder_input()` 和 `free()`

位置：`vllm/v1/core/encoder_cache_manager.py:673-689`

```python
def free_encoder_input(self, request: Request, input_id: int) -> None:
    req_id = request.request_id
    mm_hash = request.mm_features[input_id].identifier
    if not self.cached.get(mm_hash, None):
        return
    self.cached[mm_hash].discard(req_id)
    if self.cached[mm_hash]:
        return
    num_encoder_embeds = request.get_num_encoder_embeds(input_id)
    if mm_hash in self.cpu_cache:
        self.cpu_freeable[mm_hash] = self.cpu_cache[mm_hash]
        self.cpu_num_freeable_slots += num_encoder_embeds
    if mm_hash in self.gpu_cache:
        self.gpu_freeable[mm_hash] = self.gpu_cache[mm_hash]
        self.gpu_num_freeable_slots += num_encoder_embeds
```

请求完成或取消时减少引用；引用为 0 时放入对应层级的 `freeable`。

---

## 5. 与 Worker 的交互

Manager 只负责**元数据调度**，真正的 tensor 存储和搬运在 worker 端执行。

Scheduler 通过以下接口把决策通知给 worker：

- `get_promoting_mm_hashes()`：需要从 CPU 提升到 GPU 的 mm_hash 列表。
- `get_cpu_get_encoder_mm_hashes()`：需要从 CPU 加载的 mm_hash 列表。
- `get_freed_mm_hashes()`：需要释放的 mm_hash 列表。

worker 端在 `vllm/v1/worker/gpu_model_runner.py` 有相关检查：

```python
from vllm.config.score_encoder_cache import get_score_encoder_cache_config

# gpu_model_runner.py:1060
if not get_score_encoder_cache_config(self.vllm_config).enabled:
    ...

# gpu_model_runner.py:2936
if get_score_encoder_cache_config(self.vllm_config).enabled:
    ...
```

---

## 6. 不变式检查

`_check_invariant()`（`encoder_cache_manager.py:700-759`）每 1000 个请求检查一次：

- CPU/GPU 总占用 + 空闲 = 总容量。
- `freeable_slots = free_slots + freeable 中条目占用之和`。
- `freeable` 中的条目必须没有请求引用。

---

## 7. Worker 端执行：`_async_process_scheduler_output()`

位置：`vllm/v1/worker/gpu_model_runner.py:1076-1107`

这是 `ScoreEncoderCacheManager` 两级缓存策略在 worker 端的实际执行入口，由 `_update_states()` 调用。它根据 scheduler output 异步处理 encoder cache 的释放、晋升和预取。

### 7.1 涉及的数据结构

在 `GPUModelRunner` 中初始化了三个缓存字典：

```python
# gpu_model_runner.py:501
self.encoder_cache: dict[str, torch.Tensor] = {}       # GPU 侧主 cache

# gpu_model_runner.py:862-863
self.tmp_encoder_cache: dict[str, torch.Tensor] = {}   # GPU 侧临时 cache
self.cpu_encoder_cache: dict[str, torch.Tensor] = {}   # CPU 侧 cache
```

| 缓存 | 位置 | 用途 |
|---|---|---|
| `encoder_cache` | GPU | 热数据，长期保留在 GPU |
| `tmp_encoder_cache` | GPU | 刚从 CPU 加载或新计算但未晋升的数据 |
| `cpu_encoder_cache` | CPU (pinned memory) | 冷数据，需要时异步搬到 GPU |

### 7.2 释放 GPU encoder cache

```python
for mm_hash in scheduler_output.free_encoder_mm_hashes:
    value = self.encoder_cache.pop(mm_hash, None)
    if value is None and mm_hash not in scheduler_output.promoting_mm_hashes:
        self.cpu_encoder_cache.pop(mm_hash, None)
```

- 先从 GPU 主 cache 中删除。
- 如果 GPU 没有，且不在晋升列表中，再从 CPU cache 删除。
- 排除 `promoting_mm_hashes` 是为了避免先删掉 CPU 版本导致后续晋升无数据可搬。

### 7.3 晋升 CPU → GPU

```python
for mm_hash in scheduler_output.promoting_mm_hashes:
    cpu_value = self.cpu_encoder_cache.get(mm_hash, None)
    if cpu_value is None:
        continue

    staging = cpu_value.pin_memory()
    gpu_value = staging.detach().to(self.device, non_blocking=True)
    gpu_value.record_stream(torch.cuda.current_stream())

    self.encoder_cache[mm_hash] = gpu_value
    del staging
```

- 从 CPU cache 取出 tensor。
- `pin_memory()` 确保 page-locked 内存，便于异步 DMA。
- `non_blocking=True` 实现异步 CPU→GPU 拷贝。
- `record_stream()` 保证 tensor 生命周期覆盖 GPU 使用。
- 放入 `encoder_cache`，后续 `_gather_mm_embeddings()` 优先从这里取。

### 7.4 CPU → GPU 临时预取

```python
for mm_hash in scheduler_output.cpu_get_encoder_mm_hashes:
    if mm_hash in self.encoder_cache or mm_hash in self.tmp_encoder_cache:
        continue
    cpu_value = self.cpu_encoder_cache.get(mm_hash, None)
    if cpu_value is None:
        continue
    staging = cpu_value.pin_memory()
    gpu_value = staging.detach().to(self.device, non_blocking=True)
    gpu_value.record_stream(torch.cuda.current_stream())
    self.tmp_encoder_cache[mm_hash] = gpu_value
    del staging
```

- 遍历 scheduler 标记为需要从 CPU 获取的 `mm_hash`。
- 如果已经在 GPU 主 cache 或临时 cache 中，跳过。
- 否则异步搬到 GPU，但放入 `tmp_encoder_cache`。

`promoting` 与 `cpu_get` 的区别：

| 列表 | 放入的 GPU cache | 含义 |
|---|---|---|
| `promoting_mm_hashes` | `encoder_cache` | 热点数据，值得长期留在 GPU |
| `cpu_get_encoder_mm_hashes` | `tmp_encoder_cache` | 当前 step 需要，但不够热，用完可能释放 |

### 7.5 异步特性

- `non_blocking=True` 的 CPU→GPU 拷贝。
- `record_stream` 避免同步等待。
- 实际 tensor 使用发生在后续 `_gather_mm_embeddings()` 和模型前向中，GPU stream 保证依赖关系。

### 7.6 与其他方法的配合

#### 计算后写入缓存

`gpu_model_runner.py:2935-2958`：

```python
for mm_hash, output in zip(mm_hashes, encoder_outputs):
    if get_score_encoder_cache_config(self.vllm_config).enabled:
        staging = torch.empty_like(output, device="cpu", pin_memory=True)
        staging.copy_(output.detach(), non_blocking=True)
        self.cpu_encoder_cache[mm_hash] = staging

        if (
            mm_hash in scheduler_output.promoting_mm_hashes and
            mm_hash not in scheduler_output.free_encoder_mm_hashes
        ):
            self.encoder_cache[mm_hash] = output
        else:
            self.tmp_encoder_cache[mm_hash] = output
    else:
        self.encoder_cache[mm_hash] = output
```

新算出的 embedding：

- 复制一份到 `cpu_encoder_cache`。
- 如果 scheduler 要求晋升且不被释放，同时保留在 `encoder_cache`。
- 否则只放在 `tmp_encoder_cache`。

#### 使用前读取缓存

`gpu_model_runner.py:3020-3024`：

```python
mm_hash = mm_feature.identifier
encoder_output = self.encoder_cache.get(mm_hash, None)
if encoder_output is None:
    encoder_output = self.tmp_encoder_cache.get(mm_hash, None)
assert encoder_output is not None, f"Encoder cache miss for {mm_hash}"
```

优先从 `encoder_cache` 取，没有再查 `tmp_encoder_cache`。

---

## 8. 异步 CPU→GPU 传输的三个关键操作

在 `_async_process_scheduler_output()` 中，CPU→GPU 的 embedding 搬运使用了标准的三段式组合：

```python
staging = cpu_value.pin_memory()
gpu_value = staging.detach().to(self.device, non_blocking=True)
gpu_value.record_stream(torch.cuda.current_stream())
```

下面分别解释它们的作用和相互配合关系。

### 8.1 `pin_memory()` — 固定 CPU 内存为 page-locked memory

普通 CPU 内存可能被操作系统换页。当 GPU 通过 DMA 从 CPU 拷贝数据时，如果内存不是 page-locked，GPU 驱动需要先把数据复制到一个临时的 pinned buffer，再启动 DMA，这个中间步骤会和 GPU kernel 串行，导致同步等待。

`pin_memory()` 把 CPU tensor 放到操作系统无法换出的物理页上，称为 **page-locked** 或 **pinned memory**。

**作用**：

- 允许 GPU 直接通过 DMA 从 CPU 取数据，无需额外中间拷贝。
- 是实现 `non_blocking=True` 异步传输的前提条件。

**代价**：

- pinned memory 占用物理内存，不能换页，不宜过多。
- 分配和释放比普通 CPU 内存慢。

### 8.2 `to(self.device, non_blocking=True)` — 异步拷贝到 GPU

```python
# 同步拷贝，CPU 阻塞等待完成
gpu_value = cpu_value.to("cuda")

# 异步拷贝，CPU 立即返回继续执行后续代码
gpu_value = cpu_value.to("cuda", non_blocking=True)
```

`non_blocking=True` 让 GPU 在后台执行拷贝，同时 CPU 可以继续处理其他逻辑（如调度、准备下一个 batch）。

**注意**：`non_blocking=True` 要真正生效，CPU 内存通常需要是 pinned 的：

```python
cpu_value.pin_memory().to("cuda", non_blocking=True)  # 正确，异步
cpu_value.to("cuda", non_blocking=True)               # 可能回退为同步
```

### 8.3 `record_stream(torch.cuda.current_stream())` — 延长 tensor 生命周期

这是异步拷贝中最容易忽略但最关键的一步。

**问题背景**：

异步拷贝的语义是 `to()` 调用立即返回，GPU 上的实际拷贝操作被排到 CUDA stream 队列里，可能在未来某个时刻才执行。CPU 继续往下执行，可能很快就会：

- 删除 `staging`
- 覆盖 `cpu_value`
- 让 tensor 离开作用域

如果 CPU 侧的 tensor 在 GPU 还没用完之前就被释放或修改，会导致：

- **use-after-free**：GPU 读取已被释放的 CPU 内存。
- **数据竞争**：GPU 读到被覆盖的数据。
- **崩溃或错误结果**。

**`record_stream()` 的作用**：

```python
gpu_value.record_stream(torch.cuda.current_stream())
```

它告诉 PyTorch：

> 这个 tensor 的生命周期必须至少延续到当前 CUDA stream 上所有已提交的工作完成。

具体机制：

- PyTorch 记录哪些 tensor 被哪些 stream 使用。
- 调用 `record_stream(stream)` 后，tensor 的引用被关联到这个 stream。
- 只有当 stream 上所有已排队的工作都完成后，tensor 的内存才会被释放或重用。

**在 vLLM 中的具体场景**：

```python
staging = cpu_value.pin_memory()
gpu_value = staging.detach().to(self.device, non_blocking=True)
gpu_value.record_stream(torch.cuda.current_stream())

self.encoder_cache[mm_hash] = gpu_value
del staging
```

如果没有 `record_stream()`：

- `del staging` 之后，pinned CPU buffer 可能被释放。
- 但 GPU 上的异步拷贝可能还没开始执行。
- 结果：GPU 拷贝时读到无效内存。

有了 `record_stream()`：

- `gpu_value` 被加入当前 stream 的待释放列表。
- 即使 `staging` 被删除，`gpu_value` 的底层数据在 GPU stream 完成前不会真正释放。
- 拷贝安全完成。

### 8.4 三者组合的标准模式

```python
cpu_value = ...                    # CPU 上的源数据
staging = cpu_value.pin_memory()   # 1. 固定内存，使能异步 DMA
gpu_value = staging.detach().to(  # 2. 启动异步拷贝
    self.device,
    non_blocking=True
)
gpu_value.record_stream(           # 3. 延长生命周期到 stream 完成
    torch.cuda.current_stream()
)
self.cache[key] = gpu_value
del staging                        # CPU 中间 buffer 可以安全删除
```

### 8.5 一句话总结

`pin_memory()` 让 CPU 内存可被 GPU 直接 DMA；`non_blocking=True` 让拷贝和 CPU 计算并行；`record_stream()` 保证源/目标 tensor 在 GPU 异步使用完成前不被释放或覆盖。三者配合才能实现安全高效的异步 CPU→GPU 数据传输。

---

## 9. 总结

`ScoreEncoderCacheManager` 实现了一个 **GPU-CPU 两级 encoder embedding 缓存**：

- 新算出的 embedding 默认进入 CPU cache。
- 热点 embedding 根据 `score = (freq + clock) * cal_cost` 晋升到 GPU cache。
- GPU 满时按 score 淘汰低价值条目。
- CPU 满时按 FIFO 淘汰。
- Manager 只负责调度决策，实际 tensor 存储和跨层搬运由 worker 执行。
- Worker 端通过 `_async_process_scheduler_output()` 异步完成释放、晋升和预取操作。
- 异步 CPU→GPU 传输依赖 `pin_memory()`、`non_blocking=True` 和 `record_stream()` 的配合。
