# vLLM 中 `--max-model-len` 参数研究

## 1. 参数入口与配置推导

### 1.1 参数定义

`--max-model-len` 对应 `ModelConfig.max_model_len`，定义在 `vllm/config/model.py:188-200`：

```python
max_model_len: int = Field(default=None, ge=-1)
"""Model context length (prompt and output). If unspecified, will be
automatically derived from the model config.

When passing via `--max-model-len`, supports k/m/g/K/M/G in human-readable
format. Examples:

- 1k -> 1000
- 1K -> 1024
- 25.6k -> 25,600
- -1 or 'auto' -> Automatically choose the maximum model length that fits in
  GPU memory. This will use the model's maximum context length if it fits,
  otherwise it will find the largest length that can be accommodated."""
```

在 `ModelConfig.__post_init__` 中完成推导与校验：

```python
# vllm/config/model.py:675-677
self.original_max_model_len = self.max_model_len
self.max_model_len = self.get_and_verify_max_len(self.max_model_len)
```

### 1.2 `get_and_verify_max_len` 实现细节

`vllm/config/model.py:1753-1777`：

```python
def get_and_verify_max_len(self, max_model_len: int):
    tokenizer_config = None
    if (
        self.runner_type == "pooling"
        and getattr(self.hf_config, "position_embedding_type", "") == "absolute"
    ):
        tokenizer_config = try_get_tokenizer_config(...)
    max_model_len = _get_and_verify_max_len(
        hf_config=self.hf_text_config,
        model_arch_config=self.model_arch_config,
        tokenizer_config=tokenizer_config,
        max_model_len=max_model_len,
        disable_sliding_window=self.disable_sliding_window,
        sliding_window=self.get_sliding_window(),
        spec_target_max_model_len=self.spec_target_max_model_len,
        encoder_config=self.encoder_config,
    )
    logger.info("Using max model len %s", max_model_len)
    return max_model_len
```

实际逻辑在 `_get_and_verify_max_len`（`vllm/config/model.py:2143-2280`）。

### 1.3 `derived_max_model_len_and_key` 的作用与含义

`derived_max_model_len_and_key` 定义在 `vllm/config/model_arch.py:62-63`：

```python
derived_max_model_len_and_key: tuple[float, str | None]
"""Derived maximum model length and key from the hf config."""
```

实际生成在 `vllm/transformers_utils/model_arch_config_convertor.py:320-353`：

```python
def derive_max_model_len_and_key(self) -> tuple[float, str | None]:
    derived_max_model_len = float("inf")
    possible_keys = [
        "max_position_embeddings",  # OPT, Llama, Qwen, Gemma 等
        "n_positions",              # GPT-2
        "max_seq_len",              # MPT
        "seq_length",               # ChatGLM2
        "model_max_length",         # Command-R / Cohere
        "max_target_positions",     # Whisper
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(self.hf_text_config, key, None)
        if max_len is not None:
            if max_len < derived_max_model_len:
                max_len_key = key
            derived_max_model_len = min(derived_max_model_len, max_len)

    # For Command-R / Cohere, Cohere2 models
    if tmp_max_len := getattr(self.hf_text_config, "model_max_length", None):
        max_len_key = "model_max_length"
        derived_max_model_len = tmp_max_len
    return derived_max_model_len, max_len_key
```

**作用与含义**：

- `derived_max_model_len`：从 Hugging Face 模型配置中推导出的最大上下文长度。
- `max_len_key`：实际使用的是哪个配置字段。

**为什么要扫描多个 key 并取最小值？**

不同模型架构使用不同的字段名表示最大序列长度。vLLM 需要支持各种模型架构，因此扫描所有可能的字段。取最小值的原因是：模型上下文长度受限于所有相关配置中最严格的那个。

### 1.4 Command-R / Cohere 系列特殊处理

代码：

```python
# For Command-R / Cohere, Cohere2 models
if tmp_max_len := getattr(self.hf_text_config, "model_max_length", None):
    max_len_key = "model_max_length"
    derived_max_model_len = tmp_max_len
```

Command-R、Cohere、Cohere2 等模型的 `config.json` 中通常同时存在：

- `max_position_embeddings`：一个较大的值，表示 RoPE/position embedding 的理论上限。
- `model_max_length`：模型实际训练或推荐使用的最大上下文长度。

对于这类模型，`model_max_length` 才是真正有效的上下文长度限制，`max_position_embeddings` 可能更大但并不代表模型在该长度上表现可靠。因此 vLLM 特殊处理：如果 `model_max_length` 存在，直接用它覆盖之前扫描得到的最小值。

### 1.5 Sliding Window 限制

代码：

```python
if (
    disable_sliding_window
    and sliding_window is not None
    and sliding_window < derived_max_model_len
):
    max_len_key = "sliding_window"
    derived_max_model_len = sliding_window
```

**Sliding Window 是什么？**

Sliding window attention（滑动窗口注意力）是一种注意力机制，每个 token 只关注其左右固定窗口大小内的 token，而不是整个历史序列。

**为什么禁用后还要用 `sliding_window` 作为上限？**

当 `disable_sliding_window=True` 时，vLLM 会关闭滑动窗口机制，退化为普通的 full attention。但是模型的 position embeddings / RoPE 可能仍然是按照 `sliding_window` 大小训练的。如果 `sliding_window < derived_max_model_len`，那么即使关闭滑动窗口，也不能让序列长度超过 `sliding_window`，否则 position embedding 会越界。

### 1.6 Tokenizer Config 限制

代码：

```python
if tokenizer_config:
    tokenizer_model_max_length = tokenizer_config.get(
        "model_max_length", derived_max_model_len
    )
    derived_max_model_len = min(derived_max_model_len, tokenizer_model_max_length)
```

Tokenizer config 是 `tokenizer_config.json` 中的配置，其中可能包含 `model_max_length` 字段，表示 tokenizer 能够有效处理的最大 token 数。vLLM 只对 pooling + absolute position embedding 的模型启用这个检查，因为这类模型对 tokenizer 长度限制最敏感。最终 `derived_max_model_len` 取模型配置推导值和 tokenizer `model_max_length` 的最小值。

### 1.7 RoPE Scaling

代码：

```python
rope_parameters = getattr(hf_config, "rope_parameters", None)
if rope_parameters is not None and "gemma3" not in hf_config.model_type:
    scaling_factor = 1.0
    for rp in rope_parameters.values():
        rope_type = rp["rope_type"]
        if rope_type not in ("su", "longrope", "llama3"):
            scaling_factor = rp.get("factor", scaling_factor)
            if rope_type == "yarn":
                derived_max_model_len = rp["original_max_position_embeddings"]
    derived_max_model_len *= scaling_factor
```

RoPE scaling 会按 scaling factor 扩大模型可处理的上下文长度。Gemma3 会跳过，因为其 `max_model_len` 已经被 RoPE scaling 过。

### 1.8 Encoder Config 覆盖

对于 Sentence Transformer 等 encoder model，若 `encoder_config.max_seq_length` 存在：

```python
if encoder_config and "max_seq_length" in encoder_config:
    derived_max_model_len = encoder_config["max_seq_length"]
```

这些模型通常使用 `--runner pooling` 运行，它们的 `max_seq_length` 是训练时确定的有效上限。

### 1.9 用户指定值的优先级

代码：

```python
if max_model_len is None or max_model_len == -1:
    max_model_len = int(derived_max_model_len)
elif max_model_len > derived_max_model_len:
    if envs.VLLM_ALLOW_LONG_MAX_MODEL_LEN:
        logger.warning_once(...)
    else:
        raise ValueError(...)
```

用户指定的 `--max-model-len` 优先级很高，但有约束：

| 情况 | 行为 |
|---|---|
| 未指定 / `-1` / `auto` | 使用推导值 `derived_max_model_len` |
| 用户指定且 `<= derived_max_model_len` | 使用用户指定值 |
| 用户指定且 `> derived_max_model_len` | 默认报错；设置 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 后可强制使用 |

推导值是"安全上限"，用户值是"实际使用的上限"。

---

## 2. GPU 内存管理与 `gpu-memory-utilization`

### 2.1 参数定义

`vllm/config/cache.py:68-75`：

```python
gpu_memory_utilization: float = Field(default=0.92, gt=0, le=1)
"""The fraction of GPU memory to be used for the model executor, which can
range from 0 to 1. For example, a value of 0.5 would imply 50% GPU memory
utilization."""
```

### 2.2 正式生效位置

Worker 初始化时，`vllm/v1/worker/gpu_worker.py:374-376`：

```python
self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
self.requested_memory = request_memory(init_snapshot, self.cache_config)
```

`request_memory` 在 `vllm/v1/worker/utils.py:422-442`：

```python
def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )
    if init_snapshot.free_memory < requested_memory:
        raise ValueError(...)
    return requested_memory
```

这就是 `gpu_memory_utilization` 的正式生效点：它计算 vLLM 这个实例可以使用的 GPU memory 上限（bytes）。

### 2.3 `MemorySnapshot` 底层实现

`vllm/utils/mem_utils.py:108-164`：

```python
@dataclass
class MemorySnapshot:
    torch_peak: int = 0
    free_memory: int = 0
    total_memory: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0

    def measure(self) -> None:
        device = self.device_
        self.torch_peak = torch.accelerator.memory_stats(device).get(
            "allocated_bytes.all.peak", 0
        )
        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)
        self.cuda_memory = self.total_memory - self.free_memory
        self.torch_memory = torch.accelerator.memory_reserved(device)
        self.non_torch_memory = self.cuda_memory - self.torch_memory
        self.timestamp = time.time()
```

### 2.4 HBM 划分：两部分

vLLM 把 GPU memory 分成两大块：

#### 第一块：non-KV-cache memory

包括：

- 模型权重（weights）
- 峰值激活（peak activation tensors）
- 非 torch 内存（NCCL buffer、CUDA context、attention backend workspace 等）
- CUDA graph 预留内存

在 `determine_available_memory()` 中通过 profiling 得到：

```python
# vllm/v1/worker/gpu_worker.py:470-529
with memory_profiling(
    self.init_snapshot,
    weights_memory=int(self.model_runner.model_memory_usage),
) as profile_result:
    self.model_runner.profile_run()
    profile_torch_peak = torch.accelerator.memory_stats(self.device).get(
        "allocated_bytes.all.peak", 0
    )
    cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()

self.available_kv_cache_memory_bytes = (
    self.requested_memory
    - profile_result.non_kv_cache_memory
    - cudagraph_memory_estimate_applied
)
```

其中：

```python
# vllm/utils/mem_utils.py:309-313
profile_result.non_kv_cache_memory = (
    profile_result.non_torch_increase
    + profile_result.torch_peak_increase
    + profile_result.weights_memory
)
```

#### 第二块：KV cache memory

```python
self.available_kv_cache_memory_bytes = (
    self.requested_memory
    - profile_result.non_kv_cache_memory
    - cudagraph_memory_estimate_applied
)
```

这是留给 KV cache 的显存。

### 2.5 `determine_available_memory()` 详细实现

`vllm/v1/worker/gpu_worker.py:434-560`：

1. 检查是否手动指定 KV cache 内存（`kv_cache_memory_bytes`），若是则跳过 profiling。
2. 启动 `memory_profiling` 上下文。
3. 执行 `model_runner.profile_run()` 模拟 dummy forward。
4. 记录 `weights_memory`、`torch_peak_increase`、`non_torch_increase`。
5. 计算 `non_kv_cache_memory = weights + torch_peak + non_torch`。
6. 估算 CUDA graph 内存（可选）。
7. 计算 `available_kv_cache_memory_bytes = requested_memory - non_kv_cache_memory - cudagraph_estimate`。
8. 返回该值。

---

## 3. KV Cache 空间分配流程

### 3.1 调用链路

```
EngineCore._initialize_kv_caches()
  -> model_executor.get_kv_cache_spec()
  -> model_executor.determine_available_memory()   # 每 worker 可用 KV cache 显存
  -> get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory)
       -> 合并 kv_cache_specs
       -> 生成全局 KV cache groups
       -> 若 original_max_model_len == -1：
              _auto_fit_max_model_len(...)
       -> _check_enough_kv_cache_memory(...)
       -> 每 worker 生成 KVCacheConfig
       -> 取所有 worker 最小 num_blocks
  -> 若 auto-fit 改变 max_model_len：
       collective_rpc("update_max_model_len", ...)
  -> cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks
  -> model_executor.initialize_from_config(kv_cache_configs)   # 实际分配 GPU tensor
```

### 3.2 `_auto_fit_max_model_len` 调用链路

```
EngineCore._initialize_kv_caches()
  -> get_kv_cache_configs(...)
       -> if vllm_config.model_config.original_max_model_len == -1:
              _auto_fit_max_model_len(vllm_config, projected_groups_per_worker, available_memory)
                 -> for each worker:
                        _estimate_max_model_len_from_groups(...)
                           -> 二分查找 max_model_len
                              -> _max_memory_usage_bytes_from_groups(...)
                                 -> kv_cache_spec.max_memory_usage_bytes(vllm_config)
                 -> vllm_config.model_config.max_model_len = min(worker_max_values)
```

代码：`vllm/v1/core/kv_cache_utils.py:1921-1982`。

### 3.3 `max_model_len` 如何影响 KV cache 大小

#### (1) 直接影响单请求最大 KV cache 用量

`vllm/v1/kv_cache_interface.py:237-245`：

```python
def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
    max_model_len = vllm_config.model_config.max_model_len
    dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
    if dcp_world_size * pcp_world_size > 1:
        max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
    return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

#### (2) 决定每请求 block 数与最大并发

```python
# vllm/v1/core/kv_cache_utils.py:933-951
def get_max_concurrency_for_kv_cache_config(...):
    num_layer_per_group = max(len(group.layer_names) ...)
    max_memory_usage_per_request = num_layer_per_group * max_memory_usage_bytes(...)
    memory_per_block = ...
    num_block_per_request = cdiv(max_memory_usage_per_request, memory_per_block)
    max_concurrency = kv_cache_config.num_blocks / num_block_per_request
    return max_concurrency
```

#### (3) KV cache 容量（tokens）

```python
# vllm/v1/core/kv_cache_utils.py:1810-1820
def get_kv_cache_capacity(...):
    max_model_len = vllm_config.model_config.max_model_len
    max_concurrency = get_max_concurrency_for_kv_cache_config(...)
    return int(max_concurrency * max_model_len), max_concurrency
```

**结论**：`max_model_len` 越大，单请求 KV cache 用量越大，同样 block 数下最大并发越低。

### 3.4 `num_blocks` 的含义

`vllm/v1/core/kv_cache_utils.py:986-1003`：

```python
def get_num_blocks(
    vllm_config: VllmConfig,
    num_layers: int,
    available_memory: int,
    page_size: int,
) -> int:
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    return may_override_num_blocks(vllm_config, num_blocks)
```

`num_blocks` 是 KV cache pool 中逻辑块的总数。每个 block 可以容纳 `block_size` 个 token 的 K/V 值。

### 3.5 KV cache 空间确定后还会改变吗？

通常不会动态改变：

- `num_gpu_blocks` 在启动时确定。
- `KVCacheTensor` 的大小在 `initialize_from_config()` 中分配后固定。

例外：sleep mode、weight update、LoRA swap、EEP scale-up 等场景可能重新初始化。

---

## 4. KVCacheManager 的 Token 硬上限

### 4.1 实现位置

`vllm/v1/core/kv_cache_manager.py:127`：

```python
self.max_model_len = max_model_len
```

在 `allocate_slots` 中多处截断：

```python
# vllm/v1/core/kv_cache_manager.py:358-361
total_computed_tokens = min(
    num_local_computed_tokens + num_external_computed_tokens,
    self.max_model_len,
)

# :374
full_num_tokens = min(request.num_tokens, self.max_model_len)

# :389-392
num_tokens_need_slot = min(
    num_tokens_main_model + num_lookahead_tokens, self.max_model_len
)
```

### 4.2 Scheduler 层的截断

`vllm/v1/core/sched/scheduler.py:484-489`：

```python
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len
    - request.num_computed_tokens
    - self.num_sampled_tokens_per_step,
)
```

### 4.3 停止检查

`vllm/v1/core/sched/scheduler.py:1917`：

```python
stopped = check_stop(request, self.max_model_len)
```

### 4.4 作用

1. 防止调度超过模型 position embedding / RoPE 可表示的位置。
2. 确保 KV cache 分配不超出物理 block pool。
3. 控制生成长度，达到 `max_model_len` 时停止生成。

---

## 5. 输入层校验与默认 max_tokens

### 5.1 默认生成长度

`vllm/v1/engine/input_processor.py:316-321`：

```python
if sampling_params.max_tokens is None:
    seq_len = length_from_prompt_token_ids_or_embeds(prompt_token_ids, prompt_embeds)
    sampling_params.max_tokens = self.model_config.max_model_len - seq_len
```

### 5.2 Prompt 长度校验

`vllm/v1/engine/input_processor.py:399-432`：

```python
max_prompt_len = model_config.max_model_len
if prompt_len > max_prompt_len:
    raise ValueError(...)
elif prompt_len == max_prompt_len and model_config.runner_type == "generate":
    raise ValueError(...)
```

---

## 6. EncoderCacheManager 与 `max_model_len`

### 6.1 `EncoderCacheManager` 初始化

`vllm/v1/core/sched/scheduler.py:221-229`：

```python
self.max_num_encoder_input_tokens = (
    mm_budget.encoder_compute_budget if mm_budget else 0
)
encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
self.encoder_cache_manager = (
    EncoderDecoderCacheManager(cache_size=encoder_cache_size)
    if self.is_encoder_decoder
    else EncoderCacheManager(cache_size=encoder_cache_size)
)
```

### 6.2 `cache_size` 的来源

`vllm/multimodal/encoder_budget.py:117-123`：

```python
encoder_compute_budget, encoder_cache_size = compute_mm_encoder_budget(
    scheduler_config,
    active_mm_max_toks_per_item,
)
self.encoder_compute_budget = encoder_compute_budget
self.encoder_cache_size = encoder_cache_size
```

`compute_mm_encoder_budget` 在 `vllm/v1/core/encoder_cache_manager.py:309-316`：

```python
encoder_compute_budget = max(
    scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item
)
encoder_cache_size = max(
    scheduler_config.encoder_cache_size, max_tokens_per_mm_item
)
```

### 6.3 是否与 `max_model_len` 有关？

**直接关系不大。** `EncoderCacheManager.cache_size` 的单位是 encoder embeddings 数量，不直接来自 `max_model_len`。

但间接相关：`MultiModalBudget._get_max_items` 中用 `max_model_len` 计算 `max_items_per_prompt`：

```python
# vllm/multimodal/encoder_budget.py:159-162
max_items_per_prompt = max(
    1,
    min(mm_limit, self.max_model_len // max_tokens_per_item),
)
```

这限制单个 prompt 中的多模态 item 数量，从而影响 encoder cache 的需求。

### 6.4 `max_items_per_prompt` 参数解释

- `mm_limit`：该模态每个 prompt 允许的最大 item 数。
- `self.max_model_len`：模型最大上下文长度。
- `max_tokens_per_item`：单个 item 占用的最大 token 数。
- `self.max_model_len // max_tokens_per_item`：理论上一个 prompt 最多能塞多少个该模态 item。
- `min(mm_limit, ...)`：取用户/模型限制和上下文长度限制的较小值。
- `max(1, ...)`：确保至少为 1。

---

## 7. 完整流程图

```
开始
  │
  ▼
[解析 CLI 参数]
  │  --gpu-memory-utilization (默认 0.92)
  │  --max-model-len / --max-model-length
  ▼
[创建 ModelConfig]
  │  推导 max_model_len：
  │    - 扫描 HF config 中 max_position_embeddings 等字段取最小值
  │    - Command-R/Cohere 用 model_max_length 覆盖
  │    - 应用 sliding window 限制
  │    - pooling + absolute position 应用 tokenizer model_max_length 限制
  │    - 应用 RoPE scaling
  │    - encoder model 应用 max_seq_length 覆盖
  │    - 用户指定值校验
  ▼
[创建 EngineCore]
  │
  ▼
[EngineCore._initialize_kv_caches()]
  │
  ├──▶ [获取 kv_cache_specs]
  │         -> model_executor.get_kv_cache_spec()
  │
  ├──▶ [worker.determine_available_memory()]
  │      │
  │      ├──▶ [MemorySnapshot] 记录 GPU 总内存 / 空闲内存
  │      │         -> torch.accelerator.get_memory_info()
  │      │
  │      ├──▶ [request_memory(total_memory * gpu_memory_utilization)]
  │      │         计算 vLLM 可用内存上限
  │      │
  │      ├──▶ [memory_profiling]
  │      │      │
  │      │      ├──▶ profile_run() 执行 dummy forward
  │      │      ├──▶ 记录 weights_memory
  │      │      ├──▶ 记录 torch_peak_increase
  │      │      └──▶ 记录 non_torch_increase
  │      │
  │      ├──▶ [non_kv_cache_memory = weights + torch_peak + non_torch]
  │      │
  │      ├──▶ [cudagraph_memory_estimate]
  │      │
  │      └──▶ [available_kv_cache_memory = requested - non_kv_cache - cudagraph]
  │
  ├──▶ [get_kv_cache_configs(...)]
  │      │
  │      ├──▶ 合并所有 worker 的 kv_cache_specs
  │      │
  │      ├──▶ 生成全局 KV cache groups
  │      │
  │      ├──▶ 若 original_max_model_len == -1：
  │      │      └──▶ _auto_fit_max_model_len(...)
  │      │             二分查找显存可容纳的最大 max_model_len
  │      │
  │      ├──▶ 检查每 worker 是否有足够显存
  │      │
  │      ├──▶ 为每 worker 生成 KVCacheConfig
  │      │       num_blocks = available_memory // page_size // num_layers
  │      │
  │      └──▶ 取所有 worker 中最小 num_blocks，同步 tensor 大小
  │
  ├──▶ [若 auto-fit 改变 max_model_len]
  │      └──▶ collective_rpc("update_max_model_len", ...)
  │
  ├──▶ [设置 cache_config.num_gpu_blocks]
  │
  ├──▶ [计算 KV cache 容量]
  │      num_tokens = max_concurrency * max_model_len
  │
  └──▶ [model_executor.initialize_from_config(kv_cache_configs)]
           │
           └──▶ 实际分配 GPU KV cache tensors
  ▼
[Scheduler 使用 KVCacheManager 运行]
  │
  └──▶ 调度请求，按需分配/回收 KV cache block
       └──▶ check_stop(request, max_model_len)
```

### 关键说明

| 阶段 | 说明 |
|---|---|
| `--max-model-len` | 用户指定的目标上下文长度，受推导安全上限约束 |
| `--gpu-memory-utilization` | 决定 vLLM 可使用 GPU 总内存的比例 |
| `determine_available_memory` | profiling 后得到 non-KV-cache 开销，从而算出 KV cache 可用显存 |
| `get_kv_cache_configs` | 根据可用显存和 `max_model_len` 计算 `num_blocks` |
| `_auto_fit_max_model_len` | 仅当 `--max-model-len -1` 时启用，二分查找适配显存的最大长度 |
| `initialize_from_config` | 实际 GPU tensor 分配点，分配后 KV cache pool 大小固定 |
| Scheduler | 运行时按 block 粒度分配和回收，但 pool 总大小不变 |

---

## 8. 总结

`--max-model-len` 是 vLLM 的上下文长度硬上限：

1. **配置阶段**：由模型 HF config 推导、用户显式值校验、`-1` 时按显存 auto-fit。
2. **KV cache 规划**：`max_model_len` 决定单请求最大 KV cache 用量，进而影响最大并发数。
3. **运行时硬上限**：`KVCacheManager` 和 `Scheduler` 通过 `min(..., max_model_len)` 截断 token 位置，`check_stop` 在达到上限时停止生成。
4. **输入校验**：`InputProcessor` 校验 prompt 长度，并设置默认 `max_tokens`。
5. **多模态间接影响**：`max_model_len` 通过 `MultiModalBudget` 限制单个 prompt 的模态 item 数量，但不直接控制 `EncoderCacheManager` 的缓存分配。

---

## 9. vLLM serve 启动参数示例

### 9.1 `scheduler_config.max_num_encoder_input_tokens`

对应 CLI 参数：

```bash
--max-num-encoder-input-tokens
```

控制每步允许编码的最大 encoder input tokens 数。

### 9.2 `scheduler_config.encoder_cache_size`

对应 CLI 参数：

```bash
--encoder-cache-size
```

控制 encoder cache 的容量（以 encoder embeddings 数量计）。

### 9.3 `max_tokens_per_mm_item`

这个**没有直接的 CLI 参数**。它由模型自身的 multimodal processor 决定，取决于：

- 模型架构
- 图像分辨率
- `limit_mm_per_prompt` 等配置
- 某些模型支持通过 `mm_processor_kwargs` 传入 processor-specific 参数间接影响

例如 Qwen2-VL 可以通过 `min_pixels`/`max_pixels` 控制每张图对应的 token 数：

```bash
--mm-processor-kwargs '{"min_pixels": 3136, "max_pixels": 12845056}'
```

但这不是通用参数，具体可用的 key 取决于模型 processor。

### 9.4 完整示例命令

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --max-num-encoder-input-tokens 65536 \
    --encoder-cache-size 65536 \
    --limit-mm-per-prompt image=4 \
    --mm-processor-kwargs '{"max_pixels": 401408}'
```

---

## 10. `MultiModalBudget._get_max_items` 具体代码

`vllm/multimodal/encoder_budget.py:140-180`：

```python
def _get_max_items(
    self,
    modality: str,
    max_tokens_per_item: int,
) -> tuple[int, int]:
    if max_tokens_per_item == 0:
        return 0, 0

    # Check how many items of this modality can be supported by
    # the encoder budget.
    if (encoder_budget := self.get_encoder_budget()) == 0:
        return 0, 0

    max_encoder_items_per_batch = encoder_budget // max_tokens_per_item

    # Check how many items of this modality can be supported by
    # the decoder budget.
    mm_limit = self.mm_limits[modality]

    max_items_per_prompt = max(
        1,
        min(mm_limit, self.max_model_len // max_tokens_per_item),
    )

    scheduler_config = self.scheduler_config
    max_num_reqs = self.max_num_reqs

    if not scheduler_config.enable_chunked_prefill:
        max_num_reqs = min(
            max_num_reqs,
            scheduler_config.max_num_batched_tokens // max_tokens_per_item,
        )

    max_decoder_items_per_batch = max_num_reqs * max_items_per_prompt

    max_items_per_batch = max(
        1,
        min(max_encoder_items_per_batch, max_decoder_items_per_batch),
    )

    return max_items_per_prompt, max_items_per_batch
```

### 10.1 代码解释

| 变量 | 含义 |
|---|---|
| `max_tokens_per_item` | 单个多模态 item 占用的最大 token 数 |
| `encoder_budget` | `min(encoder_compute_budget, encoder_cache_size)` |
| `max_encoder_items_per_batch` | 当前 encoder budget 能支持的每 batch 最大 item 数 |
| `mm_limit` | 该模态每个 prompt 允许的最大 item 数 |
| `self.max_model_len` | 模型最大上下文长度 |
| `self.max_model_len // max_tokens_per_item` | 在不考虑 text token 的情况下，一个 prompt 最多能放多少个 item |
| `max_items_per_prompt` | 取 `mm_limit` 和 `max_model_len` 限制的较小值，并确保至少为 1 |
| `max_num_reqs` | 每 batch 最大请求数 |
| `max_decoder_items_per_batch` | decoder budget 能支持的每 batch 最大 item 数 |
| `max_items_per_batch` | 最终每 batch 最大 item 数，受 encoder 和 decoder 预算共同限制 |

---

## 11. 总结

`--max-model-len` 是 vLLM 的上下文长度硬上限：

1. **配置阶段**：由模型 HF config 推导、用户显式值校验、`-1` 时按显存 auto-fit。
2. **KV cache 规划**：`max_model_len` 决定单请求最大 KV cache 用量，进而影响最大并发数。
3. **运行时硬上限**：`KVCacheManager` 和 `Scheduler` 通过 `min(..., max_model_len)` 截断 token 位置，`check_stop` 在达到上限时停止生成。
4. **输入校验**：`InputProcessor` 校验 prompt 长度，并设置默认 `max_tokens`。
5. **多模态间接影响**：`max_model_len` 通过 `MultiModalBudget` 限制单个 prompt 的模态 item 数量，但不直接控制 `EncoderCacheManager` 的缓存分配。
