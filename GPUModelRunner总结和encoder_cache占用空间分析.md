# GPUModelRunner 总结与 Encoder Cache 占用空间分析

## 目录

1. [GPUModelRunner 类方法详细分析](#1-gpumodelrunner-类方法详细分析)
   - 1.1 [初始化与配置方法](#11-初始化与配置方法)
     - 1.1.1 [`__init__(self, vllm_config, device)`](#111-__init__self-vllm_config-device)
     - 1.1.2 [`load_model(self, load_dummy_weights=False)`](#112-load_modelself-load_dummy_weightsfalse)
     - 1.1.3 [`initialize_kv_cache(self, kv_cache_config, is_profiling=False)`](#113-initialize_kv_cacheself-kv_cache_config-is_profilingfalse)
     - 1.1.4 [`profile_run(self)`](#114-profile_runself)
     - 1.1.5 [`capture_model(self)`](#115-capture_modelself)
   - 1.2 [核心推理执行方法](#12-核心推理执行方法)
     - 1.2.1 [`execute_model(self, scheduler_output, intermediate_tensors=None)`](#121-execute_modelself-scheduler_output-intermediate_tensorsnone)
     - 1.2.2 [`sample_tokens(self, grammar_output)`](#122-sample_tokensself-grammar_output)
     - 1.2.3 [`_prepare_inputs(self, scheduler_output, num_scheduled_tokens)`](#123-_prepare_inputsself-scheduler_output-num_scheduled_tokens)
     - 1.2.4 [`_preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors)`](#124-_preprocessself-scheduler_output-num_input_tokens-intermediate_tensors)
     - 1.2.5 [`_model_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)`](#125-_model_forwardself-input_ids-positions-intermediate_tensors-inputs_embeds-model_kwargs)
     - 1.2.6 [`_sample(self, logits, spec_decode_metadata)`](#126-_sampleself-logits-spec_decode_metadata)
     - 1.2.7 [`_bookkeeping_sync(self, scheduler_output, sampler_output, logits, hidden_states, num_scheduled_tokens)`](#127-_bookkeeping_syncself-scheduler_output-sampler_output-logits-hidden_states-num_scheduled_tokens)
     - 1.2.8 [`propose_draft_token_ids(...)`](#128-propose_draft_token_idsself-scheduler_output-sampled_token_ids-sampling_metadata-hidden_states-sample_hidden_states-aux_hidden_states-spec_decode_metadata-common_attn_metadata-slot_mappings)
   - 1.3 [辅助执行方法](#13-辅助执行方法)
     - 1.3.1 [`_update_states(self, scheduler_output)`](#131-_update_statesself-scheduler_output)
     - 1.3.2 [`_build_attention_metadata(...)`](#132-_build_attention_metadataself-num_tokens-num_reqs-max_query_len-)
     - 1.3.3 [`_determine_batch_execution_and_padding(...)`](#133-_determine_batch_execution_and_paddingself-num_tokens-num_reqs-num_scheduled_tokens_np-max_num_scheduled_tokens-use_cascade_attn-)
     - 1.3.4 [`_get_slot_mappings(self, num_tokens_padded, num_reqs_padded, num_tokens_unpadded, ubatch_slices)`](#134-_get_slot_mappingsself-num_tokens_padded-num_reqs_padded-num_tokens_unpadded-ubatch_slices)
     - 1.3.5 [`_execute_mm_encoder(self, scheduler_output)`](#135-_execute_mm_encoderself-scheduler_output)
     - 1.3.6 [`_gather_mm_embeddings(self, scheduler_output, shift_computed_tokens=0)`](#136-_gather_mm_embeddingsself-scheduler_output-shift_computed_tokens0)
     - 1.3.7 [`_pool(self, hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output)`](#137-_poolself-hidden_states-num_scheduled_tokens-num_scheduled_tokens_np-kv_connector_output)
   - 1.4 [虚拟运行与分析方法](#14-虚拟运行与分析方法)
     - 1.4.1 [`_dummy_run(...)`](#141-_dummy_runself-num_tokens-cudagraph_runtime_mode-force_attention-uniform_decode-)
     - 1.4.2 [`_dummy_sampler_run(self, hidden_states)`](#142-_dummy_sampler_runself-hidden_states)
     - 1.4.3 [`_dummy_pooler_run(self, hidden_states)`](#143-_dummy_pooler_runself-hidden_states)
   - 1.5 [工具与辅助方法](#15-工具与辅助方法)
     - 1.5.1 [`_calc_spec_decode_metadata(self, num_draft_tokens, cu_num_scheduled_tokens)`](#151-_calc_spec_decode_metadataself-num_draft_tokens-cu_num_scheduled_tokens)
     - 1.5.2 [`_update_states_after_model_execute(self, output_token_ids, scheduler_output)`](#152-_update_states_after_model_executeself-output_token_ids-scheduler_output)
     - 1.5.3 [`_compute_cascade_attn_prefix_lens(self, num_scheduled_tokens, num_computed_tokens, num_common_prefix_blocks)`](#153-_compute_cascade_attn_prefix_lensself-num_scheduled_tokens-num_computed_tokens-num_common_prefix_blocks)
     - 1.5.4 [`get_kv_cache_spec(self)`](#154-get_kv_cache_specself)
     - 1.5.5 [`update_config(self, overrides)`](#155-update_configself-overrides)
     - 1.5.6 [`shutdown(self)`](#156-shutdownself)
2. [max_num_batched_tokens 参数影响分析](#2-max_num_batched_tokens-参数影响分析)
   - 2.1 [调度器 (Scheduler) 相关](#21-调度器-scheduler-相关)
   - 2.2 [GPUModelRunner 缓冲区分配](#22-gpumodelrunner-缓冲区分配)
   - 2.3 [InputBatch 和 BlockTable](#23-inputbatch-和-blocktable)
   - 2.4 [CUDA 图捕获](#24-cuda-图捕获)
   - 2.5 [内存和并行相关](#25-内存和并行相关)
   - 2.6 [其他影响](#26-其他影响)
3. [Encoder Cache 占用空间分析](#3-encoder-cache-占用空间分析)
   - 3.1 [参数传递链路](#31-参数传递链路)
   - 3.2 [关键代码分析](#32-关键代码分析)
     - 3.2.1 [`SchedulerConfig` 中的初始化 (`vllm/config/scheduler.py:248-249`)](#321-schedulerconfig-中的初始化-vllmconfigschedulerpy248-249)
     - 3.2.2 [`compute_mm_encoder_budget` 函数 (`vllm/v1/core/encoder_cache_manager.py:273-320`)](#322-compute_mm_encoder_budget-函数-vllmv1coreencoder_cache_managerpy273-320)
     - 3.2.3 [`EncoderCacheManager` 初始化 (`vllm/v1/core/sched/scheduler.py:226-232`)](#323-encodercachemanager-初始化-vllmv1coreschedschedulerpy226-232)
   - 3.3 [`encoder_cache` 占用空间上限计算](#33-encoder_cache-占用空间上限计算)
     - 3.3.1 [`EncoderCacheManager` 的 `cache_size` 单位](#331-encodercachemanager-的-cache_size-单位)
     - 3.3.2 [每个 encoder embedding 的大小](#332-每个-encoder-embedding-的大小)
     - 3.3.3 [占用空间上限公式](#333-占用空间上限公式)
     - 3.3.4 [具体数值示例](#334-具体数值示例)
     - 3.3.5 [实际代码中的验证](#335-实际代码中的验证)
   - 3.4 [重要说明](#34-重要说明)
     - 3.4.1 [`encoder_cache_size` 的单位是"嵌入数量"而非"字节"](#341-encoder_cache_size-的单位是嵌入数量而非字节)
     - 3.4.2 [`max_num_batched_tokens` 的默认值](#342-max_num_batched_tokens-的默认值)
     - 3.4.3 [`encoder_cache_size` 可能被 `max_tokens_per_mm_item` 覆盖](#343-encoder_cache_size-可能被-max_tokens_per_mm_item-覆盖)
     - 3.4.4 [缓存驱逐机制](#344-缓存驱逐机制)
   - 3.5 [`encoder_cache` 的内存分配机制](#35-encoder_cache-的内存分配机制)
     - 3.5.1 [核心结论](#351-核心结论)
     - 3.5.2 [详细分析](#352-详细分析)
     - 3.5.3 [与 KV Cache 的对比](#353-与-kv-cache-的对比)
     - 3.5.4 [总结](#354-总结)
4. [核心问题解答](#4-核心问题解答)
   - 4.1 [采样器 (Sampler)、投机解码器 (drafter)、输入批处理器 (InputBatch) 详解](#41-采样器-sampler投机解码器-drafter输入批处理器-inputbatch-详解)
     - 4.1.1 [采样器 (Sampler)](#411-采样器-sampler)
     - 4.1.2 [投机解码器 (drafter)](#412-投机解码器-drafter)
     - 4.1.3 [输入批处理器 (InputBatch)](#413-输入批处理器-inputbatch)
   - 4.2 [`load_dummy_weights` 的含义](#42-load_dummy_weights-的含义)
     - 4.2.1 [什么是 `load_dummy_weights`？](#421-什么是-load_dummy_weights)
     - 4.2.2 [虚拟权重 vs 真实权重](#422-虚拟权重-vs-真实权重)
     - 4.2.3 [代码实现](#423-代码实现)
     - 4.2.4 [使用场景](#424-使用场景)
     - 4.2.5 [为什么使用虚拟权重？](#425-为什么使用虚拟权重)
   - 4.3 [`GPUModelRunner.initialize_kv_cache` vs `EngineCore._initialize_kv_caches`](#43-gpumodelrunnerinitialize_kv_cache-vs-enginecore_initialize_kv_caches)
     - 4.3.1 [区别与联系](#431-区别与联系)
     - 4.3.2 [调用关系](#432-调用关系)
     - 4.3.3 [详细说明](#433-详细说明)
     - 4.3.4 [总结](#434-总结)
   - 4.4 [`GPUModelRunner` 与 `Worker` 的关系](#44-gpumodelrunner-与-worker-的关系)
     - 4.4.1 [`GPUModelRunner` 是由 `Worker` 创建的吗？](#441-gpumodelrunner-是由-worker-创建的吗)
     - 4.4.2 [`GPUModelRunner.load_model` 是由 `Worker.load_model` 调用的吗？](#442-gpumodelrunnerload_model-是由-workerload_model-调用的吗)
     - 4.4.3 [完整的调用链](#443-完整的调用链)
     - 4.4.4 [`Worker` 对 `GPUModelRunner` 的其他调用](#444-worker-对-gpumodelrunner-的其他调用)
   - 4.5 [`execute_model` 返回 `None` 的场景](#45-execute_model-返回-none-的场景)
     - 4.5.1 [说法正确性](#451-说法正确性)
     - 4.5.2 [为什么返回 `None`？](#452-为什么返回-none)
     - 4.5.3 [总结](#453-总结)
5. [补充问答](#5-补充问答)
   - 5.1 [池化模型和提示嵌入是什么？](#51-池化模型和提示嵌入是什么)
   - 5.2 ["Sampler 是负责从模型输出的 logits 中采样下一个 token 的模块" 中的 logits 是什么？](#52-sampler-是负责从模型输出的-logits-中采样下一个-token-的模块-中的-logits-是什么)
   - 5.3 [CUDA 图捕获是什么？](#53-cuda-图捕获是什么)
   - 5.4 [EngineCore、Worker 和 GPUModelRunner 三者关系](#54-enginecoreworker-和-gpumodelrunner-三者关系)
   - 5.5 [vLLM 实际执行中会为每个 GPU 分配一个 GPUModelRunner 对象吗？](#55-vllm-实际执行中会为每个-gpu-分配一个-gpumodelrunner-对象吗)
   - 5.6 [`execute_model` 中"计算级联注意力前缀长度"在做什么？](#56-execute_model-中计算级联注意力前缀长度在做什么)
   - 5.7 [`_execute_mm_encoder` 的执行时机](#57-_execute_mm_encoder-的执行时机)
   - 5.8 [`_execute_mm_encoder` 中 `prompt_embeds` 相关代码](#58-_execute_mm_encoder-中-prompt_embeds-相关代码)
   - 5.9 [`_gather_mm_embeddings` 方法分析](#59-_gather_mm_embeddings-方法分析)
   - 5.10 [多模态特征和多模态嵌入的区别是什么？计算过程是什么？](#510-多模态特征和多模态嵌入的区别是什么计算过程是什么)
- [附录：关键调用关系总结](#附录关键调用关系总结)

---

## 1. GPUModelRunner 类方法详细分析

`GPUModelRunner` 是 vLLM v1 引擎中负责 GPU 上模型执行的核心类，继承自 `LoRAModelRunnerMixin`、`KVConnectorModelRunnerMixin` 和 `ECConnectorModelRunnerMixin`。它管理从输入准备、模型前向传播、采样到投机解码的完整推理流水线。

### 1.1 初始化与配置方法

#### 1.1.1 `__init__(self, vllm_config, device)`

**功能**: 初始化 GPUModelRunner 实例，配置所有运行时状态。

**主要初始化内容**:

```python
def __init__(self, vllm_config: VllmConfig, device: torch.device):
    # 1. 配置解析
    self.vllm_config = vllm_config
    self.model_config = vllm_config.model_config      # 模型配置
    self.cache_config = vllm_config.cache_config      # 缓存配置
    self.parallel_config = vllm_config.parallel_config  # 并行配置
    self.scheduler_config = vllm_config.scheduler_config  # 调度配置
    self.speculative_config = vllm_config.speculative_config  # 投机解码配置

    # 2. 设备与数据类型
    self.device = device
    self.dtype = self.model_config.dtype  # 模型数据类型 (如 bfloat16)

    # 3. KV 缓存数据类型
    self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
        cache_config.cache_dtype, self.model_config
    )

    # 4. 模型特性标志
    self.is_pooling_model = model_config.runner_type == "pooling"
    self.enable_prompt_embeds = model_config.enable_prompt_embeds
    self.is_multimodal_raw_input_only_model = model_config.is_multimodal_raw_input_only_model

    # 5. 并行相关
    self.dcp_world_size = self.parallel_config.decode_context_parallel_size
    self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group

    # 6. 调度限制
    self.max_num_tokens = scheduler_config.max_num_batched_tokens
    self.max_num_reqs = scheduler_config.max_num_seqs

    # 7. 多模态支持
    self.mm_registry = MULTIMODAL_REGISTRY
    self.uses_mrope = model_config.uses_mrope
    self.uses_xdrope_dim = model_config.uses_xdrope_dim
    self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(model_config)

    # 8. 编码器长度 (encoder-decoder 模型)
    if self.model_config.is_encoder_decoder:
        self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
    else:
        self.max_encoder_len = 0

    # 9. 异步调度
    self.use_async_scheduling = self.scheduler_config.async_scheduling

    # 10. 采样器初始化
    self.sampler = Sampler(
        logprobs_mode=self.model_config.logprobs_mode,
        use_fp64_gumbel=self.model_config.use_fp64_gumbel,
    )

    # 11. 投机解码器 (drafter) 初始化
    if self.speculative_config and get_pp_group().is_last_rank:
        if self.speculative_config.method == "ngram":
            self.drafter = NgramProposer(self.vllm_config)
        elif self.speculative_config.use_eagle():
            self.drafter = EagleProposer(self.vllm_config, self.device, self)
        elif self.speculative_config.method == "medusa":
            self.drafter = MedusaProposer(...)
        # ... 其他方法
        self.rejection_sampler = RejectionSampler(
            self.sampler, self.speculative_config, self.device
        )

    # 12. 请求状态管理
    self.requests: dict[str, CachedRequestState] = {}
    self.num_prompt_logprobs: dict[str, int] = {}

    # 13. 输入批处理器 (InputBatch) 初始化
    self.input_batch = InputBatch(
        max_num_reqs=self.max_num_reqs,
        max_model_len=max(self.max_model_len, self.max_encoder_len),
        max_num_batched_tokens=self.max_num_tokens,
        device=self.device,
        vocab_size=self.model_config.get_vocab_size(),
        block_sizes=[placeholder_block_size],
        # ... 其他参数
    )

    # 14. CUDA 流和事件 (用于异步调度)
    if self.use_async_scheduling:
        self.async_output_copy_stream = torch.cuda.Stream()
        self.prepare_inputs_event = torch.cuda.Event(blocking=True)

    # 15. CUDA 图批次大小
    if self.compilation_config.cudagraph_capture_sizes:
        self.cudagraph_batch_sizes = sorted(self.compilation_config.cudagraph_capture_sizes)

    # 16. 预分配 GPU 缓冲区
    self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
    self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=self.device)
    self.query_start_loc = self._make_buffer(self.max_num_reqs + 1, dtype=torch.int32)
    self.seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
    # ... 更多缓冲区

    # 17. M-RoPE / XD-RoPE 位置缓冲区
    if self.uses_mrope:
        self.mrope_positions = self._make_buffer((3, self.max_num_tokens + 1), dtype=torch.int64)
    if self.uses_xdrope_dim > 0:
        self.xdrope_positions = self._make_buffer((self.uses_xdrope_dim, self.max_num_tokens + 1), dtype=torch.int64)

    # 18. 编码器缓存
    self.encoder_cache: dict[str, torch.Tensor] = {}  # mm_hash -> encoder_output

    # 19. CUDA 图调度器
    self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

    # 20. 多模态预算
    self.mm_budget = MultiModalBudget(self.vllm_config, self.mm_registry) if self.supports_mm_inputs else None
```

**调用时机**: 由 `GPUWorker` 在引擎启动时创建。

**初始化流程图**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPUModelRunner.__init__                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. 解析配置 (vllm_config)                                      │
│     ├── model_config, cache_config, parallel_config             │
│     ├── scheduler_config, speculative_config                    │
│     └── lora_config, load_config, observability_config          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 设置设备与数据类型                                          │
│     ├── device, dtype, kv_cache_dtype                           │
│     └── max_model_len, max_num_tokens, max_num_reqs             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. 初始化模型特性标志                                          │
│     ├── is_pooling_model, enable_prompt_embeds                  │
│     ├── is_multimodal_raw_input_only_model                      │
│     └── uses_mrope, uses_xdrope_dim, supports_mm_inputs         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. 初始化并行相关                                              │
│     ├── dcp_world_size, dcp_rank                                │
│     └── broadcast_pp_output                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. 初始化采样器 (Sampler)                                      │
│     └── Sampler(logprobs_mode, use_fp64_gumbel)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. 初始化投机解码器 (drafter) [如果启用]                       │
│     ├── NgramProposer / EagleProposer / MedusaProposer / ...    │
│     └── RejectionSampler                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. 初始化输入批处理器 (InputBatch)                             │
│     └── InputBatch(max_num_reqs, max_model_len, ...)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  8. 初始化 CUDA 流和事件 [如果异步调度]                         │
│     ├── async_output_copy_stream                                │
│     └── prepare_inputs_event                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  9. 预分配 GPU 缓冲区                                           │
│     ├── input_ids, positions, query_start_loc, seq_lens         │
│     ├── mrope_positions / xdrope_positions [如果启用]           │
│     └── encoder_cache, intermediate_tensors                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  10. 初始化工具类                                               │
│      ├── CudagraphDispatcher                                    │
│      ├── MultiModalBudget [如果支持多模态]                      │
│      └── LateInteractionRunner                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.1.2 `load_model(self, load_dummy_weights=False)`

**功能**: 加载模型权重到 GPU。

**主要步骤**:
- 通过 `model_loader` 加载模型
- 处理 LoRA 模型加载
- 初始化 EPLB（专家并行负载均衡）状态
- 根据编译配置包装模型（`CUDAGraphWrapper` / `UBatchWrapper` / `BreakableCUDAGraphWrapper`）
- 设置 Eagle3 辅助隐藏状态输出

**调用时机**: 引擎初始化阶段，由 `GPUWorker` 调用。

#### 1.1.3 `initialize_kv_cache(self, kv_cache_config, is_profiling=False)`

**功能**: 初始化 KV 缓存。

**主要步骤**:
- 添加编码器专用层到 KV 缓存配置
- 初始化注意力后端 (`initialize_attn_backend`)
- 准备内核块大小 (`prepare_kernel_block_sizes`)
- 创建元数据构建器 (`initialize_metadata_builders`)
- 重新初始化输入批处理 (`may_reinitialize_input_batch`)
- 分配和重塑 KV 缓存张量 (`initialize_kv_cache_tensors`)
- 注册 KV 传输组

**调用时机**: 在 `profile_run` 之后，由 `GPUWorker` 调用以分配实际的 KV 缓存。

#### 1.1.4 `profile_run(self)`

**功能**: 内存分析运行，用于确定可用的 KV 缓存大小。

**主要步骤**:
- 运行多模态编码器分析（如果支持多模态）
- 执行 `_dummy_run` 进行模型前向传播分析
- 运行 `_dummy_sampler_run` 或 `_dummy_pooler_run` 分析采样/池化内存
- 清理编码器缓存

**调用时机**: 在 `load_model` 之后、`initialize_kv_cache` 之前，由 `GPUWorker` 调用。

#### 1.1.5 `capture_model(self)`

**功能**: 捕获 CUDA 图以优化推理性能。

**主要步骤**:
- 初始化编码器 CUDA 图管理器
- 遍历所有批次描述符，调用 `_capture_cudagraphs` 捕获图
- 捕获编码器 CUDA 图（如果启用）
- 锁定工作区防止执行期间调整大小

**调用时机**: 在 `initialize_kv_cache` 之后，由 `GPUWorker` 调用。

### 1.2 核心推理执行方法

#### 1.2.1 `execute_model(self, scheduler_output, intermediate_tensors=None)`

**功能**: 执行模型前向传播的核心入口。

**主要步骤**:
1. 更新批次状态 (`_update_states`)
2. 处理多模态编码器（如果是 EC 传输生产者）
3. 准备输入 (`_prepare_inputs`)
4. 计算级联注意力前缀长度 (`_compute_cascade_attn_prefix_lens`)
5. 确定批次执行和填充 (`_determine_batch_execution_and_padding`)
6. 获取槽映射 (`_get_slot_mappings`)
7. 构建注意力元数据 (`_build_attention_metadata`)
8. 预处理输入 (`_preprocess`)
9. 执行模型前向传播 (`_model_forward`)
10. 处理输出（池化/采样/中间张量）
11. 保存执行状态到 `execute_model_state`

**调用时机**: 每个推理步骤由 `GPUWorker` 调用。返回 `None` 表示需要继续调用 `sample_tokens()`。

**流程图**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         execute_model 流程图                                │
└─────────────────────────────────────────────────────────────────────────────┘

开始
  │
  ▼
┌─────────────────┐
│ 1. 状态检查      │─── execute_model_state 不为 None? ───▶ 抛出异常
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ 2. 清理 routed  │
│    experts 缓冲区│
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ 3. N-gram GPU   │─── 使用 ngram_gpu? ───▶ 复制 scheduler_output
│    特殊处理      │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ 4. KV 传输组处理 │─── has_kv_transfer_group? ───▶ handle_preemptions
└─────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. 预处理阶段 (synchronize_input_prep)                          │
│                                                                 │
│  ┌─────────────┐                                                │
│  │ 5.1 _update │─── 更新批次状态，返回 deferred_state_corrections_fn│
│  │   _states   │                                                │
│  └─────────────┘                                                │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐     ┌─────────────────┐                       │
│  │ 5.2 EC 传输  │───▶│ 是 EC 生产者?    │───▶ 执行 _execute_mm_encoder│
│  │   处理      │     │                 │     返回空输出          │
│  └─────────────┘     └─────────────────┘                       │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐     ┌─────────────────┐                       │
│  │ 5.3 空批次   │───▶│ num_scheduled_  │───▶ 返回 EMPTY_MODEL_RUNNER_OUTPUT│
│  │   处理      │     │ tokens == 0?    │     或 kv_connector_no_forward│
│  └─────────────┘     └─────────────────┘                       │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐                                                │
│  │ 5.4 _prepare│─── 准备输入张量，返回 logits_indices 和         │
│  │   _inputs   │    spec_decode_metadata                       │
│  └─────────────┘                                                │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐                                                │
│  │ 5.5 级联注意力│─── 计算 cascade_attn_prefix_lens              │
│  │   前缀长度   │                                                │
│  └─────────────┘                                                │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐                                                │
│  │ 5.6 确定批次 │─── 返回 cudagraph_mode, batch_desc,             │
│  │   执行和填充 │    should_ubatch, num_tokens_across_dp          │
│  └─────────────┘                                                │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐                                                │
│  │ 5.7 获取槽   │─── 返回 slot_mappings_by_group, slot_mappings   │
│  │   映射      │                                                │
│  └─────────────┘                                                │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐                                                │
│  │ 5.8 构建注意力│─── 返回 attn_metadata,                         │
│  │   元数据    │    spec_decode_common_attn_metadata             │
│  └─────────────┘                                                │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐                                                │
│  │ 5.9 _prepro-│─── 返回 input_ids, inputs_embeds, positions,    │
│  │   cess      │    intermediate_tensors, model_kwargs,          │
│  │             │    ec_connector_output                          │
│  └─────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────┐
│ 6. KV scales    │─── calculate_kv_scales? ───▶ cudagraph_mode = NONE
│    计算         │
└─────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. 模型前向传播 (set_forward_context)                           │
│                                                                 │
│  ┌─────────────┐                                                │
│  │ _model_for- │─── 调用 self.model(...) 执行前向传播            │
│  │   ward      │    返回 model_output                            │
│  └─────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. 后处理阶段                                                   │
│                                                                 │
│  ┌─────────────┐                                                │
│  │ 8.1 处理辅助 │─── use_aux_hidden_state_outputs?               │
│  │   隐藏状态   │    分离 hidden_states 和 aux_hidden_states      │
│  └─────────────┘                                                │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐     ┌─────────────────┐                        │
│  │ 8.2 PP 并行  │───▶│ 非最后 rank?     │───▶ 返回 intermediate_tensors│
│  │   处理      │     │                 │                        │
│  └─────────────┘     └─────────────────┘                        │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐     ┌─────────────────┐                        │
│  │             │───▶│ 是池化模型?      │───▶ 返回 _pool() 输出   │
│  │             │     │                 │                        │
│  └─────────────┘     └─────────────────┘                        │
│        │                                                        │
│        ▼                                                        │
│  ┌─────────────┐                                                │
│  │ 计算 logits  │─── sample_hidden_states = hidden_states[logits_indices]│
│  │             │    logits = model.compute_logits(sample_hidden_states)  │
│  └─────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────┐
│ 9. 保存执行状态  │─── execute_model_state = ExecuteModelState(...)
│    到 execute_  │
│    model_state  │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ 10. 应用延迟的   │─── deferred_state_corrections_fn? ───▶ 调用修正函数
│    状态修正     │
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ 11. 返回 None   │─── 表示需要调用 sample_tokens() 进行采样
└─────────────────┘
```

#### 1.2.2 `sample_tokens(self, grammar_output)`

**功能**: 从模型输出中采样 token。

**主要步骤**:
1. 解包 `execute_model_state`
2. 应用语法位掩码（结构化输出）
3. 执行采样 (`_sample`)
4. 更新模型执行后状态 (`_update_states_after_model_execute`)
5. 广播采样 token（PP 并行）
6. 提出投机解码 token (`propose_draft_token_ids`)
7. 执行簿记同步 (`_bookkeeping_sync`)
8. 构建 `ModelRunnerOutput` 或 `AsyncGPUModelRunnerOutput`

**调用时机**: 在 `execute_model` 返回 `None` 后，由 `GPUWorker` 调用。

#### 1.2.3 `_prepare_inputs(self, scheduler_output, num_scheduled_tokens)`

**功能**: 准备模型输入张量。

**主要步骤**:
- 提交块表 (`commit_block_table`)
- 计算累积和与范围 (`_get_cumsum_and_arange`)
- 计算位置索引
- 处理 M-RoPE/XD-RoPE 位置
- 提取 token ID 到 `input_ids`
- 准备注意力元数据（`query_start_loc`, `seq_lens` 等）
- 计算槽映射 (`compute_slot_mapping`)
- 准备输入 ID (`_prepare_input_ids`)
- 处理投机解码元数据 (`_calc_spec_decode_metadata`)

**调用时机**: 在 `execute_model` 中被调用。

#### 1.2.4 `_preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors)`

**功能**: 预处理输入，准备模型前向传播所需的张量。

**主要步骤**:
- 处理多模态输入（执行编码器、收集嵌入）
- 处理提示嵌入 (`prompt_embeds`)
- 准备位置张量
- 同步和收集中间张量（PP 并行）
- 处理编码器-解码器模型的编码器输出

**调用时机**: 在 `execute_model` 中被调用，位于 `_prepare_inputs` 之后。

#### 1.2.5 `_model_forward(self, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)`

**功能**: 调用模型的前向传播。

**主要步骤**:
- 直接调用 `self.model(...)`

**调用时机**: 在 `execute_model` 中被调用，位于 `_preprocess` 之后。

#### 1.2.6 `_sample(self, logits, spec_decode_metadata)`

**功能**: 执行 token 采样。

**主要步骤**:
- 更新异步输出 token ID
- 无投机解码时直接调用 `self.sampler`
- 有投机解码时调用 `self.rejection_sampler`

**调用时机**: 在 `sample_tokens` 中被调用。

#### 1.2.7 `_bookkeeping_sync(self, scheduler_output, sampler_output, logits, hidden_states, num_scheduled_tokens)`

**功能**: 同步簿记，更新请求状态。

**主要步骤**:
- 计算 logits 中的 NaN 数量
- 处理丢弃的采样 token
- 更新 `input_batch` 中的 token ID
- 计算提示对数概率 (`_get_prompt_logprobs_dict`)
- 返回各种输出数据

**调用时机**: 在 `sample_tokens` 中被调用，位于 `_sample` 之后。

#### 1.2.8 `propose_draft_token_ids(self, scheduler_output, sampled_token_ids, sampling_metadata, hidden_states, sample_hidden_states, aux_hidden_states, spec_decode_metadata, common_attn_metadata, slot_mappings)`

**功能**: 提出投机解码的草稿 token。

**主要步骤**:
- 根据投机解码方法（ngram、Eagle、Medusa、DFlash 等）调用相应的提议器
- 准备下一个 token ID (`prepare_next_token_ids_padded` / `prepare_next_token_ids_cpu`)
- 调用 `drafter.propose` 生成草稿 token

**调用时机**: 在 `sample_tokens` 中被调用，当启用投机解码时。

### 1.3 辅助执行方法

#### 1.3.1 `_update_states(self, scheduler_output)`

**功能**: 根据调度器输出更新缓存状态和持久批次。

**主要步骤**:
1. 移除已完成的请求
2. 零化新分配的缓存块
3. 复制 KV 缓存块
4. 处理编码器缓存生命周期
5. 添加新请求到缓存状态
6. 更新运行/恢复请求的状态
7. 压缩批次状态 (`condense`)
8. 重新排序批次 (`_may_reorder_batch`)

**调用时机**: 在 `execute_model` 开头被调用。

**流程图**:

```
┌─────────────────────────────────────────────────────────────────┐
│                      _update_states 流程                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. 移除已完成的请求                                             │
│    ├── 从 self.requests 中移除                                  │
│    ├── 从 self.input_batch 中移除                               │
│    └── 通知 late_interaction_runner                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 零化新分配的缓存块                                           │
│    └── _zero_block_ids(new_block_ids_to_zero)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 复制 KV 缓存块                                               │
│    └── copy_kv_cache_blocks_inplace(kv_caches, ...)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. 释放编码器缓存                                               │
│    └── _process_encoder_cache_scheduler_output()                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. 移除未调度的请求                                             │
│    └── input_batch.remove_request(req_id)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. 添加新请求                                                   │
│    ├── 创建 CachedRequestState                                  │
│    ├── 初始化 M-RoPE / XD-RoPE 位置                             │
│    └── 添加到 reqs_to_add                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. 更新运行/恢复请求的状态                                      │
│    ├── 更新 num_computed_tokens                                 │
│    ├── 更新 output_token_ids                                    │
│    ├── 更新 block_ids                                           │
│    └── 更新 input_batch                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. 添加新/恢复请求到持久批次                                    │
│    └── input_batch.add_request(request)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. 压缩批次状态                                                 │
│    └── input_batch.condense()                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 10. 重新排序批次                                                │
│     └── _may_reorder_batch(scheduler_output)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 11. 刷新批次元数据                                              │
│     └── input_batch.refresh_metadata()                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 12. 更新 N-gram GPU 张量 [如果启用]                             │
│     └── update_ngram_gpu_tensors_incremental(...)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 13. 返回延迟修正函数 [如果有投机解码修正]                       │
│     └── correct_spec_decode_token_counts()                      │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.3.2 `_build_attention_metadata(self, num_tokens, num_reqs, max_query_len, ...)`

**功能**: 构建注意力元数据。

**主要步骤**:
- 计算最大序列长度
- 获取块表和槽映射
- 构建公共注意力元数据 (`CommonAttentionMetadata`)
- 为每个 KV 缓存组构建注意力元数据
- 处理投机解码的公共注意力元数据

**调用时机**: 在 `execute_model` 中被调用，位于 `_get_slot_mappings` 之后。

#### 1.3.3 `_determine_batch_execution_and_padding(self, num_tokens, num_reqs, num_scheduled_tokens_np, max_num_scheduled_tokens, use_cascade_attn, ...)`

**功能**: 确定批次执行方式和填充。

**主要步骤**:
- 检查是否为均匀解码 (`_is_uniform_decode`)
- 计算 LoRA 状态
- 序列并行填充 (`_pad_for_sequence_parallelism`)
- 调度 CUDA 图 (`cudagraph_dispatcher.dispatch`)
- 跨 DP 协调批次 (`coordinate_batch_across_dp`)

**调用时机**: 在 `execute_model` 中被调用，位于 `_prepare_inputs` 之后。

#### 1.3.4 `_get_slot_mappings(self, num_tokens_padded, num_reqs_padded, num_tokens_unpadded, ubatch_slices)`

**功能**: 构建槽映射。

**主要步骤**:
- 为每个 KV 缓存组获取槽映射
- 填充未使用的槽为 -1
- 处理 ubatch 切片

**调用时机**: 在 `execute_model` 中被调用，位于 `_determine_batch_execution_and_padding` 之后。

#### 1.3.5 `_execute_mm_encoder(self, scheduler_output)`

**功能**: 执行多模态编码器。

**主要步骤**:
- 批量处理调度器中的多模态输入
- 处理提示嵌入（直接注入编码器缓存）
- 运行编码器（支持 LoRA、CUDA 图）
- 缓存编码器输出

**调用时机**: 在 `_preprocess` 中被调用，当支持多模态且为第一 PP rank 时。

**流程图**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    _execute_mm_encoder 流程                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. _batch_mm_inputs_from_scheduler                              │
│    └── 从 scheduler_output 获取 mm_hashes, mm_kwargs, mm_lora_refs│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 处理 prompt_embeds                                           │
│    └── 直接注入 encoder_cache，无需编码                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 处理 LoRA [如果启用]                                         │
│    └── 构建 tower/connector LoRA 映射                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. 批量执行编码器                                               │
│    ├── 视频特殊处理: 顺序编码                                    │
│    ├── CUDA 图: encoder_cudagraph_manager.execute()             │
│    └── 普通: model.embed_multimodal()                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. 缓存编码器输出                                               │
│    └── _cache_encoder_output(mm_hash, output)                   │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.3.6 `_gather_mm_embeddings(self, scheduler_output, shift_computed_tokens=0)`

**功能**: 收集多模态嵌入。

**主要步骤**:
- 遍历批次中的请求
- 从编码器缓存中获取嵌入
- 处理 M-RoPE/XD-RoPE 位置同步
- 返回嵌入和掩码

**调用时机**: 在 `_preprocess` 中被调用，位于 `_execute_mm_encoder` 之后。

**流程图**:

```
┌─────────────────────────────────────────────────────────────────┐
│                   _gather_mm_embeddings 流程                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. 初始化                                                       │
│    ├── mm_embeds = []                                           │
│    └── is_mm_embed = zeros(total_num_scheduled_tokens, bool)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 遍历批次中的每个请求                                         │
│    for req_id in self.input_batch.req_ids:                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 获取当前窗口内的多模态特征                                   │
│    lo, hi = get_mm_features_in_window(                          │
│        mm_features,                                             │
│        start=num_computed_tokens,                               │
│        end=num_computed_tokens + num_scheduled_tokens           │
│    )                                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. 遍历每个多模态特征                                           │
│    for i in range(lo, hi):                                      │
│      ├── 计算嵌入范围 (start_idx, end_idx)                      │
│      ├── 从缓存获取编码器输出 (_get_encoder_output_from_cache)   │
│      ├── 提取嵌入 (encoder_output[curr_embeds_start:curr_embeds_end])│
│      ├── 标记嵌入位置 (is_mm_embed[...] = True)                 │
│      └── 添加到 mm_embeds_req                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. 处理 M-RoPE 位置同步 [如果启用多模态剪枝]                    │
│    └── recompute_mrope_positions()                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. 同步 M-RoPE / XD-RoPE 位置                                   │
│    └── _calc_mrope_positions() / _calc_xdrope_positions()       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. 返回 (mm_embeds, is_mm_embed)                                │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.3.7 `_pool(self, hidden_states, num_scheduled_tokens, num_scheduled_tokens_np, kv_connector_output)`

**功能**: 执行池化操作（用于嵌入模型）。

**主要步骤**:
- 构建池化元数据
- 调用 `model.pooler`
- 处理后期交互运行器
- 返回池化输出

**调用时机**: 在 `execute_model` 中被调用，当模型为池化模型时。

### 1.4 虚拟运行与分析方法

#### 1.4.1 `_dummy_run(self, num_tokens, cudagraph_runtime_mode, force_attention, uniform_decode, ...)`

**功能**: 执行虚拟前向传播，用于预热、分析和 CUDA 图捕获。

**主要步骤**:
- 构建虚拟批次
- 确定批次执行和填充
- 构建注意力元数据（可选）
- 执行模型前向传播
- 处理投机解码虚拟运行
- 注册层间 NVTX 钩子

**调用时机**:
- `profile_run` 中调用（内存分析）
- `capture_model` 中调用（CUDA 图捕获）
- `execute_model` 中调用（DP 协调时）

#### 1.4.2 `_dummy_sampler_run(self, hidden_states)`

**功能**: 执行虚拟采样器运行，用于内存分析。

**主要步骤**:
- 使用随机张量避免特殊值
- 计算 logits
- 运行采样器
- 运行拒绝采样器（如果启用投机解码）

**调用时机**: 在 `profile_run` 中被调用。

#### 1.4.3 `_dummy_pooler_run(self, hidden_states)`

**功能**: 执行虚拟池化运行，用于内存分析。

**主要步骤**:
- 遍历所有支持的池化任务
- 运行每个任务的虚拟池化
- 返回最大输出的任务结果

**调用时机**: 在 `profile_run` 中被调用，当模型为池化模型时。

### 1.5 工具与辅助方法

#### 1.5.1 `_calc_spec_decode_metadata(self, num_draft_tokens, cu_num_scheduled_tokens)`

**功能**: 计算投机解码元数据。

**主要步骤**:
- 计算 logits 索引
- 计算奖励 logits 索引
- 计算草稿 logits 索引
- 提取草稿 token ID

**调用时机**: 在 `_prepare_inputs` 中被调用，当启用投机解码时。

#### 1.5.2 `_update_states_after_model_execute(self, output_token_ids, scheduler_output)`

**功能**: 模型执行后更新缓存状态（用于 MTP/EAGLE 混合模型）。

**主要步骤**:
- 计算接受的 token 数量
- 处理 Mamba 状态对齐

**调用时机**: 在 `sample_tokens` 中被调用，位于 `_sample` 之后。

#### 1.5.3 `_compute_cascade_attn_prefix_lens(self, num_scheduled_tokens, num_computed_tokens, num_common_prefix_blocks)`

**功能**: 计算级联注意力前缀长度。

**主要步骤**:
- 遍历所有 KV 缓存组和注意力组
- 计算每个组的级联注意力前缀长度

**调用时机**: 在 `execute_model` 中被调用，当启用级联注意力时。

#### 1.5.4 `get_kv_cache_spec(self)`

**功能**: 生成 KV 缓存规格。

**主要步骤**:
- 解析静态前向上下文中的注意力模块
- 处理 KV 共享层
- 返回每个层的 KV 缓存规格

**调用时机**: 在 `initialize_kv_cache` 和 `_init_minimal_kv_cache_for_profiling` 中被调用。

#### 1.5.5 `update_config(self, overrides)`

**功能**: 更新配置（如 `load_config`, `model_config`）。

**调用时机**: 动态更新配置时由外部调用。

#### 1.5.6 `shutdown(self)`

**功能**: 释放 GPU 资源。

**主要步骤**:
- 清理分析 KV 缓存
- 清除 CUDA 图
- 清理静态前向上下文
- 重置工作区管理器

**调用时机**: 引擎关闭时由 `GPUWorker` 调用。

---

## 2. max_num_batched_tokens 参数影响分析

`max_num_batched_tokens` 是 vLLM 中最核心的调度参数之一，除了影响 `EncoderCacheManager` 的 `encoder_cache_size` 外，还影响以下多个方面：

### 2.1 调度器 (Scheduler) 相关

| 影响项 | 说明 | 代码位置 |
|--------|------|----------|
| **`max_num_scheduled_tokens`** | 调度器每步最多调度的 token 数，默认等于 `max_num_batched_tokens` | `vllm/v1/core/sched/scheduler.py:111-114` |
| **`max_num_encoder_input_tokens`** | 编码器计算预算，默认等于 `max_num_batched_tokens` | `vllm/config/scheduler.py:248` |
| **`token_budget`** | 每步调度的 token 预算上限 | `vllm/v1/core/sched/scheduler.py:454` |
| **投机解码调度** | `max_num_scheduled_tokens = max_num_batched_tokens - scheduled_token_delta` | `vllm/config/vllm.py:1684-1688` |

```python
# vllm/v1/core/sched/scheduler.py:111-114
self.max_num_scheduled_tokens = (
    self.scheduler_config.max_num_scheduled_tokens
    if self.scheduler_config.max_num_scheduled_tokens is not None
    else self.scheduler_config.max_num_batched_tokens
)

# vllm/config/vllm.py:1684-1688 (投机解码时)
max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
if self.scheduler_config.max_num_scheduled_tokens is None:
    self.scheduler_config.max_num_scheduled_tokens = (
        max_num_batched_tokens - scheduled_token_delta
    )
```

### 2.2 GPUModelRunner 缓冲区分配

`max_num_batched_tokens` 直接决定了 `GPUModelRunner` 中多个 GPU 缓冲区的大小：

| 缓冲区 | 大小 | 说明 |
|--------|------|------|
| `input_ids` | `(max_num_tokens,)` | 输入 token IDs |
| `positions` | `(max_num_tokens,)` | 位置编码 |
| `req_indices` | `(max_num_tokens,)` | 请求索引 |
| `inputs_embeds` | `(max_num_tokens, hidden_size)` | 输入嵌入 |
| `is_token_ids` | `(max_num_tokens,)` | 标记是否为 token IDs |
| `mrope_positions` | `(3, max_num_tokens + 1)` | M-RoPE 位置 |
| `xdrope_positions` | `(xdrope_dim, max_num_tokens + 1)` | XD-RoPE 位置 |
| `query_pos` | `(arange_size,)` | 查询位置 |

```python
# vllm/v1/worker/gpu_model_runner.py:497
self.max_num_tokens = scheduler_config.max_num_batched_tokens

# vllm/v1/worker/gpu_model_runner.py:755-792
self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=self.device)
self.req_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
self.inputs_embeds = self._make_buffer(
    self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
)
self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
```

### 2.3 InputBatch 和 BlockTable

| 影响项 | 说明 | 代码位置 |
|--------|------|----------|
| **`InputBatch.max_num_batched_tokens`** | 输入批处理器的最大 token 数 | `vllm/v1/worker/gpu_input_batch.py:97,122` |
| **`BlockTable.slot_mapping`** | 槽映射缓冲区大小 | `vllm/v1/worker/block_table.py:86-88` |
| **`MultiGroupBlockTable`** | 多组块表的最大 token 数 | `vllm/v1/worker/block_table.py:249` |

```python
# vllm/v1/worker/gpu_input_batch.py:174-184
self.block_table = MultiGroupBlockTable(
    max_num_reqs=max_num_reqs,
    max_num_batched_tokens=max_num_batched_tokens,  # 传入
    pin_memory=PIN_MEMORY,
    device=device,
    block_sizes=block_sizes,
    kernel_block_sizes=kernel_block_sizes,
    max_num_blocks=max_num_blocks_per_req,
    cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    slot_mapping_modes=slot_mapping_modes,
)

# vllm/v1/worker/block_table.py:86-88
self.slot_mapping = self._make_buffer(
    self.max_num_batched_tokens, dtype=torch.int64
)
```

### 2.4 CUDA 图捕获

| 影响项 | 说明 | 代码位置 |
|--------|------|----------|
| **`max_cudagraph_capture_size`** | CUDA 图最大捕获大小 | `vllm/config/vllm.py:1779-1780` |
| **`cudagraph_capture_sizes`** | CUDA 图捕获大小列表 | `vllm/config/vllm.py:1820-1825` |
| **编译范围上限** | `compile_range_end = max_num_batched_tokens` | `vllm/config/vllm.py:1896` |

```python
# vllm/config/vllm.py:1779-1780
max_num_tokens = self.scheduler_config.max_num_batched_tokens
max_cudagraph_capture_size = min(max_num_tokens, max_cudagraph_capture_size)

# vllm/config/vllm.py:1820-1825
if (
    max_num_tokens <= max_cudagraph_capture_size
    and max_num_tokens not in cudagraph_capture_sizes
):
    cudagraph_capture_sizes.append(max_num_tokens)
```

### 2.5 内存和并行相关

| 影响项 | 说明 | 代码位置 |
|--------|------|----------|
| **`max_in_flight_tokens`** | 在途 token 数上限 | `vllm/config/vllm.py:515-516` |
| **`max_concurrent_batches`** | 最大并发批次数 | `vllm/config/vllm.py:496-506` |
| **LoRA 静态缓冲区** | LoRA 创建的静态缓冲区大小 | `vllm/config/scheduler.py:219` |
| **Inductor 索引类型** | 32-bit 或 64-bit 索引整数 | `vllm/config/scheduler.py:223-225` |

```python
# vllm/config/vllm.py:515-516
return (
    self.max_concurrent_batches * self.scheduler_config.max_num_batched_tokens
)

# vllm/config/scheduler.py:217-226
# max_num_batched_tokens need to be included in the hash due
# to two reasons:
# 1. LoRA creates static buffers based on max_num_batched_tokens.
#   The tensor sizes and strides get captured in the torch.compile
#   graph explicitly.
# 2. Inductor decides whether using 32-bit or 64-bit indexing integer
#   based on the data sizes. `max_num_batched_tokens` has an
#   impact on that.
```

### 2.6 其他影响

| 影响项 | 说明 | 代码位置 |
|--------|------|----------|
| **Mamba 缓存对齐** | `block_size <= max_num_batched_tokens` | `vllm/config/vllm.py:2239` |
| **序列并行填充** | `_pad_for_sequence_parallelism` | `vllm/v1/worker/gpu_model_runner.py:3490` |
| **DP 协调** | `coordinate_batch_across_dp` | `vllm/v1/worker/gpu_model_runner.py:3968` |
| **Routed Experts** | `max_num_batched_tokens` 用于路由专家捕获 | `vllm/v1/worker/gpu_model_runner.py:7583` |

---

## 3. Encoder Cache 占用空间分析

### 3.1 参数传递链路

```
EngineArgs.max_num_batched_tokens
    ↓
SchedulerConfig.max_num_batched_tokens
    ↓
SchedulerConfig.encoder_cache_size = max_num_batched_tokens  (scheduler.py:249)
    ↓
MultiModalBudget.encoder_cache_size = max(encoder_cache_size, max_tokens_per_mm_item)  (encoder_budget.py:117-120)
    ↓
EncoderCacheManager(cache_size=encoder_cache_size)  (scheduler.py:226-232)
```

### 3.2 关键代码分析

#### 3.2.1 `SchedulerConfig` 中的初始化 (`vllm/config/scheduler.py:248-249`)

```python
self.max_num_encoder_input_tokens = self.max_num_batched_tokens
self.encoder_cache_size = self.max_num_batched_tokens
```

**说明**: `encoder_cache_size` 默认等于 `max_num_batched_tokens`。

#### 3.2.2 `compute_mm_encoder_budget` 函数 (`vllm/v1/core/encoder_cache_manager.py:273-320`)

```python
def compute_mm_encoder_budget(
    scheduler_config: "SchedulerConfig",
    mm_max_toks_per_item: Mapping[str, int],
) -> tuple[int, int]:
    max_tokens_per_mm_item = max(mm_max_toks_per_item.values())

    encoder_compute_budget = max(
        scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item
    )
    encoder_cache_size = max(
        scheduler_config.encoder_cache_size, max_tokens_per_mm_item
    )

    return encoder_compute_budget, encoder_cache_size
```

**说明**: `encoder_cache_size` 取 `max_num_batched_tokens` 和 `max_tokens_per_mm_item` 中的较大值。

#### 3.2.3 `EncoderCacheManager` 初始化 (`vllm/v1/core/sched/scheduler.py:226-232`)

```python
encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
self.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)
```

**说明**: `EncoderCacheManager` 的 `cache_size` 就是 `encoder_cache_size`。

### 3.3 `encoder_cache` 占用空间上限计算

#### 3.3.1 `EncoderCacheManager` 的 `cache_size` 单位

根据 `EncoderCacheManager` 的文档字符串 (`vllm/v1/core/encoder_cache_manager.py:48-54`):

```python
"""
Args:
    cache_size: Limit the size of the cache, measured by the number of
                encoder embeddings from the input sequence.

Attributes:
    cache_size: Total cache capacity in encoder embeddings.
    num_free_slots: Current available cache capacity in encoder embeddings.
"""
```

**关键**: `cache_size` 的单位是 **encoder embeddings 的数量**，而不是字节数。

#### 3.3.2 每个 encoder embedding 的大小

每个 encoder embedding 是一个形状为 `(feature_size, hidden_size)` 的 2D 张量：
- `feature_size`: 可变，取决于多模态输入（如图像块数量）
- `hidden_size`: 模型的隐藏层大小（如 4096、8192 等）

从 `MultiModalEmbeddings` 的定义 (`vllm/model_executor/models/interfaces.py:63-70`):
```python
MultiModalEmbeddings: TypeAlias = list[Tensor] | Tensor | tuple[Tensor, ...]
"""
The output embeddings must be one of the following formats:
- A list or tuple of 2D tensors, where each tensor corresponds to
    each input multimodal data item (e.g, image).
- A single 3D tensor, with the batch dimension grouping the 2D tensors.
"""
```

#### 3.3.3 占用空间上限公式

```
最大占用空间（字节） = encoder_cache_size × hidden_size × dtype_size
```

其中：
- `encoder_cache_size`: 由 `max_num_batched_tokens` 决定（默认等于它）
- `hidden_size`: 模型隐藏层大小
- `dtype_size`: 数据类型大小（通常为 2 字节，即 bfloat16/float16）

#### 3.3.4 具体数值示例

| 场景 | `max_num_batched_tokens` | `hidden_size` | `dtype_size` | 最大占用空间 |
|------|-------------------------|---------------|--------------|-------------|
| 默认配置 (A100) | 8192 | 4096 | 2 bytes | 8192 × 4096 × 2 = **64 MB** |
| 默认配置 (H100) | 16384 | 4096 | 2 bytes | 16384 × 4096 × 2 = **128 MB** |
| 大模型 (H100) | 16384 | 8192 | 2 bytes | 16384 × 8192 × 2 = **256 MB** |
| 自定义配置 | 32768 | 8192 | 2 bytes | 32768 × 8192 × 2 = **512 MB** |

#### 3.3.5 实际代码中的验证

在 `GPUModelRunner` 中，`encoder_cache` 是一个字典：
```python
# vllm/v1/worker/gpu_model_runner.py:564
self.encoder_cache: dict[str, torch.Tensor] = {}
```

每个缓存的 encoder output 是一个张量，形状为 `(num_embeds, hidden_size)`。

`EncoderCacheManager` 通过 `num_free_slots` 跟踪可用的嵌入数量：
```python
# vllm/v1/core/encoder_cache_manager.py:68-71
def __init__(self, cache_size: int):
    self.cache_size = cache_size
    self.num_free_slots = cache_size
    self.num_freeable_slots = cache_size
```

### 3.4 重要说明

#### 3.4.1 `encoder_cache_size` 的单位是"嵌入数量"而非"字节"

`EncoderCacheManager` 的 `cache_size` 限制的是 **encoder embeddings 的数量**，而不是直接的内存字节数。这意味着：

- 每个多模态输入项（如一张图片）会产生一定数量的 embeddings
- `cache_size` 限制的是所有这些 embeddings 的总数量
- 实际内存占用 = 嵌入数量 × `hidden_size` × `dtype_size`

#### 3.4.2 `max_num_batched_tokens` 的默认值

根据 `vllm/engine/arg_utils.py:2443-2462`：

```python
if device_memory >= 70 * GiB_bytes and "a100" not in device_name:
    # For GPUs like H100 and MI300x, use larger default values.
    default_max_num_batched_tokens = {
        UsageContext.LLM_CLASS: 16384,
        UsageContext.OPENAI_API_SERVER: 8192,
    }
else:
    default_max_num_batched_tokens = {
        UsageContext.LLM_CLASS: 8192,
        UsageContext.OPENAI_API_SERVER: 2048,
    }
```

#### 3.4.3 `encoder_cache_size` 可能被 `max_tokens_per_mm_item` 覆盖

从 `compute_mm_encoder_budget` 函数：
```python
encoder_cache_size = max(
    scheduler_config.encoder_cache_size, max_tokens_per_mm_item
)
```

如果单个多模态项的最大 token 数超过 `max_num_batched_tokens`，则 `encoder_cache_size` 会被提升以容纳至少一个完整的多模态项。

#### 3.4.4 缓存驱逐机制

`EncoderCacheManager` 使用 LRU 驱逐策略：
```python
# vllm/v1/core/encoder_cache_manager.py:178-183
while num_embeds > self.num_free_slots:
    mm_hash, num_free_embeds = self.freeable.popitem(last=False)
    del self.cached[mm_hash]
    self.freed.append(mm_hash)
    self.num_free_slots += num_free_embeds
```

当缓存满时，最旧的未被引用的条目会被驱逐。

### 3.5 `encoder_cache` 的内存分配机制

#### 3.5.1 核心结论

**`encoder_cache` 的内存占用是动态增长的，不是启动时固定分配的。**

`max_num_batched_tokens` 仅仅定义了一个**上限**（通过 `encoder_cache_size`），实际的内存占用是随着多模态请求的处理而**动态增加**的。

#### 3.5.2 详细分析

##### 3.5.2.1 `EncoderCacheManager` 是逻辑管理器，不分配物理内存

```python
# vllm/v1/core/encoder_cache_manager.py:68-71
class EncoderCacheManager:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size          # 逻辑上限（嵌入数量）
        self.num_free_slots = cache_size      # 当前可用槽位
        self.num_freeable_slots = cache_size  # 可回收槽位

        self.cached: dict[str, set[str]] = {}      # mm_hash -> request_ids
        self.request_cached_ids: dict[str, set[int]] = {}  # request_id -> input_ids
        self.freeable: OrderedDict[str, int] = OrderedDict()  # mm_hash -> num_embeds
        self.freed: list[str] = []  # 已驱逐的 mm_hash
```

**关键点**: `EncoderCacheManager` 只维护**逻辑状态**（哪些 mm_hash 被缓存、被哪些请求引用、可用槽位数量），**不分配任何 GPU 内存**。

##### 3.5.2.2 `GPUModelRunner.encoder_cache` 是物理存储，动态增长

```python
# vllm/v1/worker/gpu_model_runner.py:564
self.encoder_cache: dict[str, torch.Tensor] = {}  # mm_hash -> encoder_output
```

**关键点**:
- `encoder_cache` 是一个普通的 Python 字典
- 键是 `mm_hash`（多模态数据的哈希值）
- 值是 `torch.Tensor`（编码器输出的嵌入向量）
- **初始为空**，随着多模态请求的处理而动态添加条目

##### 3.5.2.3 内存分配流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    多模态请求处理流程                           │
└─────────────────────────────────────────────────────────────────┘

1. 调度器调度多模态请求
   └── EncoderCacheManager.can_allocate() 检查逻辑空间
       ├── 如果空间不足，驱逐旧条目（逻辑驱逐）
       └── 返回 True/False

2. GPUModelRunner._execute_mm_encoder()
   └── 执行编码器模型，生成嵌入向量
       └── _cache_encoder_output(mm_hash, output)
           └── self.encoder_cache[mm_hash] = output  # 物理内存分配

3. GPUModelRunner._gather_mm_embeddings()
   └── 从 encoder_cache 中提取嵌入
       └── encoder_output = self.encoder_cache.get(mm_hash)

4. 请求完成或缓存驱逐
   └── SchedulerOutput.free_encoder_mm_hashes
       └── GPUModelRunner._process_encoder_cache_scheduler_output()
           └── self.encoder_cache.pop(mm_hash, None)  # 物理内存释放
```

##### 3.5.2.4 代码证据

**物理内存分配**（动态）:
```python
# vllm/v1/worker/gpu_model_runner.py:2972-2981
def _cache_encoder_output(
    self,
    mm_hash: str,
    output: torch.Tensor,
    scheduler_output: "SchedulerOutput",
) -> None:
    """Store an encoder output for later multimodal embedding gather."""
    del scheduler_output
    self.encoder_cache[mm_hash] = output  # 动态添加，分配 GPU 内存
    self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

**物理内存释放**（动态）:
```python
# vllm/v1/worker/gpu_model_runner.py:1170-1176
def _process_encoder_cache_scheduler_output(
    self,
    scheduler_output: "SchedulerOutput",
) -> None:
    """Apply scheduler-side encoder cache lifecycle updates."""
    for mm_hash in scheduler_output.free_encoder_mm_hashes:
        self.encoder_cache.pop(mm_hash, None)  # 动态删除，释放 GPU 内存
```

**逻辑驱逐**（不释放物理内存）:
```python
# vllm/v1/core/encoder_cache_manager.py:175-183
# Not enough free slots but enough reclaimable slots
# NOTE: Eviction takes place here, but physical memory is not freed
# until model runner is notified by the scheduler output.
while num_embeds > self.num_free_slots:
    mm_hash, num_free_embeds = self.freeable.popitem(last=False)
    del self.cached[mm_hash]      # 逻辑删除
    self.freed.append(mm_hash)    # 记录待释放
    self.num_free_slots += num_free_embeds
```

##### 3.5.2.5 内存占用上限

虽然 `encoder_cache` 是动态增长的，但它的**上限**受到 `encoder_cache_size` 的限制：

```
最大内存占用（字节） = encoder_cache_size × hidden_size × dtype_size
```

其中：
- `encoder_cache_size = max(max_num_batched_tokens, max_tokens_per_mm_item)`
- `hidden_size`: 模型隐藏层大小
- `dtype_size`: 数据类型大小（通常为 2 字节）

**示例**（`max_num_batched_tokens=32768`, `hidden_size=4096`, `dtype=bfloat16`）:
```
最大内存占用 = 32768 × 4096 × 2 = 256 MB
```

##### 3.5.2.6 动态增长的实际表现

| 阶段 | `encoder_cache` 状态 | 内存占用 |
|------|---------------------|----------|
| **启动时** | 空字典 `{}` | 0 字节 |
| **处理第一个多模态请求** | 添加 1 个条目 | `1 × num_embeds × hidden_size × dtype_size` |
| **处理 N 个多模态请求** | 添加 N 个条目 | `N × num_embeds × hidden_size × dtype_size` |
| **达到上限** | 驱逐旧条目，添加新条目 | 稳定在 `encoder_cache_size × hidden_size × dtype_size` |
| **请求完成** | 删除对应条目 | 减少相应内存 |

#### 3.5.3 与 KV Cache 的对比

| 特性 | `encoder_cache` | KV Cache |
|------|----------------|----------|
| **分配时机** | 动态（运行时） | 静态（启动时） |
| **分配方式** | Python 字典动态添加 | 预分配连续 GPU 内存 |
| **大小** | 动态增长，上限固定 | 启动时固定 |
| **管理器** | `EncoderCacheManager`（逻辑） | `KVCacheManager`（物理+逻辑） |
| **驱逐机制** | LRU 驱逐 | 基于块表的引用计数 |

#### 3.5.4 总结

| 问题 | 答案 |
|------|------|
| `encoder_cache` 是启动时固定分配的吗？ | **否**，是运行时动态增长的 |
| `max_num_batched_tokens` 定义了什么？ | 定义了 `encoder_cache_size` 的**上限** |
| 实际内存占用何时达到上限？ | 当缓存的多模态嵌入数量达到 `encoder_cache_size` 时 |
| 内存占用会减少吗？ | **是**，当请求完成或缓存驱逐时 |
| 最大内存占用如何计算？ | `encoder_cache_size × hidden_size × dtype_size` |

**核心设计思想**: vLLM 采用**逻辑上限 + 动态分配**的策略，既保证了内存使用的可控性，又避免了启动时的过度分配。这种设计对于多模态场景尤为重要，因为多模态请求通常是稀疏的（不是每个请求都包含图像/视频）。

---

## 4. 核心问题解答

### 4.1 采样器 (Sampler)、投机解码器 (drafter)、输入批处理器 (InputBatch) 详解

#### 4.1.1 采样器 (Sampler)

**定义**: `Sampler` 是负责从模型输出的 logits 中采样下一个 token 的模块。

**位置**: `vllm/v1/sample/sampler.py`

**核心功能**:
1. 将 logits 转换为概率分布
2. 应用各种采样策略（温度、top-k、top-p、min-p 等）
3. 应用惩罚（重复惩罚、频率惩罚、存在惩罚）
4. 处理 bad words 和 allowed token ids
5. 计算 logprobs（如果请求）

**关键代码**:
```python
class Sampler(nn.Module):
    def __init__(self, logprobs_mode="raw_logprobs", use_fp64_gumbel=False):
        self.topk_topp_sampler = TopKTopPSampler(logprobs_mode, use_fp64_gumbel)

    def forward(self, logits, sampling_metadata, ...):
        # 1. 应用 logit processors
        # 2. 应用 penalties
        # 3. 应用 temperature
        # 4. 应用 top-k / top-p
        # 5. 采样
        # 6. 返回 SamplerOutput
```

**在 GPUModelRunner 中的使用**:
```python
# 初始化
self.sampler = Sampler(
    logprobs_mode=self.model_config.logprobs_mode,
    use_fp64_gumbel=self.model_config.use_fp64_gumbel,
)

# 使用 (在 _sample 方法中)
sampler_output = self.sampler(
    logits=logits,
    sampling_metadata=sampling_metadata,
)
```

#### 4.1.2 投机解码器 (drafter)

**定义**: `drafter` 是用于投机解码 (Speculative Decoding) 的模块，负责提出候选 token (draft tokens)，由目标模型验证。

**位置**: `vllm/v1/spec_decode/` 目录下的各种 Proposer 类

**核心功能**:
1. 基于当前上下文提出多个候选 token
2. 使用较小的草稿模型或启发式方法快速生成候选
3. 目标模型一次性验证所有候选，提高解码效率

**主要类型**:

| 类型 | 类名 | 说明 |
|------|------|------|
| N-gram | `NgramProposer` | 基于 N-gram 统计的启发式方法 |
| N-gram GPU | `NgramProposerGPU` | GPU 加速的 N-gram 方法 |
| EAGLE | `EagleProposer` | 使用草稿模型 (draft model) 的方法 |
| EAGLE3 | `EagleProposer` (eagle3) | EAGLE 的改进版本 |
| Medusa | `MedusaProposer` | 使用多个解码头的方法 |
| DFlash | `DFlashProposer` | 使用填充式解码的方法 |
| Draft Model | `DraftModelProposer` | 使用独立草稿模型的方法 |
| Suffix | `SuffixDecodingProposer` | 基于后缀匹配的方法 |
| Extract Hidden States | `ExtractHiddenStatesProposer` | 提取隐藏状态的方法 |

**在 GPUModelRunner 中的初始化**:
```python
if self.speculative_config and get_pp_group().is_last_rank:
    if self.speculative_config.method == "ngram":
        self.drafter = NgramProposer(self.vllm_config)
    elif self.speculative_config.use_eagle():
        self.drafter = EagleProposer(self.vllm_config, self.device, self)
    elif self.speculative_config.method == "medusa":
        self.drafter = MedusaProposer(...)
    # ...

    # 拒绝采样器，用于验证 draft tokens
    self.rejection_sampler = RejectionSampler(
        self.sampler, self.speculative_config, self.device
    )
```

**使用流程**:
```
1. 目标模型前向传播 → 得到 hidden_states
2. drafter.propose() → 提出 draft tokens
3. 目标模型验证 draft tokens → rejection_sampler
4. 接受部分 draft tokens，拒绝其余
```

#### 4.1.3 输入批处理器 (InputBatch)

**定义**: `InputBatch` 管理当前批次中所有请求的输入状态，包括 token IDs、块表、采样参数等。

**位置**: `vllm/v1/worker/gpu_input_batch.py`

**核心功能**:
1. 管理批次中所有请求的 token IDs (CPU 和 GPU)
2. 维护块表 (block table) 用于 KV 缓存管理
3. 跟踪每个请求的计算状态 (num_computed_tokens, num_prompt_tokens 等)
4. 管理采样参数 (temperature, top_p, top_k 等)
5. 处理 LoRA 请求映射
6. 支持异步调度中的 token ID 更新

**关键属性**:
```python
class InputBatch:
    def __init__(self, ...):
        # 请求管理
        self._req_ids: list[str | None] = []  # 批次中的请求 ID
        self.req_id_to_index: dict[str, int] = {}  # 请求 ID -> 批次索引

        # Token IDs (CPU)
        self.token_ids_cpu_tensor: torch.Tensor  # (max_num_reqs, max_model_len)
        self.token_ids_cpu: np.ndarray  # numpy 视图
        self.is_token_ids: np.ndarray  # 标记是否为 token IDs (vs embeddings)

        # 请求状态
        self.num_tokens_no_spec: np.ndarray  # 每个请求的 token 数 (不含投机)
        self.num_prompt_tokens: np.ndarray  # 每个请求的 prompt token 数
        self.num_computed_tokens_cpu: np.ndarray  # 已计算的 token 数

        # 块表 (KV 缓存管理)
        self.block_table: MultiGroupBlockTable

        # 采样参数
        self.temperature: torch.Tensor
        self.top_p: torch.Tensor
        self.top_k: torch.Tensor
        # ...

        # 异步调度
        self.prev_sampled_token_ids: torch.Tensor | None  # 上一步的采样结果
        self.prev_req_id_to_index: dict[str, int] | None  # 上一步的请求映射
```

**在 GPUModelRunner 中的使用**:
```python
# 初始化
self.input_batch = InputBatch(
    max_num_reqs=self.max_num_reqs,
    max_model_len=max(self.max_model_len, self.max_encoder_len),
    max_num_batched_tokens=self.max_num_tokens,
    device=self.device,
    vocab_size=self.model_config.get_vocab_size(),
    # ...
)

# 使用 (在 _update_states 中)
self.input_batch.remove_request(req_id)  # 移除完成的请求
self.input_batch.add_request(req_state)  # 添加新请求
self.input_batch.condense()  # 压缩批次
self.input_batch.refresh_metadata()  # 刷新元数据
```

### 4.2 `load_dummy_weights` 的含义

#### 4.2.1 什么是 `load_dummy_weights`？

`load_dummy_weights` 是一个布尔参数，用于控制是否加载**虚拟权重**（dummy weights）而不是真实的模型权重。

#### 4.2.2 虚拟权重 vs 真实权重

| 特性 | 虚拟权重 (Dummy Weights) | 真实权重 (Real Weights) |
|------|-------------------------|------------------------|
| **来源** | 随机生成 | 从预训练模型文件加载 |
| **值** | 均匀分布随机值 (-1e-3 ~ 1e-3) | 训练好的参数值 |
| **用途** | 性能测试、内存分析、CUDA 图捕获 | 实际推理 |
| **加载速度** | 极快（无需读取磁盘） | 慢（需读取 GB 级文件） |
| **内存占用** | 与真实权重相同 | 与虚拟权重相同 |
| **输出质量** | 无意义（随机输出） | 有意义 |

#### 4.2.3 代码实现

**`DummyModelLoader`** (`vllm/model_executor/model_loader/dummy_loader.py`):
```python
class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        for layer in model.modules():
            info = get_layerwise_info(layer)
            if info.can_load():
                self._process_online_quant_layer(layer, info)
            else:
                # 为每一层分配随机值
                initialize_dummy_weights(layer, model_config)
```

**`initialize_dummy_weights`** (`vllm/model_executor/model_loader/weight_utils.py:1273`):
```python
def initialize_dummy_weights(
    model: torch.nn.Module,
    model_config: ModelConfig,
    low: float = -1e-3,
    high: float = 1e-3,
    seed: int = 1234,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.
    """
    for param in model.state_dict().values():
        initialize_single_dummy_weight(param, low, high, seed)
```

#### 4.2.4 使用场景

**在 `load_model` 中的使用**:
```python
def load_model(self, load_dummy_weights: bool = False) -> None:
    if load_dummy_weights:
        self.load_config.load_format = "dummy"  # 设置为 dummy 格式

    model_loader = get_model_loader(self.load_config)
    self.model = model_loader.load_model(
        vllm_config=self.vllm_config, model_config=self.model_config
    )
```

**主要使用场景**:
1. **内存分析 (Memory Profiling)**: 在 `profile_run` 中使用虚拟权重快速评估内存需求
2. **CUDA 图捕获**: 在 `capture_model` 中使用虚拟权重捕获计算图，无需加载真实权重
3. **性能测试**: 在不关心输出质量的情况下测试推理速度
4. **CI/CD 测试**: 在测试环境中快速启动服务

#### 4.2.5 为什么使用虚拟权重？

```python
# 场景 1: 内存分析
def profile_run(self) -> None:
    # 使用虚拟权重快速分析内存，无需等待真实权重加载
    hidden_states, last_hidden_states = self._dummy_run(
        self.max_num_tokens, is_profile=True
    )

# 场景 2: CUDA 图捕获
def capture_model(self) -> int:
    # 使用虚拟权重捕获计算图，真实权重可能还未加载
    for batch_desc in batch_descriptors:
        self._warmup_and_capture(batch_desc, ...)
```

**优势**:
- **速度快**: 无需从磁盘读取 GB 级的模型文件
- **确定性**: 使用固定种子，每次生成的虚拟权重相同
- **安全性**: 随机值在合理范围内 (-1e-3 ~ 1e-3)，不会导致 NaN
- **一致性**: 虚拟权重与真实权重占用相同的内存空间

### 4.3 `GPUModelRunner.initialize_kv_cache` vs `EngineCore._initialize_kv_caches`

#### 4.3.1 区别与联系

| 特性 | `EngineCore._initialize_kv_caches` | `GPUModelRunner.initialize_kv_cache` |
|------|-----------------------------------|-------------------------------------|
| **层级** | 引擎核心层 (Engine Core) | 工作器层 (Worker) |
| **职责** | 协调所有 worker 的 KV 缓存初始化 | 初始化单个 worker 的 KV 缓存 |
| **调用者** | 引擎启动流程 | `Worker.initialize_from_config` |
| **被调用者** | `model_executor.initialize_from_config` | `initialize_attn_backend`, `initialize_kv_cache_tensors` 等 |
| **返回值** | `KVCacheConfig` (调度器使用的配置) | `None` |
| **副作用** | 设置 `vllm_config.cache_config.num_gpu_blocks` | 分配 GPU 内存，初始化注意力后端 |

#### 4.3.2 调用关系

```
┌─────────────────────────────────────────────────────────────────┐
│                    EngineCore._initialize_kv_caches             │
│                         (引擎核心进程)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ 1. 收集所有 worker 的 KV cache specs
                              │    model_executor.get_kv_cache_specs()
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 确定可用 GPU 内存                                           │
│     model_executor.determine_available_memory()                 │
│     └── 每个 worker 调用 profile_run() 分析内存                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. 计算 KV cache 配置                                          │
│     get_kv_cache_configs(vllm_config, kv_cache_specs,           │
│                          available_gpu_memory)                  │
│     └── 生成每个 worker 的 KVCacheConfig                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. 生成调度器 KV cache 配置                                    │
│     generate_scheduler_kv_cache_config(kv_cache_configs)        │
│     └── 设置 vllm_config.cache_config.num_gpu_blocks            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. 初始化所有 worker 的 KV cache                               │
│     model_executor.initialize_from_config(kv_cache_configs)     │
│     └── collective_rpc("initialize_from_config", ...)           │
│         └── 每个 Worker.initialize_from_config(kv_cache_config) │
│             └── GPUModelRunner.initialize_kv_cache(kv_cache_config) │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.3.3 详细说明

**`EngineCore._initialize_kv_caches`** (`vllm/v1/engine/core.py:240`):
```python
def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
    # 1. 注册所有 KV cache specs
    register_all_kvcache_specs(vllm_config)

    # 2. 获取所有 worker 的 KV cache specs
    kv_cache_specs = self.model_executor.get_kv_cache_specs()

    # 3. 确定可用 GPU 内存 (通过 profile_run)
    available_gpu_memory = self.model_executor.determine_available_memory()

    # 4. 计算 KV cache 配置
    kv_cache_configs = get_kv_cache_configs(
        vllm_config, kv_cache_specs, available_gpu_memory
    )

    # 5. 生成调度器配置
    scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
    vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks

    # 6. 初始化所有 worker 的 KV cache
    self.model_executor.initialize_from_config(kv_cache_configs)

    return scheduler_kv_cache_config
```

**`GPUModelRunner.initialize_kv_cache`** (`vllm/v1/worker/gpu_model_runner.py:7504`):
```python
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig, is_profiling: bool = False) -> None:
    # 1. 深拷贝配置
    kv_cache_config = deepcopy(kv_cache_config)
    self.kv_cache_config = kv_cache_config

    # 2. 添加编码器专用层
    self.may_add_encoder_only_layers_to_kv_cache_config()

    # 3. 添加 KV 共享层
    self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)

    # 4. 初始化注意力后端
    self.initialize_attn_backend(kv_cache_config, is_profiling=is_profiling)

    # 5. 准备内核块大小
    kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, self.attn_groups)

    # 6. 创建元数据构建器
    self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

    # 7. 重新初始化输入批处理
    self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)

    # 8. 初始化 KV cache 张量
    kv_caches = self.initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)

    # 9. 注册 KV 传输组
    if has_kv_transfer_group() and not is_profiling:
        kv_transfer_group.register_kv_caches(kv_caches)
```

#### 4.3.4 总结

- **`EngineCore._initialize_kv_caches`** 是**协调者**，负责：
  - 收集所有 worker 的 KV cache 需求
  - 通过内存分析确定可用内存
  - 计算全局 KV cache 配置
  - 触发所有 worker 初始化

- **`GPUModelRunner.initialize_kv_cache`** 是**执行者**，负责：
  - 在单个 worker 上分配 GPU 内存
  - 初始化注意力后端和元数据构建器
  - 设置 KV cache 张量

### 4.4 `GPUModelRunner` 与 `Worker` 的关系

#### 4.4.1 `GPUModelRunner` 是由 `Worker` 创建的吗？

**是的**。从代码中可以明确看到：

```python
# vllm/v1/worker/gpu_worker.py:401-416
# Construct the model runner
if self.use_v2_model_runner:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    self.model_runner: GPUModelRunner = GPUModelRunnerV2(
        self.vllm_config, self.device
    )
else:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
    self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)
```

#### 4.4.2 `GPUModelRunner.load_model` 是由 `Worker.load_model` 调用的吗？

**是的**。从代码中可以明确看到：

```python
# vllm/v1/worker/gpu_worker.py:424-431
def load_model(self, *, load_dummy_weights: bool = False) -> None:
    with (
        self._maybe_get_memory_pool_context(tag="weights"),
        set_current_vllm_config(self.vllm_config),
        self._scoped_allocator_max_split(max_split_size_mb=20),
    ):
        self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
```

#### 4.4.3 完整的调用链

```
┌─────────────────────────────────────────────────────────────────┐
│  EngineCore / Executor                                          │
│  └── collective_rpc("load_model", ...)                          │
│      └── Worker.load_model(load_dummy_weights=...)              │
│          └── GPUModelRunner.load_model(load_dummy_weights=...)  │
│              └── model_loader.load_model(...)                   │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.4.4 `Worker` 对 `GPUModelRunner` 的其他调用

| Worker 方法 | 调用的 GPUModelRunner 方法 | 说明 |
|------------|---------------------------|------|
| `load_model()` | `load_model()` | 加载模型权重 |
| `initialize_from_config()` | `initialize_kv_cache()` | 初始化 KV 缓存 |
| `determine_available_memory()` | `profile_run()` | 内存分析 |
| `compile_or_warm_up_model()` | `_dummy_run()`, `capture_model()` | 编译/预热 |
| `execute_model()` | `execute_model()` | 执行模型 |
| `sample_tokens()` | `sample_tokens()` | 采样 token |
| `shutdown()` | `shutdown()` | 关闭清理 |

### 4.5 `execute_model` 返回 `None` 的场景

#### 4.5.1 说法正确性

**是的，这个说法是正确的。** 当 `execute_model` 返回 `None` 时，表示需要调用 `sample_tokens()` 来完成采样过程。

#### 4.5.2 为什么返回 `None`？

`execute_model` 返回 `None` 的设计是为了将**模型前向传播**和**采样**分离，以支持**异步调度**和**投机解码**。

##### 4.5.2.1 代码证据

```python
# vllm/v1/worker/gpu_model_runner.py:4130-4498
@torch.inference_mode()
def execute_model(self, scheduler_output, intermediate_tensors=None):
    # ... 执行模型前向传播 ...

    # 保存执行状态
    self.execute_model_state = ExecuteModelState(
        scheduler_output, logits, spec_decode_metadata,
        spec_decode_common_attn_metadata, hidden_states,
        sample_hidden_states, aux_hidden_states,
        ec_connector_output, cudagraph_stats, slot_mappings,
    )

    # 返回 None 表示需要调用 sample_tokens
    return None
```

##### 4.5.2.2 返回 `None` 的条件

`execute_model` 在以下情况下返回 `None`：

1. **正常生成任务**: 模型前向传播完成，需要采样下一个 token
2. **非最后 PP rank**: 返回中间张量给下一个 PP rank（此时返回 `IntermediateTensors`，不是 `None`）
3. **池化模型**: 直接返回池化输出（此时返回 `ModelRunnerOutput`，不是 `None`）

```python
# 关键代码路径
if not self.broadcast_pp_output:
    if not get_pp_group().is_last_rank:
        # 非最后 PP rank，返回中间张量
        return hidden_states  # 返回 IntermediateTensors，不是 None

    if self.is_pooling_model:
        # 池化模型，返回池化输出
        return self._pool(...)  # 返回 ModelRunnerOutput，不是 None

    # 计算 logits
    sample_hidden_states = hidden_states[logits_indices]
    logits = self.model.compute_logits(sample_hidden_states)
else:
    # 广播 PP 输出
    ...

# 保存执行状态并返回 None
self.execute_model_state = ExecuteModelState(...)
return None  # 需要调用 sample_tokens
```

##### 4.5.2.3 为什么需要分离？

**异步调度 (Async Scheduling)**:

```
传统同步调度:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  准备输入    │───▶│  模型前向    │───▶│  采样       │
│             │     │  传播        │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
       ↑                                           │
       └───────────────────────────────────────────┘
                    (等待采样完成才能准备下一步)

异步调度:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  准备输入    │───▶│  模型前向    │───▶│  采样       │
│  (步骤 N)   │     │  传播 (N)   │     │  (步骤 N-1) │
└─────────────┘     └─────────────┘     └─────────────┘
       ↑                                           │
       └───────────────────────────────────────────┘
       (准备下一步和采样当前步可以并行)
```

**投机解码 (Speculative Decoding)**:

```
┌─────────────────────────────────────────────────────────────────┐
│  execute_model (目标模型)                                       │
│  ├── 前向传播目标模型                                           │
│  ├── 保存 hidden_states 到 execute_model_state                  │
│  └── 返回 None                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  sample_tokens                                                  │
│  ├── 从 execute_model_state 恢复 hidden_states                  │
│  ├── 使用 drafter 提出 draft tokens                             │
│  ├── 使用 rejection_sampler 验证 draft tokens                   │
│  └── 返回最终采样结果                                           │
└─────────────────────────────────────────────────────────────────┘
```

##### 4.5.2.4 `execute_model_state` 的作用

`execute_model_state` 是一个 `NamedTuple`，用于在 `execute_model` 和 `sample_tokens` 之间传递状态：

```python
class ExecuteModelState(NamedTuple):
    scheduler_output: "SchedulerOutput"           # 调度器输出
    logits: torch.Tensor                          # 模型输出的 logits
    spec_decode_metadata: SpecDecodeMetadata | None  # 投机解码元数据
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None  # 投机解码注意力元数据
    hidden_states: torch.Tensor                   # 隐藏状态
    sample_hidden_states: torch.Tensor            # 采样用的隐藏状态
    aux_hidden_states: list[torch.Tensor] | None  # 辅助隐藏状态 (EAGLE3)
    ec_connector_output: ECConnectorOutput | None  # EC 连接器输出
    cudagraph_stats: CUDAGraphStat | None         # CUDA 图统计
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None  # 槽映射
```

##### 4.5.2.5 完整调用流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Worker.execute_model(scheduler_output)                         │
│  └── GPUModelRunner.execute_model(scheduler_output)             │
│      ├── 执行模型前向传播                                       │
│      ├── 保存 execute_model_state                               │
│      └── 返回 None                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Worker.sample_tokens(grammar_output)                           │
│  └── GPUModelRunner.sample_tokens(grammar_output)               │
│      ├── 从 execute_model_state 恢复状态                        │
│      ├── 执行采样 (_sample)                                     │
│      ├── 执行投机解码 (propose_draft_token_ids)                 │
│      ├── 执行簿记同步 (_bookkeeping_sync)                       │
│      └── 返回 ModelRunnerOutput / AsyncGPUModelRunnerOutput     │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.5.3 总结

| 场景 | `execute_model` 返回值 | 是否需要 `sample_tokens` |
|------|----------------------|------------------------|
| 正常生成任务 | `None` | 是 |
| 非最后 PP rank | `IntermediateTensors` | 否 |
| 池化模型 | `ModelRunnerOutput` | 否 |
| 空批次 | `EMPTY_MODEL_RUNNER_OUTPUT` | 否 |
| EC 传输生产者 | `ModelRunnerOutput` | 否 |

**核心设计思想**: 通过将前向传播和采样分离，vLLM 实现了：
1. **异步调度**: 前向传播和采样可以并行执行
2. **投机解码**: 在采样阶段使用 drafter 提出候选 token
3. **流水线并行**: 不同 PP rank 可以并行处理不同阶段

---

## 附录：关键调用关系总结

| 方法 | 主要调用者 | 被调用方法 |
|------|-----------|-----------|
| `execute_model` | GPUWorker | `_update_states`, `_prepare_inputs`, `_preprocess`, `_model_forward`, `_pool` |
| `sample_tokens` | GPUWorker | `_sample`, `_update_states_after_model_execute`, `propose_draft_token_ids`, `_bookkeeping_sync` |
| `_prepare_inputs` | `execute_model` | `_get_cumsum_and_arange`, `_calc_spec_decode_metadata`, `_prepare_input_ids` |
| `_preprocess` | `execute_model` | `_execute_mm_encoder`, `_gather_mm_embeddings`, `_init_model_kwargs` |
| `_build_attention_metadata` | `execute_model` | `_get_encoder_seq_lens`, `builder.build` |
| `propose_draft_token_ids` | `sample_tokens` | `drafter.propose`, `prepare_next_token_ids_padded` |
| `load_model` | GPUWorker | `model_loader.load_model`, `load_lora_model` |
| `profile_run` | GPUWorker | `_dummy_run`, `_dummy_sampler_run`, `_dummy_pooler_run` |
| `capture_model` | GPUWorker | `_capture_cudagraphs`, `_warmup_and_capture` |

---

## 5. 补充问答

本节汇总对 `GPUModelRunner`、多模态编码器、采样器及执行架构的常见追问。

### 5.1 池化模型和提示嵌入是什么？

**池化模型（Pooling Model）**

池化模型是**不生成 token，而是输出向量表示**的模型，典型用途包括 Embedding、Reranker、Reward 等。在 vLLM v1 中通过 `model_config.runner_type == "pooling"` 判断：

```python
# vllm/v1/worker/gpu_model_runner.py:480
self.is_pooling_model = model_config.runner_type == "pooling"
```

如果是池化模型，`execute_model` 不会走采样器，而是调用 `_pool()`：

```python
# vllm/v1/worker/gpu_model_runner.py:4438-4445
if self.is_pooling_model:
    return self._pool(
        hidden_states,
        num_scheduled_tokens,
        num_scheduled_tokens_np,
        kv_connector_output,
    )
```

`_pool()` 内部调用 `model.pooler(...)` 生成池化结果（如 embedding 向量）：

```python
# vllm/v1/worker/gpu_model_runner.py:3448-3451
model = cast(VllmModelForPooling, self.model)
raw_pooler_output: PoolerOutput = model.pooler(
    hidden_states=hidden_states, pooling_metadata=pooling_metadata
)
```

**提示嵌入（Prompt Embeds）**

`prompt_embeds` 是用户**直接传入的、已经计算好的文本/多模态嵌入向量**，而不是让 vLLM 通过 tokenizer + embedding layer 计算。需要显式开启：

```python
# vllm/config/model.py:255-260
enable_prompt_embeds: bool = False
"""If `True`, enables passing text embeddings as inputs via the
`prompt_embeds` key."""
```

在请求入口 `chat_utils.py` 中，`prompt_embeds` 被当作一种特殊 modality 处理，不需要经过视觉编码器。

---

### 5.2 "Sampler 是负责从模型输出的 logits 中采样下一个 token 的模块" 中的 logits 是什么？

**logits** 是语言模型最后一层输出的**未归一化分数向量**，长度等于词表大小 `vocab_size`。例如：

- 词表大小 `V = 32000`
- 模型输出 `logits.shape = (batch_size, V)`
- 对位置 `i`，`logits[i, :]` 表示该位置取每个 token 的原始分数

Sampler 拿到 logits 后依次做：转 float32、应用 allowed token ids / bad words、应用 logit processors 与 penalties、应用 temperature / top-k / top-p，最后采样：

```python
# vllm/v1/sample/sampler.py:72-119
def forward(self, logits: torch.Tensor, sampling_metadata: SamplingMetadata, ...):
    logits = logits.to(torch.float32)
    logits = self.apply_logits_processors(logits, ...)
    sampled, processed_logprobs = self.sample(logits, sampling_metadata)
```

---

### 5.3 CUDA 图捕获是什么？

**CUDA Graph** 是 NVIDIA GPU 提供的特性：把一系列 GPU 操作（kernel launch、内存拷贝等）预先记录成一张“图”，之后可整体 replay，避免每次重复做 kernel 调度、参数校验等 CPU 开销。

在 vLLM 中：

- 捕获时机：`capture_model()`
- 运行时选择：`cudagraph_dispatcher.dispatch(...)`
- 模型被包装为 `CUDAGraphWrapper` / `BreakableCUDAGraphWrapper`
- 多模态编码器有独立的 `EncoderCudaGraphManager`

```python
# vllm/v1/worker/gpu_model_runner.py:853
self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)
```

`_dummy_run` 和 `capture_model` 会针对不同 batch size 分别捕获 CUDA 图，运行时按 `num_tokens` / `num_reqs` 匹配：

```python
# vllm/v1/worker/gpu_model_runner.py:3951
cudagraph_mode, batch_descriptor = dispatch_cudagraph(...)
```

CUDA 图要求固定 shape，因此 vLLM 会对输入做 padding。

---

### 5.4 EngineCore、Worker 和 GPUModelRunner 三者关系

| 组件 | 层级 | 职责 |
|------|------|------|
| **EngineCore** | 引擎核心进程 | 持有 Scheduler 和 Executor，协调调度、KV cache 初始化、执行模型 |
| **Executor**（如 `MultiprocExecutor`） | 执行器层 | 管理多个 Worker 进程，通过 `collective_rpc` 广播调用 |
| **Worker**（如 `gpu_worker.Worker`） | Worker 进程 | 每个 GPU 一个进程，初始化设备并创建 `GPUModelRunner` |
| **GPUModelRunner** | 模型运行器 | 真正的模型执行逻辑：加载模型、KV cache、前向传播、采样 |

**创建关系**

```
EngineCore.__init__()
    └── self.model_executor = Executor(vllm_config)
            └── 创建 N 个 Worker 进程
                    └── Worker.init_device()
                            └── self.model_runner = GPUModelRunner(...)
```

**调用关系**

```
EngineCore.step()
    └── model_executor.execute_model(scheduler_output)
            └── collective_rpc("execute_model", args=(scheduler_output,))
                    └── 每个 Worker.execute_model(scheduler_output)
                            └── GPUModelRunner.execute_model(scheduler_output)
                                    └── 返回 None（需要采样）
    └── model_executor.sample_tokens(grammar_output)
            └── collective_rpc("sample_tokens", args=(grammar_output,))
                    └── 每个 Worker.sample_tokens(grammar_output)
                            └── GPUModelRunner.sample_tokens(grammar_output)
```

**关系图**

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         EngineCore                                  │
│  ┌─────────────────┐    ┌─────────────────────────────────────────┐ │
│  │   Scheduler     │───▶│         model_executor                  │ │
│  │                 │    │  (MultiprocExecutor / RayExecutor / ...)│ │
│  └─────────────────┘    └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ collective_rpc
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Executor                                    │
│  ┌─────────────┐  ┌─────────────┐        ┌─────────────────────┐   │
│  │  Worker 0   │  │  Worker 1   │  ...   │  Worker N-1         │   │
│  │  (GPU 0)    │  │  (GPU 1)    │        │  (GPU N-1)          │   │
│  └──────┬──────┘  └──────┬──────┘        └──────────┬──────────┘   │
└─────────┼────────────────┼──────────────────────────┼──────────────┘
          │                │                          │
          ▼                ▼                          ▼
   ┌──────────────┐ ┌──────────────┐        ┌──────────────┐
   │ GPUModelRunner│ │ GPUModelRunner│  ...   │ GPUModelRunner│
   │   on GPU 0   │ │   on GPU 1   │        │  on GPU N-1  │
   └──────────────┘ └──────────────┘        └──────────────┘
```

---

### 5.5 vLLM 实际执行中会为每个 GPU 分配一个 GPUModelRunner 对象吗？

**是的。**

vLLM 采用**每个 GPU 一个独立 Worker 进程**的设计。每个 Worker 在 `init_device()` 中创建**一个** `GPUModelRunner` 实例。例如 `tensor_parallel_size=2` 时，两个 GPU 各有一个 Worker，各创建一个 `GPUModelRunner`，模型权重切分到两个 GPU 上。

```python
# vllm/v1/executor/multiproc_executor.py:174-191
for local_rank in range(self.local_world_size):
    global_rank = global_start_rank + local_rank
    unready_worker_handle = WorkerProc.make_worker_process(
        vllm_config=self.vllm_config,
        local_rank=local_rank,
        rank=global_rank,
        ...
    )
```

`local_world_size` 即当前节点 GPU 数。

---

### 5.6 `execute_model` 中“计算级联注意力前缀长度”在做什么？

代码位置：

```python
# vllm/v1/worker/gpu_model_runner.py:4219-4227
cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
    num_scheduled_tokens_np,
    self.input_batch.num_computed_tokens_cpu[:num_reqs],
    scheduler_output.num_common_prefix_blocks,
)
```

**级联注意力（Cascade Attention）** 用于优化多个请求共享相同 prompt 前缀的场景：只计算一次公共前缀的 attention，而不是每个请求分别计算。

计算逻辑：

```python
# vllm/v1/worker/gpu_model_runner.py:2663-2711
common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
common_prefix_len = min(common_prefix_len, num_computed_tokens.min())
common_prefix_len = (
    common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size
)
```

1. 从 Scheduler 拿到共享 KV cache 块数。
2. 乘以 `block_size` 得到 token 数。
3. 限制为不超过 batch 中 `num_computed_tokens` 的最小值（避免包含某些请求的未来 token）。
4. 对齐到 `block_size` 的整数倍。

该特性默认关闭（`disable_cascade_attn=True`），需显式开启。

---

### 5.7 `_execute_mm_encoder` 的执行时机

`_execute_mm_encoder` 只在 `_preprocess()` 中被调用，且需同时满足：

```python
# vllm/v1/worker/gpu_model_runner.py:3534-3540
if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
    self._execute_mm_encoder(scheduler_output)
    mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)
```

条件拆解：

| 条件 | 含义 |
|------|------|
| `self.supports_mm_inputs` | 模型支持多模态输入 |
| `is_first_rank` | 当前是 PP 的第一个 rank，负责输入/编码器处理 |
| `not is_encoder_decoder` | 普通多模态生成模型，非 encoder-decoder 架构 |

对于 **encoder-decoder** 模型，调用路径不同：

```python
# vllm/v1/worker/gpu_model_runner.py:3638-3645
if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
    encoder_outputs = self._execute_mm_encoder(scheduler_output)
    model_kwargs.update({"encoder_outputs": encoder_outputs})
```

**总结**

1. 普通多模态生成模型：在 `_preprocess` 第一分支执行，为 `_gather_mm_embeddings` 准备缓存。
2. Encoder-Decoder 模型：在 `_preprocess` 末尾执行，输出直接作为 decoder 的 `encoder_outputs`。
3. 只在 `is_first_rank` 为真时执行。

---

### 5.8 `_execute_mm_encoder` 中 `prompt_embeds` 相关代码

#### `pe_indices` 的含义

```python
# vllm/v1/worker/gpu_model_runner.py:2998-3002
pe_indices = [
    i
    for i, (modality, _) in enumerate(mm_kwargs)
    if modality == "prompt_embeds"
]
```

`mm_kwargs` 是调度器安排给编码器的多模态输入列表，`pe_indices` 找出其中 modality 为 `prompt_embeds` 的索引。

#### `modality == "prompt_embeds"` 的含义

`prompt_embeds` 是一种**伪多模态 modality**：

- 不是图像/音频/视频，而是用户直接提供的**预计算嵌入向量**。
- 不需要经过视觉编码器（ViT），因为已经在嵌入空间中。
- 为复用多模态嵌入的 gather 流程，被包装成 modality。

#### `_cache_encoder_output(...)` 做了什么

```python
# vllm/v1/worker/gpu_model_runner.py:3004-3012
for i in pe_indices:
    pe_tensor = mm_kwargs[i][1]["embedding"].data
    self._cache_encoder_output(
        mm_hashes[i],
        pe_tensor.to(self.device),
        scheduler_output,
    )
```

```python
# vllm/v1/worker/gpu_model_runner.py:2972-2981
def _cache_encoder_output(...):
    self.encoder_cache[mm_hash] = output
    self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)
```

1. 从 `mm_kwargs[i][1]["embedding"].data` 取出用户传入的 prompt_embeds tensor。
2. 搬到当前 GPU。
3. 以 `mm_hashes[i]` 为 key 存入 `self.encoder_cache`。
4. 后续 `_gather_mm_embeddings` 即可像普通多模态嵌入一样取出。

#### `encoder_outputs: list[torch.Tensor] = []` 如何获取

过滤掉 `prompt_embeds` 后，剩余项才需要真正编码：

```python
# vllm/v1/worker/gpu_model_runner.py:3107-3189
encoder_outputs: list[torch.Tensor] = []
current_item_idx = 0
for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
    mm_kwargs, device=self.device, pin_memory=PIN_MEMORY
):
    batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
    encoder_outputs.extend(batch_outputs)
    current_item_idx += num_items

for mm_hash, output in zip(mm_hashes, encoder_outputs):
    self._cache_encoder_output(mm_hash, output, scheduler_output)
```

流程：

1. `group_and_batch_mm_kwargs` 按 modality 分组 batch。
2. 调用 `model.embed_multimodal(**mm_kwargs_batch)` 做编码器前向传播。
3. 返回 `batch_outputs`（列表/张量），每个元素对应一个输入项的嵌入。
4. 追加到 `encoder_outputs`。
5. 按 `mm_hash` 缓存。

---

### 5.9 `_gather_mm_embeddings` 方法分析

#### `mm_features = req_state.mm_features` 含义

```python
# vllm/v1/worker/gpu_model_runner.py:3221-3224
req_state = self.requests[req_id]
mm_features = req_state.mm_features
```

`req_state` 是 `CachedRequestState`，代表请求在 `GPUModelRunner` 中的缓存状态。`mm_features` 是该请求包含的多模态输入项列表，每个元素是 `MultiModalFeatureSpec`：

```python
# vllm/multimodal/inputs.py:302-329
@dataclass
class MultiModalFeatureSpec:
    data: "MultiModalKwargsItem | None"
    modality: str
    identifier: str           # 缓存 hash
    mm_position: PlaceholderRange  # 在 prompt 中的位置
```

#### `get_embeds_indices_in_range(start_idx, end_idx)` 在做什么

```python
# vllm/v1/worker/gpu_model_runner.py:3242-3244
curr_embeds_start, curr_embeds_end = (
    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
)
```

`pos_info` 是 `PlaceholderRange`。当前 step 只处理 prompt 的一部分，而多模态占位符中某些位置可能不是真正的嵌入（由 `is_embed` 标记）。该函数把“prompt 中的位置范围”映射到“实际嵌入张量中的索引范围”：

```python
# vllm/multimodal/inputs.py:158-177
def get_embeds_indices_in_range(self, start_idx, end_idx):
    if self.embeds_cumsum is None:
        return start_idx, end_idx
    embeds_start_idx = self.embeds_cumsum[start_idx - 1] if start_idx > 0 else 0
    embeds_end_idx = self.embeds_cumsum[end_idx - 1] if end_idx > 0 else 0
    return embeds_start_idx, embeds_end_idx
```

#### 得到 `mm_embeds` 的过程

```python
# vllm/v1/worker/gpu_model_runner.py:3225-3295
lo, hi = get_mm_features_in_window(
    mm_features,
    start=num_computed_tokens,
    end=num_computed_tokens + num_scheduled_tokens,
)
for i in range(lo, hi):
    mm_feature = mm_features[i]
    pos_info = mm_feature.mm_position
    mm_hash = mm_feature.identifier
    encoder_output = self._get_encoder_output_from_cache(mm_hash)
    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
    mm_embeds_req.append(mm_embeds_item)
mm_embeds.extend(mm_embeds_req)
```

1. 用 `get_mm_features_in_window` 找出当前窗口内涉及的多模态项。
2. 计算每个项在当前窗口内的范围。
3. 用 `get_embeds_indices_in_range` 映射到嵌入张量索引。
4. 从 `encoder_cache` 按 `mm_hash` 取出完整编码器输出并切片。
5. 收集到 `mm_embeds_req`，再追加到全局 `mm_embeds`。
6. 同时用 `is_mm_embed` 标记哪些位置是多模态嵌入。

#### `scheduler_output.free_encoder_mm_hashes` 如何获取

```python
# vllm/v1/core/sched/scheduler.py:1155
free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
```

`EncoderCacheManager` 在 `can_allocate()` / `free()` 时把被驱逐的 `mm_hash` 放入 `self.freed`，`get_freed_mm_hashes()` 取出并清空：

```python
# vllm/v1/core/encoder_cache_manager.py:256-267
def get_freed_mm_hashes(self) -> list[str]:
    freed = self.freed
    self.freed = []
    return freed
```

然后在 `_process_encoder_cache_scheduler_output` 中真正释放 GPU 内存：

```python
# vllm/v1/worker/gpu_model_runner.py:1170-1176
def _process_encoder_cache_scheduler_output(...):
    for mm_hash in scheduler_output.free_encoder_mm_hashes:
        self.encoder_cache.pop(mm_hash, None)
```

**注意**：调度器只是逻辑驱逐，真正的 GPU 张量释放在 `GPUModelRunner` 里完成。

---

### 5.10 多模态特征和多模态嵌入的区别是什么？计算过程是什么？

#### 区别

| 概念 | 含义 | 例子 |
|------|------|------|
| **多模态特征（MultiModal Feature）** | 原始/预处理后的多模态数据及其在 prompt 中的位置信息 | 图片像素张量、音频 mel 谱 |
| **多模态嵌入（MultiModal Embedding）** | 经过编码器后可直接输入 LLM 的向量 | 图片经 ViT 后得到的 `(num_patches, hidden_size)` 张量 |

`MultiModalFeatureSpec` 描述的是**特征**；`_execute_mm_encoder` 输出的是**嵌入**。

#### 多模态嵌入计算过程

```text
用户输入（图片/音频/视频）
    ↓
MultiModalProcessor 处理
    ↓
MultiModalFeatureSpec 列表（data + modality + position）
    ↓
Scheduler 决定哪些需要编码
    ↓
GPUModelRunner._execute_mm_encoder()
    ↓
group_and_batch_mm_kwargs() 按 modality batch
    ↓
model.embed_multimodal(**mm_kwargs_batch)  # 编码器前向
    ↓
MultiModalEmbeddings（编码器输出）
    ↓
_cache_encoder_output() 存入 encoder_cache
    ↓
GPUModelRunner._gather_mm_embeddings()
    ↓
按当前 step 窗口切片
    ↓
mm_embeds + is_mm_embed 传给 model.embed_input_ids()
    ↓
inputs_embeds 输入 LLM
```

#### 是否通过编码器模型（如 ViT）生成？

**是的，对于图像/音频/视频等多模态输入**，编码器通常是：

- 图像：`CLIP ViT`、`SigLIP`、`EVA-CLIP` 等
- 音频：`Whisper` encoder、BEATs 等
- 视频：图像编码器 + 时间维度处理

调用点：

```python
# vllm/v1/worker/gpu_model_runner.py:3177-3180
if cudagraph_output is not None:
    batch_outputs = cudagraph_output
else:
    batch_outputs = model.embed_multimodal(**mm_kwargs_batch)
```

`model.embed_multimodal` 就是模型的多模态编码器入口。

#### `prompt_embeds` 的特殊性

`prompt_embeds`**不经过编码器**，因为用户已经提供了嵌入向量。它只是为了复用多模态嵌入的 gather 流程，被包装成一种 modality。

---

*文档生成时间: 2026-07-16*
*基于 vLLM 代码库分析*
