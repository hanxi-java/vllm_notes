# vLLM Qwen3-VL 阅读笔记
> 主题：`Qwen3VLMultiModalProcessor._call_hf_processor` → `BaseMultiModalProcessor._call_hf_processor` → HF processor 调用链路的细节问答
> 涉及文件：`vllm/model_executor/models/qwen3_vl.py`、`vllm/multimodal/processing/processor.py`、`vllm/multimodal/processing/context.py`、`vllm/transformers_utils/processor.py`

## 目录

- [问题 1：两个 `_call_hf_processor` 的分工与区别](#问题-1两个-_call_hf_processor-的分工与区别)
- [问题 2：`enable_hf_prompt_update=True` 为什么对应无缓存路径，为什么要文本+mm 一起进 HF](#问题-2enable_hf_prompt_updatetrue-为什么对应无缓存路径为什么要文本mm-一起进-hf)
- [问题 3：`typ` 是怎么变成 `Qwen3VLProcessor` 的](#问题-3typ-是怎么变成-qwen3vlprocessor-的)
- [问题 4：`call_hf_processor` 的入参 `hf_processor` 就是 Qwen3VLProcessor 吗](#问题-4call_hf_processor-的入参-hf_processor-就是-qwen3vlprocessor-吗)
- [问题 5：`_get_prompt_updates` 生成的 `PromptReplacement` 详解](#问题-5_get_prompt_updates-生成的-promptreplacement-详解)
- [问题 6：`MultiModalInput.prompt_token_ids: list[int]` 的含义](#问题-6multimodalinputprompt_token_ids-listint-的含义)
- [问题 7：`qwen3_vl.py` 中的主要对象及其作用](#问题-7qwen3_vlpy-中的主要对象及其作用)
- [问题 8：`Qwen3_VisionTransformer.forward` 详解](#问题-8qwen3_visiontransformerforward-详解)

---

## 问题 1：两个 `_call_hf_processor` 的分工与区别

> 原问题：Qwen3VLMultiModalProcessor 的 `_call_hf_processor` 和其基类的 `_call_hf_processor` 分别都做了哪些事情？有什么区别？

**基类版**（`processor.py:1097`，所有模型共用）——只做"标准的一次性调用"：

```python
def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
    return self.info.ctx.call_hf_processor(
        self.info.get_hf_processor(**mm_kwargs),   # 取 HF processor 实例
        dict(text=prompt, **mm_data),              # text+图片+视频 一起给
        dict(**mm_kwargs, **tok_kwargs),
    )
```

职责就是桥接：拿实例 → 组 data → 透传 kwargs → 调用 → 返回 `BatchFeature`。它假设**一次调用就能处理整个 prompt 和所有模态**。

**Qwen3-VL 覆写版**（`qwen3_vl.py:1239`）——因为标准假设对视频不成立，拆成了 **N+1 次**调用：

| 步骤 | 基类 | Qwen3-VL 覆写 |
|---|---|---|
| 图片 | text+图+视频一次进 HF | 弹出 videos，text+images 调一次 `super()` |
| 视频 | 同上（HF 一把梭） | **每条视频单独**调一次 `super()`（只传 `"<|vision_start|><|video_pad|><|vision_end|>"` 假 prompt + 单条视频） |
| 视频 token 序列 | 直接用 HF 展开的 input_ids | **丢弃** HF 的，用 `get_video_repl` 自造带时间戳/EVS 剪枝的序列 |
| 合并 | 无 | `_replace_video_token_placeholders` 把自造的序列拼回文本 token 流 |
| 额外产出 | 无 | `timestamps` 附加到输出里 |

为什么必须覆写：Qwen3-VL 的视频占位符中**交错着时间戳文本 token**（`<0.5秒>`），且 `video_pruning_rate`（EVS）会改变每帧 token 数——这两件事 HF processor 的展开逻辑都不知道，所以 vLLM 只借 HF 做"抽帧+resize+normalize"，token 拼接自己来。

---

## 问题 2：`enable_hf_prompt_update=True` 为什么对应无缓存路径，为什么要文本+mm 一起进 HF

> 原问题：`enable_hf_prompt_update=True` 为什么是无缓存路径？这时候为什么要调用 `_apply_hf_processor_text_mm` 把文本和 mm 数据一起进 HF processor？

先看两处调用点：

```
_apply_hf_processor        (:1398, 缓存关闭/passthrough 时走)
    → _apply_hf_processor_main(..., enable_hf_prompt_update=True)

_cached_apply_hf_processor (:1441, 有缓存且只处理 miss 的 item)
    → _apply_hf_processor_main(..., enable_hf_prompt_update=False)
```

关键在于：**缓存路径下，送进 HF 的 mm 数据是"缺缓存的那几张图"，和完整 prompt 对不上**。

HF processor 有个硬性约束：文本里的占位符数量必须和传入的 mm item 数量一致（它要按 `grid_thw` 把每个 `<|image_pad|>` 展开成 N 个）。假设 prompt 里有 3 张图，其中 2 张命中缓存，此时只把缺的 1 张图给 HF——如果连同完整 prompt 一起给，HF 会发现 3 个占位符只有 1 张图，直接报错。

所以两条路径的策略不同：

```
无缓存 (True):  文本 + 全部 mm 一起进 HF (_apply_hf_processor_text_mm)
                → HF 顺手把占位符展开好, 返回 is_update_applied=True
                → vLLM 只需 _find_mm_placeholders 定位区间

有缓存 (False): 文本单独 tokenize (_apply_hf_processor_text_only, 不给 mm)
                + 缺失 mm 单独处理 (_apply_hf_processor_mm_only, 不给文本)
                → 返回 is_update_applied=False
                → vLLM 自己 _apply_prompt_updates 做展开,
                  并与缓存中该图的 prompt_updates 合并 (_merge_mm_kwargs)
```

换句话说：`True` = "prompt 和 mm 数据是完整配套的，让 HF 一把处理掉"；`False` = "mm 只是子集，拆开处理，展开工作 vLLM 自己来"。这就是为什么它恰好和无缓存/有缓存一一对应——不是因为功能上绑定，而是因为**只有无缓存时 prompt 和 mm 才是完整的配对**。

---

## 问题 3：`typ` 是怎么变成 `Qwen3VLProcessor` 的

> 原问题：`cached_processor_from_config(model_config, processor_cls=typ, ...)` 代码里面方法接受的参数明明是 `processor_cls=typ`，`if typ is None: typ = ProcessorMixin`，怎么 typ 就变成了 Qwen3VLProcessor 了呢？

`typ` 不是在 `cached_processor_from_config` 内部变的，而是**调用方传进去的**。完整传递链：

```python
# qwen3_vl.py:852  —— 模型自己指定了要用的类
class Qwen3VLProcessingInfo:
    def get_hf_processor(self, **kwargs):
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,      # ← typ 就是在这里被赋值的
            use_fast=..., **kwargs)

# context.py:~190
def get_hf_processor(self, typ, /, **kwargs):
    ...
    return cached_processor_from_config(
        self.model_config,
        processor_cls=typ,          # ← 原样透传 Qwen3VLProcessor
        ...)

# transformers_utils/processor.py:374
def cached_processor_from_config(model_config,
                                 processor_cls=ProcessorMixin,  # 默认值, 用不上
                                 ...):
    return cached_get_processor(..., processor_cls=processor_cls, ...)
```

`if typ is None: typ = ProcessorMixin` 只是 `ctx.get_hf_processor()` 的**兜底默认值**——当某个模型的 ProcessingInfo 不指定具体类时（`self.ctx.get_hf_processor()` 裸调），就用通用的 `ProcessorMixin`，进而在 `get_processor`（:223-224）里走 `AutoProcessor.from_pretrained` 自动推断。Qwen3-VL 显式传了 `Qwen3VLProcessor`，所以走 :232 分支：

```python
elif issubclass(processor_cls, ProcessorMixin):
    processor = processor_cls.from_pretrained(model, ...)   # Qwen3VLProcessor.from_pretrained
```

之所以要显式指定而不靠 AutoProcessor：保证类型确切（`get_image_processor`/`get_video_processor` 要从它身上取子处理器）、可以加 `use_fast=True` 等参数、且类型标注精确。

---

## 问题 4：`call_hf_processor` 的入参 `hf_processor` 就是 Qwen3VLProcessor 吗

> 原问题：InputProcessingContext.call_hf_processor 中的入参 hf_processor，就是前面的 Qwen3VLProcessor 吗？

是的。看基类的调用（`processor.py:1110`）：

```python
return self.info.ctx.call_hf_processor(
    self.info.get_hf_processor(**mm_kwargs),   # ← 返回值就是第一个入参 hf_processor
    ...)
```

`self.info` 是 `Qwen3VLProcessingInfo`，其 `get_hf_processor`（问题 3 的链路）返回的就是 **`Qwen3VLProcessor` 实例**（经 `from_pretrained` 加载并缓存）。所以 `ctx.call_hf_processor` 里：

```python
output = hf_processor(**data, **allowed_kwargs)
```

等价于 `Qwen3VLProcessor(...)(text=prompt, images=..., videos=..., fps=..., return_tensors="pt")`。

`ctx.call_hf_processor` 自己不再关心具体类型——`assert callable(hf_processor)` 即可，它只负责 kwargs 过滤和 dtype 后处理。这是典型的依赖注入：谁实例化（`info.get_hf_processor`）、谁调用（`ctx.call_hf_processor`）分离。

---

## 问题 5：`_get_prompt_updates` 生成的 `PromptReplacement` 详解

> 原问题：Qwen3-VL 覆写的 `_get_prompt_updates` 生成 PromptReplacement 规则（image: 展开成 `grid_thw.prod()//merge²` 个 token）需要仔细解释一下。

先看 Qwen3-VL 返回的两条规则（`qwen3_vl.py:1498-1509`）：

```python
PromptReplacement(
    modality="image",
    target=hf_processor.image_token,          # "<|image_pad|>" 这个字符串
    replacement=get_image_replacement_qwen3vl, # 一个函数, 按 item_idx 计算
)
```

### 它要解决的问题

用户 prompt 里每张图只有**一个** `<|image_pad|>`，但模型实际需要 N 个占位 token（N = 该图经 ViT+merger 后的 token 数）。N 取决于图片分辨率，**只有跑完 HF processor 拿到 `image_grid_thw` 才知道**——所以 replacement 是函数而不是常量。

### replacement 函数的计算

```python
def get_image_replacement_qwen3vl(item_idx: int):
    out_item = out_mm_kwargs["image"][item_idx]   # 第 item_idx 张图的处理结果
    grid_thw = out_item["image_grid_thw"].data    # 例如 [1, 60, 90] (t,h,w 网格)
    num_tokens = int(grid_thw.prod()) // merge_length   # 1×60×90 // (2×2) = 1350
    return [hf_processor.image_token_id] * num_tokens   # [151655] × 1350
```

`grid_thw.prod()` 是 patch 总数，`merge_length = merge_size²` 是 merger 的降采样倍率——除完正好等于 encoder 输出的 embedding 个数，保证"占位 token 数 == 实际 embedding 数"，否则后面 merge embedding 时会错位。

### 这条规则如何被使用

```
PromptReplacement(modality, target, replacement)
        │ resolve(item_idx)   → ResolvedPromptUpdate(target, content, mode=REPLACE)
        ▼
两个用途, 取决于 is_update_applied:

① HF 已展开 (is_update_applied=True, 无缓存路径):
   _find_mm_placeholders —— 拿 target/content 在 prompt_ids 里"找"
   连续的 image_token_id 区段, 记录 PlaceholderRange(offset, length=1350)
   (此时 HF 已经替换过了, vLLM 只是定位)

② HF 未展开 (is_update_applied=False, 缓存路径):
   _apply_prompt_updates —— 真的执行替换:
   "...<|image_pad|>..." → "...<|image_pad|>×1350..."
   同时记录 PlaceholderRange
```

`REPLACE` 模式（对照 `PromptInsertion` 的 `INSERT`）意思是"用 content **替换** target"；INSERT 则是把 content 插到 target 旁边（用于 BosToken 这类场景）。video 的规则同理，只是 replacement 是 `get_video_repl` 生成的带时间戳的 `PromptUpdateDetails(full=..., features=...)`——`full` 是完整替换文本，`features` 是其中真正对应 embedding 的那段（时间戳 token 不算），两者区分使 placeholder 区间精确对应 embedding 数量。

---

## 问题 6：`MultiModalInput.prompt_token_ids: list[int]` 的含义

> 原问题：解释一下 MultiModalInput 对象中 `prompt_token_ids: list[int]` 的含义。

它是**这一条请求最终要喂给模型的、完整的、一维的 token id 序列**——文本 token 和展开后的多模态占位 token 交错排列。

举例，用户发"图1 + 描述一下这张图"：

```
prompt_token_ids =
[<|im_start|>, user_tokens...,           ← chat 模板/文本 token
 <|vision_start|>,
 151655, 151655, ..., 151655,            ← image_pad × 1350 (问题5展开的结果)
 <|vision_end|>,
 描述一下这张图的 token...,
 <|im_end|>, ...]
```

要点：

- **不是 batch 维**：`list[int]` 是一条请求的一维序列（vLLM v1 中 batch 由 GPUModelRunner 在 GPU 侧把多条请求的序列 concat 成扁平张量）；
- **占位 token 已展开**：长度 = 文本 token 数 + Σ 每张图的 encoder 输出 token 数。调度器按这个长度做 token 预算、KV cache 分配，所以它必须是"最终形态"；
- **占位 token 的值本身不重要**（都是 image_token_id），它们永远不会过 embedding 表——`embed_input_ids` 时用 `is_multimodal` 掩码把这些位置替换成 ViT 算出的 image embedding（`_merge_multimodal_embeddings`）；它们存在的意义是**占住位置**，让文本 embedding 和图像 embedding 在序列中对齐；
- 与 `mm_placeholders` 配套：`mm_placeholders={"image": [PlaceholderRange(offset=12, length=1350)]}` 记录"这个序列里第 12~1361 个 token 是第 0 张图"，encoder cache 存取、embedding scatter、partial prefill 断点判断都靠它定位。

---

## 问题 7：`qwen3_vl.py` 中的主要对象及其作用

> 原问题：请给出文件 `C:\Code\vllm\vllm\model_executor\models\qwen3_vl.py` 中的主要对象，并且说明其作用是什么。

`qwen3_vl.py` 中的主要对象（类）按职责分四组：

### 一、视觉编码器（ViT 塔）

| 类 | 行号 | 作用 |
|---|---|---|
| `Qwen3_VisionPatchEmbed` | :347 | 用 Conv3d 把图片/视频帧切成 patch 并投影到 hidden_size（视频按 `temporal_patch_size` 分组） |
| `Qwen3_VisionMLP` | :376 | ViT block 里的 FFN（fc1 → act → fc2） |
| `Qwen3_VisionBlock` | :413 | 一个 ViT transformer block：LayerNorm + `MMEncoderAttention`（varlen，靠 `cu_seqlens`）+ MLP |
| `Qwen3_VisionPatchMerger` | :467 | "Connector"：把 m×m 个 patch 合并降采样，投影到 LLM 的 hidden 维度；`deepstack_merger_list` 用的是带 postshuffle norm 的变体 |
| **`Qwen3_VisionTransformer`** | :519 | **完整 ViT 塔**。组合以上部件：patch_embed → 位置嵌入插值 → N 个 block（在 `deepstack_visual_indexes` 层额外抽中间特征）→ merger。还负责算 `rot_pos_emb`（2D RoPE）、`prepare_encoder_metadata`（eager/cudagraph capture/replay 三路径共用的元数据） |

### 二、多模态输入处理（CPU 侧三件套）

| 类 | 行号 | 作用 |
|---|---|---|
| `Qwen3VLProcessingInfo` | :848 | 配置与工厂：提供 HF config / `Qwen3VLProcessor` / 图片·视频 processor 的获取入口；估算图片/视频的 token 数（`_get_vision_info`、`get_max_video_tokens`），供 profiling 和调度器容量规划 |
| `Qwen3VLDummyInputsBuilder` | :1041 | 启动 profiling 时构造"最坏情况"的假图片/假视频（`get_dummy_mm_data`、`_get_dummy_videos`），用于测显存、定 KV/encoder cache 大小 |
| `Qwen3VLMultiModalProcessor` | :1229 | **请求处理核心**：覆写 `_call_hf_processor`（视频逐条处理 + 时间戳 + EVS token 规划）、`_get_mm_fields_config`（张量 batching 规则）、`_get_prompt_updates`（`<|image_pad|>`/`<|video_pad|>` 展开规则） |

### 三、语言模型封装

| 类 | 行号 | 作用 |
|---|---|---|
| `Qwen3LLMModel` | :1587 | 继承 `Qwen3Model`，forward 增加 `deepstack_input_embeds` 参数——把 deepstack 中间层视觉特征逐层注入 LLM |
| `Qwen3LLMForCausalLM` | :1640 | 继承 `Qwen3ForCausalLM`，组合上面的 model + lm_head + logits_processor；处理 PP（流水线并行）下的权重加载与缺层 |

### 四、顶层模型（与 vLLM 核心对接）

| 类 | 行号 | 作用 |
|---|---|---|
| **`Qwen3VLForConditionalGeneration`** | :1678 | **vLLM 加载的入口类**。组合 `self.visual`（ViT）+ `self.language_model`（LLM），并实现 8 个协议接口 |

它实现的接口及对应职责：

```
SupportsMultiModal        → embed_multimodal: 跑 ViT 产出图片/视频 embedding
                            embed_input_ids: 文本 embedding 与视觉 embedding 合并
SupportsMRoPE             → get_mrope_input_positions: 预计算 3 维位置(时间/高/宽)
                            recompute_mrope_positions: EVS 剪枝后重算位置
SupportsEncoderCudaGraph  → get_encoder_cudagraph_config / item_specs /
                            select_items / prepare_capture/replay /
                            encoder_cudagraph_forward / encoder_eager_forward
SupportsPP                → deepstack buffer 的存取/清理，跨 stage 传 intermediate_tensors
SupportsLoRA              → packed_modules_mapping (qkv_proj/gate_up_proj 打包)
SupportsEagle / Eagle3    → 投机解码支持
SupportsMultiModalPruning → EVS 视频 token 剪枝 (_postprocess_video_embeds_evs)
```

### 五、模块级辅助对象（不是类）

| 对象 | 作用 |
|---|---|
| `triton_pos_embed_interpolate` / `pos_embed_interpolate_native` | ViT 位置嵌入的双线性插值（动态分辨率适配），有 Triton 用 fused kernel |
| `_replace_video_token_placeholders` | 把自造的视频 token 序列拼回文本 token 流 |
| `DUMMY_VIDEO_NUM_FRAMES = 2048` | profiling 假视频的帧数常量 |
| `_cached_tensor` | 带 lru_cache 的小工具，避免重复创建相同张量 |

### 整体关系图

```
Qwen3VLForConditionalGeneration  ← vLLM ModelRegistry 加载的唯一入口
    │
    ├─ self.visual = Qwen3_VisionTransformer        (视觉塔)
    │     ├─ Qwen3_VisionPatchEmbed
    │     ├─ Qwen3_VisionBlock × depth
    │     │     └─ (MMEncoderAttention + Qwen3_VisionMLP)
    │     ├─ Qwen3_VisionPatchMerger                (主 connector)
    │     └─ deepstack_merger_list: PatchMerger × k (deepstack connector)
    │
    ├─ self.language_model = Qwen3LLMForCausalLM    (语言模型)
    │     └─ Qwen3LLMModel (Qwen3 + deepstack 注入)
    │
    └─ 注册三件套 (类装饰器 MULTIMODAL_REGISTRY.register_processor):
          Qwen3VLProcessingInfo      —— 配置/token 数估算
          Qwen3VLDummyInputsBuilder  —— profiling 假数据
          Qwen3VLMultiModalProcessor —— 请求级预处理
```

一句话总结：**顶层类是"协议适配器"，视觉塔和语言模型是"算子"，处理三件套是"输入侧翻译官"**——vLLM 核心只认识顶层类暴露的标准协议方法，所有 Qwen3-VL 特有的逻辑（deepstack、mrope、EVS、时间戳）都封装在这些对象内部。

---

## 问题 8：`Qwen3_VisionTransformer.forward` 详解

> 原问题：请详细解释一下 model_executor.models.qwen3_vl.Qwen3_VisionTransformer.forward 方法都做了哪些事情。

`Qwen3_VisionTransformer.forward`（`qwen3_vl.py:800-841`）是 ViT 塔的核心前向：把**一个 batch 内所有图片/视频帧的 patch（已展平拼接成一条长序列）**一次性跑完 ViT，输出按 merger 降采样后的视觉 embedding（含 deepstack 中间层特征）。

### 0. 输入输出契约

```
输入:
  x:          pixel_values, shape (L, C·T·P·P)
              L = batch 内所有图/视频帧的 patch 总数
              例如 grid_thw=[[1,60,90],[1,40,40]] → L = 5400+1600 = 7000
  grid_thw:   每个 mm item 的 [t, h, w] 网格（patch 数维度）
  encoder_metadata: 可选。eager 时为 None 现场算;
              CUDA graph capture/replay 时由外部预填（固定 shape 的 buffer）

输出:
  hidden_states: (N, out_hidden_size)
              N = Σ t·(h/m)·(w/m)  ← merger 降采样后的 token 总数
              out_hidden_size = D·(1+k)，D=LLM hidden，k=deepstack 层数
              （主特征 + deepstack 特征在 hidden 维上 cat）
```

### 1. 逐段解析

```python
hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
hidden_states = self.patch_embed(hidden_states)
```

**① patch embedding**（`:807-808`）：像素先从 CPU 异步拷到 GPU（`non_blocking`），然后 `Qwen3_VisionPatchEmbed.forward` 把 `(L, C·T·P·P)` reshape 成 `(L, C, 2, 14, 14)`，过 stride=kernel 的 Conv3d——等价于每 2 帧 × 14×14 像素一组做线性投影——得到 `(L, 1152)`。

```python
if encoder_metadata is None:
    encoder_metadata = self.prepare_encoder_metadata(grid_thw_list)
```

**② 元数据准备**（`:810-815`）：这是这个 forward 最关键的设计。`prepare_encoder_metadata`（`:704`）从 `grid_thw` 一次性算出四样东西：

| 元数据 | 内容 | 用途 |
|---|---|---|
| `pos_embeds` | 按每张图 (t,h,w) 双线性插值出的绝对位置嵌入 `(L, 1152)` | 动态分辨率适配 |
| `rotary_pos_emb_cos/sin` | 2D RoPE 的 cos/sin，按 spatial-merge 顺序重排 | block 内 attention 用 |
| `cu_seqlens` | varlen 边界：每张图/每帧的起止 patch 下标 | 告诉 attention "哪些 patch 属于同一张图" |
| `max_seqlen` / `sequence_lengths` | 最大序列长（CPU 标量）/ 各序列长度 | FlashAttention / FlashInfer 后端参数 |

**为什么允许外部传入**：CUDA graph 要求图内没有数据依赖的控制流和动态分配。capture 时用 dummy 元数据录制，replay 时把真实元数据拷进固定 buffer——所以 eager 路径现场算、graph 路径外部给，**同一份 forward 代码三种场景复用**。

```python
pos_embeds = encoder_metadata["pos_embeds"]
hidden_states = hidden_states + pos_embeds
hidden_states = hidden_states.unsqueeze(1)     # (L, 1, 1152)
```

**③ 加位置嵌入**（`:817-819`）：绝对位置嵌入直接加到 patch embedding 上。`unsqueeze(1)` 多出的维度是 `MMEncoderAttention` 包装器约定的形状（seqlen, batch=1, hidden）。

```python
deepstack_feature_lists = []
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(hidden_states,
                        cu_seqlens=..., rotary_pos_emb_cos=..., ...)
    if layer_num in self.deepstack_visual_indexes:
        deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
        deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](hidden_states)
        deepstack_feature_lists.append(deepstack_feature)
```

**④ N 个 transformer block + deepstack 抽取**（`:821-836`）。每个 `Qwen3_VisionBlock.forward` 是标准的 pre-norm 残差结构：

```
x = x + attn(LN1(x), cu_seqlens, cos, sin, ...)   # varlen 全注意力 + 2D RoPE
x = x + mlp(LN2(x))
```

attention 是**图内全注意力**（Qwen3-VL 无 window attention），`cu_seqlens` 保证不同图/不同帧之间不互相注意。

**Deepstack 是 Qwen3-VL 特有的多尺度设计**：在指定的中间层（如第 8/16/24 层），把该层输出额外过一个 `deepstack_merger_list[i]` 存下来。这样 LLM 不仅拿到最后一层的高层语义特征，还能拿到中间层的细粒度特征（对 OCR、小目标有帮助）。

```python
hidden_states = self.merger(hidden_states)
hidden_states = torch.cat([hidden_states] + deepstack_feature_lists, dim=1)
```

**⑤ 主 merger + 拼接**（`:837-840`）。`Qwen3_VisionPatchMerger.forward` 把 `(L, 1, 1152)` view 成 `(L/4, 4×1152)`（m=2，4 个 patch 合一），LayerNorm → fc1 → GELU → fc2 投影到 LLM 的 hidden 维度 D，输出 `(N, D)`。

最后把主特征和 k 个 deepstack 特征在 **hidden 维**上 cat：`(N, D·(1+k))`。

> 注意：deepstack merger 用 `use_postshuffle_norm=True` 变体——先 view 成 `(N, 4×1152)` 再做 norm，而主 merger 是先 norm 再 view，这是对齐 HF 权重行为的细节差异。

### 2. 全流程图

```
pixel_values (L, C·T·P·P)          grid_thw [[t,h,w],...]
        │                                  │
        ▼                                  ▼
  H2D + patch_embed (Conv3d)      prepare_encoder_metadata
        │ (L,1152)                 ├─ pos_embeds (L,1152) ──┐
        │                          ├─ rope cos/sin          │
        │                          ├─ cu_seqlens            │ (eager 现算 /
        ▼                          └─ max_seqlen, seq_lens  │  graph 预填)
  + pos_embeds ◄───────────────────────────────────────────┘
        │ (L,1,1152)
        ▼
  ┌─ for layer_num, blk in blocks: ─────────────────────────┐
  │  x = blk(x, cu_seqlens, cos, sin, max_seqlen, seq_lens) │
  │     = x + Attn(LN x) + MLP(LN x)                        │
  │  if layer_num ∈ deepstack_visual_indexes:               │
  │     deepstack_feat[i] = deepstack_merger[i](x)  → (N,D) │
  └──────────────────────────────────────────────────────────┘
        │ (L,1,1152)
        ▼
  merger (4 patch 合 1, 投影到 D)  → (N, D)
        │
        ▼
  cat([主特征, deepstack×k], dim=1)  → (N, D·(1+k))
        │
        ▼
  返回给 _process_image/video_input → 按 item split
  → LLM 侧 _compute_deepstack_embeds 再把 D 和 k·D 拆开:
    主特征进输入 embedding, deepstack 特征逐层注入 LLM
```

### 3. 与上下游的衔接

- **上游**：`Qwen3VLForConditionalGeneration._process_image_input / _process_video_input` 调它，传拼好的 `pixel_values` 和 `grid_thw`；返回值再按每个 item 的 token 数 `split`。
- **下游**：输出的 `(N, D·(1+k))` 在 `_compute_deepstack_embeds` 里被切成主特征（D，merge 进 input embedding）和 deepstack 特征（k·D，存入 buffer，LLM forward 时逐层加上）。
- **CUDA graph**：`encoder_cudagraph_forward` 也调它，只是提前用 `prepare_encoder_metadata` 填好固定 buffer 的 `encoder_metadata` 传进来，让整段 ViT 前向可以被录制成 graph 重放。

一句话总结：这个 forward 做的是"**打包 batch 内全部视觉 patch → 加动态分辨率位置编码 → varlen ViT → merger 降采样 → 主特征与 deepstack 特征拼接**"，并通过 `encoder_metadata` 参数化把元数据计算外置，使同一段代码能同时服务 eager 和 CUDA graph 两条执行路径。


---

# 附：vLLM ↔ transformers 图像预处理链路问答笔记（Qwen2-VL，2026-08-03）

> 基于 `C:\Code\vllm` 与 `C:\Code\transformers` 源码走读整理，围绕"从 vLLM 启动到 `Qwen2VLImageProcessor._preprocess` 的完整调用链路"以及"如何在不修改 transformers 的前提下给 resize 加缓存"展开。

## 目录

- [Q1. 从 vllm 启动到 Qwen2VLImageProcessor._preprocess 的调用链路](#q1)
- [Q2. call_hf_processor 方法中 hf_processor 参数如何理解](#q2)
- [Q3. 不改 transformers、仅改 vllm 给 _preprocess 的 resize 加 ECCPUConnector 缓存复用的方案](#q3)
- [Q4. _preprocess 中 self.resize 实际在哪个对象中执行？调用链路如何？](#q4)
- [Q5. 为什么包装 processor.image_processor.resize 而不是直接 processor](#q5)
- [Q6. _cached_apply_hf_processor 做什么？缓存 key 是什么？缓存在哪里？](#q6)
- [Q7. BaseMultiModalProcessorCache 详解：作用、抽象方法、子类实现与区别](#q7)
- [Q8. BaseMultiModalProcessor.cache 的具体类型如何确定](#q8)
- [Q9. 什么是 IPC 缓存](#q9)
- [Q10. Qwen3-VL 的 vision ViT 在 P0 还是 P1 进程执行](#q10)
- [Q11. ShmObjectStoreSenderCache 与 ShmObjectStoreReceiverCache 在 P0/P1 侧的实现](#q11)
- [Q12. ShmObjectStoreSenderCache.get_and_update_item 命中/未命中时的 item 与 put 细节](#q12)
- [Q13. `_cached_apply_hf_processor` 返回对象的含义与返回值示例](#q13)
- [Q14. `MultiModalProcessingInfo` 对象中的 `hashes` 字段是怎样得到的](#q14)

---

<a id="q1"></a>
## Q1. 从 vllm 启动到 `Qwen2VLImageProcessor._preprocess` 的调用链路

以 vLLM V1 架构、`vllm serve` 启动 OpenAI 服务、请求中携带图片为例。

### 一、vLLM 启动阶段（引擎与多模态处理器注册）

1. **`vllm serve`** → `vllm/entrypoints/cli/serve.py` → `vllm/entrypoints/openai/api_server.py:110` `build_async_engine_client()`
2. `api_server.py:168` `AsyncLLM.from_vllm_config(...)` 创建 V1 引擎
3. `vllm/v1/engine/async_llm.py:138` 构造 `InputProcessor(self.vllm_config, renderer)`
4. `vllm/renderers/base.py:127` 渲染器初始化时通过 `mm_registry.create_processor(...)` 创建多模态处理器。对于 Qwen2-VL，注册表（`vllm/multimodal/registry.py`）根据模型类型实例化 **`Qwen2VLMultiModalProcessor`**（`vllm/model_executor/models/qwen2_vl.py:1112`，继承 `BaseMultiModalProcessor`）
5. 启动时还会做 profiling/dummy run，同样会走一次下面的链路

### 二、请求阶段（每条带图请求）

```
OpenAI Chat API 请求 (含 image_url)
  └─ serving_chat / Renderer.render_chat()
       └─ Renderer._process_multimodal()                vllm/renderers/base.py:729
            └─ mm_processor.apply(...)                  vllm/renderers/base.py:763
```

离线 `LLM` 类路径则走：`AsyncLLM.add_request()` → `InputProcessor.process_inputs()`（`vllm/v1/engine/input_processor.py:251`）→ `input_preprocessor.preprocess()`（`vllm/inputs/preprocess.py:274`）→ `_process_text()` → `renderer._process_multimodal()`（`preprocess.py:90`），殊途同归。

### 三、vLLM 多模态处理核心（`vllm/multimodal/processing/processor.py`）

```
BaseMultiModalProcessor.apply()                       processor.py:1669
  └─ _cached_apply_hf_processor()                     processor.py:1444   (查 mm 缓存，只处理缺失项)
       └─ _apply_hf_processor_main()                  processor.py:1258   (按 纯文本/纯mm/文本+mm 分发)
            └─ _apply_hf_processor_text_mm()          processor.py:1135
                 └─ _call_hf_processor()              processor.py:1097
                      ├─ self.info.get_hf_processor() → Qwen2VLProcessingInfo.get_hf_processor()
                      │     vllm/model_executor/models/qwen2_vl.py:835
                      │     内部经 ctx.get_hf_processor(Qwen2VLProcessor)
                      │     → AutoProcessor 加载出 transformers 的 Qwen2VLProcessor
                      └─ info.ctx.call_hf_processor() vllm/multimodal/processing/context.py:244
                            → hf_processor(text=prompt, images=..., return_tensors="pt")
                            即调用 transformers 的 Qwen2VLProcessor.__call__()
```

### 四、transformers 侧（到达 `_preprocess`）

```
ProcessorMixin.__call__()                     transformers/processing_utils.py:649
  └─ _process_images(images, **kwargs)        processing_utils.py:762
       └─ self.image_processor(images, ...)   即 Qwen2VLImageProcessor.__call__
            └─ BaseImageProcessor.__call__()  image_processing_utils.py:215  (→ self.preprocess)
                 └─ Qwen2VLImageProcessor.preprocess()     models/qwen2_vl/image_processing_qwen2_vl.py:141
                      └─ BaseImageProcessor.preprocess()   image_processing_utils.py:383
                           (校验/标准化 kwargs)
                           └─ _preprocess_image_like_inputs()  image_processing_utils.py:297
                                ├─ _prepare_image_like_inputs()  (PIL/np → torch.Tensor, 通道前置)
                                └─ ★ Qwen2VLImageProcessor._preprocess()  image_processing_qwen2_vl.py:148
                                     (smart_resize → resize → rescale_and_normalize
                                      → reshape/permute 成 patches → 返回 pixel_values + image_grid_thw)
```

### 五、处理结果回流

`_preprocess` 返回的 `pixel_values` / `image_grid_thw` 作为 `BatchFeature` 一路返回到 `BaseMultiModalProcessor.apply()`，在那里：

1. `MultiModalKwargsItems.from_hf_inputs()` 把 HF 输出转成 vLLM 内部 mm kwargs（`processor.py:1416`）
2. `_maybe_apply_prompt_updates()` 用 `<|image_pad|>` 占位符按图像 token 数展开（`processor.py:1695`）
3. 生成 `MultiModalInput`（token ids + mm_kwargs + mm_placeholders）交给 `InputProcessor.process_inputs()` 包装成 `EngineCoreRequest`，送入调度器

**补充说明**：这个版本的 transformers 中 `Qwen2VLProcessor` 没有自己覆写 `__call__`，图像处理完全走 `ProcessorMixin` 的通用 `_process_images` 路径；真正模型定制的部分就在 `Qwen2VLImageProcessor._preprocess`（patch 分组 reshape）和 vLLM 侧的 `Qwen2VLMultiModalProcessor._get_mm_fields_config`（`qwen2_vl.py:1148`，声明 `pixel_values`/`image_grid_thw` 的字段映射）。另外 `image_processing_pil_qwen2_vl.py` 里的 `Qwen2VLImageProcessorPil` 是 PIL 后端的同类实现，链路结构相同，区别只在后端基类（`TorchvisionBackend` vs `PilBackend`）。

---

<a id="q2"></a>
## Q2. `call_hf_processor` 方法中 `hf_processor` 参数如何理解

`hf_processor` 就是**从 HuggingFace 加载出来的那个模型 Processor 实例**——对 Qwen2-VL 来说，就是 transformers 里的 `Qwen2VLProcessor` 对象。可从四个层面理解：

### 1. 它是谁：HF 侧的 Processor 实例

类型注解 `Callable[..., BatchFeature] | ProcessorMixin` 说明它可以是：

- **`ProcessorMixin` 的子类实例**（绝大多数情况）——如 `Qwen2VLProcessor`、`LlavaNextProcessor` 等。它内部打包了 `tokenizer` + `image_processor` + `video_processor`，是"文本+多模态"的统一预处理入口。
- **任意可调用对象**——只要签名是 `(**data, **kwargs) -> BatchFeature` 即可，给一些不走标准 HF Processor 的模型留了余地（比如某些模型覆写 `_call_hf_processor` 传入自定义函数）。

### 2. 它从哪来：`_call_hf_processor` 里现取

```python
# processor.py:1110
return self.info.ctx.call_hf_processor(
    self.info.get_hf_processor(**mm_kwargs),   # ← 就是这里拿到的
    dict(text=prompt, **mm_data),              # → 变成 hf_processor 的 data 参数
    dict(**mm_kwargs, **tok_kwargs),
)
```

对 Qwen2-VL：

- `Qwen2VLProcessingInfo.get_hf_processor()`（`qwen2_vl.py:835`）→ `ctx.get_hf_processor(Qwen2VLProcessor)`（`context.py:179`）
- `ctx.get_hf_processor` 内部通过 `AutoProcessor` 从模型目录（`preprocessor_config.json`、`chat_template.json` 等）加载实例，并**校验它确实是指定类型**，然后缓存复用（不是每个请求都重新加载）

所以 `call_hf_processor` 自己并不创建 processor，它只是个"带参数过滤和错误包装的调用器"。

### 3. 它怎么被用：函数体内的唯一用途

```python
output = hf_processor(**data, **allowed_kwargs)   # context.py:270
```

这一行等价于调用 `Qwen2VLProcessor.__call__(text=prompt, images=[...], videos=[...], return_tensors="pt", ...)`，即进入 transformers 链路（`processing_utils.py:649` → `_process_images` → `Qwen2VLImageProcessor._preprocess`）。

调用前 vLLM 做了两件事：

- `get_allowed_kwarg_only_overrides(...)`（`context.py:261`）：过滤 `mm_processor_kwargs` 中 HF processor 实际接受的参数，避免传入不认的 kwargs 报错
- `allowed_kwargs.setdefault("return_tensors", "pt")`：强制让 HF 返回 torch 张量，供后续 vLLM 组 batch 使用

### 4. 为什么设计成参数传入，而不是方法内部自取

把 processor 作为参数传入 `call_hf_processor`，是**策略分离**：`InputProcessingContext` 只负责"怎么安全地调用一个 HF processor"（kwargs 合并、过滤、dtype 后处理、异常包装成 `ValueError`），而"用哪个 processor、传什么初始化参数"由模型的 `ProcessingInfo.get_hf_processor()` 决定。这样同一个 `call_hf_processor` 可以服务所有模型。

`num_tries` / `max_tries` 参数则是给会覆写此方法的子类预留的重试控制（默认实现中 `num_tries=1` 即不重试，加载/调用失败直接抛带上下文的 `ValueError`）。

**一句话总结**：`hf_processor` 是 HF `Qwen2VLProcessor` 实例本身，`call_hf_processor` 是 vLLM 包装的一次受控 `__call__` 调用，是 vLLM 世界跨入 transformers 世界的那个边界点。

---

<a id="q3"></a>
## Q3. 不改 transformers、仅改 vllm 给 `_preprocess` 的 resize 加 ECCPUConnector 缓存复用的方案

需求：在 `_preprocess` 的 `stacked_images = self.resize(...)` 前后加缓存（执行前读缓存，miss 则 resize，之后写缓存），存储用 ECCPUConnector，且不修改 transformers 目录。

核心思路：**不碰 `_preprocess` 本体，而是包装它内部调用的 `self.resize` 实例方法**，再在 vLLM 侧找一个合适的注入点把包装装上去。

### 一、拦截点：包装 `resize`，而不是覆写 `_preprocess`

`_preprocess` 里的调用是 `self.resize(...)`——这是通过实例的动态分派。所以只要在实例上替换 `resize` 属性，就能精确实现"执行前读缓存 → miss 则 resize → 写缓存"，而且：

- 不用拷贝 `_preprocess` 那 ~70 行分组/reshape/permute 逻辑，**transformers 升级后不会漂移**；
- 缓存命中时直接返回，跳过的正好是计算最重的部分。

对比其他候选拦截点：

| 方案 | 问题 |
|---|---|
| 子类化 `Qwen2VLImageProcessor` 覆写 `_preprocess` | 整体 fork 上游逻辑，随 transformers 版本漂移 |
| 全局 monkey-patch 模块级类方法 | 影响所有模型/所有进程，不可控 |
| 覆写 vllm 的 `_call_hf_processor` | 粒度太粗，拿不到 resize 前后的中间态 |
| **包装实例的 `resize`** | ✅ 粒度恰好、改动面最小 |

### 二、注入点：修改 `Qwen2VLProcessingInfo.get_hf_processor`

vLLM 拿 HF processor 的唯一入口就是这里（`vllm/model_executor/models/qwen2_vl.py:835`），processor 实例由 vllm 的 ctx 缓存复用，所以包装只会发生一次、且只对 Qwen2-VL 生效：

```python
# vllm/model_executor/models/qwen2_vl.py
class Qwen2VLProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Qwen2VLProcessor:
        processor = self.ctx.get_hf_processor(          # 保持原有调用不变
            Qwen2VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )
        from vllm.multimodal.resize_cache import maybe_wrap_resize_cache
        maybe_wrap_resize_cache(processor.image_processor, self.ctx)   # 新增
        return processor
```

注意 vllm 默认 `use_fast=True`，走的是 `TorchvisionBackend` 那个 `_preprocess`；如果关心 PIL 后端，同样包装 `Qwen2VLImageProcessorPil.resize` 即可，包装器是通用的。

> **教训（见 Q5）**：不要写成 `super().get_hf_processor(Qwen2VLProcessor, ...)`——父类 `BaseProcessingInfo.get_hf_processor` 只接受 `**kwargs`，没有 `typ` 位置参数，那样写会 TypeError。既然 `qwen2_vl.py` 在 vllm 仓库内允许直接改，直接改原方法体最简洁。

### 三、包装器模块（新增 `vllm/multimodal/resize_cache.py`）

```python
import functools
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)
_WRAP_FLAG = "_vllm_resize_cache_wrapped"

def _make_key(image: torch.Tensor, size, resample) -> str:
    import blake3  # vllm 已依赖 blake3（MultiModalHasher 用的就是它）
    h = blake3.blake3()
    img = image.contiguous()
    h.update(memoryview(img.numpy().tobytes()))
    h.update(f"{size.height}x{size.width}:{resample}:{image.dtype}".encode())
    return h.hexdigest()

def maybe_wrap_resize_cache(image_processor, ctx) -> None:
    if getattr(image_processor, _WRAP_FLAG, False):   # 幂等，防止重复包装
        return
    cache = get_resize_cache(ctx)                      # 见第四节
    if cache is None:
        return
    orig_resize = image_processor.resize

    @functools.wraps(orig_resize)
    def cached_resize(image, size, resample=None, **kwargs):
        key = _make_key(image, size, resample)
        hit = cache.get(key)
        if hit is not None:
            return hit.clone()          # 防止下游原地修改污染缓存
        out = orig_resize(image=image, size=size, resample=resample, **kwargs)
        cache.put(key, out)
        return out

    image_processor.resize = cached_resize
    setattr(image_processor, _WRAP_FLAG, True)
    logger.info("Qwen2VLImageProcessor.resize wrapped with EC cache")
```

**Key 设计要点**：`_preprocess` 里 `stacked_images` 是同 shape 图片堆叠的批次张量，所以 key = 张量内容哈希 + 目标 `(resized_height, resized_width)` + `resample` + dtype，缺一不可。哈希一次大张量有 CPU 开销，但远小于 resize 本身（双三次插值），仍划算。

### 四、缓存存储层：怎么和 ECCPUConnector 结合

**`ECCPUConnector` 本体不适合直接拿来用**，原因有三：

1. **进程不匹配**：它是 role 分离设计——`SCHEDULER` 角色在 EngineCore 进程、`WORKER` 角色在 worker 进程（`factory.py` 注释写得很明确）。而 `_preprocess` 运行在**前端进程**（Renderer / `InputProcessor.process_inputs` 所在进程），那里没有 connector 实例；
2. **block 形状不匹配**：block 大小按 `hidden_dim * element_size` 设计（`common.py:_get_encoder_cache_hidden_dim`），是为 ViT 输出嵌入准备的；
3. **语义不匹配**：它的读写是 step 同步的 GPU↔mmap DMA（`StepTracker`、pin/unpin），而 resize 缓存要的是简单的 CPU 张量 KV。

按优雅程度排序的三个落法：

- **方案 B（推荐）：复用底层 `ECSharedRegion` + `EmbeddingCache`，建一个独立的 resize 专用 region。** `ECSharedRegion` 是 `/dev/shm/vllm_ec_{engine_id}.mmap` 的 MAP_SHARED mmap，天然跨进程共享——这正好解决 vLLM 起多个 API/renderer 进程（`api_process_rank`）时缓存不能共享的问题。新模块里用相同 engine_id 规则、自己的文件名和 block 尺寸（按 resize 输出张量典型大小）创建 region，配 `EmbeddingCache` 的 LRU 逐出逻辑：

```python
def get_resize_cache(ctx):
    global _cache
    if _cache is None:
        region = ECSharedRegion(engine_id=f"{...}_resize", ...)
        _cache = ResizeTensorCache(region)   # get/put: 张量序列化进 block，LRU 逐出
    return _cache
```

- **方案 C：给 `ECCPUConnector` 增加通用 CPU 张量 KV 接口**（`get_tensor(key) / put_tensor(key, tensor)`），并允许在前端进程以第三种角色实例化（attach 到同一 mmap）。改动更"正统"但侵入面大，要动 role 模型和 metadata 协议。
- **方案 A（最简，先验证收益）**：先用进程内 `collections.OrderedDict` LRU 跑通整条链路，确认命中率收益后再升级到方案 B 的 mmap 共享。

### 五、其他注意事项

1. **线程安全**：renderer 可能并发处理请求，`cache.get/put` 需要加锁（`EmbeddingCache` 目前假设单线程 scheduler，直接用要包一层锁）；
2. **开关默认关闭**：用 env var 或 `mm_processor_kwargs` 控制，不启用时 `maybe_wrap_resize_cache` 直接返回，零行为变化；
3. **命中返回 clone**：缓存里的张量可能被后续 `rescale_and_normalize` 等步骤原地修改，返回副本更安全（开销远小于 resize）；
4. **一致性自检**：建议先写个单测——同一图片连续两次 `_preprocess`，第二次的 `pixel_values`/`image_grid_thw` 必须与未包装时 bitwise 一致。

**总结**：拦截在 `resize`、注入在 `Qwen2VLProcessingInfo.get_hf_processor`、存储复用 `ECSharedRegion` 而非 `ECCPUConnector` 本体——transformers 目录零改动，vllm 侧只动 `qwen2_vl.py` 一处加一个独立新模块。

---

<a id="q4"></a>
## Q4. `_preprocess` 中 `self.resize` 实际在哪个对象中执行？调用链路如何？

`self.resize` 里的 `self` 是 **`Qwen2VLImageProcessor` 的实例**，但 `resize` 方法本身并不在这个类里——它来自父类 **`TorchvisionBackend`**。

### 一、方法归属（MRO 查找）

```
Qwen2VLImageProcessor            models/qwen2_vl/image_processing_qwen2_vl.py:92
  └─ TorchvisionBackend          image_processing_backends.py:86   ← resize 定义在这里 (line 205)
       └─ BaseImageProcessor     image_processing_utils.py:60
            └─ ImageProcessingMixin
```

`Qwen2VLImageProcessor` 只覆写了 `_standardize_kwargs` / `preprocess` / `_preprocess`，**没有覆写 `resize`**，所以 Python 沿 MRO 找到的第一个实现是 **`TorchvisionBackend.resize()`（`transformers/image_processing_backends.py:205`）**。

### 二、`TorchvisionBackend.resize` 内部执行逻辑

对照 `_preprocess` 的实参（`size=SizeDict(height=resized_height, width=resized_width)`, `resample=resample`）逐条走：

1. **插值方式转换**（`image_processing_backends.py:215-221`）：
   - `Qwen2VLImageProcessor.resample = PILImageResampling.BICUBIC`（类属性，`image_processing_qwen2_vl.py:94`）
   - 经 `pil_torch_interpolation_mapping` 映射为 `tvF.InterpolationMode.BICUBIC`
2. **目标尺寸分支**（`image_processing_backends.py:231-252`）：由于传入的 `SizeDict` 带的是 `height`/`width`，命中 `elif size.height and size.width:` 分支（line 246-247）→ `new_size = (resized_height, resized_width)`。注意 `shortest_edge/longest_edge` 分支**不会**走到——宽高已在 `_preprocess` 里由 `smart_resize` 算好，`resize` 只做纯缩放。
3. **真正计算**（`image_processing_backends.py:257`）：

```python
return tvF.resize(image, new_size, interpolation=interpolation, antialias=True)
```

即 **`torchvision.transforms.v2.functional.resize`**（文件顶部 `from torchvision.transforms.v2 import functional as tvF`，line 74），输入是按 shape 分组堆叠的 `(B, C, H, W)` CPU 张量，底层走 torch 的 interpolate 算子批量完成。

### 三、完整调用链

```
(vllm 侧 ... → ProcessorMixin.__call__ → _process_images)
  └─ Qwen2VLImageProcessor.__call__          image_processing_utils.py:215
       └─ BaseImageProcessor.preprocess      image_processing_utils.py:383
            └─ _preprocess_image_like_inputs image_processing_utils.py:297
                 └─ Qwen2VLImageProcessor._preprocess      image_processing_qwen2_vl.py:148
                      └─ self.resize(...)   ← 动态分派
                           └─ TorchvisionBackend.resize    image_processing_backends.py:205
                                └─ torchvision.transforms.v2.functional.resize  (tvF.resize)
                                     └─ torch interpolate (CPU 批量张量缩放)
```

### 四、两点补充

1. **存在孪生实现**：`image_processing_backends.py:521` 的 `PilBackend.resize` 是 PIL 后端的对应实现，服务于 `Qwen2VLImageProcessorPil`。vLLM 在 `Qwen2VLProcessingInfo.get_hf_processor` 里默认 `use_fast=True`，加载的是 Torchvision 版本。
2. **对缓存方案的印证**：包装的正是 `TorchvisionBackend.resize` 这个实例方法——它是 `_preprocess` 内唯一的重计算点，且入参（堆叠张量 + 目标 h/w + resample，antialias 恒为默认 `True`）完全决定输出，作为缓存 key 是完备的。包装实例属性后，`self.resize(...)` 的动态分派会先命中包装函数，未命中再落入 `TorchvisionBackend.resize` 原实现。

---

<a id="q5"></a>
## Q5. 为什么包装 `processor.image_processor.resize` 而不是直接 `processor`

`processor.image_processor` 这一层是必须去的。关键在于：**要拦截的 `self.resize`，它的 `self` 是 `Qwen2VLImageProcessor` 实例，而不是 `Qwen2VLProcessor` 实例**——这是两个不同的对象。

### 对象关系

```python
processor = Qwen2VLProcessor(...)        # ProcessorMixin 子类，一个"容器"
processor.image_processor                # = Qwen2VLImageProcessor 实例 ← resize 挂在这里
processor.tokenizer                      # = Qwen2TokenizerFast
processor.video_processor                # = Qwen2VLVideoProcessor
```

`Qwen2VLProcessor`（`ProcessorMixin`）本身**没有 `resize` 方法**。`_preprocess` 里的 `self.resize(...)` 是在 `Qwen2VLImageProcessor` 实例上动态分派的。执行路径：

```
vllm call_hf_processor:  processor(text=..., images=...)          # Qwen2VLProcessor.__call__
  └─ ProcessorMixin._process_images:  self.image_processor(images, ...)   processing_utils.py:763
       └─ Qwen2VLImageProcessor.preprocess → _preprocess
            └─ self.resize(...)   ← self 是 image_processor，不是 processor
```

所以：

- 如果写 `processor.resize = cached_resize` → 只是给容器对象挂了一个永远没人调用的属性，`self.resize` 的分派路径根本不经过它，**拦截无效**；
- 写 `maybe_wrap_resize_cache(processor, ...)` 同理，包装器内部取 `processor.resize` 会 `AttributeError`。

### 什么时候"直接 processor"才对

如果想把缓存粒度**上移**，拦截点才轮到 `processor` 本身——但那拦截的是另一个方法：

| 拦截点 | 缓存对象 | 粒度 |
|---|---|---|
| `processor.image_processor.resize` | resize 后的中间张量 | 细（只省 resize） |
| `processor.__call__`（或覆写 vllm 的 `_call_hf_processor`） | 整个 `BatchFeature`（pixel_values + grid_thw + input_ids） | 粗（省掉整个 HF 预处理） |

第二种也是合法设计，甚至更简单——但那是"整图预处理结果缓存"，不是"在 `resize` 前后加缓存"。两者 key 的构造、命中收益、内存占用都不同。

顺带一提：vLLM 自己在 `_cached_apply_hf_processor`（`processor.py:1444`）就有一层按 `mm_hash` 缓存 HF processor 输出的机制（mm processor cache）。选择在 resize 这一层做，价值在于即使上层缓存关闭/失效时仍能省掉重计算——这也再次说明拦截点必须在 `image_processor.resize` 上，否则就和 vLLM 已有的那层粒度重复了。

---

<a id="q6"></a>
## Q6. `_cached_apply_hf_processor` 做什么？缓存 key 是什么？缓存在哪里？

### 一、主要做什么

它是 `BaseMultiModalProcessor.apply()` 的第一步（`processor.py:1691`），作用是**以"多模态条目"为粒度，避免对重复图片/视频重复执行 HF processor**（即避免重复走 `_preprocess`、resize、tokenize 这一整条重链路）。流程（`processor.py:1444-1516`）：

1. `inputs.get_mm_hashes(...)`：给请求里**每一个** mm item（每张图、每个视频）算出 `mm_hash`；
2. `_get_cache_missing_items(...)`：逐项查缓存，把 mm items 分成"已缓存"和"缺失"两组；
3. `_apply_hf_processor_main(..., mm_items=mm_missing_data_items)`：**只对缺失的条目**调用 HF processor（`enable_hf_prompt_update=False`，因为缺失子集和 prompt 不对应）；
4. `_merge_mm_kwargs(...)`：把缓存命中结果和新算出的结果按原顺序合并，新条目同时写入缓存；
5. 返回 `(prompt_ids, mm_info, is_update_applied)`，供后续 `_maybe_apply_prompt_updates` 展开 `<|image_pad|>` 占位符。

如果缓存被禁用（`self.cache is None`）或数据是 passthrough 类型，直接退化为全量 `_apply_hf_processor`（`processor.py:1456-1457`）。

### 二、缓存匹配的 key

key 是每个 mm item 的 **`mm_hash` 字符串**，计算在 `ProcessorInputs.get_mm_hashes`（`processing/inputs.py:26`）→ `MultiModalHasher.hash_kwargs`（`multimodal/hasher.py:166`），参与哈希的成分：

| 成分 | 说明 |
|---|---|
| `model_id` | 模型路径/名字，防止跨模型串用 |
| `modality: item` | 模态名 + **条目内容序列化**：PIL 图片哈希像素数据（`mode` + `np.asarray` + palette，EXIF 里有 UUID 则直接用）；`MediaWithBytes` 用原始字节；tensor/ndarray 用 shape+dtype+底层字节（`hasher.py:53-143`） |
| `hf_processor_mm_kwargs` | 用户传的 processor 参数（如 `min_pixels`/`max_pixels`），参数不同则 hash 不同 |

哈希算法由 mm 配置的 `mm_hasher_algorithm` 决定（默认 blake3）。另外如果调用方提供了 `multi_modal_uuids`，uuid 直接作为 hash 用（但此时若同时传了 processor kwargs，仍会把 uuid 和 kwargs 混合再哈希，`inputs.py:47-61`）。

要点：**key 是"原始图片内容 + 处理参数"的哈希，发生在 HF processor 之前**——同一张原图重复出现才命中；图片内容差一个字节就 miss。

### 三、缓存存在哪里

存活在 **P0 进程**（执行 input processing 的进程，即前端/Renderer 所在进程），按部署形态有三种实现（`multimodal/cache.py`）：

1. **`MultiModalProcessorOnlyCache`**（`cache.py:326`，默认形态）：进程内 `LRUCache[str, MultiModalProcessorCacheItem]`（`vllm/utils/cache.py`），容量由 `--mm-processor-cache-gb` 控制，按条目字节数计量、LRU 逐出。value 是 `(mm_kwargs, prompt_updates)`——**包含 pixel_values 等张量本体和占位符展开信息**，命中后连 prompt update 都不用重算。
2. **`MultiModalProcessorSenderCache`** / **`ShmObjectStoreSenderCache`**（`cache.py:379/437`，启用 IPC caching 时）：P0 侧把条目（或条目的地址句柄）序列化进**共享内存对象存储**（`SingleWriterShmObjectStorage`，`shm_object_storage.py`），key 仍是 mm_hash，value 变成 `(address, monotonic_id)` 句柄；
3. 对应的 **ReceiverCache** 在引擎核心进程（P1）侧按地址从共享内存读回张量，避免跨进程反复拷贝大 tensor。

生命周期上，cache 实例由 `BaseMultiModalProcessor.__init__` 根据 `model_config.get_multimodal_config()` 创建（实际由 Renderer 注入，见 Q8），挂在 mm processor 上，随引擎进程存活；`clear_cache()` / `close()` 在引擎关闭时调用。

### 四、和 resize 缓存方案的关系（重要）

这套机制意味着：**只要 `--mm-processor-cache-gb > 0`（默认开启），重复原图根本不会进入 `_preprocess`**——包装的 resize 第二次都不会被调用。所以 resize 级缓存的增量价值主要在：

- mm processor cache 被关闭或容量太小被逐出的场景；
- 原图不同但 resize 输入张量恰好相同的场景（实际上极少见，因为 key 都是内容哈希）。

换句话说，vLLM 已有的这层缓存和 resize 缓存方案 key 语义高度重叠，但缓存对象是"整个 HF 输出"而非"resize 中间张量"。动手前建议先评估现有 mm processor cache 的命中率是否已经不满足需求——如果瓶颈是**首次**处理的 resize 耗时（而非重复图片），resize 级缓存帮不上忙，优化方向应该是算子本身（如 batch/GPU resize）；如果瓶颈确实是重复图片但上层缓存因故不可用，该方案才有明确收益。

---

<a id="q7"></a>
## Q7. `BaseMultiModalProcessorCache` 详解：作用、抽象方法、子类实现与区别

### 一、作用

`BaseMultiModalProcessorCache`（`vllm/multimodal/cache.py:262`）是 **P0（前端进程）侧的 mm 处理器缓存抽象**，即 `_cached_apply_hf_processor` 里 `self.cache` 的类型。解决的问题：同一张图重复出现在不同请求中时，避免在 P0 重复执行 HF processor（含 `_preprocess`/resize/tokenize），同时避免把处理好的大张量反复通过 IPC 发给引擎核心进程（P1）。

设计基石写在祖父类 `BaseMultiModalCache` 的 docstring 里（`cache.py:175-198`）——**P0/P1 缓存镜像**：

```
              is_cached() x N     get_and_update()
P0: From API -----------------> -----------------> To P1
             get_and_update()
P1: From P0 -----------------> To model
```

- `is_cached()` 只查不碰，可在 P0 随意调用；
- `get_and_update()` 在 P0 和 P1 **按相同顺序依次调用**，两边 LRU 逐出顺序因此完全一致；
- 这样 P0 查自己的缓存就知道 P1 有没有该条目，**无需跨进程通信**。

在 `_cached_apply_hf_processor` 中的使用方式（`processor.py:1444+`）：先用 `is_cached` 系列把 mm items 分成命中/缺失两组 → 只对缺失组跑 HF processor → 合并时通过 `get_and_update_item` 把新条目写入缓存并返回最终数据。

### 二、抽象出来的方法

**继承自 `BaseMultiModalCache`**：

| 方法 | 作用 |
|---|---|
| `get_and_update_item(mm_item, mm_hash)` | 核心操作。命中则返回缓存值；未命中则把输入存入缓存并原样返回。**会更新逐出顺序**（LRU touch + 写入）。返回 `(item, prompt_updates)` |
| `get_and_update(items, hashes)` | 批量版，非抽象，逐条调上面那个 |
| `clear_cache()` | 清空底层缓存 |

**`BaseMultiModalProcessorCache` 新增**（`cache.py:262-323`）：

| 方法 | 作用 |
|---|---|
| `is_cached_item(mm_hash)` | 纯查询，**不改变逐出顺序**。`_get_cache_missing_items` 靠它做分组 |
| `is_cached(hashes)` | 批量版 |
| `touch_sender_cache_item(mm_hash)` | 只刷新逐出顺序不传数据。用于"已知 P1 有这份数据"的场景（如后续请求只带 uuid/hash），防止条目被提前逐出 |
| `make_stats(delta)` | 命中率统计，供 `Renderer._process_multimodal` 后的 `update_mm_cache_stats` 上报 |
| `close()` | 释放底层资源（shm 实现需要），默认 no-op |

类型约束：输入 `_I = tuple[MultiModalKwargsItem, Sequence[ResolvedPromptUpdate]] | None`，输出 `_O = tuple[MultiModalKwargsItem | None, Sequence[ResolvedPromptUpdate]]`——每个缓存条目是 **(处理后的张量数据, 占位符展开信息)** 的二元组。

### 三、三个子类实现及存储方式

#### 1. `MultiModalProcessorOnlyCache`（type=`"processor_only"`，`cache.py:326`）

- **存储**：P0 进程内 `LRUCache[str, MultiModalProcessorCacheItem]`，容量 `--mm-processor-cache-gb`，按 `tensor.nbytes` 计量逐出（`MultiModalCache.get_lru_cache`，`cache.py:158`）。
- **存什么**：**完整数据**——`item`（pixel_values 等张量本体）+ `prompt_updates`。
- **命中行为**：直接返回缓存的完整条目，照常经 IPC 发给 P1。**省了重算，不省传输**。
- **适用**：不支持 IPC 缓存的部署（多 API 进程、或 DP>1 且无外部负载均衡，`registry.py:283-289`）——此时 P1 侧没有对应缓存，只能 P0 自存自发。

#### 2. `MultiModalProcessorSenderCache`（type=`"lru"`，`cache.py:379`）

- **存储**：P0 进程内 `LRUCache[str, MultiModalProcessorCacheItemMetadata]`。
- **存什么**：**只存元数据**——`item_size`（字节数，用于保持和 P1 一致的逐出策略）+ `prompt_updates`（必须留 P0，因为有些模型的 prompt update 依赖张量数据，`cache.py:62-74` 注释）。**张量本体不留在 P0**。
- **命中行为**：返回 `(None, prompt_updates)`——item 为 None，IPC 时**不传输张量**，P1 侧的 `MultiModalReceiverCache`（存有真实张量的镜像 LRU）按相同 hash 自己取。
- **收益**：P0 内存占用小 + 命中时零张量传输；代价是张量在 P0/P1 各存一份（首次未命中时要传一次）。

#### 3. `ShmObjectStoreSenderCache`（type=`"shm"`，`cache.py:437`）

- **存储**：共享内存对象库——`SingleWriterShmObjectStorage` 架在 `SingleWriterShmRingBuffer` 上（`/dev/shm` 环形缓冲，名字取 `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME`），msgpack 序列化，容量同 `--mm-processor-cache-gb`，单对象上限 `--mm-shm-cache-max-object-size-mb`。`mm_hash → (address, monotonic_id)` 的索引；`prompt_updates` 单独放 P0 本地 dict `_p0_cache`（并定期清理已不在 shm 里的悬空条目，`remove_dangling_items`）。
- **存什么**：张量写入共享内存一次，P0 返回的是 `address_as_item()` 构造的"地址条目"——`MultiModalKwargsItem({"address": ..., "monotonic_id": ...})`，**不含张量**。
- **命中行为**：命中直接拿地址；未命中 `put` 进 shm 拿地址。worker 进程的 `ShmObjectStoreReceiverCache`（`cache.py:678`）按地址从 shm 读回张量（`n_readers=world_size`，带 reader lock 支持 TP 多 worker 并发读）。
- **收益**：张量**只写一次、所有进程共享**，多 worker 不需要每个进程一份拷贝；写满或对象超限时优雅降级为不缓存（`cache.py:514-535`）。

### 四、三者区别一览

| | processor_only | lru | shm |
|---|---|---|---|
| 张量存哪 | P0 进程内存 | P1 进程内存（P0 只存 size 元数据） | /dev/shm 共享内存（一份，全进程共享） |
| P0 内存开销 | 高（完整张量） | 低（仅元数据） | 低（仅地址+索引） |
| 命中时张量传输 | 经 IPC 全量发 P1 | 不传（P1 自取） | 不传（worker 按地址读） |
| 冗余份数 | 1 | 2（P0 miss 时发一次，P1 存一份） | 1 |
| TP 多 worker | — | 每 worker 各自缓存 | 单写多读，天然支持 |
| prompt_updates 存哪 | P0 | P0 | P0（`_p0_cache`） |
| 配套 P1 侧实现 | 无 | `MultiModalReceiverCache`（引擎进程 LRU） | `ShmObjectStoreReceiverCache`（worker 进程 shm reader） |

选型由 `MultiModalRegistry._get_cache_type` 决定（`registry.py:268`）：模型不支持多模态或 `--mm-processor-cache-gb <= 0` → 无缓存（`_cached_apply_hf_processor` 退化为全量处理）；IPC 不受支持 → 强制 `processor_only`；否则取 `--mm-processor-cache-type`（`lru`/`shm`）。

### 五、与 resize 缓存方案的关系

注意这三个实现缓存的 key 都是 `mm_hash`（原图内容哈希），value 都是**整个 HF processor 的输出**——与 resize 级缓存是同一命中条件、不同缓存对象。如果要用 `ECCPUConnector` 体系做 resize 缓存，`ShmObjectStoreSenderCache` 的"单写者 shm 环形缓冲 + 地址句柄 + P0 侧元数据字典"模式是最值得参考的现成范本，比重用 ECCPUConnector 的 block/StepTracker 机制更贴合前端预处理场景。

---

<a id="q8"></a>
## Q8. `BaseMultiModalProcessor.cache` 的具体类型如何确定

`BaseMultiModalProcessor` 自己**不做决定**——它的 `__init__` 只是把 `cache` 作为参数存下来（`processor.py:984-990`），类型完全由**创建它的上游**决定。

### 一、注入链路：cache 从哪来

```
Renderer.__init__                                   vllm/renderers/base.py:124
  ├─ mm_registry.processor_cache_from_config(config)   ← 决策发生在这里
  └─ mm_registry.create_processor(..., cache=mm_processor_cache)   base.py:127
       └─ factories.build_processor(ctx, cache=cache)              registry.py:230
            └─ BaseMultiModalProcessor.__init__(..., cache=cache)  processor.py:979
                 └─ self.cache = cache   ← 纯透传
```

注意 `create_processor` 的 `cache` 参数默认是 `None`（`registry.py:216`）——比如 profiling 用的 dummy processor、或调用方没传时，`self.cache` 就是 `None`，`_cached_apply_hf_processor` 直接退化为全量处理（`processor.py:1456`）。所以真正的决策点在 **`MultiModalRegistry._get_cache_type`**。

### 二、决策点：`_get_cache_type`（registry.py:268-292）

按顺序判断，返回四种结果之一：

```
1. 模型不支持多模态?                → None（无缓存）
2. mm_processor_cache_gb <= 0?      → None（用户禁用）
3. IPC 缓存不受支持?                → "processor_only"
   条件: _api_process_count > 1
        或 (DP > 1 且无外部负载均衡)
4. 否则                             → mm_processor_cache_type 配置值
                                      ("lru" 或 "shm")
```

然后 `processor_cache_from_config`（`registry.py:294-309`）按返回值实例化：

| `_get_cache_type` 返回 | 实例化的类 |
|---|---|
| `None` | `None`（`self.cache = None`） |
| `"processor_only"` | `MultiModalProcessorOnlyCache` |
| `"lru"` | `MultiModalProcessorSenderCache` |
| `"shm"` | `ShmObjectStoreSenderCache` |

### 三、用户可控制的配置项

都在 `MultiModalConfig`（`vllm/config/multimodal.py`），对应 CLI 参数：

| 配置 | 默认值 | 作用 |
|---|---|---|
| `--mm-processor-cache-gb` | `4` | 容量；**设为 0 完全禁用**（`self.cache` 为 None） |
| `--mm-processor-cache-type` | `"lru"` | 在 `lru` / `shm` 之间选（仅在 IPC 支持时生效） |
| `--mm-shm-cache-max-object-size-mb` | `128` | 仅 shm 模式，单对象上限 |

所以"怎么知道用了哪个实现"：**默认部署（单 API 进程、单 DP）+ 默认配置 → `MultiModalProcessorSenderCache`（lru）**。要改 shm 需显式 `--mm-processor-cache-type shm`；多 API 进程部署会被强制降级为 `processor_only`。

### 四、P1 侧的配套选择（镜像关系）

同一 `_get_cache_type` 还决定 P1 侧建不建、建什么 receiver cache，两边必须配套：

| cache_type | P0 sender（Renderer） | P1 receiver |
|---|---|---|
| `processor_only` | `MultiModalProcessorOnlyCache` | **无**（`core.py:177` 返回 None） |
| `lru` | `MultiModalProcessorSenderCache` | `MultiModalReceiverCache`（引擎进程，`v1/engine/core.py:177`） |
| `shm` | `ShmObjectStoreSenderCache` | `ShmObjectStoreReceiverCache`（worker 进程，`v1/worker/worker_base.py:311`） |

另外 Renderer 还会建第二个 processor（`_readonly_mm_processor`，`base.py:139-146`），它固定用 `processor_only_cache_from_config` → 永远是 `MultiModalProcessorOnlyCache`，专供 tokenize-only 请求，避免污染主 sender cache 的 LRU 顺序（因为主 cache 的逐出顺序必须与 P1 镜像）。

### 五、运行时如何确认

```python
type(mm_processor.cache)   # 或 type(renderer.mm_processor.cache)
```

或在 `BaseMultiModalProcessor.__init__` 里加一行日志打印 `type(cache).__name__`。也可以通过启动日志中的 engine args（`mm_processor_cache_type`、`mm_processor_cache_gb`）加上 API 进程数/DP 拓扑，按第三节的规则静态推断。

---

<a id="q9"></a>
## Q9. 什么是 IPC 缓存

> **原问题**：上面回答中提到的 IPC 缓存，是什么？

**IPC = Inter-Process Communication（进程间通信）**，"IPC 缓存"指的是 vLLM V1 多进程架构下，**让多模态处理结果跨越"前端进程 → 引擎核心进程"这条 IPC 通道时被缓存复用**的机制。要理解它，先看它要省掉的成本是什么。

### 一、背景：P0 和 P1 之间隔着一条 IPC 通道

vLLM V1 默认是多进程架构：

- **P0（前端进程）**：运行 API server / Renderer / `InputProcessor`，`_preprocess`、resize 等 HF processor 计算都发生在这里；
- **P1（引擎核心进程）**：运行 `EngineCore`（调度、模型执行），通过 **ZMQ 消息队列**接收 P0 发来的 `EngineCoreRequest`。

每个请求处理完后，`pixel_values` 等多模态张量要**序列化后经 ZMQ 从 P0 拷贝到 P1**。一张图的 pixel_values 动辄几 MB，高并发下这条通道的序列化+拷贝开销非常可观。

### 二、"IPC 缓存"省的到底是什么

mm processor cache 有两个层面的收益，三种实现的区别正在于覆盖到哪一层：

| 模式 | 省重算（P0 不再跑 HF processor） | 省传输（命中时不过 IPC 通道） |
|---|---|---|
| `processor_only`（**IPC caching disabled**） | ✅ | ❌ 命中后仍全量发 P1 |
| `lru` / `shm`（**IPC caching enabled**） | ✅ | ✅ 命中时只发 hash/地址 |

关键代码证据在 `MultiModalFeatureSpec.data` 的注释里（`vllm/multimodal/inputs.py:331-337`）：

```python
data: "MultiModalKwargsItem | None"
"""
Represents multimodal data for this feature.

Can be `None` if the item is cached, to skip IPC between API server
and engine core processes.
"""
```

也就是说：启用 IPC 缓存后，P1 侧也持有同样的缓存条目（`lru` 模式存在 engine core 进程的 `MultiModalReceiverCache`；`shm` 模式存在共享内存、worker 进程按地址读），P0 命中缓存时把 `data` 置为 `None` 发给 P1，**大张量不再过 ZMQ**，P1 按 `mm_hash` 从自己的缓存里取回——这就是 Q7 里 `MultiModalProcessorSenderCache.get_and_update_item` 命中时返回 `(None, prompt_updates)` 的含义。

### 三、为什么"IPC 缓存受支持"是有条件的

`registry.py:283-286` 的判断：

```python
is_ipc_supported = parallel_config._api_process_count == 1 and (
    parallel_config.data_parallel_size == 1
    or parallel_config.data_parallel_external_lb
)
```

原因回到 Q7 说的 **P0/P1 镜像不变式**：P0 靠查自己的缓存来推断 P1 有没有该条目，前提是两边 LRU 逐出顺序完全一致，而只有在 **1 个 P0 对 1 个 P1** 时这个顺序才能保持：

- **多 API 进程**（`_api_process_count > 1`）：多个 P0 各自有缓存，但共享/对应 P1 的语义被打破，P1 的逐出顺序无法同时镜像多个 P0 → 退化为 `processor_only`；
- **DP > 1 且无外部 LB**：同一图片的后续请求可能被内部负载均衡发到**另一个** DP engine，那个 engine 的 P1 缓存里并没有这份数据，P0 却以为有 → 正确性出问题；有外部 LB 时可以把同 hash 请求固定路由到同一 engine，才允许启用。

不满足条件时只能用 `processor_only`：P0 自存自发，命中后照常全量经 IPC 发送——省算不省传。

### 四、一句话总结

**IPC 缓存 = "P0/P1 双侧镜像缓存"**：P0 侧缓存负责"要不要重算"，P1 侧缓存负责"要不要重传"。`processor_only` 只有前者；`lru`（P1 进程内 LRU）和 `shm`（共享内存单写多读）是后者的两种存储实现，让重复图片的请求在进程边界上只流过一个小小的 hash/地址，而不是整张 pixel_values。

注：不要把它和 `lmcache_integration` 里的 `IPCCacheEngineKey`（KV cache 的 CUDA IPC 句柄传输）或 `mm_ipc_gpu_memory_gb`（前端 GPU 解码用的显存池）混淆——那几处"IPC"是另外的机制，这里讨论的是 `multimodal/cache.py` 这条 mm processor 缓存线。

---

<a id="q10"></a>
## Q10. Qwen3-VL 的 vision ViT 在 P0 还是 P1 进程执行

> **原问题**：请问对 qwen3 vl 的 vision vit 的处理是在 vllm 中哪个进程调用的？是 P0 前端进程还是 P1 核心进程呢？

**ViT（视觉编码器）跑在 P1 一侧，不在 P0 前端进程。** 两个进程的分工非常清晰：

### 一、分工边界：P0 出 pixel_values，P1 跑 ViT

| | P0 前端进程 | P1 引擎核心侧 |
|---|---|---|
| 干什么 | HF processor CPU 预处理：`_preprocess`、resize、normalize、patch reshape | **ViT 前向（GPU）**：`Qwen3VisionTransformer` |
| 产出 | `pixel_values` / `image_grid_thw`（CPU 张量，经 IPC/共享内存发给 P1） | 图像 embeddings（GPU 张量，写入 encoder_cache） |

### 二、ViT 的实际调用链（全部在 worker 进程内）

```
EngineCore 调度器决定本步要编码哪些 mm item
  → SchedulerOutput.scheduled_encoder_inputs 发给 worker
    └─ GPUModelRunner.execute_model()
         └─ _execute_mm_encoder(scheduler_output)        vllm/v1/worker/gpu_model_runner.py:3005
              ├─ _batch_mm_inputs_from_scheduler()        (把各请求的 pixel_values 组 batch、上 GPU)
              └─ model.get_multimodal_embeddings(...)
                   └─ Qwen3VLForConditionalGeneration._process_image_input()
                        vllm/model_executor/models/qwen3_vl.py:2210
                        └─ self.visual(pixel_values, grid_thw)     qwen3_vl.py:2225
                             └─ Qwen3VisionTransformer.forward   ← ViT 在这里真正执行
```

编码结果经 `_cache_encoder_output()` 写入 **GPU encoder_cache**（`gpu_model_runner.py:3030` 一带），随后 `_gather_mm_embeddings()`（`gpu_model_runner.py:3227`）把图像 embedding 按 `is_mm_embed` 拼进 LLM 的输入序列。如果命中 encoder cache / EC connector（`ECCPUConnector`：worker 侧 `save_caches` 把 encoder 输出卸载到 CPU mmap，调度器侧 `has_cache_item` 命中后本步干脆不调度 encoder 输入），**ViT 前向会被整个跳过**。

### 三、"P1 侧"的精确含义：worker 进程

严格说 ViT 跑在 **worker 进程**里，它和 EngineCore 的关系取决于并行配置：

- **TP=1（UniProcExecutor）**：worker 和 EngineCore 在**同一个进程**（即 P1 引擎核心进程本身）；
- **TP>1（MultiprocExecutor）**：EngineCore 派生多个**独立 worker 子进程**，ViT 在每个 worker 进程里各跑一份（按 TP 切分视觉权重）。

无论哪种，都属于"引擎核心侧"——和 P0 前端进程（API server/Renderer）是明确分开的。

### 四、整条流水线的数据流向

```
P0: 原图 → _preprocess/resize (CPU) → pixel_values
     │  (IPC: ZMQ 或 shm, 命中 mm processor cache 时只发 hash/地址)
     ▼
P1 worker: pixel_values → Qwen3VisionTransformer (GPU) → 图像 embeddings → encoder_cache
     │  (ECCPUConnector: embeddings 可卸载到 CPU mmap, 下次命中直接跳过 ViT)
     ▼
P1 worker: embeddings 拼入输入序列 → LLM 前向
```

所以 Q3 里讨论的 resize 缓存（P0 侧优化）和 `ECCPUConnector`（P1 侧优化）分别作用于这条链的两端：前者省的是 **CPU 预处理**，后者省的是 **GPU ViT 前向**。

---

<a id="q11"></a>
## Q11. `ShmObjectStoreSenderCache` 与 `ShmObjectStoreReceiverCache` 在 P0/P1 侧的实现

> **原问题**：ShmObjectStoreSenderCache 和 ShmObjectStoreReceiverCache 在 P0 侧和 P1 侧分别是怎样实现的？

这两个类是**同一个共享内存对象库的"写端"和"读端"**——它们共享底层设施 `SingleWriterShmObjectStorage`（架在 `SingleWriterShmRingBuffer` 上，`vllm/distributed/device_communicators/shm_object_storage.py`），只是以不同角色 attach。

### 一、共享的底层：`SingleWriterShmRingBuffer` + `SingleWriterShmObjectStorage`

**环形缓冲（`shm_object_storage.py:22`）**：

- 一块命名共享内存（名字取 `VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME`，大小 = `mm_processor_cache_gb`），**单写者（P0）创建，多读者（各 worker）attach**；
- 每个 chunk 布局：`[4字节 monotonic_id][4字节 size][数据]`，写者进程内维护 `metadata: monotonic_id → 起始地址`；
- 线性分配 + 到末尾回绕（wraparound）；`monotonic_id` 用于**数据完整性校验**——读者按地址读时核对 id，防止缓冲回绕后旧地址被新数据覆盖导致错读；
- 空间回收是 **FIFO**：写者调 `free_buf` 从 oldest chunk 开始，用 `is_free_fn` 判断能否回收。

**对象库（`shm_object_storage.py:414`）**：

- 每个对象的布局：`[4字节引用计数 flag][pickle 的元数据][序列化数据]`；
- **写者侧**才维护三个索引：`key_index: key → (address, monotonic_id)`、`id_index: monotonic_id → key`、`writer_flag: monotonic_id → 引用次数`；
- 序列化用 `MsgpackSerde`（`shm_object_storage.py:346`）：torch.Tensor / `MultiModalKwargsItem` 走 vllm 的 `MsgpackEncoder` 编成**若干字节段**（记录每段长度 `len_arr`），其他对象走 pickle；
- **回收基于引用计数**（`default_is_free_check`，:702）：`reader_count >= writer_flag × n_readers` 才回收。语义——写者每引用一次对象（put/get_cached/touch 都会 `writer_flag++`），意味着"所有 n_readers 个读者都应消费一次"；读者每次 `get`/`touch` 给 chunk 头的 `reader_count +1`。实际消费数达到期望数，对象才可被 FIFO 回收。即**写者 touch 是"续命"（提高回收门槛），读者 touch/get 是"销账"（推进回收）**。

### 二、P0 侧：`ShmObjectStoreSenderCache`（`cache.py:437`）

**初始化**（:449-471）：`create=True` 创建环形缓冲（写者角色），`n_readers = world_size`（TP worker 数），`MsgpackSerde`；另有一个**只在 P0 存在的 dict `_p0_cache: mm_hash → prompt_updates`**——prompt_updates 不需要给 worker，留在本地最省。

**各方法实现**：

| 方法 | 实现 |
|---|---|
| `is_cached_item(mm_hash)` | `mm_hash in _shm_cache.key_index`（纯查 P0 本地索引） |
| `get_and_update_item` 命中 | `get_cached()` 取 `(address, monotonic_id)` 并 `writer_flag++`（续命）→ 返回 **`address_as_item()`**：只含 `{"address", "monotonic_id"}` 两个字段的 `MultiModalKwargsItem`（几十字节）+ 本地 `_p0_cache` 里的 prompt_updates |
| `get_and_update_item` 未命中 | `put(mm_hash, item)`：msgpack 序列化 → `allocate_buf`（空间不足先 `free_unused()` FIFO 回收最多 `2×max_object_size` 再重试）→ 写入 → 更新索引 → 同样返回地址条目 |
| 异常降级 | 对象超过 `--mm-shm-cache-max-object-size-mb`（ValueError）或缓冲满且被引用计数保护无法回收（MemoryError）→ **不缓存，原样返回完整 item**（:514-535） |
| `touch_sender_cache_item` | `_shm_cache.touch()` → 写者侧 `writer_flag++`，防回收 |
| 辅助 | `remove_dangling_items()`：`_p0_cache` 超过 `2×len(key_index)` 时清理已被 shm 回收的悬空 prompt_updates |

关键点：**P0 从来不把张量发出去**，发给 P1 的只是地址条目；张量本体在共享内存里只写一次。

### 三、P1 侧：`ShmObjectStoreReceiverCache`（`cache.py:678`）

**部署位置**：**worker 进程**（不是 engine core 主进程！`v1/worker/worker_base.py:311` 创建，TP>1 时每个 worker 各 attach 一份），传入跨 worker 共享的 `shared_worker_lock`。

**初始化**（:688-709）：`create=False` attach 已有环形缓冲（读者角色），`reader_lock` 必传（读者计数必须加锁，否则多 worker 并发改 chunk 头会竞态）。

**各方法实现**：

| 方法 | 实现 |
|---|---|
| `get_and_update_item(item, mm_hash)` | item 含 `"address"` → `_shm_cache.get(address, monotonic_id)`：`access_buf` 定位 → **核对 chunk 头 monotonic_id 是否匹配**（不匹配说明地址已被复用，报错）→ `MsgpackSerde.deserialize` 还原成 `MultiModalKwargsItem`（`share_mem=False`，数据**拷贝出**共享内存）→ 加 reader_lock 给 `reader_count+1`（销账）；item 不含地址（首次没走缓存）→ 原样透传 |
| `touch_receiver_cache_item` | 读者侧 touch：加锁把 `reader_count+1`。细节保护（:667-672）：`reader_count >= n_readers` 才加——避免新条目刚写入、各读者首次读取还没完成时 pre-touch 误销账 |
| `get_and_update_features`（父类 :589） | **先对所有 feature 统一 touch，再逐个 get**——防止批量更新过程中，正在处理的条目被 FIFO 回收 |
| `clear_cache` | 调 `_shm_cache.clear()`，但 `clear()` 内部断言写者身份，读者侧实际是 no-op |

### 四、两侧协作的完整数据流

```
P0 (前端进程)                          P1 (worker 进程)
─────────────                          ─────────────
put(mm_hash, 张量)  ──写入──▶  /dev/shm 环形缓冲 (张量只存一份)
返回 地址条目 {address, id}
     │  经 ZMQ 发给 EngineCore → 调度 → 发给 worker（全程只是几十字节）
     │                                     │
     │                                     ▼
     │                              get(address, id): 核对 id → 反序列化
     │                                → reader_count+1 (销账)
writer_flag++ (续命)                     │
     │                                     ▼
     ◀── reader_count ≥ writer_flag×n_readers 时, 下次 put 空间不足可 FIFO 回收
```

### 五、设计要点总结

1. **单写多读**：P0 是唯一写者，避免写竞争；多 worker 读者靠 chunk 头引用计数 + 共享锁同步；
2. **张量零冗余**：与 `lru` 模式（P0/P1 各存一份、首次要过 IPC）不同，shm 模式全系统只有一份拷贝；
3. **FIFO + 引用计数模拟缓存语义**：环形缓冲本身 FIFO 回收，"LRU 续命"效果通过写者/读者两侧的 touch 计数差实现；
4. **优雅降级**：对象过大、缓冲满都不报错，只是不缓存（退化为全量传输）；
5. **prompt_updates 不走 shm**：只留在 P0 的 `_p0_cache`，因为它是 P0 展开占位符用的，worker 不需要。

---

<a id="q12"></a>
## Q12. `ShmObjectStoreSenderCache.get_and_update_item` 命中/未命中时的 item 与 put 细节

> **原问题**：ShmObjectStoreSenderCache 中，get_and_update_item 方法命中时候 item 是什么？当 get_and_update_item 方法未命中时候的 put 方法是在哪里的？存入的 item 又是什么？

这里"item"在这条路径上有三种形态，先看调用方 `_merge_mm_kwargs`（`processor.py:1371-1383`）传入的是什么：

```python
for item_idx, item_hash in enumerate(hashes):
    if not mm_is_cached[modality][item_idx]:
        item = (missing_kwargs_item, missing_updates_item)   # 未命中: 真实数据二元组
    else:
        item = None                                           # 命中: None
    kwargs, updates = cache.get_and_update_item(item, item_hash)
```

### 一、命中时：item 是什么

**传入的 `mm_item` 是 `None`**（命中意味着 P0 没跑 HF processor，手里本来就没有数据）。

**返回的 item 是"地址条目"**（`cache.py:493-499`）：

```python
if self._shm_cache.is_cached(mm_hash):
    address, monotonic_id = self._shm_cache.get_cached(mm_hash)   # 查 key_index, writer_flag++
    prompt_updates = self._p0_cache[mm_hash]                      # prompt_updates 从 P0 本地 dict 取
    return self.address_as_item(address, monotonic_id), prompt_updates
```

`address_as_item`（`cache.py:567-581`）构造的是一个**假的 `MultiModalKwargsItem`**：

```python
MultiModalKwargsItem({
    "address":      MultiModalFieldElem(data=address,      field=MultiModalBatchedField()),
    "monotonic_id": MultiModalFieldElem(data=monotonic_id, field=MultiModalBatchedField()),
})
```

——只有两个整数、几十字节，**不含任何张量**。它就是随后经 ZMQ 发给 P1 的东西；worker 进程的 `ShmObjectStoreReceiverCache` 看到 `"address" in item` 就知道该去共享内存读真数据。

### 二、未命中时：put 在哪、存的是什么

**put 不在 sender cache 里**——sender 只是委托（`cache.py:507`）：

```python
address, monotonic_id = self._shm_cache.put(mm_hash, item)
```

真正实现是 **`SingleWriterShmObjectStorage.put`（`shm_object_storage.py:563-611`）**，流程：序列化 → 分配 chunk（空间不足先 `free_unused()` FIFO 回收再重试）→ 写入共享内存 → 更新索引：

```python
object_data, data_bytes, object_metadata, md_bytes = self.ser_de.serialize(value)  # MsgpackSerde
address, monotonic_id = self.ring_buffer.allocate_buf(buffer_size)
... 写入 [4字节引用计数][元数据][数据] ...
self.key_index[key] = (address, monotonic_id)   # mm_hash → 地址
```

**存入的 `item`** 就是调用方传来的二元组的第一部分 `missing_kwargs_item`——一个**真的 `MultiModalKwargsItem`**，即这张缺失图片刚被 HF processor 处理完的输出。对 Qwen2/3-VL 的图片，里面大致是：

```python
MultiModalKwargsItem({
    "pixel_values":   MultiModalFieldElem(data=torch.Tensor(...)),   # (num_patches, C·t·p·p)
    "image_grid_thw": MultiModalFieldElem(data=torch.Tensor([t,h,w])),
})
```

它被 `MsgpackSerde`（vllm 的 `MsgpackEncoder`）序列化成若干字节段写进共享内存——**所以 shm 里存的是"单张图 HF 预处理结果（含 pixel_values 张量本体）的序列化字节"**，`key_index` 里只存 `mm_hash → (address, monotonic_id)` 的映射。

注意二元组的第二部分 `missing_updates_item`（prompt_updates）**不进 shm**，单独存到 P0 本地 `_p0_cache[mm_hash]`（`cache.py:512`）。

### 三、一张表收尾

| | 命中 | 未命中 |
|---|---|---|
| 传入 `mm_item` | `None` | `(真 MultiModalKwargsItem, prompt_updates)` |
| 动作 | 查 `key_index` 拿地址，`writer_flag++` | `put` 把张量序列化写进 shm，记索引 |
| **返回的 item** | 地址条目（两个 int） | **同样是地址条目**（`cache.py:513`） |
| prompt_updates 来源 | `_p0_cache` 读 | 直接返回，同时写入 `_p0_cache` |

所以无论命中与否，`get_and_update_item` 返回给下游的都是**地址条目**——P0 永远不向 P1 直接发张量。唯一的例外是 `put` 失败（对象超过 `--mm-shm-cache-max-object-size-mb`，或缓冲满且被引用计数卡住）：此时原样返回传入的真 item（含张量），降级为全量 IPC 传输（`cache.py:514-535`）。

---

<a id="q13"></a>
## Q13. `_cached_apply_hf_processor` 返回对象的含义与返回值示例

> **原问题**：multimodal.processing.processor.BaseMultiModalProcessor._cached_apply_hf_processor 方法返回的对象是什么含义。请给它的返回值举个例子。

### 返回值签名

`vllm/multimodal/processing/processor.py:1441`：

```python
def _cached_apply_hf_processor(
    self,
    inputs: ProcessorInputs,
    timing_ctx: TimingContext,
) -> tuple[list[int], MultiModalProcessingInfo, bool]:
```

返回一个三元组：**`(prompt_ids, mm_info, is_update_applied)`**。

### 三个元素的含义

**① `prompt_ids: list[int]` — 分词后的 prompt（占位符尚未展开）**

对 prompt 文本做 tokenize 得到的 token id 序列。注意在缓存路径下，`_apply_hf_processor_main` 是以 `enable_hf_prompt_update=False` 调用的（`processor.py:1479`），所以 `<|image_pad|>` 这类占位符**只出现 1 次**，还没有被展开成实际数量——展开动作推迟到后面 `apply()` 里的 `_apply_prompt_updates`（`processor.py:1528`）统一做。原因写在注释里（`processor.py:1466`）：缓存路径下 HF processor 只处理了缺失的 item，prompt 和 `mm_missing_data_items` 不对应，不能在此刻做文本替换。

**② `mm_info: MultiModalProcessingInfo` — 多模态处理结果（核心）**

一个 NamedTuple（`processor.py:966`）：

```python
class MultiModalProcessingInfo(NamedTuple):
    kwargs: MultiModalKwargsOptionalItems   # HF processor 输出的张量，按模态/按 item 组织
    hashes: MultiModalHashes                # 每个 item 的内容哈希，用于缓存
    prompt_updates: MultiModalPromptUpdates # 每个 item 对应的占位符替换规则
```

- **`kwargs`**：`{模态: [MultiModalKwargsItem, ...]}`，每个 item 里是 HF processor 算出的张量字段（如 `pixel_values`、`image_grid_thw`）。缓存路径下它是 `_merge_mm_kwargs` 把"缓存命中的旧 item"和"新算的缺失 item"按原始顺序合并后的结果。
- **`hashes`**：`{模态: [hash_str, ...]}`，每个 item 的内容哈希（或用户传入的 uuid），是 processor cache 的 key。
- **`prompt_updates`**：`{模态: [[ResolvedPromptUpdate, ...], ...]}`，每个 item 一组替换规则：找到 `target`（如 `<|image_pad|>`），用 `content`（N 个占位 token id）替换/插入。

**③ `is_update_applied: bool` — HF processor 是否已经展开过文本**

表示 HF processor 在返回结果前**是否已经把占位符替换进 prompt 文本**。缓存路径下恒为 `False`；无缓存时走 `_apply_hf_processor`（`enable_hf_prompt_update=True`），若模型覆写了 `_call_hf_processor` 让 HF 端展开文本，则可能为 `True`。

### 具体例子（Qwen3-VL）

假设请求为：prompt = `"<|im_start|>user\n描述这张图: <|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"`，附带 1 张图片，resize 后 448×448（patch=14，merge=2 → grid_thw=[1,32,32]，合并后 **256 个视觉 token**，`<|image_pad|>` 的 token id 是 151655）。返回值大致是：

```python
(
    # ① prompt_ids：占位符 <|image_pad|> 只出现 1 次，未展开
    [151644, 872, 198, ..., 151652, 151655, 151653, 151645, 198],
    #   user  “描述这张图:”  <v_start> <img_pad> <v_end> <im_end>

    # ② mm_info
    MultiModalProcessingInfo(
        kwargs=MultiModalKwargsItems({
            "image": [
                MultiModalKwargsItem({
                    # 1024 个 patch（1×32×32），每 patch 3×2×14×14=1176 维
                    "pixel_values":    Tensor(shape=(1024, 1176), dtype=bfloat16),
                    "image_grid_thw":  Tensor([[1, 32, 32]]),
                })
            ]
        }),
        hashes={
            "image": ["a3f5c8…64位hex"]   # 该图片的内容哈希，缓存 key
        },
        prompt_updates={
            "image": [
                [   # 第 0 张图对应的一组替换规则
                    ResolvedPromptUpdate(
                        modality="image",
                        item_idx=0,
                        mode=UpdateMode.REPLACE,
                        target="<|image_pad|>",               # 要查找的占位文本
                        content=PromptUpdateDetails.full(
                            [151655] * 256                    # 展开成 256 个占位 token
                        ),
                    )
                ]
            ]
        },
    ),

    # ③ is_update_applied
    False,   # 缓存路径下恒为 False，展开由后续 _apply_prompt_updates 完成
)
```

下游消费者：`apply()`（`processor.py:1685` 附近）拿到这个三元组后，用 `prompt_updates` 把 `prompt_ids` 中那个单独的 `151655` 替换成 256 个，同时生成 `PlaceholderRange(offset, length=256)` 告诉 engine core 这段区间要用 `kwargs` 里的视觉 embedding 来填充；`hashes` 则用于 processor cache 和后续 encoder cache 的命中判断。

**一句话总结**：它返回的是"一次多模态请求经过 HF processor 后的完整中间产物"——未展开占位符的 token 序列 + 按 item 组织好的视觉张量/哈希/占位符展开规则，缓存路径下则是缓存命中项与新计算项合并后的结果。

---

<a id="q14"></a>
## Q14. `MultiModalProcessingInfo` 对象中的 `hashes` 字段是怎样得到的

> **原问题**：MultiModalProcessingInfo对象中的hashes字段是怎样得到的？

`hashes` 是由 `ProcessorInputs.get_mm_hashes()` 现算出来的。链路如下：

### 调用点

在 `_cached_apply_hf_processor`（`processor.py:1456`）和 `_apply_hf_processor`（`processor.py:1424`）里：

```python
with timing_ctx.record("get_mm_hashes"):
    mm_hashes = inputs.get_mm_hashes(self.info.model_id)
```

### 计算逻辑（`vllm/multimodal/processing/inputs.py:25`）

对 `mm_data_items` 里的**每个模态、每个 item**，分两种情况：

**情况 A：用户没有提供 UUID（最常见）**

直接对每个 item 做内容哈希：

```python
mm_hashes[modality] = [
    hasher.hash_kwargs(
        model_id=model_id,          # 模型 ID，混入哈希避免跨模型串缓存
        **{modality: item},         # item 本体（PIL.Image / ndarray / tensor / bytes…）
        **hf_processor_mm_kwargs,   # 处理参数（如 min_pixels/max_pixels），
                                    # 参数不同 → 处理结果不同 → 哈希必须不同
    )
    for item in data_items
]
```

**情况 B：用户通过 API 提供了 UUID（`multi_modal_uuids`，经 `parse_mm_uuids` 解析成 `mm_uuid_items`）**

逐 item 判断（`inputs.py:46-58`）：

- **`uuid_item` 不为 None 且没有 `hf_processor_mm_kwargs`** → 直接把用户给的 UUID 字符串当作 hash，**跳过昂贵的内容哈希**；
- **`uuid_item` 为 None，或存在 `hf_processor_mm_kwargs`** → 仍然计算哈希，但参与哈希的"内容"优先用 UUID 字符串而不是原始数据（性能优化，注释见 `inputs.py:47-49`）。

### 哈希本身怎么算（`vllm/multimodal/hasher.py`）

`MultiModalHasher.hash_kwargs`（`hasher.py:154`）：

1. 按 key 排序后，对每个 value 递归调用 `iter_item_to_bytes` → `serialize_item` 序列化成字节流：
   - `PIL.Image` → `mode` + 像素 ndarray（若 EXIF 里有 `ImageID` UUID，直接用它的字节，免读像素）；
   - `torch.Tensor` → 转 CPU numpy 字节（bfloat16 特判：view 成 uint8）；
   - `np.ndarray` → `dtype.str + shape + 数据字节`；
   - `bytes/str/int/float` → 直接编码；
   - 其他 → 兜底 `pickle.dumps`；
2. 逐块喂给哈希器，算法由环境变量 `VLLM_MM_HASHER_ALGORITHM` 决定：**默认 blake3**，可选 sha256/sha512（FIPS 合规场景）；
3. 返回 `hasher.hexdigest()` 十六进制字符串。

### 例子（Qwen3-VL，1 张图，无 UUID）

```python
mm_hashes = {
    "image": [
        # blake3(model_id="Qwen/Qwen3-VL-8B-Instruct",
        #        image=<PIL.Image 模式+像素>,
        #        /* hf_processor_mm_kwargs 为空则不参与 */)
        "b7e2c94a1d..."
    ]
}
```

若同一请求还带 1 个视频且用户传了 UUID：

```python
mm_hashes = {
    "image": ["b7e2c94a1d..."],          # 内容哈希
    "video": ["user-uuid-video-0001"],   # 用户提供的 UUID 原样使用
}
```

### 这个 hash 的用途

它贯穿整条缓存链：

1. **processor cache**：`_get_cache_missing_items`（`processor.py:1299`）用它判断哪些 item 已处理过，命中就跳过 HF processor；
2. **下游 encoder cache / prefix caching**：hash 会随 `mm_info` 一路传到 engine core，作为该多模态数据的身份标识。

设计上两个关键混入因子值得注意：`model_id` 保证不同模型的处理结果不共享缓存项；`hf_processor_mm_kwargs` 保证"同一张图、不同 resize 参数"被视为不同 item——这正是情况 B 中"有 kwargs 时即使给了 UUID 也要重算哈希"的原因。
