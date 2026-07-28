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
