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
