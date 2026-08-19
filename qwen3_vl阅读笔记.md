# Qwen3-VL 阅读笔记（基于 transformers 源码）

> 源码仓库：`C:\Code\transformers`（HuggingFace transformers，main 分支）
> 关键结论先行：**Qwen3VLProcessor 自己没有图像预处理实现**，auto-mapping 把它指向 Qwen2-VL 的图像处理器：
>
> ```python
> # src/transformers/models/auto/image_processing_auto.py:138
> ("qwen3_vl", {"torchvision": "Qwen2VLImageProcessor", "pil": "Qwen2VLImageProcessorPil"})
> ```
>
> 完整链路：`Qwen3VLProcessor`（`models/qwen3_vl/processing_qwen3_vl.py`）→ `Qwen2VLImageProcessor`（`models/qwen2_vl/image_processing_qwen2_vl.py`）。

## 目录

- [Q1：用 vLLM 部署 Qwen3-VL 时，能从 transformers 仓库看到所有推理代码吗？](#q1用-vllm-部署-qwen3-vl-时能从-transformers-仓库看到所有推理代码吗)
- [Q2：smart_resize → resize → rescale/normalize → patches 重排的完整分析](#q2smart_resize--resize--rescalenormalize--patches-重排的完整分析)
  - [1. smart_resize — 动态分辨率计算](#1-smart_resize--动态分辨率计算)
  - [2. _preprocess — 完整流水线](#2-_preprocess--完整流水线)
  - [3. Processor 侧：占位 token 展开](#3-processor-侧占位-token-展开)
  - [4. 端到端例子](#4-端到端例子)
- [Q3：TorchvisionBackend.rescale_and_normalize 在做什么？](#q3torchvisionbackendrescale_and_normalize-在做什么)
- [Q4：reshape 拆 H 为 (grid_h//2, 2, 14) 与 permute 怎么理解？](#q4reshape-拆-h-为-grid_h2-2-14-与-permute-怎么理解)
- [Q5：unsqueeze/expand/reshape 复制"帧"生成 1176 维 patch 的细节](#q5unsqueezeexpandreshape-复制帧生成-1176-维-patch-的细节)
- [Q6：pixel_values 变长拼接与 image_grid_thw 的作用](#q6pixel_values-变长拼接与-image_grid_thw-的作用)
- [Q7：从 _preprocess 到占位 token 展开，Processor 还做了哪些事？](#q7从-_preprocess-到占位-token-展开processor-还做了哪些事)
- [Q8：从 pixel_values 被 ViT 消费到 image_grid_thw 驱动 MRoPE，模型做了哪些事？](#q8从-pixel_values-被-vit-消费到-image_grid_thw-驱动-mrope模型做了哪些事)
- [Q9：从 preprocess 产生的 pixel_values 到被 ViT 消费，中间的链路？](#q9从-preprocess-产生的-pixel_values-到被-vit-消费中间的链路)
- [Q10：带图片请求的推理整体流程、参与对象与关键方法（总览）](#q10带图片请求的推理整体流程参与对象与关键方法总览)
- [Q11：Qwen3VLModel.get_image_features 方法详解](#q11qwen3vlmodelget_image_features-方法详解)
- [Q12：Qwen3VLVisionModel / Qwen3VLTextModel / Qwen3VLModel 的区别](#q12qwen3vlvisionmodel--qwen3vltextmodel--qwen3vlmodel-的区别)
- [Q13：vLLM 端 Qwen3_VisionTransformer 对象的各个字段作用是什么？](#q13vllm-端-qwen3_visiontransformer-对象的各个字段作用是什么)
- [Q14：Qwen3_VisionTransformer.forward 方法一步步做了什么（含原理）](#q14qwen3_visiontransformerforward-方法一步步做了什么含原理)
- [Q15：Qwen3_VisionTransformer.prepare_encoder_metadata 具体做了哪些事情？](#q15qwen3_visiontransformerprepare_encoder_metadata-具体做了哪些事情)
- [Q16：2D RoPE 的原理](#q162d-rope-的原理)

---

## Q1：用 vLLM 部署 Qwen3-VL 时，能从 transformers 仓库看到所有推理代码吗？

**部分正确，但有关键误解。**

transformers 仓库里有 Qwen3-VL 的完整参考实现（`src/transformers/models/qwen3_vl/` 及 `qwen3_vl_moe/`），是理解模型架构和数学逻辑最权威的资料。**但 vLLM 部署时并不执行这里的 `modeling_*.py`**：vLLM 仓库（`vllm/model_executor/models/qwen3_vl.py`）有一份独立重写的实现，因为要接入 PagedAttention、continuous batching、KV cache 分页、CUDA graph、张量并行等 serving 级机制。

vLLM 从 transformers **实际复用的**只有：

- `configuration` / `config.json` 的解析
- `AutoProcessor` / tokenizer（多模态输入预处理）

| 想了解的东西 | 去哪看 |
|---|---|
| 模型架构、数学逻辑 | transformers `qwen3_vl/modeling_qwen3_vl.py` |
| 图像/视频预处理成 tensor | transformers processing 相关文件 |
| vLLM 实际推理代码（attention、KV cache、并行、权重加载） | vLLM 仓库 `vllm/model_executor/models/qwen3_vl.py` |
| 采样、调度、batching、serving | vLLM 核心引擎 |

实践中读 vLLM 的模型文件时，看不懂的结构逻辑应对照 transformers 的参考实现——两者张量形状和计算流程一一对应。

---

## Q2：smart_resize → resize → rescale/normalize → patches 重排的完整分析

### 1. smart_resize — 动态分辨率计算

`models/qwen2_vl/image_processing_qwen2_vl.py:62-88`：

```python
def smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(...)                     # 极端长条图直接拒绝
    h_bar = round(height / factor) * factor       # ① 先对齐到 factor 的倍数
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:                # ② 太大 → 等比缩小
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:              # ③ 太小 → 等比放大
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar
```

**设计意图**：不做固定分辨率，而是**保持宽高比的动态分辨率**，只施加三个约束：

| 约束 | 原因 |
|---|---|
| h、w 必须是 `factor = patch_size × merge_size = 14×2 = 28` 的倍数 | ViT 按 14×14 切 patch，之后 2×2 patch merge 进 LLM；只对齐 14 会在 merge 边界出现半个 patch |
| 总像素 ≤ `max_pixels`（默认 28·28·1280 ≈ 100 万） | 限制单图 token 数上限，控制显存/时延 |
| 总像素 ≥ `min_pixels`（默认 56×56，即 4×4 个 patch） | 太小的图放大后才有足够视觉信息 |

缩放系数 `beta = sqrt(面积比)`：保持宽高比时，面积比的平方根才是边长缩放比；缩小用 `floor`（不超上限），放大用 `ceil`（不低于下限），最后乘回 factor 对齐。缩小分支套 `max(factor, ...)` 防止极端宽高比下某边被压到 0。

> 视频版（`qwen3_vl/video_processing_qwen3_vl.py:35`）多了 `temporal_factor` 对齐帧数、`factor` 默认 32，逻辑同构。

### 2. _preprocess — 完整流水线

`image_processing_qwen2_vl.py:148-230`：

**① 按 shape 分组**（:166）：同尺寸图堆叠成一个 tensor 一起处理（向量化），`reorder_images` 还原顺序。这解释了为什么 pixel_values 是变长的——每张图分辨率不同，无法堆成规则的 `(B, C, H, W)` batch。

**② smart_resize → resize**（:171-182）：双三次插值（BICUBIC）缩放到 `(h_bar, w_bar)`。

**③ rescale / normalize**（:191-193）：先 `[0,255]→[0,1]`，再用 CLIP mean/std 标准化（详见 Q3）。

**④ 重排成 patches**（:194-208，详见 Q4）：

```python
patches = patches.reshape(B, C, grid_h//2, 2, 14, grid_w//2, 2, 14)
patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
# → (B, grid_h//2, grid_w//2, 2, 2, C, 14, 14)
```

**⑤ 时间维复制 + 展平**（:210-218，详见 Q5）：

```python
flatten_patches = (patches.unsqueeze(6)
    .expand(-1,-1,-1,-1,-1,-1, temporal_patch_size, -1, -1)
    .reshape(B, grid_h*grid_w, C * 2 * 14 * 14))     # (B, n_patches, 1176)
```

**⑥ 汇总输出**（:223-230，详见 Q6）：

```python
pixel_values = torch.cat(processed_images, dim=0)           # (Σn_patches, 1176)
image_grid_thw = torch.tensor(processed_grids_ordered)      # (n, 3)，每行 [1, grid_h, grid_w]
```

### 3. Processor 侧：占位 token 展开

预处理后，`Qwen3VLProcessor` 把 prompt 里的 `<|image_pad|>` 展开成正确数量（`processing_qwen3_vl.py:76-79`）：

```python
merge_length = self.image_processor.merge_size**2          # 4
num_image_tokens = image_grid_thw[i].prod() // merge_length
```

**LLM 侧每张图占 `t·h·w / 4` 个 token**——ViT 输出 `t·h·w` 个 patch 特征，经 2×2 merge 后送给语言模型。这就是 smart_resize 的 factor 是 28 而非 14 的原因：保证 `grid_h`、`grid_w` 为偶数，merge 整除。

`get_number_of_image_patches()`（`image_processing_qwen2_vl.py:232`，注释明确写着 **"used by vLLM"**）复现同一套 smart_resize 计算，让 vLLM 在没有真实图片时也能算出占位 token 数、规划 KV cache。

### 4. 端到端例子

输入一张 1000×800 的图：

1. `smart_resize(1000, 800, factor=28)`：面积 80 万 < max_pixels ≈ 100 万，只需对齐 → `h_bar = 1008`，`w_bar = 784`
2. resize 到 1008×784，normalize
3. `grid_h=72, grid_w=56` → reshape/permute/展平
4. 输出：`pixel_values (72×56=4032, 1176)`，`image_grid_thw = [[1, 72, 56]]`
5. LLM 侧占位 token 数 = 4032 / 4 = **1008 个 `<|image_pad|>`**

---

## Q3：TorchvisionBackend.rescale_and_normalize 在做什么？

`src/transformers/image_processing_backends.py:314-337`。作用是把图像预处理中两步逐像素操作——**rescale（值域缩放）和 normalize（标准化）——融合成一次计算**。

两步分别是：

```python
def rescale(self, image, scale):      return image * scale          # :280-285，[0,255]×(1/255)→[0,1]
def normalize(self, image, mean, std): return tvF.normalize(...)    # :287-295，(x-μ)/σ
```

Qwen2/3-VL 用 `OPENAI_CLIP_MEAN/STD`（约 `[0.481, 0.458, 0.408]` / `[0.269, 0.261, 0.276]`），匹配视觉塔预训练输入分布。

**融合原理**：两步都是仿射变换，可合并：

$$
\frac{x \cdot s - \mu}{\sigma} = \frac{x - \mu/s}{\sigma/s}
$$

即不缩放图像，而是把 mean/std 预除以 rescale_factor，只做一次 normalize（`_fuse_mean_std_and_rescale_factor`，:297-312）：

```python
if do_rescale and do_normalize:
    image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)  # μ × 255
    image_std  = torch.tensor(image_std,  device=device) * (1.0 / rescale_factor)  # σ × 255
    do_rescale = False   # rescale 被吸收掉了
```

省掉对整个 `(B, C, H, W)` tensor 的一次完整乘法和中间显存分配。两个细节：

- **`@lru_cache(maxsize=10)`**（:297）：融合后 mean/std 只依赖参数不依赖图像，缓存避免重复建 tensor；
- **dtype 提升**（:333）：`images.to(torch.float32)` 在 normalize 前做，防止 uint8 直接 `(x-μ)/σ` 丢精度/溢出。

分支逻辑：都开 → 融合 normalize；只 rescale → 单独 `image * factor`；都关 → 原样返回。Qwen2/3-VL 默认走融合路径。

---

## Q4：reshape 拆 H 为 (grid_h//2, 2, 14) 与 permute 怎么理解？

关键：**reshape 不移动任何数据，只是给同一段内存换一种"下标解读方式"**。

### 1. reshape 为什么能"拆"出三个因子

`(C, H, W)` tensor 行优先连续存储，第 `h` 行起始地址 = `h × W`。设 `H = 56`、`patch = 14`、`merge = 2`，则 `grid_h = 4`，且：

$$
h \in [0, 56), \quad 56 = \underbrace{2}_{grid_h//2} \times \underbrace{2}_{merge} \times \underbrace{14}_{patch}
$$

任意行号唯一分解：`h = i×28 + j×14 + k`（`i∈[0,2), j∈{0,1}, k∈[0,14)`）。如 `h=30 = 1×28 + 0×14 + 2` → `i=1, j=0, k=2`。

`reshape(B, C, grid_h//2, 2, 14, grid_w//2, 2, 14)` 就是把一维下标 `h` 按此除法拆成三维 `(i, j, k)`，数据不动：

| 因子 | 含义 | 步长 |
|---|---|---|
| `i`（`grid_h//2`） | merge 块的行号：第几个 28×28 大块 | 28 像素 |
| `j`（`merge_size=2`） | 块内 patch 的行偏移 | 14 像素 |
| `k`（`patch_size=14`） | patch 内像素行偏移 | 1 像素 |

W 方向对称。拆完为 8 维：

```
(B, C, 块行i, 块内patch行j, patch内行k, 块列i', 块内patch列j', patch内列k')
 0  1    2         3            4          5         6            7
```

### 2. permute 在做什么

目标：让**同一个 2×2 merge 块里的 4 个 patch 在展平后的序列中相邻**（模型端按每 4 个连续 patch 一组做 merge）。

`permute(0, 2, 5, 3, 6, 1, 4, 7)` → `(B, 块行, 块列, 块内行, 块内列, C, patch行, patch列)`。后续 reshape 按行优先遍历，patch 顺序为：

```
块(0,0): 左上 右上 左下 右下   ← 4 个相邻
块(0,1): 左上 右上 左下 右下
块(1,0): ...
```

每个 patch 内容 `(C, 14, 14)`，连同时间维复制展平后为 `C×2×14×14 = 1176` 维。

### 3. 56×56 图走一遍

```
(1, 3, 56, 56)
  → reshape → (1, 3, 2, 2, 14, 2, 2, 14)
  → permute → (1, 2, 2, 2, 2, 3, 14, 14)
  → expand+reshape → (1, 16, 1176)     16 = 4×4 个 patch
```

patch 0~3 属左上 merge 块，4~7 属右上块……merger 直接把每连续 4 个 1176 维向量拼起来过 MLP，得 `16/4 = 4` 个视觉 token。

**一句话总结**：reshape 利用"行号 = 块行×28 + 块内行×14 + 像素行"恒等式，零拷贝地把空间位置编码进下标结构；permute 调整遍历顺序，让同一 merge 块的 patch 在序列中相邻，为 2×2 merge 和位置编码做准备。

---

## Q5：unsqueeze/expand/reshape 复制"帧"生成 1176 维 patch 的细节

### 0. 模型端期待什么

`modeling_qwen3_vl.py:92-100`——patch embedding 是 **3D 卷积**（图/视频共用同一套权重）：

```python
kernel_size = [temporal_patch_size, patch_size, patch_size]   # [2, 14, 14]
self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size)

def forward(self, hidden_states):
    hidden_states = hidden_states.view(-1, in_channels, temporal_patch_size, patch_size, patch_size)
    hidden_states = self.proj(hidden_states).view(-1, embed_dim)
```

kernel = stride = `(2, 14, 14)`，一次吃掉"连续 2 帧上同一 14×14 位置"的像素管（tube）→ 一个 `embed_dim` 维向量。所以处理器必须保证每个 patch 是 `C×2×14×14 = 1176` 个数，且内存排列恰好是 `(C, T, H', W')`。

### 1. 逐操作分解

进入前 shape：`(B, grid_h//2, grid_w//2, 2, 2, C, 14, 14)`。

**① `unsqueeze(6)`**：在 C 之后、像素维之前插入大小为 1 的时间维 → `(..., C, 1, 14, 14)`。位置对应模型端 `(C, T, 14, 14)` 的顺序。

**② `expand(-1,...,-1, 2, -1, -1)`**：时间维扩成 `temporal_patch_size = 2`。**零拷贝**（stride=0），两个"帧"指向同一份像素——同一张图被当成"两帧完全相同的画面"。

> **为什么复制而不是补零帧？** Conv3d 权重在视频上训练，每个 tube 都是 2 帧真实画面；补零帧会使一半卷积输入恒为 0，偏离训练分布。复制同一帧等价于"静止视频"，在分布内。这也是 `image_grid_thw` 里图片 `t=1` 的原因：像素层面给了 2 帧，位置编码层面告诉模型"时间没有推进"。

**③ `reshape(B, grid_h*grid_w, C*2*14*14)`**：中间 4 个网格维相乘 = patch 总数（保持 merge 块相邻顺序）；尾部 `(C, 2, 14, 14)` 展平成 1176 维。因 expand 产生 stride=0 非连续维，**此 reshape 会实际拷贝数据**——复制帧的开销在此发生。

### 2. 两头格式的"暗号"

```
处理器每行 1176 个数的排列:  (C, T=2, 14, 14) 展平
模型端 view 的还原方式:      (-1, C, 2, 14, 14)
```

两者必须逐维对应，否则通道、时间、像素错位纠缠。前面 permute 把 C 挪到像素维前面、unsqueeze 插在 C 之后，都是为了让这一刻 `view` 无损还原——处理器与模型之间没有元数据传递排列方式，全靠这个约定。

### 3. 完整图景（1008×784 的图，grid 72×56）

```
permute 后:     (1, 36, 28, 2, 2, 3, 14, 14)
unsqueeze(6):   (1, 36, 28, 2, 2, 3, 1, 14, 14)
expand:         (1, 36, 28, 2, 2, 3, 2, 14, 14)   ← 每 patch 变"2 帧静止画面"
reshape:        (1, 4032, 1176)
↓ 跨图 cat 后去掉 B 维
pixel_values:   (Σpatch, 1176)
↓ 模型端
view(-1, 3, 2, 14, 14) → Conv3d → (Σpatch, embed_dim)
```

视频走同一条路，区别是视频本有时序帧：`t_bar` 帧两两一组天然填满 T 维，无需 expand，`video_grid_thw` 的 `t > 1`。图片只是"视频流水线"的特例（t=1 的静止视频）——这是 Qwen-VL 系列用一套架构统一图/视频的核心技巧。

---

## Q6：pixel_values 变长拼接与 image_grid_thw 的作用

这两行是"处理器输出协议"的核心：**数据（pixel_values）和索引（image_grid_thw）分离**。

### 1. 为什么拼成无 batch 维的变长序列

动态分辨率下一个 batch 内各图 patch 数不同：

```
图1: 1008×784 → 4032 patch；图2: 336×252 → 288 patch；图3: ~3570 patch
```

无法堆成规则 `(B, C, H, W)`，pad 到最大图又浪费计算。选择 **packing**：所有图的 patch 沿 dim 0 首尾相接：

```python
pixel_values = torch.cat(processed_images, dim=0)   # (7890, 1176)
```

借用 LLM 的**变长序列打包（packing / varlen）**思路：多条不同长度序列拼成一条长序列 + 边界索引，用 FlashAttention varlen kernel 一次算完、注意力不跨序列。零填充浪费。

代价：**像素张量自己不知道属于哪张图**，边界信息全部外置到 `image_grid_thw`。

### 2. image_grid_thw：拼接序列的"目录"

```python
image_grid_thw = [[1, 72, 56], [1, 24, 18], [1, h3, w3]]
```

- **t**：图片恒为 1；视频为 patch 化后的帧网格数；
- **h, w**：patch 网格尺寸（`resized // 14`）。

约束：第 i 张图占 pixel_values 的行数 = `tᵢ·hᵢ·wᵢ`，按图片顺序连续排列；`cumsum(t·h·w)` 即切分点。

### 3. 消费方式一：切分与注意力边界

进视觉塔先算 `cu_seqlens`（`vision_utils.py:50-53`）：

```python
cu_seqlens = repeat_interleave(grid_thw[:,1] * grid_thw[:,2], grid_thw[:,0]).cumsum(0)
cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)     # [0, 288, 4320, ..., 7890]
```

传给 FlashAttention varlen 接口，**把 pack 成一条的序列在注意力层面重新隔开**——图 1 的 patch 不会 attend 到图 2。packing 省算力、隔离保语义。

视觉塔输出经 2×2 merger 压缩后按图切开（`modeling_qwen3_vl.py:1062-1063`）：

```python
split_sizes = (image_grid_thw.prod(-1) // spatial_merge_size**2).tolist()
image_embeds = torch.split(image_embeds, split_sizes)
```

每张图 `t·h·w/4` 个视觉 token，scatter 进文本序列中 `<|image_pad|>` 占位符位置——与处理器展开占位符时 `grid_thw.prod() // 4` 的数量严格一致，供需两侧都用 `image_grid_thw`。

### 4. 消费方式二：给每个 patch 生成位置坐标

视觉塔两个位置编码（`modeling_qwen3_vl.py:695-707`）：

**① 可学习绝对位置嵌入（双线性插值）**：`pos_embed` 是固定 `num_grid_per_side²` 的嵌入表，每张图按自己的 `(h, w)` 网格双线性插值采样出 `h×w` 个位置向量——动态分辨率下绝对位置编码的标准技巧。

**② 2D RoPE**：`get_vision_position_ids`（`vision_utils.py:97-113`）生成每个 patch 的 `(h_idx, w_idx)`：

```python
hpos, wpos = meshgrid(arange(h), arange(w))
hpos.reshape(h//m, m, w//m, m).transpose(1, 2).flatten()   # 重排成"merge块优先"顺序
```

注意这与处理器端 patch 排列顺序**同构**——先按 merge 块走、块内 2×2 相邻，两者逐行对齐。视频把同一组 `(h,w)` 坐标 `repeat(t, 1)`。

### 5. 消费方式三：LLM 侧的 MRoPE

视觉 token 进语言模型后，`image_grid_thw` 参与 `get_rope_index`（`modeling_qwen3_vl.py:1009-1014`）。MRoPE 把 RoPE 位置维拆成 **(t, h, w) 三轴**：

- **纯文本 token**：三轴共用同一递增序号，退化为普通 1D RoPE；
- **视觉 token**：t 轴标帧号，h/w 轴标 merge 后网格坐标（`h//2, w//2`），由 `grid_thw` 展开生成；
- **图像之后的文本**：起始位置从 `start + max(h, w) // 2` 继续，保证位置单调连续。

### 6. 总结

```
pixel_values    (Σpatch, 1176)   ← 纯数据，变长打包，无边界信息
image_grid_thw  (n, 3)           ← 纯元数据，承担 4 个角色：
   ① 切分点：t·h·w 划分每张图占多少行
   ② 注意力边界：cu_seqlens 防止跨图/跨帧注意力
   ③ ViT 位置编码坐标源：插值 pos_embed + 2D RoPE 的 (h,w)
   ④ LLM MRoPE 坐标源：(t,h,w) 三轴位置 + 占位 token 数量校验
```

这是变长多模态输入的通用范式：**把"形状"从张量里抽出来变成显式元数据**，数据平面可自由打包、切分、跨设备搬运，语义边界随时可从元数据精确重建。vLLM 端的 Qwen3-VL 实现沿用同一套 `(pixel_values, image_grid_thw)` 协议。

---

## Q7：从 _preprocess 到占位 token 展开，Processor 还做了哪些事？

发生在 `ProcessorMixin.__call__`（`processing_utils.py:648-706`）。按执行顺序：

```
processor(images=..., text=..., videos=...)
  │
  ├─ 0. prepare_inputs_layout        输入归一化、抓取远程图像
  ├─ 1. validate_inputs              至少有一种输入
  ├─ 2. _merge_kwargs                按模态分发参数
  ├─ 3. _process_images  ──► image_processor(...)  ← _preprocess 在此被调用
  │                      ──► replace_image_token × n 张图   ← 占位符展开在此
  ├─ 4. _process_videos  ──► video_processor(...) + replace_video_token（Qwen3 特有：时间戳）
  ├─ 5. get_text_with_replacements   展开后的占位字符串写回 text
  ├─ 6. tokenizer(text)              分词
  ├─ 7. _check_special_mm_tokens     防截断校验
  ├─ 8. create_mm_token_type_ids     生成模态类型 id（Qwen3 默认开启）
  └─ 9. 合并所有输出 → BatchFeature
```

### 0. prepare_inputs_layout（:656, :708-736）
单 `str` 包成 `[str]`；`fetch_images`：URL/路径在此才真正下载/打开成 PIL Image。

### 1-2. 校验与参数分发（:659-665）
`_merge_kwargs` 按 `Qwen3VLProcessorKwargs` 分成三组 kwargs，注入默认值（`processing_qwen3_vl.py:30-38`）：

```python
"text_kwargs": {"padding": False, "return_token_type_ids": False, "return_mm_token_type_ids": True},
"videos_kwargs": {"return_metadata": True},
```

`return_mm_token_type_ids=True` 默认开启——Qwen3-VL 与 Qwen2-VL 在 processor 侧的重要差异。

### 3. _process_images（:761-771）——图像处理 + 占位符展开绑在一起

```python
processed_images = self.image_processor(images, **kwargs)   # ← _preprocess 在此执行
for idx in range(len(images)):
    replacement_text = self.replace_image_token(processed_images, image_idx=idx)
```

**占位符展开是图像预处理完成后立刻做的**——展开数量依赖 `_preprocess` 输出的 `image_grid_thw`。Qwen3-VL 覆盖版（`processing_qwen3_vl.py:76-79`）：`num_image_tokens = image_grid_thw[idx].prod() // 4`，返回 `"<|image_pad|>" * N`。替换串暂存 `images_replacements`，**此时还没写回文本**。

### 4. _process_videos（:773-783）——Qwen3-VL 视频占位符更复杂

`replace_video_token`（`processing_qwen3_vl.py:81-107`）为**每一帧**生成：

```
<0.0 seconds><|vision_start|><|video_pad|>×frame_seqlen<|vision_end|>
```

时间戳由 `_calculate_timestamps` 根据 `video_metadata`（fps + 采样帧索引）算出，temporal patch 内取首尾帧时间平均；缺 fps 元数据则警告并默认 `fps=24`。这就是 `videos_kwargs` 默认 `return_metadata=True` 的原因。

### 5. get_text_with_replacements（:684, :806-909）——真正的"写回"

用分组正则扫描每条文本，**按出现顺序**消费替换串：第 i 个 `<|image_pad|>` 换成 `images_replacements[i]`。同时记录字符偏移（`text_replacement_offsets`，供 vLLM 等对账）。隐含约定：**文本占位符的数量和顺序必须与传入图片一致**。

### 6. tokenizer(text)（:690）
对**展开后**的文本分词。每个 `<|image_pad|>` 是词表里的单个 special token，N 个占位符 → input_ids 里 N 个连续相同 id——这 N 个位置就是模型端 scatter 视觉 embedding 的"落点"。

### 7. _check_special_mm_tokens（:691, :2321-2337）
校验分词前后特殊 token 数量一致，防止用户开截断把 `<|image_pad|>` 截掉（会导致模型端视觉 token 数与占位符数对不上，scatter 时崩），提前 fail-fast。

### 8. create_mm_token_type_ids（:696-697, :911-941）
生成与 `input_ids` 等长的标记序列：文本=0，image_pad=1，video_pad=2，audio=3。模型端用它决定当前占位符段该消费哪张图的 `grid_thw`（`modeling_qwen3_vl.py:980-1011`）。

### 9. 合并输出（:700-706）
最终 `BatchFeature`：`input_ids`、`attention_mask`、`mm_token_type_ids`、`pixel_values`、`image_grid_thw`（视频另有 `pixel_values_videos`、`video_grid_thw`、`video_metadata`）。

### 时序图

```
图像通路                          文本通路
─────────                        ─────────
fetch_images (URL→PIL)
   │
_preprocess ──► pixel_values
        └──► image_grid_thw ──► replace_image_token
                                   │  "<|image_pad|>"×N  ──┐
video 同理（+时间戳 placeholder）   │                       ├─► 正则写回 text ──► tokenizer
                                                           │                       │
                                                           │              ┌────────┴────────┐
                                                           │              │ 校验数量不截断     │
                                                           │              │ mm_token_type_ids │
                                                           │              └────────┬────────┘
                                                           └──────────────────────►│ 合并输出
```

**一句话总结**：`_preprocess` 解决"像素 → patch 张量"，其后 processor 做的是**图文对账**——用图像侧算出的网格尺寸决定文本侧占位符的数量与形式（视频还注入时间戳），把两者缝成一条 input_ids，并附上 `mm_token_type_ids` 和校验，保证模型端 scatter 视觉 embedding 时供需严格相等。

---

## Q8：从 pixel_values 被 ViT 消费到 image_grid_thw 驱动 MRoPE，模型做了哪些事？

主入口 `Qwen3VLModel.forward`（`modeling_qwen3_vl.py:1160`）。

### 端到端全景

```
pixel_values (Σpatch,1176) + image_grid_thw (n,3) + input_ids + mm_token_type_ids
        │
        ▼
Qwen3VLModel.forward (:1160)
  ① input_ids → inputs_embeds
  ② ViT 消费 pixel_values（grid_thw 驱动 3 次）
  ③ masked_scatter 灌入文本序列
  ④ grid_thw 第 4 次驱动：MRoPE 位置 id
  ⑤ 组装 DeepStack 注入材料
  ⑥ 文本解码器（MRoPE 生效 + DeepStack 注入）
```

### ① 文本侧先查 embedding（:1183-1184）

整条 `input_ids`（含 `<|image_pad|>` 占位符）先过文本 embedding 表。占位 token 得到"假的"文本向量把位置占住，稍后被视觉特征覆盖。

### ② ViT 消费 pixel_values（`get_image_features` :1045 → `Qwen3VLVisionModel.forward` :682）

**a) PatchEmbed（:704）**：`(Σpatch, 1176)` → `view(-1, 3, 2, 14, 14)` → Conv3d(kernel=stride=(2,14,14)) → `(Σpatch, hidden)`。

**b) 绝对位置嵌入插值（:695, :705-706）**——`grid_thw` **第 1 次驱动**：每张图按自己的 `(h,w)` 网格从固定嵌入表双线性插值出位置向量，加到 patch 特征上。

**c) 2D RoPE 位置 id（:701, :707-713）**——`grid_thw` **第 2 次驱动**：`get_vision_position_ids` 生成每个 patch 的 `(h_idx, w_idx)`（merge 块优先顺序，与处理器端 patch 排列同构），得到 cos/sin 供 attention 施加到 q/k。

**d) N 个 VisionBlock（:716-723）**——`grid_thw` **第 3 次驱动**：`cu_seqlens`（由 `t·h·w` 算出）把打包序列在注意力层面隔开。视觉注意力 `is_causal=False`（:204）——**同图内 patch 双向可见**，隔离只发生在图间/帧间。

**e) merger + deepstack（:715-736）**：

```python
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(...)
    if layer_num in self.deepstack_visual_indexes:     # 中间层抽特征
        deepstack_feature_lists.append(deepstack_merger(hidden_states))
merged = self.merger(hidden_states)                    # 末层过主 merger
```

- **merger**（:118-131）：`view(-1, hidden×4)` 把每连续 4 个 patch（同一 2×2 merge 块——处理器 permute 保证的相邻性在此兑现）→ LayerNorm → MLP → 投影到 LLM hidden_size，输出 `(Σpatch/4, llm_hidden)`；
- **deepstack**：额外从几个中间层抽特征，稍后注入 LLM 浅层。

出视觉塔后按 `grid_thw.prod(-1) // 4` 切成每张图一段（:1062-1063）。

### ③ 视觉特征灌入文本序列（:1196-1199）

```python
image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

`get_placeholder_mask`（:1068-1107）：`input_ids == image_token_id` 定位所有占位符，并做**供需校验**——`n_image_tokens × hidden == image_features.numel()`，占位符数（处理器承诺）与视觉 token 数（ViT 产出）必须严格相等。`masked_scatter` 按掩码覆盖占位向量。视频同理（:1201-1211）。

### ④ image_grid_thw 驱动 MRoPE（:1237-1246 → `get_rope_index` :931）

第 4 次驱动。`compute_3d_position_ids`（:1109）先强制要求 `mm_token_type_ids` 存在，然后：

**a) 视频 grid 拆分（:966-968）**：Qwen3 视频按帧插时间戳，t 帧视频拆成 t 个 `t=1` 的 grid：

```python
video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
video_grid_thw[:, 0] = 1
```

**b) 按模态分段（:984-995）**：`itertools.groupby` 把 `mm_token_type_ids` 切成连续的文本段(0)/图像段(1)/视频段(2)。

**c) 逐段生成 3D 位置（:999-1014）**：
- 文本段：`arange(len)` 三轴复制，从 `current_pos` 接续；
- 视觉段：`next(grid_iters[modality])` 按序消费一张图的 `grid_thw`，展开成 (t, h/2, w/2) 网格坐标；
- 段间衔接：`current_pos += max(grid_h, grid_w) // 2`，保证后续文本位置单调。

**d) 缓存 `rope_deltas`（:1129-1152）**：记录 3D 位置与 1D 序号的差值。**增量生成时**（KV cache 非空）不重算全序列，直接 `arange + delta` 推算新 token 位置——MRoPE 支持逐 token 解码的关键。

输出 `position_ids (3, B, L)`。

### ⑤ 组装 DeepStack 注入材料（:1213-1235）

图/视频掩码合并成 `visual_pos_masks`，deepstack 特征按掩码拼成统一布局的 list。

### ⑥ 文本解码器（:1248 → `Qwen3VLTextModel.forward` :768）

1. **位置 id 拆包（:807-811）**：`(4,B,L)` 第 0 路是文本位置（建因果掩码用），后 3 路是 MRoPE 的 (t,h,w)，`rotary_emb` 按三轴分别旋转 q/k 的不同频段——**MRoPE 在此真正生效**；
2. **因果掩码（:813-819）**：视觉 token 进 LLM 后遵守与文本相同的因果规则；
3. **DeepStack 注入（:838-844）**：前几层每层输出后 `hidden_states[visual_pos_masks] += deepstack_visual_embeds[layer_idx]`；
4. 过剩余层、RMSNorm、lm_head 输出 logits。

### 总结：image_grid_thw 的全部驱动点

```
                        image_grid_thw (n,3)
                               │
   ┌───────────┬───────────────┼───────────────┬──────────────┐
   ▼           ▼               ▼               ▼              ▼
pos_embed   2D RoPE         cu_seqlens      merger 后       get_rope_index
双线性插值   (h,w) 坐标     注意力隔离边界    按图切分        MRoPE (t,h,w)
(ViT 输入)  (ViT attn)     (ViT attn)      (ViT 输出)      + rope_deltas
```

**一句话总结**：模型前向是一次精密的"供需对接"——处理器承诺的每件事（patch 排列顺序、占位符数量、t=1 静止帧约定）在模型侧都有对应消费点，`image_grid_thw` 和 `mm_token_type_ids` 就是连接两侧的契约。vLLM 的实现逐项复刻同样的对接逻辑，只是每个消费点换成了高性能内核。

---

## Q9：从 preprocess 产生的 pixel_values 到被 ViT 消费，中间的链路？

链路不长，但有几个容易忽略的关键环节（tensor 化、设备/dtype 迁移、两层 forward 转发、Conv3d 消费点）。

### 链路总览

```
Qwen2VLImageProcessor._preprocess
   │  输出: pixel_values (Σpatch, 1176) float32 @ CPU + image_grid_thw (n,3) long
   ▼
① BatchFeature 打包 & return_tensors="pt" 张量化
   ▼
② 用户侧 .to(device) 迁到 GPU（dtype 此时仍是 float32）
   ▼
③ Qwen3VLForConditionalGeneration.forward (:1320)  ── 纯转发
   ▼
④ Qwen3VLModel.forward (:1160)  ── 路由: pixel_values is not None → 走图像分支
   ▼
⑤ get_image_features (:1045)  ── dtype 对齐: .type(self.visual.dtype)
   ▼
⑥ Qwen3VLVisionModel.forward (:682)  ── 先用 grid_thw 准备 3 样辅料
   ▼
⑦ 【消费点】patch_embed (:704) → Conv3d (:92-100)
   ▼
(Σpatch, embed_dim) —— pixel_values 正式变成 ViT 的 hidden states
```

### ① BatchFeature 打包（`processing_utils.py:706`）

processor 把所有模态输出合并为 `BatchFeature`（transformers 的"智能 dict"），`return_tensors="pt"` 时统一转成 torch tensor。此时：

- `pixel_values`：`(Σpatch, 1176)`，**float32，在 CPU 上**（normalize 在 float32 下完成，见 Q3）；
- `image_grid_thw`：`(n, 3)`，`torch.long`。

**不需要 DataLoader/collate_fn**——packing 设计下多图已拼成单个二维 tensor，天然就是"一个 batch"，无堆叠对齐问题。这也是该协议对 serving 友好的原因之一。

### ② 设备迁移（用户代码侧）

```python
inputs = processor(images=..., text=..., return_tensors="pt").to(model.device)
generated = model.generate(**inputs)
```

`BatchFeature.to(device)` 把所有 tensor 搬上 GPU，**dtype 不变**（仍 float32）。float32→bf16 的转换被刻意推迟到模型内部（第 ⑤ 步）。

### ③ 外层包装：`Qwen3VLForConditionalGeneration.forward`（:1320-1395）

带 `lm_head` 的生成类，本身**不碰 pixel_values**，原样转发给基座 `self.model(...)`。（`model.generate(**inputs)` 内部第一步也是调这个 forward。）

### ④ 路由：`Qwen3VLModel.forward`（:1160）

多模态分支路由（:1189）：

```python
if pixel_values is not None:
    image_outputs = self.get_image_features(pixel_values, image_grid_thw, ...)
```

纯文本请求整个图像分支跳过，ViT 一次都不会被调用。图/视频是两条平行分支（:1189 / :1201），可同时在场。

### ⑤ dtype 对齐：`get_image_features`（:1045-1059）

```python
pixel_values = pixel_values.type(self.visual.dtype)          # float32 → bf16/fp16
vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, ...)
```

**float32 → 视觉塔权重 dtype 在此发生**。不在处理器里直接输出 bf16 的原因：预处理的 normalize 需要 float32 精度，且处理器输出应与模型 dtype 解耦（同一 checkpoint 可按不同 dtype 加载）。转换点放在进视觉塔前一刻，是精度与通用性的折中。

### ⑥ 视觉塔入口：`Qwen3VLVisionModel.forward`（:682-704）

消费 pixel_values **之前**，先用 `grid_thw` 备好三样辅料：

```python
bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(grid_thw, ...)  # 绝对位置嵌入插值表
position_ids = get_vision_position_ids(grid_thw, ...)                                        # 2D RoPE 坐标
cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, ...)                         # varlen 注意力边界
```

这些准备**只依赖 grid_thw，不依赖 pixel_values 内容**——"形状元数据与像素数据分离"的又一体现（vLLM 因此可在无真实图片时预计算）。

### ⑦ 消费点：PatchEmbed（:704 → :95-101）

```python
# Qwen3VLVisionModel.forward :704
hidden_states = self.patch_embed(hidden_states)   # hidden_states 此刻就是 pixel_values

# Qwen3VLVisionPatchEmbed.forward :95-101
def forward(self, hidden_states):
    target_dtype = self.proj.weight.dtype
    hidden_states = hidden_states.view(
        -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
    )                                          # (Σpatch, 1176) → (Σpatch, 3, 2, 14, 14)
    hidden_states = self.proj(hidden_states.to(dtype=target_dtype))
                                               # Conv3d(3→embed_dim, kernel=stride=(2,14,14))
    return hidden_states.view(-1, self.embed_dim)   # (Σpatch, embed_dim)
```

三个动作：

1. **`view` 还原五维**：1176 维按 `(C, T, H', W') = (3, 2, 14, 14)` 重新解读——Q5 的"暗号"在此对账，零拷贝，依赖处理器端排列约定；
2. **二次 dtype 保险**：`.to(self.proj.weight.dtype)`——即使第 ⑤ 步被绕过（如直接调 visual），Conv3d 输入也必然与权重同 dtype；
3. **Conv3d 滑动**：kernel = stride，无重叠地把每个 `(3,2,14,14)` tube 线性投影成 `embed_dim` 维向量。

**到这里 pixel_values 被"消费完毕"**：从像素张量变成 ViT 语义空间的 `(Σpatch, embed_dim)` 特征序列，此后只以 hidden_states 身份参与后续计算（+ 插值位置嵌入 → RoPE → N×Block → merger，见 Q8）。

### 逐环节张量状态一览

| 环节 | 形状 | dtype | 设备 |
|---|---|---|---|
| ① processor 输出 | (Σpatch, 1176) | float32 | CPU |
| ② .to(device) 后 | (Σpatch, 1176) | float32 | GPU |
| ⑤ get_image_features 入口 | (Σpatch, 1176) | **bf16** | GPU |
| ⑦ view 后 | (Σpatch, 3, 2, 14, 14) | bf16 | GPU |
| ⑦ Conv3d 后（消费完成） | (Σpatch, embed_dim) | bf16 | GPU |

### 设计要点回顾

1. **零 collate**：packing 让 processor 直接产出"单 batch"，绕过 DataLoader 对齐问题；
2. **dtype 延迟转换**：处理器输出与模型 dtype 解耦，float32→bf16 推迟到进塔前一刻（两处设防：`get_image_features` 和 patch_embed 内部）；
3. **辅料先行**：grid_thw 驱动的位置/边界准备与 pixel_values 内容无关，可预计算——vLLM 利用这点做 multimodal 输入的 profiling 和缓存；
4. **消费点唯一**：pixel_values 全程只被 `patch_embed` 的 `view + Conv3d` 读一次，之后计算全部基于 hidden_states——这就是 vLLM 侧只需对齐这一个接口形状 `(Σpatch, 1176)` 的原因。

---

## Q10：带图片请求的推理整体流程、参与对象与关键方法（总览）

> 本章是全笔记的总览，细节回查 Q1-Q9。行号见 `models/qwen3_vl/` 与相关基础设施文件。

### 一、参与对象清单

| 对象 | 角色 | 关键职责 |
|---|---|---|
| `Qwen3VLProcessor` | 多模态编排器 | 串联三个子处理器 + 图文对账（`processing_qwen3_vl.py:42`） |
| ├ `Qwen2VLImageProcessor` | 图像预处理 | smart_resize、patch 重排、packing（`qwen2_vl/image_processing_qwen2_vl.py:92`） |
| ├ `Qwen2TokenizerFast` | 分词器 | 文本 ↔ token ids |
| └ chat_template | 对话模板 | messages → 带 `<\|im_start\|>`/`<\|vision_start\|>` 的 prompt 字符串 |
| `BatchFeature` | 数据容器 | 打包所有输出、张量化、设备迁移（`processing_utils.py:706`） |
| `Qwen3VLForConditionalGeneration` | 生成入口 | lm_head + `GenerationMixin.generate` |
| ├ `Qwen3VLModel` | 多模态融合层 | 视觉特征灌入、MRoPE 计算（:865） |
| │  ├ `Qwen3VLVisionModel` | 视觉塔 ViT | pixel_values → 视觉 token（:612） |
| │  │  ├ `Qwen3VLVisionPatchEmbed` | Conv3d patch 嵌入（:84） |
| │  │  ├ `Qwen3VLVisionBlock` ×N | 双向注意力 + MLP（含 `Qwen3VLVisionAttention`） |
| │  │  ├ `Qwen3VLVisionPatchMerger` | 2×2 merge + 投影到 LLM 维度（:118） |
| │  │  └ `deepstack_merger_list` | 中间层特征抽取（:643） |
| │  └ `Qwen3VLTextModel` | 语言模型 | 文本解码（:745） |
| │     ├ `embed_tokens` | token embedding（:754） |
| │     ├ `Qwen3VLTextDecoderLayer` ×N | 因果注意力 + MLP |
| │     └ `Qwen3VLTextRotaryEmbedding` | MRoPE 三轴旋转（:759） |
| └ `lm_head` | 输出投影 | hidden → vocab logits |
| `DynamicCache` | KV 缓存 | 增量解码的注意力缓存（:794） |
| `rope_deltas`（模型属性） | 位置增量缓存 | 解码期 MRoPE 位置推算（:1129-1152） |

### 二、整体流程（三个阶段）

```
═══════════ 阶段 A：预处理（CPU，每请求一次） ═══════════

messages / (text + images)
   │
   ├─ [可选] Qwen3VLProcessor.apply_chat_template()
   │     messages → prompt 字符串（含 <|vision_start|><|image_pad|><|vision_end|>）
   │
   ▼
Qwen3VLProcessor.__call__()                        processing_utils.py:648
   ├─ prepare_inputs_layout()          :708    URL→PIL、text 包成 list
   ├─ _merge_kwargs()                          按模态分发参数（Qwen3 默认 return_mm_token_type_ids=True）
   ├─ _process_images()                :761
   │    ├─ Qwen2VLImageProcessor.__call__ → _preprocess()    image_processing_qwen2_vl.py:148
   │    │     ├─ group_images_by_shape()             同尺寸分组
   │    │     ├─ smart_resize()              :62     动态分辨率（28 对齐 + min/max_pixels）
   │    │     ├─ resize()                            双三次插值
   │    │     ├─ rescale_and_normalize()             融合标准化（image_processing_backends.py:314）
   │    │     ├─ reshape + permute                   merge 块优先的 patch 排序
   │    │     └─ unsqueeze + expand + reshape        复制 2 帧 → (Σpatch, 1176)
   │    │     输出: pixel_values (Σpatch,1176) + image_grid_thw (n,3)
   │    └─ replace_image_token()           processing_qwen3_vl.py:76
   │          grid_thw.prod() // 4 → "<|image_pad|>" × N
   ├─ get_text_with_replacements()     :806    占位符展开写回 text
   ├─ tokenizer(text)                          分词 → input_ids
   ├─ _check_special_mm_tokens()       :2321   防截断校验
   └─ create_mm_token_type_ids()       :911    文本=0/图=1/视频=2
   ▼
BatchFeature {input_ids, attention_mask, mm_token_type_ids,
              pixel_values, image_grid_thw}  → .to(GPU)

═══════════ 阶段 B：Prefill（GPU，首个 token） ═══════════

GenerationMixin.generate()
   └─ 首轮 forward:
      Qwen3VLForConditionalGeneration.forward()   modeling_qwen3_vl.py:1320
        └─ Qwen3VLModel.forward()                  :1160
             ├─ embed_tokens(input_ids)            :1184   占位符先占坑
             ├─ get_image_features()               :1045
             │    ├─ pixel_values.type(visual.dtype)      float32→bf16
             │    └─ Qwen3VLVisionModel.forward()  :682
             │         ├─ get_vision_bilinear_indices_and_weights()  绝对位置嵌入插值表
             │         ├─ get_vision_position_ids()         2D RoPE 坐标
             │         ├─ get_vision_attention_seqlens()    cu_seqlens 注意力边界
             │         ├─ 【消费点】patch_embed()    :704   view(Σpatch,3,2,14,14) → Conv3d
             │         ├─ + pos_embed（双线性插值）
             │         ├─ N × Qwen3VLVisionBlock     :716   双向注意力（is_causal=False）
             │         │    └─ 中途按 deepstack_visual_indexes 抽中间层特征  :724
             │         ├─ merger()                  :730   4 patch→1 token，投影到 LLM 维度
             │         └─ split by grid_thw.prod()//4      :1062  按图切段
             ├─ get_placeholder_mask()              :1068   定位占位符 + 供需校验
             ├─ inputs_embeds.masked_scatter()      :1199   视觉特征灌入文本序列
             ├─ compute_3d_position_ids()           :1109
             │    └─ get_rope_index()               :931
             │         ├─ mm_token_type_ids 按模态分段（itertools.groupby）
             │         ├─ 文本段: arange×3；视觉段: grid_thw 展开 (t,h,w)
             │         └─ 缓存 rope_deltas                 供解码期使用
             └─ Qwen3VLTextModel.forward()          :768
                  ├─ position_ids 拆包 (4,B,L)      :807   文本位 + (t,h,w) 三轴
                  ├─ create_causal_mask()           :813
                  ├─ N × Qwen3VLTextDecoderLayer    :827   MRoPE 施加到 q/k
                  │    └─ 前 K 层: _deepstack_process()    :840   视觉中间特征加到浅层
                  └─ norm
        └─ lm_head(hidden[:, -1:])                 :1401   只算最后位置的 logits
   → 采样得到第 1 个新 token；KV 写入 DynamicCache

═══════════ 阶段 C：Decode（GPU，逐 token 循环） ═══════════

generate() 循环，每步:
   Qwen3VLModel.forward()  ⚡ 与 Prefill 的差异:
     ├─ pixel_values=None                ViT 不再运行（视觉信息已在 KV cache 里）
     ├─ 无 masked_scatter                 无图可灌
     ├─ compute_3d_position_ids() :1142   走增量分支:
     │    position_ids = arange(past_len, past_len+1) + rope_deltas
     │    （不重算全序列 MRoPE）
     └─ Qwen3VLTextModel 用 past_key_values 增量注意力，只处理 1 个新 token
   → 采样 → 拼接 → 直到 EOS / max_new_tokens
   → processor.batch_decode() 回文本
```

### 三、关键方法索引（按调用顺序）

| # | 方法 | 位置 | 作用 |
|---|---|---|---|
| 1 | `Qwen3VLProcessor.__call__` | processing_utils.py:648 | 预处理总编排 |
| 2 | `Qwen2VLImageProcessor._preprocess` | image_processing_qwen2_vl.py:148 | 像素 → patches |
| 3 | `smart_resize` | image_processing_qwen2_vl.py:62 | 动态分辨率 |
| 4 | `Qwen3VLProcessor.replace_image_token` | processing_qwen3_vl.py:76 | 占位符展开 |
| 5 | `get_text_with_replacements` | processing_utils.py:806 | 展开写回文本 |
| 6 | `create_mm_token_type_ids` | processing_utils.py:911 | 模态标记 |
| 7 | `Qwen3VLModel.forward` | modeling_qwen3_vl.py:1160 | 多模态融合主入口 |
| 8 | `get_image_features` | :1045 | dtype 对齐 + 调视觉塔 |
| 9 | `Qwen3VLVisionModel.forward` | :682 | ViT 主体 |
| 10 | `Qwen3VLVisionPatchEmbed.forward` | :95 | **pixel_values 消费点** |
| 11 | `Qwen3VLVisionPatchMerger.forward` | :128 | patch → LLM token |
| 12 | `get_placeholder_mask` | :1068 | 定位 + 供需校验 |
| 13 | `get_rope_index` | :931 | MRoPE 位置生成 |
| 14 | `compute_3d_position_ids` | :1109 | 位置 id 编排（含增量分支） |
| 15 | `Qwen3VLTextModel.forward` | :768 | LLM 解码（MRoPE 生效点） |
| 16 | `_deepstack_process` | :853 | 视觉特征注入浅层 |

### 四、对象协作图

```
            ┌──────────────────────────────────────────────┐
  请求 ───► │  Qwen3VLProcessor（编排器）                    │
            │   ├─ Qwen2VLImageProcessor ─ pixel_values ─┐ │
            │   ├─ Tokenizer ──────────── input_ids ────┤ │
            │   └─ (图文对账: 占位符展开 + 校验)            │ │
            └──────────────────────────────────────────┼─┘
                                                        ▼
                                              BatchFeature (.to GPU)
                                                        ▼
            ┌──────────────────────────────────────────────┐
            │  Qwen3VLForConditionalGeneration (generate)  │
            │   └─ Qwen3VLModel                            │
            │       ├─ Qwen3VLVisionModel (仅 Prefill)      │
            │       │    PatchEmbed→Blocks→Merger           │
            │       ├─ masked_scatter（视觉灌入文本）         │
            │       ├─ get_rope_index（MRoPE + rope_deltas）│
            │       └─ Qwen3VLTextModel ◄── DynamicCache    │
            │            （DeepStack 注入浅层）               │
            │   └─ lm_head → 采样                           │
            └──────────────────────────────────────────────┘
```

### 五、核心记忆点

1. **图像只在 Prefill 过一次 ViT**；Decode 阶段视觉信息全靠 KV cache + `rope_deltas` 延续；
2. **两份元数据契约**贯穿全程：`image_grid_thw`（视觉侧 4 次驱动）和 `mm_token_type_ids`（MRoPE 分段依据）；
3. **三个校验点**保证图文对齐：处理器防截断校验（`_check_special_mm_tokens`）、供需校验（`get_placeholder_mask`）、MRoPE 前置检查（`compute_3d_position_ids`）。

---

## Q11：Qwen3VLModel.get_image_features 方法详解

`modeling_qwen3_vl.py:1044-1066`，仅 8 行，但它是**"视觉塔输出 → LLM 可用特征"的收口点**。逐行拆解：

```python
def get_image_features(
    self,
    pixel_values: torch.FloatTensor,
    image_grid_thw: torch.LongTensor | None = None,
    **kwargs,
):
    # 1057  ① dtype 对齐
    pixel_values = pixel_values.type(self.visual.dtype)
    # 1058  ② 调用视觉塔
    vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
    # 1061  ③ 取出 merge 后的特征
    image_embeds = vision_output.pooler_output
    # 1062  ④ 按图切分
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    image_embeds = torch.split(image_embeds, split_sizes)
    # 1064  ⑤ 改写返回值
    vision_output.pooler_output = image_embeds
    return vision_output
```

### ① dtype 对齐（:1057）

```python
pixel_values = pixel_values.type(self.visual.dtype)
```

处理器输出的 pixel_values 是 **float32**（normalize 需要 float32 精度，且处理器不应与模型 dtype 耦合——见 Q9）。这里转成视觉塔权重的 dtype（通常 bf16）。转换点刻意放在进塔前一刻：预处理保精度、计算保效率。

### ② 调用视觉塔（:1058-1060）

```python
vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
```

把打包的 `(Σpatch, 1176)` 和元数据 `image_grid_thw` 一起交给 `Qwen3VLVisionModel`，内部完成 patch_embed → 位置编码 → N×Block → merger 全流程（见 Q8）。`return_dict=True` 要求返回结构化对象而非裸 tuple——因为接下来要读写它的字段。

返回值是 `BaseModelOutputWithDeepstackFeatures`，三个字段分工：

| 字段 | 内容 | 形状 |
|---|---|---|
| `last_hidden_state` | 最后一层 Block 输出（**未 merge**） | `(Σpatch, hidden)` |
| `pooler_output` | 主 merger 输出（**已 merge**，投影到 LLM 维度） | `(Σpatch/4, llm_hidden)` |
| `deepstack_features` | 若干中间层各过一个小 merger 的输出 list | `K × (Σpatch/4, llm_hidden)` |

### ③ 取出 merge 后特征（:1061）

LLM 需要的是 `pooler_output`——经过 2×2 merge、且维度已投影到 `llm_hidden` 的特征。`last_hidden_state` 是给训练/分析用的中间产物，推理链路不用。

### ④ 按图切分（:1062-1063）——本方法的核心增量

```python
split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
image_embeds = torch.split(image_embeds, split_sizes)
```

视觉塔输出仍是**打包的一条变长序列**（`(Σpatch/4, llm_hidden)`，多图首尾相接）。这里用 `image_grid_thw` 切成**每张图一段**的 tuple：

- `grid_thw.prod(-1)` = 每张图的 patch 数（`t·h·w`）；
- `// spatial_merge_size**2`（÷4）= merge 后每张图的视觉 token 数；
- `torch.split` 按此长度列表切开 → `tuple[(n₁, d), (n₂, d), ...]`。

这正是处理器侧展开占位符时 `grid_thw.prod() // 4` 的镜像计算——**供（这里切出的每段长度）需（占位符数量）两侧用同一份元数据、同一个公式**。

### ⑤ 改写返回值（:1064）

把切分后的 tuple 写回结构化输出再返回。`get_image_features` 的契约：**输入打包的像素，输出按图组织的特征**。

### 一个看似矛盾的细节：切开后又被拼回去

调用方 `Qwen3VLModel.forward`（:1193-1195）拿到结果后立刻：

```python
image_embeds = image_outputs.pooler_output        # tuple，每图一段
image_embeds = torch.cat(image_embeds, dim=0)     # 又拼回一条！
```

为什么切了再拼？因为 `get_image_features` 同时服务两类调用者：

1. **模型内部 forward**：需要一条完整序列做 `masked_scatter`（scatter 按掩码展平消费，与 per-image 无关）——拼回去即可；
2. **外部 API 用户**：直接调 `model.get_image_features(pixel_values, grid_thw)` 做图像编码（不接 LLM），这时**按图分段的 tuple 才是有用的形态**——每张图拿到自己的特征矩阵。

split 是为 API 契约做的，cat 是内部链路的成本（一次拼接，开销可忽略）。vLLM 等外部系统调用的也正是这个 per-image 接口。

### 与 get_video_features 的关系

`get_video_features`（:1024-1040）完全同构，只是把 `pixel_values_videos + video_grid_thw` 走同一条视觉塔。Qwen3-VL 图/视频共塔，两个方法只是入口包装，内部都是 `self.visual(...)`。

### 总结

```
输入:  pixel_values (Σpatch,1176) fp32  +  image_grid_thw (n,3)
  ① .type(visual.dtype)                    fp32 → bf16
  ② self.visual(...)                       → BaseModelOutputWithDeepstackFeatures
  ③ 取 pooler_output                       (Σpatch/4, llm_hidden)，已 merge、已投影
  ④ split by grid_thw.prod(-1)//4          → tuple，每张图一段（供需公式的"供"侧）
  ⑤ 写回 pooler_output 返回
输出:  pooler_output=tuple[per-image 特征], deepstack_features=中间层特征 list
```

**一句话**：`get_image_features` 是视觉塔的"外交接口"——对内负责 dtype 对齐和调塔，对外把打包序列按图切分，同时保留 deepstack 中间层特征，供模型 forward 或外部调用者按需取用。

---

## Q12：Qwen3VLVisionModel / Qwen3VLTextModel / Qwen3VLModel 的区别

三个类是 Qwen3-VL 的**三层俄罗斯套娃**——从外到内每层包装一层，职责完全正交。都在 `modeling_qwen3_vl.py` 里。

### 一句话区分

| 类 | 行号 | 一句话定位 | 输入 → 输出 |
|---|---|---|---|
| `Qwen3VLVisionModel` | :612 | **视觉塔**：只管看图 | `pixel_values + grid_thw` → 视觉 token 特征 |
| `Qwen3VLTextModel` | :745 | **语言模型**：只管读写文字 | `inputs_embeds + position_ids` → 文本 hidden states |
| `Qwen3VLModel` | :865 | **融合层**：把图"翻译"成文，再交给语言模型 | 原始多模态输入 → 融合后的 hidden states |

### 1. Qwen3VLVisionModel —— 视觉编码器（:612-736）

- **看见什么**：只认识 `pixel_values (Σpatch, 1176)` 和 `grid_thw (n,3)`，**完全不认识 token、文本、词表**；
- **内部组成**：`patch_embed`（Conv3d）→ `pos_embed`（可学习绝对位置）+ `rotary_pos_emb`（2D RoPE）→ `blocks ×N`（**双向**注意力，`is_causal=False`）→ `merger` + `deepstack_merger_list`；
- **输出**：`BaseModelOutputWithDeepstackFeatures`（merge 后特征 + 中间层特征）；
- **类比**：眼睛 + 视觉皮层。把像素转成"语义特征"，但自己不会说话。

### 2. Qwen3VLTextModel —— 文本解码器（:745-861）

- **看见什么**：只认识 embedding 序列和位置 id，**不知道哪些向量来自图像**——视觉 token 对它只是"一串普通的输入向量"；
- **内部组成**：`embed_tokens` → `layers ×N`（**因果**注意力）→ `norm`，外加两个多模态钩子：
  - `position_ids (4, B, L)` 拆包，MRoPE 三轴施加到 q/k（:807-824）；
  - `_deepstack_process`：前几层把视觉中间特征加到视觉位置（:853-861）；
- **输出**：`BaseModelOutputWithPast`（hidden states + KV cache）；
- **类比**：大脑语言中枢。收到的是"已经翻译好的"信息，不需要知道信息来自眼睛还是耳朵；
- **配置独立**：有自己的 `Qwen3VLTextConfig`，可单独实例化（注释写明 "not a pure text-only model"，就是因为 deepstack 钩子）。

### 3. Qwen3VLModel —— 多模态融合器（:865-1258）

上面两者的**粘合层 + 翻译官**，本身几乎没有计算层，全是编排逻辑：

1. 持有 `self.visual` 和 `self.language_model` 两个子模块（:872-873，均用 `AutoModel.from_config` 构建，体现解耦）；
2. `get_image_features` / `get_video_features`：调视觉塔并按图切分（见 Q11）；
3. `get_placeholder_mask` + `masked_scatter`：视觉特征灌进文本 embedding 序列（:1068, :1199）——**"翻译"发生在这里**：图像从像素空间进入文本 embedding 空间；
4. `compute_3d_position_ids` / `get_rope_index`：为混合序列计算 MRoPE 位置（:931, :1109）；
5. 组装 `visual_pos_masks` + `deepstack_visual_embeds`，传给文本模型（:1213-1246）。

**类比**：大脑的联合皮层——接收眼睛的信号，转换成语言中枢能处理的格式，并告诉它"第 5-1012 号位置的信息来自一张 72×56 网格的图"。

### 关键区别维度

| 维度 | VisionModel | TextModel | VLModel |
|---|---|---|---|
| 注意力类型 | 双向（图内） | 因果 | 不自己做注意力 |
| 位置编码 | 2D RoPE（h,w） | **3D MRoPE**（t,h,w） | 计算 MRoPE 的 position_ids |
| 是否接触 pixel_values | ✅ 唯一直接消费者 | ❌ | 只做转发，不计算 |
| 是否接触 input_ids/词表 | ❌ | ✅ | ✅（但只为查 embedding 和找占位符） |
| 是否有新参数 | 全部视觉参数 | 全部文本参数 | **几乎没有**（纯编排） |
| 可独立使用 | ✅ 图像编码器 | ✅ 可当纯文本 LLM | 多模态时才需要 |

### 套娃关系全景（含最外层）

```
Qwen3VLForConditionalGeneration        ← 生成接口：lm_head + generate() + loss
   └─ Qwen3VLModel                     ← 融合：masked_scatter + MRoPE + DeepStack 组装
       ├─ Qwen3VLVisionModel (visual)  ← 看图：pixel_values → 视觉 token
       └─ Qwen3VLTextModel (language)  ← 读写：embeds → hidden states
```

### 为什么这样拆

1. **图/视频/文本复用**：VisionModel 同时服务图像和视频入口；TextModel 理论上可独立加载为纯文本模型；
2. **组合式配置**：`Qwen3VLConfig` 内含 `vision_config` + `text_config` 两个子配置，两个子模型各自用 `AutoModel.from_config` 构建（:872-873），视觉塔换架构不用动文本侧；
3. **vLLM 移植友好**：vLLM 侧对应拆成 `Qwen3VLVisionTransformer` 和 `Qwen3LLMModel`，融合逻辑（scatter、MRoPE）重写在 vLLM 的 model 类里——三方职责一一对应。

**一句话总结**：VisionModel 管"看"，TextModel 管"说"，VLModel 管"把看到的变成能说的"——三者通过 `(pixel_values, grid_thw)` 和 `(inputs_embeds, position_ids)` 两份标准化接口通信，互不越界。

---

## Q13：vLLM 端 Qwen3_VisionTransformer 对象的各个字段作用是什么？

> 本章从 transformers 参考实现切到 **vLLM 侧的重写实现**——正是 Q12 结尾提到的「vLLM 侧对应拆成 `Qwen3VLVisionTransformer` 和 `Qwen3LLMModel`」。
> 源码：`vllm/model_executor/models/qwen3_vl.py:552`。与 transformers 的 `Qwen3VLVisionModel`（笔记 Q8/Q11/Q12）数学等价，但字段按 serving 需求重新组织。

### 0. 类总览

```python
class Qwen3_VisionTransformer(nn.Module):      # qwen3_vl.py:552
    hf_to_vllm_mapper = WeightsMapper(          # HF 权重名 → vLLM 权重名的映射
        orig_to_new_stacked={
            "attn.q.": ("attn.qkv.", "q"),      # HF 独立 q/k/v → vLLM 的 stacked qkv
            "attn.k.": ("attn.qkv.", "k"),
            "attn.v.": ("attn.qkv.", "v"),
        }
    )

    def __init__(self, vision_config, norm_eps=1e-6, quant_config=None, prefix=""):
        ...
```

`__init__`（:561-663）里的 20 多个字段按性质分四类：**① 从 `Qwen3VLVisionConfig` 直接拷贝的标量配置**、**② 并行/部署相关派生量**、**③ 可学习子模块（参数所在）**、**④ 运行时状态**。逐类拆解。

### 1. 标量配置字段 —— `vision_config` 直接映射（:569-581）

| 字段 | 行号 | 默认值 | 作用 |
|---|---|---|---|
| `hidden_size` | :569 | 1152 | ViT 每个 patch 的特征维度（= `head_dim × num_heads`） |
| `num_heads` | :570 | 16 | 视觉注意力头数 |
| `num_position_embeddings` | :571 | 2304 | 绝对位置嵌入表的行数（= 48×48 网格） |
| `patch_size` | :572 | 16 | 空间 patch 边长 |
| `spatial_merge_size` | :573 | 2 | 2×2 merge 的边长 |
| `spatial_merge_unit` | :574 | 4 | `= merge_size²`，一个 merge 块合并的 patch 数 |
| `temporal_patch_size` | :575 | 2 | 时间维 patch（一个 tube 覆盖的帧数） |
| `deepstack_visual_indexes` | :576-580 | (8, 16, 24) | 抽中间层特征的层号；config 无此属性时回退 `[]` |
| `num_grid_per_side` | :581 | 48 | `= int(2304**0.5)`，pos_embed 表对应的网格边长 |

几个值得注意的点：

- **`patch_size = 16` 是 Qwen3-VL 相对 Qwen2-VL 的变化**。笔记 Q2-Q9 里反复出现的 `1176 = 3·2·14·14` 是 Qwen2-VL（patch=14）的 patch 维度；Qwen3-VL 用 patch=16，扁平 patch 维度是 `3·2·16·16 = 1536`。相应地 smart_resize 的 `factor = patch_size × merge_size = 32`（Qwen2-VL 是 28）。`Qwen3_VisionPatchEmbed` 构造函数里那个 `patch_size: int = 14` 只是默认值（:372），实际由 config 传入 16 覆盖。
- **`spatial_merge_unit` 不是独立配置**，是 `spatial_merge_size**2` 的缓存，merger 里把连续 4 个 patch 拼成一个视觉 token（对应 Q4/Q11 的 2×2 merge）。
- **`num_grid_per_side`**：`pos_embed` 是 `num_position_embeddings × hidden_size` 的嵌入表，按 `num_grid_per_side × num_grid_per_side`（48×48）网格理解。动态分辨率下每张图按自己的 `(h,w)` 网格在表上双线性插值（见 `fast_pos_embed_interpolate` :718）。

### 2. 并行 / 部署派生字段（:583-612）

```python
use_data_parallel = is_vit_use_data_parallel()                          # :583
self.tp_size = (                                                        # :584-588
    1 if use_data_parallel
    else parallel_state.get_tensor_model_parallel_world_size()
)
self.out_hidden_size = (                                                # :592-594
    vision_config.out_hidden_size * (1 + len(self.deepstack_visual_indexes))
)
self.fp8_padded_hidden_size = get_fp8_padded_hidden_size(               # :610-612
    self.num_heads, head_dim
)
```

| 字段 | 作用 |
|---|---|
| `tp_size` | 视觉塔张量并行世界大小；ViT 走数据并行（复制）时为 1。注意力后端据此把 `cu_seqlens` 按 TP 切片 |
| `out_hidden_size` | 视觉塔**最终**输出维度。`vision_config.out_hidden_size`（=3584，即 LLM hidden 维度）是单路 merger 的输出维度，但 forward 会把 `[主 merger 输出] + deepstack 各层输出` 沿 dim 1 拼接（:871-873），所以预先 `×(1 + deepstack 层数)`——DP ViT 下为 all_gather 空张量预留正确大小 |
| `fp8_padded_hidden_size` | FP8 注意力下 Q/K/V 量化后是三个独立连续张量，`cu_seqlens` 用统一 stride（不再有 3×V 的跳变），这是 padding 后的 hidden size（:608-612 注释说明） |

### 3. 可学习子模块 —— 参数真正所在（:596-663）

```python
self.patch_embed = Qwen3_VisionPatchEmbed(...)     # :596   Conv3d patch 嵌入
self.pos_embed = nn.Embedding(...)                 # :603   可学习绝对位置嵌入表
self.rotary_pos_emb = get_rope(...)                # :614   2D RoPE（cos/sin 缓存）
self.merger = Qwen3_VisionPatchMerger(...)         # :621   主 merger（末层）
self.deepstack_merger_list = nn.ModuleList([...])  # :630   deepstack 各中间层的小 merger
self.blocks = nn.ModuleList([...])                 # :650   depth×Qwen3_VisionBlock
```

**`patch_embed`**（`Qwen3_VisionPatchEmbed` :369）：`Conv3d(in_channels=3 → hidden_size=1152, kernel=stride=(2,16,16))`。对应笔记 Q5 的 3D 卷积 patch embedding——把打包像素 `(Σpatch, 1536)` 按 `view(L, -1, 2, 16, 16)` 还原成 `(C,T,H',W')`，一次卷积吃掉一个 tube，输出 `(Σpatch, 1152)`。`forward` 里 `-1` 自动解析为 `in_channels=3`，即 Q5 的"暗号"在 vLLM 侧的对账点。

**`pos_embed`**（:603）：`nn.Embedding(2304, 1152)`，48×48 网格上每个位置的绝对位置向量。动态分辨率下由 `fast_pos_embed_interpolate`（:718，优先 Triton、无 Triton 退回 native）双线性插值出任意 `(h,w)` 网格的位置嵌入，`merge` 重排顺序也 baked 进插值索引。

**`rotary_pos_emb`**（:614-619）：`get_rope(head_size=72, max_position=8192, is_neox_style=True, partial_rotary_factor=0.5)`——2D RoPE，只旋转 head_dim 前一半（partial 0.5），本身无参数，但提供 `get_cos_sin(max_grid_size)` 从缓存取 cos/sin，配合 `rot_pos_ids`（:673，`@lru_cache`）生成每个 patch 的 `(h,w)` 旋转角。

**`merger`**（:621，`Qwen3_VisionPatchMerger` :500）：把连续 `spatial_merge_size²=4` 个 patch（`4×1152=4608` 维）→ LayerNorm → 线性 + GELU → 线性投影到 `out_hidden_size=3584`（LLM 维度）。这是「patch 特征 → 视觉 token」的收口（对应 Q8/Q11 的 merger）。

**`deepstack_merger_list`**（:630-643）：`deepstack_visual_indexes` 里每个层号对应一个 merger，与主 merger 的唯一差异是 `use_postshuffle_norm=True`——先 `view(-1, hidden_size)` 再 norm（:541-544），而非先 norm 再 view。Qwen3-VL 的 deepstack 机制：从第 8/16/24 层各抽一次中间特征、各过一个小 merger，最后与主 merger 输出拼一起（:871-873），供 LLM 浅层注入（对应 Q8 的 DeepStack）。

**`blocks`**（:650-663）：`vision_config.depth=27` 个 `Qwen3_VisionBlock`（:446）。每块 = `norm1` + 双向注意力（`Qwen2_5_VisionAttention`，`is_causal=False`）+ `norm2` + MLP（`Qwen3_VisionMLP`）。注意力靠 `cu_seqlens` 隔开图间/帧间（对应 Q6/Q8 的 varlen 隔离）。

### 4. 运行时状态字段与属性（:605-606, :645-671）

```python
head_dim = self.hidden_size // self.num_heads      # :606  局部变量，72
self.attn_backend = get_vit_attn_backend(          # :645  视觉注意力后端
    head_size=head_dim, dtype=torch.get_default_dtype()
)
@property
def dtype(self):  return self.patch_embed.proj.weight.dtype    # :665-667
@property
def device(self): return self.patch_embed.proj.weight.device   # :669-671
```

- **`attn_backend`**（:645）：视觉塔注意力后端选择（FlashAttention / FlashInfer / Triton 等）。`prepare_encoder_metadata` 按后端决定 `cu_seqlens` 是否需要重算（:822）、是否生成 `sequence_lengths`/`max_seqlen`（:805/:813）——同一份元数据同时服务 eager、CUDA graph capture 和 replay 三条路径。
- **`dtype` / `device`**（:665/:669）：**不存字段，而是从 `patch_embed.proj.weight` 动态读**——保证"视觉塔的 dtype/device"与权重真实所在一致，load 权重后自动跟随，避免状态不同步。`forward` 入口用它做 `x.to(device=self.device, dtype=self.dtype)`（:840）。

### 5. forward 里字段如何串起来（:833-874）

```
x (Σpatch, C·T·P·P)                # Qwen3-VL: 3·2·16·16 = 1536（Qwen2-VL 是 1176）
  → patch_embed                    # Conv3d → (Σpatch, hidden_size=1152)
  → + pos_embeds                   # fast_pos_embed_interpolate 用 pos_embed 插值
  → unsqueeze(1)                   # (Σpatch, 1, 1152)
  → depth × blocks                 # 每块用 rotary_pos_emb 的 cos/sin + cu_seqlens 做双向注意力
        └─ layer ∈ deepstack_visual_indexes → deepstack_merger_list 抽中间特征
  → merger                         # (Σpatch/4, out_hidden_size=3584)
  → cat([主 merger] + deepstack 各层)  # (Σpatch/4, out_hidden_size × (1+3)) = 14336
```

### 总结

字段分四层，职责清晰：

| 层 | 字段 | 特点 |
|---|---|---|
| 标量配置 | hidden_size/num_heads/patch_size/merge/temporal/num_position_embeddings/deepstack_visual_indexes | 从 `Qwen3VLVisionConfig` 原样拷来，超参 |
| 并行派生 | tp_size/out_hidden_size/fp8_padded_hidden_size | 为 TP/DP、deepstack 拼接、FP8 对齐预先算好 |
| 可学习子模块 | patch_embed/pos_embed/merger/deepstack_merger_list/blocks（27×） | 真正含参数的部分，与 transformers 的 `Qwen3VLVisionModel` 一一对应 |
| 运行时状态 | attn_backend + dtype/device 属性 | 后端选择 + 从权重动态读，服务 CUDA graph 与部署 |

**一句话总结**：`Qwen3_VisionTransformer` 是 `Qwen3VLVisionModel` 的 vLLM 重写——把 serving 关心的并行切分、注意力后端、FP8 对齐、deepstack 拼接从"运行时逻辑"提前显式化为字段，可学习子模块（patch_embed → pos_embed → blocks → merger/deepstack）仍与 transformers 参考实现逐项对应，只是每个消费点换成了高性能内核。

---

## Q14：Qwen3_VisionTransformer.forward 方法一步步做了什么（含原理）

> 源码：`vllm/model_executor/models/qwen3_vl.py:833-874`。本章把 Q13 介绍的字段"串"起来，逐行讲 forward 做了什么、为什么这么做，并补齐每一步背后的原理（patch/tube、位置编码、RoPE、变长注意力、双向注意力、merge、deepstack）。适合对 ViT 视觉塔不熟的读者。

### 0. 先看整体：forward 就 7 步

```python
def forward(self, x, grid_thw, *, encoder_metadata=None):   # :833
    hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)  # ① :840
    hidden_states = self.patch_embed(hidden_states)          # ② :841
    if encoder_metadata is None:                             # ③ :843-848
        grid_thw_list = grid_thw if isinstance(grid_thw, list) else grid_thw.tolist()
        encoder_metadata = self.prepare_encoder_metadata(grid_thw_list)
    pos_embeds = encoder_metadata["pos_embeds"]              # ④ :850-852
    hidden_states = hidden_states + pos_embeds
    hidden_states = hidden_states.unsqueeze(1)
    deepstack_feature_lists = []                             # ⑤ :854-869
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(hidden_states, ...)
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature_lists.append(self.deepstack_merger_list[...](hidden_states))
    hidden_states = self.merger(hidden_states)               # ⑥ :870
    hidden_states = torch.cat([hidden_states] + deepstack_feature_lists, dim=1)  # ⑦ :871-873
    return hidden_states
```

七步记忆口诀：**搬设备 → 像素变特征 → 算辅料 → 加位置 → 过层堆 → 降采样 → 拼 deepstack**。

### 1. 输入长什么样（:833-839）

- `x`：形状 `(Σpatch, 1536)` 的浮点张量。**一行 = 一个 patch**。`1536 = 3 通道 × 2 帧 × 16×16`（Q13 讲过，Qwen3-VL 的 patch=16）。这个 x 就是 processor 输出的 `pixel_values`（见 Q6/Q9）。
- `grid_thw`：`list[list[int]]`，每张图/视频一行 `[t, h, w]`，即 patch 网格尺寸。
- **关键前提**：x 是"多张图打包成一条序列"——所有图的所有 patch 沿 dim 0 首尾相接，**没有 batch 维**。哪段属于哪张图，全靠 `grid_thw` 记录。这就是 Q6 讲的"数据与索引分离"。

> 为什么能打包？因为每张图分辨率不同、patch 数不同，无法堆成规则的 `(B, N, D)`；pad 到最大图又浪费算力。于是借用 LLM 的 varlen（变长序列打包）思路：拼成一条，用边界索引告诉注意力"哪里不能互相看"。后面 4.3 展开。

### 2. 第①步：迁移设备与 dtype（:840）

```python
hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
```

- `self.device` / `self.dtype` 是 Q13 介绍的两个 property，从 `patch_embed.proj.weight` 动态读出（:665-671），保证和权重真实所在一致。
- **为什么在这里做**：processor 输出在 CPU 上、float32（normalize 需要 float32 精度，见 Q9）；视觉塔权重通常 bf16、在 GPU。这一步把像素搬上 GPU、转成 bf16。
- `non_blocking=True`：异步拷贝，不阻塞调用线程（vLLM 输入张量常由 `async_tensor_h2d` 预取，这里再保险一次）。

### 3. 第②步：patch_embed —— 像素变特征（:841 → :391-395）

```python
def forward(self, x):                       # Qwen3_VisionPatchEmbed.forward :391
    L, C = x.shape                          # L=Σpatch, C=1536
    x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)  # :393
    x = self.proj(x).view(L, self.hidden_size)   # :394
    return x                                 # (Σpatch, 1152)
```

**先讲原理：什么是 patch / tube？**

ViT 不能像 LLM 那样直接"吃"像素，因为自注意力是 O(n²) 的——一张 1024×1024 的图有 100 万像素，两两注意力就是 10¹² 次运算，不可行。解法是**先把图切成小块（patch）**，每个 patch 作为一个"token"。1024×1024 按 16×16 切，得到 64×64 = 4096 个 patch，规模就正常了。

Qwen3-VL 更进一步：它是**图/视频共塔**（Q5 讲过），所以 patch 不是 2D 的，而是 3D 的 **tube（管子）**——同时覆盖空间 `patch_size × patch_size` 和时间 `temporal_patch_size` 帧。一个 tube = 连续 2 帧上同一 16×16 位置的一块像素。

**为什么用卷积（Conv3d）而不是直接 flatten？**

最朴素的 patch embedding 是：把每个 patch 的像素 `flatten` 成一个长向量，再过一个线性层。而 `Conv3d(kernel=stride=(2,16,16))` 在数学上和它**完全等价**——卷积核在图上按 stride 滑动、每次取一个 tube 做内积，就是"每个 patch 的线性投影"。用卷积实现的好处：

1. **表达紧凑**：一个 Conv3d 层把"切块 + 线性投影"合并，权重 `(3→1152, kernel=(2,16,16))` 直接就是 patch embedding 矩阵；
2. **kernel=stride 无重叠**：stride 等于 kernel 尺寸，tube 之间不重叠、不遗漏，恰好无浪费地铺满整张图（这是"切块"而非"滑窗"的关键）。

`x.view(L, -1, 2, 16, 16)` 里的 `-1` 自动解析为 3（in_channels），就是把"一行 1536 个数"还原成 `(通道, 时间, 高, 宽)` 的 tube——这是 Q5 讲的"暗号"，处理器和模型约定好的排列方式。

经过这一步，`(Σpatch, 1536)` 像素 → `(Σpatch, 1152)` 特征。**从此 x 不再叫像素，而是 hidden_states**。

### 4. 第③步：prepare_encoder_metadata —— 用 grid_thw 算 5 样"辅料"（:737-831）

这步**不碰像素内容**，只用 `grid_thw`（形状元数据）预先算出 5 样东西，供后面加位置和做注意力用：

```python
metadata["pos_embeds"]        = self.fast_pos_embed_interpolate(grid_thw_list)   # :770
metadata["rotary_pos_emb_cos"], metadata["rotary_pos_emb_sin"] = self.rot_pos_emb(grid_thw_list)  # :771-773
metadata["sequence_lengths"]  = ...                                             # :805
metadata["max_seqlen"]        = ...                                             # :819
metadata["cu_seqlens"]        = ...                                             # :822
```

分三组，原理各不相同，逐个讲。

#### 4.1 绝对位置嵌入 + 双线性插值（pos_embeds，:770 → :718-735）

**原理一：Transformer 的注意力是"排列不变"的，必须注入位置信息。**

自注意力公式 `softmax(QK^T/√d)·V` 里，若把输入 token 顺序打乱，每个 token 的注意力输出会跟着打乱，但"每个位置算出来的东西"是相同的——它**天生不知道 token 之间谁在前谁在后**。对文本，这等于丢失"词序"；对图像，等于丢失"这个 patch 在图的哪个位置"。所以必须人为注入位置信息。

最简单的一类做法是**绝对位置嵌入**：给每个位置编一个向量，加到 token 上。Qwen3 视觉塔用的是**可学习**版本：`pos_embed = nn.Embedding(2304, 1152)`（Q13 的字段），即一张 `2304 × 1152` 的查表，第 i 行就是"第 i 个位置"的向量。这些向量在训练中学习，最终编码了空间结构。

**原理二：为什么动态分辨率下要插值，不能直接查表？**

`pos_embed` 只有 2304 行，对应 `num_grid_per_side = 48` 的 `48×48` 网格（2304 = 48²）。但 Qwen-VL 是**动态分辨率**的（Q2 的 smart_resize），一张图 resize 后 patch 网格可能是 72×56、24×18……任意尺寸，无法直接对齐 48×48 的表。

解法是**双线性插值**：把 48×48 的表当成一张"位置图"，按目标图自己的 `(h, w)` 网格，在每个目标位置周围找表里最近的 2×2 个点做加权平均，采样出 `h×w` 个位置向量。这就是 `fast_pos_embed_interpolate`（:718，优先 Triton 核、无 Triton 退回 `pos_embed_interpolate_native`）做的事。

> 双线性插值 = 二维的线性插值：目标点落在四个已知点围成的格子里，先上下两行各做一次横向线性插值，再对两个结果做一次纵向线性插值，权重由目标点离四边的距离决定。Triton 核（:176-289）里 `w00/w01/w10/w11` 就是这四个插值系数，`h_scale = (num_grid_per_side-1)/(h-1)` 把目标网格坐标归一化到表坐标。

**一个易忽略的细节**：插值出来的顺序是"merge 块优先"（`pos_embed_interpolate_native` 末尾的 `reshape(h//m, m, w//m, m, ...).permute(0,2,1,3,4)`），和 Q4 讲的 patch 排列顺序同构——保证加位置时逐 patch 对齐。

#### 4.2 2D RoPE（rotary_pos_emb_cos/sin，:771-773 → :700-716）

**原理：RoPE（旋转位置编码）是什么？**

绝对位置嵌入有个弱点：它学到的是"绝对位置 5 的向量"，但注意力真正关心的是**相对位置**（"这个 patch 在我右边 3 格"），而绝对编码表达相对关系很绕。RoPE 换了个思路——**不"加"位置向量，而是"旋转" q/k 向量**。

具体做法：把 q（或 k）向量按相邻两维配对 `(x0, x1)`，把每一对看成一个 2D 平面上的点，然后按位置 `m` 把它**旋转** `m·θ` 角度（θ 是每个维度对专属的"转速"）：

```
x0' = x0·cos(mθ) - x1·sin(mθ)
x1' = x0·sin(mθ) + x1·cos(mθ)
```

不同维度对用不同频率 `θ_i = base^(-2i/d)`：低频维度对转得慢、捕捉远距离，高频转得快、捕捉近距离，像傅里叶变换一样把不同尺度的相对位置信息铺开。

**为什么"旋转"能编码相对位置？** 这是 RoPE 的数学核心：位置 m 的向量转了 mθ，位置 n 的向量转了 nθ，它们做点积（注意力里的 QK^T）时，旋转角只以 `(m-n)θ` 出现——**点积只依赖相对位置 m-n，与绝对位置无关**。于是模型无需显式学"位置 5 和位置 8 差 3 格"，旋转相位天然携带了它。

实现上不需要真做矩阵旋转：预先算好每个位置的 `cos(mθ)`、`sin(mθ)`（`rotary_pos_emb.get_cos_sin` 缓存），前向时按元素乘加（`ApplyRotaryEmb`，qwen2_5_vl.py:397/:125-185）。代码里 `is_neox_style=True` 表示"把 head 前半和后半配对旋转"这种布局。

**2D RoPE：图像位置是二维的。**

文本位置一个数就够，图像 patch 的位置是 `(h, w)` 两个坐标。于是把 head 维度**劈成两半**：前一半用 h 坐标旋转，后一半用 w 坐标旋转。这样注意力既能感知"上下"又能感知"左右"的相对位置。`rot_pos_emb`（:700-716）里 `pos_ids` 是 `(Σpatch, 2)` 的 `[h_idx, w_idx]`（由 `rot_pos_ids` :673 生成），`cos[pos_ids].flatten(1)` 把 h 的 cos 和 w 的 cos 前后拼接，正好对应"前半 h、后半 w"。

`partial_rotary_factor=0.5`（Q13 的字段）：每个 head 只让一半维度参与旋转。这里配合 2D，恰好让 h、w 两轴各分到一半、覆盖整个 head。**只旋转 q/k，不旋转 v**（value 不携带位置，位置信息通过 q·k 的注意力权重进入）。

> 一个直觉类比：绝对位置嵌入是"给每个座位贴门牌号"；RoPE 是"给向量一个朝向"，两个向量的"夹角差"就表达了它们的相对位置。视觉塔同时用两种（绝对 pos_embed + 2D RoPE），各司其职：绝对位置给 patch 全局定位，RoPE 给注意力精确的相对位置。

#### 4.3 cu_seqlens —— 变长注意力的"边界"（:776-829）

**这是 forward 里最"serving"的一步。** 回到第 1 节的"打包"：多张图的 patch 被拼成一条序列，但注意力**绝不能让图 A 的 patch 去 attend 图 B 的 patch**（两张无关的图互相"看"毫无意义，还会串信息）。

`cu_seqlens`（cumulative sequence lengths，累积序列长度）就是用来划边界的：它是一个 `[0, len1, len1+len2, len1+len2+len3, ...]` 的数组，第 i 段 `[cu_seqlens[i], cu_seqlens[i+1])` 就是第 i 张图的 patch 区间。

```python
patches_per_frame = grid_thw_np[:, 1] * grid_thw_np[:, 2]     # :777  每帧 h*w 个 patch
cu_seqlens = np.repeat(patches_per_frame, grid_thw_np[:, 0]).cumsum()  # :778-780  按 t 展开后累加
cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])  # :781  前面补 0
```

注意这里按 `t` 展开：一张 `t=2` 的视频会被拆成 **2 段**（每帧一段，各 `h*w` 个 patch），因为视觉注意力按帧隔离（帧间不互相 attend，时间信息靠后续位置编码和 LLM 侧处理）。

**原理：varlen（变长序列）注意力。** 传统注意力要求一个 batch 里所有序列等长（pad 到 max_len）；varlen 注意力用一个 `cu_seqlens` 数组替代"固定长度"的假设，让不同长度的序列打包进同一个 kernel 调用，每个序列独立做注意力（softmax 分母只在段内求和）。这是 FlashAttention 的 `varlen` 接口标准输入，也是 LLM 连续 batching 的核心机制——视觉塔打包多图，正是借用了它。

后面两行（:787-829）在为不同注意力后端和 CUDA graph 做**对齐**：

- `pad_to`（:787-802）：把 `cu_seqlens` pad 到固定的 batch/帧数上限，这样 CUDA graph 捕获时缓冲区大小固定，replay 时不会越界（vLLM 的 encoder CUDA graph 机制）。
- `maybe_recompute_cu_seqlens`（:822-829）：FlashInfer 后端需要"字节偏移"而非"token 偏移"，还要分 Q/K/O 和 V 两条（FP8 下 Q/K/V 是独立张量、stride 一致；bf16 下 V 有 3× stride），这里按后端把 token 数换算成内存偏移。
- `sequence_lengths`（:805-807）、`max_seqlen`（:810-819）：同样是后端特需（FlashInfer CuDNN 要逐序列长度，FlashAttention 要最大序列长度）。

一句话：**cu_seqlens 是"打包序列的分隔符"，其余三个量是"为特定注意力后端/图捕获做的适配"**。

#### 4.4 小结：这一步为什么"预先"算好

这 5 样辅料**只依赖 grid_thw，不依赖像素内容**。所以可以在 forward 里一次性算好（甚至在无真实图片时预计算），CUDA graph capture 和 replay 也复用同一份（:746 的 docstring 明确写了这个设计意图）。这是"形状元数据与像素数据分离"的又一体现。

### 5. 第④步：加位置嵌入 + 加 batch 维（:850-852）

```python
pos_embeds = encoder_metadata["pos_embeds"]     # (Σpatch, 1152)
hidden_states = hidden_states + pos_embeds      # 绝对位置嵌入直接加
hidden_states = hidden_states.unsqueeze(1)      # (Σpatch, 1152) → (Σpatch, 1, 1152)
```

- **加 pos_embeds**：把 4.1 插值出的绝对位置向量加到每个 patch 的特征上。这是 ViT 的经典做法（patch_embed 输出 + 位置嵌入）。
- **`unsqueeze(1)`**：在第 1 维插入一个大小为 1 的维度。为什么？因为下面的注意力层期望输入是 `[seq_len, batch_size, hidden]` 的格式（qwen2_5_vl.py:409 注释 `[s, b, c]`）。我们打包成一条序列，batch 就是 1，于是 `(seq, 1, hidden)`。**这个"batch=1"不是真的只有一个样本，而是"所有 patch 打包成一个 batch 项"**——真正的"多序列"由 cu_seqlens 在注意力内部区分。

### 6. 第⑤步：循环 depth 个 VisionBlock（:855-869）

```python
for layer_num, blk in enumerate(self.blocks):    # depth=27 层
    hidden_states = blk(
        hidden_states,
        cu_seqlens=encoder_metadata["cu_seqlens"],
        rotary_pos_emb_cos=encoder_metadata["rotary_pos_emb_cos"],
        rotary_pos_emb_sin=encoder_metadata["rotary_pos_emb_sin"],
        max_seqlen=encoder_metadata["max_seqlen"],
        sequence_lengths=encoder_metadata.get("sequence_lengths"),
    )
    if layer_num in self.deepstack_visual_indexes:   # 第 8/16/24 层
        deepstack_feature_lists.append(self.deepstack_merger_list[...](hidden_states))
```

每个 `Qwen3_VisionBlock`（:446-497）是一个标准的 Transformer 编码器块：

```python
x = x + self.attn(self.norm1(x), ...)   # :487-494  残差 + 注意力
x = x + self.mlp(self.norm2(x))         # :496      残差 + MLP
```

两个要点（原理）：

**① 双向注意力（is_causal=False）。** 文本 LLM 用**因果注意力**——每个 token 只能看它前面的 token（因为生成时未来 token 还不存在）。但视觉编码器是**双向**的：一张图的 patch 之间没有先后，左上的 patch 完全可以"看"右下角的 patch，这样才能理解整张图。代码里注意力没有因果 mask，隔离只发生在**图间/帧间**（靠 cu_seqlens，见 4.3），图内全连通。

**② 残差连接 + LayerNorm（Pre-Norm）。** `x = x + attn(norm(x))` 的结构：先归一化再算注意力、再残差相加。残差让梯度能"抄近路"回流，是训练几十层深网络不崩的关键；LayerNorm 稳定每层输入的分布。这是几乎所有现代 Transformer 的标配。

**注意力内部**（`Qwen2_5_VisionAttention.forward`，qwen2_5_vl.py:399-457）：QKV 一次投影（:410）→ 拆出 q/k/v（:413-418）→ **q/k 施加 2D RoPE**（:420-441）→ varlen 注意力（:443-450）→ 输出投影（:456）。注意 RoPE 只施加到 q/k，v 不旋转。

**deepstack 中间层抽取**（:864-869）：若当前层号在 `deepstack_visual_indexes = (8, 16, 24)` 里，就把这一层输出另存一份，过一个小 merger（`deepstack_merger_list`），存进 `deepstack_feature_lists`。**原理**：Qwen3-VL 的 deepstack 机制——不只把最后一层视觉特征喂给 LLM，还额外把几个**中间层**的特征也抽出来，让 LLM 浅层能拿到视觉塔的低级/中级特征（边缘、纹理 vs 语义），类似多尺度/多分辨率特征复用。主 merger 只处理最后一层，deepstack 处理中间层。

### 7. 第⑥步：merger —— 2×2 merge 降采样（:870）

```python
hidden_states = self.merger(hidden_states)   # (Σpatch, 1, 1152) → (Σpatch/4, 3584)
```

`Qwen3_VisionPatchMerger.forward`（:540-549）：

```python
x = self.norm(x).view(-1, self.hidden_size)   # hidden_size = 1152×4 = 4608
x, _ = self.linear_fc1(x); x = self.act_fn(x); out, _ = self.linear_fc2(x)
```

**原理：为什么要 merge？** ViT 输出的 patch 太多了——一张图几百上千个 patch，若每个 patch 都变成一个视觉 token 喂给 LLM，序列会爆炸（token 数 = 计算量 = 显存）。所以把**空间上相邻的 `spatial_merge_size²=4` 个 patch 合并成 1 个 token**：`view(-1, 4608)` 把连续 4 个 1152 维 patch 拼成一个 4608 维向量，再过 LayerNorm + MLP 投影到 LLM 的 hidden 维度（3584）。

**为什么连续 4 个 patch 恰好是空间上相邻的 2×2 块？** 这是 Q4 埋的伏笔——processor 在预处理时用 permute 把 patch 排成"merge 块优先"顺序，同一 2×2 块的 4 个 patch 在序列里相邻。merger 只需"每连续 4 个拼一个"，就等价于"每 2×2 块合并"，无需任何索引运算。vLLM 侧沿用同一约定。

输出 `(Σpatch/4, 3584)`：patch 数降为 1/4，维度投影到 LLM 维度，每个元素就是一个"视觉 token"。

### 8. 第⑦步：拼接 deepstack 特征（:871-873）

```python
hidden_states = torch.cat([hidden_states] + deepstack_feature_lists, dim=1)
# (Σpatch/4, 3584) + 3×(Σpatch/4, 3584) → (Σpatch/4, 3584×4=14336)
```

把主 merger 输出和 3 个 deepstack 层（第 8/16/24 层）的输出沿 **dim 1（特征维）拼接**。这正好对上 Q13 里 `out_hidden_size = 3584 × (1+3) = 14336` 的预留——视觉塔最终输出不是 3584 维，而是"主特征 + 3 份中间特征"拼成的 14336 维。

> deepstack 各层输出在 merge 后维度都已是 3584（都过了各自的 merger），所以能直接和主特征在特征维拼接。拼接后不是 4 个独立 token 流，而是每个视觉 token 的"特征向量"变长了——LLM 侧会再把它拆回 4 份，分别喂给不同的层（见 Q8 的 DeepStack 注入）。

### 9. 完整数据流一览

| 步骤 | 位置 | 形状变化 | 做了什么 |
|---|---|---|---|
| 输入 | :833 | `(Σpatch, 1536)` | 打包的像素 |
| ① 搬设备 | :840 | `(Σpatch, 1536)` | CPU/fp32 → GPU/bf16 |
| ② patch_embed | :841 | → `(Σpatch, 1152)` | Conv3d：像素→特征 |
| ③ 算辅料 | :843 | — | 用 grid_thw 算 pos/rotary/cu_seqlens |
| ④ 加位置 | :850-852 | `(Σpatch, 1, 1152)` | +绝对位置，加 batch 维 |
| ⑤ 27×block | :855 | `(Σpatch, 1, 1152)` | 双向注意力+MLP，抽 deepstack |
| ⑥ merger | :870 | → `(Σpatch/4, 3584)` | 2×2 merge，投影到 LLM 维度 |
| ⑦ 拼 deepstack | :871-873 | → `(Σpatch/4, 14336)` | 主特征 + 3 份中间特征 |

### 总结：三个层次的"为什么"

1. **为什么有这些步骤**：像素 →（patch_embed）→ 特征 →（加位置）→ 有空间语义的特征 →（双向注意力堆叠）→ 深度融合的特征 →（merge）→ 视觉 token →（拼 deepstack）→ 给 LLM 的多尺度视觉表示。
2. **为什么打包 + cu_seqlens**：动态分辨率下多图无法规整批处理，借 varlen 注意力"拼一条 + 划边界"，省算力又不串图。
3. **为什么位置编码要两套**：绝对 pos_embed 给全局定位，2D RoPE 给注意力精确的相对位置（h、w 两轴），二者互补。

### 附：Conv3d 的 kernel=stride 为什么等价于"切块 + 线性投影"

> 这是 Q14 第 ② 步里"数学上等价于切块+线性投影，但更紧凑"一句的展开。核心结论：**卷积核的"滑动取块做内积"和"把块 flatten 后乘一个矩阵"是同一个运算的两种写法**，区别只在数据怎么排布、谁来帮你优化。

#### 1. "线性投影"在算什么

一个 patch 是 `3 通道 × 2 帧 × 16×16 = 1536` 个数。线性投影把它当成一个 1536 维列向量 `x` **1536 维列向量**：形状 = [1536,1]，用权重矩阵 `W ∈ R^(1152×1536)` 乘，得到 1152 维输出 `y`：


$$y[j] = Σ_i  W[j, i] · x[i]$$        # 第 j 维 = W 第 j 行与 x 逐元素乘加


即 `y = W·x + b`，第 `j` 个输出 = 权重表第 `j` 行（1536 个系数）与 patch 的 1536 个数对应相乘再求和。

#### 2. "Conv3d"在算什么

Conv3d 权重是 `K ∈ R^(1152 × 3 × 2 × 16 × 16)`（输出通道 × 输入通道 × 时间 × 高 × 宽）。第 `o` 个输出通道在某窗口位置算：

$$y[o] = b[o] + Σ_c Σ_t Σ_h Σ_w  K[o, c, t, h, w] · X[c, t, h, w]$$

即：**把卷积核 K[o]（1536 个系数）和窗口覆盖的输入块（1536 个值）逐元素相乘再求和**。

#### 3. 逐项对应——是同一个运算

| 线性投影 | Conv3d | 对应关系 |
|---|---|---|
| `W` 第 `j` 行（1536 个数） | `K` 第 `j` 个输出通道（1536 个数） | 一样，只是排成 `(3,2,16,16)` 而非一条 1536 的线 |
| `x`（flatten 后的 patch） | 卷积窗口覆盖的输入块 | 同一块，只是保持 `(3,2,16,16)` 形状 |
| `y[j]` | 卷积输出的第 `j` 通道 | 同一个数 |
| 参数 `1152×1536` | `1152×3×2×16×16` | **完全相同**（`3×2×16×16=1536`） |

所以两者**计算量、参数量完全相同，一个不省**。差别只在：线性层要求先把 patch flatten 成一条线，卷积则让数据保持 3D 形状、由卷积核自己"对齐"。

#### 4. "stride = kernel" 凭什么等价于"切块"

卷积的 **stride（步长）** 决定相邻两次窗口移动多少：

- **stride < kernel**：窗口重叠，相邻 patch 共享像素——这是"滑窗"，不是切块；
- **stride = kernel**：窗口刚好一个挨一个，相邻 patch 不重叠、不遗漏，恰好铺满整张图——这才是"切块"。

所以 `kernel = stride = (2, 16, 16)` 意味着：卷积核每次正好跨过一整个 tube（2 帧 × 16×16），下一个窗口从下一个 tube 开始，绝不回头、绝不重复。**"把图切成互不相交的块"由 stride 自动保证，无需手写切片代码。**

#### 5. "更紧凑"到底省了什么

**不是省参数、不是省计算**（第 3 点说了两边一样）。省的是：

1. **代码表达**：一行 `nn.Conv3d(3, 1152, kernel=(2,16,16), stride=(2,16,16))` 同时表达"切块 + 投影"；朴素写法要切片 → flatten → 线性层三步。
2. **中间张量**：朴素写法要把所有 patch 抽出堆成 `(n_patches, 1536)` 大矩阵再乘；卷积在 kernel 内部"边取块边内积"，不显式 materialize 中间矩阵。
3. **底层优化**：框架的卷积 kernel 深度优化过（im2col+GEMM / Winograd / direct conv），比手写 `view` + `matmul` 更快、缓存更友好。

#### 6. 回到 vLLM 代码——这里 Conv3d 其实只做了"投影"

```python
x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)  # (L, 3, 2, 16, 16)
x = self.proj(x).view(L, self.hidden_size)                                     # Conv3d
```

**输入 x 已被 processor 切好、flatten 成 `(L, 1536)`**（"切块"在预处理阶段就完成了）。`.view` 只把每个 1536 维行还原成 `(3,2,16,16)`，Conv3d 的 kernel=stride 恰好等于整个小图尺寸，所以**每个 patch 只产出 1 个输出**。因此在这段代码里，Conv3d 扮演的**纯粹是"线性投影"**（切块已在 processor 完成）。

而"Conv3d 既能切块又能投影"这句更一般的话，针对的是"喂进去的是整张图"的场景——那时 stride=kernel 的非重叠滑动才同时承担了切块。

**一句话**：Conv3d 的 kernel=stride 就是"用卷积这种高效写法，对每个互不重叠的 patch 做一次线性投影"——数学上与 `flatten + 全连接` 完全等价，省的是代码、中间张量和能否用上优化内核，而不是参数或计算量。

### 附：blk(...) 一行做了什么

> 这是 Q14 第 ⑤ 步里 `for layer_num, blk in enumerate(self.blocks): hidden_states = blk(...)` 一行的展开。核心结论：**这一行是整座视觉塔的"本体"——forward 其余 6 步只是搬设备、加位置、降采样的包装，真正把像素特征"炼"成视觉语义的 54 次计算（27 层 × 2 子层）全藏在这一个循环里。**

#### 1. `self.blocks` 是什么

`self.blocks` 是 `nn.ModuleList`（:651-664），装着 `depth = 27` 个**结构完全相同、参数各自独立**的 `Qwen3_VisionBlock`：

```python
self.blocks = nn.ModuleList(
    [Qwen3_VisionBlock(dim=..., num_heads=..., mlp_hidden_dim=..., ...)
     for layer_idx in range(vision_config.depth)]   # depth = 27
)
```

`ModuleList` 是 PyTorch 专门用来装"一串子模块"的容器，它本身不做任何计算，只负责让这些子模块能被 `.to()`、`.parameters()`、权重加载等批量管理——迭代它，拿到的就是一个个真正的 block。

#### 2. `enumerate` 拆出来的两个东西

```python
for layer_num, blk in enumerate(self.blocks):
```

`enumerate` 每次吐出 `(层号, 块)` 两样：

- `layer_num`：`0, 1, ..., 26`，只用来判断"这一层要不要抽 deepstack 特征"（`if layer_num in self.deepstack_visual_indexes`，即第 8/16/24 层）；
- `blk`：第 `layer_num` 个 `Qwen3_VisionBlock` 实例——就是接下来要被"调用"的那个块。

#### 3. `blk(...)` 一行到底执行了什么

`blk` 是 `nn.Module`，`blk(hidden_states, ...)` 不是普通函数调用，而是走 PyTorch 的 `__call__` → `forward` 约定。展开后（:479-498）就是两段"残差 + 子层"：

```python
x = x + self.attn(self.norm1(x), cu_seqlens=..., rotary_pos_emb_cos=..., ...)  # ① 注意力子层
x = x + self.mlp(self.norm2(x))                                               # ② 前馈子层
```

五个形参里，`hidden_states`（即 `x`）是真正被"加工"的数据，其余四个（`cu_seqlens`、`rotary_pos_emb_cos/sin`、`max_seqlen`、`sequence_lengths`）是第 ③ 步算好的**环境辅料**，只有注意力子层消费它们（划边界、加 RoPE），block 的 `mlp` 不碰这些元数据。

#### 4. 为什么"一行"就写完了 27 层

关键在 `hidden_states = ...` 的**反复覆盖**：第 0 层吃进输入、吐出特征；第 1 层吃进第 0 层的输出、再吐一次……`hidden_states` 变量名不变，内容每轮被替换一次。循环只写一行，展开却是 `27 层 × 2 子层 = 54` 次前向计算——这就是 Transformer "深度"的由来：**同样的块结构重复堆叠、靠残差让信息层层提炼，而不是一个巨大的单层一次算完**。

也正因如此，forward 的 7 步里其余 6 步（搬设备、patch_embed、算辅料、加位置、merger、拼 deepstack）都是在为这个循环"备料"或"收尾"：备好 `(Σpatch, 1, 1152)` 的序列和 5 样辅料 → 交给 27 个 block 循环加工 → 取出最后输出再过 merger 降采样。

**一句话**：`blk(...)` 这一行 = 把 hidden_states 依次穿过 27 个 `Qwen3_VisionBlock`，每个块各做一次"注意力残差 + MLP 残差"——ViT 的所有"思考"都发生在这个循环里。

### 附：deepstack 中间层抽取那几行在做什么

> 这是 Q14 第 ⑤ 步里 `if layer_num in self.deepstack_visual_indexes:` 那几行（:873-878）的展开。核心结论：**在 27 层视觉塔循环里，除了最后一层交给主 merger，还额外把第 8/16/24 三层的输出「快照」一份、各过一个小 merger，收集起来供最后拼进视觉塔输出——这就是 Qwen3-VL 的 deepstack 多级特征复用。**

#### 1. 逐行拆解

```python
if layer_num in self.deepstack_visual_indexes:                     # ① :873 只有第 8/16/24 层触发
    deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)  # ② :874 层号 → 列表下标
    deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](
        hidden_states
    )                                                             # ③ :875-877 过对应 merger
    deepstack_feature_lists.append(deepstack_feature)              # ④ :878 收进列表
```

**① 触发条件（:873）**：`layer_num` 是 `0..26`（27 层），`deepstack_visual_indexes = (8, 16, 24)`，所以只有走到第 8/16/24 层才进分支，其余层只做普通 `blk(...)` 加工后继续往下传。

**② `.index(layer_num)`（:874）——一个关键细节**：`deepstack_merger_list` 只有 3 个元素（每个 deepstack 层配一个），下标是 `0/1/2` 而不是 `8/16/24`，所以要把层号映射成列表位置：

| `layer_num` | `.index()` | 用哪个 merger |
|---|---|---|
| 8 | 0 | `deepstack_merger_list[0]` |
| 16 | 1 | `deepstack_merger_list[1]` |
| 24 | 2 | `deepstack_merger_list[2]` |

`tuple.index(x)` 返回 `x` 首次出现的位置；`if` 已保证 `layer_num` 一定在里面，这里必然得到 `0/1/2` 之一。

**③ 过 merger（:875-877）**：`deepstack_merger_list` 里也是 `Qwen3_VisionPatchMerger`（:631-644），做和主 merger 一样的事——2×2 merge 把 patch 数降到 1/4、再投影到 LLM 维度 3584。唯一区别是构造时传了 `use_postshuffle_norm=True`（:637）：它是「先拼成 4608 维、再对合并后的向量做 LayerNorm」，主 merger 则「先各自归一化、再拼」。输出 `(Σpatch/4, 3584)`，与主 merger 同形。

**④ 收集（:878）**：只是存进列表，`hidden_states` 本身不被改动——它继续作为下一层输入往前走，相当于「边往下传边在旁边复印一份存档」。

#### 2. 这 3 份特征最后去哪了

循环结束后第 ⑦ 步（:880-882）拼回主特征：

```python
hidden_states = torch.cat([hidden_states] + deepstack_feature_lists, dim=1)
# (Σpatch/4, 3584) + 3×(Σpatch/4, 3584) → (Σpatch/4, 14336)
```

视觉塔最终输出 `14336 = 3584 × 4` 维（主特征 + 第 8/16/24 层各一份），LLM 侧再拆回 4 份、分别注入不同深度的层（见 Q8 DeepStack 注入）。

#### 3. 为什么抽中间层（DeepStack 的原理）

只喂最后一层，LLM 只能拿到视觉塔最深的**语义级**特征；而浅层学的是**低级特征**（边缘、纹理），深层才是语义。DeepStack 同时抽几层，等于给 LLM 提供**多尺度/多分辨率**的视觉信息——让 LLM 浅层也能拿到视觉塔的底层细节，而不是只能等最高层抽象好的结果。

**一句话**：这几行在 27 层循环的第 8/16/24 层各复印一份当前特征，用各自的小 merger 做 merge+投影后存进列表，最终与主特征拼成 14336 维输出——Qwen3-VL 的 deepstack 多级特征复用。

### 附：torch.cat(...) 拼出的 14336 维做了什么、后续怎么被使用

> 这是 Q14 第 ⑦ 步 `hidden_states = torch.cat([hidden_states] + deepstack_feature_lists, dim=1)`（:880-882）的展开。核心结论：**这一步把主特征和 3 份 deepstack 中间特征沿特征维拼成一条 14336 维输出；到 LLM 侧再按约定拆回两半——前半走正常 scatter、后半走 deepstack 注入。**

#### 1. 这一步在做什么

```python
hidden_states = self.merger(hidden_states)   # :879  → (Σpatch/4, 3584)
hidden_states = torch.cat(
    [hidden_states] + deepstack_feature_lists, dim=1
)  # :880-882  → (Σpatch/4, 14336)
```

`[hidden_states] + deepstack_feature_lists` 是 `[主特征, ds_8, ds_16, ds_24]` 四段，每段都已是 `(Σpatch/4, 3584)`；`dim=1` 沿**特征维**拼接——token 数不变，每个 token 的特征向量变长 4 倍。

#### 2. 得到的 hidden_states 是什么

形状 `(Σpatch/4, 14336)`，每行一个视觉 token，其 14336 维按顺序是 4 段：

| 维段 | 内容 | 来源 |
|---|---|---|
| `[:, 0:3584]` | **主特征** | 第 26 层（最后一层）过主 merger |
| `[:, 3584:7168]` | deepstack level 0 | 第 8 层过 `deepstack_merger_list[0]` |
| `[:, 7168:10752]` | deepstack level 1 | 第 16 层过 `deepstack_merger_list[1]` |
| `[:, 10752:14336]` | deepstack level 2 | 第 24 层过 `deepstack_merger_list[2]` |

顺序是**主特征在前、deepstack 按层号升序在后**（`deepstack_feature_lists` 在循环里按 8→16→24 依次 append）。

#### 3. 后续怎样被使用

这个 14336 维不是喂给 LLM 的最终形态。它先作为视觉塔输出返回，经 `_process_image_input`（:2254-2259）按图 `split` 后进入多模态嵌入，再在 `_compute_deepstack_embeds`（:2860-2899）里**拆回两份**：

```python
(multimodal_embeddings_main, multimodal_embeddings_multiscale) = torch.split(
    multimodal_embeddings_cat, [self.visual_dim, self.multiscale_dim], dim=-1
)   # visual_dim=3584, multiscale_dim=10752
```

于是分两路走：

**① 主特征（前 3584 维）→ 正常 scatter 进文本序列**：`multimodal_embeddings_main` 就是普通视觉 token 嵌入，被 `_merge_multimodal_embeddings` 按 `<|image_pad|>` 占位符位置灌进 `inputs_embeds`（同 Q8 的 `masked_scatter`）。

**② 多尺度特征（后 10752 维）→ 注入 LLM 浅层**：`multimodal_embeddings_multiscale` 被 reshape 成 `(num_level=3, L, 3584)`（:2894-2897），作为 `deepstack_input_embeds` 存进 buffer，最后传给 `Qwen3LLMModel.forward`（:1651, :1674-1680）：

```python
if deepstack_input_embeds is not None and layer_idx in range(0, len(deepstack_input_embeds)):
    hidden_states = hidden_states + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
```

`len(deepstack_input_embeds) = 3`，所以 `layer_idx ∈ {0, 1, 2}`——**三份 deepstack 特征分别加到 LLM 第 0、1、2 层输出上**：

| deepstack 来源（ViT 层） | 注入到 LLM 层 |
|---|---|
| 第 8 层 | 第 0 层 |
| 第 16 层 | 第 1 层 |
| 第 24 层 | 第 2 层 |

**一句话**：

```
torch.cat(dim=1) 把 4 段 3584 维拼成 14336 维
   ↓ 进 encoder cache / multimodal_embeddings
_compute_deepstack_embeds 再拆回两半:
   ├─ 前 3584（主特征）──► scatter 进 <|image_pad|> 占位符
   └─ 后 10752（3 级多尺度）──► reshape (3,L,3584) ──► 逐层加到 LLM 前 3 层
```

即：**用一个张量把「最终视觉 token」和「给 LLM 浅层的 3 份中间视觉提示」打包成一条，省去多路传递**——到 LLM 侧再拆开，主特征走正常 scatter、多尺度走 deepstack 注入。

### 附：从 14336 维到 Qwen3LLMModel 的完整调用链路

> 这是上一节「14336 维后续怎么被使用」的完整版：把 `Qwen3_VisionTransformer.forward` 产出 hidden_states → `_compute_deepstack_embeds` 拆两路 → `Qwen3LLMModel` 注入前 3 层整条链路串起来。核心结论：**主特征走「文本嵌入」正常路径，多尺度特征走「buffer 旁路」——两者在 `_compute_deepstack_embeds` 处从同一 14336 维里拆开，在 `Qwen3LLMModel` 前 3 层重新汇合。**

#### 全景链路图

```
┌─ Phase A：编码器（GPUModelRunner 驱动，视觉塔前向，产出 14336 维）──────────────┐
│ GPUModelRunner._execute_mm_encoder()                                             │
│   └─ Qwen3VLForConditionalGeneration.embed_multimodal(**mm_kwargs)   :2830       │
│        └─ _process_image_input(image_input)                          :2239       │
│             ├─ self.visual(pixel_values, grid_thw)                   :2254       │
│             │    └─ Qwen3_VisionTransformer.forward                  :842        │
│             │         └─ return hidden_states  (Σpatch/4, 14336)                │
│             └─ image_embeds.split(sizes)                             :2259       │
│                  → tuple[(n₁,14336), (n₂,14336), …]（按图切段）                 │
│        └─ 返回 multimodal_embeddings（tuple）→ 存入 encoder_cache               │
└──────────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Phase B：收集 + 拆两路（_compute_deepstack_embeds）─────────────────────────────┐
│ GPUModelRunner._gather_mm_embeddings() → 从 encoder_cache 取出                   │
│   └─ Qwen3VLForConditionalGeneration.embed_input_ids(                            │
│         input_ids, multimodal_embeddings=..., is_multimodal=...)  :2901          │
│        ├─ _embed_text_input_ids(...)                                  :2908     │
│        └─ _compute_deepstack_embeds(inputs_embeds, mm_embeddings, ...):2860     │
│             ├─ torch.cat(mm_embeddings)          → (Σnᵢ, 14336)                 │
│             ├─ torch.split(..., [3584, 10752], dim=-1)             :2869         │
│             │     ├─ multimodal_embeddings_main        (Σnᵢ, 3584)              │
│             │     └─ multimodal_embeddings_multiscale  (Σnᵢ, 10752)             │
│             ├─ 主特征 split 回每图 → multimodal_embeddings         :2878         │
│             └─ 多尺度 → reshape (3, L, 3584) → deepstack_input_embeds :2894     │
│        ├─ _merge_multimodal_embeddings(主特征 scatter 进文本)      :2931         │
│        └─ _set_deepstack_input_embeds(多尺度存入 buffer)           :2938         │
│        └─ return inputs_embeds（只含主特征）                                     │
└──────────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Phase C：模型前向（注入 Qwen3LLMModel 前 3 层）────────────────────────────────┐
│ Qwen3VLForConditionalGeneration.forward(...)                      :2942          │
│   ├─ _get_deepstack_input_embeds(n_tokens) → IntermediateTensors  :2978         │
│   └─ self.language_model.model(                                      :2984      │
│         input_ids, positions, ..., deepstack_input_embeds=...)                   │
│        └─ Qwen3LLMModel.forward                                    :1644         │
│             └─ for layer_idx, layer in enumerate(layers):                        │
│                  hidden_states, residual = layer(...)                            │
│                  if deepstack_input_embeds is not None and                       │
│                     layer_idx in range(0, 3):                     :1674          │
│                      hidden_states += deepstack_input_embeds[                   │
│                          f"deepstack_input_embeds_{layer_idx}"]  :1679           │
│   └─ _clear_deepstack_input_embeds(...)                            :2994          │
└──────────────────────────────────────────────────────────────────────────────────┘
```

#### Phase A：视觉塔前向 → 14336 维（:842 → :2254 → :2259）

1. **`GPUModelRunner._execute_mm_encoder`** 调用模型的 `embed_multimodal(**mm_kwargs)`（vLLM v1 的多模态编码入口）。
2. **`embed_multimodal`**（:2830）解析输入，图像走 `_process_image_input`（:2844）。
3. **`_process_image_input`**（:2239）里 `self.visual(pixel_values, grid_thw)`（:2254）就是 `Qwen3_VisionTransformer.forward`，返回 `(Σpatch/4, 14336)`。
4. `image_embeds.split(sizes)`（:2259，`sizes = grid_thw.prod(-1)//4`）把打包序列按图切成 `tuple[(n₁,14336), (n₂,14336), …]`。
5. 结果作为 `multimodal_embeddings`（tuple）返回，被 `GPUModelRunner` 缓存进 `encoder_cache`（key 是 mm_hash）。

> 关键：此时**还没拆主/多尺度**，每张图的 embedding 仍是完整 14336 维。

#### Phase B：`_compute_deepstack_embeds` 拆两路（:2860）

`embed_input_ids`（:2901）在 `_gather_mm_embeddings` 之后被调用，先 `_embed_text_input_ids` 得到文本嵌入，然后（`use_deepstack` 时）调 `_compute_deepstack_embeds`（:2919-2927）：

```python
multimodal_embeddings_cat = torch.cat(multimodal_embeddings, dim=0)          # (Σnᵢ, 14336)
(multimodal_embeddings_main, multimodal_embeddings_multiscale) = \
    torch.split(multimodal_embeddings_cat, [self.visual_dim, self.multiscale_dim], dim=-1)
# visual_dim=3584, multiscale_dim=10752
```

- **`multimodal_embeddings_main`（前 3584）**：`torch.split(..., visual_lens)` 切回每图 → 作为「真正的视觉 token」交给 `_merge_multimodal_embeddings`（:2931）scatter 进 `inputs_embeds` 的 `<|image_pad|>` 位置。
- **`multimodal_embeddings_multiscale`（后 10752）**：经 `_merge_multimodal_embeddings` 摆到占位位置后 `view(L, 3, 3584).permute(1,0,2)`（:2894-2897）→ `(3, L, 3584)`，作为 `deepstack_input_embeds` 存入 buffer（`_set_deepstack_input_embeds` :2938）。

最终 `embed_input_ids` 返回的 `inputs_embeds` **只含主特征**；多尺度部分走了 buffer 旁路。

#### Phase C：`Qwen3LLMModel.forward` 注入前 3 层（:1644）

`Qwen3VLForConditionalGeneration.forward`（:2942）先 `_get_deepstack_input_embeds`（:2978）从 buffer 取出多尺度嵌入（`IntermediateTensors`，key 为 `deepstack_input_embeds_0/1/2`），再传给 `self.language_model.model(...)`（:2984）。

`Qwen3LLMModel.forward`（:1644）在层循环里（:1665-1683）：

```python
for layer_idx, layer in islice(enumerate(self.layers), self.start_layer, self.end_layer):
    hidden_states, residual = layer(positions, hidden_states, residual)
    if deepstack_input_embeds is not None and layer_idx in range(0, len(deepstack_input_embeds)):
        hidden_states = hidden_states + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
```

`len(deepstack_input_embeds) = 3`，故 `layer_idx ∈ {0,1,2}`：

| deepstack 来源（ViT 层） | 注入的 LLM 层 |
|---|---|
| 第 8 层 | 第 0 层 |
| 第 16 层 | 第 1 层 |
| 第 24 层 | 第 2 层 |

用完后 `_clear_deepstack_input_embeds`（:2994）清空 buffer。

**一句话总结**：

```
Qwen3_VisionTransformer.forward ──► (Σpatch/4, 14336)
   │ embed_multimodal → _process_image_input → split 按图 → encoder_cache
   ▼
embed_input_ids → _compute_deepstack_embeds
   ├─ 前 3584（主特征）──► _merge_multimodal_embeddings ──► inputs_embeds
   └─ 后 10752（多尺度）──► (3,L,3584) ──► buffer 旁路
   ▼
forward → _get_deepstack_input_embeds → language_model.model(...)
   ▼
Qwen3LLMModel.forward：layer 0/1/2 分别 += deepstack_input_embeds_{0,1,2}
```

核心设计：**主特征走「文本嵌入」正常路径，多尺度特征走「buffer 旁路」**——两者在 `_compute_deepstack_embeds` 处从同一 14336 维里拆开，在 `Qwen3LLMModel` 前 3 层重新汇合。

---

## Q15：Qwen3_VisionTransformer.prepare_encoder_metadata 具体做了哪些事情？

> 源码：`vllm/model_executor/models/qwen3_vl.py:746-831`。它是 forward 第 ③ 步真正干活的函数，也是 Q14 里"算 5 样辅料"的展开。一句话定位：**把 `grid_thw`（形状元数据）翻译成注意力层需要的 5 样"辅料"**。

### 0. 为什么要单独抽出这个方法

看 docstring（:755-758）：

> Shared by the eager forward path, CUDA graph capture, and CUDA graph replay to avoid duplicated implementation.

它被**三条代码路径共用**：普通 eager 前向、CUDA graph 捕获（capture）、CUDA graph 回放（replay）。vLLM 为了省显存/时延，会把视觉编码器"录制"成 CUDA graph 复用，而录制和回放时输入形状必须固定、辅料必须能预分配。把"算辅料"的逻辑抽成唯一入口，保证三条路径行为一致、不重复实现——这是它存在的**根本原因**，理解了它，后面那些 `pad_to`、`override` 参数才好懂。

### 1. 返回值全景：5 个 key

| key | 形状 | 用途 | 谁消费 |
|---|---|---|---|
| `pos_embeds` | `(Σpatch, 1152)` | 绝对位置嵌入（已插值） | forward 里 `hidden_states + pos_embeds` |
| `rotary_pos_emb_cos` | `(Σpatch, 72)` | 2D RoPE 的 cos | 每个 VisionBlock 的注意力 |
| `rotary_pos_emb_sin` | `(Σpatch, 72)` | 2D RoPE 的 sin | 同上 |
| `cu_seqlens` | `(num_seqs+1,)` | 变长注意力边界 | 注意力层（varlen） |
| `sequence_lengths` | `(num_seqs,)` 或 None | 逐序列长度 | 仅 FlashInfer CuDNN 后端 |
| `max_seqlen` | 标量 tensor（CPU） | 最长序列长度 | FlashAttention / Triton 后端 |

前三个是"位置"，后三个是"注意力边界"，逻辑上分两组。

### 2. 输入参数（4 个可选）

- `grid_thw_list`：必填，`list[[t,h,w], ...]`；
- `max_batch_size` / `max_frames_per_batch`：cu_seqlens 的 pad 目标（CUDA graph 用）；
- `max_seqlen_override`：覆盖 max_seqlen（CUDA graph 捕获最坏情况用）；
- `device`：张量放哪个设备，默认 `self.device`。

### 3. 逐项拆解

#### 3.1 pos_embeds —— 绝对位置嵌入（:779）

```python
metadata["pos_embeds"] = self.fast_pos_embed_interpolate(grid_thw_list)
```

就是 Q14 4.1 讲的：对每张图按自己的 `(h,w)` 网格，从 `pos_embed`（48×48 可学习表）**双线性插值**出位置向量。视频（t>1）会把同一份 `(h,w)` 位置向量重复 t 次（`t*h*w` 个）。输出 `(Σpatch, 1152)`，和 patch_embed 的输出等长，可直接相加。

#### 3.2 rotary cos/sin —— 2D RoPE（:780-782）

```python
rotary_cos, rotary_sin = self.rot_pos_emb(grid_thw_list)
```

`rot_pos_emb`（:700-716）做三件事：

1. **生成 2D 位置 id**：`rot_pos_ids(h,w,merge)`（:673，`@lru_cache` 缓存，同尺寸图复用）给每个 patch 一个 `[h_idx, w_idx]` 坐标，顺序是"merge 块优先"（与 patch 排列同构）；视频把同一组坐标 `repeat(t,1)`；
2. **取缓存**：`get_cos_sin(max_grid_size)` 从 RoPE 的预计算表里取出 0~max_grid_size-1 每个位置的 cos/sin；
3. **查表拼接**：`cos[pos_ids].flatten(1)` 得到 `(Σpatch, 72)`——前 36 维按 h 旋转、后 36 维按 w 旋转（2D RoPE，见 Q14 4.2）。

#### 3.3 cu_seqlens —— 打包序列的分隔符（:784-790）

```python
grid_thw_np = np.array(grid_thw_list, dtype=np.int32)      # (n, 3)
patches_per_frame = grid_thw_np[:, 1] * grid_thw_np[:, 2]   # 每帧 h*w 个 patch
cu_seqlens = np.repeat(patches_per_frame, grid_thw_np[:, 0]).cumsum()  # 按 t 展开再累加
cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])  # 前面补 0
```

**关键**：`np.repeat(..., grid_thw[:, 0])` 把一张 `t=2` 的视频展开成 **2 段**（每段 `h*w`）。为什么？因为视觉注意力按**帧**隔离——视频的不同帧之间不互相 attend（时间信息留给位置编码和 LLM 侧），所以一帧 = 一个独立注意力序列。

结果是个 `[0, len1, len1+len2, ...]` 的累积数组，`cu_seqlens[i]~cu_seqlens[i+1]` 就是第 i 段（一图或一帧）的 patch 区间。

#### 3.4 padding —— 为 CUDA graph 固定缓冲区（:792-811）

```python
pad_to = max_frames_per_batch if max_frames_per_batch is not None else max_batch_size
if pad_to is not None:
    num_seqs = len(cu_seqlens) - 1
    if num_seqs < pad_to:
        cu_seqlens = np.concatenate([cu_seqlens, np.full(pad_to - num_seqs, cu_seqlens[-1], ...)])
```

**原理**：CUDA graph 录制后，所有张量的形状/大小就"冻死"了。为了让回放时任何输入都能塞进同一个 graph，录制时就按**最坏情况**（最多多少个序列）把 cu_seqlens pad 满，末尾用 `cu_seqlens[-1]`（总长度）填——多出来的"空序列"长度为 0，注意力算出来是空段，不影响正确性。

`max_frames_per_batch` 优先于 `max_batch_size`：视频一张图就贡献 T 帧（= T 个注意力序列），序列总数会超过 batch 数，所以视频场景用"帧预算"来定 pad 目标更准确。

#### 3.5 sequence_lengths —— FlashInfer 特需（:813-816）

```python
metadata["sequence_lengths"] = MMEncoderAttention.maybe_compute_seq_lens(self.attn_backend, cu_seqlens, device)
```

`maybe_compute_seq_lens`（mm_encoder_attention.py:245）只有后端是 **FLASHINFER** 时才返回非 None，否则直接 `None`。它算 `cu_seqlens[1:] - cu_seqlens[:-1]`（每段长度），并按 bucket pad 成 tensor。**FlashInfer 的 CuDNN 后端需要逐序列长度**，别的后端不需要，所以这里按后端按需生成——省掉无谓的 tensor 分配。

#### 3.6 max_seqlen —— 后端特需 + 为什么留在 CPU（:818-828）

```python
if max_seqlen_override is not None:
    max_seqlen_val = max_seqlen_override
else:
    max_seqlen_val = MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens)
metadata["max_seqlen"] = torch.tensor(max_seqlen_val, dtype=torch.int32)   # 注意：没 .to(device)
```

两个点：

- **`compute_max_seqlen`**（:223）只在 FlashAttention / Triton 等后端才需要，取 `cu_seqlens` 相邻差的最大值（FlashInfer 还要 bucket 到 2 的幂档位）；其他后端返回 0。
- **为什么故意留在 CPU**（:825-827 注释）：注意力 wrapper 要调 `.item()` 把这个标量变成 Python int。如果它放在 GPU 上，`.item()` 会触发一次 **D2H（设备到主机）拷贝**——在 CUDA graph 捕获时，这次拷贝会被"录进图里"成为固定开销。放 CPU 上，捕获时这个标量直接被烤进图，回放零成本。
- **`max_seqlen_override`**：CUDA graph 捕获时为了覆盖回放的最坏情况，直接指定一个上限值，不按当前输入算。

#### 3.7 后端重算 cu_seqlens —— token 数 → 内存偏移（:830-831）

```python
metadata["cu_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
    self.attn_backend, cu_seqlens, self.hidden_size, self.tp_size, device,
    fp8_padded_hidden_size=self.fp8_padded_hidden_size,
)
```

这是最"后端"的一步（mm_encoder_attention.py:267-318）。前面算的 cu_seqlens 单位是 **token 数**，但 FlashInfer 的 varlen 接口要的是**元素/内存偏移**（offset into Q/K/V buffer）。换算逻辑：

- **bf16 路径**：Q/K/V 是交错共享 buffer 的非连续视图，V 相对 Q/K 有 **3× stride**（Q、K、V 三者交错排布）。于是 `scale = hidden_size // tp_size`，Q/K/O 的 cu_seqlens = token × scale，V 的 = token × scale × 3，两条各自 pad 后拼起来。
- **FP8 路径**：Q/K/V 量化后是**三个独立连续张量**，stride 统一（都是 `H × padded_D`），没有 3× 跳变。`scale = fp8_padded_hidden_size // tp_size`，两条相同，所以拼 `[padded, padded]`。
- **其他后端**（FlashAttention/Triton/SDPA）：不需要换算，`async_tensor_h2d` 直接把 token 数的 cu_seqlens 异步搬到设备即可。

### 4. 完整数据流

```
grid_thw_list [[t,h,w], ...]
        │
        ├─► fast_pos_embed_interpolate ──► pos_embeds         (Σpatch, 1152)
        ├─► rot_pos_emb                ──► rotary cos/sin      (Σpatch, 72)×2
        └─► np 计算 cu_seqlens（token 数，视频按帧拆段）
                ├─► pad 到 max_batch/max_frames（CUDA graph 固定缓冲）
                ├─► maybe_compute_seq_lens   ──► sequence_lengths（仅 FlashInfer）
                ├─► compute_max_seqlen       ──► max_seqlen（标量，留 CPU）
                └─► maybe_recompute_cu_seqlens ─► cu_seqlens（后端换算：字节偏移/TP/FP8）
```

### 小结

`prepare_encoder_metadata` 就干一件事：**把"形状元数据" grid_thw 一次性翻译成"位置 + 注意力边界"两套辅料**。它的设计处处体现 serving 的考量——

1. **只依赖形状、不碰像素**：所以能预计算、能被 CUDA graph 复用（三条路径共用一个入口）；
2. **按后端按需生成**：`sequence_lengths`、`max_seqlen`、cu_seqlens 的重算都只在特定后端才做，省掉无谓分配；
3. **pad + override 为图捕获服务**：固定缓冲区大小、覆盖最坏情况，保证回放不越界。

---

## Q16：2D RoPE 的原理

> 本文是 Q14 4.2「2D RoPE」一段的完整展开。先纠正一个说法：**"head 劈成前 36 维（h）、后 36 维（w）"只是高层直觉**，代码里的确切布局见第 4 节。核心结论：1D RoPE 是"给向量一个朝向"，2D RoPE 是"给向量两个朝向"——一半维度按行坐标 h 旋转、一半按列坐标 w 旋转，注意力从而能分别感知"上下"和"左右"的相对位置。

### 0. 一句话

**1D RoPE 是"给向量一个朝向"，2D RoPE 是"给向量两个朝向"**——把注意力头的维度劈成两半，一半用行坐标 h 旋转、一半用列坐标 w 旋转，于是注意力能分别感知"上下"和"左右"的相对位置。

### 1. 先回顾 1D RoPE（2D 是它的推广）

RoPE 不"加"位置向量，而是"旋转" q/k 向量。把相邻两维 `(x₀, x₁)` 看成一个 2D 平面上的点（等价于复数 `x₀ + i·x₁`），按位置 `m` 旋转角度 `mθ`：

```
[x₀']   [cos(mθ)  -sin(mθ)] [x₀]
[x₁'] = [sin(mθ)   cos(mθ)] [x₁]
```

复数形式更简洁：**乘以 $e^{i·mθ}$**。

**RoPE 的核心魔法**：位置 m 的 q 转了 `mθ`，位置 n 的 k 转了 `nθ`，它们点积时：

$$(q·e^{imθ}) · (k·e^{inθ})ᶜᵒⁿʲ = q·k*·e^{i(m-n)θ}$$

取实部后**只依赖 `(m-n)`**——相对位置，与绝对位置无关。所以模型不需要显式学"位置 5 和位置 8 差 3"，旋转相位天然携带了相对关系。

> 不同维度对用不同频率 $θ_j = base^(-2j/d)$：高频转得快、捕捉近距离，低频转得慢、捕捉远距离，像傅里叶变换把不同尺度的相对位置铺开。

### 2. 为什么 1D 不够：图像位置是二维的

文本 token 的位置是一个标量 `m`。但图像 patch 的位置是 `(h, w)` 两个坐标——"第 3 行第 5 列"。1D RoPE 只有一个"旋转角"，装不下两个独立的坐标。

三个候选方案及各自问题：

| 方案 | 问题 |
|---|---|
| 把 2D 网格拍平成一维序号再套 1D RoPE | 丢失空间结构：`(h,w)` 和 `(h+1,w)` 的 1D 距离是"宽度 W"，而 `(h,w)` 和 `(h,w+1)` 距离是 1——上下相邻和左右相邻被赋予完全不同的"距离"，不自然 |
| 给每个 `(h,w)` 学一个绝对位置向量 | 表随分辨率爆炸，难泛化到 smart_resize 出来的任意 `(h,w)` 网格 |
| **2D RoPE** | 把两个坐标各自交给一半维度，天然分离行/列相对信息 ✅ |

### 3. 2D RoPE 的做法：head 劈两半，各管一轴

把注意力头的向量 `x`（72 维）分成两半：

```
x = [ x^h (前 36 维) | x^w (后 36 维) ]
```

对位置 `(h, w)`：

$$x^h → x^h · e^{i·h·θ}$$      # 前一半用「行坐标 h」旋转
$$x^w → x^w · e^{i·w·θ}$$      # 后一半用「列坐标 w」旋转


于是两个 patch `(h₁,w₁)` 和 `(h₂,w₂)` 的点积：
$$x^h 部分 → 只依赖 (h₁ - h₂) = Δh   ← 行方向相对位置$$
$$x^w 部分 → 只依赖 (w₁ - w₂) = Δw   ← 列方向相对位置$$

**注意力权重里，一半维度在回答"我们隔几行"，另一半在回答"我们隔几列"**。这就是 2D RoPE 的完整图景——本质上是**两个独立的 1D RoPE，分别在 h 轴和 w 轴上叠加**。

### 4. 代码里具体怎么实现（Qwen3-VL 视觉塔）

对应 `qwen3_vl.py:614-619` 和 `:700-716`：

1. **`get_rope(head_size=72, partial_rotary_factor=0.5)`** → `rotary_dim = 72 × 0.5 = 36`（`__init__.py:72`）。即每个 head 里参与旋转的"容量"是 36 维（`partial_rotary_factor` 决定让多少维度承载位置信息）。

2. **`rotary_dim=36` 生成 `36/2 = 18` 个频率**（`base.py:89` 的 `arange(0, rotary_dim, 2)`）。每个频率旋转一对维度（一个 2D 平面），18 个频率覆盖 36 维。

3. **`rot_pos_ids(h, w, merge)`**（:673，`@lru_cache`）给每个 patch 生成 `(h_idx, w_idx)` 二维坐标，顺序是"merge 块优先"（与 patch 排列对齐）。

4. **`rot_pos_emb`**（:700-716）里的关键一步：

```python
cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)  # 各 [max_grid_size, 18]
pos_ids = ...                                              # [n_patch, 2] = (h, w)
cos_combined = cos[pos_ids].flatten(1)                     # [n_patch, 36] = [h 的 cos(18) | w 的 cos(18)]
```

`cos[pos_ids]` 按 `(h, w)` 两个坐标**各查一次表**，得到 `[n_patch, 2, 18]`；`flatten(1)` 把"h 的 18 个 cos"和"w 的 18 个 cos"前后拼起来，得到 `[n_patch, 36]`。**这个"前后拼接"就是 2D 的核心动作**——前 18 个频率对应 h 轴、后 18 个频率对应 w 轴。

5. **neox 布局配对旋转**（`ApplyRotaryEmb`，common.py:169-179）：把 72 维 head 按 `(dim d, dim d+36)` 两两配对成 36 个旋转平面，用上面的 cos/sin 逐平面旋转。于是前 18 个平面按 h 旋转、后 18 个平面按 w 旋转，72 维 head 全部被覆盖。

> 注意一个容易踩的坑：`partial_rotary_factor=0.5` 单独看是"只转一半维度"（1D RoPE 场景下会留一半不转），但**配合 2D 就恰好满配**——h、w 两轴各分到 18 个频率（36 维），加起来正好 72 维，没有"不旋转"的剩余。这也是为什么视觉塔要设 0.5：它要为 h、w 两个轴各留一半。

### 5. 只旋转 q/k，不旋转 v

在 `Qwen2_5_VisionAttention.forward`（qwen2_5_vl.py:420-441）：

```python
qk, v = qkv[:, :, :2], qkv[:, :, 2]     # 拆出 q/k 和 v
qk_rotated = self.apply_rotary_emb(...)  # 只对 q/k 旋转
```

**value 不旋转**。原因：位置信息只需要通过注意力权重（`q·k` 决定"该看谁、看多重"）进入模型，`v` 本身是"被看的那个 patch 的内容"，与它自己的位置无关——旋转 `v` 只会无谓地扰动内容表示。

### 6. 一句话总结 + 直觉

- **1D RoPE**：给向量一个"旋转角"，夹角差 = 相对位置。
- **2D RoPE**：给向量两个"旋转角"（h 角 + w 角），各自占一半维度，注意力同时读出 Δh 和 Δw。
- **直觉类比**：1D RoPE 像"给每个座位一个方向"，2D RoPE 像"给每个座位一个经纬度方向"——模型能同时判断"你在第几排"和"你在第几列"。
