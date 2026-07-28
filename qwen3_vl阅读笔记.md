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
