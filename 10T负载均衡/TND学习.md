# TND 学习

> 本文结合代码库（`C:\Code\train`，含 `MindSpeed`、`MindSpeed-MM`、`Megatron-LM` 三个子仓库）整理，回答两个问题：什么是 TND Attention，以及代码里都用它做了什么。

## 目录

- [什么是 TND Attention](#什么是-tnd-attention)
- [代码里都用 TND 做了什么](#代码里都用-tnd-做了什么)
  - [1. 序列打包 / SeqPack（变长序列训练）](#1-序列打包--seqpack变长序列训练)
  - [2. 融合注意力算子入口（区分 TND / SBH 两条路径）](#2-融合注意力算子入口区分-tnd--sbh-两条路径)
  - [3. Ring Attention 上下文并行（TND 布局下的 online-softmax 合并）](#3-ring-attention-上下文并行tnd-布局下的-online-softmax-合并)
  - [4. Hamilton Attention、Ulysses CP、KV-allgather CP 等也支持 TND](#4-hamilton-attentionulysses-cpkv-allgather-cp-等也支持-tnd)
  - [5. 多模态模型（MindSpeed-MM）的 ViT / LLM 配置](#5-多模态模型mindspeed-mm的-vit--llm-配置)
  - [6. 配套工具函数与底层算子](#6-配套工具函数与底层算子)
- [小结](#小结)

---

## 什么是 TND Attention

**TND 不是一种新的注意力算法，而是一种 Q/K/V 张量的数据排布格式（layout）**，用于昇腾 NPU 上的融合注意力算子（`npu_fusion_attention`）和上下文并行（CP）实现。三个字母分别代表张量的三个维度：

- **T** = Total tokens —— 所有样本拼接后的**总 token 数**（`t = b * s`），即把 batch 维度折叠进 sequence 维度；
- **N** = number of attention heads（注意力头数）；
- **D** = head dimension（每个头的维度）。

与之对照，代码里常见的其他 layout 有 `SBH`（S×B×H）、`BSH`、`BSND`、`BNSD`。文档里明确写了 TND 的含义和约束（`MindSpeed/docs/zh/ops/fusion_attention.md:73`）：

> `input_layout`：…支持 `BSH、SBH、BSND、BNSD、TND(actual_seq_qlen/actual_seq_kvlen 需传值)`

也就是说，**TND 本质上是“变长序列（varlen）/ packed 序列”的表示方式**，配合 `actual_seq_qlen` / `actual_seq_kvlen`（即 `cu_seqlens`，各条序列长度的累加和）来记录每条序列的边界。这其实就是 Megatron 里常说的 **THD / varseq 格式**。

代码里给出了精确的转换实现（`MindSpeed/mindspeed/core/context_parallel/utils.py:29-40`）：

```python
# SBH -> TND
def sbh_to_tnd(x, n):
    s, b, h = x.shape
    d, t = h // n, int(b * s)
    return x.transpose(0, 1).view(t, h).view(t, n, d)

# TND -> SBH
def tnd_to_sbh(x, b):
    t, n, d = x.shape
    s, h = t // b, int(n * d)
    return x.view(b, s, n, d).transpose(0, 1).view(s, b, h)
```

在 `MindSpeed-MM/mindspeed_mm/utils/utils.py:525` 里也有同样的映射：`("sbnd", "tnd"): rearrange(x, "s b n d -> (s b) n d")`。

**核心目的**：多条序列长度异构时，把它们首尾拼接成一条连续的 token 流，去掉 batch 维度，从而避免按 batch 内最长序列做 padding 造成的显存浪费，并均衡各卡间的负载。

## 代码里都用 TND 做了什么

### 1. 序列打包 / SeqPack（变长序列训练）

`MindSpeed-MM/docs/zh/features/seqpack.md` 明确说明了动机和用法：

> 将多条序列拼接成近似于 `max-seq-len` 的长度，并将拼接后的数据作为一个批次输入模型，**模型以 TND 的 layout 模式处理拼接后的数据**…每张卡上的 token 总数一致，在节约 padding 的显存的同时，均衡卡间数据负载。

即多模态大模型训练中，图像/文本 token 长度差异大，传统方式要 padding 到最长，浪费显存；改成 TND + `cu_seqlens` 后做 varlen attention。

### 2. 融合注意力算子入口（区分 TND / SBH 两条路径）

`MindSpeed/mindspeed/core/context_parallel/dot_product_attention.py:160-178` 是最核心的分支判断：

```python
if packed_seq_params is not None and not is_ulysses_algo:
    # TND
    T, n_head, D = query.shape[0], query.shape[1], query.shape[2]
else:
    seq_length, bsz, n_head, head_dim = ...
...
if packed_seq_params is not None and not is_ulysses_algo:
    cp_size = ...; actual_seq_qlen = packed_seq_params.cu_seqlens_q.tolist()
    actual_seq_kvlen = packed_seq_params.cu_seqlens_kv.tolist()
    shape_order = 'TND'
else:
    query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in ...]
    shape_order = 'SBH'
```

随后把 `shape_order='TND'` 和 `actual_seq_qlen/actual_seq_kvlen` 传给 `npu_fusion_attention`（`dot_product_attention.py:286-301`），NPU 融合算子据此做 varlen 的 flash attention。

### 3. Ring Attention 上下文并行（TND 布局下的 online-softmax 合并）

这是 TND 用得最重的地方。ring attention 里各 rank 只持有自己负责的 token block，Q/K/V 以 TND 布局在环上通信，每算完一个 KV block 就用 `forward_update` 做 online softmax 的 rescale 合并。

- `MindSpeed/mindspeed/core/context_parallel/utils.py:263-297` 的 `tnd_out_update`：专门处理 TND 布局下的 softmax 统计量（max/sum）合并，内部调用 `forward_update(..., layout='TND')`（`utils.py:77-119` 的 `forward_update_without_fused` 里对 TND 用 `flatten_softmax/unflatten_softmax` 处理子序列切分，SBH 则用 `rearrange`）。
- `MindSpeed-MM/mindspeed_mm/fsdp/distributed/context_parallel/ring_context_parallel/ring_context_parallel.py:1240-1314` 定义了 `TNDGeneralAttentionStrategy`，正反向都用 `layout="TND"` 调 `torch_npu.npu_fusion_attention(_grad)`；`1317` 的 `AttentionWithCpTNDGeneral` 是 TND 布局下带 CP 的注意力 `autograd.Function`，在内外环循环里做 KV 的 P2P 收发与分块计算。
- 配套的 `get_selection_indices_for_tnd_softmax_update`（`MindSpeed-MM/.../ring_context_parallel/utils.py:18`）为 TND 布局生成 softmax 合并时需要的索引。

### 4. Hamilton Attention、Ulysses CP、KV-allgather CP 等也支持 TND

- `MindSpeed/mindspeed/te/pytorch/attention/dot_product_attention/hamilton_context_parallel.py`：HAM 注意力支持 TND 格式（`general_output_update_for_ha_of_tnd_format`，`backend.py:73-83` 里 THD 即 `shape_order='TND'`）。
- `ulysses_context_parallel.py:78/97`、`kvallgather_context_parallel.py:460/518/588` 里同样有 `shape_order='TND'` 分支。

### 5. 多模态模型（MindSpeed-MM）的 ViT / LLM 配置

Qwen3-VL、Qwen3-Omni、HunyuanVideo 等模型直接以 `attn_layout: TND` 配置注意力：

- `MindSpeed-MM/examples/qwen3vl/*.yaml`（如 `qwen3vl_full_sft_8B.yaml:133/148`）：`attn_layout: TND`
- `MindSpeed-MM/examples/qwen3omni/model.json`：`"attn_layout": "TND"`
- `MindSpeed-MM/examples/qwen3vl/README.md:214-217`：ViT 和 LLM 的 `flash_attention_2` 均用 `TND`。

### 6. 配套工具函数与底层算子

- 布局转换：`MindSpeed/mindspeed/core/context_parallel/utils.py` 的 `sbh_to_tnd/tnd_to_sbh`；`MindSpeed-MM/mindspeed_mm/utils/utils.py:500-545` 的 `change_tensor_layout`（支持 `sbnd↔tnd`、`tnd↔sbh/bsh`）。
- 底层 flop 统计也识别 TND：`MindSpeed/mindspeed/ops/csrc/flop_counter/flop_counter.cpp:138-140` 里 `input_layer_str == "TND"` 时要求 `actual_seq_qlen/actual_seq_kvlen` 非空，用于按 varlen 精确统计 GQA/MQA 的浮点运算量。
- 测试：`test_ringattn_context_parallel_tnd.py`、`test_hamiltonattention.py` 等专门验证 TND 路径的正确性（用 `sbh_to_tnd` 构造输入再与参考结果对拍）。

## 小结

- **TND Attention** = 以 `[总token数 T, 头数 N, 头维 D]` 布局组织 Q/K/V、并携带 `cu_seqlens`（`actual_seq_qlen/kvlen`）做**变长/打包序列的注意力计算**，主要在昇腾 NPU 的 `npu_fusion_attention` 融合算子上落地。
- 在本代码库中，它被用于：**① SeqPack 变长序列训练（省 padding、负载均衡）；② ring/Hamilton/Ulysses 等上下文并行下的分块 attention 与 online-softmax 合并；③ 多模态模型（Qwen3-VL/Omni 等）ViT 与 LLM 的注意力配置；④ 布局转换工具与 flop 统计**等场景。
