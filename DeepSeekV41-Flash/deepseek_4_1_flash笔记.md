# DeepSeek-V4.1-Flash 学习笔记

## 目录

- [DeepSeek-V4.1-Flash 学习笔记](#deepseek-v41-flash-学习笔记)
  - [目录](#目录)
  - [问题 1：20-layer causal encoder 是什么](#问题-120-layer-causal-encoder-是什么)
    - [问题原文](#问题原文)
    - [结论](#结论)
    - [1. 20-layer causal encoder 的结构](#1-20-layer-causal-encoder-的结构)
    - [2. Causal Encoder 的主要作用](#2-causal-encoder-的主要作用)
      - [2.1 普通 Decoder-only Transformer](#21-普通-decoder-only-transformer)
      - [2.2 CED 的做法](#22-ced-的做法)
    - [3. 为什么还要保留 Decoder 每层自己的 SWA](#3-为什么还要保留-decoder-每层自己的-swa)
    - [4. CED 为什么能降低 Prefill 计算量](#4-ced-为什么能降低-prefill-计算量)
      - [4.1 100K Prompt 示例](#41-100k-prompt-示例)
    - [5. Decode 时是否也只执行前 20 层](#5-decode-时是否也只执行前-20-层)
    - [6. 为什么它仍然属于 Decoder-only 自回归模型](#6-为什么它仍然属于-decoder-only-自回归模型)
      - [6.1 传统 Seq2Seq Encoder](#61-传统-seq2seq-encoder)
      - [6.2 CED 的 Causal Encoder](#62-ced-的-causal-encoder)
    - [7. 为什么称为 Causal Encoder](#7-为什么称为-causal-encoder)
    - [8. CED 与 YOCO 的关系](#8-ced-与-yoco-的关系)
    - [9. 为什么代码中的层仍叫 DecoderLayer](#9-为什么代码中的层仍叫-decoderlayer)
    - [最终总结](#最终总结)
  - [问题 2：普通 Decoder-only 与 CED 的 Prefill、Decode 计算量](#问题-2普通-decoder-only-与-ced-的-prefilldecode-计算量)
    - [问题 2 原文](#问题-2-原文)
    - [问题 2 结论](#问题-2-结论)
    - [1. 统一符号与计算口径](#1-统一符号与计算口径)
    - [2. 普通模型的 Prefill](#2-普通模型的-prefill)
    - [3. 普通模型的 Decode 与 KV Cache](#3-普通模型的-decode-与-kv-cache)
    - [4. vLLM 代码对应关系](#4-vllm-代码对应关系)
      - [4.1 调度器只调度尚未计算的 token](#41-调度器只调度尚未计算的-token)
      - [4.2 新 token 仍然经过全部层](#42-新-token-仍然经过全部层)
      - [4.3 新 K/V 被追加到 Cache](#43-新-kv-被追加到-cache)
    - [5. 普通模型总计算量](#5-普通模型总计算量)
    - [6. 三角和出现在哪里](#6-三角和出现在哪里)
    - [7. 完全关闭 KV Cache 的计算量](#7-完全关闭-kv-cache-的计算量)
    - [8. Dense Attention 的二次复杂度](#8-dense-attention-的二次复杂度)
    - [9. 论文 CED 的 Prefill 与总计算量](#9-论文-ced-的-prefill-与总计算量)
      - [9.1 加上 1K 输出的 Decode](#91-加上-1k-输出的-decode)
    - [10. 本地 CUDA 与 Ascend 实现差异](#10-本地-cuda-与-ascend-实现差异)
      - [10.1 上游 vLLM CUDA 实现](#101-上游-vllm-cuda-实现)
      - [10.2 当前 vLLM-Ascend 静态代码](#102-当前-vllm-ascend-静态代码)
    - [问题 2 最终总结](#问题-2-最终总结)
  - [问题 3：Backbone、Engram 与每 Token 激活参数](#问题-3backboneengram-与每-token-激活参数)
    - [问题 3 原文](#问题-3-原文)
    - [问题 3 核心结论](#问题-3-核心结论)
    - [1. 什么是 Backbone Parameters](#1-什么是-backbone-parameters)
      - [1.1 552B 为什么可以这么大](#11-552b-为什么可以这么大)
      - [1.2 Backbone Parameters 不等于 Active Parameters](#12-backbone-parameters-不等于-active-parameters)
    - [2. 什么是 Engram Parameters](#2-什么是-engram-parameters)
      - [2.1 196B 是怎么来的](#21-196b-是怎么来的)
      - [2.2 为什么 196B 很大，计算却可以很小](#22-为什么-196b-很大计算却可以很小)
      - [2.3 Engram 与 MoE 的差异](#23-engram-与-moe-的差异)
    - [3. “每个 Token 激活 8B 参数”是什么意思](#3-每个-token-激活-8b-参数是什么意思)
- [P\_{\\text{active/token}}](#p_textactivetoken)
      - [3.1 它不表示什么](#31-它不表示什么)
    - [4. 为什么 Prefill 是 8B，而 Decode 是 16B](#4-为什么-prefill-是-8b而-decode-是-16b)
      - [4.1 Prefill](#41-prefill)
- [C\_{\\text{CED,prefill}}](#c_textcedprefill)
      - [4.2 Decode](#42-decode)
      - [4.3 用一个具体例子理解](#43-用一个具体例子理解)
    - [5. 8B/16B 是否包含 196B Engram](#5-8b16b-是否包含-196b-engram)
    - [6. vLLM 与 vLLM-Ascend 代码对应关系](#6-vllm-与-vllm-ascend-代码对应关系)
      - [6.1 上游 vLLM CUDA 路径](#61-上游-vllm-cuda-路径)
      - [6.2 当前本地 vLLM-Ascend 路径](#62-当前本地-vllm-ascend-路径)
    - [问题 3 最终总结](#问题-3-最终总结)
  - [问题 4：CED 如何从 Encoder 输出构造 Decoder Global KV](#问题-4ced-如何从-encoder-输出构造-decoder-global-kv)
    - [问题 4 原文](#问题-4-原文)
    - [问题 4 核心结论](#问题-4-核心结论)
    - [1. 普通 Decoder-only 为什么不能跳过 Prompt 的 Decoder 层](#1-普通-decoder-only-为什么不能跳过-prompt-的-decoder-层)
    - [2. CED 改变了什么依赖关系](#2-ced-改变了什么依赖关系)
- [K\_l^{\\text{global}},V\_l^{\\text{global}}](#k_ltextglobalv_ltextglobal)
      - [2.1 两个重要限定](#21-两个重要限定)
    - [3. 为什么 Encoder 输出可以让大部分 Prompt 跳过 Decoder](#3-为什么-encoder-输出可以让大部分-prompt-跳过-decoder)
- [K\_{\\text{global}},V\_{\\text{global}}](#k_textglobalv_textglobal)
      - [3.1 实际不是普通 Dense Global Attention](#31-实际不是普通-dense-global-attention)
    - [4. 为什么 Decoder 仍需要每层自己的 SWA KV](#4-为什么-decoder-仍需要每层自己的-swa-kv)
- [K\_l^{\\text{SWA}},V\_l^{\\text{SWA}}](#k_ltextswav_ltextswa)
    - [5. 为什么只需 Replay 最后一个滑窗](#5-为什么只需-replay-最后一个滑窗)
    - [6. 为什么最后 128 个 Token Replay 只是近似](#6-为什么最后-128-个-token-replay-只是近似)
      - [6.1 单层 SWA 的依赖范围](#61-单层-swa-的依赖范围)
      - [6.2 多层以后依赖会扩散](#62-多层以后依赖会扩散)
- [L\_{\\text{dec}}\\times W](#l_textdectimes-w)
- [20\\times128](#20times128)
      - [6.3 Bounded Replay 做了什么截断](#63-bounded-replay-做了什么截断)
    - [7. Retaining Layer-local SWA 不等于持久化保存](#7-retaining-layer-local-swa-不等于持久化保存)
    - [8. 100K Prompt 计算示例](#8-100k-prompt-计算示例)
      - [8.1 普通模型](#81-普通模型)
- [C\_{\\text{normal}}](#c_textnormal)
- [100000\\times40](#100000times40)
      - [8.2 论文理想 CED](#82-论文理想-ced)
- [100000\\times20](#100000times20)
- [128\\times20](#128times20)
- [C\_{\\text{CED}}](#c_textced)
- [2,000,000+2,560](#20000002560)
- [\\frac{2,002,560}{4,000,000}](#frac20025604000000)
      - [8.3 第一 Decode 步能访问什么](#83-第一-decode-步能访问什么)
    - [9. 上游 vLLM 代码对应关系](#9-上游-vllm-代码对应关系)
      - [9.1 KV Source 与 Consumer](#91-kv-source-与-consumer)
      - [9.2 Global KV 与 SWA KV 是独立缓存](#92-global-kv-与-swa-kv-是独立缓存)
      - [9.3 Decoder Replay 只保留末尾窗口行](#93-decoder-replay-只保留末尾窗口行)
      - [9.4 Replay 分界点](#94-replay-分界点)
      - [9.5 Replay Token 不覆盖已经缓存的 Global KV](#95-replay-token-不覆盖已经缓存的-global-kv)
      - [9.6 为什么上游代码不是严格的 $20N+20W$](#96-为什么上游代码不是严格的-20n20w)
- [21\\times100000+19\\times128](#21times10000019times128)
    - [10. vLLM-Ascend 代码对应关系](#10-vllm-ascend-代码对应关系)
      - [10.1 已实现 Global KV Source/Consumer 拓扑](#101-已实现-global-kv-sourceconsumer-拓扑)
      - [10.2 每层仍然有自己的 SWA Cache](#102-每层仍然有自己的-swa-cache)
      - [10.3 Attention 算子同时读取 Local SWA 与 Global KV](#103-attention-算子同时读取-local-swa-与-global-kv)
      - [10.4 当前 Ascend 代码的限制](#104-当前-ascend-代码的限制)
    - [问题 4 最终总结](#问题-4-最终总结)
  - [问题 5：Causal Encoder、CED、Global/Local KV 与 SWA Replay](#问题-5causal-encodercedgloballocal-kv-与-swa-replay)
    - [问题 5 原文](#问题-5-原文)
    - [问题 5 核心结论](#问题-5-核心结论)
    - [5.1 40 层整体结构](#51-40-层整体结构)
    - [5.2 Layer 0～19 与普通 KV 的异同](#52-layer-019-与普通-kv-的异同)
      - [普通 Decoder-only 模型](#普通-decoder-only-模型)
      - [DeepSeek-V4.1-Flash 的双记忆分支](#deepseek-v41-flash-的双记忆分支)
    - [5.3 Causal Encoder 的特殊之处](#53-causal-encoder-的特殊之处)
      - [5.3.1 它仍然是 Causal 的](#531-它仍然是-causal-的)
      - [5.3.2 Layer 0～1 是纯 SWA](#532-layer-01-是纯-swa)
      - [5.3.3 Layer 2～19 使用 ratio=2 压缩](#533-layer-219-使用-ratio2-压缩)
      - [5.3.4 Global KV 不按每层各存一份](#534-global-kv-不按每层各存一份)
    - [5.4 CED 是什么](#54-ced-是什么)
    - [5.5 WKV 与 WZ 分别是什么](#55-wkv-与-wz-分别是什么)
      - [5.5.1 $W\_{KV}$：内容投影](#551-w_kv内容投影)
      - [5.5.2 $W\_Z$：压缩门控投影](#552-w_z压缩门控投影)
    - [5.6 Ratio=2 时如何压缩 Global KV](#56-ratio2-时如何压缩-global-kv)
    - [5.7 Ratio=1 时是否还需要 WZ](#57-ratio1-时是否还需要-wz)
    - [5.8 Global KV 来自 H20 的含义](#58-global-kv-来自-h20-的含义)
    - [5.9 Layer-specific Projection 的含义](#59-layer-specific-projection-的含义)
    - [5.10 Local KV 来自各层 SWA 的含义](#510-local-kv-来自各层-swa-的含义)
    - [5.11 为什么必须做 SWA Replay](#511-为什么必须做-swa-replay)
    - [5.12 最近 nwin 个 Token 做 Replay 的过程](#512-最近-nwin-个-token-做-replay-的过程)
    - [5.13 为什么 Bounded Replay 是近似的](#513-为什么-bounded-replay-是近似的)
    - [5.14 完整小例子](#514-完整小例子)
      - [Encoder 阶段](#encoder-阶段)
      - [构造 Decoder Global KV](#构造-decoder-global-kv)
      - [SWA Replay](#swa-replay)
      - [生成第一个新 token 8](#生成第一个新-token-8)
    - [5.15 vLLM 与 vLLM-Ascend 当前实现差异](#515-vllm-与-vllm-ascend-当前实现差异)
      - [5.15.1 vLLM CUDA 路径](#5151-vllm-cuda-路径)
      - [5.15.2 当前 vLLM-Ascend 快照](#5152-当前-vllm-ascend-快照)
    - [问题 5 最终总结](#问题-5-最终总结)
  - [问题 6：Ratio=2 的 Token/KV 压缩原理与是否需要复原](#问题-6ratio2-的-tokenkv-压缩原理与是否需要复原)
    - [问题 6 原文](#问题-6-原文)
    - [问题 6 核心结论](#问题-6-核心结论)
    - [6.1 Token 压缩到底压缩了什么](#61-token-压缩到底压缩了什么)
    - [6.2 Ratio=2 的整体处理流程](#62-ratio2-的整体处理流程)
    - [6.3 每个 Token 如何产生 KV 内容和门控](#63-每个-token-如何产生-kv-内容和门控)
      - [6.3.1 KV 内容投影](#631-kv-内容投影)
      - [6.3.2 压缩门控投影](#632-压缩门控投影)
    - [6.4 两个 Token 如何压成一个 Entry](#64-两个-token-如何压成一个-entry)
    - [6.5 数值例子](#65-数值例子)
    - [6.6 代码中的压缩公式](#66-代码中的压缩公式)
    - [6.7 RMSNorm、RoPE 与缓存写入](#67-rmsnormrope-与缓存写入)
    - [6.8 压缩位置和 Slot 映射](#68-压缩位置和-slot-映射)
      - [6.8.1 RoPE 锚点](#681-rope-锚点)
    - [6.9 奇数长度与 Residual 的处理](#69-奇数长度与-residual-的处理)
    - [6.10 Chunked Prefill 如何跨 Chunk 配对](#610-chunked-prefill-如何跨-chunk-配对)
    - [6.11 压缩 KV 如何参加注意力](#611-压缩-kv-如何参加注意力)
    - [6.12 生成时是否需要 Token 复原](#612-生成时是否需要-token-复原)
      - [6.12.1 为什么不需要复原](#6121-为什么不需要复原)
      - [6.12.2 为什么无法完整复原](#6122-为什么无法完整复原)
      - [6.12.3 原始 token 是否全部消失](#6123-原始-token-是否全部消失)
    - [6.13 Decode 阶段的 Ratio=2 缓存更新](#613-decode-阶段的-ratio2-缓存更新)
    - [6.14 为什么不会破坏因果性](#614-为什么不会破坏因果性)
    - [6.15 Token 压缩的实际收益与边界](#615-token-压缩的实际收益与边界)
    - [问题 6 最终总结](#问题-6-最终总结)
  - [问题 7：旧版 CSA 的 Overlap 与 CSA2 的 Softmax Pooling](#问题-7旧版-csa-的-overlap-与-csa2-的-softmax-pooling)
    - [问题 7 原文](#问题-7-原文)
    - [问题 7 核心结论](#问题-7-核心结论)
    - [7.1 这个说法来自哪篇论文](#71-这个说法来自哪篇论文)
      - [7.1.1 原始来源：DeepSeek-V4](#711-原始来源deepseek-v4)
- [\[S^a;S^b\]](#sasb)
- [C\_i^{\\mathrm{Comp}}](#c_imathrmcomp)
      - [7.1.2 V4.1 论文为什么再次提到它](#712-v41-论文为什么再次提到它)
    - [7.2 旧版 CSA 的 Overlap 如何发生](#72-旧版-csa-的-overlap-如何发生)
    - [7.3 vLLM 的 DeepSeek-V4 代码如何实现 Overlap](#73-vllm-的-deepseek-v4-代码如何实现-overlap)
    - [7.4 DeepSeek-V4.1 CSA2 中是否还有 Overlap](#74-deepseek-v41-csa2-中是否还有-overlap)
      - [7.4.1 V4.1 的 ratio=2 应如何理解](#741-v41-的-ratio2-应如何理解)
      - [7.4.2 上游 vLLM 的代码证据](#742-上游-vllm-的代码证据)
      - [7.4.3 vLLM-Ascend 的代码证据](#743-vllm-ascend-的代码证据)
    - [7.5 Ratio=1 的特殊含义](#75-ratio1-的特殊含义)
    - [7.6 为什么称为 Softmax Pooling](#76-为什么称为-softmax-pooling)
      - [7.6.1 Softmax 只负责把 Logits 变成归一化权重](#761-softmax-只负责把-logits-变成归一化权重)
- [\\alpha\_{i,k}](#alpha_ik)
      - [7.6.2 Pooling 是后续的加权归约](#762-pooling-是后续的加权归约)
- [\\bar{\\mathbf c}](#barmathbf-c)
- [\\operatorname{SoftmaxPooling}(C,Z)](#operatornamesoftmaxpoolingcz)
    - [7.7 Gate 是逐通道权重而非单一标量](#77-gate-是逐通道权重而非单一标量)
    - [7.8 CSA、HCA 与 CSA2 对比](#78-csahca-与-csa2-对比)
    - [7.9 为什么前面的 V4.1 分析没有提到 Overlap](#79-为什么前面的-v41-分析没有提到-overlap)
    - [问题 7 最终总结](#问题-7-最终总结)
  - [问题 8：SparseFlashMla 的仓库归属与 Local/Global KV 联合注意力](#问题-8sparseflashmla-的仓库归属与-localglobal-kv-联合注意力)
    - [问题 8 原文](#问题-8-原文)
    - [问题 8 核心结论](#问题-8-核心结论)
    - [8.1 稀疏注意力算子究竟在哪个仓库](#81-稀疏注意力算子究竟在哪个仓库)
      - [8.1.1 上游算子仓库](#811-上游算子仓库)
      - [8.1.2 当前 vLLM-Ascend 已集成算子源码](#812-当前-vllm-ascend-已集成算子源码)
      - [8.1.3 vllm-github 中是什么](#813-vllm-github-中是什么)
    - [8.2 vLLM-Ascend 中的完整调用链](#82-vllm-ascend-中的完整调用链)
    - [8.3 算子的输入分别代表什么](#83-算子的输入分别代表什么)
    - [8.4 算子负责和不负责什么](#84-算子负责和不负责什么)
    - [8.5 Selected Global KV 与 Layer-local SWA KV 如何联合计算](#85-selected-global-kv-与-layer-local-swa-kv-如何联合计算)
    - [8.6 为什么不能分别 Softmax 再相加](#86-为什么不能分别-softmax-再相加)
    - [8.7 联合注意力数值例子](#87-联合注意力数值例子)
    - [8.8 算子内部的在线 Softmax 流水](#88-算子内部的在线-softmax-流水)
    - [8.9 Full、Reindex、Reuse 是否改变注意力公式](#89-fullreindexreuse-是否改变注意力公式)
    - [8.10 这个问题是否可以改写](#810-这个问题是否可以改写)
    - [8.11 当前 Ascend 实现与论文配置的差异](#811-当前-ascend-实现与论文配置的差异)
    - [问题 8 最终总结](#问题-8-最终总结)
  - [问题 9：Indexer Top-K、Local/Global 顺序与在线 Softmax](#问题-9indexer-top-klocalglobal-顺序与在线-softmax)
    - [问题 9 原文](#问题-9-原文)
    - [问题 9 核心结论](#问题-9-核心结论)
    - [9.1 Indexer 与 Main Attention 是两次不同的打分](#91-indexer-与-main-attention-是两次不同的打分)
    - [9.2 Indexer 如何选出 Top-K Global KV](#92-indexer-如何选出-top-k-global-kv)
      - [9.2.1 为每个 Main KV 条目生成轻量 Indexer K](#921-为每个-main-kv-条目生成轻量-indexer-k)
- [k\_j^{\\mathrm{idx}}](#k_jmathrmidx)
      - [9.2.2 生成当前 token 的 Indexer Query](#922-生成当前-token-的-indexer-query)
      - [9.2.3 计算每个压缩位置的检索分数](#923-计算每个压缩位置的检索分数)
- [r\_{t,j,h}](#r_tjh)
- [s\_{t,j}](#s_tj)
      - [9.2.4 因果过滤和 Top-K](#924-因果过滤和-top-k)
- [N\_{\\mathrm{visible}}](#n_mathrmvisible)
- [\\mathcal I\_t](#mathcal-i_t)
    - [9.3 Hierarchical Sparse Indexer](#93-hierarchical-sparse-indexer)
    - [9.4 Local 与 Global 的拼接顺序](#94-local-与-global-的拼接顺序)
- [\\operatorname{softmax}(QK'^\\top)V'](#operatornamesoftmaxqktopv)
- [\\operatorname{softmax}(QK^\\top P^\\top)PV](#operatornamesoftmaxqktop-ptoppv)
    - [9.5 Local Attention Logit 公式详解](#95-local-attention-logit-公式详解)
- [a\_{t,j,h}^{L}](#a_tjhl)
- [o\_t](#o_t)
    - [9.6 为什么 Local SWA 与 Global KV 同时存在](#96-为什么-local-swa-与-global-kv-同时存在)
    - [9.7 在线 Softmax 为什么维护三个状态](#97-在线-softmax-为什么维护三个状态)
- [O](#o)
    - [9.8 在线 Softmax 更新和数值例子](#98-在线-softmax-更新和数值例子)
    - [问题 9 最终总结](#问题-9-最终总结)
  - [问题 10：由压缩 Main KV 生成 Indexer K](#问题-10由压缩-main-kv-生成-indexer-k)
    - [问题 10 原文](#问题-10-原文)
    - [问题 10 核心结论](#问题-10-核心结论)
    - [10.1 压缩条目 Cj 是什么](#101-压缩条目-cj-是什么)
    - [10.2 Indexer 投影矩阵是什么](#102-indexer-投影矩阵是什么)
    - [10.3 RMSNorm 的作用](#103-rmsnorm-的作用)
- [\\operatorname{RMS}(x)](#operatornamermsx)
- [\\operatorname{RMSNorm}(x)](#operatornamermsnormx)
    - [10.4 实际实现中的 RoPE 和 INT8 量化](#104-实际实现中的-rope-和-int8-量化)
- [\\widehat{k}\_j^{\\mathrm{idx}}](#widehatk_jmathrmidx)
- [\\left(k\_j^{\\mathrm{idx,int8}},s\_j^K\\right)](#leftk_jmathrmidxint8s_jkright)
    - [10.5 Main KV 与 Indexer K 的一一对应](#105-main-kv-与-indexer-k-的一一对应)
    - [10.6 为什么从 Main KV 投影 Indexer K](#106-为什么从-main-kv-投影-indexer-k)
    - [问题 10 最终总结](#问题-10-最终总结)
- [k\_j^{\\mathrm{idx}}](#k_jmathrmidx-1)
  - [问题 11：什么是 Main KV](#问题-11什么是-main-kv)
    - [问题 11 原文](#问题-11-原文)
    - [问题 11 核心结论](#问题-11-核心结论)
    - [11.1 Main、Global、Long 与 Compressed KV 的关系](#111-maingloballong-与-compressed-kv-的关系)
    - [11.2 Main KV 怎样生成](#112-main-kv-怎样生成)
      - [Ratio=1](#ratio1)
      - [Ratio=2](#ratio2)
- [C\_j](#c_j)
    - [11.3 Main KV 的存储表示](#113-main-kv-的存储表示)
    - [11.4 Main KV、Indexer K 与 SWA KV 的区别](#114-main-kvindexer-k-与-swa-kv-的区别)
    - [11.5 CED 中 Main KV 的特殊来源](#115-ced-中-main-kv-的特殊来源)
    - [11.6 长上下文例子](#116-长上下文例子)
    - [问题 11 最终总结](#问题-11-最终总结)
  - [问题 12：Full、Reindex、Reuse 与 Main Q](#问题-12fullreindexreuse-与-main-q)
    - [问题 12 原文](#问题-12-原文)
    - [问题 12 核心结论](#问题-12-核心结论)
    - [12.1 Top-K 选择的是 Main KV 条目](#121-top-k-选择的是-main-kv-条目)
    - [12.2 Full Mode](#122-full-mode)
    - [12.3 Reindex Mode](#123-reindex-mode)
    - [12.4 Reuse Mode](#124-reuse-mode)
    - [12.5 代码如何表示三种模式](#125-代码如何表示三种模式)
    - [12.6 40 层的实际模式布局](#126-40-层的实际模式布局)
    - [12.7 Main Q 是什么](#127-main-q-是什么)
    - [12.8 Main Q 的生成过程](#128-main-q-的生成过程)
    - [12.9 Main Q 与 Indexer Q 的区别](#129-main-q-与-indexer-q-的区别)
    - [12.10 为什么三种模式都计算 Main Q 和 SWA KV](#1210-为什么三种模式都计算-main-q-和-swa-kv)
- [a\_{l,t,j}](#a_ltj)
    - [12.11 四层组示例](#1211-四层组示例)
    - [问题 12 最终总结](#问题-12-最终总结)
  - [问题 13：Decoder Hierarchical Indexer 的 Full 层与共享候选池](#问题-13decoder-hierarchical-indexer-的-full-层与共享候选池)
    - [问题 13 原文](#问题-13-原文)
    - [问题 13 核心结论](#问题-13-核心结论)
    - [13.1 “扫描全部 Main KV”究竟扫描什么](#131-扫描全部-main-kv究竟扫描什么)
- [k\_p^{\\mathrm{idx}}](#k_pmathrmidx)
    - [13.2 Full 层的全长 Indexer Score](#132-full-层的全长-indexer-score)
- [s\_{t,p}](#s_tp)
    - [13.3 两条并行支路：本层 Top-512 与候选块](#133-两条并行支路本层-top-512-与候选块)
      - [支路 A：本层自己的 Top-512](#支路-a本层自己的-top-512)
- [\\mathcal I\_t^{(20)}](#mathcal-i_t20)
      - [支路 B：构造共享候选块](#支路-b构造共享候选块)
    - [13.4 每 8 个位置划块及块内最大值](#134-每-8-个位置划块及块内最大值)
- [N\_{\\mathrm{block}}](#n_mathrmblock)
- [S\_{t,b}^{\\mathrm{block}}](#s_tbmathrmblock)
    - [13.5 选择 2048 个块并形成最多 16384 个候选位置](#135-选择-2048-个块并形成最多-16384-个候选位置)
- [\\mathcal B\_t](#mathcal-b_t)
- [|\\mathcal B\_t|](#mathcal-b_t-1)
- [\\mathcal C\_t](#mathcal-c_t)
    - [13.6 为什么候选池远大于最终 Top-512](#136-为什么候选池远大于最终-top-512)
    - [13.7 Decoder 各层的 Full、Reindex、Reuse 数据流](#137-decoder-各层的-fullreindexreuse-数据流)
    - [13.8 Reindex 是否真的不扫描完整长上下文](#138-reindex-是否真的不扫描完整长上下文)
- [s'\_{t,p}](#s_tp-1)
- [\\mathcal I\_t^{(l)}](#mathcal-i_tl)
    - [13.9 理论复杂度与长上下文示例](#139-理论复杂度与长上下文示例)
    - [13.10 当前 vLLM 与 vLLM-Ascend 实现差异](#1310-当前-vllm-与-vllm-ascend-实现差异)
      - [上游 vLLM](#上游-vllm)
      - [当前本地 vLLM-Ascend 910B 路径](#当前本地-vllm-ascend-910b-路径)
    - [问题 13 最终总结](#问题-13-最终总结)
  - [问题 14：普通 KV Cache 究竟存储什么](#问题-14普通-kv-cache-究竟存储什么)
    - [问题 14 原文](#问题-14-原文)
    - [问题 14 核心结论](#问题-14-核心结论)
    - [14.1 K 和 V 是怎样产生的](#141-k-和-v-是怎样产生的)
    - [14.2 为什么必须同时保存 K 和 V](#142-为什么必须同时保存-k-和-v)
    - [14.3 为什么不保存历史 Query](#143-为什么不保存历史-query)
    - [14.4 为什么 KV Cache 必须逐层保存](#144-为什么-kv-cache-必须逐层保存)
    - [14.5 Prefill 和 Decode 怎样写入缓存](#145-prefill-和-decode-怎样写入缓存)
    - [14.6 vLLM 的分页物理布局](#146-vllm-的分页物理布局)
    - [14.7 Slot Mapping 与 Block Table](#147-slot-mapping-与-block-table)
    - [14.8 KV Cache 的数据类型与量化内容](#148-kv-cache-的数据类型与量化内容)
    - [14.9 普通 KV Cache 的显存计算](#149-普通-kv-cache-的显存计算)
    - [14.10 MHA、GQA 与 MQA 的差别](#1410-mhagqa-与-mqa-的差别)
    - [14.11 普通 KV Cache 与 DeepSeek-V4.1 的区别](#1411-普通-kv-cache-与-deepseek-v41-的区别)
    - [14.12 常见误区](#1412-常见误区)
    - [问题 14 最终总结](#问题-14-最终总结)
  - [问题 15：FP4 Indexer Q/K、QAT、MXFP4 与 FP4 Main KV](#问题-15fp4-indexer-qkqatmxfp4-与-fp4-main-kv)
    - [问题 15 原文](#问题-15-原文)
    - [问题 15 核心结论](#问题-15-核心结论)
    - [15.1 Indexer Q/K 为什么适合 FP4](#151-indexer-qk-为什么适合-fp4)
    - [15.2 Indexer Q/K 的 QAT 过程与原理](#152-indexer-qk-的-qat-过程与原理)
    - [15.3 OCP MXFP4 的位级格式](#153-ocp-mxfp4-的位级格式)
    - [15.4 MXFP4 的量化公式和数值例子](#154-mxfp4-的量化公式和数值例子)
    - [15.5 vLLM 中 Indexer Q/K 的 MXFP4 实现](#155-vllm-中-indexer-qk-的-mxfp4-实现)
    - [15.6 V4.1 Main KV 为什么不照搬 OCP MXFP4](#156-v41-main-kv-为什么不照搬-ocp-mxfp4)
    - [15.7 V4.1 Main KV 的 E2M1 加 E4M3 格式](#157-v41-main-kv-的-e2m1-加-e4m3-格式)
    - [15.8 为什么可以省略 Global Scale](#158-为什么可以省略-global-scale)
    - [15.9 为什么在 RoPE 后量化](#159-为什么在-rope-后量化)
    - [15.10 反量化为何解除原生 FP4 MatMul 要求](#1510-反量化为何解除原生-fp4-matmul-要求)
    - [15.11 反量化是否抵消 FP4 收益](#1511-反量化是否抵消-fp4-收益)
    - [15.12 Indexer 与 Main Attention 的不同选择](#1512-indexer-与-main-attention-的不同选择)
    - [15.13 vLLM 与 vLLM-Ascend 当前实现差异](#1513-vllm-与-vllm-ascend-当前实现差异)
    - [问题 15 最终总结](#问题-15-最终总结)
  - [问题 16：E2M1 与 E4M3 Scale 的反量化及 Ascend 950 原生 MXFP4 MatMul](#问题-16e2m1-与-e4m3-scale-的反量化及-ascend-950-原生-mxfp4-matmul)
    - [问题 16 原文](#问题-16-原文)
    - [问题 16 核心结论](#问题-16-核心结论)
- [\\widehat{x}\_i](#widehatx_i)
    - [16.1 E2M1 能否与 E4M3 Scale 直接相乘](#161-e2m1-能否与-e4m3-scale-直接相乘)
    - [16.2 vLLM Main KV 的实际反量化过程](#162-vllm-main-kv-的实际反量化过程)
    - [16.3 DeepSeek Main KV 为什么不是标准 OCP MXFP4](#163-deepseek-main-kv-为什么不是标准-ocp-mxfp4)
    - [16.4 Ascend 950 是否支持原生 MXFP4 MatMul](#164-ascend-950-是否支持原生-mxfp4-matmul)
    - [16.5 vLLM-Ascend A8W4 内核的真实计算路径](#165-vllm-ascend-a8w4-内核的真实计算路径)
    - [16.6 FP4 MatMul 为什么受硬件指令约束](#166-fp4-matmul-为什么受硬件指令约束)
- [C\_{mn}](#c_mn)
    - [16.7 反量化为什么能绕开原生格式限制](#167-反量化为什么能绕开原生格式限制)
    - [16.8 当前 vLLM-Ascend DeepSeek-V4.1 的实际数据类型](#168-当前-vllm-ascend-deepseek-v41-的实际数据类型)
    - [问题 16 最终总结](#问题-16-最终总结)
  - [问题 17：CSA2 训练中的 Shadow Indexer](#问题-17csa2-训练中的-shadow-indexer)
    - [问题 17 原文](#问题-17-原文)
    - [问题 17 核心结论](#问题-17-核心结论)
    - [17.1 为什么 CSA2 会遇到 Pipeline Parallel 问题](#171-为什么-csa2-会遇到-pipeline-parallel-问题)
    - [17.2 Shadow Indexer 的核心结构](#172-shadow-indexer-的核心结构)
    - [17.3 为什么需要单一 Logical Owner](#173-为什么需要单一-logical-owner)
    - [17.4 完整的前向、反向和参数更新流程](#174-完整的前向反向和参数更新流程)
      - [第 1 步：参数同步](#第-1-步参数同步)
      - [第 2 步：各 Stage 本地执行](#第-2-步各-stage-本地执行)
- [s\_j](#s_j)
      - [第 3 步：各 Stage 反向计算局部梯度](#第-3-步各-stage-反向计算局部梯度)
- [g\_s](#g_s)
      - [第 4 步：梯度聚合](#第-4-步梯度聚合)
- [g\_{\\mathrm{owner}}](#g_mathrmowner)
      - [第 5 步：Owner 更新参数](#第-5-步owner-更新参数)
      - [第 6 步：将新参数重新同步给 Shadow](#第-6-步将新参数重新同步给-shadow)
    - [17.5 Shadow Indexer 复制什么、不复制什么](#175-shadow-indexer-复制什么不复制什么)
      - [复制的主要对象](#复制的主要对象)
      - [不应理解为复制整个 Decoder 层](#不应理解为复制整个-decoder-层)
      - [也不等于复制全部百万 Token KV Cache](#也不等于复制全部百万-token-kv-cache)
    - [17.6 三项训练机制怎样配合](#176-三项训练机制怎样配合)
      - [机制一：Shadow Indexers](#机制一shadow-indexers)
      - [机制二：Pipeline Payload Extensions](#机制二pipeline-payload-extensions)
      - [机制三：Microbatch-aware Shared-state Manager](#机制三microbatch-aware-shared-state-manager)
    - [17.7 与 Full、Reindex、Reuse 的关系](#177-与-fullreindexreuse-的关系)
    - [17.8 为什么不把共享层限制在同一个 Pipeline Stage](#178-为什么不把共享层限制在同一个-pipeline-stage)
    - [17.9 vLLM 与 vLLM-Ascend 代码能证明什么](#179-vllm-与-vllm-ascend-代码能证明什么)
      - [17.9.1 上游 vLLM](#1791-上游-vllm)
      - [17.9.2 vLLM-Ascend](#1792-vllm-ascend)
    - [17.10 论文明确内容与未披露细节](#1710-论文明确内容与未披露细节)
    - [问题 17 最终总结](#问题-17-最终总结)
- [\\text{Shadow Indexer}](#textshadow-indexer)
  - [问题 18：Inference System 中的融合算子及其对算子流的简化](#问题-18inference-system-中的融合算子及其对算子流的简化)
    - [问题 18 原文](#问题-18-原文)
    - [问题 18 核心结论](#问题-18-核心结论)
    - [18.1 为什么 DeepSeek-V4.1-Flash 特别依赖融合算子](#181-为什么-deepseek-v41-flash-特别依赖融合算子)
    - [18.2 FlashMLA 的 fused-RoPE-attention-RoPE-cast](#182-flashmla-的-fused-rope-attention-rope-cast)
      - [18.2.1 它融合了什么](#1821-它融合了什么)
      - [18.2.2 为什么最后还要 cast](#1822-为什么最后还要-cast)
      - [18.2.3 vLLM 代码证据](#1823-vllm-代码证据)
    - [18.3 DeepGEMM 的 Mega-Gate](#183-deepgemm-的-mega-gate)
      - [18.3.1 普通 MoE Gate 的逻辑](#1831-普通-moe-gate-的逻辑)
      - [18.3.2 Mega-Gate 做什么](#1832-mega-gate-做什么)
    - [18.4 DeepGEMM 的 Mega-mHC](#184-deepgemm-的-mega-mhc)
      - [18.4.1 mHC 为什么容易产生碎片化小算子](#1841-mhc-为什么容易产生碎片化小算子)
      - [18.4.2 Mega-mHC 的关键思想](#1842-mega-mhc-的关键思想)
    - [18.5 DeepGEMM 的 Mega-MoE](#185-deepgemm-的-mega-moe)
      - [18.5.1 普通 Expert Parallel MoE 的执行链](#1851-普通-expert-parallel-moe-的执行链)
      - [18.5.2 Mega-MoE 做什么](#1852-mega-moe-做什么)
    - [18.6 TileKernels 是什么](#186-tilekernels-是什么)
    - [18.7 DeepSelect 的 TopK Kernel](#187-deepselect-的-topk-kernel)
      - [18.7.1 输入和输出](#1871-输入和输出)
- [\\mathcal G\_t](#mathcal-g_t)
      - [18.7.2 它不负责什么](#1872-它不负责什么)
    - [18.8 这些算子如何把复杂架构变成简单算子流](#188-这些算子如何把复杂架构变成简单算子流)
      - [第一层：Reuse Mode 先删掉不必要工作](#第一层reuse-mode-先删掉不必要工作)
      - [第二层：大 kernel 合并相邻计算](#第二层大-kernel-合并相邻计算)
      - [第三层：让生产者和消费者共用 layout](#第三层让生产者和消费者共用-layout)
      - [第四层：持久化调度覆盖通信空泡](#第四层持久化调度覆盖通信空泡)
    - [18.9 Prefill 与 Decode 的典型数据流](#189-prefill-与-decode-的典型数据流)
      - [18.9.1 Reuse Mode 的概念流程](#1891-reuse-mode-的概念流程)
      - [18.9.2 为什么 decode 往往比 prefill kernel 更少](#1892-为什么-decode-往往比-prefill-kernel-更少)
    - [18.10 vLLM 与 vLLM-Ascend 中的对应实现](#1810-vllm-与-vllm-ascend-中的对应实现)
      - [18.10.1 NVIDIA 路径](#18101-nvidia-路径)
      - [18.10.2 Ascend 路径不是逐名字照搬](#18102-ascend-路径不是逐名字照搬)
    - [18.11 容易混淆的几个 Top-K](#1811-容易混淆的几个-top-k)
    - [问题 18 最终总结](#问题-18-最终总结)
  - [问题 19：SWA Bounded Replay 与在 Encoder 处结束 Prefill](#问题-19swa-bounded-replay-与在-encoder-处结束-prefill)
    - [问题 19 原文](#问题-19-原文)
    - [问题 19 核心结论](#问题-19-核心结论)
    - [19.1 CED 中 Encoder 和 Decoder 的工作划分](#191-ced-中-encoder-和-decoder-的工作划分)
      - [19.1.1 Global KV 的来源](#1911-global-kv-的来源)
- [C^{(l)}](#cl)
      - [19.1.2 Local SWA KV 的来源](#1912-local-swa-kv-的来源)
- [K\_{\\mathrm{SWA}}^{(l)},V\_{\\mathrm{SWA}}^{(l)}](#k_mathrmswalv_mathrmswal)
    - [19.2 在 Encoder 处结束 Prefill 是什么意思](#192-在-encoder-处结束-prefill-是什么意思)
    - [19.3 是否从不缓存 Decoder SWA KV](#193-是否从不缓存-decoder-swa-kv)
      - [19.3.1 运行时设备缓存：会缓存](#1931-运行时设备缓存会缓存)
      - [19.3.2 Prefix Cache：Bounded Replay 模式下不缓存](#1932-prefix-cachebounded-replay-模式下不缓存)
      - [19.3.3 Persistent KV Cache：不长期保存](#1933-persistent-kv-cache不长期保存)
    - [19.4 为什么精确重建需要最后 L/2 乘 nwin 个 Prompt Token](#194-为什么精确重建需要最后-l2-乘-nwin-个-prompt-token)
      - [一层时](#一层时)
      - [两层时](#两层时)
      - [三层及一般情形](#三层及一般情形)
- [R\_D](#r_d)
- [\\approx DW](#approx-dw)
      - [数值例子](#数值例子)
    - [19.5 为什么只 Replay 最后 nwin 个 Token 是近似的](#195-为什么只-replay-最后-nwin-个-token-是近似的)
    - [19.6 Encoder SWA Bounded Replay 的策略](#196-encoder-swa-bounded-replay-的策略)
      - [对 replay Prefix token](#对-replay-prefix-token)
      - [对未缓存 suffix token](#对未缓存-suffix-token)
    - [19.7 Encoder 与 Decoder Bounded Replay 的区别](#197-encoder-与-decoder-bounded-replay-的区别)
- [\\mathcal L\_i](#mathcal-l_i)
    - [19.8 vLLM 中的实现链路](#198-vllm-中的实现链路)
      - [19.8.1 Scheduler 将 Prefix 命中位置向前回退一个窗口](#1981-scheduler-将-prefix-命中位置向前回退一个窗口)
      - [19.8.2 Replay token 不覆盖已经缓存的 Global KV](#1982-replay-token-不覆盖已经缓存的-global-kv)
      - [19.8.3 Attention 不读取 replay\_start 之前的 SWA KV](#1983-attention-不读取-replay_start-之前的-swa-kv)
- [\\operatorname{SWAStart}(i)](#operatornameswastarti)
      - [19.8.4 Decoder 后半部分只执行尾部窗口](#1984-decoder-后半部分只执行尾部窗口)
    - [19.9 vLLM-Ascend 当前代码状态](#199-vllm-ascend-当前代码状态)
    - [19.10 100K Prompt 的计算量示例](#1910-100k-prompt-的计算量示例)
      - [不使用 CED](#不使用-ced)
- [100,000\\times40](#100000times40-1)
      - [CED，但精确重建 Decoder SWA](#ced但精确重建-decoder-swa)
- [100,000\\times20](#100000times20-1)
- [2560\\times20](#2560times20)
      - [CED 加 Decoder Bounded Replay](#ced-加-decoder-bounded-replay)
- [128\\times20](#128times20-1)
- [2,000,000+2,560](#20000002560-1)
    - [问题 19 最终总结](#问题-19-最终总结)
- [C\_{\\text{decoder replay}}](#c_textdecoder-replay)
  - [问题 20：DeepSeek-V4 与 DeepSeek-V4.1-Flash 的 KV Cache 管理差异](#问题-20deepseek-v4-与-deepseek-v41-flash-的-kv-cache-管理差异)
    - [问题 20 原文](#问题-20-原文)
    - [问题 20 核心结论](#问题-20-核心结论)
    - [20.1 必须区分运行时 KV Cache 与持久化 Prefix Cache](#201-必须区分运行时-kv-cache-与持久化-prefix-cache)
    - [20.2 两代方案总览对比](#202-两代方案总览对比)
    - [20.3 DeepSeek-V4 的设备端 KV Cache 组织](#203-deepseek-v4-的设备端-kv-cache-组织)
    - [20.4 V4 的 Indexer K 与重叠 Token 压缩](#204-v4-的-indexer-k-与重叠-token-压缩)
      - [V4 的 Indexer K 是相对独立的缓存路径](#v4-的-indexer-k-是相对独立的缓存路径)
      - [V4 的 CSA 存在 overlap](#v4-的-csa-存在-overlap)
    - [20.5 V4.1 的 CSA2 跨层共享](#205-v41-的-csa2-跨层共享)
    - [20.6 Full、Reindex、Reuse 改变了什么](#206-fullreindexreuse-改变了什么)
    - [20.7 V4.1 的 FP4 Main KV](#207-v41-的-fp4-main-kv)
    - [20.8 SWA KV 的运行时与持久化管理差异](#208-swa-kv-的运行时与持久化管理差异)
      - [V4.1 运行时仍然需要 SWA KV](#v41-运行时仍然需要-swa-kv)
      - [V4 的持久化策略](#v4-的持久化策略)
      - [V4.1 的持久化策略](#v41-的持久化策略)
    - [20.9 Bounded Replay 如何替代持久化 SWA](#209-bounded-replay-如何替代持久化-swa)
      - [V4 精确恢复为何昂贵](#v4-精确恢复为何昂贵)
      - [V4.1 的 CED 改变了 Global KV 的生成依赖](#v41-的-ced-改变了-global-kv-的生成依赖)
    - [20.10 100K Prompt 示例](#2010-100k-prompt-示例)
      - [V4](#v4)
      - [V4.1](#v41)
    - [20.11 890 Bytes/Token 和八分之一容量的含义](#2011-890-bytestoken-和八分之一容量的含义)
      - [为什么持久化总量约为八分之一](#为什么持久化总量约为八分之一)
    - [20.12 当前 vLLM 与 vLLM-Ascend 的实现边界](#2012-当前-vllm-与-vllm-ascend-的实现边界)
    - [问题 20 最终总结](#问题-20-最终总结)
  - [问题 21：1000 Token Prefill 中 Causal Encoder 的 SWA 计算范围](#问题-211000-token-prefill-中-causal-encoder-的-swa-计算范围)
    - [问题 21 原文](#问题-21-原文)
    - [问题 21 核心结论](#问题-21-核心结论)
    - [21.1 窗口 128 限制的是 Key 范围而不是 Query 数量](#211-窗口-128-限制的是-key-范围而不是-query-数量)
    - [21.2 必须区分计算、直接可见和最终保留](#212-必须区分计算直接可见和最终保留)
      - [位置 0～871 是否作为 Query 执行 Attention](#位置-0871-是否作为-query-执行-attention)
      - [位置 999 是否通过本层 SWA 直接读取 0～871](#位置-999-是否通过本层-swa-直接读取-0871)
      - [Prefill 结束后是否仍要保存 0～871 的 SWA KV](#prefill-结束后是否仍要保存-0871-的-swa-kv)
    - [21.3 为什么位置 0 到 871 必须计算](#213-为什么位置-0-到-871-必须计算)
      - [它们必须生成自己的 Encoder hidden states](#它们必须生成自己的-encoder-hidden-states)
      - [CED 需要整个 Prompt 的 Encoder 最终输出](#ced-需要整个-prompt-的-encoder-最终输出)
      - [CED 复杂度公式也包含全部 Encoder token](#ced-复杂度公式也包含全部-encoder-token)
    - [21.4 多层 SWA 的间接感受野](#214-多层-swa-的间接感受野)
    - [21.5 单层 SWA 配对数](#215-单层-swa-配对数)
    - [21.6 vLLM 与 vLLM-Ascend 代码对应](#216-vllm-与-vllm-ascend-代码对应)
      - [上游 vLLM 的窗口公式](#上游-vllm-的窗口公式)
      - [Ascend 的窗口与缓存写入](#ascend-的窗口与缓存写入)
    - [21.7 Bounded Replay 是特殊情况](#217-bounded-replay-是特殊情况)
    - [问题 21 最终总结](#问题-21-最终总结)
  - [问题 22：第 2 层 Full 层需要计算哪些位置的 SWA](#问题-22第-2-层-full-层需要计算哪些位置的-swa)
    - [问题 22 原文](#问题-22-原文)
    - [问题 22 核心结论](#问题-22-核心结论)
    - [22.1 为什么第 2 层是 Full 层](#221-为什么第-2-层是-full-层)
    - [22.2 Full 不是 Full Attention](#222-full-不是-full-attention)
    - [22.3 第 2 层的 SWA 位置与窗口](#223-第-2-层的-swa-位置与窗口)
    - [22.4 第 2 层 Full 的完整数据流](#224-第-2-层-full-的完整数据流)
    - [22.5 Ratio 等于 2 的 Main KV 与 Top-K](#225-ratio-等于-2-的-main-kv-与-top-k)
    - [22.6 第 2 层 SWA 计算量](#226-第-2-层-swa-计算量)
    - [22.7 Prefill 结束后留下哪些缓存](#227-prefill-结束后留下哪些缓存)
    - [22.8 代码实现证据](#228-代码实现证据)
      - [上游 vLLM](#上游-vllm-1)
      - [vLLM-Ascend](#vllm-ascend)
    - [22.9 Bounded Replay 时答案如何变化](#229-bounded-replay-时答案如何变化)
    - [问题 22 最终总结](#问题-22-最终总结)
  - [问题 23：第 3～7 层 Reuse 层需要计算哪些位置的 SWA](#问题-23第-37-层-reuse-层需要计算哪些位置的-swa)
    - [问题 23 原文](#问题-23-原文)
    - [问题 23 核心结论](#问题-23-核心结论)
    - [23.1 为什么第 3～7 层是 Reuse 层](#231-为什么第-37-层是-reuse-层)
    - [23.2 Reuse 复用什么、不复用什么](#232-reuse-复用什么不复用什么)
    - [23.3 首次完整 Prefill 的 SWA 计算位置](#233-首次完整-prefill-的-swa-计算位置)
    - [23.4 第 3～7 层的数据流](#234-第-37-层的数据流)
    - [23.5 SWA 计算量](#235-swa-计算量)
    - [23.6 Bounded Replay 时答案如何变化](#236-bounded-replay-时答案如何变化)
    - [23.7 代码实现证据](#237-代码实现证据)
    - [问题 23 最终总结](#问题-23-最终总结)
  - [问题 24：第 20 层 Full 层需要计算哪些位置的 SWA](#问题-24第-20-层-full-层需要计算哪些位置的-swa)
    - [问题 24 原文](#问题-24-原文)
    - [问题 24 核心结论](#问题-24-核心结论)
    - [24.1 为什么第 20 层是 Full 层](#241-为什么第-20-层是-full-层)
    - [24.2 论文理想 CED 与当前上游实现的口径](#242-论文理想-ced-与当前上游实现的口径)
    - [24.3 上游 vLLM 中第 20 层的实际计算范围](#243-上游-vllm-中第-20-层的实际计算范围)
    - [24.4 第 20 层完整数据流](#244-第-20-层完整数据流)
    - [24.5 当前 vLLM-Ascend 的行为](#245-当前-vllm-ascend-的行为)
    - [问题 24 最终总结](#问题-24-最终总结)
  - [问题 25：第 21～23 层 Reuse 的 SWA 位置与缓存空间](#问题-25第-2123-层-reuse-的-swa-位置与缓存空间)
    - [问题 25 原文](#问题-25-原文)
    - [问题 25 核心结论](#问题-25-核心结论)
    - [25.1 第 21～23 层的角色](#251-第-2123-层的角色)
    - [25.2 Bounded Replay 中计算哪些位置](#252-bounded-replay-中计算哪些位置)
    - [25.3 Replay 内部的 SWA 窗口](#253-replay-内部的-swa-窗口)
    - [25.4 三层的 SWA 计算量](#254-三层的-swa-计算量)
    - [25.5 每层 66 KiB 的推导](#255-每层-66-kib-的推导)
    - [25.6 不同缓存格式的容量对比](#256-不同缓存格式的容量对比)
    - [25.7 当前 vLLM-Ascend 为什么通常不是 66 KiB](#257-当前-vllm-ascend-为什么通常不是-66-kib)
    - [25.8 论文、上游 NVIDIA 与 Ascend 的最终对比](#258-论文上游-nvidia-与-ascend-的最终对比)
    - [问题 25 最终总结](#问题-25-最终总结)
  - [问题 26：如何理解 Decoder SWA Bounded Replay](#问题-26如何理解-decoder-swa-bounded-replay)
    - [问题 26 原文](#问题-26-原文)
    - [问题 26 核心结论](#问题-26-核心结论)
    - [26.1 原文逐句拆解](#261-原文逐句拆解)
    - [26.2 为什么只把 Encoder 输出送入 Decoder](#262-为什么只把-encoder-输出送入-decoder)
    - [26.3 Replay 不是重新生成 Token](#263-replay-不是重新生成-token)
    - [26.4 1000 Token、窗口 128 的完整流程](#264-1000-token窗口-128-的完整流程)
    - [26.5 Feed Their Encoder Outputs 的含义](#265-feed-their-encoder-outputs-的含义)
    - [26.6 Same SWA Truncation 的含义](#266-same-swa-truncation-的含义)
    - [26.7 为什么重建结果只是近似](#267-为什么重建结果只是近似)
    - [26.8 为什么精确重建非常昂贵](#268-为什么精确重建非常昂贵)
    - [26.9 At Every Prefill 的准确含义](#269-at-every-prefill-的准确含义)
    - [26.10 Only for Decoding 不等于不缓存](#2610-only-for-decoding-不等于不缓存)
    - [26.11 为什么不写入 Prefix Cache](#2611-为什么不写入-prefix-cache)
      - [1. 节省持久化缓存空间](#1-节省持久化缓存空间)
      - [2. 它可以低成本重建](#2-它可以低成本重建)
      - [3. Replay 状态不是完整 Forward 的规范状态](#3-replay-状态不是完整-forward-的规范状态)
    - [26.12 Prefix Cache 命中示例](#2612-prefix-cache-命中示例)
    - [26.13 为什么 Post-training 要模拟 Replay](#2613-为什么-post-training-要模拟-replay)
    - [问题 26 最终总结](#问题-26-最终总结)
  - [问题 27：Prefill 计算量接近减半的说法是否准确](#问题-27prefill-计算量接近减半的说法是否准确)
    - [问题 27 原文](#问题-27-原文)
    - [问题 27 核心结论](#问题-27-核心结论)
    - [27.1 论文的减半公式在计算什么](#271-论文的减半公式在计算什么)
    - [27.2 Encoder 与 Decoder 的模式布局](#272-encoder-与-decoder-的模式布局)
      - [Causal Encoder：层 0～19](#causal-encoder层-019)
      - [Decoder：层 20～39](#decoder层-2039)
    - [27.3 Reuse 不是复用整层输出](#273-reuse-不是复用整层输出)
    - [27.4 Reindex 仍执行哪些计算](#274-reindex-仍执行哪些计算)
    - [27.5 三种模式的成本分解](#275-三种模式的成本分解)
    - [27.6 CED 与 CSA2 优化不同的计算维度](#276-ced-与-csa2-优化不同的计算维度)
      - [CED：减少完整 Prompt 穿过的层数](#ced减少完整-prompt-穿过的层数)
      - [CSA2 Reuse/Reindex：减少每层内部的 KV 与索引工作](#csa2-reusereindex减少每层内部的-kv-与索引工作)
    - [27.7 更精确的模式加权公式](#277-更精确的模式加权公式)
    - [27.8 实际比例为什么可能偏离一半](#278-实际比例为什么可能偏离一半)
      - [情况一：Encoder 和 Decoder 总成本接近](#情况一encoder-和-decoder-总成本接近)
      - [情况二：Decoder 因大量 Reuse 而明显更便宜](#情况二decoder-因大量-reuse-而明显更便宜)
      - [情况三：Decoder 比 Encoder 更昂贵](#情况三decoder-比-encoder-更昂贵)
    - [27.9 不同 Prompt 长度下的理论比例](#279-不同-prompt-长度下的理论比例)
    - [27.10 为什么实际延迟不一定减半](#2710-为什么实际延迟不一定减半)
      - [1. 不同模式的层成本不一样](#1-不同模式的层成本不一样)
      - [2. Encoder 与 Decoder 压缩比例不同](#2-encoder-与-decoder-压缩比例不同)
      - [3. Replay batch 较小，硬件利用率可能较低](#3-replay-batch-较小硬件利用率可能较低)
      - [4. Indexer 成本不完全服从简单的 $NL$](#4-indexer-成本不完全服从简单的-nl)
      - [5. MoE 和通信可能成为瓶颈](#5-moe-和通信可能成为瓶颈)
    - [27.11 如何准确表述论文结论](#2711-如何准确表述论文结论)
    - [问题 27 最终总结](#问题-27-最终总结)
      - [判断一：Reuse/Reindex 会不会让论文公式失效？](#判断一reusereindex-会不会让论文公式失效)
      - [判断二：“计算量减半”是不是精确结论？](#判断二计算量减半是不是精确结论)
  - [问题 28：为什么长期 Global KV 要保存在 SSD](#问题-28为什么长期-global-kv-要保存在-ssd)
    - [问题 28 原文](#问题-28-原文)
    - [问题 28 核心结论](#问题-28-核心结论)
    - [28.1 Runtime KV 与 Persistent KV](#281-runtime-kv-与-persistent-kv)
      - [1. Runtime KV Cache](#1-runtime-kv-cache)
      - [2. Persistent KV Cache](#2-persistent-kv-cache)
    - [28.2 为什么不能长期全部放在 HBM](#282-为什么不能长期全部放在-hbm)
      - [单个 Prefix 的 Global KV 大小](#单个-prefix-的-global-kv-大小)
    - [28.3 请求数量与保留时间造成容量爆炸](#283-请求数量与保留时间造成容量爆炸)
    - [28.4 为什么 Global KV 值得长期保存](#284-为什么-global-kv-值得长期保存)
    - [28.5 SSD 较慢为什么仍可使用](#285-ssd-较慢为什么仍可使用)
    - [28.6 Global KV 与 SWA KV 的保留策略](#286-global-kv-与-swa-kv-的保留策略)
      - [Global KV](#global-kv)
      - [SWA KV](#swa-kv)
    - [28.7 为什么 V4.1 Persistent KV 约为 V4 的八分之一](#287-为什么-v41-persistent-kv-约为-v4-的八分之一)
      - [第一个因素：不再持久化 SWA KV](#第一个因素不再持久化-swa-kv)
      - [第二个因素：Global KV 压缩到 V4 的约四分之一](#第二个因素global-kv-压缩到-v4-的约四分之一)
    - [28.8 为什么不能完全不保存 Global KV](#288-为什么不能完全不保存-global-kv)
    - [28.9 SSD 容量之外的原因](#289-ssd-容量之外的原因)
      - [1. 每字节成本低](#1-每字节成本低)
      - [2. 数据需要跨请求存在](#2-数据需要跨请求存在)
      - [3. Global KV 的访问具有长尾特征](#3-global-kv-的访问具有长尾特征)
      - [4. 可以分层管理冷热数据](#4-可以分层管理冷热数据)
    - [28.10 常见误解](#2810-常见误解)
      - [误解一：模型直接在 SSD 上做 Attention](#误解一模型直接在-ssd-上做-attention)
      - [误解二：所有请求的 KV 都永久保存在 SSD](#误解二所有请求的-kv-都永久保存在-ssd)
      - [误解三：V4.1 完全不保存 SWA KV](#误解三v41-完全不保存-swa-kv)
    - [问题 28 最终总结](#问题-28-最终总结)
  - [问题 29：为什么 Prefill 激活 8B、Decode 激活 16B 参数](#问题-29为什么-prefill-激活-8bdecode-激活-16b-参数)
    - [问题 29 原文](#问题-29-原文)
    - [问题 29 核心结论](#问题-29-核心结论)
    - [29.1 激活参数是什么意思](#291-激活参数是什么意思)
    - [29.2 为什么 552B 总参数只激活约 16B](#292-为什么-552b-总参数只激活约-16b)
    - [29.3 552B 总参数的 MoE 数量级核算](#293-552b-总参数的-moe-数量级核算)
    - [29.4 每个 Token 在单层激活多少专家参数](#294-每个-token-在单层激活多少专家参数)
    - [29.5 从 16B 反推 8B](#295-从-16b-反推-8b)
    - [29.6 为什么普通 Prefill Token 主要只经过 20 层](#296-为什么普通-prefill-token-主要只经过-20-层)
    - [29.7 为什么 Decode Token 必须经过 40 层](#297-为什么-decode-token-必须经过-40-层)
    - [29.8 Full、Reindex、Reuse 对激活参数的影响](#298-fullreindexreuse-对激活参数的影响)
    - [29.9 用 MoE 参数核验 8B 和 16B](#299-用-moe-参数核验-8b-和-16b)
    - [29.10 为什么 196B Engram 不直接加到 8B/16B](#2910-为什么-196b-engram-不直接加到-8b16b)
    - [29.11 Decoder SWA Bounded Replay 的例外](#2911-decoder-swa-bounded-replay-的例外)
    - [29.12 激活参数不等于 FLOPs](#2912-激活参数不等于-flops)
    - [问题 29 最终总结](#问题-29-最终总结)

---

## 问题 1：20-layer causal encoder 是什么

### 问题原文

> 2.1. Overview 中提到 20-layer causal encoder，请问这里面的 20-layer causal encoder 是什么？其作用是？不是说现在的推理模型主要架构都是 decoder only 的，为什么这个模型设计了 encoder 呢？

### 结论

论文里的 **20-layer causal encoder** 不是 BERT、T5 或原始 Transformer Encoder 那种可以同时看左右两侧 token 的双向 Encoder，也不意味着 DeepSeek-V4.1-Flash 从 decoder-only 模型变成了传统的 encoder-decoder 模型。

更准确地说，它是：

> 40 层自回归语言模型的前 20 层，被赋予了“把长 Prompt 编码成可供后 20 层复用的全局 KV 表示”的职责，因此论文将其命名为 causal encoder。

它依然使用因果注意力：

$$
H_t^{(l)} \text{ 只能依赖 } x_{\le t}
$$

也就是说，它不能看到未来 token。整个模型对外仍然是自回归 Causal LM，仍然执行 next-token prediction；vLLM 代码中也明确注册为 `DeepseekV41LLMForCausalLM`，而不是 seq2seq 模型。

本次分析所核对的本地代码版本为：

- `D:\Code\vllm-github`：提交 `027b6f3a22`，分支 `main`；
- `D:\Code\vllm-ascend\vllm-ascend`：提交 `32f29b0d4`，分支 `l2_929`；
- 论文：`D:\Code\vllm_notes\DeepSeekV41-Flash\DeepSeek-V4.1-Flash Pushing the Limits of KV Cache Compression.pdf`，重点参考第 7～9 页的 Figure 3、Section 2.1 和 Section 2.2。

### 1. 20-layer causal encoder 的结构

DeepSeek-V4.1-Flash 的语言主干一共有 40 个 causal Transformer block，并被划分为两段：

| 部分 | 论文中的层号 | 代码中的索引 | 主要职责 |
|---|---:|---:|---|
| Causal Encoder | 1～20 | 0～19 | 对完整 Prompt 做深层因果编码，产生 $H^{20}$ 及前半段所需的 KV |
| Decoder | 21～40 | 20～39 | 基于 $H^{20}$ 构造或复用全局 KV，对最近窗口和生成 token 做进一步计算 |

论文 Figure 3 给出了更加具体的布局：

- 前两层只使用 Sliding Window Attention，简称 SWA；
- Causal Encoder 的其余 18 层使用 CSA2，压缩比为 2；
- Decoder 的 20 层使用压缩比为 1 的 CSA2，并通过 Full、Reindex、Reuse 三种模式复用 KV 和稀疏索引；
- 每个 Transformer block 仍然包含 DeepSeekMoE；
- 这里讨论的是语言 backbone 中的 causal encoder，不要与 Figure 3 底部单独处理图像的 `Vision Encoder` 混淆。

vLLM-Ascend 的测试配置把这个结构直接表示为：

```python
num_hidden_layers = 40
compress_ratios = [0, 0] + [2] * 18 + [1] * 20
kv_source_layer_ids = [2, 8, 14, 20]
index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer_id = 20
sliding_window = 128
```

对应源码位置为：

- `D:\Code\vllm-ascend\vllm-ascend\tests\deepseek_v41_utils.py:273-287`。

各字段含义如下：

- `[0, 0]`：代码索引 0、1 的前两层只有 SWA；
- `[2] * 18`：索引 2～19 的 18 层属于 causal encoder 的 CSA2 部分，主 KV 按两个 token 压缩成一个 entry；
- `[1] * 20`：索引 20～39 的 20 层属于 decoder；
- `kv_source_layer_ids=[2, 8, 14, 20]`：物理长程 KV cache 并不是每一层都单独保存，而是由少数 source layer 产生，其余 consumer layer 复用；
- `index_source_layer_ids=[2,8,14,20,24,28,32,36]`：只有这些层重新计算稀疏 Top-K 索引，其余层可以复用已有索引；
- `candidate_source_layer_id=20`：decoder 的第一个 block 建立 Hierarchical Sparse Indexer 使用的 candidate pool。

这里有一个容易混淆的编号问题：`20` 是 decoder 第一个 block 的零基代码索引，但该 block 的输入正是前 20 层结束后的 $H^{20}$。因此，论文公式说“从 $H^{L/2}$ 投影 KV”，代码则把相关 compressor 和 cache owner 挂在索引 20 的 attention 模块上，两者并不矛盾。

vLLM-Ascend 在以下位置将每一层解析为 `kv_source_layer`、`index_source_layer`、candidate source 等角色：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:496-545`。

长程 KV cache 只在 `is_kv_source` 为真的层上创建，consumer 层通过 source prefix 访问对应 cache：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:663-746`。

### 2. Causal Encoder 的主要作用

它的核心作用可以概括成一句话：

> 用前 20 层一次性把长 Prompt 转换成后 20 层所需要的全局记忆，从而避免长 Prompt 的每一个 token 都完整执行后 20 层。

#### 2.1 普通 Decoder-only Transformer

对于普通 40 层 decoder-only 模型，每一层通常从自己的 hidden state 产生自己的 K/V：

$$
K^{(l)}=H^{(l)}W_K^{(l)},\qquad
V^{(l)}=H^{(l)}W_V^{(l)}
$$

因此，长度为 $N$ 的 Prompt 做 prefill 时，大致需要执行 $N\times40$ 份 layer-token 计算；不同层通常还要保存各自的 KV cache：

```text
Prompt 的 N 个 token
       │
       ▼
Layer 1  ── 保存 KV1
       ▼
Layer 2  ── 保存 KV2
       ▼
      ...
       ▼
Layer 40 ── 保存 KV40
```

#### 2.2 CED 的做法

CED 首先让全部 Prompt token 经过前 20 层：

$$
X_{1:N}
\rightarrow H^{(1)}_{1:N}
\rightarrow \cdots
\rightarrow H^{(20)}_{1:N}
$$

随后，decoder 层的全局 KV 不再从各 decoder 层自己的 $H^{(l)}$ 生成，而是从 causal encoder 最后一层的输出 $H^{(20)}$ 投影得到。论文 Section 2.2 给出的公式为：

$$
C^{(l)}=H^{(20)}W_{KV}^{(l)},\qquad
Z^{(l)}=H^{(20)}W_Z^{(l)},\qquad l>20
$$

其中：

- $C^{(l)}$ 表示第 $l$ 个 decoder 层需要的全局 KV entry；
- $Z^{(l)}$ 表示相应的压缩权重；
- $W_{KV}^{(l)}$、$W_Z^{(l)}$ 仍然可以是 layer-dependent 的，所以不同 decoder 层并不是被迫使用完全相同的语义表示；
- 与 CSA2 结合后，并不是所有 decoder 层都物理保存一份独立的 $C^{(l)}$。Full 层产生主 KV，Reindex 和 Reuse 层继续共享这些 KV。

上游 vLLM 的代码关系如下：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:293-341`：解析每一层的 compression ratio、KV source 和 index source；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:563-593`：只在 KV source 层创建 compressor；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:895-945`：source 层的 compressor 直接投影进入该层的 hidden states；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:1100-1134`：consumer 层不分配自己的主 KV cache，而是读取 source cache。

整体数据流可简化为：

```text
                       完整 Prompt：N 个 token
                                │
                                ▼
                  ┌─────────────────────────┐
                  │ 20-layer Causal Encoder │
                  │ 每个位置只能看左侧 token │
                  └────────────┬────────────┘
                               │ H^(20)
                 ┌─────────────┴──────────────┐
                 │                            │
                 ▼                            ▼
      Layer-dependent projection       最近 n_win 个 token
       生成或压缩全局 KV                 做 SWA replay
                 │                            │
                 └─────────────┬──────────────┘
                               ▼
                     20-layer Decoder
                   Q：来自 decoder 当前层
              Global K/V：主要来自 H^(20)
              Local K/V：来自各层自己的 SWA
                               │
                               ▼
                          Next-token logits
```

所以，这个 causal encoder 实际上可以理解为一个“全局 KV 表示生产器”或“Prompt memory encoder”。

### 3. 为什么还要保留 Decoder 每层自己的 SWA

如果 decoder 的所有 K/V 都直接来自 $H^{20}$，后 20 层对局部上下文的逐层计算深度可能不足。因此，CED 做了一个折中：

- 长程、全局信息：来自 causal encoder 输出构造的 global KV；
- 短程、局部信息：每个 decoder 层仍然从自己的 hidden state 生成 SWA K/V。

对于 decoder 第 $l$ 层、位置 $t$，可以抽象为：

$$
Q_t^{(l)}=H_t^{(l)}W_Q^{(l)}
$$

全局部分为：

$$
K_{\text{global}}^{(l)},V_{\text{global}}^{(l)}
\leftarrow H^{(20)}
$$

局部部分为：

$$
K_{\text{local},t}^{(l)},V_{\text{local},t}^{(l)}
\leftarrow H_t^{(l)}
$$

最终 attention 同时使用两部分信息：

$$
\operatorname{Attention}\left(
Q_t^{(l)},
[K_{\text{global}}^{(l)};K_{\text{local}}^{(l)}],
[V_{\text{global}}^{(l)};V_{\text{local}}^{(l)}]
\right)
$$

这样，decoder 可以保留逐层局部推理能力，而不是退化成“后 20 层都读取同一份静态 memory”。

相应代价是：如果只持久化 global KV，那么重新进入 decoder 时，需要重建最近窗口的 layer-local SWA KV。这就是论文所说的 **Decoder SWA Bounded Replay**。

本地配置中的 SWA window 是 128。上游 vLLM 调度器在 prefix 命中场景中会把命中前缀的最后一个窗口标记为 replay 区间，用于重建 SWA KV，同时保持已经命中的主 KV 不变：

- `D:\Code\vllm-github\vllm\v1\core\sched\scheduler.py:3018-3030`。

### 4. CED 为什么能降低 Prefill 计算量

设：

- 总层数 $L=40$；
- Prompt 长度为 $N$；
- SWA window 为 $n_{\text{win}}$，本地配置中为 128；
- 且 $N\gg n_{\text{win}}$。

普通 40 层模型的主要 prefill 工作量近似为：

$$
C_{\text{normal}}=NL
$$

CED 的主要工作量近似为：

$$
C_{\text{CED}}
=N\frac{L}{2}
+n_{\text{win}}\frac{L}{2}
$$

第一项表示完整 Prompt 经过 20 层 causal encoder；第二项表示最近一个窗口经过 20 层 decoder，以重建各层局部 SWA 状态并得到最终输出。因此：

$$
C_{\text{CED}}
=20N+20n_{\text{win}}
\approx20N
$$

普通模型则约为 $40N$，所以在长 Prompt 下接近减少一半。

#### 4.1 100K Prompt 示例

假设：

$$
N=100000,\qquad n_{\text{win}}=128
$$

普通 40 层模型的简化 layer-token 工作量为：

$$
100000\times40=4,000,000
$$

CED 的简化工作量为：

$$
100000\times20+128\times20
=2,002,560
$$

简化估算的节省比例为：

$$
1-\frac{2,002,560}{4,000,000}
\approx49.94\%
$$

即接近减半。

这只是结构层面的 layer-token 估算，没有包含 compressor、稀疏索引、通信、MoE 路由和 kernel 利用率等实际开销，所以论文使用的是 “nearly halves”，而不是声称真实 wall-clock 时间一定严格减半。

论文和官方介绍将其概括为：Prefill 时每个普通输入 token 大致激活 8B backbone 参数，而 Decode 时新 token 激活 16B 参数。

### 5. Decode 时是否也只执行前 20 层

不是。CED 最主要节省的是长 Prompt 的 prefill，而不是把生成阶段永久削减成 20 层。

在 decode 阶段，每次只有一个或少量新 token：

1. 新 token 先经过 20 层 causal encoder，得到新的 $H_t^{20}$；
2. 根据 $H_t^{20}$ 更新相应的全局压缩 KV；
3. 新 token 再经过后 20 层 decoder；
4. decoder 使用已有 global KV、candidate pool、Top-K 索引以及各层 SWA cache；
5. 最终产生下一个 token 的 logits。

因此：

| 阶段 | 历史 Prompt token | 当前或新 token |
|---|---|---|
| Prefill | 大部分只完整经过前 20 层 | Prompt 尾部窗口还需要 decoder replay |
| Decode | 已经转换成 KV，不需要重新执行 | 新 token 依次经过全部 40 层 |

这也解释了论文中的：

- Prefill：每 token 激活约 8B 参数；
- Decode：每 token 激活约 16B 参数。

这不是在推理时随意砍掉一个普通 40 层模型的后半部分。模型从训练开始就采用了这种非对称信息流，decoder 被训练成消费 causal encoder 产生的 global KV。

### 6. 为什么它仍然属于 Decoder-only 自回归模型

关键在于，“encoder”这个词在这里与传统 seq2seq 模型中的含义不同。

#### 6.1 传统 Seq2Seq Encoder

以 T5 或原始 Transformer 翻译模型为例：

- Encoder 输入是源序列；
- Encoder attention 通常是双向的；
- Decoder 输入是另一个目标序列；
- Decoder 先做 masked self-attention，再通过 cross-attention 读取 encoder memory；
- 模型通常设置 `is_encoder_decoder=True`。

例如机器翻译：

```text
Encoder 输入："I love Beijing"
                       │
                       ▼
                双向 source memory
                       │ cross-attention
                       ▼
Decoder 输出："我 爱 北京"
```

在传统双向 Encoder 中，`I` 的表示可以使用后面的 `Beijing` 信息。

#### 6.2 CED 的 Causal Encoder

DeepSeek-V4.1-Flash 中：

- causal encoder 和 decoder 处理的是同一条连续 token 序列；
- causal encoder 不能看到未来；
- 不存在独立的 source sequence 和 target sequence；
- 最终训练目标仍然是 next-token prediction；
- 对外接口仍然是 `ForCausalLM`；
- decoder 的 global KV 虽然来自 encoder 输出，但仍然受相同的因果位置约束。

两者的区别如下：

| 属性 | 传统 Encoder-Decoder | DeepSeek CED |
|---|---|---|
| Encoder mask | 通常双向 | 严格 causal |
| 输入序列 | Source 与 Target 分开 | 同一条自回归序列 |
| Decoder 读取内容 | 独立 source memory | 前 20 层产生的 causal global KV |
| 是否可以看到未来 token | Encoder 通常可以 | 不可以 |
| 常见训练目标 | Conditional seq2seq | Next-token prediction |
| 外部模型接口 | Seq2Seq LM | Causal LM |
| 主要设计目的 | 输入输出序列转换 | 降低 Prefill 和 KV cache 成本 |

上游 vLLM 的配置展开 `text_config` 时，特意避免覆盖 `PreTrainedConfig.is_encoder_decoder`：

- `D:\Code\vllm-github\vllm\transformers_utils\configs\deepseek_v41.py:34-42`。

模型类则明确注册为：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:1302`，`DeepseekV41LLMForCausalLM`。

因此，从 Hugging Face/vLLM 的任务分类和外部行为看，它仍然是 Causal LM，而不是经典 Encoder-Decoder LM。

### 7. 为什么称为 Causal Encoder

“Encoder”强调它在 CED 内部承担的功能：

> 把完整的因果前缀编码成一种共享、可压缩、可供后续 decoder 层消费的全局记忆。

“Causal”则强调：

> 这种编码没有使用未来信息，因此不会破坏自回归生成。

可以用一个简单的归纳关系说明这一点。输入 embedding 为：

$$
H_t^{(0)}=f(x_t)
$$

假设第 $l-1$ 层的 $H_t^{(l-1)}$ 只依赖 $x_{\le t}$，第 $l$ 层使用 causal attention，只访问 $j\le t$，则：

$$
H_t^{(l)}
=F\left(H_{\le t}^{(l-1)}\right)
$$

仍然只依赖 $x_{\le t}$。

于是 encoder 输出 $H_j^{(20)}$ 只包含 $x_{\le j}$ 的信息。decoder global KV：

$$
K_j^{(l)},V_j^{(l)}=g_l(H_j^{(20)})
$$

同样没有未来信息。decoder 在位置 $t$ 又只允许选择 $j\le t$ 的 KV，因此最终 logits 仍满足：

$$
p(x_{t+1}\mid x_{\le t})
$$

这就是它依然属于自回归语言模型的根本原因。

### 8. CED 与 YOCO 的关系

论文明确说明 CED 受到 YOCO（You Only Cache Once）启发。

YOCO 原论文把自身称为 **decoder-decoder architecture**：

- 下半部分 self-decoder 产生一次全局 KV；
- 上半部分 cross-decoder 复用这些 KV；
- 从外部看仍然表现为 decoder-only language model。

YOCO 原论文：<https://arxiv.org/abs/2405.05254>

DeepSeek CED 可以理解为沿这个方向做了进一步改造：

- YOCO 强调 global KV “只缓存一次”以及 prefill early exit；
- DeepSeek CED 保留 decoder 各层的 local SWA；
- 使用 $H^{20}$ 和 layer-dependent projection 构造 decoder global KV；
- 结合 CSA2 做跨层主 KV 和 Top-K 索引复用；
- 使用 SWA Bounded Replay 控制局部状态重建成本；
- 再结合 FP4 main KV cache 降低存储开销。

因此，如果 “causal encoder-decoder” 这个名称容易让人联想到 T5，可以把它理解为：

> **共享全局 KV 的两段式 causal decoder**

或者：

> **前半段负责 Prompt 编码、后半段负责深层生成的非对称 decoder-only 模型**

这两种表述更接近它的真实工作方式。

### 9. 为什么代码中的层仍叫 DecoderLayer

这是代码复用命名与论文语义分层不同造成的。

上游 vLLM 使用同一个 `DeepseekV4DecoderLayer` 类创建全部 `config.num_hidden_layers` 个 block：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:717-732`。

这并不表示论文语义上的前 20 层不是 causal encoder。CED 的角色差别主要编码在以下配置和运行关系中：

- `compress_ratios`；
- `kv_source_layer_ids`；
- `index_source_layer_ids`；
- `candidate_source_layer_id`；
- compressor 和 cache 的实际 ownership；
- source/consumer cache 的共享关系；
- SWA replay 元数据。

代码没有必要为两段分别定义：

```python
CausalEncoderLayer
DecoderLayer
```

所以阅读实现时，不能只根据 Python 类名判断一层在论文中的角色，而应结合 layer index、compression ratio 和 KV/cache topology 判断。

### 最终总结

论文中的 **20-layer causal encoder** 是 DeepSeek-V4.1-Flash 40 层自回归语言主干的前 20 层。它对完整 Prompt 做因果编码，输出 $H^{20}$，并以此构造、压缩和共享后 20 层 decoder 所需要的全局 KV cache。

它的主要作用是：

1. 让长 Prompt 的大部分 token 不必完整执行后 20 层；
2. 将 prefill 的主要复杂度从 $O(NL)$ 降到约 $O(NL/2)$；
3. 为 decoder 提供统一的全局上下文表示；
4. 配合 CSA2 跨层共享 KV 与稀疏 Top-K 索引；
5. 通过仅回放最后 $n_{\text{win}}$ 个 token，保留 decoder 各层的局部 SWA 计算深度。

它之所以不违背“现在的推理模型主要采用 decoder-only 架构”的说法，是因为这个 encoder：

- 仍然使用 causal mask；
- 处理的不是独立 source sequence；
- 不能看到未来 token；
- 仍然使用 next-token prediction；
- 模型对外仍然是 `ForCausalLM`；
- 本质上是 decoder-only 模型内部的一种非对称、两阶段 KV 生成结构，而不是传统的双向 Encoder 加 Cross-Attention Decoder。

---

## 问题 2：普通 Decoder-only 与 CED 的 Prefill、Decode 计算量

### 问题 2 原文

> 对于普通 40 层 decoder-only 模型，其 prefill 和 decode 计算过程是怎样的？假如全部的 prompt 在 prefill 阶段经过了 40 层计算后，生成了第一个 token，后面每生成一个新 token，都要和前面的 prompt 再经过这 40 层网络再生成一个 token 吗？因此对于公式 $C_{\text{normal}}=NL$，如果输入序列 100K，生成序列 1K，不开 MTP，那就要有 $N\times[100K+(1+2+\cdots+1K)]$ 的计算吗？而对于论文中的 CED 实现，就有公式 $C_{\text{CED}}=N\frac{L}{2}+n_{\text{win}}\frac{L}{2}$，这时候的总计算是多少呢？

### 问题 2 结论

这里最关键的修正是：

> 开启 KV Cache 后，普通 decoder-only 模型在 Decode 阶段不会让“全部 Prompt 加已生成 token”重新经过 40 层网络。

实际过程是：

- Prefill：全部 Prompt token 经过 40 层，并保存每一层的 KV Cache；
- 第一个输出 token：直接从 Prompt 最后一个位置的 logits 采样得到；
- 后续 Decode：每一步只有最新的 1 个 token 经过 40 层；
- 新 token 在每一层 attention 中读取历史 KV Cache，但不会重新计算历史 token 的 Q/K/V、MLP 和 hidden state。

所以，对于 100K Prompt、生成 1K token、40 层且不开 MTP 的普通模型，“token 穿层次数”近似为：

$$
40\times(100K+1K)
$$

更精确地考虑第一个输出 token 已经由 Prefill 产生，则是：

$$
40\times[100K+(1K-1)]
$$

三角和并非完全没有意义，但它描述的是 Decode attention 中“新 Query 读取多少历史 KV”，而不是“多少历史 token 重新执行全部 40 层”。

### 1. 统一符号与计算口径

为避免论文中的 $N$ 和问题中的 $N$ 含义混淆，统一使用：

| 符号 | 含义 | 示例 |
|---|---|---:|
| $L$ | Transformer 层数 | 40 |
| $P$ | Prompt token 数 | 100,000 |
| $G$ | 最终生成 token 数 | 1,000 |
| $W$ | SWA replay window | 128 |
| $d$ | hidden dimension | 未指定 |

论文 Section 2.2 中的 $O(NL)$ 里，$N$ 是输入序列长度，相当于这里的 $P$；$L$ 才是模型层数。

此外，$PL$ 统计的是简化的 layer-token 数，即“多少 token 穿过多少个 Transformer block”，并不是完整 FLOPs。它没有显式计入：

- Dense attention 的 $P^2$ 项；
- hidden size $d$；
- Q/K/V/O projection；
- MLP 或 MoE；
- Top-K indexer；
- 通信、量化和 kernel 利用率。

### 2. 普通模型的 Prefill

假设 Prompt 为：

```text
[p1, p2, p3, ..., p100000]
```

Prefill 时，这 100K 个 token 会经过全部 40 层：

```text
100K Prompt tokens
        │
        ▼
┌─────────────────┐
│ Transformer L1  │ ── 保存 K1、V1
├─────────────────┤
│ Transformer L2  │ ── 保存 K2、V2
├─────────────────┤
│       ...       │
├─────────────────┤
│ Transformer L40 │ ── 保存 K40、V40
└─────────────────┘
        │
        ▼
最后一个 Prompt 位置的 logits
        │
        ▼
采样得到第一个输出 token y1
```

在第 $l$ 层，Prompt token 会产生：

$$
Q^{(l)}=H^{(l)}W_Q^{(l)}
$$

$$
K^{(l)}=H^{(l)}W_K^{(l)},\qquad
V^{(l)}=H^{(l)}W_V^{(l)}
$$

其中每层的 $K^{(l)},V^{(l)}$ 会写入 KV Cache。因此，忽略 attention 内部的二次项，Prefill 的 layer-token 数为：

$$
C_{\text{normal,prefill}}=PL
$$

代入 $P=100000,L=40$：

$$
C_{\text{normal,prefill}}
=100000\times40
=\boxed{4,000,000}
$$

Prefill 在数学上是因果的，但在 GPU/NPU 上通常不是逐 token 串行运行。一个 Prompt 的大量 token 会在同一层中并行处理，因果 mask 只负责限制位置 $i$ 不能看到 $j>i$。

### 3. 普通模型的 Decode 与 KV Cache

Prefill 结束后，每一层已经缓存了 Prompt 的 K/V：

```text
Layer 1 KV Cache:  K1[p1...p100000],  V1[p1...p100000]
Layer 2 KV Cache:  K2[p1...p100000],  V2[p1...p100000]
...
Layer 40 KV Cache: K40[p1...p100000], V40[p1...p100000]
```

Prompt 最后一个位置的 logits 用于采样 $y_1$。为了产生 $y_2$，只把 $y_1$ 送入模型：

```text
输入：y1，一个 token
  │
  ▼
Layer 1:
  计算 y1 的 Q1/K1/V1
  Q1(y1) 读取历史 K1[prompt]
  把 K1(y1), V1(y1) 追加到 Layer 1 KV Cache
  │
  ▼
Layer 2:
  计算 y1 在第 2 层的 Q2/K2/V2
  Q2(y1) 读取历史 K2[prompt]
  把 K2(y1), V2(y1) 追加到 Layer 2 KV Cache
  │
 ...
  ▼
Layer 40 → logits → 采样 y2
```

为了产生 $y_3$，只把 $y_2$ 送入模型，读取 `[Prompt KV, y1 KV]`，再得到 $y_3$。

例如 Prompt 有 5 个 token，要生成 3 个 token：

| 阶段 | 实际穿过 40 层的 token | 读取的历史 KV | 得到 |
|---|---|---|---|
| Prefill | `p1...p5` | 在 Prefill 内部生成 | $y_1$ |
| Decode 1 | 只有 $y_1$ | `p1...p5` | $y_2$ |
| Decode 2 | 只有 $y_2$ | `p1...p5,y1` | $y_3$ |

历史 token 不需要重新计算，是因为 causal Transformer 中过去位置的 hidden state 不会被未来 token 改变：

$$
H_i^{(l)}=F(H_{\le i}^{(l-1)})
$$

加入位置 $t>i$ 后，位置 $i$ 仍不能看到 $t$，所以原有的 $K_i^{(l)},V_i^{(l)}$ 仍然有效。

### 4. vLLM 代码对应关系

#### 4.1 调度器只调度尚未计算的 token

上游 vLLM 在：

`D:\Code\vllm-github\vllm\v1\core\sched\scheduler.py:683-700`

计算：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

其中：

- `num_computed_tokens` 是已经计算、已有 KV 的 token 数；
- 二者相减得到当前真正需要送入模型的新 token 数；
- 普通 Decode、不开 speculative decoding 时，稳定阶段通常是 `num_new_tokens = 1`。

vLLM-Ascend 中采用相同逻辑：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\core\dyntra_lb_scheduler.py:503-515`；
- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\core\dyntra_lb_scheduler.py:845-878`。

#### 4.2 新 token 仍然经过全部层

普通 Llama 风格模型创建 `config.num_hidden_layers` 个 `LlamaDecoderLayer`：

- `D:\Code\vllm-github\vllm\model_executor\models\llama.py:389-393`。

当前调度进来的 hidden states 依次经过所有层：

- `D:\Code\vllm-github\vllm\model_executor\models\llama.py:432-440`。

Decode 时，对单个请求，这里的 token 维通常只有 1，而不是整个 100K Prompt。

#### 4.3 新 K/V 被追加到 Cache

vLLM 的 attention layer 将 KV Cache 保存在 `self.kv_cache` 中，并使用当前新 token 的 K/V 更新指定 slot：

- `D:\Code\vllm-github\vllm\model_executor\layers\attention\attention.py:488-578`；
- `D:\Code\vllm-github\vllm\model_executor\layers\attention\attention.py:724-744`。

然后 attention backend 接收当前 query 和历史 `kv_cache` 执行 attention：

- `D:\Code\vllm-github\vllm\model_executor\layers\attention\attention.py:767-793`。

这对应的是“只计算新 token 的 Q/K/V，同时读取历史 token 的 KV Cache”，而不是重新执行历史 token 的整个 Transformer forward。

### 5. 普通模型总计算量

Prefill 已经根据最后一个 Prompt token 的 hidden state 得到第一个输出 token $y_1$。因此，为了最终得到 $G$ 个输出 token，后续只需执行 $G-1$ 次 Decode forward：

$$
C_{\text{normal,block}}
=PL+(G-1)L
=L(P+G-1)
$$

代入：

$$
L=40,\quad P=100000,\quad G=1000
$$

得到：

$$
C_{\text{normal,block}}
=40(100000+999)
=\boxed{4,039,960}
$$

| 部分 | Layer-token 次数 |
|---|---:|
| Prefill | $100000\times40=4,000,000$ |
| Decode | $999\times40=39,960$ |
| 合计 | $\boxed{4,039,960}$ |

如果使用常见近似，把每个输出 token 都记成一个 Decode step：

$$
C_{\text{normal,block}}\approx L(P+G)
$$

则为：

$$
40(100000+1000)=4,040,000
$$

与精确值只差一个 token 穿过 40 层的 40 次 layer-token。

### 6. 三角和出现在哪里

三角和主要出现在 Decode attention 读取的 KV 长度中，而不是完整 Transformer block 的重复执行次数中。

为了得到第 2～$G$ 个 token，每层 Decode attention 读取的历史 KV 总长度近似为：

$$
\sum_{i=1}^{G-1}(P+i)
=(G-1)P+\frac{G(G-1)}{2}
$$

代入 $P=100000,G=1000$：

$$
999\times100000+\frac{1000\times999}{2}
$$

$$
=99,900,000+499,500
=\boxed{100,399,500}
$$

这是每一层大致处理的 query-key 位置对数量。40 层约为：

$$
40\times100,399,500
=4,015,980,000
$$

但它表示的是 999 个新 Query 分别与缓存中的历史 K/V 做 attention，而不是 100,399,500 个历史 token 全部重新执行 QKV projection、MLP、LayerNorm 或 MoE。

| 统计维度 | 使用 KV Cache 的普通 Decode |
|---|---|
| 新 token 的 QKV projection | 每步 1 个 token |
| 新 token 的 MLP/MoE | 每步 1 个 token |
| 新 token 穿过 Transformer 层 | 每步 $L$ 次 |
| 历史 token 的 QKV projection | 不重算 |
| 历史 token 的 MLP/MoE | 不重算 |
| 历史 K/V | 从 Cache 读取 |
| Attention 读取长度 | 随上下文从 $P$ 增长到 $P+G$ |

### 7. 完全关闭 KV Cache 的计算量

如果完全不用 KV Cache，才需要为每个输出 token 重跑整个当前上下文：

$$
P+(P+1)+\cdots+(P+G-1)
$$

总 layer-token 数为：

$$
C_{\text{no-cache}}
=L\left[GP+\frac{G(G-1)}{2}\right]
$$

代入：

$$
C_{\text{no-cache}}
=40\left[1000\times100000+\frac{1000\times999}{2}\right]
$$

$$
=40(100,000,000+499,500)
=\boxed{4,019,980,000}
$$

对比：

| 场景 | Layer-token 次数 |
|---|---:|
| 正常使用 KV Cache | 4,039,960 |
| 完全不用 KV Cache | 4,019,980,000 |

后者约是前者的 995 倍。问题中原先写的：

$$
L\left[P+(1+2+\cdots+G)\right]
$$

既不是正常 KV Cache 情况，也不是完全无 KV Cache 情况，因为它遗漏了无 Cache 时每一步都会重复处理的 $P$。

### 8. Dense Attention 的二次复杂度

论文的 $C_{\text{normal}}=PL$ 只统计 token 穿过 block 的数量。对于普通 dense causal attention，单层 Prefill 的 query-key 位置对数量是：

$$
1+2+\cdots+P
=\frac{P(P+1)}{2}
$$

当 $P=100000$：

$$
\frac{100000\times100001}{2}
=5,000,050,000
$$

每层约 50 亿个因果 query-key 位置对；40 层约为：

$$
200,002,000,000
$$

更完整的普通 dense Transformer 复杂度应区分：

$$
C_{\text{proj+MLP,prefill}}=O(LPd^2)
$$

$$
C_{\text{attention,prefill}}=O(LP^2d)
$$

DeepSeek-V4.1-Flash 使用 CSA2、压缩主 KV、Top-K 稀疏选择和 SWA，因此不能把它的 attention FLOPs 直接按普通 dense attention 的 $P^2$ 估算。

### 9. 论文 CED 的 Prefill 与总计算量

论文 Section 2.2 给出的 Prefill 简化公式是：

$$
C_{\text{CED,prefill}}
=P\frac{L}{2}+W\frac{L}{2}
$$

第一项表示全部 $P$ 个 Prompt token 经过前 $L/2$ 层 causal encoder；第二项表示仅 Prompt 最后的 $W$ 个 token 经过后 $L/2$ 层 decoder，用来重建各层 SWA KV。

当 $P\gg W$：

$$
C_{\text{CED,prefill}}\approx\frac{PL}{2}
$$

代入 $P=100000,L=40,W=128$：

$$
C_{\text{CED,prefill}}
=100000\times20+128\times20
$$

$$
=2,000,000+2,560
=\boxed{2,002,560}
$$

普通模型 Prefill 是 4,000,000，因此 CED Prefill 的 layer-token 数为普通模型的 50.064%，即减少约 49.936%。

#### 9.1 加上 1K 输出的 Decode

论文公式只描述 Prefill，不包含 Decode。CED 在 Decode 阶段，每个新 token 仍然经过全部 40 层，因此总量为：

$$
C_{\text{CED,total}}
=P\frac{L}{2}+W\frac{L}{2}+(G-1)L
$$

代入：

$$
C_{\text{CED,total}}
=100000\times20+128\times20+999\times40
$$

$$
=2,000,000+2,560+39,960
=\boxed{2,042,520}
$$

如果使用 $G$ 次 Decode 的近似计数，则为：

$$
2,000,000+2,560+40,000
=\boxed{2,042,560}
$$

比较如下：

| 模型 | Prefill | Decode 999 步 | 总 layer-token |
|---|---:|---:|---:|
| 普通 40 层模型 | 4,000,000 | 39,960 | 4,039,960 |
| 论文理想 CED | 2,002,560 | 39,960 | 2,042,520 |

总量比例为：

$$
\frac{2,042,520}{4,039,960}\approx50.56\%
$$

即简化 layer-token 数减少约 49.44%。端到端节省比例略低于 Prefill，是因为两种模型在 Decode 阶段都会让新 token 经过全部 40 层。

CED 的完整阶段可以画成：

```text
阶段 A：Causal Encoder Prefill
100K Prompt → 前 20 层全部计算 → 得到全部位置 H^(20)
                                      │
                                      └─ 构造 Decoder Global KV

阶段 B：Decoder SWA Bounded Replay
Prompt 最后 128 个位置 → 后 20 层 → 重建 Decoder SWA KV → logits → y1

阶段 C：Autoregressive Decode
y1   → 全部 40 层 → y2
y2   → 全部 40 层 → y3
...
y999 → 全部 40 层 → y1000
```

论文 Section 3.2.2 还说明，Decoder SWA Bounded Replay 得到的是近似状态，不一定与后 20 层覆盖整个 Prompt 完整执行时的 SWA KV 数学等价；论文通过在 post-training 中模拟相同 replay 做 train-aware adaptation，将质量损失控制在较小范围。

### 10. 本地 CUDA 与 Ascend 实现差异

#### 10.1 上游 vLLM CUDA 实现

`D:\Code\vllm-github` 已实现专门的 `DecoderReplayLayers`：最后一个 KV source 之后的层，在 eager prefill 中只运行每个请求最后 `window` 行：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\decoder_replay_layers.py:3-12`；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\decoder_replay_layers.py:45-67`。

模型使用：

```python
cut = max(config.kv_source_layer_ids)
```

当前配置中 `kv_source_layer_ids=[2,8,14,20]`，所以 `cut=20`。代码实际执行：

- 索引 0～20：完整 Prefill；
- 索引 21～39：只对尾部 window replay。

对应位置：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:757-784`；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:955-978`。

因此当前 CUDA 代码按静态 layer-row 数量更接近：

$$
C_{\text{vLLM-CUDA,prefill}}=21P+19W
$$

代入：

$$
21\times100000+19\times128
=2,100,000+2,432
=\boxed{2,102,432}
$$

加上 999 次完整 40 层 Decode：

$$
C_{\text{vLLM-CUDA,total}}
=2,102,432+39,960
=\boxed{2,142,392}
$$

约为普通模型的 53.03%，即按这一简化计数减少约 46.97%。多出的一层来自实现边界：索引 20 是最后一个 global KV source 和 candidate source，当前实现把它放在完整 Prefill 段，索引 21～39 才进入尾部 replay。

此外，CUDA replay trimming 有启用条件。在以下场景中可能关闭：

- Pipeline stage 没有同时持有最后 KV source 和全部后续层；
- 使用 sequence parallel；
- `prefill_context_parallel_size > 1`；
- 使用 ubatching；
- 最后 KV source 后还有 Engram 层；
- draft model 的 sliding-window 或 hidden-state 需求不兼容。

相关判断位于：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:1146-1183`。

而且当前 `model_state.py:403-435` 表明只有 eager step 会真正裁剪；CUDA Graph step 保持完整 batch。因此论文公式代表架构目标和优化后的理想执行结构，不能在不检查运行配置的情况下直接等同于某次部署的实测计算量。

#### 10.2 当前 vLLM-Ascend 静态代码

本地 vLLM-Ascend 已实现：

- 40 层 V4.1 topology；
- `compress_ratios`；
- `kv_source_layer_ids`；
- `index_source_layer_ids`；
- CSA2 KV source/consumer 共享；
- candidate pool；
- 独立 SWA cache。

但当前 Python forward 仍表现为对 `self.layers` 逐层循环：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:1224-1255`。

在本次检索范围内，没有发现与上游 CUDA `DecoderReplayLayers` 对等的、在模型层之间裁剪 token rows 的实现。因此需要区分：

| 结论 | 状态 |
|---|---|
| CED/CSA2 的 KV source/consumer 拓扑 | Ascend 代码已实现 |
| 论文的 $20P+20W$ 复杂度 | 论文架构目标 |
| CUDA 尾部 replay row trimming | 上游 vLLM 有条件实现 |
| 当前 Ascend 是否有等价 row trimming | 静态 `model.py` 尚不能确认，显式循环仍处理全部层 |
| Ascend 实测是否接近 8B Prefill | 需要 profiler/kernel shape 验证 |

验证 Ascend 路径时，应观察第 21～40 层 MoE/attention kernel 的 token 第一维：

- 若前半层约为 100K、后半层约为 128，说明 decoder replay trimming 生效；
- 若 40 层都是约 100K，说明 KV cache 虽然已经共享和压缩，但论文所说的近半 Prefill block 计算节省没有在该执行路径完全实现。

### 问题 2 最终总结

1. 普通模型开启 KV Cache 后，Decode 不会重新计算全部 Prompt。每一步只有最新 token 经过 40 层，历史 token 以 KV Cache 形式被 attention 读取。
2. 普通 40 层模型、100K 输入、1K 输出的精确简化计算量为：

   $$
   \boxed{C_{\text{normal}}=40(100000+999)=4,039,960}
   $$

3. 三角和对应 Decode attention 累积读取的历史 KV 长度：

   $$
   \boxed{(G-1)P+\frac{G(G-1)}{2}}
   $$

   它不代表历史 token 重新经过全部网络。
4. 完全关闭 KV Cache 才会产生：

   $$
   \boxed{C_{\text{no-cache}}=L\left[GP+\frac{G(G-1)}{2}\right]}
   $$

5. 论文理想 CED 的 Prefill 为：

   $$
   \boxed{C_{\text{CED,prefill}}=100000\times20+128\times20=2,002,560}
   $$

6. 加上 999 次完整 40 层 Decode，论文理想 CED 总量为：

   $$
   \boxed{C_{\text{CED,total}}=2,042,520}
   $$

   相较普通模型的 4,039,960，简化 layer-token 数约减少 49.44%。
7. 论文公式、上游 CUDA 的有条件 replay trimming、当前 Ascend 静态实现需要分别看待；最终是否达到该计算量，应以运行配置和 profiler 中各层实际 token shape 为准。

---

## 问题 3：Backbone、Engram 与每 Token 激活参数

### 问题 3 原文

> “DeepSeek-V4.1-Flash has 552B backbone parameters and 196B Engram parameters, activating 8B parameters per token during prefill and 16B during decode.”这句话中，backbone parameters、Engram parameters 是什么意思？每个 token 有 8B 的激活参数，又该怎么解释？

### 问题 3 核心结论

这句话里的三个数字属于两个不同维度：

| 数字 | 表示什么 | 是否每个 token 都会全部参与计算 |
|---:|---|---|
| 552B backbone parameters | 主干网络的总参数容量，主要是 40 层 Transformer 中的 Attention、MoE、mHC、Embedding 等 | 否。MoE 中每层 384 个 routed expert 只选择 6 个 |
| 196B Engram parameters | 两个超大的条件记忆查找表及其相关投影参数 | 否。每个 token 只查出极少数表项 |
| 8B / 16B activated parameters | 一个 token 在一次前向计算中实际使用到的主干参数规模 | Prefill 约 8B，Decode 约 16B |

最重要的两点是：

1. `activated parameters` 不是激活张量占用，也不是显存占用，而是“该 token 的计算路径实际经过了多少参数”。
2. 8B 与 16B 的约 2 倍关系，主要来自 CED：大部分 Prompt token 在 Prefill 时只完整经过前 20 层；Decode 的新 token 要经过完整 40 层。

论文依据主要位于本地论文的第 7、13、22、24 页：

`D:\Code\vllm_notes\DeepSeekV41-Flash\DeepSeek-V4.1-Flash Pushing the Limits of KV Cache Compression.pdf`

### 1. 什么是 Backbone Parameters

这里的 backbone 可以理解成：

> 模型执行语言理解、推理和生成的主要神经网络计算骨架。

DeepSeek-V4.1-Flash 的语言 backbone 是 40 层因果 Transformer：

```text
输入 Embedding
      │
      ▼
┌──────────────────────┐
│ 20 层 Causal Encoder │
│ Attention + MoE      │
└──────────────────────┘
      │
      ▼
┌──────────────────────┐
│ 20 层 Decoder        │
│ Attention + MoE      │
└──────────────────────┘
      │
      ▼
   LM Head
```

Backbone 参数主要包括：

- 40 层 Transformer 中的注意力投影参数；
- 每层的 MoE routed experts；
- 每层的 shared expert；
- MoE router；
- Single-Pass mHC 参数；
- 输入 Embedding、输出预测头以及归一化参数等。

它不等于“552B 个参数全部是普通 Dense FFN”。实际上，552B 中绝大部分来自 MoE 的 384 个 routed experts。

#### 1.1 552B 为什么可以这么大

论文给出的 MoE 配置是：

- 40 个 Transformer block；
- 每层 384 个 routed experts；
- 每个 token 每层只选择 6 个；
- 每层还有 1 个 shared expert；
- hidden size 为 5120；
- 每个 expert 的中间维度为 2304；
- expert 使用 SwiGLU。

一个 SwiGLU expert 通常有三组主要矩阵：

$$
W_{\text{gate}},W_{\text{up}}\in
\mathbb{R}^{5120\times2304},
\qquad
W_{\text{down}}\in
\mathbb{R}^{2304\times5120}
$$

所以一个 expert 大约有：

$$
P_{\text{expert}}
=3\times5120\times2304
=35,389,440
\approx35.39M
$$

仅全部 routed experts 就约为：

$$
40\times384\times35.39M
\approx543.58B
$$

这已经非常接近论文报告的 552B。剩余参数来自 shared experts、Attention、mHC、Embedding、router 等。由于论文的 552B 是整体四舍五入口径，不能简单依靠上述近似公式精确反推出每一项，但可以清楚看出：

> 552B 的主体是“所有可能被路由到的专家参数总和”。

Ascend 实现中，模型从配置读取 routed/shared expert 数量：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:161-162`

构造 MoE 时将 `num_experts` 设置为 `n_routed_experts`，将 `top_k` 设置为 `num_experts_per_tok`：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:226-249`

上游 vLLM 的 NVIDIA 实现也直接读取：

```python
n_routed_experts = config.n_routed_experts
n_activated_experts = config.num_experts_per_tok
```

对应代码位置：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:124-148`

官方 checkpoint 配置明确给出：

```json
{
  "n_layers": 40,
  "dim": 5120,
  "moe_inter_dim": 2304,
  "n_routed_experts": 384,
  "n_shared_experts": 1,
  "n_activated_experts": 6
}
```

#### 1.2 Backbone Parameters 不等于 Active Parameters

假设某一层有 384 个专家：

```text
                     ┌─ Expert 3
                     ├─ Expert 17
当前 token ─ Router ─┼─ Expert 62
                     ├─ Expert 105     实际选择 6 个
                     ├─ Expert 201
                     └─ Expert 344

另外 378 个 routed experts
本 token 不执行其矩阵乘法
```

模型容量仍然拥有全部 384 个专家，因此它们都计入 552B；但当前 token 只经过其中 6 个，因此只将被选中的专家计入 activated parameters。

路由比例为：

$$
\frac{6}{384}=\frac{1}{64}
$$

但是不能直接认为：

$$
552B/64=8.625B
$$

因为不是所有参数都受 MoE 路由控制：

- Attention 参数是每个 token 都会使用的；
- mHC 参数是共享执行的；
- shared expert 每个 token 都要执行；
- Embedding、Normalization、router 等也有自己的计数方式；
- 不同组件对 activated parameters 的统计约定可能不同。

### 2. 什么是 Engram Parameters

Engram 不是另外一套 Transformer，也不是另外 196B 的普通 Dense 层。它是一种：

> 基于局部 N-gram 的超大条件记忆表，通过哈希查表提取少量记忆，再用当前上下文对记忆进行门控融合。

可以把它类比成一个巨大的“可学习短语知识库”：

```text
输入 token 序列
   │
   ├─ tokenizer 归一化与压缩
   │
   ├─ 构造最近的 2-gram、3-gram、4-gram
   │
   ├─ 每种 N-gram 使用 8 个独立 hash head
   │
   ├─ 从巨型 Engram 表中读取少量行
   │
   ├─ 投影成 key/value
   │
   ├─ 根据当前隐藏状态计算 gate
   │
   └─ hidden_state + gate × Engram_value
```

vLLM 文档将 Engram 描述为 conditional memory：它不是像 MoE expert 那样对一大块权重执行矩阵乘法，而是将局部 token 上下文哈希到大表中的少数行，再将查出的结果融合到隐藏状态：

- `D:\Code\vllm-github\docs\features\engram.md:3-9`
- `D:\Code\vllm-github\docs\features\engram.md:29-58`

#### 2.1 196B 是怎么来的

论文和 checkpoint 的配置是：

- 共有两个 Engram 模块；
- 位于第 1、14 层，使用 zero-based 编号；
- 每个模块处理 2-gram、3-gram、4-gram；
- 每个 N-gram order 有 8 个 hash head；
- 每个 head 的向量维度为 256；
- 因而每个 N-gram order 的总维度是：

$$
8\times256=2048
$$

三种 N-gram 合起来，每个 token 查出的向量大小为：

$$
3\times8\times256=6144
$$

checkpoint 中两个 Engram 表的总行数分别为：

$$
384,006,168,\qquad384,016,682
$$

每一行有 256 个参数，因此表参数总量为：

$$
384,006,168\times256
=98,305,579,008
$$

$$
384,016,682\times256
=98,308,270,592
$$

合计：

$$
\boxed{
196,613,849,600
\approx196.6B
}
$$

这就是论文“196B Engram parameters”的主要来源。

vLLM 从配置读取 `engram_num_embeddings`、`engram_n_heads`、`engram_head_dim` 和最大 N-gram 阶数，并计算：

$$
n_{\text{hash columns}}
=(n_{\max}-1)\times n_{\text{heads}}
=3\times8=24
$$

对应代码位置：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\common\engram.py:197-235`

Ascend 代码使用下面的结构创建两个大表：

```python
AscendParallelEngramEmbedding(
    rows,
    config.engram_head_dim,
    ...
)
```

对应：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:1028-1040`

#### 2.2 为什么 196B 很大，计算却可以很小

虽然 Engram 总共有约 196.6B 个表参数，但对于一个普通文本 token，每个 Engram 模块只读取：

$$
3\text{种 N-gram}\times8\text{个 head}
=24\text{行}
$$

每行 256 个值，因此每个模块只取出：

$$
24\times256=6144
$$

两个模块合计约：

$$
2\times6144=12,288
$$

个表元素。

也就是说：

```text
Engram 总容量：约 196,600,000,000 个表参数
单 token 查表：约 12,288 个表元素
```

二者相差约七个数量级。因此 Engram 可以显著扩大模型的“记忆容量”，却不需要让每个 token 对 196B 权重执行矩阵乘法。

上游代码先为整个 token 流生成 N-gram hash：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:878-921`

然后在进入 Transformer 层之前预取相应 Engram 行：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:926-937`

查出的 6144 维向量再通过 `wkv` 投影为 key/value，并通过门控注入 residual stream：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\common\engram.py:1314-1389`

Ascend 实现中的投影与门控过程位于：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\engram\layer.py:11-45`

其核心过程是：

```python
kv = self.wkv(rows)
key, value = kv.split(...)
return engram_gate(...)
```

#### 2.3 Engram 与 MoE 的差异

| 对比项 | MoE expert | Engram |
|---|---|---|
| 稀疏维度 | 在多个计算专家中选择少数专家 | 在巨型记忆表中选择少数行 |
| 核心操作 | 大规模矩阵乘法 | N-gram 哈希、查表、小型投影和门控 |
| 扩大什么 | 条件计算容量 | 条件记忆容量 |
| 单 token 访问 | 每层 384 选 6 | 每模块通常查 24 行 |
| 是否计入 552B backbone | 是 | 否，论文单独报告为 196B |
| 是否全部常驻设备 HBM | 通常需要分片部署 | 可以放在 Host Memory，按需预取 |

vLLM 默认可以把 Engram 表放在 pinned host memory，通过 UVA 和异步预取与前面网络层的计算重叠：

- `D:\Code\vllm-github\docs\features\engram.md:74-84`

这也说明：

> 参数属于模型，不代表这些参数必须同时常驻一张 GPU，也不代表每个 token 都读取全部参数。

### 3. “每个 Token 激活 8B 参数”是什么意思

更严格地说，它表示：

> 针对一个 token 的一次目标模型前向过程，将该 token 实际经过的 Attention、MoE 选中专家、shared expert、mHC 等模块的参数量相加，得到约 8B 或 16B。

可以写成概念公式：

$$
P_{\text{active/token}}
=
P_{\text{always-active}}
+
\sum_{l\in\text{executed layers}}
\left(
P_{\text{shared expert},l}
+
\sum_{e\in\operatorname{TopK}_l(token)}
P_{l,e}
\right)
$$

其中：

$$
|\operatorname{TopK}_l(token)|=6
$$

#### 3.1 它不表示什么

“激活 8B 参数”不表示：

- 产生了 8B 个 activation tensor 元素；
- 需要 8B 字节显存；
- 模型只保存 8B 权重；
- 一个 token 要读取全部 552B 权重；
- 计算量严格等于 8B FLOPs；
- Engram 的 196B 表全部参与了这个 token 的计算。

尤其要区分下面几个概念：

| 概念 | 单位 | 含义 |
|---|---|---|
| 总参数量 | 参数个数 | 模型拥有多少容量 |
| 激活参数量 | 参数个数/token | 一个 token 的执行路径实际使用多少权重 |
| activation memory | Byte | 前向过程产生的中间张量占多少内存 |
| FLOPs | 浮点运算次数 | 实际完成多少乘加、Attention 等计算 |

Activated parameters 是估算稀疏模型单 token 计算成本的一个方便指标，但不等于严格 FLOPs。Attention 的 FLOPs 还会随序列长度、稀疏 Top-K、KV 维度和精度变化。

### 4. 为什么 Prefill 是 8B，而 Decode 是 16B

这不是因为 Prefill 的 MoE 从 6 个专家减少到了 3 个。Prefill 和目标模型 Decode 中，正常 MoE 都是每层 384 选 6。

真正原因是 CED 改变了 Prompt token 要完整经过的层数。

#### 4.1 Prefill

对于长度为 $N$ 的长 Prompt：

```text
绝大多数 Prompt token
        │
        ▼
完整执行前 20 层 Causal Encoder
        │
        ├─ 产生 encoder hidden states
        └─ 投影生成 decoder 所需 global KV

不再让所有 Prompt token 完整执行后 20 层 Decoder
只有末尾 SWA window 的少量 token 做 bounded replay
```

论文给出的理想复杂度是：

$$
C_{\text{CED,prefill}}
=
N\frac{L}{2}
+
n_{\text{win}}\frac{L}{2}
$$

其中：

- $L=40$；
- $n_{\text{win}}=128$；
- 当 $N\gg128$ 时，第二项很小。

因此大部分 Prompt token 只完整经过约 20 层：

$$
P_{\text{active,prefill}}
\approx8B/token
$$

这里的 8B 是一个总体架构统计值，不应理解为每个 Prompt 位置都严格、毫无例外地使用完全相同的 8,000,000,000 个参数。Prompt 尾部参与 SWA replay 的 token 会有额外计算。

#### 4.2 Decode

生成一个新 token 时，该 token 需要通过完整生成路径：

```text
新 token
   │
   ▼
前 20 层 Causal Encoder
   │
   ▼
后 20 层 Decoder
   │
   ▼
LM Head → 下一个 token
```

因此 Decode 阶段大致执行 40 层：

$$
P_{\text{active,decode}}
\approx16B/token
$$

于是得到近似的 2:1：

$$
\frac{16B}{8B}
\approx
\frac{40\text{层}}{20\text{层}}
=2
$$

#### 4.3 用一个具体例子理解

假设 Prompt 长度为 100,000，SWA window 为 128：

| 部分 | token 数 | 大致执行深度 | 激活参数口径 |
|---|---:|---:|---:|
| 大部分 Prompt token | 100,000 | 前 20 层 | 约 8B/token |
| Decoder SWA replay | 最后 128 个 | 补充后半层局部计算 | 额外开销 |
| 每个新生成 token | 1 | 完整 40 层 | 约 16B/token |

如果仅使用 activated-parameter-token 做很粗略的估算：

$$
100000\times8B
$$

是 Prompt 主体的参数访问规模，Decode 每生成一个 token 约为：

$$
1\times16B
$$

但这种乘法不是 FLOPs，不能用于精确推导吞吐，因为它没有体现 Attention 序列长度、KV Cache、量化精度和通信开销。

### 5. 8B/16B 是否包含 196B Engram

从论文的表述和 Table 1 的统计结构看，最合理的理解是：

> 8B/16B 主要是 552B backbone 的 activated-parameter 口径；196B Engram 作为条件记忆容量单独报告，没有把整个 196B 加入每 token 激活量。

原因有三点：

1. 论文把 `# Backbone Params = 552B` 与 `# Activated Params = 8B/16B` 放在一起比较，而 Engram 在架构章节单独报告。
2. Engram 每 token 只查极少数行，不能把 196B 全算作激活参数。
3. Engram 的优势正是把 memorization 和 computation 解耦：增加巨大记忆容量，而单 token 查找成本近似保持常数。

因此不要把它解释为：

$$
8B+196B
$$

或：

$$
16B+196B
$$

如果把实际取出的 Engram 表项也记作“本 token 访问的参数”，两个模块大约只增加 12,288 个表元素，外加两个模块的投影和门控参数。相对于 8B/16B，这个查表元素数量可以忽略；而投影层是否纳入 activated-parameter 统计，取决于论文的具体计数规则。

正确理解是：

```text
模型容量：
    552B 主干参数
  + 196B Engram 条件记忆参数
  ≈ 748B 两个主要参数桶

单 token 主干执行：
  Prefill ≈ 8B
  Decode  ≈ 16B

Engram：
  每个文本 token 只从 196B 表中稀疏读取少量行
```

“约 748B”只是将论文的两个主要参数桶相加，不一定等价于整个推理 checkpoint 的全部存储项；单独训练的 DSpark、量化 scale、视觉模块等还可能采用独立的统计口径。

此外，DSpark 是另行训练的 speculative decoding 模块；论文中的 16B 更适合看作目标 backbone 的单 token 口径，而不是“开启 DSpark 后整个推测生成系统的一切额外计算”。

### 6. vLLM 与 vLLM-Ascend 代码对应关系

#### 6.1 上游 vLLM CUDA 路径

上游实现会：

1. 构造完整 40 层：
   `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:730-745`。
2. 根据最后一个 KV source layer 确定 replay 分界点。
3. 在 Prefill 时让后面的层只处理每个请求的末尾窗口：
   `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:757-784`。
4. 先执行主体层，再进入 `decoder_replay_layers`：
   `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:955-970`。

这与论文“长 Prompt 主体只经过约半个网络、末尾窗口做 replay”的思路一致。不过具体实现根据 `kv_source_layer_ids` 划分，不能机械地把实际 CUDA kernel 次数等同于理想公式中的严格 20+20。

本次核验的上游 vLLM 提交为：

```text
5f30fc7031cae49bf51073fc953d419b08f8887c
```

#### 6.2 当前本地 vLLM-Ascend 路径

Ascend 已实现：

- 384 routed experts 中 Top-6 路由；
- shared expert；
- 两层 Engram 大表；
- Engram 哈希、查表和门控融合。

例如：

- Engram 只挂到配置指定的层：
  `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:830-835`；
- 查表结果按层准备：
  `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:1143-1155`；
- 在相应层进入 Engram 门控融合：
  `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:1263-1275`。

但从当前本地提交 `4ebb3090fc3c13c26557dcaae353882e70246649` 的静态 Python 路径看：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:1254-1275`

仍然是：

```python
for layer in self.layers:
    ...
    hidden_states, pre_mix = layer(...)
```

即显式遍历全部层，目前没有看到上游 CUDA 那种 `decoder_replay_start + DecoderReplayLayers` 的层级裁剪结构。

所以应区分：

| 结论 | 状态 |
|---|---|
| 论文架构口径：Prefill 约 8B、Decode 约 16B | 论文明确给出 |
| 上游 CUDA 已有 decoder bounded replay 路径 | 源码可确认 |
| 当前 Ascend Python 路径已实现 Engram 和 Top-6 MoE | 源码可确认 |
| 当前 Ascend 实际 Prefill 已达到 8B/token | 仅凭当前静态源码不能确认；从显式遍历全部层看，甚至有理由怀疑尚未完整落地 CED 计算裁剪 |
| 实机到底执行多少层和多少 token | 需要 profiler 的逐层 token shape/kernel 记录验证 |

### 问题 3 最终总结

这句话可以翻译成更直白的表达：

> DeepSeek-V4.1-Flash 的主干网络拥有约 5520 亿参数，其中绝大部分是分布在 40 层中的 MoE 专家；模型另外拥有约 1960 亿参数的 Engram 条件记忆表。虽然总容量很大，但一个 token 在每层只选择 384 个 routed experts 中的 6 个，并且 Engram 只查少量表项，所以不会使用全部参数。长 Prompt 的大多数 token 在 CED Prefill 中只完整经过前 20 层，对应约 80 亿激活参数；生成阶段的新 token 需要经过完整 40 层，对应约 160 亿激活参数。

一句话概括：

$$
\boxed{
\text{大总容量}
\ne
\text{大单-token计算量}
}
$$

DeepSeek-V4.1-Flash 同时利用了两种稀疏性：

$$
\boxed{
\text{MoE：计算稀疏}
\quad+\quad
\text{Engram：记忆访问稀疏}
}
$$

再通过 CED 让 Prefill 具有额外的“层深度稀疏”，于是形成了 `552B backbone + 196B memory`，但 `8B/16B active` 的组合。

---

## 问题 4：CED 如何从 Encoder 输出构造 Decoder Global KV

### 问题 4 原文

> “CED constructs the decoder’s global key-value (KV) cache from encoder outputs”应该怎么解释？为什么保存层滑窗注意力就可以跳过大部分提示词的 Decoder 阶段呢？

### 问题 4 核心结论

论文中这句话的准确含义是：

> Decoder 的长程全局 KV，不需要先让所有 Prompt token 依次经过 Decoder 的每一层再生成；它可以直接由第 20 层 Encoder 的输出表示投影得到。

因此，长度为 $N$ 的 Prompt 不再需要完整执行后 20 层 Decoder。真正阻止 Prefill 在 Encoder 结束处立即停止的，只剩下 Decoder 每层自己的 Sliding-Window Attention（SWA）状态。

但需要纠正一个容易产生的误解：

> 不是“保存滑窗注意力就可以跳过 Decoder”，而是“全局 KV 已由 Encoder 输出生成，所以 Decoder 只剩局部 SWA KV 需要恢复”；又因为 SWA 只关心末尾窗口，所以只对最后 $n_{\text{win}}$ 个 token 做 Decoder replay。

而且论文明确承认：只 replay 最后一个窗口是一种近似，不是和完整 Decoder Prefill 数学等价的精确结果。

论文依据主要位于本地论文的第 8、9、19、20 页：

`D:\Code\vllm_notes\DeepSeekV41-Flash\DeepSeek-V4.1-Flash Pushing the Limits of KV Cache Compression.pdf`

### 1. 普通 Decoder-only 为什么不能跳过 Prompt 的 Decoder 层

先看普通 40 层 Decoder-only 模型。

设 Prompt 有 $N$ 个 token，第 $l$ 层的隐藏状态为：

$$
H_l\in\mathbb{R}^{N\times d}
$$

普通模型中，第 $l$ 层自己的 KV 通常由该层输入或隐藏状态生成：

$$
K_l=H_lW_l^K,\qquad V_l=H_lW_l^V
$$

关键依赖是：

$$
H_l=f_l(H_{l-1})
$$

因此要得到第 35 层的 Prompt KV，必须先得到第 35 层对应的 Prompt hidden states；而要得到 $H_{35}$，就必须依次运行：

$$
H_0\to H_1\to H_2\to\cdots\to H_{35}
$$

所以普通模型的依赖关系是：

```text
Prompt Embedding
      │
      ▼
Layer 0 → 产生 K0/V0
      │
      ▼
Layer 1 → 产生 K1/V1
      │
     ...
      │
      ▼
Layer 39 → 产生 K39/V39
```

不能直接说：

> “我不计算 Layer 20～39 的 Prompt hidden states，但仍然希望得到它们的 KV。”

因为普通模型的第 $l$ 层 KV 本来就定义为 $H_l$ 的投影。

### 2. CED 改变了什么依赖关系

CED 最关键的改动不是简单地“共享 KV Cache”，而是改变了 Decoder 全局 KV 的来源。

论文把 40 层分成：

```text
Layer 0～19：Causal Encoder
Layer 20～39：Decoder
```

设 Encoder 最后一层输出为：

$$
H_{\text{enc}}=H_{L/2}=H_{20}
$$

对于后半部分 Decoder 层 $l>L/2$，普通方式原本是：

$$
K_l^{\text{global}},V_l^{\text{global}}
=
\operatorname{Proj}_l(H_l)
$$

而 CED 改成：

$$
C_l=H_{L/2}W_l^{KV}
$$

$$
Z_l=H_{L/2}W_l^Z
$$

其中：

- $H_{L/2}$ 是 Encoder 最后一层输出；
- $C_l$ 是 Decoder 使用的全局压缩 KV；
- $Z_l$ 是对应的压缩权重；
- $W_l^{KV}$、$W_l^Z$ 是投影参数。

所以依赖关系从：

```text
Prompt → Decoder Layer l 的 hidden state Hl
                     │
                     └→ Decoder Layer l 的 global KV
```

变成：

```text
Prompt → Encoder 最终输出 H20
                     │
                     ├→ Decoder global KV
                     ├→ Decoder global KV
                     ├→ Decoder global KV
                     └→ Decoder global KV
```

这就是：

> “CED constructs the decoder’s global KV cache from encoder outputs”

的具体含义。

#### 2.1 两个重要限定

第一，Encoder 并不是直接把自己的同一份 K/V 原封不动交给 Decoder，而是：

> Decoder 使用相应的投影参数，把 Encoder 最终表示转换成 Decoder 所需的全局 KV 表示。

第二，Decoder 仍然有自己的 Q：

$$
Q_l(t)=H_l(t)W_l^Q
$$

Decode 新 token 经过 Decoder 第 $l$ 层时，会用该层自己的 Query 去检索由 Encoder 表示构造出来的全局 KV。

### 3. 为什么 Encoder 输出可以让大部分 Prompt 跳过 Decoder

假设 Prompt 长度是 100,000。

普通模型需要做：

```text
100,000 个 Prompt token
  × 40 层
```

因为每一层都要为这 100,000 个 token 生成本层 KV。

CED 则把状态拆成两部分：

| 状态 | 覆盖范围 | 来源 | 是否需要所有 Prompt token 经过 Decoder |
|---|---|---|---|
| Global KV | 整个长上下文 | Encoder 最终输出 $H_{20}$ | 不需要 |
| Decoder SWA KV | 最近一个局部窗口 | Decoder 每层自己的 hidden state | 需要，但只需末尾窗口 |

长上下文信息已经包含在：

$$
H_{20}(0),H_{20}(1),\ldots,H_{20}(N-1)
$$

中。

然后由这些 Encoder 输出直接构造 Decoder global KV：

$$
K_{\text{global}},V_{\text{global}}
=
\operatorname{Proj}(H_{20})
$$

因此，为了让未来 Decode token 可以访问 Prompt 的全局信息，不再需要计算：

$$
H_{21}(0:N),H_{22}(0:N),\ldots,H_{39}(0:N)
$$

只要得到全局 KV，未来新 token 在 Decoder 中就可以这样工作：

```text
新生成 token 在 Decoder 第 l 层的 Query
              │
              ├── Global Attention
              │      └─ 检索完整 Prompt 的 Encoder-derived global KV
              │
              └── Sliding-Window Attention
                     └─ 检索最近 W 个 token 的 layer-local SWA KV
```

因此，远处上下文由 global branch 提供，最近邻的细粒度局部信息由 SWA branch 提供。

#### 3.1 实际不是普通 Dense Global Attention

DeepSeek-V4.1-Flash 的 global branch 是 CSA2 稀疏全局注意力，不是每个 Query 与全部 100K token 做完整 Dense Attention。

大体上是：

1. Indexer 从全局 KV 中选择相关位置；
2. 每个 Query 选择 Top-K 全局条目；
3. Sparse Attention 只读取被选中的全局 KV。

论文配置中 `index_topk=512`，官方配置还给出：

```json
"window_size": 128,
"kv_source_layers": [2, 8, 14, 20],
"index_source_layers": [2, 8, 14, 20, 24, 28, 32, 36]
```

其中 Decoder 部分只有 layer 20 是 global KV source，后续 Decoder 层都可以消费该 source 的全局 KV；layer 24、28、32、36 主要重新执行 index selection，而不是重新生成一份完整 global KV。

### 4. 为什么 Decoder 仍需要每层自己的 SWA KV

因为论文没有把 Decoder 的所有 KV 都改成来自 Encoder 输出。

对于 Global Attention：

$$
K_l^{\text{global}},V_l^{\text{global}}
\leftarrow H_{20}
$$

但对于 Sliding-Window Attention：

$$
K_l^{\text{SWA}},V_l^{\text{SWA}}
=
\operatorname{Proj}_l(H_l)
$$

也就是说：

> Decoder 第 30 层的 SWA KV 必须来自第 30 层的隐藏状态，不能简单使用 Encoder 的 $H_{20}$ 代替。

这种设计保留了 Decoder 的计算深度。

如果后 20 层所有 KV 都直接使用完全相同的 Encoder 表示，那么后半部分网络的上下文建模能力可能会过度受限。保留 layer-local SWA，相当于让每个 Decoder 层仍然拥有：

- 本层自己的局部 K/V；
- 本层自己的 Q；
- 本层的 MoE 和非线性变换；
- 对最近 token 的逐层深度建模能力。

因此 CED 是一种混合结构：

```text
长程信息：
Encoder H20
   └─ 生成 Decoder global KV
      └─ 不需要完整 Decoder Prompt forward

局部信息：
Decoder Hl
   └─ 生成每层自己的 SWA KV
      └─ 仍然需要少量 Decoder replay
```

### 5. 为什么只需 Replay 最后一个滑窗

设 SWA window：

$$
W=n_{\text{win}}=128
$$

对于下一次 Decode 的新 token $x_N$，Decoder 的 SWA 只会关注最近约 128 个位置：

$$
x_{N-128},\ldots,x_{N-1}
$$

更早的 token：

$$
x_0,\ldots,x_{N-129}
$$

不会被下一 token 的 SWA branch 直接访问。

因此，第一步 Decode 真正需要的局部状态是：

```text
每个 Decoder 层：
  最近 W 个 Prompt token 的 layer-local SWA K/V
```

不需要为 100K Prompt 中所有位置保存 Decoder SWA KV。

完整的数据流可以表示为：

```text
100K Prompt
    │
    ▼
20 层 Encoder，对全部 100K token 计算
    │
    ├─────────────── 全部 H20 ───────────────┐
    │                                        │
    │                                        ▼
    │                             构造完整 global KV
    │                             覆盖整个 100K Prompt
    │
    └─ 截取最后 128 个 H20
                │
                ▼
        replay 20 层 Decoder
                │
                ▼
      生成每层最近 128 个 SWA KV
                │
                ▼
       得到最后 Prompt token 的最终输出
                │
                ▼
            生成第一个 token
```

所以真正的逻辑是：

$$
\boxed{
\text{完整 Prompt 的全局信息}
+
\text{最近窗口的逐层局部信息}
}
$$

而不是：

$$
\text{仅靠最近 128 个 token 表示整个 Prompt}
$$

旧 Prompt token 并没有消失；它们仍然存在于 global KV 中。只是它们不再需要存在于 Decoder 每层的 local SWA KV 中。

### 6. 为什么最后 128 个 Token Replay 只是近似

这是这套方法最容易被忽略的地方。

#### 6.1 单层 SWA 的依赖范围

假设在某一层，token $i$ 可以注意最近 $W$ 个 token：

$$
H_l(i)
\leftarrow
H_{l-1}(i-W+1:i)
$$

#### 6.2 多层以后依赖会扩散

虽然每层只看 128 个 token，但第二层所看到的每一个输入已经包含第一层的局部上下文。

例如令 $W=4$：

```text
Layer 1:
位置 12 依赖 9～12

Layer 2:
位置 12 依赖 Layer 1 的 9～12
而 Layer 1 的位置 9 又依赖 6～9

所以 Layer 2 的位置 12
可能间接依赖位置 6～12
```

随着层数增加，理论感受野不断向前扩张。

对于 $L_{\text{dec}}=20$、$W=128$，精确恢复顶层局部状态，理论上可能需要回放大约：

$$
L_{\text{dec}}\times W
=
20\times128
=
2560
$$

个 Prompt token。

#### 6.3 Bounded Replay 做了什么截断

论文没有 replay 2560 个 token，而是只 replay 最后 128 个：

$$
s=N-W
$$

对于 replay 段中的位置 $i$，SWA 的实际可见范围被截断成：

$$
\left[
\max(s,i-W+1),i
\right]
$$

这意味着 replay 段开头的 token 无法看到位于 $s$ 之前、本来应该在其 SWA window 内的 token。

例如：

```text
完整计算中：

token 99872 的 SWA
可能需要 99745～99872

但 bounded replay 从 99872 开始：

token 99872 只能看到 replay 段中的自己
无法看到 99745～99871
```

随着 replay 在 Decoder 中向后传播，后面位置逐渐拥有更多 replay 段内部上下文，但仍不等价于完整 100K Prompt 经过 Decoder 的结果。

所以论文明确指出：

> reconstructed decoder SWA KV is not mathematically equivalent to that from a full decoder forward pass

之所以可用，是因为：

1. Global KV 仍覆盖完整 Prompt；
2. 缺失的只是 replay 边界之前的一部分局部 SWA 传播；
3. SWA 的实际有效感受野通常小于理论最大感受野；
4. 论文在 post-training 中模拟了同样的 replay，让模型适应这种截断；
5. 论文实验发现响应质量下降可以忽略。

因此这不是严格等价变换，而是经过训练适配和实验验证的部署近似。

### 7. Retaining Layer-local SWA 不等于持久化保存

论文第 8 页的：

> retaining layer-local sliding-window attention

表示：

> 模型架构中仍保留每一层自己的 SWA 分支。

它不等于：

> 将所有层的 SWA KV 长期保存在 SSD persistent cache 中。

实际上论文第 19～20 页明确说明：

- Global KV 长期放在 persistent KV cache；
- SWA KV 不再长期持久化；
- 活跃会话可以在短 TTL 的 host memory pool 中暂存 SWA KV；
- SWA KV miss 时，用 bounded replay 重建；
- Decoder replay 得到的 SWA KV 只用于接下来的 Decode，不用于 prefix caching。

可以整理为：

| 状态 | 长期持久化 | Cache miss 后如何处理 |
|---|---|---|
| Global KV | 是 | 命中后直接复用完整前缀 |
| Encoder SWA KV | 否或短期保存 | replay 最近窗口，重建 Encoder SWA |
| Decoder SWA KV | 不用于 persistent prefix caching | 每次 Prefill 对最后窗口做 Decoder replay |

因此更准确的表述不是“保存层滑窗注意力”，而是：

> 保留 layer-local SWA 的计算机制，并在需要 Decode 时只重建末尾窗口的 SWA KV。

### 8. 100K Prompt 计算示例

设：

$$
N=100000,\quad L=40,\quad L_{\text{enc}}=20,\quad
L_{\text{dec}}=20,\quad W=128
$$

#### 8.1 普通模型

所有 Prompt token 经过全部 40 层：

$$
C_{\text{normal}}
=
100000\times40
=
4,000,000
$$

单位是简化的 layer-token。

#### 8.2 论文理想 CED

全 Prompt 经过 Encoder：

$$
100000\times20
=
2,000,000
$$

最后 128 个 token 经过 Decoder replay：

$$
128\times20
=
2,560
$$

总计：

$$
C_{\text{CED}}
=
2,000,000+2,560
=
2,002,560
$$

相对普通模型：

$$
\frac{2,002,560}{4,000,000}
=
50.064\%
$$

即 Prefill 计算量接近减半。

#### 8.3 第一 Decode 步能访问什么

第一个生成 token 进入 Decoder 时，每层同时使用：

```text
Global branch:
    完整 100K Prompt 的压缩全局 KV
    └─ 通过 sparse indexer 选择相关条目

Local SWA branch:
    最近 128 个 Prompt token 的本层 SWA KV
```

因此它同时拥有：

- 全局长程记忆；
- 最近上下文的高分辨率局部状态。

这正是能够跳过大多数 Prompt Decoder 计算而不完全丢失长上下文的原因。

### 9. 上游 vLLM 代码对应关系

以下位置基于本地上游提交：

```text
5f30fc7031cae49bf51073fc953d419b08f8887c
```

#### 9.1 KV Source 与 Consumer

上游代码把各层分成 KV source 和 consumer：

- `compress_ratios=0` 表示纯 SWA；
- `1/2` 表示具有压缩 global KV；
- Compressor 只存在于 `kv_source_layer_ids`；
- consumer 读取最近的 source 所发布的 KV。

代码位置：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:303-349`

只有 source layer 创建 compressor；consumer 记录 source cache 的路径：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:573-603`

获取 global KV 时：

```python
if self.is_kv_source:
    return self.kv_cache

source = static_forward_context[self.compressed_cache_prefix]
return source.kv_cache
```

对应：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:1137-1144`

这说明后续 Decoder layer 不需要各自拥有完整 global KV cache，而可以指向 source layer 的 cache。

#### 9.2 Global KV 与 SWA KV 是独立缓存

SWA cache 单独创建：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:529-568`

Global compressed cache 则只由 KV source layer 的 compressor 创建：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:573-585`

因此代码结构本身就是：

```text
每层：
  自己的 SWA cache

若是 KV source：
  再拥有 compressed global KV cache

若是 consumer：
  读取 source 的 compressed global KV cache
```

#### 9.3 Decoder Replay 只保留末尾窗口行

`decoder_replay_layers.py` 对机制的说明非常直接：

> last KV-source 之后的层只拥有 sliding-window KV，所以 eager Prefill 中只在每个请求最后一个 window 的行上运行。

代码位置：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\decoder_replay_layers.py:3-11`

它通过 `index_select` 取出 replay rows，在较小的 sub-batch 上执行后半层，再把输出 scatter 回原 batch：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\decoder_replay_layers.py:45-67`

#### 9.4 Replay 分界点

模型取最后一个 KV source：

```python
cut = max(config.kv_source_layer_ids)
```

然后将：

```python
decoder_replay_start = cut + 1
```

后面的层放入 replay 路径：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:757-784`

主 forward 先执行到 replay 边界，再让 `DecoderReplayLayers` 处理剩余层：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:955-978`

#### 9.5 Replay Token 不覆盖已经缓存的 Global KV

上游 model state 会把 replay token 在 prefix-cacheable cache group 中的 slot mapping 改成 padding slot，注释明确说明：

> replayed tokens rebuild window KV only; cached KV stays as is.

代码位置：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model_state.py:303-333`

它还通过：

```python
np.minimum(query_lens, layers.window)
```

把每个 Prefill request 的 replay batch 截断到最多一个 window：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model_state.py:403-456`

#### 9.6 为什么上游代码不是严格的 $20N+20W$

官方配置中的最后一个 KV source 是 layer 20：

```json
"kv_source_layers": [2, 8, 14, 20]
```

上游代码使用：

```python
cut = max(kv_source_layer_ids)  # 20
decoder_replay_start = cut + 1  # 21
```

因此，当前 CUDA 实现的简化层执行范围是：

```text
Layer 0～20：处理完整 Prompt
Layer 21～39：只处理末尾 replay window
```

对应的简化计算量更接近：

$$
21N+19W
$$

而不是论文理想公式中的：

$$
20N+20W
$$

对于 $N=100000,W=128$：

$$
21\times100000+19\times128
=
2,102,432
$$

这仍然比普通的 4,000,000 小很多，但它说明：

> 论文公式表达架构级复杂度；具体实现还需要考虑 source layer 的执行边界、索引、图模式和 kernel 限制。

### 10. vLLM-Ascend 代码对应关系

以下位置基于本地 Ascend 提交：

```text
4ebb3090fc3c13c26557dcaae353882e70246649
```

#### 10.1 已实现 Global KV Source/Consumer 拓扑

Ascend 会根据：

- `kv_source_layer_ids`；
- `index_source_layer_ids`；
- `compress_ratios`；

为每一层计算 `kv_source_layer`、`is_kv_source` 等角色：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:440-548`

KV source layer 创建：

- `long_kv_cache`；
- `DeepseekV41Compressor`。

consumer 则记录 `long_kv_source_prefix`，指向 source layer 的全局 KV：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:665-748`

#### 10.2 每层仍然有自己的 SWA Cache

Ascend 为每层单独创建 `swa_cache_layer`：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:577-660`

在执行时，每层当前 hidden state 被投影为 SWA KV，并写入本层 SWA cache：

```python
kv = self._project_kv(attn, hidden_states, cos, sin)
scatter_cache_sk(
    attn.dsa_attn.swa_cache_layer.kv_cache[0],
    metadata.swa.slot_mapping,
    kv,
)
```

对应：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:271-291`

只有 KV source layer 会额外生成 compressed global KV：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:380-438`

#### 10.3 Attention 算子同时读取 Local SWA 与 Global KV

Ascend 的 Sparse Flash MLA 调用中同时传入：

```python
ori_kv = 本层 SWA KV
cmp_kv = source layer 的 compressed global KV
```

对应：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:474-520`

这正好对应：

```text
ori_kv → 最近窗口的高分辨率局部信息
cmp_kv → 完整 Prompt 的压缩全局信息
```

#### 10.4 当前 Ascend 代码的限制

当前本地 Ascend 的高层 forward 仍然是：

```python
for layer in self.layers:
    hidden_states, pre_mix = layer(...)
```

代码位置：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:1245-1275`

也就是说，静态源码已经具备：

- Global KV source/consumer；
- 独立的 SWA cache；
- 两类 KV 同时进入 Attention。

但是没有看到上游 CUDA 中完整的：

```text
decoder_replay_start
+ replay rows sub-batch
+ DecoderReplayLayers
```

因此应谨慎区分：

| 能力 | 上游 CUDA | 当前本地 Ascend |
|---|---|---|
| Global KV source/consumer | 已实现 | 已实现 |
| 每层独立 SWA KV | 已实现 | 已实现 |
| Replay token 不覆盖 cached global KV | 已实现 | 未看到对应完整高层路径 |
| 后半 Decoder 仅执行末尾窗口 | 已实现但有条件 | 当前静态 Python forward 未体现 |
| 论文理想 Prefill 减半 | 接近实现 | 不能仅凭当前源码认定已经达到 |

### 问题 4 最终总结

“CED constructs the decoder’s global KV cache from encoder outputs”可以解释为：

> 对于完整 Prompt，CED 先用前 20 层 Encoder 得到每个位置的最终 Encoder 表示，再用这些表示生成后 20 层所需的长程全局 KV；因此，不必为了构造全局 KV 而让所有 Prompt token 通过完整 Decoder。Decoder 只需要对 Prompt 尾部一个 SWA window 做近似 replay，以生成每层自己的局部 SWA KV，并产生最后一个 Prompt token 的最终表示。Decode 时，新 token 同时查询完整 Prompt 的 global KV 和最近窗口的 layer-local SWA KV。

其核心不是“只保留窗口就足够理解整个 Prompt”，而是：

$$
\boxed{
\underbrace{\text{完整 Prompt 的 Encoder-derived Global KV}}_{\text{负责长程信息}}
+
\underbrace{\text{最后窗口的 Decoder layer-local SWA KV}}_{\text{负责局部信息}}
}
$$

最后再强调一次：

$$
\boxed{
\text{能跳过完整 Decoder Prompt forward 的根因是 Global KV 来源被解耦}
}
$$

而：

$$
\boxed{
\text{SWA Bounded Replay 是为补齐剩余局部状态所做的近似恢复}
}
$$

---

## 问题 5：Causal Encoder、CED、Global/Local KV 与 SWA Replay

### 问题 5 原文

> Layer 0～19：Causal Encoder 的计算过程，是不是和普通的 KV 是一样的？有什么特殊之处吗？CED 是什么？$W_{KV}^{(l)}$、$W_Z^{(l)}$ 是什么？Global K/V：主要来自 $H^{(20)}$，Local K/V：来自各层自己的 SWA，这两句话应该怎样解释？生成/压缩全局 KV 的 layer-specific projection 是什么过程？最近 $n_{win}$ 个 token 做 SWA Replay 是什么意思，这个过程的原理是什么？

### 问题 5 核心结论

Layer 0～19 虽然叫 **Causal Encoder**，但它们本质上仍然是带因果掩码的 Transformer 自回归计算块，并不是 BERT 式双向 Encoder。

它们的局部 SWA KV 与普通自回归模型的逐层 KV 很接近；真正特殊的地方在于：模型另外增加了一条 CSA2 全局 KV 分支，而且这条全局 KV 会在 token 维度压缩、在 layer 维度复用。

必须先区分两个概念：

- **Decoder Global KV 来自 $H^{(20)}$**：Decoder 的长程 KV 不必再从每个 Decoder 层自己的隐藏状态产生，而是可以从最后一层 Causal Encoder 的输出投影得到。
- **Local KV 来自各层自己的 SWA**：每个 Decoder 层最近窗口的局部 KV 仍必须从该层自己的隐藏状态产生，不能用 $H^{(20)}$ 统一替代。因此需要 SWA Replay。

本节核对的本地代码版本为：

- `D:\Code\vllm-github`：`5f30fc7031cae49bf51073fc953d419b08f8887c`；
- `D:\Code\vllm-ascend\vllm-ascend`：`4ebb3090fc3c13c26557dcaae353882e70246649`；
- 论文重点参考 Section 2.2、2.3、3.2.2 和 4.2.1。

### 5.1 40 层整体结构

代码使用从 0 开始的层编号：

```text
输入 Token
   │
   ▼
┌────────────────── Causal Encoder：Layer 0～19 ──────────────────┐
│                                                                │
│ Layer 0～1   ：纯 SWA，只保留每层自己的局部 KV                 │
│                                                                │
│ Layer 2      ：Full，生成 ratio=2 的全局 KV + Indexer K/Top-K  │
│ Layer 3～7   ：Reuse，复用 Layer 2 的全局 KV 和 Top-K           │
│                                                                │
│ Layer 8      ：Full，生成新的 ratio=2 全局 KV                   │
│ Layer 9～13  ：Reuse Layer 8                                    │
│                                                                │
│ Layer 14     ：Full，生成新的 ratio=2 全局 KV                   │
│ Layer 15～19 ：Reuse Layer 14                                   │
└────────────────────────────────────────────────────────────────┘
   │
   │ 输出：H^(20)，即经过前 20 层后的所有 Token 表示
   ▼
┌──────────────── Decoder：Layer 20～39 ─────────────────────────┐
│                                                                │
│ Layer 20     ：Full，由 H^(20) 投影出 Decoder 全局 KV           │
│ Layer 21～23 ：Reuse Layer 20 的全局 KV 和 Top-K                │
│                                                                │
│ Layer 24     ：Reindex，不生成新全局 KV，只重新选择 Top-K       │
│ Layer 25～27 ：Reuse                                            │
│ Layer 28/32/36：Reindex                                        │
│ 其余层       ：Reuse                                            │
│                                                                │
│ 但是：所有 Decoder 层仍然需要各自独立的局部 SWA KV             │
└────────────────────────────────────────────────────────────────┘
```

官方模型配置给出：

```json
"window_size": 128,
"kv_source_layers": [2, 8, 14, 20],
"index_source_layers": [2, 8, 14, 20, 24, 28, 32, 36],
"compress_ratios": [0, 0, 2, ..., 2, 1, ..., 1, 0, 0, 0]
```

含义如下：

- Layer 0～1：`compress_ratio=0`，纯 SWA；
- Layer 2～19：`compress_ratio=2`，每两个 token 压缩成一个全局 KV entry；
- Layer 20～39：`compress_ratio=1`，一个 token 对应一个全局 KV entry；
- 最后三个 0 对应 MTP 层，不属于 40 层 backbone。

vLLM 对这三种压缩比的解释位于：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:303-349`。

### 5.2 Layer 0～19 与普通 KV 的异同

#### 普通 Decoder-only 模型

普通第 $l$ 层一般计算：

$$
Q^{(l)}=H^{(l)}W_Q^{(l)}
$$

$$
K^{(l)}=H^{(l)}W_K^{(l)},\qquad
V^{(l)}=H^{(l)}W_V^{(l)}
$$

然后执行：

$$
A^{(l)}=operatorname{Softmax}
\left(
\frac{Q^{(l)}K^{(l)T}}{\sqrt d}+M_{\text{causal}}
\right)V^{(l)}
$$

每层都保存自己完整的逻辑 KV cache：

$$
\operatorname{KVCache}^{(l)}=
\left(K^{(l)}_{0:N-1},V^{(l)}_{0:N-1}\right)
$$

#### DeepSeek-V4.1-Flash 的双记忆分支

CSA2 层的注意力可以抽象成：

$$
\operatorname{Attention}^{(l)}=
\operatorname{Attention}
\left(
Q^{(l)},
\underbrace{\operatorname{SWA\text{-}KV}^{(l)}}_{\text{局部、逐层独立}},
\underbrace{\operatorname{Global\text{-}KV}^{(s(l))}}_{\text{长程、跨层复用}}
\right)
$$

其中 $s(l)$ 是不大于当前层的最近 KV source layer。

| 当前层 | 使用的 Global KV | Local SWA KV |
|---:|---:|---:|
| 2 | Layer 2 生成 | Layer 2 自己生成 |
| 3 | 复用 Layer 2 | Layer 3 自己生成 |
| 4 | 复用 Layer 2 | Layer 4 自己生成 |
| 8 | Layer 8 重新生成 | Layer 8 自己生成 |
| 9 | 复用 Layer 8 | Layer 9 自己生成 |
| 15 | 复用 Layer 14 | Layer 15 自己生成 |

所以答案不是简单的“相同”或“不同”，而是：

> 局部 SWA KV 仍然是普通的逐层生成思路；全局 KV 则采用 token 压缩、稀疏 Top-K 和跨层复用，不再是一层一份完整长程 KV cache。

代码对应关系：

- KV source、index source 和最近 source 的解析：`D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:303-349`；
- 只有 source layer 创建 Compressor：`D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:573-603`；
- consumer 读取 source cache：`D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:1110-1144`；
- Ascend 的 layer role 拓扑：`D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:503-548`。

### 5.3 Causal Encoder 的特殊之处

#### 5.3.1 它仍然是 Causal 的

对位置 $i$，第 $l$ 层隐藏状态只能依赖当前位置和之前的位置：

$$
H_i^{(l)}=f_l\left(H_0^{(l-1)},\ldots,H_i^{(l-1)}\right)
$$

不能依赖 $i+1,i+2,\ldots$，因此它不是传统 BERT 式双向 Encoder。它只是承担了“生成可供上半网络共享的中间表示”的 Encoder 职责。

#### 5.3.2 Layer 0～1 是纯 SWA

窗口大小为 128：

$$
\mathcal W(i)=[\max(0,i-127),i]
$$

例如位置 1000 的 token 只通过 SWA 直接访问位置 873～1000。Layer 0～1 没有全局 compressed KV，所以 `compress_ratio=0`。

#### 5.3.3 Layer 2～19 使用 ratio=2 压缩

```text
Token 0、1   ──压缩──> Global KV entry 0
Token 2、3   ──压缩──> Global KV entry 1
Token 4、5   ──压缩──> Global KV entry 2
...
```

100K token 大约形成 50K 个全局 KV entry。最近窗口仍通过未进行 token 压缩的 layer-local SWA KV 保存，因此总体策略是：

```text
远处历史：压缩、稀疏选择、跨层复用
附近历史：高分辨率、逐层独立的 SWA KV
```

#### 5.3.4 Global KV 不按每层各存一份

Encoder 的 KV source 是 Layer 2、8、14：

- Layer 2～7 共用 Layer 2 的 Global KV；
- Layer 8～13 共用 Layer 8；
- Layer 14～19 共用 Layer 14。

这同时在三个方向降低成本：

- entry size：MLA 风格的 512 维 latent；
- sequence dimension：ratio=2；
- layer dimension：六层共享一份 Global KV。

### 5.4 CED 是什么

CED 是：

$$
\boxed{\text{CED}=\text{Causal Encoder-Decoder}}
$$

它解决的是长 Prompt 或 prefix cache miss 时的 prefill 计算问题。

普通 40 层模型的 layer-token 主干计算量近似为：

$$
C_{\text{normal}}=N\times40
$$

CED 把模型划分为：

- 前 20 层 Causal Encoder；
- 后 20 层 Decoder。

对绝大多数 Prompt token：

1. 完整经过前 20 层，得到 $H^{(20)}$；
2. 直接从 $H^{(20)}$ 投影出 Decoder Global KV；
3. 不要求全部 Prompt token 完整执行后 20 层；
4. 仅对末尾 $n_{win}$ 个 token 做 Decoder SWA Replay。

理想化计算量为：

$$
C_{\text{CED}}=N\frac L2+n_{win}\frac L2
$$

例如 $N=100000$、$L=40$、$n_{win}=128$：

$$
C_{\text{CED}}=100000\times20+128\times20=2,002,560
$$

普通模型为：

$$
C_{\text{normal}}=100000\times40=4,000,000
$$

所以 prefill 接近减少一半。CED 主要降低 prefill；进入逐 token decode 后，新 token 仍需要正常通过完整 backbone。

### 5.5 WKV 与 WZ 分别是什么

论文 CED 公式为：

$$
C_l=H_{L/2}W_{KV}^{(l)}
$$

$$
Z_l=H_{L/2}W_Z^{(l)}
$$

其中：

- $H_{L/2}$：最后一层 Causal Encoder 的输出；
- $W_{KV}^{(l)}$：把隐藏状态投影为 KV latent 的权重；
- $W_Z^{(l)}$：生成 KV 压缩门控 logit 的权重；
- $C_l$：候选 KV latent；
- $Z_l$：压缩权重的 logit，不是 Value，也不是最终缓存。

#### 5.5.1 $W_{KV}$：内容投影

模型隐藏维度为：

$$
d_{model}=5120
$$

Global KV latent 维度为：

$$
d_{KV}=512
$$

所以可以理解为：

$$
W_{KV}\in\mathbb R^{5120\times512}
$$

对第 $i$ 个 token：

$$
c_i=h_iW_{KV},\qquad c_i\in\mathbb R^{512}
$$

这里不是传统 MHA 中分别保存大规模 $K_i$ 和 $V_i$，而是 MLA/CSA2 风格的共享 KV latent。

Ascend 中对应：

```python
self.wkv = nn.Linear(dim, self.width, bias=False)
```

代码位置：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\compressor.py:14-23`。

#### 5.5.2 $W_Z$：压缩门控投影

Ascend 代码将论文中的 $W_Z$ 命名为 `wgate`：

```python
self.wgate = nn.Linear(dim, self.width, bias=False)
```

对 token $i$：

$$
z_i=h_iW_Z
$$

当前代码中的 $z_i$ 不是一个标量，而是 512 维向量：

$$
z_i\in\mathbb R^{512}
$$

因此压缩采用逐通道 learned softmax pooling，而不是让所有 512 个通道共用一个标量权重。

### 5.6 Ratio=2 时如何压缩 Global KV

以两个 token $h_0,h_1$ 压缩成一个 entry 为例。

先做两组投影：

$$
c_0=h_0W_{KV},\qquad c_1=h_1W_{KV}
$$

$$
z_0=h_0W_Z,\qquad z_1=h_1W_Z
$$

然后沿两个 token 的维度做逐通道 softmax：

$$
p_0[d]=\frac{\exp(z_0[d])}{\exp(z_0[d])+\exp(z_1[d])}
$$

$$
p_1[d]=\frac{\exp(z_1[d])}{\exp(z_0[d])+\exp(z_1[d])}
$$

最终第 $d$ 个通道为：

$$
\widetilde c[d]=p_0[d]c_0[d]+p_1[d]c_1[d]
$$

整体写为：

$$
\widetilde c=
\sum_{r=0}^{1}
\operatorname{Softmax}_{r}(z_r)\odot c_r
$$

之后还会执行：

1. RMSNorm；
2. 部分 RoPE；
3. 写入 compressed Global KV cache；
4. 从 compressed latent 投影生成 Indexer K。

Ascend 的纯 PyTorch 参考实现直接展示了上述过程：

```python
kv = torch.nn.functional.linear(xs.float(), wkv.float())
scores = torch.nn.functional.linear(xs.float(), wgate.float())
prob = score_part.softmax(dim=0)
pooled = (kv_part * prob).sum(dim=0)
```

对应：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\ops\triton\compressor\compressor_triton.py:567-626`。

实际 NPU 路径为：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:380-438`。

其中：

- 第 406 行执行 `compressor.wkv(hidden_states_fp32)`；
- 第 407 行执行 `compressor.wgate(hidden_states_fp32)`；
- 第 408 行执行 `pool_projected(kv, score, ...)`；
- 第 420～425 行更新 Indexer K；
- 第 426～438 行执行 RoPE 并写入 long Global KV cache。

如果 chunked prefill 把一对 token 拆到两个 chunk，Compressor 会在 ring state cache 中暂存未完成分组的 $[c_i,z_i]$，等下一个 token 到来后再完成 pooling。参考实现第 589～608 行读取残余行，第 623～626 行更新状态缓存。

### 5.7 Ratio=1 时是否还需要 WZ

论文的一般 CED 公式同时写出了 $W_{KV}^{(l)}$ 和 $W_Z^{(l)}$。但 Decoder 的实际压缩比为：

$$
m=1
$$

一个 token 自己形成一个 entry，softmax 只有一个元素：

$$
\operatorname{Softmax}(z_i)=1
$$

因此 $W_Z$ 没有实际作用，代码不会为 ratio=1 创建 gate：

```python
self.has_gate = compress_ratio > 1
```

vLLM 代码注释也明确说明 ratio=1 没有 gate，见：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\compressor.py:171-228`。

Decoder boundary 的实际路径可写为：

$$
H^{(20)}
\xrightarrow{W_{KV}^{(20)}}
C^{(20)}
\xrightarrow{\operatorname{RMSNorm}}
\widetilde C^{(20)}
\xrightarrow{\operatorname{RoPE}}
\operatorname{GlobalKVCache}
$$

### 5.8 Global KV 来自 H20 的含义

这里说的是 **Decoder 的 Global KV**，不是 Encoder 内所有 CSA2 层的 Global KV。

设 Prompt 长度为 $N$。经过 Layer 0～19 后得到：

$$
H^{(20)}=
[h_0^{(20)},h_1^{(20)},\ldots,h_{N-1}^{(20)}]
$$

CED 不再要求每个 Prompt token 完整经过 Layer 20～39 才能生成 Decoder Global KV，而是直接：

$$
\operatorname{DecoderGlobalKV}
=\operatorname{Project}\left(H^{(20)}\right)
$$

```text
全部 Prompt
    │
    ▼
Encoder Layer 0～19
    │
    ▼
H^(20)[0:N]
    │
    ├── W_KV / RMSNorm / RoPE ──> Decoder Global KV Cache
    │
    └── 最后 128 个位置 ─────────> Decoder SWA Replay
```

对于很早的 token，例如位置 100，Decoder Layer 35 不需要保存一份由 $H_{100}^{(35)}$ 生成的独立长程 KV。它可以查询由 $H_{100}^{(20)}$ 投影、并由 KV source Layer 20 发布的共享 Global KV entry。

### 5.9 Layer-specific Projection 的含义

单独看 CED 的一般公式，不同 Decoder 层可以拥有不同的投影权重：

$$
C_{21}=H^{(20)}W_{KV}^{(21)}
$$

$$
C_{22}=H^{(20)}W_{KV}^{(22)}
$$

虽然输入都是 $H^{(20)}$，不同的 $W_{KV}^{(l)}$ 可以生成不同层所需的 KV，这就是论文所说的 layer-dependent 或 layer-specific projection。

但是 DeepSeek-V4.1-Flash 又结合了 CSA2 的跨层复用。最终配置不是 20 个 Decoder 层分别保存一份 Global KV，而是：

- Layer 20 是 Decoder 唯一的 KV source；
- Layer 21～39 复用 Layer 20 的 Global KV；
- Layer 24、28、32、36 只重新计算 Indexer Q 和 Top-K，不重新生成 Global KV。

所以更准确的解释是：

> Layer-specific projection 是 CED 的一般机制；在 V4.1-Flash 的具体 CSA2 配置中，投影权重属于 Full/KV-source layer，投影结果由后续共享组复用。

代码对应：

- 只有 `is_kv_source=True` 才实例化 Compressor：`D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:573-585`；
- consumer 通过 source prefix 取得 cache：`D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:1137-1144`。

### 5.10 Local KV 来自各层 SWA 的含义

假设 Decoder Layer 20、21、22 都处理位置 $i$，局部 KV 分别为：

$$
K_i^{\text{local},20},V_i^{\text{local},20}
=f_{KV}^{20}(H_i^{(20)})
$$

$$
K_i^{\text{local},21},V_i^{\text{local},21}
=f_{KV}^{21}(H_i^{(21)})
$$

$$
K_i^{\text{local},22},V_i^{\text{local},22}
=f_{KV}^{22}(H_i^{(22)})
$$

因为：

$$
H_i^{(20)}\ne H_i^{(21)}\ne H_i^{(22)}
$$

所以这些 Local KV 不能共享。

| KV 类型 | 主要职责 | 是否跨层复用 | 来源 |
|---|---|---:|---|
| Global KV | 长距离记忆 | 是 | KV source 的隐藏状态；Decoder 来源是 $H^{(20)}$ |
| Local SWA KV | 最近 token 的逐层精细依赖 | 否 | 当前层自己的隐藏状态 |

Ascend 中每层都用当前 `hidden_states` 生成 SWA KV：

```python
kv = attn.kv_norm(attn.wkv(hidden_states))
scatter_cache_sk(attn.dsa_attn.swa_cache_layer.kv_cache[0], ..., kv)
```

对应：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:271-291`。

稀疏注意力算子同时接收：

```python
ori_kv = 当前层自己的 SWA KV
cmp_kv = KV source layer 的 Global KV
```

对应：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:474-520`。

因此可以概括为：

$$
\boxed{
\text{当前层注意力}
=\text{当前层 Local SWA}
+\text{共享 Global KV 的稀疏 Top-K}
}
$$

### 5.11 为什么必须做 SWA Replay

$H^{(20)}$ 可以直接用于生成 Decoder Global KV，但不能直接生成所有 Decoder 层正确的 Local KV。

例如 Layer 35 的局部 KV 应来自：

$$
H^{(35)}
$$

而不是统一来自：

$$
H^{(20)}
$$

如果完整 Prompt 在 Layer 20～39 全部跳过，那么：

- Decoder Global KV 已经存在；
- 但 Layer 20～39 各层最近 128 个 token 的 Local SWA KV 不存在。

第一个 decode token 到达 Decoder Layer 35 时，需要同时访问：

- 整个 Prompt 中稀疏选出的 Global Top-K；
- 最近 128 个 Prompt token 的 layer-local SWA KV。

因此必须补建每个 Decoder 层的 Local SWA KV，这个补建过程就是 SWA Replay。

### 5.12 最近 nwin 个 Token 做 Replay 的过程

假设：

$$
N=100000,\qquad n_{win}=128
$$

Replay 起点为：

$$
s=N-n_{win}=99872
$$

取 Encoder 最终输出：

$$
H^{(20)}_{99872:100000}
$$

只把这 128 个 token 送过 Decoder 层：

```text
H^(20)[99872:100000]
        │
        ▼
Decoder Layer 20
  生成 Layer 20 最近 128 个位置的 SWA KV
        │
        ▼
Decoder Layer 21
  生成 Layer 21 最近 128 个位置的 SWA KV
        │
        ▼
       ...
        │
        ▼
Decoder Layer 39
  生成 Layer 39 最近 128 个位置的 SWA KV
```

Replay 的目标不是重建整个 Prompt 的 Decoder Global KV，而是：

$$
\boxed{\text{补建各 Decoder 层最近窗口的 Local SWA KV}}
$$

这些 replay 产生的 Decoder SWA KV：

- 只供后续 decode 使用；
- 不写入持久化 prefix cache；
- 不重算或覆盖从 $H^{(20)}$ 构造好的 Global KV。

### 5.13 为什么 Bounded Replay 是近似的

SWA 的单层窗口虽然只有 $W=128$，但多层依赖会逐层向前扩散。

以两层为例：

```text
Layer 21 的位置 99900
    需要读取本层最近 128 个 KV

这些 Layer 21 KV 来自 Layer 20 的输出

而 Layer 20 最早那个位置的输出
    又依赖它之前 128 个 Layer 20 KV
```

因此两层理论有效历史可能扩展到约 $2W$；20 个 Decoder 层则可能需要约：

$$
20W=2560
$$

个 Prompt token，才能严格重构最后窗口中所有层的隐藏状态和 KV。

Bounded Replay 只 replay 最后 128 个 token，会截断 replay 起点之前的局部依赖。论文定义：如果 replay 从位置 $s$ 开始，位置 $i$ 的 query 只允许看到：

$$
[\max(s,i-W+1),i]
$$

而不是完整前向中的：

$$
[\max(0,i-W+1),i]
$$

例如 $W=4$，只 replay 位置 6～9：

```text
完整前向中，位置 6 的 SWA 可看：3, 4, 5, 6
Bounded Replay 中，位置 6 只能看：6

完整前向中，位置 7 的 SWA 可看：4, 5, 6, 7
Bounded Replay 中，位置 7 只能看：6, 7

位置 9：
完整前向和 Replay 都可看：6, 7, 8, 9
```

Replay 区间最前面的 token 误差较大，越靠近区间末尾，可见局部窗口越完整。

这种近似仍然有效的原因主要有：

1. Global KV 仍然覆盖完整 Prompt，局部依赖被截断不等于完整历史消失；
2. 越接近生成边界，Replay token 的局部窗口越完整；
3. Post-training 会模拟相同 replay，使模型适应这种近似状态。

### 5.14 完整小例子

假设简化为：

- 4 层 Encoder；
- 4 层 Decoder；
- Prompt 长度 $N=8$；
- SWA 窗口 $W=3$。

#### Encoder 阶段

```text
Token 0～7
   │
   ▼
Encoder Layer 0～3
   │
   ▼
H^(4)[0～7]
```

#### 构造 Decoder Global KV

```text
H^(4)[0～7]
   │
   └── W_KV ──> Global KV[0～7]
```

不需要让 token 0～4 完整运行全部四层 Decoder，才能得到长程 KV。

#### SWA Replay

只取最后三个 Encoder output：

$$
H_5^{(4)},H_6^{(4)},H_7^{(4)}
$$

```text
H^(4)[5,6,7]
   │
   ▼
Decoder 0：生成 local KV_0[5,6,7]
   │
   ▼
Decoder 1：生成 local KV_1[5,6,7]
   │
   ▼
Decoder 2：生成 local KV_2[5,6,7]
   │
   ▼
Decoder 3：生成 local KV_3[5,6,7]
```

#### 生成第一个新 token 8

某个 Decoder 层可以使用：

```text
Global 分支：从位置 0～7 的 Global KV 中选 Top-K
Local 分支：访问当前层 replay 出来的位置 5、6、7
```

因此：

$$
\operatorname{Context}(8)=
\operatorname{GlobalTopK}(0:7)+
\operatorname{LocalSWA}(5:7)
$$

CED 的核心就是：

> 远距离历史通过共享 Global KV 保留；近距离逐层状态通过少量 SWA Replay 恢复。

### 5.15 vLLM 与 vLLM-Ascend 当前实现差异

#### 5.15.1 vLLM CUDA 路径

vLLM 有专门的 `DecoderReplayLayers`：

- 每个请求只收集最后 `window` 行；
- 用子 batch 执行 replay layers；
- 再将结果 scatter 回原 batch。

对应：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\decoder_replay_layers.py:3-11,45-67`。

模型以最后一个 KV source 为切分点：

```python
cut = max(config.kv_source_layer_ids)  # 20
decoder_replay_start = cut + 1         # 21
```

在 eager prefill 中：

- Layer 0～20 运行完整 token batch；
- Layer 21～39 只运行最后一个窗口。

代码位置：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:757-784`；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:955-978`。

所以当前 vLLM 实现的粗略 layer-token 次数更接近：

$$
N\times21+W\times19
$$

而不是论文抽象公式中严格的：

$$
N\times20+W\times20
$$

因为 Decoder Full/KV-source Layer 20 被用作实际执行边界。对于 $N\gg W$，两者仍然接近“prefill 减半”。

#### 5.15.2 当前 vLLM-Ascend 快照

Ascend 已经实现：

- 每层自己的 SWA cache；
- KV source 的 long compressed KV cache；
- ratio=1/2 Compressor；
- consumer 读取 source cache；
- `ori_kv + cmp_kv` 联合注意力。

但当前高层 forward 仍然是：

```python
for layer in self.layers:
    hidden_states, pre_mix = layer(...)
```

代码位置：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:1245-1275`。

当前快照中没有看到与上游 `DecoderReplayLayers` 完全对等的“抽取每个请求最后 128 行，再只执行 Layer 21～39”的高层路径。因此必须区分：

- KV 数据结构和 Global/Local cache 语义已经实现；
- 但不能只凭当前 Ascend 静态源码认定论文所述 prefill 减半已经完整落地；
- 这是静态源码观察，不是 NPU runtime profiling 结论。

### 问题 5 最终总结

对于 Decoder 第 $l$ 层，当前 query 的注意力输入可以抽象为：

$$
O_i^{(l)}=
\operatorname{Attention}
\left(
Q_i^{(l)},
\left[
\underbrace{K_{\text{global},\operatorname{TopK}(i)}^{(20)}}_{\text{由 }H^{(20)}\text{ 投影并跨层复用}},
\underbrace{K_{\text{local},i-W+1:i}^{(l)}}_{\text{由当前层隐藏状态生成}}
\right],
\left[V_{\text{global}},V_{\text{local}}^{(l)}\right]
\right)
$$

核心结论如下：

1. Layer 0～19 仍然是因果自回归 Transformer 计算，不是双向 Encoder。
2. Layer 0～1 为纯 SWA；Layer 2～19 使用 ratio=2 的 CSA2 Global KV，并按 2/8/14 三个 source group 跨层复用。
3. $W_{KV}$ 把隐藏状态投影成 512 维 KV latent；$W_Z$ 在 ratio 大于 1 时生成逐通道压缩门控，代码中叫 `wgate`。
4. Decoder Global KV 由 $H^{(20)}$ 投影而来；具体 V4.1-Flash 配置中只有 Layer 20 真正生成 Decoder Global KV，后续层复用。
5. Local SWA KV 必须由每个 Decoder 层自己的隐藏状态生成，无法直接用 $H^{(20)}$ 代替。
6. SWA Replay 只把最近 $n_{win}=128$ 个 Encoder output 送过 Decoder，用于补建逐层 Local KV。
7. Bounded Replay 会截断 replay 区间之前的局部依赖，因此不是数学上的完整 Decoder forward；Global KV、生成边界附近逐渐完整的窗口和训练适配共同降低其影响。
8. CED 能让大部分 Prompt 跳过完整 Decoder 的根因是 Global KV 的来源从各 Decoder 层隐藏状态中解耦，而不是“只保存滑窗就足够理解整个 Prompt”。

最终可以用一条式子概括：

$$
\boxed{
\text{完整 Prompt 的 Encoder-derived Global KV}
+
\text{最近窗口的 Decoder layer-local SWA KV}
}
$$

---

## 问题 6：Ratio=2 的 Token/KV 压缩原理与是否需要复原

### 问题 6 原文

> Token 压缩的原理和具体实施方式是怎样做的？可以用 ratio=2 作为例子。既然有 token 压缩，生成输出的时候是不是也有 token 复原？

### 问题 6 核心结论

DeepSeek-V4.1-Flash 所说的“token 压缩”不是把两个离散文本 token 合成一个新 token，也不会把 Transformer 的两行 hidden state 合成一行后继续向后计算。

它真正压缩的是 **Global/Main KV cache 的时间轴**：两个 token 各自产生一份候选 KV latent，再通过学习得到的逐通道门控，将两份 latent 汇聚成一个缓存 entry。

因此，生成输出时不存在“把一个压缩 token 还原成两个 token”的过程。这个压缩是有损的、面向注意力记忆的聚合，并不是可逆编码。

在 DeepSeek-V4.1-Flash 的具体配置中：

- Causal Encoder 的 Layer 2～19 使用 `ratio=2`；
- Decoder 的 Layer 20～39 使用 `ratio=1`；
- 所有 token 的 hidden state 主干仍然逐 token 计算；
- 只有 Encoder 的 Global KV 时间轴按 2:1 压缩。

本节核对的本地源码版本为：

- `D:\Code\vllm-github`：`5f30fc7031cae49bf51073fc953d419b08f8887c`；
- `D:\Code\vllm-ascend\vllm-ascend`：`4ebb3090fc3c13c26557dcaae353882e70246649`；
- 论文重点参考 Section 2.2、2.3、3.2.2 和 4.2.1。

### 6.1 Token 压缩到底压缩了什么

设输入序列包含 $N$ 个 token，某层 hidden states 为：

$$
H\in\mathbb R^{N\times d_{model}}
$$

ratio=2 并不会把 hidden states 改成：

$$
H'\in\mathbb R^{N/2\times d_{model}}
$$

实际 hidden state 的 token 维度保持不变：

$$
H\in\mathbb R^{N\times5120}
$$

压缩的是 Global/Main KV：

$$
\operatorname{GlobalKV}\in
\mathbb R^{\lfloor N/2\rfloor\times512}
$$

| 数据 | ratio=2 后 token 数量是否减半 | 说明 |
|---|---:|---|
| 输入 token IDs | 否 | 文本 token 没有合并 |
| Transformer hidden states | 否 | 每个 token 仍有一行 |
| Query | 否 | 每个 token 仍生成自己的 Query |
| MoE/MLP 输入 | 否 | 每个 token 仍单独通过网络 |
| Layer-local SWA KV | 否 | 最近窗口仍按原始 token 保存 |
| Global/Main KV cache | 是 | 每两个 token 汇聚成一个 entry |
| Indexer K | 是 | 从压缩后的 Main KV 生成 |
| 最终输出 token | 否 | 仍然一次生成一个离散 token |

所以更准确的名称是：

$$
\boxed{\text{Global KV sequence compression}}
$$

### 6.2 Ratio=2 的整体处理流程

假设某个 KV source layer 收到四个 token 的隐藏状态：

$$
h_0,h_1,h_2,h_3,\qquad h_i\in\mathbb R^{5120}
$$

ratio=2 将它们划分为两个不重叠分组：

$$
G_0=\{h_0,h_1\},\qquad G_1=\{h_2,h_3\}
$$

最终得到两份 Global KV latent：

$$
\widetilde c_0=\operatorname{Compress}(h_0,h_1)
$$

$$
\widetilde c_1=\operatorname{Compress}(h_2,h_3)
$$

整体流程如下：

```text
原始 hidden states，仍然有 4 行
──────────────────────────────────────────────
h0 ── W_KV ──> c0 ─┐
                    ├─ 逐通道 Softmax Pool ─> c̃0 ─> RMSNorm ─> RoPE ─> KV entry 0
h1 ── W_KV ──> c1 ─┘
      W_Z 产生 z0、z1，作为汇聚权重

h2 ── W_KV ──> c2 ─┐
                    ├─ 逐通道 Softmax Pool ─> c̃1 ─> RMSNorm ─> RoPE ─> KV entry 1
h3 ── W_KV ──> c3 ─┘
      W_Z 产生 z2、z3，作为汇聚权重
──────────────────────────────────────────────

Global KV cache：4 个 token → 2 个 entry
Hidden state 主干：仍然是 4 个 token
```

论文 Section 2.3 指出，CSA2 的 ratio $m$ 表示将连续 $m$ 个 token 压缩成一个 Main KV entry。与之前 CSA 使用重叠窗口不同，CSA2 使用不重叠分组，并移除了额外的绝对位置编码。

### 6.3 每个 Token 如何产生 KV 内容和门控

对于每个 token 的隐藏状态 $h_i$，Compressor 做两次线性投影。

#### 6.3.1 KV 内容投影

$$
c_i=h_iW_{KV}
$$

其中：

$$
h_i\in\mathbb R^{5120},\qquad
W_{KV}\in\mathbb R^{5120\times512}
$$

因此：

$$
c_i\in\mathbb R^{512}
$$

$c_i$ 表示 token $i$ 对 Global KV 内容的候选贡献。

#### 6.3.2 压缩门控投影

$$
z_i=h_iW_Z
$$

代码中 $W_Z$ 称为 `wgate`：

$$
W_Z\in\mathbb R^{5120\times512},\qquad
z_i\in\mathbb R^{512}
$$

$z_i$ 不是一个标量，而是一个 512 维向量。这意味着对于不同 latent 通道，两个 token 可以具有不同的重要性。

vLLM-Ascend 中对应：

```python
self.wkv = nn.Linear(dim, self.width, bias=False, ...)
self.wgate = nn.Linear(dim, self.width, bias=False, ...)
```

代码位置：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\compressor.py:14-23`。

上游 vLLM 将两个投影融合到一个 `MergedColumnParallelLinear` 中。ratio=2 时输出形状为：

$$
[\text{num tokens},1024]
$$

前 512 维为 $c_i$，后 512 维为 $z_i$。对应：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\compressor.py:204-237`；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:920-955`。

### 6.4 两个 Token 如何压成一个 Entry

对于分组：

$$
G_g=\{2g,2g+1\}
$$

先计算：

$$
c_{2g}=h_{2g}W_{KV},\qquad
c_{2g+1}=h_{2g+1}W_{KV}
$$

$$
z_{2g}=h_{2g}W_Z,\qquad
z_{2g+1}=h_{2g+1}W_Z
$$

对每个 latent 通道 $d$，在两个 token 之间执行 softmax：

$$
p_{2g}[d]=
\frac{\exp(z_{2g}[d])}
{\exp(z_{2g}[d])+\exp(z_{2g+1}[d])}
$$

$$
p_{2g+1}[d]=
\frac{\exp(z_{2g+1}[d])}
{\exp(z_{2g}[d])+\exp(z_{2g+1}[d])}
$$

满足：

$$
p_{2g}[d]+p_{2g+1}[d]=1
$$

最终压缩 latent 为：

$$
\widetilde c_g[d]
=p_{2g}[d]c_{2g}[d]
+p_{2g+1}[d]c_{2g+1}[d]
$$

向量形式为：

$$
\boxed{
\widetilde c_g
=\sum_{r=0}^{1}
\operatorname{Softmax}_{r}(z_{2g+r})
\odot c_{2g+r}
}
$$

这不是简单平均：

$$
\widetilde c_g\ne\frac{c_{2g}+c_{2g+1}}2
$$

而是训练得到的逐通道加权汇聚。

### 6.5 数值例子

为了便于观察，假设 KV latent 只有两个通道：

$$
c_0=[10,0],\qquad c_1=[0,8]
$$

门控分数为：

$$
z_0=[2,0],\qquad z_1=[0,2]
$$

第一个通道：

$$
\operatorname{Softmax}(2,0)\approx(0.881,0.119)
$$

$$
\widetilde c[0]=0.881\times10+0.119\times0=8.81
$$

第一个通道主要保留 token 0 的信息。

第二个通道：

$$
\operatorname{Softmax}(0,2)\approx(0.119,0.881)
$$

$$
\widetilde c[1]=0.119\times0+0.881\times8\approx7.05
$$

第二个通道主要保留 token 1 的信息。

最终：

$$
\widetilde c=[8.81,7.05]
$$

因此，逐通道门控可以表现为：

```text
通道 0 主要选择 token 0
通道 1 主要选择 token 1
通道 2 可能平均二者
通道 3 可能更关注 token 1
...
```

不过从维度看：

$$
\mathbb R^{2\times512}\rightarrow\mathbb R^{512}
$$

仍然是有损映射，无法完整恢复原来的两个向量。

### 6.6 代码中的压缩公式

vLLM-Ascend 提供了清晰的纯 PyTorch 参考实现：

```python
kv = torch.nn.functional.linear(xs.float(), wkv.float())
scores = torch.nn.functional.linear(xs.float(), wgate.float())

group_tokens = tokens[g * ratio : (g + 1) * ratio]
kv_part = group_tokens[:, :head_dim]
score_part = group_tokens[:, head_dim:]

prob = score_part.softmax(dim=0)
pooled = (kv_part * prob).sum(dim=0)
```

代码位置：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\ops\triton\compressor\compressor_triton.py:567-620`。

其中：

- `kv_part.shape=[2,512]`；
- `score_part.shape=[2,512]`；
- `softmax(dim=0)` 表示在两个 token 之间做 softmax；
- 每个通道有独立的两个权重；
- `pooled.shape=[512]`。

上游 vLLM Triton kernel 使用相同公式：

```python
pooled = tl.sum(kv * tl.softmax(score, 0), 0)
```

对应：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\common\ops\fused_compress_quant_cache.py:140-221`。

单元测试也按照下式计算参考输出：

```python
pooled = (group[:, :512] * group[:, 512:].softmax(0)).sum(0)
```

对应：

- `D:\Code\vllm-github\tests\kernels\test_compressor_kv_cache.py:280-299`。

### 6.7 RMSNorm、RoPE 与缓存写入

两个 token 汇聚后得到：

$$
\widetilde c_g\in\mathbb R^{512}
$$

随后执行 RMSNorm：

$$
\widehat c_g=
\frac{\widetilde c_g}
{\sqrt{\frac1{512}\sum_{d=1}^{512}\widetilde c_g[d]^2+\epsilon}}
\odot\gamma
$$

上游 kernel 对应：

```python
variance = tl.sum(pooled * pooled, 0) / 512
normed = pooled * tl.rsqrt(variance + EPS) * weight
```

代码位置：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\common\ops\fused_compress_quant_cache.py:215-221`。

完整缓存路径为：

```text
压缩 latent
    │
    ▼
RMSNorm
    │
    ▼
部分 RoPE
    │
    ▼
BF16 / FP8 / FP4 数值格式处理
    │
    ├── 写入 Main/Global KV Cache
    └── 投影生成 Indexer K
```

需要区分三种不同的压缩维度：

| 压缩维度 | 方式 | ratio=2 是否属于它 |
|---|---|---:|
| Entry 大小 | MLA latent、FP8、FP4 | 否 |
| Token/序列维度 | 每两个 token 合成一个 KV entry | 是 |
| Layer 维度 | 多层复用同一个 KV source | 否 |

所以 `ratio=2` 不是“两比特量化”，而是：

$$
\boxed{\text{2 个原始 token 对应 1 个 Global KV entry}}
$$

### 6.8 压缩位置和 Slot 映射

对于零基 token 位置：

```text
位置 0、1 → compressed entry 0
位置 2、3 → compressed entry 1
位置 4、5 → compressed entry 2
```

只有分组末尾 token 触发写入：

$$
(p+1)\bmod2=0
$$

即位置：

$$
1,3,5,7,\ldots
$$

上游 slot mapping 代码为：

```python
is_valid = ((pos + 1) % COMPRESS_RATIO == 0)
pos_after_compress = pos // COMPRESS_RATIO
```

对应：

- `D:\Code\vllm-github\vllm\v1\attention\backends\mla\compressor_utils.py:62-92`。

代码注释明确说明：每 `compress_ratio` 个 token 共用一个 compressed state，由分组最后一个 token 负责写入，其他 token 的 compressed slot 为 PAD：

- `D:\Code\vllm-github\vllm\v1\attention\backends\mla\compressor_utils.py:161-200`。

Ascend 对应逻辑为：

```python
valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)
return torch.where(valid, slot_mapping // ratio, -1)
```

代码位置：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:177-184`。

#### 6.8.1 RoPE 锚点

当位置 1 完成 $\{0,1\}$ 分组时，压缩 entry 使用分组起点 0 的 RoPE；位置 3 完成 $\{2,3\}$ 时使用位置 2。

代码使用：

```python
position // COMPRESS_RATIO * COMPRESS_RATIO
```

即：

$$
p_{anchor}=\left\lfloor\frac p2\right\rfloor\times2
$$

对应：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\common\ops\fused_compress_quant_cache.py:320-345`。

### 6.9 奇数长度与 Residual 的处理

假设当前序列长度为 7，位置为 0～6：

```text
位置 0、1 → entry 0，已经完成
位置 2、3 → entry 1，已经完成
位置 4、5 → entry 2，已经完成
位置 6   → 尚未凑到位置 7，暂时是 residual
```

此时：

$$
N_{completed}=\left\lfloor\frac72\right\rfloor=3
$$

$$
N_{residual}=7\bmod2=1
$$

Global KV cache 中有三个完成 entry，同时 Compressor state cache 暂存位置 6 的：

$$
[c_6,z_6]
$$

当位置 7 到来时：

$$
[c_6,z_6]+[c_7,z_7]
\longrightarrow\widetilde c_3
$$

然后写入第 4 个 Global KV entry。

Ascend metadata 直接计算：

```python
compressed_length = seq_len // ratio
residual = seq_len % ratio
```

对应：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:823-833`。

上游实现使用每请求独立的 ring state cache 来保存尚未闭合分组的 KV/score 状态，可以处理 chunked prefill、单 token decode、不同请求的奇偶长度以及 paged-cache 边界：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\common\ops\fused_compress_quant_cache.py:22-60`。

### 6.10 Chunked Prefill 如何跨 Chunk 配对

假设 ratio=2，但一对 token 被调度到不同 chunk：

```text
Chunk A：... token 100
Chunk B：token 101, token 102, ...
```

token 100 到来时还不能闭合分组：

```text
Ring state：保存 c100 和 z100
```

处理 Chunk B 时，token 101 到来，Compressor 读取 ring state 中的 token 100：

$$
\widetilde c_{50}=
\operatorname{Pool}
\left([c_{100},z_{100}],[c_{101},z_{101}]\right)
$$

随后 token 102 又成为下一组的 residual。

上游 kernel 在 chunk 第一个 token 恰好闭合历史分组时，会从 ring 中读取前一个 token：

```python
prev = ring[(position - 1) % STATE_BLOCK]
current = raw[start]
pooled = weighted_pool(prev, current)
```

对应：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\common\ops\fused_compress_quant_cache.py:140-193`。

所以分组依据绝对 token position，不会因 chunked prefill 的切分位置而改变。

### 6.11 压缩 KV 如何参加注意力

如果原始序列有 100K token，ratio=2 后 Main KV 大约有：

$$
\frac{100000}{2}=50000
$$

个 entry。

Indexer 在约 50K 个压缩 entry 上打分并选择 Top-K。对于 query $q_i$：

$$
s_{i,g}=q_i^{index}\cdot k_g^{index}
$$

$$
\mathcal T_i=\operatorname{TopK}_g(s_{i,g})
$$

Global attention 输出可近似写为：

$$
o_i^{global}=
\sum_{g\in\mathcal T_i}\alpha_{i,g}\widehat c_g
$$

同时，当前 query 还会读取当前层的未压缩 SWA KV：

$$
o_i=\operatorname{Attention}
\left(
q_i,
\text{Selected Compressed Global KV},
\text{Layer-local SWA KV}
\right)
$$

Ascend 稀疏注意力算子同时接收：

```python
ori_kv = 当前层自己的 SWA KV
cmp_kv = 压缩后的 Global KV
```

对应：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:474-520`。

因此模型的记忆可以理解为：

```text
较远历史
    └── 压缩 Global KV，降低缓存和索引规模

最近 128 个 token
    └── 未压缩、逐层独立的 SWA KV

当前 token
    └── 仍有自己的 Query、hidden state 和完整 Transformer 计算
```

### 6.12 生成时是否需要 Token 复原

不需要，而且原则上也无法完整复原。

#### 6.12.1 为什么不需要复原

Attention 的目标不是从 KV cache 中反解原始 token，而是根据当前 Query 读取对当前计算有帮助的向量摘要。

普通 Attention 也不会从 $V_j$ 中还原 token，而是直接计算：

$$
o_i=\sum_j\alpha_{ij}V_j
$$

CSA2 将远程历史从逐 token 的 $V_j$ 换成按组汇聚的 $\widetilde V_g$：

$$
o_i^{global}=\sum_g\alpha_{i,g}\widetilde V_g
$$

Attention output 随后继续经过输出投影、残差/mHC、MoE 和后续层，最终由 LM Head 计算词表 logits：

$$
\operatorname{logits}_t=h_t^{final}W_{vocab}^{T}
$$

代码为：

```python
logits = self.logits_processor(self.lm_head, hidden_states)
```

对应：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:1564-1569`；
- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:1414-1419`。

生成的是新的词表 token，而不是从 compressed KV 中恢复旧 token。

#### 6.12.2 为什么无法完整复原

压缩前有两个 512 维向量：

$$
(c_0,c_1)\in\mathbb R^{1024}
$$

压缩后只有一个 512 维向量：

$$
\widetilde c\in\mathbb R^{512}
$$

这是：

$$
\mathbb R^{1024}\rightarrow\mathbb R^{512}
$$

大量不同的 $(c_0,c_1)$ 可能得到相同的 $\widetilde c$，所以不存在唯一逆映射：

$$
\widetilde c\not\Rightarrow(c_0,c_1)
$$

它是学习得到的有损摘要，而不是无损编码。

#### 6.12.3 原始 token 是否全部消失

需要分层理解：

- 原始 token ID 可能仍存在于请求管理和调度侧；
- Transformer hidden-state 主干仍然为每个 token 保留独立行；
- 最近 128 个 token 的 layer-local SWA KV 仍按 token 保存；
- 只有较远历史的 Global KV 不再保存两份独立 latent，而只保存一个 learned summary。

所以“没有复原”不等于“模型只知道半个 Prompt”。模型依靠：

$$
\boxed{
\text{压缩后的全局长程记忆}
+
\text{未压缩的局部 SWA 记忆}
}
$$

共同工作。

### 6.13 Decode 阶段的 Ratio=2 缓存更新

生成时，每个新 token 仍然逐个进入模型。

假设 prefill 后已有位置 0～4：

```text
位置 0、1 → Global entry 0
位置 2、3 → Global entry 1
位置 4   → residual，等待位置 5
```

位置 4 的最终 hidden state 用于采样位置 5 的 token。

下一次 decode forward 输入位置 5：

```text
位置 4 的历史 state
+
位置 5 的当前 state
──────────────
形成 Global entry 2
```

位置 5 的最终 hidden state 用来预测位置 6。

输入位置 6 时，位置 6 是下一组的第一个 token，只保存 residual；输入位置 7 时，位置 6 和 7 合并形成 entry 3。

| 当前输入位置 | Global KV 动作 | 是否预测下一个 token |
|---:|---|---:|
| 4 | 保存 residual | 是，预测位置 5 |
| 5 | 将 4、5 压成 entry 2 | 是，预测位置 6 |
| 6 | 保存 residual | 是，预测位置 7 |
| 7 | 将 6、7 压成 entry 3 | 是，预测位置 8 |

因此：

> 每两个 token 才写出一个 ratio=2 Global KV entry，不等于每两个 token 才生成一次输出。

生成频率仍然是：

$$
\boxed{\text{一次 decode forward 生成一个新 token}}
$$

而且 DeepSeek-V4.1-Flash 的 Decoder 本身使用 ratio=1；ratio=2 主要位于 Causal Encoder 的 Layer 2～19。

### 6.14 为什么不会破坏因果性

以分组 $\{4,5\}$ 为例：

- query 位置 4 执行时，位置 5 还不存在；
- 此时不会提前产生 $\{4,5\}$ 的 compressed entry；
- 只保存位置 4 的 residual state；
- 位置 5 到来后，才闭合并发布 $\{4,5\}$ 的 entry。

写入条件为：

$$
(p+1)\bmod2=0
$$

因此位置 4 不会看到未来位置 5。位置 5 可以使用包含自身的 KV，这与普通 causal self-attention 中当前 query 可以访问自己的 K/V 一致。

另外，位置 4 仍存在于 Local SWA KV 中，所以即使 Global 分组尚未闭合，它也不会从当前局部注意力上下文中消失。

### 6.15 Token 压缩的实际收益与边界

假设上下文长度为 100K。

不做 token 压缩：

$$
N_{GlobalKV}=100000
$$

ratio=2：

$$
N_{GlobalKV}=50000
$$

直接收益包括：

1. Global KV cache 的序列长度约减半；
2. Indexer K cache 长度约减半；
3. Indexer 搜索空间约减半；
4. Sparse attention 的候选 entry 数减少；
5. 配合跨层 KV reuse 后，多层不再各自保存完整长程缓存。

但边界也很明确：

- 它不直接把 MoE/MLP 的 token 计算量减半；
- 它只压缩 Global KV，SWA local cache 仍按原始 token 工作；
- 从两个 512 维 latent 到一个 512 维 latent 是有损压缩；
- 长程细粒度信息的损失由 learned pooling、稀疏检索、Local SWA 和端到端训练共同补偿。

DeepSeek-V4.1-Flash 的整体收益来自多种机制叠加：

```text
CED
└── 大部分 Prompt 跳过完整 Decoder 计算

Ratio=2 token compression
└── Encoder Global KV 时间轴减半

MLA / FP8 / FP4
└── 单个 KV entry 更小

Cross-layer reuse
└── 多层共享 Global KV

Sparse Top-K
└── 每个 Query 只读取少量 Global KV entry

SWA
└── 最近窗口保留高分辨率逐 token 信息
```

### 问题 6 最终总结

ratio=2 的 token/KV 压缩可以概括为：

$$
\boxed{
\begin{aligned}
c_i &= h_iW_{KV}\\
z_i &= h_iW_Z\\
\widetilde c_g
&=\sum_{i=2g}^{2g+1}
\operatorname{Softmax}_{i}(z_i)\odot c_i\\
\widehat c_g
&=\operatorname{RMSNorm}(\widetilde c_g)
\end{aligned}
}
$$

随后将 $\widehat c_g$ 经过 RoPE 和数值格式处理，写入一个 Global KV cache entry。

最终结论如下：

1. 压缩的是 Global KV cache 的序列维度，不是 tokenizer token 或 hidden-state 主干。
2. ratio=2 表示两个连续、不重叠的 token 形成一个 Global KV entry。
3. 每个 token 先分别通过 $W_{KV}$ 产生内容向量，通过 $W_Z$ / `wgate` 产生逐通道门控。
4. 两个 token 通过逐通道 softmax weighted pooling 汇聚，不是简单平均。
5. 只有分组末尾 token 发布 compressed entry，未闭合 token 作为 residual 保存在 state cache 中。
6. Chunked prefill 和单 token decode 可以通过 ring state 跨调用完成配对。
7. 当前 query 同时使用压缩 Global KV 和未压缩 Local SWA KV。
8. 生成输出时不需要 token 复原；最终 token 由最终 hidden state 经过 LM Head 产生。
9. 从两个 512 维 latent 到一个 512 维 latent 是有损映射，原则上无法唯一恢复原始两个 latent。
10. 每两个 token 写一次 ratio=2 Global KV，不影响模型仍然每次 decode 生成一个 token。

一句话概括：

$$
\boxed{
\text{Token compression 是“把远程注意力记忆做成学习型摘要”，而不是压缩后再解压文本 Token}
}
$$

---

## 问题 7：旧版 CSA 的 Overlap 与 CSA2 的 Softmax Pooling

### 问题 7 原文

> 对于 token 压缩，有如下说法：“每 $2m$ 个相邻的 token 会被压缩成一组，2 个相邻的组会有 $m$ 个 overlap，所以实际压缩比例是 $1:m$。这里的压缩就是 softmax 池化：使用一个可训练矩阵计算每个 token 的权重，然后加权平均。DeepSeek-V4.1 用的 $m=2$ 和 $1$。”我想知道这个说法是不是来自于什么论文？以及它是否正确，其中的 overlap 真的存在吗？毕竟你之前的分析都没有提到。还有，softmax 计算就是 softmax 计算，为什么强调 softmax 池化？

### 问题 7 核心结论

这段说法不是凭空来的，它主要源自两处：

1. **DeepSeek-V4 原始 CSA 论文**：明确提出一个压缩 KV 由 $2m$ 个源位置参与计算，相邻压缩 KV 的源位置有 $m$ 个重叠，因此序列长度最终缩短为原来的 $1/m$。
2. **DeepSeek-V4.1-Flash 论文 Section 2.3**：在解释 CSA2 相比 CSA 的简化时，再次概括了上述机制，并明确说明 **CSA2 删除了 overlap 和绝对位置偏置**。

但是，题目中引用的整句话把两代设计混在了一起：

| 说法 | 判断 |
|---|---|
| 旧版 CSA 每个压缩 KV 由 $2m$ 个源 token 位置参与计算 | 基本正确 |
| 相邻压缩 KV 有 $m$ 个源 token 位置重叠 | 对旧版 CSA 正确 |
| 所以输出长度是输入的 $1/m$ | 正确，决定比例的是步长 $m$，不是窗口宽度 $2m$ |
| DeepSeek-V4.1 使用 $m=2$ 和 $m=1$ | 正确 |
| DeepSeek-V4.1 的 $m=2$ 也使用 $2m=4$ 的重叠窗口 | **错误**，CSA2 已经删除 overlap |
| “压缩就是 softmax 池化” | 大方向正确，但完整操作是“线性投影＋softmax 归一化＋沿 token 维加权求和” |

最准确的表述是：

> DeepSeek-V4 的 CSA 使用宽度 $2m$、步长 $m$ 的重叠式 gated pooling；DeepSeek-V4.1 的 CSA2 改成宽度 $m$、步长 $m$ 的非重叠 gated pooling。V4.1 中 $m=2$ 才真正把两个 token 压成一个，$m=1$ 则退化为不压缩，并省略 gate。

本次分析核对的代码版本为：

- `D:\Code\vllm-github`：commit `5f30fc7031cae49bf51073fc953d419b08f8887c`；
- `D:\Code\vllm-ascend\vllm-ascend`：commit `4ebb3090fc3c13c26557dcaae353882e70246649`；
- 本地论文：`D:\Code\vllm_notes\DeepSeekV41-Flash\DeepSeek-V4.1-Flash Pushing the Limits of KV Cache Compression.pdf`，重点参考 Section 2.3。

### 7.1 这个说法来自哪篇论文

#### 7.1.1 原始来源：DeepSeek-V4

DeepSeek-V4 论文中的 CSA 压缩公式为：

$$
C^a=HW^{aKV},\qquad C^b=HW^{bKV}
$$

$$
Z^a=HW^{aZ},\qquad Z^b=HW^{bZ}
$$

其中：

- $C^a,C^b$ 是同一批 hidden states 经过两个不同投影得到的两组 KV 内容；
- $Z^a,Z^b$ 是对应的压缩权重 logits；
- 每个 token 的权重不是一个标量，而是一个 $c$ 维向量。

第 $i$ 个压缩 KV 使用当前 block 的：

$$
C^a_{mi:m(i+1)-1}
$$

以及前一个 block 的：

$$
C^b_{m(i-1):mi-1}
$$

两部分拼起来一共是 $2m$ 个条目。权重计算为：

$$
[S^a;S^b]
=
\operatorname{Softmax}_{\text{token}}
([Z^a+B^a;Z^b+B^b])
$$

最终压缩结果为：

$$
C_i^{\mathrm{Comp}}
=
\sum_{j=mi}^{m(i+1)-1}S_j^a\odot C_j^a
+
\sum_{j=m(i-1)}^{mi-1}S_j^b\odot C_j^b
$$

论文随后明确说明：每个压缩条目虽然涉及 $2m$ 个源条目，但相邻压缩条目会共享 $m$ 个源 token 位置，所以最终长度仍然是 $n/m$。

论文来源：

- DeepSeek-AI, *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, Section 2.3.1，公式 9～12；
- 在线版本：<https://arxiv.org/html/2606.19348v1>。

因此，“$2m$ 个 token、相邻 overlap $m$、实际压缩到 $1/m$”确实来自 DeepSeek-V4 的 CSA，而不是其他无关的 token merging 论文。

#### 7.1.2 V4.1 论文为什么再次提到它

DeepSeek-V4.1-Flash 论文 Section 2.3 专门对比了 CSA 和 CSA2：

- 旧 CSA：每个压缩 main KV 来自 $2m$ 个源条目；
- 相邻输出存在源位置重叠；
- 旧 CSA 还使用 absolute positional embedding 表示这 $2m$ 个位置；
- CSA2 删除 overlap 和 absolute positional embedding；
- CSA2 的 indexer K 也不再从 hidden states 走独立压缩路径，而是从 main KV 投影得到。

因此，题目中的说法很可能是对 V4.1 论文这段比较文字的转述。但转述者又把 V4.1 的 $m=2,1$ 接在旧 CSA 的 $2m$ overlap 机制之后，造成了机制混用。

论文来源：

- DeepSeek-AI, *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*, Section 2.3；
- 在线版本：<https://arxiv.org/html/2609.19969v1>。

### 7.2 旧版 CSA 的 Overlap 如何发生

以 $m=2$ 为例。先忽略 $a/b$ 两套投影的区别，只观察原始 token 位置：

```text
原始位置：
t0  t1  t2  t3  t4  t5  t6  t7 ...

压缩项 C0： [padding] [padding] [t0 t1]
压缩项 C1：                   [t0 t1] [t2 t3]
压缩项 C2：                            [t2 t3] [t4 t5]
压缩项 C3：                                     [t4 t5] [t6 t7]
```

更抽象地说：

$$
G_i=\{(i-1)m,\ldots,im-1\}\cup\{im,\ldots,(i+1)m-1\}
$$

窗口宽度为：

$$
|G_i|=2m
$$

相邻窗口的交集为：

$$
G_i\cap G_{i+1}=\{im,\ldots,(i+1)m-1\}
$$

所以：

$$
|G_i\cap G_{i+1}|=m
$$

但窗口每次向前移动的距离也是 $m$，也就是 stride 为 $m$。因此，每加入 $m$ 个新 token 才产生一个新压缩条目：

$$
N_{\mathrm{compressed}}\approx\frac{N}{m}
$$

所以：

$$
\text{输出条目数}:\text{输入 token 数}=1:m
$$

这里的关键是：

> 压缩比例由 stride 决定，而不是由窗口宽度决定。

这和滑动窗口卷积、滑动窗口池化的计数逻辑相同。窗口宽度为 4、步长为 2 时，虽然每个输出观察 4 个输入，输出数量仍然约为输入的一半。

不过，旧 CSA 不只是把同一套 KV 放进一个普通的 $2m$ 滑窗。以 $m=2$ 为例：

$$
C_1^{\mathrm{Comp}}
=S^b_0\odot C^b_0+S^b_1\odot C^b_1
+S^a_2\odot C^a_2+S^a_3\odot C^a_3
$$

下一个压缩条目是：

$$
C_2^{\mathrm{Comp}}
=S^b_2\odot C^b_2+S^b_3\odot C^b_3
+S^a_4\odot C^a_4+S^a_5\odot C^a_5
$$

二者共享原始 token 位置 2、3，但：

$$
C^a_2\ne C^b_2
$$

因为：

$$
C^a_2=H_2W^{aKV},\qquad C^b_2=H_2W^{bKV}
$$

也就是说，重叠的是源 token 位置；相邻窗口中同一个位置经过的投影分支不同，并不是把完全相同的 KV 向量机械复制两次。

### 7.3 vLLM 的 DeepSeek-V4 代码如何实现 Overlap

旧版 V4 compressor 在：

- `D:\Code\vllm-github\vllm\models\deepseek_v4\compressor.py:239-240`

直接定义：

```python
self.overlap = compress_ratio == 4
self.coff = 1 + self.overlap
```

因此：

- CSA，`compress_ratio=4`：
  - `overlap=True`；
  - `coff=2`；
  - 源窗口宽度为 $2m=8$。
- HCA，`compress_ratio=128`：
  - `overlap=False`；
  - `coff=1`；
  - 源窗口宽度为 $m=128$。

状态缓存规格也直接计算了窗口宽度，位置为：

- `D:\Code\vllm-github\vllm\models\deepseek_v4\compressor.py:157-160`

```python
assert compress_ratio in [4, 128]
coff = 1 + (compress_ratio == 4)
self.sliding_window = coff * compress_ratio
```

CPU 参考实现进一步在：

- `D:\Code\vllm-github\vllm\models\deepseek_v4\cpu\cpu_compressor.py:85-90`

构造窗口：

```python
window = (1 + self.overlap) * self.compress_ratio
win_offsets = torch.arange(window, device=positions.device)
win_local = positions.unsqueeze(-1) - window + 1 + win_offsets
```

所以，对 V4 的 CSA，窗口就是：

$$
(1+1)\times4=8=2m
$$

但这也揭示了题目说法的一处不严谨：

> 不能把“所有 DeepSeek 压缩都是 $2m$ overlap”当成通用规律。即使在 V4 中，也只有 `compress_ratio=4` 的 CSA 使用 overlap；`compress_ratio=128` 的 HCA 是非重叠压缩。

### 7.4 DeepSeek-V4.1 CSA2 中是否还有 Overlap

**没有。**

这是论文和两份代码实现共同确认的结论。

#### 7.4.1 V4.1 的 ratio=2 应如何理解

```text
t0  t1  t2  t3  t4  t5  t6  t7
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   C0        C1        C2        C3
```

也就是：

$$
G_i=\{2i,2i+1\}
$$

$$
G_i\cap G_{i+1}=\varnothing
$$

每两个 token 产生一个压缩 KV：

$$
N_{\mathrm{compressed}}=\left\lfloor\frac{N}{2}\right\rfloor
$$

如果最后剩下一个未配对 token，它会暂存在 compressor state 中，等待下一个 token 到来。这个 state 是“未完成分组的 residual state”，不是 overlap。

#### 7.4.2 上游 vLLM 的代码证据

V4.1 compressor 的类注释位于：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\compressor.py:171-178`

```python
Pools ``compress_ratio`` consecutive tokens into one KV latent with a
learned softmax gate (ratio 1 has no gate and no pooling).
```

意思是把连续的 `compress_ratio` 个 token 池化成一个 KV latent，而不是把连续 $2m$ 个 token 池化成一个 latent。

该实现只接受 1 和 2 两种 ratio：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\compressor.py:192-197`

```python
if compress_ratio not in (1, 2):
    raise NotImplementedError(...)
```

gate 只在 $m>1$ 时存在：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\compressor.py:204-205`

```python
self.has_gate = compress_ratio > 1
```

压缩 KV 的 slot 写入规则位于：

- `D:\Code\vllm-github\vllm\v1\attention\backends\mla\compressor_utils.py:73-87`

```python
is_valid = ((pos + 1) % COMPRESS_RATIO == 0) & (slot != PAD_ID)
pos_after_compress = pos // COMPRESS_RATIO
```

对 $m=2$：

| 原始位置 `pos` | 是否写 compressed KV | 压缩槽位 |
|---:|---:|---:|
| 0 | 否 | 0，但不写 |
| 1 | 是 | 0 |
| 2 | 否 | 1，但不写 |
| 3 | 是 | 1 |
| 4 | 否 | 2，但不写 |
| 5 | 是 | 2 |

函数注释也明确说明，每 `compress_ratio` 个 token 共享一个压缩 state，由这一组的最后一个 token 完成写入：

- `D:\Code\vllm-github\vllm\v1\attention\backends\mla\compressor_utils.py:161-177`。

#### 7.4.3 vLLM-Ascend 的代码证据

Ascend reference compressor 位于：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\ops\triton\compressor\compressor_triton.py:567-620`。

它对每个 group 直接切片：

```python
group_tokens = tokens[g * ratio : (g + 1) * ratio]
```

位置为：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\ops\triton\compressor\compressor_triton.py:612-617`。

当 `ratio=2` 时：

```text
g=0 -> tokens[0:2]
g=1 -> tokens[2:4]
g=2 -> tokens[4:6]
```

切片之间没有交集。

Ascend 的 compressed slot mapping 也只允许每组最后一个 token 写入：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:177-184`

```python
valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)
return torch.where(valid, slot_mapping // ratio, -1)
```

序列长度被拆成：

$$
N_{\mathrm{complete}}=\left\lfloor\frac N2\right\rfloor,\qquad
N_{\mathrm{residual}}=N\bmod2
$$

对应代码为：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:826-833`

```python
torch.div(seq_lens, ratio, rounding_mode="floor", ...)
torch.remainder(seq_lens, ratio, ...)
```

这些实现中不存在隐藏的 overlap。

### 7.5 Ratio=1 的特殊含义

V4.1 的 decoder 使用的 CSA2 ratio 之一是 $m=1$。此时分组为：

$$
G_i=\{i\}
$$

如果仍然计算 softmax：

$$
\operatorname{Softmax}([z_i])=[1]
$$

于是：

$$
\bar C_i=1\odot C_i=C_i
$$

因此：

- 没有序列长度压缩；
- 不需要计算 $W_Z$；
- 不需要保存 gate score；
- 没有真正的 pooling；
- 每个 token 直接生成一个 main KV entry。

上游代码因此专门优化了 ratio=1：

```python
self.has_gate = compress_ratio > 1
```

线性层输出尺寸在 ratio=1 时也只有 KV，没有 gate，位置为：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\compressor.py:216-228`

```python
wkv_wgate_sizes = (
    [self.head_dim, self.head_dim]
    if self.has_gate
    else [self.head_dim]
)
```

所以，“V4.1 使用 $m=2$ 和 $m=1$”虽然正确，但 $m=1$ 更准确的含义是：

> CSA2 支持“不压缩的 main KV”作为特殊情况，而不是把一个 token 做了一次有实际意义的池化。

### 7.6 为什么称为 Softmax Pooling

这里需要区分两个操作。

#### 7.6.1 Softmax 只负责把 Logits 变成归一化权重

设一个非重叠组中有 $m$ 个 token：

$$
h_0,h_1,\ldots,h_{m-1}
$$

先经过两个投影：

$$
c_i=h_iW_{KV}
$$

$$
z_i=h_iW_Z
$$

其中：

- $c_i\in\mathbb R^d$ 是待压缩的内容；
- $z_i\in\mathbb R^d$ 是压缩 gate 的 logits。

然后沿 token 维做 softmax：

$$
\alpha_{i,k}
=
\frac{\exp(z_{i,k})}
{\sum_{j=0}^{m-1}\exp(z_{j,k})}
$$

对每个通道 $k$，都有：

$$
\sum_{i=0}^{m-1}\alpha_{i,k}=1,\qquad \alpha_{i,k}\ge0
$$

到这里为止只完成了 softmax。输入仍然是 $m\times d$ 个权重，并没有压成一个向量。

#### 7.6.2 Pooling 是后续的加权归约

压缩结果为：

$$
\bar c_k=\sum_{i=0}^{m-1}\alpha_{i,k}c_{i,k}
$$

向量形式为：

$$
\bar{\mathbf c}
=
\sum_{i=0}^{m-1}\boldsymbol{\alpha}_i\odot\mathbf c_i
$$

这一步将：

$$
[m,d]\longrightarrow[d]
$$

即把 $m$ 个 token 表示归约成一个表示，这才是 pooling。

```text
gate logits z
      │
      ▼
softmax over token dimension   ← 产生归一化权重
      │
      ▼
α × content c
      │
      ▼
sum over token dimension       ← 真正完成 pooling
      │
      ▼
one compressed KV latent
```

所以，“softmax pooling”不是一种不同于普通 softmax 的新数学函数，而是整个组合操作的名称：

$$
\operatorname{SoftmaxPooling}(C,Z)
=
\sum_{\text{token}}
\operatorname{Softmax}_{\text{token}}(Z)\odot C
$$

类似地，人们也会使用这些术语：

- mean pooling：权重固定为 $1/m$；
- max pooling：每个通道取最大值；
- attention pooling：根据 query-key 分数加权求和；
- softmax-gated pooling：用学习到的 gate logits 经 softmax 后加权求和。

DeepSeek 的做法更准确地说是“逐通道的 softmax-gated pooling”。

### 7.7 Gate 是逐通道权重而非单一标量

题目中“使用一个可训练矩阵计算每个 token 的权重”容易让人联想到：

$$
z_i\in\mathbb R
$$

即一个 token 只有一个标量权重。但 DeepSeek 实际上是：

$$
z_i=h_iW_Z\in\mathbb R^d
$$

所以不同通道可以从不同 token 中提取信息。

例如 $m=2,d=3$：

$$
c_0=[10,1,4],\qquad c_1=[2,9,6]
$$

假设 softmax 后的权重为：

$$
\alpha_0=[0.9,0.2,0.6]
$$

$$
\alpha_1=[0.1,0.8,0.4]
$$

压缩结果为：

$$
\bar c=\alpha_0\odot c_0+\alpha_1\odot c_1
$$

$$
=[0.9\times10+0.1\times2,\;
0.2\times1+0.8\times9,\;
0.6\times4+0.4\times6]
$$

$$
=[9.2,7.4,4.8]
$$

可以看到：

- 第一个通道主要选择 token 0；
- 第二个通道主要选择 token 1；
- 第三个通道混合两个 token。

所以它比“给两个 token 各分配一个统一标量再求加权平均”更灵活。

Ascend reference 实现正好展示了这一过程：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\ops\triton\compressor\compressor_triton.py:613-620`

```python
kv_part = group_tokens[:, :head_dim]
score_part = group_tokens[:, head_dim:]
prob = score_part.softmax(dim=0)
pooled = (kv_part * prob).sum(dim=0)
```

这里：

- `score_part.shape == [m, head_dim]`；
- `softmax(dim=0)` 沿 token/group 维归一化；
- `.sum(dim=0)` 沿 token/group 维归约；
- 输出 `pooled.shape == [head_dim]`。

所以强调“softmax pooling”的原因是：

> Softmax 只生成权重；将权重乘到 KV 上并沿 token 维求和，才真正完成序列长度压缩。

### 7.8 CSA、HCA 与 CSA2 对比

| 项目 | DeepSeek-V4 CSA | DeepSeek-V4 HCA | DeepSeek-V4.1 CSA2 $m=2$ | DeepSeek-V4.1 CSA2 $m=1$ |
|---|---:|---:|---:|---:|
| 压缩比例 | $m=4$ | $m'=128$ | $m=2$ | $m=1$ |
| 每个输出涉及的源位置 | $2m=8$ | $m'=128$ | $m=2$ | 1 |
| 相邻分组重叠 | 有，重叠 $m=4$ | 无 | **无** | 无 |
| 输出长度 | $N/4$ | $N/128$ | $N/2$ | $N$ |
| gate | 有 | 有 | 有 | 无 |
| 绝对位置压缩偏置 APE | 有 | 有 | **删除** | 无 |
| 压缩 indexer K 路径 | 从 hidden states 独立压缩 | 不适用 CSA indexer | 从 main KV 投影 | 从 main KV 投影 |
| 相邻源窗口示例 | `[0..7]`,`[4..11]` | `[0..127]`,`[128..255]` | `[0,1]`,`[2,3]` | `[0]`,`[1]` |

### 7.9 为什么前面的 V4.1 分析没有提到 Overlap

因为前面的分析对象是 **DeepSeek-V4.1-Flash 的 CSA2 实现**。在 CSA2 中，overlap 已经被明确删除：

- 论文明确说删除 overlap；
- 上游 vLLM 使用 `g*m:(g+1)*m` 的连续非重叠分组；
- vLLM-Ascend 使用同样的非重叠分组；
- slot mapping 只在每组最后一个 token 写入；
- ratio=2 的 state 只保存尚未配对的 residual token，而不是保存重叠窗口。

因此，前面将 V4.1 的 ratio=2 解释为：

```text
[t0,t1] -> C0
[t2,t3] -> C1
[t4,t5] -> C2
```

是正确的。如果要补充得更完整，应该再加一句历史对比：

> 旧版 V4 的 CSA 确实有 $2m$ overlap，但 V4.1 的 CSA2 正是把它删除了。

### 问题 7 最终总结

1. “每个压缩 KV 由 $2m$ 个源条目产生，相邻压缩 KV 重叠 $m$ 个源位置”来自 DeepSeek-V4 的 CSA 论文公式，而不是 V4.1 CSA2 的实际分组方式。
2. 旧 CSA 的窗口宽度为 $2m$、步长为 $m$，因此相邻窗口重叠 $m$，但输出数仍约为 $N/m$。
3. 旧 CSA 的重叠是源 token 位置的重叠；同一位置在相邻输出中会经过 $a/b$ 两套不同投影，不等于直接重复同一个 KV 向量。
4. DeepSeek-V4 的实际实现只对 ratio=4 的 CSA 启用 overlap；ratio=128 的 HCA 不重叠。
5. DeepSeek-V4.1 的 CSA2 明确删除 overlap 和压缩 APE。
6. V4.1 的 ratio=2 是 `[0,1]`、`[2,3]`、`[4,5]` 这样的非重叠分组。
7. ratio=2 的 state cache 用于保存尚未闭合分组的 residual token，不代表存在 overlap。
8. V4.1 的 ratio=1 是不压缩的特殊情况，单元素 softmax 恒等于 1，所以代码直接省略 gate 和 pooling。
9. “softmax pooling”不是特殊版本的 softmax，而是“softmax 产生归一化权重＋与 KV 相乘＋沿 token 维求和”的组合操作。
10. DeepSeek 的 gate 是逐通道向量，而不是每个 token 一个标量，因此不同 KV 通道可以选择不同 token 的信息。

一句话概括：

$$
\boxed{
\text{旧 CSA：宽度 }2m\text{、步长 }m\text{、存在重叠；CSA2：宽度 }m\text{、步长 }m\text{、没有重叠}
}
$$

---

## 问题 8：SparseFlashMla 的仓库归属与 Local/Global KV 联合注意力

### 问题 8 原文

> vllm-ascend 中 DeepSeek-V4.1-Flash 的稀疏注意力算子是在哪个仓库实现的？它主要做了什么事情？这个问题是否可以换成：当 Selected Compressed Global KV 和 Layer-local SWA KV 同时存在的时候，注意力是怎样计算的？

### 问题 8 核心结论

1. 从算子的上游归属看，DeepSeek-V4.1-Flash 使用的 Ascend 稀疏注意力算子来自 CANN 的 `ops-transformer` 仓库，对应目录为：

   ```text
   D:\Code\ops-transformer\attention\sparse_flash_mla
   ```

2. 从当前 vLLM-Ascend 的实际构建和调用看，这套算子源码已经集成到 vLLM-Ascend 自己的 `csrc` 中：

   ```text
   D:\Code\vllm-ascend\vllm-ascend\csrc\attention\sparse_flash_mla
   ```

   因此，运行当前 vLLM-Ascend 时，并不是动态读取旁边的 `D:\Code\ops-transformer` 源码目录，而是编译和加载 vLLM-Ascend 仓库中已经集成的 `aclnnSparseFlashMla` 算子。

3. “当 Selected Compressed Global KV 和 Layer-local SWA KV 同时存在时，注意力怎样计算”准确抓住了 CSA2 的核心数学语义：两类 KV 共同参与一次注意力，并且在**同一个 Softmax 归一化域**中竞争权重。

4. 但这个改写不能完全替代“算子在哪里实现”的工程问题，因为实际算子还负责稀疏 Gather、Paged KV Cache 地址换算、窗口与因果 Mask、在线 Softmax、Sink 处理以及跨核归并等工作。

本文的本地源码快照为：

- `D:\Code\vllm-ascend\vllm-ascend`：`4ebb3090fc3c13c26557dcaae353882e70246649`
- `D:\Code\vllm-github`：`5f30fc7031cae49bf51073fc953d419b08f8887c`
- `D:\Code\ops-transformer`：`2f0ffba13ab5d0ae6b92d4e12b097d1fb29ee4d0`

### 8.1 稀疏注意力算子究竟在哪个仓库

需要区分“上游算子仓库”和“当前 vLLM-Ascend 实际编译的源码位置”。

| 层次 | 代码位置 | 主要职责 |
|---|---|---|
| 模型公共定义 | `D:\Code\vllm-github\vllm\models\deepseek_v41` | 描述 CSA2、压缩、Indexer 和平台抽象接口 |
| Ascend Python 接入 | `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py` | 准备 Q、两类 KV Cache、Top-K 索引和 Metadata |
| Ascend Torch/ACLNN 绑定 | `D:\Code\vllm-ascend\vllm-ascend\csrc\attention\sparse_flash_mla` | 注册 PyTorch Custom Op 并调用 ACLNN 接口 |
| AscendC Kernel | `...\sparse_flash_mla\op_kernel` | 在 AI Core 上执行 Gather、QK、Softmax、PV 和归并 |

#### 8.1.1 上游算子仓库

独立的 CANN 算子源码位于：

```text
D:\Code\ops-transformer\attention\sparse_flash_mla
```

Kernel 入口为：

```text
D:\Code\ops-transformer\attention\sparse_flash_mla\op_kernel\sparse_flash_mla.cpp:66-130
```

`ops-transformer` 的算子 API 列表中也登记了 `aclnnSparseFlashMla` 和 `aclnnSparseFlashMlaMetadata`。因此，如果问题是“这个 Ascend 算子最上游属于哪个算子仓库”，答案是 CANN `ops-transformer`。

#### 8.1.2 当前 vLLM-Ascend 已集成算子源码

当前 vLLM-Ascend 已包含完整实现：

- 算子说明：`csrc/attention/sparse_flash_mla/README.md:18-36`
- 详细设计：`csrc/attention/sparse_flash_mla/docs/design.md:10-45`
- AscendC Kernel 入口：`csrc/attention/sparse_flash_mla/op_kernel/sparse_flash_mla.cpp:66-130`
- Torch Schema 注册：`csrc/torch_binding.cpp:3077-3103`
- ACLNN 适配：`csrc/attention/sparse_flash_mla/sparse_flash_mla_torch_adpt.h:148-177`

C++ 适配层最终执行：

```cpp
EXEC_NPU_CMD(aclnnSparseFlashMla, ...)
```

位置为 `sparse_flash_mla_torch_adpt.h:173-176`。

因此，更严谨的表述是：

> 算子的上游/独立实现仓库是 CANN `ops-transformer`；当前 vLLM-Ascend 已把这套算子源码集成到自己的 `csrc` 中，并从本仓完成编译和 PyTorch 注册。

#### 8.1.3 vllm-github 中是什么

`vllm-github` 主要提供模型公共逻辑和不同平台的抽象接口。例如：

- `vllm/models/deepseek_v41/attention.py:237-245` 定义平台相关的 `forward_mqa`；
- `vllm/models/deepseek_v41/nvidia/flashmla.py:230-245` 是 NVIDIA FlashMLA 路径。

CUDA 实现同样把两类缓存一起传给底层算子：

```python
k_cache=swa_cache,
indices=swa_indices,

extra_k_cache=kv_cache if not swa_only else None,
extra_indices_in_kvcache=topk_indices,
```

这说明“本层 SWA KV 加选中的 Main/Global KV 联合计算”是 DeepSeek-V4.1 CSA2 的模型语义，不是 Ascend 独有设计；Ascend 与 CUDA 只是采用不同的 Kernel 实现。

### 8.2 vLLM-Ascend 中的完整调用链

当前代码的主要执行过程可以概括为：

```text
当前层 hidden_states
        │
        ├── 生成当前层 Query
        │
        ├── 生成并写入当前层 SWA KV
        │        └── layer-local、未压缩、窗口大小 128
        │
        ├── 获得 Main/Global Compressed KV
        │        ├── Full Mode：本层生成
        │        ├── Reindex Mode：复用前层 Main KV
        │        └── Reuse Mode：复用前层 Main KV
        │
        ├── Indexer
        │        ├── Full/Reindex：计算新的 Top-K
        │        └── Reuse：直接复用已有 Top-K
        │
        └── npu_sparse_flash_mla
                 ├── ori_kv = Layer-local SWA KV
                 ├── cmp_kv = Compressed Global/Main KV
                 ├── cmp_sparse_indices = Top-K 索引
                 ├── 联合 Mask
                 ├── 联合 Online Softmax
                 └── 输出 Attention Result
```

V4.1 模型入口在：

```text
vllm_ascend/models/deepseek_v41/model.py:753-756
```

这里调用 `torch.ops.vllm.dsa_v41_forward`。注册和分发代码位于：

```text
vllm_ascend/attention/dsa_v41.py:51-77
```

完整 Forward 顺序位于 `dsa_v41.py:535-565`，包括准备缓存、生成 Query、选择索引、执行注意力、执行逆 Partial RoPE 以及输出投影。

真正进入 Ascend 稀疏注意力算子的调用位于：

```text
vllm_ascend/attention/dsa_v41.py:474-525
```

核心调用是：

```python
output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
    q=q,
    ori_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0],
    cmp_kv=source_cache,
    ori_sparse_indices=swa_indices,
    cmp_sparse_indices=cmp_indices,
    ...
)
```

### 8.3 算子的输入分别代表什么

| 算子参数 | 模型语义 |
|---|---|
| `q` | 当前层的 Main Query |
| `ori_kv` | 当前层自己的、未压缩的 SWA KV |
| `ori_sparse_indices` | SWA 中需要访问的位置或窗口索引 |
| `cmp_kv` | 压缩的 Global/Main KV Cache |
| `cmp_sparse_indices` | Indexer 选出的 Global KV Top-K 索引 |
| `ori_block_table` | SWA Paged Cache 的逻辑块到物理块映射 |
| `cmp_block_table` | Global KV Paged Cache 的逻辑块到物理块映射 |
| `seqused_ori_kv` | Original/SWA KV 的有效序列长度 |
| `seqused_cmp_kv` | Compressed KV 的有效序列长度 |
| `sinks` | Attention Sink，对 Softmax 分母有影响 |
| `cmp_mask_mode`、`residual` | 压缩分组的因果可见性控制 |

需要特别注意，`cmp_kv` 通常不是已经复制好的一个连续 Top-K 小张量。物理实现更接近：

```text
完整 cmp_kv 分页缓存
        +
cmp_sparse_indices
        +
cmp_block_table
```

即“Selected Compressed Global KV”是逻辑概念。算子收到完整的分页压缩缓存和索引，再在 Kernel 内按索引 Gather 真正参与当前 Query 计算的条目。

### 8.4 算子负责和不负责什么

`npu_sparse_flash_mla` 不负责：

- 从 Hidden States 生成压缩 KV；
- 计算所有 Main KV 的 Indexer 分数；
- 从全量 Main KV 中求 Top-K 排名。

算子设计文档在 `csrc/attention/sparse_flash_mla/docs/design.md:31` 明确说明：调用方提供压缩结果以及稀疏索引。

调用算子之前的流程是：

```text
hidden states
   │
   ├── Compressor ──> Compressed Main KV
   │
   └── Indexer ──> Top-K indices
                         │
                         ▼
                Sparse Flash MLA
```

在 vLLM-Ascend 中：

- 压缩 KV 的生成和写缓存逻辑位于 `dsa_v41.py:380-438`；
- Top-K 索引的选择或复用位于 `dsa_v41.py:440-472`；
- 稀疏注意力计算位于 `dsa_v41.py:474-528`。

Sparse Flash MLA 真正负责的是：

1. 根据稀疏索引和 Block Table 定位分页 KV；
2. Gather 选中的压缩 Global KV；
3. 读取窗口内的 Layer-local SWA KV；
4. 执行 $QK^T$；
5. 应用 Scale、Causal Mask、Window Mask 和压缩 Mask；
6. 对 Local 与 Global logits 做统一的 Online Softmax；
7. 执行 $PV$；
8. 必要时执行 Flash Decode 的跨核结果归并。

### 8.5 Selected Global KV 与 Layer-local SWA KV 如何联合计算

设当前时刻的 Query 为 $q_t$。

本层 SWA 中有效的局部 KV 集合为：

$$
\mathcal L_t=
\left\{
(k_j^L,v_j^L)\mid j\in[t-w+1,t]
\right\}.
$$

Indexer 从压缩 Global KV Cache 中选出的 Top-K 集合为：

$$
\mathcal G_t=
\left\{
(k_r^G,v_r^G)\mid r\in\operatorname{TopK}(q_t)
\right\}.
$$

实际参与注意力的 KV 序列是两者的逻辑拼接：

$$
\mathcal J_t=\mathcal L_t\mathbin{\Vert}\mathcal G_t.
$$

Local 与 Global 的 logits 分别是：

$$
a_j^L=\mathrm{scale}\cdot q_t^\top k_j^L,
$$

$$
a_r^G=\mathrm{scale}\cdot q_t^\top k_r^G.
$$

两类 KV 共享同一个归一化分母：

$$
Z=
\sum_{j\in\mathcal L_t}e^{a_j^L}
+
\sum_{r\in\mathcal G_t}e^{a_r^G}
+e^{s_h},
$$

其中 $s_h$ 是该注意力头的 Sink 值。最终输出是：

$$
o_t=
\frac{
\sum_{j\in\mathcal L_t}e^{a_j^L}v_j^L
+
\sum_{r\in\mathcal G_t}e^{a_r^G}v_r^G
}{Z}.
$$

Sink 进入 Softmax 分母，但其 Value 为零，因此不会向输出分子增加内容。

算子设计文档在 `design.md:15-31` 对实际参与计算的条目序列和 Sink 给出了相同定义。

### 8.6 为什么不能分别 Softmax 再相加

下面这种计算不等价：

$$
o_{\mathrm{wrong}}=
\operatorname{Attention}(q_t,\mathcal L_t)
+
\operatorname{Attention}(q_t,\mathcal G_t).
$$

因为它会让 Local 和 Global 分别归一化，两边各自得到总权重 1，破坏 Local 与 Global 条目之间的直接竞争。

正确计算是：

$$
o_{\mathrm{correct}}=
\operatorname{Attention}
\left(q_t,\mathcal L_t\Vert\mathcal G_t\right).
$$

即：

```text
Local logits  ─┐
               ├── 共享同一个 Online Softmax ──> 加权求和
Global logits ─┘
```

实现中不必真的构造一个完整的拼接 logits 张量。Local 与 Global KV 可以分轮、分块读取，但每一轮都必须更新同一组 Softmax 状态。`design.md:287` 明确指出：`ori` 和 `cmp` 可以分多轮读取，但必须共享同一行 Softmax 状态；分别 Softmax 后直接相加不等价。

### 8.7 联合注意力数值例子

假设当前 Query：

- 在 SWA 中看到两个局部条目；
- 从 Global Compressed KV 中选出两个条目；
- 四个 logits 分别为：

$$
a^L=[2,1],\qquad a^G=[3,0].
$$

暂时忽略 Sink，则统一归一化分母为：

$$
Z=e^2+e^1+e^3+e^0
  \approx 7.389+2.718+20.086+1
  =31.193.
$$

| 来源 | Logit | 联合 Softmax 权重 |
|---|---:|---:|
| SWA token 1 | 2 | 0.237 |
| SWA token 2 | 1 | 0.087 |
| Global token 1 | 3 | 0.644 |
| Global token 2 | 0 | 0.032 |

所以：

$$
o_t=
0.237v_1^L
+0.087v_2^L
+0.644v_1^G
+0.032v_2^G.
$$

这说明：

- 远处的 Global KV 如果和 Query 最相关，可以得到最大的权重；
- 最近的 SWA 条目仍然参与竞争；
- Local 与 Global 并不是预先各占 50%；
- 两类记忆的相对权重由同一个 Softmax 自适应决定。

如果还有 $s_h=0$ 的 Sink，分母再增加 $e^0=1$。所有实际 KV 权重之和将小于 1，剩余概率质量落在 Value 为零的 Sink 上。

### 8.8 算子内部的在线 Softmax 流水

算子内部的概念流程是：

```text
               cmp_sparse_indices
                        │
                        ▼
cmp_kv 分页缓存 ──> 稀疏 Gather ──┐
                                  │
                                  ├── QKᵀ
ori_kv / SWA KV ──> 窗口读取 ─────┘
                                      │
                                      ▼
                           Scale + Causal/Window Mask
                                      │
                                      ▼
                               Online Softmax
                                      │
                                      ▼
                                   P × V
                                      │
                                      ▼
                          跨块/跨核修正、累积与归一化
```

在线 Softmax 维护三个状态：

- 当前最大值 $m$；
- 指数和 $l$；
- 未归一化输出累积量 $a$。

读入新的一块 logit $x_j$ 时：

$$
m'=\max(m,\max_jx_j),
$$

$$
\alpha=e^{m-m'},
$$

$$
p_j=e^{x_j-m'},
$$

$$
l'=\alpha l+\sum_jp_j,
$$

$$
a'=\alpha a+\sum_jp_jv_j.
$$

所有 Local/Global 块处理完以后：

$$
o=\frac{a}{l}.
$$

因此，即使执行顺序是先处理 SWA，再分多批处理 Global KV，数学结果仍然等价于把全部有效 Local/Global logits 一次性放入同一个 Softmax。对应的流水与递推说明位于 `design.md:255-300`。

### 8.9 Full、Reindex、Reuse 是否改变注意力公式

不改变。三种模式只改变 Main KV、Indexer K 和 Top-K 索引从哪里获得：

| 模式 | Main/Global KV | Indexer K | Top-K indices |
|---|---|---|---|
| Full | 当前层生成 | 当前层生成 | 当前层重新计算 |
| Reindex | 复用之前的 Full 层 | 随 Main KV 复用 | 当前层重新打分选择 |
| Reuse | 复用之前的 Full 层 | 不再使用 | 直接复用最近一次 Top-K |

三种模式在当前层都会生成自己的 Main Query 和 Layer-local SWA KV，最后都归结为：

$$
\operatorname{Attention}
\left(
Q_{\mathrm{current}},
KV_{\mathrm{SWA,current}}
\Vert
KV_{\mathrm{Global,selected}}
\right).
$$

论文 Figure 4 对此说明为：三种模式的区别在于如何得到 Main KV、Indexer K 和 Top-K indices；三种模式都会在当前层计算 Main Q 与 SWA KV。

论文的模型配置为：

- SWA window：128；
- Sparse Attention Top-K：512；
- Encoder CSA2 compression ratio：$m=2$；
- Decoder CSA2 compression ratio：$m=1$。

因此，在典型单步 Decode 的 Core Attention 中，一个 Query 大约面对：

$$
128+512=640
$$

个候选 KV 条目，而不是直接对 100K 或 1M 全上下文做稠密 Attention。Indexer 的搜索成本是另一部分，再通过跨层索引复用和 Hierarchical Sparse Indexer 降低。

### 8.10 这个问题是否可以改写

可以，但需要限定改写所覆盖的范围。

如果问题关注的是**算法语义**，则可以准确地改写为：

> 给定当前层 Query、Selected Compressed Global KV 和 Layer-local SWA KV，怎样在同一个 Softmax 中完成联合注意力？

这个改写直接抓住了 Sparse Flash MLA 的核心数学工作。

如果问题关注的是**完整工程实现**，则还需要继续回答：

- Global KV 在哪里压缩并写入缓存；
- Top-K 由哪个 Indexer 产生或复用；
- 逻辑索引怎样通过 Block Table 映射到物理 Paged Cache；
- Causal、SWA window 和压缩 residual 怎样生成 Mask；
- Metadata 算子如何为 AI Core 划分任务；
- Kernel 怎样完成 Gather、QK、Online Softmax、PV 和跨核归并。

所以，第二种问法是第一种问题的“核心算法版”，但不是完整的“仓库位置与工程实现版”。

### 8.11 当前 Ascend 实现与论文配置的差异

论文还描述了进一步的 Main KV 低比特量化设计。但是当前检出的 vLLM-Ascend V4.1 路径在：

```text
vllm_ascend/models/deepseek_v41/model.py:704-716
```

把 long/compressed cache 按 `torch.bfloat16` 分配，并调用 BF16 SparseFlashMla 路径。

因此需要区分：

- **论文完整目标架构**：包含更加激进的 KV 压缩和低比特存储；
- **当前 vLLM-Ascend 源码快照**：已经实现 CSA2 拓扑、索引复用与 Sparse Flash MLA，但当前这条专用缓存路径仍为 BF16。

该差异不会改变“Layer-local SWA KV 与 Selected Global KV 共享一个 Softmax”的算法语义，但会影响实际 KV Cache 占用、内存带宽和性能收益。

### 问题 8 最终总结

1. DeepSeek-V4.1-Flash 的 Ascend 稀疏注意力上游来自 CANN `ops-transformer/attention/sparse_flash_mla`。
2. 当前 vLLM-Ascend 已将完整 SparseFlashMla 源码集成到自己的 `csrc/attention/sparse_flash_mla`，实际构建不依赖旁边的 `D:\Code\ops-transformer` 源码目录。
3. Python 调用入口是 `vllm_ascend/attention/dsa_v41.py:502` 的 `torch.ops._C_ascend.npu_sparse_flash_mla`。
4. `ori_kv` 表示当前层 Layer-local SWA KV；`cmp_kv` 配合 `cmp_sparse_indices` 表示 Selected Compressed Global/Main KV。
5. 压缩 KV 的生成和 Top-K 选择发生在调用稀疏注意力算子之前；算子本身不负责压缩，也不负责计算 Top-K 排名。
6. Local 与 Global KV 必须进入同一个 Online Softmax，而不能分别 Softmax 后直接相加。
7. 算子可以分轮读取 `ori_kv` 和 `cmp_kv`，但所有轮次必须共享同一组最大值、指数和与输出累积状态。
8. Full、Reindex、Reuse 只改变 Main KV 和 Top-K 的来源，不改变最终的联合注意力公式。
9. 因此，该问题在算法层面可以改写为：“Selected Compressed Global KV 和 Layer-local SWA KV 同时存在时，怎样完成统一归一化的稀疏注意力？”

一句话概括：

> SparseFlashMla 根据 Top-K 索引从压缩 Global KV Cache 中 Gather 选中条目，再与当前层自己的 SWA KV 共享同一个在线 Softmax，完成一次统一的稀疏 MLA Attention。

---

## 问题 9：Indexer Top-K、Local/Global 顺序与在线 Softmax

### 问题 9 原文

> Indexer 从压缩 Global KV Cache 中选出的 Top-K 集合的过程是怎样的？实际参与注意力计算的 KV 序列是两者的拼接：$\mathcal J_t=\mathcal L_t\mathbin{\Vert}\mathcal G_t$，为什么不是 $\Vert$ 符号前后的顺序相反？$a_j^L=\text{scale}\cdot q^\top k_j^L$，请详细解释这个式子各项的含义。当前 Query 在 SWA 中看到两个局部 token，和从 Global Compressed KV 中选出两个 token，为什么会有这两种状况同时存在？为什么在线 Softmax 维护三个状态：当前最大值 $m$、指数和 $l$、未归一化输出累积量 $a$？

### 问题 9 核心结论

1. Indexer 检索和最终 Main Attention 是两次不同的打分。Indexer 使用轻量的 Indexer Q/K 生成位置索引；SparseFlashMla 根据这些索引读取完整 Main KV，再做最终注意力。
2. $\mathcal L_t\Vert\mathcal G_t$ 与 $\mathcal G_t\Vert\mathcal L_t$ 在数学上等价，只要 K、V、Mask 和位置元数据同步重排；前者只是叙述和接口习惯。
3. $a_j^L$ 是 Local 条目 $j$ 的预 Softmax 标量 Logit，不是概率；Value 在 Softmax 之后才参与加权求和。
4. SWA 是保证可见的精确局部记忆，Global KV 是通过 Indexer 检索的长距离记忆，CSA2 有意让二者同时存在并共同归一化。
5. 在线 Softmax 的 $m,l,A$ 分别承担数值稳定基准、归一化分母和加权 Value 分子，三者缺一不可。为避免与 Logit $a_j$ 混淆，本文把输出累积量写成大写 $A$。

### 9.1 Indexer 与 Main Attention 是两次不同的打分

完整链路为：

```text
当前 token 的 hidden state
          │
          ├── Indexer Q
          │       ├── 与可见的 Indexer K 打分
          │       └── Top-K → 位置索引 I_t
          │
          └── Main Query q_t
                  ├── 读取本层 SWA KV
                  ├── 根据 I_t 读取 Main/Global KV
                  └── 联合 Softmax → Attention 输出
```

Indexer 是检索器，只回答“长上下文中哪些压缩位置值得读取”；SparseFlashMla 才是最终注意力计算器。

### 9.2 Indexer 如何选出 Top-K Global KV

#### 9.2.1 为每个 Main KV 条目生成轻量 Indexer K

对压缩 Main KV 条目 $C_j$，生成：

$$
k_j^{\mathrm{idx}}
=
\operatorname{RMSNorm}\left(W_K^{\mathrm{idx}}C_j\right).
$$

Ascend 代码在 `vllm_ascend/models/deepseek_v41/indexer.py:94-114` 中完成投影、RMSNorm、Partial RoPE、INT8 量化以及写入 Indexer K Cache。

#### 9.2.2 生成当前 token 的 Indexer Query

当前 token $t$ 生成 $g$ 个 Indexer Query head：

$$
q_{t,h}^{\mathrm{idx}}
=W_Q^{\mathrm{idx},h}qr_t,
\qquad h=1,\ldots,g.
$$

当前模型配置为 $g=32$、每个 Indexer head 维度为 128。代码位于 `indexer.py:116-143`。

#### 9.2.3 计算每个压缩位置的检索分数

先对每个 head 计算：

$$
r_{t,j,h}
=
\operatorname{ReLU}
\left[
\left(q_{t,h}^{\mathrm{idx}}\right)^\top
k_j^{\mathrm{idx}}
\right].
$$

实际 INT8 算子还会乘 Query/Key 的反量化 Scale。随后由当前 Hidden State 产生 head 权重 $w_{t,h}$，并合并各 head：

$$
s_{t,j}
=
\sum_{h=1}^{g}
w_{t,h}
\operatorname{ReLU}
\left[
\left(q_{t,h}^{\mathrm{idx}}\right)^\top
k_j^{\mathrm{idx}}
\right].
$$

$s_{t,j}$ 是检索分数，不是最终 Attention Logit。算子公式和流程见 `csrc/attention/quant_lightning_indexer_v2/README.md:16-23`。

#### 9.2.4 因果过滤和 Top-K

只允许选择当前 Query 因果可见的压缩位置。压缩率为 $m$ 时，有效压缩条目数近似为：

$$
N_{\mathrm{visible}}
=
\left\lfloor\frac{t+1}{m}\right\rfloor.
$$

Ascend 后处理直接使用：

```python
visible = (positions + 1) // COMPRESS_RATIO
valid = (selected >= 0) & (selected < visible[:, None])
```

见 `vllm_ascend/ops/triton/prepare_indexer_indices.py:35-49`。最终：

$$
\mathcal I_t
=
\operatorname{TopK}_{j<N_{\mathrm{visible}}}
\left(s_{t,j},K\right),
$$

论文配置中 $K=512$。选中的是位置集合：

$$
\mathcal I_t=\{j_1,j_2,\ldots,j_K\},
$$

再由这些位置取得真正 Main KV：

$$
\mathcal G_t=\{C_{j_1},C_{j_2},\ldots,C_{j_K}\}.
$$

Ascend 执行量化打分和 Top-K 的入口是 `indexer.py:193-228` 的 `npu_quant_lightning_indexer_v3`。

### 9.3 Hierarchical Sparse Indexer

对于百万 token 上下文，每个 Indexer 都全量扫描仍然昂贵，所以论文采用两级检索：

1. **Full Mode**：扫描全部因果可见 Main KV，产生本层 Top-512；同时每 8 个压缩位置组成一块，以块内最大分数作为块分数，选择 2048 个块，形成最多 $2048\times8=16384$ 个候选位置。
2. **Reindex Mode**：只在这个候选池里重新打分，产生当前层自己的 Top-512。
3. **Reuse Mode**：不执行新 Indexer，直接复用最近一次 Full/Reindex 层的 Top-K indices。

Reuse 对应 `vllm_ascend/attention/dsa_v41.py:440-446`：非 index source 层直接读取共享的 `topk_indices`。

### 9.4 Local 与 Global 的拼接顺序

$$
\mathcal J_t=\mathcal L_t\Vert\mathcal G_t
$$

中的 $\Vert$ 表示逻辑拼接，不是逻辑或，也不要求固定物理顺序。反过来写：

$$
\mathcal J'_t=\mathcal G_t\Vert\mathcal L_t
$$

也可以得到相同 Attention 结果，只要 K、V、Mask、有效长度与位置元数据同步重排。

令 $P$ 为置换矩阵，$K'=PK,V'=PV$，则：

$$
\operatorname{softmax}(QK'^\top)V'
=
\operatorname{softmax}(QK^\top P^\top)PV
=
\operatorname{softmax}(QK^\top)V.
$$

之前把 Local 写在前面只是因为：先介绍始终保证存在的本层局部窗口，再介绍检索得到的 Global 记忆；Ascend 接口也先传 `ori_kv`，再传 `cmp_kv`。实际上 Ascend 将两类缓存作为独立输入，CUDA/ROCm 的 Combined Indices 顺序也可以不同。

顺序不重要不等于可以只交换 K 而不交换 V/Mask。若配对关系被破坏，结果必然错误；若同一时间区域在 SWA 和 Main KV 中都有表示，交换顺序不影响结果，但删除其中一个会改变结果。

### 9.5 Local Attention Logit 公式详解

更完整地写：

$$
a_{t,j,h}^{L}
=
\gamma_h\left(q_{t,h}\right)^\top k_{j,h}^{L}.
$$

| 项 | 含义 |
|---|---|
| $t$ | 当前 Query token 的位置 |
| $h$ | Query head 编号 |
| $j$ | SWA 窗口中的某个历史位置 |
| 上标 $L$ | Local/SWA，不是层编号 |
| $q_{t,h}\in\mathbb R^d$ | 当前位置、第 $h$ 个主注意力 Query |
| $k_{j,h}^{L}\in\mathbb R^d$ | 本层 SWA 中位置 $j$ 的 Key |
| $q^\top k=\sum_cq_ck_c$ | Query 与 Key 的点积相关性 |
| $\gamma_h$ 或 `scale` | 通常接近 $1/\sqrt d$ 的缩放系数 |
| $a_{t,j,h}^{L}$ | 进入 Mask/Softmax 前的标量 Logit |

RoPE 已在点积前作用到对应维度，因此点积同时包含内容和相对位置信息。Value 不出现在该公式中，因为 Key 用于产生权重，Value 在 Softmax 后参与：

$$
o_t
=
\sum_jp_j^Lv_j^L
+
\sum_rp_r^Gv_r^G.
$$

### 9.6 为什么 Local SWA 与 Global KV 同时存在

CSA2 是一个双记忆系统：

```text
                    当前 Query
                        │
             ┌──────────┴──────────┐
             │                     │
     Layer-local SWA          Selected Global KV
       最近 128 个              长上下文 Top-512
       精细、保证可见             压缩、按相关性检索
             │                     │
             └──────────┬──────────┘
                        │
                  联合 Softmax
```

SWA 保证最近 token 的精确逐 token 表示，不依赖 Indexer 是否命中，适合局部语法、当前代码行和最近推理步骤。Global/Main KV 覆盖整个长上下文，适合找回早期系统要求、远处变量定义或长文档事实。

两者不是互斥集合。Global Top-K 可能选择到最近时间区域，但其表示和本层 SWA 不同：SWA 是当前层未压缩局部表示；Global/Main KV 可能来自共享 Source 层并经过时间压缩。即使时间范围重叠，也作为两个不同条目共同竞争权重。

### 9.7 在线 Softmax 为什么维护三个状态

普通注意力输出为：

$$
O
=
\frac{\sum_i e^{x_i}v_i}{\sum_i e^{x_i}}.
$$

为避免 $e^{1000}$ 一类上溢，稳定 Softmax 减去最大值。分块处理 Local/Global KV 时，后面的块可能出现更大 Logit，因此维护：

1. 当前最大值：

   $$
   m=\max_{\text{已处理}}x_i.
   $$

2. 以当前最大值为基准的指数和：

   $$
   l=\sum_{\text{已处理}}e^{x_i-m}.
   $$

3. 未归一化 Value 分子：

   $$
   A=\sum_{\text{已处理}}e^{x_i-m}v_i.
   $$

最终：

$$
O=\frac{A}{l},
\qquad
\operatorname{LSE}=m+\log l.
$$

没有 $m$ 会溢出/下溢；没有 $l$ 无法归一化；没有 $A$ 则无法得到 Value 加权输出。

### 9.8 在线 Softmax 更新和数值例子

旧状态为 $(m,l,A)$，新块 Logits 为 $x_j$：

$$
m'=\max(m,\max_jx_j),
$$

$$
\alpha=e^{m-m'},\qquad p_j=e^{x_j-m'},
$$

$$
l'=\alpha l+\sum_jp_j,
$$

$$
A'=\alpha A+\sum_jp_jv_j.
$$

例如 Local Logits 为 $[2,1]$，Global Logits 为 $[3,0]$。处理 Local 后：

$$
m_L=2,qquad l_L=1+e^{-1}\approx1.3679.
$$

处理 Global 时新最大值为 3，旧状态按 $\alpha=e^{-1}$ 重缩放：

$$
l'=e^{-1}(1+e^{-1})+1+e^{-3}\approx1.5530.
$$

它等于一次性计算：

$$
e^{2-3}+e^{1-3}+e^{3-3}+e^{0-3}.
$$

所以分块计算与一次性联合 Softmax 数学等价；先处理 Local 或先处理 Global，在精确算术下也得到相同最终状态。

### 问题 9 最终总结

1. Indexer 用轻量 Q/K 检索位置，Main Attention 使用完整 Main KV 计算输出。
2. Top-K 是压缩 Main KV 的位置集合，随后根据位置读取真正缓存内容。
3. Full 扫描全局并建立块候选池，Reindex 在候选池中重选，Reuse 直接复用索引。
4. $\mathcal L_t\Vert\mathcal G_t$ 的顺序只是表达/布局选择；同步置换 K、V 和元数据不改变注意力结果。
5. Local Logit 是缩放后的 Query-Key 点积，Value 只在 Softmax 后参与聚合。
6. SWA 提供保证可见的精确局部记忆，Global KV 提供按相关性检索的长距离记忆。
7. 在线 Softmax 的 $m,l,A$ 分别负责数值稳定、分母和分子，使分块处理与完整 Softmax 等价。

---

## 问题 10：由压缩 Main KV 生成 Indexer K

### 问题 10 原文

> “先生成压缩 Main KV 对应的 Indexer K”是什么意思？请解释公式每一项的含义：
>
> $$
> k_j^{\mathrm{idx}}
> =
> \operatorname{RMSNorm}
> \left(W_K^{\mathrm{idx}}C_j\right).
> $$

### 问题 10 核心结论

对压缩 Main KV Cache 中每一个完整内容条目 $C_j$，额外生成一个更小、专门用于检索的向量 $k_j^{\mathrm{idx}}$。Indexer 先使用这个小向量判断“是否值得读取 $C_j$”；选中后，真正 Attention 读取的是 $C_j$，不是 Indexer K。

### 10.1 压缩条目 Cj 是什么

$j$ 是压缩 Main KV Cache 的逻辑槽位，而不一定是原始 token 位置。ratio=2 时：

```text
H0、H1 ──压缩──> C0
H2、H3 ──压缩──> C1
H4、H5 ──压缩──> C2
```

Ascend 中 ratio=2 的生成代码位于 `vllm_ascend/attention/dsa_v41.py:402-408`：

```python
kv = compressor.wkv(hidden_states_fp32)
score = compressor.wgate(hidden_states_fp32)
latent = compressor.pool_projected(kv, score, state_metadata)
```

这里 `latent[j]` 就是 $C_j$。Sparse MLA 中它是共享 Main KV latent，而不是传统 Attention 中两个完全独立的 $(K_j,V_j)$ 张量。当前主 latent 维度为 512：

$$
C_j\in\mathbb R^{512}.
$$

### 10.2 Indexer 投影矩阵是什么

$W_K^{\mathrm{idx}}$ 是可训练线性投影，将 512 维 Main KV latent 转换为 128 维轻量 Indexer Key：

$$
W_K^{\mathrm{idx}}\in\mathbb R^{128\times512},
$$

$$
\widetilde{k}_j^{\mathrm{idx}}
=W_K^{\mathrm{idx}}C_j
\in\mathbb R^{128}.
$$

代码位于 `vllm_ascend/models/deepseek_v41/indexer.py:65-71`：

```python
self.wk = nn.Linear(
    config.head_dim,
    self.width,
    bias=False,
    dtype=torch.bfloat16,
)
```

`config.head_dim=512`，`self.width=index_head_dim=128`，不使用 Bias。低维 Indexer K 用于扫描长上下文，可显著降低检索计算和缓存带宽；完整 512 维 Main KV 只有在位置被选中后才读取。

### 10.3 RMSNorm 的作用

令：

$$
x=W_K^{\mathrm{idx}}C_j\in\mathbb R^{d_{\mathrm{idx}}}.
$$

RMSNorm 计算：

$$
\operatorname{RMS}(x)
=
\sqrt{
\frac{1}{d_{\mathrm{idx}}}
\sum_{c=1}^{d_{\mathrm{idx}}}x_c^2
+\epsilon
},
$$

$$
\operatorname{RMSNorm}(x)
=
\gamma\odot\frac{x}{\operatorname{RMS}(x)}.
$$

$\epsilon$ 防止除零，$\gamma$ 是可训练逐通道缩放，$\odot$ 表示逐元素乘法。RMSNorm 不做均值中心化，它主要稳定不同条目的向量幅值，避免某个条目仅因模长较大就得到不合理高分。

代码为：

```python
key = self.k_norm(self.wk(latent)).view(-1, 1, self.width)
```

见 `indexer.py:94-99`。形状变化为：

$$
\underbrace{C_j}_{[512]}
\xrightarrow{W_K^{\mathrm{idx}}}
\underbrace{\widetilde{k}_j^{\mathrm{idx}}}_{[128]}
\xrightarrow{\mathrm{RMSNorm}}
\underbrace{k_j^{\mathrm{idx}}}_{[128]}.
$$

### 10.4 实际实现中的 RoPE 和 INT8 量化

公式只描述核心投影和归一化。实际缓存还会执行：

$$
\widehat{k}_j^{\mathrm{idx}}
=
\operatorname{PartialRoPE}
\left(k_j^{\mathrm{idx}},p_j\right),
$$

$$
\left(k_j^{\mathrm{idx,int8}},s_j^K\right)
=
\operatorname{QuantINT8}
\left(\widehat{k}_j^{\mathrm{idx}}\right).
$$

完整链路为：

```text
C_j
 │ W_K^idx：512 → 128
 ▼
未归一化 Indexer K
 │ RMSNorm
 ▼
归一化 Indexer K
 │ Partial RoPE
 ▼
带位置信息的 Indexer K
 │ Dynamic INT8 Quantization
 ▼
INT8 Indexer K + FP16 Scale
 │
 ▼
Indexer K Paged Cache
```

Partial RoPE 和 INT8 量化位于 `indexer.py:99-114`。Indexer K Cache 配置为 1 个 KV head、128 维、INT8 数据和 FP16 Scale，见 `indexer.py:73-86`。

同一个原始 `latent` 随后走 Main KV 自己的 Partial RoPE，并以完整 Main KV 写入 `long_kv_cache`，见 `dsa_v41.py:420-438`：

```text
                         ┌─ W_K^idx → RMSNorm → RoPE → INT8
                         │                  └─ Indexer K Cache
压缩 Main KV latent C_j ─┤
                         │
                         └─ Main Partial RoPE → BF16
                                            └─ Main KV Cache
```

### 10.5 Main KV 与 Indexer K 的一一对应

“对应”表示共享同一逻辑槽位 $j$：

```text
Main KV Cache                 Indexer K Cache
slot 0：C0，512 维  <──────>  slot 0：k0_idx，128 维
slot 1：C1，512 维  <──────>  slot 1：k1_idx，128 维
slot 2：C2，512 维  <──────>  slot 2：k2_idx，128 维
```

Indexer 若选出：

$$
\mathcal I_t=\{1,7,20\},
$$

最终 Sparse Attention 读取：

$$
\{C_1,C_7,C_{20}\},
$$

而不是把 $k_j^{\mathrm{idx}}$ 当作最终 KV。Indexer K 是检索目录，Main KV 是目录指向的真实内容。

“先生成”也不表示每个 Decode 步重算所有历史 Indexer K。每当新压缩条目 $C_j$ 产生时，对应的 $k_j^{\mathrm{idx}}$ 只生成并缓存一次；后续 Query 生成自己的 Indexer Query，读取历史 Indexer K 完成检索。

### 10.6 为什么从 Main KV 投影 Indexer K

从实际 Main KV 条目直接投影可以保证：

1. 每个 Indexer K 都对应一个真实存在的 Main KV 条目；
2. Top-K 索引可以直接用于读取 Main KV Cache；
3. Main KV 跨层复用时，Indexer K 也可以随之复用；
4. 不需要维护另一套独立的时间压缩路径；
5. Indexer 学到的是“这个真实缓存条目是否值得被最终注意力读取”。

### 问题 10 最终总结

$$
\boxed{
k_j^{\mathrm{idx}}
=
\operatorname{RMSNorm}\left(W_K^{\mathrm{idx}}C_j\right)
}
$$

表示把第 $j$ 个完整 Main KV latent 从 512 维投影并归一化成 128 维检索 Key。实际实现还会加 Partial RoPE，并量化为 INT8 写入独立缓存。Indexer 用它选位置，SparseFlashMla 根据选中的位置读取完整 Main KV；它们是“检索卡片”和“真实内容”的关系。

---

## 问题 11：什么是 Main KV

### 问题 11 原文

> 什么是上面提到的 Main KV？

### 问题 11 核心结论

Main KV 是 DeepSeek-V4.1 CSA2 中用于保存和检索整个历史上下文信息的长上下文主缓存分支。Indexer 从 Main KV 中选出 Top-K 条目，随后这些条目与当前层自己的 SWA KV 一起进入最终联合注意力。

```text
当前层 Query
    │
    ├── Layer-local SWA KV
    │      最近窗口、每层独立、精确局部信息
    │
    └── Main KV
           整个长上下文、可压缩、可跨层共享
           通过 Indexer 选择 Top-K
```

### 11.1 Main、Global、Long 与 Compressed KV 的关系

在本文上下文中：

$$
\text{Main KV}
\approx
\text{Global KV}
\approx
\text{Long-context KV}.
$$

| 名称 | 强调点 |
|---|---|
| Main KV | 论文 CSA2 中与 SWA 分支相对的主 KV 分支 |
| Global KV | 能提供整个历史上下文的信息 |
| Long KV | vLLM-Ascend 代码中长上下文缓存的命名 |
| Compressed KV | Main KV 使用时间维压缩时的称呼 |
| Selected Main KV | 经过 Indexer Top-K 选择后的 Main KV 子集 |

Main KV 不一定总是压缩的：

```text
Main KV
 ├── ratio=2：两个原始位置压成一个条目
 └── ratio=1：一个位置对应一个条目，不做时间维压缩
```

论文配置中 Encoder 大部分 CSA2 层使用 $m=2$，Decoder CSA2 层使用 $m=1$。因此“Main KV”是分支名称，“Compressed KV”是其中一种配置。

Main KV 也不是 CPU 主存，不表示普通 Dense Attention 的全部 K/V，更不是一个必须叫 `main_kv` 的代码变量。Ascend 中对应名称主要是 `long_kv_cache`、`source_cache` 和算子参数 `cmp_kv`。

### 11.2 Main KV 怎样生成

#### Ratio=1

每个 Hidden State 投影出一个 Main KV 条目：

$$
C_t=W_{KV}H_t.
$$

对应：

```text
H0 → C0
H1 → C1
H2 → C2
```

Ascend 代码在 `dsa_v41.py:392-401`：

```python
if ratio == 1:
    latent = compressor(hidden_states)
```

#### Ratio=2

相邻两个位置先分别产生候选内容和 Gate，再池化成一个条目：

$$
C_j
=
\operatorname{Pool}
\left(
W_{KV}H_{2j},
W_{KV}H_{2j+1}
\right).
$$

对应：

```text
H0、H1 → C0
H2、H3 → C1
H4、H5 → C2
```

代码位于 `dsa_v41.py:402-408`。

### 11.3 Main KV 的存储表示

普通 Attention 通常分别缓存 $K_j$ 和 $V_j$；Sparse MLA 使用共享的 Main KV latent：

$$
C_j\in\mathbb R^{512}.
$$

同一缓存向量在 QK 阶段承担 Key latent 作用，在 PV 阶段承担 Value latent 作用。实际模型还包含 Partial RoPE、逆 RoPE 和输出投影。

Ascend 先给 `latent` 施加 Main KV Partial RoPE：

```python
torch.ops._C_ascend.inplace_partial_rotary_mul(...)
```

再写入：

```python
attn.long_kv_cache.kv_cache[0]
```

见 `dsa_v41.py:426-438`。因此 `long_kv_cache` 是论文 Main KV 在当前 Ascend Backend 中的主要物理缓存。

### 11.4 Main KV、Indexer K 与 SWA KV 的区别

| 对象 | 典型维度/类型 | 作用 |
|---|---|---|
| Main KV $C_j$ | 512 维，当前专用 Ascend 路径为 BF16 | 真正参与最终 Sparse Attention |
| Indexer K $k_j^{\mathrm{idx}}$ | 128 维 INT8 + FP16 Scale | 快速判断 Main KV 位置 $j$ 是否值得读取 |
| Layer-local SWA KV | 当前层局部 KV | 保证最近窗口精确、无条件可见 |

Main KV 与 Indexer K 一一对应，但用途不同：

```text
Indexer K = 检索目录
Main KV   = 目录指向的真实内容
```

SWA 与 Main KV 的区别为：

| 属性 | Main KV | Layer-local SWA KV |
|---|---|---|
| 作用范围 | 整个长上下文 | 最近窗口 |
| 访问方式 | Indexer Top-K | 固定滑动窗口 |
| 时间压缩 | ratio=1 或 2 | 通常不压缩 |
| 跨层共享 | 可以 | 一般不共享 |
| 来源 | Source 层或 CED Encoder 输出 | 当前层自己的 Hidden State |
| 算子参数 | `cmp_kv` | `ori_kv` |

Ascend 算子调用位于 `dsa_v41.py:502-508`：

```python
torch.ops._C_ascend.npu_sparse_flash_mla(
    q=q,
    ori_kv=attn.dsa_attn.swa_cache_layer.kv_cache[0],
    cmp_kv=source_cache,
    cmp_sparse_indices=cmp_indices,
    ...
)
```

SWA 负责“最近发生了什么”，Main KV 负责“遥远历史中什么和当前 Query 相关”。Main KV 可能压缩并依赖检索，SWA 则为局部连续性提供可靠路径。

### 11.5 CED 中 Main KV 的特殊来源

普通 Decoder-only 第 $l$ 层的 KV 来自该层自己的 Hidden State：

$$
KV_l=W_{KV}^{(l)}H_l.
$$

CED 上半部分 Decoder 的 Main KV 可以直接从 Causal Encoder 最终输出 $H_{20}$ 投影：

$$
C_l=H_{20}W_{KV}^{(l)}.
$$

此时：

```text
Main Query：来自当前 Decoder 层正在处理的 Query token
Main KV：可从 Causal Encoder 的 H20 投影得到
SWA KV：仍来自当前 Decoder 层自己的 Hidden State
```

这使大部分 Prompt 不必逐层执行上半 Decoder 来生成 Main KV；只需对最近窗口 Replay，以构造每层自己的 SWA KV。

### 11.6 长上下文例子

假设 Prompt 为 100K token、SWA window=128、Main KV ratio=2、Attention Top-K=512。Main KV Cache 大约有：

$$
100000/2=50000
$$

个条目 $C_0,\ldots,C_{49999}$，并有一一对应的 Indexer K。

当前 Query：

1. 用轻量 Indexer Query 对历史 Indexer K 检索；
2. 选出 512 个位置，如 $\{12,95,801,\ldots\}$；
3. 从 Main KV Cache 读取 $C_{12},C_{95},C_{801},\ldots$；
4. 同时读取当前层最近约 128 个 SWA KV；
5. 两类 KV 进入同一个 Online Softmax。

Core Attention 因而大约处理：

$$
512+128=640
$$

个候选条目，而不是直接读取全部 100K 个原始 token。Indexer 搜索成本另行通过低维 Q/K、跨层 Reuse 与 Hierarchical Candidate Pool 降低。

### 问题 11 最终总结

```text
原始 Hidden States
        │
        │ W_KV + 可选 ratio=2 Pooling
        ▼
Main KV 条目 C_j
        │
        ├──────────────> Main/Long KV Cache
        │                     │
        │                     │ Top-K 索引读取
        │                     ▼
        │                SparseFlashMla
        │
        └── W_K^idx + RMSNorm + RoPE + INT8
                              │
                              ▼
                       Indexer K Cache
                              │
                              │ 轻量检索
                              ▼
                         Top-K indices
```

一句话概括：Main KV 是 CSA2 保存长上下文真实内容的主 KV latent 缓存；Indexer K 是从它派生的轻量检索标签。Indexer 用标签选择位置，最终 SparseFlashMla 读取的仍然是 Main KV，并与当前层 SWA KV 共同完成注意力。

---

## 问题 12：Full、Reindex、Reuse 与 Main Q

### 问题 12 原文

> KV 层被划分为三种模式，下面对这三种模式的理解对吗？
>
> - Full Mode：完整生成 Main KV、Indexer K，并重新搜索 Top-K 历史 Token。
> - Reindex Mode：复用此前生成的 Main KV、Indexer K，但重新计算新的 Top-K 索引。
> - Reuse Mode：进一步连 Top-K 索引也一起复用，仅执行真正的 Attention 计算。
>
> “Full、Reindex、Reuse 三种模式虽然对 Main KV、Indexer K 和 Top-K 的取得方式不同，但三种模式都会计算当前层自己的 Main Q 和 SWA KV”，这里的 Main Q 指的是什么？

### 问题 12 核心结论

整体理解正确，但需要三个限定：

1. Top-K 选择的是因果可见的 Main KV 条目/压缩位置，不一定是原始 token；ratio=2 时一个条目概括两个原始位置。
2. Reindex 使用新的 Indexer Q 重新打分，但在 Decoder Hierarchical Indexer 中通常只搜索共享候选池，而不是完整上下文。
3. Reuse 只是在 Main-KV/Indexer 准备路径中跳过了生成和检索；它仍计算当前层 Main Q、SWA KV、Sparse Attention、输出投影以及该 Transformer Block 的残差、mHC、MoE/MLP。

三种模式的精确对照如下：

| 模式 | 新 Main KV | 新 Indexer K | 新 Indexer Q/打分 | 新 Top-K | 当前层 Main Q、SWA KV 与 Core Attention |
|---|---:|---:|---:|---:|---:|
| Full | 是 | 是 | 是 | 是 | 是 |
| Reindex | 否，复用 | 否，复用 | 是 | 是 | 是 |
| Reuse | 否，复用 | 否，复用 | 否 | 否，复用 | 是 |

可以概括为：

```text
Full：
新的 Main KV + 新的 Indexer K + 新的 Indexer Q + 新 Top-K

Reindex：
复用 Main KV/Indexer K + 新的 Indexer Q + 新 Top-K

Reuse：
复用 Main KV/Indexer K + 复用 Top-K
```

本文核对的本地源码快照为：

- `D:\Code\vllm-ascend\vllm-ascend`：`4ebb3090fc3c13c26557dcaae353882e70246649`
- `D:\Code\vllm-github`：`5f30fc7031cae49bf51073fc953d419b08f8887c`

### 12.1 Top-K 选择的是 Main KV 条目

“重新搜索 Top-K 历史 Token”更准确应写为：

> 重新搜索 Top-K 个因果可见的 Main KV 条目或压缩位置。

Main KV 支持不同压缩率：

- Encoder CSA2 使用 $m=2$ 时，一个 Main KV 条目代表两个原始 token；
- Decoder CSA2 使用 $m=1$ 时，一个 Main KV 条目约对应一个原始 token。

例如 $m=2$：

```text
原始 token 0、1 → Main KV entry C0
原始 token 2、3 → Main KV entry C1
原始 token 4、5 → Main KV entry C2
```

Indexer 若选中 $\{C_0,C_2\}$，选中的是两个压缩条目，分别概括原始时间范围 $[0,1]$ 和 $[4,5]$。论文 Top-K=512 指最多选择 512 个 Main KV entries，而不一定是 512 个原始 token。

### 12.2 Full Mode

Full 层拥有并发布一份新的 Main KV Cache，从 Main KV 投影对应的 Indexer K，计算当前层 Indexer Q，并运行 Indexer 产生新的 Top-K Main-KV indices：

```text
当前层输入
   │
   ├── Main Q ─────────────────────────────────────┐
   ├── 当前层 SWA KV ──────────────────────────────┤
   │                                               │
   ├── 新 Main KV ───────────────┐                 │
   ├── 从 Main KV 生成 Indexer K │                 │
   └── 新 Indexer Q → 打分 → Top-K indices ───────┤
                                                   ▼
                                            Core Attention
```

Full 层生成：

- 当前层 Main Q；
- 当前层 SWA KV；
- 新 Main KV；
- 对应的新 Indexer K；
- 当前层 Indexer Q；
- 新 Top-K indices；
- 最终 Attention 输出。

对于 CED Decoder，论文还强调 Full 层的 Global/Main KV 可以从 Causal Encoder 最后一层输出 $H_{20}$ 投影：

$$
C_l=H_{20}W_{KV}^{(l)}.
$$

所以“Full 层生成自己的 Main KV”是指它拥有自己的 Main-KV 投影参数和缓存结果，不意味着 Prompt 必须完整经过该 Decoder 层后才能产生 Main KV。

Decoder 第一个 Full 层还充当 Candidate Source：扫描全部因果可见 Main KV，产生自己的 Top-512，同时构建块级候选池，供后续 Reindex 层使用。

### 12.3 Reindex Mode

Reindex 的理解是正确的：它复用最近 Full 层的 Main KV 及对应 Indexer K，但计算当前层自己的 Indexer Q，并产生新的 Top-K：

```text
前序 Full 层 Main KV ─────────────────────┐
前序 Full 层 Indexer K ──┐               │
                         │               │
当前 Reindex 层 Indexer Q│               │
                         ▼               │
                    重新打分              │
                         │               │
                    新 Top-K indices ────┤
当前层 Main Q ───────────────────────────┤
当前层 SWA KV ───────────────────────────┤
                                         ▼
                                  Core Attention
```

不同深度的层对“哪些历史信息重要”可能有不同判断，所以即使记忆库不变，也需要新的检索 Query。例如前一层可能选中函数定义和变量说明，更深层可能改为选择用户要求和中间推理结果。

Hierarchical Sparse Indexer 下，Decoder 第一个 Full 层将位置按每 8 个划分成块，选择最多 2048 个高分块，形成最多：

$$
2048\times8=16384
$$

个候选位置。后续 Reindex 通常只在该候选池内重新打分并选择 Top-512，而不是扫描整个百万 token 上下文。

因此最精确的表述是：

> Reindex 复用 Main KV 和 Indexer K，生成当前层 Indexer Q，并在共享候选池中重新打分、选择新的 Top-K。

### 12.4 Reuse Mode

Reuse 复用：

- 最近可用的 Main KV；
- 与该 Main KV 对应的 Indexer K 关系；
- 最近一次 Full/Reindex 针对这份 Main KV 产生的 Top-K indices。

它不再执行：

- Main KV 压缩/投影；
- Indexer K 生成；
- Indexer Q 生成；
- Indexer 打分；
- Top-K 排序。

Ascend 代码在 `vllm_ascend/attention/dsa_v41.py:440-446` 直接返回共享索引：

```python
if not self.role.is_index_source:
    return shared.topk_indices[: hidden_states.shape[0]]
```

但 Reuse 仍执行：

```text
当前层 Hidden State
    ├── 当前层 Main Q 投影
    ├── 当前层 SWA KV 投影并写缓存
    ├── 用复用 Top-K 读取 Selected Main KV
    ├── SparseFlashMla
    ├── Attention 输出投影
    ├── 残差/mHC
    └── MoE/MLP
```

因此，“仅执行真正 Attention”中的“仅”过强。更准确地说：

> Reuse 跳过 Main-KV 生成和整个 Indexer 流程，但仍执行当前层 Query/SWA/Attention/输出投影以及完整 Transformer Block 的其他计算。

Reuse 也不能复用任意 Top-K；它必须复用针对当前这份 Main KV 计算出的最近 Top-K，否则相同索引编号可能指向另一份缓存里的不同内容。

### 12.5 代码如何表示三种模式

当前 vLLM-Ascend 没有直接保存 `mode=FULL/REINDEX/REUSE` 枚举，而是用两个独立职责组合：

- `is_kv_source`：当前层是否产生新的 Main KV；
- `is_index_source`：当前层是否运行 Indexer 并发布新的 Top-K。

因此：

| 模式 | `is_kv_source` | `is_index_source` |
|---|---:|---:|
| Full | True | True |
| Reindex | False | True |
| Reuse | False | False |

层角色定义见 `vllm_ascend/models/deepseek_v41/model.py:439-455`。每一层通过：

```python
kv_source = _latest_source(layer_idx, kv_sources)
index_source = _latest_source(layer_idx, index_sources)
```

找到不高于自己的最近 KV Source 和 Index Source，见 `model.py:494-536`。

只有 KV Source 层创建：

- `long_kv_cache`；
- `compressor`。

对应 `model.py:704-721`。只有 Index Source 层创建 `DeepseekV41Indexer`，对应 `model.py:723-733`。所以：

```text
Full：compressor + long_kv_cache + indexer
Reindex：indexer，但没有自己的 compressor/long_kv_cache
Reuse：三者都不创建，消费 Source 层和共享索引
```

消费者通过 `long_kv_source_prefix` 和 `index_k_source_prefix` 找到来源，见 `model.py:735-739`。真正执行 Attention 时，再从 Source 层取得 Main KV，见 `dsa_v41.py:474-477`。

### 12.6 40 层的实际模式布局

本地测试配置在 `tests/deepseek_v41_utils.py:276-279` 给出：

```python
compress_ratios=[0, 0] + [2] * 18 + [1] * 20
kv_source_layer_ids=[2, 8, 14, 20]
index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer_id=20
```

对应布局：

```text
Causal Encoder
────────────────────────────────────────
Layer  0-1   Pure SWA

Layer  2     Full
Layer  3-7   Reuse

Layer  8     Full
Layer  9-13  Reuse

Layer 14     Full
Layer 15-19  Reuse


Decoder
────────────────────────────────────────
Layer 20     Full + Candidate Source
Layer 21-23  Reuse

Layer 24     Reindex
Layer 25-27  Reuse

Layer 28     Reindex
Layer 29-31  Reuse

Layer 32     Reindex
Layer 33-35  Reuse

Layer 36     Reindex
Layer 37-39  Reuse
```

这与论文配置一致：Encoder 前两层为纯 SWA，余下 18 层分成三个六层组，每组第一层 Full、其余五层 Reuse；Decoder 分成五个四层组，第一组第一层 Full，后四组第一层 Reindex，每组其余三层 Reuse。

### 12.7 Main Q 是什么

Main Q 是：

> 当前层用于真正 Sparse MLA/Core Attention 的 Query 向量。

它会同时与：

1. 当前层 Layer-local SWA KV；
2. Indexer 选出的 Main/Global KV；

计算最终注意力：

```text
                      Main Q
                         │
             ┌───────────┴───────────┐
             │                       │
        Local SWA KV          Selected Main KV
             │                       │
             └───────────┬───────────┘
                         │
                  Joint Softmax
                         │
                  Attention Output
```

“Main”主要用于把它和 `Indexer Q` 区分开，并不表示它只查询 Main KV。当前实现没有另一份独立的 `SWA Q`；同一份 Main Q 同时查询 Local SWA KV 与 Selected Main KV。

### 12.8 Main Q 的生成过程

设第 $l$ 层输入为：

$$
H_l\in\mathbb R^{T\times d_{\mathrm{model}}}.
$$

Main Q 使用低秩 Query 投影：

$$
Q_{a,l}=H_lW_{Q_a}^{(l)},
$$

$$
Q_{r,l}=\operatorname{RMSNorm}(Q_{a,l}),
$$

$$
\widetilde Q_l^{\mathrm{main}}
=Q_{r,l}W_{Q_b}^{(l)},
$$

$$
Q_l^{\mathrm{main}}
=\operatorname{PartialRoPE}
\left(\operatorname{reshape}(\widetilde Q_l^{\mathrm{main}})\right).
$$

流程为：

```text
当前层 Hidden State H_l
          │ Wq_a
          ▼
     低秩 Query 表示
          │ RMSNorm
          ▼
          qr
          │ Wq_b
          ▼
 [T, Query Heads, 512]
          │ Partial RoPE
          ▼
       Main Q
```

Ascend 代码对应为：

- `q_a = wq_a.matmul(...)`：`dsa_v41.py:327-338`；
- `qr = attn.q_norm(q_a)`：`dsa_v41.py:340-349`；
- `q = wq_b.matmul(...).unflatten(...)`：`dsa_v41.py:351-370`；
- Main Q Partial RoPE：`dsa_v41.py:371-377`。

最终形状为：

$$
Q_l^{\mathrm{main}}
\in
\mathbb R^{T\times H_q^{\mathrm{local}}\times512}.
$$

论文配置中全模型 Query heads 为 64；Tensor Parallel 下每个 Rank 保存本地 head 分片。

### 12.9 Main Q 与 Indexer Q 的区别

| 属性 | Main Q | Indexer Q |
|---|---|---|
| 用途 | 最终 Sparse Attention | 选择 Main KV 的 Top-K 位置 |
| Head 数 | 论文配置 64 | 32 |
| Head Dim | 512 | 128 |
| 查询 SWA KV | 是 | 否 |
| 查询 Selected Main KV | 是，执行最终 Attention | 只对 Indexer K 打分，不执行最终 Attention |
| Full | 计算 | 计算 |
| Reindex | 计算 | 计算 |
| Reuse | 计算 | 不计算 |
| 跨层复用 | 不复用 | Query 本身不复用，但其 Top-K 结果可以复用 |

二者都可从中间低秩表示 `qr` 出发，但使用不同投影矩阵：

```text
                   qr
          ┌────────┴─────────┐
          │                  │
     Main Wq_b          Indexer Wq_b
          │                  │
          ▼                  ▼
       Main Q            Indexer Q
  64 heads × 512      32 heads × 128
          │                  │
          │                  └── 对 Indexer K 打分 → Top-K
          │
          └── 对 SWA KV + Selected Main KV 做最终 Attention
```

Indexer Query 构造位于 `vllm_ascend/models/deepseek_v41/indexer.py:116-157`。

### 12.10 为什么三种模式都计算 Main Q 和 SWA KV

第 $l$ 层 Main Q 来自当前层输入和层专属参数：

$$
Q_l^{\mathrm{main}}=f_l(H_l).
$$

下一层则为：

$$
Q_{l+1}^{\mathrm{main}}=f_{l+1}(H_{l+1}).
$$

通常 $H_{l+1}\neq H_l$，且不同层的 $W_{Q_a}^{(l)},W_{Q_b}^{(l)}$ 不同，因此不能跨层复用 Main Q。即使复用了 Main KV 和 Top-K，本层仍要重新计算权重：

$$
a_{l,t,j}
=
\frac{
\left(Q_{l,t}^{\mathrm{main}}\right)^\top K_j
}{\sqrt d}.
$$

即 Reuse 复用的是“去哪里找”，不是“当前层如何读取”。Top-K 指定读取哪些 Main KV，Main Q 决定这些 Main KV 和当前层 SWA KV 在本层各获多大权重。

SWA KV 同样是 Layer-local：

$$
KV_{l,t}^{\mathrm{SWA}}
=W_{KV,\mathrm{SWA}}^{(l)}H_{l,t}.
$$

不同层 Hidden State 和投影参数不同，所以每层必须生成自己的 SWA KV。Ascend 在 `dsa_v41.py:312-378` 并行准备 Main Q 与当前层 SWA KV，随后调用：

```python
torch.ops._C_ascend.npu_sparse_flash_mla(
    q,
    ori_kv=current_layer_swa_cache,
    cmp_kv=reused_or_new_main_kv,
    cmp_sparse_indices=new_or_reused_topk,
    ...
)
```

见 `dsa_v41.py:502-525`。这里：

- `q`：当前层 Main Q；
- `ori_kv`：当前层 SWA KV；
- `cmp_kv`：新生成或复用的 Main KV；
- `cmp_sparse_indices`：新生成或复用的 Top-K。

### 12.11 四层组示例

以 Decoder 的一个四层组为例：

```text
Layer 24：Reindex
Layer 25：Reuse
Layer 26：Reuse
Layer 27：Reuse
```

这些层共同复用 Layer 20 Full 生成的 Main KV 和 Indexer K。

Layer 24：

```text
复用 Layer 20 Main KV
复用 Layer 20 Indexer K
生成 Layer 24 Indexer Q
重新选出 Top-K = {C3, C8, C100, ...}
生成 Layer 24 Main Q
生成 Layer 24 SWA KV
执行联合 Attention
```

Layer 25：

```text
复用 Layer 20 Main KV
复用 Layer 24 Top-K
不生成 Indexer Q
生成 Layer 25 Main Q
生成 Layer 25 SWA KV
执行联合 Attention
```

Layer 26、27继续复用 Layer 24 的 Top-K，但分别生成自己的 Main Q 和 SWA KV，并重新计算各层自己的 Attention 权重。因此，即使几个 Reuse 层使用相同 Main KV 和相同位置集合，Attention 输出仍然不同。

### 问题 12 最终总结

对原理解逐项判定：

1. **Full Mode**：基本正确。建议表述为“生成并发布新的 Main KV 及对应 Indexer K，计算当前层 Indexer Q，并在因果可见的 Main KV 条目中选择新的 Top-K；Decoder 第一个 Full 层还构建分块候选池”。
2. **Reindex Mode**：正确。它复用 Main KV/Indexer K，计算自己的 Indexer Q，并通常在共享候选池内产生新 Top-K。
3. **Reuse Mode**：核心正确，但不能说整个层‘仅’执行 Attention。它跳过 Main-KV 生成和 Indexer 流程，仍执行当前层 Main Q、SWA KV、Core Attention、输出投影和其余 Transformer Block。
4. **Main Q**：当前层真正参与 Sparse MLA 的 Query，同一份 Main Q 同时查询当前层 SWA KV 和 Selected Main KV；它与只负责产生 Top-K 的轻量 Indexer Q 不同。

一句话概括：

$$
\boxed{
\text{Main KV 是记忆库，Indexer K 是目录，Top-K 是书单，Main Q 是当前层真正提出的阅读问题。}
}
$$

Full 建新记忆库并重新列书单；Reindex 复用记忆库但重新列书单；Reuse 连书单也复用；然而三者都必须使用当前层自己的 Main Q 和 SWA KV 真正读取这些内容。

---

## 问题 13：Decoder Hierarchical Indexer 的 Full 层与共享候选池

### 问题 13 原文

> （1）论文 Decoder 中第一个 Full 层 Hierarchical Indexer 会：
>
> 1. 扫描全部因果可见 Main KV；
> 2. 产生自己的 Top-512；
> 3. 将压缩位置每 8 个划分为一块；
> 4. 选择最多 2048 个高分块；
> 5. 形成最多 16384 个候选位置。
>
> 请具体解释一下这个过程。
>
> （2）Decoder Hierarchical Indexer 下通常只对共享候选池重新打分，不重新扫描完整长上下文。

### 问题 13 核心结论

上述理解在论文算法层面基本正确，但必须修正一个关键的顺序关系：

> 第一个 Full 层不是“先得到 Top-512，再把这 512 个位置按 8 个分块”。

正确过程是：Full 层先为所有因果可见的 Main-KV 位置计算完整 Indexer Score，然后从同一份完整分数向量分出两条并行支路：

```text
全长 Indexer Score：s[0:N)
                │
        ┌───────┴────────────────┐
        │                        │
支路 A：位置级 Top-512      支路 B：每 8 个位置组成一块
        │                        │
本层 Attention 使用        块内 max → Top-2048 块
                                 │
                         最多 2048×8=16384 个
                         跨层共享候选位置
```

因此：

- 本层 Top-512 给当前 Full 层自己的稀疏 Attention 使用；
- 2048 个候选块给后面的 Reindex 层缩小搜索范围；
- 两者都来自完整位置分数，但候选块不是从本层 Top-512 中产生的。

### 13.1 “扫描全部 Main KV”究竟扫描什么

严格地说，Indexer 不是直接拿 Main K 做普通 Attention 打分，而是扫描与 Main-KV 逻辑位置一一对齐的 `Indexer K`：

$$
k_p^{\mathrm{idx}}
=
\operatorname{RMSNorm}
\left(
W_K^{\mathrm{idx}}C_p
\right).
$$

其中：

- $p$ 是 Main-KV Cache 中的逻辑位置；
- $C_p$ 是该位置对应的 compressor latent；
- $W_K^{\mathrm{idx}}$ 是 Indexer 专用的 K 投影矩阵；
- $k_p^{\mathrm{idx}}$ 是用于检索 Main KV 的轻量索引键；
- 真正的 Main K/V 只在 Top-512 被选出后参与 Main Attention。

所以“扫描全部 Main KV”更准确的表述是：

> 对全部因果可见 Main-KV 位置对应的 Indexer K 计算检索分数，但尚未对全部 Main V 执行正式 Attention。

上游 vLLM 的 `DeepseekV4Indexer` 说明，Indexer K 来自 KV-source layer 的 compressor latent；非 K-owner 的 Index Source 共享 KV Source 的分页 K Cache，见 `vllm/models/deepseek_v41/attention.py:1227-1240`。

Decoder 的 `compress_ratio=1`。本地测试配置为：

```python
compress_ratios = [0, 0] + [2] * 18 + [1] * 20
```

见 `vllm-ascend/tests/deepseek_v41_utils.py:273-282`。因此，在 Decoder Hierarchical Indexer 的语境中，一个 Main-KV 逻辑位置通常对应一个原始 token。这里所谓的“压缩位置”，更准确地说是“压缩 KV 架构中的 Main-KV 位置”；Decoder 的时间维压缩比例本身是 1。

### 13.2 Full 层的全长 Indexer Score

假设当前 Query 为 $q_t$，共有 $N_t$ 个因果可见 Main-KV 位置：

$$
p=0,1,\ldots,N_t-1.
$$

概念上，每个位置的 Indexer Score 为：

$$
s_{t,p}
=
\sum_{h=1}^{H_{\mathrm{idx}}}
w_{t,h}
\operatorname{ReLU}
\left(
\frac{
(q^{\mathrm{idx}}_{t,h})^\top k^{\mathrm{idx}}_p
}{\sqrt{d_{\mathrm{idx}}}}
\right).
$$

各项含义如下：

- $q^{\mathrm{idx}}_{t,h}$：当前 token 在第 $h$ 个 Indexer Head 上的 Query；
- $k^{\mathrm{idx}}_p$：第 $p$ 个 Main-KV 位置对应的 Indexer K；
- $w_{t,h}$：当前 Query 对不同 Indexer Head 的动态权重；
- ReLU：将负相关得分截断；
- $s_{t,p}$：位置 $p$ 对当前 Query 的最终检索重要性。

因果不可见、越界和 padding 位置的分数被设为：

$$
s_{t,p}=-\infty.
$$

Full 层需要得到完整位置分数向量：

$$
\mathbf s_t=
[s_{t,0},s_{t,1},\ldots,s_{t,N_t-1}].
$$

Ascend 侧在 `vllm_ascend/models/deepseek_v41/indexer.py:193-227` 中量化 Indexer Query，并将分页 INT8 Indexer-K Cache、Query Scale、Key Scale、分页元数据和候选模式传给 QLI 算子。

### 13.3 两条并行支路：本层 Top-512 与候选块

#### 支路 A：本层自己的 Top-512

Full 层直接从完整位置分数中选择：

$$
\mathcal I_t^{(20)}
=
\operatorname{TopK}_{512}
\left\{
s_{t,p}\mid 0\leq p<N_t
\right\}.
$$

$\mathcal I_t^{(20)}$ 是最多 512 个 Main-KV 位置编号，随后用来读取真正的 Selected Main KV，并与当前层 Local SWA KV 一起执行稀疏 Attention。

#### 支路 B：构造共享候选块

同一份完整分数还会进入块级聚合分支：每 8 个连续位置组成一块，先产生块分数，再选择最多 2048 个块。

Ascend 算子设计中的 source 模式明确规定：

```text
sparse_indices       = topK(full_score)
candidateTopkIndex   = topK(block_score)
```

即本层 Top-512 和跨层候选块是并行产生的两个输出。对应形式化定义见 `csrc/attention/quant_lightning_indexer_v2/docs/qli_v2_two_level_topk_design.md:105-127`。该文档还明确说明 source 层自身继续执行正常的全量 Top-K，而不是先用候选块 Mask 自己。

### 13.4 每 8 个位置划块及块内最大值

候选块大小为：

$$
B_s=8.
$$

位置 $p$ 所属的块为：

$$
b(p)=\left\lfloor\frac{p}{8}\right\rfloor.
$$

块总数为：

$$
N_{\mathrm{block}}
=
\left\lceil\frac{N_t}{8}\right\rceil.
$$

| 块编号 | 包含的 Main-KV 位置 |
|---:|---|
| 0 | 0～7 |
| 1 | 8～15 |
| 2 | 16～23 |
| 3 | 24～31 |
| $b$ | $8b$～$8b+7$ |

这里的 8-position candidate block 是 Hierarchical Indexer 的算法逻辑块，不是 vLLM PagedAttention 的 KV Cache page/block。PagedAttention 的物理 Cache Block 可能是 64、128 等其他大小，两者不能混用。

每个候选块的分数定义为块内位置分数的最大值：

$$
S_{t,b}^{\mathrm{block}}
=
\max_{\substack{p\in[8b,8b+7]\\p<N_t}}
s_{t,p}.
$$

使用最大值的含义是：只要一个块中存在一个很重要的位置，就保留这个位置所在的整个局部块。

假设为了演示只选择两个候选块：

| 块 | 位置范围 | 块内位置分数示意 | 块分数 |
|---:|---|---|---:|
| 0 | 0～7 | 0.1, 0.8, 0.2, … | 0.8 |
| 1 | 8～15 | 0.4, 3.2, 0.7, … | 3.2 |
| 2 | 16～23 | 1.1, 0.3, 0.6, … | 1.1 |
| 3 | 24～31 | 0.5, 2.7, 0.1, … | 2.7 |
| 4 | 32～39 | 0.4, 0.2, 0.1, … | 0.4 |

块级 Top-2 为 $\{B_1,B_3\}$，展开后得到：

$$
\{8,\ldots,15\}\cup\{24,\ldots,31\},
$$

一共 16 个候选位置。

Ascend 设计中的块定义、尾块补 $-\infty$ 和块内 `amax` 见 `qli_v2_two_level_topk_design.md:15-26`、`:38-44`。

当前 Ascend 910B 实现还有一个工程细节：将当前 Query 因果可见范围中的最后一个块强制置为 $+\infty$，确保最新位置所在块进入候选池：

$$
S^{\mathrm{block}}_{t,b_{\mathrm{last}}}=+\infty.
$$

该 pin 规则只影响候选块分支，不会强制最新位置进入 Full 层自己的位置级 Top-512。

### 13.5 选择 2048 个块并形成最多 16384 个候选位置

在块分数上执行：

$$
\mathcal B_t
=
\operatorname{TopK}_{2048}
\left(
S_{t,0}^{\mathrm{block}},
S_{t,1}^{\mathrm{block}},
\ldots
\right).
$$

实际有效块数为：

$$
|\mathcal B_t|
=
\min
\left(
2048,
\left\lceil N_t/8\right\rceil
\right).
$$

如果有效块不足 2048，就只保留全部有效块，多余输出槽写为 `-1`。候选位置集合为：

$$
\mathcal C_t
=
\bigcup_{b\in\mathcal B_t}
\{8b,8b+1,\ldots,8b+7\}.
$$

因此：

$$
|\mathcal C_t|\leq2048\times8=16384.
$$

但代码通常不直接保存 16384 个位置索引，而只保存 2048 个 INT32 块编号。消费者再把块 $b$ 展开为：

$$
[8b,8b+1,\ldots,8b+7].
$$

这样每个 Query 保存块编号只需要约 $2048\times4=8$ KB，而显式保存 16384 个位置索引需要约 64 KB，索引存储量缩小 8 倍。

Ascend Python 层明确说明 candidates 是 block IDs 而不是 position IDs，形状为 `[tokens, 1, candidate_topk_blocks]`，见 `vllm_ascend/models/deepseek_v41/indexer.py:159-180`。块编号展开语义见 `qli_v2_two_level_topk_design.md:79-89`。

只有在：

$$
N_t>2048\times8=16384
$$

时，候选池才真正裁剪搜索范围。若上下文只有 10000 个 Main-KV 位置，则块数为 1250，小于 2048，全部块都会被保留。

### 13.6 为什么候选池远大于最终 Top-512

候选池不是最终 Attention 集合，而是后续 Reindex 层的高召回率搜索空间。

如果 Full 层只把自己的 512 个最终位置传下去，后续层即使拥有新的 Hidden State 和 Indexer Query，也无法选择 Full 层丢弃的位置：

```text
Full 层认为重要的 512 个位置
             ↓
后续层只能在这 512 个位置中改变排序
```

保留 16384 个候选位置后，后续层可以执行更充分的重新排序：

```text
Full 层粗筛：全部 N 个位置 → 最多 16384 个候选位置
                                      ↓
Reindex 层精排：候选位置 → 自己的 Top-512
```

候选池最大是最终 Top-512 的：

$$
\frac{16384}{512}=32
$$

倍，为不同层的表示差异保留了较大的调整空间。块级选取还会把高分位置附近的 7 个邻居一同保留；即使后续层认为相邻的另一个 token 更重要，它仍有机会被选中。

### 13.7 Decoder 各层的 Full、Reindex、Reuse 数据流

DeepSeek-V4.1-Flash 的 20 层 Decoder 从 Layer 20 开始，按 5 组、每组 4 层组织：

| Decoder 层 | 模式 | Indexer 行为 |
|---|---|---|
| 20 | Full / Candidate Source | 扫描全部位置，产生本层 Top-512，同时生成共享候选块 |
| 21～23 | Reuse | 不重新运行 Indexer，复用 Layer 20 的 Top-512 |
| 24 | Reindex | 在共享候选池内重新打分，产生自己的 Top-512 |
| 25～27 | Reuse | 复用 Layer 24 的 Top-512 |
| 28 | Reindex | 在同一共享候选池内重新打分 |
| 29～31 | Reuse | 复用 Layer 28 的 Top-512 |
| 32 | Reindex | 在同一共享候选池内重新打分 |
| 33～35 | Reuse | 复用 Layer 32 的 Top-512 |
| 36 | Reindex | 在同一共享候选池内重新打分 |
| 37～39 | Reuse | 复用 Layer 36 的 Top-512 |

对应配置为：

```python
index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer_id = 20
```

见 `tests/deepseek_v41_utils.py:276-282`。Ascend 在 `vllm_ascend/models/deepseek_v41/model.py:494-536` 中根据最近的 KV Source 和 Index Source 构造每层角色；Layer 20 是 Candidate Source，后续 Index Source 的 `uses_candidate_filter=True`。

整体数据流为：

```text
Layer 20 Full
  ├─ TopK20 ──────────────────────→ Layer 20 Attention
  ├─ TopK20 ─→ Layer 21/22/23 Reuse
  └─ Candidate Blocks P
          │
          ├─ Layer 24 Reindex(P) → TopK24 → Layer 25/26/27 Reuse
          ├─ Layer 28 Reindex(P) → TopK28 → Layer 29/30/31 Reuse
          ├─ Layer 32 Reindex(P) → TopK32 → Layer 33/34/35 Reuse
          └─ Layer 36 Reindex(P) → TopK36 → Layer 37/38/39 Reuse
```

因此，共享的是 Full 层产生的候选池；不同 Reindex 层使用自己的 Indexer Query 和 Head Weight 重新打分，各自得到不同的最终 Top-512。

Ascend 执行代码位于 `vllm_ascend/attention/dsa_v41.py:440-472`：非 Index Source 直接读取共享 `topk_indices`；Index Source 调用 Indexer；Candidate Source 另外把 `candidates` 写入共享状态。

候选池只在同一次 forward 内共享，不能直接跨 decode step 复用。新 token 到来后，Query 和因果可见范围均发生变化，需要重新生成本步候选池。该约束见 `qli_v2_two_level_topk_design.md:55-56`。

### 13.8 Reindex 是否真的不扫描完整长上下文

从论文算法语义看，这句话正确。理想实现中：

$$
C_{\mathrm{Full}}
\propto N_t,
$$

而每个 Reindex 层只对共享候选池重新打分：

$$
C_{\mathrm{Reindex}}
\propto\min(N_t,16384).
$$

Reindex 层自己的打分可形式化为：

$$
s'_{t,p}
=
\begin{cases}
s^{(l)}_{t,p}, & p\in\mathcal C_t,\\
-\infty, & p\notin\mathcal C_t,
\end{cases}
$$

然后选择：

$$
\mathcal I_t^{(l)}
=
\operatorname{TopK}_{512}(s'_{t,:}).
$$

但必须区分两种含义：

1. **逻辑搜索域缩小**：最终 Top-512 只能从最多 16384 个共享候选位置中产生；
2. **物理计算量缩小**：底层 kernel 是否真的只加载并计算这 16384 个位置的 QK，而不计算候选池外位置。

论文描述的是理想的层级检索算法；具体后端未必已经完全实现第二种优化。

### 13.9 理论复杂度与长上下文示例

假设 Decoder 有一个 Full Indexer 和四个 Reindex Indexer。如果五个 Index Source 都扫描完整上下文，则位置级打分数为：

$$
C_{\mathrm{dense}}\propto5N_t.
$$

Hierarchical Indexer 的理想位置级打分数为：

$$
C_{\mathrm{hier}}
\propto
N_t+4\min(N_t,16384).
$$

当 $N_t=100000$ 时：

$$
C_{\mathrm{dense}}=500000,
$$

$$
C_{\mathrm{hier}}
=100000+4\times16384
=165536.
$$

理论缩减比例约为：

$$
\frac{500000}{165536}\approx3.02.
$$

单个 Reindex 层的打分位置数理论上缩小：

$$
\frac{100000}{16384}\approx6.10
$$

倍。

当 $N_t=1000000$ 时：

$$
C_{\mathrm{dense}}=5000000,
$$

$$
C_{\mathrm{hier}}
=1000000+4\times16384
=1065536,
$$

约缩小：

$$
\frac{5000000}{1065536}\approx4.69
$$

倍。

上述只是 Decoder Indexer 的检索打分量，不是整个 Transformer 的总 FLOPs，也不包括 Main Attention、SWA、MoE/MLP 等计算。

### 13.10 当前 vLLM 与 vLLM-Ascend 实现差异

#### 上游 vLLM

上游 vLLM 支持可选的 `attention_config.indexer_sparse_logits`。Candidate Consumer 可以使用 `SparseMQAIndexer` 只对候选块计算 logits；当前代码还要求搭配 MXFP4 Indexer Cache。对应代码见 `vllm/models/deepseek_v41/attention.py:1329-1369`。

这说明上游实现显式区分：

```text
候选 Mask 保证算法语义正确
              和
只读取、计算候选块以真正减少 QK 工作量
```

#### 当前本地 vLLM-Ascend 910B 路径

从本地 910B QLI 两级候选设计能够确认，Consumer 模式当前的执行顺序是：

```text
全长 QK
  → 全长 score 后处理
  → 根据候选块生成 Mask
  → 候选块外分数置为 -∞
  → Top-K 排序
```

`qli_v2_two_level_topk_design.md:157-167` 明确显示：

- MatMul1 的 QK 阶段不变；
- score 后处理阶段不变；
- 在完整 score 已经生成后，才追加 candidate block mask；
- 最后执行 SortAll/Top-K。

因此，对这条具体实现路径，最严谨的结论是：

| 判断 | 结论 |
|---|---|
| Reindex 的 Top-512 被限制在最多 16384 个候选位置内 | 成立 |
| Reindex 不重新生成完整的全局候选池 | 成立 |
| Reindex 的最终逻辑搜索域不包含候选池外位置 | 成立 |
| 当前 910B kernel 完全不计算候选池外 QK | 从当前代码看不成立 |
| 论文理论上的 QK 缩减已在所有后端实现 | 不能这样断言 |

Ascend Python 层在 `vllm_ascend/models/deepseek_v41/indexer.py:212-226` 把三种候选模式映射为：

```python
mode = 1  # Candidate Source / Full
mode = 2  # Candidate Consumer / Reindex
mode = 3  # Candidate 功能关闭
```

该模式保证 source 生成候选块、consumer 在候选块内得到最终 Top-K，但是否在 QK 之前真正稀疏读取，仍取决于底层 kernel。

本地设计文档还说明该候选扩展只在 `ascend910b` 配置中声明，950 配置没有同步修改，见 `qli_v2_two_level_topk_design.md:308-311`。因此不能仅根据这份本地源码断言 Ascend 950 是否存在另一套外部优化实现。

### 问题 13 最终总结

第一个 Decoder Full Indexer 的完整过程可以概括为：

1. 对全部因果可见 Main-KV 位置对应的 Indexer K 计算完整位置分数；
2. 从完整分数直接选择本层自己的 Top-512；
3. 同时将完整位置空间按每 8 个相邻位置划为一个候选块；
4. 对每个块取块内最大位置分数；
5. 选择最多 2048 个高分块；
6. 这 2048 个块在逻辑上展开为最多 16384 个候选位置，并在同一次 forward 内共享给后续 Reindex 层；
7. 后续 Reindex 层使用自己的 Indexer Query 对共享候选池重新排序，分别产生新的 Top-512；
8. Reuse 层不重新运行 Indexer，直接复用最近一个 Full/Reindex 层的 Top-512。

一句话概括：

$$
\boxed{
\text{Full 层在全库中粗筛候选区域并完成自己的精排；Reindex 层在候选区域内重新精排。}
}
$$

同时要注意：论文算法上的“只在候选池中重新打分”不必然意味着所有当前硬件实现都已经跳过候选池外的 QK 计算。当前本地 Ascend 910B 路径能够确认的是“全长 Score 后候选 Mask”；是否实现真正的候选块稀疏 QK，需要继续查看对应后端的专用 kernel。

---

## 问题 14：普通 KV Cache 究竟存储什么

### 问题 14 原文

> 普通 KV Cache 存储的内容究竟是什么？

### 问题 14 核心结论

普通 decoder-only Transformer 的 KV Cache 保存的是：

> 每一层 Attention 对已经处理过的 token 计算出的 Key 向量和 Value 向量。

对于第 $l$ 层、第 $i$ 个 token：

$$
K_i^{(l)}=H_i^{(l)}W_K^{(l)},qquad
V_i^{(l)}=H_i^{(l)}W_V^{(l)}.
$$

如果模型使用 RoPE，普通 Llama 类路径通常缓存应用位置旋转后的 Key：

$$
\widetilde K_i^{(l)}=R(i)K_i^{(l)}.
$$

因此，一层普通 KV Cache 的逻辑内容为：

$$
\boxed{
\mathrm{KVCache}^{(l)}=
\{\widetilde K_1^{(l)},V_1^{(l)},\ldots,
\widetilde K_N^{(l)},V_N^{(l)}\}
}.
$$

它不是 token ID、token embedding、模型权重、完整 Hidden State、历史 Query、Attention Score、Softmax 权重、Attention 输出或最终 logits。它是特定请求、特定 Attention 层、特定历史 token 的运行时中间激活状态。

### 14.1 K 和 V 是怎样产生的

第 $l$ 层输入 Hidden State 为：

$$
H^{(l)}=[h_1^{(l)},h_2^{(l)},\ldots,h_N^{(l)}],qquad
h_i^{(l)}\in\mathbb R^{d_{\mathrm{model}}}.
$$

通过三个投影得到：

$$
Q^{(l)}=H^{(l)}W_Q^{(l)},qquad
K^{(l)}=H^{(l)}W_K^{(l)},qquad
V^{(l)}=H^{(l)}W_V^{(l)}.
$$

可将其理解为：

- Query：当前 token 想查找什么；
- Key：历史 token 提供的检索标签；
- Value：该历史 token 被命中后真正提供的内容。

vLLM 的 Llama 实现先做融合 QKV 投影：

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split(
    [self.q_size, self.kv_size, self.kv_size],
    dim=-1,
)
```

随后对 Q、K 应用 RoPE，并进入 Attention：

```python
q, k = self.rotary_emb(positions, q, k)
attn_output = self.attn(q, k, v)
```

对应 `vllm/model_executor/models/llama.py:224-233`。因此在该标准路径里，Attention 后端接收到的是旋转后的 Q、K 和未做 RoPE 的 V，随后把 K/V 写入缓存。

### 14.2 为什么必须同时保存 K 和 V

普通 Attention 为：

$$
A_t=operatorname{Softmax}
\left(
\frac{Q_tK_{\leq t}^{\top}}{\sqrt{d_h}}+\mathrm{Mask}
\right),
$$

$$
O_t=A_tV_{\leq t}.
$$

其中：

- 当前 Query 与历史 Key 点积，决定应该关注哪些历史位置；
- 得到的 Attention 权重再对历史 Value 加权求和，形成输出。

因此必须保存 K，未来 Query 才能检索历史位置；也必须保存 V，被命中后才能读取历史内容。只有 K 没有 V，相当于只有目录没有正文；只有 V 没有 K，则不知道应该读取哪部分正文。

### 14.3 为什么不保存历史 Query

对新 token $t$，只需计算自己的 Query：

$$
Q_t=H_tW_Q,
$$

然后查询历史 Key：

$$
Q_tK_1^\top,Q_tK_2^\top,\ldots,Q_tK_t^\top.
$$

下一 token 使用的是新的 $Q_{t+1}$，不会再次使用 $Q_1,\ldots,Q_t$。所以历史 Query 完成当步查询后没有复用价值，不需要写入 Cache。

```text
当前 Query
    │
    ├── 与历史 Key 比较，产生 Attention 权重
    └── 用这些权重加权历史 Value，产生输出
```

### 14.4 为什么 KV Cache 必须逐层保存

相邻层 Hidden State 不同：

$$
H^{(l+1)}=F_l(H^{(l)}),
$$

而且每层投影参数也不同：

$$
W_K^{(l)}\neq W_K^{(l+1)},qquad
W_V^{(l)}\neq W_V^{(l+1)}.
$$

所以同一个 token 在不同层的 K/V 一般也不同：

$$
K_i^{(l)}\neq K_i^{(l+1)},qquad
V_i^{(l)}\neq V_i^{(l+1)}.
$$

普通 40 层模型的缓存逻辑上是：

```text
Layer 0 : K0[token 0...N-1], V0[token 0...N-1]
Layer 1 : K1[token 0...N-1], V1[token 0...N-1]
...
Layer 39: K39[token 0...N-1], V39[token 0...N-1]
```

完整逻辑形状可抽象为：

$$
[L,2,N,H_{\mathrm{KV}},D_h],
$$

其中 $L$ 为层数，2 表示 K/V，$N$ 是已缓存 token 数，$H_{\mathrm{KV}}$ 是 KV Head 数，$D_h$ 是 Head Dim。实际实现通常按层和物理块分别管理，并不一定分配一个整体连续张量。

### 14.5 Prefill 和 Decode 怎样写入缓存

假设 Prompt 为：

```text
[token1, token2, token3]
```

Prefill 并行计算三个 token 在每层的 Q/K/V，但写入缓存的只有：

```text
K Cache = [K1, K2, K3]
V Cache = [V1, V2, V3]
```

Q1、Q2、Q3 只服务于本次 Prefill Attention，用完不缓存。Prefill 输出 logits 后采样出 `token4`。

下一次 Decode 把 `token4` 作为输入，在每层只计算：

$$
Q_4,K_4,V_4,
$$

并追加：

```text
K Cache = [K1, K2, K3, K4]
V Cache = [V1, V2, V3, V4]
```

当前 Query 执行：

$$
A_4=operatorname{Softmax}
\left(
\frac{Q_4[K_1,K_2,K_3,K_4]^\top}{\sqrt{d_h}}
\right),
$$

$$
O_4=A_4[V_1,V_2,V_3,V_4].
$$

| 阶段 | 本次输入 | Cache 写入后 | 采样出的 token |
|---|---|---|---|
| Prefill | token1～token3 | K/V 1～3 | token4 |
| Decode 1 | token4 | K/V 1～4 | token5 |
| Decode 2 | token5 | K/V 1～5 | token6 |

因此 Decode 不重新计算历史 token 的 K/V，只计算并追加当前输入 token 的 K/V。它仍需让当前 Query 读取越来越长的历史 K/V，但避免了历史 token 的投影、Attention、MLP/MoE 等完整前向计算。

### 14.6 vLLM 的分页物理布局

逻辑上，一层 Cache 可表示为：

$$
K_{\mathrm{cache}}\in\mathbb R^{N\times H_{\mathrm{KV}}\times D_h},qquad
V_{\mathrm{cache}}\in\mathbb R^{N\times H_{\mathrm{KV}}\times D_v}.
$$

标准 Attention 通常有 $D_v=D_h$。vLLM 不要求一个请求的缓存物理连续，而是划分为固定 Page/Block。概念布局为：

$$
[2,N_{\mathrm{blocks}},B,H_{\mathrm{KV}},D_h].
$$

Ascend 后端支持：

```python
(2, num_blocks, num_kv_heads, block_size, head_size)
```

或：

```python
(2, num_blocks, block_size, num_kv_heads, head_size)
```

见 `vllm_ascend/attention/attention_v1.py:102-112`。维度顺序虽不同，逻辑内容仍是每层、每个 token 的 K 和 V。

### 14.7 Slot Mapping 与 Block Table

`slot_mapping` 负责把当前输入 token 映射到具体物理缓存位置。若：

```text
block_size = 4
物理 Block = 7
块内偏移 = 2
```

其线性 slot 为：

$$
\mathrm{slot}=7\times4+2=30.
$$

Ascend 在 `attention_v1.py:1122-1155` 中把 `key_to_cache`、`value_to_cache`、`key_cache`、`value_cache` 和 `slot_mapping` 交给 `DeviceOperator.reshape_and_cache`；底层 NPU scatter 写入逻辑见 `device_op.py:93-122`。CUDA/FlashAttention 路径见 `vllm/v1/attention/backends/flash_attn.py:1511-1545`。

`block_table` 则负责读取时将请求的逻辑块映射到物理块。例如：

```text
Request A 逻辑块：[0, 1, 2]
对应物理块      ：[17, 3, 42]
```

CUDA FlashAttention 将 `query`、`key_cache`、`value_cache` 和 `block_table` 一起传给 Kernel，见 `flash_attn.py:1454-1479`；Ascend Paged Attention 的对应读取见 `attention_v1.py:1054-1072`。

### 14.8 KV Cache 的数据类型与量化内容

数学上缓存的是 K/V 向量，物理表示可能是 FP32、FP16、BF16、FP8、INT8、INT4、NVFP4 或其他专用格式。

BF16 Cache 通常直接保存 BF16 K/V；量化 Cache 则保存：

```text
量化后的 K/V 数据 + Scale
```

某些非对称量化还要保存 Zero Point。vLLM 的 KV Cache 类型枚举包含 FP8、INT8、INT4、NVFP4 等，见 `vllm/v1/kv_cache_interface.py:40-57`。

所以“KV Cache 保存 K/V”是数学层描述；实际显存中可能是 K/V 的量化表示和恢复它们所需的量化参数。

### 14.9 普通 KV Cache 的显存计算

普通 Attention 每层、每个 token 的缓存元素数为：

$$
2H_{\mathrm{KV}}D_h.
$$

若模型有 $L$ 层、序列长度 $N$、每元素 $b$ 字节，则总大小近似为：

$$
\boxed{
M_{\mathrm{KV}}
=L\times N\times2\times H_{\mathrm{KV}}\times D_h\times b
}.
$$

vLLM Cache Spec 也按：

$$
(\mathrm{head\_size}+\mathrm{head\_size\_v})
\times\mathrm{dtype\_size}
$$

计算单个 KV Head、单个存储状态的内容字节数，见 `vllm/v1/kv_cache_interface.py:486-536`。

例如：32 层、8 个 KV Heads、Head Dim 128、BF16 两字节、上下文 100K。每层每 token 为：

$$
2\times8\times128\times2=4096\ \mathrm{bytes}.
$$

全模型每 token：

$$
32\times4096=131072\ \mathrm{bytes}=128\ \mathrm{KiB}.
$$

100K token 约为：

$$
100000\times128\ \mathrm{KiB}\approx12.2\ \mathrm{GiB}.
$$

这还不包含模型权重、临时激活、Attention Workspace、Graph 内存、缓存块碎片和同一 Batch 中的其他请求。

### 14.10 MHA、GQA 与 MQA 的差别

三者保存的仍然是 K/V，主要差别是 KV Head 数：

| Attention 类型 | Query Heads | KV Heads | KV Cache 大小 |
|---|---:|---:|---:|
| MHA | $H_q$ | $H_q$ | 最大 |
| GQA | $H_q$ | 小于 $H_q$ | 较小 |
| MQA | $H_q$ | 1 | 最小 |

例如 Query Heads 为 32，MHA 有 32 个 KV Heads，而 GQA 只有 8 个，则 GQA KV Cache 约为 MHA 的：

$$
\frac{8}{32}=\frac14.
$$

vLLM Llama 实现分别记录 `num_heads`、`num_kv_heads`、`q_size` 和 `kv_size`，见 `llama.py:150-162`。因此 KV Cache 大小主要由 KV Head 数而不是 Query Head 数决定。

### 14.11 普通 KV Cache 与 DeepSeek-V4.1 的区别

DeepSeek-V4.1-Flash 从多个维度压缩 KV Cache：

1. GQA 减少 KV Head 数；
2. MLA 保存较小 KV Latent，而不是全部展开后的多头 K/V；
3. Token/KV Compression 将多个 token 压成一个 Main-KV Entry；
4. Cross-layer Reuse 让多个层共享 Main KV；
5. FP4 减少单个缓存元素的字节数；
6. Sparse Attention 只选择部分 Main KV 做最终 Attention。

论文说明 DeepSeek-V4.1 的 Global KV 包括 Main KV 和 Indexer K，而 Local SWA 保存 Layer-local SWA KV。对比如下：

| 类型 | 实际缓存内容 |
|---|---|
| 普通 MHA/GQA | 每层历史 token 的 K 和 V |
| 普通量化 KV Cache | 量化 K、量化 V、Scale，必要时还有 Zero Point |
| MLA | KV Latent，必要时另存 RoPE Key 分量 |
| DeepSeek-V4.1 Main KV | 压缩后的 Main-KV latent/entry |
| DeepSeek-V4.1 Indexer Cache | 与 Main-KV 位置对齐的 Indexer K 和量化 Scale |
| DeepSeek-V4.1 SWA KV | 每层最近滑动窗口内的局部 K/V |

因此 DeepSeek 论文中的“KV Cache”不一定是普通意义上的完整 $[K_1,V_1,K_2,V_2,\ldots]$，而可能是能够在 Attention 中等价或近似承担 K/V 功能的压缩表示。

### 14.12 常见误区

1. **KV Cache 是否保存原始文本？** 不保存。缓存中没有可直接阅读的文本或 token ID，而是模型内部的浮点或量化向量。
2. **能否从一条 K/V 恢复原 token？** 一般不能可靠恢复。K/V 是经过上下文混合和投影的内部表示，不是可逆文本编码。
3. **KV Cache 是否属于模型参数？** 不属于。$W_Q,W_K,W_V,W_O$ 等是模型参数；KV Cache 是随请求变化的运行时激活。
4. **不同请求能否共享 Cache？** 只有前缀完全相同且位置、模型版本、Adapter、缓存精度等兼容时，Prefix Caching 才能共享对应 K/V Block；语义相似并不足以直接复用。
5. **K 一定以 RoPE 后形式保存吗？** 前述 vLLM Llama 路径是先做 RoPE 再写 K；但某些 MLA、解耦 RoPE、相对位置编码或专用融合算子可能缓存未旋转 K、KV Latent、独立 RoPE K 分量或其他可恢复表示，不能推广为所有架构的绝对规则。

### 问题 14 最终总结

普通 KV Cache 的本质是：

$$
\boxed{
\text{缓存每一层历史 token 的 Key 和 Value，使未来 token 只需计算新的 Q/K/V，而不必重算完整历史网络。}
}
$$

形象地说：

```text
历史 Key   = 可被未来 Query 检索的目录
历史 Value = 被命中后真正读取的内容
当前 Query = 当前 token 提出的问题
```

它缓存的是历史 token 在每一层 Attention 中已经加工好的可检索记忆，而不是 token 本身，也不是完整 Hidden State。普通模型通常逐层、逐 token 保存 K/V；DeepSeek-V4.1 则进一步将其改造成 Main KV、Indexer K 和 SWA KV 等压缩、共享和稀疏化表示。

---

## 问题 15：FP4 Indexer Q/K、QAT、MXFP4 与 FP4 Main KV

### 问题 15 原文

> 在论文的 2.4.4. FP4 Main KV Cache 中提到的内容，我不是很明白，DeepSeek-V4 为 FP4 indexer Q 和 FP4 indexer Q 使用 QAT 的具体实施过程和原理是怎样的？此时使用的 OCP-standard MXFP4 格式具体是怎样的格式呢？在注意力计算之前，反量化缓存值又为什么可以允许我们使用更精确的格式，从而可以不再要求对这种格式的本地矩阵乘法的支持呢？

### 问题 15 核心结论

原问题中第二个 “FP4 Indexer Q” 应为 **FP4 Indexer K**。论文这一节实际包含两种 FP4 路线：

| 对象 | 模型 | 格式 | 主要目标 |
|---|---|---|---|
| Indexer Q/K | DeepSeek-V4 已使用，V4.1 延续 | OCP MXFP4：E2M1 + 每 32 元素一个 E8M0 Scale | 压缩 Indexer Cache，同时加速大规模检索打分 |
| Main KV | DeepSeek-V4.1 新扩展 | E2M1 + 每 16 通道一个 E4M3 Scale，无二级 Global Scale | 主要压缩长期 Cache 存储 |
| SWA KV | DeepSeek-V4.1 | FP8 | SWA 对量化更敏感 |

所以 OCP-standard MXFP4 与后文“每 16 通道一个 E4M3 Scale”不是同一种格式：前者用于 Indexer Q/K；后者是 Main KV 使用的 NVFP4 风格单级 Block-Scaled FP4。

### 15.1 Indexer Q/K 为什么适合 FP4

Indexer 给 Main-KV 位置打检索分数：

$$
s_{t,j}=
\sum_h w_{t,h}
\operatorname{ReLU}
\left(
(q^{\mathrm{idx}}_{t,h})^\top k^{\mathrm{idx}}_j
\right).
$$

长上下文中，一个 Query 需要与几十万甚至一百万个 Indexer K 打分。Indexer 的特点是：

```text
Indexer K Cache 很长
        +
Q × K 检索矩阵乘法规模很大
        +
最终主要关心 Top-K 排序
```

将 Indexer Q/K 同时压成 FP4，可以减少 Indexer K Cache；如果硬件支持原生 MXFP4 点积，还能直接加速检索。但 FP4 误差可能改变得分排序乃至 Top-K 集合，因此 DeepSeek 使用 QAT，而不是简单训练后量化。

### 15.2 Indexer Q/K 的 QAT 过程与原理

QAT 即 Quantization-Aware Training。训练前向中模拟部署时的量化和反量化：

$$
\widehat x=D(Q(x)).
$$

对 Indexer Q/K：

$$
\widehat q^{\mathrm{idx}}
=D_{\mathrm{MXFP4}}(Q_{\mathrm{MXFP4}}(q^{\mathrm{idx}})),
$$

$$
\widehat k^{\mathrm{idx}}
=D_{\mathrm{MXFP4}}(Q_{\mathrm{MXFP4}}(k^{\mathrm{idx}})).
$$

训练打分使用量化误差污染后的 Q/K：

$$
\widehat s_{t,j}=
\sum_h w_{t,h}
\operatorname{ReLU}
\left(
(\widehat q^{\mathrm{idx}}_{t,h})^\top
\widehat k^{\mathrm{idx}}_j
\right).
$$

概念流程为：

```text
Indexer Q/K 投影
      ↓
RMSNorm / RoPE
      ↓
按 32 通道分块
      ↓
计算 E8M0 Scale，量化为 E2M1
      ↓
立即反量化，得到 Q̂/K̂
      ↓
Indexer Score、Top-K、训练 Loss
```

由于舍入函数几乎处处导数为零，QAT 通常采用 Straight-Through Estimator：前向真实模拟舍入、饱和和反量化，反向在有效范围内近似令：

$$
\frac{\partial\widehat x}{\partial x}\approx1.
$$

这样模型可以主动减少离群值、调整 Q/K 分布、扩大重要位置与非重要位置的得分间隔，使 Top-K 对量化误差更稳健。

例如高精度分数为 $2.031$ 和 $2.028$ 时，FP4 误差很容易使排序反转；QAT 可以把有用位置训练成更大的 margin，例如 $2.5$ 对 $1.8$。

论文和这两个推理仓库没有公开完整训练代码，因此训练步数、学习率、STE 精确形式、是否采用随机舍入或额外排序蒸馏损失均不能从当前源码确认。能够确认的是论文声明 DeepSeek-V4 对 Indexer Q/K 使用 QAT，V4.1 又在 post-training 中把 QAT 扩展到 Main KV。

### 15.3 OCP MXFP4 的位级格式

OCP MXFP4 是 Microscaling Format：

```text
32 个 FP4 E2M1 Element
        +
1 个由这 32 个 Element 共享的 E8M0 Scale
```

| 属性 | MXFP4 |
|---|---|
| Element 类型 | FP4 E2M1 |
| Element 位宽 | 4 bit |
| Block Size | 32 |
| Block Scale | E8M0 |
| Scale 位宽 | 8 bit |

一个 E2M1 值的 4 bit 为：

```text
┌──────┬──────────┬──────────┐
│ Sign │ Exponent │ Mantissa │
│ 1bit │   2bit   │   1bit   │
└──────┴──────────┴──────────┘
```

可表示的非负幅值为：

$$
\{0,0.5,1,1.5,2,3,4,6\}.
$$

E8M0 没有符号位和 Mantissa，Bias 为 127，表示的 Scale 基本为 2 的整数次幂：

$$
s=2^{e-127}.
$$

它硬件实现简单，但 Scale 只能取 $\ldots,0.5,1,2,4,8,\ldots$，不够细致。

### 15.4 MXFP4 的量化公式和数值例子

对 32 元素块 $X_b$：

$$
a_{\max}=\max_i|x_i|,
$$

vLLM 当前实现选择：

$$
s_b=2^{\left\lceil\log_2(a_{\max}/6)\right\rceil}.
$$

随后：

$$
p_i=Q_{\mathrm{E2M1}}(x_i/s_b),qquad
\widehat x_i=s_bp_i.
$$

假设块中代表性数值为：

$$
[1.2,2.7,-5.2,9.2,\ldots].
$$

有 $a_{\max}=9.2$，所以：

$$
s=2^{\lceil\log_2(9.2/6)\rceil}=2.
$$

归一化为 $[0.6,1.35,-2.6,4.6]$，舍入到 E2M1 后为 $[0.5,1.5,-3,4]$，反量化得到 $[1,3,-6,8]$。QAT 就是让模型在训练阶段看到这种误差。

32 个 E2M1 共 16 字节，再加 1 字节 E8M0 Scale，总计 17 字节，即平均 4.25 bit/元素。128 维 Indexer K 需要：

$$
128/2+128/32=64+4=68\ \mathrm{bytes},
$$

而 BF16 需要 256 字节，约缩小 3.76 倍。

### 15.5 vLLM 中 Indexer Q/K 的 MXFP4 实现

Indexer Q 的 MXFP4 表示为：

```text
q_packed: [T, H, head_dim/2]，uint8，每字节两个 E2M1
q_scale : [T, H, head_dim/32]，uint8 E8M0
```

具体说明见 `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:555-588`；量化 Scale 计算和 E2M1 打包见同文件 `:57-76`。

V4.1 Indexer Q 的执行链为：

```text
qr → Wq_b → [token, 32 heads, 128]
   → RoPE + MXFP4 Quant → (q_packed, q_scale)
```

调用见 `vllm/models/deepseek_v41/attention.py:1411-1440`。

Indexer K 来自 Main Compressor Latent：

$$
k^{\mathrm{idx}}=\operatorname{RMSNorm}(W_K^{\mathrm{idx}}C),
$$

然后执行 RoPE、MXFP4 Quant，并写入 Paged Indexer-K Cache，见 `common/ops/indexer_k_store.py:3-54`、`:210-236`。

原生 MXFP4 块点积可写为：

$$
\operatorname{Dot}(Q_b,K_b)
=s_b^Qs_b^K\sum_{i=0}^{31}p_{b,i}^Qp_{b,i}^K.
$$

支持 MXFP4 的硬件可以读取 E2M1 和 E8M0 Scale，直接完成低精度乘法并用更高精度累加。

### 15.6 V4.1 Main KV 为什么不照搬 OCP MXFP4

Indexer 与 Main KV 的目标不同：

```text
Indexer Q/K：既压缩 Cache，又要直接加速超长 QK 扫描
Main KV    ：主要压缩长期存储，只对选出的 Top-K 做最终 Attention
```

Main KV 可以采用：

```text
FP4 长期存储
   ↓
Indexer 选择 Top-K
   ↓
只 Gather Selected Main KV
   ↓
反量化为 BF16
   ↓
普通 BF16 Sparse Attention
```

因此它无需迁就硬件原生 FP4 MatMul 的 Operand 格式，可以选择精度更高的自定义 4-bit 存储格式。

### 15.7 V4.1 Main KV 的 E2M1 加 E4M3 格式

论文选择：

```text
E2M1 数值
+ 每 16 通道一个 E4M3 Scale
+ 不保存第二级 Global Scale
```

即：

$$
\widehat x_i=p_i^{\mathrm{E2M1}}s_b^{\mathrm{E4M3}}.
$$

标准 NVFP4 通常还包含整个张量共享的 FP32 Global Scale：

$$
\widehat x_i=p_i^{\mathrm{E2M1}}s_b^{\mathrm{E4M3}}s_{\mathrm{global}}^{\mathrm{FP32}}.
$$

所以 V4.1 Main KV 更准确地说是 NVFP4 风格、但省略 Global Scale 的单级格式，不是完整标准 NVFP4 Recipe。

它比 OCP MXFP4 更精确，原因是：

1. Block Size 从 32 缩小到 16，一个离群值只影响另外 15 个元素；
2. E4M3 Scale 带 3 位 Mantissa，比只能表示 2 的幂的 E8M0 更贴近理想的 $a_{\max}/6$。

512 维 Main-KV Entry 被 reshape 成 $[32,16]$：512 个 E2M1 占 256 字节，32 个 E4M3 Scale 占 32 字节，总计 288 字节。BF16 需要 1024 字节，因此约缩小 3.56 倍。对应实现见 `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py:233-259`、`:421-461`。

### 15.8 为什么可以省略 Global Scale

论文说明 Main-KV Latent 经过 RMSNorm：最大的训练后 RMSNorm 权重大约为 1；512 维归一化向量的 $L_2$ 范数约为：

$$
\sqrt{512}\approx22.6,
$$

实际训练观察到的最大幅值约为 10。

E4M3 Scale 最大约 448，E2M1 最大幅值为 6，不带 Global Scale 时覆盖范围仍达：

$$
448\times6=2688,
$$

远高于实际约 10～22.6 的幅值。省略 Global Scale 因此不会带来可测量精度下降，同时可以简化 Cache Layout、减少量化/反量化步骤，并避免 Global Amax 同步。

### 15.9 为什么在 RoPE 后量化

论文对 NoPE 和 RoPE 部分采用相同格式，并选择：

$$
C\xrightarrow{\mathrm{RoPE}}C_{\mathrm{rotated}}
\xrightarrow{Q_{\mathrm{FP4}}}C_{\mathrm{cache}}.
$$

RoPE 虽然理想上保持二范数，但会改变通道分布。若在 RoPE 前量化，读取时还需要恢复数据并完成位置旋转，增加 Decode 热路径开销。论文实验认为 RoPE 前量化仅有很小的精度收益，不值得额外开销。vLLM 也是先产生 `rotated`，再按 `[32,16]` 量化，见 `fused_compress_quant_cache.py:433-461`。

### 15.10 反量化为何解除原生 FP4 MatMul 要求

若直接执行 FP4 MatMul，硬件必须原生识别 E2M1、Scale 类型、Block Size、Scale Layout 和相应矩阵乘指令：

```text
FP4 Q × FP4 K → 原生 FP4 Matrix Core → 高精度累加
```

Main KV 则采用：

```text
E2M1 + E4M3/16 Cache
        ↓
解包高低 4-bit nibble
        ↓
E2M1 恢复 + 乘 E4M3 Scale
        ↓
BF16 Main KV Workspace
        ↓
普通 BF16 Attention
```

反量化公式为：

$$
\widehat x_i=\operatorname{BF16}
\left(p_i^{\mathrm{E2M1}}s_b^{\mathrm{E4M3}}\right).
$$

矩阵乘法实际看到的是 BF16：

$$
Q_{\mathrm{BF16}}\widehat K_{\mathrm{BF16}}^\top,
\qquad
P_{\mathrm{BF16}}\widehat V_{\mathrm{BF16}}.
$$

所以硬件只需支持 4-bit 解包、Scale 乘法和普通 BF16 Attention，不需要支持“E2M1 + E4M3/16 + 无 Global Scale”的原生 FP4 MMA。

本质上是：

$$
\boxed{\text{Cache Encoding 与 MatMul Operand Encoding 解耦。}}
$$

不是反量化让 FP4 本身更准确，而是 Cache 不再受硬件原生 MMA 格式限制，因此可以选用更细粒度、量化误差更小的存储格式。

vLLM 的反量化 Kernel 从分页 Cache 读取打包字节，分离两个 nibble，恢复 E2M1 幅值，读取 E4M3 Scale，相乘后写入 BF16 Workspace，见 `common/ops/cache_utils.py:503-566`。

### 15.11 反量化是否抵消 FP4 收益

反量化确实增加计算和临时 Workspace，但长期收益仍然存在：

- Main KV 在 HBM、Host Memory、SSD 中长期只占 288 字节/Entry，而 BF16 为 1024 字节；
- Cache 迁移和跨节点传输流量下降；
- 反量化后的 BF16 只是短生命周期 Workspace；
- Sparse Attention 先选 Top-512，只反量化 Selected Main KV，而不是展开完整百万上下文。

因此反量化量近似从全长的 $N\times512$ 个元素缩小为 $K\times512$，其中 $K=512$。如果是全量 Dense Attention，每一步都反量化百万 token，收益可能被开销部分抵消；Sparse Attention 是该设计成立的重要前提。

### 15.12 Indexer 与 Main Attention 的不同选择

Indexer 要扫描长上下文，若先把全部 Indexer K 反量化，会产生巨大的 BF16 临时数据并失去 FP4 QK 加速，所以更适合直接用原生 OCP MXFP4 点积。

Main Attention 只读取 Top-K，把约 512 个 Entry 反量化为 BF16 的成本较小。因此合理分工为：

| 路径 | FP4 的主要作用 | 计算方式 |
|---|---|---|
| Indexer Q/K | 存储压缩 + 检索计算加速 | 原生 OCP MXFP4 点积 |
| Selected Main KV | 长期存储压缩 | 先反量化，再做 BF16 Attention |
| SWA KV | 局部窗口且量化敏感 | 保留 FP8 |

### 15.13 vLLM 与 vLLM-Ascend 当前实现差异

当前 `vllm-github` 能看到两条路径：

- `attention_config.indexer_kv_dtype="mxfp4"`：Indexer Q/K 使用 E2M1 + E8M0/32；配置见 `vllm/config/attention.py:83-99`；
- `nvfp4_ds_mla`：Main KV 保存 256 字节 E2M1 加 32 字节 E4M3 Scale，并提供 Gather + Dequantize-to-BF16 路径。

但当前本地 `vllm-ascend` 尚不能作为论文 FP4 Cache 的直接实现证据：

| 项目 | 论文目标 | 当前本地 vLLM-Ascend |
|---|---|---|
| Indexer Q | OCP MXFP4 | INT8 + 每 Head FP16 Scale |
| Indexer K | OCP MXFP4 | INT8 + FP16 Scale |
| Main KV | E2M1 + E4M3/16 | BF16 |

Ascend Indexer K Cache Spec 在 `vllm_ascend/models/deepseek_v41/indexer.py:73-86` 中声明为 INT8，`:94-112` 使用 `npu_dynamic_quant(..., int8)`；Indexer Q 的 INT8 量化见 `ops/triton/quantize_indexer_query.py:1-69`；Main `long_kv_cache` 在 `model.py:704-716` 中声明为 BF16。

因此本地 Ascend 版本主要实现了 V4.1 架构、压缩和跨层共享语义，但尚未把论文第 2.4.4 节的 FP4 Cache 格式完整落地到当前 DSA V4.1 路径。

### 问题 15 最终总结

1. DeepSeek-V4 已通过 QAT 让 Indexer Q/K 适应 OCP MXFP4，即每 32 个 E2M1 数值共享一个 E8M0 Scale；该格式同时优化 Indexer Cache 和长上下文检索计算。
2. DeepSeek-V4.1 将 QAT 扩展到 Main KV，但使用误差更小的 E2M1 + 每 16 通道一个 E4M3 Scale，并利用 RMSNorm 后有限的动态范围省略标准 NVFP4 的 Global Scale。
3. Main KV FP4 主要承担存储压缩；Top-K 选出后先反量化到 BF16，再执行普通 Sparse Attention，因此其 Cache 格式无需获得硬件原生 FP4 MatMul 支持。

最核心的设计是：

$$
\boxed{
\text{Indexer FP4 同时优化存储和计算；Main-KV FP4 主要优化存储，并在计算前恢复为高精度。}
}
$$

---

## 问题 16：E2M1 与 E4M3 Scale 的反量化及 Ascend 950 原生 MXFP4 MatMul

### 问题 16 原文

> 如果量化后是 E2M1 的 4-bit 格式，scale 是 E4M3 的，做反量化的时候，4 bit 可以和 8 bit 的 scale 直接相乘吗？昇腾 Ascend 950 硬件支持原生 MXFP4 矩阵乘法吗？对 FP4 MatMul 的支持，该格式为什么会受硬件指令约束呢？

### 问题 16 核心结论

1. **E2M1 的 4-bit 数据可以与 E4M3 的 8-bit Scale 相乘**，但这里说的是解码后的数值相乘：

   $$
   \widehat{x}_i
   =
   \operatorname{decode}_{\mathrm{E2M1}}(q_i)
   \times
   \operatorname{decode}_{\mathrm{E4M3}}(s_g).
   $$

   不能把 4-bit 和 8-bit 原始位模式当作普通整数直接相乘。软件实现通常先将二者解码或提升到 FP16、BF16、FP32，再完成乘法；专用硬件也可以把这一过程融合在反量化数据通路中。

2. **Ascend 950PR/950DT 支持原生 MXFP4 Cube 矩阵乘法能力**。官方 Ascend C `MX Mmad` 明确支持：

   $$
   \texttt{fp4x2\_e2m1\_t}
   \times
   \texttt{fp4x2\_e2m1\_t}
   \rightarrow
   \texttt{float}.
   $$

   但“支持 MXFP4”不等于支持任意 E2M1 数据、任意 Scale 格式、任意分组大小和任意布局。

3. DeepSeek-V4.1-Flash 的 Main KV 格式是 E2M1 加每 16 维一个 E4M3 Scale；标准 OCP MXFP4 则是 E2M1 加每 32 维一个 E8M0 Scale。论文通过在 Attention 前反量化 Main KV，绕开了对 E2M1 + E4M3/16 这一自定义 Cache 格式的原生 MatMul 指令要求。

### 16.1 E2M1 能否与 E4M3 Scale 直接相乘

需要把“直接相乘”区分为三个层次：

| 所谓“直接相乘” | 是否可以 | 实际含义 |
|---|---:|---|
| 数学上的数值相乘 | 可以 | 解码 E2M1 和 E4M3 后计算 $q_i s_g$ |
| 把原始 4-bit、8-bit 位模式当整数相乘 | 不可以 | 位模式不是实际浮点数值 |
| 直接交给某条 FP4×Scale 硬件指令 | 取决于硬件 | Scale 类型、Group Size、布局和 Operand 组合必须匹配指令 |

E2M1 的四个位包含 1 位符号、2 位指数和 1 位尾数。在这里使用的 E2M1 可表示的非负幅值为：

$$
\{0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6\},
$$

再由符号位决定正负。E4M3 Scale 同样需要按浮点格式解释，不能把其原始字节当作 `uint8` 数值。

例如某个 E2M1 元素解码为 $q_i=3$，该 16 维分组的 E4M3 Scale 解码为 $s_g=0.75$，则：

$$
\widehat{x}_i=3\times0.75=2.25.
$$

其数据流为：

```text
E2M1 nibble ──解码──> 3.0 ──┐
                             ├── FP32 multiply ──> 2.25 ──> BF16
E4M3 byte   ──解码──> 0.75 ─┘
```

输入存储位宽可以不同，因为真正执行乘法时，二者已经进入共同的内部计算精度。

### 16.2 vLLM Main KV 的实际反量化过程

Main KV 量化实现位于 `vllm/models/deepseek_v41/common/ops/fused_compress_quant_cache.py:421-461`：

- `:421-424` 说明格式是 E2M1，并且每 16 维共享一个 E4M3 Scale；
- `:446` 将 512 维重排为 `[32,16]`；
- `:447-449` 计算：

  $$
  s_g=\operatorname{E4M3}\left(
  \operatorname{clamp}\left(
  \frac{\max_i|x_i|}{6},2^{-9},448
  \right)\right);
  $$

- `:452` 计算 $x_i/s_g$；
- `:453-459` 把两个 E2M1 元素打包到一个字节；
- `:460-461` 将每组的 E4M3 Scale 保存为一个 8-bit 字节。

反量化实现位于 `vllm/models/deepseek_v41/common/ops/cache_utils.py:503-566`：

1. `:543-547` 从每个字节中拆出高、低两个 nibble；
2. `:548-553` 按 E2M1 规则重建数值；
3. `:555-561` 读取 E4M3 Scale，并提升到 FP32；
4. `:562-563` 在 FP32 中执行 E2M1 数值与 E4M3 Scale 的乘法；
5. `:566` 将反量化结果写入 BF16 Workspace。

因此代码中的真实流程是：

```text
Packed uint8
   │
   ├──拆出高/低 4-bit nibble
   ├──按照 E2M1 规则恢复并转成 FP32
   │
E4M3 Scale byte
   ├──按照 E4M3 bitcast
   └──转成 FP32
             │
             ▼
          FP32 multiply
             │
             ▼
          BF16 Workspace
```

### 16.3 DeepSeek Main KV 为什么不是标准 OCP MXFP4

OCP Microscaling 将一个分组定义为：

$$
\{X,[P_i]_{i=1}^{k}\},
$$

其中 $P_i$ 是低精度元素，$X$ 是整个分组共享的 Scale，实际值为：

$$
v_i=XP_i.
$$

所以 4-bit 元素和 8-bit Scale 具有不同位宽完全合理。但是标准 OCP MXFP4 和 DeepSeek Main KV 使用了不同的具体参数：

| 格式 | 元素格式 | 每元素位宽 | Group Size | Scale 格式 |
|---|---|---:|---:|---|
| OCP MXFP4 | E2M1 | 4 bit | 32 | E8M0，8 bit |
| DeepSeek-V4.1 Main KV | E2M1 | 4 bit | 16 | E4M3，8 bit |

上游 vLLM 也明确区分二者：

- `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:19-20` 将 Indexer 使用的 MXFP4 定义为每 32 个元素共享一个 UE8M0 Scale；
- `:65-75` 将 Scale 取为：

  $$
  s=2^{\lceil\log_2(\operatorname{amax}/6)\rceil};
  $$

- `vllm/models/deepseek_v41/common/ops/cache_utils.py:45-47` 则把 Main KV 定义为 512 个 E2M1，每 16 维一个 E4M3 Scale。

Main KV 每个 Entry 的存储量为：

$$
\frac{512\times4}{8}+\frac{512}{16}\times1
=256+32
=288\ \text{bytes}.
$$

因此，更准确的区分是：

```text
Indexer Q/K：OCP MXFP4，E2M1 + E8M0/32
Main KV：NVFP4-like 单层格式，E2M1 + E4M3/16
```

不能仅因为二者的元素都是 E2M1，就认为它们属于同一种硬件 Operand 格式。

### 16.4 Ascend 950 是否支持原生 MXFP4 MatMul

答案是：**支持，但必须限定在硬件和 CANN 已定义的 Operand 组合内。**

CANN 9.0 发布说明明确指出 Ascend 950PR 新增 FP8、MXFP8、MXFP4 支持，高阶 MatMul API 支持 MXFP4/8 低比特矩阵操作。

Ascend C `Mmad` 文档进一步说明：

- 该 Tensor API 支持 Ascend 950PR/Ascend 950DT；
- `MmadType::MX` 用来启用 MX Mmad；
- 原生支持 `fp4x2_e2m1_t × fp4x2_e2m1_t → float`；
- 左右矩阵的 Scale 要预先搬入 `L0ScaleA` 和 `L0ScaleB`；
- 结果在 L0C 中以 FP32 累加。

官方资料：

- [Ascend C Mmad API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_TensorAPI/mmad_compute/Mmad.md)
- [CANN 9.0 Release Notes](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/releasenote/release-notes.md)
- [OCP Microscaling Formats MX Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

这证明 Ascend 950 具有硬件 Cube 层面的 MXFP4 路径，而不只是完全展开成 BF16 后再执行 BF16 MatMul。

### 16.5 vLLM-Ascend A8W4 内核的真实计算路径

当前本地 vLLM-Ascend 的 `csrc/moe/grouped_matmul_situ_quant/README.md:3-4` 将相关算子描述为：

```text
GroupedMatmul(MXFP8 x MXFP4) + SiTU + dynamic MX quant
Ascend950PR (arch35) only
```

其 Host 侧数据类型在 `op_host/grouped_matmul_situ_quant_def.cpp` 中定义：

- `:11-17`：激活为 E4M3，激活 Scale 为 E8M0；
- `:23-29`：权重为 E2M1 FP4，使用 `FRACTAL_NZ_C0_16`；
- `:31-34`：权重 Scale 为 E8M0；
- `:62`：注册到 `ascend950`。

设备侧模板在 `op_kernel/gmsq_vcv_controller.h:30-32` 中使用：

```cpp
GMMSQWeightQuantVcvBasicBlock<
    fp8_e4m3fn_t,
    fp4x2_e2m1_t,
    fp8_e8m0_t,
    fp8_e8m0_t,
    ...
>
```

但这里有一个重要细节：官方 `MX Mmad` 类型组合包含 FP4×FP4、FP8×FP8，并没有列出 FP8×FP4。这个 A8W4 算子虽然以 FP4 保存权重，但进入 Cube 前会先把权重拓宽成 FP8。

`op_kernel/vendor/gmsq2/gmmsq_weight_quant_vec_compute.h:13-22` 明确写有：

```text
weight FP4->FP8 widen datapath
FP4-in MTE2 ring
FP8-out widen ring
```

`:343-355` 调用 `AntiQuantProcessNzMxA8W4`，`:363-383` 将 FP4 权重转换成高位 `xType`；这里的 `xType` 是 E4M3 FP8。随后 `gmmsq_weight_quant_cube_compute_tools.h:187-200` 才调用 `AscendC::Mmad`。L0C 在 `gmmsq_weight_quant_cube_compute.h:106-108` 中声明为 `LocalTensor<float>`，即 FP32 累加。

因此该内核的准确数据流是：

```text
Activation: MXFP8 E4M3 + E8M0 ───────────────┐
                                              │
Weight: MXFP4 E2M1 + E8M0                     │
          │                                   │
          └─ AIV FP4 → FP8 widen ─────────────┤
                                              ▼
                                     Cube MX FP8 × FP8
                                              │
                                              ▼
                                        FP32 L0C 累加
```

所以应当区分：

- Ascend 950 原生支持 FP4×FP4 MX Mmad；
- 当前这个 A8W4 MoE 算子不是原生的混合 FP8×FP4 Mmad；
- 它利用 FP4 降低权重存储和搬运量，再在 Vector 侧拓宽为 FP8 后交给 Cube。

### 16.6 FP4 MatMul 为什么受硬件指令约束

低精度矩阵乘法不是简单的 $C=A\times B$，而更接近：

$$
C_{mn}
=
\sum_g
s^A_{m,g}s^B_{g,n}
\left(
\sum_{k\in g}q^A_{m,k}q^B_{k,n}
\right).
$$

硬件除了读取量化元素 $q$，还必须知道：

1. 每多少个元素共享一个 Scale；
2. Scale 是 E8M0、E4M3、FP16 还是 FP32；
3. Scale 在内存中怎样排列；
4. Scale 对应 K 轴的哪个分组；
5. 两个 4-bit 元素以什么高低位顺序打包；
6. 矩阵采用 ND、NZ、ZN 还是其他分形布局；
7. 地址、K 轴和 Tile 的对齐要求；
8. 乘积和累加使用 FP16、BF16 还是 FP32。

标准 MXFP4 是：

```text
q0 ... q31  | E8M0 scale_0
q32 ... q63 | E8M0 scale_1
```

DeepSeek Main KV 则是：

```text
q0 ... q15  | E4M3 scale_0
q16 ... q31 | E4M3 scale_1
```

如果把 E4M3/16 的 Main KV 直接交给假设 E8M0/32 的 MX 指令：

- 硬件会按错误的格式解释 Scale 位模式；
- 第 16～31 维会关联到错误的 Scale；
- Scale 地址步长和分形布局也不匹配；
- 结果不是普通舍入误差，而是格式语义错误。

E8M0 Scale 没有尾数，本质上表示 2 的幂：

$$
s=2^e.
$$

硬件可以用指数调整式的数据通路处理。E4M3 带有 3 位尾数：

$$
s=(-1)^S2^{E-\mathrm{bias}}
\left(1+\frac{M}{8}\right),
$$

能表达 $0.75$、$1.375$ 等非 2 次幂 Scale，通常有更小的量化误差，但需要更一般的浮点缩放逻辑。

因此原生低比特 MatMul 的接口实际更像：

```text
MatMul<
    element_type,
    scale_type,
    group_size,
    packing_order,
    matrix_layout,
    accumulator_type,
    alignment
>
```

而不是只有一个笼统的 `FP4 MatMul` 标签。

### 16.7 反量化为什么能绕开原生格式限制

DeepSeek Main KV 的实际计算链路是：

```text
E2M1 + E4M3/16 Main KV Cache
               │
               ▼
      Vector/Triton 反量化
               │
               ▼
        BF16 Main KV Workspace
               │
               ▼
        BF16 Attention/MatMul
```

矩阵乘法硬件最终看到的是：

$$
Q_{\mathrm{BF16}}K_{\mathrm{BF16}}^\top,
$$

而不是 E2M1 + E4M3/16。因此硬件只需要支持 4-bit 解包、E4M3 解码、普通浮点乘法和 BF16 Attention，不必提供完全匹配 DeepSeek Cache Layout 的原生 FP4 MMA。

其代价包括：

- 反量化计算；
- BF16 临时 Workspace；
- 反量化后的数据带宽；
- FP4 数据不能贯穿 Main Attention MatMul。

但 Main Attention 只反量化 Indexer 选出的有限 Top-K Main KV，而不是每层展开全部百万级上下文，所以该成本仍然可控。

### 16.8 当前 vLLM-Ascend DeepSeek-V4.1 的实际数据类型

硬件具备 MXFP4 能力，不代表当前本地 DeepSeek-V4.1 实现已经采用论文中的 FP4 Main KV。

当前本地代码中：

- `vllm_ascend/models/deepseek_v41/indexer.py:73-86` 将 Indexer K Cache 声明为 INT8，并使用 FP16 Scale；
- 同文件 `:107-113` 调用 `npu_dynamic_quant` 生成 INT8 K 和 FP16 Scale；
- `vllm_ascend/ops/triton/quantize_indexer_query.py:28-37` 将 Indexer Q 量化为 INT8 + FP16 Scale；
- `vllm_ascend/models/deepseek_v41/model.py:704-716` 将压缩后的 `long_kv_cache` 声明为 BF16。

应当区分以下层次：

| 层次 | 当前结论 |
|---|---|
| Ascend 950 硬件/CANN | 支持 OCP 风格 MXFP4 和 FP4×FP4 MX Mmad |
| vLLM-Ascend 通用 MoE 路径 | 已有 Ascend 950 MXFP8/MXFP4 存储与计算优化 |
| 当前本地 DeepSeek-V4.1 Indexer | INT8 + FP16 Scale |
| 当前本地 DeepSeek-V4.1 long/Main KV | BF16 |
| 论文 Main KV | E2M1 + E4M3/16，Attention 前反量化 |

### 问题 16 最终总结

1. E2M1 的 4-bit 元素和 E4M3 的 8-bit Scale 可以按数值语义相乘；位宽不同不是障碍，但必须先正确解释二者的浮点编码，不能把原始 bit pattern 当普通整数相乘。
2. Ascend 950PR/950DT 原生支持 MX Mmad，包括 FP4 E2M1 × FP4 E2M1 → FP32，但该能力绑定于硬件定义的 Operand 类型、E8M0 Scale、Group Size、Scale Layout、分形布局和对齐条件。
3. DeepSeek-V4.1 Main KV 的 E2M1 + E4M3/16 不是标准 OCP MXFP4，不能因为元素都是 E2M1 就直接送入标准 MXFP4 Mmad。
4. 论文采用“FP4 Cache 存储、Attention 前反量化为 BF16”的路径，使 Cache Encoding 与 MatMul Operand Encoding 解耦，因此不需要硬件原生支持 E2M1 + E4M3/16 这一特定格式。
5. 当前本地 vLLM-Ascend 的 DeepSeek-V4.1 路径仍是 INT8 Indexer + BF16 long/Main KV；仓库中的 Ascend 950 MXFP4 MoE 内核可以证明硬件能力，但不是论文 FP4 Main KV 已落地的直接证据。

最核心的结论是：

$$
\boxed{
\text{“支持 FP4”必须同时说明元素格式、Scale 格式、Group Size、布局和实际矩阵指令。}
}
$$

---

## 问题 17：CSA2 训练中的 Shadow Indexer

### 问题 17 原文

> 在论文 3.1.2 Attention Sharing Training for CSA2 中的 Shadow indexers 具体是怎么回事？请帮忙解答。

### 问题 17 核心结论

先用一句话概括：

> **Shadow Indexer（影子索引器）不是一种新的注意力模式，也不是额外训练出来的“教师 Indexer”；它是为了让跨 Pipeline Stage 共享的同一个逻辑 Indexer 能在各个 Stage 本地执行，而放置在这些 Stage 上的轻量参数副本。**

这些副本在物理上分布于不同 Pipeline Stage，但在模型语义上仍然只代表**同一个共享 Indexer**：

- 只有一个 **logical owner（逻辑所有者）**负责优化器状态和 checkpoint；
- 各参与 Stage 上放置可执行的 **shadow replica（影子副本）**；
- 前向时，每个 Stage 使用本地副本计算，避免跨 Stage 远程调用模块；
- 反向时，将各副本产生的参数梯度聚合到 owner；
- owner 更新参数后，再将新参数同步给所有 shadow replica。

因此，它解决的是一个**分布式训练与 Pipeline Parallel 调度问题**，不是改变 Indexer 的数学功能。

论文 3.1.2 对这一设计的核心描述可以概括为：

> 每个参与的 Pipeline Stage 放置一个轻量、可执行的 Indexer 副本，同时保留单一逻辑 owner。Owner 负责优化和 checkpoint；参数同步与梯度聚合保证 shadow replicas 在训练过程中保持一致。这样可以保持原模型的共享语义，又不需要让 Pipeline scheduler 把共享层当作特殊的跨 Stage 执行单元。

论文位置：

- 本地论文：`D:\Code\vllm_notes\DeepSeekV41-Flash\DeepSeek-V4.1-Flash Pushing the Limits of KV Cache Compression.pdf`，第 3.1.2 节；
- arXiv 页面：<https://arxiv.org/abs/2609.19969>；
- DeepSeek-V4.1-Flash 模型说明：<https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md>。

---

### 17.1 为什么 CSA2 会遇到 Pipeline Parallel 问题

CSA2 中相邻若干 Decoder 层会共享稀疏注意力的某些中间结果。结合前面讨论过的三种层角色，可以抽象为：

| 层模式 | 是否构造 Main KV / Indexer K | 是否重新进行候选搜索或打分 | 是否取得新的 Top-K |
|---|---:|---:|---:|
| Full | 是 | 扫描完整可见范围并构造候选池 | 是 |
| Reindex | 复用已有缓存 | 对共享候选池重新打分 | 是 |
| Reuse | 复用 | 否 | 直接复用 |

这里的共享关系是按模型层组织的，而 Pipeline Parallel（PP）按“将连续层切到不同设备”组织。例如一个 40 层 Decoder 被切成：

```text
Pipeline Stage 0：Layer  0～ 9
Pipeline Stage 1：Layer 10～19
Pipeline Stage 2：Layer 20～29
Pipeline Stage 3：Layer 30～39
```

假设一个 attention-sharing group 横跨 Layer 28～33：

```text
Stage 2：L28  L29
              │
              ├── 一个共享 Indexer 组
              │
Stage 3：L30  L31  L32  L33
```

如果共享 Indexer 只作为普通子模块放在 Stage 2，那么 Stage 3 中需要执行 Indexer 的层就会遇到问题：

1. Stage 3 的激活需要发送回 Stage 2；
2. Stage 2 执行 Indexer 后，再把结果发送到 Stage 3；
3. 这种“回跳”通信破坏常规 PP 的顺向数据流；
4. 还会让 1F1B、激活重计算和反向调度变得复杂；
5. Pipeline scheduler 必须理解“某个逻辑模块被多个 Stage 调用”这种特殊关系。

也就是说，**模型语义上的跨层共享边界不一定与物理 PP 切分边界重合**。Shadow Indexer 就是为了解除这两个边界之间的耦合。

---

### 17.2 Shadow Indexer 的核心结构

设共享 Indexer 的逻辑参数为：

$$
\theta_{\mathrm{idx}}.
$$

如果这个共享组横跨三个 Pipeline Stage，那么物理上可以出现三个副本：

$$
\theta_{\mathrm{idx}}^{(0)},\quad
\theta_{\mathrm{idx}}^{(1)},\quad
\theta_{\mathrm{idx}}^{(2)}.
$$

它们并不是三个独立学习的 Indexer，而必须保持：

$$
\theta_{\mathrm{idx}}^{(0)}
=\theta_{\mathrm{idx}}^{(1)}
=\theta_{\mathrm{idx}}^{(2)}
=\theta_{\mathrm{idx}}.
$$

可以把结构理解为：

```text
                         一个逻辑 Indexer
                       θ_idx（logical owner）
                          /      |      \
                 参数同步       |       参数同步
                    /            |            \
        Stage 0 shadow    Stage 1 shadow    Stage 2 shadow
             θ⁽⁰⁾             θ⁽¹⁾             θ⁽²⁾
               │                │                │
          本地输入 X₀       本地输入 X₁       本地输入 X₂
               │                │                │
          本地执行前向      本地执行前向      本地执行前向
               \                |                /
                 各副本梯度聚合回 logical owner
```

因此，“shadow”的含义是：

- **物理上有副本**：使每个参与 Stage 都能够本地执行；
- **逻辑上没有多出模型参数**：这些副本共同表示同一个参数对象；
- **训练上必须同步**：不能让不同 Stage 上的副本各自独立更新。

这与张量并行中“同一参数被不同 rank 持有分片”有一点相似，但并不完全相同：

- Tensor Parallel 通常是把一个运算或参数张量切分；
- Shadow Indexer 更接近把同一个轻量模块复制到需要执行它的 PP Stage；
- 复制的目的是避免跨 Stage 调用，同时通过同步维持参数共享语义。

---

### 17.3 为什么需要单一 Logical Owner

如果每个 Stage 都把 shadow replica 当作普通、独立参数交给自己的 optimizer，副本会很快发生漂移。

假设初始参数都是：

$$
\theta^{(A)}=\theta^{(B)}=1.0.
$$

两个 Stage 分别得到梯度：

$$
g_A=0.2,\qquad g_B=0.3.
$$

如果学习率为 $0.1$，各 Stage 独立更新，那么：

$$
\theta^{(A)}_{\mathrm{new}}=1.0-0.1\times0.2=0.98,
$$

$$
\theta^{(B)}_{\mathrm{new}}=1.0-0.1\times0.3=0.97.
$$

此时“共享 Indexer”已经变成两个不同 Indexer，破坏模型定义。

正确做法是先聚合所有副本贡献的梯度：

$$
g_{\mathrm{owner}}=g_A+g_B=0.5,
$$

然后由 owner 只更新一次：

$$
\theta_{\mathrm{new}}
=1.0-0.1\times0.5
=0.95.
$$

最后再将 $0.95$ 同步到所有 shadow：

$$
\theta^{(A)}_{\mathrm{new}}
=\theta^{(B)}_{\mathrm{new}}
=0.95.
$$

这里用“求和”只是展示共享参数的链式法则。实际训练系统可能因为 loss reduction、数据并行或梯度累积策略而表现为 sum、average 或带缩放的归约；关键点不是具体缩放，而是：

> 所有 shadow 对同一个逻辑参数产生的梯度贡献，必须在同一次逻辑参数更新之前被正确合并。

单一 owner 通常承担：

- 保存 master parameter；
- 保存 optimizer state，例如 Adam 的一阶、二阶矩；
- 执行 optimizer step；
- 负责 checkpoint 保存与恢复；
- 将更新后的权重同步给 shadow replicas。

这样 checkpoint 中不需要把相同的共享参数当作多个独立模块保存，也不会在恢复后得到互相不一致的副本。

---

### 17.4 完整的前向、反向和参数更新流程

一次训练迭代可以抽象成以下过程。

#### 第 1 步：参数同步

在需要执行本轮前向之前，owner 将当前参数广播或同步给相关 Stage：

$$
\theta_{\mathrm{idx}}^{(s)}
\leftarrow
\theta_{\mathrm{owner}},
\qquad s\in\mathcal S,
$$

其中 $\mathcal S$ 是使用这个共享 Indexer 的 Stage 集合。

#### 第 2 步：各 Stage 本地执行

每个 Stage 使用自己的局部激活和本地 shadow：

$$
Y_s=I_{\theta_{\mathrm{idx}}^{(s)}}(X_s).
$$

Indexer 的典型计算仍然是原来的计算，例如可抽象为：

$$
Q^{\mathrm{idx}}
=W_Q^{\mathrm{idx}}H,
$$

$$
K_j^{\mathrm{idx}}
=\operatorname{RMSNorm}
\left(
W_K^{\mathrm{idx}}C_j
\right),
$$

并利用 Indexer Q、Indexer K 得到位置分数：

$$
s_j
=
\sum_h w_h
\left(q_h^{\mathrm{idx}}\right)^\top
k_{j,h}^{\mathrm{idx}}.
$$

Shadow 机制不会改变这些数学公式；它只改变“该公式在哪个 Pipeline Stage 上执行，以及参数如何保持一致”。

#### 第 3 步：各 Stage 反向计算局部梯度

对每个 shadow replica：

$$
g_s
=
\frac{\partial\mathcal L}
{\partial\theta_{\mathrm{idx}}^{(s)}}.
$$

每个 $g_s$ 都只是对应 Stage、对应 microbatch 的一部分梯度贡献。

#### 第 4 步：梯度聚合

将各 Stage 的梯度归约到逻辑 owner：

$$
g_{\mathrm{owner}}
=
\sum_{s\in\mathcal S}g_s.
$$

如果还使用 Data Parallel，那么还需与 DP 维度的梯度归约配合。概念上可以理解为：

```text
同一数据副本内部：
多个 PP Stage 的 shadow 梯度 → 合并为一个共享参数梯度

多个数据副本之间：
各 DP replica 的共享参数梯度 → 再做 DP 梯度归约
```

具体先后顺序和通信编排属于实现细节，论文没有给出完整 collective 时序。

#### 第 5 步：Owner 更新参数

$$
\theta_{\mathrm{owner}}
\leftarrow
\operatorname{OptimizerStep}
\left(
\theta_{\mathrm{owner}},g_{\mathrm{owner}}
\right).
$$

#### 第 6 步：将新参数重新同步给 Shadow

所有 shadow 再次取得 owner 的最新版本，为下一轮计算做准备。

整体流程如下：

```text
Owner 参数
   │
   ├──同步──> Shadow@Stage A ──前向/反向──> g_A ──┐
   ├──同步──> Shadow@Stage B ──前向/反向──> g_B ──┼──梯度聚合
   └──同步──> Shadow@Stage C ──前向/反向──> g_C ──┘
                                                   │
                                              Owner optimizer
                                                   │
                                              更新后的参数
                                                   │
                                              再同步给 shadows
```

---

### 17.5 Shadow Indexer 复制什么、不复制什么

理解 Shadow Indexer 最容易出现的误区，是把“Indexer 参数副本”和“长上下文运行时缓存副本”混为一谈。

#### 复制的主要对象

论文所说的 shadow replica，核心是让 Indexer 可以在目标 Stage 本地执行，因此复制的是该 Indexer 所需的**轻量可训练模块/参数**。从功能上看，可能涉及：

- $W_Q^{\mathrm{idx}}$；
- $W_K^{\mathrm{idx}}$；
- Indexer 的归一化参数；
- 多头得分组合权重 $w_h$；
- 以及执行 Indexer 必须依赖的其他少量参数。

但是，论文没有公开到“逐个张量列出 shadow 参数表”的程度，所以不能把上面的功能性列表误写成已由论文逐项确认的实现清单。

#### 不应理解为复制整个 Decoder 层

Shadow Indexer 不是：

- 复制整个 Transformer block；
- 复制完整 Attention 和 FFN/MoE；
- 为每个 Stage 增加一套独立训练的 Indexer；
- 增加一组新的 Full/Reindex/Reuse 模式。

#### 也不等于复制全部百万 Token KV Cache

需要严格区分三类对象：

| 对象 | 例子 | 解决机制 |
|---|---|---|
| 可训练参数 | $W_Q^{\mathrm{idx}}$、$W_K^{\mathrm{idx}}$、归一化参数 | Shadow replica、参数同步、梯度聚合 |
| 运行时中间数据 | Main KV、Indexer K、候选池、Top-K 索引 | Pipeline payload 扩展、跨 Stage 状态传递 |
| microbatch 生命周期状态 | 某个 microbatch 当前复用的候选与 Top-K | microbatch-aware state manager |

因此，“每个 Stage 有 Shadow Indexer”并不意味着每个 Stage 都完整复制超长上下文的所有 Indexer K 或 Main KV。参数复制和运行时数据传递是两个不同问题。

---

### 17.6 三项训练机制怎样配合

论文在 3.1.2 中讨论的不只是一个孤立的 shadow module，而是一组相互配合的训练机制。可以归纳为三部分。

#### 机制一：Shadow Indexers

解决：

> 跨 PP Stage 共享的 Indexer 在哪里执行？

答案是：在每个需要执行它的 Stage 上放一个本地可执行副本，同时保持单一逻辑 owner。

#### 机制二：Pipeline Payload Extensions

解决：

> 上游层产生并被下游共享的候选集、稀疏路由信息和相关中间状态，怎样随 Pipeline 激活一起传递？

普通 PP payload 主要传递 hidden states。CSA2 还需要把 attention sharing 所需的额外状态沿 Pipeline 传递。论文指出这些 payload 与 Context Parallel 的划分对齐，并为需要梯度的张量保持反向传播关系。

这里要特别区分：

- 浮点中间张量可以是 autograd graph 的一部分；
- Top-K 的整数索引是离散选择结果，不能简单地说“整数索引本身按普通链式法则传梯度”；
- 如果训练使用某种近似梯度、辅助目标或可微代理，那是另一个算法细节，不能仅凭“payload supports gradients”推断出来。

#### 机制三：Microbatch-aware Shared-state Manager

解决：

> Pipeline 同时在飞行多个 microbatch，而且还可能进行 activation recomputation，怎样保证每个 microbatch 读取到自己的共享状态？

共享状态需要按 microbatch 区分，并跨越：

- 首次 forward；
- activation recomputation；
- backward；
- 多个 consumer layer；

一直保留到最后一个消费者完成，才能释放。否则可能出现：

- microbatch A 覆盖 microbatch B 的候选池；
- 重计算时找不到第一次 forward 的共享状态；
- 状态释放过早；
- 不同层错误地复用其他 microbatch 的 Top-K。

三者的关系可以画成：

```text
Shadow Indexer
    └─解决“共享模块在哪里算、参数怎样保持同一个”

Pipeline Payload Extension
    └─解决“共享中间结果怎样跨 Stage 搬运”

Microbatch-aware State Manager
    └─解决“多个 microbatch 的共享结果怎样隔离与管理生命周期”
```

只实现其中一项，通常不足以完整支持 CSA2 的跨 Stage 训练。

---

### 17.7 与 Full、Reindex、Reuse 的关系

这两组概念处于不同层次：

| 概念 | 所属层次 | 回答的问题 |
|---|---|---|
| Full / Reindex / Reuse | 模型算法与层角色 | 当前层怎样获得 Main KV、候选与 Top-K？ |
| Shadow Indexer | 分布式训练系统 | 共享 Indexer 跨越 PP Stage 时，怎样在本地执行并维持参数共享？ |

所以不能把 Shadow Indexer 理解成“第四种模式”。

进一步说：

- **Full 层**通常是完整索引与候选生成的 source；
- **Reindex 层**需要对共享候选重新计算得分，因此可能需要执行共享 Indexer 的一部分逻辑；
- **Reuse 层**直接复用已有 Top-K，通常不需要再次执行 Indexer；
- 哪些 Stage 实际需要 shadow，取决于这个共享组的图结构和训练实现。

因此，“每一个 Reuse 层也有一个 Shadow Indexer”不是一个可以直接成立的说法。论文的表述是给**participating stages**放置可执行副本，不等于给所有层无条件复制。

---

### 17.8 为什么不把共享层限制在同一个 Pipeline Stage

一种看似简单的方案是：强制一个 attention-sharing group 永远不跨 PP 边界。这样似乎就不需要 shadow。

但这会带来几个问题：

1. **限制 Pipeline 切分自由度**  
   PP 切分必须迁就共享组边界，不能按每层计算量、显存或 MoE 负载灵活均衡。

2. **可能造成 Stage 负载不均**  
   如果共享组大小与理想 PP 分层数不整除，一些 Stage 会多放几层，形成 pipeline bubble。

3. **限制不同并行规模**  
   换一个 PP degree，就可能要重新设计层划分或共享组。

4. **使模型语义绑死在训练拓扑上**  
   一个纯模型结构选择不应过度依赖某次训练使用多少设备、多少 Pipeline Stage。

Shadow Indexer 的价值就是：

$$
\boxed{
\text{模型的逻辑共享边界}
\quad\text{与}\quad
\text{Pipeline 的物理切分边界}
\text{解耦}
}
$$

这样 PP scheduler 仍可以把每个 Stage 看作一个常规的本地执行单元，而不用为“远程调用共享模块”加入特殊调度语义。

---

### 17.9 vLLM 与 vLLM-Ascend 代码能证明什么

这里必须区分：

- 论文第 3.1.2 节描述的是**训练系统**；
- 当前 vLLM 和 vLLM-Ascend 中能看到的 DeepSeek-V4.1 支持主要是**推理执行与缓存共享**；
- 推理代码可以帮助理解 source/consumer、共享 Top-K 和共享 KV，但不能据此宣称已经实现了训练用 Shadow Indexer。

#### 17.9.1 上游 vLLM

文件：

`D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py`

关键代码关系如下：

1. **第 303～309 行**  
   Compressor/cache 只在 KV source 层创建，Indexer 只在 index source 层创建，consumer 层复用最近 source 的结果。

2. **第 322～348 行**  
   按层解析 KV source 和 index source，建立层间共享拓扑。

3. **第 431～472 行**  
   只有 `is_index_source` 的层实际持有 Indexer；非 owner/source 层通过共享关系使用已有缓存或索引结果。

4. **第 448～458 行**  
   如果 Indexer K 的 source 不在同一 rank，当前代码会显式报错，错误信息指出 PP 不能切在 V4.1 的 KV-sharing group 内。

5. **第 586～600 行**  
   对 compressed KV 的跨 rank source 也有相应限制。

6. **第 1227～1240 行**  
   Indexer 说明中明确：只有 index source 层执行索引，consumer 复用共享 Top-K buffer。

这些代码说明当前推理实现具有“source 计算、consumer 复用”的逻辑，但还把共享组限制在相同 PP rank。恰恰因为它没有论文训练系统中的 shadow/payload/state-manager 机制，所以它不能直接跨 PP Stage 复用这个共享组。

换句话说：

> vLLM 中的“跨 rank 不支持”是 Shadow Indexer 所要解决的问题的一种具体体现，但不能反过来说 vLLM 已经实现了 Shadow Indexer。

#### 17.9.2 vLLM-Ascend

模型拓扑文件：

`D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py`

关键位置：

- **第 439～477 行**：定义层角色及 source/consumer 拓扑；
- **第 498～548 行**：构造各层的执行计划；
- **第 704～721 行**：只有 KV source 层创建相应 cache/compressor；
- **第 723～733 行**：只有 index source 层创建 `DeepseekV41Indexer`；
- **第 480～491 行**：定义共享 attention state，例如 Top-K 与候选信息；
- **第 997～1003 行**：将同一个共享状态绑定到组内相关层。

注意力/索引执行文件：

`D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py`

关键位置：

- **第 440～446 行**：非 index source 层直接返回共享状态中的 `topk_indices`；
- **第 448～472 行**：index source 层执行选择，并把 Top-K 或候选结果发布到共享状态。

这同样是在说明**推理期 buffer/state sharing**：

```text
Index source：计算并写入 topk_indices
Consumer：直接读取并复用 topk_indices
```

它不等同于训练期 Shadow Indexer，原因是上述代码路径中没有展示：

- 多个 PP Stage 上的可训练 Indexer 参数副本；
- logical owner；
- shadow 参数广播；
- shadow 梯度聚合；
- owner-only optimizer state；
- shadow-aware checkpoint。

所以从代码能得到的严谨结论是：

> 当前代码实现验证了 CSA2 的 source/consumer 和共享运行时状态思想；论文所述 Shadow Indexer 则是解决跨 PP Stage 训练的额外机制，两者相关但不是同一件事。

---

### 17.10 论文明确内容与未披露细节

为了避免把合理推测写成确定事实，可以将信息边界列成下表：

| 内容 | 论文是否明确 |
|---|---|
| 每个参与 Stage 放置轻量可执行 replica | 明确 |
| 共享参数只有一个 logical owner | 明确 |
| owner 负责优化和 checkpoint | 明确 |
| 参数同步保持 shadow 一致 | 明确 |
| shadow 梯度需要聚合 | 明确 |
| Pipeline payload 扩展携带共享状态 | 明确 |
| 与 Context Parallel 分区对齐 | 明确 |
| microbatch 状态跨 forward/recompute/backward 保留到最终消费者 | 明确 |
| 使用哪一种 collective（broadcast、all-reduce、reduce 等） | 未详细披露 |
| 每次同步发生在每个 microbatch、每个 iteration，还是分桶重叠通信 | 未详细披露 |
| Shadow Indexer 精确包含哪些参数张量 | 未逐项披露 |
| 参数和梯度通信的实际字节数与性能开销 | 未给出完整明细 |
| Top-K 离散选择具体采用何种梯度近似 | 不能仅由 shadow 机制推出 |

因此，如果进一步讨论实现方案，比较合理的工程推演是：

1. owner 持有唯一 optimizer state；
2. shadows 注册为不被本地 optimizer 独立更新的参数副本；
3. forward 前或参数更新后同步 owner 权重；
4. backward 后把所有 shadow 梯度归约到 owner；
5. checkpoint 只保存一次逻辑参数；
6. 恢复后由 owner 再初始化 shadows。

但这些属于基于论文语义的实现推演，不能假装成论文已经公开的逐行训练代码。

---

### 问题 17 最终总结

1. **Shadow Indexer 是什么？**  
   它是同一个共享 Indexer 在不同 Pipeline Stage 上的本地可执行参数副本。

2. **为什么需要它？**  
   CSA2 的 attention-sharing group 可能跨 PP 边界；如果只有一个物理 Indexer，其他 Stage 就要远程回调该模块，破坏普通 Pipeline 调度。

3. **它是不是多个独立 Indexer？**  
   不是。它们共享一个逻辑参数身份，必须保持数值一致。

4. **谁更新参数？**  
   单一 logical owner 负责 optimizer 和 checkpoint；各 shadow 的梯度聚合到 owner 后只更新一次。

5. **它是否复制完整长上下文 KV？**  
   不能这样理解。Shadow 主要解决可训练 Indexer 模块的本地执行；Main KV、Indexer K、候选池和 Top-K 属于运行时状态，由 payload/state-management 机制处理。

6. **它与 Full/Reindex/Reuse 是什么关系？**  
   Full/Reindex/Reuse 是模型层的索引复用模式；Shadow Indexer 是跨 Pipeline Stage 的分布式训练实现，两者不是同一分类维度。

7. **当前 vLLM/vLLM-Ascend 是否已经实现论文的 Shadow Indexer？**  
   从当前代码能确认的是推理期 source/consumer 与共享 Top-K 状态；上游 vLLM 还明确限制共享组跨 PP rank。因此不能把这些推理代码称为训练期 Shadow Indexer 实现。

最核心的概念可以写成：

$$
\boxed{
\text{Shadow Indexer}
=
\text{同一逻辑共享 Indexer}
\text{在多个 PP Stage 上的物理可执行副本}
}
$$

它以少量参数同步和梯度聚合为代价，换来了：

$$
\boxed{
\text{保持模型共享语义}
+
\text{保持常规 Pipeline 调度}
+
\text{允许共享组跨越 PP 边界}
}
$$

---

## 问题 18：Inference System 中的融合算子及其对算子流的简化

### 问题 18 原文

> 请你结合 `D:\Code\vllm-ascend\vllm-ascend` 和 `D:\Code\vllm-github` 的代码内容，以及论文
> `D:\Code\vllm_notes\DeepSeekV41-Flash\DeepSeek-V4.1-Flash Pushing the Limits of KV Cache Compression.pdf`
> 的内容，回答以下问题。你在回答问题的时候可以辅助进行一些网络知识的搜索，请尽量详细回答，并配合适当举例和图表。
>
> 论文 3.2. Inference System 章节中提到的一系列融合算子，包括
> fused-RoPE-attention-RoPE-cast kernel in FlashMLA、Mega-Gate、Mega-mHC、
> Mega-MoE kernels in DeepGEMM、the kernels in TileKernels、TopK kernel in
> DeepSelect，具体是做什么的？它们的存在，又是怎样使得概念上复杂的架构，
> 在推理算子流上变得简单了呢？

### 问题 18 核心结论

这些名字并不是六个彼此孤立的“数学模块”，而是把 DeepSeek-V4.1-Flash 中原本需要很多
GPU/NPU kernel、许多中间张量和多次通信才能完成的逻辑，重新封装成少数几个大粒度执行单元。

需要特别区分两个层面：

1. **模型/算法层仍然复杂**：CSA2、Local SWA、Compressed Global KV、Indexer、
   mHC 多残差流、Engram、MoE 路由与 Expert Parallel 都没有消失。
2. **设备执行层被压缩**：RoPE、量化、Top-K、路由、激活、通信、多个 GEMM 之间的边界
   被融合或流水化，从调度器看来只剩少数几个“大算子”。

可以用下面的对应关系概括：

| 架构功能 | 主要融合实现 | 核心作用 |
|---|---|---|
| CSA2 稀疏注意力 | FlashMLA + DeepSelect | 选择少量 Global KV，并与 Local SWA 一起计算注意力 |
| MoE 路由 | Mega-Gate | 将 router GEMM、打分修正和 Expert Top-K 合并 |
| 多流残差连接 | Mega-mHC | 将相邻层的 mHC post/pre、归一化和混合系数计算合并 |
| Expert Parallel MoE | Mega-MoE | 将 dispatch、专家计算、激活、量化和 combine 流水化 |
| 其余细粒度算子 | TileKernels | 提供面向具体 shape/layout 的高性能 tile kernel |
| Indexer 位置筛选 | DeepSelect TopK | 从 Indexer logits 中选出压缩历史位置 |

论文所说“多数 Reuse Mode 层在 prefill 约 15 个 kernel、decode 约 11 个 kernel”，
应该理解为论文生产栈中的端到端算子数量，而不是说模型只有 11～15 个数学步骤，
也不能机械外推到所有 vLLM 后端。

---

### 18.1 为什么 DeepSeek-V4.1-Flash 特别依赖融合算子

如果按概念公式逐项实现，一个 Decoder 层可能包含：

```text
读取共享 Top-K
  → 生成 Main Q
  → Q-RoPE
  → 读取 Local SWA KV
  → 根据 Top-K gather Compressed Global KV
  → Local/Global 两路 attention logits
  → 统一 online softmax
  → attention output
  → inverse/output RoPE
  → cast/quantize
  → output projection
  → mHC post
  → 下一层 mHC pre
  → MoE router GEMM
  → score correction / mask
  → Expert Top-K
  → EP dispatch
  → Expert GEMM1
  → SwiGLU
  → activation quantization
  → Expert GEMM2
  → weighted combine
  → EP combine
```

如果每个箭头都对应一次独立 kernel launch，主要成本不只来自 FLOPs，还包括：

- 每个 kernel 的 launch 与同步开销；
- 中间结果反复写回、读出 HBM；
- 张量 layout 转换、转置和重新打包；
- 小 batch decode 时大量小算子不能充分占满设备；
- EP 通信与本地专家计算串行，产生空泡；
- 多个框架算子之间难以进行跨算子的流水化。

融合的目的就是尽量把：

$$
\text{中间张量写 HBM}
\rightarrow
\text{再次从 HBM 读取}
$$

变成：

$$
\text{寄存器/片上缓存中的生产者结果}
\rightarrow
\text{同一 kernel 内的消费者}
$$

所以它优化的是“执行图的物理实现”，并没有修改模型的语义。

---

### 18.2 FlashMLA 的 fused-RoPE-attention-RoPE-cast

#### 18.2.1 它融合了什么

这个 kernel 可抽象为：

$$
Q
\xrightarrow{\mathrm{RoPE}}
Q_{\mathrm{rot}}
\xrightarrow{\mathrm{SparseAttention}
(\mathrm{Local\ SWA},\,\mathrm{Selected\ Global\ KV})}
O_{\mathrm{rot}}
\xrightarrow{\mathrm{inverse/output\ RoPE}}
O
\xrightarrow{\mathrm{cast/quantize}}
O_{\mathrm{lowp}}.
$$

其中稀疏注意力同时消费：

- 当前层保存的 **Local SWA KV**；
- Indexer Top-K 选中的 **Compressed Global KV**；
- 当前 query 及其位置、有效长度和块表等元数据。

“attention 前后各有一次 RoPE”不是重复做同一件事。第一处将 query 放进与缓存 key
一致的旋转位置空间；第二处对应 V4.1 MLA/输出投影所要求的输出空间变换，可视为
inverse/conjugate RoPE。融合后，旋转前后的大张量不必单独落到 HBM。

#### 18.2.2 为什么最后还要 cast

attention 的 softmax 和累加往往需要较高的内部精度，但输出投影可能接收 FP8 等低精度输入。
因此 kernel 可以：

1. 用较高精度完成 logits、online softmax 和 value 累加；
2. 在输出端完成旋转；
3. 直接生成后续 `wo_a/Wv` 能消费的低精度及目标 layout。

这样避免：

```text
attention 输出
 → 单独 inverse RoPE kernel
 → 单独 cast/quant kernel
 → 单独 layout/repack kernel
 → output projection
```

而变成：

```text
fused sparse attention（含前后 RoPE 和 cast）
 → output projection
```

#### 18.2.3 vLLM 代码证据

在 NVIDIA DeepSeek-V4.1 路径中：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\sparse_mla.py:235-243`
  配置 FlashMLA Mega Attention，并明确 `enable_q_norm=False`；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\flash_mla_mega_attn.py:3-15`
  说明该实现面向 DeepSeek-V4/V4.1 的 fused attention 路径；
- 同文件 `:361-378` 是 decode 调用路径；
- 同文件 `:465-478` 是 prefill 调用路径；
- 同文件 `:201-228` 展示融合输出被后续 `wo_a` 路径直接消费。

因此不能把这个 fused kernel 简化成“只把 RoPE 和普通 attention 写在一个函数里”；
关键还包括 Local/Global KV 联合注意力、输出旋转、低精度输出以及 layout 对接。

---

### 18.3 DeepGEMM 的 Mega-Gate

#### 18.3.1 普通 MoE Gate 的逻辑

普通 MoE 路由大致为：

$$
R=XW_{\mathrm{router}},
$$

然后依次进行：

1. 对 router logits 计算 score；
2. 加 correction bias 或其他路由修正；
3. 对图像 token、hash 路由或特定 expert 施加 mask；
4. 执行 Expert Top-K；
5. 对被选 expert 的权重归一化并乘 routed scaling factor；
6. 生成 `topk_ids` 和 `topk_weights`。

这些步骤若拆成许多小 kernel，在 decode 的小 token batch 上尤其昂贵。

#### 18.3.2 Mega-Gate 做什么

Mega-Gate 将 router GEMM 和大部分后处理合并为一个更大的路由执行单元：

$$
X
\longrightarrow
\boxed{
\text{Router GEMM}
+\text{score/correction}
+\text{mask}
+\text{Expert Top-K}
+\text{normalize/scale}
}
\longrightarrow
(\mathrm{topk\_ids},\mathrm{topk\_weights}).
$$

它的输出不是 attention 历史位置，而是 MoE 专家编号及相应权重。

本地代码证据：

- `D:\Code\vllm-github\vllm\utils\deep_gemm.py:489-535`
  封装 Mega-Gate 的调用接口和输出；
- `D:\Code\vllm-github\vllm\models\deepseek_v4\nvidia\model.py:1033-1091`
  展示模型路由侧如何选择 Mega-Gate 或普通 fallback。

小 batch、特定 shape 或不满足内核约束时仍可能退回普通路由路径，所以
“存在 Mega-Gate”不等于每个请求、每个阶段都必然调用它。

---

### 18.4 DeepGEMM 的 Mega-mHC

#### 18.4.1 mHC 为什么容易产生碎片化小算子

mHC（manifold-constrained hyper-connections）不再只维护一条传统 residual stream，
而是维护多条残差流，并使用受约束的混合矩阵在主分支和残差分支之间进行变换。
从算子图看，容易出现：

- 多流 reshape/split；
- RMSNorm；
- 混合系数生成；
- Sinkhorn 或约束归一化；
- 当前层 post mixing；
- 下一层 pre mixing；
- 多个 elementwise add/mul。

这些操作算术量未必很大，但 kernel 很碎。

#### 18.4.2 Mega-mHC 的关键思想

相邻层之间的：

$$
\mathrm{mHC\ post}^{(l)}
\quad\text{和}\quad
\mathrm{mHC\ pre}^{(l+1)}
$$

存在天然的生产者—消费者关系。因此 Mega-mHC 可以把当前层 post、下一层 pre、
混合系数生成、Sinkhorn 和 RMSNorm 等合在一个大 kernel 中：

```text
Layer l 主分支输出 + 4 条 residual streams
       │
       ▼
Mega-mHC:
  current post mixing
  + next pre coefficients
  + Sinkhorn/constraint
  + RMSNorm
       │
       ├── 下一层主分支输入
       └── 更新后的多条 residual streams
```

这是一种“跨层边界融合”，不只是单个线性层加一个激活函数。

本地代码证据：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\ops\mega_mhc.py:47-102`
  展示 Mega-mHC 的输入、约束与融合调用；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\ops\mhc.py:283-321`
  展示 Mega-mHC 的条件分派与 fallback。

当前实现带有硬件和 shape 前提，例如特定 NVIDIA 架构、`hc_mult=4`、
hidden size 对齐等；条件不满足时，模型语义不变，但执行会退回多个普通算子。

---

### 18.5 DeepGEMM 的 Mega-MoE

#### 18.5.1 普通 Expert Parallel MoE 的执行链

MoE 的完整路径不是只有两个专家 GEMM，而是：

```text
topk_ids/topk_weights
 → 统计目标 rank/expert
 → EP dispatch / all-to-all
 → expert GEMM1
 → SwiGLU
 → activation quantization
 → expert GEMM2
 → 按 gate weight 加权
 → EP combine / all-to-all
 → 恢复原 token 顺序
```

若严格串行，通信和计算之间会有明显空泡。

#### 18.5.2 Mega-MoE 做什么

Mega-MoE 将上述步骤放到一个持久化或深度流水化的执行框架中，使不同 token tile、
expert tile、通信 tile 可以重叠：

```text
Tile A: dispatch ── GEMM1 ── SwiGLU/Quant ── GEMM2 ── combine
Tile B:          dispatch ── GEMM1 ── SwiGLU/Quant ── GEMM2 ── combine
Tile C:                   dispatch ── GEMM1 ── ...
```

因此“Mega”不只是把源代码函数拼成一个函数，而是尝试做到：

$$
T_{\mathrm{total}}
\approx
\max(T_{\mathrm{communication}},T_{\mathrm{compute}})
+T_{\mathrm{pipeline\ overhead}},
$$

而不是完全串行的：

$$
T_{\mathrm{total}}
\approx
T_{\mathrm{communication}}+T_{\mathrm{compute}}.
$$

本地 NVIDIA 模型调用可见：

- `D:\Code\vllm-github\vllm\models\deepseek_v4\nvidia\model.py:734-775`，
  该处将路由结果交给 Mega-MoE 路径，完成专家并行计算和组合。

需要注意，Mega-MoE 仍然执行相同的专家网络：

$$
Y_e=W_{2,e}\bigl(\mathrm{SwiGLU}(W_{1,e}X_e)\bigr),
$$

融合改变的是 token 搬运、通信和计算调度方式，不是把多个 expert 合成一个 expert。

---

### 18.6 TileKernels 是什么

TileKernels 更应被理解为一套面向 tile 编程和特定硬件布局的 kernel 库，而不是一个单独算子。
它可承载或辅助实现：

- MoE routing 的若干步骤；
- 低精度量化和 layout 转换；
- fused SwiGLU + cast；
- Engram 相关操作；
- mHC 的 pre/post 变换；
- RoPE 等细粒度、但高频的计算。

“tile”表示把张量划分为适配硬件矩阵单元、shared memory/片上缓存和线程组织方式的块。
库可以针对固定的 hidden size、head dimension、量化 block size 和 batch 区间，
提供比通用框架组合算子更合适的 memory layout 与调度。

vLLM 中可以看到类似的 TileLang mHC 实现：

- `D:\Code\vllm-github\vllm\model_executor\kernels\mhc\tilelang.py:74-120`
  定义 mHC 所需 tile kernel 的计算结构；
- 同文件 `:791-811` 给出对应调用/封装路径。

所以论文中 “the kernels in TileKernels” 不是在指一个叫 `TileKernels()` 的万能 kernel，
而是指若干由该库提供、填补融合图中剩余热点的专用 kernel。

---

### 18.7 DeepSelect 的 TopK Kernel

#### 18.7.1 输入和输出

DeepSelect TopK 位于 attention Indexer 路径：

$$
\text{Indexer logits over compressed history}
\xrightarrow{\mathrm{TopK}}
\text{selected compressed positions}.
$$

假设压缩 Global KV 一共有 100,000 个候选位置，Indexer 为当前 query 生成：

$$
s=[s_0,s_1,\ldots,s_{99999}].
$$

TopK kernel 选择最大的 512 个 score：

$$
\mathcal G_t
=
\operatorname{TopKIndices}(s,512),
$$

输出例如：

```text
[17, 205, 1308, ..., 99871]
```

后续 sparse attention 才根据这些位置读取 Compressed Global KV。

#### 18.7.2 它不负责什么

DeepSelect TopK：

- 不生成 Indexer Q/K；
- 不计算完整 attention；
- 不选择 MoE experts；
- 不生成 Local SWA 窗口；
- 只负责从 Indexer score 中高效选出历史位置。

vLLM 的统一入口及后端分派可见：

- `D:\Code\vllm-github\vllm\model_executor\layers\indexer_topk.py:74-116`；
- 同文件 `:302-317`。

因此，DeepSelect TopK 和 Mega-Gate 内部 Expert Top-K 虽然都叫 Top-K，
选择对象完全不同：

| Top-K | 候选对象 | 输出 |
|---|---|---|
| DeepSelect TopK | 压缩后的历史 token/KV 位置 | Global KV position indices |
| Mega-Gate Expert TopK | MoE experts | expert ids 和 gate weights |

---

### 18.8 这些算子如何把复杂架构变成简单算子流

“简单”主要来自四个层次。

#### 第一层：Reuse Mode 先删掉不必要工作

CSA2 的 Reuse 层直接复用共享 Top-K，因而无需每层都执行：

$$
\text{Compressor}
+\text{Indexer Q/K}
+\text{全量 Indexer score}
+\text{Top-K}.
$$

这是算法/架构层面的删减。

#### 第二层：大 kernel 合并相邻计算

例如：

$$
\underbrace{
\mathrm{RoPE}
+\mathrm{Attention}
+\mathrm{inverse\ RoPE}
+\mathrm{cast}
}_{\text{FlashMLA fused kernel}},
$$

以及：

$$
\underbrace{
\mathrm{router\ GEMM}
+\mathrm{score\ correction}
+\mathrm{Expert\ TopK}
}_{\text{Mega-Gate}}.
$$

这是 kernel launch 数量的删减。

#### 第三层：让生产者和消费者共用 layout

如果前一个 kernel 的输出就是下一个 kernel 要求的精度和排布，就不再需要：

- transpose；
- contiguous copy；
- dequant/requant；
- 单独的 cast；
- 中间 tensor materialization。

这是数据搬运和 HBM 流量的删减。

#### 第四层：持久化调度覆盖通信空泡

Mega-MoE 将 EP 通信与专家计算交错执行，使设备无需在每个阶段全局等待。
这是同步和通信空泡的删减。

综合起来：

$$
\boxed{
\text{复杂模型语义}
\neq
\text{必须拥有复杂、碎片化的 kernel DAG}
}
$$

只要输入输出契约和布局被统一，内部复杂逻辑可以由少数 fused/persistent kernels 承担。

---

### 18.9 Prefill 与 Decode 的典型数据流

#### 18.9.1 Reuse Mode 的概念流程

```text
共享的 selected_global_indices
              │
当前 hidden ─┼─→ Main Q
              │
当前层 SWA KV cache ───────────┐
Compressed Global KV + indices ├─→ fused FlashMLA attention
                               │     (Q-RoPE + Local/Global attention
                               │      + output RoPE + cast)
                               └─→ output projection
                                      │
                                      ▼
                                  Mega-mHC
                                      │
                                      ▼
                                  Mega-Gate
                                      │
                                      ▼
                                  Mega-MoE
```

Indexer source 层会在这条链之前额外执行 Indexer score 和 DeepSelect TopK；
Reuse 层直接接收已有 indices，所以算子数更少。

#### 18.9.2 为什么 decode 往往比 prefill kernel 更少

decode 每步通常只处理少量新 token，历史 KV 已经存在：

- 不必重新为整段 prompt 写入所有 KV；
- 很多元数据、共享索引状态和 buffer 可以复用；
- 不需要 prefill 中的大块 cache populate、分块元数据构造和某些 gather/reorder；
- persistent kernel 更容易围绕固定 decode 数据流工作。

因此论文给出的多数 Reuse 层约 15 个 prefill kernels、11 个 decode kernels 是合理的。
但这属于特定软件版本、硬件和生产配置的统计，而不是模型定义中的恒等式。

---

### 18.10 vLLM 与 vLLM-Ascend 中的对应实现

#### 18.10.1 NVIDIA 路径

论文列举的 FlashMLA、DeepGEMM、TileKernels 和 DeepSelect 主要对应 NVIDIA 生产栈。
在本地 `D:\Code\vllm-github` 中，可以看到这些能力通过模型专用后端和 kernel wrapper 接入。

#### 18.10.2 Ascend 路径不是逐名字照搬

Ascend 后端实现相同模型语义，但算子边界并不必然与 NVIDIA 一致：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:480-528`
  调用 `npu_sparse_flash_mla`，一次消费 Local KV、Compressed KV 和 Top-K indices，
  对应 Local/Global 联合稀疏注意力；
- 同文件 `:554-560` 显示当前路径中的 inverse RoPE 仍是独立步骤，
  因此其融合边界与 NVIDIA 的 fused-RoPE-attention-RoPE-cast 不完全相同；
- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\indexer.py:193-228`
  使用 `npu_quant_lightning_indexer_v3` 完成 Ascend 侧量化 Indexer 计算；
- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\ops\fused_moe\moe_comm_method.py:451-530`
  展示 CANN MegaMoE 的接入和执行分支。

所以更准确的表述是：

> NVIDIA 与 Ascend 后端实现相同的模型数据依赖，但会根据各自硬件指令、
> 通信库、片上存储和编程模型选择不同的融合边界。

不能仅凭论文里的 NVIDIA kernel 名称，就断言 Ascend 也以同名、同边界、同 kernel 数实现。

---

### 18.11 容易混淆的几个 Top-K

DeepSeek-V4.1-Flash 的执行链中至少存在三类容易混淆的“选择”：

| 名称 | 选择什么 | 位于哪里 | 典型输出 |
|---|---|---|---|
| Hierarchical Indexer 候选块选择 | 历史 KV 块 | attention Indexer | 候选位置集合 |
| DeepSelect TopK | 压缩历史位置 | sparse attention 前 | 例如 512 个 Global KV indices |
| Mega-Gate Expert TopK | MoE experts | FFN/MoE 前 | expert ids + weights |

它们都可能使用 Top-K 算法，但不能共用语义：

$$
\text{历史位置 Top-K}
\neq
\text{专家 Top-K}.
$$

前者回答“当前 query 应该读哪些远程历史 KV”，后者回答“当前 token 应该送入哪些专家”。

---

### 问题 18 最终总结

1. **FlashMLA fused-RoPE-attention-RoPE-cast**  
   将 query RoPE、Local SWA + Selected Global KV 的稀疏注意力、输出 inverse/conjugate
   RoPE 和低精度 cast/layout 对接合并，减少中间张量、launch 和 HBM 往返。

2. **Mega-Gate**  
   将 MoE router GEMM、score/correction/mask、Expert Top-K、权重归一化与 scaling 合并，
   直接产出 `topk_ids/topk_weights`。

3. **Mega-mHC**  
   将当前层 mHC post 与下一层 mHC pre、混合系数、Sinkhorn、RMSNorm 等跨层融合，
   避免多残差流产生大量小算子。

4. **Mega-MoE**  
   把 EP dispatch、专家 GEMM1、SwiGLU、量化、GEMM2、加权和 EP combine 组织成融合或
   持久化流水线，以重叠通信和计算。

5. **TileKernels**  
   是一组面向特定 shape、layout 和硬件 tile 的专用 kernels，不是单一算子；
   它负责补齐量化、激活、mHC、Engram、RoPE 等热点。

6. **DeepSelect TopK**  
   从 Indexer logits 中选出需要读取的压缩历史位置；它既不计算 attention，
   也不是 MoE 的 Expert Top-K。

7. **为什么算子流变简单**  
   Reuse Mode 先删除重复的 Indexer 工作；融合再减少 kernel launch；
   对齐 layout 消除中间 HBM 搬运；persistent/Mega kernel 进一步重叠通信与计算。

最终可概括为：

$$
\boxed{
\begin{aligned}
\text{模型复杂度} &:\quad
\text{CSA2}+\text{mHC}+\text{Engram}+\text{MoE},\\
\text{运行时复杂度} &:\quad
\text{少量大粒度、布局相容、可流水化的 kernels}.
\end{aligned}
}
$$

这些融合算子的真正价值，不是让架构逻辑消失，而是把复杂性封装在经过硬件优化的 kernel
内部，使上层推理引擎看到一条更短、更稳定、更少同步点的执行链。

---

## 问题 19：SWA Bounded Replay 与在 Encoder 处结束 Prefill

### 问题 19 原文

> 请你结合 `D:\Code\vllm-ascend\vllm-ascend` 和 `D:\Code\vllm-github` 的代码内容，以及论文
> `D:\Code\vllm_notes\DeepSeekV41-Flash\DeepSeek-V4.1-Flash Pushing the Limits of KV Cache Compression.pdf`
> 的内容，回答以下问题。你在回答问题的时候可以辅助进行一些网络知识的搜索，请尽量详细回答，
> 并配合适当举例和图表。
>
> 论文 3.2.2. SWA Bounded Replay 章节中，Decoder SWA Bounded Replay 部分提到，
> decoder SWA KV 是在 encoder 上停止 prefill 的唯一障碍。这是什么意思，为什么要在
> encoder 上停止 prefill 呢？我们是从不缓存 decoder SWA KV 的吗？为什么精确重建
> SWA KV，需要在 $L/2\times n_{\text{win}}$ 个提示词上运行 $L/2$ decoder 层数
> 的计算呢？为了解决这个问题，论文后面给出的 encoder bounded replay strategy 是什么呢？

### 问题 19 核心结论

“Decoder SWA KV 是在 Encoder 处结束 Prefill 的唯一障碍”是说：

> CED 已经可以直接从第 20 层 Encoder 输出构造第 21～40 层所需的 Global KV，
> 因此没有必要让全部 Prompt 继续完整通过 20 个 Decoder 层；但每个 Decoder 层还有一份
> 由该层自身隐藏状态生成的 Local SWA KV。首个 Decode token 立即需要这些 SWA KV，
> 所以仍必须对 Prompt 尾部做一次 Decoder replay。

这里的“在 Encoder 处结束 Prefill”不是停止整个推理，而是：

- 大部分 Prompt token 只运行前 20 个 Encoder 层；
- 不再让全部 Prompt 继续通过后 20 个 Decoder 层；
- 只让最后 $n_{\text{win}}$ 个 Prompt token 通过 Decoder，以近似重建 Decoder SWA KV；
- 然后进入正常的逐 token Decode。

论文中的 “never cache decoder SWA KV” 是指不将其纳入可复用的持久化 Prefix Cache，
不是说运行时完全不存在 Decoder SWA KV。Decode 期间它仍然必须保存在设备侧的活动
SWA KV Cache 中。

---

### 19.1 CED 中 Encoder 和 Decoder 的工作划分

DeepSeek-V4.1-Flash 的 Backbone 有 $L=40$ 个 Transformer 层：

$$
\underbrace{\text{Layer 1～20}}_{\text{Causal Encoder}}
\quad+\quad
\underbrace{\text{Layer 21～40}}_{\text{Decoder}}.
$$

令：

$$
D=\frac L2=20.
$$

#### 19.1.1 Global KV 的来源

对于后 20 个 Decoder 层，其 Global KV 不再由各层自己的隐藏状态 $H^{(l)}$ 生成，
而是直接从 Encoder 最后一层输出 $H^{(20)}$ 投影：

$$
C^{(l)}
=
H^{(20)}W_{KV}^{(l)},\qquad l>20.
$$

因此，对长度为 $N$ 的 Prompt，生成 Decoder Global KV 不需要让全部 Prompt 执行：

$$
H^{(20)}
\rightarrow \text{Decoder Layer 21}
\rightarrow\cdots
\rightarrow \text{Decoder Layer 40}.
$$

只需要对 $H^{(20)}$ 做相应的层专属投影和压缩：

```text
全部 Prompt token
        │
        ▼
Encoder Layer 1～20
        │
        ├── 生成 Encoder 自己的 SWA KV
        │
        └── H^(20)
              │
              └── 投影得到 Decoder 各层的 Global KV
```

#### 19.1.2 Local SWA KV 的来源

Decoder SWA KV 不能直接由 $H^{(20)}$ 为所有层统一产生。第 $l$ 个 Decoder 层的
SWA KV 来自该层自己的输入隐藏状态：

$$
K_{\mathrm{SWA}}^{(l)},V_{\mathrm{SWA}}^{(l)}
=
f_{KV}^{(l)}\left(H^{(l-1)}\right).
$$

例如：

$$
\begin{aligned}
\text{Layer 21 SWA KV}&\leftarrow H^{(20)},\\
\text{Layer 22 SWA KV}&\leftarrow H^{(21)},\\
\text{Layer 23 SWA KV}&\leftarrow H^{(22)},\\
&\vdots\\
\text{Layer 40 SWA KV}&\leftarrow H^{(39)}.
\end{aligned}
$$

所以只有 Layer 21 的 SWA KV 能直接从 Encoder 输出生成；Layer 22 必须等 Layer 21
执行完成，Layer 23 又必须等待 Layer 22，依此类推。

| Decode 所需历史状态 | 能否只依赖 Encoder 输出构造 | 是否要求 Prompt 执行 Decoder |
|---|---:|---:|
| Decoder Global KV | 可以 | 不需要 |
| Decoder Indexer K | 可从 Main/Global KV 派生 | 不一定需要 |
| Decoder Top-K | 可由 Indexer 计算或跨层复用 | 不需要完整 Decoder Prompt forward |
| Decoder 各层 SWA KV | 不可以，依赖各 Decoder 层自己的输入隐藏状态 | 需要某种 Decoder replay |

因此论文说 Decoder SWA KV 是阻止 Prefill 完全停在 Encoder 的唯一障碍。

---

### 19.2 在 Encoder 处结束 Prefill 是什么意思

普通 40 层模型的 Prompt Prefill 是：

```text
N 个 Prompt token
 → Layer 1
 → Layer 2
 → ...
 → Layer 40
 → 得到每层 KV Cache
 → 开始生成
```

计算规模大致为：

$$
C_{\text{normal}}\sim NL.
$$

CED 希望把它变成：

```text
N 个 Prompt token
 → Encoder Layer 1～20
 → 得到 H^(20)
 → 直接投影 Decoder Global KV

最后少量 Prompt token
 → Decoder Layer 21～40
 → 只重建 Decoder SWA KV
```

因而：

$$
C_{\text{CED}}
\approx
N\frac L2
+
n_{\text{win}}\frac L2.
$$

当：

$$
N\gg n_{\text{win}},
$$

有：

$$
C_{\text{CED}}
\approx
N\frac L2,
$$

接近普通 40 层 Prefill 的一半。

所以 “ending prefill at the encoder” 并不是说 Encoder 一执行完就直接采样 token，而是：

> 长 Prompt 的主体计算只执行 Encoder；Decoder 只接收一个有界的尾部 replay，
> 而不是接收完整 Prompt。

---

### 19.3 是否从不缓存 Decoder SWA KV

答案需要分三个层次。

#### 19.3.1 运行时设备缓存：会缓存

Decode 每一步都要读取各个 Decoder 层最近窗口的 SWA KV，因此运行时必须存在
Decoder SWA KV Cache。

在 vLLM 中：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:557-568`
  创建 `DeepseekV4SWACache`；
- 同文件 `:1021` 取得实际的 `swa_kv_cache`；
- 同文件 `:1044-1058` 将 RoPE、量化和 SWA cache insert 融合；
- BF16 写入路径在 `:1071-1083`；
- FP8 写入路径在 `:1085-1101`。

实际过程是：

```text
Decoder bounded replay
        │
        ▼
生成各 Decoder 层最近 W 个位置的 SWA KV
        │
        ▼
写入当前活动请求的设备 SWA KV Cache
        │
        ▼
首个以及后续 Decode token 读取这些 SWA KV
```

#### 19.3.2 Prefix Cache：Bounded Replay 模式下不缓存

论文说的 “we never cache decoder SWA KV”，主要指它不作为 Prefix Cache 的可命中内容保存。

vLLM 在：

- `D:\Code\vllm-github\vllm\v1\kv_cache_interface.py:937` 定义 `bounded_replay`；
- 同文件 `:949-955` 指定开启 bounded replay 时，SWA cache 的
  `prefix_cacheable=False`，Prefix 命中时需要 replay 一个 `sliding_window`。

所以：

$$
\boxed{
\text{Decoder SWA KV 存在于活动 Decode 状态}
\quad\text{但不属于可长期复用的 Prefix Cache}
}
$$

#### 19.3.3 Persistent KV Cache：不长期保存

论文 3.2.1 进一步区分：

- Global KV 有长尾复用价值，仍保存在 Persistent KV Cache；
- SWA KV 通常只对活跃会话的近期 turn 有价值；
- 将 SWA KV 在 SSD 中保存数十小时，成本高但收益低；
- Encoder SWA KV 可以短期进入使用约 10% Host DRAM 的分布式内存池；
- Decoder SWA KV 在每次 Prefill 尾部重建，只用于随后的 Decode，不用于 Prefix Cache。

| 缓存层级 | Decoder SWA KV |
|---|---|
| Decode 期间的设备活动缓存 | 有 |
| 当前请求/会话的短期状态 | 可以存在 |
| 可跨 Prefix 命中复用的缓存 | 不保存 |
| SSD/Host 上的长期 Persistent Prefix Cache | 不保存 |

---

### 19.4 为什么精确重建需要最后 L/2 乘 nwin 个 Prompt Token

关键是：

> SWA 虽然每层只看一个窗口，但多层叠加后，理论感受野会逐层向前扩张。

设：

$$
W=n_{\text{win}},
\qquad
D=\frac L2.
$$

每个 SWA 层中，位置 $i$ 可以看：

$$
[i-W+1,\ i].
$$

#### 一层时

要构造 Decoder 第 1 层最后 $W$ 个位置的 SWA KV，只需要最后 $W$ 个 Encoder 输出：

```text
Decoder Layer 1 需要：
[N-W, ..., N-1]
```

长度约为：

$$
W.
$$

#### 两层时

Decoder 第 2 层最后 $W$ 个位置的输入来自第 1 层输出。第 1 层最后 $W$ 个输出中
最早的位置 $N-W$，还需要向前看 $W-1$ 个位置。

因此第 2 层要精确重建最后 $W$ 个 KV，在 Encoder/Decoder 边界处实际需要约：

$$
2W-1
$$

个位置。

#### 三层及一般情形

继续递推，第 3 层需要约：

$$
3W-2.
$$

一般地，第 $d$ 个 Decoder 层最后 $W$ 个位置的精确理论感受野约为：

$$
R_d=d(W-1)+1.
$$

当 $d=D=L/2$ 时：

$$
R_D
=
D(W-1)+1
\approx DW
=
\frac L2n_{\text{win}}.
$$

```text
最后 W 个 Layer 23 状态
        │ 需要 Layer 22 更早的 W-1 个状态
        ▼
最后约 2W 个 Layer 22 状态
        │ 需要 Layer 21 更早的 W-1 个状态
        ▼
最后约 3W 个 Layer 21/Encoder 边界状态
```

所以论文是在说：

> 为了精确重建 20 个 Decoder 层的 SWA KV，需要从 Prompt 尾部向前取约
> $20W$ 个 Encoder 输出，再让它们通过 Decoder。

这里确实是约 $D\times W$ 个不同 Prompt 位置，不只是 $W$ 个 token 的
token-layer 计数。

#### 数值例子

DeepSeek-V4.1-Flash 中：

$$
L=40,\quad D=20,\quad W=128.
$$

最上层 Decoder SWA KV 的理论精确感受野为：

$$
20(128-1)+1=2541.
$$

论文用近似写法：

$$
D\times W=20\times128=2560.
$$

若简单地让这些 token 全部运行 20 个 Decoder 层，计算约为：

$$
2560\times20=51,200
$$

次 token-layer。

理论上可以每经过一层就裁掉一部分较早位置，形成三角形计算：

$$
128(1+2+\cdots+20)=26,880
$$

个近似 token-layer，但仍远高于 bounded replay 的：

$$
128\times20=2,560.
$$

---

### 19.5 为什么只 Replay 最后 nwin 个 Token 是近似的

Bounded Replay 不再追求恢复完整的理论感受野，只取：

$$
s=N-W
$$

到 $N-1$ 这最后 $W$ 个 Prompt token，并在每一层强制把 SWA 可见范围截断到
replay 起点 $s$：

$\mathcal L_i=[\max(s,i-W+1),\ i]$

例如 $W=4$，只 replay token 7～10：

| Query 位置 | 完整 SWA 原本可看 | Bounded Replay 实际可看 |
|---:|---|---|
| 7 | 4, 5, 6, 7 | 7 |
| 8 | 5, 6, 7, 8 | 7, 8 |
| 9 | 6, 7, 8, 9 | 7, 8, 9 |
| 10 | 7, 8, 9, 10 | 7, 8, 9, 10 |

Replay 区间开始处误差最大，因为位置 7 原本还应看到 4、5、6；越靠近区间尾部，
当前层的局部窗口越完整。但进入下一层后，早期位置的近似误差仍会传播，因此：

$$
\text{Bounded Replay SWA KV}
\ne
\text{Full Decoder Prefill SWA KV}.
$$

之所以可行，是因为：

1. Global KV 仍覆盖长程信息；
2. SWA 主要承担局部、高分辨率上下文；
3. SWA 的有效感受野通常小于理论上的 $DW$；
4. Post-training 中模拟相同 replay，使模型适应这种边界截断。

---

### 19.6 Encoder SWA Bounded Replay 的策略

Encoder Bounded Replay 解决的是：

> Prefix Cache 命中了 Global KV，但 Encoder SWA KV 没命中或没有被持久化保存。

假设：

- 原 Prompt 长度为 $N$；
- Prefix Cache 命中到位置 $h-1$；
- 未命中后缀为 $[h,N-1]$；
- replay window 为 $W$；
- replay 起点为：

$$
s=\max(0,h-W).
$$

流程为：

```text
缓存命中的 Prefix                    未缓存的新后缀
[0 ........ s-1 | s ........ h-1] | [h ........ N-1]
   不重算             replay W 个         正常 Prefill
                       Prefix token
```

把最后 $W$ 个已缓存 Prefix token 与未缓存 suffix 一起送入 Encoder。

#### 对 replay Prefix token

只重新生成 SWA KV：

$$
\text{replayed prefix}
\rightarrow
\text{SWA KV only}.
$$

其已缓存的 Global KV：

- 直接复用；
- 不重新计算；
- 不覆盖原缓存。

#### 对未缓存 suffix token

正常生成：

$$
\text{uncached suffix}
\rightarrow
\text{Global KV}+\text{SWA KV}.
$$

```text
                    Prefix Cache hit
                           │
             ┌─────────────┴─────────────┐
             │                           │
       Cached Global KV          Missing Encoder SWA KV
             │                           │
             │                 replay Prefix 最后 W 个 token
             │                           │
             │                 只重建 replay 段的 SWA KV
             │                           │
             └─────────────┬─────────────┘
                           │
                  继续处理 uncached suffix
                           │
             suffix 同时生成 Global KV 和 SWA KV
```

Encoder replay 同样是近似的：位置 $s$ 原本能读取 $s-W+1$ 到 $s$，现在只能从
$s$ 开始。因此 replay Prefix 的隐藏状态是近似的，未缓存 suffix 的隐藏状态也会受影响。

结果是：

- replay Prefix 的 Global KV 不受影响，因为直接复用已缓存值；
- replay Prefix 只重建近似 SWA KV；
- suffix 新产生的 Global KV 和 SWA KV 会依赖这个近似状态；
- Prefix Cache 命中位置不同，suffix 状态也可能略有不同。

论文明确指出，不同 cache-hit position 下结果不是数学完全一致，但实验显示响应质量几乎不受影响。

---

### 19.7 Encoder 与 Decoder Bounded Replay 的区别

| 对比项 | Encoder SWA Bounded Replay | Decoder SWA Bounded Replay |
|---|---|---|
| 触发原因 | Prefix Global KV 命中，但 Encoder SWA KV 缺失 | CED 希望避免完整 Prompt 运行 Decoder |
| Replay 输入 | 已命中 Prefix 的最后 $W$ 个 token，加未命中 suffix | 完整 Prompt 的最后 $W$ 个 token |
| Replay 哪些层 | 前 $L/2$ 个 Encoder 层 | 后 $L/2$ 个 Decoder 层 |
| Replay token 的 Global KV | 已缓存，禁止重算和覆盖 | Decoder Global KV 已从 Encoder 输出投影 |
| Replay 主要产物 | Encoder SWA KV | Decoder SWA KV |
| suffix 是否生成 Global KV | 是 | Decoder Global KV 走 CED 投影路径 |
| 结果是否精确 | 否 | 否 |
| 用途 | 恢复 Prefix 命中后的 Encoder 局部状态 | 为首个和后续 Decode token 准备 Decoder 局部状态 |

两者共享同一条截断规则：

$$
\mathcal L_i
=
[\max(s,i-W+1),i].
$$

---

### 19.8 vLLM 中的实现链路

#### 19.8.1 Scheduler 将 Prefix 命中位置向前回退一个窗口

`D:\Code\vllm-github\vllm\v1\core\sched\scheduler.py:3056-3068`：

```python
num_replay_tokens = min(self.prefix_replay_tokens, num_hit_tokens)
request.replay_start = num_hit_tokens - num_replay_tokens
```

也就是：

$$
s=h-\min(W,h).
$$

随后 `scheduler.py:1085-1096` 将 `num_computed_tokens` 向前回退，使最后 $W$ 个
Prefix token 再次进入 Prefill 调度。

#### 19.8.2 Replay token 不覆盖已经缓存的 Global KV

`D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model_state.py:317-331`
把 replay token 在可 Prefix Cache 的缓存组中的 `slot_mapping` 改为 padding slot：

```text
Replay token：
  SWA cache group     → 允许写入，重建 SWA
  Global cache group  → padding slot，不覆盖缓存
```

这对应论文所说：

> Replayed tokens regenerate only SWA KV, reusing the cached global KV without
> recomputation or overwriting.

#### 19.8.3 Attention 不读取 replay_start 之前的 SWA KV

`D:\Code\vllm-github\vllm\models\deepseek_v41\common\ops\cache_utils.py:1407-1410`：

```python
swa_start_pos = max(pos - (WINDOW_SIZE - 1), replay_start)
```

对应：

$$
\operatorname{SWAStart}(i)
=
\max(s,i-W+1).
$$

模型状态的总体说明位于
`D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model_state.py:160-179`：

- SWA 不进入 Prefix Cache；
- Prefix 命中后重算最后一个窗口；
- replay token 不覆写已缓存 KV；
- SWA kernel 不读取 `replay_start` 之前的状态。

#### 19.8.4 Decoder 后半部分只执行尾部窗口

`D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:757-784` 先找到最后一个
KV source：

```python
cut = max(config.kv_source_layer_ids)
self.decoder_replay_start = cut + 1
```

含义是：

- 最后一个 KV source 之前照常执行；
- 其后的层不需要为完整 Prompt 生成新的 Global KV；
- 后续层只对尾部窗口 replay，以重建各层 SWA KV。

`D:\Code\vllm-github\vllm\models\deepseek_v41\decoder_replay_layers.py:3-11`
明确说明：最后一个 KV source 后面的层，在 eager prefill 中只运行每个请求最后
`window` 行。

实际执行位于同文件 `:45-67`：

1. 用 `rows` 选出尾部 replay token；
2. 建立独立 forward context；
3. 只对这些行执行后续 Decoder layers；
4. 将结果散回原 batch 对应行。

模型主流程：

- `model.py:955-967`：先执行完整 batch 的前半部分；
- `model.py:969-978`：后半部分进入 `decoder_replay_layers`；
- `model.py:1103-1127`：实际运行 last KV source 后的层。

`model_state.py:423-429` 为每个请求保留：

```python
kept_lens = min(query_lens, layers.window)
```

然后 `model_state.py:466-507` 构造尾部子 batch 的 token rows、positions、
query boundaries、slot mappings 和 replay 起点。

---

### 19.9 vLLM-Ascend 当前代码状态

Ascend 路径明确实现了运行时 SWA KV Cache：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:552-574`
  定义 `AscendDeepseekV41SWACache`；
- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:283-291`
  计算并写入 SWA KV；
- 同文件 `:351-368` 将 KV projection、归一化、RoPE 和 cache store 与 Q 路径重叠。

但是，就当前本地代码快照的静态搜索结果：

- 没有找到对应的 `decoder_replay_layers`；
- 没有找到 `replay_start`；
- 没有找到 `prefix_replay_tokens` 或 `bounded_replay` 的 DeepSeek-V4.1 实现；
- `vllm_ascend\models\deepseek_v41\model.py:559-568` 创建 SWA spec 时没有开启 bounded replay；
- `vllm_ascend\core\kv_cache_interface.py:236-286` 没有像上游 NVIDIA 路径那样增加
  专门的 replay 字段和逻辑。

因此，当前代码能确认的是：

> vLLM NVIDIA DeepSeek-V4.1 路径已经实现 Encoder/Decoder SWA bounded replay
> 的调度和执行；当前 vLLM-Ascend 快照实现了普通运行时 SWA cache，但没有看到同等的
> bounded replay 机制。

这是当前代码快照的静态结论，不能据此断言其他内部 CANN 或部署分支也一定没有相关实现。

---

### 19.10 100K Prompt 的计算量示例

假设：

$$
N=100,000,\quad L=40,\quad D=20,\quad W=128.
$$

#### 不使用 CED

$$
100,000\times40
=
4,000,000
$$

个主要 token-layer。

#### CED，但精确重建 Decoder SWA

Encoder 主体：

$$
100,000\times20
=
2,000,000.
$$

Decoder 精确 replay 约取 2560 个 Prompt token：

$$
2560\times20
=
51,200.
$$

合计约：

$$
2,051,200.
$$

#### CED 加 Decoder Bounded Replay

只 replay 最后 128 个 token：

$$
128\times20
=
2,560.
$$

合计：

$$
2,000,000+2,560
=
2,002,560.
$$

与普通模型相比：

$$
\frac{2,002,560}{4,000,000}
\approx50.06\%.
$$

这就是论文所说的 “nearly halving total prefill computation”。

---

### 问题 19 最终总结

1. **在 Encoder 处结束 Prefill**  
   指绝大部分 Prompt 只运行前 $L/2$ 层，不是停止推理或跳过 Decode。

2. **Global KV 不是障碍**  
   Decoder Global KV 可以从 $H^{(L/2)}$ 直接投影，不要求全部 Prompt 穿过 Decoder。

3. **Decoder SWA KV 是唯一障碍**  
   它是 layer-local 的，后层 SWA KV 依赖前层输出，不能仅从 Encoder 最终输出一次性生成。

4. **“不缓存”不等于运行时不存在**  
   Decoder SWA KV 不纳入持久化 Prefix Cache，但 replay 后仍写入活动设备缓存，供 Decode 使用。

5. **精确重建的感受野会逐层扩大**

   $$
   R_D=D(W-1)+1\approx DW.
   $$

   因此 20 个 Decoder 层、窗口 128 时，需要约 2560 个尾部 Prompt token。

6. **Decoder Bounded Replay**  
   放弃精确恢复，只让最后 $W$ 个 Prompt token 运行后 $L/2$ 个 Decoder 层：

   $$
   C_{\text{decoder replay}}
   =
   W\frac L2.
   $$

7. **Encoder Bounded Replay**  
   当 Global KV Prefix 命中而 Encoder SWA 缺失时，回放命中 Prefix 的最后 $W$ 个 token：
   replay Prefix 只重建 SWA，已缓存 Global KV 不重算、不覆盖，uncached suffix 同时生成
   Global KV 和 SWA KV。

8. **两种 Replay 都是近似计算**  
   它们都把 SWA 截断在 `replay_start`：

   $$
   \mathcal L_i=[\max(s,i-W+1),i].
   $$

   论文通过实验以及 Post-training 中的 replay 模拟，使这种近似对响应质量的影响保持很小。

---

## 问题 20：DeepSeek-V4 与 DeepSeek-V4.1-Flash 的 KV Cache 管理差异

### 问题 20 原文

> DeepSeek-V4 的 KV cache 管理方式，和 DeepSeek-V4.1 flash 的 KV cache 管理方式，有什么区别？

### 问题 20 核心结论

这里的 DeepSeek-V4 应以论文实际对比的 **DeepSeek-V4-Flash** 为基线。
V4.1-Flash 的变化不只是“把 FP8 KV 改成 FP4 KV”，而是同时改变了：

1. Global/Main KV 在不同层之间的所有权；
2. Indexer K 和 Top-K 索引的复用方式；
3. Main KV 的存储精度；
4. SWA KV 的持久化与缺失重建方式；
5. 长 Prompt 在 Prefill 阶段生成 Decoder KV 的路径。

可以概括为：

> **DeepSeek-V4：** 每个压缩注意力层主要维护自己的长程 KV，另有每层 SWA 状态；持久化缓存可同时保存 Global KV 与 SWA 快照。  
> **DeepSeek-V4.1-Flash：** 多层共享 Main KV、Indexer K 和部分 Top-K 结果，Main KV 进一步 FP4 化；长期持久化只保存 Global KV，SWA 缺失时用 Bounded Replay 重建。

---

### 20.1 必须区分运行时 KV Cache 与持久化 Prefix Cache

论文中的“KV Cache 管理”包含两个不同层级：

1. **设备端运行时 KV Cache**
   - 当前活跃请求在 HBM 中使用；
   - 包含 Main/Global KV、Indexer K、SWA KV、压缩尾部状态等；
   - 直接参与当前 Prefill 或 Decode。

2. **持久化 Prefix Cache**
   - 将已经计算过的长前缀存入 Host DRAM、分布式缓存或 SSD；
   - 后续请求命中相同前缀时，可以避免重新执行大段 Prefill；
   - 论文所说 V4.1 持久化容量约为 V4 的八分之一，主要指这一层。

因此，不能把“V4.1 不再持久化 SWA”理解成“V4.1 运行时没有 SWA KV”。V4.1 在活跃请求中仍需要每层 SWA KV。

---

### 20.2 两代方案总览对比

| 对比维度 | DeepSeek-V4-Flash | DeepSeek-V4.1-Flash |
|---|---|---|
| 长程注意力 | CSA 与 HCA 混合 | 统一为纯 CSA2 |
| 长程压缩比 | CSA 通常为 $m=4$；HCA 通常为 $m'=128$ | 通常为 $m=1/2$，再依靠层间共享降低总量 |
| Main/Global KV 所有权 | 每个压缩层主要分别生成、分别缓存 | 只有 KV source layer 生成，多个 consumer layer 共享 |
| Indexer K | CSA 层从本层 hidden states 单独压缩得到 | 从共享 Main KV latent 投影得到，并在层间共享 |
| Top-K 搜索 | CSA 层通常执行自己的检索 | Full、Reindex、Reuse：重新搜索、重新打分或直接复用 |
| Token 压缩窗口 | CSA 存在重叠窗口和绝对位置嵌入 | 去掉重叠和额外绝对位置嵌入 |
| Main KV 精度 | FP8 | FP4，E2M1 数据配 E4M3 block scale |
| Indexer Q/K | 已使用 FP4 QAT | 延续 FP4，并与共享 Main KV 配合 |
| SWA KV 精度 | 通常 FP8 | 仍为 FP8，没有激进改成 FP4 |
| 设备端 SWA | 每层保存固定窗口 | 仍然每层保存固定窗口 |
| 持久化 SWA | Full、Periodic 或 Zero SWA caching | 不再进入长期持久化缓存 |
| SWA 缺失恢复 | Zero 策略精确恢复代价可达 $L n_{win}$ | Bounded Replay，只回放最后 $n_{win}$ 个 token |
| Prompt Prefill | Prompt 通过全部 Decoder 层 | CED：主体只通过 Encoder，Decoder 仅处理尾部 Replay |
| Global KV 占用 | 比较基线 | 约为 V4 的 $1/4$ |
| 持久化总占用 | Global KV 加 SWA 快照 | 约为 V4 的 $1/8$ |

---

### 20.3 DeepSeek-V4 的设备端 KV Cache 组织

V4 的设备端不只有一种 KV Cache，可以概括成：

```text
一个请求的设备端缓存
│
├── Classical / Compressed KV Cache
│   ├── CSA 压缩 KV，通常压缩比 m = 4
│   ├── HCA 压缩 KV，通常压缩比 m' = 128
│   └── CSA Indexer K Cache
│
└── State Cache
    ├── 每层最近 n_win 个 token 的 SWA KV
    └── 尚未凑满压缩窗口的尾部 token 状态
```

其中：

- 压缩 KV Cache 保存已经完成压缩的长程历史；
- SWA KV 保存最近窗口内的逐 token 精细信息；
- Compressor state 保存不足以形成下一条压缩 KV 的尾部 token。

例如，CSA 每 4 个新 token 产生一条压缩表示。如果当前有 102 个 token，已经满足压缩条件的部分可进入 compressed cache，最后尚未完成下一组的 token 则保留在 state cache。与此同时，SWA 独立保存最后 $n_{win}$ 个 token 的逐 token KV。

vLLM-Ascend 中对应代码为：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v4\compressor.py:50-86`：按照压缩比建立 CSA/HCA 缓存和状态规格；
- 同文件 `:119-125`：`ratio=4` 的 CSA 启用 overlap 和绝对位置嵌入；
- 同文件 `:154-174`：为尚未完成压缩的 token 分配 compressor state；
- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v4\model.py:567-613`：各压缩注意力层分别创建 compressor；
- 同文件 `:615-636`：每层另外创建自己的 SWA Cache 和稀疏注意力模块。

上游 vLLM 也体现了逐层所有权：

- `D:\Code\vllm-github\vllm\models\deepseek_v4\attention.py:297-317`：`ratio=4` 的 CSA 层分别创建 `DeepseekV4Indexer`；
- 同文件 `:339-352`：每层创建自己的 SWA Cache；
- 同文件 `:360-374`：每个压缩层创建自己的 compressor；
- 同文件 `:440-444`：`ratio=128` 进入 HCA 分支。

所以 V4 虽然已经在 token 维度压缩长程 KV，但同类历史表示仍需要在许多层中分别生成和保存。

---

### 20.4 V4 的 Indexer K 与重叠 Token 压缩

#### V4 的 Indexer K 是相对独立的缓存路径

CSA 要从压缩长程历史中选出 Top-K，除了真正参与 Main Attention 的 Main KV，还需要 Indexer K：

```text
hidden states
   │
   ├── Main KV compressor ──> Main KV Cache
   │
   └── Indexer compressor ──> Indexer K Cache
                                  │
Indexer Query ────────────────────┘
                 │
               Top-K
                 │
           Main Attention
```

在 V4 上游实现中：

- `attention.py:882-920` 定义独立的 Indexer K cache spec；
- `attention.py:953-957` 支持 FP4 或 FP8 Indexer Cache；
- `attention.py:1005-1021` 计算 FP4 Indexer K 的存储记录大小；
- `attention.py:1022-1031` 为 Indexer K 建立独立的 hidden-state compressor。

V4 已经对 Indexer Q/K 使用 FP4 QAT，但 Main KV 通常仍为 FP8，而且 Indexer K 是从 hidden states 通过单独压缩路径得到的。

#### V4 的 CSA 存在 overlap

对于压缩比 $m$：

- 一条压缩表示覆盖 $2m$ 个原始 token；
- 相邻窗口之间重叠 $m$ 个 token；
- 每前进 $m$ 个 token 产生一条新压缩表示；
- 长期平均压缩率仍近似为 $1:m$。

以 $m=4$ 为例：

```text
原始 token：

0 1 2 3 4 5 6 7 | 8 9 10 11 | 12 13 14 15
└──── 窗口 0 ────┘
        └──────── 窗口 1 ────────┘
                └──────── 窗口 2 ────────┘

窗口长度：8 = 2m
滑动步长：4 = m
重叠长度：4 = m
```

这种设计增强了边界附近的信息覆盖，但也增加了压缩状态、尾部 token 和缓存布局的复杂度。V4.1 的 CSA2 去掉了 overlap 和对应的额外绝对位置嵌入。

---

### 20.5 V4.1 的 CSA2 跨层共享

CSA2 把 Decoder 层区分成：

- KV source layer；
- Index source layer；
- 普通 consumer/reuse layer。

```text
KV source layer
   │
   ├── 生成共享 Main KV Cache ─────────────┐
   └── 生成/提供共享 Indexer K             │
                                           │
Index source / Reindex layer               │
   ├── 读取共享 Indexer K                  │
   ├── 对共享候选池重新打分                 │
   └── 产生新的 Top-K indices              │
                                           │
Reuse layer                                │
   ├── 读取共享 Main KV ◀──────────────────┘
   └── 直接复用之前的 Top-K indices
```

因此，V4.1 的共享包含三层含义：

1. 多层共享 Main KV；
2. 多层共享 Indexer K；
3. 若干层还可以共享 Top-K indices。

上游 vLLM 中：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:322-348`：读取 `kv_source_layer_ids` 和 `index_source_layer_ids`，consumer 解析最近的 source；
- 同文件 `:431-482`：只有相应源层拥有 Indexer K cache，非源层引用共享缓存；
- 同文件 `:573-603`：只有 KV source layer 创建 compressor；
- 同文件 `:1110-1144`：只有 KV source layer 申明压缩 Main KV Cache；
- 同文件 `:1227-1240`：只有 Index source layer 创建 Indexer，其他层复用其结果。

vLLM-Ascend 中：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:498-549`：建立 KV source 和 Index source 的层映射；
- 同文件 `:680-714`：只有 `role.is_kv_source` 的层创建 long KV cache 和 compressor；
- 同文件 `:715-739`：只有 Index source 层创建 Indexer，consumer 指向对应 source 的共享缓存。

---

### 20.6 Full、Reindex、Reuse 改变了什么

| 模式 | Main KV | Indexer K | Top-K indices | 当前层 Query/SWA |
|---|---|---|---|---|
| Full | 当前 source 层生成 | 当前 source/index 层生成 | 扫描完整可见历史并产生 | 当前层仍正常计算 |
| Reindex | 复用共享 Main KV | 复用共享 Indexer K | 对共享候选池重新打分 | 当前层仍正常计算 |
| Reuse | 复用共享 Main KV | 复用 | 直接复用已有 Top-K | 当前层仍正常计算 |

共享 KV 并不意味着整个 Attention 输出共享。每层仍有自己的 hidden state，因此仍会生成本层 Query，并通过本层 Attention 和 FFN/MoE 更新 hidden state。

可以类比为：

```text
每层自己的 Query     = 不同的读者
共享 Indexer K       = 一本共享索引
共享 Main KV         = 一间共享档案库
每层自己的 SWA KV    = 每位读者身边的局部便签
```

---

### 20.7 V4.1 的 FP4 Main KV

两代模型的典型精度可以简化表示为：

```text
V4：
Main KV       : FP8
Indexer Q/K   : FP4 QAT
SWA KV        : FP8

V4.1：
Main KV       : FP4 QAT
Indexer Q/K   : FP4 QAT
SWA KV        : FP8
```

V4.1 Main KV 使用 E2M1 4-bit 数据，并为每 16 个通道配置一个 E4M3 8-bit scale。假设 Main KV latent 有 512 个 FP4 数值：

$$
512\times4/8=256\ \text{bytes},
$$

scale 数量为：

$$
512/16=32,
$$

每个 scale 占 1 byte，所以合计：

$$
256+32=288\ \text{bytes}.
$$

上游 vLLM 的对应位置：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:504-518`：根据 cache dtype 计算记录大小，其中 NVFP4 compressed Main KV 约为 288 bytes；
- 同文件 `:1147-1210`：Main KV 与 Indexer K 的物理缓存规格和布局管理。

SWA 没有随 Main KV 一起改为 FP4，是因为最近窗口中的逐 token 局部信息对量化误差更敏感；论文继续为 SWA 保留 FP8。

---

### 20.8 SWA KV 的运行时与持久化管理差异

#### V4.1 运行时仍然需要 SWA KV

V4.1 中实际参与注意力的历史 KV 可抽象为：

$$
\mathcal J_t=\mathcal L_t\mathbin{\Vert}\mathcal G_t,
$$

其中：

- $\mathcal L_t$ 是当前层自己的 layer-local SWA KV；
- $\mathcal G_t$ 是从共享 compressed Global/Main KV 中选出的 Top-K；
- 两部分必须在同一个 Softmax 归一化域中联合计算。

所以“V4.1 不缓存 SWA”并不准确。准确说法是：

> V4.1 在活跃请求运行期间仍缓存 SWA KV，但不再把 SWA KV 放入长期持久化 Prefix Cache。

Ascend 代码 `deepseek_v41/model.py:552-574` 仍为每层建立 `AscendDeepseekV41SWACache`，证明运行时 SWA 并未消失。

#### V4 的持久化策略

V4 的持久化缓存可包含：

```text
Global / Compressed KV
+
SWA KV checkpoints
  ├── Prompt 末尾
  └── Output 末尾
```

Global KV 与 SWA KV 分别执行 LRU 管理。论文讨论三种 SWA 策略：

1. Full SWA Caching：完整保存 SWA，恢复快但容量大；
2. Periodic Checkpointing：周期性保存，从最近检查点向后重放；
3. Zero SWA Caching：不持久化 SWA，命中时精确重建。

Zero SWA Caching 的问题是精确恢复可能要求最后 $n_{win}$ 个 token 重新通过全部 $L$ 层：

$$
C_{\text{V4 zero-SWA}}\approx L n_{win}.
$$

#### V4.1 的持久化策略

V4.1 改为：

```text
长期 SSD Prefix Cache（约 72 小时或更长）
└── Global/Main KV

短期分布式 Host DRAM Pool（分钟级 TTL）
└── SWA KV；命中时直接使用

如果 SWA DRAM 未命中
└── Bounded Replay 重建最后 n_win 个 token 的 SWA
```

论文给出的部署方案会从 Host DRAM 中划出约 10% 用于短期 SWA Pool，但不再将 SWA 纳入长期 SSD 持久化路径。

---

### 20.9 Bounded Replay 如何替代持久化 SWA

#### V4 精确恢复为何昂贵

在普通 Decoder-only 依赖链中，第 $l$ 层最后 $n_{win}$ 个 token 的 hidden states 依赖第 $l-1$ 层的输出。要精确恢复所有层的 SWA KV，通常必须让一段尾部 Prompt 依次通过全部层：

```text
最后一段 Prompt token
  → Decoder 第 0 层
  → Decoder 第 1 层
  → ...
  → Decoder 第 L-1 层
```

其层-token 代价约为 $L n_{win}$，而多层 SWA 感受野扩张时，实际所需输入尾段还可能更长。

#### V4.1 的 CED 改变了 Global KV 的生成依赖

V4.1 中：

1. Prompt 主体先通过 20 层 Causal Encoder；
2. 得到 Encoder 输出 $H^{(20)}$；
3. Decoder 各 KV source layer 的 Global KV 可直接从该共享 Encoder 输出投影；
4. 不再要求全部 Prompt token 先完整通过后半段 Decoder，才能构建 Global KV。

于是 Prefix Cache 命中但 SWA 丢失时：

```text
缓存命中的前缀
├── Global KV：从持久化缓存直接读取
└── SWA KV：只回放最后 n_win 个 token 重建
```

Bounded Replay 将回放段左边界设为 $s$，对回放 token $i$ 使用：

$$
\mathcal L_i=[\max(s,i-W+1),i],\qquad W=n_{win}.
$$

这意味着回放 token 不能通过 SWA 越过 $s$ 看到更早的局部 token。它不是旧执行过程的逐位精确恢复，而是一种由训练阶段适配过的近似恢复；更早的历史仍可通过 Global KV 被访问。

上游 vLLM 中：

- `D:\Code\vllm-github\vllm\v1\kv_cache_interface.py:937-955`：Bounded Replay 的 SWA spec 被标记为不可直接做 prefix cache，并设置 replay token 数等于 sliding window；
- `D:\Code\vllm-github\vllm\v1\core\sched\scheduler.py:3056-3068`：计算 replay 起点与 token 数；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model_state.py:160-179`：缓存命中后重建窗口内 SWA，同时不覆盖已缓存 Global KV。

---

### 20.10 100K Prompt 示例

假设：

- Prompt 长度为 100,000；
- SWA window 为 128；
- V4 有 43 层；
- V4.1 Causal Encoder 为 20 层；
- 暂时忽略 batch、padding、物理块碎片和元数据。

#### V4

Prompt Prefill 需要经过全部 Decoder 层：

```text
100,000 个 Prompt token × 43 层
```

运行时每层保留最近 128 个 token 的 SWA，同时压缩层分别维护自己的 CSA/HCA KV；CSA 层还有自己的 Indexer K。

若持久化采用 Zero SWA Caching，简单口径下恢复代价约为：

$$
43\times128=5504
$$

个层-token 计算单元。

#### V4.1

Prompt 主体首先执行：

```text
100,000 个 token × 20 层 Causal Encoder
```

Global KV 从 Encoder 输出投影，只由部分 KV source layer 生成，再由其余 Decoder 层共享。Decoder 仅对 Prompt 尾部约 128 个 token 做 Bounded Replay，以建立 Decode 所需的 layer-local SWA 状态。

持久化时长期保存 FP4 Global/Main KV，不保存 SWA；恢复时 Global KV 直接加载，SWA 若未在短期 DRAM 中命中，则回放最后 128 个 token。

所以 V4.1 同时减少了：

- Prompt 在全部 Decoder 层中的重复处理；
- 多层重复保存 Main KV 的空间；
- 多层重复保存 Indexer K 的空间；
- 多层重复执行 Top-K 搜索的开销；
- Main KV 的元素位宽；
- 长期保存 SWA 快照的容量。

---

### 20.11 890 Bytes/Token 和八分之一容量的含义

论文给出的 V4.1 Global KV footprint 为：

$$
890\ \text{bytes/token},
$$

约为 V4-Flash 的四分之一。这是整个模型范围内平均到一个 token 的 Global KV 开销，不是某一层单个 K 向量的大小。它综合了：

- 只有部分层是 KV source；
- 多个 consumer 层共享 Main KV；
- Main KV 使用 FP4；
- Indexer K 使用紧凑格式；
- CSA2 去除了旧 CSA 的部分冗余结构。

反推作直观估算：

$$
\text{V4 Global KV}\approx4\times890=3560\ \text{bytes/token}.
$$

一百万 token 的粗略对比如下：

| 模型 | Global KV 粗略容量 |
|---|---:|
| V4-Flash | 约 3.56 GB |
| V4.1-Flash | 约 0.89 GB |

这只是由论文比例反推的直观值，不包括 block/page 对齐、allocator 碎片、metadata、临时 Top-K buffer、SWA 运行时缓存及通信缓冲区。

#### 为什么持久化总量约为八分之一

设 V4 的持久化容量为：

$$
P_{V4}=G_{V4}+S_{V4}.
$$

论文指出 V4 部署中的 SWA 大约占持久化容量的一半，因此近似有：

$$
G_{V4}\approx\frac12P_{V4},\qquad
S_{V4}\approx\frac12P_{V4}.
$$

V4.1 将 Global KV 压到 V4 的约四分之一，同时不再持久化 SWA：

$$
P_{V4.1}
\approx\frac{G_{V4}}4
\approx\frac{P_{V4}/2}{4}
=\frac{P_{V4}}8.
$$

因此八倍收益不是 FP4 单独产生的，而是两部分相乘：

```text
跨层共享 + FP4 Main KV
             ↓
       Global KV ≈ 1/4

取消长期持久化 SWA
             ↓
       总量再减少约一半

合计：持久化总容量 ≈ 1/8
```

这里的 $1/8$ 不能理解为所有运行时显存也缩到八分之一。活跃请求仍需要 SWA、Main KV、Indexer K、Top-K indices、Replay 临时状态及调度缓冲区。

---

### 20.12 当前 vLLM 与 vLLM-Ascend 的实现边界

从当前本地代码可以确认：

1. **上游 vLLM DeepSeek-V4.1 路径**
   - 实现了 KV source、Index source 和 consumer 的 CSA2 共享关系；
   - 可见 NVFP4 Main KV 的缓存规格；
   - 可见 Bounded Replay 的缓存接口、Scheduler 与 NVIDIA model state 路径。

2. **当前 vLLM-Ascend DeepSeek-V4.1 路径**
   - 已实现 KV source、Index source 和 consumer 的共享拓扑；
   - 已实现每层运行时 SWA Cache；
   - `deepseek_v41/model.py:712` 当前可见的 `long_kv_cache` 使用 BF16 路径；
   - 当前静态搜索未发现与上游 NVIDIA 路径相同的 `bounded_replay`、`replay_start` 或 `decoder_replay_layers` 完整实现。

因此必须区分三件事：

- 论文定义的 V4.1 能力：FP4 Main KV、CSA2、CED 和 Bounded Replay；
- 上游 NVIDIA vLLM 当前可见实现：已体现 NVFP4 和 Bounded Replay 路径；
- 当前本地 Ascend 快照：共享拓扑和运行时 SWA 已实现，但不能仅凭当前静态代码断言论文的 FP4 Main KV 与 Bounded Replay 已完整落地。

这只是当前本地代码快照的静态结论，并不能排除内部 CANN 算子或其他部署分支存在额外实现。

---

### 问题 20 最终总结

DeepSeek-V4 的主要思想是：

> 在各注意力层内部用 CSA/HCA 压缩长程历史，再用每层 SWA 保存最近上下文。

DeepSeek-V4.1-Flash 则进一步变成：

> 用 CED 避免绝大多数 Prompt token 通过全部 Decoder；用 CSA2 让多个 Decoder 层共享 Main KV、Indexer K 和 Top-K；将 Main KV 压成 FP4；长期只持久化 Global KV，SWA 缺失时用 Bounded Replay 重建。

最本质的区别可以写成：

```text
V4：
以“每一层内部压缩”为主

V4.1：
以“Encoder 生成全局表征
  + Decoder 跨层共享
  + FP4 低精度存储
  + Global/SWA 分级生命周期管理”为主
```

还要特别记住四点：

1. V4.1 仍有运行时 SWA，只是不再长期持久化；
2. FP4 只贡献了部分容量下降，跨层共享同样关键；
3. `1/4` 主要描述 Global KV，`1/8` 主要描述持久化总容量；
4. 跨层共享的是历史 KV 和检索结果，每层自己的 Query、SWA、Attention 输出和 FFN/MoE 计算仍然存在。

---

## 问题 21：1000 Token Prefill 中 Causal Encoder 的 SWA 计算范围

### 问题 21 原文

> 对于 Prefill 阶段的 SWA，window 长度是 128。如果输入序列长度是 1000，那么对于 Causal Encoder 的第 0～1 层，第 872～999 位置的 SWA 需要计算，那第 0～871 位置的 Attention 需要计算吗？

### 问题 21 核心结论

如果这是一次**没有 Prefix Cache 命中的首次完整 Prefill**，那么 Causal Encoder 的第 0 层、第 1 层，乃至整个第 0～19 层，都必须对位置 0～999 执行本层计算。位置 0～871 的 SWA Attention 也必须计算。

窗口 128 限制的是：

> 每个 Query 最多直接读取多少个局部 Key/Value。

它不表示：

> 整个序列只计算最后 128 个 Query。

Prefill 完成以后，供后续 Decode 使用的**逻辑有效 SWA 尾窗**通常只需保留位置 872～999。因而需要区别“计算过哪些位置”和“最后还保留哪些位置”。

只有在 Global KV Prefix 已命中、Encoder SWA 缺失并触发 **Encoder SWA Bounded Replay** 时，才可能不重算 0～871，只回放 872～999。

---

### 21.1 窗口 128 限制的是 Key 范围而不是 Query 数量

令：

$$
N=1000,\qquad W=128.
$$

位置从 0 开始编号。普通完整 Prefill 中，位置 $i$ 的 Query 能直接看到的 SWA Key/Value 范围为：

$$
\mathcal L_i=[\max(0,i-W+1),i]
=[\max(0,i-127),i].
$$

具体例子：

| Query 位置 | SWA 可见的 Key/Value 位置 | 数量 |
|---:|---|---:|
| 0 | `[0,0]` | 1 |
| 1 | `[0,1]` | 2 |
| 127 | `[0,127]` | 128 |
| 128 | `[1,128]` | 128 |
| 500 | `[373,500]` | 128 |
| 871 | `[744,871]` | 128 |
| 872 | `[745,872]` | 128 |
| 998 | `[871,998]` | 128 |
| 999 | `[872,999]` | 128 |

因此全部 Query：

$$
Q_0,Q_1,\ldots,Q_{999}
$$

都需要计算，只是每个 Query 的局部 Key 范围最多为 128。

例如位置 500 仍需执行：

```text
Query 500
    ↓
读取 K/V 373～500
    ↓
得到位置 500 的 Attention 输出
    ↓
残差、归一化、FFN/MoE
    ↓
作为下一层位置 500 的输入
```

---

### 21.2 必须区分计算、直接可见和最终保留

对于长度 1000 的完整 Prefill，可以分别问三个问题。

#### 位置 0～871 是否作为 Query 执行 Attention

是。完整 Prefill 中，0～999 全部作为 Query 执行本层计算。

#### 位置 999 是否通过本层 SWA 直接读取 0～871

否。位置 999 的直接 SWA 窗口是：

$$
[999-128+1,999]=[872,999].
$$

#### Prefill 结束后是否仍要保存 0～871 的 SWA KV

逻辑上不必全部保留。处理完位置 999 后，供下一个 Decode token 使用的局部尾窗主要是 872～999。位置 0～871 的 SWA KV 已经完成历史使命，可以被回收；但这些位置的 Attention 输出已经传递到后层，并参与 Encoder 最终表示和 Global KV 的生成。

所以正确关系是：

```text
已经作为 Query 计算过的位置：0～999

位置 999 直接通过 SWA 读取：872～999

Prefill 结束时逻辑上继续保留的 SWA 尾窗：872～999
```

---

### 21.3 为什么位置 0 到 871 必须计算

#### 它们必须生成自己的 Encoder hidden states

第 0 层需要为所有位置产生输出：

$$
H_0^{(1)},H_1^{(1)},\ldots,H_{999}^{(1)}.
$$

这些输出是第 1 层的输入。比如第 1 层位置 872 的 SWA 窗口需要第 0 层位置 745～872 的 hidden states；如果第 0 层跳过 0～871，就无法得到正确的后层输入。

#### CED 需要整个 Prompt 的 Encoder 最终输出

CED 的前 $L/2$ 层处理全部 $N$ 个 Prompt token，得到：

$$
H^{(20)}=\{H_0^{(20)},H_1^{(20)},\ldots,H_{999}^{(20)}\}.
$$

Decoder KV source layer 再从这些 Encoder 输出投影 Global/Main KV：

$$
C^{(l)}=H^{(20)}W_{KV}^{(l)}.
$$

如果 Encoder 只计算位置 872～999，Global KV 中就会丢失 Prompt 前 872 个位置的表征。

#### CED 复杂度公式也包含全部 Encoder token

论文给出的：

$$
C_{\text{CED}}
=N\frac L2+n_{\text{win}}\frac L2
$$

第一项 $N L/2$ 表示全部 $N$ 个 token 经过全部 $L/2$ 个 Causal Encoder 层；第二项才是 Decoder 尾部 Replay。

若：

$$
N=1000,\quad L=40,\quad n_{\text{win}}=128,
$$

那么：

$$
C_{\text{encoder}}=1000\times20=20000,
$$

而 Decoder Bounded Replay 为：

$$
C_{\text{decoder replay}}=128\times20=2560.
$$

---

### 21.4 多层 SWA 的间接感受野

一层 SWA 直接看到 128 个位置，但多层堆叠后，早期信息会通过 hidden states 逐层向后传播。

只考虑局部分支，经过 $d$ 层后的理论感受野宽度为：

$$
R_d=1+d(W-1).
$$

对于 $W=128$：

- 一层：$R_1=128$；
- 两层：$R_2=255$；
- 20 层：$R_{20}=1+20\times127=2541$。

例如第 1 层位置 999 直接读取第 0 层位置 872～999；但第 0 层位置 872 自己已经聚合过原始位置 745～872，因此两层以后位置 999 可以间接依赖 745～999。

这说明早期位置的计算结果即使不再驻留于当前 SWA 尾窗，也已经写入后续 hidden states。

---

### 21.5 单层 SWA 配对数

单层长度 1000、窗口 128 的 Query-Key 配对数为：

$$
C_{\text{SWA}}=
\sum_{i=0}^{999}\min(i+1,128).
$$

拆开：

$$
C_{\text{SWA}}
=\sum_{j=1}^{128}j+(1000-128)\times128.
$$

其中：

$$
\sum_{j=1}^{128}j=\frac{128\times129}{2}=8256,
$$

$$
872\times128=111616.
$$

因此：

$$
C_{\text{SWA}}=119872.
$$

分段如下：

| Query 范围 | Query 数 | SWA 配对数 |
|---|---:|---:|
| 0～127 | 128 | 8,256 |
| 128～871 | 744 | 95,232 |
| 872～999 | 128 | 16,384 |
| 合计 | 1,000 | 119,872 |

位置 0～871 对应的配对数为：

$$
8256+95232=103488,
$$

所以首次完整 Encoder Prefill 绝不是只计算最后 128 个 Query。

---

### 21.6 vLLM 与 vLLM-Ascend 代码对应

#### 上游 vLLM 的窗口公式

`D:\Code\vllm-github\vllm\models\deepseek_v41\common\ops\cache_utils.py:1516-1528` 对每一个 Prefill token 计算：

```python
pos = start_pos + token_idx_in_query
swa_start_pos = max(pos - (WINDOW_SIZE - 1), 0)
swa_len = pos - swa_start_pos + 1
```

这对应：

$$
[\max(0,i-W+1),i].
$$

代码是为全部 Prefill token 构建局部窗口，并没有只为最后 `window_size` 个 token 建立 Query。

#### Ascend 的窗口与缓存写入

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:379` 从配置读取 `sliding_window`；
- 同文件 `:618-654` 创建每层 SWA Cache，并把窗口传给稀疏注意力；
- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:849-877` 设置因果模式和 `ori_win_left=window_size-1`；
- 同文件 `:283-291` 对当前批次的 hidden states 投影 KV，并通过 `scatter_cache_sk` 写入本层 SWA Cache。

当窗口为 128 时：

```text
左侧历史 = 127
当前位置 = 1
合计     = 128
```

---

### 21.7 Bounded Replay 是特殊情况

如果完整 1000-token Global KV 已命中，但 Encoder SWA KV 丢失，Encoder Bounded Replay 会设置：

$$
s=1000-128=872.
$$

此时只回放位置 872～999，位置 0～871 不重新执行 SWA。上游：

- `D:\Code\vllm-github\vllm\v1\kv_cache_interface.py:956-962` 令 `prefix_replay_tokens=sliding_window`；
- `D:\Code\vllm-github\vllm\v1\core\sched\scheduler.py:3114-3132` 计算 `replay_start=num_hit_tokens-num_replay_tokens`。

但是 Replay 会在 $s$ 处截断 SWA：

$$
\mathcal L_i^{\text{replay}}
=[\max(s,i-W+1),i].
$$

| Query | 完整 Prefill 窗口 | 从 872 开始的 Replay 窗口 |
|---:|---|---|
| 872 | `[745,872]` | `[872,872]` |
| 873 | `[746,873]` | `[872,873]` |
| 900 | `[773,900]` | `[872,900]` |
| 999 | `[872,999]` | `[872,999]` |

`cache_utils.py:1525-1528` 先按普通窗口计算起点，再用 `replay_start` 对起点取最大值，正是这一截断公式。因此 Bounded Replay 是训练适配过的近似恢复，并不等价于完整 Prefill。

---

### 问题 21 最终总结

对于长度 1000、窗口 128 的首次完整 Causal Encoder Prefill：

1. 位置 0～999 全部作为 Query 执行 SWA；
2. 每个 Query 最多读取自己和左侧 127 个位置；
3. 位置 999 只直接读取位置 872～999；
4. Prefill 完成后，逻辑上只需继续保留最后约 128 个 SWA KV；
5. 位置 0～871 虽然退出尾窗，但其输出已进入后层 hidden states 和 Global KV；
6. 只有 Prefix Cache 命中后的 Bounded Replay 才只重算 872～999，并在 872 截断窗口。

一句话概括：

> `window=128` 限制每个 token 的局部注意范围，不限制首次 Prefill 需要处理的 token 总数。

---

## 问题 22：第 2 层 Full 层需要计算哪些位置的 SWA

### 问题 22 原文

> 继续上一个问题，对于 Prefill 阶段的 SWA，window 长度是 128。如果输入序列长度是 1000，那么对于 Causal Encoder 的第 2 层 Full 层，哪些位置需要计算 SWA？

### 问题 22 核心结论

对于一次没有 Prefix Cache 命中的完整 Prefill，第 2 层虽然是 CSA2 **Full Mode**，但仍然需要对位置：

$$
0,1,2,\ldots,999
$$

全部计算 SWA。

Full 并不是 Full Attention，也不表示 SWA 窗口扩大为 1000。Full 的含义是：这一层除了计算自己的 Main Q 和 SWA KV，还负责完整执行 CSA2 全局分支，即生成 Main KV、Indexer K，并执行新的 Top-K 检索。

第 2 层与第 0、1 层在“哪些 Query 位置需要做 SWA”这一点上完全相同：都是 0～999。区别在于：

```text
Layer 0、1：纯 SWA

Layer 2 Full：SWA
              + ratio=2 Main KV 压缩
              + Indexer K
              + 新 Top-K
              + Selected Global/Main KV Attention
```

---

### 22.1 为什么第 2 层是 Full 层

DeepSeek-V4.1-Flash 官方配置的关键部分是：

```json
sliding_window = 128

compress_ratios = [0, 0, 2, 2, ...]

kv_source_layer_ids = [2, 8, 14, 20]

index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
```

因此：

| 层 | 压缩比 | KV source | Index source | 模式 |
|---:|---:|---:|---:|---|
| 0 | 0 | 否 | 否 | 纯 SWA |
| 1 | 0 | 否 | 否 | 纯 SWA |
| 2 | 2 | 是 | 是 | Full |
| 3～7 | 2 | 否 | 否 | Reuse |
| 8 | 2 | 是 | 是 | Full |

第 2 层同时出现在 KV source 和 Index source 列表中，所以它既生成新的 Main KV，又生成 Indexer K 并执行 Indexer，是第一个 Full 层。

上游 `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:303-348`：

- 读取每层 `compress_ratios`；
- 判断本层是否属于 `kv_source_layer_ids`；
- 判断本层是否属于 `index_source_layer_ids`；
- 消费层则寻找不高于当前层的最近 source。

Ascend 的 `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:498-535` 也以相同方式建立不可变 layer topology。

---

### 22.2 Full 不是 Full Attention

论文 Figure 4 的 Full、Reindex、Reuse 区分的是：

> Main KV、Indexer K、Top-K indices 从哪里取得。

三种模式都会计算当前层自己的：

- Main Q；
- Layer-local SWA KV；
- Attention 输出。

所以：

```text
Full Mode
    ≠ 对 1000 个位置执行普通 Dense Full Attention
    ≠ 把 SWA 窗口从 128 扩大到 1000
    ≠ 只计算最后 128 个 Query

Full Mode
    = 完整承担 CSA2 全局缓存与检索职责
      + 照常执行窗口为 128 的本层 SWA
```

---

### 22.3 第 2 层的 SWA 位置与窗口

首次完整 Prefill 时，第 2 层需要计算的 Query 位置是：

$$
\boxed{0\sim999\text{ 全部位置}}
$$

位置 $i$ 的局部窗口仍为：

$$
\mathcal L_i=[\max(0,i-127),i].
$$

| 第 2 层 Query | 第 2 层 SWA 读取的 K/V | 长度 |
|---:|---|---:|
| 0 | `[0]` | 1 |
| 1 | `[0,1]` | 2 |
| 2 | `[0,1,2]` | 3 |
| 127 | `[0,127]` | 128 |
| 128 | `[1,128]` | 128 |
| 500 | `[373,500]` | 128 |
| 871 | `[744,871]` | 128 |
| 872 | `[745,872]` | 128 |
| 999 | `[872,999]` | 128 |

Full 只增加全局分支的工作，不改变这个 SWA 窗口公式。

---

### 22.4 第 2 层 Full 的完整数据流

第 2 层接收第 1 层的全部输出：

$$
H^{(2)}=[h_0^{(2)},h_1^{(2)},\ldots,h_{999}^{(2)}].
$$

其逻辑数据流是：

```text
第 1 层输出 H[0:999]
          │
          ├────────────────────────────────────────┐
          │                                        │
          ▼                                        ▼
    Layer-local SWA 分支                       Global/Main 分支
          │                                        │
    对 0～999 生成 Query                     ratio=2 压缩
    对 0～999 生成 SWA KV                   生成 Main KV
          │                                        │
    每个 Q 看最多 128 个位置                  生成 Indexer K
          │                                        │
          │                                  为 Query 生成新 Top-K
          │                                        │
          └──────── Local SWA + Selected Main KV ──┘
                              │
                        联合 Core Attention
                              │
                      第 2 层输出 H'[0:999]
```

因此它不是在 SWA 与 Global 之间二选一，而是：

$$
\text{Layer 2 Attention}
=\text{Local SWA}
+\text{Selected Compressed Global KV}.
$$

两类 Logit 在算子中按联合 Attention/Softmax 语义组合，而不是各自 Softmax 后简单相加。

---

### 22.5 Ratio 等于 2 的 Main KV 与 Top-K

第 2 层的压缩比为 2，每两个相邻 token 形成一条 Main KV：

```text
原始位置 0、1       → Main KV C0
原始位置 2、3       → Main KV C1
原始位置 4、5       → Main KV C2
...
原始位置 998、999   → Main KV C499
```

长度 1000 最终得到：

$$
1000/2=500
$$

条 Main KV。

因果执行中，只有分组内 token 都已出现时，相应 compressed entry 才完整可用。对 Query 位置 $i$，可用的 ratio=2 条目数近似为：

$$
N_C(i)=\left\lfloor\frac{i+1}{2}\right\rfloor.
$$

`D:\Code\vllm-github\vllm\models\deepseek_v41\common\ops\cache_utils.py:1529-1530` 实现为：

```python
topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)
```

例如：

| Query 位置 | 已完成的 ratio=2 Main KV 数 |
|---:|---:|
| 0 | 0 |
| 1 | 1 |
| 2 | 1 |
| 3 | 2 |
| 99 | 50 |
| 499 | 250 |
| 999 | 500 |

模型 `index_topk=512`。本例最后只有 500 条完整 compressed entries，因此位置 999 的候选数量小于 Top-512 上限。

位置 999 的两条注意力支路可以直观表示为：

```text
Local SWA：
原始位置 872～999，共 128 条细粒度 KV

Global/Main：
覆盖位置 0～999 的 500 条 ratio=2 压缩 KV

                 ↓
         联合参加 Core Attention
```

---

### 22.6 第 2 层 SWA 计算量

第 2 层的 SWA Query-Key 配对数与第 0、1 层相同：

$$
C_{\text{SWA}}
=\sum_{i=0}^{999}\min(i+1,128)
=119872.
$$

其中：

$$
1+2+\cdots+128=8256,
$$

$$
872\times128=111616.
$$

完整因果 Dense Attention 的配对数为：

$$
C_{\text{dense}}=\frac{1000\times1001}{2}=500500.
$$

所以第 2 层 SWA 分支的理论配对数约为 Dense Causal Attention 的：

$$
\frac{119872}{500500}\approx23.95\%.
$$

但第 2 层总计算量仍高于第 0、1 层，因为它还执行 Main KV 压缩、Indexer Q/K、Top-K 与 Selected Main KV Attention。

---

### 22.7 Prefill 结束后留下哪些缓存

首次 Prefill 期间，第 2 层位置 0～999 全部生成和使用过 SWA KV。但当处理到位置 999 后，供后续 Decode 使用的逻辑有效 SWA 尾窗主要是：

$$
872\sim999.
$$

第 2 层作为 Full/KV source 层还会保留可供后续层共享的长程缓存：

```text
Layer 2
│
├── 长程共享缓存
│   ├── 500 条 ratio=2 Main KV
│   └── 对应的 Indexer K
│
├── 本层生成的索引结果
│   └── 新的 Top-K indices
│
└── 本层局部缓存
    └── 最后约 128 个 token 的 SWA KV
```

这里的 128 是逻辑有效窗口。实际物理页数量还受 block size、页对齐、Chunked Prefill 和 allocator 回收时机影响，不一定精确等于 128 个独立物理槽。

---

### 22.8 代码实现证据

#### 上游 vLLM

`D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:557-568` 对当前层创建自己的 SWA Cache，不取决于它是 Full、Reindex 还是 Reuse。

随后 `:573-585` 才判断：

```python
if self.is_kv_source:
    self.compressor = DeepseekCompressor(...)
```

所以结构是：

```text
所有 CSA2 层：都有自己的 SWA Cache

KV source / Full 层：额外拥有 Compressor 和 Main KV Cache
```

#### vLLM-Ascend

`D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:283-291`：

```python
kv = self._project_kv(attn, hidden_states, cos, sin)
scatter_cache_sk(... swa_cache ..., kv)

if self.role.is_kv_source:
    self._write_compressed_source(...)
```

其顺序清楚表明：

1. 为当前有效 hidden states 生成并写入本层 SWA KV；
2. 如果是 KV source，再额外生成 compressed Main KV。

同文件 `:849-877` 读取窗口并设置：

```python
ori_win_left = window_size - 1
ori_win_right = 0
```

当窗口为 128 时，本层每个 Query 最多使用左侧 127 个历史位置和当前自身。

---

### 22.9 Bounded Replay 时答案如何变化

以上结论针对首次、未命中 Prefix Cache 的完整 Encoder Prefill。

如果 Global KV 已命中、Encoder SWA 丢失，Encoder SWA Bounded Replay 只回放：

$$
872\sim999.
$$

此时第 2 层只对这 128 个 Query 重新执行 SWA；位置 0～871 不重算。窗口还会在 replay 起点 $s=872$ 截断：

$$
\mathcal L_i^{\text{replay}}
=[\max(872,i-127),i].
$$

| Query | 完整 Prefill 窗口 | Bounded Replay 窗口 |
|---:|---|---|
| 872 | `[745,872]` | `[872,872]` |
| 873 | `[746,873]` | `[872,873]` |
| 900 | `[773,900]` | `[872,900]` |
| 999 | `[872,999]` | `[872,999]` |

Replay 的 Prefix token 只用于重建 SWA；已经命中的 Global/Main KV 不应重新计算或覆盖。因此：

```text
首次完整 Encoder Prefill：
第 2 层对 0～999 全部计算 SWA

Encoder Bounded Replay：
第 2 层只重新计算 872～999 的 SWA
```

---

### 问题 22 最终总结

对于输入长度 1000、窗口 128 的首次完整 Prefill，第 2 层 Full 层需要计算 SWA 的位置是：

$$
\boxed{0,1,2,\ldots,999}
$$

每个位置 $i$ 的窗口为：

$$
\boxed{[\max(0,i-127),i]}.
$$

第 2 层作为 Full 层比纯 SWA 的第 0、1 层额外执行：

1. ratio=2 Main KV 压缩，1000 个 token 形成 500 条 Main KV；
2. 从 Main KV 投影 Indexer K；
3. 计算本层 Indexer Q；
4. 执行新的 Top-K 检索；
5. 将 Selected Main KV 与本层 SWA KV 联合送入 Core Attention。

最需要记住的是：

> Full 描述 CSA2 全局缓存与索引职责，不表示 SWA 变成全长 Attention；第 2 层仍对全部 1000 个 Query 执行窗口为 128 的 SWA。

---

## 问题 23：第 3～7 层 Reuse 层需要计算哪些位置的 SWA

### 问题 23 原文

> 继续上一个问题，对于 prefill 阶段的 SWA，win 长度是 128，如果输入序列长度是 1000，那么对于 causal encoder 的第 3～7 层 Reuse 层，哪些位置需要计算 SWA？

### 问题 23 核心结论

对于**首次、未命中 Prefix Cache 的完整 Prefill**，代码索引第 3～7 层虽然是 Reuse 层，但它们仍然必须为全部 1000 个 Query 位置计算本层自己的 SWA：

$$
\boxed{0,1,2,\ldots,999}.
$$

窗口长度 128 限制的是每个 Query 能看见多少个局部 Key，而不是只让最后 128 个 Query 进入该层。对于位置 $i$：

$$
\boxed{\mathcal L_i=[\max(0,i-127),i]}.
$$

“Reuse”复用的是第 2 层提供的压缩 Main KV 和 Top-K 索引结果，不复用第 2 层的 Query、hidden state 或 SWA KV。第 3、4、5、6、7 层都必须根据自己的输入 hidden states 生成各自独立的 SWA KV。

只有进入 **Encoder SWA Bounded Replay** 的特殊路径时，这些层才只重新处理尾部位置 872～999。

---

### 23.1 为什么第 3～7 层是 Reuse 层

DeepSeek-V4.1-Flash 的关键配置为：

```python
kv_source_layer_ids = [2, 8, 14, 20]
index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
compress_ratios = [0, 0] + [2] * 18 + [1] * 20
sliding_window = 128
```

在第 2～7 层这一组中：

| 层索引 | 模式 | Main KV 来源 | Top-K 来源 | 本层 SWA |
|---:|---|---|---|---|
| 2 | Full | 本层生成 | 本层搜索 | 本层生成 |
| 3 | Reuse | 复用第 2 层 | 复用第 2 层 | 本层生成 |
| 4 | Reuse | 复用第 2 层 | 复用第 2 层 | 本层生成 |
| 5 | Reuse | 复用第 2 层 | 复用第 2 层 | 本层生成 |
| 6 | Reuse | 复用第 2 层 | 复用第 2 层 | 本层生成 |
| 7 | Reuse | 复用第 2 层 | 复用第 2 层 | 本层生成 |

直到第 8 层才再次生成新的 Main KV，并重新建立新的索引结果。

---

### 23.2 Reuse 复用什么、不复用什么

Reuse 层的完整语义可以写成：

$$
\text{Reuse Layer}
=\text{Shared Main KV}
+\text{Shared Top-K Indices}
+\text{Layer-local SWA}
+\text{Layer-local Transformer Computation}.
$$

具体区分如下：

| 内容 | 是否复用第 2 层结果 | 原因 |
|---|---|---|
| 压缩 Main KV Cache | 是 | 这是 CSA2 的跨层共享对象 |
| Indexer K Cache | 是 | 与 Main KV 条目一一对应，由 source 层持有 |
| Top-K indices | 是 | Reuse 层不重新搜索 |
| 当前层 Main Query | 否 | 每层 hidden state 和投影权重不同 |
| 当前层 SWA Query | 否 | 来自本层输入 hidden state |
| 当前层 SWA KV | 否 | 必须由本层 hidden state 生成 |
| Attention 输出 | 否 | 每层都要执行自己的 Attention |
| MoE/FFN 输出 | 否 | 每层参数不同 |

因此，Reuse 绝不等于“跳过这一层”，也不等于“把上一层 SWA KV 拿来直接用”。

---

### 23.3 首次完整 Prefill 的 SWA 计算位置

对于输入长度 1000，每个位置都会作为一次 Query：

```text
Query 位置：0, 1, 2, ..., 871, 872, ..., 999
             └──────── 全部需要进入每个 Reuse 层 ────────┘
```

每个 Query 的局部窗口为：

| Query 位置 | 本层 SWA 能直接看见的 Key 位置 |
|---:|---|
| 0 | 0 |
| 1 | 0～1 |
| 10 | 0～10 |
| 127 | 0～127 |
| 128 | 1～128 |
| 500 | 373～500 |
| 871 | 744～871 |
| 872 | 745～872 |
| 999 | 872～999 |

位置 0～871 不但需要计算，而且它们的输出还会成为下一层相同位置的输入：

```text
Layer 2 输出 H²[0:999]
          │
          ▼
Layer 3 计算 Q/KV/SWA/FFN，得到 H³[0:999]
          │
          ▼
Layer 4 计算 Q/KV/SWA/FFN，得到 H⁴[0:999]
          │
         ...
          ▼
Layer 7 计算 Q/KV/SWA/FFN，得到 H⁷[0:999]
```

如果省略位置 0～871，那么下一层就缺少这些位置对应的 hidden states，后续第 8 层也无法为整个 prompt 建立正确的压缩 Main KV。

---

### 23.4 第 3～7 层的数据流

以第 3 层为例：

```text
第 3 层输入 H²[0:999]
        │
        ├── 本层 Q/KV 投影
        │     ├── Q³[0:999]
        │     └── SWA KV³[0:999]
        │
        ├── 读取第 2 层共享的 Main KV
        ├── 读取第 2 层共享的 Top-K indices
        │
        └── 对每个 Query 联合计算：
              Local SWA³
                    +
              Selected Main KV²
                    │
                    ▼
              Attention Output³
                    │
                    ▼
                 MoE/FFN³
                    │
                    ▼
                 H³[0:999]
```

第 4～7 层的过程相同，只是输入 hidden state 和当前层投影权重不同。

---

### 23.5 SWA 计算量

单层的局部 Query-Key 配对数为：

$$
C_{\text{SWA,1 layer}}
=\sum_{i=0}^{999}\min(i+1,128)
=8256+872\times128
=119872.
$$

第 3～7 层共有 5 层，因此：

$$
C_{\text{SWA,3--7}}
=5\times119872
=599360.
$$

这里统计的是局部 SWA score 配对数，不包括：

- Query/KV 投影；
- Selected Main KV Attention；
- 输出投影；
- RMSNorm；
- MoE/FFN。

---

### 23.6 Bounded Replay 时答案如何变化

如果讨论的不是首次完整 Prefill，而是论文中的 Encoder SWA Bounded Replay，那么 replay 起点为：

$$
s=1000-128=872.
$$

此时第 3～7 层只重新处理：

$$
\boxed{872,873,\ldots,999}.
$$

并且窗口被截断在 replay segment 内：

$$
\mathcal L_i^{\text{replay}}
=[\max(872,i-127),i].
$$

例如：

| Query | 完整 Prefill 的局部窗口 | Bounded Replay 窗口 |
|---:|---|---|
| 872 | 745～872 | 872 |
| 873 | 746～873 | 872～873 |
| 900 | 773～900 | 872～900 |
| 999 | 872～999 | 872～999 |

所以必须先明确讨论的是哪一种场景：

```text
首次完整 Prefill：第 3～7 层计算 0～999
Encoder Bounded Replay：第 3～7 层只重算 872～999
```

---

### 23.7 代码实现证据

上游 vLLM 在每个 DeepSeek-V4.1 Attention 层中都会创建独立的 SWA cache；随后才根据 `is_kv_source` 决定是否额外创建 compressor。对应：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:557-568`：所有层创建本层 SWA cache；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:573-585`：只有 KV source 层创建 Main KV compressor。

当前 vLLM-Ascend 同样先为本层投影并写入 SWA KV，再根据 `role.is_kv_source` 决定是否写入压缩 Main KV：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\attention\dsa_v41.py:283-291`。

这说明 Reuse 层没有省略本层 SWA。

---

### 问题 23 最终总结

对于长度 1000、窗口 128 的首次完整 Prefill：

$$
\boxed{\text{第 3～7 层均计算位置 }0～999\text{ 的 SWA。}}
$$

每个位置只查看最多 128 个局部 Key：

$$
\boxed{\mathcal L_i=[\max(0,i-127),i]。}
$$

第 3～7 层复用第 2 层的 Main KV 和 Top-K，但各自重新生成 Q、SWA KV、Attention 输出和 MoE/FFN 输出。只有在 Encoder SWA Bounded Replay 场景下，它们才只重算 872～999。

---

## 问题 24：第 20 层 Full 层需要计算哪些位置的 SWA

### 问题 24 原文

> 继续上一个问题，对于 prefill 阶段的 SWA，win 长度是 128，如果输入序列长度是 1000，那么对于模型的第 20 层 Full 层，哪些位置需要计算 SWA？

### 问题 24 核心结论

代码索引第 20 层是最后一个 KV source，也是 decoder 部分的第一个 Full 层。这里需要区分三个口径：

1. **论文的理想 CED 计算公式**希望 decoder 只 replay 最后 $n_{\text{win}}=128$ 个位置，因此从算法目标看，第 20～39 层只需要处理 872～999；
2. **当前上游 vLLM NVIDIA 实现**把第 20 层作为最后一个完整 KV source，在裁剪 replay batch 之前执行它，因此第 20 层实际处理 0～999，第 21～39 层才处理 872～999；
3. **当前 vLLM-Ascend 实现**还没有 decoder bounded replay 裁剪，所有层都继续处理 0～999。

因此，如果问题问的是当前本地 vLLM/vLLM-Ascend 代码的实际行为，第 20 层对全部位置计算本层 SWA：

$$
\boxed{0,1,2,\ldots,999}.
$$

每个 Query 的窗口仍然是：

$$
\boxed{[\max(0,i-127),i]}.
$$

---

### 24.1 为什么第 20 层是 Full 层

配置指定：

```python
kv_source_layer_ids = [2, 8, 14, 20]
index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer_id = 20
compress_ratios[20] = 1
```

所以第 20 层同时承担：

- 生成新的 Main KV；
- 生成对应 Indexer K；
- 对全部因果可见 Main KV 做新的 Top-K 搜索；
- 生成供后续层使用的层级候选池；
- 生成第 20 层自己的 SWA KV；
- 执行第 20 层 Attention 与 MoE/FFN。

它是 decoder 后半段共享状态的根节点：

```text
Layer 20 Full
├── Main KV source for layers 20～39
├── Candidate source for hierarchical indexer
├── Top-K source for layers 20～23
└── Layer-local SWA KV for layer 20 itself
```

---

### 24.2 论文理想 CED 与当前上游实现的口径

论文给出的 CED prefill 近似计算量为：

$$
C_{\text{CED}}
=N\frac{L}{2}+n_{\text{win}}\frac{L}{2}.
$$

当 $N=1000$、$L=40$、$n_{\text{win}}=128$ 时，理想语义是：

```text
Encoder 20 层：处理 1000 个 token
Decoder 20 层：只 replay 最后 128 个 token
```

即 decoder 第一个层在算法概念上也属于尾窗 replay。

但工程实现还要在 encoder 输出 $H^{20}$ 上构造供 decoder 共享的 Main KV、Indexer K 与候选池。当前上游 vLLM 选择让最后一个 KV source 层 20 仍处在完整 batch 区域，直到它生成共享状态后，才把 token batch 裁剪到最后 128 个位置。

因此当前实现的分界不是：

```text
0～19 full，20～39 replay
```

而是：

```text
0～20 full，21～39 replay
```

这是论文理想复杂度与当前工程实现之间的一层边界差异。

---

### 24.3 上游 vLLM 中第 20 层的实际计算范围

上游 NVIDIA 实现首先取最后一个 KV source：

```python
cut = max(config.kv_source_layer_ids)
```

这里：

$$
cut=20.
$$

然后把 decoder replay 起点设置为：

```python
self.decoder_replay_start = cut + 1
```

所以：

$$
\text{decoder\_replay\_start}=21.
$$

对应源码：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:757-768`。

完整 batch 先运行：

```python
self._run_layers(
    range(self.start_layer, self.decoder_replay_start),
    ...
)
```

对应 `model.py:965-977`。当 start layer 为 0、replay start 为 21 时，Python 的 `range(0, 21)` 包含 0～20，因此第 20 层仍收到全部 1000 个位置。

第 20 层位置 $i$ 的 SWA 窗口为：

$$
\mathcal L_i=[\max(0,i-127),i].
$$

例如：

| Query | 第 20 层 SWA Key 范围 |
|---:|---|
| 0 | 0 |
| 127 | 0～127 |
| 128 | 1～128 |
| 871 | 744～871 |
| 872 | 745～872 |
| 999 | 872～999 |

---

### 24.4 第 20 层完整数据流

```text
Encoder 最终输出 H²⁰[0:999]
             │
             ▼
Layer 20 Full，处理 0～999
├── 生成本层 Q[0:999]
├── 生成本层 SWA KV[0:999]
├── ratio=1 生成 Main KV[0:999]
├── 投影 Indexer K[0:999]
├── 搜索本层 Top-K
├── 构造共享候选块
├── 联合 Local SWA + Selected Main KV
└── 生成 H²¹[0:999]
             │
             └── bounded replay 仅保留 H²¹[872:999]
                              │
                              ▼
                   Layer 21～39 replay
```

由于 ratio=1，第 20 层的 Main KV 不再把两个 token 合并为一个：每个输入位置对应一个 Main KV 条目。

---

### 24.5 当前 vLLM-Ascend 的行为

当前 Ascend 模型 forward 先记录完整 token 数：

```python
full_num_tokens = positions.shape[0]
```

随后直接遍历所有层，并把同一个 `positions` 和当前完整 token 轴继续传给下一层：

```python
for layer in self.layers:
    ...
    hidden_states, pre_mix = layer(
        positions, hidden_states, pre_mix, ...
    )
```

对应：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:1231-1234`；
- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:1254-1275`。

当前该目录没有 NVIDIA 实现里的 `decoder_replay_start` 或 `DecoderReplayLayers` 裁剪逻辑。因此第 20 层以及后续层当前都会处理 0～999。

---

### 问题 24 最终总结

对于输入长度 1000、窗口 128：

| 口径 | 第 20 层处理的位置 |
|---|---|
| 论文理想 CED Decoder Replay | 872～999 |
| 当前上游 vLLM NVIDIA 实现 | 0～999 |
| 当前 vLLM-Ascend 实现 | 0～999 |

当前上游之所以让第 20 层仍执行完整位置，是因为它是最后一个 KV source 和候选池 source；完成它之后，才从第 21 层开始把 decoder batch 裁剪成最后 128 个位置。

---

## 问题 25：第 21～23 层 Reuse 的 SWA 位置与缓存空间

### 问题 25 原文

> 继续上一个问题，对于 prefill 阶段的 SWA，win 长度是 128，如果输入序列长度是 1000，那么对于模型的第 21～23 层 Reuse 层，哪些位置需要计算 SWA？SWA KV cache 占用的空间大小是多少？是每层 66 KB 吗？

### 问题 25 核心结论

- 按论文的 Decoder SWA Bounded Replay 设计，第 21、22、23 层都只计算最后 128 个提示词，即位置 **872～999**。
- 上游 vLLM NVIDIA 在 bounded replay 已启用且走可裁剪路径时也是如此。
- 这三层虽然是 Reuse 层，但复用的是第 20 层生成的 Main KV 和 Top-K；每一层自己的 Query、SWA KV、Attention 和 FFN 仍要计算。
- “每层 66 KB”在特定格式下准确：上游 vLLM 在 NVIDIA SM100 上采用的 528 byte/token MXFP8 SWA 记录满足：

$$
128\times528=67584\text{ bytes}=66\text{ KiB}.
$$

严格说应是 **66 KiB/层/请求**，而不是十进制 66 KB。

- 当前 vLLM-Ascend 代码没有实现 decoder bounded replay 的 token 裁剪，因此当前代码实际会让第 21～23 层继续处理位置 0～999；而且 `cache_dtype=auto` 时通常是 BF16，对应约 **128 KiB/层/请求**，不是 66 KiB。

---

### 25.1 第 21～23 层的角色

官方配置指定：

```python
num_hidden_layers = 40
sliding_window = 128
kv_source_layer_ids = [2, 8, 14, 20]
index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer_id = 20
```

第 21、22、23 层既不是新的 KV source，也不是新的 Index source，因此它们属于第 20 层之后的 Reuse 层：

```text
Layer 20：Full
  ├── 生成/保存 Main KV
  ├── 生成 Indexer K
  ├── 搜索 Top-K
  └── 生成 Layer 20 自己的 SWA KV
          │
          ├──────── 共享 Main KV、Top-K ────────┐
          ▼                                     │
Layer 21：Reuse ── 自己计算 Q 和自己的 SWA KV   │
Layer 22：Reuse ── 自己计算 Q 和自己的 SWA KV   │
Layer 23：Reuse ── 自己计算 Q 和自己的 SWA KV   │
          │                                     │
          ▼                                     │
Layer 24：Reindex ── 重新打分产生新的 Top-K，但仍复用 Main KV
```

所以更准确的公式是：

$$
\text{Reuse}
=\text{复用 Main KV 和索引结果}
+\text{计算本层 Q、本层 SWA KV、本层 Attention、本层 FFN}.
$$

---

### 25.2 Bounded Replay 中计算哪些位置

设：

$$
L_{\text{prompt}}=1000,
\qquad n_{\text{win}}=128.
$$

Replay 起点为：

$$
s=L_{\text{prompt}}-n_{\text{win}}
=1000-128
=872.
$$

因此送入 decoder bounded replay 的位置集合是：

$$
\boxed{\{872,873,\ldots,999\}}.
$$

当前上游 NVIDIA 实现取：

```python
cut = max(config.kv_source_layer_ids)  # 20
self.decoder_replay_start = cut + 1    # 21
```

所以层 0～20 先走完整 token batch，层 21～39 再走 replay batch。对应：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:757-793`；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:965-988`。

模型状态构造 replay batch 时，对每个请求保留：

$$
\text{kept\_len}
=\min(\text{query\_len},\text{window})
=128.
$$

对应：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model_state.py:463-469`。

因此：

| 层索引 | 模式 | Prefill 中进入该层的位置 |
|---:|---|---|
| 20 | Full | 0～999 |
| 21 | Reuse | 872～999 |
| 22 | Reuse | 872～999 |
| 23 | Reuse | 872～999 |
| 24 | Reindex | 872～999 |

---

### 25.3 Replay 内部的 SWA 窗口

论文给出的 bounded replay 局部范围为：

$$
\mathcal L_i^{\text{replay}}
=[\max(s,i-W+1),i],
$$

其中：

- $s=872$：replay 起点；
- $W=128$：SWA 窗口；
- $i$：当前 Query 的绝对位置。

因此：

| Query 位置 $i$ | 普通完整 SWA 原本看见 | Bounded Replay 实际看见 |
|---:|---|---|
| 872 | 745～872 | 872 |
| 873 | 746～873 | 872～873 |
| 900 | 773～900 | 872～900 |
| 998 | 871～998 | 872～998 |
| 999 | 872～999 | 872～999 |

位置 872 的标准 SWA 原本依赖位置 745～871，但 bounded replay 没有重放这些位置，所以它只能看见位置 872。越接近 999，缺失的局部历史越少；位置 999 的 replay 窗口与完整 SWA 窗口相同。

第 22、23 层收到的是前一 replay 层产生的 128 行 hidden states：

```text
Layer 20 完整输出：[h₀²⁰, h₁²⁰, ..., h₉₉₉²⁰]
                           │
                           └─ 截取 h₈₇₂²⁰ ... h₉₉₉²⁰
                                      │
Layer 21 replay：                 872 ... 999
                                      │
Layer 22 replay：                 872 ... 999
                                      │
Layer 23 replay：                 872 ... 999
```

所以第 22、23 层不会恢复位置 0～871 的 layer-local SWA states。这也是论文所说 reconstructed decoder SWA KV 不与完整 decoder forward 数学等价的原因。

不过它们仍可通过第 20 层共享的压缩 Main KV 和 Top-K 取得长距离信息：

$$
\text{Attention Context}
=\underbrace{\text{Selected Global Main KV}}_{\text{长上下文}}
\mathbin{\Vert}
\underbrace{\text{Layer-local Replay SWA KV}}_{\text{位置 }872\text{ 到当前}}.
$$

---

### 25.4 三层的 SWA 计算量

Replay segment 内各位置看见的局部 KV 数依次为：

$$
1,2,3,\ldots,128.
$$

因此每层的 SWA Query-Key 配对数为：

$$
1+2+\cdots+128
=\frac{128\times129}{2}
=8256.
$$

第 21～23 三层合计：

$$
3\times8256=24768.
$$

对比完整 1000-token SWA，每层的配对数为：

$$
\sum_{i=0}^{999}\min(i+1,128)
=8256+872\times128
=119872.
$$

Replay 每层局部配对数约为完整 SWA 的：

$$
\frac{8256}{119872}\approx6.89\%.
$$

这只是局部 Attention score 的配对数，不包括 Main KV sparse attention、投影、输出层与 MoE/FFN 的成本。

---

### 25.5 每层 66 KiB 的推导

DeepSeek-V4.1 的 SWA KV 使用 MLA 风格的单个 512-channel KV latent：

$$
d_{\text{KV-latent}}=512,
\qquad n_{\text{KV-head}}=1.
$$

它不是传统 MHA 中分别缓存一个 512 维 K 和一个 512 维 V，因此不能再机械地乘以 2。

论文明确保留 FP8 SWA KV。如果暂时忽略 scale，纯 FP8 数据部分为：

$$
512\text{ channels}\times1\text{ byte}=512\text{ bytes/token}.
$$

128 token 的原始数据为：

$$
128\times512=65536\text{ bytes}=64\text{ KiB}.
$$

上游 vLLM 在 NVIDIA SM100 路径中使用 V4.1 MXFP8 SWA record：

- 512 个 E4M3 FP8 数据：512 bytes；
- 每 32 个通道一个 UE8M0 scale；
- scale 数量：$512/32=16$；
- 每个 scale 1 byte：16 bytes。

所以每 token：

$$
S_{\text{token}}=512+16=528\text{ bytes}.
$$

每层 128-token SWA：

$$
S_{\text{layer}}
=128\times528
=67584\text{ bytes}
=66\text{ KiB}.
$$

代码直接定义：

```python
self.swa_bytes_per_token = 528 if self.kv_mxfp8 else 584
```

对应：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:513-518`。

512 个 FP8 数据加 16 个 UE8M0 scale 的格式说明位于：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\attention.py:122-129`。

所以“每层 66 KB”应精确表述为：

> 在 NVIDIA SM100 的 V4.1 MXFP8 packed SWA cache 格式下，每层、每请求、128 个有效 token 的逻辑容量为 66 KiB。

---

### 25.6 不同缓存格式的容量对比

不同实现并不都使用 528-byte record：

| SWA 存储格式 | 每 token | 128 token/层 |
|---|---:|---:|
| 原始 FP8 latent，不计 scale | 512 B | 64 KiB |
| V4.1 MXFP8：512 FP8 + 16 UE8M0 scale | 528 B | 66 KiB |
| 旧 V4 packed 格式 | 584 B | 73 KiB |
| BF16 latent | 1024 B | 128 KiB |

旧 V4 packed record 包含：

- 448 个 FP8 NoPE 通道；
- 64 个 BF16 RoPE 通道；
- 7 个 UE8M0 scale；
- 1 byte padding。

所以：

$$
448+64\times2+7+1=584\text{ bytes/token},
$$

$$
584\times128=74752\text{ bytes}=73\text{ KiB}.
$$

第 21～23 层在 528-byte MXFP8 格式下合计：

$$
3\times66=198\text{ KiB/请求}.
$$

如果 decoder 20 层都按这一格式保存 128-token SWA：

$$
20\times66=1320\text{ KiB}
\approx1.289\text{ MiB/请求}.
$$

这些数字只统计 SWA KV 数据，不包括 Main KV、Indexer K、Top-K indices、候选块、block table、页对齐和 allocator 元数据。

---

### 25.7 当前 vLLM-Ascend 为什么通常不是 66 KiB

Ascend 模型通过以下函数决定 SWA cache dtype：

```python
kv_cache_dtype = kv_cache_dtype_str_to_dtype(
    vllm_config.cache_config.cache_dtype,
    vllm_config.model_config,
)
```

对应：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:617-624`。

当 `cache_dtype == "auto"` 时，公共函数返回模型 dtype：

```python
if kv_cache_dtype == "auto":
    return model_config.dtype
```

对应：

- `D:\Code\vllm-github\vllm\utils\torch_utils.py:525-531`。

官方模型 dtype 为 BF16，因此默认情况下：

$$
\text{SWA dtype}=\text{BF16}.
$$

Ascend SWA cache 的实际页大小公式为：

$$
\text{page bytes}
=\text{storage block size}
\times n_{\text{KV-head}}
\times d_{\text{head}}
\times\operatorname{sizeof}(\text{dtype}).
$$

对应：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\core\kv_cache_interface.py:261-267`。

代入 BF16：

$$
128\times1\times512\times2
=131072\text{ bytes}
=128\text{ KiB/层/请求}.
$$

若显式使用普通 FP8，则纯 latent 数据为：

$$
128\times1\times512\times1=64\text{ KiB}.
$$

当前 Ascend 路径没有采用 NVIDIA SM100 的 528-byte/token V4.1 packed SWA record，所以不能把 66 KiB 直接套用到 Ascend。

此外，当前 Ascend 模型 forward 中没有 decoder replay token 裁剪。它记录完整 `positions.shape[0]` 后，继续以相同 token 轴遍历所有层：

- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:1231-1234`；
- `D:\Code\vllm-ascend\vllm-ascend\vllm_ascend\models\deepseek_v41\model.py:1254-1275`。

因此当前这份 Ascend 代码中，第 21～23 层实际仍计算位置 0～999。

---

### 25.8 论文、上游 NVIDIA 与 Ascend 的最终对比

| 讨论对象 | 第 21～23 层处理的位置 | 典型 SWA 容量/层/请求 |
|---|---|---:|
| 论文 Decoder SWA Bounded Replay | 872～999 | 论文规定 FP8，但未把所有后端物理布局固定成 528 B/token |
| 上游 vLLM NVIDIA SM100，bounded replay 已启用 | 872～999 | 66 KiB |
| 上游 vLLM 旧 V4 packed record | 视 replay 是否启用 | 73 KiB |
| 上游 vLLM bounded replay 被关闭或保留完整 batch 的图模式 | 0～999 | 依后端格式而定 |
| 当前 vLLM-Ascend，`cache_dtype=auto` | 0～999 | 通常 128 KiB |
| 当前 vLLM-Ascend，普通 FP8 cache | 0～999 | 约 64 KiB，加后端相关开销 |

上游 vLLM 也明确限制了 bounded replay 的启用条件；例如部分 Pipeline Parallel、Sequence Parallel、Prefill Context Parallel、microbatch 或 draft 配置下会关闭裁剪，参见：

- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model.py:1156-1193`；
- `D:\Code\vllm-github\vllm\models\deepseek_v41\nvidia\model_state.py:448-469`。

所以不能只看到 `window_size=128` 就断言任何设备、任何运行模式下都只计算尾部 128 个 Query，也不能断言任何实现都固定占用 66 KiB。

---

### 问题 25 最终总结

1. 按 Decoder SWA Bounded Replay，第 21、22、23 层都只处理：

   $$
   \boxed{872～999}.
   $$

2. Reuse 只复用第 20 层的 Main KV 和 Top-K；三层仍各自生成独立 SWA KV。

3. 528-byte MXFP8 record 下：

   $$
   \boxed{128\times528=66\text{ KiB/层/请求}}.
   $$

   第 21～23 层合计为 198 KiB/请求。

4. 66 KiB 不是跨平台固定值：纯 FP8 latent 是 64 KiB，旧 packed record 是 73 KiB，BF16 latent 是 128 KiB。

5. 当前 vLLM-Ascend 默认 BF16 SWA cache 通常约为 128 KiB/层/请求，并且当前实现尚未把第 21～23 层裁剪为只处理最后 128 个位置。

---

## 问题 26：如何理解 Decoder SWA Bounded Replay

### 问题 26 原文

> 论文 3.2.2. SWA Bounded Replay 中的 Decoder SWA Bounded Replay 部分，“at every prefill, we replay the last $n_{\text{win}}$ tokens of the prompt, feed their encoder outputs through the decoder layers under the same SWA truncation, and use the resulting decoder SWA KV only for decoding, not for prefix caching.”，这句话应该如何理解？

### 问题 26 核心结论

这句话描述的是：

> 在进入自回归 Decode 之前，DeepSeek-V4.1-Flash 不让整个 Prompt 经过后半段 20 层 Decoder，而只取 Prompt 最后的 $n_{\text{win}}$ 个 token，把它们在 Causal Encoder 中得到的输出送入 Decoder，临时重建每个 Decoder 层的 SWA KV Cache。

论文 3.2.1～3.2.2 的关键背景是：持久化 Prefix Cache 主要保存 Global KV，不再长期保存 Decoder SWA KV。因此，每次 Prefill 结束、即将开始 Decode 时，都要临时准备一份 Decoder SWA KV。

本节依据：

- `D:\Code\vllm_notes\DeepSeekV41-Flash\DeepSeek-V4.1-Flash Pushing the Limits of KV Cache Compression.pdf`，第 19～20 页，Section 3.2.1 和 Section 3.2.2。

---

### 26.1 原文逐句拆解

| 原文 | 实际含义 |
|---|---|
| `at every prefill` | 每次请求从 Prompt 处理转入 Decode 前，都要确保 Decoder SWA KV 已经被重建 |
| `replay the last nwin tokens` | 只重新运行 Prompt 尾部一个 SWA 窗口，而不是让整个 Prompt 经过 Decoder |
| `feed their encoder outputs through decoder layers` | 输入 Decoder 的不是原始 token embedding，而是这些 token 的最终 Causal Encoder hidden states |
| `under the same SWA truncation` | 每层局部 SWA 都被限制在 replay segment 内，不能访问 replay 起点之前的局部 SWA KV |
| `only for decoding, not for prefix caching` | 重建出来的 SWA KV 是当前请求的临时运行时状态，只服务随后的 Decode，不写入持久化 Prefix Cache |

这五个短语连起来，就是：

```text
完整 Prompt 经过 Encoder
        │
        ├── 构造完整 Global KV
        │
        └── 截取最后 nwin 个 Encoder outputs
                         │
                         ▼
                 运行 Decoder Replay
                         │
                         ▼
             临时生成各 Decoder 层 SWA KV
                         │
                         ▼
                    开始 Decode
```

---

### 26.2 为什么只把 Encoder 输出送入 Decoder

DeepSeek-V4.1-Flash 的 40 层 backbone 可以抽象成：

```text
前 20 层 Causal Encoder                  后 20 层 Decoder
Layer 0～19                             Layer 20～39
      │                                       │
      │ 产生最终 Encoder hidden states        │
      └──────────── Henc ─────────────────────►
```

在普通 decoder-only 模型中，Prompt 的每个 token 通常都要顺序经过全部 40 层。

而在 CED 中：

1. 完整 Prompt 经过前 20 层 Causal Encoder；
2. Decoder 的 Global KV 可以直接从最终 Encoder hidden states 投影得到；
3. 因而不必为了生成 Decoder Global KV，让全部 Prompt 再经过后 20 层；
4. 但 Decoder 每层自己的 SWA KV 不能直接从同一份 $H_{\text{enc}}$ 一次性投影得到。

原因是每个 Decoder 层的 SWA KV 依赖该层自己的输入 hidden states：

$$
KV_{\text{SWA}}^{(l)}
=W_{KV,\text{SWA}}^{(l)}H^{(l-1)}.
$$

其中 $H^{(l-1)}$ 必须通过前面的 Decoder 层逐层计算才能得到：

```text
Encoder 输出 Henc
       │
       ▼
Decoder Layer 20
产生 Layer 20 SWA KV 和 H20-out
       │
       ▼
Decoder Layer 21
产生 Layer 21 SWA KV 和 H21-out
       │
       ▼
Decoder Layer 22
产生 Layer 22 SWA KV
       │
      ...
```

因此，Global KV 可以从 Encoder 输出直接构造；Decoder SWA KV 仍必须运行 Decoder 层才能生成。

---

### 26.3 Replay 不是重新生成 Token

这里的 replay 不表示：

- 重新分词；
- 把这些 token 再追加到 Prompt；
- 重新调用模型生成这 128 个 token；
- 对这些 token 重新采样。

它表示的是：

> 把这些已知 Prompt token 对应的 Encoder hidden states，再作为一个小型 Prefill batch，送入 Decoder 层进行一次确定性的前向计算。

因此它更接近“重新计算尾部 hidden states 和 KV”，而不是“重新生成尾部文本”。

---

### 26.4 1000 Token、窗口 128 的完整流程

设：

$$
N=1000,
\qquad n_{\text{win}}=W=128.
$$

Replay 起点为：

$$
s=N-W=1000-128=872.
$$

所以尾部 replay segment 是：

$$
[872,999].
$$

整体数据流如下：

```text
Prompt token
0 1 2 ... 871 872 873 ... 999
│                │            │
└────── 全部进入 Causal Encoder ──────┘
                         │
                         ▼
Encoder 输出
Henc[0], ..., Henc[871], Henc[872], ..., Henc[999]
                         │
           ┌─────────────┴─────────────────┐
           │                               │
           ▼                               ▼
从全部 Encoder 输出构造              只截取尾部 Encoder 输出
Global/Main KV                      Henc[872:1000]
可持久化 Prefix Cache                    │
                                           ▼
                                  依次经过 Decoder 20 层
                                           │
                                           ▼
                              每层产生自己的 128-token SWA KV
                                           │
                                           ▼
                                   开始生成 token 1000
```

关键点是：

- Global KV 路径覆盖完整 Prompt；
- Decoder replay 路径只处理最后 128 个 Prompt token；
- Decoder 的每一层仍分别生成自己的 SWA KV；
- replay 完成后才具备执行第一个 Decode step 的完整状态。

---

### 26.5 Feed Their Encoder Outputs 的含义

对于位置 $i$，经过前 20 层 Causal Encoder 后得到：

$$
H_i^{\text{enc}}.
$$

Decoder replay 的输入是：

$$
H_{872}^{\text{enc}},
H_{873}^{\text{enc}},
\ldots,
H_{999}^{\text{enc}},
$$

而不是重新从：

$$
E(x_{872}),E(x_{873}),\ldots,E(x_{999})
$$

这些原始 token embeddings 开始运行全部模型。

可以写成：

$$
\left[
H_{872}^{\text{enc}},
\ldots,
H_{999}^{\text{enc}}
\right]
\xrightarrow{\text{Decoder Layer 20～39}}
\text{Decoder SWA KV}.
$$

这也是为什么这项优化能够省掉大量计算：前半段 Encoder 已经完成完整 Prompt 的因果语义编码，Decoder 只需要处理最后一个局部窗口。

---

### 26.6 Same SWA Truncation 的含义

普通完整 Prefill 中，位置 $i$ 的 SWA 范围是：

$$
\mathcal L_i^{\text{full}}
=[\max(0,i-W+1),i].
$$

但 bounded replay 从位置 $s$ 开始，位置 $s$ 之前的 Decoder SWA KV 根本不存在。因此 replay 时必须把窗口左边界截断在 $s$：

$$
\boxed{
\mathcal L_i^{\text{replay}}
=[\max(s,i-W+1),i]
}.
$$

对于：

$$
s=872,
\qquad W=128,
$$

有：

| Query | 完整 Decoder Prefill 应看到 | Bounded Replay 实际看到 |
|---:|---|---|
| 872 | 745～872 | 872 |
| 873 | 746～873 | 872～873 |
| 874 | 747～874 | 872～874 |
| 900 | 773～900 | 872～900 |
| 998 | 871～998 | 872～998 |
| 999 | 872～999 | 872～999 |

Replay segment 内部形成一个三角形因果区域：

```text
Key →
        872 873 874 ... 998 999
Q 872    ●
u 873    ●   ●
e 874    ●   ●   ●
r ...    ●   ●   ●  ...
y 998    ●   ●   ●  ...  ●
↓ 999    ●   ●   ●  ...  ●   ●
```

它不会让位置 872 去访问不存在的 Decoder SWA KV 位置 745～871。

---

### 26.7 为什么重建结果只是近似

假设要得到 Decoder Layer 20 的位置 872 hidden state。

完整 Prefill 时，位置 872 的 SWA 可以看见：

$$
745～872.
$$

但 bounded replay 时它只能看见：

$$
872.
$$

于是：

$$
\widetilde H_{872}^{(20)}
\neq
H_{872}^{(20),\text{full}}.
$$

这个近似 hidden state 又会送入下一层：

$$
\widetilde H_{872}^{(20)}
\rightarrow
\widetilde H_{872}^{(21)}
\rightarrow
\widetilde H_{872}^{(22)}
\rightarrow\cdots
$$

因此误差会沿 Decoder 层传播，最终得到：

$$
\widetilde{KV}_{\text{SWA}}^{(l)}
\neq
KV_{\text{SWA,full}}^{(l)}.
$$

不过这种近似主要影响 replay segment 的左端：

- 位置 872 缺少 127 个局部历史 token；
- 位置 900 缺少一部分局部历史；
- 位置 999 的 128-token SWA 窗口已经完整。

第一个真正重要的 Decode Query 位于 1000，它需要的局部历史正是最近窗口。因此 bounded replay 对后续生成的影响较小。

此外，每个 Decoder Query 仍然可以通过 Selected Global KV 分支获取早期 Prompt 信息：

```text
当前 Decoder Query
        │
        ├── Local SWA：
        │     replay segment 中的细粒度最近上下文
        │
        └── Selected Global/Main KV：
              从完整 Prompt 的压缩全局缓存中选择长距离信息
```

因此位置 872 即使在 Local SWA 中看不到 0～871，仍有可能通过 Global KV 获取更早的 Prompt 信息。

---

### 26.8 为什么精确重建非常昂贵

乍看之下，最终 Decode 只需要每层最近 128 个 SWA KV，那么似乎让每层都重算 128 个位置就应该是精确的。

问题在于：跨层 SWA 依赖会向前扩散。

用一个简化例子：

- 3 个 Decoder 层；
- 窗口 $W=4$；
- 希望重建最后四个位置 8～11。

最顶层位置 8～11 需要前一层更早的状态：

```text
Layer 3 需要：8～11
Layer 2 需要：5～11
Layer 1 需要：2～11
```

随着层数增加，最底部 Decoder 层必须向前补越来越多的位置。

一般而言，如果有 $D$ 个 Decoder 层，要精确恢复最后 $W$ 个顶层状态，最底层需要覆盖的范围大约为：

$$
W+(D-1)(W-1)
\approx D\times W.
$$

DeepSeek-V4.1-Flash 中：

$$
D=L/2=20,
\qquad W=128.
$$

于是底部所需历史范围大约为：

$$
20\times128=2560\text{ tokens}.
$$

当长 Prefix 已经命中 Global KV、而新增 suffix 很短时，为了恢复 SWA 再处理数千个历史 token，会大幅抵消 Prefix Cache 的收益。

Bounded Replay 强制每一层都只处理同一个 128-token segment：

```text
精确恢复：

Decoder 最低层：需要很长的历史范围
中间层：历史范围逐层缩短
Decoder 最高层：需要最后 128 个位置


Bounded Replay：

Decoder Layer 20：只处理 128 个位置
Decoder Layer 21：只处理 128 个位置
...
Decoder Layer 39：只处理 128 个位置
```

所以 Decoder replay 的 token-layer 工作量被限制为：

$$
C_{\text{decoder replay}}
=\frac L2\times n_{\text{win}}.
$$

对本例：

$$
20\times128=2560
$$

个 token-layer。

---

### 26.9 At Every Prefill 的准确含义

`at every prefill` 指的是每次从 Prompt Prefill 转入 Decode 时进行一次，而不是每生成一个 token 都 replay。

完整生命周期如下：

```text
1. Prompt Prefill
   ├── Encoder 处理 Prompt
   ├── 构造或读取 Global KV
   └── Replay 最后 128 个 Encoder outputs
                         │
                         ▼
2. 得到 Decoder 各层的 SWA KV
                         │
                         ▼
3. Decode token 1000
                         │
                         ▼
4. Decode token 1001
                         │
                         ▼
5. Decode token 1002
   ...
```

Decoder replay 只发生在 Prefill 的末尾。

进入正常 Decode 后，不会为每个输出 token 再 replay 一遍 128 个 Prompt token。每生成一个新 token，只需：

1. 为当前 token 运行 Decoder；
2. 生成本 token 在各层的 SWA KV；
3. 将它追加到本层运行时 SWA cache；
4. 淘汰超过窗口的最老条目。

例如 Prefill 完成后：

```text
SWA Cache：872～999
```

生成 token 1000 后：

```text
SWA Cache：873～1000
```

生成 token 1001 后：

```text
SWA Cache：874～1001
```

它是一个长度不超过 128 的滚动窗口。

---

### 26.10 Only for Decoding 不等于不缓存

这里必须区分两种 cache：

| Cache 类型 | 生命周期 | 是否保存 Decoder replay SWA |
|---|---|---|
| 当前活跃请求的运行时 KV Cache | 当前请求的 Decode 期间 | 是 |
| 持久化 Prefix Cache | 跨请求、跨会话或长时间复用 | 否 |

Replay 产生的 Decoder SWA KV 必须暂时保存在运行时 KV Cache 中，否则第一个 Decode token 无法使用最近的 Prompt 上下文。

所以：

```text
“only for decoding”
不等于“完全不保存”
```

而是：

```text
只保存在当前请求的运行时状态中
不作为可跨请求复用的 Prefix Cache 条目
```

---

### 26.11 为什么不写入 Prefix Cache

主要有三个原因。

#### 1. 节省持久化缓存空间

论文 3.2.1 指出，DeepSeek-V4 中 SWA KV 接近持久化 KV Cache 容量的一半。

如果每个 Prefix 都长期保存每层 SWA KV：

$$
\text{请求数}
\times\text{层数}
\times n_{\text{win}}
\times\text{KV 宽度}
$$

会产生很高的 SSD、DRAM 或 HBM 存储成本。

而 Decoder SWA 只在当前会话、当前 Prefix 尾部以及紧接着的 Decode 中有价值，其长期复用率远低于 Global KV。

#### 2. 它可以低成本重建

既然只需：

$$
\frac L2\times n_{\text{win}}
$$

的 bounded replay 就能得到足以 Decode 的近似状态，那么长期保存的经济性就下降了。

论文选择：

```text
Global KV：长期保存，适合 Prefix Cache
Decoder SWA KV：需要时 replay，临时保存
```

#### 3. Replay 状态不是完整 Forward 的规范状态

Bounded replay 在位置 $s$ 截断了局部依赖，因此产生的是近似状态：

$$
\widetilde{KV}_{\text{SWA}}
\neq
KV_{\text{SWA,full}}.
$$

它非常适合服务紧接着的 Decode，但没有必要把这种为当前 replay 边界构造的临时状态加入长期 Prefix Cache。

---

### 26.12 Prefix Cache 命中示例

假设已有缓存 Prefix 长度为 990，这次新增 10 个 token，最终 Prompt 长度为 1000：

```text
Cached prefix：0～989
New suffix：   990～999
Window：       128
```

最终 Decoder 需要的尾部窗口仍是：

$$
872～999.
$$

它由两部分组成：

- 缓存 Prefix 尾部：872～989，共 118 个 token；
- 新 suffix：990～999，共 10 个 token。

```text
完整 Prompt：
0 ................ 871 | 872 ........ 989 | 990 ..... 999
      不进 Decoder replay | 缓存前缀尾部 118 | 新后缀 10
                          └──── 共 128 ────┘
```

流程为：

```text
持久化 Global KV 命中
        │
        ├── 不重新计算完整 Prefix 的 Global KV
        │
        └── 准备最后 128 个位置的 Encoder outputs
                       │
                       ▼
             送入 Decoder 20 层 replay
                       │
                       ▼
             临时生成 Decoder SWA KV
                       │
                       ▼
                  开始 Decode
```

这正是 bounded replay 在“长缓存 Prefix + 短新增 suffix”场景中的价值：不会因为缺少 Decoder SWA KV，又把整个长 Prefix 送入 Decoder。

---

### 26.13 为什么 Post-training 要模拟 Replay

如果训练时始终使用完整 SWA，而推理时突然采用截断 replay，就会出现训练和推理不一致：

```text
训练：
位置 872 看见 745～872

推理 replay：
位置 872 只看见 872
```

论文因此在 post-training 中模拟同样的 replay，让模型学习适应：

- replay 左边界附近缺少局部历史；
- SWA KV 是近似重建的；
- 长距离信息更多依赖 Global KV；
- 越靠近 Prompt 尾部，局部窗口逐步恢复完整。

这属于 train-aware adaptation：不是让 bounded replay 变得数学精确，而是让模型学会在这种近似状态下保持输出质量。

---

### 问题 26 最终总结

原句可以改写为：

> 每次完成 Prompt 的 Encoder Prefill、准备开始生成第一个输出 token 时，只取 Prompt 最后 $n_{\text{win}}$ 个位置在 Causal Encoder 中得到的 hidden states，让它们顺序经过后半段所有 Decoder 层。每个 Decoder 层只允许局部 SWA 访问这段 replay 区间内部的 Key，由此临时重建该层最近窗口的 SWA KV Cache。这些 SWA KV 只保留在当前请求的运行时缓存中，供接下来的自回归生成使用，不加入可被未来请求复用的持久化 Prefix Cache。

完整数据流为：

```text
完整 Prompt
   │
   ▼
Causal Encoder
   │
   ├──────────── 全部 Encoder 输出 ───────► Global KV
   │                                      持久化 Prefix Cache
   │
   └── 最后 nwin 个 Encoder 输出
                    │
                    ▼
         Decoder Bounded Replay
         每层只处理同一尾部窗口
                    │
                    ▼
         每层临时 Decoder SWA KV
                    │
                    ▼
              当前请求 Decode
                    │
                    ▼
       请求结束后不进入 Prefix Cache
```

一句话概括：

$$
\boxed{
\text{完整 Prompt 只走 Encoder；Prompt 尾部一个窗口再走 Decoder；}
\text{重建出的 Decoder SWA 只服务随后的生成。}
}
$$

---

## 问题 27：Prefill 计算量接近减半的说法是否准确

### 问题 27 原文

> 论文中提及的 prefill 阶段计算量减半的说法是否不准确？因为在 causal encoder 中的 3～7、9～13、15～19 的 CSA2 模式不是 Reuse 吗？后面 decoder 也还是有很多的 Reuse 和 Reindex。

### 问题 27 核心结论

论文“Prefill 计算量接近减半”的说法，在它明确采用的**渐近 token-layer 计算口径**下是成立的；但它不是“精确 FLOPs 减少 50%”，更不是“任何长度、任何硬件上的 Prefill 延迟都降低 50%”。

用户指出 Encoder 中存在大量 Reuse、Decoder 中存在 Reuse/Reindex，是非常重要的。这说明不同层的实际计算成本并不完全相等，因此论文的：

$$
O(NL)
\rightarrow
O\left(\frac{NL}{2}+n_{\text{win}}\frac L2\right)
\approx O\left(\frac{NL}{2}\right)
$$

是高层复杂度估算，而不是逐算子精确统计。

但 Reuse/Reindex 并不会跳过整层 Transformer 计算，所以它们不能直接推翻“只有前半层处理完整 Prompt”这一推导。论文 Figure 3、Figure 4、Section 2.2、2.3 和 3.2.2 对 CED 与 CSA2 采用了两个不同的优化维度。

本节依据：

- `D:\Code\vllm_notes\DeepSeekV41-Flash\DeepSeek-V4.1-Flash Pushing the Limits of KV Cache Compression.pdf`，重点参考第 7～11 页和第 20 页。

---

### 27.1 论文的减半公式在计算什么

论文 Section 2.2 给出的口径，本质上是在统计：

$$
\text{处理的 token 数}
\times
\text{经过的 Transformer 层数}.
$$

也就是 token-layer 数量。

设：

- Prompt 长度为 $N$；
- 模型总层数为 $L=40$；
- Causal Encoder 为前 $L/2=20$ 层；
- Decoder 为后 $L/2=20$ 层；
- SWA replay 长度为 $n_{\text{win}}=128$。

普通 40 层模型的 Prefill token-layer 数为：

$$
C_{\text{normal}}=NL.
$$

CED 中：

1. 完整 Prompt 经过 20 层 Causal Encoder：

   $$
   N\frac L2;
   $$

2. 只有最后 $n_{\text{win}}$ 个 token 经过 20 层 Decoder：

   $$
   n_{\text{win}}\frac L2.
   $$

所以：

$$
C_{\text{CED}}
=N\frac L2+n_{\text{win}}\frac L2.
$$

相对于普通模型：

$$
\frac{C_{\text{CED}}}{C_{\text{normal}}}
=\frac{NL/2+n_{\text{win}}L/2}{NL}
=\frac12+\frac{n_{\text{win}}}{2N}.
$$

当：

$$
N\gg n_{\text{win}},
$$

第二项趋近于 0：

$$
\frac{C_{\text{CED}}}{C_{\text{normal}}}
\approx\frac12.
$$

因此论文使用的是：

> 对长序列，处理完整 Prompt 的网络深度从大约 40 层降低到大约 20 层。

它并没有证明所有算子的精确 FLOPs 必然减少一半。

---

### 27.2 Encoder 与 Decoder 的模式布局

#### Causal Encoder：层 0～19

```text
Layer 0～1：SWA only

Layer 2：Full
Layer 3～7：Reuse

Layer 8：Full
Layer 9～13：Reuse

Layer 14：Full
Layer 15～19：Reuse
```

即：

- 2 个纯 SWA 层；
- 3 个 Full 层；
- 15 个 Reuse 层。

可以写成：

$$
C_E
=2c_{\text{SWA}}
+3c_{\text{Full}}^{(r=2)}
+15c_{\text{Reuse}}^{(r=2)}.
$$

#### Decoder：层 20～39

```text
Layer 20：Full
Layer 21～23：Reuse

Layer 24：Reindex
Layer 25～27：Reuse

Layer 28：Reindex
Layer 29～31：Reuse

Layer 32：Reindex
Layer 33～35：Reuse

Layer 36：Reindex
Layer 37～39：Reuse
```

即：

- 1 个 Full 层；
- 4 个 Reindex 层；
- 15 个 Reuse 层。

可以写成：

$$
C_D
=c_{\text{Full}}^{(r=1)}
+4c_{\text{Reindex}}^{(r=1)}
+15c_{\text{Reuse}}^{(r=1)}.
$$

所以“后面有很多 Reuse 和 Reindex”这个观察完全正确。但关键在于，Reuse/Reindex 复用的是 Main KV 和索引相关状态，不是整个 Transformer 层的输出。

---

### 27.3 Reuse 不是复用整层输出

Reuse 模式复用的是：

- Main KV；
- Indexer K；
- 最新的 Top-K indices。

它不复用：

- 当前层 hidden states；
- 当前层 Main Q；
- 当前层 SWA KV；
- 当前层 Core Attention；
- 当前层 Attention output；
- 当前层 MoE/FFN；
- 当前层 mHC/residual 更新。

论文 Figure 4 明确说明，Full、Reindex、Reuse 三种模式都会计算当前层自己的：

$$
\text{Main Q}
\quad\text{和}\quad
\text{SWA KV}.
$$

并且三种模式都会进行：

$$
\text{Selected Main KV}
+\text{Layer-local SWA KV}
\rightarrow
\text{Core Attention}
\rightarrow
\text{Attention Output}.
$$

Figure 3 还显示，每层 Attention 后仍然有自己的 MoE。

因此 Reuse 层的数据流是：

```text
当前层输入 hidden state
        │
        ├── 计算当前层 Main Q
        ├── 计算当前层 SWA KV
        │
        ├── 读取共享 Main KV
        ├── 读取共享 Top-K indices
        │
        ▼
Selected Main KV + Layer-local SWA KV
        │
        ▼
当前层 Core Attention
        │
        ▼
当前层输出投影
        │
        ▼
当前层 MoE/FFN
        │
        ▼
下一层 hidden state
```

所以：

$$
\boxed{\text{Reuse}\neq\text{跳过 Transformer 层}}.
$$

更准确地说：

$$
\boxed{
\text{Reuse}
=\text{省掉 Main KV 构造与 Indexer 搜索的一部分计算}
}.
$$

这就是为什么 Encoder 的第 3～7、9～13、15～19 层仍然必须处理全部 $N$ 个 Prompt token。

---

### 27.4 Reindex 仍执行哪些计算

Reindex 层复用：

- 前面 Full 层生成的 Main KV；
- 对应的 Indexer K；
- Decoder 第一个 Full 层构造的共享候选池。

但它仍然需要：

1. 计算本层 Main Q；
2. 计算本层 SWA KV；
3. 计算本层 Indexer Q；
4. 对共享候选池重新打分；
5. 产生新的 Top-K；
6. Gather Selected Main KV；
7. 计算 Local + Global 联合 Attention；
8. 运行输出投影；
9. 运行本层 MoE/FFN。

因此：

$$
\text{Reindex Cost}
=\text{Common Transformer Cost}
+\text{Candidate Re-scoring Cost}.
$$

它确实比 Full 便宜，但绝不是“这一层被跳过”。

---

### 27.5 三种模式的成本分解

设 $c_{\text{common}}$ 表示所有 CSA2 层都要执行的公共部分：

- Norm；
- Main Q 投影；
- SWA KV 投影；
- Local SWA Attention；
- Selected Main KV Attention；
- 输出投影；
- MoE/FFN；
- residual/mHC。

再定义：

- $c_{\text{mainKV}}$：生成 Main KV；
- $c_{\text{indexK}}$：生成 Indexer K；
- $c_{\text{indexQ}}$：生成 Indexer Q；
- $c_{\text{full-score}}$：扫描完整上下文；
- $c_{\text{candidate-score}}$：只对候选池重新打分。

那么：

$$
c_{\text{Full}}
=c_{\text{common}}
+c_{\text{mainKV}}
+c_{\text{indexK}}
+c_{\text{indexQ}}
+c_{\text{full-score}},
$$

$$
c_{\text{Reindex}}
=c_{\text{common}}
+c_{\text{indexQ}}
+c_{\text{candidate-score}},
$$

$$
c_{\text{Reuse}}
=c_{\text{common}}.
$$

因此：

$$
c_{\text{Reuse}}
<c_{\text{Reindex}}
<c_{\text{Full}},
$$

但：

$$
c_{\text{Reuse}}>0
$$

且不一定很小，因为 MoE、Q 投影、SWA KV、Attention 和输出投影仍然存在。

---

### 27.6 CED 与 CSA2 优化不同的计算维度

这是理解论文结论的关键。

#### CED：减少完整 Prompt 穿过的层数

CED 主要优化 token-layer 维度：

```text
普通模型：

N 个 Prompt token
    ×
40 层


CED：

N 个 Prompt token × 20 层 Encoder
+
128 个尾部 token × 20 层 Decoder
```

它回答的问题是：

> 一个 Prompt token 是否需要进入后半段 Decoder？

#### CSA2 Reuse/Reindex：减少每层内部的 KV 与索引工作

CSA2 主要优化层内常数和跨层缓存：

```text
Full：
生成 Main KV + Indexer K + 新 Top-K

Reindex：
复用 Main KV/Indexer K，只重新计算 Top-K

Reuse：
Main KV、Indexer K、Top-K 都复用
```

它回答的问题是：

> 当一个 token 已经进入这一层后，这层要重新计算多少 Main KV 和 Indexer 工作？

二者可以正交叠加：

```text
                         是否进入 Decoder 层
                               │
                               │ CED
                               ▼
                    只让最后 nwin 个 token 进入
                               │
                               ▼
                   进入层以后做哪些计算
                               │
                         CSA2 mode
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
            Full            Reindex           Reuse
```

所以 Encoder 中有大量 Reuse，并不意味着这些 Encoder 层没有处理完整 Prompt；它只意味着处理完整 Prompt 时少做了某些全局 KV/Indexer 操作。

---

### 27.7 更精确的模式加权公式

论文的简单公式近似假设每层成本相同：

$$
C_{\text{normal}}\approx NL.
$$

如果区分不同模式，定义：

$$
C_E
=2c_{\text{SWA}}
+3c_{\text{Full}}^{(2)}
+15c_{\text{Reuse}}^{(2)},
$$

$$
C_D
=c_{\text{Full}}^{(1)}
+4c_{\text{Reindex}}^{(1)}
+15c_{\text{Reuse}}^{(1)}.
$$

其中上标 $(2)$、$(1)$ 分别表示 Main KV 压缩比例 2 和 1。

如果完整 Prompt 经过全部 40 层，模式加权成本约为：

$$
C_{\text{full-pass}}
=N(C_E+C_D).
$$

采用 CED 和 Decoder Bounded Replay 后：

$$
C_{\text{CED}}
=NC_E+n_{\text{win}}C_D+C_{\text{aux}}.
$$

其中 $C_{\text{aux}}$ 包括：

- 从 Encoder 输出构造 Decoder Global KV；
- replay batch 准备；
- Prefix Cache 管理；
- 候选池构造；
- 其他投影和调度开销。

因此更精确的成本比例是：

$$
R
=\frac{NC_E+n_{\text{win}}C_D+C_{\text{aux}}}
{N(C_E+C_D)}.
$$

当：

$$
N\gg n_{\text{win}},
\qquad
C_{\text{aux}}\ll N(C_E+C_D),
$$

有：

$$
R\approx\frac{C_E}{C_E+C_D}.
$$

只有在：

$$
C_E\approx C_D
$$

时，才得到：

$$
R\approx\frac12.
$$

这就是论文 `nearly halving` 的隐含近似。

---

### 27.8 实际比例为什么可能偏离一半

偏离方向不是固定的，要看 $C_E$ 与 $C_D$ 谁更大。

#### 情况一：Encoder 和 Decoder 总成本接近

如果：

$$
C_E\approx C_D,
$$

那么：

$$
R\approx50\%.
$$

这种估算并非毫无依据，因为两边都是 20 层，而且绝大多数层仍然执行 Main Q、SWA KV、Core Attention、输出投影和 MoE。特别是每层都存在 MoE，Reuse 省掉的 Indexer 工作未必占整层成本的主体。

#### 情况二：Decoder 因大量 Reuse 而明显更便宜

假设完整 40 层成本中：

$$
C_E:C_D=60:40.
$$

当 $N\gg n_{\text{win}}$ 时：

$$
R\approx\frac{60}{60+40}=60\%.
$$

这时只节省约 40%，而不是 50%。所以如果 Decoder 半段本来就显著比 Encoder 半段便宜，论文的“减半”会高估实际 FLOPs 节省。

#### 情况三：Decoder 比 Encoder 更昂贵

如果由于 ratio=1 Main KV 数量更大、Hierarchical Indexer、Reindex 或其他 Decoder 特有开销，使得：

$$
C_D>C_E,
$$

那么跳过大部分 Decoder Prefill 可能节省超过 50%。

因此，单纯看到 Decoder 有 15 个 Reuse 层，并不能直接判断实际节省一定低于 50%；还需要知道：

- Reuse 省掉的部分占整层多少；
- MoE 占多少；
- Attention 与 Indexer 各占多少；
- ratio=1 与 ratio=2 的差异；
- 硬件上的 kernel 效率。

论文没有在这条复杂度公式中给出逐模式 FLOPs 权重，因此不能从该公式得到精确百分比。

---

### 27.9 不同 Prompt 长度下的理论比例

即使接受均匀层成本假设：

$$
R=\frac12+\frac{n_{\text{win}}}{2N}.
$$

当 $n_{\text{win}}=128$ 时：

| Prompt 长度 $N$ | CED/普通计算量 | 理论减少比例 |
|---:|---:|---:|
| 256 | 75% | 25% |
| 512 | 62.5% | 37.5% |
| 1000 | 56.4% | 43.6% |
| 4096 | 51.5625% | 48.4375% |
| 32768 | 50.1953% | 49.8047% |
| 100000 | 50.064% | 49.936% |

以 $N=1000$ 为例：

$$
C_{\text{normal}}
=1000\times40
=40000
$$

个 token-layer。

CED：

$$
C_{\text{CED}}
=1000\times20+128\times20
=22560.
$$

所以：

$$
\frac{22560}{40000}=56.4\%,
$$

节省：

$$
1-56.4\%=43.6\%.
$$

因此对 1000-token Prompt，说“精确减半”是不准确的；说“接近减半”已经是比较宽松的表达。

对于 100K 长上下文：

$$
C_{\text{normal}}
=100000\times40
=4,000,000,
$$

$$
C_{\text{CED}}
=100000\times20+128\times20
=2,002,560.
$$

比例为：

$$
50.064\%,
$$

节省：

$$
49.936\%.
$$

这时 `nearly halving` 就非常准确。论文也明确加了：

$$
N\gg n_{\text{win}}.
$$

---

### 27.10 为什么实际延迟不一定减半

即使 token-layer 数减少约 50%，实际 Prefill latency 也未必减少 50%，原因包括：

#### 1. 不同模式的层成本不一样

$$
c_{\text{Full}}
\neq c_{\text{Reindex}}
\neq c_{\text{Reuse}}.
$$

#### 2. Encoder 与 Decoder 压缩比例不同

Encoder CSA2 通常：

$$
\text{ratio}=2,
$$

Decoder：

$$
\text{ratio}=1.
$$

两者 Main KV 数量和相关操作成本不同。

#### 3. Replay batch 较小，硬件利用率可能较低

完整 Encoder Prefill 处理大量 token，矩阵乘法容易达到较高吞吐。Decoder replay 只有 128 个 token，虽然 FLOPs 少，但 kernel launch、同步、通信、MoE dispatch 和小矩阵利用率可能使每 token 成本更高。

#### 4. Indexer 成本不完全服从简单的 $NL$

Full Indexer 需要扫描较长上下文，Reindex 需要扫描候选池。它们的复杂度还和 Main KV 长度、Top-K、candidate pool 大小与压缩比例有关。

论文的 $O(NL)$ 是在固定稀疏宽度和高层结构下的简化表示。

#### 5. MoE 和通信可能成为瓶颈

如果实际运行主要受专家路由、All-to-All、内存带宽、KV gather 或 kernel launch 限制，那么 token-layer 减半不会线性转化为端到端延迟减半。

---

### 27.11 如何准确表述论文结论

不够严谨的说法是：

> DeepSeek-V4.1-Flash 的 Prefill FLOPs 和延迟必然减少 50%。

更准确的说法是：

> 在长上下文 $N\gg n_{\text{win}}$ 下，CED 使完整 Prompt 主要只通过前 20 层 Causal Encoder，后 20 层 Decoder 仅处理最后 $n_{\text{win}}$ 个 token。因此按 token-layer 或均匀层成本计算，Prefill 主干计算量从约 $NL$ 降为 $NL/2+n_{\text{win}}L/2$，渐近接近原来的一半。CSA2 的 Full、Reindex、Reuse 会进一步改变每层内部的实际成本，因此精确 FLOPs 与端到端延迟需要按模式加权或通过系统 profiling 测量。

---

### 问题 27 最终总结

#### 判断一：Reuse/Reindex 会不会让论文公式失效？

不会。

因为 Reuse/Reindex 没有跳过整个 Transformer 层。Causal Encoder 的第 3～7、9～13、15～19 层仍然对完整 Prompt 的每个 token 执行本层的：

- Query；
- SWA KV；
- 联合 Attention；
- 输出投影；
- MoE/FFN。

Decoder 的 Reuse/Reindex 层也一样，只不过在 bounded replay 中，它们只对最后 $n_{\text{win}}$ 个 token 执行。

#### 判断二：“计算量减半”是不是精确结论？

不是。

它是满足下列条件的渐近近似：

$$
N\gg n_{\text{win}},
$$

并隐含假设：

$$
C_E\approx C_D,
\qquad
C_{\text{aux}}\text{ 相对较小}.
$$

因此最准确的结论是：

$$
\boxed{
\text{论文的 nearly halving 作为长上下文 token-layer 复杂度描述是合理的；}
}
$$

$$
\boxed{
\text{若把它理解为精确 FLOPs 或端到端延迟降低 50%，则确实不够准确。}
}
$$

Reuse/Reindex 的存在不是该结论的反例，而是说明真实节省比例应使用模式加权公式：

$$
\boxed{
C_{\text{CED}}
=NC_E+n_{\text{win}}C_D+C_{\text{aux}}
}
$$

而不能只使用统一的单位层成本。

---

## 问题 28：为什么长期 Global KV 要保存在 SSD

### 问题 28 原文

> 论文中说 V4.1 长期保存 Global KV，为什么要在 SSD 上面保存呢？是因为这个量很大？

### 问题 28 核心结论

是的，最直接的原因就是：**长期保存的 Global KV 总量非常大**。但更准确地说，不只是“单个请求很大”，而是：

$$
\boxed{
\text{单个长上下文的 Global KV 很大}
\times
\text{需要保留的大量 Prefix}
\times
\text{至少 72 小时的保留期}
}
$$

最终形成了远超 HBM、甚至远超普通主机 DRAM 容量的持久化缓存池，所以需要使用容量更大、每字节成本更低的 SSD。

同时必须区分：

- 活跃请求正在执行 Attention 时使用的 Runtime Global KV：在 HBM 中；
- 为未来请求复用而长期保存的 Persistent Global KV：在 SSD 或 Host Memory 中。

论文在摘要、Introduction 和 Section 3.2.1 中明确区分了 runtime KV 与 persistent KV，并说明 V4 部署为了获得较高命中率，配置了足够大的 SSD Persistent KV Cache，使典型工作负载下 KV 能保留超过 72 小时。

本节依据：

- `D:\Code\vllm_notes\DeepSeekV41-Flash\DeepSeek-V4.1-Flash Pushing the Limits of KV Cache Compression.pdf`，重点参考第 1、4、5、19、20 和 37 页。

---

### 28.1 Runtime KV 与 Persistent KV

#### 1. Runtime KV Cache

这是当前正在运行的请求马上要用于 Attention 的 KV。

例如当前请求正在生成 token 100001：

```text
当前 Query
   │
   ▼
从 HBM 中读取 Global KV / SWA KV
   │
   ▼
Attention 计算
```

这种 KV 需要高带宽、低延迟访问，因此必须放在 HBM，或者至少在计算前加载到设备端可直接访问的高速内存。

论文所说的 Global KV cache footprint “always in HBM”，指的是活跃请求真正参加 Attention 计算时的运行时 Global KV。

#### 2. Persistent KV Cache

这是为了以后再次遇到相同 Prefix 时，不必重新做 Prefill 而保存的 KV。

例如多个请求拥有相同的系统提示词：

```text
System Prompt：
“你是一个编程助手……”
+ 工具定义
+ API Schema
+ 仓库上下文
```

第一次请求：

```text
长 Prefix
   │
   ▼
执行 Prefill
   │
   ▼
生成 Global KV
   │
   ├── 当前请求使用
   │
   └── 保存到 Persistent KV Cache
```

后续请求命中同一 Prefix：

```text
相同 Prefix
   │
   ▼
命中 Persistent Global KV
   │
   ▼
从 SSD/Host Memory 加载
   │
   ▼
放入运行时缓存
   │
   ▼
跳过已缓存 Prefix 的大量 Prefill
```

Persistent KV 不是当前 Attention 每一步都直接从 SSD 随机读取，而是一个长期保存层。命中后，缓存数据需要进入执行引擎可以使用的运行时存储路径。

---

### 28.2 为什么不能长期全部放在 HBM

HBM 有三个问题：

1. 容量有限；
2. 每字节成本极高；
3. 还要同时存放模型权重、激活、工作区和活跃请求 KV。

即使 V4.1 已经把 Global KV 压缩到约：

$$
890\text{ bytes/token},
$$

单个长上下文仍然不小。

#### 单个 Prefix 的 Global KV 大小

| Prefix 长度 | Global KV 大小 |
|---:|---:|
| 10K token | $10,000\times890=8.9$ MB |
| 100K token | $100,000\times890=89$ MB |
| 1M token | $1,000,000\times890=890$ MB |

换算成二进制单位，1M token 大约是：

$$
\frac{890,000,000}{1024^2}
\approx848.8\text{ MiB}.
$$

也就是说，一个 1M-token Prefix 的 Global KV 就接近 0.85 GiB。

假设单张设备有 80 GiB HBM，即使完全不考虑模型权重、运行时工作区等，也只能存几十个这样的 Prefix。实际可用于 Persistent KV 的空间要小得多。

所以 HBM 只适合保存：

- 当前活跃请求；
- 即将被使用的热 KV；
- Attention 马上要访问的数据。

不适合保存数小时甚至数天都可能暂时不用的大量 Prefix。

---

### 28.3 请求数量与保留时间造成容量爆炸

单个 100K Prefix 的 V4.1 Global KV 约为：

$$
89\text{ MB}.
$$

如果长期缓存 10 万个这样的 Prefix：

$$
89\text{ MB}\times100,000
=8.9\text{ TB}.
$$

即使实际系统会进行 Prefix block 共享、去重、LRU 淘汰、分块存储、压缩和多机器分布，整体容量仍可能达到 TB 级甚至更高。

因此真正的问题是：

```text
单 Prefix 数十 MB～数百 MB
            ×
同时保存成千上万或更多 Prefix
            ×
72 小时以上保留期
            =
TB/PB 级潜在持久化容量需求
```

HBM 无法承担，普通主机 DRAM 的成本也很高，而 SSD 更适合这种大容量、长保留期的数据。

---

### 28.4 为什么 Global KV 值得长期保存

不是所有 KV 都值得写入 SSD。

论文认为 Global KV 有明显的 long-tail reuse，即某个 Prefix 在生成后，可能过了较长时间才再次被访问，但再次访问时仍然很有价值。

典型场景包括：

- 多轮 Agent 任务；
- 相同系统提示词；
- 大型工具定义与 API Schema；
- 相同代码仓库上下文；
- 相同知识库文档；
- 多个请求共享的对话前缀；
- 长时间运行的 Agent 恢复执行；
- 多用户共享公共 Prefix。

重新 Prefill 一个 100K 或 1M token 的 Prefix，需要读取并处理所有输入 token、运行 Causal Encoder、构造 Global KV 并执行 Indexer 相关操作，计算代价很大。

如果 SSD 中已有 Global KV，系统就可以用：

$$
\text{SSD/Host I/O 成本}
$$

换取避免：

$$
\text{长 Prefix Prefill 计算成本}.
$$

对于超长上下文，这种交换通常是值得的。

---

### 28.5 SSD 较慢为什么仍可使用

SSD 的带宽和延迟不如 HBM，但 Persistent KV 的目标不是直接服务每次 Attention 内积，而是在新请求开始或 Prefix Cache 命中时，把已经计算好的 KV 恢复到运行时缓存。

数据路径可以抽象为：

```text
Persistent Global KV
SSD / Host Memory
        │
        │ Prefix 命中，加载缓存块
        ▼
Host DRAM / 传输通道
        │
        ▼
HBM 中的 Runtime Global KV
        │
        ▼
Attention
```

所以这是缓存层级问题：

| 层级 | 特点 | 适合的数据 |
|---|---|---|
| HBM | 最快、最贵、容量最小 | 当前活跃请求的 Runtime KV |
| Host DRAM | 较快、容量较大 | 热 Prefix、短期状态、短 TTL SWA |
| SSD | 较慢、容量很大、每字节成本低 | 长期 Persistent Global KV |

论文没有披露 Persistent KV Cache 的完整软件实现，例如：

- SSD 是本地 NVMe 还是分布式存储；
- 是否存在额外的 DRAM 热缓存；
- KV block 的具体加载与预取方式；
- 压缩数据是否在 Host 或 Device 解码；
- 每级缓存的淘汰策略细节。

因此不能仅凭论文断言具体数据链路，但可以确认：SSD 是持久化容量层，不是每个 Attention step 的直接工作内存。

---

### 28.6 Global KV 与 SWA KV 的保留策略

这是论文 Section 3.2.1 最核心的设计差异。

#### Global KV

特点：

- 覆盖完整 Prefix；
- Prefix 命中时可以复用完整的全局上下文；
- 可能在很久之后再次命中；
- 重新计算长 Prefix 的成本很高；
- 具有 long-tail reuse。

因此：

```text
Global KV
→ Persistent KV Cache
→ SSD/Host Memory
→ 至少 72 小时保留期
```

#### SWA KV

特点：

- 每层只保留最近 $n_{\text{win}}$ 个状态；
- 主要服务当前活跃会话；
- 通常只在分钟级时间窗口内有价值；
- 会话结束或进入下一轮后，旧状态很快失去价值；
- 缺失时可以通过 SWA Bounded Replay 近似恢复。

因此 V4.1 不再把 SWA KV 长期写入 Persistent KV Cache，而是放在使用约 10% Host DRAM 构建的分布式短期内存池中：

```text
SWA KV
→ Host DRAM 短期池
→ TTL 只有几分钟
→ 过期后立即回收
→ Miss 时执行 Bounded Replay
```

两类缓存的策略对比如下：

| 项目 | Global KV | SWA KV |
|---|---|---|
| 覆盖范围 | 完整 Prefix | 最近滑动窗口 |
| 复用时间尺度 | 长尾、可达数天 | 分钟级、活跃会话内 |
| Miss 重建成本 | 长 Prefix Prefill，昂贵 | Bounded Replay，较便宜 |
| V4.1 长期保存位置 | Persistent Cache，SSD/Host Memory | 不进入 Persistent Cache |
| 典型保留期 | 至少 72 小时 | 数分钟 |
| 淘汰后恢复 | 可能重新 Prefill | Replay 最近 $n_{\text{win}}$ token |

---

### 28.7 为什么 V4.1 Persistent KV 约为 V4 的八分之一

论文给出两个乘法因素。

#### 第一个因素：不再持久化 SWA KV

在 V4 中，Persistent KV 大致由两部分组成：

```text
Persistent KV
├── Global KV：约一半
└── SWA KV：约一半
```

V4.1 移除持久化 SWA KV，容量先大约减半：

$$
1\rightarrow\frac12.
$$

#### 第二个因素：Global KV 压缩到 V4 的约四分之一

通过 CSA2 跨层 KV 共享、Main KV 压缩、Indexer K 共享、FP4 Main KV 和更紧凑的缓存布局，V4.1 的 Global KV 约为 V4 对应 Global KV 的：

$$
\frac14.
$$

两个因素相乘：

$$
\frac12\times\frac14
=\frac18.
$$

用一个归一化例子：

```text
假设 V4：
Global KV = 1
SWA KV    = 1
总计      = 2

V4.1：
Global KV = 1/4
持久化 SWA KV = 0
总计      = 1/4
```

比例为：

$$
\frac{1/4}{2}
=\frac18.
$$

所以 V4.1 仍然需要 SSD Persistent Cache，只是相同工作负载下的容量已经降低到 V4 的约八分之一。

---

### 28.8 为什么不能完全不保存 Global KV

如果完全不保存，那么每次多轮对话或 Agent 恢复都要重新计算全部 Prefix。

例如：

```text
第一轮：
100K Prompt → Prefill → 生成 Global KV → 输出

第二轮新增 100 token：

如果没有 Persistent Global KV：
重新 Prefill 前面 100K + 新增 100

如果有 Persistent Global KV：
加载前面 100K 对应的 Global KV
只处理未缓存的新 suffix
```

Agent 工作负载往往是：

```text
非常长的历史 Prefix
+
很短的新工具结果/用户输入
```

所以不保存 Global KV 会造成巨大的重复 Prefill。

Persistent Global KV 本质上是在做：

$$
\boxed{
\text{用廉价的大容量存储换昂贵的重复计算}
}.
$$

---

### 28.9 SSD 容量之外的原因

除了容量，SSD 还适合这个场景的几个原因是：

#### 1. 每字节成本低

长期保存 TB 级 KV 时，SSD 比 HBM、DRAM 更经济。

#### 2. 数据需要跨请求存在

HBM Runtime KV 通常跟随活跃请求生命周期；请求结束后，设备内存需要尽快释放给其他请求。Persistent Prefix Cache 必须跨请求、跨调度批次，甚至跨较长时间存在。

#### 3. Global KV 的访问具有长尾特征

它可能长时间不被访问，但一旦命中就能节省大量 Prefill。SSD 适合保存这种“容量大、访问不频繁、单次命中价值高”的数据。

#### 4. 可以分层管理冷热数据

热 Prefix 可以保留在 Host DRAM，较冷的 Prefix 放在 SSD。论文只给出了总体部署原则，没有完整披露每一级缓存的具体实现。

---

### 28.10 常见误解

#### 误解一：模型直接在 SSD 上做 Attention

不是。

Attention 所需的活跃 Global KV 最终必须位于设备能够高效访问的运行时存储中。SSD 主要负责长期保存。

#### 误解二：所有请求的 KV 都永久保存在 SSD

不是。

论文描述了 LRU 管理和保留期。容量不足时仍然需要淘汰，只是系统把 Persistent Cache 配置得足够大，使典型工作负载下 Global KV 能保证至少 72 小时寿命。

#### 误解三：V4.1 完全不保存 SWA KV

也不是。

它不把 SWA KV 放入长期 Persistent KV Cache，但活跃请求 Decode 时仍然需要运行时 SWA KV。Encoder SWA 还可以进入短 TTL 的 Host DRAM 分布式池；Decoder replay 产生的 SWA KV 则用于当前 Decode。

---

### 问题 28 最终总结

V4.1 把长期 Global KV 放在 SSD，首要原因确实是容量巨大。但完整原因是：

$$
\boxed{
\text{Global KV 单 Prefix 较大}
+\text{缓存 Prefix 数量巨大}
+\text{需要跨请求保存至少 72 小时}
+\text{Global KV 具有长尾复用价值}
}.
$$

因此形成了如下分层：

```text
HBM
└── 当前活跃请求的 Runtime Global KV / SWA KV
    最快，但容量最小、成本最高

Host DRAM
└── 热缓存和短 TTL SWA KV
    V4.1 使用约 10% Host DRAM 构建 SWA 短期池

SSD / Persistent Cache
└── 长期 Global KV
    容量大、每字节成本低、保留至少 72 小时
```

V4.1 并不是因为 SSD 速度快才使用 SSD，而是因为：

$$
\boxed{
\text{SSD 足够大且足够便宜，可以保留大量低频但高价值的 Global KV。}
}
$$

命中后付出一次数据加载成本，通常仍比重新对几十万乃至上百万 token 执行 Prefill 更划算。

---

## 问题 29：为什么 Prefill 激活 8B、Decode 激活 16B 参数

### 问题 29 原文

论文中写道：

> DeepSeek-V4.1-Flash has 552B backbone parameters and 196B Engram parameters, activating 8B parameters per token during prefill and 16B during decode.

为什么 Prefill 是 8B，而 Decode 是 16B？下面给出计算过程。

### 问题 29 核心结论

最关键的原因不是 Prefill 和 Decode 使用了不同数量的 MoE 专家，而是 CED（Causal Encoder-Decoder）让两类 token 实际经过的主干层数不同：

```text
普通 Prompt token：20 层 Causal Encoder              → 约激活 8B 参数
Decode 新 token：   20 层 Encoder + 20 层 Decoder    → 约激活 16B 参数
```

所以可以把论文的数字理解成：

$$
P_{\text{active,prefill}}
\approx \frac{20}{40}P_{\text{active,decode}}
\approx \frac{1}{2}\times16\text{B}
=8\text{B}.
$$

这里的“一半”来自主路径的网络深度从 40 层降为 20 层，而不是因为 Prefill 每层只激活一半专家。

### 29.1 激活参数是什么意思

“每个 token 激活 8B/16B 参数”指：处理这个 token 的一次前向传播中，实际参与运算的权重参数总量约为 80 亿/160 亿。

它不等于：

- 模型总参数量；
- KV Cache 的元素数或字节数；
- 中间激活张量占用的显存；
- 实际 FLOPs；
- 整个请求中所有 token 累计访问的参数量。

DeepSeek-V4.1-Flash 的 backbone 总共有 552B 参数，但它是稀疏 MoE 模型。每个 token 在每层只访问少量被路由到的专家，而不是访问全部 384 个 routed experts，因此单 token 激活参数量远小于 552B。

### 29.2 为什么 552B 总参数只激活约 16B

论文给出的主要结构参数包括：

| 项目 | 数值 |
|---|---:|
| Backbone 层数 | 40 |
| 隐藏维度 $d$ | 5120 |
| 每层 routed experts 数 | 384 |
| 每个 token 选中的 routed experts 数 | 6 |
| 每层 shared expert 数 | 1 |
| Expert 中间维度 $d_{ff}$ | 2304 |
| Expert MLP 形式 | SwiGLU |

SwiGLU expert 通常包含三个主要矩阵：gate projection、up projection 和 down projection。因此忽略 bias 等小项后，一个 expert 的参数量近似为：

$$
P_{\text{expert}}
=3d d_{ff}
=3\times5120\times2304
=35{,}389{,}440
\approx35.39\text{M}.
$$

虽然每层一共有 384 个 routed experts，但一个 token 每层只选择 6 个 routed experts，同时还经过 1 个 shared expert。因此，一层中该 token 激活的 expert 参数约为：

$$
P_{\text{MoE,layer}}
=(6+1)\times35.38944\text{M}
=247.72608\text{M}
\approx0.248\text{B}.
$$

这正是“总参数很大、单 token 激活参数较小”的来源。

### 29.3 552B 总参数的 MoE 数量级核算

仅计算所有 routed experts：

$$
\begin{aligned}
P_{\text{routed}}
&=40\times384\times35{,}389{,}440\\
&=543{,}581{,}798{,}400\\
&\approx543.58\text{B}.
\end{aligned}
$$

所有 shared experts 的参数量约为：

$$
P_{\text{shared}}
=40\times35{,}389{,}440
=1.4156\text{B}.
$$

两者合计：

$$
P_{\text{experts,total}}
\approx543.58\text{B}+1.42\text{B}
\approx545.00\text{B}.
$$

它已经占到 552B backbone 的绝大部分。剩余参数主要属于 attention、indexer、mHC、归一化、embedding、输出头等模块。这个核算也说明：552B 的主体是 384 路专家的“参数库”，而每个 token 只会访问其中很小的一部分。

### 29.4 每个 Token 在单层激活多少专家参数

每层中，一个 token 的 routed expert 激活量为：

$$
6\times35.38944\text{M}
=212.33664\text{M}.
$$

再加 1 个 shared expert：

$$
212.33664\text{M}+35.38944\text{M}
=247.72608\text{M}.
$$

于是：

- 20 层的 active MoE 参数约为

  $$
  20\times247.72608\text{M}
  =4.9545216\text{B};
  $$

- 40 层的 active MoE 参数约为

  $$
  40\times247.72608\text{M}
  =9.9090432\text{B}.
  $$

这里只计算了专家 MLP；还需要加每层共同执行的 attention、indexer、mHC、归一化以及其他 dense 参数，才得到论文报告的约 8B 和 16B。

### 29.5 从 16B 反推 8B

因为 Decode token 要经过全部 40 层，所以论文给出的 Decode 激活量约为：

$$
P_{\text{active,decode}}
\approx16\text{B}.
$$

按 40 层平均，每层激活参数约为：

$$
\bar P_{\text{active,layer}}
\approx\frac{16\text{B}}{40}
=0.4\text{B/layer}.
$$

普通 Prefill token 只走 20 层 Causal Encoder，因此：

$$
P_{\text{active,prefill}}
\approx20\times0.4\text{B}
=8\text{B}.
$$

Decode token 走 20 层 Encoder 和 20 层 Decoder：

$$
P_{\text{active,decode}}
\approx(20+20)\times0.4\text{B}
=16\text{B}.
$$

这就是论文中 8B 与 16B 最直接的计算逻辑。

### 29.6 为什么普通 Prefill Token 主要只经过 20 层

CED 把 40 层主干拆成：

```text
Layer 0～19   ：Causal Encoder
Layer 20～39  ：Decoder
```

对长 Prompt 执行 Prefill 时，绝大多数提示词只计算 Causal Encoder。Encoder 输出一方面被压缩并投影为各个 Decoder 层可使用的 Global KV，另一方面保留最后一个局部窗口所需的信息。这样就不必把完整的长 Prompt 再逐 token 送过后 20 层 Decoder。

因此论文用近似式表示主路径计算量：

$$
C_{\text{CED}}
=N\frac{L}{2}+n_{\text{win}}\frac{L}{2},
$$

其中第一项是全部 $N$ 个 Prompt token 经过前半段 Encoder，第二项是末尾 $n_{\text{win}}$ 个 token 为重建 Decoder SWA KV 而执行的 bounded replay。

### 29.7 为什么 Decode Token 必须经过 40 层

生成阶段每产生一个新 token，都需要计算这个新 token 在整张网络中的表示：

```text
新 token
  ↓
Encoder Layer 0～19
  ↓
Decoder Layer 20～39
  ↓
LM Head / 下一个 token 概率
```

KV Cache 能避免重新计算以前所有 token 的 K/V，但不能跳过当前新 token 自己的层计算。当前 token 仍要在每层生成 Q，并生成和写入相应的 K/V，完成 attention、MoE、残差等计算。因此 Decode 每个 token 激活约 40 层的权重，也就是约 16B 参数。

### 29.8 Full、Reindex、Reuse 对激活参数的影响

CSA2 的 Full、Reindex、Reuse 会改变 indexer 和 Main KV 的产生或复用方式：

- Full：生成/取得 Main KV 和 Indexer K，并搜索 Top-K；
- Reindex：复用共享的 Main KV、Indexer K，但重新为本层 Query 打分并选择 Top-K；
- Reuse：连 Top-K 索引也复用。

这会改变每层精确执行的算子和参数量，但并不意味着 Reuse 层完全不计算。三种模式仍需执行本层的 Main Q、局部 SWA KV、注意力输出投影、mHC 和 MoE 等主干计算。所以 8B/16B 是论文对整个 token 前向所给的汇总值，而不是只把 Full 层计入、把 Reindex/Reuse 层全部排除。

### 29.9 用 MoE 参数核验 8B 和 16B

前面得到：

$$
P_{\text{active MoE,40 layers}}
\approx9.91\text{B}.
$$

论文报告 Decode 总激活约 16B，因此可以用数量级做一致性检查：

$$
P_{\text{other active,40 layers}}
\approx16-9.91
=6.09\text{B}.
$$

如果 Prefill 主路径只运行一半层数，则：

$$
\begin{aligned}
P_{\text{active MoE,20 layers}}&\approx4.95\text{B},\\
P_{\text{other active,20 layers}}&\approx3.05\text{B},\\
P_{\text{total,20 layers}}&\approx4.95+3.05=8.00\text{B}.
\end{aligned}
$$

这不是论文公布的逐模块精确参数表，而是利用模型结构做的数量级核验；结果恰好与 8B/16B 的两倍关系吻合。

### 29.10 为什么 196B Engram 不直接加到 8B/16B

196B Engram 是巨大的稀疏 n-gram embedding table 总容量，不代表每个 token 会把 196B 参数全部读出。论文给出的 Engram 配置包括：

- 2 个 Engram 模块，位于 Encoder 的第 1 层和第 14 层附近；
- n-gram 阶数为 $\{2,3,4\}$；
- 每种阶数有 8 个 hash heads；
- 每种阶数的总 embedding dimension 为 2048；
- 每个 hash head 约有 16M 个 table entries。

每个 head 的向量维度为：

$$
d_{\text{head}}
=\frac{2048}{8}
=256.
$$

单个 Engram 模块的总表参数量约为：

$$
3\times8\times16\text{M}\times256
=98.304\text{B}.
$$

两个模块合计：

$$
2\times98.304\text{B}
=196.608\text{B},
$$

与论文所说的 196B 一致。

但一个 token 在单个模块只会对每个 n-gram 阶数、每个 hash head 查询一个 embedding 向量，因此粗略查询的 table values 数量是：

$$
3\times8\times256
=6144.
$$

两个模块合计约为：

$$
2\times6144=12{,}288
=0.000012288\text{B}.
$$

与 8B/16B 相比可以忽略不计。并且这两个 Engram 模块都位于 Encoder 中，普通 Prefill token 和 Decode token 都会经过它们，所以 Engram 也不是 Decode 比 Prefill 多一倍的原因。

### 29.11 Decoder SWA Bounded Replay 的例外

“Prefill 每 token 激活 8B”是对长上下文主路径的概括。为了构造解码开始时所需的 Decoder SWA KV，Prefill 还会把 Prompt 末尾 $W=n_{\text{win}}$ 个 token 的 Encoder 输出送入后 20 层 Decoder 做 bounded replay。

因此更精确地说：

- 前 $N-W$ 个 Prompt token：只走 20 层 Encoder，约 8B；
- 最后 $W$ 个 Prompt token：还参与 20 层 Decoder replay，就这一次 replay 而言约相当于完整 40 层，即约 16B。

整个 Prefill 的累计激活参数访问量可近似写成：

$$
\begin{aligned}
A_{\text{prefill}}
&\approx(N-W)\times8\text{B}+W\times16\text{B}\\
&=8N\text{B}+8W\text{B}.
\end{aligned}
$$

平均到每个 Prompt token：

$$
\bar P_{\text{active,prefill}}
\approx8\text{B}\left(1+\frac{W}{N}\right).
$$

以 $W=128$ 为例：

| Prompt 长度 $N$ | 平均激活参数量近似值 |
|---:|---:|
| 1,000 | $8\times(1+128/1000)=9.024\text{B}$ |
| 4,096 | $8\times(1+128/4096)=8.25\text{B}$ |
| 100,000 | $8\times(1+128/100000)=8.01024\text{B}$ |

所以，对长 Prompt 而言，bounded replay 是一个被固定窗口限制的附加项，平均值快速趋近 8B。论文的“8B during prefill”应理解为架构主路径或长上下文渐近意义下的数字，而不是宣称包括尾部 replay 在内的每一个 Prompt 位置都严格只访问 8B 权重。

### 29.12 激活参数不等于 FLOPs

即使 Prefill 是 8B、Decode 是 16B，也不能直接得出 Decode 单 token 延迟严格是 Prefill 的两倍。原因包括：

- Prefill 的 attention 要处理许多 Query，矩阵形状大，通常有更高并行度；
- Decode 一次通常只有很少 Query，但需要读取长历史 KV，容易受显存/内存带宽限制；
- Indexer、Top-K、Main KV、SWA KV 的访问方式不同；
- MoE 的 expert routing 和跨卡通信成本不同；
- Full、Reindex、Reuse 层的实际算子量不同；
- Prefill 还存在 bounded replay；
- 融合算子、量化和硬件利用率会改变实际耗时。

因此，8B/16B 描述的是“每个 token 使用了多少权重参数”，而不是完整性能模型。

### 问题 29 最终总结

```text
552B backbone
  └─ 绝大多数参数存在于每层 384 个 routed experts 中
       └─ 每个 token 只选 6 个 routed experts + 1 个 shared expert

普通 Prefill token
  └─ 主要经过 20 层 Causal Encoder
       └─ 约激活 8B 参数

最后 n_win 个 Prompt token
  └─ Encoder 后还做 Decoder SWA bounded replay
       └─ 这是固定窗口附加项，不改变长上下文平均值趋近 8B

Decode 新 token
  └─ 经过 20 层 Encoder + 20 层 Decoder
       └─ 约激活 16B 参数

196B Engram
  └─ 是稀疏哈希 embedding table 的总容量
       └─ 每个 token 只查极少数条目，不把 196B 全部激活
```

一句话概括：**Prefill 的普通 Prompt token 主要只通过 40 层主干的前 20 层，而 Decode 的新 token 必须通过全部 40 层；在每层稀疏激活模式大体相同的情况下，激活参数量因此从约 8B 变为约 16B。**

本文相关结构和配置依据：`D:\Code\vllm_notes\DeepSeekV41-Flash\DeepSeek-V4.1-Flash Pushing the Limits of KV Cache Compression.pdf`，重点参见模型总览、CED/CSA2、Engram 配置和 SWA Bounded Replay 相关章节。
