# Qwen3.6 推理过程中的昇腾关键算子

## 问题

> 在本目录下的 vllm 和 vllm-ascend 文件夹寻找,qwen3.6 推理过程中的昇腾关键算子。最好可以画图给出执行次序。

---

## 回答

Qwen3.6 在 vllm-ascend 里是一条完整走通昇腾 NPU 自定义算子的路径。关键结论先说:

**Qwen3.6 是「GDN 线性注意力 + 全注意力」混合的 Mamba-Transformer 模型**(`layer_types` 数组决定每层是 `full_attention` 还是 `linear_attention`),昇腾关键算子集中在三条线上:

1. **GDN 线性注意力**(昇腾独有、最重)—— `npu_causal_conv1d_custom` / `npu_recurrent_gated_delta_rule` / `npu_chunk_gated_delta_rule`
2. **全注意力** —— `flash_attn_npu_v3`(FA3)/ `npu_fused_infer_attention_score`(FIA)
3. **通用/MLP/MoE** —— `npu_swiglu`、W8A8 量化 GEMM、MoE GMM

---

## 一、整体执行次序(算子级)

```text
  input_ids
     │
     ▼
  embed_tokens (VocabParallelEmbedding)
     │  hidden_states
     ▼
╔═══════════════════════════════════════════════════════════════════╗
║  for layer in model.layers (按 config.layer_types 混合排列)          ║
║                                                                    ║
║  ┌─ full_attention 层 ──────────────────────────────────────────┐  ║
║  │ input_layernorm (RMSNorm)                                   │  ║
║  │ qkv_proj   (AscendQKVParallelLinear / 量化 GEMM)            │  ║
║  │ q_norm / k_norm (逐头 RMSNorm)                              │  ║
║  │ RoPE       (torch.ops.vllm.npu_rotary_embedding)           │  ║
║  │ FlashAttention:  fa3_v1(flash_attn_npu_v3) / FIA           │  ║
║  │                 └ npu_scatter_pa_kv_cache 写 KV            │  ║
║  │ o_proj     (RowParallelLinear + all-reduce)                │  ║
║  └──────────────────────────────────────────────────────────────┘  ║
║                                                                    ║
║  ┌─ linear_attention (GDN) 层 ─────────────────────────────────┐  ║
║  │ input_layernorm (RMSNorm)                                   │  ║
║  │ in_proj_qkvz / in_proj_ba (线性投影)                        │  ║
║  │ fused_qkvzba_split_reshape_cat (Triton, 仅 GQA 交错布局)   │  ║
║  │ npu_causal_conv1d_custom     ← 短卷积/状态更新(AscendC)    │  ║
║  │ fused_gdn_gating (Triton)    ← 算 g(衰减) 与 β(sigmoid门) │  ║
║  │ l2norm_fwd (q、k 做 L2 归一)                                │  ║
║  │ prefill: npu_chunk_gated_delta_rule   (CANN 融合算子)       │  ║
║  │ decode : npu_recurrent_gated_delta_rule (AscendC 自研算子) │  ║
║  │ output norm (RMSNorm) + out_proj                            │  ║
║  └──────────────────────────────────────────────────────────────┘  ║
║                                                                    ║
║  post_attention_layernorm (RMSNorm)                                ║
║  MLP:  27B→ dense (gate/up/down + npu_swiglu)                     ║
║        35B-A3B→ MoE (moe_gating_top_k + npu_grouped_matmul)       ║
╚═══════════════════════════════════════════════════════════════════╝
     │
     ▼
  final norm (RMSNorm) → lm_head (ParallelLMHead)
     │
     ▼
  logits → sampler → 下一 token
     └ (可选) qwen3_5_mtp 多头投机解码(与 Qwen3.5 共用 MTP head)
```

---

## 二、GDN 线性注意力层 —— 昇腾核心算子细节

这是 Qwen3.6 相比普通 Dense 模型最「昇腾化」的部分,实现主体在 `vllm_ascend/ops/gdn.py` 的 `AscendGatedDeltaNetAttention._forward_core`(gdn.py:259)。

核心算子逐个说明:

| 算子 | 作用 | 实现位置 |
|---|---|---|
| `torch.ops.vllm.qwen_gdn_attention_core` | 统一入口,把 GDN 核心转发到 `_forward_core` | gdn.py:235 |
| `torch.ops._C_ascend.npu_causal_conv1d_custom` | 短卷积(sequence transformation),prefill/decode/spec 三条路径都走它 | gdn.py:312 起;C++ 在 `csrc/moe/causal_conv1d` |
| `fused_gdn_gating`(Triton) | 用 `A_log/a/b/dt_bias` 算衰减 `g` 与门控 `β` | `ops/triton/fused_gdn_gating.py` |
| `l2norm_fwd` | Q/K 的 L2 归一化(来自 flash-linear-attention) | gdn.py:451 |
| `torch_npu.npu_chunk_gated_delta_rule` | **prefill** 的融合 chunk 算子(CANN 内置,一次替代 chunk_scaled_dot_kkt+solve_tril 等整条 Triton 流水线) | gdn.py:138 |
| `torch.ops._C_ascend.npu_recurrent_gated_delta_rule` | **decode** 的递归算子(AscendC 自研,支持 fp32 state、dtype 更宽) | gdn.py:457/565;C++ 在 `csrc/attention/recurrent_gated_delta_rule` |
| `chunk_gated_delta_rule`(Triton) | prefill 的 **fallback** 路径(当 CANN 融合算子不可用或 PCP 场景) | gdn.py:538 |

关键分支逻辑(`gdn.py:515` 附近):

- prefill 优先探测 `torch_npu.npu_chunk_gated_delta_rule`(class 级缓存一次冒烟测试,`_probe_fused_chunk`),可用且非 PCP 时走融合算子,否则退回 Triton `chunk_gated_delta_rule`。
- decode 始终走 `_C_ascend.npu_recurrent_gated_delta_rule`。
- Atlas 300I DUO(310P)走 `npu_recurrent_gated_delta_rule_v310`(`csrc/attention/recurrent_gated_delta_rule_v310`)。

对应的 metadata/图模式封装在 `ops/gdn_attn_builder.py`(`AscendGDNAttentionMetadataBuilder`),负责把 prefill/decode/spec 三类序列切开、生成 `cu_seqlens`、chunk 索引、conv1d 元数据等。

---

## 三、全注意力层算子

Qwen3.6 的 `full_attention` 层复用 `Qwen3NextAttention`,昇腾上有两套后端:

- **FA3**(`attention/fa3_v1.py`):调用 `flash_attn_npu_v3.flash_attn_with_kvcache`(PagedAttention),block_size 128。
- **FIA**(默认):调用 `torch_npu.npu_fused_infer_attention_score`(见 `device/device_op.py:56`)。
- KV 写入:`torch_npu.npu_scatter_pa_kv_cache`(`device_op.py:45`)。
- RoPE:`torch.ops.vllm.npu_rotary_embedding`(`ops/register_custom_ops.py:244` 注册,实现 `ops/rotary_embedding.py`);qwen3_5 风格还用 `triton_split_qkv_rmsnorm_mrope`(`patch_qwen3_5.py:75`)。

---

## 四、通用 / MLP / MoE 算子(两变体都涉及)

| 类别 | 算子 | 说明 |
|---|---|---|
| 归一化 | `ops.rms_norm`、`torch_npu.npu_add_rms_norm_quant` | RMSNorm(量化时融合) |
| 线性 | `torch.ops.vllm.unquantized_gemm` | BF16 走 FRACTAL_NZ 矩阵乘 |
| W8A8 量化 | `torch_npu.npu_quantize`、`torch.ops.vllm.quantize`、`matmul_and_reduce` | `--quantization ascend` 时启用 |
| 激活 | `torch_npu.npu_swiglu`、`npu_fast_gelu` | MLP SwiGLU(`ops/activation.py`) |
| MoE(A3B) | `npu_moe_init_routing_v2`、`moe_gating_top_k`、`npu_grouped_matmul` / `npu_grouped_matmul_swiglu_quant`、`npu_mm_reduce_scatter_base` | `ops/fused_moe/` |
| 集合通信 | `tensor_model_parallel_all_reduce/gather` + FlashComm1 优化 | `ops/register_custom_ops.py:195-222` |

MoE 部分(`Qwen3.6-35B-A3B`)额外涉及 `--enable-expert-parallel` 与 `multistream_overlap_shared_expert`(共享专家与路由专家重叠),算子见 `device_op.py:104/134/209`。

---

## 五、关键算子 → 源文件索引

| 文件 | 关键内容 |
|---|---|
| `vllm_ascend/ops/gdn.py` | GDN 核心:conv1d、gating、recurrent/chunk delta rule |
| `vllm_ascend/ops/gdn_attn_builder.py` | GDN 的 attention metadata/图模式构建 |
| `vllm_ascend/ops/triton/fused_gdn_gating.py` | g/β 门控 Triton 内核 |
| `vllm_ascend/attention/fa3_v1.py` | FA3 全注意力后端 |
| `vllm_ascend/device/device_op.py` | FIA、MoE routing/GMM、scatter、量化等 `torch_npu` 封装 |
| `vllm_ascend/ops/register_custom_ops.py` | `_C_ascend`/`vllm` 自定义 op 注册(量化、RoPE、集合通信) |
| `vllm_ascend/patch/worker/patch_qwen3_5.py` | Qwen3.5/3.6 解码层 + GDN 的 patch 注入 |
| `vllm_ascend/patch/worker/patch_qwen3_dflash.py` | DFlash(DSpark 投机)上下文 KV 预计算 patch |
| `vllm_ascend/models/qwen3_dspark.py` | Qwen3.6 DSpark 投机草稿模型的昇腾版 |
| `csrc/attention/recurrent_gated_delta_rule*/` | 递归 GDN decode 的 AscendC 内核(含 310P 变体) |
| `csrc/moe/causal_conv1d*/` | 因果 conv1d 的 AscendC 内核 |
| `vllm/vllm/model_executor/models/qwen3_5.py` | 上游 Qwen3.5/3.6 混合层结构定义(`layer_types`) |

---

## 总结

**Qwen3.6 在昇腾上的「关键算子」= GDN 线的 `npu_causal_conv1d_custom` + `npu_recurrent_gated_delta_rule`(decode)/ `npu_chunk_gated_delta_rule`(prefill)+ `fused_gdn_gating`,叠加上全注意力线的 `flash_attn_npu_v3`/`npu_fused_infer_attention_score`,以及 W8A8 量化 GEMM、`npu_swiglu` 和(A3B 变体)MoE GMM 算子。**

其中 `npu_recurrent_gated_delta_rule` 与 `npu_causal_conv1d_custom` 是 vllm-ascend 为 GDN 专门自研的 AscendC 内核,是整个推理链路里最「昇腾定制」的部分。
