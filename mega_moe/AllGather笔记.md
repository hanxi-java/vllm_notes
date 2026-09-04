# A5 上 MoE ALLGATHER 场景笔记

> 来源：vllm-ascend 代码梳理 + 一条 NPU profiler 算子序列的对照分析。
> 核心文件：`vllm_ascend/ascend_forward_context.py`、`vllm_ascend/ops/fused_moe/*`、`vllm_ascend/device/device_op.py`、`vllm_ascend/device/hardware_profile.py`。

---

## 1. A5 上 MoE 通信方法如何选择

入口 `select_moe_comm_method()`（`ascend_forward_context.py:367`），A5 走 `_select_a5_moe_comm_method()`（`ascend_forward_context.py:337`）：

```python
def _select_a5_moe_comm_method(num_tokens, vllm_config, mc2_tokens_capacity,
                               is_draft_model=False, draft_moe_quant_type=QuantType.NONE):
    if is_mega_moe_enabled():
        # A5 的 FUSED_MC2（mega moe）只支持部分 MXFP 量化；MTP draft 不支持时回退
        if is_draft_model and draft_moe_quant_type not in A5_SUPPORT_MEGA_MOE_QUANT_TYPES:
            pass
        else:
            return MoECommType.FUSED_MC2
    num_experts_per_tok = getattr(hf_text_config, "num_experts_per_tok", top_k_experts)
    world_size = vllm_config.parallel_config.world_size_across_dp
    if (num_tokens is None or num_tokens <= mc2_tokens_capacity) and world_size > 1:
        return MoECommType.MC2
    if world_size <= num_experts_per_tok:
        return MoECommType.ALLGATHER      # ← 本文讨论的分支
    return MoECommType.ALLTOALL
```

要点：

- `A5_SUPPORT_MEGA_MOE_QUANT_TYPES = {W4A4MXFP, W4A8MXFP, W8A8MXFP}`（`ascend_forward_context.py:62`）。
- `is_mega_moe_enabled() = (enable_fused_mc2 == 1 and is_mega_moe_supported())`。
- 返回 `ALLGATHER` 的触发条件：`world_size_across_dp <= num_experts_per_tok`（且没走 mega moe / MC2）。
- **`dispatch_ffn_combine`（FUSED_MC2 非 mega 分支）在 A5 上不可达**：它只在 `FusedMC2CommImpl.fused_experts` 里，而 A5 的 policy（`CAPACITY_AND_WORLD_SIZE`，见 `hardware_profile.py:219`）与 `_select_a5_moe_comm_method` 不会选它。

---

## 2. A5 + ALLGATHER 时 MoE 过程执行的算子/方法（总表）

入口 `AscendRoutedExperts.forward_impl`（`routed_experts.py:503`），分 **prepare → 路由 → fused_experts → finalize** 四段。

| 阶段 | 调用链（方法） | 实际执行的算子 | 作用 |
|---|---|---|---|
| **1. prepare** | `moe_comm_method.prepare` → `PrepareAndFinalizeWithAllGather.prepare`（`prepare_finalize.py:381`） | | 输入 gather |
| 1a. 动态量化（仅量化） | `_prepare_with_ep_group`（`prepare_finalize.py:400`） | `torch_npu.npu_dynamic_quant`（W8A8）/ `torch_npu.npu_dynamic_mx_quant`（MXFP，`dst_type=float4_e2m1fn_x2` 或 `float8_e4m3fn`） | 激活动态量化，产出 per-token scale |
| 1b. EP 序列并行 gather | `_prepare_with_ep_group` | `torch.ops.vllm.maybe_all_gather_and_maybe_unpad` | EP 组内 all-gather |
| 1c. DP 路径 gather | `_prepare_with_dp_group`（`prepare_finalize.py:455`） | `nn.functional.pad` + `get_dp_group().all_gather` / `get_pcp_group().all_gather`（HCCL） | DP 组内 pad + all-gather |
| **2. 路由** | `_select_experts`（`routed_experts.py:443`）→ `router._select_experts` | | 选专家 |
| 2a. topk（hash MoE） | `AscendFusedTopKRouter._select_experts`（`fused_topk_router.py:124`） | `torch.ops._C_ascend.moe_gating_top_k_hash` | DeepSeek hash 门控 top-k |
| 2b. topk（普通） | 同上（`fused_topk_router.py:145`） | `DeviceOperator.moe_gating_top_k` → `torch_npu.npu_moe_gating_top_k_softmax`/`..._sigmoid`；或 `torch.topk` | softmax/sigmoid 门控 top-k |
| 2c. 映射/混合 | `_select_experts`（`routed_experts.py:457-501`） | `self.log2phy[topk_ids]`、`torch.cat`、`torch.argsort` | 逻辑→物理映射、shared expert 混合 |
| **3. fused_experts** | 基类 `MoECommMethod.fused_experts`（`moe_comm_method.py:117`） | | dispatch → MLP → combine |
| 3a. dtype 断言 | `fused_experts` | `assert dtype ∈ [fp32/fp16/bf16/int8/fp8_e4m3fn/uint8]` | 入口校验 |
| 3b. token dispatch | `TokenDispatcherWithAllGather.token_dispatch`（`token_dispatcher.py:342`） | `DeviceOperator.npu_moe_init_routing` → `torch_npu.npu_moe_init_routing_v2` | 按专家排序 token，返回 `expanded_row_idx`/`expert_tokens` |
| 3c. MLP（非量化） | `unified_apply_mlp` → `unquant_apply_mlp`（`moe_mlp.py:653`） | `torch_npu.npu_grouped_matmul`（w1）+ 激活（`npu_swiglu`/`F.gelu`/situ/swigluoai/swiglustep）+ `npu_grouped_matmul`（w2） | 分组 matmul + 激活 |
| 3d. MLP（量化，A5） | `unified_apply_mlp` → `quant_apply_mlp`（`moe_mlp.py:194`） | `DeviceOperator.npu_dynamic_quant`；`A5DeviceAdaptor.npu_grouped_matmul_swiglu_quant` → `torch_npu.npu_grouped_matmul_swiglu_quant_v2`（或 MXFP：`npu_grouped_matmul` + `_C_ascend.npu_swiglu_group_quant`）；`A5DeviceAdaptor.npu_grouped_matmul_gmm2` → `npu_grouped_matmul` | 动态量化 + fused GMM-SwiGLU-quant + gmm2 |
| 3e. token combine | `TokenDispatcherWithAllGather.token_combine`（`token_dispatcher.py:426`） | `DeviceOperator.npu_moe_token_unpermute` → `torch_npu.npu_moe_token_unpermute` | 按 `expanded_row_idx` 还原 token 顺序并加权 |
| **4. finalize** | `moe_comm_method.finalize` → `PrepareAndFinalizeWithAllGather.finalize`（`prepare_finalize.py:520`） | | 输出 reduce-scatter |
| 4a. SP 路径 | `_finalize_with_ep_group`（`prepare_finalize.py:538`） | `torch.ops.vllm.maybe_pad_and_reduce`（+ pcp reduce_scatter） | EP reduce-scatter |
| 4b. DP 路径 | `_finalize_with_dp_group`（`prepare_finalize.py:556`） | `get_pcp_group().reduce_scatter`（pcp>1）、`get_dp_group().reduce_scatter`（dp>1）+ 切片 `[:num_tokens]` | DP reduce-scatter 并截断 |

---

## 3. 调用流程链路图

```
AscendRoutedExperts.forward_impl
│
├─① _EXTRA_CTX.moe_comm_method.prepare  (moe_comm_method.py:94)
│   └─ PrepareAndFinalizeWithAllGather.prepare  (prepare_finalize.py:381)
│       ├─ [SP] _prepare_with_ep_group  (prepare_finalize.py:400)
│       │   ├─ torch_npu.npu_dynamic_quant / npu_dynamic_mx_quant  (量化时)
│       │   └─ torch.ops.vllm.maybe_all_gather_and_maybe_unpad
│       └─ [DP] _prepare_with_dp_group  (prepare_finalize.py:455)
│           ├─ nn.functional.pad
│           └─ get_dp_group().all_gather / get_pcp_group().all_gather
│
├─② self._select_experts  (routed_experts.py:443)
│   └─ self.router._select_experts
│       ├─ torch.ops._C_ascend.moe_gating_top_k_hash   (hash MoE)
│       │   或 DeviceOperator.moe_gating_top_k → npu_moe_gating_top_k_softmax/sigmoid
│       ├─ norm_topk_prob（renorm）→ torch.sum / div
│       ├─ self.log2phy[topk_ids]
│       └─ mix_placement: torch.cat / torch.argsort
│
├─③ self.quant_method.apply → moe_comm_method.fused_experts
│   └─ MoECommMethod.fused_experts（基类）(moe_comm_method.py:117)
│       ├─ assert dtype ∈ [...]
│       ├─ TokenDispatcherWithAllGather.token_dispatch  (token_dispatcher.py:342)
│       │   ├─ hidden_states * topk_weights  (apply_router_weight_on_input)
│       │   ├─ expert_map[topk_ids] != -1 → topk_weights * mask
│       │   └─ DeviceOperator.npu_moe_init_routing → torch_npu.npu_moe_init_routing_v2
│       ├─ build_mlp_compute_input
│       ├─ _apply_mlp → unified_apply_mlp  (moe_mlp.py:778)
│       │   ├─ [非量化] unquant_apply_mlp  (moe_mlp.py:653)
│       │   │   ├─ torch_npu.npu_grouped_matmul (w1 gate_up)
│       │   │   ├─ torch_npu.npu_swiglu / F.gelu / situ / swigluoai / swiglustep
│       │   │   └─ torch_npu.npu_grouped_matmul (w2 down)
│       │   └─ [量化 MXFP] quant_apply_mlp  (moe_mlp.py:194)
│       │       ├─ DeviceOperator.npu_dynamic_quant → npu_dynamic_quant / npu_dynamic_mx_quant
│       │       ├─ cumsum_group_list → torch.cumsum
│       │       ├─ A5DeviceAdaptor.npu_grouped_matmul_swiglu_quant → npu_grouped_matmul_swiglu_quant_v2
│       │       └─ A5DeviceAdaptor.npu_grouped_matmul_gmm2 → npu_grouped_matmul
│       └─ TokenDispatcherWithAllGather.token_combine  (token_dispatcher.py:426)
│           └─ DeviceOperator.npu_moe_token_unpermute → torch_npu.npu_moe_token_unpermute
│
└─④ _EXTRA_CTX.moe_comm_method.finalize
    └─ PrepareAndFinalizeWithAllGather.finalize  (prepare_finalize.py:520)
        ├─ [SP] _finalize_with_ep_group → torch.ops.vllm.maybe_pad_and_reduce
        └─ [DP] _finalize_with_dp_group → get_dp_group().reduce_scatter + [ :num_tokens ]
```

关键点：

- **dispatch/combine 固定一对**：`npu_moe_init_routing_v2` + `npu_moe_token_unpermute`，与量化无关。
- **量化只改变 MLP 段**：非量化走 `npu_grouped_matmul ×2 + npu_swiglu`；MXFP 走 `npu_dynamic_mx_quant → cumsum → npu_grouped_matmul_swiglu_quant_v2 → npu_grouped_matmul`。
- **`DeviceOperator` 在 A5 上解析到 `A5DeviceAdaptor`**（`device_op.py:770`），MLP 段用的是 A5 重载版本（才支持 MXFP；`BaseDeviceAdaptor` 对应方法会直接 `raise` "MXFP only supported on A5"）。
- **ALLGATHER 通信（all-gather / reduce-scatter）只在 `dp_size>1` / `pcp_size>1` / SP 时真正触发**；若 `ep_world_size==1` 或 DP=1，这两段退化为空，只剩末尾 TP all-reduce。

---

## 4. trace 算子序列 ↔ ALLGATHER MoE 的对应范围

给定一条 profiler 算子序列，ALLGATHER MoE 的“指纹”范围是 **从 `MoeGatingTopK` 到 `MoeTokenUnpermute`**（含中间 routing 算术），每个 MoE 层出现一次：

```
MoeGatingTopK                                          ← 路由
aclnnReduceSum / aclnnDiv / aclnnInplaceCopy_Cast        ← norm_topk_prob 重归一化
aclnnIndex_IndexAiCore_Index                             ← log2phy[topk_ids] / expert_map[topk_ids]
aclnnNeScalar / aclnnMul                                 ← expert_map != -1, topk_weights * mask
aclnnMoeInitRoutingV3_...                                ← dispatch（npu_moe_init_routing_v2）
DynamicMxQuant                                           ← 激活 MX 量化（npu_dynamic_mx_quant）
aclnnCumsum_...                                          ← group_list 前缀和（cumsum_group_list）
aclnnGroupedMatmulSwigluQuantV2_...                      ← gmm1 + SwiGLU + 量化（npu_grouped_matmul_swiglu_quant_v2）
aclnnGroupedMatmulV4_...                                 ← gmm2（npu_grouped_matmul）
aclnnMoeTokenUnpermute_MoeFinalizeRoutingV2_...          ← combine（npu_moe_token_unpermute）
```

紧跟 `MoeTokenUnpermute` 之后是 shared expert 的 FFN（并行分支）：

```
DynamicMxQuant → aclnnQuantMatmulV5 → SwiGlu → DynamicMxQuant → aclnnQuantMatmulV5
→ aclnnMatmul → aclnnSigmoid → aclnnMul → hcom_allReduce → aclnnAdd → aclnnAdds → aclnnAddRmsNorm
```

trace 算子 ↔ 代码算子映射：

| trace 算子（kernel） | 代码算子/方法 | 阶段 |
|---|---|---|
| `aclnnMatmul_MatMulV3Common`（MoeGatingTopK 前） | `gate_linear.forward`（fp32 线性） | 门控线性 |
| `MoeGatingTopK` | `_C_ascend.moe_gating_top_k_hash` / `npu_moe_gating_top_k_softmax·sigmoid` | 路由 top-k |
| `ReduceSum`+`Div`+`Cast` | `norm_topk_prob` 重归一化 | 路由后处理 |
| `Index` | `log2phy[topk_ids]` / `expert_map[topk_ids]` | 逻辑→物理映射 |
| `NeScalar`+`Mul` | `expert_map != -1` + `topk_weights * mask` | 专家 mask 加权 |
| `MoeInitRoutingV3` | `npu_moe_init_routing_v2` | token dispatch |
| `DynamicMxQuant` | `npu_dynamic_mx_quant` | 激活 MX 量化 |
| `Cumsum` | `cumsum_group_list` | group_list 前缀和 |
| `GroupedMatmulSwigluQuantV2` | `npu_grouped_matmul_swiglu_quant_v2` | gmm1+swiglu+quant |
| `GroupedMatmulV4` | `npu_grouped_matmul` | gmm2 down 投影 |
| `MoeTokenUnpermute` | `npu_moe_token_unpermute` | token combine |
| `QuantMatmulV5`（shared expert） | `npu_quant_matmul` | shared expert FFN |
| `SwiGlu` | `npu_swiglu` | shared expert 激活 |
| `Matmul`+`Sigmoid`+`Mul` | shared expert gate | shared expert 门控 |

关于通信算子：

- trace 里 MoE 前后没有 `hcom_allGather` / `hcom_reduceScatter` ⇒ 该 ALLGATHER 场景是退化的（`ep_world_size==1` 或 `world_size<=num_experts_per_tok` 且无跨 rank token 交换），`dp_size==1` 时 `_prepare_with_dp_group` / `_finalize_with_dp_group` 里的 all_gather/reduce_scatter 都不触发。
- `hcom_allReduce_AicpuKernel_503_X_1` 是 **TP all-reduce**（`tensor_model_parallel_all_reduce`），用于把 routed + shared expert 结果在 TP 组内归约，不是 MoE 内部 dispatch/combine 通信。
- 唯一的 `hcom_allGather__503_0_1`（后接 `ArgMax`）与 MoE 无关。

---

## 5. `aclnnMatmul_MatMulV3Common_MatMulV3` 的调用栈

该 kernel 是 Ascend 上的**标准稠密 GEMM**（非量化、非分组），即 `torch.matmul / torch.mm / torch.addmm / torch.bmm / F.linear` 在 NPU 上最终落到的 ACLNN 算子。

matmul 族对照：

| trace kernel | 对应 | 用途 |
|---|---|---|
| `aclnnMatmul_MatMulV3Common_MatMulV3` | `torch.matmul / F.linear` | 非量化稠密线性 |
| `aclnnQuantMatmulV5_QuantBatchMatmulV3` | `torch_npu.npu_quant_matmul` | 量化线性 |
| `aclnnGroupedMatmulV4_GroupedMatmul` | `torch_npu.npu_grouped_matmul` | MoE 分组 matmul（gmm2） |

通用调用栈：

```
<模型/层>.forward
 └─ nn.Linear / LinearBase / ReplicatedLinear / ColumnParallelLinear.forward
     └─ torch.nn.functional.linear(input, weight, bias)
         └─ torch.matmul / torch.addmm / torch.mm / torch.bmm
             └─ [torch_npu ATen dispatch → ACLNN]
                 └─ aclnnMatmul → MatMulV3Common_MatMulV3 (fp32/fp16/bf16 标准 GEMM)
```

在 MoE trace 里的三处具体调用栈：

**① 路由门控线性（`MoeGatingTopK` 前）**

```
AscendGateLinear.forward  (gate_linear.py:54)
 └─ x.to(torch.float32)                 # 强制 fp32
 └─ ReplicatedLinear.forward(self, x)   # vllm
     └─ F.linear → torch.matmul/addmm → aclnnMatmul → MatMulV3Common
```

**② Shared expert 门控（`Sigmoid`/`Mul` 前）**

```
SharedExperts.part2  (shared_experts.py:158-165)
 └─ self.layer.expert_gate(hidden_states)   # 仅带 expert_gate 的模型（如 Qwen3-Next）
     └─ F.linear → torch.matmul → MatMulV3Common
 └─ F.sigmoid(gate_out) * shared_out        # (Sigmoid → Mul)
```

**③ Shared expert 非量化 FFN（无 quant fallback 分支）**

```
SharedExperts.forward  (shared_experts.py:391-399, else 分支)
 ├─ self.part1(hidden_states)
 │    └─ self.layer.gate_up_proj(hidden_states)   # F.linear → matmul → MatMulV3Common
 └─ self.part2(hidden_states, part1_out)
      ├─ self.layer.act_fn(...)                    # swiglu
      └─ self.layer.down_proj(shared_act)          # F.linear → matmul → MatMulV3Common
```

一句话：`aclnnMatmul_MatMulV3Common_MatMulV3` = `torch.matmul/F.linear` 的非量化稠密 GEMM，来自路由 gate 线性（fp32）、shared expert 门控线性、shared expert 非量化 gate_up/down 线性。量化路径（`npu_quant_matmul`）与 MoE 分组路径（`npu_grouped_matmul`）分别落到 `QuantMatmulV5` / `GroupedMatmulV4`，不会产生这个 kernel。

---

## 6. 附：W4A4_MXFP 与 FUSED_MC2 / dispatch_ffn_combine 的关系（背景）

- `torch.ops._C_ascend.dispatch_ffn_combine` 只在 `FusedMC2CommImpl.fused_experts`（`moe_comm_method.py:448-486`）中，触发条件：`enable_fused_mc2==1` 且 `use_mega_moe==False`，且断言 `w1_scale_bias`/`w2_scale_bias` 均非 None。
- W4A4_MXFP 的 `apply`（`w4a4_mxfp4.py:172`）不传 `w1_scale_bias`/`w2_scale_bias`（默认 None），因此**该断言不会通过**；且 `use_mega_moe==True` 时 `_get_cann_mega_moe_quant_settings(W4A4MXFP)` 会 `raise RuntimeError`。所以 W4A4_MXFP 与 `dispatch_ffn_combine`（FUSED_MC2 非 mega 路径）不兼容。
- 提供 scale_bias 的只有：W4A8（`w4a8.py:205-240`）、W8A8 且 `fused_scale_flag`、非量化 MoE（`routed_experts.py:142-143` 填空 tensor）。
- 基类 `MoECommMethod.fused_experts` 被 `AllGatherCommImpl` / `MC2CommImpl` / `AlltoAllCommImpl` 共用；只有 `FusedMC2CommImpl` 覆写了它。A5 上从不选 `FUSED_MC2`，所以基类 `fused_experts` 就是 A5 MoE 的唯一执行路径。
