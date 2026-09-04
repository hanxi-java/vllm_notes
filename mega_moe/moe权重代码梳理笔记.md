# MoE 权重代码梳理笔记

> 本文记录 vllm / vllm-ascend 中 MoE 权重从「加载 → 处理 → 计算」的完整代码链路,以及每一阶段涉及的关键方法。

## 目录

- [一、问题](#一问题)
- [二、背景:MoE 权重的整体架构](#二背景moe-权重的整体架构)
- [三、阶段 0:量化方法选择](#三阶段-0量化方法选择)
- [四、阶段 1:权重创建(声明空张量)](#四阶段-1权重创建声明空张量)
- [五、阶段 2:权重加载(填值)](#五阶段-2权重加载填值)
- [六、阶段 3:加载后处理(转格式、打包、拆专家)](#六阶段-3加载后处理转格式打包拆专家)
- [七、阶段 4:前向计算(真正算 MoE)](#七阶段-4前向计算真正算-moe)
- [八、完整链路图](#八完整链路图)
- [九、两条路径对比(未量化 vs 量化)](#九两条路径对比未量化-vs-量化)
- [十、select_moe_comm_method 调用链路](#十select_moe_comm_method-调用链路)

---

## 一、问题

结合 vllm 和 vllm-ascend 的代码,说说 MoE 权重从加载到处理、再到后面的 MoE 计算,都需要经过哪些关键方法?

---

## 二、背景:MoE 权重的整体架构

在 vLLM 的 fused MoE 架构里,一个 MoE 层的职责被拆成两部分:

1. **权重所有者 `RoutedExperts`**:持有 MoE 的权重张量(`w13_weight`、`w2_weight` 等)。昇腾侧对应 `AscendRoutedExperts`(`vllm_ascend/ops/fused_moe/routed_experts.py:202`)。
2. **执行方法 `quant_method`**:决定"怎么算"。它是一组方法的集合,负责声明权重、处理权重、执行前向。昇腾侧对应 `AscendUnquantizedFusedMoEMethod`(未量化)或 `AscendFusedMoEMethod`(量化,内部包着 `AscendMoEScheme` 子类)。

所以 MoE 权重生命周期里的每一步,本质都是「`quant_method` 里的某个方法」被 vLLM 的框架在合适时机调用。

下面按时间顺序,把四个阶段的关键方法串起来。

---

## 三、阶段 0:量化方法选择

**关键方法:`QuantizationConfig.get_quant_method()`**

构建模型时,vLLM 为每个 MoE 层调用 `get_quant_method(layer, prefix)`,决定该层用哪个 `quant_method`。昇腾的量化配置在这里返回昇腾自己的方法类:

- 未量化 → `AscendUnquantizedFusedMoEMethod`
  - 参考:`vllm_ascend/quantization/modelopt_mxfp8_config.py:67-68`
  - 参考:`vllm_ascend/quantization/modelslim_config.py:768-774`
- 量化(如 W4A8)→ `AscendFusedMoEMethod`(适配器),内部包着 `AscendMoEScheme` 子类(如 `AscendW4A8DynamicFusedMoEMethod`)
  - 参考:`vllm_ascend/quantization/modelslim_config.py:781`
  - 适配器定义:`vllm_ascend/quantization/method_adapters.py:190`

这一步是后续所有阶段的"分水岭"——选了哪个方法,后面就走哪套逻辑。

---

## 四、阶段 1:权重创建(声明空张量)

**关键方法:`create_weights` → `get_weight` / `get_dynamic_quant_param`**

模型构建时,需要先声明各权重的形状和 dtype(此时值还没填)。

- **未量化路径**:`UnquantizedFusedMoEMethod.create_weights`
  - 位置:`vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py:55`
  - 创建 `w13_weight`(gate_proj + up_proj 融合,形状 `[num_experts, 2*intermediate, hidden]`)和 `w2_weight`(down_proj,形状 `[num_experts, hidden, intermediate]`)。

- **量化路径(以 W4A8 为例)**:`AscendW4A8DynamicFusedMoEMethod` 的方法
  - `get_weight`(`vllm_ascend/quantization/methods/w4a8.py:146`):声明 int8 权重张量(INT4 打包在 int8 里)。
  - `get_dynamic_quant_param`(`w4a8.py:163`):声明 per-channel 的 scale/offset 等量化参数。
  - `get_dynamic_quant_param_compressed_tensors`(`w4a8.py:173`)与 `get_dynamic_quant_param_modelslim`(`w4a8.py:183`):分别对应 LLM-Compressor 和 msModelSlim 两种权重来源。

这个阶段的产出:一批**形状正确、值为空**的参数对象。

---

## 五、阶段 2:权重加载(填值)

**关键方法:`load_weights` → `weight_loader`**

vLLM 的 model loader(`vllm/model_executor/model_loader/base_loader.py:80`)遍历 checkpoint,把磁盘上的权重数值填进阶段 1 声明的参数里。

- 入口:`base_loader.py` 里的 `load_weights`。
- 每层通过自己的 `weight_loader` 完成具体填充(处理 TP/EP 切分、packed module 映射等)。

这一阶段基本是上游 vLLM 的逻辑,昇腾的差异主要在于阶段 1 声明了正确的形状、以及给 MoE 权重装了特定的 loader(例如 `modelslim_config.py:778` 里 `layer.weight_loader = _make_modelslim_moe_weight_loader(...)`),让 checkpoint 能正确对位。

---

## 六、阶段 3:加载后处理(转格式、打包、拆专家)

> 这是昇腾差异最大、也最关键的一个阶段。

**关键入口:`vllm/model_executor/model_loader/utils.py:97` 的 `process_weights_after_loading()`**

所有权重加载完成后,vLLM 会遍历整个模型,对每个带 `quant_method` 的模块调用:

```python
quant_method.process_weights_after_loading(module)   # utils.py:113
```

这一步把「checkpoint 里的原始权重」加工成「昇腾融合算子要的形态」。分两条路径:

### 6.1 未量化路径:`AscendUnquantizedFusedMoEMethod.process_weights_after_loading`

位置:`vllm_ascend/ops/fused_moe/routed_experts.py:75`

```python
def process_weights_after_loading(self, layer):
    super(...).process_weights_after_loading(layer)
    # 1. 转置:把 [E, out, in] 变成 [E, in, out]
    w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
    replace_parameter(layer, "w13_weight", w13_data)
    w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
    replace_parameter(layer, "w2_weight", w2_data)

    # 2. 根据是否开启 fused_mc2 决定格式
    enable_fused_mc2 = get_ascend_config().enable_fused_mc2
    if enable_fused_mc2:
        # 融合算子只认 FRACTAL_NZ 分形布局
        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
        # 动态 EPLB 时再按专家拆成 list
        if enable_fused_mc2 == 1 and self.dynamic_eplb:
            layer.w13_weight_list = [w.clone() for w in layer.w13_weight.data.unbind(dim=0)]
            layer.w2_weight_list = [w.clone() for w in layer.w2_weight.data.unbind(dim=0)]
            del layer.w13_weight
            del layer.w2_weight
            torch.npu.empty_cache()
    else:
        layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
        layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)
```

关键点:
- **`transpose(1, 2)`**:把权重从 `[E, out, in]` 转成 `[E, in, out]`,匹配昇腾算子的权重方向约定。
- **`npu_format_cast(FRACTAL_NZ)`**:把权重从 ND 连续布局转成分形分块布局(`[E, in/16, out/16, 16, 16]`),这是昇腾 Cube 矩阵计算单元需要的权重排布。
- **`unbind(dim=0)` + `clone`**:按第 0 维(专家维)把整块 `[E, ...]` 拆成"每个专家一个 tensor"的 list,供动态 EPLB 单独迁移专家。

### 6.2 量化路径:`AscendW4A8DynamicFusedMoEMethod.process_weights_after_loading`

位置:`vllm_ascend/quantization/methods/w4a8.py:326`

```python
def process_weights_after_loading(self, layer):
    # 1. 按来源分发
    if self.quant_method == COMPRESSED_TENSORS_METHOD:
        self.process_weights_after_loading_compressed_tensors(layer)
    else:
        self.process_weights_after_loading_modelslim(layer)
    # 2. 转 NZ
    layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
    layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)
    # 3. 拆 per-expert list 或打包 int32
    ...
```

两个子方法做的事:

- `process_weights_after_loading_compressed_tensors`(`w4a8.py:369`):转置 → 在线算 `scale_bias = 8 * (weight * scale).sum(axis=1)` → `_process_scale` 处理 scale → `_pack_int4_to_int8` 把 int4 打包成 int8。
- `process_weights_after_loading_modelslim`(`w4a8.py:396`):转置 → `_process_scale` → scale_bias 转置求和 → squeeze。

几个关键辅助方法(都在 `w4a8.py`):

| 方法 | 作用 |
|---|---|
| `_pack_int4_to_int8`(315) | 两个 int4 按位打包进一个 int8 |
| `_pack_to_int32`(301) | 4 个 int8(8 个 int4)打包成 1 个 int32 |
| `_process_scale`(308) | float32 scale 的二进制位重解释成 int64 |
| `maybe_squeeze_per_channel_weight_scale`(295) | `[E, out, 1]` 压成 `[E, out]` |

---

## 七、阶段 4:前向计算(真正算 MoE)

**关键链路:`forward_modular` → `apply` → `fused_experts` → token dispatch/MLP/combine**

### 7.1 入口:`RoutedExperts.forward_modular`

位置:`vllm/model_executor/layers/fused_moe/routed_experts.py:1196`

```python
def forward_modular(self, x, topk_weights, topk_ids, ...):
    return self.quant_method.apply(
        layer=self, x=x, topk_weights=topk_weights, topk_ids=topk_ids, ...
    )
```

把路由结果(`topk_weights` / `topk_ids`)和输入 `x` 交给 `quant_method.apply`。

### 7.2 打包:`apply`

- 未量化:`AscendUnquantizedFusedMoEMethod.apply`(`routed_experts.py:107`)
- 量化:`AscendW4A8DynamicFusedMoEMethod.apply`(`w4a8.py:201`)

核心动作:
1. 根据当前算子路径(MegaMoe / 旧 dispatch_ffn_combine / 普通)选权重的正确形态(原始 int8 list、`view(int32)` list、或整块 tensor);
2. 调用 `build_fused_experts_input(...)` 打包成 `MoEFusedExpertsInput`。

### 7.3 执行:`moe_comm_method.fused_experts`

位置:`vllm_ascend/ops/fused_moe/moe_comm_method.py:118`

这是昇腾 MoE 通信方法的统一入口,内部三步:

```python
# ① token 分发(EP 通信:把 token 按专家路由到对应 rank)
token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input)

# ② 专家 MLP 计算
mlp_output, before_gmm2_evt = self._apply_mlp(mlp_compute_input)

# ③ token 回收(EP 通信:把结果送回原 rank)
routed_out = self.token_dispatcher.token_combine(mlp_output, ...)
```

- `token_dispatch` / `token_combine` 由不同的 `TokenDispatcherWith*` 实现(`MC2` / `AllGather` / `All2AllV`),负责 token 的跨卡交换。
- `_apply_mlp` → `unified_apply_mlp` 调用底层算子。
- 融合路径(FusedMC2)则走 `_apply_cann_mega_moe`(`moe_comm_method.py:384`)直接调 `mega_moe`,或 `dispatch_ffn_combine`(`moe_comm_method.py:503`)。

### 7.4 底层算子

最终落到 CANN 的融合算子:`mega_moe`、`dispatch_ffn_combine`、`npu_moe_distribute_dispatch/combine` 等,在 NPU 上完成「token 交换 + 专家 GEMM」。

---

## 八、完整链路图

```
[阶段0] get_quant_method()
          └─→ 选 quant_method(未量化 / 量化)
                                   │
[阶段1] create_weights / get_weight / get_dynamic_quant_param
          └─→ 声明空权重(形状 + dtype)
                                   │
[阶段2] load_weights / weight_loader
          └─→ 从 checkpoint 填数值
                                   │
[阶段3] utils.process_weights_after_loading()
          └─→ quant_method.process_weights_after_loading(layer)
                ├─ transpose(1, 2)
                ├─ pack int4 / int32 / scale 位重解释
                ├─ npu_format_cast(FRACTAL_NZ)
                └─ unbind → per-expert list
                                   │
[阶段4] forward_modular()
          └─→ quant_method.apply()
                ├─ 选权重形态(int8 list / int32 view / 整块 tensor)
                └─→ build_fused_experts_input()
                      └─→ moe_comm_method.fused_experts()
                            ├─ token_dispatcher.token_dispatch()   (EP 分发)
                            ├─ _apply_mlp / _apply_cann_mega_moe   (专家计算)
                            └─ token_dispatcher.token_combine()    (EP 回收)
                                  └─→ CANN 融合算子 (mega_moe / dispatch_ffn_combine / MC2)
```

---

## 九、两条路径对比(未量化 vs 量化)

| 阶段 | 未量化 `AscendUnquantizedFusedMoEMethod` | 量化 `AscendW4A8DynamicFusedMoEMethod` |
|---|---|---|
| 阶段1 声明 | `create_weights` 直接建 `w13_weight`/`w2_weight` | `get_weight` + `get_dynamic_quant_param` |
| 阶段3 后处理 | transpose + FRACTAL_NZ + per-expert split | transpose + int4/int32 打包 + scale 位重解释 + scale_bias + NZ + split |
| 阶段4 前向 | `quant_type=NONE`,空 scale/bias 占位 | `quant_type=W4A8`,带 per-channel scale/bias |
| 权重 dtype | BF16/FP16(原生浮点) | int8(内含打包的 int4) |

**总结一句话**:MoE 权重走 4 个阶段——**声明 → 加载 → 后处理 → 计算**,跨 `get_quant_method` / `create_weights` / `load_weights` / `process_weights_after_loading` / `apply` / `fused_experts` 这几个关键方法。昇腾的差异集中在**阶段 3(格式加工)**和**阶段 4(走昇腾自己的 MC2/MegaMoe 融合算子)**,阶段 0-2 基本复用 vLLM 框架。

---

## 十、select_moe_comm_method 调用链路

> 记录 `select_moe_comm_method`(决定 MoE 走 MegaMoe / MC2 / AllGather / AllToAll 哪条算子路径)的**完整调用链路**,重点标出**前向入口**。

### 10.1 关键前提:per-forward,不是 per-layer

- `select_moe_comm_method` 是**每次前向只调一次**的函数,在模型前向开始前决定整次前向的通信方式,对这次前向的**所有层**统一生效。
- 结果 `moe_comm_type` / `moe_comm_method` 缓存进 `forward_context`;之后各 MoE 层只**读缓存**,不重算。
- 层读缓存的位置:`ops/fused_moe/routed_experts.py` 的 `apply()`(读 `_EXTRA_CTX.moe_comm_method` @137、`_EXTRA_CTX.moe_comm_type` @141)和 `forward_impl()`(读 `_EXTRA_CTX.moe_comm_method` @532/604)。

因此 `skip_mega_moe`(per-layer 字段)只能**提升成 per-forward 布尔**(主=0 / 草稿=1)传给 `select_moe_comm_method`,无法被它直接读到某个层的 `self.skip_mega_moe`。

### 10.2 调用点清单

| # | 场景 | 调用位置 | num_tokens 传入 | 是否草稿 | 是否前向 |
|---|---|---|---|---|---|
| ① | profile_run 判断是否额外 dummy | `model_runner_v1.py:3478` / `worker/v2/model_runner.py:255` | `mc2_capacity` | 否 | 否(判条件) |
| ② | 主模型前向(含 dummy 前向) | `ascend_forward_context.py:119`(v1) / `platform.py:553`(v2) | 真实 batch 或 dummy capacity | 否 | **是** |
| ③ | 草稿模型前向(propose / dummy) | `ascend_forward_context.py:119`(v1,`is_draft_model=True`) | draft batch | **是** | **是** |
| ④ | DP allreduce 跳过决策 | `utils.py:1140` | `potential_max_tokens` / `max_num_batched_tokens` | 可草稿 | 否(init) |
| ✗ | v2 草稿图重放 | (无) | — | 是 | **是**(但不调) |

### 10.3 完整链路图

图例:`▶▶` = 前向入口;`──select_moe_comm_method──` = 调用点;`~~~` = 前向算子内部(每层只读缓存);`✗` = 不调 `select_moe_comm_method`。

**阶段 0:初始化(非前向,进程启动时)**

```
model_runner.__init__ / DP 配置决策
 └─ utils.should_skip_allreduce_across_dp_group()        utils.py:1104
     └─ needs_mc2(n)                                      utils.py:1139
         └──select_moe_comm_method(n, vllm_config)──      utils.py:1140
```

**阶段 1:启动 profile / dummy run —— V1 runner(`model_runner_v1.py`)**

```
profile_run()                                              model_runner_v1.py:3468
 └─ if max_num_tokens > mc2_capacity:
 │   └──select_moe_comm_method(mc2_capacity, ...)──        model_runner_v1.py:3478   ← ① 判断
 └─ _dummy_run(mc2_capacity, is_profile=True)              model_runner_v1.py:3147
     ├─▶▶ [主模型 dummy 前向入口]
     │   with set_ascend_forward_context(                  model_runner_v1.py:3405
     │            in_profile_run=True)
     │   └──select_moe_comm_method(...)──                  ascend_forward_context.py:119   ← ②
     │   └─▶▶ _model_forward(...)                          model_runner_v1.py:3418 → 2637
     └─ if self.drafter:                                   model_runner_v1.py:3427
         └─▶▶ [草稿模型 dummy 前向入口]
             self.drafter.dummy_run(...)                   llm_base_proposer.py:548
             └─ with set_ascend_forward_context(           llm_base_proposer.py:699
                      is_draft_model=True)
             └──select_moe_comm_method(..., is_draft_model=True)──  ascend_forward_context.py:119   ← ③
             └─▶▶ self._runnable(...)                      llm_base_proposer.py:719
```

**阶段 1:启动 profile / dummy run —— V2 runner(`worker/v2/model_runner.py`)**

```
profile_run()                                              worker/v2/model_runner.py:244
 └─ if max_num_tokens > mc2_capacity:
 │   └──select_moe_comm_method(mc2_capacity, ...)──        worker/v2/model_runner.py:255   ← ①
 └─ _dummy_run(...)
     └─▶▶ platform.set_additional_forward_context()        platform.py:467
         └──select_moe_comm_method(...)──                  platform.py:553   ← ②
         └─▶▶ model forward
```

**阶段 2:真实请求处理(execute_model,每步一次)—— V1 runner**

```
execute_model()                                            model_runner_v1.py:1789
 ├─▶▶ [主模型前向入口]
 │   with set_ascend_forward_context(...)                  model_runner_v1.py:2114
 │   └──select_moe_comm_method(...)──                      ascend_forward_context.py:119   ← 主前向
 │   └─▶▶ _model_forward(...)                              model_runner_v1.py:2136 → 2637
 └─ if 投机解码 (MTP):
     propose_draft_token_ids(...)                          model_runner_v1.py:1458
     └─ self.drafter._propose(...)                         model_runner_v1.py:1658
         └─ _propose(...)                                  llm_base_proposer.py:742
             └─▶▶ [草稿模型前向入口]
                 with set_ascend_forward_context(          llm_base_proposer.py:1016
                          is_draft_model=True)
                 └──select_moe_comm_method(..., is_draft_model=True)──  ascend_forward_context.py:119   ← 草稿前向
                 └─▶▶ self._runnable(...)                  llm_base_proposer.py:1046
```

**阶段 2:真实请求处理 —— V2 runner(ACL graph)**

```
execute_model()                                            worker/v2/model_runner.py:203
 ├─▶▶ [主模型前向入口]
 │   platform.set_additional_forward_context(...)          platform.py:467
 │   └──select_moe_comm_method(...)──                      platform.py:553   ← 主前向
 │   └─▶▶ model forward(eager 或 graph replay)
 └─ if 投机解码:
     speculator.propose()                                  worker/v2/spec_decode/*/speculator.py:115/106/127
     └─ ACL graph run_fullgraph()                          autoregressive/aclgraph.py:127 / dflash/aclgraph.py:81
         └─▶▶ [草稿模型前向入口 —— 图模式]
             with set_forward_context(...)                 aclgraph.py:152 / :96
             _EXTRA_CTX.is_draft_model = True              aclgraph.py:163 / :106
             ✗ 不调 select_moe_comm_method,不重算 moe_comm_type
             └─▶▶ graph replay(草稿前向)
```

### 10.4 三个关键结论

1. **真正「选 comm」的前向入口共 4 处**(v1):主前向(`model_runner_v1.py:2114`)、草稿前向(`llm_base_proposer.py:1016`)、主 dummy(`:3405`)、草稿 dummy(`:699`);v2 侧只有 `platform.py:553` 一处主前向。
2. **草稿标志 `is_draft_model=True` 已在两个入口传了**(`:1016` propose、`:699` dummy),但主目录 `set_ascend_forward_context` / `select_moe_comm_method` 签名还没接这个参数——这是要让 `skip_mega_moe` 生效的第一处补丁。
3. **v2 图模式的草稿前向(`aclgraph.py:152/96`)既不调 `select_moe_comm_method` 也不重算 `moe_comm_type`**,只设 `_EXTRA_CTX.is_draft_model = True`。要让 `skip_mega_moe` 在 v2 图模式生效,必须在这两处额外:设 `_EXTRA_CTX.skip_mega_moe = True` + 用草稿 `draft_vllm_config` 重算并写回 `moe_comm_type`/`moe_comm_method`。
