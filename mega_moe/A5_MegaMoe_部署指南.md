# A5 (Ascend950DT) MegaMoe 安装部署指南

适用范围：Kimi-K3 W4A8 MXFP dummy 权重在 Ascend 950DT (A5) 上启用 CANN MegaMoe 融合算子（可选叠加 DSpark 投机解码）。

参考脚本：`/mnt/share/l00889328/10t/224_dummy.sh`（同目录另有 220/228/232 版本，仅 `local_ip` 不同）。

---

## 一、环境准备清单（新容器必做）

| # | 步骤 | 命令 / 说明 |
|---|------|------------|
| 1 | **安装算子二进制** | `cd /mnt/share/l00889328/10t/ops-transformer && bash build.sh --pkg --soc=ascend950 --vendor_name=custom_mega_moe --ops=mega_moe -j16` 然后 `bash build/cann-ops-transformer-custom_mega_moe_linux-aarch64.run`（装到 `${ASCEND_HOME_PATH}/opp/vendors/custom_mega_moe_transformer`） |
| 2 | **安装 Python 包装层（必须，漏了必挂）** | `pip install --no-deps --target /usr/local/Ascend/cann-9.1.0/python/site-packages --upgrade /mnt/share/l00889328/10t/ops-transformer/torch_extension` |
| 3 | **vllm-ascend 代码** | `git fetch myfork && git checkout A5_megamoe_with_DSpark_dummy`（或纯净版 `A5_megamoe`） |
| 4 | **模型配置** | `dummy_kimik3/`（8 层 / 2048 专家 / W4A8 MXFP）与 `dummy_dspark/`（叠加 DSpark 时）目录就位 |
| 5 | **编包联网**（仅重编算子包时） | 需公司网关 CA 已入系统信任库，否则 eigen 等三方下载报 SSL status 60 |

### 安装自检

```bash
python3 -c "
from vllm_ascend.ops.fused_moe.mega_moe_adapter import probe_cann_mega_moe_api as p
c = p(); print(c.available, c.supports_situ, c.supports_comm_context_preload)"
# 预期输出: True True True
```

任一为 False 的含义：

- `available=False` → 包装层/算子 API 缺失（第 1 或 2 步没做）
- `supports_situ=False` → **第 2 步没生效**（旧包装层在位，服务日志会报 `installed MegaMoe API does not expose SiTU parameters`）
- 首次运行出现 ninja `c++ ...` 编译打屏属正常（JIT 一次性缓存）

---

## 二、参考脚本与环境变量解释（224_dummy.sh）

```bash
#!/bin/bash
# Kimi-K3 TP8+EP dummy-weights serve on Ascend A5 (Ascend950DT, 8 NPUs)
# 随机初始化权重(--load-format dummy), 无需真实 checkpoint
# 层数已在 dummy_kimik3/config.json 中减为 4 层:
#   num_hidden_layers=4, full_attn_layers=[4], kda_layers=[1,2,3] (层编号 1-indexed)
local_ip="141.61.52.224"
nic_name="enp35s0f2"

# 以下环境变量无需修改
# triton 编译的临时目录必须指向本地盘, /mnt/share 是网络盘会导致 TemporaryDirectory 清理失败
export TMPDIR=/tmp
# 本机 8 卡全空闲, 默认全部使用; 需要指定芯片时再取消注释
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

export HCCL_BUFFSIZE=800
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"

export VLLM_ASCEND_FAKE_ACCEPT_RATE=0.8

SPECULATIVE_CONFIG="$(
  printf \
'{"method":"dspark","model":"%s","num_speculative_tokens":7,"draft_tensor_parallel_size":8,"max_model_len":4096,"draft_sample_method":"greedy","enforce_eager":true}' \
    "/mnt/share/l00889328/10t/dummy_dspark"
)"
# /mnt/share/weights/Kimi-K3-DSpark
# /mnt/share/y00823936/Inferact-Kimi-K3-DSpark

# A3 机器上的 mooncake 传输库, 本机(A5)未安装, 暂不需要
# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
# vllm-ascend 启动时会把 _cann_ops_custom/vendors 设为 ASCEND_CUSTOM_OPP_PATH, 会挡住
# CANN opp/vendors 下自装的自定义算子包(如 custom_mega_moe)。这里显式追加, 保证
# mega_moe 新算子也被加载(否则 aclnnMegaMoe 解析到内置旧版, 报 moeExpertNum=0)。
# export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/custom_mega_moe_transformer${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/custom_mega_moe_transformer/op_api/lib/:${LD_LIBRARY_PATH}
# 本机 checkout 位于 /mnt/share/l00889328/10t (与 editable 安装同源)
export PYTHONPATH=/mnt/share/l00889328/10t/vllm:/mnt/share/l00889328/10t/vllm-ascend:$PYTHONPATH
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=10000
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

# 清理 torch 编译缓存: 缓存 key 未覆盖模型结构类配置(层数/aux层/专家数),
# 改配置后撞上旧编译产物会报 "too many values to unpack (expected N)"
rm -rf /root/.cache/vllm/torch_compile_cache

MODEL_DIR=${MODEL_DIR:=/mnt/share/l00889328/10t/dummy_kimik3}

vllm serve "${MODEL_DIR}" \
    --served-model-name kimi \
    --load-format dummy \
    --port 8088 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-num-seqs 1 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --tokenizer-mode kimi_k3 \
    --limit-mm-per-prompt '{"vision_chunk": 40}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --no-enforce-eager \
    --quantization ascend \
    --additional-config '{"enable_flashcomm1":false, "enable_shared_expert_dp": false, "multistream_overlap_shared_expert": false, "enable_fused_mc2": 1}' \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode data \
    --speculative-config "$SPECULATIVE_CONFIG" \

```

### 通用环境变量

| 变量 | 含义 |
|------|------|
| `TMPDIR=/tmp` | triton 编译临时目录。必须指本地盘——`/mnt/share` 是网络盘，会导致 TemporaryDirectory 清理失败 |
| `HCCL_IF_IP` / `HCCL_SOCKET_IFNAME` | HCCL 通信使用的 IP / 网卡 |
| `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` | GLOO 与 TP 进程组通信网卡 |
| `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` | NPU 内存分配器启用可扩展段，减少碎片 |
| `OMP_PROC_BIND=false` / `OMP_NUM_THREADS=1` | 关闭 OpenMP 绑核、单线程——多进程 rank 场景避免线程争抢 |
| `TASK_QUEUE_ENABLE=1` | 启用 NPU 任务队列异步下发 |
| `HCCL_BUFFSIZE=800` | HCCL 通信缓冲大小（MB） |
| `HCCL_OP_EXPANSION_MODE="CCU_SCHED"` | HCCL 算子展开模式走 CCU 调度 |
| `ASCEND_CONNECT_TIMEOUT` / `ASCEND_TRANSFER_TIMEOUT` | NPU 连接 / 传输超时（ms） |
| `VLLM_ENGINE_READY_TIMEOUT_S` / `VLLM_RPC_TIMEOUT` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | vLLM 引擎就绪 / RPC / 单步执行超时——大模型冷启动慢，全部放大 |
| `PYTHONPATH=.../vllm:.../vllm-ascend` | 使用 10t 下的本地 checkout（与 editable 安装同源） |

### MegaMoe / DSpark 专属环境变量

| 变量 | 含义 |
|------|------|
| `LD_LIBRARY_PATH+=.../custom_mega_moe_transformer/op_api/lib/` | **MegaMoe 关键**：让自定义算子包的 `libcust_opapi.so`（含新版 `aclnnMegaMoe`）可被动态链接找到。不设或算子包没装 → 解析到 CANN 内置旧版算子，服务报 `moeExpertNum must be > 0, got 0` |
| `ASCEND_CUSTOM_OPP_PATH`（脚本中注释态备选） | 另一条生效路径：把自定义 vendor 目录加入 CANN 算子发现路径。注意 vLLM-Ascend 启动时会把自己的 vendors 目录 prepend 到此变量，可能遮蔽自装包——如遇到遮蔽问题取消注释此行 |
| `VLLM_ASCEND_FAKE_ACCEPT_RATE=0.8` | **DSpark 测试用**：强制投机解码采信率。每条序列前 `ceil(0.8×(投机步长+1))` 个槽位直接以 `target_argmax[0]` 填充并跳过 verify（dummy 权重下真实采信率为 0，用于压测/演示）。设 `0` 或 `1` 恢复正常 verify |

### 启动前动作与关键启动参数

```bash
rm -rf /root/.cache/vllm/torch_compile_cache
```

> torch 编译缓存 key 未覆盖模型结构类配置（层数 / aux 层 / 专家数），改配置后撞旧编译产物会报
> `too many values to unpack (expected N)`。脚本每次启动前清理，代价是每次多 1~2 分钟编译。

| 启动参数 | 说明 |
|---------|------|
| `--quantization ascend` + 模型 W4A8 MXFP (group_size=32) | A5 MegaMoe 契约仅支持此量化 |
| `--tensor-parallel-size 8 --enable-expert-parallel` | TP8 + EP8，2048 专家需被 EP 整除 |
| `--additional-config '{"enable_fused_mc2": 1, ...}'` | **MegaMoe 总开关**；与 `multistream_overlap_shared_expert` 互斥（后者会被强制关闭） |
| `--speculative-config '{"method":"dspark","model":".../dummy_dspark","num_speculative_tokens":7,...}'` | DSpark 投机解码；`num_speculative_tokens=7` → verify 输出 `bs×8` |
| `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` + `--no-enforce-eager` | decode 阶段入图 |

---

## 三、启动与验证

```bash
cd /mnt/share/l00889328/10t
bash 224_dummy.sh > serve.log 2>&1 &
tail -f serve.log    # 等待 "Application startup complete"
```

日志关键检查点（按出现顺序）：

1. `CANN MegaMoe layer capability: supported=True, quant=QuantType.W4A8MXFP, activation=...situglu..., reason=supported` ← 能力判定通过
2. `CANN MegaMoe sym-buffer alloc (must match across all EP ranks): ... num_experts=2048 num_topk=32` ← MegaMoe 缓冲分配成功
3. `Application startup complete`

curl 验证：

```bash
unset http_proxy https_proxy
curl -s http://141.61.52.224:8088/v1/models    # 200
curl -s http://141.61.52.224:8088/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"kimi","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
# 输出乱码属预期（dummy 随机权重），finish_reason=length 即正常
```

投机解码指标：

```bash
curl -s http://141.61.52.224:8088/metrics | grep -E "draft_tokens_total|accepted_tokens_total" | grep -v "#"
```

---

## 四、常见问题速查

| 症状 | 原因 | 修法 |
|------|------|------|
| `moeExpertNum must be > 0, got 0` | 算子解析到内置旧版（`LD_LIBRARY_PATH` 没指 vendor 或被 `ASCEND_CUSTOM_OPP_PATH` 遮蔽） | 确认第 1 步已装包；确认脚本 `LD_LIBRARY_PATH` 行生效；必要时取消 `ASCEND_CUSTOM_OPP_PATH` 注释 |
| capability `reason=installed MegaMoe API does not expose SiTU parameters` | 第 2 步（torch_extension pip）没生效 | 在该机重跑第 2 步命令，用自检命令验收 `True True True` |
| `too many values to unpack (expected N)`（N 较大） | torch 编译缓存与当前模型结构不匹配 | `rm -rf /root/.cache/vllm/torch_compile_cache` 后重启 |
| `DFlash drafter expects 35840 ... but received 7168` | DSpark draft 需 5 个不同层隐状态，目标模型层数不足或 `target_layer_ids` 越界 | 目标模型 ≥6 层（当前 8 层）；`dummy_dspark/config.json` 的 `target_layer_ids=[1,2,3,5,6]`（+1 后须落在层范围内，优先含 MLA 层） |
| capability `supported=False, reason=A5 MegaMoe does not support <量化>` | 模型量化不是 W4A8 MXFP group_size=32 | 检查 `quant_model_description.json` 专家权重为 `W4A8_MXFP`、`group_size=32` |
| 编包下载 eigen 报 SSL status 60 | 公司代理 TLS 拦截，网关 CA 不在信任库 | 将 `Huawei Web Secure Internet Gateway CA V2` 装入 `/etc/pki/ca-trust/source/anchors/` 后 `update-ca-trust` |
| 重启前残留进程占卡 | 上次服务未清干净 | `bash /mnt/share/l00889328/10t/clean_npu.sh` |

---

## 五、单算子自检 UT（mega_moe 边界验证）

脚本：`/mnt/share/l00889328/10t/megamoe_selftest.py`（官方 demo 移植版，绕开 vllm-ascend 框架直接调算子，用于隔离定位"框架问题 vs 算子/环境问题"）。

**验证内容**：A8W4 场景（FP8 激活 + MXFP4 权重，对应 Kimi-K3 W4A8 MXFP），
**2048 专家 + topK 32 双边界值**，单机 2 卡 EP2，权重 FP4 打包 format-29 + E8M0 scale。

```bash
cd /mnt/share/l00889328/10t
ASCEND_RT_VISIBLE_DEVICES=0,1 python3 megamoe_selftest.py
```

**预期输出**：

```
[rank N] symm buffer ok: num_experts=2048 ep_world=2
[OK] device_N finish, y=(256, 4096) dtype=torch.bfloat16 expert_token_nums[:4]=[...]
RESULT: all ranks finished -> [0, 1]
```

**判读**：
- 通过 → 算子包/环境层（二进制 + Python 包装层 + 加载路径）全部正常；若服务仍异常，问题在框架传参
- `moeExpertNum must be > 0, got 0` → 环境层问题（vendor 未加载，见第四节速查表第一条）
- 卡在权重生成阶段无输出 → 正常现象，每 rank 需生成 ~24GiB float32 随机权重，等 2~5 分钟

**参数速记**（改脚本头部变量即可调整）：`E`=每 rank 本地专家数（= num_experts / EP）、
`topK`、`num_experts` 全局专家数、`scene` 可选 A8W8 / A8W4 / A4W4。
