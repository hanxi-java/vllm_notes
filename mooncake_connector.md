# MooncakeConnector 技术文档

> 文件路径：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`
>
> 功能：基于 Mooncake TransferEngine 的 KV Cache P2P 传输连接器，用于 EPD（Encoder-Prefill-Decode）分离式推理架构中，在 Prefill 节点和 Decode 节点之间高效传输 KV Cache。

---

## 目录

1. [整体架构](#1-整体架构)
2. [核心数据结构](#2-核心数据结构)
3. [KVCacheTaskTracker：任务生命周期管理](#3-kvcachetasktracker任务生命周期管理)
4. [KVCacheSendingThread：发送端后台线程](#4-kvcachesendingthread发送端后台线程)
5. [KVCacheRecvingThread：接收端后台线程](#5-kvcacherecvingthread接收端后台线程)
6. [MooncakeConnectorWorker：Worker 侧连接器](#6-mooncakeconnectorworkerworker-侧连接器)
7. [MooncakeConnectorScheduler：Scheduler 侧连接器](#7-mooncakeconnectorschedulerscheduler-侧连接器)
8. [MooncakeConnector：统一入口](#8-mooncakeconnector统一入口)
9. [关键流程详解](#9-关键流程详解)
10. [与 EPD Proxy 的协作关系](#10-与-epd-proxy-的协作关系)

---

## 1. 整体架构

### 1.1 文件定位

`mooncake_connector.py` 是 **vLLM-Ascend 的 Mooncake KV 传输连接器实现**，负责在 **Prefill 节点**（kv_producer）和 **Decode 节点**（kv_consumer）之间通过 RDMA/高带宽网络传输 KV Cache。

### 1.2 核心类架构

| 类名 | 职责 | 运行位置 |
|------|------|---------|
| `MooncakeAgentMetadata` | 节点元数据（engine_id、端口、KV cache 内存地址等） | 所有节点 |
| `KVCacheTaskTracker` | 跟踪 KV 传输任务的完成状态 | 发送/接收线程内部 |
| `KVCacheSendingThread` | **发送端后台线程**，监听 decode 节点的元数据请求和完成通知 | Prefill 节点 |
| `KVCacheRecvingThread` | **接收端后台线程**，从 prefill 节点拉取 KV cache 并做格式重组 | Decode 节点 |
| `MooncakeConnectorWorker` | Worker 侧的连接器逻辑（注册内存、启动线程、执行传输） | Worker 进程 |
| `MooncakeConnectorScheduler` | Scheduler 侧的调度逻辑（决定哪些请求需要 KV 传输） | Scheduler 进程 |
| `MooncakeConnectorMetadata` | 请求级别的传输元数据 | Scheduler |
| `MooncakeConnector` | 统一入口，根据 role 分发到 Scheduler 或 Worker | 两者 |

### 1.3 传输流程概览

```
[Prefill Worker]                              [Decode Worker]
     │                                               │
     │ 1. Prefill 完成后，KV cache 在 NPU 内存中      │
     │                                               │
     │ 2. Decode Scheduler 决定需要远程 prefill       │
     │    通过 kv_transfer_params 标记 do_remote_prefill=True
     │                                               │
     │ 3. Decode Worker 启动 KVCacheRecvingThread     │
     │    向 Prefill 的 KVCacheSendingThread 请求元数据 │
     │◄──────── GET_META_MSG ─────────────────────────┤
     │───────── MooncakeAgentMetadata ───────────────►│
     │                                               │
     │ 4. Decode 根据元数据计算 src/dst/length 列表     │
     │    调用 TransferEngine.batch_transfer_sync_read │
     │◄═════════ KV Cache Data (RDMA/Network) ══════►│
     │                                               │
     │ 5. Decode 做 KV cache 格式重组（如果需要）      │
     │                                               │
     │ 6. Decode 发送 DONE_RECVING_MSG                 │
     │◄──────── DONE_RECVING_MSG ─────────────────────┤
     │───────── ACK ────────────────────────────────►│
     │                                               │
     │ 7. Prefill 收到 ACK 后释放 KV block            │
```

---

## 2. 核心数据结构

### 2.1 MooncakeAgentMetadata：节点握手元数据

```python
class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str                          # 当前引擎唯一标识
    te_rpc_port: int                        # Mooncake TransferEngine 的 RPC 端口
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]]  # KV cache 分组到层索引的映射
    block_size: int                         # 逻辑 KV block 的大小（token 数）
    kv_caches_base_addr: list[list[int]]    # 每层 KV cache 张量的内存起始地址
    block_size_scale: list[list[int]]       # 逻辑 block 到物理 tensor block 的缩放比
    num_blocks: int                         # 总逻辑 block 数
    block_lens: list[list[int]]             # 每个 block 的字节长度
    local_ip: str = ""                      # 本地 IP 地址
```

**字段详解与示例**：

假设一个 32 层模型，分 2 个 KV cache group，TP=2，block_size=16：

```python
# kv_group2layeridx: 分组到层索引的映射
{
    0: (
        {
            "layer_names": ["model.layers.0.self_attn", "model.layers.1.self_attn", ..., "model.layers.15.self_attn"],
            "kv_cache_spec_type": "FullAttentionSpec",
            "kv_cache_spec": {"block_size": 16, "num_kv_heads": 8, "head_dim": 128, "dtype": "torch.float16"}
        },
        [0, 1, 2, ..., 15]  # 物理层索引
    ),
    1: (
        {
            "layer_names": ["model.layers.16.self_attn", ..., "model.layers.31.self_attn"],
            "kv_cache_spec_type": "FullAttentionSpec",
            "kv_cache_spec": {"block_size": 16, "num_kv_heads": 8, "head_dim": 128, "dtype": "torch.float16"}
        },
        [16, 17, ..., 31]
    )
}

# kv_caches_base_addr: 每层 K/V cache 的内存地址
# 假设 32 层，每层有 K 和 V 两个 cache
[
    [0x7f0000000000, 0x7f0000100000],  # layer 0: K_cache_addr, V_cache_addr
    [0x7f0000200000, 0x7f0000300000],  # layer 1
    ...
]

# block_size_scale: 逻辑 block 到物理 tensor block 的缩放
# 如果 num_blocks=1024，但 tensor.shape[0]=2048，则 scale=2
[[2, 2], [2, 2], ...]  # 每层每个 cache 的 scale

# block_lens: 每个 block 的字节长度
# block_size=16, num_kv_heads=8, head_dim=128, dtype=fp16 (2 bytes)
# 一个 block 长度 = 16 * 8 * 128 * 2 = 32768 bytes
[[32768, 32768], [32768, 32768], ...]
```

### 2.2 ReqMeta：请求级传输元数据

```python
@dataclass
class ReqMeta:
    local_block_ids: BlockIds               # 本地 block ID 列表
    num_external_tokens: int                # 需要从外部拉取的 token 数
    num_computed_tokens: int                # 本地已计算的 token 数
    remote_block_ids: BlockIds              # 远程 block ID 列表
    remote_host: str                        # 远程节点 IP
    remote_port: int                        # 远程节点 handshake 端口
    remote_engine_id: str                   # 远程引擎 ID
    remote_request_id: str                  # 远程请求 ID
    remote_pcp_size: int                    # 远程 PCP (Prefill Context Parallel) 大小
    remote_dcp_size: int                    # 远程 DCP (Decode Context Parallel) 大小
    remote_ptp_size: int | None             # 远程 PTP (Prefill Tensor Parallel) 大小
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]]  # 跨节点元数据映射
    num_prompt_blocks: int                  # prompt 占用的 block 数
```

### 2.3 GroupPull：分组拉取信息

```python
@dataclass(frozen=True)
class GroupPull:
    group_id: int                           # KV cache 组 ID
    remote_tp_offset: int                   # 远程 TP 偏移
    num_group_pulls: int                    # 该组需要拉取的次数（TP 分片相关）
    prefill_pp_rank: int = 0                # Prefill 的 PP rank
    is_group_transfer_end: bool = False     # 是否该组传输结束
```

### 2.4 SizedDict：带大小限制的 LRU 字典

```python
class SizedDict(OrderedDict):
    def __init__(self, max_size=16000, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)  # LRU 淘汰最旧的

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            value: dict[int, list[int]] = {}
            self[key] = value
            return value
```

用于缓存远程节点的元数据，避免重复请求。

---

## 3. KVCacheTaskTracker：任务生命周期管理

### 3.1 功能

跟踪 KV 传输任务的完成状态，管理请求的生命周期：
- 记录待处理的请求 (`reqs_to_process`)
- 记录已完成的请求 (`finished_requests`)
- 支持延迟释放（等待 decode 节点确认接收完毕）

### 3.2 核心代码

```python
class KVCacheTaskTracker:
    def __init__(self):
        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()           # 已完成的请求
        # 延迟释放队列：记录需要延迟释放的请求及其开始时间
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        self.reqs_to_process: set[str] = set()             # 待处理的请求

    def add_req_to_process(self, request_id: str):
        """将请求加入待处理集合"""
        self.reqs_to_process.add(request_id)

    def add_not_transfer_request(self, request_id: str):
        """标记请求无需传输（本地已完成）"""
        with self.done_task_lock:
            self.finished_requests.add(request_id)
            self.reqs_to_process.discard(request_id)

    def update_done_task_count(self, request_id: str):
        """标记请求传输完成"""
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.finished_requests.add(request_id)
                self.reqs_to_process.discard(request_id)
                self.delayed_free_requests.pop(request_id, None)
            else:
                logger.warning(
                    "MooncakeConnector finish req not in reqs to process. "
                    "request_id=%s. "
                    "Possible cause: Request was already completed or not properly tracked.",
                    request_id,
                )

    def get_and_clear_finished_requests(self) -> set[str]:
        """获取并清空已完成请求集合"""
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            expired_requests = self._retrieve_expired_requests()
            finished_requests.update(expired_requests)
            self.finished_requests.clear()
        return finished_requests

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        """添加延迟释放请求"""
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.delayed_free_requests[request_id] = delay_start_time

    def _retrieve_expired_requests(self):
        """检索所有超时的延迟请求"""
        expired_requests: set[str] = set()
        current_time = time.time()
        while self.delayed_free_requests:
            request_id = next(iter(self.delayed_free_requests))
            delay_start_time = self.delayed_free_requests[request_id]
            if current_time - delay_start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                self.delayed_free_requests.popitem(last=False)
                self.reqs_to_process.discard(request_id)
                expired_requests.add(request_id)
                logger.info(
                    "Force freed expired request: %s. "
                    "Reason: Request exceeded timeout threshold (%s seconds). "
                    "Action: Resources have been forcibly released to prevent memory leak.",
                    request_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
            else:
                break
        return expired_requests
```

### 3.3 使用场景

**延迟释放机制**：
- Prefill 节点完成 KV 计算后，不能立即释放 block
- 需要等待 Decode 节点通过 `DONE_RECVING_MSG` 确认已接收完毕
- 如果 Decode 长时间未确认，超时后强制释放（防止内存泄漏）

---

## 4. KVCacheSendingThread：发送端后台线程

### 4.1 功能

运行在 **Prefill 节点**（kv_producer）上的后台线程，负责：
1. 监听 ZMQ 路由 socket，等待 Decode 节点的连接
2. 响应 `GET_META_MSG`：返回本节点的 `MooncakeAgentMetadata`
3. 响应 `DONE_RECVING_MSG`：Decode 节点通知 KV 已接收完毕，更新任务状态并释放资源

### 4.2 初始化

```python
class KVCacheSendingThread(threading.Thread):
    def __init__(
        self,
        vllm_config: VllmConfig,
        tp_rank: int,
        prefill_tp_size: int,
        local_engine_id: str,
        side_channel_host: str,
        side_channel_port: int,
        metadata: MooncakeAgentMetadata,
        ready_event: threading.Event,
        kv_caches: dict[str, Any],
        pcp_rank: int,
    ):
        super().__init__(daemon=True, name="KVCacheSendingThread")
        self.tp_rank = tp_rank
        self.prefill_tp_size = prefill_tp_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.local_engine_id = local_engine_id
        self.side_channel_host = side_channel_host
        self.side_channel_port = side_channel_port
        self.metadata = metadata
        self.ready_event = ready_event
        self.kv_caches = kv_caches
        self.pcp_rank = pcp_rank
        self.port_send_num: dict[str, int] = {}
        self.task_tracker = KVCacheTaskTracker()
```

### 4.3 端口计算

```python
def run(self):
    """Run the thread to handle KV cache transfer requests."""
    try:
        # 每个 rank 有独立的 handshake 端口，避免冲突
        device_index = self.pp_rank * self.tp_size + self.tp_rank + self.pcp_rank * self.prefill_tp_size
        handshake_port = self.side_channel_port + device_index
        path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
        logger.info(
            "KVCacheSendingThread started listening on path: %s. "
            "Thread: tp_rank=%d, pp_rank=%d, pcp_rank=%d",
            path, self.tp_rank, self.pp_rank, self.pcp_rank,
        )
        with zmq_ctx(zmq.ROUTER, path) as sock:
            self.ready_event.set()
            self.run_busy_loop(sock)
    except Exception as e:
        logger.exception(
            "Mooncake KVCacheSendingThread encountered exception. "
            "Thread: tp_rank=%d, pp_rank=%d, listening_path=%s. "
            "Error: %s",
            self.tp_rank, self.pp_rank, path, e,
        )
```

**端口计算公式**：
```
device_index = pp_rank * tp_size + tp_rank + pcp_rank * prefill_tp_size
handshake_port = side_channel_port + device_index
```

**示例**：TP=4, PP=2, PCP=2, base_port=50000

| PP | PCP | TP | device_index | Port |
|----|-----|----|-------------|------|
| 0  | 0   | 0  | 0*4+0+0*4=0 | 50000 |
| 0  | 0   | 1  | 0*4+1+0*4=1 | 50001 |
| 0  | 1   | 0  | 0*4+0+1*4=4 | 50004 |
| 1  | 0   | 0  | 1*4+0+0*4=4 | 50004 |

注意：代码断言 `pp_size > 1 and pcp_size > 1` 不能同时成立，所以不会有冲突。

### 4.4 主循环 run_busy_loop

```python
def run_busy_loop(self, sock: zmq.Socket):
    """主循环：处理来自 Decode 节点的消息"""
    encoder = msgspec.msgpack.Encoder()
    encoded_data = encoder.encode(self.metadata)  # 预编码元数据，提高性能
    size_in_bytes = len(encoded_data)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes))

    decoder = msgspec.msgpack.Decoder(type=tuple)
    while True:
        try:
            frames = sock.recv_multipart()
            if len(frames) < 2:
                logger.error("Invalid message format. Expected at least 2 frames.")
                continue

            identity = frames[0]  # ROUTER socket 的客户端标识
            payload = [f for f in frames[1:] if f != b""]
            if len(payload) != 1:
                logger.error("Invalid message format. Expected exactly 1 payload frame.")
                continue

            msg = decoder.decode(payload[0])
            if msg[0] == GET_META_MSG:
                # 返回本节点的元数据
                sock.send_multipart((identity, b"", encoded_data))
            elif msg[0] == DONE_RECVING_MSG:
                # Decode 节点通知接收完毕
                logger.debug("Got DONE_RECVING_MSG for request %s", msg[1])
                request_id = msg[1]
                remote_port_send_num = msg[2]
                if remote_port_send_num:
                    # 多端口场景：统计所有端口都完成后才释放
                    if request_id not in self.port_send_num:
                        self.port_send_num[request_id] = 0
                    self.port_send_num[request_id] += 1
                    device_index = self.pp_rank * self.tp_size + self.tp_rank + self.pcp_rank * self.prefill_tp_size
                    handshake_port = self.side_channel_port + device_index
                    if self.port_send_num[request_id] >= remote_port_send_num[handshake_port]["num"]:
                        self.task_tracker.update_done_task_count(request_id)
                        del self.port_send_num[request_id]
                else:
                    self.task_tracker.update_done_task_count(request_id)
                # 发送 ACK 确认
                while True:
                    try:
                        sock.send_multipart((identity, b"", b"ACK"), flags=zmq.NOBLOCK)
                        break
                    except zmq.Again:
                        logger.debug("Socket not ready, retrying to send ACK for request %s", msg[1])
                        time.sleep(0.01)
            else:
                logger.error("Unexpected message type: %s", msg[0] if msg else "empty")
        except Exception as e:
            logger.error("Error in connection listener: %s", e)
```

**关键点**：
- 使用 `zmq.ROUTER` socket：支持异步多客户端，通过 `identity` 帧路由回复
- `GET_META_MSG` 是高频调用：每个新请求都会触发，所以 metadata 预编码
- `DONE_RECVING_MSG` 是完成信号：触发资源释放

---

## 5. KVCacheRecvingThread：接收端后台线程

### 5.1 功能

运行在 **Decode 节点**（kv_consumer）上的后台线程，负责：
1. 从任务队列取出拉取请求
2. 通过 ZMQ 向 Prefill 节点请求元数据
3. 调用 Mooncake `TransferEngine` 批量传输 KV cache
4. 传输完成后做格式重组（如果需要）
5. 发送 `DONE_RECVING_MSG` 通知 Prefill 释放资源

### 5.2 初始化

```python
class KVCacheRecvingThread(threading.Thread):
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        _prefill_pp_size: int,
        engine: TransferEngine,
        local_engine_id: str,
        local_handshake_port: int,
        side_channel_port: int,
        local_kv_caches_base_addr: list[list[int]],
        block_len_per_addr: list[list[int]],
        is_hma_required=False,
        ready_event: threading.Event | None = None,
        vllm_config: VllmConfig | None = None,
        kv_caches: dict[str, Any] | None = None,
        prefill_pp_layer_partition: str | None = None,
        kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] | None = None,
        block_size_scale: list[list[int]] | None = None,
    ):
        super().__init__(daemon=True, name="KVCacheRecvingThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._prefill_pp_size = _prefill_pp_size
        self.local_engine_id = local_engine_id
        self.local_handshake_port = local_handshake_port
        self.side_channel_port = side_channel_port
        self.engine = engine
        self.ready_event = ready_event or threading.Event()
        self.kv_caches = kv_caches or {}
        
        # 使用 SizedDict 缓存远程节点元数据，避免重复请求
        self.kv_caches_base_addr: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.kv_caches_base_addr[local_engine_id][local_handshake_port] = local_kv_caches_base_addr
        self.block_len_per_addr = block_len_per_addr
        self.kv_group2layeridx = kv_group2layeridx or {}
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.remote_block_size_scale: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_kv_group2layeridx: dict[str, dict[int, dict[int, tuple[dict[str, Any], list[int]]]]] = SizedDict()

        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=32)  # 线程池处理传输任务
        self.task_tracker = KVCacheTaskTracker()
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[str, deque[zmq.Socket]] = defaultdict(deque)
        self.remote_poller = zmq.Poller()
        self.timeout = 1.0

        assert vllm_config is not None
        self.vllm_config: VllmConfig = vllm_config
        self.model_config = self.vllm_config.model_config
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )
        self.use_mla = self.model_config.is_deepseek_mla
        self.is_hma_required = is_hma_required
        self.block_size = self.vllm_config.cache_config.block_size
        # ... 更多初始化
```

### 5.3 `_transfer_kv_cache_all_groups` 使用的字段详解

`_transfer_kv_cache_all_groups` 是 `KVCacheRecvingThread` 中**实际执行 KV 传输**的核心方法。它从 `req_meta` 获取请求信息，然后访问 `self` 的多个字段来构建传输参数。以下是该方法使用的所有字段及其含义、作用与示例。

#### 字段总览表

| 字段名 | 类型 | 含义 | 在 `_transfer_kv_cache_all_groups` 中的使用位置 |
|--------|------|------|------------------------------------------|
| `self.kv_caches_base_addr` | `dict[str, dict[int, list[list[int]]]]` | 各引擎各端口的 KV cache 内存基地址 | 行 636-637：获取本地和远程层基地址 |
| `self.remote_te_port` | `dict[str, dict[int, int]]` | 远程引擎的 TransferEngine RPC 端口 | 行 638：构建 `session_id` |
| `self.remote_block_size_scale` | `dict[str, dict[int, list[list[int]]]]` | 远程引擎的 block 缩放比例 | 行 639：获取远程 block 缩放 |
| `self.block_size` | `int` | 逻辑 block 大小（token 数） | 行 678：计算远程 kernel block 大小 |
| `self.block_size_scale` | `list[list[int]]` | 本地 block 缩放比例 | 行 672：获取本地 block 缩放 |
| `self.pp_layer_indices` | `dict[int, tuple[int, int]]` | PP rank 到层索引范围的映射 | 行 652：筛选当前 PP rank 负责的层 |
| `self.vllm_config` | `VllmConfig` | vLLM 全局配置 | 行 653：检查是否启用 speculative decoding |
| `self.num_draft_layers` | `int` | Speculative decoding 的 draft 层数 | 行 654：调整 end_layer_index |
| `self.kv_group2layeridx` | `dict[int, tuple[dict, list[int]]]` | KV group 到层索引的映射 | 行 659：获取 group 的 spec 和层索引 |
| `self.engine` | `TransferEngine` | Mooncake 传输引擎实例 | 行 777：执行 `batch_transfer_sync_read` |
| `self.block_len_per_addr` | `list[list[int]]` | 每层每个 cache 的 block 字节长度 | 行 745：计算传输长度 |
| `self.local_engine_id` | `str` | 本地引擎 ID | 行 637：索引本地基地址 |
| `self.local_handshake_port` | `int` | 本地 handshake 端口 | 行 637：索引本地基地址 |
| `self._prefill_pp_size` | `int` | Prefill 节点的 PP 大小 | 行 653：判断是否为最后一个 PP rank |
| `self.num_speculative_tokens` | `int` | 投机采样的 token 数 | 行 702：Mamba 组计算 transfer block 索引 |
| `self.is_hma_required` | `bool` | 是否需要 HMA 重组 | 行 685：判断传输后是否执行 HMA 重组 |

---

#### 1. `self.kv_caches_base_addr`

**类型**：`dict[str, dict[int, list[list[int]]]]`

**含义**：一个嵌套字典，缓存了本地和远程节点的 KV cache 内存基地址。
- 第一层 key：`engine_id`（字符串，标识一个 Prefill/Decode 引擎实例）
- 第二层 key：`handshake_port`（整数，标识该引擎上的具体端口/设备）
- 值：`list[list[int]]`，即 `[[layer0_K_addr, layer0_V_addr], [layer1_K_addr, layer1_V_addr], ...]`

**作用**：在传输时，通过 `remote_engine_id` 和 `remote_handshake_port` 索引到远程节点的基地址，结合 `local_engine_id` 和 `local_handshake_port` 索引到本地基地址，从而计算出每个 layer 每个 cache（K/V）的具体 `src` 和 `dst` 地址。

**示例**：
```python
# 假设本地是 decode 节点，engine_id="decode_0", handshake_port=50001
# 远程是 prefill 节点，engine_id="prefill_0", handshake_port=50002
self.kv_caches_base_addr = {
    "decode_0": {
        50001: [
            [0x7f0000000000, 0x7f0000100000],  # layer 0: K_addr, V_addr
            [0x7f0000200000, 0x7f0000300000],  # layer 1
            ...
        ]
    },
    "prefill_0": {
        50002: [
            [0x7f1000000000, 0x7f1000100000],  # layer 0: K_addr, V_addr
            [0x7f1000200000, 0x7f1000300000],  # layer 1
            ...
        ]
    }
}
```

在代码中（行 636-637）：
```python
remote_kv_caches_base_addrs = self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
local_kv_caches_base_addrs = self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port]
```

---

#### 2. `self.remote_te_port`

**类型**：`dict[str, dict[int, int]]`

**含义**：缓存远程节点的 Mooncake TransferEngine RPC 端口。
- 第一层 key：`engine_id`
- 第二层 key：`handshake_port`
- 值：该远程节点上 TransferEngine 监听的 RPC 端口号

**作用**：构建 Mooncake 传输的 `session_id`。`session_id` 是 `batch_transfer_sync_read` 的第一个参数，格式为 `"host:port"`，用于标识远程传输端点。

**示例**：
```python
self.remote_te_port = {
    "prefill_0": {
        50002: 50010  # prefill 节点在 handshake_port=50002 上的 TE RPC 端口是 50010
    }
}

# 代码中（行 638-640）：
remote_transfer_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
session_id = f"{remote_host}:{remote_transfer_port}"  # 例如 "192.168.1.10:50010"
```

---

#### 3. `self.remote_block_size_scale`

**类型**：`dict[str, dict[int, list[list[int]]]]`

**含义**：缓存远程节点的 block 缩放比例。与 `kv_caches_base_addr` 结构相同，但存储的是缩放因子而非地址。

**作用**：逻辑 block ID 需要转换为物理 tensor block ID。远程节点和本地节点可能使用不同的缩放比例（例如远程 prefill 的 tensor block 数是本地 decode 的 2 倍），所以需要分别获取本地和远程的 `block_size_scale` 进行扩展。

**示例**：
```python
# 假设 block_size=16，但 tensor 物理 shape[0]=32，则 scale=2
self.remote_block_size_scale = {
    "prefill_0": {
        50002: [[2, 2], [2, 2], ...]  # 每层 [K_scale, V_scale]
    }
}
self.block_size_scale = [[1, 1], [1, 1], ...]  # 本地 decode 的 scale

# 代码中（行 672-673）：
local_scale = self.block_size_scale[layer_indices[0]][0]   # 例如 1
remote_scale = remote_block_size_scale[layer_indices[0]][0] # 例如 2

# 逻辑 block_id=10 扩展为物理 block_id：
# 本地: [10] (scale=1)
# 远程: [20, 21] (scale=2，即 10*2+0, 10*2+1)
```

---

#### 4. `self.block_size`

**类型**：`int`

**含义**：vLLM cache config 中定义的逻辑 block 大小，即一个 block 能容纳多少个 token。

**作用**：计算远程 kernel block 大小，用于处理 prefix cache 场景下跳过已计算 token 的逻辑。

**示例**：
```python
# 假设 block_size=16, remote_scale=2
remote_kernel_block_size = self.block_size // remote_scale  # 16 // 2 = 8

# 如果 num_computed_tokens=8，表示前 8 个 token 已经在本地 prefix cache 中命中
remote_start_idx = num_computed_tokens // remote_kernel_block_size  # 8 // 8 = 1
# 则跳过第 0 个 kernel block，从第 1 个开始传输
```

---

#### 5. `self.block_size_scale`

**类型**：`list[list[int]]`

**含义**：本地节点的 block 缩放比例。与 `remote_block_size_scale` 对应，但只存储本地数据，不需要按 engine_id 索引。

**作用**：将本地逻辑 block ID 扩展为物理 tensor block ID。

**示例**：见 `self.remote_block_size_scale` 的示例。

---

#### 6. `self.pp_layer_indices`

**类型**：`dict[int, tuple[int, int]]`

**含义**：Prefill 节点 PP rank 到层索引范围的映射。key 是 PP rank，value 是 `(first_layer_index, end_layer_index)`。

**作用**：在 PP 并行场景下，每个 Prefill 节点只负责模型的一部分层。Decode 节点需要根据 `group_pull.prefill_pp_rank` 筛选出该节点实际负责的层，避免请求不存在的层。

**示例**：
```python
# 假设 32 层模型，PP=2，则每个 rank 负责 16 层
self.pp_layer_indices = {
    0: (0, 16),   # PP rank 0 负责 layer 0-15
    1: (16, 32)   # PP rank 1 负责 layer 16-31
}

# 代码中（行 651-655）：
def pp_layer_indices(layer_indices, prefill_pp_rank):
    first, end = self.pp_layer_indices[prefill_pp_rank]  # 例如 (0, 16)
    # 如果启用了 speculative decoding 且是最后一个 PP rank，end 需要加上 draft 层数
    if self.vllm_config.speculative_config is not None and prefill_pp_rank == self._prefill_pp_size - 1:
        end += self.num_draft_layers
    return [idx for idx in layer_indices if first <= idx < end]
```

---

#### 7. `self.vllm_config`

**类型**：`VllmConfig`

**含义**：vLLM 的全局配置对象，包含 model_config、cache_config、parallel_config、speculative_config 等。

**作用**：在 `_transfer_kv_cache_all_groups` 中主要用于检查是否启用了 speculative decoding（`self.vllm_config.speculative_config is not None`）。

**示例**：
```python
# 行 653：检查是否启用 speculative decoding
if self.vllm_config.speculative_config is not None and prefill_pp_rank == self._prefill_pp_size - 1:
    end_layer_index += self.num_draft_layers
```

---

#### 8. `self.num_draft_layers`

**类型**：`int`

**含义**：Speculative decoding 中 draft 模型的层数。对于 MTP（Multi-Token Prediction）方法，所有 draft 层共享同一个 KV cache，所以值为 1；对于其他方法，值为 draft 模型的实际隐藏层数。

MTP（Multi-token Prediction，多词元预测）是大语言模型（LLM）领域一项前沿的训练和推理架构技术。
简单来说，传统的语言模型是“走一步看一步”（每次只预测下一个词），而采用 MTP 方法的模型被训练成“走一步看多步”（一次性预测未来连续的多个词）。

**作用**：在最后一个 PP rank 的 `end_layer_index` 上加上 draft 层数，确保 KV 传输覆盖到 draft 层。

**示例**：
```python
# 假设主模型 32 层，draft 模型 2 层，PP=2
# PP rank 1 原本负责 layer 16-32
# 加上 num_draft_layers=2 后，负责 layer 16-34
self.pp_layer_indices = {1: (16, 34)}
```

---

#### 9. `self.kv_group2layeridx`

**类型**：`dict[int, tuple[dict[str, Any], list[int]]]`

**含义**：KV cache 分组到层索引的映射。key 是 group_id，value 是 `(group_spec, layer_indices)`。
- `group_spec`：包含 `kv_cache_spec_type`（如 `"FullAttentionSpec"` 或 `"MambaSpec"`）、`layer_names` 等
- `layer_indices`：该 group 包含的物理层索引列表

**作用**：根据 `group_pull.group_id` 获取该组的 spec 和层索引，然后判断是 Attention 组还是 Mamba 组，走不同的传输逻辑。

**示例**：
```python
self.kv_group2layeridx = {
    0: (
        {"kv_cache_spec_type": "FullAttentionSpec", "layer_names": ["layers.0", ...]},
        [0, 1, 2, ..., 15]
    ),
    1: (
        {"kv_cache_spec_type": "FullAttentionSpec", "layer_names": ["layers.16", ...]},
        [16, 17, ..., 31]
    )
}

# 代码中（行 659-665）：
group_spec, layer_indices = self.kv_group2layeridx[group_idx]
is_mamba_group = group_spec["kv_cache_spec_type"] == "MambaSpec"
```

---

#### 10. `self.engine`

**类型**：`TransferEngine`

**含义**：Mooncake 传输引擎实例，封装了 RDMA/高带宽网络的底层传输能力。

**作用**：调用 `batch_transfer_sync_read` 执行实际的批量内存传输。

**示例**：
```python
# 行 777：执行批量传输
ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
# session_id: "192.168.1.10:50010"
# src_list: [本地 decode 目标地址1, 地址2, ...]
# dst_list: [远程 prefill 源地址1, 地址2, ...]
# length_list: [传输长度1, 长度2, ...]
```

---

#### 11. `self.block_len_per_addr`

**类型**：`list[list[int]]`

**含义**：每层每个 cache（K/V）的单个 block 字节长度。

**作用**：计算传输长度。`block_len` 是一个完整 block 的字节数，当 TP 分片时，需要除以 `tp_num_need_pulls` 得到每个分片的长度。

**示例**：
```python
# 假设 block_size=16, num_kv_heads=8, head_dim=128, dtype=fp16 (2 bytes)
# block_len = 16 * 8 * 128 * 2 = 32768 bytes
self.block_len_per_addr = [
    [32768, 32768],  # layer 0: [K_block_len, V_block_len]
    [32768, 32768],  # layer 1
    ...
]

# 代码中（行 745-746）：
block_len = self.block_len_per_addr[layer_idx][cache_idx]  # 32768
inner_block_len = block_len // tp_num_need_pulls  # 如果 tp_num_need_pulls=2, 则为 16384
```

---

#### 12. `self.local_engine_id` 和 `self.local_handshake_port`

**类型**：`str` 和 `int`

**含义**：本地 Decode 节点的引擎标识和 handshake 端口。

**作用**：在 `kv_caches_base_addr` 中索引本地基地址。

**示例**：
```python
# 行 637：
local_kv_caches_base_addrs = self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port]
# 例如：self.local_engine_id="decode_0", self.local_handshake_port=50001
```

---

#### 13. `self._prefill_pp_size`

**类型**：`int`

**含义**：Prefill 节点的 Pipeline Parallel 大小。

**作用**：判断当前 group 的 `prefill_pp_rank` 是否是最后一个 PP rank。如果是最后一个 rank 且启用了 speculative decoding，则需要调整 `end_layer_index` 以包含 draft 层。

**示例**：
```python
# 行 653：
if self.vllm_config.speculative_config is not None and prefill_pp_rank == self._prefill_pp_size - 1:
    end_layer_index += self.num_draft_layers
```

---

#### 14. `self.num_speculative_tokens`

**类型**：`int`

**含义**：Speculative decoding 中使用的投机采样 token 数量。

**作用**：在 Mamba 组的特殊处理中，计算需要从远程拉取的特定 block 索引。

**示例**：
```python
# 行 702：Mamba 组计算 transfer block 索引
transfer_block_idx = len(remote_group_block_ids) - self.num_speculative_tokens - 1
# 假设远程有 10 个 block，num_speculative_tokens=3
# 则 transfer_block_idx = 10 - 3 - 1 = 6
# 只拉取第 6 个 block
```

---

#### 15. `self.is_hma_required`

**类型**：`bool`

**含义**：是否启用了 HMA（Hybrid Memory Attention，混合内存注意力）模式。

**作用**：传输完成后，如果需要格式重组，判断使用 HMA 专用的重组逻辑还是通用重组逻辑。

**示例**：
```python
# 行 685-692：
if self.is_hma_required:
    for group_idx, grouped_local_block_ids, num_group_pulls, layer_indices in gqa_reformat_groups:
        group_kv_caches = self._get_group_kv_caches(group_idx, layer_indices)
        self.reformat_kv_cache_hybrid_linear_torch(grouped_local_block_ids, num_group_pulls, group_kv_caches)
```

---

### 5.4 核心方法：`_transfer_kv_cache_all_groups` 完整流程

```python
def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]):
    """Handle a KV cache transfer request."""
    remote_request_id = req_meta["remote_request_id"]
    local_block_ids: BlockIds = req_meta["local_block_ids"]
    remote_block_ids: BlockIds = req_meta["remote_block_ids"]
    group_pulls: list[GroupPull] = req_meta["group_pulls"]
    remote_engine_id = req_meta["remote_engine_id"]
    remote_host = req_meta["remote_host"]
    remote_handshake_port = req_meta["remote_handshake_port"]

    # 1. 检查是否需要传输（本地全命中则跳过）
    num_local_blocks = sum(len(group_block_ids) for group_block_ids in local_block_ids)
    if num_local_blocks == 0:
        return

    # 2. 获取远程元数据（缓存或请求）
    if (
        remote_engine_id not in self.kv_caches_base_addr
        or remote_handshake_port not in self.kv_caches_base_addr[remote_engine_id]
    ):
        self._get_remote_metadata(remote_host, remote_handshake_port)
    
    remote_kv_caches_base_addrs = self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
    local_kv_caches_base_addrs = self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port]
    remote_transfer_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
    remote_block_size_scale = self.remote_block_size_scale[remote_engine_id][remote_handshake_port]
    session_id = f"{remote_host}:{remote_transfer_port}"

    # 3. 构建批量传输参数
    req_start_time = time.perf_counter()
    src_list: list[int] = []   # 本地（decode）目标地址
    dst_list: list[int] = []   # 远程（prefill）源地址
    length_list: list[int] = [] # 传输长度
    attention_group_reformat_block_ids: list[tuple[tuple[int, list[list[int]], int, list[int]], bool]] = []

    def expand_block_ids(block_ids, scale):
        """将逻辑 block ID 扩展为物理 tensor block ID"""
        return [bid * scale + offset for bid in block_ids for offset in range(scale)]

    def pp_layer_indices(layer_indices: list[int], prefill_pp_rank: int) -> list[int]:
        """根据 PP rank 筛选该节点负责的层"""
        first_layer_index, end_layer_index = self.pp_layer_indices[prefill_pp_rank]
        if self.vllm_config.speculative_config is not None and prefill_pp_rank == self._prefill_pp_size - 1:
            end_layer_index += self.num_draft_layers
        return [layer_idx for layer_idx in layer_indices if first_layer_index <= layer_idx < end_layer_index]

    # 4. 遍历每个 group，计算传输地址
    for group_pull in group_pulls:
        group_idx = group_pull.group_id
        group_spec, layer_indices = self.kv_group2layeridx[group_idx]
        layer_indices = pp_layer_indices(layer_indices, group_pull.prefill_pp_rank)
        if not layer_indices:
            continue
        
        tp_num_need_pulls = group_pull.num_group_pulls
        inner_offset = group_pull.remote_tp_offset
        is_mamba_group = group_spec["kv_cache_spec_type"] == "MambaSpec"
        local_group_block_ids = local_block_ids[group_idx]
        remote_group_block_ids = remote_block_ids[group_idx]
        if not local_group_block_ids:
            continue

        if not is_mamba_group:
            # 标准 Attention 层的处理
            is_group_transfer_end = group_pull.is_group_transfer_end
            local_scale = self.block_size_scale[layer_indices[0]][0]
            remote_scale = remote_block_size_scale[layer_indices[0]][0]
            kernel_local_block_ids = expand_block_ids(local_group_block_ids, local_scale)
            kernel_remote_block_ids = expand_block_ids(remote_group_block_ids, remote_scale)
            
            # 处理 prefix cache：跳过已计算的 token
            num_computed_tokens = req_meta.get("num_computed_tokens", 0)
            remote_kernel_block_size = self.block_size // remote_scale
            remote_start_idx = num_computed_tokens // remote_kernel_block_size
            kernel_remote_block_ids = kernel_remote_block_ids[remote_start_idx:]
            num_kernel_blocks = min(len(kernel_remote_block_ids), len(kernel_local_block_ids))
            kernel_remote_block_ids = kernel_remote_block_ids[:num_kernel_blocks]
            kernel_local_block_ids = kernel_local_block_ids[:num_kernel_blocks]

            # 根据 TP 分片策略分组
            if tp_num_need_pulls == 1:
                grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
                    kernel_remote_block_ids, kernel_local_block_ids
                )
            else:
                grouped_remote_block_ids = [[block_id] for block_id in kernel_remote_block_ids]
                grouped_local_block_ids = [[block_id] for block_id in kernel_local_block_ids]
            
            attention_group_reformat_block_ids.append(
                ((group_idx, grouped_local_block_ids, tp_num_need_pulls, layer_indices), is_group_transfer_end)
            )
        else:
            # Mamba 层的特殊处理
            if len(local_group_block_ids) != len(remote_group_block_ids):
                raise RuntimeError("For MambaSpec num block should equal on P node and D node.")
            transfer_block_idx = len(remote_group_block_ids) - self.num_speculative_tokens - 1
            grouped_remote_block_ids = [[remote_group_block_ids[transfer_block_idx]]]
            grouped_local_block_ids = [[local_group_block_ids[0]]]

        # 5. 计算每个 layer 的 src/dst/length
        if is_mamba_group:
            for layer_idx in layer_indices:
                start_meta_idx = len(src_list)
                self._append_mamba_transfer_meta(
                    src_list, dst_list, length_list,
                    group_spec=group_spec,
                    src_layer_base_addr=local_kv_caches_base_addrs[layer_idx],
                    dst_layer_base_addr=remote_kv_caches_base_addrs[layer_idx],
                    block_len=self.block_len_per_addr[layer_idx],
                    remote_block_id=grouped_remote_block_ids[0][0],
                    local_block_id=grouped_local_block_ids[0][0],
                    tp_num_need_pulls=tp_num_need_pulls,
                    remote_tp_offset=inner_offset,
                )
            continue

        for layer_idx in layer_indices:
            # 对每一层来说，K 循环一次， V循环一次
            for cache_idx in range(len(local_kv_caches_base_addrs[layer_idx])):
                # local_kv_caches_base_addrs 格式样例如下，所以这个循环的意思就是K V 分别传输：
                # [0x7f1000000000, 0x7f1000100000],  # layer 0: K_addr, V_addr
                # [0x7f1000200000, 0x7f1000300000],  # layer 1
                src_layer_base_addr = local_kv_caches_base_addrs[layer_idx][cache_idx]
                dst_layer_base_addr = remote_kv_caches_base_addrs[layer_idx][cache_idx]
                # block_len_per_addr 各层 K V cache的长度
                block_len = self.block_len_per_addr[layer_idx][cache_idx]
                inner_block_len = block_len // tp_num_need_pulls
                for remote_block_id, local_block_id in zip(grouped_remote_block_ids, grouped_local_block_ids):
                    # 计算具体内存地址
                    src = src_layer_base_addr + local_block_id[0] * block_len + inner_offset * inner_block_len
                    dst = dst_layer_base_addr + remote_block_id[0] * inner_block_len
                    length = inner_block_len * len(local_block_id)
                    # decoder 本地的 K V cache的物理地址
                    src_list.append(src)
                    # prefiller 本地的 K V cache的物理地址
                    dst_list.append(dst)
                    length_list.append(length)

    # 6. 调用 Mooncake TransferEngine 执行批量传输
    if not src_list:
        return

    logger.debug(
        "Mooncake transfer request=%s session id=%s src=%s dst=%s length=%s",
        remote_request_id, session_id, src_list, dst_list, length_list
    )
    # 实际执行传输动作。可以看到直接使用了物理地址
    ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
    if ret < 0:
        logger.error("Mooncake transfer failed for request. remote_request_id=%s, ret=%d", req_meta["remote_request_id"], ret)
        raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

    req_end_time = time.perf_counter()
    req_transfer_elapsed = (req_end_time - req_start_time) * 1000
    logger.info(
        "KV cache transfer for request %s took %.2f ms. local_ip %s local_device_id %s remote_session_id %s",
        remote_request_id, req_transfer_elapsed, get_ip(), self.tp_rank, session_id
    )

    # 7. 格式重组
    ready_attention_group_reformat_block_ids = []
    for reformat_group, is_group_transfer_end in attention_group_reformat_block_ids:
        if is_group_transfer_end:
            ready_attention_group_reformat_block_ids.append(reformat_group)
    if not ready_attention_group_reformat_block_ids:
        return

    gqa_reformat_groups = [
        (group_idx, grouped_local_block_ids, num_group_pulls, layer_indices)
        for (group_idx, grouped_local_block_ids, num_group_pulls, layer_indices) in ready_attention_group_reformat_block_ids
        if num_group_pulls > 1
    ]

    if self.is_hma_required:
        # HMA（混合注意力）场景的重组
        for group_idx, grouped_local_block_ids, num_group_pulls, layer_indices in gqa_reformat_groups:
            group_kv_caches = self._get_group_kv_caches(group_idx, layer_indices)
            if not group_kv_caches:
                continue
            self.reformat_kv_cache_hybrid_linear_torch(grouped_local_block_ids, num_group_pulls, group_kv_caches)
        return

    # 非 HMA 场景的统一重组
    uniform_num_pulls = {num_group_pulls for _, _, num_group_pulls, _ in ready_attention_group_reformat_block_ids}
    if len(uniform_num_pulls) != 1:
        raise RuntimeError(f"Non-hybrid Mooncake KV reformat expects uniform group pulls, but got {uniform_num_pulls}.")

    num_group_pulls = next(iter(uniform_num_pulls))
    need_cat_cache = num_group_pulls > 1
    need_nz_cache = get_ascend_config().enable_kv_nz
    if not (need_cat_cache or need_nz_cache):
        return

    use_fused_op = ascend_envs.VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK
    for group_idx, reformat_block_ids, _, layer_indices in ready_attention_group_reformat_block_ids:
        group_kv_caches = self._get_group_kv_caches(group_idx, layer_indices)
        if not group_kv_caches:
            continue
        if use_fused_op and enable_custom_op():
            if need_cat_cache:
                self.reformat_kv_cache_with_fused_op(reformat_block_ids, num_group_pulls, group_kv_caches)
            if need_nz_cache:
                self.reformat_kv_cache(reformat_block_ids, num_group_pulls, False, need_nz_cache, group_kv_caches)
        else:
            self.reformat_kv_cache(reformat_block_ids, num_group_pulls, need_cat_cache, need_nz_cache, group_kv_caches)
```

#### 标准 Attention 层处理详解

上述代码中 `if not is_mamba_group:` 分支处理 **标准 Attention 层** 的 KV cache 传输准备，核心任务是：**将逻辑 block ID 转换为物理 tensor block ID，处理 prefix cache 跳过，然后按 TP 策略分组，最后记录需要格式重组的信息**。

##### 1. 获取 block 缩放比例并扩展

```python
local_scale = self.block_size_scale[layer_indices[0]][0]
remote_scale = remote_block_size_scale[layer_indices[0]][0]
kernel_local_block_ids = expand_block_ids(local_group_block_ids, local_scale)
kernel_remote_block_ids = expand_block_ids(remote_group_block_ids, remote_scale)
```

**逻辑 block → 物理 tensor block 的映射**。

vLLM 的 `block_size` 是逻辑概念（如 16 tokens），但底层 tensor 的 shape[0] 可能更大。如果 `tensor_num_blocks = 2048` 而 `num_blocks = 1024`，则 `scale = 2`。

`expand_block_ids(10, 2)` → `[20, 21]`，即逻辑 block 10 对应物理 block 20 和 21。

**示例**：
- 本地 decode：`local_scale=1`，逻辑 block `[5,6]` → `[5,6]`
- 远程 prefill：`remote_scale=2`，逻辑 block `[5,6]` → `[10,11,12,13]`

##### 2. 处理 prefix cache：跳过已计算的 token

```python
num_computed_tokens = req_meta.get("num_computed_tokens", 0)
remote_kernel_block_size = self.block_size // remote_scale
remote_start_idx = num_computed_tokens // remote_kernel_block_size
kernel_remote_block_ids = kernel_remote_block_ids[remote_start_idx:]
```

**跳过本地 prefix cache 已命中的部分，只传输需要的 token**。

假设 `block_size=16`，`remote_scale=2`，则 `remote_kernel_block_size=8`。如果 `num_computed_tokens=8`，则 `remote_start_idx=1`，跳过第 0 个 kernel block。

**为什么只裁剪 remote？** 因为本地 decode 的 block 已经分配好了，远程 prefill 的 block 可能包含更多已计算的 token。

##### 3. 对齐本地和远程的 block 数量

```python
num_kernel_blocks = min(len(kernel_remote_block_ids), len(kernel_local_block_ids))
kernel_remote_block_ids = kernel_remote_block_ids[:num_kernel_blocks]
kernel_local_block_ids = kernel_local_block_ids[:num_kernel_blocks]
```

**取最小值，确保两边 block 数量一致**，防止越界。

##### 4. 根据 TP 分片策略分组

```python
if tp_num_need_pulls == 1:
    # TP 不需要分片拉取：合并连续 block 优化传输
    grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
        kernel_remote_block_ids, kernel_local_block_ids
    )
else:
    # TP 需要分片拉取：每个 block 单独一组
    grouped_remote_block_ids = [[block_id] for block_id in kernel_remote_block_ids]
    grouped_local_block_ids = [[block_id] for block_id in kernel_local_block_ids]
```

**`tp_num_need_pulls`** 表示 decode 需要从多少个 prefill TP rank 拉取 KV 分片。

- **= 1**：Prefill TP = Decode TP，一一对应，可以合并连续 block 做批量传输（减少 RDMA 请求数）
- **> 1**：Prefill TP > Decode TP，每个 block 独立处理，后续按 TP offset 分别拉取

**`group_concurrent_contiguous`** 的作用：将连续的 block ID 合并为区间，例如 `[10,11,12,15,16]` → `[[10,11,12], [15,16]]`，这样可以用一次 RDMA 传输连续内存。
**group_concurrent_contiguous 直观的例子**
假设我们有以下参数，且步长（stride）和块长度（len）都为默认值 1：src = [10, 11, 12,  15, 16] (源地址块)  dst = [20, 21, 22,  23, 25] (目标地址块)
**步骤演算：
**求差值 (np.diff)：**
    src 的差值: [1, 1, 3, 1]
    dst 的差值: [1, 1, 1, 2]
**判断连续性 (== block_len)：**
    src 连续性: [True, True, False, True]
    dst 连续性: [True, True, True, False]
**同时连续 (&)：**
    结果: [True, True, False, False] (只有前两个间隔是双端连续的)
**找断点 (~ 且 + 1)：**
    反转结果 (~): [False, False, True, True]
    断点索引在第 2 位和第 3 位，加上 1 后，brk = [3, 4]。这意味着要在原数组索引为 3 和 4 的地方切一刀。
**执行分割 (np.split)：**
    src 被切分成: [10, 11, 12], [15], [16]
    dst 被切分成: [20, 21, 22], [23], [25]
**最终结果：**
    返回的 src_groups 是 [[10, 11, 12], [15], [16]]
    返回的 dst_groups 是 [[20, 21, 22], [23], [25]]

##### 5. 记录格式重组信息

```python
attention_group_reformat_block_ids.append(
    ((group_idx, grouped_local_block_ids, tp_num_need_pulls, layer_indices), is_group_transfer_end)
)
```

**记录该 group 的传输信息，供后续格式重组使用**。

`is_group_transfer_end` 标记该 group 是否传输完成。如果完成，后续会调用 `reformat_kv_cache_*` 方法进行内存布局重组（例如 TP 分片合并、NZ 格式转换等）。

##### 数据结构详解

`attention_group_reformat_block_ids` 是一个列表，每个元素是一个 **元组**：

```
((group_idx, grouped_local_block_ids, tp_num_need_pulls, layer_indices), is_group_transfer_end)
     └─ 重组信息元组 ─────────────────────────────────────────────┘  └─ 是否传输完成标记
```

| 字段 | 类型 | 含义 |
|------|------|------|
| `group_idx` | `int` | KV cache 组 ID（如 0, 1, ...） |
| `grouped_local_block_ids` | `list[list[int]]` | 分组后的本地 block ID（如 `[[10,11,12], [15,16]]`） |
| `tp_num_need_pulls` | `int` | 需要从多少个 prefill TP rank 拉取 |
| `layer_indices` | `list[int]` | 该 group 包含的层索引 |
| `is_group_transfer_end` | `bool` | 该 group 是否传输完成 |

##### 为什么需要记录？

因为 **RDMA 传输的内存布局可能与本地 decode 期望的布局不一致**，传输完成后需要重组：

**场景 1：TP 分片合并（`tp_num_need_pulls > 1`）**

当 Prefill TP > Decode TP 时，Decode 从多个 prefill rank 分别拉取 KV 分片。传输后，这些分片是分散的，需要合并：

```
传输后布局（分散）:  [block0_tp0, block0_tp1, block1_tp0, block1_tp1, ...]
目标布局（合并）:    [block0_all_tp, block1_all_tp, ...]
```

**场景 2：HMA（混合注意力）重组**

`is_hma_required=True` 时，需要调用 `reformat_kv_cache_hybrid_linear_torch` 进行维度转置：

```python
# 传输后布局：[block, split, token, head_per_split, dim]
# 目标布局：  [block, token, split, head_per_split, dim]
```

**场景 3：NZ 格式转换**

Ascend NPU 可能需要将 KV cache 转换为 NZ（非零）格式以优化计算。

##### 后续如何使用？

在 `_transfer_kv_cache_all_groups` 方法末尾：

```python
# 7. 格式重组
ready_attention_group_reformat_block_ids = []
for reformat_group, is_group_transfer_end in attention_group_reformat_block_ids:
    if is_group_transfer_end:
        ready_attention_group_reformat_block_ids.append(reformat_group)
```

**只有 `is_group_transfer_end=True` 的 group 才会进入重组**。这是因为：
- 一个请求可能分多次传输（如多轮迭代）
- 只有最后一次传输完成后，才做最终重组

然后：

```python
if self.is_hma_required:
    # HMA 场景：逐层转置
    for group_idx, grouped_local_block_ids, num_group_pulls, layer_indices in gqa_reformat_groups:
        self.reformat_kv_cache_hybrid_linear_torch(...)
else:
    # 通用场景：TP 合并 + NZ 转换
    self.reformat_kv_cache(...)
```

##### 总结

这段代码的本质是：**为"传输后处理"阶段收集元数据**。RDMA 只负责把字节从 A 搬到 B，但内存布局的对齐、TP 分片的合并、格式的转换，需要在传输完成后由 CPU/NPU 额外处理。`attention_group_reformat_block_ids` 就是连接"传输阶段"和"重组阶段"的桥梁。

##### 整体流程图

```
逻辑 block IDs ──► 物理 block IDs ──► 跳过 prefix cache ──► 对齐数量 ──► TP 分组 ──► 记录重组信息
     │                  │                  │                  │           │
     │              local_scale        remote_start_idx    min()    group_concurrent_contiguous
     │              remote_scale
```

这段代码的本质是：**为批量 RDMA 传输做地址预处理，同时收集后续格式重组所需的元数据**。

---

#### Mamba 层处理详解

与标准 Attention 层不同，Mamba 层（`is_mamba_group=True`）走另一条分支：

```python
# Mamba 层的特殊处理
if len(local_group_block_ids) != len(remote_group_block_ids):
    raise RuntimeError("For MambaSpec num block should equal on P node and D node.")
transfer_block_idx = len(remote_group_block_ids) - self.num_speculative_tokens - 1
grouped_remote_block_ids = [[remote_group_block_ids[transfer_block_idx]]]
grouped_local_block_ids = [[local_group_block_ids[0]]]
```

这段代码处理 **Mamba 层**（非标准 Attention 架构，如 State Space Model）的 KV cache 传输准备。

##### 1. Block 数量校验

```python
if len(local_group_block_ids) != len(remote_group_block_ids):
    raise RuntimeError("For MambaSpec num block should equal on P node and D node.")
```

**校验**：Mamba 架构要求 Prefill 节点和 Decode 节点的 block 数量必须相等。与 Attention 层不同，Mamba 的 state 不能部分传输，必须完整对齐。

##### 2. 计算需要传输的特定 block 索引

```python
transfer_block_idx = len(remote_group_block_ids) - self.num_speculative_tokens - 1
```

**计算需要传输的特定 block 索引**。Mamba 不像 Attention 那样逐层存储 KV cache，而是维护一个压缩的 state。这里只传输一个关键 block。

**示例**：
- 远程有 10 个 block，`num_speculative_tokens=3`
- `transfer_block_idx = 10 - 3 - 1 = 6`
- 只拉取第 6 个 block（保留最后 3 个 speculative token 的空间）

##### 3. 构造分组格式

```python
grouped_remote_block_ids = [[remote_group_block_ids[transfer_block_idx]]]
grouped_local_block_ids = [[local_group_block_ids[0]]]
```

**构造分组格式**。与 Attention 层的多个 block 分组不同，Mamba 只传输 **单个 block**：
- `grouped_remote_block_ids`：`[[block_6]]`（只取第 6 个远程 block）
- `grouped_local_block_ids`：`[[block_0]]`（映射到本地第 0 个 block）

##### 为什么 Mamba 特殊？

| 特性 | Attention (FullAttentionSpec) | Mamba (MambaSpec) |
|------|------------------------------|-------------------|
| KV cache 存储 | 每层独立的 K/V 张量 | 压缩的 state 向量 |
| 传输粒度 | 多个 block，逐层逐 cache | 单个 block，跨层共享 |
| Block 数量要求 | 本地和远程可以对齐裁剪 | 必须严格相等 |
| Speculative tokens | 通过 draft 层扩展 | 通过 state 偏移计算 |

##### 后续处理

这段代码构造的 `grouped_*_block_ids` 会进入下面的 Mamba 专用传输逻辑：

```python
if is_mamba_group:
    for layer_idx in layer_indices:
        self._append_mamba_transfer_meta(
            src_list, dst_list, length_list,
            group_spec=group_spec,
            src_layer_base_addr=local_kv_caches_base_addrs[layer_idx],
            dst_layer_base_addr=remote_kv_caches_base_addrs[layer_idx],
            block_len=self.block_len_per_addr[layer_idx],
            remote_block_id=grouped_remote_block_ids[0][0],   # 单个 block
            local_block_id=grouped_local_block_ids[0][0],      # 单个 block
            ...
        )
```

本质上是：**Mamba 的 state 传输是"单点式"的，只取一个关键 state block，而不是像 Attention 那样批量传输多个 KV block**。

---

### 5.5 地址计算详解

```python
# 核心地址计算逻辑
src = src_layer_base_addr + local_block_id[0] * block_len + inner_offset * inner_block_len
dst = dst_layer_base_addr + remote_block_id[0] * inner_block_len
length = inner_block_len * len(local_block_id)
```

| 变量 | 含义 | 示例 |
|------|------|------|
| `src_layer_base_addr` | 本地 decode 某层某 cache 的基地址 | `0x7f0000000000` |
| `local_block_id[0]` | 本地 block ID | `10` |
| `block_len` | 一个 block 的总字节长度 | `32768` |
| `inner_offset` | TP 分片偏移 | `0` (TP=1) |
| `inner_block_len` | 单个 TP 分片的 block 长度 | `32768` |
| `dst_layer_base_addr` | 远程 prefill 某层某 cache 的基地址 | `0x7f1000000000` |
| `remote_block_id[0]` | 远程 block ID | `10` |

**示例计算**：
```
src = 0x7f0000000000 + 10 * 32768 + 0 * 32768 = 0x7f0000800000
dst = 0x7f1000000000 + 10 * 32768 = 0x7f1000800000
length = 32768 * 1 = 32768
```

### 5.7 KV Cache 格式重组

传输后可能需要重组，因为 Prefill 和 Decode 可能使用不同的 TP 策略或内存布局：

#### 5.6.1 HMA 场景的重组

```python
@torch.no_grad()
def reformat_kv_cache_hybrid_linear_torch(self, block_ids: list[list[int]], tp_num_need_pulls: int, group_kv_caches):
    """HMA（混合注意力）场景的 KV cache 重组"""
    flat_block_ids = [item for sublist in block_ids for item in sublist]
    if not flat_block_ids or tp_num_need_pulls == 1:
        return
    device = list(self.kv_caches.values())[0][0].device
    block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.long, device=device)
    num_blocks = block_ids_tensor.numel()

    def _transpose_cache_by_block(cache: torch.Tensor):
        # 传输后的布局：[block, split, token, head_per_split, dim]
        # 目标布局：    [block, token, split, head_per_split, dim]
        selected = cache.index_select(0, block_ids_tensor)
        block_size = cache.shape[1]
        transposed = (
            selected.reshape(num_blocks, tp_num_need_pulls, block_size, -1)
            .transpose(1, 2)  # 交换 split 和 token 维度
            .contiguous()
            .reshape_as(selected)
        )
        cache.index_copy_(0, block_ids_tensor, transposed)

    for _, (k_cache_layer, v_cache_layer) in group_kv_caches.items():
        _transpose_cache_by_block(k_cache_layer)
        _transpose_cache_by_block(v_cache_layer)
```

#### 5.6.2 通用场景的重组

```python
def reformat_kv_cache(self, block_ids, tp_num_need_pulls, need_cat_cache=False, need_nz_cache=False, kv_caches=None):
    if kv_caches is None:
        kv_caches = self.kv_caches
    k_cache = list(kv_caches.values())[0][0]
    dtype = k_cache.dtype
    device = k_cache.device
    num_kv_heads, k_head_dim, v_head_dim = self._get_kv_cache_dims_from_tensors(kv_caches)

    flat_block_ids = [item for sublist in block_ids for item in sublist]
    block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int32, device=device)
    num_blocks = len(flat_block_ids)
    num_tokens = num_blocks * self.block_size

    # 创建设备张量用于拷贝操作
    block_table = block_ids_tensor.view(1, -1)
    block_len_tensor = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    seq_start_tensor = torch.tensor([0], dtype=torch.int32, device=device)

    k_buffer = torch.empty((num_tokens, num_kv_heads, k_head_dim), dtype=dtype, device=device)
    v_buffer = torch.empty((num_tokens, num_kv_heads, v_head_dim), dtype=dtype, device=device)

    # 创建 slot mapping 用于 reshape 操作
    block_offsets = torch.arange(0, self.block_size, dtype=torch.int32, device=device)
    slot_mapping = (
        block_offsets.reshape((1, self.block_size)) + block_ids_tensor.reshape((num_blocks, 1)) * self.block_size
    ).flatten()

    # 同步确保数据就绪
    torch.npu.synchronize()

    # 逐层处理 KV cache
    for _, (k_cache_layer, v_cache_layer) in kv_caches.items():
        # 加载 cache 数据到 buffer
        torch_npu.atb.npu_paged_cache_load(
            k_cache_layer, v_cache_layer,
            block_table, block_len_tensor,
            seq_starts=seq_start_tensor,
            key=k_buffer, value=v_buffer,
        )
        if need_cat_cache:
            # TP 合并：将多个 TP 分片拼成一个
            self._cat_kv_cache(k_cache_layer, v_cache_layer, k_buffer, v_buffer, tp_num_need_pulls, num_blocks, num_tokens, slot_mapping, num_kv_heads)
        if need_nz_cache:
            # NZ 格式转换
            self._nz_kv_cache(k_cache_layer, v_cache_layer, k_buffer, v_buffer, slot_mapping, num_kv_heads, k_head_dim, v_head_dim)

    del k_buffer, v_buffer
```

---

## 6. MooncakeConnectorWorker：Worker 侧连接器

### 6.1 功能

Worker 侧的连接器，负责：
1. 解析 prefill/decode 的并行配置（TP/DP/PP/PCP）
2. 计算 handshake 端口
3. 注册 KV cache 内存到 Mooncake TransferEngine
4. 根据角色（kv_producer/kv_consumer）启动发送或接收线程

### 6.2 初始化与并行配置解析

```python
class MooncakeConnectorWorker:
    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        self._get_prefill_decode_size(vllm_config)
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
        if self._prefill_tp_size < self._decode_tp_size:
            raise ValueError(
                f"prefill_tp_size: {self._prefill_tp_size} must be greater than"
                f" or equal to the decode_tp_size: {self._decode_tp_size}"
            )

        # 并行配置
        self.vllm_config = vllm_config
        self.ascend_config = get_ascend_config()
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_group = get_tp_group()
        self.pp_rank = get_pp_group().rank_in_group
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pcp_size = get_pcp_group().world_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        # 总设备数 = TP × DP × PCP × PP
        self.max_device_id = self.tp_size * self.dp_size * self.pcp_size * self.pp_size
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.num_key_value_heads = self.vllm_config.model_config.hf_text_config.num_key_value_heads

        # Handshake 基础端口计算
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
            * self.pcp_size
        )
        device_index = (self.pp_rank + self.pcp_rank) * self.tp_size + self.tp_rank
        self.handshake_port = self.side_channel_port + device_index

        # Mooncake TransferEngine 实例
        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()

        # 后台线程
        self.kv_send_thread: KVCacheSendingThread | None = None
        self.kv_recv_thread: KVCacheRecvingThread | None = None
```

### 6.3 解析 Prefill/Decode 并行配置

```python
def _get_prefill_decode_size(self, vllm_config: VllmConfig):
    # 从 extra config 获取 prefill 的并行配置
    prefill_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("prefill", {})
    assert "tp_size" in prefill_parallel_config
    self._prefill_tp_size = prefill_parallel_config["tp_size"]
    assert "dp_size" in prefill_parallel_config
    self._prefill_dp_size = prefill_parallel_config["dp_size"]
    self._prefill_pp_size = prefill_parallel_config.get("pp_size", 1)

    # 从 extra config 获取 decode 的并行配置
    decode_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("decode", {})
    assert "tp_size" in decode_parallel_config
    self._decode_tp_size = decode_parallel_config["tp_size"]
    assert "dp_size" in decode_parallel_config
    self._decode_dp_size = decode_parallel_config["dp_size"]
    self._decode_pp_size = decode_parallel_config.get("pp_size", 1)
    assert self._decode_pp_size == 1, "decode pp size must be 1"
    self._prefill_pp_layer_partition = prefill_parallel_config.get("pp_layer_partition")
```

### 6.4 注册 KV Cache

```python
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    """Register the KV Cache data."""
    self.use_mla = self.vllm_config.model_config.is_deepseek_mla
    self.use_sparse = hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")
    self.num_blocks = self.kv_cache_config.num_blocks
    self.kv_caches = kv_caches

    # 构建 group 到 layer 的映射
    self.kv_group2layeridx = self._build_kv_group2layeridx()
    has_mamba_group = self._has_mamba_group()

    # 计算每层 KV cache 的基地址、block 长度、缩放比例
    layer_name_to_idx = {
        layer_name: layer_idx
        for _, (group_spec, layer_indices) in self.kv_group2layeridx.items()
        for layer_name, layer_idx in zip(group_spec["layer_names"], layer_indices)
    }
    metadata_layers = max(layer_name_to_idx.values(), default=-1) + 1

    self.kv_caches_base_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
    self.block_size_scale: list[list[int]] = [[] for _ in range(metadata_layers)]
    self.block_len_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]

    for layer_name, kv_cache_tuple in kv_caches.items():
        layer_idx = layer_name_to_idx[layer_name]
        for single_kv_cache in self._as_kv_cache_tuple(kv_cache_tuple):
            tensor_num_blocks = single_kv_cache.shape[0]
            block_size_scale = tensor_num_blocks // self.num_blocks
            block_shape = single_kv_cache.shape[1:]
            self.block_len_per_addr[layer_idx].append(single_kv_cache.element_size() * math.prod(block_shape))
            self.block_size_scale[layer_idx].append(block_size_scale)
            self.kv_caches_base_addr[layer_idx].append(single_kv_cache.data_ptr())

    # 注册内存到 Mooncake TransferEngine
    if has_mamba_group:
        ptrs, lengths = self._get_registered_kv_tensor_buffers(kv_caches)
        register_regions = RegisterRegions(ptrs=ptrs, lengths=lengths)
    else:
        register_regions = collect_storage_merged_register_regions(kv_caches)

    validate_register_region_count(register_regions)
    global_te.register_buffer(register_regions.ptrs, register_regions.lengths)

    # 创建元数据并启动线程
    metadata = MooncakeAgentMetadata(
        engine_id=self.engine_id,
        te_rpc_port=self.te_rpc_port,
        kv_group2layeridx=self.kv_group2layeridx,
        block_size=self.block_size,
        kv_caches_base_addr=self.kv_caches_base_addr,
        block_size_scale=self.block_size_scale,
        num_blocks=self.num_blocks,
        block_lens=self.block_len_per_addr,
        local_ip=get_ip(),
    )
    self.xfer_handshake_metadata = metadata

    ready_event = threading.Event()
    if self.kv_role == "kv_producer":
        self.kv_send_thread = KVCacheSendingThread(...)
        self.kv_send_thread.start()
    else:
        self.kv_recv_thread = KVCacheRecvingThread(...)
        self.kv_recv_thread.start()
```

### 6.5 KV Cache 分片计算

```python
if self.vllm_config.model_config.is_deepseek_mla:
    self.tp_num_need_pulls = 1  # MLA 的 TP 处理不同
else:
    num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
    num_p_block_heads = max(1, self.num_key_value_heads // self._prefill_tp_size)
    self.tp_num_need_pulls = num_d_block_heads // num_p_block_heads
```

当 Prefill TP > Decode TP 时，Decode 需要从多个 Prefill rank 拉取 KV 分片。

---

## 7. MooncakeConnectorScheduler：Scheduler 侧连接器

### 7.1 功能

Scheduler 侧的连接器，负责**决策层**逻辑：
1. 决定哪些请求需要 KV 传输
2. 计算需要从远程拉取的 token 数
3. 在请求分配 block 后记录传输元数据
4. 请求完成时决定是否延迟释放 block

### 7.2 核心方法

#### 7.2.1 get_num_new_matched_tokens

```python
def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
    """
    对于远程 prefill，计算需要从外部 KV cache 拉取的 token 数。
    
    Returns:
        - 可以从外部 KV cache 加载的 token 数（超出本地已计算的部分）
        - 是否异步加载（在 scheduler 步骤之间）
    """
    params = request.kv_transfer_params
    logger.debug(
        "MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
        num_computed_tokens, params,
    )

    if params is not None and params.get("do_remote_prefill"):
        # 远程 prefill：获取所有 prompt blocks
        assert num_computed_tokens % self.block_size == 0
        params["num_computed_tokens"] = num_computed_tokens
        count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
        return count, count > 0

    # 该请求无需远程 prefill
    return 0, False
```

#### 7.2.2 update_state_after_alloc

```python
def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
    params = request.kv_transfer_params
    logger.debug(
        "MooncakeConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
        num_external_tokens, params,
    )

    if params is not None and (params.get("do_remote_prefill", False) or params.get("do_remote_decode", False)):
        self._reqs_in_batch.add(request.request_id)
    
    if params is not None and params.get("do_remote_prefill"):
        if params.get("remote_block_ids"):
            if all(p in params for p in ("remote_engine_id", "remote_host", "remote_port", "remote_request_id")):
                local_block_ids = blocks.get_unhashed_block_ids_all_groups() if num_external_tokens > 0 else []
                # 获取需要从远程拉取的 unhashed blocks
                self._reqs_need_recv[request.request_id] = (request, local_block_ids, num_external_tokens)
            else:
                logger.warning("Got invalid KVTransferParams. params=%s", params)
        else:
            assert num_external_tokens == 0
        # 每个请求只触发 1 次 KV 传输
        params["do_remote_prefill"] = False
```

#### 7.2.3 build_connector_meta

```python
def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
    meta = MooncakeConnectorMetadata()

    # 遍历需要接收的请求，转换为 ReqMeta
    for req_id, (req, block_ids, num_external_tokens) in self._reqs_need_recv.items():
        assert req.kv_transfer_params is not None
        meta.add_new_req(
            request_id=req_id,
            local_block_ids=block_ids,
            num_external_tokens=num_external_tokens,
            kv_transfer_params=req.kv_transfer_params,
        )

    # 清空列表，避免重复调度
    self._reqs_need_recv.clear()
    meta.requests_to_send = self._reqs_need_send
    self._reqs_need_send = {}
    meta.reqs_in_batch = self._reqs_in_batch
    self._reqs_in_batch = set()

    return meta
```

#### 7.2.4 request_finished

```python
def request_finished(self, request: "Request", block_ids: BlockIds) -> tuple[bool, dict[str, Any] | None]:
    """
    请求完成后，判断是否需要延迟释放 block。
    """
    params = request.kv_transfer_params
    logger.debug(
        "MooncakeConnector request_finished, request_status=%s, kv_transfer_params=%s",
        request.status, params,
    )

    if (
        params is None
        or not params.get("do_remote_decode")
        or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
    ):
        return False, None

    computed_block_ids = block_ids
    computed_block_lens = [len(block_id_list) for block_id_list in computed_block_ids]
    delay_free_blocks = sum(computed_block_lens) > 0
    if delay_free_blocks:
        logger.info("Delaying free of %d blocks for request %s", len(computed_block_ids), request.request_id)
        self._reqs_need_send[request.request_id] = time.time()

    num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)
    computed_block_ids = tuple(
        block_ids[:num_prompt_blocks]
        if not isinstance(self.kv_cache_groups[i].kv_cache_spec, MambaSpec)
        else block_ids
        for i, block_ids in enumerate(computed_block_ids)
    )

    return delay_free_blocks, dict(
        do_remote_prefill=True,
        do_remote_decode=False,
        remote_block_ids=computed_block_ids,
        remote_engine_id=self.engine_id,
        remote_request_id=request.request_id,
        remote_host=self.side_channel_host,
        remote_port=self.side_channel_port,
        remote_pcp_size=self.pcp_size,
        remote_dcp_size=self.dcp_size,
        remote_ptp_size=self.tp_size,
        last_token_id=request.output_token_ids[-1],
        remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
        num_prompt_blocks=num_prompt_blocks,
    )
```

**延迟释放机制**：
- 当请求因长度限制完成（`FINISHED_LENGTH_CAPPED`）且需要远程 decode 时
- 不能立即释放 block，因为 decode 节点可能需要继续拉取 KV
- 记录到 `_reqs_need_send`，等待 decode 确认后再释放

---

## 8. MooncakeConnector：统一入口

### 8.1 功能

根据运行角色（Scheduler 或 Worker）创建对应的连接器实例，并暴露统一接口。

### 8.2 代码

```python
class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = MooncakeConnectorScheduler(
                vllm_config, str(self.engine_id), kv_cache_config
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(self, request: "Request", block_ids: tuple[list[int], ...]) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        """MooncakeConnector does not save explicitly."""
        pass

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata

    def set_xfer_handshake_metadata(self, metadata: dict[int, KVConnectorHandshakeMetadata]) -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)
```

### 8.3 继承的接口含义

| 接口 | 含义 |
|------|------|
| `KVConnectorBase_V1` | vLLM 定义的 KV 连接器基类（V1 版本），规定所有 KV 传输插件必须实现的接口 |
| `SupportsHMA` | Hybrid Memory Attention（混合内存注意力）支持标记接口，用于标识该连接器支持混合注意力架构 |

---

## 9. 关键流程详解

### 9.1 完整 KV 传输流程

```
Step 1: Decode Scheduler 调度请求
        ├── 请求带有 kv_transfer_params: {do_remote_prefill: True, ...}
        ├── 调用 get_num_new_matched_tokens() → 返回需要拉取的 token 数
        └── 调用 update_state_after_alloc() → 记录到 _reqs_need_recv

Step 2: Decode Scheduler 构建 connector meta
        ├── 调用 build_connector_meta()
        ├── 将 _reqs_need_recv 转换为 MooncakeConnectorMetadata
        └── 传递给 Worker

Step 3: Decode Worker 启动加载
        ├── 调用 start_load_kv()
        ├── KVCacheRecvingThread 从队列取出请求
        └── 调用 _transfer_kv_cache_all_groups()

Step 4: 元数据握手
        ├── Decode → Prefill: GET_META_MSG
        ├── Prefill → Decode: MooncakeAgentMetadata
        └── Decode 缓存元数据到 SizedDict

Step 5: 批量传输
        ├── 计算所有 layer 的 src/dst/length 列表
        ├── 调用 engine.batch_transfer_sync_read()
        └── 等待传输完成

Step 6: 格式重组
        ├── 如果需要：reformat_kv_cache_hybrid_linear_torch()
        ├── 如果需要：reformat_kv_cache_with_fused_op()
        └── 或不重组

Step 7: 完成通知
        ├── Decode → Prefill: DONE_RECVING_MSG
        ├── Prefill → Decode: ACK
        └── Prefill 释放 KV block
```

### 9.2 PCP/DCP 场景下的分片传输

当启用 Context Parallel 时，序列被切分到多卡：

```python
def get_kv_head_groups(tp_size):
    """获取 KV head 分组，用于 CP 场景"""
    if self.use_mla or self.use_sparse:
        return [tuple([0])]
    if self.num_key_value_heads // tp_size >= 1:
        kv_head_groups = []
        for tp_rank in range(tp_size):
            kv_head_ids = [
                head_idx + tp_rank * (self.num_key_value_heads // tp_size)
                for head_idx in range(self.num_key_value_heads // tp_size)
            ]
            kv_head_groups.append(tuple(kv_head_ids))
        return kv_head_groups
    if tp_size // self.num_key_value_heads > 1:
        kv_head_groups = []
        for kv_head_ids_ in range(self.num_key_value_heads):
            kv_head_groups.append(tuple([kv_head_ids_]))
        return kv_head_groups
```

---

## 10. 与 EPD Proxy 的协作关系

### 10.1 架构位置

```
Client
  │
  ▼
[Proxy: epd_load_balance_proxy_layerwise_server_example.py]
  ├── 分发多模态请求到 Encoder
  ├── 分发文本请求到 Decoder（注入 metaserver 回调）
  └── 接收 metaserver 回调，触发 Prefiller
  │
  ▼
[Encoder] → 处理图片/视频 → 输出视觉 token
  │
  ▼
[Decoder] → 接收请求 → 发现 do_remote_prefill=True
  │           └── 向 metaserver 发送 KV 传输参数
  │
  ▼
[Proxy /v1/metaserver] → 选择 Prefiller → 发送请求
  │
  ▼
[Prefiller] → 执行 Prefill → KV cache 写入 NPU 内存
  │           └── MooncakeConnector (kv_producer)
  │               └── KVCacheSendingThread 监听元数据请求
  │
  ▼
[Decode Worker] → KVCacheRecvingThread 拉取 KV
  │               └── Mooncake TransferEngine (RDMA/Network)
  │
  ▼
[Decode] → 收到 KV → 开始 Decode → 流式返回 token
```

### 10.2 Proxy 代码中的关键注入

```python
# _handle_completions 中，Decoder 请求注入 metaserver 地址
req_data["kv_transfer_params"] = {
    "do_remote_decode": False,
    "do_remote_prefill": True,
    "metaserver": f"http://{global_args.host}:{global_args.port}/v1/metaserver",
}

# metaserver 回调中，选择 Prefiller 并发送请求
@app.post("/v1/metaserver")
async def metaserver(request: Request):
    kv_transfer_params = await request.json()
    request_id = kv_transfer_params["request_id"]
    req_data, token_score, api = proxy_state.req_data_dict[request_id]
    req_data["kv_transfer_params"] = kv_transfer_params
    
    prefiller_idx = proxy_state.select_prefiller(token_score)
    prefiller = proxy_state.prefillers[prefiller_idx]
    _ = await send_request_to_service(prefiller.client, prefiller_idx, api, req_data, request_id)
    proxy_state.release_prefiller(prefiller_idx, token_score)
```

### 10.3 配置示例

```bash
# Prefiller 启动（kv_producer）
vllm serve "/path/to/model" \
    --kv-transfer-config '{
        "kv_connector": "MooncakeLayerwiseConnector",
        "kv_role": "kv_producer",
        "kv_port": "50001",
        "kv_connector_extra_config": {
            "prefill": {"tp_size": 2, "dp_size": 1},
            "decode": {"tp_size": 1, "dp_size": 1}
        }
    }'

# Decoder 启动（kv_consumer）
vllm serve "/path/to/model" \
    --kv-transfer-config '{
        "kv_connector": "MooncakeLayerwiseConnector",
        "kv_role": "kv_consumer",
        "kv_port": "50001",
        "kv_connector_extra_config": {
            "prefill": {"tp_size": 2, "dp_size": 1},
            "decode": {"tp_size": 1, "dp_size": 1}
        }
    }'
```

---

## 附录：关键术语表

| 术语 | 全称 | 含义 |
|------|------|------|
| TP | Tensor Parallel | 张量并行 |
| PP | Pipeline Parallel | 流水线并行 |
| DP | Data Parallel | 数据并行 |
| PCP | Prefill Context Parallel | Prefill 上下文并行 |
| DCP | Decode Context Parallel | Decode 上下文并行 |
| HMA | Hybrid Memory Attention | 混合内存注意力 |
| MLA | Multi-head Latent Attention | 多头潜在注意力（DeepSeek） |
| KV Cache | Key-Value Cache | 注意力机制的键值缓存 |
| Block | - | KV cache 的内存分配单元 |
| Mooncake | - | 开源高性能 KV 传输引擎 |
| ZMQ | ZeroMQ | 消息队列库，用于元数据握手 |
| RDMA | Remote Direct Memory Access | 远程直接内存访问 |
| EPD | Encoder-Prefill-Decode | 分离式推理架构 |

---

> 文档版本：基于 vllm-ascend 代码分析
> 生成日期：2026-06-15
