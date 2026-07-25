## MemCache 配置说明

#### MetaService（元数据服务）配置

| 配置项 | 值类型 | 是否必填 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|--------|----------|------|
| ock.mmc.meta_service_url | string | 可选 | tcp://127.0.0.1:5000 | tcp://\<host>\<port> | host 支持 IP 和域名，端口范围 [1025, 65535] |
| ock.mmc.meta_service.config_store_url | string | 可选 | tcp://127.0.0.1:6000 | tcp://\<host>\<port> | host 支持 IP 和域名，端口范围 [1025, 65535] |
| ock.mmc.meta_service.metrics_url | string | 可选 | http://127.0.0.1:8000 | http://127.0.0.1:\<port> | host 必须为 127.0.0.1，端口范围 [1025, 65535] |
| ock.mmc.meta.ha.enable | bool | 可选 | false | true/false | 在 k8s 集群中启用元数据服务的主备高可用（HA） |
| ock.mmc.meta_service.metrics_report_interval_seconds | integer | 可选 | 30 | [0, 86400] | 指标汇总打印间隔（秒），设置为 0 表示关闭周期性指标打印 |
| ock.mmc.log_level | string | 可选 | info | debug/info/warn/error | 日志级别 |
| ock.mmc.log_path | string | 可选 | /var/log/memcache_hybrid | 相对路径或绝对路径 | 日志路径，绝对路径以 '/' 开头 |
| ock.mmc.log_rotation_file_size | int | 可选 | 20 | 1 <= n <= 500 | 日志轮转文件大小（MB） |
| ock.mmc.log_rotation_file_count | int | 可选 | 50 | 1 <= n <= 50 | 日志轮转文件数量 |
| ock.mmc.evict_threshold_high | int | 可选 | 90 | 1 <= n <= 99 | 淘汰阈值，90 表示 90%，最大阈值为 99%。注意：当单个 put 值的大小超过容量的 1% 时，无法触发淘汰 |
| ock.mmc.evict_threshold_low | int | 可选 | 80 | 1 <= n <= 98 | 淘汰后的目标阈值 |
| | | | | | |
| ock.mmc.tls.enable | bool | 可选 | false | true/false | metaservice 的 TLS 开关 |
| ock.mmc.tls.ca.path | string | 可选 | | 0 <= len < 256 | 根证书路径 |
| ock.mmc.tls.ca.crl.path | string | 可选 | | 0 <= len < 256 | 证书吊销列表（CRL）路径 |
| ock.mmc.tls.cert.path | string | 可选 | | 0 <= len < 256 | 服务端证书路径 |
| ock.mmc.tls.key.path | string | 可选 | | 0 <= len < 256 | 服务端私钥路径 |
| ock.mmc.tls.key.pass.path | string | 可选 | | 0 <= len < 256 | 服务端私钥口令文件路径（若私钥未加密则留空） |
| ock.mmc.tls.package.path | string | 可选 | | 0 <= len < 256 | openssl 动态库路径 |
| ock.mmc.tls.decrypter.path | string | 可选 | | 0 <= len < 256 | 口令解密库路径（若口令未加密则留空） |
| | | | | | |
| ock.mmc.config_store.tls.enable | bool | 可选 | false | true/false | config store（配置存储）的 TLS 开关 |
| ock.mmc.config_store.tls.ca.path | string | 可选 | | 0 <= len < 256 | config store 的根证书路径 |
| ock.mmc.config_store.tls.ca.crl.path | string | 可选 | | 0 <= len < 256 | config store 的证书吊销列表路径 |
| ock.mmc.config_store.tls.cert.path | string | 可选 | | 0 <= len < 256 | config store 的服务端证书路径 |
| ock.mmc.config_store.tls.key.path | string | 可选 | | 0 <= len < 256 | config store 的服务端私钥路径 |
| ock.mmc.config_store.tls.key.pass.path | string | 可选 | | 0 <= len < 256 | config store 的服务端私钥口令文件路径（若私钥未加密则留空） |
| ock.mmc.config_store.tls.package.path | string | 可选 | | 0 <= len < 256 | config store 的 openssl 动态库路径 |
| ock.mmc.config_store.tls.decrypter.path | string | 可选 | | 0 <= len < 256 | config store 的口令解密库路径（若口令未加密则留空） |

#### LocalService（本地服务）配置

| 配置项 | 值类型 | 是否必填 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|--------|----------|------|
| ock.mmc.meta_service_url | string | 可选 | tcp://127.0.0.1:5000 | tcp://\<host>\<port> | HA 场景下 host 为集群 IP/域名，端口范围 [1025, 65535] |
| ock.mmc.local_service.config_store_url | string | 可选 | tcp://127.0.0.1:6000 | tcp://\<host>\<port> | host 支持 IP 和域名，端口范围 [1025, 65535] |
| ock.mmc.log_level | string | 可选 | info | debug/info/warn/error | 日志级别 |
| | | | | | |
| ock.mmc.local_service.world_size | integer | 可选 | 256 | [1, 1024] | 支持的最大 rank 数量；一旦 rank 完成连接，不允许再修改——需要重启 meta 服务 |
| | | | | | |
| ock.mmc.local_service.protocol | string | **必填** | host_rdma | host_rdma/host_urma/host_tcp/host_shm/device_sdma/device_rdma/device_urma | host_shm 要求 DRAM > 0、HBM = 0，且不使用 hcom |
| ock.mmc.local_service.hcom_url | string | 可选 | tcp://127.0.0.1:7000 | tcp://\<host>\<port> | 用于 DRAM 池，host 支持 IP 和域名，端口范围 [1024, 65535] |
| | | | | | |
| ock.mmc.local_service.dram.size | integer | **必填** | 1GB | [0, 1TB] | 支持 134217728、2048KB、200mb、2.5G、1TB 等格式，<br/> 并自动对齐到 2MB（host_rdma、host_tcp 或 host_shm）或 1GB（device_sdma、device_rdma 或 device_urma） |
| ock.mmc.local_service.hbm.size | integer | 可选 | 0 | [0, 1TB] | 支持 134217728、2048KB、200mb、2.5G、1TB 等格式，<br/> 并自动对齐到 2MB（host_rdma、host_tcp 或 host_shm）或 1GB（device_sdma、device_rdma 或 device_urma）。host_shm 要求 HBM 保持为 0 |
| ock.mmc.local_service.max.dram.size | integer | 可选 | 取 dram.size 的值 | [0, 1TB] | 所有本地进程中 `ock.mmc.local_service.dram.size` 的最大值，当各 rank 贡献的 DRAM 大小不同时必须配置 |
| ock.mmc.local_service.max.hbm.size | integer | 可选 | 取 hbm.size 的值 | [0, 1TB] | 所有本地进程中 `ock.mmc.local_service.hbm.size` 的最大值，当各 rank 贡献的 HBM 大小不同时必须配置 |
| | | | | | |
| ock.mmc.client.retry_milliseconds | integer | 可选 | 0 | [0, 600000] | 客户端请求元数据服务且连接不存在时的总重试时长（重试间隔为 200ms） |
| ock.mmc.client.timeout.seconds | integer | 可选 | 60 | [1, 600] | 客户端请求超时时间（秒） |
| ock.mmc.client.read_thread_pool.size | integer | 可选 | 4 | [1, 64] | 读线程池的线程数 |
| ock.mmc.client.write_thread_pool.size | integer | 可选 | 4 | [1, 64] | 写线程池的线程数 |
| ock.mmc.client.aggregate.io | bool | 可选 | true | true/false | 是否为读写操作启用 IO 请求聚合 |
| ock.mmc.client.aggregate.num | integer | 可选 | 122 | [1, 131072] | 单批次聚合的 IO 请求数量 |
| ock.mmc.client.batch_option.chunk.size | integer | 可选 | 8MB | [0, 1TB] | 支持 134217728、2048KB、200mb、2.5G、1TB 等格式 |
| ock.mmc.client.batch_option.chunk.count | integer | 可选 | 3 | [1, 64] | 在批量拷贝操作中，仅当批量数量小于 chunk 数量，或总数据大小小于（chunk size × chunk count）时，系统才不执行切片和并发拷贝操作 |
| | | | | | |
| ock.mmc.tls.enable | bool | 可选 | false | true/false | metaservice 的 TLS 开关 |
| ock.mmc.tls.ca.path | string | 可选 | | 0 <= len < 256 | 根证书路径 |
| ock.mmc.tls.ca.crl.path | string | 可选 | | 0 <= len < 256 | 证书吊销列表（CRL）路径 |
| ock.mmc.tls.cert.path | string | 可选 | | 0 <= len < 256 | 客户端证书路径 |
| ock.mmc.tls.key.path | string | 可选 | | 0 <= len < 256 | 客户端私钥路径 |
| ock.mmc.tls.key.pass.path | string | 可选 | | 0 <= len < 256 | 客户端私钥口令文件路径（若私钥未加密则留空） |
| ock.mmc.tls.package.path | string | 可选 | | 0 <= len < 256 | openssl 动态库路径 |
| ock.mmc.tls.decrypter.path | string | 可选 | | 0 <= len < 256 | 口令解密库路径（若口令未加密则留空） |
| | | | | | |
| ock.mmc.config_store.tls.enable | bool | 可选 | false | true/false | config store（配置存储）的 TLS 开关 |
| ock.mmc.config_store.tls.ca.path | string | 可选 | | 0 <= len < 256 | config store 的根证书路径 |
| ock.mmc.config_store.tls.ca.crl.path | string | 可选 | | 0 <= len < 256 | config store 的证书吊销列表路径 |
| ock.mmc.config_store.tls.cert.path | string | 可选 | | 0 <= len < 256 | config store 的客户端证书路径 |
| ock.mmc.config_store.tls.key.path | string | 可选 | | 0 <= len < 256 | config store 的客户端私钥路径 |
| ock.mmc.config_store.tls.key.pass.path | string | 可选 | | 0 <= len < 256 | config store 的客户端私钥口令文件路径（若私钥未加密则留空） |
| ock.mmc.config_store.tls.package.path | string | 可选 | | 0 <= len < 256 | config store 的 openssl 动态库路径 |
| ock.mmc.config_store.tls.decrypter.path | string | 可选 | | 0 <= len < 256 | config store 的口令解密库路径（若口令未加密则留空） |
| | | | | | |
| ock.mmc.local_service.hcom.tls.enable | bool | 可选 | false | true/false | hcom 的 TLS 开关 |
| ock.mmc.local_service.hcom.tls.ca.path | string | 可选 | | 0 <= len < 256 | hcom 的根证书路径 |
| ock.mmc.local_service.hcom.tls.ca.crl.path | string | 可选 | | 0 <= len < 256 | hcom 的证书吊销列表路径 |
| ock.mmc.local_service.hcom.tls.cert.path | string | 可选 | | 0 <= len < 256 | hcom 的客户端证书路径 |
| ock.mmc.local_service.hcom.tls.key.path | string | 可选 | | 0 <= len < 256 | hcom 的客户端私钥路径 |
| ock.mmc.local_service.hcom.tls.key.pass.path | string | 可选 | | 0 <= len < 256 | hcom 的客户端私钥口令文件路径（若私钥未加密则留空） |
| ock.mmc.local_service.hcom.tls.decrypter.path | string | 可选 | | 0 <= len < 256 | hcom 的口令解密库路径（若口令未加密则留空） |

---

## 多级缓存：淘汰与回温（Rewarm）机制详解

> 背景：仓库支持 HBM / DRAM / SSD 三级存储介质（`MediaType` 枚举见 `src/memcache/csrc/common/mmc_types.h:78`），
> 层间迁移路径由 `MoveUp()` / `MoveDown()`（同文件 :85-109）定义，**只支持相邻层单步迁移**：
> 下行 HBM→DRAM→SSD，上行 SSD→DRAM（回温到 HBM 当前版本暂不支持），**不存在 HBM↔SSD 的直接迁移路径**。
>
> ```
> 写/回温上行： SSD ──rewarm──> DRAM        （DRAM→HBM 回温暂不支持）
> 淘汰下行：    HBM ──evict──> DRAM ──evict──> SSD ──evict──> 删除
> ```

### 问题 1：按 LRU 淘汰，具体的淘汰计算数量和规则是怎么样的？

#### 数据结构：每层一条独立的 LRU 链表

`MmcMetaContainerLRU`（`mmc_meta_container_lru.cpp:38-41`）：

```
metaMap_ (unordered_map)                lruLists_[3] —— 每层一条 list
┌──────────┬──────────────┐            HBM: [key5] ←→ [key2] ←→ [key9]
│   key    │ ValueLruItem │            DRAM: [key7] ←→ [key1] ←→ [key3] ←→ [key8]
│          │  ├ value_    │            SSD: [key4] ←→ [key6]
│          │  ├ mediaType_ │             ↑push_front       ↑prev(end) = 最久未用
│          │  └ lruIter_ ──┼──→ 指向其在 LRU 链表中的位置   (MRU在头, LRU在尾)
└──────────┴──────────────┘
```

**LRU 维护规则**：
- `Insert`：push_front 到对应层的链表头（`:62`）
- `Get` / `ExistKey` 命中：`Promote` → `UpdateLRU`，把 key 移到链表头（`mmc_meta_manager.cpp:57`、`:186`）
- 回温完成：`InsertLru(key, dstType)` 挂到目标层链表头
- 淘汰中：`mediaType_` 暂置 `NONE`，表示"在迁移中"，此时 `Promote` 会跳过（`:174-177`）

#### 淘汰触发与数量计算

```
Alloc/BatchAlloc 请求
   └─> CheckAndEvict(media, wantAllocSize)              (mmc_meta_mgr_proxy.cpp:52,85)
        └─> GetNeedEvictList(高水位90)                    (mmc_global_allocator.h:286)
             按 SSD→DRAM→HBM 顺序检查每层:
                usedSize*100 > totalSize*90  → 该层进入 needEvictList
        └─> CAS 保证只有一轮淘汰在跑 → 线程池异步执行
        └─> MultiLevelElimination(...)                   (mmc_meta_container_lru.cpp:249)
             对 needEvictList 中的每一层:
```

**数量公式**（`mmc_meta_container_lru.cpp:266-268`）：

```
numEvictObjs = max( min(oriNum × (nowThreshold − low) / high, oriNum), 1 )
```

其中 `nowThreshold = usedSize×100/totalSize`（当前水位），`high=90`、`low=80`（可配，`ock.mmc.evict_threshold_high/low`）。

**举例**：某层有 1000 个 key，当前水位 95%：
`1000 × (95−80) / 90 = 166` 个 → 淘汰后水位约降到 83%，接近低水位。即**超得越多，淘汰越多，目标是把水位压回低水位附近**。

**选择规则**：`EvictOneLeastRecentlyUsed`（`:204`）每次取该层 LRU **链表尾**（`std::prev(end())`），回调结果决定去向：

```
取尾部 key ──> EvictCallBackFunction(key, meta, srcMedia)   (mmc_meta_manager.cpp:938)
                 │
                 ├─ src=SSD(无下层) ────────────→ REMOVE：彻底删除
                 ├─ 下层剩余空间 < objSize ──────→ REMOVE：降级无望，直接删除
                 └─ 否则 ──→ 异步 MoveBlob(src→下一层)，返回 MOVE_DOWN
                              （从本层LRU摘除，mediaType_置NONE，搬完再更新）
```

### 问题 2：Get 路径同步回温和 ExistKey 路径异步回温的触发条件是什么？过程如何？

先说结论对比：

| | Get 路径（同步回温） | ExistKey 路径（异步回温） |
|---|---|---|
| 入口 | `MmcMetaManager::Get` → `FillObjMetaWithRewarm` | `MmcMetaManager::ExistKey` |
| 触发条件 | 要读数据，但高层没有可读 blob，只有低层可读 blob | key 存在，**只有** SSD 可读 blob（无高层可读、无高层回温中） |
| 执行方式 | **在请求线程上同步执行** `RewarmBlob`，读完回温后的新 blob 才返回 | **丢线程池异步执行**，`ExistKey` 立即返回 OK |
| 目的 | 本次读就要用高层数据（或至少触发回温） | 探测存在性时顺便"预热"，下次读就快了 |
| 代码 | `mmc_meta_manager.cpp:66-171` | `:173-242` + `TriggerAsyncRewarm:509` |

#### Get 路径详细流程（`FillObjMetaWithRewarm`）

```
Get(key)
  ├─ metaContainer_->Get(key) 找到 memObj
  ├─ Promote(key)                        // LRU 提到链表头
  └─ FillObjMetaWithRewarm(key, ...)     // 加 memObj 锁，遍历 blob（见问题4）
        │
        ▼ 遍历结果分 4 种情况：
  ┌─────────────────────────────────────────────────────────────┐
  │ ① 高层有 READABLE blob → 直接选它读，不回温                    │
  │                                                             │
  │ ② 高层只有 ALLOCATED + 低层有 READABLE（回温进行中）           │
  │    → WaitUntilReadable 等最多 100ms，被 cv_ 唤醒后读新 blob   │
  │      超时返回 MMC_TIMEOUT                      (:103-110)   │
  │                                                             │
  │ ③ 只有 ALLOCATED、无低层伴生（写入进行中）                     │
  │    → 对读不可见，返回 numBlobs_=0              (:113-120)   │
  │                                                             │
  │ ④ 高层啥都没有，只有低层 READABLE（如只有 SSD）                │
  │    → dst = MoveUp(src)；若 dst==HBM 则跳过回温直接读低层      │
  │    → 否则同步调 RewarmBlob 回温 SSD→DRAM，                    │
  │      成功后改读新 blob；失败则报错并 READ_FINISH (:122-154)  │
  └─────────────────────────────────────────────────────────────┘
        │
        ▼
  对选中的 blob 做 MMC_READ_START（租约/引用计数），把 blob 描述返回给客户端
```

#### ExistKey 路径详细流程

```
ExistKey(key)
  ├─ metaContainer_->Get + Promote
  ├─ 加锁，filter=READABLE 取所有可读 blob，统计：
  │     hasReadableHigher = 有没有 HBM/DRAM 的 READABLE blob
  │     ssdDesc           = SSD blob 的描述
  ├─ 若高层不可读，再查有没有 ALLOCATED 的 HBM/DRAM blob（hasPendingDram）
  │     —— 防止和进行中的回温重复触发
  ├─ hasOnlyReadableSsd = !hasReadableHigher && !hasPendingDram && ssdDesc.size_>0
  │                                                             (:235)
  └─ 若 hasOnlyReadableSsd → TriggerAsyncRewarm(key, memObj, ssdDesc)  (:240)
        │
        ▼ (mmc_meta_manager.cpp:509)
        dst = MoveUp(SSD) = DRAM；dst==HBM/NONE 则跳过
        丢线程池异步执行：
          ① CheckAndEvict(DRAM, size)   // 先给 DRAM 腾地方
          ② 加锁二次检查：是否已有 DRAM/HBM 的 READABLE/ALLOCATED blob
             （异步任务排队期间可能已被别的路径回温过了） (:521-535)
          ③ RewarmBlob(...) 执行回温
```

### 问题 3：可读 blob 是什么概念？blob 是什么概念？

**blob = 一个 key 的数据在某一个（节点, 介质）上的一份物理副本的元信息**（`mmc_mem_blob.h:82`）：

```
MmcMemBlob {
    rank_       // 所在节点（rank id）
    gva_        // 全局虚拟地址（数据实际位置）
    size_       // 数据大小
    mediaType_  // 所在介质：HBM / DRAM / SSD
    state_      // 状态机（见下）
    prot_       // 读写权限
    metaLeaseManager_  // 租约管理（读写引用、TTL）
}
```

一个 key（`MmcMemObjMeta`）下可以挂**多个 blob**（通过 `Next()` 串成链），比如回温后 SSD 一份 + DRAM 一份同时存在：

```
key="tensor_A" (MmcMemObjMeta)
   └── blob#1 {DRAM, rank2, gva=0x..., READABLE} ──Next()──> blob#2 {SSD, rank2, READABLE}
```

**blob 状态机**（`mmc_blob_state.h:29-41`）：

```
 NONE ──alloc──> ALLOCATED ──write ok──> READABLE ──remove──> REMOVING ──> NONE
                  │                         ▲
                  └── 空间已分配但数据未写完 ──┘
                  （对读路径不可见，除非是回温中且有低层伴生）
```

**"可读 blob" = 状态为 `READABLE` 的 blob**，即数据已完整写入、可以被 Get 直接读走。`ALLOCATED` 表示"坑位已占、数据在写"，`REMOVING` 表示正在删除。

### 问题 4：Get 时遍历 blob 的详细过程

遍历代码在 `mmc_meta_manager.cpp:80-100`，用三个指针分类收集：

```
blobs = memObj->GetBlobs(filterPtr)   // 该 key 的 blob 链

selectedBlob = null    // 最终选来读的
pendingBlob  = null    // 回温/写入进行中的
lowerBlob    = null    // 最低层(SSD)的可读副本

for each blob in blobs:
    type = blob->Type()
    ┌────────────────────────────────────────────────────────┐
    │ if MoveDown(type)==NONE  → 是最低层(SSD):               │
    │      state==READABLE 且 lowerBlob为空 → lowerBlob=blob │
    │ else (HBM/DRAM):                                       │
    │      state==READABLE  → selectedBlob=blob; break;  ◄── 找到立即可读的，停止
    │      state==ALLOCATED 且 pendingBlob为空 → pendingBlob │
    └────────────────────────────────────────────────────────┘
```

遍历逻辑的三个要点：

1. **高层优先**：遇到 HBM/DRAM 的 READABLE 立即 `break`，不再看后面的——哪怕 SSD 也有副本，也读快的；
2. **SSD 只当"保底"**：SSD 的 READABLE 不直接选中，只记入 `lowerBlob`，用于"高层都没有时触发回温"或"回温跳过时的兜底读取"；
3. **ALLOCATED 单独记账**：`pendingBlob` 区分两种后续——有 `lowerBlob` 伴生 → 是回温中，可以等；没有伴生 → 是写入中，对读不可见。

遍历完后的决策树（即问题 2 的 ①②③④）：

```
selectedBlob? ──是──> 读它（走高层，不回温）
     │否
pendingBlob 且 lowerBlob? ──是──> 等≤100ms 回温完成，读新 blob
     │否
pendingBlob? ──是──> 写入中，返回空（客户端视为未就绪）
     │否
lowerBlob? ──是──> 触发回温（或直接读 SSD 兜底）
```

### 问题 5：回温执行（RewarmBlob）的详细过程

`RewarmBlob`（`mmc_meta_manager.cpp:1000-1080`），以 SSD→DRAM 为例：

```
                      RewarmBlob(key, srcDesc=SSD blob, dst=DRAM)
                                │
 ① 分配目标 blob                ▼
    AllocOptions: size=srcDesc.size_, media=DRAM
    flags: dst==HBM ? 随机节点 : ALLOC_FORCE_BY_RANK
           → DRAM 优先分配到源数据所在节点，利用本地内存带宽 (:1008-1012)
    globalAllocator_->Alloc() 失败 → 返回 MMC_MALLOC_FAILED
                                │
 ② 先登记、后拷贝               ▼
    newBlob->UpdateState(MMC_ALLOCATED_OK)
    objMeta->AddBlob(newBlob)          ← 关键：拷贝前先挂到 key 上，
    此时新 blob 是 ALLOCATED 状态           让并发 Get 能"看见回温进行中"
                                │      从而走问题4的等待分支 (:1022-1034)
 ③ 解锁后 RPC 拷数据            ▼
    guard.unlock()                     ← 拷贝期间不 holding key 的锁
    BlobCopyRequest{key, srcDesc, dstDesc}
    metaNetServer_->SyncCall(dstRank)  → 数据 SSD ──RPC──> DRAM
         │失败                          (:1053-1055)
         ▼
      rollback(): 加锁、按 (rank,DRAM) filter 把新 blob FreeBlobs 掉，返回错误
         │成功
 ④ 置可读                       ▼
    guard.lock()
    newBlob->UpdateState(MMC_WRITE_OK) → 状态变 READABLE
    （此时 cv_.NotifyReadable 唤醒等待中的并发 Get）
                                │
 ⑤ 挂入目标层 LRU               ▼
    guard.unlock()
    metaContainer_->InsertLru(key, DRAM)   ← 故意在 objMeta 锁外调用，
    IncrementRewarmCounter()                  避免与淘汰路径锁序死锁 (:1073)
```

时序上看两个并发 Get 的协同：

```
线程A (先到的Get)                线程B (后到的Get)
   │ 发现只有SSD blob               │
   │ RewarmBlob:                   │ 遍历blob: 无READABLE高层，
   │   AddBlob(ALLOCATED) ────────>│   但 pendingBlob(ALLOCATED)
   │   解锁, RPC拷贝中...          │   + lowerBlob(SSD) →
   │          │                    │ WaitUntilReadable(≤100ms)
   │   WRITE_OK → READABLE ──────> │ cv_ 唤醒，直接读 DRAM 新 blob
   ▼                               ▼
```

这样设计的核心：**回温只执行一次**（A 做），**并发的读者不重复回温也不读慢速 SSD**（B 等待后直接读 DRAM），同时拷贝长耗时阶段不持锁，不阻塞其他 key 的操作。

---

## Q&A：MemcacheBackend 的创建、数量与 key 语义

> **问题：**
> 1. 每个 NPU rank 创建一个 store 实例并初始化，这个动作请给出代码依据。
> 2. MemcacheBackend 对象会创建几次？结合 DP=2, PP=2, TP=2 来说明。
> 3. 请举例解释一下 key 的语义：`model@pcp@dcp@tp_rank@pp_rank@group@cache_role@cache_family@chunk_hash`

### 1. "每个 NPU rank 创建一个 store 实例并初始化"的代码依据

**调用链（每个 worker 进程各走一遍）：**

```
vllm/distributed/kv_transfer/kv_transfer_state.py:90
  _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(role=KVConnectorRole.WORKER)
      │   ← 进程级模块全局变量，每个 worker 进程只创建一次
      ▼
AscendStoreConnector.__init__ (ascend_store_connector.py:106)
  self.connector_worker = KVPoolWorker(...)          # role==WORKER 分支
      ▼
KVPoolWorker.__init__ (pool_worker.py:205-221)
  backend_module = importlib.import_module("...memcache_backend")
  self.m_store = MemcacheBackend(parallel_config, **backend_kwargs)   # ← 创建点
      ▼
MemcacheBackend.__init__ (memcache_backend.py:23-35)
  self.local_rank = get_world_group().local_rank     # line 24：拿到本进程的本地设备号
  ...
  if not self._lazy_init:
      self.store = self._setup_store()               # line 34
```

**初始化的直接证据**（`memcache_backend.py:50-67`）：

```python
def _setup_store(self):
    from memcache_hybrid import DistributedObjectStore
    ...
    if self._is_a2:                                   # A2 设备先做集合通信 warmup
        tmp_tensor = torch.zeros(1, device="npu")
        ...
        torch.distributed.all_gather(..., group=get_world_group().device_group)
    store = DistributedObjectStore()                  # line 64：创建本 rank 的 store
    res = store.init(self.local_rank)                 # line 65：用 local_rank 初始化
    assert res == 0
    return store
```

要点：`local_rank` 来自 `get_world_group().local_rank`（vllm `parallel_state.py:379` 注释明确写着 "local rank used to assign devices"），即**该 worker 进程绑定的 NPU 设备号**；`store.init(local_rank)` 把这个 store 实例锚定到该设备上。之后传输线程启动时还会 `set_device()`（`memcache_backend.py:75-77`，被 `kv_transfer.py:80` 调用）再次确认 `torch.npu.set_device(npu:{local_rank})`。

### 2. MemcacheBackend 对象会创建几次？（DP=2, PP=2, TP=2）

**答案：8 次。** 公式：`DP × TP × PP = 2 × 2 × 2 = 8`，即**每个 worker 进程（每块 NPU）一个**。

| 进程 | 角色 | 创建的对象 | 有无 MemcacheBackend |
|---|---|---|---|
| DP engine 0 的 Scheduler 进程 ×1 | SCHEDULER | `KVPoolScheduler`（只持有 ZMQ `LookupKeyClient`，`pool_scheduler.py:82`） | ❌ 无 |
| DP engine 1 的 Scheduler 进程 ×1 | SCHEDULER | 同上 | ❌ 无 |
| Worker 进程 ×8（每 engine TP×PP=4 个） | WORKER | 各 1 个 `KVPoolWorker` → 各 1 个 `MemcacheBackend` → 各 1 个 `DistributedObjectStore` | ✅ 8 个 |

依据：

- vLLM 工厂注释（`factory.py:67-74`）明确两种角色严格分离：scheduler 连接器在 scheduler 进程，worker 连接器在 worker 进程。`MemcacheBackend` 只在 `KVPoolWorker.__init__` 里构造（`pool_worker.py:218`），`KVPoolScheduler` 里没有任何 backend——**调度侧查 memcache 是走 ZMQ 到 rank0 worker 的 `LookupKeyServer` 间接查询的**（ZMQ 路径按 dp_rank 区分：`ipc://.../lookup_rpc_port_{port}_dp_rank{dp_rank}`，`pool_scheduler.py:693`）。
- `_KV_CONNECTOR_AGENT` 是 worker 进程内的模块级单例（`kv_transfer_state.py:79-94`），所以一个进程无论多少层、多少请求，只有 **1 个** backend。
- 单机 8 卡场景下，8 个 worker 的 `local_rank` 分别为 0~7，各自 `store.init(local_rank)` 并 `register_buffer` 注册**自己那块卡**的 KV 显存。8 个 store 实例连接的是**同一个分布式 memcache 全局池**，靠 key 中的 rank 字段区分数据分片（见问题 3）。

两个细节：

- **lazy_init 不改变对象数量**：DSV4 压缩模型下 8 个 `MemcacheBackend` Python 对象照常在构造时创建，只是 `DistributedObjectStore.init` 推迟到该 rank 第一次 `put` 时（`_ensure_initialized`，`memcache_backend.py:37-48`）；且 `lazy_init` 在 A2 上被禁用（`memcache_backend.py:27`）。
- 多线程共享：每个 worker 内的 `KVCacheStoreSendingThread` / `KVCacheStoreRecvingThread` 共享这同一个 `m_store`，不会再建。

### 3. Key 语义详解：`model@pcp@dcp@tp_rank@pp_rank@group@cache_role@cache_family@chunk_hash`

生成代码在 `config_data.py:61-71`（`PoolKey.to_string`），字段来自 `KeyMetadata`（`config_data.py:21-38`）+ `chunk_hash`。以 **DeepSeek 类模型（MLA，`num_kv_head=1`），TP=8、PP=2、无 PCP/DCP，hash_block_size=128** 为例：

某请求前 256 个 token 切成 2 个 chunk，chunk_hash 分别为 `a1b2...`（token 0-127）和 `c3d4...`（token 128-255，前缀链式哈希）。TP rank5、PP rank0 的 worker 对 chunk 0 生成的 key 是：

```
DeepSeek-V3@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:default@a1b2...
```

逐字段含义：

| 字段 | 本例值 | 语义（为什么必须在 key 里） |
|---|---|---|
| `model_name` | `DeepSeek-V3` | 模型目录名（`pool_worker.py:190`）。多个模型共用一个全局池时隔离 keyspace，防止 A 模型的 KV 被 B 模型命中 |
| `pcp` | `0` | Prefill Context Parallel rank。PCP 下每个 rank 只算上下文的一段，KV 不重叠，必须按 rank 区分 |
| `dcp` | `0` | Decode Context Parallel rank，同理 |
| `head_or_tp_rank` | `0` | **TP 分片标识**。GQA 模型（如 4 个 KV head、TP=2）时 `head_or_tp_rank = tp_rank`，rank0/rank1 各存自己的 head 切片，key 分别为 `...:0` / `...:1`；消费侧 lookup 会把所有 tp 变体展开查询、**全部命中才算命中**（`pool_worker.py:1166-1173`）。本例 MLA 只有 1 个 KV head，`put_step = tp_size // num_kv_head = 8`（`pool_worker.py:152-154`），`head_or_tp_rank = tp_rank // put_step = 0`——8 个 rank 数据完全相同，共用 `:0` 一个 keyspace，put 时按 `keys[tp_rank % 8 :: 8]` 把 chunk 摊派给 8 个 rank 各存 1/8（`kv_transfer.py:343-347`），既去重又负载均衡 |
| `pp_rank` | `0` | PP=2 时 stage0 持有 layer 0-29，stage1 持有 layer 30-59。同一个 chunk_hash 在两个 stage 对应**不同层**的 KV 数据，必须用 pp_rank 区分，各存各的 |
| `group` | `0` | KV cache group id。混合模型（full attention + sliding window + mamba）有多个 block pool，同一 chunk 在不同 group 的物理块不同，分开命名空间 |
| `cache_role` | `kv` | 区分同一个 chunk_hash 下的 **KV 数据**与**状态数据**（`state`：mamba 状态 / DSV4 compressor、indexer 状态），二者哈希相同但内容完全不同（`config_data.py:36` 注释） |
| `cache_family` | `default` | 压缩家族（`config_data.py:121-126`）：DSV4 不同层压缩率不同，ratio=4 的层 family 为 `c4`，存取粒度 = block_size × 4。同一 chunk 以不同压缩率存储时靠 family 区分 |
| `chunk_hash` | `a1b2...` | 来自 vLLM 的 `request.block_hashes`，是对**前 N 个 token 的前缀链式哈希**。这是"跨请求、跨实例前缀复用"的根基：任何请求只要前 128 个 token 相同，chunk 0 的哈希就相同 → key 完全相同 → 全局池直接命中 |

**layerwise 模式**还会追加 `@layer_id:{i}`（`LayerPoolKey.to_string`，`config_data.py:108-118`），因为一个 chunk 的各层是逐层传输、逐层落池的。

一句话概括这个 key 的设计：**`chunk_hash` 回答"数据是什么"（哪段 token 前缀），前面所有字段回答"这是谁的那一份"（哪个模型、哪个并行切分、哪个 cache 组、哪种数据形态、哪种压缩率）——两者相乘，保证全局池里任意两个物理上不同的 KV 片段 key 必不同，逻辑上相同的必相同。**

---

## Q&A：MemcacheBackend 的作用与请求处理全链路

> **原始问题：** vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend.MemcacheBackend 中的方法在 kv cache 的传输过程起到了什么作用？在整个 vllm / vllm-ascend 共同工作的背景下，试着画图说明，从请求进来，到和 memcache 发生关系，请求的处理链路是怎么样的。

（依据代码：`memcache_backend.py`、`backend.py`、`kv_transfer.py`、`pool_worker.py`、`pool_scheduler.py`、`ascend_store_connector.py`、`config_data.py`，以及 memcache 仓库的 Python API 文档。）

### 一、MemcacheBackend 在 KV cache 传输中的作用

`MemcacheBackend` 是整个 AscendStore 传输栈的**最底层"存储 I/O 执行器"**——它是 vllm-ascend 与华为 memcache_hybrid（MemFabric Hybrid）C++ 库 `DistributedObjectStore` 之间的一层薄适配器。它本身不做任何调度、不做 key 生成、不做地址计算，只负责把上层准备好的 `(keys, addrs, sizes)` 三元组真正落到全局 KV 池里。

各方法职责：

| 方法 | 底层调用 | 作用 |
|---|---|---|
| `__init__` / `_setup_store` / `_ensure_initialized` | `DistributedObjectStore().init(local_rank)` | 每个 NPU rank 创建一个 store 实例并初始化；DSV4 压缩模型走 `lazy_init`（第一次 `put` 时才初始化，避免初始化时序问题）；A2 设备先做一次 `all_gather` warmup |
| `set_device()` | `torch.npu.set_device` | 传输线程启动时绑定本 rank 的 NPU 设备（`kv_transfer.py:80`） |
| `register_buffer(ptrs, sizes)` | `store.register_buffer` | **把 NPU 上 KV cache 显存区间注册给 memcache**，使后续 put/get 可以直接对设备显存做 DMA（零拷贝），无需经过 host 中转。地址来自 `KVPoolWorker.register_kv_caches` 计算的各 layer KV tensor 的 base addr + region 长度（`pool_worker.py:451`） |
| `exists(keys)` | `batch_is_exist` | **纯元数据查询**：批量检查 chunk key 是否已存在于全局池。用于两个地方：① scheduler 侧的前缀命中查询（lookup）；② 发送线程 put 之前去重（已存在的块不再重复存） |
| `put(keys, addrs, sizes)` | `batch_put_from_layers(..., COPY_L2G)` | **存**：把本地（Local）NPU 显存中各 layer 的 KV block 数据按 key 写入全局（Global）memcache 池。`addrs` 是由 `ChunkedTokenDatabase.prepare_value` 算出的每层 KV tensor 内 `base + block_id * stride` 的显存地址，一个 key 对应"所有层在该 chunk 上的一组地址" |
| `get(keys, addrs, sizes)` | `batch_get_into_layers(..., COPY_G2L)` | **取**：从全局池按 key 读出 KV 数据，直接 DMA 写入本地已分配好的 KV cache block 地址里（prefill 跳算 / P 传 D 的接收侧） |

`MmcDirect` 枚举说明拷贝方向：`L2G`（本地→全局，存）、`G2L`（全局→本地，取）、`G2H`/`H2G`（全局↔主机内存）。

一句话总结：**上层（KVTransferThread / KVPoolWorker）负责"算什么 key、从哪块显存、传多少"，MemcacheBackend 只负责"存在不在、写进去、读出来"这三件事，是 NPU 显存与分布式 KV 池之间的数据搬运闸门。**

### 二、从请求进来到与 memcache 发生关系的完整链路

```
┌──────────────────────────── vLLM (EngineCore 进程) ────────────────────────────┐
│                                                                                │
│  HTTP 请求 ──▶ API Server ──▶ EngineCore                                       │
│                                  │                                             │
│                                  ▼                                             │
│                     ┌── Ascend Scheduler (scheduler 侧) ──┐                    │
│                     │ connector.get_num_new_matched_tokens│                    │
│                     │  = KVPoolScheduler                  │                    │
│                     │    .get_num_new_matched_tokens()    │                    │
│                     │      │ block_hashes 按 granularity  │                    │
│                     │      │ 切 chunk                     │                    │
│                     │      ▼                              │                    │
│                     │  LookupKeyClient ── ZMQ REQ ──┐     │                    │
│                     └───────────────────────────────┼─────┘                    │
└─────────────────────────────────────────────────────┼──────────────────────────┘
                                                      ▼
┌──────────────────── Worker 进程 rank0 ── ZMQ REP ──────────────────────────────┐
│  LookupKeyServer (ascend_store_connector.py:272)                               │
│      └─▶ KVPoolWorker.lookup_scheduler()  (pool_worker.py:1120)                │
│            └─▶ ChunkedTokenDatabase.process_tokens() 生成 PoolKey              │
│                 "model@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@group:0@...@hash" │
│            └─▶ MemcacheBackend.exists(keys) ══▶ batch_is_exist ══▶ ┌────────┐  │
│                                                                     │memcache│  │
│  返回命中 token 数 ◀═══════════════════════════════════════════════│ 全局池 │  │
└─────────────────────────────────────────────────────────────────────┴────────┴──┘
```

命中数回到 scheduler 后：

```
Scheduler: 为命中部分分配本地 KV blocks
   └─▶ update_state_after_alloc()  → LoadSpec.can_load = True
   └─▶ build_connector_meta()      → 生成 ReqMeta (keys/block_ids/load_spec)
        随 SchedulerOutput 发给 Worker
```

Worker 侧执行（`kv_connector_model_runner_mixin` 在 forward 前后插入钩子）：

```
                        Worker 进程 (每个 NPU rank)
 ┌──────────────────────────────────────────────────────────────────────┐
 │ [初始化时] register_kv_caches(kv_caches)                              │
 │   └─▶ 计算每层 KV tensor 的 base_addr/block_len/block_stride          │
 │   └─▶ MemcacheBackend.register_buffer(ptrs, sizes) ──▶ 注册显存给池   │
 │                                                                      │
 │ [forward 前] start_load_kv(metadata)                                  │
 │   └─▶ KVPoolWorker.start_load_kv (pool_worker.py:522)                 │
 │        ├─ 对每个 can_load 的请求:                                     │
 │        │   process_tokens_with_block_ids → (start,end,key,block_id)  │
 │        │   prepare_value → addrs = base + block_id*stride (每层)      │
 │        ├─ 同步模式: MemcacheBackend.get(keys,addrs,sizes)  ─┐         │
 │        ├─ load_async: 丢给 KVCacheStoreRecvingThread 异步 get┤ G2L    │
 │        └─ layerwise:  retrieve_layer 生成器逐层 get          │ DMA     │
 │                                                            ▼         │
 │                            ┌───────── memcache 全局 KV 池 ─────────┐ │
 │                            │  KV 数据直接写入本地 NPU KV block       │ │
 │                            └───────────────────────────────────────┘ │
 │   ▶ 命中前缀无需重算，模型只对剩余 token 做 prefill/decode             │
 │                                                                      │
 │ [forward 中]  (layerwise 模式) wait_for_layer_load / save_kv_layer    │
 │                                                                      │
 │ [forward 后] wait_for_save()                                          │
 │   └─▶ KVPoolWorker.wait_for_save (pool_worker.py:703)                 │
 │        └─▶ KVCacheStoreSendingThread (kv_transfer.py:299)             │
 │             1. process_tokens_with_block_ids → keys                   │
 │             2. TP 分片: keys[tp_rank % put_step :: put_step]          │
 │             3. MemcacheBackend.exists(keys) ──▶ 去重，只存缺失块      │
 │             4. prepare_value → addrs/sizes                            │
 │             5. current_event.synchronize()  ← 等 NPU 算完该 block     │
 │             6. MemcacheBackend.put(keys,addrs,sizes) ══ L2G ══▶ 池   │
 │             7. queue.join() 确保写完才向 scheduler 报 finished        │
 │                                                                      │
 │ get_finished() ─▶ scheduler 收到 done_sending 后才释放 GPU blocks     │
 │ (request_finished 里 delay_free_blocks 防止块被提前复用)              │
 └──────────────────────────────────────────────────────────────────────┘
```

### 关键点提炼

1. **三次与 memcache 交互，对应三个方法**：
   - **调度前**：`exists`（经 ZMQ lookup 通道）→ 决定"这个请求有多少前缀可以直接加载、少算多少 token"；
   - **forward 前**：`get`（G2L）→ 把命中前缀的 KV 直接灌进本地显存块，实现 prefill 跳算 / PD 分离中 D 节点接收；
   - **forward 后**：`put`（L2G）→ 把新算出的 KV 写回全局池，供后续请求或 decode 实例复用（写前先 `exists` 去重）。

2. **key 的语义**：`model@pcp@dcp@tp_rank@pp_rank@group@cache_role@cache_family@chunk_hash`（`config_data.py:61`），按 `hash_block_size` 对齐的 token chunk 哈希，因此跨实例、跨请求的同前缀天然命中同一个 key——这是全局前缀缓存和 PD 分离能工作的基础。

3. **地址的语义**：put/get 不拷贝数据到 python 对象，而是传**显存裸地址列表**（每层一个 `base + block_id × stride`），配合 `register_buffer` 实现 NPU 显存 ↔ 全局池的直接 DMA，Python 线程只提交描述符。

4. **同步点**：`put` 前的 `current_event.synchronize()`（`kv_transfer.py:428`）保证 NPU 已写完该 block 的 KV 才让 memcache 去读；`wait_for_save` 里的 `request_queue.join()` 保证 scheduler 释放 block 前数据已离卡。

5. **P/D 分离角色**：`kv_role=kv_producer` 的 P 实例只走 ③（put），`kv_consumer` 的 D 实例走 ①②（exists/get），`kv_both` 或单机场景三者都走——MemcacheBackend 对这三种角色是同一份代码，区别只在上层调用哪些方法。

---

## Python 客户端接口：DistributedObjectStore、layers、KeyInfo、GVA 与监控

### 问题 1：DistributedObjectStore 类主要功能是怎么样的？

`DistributedObjectStore` 本身**不是 Python 实现的**，它是 C++ 类 `MmcacheStore` 通过 pybind11 暴露的绑定（`pymmc.cpp:617`），是**客户端侧的分布式对象存储入口**。`memcache_hybrid/__init__.py:49` 只是从 `_pymmc` 导入。

按功能分组（绑定代码都在 `pymmc.cpp:617-908`）：

```
                    DistributedObjectStore (C++ MmcacheStore)
   ┌──────────────────────────────────────────────────────────────────┐
   │ 生命周期                                                          │
   │   setup(config) / init(device_id, init_bm) / close()              │
   ├──────────────────────────────────────────────────────────────────┤
   │ ① 简单读写（bytes，内部拷贝）                                       │
   │   put(key, buf) / put_batch          get(key) / get_batch         │
   ├──────────────────────────────────────────────────────────────────┤
   │ ② 零拷贝读写（预分配 buffer，整块数据）                              │
   │   put_from(key, ptr, size)         get_into(key, ptr, size)       │
   │   batch_put_from                   batch_get_into                 │
   ├──────────────────────────────────────────────────────────────────┤
   │ ③ 分层读写（一个 key = N 层数据，见问题2）                           │
   │   put_from_layers / batch_put_from_layers                         │
   │   get_into_layers / batch_get_into_layers                         │
   ├──────────────────────────────────────────────────────────────────┤
   │ ④ GVA 直接读写（绕过 key 查找，见问题4）                             │
   │   batch_alloc(keys, sizes, media) → 返回 GVA                      │
   │   batch_copy / batch_copy_layers（GVA ↔ 本地 buffer 直接拷贝）      │
   ├──────────────────────────────────────────────────────────────────┤
   │ ⑤ 元信息/管理                                                      │
   │   is_exist / batch_is_exist        get_key_info / batch_get_key_info│
   │   remove / remove_batch / remove_all                              │
   │   register_buffer / unregister_buffer（注册 RDMA 直通内存）          │
   │   get_local_service_id                                            │
   └──────────────────────────────────────────────────────────────────┘
```

一句话：它把"key → 分布式内存池（HBM/DRAM/SSD）中的 blob"的读写、查询、删除全部封装起来，数据面通过 `direct` 参数指定拷贝方向（`H2G/L2G/G2H/G2L`，即 Host DRAM ↔ 全局空间、NPU HBM ↔ 全局空间）。

### 问题 2：DistributedObjectStore 类的 put_from_layers，这里的 layers 是什么概念？

**layers = 一个逻辑对象（key）由 N 个不连续的内存片段按顺序拼接而成，每个片段叫一层**。

- `buffer_ptrs`: N 个源缓冲区指针（每个指针对应一层）
- `sizes`: 每层的大小（**各层可以不等长**）
- 存储侧把 N 层按顺序拼接成**一个连续的 blob** 存入全局内存池；读时 `get_into_layers` 再按同样的切分写回 N 个目标缓冲区

```
应用侧内存（分散）                     全局内存池（一个连续 blob）
┌─────────┐ layer0 (size0)          ┌──────────────────────────────┐
├─────────┤ layer1 (size1)   put    │ layer0│layer1│layer2│layer3  │
├─────────┤ layer2 (size2)  ────>   └──────────────────────────────┘
├─────────┤ layer3 (size3)              gva 起始地址，按 size 累加
   N 个 data_ptr()，地址不连续
```

**典型场景就是 vLLM 的 KV Cache**（见官方示例 `example/python/test_mmc_layers.py:48-74`）：

```python
# KV cache 张量形状: (layer_number=10, block_number=10, block_size=1024)
# 第 0 维是 transformer 层号 —— 这就是 "layers" 的来源
cpu_tensor = torch.empty((layer_number, block_number, block_size))
# 一个 block 的 KV = 所有层在该 block 上的切片，共 layer_number 个片段
block = [cpu_tensor[i][block_id] for i in range(layer_number)]

store.put_from_layers("2d",
                      [layer.data_ptr() for layer in block],   # 10 个层的指针
                      [block_size] * layer_number,             # 每层大小
                      COPY_L2G)
```

即：**一个 key = 一个 block 的 KV cache，layers = 各 transformer 层在该 block 上的切片**。这样一次调用就把分散在 N 处的层数据聚合成一个对象存取，不用先自己做 concat 拷贝。`batch_copy_layers` 的文档（`memcache_python_api.md:791`）也印证了拼接语义："每层大小会从对应起始 GVA 开始按顺序累加"。

### 问题 3：DistributedObjectStore 类的 get_key_info 返回的 KeyInfo 对象，每个字段的含义

`KeyInfo` 定义在 `src/memcache/include/cpp/mmcache.h:25-104`，Python 侧暴露 4 个查询方法（`pymmc.cpp:29-35`）。它描述的是**一个 key 的所有 blob 副本的分布信息**：

```
key ──> KeyInfo {
            size_:     uint64        数据总大小（<=0 表示 key 不存在或无效）
            blobNum_:  uint32        blob 副本数量
            loc_:  [int, ...]        每个 blob 所在位置 ── loc_list()
            type_: [int, ...]        每个 blob 的介质类型 ── type_list()
            gva_:  [uint64, ...]     每个 blob 的起始 GVA ── gva_list()
        }
```

逐字段解释：

| 字段 / 方法 | 类型 | 含义 |
|---|---|---|
| `size()` → `size_` | uint64 | 该 key 数据的**总字节数**（一个 blob 的完整数据大小，不是各副本之和）。注释明确：`size <= 0` 表示 key 不存在或无效 |
| `blobNum_`（未直接暴露，见 `__str__`） | uint32 | 该 key 名下的 **blob 副本个数**。多副本（`ReplicateConfig.replicaNum`）或多级介质（回温后 DRAM+SSD 各一份）时会 >1 |
| `loc_list()` → `loc_` | List[int] | 每个 blob 的 **location，即所在节点的 rank id**（对应 C++ 里 `MmcMemBlob::rank_`）。用来知道数据在哪个节点上，便于本地化读取 |
| `type_list()` → `type_` | List[int] | 每个 blob 的 **介质类型**：`0 = MEDIA_HBM`，`1 = MEDIA_DRAM`，`2 = MEDIA_SSD`（对应 `mmc_types.h:78` 的枚举值） |
| `gva_list()` → `gva_` | List[int] | 每个 blob 的**起始 GVA（全局虚拟地址）**，是后续 `batch_copy` 直接读写的地址 |

三个列表是**平行数组**，按下标一一对应：

```
          blob#0        blob#1
loc_:   [ rank2   ,     rank2    ]
type_:  [ 1(DRAM) ,     2(SSD)   ]
gva_:   [ 0x7f3a... ,   0x81c0...]
           └── 同一份数据在两个介质上的两个副本 ──┘
```

**关于 `flag` 参数**（`memcache_python_api.md:689`）：`flag=0` 为普通查询；`flag=1` 用于单 blob 场景下为后续**基于 GVA 的直读流程**做准备（即拿到 `gva_list()` 后用 `batch_copy` 绕过 key 查找直接读写）。

### 问题 4：doc/memcache_python_api.md 文档中"GVA 相关接口与常量"中的 GVA 是什么意思？

**GVA = Global Virtual Address（全局虚拟地址）**，是分布式内存池统一编址的地址空间。

普通 put/get 流程是"key → 查元数据服务 → 找到 blob → 拷贝数据"；而 GVA 接口把"寻址"这一步开放给上层，拿到地址后**直接对全局内存做读写，跳过 key 查找**：

```
普通流程：   put(key, data)  ──> MetaService 查 key ──> 定位 blob ──> 拷贝
GVA 流程：   batch_alloc(keys, sizes) ──> 返回每个 key 的起始 GVA
             batch_copy(gva_ptrs, buf_ptrs, sizes, H2G)  ──> 按地址直接写
             get_key_info(key, flag=1) ──> gva_list()
             batch_copy(gva_ptrs, buf_ptrs, sizes, G2L)  ──> 按地址直接读
                    （文档 memcache_python_api.md:799 给出的典型流程）
```

每个 blob 在分配时就有自己的 `gva_` 字段（`mmc_mem_blob.h:198`），集群内任何节点拿到这个地址都能通过 RDMA/URMA 等协议直接访问这块内存（配合 `register_buffer` 注册本地缓冲区实现零拷贝）。另外注意仓库最近的约束（git 提交 `ebdb48a7`）：**使用 URMA 协议通讯时须使用 56 bit GVA**。

### 问题 5：memcache 有暴露监控能力或者接口吗？比如查询各级存储剩余容量的功能

**有，而且很完整。** MetaService 启动一个 HTTP 服务（地址由 `ock.mmc.meta_service.metrics_url` 配置，默认 `http://127.0.0.1:8000`），接口文档在 `doc/memcache_restful_api.md`：

#### 容量查询

```
GET /api/v1/capacity/usage                # 按介质汇总：npu(=HBM) / cpu(=DRAM)
     → { "npu": {total_bytes, used_bytes, free_bytes},
          "cpu": {total_bytes, used_bytes, free_bytes} }

GET /api/v1/capacity/segment_remaining    # 按 segment 明细（每节点每介质）
     → { "segments": [ {segment_name: "rank-0-hbm",
                         total_bytes, used_bytes, remaining_bytes}, ... ] }

GET /query_segment?segment=rank-0-hbm     # 单个 segment 详情
     → { segment, medium, total_bytes, used_bytes,
          remaining_bytes, remaining_ratio }

GET /get_all_segments                     # 列出所有 segment_id（rank-<id>-<介质>）
```

> segment 是容量管理的最小单位：一个节点的一种介质 = 一个 segment，命名 `rank-<rank>-<medium>`。

#### 指标监控（Prometheus 格式）

```
GET /metrics            # Prometheus 文本格式，可直接被 Prometheus 抓取
                        # 包含：alloc/get/remove 等请求数、成功、失败、not_found 计数
                        #       evict 淘汰计数、rewarm 回温计数、SSD 淘汰删除计数等
GET /api/v1/analysis/alloc_free_latency   # alloc/free 延迟统计（ptracer）
```

这些计数器来自 `MmcMetaMetricManager`（淘汰/回温流程里的 `IncrementEvictCounter`、`IncrementRewarmCounter` 等就是喂给它的），另外配置项 `ock.mmc.meta_service.metrics_report_interval_seconds`（默认 30s）控制周期性指标汇总打印。

#### 运维/调试接口

```
GET /health                        # 健康检查
GET /query_key?key=...             # 查单个 key 的元信息（介质、gva、状态、租约）
GET /batch_query_keys?keys=...     # 批量查 key
GET /get_all_keys                  # 列出全部 key
GET /api/v1/segments/status?segment=...   # segment 状态
POST /api/v1/drain_jobs            # 数据排空任务（配合 segment 下线）
```

所以做容量监控最简单的做法是定时抓 `/api/v1/capacity/usage` 看 HBM/DRAM 整体水位，再抓 `/metrics` 接 Prometheus 看请求和淘汰/回温速率。

---

## Q&A：MemcacheBackend 加载配置、head_or_tp_rank 与 chunk_hash 详解

> **原始问题：**
> 1. `KVPoolWorker.__init__ (pool_worker.py:205-221) backend_module = importlib.import_module("...memcache_backend"); self.m_store = MemcacheBackend(parallel_config, **backend_kwargs)` 这个没有找到原来的代码，看看呢？启动参数怎样配会加载 MemcacheBackend？
> 2. `head_or_tp_rank` 的解释（TP 分片标识、GQA/MLA 差异、put_step 摊派、lookup 展开）需要再详细解释一下。
> 3. `chunk_hash` 是来自 vLLM 的 `request.block_hashes`，是对前 N 个 token 的前缀链式哈希。请详细解释一下。

### 1. `MemcacheBackend` 创建代码的原文 & 启动参数怎么配

#### 原文代码

上次回答把 205-221 行压缩成了伪代码。`pool_worker.py:205-221` 的**原文**是：

```python
backend = backend_map.get(self.backend.lower())          # line 205
assert backend is not None
backend_path = backend.get("path")
backend_name = backend.get("name")
assert backend_path is not None and backend_name is not None
backend_module = importlib.import_module(backend_path)   # line 210
real_backend = getattr(backend_module, backend_name)     # line 211 → MemcacheBackend 类

backend_kwargs = {}
if self.backend.lower() in {"mooncake", "memcache"}:
    # DSV4 exposes compress_ratios; only use lazy store init for this
    # compressed-model path.
    backend_kwargs["lazy_init"] = self.use_compress
self.m_store = real_backend(                             # line 218 → 构造 MemcacheBackend
    parallel_config,
    **backend_kwargs,
)
```

其中 `backend_map` 就在同文件 `pool_worker.py:54-67`：

```python
backend_map = {
    "mooncake": {"name": "MooncakeBackend",  "path": "...backend.mooncake_backend"},
    "memcache": {"name": "MemcacheBackend",  "path": "...backend.memcache_backend"},
    "yuanrong": {"name": "YuanrongBackend",  "path": "...backend.yuanrong_backend"},
}
```

而 `self.backend` 来自 `pool_worker.py:113`：

```python
self.backend = vllm_config.kv_transfer_config.kv_connector_extra_config.get("backend", "mooncake")
```

#### 启动参数配置

两级配置缺一不可：

```bash
vllm serve /path/to/DeepSeek-V3 \
  --kv-transfer-config '{
      "kv_connector": "AscendStoreConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
          "backend": "memcache",          # ← 决定加载 MemcacheBackend（缺省是 "mooncake"!）
          "use_layerwise": false,
          "load_async": true,
          "lookup_rpc_port": 0
      }
  }'
```

链路是：

1. `"kv_connector": "AscendStoreConnector"` → `KVConnectorFactory` 注册表查找（注册点在 `vllm_ascend/distributed/kv_transfer/__init__.py:45-49`，`MooncakeConnectorStoreV1` 是同一类的旧名字）→ 每个 worker 进程创建 `AscendStoreConnector` → `KVPoolWorker`；
2. `"backend": "memcache"` → `KVPoolWorker.__init__` 查 `backend_map["memcache"]` → `importlib.import_module("...memcache_backend")` + `getattr(module, "MemcacheBackend")` → 构造对象。

另外两个隐含条件：环境里必须装有 `memcache_hybrid` 包（否则 `_setup_store` 抛 `ImportError` 并提示去 gitee.com/ascend/memfabric_hybrid 安装）；`lazy_init` 仅当模型带 `compress_ratios`（DSV4 压缩路径）时为 True。

### 2. `head_or_tp_rank` / `put_step` 详细解释

#### 涉及的全部代码

**计算规则**（`pool_worker.py:147-157`）：

```python
if self.use_mla:
    self.num_kv_head = 1
else:
    self.num_kv_head = model_config.get_total_num_kv_heads()

if self.num_kv_head < self.tp_size:
    self.put_step = self.tp_size // self.num_kv_head
    self.head_or_tp_rank = self.tp_rank // self.put_step
else:
    self.head_or_tp_rank = self.tp_rank
    self.put_step = 1
```

`head_or_tp_rank` 随后被写进 key 的元数据（`pool_worker.py:187`），成为 key 字符串里的 `@head_or_tp_rank:{x}` 字段。

**chunk 摊派**（发送线程 `kv_transfer.py:338-347`）：

```python
if (not self.dcp_size > 1
        and not req_meta.disable_tp_key_sharding
        and not self.group_uses_align_state[group_id]):
    starts = starts[self.tp_rank % self.put_step :: self.put_step]
    ends   = ends  [self.tp_rank % self.put_step :: self.put_step]
    keys   = keys  [self.tp_rank % self.put_step :: self.put_step]
    ...
```

**lookup 展开**（`pool_worker.py:1166-1173`，消费侧查所有分片）：

```python
multi_tp_keys = keys[:]                                  # 原始 key 都是 :0
group_tp_size = self.get_group_tp_size(group_id)         # = min(tp_size, num_kv_head)
for i in range(1, group_tp_size):
    for item in keys:
        new_str = item.replace("@head_or_tp_rank:0", f"@head_or_tp_rank:{i}", 1)
        multi_tp_keys.append(new_str)
```

#### 设计动机：区分"数据不同"和"数据重复"两种 TP 场景

背景：每个 TP rank 的 NPU 上都有自己的一份 KV cache 显存。但**这份显存里的内容是不是和其他 rank 一样，取决于注意力结构**：

- **GQA/MHA**：KV head 被切分给各 TP rank，rank0 存 head 0-1，rank1 存 head 2-3——**各 rank 数据互不相同**；
- **MLA**（DeepSeek）：KV 被压缩成一个共享的 latent 向量，所有 rank 保存**完全相同的副本**。

这个区别直接决定 key 该怎么命名、存几份。分三种情形：

**情形 A：GQA，`num_kv_head=4，TP=2`**（走 `else` 分支）

- `put_step = 1`，`head_or_tp_rank = tp_rank` → rank0 用 `:0`，rank1 用 `:1`；
- rank0 把自己的 head 0-1 切片 put 到 `...@head_or_tp_rank:0@...@hash`，rank1 把 head 2-3 put 到 `...:1@...@hash`。两个 key 指向**不同的物理数据**，各存一份，总量无冗余；
- 消费侧（D 实例）lookup 时把 `:0` 和 `:1` 两个变体都展开查（`group_tp_size = min(2, 4) = 2`），**两个分片都存在才算该 chunk 命中**；get 时 D 侧 rank0 取 `:0`、rank1 取 `:1`，各拿回自己负责的那几个 head。

**情形 B：MLA，`num_kv_head=1，TP=8`**（走 `if` 分支）

- `put_step = 8 // 1 = 8`，`head_or_tp_rank = tp_rank // 8 = 0`（所有 rank 都是 0）；
- 因为 8 个 rank 的 KV 内容**一模一样**，如果按情形 A 各用各的 rank 号做 key，同一份数据要在全局池里存 8 份，浪费 8 倍空间。所以统一只用 `:0` 一个 keyspace；
- 但新问题来了：8 个 rank 都往同一个 keyspace put 相同内容，即使 `exists()` 去重，也存在并发竞争，而且每个 rank 都要发起完整的传输，浪费带宽。于是发送线程做 **chunk 级摊派**：`keys[tp_rank % 8 :: 8]`——
  - rank0 只负责 chunk 0, 8, 16, ...
  - rank1 只负责 chunk 1, 9, 17, ...
  - …
  - 8 个 rank 合起来恰好覆盖全部 chunk 一遍：**空间上只存一份，传输工作量每 rank 只有 1/8**，这就是"既去重又负载均衡"；
- 消费侧 `group_tp_size = min(8, 1) = 1`，lookup 只查 `:0` 一个变体即可，D 侧任意 rank 取回的就是完整数据。

**情形 C：中间态，`num_kv_head=2，TP=8`**

- `put_step = 8 // 2 = 4`，`head_or_tp_rank = tp_rank // 4` → rank 0-3 是 `:0`，rank 4-7 是 `:1`；
- 语义：head 0 的数据在 rank 0-3 上重复了 4 份，head 1 在 rank 4-7 上重复 4 份。keyspace 有 `:0`/`:1` 两个（对应两种不同数据），每个 keyspace 内部 4 个 rank 再按 `tp_rank % 4` 摊派 chunk，各存 1/4；
- 消费侧 `group_tp_size = min(8, 2) = 2`，展开 `:0`、`:1` 都命中才算命中。

**一句话总结**：`head_or_tp_rank` 标记的是"这是第几份**不同的数据**"（不同的 KV head 切片），相同副本共用一个编号；`put_step` 标记"持有相同副本的 rank 数量"，用于把 chunk 在副本之间摊派，避免重复传输。两个例外：DCP>1 时或 mamba align 模式的 group 不做这套摊派，直接用 `tp_rank`。

### 3. `chunk_hash`（`request.block_hashes` 前缀链式哈希）详细解释

#### vLLM 侧：哈希怎么算出来的

**入口**：每个 `Request` 维护一个 `block_hashes: list[BlockHash]`（`vllm/v1/request.py:199`），每当 token 增加（prefill 接收 prompt、decode 每生成 token）就调 `update_block_hashes()`（`request.py:257-260`）增量补齐新满块的哈希。

**核心算法**（`kv_cache_utils.py:596-623`）：

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH          # 第一个块的"父哈希"是固定随机种子
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
```

**驱动循环**（`kv_cache_utils.py:705-746`）：按 `hash_block_size` 一个块一个块地算，`prev_block_hash_value` 不断滚动传入下一块；**只哈希完整块**（`# We only hash full blocks`，line 728）。

#### 数值例子

`hash_block_size = 128`，prompt 共 384 个 token（t₀~t₃₈₃），恰好 3 块：

```
h₀ = H( NONE_HASH, (t₀…t₁₂₇),     extra_keys )
h₁ = H( h₀,        (t₁₂₈…t₂₅₅),   extra_keys )
h₂ = H( h₁,        (t₂₅₆…t₃₈₃),   extra_keys )
```

所谓"**前缀链式**"：`h₁` 的输入里包含 `h₀`，而 `h₀` 又承诺了 t₀~t₁₂₇，所以 **`h₁` 实际上是对整个前缀 t₀~t₂₅₅ 的承诺（commitment）**，而不只是第 128-255 这一段。这带来两个关键性质：

1. **相同前缀 ⇒ 相同哈希**：另一个请求只要前 256 个 token 完全相同，它算出的 `h₁` 逐字节相同——不管第 257 个 token 之后是什么。这就是跨请求、跨实例前缀复用的基础：key 里带着 `h₁`，谁都能命中；
2. **不同前缀 ⇒ 哈希必不同**：即使两个请求第 128-255 段碰巧一样，但开头不同，它们的 `h₁` 也不同（父哈希不同）。这防止了"中段相同但前缀不同"的错误命中——KV cache 是前缀敏感的，第 200 个 token 的 KV 只有在前面 200 个 token 都一样时才有意义。

`extra_keys`（`kv_cache_utils.py:560-593`）把 LoRA 名、多模态输入哈希、`cache_salt`、prompt embedding 等也混入，保证"同 token 但不同 LoRA/图片"不互相污染。

#### vllm-ascend 侧：粒度对齐

vLLM 按 `hash_block_size`（如 16）算哈希，而 memcache 池希望按更大的 `group_block_size`（如 128）一个 chunk 一个 key。对齐代码在 `config_data.py:522-545`：

```python
def get_block_hashes(block_hashes, group_block_size, hash_block_size):
    scale_factor = group_block_size // hash_block_size          # 如 128/16 = 8
    return [_rehash_block_hash_group(block_hashes[idx : idx+scale_factor])
            for idx in range(0, ..., scale_factor)]

def _rehash_block_hash_group(block_hashes):
    hasher = hashlib.sha256()
    hasher.update(_GROUPED_BLOCK_HASH_DOMAIN)
    hasher.update(len(block_hashes).to_bytes(...))
    for block_hash in block_hashes:
        hasher.update(...)                                       # 把 8 个小块哈希一起 sha256
    return BlockHash(hasher.digest())
```

即 **chunk_hash_k = sha256(h_{8k}, h_{8k+1}, …, h_{8k+7})**。由于输入的 8 个小哈希本身都是前缀承诺，组合出来的大 chunk 哈希依然是对"前 (k+1)×128 个 token"的承诺，链式性质无损保留。

最后 `ChunkedTokenDatabase.process_tokens`（`config_data.py:409-430`）把每个大 chunk 的哈希 hex 化，拼进 `PoolKey`，成为 key 字符串的最后一个字段 `@a1b2...`，交给 `MemcacheBackend.exists/put/get` 使用。

**串起来看**：哈希是纯内容寻址、确定性计算、不需要任何中心协调——P 实例（producer）和 D 实例（consumer）各自独立地对同一个 prompt 算出**完全相同**的 key 字符串，这就是 PD 分离场景下跨机器传输 KV 能对上号的根本原因。同时"只哈希完整块"也解释了调度侧 `discard_partial_chunks` 和 `_floor_to_cache_transfer_granularity` 的存在：尾部不足一个 chunk 的 token 没有哈希、没有 key，自然不参与池化。

---

## KV cache 张量布局 与 按介质精确删除 key

### 问题 1：KV cache 张量形状 (layer_number=10, block_number=10, block_size=1024) 请再好好解释一下？

#### 三个维度分别是什么

这是 vLLM PagedAttention 管理 KV cache 的经典内存布局（示例里做了简化，用 `uint8` 表示）：

```
cpu_tensor = torch.empty((10, 10, 1024), dtype=uint8)
                           │   │    │
                           │   │    └─ block_size=1024：一个 block 在一层里占 1024 字节
                           │   │       （真实场景 = block内token数 × kv_head数 × head_dim × dtype字节数）
                           │   │
                           │   └────── block_number=10：PagedAttention 的"页"，
                           │            KV cache 按 block（页）切块管理，一个 block
                           │            装固定数量 token 的 K/V
                           │
                           └────────── layer_number=10：Transformer 的层数，
                                       每层 attention 都有自己独立的 K/V cache
```

把它想象成一栋楼：

```
                 block0   block1   block2  ...  block9
  layer0      [  1024B ][  1024B ][  1024B ]...[  1024B ]   ← 第0层的KV cache
  layer1      [  1024B ][  1024B ][  1024B ]...[  1024B ]   ← 第1层的KV cache
   ...
  layer9      [  1024B ][  1024B ][  1024B ]...[  1024B ]   ← 第9层的KV cache

  整个张量 = 10层 × 10块 × 1024B = 100KB，内存上按 layer 优先连续存放
```

#### 关键问题：一个 block 的完整 KV 在内存里是**不连续**的

推理时要存取的是一个 **block**（比如某个请求的一段 token 的 KV），而 attention 是逐层计算的——**一个 block 的完整数据 = 所有 10 层在这个 block 上的切片**：

```python
# 示例 test_mmc_layers.py:54-56 干的事：
block = [cpu_tensor[i][block_id] for i in range(layer_number)]
#       └── 第i层、第block_id块、1024字节的切片，共10个
```

```
cpu_tensor 内存布局（layer 优先连续）：

[layer0: b0|b1|...|b9][layer1: b0|b1|...|b9]...[layer9: b0|b1|...|b9]
          ↑                    ↑                        ↑
        b0@L0                b0@L1                    b0@L9
        └────────  block0 的完整 KV = 这 10 个分散的切片  ────────┘
                   地址间隔 = 10×1024B，互不相邻
```

#### put_from_layers 解决的就是这个"分散"问题

```
没有 layers 接口时（笨办法）：
   先在本地 concat：malloc 10×1024B 连续buffer → 拷10次 → put(key, buffer)
   （多一次完整的数据拷贝，10KB变20KB流量）

用 put_from_layers（示例的做法）：
   put_from_layers("2d",
       [b0@L0地址, b0@L1地址, ..., b0@L9地址],   ← 10个指针直接给它
       [1024]*10)                                ← 每层大小
        │
        ▼  存储侧按顺序拼接，全局池里是一个连续 blob
   key="2d" → blob{ [b0@L0|b0@L1|...|b0@L9] 共10240B }
```

读的时候 `get_into_layers` 做逆操作：把连续 blob 按同样的 10 段切分，直接 scatter 回 NPU 张量各层的对应位置。示例里 `test_equidistant` 就是验证：把 block0 写入、读回到 block1 的位置，然后断言 `tensor[0][0] == tensor[0][1]`（`test_mmc_layers.py:83`）。

一句话总结：**"layers" 的 layer 就是 Transformer 的层**；这个 API 是为 KV cache"层优先存储、块优先访问"的布局矛盾量身定做的零拼接读写通道。

### 问题 2：假如想精确地删掉 memcache 上存储的某个介质上的某个 key，能做到吗？

**对外接口做不到；内部机制支持，但没有暴露。**

#### 对外接口：只能删整个 key

```
Python:  store.remove(key)                 → 删除该 key 在所有介质上的所有 blob
C:       mmcc_remove(key, flags)           → flags 是 reserved（保留未用，mmc_client.h:111）
REST:    DELETE /key?key=...               → 同上，整 key 删除
```

删除路径最终走到 `MmcMetaManager::Remove`（`mmc_meta_manager.cpp:545`）：把 key 从容器摘除后 `PushRemoveList(key, objMeta)` ——**不传 filter**，即默认释放全部 blob（HBM/DRAM/SSD 上的副本一起删）。

#### 内部机制：按介质过滤删除是存在的

底层的 `FreeBlobs` 本身就支持 `MmcBlobFilter{rank, mediaType, state}` 三维过滤（`mmc_mem_obj_meta.cpp:61`），而且系统内部已经在用：

```
淘汰路径 (EvictCallBackFunction, mmc_meta_manager.cpp:956):
    filter = {rank=任意, media=srcMediaType, state=任意}
    → 注释原文："淘汰时仅释放 srcMediaType 的 blob，而非全部"

回温回滚 (RewarmBlob, mmc_meta_manager.cpp:1042):
    filter = {rank=dstRank, media=dstMediaType, state=任意}
    → 只释放刚分配的那份失败副本，保留低层原数据
```

```
                ┌─────────────────────────────┐
   remove(key)  │  filter = null → 全删        │  ← 唯一暴露给用户的入口
                └─────────────────────────────┘
                ┌─────────────────────────────┐
   内部能力      │  filter = {rank, media, state}│  ← 精确删某介质副本，
   (未暴露)      │  → 只删匹配的 blob            │    只在淘汰/回滚路径使用
                └─────────────────────────────┘
```

#### 实际可行的替代办法

| 需求 | 可行做法 |
|---|---|
| 删掉 key 在某个介质上的副本 | **没有直接 API**。只能 `remove(key)` 整删后按需重新 put（会落到高层介质） |
| 让某 key 从高层介质"降级"而不是删除 | 靠系统的淘汰机制自然发生（高层超水位时按 LRU 降级到下一层），无法对指定 key 触发 |
| 清空某个介质/节点的数据 | segment 粒度的 **drain job**（`POST /api/v1/drain_jobs`），但那是整个 segment 排空，不是单 key |

所以如果需要"精确删除 key 在 SSD 上的副本但保留 DRAM 副本"这类操作，当前版本需要改代码——把 `PushRemoveList` 的 filter 能力通过一个新的 RPC/REST 参数透出来即可，底层机制是现成的。

---

## 深入：容器结构、Insert 触发、batch_alloc vs batch_put_from、Get 回温性能、高低层含义、MmcMemObjMeta

### 问题 1：MmcMetaContainerLRU 对象的结构是什么样的？

`MmcMetaContainerLRU<std::string, MmcMemObjMetaPtr>`（`mmc_meta_container_lru.cpp:27-42`）= **1 张哈希表 + 3 条 LRU 链表 + 2 把读写锁 + 1 个类型回调**：

```
┌──────────────────────────── MmcMetaContainerLRU ─────────────────────────────┐
│                                                                              │
│  metaMap_: unordered_map<key, ValueLruItem>      lruLists_[MEDIA_NONE=3]:    │
│  ┌─────────┬───────────────────────────┐        ┌─────────────────────────┐  │
│  │ "blk_0" │┌─────────────────────────┐│  [0]HBM: head⇄"blk_7"⇄"blk_2"⇄tail │
│  │         ││ value_ ──────┐          ││        └─────────────────────────┘  │
│  │         ││ mediaType_=DRAM         ││  ┌─────────────────────────┐        │
│  │         ││ lruIter_ ────┐│         ││  [1]DRAM: head⇄"blk_0"⇄"blk_5"⇄tail│
│  ├─────────┤└──────────────┼│─────────┘│  └────────────▲────────────┘        │
│  │ "blk_5" │ ...           ││          └───────────────┘ lruIter_ 指回链表节点 │
│  └─────────┴───────────────┼┘          ┌─────────────────────────┐           │
│                            ▼           [2]SSD : head⇄"blk_9"⇄tail            │
│              MmcMemObjMetaPtr ──> MmcMemObjMeta（见问题7）└──────────────────┘
│                                                                              │
│  metaLock_: 读写锁，保护 metaMap_      lruLock_: 读写锁，保护 3 条链表         │
│  getTypeFunc_: 回调，从 Value 算出该 key 属于哪一层（见问题3）                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

关键设计：
- **一个 key 只在一条链表里**（`mediaType_` 记录在哪条），它表示"这个 key 当前主要挂在哪一层"，淘汰按层独立进行；
- `lruIter_` 缓存了自己在链表中的迭代器，使 `Promote`/摘除都是 **O(1)**；
- **两把锁分离**：查/改 map 用 `metaLock_`，动链表用 `lruLock_`，减少竞争；
- 链表头 = 最近使用（MRU），链表尾 = 最久未用（淘汰受害者）。

### 问题 2：MmcMetaContainer 的子类除了 MmcMetaContainerLRU，还有其他的吗？

**没有。** 全仓库只有一个实现：

```
        MmcMetaContainer<Key, Value>        （抽象接口，mmc_meta_container.h）
                    △
                    │ 唯一子类
        MmcMetaContainerLRU<Key, Value>     （mmc_meta_container_lru.cpp:28）
```

工厂函数 `MmcMetaContainer::Create()`（`mmc_meta_container_lru.cpp:305-309`）也只 `new` LRU 这一种。接口（`Insert/Get/Erase/EraseIf/IterateIf/MultiLevelElimination...`）是模板化的，理论上可以替换成 LFU 等策略，但当前版本没有第二个实现。

### 问题 3：MmcMetaContainerLRU 的 Insert 方法是由谁触发的？blobType 是什么概念？

#### Insert 的两个触发点

```
触发点①（主路径）：客户端 Put/Alloc
   client put(key) ──RPC──> MmcMetaMgrProxy::Alloc
                               └─> MmcMetaManager::Alloc
                                    ├─ globalAllocator_->Alloc(...)  // 全局池分配 blob
                                    └─ metaContainer_->Insert(key, tempMetaObj)
                                                        (mmc_meta_manager.cpp:301)

触发点②（恢复路径）：元数据重建
   MmcMetaManager::RebuildMeta(blobMap)                 (mmc_meta_manager.cpp:666)
   —— meta 服务重启/主备切换后，把备份的 blob 描述重新 Insert 回容器
```

#### blobType 是什么

Insert 时需要决定把这个 key 挂到**哪一层的 LRU 链表**，`blobType` 就是干这个的：

```cpp
// Insert 内 (mmc_meta_container_lru.cpp:57)
MediaType blobType = GetBlobType(value);
//   → getTypeFunc_(value)（构造时注入的回调，mmc_meta_manager.h:111）
//   → objMeta->GetBlobType()（mmc_mem_obj_meta.cpp:147）
MediaType MmcMemObjMeta::GetBlobType() {
    for (auto blob : blobs_)
        if (blob != nullptr) return blob->Type();   // 取第一个有效 blob 的介质类型
    return MEDIA_NONE;
}
```

即 **blobType = 该 key 第一个 blob 所在的介质层（HBM/DRAM/SSD）**。在 Put 路径上，刚分配的 blob 都在用户指定的目标介质上，所以第一个 blob 的类型就是本次写入的落点层。若返回 `MEDIA_NONE`（没有任何 blob），Insert 直接报错（`:58-61`）。

注意它的局限性：key 回温后会有多层 blob，但链表归属仍只记一层——所以回温完成后要用 `InsertLru(key, dstType)` 显式把它挂到新的层。

### 问题 4：对外暴露的 API 中，batch_alloc 和 batch_put_from 有什么区别？

一句话：**batch_put_from 是"一步到位按 key 写数据"；batch_alloc 是"只圈地不搬货"，把数据搬运留给 batch_copy 按 GVA 直写。**

```
batch_put_from(keys, buf_ptrs, sizes, direct, replicateConfig)   —— 一个调用完成：
   key ──> MetaService Alloc(按 replicateConfig 选介质/副本数)
        ──> 数据从本地 buffer 拷入全局池（按 key 寻址）
        ──> 同步更新状态 READABLE
   调用者自始至终只认 key，拿不到也不需要 GVA

batch_alloc(keys, sizes, media) ──> 返回 [gva0, gva1, ...]      —— 只做分配：
   key ──> MetaService Alloc(media 参数指定介质, NATIVE_AFFINITY, 单副本)
        ──> 返回每个 key 的起始 GVA
   ✗ 不拷任何数据！blob 处于"已分配、内容未写"状态

   之后由调用者自己驱动数据面：
   batch_copy(gva_ptrs, buf_ptrs, sizes, H2G)   // 按地址直写
   ...get_key_info(key, flag=1) 拿回 gva...
   batch_copy(gva_ptrs, buf_ptrs, sizes, G2L)   // 按地址直读
```

对比表：

| | batch_put_from | batch_alloc (+batch_copy) |
|---|---|---|
| 语义 | 完整的写入（分配+拷贝+置可读） | 仅分配全局内存，返回 GVA |
| 寻址方式 | 按 key（每次都查元数据） | 按 GVA（后续读写跳过 key 查找） |
| 介质/副本 | 由 `replicateConfig` 决定 | `media` 参数直给，固定单副本 |
| 典型用途 | 常规 KV 写入 | 高频 KV cache 场景：地址一次拿、反复直读直写，省掉每步的元数据交互 |

（依据：`mmcache_store.cpp:949-972` 的 `BatchMalloc` 实现与文档 `memcache_python_api.md:736-799` 的典型 GVA 流程。）

### 问题 5：Get 路径中对 blob 的读取是不是在 NPU 进程上触发的？同步执行 RewarmBlob 会不会很花时间、影响推理性能？

#### 读取在哪里触发

```
 vLLM 进程（NPU 所在节点）            MetaService 进程              远端/本地存储节点
 ┌───────────────────────┐   ①Get RPC   ┌──────────────────┐
 │ store.get_into(key)   │─────────────>│ FillObjMetaWithRewarm
 │ (MmcClientDefault::Get│              │  若只有SSD blob：  │ ②CopyBlob RPC
 │   mmc_client_default. │              │  RewarmBlob ───────┼──> SSD→DRAM 拷贝
 │   cpp:347)            │<─────────────│  返回新blob描述     │   (local service 之间)
 │                       │ ③返回blob描述 └──────────────────┘
 │ ④bmProxy_->BatchGet   │─────────────────────────────────────> 按 gva 直读数据
 │   (本地 local service │      ④ RDMA/sdma，数据进本地 buffer      (从 DRAM 读)
 │    线程执行, cpp:364) │
 └───────────────────────┘
```

- **触发方确实是 NPU 所在主机上的 vLLM/客户端进程**（client 库就在这个进程里），但真正搬数据的是 local service 的 IO 线程 + RDMA 网卡/sdma 硬件，不占 NPU 算力；
- **RewarmBlob 不在 vLLM 进程里执行**，它在 MetaService 进程里跑；vLLM 线程只是**阻塞在 ①③ 之间的 RPC 等待上**。

#### 会不会很慢、影响推理？

**会付出代价，但只发生在"冷数据首次读取"这一条路径上**，且有多层缓解：

```
读一个 key 的延迟分解：
  热数据（DRAM/HBM 有 READABLE blob）：①③ RPC + ④ RDMA        → 正常，无回温
  冷数据（只有 SSD blob）首次读：      ①③ RPC 内含 SSD→DRAM 拷贝 ← 多花的在这里
                                      + ④ 从 DRAM 读            → 明显变慢
  冷数据第二次读起：                   已在 DRAM，同热数据         → 快
```

缓解机制（前面分析过的设计在这里闭环）：

1. **ExistKey 异步回温 = 预取**：vLLM connector 的典型用法是先 `batch_is_exist` 再 `get`。`is_exist` 命中"只有 SSD"时就在后台提前回温（`ExistKey` 立即返回，不阻塞），等真正 `get` 时数据往往已在 DRAM；
2. **回温去重**：多个并发 Get 撞上同一个回温中的 key，只有第一个执行 RewarmBlob，其余在条件变量上**最多等 100ms**（`rewarmWaitMs`），然后直接读 DRAM 新 blob，不会每个请求都拷一遍 SSD；
3. **回温只升一级到 DRAM**，不追求一步到位到 HBM，控制了冷启动开销；
4. 淘汰是从高层往低层逐级的，**热数据平时就在 DRAM/HBM**，回温属于 cache miss 的兜底路径而不是常态。

所以准确说：**它把"首次冷读"的延迟换成了"后续热读"的加速**，对推理的影响集中在 cache miss 的那一次请求；如果上层（vLLM）先用 `is_exist` 探一遍，这个延迟还能被进一步隐藏。

### 问题 6：Get 路径中的"高层"是什么含义？一定是 HBM 吗？"低层伴生（写入进行中）"是什么、如何判断？

#### "高层"≠ 特指 HBM

遍历代码里的判定（`mmc_meta_manager.cpp:85-90`）：

```cpp
bool isLowestTier = (MoveDown(type) == MEDIA_NONE);  // 唯一能"再往下没有层"的是 SSD
if (isLowestTier) { ...记 lowerBlob... }
else              { ...这是"高层"，检查 READABLE/ALLOCATED... }
```

```
三级部署：  HBM ─┐
                 ├─ 都是"高层"（isLowestTier=false）     SSD = 最低层
            DRAM ┘
二级部署（hbm.size=0）：DRAM 自己就是"高层"               SSD = 最低层
```

所以**高层 = HBM 或 DRAM 的统称**，即"内存介质层"，不一定是 HBM。`ExistKey` 里的 `hasReadableHigher` 也印证了这一点（`mmc_meta_manager.cpp:213`）：

```cpp
if (type == MEDIA_HBM || type == MEDIA_DRAM) { hasReadableHigher = true; }
```

——它判断的是"HBM **或** DRAM 有没有可读副本"，两者都算高层。

#### "低层伴生"是什么、怎么判断

**低层伴生 = `lowerBlob`：同一个 key 在最低层（SSD）上那份 READABLE 的 blob。** 判断方式就在遍历里：

```
for each blob:
    if MoveDown(type)==NONE (是SSD) 且 state==READABLE ──> lowerBlob = 它  ← 伴生副本
    else (HBM/DRAM) 且 state==ALLOCATED                ──> pendingBlob = 它
```

它的语义是"**高层那份 ALLOCATED 的 blob 有数据来源**"：

- 回温（RewarmBlob）的特征就是：SSD 老副本保持 READABLE 不动，同时 DRAM 出现一个 ALLOCATED 新副本 → `pendingBlob + lowerBlob` 同时存在；
- 普通写入（Put）的特征是：只有新分配的 ALLOCATED blob，低层什么都没有 → `pendingBlob` 单独存在。

### 问题 7：MmcMemObjMeta 对象的结构是什么样的，功能是什么？

定义在 `mmc_mem_obj_meta.h:30-136`，是**一个 key 的元数据对象**（容器里的 Value）：

```
┌─────────────────────── MmcMemObjMeta（一个 key）───────────────────────┐
│  prot_      uint16   访问权限（读/写）                                   │
│  priority_  uint8    优先级（预留给淘汰策略）                             │
│  numBlobs_  uint8    blob 副本数                                        │
│  size_      uint64   单个 blob 的数据字节数                              │
│  mutex_     mutex    本对象级锁（注释：make sure the size is 64 bytes，  │
│                      按缓存行优化，读写本对象前必须持有）                  │
│  blobs_     list<MmcMemBlobPtr> ─┐                                      │
└──────────────────────────────────┼──────────────────────────────────────┘
                                   ▼   一个 key 的 blob 链（多层/多副本）
        ┌──────────────┐   Next()  ┌──────────────┐   Next()  ┌──────────────┐
        │ blob: DRAM   │──────────>│ blob: SSD    │──────────>│     ...      │
        │ rank2 READABLE│          │ rank2 READABLE│          │              │
        └──────────────┘           └──────────────┘           └──────────────┘
```

功能（即它提供的方法族）：

| 方法 | 功能 |
|---|---|
| `AddBlob` | 挂一个新 blob（多副本/回温新副本） |
| `GetBlobs(filter, revert)` | 按 `{rank, 介质, 状态}` 过滤取 blob 子集——Get 遍历、ExistKey 统计都靠它 |
| `FreeBlobs(key, allocator, filter)` | 按过滤器释放 blob（淘汰只删源层、回滚只删目标层） |
| `UpdateBlobsState` | 按过滤器批量推进 blob 状态机 |
| `GetBlobType()` / `MoveTo(down)` | 回答"这个 key 在哪层/上下一层是哪层"（Insert 和淘汰迁移用） |
| `Mutex()` | 对象级互斥锁，所有 blob 增删、遍历、回温决策都在它保护下完成 |

一句话：**容器（问题 1）管"key 之间的 LRU 顺序"，MmcMemObjMeta 管"key 内部有哪些副本、分别在哪个节点哪层、什么状态"。**

### 问题 8："pendingBlob 有 lowerBlob 伴生 → 回温中可等；没有伴生 → 写入中对读不可见"如何理解？

回看 `FillObjMetaWithRewarm` 的两个分支（`mmc_meta_manager.cpp:102-120`），这是两种**外观相似但本质不同**的场景：

#### 场景 A：pendingBlob **有** lowerBlob 伴生 → 回温进行中，值得等

时间线上这个 key 的状态是：

```
  时刻1: key 只有 SSD blob(READABLE)                     ← 冷数据
  时刻2: 某线程触发 RewarmBlob
           ├─ DRAM 分配新 blob，AddBlob(ALLOCATED)        ← pendingBlob 出现
           └─ SSD blob(READABLE) 原地不动                 ← lowerBlob 伴生
  时刻3: RPC 拷贝 SSD→DRAM 完成，新 blob → READABLE
```

此时你来 Get，看到的是 `ALLOCATED(DRAM) + READABLE(SSD)`。因为 **SSD 那份老数据完好存在**，说明 DRAM 那份 ALLOCATED 一定是回温产生的（写入的数据源就是 SSD），它**马上就会变 READABLE**。所以代码选择等待：

```cpp
pendingBlob->WaitUntilReadable(guard, 100ms);   // 被 cv_ 唤醒后直接读热腾腾的 DRAM
```

等 100ms 换来之后所有读取都走 DRAM，划算；超时则报错由上层重试。

#### 场景 B：pendingBlob **没有** lowerBlob 伴生 → 首次写入进行中，不能等

时间线是：

```
  时刻1: 客户端 put(key) → Alloc 出 DRAM blob(ALLOCATED)   ← pendingBlob
         （这是这份数据在系统里的唯一副本，低层什么都没有）
  时刻2: 客户端正在往里写数据...（可能刚开始，也可能写一半）
  时刻?: 写完后 WRITE_OK → READABLE
```

此时 blob 里的**内容是未定义的**——可能半个字都没写。没有低层副本意味着没有可兜底的数据源。如果让你读，要么读到垃圾，要么不知道要等多久（大对象首次写入耗时无界）。所以代码把它**对读路径隐藏**：

```cpp
objMeta.numBlobs_ = 0;   // 返回"现在没有可读的 blob"
                         // 客户端视为未就绪，稍后重试
```

#### 一句话对比

```
 ALLOCATED 高层 blob + READABLE 低层 blob → 回温：数据有源，完成可期，【等】
 ALLOCATED 高层 blob + 低层什么都没有     → 首写：内容未定，等待无界，【藏】
```

区分依据之所以可靠，是因为两条路径**创建 ALLOCATED blob 的方式不同**：回温一定先保旧副本再建新副本（先 `AddBlob` 后拷贝，旧副本全程不动）；而首次 Put 创建的是无伴生的全新 blob。"有没有低层伴生"就成了区分二者的天然标志。

---

## Q&A：MemcacheBackend 各方法在请求处理中的调用时序

> **原始问题：** C:\Code\vllm-ascend\vllm_ascend\distributed\kv_transfer\kv_pool\ascend_store\backend\memcache_backend.py 文件中的各个方法，在处理请求时候的调用时序是怎样的？请画图说明，并给出必要的说明。

（关键前提：backend 只存在于 **Worker 进程**（每 NPU rank 一个），scheduler 进程对它的调用实际是经 ZMQ 转发到 rank0 worker 上执行的。）

### 一、总体时序图

```
时间 ──────────────────────────────────────────────────────────────────────▶

┌─ 阶段0：Worker 进程初始化（整个生命周期只跑一次）───────────────────────────┐
│                                                                            │
│  vLLM Worker.init / KVConnectorFactory.create_connector(WORKER)            │
│      │                                                                     │
│      ▼                                                                     │
│  ① __init__(parallel_config, lazy_init)                                    │
│      ├─ get_world_group().local_rank        # 确定本 rank 的 NPU 设备号    │
│      ├─ 非 lazy ──▶ _setup_store() ──▶ DistributedObjectStore().init()    │
│      │              [A2: 先 all_gather warmup]   ★store 就绪               │
│      └─ lazy(DSV4)──▶ store=None，推迟到第一次 put                          │
│      │                                                                     │
│      ▼  (model runner 分配完 KV cache 后)                                  │
│  ② register_buffer(ptrs, sizes)   ◀── KVPoolWorker.register_kv_caches      │
│      ├─ 记录显存区间 (base_addr, region_len)                                │
│      └─ [仅 A2 且 store 已就绪] store.register_buffer() × N 个区间          │
│      │                                                                     │
│      ▼  (传输线程启动，kv_transfer.py:80)                                  │
│  ③ set_device() ──▶ torch.npu.set_device(npu:{local_rank})                 │
│         [发送线程 / 接收线程 各调一次]                                      │
└────────────────────────────────────────────────────────────────────────────┘

┌─ 阶段1：每个新请求 · 调度期（scheduler 进程发起，rank0 worker 执行）────────┐
│                                                                            │
│  Scheduler.get_num_new_matched_tokens()                                    │
│      └─ZMQ REQ──▶ rank0 Worker: LookupKeyServer ──▶ lookup_scheduler()     │
│                        │                                                   │
│                        ▼                                                   │
│                   ④ exists(keys) ──▶ batch_is_exist                        │
│                      [每个 kv group 一次；TP/PP 变体展开后一批查完]         │
│                      [lazy 且未初始化 → 直接返回全 0，视为未命中]           │
│                        │                                                   │
│                   命中 token 数 ◀──ZMQ RESP── 回到 scheduler，决定分配      │
└────────────────────────────────────────────────────────────────────────────┘

┌─ 阶段2：每个命中请求 · forward 之前（本 rank worker，load 路径）────────────┐
│                                                                            │
│  start_load_kv(metadata)                                                   │
│      ├─ 同步模式:        ⑤ get(keys,addrs,sizes) ──▶ batch_get_into_layers │
│      │                     (pool_worker.py:615)                 (G2L DMA)  │
│      ├─ load_async:      ⑤ get(...)  在 KVCacheStoreRecvingThread 中       │
│      │                     (kv_transfer.py:515)                            │
│      └─ layerwise:       ⑤ get(...)  逐层，见阶段3                         │
│                      [lazy 且未初始化 → 报错返回 None，标记块失效]          │
│      ▶ 返回后：命中前缀的 KV 已在本地显存块中，可直接参与 attention          │
└────────────────────────────────────────────────────────────────────────────┘

┌─ 阶段3：forward 之中（仅 use_layerwise=True）──────────────────────────────┐
│                                                                            │
│  逐层交替：                                                                 │
│    wait_for_layer_load(layer_i) ──▶ ⑤ get(layer_i 的 keys)  ── 边下边算    │
│    save_kv_layer(layer_i)       ──▶ ⑦ put(layer_i 的 keys)  ── 边算边存    │
│  （生成器 retrieve_layer/store_layer 驱动，key 带 @layer_id 后缀）          │
└────────────────────────────────────────────────────────────────────────────┘

┌─ 阶段4：每个请求 · forward 之后（本 rank worker，save 路径）────────────────┐
│                                                                            │
│  wait_for_save() ──▶ KVCacheStoreSendingThread._handle_request             │
│      │                                                                     │
│      ▼  去重（kv_transfer.py:352）                                         │
│  ⑥ exists(keys) ──▶ batch_is_exist      # 已在池里的 chunk 不再重复存      │
│      │                                                                     │
│      ▼  current_event.synchronize()  # 等 NPU 算完                         │
│  ⑦ put(keys,addrs,sizes) ──▶ batch_put_from_layers            (L2G DMA)   │
│      ├─ [lazy 且首次] _ensure_initialized() → _setup_store()               │
│      │                 + _register_buffers_if_needed()  ← 补做阶段0的事    │
│      └─ 失败仅记日志，不抛异常（容错：池满/网络问题不拖垮推理）              │
│      │                                                                     │
│      ▼  queue.join() → get_finished() 上报 scheduler → 释放本地 block      │
└────────────────────────────────────────────────────────────────────────────┘
```

### 二、必要说明

#### 1. 调用次数统计（一个"命中且需要回存"的请求，非 layerwise）

| 方法 | 调用次数 | 发生位置 |
|---|---|---|
| `__init__` / `_setup_store` | 全程 1 次（lazy 时推迟到首次 put） | 进程启动 / 首个 put |
| `register_buffer` | 全程 1 次（内部对 N 个显存区间循环注册，仅 A2 真正下发） | KV cache 分配后 |
| `set_device` | 每个传输线程 1 次 | 线程 `run()` 开头 |
| `exists` | **每请求 2 次**：① 调度期 lookup（只在 rank0 worker）；② 保存前去重（每个 worker rank） | 阶段1、阶段4 |
| `get` | 每请求 0 或 1 次（仅命中外部缓存才调；layerwise 时每层 1 次） | 阶段2/3 |
| `put` | 每请求 0 或 1 次（kv_consumer 且未开 `consumer_is_to_put` 时不调；layerwise 时每层 1 次） | 阶段4/3 |

#### 2. 三条路径对应的角色

- **kv_producer（P 实例）**：只走 阶段0 → 阶段4（`exists` 去重 + `put`）；
- **kv_consumer（D 实例）**：走 阶段0 → 阶段1（`exists` lookup）→ 阶段2（`get`）；默认不 put；
- **kv_both（单机/共池）**：全部走一遍，顺序是 `exists(lookup)` → `get` → `exists(dedup)` → `put`。

#### 3. lazy_init（DSV4 压缩模型）对时序的改变

非 lazy 时 `_setup_store` 在 ① 同步完成；lazy 时整个初始化被"折叠"进**第一次 `put`**：

```
__init__(store=None) → register_buffer(只记录,不下发) → set_device 照常
   → exists: 未初始化,返回全 0（不触发初始化,lookup 永远 miss,无害）
   → get:    未初始化,报错返回 None（设计前提:put 一定先于 get 发生）
   → 首次 put: _ensure_initialized() { _setup_store() + 补注册 buffer } → 正常 put
```

依据：`memcache_backend.py:27`（A2 禁用 lazy）、`37-48`（加锁的 `_ensure_initialized`）、`96-104`（exists 短路）、`107-115`（get 短路）、`147`（put 触发初始化）。

#### 4. 线程模型

`exists`/`get`/`put` 的实际调用线程不是统一的：调度期 `exists` 跑在 **LookupKeyServer 的 ZMQ 线程**（rank0）；load 的 `get` 跑在**主推理线程**（同步）或 **RecvingThread**（异步）；save 的 `exists`+`put` 跑在 **SendingThread**。它们共享同一个 `DistributedObjectStore`，线程安全由 memcache_hybrid C++ 库内部保证；`set_device` 在每个传输线程入口各调一次正是为了让该线程的后续 DMA 落在正确的 NPU 上下文上。

#### 5. 容错语义

`get`/`put` 内部都是 try/except + 逐 key 返回码检查：失败只记日志并（对 get）把失败 block 记入 `_invalid_block_ids` 反馈给 scheduler 重算，**不会把异常抛进推理主路径**——memcache 不可用降级为"缓存未命中"，而不是推理失败。

---

## Q&A：基础概念澄清（num_kv_head / TP rank / 三种情形 / 注意力结构 / update_block_hashes / 哈希函数 H）

> **原始问题：**
> 1. 上面回答提到的 `self.num_kv_head = model_config.get_total_num_kv_heads()`，num_kv_head 具体是什么意思？
> 2. 上面回答提到的 TP rank 是一个什么概念？为什么"每个 TP rank 的 NPU 上都有自己的一份 KV cache 显存"？
> 3. 上面回答提到的三个情形是什么使用场景会出现的？
> 4. GQA/MHA、MLA（DeepSeek）分别是什么注意力结构？
> 5. "每当 token 增加（prefill 接收 prompt、decode 每生成 token）就调 `update_block_hashes()`（request.py:257-260）增量补齐新满块的哈希。"这个方法要详细解释一下，并且画图给出它在整个请求处理链的位置。
> 6. `h₀ = H( NONE_HASH, (t₀…t₁₂₇), extra_keys )` 中的 H，是 hash 的意思吗？

### 1. `num_kv_head` 是什么意思？

**`num_kv_head` = 模型每层 attention 中 KV 头（Key/Value head）的数量**，是模型结构决定的常量，来自 HF config 的 `num_key_value_heads`。

代码依据（`vllm/config/model.py:1328-1343`）：

```python
def get_total_num_kv_heads(self) -> int:
    """Returns the total number of KV heads."""
    return self.model_arch_config.total_num_kv_heads      # 整个模型每层的 KV 头数

def get_num_kv_heads(self, parallel_config) -> int:
    """Returns the number of KV heads per GPU."""
    if self.use_mla:
        return 1                                          # MLA 按 1 处理
    total_num_kv_heads = self.get_total_num_kv_heads()
    # TP 时均分；不够分时复制，保证每卡至少 1 个
    return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```

直觉理解：attention 里每个 KV 头产生一份 K 矩阵和 V 矩阵，**KV cache 就是按头存的**——每层每 token 的缓存大小 ≈ `2（K和V）× 每卡KV头数 × head_dim`。所以它直接决定：

- 每块 NPU 上 KV cache 显存的大小；
- TP 切分时各 rank 缓存的数据是"不同切片"还是"相同副本"——这正是 `head_or_tp_rank`/`put_step` 那套逻辑的输入。

举例：Llama-3-8B 是 32 个 query 头、8 个 KV 头（GQA）；DeepSeek-V3 用 MLA，`use_mla=True` 直接按 1 算。

### 2. TP rank 是什么概念？为什么每个 TP rank 的 NPU 上有自己的一份 KV cache 显存？

**TP（Tensor Parallelism，张量并行）** 是把同一层模型的权重矩阵**切开分到多张 NPU 上**的并行方式：TP=8 就是 8 张卡，每张卡是一个 **TP rank**（编号 0~7），各存 1/8 的权重，forward 时各算各的分片，再用 AllReduce/AllGather 通信拼出完整结果。

为什么每个 rank 都有自己的 KV cache？两个层面：

- **物理层面**：8 个 rank 是 8 个独立进程、8 块独立 NPU，显存天然不共享，每块卡的 worker 进程各自调用 `register_kv_caches` 在自己卡上分配 KV cache buffer；
- **逻辑层面**：attention 的头也随 TP 切分——rank r 只负责计算分配给它的那些 query/KV 头，那么它产生的 K/V 数据也只跟这些头有关，**只存自己算的那部分就够了**。所以 GQA 下各 rank 的 KV cache 内容互不相同（合起来才是完整的一份）；MLA 下各 rank 算的是同一份 latent，内容是相同副本。

这也解释了之前讨论的必要性：既然 KV 数据按 rank 分布在不同卡上，往全局池存取时就必须在 key 里标明"这是哪个 rank 的那一份"（`head_or_tp_rank`），否则取回来会对不上号。

### 3. 三个情形分别在什么使用场景出现？

情形由 **模型结构（num_kv_head）× 部署的 TP 大小** 共同决定：

| 情形 | 条件 | 典型场景 |
|---|---|---|
| **A**：`num_kv_head ≥ tp_size` → 每 rank 一个独立 keyspace | KV 头够分 | **绝大多数模型的常规部署**。如 Llama-3-70B（8 KV 头）TP=8：rank i 存 head i，key 为 `:0`~`:7`；Qwen2-7B（4 KV 头）TP=2：每 rank 2 个头 |
| **B**：`num_kv_head=1 < tp_size` → 全 rank 共用 `:0` | MLA / MQA 模型 | **DeepSeek-V2/V3/R1 的部署**（vllm-ascend 的主战场），TP=8/16 时所有 rank 缓存内容相同，共用 `:0`，chunk 按 `tp_rank % 8` 摊派；MQA 模型（如部分 Falcon）同理 |
| **C**：`1 < num_kv_head < tp_size` → 部分复制 | 小 KV 头数的 GQA 模型用**大卡数**部署 | 如 4 KV 头的模型上 TP=8：`put_step=2`，`head_or_tp_rank=tp_rank//2` → `:0`~`:3` 四个 keyspace，每个 keyspace 有 2 个 rank 持有副本、内部再摊派 chunk。小模型（如 2 个 KV 头的 1B 模型）为了 PD 分离/大并发硬上 TP=8 时常见 |

一句话：**情形 A 是"数据不同"，情形 B 是"数据全同"，情形 C 是"部分相同"**——模型越小、TP 开得越大，越往 B/C 靠。

### 4. MHA / GQA / MLA 分别是什么注意力结构？

三者的区别只在 **K、V 头与 Q 头的对应关系**，直接影响 KV cache 大小：

**MHA（Multi-Head Attention，多头注意力）**——最原始的形态：
每个 query 头都有自己专属的 K 头和 V 头，`num_kv_head = num_query_head`。如 32 个 Q 头就配 32 个 KV 头。效果好但 KV cache 最大。GPT-2/3、Llama-1 时代的主流。

**GQA（Grouped-Query Attention，分组查询注意力）**：
把 query 头分组，**每组共享一个 KV 头**。如 32 个 Q 头分 8 组，只有 8 个 KV 头——KV cache 直接降为 MHA 的 1/4，效果损失很小。Llama-2/3-70B、Qwen 系列、Mistral 等现代模型几乎都是 GQA。特例：**MQA**（Multi-Query Attention）是只有 1 个 KV 头的极端 GQA。

**MLA（Multi-head Latent Attention，多头潜在注意力，DeepSeek 提出）**：
不再按头存 K/V，而是把 K/V **压缩成一个低秩的潜在向量 c_t**（维度远小于"头数×head_dim"），缓存里只存这个 latent（外加一小段 RoPE 分量）；用时再通过升压矩阵还原出各头的 K/V（推理时可用矩阵吸收技巧把升压合并进计算，decode 阶段等价于 MQA）。KV cache 比 GQA 再小一个数量级，这是 DeepSeek-V2/V3 长上下文低成本的关键。

对应到代码：`model.py:1334` 的 `if self.use_mla: return 1`——MLA 缓存在逻辑上就是"全局一份"，所以 vllm-ascend 里 MLA 按 `num_kv_head=1` 处理，所有 TP rank 持相同副本。

```
MHA:  Q头0 Q头1 Q头2 Q头3        GQA: Q头0 Q头1 Q头2 Q头3       MLA: Q头0 Q头1 Q头2 Q头3
       │    │    │    │                │    │    │    │                │    │    │    │
      KV0  KV1  KV2  KV3              └──KV0─┘  └──KV1─┘             └──── c_t(latent) ────┘
      cache = 4 份                  cache = 2 份                     cache = 1 个低维向量
```

### 5. `update_block_hashes()` 详解 + 它在请求链中的位置

#### 代码本体（`vllm/v1/request.py`）

```python
# line 199  请求对象里维护的哈希列表，第 i 项 = 前 (i+1)×hash_block_size 个 token 的承诺
self.block_hashes: list[BlockHash] = []
# line 203  哈希函数（由 EngineCore 注入，不带 self 绑定以避免循环引用）
self._block_hasher = block_hasher
# line 204  构造时立即算一次：prompt 的所有完整块
self.update_block_hashes()

# line 244-255  decode 每生成 token 时由 scheduler 调用
def append_output_token_ids(self, token_ids):
    ...
    self._all_token_ids.extend(token_ids)
    self.update_block_hashes()          # line 255

# line 257-260  本体
def update_block_hashes(self) -> None:
    """Compute block hashes for any new full blocks and append them."""
    if self._block_hasher is not None:
        self.block_hashes.extend(self._block_hasher(self))
```

#### 为什么说是"增量补齐"

关键在 `_block_hasher` 的实现（`kv_cache_utils.py:705-746`）：

```python
start_token_idx = len(request.block_hashes) * hash_block_size   # ← 从"已算到的位置"继续
...
while True:
    end_token_idx = start_token_idx + hash_block_size
    if end_token_idx > num_tokens:
        break                    # 只对【新凑满的完整块】算哈希，不满一块就停
    block_hash = hash_block_tokens(caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys)
    new_block_hashes.append(block_hash)
    prev_block_hash_value = block_hash     # 链式滚动
```

`start_token_idx = 已有哈希数 × 块长`，所以它是**纯增量**的：之前算过的块绝不重算，每次调用只补算"自上次以来新凑满"的块。

- **prefill 时**（`__init__`）：prompt 假设 384 token、块长 128，一次性算出 h₀、h₁、h₂；尾部不足 128 的 43 个 token 不算；
- **decode 时**（`append_output_token_ids`，被 `scheduler.py:2005` 每步调用）：每步只多 1 个 token，绝大多数调用 `while` 循环第一次就 break，**返回空列表、开销≈0**；只有当新 token 恰好跨过块边界（如第 512 个 token 落位）才补算一个 h₃。这就是为什么可以在每个 decode step 都调而不心疼性能。

`self._block_hasher` 为 None 的情况：既没开 prefix caching 也没配 KV connector 时（`engine/core.py:214`），整个机制关闭。

#### 在请求处理链中的位置

```
HTTP 请求
   │
   ▼
Processor: tokenize → EngineCoreRequest
   │
   ▼
EngineCore.add_request → Request.__init__ (request.py:204)
   │   └─▶ update_block_hashes()  ①【prefill 时刻】一次性算 prompt 全部完整块
   │                                 产出 block_hashes = [h₀, h₁, h₂, ...]
   ▼
┌── Scheduler 每轮调度 ─────────────────────────────────────────────────┐
│  ② block_hashes 的第一个消费者：                                       │
│     a. 本地前缀缓存：KVCacheManager 用它找 GPU 上可复用的 block          │
│     b. connector.get_num_new_matched_tokens(request)                  │
│        → KVPoolScheduler 把 request.block_hashes 经 ZMQ 发给 rank0     │
│        → 生成 key → MemcacheBackend.exists() 查全局池 ★memcache 入口   │
│  ③ 分配 block、build_connector_meta：block_hashes 随 ReqMeta 到 Worker │
│     → Worker 用它生成 put/get 的 key ★memcache 存取入口                │
│  ④ 执行 forward，产出新 token                                          │
│  ⑤ scheduler.update_from_output → append_output_token_ids              │
│     └─▶ update_block_hashes()  【decode 每步】增量补哈希（多数为空操作）│
│     → 凑满新块后，该块的 KV 在下一步 wait_for_save 里就能以新 key 回存   │
└────────────────────────────────────────────────────────────────────────┘
```

即：`update_block_hashes` 是**整条链的"原料供应点"**——本地前缀缓存、memcache 的 exists/get/put 三套 key，全部溯源到它维护的 `request.block_hashes`。

### 6. `H` 是 hash 的意思吗？

**是的**，`H` 就是哈希函数，具体是 vLLM 配置项 `cache_config.prefix_caching_hash_algo` 选定的算法，经 `get_hash_fn_by_name()`（`vllm/utils/hashing.py:82-97`）解析：

| 配置值 | 实际函数 |
|---|---|
| `"sha256"`（默认） | SHA-256 |
| `"sha256_cbor"` | 先 CBOR 序列化输入元组再 SHA-256 |
| `"xxhash"` / `"xxhash_cbor"` | xxHash（更快，非加密级） |

所以 `h₀ = H(NONE_HASH, (t₀…t₁₂₇), extra_keys)` 的精确含义是：

```python
h₀ = BlockHash( hash_function( (NONE_HASH, (t₀,…,t₁₂₇), extra_keys) ) )
#           ↑ 把三元组一起序列化后算摘要，输出定长字节的指纹
```

三个要点：

1. **它是确定性指纹，不是加密**：同样的 `(父哈希, token 序列, extra_keys)` 在任何进程、任何机器上算出完全相同的字节串——这正是 P/D 两台机器能对上 key 的数学基础；
2. **输入是三元组**：父块哈希（实现链式）+ 本块 token ids + extra_keys（LoRA/多模态/cache_salt 等）；
3. **`NONE_HASH`** 是进程启动时用同一个哈希函数对固定种子算出的"创世哈希"（`kv_cache_utils.py:99-114`），作为第一个块的父哈希，保证全集群一致。

---

## Q&A：Scheduler.schedule() 与 KVCacheManager 的协同工作

> **原始问题：** 我们知道 v1.core.sched.scheduler.Scheduler 对象持有了 KVCacheManager 字段，而且也知道该对象的 schedule 方法实现了请求调度逻辑。那么，想请你结合 schedule 方法的主要逻辑，阐明一下该方法是怎样和 kv_cache_manager 一起工作的。最好详细一些，可以图文共同来说明这个事情。

（依据代码：`vllm/v1/core/sched/scheduler.py:427-1226`、`vllm/v1/core/kv_cache_manager.py` 全文结构、抢占与收尾路径。）

### 一、角色分工总览

```
┌────────────────────────────────────────────────────────────────────┐
│  Scheduler（v1/core/sched/scheduler.py）                            │
│  职责：决策"这一步哪些请求跑、各跑多少个 token"                      │
│  持有：waiting / running 队列、token_budget、调度策略               │
│  不做：不直接管理显存块，全部委托给 kv_cache_manager                 │
└───────────────────────────────┬────────────────────────────────────┘
                                 │ 方法调用
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  KVCacheManager（v1/core/kv_cache_manager.py）                      │
│  职责：决策"token 放在哪些物理 block 里、哪些前缀可以复用"           │
│  接口：get_computed_blocks / allocate_slots / free / cache_blocks   │
└───────┬───────────────────────────────────────┬────────────────────┘
        │                                        │
        ▼                                        ▼
┌───────────────────────┐              ┌──────────────────────────────┐
│ KVCacheCoordinator    │              │ BlockPool                    │
│ （按 group 管理，支持  │              │ 物理 block 池：              │
│  full attn/SWA/mamba  │              │  ① 空闲 block 队列           │
│  混合模型）            │              │  ② hash→block 映射（前缀缓存）│
│ · 算需要几个 block    │              │  ③ ref_cnt + LRU 淘汰        │
│ · 查找前缀命中         │              └──────────────────────────────┘
│ · 提交/释放 block     │
└───────────────────────┘
```

一句话：**Scheduler 是"调度大脑"，KVCacheManager 是"内存管家"**——大脑每步问管家三件事：① 这请求有多少前缀已经算过了？② 给它 N 个新 token 能不能找到地方放？③ 放不下我就杀人（抢占），你收尸。

### 二、`schedule()` 主流程与 KVCacheManager 的触点

```
schedule() 开始
   │
   ▼  [触点0] kv_cache_manager.new_step_starts()          (scheduler.py:463)
   │         通知管理器"新一步开始"（用于步内状态重置）
   │
   ═══════════ 第一循环：RUNNING 队列（已在跑的请求）═══════════
   │  对每个 running 请求：
   │    ① 算 num_new_tokens（num_tokens_with_spec - num_computed，
   │       受 token_budget / long_prefill_threshold / max_model_len 裁剪）
   │    ② encoder / mamba 对齐调整
   │    ▼  [触点1] allocate_slots(request, num_new_tokens,
   │       │                     num_lookahead_tokens)      (:566)
   │       │     ┌─ 成功 → 返回新分配的 blocks，记账，budget 扣减
   │       │     └─ 返回 None（显存不够）→ 进入【抢占循环】：
   │       │          选出最低优先级请求 → _preempt_request()   (:605)
   │       │            └─ _free_request_blocks()               (:1212)
   │       │               └─ kv_cache_manager.free()           (:2248)
   │       │                  受害者 blocks 全部归还池子
   │       │          重试 allocate_slots ... 直到成功或轮到自己被杀
   │
   ═══════════ 第二循环：WAITING 队列（新到/被抢占的请求）═══════════
   │  （前提：本步没有发生抢占，且未暂停）
   │  对每个 waiting 请求：
   │    ① 若 num_computed_tokens == 0（新请求）：
   │       ▼  [触点2] 本地前缀缓存查找
   │       │   无 connector: get_computed_blocks(request)              (:739)
   │       │   有 connector: get_computed_blocks_for_connector(request)(:730)
   │       │      内部：coordinator.find_longest_cache_hit(
   │       │                request.block_hashes, num_tokens - 1)
   │       │      → (命中的blocks, num_local_computed_tokens)
   │       │      ※ 用 block_hashes 在 BlockPool 的 hash 表中找可复用块
   │       ▼  [触点3] 外部缓存查找（有 connector 时）
   │       │   connector.get_num_new_matched_tokens(request, num_local)(:744)
   │       │      → num_external_computed_tokens
   │       │      ※ 对 AscendStoreConnector 就是查 memcache 的那一步
   │       │   num_computed = 本地命中 + 外部命中
   │    ② 算 num_new_tokens = num_tokens - num_computed（同样受 budget 裁剪）
   │    ▼  [触点4] allocate_slots(request, num_new_tokens,
   │       │            num_new_computed_tokens=num_local,
   │       │            new_computed_blocks=命中块,
   │       │            num_external_computed_tokens=外部命中,
   │       │            delay_cache_blocks=load_kv_async, ...) (:919)
   │       │     └─ 返回 None → break（waiting 请求不抢占，直接停止准入）
   │    ③  [触点5] connector.update_state_after_alloc(
   │       │          request, kv_cache_manager.get_blocks(req_id),    (:947)
   │       │          num_external_computed_tokens)
   │       │     → AscendStore 侧在此标记 LoadSpec.can_load = True
   │    ④  移入 running，记录 scheduled_new_reqs / resumed_reqs
   │
   ═══════════ 收尾：组装 SchedulerOutput ═══════════
   │    [触点6] get_num_common_prefix_blocks()   (:1074，cascade attention 用)
   │    [触点7] take_kv_cache_block_copies()     (:1112，COW 复制任务)
   │    [触点8] req_to_new_blocks[req] = get_blocks(req_id) 转成 block_ids
   │            随 SchedulerOutput 发给 worker —— worker 只拿 block_id 列表，
   │            物理地址映射在 worker 侧 KV tensor 上解释
   ▼
返回 SchedulerOutput
```

### 三、`allocate_slots` 内部：管家的一次完整分配

这是两个循环共同的核心调用（`kv_cache_manager.py:340-561`）。它内部的 block 布局（docstring 原图）：

```
token 轴 →
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
   已算过的    本次本地前缀     connector      本步要算    投机/前瞻
               缓存命中        外部命中
```

三个工作阶段（`kv_cache_manager.py:424-435` 注释 + 实现）：

```
阶段1：清扫 + 容量检查
   coordinator.remove_skipped_blocks(req_id, ...)        (:500)
     释放"滑窗之外"等注意力不再需要的旧 block（先还后借，减少淘汰）
   coordinator.get_num_blocks_to_allocate(...)           (:506)
     按 group 算出还需几个新 block
   检查 free_blocks - reserved ≥ need + watermark        (:519-523)
     不够 → 返回 None（触发 scheduler 的抢占或停止准入）

阶段2：接管前缀命中块
   coordinator.allocate_new_computed_blocks(...)         (:531)
     把 find_longest_cache_hit 找到的命中块挂到本请求的 block table 上，
     ref_cnt +1（防止被 LRU 淘汰）；为 ext_comp 的外部命中块分配空壳 block
     （等 connector 把远端 KV 灌进来，memcache 的 get 就写进这些块）

阶段3：分配新计算块 + 提交哈希
   coordinator.allocate_new_blocks(...)                  (:538)
     从 BlockPool 空闲队列弹出新 block 给 new + lookahead
   coordinator.cache_blocks(request, num_tokens_to_cache)(:559)
     把"已确定算完"的 block 的 hash 注册进 BlockPool 的 hash 表
     —— 这一步让它们变成【别人的前缀缓存】，未来请求可命中
```

### 四、schedule() 之外：一轮完整生命周期中的配合

`schedule()` 只是"上半场"，管家与大脑的配合贯穿请求一生：

```
请求到来                每步调度                    执行后                结束时
    │                      │                          │                    │
    │   schedule()          │   update_from_output()   │   finish/free      │
    │  ┌──────────────┐     │  ┌──────────────────┐   │  ┌──────────────┐ │
    ▼  │get_computed_  │     ▼  │cache_blocks(req,  │   ▼  │free(request) │ │
 block │blocks() 本地命中│  token│ num_computed)     │  释放  │pop_blocks_   │ │
 hashes│connector 外部命中│  回来 │ 把新算完的块提交进 │  请求  │for_free()    │ │
 生成  │allocate_slots() │─────▶│ 前缀缓存 hash 表   │─────▶│block_pool.   │ │
(链式) │  → block_ids    │ block│ （scheduler.py:   │ block│free_blocks() │ │
    │  └──────────────┘  ids   │  2549/2568）       │ 表   │(:2248-2261)  │ │
    │                      │    └──────────────────┘   │  └──────────────┘ │
    │                      │   被抢占时：                │   块上的 hash 仍保留 │
    │                      │   _preempt_request → free  │   → 变成纯前缀缓存， │
    │                      │   → 块还池但 hash 可复用，   │   等 LRU 淘汰       │
    │                      │   重启后靠 prefix cache 找回│                     │
```

关键点：**"释放"不等于"数据作废"**。`free()` 只是把 block 的 ref_cnt 归零还进空闲队列，块上登记的 hash 还在 BlockPool 的 hash 表里；只要没被新分配覆盖或 LRU 淘汰，下一个同前缀请求通过 `get_computed_blocks` 就能零成本"复活"这些块——这就是 vLLM 前缀缓存的核心机制，也是 `block_hashes`（链式哈希）存在的意义。

### 五、和 memcache / AscendStore 如何衔接

| schedule() 触点 | AscendStoreConnector 做的事 |
|---|---|
| 触点2 `get_computed_blocks_for_connector` (:730) | 先查本地 GPU 前缀缓存；hybrid 模型还处理各 group 命中不一致（hit_diverged 时回退 `:759-767`） |
| 触点3 `connector.get_num_new_matched_tokens` (:744) | KVPoolScheduler → ZMQ → rank0 worker → `MemcacheBackend.exists` 查全局池，得到外部命中数 |
| 触点4 `allocate_slots(..., num_external_computed_tokens=..., delay_cache_blocks=load_async)` (:919) | 为外部命中的 token **先分配空壳 block**（memcache `get` 的目标地址）；`delay_cache_blocks=True` 时跳过 `cache_blocks` 提交，等异步传输完成 |
| 触点5 `connector.update_state_after_alloc(request, get_blocks(req_id), ext_tokens)` (:947) | 把分配好的 block_id 列表交给 connector → LoadSpec.can_load=True → worker 侧 `start_load_kv` 时 `MemcacheBackend.get` 就往这些块里灌数据 |

即：**KVCacheManager 管"块从哪来、放哪去"，MemcacheBackend 管"块里的数据从远端搬过来"，Scheduler 通过 allocate_slots 的两个参数（`num_external_computed_tokens`、`delay_cache_blocks`）和 connector 的两个回调（`get_num_new_matched_tokens`、`update_state_after_alloc`）把两者缝在一起。**

### 六、设计要点提炼

1. **统一的进度模型**：schedule() 没有 prefill/decode 之分（`:429-438` 的注释），每个请求只有 `num_computed_tokens` 追赶 `num_tokens_with_spec`——chunked prefill、前缀缓存、投机解码全是这个模型的特例；
2. **token 预算（软约束）与 block 分配（硬约束）分离**：budget 决定"算多少"，allocate_slots 决定"放不放得下"，两个循环都以这两个资源为界；
3. **抢占只发生在 RUNNING 循环**：running 请求可以杀别人求生（while True 重试），waiting 请求分配失败只是 break 等下轮——保证已跑请求不被饿死；
4. **先还后借**：`remove_skipped_blocks` 在检查容量之前调用，滑窗类模型能自我腾出空间；
5. **命中块的接管是原子的**：`allocate_new_computed_blocks` 在确认容量足够后才 touch 命中块，避免"查找时命中、分配时已淘汰"的竞态。
