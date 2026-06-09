# MNN PIC Server 系统流程与计算加速创新性分析

本文基于当前仓库源码梳理，重点不是接口说明，而是分析这套 PIC/PagedAttention server 在“计算加速”上的创新性支点、实际实现边界和实验验证路径。

## 一页结论

当前 `pic_server` 的核心设计是把长文档 prefill 拆成两个阶段：

1. **离线/持久 text cache 构建**：`POST /v1/prefill/text` 对文档做一次真实 prefill，把每层 KV 导出为 `.k/.v/.json`。key 在写盘前被转成 `canonical_no_rope`，value 保持 raw layout。
2. **请求内复用与稀疏重算**：`POST /v1/chat/completions` 中，prompt 必须显式包含 `{{pic_cache}}`。server 将 prompt 拆成 `prelude + PIC文档 + suffix`，再把持久 KV 作为 external segments 注入当前请求的 `PagedKVMeta/PagedCache`。后端 PagedAttention 在每层 attention 执行时把外部 KV hydrate 到 slot table 对应的位置，并对 key 按当前 logical position 重新施加 RoPE。

加速的主要来源不是 decode，而是 **prefill 侧跳过 PIC 文档 token 的 Transformer 层计算**：

- `full-reuse`：不重算 PIC token，只做磁盘读、hydrate、PagedCache 写入和 suffix prefill。
- `epic` / `explicit` / `cacheblend`：先 hydrate 全量 PIC KV，再只对选中的 PIC token 做 sparse recompute，其余 token 复用持久 KV。
- `cacheblend/delta-v`：当前代码有 native backend score/top-k 路径，score 来自请求内 full-reference PagedCache value 与持久 cached PIC value 的 mean-abs delta。

可以作为创新性讨论的强点是：

- **position-canonical persistent KV**：持久 key 去掉 RoPE，使同一份文档 KV 能复用于不同 prompt 位置。
- **request-scoped PagedCache 作为唯一 KV 交换层**：复用、重算、suffix attention 都通过 slot table/PagedAttention 执行，避免把 chat/scoring 做成 scratch KV 文件读写。
- **backend-native hydrate + sparse recompute + score/top-k**：CPU/CUDA/OpenCL 都有外部 KV hydrate；CUDA/OpenCL 的 cacheblend score/top-k 在 device kernel 中完成，最后只回传 selected indices。

需要谨慎表述的点是：

- `full-compute` 不是加速模式，它只是语义对齐 baseline。
- `cacheblend` 每个请求仍要做一次 score-layer full-reference pass，性能报告必须把 scoring 成本计入。
- 当前 `kvshare/delta-a` 在 server 侧没有独立 native selector；从 `buildExecutionPlan` 看，在没有 native selected indices 时会走 `full-compute-fallback`。
- 当前 sparse recompute 的 server 调用是 selected token 通过 `recomputeExternalPagedKV` 重新 forward；`score_layer_idx` 主要用于 cacheblend scoring，不等价于“只从某层开始重算”的完整分层重算实现。
- batch 是 request-scheduled sequential batch，不是 native paged batch。

## 关键源码位置

- Server 入口与路由：`transformers/pic_llm/engine/app/pic_server_main.cpp`，`transformers/pic_llm/engine/app/pic_server.cpp`
- LLM 对 server 暴露的 PIC/PagedKV 接口：`transformers/pic_llm/engine/include/llm/llm.hpp`
- LLM 侧 external KV / sparse recompute / cacheblend scoring：`transformers/pic_llm/engine/src/llm.cpp`
- 请求级 PagedKV 元数据：`source/core/PagedKVMeta.hpp`
- CPU PagedAttention：`source/backend/cpu/CPUPagedAttention.cpp`
- CUDA PagedAttention：`source/backend/cuda/execution/PagedAttentionExecution.cu`
- OpenCL PagedAttention：`source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
- OpenCL kernels：`source/backend/opencl/execution/cl/paged_attention_buf.cl`

## 系统总览

```text
Client
  |
  | 1. POST /v1/prefill/text
  v
PicServer
  | tokenize(text)
  | Llm::setPrefixCacheFile(...)
  | Llm::response(token_ids, max_new_tokens=0)
  v
PagedAttention PendingWrite
  | export per-layer KV:
  |   key: canonical_no_rope, [token,batch,kv_head,head_dim]
  |   value: raw, [batch,kv_head,token,head_dim]
  v
Disk text cache: meta.json + tokens.json + layers/*.k/*.v/*.json


Client
  |
  | 2. POST /v1/chat/completions with pic_cache + {{pic_cache}}
  v
PicServer
  | apply chat template
  | split prelude / PIC placeholder / suffix
  | load text cache refs into PagedKVExternalSegment
  | build execution plan
  v
Llm / PagedKVMeta
  | prefill(prelude)
  | appendExternalPagedKV(PIC token ids, external segments)
  | optional sparse recompute(selected PIC logical indices)
  | prefill(suffix)
  v
CPU/CUDA/OpenCL PagedAttention
  | hydrate external KV per layer into PagedCache slots
  | reapply RoPE to canonical key at current logical position
  | write current K/V and run attention through slot table
  v
Response
```

## Server 生命周期

`pic_server_main.cpp` 接收：

- `--config`: PIC LLM config
- `--host`, `--port`
- `--kv-cache-dir`: 默认 `.cache/kvshare/kvcache`
- `--model`: 对外暴露的模型名

`PicServer::load()` 创建 `MNN::Transformer::Llm`，把 `tmp_path` 和 `prefix_cache_path` 注入 LLM config，然后加载模型。`PicServer::start()` 注册 HTTP 路由：

- `GET /`
- `GET /healthz`
- `GET /v1/models`
- `POST /reset`
- `POST /v1/prefill/text`
- `POST /v1/kv/pic_caches`
- `POST /v1/chat/completions`
- `POST /chat/completions`

所有核心 JSON endpoint 都通过同一把 `mMutex` 串行进入 LLM。这保证了单 `Llm` 实例的状态安全，但也说明当前 server 不是并发 native batching。

## `/v1/prefill/text`: 持久 text cache 构建

输入只接受 inline text：

```json
{
  "id": "doc-1",
  "type": "text",
  "content": "actual document text",
  "force": false
}
```

流程：

1. 校验 `type=text`，拒绝 `path`，拒绝看起来像文件路径的 `content`。
2. 根据 `id/cache_name/backend` 生成 cache root：
   `kv_cache_dir/objects/<backend>/<cache_name>/`
3. tokenizer 编码文档，计算 content sha256。
4. 如果 `meta.json` 中 sha256、token ids、KV layout 和 layer files 都匹配，则直接返回 cache hit。
5. cache miss 时清理旧目录并创建 `layers/`。
6. 调用：
   - `mLlm->reset()`
   - `mLlm->setPrefixCacheFile(fileStem)`
   - `mLlm->response(tokenIds, &sink, "", 0)`
   - `mLlm->clearPrefixCacheFile()`
   - `mLlm->reset()`
7. LLM prefill 的第一轮发现 prefix cache 不存在时，把 `mMeta->file_flag` 设为 `KVMeta::PendingWrite`。
8. PagedAttention 后端在执行每层 attention 时看到 `PendingWrite`，把当前 PagedCache 导出到磁盘。
9. server 读取 sidecar shape，生成统一 `meta.json` 和 `tokens.json`。

持久 KV 的关键 layout：

```json
{
  "layout": "mnn_paged_attention_raw_v1",
  "key_layout": "[token,batch,kv_head,head_dim]",
  "value_layout": "[batch,kv_head,token,head_dim]",
  "key_rope_state": "canonical_no_rope",
  "rope_pairing": "half"
}
```

创新性含义：

- 这不是普通 decode KV cache，而是 **文档级 prefill KV 对象**。
- key 写盘前被 inverse RoPE 成 canonical 状态，后续可以在不同 prompt logical position 重新绑定 RoPE。
- 每层 `.k/.v/.json` 加 `meta.json/tokens.json`，使 cache 成为可验证、可组合的系统对象。

## `/v1/kv/pic_caches`: PIC cache metadata 组合

输入示例：

```json
{
  "id": "pic-1",
  "text_cache_refs": [
    {"id": "doc-1"}
  ],
  "selection_algorithm": "full-reuse",
  "pic_recompute_ratio": 0.2,
  "pic_recompute_score_layer_idx": 1
}
```

流程：

1. 校验只接受 `text_cache_refs/text_caches`，拒绝 legacy `merge_mode/segments`。
2. 每个 ref 只允许 `id/cache_name/meta_path`，不能带 `content/path`。
3. 读取 text cache `meta.json`，检查：
   - `format=kvshare-prefix-cache-meta-v1`
   - `type=text`
   - `layout=mnn_paged_attention_raw_v1`
   - `key_rope_state=canonical_no_rope`
   - `rope_pairing=half`
   - backend 与当前 runtime 匹配
4. 把多个 text cache 的 token ids 拼接，把每个 text cache 转成一个 `PagedKVExternalSegment`。
5. 返回 PIC metadata。

这个 endpoint 不重新计算 KV，也不写新的持久 KV。它主要把多个 text cache 组织成当前请求可用的 PIC 逻辑文档。

## `/v1/chat/completions`: 请求内复用和稀疏重算

### 无 `pic_cache`

无 PIC 时走普通 full-compute：

1. apply chat template
2. tokenizer encode
3. `mLlm->reset()`
4. `mLlm->generate_init()`
5. `mLlm->prefill(inputTokenIds)`
6. 如果 `max_tokens != 0`，再 decode

这条路径是基线，不是 PIC 加速。

### 有 `pic_cache`

有 PIC 时必须在 messages content 中包含 placeholder，默认 `{{pic_cache}}`。server 不支持 legacy 隐式插入。

流程：

1. `preparePicCacheFromRequest(...)` 解析 text cache refs，生成 token ids 和 external segments。
2. apply chat template。
3. 按 placeholder 拆成：
   - `preludeText`
   - `PIC token ids`
   - `suffixText`
4. tokenizer 编码 prelude/suffix，并做 tokenizer 空前缀裁剪。
5. 对 `cacheblend/delta-v`，先跑 native scoring：
   - 拼 full prompt: `prelude + PIC + suffix`
   - `mLlm->selectCacheBlendExternalPagedKV(...)`
   - LLM 加载 score-layer 截断 module
   - backend PagedAttention 在 score layer 计算 value delta 和 top-k
   - 返回 selected local indices
6. `buildExecutionPlan(...)` 生成执行计划。
7. 执行计划：
   - `fullCompute`: prefill `prelude + PIC + suffix`
   - 非 fullCompute:
     - prefill `prelude + plan.prefillPicTokenIds`
     - `appendExternalPagedKV(plan.externalTokenIds, plan.externalSegments)`
     - 如果需要，`recomputeExternalPagedKV(plan.sparseLogicalIndices, plan.sparseTokenIds)`
     - prefill `suffix`
8. 如果 `max_tokens != 0` 再 decode；性能分析通常设置 `max_tokens=0` 看 prefill-only。
9. 返回 OpenAI-compatible response，并在 `pic_cache/precision_recovery` metadata 中记录选择算法、重算 token、fallback 等信息。

## 执行模式与当前实现状态

| `selection_algorithm` | 当前计划 | 加速含义 | 当前代码状态 |
|---|---|---|---|
| `full-compute` | `native-full-compute` | 不读磁盘 KV，完整计算 prelude+PIC+suffix | 用于语义 baseline，不是加速 |
| `full-reuse` | `native-full-reuse` | 全量复用 PIC KV，只算 prelude/suffix | 已实现 |
| `epic` | `native-epic-sparse-recompute` | 按 ratio 选择 PIC 开头连续 token 重算 | 已实现 selection 和 sparse recompute；当前 server 执行不是显式分层重算 |
| `cacheblend` / `delta-v` | `native-cacheblend-sparse-recompute` | 请求内 score-layer value delta 选 top-k，再 sparse recompute | 已实现 native scoring/top-k 路径 |
| `explicit` | `native-explicit-sparse-recompute` | 由请求给出重算 logical/local indices | 已实现 |
| `kvshare` / `delta-a` | 期望 sparse recompute | 按当前代码，如果没有 native selected indices，会 `full-compute-fallback` | 需要补齐 native selector 后再强表述 |

注意：`score_layer_idx` 当前明确用于 `cacheblend/delta-v` 的 score-layer module。server 后续 `recomputeExternalPagedKV` 是把 selected token ids 送进 sparse query forward，当前接口没有显式表达“从 score layer 开始只重算后续层”的执行边界。因此论文或报告中如果要强调分层 sparse recompute，需要先确认实现是否已经补齐。

## PagedKVMeta: 请求内 KV 控制平面

`PagedKVMeta` 是 PIC runtime 的控制平面，关键字段包括：

- `request_active/request_capacity/logical_length`
- `slot_table_host/slot_table_version`
- `external_segments/external_loaded_layers`
- `sparse_query_active/sparse_query_logical_indices`
- `cacheblend_score_active/cacheblend_score_ready`
- `cacheblend_score_segments/cacheblend_score_selected_local_indices`

关键行为：

- `beginRequest(capacity)`：开始一个 request-scoped PagedCache 语义，初始化 slot table。
- `appendExternalSegments(...)`：把外部 text cache segments 接到当前 logical KV 序列后面。
- `beginSparseQuery(logicalIndices)`：让后续 forward 的 query token 写入指定 logical slots。
- `beginCacheBlendScoring(...)`：让指定 score layer 的 PagedAttention 执行 scoring 并返回 selected indices。
- `syncPaged()`：forward 后同步 logical length / previous / add/remove。

创新性含义：server 不直接拿 `.k/.v` 做 attention，而是把所有复用/重算决策落到 `PagedKVMeta`，由 PagedAttention 后端围绕 slot table 完成真实 KV 读写。

## PagedAttention 后端数据流

### 共同语义

CPU/CUDA/OpenCL PagedAttention 都遵循同一逻辑：

1. 根据 `PagedKVMeta` 计算 `baseLogical`、`insertLen`、`kvLen`。
2. 同步 slot table。
3. 如果存在 external segments 且当前 layer 尚未 hydrate：
   - 读取对应 layer 的 `.k/.v`
   - 校验 shape、dtype、`canonical_no_rope/half`
   - 把 external KV 写入当前 PagedCache slot
   - key 按当前 logical position 重新施加 RoPE
   - 标记当前 layer 已 hydrate
4. 把当前请求新产生的 K/V 写入 PagedCache。
5. 如果处于 sparse query，则新 K/V 写入 selected logical slots，覆盖对应 PIC token 的 KV。
6. 如果处于 cacheblend scoring layer，则比较当前请求 full-reference value 与 cached PIC value，选 top-k。
7. 从 slot table 指向的 PagedCache 执行 attention。

### CPU

CPU 后端直接读二进制 `.k/.v` 到 host vector：

- `_restoreExternalSegmentsCPU(...)` 对 canonical key 重新施加 RoPE，value 直接拷贝到 slot。
- `_runCacheBlendScoringCPU(...)` 在 host 上计算 mean-abs value delta，然后 top-k。
- `PendingWrite` 时把 key inverse RoPE 后写 `.k`，value 写 `.v`。

CPU 路径语义清晰，适合做 correctness reference。

### CUDA

CUDA 后端要点：

- external PIC KV 要求 direct mapped PagedCache。
- 支持 layer 级 async disk read/prefetch。
- hydrate 时要求 contiguous slots；value 通过 direct mapped target 写入，key 在 GPU kernel 中按 logical position 施加 RoPE。
- cacheblend scoring：
  - cached value segment 从磁盘/host staging 到 device workspace
  - `cacheBlendValueScoreKernel` 在 device 计算 score
  - `cacheBlendTopKKernel` 在 device 选 top-k
  - 只把 selected top-k indices 拷回 host
- `PendingWrite` 导出时，key 通过 `exportCanonicalPagedKeyKernel` 做 canonical export，value 从 PagedCache 拷回后写盘。

这条路径可以表述为 **device-side scoring/top-k with compact host feedback**，但不能表述为完全无 host/IO 参与，因为 cached value 仍来自磁盘并通过 host/device staging。

### OpenCL

OpenCL 后端与 CUDA 类似：

- external segments hydrate 到 mapped PagedCache。
- `pic_page_attention_hydrate_kv` kernel 对 key 重新施加 RoPE。
- `pic_cacheblend_value_score` 和 `pic_cacheblend_topk` kernel 完成 score/top-k。
- 最后读取 top-k indices 到 host。
- 同样有 `export_canonical_paged_key` 用于 text cache export。

## 加速机制的性能模型

设：

- `P`: PIC 文档 token 数
- `S`: suffix token 数
- `A`: prelude token 数
- `R`: sparse recompute token 数，约等于 `ceil(P * pic_recompute_ratio)`
- `L`: layer 数

粗略看：

- 普通 full-compute prefill：计算 `A + P + S` 个 token 的完整 Transformer。
- PIC full-compute：仍计算 `A + P + S`，但经过 PagedAttention/PagedCache，主要用于对齐 correctness 和测 PagedAttention overhead。
- full-reuse：跳过 `P` 个 PIC token 的层计算，保留：
  - 读取持久 KV
  - hydrate 到 PagedCache
  - suffix token 的 prefill
  - suffix attention 仍会 attend 到 PIC KV
- sparse recompute：
  - hydrate 全量 PIC KV
  - 重算 `R` 个 selected PIC token
  - suffix prefill
  - 对 cacheblend，还要加一次 score-layer full-reference pass 和 top-k

所以真正的加速应主要出现在：

```text
P 很大，S 较小，R/P 较低，hydrate+IO 成本低于跳过的 Transformer 计算成本
```

如果 `S` 很大，或者每个请求 cacheblend scoring 成本很高，收益会下降。

## 可主张的创新点

### 1. 位置无关的持久 KV 表示

普通 KV cache 往往绑定在生成过程的历史位置上。这里把持久 key 存成 `canonical_no_rope`，请求时按当前 logical slot 重新施加 RoPE。这样一份文档 KV 可以被放到不同 prompt 的不同位置，与不同 prelude/suffix 组合。

创新性关键词：

- position-canonical KV
- RoPE-decoupled persistent cache
- reusable document prefill state

### 2. 持久 cache 边界和请求内推理边界分离

`/v1/prefill/text` 是唯一持久写盘入口，chat 请求只消费 text cache，不生成 scratch reference KV。这让系统流程更像“文档 KV 对象 + 请求调度”，而不是传统 prompt cache 的临时复用。

创新性关键词：

- persistent text-KV object
- request-scoped inference boundary
- no scratch KV scoring path

### 3. PagedCache 是唯一真实 KV 交换层

external KV、当前请求新 KV、sparse recompute KV、suffix attention 都通过 PagedCache/slot table 交互。这个点非常重要，因为它把复用逻辑落到了 runtime/backend，而不是在 server 侧拼 tensor 或读盘比较。

创新性关键词：

- PagedAttention-native KV reuse
- slot-table based external KV hydration
- backend-consistent sparse KV overwrite

### 4. 稀疏重算恢复精度

full-reuse 最大化跳过计算，但可能损失和完整上下文重新 prefill 的一致性。`epic/cacheblend/explicit` 提供了一个中间层：大部分 PIC KV 复用，只重算少量 token。

创新性关键词：

- sparse prefill recomputation
- selected-token precision recovery
- reuse/recompute tradeoff

### 5. 后端侧 scoring/top-k

`cacheblend/delta-v` 当前不是把完整 score vector 拷回 CPU 排序，而是在 CPU/CUDA/OpenCL PagedAttention 执行 score/top-k，其中 CUDA/OpenCL 使用 device kernels，最终只回传 selected local indices。

创新性关键词：

- backend-native CacheBlend scoring
- compact top-k scheduling feedback
- device-side score/top-k for edge runtime

### 6. 面向边缘推理的 server 化落地

这不是单纯 Python prototype，而是 standalone C++ HTTP server，支持 OpenAI-compatible chat endpoint，并覆盖 CPU/CUDA/OpenCL PagedAttention runtime。

创新性关键词：

- edge-deployable PIC runtime
- C++ native MNN integration
- multi-backend reusable KV

## 不建议过度主张的点

1. **不要说 full-compute 加速。** full-compute 是对齐语义和测 overhead 的 baseline。
2. **不要说 cacheblend 只剩少量重算成本。** 当前每个 cacheblend 请求包含 score-layer full-reference pass。
3. **不要说 kvshare/delta-a 已完整 native 实现。** 当前 server 未调用独立 kvshare selector，没有 native indices 时会 fallback。
4. **不要说 CUDA/OpenCL scoring 完全无 host。** score/top-k 在 device，但 cached value 读取和最终 indices 仍涉及 host/device 传输。
5. **不要说 batch native。** 当前 response metadata 标注 `native_paged_batch=false`。
6. **不要说 sparse recompute 是严格从 score layer 开始重算。** 当前 server 接口没有显式分层重算边界。
7. **不要把 decode 加速混入 prefill 加速。** 当前分析重点应设置 `max_tokens=0`。

## 建议的创新性表述草案

可以这样概括：

> 本实现提出并落地了一套面向边缘 LLM prefill 加速的 PIC/PagedAttention runtime：首先将长文档 prefill 的 per-layer KV 持久化为 position-canonical raw KV；请求时通过 PagedKVMeta 和 slot table 将外部文档 KV hydrate 到 request-scoped PagedCache，并在 backend PagedAttention 中按当前 logical position 重新施加 RoPE。该设计把长上下文复用从传统 prompt cache 的串行历史状态中解耦出来，使文档 KV 能跨请求、跨 suffix 和跨 logical position 复用。进一步地，系统支持 selected-token sparse recompute 与 backend-native CacheBlend score/top-k，在 full-reuse 与 full-compute 之间提供可调的速度/精度折中。

更强的论文式 claim 需要实验支撑：

- full-reuse 相比 PIC full-compute 和普通 full-compute 的 prefill latency 加速比。
- cacheblend 在计入 scoring 后，低 ratio 下是否仍显著优于 full-compute。
- 质量指标证明 sparse recompute 比 full-reuse 更接近 full-compute。
- CUDA/OpenCL native scoring/top-k 相比 CPU score/top-k 或 scratch KV 方案的端到端优势。

## 实验验证建议

### 必备 baseline

1. **Normal LLM full-compute prefill**
   - 使用普通 MNN LLM 模型，不是 PIC 模型。
   - `llm_bench -n 0` 或等价 prefill-only。

2. **PIC full-compute prefill**
   - 使用 PIC/PagedAttention 模型。
   - `selection_algorithm=full-compute`。
   - 用来量化 PagedAttention/PagedCache overhead。

3. **PIC full-reuse prefill**
   - 先 `/v1/prefill/text` 构建 text cache。
   - chat 使用 `selection_algorithm=full-reuse`，`max_tokens=0`。

4. **Sparse modes**
   - `epic` / `cacheblend` / `explicit`
   - ratio 建议：`1%/5%/10%/20%/30%`
   - 每个 ratio 独立请求，不能复用 score vector 或 selected indices。

### 必报倍数

- `PIC full-compute / normal LLM full-compute`
- `full-reuse / PIC full-compute`
- `full-reuse / normal LLM full-compute`
- `cacheblend(ratio) / full-reuse`
- `cacheblend(ratio) / PIC full-compute`
- `cacheblend(ratio) / normal LLM full-compute`

### 延迟拆解

建议用 profile/NVTX/日志拆：

- text cache cold build latency
- disk read / async read wait
- hydrate external KV
- suffix prefill
- score-layer full-reference pass
- score/top-k
- sparse recompute
- decode latency，若 `max_tokens>0`

### 质量指标

至少比较：

- full-compute 输出
- full-reuse 输出
- sparse recompute 输出

可用任务：

- HotpotQA / 2Wiki / MuSiQue 等多文档 QA
- SAMSum / MultiNews 等摘要任务

建议报告：

- EM/F1/ROUGE 或任务原有 metric
- selected token ratio
- latency-quality curve

## 当前实现风险与待补齐点

1. **kvshare/delta-a selector**
   - 当前 server 没有和 cacheblend 类似的 `selectKvshareExternalPagedKV`。
   - 如果要主张 kvshare 创新，需要补齐 native scoring/top-k 或明确标注为未来工作。

2. **分层 sparse recompute**
   - metadata 里有 `score_layer_idx` 和 `pre_score_compute_layers` 的语义，但当前 server 执行没有显式“score layer 前/后分段重算”。
   - 如果论文想强调 layer-plan sparse prefill，需要核对或补齐实现。

3. **CUDA/OpenCL zero-copy/contiguous slot 约束**
   - CUDA/OpenCL hydrate 目前要求 direct mapped PagedCache 和 contiguous slots。
   - 这对当前 request_base=0 的线性请求没问题，但对更复杂 slot table/continuous batching 需要进一步支持。

4. **磁盘 IO 成本**
   - full-reuse 的理论优势可能被 KV 文件读取吞掉。
   - 需要报告 cold/warm cache、不同 doc length、不同 storage 的影响。

5. **server 并发与 batch**
   - 单 `Llm` + mutex 串行执行，batch 也是逐 item 调度。
   - 如果要讨论服务吞吐创新，需要另做并发/批处理设计。

6. **prefill-only 与 decode**
   - 当前加速主轴是 prefill。
   - decode refinement metadata 目前 disabled，不应作为当前实现贡献点。

## 最适合强调的创新路线

如果目标是分析“计算加速的创新性”，建议按这个主线组织：

1. **问题**：长文档多次被不同问题/后缀引用时，重复 prefill 文档 token 是主要浪费。
2. **核心想法**：把文档 prefill KV 变成可持久、可位置重绑定的 cache object。
3. **系统关键**：请求时不拼 scratch KV，而是 hydrate 到 PagedCache slot table，使后续 suffix attention 和 sparse recompute 都走原生 backend。
4. **速度/精度折中**：full-reuse 跳过所有 PIC token 计算；sparse recompute 只为少量重要 PIC token 恢复精度。
5. **落地优势**：C++ MNN runtime、多后端、OpenAI-compatible server、可在 Jetson/edge 环境跑。
6. **实验证明**：端到端 prefill latency、scoring 成本计入后的 ratio 曲线、质量恢复曲线。
