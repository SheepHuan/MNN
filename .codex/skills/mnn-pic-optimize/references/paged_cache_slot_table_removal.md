# PagedCache `slot_table` 间接寻址去除 — 影响分析

Status: implemented smoke + sparse-row hardening (2026-07-03). Functional compile/smoke passed; performance not accepted yet.
Scope: MNN PIC PagedAttention CPU / CUDA / OpenCL backends.
Related: `references/decode_opencl_workflow.md` (build/bench workflow),
`log/orangepi/2026-07-03-01/decode_optimization_plan_v6.md` (V6 plan, §4.2).

## TL;DR

`slot_table` 在当前所有 PIC 流程里**恒等**（`slot_table[k] == k` for all in-range k）。
去除它在**功能上零影响**：prefill 的 Q 截断靠 `sparse_query`（紧凑 active logical 索引），
持久 PIC cache 源的 hydrate 靠 kernel arg（`_picCacheSourceSlotBase`），两者都不依赖 slot_table。
去除后每层每步省一次 `slot = slot_table[k]` gather + 越界分支，并塌缩一整套
`decodeKeyReadyPrefixSlots` / `slotTableVersion` 重建判断逻辑。风险很低，唯一代价是放弃
"未来非恒等 slot 映射"这个目前未使用的扩展位。

Implementation note (2026-07-03): deleting `slot_table_version` also removed an implicit
cross-request discriminator for async persistent PIC cache read task keys. Request tokens are
still sequential in PagedCache, so slot removal itself should not corrupt logical addressing, but
async task reuse must still be invalidated across requests. The implementation therefore adds
`PagedKVMeta::request_generation` as a pure request/key nonce and uses it in CUDA/OpenCL
`externalLayerRequestKey`; it does not restore logical->physical indirection.

Sparse-row hardening (2026-07-03): after removing slot_table, Q row selection is guarded
explicitly instead of relying on implicit shape agreement. `PagedKVMeta::beginSparseQuery` now
requires strictly increasing active logical indices within `[0, logical_length)`, and
`validateSparseQueryRows(activeLen, queryLen, allowFullQueryRows)` is called by CPU/CUDA/OpenCL.
Before writing sparse K/V, full-Q input is rejected (`allowFullQueryRows=false`); after the score
layer emits active indices, full-Q/compact-output is allowed only for `PicScoreAttention` when
`queryLen > activeLen`. Later `PicSparseAttention` layers must consume compact Q/K/V rows where
`queryLen == activeLen`. Kernels then use:

```text
qLogical = sparse_query[q]
qRow     = score-layer full-Q ? qLogical : q
slot     = logical
```

This is the required transition from full context length to selected-token length while preserving
K/V reads by the original logical position.

## 1. 现状：slot_table 恒等于 identity

`slot_table` 是 `SharedPagedCache::slotTable`，一块 `[max_slots]` int32 device buffer。
`slot_table[k]` 本意是 "logical position k → physical slot"。但源码里它**永远是恒等**：

```text
source/core/PagedKVMeta.hpp:
  :68   int request_base = 0;                       // 默认 0
  :111  request_base = 0;                           // beginRequest() 硬编码 0
  :141  slot_table_host[i] = request_base + i;      // = i
  :159  slot_table_host[i] = request_base + i;      // = i
  :647  slot_table_host[i] = request_base + i;      // = i
```

`request_base` 全仓库**只有这两处赋值，都是 0**（grep `request_base\s*=` 在 `source/` 下
除 `PagedKVMeta.hpp` 外无其他写入点）。`slot_table_host[i] = request_base + i = i`。

`_slotTableIsIdentity`（`PagedAttentionBufExecution.cpp:1518-1528`）判
`meta->request_base == 0 && size >= requiredSlots` —— 永真。代码注释自己也写明
（`:1525-1527`）：

> `PagedKVMeta currently only materializes identity logical->physical slots`
> `from request_base. Avoid scanning the slot table on every decode layer.`

所以 `slot_table[k] == k` 对所有 in-range k 成立。**现在每层每步那个
`slot = slot_table[k]` 的 gather + 越界检查，是纯开销，没换任何东西。**

## 2. prefill 的 Q 截断靠 sparse_query，不是 slot_table

PIC prefill（cacheblend / epic / sparse）的 Q 截断链路完全走 `sparse_query`：

```text
score layer (PicScoreAttention, mPicAttentionMode == 1):
  full Q 输入 → 计算 score/top-k → 产出 active_indices (一组 logical position)
  → 写进 mMeta->sparse_query_logical_indices

后续 sparse layer (PicSparseAttention):
  Q 被 rearrange_q_sparse kernel 按 sparse_query 截断:
    attention_buf.cl:370   q_row0 = sparse_query[q_index0]   // 紧凑 active logical 索引
    attention_buf.cl:380   query[b*input_seq_len + q_row]    // gather 出该 logical 行
  → compact Q (长度 = active_count) 喂给 sparse_flash kernel

sparse_flash kernel 内部读 K/V:
    past_key[key_base + dim*key_max_len + k_index]   // 直接按 logical k 读
  （past_key 是提前 pack 好的连续 buffer，不是 paged cache）
```

关键事实：

- **sparse_flash kernel 根本不接 slot_table 参数**。`attention_buf.cl` 的
  `sparse_flash_attention_row64/32` 和所有 `mqtile_sparse_flash_*` 都没有 slot_table arg，
  K/V 按 logical `k` 直接索引 `past_key`/`past_value`。
- slot_table 只在更早的 `pack_paged_k_prefill` / `pack_paged_kv_prefill` 阶段用一次，
  把 paged cache 里的 K/V 按 `slot_table[logical]`（= logical）拷进连续的
  `past_key`/`past_value`。恒等下这步 pack 就是纯 memcpy。

**所以 Q 截断的逻辑（`sparse_query` / `active_indices`）完全不受去除影响**。去除间接寻址
只动 "K/V 怎么落盘、怎么读"，不动 "Q 选哪些行"。

## 3. 持久 PIC cache 源的 hydrate 也不靠 slot_table

持久 cache 源 hydrate 到 "保留 physical source slots"，但这不依赖 slot_table：

- 源 segment 落在物理区间 `[sourceBase, sourceBase + sourceSlots)`，其中
  `sourceBase = align(max(kvLen, request_capacity), 32)`（`_picCacheSourceSlotBase`，
  `PagedAttentionBufExecution.cpp:1808-1813`）—— **在 `[0, kvLen)` 逻辑区之外**，
  靠 kernel arg（`keySourceLogicalStart` / `valueSourceStart`）寻址，不靠 slot_table。
- hydrate kernel `pic_page_attention_hydrate_kv`（`paged_attention_buf.cl:423`）把源区数据
  拷到逻辑区 `[0, kvLen)` 的真实 slot：`slot = slot_table[logical]`（`:463`）= `logical`。
  即目标就是 logical 位置本身。

去掉 slot_table 后，hydrate 直接写 `slot = logical`，行为完全一致。源区寻址本来就不靠它。

## 4. 影响范围：要改的地方（机械替换，无语义变化）

### 4.1 kernel 层

`paged_decode_attention_buf.cl` / `attention_buf.cl` / `paged_attention_buf.cl` 共 ~17 个
kernel 带 `__global const int *slot_table` 参数：

```text
decode / transposed-K:
  transpose_paged_key_to_decode_key          (paged_decode_attention_buf.cl:21)
  append_decode_key_value_hd128              (:51)
  append_sparse_decode_key_value_hd128       (:88)
  DEFINE_DECODE_CAUSAL_TRANSPOSED_K_FUSED_KV_HD128          (:560, q_slot = slot_table[q_logical])
  DEFINE_DECODE_CAUSAL_TRANSPOSED_K_FUSED_KV_GQA_HD128      (Step 1 新增, q_slot)
  DEFINE_DECODE_CAUSAL_TRANSPOSED_K_QTILE_HD128             (:973, 已有 identity_slot 分支 :1147-1153)
  decode_causal_attention_row{32,64}         (attention_buf.cl:3376/3512, slot = slot_table[k])
  decode_attention_pic_rank_score_hd128      (attention_buf.cl:3647, slot = slot_table[logical])

prefill pack / hydrate / score / export:
  copy_paged_kv                              (paged_attention_buf.cl:28)
  pack_paged_kv_prefill                      (:69)
  pack_paged_k_prefill                       (:270)
  pack_paged_k_prefill_to_image              (:331)
  pic_page_attention_hydrate_kv              (:423)
  export_canonical_paged_key                 (:517)
  pic_cacheblend_value_score                 (:607)
  pic_cacheblend_value_score_cached_image    (:651)
  paged_attention_row / paged_attention      (:947 / :1053)
```

每个 kernel 的改法：

- 删除 `__global const int *slot_table` 参数；
- `slot = slot_table[k]` → `slot = k`；
- 删除 `if (slot < 0 || slot >= key_max_len) continue;` 越界 guard（logical 已在上游
  range-checked）；
- qtile 宏的 `identity_slot` 分支（`:1147-1153`）变成无条件 `value_slot = value_logical`。

**已有模板**：`matmul_qk_decode_paged_identity_hd128`（`paged_decode_attention_buf.cl:239`）
已经不带 slot_table、硬编码 logical==physical —— 就是去除后的样子。

### 4.2 host 层

`PagedAttentionBufExecution.cpp` / `.hpp`：

- 删 `SharedPagedCache::slotTable` tensor（`hpp:26`）+ 分配（`:2713, :2721`）+
  `syncSlotTable`（`:2776-2808`）及其调用（`:4008, :8515`）；
- 删 ~30 处 `openCLBuffer(mCache->slotTable.get())` setArg（含 `3496, 3945, 4175, 4223,
  4983, 4998, 5015, 5581, 5817, 6252, 6377, 6487, 6522, 6558, 6622, 7016, 7059, 7111,
  7227, 7286, 7564, 7816, 8183, 8629, 8737` 等）；
- 删 `_slotTableIsIdentity`（`:1518`）、`_slotTablePhysicalSlotAt`（`:1530`）、
  `_slotTablePrefixMatches`（`:1543`）、`_captureSlotTablePrefix`（`:1567`）、
  `decodeKeyReadyPrefixSlots`（`hpp:37`）、`slotTableVersion`（`hpp:33`）、
  `slotTableLength`（`hpp:34`）整套 —— 这些只为检测 slot_table 变化要重建 decodeKey，
  恒等下永远 match，整套塌缩；
- `PagedKVMeta.hpp`：删 `slot_table_host`（`:74`）、`slot_table_version`、`physicalSlot`
  （`:653-658`），及 `beginRequest`/`reserveRequestCapacity`/`ensureLogicalCapacity` 里
  对 `slot_table_host` 的初始化（`:139-143, :156-161, :644-649`）；
- `physicalSlot(l)` 的两处 host 调用（`PagedAttentionBufExecution.cpp:2282, :2290` 预取窗口、
  `:8776` 导出 canonical key）直接替换为 `l`。

### 4.3 自动包含的简化

- `directValuePrefill` 快路径（`:4926-4928`）当前 gate 在 `_slotTableIsIdentity`，去除后
  gate 永真，可无条件化。
- V6 Step 2（identity-slot V 快路径）整个被包含 —— 不再需要给 q=1 family 加 `identity_slot`
  arg，因为 slot_table 本身没了。

## 5. 收益

1. **每层每步省一次 gather + 越界分支**。长上下文（kv_len 512~3072）下，decode attention
   内层 K 循环每 iteration 都 `slot = slot_table[k]` + `if (slot<0||slot>=max) continue`。
   去掉后直接 `slot = k`，分支预测压力和寄存器占用都降。这是 normal LLM 路径快的原因之一
   （normal 用连续 `past_key`/`past_value`，无间接寻址）。
2. **省 slotTable buffer 带宽和同步**。`syncSlotTable` 每次请求 H2D 上传 `[max_slots]` int32
   （max_slots 可达 4096+），decode 每层还要 setArg 绑这个 buffer。
3. **decodeKey 重建判断塌缩**。`decodeKeyReadyPrefixSlots` / `_slotTablePrefixMatches` 这套
   （为应对 slot_table 版本变化要重转置历史 K）在恒等下永远 true，整套逻辑可删，decode 准备
   路径更简单。
4. **prefill pack 变纯 memcpy**。`pack_paged_k_prefill` 的 `slot = slot_table[logical]` 变
   `slot = logical`，pack kernel 可更简单，结合 `directValuePrefill` 已有的 identity 快路径
   能跳过 pack。
5. **源码量净减少**。删一套 tensor + 一套 sync + 一套 prefix-tracking + ~17 个 kernel arg，
   后续维护面变小。

## 6. 风险

- **放弃 "未来非恒等 slot 映射" 扩展位**。slot_table 是 "logical → 物理 slot 复用" 的挂钩点。
  去除等于放弃未来支持真正的 paged 碎片管理 / slot 复用 / 虚拟映射。但当前 SKILL 方向是
  "PagedCache 是唯一真实 KV owner + 持久源 hydrate 到保留 slot"，逻辑区始终 `[0, kvLen)` 连续，
  没有碎片管理需求。**如果未来不做 slot 复用，去除是净赚。**
- **Hydrate 的源区/逻辑区分离**靠 kernel arg（`_picCacheSourceSlotBase`），不靠 slot_table，
  不受影响（已确认）。
- **`physicalSlot(l)` 的 host 调用**在恒等下 `= l`，直接替换无语义变化。
- **正确性回归面**：改动跨 ~17 kernel + host，必须全矩阵 A/B（OrangePi + Rhino × 3 模型 ×
  q=1/2/4/6/8）验证 TPOT 不退化 + P0 clean（`decode_prepare_inside_decode=0`）。

## 7. 验证计划

按 `references/decode_opencl_workflow.md` 的口径：

1. 改完 `opencl_codegen.py` → `git diff --check` → build OrangePi + Rhino → `file` → rsync。
2. profile smoke（profile on）→ P0 validity scan clean → 确认所有 decode/prefill op 仍正常 fire，
   无 slot_table 相关 setArg 报错。
3. 正式 TPOT A/B（profile off）：repair qtile `old`（`MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE=0`，去除前基线）
   vs `new`（去除后，`MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE=1`）。**通过条件：全矩阵 `new <= old`（持平或微升），无 `failures.csv`，
   P0 clean。**
4. 特别检查 cacheblend/epic prefill 路径（Q 截断 + hydrate） correctness —— 跑一次
   `run_pic_prefill_latency_sweep` 确认 sparse prefill 输出与去除前一致（精度不坏）。

## 8. 2026-07-03 实际验证结果

Static checks passed:

```bash
git diff --check --cached
rg -n "slot_table|slotTable|slot_table_host|slot_table_version|physicalSlot|physicalSlots|request_base|_slotTable|syncSlotTable|decodeKeyReadyPrefixSlots|decodeKeySlotTableVersion|slotTableVersion|slotTableLength" \
  source/core/PagedKVMeta.hpp source/backend/cpu source/backend/cuda source/backend/opencl source/shape transformers/pic_llm -S
```

Exporter / shape impact:

```bash
rg -n "slot_table|slotTable|request_base|physicalSlot|slot_table_host|slot_table_version" \
  transformers/pic_llm source/shape -S
```

No matches. Export graph does not need a slot-table removal change.

Artifact builds completed and file checks passed after the sparse-row hardening:

```text
OrangePi 5 Plus OpenCL/Vulkan:  pic_server + libMNN_CL.so + libMNN_Vulkan.so
Jetson CUDA cross build:        pic_server + libMNN_Cuda_Main.so
Rhino/Aidlux Adreno OpenCL:     pic_server + libMNN_CL.so
```

Benchmark gate was not accepted. MiniCPM5-1B ctx512 full-reuse after the request-generation fix:

```text
device     benchmark.csv baseline    measured after removal      best ratio
jetson     0.313210591 s             0.481185 / 0.472853 s       1.510x slower
orangepi   0.429257000 s             0.624368 / 0.721490 s       1.455x slower
rhino      0.289987000 s             warm failed: empty reply / SSH timeout
```

These rows were not merged into `benchmark.csv`. OrangePi profile still showed native direct async
hydrate (`async_read=1 direct_segments=1 fallback_tokens=0`), so the regression is not explained by
request-generation breaking persistent-cache prefetch. Treat it as a performance isolation task,
not evidence that identity slot mapping is required.

## 9. 与 V6 计划的关系

去除 slot_table 是 V6 §4.2（identity-slot V 快路径）的**彻底版**：Step 2 原本只是给 q=1 family
加 `identity_slot` arg 跳过 gather，但既然 slot_table 恒等，不如直接全局去除，连 env gate 都不用。

建议作为 V6 的 **Step 2'**（替换原 Step 2）执行：一次改完所有 kernel + host，不再需要
`identity_slot` arg。这一步本身不期望大加速（slot_table gather 在 L1 命中率高），但简化了
后续所有步骤（Step 3 qtile、Step 5 image、Step 6 autotune 都不再需要处理 slot_table 分支）。
