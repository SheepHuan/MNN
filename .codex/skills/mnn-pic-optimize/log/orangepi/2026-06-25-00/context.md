# OrangePi 2026-06-25 00 Long Context

以下内容从旧的 `OPTIMIZATION_LOG.md` 迁移，按本设备/日期归档。

## 2026-06-25 Investigation: MiniCPM5-1B / Llama3.2 3B / Qwen3-8B OpenCL Sparse Slow Path

Goal:

- Determine whether the very poor OpenCL `cacheblend` / `epic` numbers seen for `MiniCPM5-1B` are model-specific, and whether the current OpenCL fast-prefill implementation is the direct root cause.

Evidence:

```text
MiniCPM5-1B 1024 / OrangePi OpenCL:
  normal=4.54s, pic-full=16.06s, full-reuse=0.53s
  cacheblend 5%=3.94s, 10%=4.26s, 50%=21.72s
  epic       5%=3.85s, 10%=4.08s, 50%=13.49s

MiniCPM5-1B 1024 / Rhino OpenCL:
  normal=2.32s, pic-full=2.57s, full-reuse=0.33s
  cacheblend 5%=5.43s, 10%=6.63s, 50%=24.11s
  epic       5%=5.22s, 10%=5.42s, 50%=24.46s

Llama3.2 3B 1024 / Jetson CUDA:
  normal=3.76s, pic-full=4.17s, full-reuse=1.12s
  cacheblend 5%=0.50s, epic 5%=0.49s

Llama3.2 3B 1024 / OrangePi OpenCL:
  normal=15.90s, pic-full=36.00s, full-reuse=1.11s
  cacheblend 5%=7.20s, epic 5%=6.73s

Llama3.2 3B 1024 / Rhino OpenCL:
  normal=7.74s, pic-full=8.19s, full-reuse=0.73s
  cacheblend 5%=8.89s, epic 5%=8.19s

Qwen3-8B 1024 / Rhino OpenCL targeted rerun on fixed ports 18133/19133:
  prefill/text cache_status=reused, 0.041s
  full-compute=20.85s, full-reuse=1.89s
  cacheblend 5%=23.02s, 10%=34.84s
  epic       5%=12.51s, 10%=14.22s
```

Observed pattern:

- This is not a MiniCPM-only issue. `Llama3.2 3B` and `Qwen3-8B` show the same OpenCL sparse-path pathology.
- `full-reuse` stays fast on all three models, so hydrate + suffix is not the main bottleneck.
- The regression is concentrated in sparse prefill (`cacheblend` / `epic`), and it is large enough that low-budget sparse requests can be slower than PIC `full-compute`.
- Jetson CUDA does not show the same behavior on the same workload class.

Implementation findings:

- The OpenCL prefill trial policy already treats these exact model shapes as special long-context tuples:
  - `MiniCPM5-1B`
  - `Llama-3.2-3B-Instruct`
  - `Qwen3-8B`
  See `source/backend/opencl/execution/buffer/AttentionBufExecution.cpp`.
- In OpenCL `PagedAttentionBufExecution`, dense fast prefill is explicitly disabled for sparse query:

```cpp
if (sparseQuery) {
    return false;
}
```

- OpenCL sparse fast prefill is also gated to `mHeadDim == 64`:

```cpp
if (mHeadDim != 64 || mNumHead % mKvNumHead != 0) {
    return false;
}
```

- Dispatch order in OpenCL is:
  - decode causal path
  - `runFastPrefill()` when `canUseFastPrefill(...)`
  - `runSparseFastPrefill()` when `canUseSparseFastPrefill(...)`
  - otherwise `row` kernel or generic kernel
- Because `MiniCPM5-1B`, `Llama3.2 3B`, and `Qwen3-8B` all use `headDim=128`, their sparse requests do not hit the current OpenCL sparse fast-prefill path. They also cannot use the dense fast-prefill path once `sparseQuery` is active. So they fall through to the slower `row` / `generic` path.

CUDA comparison:

- CUDA also keeps a headDim-64 sparse flash tile path, but unlike OpenCL it still has a general q-split prefill accelerator after that check.
- In `source/backend/cuda/execution/PagedAttentionExecution.cu`, if the sparse tile path does not apply, CUDA still enters `prefill_attention_fast_qk_softmax_qkv` for `attnLen > 1`, and that path accepts sparse query metadata (`sparseQueryDevice`, `queryRowsAreFull`).
- This means Jetson does not need the headDim-64 sparse flash tile path to stay fast on `headDim=128` sparse prefill. OpenCL currently does.

Conclusion:

- The main issue is not that the current OpenCL sparse fast-prefill kernel is miscomputing these models. Most bad `MiniCPM5-1B` / `Llama3.2 3B` / `Qwen3-8B` sparse requests never reach that path.
- The real gap is that OpenCL has no headDim-128 sparse fast fallback analogous to CUDA's q-split QK/softmax/QKV prefill path.
- Score layer is hit especially hard because its `full-Q + compact-output` sparse shape also cannot use the existing OpenCL headDim-64 fast path.
- OrangePi has an additional line to optimize: PIC `full-compute` is already much slower than normal full-compute for `MiniCPM5-1B` and `Llama3.2 3B`, so dense PIC full-path overhead should be treated separately from sparse-path collapse.

Kernel-level bottleneck:

- The OpenCL fallback kernels are much more naive than the dense fast-prefill path:
  - `paged_attention_row` does two full score sweeps over `valid_len`, and each sweep does a scalar `head_dim` loop. It also keeps a scalar `acc[256]` and accumulates value row-by-row.
  - `paged_attention` does the same work per output channel, again with two passes over `valid_len`.
- For sparse query this means the fallback cost is still close to:

```text
O(attnLen * validLen * headDim)
```

  with very limited reuse, poor vectorization, and repeated query/key loads.
- The kernels do honor causal truncation through:

```cpp
valid_len = min(kv_len, q_logical + 1)
```

  but for cacheblend and for suffix rows in epic, many selected logical positions still reach near the end of the prompt, so `valid_len` remains close to full context.
- This explains why `full-reuse` can be fast while sparse attention is still very poor: hydrate is not the issue, the attention kernel choice is.

Why Rhino and OrangePi differ:

- Rhino/Adreno has an extra dense-only optimization path:
  - `_useAdrenoGemmFullPrefill(...)` is enabled only for Adreno, only for dense full prefill, and only when `maskKeyLen == kvLen`.
  - `runFastPrefill()` can then enter `runAdrenoGemmPrefill()` with q-split GEMM-style kernels.
- Sparse requests never enter this path because OpenCL dense fast prefill rejects `sparseQuery`.
- So Rhino can still look healthy on `normal-full-recompute` or `pic-full-recompute`, but collapse on `cacheblend` / `epic` because sparse requests drop from Adreno GEMM/q-split back to the naive fallback kernels.
- OrangePi lacks the Adreno-only GEMM branch and also shows extra PIC full-path overhead, so it suffers both from sparse fallback and from slower dense PIC full-compute.

Borrowable ideas from CUDA / current OpenCL implementation:

1. Keep the current graph/runtime boundary:
   - OpenCL already runs `runCacheBlendScoring()` before attention dispatch, then writes active indices into `PagedKVMeta` and `mCache->sparseQuery`.
   - This means the sparse active set is already available before selecting the attention kernel. No control-flow rewrite is needed first.
2. Reuse the existing q-split dense fast-prefill structure for sparse headDim-128:
   - `runFastPrefill()` already has `rearrange_q`
   - `pack_paged_kv_prefill`
   - `matmul_qk_div_mask_prefill_piece`
   - `softmax_v4_buf`
   - `matmul_qkv_prefill_piece`
   - This is the most practical base to extend, instead of starting from a brand-new flash kernel for headDim-128.
3. Add sparse-query awareness to the q-split path:
   - gather/rearrange compact active query rows from either:
     - full query rows (`queryRowsAreFull == 1`, score layer), or
     - already compact query rows (later sparse layers)
   - feed `attnLen` compact rows into the piece kernels
   - keep packed K/V from current request PagedCache
4. Borrow the existing sparse-flash piece planning:
   - current OpenCL sparse flash already builds range-aware pieces and per-piece `activeKvLen`
   - apply the same piece planner to the headDim-128 q-split path so each q-piece only scans:

```text
activeKvLen = max(selected_logical_in_piece) + 1
```

   - this is the direct way to reduce K work for scattered cacheblend rows without changing algorithm semantics.
5. Treat score layer separately but on the same base:
   - score layer already has scoring/top-k done before kernel dispatch
   - the remaining gap is only that `full-Q / compact-output` cannot use the headDim-64 sparse flash path
   - a sparse q-split path that supports `queryRowsAreFull` closes that gap without introducing another special fallback.

Next actions:

1. Add a headDim-128 sparse q-split fast path on top of the existing dense `runFastPrefill()` kernel family.
2. Support both:
   - score-layer `full-Q / compact-output`
   - later-layer compact-Q sparse attention
3. Reuse sparse-flash's range-aware piece builder and per-piece `activeKvLen` to cap K work for scattered cacheblend rows.
4. Keep the current headDim-64 sparse flash path focused on the 1B class; do not expect it to solve `MiniCPM5-1B` / `Llama3.2 3B` / `Qwen3-8B`.
5. After sparse headDim-128 routing is fixed, separately optimize OrangePi PIC `full-compute`, because its dense PIC full-path is already slower than normal full-compute.

Implementation sketch: headDim-128 sparse q-split path

- The smallest useful change is not a new graph/runtime path. It is a new OpenCL attention dispatch branch that reuses:
  - `ensureFastPrefillTemps(...)`
  - `mRearrangeQKernel`
  - `mPackPagedKVKernel`
  - `mSoftmaxKernel`
  - `matmul_qkv_prefill_piece`
- Only the QK piece kernel truly needs sparse-aware semantics.

Recommended control flow:

```text
if (canUseFastPrefill(...)) -> dense q-split / dense Adreno GEMM
else if (canUseSparseFastPrefill(...)) -> current headDim-64 sparse flash
else if (canUseSparseQSplitPrefill(...)) -> new headDim-128 sparse q-split
else -> row/generic fallback
```

Recommended `canUseSparseQSplitPrefill(...)` scope:

- `mMeta->sparse_query_active == true`
- `externalHydrated == true`
- `attnLen > 1`, `kvLen > 0`
- `mHeadDim % 8 == 0`
- `mNumHead % mKvNumHead == 0`
- allow both:
  - score-layer `queryRowsAreFull == true`
  - later-layer compact-Q `queryRowsAreFull == false`

Kernel/dataflow plan:

1. Keep `qStorageLen` identical to current sparse-flash logic:
   - score layer: `qStorageLen = mQuerySeqLen`
   - later sparse layers: `qStorageLen = attnLen`
2. Reuse current `rearrange_q`:
   - it already works for contiguous inputs of length `qStorageLen`
   - no separate gather kernel is required for the first version
3. Reuse current `pack_paged_kv_prefill`:
   - pack all current-request K/V from PagedCache once
4. Reuse `_buildRangeAwareSparsePieces(...)`:
   - this already gives `qStart`, `qLen`, and per-piece `activeKvLen`
5. Add a sparse-aware QK piece kernel, for example:

```text
matmul_qk_div_mask_prefill_piece_sparse
```

   It should mirror CUDA `pagedPrefillQKKernel` behavior:
   - take `sparse_query`
   - take `query_rows_are_full`
   - compute:

```text
qLogical = sparse_query[q]
qRow     = query_rows_are_full ? qLogical : q
validScore = (k <= qLogical)
```

   - use `activeKvLen` as the piece axis limit
   - apply mask indexing with logical-row semantics on score layer
6. Reuse existing `softmax_v4_buf` on the piece-local QK tensor, but set axis length to `activeKvLen`, not full `kvLen`
7. Reuse `matmul_qkv_prefill_piece` with:
   - compact query count = `qPieceLen`
   - K/V axis length = `activeKvLen`
   - output written in compact row order

Why this is lower risk than a new flash kernel:

- CUDA already proves that q-split QK/softmax/QKV is enough to make sparse headDim-128 practical when it consumes:
  - sparse logical indices
  - `queryRowsAreFull`
- OpenCL already has:
  - sparse score/top-k before dispatch
  - `sparse_query` device buffer
  - range-aware sparse pieces
  - score-layer `full-Q / compact-output` semantics in the headDim-64 sparse flash path
- So the missing part is not semantics, only the headDim-128 fast kernel route.

Suggested phase order:

1. Phase A:
   - implement generic sparse q-split path on top of existing dense fast-prefill kernels
   - target both OrangePi and Rhino
   - do not touch Adreno dense GEMM path yet
2. Phase B:
   - after stable correctness/perf recovery, consider Adreno-specific sparse GEMM reuse if profiling still shows Rhino sparse path behind OrangePi
3. Phase C:
   - revisit headDim-128 flash-style sparse kernels only if q-split path still leaves a major gap versus CUDA

## 2026-06-25 Result: headDim-128 sparse q-split recovery on OrangePi

Implementation summary:

- Added a new OpenCL sparse q-split fast path for `headDim=128` in:
  - `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
  - `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp`
  - `source/backend/opencl/execution/cl/attention_buf.cl`
- New dispatch order is:
  - dense fast prefill
  - existing headDim-64 sparse flash
  - new headDim-128 sparse q-split
  - row/generic fallback
- The new sparse q-split path supports both:
  - score-layer `full-Q / compact-output` (`queryRowsAreFull=1`)
  - later-layer compact-Q sparse attention
- It reuses the existing dense q-split components:
  - `ensureFastPrefillTemps(...)`
  - `rearrange_q`
  - `pack_paged_kv_prefill`
  - `softmax_v4_buf`
  - `matmul_qkv_prefill_piece`
- The new QK kernel is `matmul_qk_div_mask_prefill_piece_sparse`, with:
  - `sparse_query`
  - `query_rows_are_full`
  - range-aware piece-local `activeKvLen`
  - per-row causal limit `k <= qLogical`

Formal OrangePi 1024 results after the new path:

```text
MiniCPM5-1B:
  normal=4.549767s, pic-full=16.065021s, full-reuse=0.514472s
  cacheblend 5%=1.622333s, 50%=7.719177s
  epic       5%=1.532307s, 50%=6.465034s

Llama3.2 3B:
  normal=15.885084s, pic-full=36.038463s, full-reuse=1.109441s
  cacheblend 5%=3.815022s, 50%=17.435609s
  epic       5%=3.654728s, 50%=15.614366s

Qwen3-8B:
  normal=35.687321s, pic-full=70.809410s, full-reuse=2.174056s
  cacheblend 5%=8.247438s, 50%=38.789479s
  epic       5%=6.874020s, 50%=32.053950s
```

Compared with the old bad OrangePi MiniCPM run:

```text
cacheblend 5%:  3.938678s -> 1.622333s
cacheblend 50%: 21.721183s -> 7.719177s
epic 5%:        3.846015s -> 1.532307s
epic 50%:       13.486879s -> 6.465034s
```

Observed pattern after the fix:

- The catastrophic `headDim=128` sparse fallback collapse is gone on OrangePi.
- `full-reuse` remains fast on all three models, so the persistent PIC cache source read + hydrate path is not the main remaining bottleneck.
- Low-budget sparse requests are now consistently much faster than `pic-full-recompute`:
  - MiniCPM5-1B: `cacheblend5 / pic-full ~= 9.90x`, `epic5 / pic-full ~= 10.48x`
  - Llama3.2 3B: `cacheblend5 / pic-full ~= 9.45x`, `epic5 / pic-full ~= 9.86x`
  - Qwen3-8B: `cacheblend5 / pic-full ~= 8.59x`, `epic5 / pic-full ~= 10.30x`
- But sparse requests are still around `3.0x-3.8x` slower than `full-reuse`, which means the remaining cost is the request-internal compute after hydrate:
  - score-layer attention
  - later sparse attention
  - compact dense graph work (`Convolution` / MLP / activation / residual)

Updated conclusion:

- The main `headDim=128` routing bug is fixed: OrangePi no longer misses a sparse fast path for `MiniCPM5-1B`, `Llama3.2 3B`, and `Qwen3-8B`.
- The next OrangePi optimization round should no longer treat this as a pure attention-routing problem.
- The correct reference is now OrangePi `Llama3.2 1B`, not the old broken `headDim=128` runs:
  - `Llama3.2 1B` 1024: `cacheblend5=1.678798s`, `epic5=1.618314s`
  - `MiniCPM5-1B` 1024 is already close in absolute low-budget latency.
  - The larger remaining gap is on wider `headDim=128` models, especially `Llama3.2 3B` and `Qwen3-8B`, where compact dense graph cost scales up.

Next actions:

1. Run OrangePi graph + attention attribution on `MiniCPM5-1B`, `Llama3.2 3B`, and `Qwen3-8B` with:
   - `MNN_PIC_GRAPH_PROFILE=1`
   - `MNN_PIC_GRAPH_PROFILE_TOP=1000`
   - `MNN_PAGED_ATTENTION_PROFILE=1`
   - `MNN_PAGED_ATTENTION_PROFILE_DETAIL=1`
2. Compare against OrangePi `Llama3.2 1B` at the same `ctx=1024` and low/high budgets.
3. If the profile shows dense dominates after the q-split recovery, move the next P1 focus to OpenCL compact dense rows:
   - compact `Convolution` / MLP kernels
   - `UnaryOp` / activation overhead on compact rows
   - ratio-sensitive tune keys for medium/high active-row counts

### MiniCPM5-1B / OrangePi / 1024 / cacheblend 50% attribution

Run:

```text
.cache/latency_budget_20260625/profile_orangepi_minicpm5_cb50_20260625_054345
server_env=MNN_PIC_GRAPH_PROFILE=1,MNN_PIC_GRAPH_PROFILE_TOP=1000,MNN_PAGED_ATTENTION_PROFILE=1,MNN_PAGED_ATTENTION_PROFILE_DETAIL=1
prefill/text cache_status=reused
measure cacheblend50 prefill_latency_s=7.882146   # profile-inflated; attribution only
```

Attention profile:

```text
layer0 full prefill:
  op=prefill_attention_fast_qk_softmax_qkv
  us=526864

layer1 score boundary:
  op=cacheblend_score
  us=7176
  read_us=6175 score_kernel_us=579 topk_us=386

layer1 score attention:
  op=score_qsplit_attention
  query=519 input_query=1024 full_q=1 q_split=2
  us=331312
  qk_us=247962 qkv_us=59957 softmax_us=2440

layer>=2 later sparse attention:
  op=sparse_qsplit_attention
  query=519 input_query=519 full_q=0 q_split=2
  per-layer us ~= 212-213 ms
  per-layer qk_us ~= 139 ms
  per-layer qkv_us ~= 60 ms
```

Graph profile summary:

```text
MNN_PIC_GRAPH_PROFILE_SUMMARY total_ms=8049.975
PicSparseAttention total=4689.729 ms calls=22
Convolution      total=2153.987 ms calls=169
PagedAttention   total=526.946 ms calls=1
PicScoreAttention total=339.899 ms calls=1
Raster           total=218.448 ms calls=396
```

Key per-op shapes:

```text
layer0 full attention:
  [1x1024x16x128] x [1x1024x2x128] -> [1x1024x2048]

layer1 score attention:
  [1x1024x16x128] x [1x1024x2x128] -> [1x519x2048]

layer>=2 sparse attention:
  [1x519x16x128] x [1x519x2x128] -> [1x519x2048]

compact MLP:
  [519x1536] -> [519x4608] gate/up
  [519x4608] -> [519x1536] down
```

Interpretation:

- The new `headDim=128` q-split path is definitely active. This request no longer falls back to `row` / `generic`.
- The remaining bottleneck is not a single block:
  - later `PicSparseAttention` is still the largest class at about `4.69 s`
  - compact dense `Convolution` is already the second largest class at about `2.15 s`
- `cacheblend_score` itself is small (`~7 ms`), and hydrate is negligible (`~0.35-0.45 ms` per layer).
- So the next OrangePi headDim-128 round must optimize both:
  1. later sparse q-split attention, especially QK cost;
  2. compact MLP / dense linear on active rows.

Refined next-step order:

1. Collect the same attribution for `Llama3.2 3B` and `Qwen3-8B` on OrangePi.
2. Compare whether their type split is still:
   - `PicSparseAttention` first
   - `Convolution` second
3. If that pattern repeats, treat `headDim=128` OrangePi optimization as a two-track problem:
   - P0: later sparse attention kernel work
   - P1: compact dense OpenCL kernels for medium/high active-row MLP shapes

### Llama3.2 3B / OrangePi / 1024 / cacheblend 50% attribution

Run:

```text
.cache/latency_budget_20260625/profile_orangepi_llama32_3b_cb50_20260625_054726
server_env=MNN_PIC_GRAPH_PROFILE=1,MNN_PIC_GRAPH_PROFILE_TOP=1000,MNN_PAGED_ATTENTION_PROFILE=1,MNN_PAGED_ATTENTION_PROFILE_DETAIL=1
prefill/text cache_status=reused
measure cacheblend50 prefill_latency_s=17.722298   # profile-inflated; attribution only
```

Attention profile:

```text
layer0 full prefill:
  op=prefill_attention_fast_qk_softmax_qkv
  us=785637

layer1 score boundary:
  op=cacheblend_score
  us=9841
  read_us=7685 score_kernel_us=1771 topk_us=349

layer1 score attention:
  op=score_qsplit_attention
  query=519 input_query=1024 full_q=1 q_split=2
  us=474550
  qk_us=353887 qkv_us=85572 softmax_us=3714

layer>=2 later sparse attention:
  op=sparse_qsplit_attention
  query=519 input_query=519 full_q=0 q_split=2
  per-layer us ~= 309 ms
  per-layer qk_us ~= 197 ms
  per-layer qkv_us ~= 86 ms
```

Graph profile summary:

```text
MNN_PIC_GRAPH_PROFILE_SUMMARY total_ms=18185.615
Convolution      total=8212.915 ms calls=197
PicSparseAttention total=8088.347 ms calls=26
PagedAttention   total=785.736 ms calls=1
PicScoreAttention total=485.760 ms calls=1
Raster           total=414.303 ms calls=460
```

Key per-op shapes:

```text
layer0 full attention:
  [1x1024x24x128] x [1x1024x8x128] -> [1x1024x3072]

layer1 score attention:
  [1x1024x24x128] x [1x1024x8x128] -> [1x519x3072]

layer>=2 sparse attention:
  [1x519x24x128] x [1x519x8x128] -> [1x519x3072]

compact MLP:
  [519x3072] -> [519x8192] gate/up
  [519x8192] -> [519x3072] down
```

Interpretation:

- `headDim=128` sparse q-split is active here too; this is not a row/generic fallback artifact.
- On `Llama3.2 3B`, compact dense `Convolution` is already as expensive as later sparse attention, slightly higher in this run:
  - `Convolution ~= 8.21 s`
  - `PicSparseAttention ~= 8.09 s`
- That is stronger evidence than MiniCPM that OrangePi `headDim=128` high-budget optimization is not an attention-only problem.
- The next default-path wins must therefore come from both sides:
  1. reduce later sparse q-split attention cost, especially QK;
  2. reduce compact MLP / q_proj / o_proj cost on medium/high active rows.

Updated direction:

- MiniCPM5-1B shows `PicSparseAttention > Convolution`.
- Llama3.2 3B shows `Convolution ~= PicSparseAttention`, with dense slightly ahead.
- So the stable cross-model statement is:
  - OrangePi `headDim=128` sparse routing is fixed;
  - the remaining gap versus the OrangePi `Llama3.2 1B` reference and versus Jetson now comes from a two-track hotspot split, not from a missing fast path.

### Qwen3-8B / OrangePi / 1024 / cacheblend 50% attribution

Run:

```text
.cache/latency_budget_20260625/profile_orangepi_qwen3_8b_cb50_20260625_055213
server_env=MNN_PIC_GRAPH_PROFILE=1,MNN_PIC_GRAPH_PROFILE_TOP=1000,MNN_PAGED_ATTENTION_PROFILE=1,MNN_PAGED_ATTENTION_PROFILE_DETAIL=1
prefill/text cache_status=reused
measure cacheblend50 prefill_latency_s=39.134997   # profile-inflated; attribution only
```

Graph profile summary:

```text
Convolution       total ~= 19652.918 ms
PicSparseAttention total ~= 17155.160 ms
PagedAttention    total ~= 1054.427 ms
PicScoreAttention total ~= 791.107 ms
```

Interpretation:

- The recovered `headDim=128` sparse q-split path is also active on `Qwen3-8B`; this is no longer a row/generic fallback case.
- On this wider model, compact dense work is at least as important as later sparse attention:
  - `Convolution ~= 19.65 s`
  - `PicSparseAttention ~= 17.16 s`
- The cross-model OrangePi pattern is now consistent:
  - `MiniCPM5-1B`: sparse attention first, dense second
  - `Llama3.2 3B`: dense and sparse nearly tied
  - `Qwen3-8B`: dense slightly ahead of sparse
- So the next `headDim=128` OrangePi gains must come from two tracks together:
  1. later sparse q-split attention, especially QK;
  2. compact dense MLP / q_proj / o_proj on medium/high active rows.

### 519-row compact dense threshold widening (`<=640`) is regressive

Goal:

- Check whether widening `usePicCompactGemmLowMemory()` from `globalY <= 512` to `globalY <= 640` helps the common 1024-token `cacheblend50` compact-row shape (`active rows = 519`).

Code change under test:

```text
source/backend/opencl/execution/buffer/ConvBufLowMemoryExecution.cpp
usePicCompactGemmLowMemory(): globalY <= 512 -> globalY <= 640
```

Runs:

```text
baseline:
  .cache/latency_budget_20260625/profile_orangepi_minicpm5_cb50_20260625_054345
  measure cacheblend50 prefill_latency_s=7.882146   # profile-inflated; attribution only

post-change:
  .cache/latency_budget_20260625/profile_orangepi_minicpm5_cb50_postcompact_20260625_055838
  measure cacheblend50 prefill_latency_s=8.531835   # profile-inflated; attribution only
```

Observed delta on `MiniCPM5-1B` cb50:

```text
total_ms:       8049.975 -> 8682.909
Convolution:    2153.987 -> 2817.135
PicSparseAttention: 4689.729 -> 4685.640
```

Conclusion:

- This is a dense-only regression. Later sparse attention stayed flat, while compact dense `Convolution` got materially slower.
- Do not keep `globalY <= 640` as the default gate.
- Code inspection explains why this is not just a kernel-symbol issue:
  - `pic_gemm_b4_c8_*` and `gemm_b4_c8_*` share the same kernel implementation body;
  - but `usePicCompactKernel` also changes the tune namespace and bypasses the generic `mUseFPWeight` best-path selection in `onResize()` for `batch > 16`.
- So widening the gate can force `519`-row shapes off the generic best-of path without reducing the underlying math cost.

Next:

1. Keep the accepted default gate at `globalY <= 512`.
2. If `519`-row compact dense is revisited, compare these paths explicitly on device instead of widening the default gate blindly:
   - generic `gemm_b4_c8_*`
   - generic `mUseFPWeight` path
   - `pic_gemm_b4_c8_*` with isolated tune cache
3. Continue the main OrangePi `headDim=128` line on the two real hotspots:
   - P0 sparse q-split attention
   - P1 compact dense graph work

### Narrow dense spill heuristic + finer qsplit pieces improve OrangePi headDim=128 cb50

Accepted code changes:

```text
source/backend/opencl/execution/buffer/ConvBufLowMemoryExecution.cpp
  - keep usePicCompactGemmLowMemory() at globalY <= 512
  - narrow the default non-FP-weight exception to:
      int4 && 512 < batch <= 576 && IC/OC >= 1024

source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
  - add _useFinerSparseQSplitPieces()
  - add _sparseQSplitChunkLen()
  - when staticWorkspace && cacheblend_score_ready && activeLen >= 384 &&
    selected_ratio >= 50%, cap qsplit chunk length to 192 before
    _buildRangeAwareSparsePieces()
```

Intent:

- Dense side: keep the generic quant GEMM default for the common `~519-row`
  int4 spill shapes instead of forcing `FP-weight + Strassen` purely because
  `batch > 512`.
- Sparse side: avoid `headDim=128` qsplit static-workspace cacheblend50
  collapsing into one or two oversized pieces with overly coarse K ranges.

Validation method:

- Rebuilt OrangePi OpenCL artifact with:

```text
MNN_TARGET_DEVICE=orangepi5plus
BUILD_TARGET=pic_server
BUILD_MNNCONVERT=0
INSTALL_AFTER_BUILD=1
JOBS=96
```

- Synced artifact to OrangePi.
- Re-ran only the exact hot bucket on device:
  - `device=orangepi`
  - `context=1024`
  - `mode=cacheblend`
  - `ratio=0.50`
- `run_pic_prefill_latency_sweep.py` kept the fixed device port and killed the
  old listener before each run.

Measured no-profile results:

```text
MiniCPM5-1B:
  benchmark.csv old: 7.719177 s
  new run:            6.570197 s
  delta:             -1.148980 s  (-14.9%)

Llama3.2 3B:
  benchmark.csv old: 17.435609 s
  new run:           15.434099 s
  delta:             -2.001510 s  (-11.5%)

Qwen3-8B:
  benchmark.csv old: 38.789479 s
  new run:           36.038715 s
  delta:             -2.750764 s  (-7.1%)
```

Interpretation:

- The combined default-path change is a real OrangePi win across all three
  headDim=128 models, not a MiniCPM-only artifact.
- The earlier broad `<=640` dense exception was too loose; the accepted narrow
  spill band is consistent with the device results.
- The qsplit-128 improvement does not require changing kernel math or adding
  fallback/env control; better piece granularity on the existing path is enough
  to move the cb50 bucket materially.

Follow-up:

1. Keep these two default-path changes.
2. Extend device validation to `epic50` and the next high-value budgets before
   touching broader heuristics.
3. Run detail profile again on the updated artifact to split the remaining
   gains between `Convolution` and later `PicSparseAttention`.

## 2026-06-25 Experiment: visible-K tile skip on headDim=128 sparse q-split

Goal:

- Stop `qk/qkv` from iterating invisible causal-K on the headDim=128 OpenCL
  sparse q-split path.
- Keep the existing `/v1/prefill/text` and sparse prefill execution path
  stable on OrangePi, fixed port `18132`, no fallback port churn.

Code changes:

```text
source/backend/opencl/execution/cl/attention_buf.cl
  - matmul_qk_div_mask_prefill_piece_sparse:
      * classify each K tile as full / partial / empty
      * empty tile: write `-FLT_MAX` and return
      * full tile: run the original fast fused MAD path
      * partial tile: only accumulate visible k0..k3 lanes
  - matmul_qkv_prefill_piece:
      * add sparse_query/output_seq_len/sparse_query_active args
      * cap the V loop to visible_kv_seq_len = max(q_logical_in_group) + 1

source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
  - wire the new matmul_qkv_prefill_piece args for:
      * runSparseQSplitPrefill()  -> sparse_query_active=1
      * runFastPrefill() piece path -> sparse_query_active=0

source/backend/opencl/execution/cl/attention_buf_mnn_cl.cpp
source/backend/opencl/execution/cl/opencl_source_map.hpp
  - regenerated with:
      python3 opencl_codegen.py .
```

Important root cause:

- The first rebuilt OrangePi artifact still logged:

```text
CL ERROR CODE : -49, info:setArg sparse qsplit matmul_qkv_prefill_piece
CL ERROR CODE : -52, info:run3d
```

- This was not a runtime kernel-cache mismatch. The real issue was that
  `attention_buf.cl` had changed, but the embedded generated source
  `attention_buf_mnn_cl.cpp` was still on the old kernel signature. After
  regenerating the OpenCL source map and rebuilding, the `CL ERROR` lines
  disappeared.

Stable device validation:

- OrangePi governors confirmed before test:

```text
CPU: performance
GPU: performance
GPU cur_freq = 1000000000
```

- Build:

```text
MNN_TARGET_DEVICE=orangepi5plus
BUILD_TARGET=pic_server
BUILD_MNNCONVERT=0
INSTALL_AFTER_BUILD=1
JOBS=96
```

- Sync:

```text
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

- Sweep:
  - script: `.codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py`
  - device: `orangepi`
  - model: `minicpm5-1b`
  - context: `1536`
  - modes: `cacheblend,epic`
  - ratio: `0.50`
  - fixed port strategy: kill old listener on `18132`, keep the same port

Stable measured results:

```text
MiniCPM5-1B / OrangePi / 1536 / cacheblend50:
  old benchmark.csv: 50.646872 s
  new stable run:    11.444419 s

MiniCPM5-1B / OrangePi / 1536 / epic50:
  old benchmark.csv: 26.854474 s
  new stable run:     8.580811 s
```

Profile result after the stable rebuild:

- run id:
  `orangepi_minicpm5_ctx1536_cb50_tilevisible_profile_20260625_172500`
- summary:
  `.cache/latency_budget_20260625/orangepi_minicpm5_ctx1536_cb50_tilevisible_profile_20260625_172500/summary.csv`
- remote log:
  `/mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/orangepi_minicpm5_ctx1536_cb50_tilevisible_profile_20260625_172500/pic_server_ctx1536.log`

Key detail:

```text
score layer:
  q_chunk=192 q_split=10
  qk_us=376.5 ms
  qkv_us=89.5 ms

later sparse layers:
  qk_us ~= 193.2 ms/layer
  qkv_us ~= 90.0 ms/layer
  qk_rect_tiles=34975
  qk_active_tiles=32173
  qk_row_tiles=127740
```

Compared with the previous cb50 detail profile:

- sparse-layer qk dropped from about `209.8 ms/layer` to about `193.2 ms/layer`
- sparse-layer qkv dropped from about `95.8 ms/layer` to about `90.0 ms/layer`
- `qk_active_tiles / qk_rect_tiles` did not change materially

Interpretation:

- The visible-K tile skip is real and stable, but it only recovers the
  over-compute inside already-issued pieces.
- The bigger blocker is unchanged piece geometry:
  cacheblend50 still has `q_chunk=192`, `q_split=10`, and highly scattered
  logical rows, so piece-level active K remains close to full context.
- Score layer full-Q q-split is still the single worst sparse-attention layer.
  Its Q gather pattern is now the next obvious hotspot, not a correctness bug.

Next step:

1. Do not touch `/v1/prefill/text` semantics.
2. Keep the stable visible-K skip.
3. Next optimize score-layer full-Q q-split input handling and/or piece
   geometry, because the current tile skip has already captured most of the
   removable invisible-K work inside the existing `192 x 10` schedule.

## 2026-06-25 Experiment: sparse q-split `COMPUTE_FLOAT4` fix repeat

Goal:

- Verify whether switching the `headDim=128` sparse q-split QK kernel from
  hardcoded `float4` accumulation to `COMPUTE_FLOAT4` is stable on real
  OrangePi runs.
- Replace the temporary `1536 / cb50, epic50` benchmark rows with the stable
  repeated result if variance stays small and no CL/runtime error shows up.

Code change under test:

```text
source/backend/opencl/execution/cl/attention_buf.cl
  - matmul_qk_div_mask_prefill_piece_sparse:
      * use COMPUTE_FLOAT4 / CONVERT_COMPUTE_FLOAT4
      * stop forcing fp32 accumulation on the sparse q-split path
```

Validation method:

- Reused the current OrangePi artifact built after regenerating embedded CL
  sources.
- Re-ran only the exact hot bucket:
  - `device=orangepi`
  - `model=minicpm5-1b`
  - `context=1536`
  - `modes=cacheblend,epic`
  - `ratio=0.50`
  - fixed ports:
    - remote `18132`
    - local `19132`
  - port policy: kill old listener and reuse the same port

Measured results:

```text
Run 1:
  cacheblend50 = 9.685507 s
  epic50       = 7.475859 s

Repeat:
  cacheblend50 = 9.703474 s
  epic50       = 7.474165 s
```

Observed variance:

```text
cacheblend50 delta = +0.017967 s  (+0.19%)
epic50       delta = -0.001694 s  (-0.02%)
```

Interpretation:

- The `COMPUTE_FLOAT4` sparse q-split fix is stable on repeated real-device
  runs.
- It materially improves the `1536 / 0.50` OrangePi MiniCPM bucket beyond the
  earlier `11.444419 / 8.580811` result from the visible-K-only build.
- The next bottleneck is no longer score-layer correctness or instability; the
  remaining work is pure performance attribution on later sparse attention and
  compact dense layers.

Follow-up:

1. Update `benchmark.csv` `MiniCPM5-1B / OrangePi / 1536 / cb50, epic50` rows
   to `9.703474 / 7.474165`.
2. Run one detail profile on the same artifact to measure how much of the gain
   came from later `PicSparseAttention` QK and whether compact dense
   `Convolution` moved at all.

Profile result on the `COMPUTE_FLOAT4` build:

- run id:
  `orangepi_minicpm5_ctx1536_cb50_qcompute_profile_20260625_175100`
- summary:
  `.cache/latency_budget_20260625/orangepi_minicpm5_ctx1536_cb50_qcompute_profile_20260625_175100/summary.csv`
- remote log:
  `/mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/orangepi_minicpm5_ctx1536_cb50_qcompute_profile_20260625_175100/pic_server_ctx1536.log`

Measured warm+profile request:

```text
cacheblend50 = 9.923847 s
```

Graph attribution on measured request (`request=2`):

```text
PicSparseAttention total_ms = 4884.290
Convolution      total_ms = 2845.598
PagedAttention   total_ms = 1140.599
PicScoreAttention total_ms = 527.295
overall total_ms = 9847.635
```

Compared with the earlier `qcompact` profile:

```text
PicSparseAttention: 6292.110 -> 4884.290   (-1407.820 ms, -22.4%)
Convolution:        2844.439 -> 2845.598   (~flat)
PicScoreAttention:   590.829 ->  527.295   (-63.534 ms, -10.8%)
```

Attention detail on the measured request:

```text
score layer (layer=1):
  q_chunk=192 q_split=10
  qk_us=124596
  qkv_us=89771

later sparse layers:
  qk_us ~= 1242xx-1249xx
  qkv_us ~= 893xx-902xx
  qk_rect_tiles=34975
  qk_active_tiles=32173
  qk_row_tiles=127740
```

Interpretation:

- The `COMPUTE_FLOAT4` fix is the first change in this branch that materially
  cuts later sparse-attention cost; it is not just a score-layer tweak.
- Dense compact `Convolution` is effectively unchanged, so the gain is not
  coming from low-memory GEMM selection.
- The remaining blocker is still sparse q-split piece geometry and active-K
  density, not score-layer correctness or a dense-kernel regression.

## 2026-06-25 Experiment: cross-model validation on OrangePi headDim=128 buckets

Goal:

- Check whether the `COMPUTE_FLOAT4` sparse q-split fix helps the other
  headDim=128 models on OrangePi, not just `MiniCPM5-1B`.

Measured results:

```text
Llama3.2 3B / OrangePi / 1024 / cacheblend50:
  old benchmark.csv: 15.434099 s
  new run:           13.260437 s

Llama3.2 3B / OrangePi / 1024 / epic50:
  old benchmark.csv: 15.614366 s
  new run:           12.070303 s

Qwen3-8B / OrangePi / 1024 / cacheblend50:
  old benchmark.csv: 36.038715 s
  new run:           31.395095 s

Qwen3-8B / OrangePi / 1024 / epic50:
  old benchmark.csv: 32.053950 s
  new run:           25.840180 s
```

Interpretation:

- The sparse QK compute-type fix is a real cross-model win for the whole
  headDim=128 family.
- It is still not enough to reach the user target on `Llama3.2 3B` /
  `Qwen3-8B`; later sparse attention and compact dense MLP remain the next
  bottlenecks.

## 2026-06-25 Change: shape-tuned sparse qsplit + OpenCL tune-level override

Code changes:

```text
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
  - add shape-keyed tune cache for headDim=128 sparse q-split q_chunk
  - reuse the existing tunedLws/tunedInfo cache instead of adding a new
    compatibility path
  - benchmark a small candidate set during Heavy/Wide warm-up only

transformers/pic_llm/engine/src/llm.cpp
  - add MNN_OPENCL_TUNE_LEVEL env override
  - warm runs can force heavy/wide tuning without changing production config
```

Validation:

- Build:

```text
MNN_TARGET_DEVICE=orangepi5plus
BUILD_TARGET=pic_server
BUILD_MNNCONVERT=0
INSTALL_AFTER_BUILD=1
JOBS=96
```

- Tune warm run:
  - run id:
    `orangepi_minicpm5_ctx1536_cb50_tuneheavy_smoke_20260625_182500`
  - server env:
    `MNN_OPENCL_TUNE_LEVEL=heavy`
  - update-cache response:
    `status=200`, `scope=mnn_runtime_cache`
  - measured result:
    `cacheblend50 = 9.734684 s`

- Default-env reuse run:
  - run id:
    `orangepi_minicpm5_ctx1536_cb50_tunecache_default_20260625_182900`
  - measured result:
    `cacheblend50 = 9.734832 s`

Interpretation:

- The new shape-tune path did not introduce crashes, port churn, or obvious
  regressions on the validated hot bucket.
- `heavy warm -> /v1/tune/update_cache -> default server reuse` is now a
  working flow for OpenCL PIC runs.
- This infrastructure is ready to be used on later sparse-attention and dense
  MLP hot shapes; the next work is improving the quality of the candidates,
  not rebuilding the cache flow.

## 2026-06-25 Change: headDim=128 later sparse flash canary

Code changes:

```text
source/backend/opencl/execution/cl/attention_buf.cl
  - extend sparse_flash_attention_row32 / row64 to accept headDim=128
  - reuse the same kernel symbols; row32/row64 still means lane width, not head_dim

source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
  - allow headDim=128 sparse flash on later sparse layers
  - keep score-layer full-Q headDim=128 on the existing q-split path for now
  - add a headDim-aware lane heuristic: headDim=128 currently prefers lane32
```

Validation on OrangePi (`Llama3.2 3B / 1024 / budget=0.50`):

```text
old benchmark.csv cacheblend50 (qsplit):   13.260437 s
old benchmark.csv epic50 (qsplit):         12.070303 s

headDim128 sparse flash lane64 formal:
  cacheblend50 = 14.427088 s

headDim128 sparse flash lane32 formal:
  cacheblend50 = 13.826558 s
  epic50       = 12.727769 s
```

Profile canary:

```text
run id:
  orangepi_llama32_3b_ctx1024_cb50_sparseflash128_lane32_profile_20260625_195500

later layers:
  op=sparse_flash_attention layer=2..27
  lane32_pieces=1 lane64_pieces=0

per-layer flash_us:
  roughly 151-155 ms with lane32
  previously about 185-188 ms with lane64
```

Interpretation:

- The `headDim=128` sparse flash path now runs stably on device and later sparse
  layers really do switch away from `sparse_qsplit_attention`.
- `lane32` is materially better than `lane64` for `headDim=128` on OrangePi,
  which matches the expected register/local-memory pressure story.
- Even after that improvement, end-to-end `cacheblend50` / `epic50` is still
  slower than the existing q-split baseline on this bucket, so these runs do
  not update `benchmark.csv`.
- The next bottleneck is no longer “can sparse flash run at all”; it is the
  remaining end-to-end cost around dense compact MLP and the still-not-good
  enough headDim=128 sparse-flash heuristic/implementation.

## 2026-06-25 Change: multi-Q-tile sparse flash variants + variant tune scaffold

Code changes:

```text
source/backend/opencl/execution/cl/attention_buf.cl
  - add mqtile_sparse_flash_hd64_q4k16
  - keep mqtile_sparse_flash_hd128_q4k16 as separate production-visible symbol

test/bench_ops/opencl/OpenCLAttentionPerf.cpp
  - extend SparseFlash/MultiQTile to include Llama3.2 1B headDim=64 cases
  - compare row32 / row64 / mqtile_hd64_q4k16 and row32 / row64 / mqtile_hd128_q4k16

source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp/.hpp
  - add sparse flash variant ids:
      row32
      row64
      mqtile_hd64_q4k16
      mqtile_hd128_q4k16
  - use OpenCL tune cache to persist the chosen sparse flash variant
  - default headDim=128 later sparse layers to mqtile_hd128_q4k16
  - keep the variant naming suffix (q4k16) explicit so future q4k8 / q8k16 can
    reuse the same tuning path without another control plane
```

OrangePi direct-op validation:

```text
log:
  /mnt/ssd/code/.cache/mnn_opencl_pic/bench_ops/orangepi_sparse_flash_mqtile_20260625_205034.log

headDim=64:
  llama3.2-1B_later_prefix_ctx1024_q519
    row32   = 82.0954 ms
    row64   = 99.3273 ms
    mqtile  = 54.6675 ms
    speedup_vs_row32 = 1.502x

  llama3.2-1B_later_scatter_ctx1024_q519
    row32   = 159.4070 ms
    row64   = 179.0286 ms
    mqtile  = 107.4200 ms
    speedup_vs_row32 = 1.484x

  llama3.2-1B_score_scatter_ctx1024_q519
    row32   = 163.5245 ms
    row64   = 181.4460 ms
    mqtile  = 107.3252 ms
    speedup_vs_row32 = 1.524x

headDim=128:
  llama3.2-3B_later_prefix_ctx1024_q519
    row32   = 210.0912 ms
    row64   = 263.9203 ms
    mqtile  = 78.5367 ms
    speedup_vs_row32 = 2.675x

  llama3.2-3B_later_scatter_ctx1024_q519
    row32   = 390.7018 ms
    row64   = 447.7782 ms
    mqtile  = 166.6525 ms
    speedup_vs_row32 = 2.344x

  llama3.2-3B_score_scatter_ctx1024_q519
    row32   = 398.7953 ms
    row64   = 454.7317 ms
    mqtile  = 171.2253 ms
    speedup_vs_row32 = 2.329x

  qwen3-8B_later_scatter_ctx1024_q519
    row32   = 524.3090 ms
    row64   = 605.9611 ms
    mqtile  = 223.1475 ms
    speedup_vs_row32 = 2.350x

  MiniCPM5-1B_later_scatter_ctx1522_q761
    row32   = 579.5079 ms
    row64   = 640.5878 ms
    mqtile  = 248.1416 ms
    speedup_vs_row32 = 2.335x
```

Interpretation:

- `q4k16` should be treated as part of the tune-able variant identity, not just
  a suffix in the kernel name. The runtime now has a place to persist exactly
  that choice.
- On OrangePi, `mqtile_hd64_q4k16` is already the best of the three tested
  variants on all current headDim=64 shapes, but one later-scatter bucket is
  only `1.484x` over row32. So `q4k16` is a solid baseline, not necessarily the
  final headDim=64 shape.
- The obvious next variant work is to add more suffix candidates such as
  `mqtile_sparse_flash_hd64_q4k8` and/or `mqtile_sparse_flash_hd64_q8k16`, then
  let the same tune cache choose among:
  `row32 / row64 / mqtile_hd64_q4k16 / ...`.

2026-06-26 OrangePi MiniCPM5-1B ctx1536 profile on `cacheblend/epic`
ratios `0.2/0.3/0.4`:

- `cacheblend 0.2`: `Convolution 30.2%`, `PagedAttention 23.1%`,
  `PicSparseAttention 20.1%`, `Raster 12.6%`.
- `cacheblend 0.3`: `Convolution 61.8%` spikes hard; the top ops are all
  dense `Convolution` layers. The OpenCL conv profile shows this request at
  `batch=623`, which falls back to `pic_compact=0`.
- `cacheblend 0.4`: `Convolution 34.0%`, `PicSparseAttention 25.7%`,
  `PagedAttention 15.0%`, `Raster 11.4%`.
- `epic 0.2`: `Convolution 32.0%`, `PagedAttention 28.3%`,
  `PicSparseAttention 12.7%`, `Raster 13.7%`.
- `epic 0.3`: `Convolution 73.8%` dominates the request.
- `epic 0.4`: `Convolution 42.1%`, `PicSparseAttention 25.5%`,
  `PagedAttention 17.0%`, `Raster 6.7%`.

Takeaway:

- The 0.3 anomaly is not primarily attention-bound. The dominant regression is
  dense `Convolution` / compact MLP work, not `PicSparseAttention`.
- `0.4` shifts more cost back into sparse attention, but still does not explain
  the 0.3 spike.
- The current compact gate was too rigid for this shape: `batch=623` dropped
  back to the generic dense path even though the row/channel ratio still fits a
  compact-row matmul bucket.
- Next profiling should keep focus on the compact dense path around
  `use_fp_weight=1 / pic_compact=1` for the 319-row active-shape bucket.

2026-06-26 follow-up on OrangePi MiniCPM5-1B ctx1536 dense anomaly:

- Re-profiled after inspecting the FP-weight path more closely and found the
  real measured `0.3` request shape is `active rows = 471`, not `623`.
- Both `471` and `623` already run with:
  - `use_fp_weight=1`
  - `pic_compact=1`
- The real cliff was inside `useFPWeightGemmLowMemory()`:
  - `M=471` previously stayed in the `mAlignM=32` bucket, so it rounded to
    `alignM=480`.
  - `M=623` used `mAlignM=64`, rounding to `alignM=640`.
  - `M=319` also stayed at `mAlignM=32`, but rounds to `alignM=320`, which is
    already a friendlier Xgemm shape.
- `StrassenMatrixComputor` is not the root cause for the `471` cliff:
  - `alignM=480 < 512`, so this bucket does not enter Strassen recursion.
  - The slow path is the basic Xgemm parameter/tune bucket for `e=480`.
- Accepted heuristic change:

```text
source/backend/opencl/execution/buffer/ConvBufLowMemoryExecution.cpp
  useFPWeightGemmLowMemory():
    if M > 384 && K >= 1024 && N >= 1024 && ratio >= 0.5
      mAlignM = 64
```

Intent:

- Promote high-arithmetic compact dense shapes around `M~=471` out of the
  `alignM=480` bucket and into `alignM=512`, where OrangePi Mali gets back to a
  much better Xgemm regime.

Validated on device:

```text
run-id:
  orangepi_minicpm1536_alignm64_profile_20260626_1
```

Summary delta vs previous profiled run:

- `cacheblend 0.2`
  - `3.796683s -> 3.693556s`
  - still healthy; no regression signal
- `cacheblend 0.3`
  - `10.165801s -> 4.773843s`
  - graph profile:
    - `Convolution 69.6% -> 39.0%`
    - `PicSparseAttention 13.1% -> 24.4%`
    - request shape stays `outputs=[1x471x2048]`
- `cacheblend 0.4`
  - `6.202831s -> 5.858921s`
  - still monotonic with budget
- `epic 0.2`
  - `3.339334s -> 3.332620s`
- `epic 0.3`
  - `9.203015s -> 4.007346s`
  - graph profile:
    - `Convolution 76.8% -> 46.2%`
    - request shape stays `outputs=[1x471x2048]`
- `epic 0.4`
  - `5.067990s -> 4.999504s`

Updated interpretation:

- The previous "623-row compact gate" explanation was only part of the older
  story. After the compact gate fix, the remaining 0.3 anomaly was the
  FP-weight dense bucket at `M=471 -> alignM=480`.
- After the alignment change, both `cacheblend` and `epic` recover logical
  budget scaling at ctx1536:
  - `0.20 < 0.30 < 0.40`
  - no more mid-budget latency spike
- The dominant remaining cost is now shared more normally between:
  - full `layer0` `PagedAttention`
  - compact dense `Convolution`
  - later `PicSparseAttention`

Next dense-side focus:

- If OrangePi still has bad mid-budget pockets on other models, inspect whether
  they also land in non-64-divisible `alignM` buckets on the FP-weight path.
- If similar cliffs appear, continue from `mAlignM` / Xgemm parameter selection
  first, before changing sparse attention again.
