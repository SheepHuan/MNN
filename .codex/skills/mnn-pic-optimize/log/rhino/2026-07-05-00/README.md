# Rhino / Adreno Decode TPOT Three-Model Check

Run:

```text
decode_adreno_three_models_vs_normal_ctx512_1024_20260704_161944
```

Scope:

- device: Rhino Pi-X1 / Adreno OpenCL, max frequency
- models: MiniCPM5-1B, Llama3.2-3B, Qwen3-4B
- contexts: 512, 1024
- normal baseline: true normal LLM, `llm_bench -n 16`
- PIC: dualgraph decode-repair x=0/1/3/5/7, `max_tokens=16`, `suffix_from_cache_tokens=1`

Before this run, local OpenCL dualgraph exports for Llama3.2-3B and Qwen3-4B were
synced to Rhino under `*-rhino-dualgraph/`; both configs validate
`llm_decode_model=llm_decode.mnn`.

P0:

- no `failures.csv`
- no `decode_prepare_inside_decode=1`
- no `ERROR`, `target unavailable`, or async PIC cache read failure
- logs show `ordinary_q1=0 repair=1` for q=1 and q=8 decode-repair routes

Initial conclusion:

Adreno was not sped up versus true normal LLM decode in the first three-model
check. PIC dualgraph x0 was still about 3.4x-5.0x slower than normal, and x>0
added more TPOT. Do not promote the current Adreno dense/direct C4 direction as
a production speedup.

Update:

The first confirmed x0 fix is to default Adreno `attnLen == 1` decode-repair to
the existing transposed-K sparse qtile q1-row route, while leaving Adreno rows
`>1` unchanged. Code change:

```text
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
_decodeRepairSparseQTileDefaultEnabled():
  MALI -> true
  ADRENO -> true only when attnLen == 1
```

Three-model Rhino x0 smoke, 16 generated tokens, `suffix_from_cache_tokens=1`:

```text
model,ctx,normal,old_pic_x0,new_pic_x0,old_pic/normal,new_pic/normal,new_vs_old
MiniCPM5-1B,512,23.219,79.247,53.708,3.41x,2.31x,1.48x faster
MiniCPM5-1B,1024,29.214,118.497,64.038,4.06x,2.19x,1.85x faster
Llama3.2-3B,512,53.256,156.851,86.500,2.95x,1.62x,1.81x faster
Llama3.2-3B,1024,55.824,240.879,99.730,4.31x,1.79x,2.42x faster
Qwen3-4B,512,70.256,227.291,110.759,3.24x,1.58x,2.05x faster
Qwen3-4B,1024,71.710,362.245,122.108,5.05x,1.70x,2.97x faster
```

Route evidence in the rebuilt default run:

```text
ordinary_q1=0 repair=1 q=1
repair_qtile_enabled=1 repair_qtile_candidate=1 repair_qtile_prefix_ready=1 repair_qtile_route=1
repair_policy=0
decode_prepare_inside_decode=0
```

This brings PIC x0 much closer to true normal across all three models but does
not finish the whole objective. Remaining work is to split the residual
1.58x-2.31x normal gap, then decide whether any separate rows>1 Adreno route is
warranted for x1/x3/x5/x7.

## Update: Align Adreno With Mali QTile Family For All Supported Rows

User direction: RhinoPi should align with OrangePi/Mali's implementation path,
not keep Adreno as an x0-only exception.

Code now defaults Adreno to the same decode-repair sparse qtile family as Mali
for every shape accepted by `_decodeRepairSparseQTileShapeSupported()`:

```text
_decodeRepairSparseQTileDefaultEnabled():
  return gpu == MALI || gpu == ADRENO
```

The shape guard still limits this default to `attnLen in [1, 8]`, so the covered
decode-repair rows are x0/x1/x3/x5/x7, i.e. q=1/2/4/6/8. This supersedes the
earlier x0-only Adreno gate.

Validation run:

```text
decode_adreno_allmodels_ctx512_1024_qtile_allrows_default_20260704_172439
```

Scope:

- device: Rhino Pi-X1 / Adreno OpenCL, max frequency
- models: MiniCPM5-1B, Llama3.2-3B, Qwen3-4B
- contexts: 512, 1024
- PIC dualgraph decode-repair x=0/1/3/5/7
- no normal rerun in this pass; compare to previous true-normal rows above

P0:

- `failures.csv`: absent
- no `ERROR`, `target unavailable`, `async persistent PIC cache read failed`,
  `Cache invalid`, `CL_OUT_OF_RESOURCES`, `decode_prepare_inside_decode=1`,
  `repair_qtile_route=0`, or `ordinary_q1=1`
- each model log shows q=1/2/4/6/8 all `repair_qtile_route=1`
- each model log shows decode graph route for seq_len=1/2/4/6/8

Current TPOT:

```text
model,ctx,x0,x1,x3,x5,x7
MiniCPM5-1B,512,54.020,62.074,70.713,99.051,104.077
MiniCPM5-1B,1024,64.640,68.763,86.797,124.069,132.243
Llama3.2-3B,512,87.014,107.552,118.707,189.927,195.877
Llama3.2-3B,1024,98.548,115.965,143.425,219.060,229.175
Qwen3-4B,512,111.503,139.731,154.459,248.829,255.691
Qwen3-4B,1024,121.775,149.947,187.898,296.221,305.274
```

Old default -> new default speedup:

```text
model,ctx,x0,x1,x3,x5,x7
MiniCPM5-1B,512,1.47x,1.42x,1.25x,1.11x,1.08x
MiniCPM5-1B,1024,1.83x,1.88x,1.48x,1.30x,1.24x
Llama3.2-3B,512,1.80x,1.67x,1.54x,1.24x,1.22x
Llama3.2-3B,1024,2.44x,2.30x,1.88x,1.64x,1.57x
Qwen3-4B,512,2.04x,1.85x,1.70x,1.35x,1.33x
Qwen3-4B,1024,2.97x,2.63x,2.12x,1.73x,1.61x
```

Conclusion:

Adreno is now on the same supported-row qtile implementation family as Mali for
PIC dualgraph decode-repair. This is a broad improvement across x0/x1/x3/x5/x7,
not only x0. The remaining gap is now highest at x5/x7 and should be profiled as
the same qtile family's rows>1 cost plus graph tail/dense overhead, not handled
by routing rows>1 to a separate default family.

## Update: Adreno Record Queue Default

After the qtile family alignment, Adreno still had a large fixed overhead gap to
true normal decode. `MNN_PIC_OPENCL_RECORD_QUEUE=op` was tested as a Qualcomm
recordable-queue A/B on the same qtile path.

Validation run:

```text
decode_20260705_015956
server_env_extra=MNN_PIC_DECODE_DEBUG=1 MNN_PIC_OPENCL_RECORD_QUEUE=op
```

P0:

- 30/30 benchmark rows succeeded, `failures.csv` absent
- all rows use `decode_runtime=mnn_token_id_sparse_decode`
- all model logs show `record_queue=1`, `repair_qtile_route=1`,
  `ordinary_q1=0`, `decode_prepare_inside_decode=0`
- no `ERROR`, `target unavailable`, async PIC cache failure, `Cache invalid`,
  or `CL_OUT_OF_RESOURCES`

Current best TPOT:

```text
model,ctx,x0,x1,x3,x5,x7
MiniCPM5-1B,512,47.764,52.282,58.526,87.317,90.177
MiniCPM5-1B,1024,56.439,58.193,76.751,117.582,122.399
Llama3.2-3B,512,73.981,93.201,108.505,174.666,180.143
Llama3.2-3B,1024,83.540,100.285,129.203,212.656,221.757
Qwen3-4B,512,96.612,123.788,139.742,233.287,233.203
Qwen3-4B,1024,108.852,135.159,175.784,279.865,286.755
```

Record-op speedup versus the qtile default is positive on all 30 rows, ranging
from `1.030x` to `1.208x`. Remaining gap to true normal x0:

```text
MiniCPM5-1B: ctx512 2.06x, ctx1024 1.93x
Llama3.2-3B: ctx512 1.39x, ctx1024 1.50x
Qwen3-4B: ctx512 1.38x, ctx1024 1.52x
```

`MNN_GPU_RECORD_OP` is now the PIC OpenCL default in `llm.cpp`; explicit
`MNN_PIC_OPENCL_RECORD_QUEUE=off` remains available for A/B, and `batch` remains
diagnostic only because the MiniCPM ctx512 x0 batch variant failed with a closed
HTTP response.

Default smoke after rebuilding and syncing the new artifact:

```text
decode_adreno_minicpm_ctx512_x0_record_default_20260705_0215
server_env_extra=MNN_PIC_DECODE_DEBUG=1
MiniCPM5-1B ctx512 x0 = 48.043 ms/token
record_queue=1, repair_qtile_route=1, ordinary_q1=0, decode_prepare_inside_decode=0
```

## Update: q1 Identity Fused-KV A/B

Hypothesis: PIC x0 and true normal q=1 are close at the attention algorithm level,
but their K read layout differs. True normal identity fused-KV scans contiguous
`key_cache + k * 128` with `vload4`; PIC qtile scans transposed `decode_key +
d * key_max_len + k`. For `q=1` there is no multi-Q K reuse, so the transposed
layout may hurt Adreno cache/bandwidth behavior.

Implemented an explicit A/B variant only:

```text
MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_IDENTITY_FUSED_KV=1
```

The candidate still stays inside the PIC dualgraph decode-repair family. It
writes `key_cache`, `value_cache`, and `decode_key`, but computes QK from
contiguous `key_cache`.

MiniCPM5-1B clean sequential A/B, same artifact, record-op on:

```text
ctx,default_qtile_ms,q1_identity_fused_kv_ms,delta_ms,speedup
512,48.479,47.302,-1.178,1.025x
1024,56.308,53.180,-3.128,1.059x
```

P0:

- both default and candidate runs had empty `failures.csv`
- no `ERROR`, `target unavailable`, async PIC cache failure, `CL_OUT_OF_RESOURCES`,
  or `decode_prepare_inside_decode=1`
- candidate logs show `repair_q1_identity_fused_kv=1`, `record_queue=1`,
  `repair_qtile_route=1`, `ordinary_q1=0`
- default logs show the same route except `repair_q1_identity_fused_kv=0`

This validates the layout hypothesis as a small Adreno x0 gain, especially at
ctx1024, but it does not explain the whole remaining normal gap. The current
best MiniCPM5-1B x0 remains about `47.3 ms` at ctx512 versus true normal
`23.2 ms`, and about `53.2 ms` at ctx1024 versus true normal `29.2 ms`.

Do not promote this variant to default yet:

- It only covers `attnLen == 1` / x0.
- x1/x3/x5/x7 still need the transposed/qtile family to share K across active
  rows; a real rows>1 identity implementation would be a separate kernel family.
- Mali may not benefit: transposed K can give better cross-lane coalescing there,
  while identity K gives each lane contiguous `d4` but makes lanes stride by 128.
  OrangePi/Mali needs its own env-gated A/B before any default change.

Other x0 A/B results on the record-op qtile path:

```text
variant,ctx512_ms,decision
record off,54.024,slower
record op,47.468,kept as default
fused append,66.826,negative
lane32,53.683,negative
lane64,49.704,negative versus default lane128-ish
```

## Update: Negative Kernel Paths Commented Out

结论：还没有消除 PIC dualgraph x0 相对 true normal LLM 的额外约 20ms+ 延迟。
后续不应继续在这些已测过的路由上反复试探，而应直接拆
PagedAttention 的 append 和 attention scan 成本。

已明确无效或不适合作为默认的 Adreno A/B：

```text
variant,result
keycache qtile v2,MiniCPM 略好但 Llama/Qwen x0 回归；x0 gap 仍约 +24..44 ms/token
q1 identity fused-KV,仍比 true normal 慢 +23..44 ms/token；Llama/Qwen 回归
q1 GQA / GQA fused,所有已测 x0 model/context pair 回归
identity qtile / no-transposed-K,只帮 MiniCPM，Llama/Qwen 回归
```

源码处理：

- active 默认路径保留为 transposed-K qtile + record queue。
- 上述低效/无效 kernel、kernel build、env gate、record queue 变体和旧调度代码都已用 `#if 0` 注释保留，没有删除实现。
- `.cl` 改动后已重新运行 `source/backend/opencl/execution/cl/opencl_codegen.py` 生成对应 C++ 内嵌源。

当前最有价值的归因是 strict normal p514 对齐后的拆分：上下文差异不是
x0 慢因；PIC x0 多出来的固定成本集中在 PagedAttention append 和
attention scan，而不是 rank/top-k、decode prepare、dense/MLP 或 Raster。

Post-cleanup smoke after rebuilding and syncing the Adreno artifact:

```text
run_id=decode_adreno_minicpm_ctx512_x0_after_cleanup_20260705_041937
MiniCPM5-1B ctx512 PIC x0 tpot_ms=48.726
failures.csv absent
route: repair_qtile_route=1 record_queue=1 ordinary_q1=0
disabled variants: repair_keycache_qtile_v2=0 repair_q1_identity_fused_kv=0 repair_q1_gqa=0
note: server log had one cold "Cache invalid, will be reset", so this is a route/build smoke, not formal TPOT
```

Graph profile with record-op and profile detail disabled still attributes most
decode time to PagedAttention:

```text
PagedAttention 71.160 ms / 72 calls
Convolution      6.523 ms
Raster           3.807 ms
ArgMax           3.251 ms
```

Conclusion: the remaining x0 gap is mainly PIC PagedAttention scan plus PIC
decode-repair/PagedCache state maintenance, with a smaller fixed graph tail.
The q1 contiguous-K candidate is useful but only a partial micro-optimization.

## Update: q1 GQA Fused A/B Was Negative

Tested an explicit PIC repair x0 GQA fused attention variant:

```text
MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_GQA=1
```

Three-model x0 A/B, ctx512/1024, `suffix_from_cache_tokens=1`, profile off:

```text
model,ctx,default_ms,gqa_ms,delta_ms
Llama3.2 3B,512,72.297,73.580,+1.283
Llama3.2 3B,1024,81.910,85.277,+3.366
MiniCPM5-1B,512,46.967,58.558,+11.591
MiniCPM5-1B,1024,54.473,75.927,+21.454
Qwen3-4B,512,95.448,98.695,+3.247
Qwen3-4B,1024,107.799,115.606,+7.807
```

MiniCPM ctx512 profile detail:

```text
default qtile: calls=360 total=431.948 ms append=162.003 ms attention=266.064 ms rank=2.130 ms
gqa fused:     calls=360 total=607.076 ms append=173.011 ms attention=429.443 ms rank=2.594 ms
```

Conclusion: missing GQA reuse is not the Adreno x0 root cause. It reduces
parallel workgroups and makes the attention body slower, especially for
MiniCPM's small KV-head count. Do not promote this variant.

## Current x0 Interpretation

PIC x0 is not just true normal decode with a different cache object. It runs the
PIC dualgraph decode-repair route:

```text
append_sparse_decode_key_value_hd128
decode_causal_attention_hd128_transposed_k_sparse_qtile
decode_attention_rank (when enabled by metadata)
```

True normal q1 uses the identity/fused-KV route and scans contiguous
`key_cache`. PIC x0 scans the prepared transposed `decode_key`, maintains
`sparse_query`, writes `key_cache/value_cache/decode_key`, and remains inside
the repair metadata path. The remaining Adreno gap should therefore be treated
as a PagedAttention/PagedCache implementation gap, not an unexplained normal
decode regression.

Next direction: stop expanding identity-buffer experiments by default. Add an
explicit Adreno image/K-read A/B for the PIC repair qtile family, keeping the
same PIC repair semantics and PagedCache/decodeKey boundary, to test whether
Adreno image/texture cache improves the transposed-K scan.

## Update: Adreno No-Transposed-K QTile Family A/B

Implemented a full qtile family as an explicit Adreno-only A/B:

```text
MNN_PAGED_ATTENTION_DECODE_REPAIR_IDENTITY_QTILE=1
MNN_PAGED_ATTENTION_DECODE_REPAIR_NO_TRANSPOSED_K=1  # alias
```

This is not a normal decode fallback. It stays in PIC dualgraph decode-repair:

- `ordinary_q1=0`
- `repair_qtile_route=1`
- append still writes `key_cache`, `value_cache`, and `decode_key`
- attention reads K from contiguous `key_cache + k * 128` instead of
  transposed `decode_key + d * key_max_len + k`

Kernel coverage:

```text
decode_causal_attention_hd128_identity_qtile_q1_row32/64/128
decode_causal_attention_hd128_identity_qtile_q2_row32/64/128
decode_causal_attention_hd128_identity_qtile_q4_row32/64/128
decode_causal_attention_hd128_identity_qtile_q8_row32/64/128
```

Profile smoke:

```text
decode_20260705_035943
server_env_extra=MNN_PIC_DECODE_DEBUG=1 MNN_PAGED_ATTENTION_PROFILE=1
                 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1
                 MNN_PAGED_ATTENTION_DECODE_REPAIR_IDENTITY_QTILE=1
```

P0:

- no `decode_prepare_inside_decode=1`
- no `ERROR`, `target unavailable`, async PIC cache failure,
  `CL_OUT_OF_RESOURCES`, or OpenCL build failure
- logs show `ordinary_q1=0`, `repair_qtile_route=1`, and
  `op=decode_causal_attention_hd128_identity_qtile`

Three-model x0 A/B, profile off, ctx512/1024, 16 generated tokens:

```text
model,ctx,default_ms,identity_qtile_ms,delta_ms,ratio
Llama3.2 3B,512,72.524,74.588,+2.064,1.0285
Llama3.2 3B,1024,81.941,86.699,+4.758,1.0581
MiniCPM5-1B,512,46.813,46.683,-0.130,0.9972
MiniCPM5-1B,1024,54.121,52.316,-1.806,0.9666
Qwen3-4B,512,95.559,98.251,+2.692,1.0282
Qwen3-4B,1024,108.206,113.718,+5.511,1.0509
```

Decision:

No-transposed-K buffer qtile is not a default Adreno win. It slightly helps
MiniCPM, but regresses Llama and Qwen by about 2.8%-5.8%. This weakens the
simple "transposed K is the root cause" hypothesis. The measured effect is
model-dependent and must be reported by component, not inferred from layout
alone.

Profile-on decode-forward attribution, same profile/debug flags, `max_tokens=2`
and x0 only:

```text
model,ctx,total_delta_ms,append_delta_ms,attention_delta_ms,rank_delta_ms
Llama3.2 3B,512,+2.167,+0.501,+1.675,-0.010
Llama3.2 3B,1024,+3.307,+0.459,+2.735,+0.108
MiniCPM5-1B,512,-3.313,+0.115,-3.400,-0.026
MiniCPM5-1B,1024,-6.683,-0.103,-6.616,+0.043
Qwen3-4B,512,+0.603,-0.198,+0.795,-0.002
Qwen3-4B,1024,+6.222,+2.382,+2.306,+0.080
```

This says the no-transposed-K candidate's profile-off TPOT gain/loss mostly
tracks the decode attention body itself. It is positive for MiniCPM and negative
for Llama/Qwen; sample time is not the cause. Qwen ctx1024 additionally shows an
append-time increase. The data does not support promoting the identity-qtiler
family as a general Adreno fix.

Keep this family as an env-gated diagnostic only. The next useful direction is
not to promote identity-buffer qtile, but to test an explicit image/K-read
variant or a more targeted append/attention split that preserves qtile semantics.

## Update: PIC x0 vs True Normal Decode Attribution

The latest attribution compared three things separately: benchmark context,
decode graph op counts, and OpenCL kernel timing.

The earlier x0-vs-normal CSVs had a small context mismatch: PIC ctx512 with
`pic_suffix_from_cache_tokens=1` enters decode at `kv_len=514`. A strict normal
MiniCPM rerun at `-p 514` measured `44.35 tok/s`, or `22.55 ms/token`; the same
command at `-p 512` measured `22.28 ms/token`. The `512 -> 514` delta is only
about `0.27 ms/token`, so context length is not the `20+ ms` PIC x0 gap.

For MiniCPM5-1B ctx512, the graph bodies are effectively aligned:

```text
op type,normal per token,PIC x0 per decode block
Attention/PagedAttention,24,24
Convolution,169,169
Raster,390,390
BinaryOp,121,121
While,96,96
UnaryOp,74,74
LayerNorm,49,49
Cast,1,1
ArgMax,0,2
```

PIC's extra `ArgMax` is only about `0.4 ms` in graph-profile mode. The dense,
Raster, BinaryOp, While, UnaryOp, and LayerNorm counts match normal per-token
counts, so the x0 gap is not caused by many extra MLP/Raster/body ops.

The measured hotspot is the attention implementation:

```text
path,attention implementation,attention total per token
normal q1,Attention kernels,~3.2 ms
PIC x0,PagedAttention decode_causal_attention_hd128_transposed_k_sparse_qtile,25.0-29.5 ms
```

PIC x0 PagedAttention split for MiniCPM ctx512:

```text
group,total_ms,append_ms,attention_ms,rank_ms,kv_len,prepare_len
1,24.976,9.099,15.814,0.028,514,513
2,29.502,11.158,18.007,0.207,514,513
```

The explicit prepare-transpose phase is outside decode timing:

```text
prepare_group,total_ms,avg_us_per_layer,inside_decode
1,11.310,471.2,0
2,5.730,238.8,0
```

Conclusion: in the current near-context comparison, PIC x0 is slower because
the PIC PagedAttention decode family is much slower than normal q1 Attention.
The extra graph op body is not the explanation. Record queue saves roughly
`6 ms/token`, but after strict p514 alignment the remaining delay is still in
PIC PagedAttention: append current K/V/decodeKey (`9.1-11.2 ms/token`) plus the
qtile attention scan (`15.8-18.0 ms/token`). The next work item is a targeted
replacement/split of those two pieces inside the PIC decode family.

## Update: Key-cache QTile V2 A/B

An env-gated Adreno diagnostic route was added:

```text
MNN_PAGED_ATTENTION_DECODE_REPAIR_KEYCACHE_QTILE_V2=1
```

It keeps the same PIC dualgraph decode-repair family, but skips writing
`decode_key` in append and lets the qtile attention read contiguous `key_cache`
directly. The route is limited to Adreno and `1 <= attnLen <= 8`.

P0 profile smoke passed before the profile-off matrix: no
`decode_prepare_inside_decode=1`, no `ERROR`, no target unavailable, no cache
invalid, and `ordinary_q1=1` count was zero. The profile-off v2 run also had no
`failures.csv`, no hard error logs, `repair_keycache_qtile_v2=1` appeared in
the route logs, and `decode_prepare_inside_decode=1` stayed zero.

Profile-off TPOT result for x0:

```text
model,ctx,true_normal,default_x0,v2_x0,default_extra,v2_extra,v2_minus_default
MiniCPM5-1B,512,23.219,48.526,47.796,+25.307,+24.577,-0.730
MiniCPM5-1B,1024,29.214,56.701,53.535,+27.487,+24.321,-3.167
Llama3.2 3B,512,53.256,73.945,75.107,+20.689,+21.851,+1.162
Llama3.2 3B,1024,55.824,83.696,87.466,+27.872,+31.642,+3.770
Qwen3-4B,512,70.256,96.914,100.540,+26.658,+30.284,+3.627
Qwen3-4B,1024,71.710,109.734,115.672,+38.024,+43.962,+5.938
```

Conclusion: the extra PIC x0 delay was not eliminated. V2 improves MiniCPM by
`0.7-3.2 ms/token`, but regresses Llama and Qwen x0 by `1.2-5.9 ms/token`. The
remaining x0 gap is still roughly `24-44 ms/token` versus true normal. Do not
promote this route to default; keep it as an explicit diagnostic variant.

## Update: x0 Residual Gap Attribution After Cleanup

Two short Rhino/MiniCPM ctx512 x0 attribution runs were added after the disabled
path cleanup. Both are profile runs only, not formal TPOT rows; P0 scans were
clean with no `decode_prepare_inside_decode=1`, no server errors, and the route
remained `ordinary_q1=0 repair_qtile_route=1`.

PagedAttention detail run:

```text
run_id=decode_adreno_minicpm_ctx512_x0_profile_attribution_20260705_043604
tpot_ms=79.458  # profile-detail attribution only
```

Aggregated over warm + measure passes, divided by two for one decode pass:

```text
kv_len,total_ms,append_ms,attention_ms,rank_ms
514,26.952,10.073,16.704,0.091
515,29.832,11.331,18.192,0.173
516,28.888,10.900,17.715,0.148
```

Graph profile run:

```text
run_id=decode_adreno_minicpm_ctx512_x0_graph_profile_20260705_0440
tpot_ms=52.697  # graph-profile attribution only
```

Decode graph body was not the missing 20ms:

```text
request,total_ms,PagedAttention,Convolution,Raster,ArgMax
1,11.361,8.425,1.040,0.721,0.409
2,2.397,0.368,0.720,0.591,0.133
```

Current answer: no, the PIC x0 extra latency versus true normal decode has not
been eliminated. After strict p514 normal alignment and code cleanup, the
remaining gap is still dominated by the PIC PagedAttention decode family:
append current K/V/decodeKey plus the qtile attention scan. The dense/Raster
graph body is small in decode-scope graph profile and should not be the next
optimization target. The next useful work is to split or replace the
PagedAttention append and attention scan, and to add clearer timing for OpenCL
queue/record replay versus graph callback timing.
