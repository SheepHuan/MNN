# OpenCL PIC Attention Optimization Log

This document is the working record for OrangePi OpenCL PIC/PagedAttention optimization. Future cacheblend/epic changes should update this file first: record the data, state the bottleneck, then list the next experiment.

## Current Baseline: 2026-06-11

Environment:

- Device: OrangePi 5 Plus, OpenCL backend.
- Model: `AI-ModelScope/Llama-3.2-1B-Instruct`, PIC graph-boundary export.
- Prompt protocol: explicit token/span, exactly 1024 prompt tokens.
- PIC token split: `doc=1010`, `prelude=7`, `suffix=7`.
- Score layer: `pic_recompute_score_layer_idx=1`.
- PIC requests: `max_tokens=0`, prefill-only.
- Normal baseline: ordinary non-PIC MNN export under `.cache/mnn-llm-export/AI-ModelScope__Llama-3.2-1B-Instruct`, `llm_bench -n 0`.
- Artifact: `.cache/output/mnn/artifacts/orangepi5plus`.
- Sweep script: `.cache/pic_opencl_1024_sweep.py`.
- Formal run dir: `.cache/latency_budget_20260611/opencl_pic_1024_current_20260611_092627`.

Remote cache policy update:

- Do not store new OrangePi experiment run dirs or PIC KV under the remote repo root `.cache`; the root filesystem filled during q-chunk sweeps.
- Use SSD-backed cache root for future OrangePi runs:

```text
/mnt/ssd/code/.cache/mnn_opencl_pic
```

- The user requested `/mnt/ssd/.cache`; current device permissions block creating that exact directory as `orangepi` (`/mnt/ssd` is root-owned and sudo requires a password). `/mnt/ssd/code` is writable by `orangepi`, so `/mnt/ssd/code/.cache/mnn_opencl_pic` is the active SSD cache root.
- `.cache/pic_opencl_1024_sweep.py` supports `PIC_SWEEP_REMOTE_CACHE_ROOT`; set it to `/mnt/ssd/.cache/mnn_opencl_pic` if that directory is later created and chowned to `orangepi`.
- Remote run dirs go under `$PIC_SWEEP_REMOTE_CACHE_ROOT/latency_budget_20260611/<tag>`.
- Remote PIC KV dirs go under `$PIC_SWEEP_REMOTE_CACHE_ROOT/kvshare/<tag>/kv`.

Build and deploy:

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=48 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/home/orangepi/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/orangepi5plus/
```

Formal sweep:

```bash
PIC_SWEEP_TAG=opencl_pic_1024_current_20260611_092627 \
PIC_SWEEP_RATIOS=0.10,0.20,0.30,0.40,0.50 \
PIC_SWEEP_MODES=cacheblend,epic \
PIC_SWEEP_RUN_NORMAL=1 \
python3 .cache/pic_opencl_1024_sweep.py
```

Formal latency:

```text
model_name,mode,budget,latency_s,effective_tps,speedup_vs_normal,status,execution_mode,recompute_token_count,reuse_token_count
Llama-3.2-1B-Instruct@orangepi-opencl,normal-full-compute,full,6.968234,146.953,1.000,200,,,
Llama-3.2-1B-Instruct@orangepi-opencl,full-reuse,full,0.394073,2598.504,17.683,200,native-full-reuse,0,1010
Llama-3.2-1B-Instruct@orangepi-opencl,cacheblend,0.10,3.595284,284.818,1.938,200,native-cacheblend-graph-boundary,101,909
Llama-3.2-1B-Instruct@orangepi-opencl,cacheblend,0.20,7.325352,139.788,0.951,200,native-cacheblend-graph-boundary,202,808
Llama-3.2-1B-Instruct@orangepi-opencl,cacheblend,0.30,5.604620,182.706,1.243,200,native-cacheblend-graph-boundary,303,707
Llama-3.2-1B-Instruct@orangepi-opencl,cacheblend,0.40,8.132663,125.912,0.857,200,native-cacheblend-graph-boundary,404,606
Llama-3.2-1B-Instruct@orangepi-opencl,cacheblend,0.50,9.587439,106.806,0.727,200,native-cacheblend-graph-boundary,505,505
Llama-3.2-1B-Instruct@orangepi-opencl,epic,0.10,1.800076,568.865,3.871,200,native-epic-graph-boundary,101,909
Llama-3.2-1B-Instruct@orangepi-opencl,epic,0.20,5.737397,178.478,1.215,200,native-epic-graph-boundary,202,808
Llama-3.2-1B-Instruct@orangepi-opencl,epic,0.30,3.456773,296.230,2.016,200,native-epic-graph-boundary,303,707
Llama-3.2-1B-Instruct@orangepi-opencl,epic,0.40,4.909510,208.575,1.419,200,native-epic-graph-boundary,404,606
Llama-3.2-1B-Instruct@orangepi-opencl,epic,0.50,6.288845,162.828,1.108,200,native-epic-graph-boundary,505,505
```

Status:

- OpenCL graph-boundary path is stable for cacheblend and epic at 10/20/30/40/50%.
- Epic is faster than normal full-compute for all tested budgets.
- Cacheblend is faster at 10% and 30%, but slower at 20%, 40%, and 50%.
- Full-reuse is very fast and not the bottleneck.
- CPU semantic validation is treated as already passed; do not repeatedly rerun CPU unless a shared semantic path changes.

## Profile Detail: 2026-06-11

Profile run:

```bash
PIC_SWEEP_TAG=opencl_pic_1024_profile_cacheblend_20260611_092848 \
PIC_SWEEP_RATIOS=0.20,0.40,0.50 \
PIC_SWEEP_MODES=cacheblend \
PIC_SWEEP_RUN_NORMAL=0 \
PIC_SWEEP_NORMAL_LATENCY=6.968234 \
PIC_SWEEP_PROFILE=1 \
PIC_SWEEP_PROFILE_DETAIL=1 \
python3 .cache/pic_opencl_1024_sweep.py
```

Important measurement rule:

- Profile detail inserts `queue.finish()` around sub-ops. Use it only for bottleneck attribution, not for speedup.
- This run's server log was partially buffered before process stop. Use the layer rows for hotspot direction, but for complete layer counts start `pic_server` with line-buffered stdout, for example `stdbuf -oL -eL`.

Profile observations:

```text
hydrate rows: 40
hydrate total: 60.594 ms
hydrate max layer: 2.025 ms
hydrate fallback_tokens: 0
async_failed: false
```

Hydrate/direct PagedCache is not the current slowdown. The cacheblend slow budgets are dominated by attention compute, not async KV loading.

Sparse attention detail from the cacheblend profile:

```text
budget,active_query_rows,sparse_logged_layers,sparse_total_ms,qk_ms,softmax_ms,qkv_ms,pack_ms,rearrange_ms,qk_active_tiles/qk_rect_tiles,worst_layer
0.20,216,15,1994.369,1380.687,45.777,386.745,21.540,150.118,0.657,layer1 total=1410.280ms qk=1073.860ms
0.40,418,15,4747.057,3285.576,115.532,712.743,23.527,274.216,0.840,layer1 total=2922.189ms qk=2422.412ms
0.50,519,12,4880.051,3461.124,101.425,764.191,16.592,314.248,0.875,layer1 total=3205.625ms qk=2634.995ms
```

Selected cacheblend logical indices are scattered and reach the end of the 1024-token context:

```text
budget,count,min,max,chunk summary
0.20,202,7,1008,chunk0 max=81; chunk1 max=240; chunk2 max=806; chunk3 max=1008
0.40,404,7,1016,chunk0 max=70; chunk1 max=149; chunk2 max=259; chunk3 max=422; chunk4 max=665; chunk5 max=942; chunk6 max=1016
0.50,505,7,1016,chunk0 max=70; chunk1 max=144; chunk2 max=229; chunk3 max=345; chunk4 max=487; chunk5 max=668; chunk6 max=860; chunk7 max=1016
```

Interpretation:

- QK is the main bottleneck, roughly 69-71% of logged sparse attention time.
- QKV is secondary, roughly 15-19%.
- Pack and hydrate are small.
- `layer=1` dominates each profiled cacheblend request. This is the score-layer boundary shape: full-Q input, compact output, and active rows produced after scoring.
- High cacheblend budgets select rows scattered across late logical positions, so causal K range per chunk approaches full context. The existing `activeKvLen = max(selected_logical) + 1` optimization loses effectiveness when selected rows reach the end of the document.

## Optimization Candidates

### P0: Make Measurement Reliable

- Start profiled `pic_server` with `stdbuf -oL -eL` so profile logs are complete before kill.
- Add timing inside `runCacheBlendScoring`: persistent value read/map, score kernel, top-k kernel, selected-index readback.
- Keep formal latency and profile detail as separate runs.
- In each future update, record run dir, git diff summary, artifact path, env vars, and whether the run used profile detail.

### P1: Remove Score-Layer First-Sparse-Layer Overhead

Symptoms:

- The worst layer in cacheblend profile is always `layer=1`.
- `layer=1` QK alone is 1.07s / 2.42s / 2.63s for 20/40/50%.

Possible work:

- Add fixed Mali-friendly LWS for `matmul_qk_sparse_prefill_piece` and `matmul_qkv_sparse_prefill_piece`, bypassing request-critical localWS autotune for sparse PIC shapes.
- Prebuild/warm sparse shape buckets for 10/20/30/40/50% before formal timing. Warm-up must be reported separately and not counted as a request latency.
- Replace the unconditional `queue.finish()` after score-layer `rearrange_sparse_q` with a safer event dependency or narrower synchronization. Current code finishes when `queryRowsAreFull` to avoid Mali faults; this may be costing real request latency.
- Consider a dedicated score-layer full-Q/compact-output kernel path instead of reusing the generic sparse fast path shape.

### P2: Reduce QK Work for Scattered CacheBlend Rows

Symptoms:

- `qk_active_tiles/qk_rect_tiles` rises to 0.84-0.875 for 40/50%.
- Cacheblend-selected rows are sorted but sparse and reach positions near 1016, forcing large K ranges.

Possible work:

- Sweep `MNN_PAGED_ATTENTION_OPENCL_SPARSE_FAST_Q_CHUNK=16,32,64,128` on cacheblend 20/40/50. Smaller chunks should reduce per-piece `activeKvLen` for scattered rows, but may increase launch overhead.
- Add automatic q-chunk selection based on selected logical-index spread:
  - Use smaller chunks for cacheblend scattered indices.
  - Keep larger chunks for epic contiguous-prefix indices.
- Replace fixed compact chunks with range-aware chunks: split active rows where logical max jumps sharply, so late selected rows do not expand the K range for earlier rows.
- Record `qk_active_tiles/qk_rect_tiles` after each q-chunk/range experiment.

### P3: Optimize QKV After QK Improves

Symptoms:

- QKV is secondary but visible: 386ms / 713ms / 764ms in partial profile.

Possible work:

- Re-test `MNN_PAGED_ATTENTION_OPENCL_DIRECT_VALUE_PREFILL=1` for 1024-token cacheblend 20/40/50 after QK chunk tuning.
- Keep direct value prefill as a conditional heuristic; older 1488-token tests showed it can help cacheblend high budgets but hurt epic low/mid budgets.
- Do not enable it globally unless epic 10-50% remains faster.

### P4: Keep Async KV and Zero-Copy Semantics Intact

Current evidence:

- Hydrate profile: `fallback_tokens=0`, total about 60ms.
- `async persistent PIC cache read failed` did not appear.

Rules:

- Do not add host-side full KV staging for production OpenCL PIC.
- Keep persistent PIC cache source -> current request PagedCache source slots -> hydrate logical slots.
- If a run reports `target unavailable`, identify whether it belongs to warm-up/full-reuse layer0 registration or the measured cacheblend request before treating it as a cacheblend bottleneck.

### P5: Budget Guard Is Only a Last Resort

Do not silently change cacheblend semantics to make numbers look faster.

Allowed only if explicitly reported as a fallback/variant:

- `cacheblend-speed-guard`: if estimated sparse QK cost exceeds normal full-compute baseline, return an unsupported/fallback marker or route to another named algorithm.
- `cacheblend-windowed` or `cacheblend-prefix-regularized`: restrict selected rows to reduce late-position K range. This changes selection semantics and must not be reported as normal cacheblend.

## Future Update Template

Append each experiment in this format:

```text
## YYYY-MM-DD Experiment: <short name>

Goal:
- ...

Code/env:
- git diff summary:
- build command:
- server env:
- remote cache root:
- script/run dir:

Formal latency:
model_name,mode,budget,latency_s,effective_tps,speedup_vs_normal,status,execution_mode,recompute_token_count,reuse_token_count
...

Profile detail:
- hydrate:
- score:
- sparse QK:
- sparse softmax:
- sparse QKV:
- qk_active_tiles/qk_rect_tiles:

Conclusion:
- ...

Next:
- ...
```

## 2026-06-11 Experiment: Sectioned Attention Profile

Goal:

- Split cacheblend latency into PagedAttention, ScoreAttention, SparseAttention, and hydrate so we optimize the right part first.
- Keep run/log/KV on SSD cache root.

Code/env:

- Added profile detail timing to `runCacheBlendScoring`: total `us`, persistent value `read_us`, score kernel `score_kernel_us`, top-k `topk_us`, and final selected-index `readback_us`.
- Added line-buffered profile launch support to `.cache/pic_opencl_1024_sweep.py`.
- Build command:

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=48 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/home/orangepi/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/orangepi5plus/
```

- Remote cache root: `/mnt/ssd/code/.cache/mnn_opencl_pic`.
- Script/run dir: `.cache/latency_budget_20260611/opencl_pic_1024_ssd_profile_sections_20260611_094647`.
- Run command:

```bash
PIC_SWEEP_TAG=opencl_pic_1024_ssd_profile_sections_20260611_094647 \
PIC_SWEEP_RATIOS=0.20,0.40,0.50 \
PIC_SWEEP_MODES=cacheblend \
PIC_SWEEP_RUN_NORMAL=0 \
PIC_SWEEP_NORMAL_LATENCY=6.968234 \
PIC_SWEEP_PROFILE=1 \
PIC_SWEEP_PROFILE_DETAIL=1 \
PIC_SWEEP_LINE_BUFFER=1 \
python3 .cache/pic_opencl_1024_sweep.py
```

Formal/profile latency:

```text
mode,budget,latency_s,speedup_vs_normal,execution_mode,recompute_token_count,reuse_token_count
full-reuse,full,0.344422,20.232,native-full-reuse,0,1010
cacheblend,0.20,7.446428,0.936,native-cacheblend-graph-boundary,202,808
cacheblend,0.40,7.672876,0.908,native-cacheblend-graph-boundary,404,606
cacheblend,0.50,10.424140,0.668,native-cacheblend-graph-boundary,505,505
```

Attention section breakdown:

```text
budget,PagedAttention_ms,ScoreAttention_scoring_ms,ScoreAttention_layer1_attention_ms,ScoreAttention_total_ms,SparseAttention_layers_gt1_ms,Hydrate_ms
0.20,367.463,57.930,832.609,890.539,562.756,20.281
0.40,63.085,220.973,1896.577,2117.550,1683.952,19.045
0.50,65.037,345.919,3148.564,3494.483,2258.664,18.858
```

Notes:

- `PagedAttention_ms` is the score-layer-before boundary (`layer < score_layer_idx`); with `score_layer_idx=1` this is only layer 0 full-prompt PagedAttention in the chat request. The 0.20 run includes a cold shape/tune spike at 367 ms; stable repeats are about 63-65 ms.
- `ScoreAttention_scoring_ms` is the cacheblend value scoring/top-k step inside `PicScoreAttention`. It is dominated by `topk_us`: 53.918 ms / 218.196 ms / 341.962 ms for 20/40/50%.
- `ScoreAttention_layer1_attention_ms` is the layer 1 full-Q/compact-output sparse attention after scoring. This is the largest hotspot.
- `SparseAttention_layers_gt1_ms` is the sum of `PicSparseAttention` layers after the score layer.
- Hydrate is not a bottleneck: it stays around 19-20 ms with `fallback_tokens=0`.

ScoreAttention layer 1 detail:

```text
budget,total_ms,qk_ms,softmax_ms,qkv_ms,pack_ms,rearrange_ms,qk_active_tiles/qk_rect_tiles
0.20,832.609,590.302,1.960,126.289,1.442,112.345,0.657
0.40,1896.577,1534.965,6.318,234.890,1.018,118.224,0.840
0.50,3148.564,2608.496,10.586,206.806,1.665,292.198,0.875
```

SparseAttention layers after score detail:

```text
budget,total_ms,layers,qk_ms,softmax_ms,qkv_ms,pack_ms,rearrange_ms,qk_active_tiles/qk_rect_tiles
0.20,562.756,14,302.717,37.909,189.972,17.588,8.757,0.657
0.40,1683.952,14,850.420,93.464,459.156,21.169,21.738,0.840
0.50,2258.664,14,1055.712,130.805,682.856,17.333,21.951,0.875
```

Conclusion:

- The first optimization target is ScoreAttention layer 1 QK, especially the full-Q/compact-output path (`queryRowsAreFull=1`). This single layer costs 0.83 s / 1.90 s / 3.15 s for cacheblend 20/40/50%.
- The second target is SparseAttention layers > 1 QK/QKV. These are smaller per layer but large in aggregate.
- `cacheblend_score` top-k should be optimized after the attention kernels; it is visible at high budgets but not the primary cause.
- PagedAttention before score and hydrate are not the bottlenecks for high-budget cacheblend.

Next:

- Optimize ScoreAttention layer 1 separately from later SparseAttention. Candidate paths:
  - special full-Q/compact-output Q rearrange to remove the current `queue.finish()` and reduce `rearrange_sparse_q` cost;
  - fixed/tuned LWS for the score-layer `matmul_qk_sparse_prefill_piece`;
  - range-aware active-row chunks to reduce `qk_active_tiles/qk_rect_tiles` for scattered cacheblend rows.
- After ScoreAttention improves, optimize SparseAttention layers > 1 QK/QKV with the same range-aware chunking and optional direct value prefill.

## 2026-06-11 Experiment: FlashMask Range Pieces + MNN Tune Cache

Goal:

- Answer whether the slowdown is PagedAttention, ScoreAttention, or SparseAttention after moving OpenCL LWS tuning out of the measured request path.
- Apply a first FlashMask-style optimization without changing cacheblend semantics: split sparse active rows into range-aware pieces so each piece has a tighter causal K range.
- Use MNN's native OpenCL tuning cache instead of adding a separate online tuner in the PIC path.

Code/env:

- Added range-aware sparse prefill piece selection in `PagedAttentionBufExecution.cpp`.
  - The active logical indices are unchanged.
  - Each piece keeps `q_len <= q_chunk`.
  - Only `activeKvLen` per piece is tightened.
  - Runtime fallback: `MNN_PAGED_ATTENTION_OPENCL_SPARSE_FAST_RANGE_PIECES=0` restores fixed q-chunk pieces.
- Added `Llm::updateRuntimeCache()` and `POST /v1/tune/update_cache` to write MNN RuntimeManager cache through the existing MNN cache path.
- Added `PIC_SWEEP_WARM_TUNE=1` to `.cache/pic_opencl_1024_sweep.py`.
  - Warm requests are not written into the formal CSV.
  - After warm requests, the script calls `/v1/tune/update_cache`.
  - Follow-up runs can start a fresh server and rely on the saved `mnn_cachefile.bin`.
- Build command:

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=48 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/home/orangepi/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/orangepi5plus/
```

Formal warm-tune run:

```bash
PIC_SWEEP_TAG=opencl_pic_1024_flashmask_warmtune_20260611_101100 \
PIC_SWEEP_RATIOS=0.10,0.20,0.30,0.40,0.50 \
PIC_SWEEP_MODES=cacheblend,epic \
PIC_SWEEP_RUN_NORMAL=0 \
PIC_SWEEP_NORMAL_LATENCY=6.968234 \
PIC_SWEEP_WARM_TUNE=1 \
PIC_SWEEP_LINE_BUFFER=1 \
python3 .cache/pic_opencl_1024_sweep.py
```

Warm-tune result:

```text
mode,budget,latency_s,speedup_vs_normal,execution_mode,recompute_token_count,reuse_token_count
full-reuse,full,0.525163,13.269,native-full-reuse,0,1010
cacheblend,0.10,4.150821,1.679,native-cacheblend-graph-boundary,101,909
cacheblend,0.20,6.366502,1.095,native-cacheblend-graph-boundary,202,808
cacheblend,0.30,3.733794,1.866,native-cacheblend-graph-boundary,303,707
cacheblend,0.40,4.753269,1.466,native-cacheblend-graph-boundary,404,606
cacheblend,0.50,6.626895,1.052,native-cacheblend-graph-boundary,505,505
epic,0.10,3.159694,2.205,native-epic-graph-boundary,101,909
epic,0.20,5.535596,1.259,native-epic-graph-boundary,202,808
epic,0.30,3.273624,2.129,native-epic-graph-boundary,303,707
epic,0.40,4.493137,1.551,native-epic-graph-boundary,404,606
epic,0.50,5.359912,1.300,native-epic-graph-boundary,505,505
```

Fresh-server cache-file repeat:

```bash
PIC_SWEEP_TAG=opencl_pic_1024_flashmask_cachefile_repeat_20260611_101802 \
PIC_SWEEP_RATIOS=0.10,0.20,0.30,0.40,0.50 \
PIC_SWEEP_MODES=cacheblend,epic \
PIC_SWEEP_RUN_NORMAL=0 \
PIC_SWEEP_NORMAL_LATENCY=6.968234 \
PIC_SWEEP_LINE_BUFFER=1 \
python3 .cache/pic_opencl_1024_sweep.py
```

Repeat result:

```text
mode,budget,latency_s,speedup_vs_normal,execution_mode,recompute_token_count,reuse_token_count
full-reuse,full,0.406432,17.145,native-full-reuse,0,1010
cacheblend,0.10,4.569623,1.525,native-cacheblend-graph-boundary,101,909
cacheblend,0.20,5.725607,1.217,native-cacheblend-graph-boundary,202,808
cacheblend,0.30,4.343375,1.604,native-cacheblend-graph-boundary,303,707
cacheblend,0.40,5.714816,1.219,native-cacheblend-graph-boundary,404,606
cacheblend,0.50,6.569642,1.061,native-cacheblend-graph-boundary,505,505
epic,0.10,3.163368,2.203,native-epic-graph-boundary,101,909
epic,0.20,5.337575,1.306,native-epic-graph-boundary,202,808
epic,0.30,3.360976,2.073,native-epic-graph-boundary,303,707
epic,0.40,4.062803,1.715,native-epic-graph-boundary,404,606
epic,0.50,6.008137,1.160,native-epic-graph-boundary,505,505
```

Profile detail runs:

```bash
PIC_SWEEP_TAG=opencl_pic_1024_flashmask_profile_10_30_20260611_101631 \
PIC_SWEEP_RATIOS=0.10,0.30 \
PIC_SWEEP_MODES=cacheblend \
PIC_SWEEP_RUN_NORMAL=0 \
PIC_SWEEP_NORMAL_LATENCY=6.968234 \
PIC_SWEEP_PROFILE=1 \
PIC_SWEEP_PROFILE_DETAIL=1 \
PIC_SWEEP_LINE_BUFFER=1 \
python3 .cache/pic_opencl_1024_sweep.py

PIC_SWEEP_TAG=opencl_pic_1024_flashmask_profile_20260611_101412 \
PIC_SWEEP_RATIOS=0.20,0.40,0.50 \
PIC_SWEEP_MODES=cacheblend \
PIC_SWEEP_RUN_NORMAL=0 \
PIC_SWEEP_NORMAL_LATENCY=6.968234 \
PIC_SWEEP_PROFILE=1 \
PIC_SWEEP_PROFILE_DETAIL=1 \
PIC_SWEEP_LINE_BUFFER=1 \
python3 .cache/pic_opencl_1024_sweep.py
```

Attention section breakdown after optimization:

```text
budget,PagedAttention_ms,ScoreAttention_scoring_ms,ScoreAttention_layer1_attention_ms,ScoreAttention_total_ms,SparseAttention_layers_gt1_ms,Hydrate_ms
0.10,246.468,18.531,22.575,41.106,337.876,18.905
0.20,150.686,57.895,52.477,110.372,740.025,19.554
0.30,68.741,124.080,71.238,195.318,1474.706,22.979
0.40,65.701,225.106,115.671,340.777,2220.714,23.202
0.50,100.777,343.987,172.663,516.650,2186.564,15.937
```

Sparse kernel detail after optimization:

```text
budget,L1_qk_ms,L1_qkv_ms,L1_qsplit,L1_tiles_ratio,Rest_qk_ms,Rest_softmax_ms,Rest_qkv_ms,Rest_layers,Rest_qsplit_avg,Rest_tiles_ratio
0.10,8.393,9.782,4,0.767,131.183,25.910,148.623,14,4.0,0.767
0.20,24.435,21.752,8,0.879,341.628,54.950,303.077,14,8.0,0.879
0.30,38.645,26.289,10,0.903,646.956,118.997,453.843,14,10.0,0.903
0.40,58.303,40.636,11,0.912,1003.818,144.895,682.275,14,11.0,0.912
0.50,77.208,50.990,13,0.928,1048.820,128.794,674.765,14,13.0,0.928
```

Conclusion:

- Before this change, cacheblend's slow point looked like `ScoreAttention layer=1`, but most of that was first sparse-shape OpenCL LWS tuning/driver work landing inside the measured request.
- The correct fix is to use MNN's native tuning cache flow:
  - warm target sparse shapes before formal measurement;
  - call `/v1/tune/update_cache`;
  - start formal runs with the saved cache loaded.
- After that, `ScoreAttention layer=1 attention` drops from 0.83 s / 1.90 s / 3.15 s at 20/40/50% to 52 ms / 116 ms / 173 ms.
- Hydrate remains small: about 16-23 ms, `fallback_tokens=0`, `async_failed=false`.
- The current bottleneck is now `SparseAttention layers > 1`, especially QK and QKV:
  - at 40%: later sparse layers cost 2.22 s, with QK 1.00 s and QKV 0.68 s;
  - at 50%: later sparse layers cost 2.19 s, with QK 1.05 s and QKV 0.67 s.
- The range-aware FlashMask-style splitter reduces rectangular QK work but does not solve scattered high-ratio rows by itself. At high budgets, active rows still reach late positions and `Rest_tiles_ratio` remains 0.91-0.93.
- `target unavailable` still appears for layer 0 in log scan, but this belongs to the known warm/full-reuse layer-0 persistent-source path; measured cacheblend requests have `async_failed=false` and direct hydrate rows with `fallback_tokens=0`.

Next:

- Treat MNN tune cache as mandatory for formal OrangePi OpenCL PIC measurements; do not report cold sparse-shape tuning as request latency.
- Optimize `PicSparseAttention` layers > 1 next:
  - fused sparse FlashAttention kernel for QK + causal softmax + QKV to remove QK/softmax/QKV intermediate buffers and launches;
  - stronger FlashMask-style row grouping/windowing only as a named semantic variant if it changes cacheblend selection;
  - direct value prefill for QKV after confirming epic 10-50% remains faster;
  - cacheblend top-k kernel is visible at 40/50% but still secondary to later sparse attention.

## 2026-06-11 Experiment: Direct Value Prefill A/B

Goal:

- Test whether the existing `MNN_PAGED_ATTENTION_OPENCL_DIRECT_VALUE_PREFILL=1` path is worth enabling by default for 1024-token cacheblend/epic.
- Keep the same graph-boundary semantics and MNN tune-cache flow; do not change CPU or persistent PIC cache source semantics.

Formal sweep:

```bash
PIC_SWEEP_TAG=opencl_pic_1024_direct_value_ab_20260611_102649 \
PIC_SWEEP_RATIOS=0.10,0.20,0.30,0.40,0.50 \
PIC_SWEEP_MODES=cacheblend,epic \
PIC_SWEEP_RUN_NORMAL=0 \
PIC_SWEEP_NORMAL_LATENCY=6.968234 \
PIC_SWEEP_WARM_TUNE=1 \
PIC_SWEEP_LINE_BUFFER=1 \
PIC_SWEEP_SERVER_ENV='MNN_PAGED_ATTENTION_OPENCL_DIRECT_VALUE_PREFILL=1' \
python3 .cache/pic_opencl_1024_sweep.py
```

Remote cache root stayed on SSD:

```text
/mnt/ssd/code/.cache/mnn_opencl_pic
```

Formal result:

```text
mode,budget,latency_s,speedup_vs_normal,execution_mode,recompute_token_count,reuse_token_count
full-reuse,full,0.536138,12.997,native-full-reuse,0,1010
cacheblend,0.10,3.524264,1.977,native-cacheblend-graph-boundary,101,909
cacheblend,0.20,5.599453,1.244,native-cacheblend-graph-boundary,202,808
cacheblend,0.30,3.687338,1.890,native-cacheblend-graph-boundary,303,707
cacheblend,0.40,5.093632,1.368,native-cacheblend-graph-boundary,404,606
cacheblend,0.50,6.748538,1.033,native-cacheblend-graph-boundary,505,505
epic,0.10,3.361028,2.073,native-epic-graph-boundary,101,909
epic,0.20,5.391049,1.293,native-epic-graph-boundary,202,808
epic,0.30,3.258290,2.139,native-epic-graph-boundary,303,707
epic,0.40,4.181462,1.666,native-epic-graph-boundary,404,606
epic,0.50,5.502349,1.266,native-epic-graph-boundary,505,505
```

Profile-detail A/B for cacheblend 20/40/50:

```bash
PIC_SWEEP_TAG=opencl_pic_1024_direct_value_profile_20260611_102921 \
PIC_SWEEP_RATIOS=0.20,0.40,0.50 \
PIC_SWEEP_MODES=cacheblend \
PIC_SWEEP_RUN_NORMAL=0 \
PIC_SWEEP_NORMAL_LATENCY=6.968234 \
PIC_SWEEP_PROFILE=1 \
PIC_SWEEP_PROFILE_DETAIL=1 \
PIC_SWEEP_LINE_BUFFER=1 \
PIC_SWEEP_SERVER_ENV='MNN_PAGED_ATTENTION_OPENCL_DIRECT_VALUE_PREFILL=1' \
python3 .cache/pic_opencl_1024_sweep.py
```

Profile-detail comparison against `opencl_pic_1024_flashmask_profile_20260611_101412`:

```text
case,query,section,total_ms,pack_ms,qk_ms,softmax_ms,qkv_ms
baseline,216,L1,52.477,1.210,24.435,3.534,21.752
baseline,216,Rest,740.025,17.581,341.628,54.950,303.077
direct,216,L1,55.856,1.240,25.189,5.129,22.595
direct,216,Rest,712.243,14.049,329.790,51.133,298.873
baseline,418,L1,115.671,1.442,58.303,8.909,40.636
baseline,418,Rest,2220.714,23.511,1003.818,144.895,682.275
direct,418,L1,119.834,1.356,59.088,11.217,41.237
direct,418,Rest,1885.031,14.158,834.214,133.739,578.022
baseline,519,L1,172.663,1.341,77.208,13.072,50.990
baseline,519,Rest,2186.564,17.741,1048.820,128.794,674.765
direct,519,L1,146.508,1.292,77.228,12.243,50.591
direct,519,Rest,2356.767,13.415,1093.032,158.856,707.871
```

Log scan:

```text
async_failed=false
target_unavailable=true
```

The `target unavailable` rows are still the known layer-0 warm/full-reuse persistent-source path; measured requests completed with status 200.

Follow-up instrumentation:

- Added `direct_value=0/1` to `op=sparse_prefill_attention_fast_qk_softmax_qkv` profile logs.
- Cross-compiled and synced OrangePi artifact after the profile-field change:

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=48 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/home/orangepi/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/orangepi5plus/
```

- Smoke profile:

```bash
PIC_SWEEP_TAG=opencl_pic_1024_direct_value_profile_field_20260611_103416 \
PIC_SWEEP_RATIOS=0.20 \
PIC_SWEEP_MODES=cacheblend \
PIC_SWEEP_RUN_NORMAL=0 \
PIC_SWEEP_NORMAL_LATENCY=6.968234 \
PIC_SWEEP_PROFILE=1 \
PIC_SWEEP_PROFILE_DETAIL=1 \
PIC_SWEEP_LINE_BUFFER=1 \
PIC_SWEEP_SERVER_ENV='MNN_PAGED_ATTENTION_OPENCL_DIRECT_VALUE_PREFILL=1' \
python3 .cache/pic_opencl_1024_sweep.py
```

- Result: request status 200; all 15 sparse attention profile rows contained `direct_value=1`, confirming the A/B actually entered the direct-value branch. The smoke latency is not a formal result because profile-detail inserts `queue.finish()`.

Conclusion:

- Direct value prefill keeps cacheblend and epic faster than normal full-compute in the formal A/B sweep.
- The improvement is mixed and not monotonic:
  - cacheblend 40% improves versus the prior fresh-cache repeat, but cacheblend 50% is slightly worse than the prior 6.57 s repeat;
  - epic 50% improves, but epic 10/20/40% are slightly slower.
- Profile-detail shows pack cost is only about 13-24 ms across later sparse layers. This is not the main bottleneck.
- QK and QKV remain the important costs. Direct value prefill can reduce later-layer QKV at 40%, but it regresses 50% in this run.
- Do not enable direct value prefill globally yet. Keep it as an explicit A/B or future heuristic after QK/fused attention work.
- Next mainline optimization remains fused sparse FlashAttention for `PicSparseAttention layers > 1`, especially reducing QK/softmax/QKV intermediate buffers and launches.

## 2026-06-11 Design: Fused Sparse FlashAttention Mainline

Goal:

- Reduce `PicSparseAttention layers > 1` cost without changing cacheblend/epic selection semantics.
- Keep `active_indices` / `sparse_query_logical_indices` exactly as produced by the score layer.
- Keep PageCache zero-copy, async persistent PIC cache source loading, and current PagedCache hydrate flow unchanged.

Why not a naive fused kernel:

- A kernel that computes one output dim-group at a time would have to recompute QK/softmax for every `head_dim/8` group. For Llama-3.2-1B `head_dim=64`, that means 8x duplicated QK/softmax and is likely slower than the current three-kernel path.
- A kernel that assigns one work-item to an entire `(query, head)` row avoids duplicate QK but serializes all K and D work too much for Mali.
- The experimental fused path should therefore be a block/streaming kernel, not just a mechanical softmax+QKV merge.

Semantics for the first fused experiment:

- Env-gated only, default off:

```text
MNN_PAGED_ATTENTION_OPENCL_SPARSE_FLASH=1
```

- Only enters after the existing sparse Q rearrange and key/value pack steps. The first version can still use `mTempQ`, `mTempK`, and packed value layout; it does not need to change PagedCache or hydrate.
- It replaces only the per-piece QK + sparse softmax + QKV section.
- It must support both score-layer and later-layer sparse attention because both have compact `mTempQ` after `rearrange_sparse_q` / `rearrange_q`.
- It must respect per-row causal bounds:
  - piece-level `activeKvLen` is only an upper bound;
  - each row still masks with its own `sparse_query_logical_indices[q]`.
- It must not reorder active rows or restrict selected rows to a window. Any semantic-changing window variant must be named separately, e.g. `cacheblend-windowed`.

Proposed V1 kernel shape:

- Start with `head_dim == 64`, `mHeadDim % 8 == 0`, and `mNumHead % mKvNumHead == 0`; fallback to current kernels otherwise.
- One workgroup handles one `(q4, head, batch)` tile.
- Local size candidate: 64 lanes over K blocks, tuned through MNN OpenCL Autotuning cache, not a custom PIC tuner.
- Maintain online softmax state per query lane:
  - `m[4]`: running max for four sparse Q rows;
  - `l[4]`: running normalizer;
  - `o[4][64]`: accumulated output vector.
- Stream K/V in blocks:
  - load packed K from `mTempK` and compute QK for q0..q3;
  - apply lane-specific causal mask using `sparse_query_logical_indices`;
  - update online softmax `(m, l, o)` and accumulate V.
- Store directly to `[batch, activeLen, head, head_dim]` output for compact rows.

Expected benefit:

- Removes `mTempQK` and `mTempSoftmax` traffic for sparse attention pieces.
- Reduces per-piece launches from three kernels to one kernel after pack.
- Keeps the same active row and causal mask semantics, so cacheblend/epic correctness remains comparable to current OpenCL sparse attention.

Risks:

- Local memory for `o[4][64]` partials may be too large if replicated per lane. The implementation should use a compact local reduction layout or split D while avoiding full QK recomputation.
- Mali occupancy may drop if the workgroup is too heavy. The first implementation must keep current kernels as fallback and report profile rows with a separate op name, e.g. `op=sparse_flash_attention`.
- Because profile-detail inserts `queue.finish()`, formal latency and fused-kernel profile runs must remain separate.

Validation plan:

1. Build with fused path compiled but disabled; confirm current cacheblend/epic formal results are unchanged.
2. Enable `MNN_PAGED_ATTENTION_OPENCL_SPARSE_FLASH=1` for cacheblend 10/20 only, profile-detail first:
   - confirm `sparse_prefill...` rows are replaced by `sparse_flash_attention` rows only for sparse layers;
   - confirm no `sparse_prefill... layer < score_layer_idx`;
   - confirm `async_failed=false` and no new PagedCache fallback.
3. Run formal 1024-token cacheblend/epic 10/20/30/40/50 with normal baseline fixed at `6.968234s`.
4. If any budget is slower than current warm-tune repeat, keep the fused path behind the env flag and record the bottleneck.
5. Only after fused QK/softmax/QKV is stable, re-test `MNN_PAGED_ATTENTION_OPENCL_DIRECT_VALUE_PREFILL=1` as a secondary heuristic.

## 2026-06-11 Decision: Keep LWS Tuning in MNN Autotuning

Question:

- Should sparse PIC LWS tuning / driver scheduling overhead be handled by MNN's own tune flow instead of a custom PIC online tuner?

Decision:

- Yes. Do not add a separate online LWS tuner in the PIC server request path.
- Use the existing MNN OpenCL Autotuning cache for all tunable sparse attention kernels:
  - `RuntimeManager::setCache(.../mnn_cachefile.bin)` loads the cache during runtime creation.
  - Warm target sparse shapes through normal PIC requests.
  - `POST /v1/tune/update_cache` calls `RuntimeManager::updateCache()` and persists the tuned `AutotuningT` entries.
  - Formal latency runs should restart/reuse the saved cache so cold shape tuning is not counted as cacheblend/epic request latency.
- Fixed LWS or disabled tuning is only an A/B diagnostic, not the production path.

Implementation note:

- Existing sparse QK/QKV fast kernels already call `localWS3DDefault(...)` through the shared `run3D` helper, so their LWS choices enter the MNN cache.
- The experimental fused softmax+QKV kernel was adjusted to use the same `run3D` path instead of a hard-coded `{64,1,1}` local size. This keeps the fused experiment aligned with MNN's tune/cache flow while remaining env-gated and default-off.

## 2026-06-11 Experiment: CacheBlend 50% Regression After LWS-Cache Retune

Goal:

- Re-test the default OpenCL PIC path after moving the fused experiment kernel to MNN's `localWS3DDefault` / cache flow.
- Verify whether cacheblend and epic remain faster than normal full compute at 1024 tokens.

Code/env:

- Artifact: `.cache/output/mnn/artifacts/orangepi5plus/`
- Remote cache root: `/mnt/ssd/code/.cache/mnn_opencl_pic`
- Normal baseline fixed: `6.968234s`.
- Formal run:

```bash
PIC_SWEEP_TAG=opencl_pic_1024_default_retune_after_lwscache_20260611_032213 \
PIC_SWEEP_RATIOS=0.10,0.20,0.30,0.40,0.50 \
PIC_SWEEP_MODES=cacheblend,epic \
PIC_SWEEP_RUN_NORMAL=0 \
PIC_SWEEP_NORMAL_LATENCY=6.968234 \
PIC_SWEEP_WARM_TUNE=1 \
PIC_SWEEP_LINE_BUFFER=1 \
PIC_SWEEP_REMOTE_CACHE_ROOT=/mnt/ssd/code/.cache/mnn_opencl_pic \
python3 .cache/pic_opencl_1024_sweep.py
```

Formal latency:

```text
mode,budget,latency_s,speedup_vs_normal,status,execution_mode,recompute_token_count,reuse_token_count
full-reuse,full,0.482712,14.436,200,native-full-reuse,0,1010
cacheblend,0.10,3.780370,1.843,200,native-cacheblend-graph-boundary,101,909
cacheblend,0.20,6.255473,1.114,200,native-cacheblend-graph-boundary,202,808
cacheblend,0.30,5.629952,1.238,200,native-cacheblend-graph-boundary,303,707
cacheblend,0.40,5.576222,1.250,200,native-cacheblend-graph-boundary,404,606
cacheblend,0.50,7.031057,0.991,200,native-cacheblend-graph-boundary,505,505
epic,0.10,4.121757,1.691,200,native-epic-graph-boundary,101,909
epic,0.20,5.955253,1.170,200,native-epic-graph-boundary,202,808
epic,0.30,3.555430,1.960,200,native-epic-graph-boundary,303,707
epic,0.40,4.906621,1.420,200,native-epic-graph-boundary,404,606
epic,0.50,6.245955,1.116,200,native-epic-graph-boundary,505,505
```

Status:

- Epic remains faster than normal at all tested budgets.
- Cacheblend 50% regressed slightly below normal (`0.991x`), so the goal is not complete under the strict "always faster" requirement.
- `log_scan.json`: `async_failed=false`; `target_unavailable=true` only reports the known layer-0 direct target registration path. This does not look like async KV or PageCache zero-copy failure.

Profile-detail for cacheblend 50:

```bash
PIC_SWEEP_TAG=opencl_pic_1024_default_cb50_profile_after_lwscache_20260611_032708 \
PIC_SWEEP_RATIOS=0.50 \
PIC_SWEEP_MODES=cacheblend \
PIC_SWEEP_RUN_NORMAL=0 \
PIC_SWEEP_NORMAL_LATENCY=6.968234 \
PIC_SWEEP_PROFILE=1 \
PIC_SWEEP_PROFILE_DETAIL=1 \
PIC_SWEEP_LINE_BUFFER=1 \
PIC_SWEEP_REMOTE_CACHE_ROOT=/mnt/ssd/code/.cache/mnn_opencl_pic \
python3 .cache/pic_opencl_1024_sweep.py
```

Breakdown:

```text
PagedAttention_layer0_ms=1111.914
ScoreAttention_scoring_ms=344.503
ScoreAttention_topk_ms=341.857
ScoreAttention_layer1_attention_ms=181.984
ScoreAttention_layer1_qk_ms=75.981
ScoreAttention_layer1_qkv_ms=48.885
SparseAttention_layers_gt1_ms=3030.656
SparseAttention_layers_gt1_qk_ms=1366.483
SparseAttention_layers_gt1_qkv_ms=884.708
SparseAttention_layers_gt1_softmax_ms=175.622
Hydrate_ms=21.143
```

Interpretation:

- Not hydrate: direct async PagedCache hydrate totals only ~21 ms and has `fallback_tokens=0`.
- Not async KV: no `async persistent PIC cache read failed`.
- Score top-k is visible at 50% (~342 ms) but still smaller than later sparse attention.
- Current main bottleneck is `PicSparseAttention layers > 1`, especially sparse QK and QKV at high active-row budgets.
- `qk_active_tiles/qk_rect_tiles` is ~0.928 for 50%, so range-aware chunking has little remaining K-range reduction when selected rows are scattered across late logical positions.

Negative A/B results:

```text
tag,env,mode,budget,latency_s,speedup_vs_normal,status
opencl_pic_1024_auto_direct50_warmtune_20260611_033830,auto direct-value heuristic,cacheblend,0.50,7.194416,0.969,200
opencl_pic_1024_auto_direct50_warmtune_20260611_033830,auto direct-value heuristic,epic,0.50,6.389405,1.091,200
opencl_pic_1024_qchunk16_cb50_20260611_034221,q_chunk=16 direct=0,cacheblend,0.50,7.987737,0.872,200
opencl_pic_1024_qchunk128_cb50_20260611_034407,q_chunk=128 direct=0,cacheblend,0.50,7.448315,0.936,200
opencl_pic_1024_qchunk256_direct_cb50_20260611_034520,q_chunk=256 direct=1,cacheblend,0.50,7.113182,0.980,200
```

Decisions from the A/B:

- Do not enable automatic direct-value prefill by default. It did not fix cacheblend 50% and can be non-monotonic.
- Do not change the default q chunk away from 64 based on these runs.
- Keep direct-value as an explicit A/B flag only.

Fused softmax+QKV V0 guard:

- `opencl_pic_1024_fused_softmax_qkv_cb50_warmtune_20260611_034659` with `MNN_PAGED_ATTENTION_OPENCL_SPARSE_FUSED_SOFTMAX_QKV=1` failed with OpenCL `CL_OUT_OF_RESOURCES (-14)` at 50% active rows.
- Added a resource guard: fused softmax+QKV only enters when `activeLen <= MNN_PAGED_ATTENTION_OPENCL_SPARSE_FUSED_SOFTMAX_QKV_MAX_ACTIVE`, default `256`.
- Guard smoke:

```text
opencl_pic_1024_fused_guard_cb50_smoke_20260611_034932
cacheblend,0.50,8.222604,0.847,200
```

- The smoke confirms the env-gated fused path no longer crashes at 50%; it falls back instead. It is not a performance improvement.

Next optimization direction:

- A small LWS/direct/chunk heuristic is not enough for cacheblend 50%.
- The remaining work should focus on a real sparse QK/QKV kernel improvement:
  - fused sparse FlashAttention that does not duplicate QK per output dim group;
  - or a stronger FlashMask-style segmentation that reduces `qk_active_tiles/qk_rect_tiles` for scattered late rows without changing selected active rows.
- Any windowing or re-ranking of cacheblend selected rows would change semantics and must be reported as a separate named variant, not normal cacheblend.

## 2026-06-11 Update: Confirm Later SparseAttention Bottleneck

User constraint update:

- Default score layer is `layer=1`. Keep this fixed.
- `layer 0` is score-before full compute.
- `layer 1` is `PicScoreAttention` scoring/top-k plus compact output.
- `layer >= 2` is later `PicSparseAttention` over compact active rows.

Fresh repeat after source-slot reserve:

```text
tag=opencl_pic_1024_source_slot_reserve_cachefile_repeat_20260611_1232
normal,full,6.968234,1.000
full-reuse,full,1.035310,6.731
cacheblend,0.10,4.187917,1.664
cacheblend,0.20,6.196139,1.125
cacheblend,0.30,4.680493,1.489
cacheblend,0.40,5.594163,1.246
cacheblend,0.50,7.188658,0.969
epic,0.10,3.761944,1.852
epic,0.20,5.863076,1.188
epic,0.30,3.706188,1.880
epic,0.40,4.979558,1.399
epic,0.50,6.012580,1.159
```

Stability scan:

```text
target_unavailable=false
async_failed=false
error_lines=[]
```

cacheblend 50% profile-detail:

```text
tag=opencl_pic_1024_source_slot_reserve_cb50_profile_20260611_1236
cacheblend_score layer=1 pic_tokens=1010 top_k=505 us=5783 read_us=3260 score_kernel_us=1505 topk_us=863 readback_us=55
ScoreAttention_layer1_attention_ms=183.203
SparseAttention_layers_gt1_ms=2958.786
SparseAttention_layers_gt1_qk_ms=1285.347
SparseAttention_layers_gt1_qkv_ms=836.536
Hydrate_ms=63.654
qk_rect_tiles=14561
qk_active_tiles=13516
qk_active_tiles/qk_rect_tiles=0.928
```

Conclusion:

- The remaining failure is narrow: epic 10-50% is faster than normal, but cacheblend 50% is still slower than normal by about 220 ms.
- This is not PageCache/zero-copy, async persistent PIC KV loading, hydrate, or cacheblend score/top-k.
- The current bottleneck is later `PicSparseAttention` at `layer >= 2`, especially per-layer sparse QK and QKV.
- At 50%, active query rows are 519 total rows and selected logical positions reach the end of the 1024-token context. The real causal K range is therefore close to full length for many rows.

Why FlashMask-style range pieces are not enough:

- The current range-aware splitter already tightens each sparse piece's K upper bound without changing active rows.
- For cacheblend 50%, `qk_active_tiles/qk_rect_tiles ~= 0.928`, leaving only about 7.2% rectangular-piece waste to remove.
- A row-level lower-bound check on the same active rows shows 4-row group causal waste is only about 0.6%. Splitting QK into 1-row/2-row groups would therefore mostly add launch overhead and not save enough math.
- FlashMask helps when many selected rows are early or clustered and piece-level K ranges can shrink. cacheblend 50% selects many rows across late positions, so the true work is not the mask waste; it is the actual active-row x long-K attention.
- Any stronger method that changes selected-row distribution, clamps windows, or regularizes toward prefix rows changes cacheblend semantics and must be reported as a named variant, not normal cacheblend.

TODO:

1. Optimize later `PicSparseAttention` sparse QK/QKV kernels directly.
2. Revisit sparse FlashAttention as a real fused kernel that streams QK/softmax/QKV once and does not recompute QK per output-dim group.
3. Keep the fused path env-gated and default-off until it beats the current sparse fast path for cacheblend/epic 10/20/30/40/50.
4. Continue recording for each profile: layer, active rows, q_split, q_chunk, QK/QKV/softmax us, `qk_active_tiles/qk_rect_tiles`, hydrate us, and `async_failed`.
5. Preserve the current PagedCache zero-copy and async loading path; do not add host KV staging or scratch `.k/.v` to mask the attention bottleneck.

## 2026-06-11 Experiment: Sparse FlashAttention Row64 V1

Goal:

- Implement a real fused sparse attention backend kernel for later `PicSparseAttention` layers.
- Keep score layer fixed at `layer=1`.
- Do not change active row semantics, PagedCache zero-copy, or async persistent PIC KV loading.

Code:

- Added `sparse_flash_attention_row64` in `source/backend/opencl/execution/cl/attention_buf.cl`.
- Added `PagedAttentionBufExecution::ensureSparseFlashKernel()`.
- New env gate:

```text
MNN_PAGED_ATTENTION_OPENCL_SPARSE_FLASH_ATTENTION=1
```

- The path only enters when:
  - sparse query is compact-Q (`full_q=0`);
  - `head_dim == 64`;
  - current layer is later `PicSparseAttention` (`layer >= 2` in the current graph-boundary model).
- `layer=1` score-layer full-Q/compact-output remains on the existing sparse fast path.
- The kernel uses one 64-lane workgroup per active row/head. Each lane streams a subset of K, computes QK once for each K, maintains local online softmax state, and reduces `(m,l,o)` across lanes.
- V0 scalar accumulator used `local_o[64 lanes][64 dims]` and ran, but was slower than the old path.
- V1 changed accumulator storage to `float8` vectors (`64 lanes x 8 float8`) and reduced local-memory loops from 64 to 8.
- Added profile op name:

```text
op=sparse_flash_attention
```

Build/deploy:

```bash
python3 opencl_codegen.py .

MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=48 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/home/orangepi/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/orangepi5plus/
```

Profile smoke:

```text
tag=opencl_pic_1024_sparse_flash_vec8_cb10_profile_20260611_125849
env=MNN_PAGED_ATTENTION_OPENCL_SPARSE_FLASH_ATTENTION=1 MNN_PAGED_ATTENTION_OPENCL_DIRECT_VALUE_PREFILL=0
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
```

Key rows:

```text
layer=1 old score-layer path:
op=sparse_prefill_attention_fast_qk_softmax_qkv layer=1 query=115 full_q=1 us=597808 qk_us=359129 qkv_us=171004

later layers with sparse flash:
layer=4 flash_us=11726
layer=5 flash_us=12196
layer=9 flash_us=13144
layer=15 flash_us=12060
```

Interpretation:

- V1 sparse flash is slower than ideal at score layer because score layer is intentionally not using it (`full_q=1`).
- For later compact-Q layers, V1 flash is materially faster than V0 and faster than the old cb10 later-layer QK/softmax/QKV profile rows.
- No OpenCL resource failure occurred at 10% smoke.

Formal latency:

```text
tag=opencl_pic_1024_sparse_flash_vec8_formal_20260611_130036
env=MNN_PAGED_ATTENTION_OPENCL_SPARSE_FLASH_ATTENTION=1
remote_cache_root=/mnt/ssd/code/.cache/mnn_opencl_pic
normal,full,6.968234,1.000
full-reuse,full,0.764264,9.118
cacheblend,0.10,3.780810,1.843
cacheblend,0.20,6.638145,1.050
cacheblend,0.30,3.903327,1.785
cacheblend,0.40,4.756370,1.465
cacheblend,0.50,5.834781,1.194
epic,0.10,4.015643,1.735
epic,0.20,5.884426,1.184
epic,0.30,3.889955,1.791
epic,0.40,4.398280,1.584
epic,0.50,5.346845,1.303
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
```

Conclusion:

- This meets the strict 1024-token requirement for the env-gated sparse flash path: cacheblend and epic are faster than normal full-compute at 10/20/30/40/50%.
- cacheblend 50% improved from the source-slot-reserve fresh repeat `7.188658s / 0.969x` to `5.834781s / 1.194x`.
- The main remaining OpenCL attention target is score layer `layer=1`, not later `layer >= 2` sparse layers.
- Keep `MNN_PAGED_ATTENTION_OPENCL_SPARSE_FLASH_ATTENTION` env-gated until one more fresh repeat confirms stability, then consider making it the default compact-Q `PicSparseAttention` path.

Direct-value note:

- Added a narrow auto direct-value heuristic for cacheblend high-budget sparse layers.
- `MNN_PAGED_ATTENTION_OPENCL_DIRECT_VALUE_PREFILL=0/1` still explicitly disables/enables it.
- This preserves A/B control while letting high-budget cacheblend avoid unnecessary packed value staging when slot table identity makes direct PagedCache value reads safe.

## 2026-06-11 Implementation: Make Sparse FlashAttention the Only OpenCL Sparse Fast Path

Goal:

- Promote the latest fastest OpenCL PIC sparse attention implementation to the default path.
- Remove the old non-improving OpenCL sparse implementation and env/fallback switches so later maintenance does not split across stale paths.
- Keep score-layer semantics, PagedCache zero-copy, and async persistent PIC cache source loading unchanged.

Code changes:

- `PicSparseAttention` compact-Q sparse fast path now always uses `sparse_flash_attention_row64`.
- Removed the env gates for sparse flash / fused sparse softmax-QKV / manual direct-value from `PagedAttentionBufExecution`.
- Removed old sparse OpenCL kernel members and build logic:
  - `matmul_qk_sparse_prefill_piece`
  - `matmul_qkv_sparse_prefill_piece`
  - `matmul_softmax_qkv_sparse_prefill_piece`
  - `rearrange_sparse_q`
- Regenerated `attention_buf_mnn_cl.cpp` and `opencl_source_map.hpp` from `attention_buf.cl`.
- Direct PagedCache value prefill remains only as a built-in narrow heuristic for cacheblend high-budget sparse layers with identity slot table.

Current production behavior:

- `layer=1` score layer is still full-Q/compact-output and does not enter sparse flash.
- `layer >= 2` compact-Q `PicSparseAttention` uses fused sparse flash by default.
- Profile logs for the later sparse fast path should now be `op=sparse_flash_attention`; seeing the old `op=sparse_prefill_attention_fast_qk_softmax_qkv` means the artifact is stale.

Next:

- Cross-compile and run a no-env fresh repeat on OrangePi:
  - no `MNN_PAGED_ATTENTION_OPENCL_SPARSE_FLASH_ATTENTION`;
  - no `MNN_PAGED_ATTENTION_OPENCL_DIRECT_VALUE_PREFILL`;
  - same 1024-token cacheblend/epic 10/20/30/40/50 sweep;
  - normal baseline fixed at `6.968234s`.
- Expected acceptance criterion remains: cacheblend and epic are faster than normal full-compute at every tested budget.

Validation:

```text
tag=opencl_pic_1024_sparse_flash_default_repeat_20260611_052347
server_env=<none for sparse flash/direct-value>
normal,full,6.968234,1.000
full-reuse,full,0.854002,8.160
cacheblend,0.10,4.116973,1.693
cacheblend,0.20,6.318665,1.103
cacheblend,0.30,4.174995,1.669
cacheblend,0.40,5.634095,1.237
cacheblend,0.50,6.424953,1.085
epic,0.10,3.795829,1.836
epic,0.20,5.948663,1.171
epic,0.30,3.831606,1.819
epic,0.40,4.720560,1.476
epic,0.50,5.842008,1.193
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
```

Default-path profile smoke:

```text
tag=opencl_pic_1024_sparse_flash_default_cb10_profile_20260611_052724
server_env=<none for sparse flash/direct-value>
old_sparse_prefill_rows=0
sparse_flash_rows=14
sparse_flash_layers=2..15
score_layer_row: op=row layer=1 query=1024 attn=115 kv_write=1024 kv_len=1024 sparse=1 full_q=1 us=99893
sparse_flash_total_ms=192.641
sparse_flash_pack_total_ms=20.316
direct_value_values=[0]
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
```

Conclusion:

- The speedup survives the default-path cleanup: cacheblend and epic remain faster than normal full-compute for every tested 10/20/30/40/50 budget.
- The new artifact no longer needs or recognizes the sparse flash/direct-value env switches in production source.
- The later sparse attention path is now exactly the intended default: score layer full-Q stays on row correctness path, compact-Q `layer >= 2` goes through fused sparse flash.
- This repeat is somewhat slower than the earlier best env-gated sparse-flash formal run at several budgets, but still much faster than the pre-flash source-slot repeat at cacheblend 50% (`7.188658s -> 6.424953s`) and satisfies the hard acceptance criterion.

## 2026-06-11 Analysis: Current Layer-Type Bottleneck After Default Sparse Flash

Profile inputs:

- Low-budget smoke: `opencl_pic_1024_sparse_flash_default_cb10_profile_20260611_052724`
- High-budget smoke: `opencl_pic_1024_sparse_flash_default_cb50_profile_20260611_053802`
- Both use default code path with no sparse-flash/direct-value env.
- Profile detail inserts `queue.finish()`, so use only for attribution, not formal latency.

cacheblend 10% measured request attention sections:

```text
layer0_full_attention=636.842 ms
score_layer_row_attention=99.893 ms
later_sparse_flash_attention_total=238.784 ms
hydrate=21.159 ms
cacheblend_score_topk=5.267 ms
```

cacheblend 50% measured request attention sections:

```text
layer0_full_attention=770.269 ms
score_layer_row_attention=1001.957 ms
later_sparse_flash_attention_total=2321.259 ms
later_sparse_flash_compute=1717.448 ms
later_sparse_pack=11.605 ms
later_sparse_rearrange=26.974 ms
hydrate=20.296 ms
cacheblend_score_topk=5.600 ms
qk_active_tiles/qk_rect_tiles=0.9282
worst_sparse_flash_layer=7 total=268.213 ms flash=228.374 ms
```

Conclusion:

- At low active-row budgets, later sparse flash is no longer the dominant request cost. Remaining latency is mostly outside the later sparse attention kernels: score-before full compute, score-layer row path, and dense graph work not covered by PagedAttention profile rows.
- At high active-row budgets, the main attention bottleneck is again `PicSparseAttention layer >= 2`, but now inside the fused sparse flash kernel rather than the old QK/softmax/QKV split path. The score layer `layer=1` row path is the second-largest attention cost.
- `PagedCache` hydrate, async persistent PIC cache source loading, score kernel, and top-k are not current bottlenecks: hydrate is about 20 ms and cacheblend score/top-k about 5-6 ms in the cb50 profile.
- FlashMask/range splitting alone is not enough for cb50 because `qk_active_tiles/qk_rect_tiles` remains about 0.928. Most work is real selected-row x long-K causal attention, not rectangular mask waste.

Next optimization priority:

1. Optimize high-budget `layer >= 2` sparse flash kernel:
   - reduce local-memory/barrier overhead in online softmax reduction;
   - consider 32-lane vs 64-lane variants through MNN tune cache, not per-request env switches;
   - improve Q/K/V vectorized loads and accumulator layout;
   - avoid increasing QK recomputation across output dim groups.
2. Add a dedicated score-layer full-Q/compact-output flash-style path for `layer=1`:
   - keep full K/V write for scoring semantics;
   - gather/read active Q rows by logical index;
   - output compact rows;
   - do not route `layer=1` through the compact-Q-only sparse flash kernel.
3. Instrument non-attention dense graph work after the score boundary:
   - verify QKV projection, o_proj, norm, MLP, and residual kernels operate on compact active rows after `layer=1`;
   - if any still process full 1024 rows, make that the next graph-level optimization.
4. Optimize `layer=0` full PagedAttention only after the above:
   - it is required by score-before full compute semantics, so it cannot be sparsified;
   - the goal is to reduce PagedAttention overhead versus normal full attention while preserving PagedCache write/read semantics.

## 2026-06-11 Implementation: Sparse Flash Lane Tuning + Score-Layer Flash

Goal:

- Continue P0/P1 from the current bottleneck analysis.
- Try 32-lane vs 64-lane sparse flash variants without reintroducing env-gated production fallbacks.
- Replace the slow `layer=1` full-Q/compact-output row kernel with a dedicated flash-style path that still reads full query rows by logical active index.

Code changes:

- Added `sparse_flash_attention_row32` beside the existing `sparse_flash_attention_row64`.
- `PagedAttentionBufExecution` now builds both row32/row64 sparse flash kernels by default.
- Lane selection is a fixed shape/plan heuristic:
  - active rows `<384`: row64.
  - cacheblend with selected PIC ratio `>=50%`: row64, because scattered high-budget rows were slower/unstable with row32 in formal repeats.
  - other high-budget sparse layers: row32.
- Extended the sparse flash kernel signature with `output_seq_len` and `query_rows_are_full`.
- `PicScoreAttention layer=1` may now use `op=score_flash_attention`:
  - K/V write and cacheblend score/top-k still happen on the full prompt.
  - active compact output rows are produced after scoring.
  - query loads use `q_row = sparse_query[q]` when `full_q=1`, so this is not the compact-Q later-layer kernel misapplied to score layer.
- Later `PicSparseAttention layer>=2` remains `op=sparse_flash_attention`.
- Regenerated `attention_buf_mnn_cl.cpp` and `opencl_source_map.hpp`.

Important failed/negative result:

```text
tag=opencl_pic_1024_sparse_flash_row32_formal_20260611_1401
cacheblend,0.50,7.029228,0.991
```

Pure `activeLen>=384 -> row32` improved some profile sub-ops but made cacheblend 50% fail the hard "faster than normal" requirement. The production heuristic therefore keeps cacheblend selected>=50% on row64.

Score-layer flash profile:

```text
tag=opencl_pic_1024_score_flash_cb50_profile_20260611_1424
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
layer0_full_attention=1076.265 ms
cacheblend_score_topk=5.980 ms
score_flash_attention layer=1 total=186.431 ms flash=110.222 ms full_q=1 lane64_pieces=13
later_sparse_flash_attention_total=1783.178 ms
later_sparse_flash_compute=1326.024 ms
```

Compared with the previous default cb50 profile, score layer attention dropped from about `0.79-1.00s` on `op=row` to `0.186s` on `op=score_flash_attention`.

Formal sweep:

```text
tag=opencl_pic_1024_score_flash_formal_20260611_1427
server_env=<none for sparse flash/direct-value>
remote_cache_root=/mnt/ssd/code/.cache/mnn_opencl_pic
normal,full,6.968234,1.000
full-reuse,full,0.840850,8.287
cacheblend,0.10,4.159067,1.675
cacheblend,0.20,5.922973,1.176
cacheblend,0.30,4.001684,1.741
cacheblend,0.40,4.801796,1.451
cacheblend,0.50,5.968761,1.167
epic,0.10,3.576207,1.948
epic,0.20,5.793736,1.203
epic,0.30,3.626172,1.922
epic,0.40,4.340059,1.606
epic,0.50,5.340461,1.305
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
```

Conclusion:

- The score-layer flash path restores clear margin at cacheblend 50%: default repeat `6.424953s -> 5.968761s`, and the slower post-row32 repeats `7.4-7.6s -> 5.97s`.
- cacheblend and epic are again faster than normal full-compute at every tested 10/20/30/40/50 budget.
- Hydrate/async load/score-topk are still not bottlenecks.
- Remaining cb50 gap is mostly non-score later sparse flash plus non-attention graph work. P2 is now more important: profile QKV projection, o_proj, MLP, norm, residual after the score layer to confirm all dense kernels operate on compact active rows.
