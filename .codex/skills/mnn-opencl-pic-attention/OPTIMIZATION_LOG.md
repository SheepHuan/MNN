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

## 2026-06-11 P2: Graph-Level Dense Profile

Goal:

- Confirm whether the non-attention dense graph after `score_layer_idx=1` still processes full 1024 rows or correctly switches to compact active rows.
- Keep this profile out of production timing. The graph profile forces output waits in the MNN debug callback, so its latency is attribution-only.

Code changes:

- Added PIC server env-gated graph op profiler:
  - `MNN_PIC_GRAPH_PROFILE=1` enables MNN debug callback before model load.
  - `MNN_PIC_GRAPH_PROFILE_TOP=N` controls printed top ops; use `1000` to dump all 690 ops for the 1024-token Llama-3.2-1B graph.
  - Output lines are `MNN_PIC_GRAPH_PROFILE_SUMMARY`, `MNN_PIC_GRAPH_PROFILE_TYPE`, and `MNN_PIC_GRAPH_PROFILE_OP`.
- Default execution is unchanged when the env var is not set.

Build / sync:

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/home/orangepi/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/orangepi5plus/
```

Validation / tags:

```text
tag=opencl_pic_1024_graph_profile_cb50_20260611_142948
server_env=MNN_PIC_GRAPH_PROFILE=1,MNN_PIC_GRAPH_PROFILE_TOP=80
attention_profile=MNN_PAGED_ATTENTION_PROFILE=1,MNN_PAGED_ATTENTION_PROFILE_DETAIL=1
cacheblend,0.50,7.022372,0.992  # wait-inflated, not formal latency

tag=opencl_pic_1024_graph_only_cb50_fullops_20260611_143125
server_env=MNN_PIC_GRAPH_PROFILE=1,MNN_PIC_GRAPH_PROFILE_TOP=1000
cacheblend,0.50,6.604463,1.055  # wait-inflated, not formal latency
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

tag=opencl_pic_1024_default_cb50_after_graphprof_20260611_143642
server_env=<none>
profile=false, detail=false, warm_tune=true
full-reuse,full,0.665456,10.471
cacheblend,0.50,6.035463,1.155
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
```

Final cb50 graph profile summary, request 4:

```text
Convolution total=3416.229 ms calls=113
PicSparseAttention total=1306.372 ms calls=14
PagedAttention layer0 total=914.879 ms calls=1
Raster total=372.903 ms calls=268
PicScoreAttention layer1 total=116.110 ms calls=1
```

Shape boundary result:

```text
conv full layer0:           442.276 ms, 7 ops
conv full layer1 score-pre:  39.809 ms, 3 ops
conv full layer>=2:           0.000 ms, 0 ops
conv active layer1+:       2923.752 ms, 102 ops
conv active layer>=2:      2764.124 ms, 98 ops
```

Important shape examples:

```text
layer0 full PagedAttention:
inputs=[1x1024x32x64]|[1x1024x8x64]|[1x1024x8x64]|[1x1x1024x1024]
outputs=[1x1024x2048]

layer1 PicScoreAttention full-Q / compact-output:
inputs=[1x1024x32x64]|[1x1024x8x64]|[1x1024x8x64]|[1x1x1024x1024]|[1]
outputs=[1x519x2048]|[519]

layer1 post-score MLP:
inputs=[519x2048x1x1] outputs=[519x8192x1x1]

layer>=2 sparse attention:
inputs=[1x519x32x64]|[1x519x8x64]|[1x519x8x64]|[1x1x519x1024]
outputs=[1x519x2048]
```

Conclusion:

- P2 did not find a graph-boundary bug where dense layers accidentally continue on full 1024 rows after the score layer.
- `layer 0` is full by required score-before full compute semantics.
- `layer 1` still has full-row pre-attention Q/K/V and rotary/norm work, which is expected because the score layer must write full K/V and score/top-k before compacting.
- After `PicScoreAttention`, layer 1 MLP and all `layer >= 2` QKV/o_proj/MLP/norm/residual paths run on compact active rows. For cb50 the active row count is 519.
- The remaining cb50 latency is therefore a real mix of mandatory full layer-0/score-pre work, later sparse flash attention, and active-row dense OpenCL convolution/MLP work. It is not a simple missing gather bug.

Next optimization order:

1. Continue P0 sparse flash kernel work for `layer >= 2`: reduce online softmax barrier/local-memory cost, improve Q/K/V load layout, and keep row32/row64 decisions in MNN tune/default heuristics.
2. Start a dense OpenCL attribution/optimization loop for active-row linear/conv shapes:
   - `[519,2048] -> [519,8192]` gate/up;
   - `[519,8192] -> [519,2048]` down;
   - `[519,2048] -> [519,2048]` q/o proj.
   If formal timing confirms this non-attention component is as large as the graph profile suggests, optimize MNN OpenCL linear/conv kernels for medium-M active rows.
3. Only then revisit layer-0 full PagedAttention / full dense overhead, because it is semantically mandatory and only one layer.

## 2026-06-11 Attention-Only: Why Sparse Flash Speedup Is Lower Than Expected

Focus:

- Ignore dense graph for this section and analyze only `PicSparseAttention layer >= 2`.
- Current production kernel is `sparse_flash_attention_row64/row32`: one active query row per workgroup, lanes scan K from `0..q_logical`.

Observed later-layer sparse flash work:

```text
cb10 default sparse flash profile:
active rows=115
later_sparse_flash_attention_total=238.784 ms
qk_active_tiles / qk_rect_tiles = 1367 / 1783 = 0.7667
qk_active_tiles / full_causal_tiles ~= 1367 / 32896 = 0.0416

cb50 score-flash profile:
active rows=519
later_sparse_flash_attention_total=1783.178 ms
later_sparse_flash_compute=1326.024 ms
qk_active_tiles / qk_rect_tiles = 13516 / 14561 = 0.9282
qk_active_tiles / full_causal_tiles ~= 13516 / 32896 = 0.4109
```

Interpretation:

- For cb50, attention QK work is not reduced to 50% of full causal attention; it is about 41% of full causal tiles because the selected active rows are scattered into late logical positions and each selected row still attends to almost the whole prefix.
- `qk_active_tiles / qk_rect_tiles ~= 0.928` means there is only about 7% rectangular/mask waste left in 4-row groups. FlashMask-style masking alone cannot deliver a large gain for cb50.
- The current sparse flash kernel already uses the exact row causal bound:

```c
active_kv_seq_len = clamp(q_logical + 1, 0, key_seq_len)
```

  Therefore it is not spending most time multiplying masked-out future K positions. The remaining work is real selected-row x long-K attention.

Why row-wise sparse flash under-delivers versus dense full attention:

1. Dense full attention is a better-shaped GEMM-like workload:
   - full kernels tile multiple Q rows and reuse K/V across rows;
   - row-wise sparse flash launches one workgroup per Q row/head and reloads K/V independently.
2. Current row kernel repeats Q loads inside the K loop:
   - `qv = vload4(query + ...)` is inside `for (k = lid; k < active_kv_seq_len; k += lanes)`;
   - Q is constant for the row/head, so this is redundant memory/conversion work.
3. Current row kernel updates `local_o` inside the K loop:
   - each K step reads/writes 8 `float8` accumulators in local memory;
   - the accumulators only need to become local memory before cross-lane reduction, so this likely costs avoidable LDS traffic and barriers pressure.
4. GQA K/V reuse is not exploited:
   - `kvh = h / NUMHEAD_GROUP_SIZE`, so four query heads share the same K/V head;
   - current z dimension is one query head, so the same K/V rows are loaded once per query head.
5. `q_split` is still inherited from old QK/softmax/QKV chunking:
   - cb50 uses `q_chunk=64`, `q_split=13`, so later sparse attention launches 13 flash kernels per layer;
   - fused row flash does not need q-piece rectangular QK buffers, so a single launch per layer may be enough. This mainly attacks driver scheduling overhead, not math.

Attention-only optimization candidates:

P0a. Single-launch sparse flash piece:

- For fused row flash, set q piece to `activeLen` instead of hard-limiting to 64.
- Expected benefit: fewer OpenCL launches (`13 -> 1` per sparse layer at cb50, `182 -> 14` for layers 2..15).
- Risk: low; local memory per workgroup is unchanged. Need verify MNN tune cache and formal cb10/cb50.

P0b. Private accumulators inside row64/row32:

- Replace per-K-loop `local_o[lid * 8 + d8]` updates with private `o0..o7` accumulators and store to local memory once before reduction.
- Keep a conservative variant first: private O only, do not hoist all Q yet.
- Expected benefit: reduce local memory traffic in the hot K loop.
- Risk: register pressure on Mali; keep env-free production only after row32/row64 variants are validated.

P0c. Hoist Q vector loads outside the K loop:

- Preload the 16 `float4` Q chunks for `head_dim=64` once per workgroup/lane before scanning K.
- Expected benefit: remove redundant Q loads/conversions repeated for every K step.
- Risk: additional private registers. This should be tested separately from P0b.

P1. Block-row sparse flash (`row2` / `row4`):

- One workgroup handles 2 or 4 adjacent active Q rows for the same head.
- Load each K/V element once and update multiple rows' softmax/output states.
- Use per-row causal checks inside the group. For cb50, the 4-row group overhead is small because active/rect is already `0.928`; the K/V reuse may be worth more than the extra masked rows.
- Expected benefit: recover some dense-attention-style K/V reuse and reduce workgroup/launch overhead.
- Risk: much higher register/local memory pressure because each row needs separate `m/l/o` state. Start with row2 before row4.

P2. GQA-head grouped sparse flash:

- Compute 2 or 4 query heads sharing the same KV head in one workgroup.
- Reuse K/V across GQA heads; keep independent softmax states per query head.
- Risk: even higher register pressure than row grouping. Consider only after row2/row4.

P3. Algorithmic K-range compression:

- Exact cacheblend cannot skip early K positions for a late selected row without changing semantics.
- To truly reduce K length, planner must produce a different algorithm variant such as `cacheblend-windowed` or `cacheblend-prefix-regularized`, where selected rows are biased/clustered or attention is approximated by a window/prefix policy.
- This must not be reported as ordinary cacheblend.

Immediate next experiment order:

1. Implement P0a single-launch sparse flash and run cb10/cb50 attention profile + formal cb50 smoke.
2. If stable, implement P0b private-O row64/row32 variant.
3. If P0b wins, test P0c Q-hoist as a separate variant.
4. Then design P1 row2 shared-K/V sparse flash for cb40/cb50, keeping current row64 as low-budget path.

## 2026-06-11 P0a Implementation: CacheBlend Single-Piece Sparse Flash

Goal:

- Reduce fused sparse flash launch overhead without changing attention semantics.
- Current row-wise sparse flash computes each active row with exact `q_logical + 1` K bound inside the kernel, so splitting Q into q64 pieces is no longer required to reduce K math. The old q64 split mostly adds kernel launches.

Code changes:

- In `PagedAttentionBufExecution::runSparseFastPrefill`:
  - if `mMeta->cacheblend_score_ready` is true, use single-piece sparse flash:

```text
q_chunk = activeLen
q_split = 1
pieces = _buildFixedSparsePieces(..., activeLen)
```

  - if not score-ready, keep the previous range-aware q64 pieces. This preserves epic / non-cacheblend behavior.
  - added `qk_row_tiles` to detail profile to show exact row-wise K span estimate; after single-piece scheduling, old `qk_rect_tiles` is just the large one-piece rectangular estimate and should not be treated as real compute.

Why cacheblend-only:

- First trial applied single-piece to every sparse flash path.
- cb50 attention profile improved, but formal epic 10/30/40/50 moved slightly worse in that run.
- Final implementation therefore uses single-piece only for cacheblend score-ready paths and keeps epic on the existing range-aware q64 path.

Build / sync:

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/home/orangepi/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/orangepi5plus/
```

Attention profile, first all-mode single-piece trial:

```text
tag=opencl_pic_1024_sparse_flash_single_piece_cb50_profile_20260611_145116
profile=true, detail=true, warm_tune=true
cacheblend,0.50,6.093415,1.144  # profile-inflated

timed request:
score_flash_attention layer=1:
  q_chunk=519 q_split=1 lane64_pieces=1 total=94.111 ms flash=89.369 ms
later sparse_flash_attention layers 2..15:
  q_chunk=519 q_split=1 lane64_pieces=14
  total=1255.392 ms
  flash=1214.291 ms
  rearrange=23.370 ms
  pack=15.720 ms
  hydrate=19.348 ms
```

Compared with the previous `opencl_pic_1024_score_flash_cb50_profile_20260611_1424`:

```text
score_flash_attention: 186.431 ms -> 94.111 ms
later sparse_flash_attention_total: 1783.178 ms -> 1255.392 ms
later sparse_flash_compute/flash: 1326.024 ms -> 1214.291 ms
```

Formal all-mode single-piece trial:

```text
tag=opencl_pic_1024_sparse_flash_single_piece_formal_20260611_145310
cacheblend,0.10,3.964500,1.758
cacheblend,0.20,6.133227,1.136
cacheblend,0.30,3.913386,1.781
cacheblend,0.40,4.644832,1.500
cacheblend,0.50,5.658548,1.231
epic,0.10,3.691532,1.888
epic,0.20,5.789389,1.204
epic,0.30,3.675803,1.896
epic,0.40,4.389616,1.587
epic,0.50,5.439305,1.281
```

Final formal cacheblend-only single-piece:

```text
tag=opencl_pic_1024_sparse_flash_single_piece_cacheblend_only_formal_20260611_145659
profile=false, detail=false, warm_tune=true
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal,full,6.968234,1.000
full-reuse,full,0.712506,9.780
cacheblend,0.10,3.970299,1.755
cacheblend,0.20,6.061056,1.150
cacheblend,0.30,3.753421,1.857
cacheblend,0.40,4.609172,1.512
cacheblend,0.50,5.508688,1.265
epic,0.10,3.953986,1.762
epic,0.20,5.877313,1.186
epic,0.30,3.504080,1.989
epic,0.40,4.264059,1.634
epic,0.50,5.118367,1.361
```

Epic sanity profile:

```text
tag=opencl_pic_1024_sparse_flash_cacheblend_only_epic10_profile_20260611_145926
epic,0.10,3.855016,1.808  # profile-inflated
timed epic10 sparse_flash_attention:
  q_chunk=64 q_split=3 lane64_pieces=42
  later layers total=174.702 ms
  flash=141.025 ms
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
```

Conclusion:

- P0a is valid for cacheblend: cb50 formal improves from `5.968761s -> 5.508688s`, speedup vs normal `1.167x -> 1.265x`.
- The attention profile confirms the intended mechanism: cb50 later sparse flash `q_split=13 -> 1`, and later attention total `1783 ms -> 1255 ms` in detail mode.
- All tested cacheblend/epic 10/20/30/40/50 remain faster than normal full-compute.
- Epic is not using the single-piece path in the final heuristic; epic10 detail shows `q_chunk=64 q_split=3`. Remaining epic variation is likely outside this P0a change or normal OpenCL timing variance.

Next attention-only optimization:

1. P0b private output accumulators in `sparse_flash_attention_row64/row32`, so the hot K loop updates private `o0..o7` registers and writes local memory only before cross-lane reduction.
2. P0c Q-load hoist, tested separately because it may increase register pressure.
3. P1 row2 sparse flash to share K/V loads across adjacent active rows for cb40/cb50.

## 2026-06-11 P0b Implementation: Private Sparse-Flash Output Accumulators

Goal:

- Reduce local memory traffic inside the row-wise sparse FlashAttention hot K loop.
- Keep semantics unchanged: exact active logical row, exact causal bound `q_logical + 1`, same online softmax recurrence, same row32/row64 production heuristic.

Final code shape:

- `sparse_flash_attention_row64` and `sparse_flash_attention_row32` now keep each lane's output accumulator in private `COMPUTE_FLOAT8 o0..o7`.
- The K loop updates private accumulators directly.
- `local_o` is written once after the K loop, immediately before cross-lane reduction.
- No row64 local-Q cache is kept. The attempted `local_q[16]` variant was removed.

Important precision note:

- Do not hard-code `half4` in these kernels. MNN OpenCL chooses storage and compute types through build options:
  - precisionLevel 2: `FLOAT=half`, `COMPUTE_FLOAT=half`.
  - precisionLevel 0: `FLOAT=half`, `COMPUTE_FLOAT=float`.
  - fp32 path: both are float.
- Attention QK / online softmax math should use `COMPUTE_FLOAT*` and `CONVERT_COMPUTE_FLOAT*`, so it follows the runtime precision policy. Hard-coding `half4` would bypass the fp16-storage/fp32-compute mode and risks changing numerical behavior.

P0b attention profile:

```text
tag=opencl_pic_1024_sparse_flash_private_o_cb50_profile_20260611_070706
profile=true, detail=true, warm_tune=true
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

timed request:
score_flash_attention layer=1:
  total=53.085 ms
  flash=48.952 ms
later sparse_flash_attention layers 2..15:
  total=745.236 ms
  rearrange=30.627 ms
  pack=17.368 ms
  flash=693.913 ms
```

Compared with P0a single-piece profile:

```text
score_flash_attention: 94.111 ms -> 53.085 ms
later sparse_flash_attention_total: 1255.392 ms -> 745.236 ms
later sparse_flash flash: 1214.291 ms -> 693.913 ms
```

P0b formal sweep:

```text
tag=opencl_pic_1024_sparse_flash_private_o_formal_20260611_071022
profile=false, detail=false, warm_tune=true
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal,full,6.968234,1.000
full-reuse,full,0.743882,9.367
cacheblend,0.10,3.617224,1.926
cacheblend,0.20,6.015361,1.158
cacheblend,0.30,3.737483,1.864
cacheblend,0.40,4.069478,1.712
cacheblend,0.50,5.262682,1.324
epic,0.10,3.656841,1.906
epic,0.20,6.142038,1.135
epic,0.30,3.476549,2.004
epic,0.40,4.005558,1.740
epic,0.50,5.514270,1.264
```

Comparison against P0a cacheblend-only single-piece formal:

```text
cacheblend,0.10: 3.970299 -> 3.617224  (-8.9%)
cacheblend,0.20: 6.061056 -> 6.015361  (-0.8%)
cacheblend,0.30: 3.753421 -> 3.737483  (-0.4%)
cacheblend,0.40: 4.609172 -> 4.069478 (-11.7%)
cacheblend,0.50: 5.508688 -> 5.262682  (-4.5%)
epic,0.10:       3.953986 -> 3.656841  (-7.5%)
epic,0.20:       5.877313 -> 6.142038  (+4.5%)
epic,0.30:       3.504080 -> 3.476549  (-0.8%)
epic,0.40:       4.264059 -> 4.005558  (-6.1%)
epic,0.50:       5.118367 -> 5.514270  (+7.7%)
```

All cacheblend and epic ratios remain faster than normal full compute. Cacheblend improves across all tested budgets. Epic improves at 10/30/40 but regresses at 20/50 in this run; since the current user focus is cacheblend sparse attention and cacheblend high budget, P0b is kept as the default kernel simplification.

Rejected P0c local-Q cache:

- Tried `COMPUTE_FLOAT4 local_q[16]` to load each row's Q once per workgroup and reuse it in the K loop.
- A full-localQ variant made cacheblend50 formal worse (`5.262682s -> 5.709010s` in comparable full sweeps) despite sometimes improving detail-mode flash microseconds.
- A row64-only localQ variant improved some epic runs but still hurt cacheblend50 repeat (`5.634133s`) and was explicitly rejected.
- Final production code does not contain `local_q`; `rg local_q source/backend/opencl/execution/cl/attention_buf.cl source/backend/opencl/execution/cl/attention_buf_mnn_cl.cpp` returns no matches.

Final sync smoke after removing localQ:

```text
tag=opencl_pic_1024_sparse_flash_private_o_final_cb50_repeat_20260611_073012
profile=false, detail=false, warm_tune=true
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
cacheblend,0.50,5.624736,1.239
```

This repeat confirms the synced artifact is stable and faster than normal, but the full P0b formal sweep above remains the main comparison point because single-ratio runs show visible end-to-end variance.

## 2026-06-11 P2 Fix: PIC Compact Dense GEMM Fast Path

Problem:

- A repeated 20/30 budget profile showed that 20% recompute was slower end-to-end than 30%, even though attention work was lower.
- Attention detail was not the cause:
  - cacheblend20 score/later attention: `20.311 ms` / `252.640 ms`.
  - cacheblend30 score/later attention: `31.056 ms` / `409.080 ms`.
  - epic20 score/later attention: `19.172 ms` / `219.471 ms`.
  - epic30 score/later attention: `27.368 ms` / `359.014 ms`.
- Graph profile identified the cliff in dense `Convolution` / Linear after the score-layer compact boundary:
  - old request=5 cacheblend20: `Convolution=4964.176 ms`, `PicSparseAttention=253.992 ms`, `PicScoreAttention=26.444 ms`.
  - old request=6 cacheblend30: `Convolution=2335.675 ms`, `PicSparseAttention=406.522 ms`, `PicScoreAttention=34.198 ms`.
  - The pathological shape was compact rows around `[216x2048x1x1]` and `[216x8192x1x1]`; attention was behaving as expected.

Implementation:

- Added PIC-prefixed OpenCL low-memory 1x1 Conv kernels:
  - `pic_gemm_b4_c8_int4_buf`
  - `pic_gemm_b4_c8_int8_buf`
- The kernel bodies share the existing quantized `gemm_b4_c8_*` implementation through inline impl functions, so quantization math and output layout stay unchanged.
- `ConvBufLowMemoryExecution` now routes large-channel compact rows (`globalY > 16 && globalY <= 512`, `inputChannels/outChannel >= 1024`) to the `pic_gemm_b4_c8_*` kernels by default.
- The fast path bypasses the generic `convBufLowMemory_*` FP-weight decision and gives PIC compact dense shapes independent MNN tune keys by appending `pic_m<active_rows>` to the LWS key. This prevents old generic GEMM tune entries from being reused for 216-row compact shapes.
- No env fallback was added; this is the default fast path for the matching shape class.

Build and sync:

```text
(cd source/backend/opencl/execution/cl && python3 opencl_codegen.py .)
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/home/orangepi/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/orangepi5plus/
```

Artifact check:

```text
strings .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN_CL.so | rg "pic_gemm_b4_c8"
pic_gemm_b4_c8
__kernel void pic_gemm_b4_c8_int4_buf
__kernel void pic_gemm_b4_c8_int8_buf
```

Smoke 20/30 sweep:

```text
tag=opencl_pic_1024_picgemm_20_30_smoke_20260611_155014
profile=false, detail=false, warm_tune=true

normal,full,6.968234,1.000
full-reuse,full,0.864710,8.058
cacheblend,0.20,3.411061,2.043
cacheblend,0.30,4.026625,1.731
epic,0.20,3.170452,2.198
epic,0.30,4.282373,1.627
```

Graph attribution after the fix:

```text
tag=opencl_pic_1024_picgemm_graph_cb20_30_20260611_155256
profile=false, detail=false, graph_profile=true, warm_tune=true
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

request=5 cacheblend20 total=3831.490 ms
  Convolution=2249.682 ms
  PagedAttention=704.097 ms
  PicSparseAttention=292.069 ms
  PicScoreAttention=26.421 ms

request=6 cacheblend30 total=4651.346 ms
  Convolution=2988.953 ms
  PagedAttention=624.255 ms
  PicSparseAttention=461.266 ms
  PicScoreAttention=33.417 ms
```

Compared to the old graph profile, cacheblend20 `Convolution` dropped from `4964.176 ms` to `2249.682 ms`. The 20% budget no longer has the inverted latency trend; the remaining higher latency at larger budgets is now consistent with more compact rows and more sparse attention work.

Formal 10/20/30/40/50 sweep:

```text
tag=opencl_pic_1024_picgemm_formal_20260611_155453
profile=false, detail=false, warm_tune=true
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal,full,6.968234,1.000
full-reuse,full,0.817891,8.520
cacheblend,0.10,2.212856,3.149
cacheblend,0.20,3.290629,2.118
cacheblend,0.30,3.889923,1.791
cacheblend,0.40,5.281809,1.319
cacheblend,0.50,5.769052,1.208
epic,0.10,2.185259,3.189
epic,0.20,3.199813,2.178
epic,0.30,3.829921,1.819
epic,0.40,4.779629,1.458
epic,0.50,5.119875,1.361
```

Comparison against the previous P0b formal sweep:

```text
cacheblend,0.10: 3.617224 -> 2.212856 (-38.8%)
cacheblend,0.20: 6.015361 -> 3.290629 (-45.3%)
cacheblend,0.30: 3.737483 -> 3.889923 (+4.1%)
cacheblend,0.40: 4.069478 -> 5.281809 (+29.8%)
cacheblend,0.50: 5.262682 -> 5.769052 (+9.6%)
epic,0.10:       3.656841 -> 2.185259 (-40.2%)
epic,0.20:       6.142038 -> 3.199813 (-47.9%)
epic,0.30:       3.476549 -> 3.829921 (+10.2%)
epic,0.40:       4.005558 -> 4.779629 (+19.3%)
epic,0.50:       5.514270 -> 5.119875 (-7.2%)
```

Conclusion:

- The 20% budget cliff is fixed. It was a dense compact Linear / OpenCL low-memory 1x1 Conv tune-path issue, not a sparse attention issue.
- All cacheblend and epic ratios remain faster than normal full compute in the formal sweep.
- The `pic_gemm_b4_c8_*` fast path is kept as default because it removes the severe 20% regression and improves low-budget cacheblend/epic substantially.
- High-budget cacheblend 40/50 can still be improved; next work should profile whether the regression versus P0b is from the new compact dense tune key selection or from normal run variance, then consider separate compact-row LWS buckets for `globalY` around 400/500.

Important MLP follow-up:

- Treat MLP as a first-class optimization track, not as incidental `Convolution` noise. After score-layer compaction, every later layer runs:
  - `mlp/gate_proj/Linear` with compact rows x 2048 -> compact rows x 8192.
  - `mlp/up_proj/Linear` with compact rows x 2048 -> compact rows x 8192.
  - `mlp/down_proj/Linear` with compact rows x 8192 -> compact rows x 2048.
- The same MLP fast path must hold for all recompute ratios from 1% to 50%. For the 1024-token setup, even 1% should usually be above the tiny-GEMV cutoff because active rows include prelude/suffix plus selected PIC rows.
- Future sweeps should include at least 1/5/10/20/30/40/50 when studying budget scaling, or explicitly say which subset was run. For each sweep, spot-check graph profile on low/mid/high ratios and confirm MLP inputs are compact rows and use the `pic_gemm_b4_c8_*` tune namespace.
- If a low budget is unexpectedly slower than a higher budget, first compare MLP `gate_proj/up_proj/down_proj` latency and tune key behavior before changing attention semantics. The historical 20% anomaly was exactly this class of MLP/dense compact-row issue.

## 2026-06-11 Jetson CUDA Graph-Boundary Adaptation

Scope:

- Ported the current PIC score-layer graph-boundary semantics to CUDA `PagedAttentionExecution`.
- CUDA now registers `OpType_PicScoreAttention` and `OpType_PicSparseAttention`.
- `kvWriteLen` and `attnLen` are separated:
  - score layer uses full Q/K/V write with compact attention output and emits `active_indices`.
  - later layers use compact Q/K/V rows while writing/reading true logical slots through `active_indices`.
- Decode and non-PIC runtime still use normal `PagedAttention` behavior.
- Async persistent PIC cache loading and PagedCache hydrate semantics are unchanged.

Rejected CUDA dense experiment:

- Tried a naive packed INT4 CUDA `PicGEMM_FpAInt4B` for compact 1x1 MLP rows.
- It was wrong for production performance even though it avoided runtime dequant:
  - old graph profile with this path, cacheblend20: `Convolution=2983.717 ms`, `PicSparseAttention=230.608 ms`.
  - per-layer compact MLP was pathological: 216-row `down_proj` around `67 ms`, `gate/up` around `51 ms`.
  - end-to-end formal cacheblend20 regressed to `3.408640s` / `0.690x` normal, and cacheblend50 to `7.818670s` / `0.301x`.
- The final CUDA code does not keep this path. CUDA compact MLP currently uses the existing tensor-core CUTLASS + runtime dequant path; a future compact GEMM must beat this path across 1%-50% before becoming default.

A/B confirmation after removing naive CUDA PicGEMM:

```text
tag=pic_cuda_1024_profile_cb20_cutlass_ab_20260611_164608
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend20 latency=0.816942s speedup_vs_normal=2.881
graph request=3 cacheblend20 total=706.398 ms
  Convolution=262.514 ms
  PicSparseAttention=230.444 ms
  PagedAttention=87.252 ms
  PicScoreAttention=30.225 ms

score layer output: [1x216x2048] + active_indices [216]
later sparse layers: [1x216x32x64] Q/K/V, [1x1x216x1024] mask
compact MLP after fix: 216-row down_proj around 3.9 ms, not 67 ms
```

Formal Jetson CUDA 1024-token sweep:

```text
tag=pic_cuda_1024_boundary_cutlass_formal_20260611_164704
model=Llama-3.2-1B-Instruct@jetson-cuda
profile=false, graph_profile=false
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.359027,0.997
full-reuse,full,0.292211,8.051
cacheblend,0.10,0.513967,4.577
cacheblend,0.20,0.690936,3.405
cacheblend,0.30,0.863871,2.723
cacheblend,0.40,1.103854,2.131
cacheblend,0.50,1.297994,1.812
epic,0.10,0.473170,4.972
epic,0.20,0.670933,3.506
epic,0.30,0.839971,2.801
epic,0.40,1.051566,2.237
epic,0.50,1.253765,1.876
```

Low-budget CUDA smoke:

```text
tag=pic_cuda_1024_low_budget_cutlass_smoke_20260611_164825
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.01,0.354654,6.633
cacheblend,0.05,0.410731,5.728
epic,0.01,0.318615,7.384
epic,0.05,0.413581,5.688
```

High-budget CUDA attribution:

```text
tag=pic_cuda_1024_profile_cb50_cutlass_20260611_164857
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50 graph total=1334.525 ms
  PicSparseAttention=535.167 ms
  Convolution=459.797 ms
  PicScoreAttention=89.685 ms
  PagedAttention=87.304 ms
  UnaryOp=74.069 ms

score layer output rows=519
later sparse attention per layer ~=38.1 ms
compact MLP per layer ~=7.4-7.8 ms
```

Conclusion:

- CUDA graph-boundary semantics are active and stable on Jetson for 1024 tokens.
- `PIC full-compute` is aligned with normal full compute (`2.359027s` vs `2.352533s`).
- `full-reuse`, cacheblend and epic are faster than normal for 1/5/10/20/30/40/50 tested budgets.
- Current remaining CUDA high-budget bottleneck is split between later-layer `PicSparseAttention` and compact dense MLP. For the next CUDA optimization round, P0 is a real CUDA sparse flash attention kernel that fuses QK/softmax/QKV without repeating QK; P1 is a tensor-core compact-row INT4/FP16 GEMM that beats CUTLASS + runtime dequant across 1%-50%.

## 2026-06-11 Jetson CUDA P0/P1 Negative A/B

Baseline to compare:

```text
tag=pic_cuda_1024_boundary_cutlass_formal_20260611_164704
cacheblend,0.20,0.690936,3.405
cacheblend,0.50,1.297994,1.812
epic,0.20,0.670933,3.506
epic,0.50,1.253765,1.876

tag=pic_cuda_1024_low_budget_cutlass_smoke_20260611_164825
cacheblend,0.01,0.354654,6.633
cacheblend,0.05,0.410731,5.728
epic,0.01,0.318615,7.384
epic,0.05,0.413581,5.688
```

Rejected P0a: existing fused row-compressed attention

- Tested by launching the existing `v2_row_compressed_mask` path with `MNN_PAGED_ATTENTION_BENCH_FORCE_V2_KERNEL=1`.
- This is mathematically closer to flash attention because it streams softmax and V accumulation, but it is one query row/head per block and loses the Q/K tiling efficiency of the existing QK kernel.

```text
tag=pic_cuda_1024_v2_sparse_ab_cb20_50_20260611_191539
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.20,1.548174,1.520
cacheblend,0.50,2.303546,1.021

attn=216 v2 sparse rows: ~58.2 ms/layer
baseline attn=216 three-stage rows: ~15.6 ms/layer
```

Conclusion: do not default the existing row-compressed fused kernel for PIC sparse. It reduces temporary buffers but destroys per-layer throughput.

Rejected P0b: sparse single-piece qSplit

- Tried forcing sparse prefill `q_split=1` while keeping normal full-compute split unchanged.
- Intended to reduce per-layer launch count for cb40/cb50, similar to the successful OpenCL single-piece result.
- On Jetson CUDA it enlarged QK/softmax temporary buffers enough to hurt high-budget layers.

```text
tag=pic_cuda_1024_sparse_singlepiece_ab_cb20_50_20260611_191852
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.20,0.813060,2.893
cacheblend,0.50,1.860186,1.265

cb50 sparse q_split=1 layer samples:
  layer=1 attn=519 q_split=1 us=60143
  layer=2 attn=519 q_split=1 us=64089
  layer=7 attn=519 q_split=1 us=70262

baseline cb50 q_split=3 later layers: ~37.3 ms/layer
```

Conclusion: CUDA P0 must be a tile-based sparse flash kernel that preserves Q/K tiling and avoids repeated QK, not a single-row fused kernel or larger single-piece QK buffer.

Rejected P1a: extend V14_MB packed GEMV to batch<=32

- Tried using existing factored-dequant packed INT4 GEMV for low-budget compact rows, processed in groups of 8 rows.
- This avoided runtime dequant for 7-32 rows, but repeated weight reads and many small GEMV launches cost more than CUTLASS + runtime dequant.

```text
tag=pic_cuda_1024_gemv32_ab_1_5_20_20260611_192317
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

full-reuse,full,0.281823,8.348
cacheblend,0.01,0.543940,4.325
cacheblend,0.05,0.414722,5.673
cacheblend,0.20,0.684411,3.437
epic,0.01,0.521939,4.507
epic,0.05,0.404292,5.819
epic,0.20,0.671286,3.505
```

Compared to the low-budget CUTLASS baseline, cacheblend1 and epic1 regressed badly. The source patch was reverted.

Next CUDA work:

- P0 needs a new tile-based sparse flash kernel:
  - keep multiple Q rows per block or per CTA tile;
  - preserve Q/K tiling and coalesced K/V loads;
  - compute online softmax and V accumulation without writing global QK/softmax;
  - avoid the existing row kernel's one-row/head occupancy problem;
  - preserve score-layer full-Q/compact-output and later compact-Q semantics.
- P1 needs either real tensor-core weight-only INT4 GEMM or a memory-aware dequant cache. Packed GEMV or naive packed GEMM should stay rejected until it beats CUTLASS across 1/5/10/20/30/40/50.

## 2026-06-11 Jetson CUDA Tile Sparse Flash Accepted Path

Implemented `pagedSparseFlashTileKernel<T, Q_TILE=8, K_TILE=32>` in CUDA `PagedAttentionExecution`:

- one CTA handles 8 active query rows for one head;
- streams K/V in 32-token tiles;
- keeps online softmax state and V accumulation inside the kernel;
- avoids global QK and softmax tensors;
- currently only supports `head_dim=64`.

The production route is intentionally narrow:

```text
fixedPlanSparseQuery =
    sparseQuery &&
    pic_graph_active_plan_ready &&
    !cacheblend_score_active
```

This means epic / fixed active-plan rows use tile sparse flash, including the score layer full-Q/compact-output case and later compact-Q sparse layers. Cacheblend scattered rows continue using the existing QK/softmax/QKV path because the all-sparse tile flash A/B was flat-to-slightly-worse on cacheblend.

Formal sweep after narrowing to epic/fixed plan:

```text
tag=pic_cuda_1024_tileflash_epic_only_formal_20260611_193310
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.367386,0.994
full-reuse,full,0.293133,8.025
cacheblend,0.10,0.516453,4.555
cacheblend,0.20,0.695926,3.380
cacheblend,0.30,0.867265,2.713
cacheblend,0.40,1.106997,2.125
cacheblend,0.50,1.294405,1.817
epic,0.10,0.461220,5.101
epic,0.20,0.643831,3.654
epic,0.30,0.812402,2.896
epic,0.40,1.008296,2.333
epic,0.50,1.211449,1.942
```

Compared with the CUTLASS graph-boundary baseline, cacheblend is unchanged within noise while epic improves:

```text
epic10: -2.5%
epic20: -4.0%
epic30: -3.3%
epic40: -4.1%
epic50: -3.4%
```

Low-budget smoke after the accepted route:

```text
tag=pic_cuda_1024_tileflash_epic_only_low_20260611_193752
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.01,0.351891,6.685
cacheblend,0.05,0.413794,5.685
epic,0.01,0.317318,7.414
epic,0.05,0.406373,5.789
```

Compared with low-budget CUTLASS baseline, cacheblend1 is slightly better, cacheblend5 is noise/slightly worse, epic1/5 are better. No extra active-row threshold is needed for epic/fixed plan right now.

Profile attribution at 50%:

```text
tag=pic_cuda_1024_tileflash_epic_only_profile_50_20260611_193417

cacheblend50:
  PicSparseAttention=534.766 ms
  Convolution=459.124 ms
  PicScoreAttention=89.024 ms
  PagedAttention=87.673 ms
  later sparse attention path=prefill_attention_fast_qk_softmax_qkv
  per later sparse layer ~=37.3 ms

epic50:
  PicSparseAttention=493.569 ms
  Convolution=466.032 ms
  PicScoreAttention=36.264 ms
  PagedAttention=80.450 ms
  sparse attention path=sparse_flash_tile_attention
  per sparse flash layer ~=34.4 ms
```

Active-index analysis explains why cacheblend budget does not map linearly to compute:

```text
mode       ratio rows max_logical avg_causal_k work_frac tail>=768 gaps>1
cacheblend 0.10 101  1006        771.1        0.753     45        83
cacheblend 0.20 202  1008        732.3        0.715     74       119
cacheblend 0.30 303  1009        737.1        0.720    126       117
cacheblend 0.40 404  1009        740.1        0.723    179        59
cacheblend 0.50 505  1009        673.6        0.658    210        59
epic       0.10 101   107         58.0        0.057      0         0
epic       0.20 202   208        108.5        0.106      0         0
epic       0.30 303   309        159.0        0.155      0         0
epic       0.40 404   410        209.5        0.205      0         0
epic       0.50 505   511        260.0        0.254      0         0
```

Cacheblend active rows are sorted but scattered and reach the tail even at 10%. The selected-row count is small, but the causal K range per selected row remains large. Epic selects a contiguous prefix, so both selected rows and causal K range shrink together. This is the main reason cacheblend speedup is weaker and less proportional to budget.

Further optimization potential:

- P0a cacheblend-specific sparse flash: keep the same active set, but bucket rows internally by logical position and scatter outputs back to the original compact row order. The kernel should skip K tiles above each bucket's max causal position and avoid loading V for invalid K. This does not change cacheblend semantics, but the upside is bounded because cacheblend average causal K is already 65%-75% of full.
- P0b tune the accepted tile flash for epic/fixed rows: A/B `Q_TILE=4/8/16`, `K_TILE=32/64`, fewer `__syncthreads()`, less `scoreShared` round trip, vectorized half/half2 K/V loads, and adaptive q-piece/bucket heuristics. Promote only a shape heuristic that wins 1/5/10/20/30/40/50; do not add production env fallbacks. The later `outAcc0/outAcc1` register-accumulator experiment was rejected on sm72.
- P1 cacheblend score layer: cacheblend50 `PicScoreAttention=89 ms`, but the attention sub-kernel is only about `37 ms`; scoring/top-k/metadata costs the other ~50 ms. Split and optimize score/top-k before changing attention semantics.
- P2 compact dense: at 50%, `Convolution ~=459-466 ms`, the same order as sparse attention. The next large end-to-end win likely needs a real tensor-core compact-row weight-only GEMM or dequant-cache strategy, not naive packed GEMV.

## 2026-06-11 Jetson CUDA Causal K Limit for Tile Flash

Follow-up optimization:

- `pagedSparseFlashTileKernel` now computes the max logical position inside each `Q_TILE`.
- The K/V streaming loop stops at `min(kvLen, max_q_logical + 1)`.
- This skips whole invalid causal K/V tiles for epic/fixed active plans.
- Cacheblend scattered rows still do not route into this kernel, so cacheblend remains protected from the previous all-sparse tile flash regression.

Build/sync:

```text
target=pic_server
CUDA architectures: 7.2
KleidiAI: OFF
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Formal sweep:

```text
tag=pic_cuda_1024_tileflash_causal_limit_formal_20260611_194416
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.378959,0.989
full-reuse,full,0.290140,8.108
cacheblend,0.10,0.512455,4.591
cacheblend,0.20,0.689947,3.410
cacheblend,0.30,0.864029,2.723
cacheblend,0.40,1.103161,2.133
cacheblend,0.50,1.298412,1.812
epic,0.10,0.382938,6.143
epic,0.20,0.494079,4.761
epic,0.30,0.596555,3.944
epic,0.40,0.734550,3.203
epic,0.50,0.888805,2.647
```

Compared with the previous epic-only tile flash run:

```text
epic10: 0.461220 -> 0.382938 (-17.0%)
epic20: 0.643831 -> 0.494079 (-23.3%)
epic30: 0.812402 -> 0.596555 (-26.6%)
epic40: 1.008296 -> 0.734550 (-27.1%)
epic50: 1.211449 -> 0.888805 (-26.6%)
```

Compared with the original CUTLASS graph-boundary baseline:

```text
epic10: 0.473170 -> 0.382938 (-19.1%)
epic20: 0.670933 -> 0.494079 (-26.4%)
epic30: 0.839971 -> 0.596555 (-29.0%)
epic40: 1.051566 -> 0.734550 (-30.2%)
epic50: 1.253765 -> 0.888805 (-29.1%)
```

Low-budget smoke:

```text
tag=pic_cuda_1024_tileflash_causal_limit_low_20260611_194507
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.01,0.359771,6.539
cacheblend,0.05,0.418441,5.622
epic,0.01,0.307299,7.656
epic,0.05,0.366858,6.413
```

Profile attribution at 50%:

```text
tag=pic_cuda_1024_tileflash_causal_limit_profile_50_20260611_194540
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50:
  PicSparseAttention=533.888 ms
  Convolution=459.820 ms
  PicScoreAttention=89.250 ms
  PagedAttention=87.287 ms
  later sparse path=prefill_attention_fast_qk_softmax_qkv

epic50:
  Convolution=465.632 ms
  PicSparseAttention=194.614 ms
  PagedAttention=80.075 ms
  UnaryOp=74.756 ms
  PicScoreAttention=15.336 ms
  sparse path=sparse_flash_tile_attention
  per sparse flash layer ~=13.0 ms
```

Conclusion:

- The main wasted work in the accepted tile sparse flash was full-length K/V streaming even when the active rows were a contiguous prefix. Limiting each Q tile by `max_q_logical` fixes that without changing sparse semantics.
- Epic/fixed-plan attention is no longer the primary bottleneck at 50%; dense `Convolution`/MLP is now larger than `PicSparseAttention`.
- Cacheblend remains attention-bound because it still uses scattered rows that reach the tail; the next cacheblend-specific path must exploit logical-position buckets or change algorithm semantics under a new mode name.
- Keep the causal K limit in the default CUDA tile flash route. Do not add an env fallback for the slower full-K tile loop.

## 2026-06-11 Jetson CUDA Cacheblend Hybrid Sparse Flash

Follow-up P0 A/B:

- The first "all sparse" route only removed `!cacheblend_score_active` but still required `pic_graph_active_plan_ready`; profile showed cacheblend remained on `prefill_attention_fast_qk_softmax_qkv`, so that was not a real cacheblend flash test.
- The real all-sparse route uses tile flash whenever `sparseQuery` is active. This finally routes cacheblend score/later sparse rows into `sparse_flash_tile_attention`.
- True all-sparse was good at high cacheblend budgets but too noisy around 20%, so the accepted route is a built-in shape heuristic:

```text
fixed/epic sparse rows: always tile flash, q_tile=8, k_tile=32
cacheblend sparse rows: tile flash only when attnLen > 256, q_tile=8, k_tile=32
cacheblend attnLen <= 256: keep existing QK/softmax/QKV path
```

Smoke that proved cacheblend can benefit when it really enters tile flash:

```text
tag=pic_cuda_1024_tileflash_true_allsparse_smoke_20_50_20260611_195425
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend20: 0.703900, 3.342x
cacheblend50: 1.084251, 2.170x
epic20:       0.496554, 4.738x
epic50:       0.906377, 2.596x
```

True all-sparse formal:

```text
tag=pic_cuda_1024_tileflash_true_allsparse_formal_20260611_195522
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.10,0.508005,4.631
cacheblend,0.20,0.671017,3.506
cacheblend,0.30,0.824639,2.853
cacheblend,0.40,1.028797,2.287
cacheblend,0.50,1.160317,2.027
epic,0.10,0.381198,6.171
epic,0.20,0.668295,3.520  # treated as outlier; repeat below returned to ~0.49s
epic,0.30,0.600304,3.919
epic,0.40,0.737021,3.192
epic,0.50,0.891496,2.639
```

Repeat for the unstable 20/30 area:

```text
tag=pic_cuda_1024_tileflash_true_allsparse_repeat_20_30_20260611_195612
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend20: 0.706689, 3.329x
cacheblend30: 0.791195, 2.973x
epic20:       0.491828, 4.783x
epic30:       0.613817, 3.833x
```

Accepted hybrid formal:

```text
tag=pic_cuda_1024_tileflash_hybrid_formal_20260611_195828
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.411054,0.976
full-reuse,full,0.296088,7.945
cacheblend,0.10,0.513422,4.582
cacheblend,0.20,0.692993,3.395
cacheblend,0.30,0.833976,2.821
cacheblend,0.40,1.048359,2.244
cacheblend,0.50,1.196245,1.967
epic,0.10,0.382526,6.150
epic,0.20,0.491652,4.785
epic,0.30,0.597417,3.938
epic,0.40,0.733922,3.205
epic,0.50,0.890640,2.641
```

Compared with the causal-limit epic-only route:

```text
cacheblend30: 0.864029 -> 0.833976 (-3.5%)
cacheblend40: 1.103161 -> 1.048359 (-5.0%)
cacheblend50: 1.298412 -> 1.196245 (-7.9%)
epic10/20/30/40/50: stays on the accepted causal-limit tile flash path
```

Hybrid low-budget smoke:

```text
tag=pic_cuda_1024_tileflash_hybrid_low_20260611_195917
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.01,0.364208,6.459
cacheblend,0.05,0.419508,5.608
epic,0.01,0.309107,7.611
epic,0.05,0.372496,6.316
```

The low-budget cacheblend path does not use tile flash under the accepted heuristic; small variation here is run noise, not a q_tile change.

Hybrid profile at 50%:

```text
tag=pic_cuda_1024_tileflash_hybrid_profile_50_20260611_195951
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50:
  PicSparseAttention=464.707 ms
  Convolution=458.810 ms
  PagedAttention=87.546 ms
  PicScoreAttention=84.367 ms
  sparse path=sparse_flash_tile_attention
  q_tile=8 k_tile=32

epic50:
  Convolution=463.017 ms
  PicSparseAttention=193.909 ms
  PagedAttention=79.806 ms
  PicScoreAttention=14.531 ms
  sparse path=sparse_flash_tile_attention
  q_tile=8 k_tile=32
```

Rejected q4 cacheblend-high variant:

```text
tag=pic_cuda_1024_tileflash_hybrid_q4_cb_smoke_30_50_20260611_200246
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend30: 1.289018, 1.825x
cacheblend50: 1.887966, 1.246x
epic30:       0.604598, 3.891x
epic50:       0.905547, 2.598x
```

Conclusion:

- Do not use `q_tile=4` for cacheblend high budgets. The smaller tile reduces some invalid K range but creates too many CTAs and loses more to scheduling/launch/memory overhead.
- Keep `q_tile=8,k_tile=32` as the accepted CUDA sparse flash tile shape for now.
- The accepted cacheblend heuristic is `attnLen > 256`. It protects 20% and low budgets while improving 30/40/50.
- Final artifact was rebuilt and re-synced after reverting q4. Final smoke `tag=pic_cuda_1024_tileflash_hybrid_final_smoke_50_20260611_200614` reported cacheblend50 `1.264013s`, epic50 `0.910181s`, confirming q4 was not left on the Jetson artifact.
- After this P0, cacheblend50 attention and compact dense are roughly tied (`PicSparseAttention ~=465 ms`, `Convolution ~=459 ms`), so the next high-value line is P1 compact tensor-core GEMM / dequant-cache work.

## 2026-06-11 Jetson CUDA Further Optimization Potential After Hybrid

Current accepted default remains the hybrid CUDA route:

```text
fixed/epic sparse rows: q8/k32 sparse flash tile with per-Q-tile causal K limit
cacheblend sparse rows: q8/k32 sparse flash tile only when attnLen > 256
cacheblend attnLen <= 256: existing QK/softmax/QKV path
```

The next potential is no longer a single "make sparse attention faster" problem. The profile at 50% splits into three different bottleneck classes:

```text
tag=pic_cuda_1024_tileflash_hybrid_profile_50_20260611_195951

cacheblend50 total_ms=1258.747
  PicSparseAttention=464.707 ms
  Convolution=458.810 ms
  PagedAttention=87.546 ms
  PicScoreAttention=84.367 ms
  UnaryOp=74.213 ms
  Raster=44.629 ms
  BinaryOp=26.187 ms

epic50 total_ms=911.664
  Convolution=463.017 ms
  PicSparseAttention=193.909 ms
  PagedAttention=79.806 ms
  UnaryOp=73.962 ms
  Raster=43.236 ms
  BinaryOp=26.038 ms
  PicScoreAttention=14.531 ms
```

Detailed attention facts:

```text
cacheblend50:
  score layer sparse flash layer=1 full_q=1 us=32670
  later sparse flash layers ~=32-33 ms/layer
  cacheblend_score metadata line precedes attention, so PicScoreAttention total has about 50 ms not explained by flash attention itself

epic50:
  score layer sparse flash layer=1 full_q=1 us=13087
  later sparse flash layers ~=13 ms/layer
```

Why cacheblend still costs much more than epic with the same active row count:

- Both have `attn=519` at 50%.
- Epic selected rows are a contiguous prefix; the causal K limit per Q tile is about 25% of full at 50%.
- Cacheblend selected rows are scattered and reach logical position 1009; average causal K is about 65%-75% of full.
- The accepted tile flash already skips whole K/V tiles above each tile's max logical position. Cacheblend still streams much more real K/V, not just mask waste.

Further attention potential:

- P0a: cacheblend-specific logical-range tiling. Keep the exact same active set and output compact order, but internally process active rows in narrower logical buckets and scatter outputs back. This can reduce invalid K/V tile loads inside a Q tile. The q4 A/B proved that simply shrinking `Q_TILE` loses too much scheduling efficiency; a useful version must keep enough rows per CTA while narrowing each tile's causal range.
- P0b: reduce tile-flash overhead without changing work: remove or reduce `scoreShared` round trips, reduce `__syncthreads()`, vectorize K/V loads, and test `K_TILE=64` only if occupancy and shared-memory pressure stay acceptable. This is likely a smaller win than P0a for cacheblend and smaller than dense work for epic.
- P0c is not to window cacheblend selected tokens. Any change that prevents scattered tail rows from being selected changes algorithm semantics and must be reported as a new mode such as `cacheblend-windowed`, not as ordinary cacheblend.

Score/top-k potential:

- `runCacheBlendScoringCUDA` still reads the cached value segment into a host `std::vector<int8_t>`, copies it to a CUDA workspace, scores one segment, runs a single-block top-k kernel, copies only top-k indices back to host, and calls `setCacheBlendScoringResult`.
- For 1024-token cacheblend50, `PicScoreAttention=84 ms` while sparse flash attention inside the same layer is about `33 ms`; the remaining roughly `50 ms` is the score/top-k/source-value path plus metadata.
- First action should be instrumentation that splits this path into `readExternalValueSegment`, H2D copy, value-score kernel, top-k kernel, D2H index copy, and metadata. Do not guess which part dominates.
- If source values already exist in the current request's mapped PagedCache source slots after async hydrate scheduling, score directly from those source slots and remove the per-request host read/H2D copy. This keeps the persistent PIC cache source/current request PagedCache boundary intact; it must not create scratch `.k/.v`.
- The current `cacheBlendTopKKernel<<<1,256>>>` is deterministic but O(topK * tokenCount) inside one CTA. For 1010 tokens this is not huge, but if instrumentation shows it is material, replace it with a two-stage block top-k or threshold/select kernel and keep deterministic tie behavior.

Compact dense/MLP potential:

- After sparse flash, dense MLP is a first-class bottleneck:
  - cacheblend50: `Convolution ~=459 ms`, tied with `PicSparseAttention`.
  - epic50: `Convolution ~=463 ms`, more than 2x `PicSparseAttention`.
- Score-layer graph boundary is working: after layer 1, MLP and QKV/o-proj run on compact rows such as `519x2048` or `519x8192`; layer 0 remains full 1024 rows by semantics.
- Per compact layer at 50%, `mlp/gate_proj`, `mlp/up_proj`, and `mlp/down_proj` are each about `7.1-7.8 ms`. The full layer-0 MLP projections are about `12.6 ms` each.
- The CUDA low-memory weight-only 1x1 Conv path currently dequantizes INT4 weights into a DYNAMIC FP16 buffer on every execute, then calls CUTLASS GEMM. This is memory-safe for large models, but for 1B PIC compact rows it creates a ratio-independent fixed cost for every Linear.
- Before adding a dequant cache, add profile-only timing around `DequantizeInt4Weight` and `runCutlassGemmFunc()` for compact MLP shapes. A dequant cache is only worth defaulting if dequant is a meaningful fraction of the 7-8 ms compact Linear time and a 1/5/10/20/30/40/50 sweep proves no OOM/regression.
- Memory math for caching all Llama-3.2-1B dequant MLP weights is large but not absurd: gate/up/down are about 32 MB each per layer, about 96 MB/layer and about 1.5 GB for 16 layers, before attention projection weights. This may be acceptable on an 8 GB Jetson for this model but is not a general production rule for larger models.

Unary/Binary/Raster potential:

- `UnaryOp ~=74 ms` and `BinaryOp ~=26 ms` at 50% are mostly MLP activation and multiply. At compact 519 rows, each `act_fn/Mul_output_0` is about `4.16 ms`, and each MLP elementwise multiply is about `0.61 ms`.
- A fused SiLU(gate) * up kernel could remove a large part of this overhead and reduce one intermediate read/write. It is less invasive than a full fused MLP kernel and should be considered alongside compact GEMM work.
- A full fused MLP kernel that combines gate/up projection, activation, multiply, and down projection could save more memory traffic but is much higher risk because it must avoid recomputing gate/up work for each down-output tile.

PagedAttention/full-prefix potential:

- `PagedAttention` layer 0 remains about `80-88 ms` and is semantically full compute. It is not the first priority now.
- Full-reuse is already about `0.296 s`; hydrate/restore lines are typically under 1 ms/layer with async reads, so the async persistent PIC cache loading path is not the current bottleneck.

Recommended next experiment order:

1. Add CUDA profile-only timers for compact weight-only Conv: dequant vs CUTLASS GEMM, at least for `mlp/gate_proj`, `mlp/up_proj`, and `mlp/down_proj` on active rows 1/5/10/20/30/40/50.
2. Add CUDA profile-only timers inside `runCacheBlendScoringCUDA`: host read, H2D copy, score kernel, top-k, D2H, metadata.
3. If dequant is large, implement a memory-aware default dequant cache for the 1B PIC artifact or a real fused dequant+tensor-core GEMM; reject it unless all ratios stay faster than the current hybrid.
4. If cacheblend score/source copy is large, score from current request PagedCache source slots and keep only final top-k indices crossing to host.
5. Only after those measurements, attempt cacheblend logical-range sparse flash. The upside is bounded by cacheblend's real causal K distribution; do not expect epic-like latency unless the algorithm changes selection distribution.

Expected ceiling from current profile:

- Halving only cacheblend sparse attention would save about `230 ms` at cb50 and move `1.20 s` toward `0.96-1.03 s`.
- Halving only compact dense would save about `230 ms` for both cb50 and epic50; epic50 would move from about `0.89 s` toward `0.66 s`.
- Removing MLP Unary/Binary overhead could save up to about `100 ms` at 50%, but real savings depend on fusion and memory traffic.
- Therefore the most realistic next end-to-end gain is not one giant kernel change; it is dense dequant/GEMM + MLP elementwise fusion for all sparse modes, plus cacheblend-specific score/source-value cleanup.

## 2026-06-11 Jetson CUDA P0 Tail K/V Cut and P1 Dequant Profile

Implemented two low-risk CUDA changes:

- P0: in `pagedSparseFlashTileKernel`, the final K tile now only loads K/V when `logical < causalKLimit`, not merely `logical < kvLen`. This keeps the same causal softmax range and avoids reading tail lanes above `max_q_logical + 1` in the last K tile.
- P1 measurement: `ConvFpAIntBExecution` now prints profile-only split timing for batched 1x1 weight-only Conv when `MNN_PAGED_ATTENTION_PROFILE=1` or `MNN_PIC_GRAPH_PROFILE=1`: `dequant_us`, `convert_us`, `cutlass_us`, and `total_us`. It does not change the production GEMM path or add an env fallback.

Build:

```text
target=pic_server
CUDA support: ON
CUDA architectures: 7.2
Enabling CUDA support (... archs: sm_72)
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Profile run:

```text
tag=pic_cuda_1024_p0_tail_p1_dequant_profile_50_20260611_202624
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000  # reused baseline
full-compute,full,2.566778,0.917         # profile sync overhead
full-reuse,full,0.433422,5.428           # profile sync overhead
cacheblend,0.50,1.375528,1.710           # profile sync overhead
epic,0.50,1.014570,2.319                 # profile sync overhead
```

Graph profile at 50%:

```text
cacheblend50 total_ms=1266.352
  Convolution=475.315 ms
  PicSparseAttention=454.714 ms
  PagedAttention=87.991 ms
  PicScoreAttention=84.041 ms
  UnaryOp=74.372 ms

epic50 total_ms=924.531
  Convolution=479.164 ms
  PicSparseAttention=189.692 ms
  PagedAttention=80.020 ms
  UnaryOp=73.800 ms
  PicScoreAttention=14.240 ms
```

P0 effect:

- Previous hybrid profile had cacheblend50 `PicSparseAttention=464.707 ms`; this run reports `454.714 ms`, about `10 ms` lower under profile.
- Raw cacheblend50 sparse flash layer times moved from roughly `32.0-32.6 ms/layer` to roughly `31.3-31.8 ms/layer`.
- Previous epic50 sparse flash layers were roughly `13.0 ms/layer`; this run is roughly `12.7 ms/layer` for normal layers. Treat this as a small win, not a new major P0 line.
- Formal no-profile smoke still passed:

```text
tag=pic_cuda_1024_p0_tail_smoke_50_20260611_202809
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

full-compute,full,2.354317,0.999
full-reuse,full,0.291271,8.077
cacheblend,0.50,1.251329,1.880
epic,0.50,0.896326,2.625
```

20% smoke:

```text
tag=pic_cuda_1024_p0_tail_smoke_20_20260611_202849
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

full-compute,full,2.351203,1.001
full-reuse,full,0.289471,8.127
cacheblend,0.20,0.720039,3.267
epic,0.20,0.499301,4.712
```

The cb20 path does not use tile flash under the accepted `attnLen > 256` cacheblend heuristic, so its single-run latency should be treated as smoke/noise rather than evidence for or against the P0 tail cut.

P1 dequant/GEMM split:

```text
profile aggregation from tag=pic_cuda_1024_p0_tail_p1_dequant_profile_50_20260611_202624

cacheblend50 Conv profile:
  lines=121
  dequant_ms=60.869
  cutlass_ms=392.414
  total_ms=453.380
  dequant_share=13.4%

epic50 Conv profile:
  lines=103
  dequant_ms=50.893
  cutlass_ms=380.283
  total_ms=431.266
  dequant_share=11.8%

compact M=519 representative shapes:
  [519,2048,8192] gate/up:
    avg_dequant ~=0.87 ms
    avg_cutlass ~=6.6-6.7 ms
    avg_total   ~=7.5-7.6 ms
  [519,8192,2048] down:
    avg_dequant ~=0.85-0.86 ms
    avg_cutlass ~=6.6 ms
    avg_total   ~=7.5 ms
  [519,2048,2048] q/o:
    avg_dequant ~=0.28-0.30 ms
    avg_cutlass ~=1.7 ms
    avg_total   ~=2.0 ms
  [519,2048,512] k/v:
    avg_dequant ~=0.14-0.15 ms
    avg_cutlass ~=0.51 ms
    avg_total   ~=0.65 ms
```

Conclusion:

- P0 tail K/V cut is safe and slightly improves sparse flash, but it is not enough to change the bottleneck class.
- P1 dequant is real but smaller than expected: full dequant caching would save roughly `50-60 ms` at cb50/epic50 after warm-up, not `100-140 ms` by itself. The larger dense cost is CUTLASS GEMM work.
- Do not blindly default a full static dequant cache for all 1B weights yet. It may need around GB-scale FP16 weight memory across MLP and attention projections, and larger models would be unsafe.
- The better P1 directions are now:
  - a memory-capped dequant cache validated across 1/5/10/20/30/40/50 and no OOM;
  - or a real fused dequant + tensor-core GEMM / weight-only tensor-core GEMM that reduces both dequant traffic and compact GEMM time;
  - plus MLP activation/multiply fusion, because `UnaryOp+BinaryOp` remains about `100 ms` at 50%.

## 2026-06-11 Jetson CUDA Further Potential Recheck

This pass rechecked the remaining CUDA upside after the accepted hybrid sparse flash and the P1 dequant split. The main conclusion is that there is still optimization potential, but it is now distributed across three smaller buckets instead of one obvious bottleneck.

Current 50% profile shape:

```text
cacheblend50:
  PicSparseAttention ~=455 ms
  Convolution        ~=475 ms
  PicScoreAttention  ~=84 ms
  UnaryOp+BinaryOp   ~=100 ms

epic50:
  Convolution        ~=479 ms
  PicSparseAttention ~=190 ms
  UnaryOp+BinaryOp   ~=100 ms
  PicScoreAttention  ~=14 ms
```

Attention recheck:

- Cacheblend active logical indices are already sorted before graph-boundary sparse rows are built (`buildCacheBlendActiveLogicalIndices` sorts selected local indices and appends active rows in logical order). Therefore a simple "sort/bucket then scatter" rewrite is less valuable than initially assumed; fixed q8 tiles already see mostly monotonic logical rows.
- The remaining cacheblend attention waste comes from q8 tiles that span large logical gaps. A useful next attention experiment is adaptive piece building: keep q8 for dense/contiguous active ranges, split only the tiles whose logical span is too large, and keep output order unchanged. Do not globally switch to q4; the q4 A/B already regressed cb30/cb50.
- `K_TILE=64` is not a drop-in change for the current kernel because the tile code relies on warp-level reductions over `threadIdx.x == 0..31`. A 64-wide K tile needs a different two-warp reduction design before it can be tested.
- WMMA/tensor-core sparse flash is theoretically interesting for the 8x32x64 QK and 8x32x64 PV tiles, but it would require a different shared-memory score/softmax layout and likely half probability staging. Treat this as P2 research, not the next production default.

Score/top-k recheck:

- CUDA cacheblend scoring still reads the persistent value segment into a host vector, copies it to a CUDA workspace, runs value-delta scoring, then runs `cacheBlendTopKKernel`.
- OpenCL already uses PagedCache source slots for scoring. CUDA can copy that design, but it first needs CUDA `onResize`/`ensureCache` to reserve source slots beyond the real logical slots, like OpenCL's `_picCacheSourceSlotBase/_picCacheSourceSlotCount`.
- The current top-k kernel is deterministic but has an expensive inner `for prev < k` scan. At 1024-token cb50, `topK ~=505`, so the algorithm does much more work than the token count suggests. First add split timers; if top-k is visible, replace the used-check scan with a shared used flag or two-stage select while preserving tie order.
- This line is likely a tens-of-ms cleanup for cacheblend, not the main epic bottleneck.

Dense/MLP recheck:

- Dequant is real but only about 11%-14% of profiled Conv time at 50%. Full dequant caching alone would likely save only `50-60 ms` at cb50/epic50 after warm-up.
- Most dense time is CUTLASS GEMM on compact rows. On sm72, a true INT4 tensor-core GEMM is not the safest assumption; the realistic CUDA path is either memory-capped FP16 dequant cache, fused dequant+FP16 GEMM, or export/runtime fusion that improves the existing CUTLASS shapes.
- The most promising graph-level dense fusion is `gate_proj` + `up_proj` because both read the same compact hidden input and produce the two operands for SwiGLU. A combined output-channel projection could improve launch/input-read efficiency, then a fused `SiLU(gate) * up` kernel can remove much of `UnaryOp+BinaryOp`.
- This dense line benefits cacheblend and epic at every ratio, while cacheblend-specific attention work mostly helps higher budgets.

Recommended next order:

1. Add split timers for CUDA cacheblend scoring: host read, H2D copy, value-score kernel, top-k, D2H indices, metadata.
2. Prototype fused `SiLU(gate) * up` or graph-level gate/up handling for compact rows, because `UnaryOp+BinaryOp ~=100 ms` is now comparable to the remaining attention micro-optimization ceiling.
3. If touching attention, implement adaptive q-piece construction for cacheblend scattered rows rather than global q4 or K_TILE=64.
4. Only after those, evaluate CUDA PagedCache source-slot scoring to remove the host value workspace path.

Practical ceiling:

- Cacheblend50 can probably still save `100-180 ms` without changing the algorithm: around `50-80 ms` from adaptive attention pieces, `20-50 ms` from score/top-k/source cleanup, and `50-100 ms` from MLP elementwise/dense cleanup.
- Epic50's best remaining win is dense/MLP, not attention. A good dense/MLP change could move epic50 from about `0.90 s` toward `0.75-0.80 s`; attention-only work will not.

## 2026-06-11 Jetson CUDA Cacheblend Score Source Slots

Implemented CUDA cacheblend score cleanup:

- `cacheBlendTopKKernel` now uses a device `used` bitmap instead of scanning `selected[0..k)` for every candidate. Tie behavior is unchanged: higher score wins, equal score picks smaller local index.
- CUDA cacheblend scoring now reserves PagedCache source slots beyond the real logical slots, matching the OpenCL source-slot design.
- On mapped PagedCache, persistent PIC value segments are read directly into `mCache->mappedValue->host` at `sourceSlotStart`, and `cacheBlendValueScoreFromPagedSourceKernel` compares reference logical slots and source slots inside the same `mCache->value` device buffer.
- This removes the previous `std::vector<int8_t> cachedValue` + H2D value workspace from the Jetson default path. It is "zero-copy" in the PagedCache/device-buffer sense; disk read still lands in CPU mapped pinned memory, not GPUDirect Storage.

Build:

```text
target=pic_server
CUDA support: ON
CUDA architectures: 7.2
Enabling CUDA support (... archs: sm_72)
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Profile before source-slot scoring, after used-bitmap top-k:

```text
tag=pic_cuda_1024_topk_used_profile_50_20260611_124634
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend_score:
  us=7346
  alloc_us=206
  read_us=3894
  h2d_us=641
  score_kernel_us=548
  topk_init_us=43
  topk_us=1862
  d2h_us=119
  metadata_us=14

cacheblend50 graph:
  PicSparseAttention=453.563 ms
  Convolution=473.952 ms
  PicScoreAttention=40.906 ms
```

Profile after source-slot scoring:

```text
tag=pic_cuda_1024_source_slot_profile_50_20260611_125345
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend_score:
  source_slot_segments=1
  source_slot_tokens=1010
  us=3477
  alloc_us=163
  read_us=501
  h2d_us=0
  score_kernel_us=778
  topk_init_us=42
  topk_us=1829
  d2h_us=119
  metadata_us=27

cacheblend50 graph:
  PicSparseAttention=455.965 ms
  Convolution=471.301 ms
  PicScoreAttention=37.116 ms
```

No-profile smoke:

```text
tag=pic_cuda_1024_source_slot_smoke_20_50_20260611_125449
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend20: 0.707247s, 3.326x vs normal
cacheblend50: 1.047366s, 2.246x vs normal

tag=pic_cuda_1024_source_slot_repeat_50_20260611_125538
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50 repeat: 1.200309s, 1.960x vs normal
```

Conclusion:

- Source-slot scoring is semantically cleaner and removes the extra value H2D copy from CUDA cacheblend score. Keep it as the default Jetson path.
- The measured score subpath improvement is about `3.9 ms` in profile (`7.346 ms -> 3.477 ms`). This is useful but not the main bottleneck.
- `topk_us` is still about `1.8 ms` at `topK=505`; further top-k work is possible but only a small cleanup unless larger context lengths make `topK * tokenCount` worse.
- End-to-end cacheblend50 remains dominated by later `PicSparseAttention` and compact dense `Convolution`; source-slot scoring should not change the next priority order.

## 2026-06-11 Jetson CUDA Static Dequant Cache For Compact Dense

Implemented a memory-capped static FP16 dequant cache for CUDA low-memory weight-only INT4 1x1 Conv:

- `ConvFpAIntBExecution::Resource` now can pre-dequant its packed INT4 weight once into a STATIC FP16 tensor.
- Default cap is `min(2 GiB, totalGlobalMem / 16)`, with a `256 MiB` floor. This keeps the fast path default on Jetson 1B while avoiding unbounded FP16 expansion for larger models or smaller devices.
- `onResize` uses the Resource static dequant filter when available; otherwise it keeps the existing DYNAMIC runtime-dequant path for tensors beyond the cap or unsupported precision.
- Profile lines now include `static_dequant`, `static_cache_bytes`, and `static_cache_total`.

Build:

```text
target=pic_server
CUDA support: ON
CUDA architectures: 7.2
Enabling CUDA support (... archs: sm_72)
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Profile:

```text
tag=pic_cuda_1024_static_dequant_2g_profile_50_20260611_211227
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

static dequant profile:
  runtime_dequant=1 lines: 0
  static_dequant=1 lines: 555
  static_cache_total=1946157056 bytes

cacheblend50 graph:
  Convolution        = 409.255 ms
  PicSparseAttention = 401.494 ms
  PicScoreAttention  = 33.362 ms
  UnaryOp            = 74.109 ms
  BinaryOp           = 25.779 ms
```

Compared with `pic_cuda_1024_source_slot_profile_50_20260611_125345`:

```text
Convolution:        471.301 ms -> 409.255 ms  (-62.046 ms)
PicSparseAttention: 455.965 ms -> 401.494 ms  (also lower in this profiled run; treat as run-to-run/profile interaction, not solely dequant)
PicScoreAttention:   37.116 ms ->  33.362 ms
cacheblend50 e2e:  1.329647 s -> 1.204583 s in profile run
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_static_dequant_2g_sweep_1_50_20260611_211338
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.312875,1.017
full-reuse,full,0.636679,3.695

cacheblend,0.01,0.321178,7.325
cacheblend,0.05,0.373273,6.302
cacheblend,0.10,0.432630,5.438
cacheblend,0.20,0.642387,3.662
cacheblend,0.30,0.762592,3.085
cacheblend,0.40,0.906735,2.595
cacheblend,0.50,1.045583,2.250

epic,0.01,0.267146,8.806
epic,0.05,0.307547,7.649
epic,0.10,0.336170,6.998
epic,0.20,0.447952,5.252
epic,0.30,0.548095,4.292
epic,0.40,0.686786,3.425
epic,0.50,0.833466,2.823
```

Conclusion:

- Keep the static dequant cache as the default CUDA compact dense fast path. It removes runtime dequant from all profiled 1B Linear ops under the 2 GiB / 1/16-memory cap.
- This is a P1 win, not the final dense solution: cb50 `Convolution` is still about `409 ms`, so most remaining dense cost is CUTLASS GEMM itself, not dequant.
- After this change, cacheblend50 is roughly split between `PicSparseAttention` and `Convolution`; epic50 remains more dense-bound than attention-bound.
- Next dense work should target graph/export fusion for gate/up and SwiGLU or a real fused dequant+GEMM/tensor-core path, not another naive packed GEMM.

## 2026-06-11 Jetson CUDA Sparse Tile Full-Causal Mask Skip

Implemented a small P0 cleanup for CUDA q8/k32 sparse tile flash:

- `PagedKVMeta` now records whether the model uses a float full-causal attention mask (`attention_mask=float`, `attention_type=full`).
- `Llm` sets this once from `llm_config.json`.
- `CUDAPagedAttention` passes `nullptr` / `mask_elements=0` to `pagedSparseFlashTileKernel` only for that full-causal sparse tile path. The tile kernel already enforces true logical causality through `active_indices`, `logical <= qLogical`, and per-Q-tile `causalKLimit`.
- Sliding/mix attention must keep the additive mask because it encodes a real window, not just causal order.

Build:

```text
target=pic_server
CUDA support: ON
CUDA architectures: 7.2
Enabling CUDA support (... archs: sm_72)
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Profile:

```text
tag=pic_cuda_1024_sparse_tile_skip_mask_profile_50_20260611_132617
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend_score:
  source_slot_segments=1
  source_slot_tokens=1010
  us=3392
  h2d_us=0
  score_kernel_us=750
  topk_us=1846

sparse tile:
  layer=1  query=1024 attn=519 mask_elements=0 input_mask_elements=1048576 causal_mask_skipped=1 us=26893
  layer=2+ query=519  attn=519 mask_elements=0 input_mask_elements=531456  causal_mask_skipped=1 us~=26277-26912

cacheblend50 graph:
  Convolution        = 410.185 ms
  PicSparseAttention = 383.119 ms
  PagedAttention     = 87.673 ms
  UnaryOp            = 73.701 ms
  PicScoreAttention  = 32.100 ms
  BinaryOp           = 25.966 ms
```

Compared with `pic_cuda_1024_static_dequant_2g_profile_50_20260611_211227`:

```text
PicSparseAttention: 401.494 ms -> 383.119 ms (-18.375 ms)
Convolution:        409.255 ms -> 410.185 ms (noise-level flat)
cacheblend50 e2e profile: 1.204583 s -> 1.186406 s
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_sparse_tile_skip_mask_sweep_1_50_20260611_132737
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.319955,1.014
full-reuse,full,0.638572,3.684

cacheblend,0.01,0.314500,7.480
cacheblend,0.05,0.374485,6.282
cacheblend,0.10,0.431842,5.448
cacheblend,0.20,0.642368,3.662
cacheblend,0.30,0.748229,3.144
cacheblend,0.40,0.889348,2.645
cacheblend,0.50,1.027361,2.290

epic,0.01,0.265385,8.865
epic,0.05,0.313428,7.506
epic,0.10,0.333599,7.052
epic,0.20,0.445318,5.283
epic,0.30,0.547138,4.300
epic,0.40,0.682878,3.445
epic,0.50,0.821617,2.863
```

Repeat smoke:

```text
tag=pic_cuda_1024_sparse_tile_skip_mask_repeat_5_50_20260611_132830
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.05,0.390996,6.017
cacheblend,0.50,1.047438,2.246
epic,0.05,0.309786,7.594
epic,0.50,0.835076,2.817
```

Conclusion:

- Keep this as a safe full-causal sparse tile cleanup. It removes redundant mask loads/branches in the target Llama-3.2 full-attention PIC graph-boundary path and shows a clear `PicSparseAttention` profile reduction.
- Do not count it as a large end-to-end win: no-profile latency still has run-to-run noise and repeat cb50 is close to the previous static-dequant sweep.
- The next P0 work must reduce real tile math / synchronization cost, not just mask overhead. The most promising direction remains adaptive q-piece / bucketed cacheblend sparse flash that reduces K work for scattered active rows while preserving compact output order.
- P1 dense work is still equally important: cacheblend50 remains split between `PicSparseAttention` and compact `Convolution`, while epic50 is dense-bound.

## 2026-06-11 Jetson CUDA High-Budget q16 Sparse Tile

Implemented a high-budget sparse tile variant:

- Keep q8/k32 as the default sparse tile.
- Use q16/k32 only when `attnLen >= 512`. This targets 1024-token 50% budgets where active rows are 519 and q8 launches many small blocks.
- Low budgets and mid budgets remain on q8/k32; cacheblend `attnLen <= 256` still keeps the existing QK/softmax/QKV path.
- This is not the rejected q4 path. q4 increased block count and regressed high budgets; q16 reduces block count and improves K/V reuse at the cost of larger CTAs.

Build:

```text
target=pic_server
CUDA support: ON
CUDA architectures: 7.2
Enabling CUDA support (... archs: sm_72)
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Profile:

```text
tag=pic_cuda_1024_q16_sparse_tile_profile_50_20260611_133723
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50 sparse tile:
  q_tile=16 k_tile=32
  layer=1 us=19476
  layer=2+ us~=19184-19426

cacheblend50 graph:
  Convolution        = 410.138 ms
  PicSparseAttention = 282.117 ms
  PagedAttention     = 87.561 ms
  UnaryOp            = 74.427 ms
  BinaryOp           = 26.398 ms
  PicScoreAttention  = 25.754 ms

epic50 graph:
  Convolution        = 418.182 ms
  PicSparseAttention = 136.829 ms
  PagedAttention     = 80.605 ms
  UnaryOp            = 74.018 ms
  BinaryOp           = 25.848 ms
  PicScoreAttention  = 10.632 ms
```

Compared with q8 + mask-skip profile `pic_cuda_1024_sparse_tile_skip_mask_profile_50_20260611_132617`:

```text
cacheblend50 PicSparseAttention: 383.119 ms -> 282.117 ms (-101.002 ms)
cacheblend50 profile e2e:        1.186406 s -> 1.086636 s
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_q16_sparse_tile_sweep_1_50_20260611_133816
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.318760,1.015
full-reuse,full,0.641728,3.666

cacheblend,0.01,0.319125,7.372
cacheblend,0.05,0.373730,6.295
cacheblend,0.10,0.430131,5.469
cacheblend,0.20,0.641187,3.669
cacheblend,0.30,0.749005,3.141
cacheblend,0.40,0.887050,2.652
cacheblend,0.50,0.912159,2.579

epic,0.01,0.264764,8.885
epic,0.05,0.306151,7.684
epic,0.10,0.334668,7.029
epic,0.20,0.442513,5.316
epic,0.30,0.543946,4.325
epic,0.40,0.678555,3.467
epic,0.50,0.773988,3.039
```

Repeat smoke:

```text
tag=pic_cuda_1024_q16_sparse_tile_repeat_40_50_20260611_133907
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.40,0.954381,2.465
cacheblend,0.50,0.937722,2.509
epic,0.40,0.684346,3.438
epic,0.50,0.794305,2.962
```

Conclusion:

- q16/k32 was a real P0 gain for the 50% budget where sparse attention was still a major bottleneck, but it is now superseded by q32/k32 for the same `attnLen >= 512` threshold.
- q16 does not apply to 40% in the 1024-token setup (`attnLen` is below 512), so the slower cacheblend40 repeat is treated as run-to-run noise rather than q16 regression.
- Compared with the prior q8 no-profile sweep, cacheblend50 improved from `1.027361 s` to `0.912159 s`, and epic50 improved from `0.821617 s` to `0.773988 s`.
- Further sparse attention work should look at wide-Q threshold tuning for larger contexts and cacheblend-specific logical bucket/scatter only if it preserves compact output order. q4 remains rejected.

## 2026-06-11 Jetson CUDA High-Budget q32 Sparse Tile

Tested and accepted q32/k32 for high-budget sparse tile:

- Keep q8/k32 as the default sparse tile for lower budgets.
- Use q32/k32 when `attnLen >= 512`, replacing the q16/k32 intermediate path.
- q32 uses the largest legal CTA size here (`32 x 32 = 1024` threads), so this acceptance is specific to the current sm72 1024-token profile/sweep and must be rechecked for larger contexts or different head dimensions.

Profile:

```text
tag=pic_cuda_1024_q32_sparse_tile_profile_50_20260611_135111
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50 sparse tile:
  q_tile=32 k_tile=32
  layer=1 us=17404
  layer=2+ us~=17099-17449

cacheblend50 graph:
  Convolution        = 410.933 ms
  PicSparseAttention = 252.729 ms
  PagedAttention     = 87.644 ms
  UnaryOp            = 74.383 ms
  BinaryOp           = 26.196 ms
  PicScoreAttention  = 23.336 ms

epic50 graph:
  Convolution        = 416.042 ms
  PicSparseAttention = 123.640 ms
  PagedAttention     = 81.209 ms
  UnaryOp            = 73.979 ms
  BinaryOp           = 26.403 ms
  PicScoreAttention  = 9.505 ms
```

No-profile 40/50 repeat:

```text
tag=pic_cuda_1024_q32_sparse_tile_repeat_40_50_20260611_135207
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.40,0.949852,2.477
cacheblend,0.50,0.908365,2.590
epic,0.40,0.689273,3.413
epic,0.50,0.785227,2.996
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_q32_sparse_tile_sweep_1_50_20260611_135252
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.308713,1.019
full-reuse,full,0.636979,3.693

cacheblend,0.01,0.313335,7.508
cacheblend,0.05,0.372454,6.316
cacheblend,0.10,0.432448,5.440
cacheblend,0.20,0.641337,3.668
cacheblend,0.30,0.747125,3.149
cacheblend,0.40,0.890466,2.642
cacheblend,0.50,0.884471,2.660

epic,0.01,0.264879,8.882
epic,0.05,0.309508,7.601
epic,0.10,0.331932,7.087
epic,0.20,0.442790,5.313
epic,0.30,0.544793,4.318
epic,0.40,0.678758,3.466
epic,0.50,0.764576,3.077
```

Compared with q16:

```text
cacheblend50: 0.912159 s -> 0.884471 s
epic50:       0.773988 s -> 0.764576 s
```

Conclusion:

- Keep q32/k32 for `attnLen >= 512` as the current Jetson CUDA high-budget sparse tile default.
- This is the first P0 change in this round that pushes cacheblend50 under `0.9 s` in a full no-profile sweep while preserving every 1%-50% ratio faster than normal.
- Remaining cb50 bottleneck after q32 is no longer dominated by sparse attention alone; compact `Convolution`, `UnaryOp`, and residual graph overhead need the next P1/P2 work.

## 2026-06-11 Jetson CUDA Sparse Tile K-Slot Cache

Implemented a small P0 cleanup inside `pagedSparseFlashTileKernel`:

- Each K tile now resolves `slotTable[logical]` once into shared `kSlotShared[K_TILE]`.
- K and V shared-memory load phases reuse the same cached physical slot.
- This avoids repeating the same slot-table global load and bounds check for every head_dim lane in both K and V loads.
- No semantic change: active logical indices, causal K limit, PagedCache physical slots, GQA head mapping, q8/q32 tile selection, and full-causal mask skip all stay unchanged.

Build:

```text
target=pic_server
CUDA support: ON
CUDA architectures: 7.2
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Initial no-profile 40/50 smoke had visible run-to-run noise:

```text
tag=pic_cuda_1024_q32_kslot_smoke_40_50_20260611_2204
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.40,0.923296,2.548
cacheblend,0.50,0.890479,2.642
epic,0.40,0.675901,3.481
epic,0.50,0.776560,3.029
```

Graph/profile showed the kernel-level win clearly, compared with q32 baseline `pic_cuda_1024_q32_sparse_tile_profile_50_20260611_135111`:

```text
tag=pic_cuda_1024_q32_kslot_profile_50_20260611_2205
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50 graph:
  Convolution        = 409.960 ms
  PicSparseAttention = 242.378 ms
  PagedAttention     = 87.480 ms
  UnaryOp            = 73.924 ms
  BinaryOp           = 25.910 ms
  PicScoreAttention  = 22.101 ms

epic50 graph:
  Convolution        = 415.936 ms
  PicSparseAttention = 118.928 ms
  PagedAttention     = 80.307 ms
  UnaryOp            = 73.895 ms
  BinaryOp           = 26.017 ms
  PicScoreAttention  = 9.350 ms

Compared with q32 baseline:
  cacheblend50 PicSparseAttention: 252.729 ms -> 242.378 ms
  epic50 PicSparseAttention:       123.640 ms -> 118.928 ms
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_q32_kslot_sweep_1_50_20260611_2207
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.313485,1.017
full-reuse,full,0.634177,3.710

cacheblend,0.01,0.312411,7.530
cacheblend,0.05,0.360835,6.520
cacheblend,0.10,0.427082,5.508
cacheblend,0.20,0.640144,3.675
cacheblend,0.30,0.718657,3.274
cacheblend,0.40,0.859711,2.736
cacheblend,0.50,0.871012,2.701

epic,0.01,0.260832,9.019
epic,0.05,0.300341,7.833
epic,0.10,0.327441,7.185
epic,0.20,0.438567,5.364
epic,0.30,0.534851,4.398
epic,0.40,0.666852,3.528
epic,0.50,0.757372,3.106
```

Compared with the q32 no-profile sweep:

```text
cacheblend50: 0.884471 s -> 0.871012 s
epic50:       0.764576 s -> 0.757372 s
```

Conclusion:

- Keep the shared K-slot cache as part of the default CUDA sparse tile path. It is a small but real P0 attention cleanup, confirmed by profile attribution and a full 1%-50% sweep.
- The remaining high-budget bottleneck is now even more balanced: cacheblend50 still has `PicSparseAttention ~=242 ms` and `Convolution ~=410 ms`; epic50 is clearly compact-dense bound.
- Next CUDA work should move to P1/P2 dense/activation fusion unless doing a larger cacheblend-specific sparse flash redesign that reduces true causal K work for scattered active rows.

Rejected follow-up in the same area:

```text
tag=pic_cuda_1024_q32_kslot_skipclear_profile_50_20260611_2211
change=skip cudaMemset(output) before sparse tile flash

cacheblend50 PicSparseAttention: 242.378 ms -> 242.429 ms
cacheblend50 graph total:        923.615 ms -> 924.581 ms
epic50 PicSparseAttention:       118.928 ms -> 118.493 ms
epic50 graph total:              784.749 ms -> 788.512 ms
```

Do not keep the skip-output-clear change as default. The sparse tile kernel writes all active rows in the current graph-boundary path, but the measured benefit is noise-level and graph total did not improve. Keep the conservative output clear outside any future explicit A/B branch.

## 2026-06-11 Jetson CUDA q32 Threshold 384

Lowered the wide-Q sparse tile threshold:

- Previous default: q32/k32 only when `attnLen >= 512`, so in the 1024-token sweep only 50% budgets used q32.
- New default: q32/k32 when `attnLen >= 384`, so 40% budgets with active rows around 418 also use q32.
- Lower budgets keep the existing q8/k32 or cacheblend `attnLen <= 256` QK/softmax/QKV path.
- This does not change attention semantics, active indices, causal K limit, or PagedCache slot mapping.

No-profile 40/50 smoke:

```text
tag=pic_cuda_1024_q32_kslot_thr384_smoke_40_50_20260611_2223
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.40,0.819659,2.870
cacheblend,0.50,0.897994,2.620
epic,0.40,0.643837,3.654
epic,0.50,0.783233,3.004
```

Graph/profile:

```text
tag=pic_cuda_1024_q32_kslot_thr384_profile_40_50_20260611_2224
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend40 graph:
  Convolution        = 342.747 ms
  PicSparseAttention = 228.638 ms
  PagedAttention     = 87.619 ms
  UnaryOp            = 62.099 ms
  BinaryOp           = 23.607 ms
  PicScoreAttention  = 21.264 ms

cacheblend50 graph:
  Convolution        = 410.863 ms
  PicSparseAttention = 236.728 ms
  PagedAttention     = 80.738 ms
  UnaryOp            = 73.770 ms
  BinaryOp           = 25.774 ms
  PicScoreAttention  = 20.555 ms

epic40 graph:
  Convolution        = 343.400 ms
  PicSparseAttention = 92.426 ms
  PagedAttention     = 80.068 ms
  UnaryOp            = 61.799 ms
  BinaryOp           = 22.298 ms
  PicScoreAttention  = 7.294 ms

epic50 graph:
  Convolution        = 414.465 ms
  PicSparseAttention = 118.701 ms
  PagedAttention     = 80.117 ms
  UnaryOp            = 73.924 ms
  BinaryOp           = 25.664 ms
  PicScoreAttention  = 8.883 ms
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_q32_kslot_thr384_sweep_1_50_20260611_2226
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.303239,1.021
full-reuse,full,0.640213,3.675

cacheblend,0.01,0.311590,7.550
cacheblend,0.05,0.369951,6.359
cacheblend,0.10,0.429822,5.473
cacheblend,0.20,0.641737,3.666
cacheblend,0.30,0.723793,3.250
cacheblend,0.40,0.755525,3.114
cacheblend,0.50,0.872416,2.697

epic,0.01,0.259708,9.058
epic,0.05,0.300698,7.824
epic,0.10,0.326507,7.205
epic,0.20,0.438586,5.364
epic,0.30,0.534949,4.398
epic,0.40,0.636039,3.699
epic,0.50,0.757277,3.107
```

Compared with q32 + k-slot-cache threshold 512 sweep:

```text
cacheblend40: 0.859711 s -> 0.755525 s
cacheblend50: 0.871012 s -> 0.872416 s
epic40:       0.666852 s -> 0.636039 s
epic50:       0.757372 s -> 0.757277 s
```

Conclusion:

- Keep q32/k32 threshold at `attnLen >= 384`. It is a large win for 40% budgets and neutral at 50% in the full sweep.
- Do not push the threshold lower without another full 1%-50% sweep. 30% has active rows around 317 and may not have enough Q work to offset q32's 1024-thread CTA cost.
- After this change, cacheblend40/50 are still split between `PicSparseAttention` and compact dense `Convolution`; epic40/50 are more compact-dense bound. P1 compact dense and P2 activation fusion remain the next larger opportunities.

## 2026-06-11 Jetson CUDA SM70 Compact CUTLASS Dense Fast Path

Tested and accepted a narrow P1 compact dense fast path for score-layer graph-boundary rows:

- Add a sm70 tensor-core CUTLASS Linear variant with `GemmShape<128,64,64>` / warp `64x32x64`.
- Use it only for low-memory INT4 1x1 Linear with static FP16 dequant cache, fp16 inference, no activation, `M in [384,768]`, and padded input channels at least 1024.
- The condition targets 1024-token high-budget compact rows (`cacheblend/epic 40%-50%`) and intentionally avoids low-budget tiny rows and full 1010-row full-compute.
- No environment gate is needed; the default path selects this variant only for the measured compact range and falls back to the existing CUTLASS path for other shapes.

Profile A/B against q32 + k-slot-cache threshold 384:

```text
old tag=pic_cuda_1024_q32_kslot_thr384_profile_40_50_20260611_2224
new tag=pic_cuda_1024_piccompact_sm70_profile_40_50_20260611_223500
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend40:
  Convolution        342.747 ms -> 273.260 ms
  PicSparseAttention 228.638 ms -> 227.874 ms
  graph total        824.835 ms -> 750.962 ms

cacheblend50:
  Convolution        410.863 ms -> 354.839 ms
  PicSparseAttention 236.728 ms -> 236.047 ms
  graph total        908.114 ms -> 851.371 ms

epic40:
  Convolution        343.400 ms -> 272.537 ms
  PicSparseAttention  92.426 ms ->  92.691 ms
  graph total        662.669 ms -> 595.666 ms

epic50:
  Convolution        414.465 ms -> 361.439 ms
  PicSparseAttention 118.701 ms -> 119.267 ms
  graph total        781.244 ms -> 732.635 ms

profile hit counts:
  pic_compact_sm70=1: 403 lines
  pic_compact_sm70=0: 488 lines
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_piccompact_sm70_sweep_1_50_20260611_223627
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.321451,1.013
full-reuse,full,0.636562,3.696

cacheblend,0.01,0.314889,7.471
cacheblend,0.05,0.374720,6.278
cacheblend,0.10,0.430173,5.469
cacheblend,0.20,0.642467,3.662
cacheblend,0.30,0.720687,3.264
cacheblend,0.40,0.687828,3.420
cacheblend,0.50,0.818838,2.873

epic,0.01,0.265916,8.847
epic,0.05,0.303827,7.743
epic,0.10,0.329392,7.142
epic,0.20,0.440432,5.341
epic,0.30,0.538072,4.372
epic,0.40,0.570946,4.120
epic,0.50,0.705095,3.336
```

Compared with q32 + k-slot-cache threshold 384 no-profile sweep:

```text
cacheblend40: 0.755525 s -> 0.687828 s  (+8.96%)
cacheblend50: 0.872416 s -> 0.818838 s  (+6.14%)
epic40:       0.636039 s -> 0.570946 s  (+10.23%)
epic50:       0.757277 s -> 0.705095 s  (+6.89%)
```

Low-budget 1%-30% results are within small noise/regression because the new path does not trigger below `M=384`; all cacheblend/epic ratios remain faster than normal full-compute.

Conclusion:

- Keep the SM70 compact CUTLASS fast path as the default CUDA compact dense high-budget path.
- Do not broaden it below `M=384` without another full 1%-50% sweep; low-budget compact rows previously rejected packed GEMV/PicGEMM attempts and are sensitive to launch/tiling overhead.
- P0 sparse flash remains the cacheblend-specific bottleneck; P1 dense is improved for high budgets, but `Convolution` is still large enough that graph-level gate/up fusion and fused SiLU*up remain worthwhile.

Rejected follow-up: lower the SM70 compact CUTLASS threshold to `M >= 256`.

```text
tag=pic_cuda_1024_piccompact_sm70_thr256_smoke_30_50_20260611_224503
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

Compared with accepted M>=384 sweep:
cacheblend30: 0.720687 s -> 0.736422 s  (-15.735 ms)
cacheblend40: 0.687828 s -> 0.712581 s  (-24.753 ms)
cacheblend50: 0.818838 s -> 0.832377 s  (-13.539 ms)
epic30:       0.538072 s -> 0.528442 s  (+9.630 ms)
epic40:       0.570946 s -> 0.574024 s  (-3.078 ms)
epic50:       0.705095 s -> 0.704313 s  (+0.782 ms)
```

Do not lower the default threshold to 256. It gives a small epic30 win but regresses cacheblend30/40/50, so it violates the default-path rule that cacheblend and epic must both remain faster/no-regression across the sweep.

## 2026-06-11 Jetson CUDA half2 SiLU Unary Fast Path

Tested and accepted a P2 activation optimization:

- Add a contiguous FP16 `UnaryOpOperation_SILU` half2 kernel in CUDA `UnaryBlit`.
- Keep the existing generic unary path for non-contiguous tensors, non-FP16, odd element counts, and other unary ops.
- This is not the full graph-level `SiLU(gate) * up` fusion, but it removes most of the expensive generic SiLU unary time while preserving graph structure and tensor semantics.

Profile A/B against accepted SM70 compact CUTLASS path:

```text
old tag=pic_cuda_1024_piccompact_sm70_profile_40_50_20260611_223500
new tag=pic_cuda_1024_half2_silu_profile_50_20260611_225119
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50 graph:
  UnaryOp            73.874 ms ->   9.009 ms
  BinaryOp           25.802 ms ->  26.302 ms
  Convolution       354.839 ms -> 356.754 ms
  PicSparseAttention 236.047 ms -> 243.252 ms
  graph total       851.371 ms -> 807.220 ms

epic50 graph:
  UnaryOp            74.187 ms ->   8.979 ms
  BinaryOp           26.307 ms ->  26.158 ms
  Convolution       361.439 ms -> 358.732 ms
  PicSparseAttention 119.267 ms -> 119.180 ms
  graph total       732.635 ms -> 664.275 ms

compact SiLU representative:
  /blocks.* /mlp/act_fn/Mul_output_0, input [1x519x8192]
  about 4.16 ms/layer -> about 0.34 ms/layer
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_half2_silu_sweep_1_50_20260611_225217
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.190676,1.074
full-reuse,full,0.637651,3.689

cacheblend,0.01,0.310606,7.574
cacheblend,0.05,0.354184,6.642
cacheblend,0.10,0.410112,5.736
cacheblend,0.20,0.609088,3.862
cacheblend,0.30,0.678736,3.466
cacheblend,0.40,0.635933,3.699
cacheblend,0.50,0.753021,3.124

epic,0.01,0.252736,9.308
epic,0.05,0.287881,8.172
epic,0.10,0.307184,7.658
epic,0.20,0.406571,5.786
epic,0.30,0.494993,4.753
epic,0.40,0.515728,4.562
epic,0.50,0.635977,3.699
```

Correctness smoke:

```text
Jetson command:
  run_test.out op/unary/silu 2 1

result:
  all <op/unary/silu> tests passed
```

Compared with accepted SM70 compact CUTLASS no-profile sweep:

```text
full-compute: 2.321451 s -> 2.190676 s  (+5.63%)
cacheblend10: 0.430173 s -> 0.410112 s  (+4.66%)
cacheblend20: 0.642467 s -> 0.609088 s  (+5.20%)
cacheblend30: 0.720687 s -> 0.678736 s  (+5.82%)
cacheblend40: 0.687828 s -> 0.635933 s  (+7.54%)
cacheblend50: 0.818838 s -> 0.753021 s  (+8.04%)
epic10:       0.329392 s -> 0.307184 s  (+6.74%)
epic20:       0.440432 s -> 0.406571 s  (+7.69%)
epic30:       0.538072 s -> 0.494993 s  (+8.01%)
epic40:       0.570946 s -> 0.515728 s  (+9.67%)
epic50:       0.705095 s -> 0.635977 s  (+9.80%)
```

Conclusion:

- Keep the CUDA half2 SiLU fast path as default. It benefits full-compute and all sparse ratios, and no target/async/error log lines appeared.
- The remaining elementwise opportunity is the actual graph-level `SiLU(gate) * up` fusion, which would remove the separate BinaryOp and some raster/intermediate traffic. Do not call that done just because unary SiLU is now fast.
- After this optimization, cacheblend50 is again mostly split between `PicSparseAttention` and compact `Convolution`; `UnaryOp` is no longer the first P2 bottleneck.

## 2026-06-11 Jetson CUDA Rejected Sparse Flash Register Accumulator

Goal:

- Reduce CUDA `pagedSparseFlashTileKernel` hot-loop shared-memory traffic by moving the output accumulator from
  `outShared[Q_TILE][HEAD_DIM]` into two per-thread register accumulators.
- Test this as a P0 sparse flash attention follow-up after the accepted k-slot cache and q32/k32 threshold.

Code/env:

- Changed only `source/backend/cuda/execution/PagedAttentionExecution.cu` after the accepted q32/k-slot path:
  removed `outShared`, used `accIdx0=linearTid` and `accIdx1=linearTid+linearThreads`, and wrote output directly
  from `outAcc0/outAcc1`.
- Built locally with Jetson CUDA cross compile and synced to Jetson.
- Profile tag:

```text
pic_cuda_1024_sparse_regacc_profile_50_20260611_150559
```

Profile result:

```text
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

profile latency:
cacheblend50: 0.982005 s
epic50:       0.776255 s

CUDAPagedAttention op=sparse_flash_tile_attention across cacheblend50+epic50:
  count=30
  total=441.449 ms
  max_layer=9.237 ms
```

Compared with accepted half2 SiLU profile:

```text
accepted profile latency:
cacheblend50: 0.807220 s
epic50:       0.664275 s

accepted PicSparseAttention graph total:
cacheblend50: 243.252 ms
epic50:       119.180 ms
total:        362.432 ms
```

Conclusion:

- Reject this variant. It worsens the sparse flash portion instead of improving it, likely because the extra private
  registers reduce occupancy / scheduling efficiency on sm72 more than the removed shared-memory accumulator helps.
- Reverted the register-accumulator source change and rebuilt/synced the accepted artifact back to Jetson.
- Do not reintroduce this `outAcc0/outAcc1` variant as a default CUDA sparse flash path.

Post-revert smoke:

```text
tag=pic_cuda_1024_after_regacc_revert_smoke_50_20260611_150905
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

full-compute: 2.201449 s
full-reuse:   0.638487 s
cacheblend50: 0.806360 s
epic50:       0.656208 s
```

## 2026-06-11 Jetson CUDA SM70 Compact CUTLASS Hybrid Tile

Goal:

- Continue P1 compact dense optimization after static dequant cache and the accepted `128x64x64` SM70 compact CUTLASS path.
- Target the remaining 50% budget compact MLP shapes:
  - `M=519,K=2048,N=8192` gate/up
  - `M=519,K=8192,N=2048` down

Code/env:

- Added a second SM70 FP16 tensor-core Linear instance:

```text
GemmTensor_F16_F16_Linear_AlignTensor_Sm70_64x128x64
```

- Added `mPicCompactSm70Tile` profile metadata:
  - `0`: normal CUTLASS path
  - `1`: existing PIC compact `128x64x64`
  - `2`: new PIC compact `64x128x64`
- Kept the path default and shape-gated, with no env switch:

```text
384 <= M < 512: 128x64x64
M >= 512:       64x128x64
```

Rejected intermediate:

```text
tag=pic_cuda_1024_piccompact_nwide_smoke_40_50_20260611_151644

cacheblend40: 0.695613 s
cacheblend50: 0.739442 s
epic40:       0.517384 s
epic50:       0.624207 s
```

The all-compact `64x128x64` tile improves 50% but regresses cacheblend40 badly compared with the accepted half2 SiLU sweep
(`cacheblend40=0.635933 s`). Do not use it for `M < 512`.

Profile evidence for `M=519`:

```text
tag=pic_cuda_1024_piccompact_nwide_profile_50_20260611_151532
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

accepted 128x64x64 profile, compact M=519:
  2048->8192: avg cutlass 5452.2 us
  8192->2048: avg cutlass 5552.8 us
  2048->2048: avg cutlass 1397.7 us

64x128x64 profile, compact M=519:
  2048->8192: avg cutlass 4916.4 us
  8192->2048: avg cutlass 4722.0 us
  2048->2048: avg cutlass 1267.6 us
```

Accepted no-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_piccompact_hybrid_nwide_sweep_1_50_20260611_151823
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.192074,1.073
full-reuse,full,0.636807,3.694

cacheblend,0.01,0.308307,7.630
cacheblend,0.05,0.363945,6.464
cacheblend,0.10,0.414109,5.681
cacheblend,0.20,0.611740,3.846
cacheblend,0.30,0.678835,3.466
cacheblend,0.40,0.634798,3.706
cacheblend,0.50,0.721868,3.259

epic,0.01,0.254410,9.247
epic,0.05,0.290160,8.108
epic,0.10,0.308750,7.620
epic,0.20,0.408996,5.752
epic,0.30,0.494652,4.756
epic,0.40,0.514860,4.569
epic,0.50,0.606127,3.881
```

Compared with the accepted half2 SiLU sweep:

```text
cacheblend50: 0.753021 s -> 0.721868 s  (+4.14%)
epic50:       0.635977 s -> 0.606127 s  (+4.69%)
cacheblend40: 0.635933 s -> 0.634798 s  (+0.18%)
epic40:       0.515728 s -> 0.514860 s  (+0.17%)
```

Conclusion:

- Accept the hybrid SM70 compact CUTLASS tile as the default P1 dense path.
- Keep `64x128x64` gated to `M >= 512`; below that, the existing `128x64x64` path is safer.
- This is a compact dense win, not an attention win. After it, cacheblend50 still needs P0 sparse attention work, while epic50 is pushed further toward dense/MLP and graph overhead.

## 2026-06-11 Jetson CUDA Rejected Mid-Budget q16 Sparse Tile

Goal:

- Test whether `384 <= attnLen < 512` should use a q16/k32 sparse flash tile instead of the accepted q32/k32 tile.
- Motivation: 40% budgets have active rows around 418, so q32 might waste K streaming for scattered cacheblend rows even though it reduces block count.

Code/env:

- Temporary change in `source/backend/cuda/execution/PagedAttentionExecution.cu`:

```text
attnLen < 384:   q8/k32
384 <= attnLen < 512: q16/k32
attnLen >= 512:  q32/k32
```

- Built locally with Jetson CUDA cross compile and synced to Jetson.

Profile run:

```text
tag=pic_cuda_1024_sparse_midq16_profile_40_50_20260611_152446
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

sparse tile summary:
  attn=418 q_tile=16 count=29 total=339.458 ms max=17.075 ms
  attn=519 q_tile=32 count=28 total=332.758 ms max=16.353 ms

graph profile:
cacheblend40 PicSparseAttention: 247.913 ms
cacheblend50 PicSparseAttention: 235.928 ms
epic40       PicSparseAttention:  96.965 ms
epic50       PicSparseAttention: 118.787 ms
```

No-profile smoke:

```text
tag=pic_cuda_1024_sparse_midq16_smoke_40_50_20260611_152613
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend40: 0.720326 s
cacheblend50: 0.743643 s
epic40:       0.522904 s
epic50:       0.623024 s
```

Compared with accepted hybrid dense + q32 threshold sweep:

```text
cacheblend40: 0.634798 s -> 0.720326 s  (regression)
cacheblend50: 0.721868 s -> 0.743643 s  (regression/noise)
epic40:       0.514860 s -> 0.522904 s  (regression)
epic50:       0.606127 s -> 0.623024 s  (regression)
```

Conclusion:

- Reject q16 for the `384 <= attnLen < 512` mid-budget range.
- Reverted to the accepted rule: q32/k32 when `attnLen >= 384`, q8/k32 below that.
- q32's larger CTA still wins for the current 1024-token 40% shape despite potential scattered-row K waste. Future P0 must reduce causal K work with a more explicit bucket/adaptive-piece design, not by simply shrinking q32 to q16.

Post-revert smoke:

```text
tag=pic_cuda_1024_after_midq16_revert_smoke_40_50_20260611_152900
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend40: 0.701642 s
cacheblend50: 0.737961 s
epic40:       0.521909 s
epic50:       0.626386 s
```

## 2026-06-23 Rhino Pi-X1 / Adreno q-split Full-Prefill Fix

Goal:

- Verify whether Rhino Pi-X1 normal `llm_bench` and PIC server prefill are using chunk forward.
- Fix the Adreno OpenCL full-prefill path without adding a temporary branch.
- Re-run formal `max_tokens=0` latency for 1024/2048 prompt lengths at cacheblend/epic budgets `0.05/0.40/0.50`, plus PIC full-compute/full-reuse and normal `llm_bench` baseline.

Chunk-forward conclusion:

- `Llm::set_config` only enables chunk forward when `config["chunk"]` or `config["chunk_limits"]` is present.
- Rhino normal and PIC OpenCL configs both have `chunk=None` and `chunk_limits=None`.
- Therefore the current Rhino normal baseline and PIC graph-boundary prefill are not chunk forward; they run a single prefill forward for these prompt lengths.

Code change:

- `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
- In the Adreno GEMM full-prefill path, `q_split > 1` wrote QK for each piece at an offset, while softmax/QKV consumed from offset 0. The production fix reuses the same per-piece QK/softmax scratch instead of carrying a stale piece offset.
- The QK/softmax scratch allocation was also corrected from `qChunkPack * kvLen` to `qChunkPack * kvPack`, matching the 32-aligned K dimension used by the GEMM path.

Build/deploy:

```text
MNN_TARGET_DEVICE=aidlux_adreno_opencl BUILD_TARGET=MNN_CL BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

MNN_TARGET_DEVICE=aidlux_adreno_opencl BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Remote sanity:

```text
device: Rhino Pi-X1 / Adreno740v2
kgsl: /dev/kgsl-3d0 present
normal_config: /mnt/nvme/mnn_pic_opencl/models/normal/AI-ModelScope__Llama-3.2-1B-Instruct/config_opencl_greedy.json
pic_config:    /mnt/nvme/mnn_pic_opencl/models/pic/AI-ModelScope__Llama-3___2-1B-Instruct-pic-boundary/config_opencl_greedy.json
configs: chunk=None, chunk_limits=None
artifact: /mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl
```

Formal run:

```text
run_id=rhino_adreno_qsplit_fix_clean_ctx1024_2048_cbepic_005_040_050_20260623
local_summary=.cache/latency_budget_20260622/rhino_adreno_qsplit_fix_clean_ctx1024_2048_cbepic_005_040_050_20260623/summary.csv
remote_cache=/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/rhino_adreno_qsplit_fix_clean_ctx1024_2048_cbepic_005_040_050_20260623
warm: each context/mode/ratio before formal measure
/v1/tune/update_cache: status=200 for ctx1024 and ctx2048
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
```

Results:

```text
context,mode,budget,latency_s,recompute,reuse,selected,speedup_vs_normal,speedup_vs_pic_full
1024,normal-full-compute,full,3.014913,,,,1.000,
1024,PIC full-compute,full,3.209506,1010,0,,0.939,1.000
1024,full-reuse,full,0.364463,0,1010,,8.272,8.806
1024,cacheblend,0.05,0.657118,51,959,51,4.588,4.884
1024,cacheblend,0.40,1.767268,404,606,404,1.706,1.816
1024,cacheblend,0.50,2.218288,505,505,505,1.359,1.447
1024,epic,0.05,0.612807,51,959,,4.920,5.237
1024,epic,0.40,1.651901,404,606,,1.825,1.943
1024,epic,0.50,2.007737,505,505,,1.502,1.599
2048,normal-full-compute,full,6.323234,,,,1.000,
2048,PIC full-compute,full,6.696720,2034,0,,0.944,1.000
2048,full-reuse,full,0.442758,0,2034,,14.281,15.125
2048,cacheblend,0.05,1.230309,102,1932,102,5.140,5.443
2048,cacheblend,0.40,4.228442,814,1220,814,1.495,1.584
2048,cacheblend,0.50,5.207544,1017,1017,1017,1.214,1.286
2048,epic,0.05,1.058596,102,1932,,5.973,6.326
2048,epic,0.40,3.279326,814,1220,,1.928,2.042
2048,epic,0.50,3.902066,1017,1017,,1.620,1.716
```

Conclusion:

- The Rhino performance issue was not caused by LLM chunk forward; both configs are non-chunked.
- PIC full-compute is now close to normal full-compute on Adreno: about 6% overhead at 1024/2048.
- full-reuse remains dominated by hydrate/suffix work and is 8.27x faster than normal at 1024, 14.28x at 2048.
- cacheblend and epic remain faster than normal even at 50% recompute on Rhino/Adreno. Epic is consistently faster than cacheblend at high budgets because its selected rows are prefix-contiguous rather than cacheblend's scattered top-k rows.
- No production env fallback or temporary branch is required; keep the Adreno GEMM q-split scratch fix as the default path.
