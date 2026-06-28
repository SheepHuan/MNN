# Jetson CUDA decode repair 08:00 context

## Baseline

Accepted baseline from 07:00:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_smoke2_20260628_0710/summary.csv
x=0 none                  TPOT=86.377 ms
x=1 lagged_attention_hkvd TPOT=97.471 ms
x=3 lagged_attention_hkvd TPOT=123.602 ms
x=5 lagged_attention_hkvd TPOT=124.975 ms
x=7 lagged_attention_hkvd TPOT=126.360 ms
```

Profile conclusion carried forward:

- `decode_attention_rank` is not a bottleneck.
- CPU bookkeeping/rank/result handling is microsecond-level.
- Remaining gap is dominated by `forward_raw`: qtile attention plus compact dense/launch overhead.

## cuBLAS rows4-8 runtime policy A/B

All runs used the accepted rows4-8 cuBLAS artifact and only changed server env.

### `32f_fast16`

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_fast16_20260628_0725/summary.csv
x=0 none                  TPOT=86.330 ms
x=1 lagged_attention_hkvd TPOT=97.258 ms
x=3 lagged_attention_hkvd TPOT=124.134 ms
x=5 lagged_attention_hkvd TPOT=125.458 ms
x=7 lagged_attention_hkvd TPOT=126.811 ms
```

Decision: reject. No broad win over default.

### `16f`

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_16f_20260628_0730/summary.csv
x=0 none                  TPOT=87.911 ms
x=1 lagged_attention_hkvd TPOT=97.309 ms
x=3 lagged_attention_hkvd TPOT=127.130 ms
x=5 lagged_attention_hkvd TPOT=124.458 ms
x=7 lagged_attention_hkvd TPOT=126.324 ms
```

Decision: reject. `x=5` has noise-level gain, but `x=3` regresses.

### `rows45_cublas=down`

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_down_20260628_0735/summary.csv
x=0 none                  TPOT=87.137 ms
x=1 lagged_attention_hkvd TPOT=97.080 ms
x=3 lagged_attention_hkvd TPOT=128.392 ms
x=5 lagged_attention_hkvd TPOT=131.374 ms
x=7 lagged_attention_hkvd TPOT=134.367 ms
```

Decision: reject. Gate/up also need the cuBLAS route; down-only is worse.

### `algo=default`

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_algo_default_20260628_0740/summary.csv
x=0 none                  TPOT=86.042 ms
x=1 lagged_attention_hkvd TPOT=97.471 ms
x=3 lagged_attention_hkvd TPOT=123.336 ms
x=5 lagged_attention_hkvd TPOT=125.084 ms
x=7 lagged_attention_hkvd TPOT=126.856 ms
```

Decision: reject as default. `x=3` is a small noise-level win, but `x=7` is worse and the improvement is not broad.

### Same-time default repeat

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_default_repeat_20260628_0745/summary.csv
x=0 none                  TPOT=86.379 ms
x=1 lagged_attention_hkvd TPOT=96.730 ms
x=3 lagged_attention_hkvd TPOT=122.668 ms
x=5 lagged_attention_hkvd TPOT=124.696 ms
x=7 lagged_attention_hkvd TPOT=225.167 ms
```

`x=7` had one repeat outlier:

```text
repeat_1 TPOT=126.852 ms
repeat_2 TPOT=126.733 ms
repeat_3 TPOT=421.914 ms
```

Treat this as run noise, not a default regression.

## qtile causal-guard source A/B

Hypothesis:

- Decode repair qtile packs sorted PIC repair rows plus the final normal decode row in one tile.
- The qtile kernel uses tile-level `qMaxLogical` for K streaming.
- Existing score code rejects `logical > qLogical`, but partial QK dot was still computed first.
- Add the same causal condition before partial dot to reduce invalid QK work.

Source change tested:

```text
source/backend/cuda/execution/PagedAttentionExecution.cu
```

Patch shape:

```text
if (qValid && kLocal < kTile && (kTileFull || logical <= qLogical)) {
    partial += ...
}
```

Build/sync:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/jetson/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

Result:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_causal_guard_20260628_0750/summary.csv
x=0 none                  TPOT=87.790 ms
x=1 lagged_attention_hkvd TPOT=98.570 ms
x=3 lagged_attention_hkvd TPOT=124.040 ms
x=5 lagged_attention_hkvd TPOT=127.093 ms
x=7 lagged_attention_hkvd TPOT=128.056 ms
```

Decision:

- Reject and revert.
- The extra per-`kLocal` branch/condition costs more than the skipped invalid dot work on Xavier for this shape.
- Future attention work should avoid mixed-logical tile waste structurally, not by adding a hot-loop branch in the current qtile kernel.

## Restore

The causal-guard patch was reverted. The accepted rows4-8 cuBLAS source was rebuilt and synced back to Jetson.

Restore smoke, repeat1 only:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_restore_smoke_20260628_0800/summary.csv
x=0 none                  TPOT=88.113 ms
x=1 lagged_attention_hkvd TPOT=100.341 ms
x=3 lagged_attention_hkvd TPOT=127.629 ms
x=5 lagged_attention_hkvd TPOT=126.457 ms
x=7 lagged_attention_hkvd TPOT=128.031 ms
```

This repeat1 smoke confirms the endpoint runs after restore, but it is too noisy to replace the 07:00 repeat3 accepted baseline.

## Build note

Both rebuilds still hit the known cross cache false-stale issue:

```text
cached: <empty>
expected: .../aarch64-none-linux-gnu-gcc
```

This remains build-time overhead, not a runtime blocker.

## Active A/B: square projection cuBLAS route

Source file:

```text
source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
```

Patch shape under test:

```text
const bool isSquareProjection = minDim >= 1024 && ic == oc;
return isMlpDown || isMlpGateOrUp || isSquareProjection;
```

Rationale:

- The accepted rows4-8 cuBLAS route fixed the largest MLP-like compact dense overhead for decode repair.
- Profile-only attribution still shows batch4/6/8 `3072 -> 3072` square projection Linears taking about `270 us` median on CUTLASS.
- This A/B tests whether moving those compact square projections to cuBLAS reduces `forward_raw` overhead for `x=3/5/7`.

Validation plan:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/jetson/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/

conda run -n kvshare-edge python .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://127.0.0.1:19131 \
  --device jetson --device-display Jetson --backend cuda \
  --model llama-pic --model-name 'Llama3.2 3B' \
  --model-config .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json \
  --contexts 1024 --budgets 0.00 \
  --decode-selectors lagged_attention_hkvd \
  --repair-tokens 0,1,3,5,7 \
  --max-tokens 8 --repeats 3 --warm-repeats 1 \
  --output-csv .cache/mnn-pic-benchmark/decode_repair_rows48_square_cublas_20260628_08xx/summary.csv \
  --output-dir .cache/mnn-pic-benchmark/decode_repair_rows48_square_cublas_20260628_08xx/client
```

Decision rule:

- Keep only if repeat3 improves broadly for `x=3/5/7` and does not regress `x=0/x=1`.
- Revert if wins are noise-level or isolated, because square projections are frequent and a neutral route adds risk without enough TPOT benefit.

Result:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_square_cublas_20260628_0848/summary.csv
x=0 none                  TPOT=87.419 ms
x=1 lagged_attention_hkvd TPOT=98.843 ms
x=3 lagged_attention_hkvd TPOT=123.998 ms
x=5 lagged_attention_hkvd TPOT=125.594 ms
x=7 lagged_attention_hkvd TPOT=126.822 ms
```

Decision:

- Reject. Compared with the accepted 07:00 repeat3 baseline, `x=0/x=1` are worse and `x=3/5/7` do not get a broad win.
- Reverted only the square-projection extension in `picRows45CublasMatches`; the accepted rows4-8 MLP-like cuBLAS route stays in place.
- Interpretation: square `3072 -> 3072` projections are not good candidates for this cuBLAS route on Xavier at batch4/6/8; continue looking at launch/fusion or a dedicated tiny-row dense path instead of expanding the rows4-8 cuBLAS matcher further.

## Active A/B: decode-repair qtile variants

After rejecting square-projection cuBLAS, the next low-risk check is qtile variant selection. Current source default for decode repair is:

```text
headDim=128, attnLen>=2 -> hd128_q8k16
```

Env-only candidates:

```text
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=2
MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q4k16

MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q4k8
```

Decision rule:

- Keep source default unchanged unless a candidate improves repeat3 `x=1/3/5/7` broadly without hurting `x=0`.
- If a candidate only helps one repair count or is within noise, reject it as an env/profile-only finding.

### `hd128_q4k16`

```text
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_q4k16_20260628_0855/summary.csv
x=0 none                  TPOT=88.035 ms
x=1 lagged_attention_hkvd TPOT=114.705 ms
x=3 lagged_attention_hkvd TPOT=141.308 ms
x=5 lagged_attention_hkvd TPOT=142.521 ms
x=7 lagged_attention_hkvd TPOT=145.220 ms
```

Decision: reject. This is much slower than default `hd128_q8k16` for all repair counts.

### `hd128_q4k8`

```text
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_q4k8_20260628_0858/summary.csv
x=0 none                  TPOT=86.779 ms
x=1 lagged_attention_hkvd TPOT=116.506 ms
x=3 lagged_attention_hkvd TPOT=143.489 ms
x=5 lagged_attention_hkvd TPOT=145.505 ms
x=7 lagged_attention_hkvd TPOT=149.690 ms
```

Decision: reject. Smaller K tile worsens all repair counts. Keep source default `hd128_q8k16`.

Conclusion:

- The qtile regression is too large to be run noise.
- Current decode-repair qtile bottleneck is not fixed by reducing Q tile from 8 to 4 or K tile from 16 to 8.
- Further attention work needs a structural tiny-row kernel or less launch overhead, not another existing qtile variant.

## Active source A/B: `hd128_q8k8`

Patch shape:

```text
source/backend/cuda/execution/PagedAttentionExecution.cu
add env-gated qtile variant hd128_q8k8 -> pagedSparseFlashMQTileKernel<half, 128, 8, 8>
```

Rationale:

- `q4k16` and `q4k8` proved that reducing Q tile to 4 is bad.
- `q8k8` keeps the accepted Q tile 8 but halves K tile shared state, testing whether smaller K chunks help tiny decode repair without the Q-tile regression.
- The variant is env-only; source default remains `hd128_q8k16` unless the A/B wins.

Run env:

```text
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=2
MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q8k8
```

Result:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_q8k8_20260628_0906/summary.csv
x=0 none                  TPOT=87.766 ms
x=1 lagged_attention_hkvd TPOT=102.254 ms
x=3 lagged_attention_hkvd TPOT=128.661 ms
x=5 lagged_attention_hkvd TPOT=129.817 ms
x=7 lagged_attention_hkvd TPOT=131.013 ms
```

Decision:

- Reject. `q8k8` is less bad than the Q-tile-4 variants, but still loses to accepted default `q8k16` across repair counts.
- Revert the env-only source variant. Keep default decode repair qtile unchanged.

## Restore after rejected A/B

Rejected source changes reverted:

- Square-projection cuBLAS admission in `ConvFpAIntBExecution.cu`.
- Env-only `hd128_q8k8` qtile variant in `PagedAttentionExecution.cu`.

Restore build:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Restore sync:

```bash
rsync -a --delete .cache/output/mnn/artifacts/jetson/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

Checks:

```text
file pic_server/libMNN_Cuda_Main.so: ARM aarch64
/v1/models via restored default server: HTTP 200
q8k8 strings in PagedAttentionExecution.cu: none
```

Cleanup:

```text
remote 18131 pic_server: stopped
local 19131 tunnel: stopped
```

## Continuation goal and current hypothesis

User goal for the next pass:

- Keep measuring normal decode (`x=0 none`) alongside decode repair.
- Reduce `x=1/3/5/7 lagged_attention_hkvd` TPOT toward `x=0`.
- Reduce both non-compute overhead and actual kernel compute time.
- Prefer targeted changes that preserve the current lagged attention + HKVD selection semantics.

Accepted source state to preserve:

- `ConvFpAIntBExecution.cu` rows4-8 MLP-like cuBLAS admission remains enabled.
- Static decode-hot dequant cache remains enabled.
- Rejected A/B changes remain removed:
  - square projection cuBLAS admission;
  - qtile causal hot-loop guard;
  - qtile `hd128_q4k16`, `hd128_q4k8`, and env-only `hd128_q8k8` as defaults.

Working hypothesis:

- The compact attention ranking design is already mostly fused in CUDA: row-compressed decode attention can accumulate ranking scores while doing attention, then only compact top-M indices are copied back.
- Prior profile attributed `decode_attention_rank` to about `4.019 ms` total over the profile run, so the current large TPOT gap is unlikely to come from CPU ranking bookkeeping alone.
- The next optimization should first confirm the fused rank path in source, then use profile to split `forward_raw` into attention, dense/MLP, activation/BinaryOp, and launch overhead.

Immediate next checks:

1. Audit `PagedAttentionExecution.cu` around decode repair rank capture and row-compressed attention to confirm whether default non-qtile decode repair still uses fused score accumulation.
2. Audit `Llm::selectPicDecodeRepairLogicalIndices()` to confirm `lagged_attention_hkvd` only scans compact rank vectors and does not copy full attention weights.
3. If source confirms the rank path is compact, run a targeted Jetson profile with `MNN_PIC_DECODE_REPAIR_PROFILE=1` and `MNN_PAGED_ATTENTION_PROFILE=1` for `x=1/3/7`, then decide whether the next code change should target dense/MLP or attention/launch overhead.

## Continuation profile: rank is already compact/fused

Source audit:

- `PagedKVMeta` stores only compact decode attention rank state:
  - `pic_decode_attention_ranked_local_indices`
  - `pic_decode_attention_rank_source_step_idx`
  - `pic_decode_attention_top_m`
- `Llm::selectPicDecodeRepairLogicalIndices()` builds a local bitmap from `lastAttentionRankedPicLocalIndices`, scans `rankedLogicalIndices` in HKVD order, selects tokens in the attention candidate pool, then falls back to HKVD fill.
- CUDA default decode repair uses `pagedAttentionRowCompressedMaskKernel` or qtile attention with `decodeAttentionScores` accumulation when rank capture is active.
- `finishDecodeAttentionRankCaptureCUDA()` only runs compact top-k and copies `top_m` int indices back to host.

Profile run:

```text
run_id=decode_repair_profile_cont_20260628_08
summary=.cache/mnn-pic-benchmark/decode_repair_profile_cont_20260628_08/summary.csv
server_log=.cache/mnn-pic-benchmark/decode_repair_profile_cont_20260628_08/remote/pic_server.log
server_env=MNN_PIC_DECODE_REPAIR_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE=1
```

Profile-only TPOT:

```text
x=0 none                  TPOT=91.944 ms
x=1 lagged_attention_hkvd TPOT=119.551 ms
x=3 lagged_attention_hkvd TPOT=152.461 ms
x=5 lagged_attention_hkvd TPOT=155.550 ms
x=7 lagged_attention_hkvd TPOT=156.818 ms
```

Step-level attribution, excluding step0 to avoid prefill contamination:

```text
x=1 segments=14 forward_raw_med=104.873ms attention_per_step=23.747ms conv_per_step=52.393ms rank_per_step=0.238ms
x=3 segments=14 forward_raw_med=135.382ms attention_per_step=24.946ms conv_per_step=79.078ms rank_per_step=0.227ms
x=5 segments=14 forward_raw_med=138.214ms attention_per_step=25.900ms conv_per_step=78.443ms rank_per_step=0.241ms
x=7 segments=14 forward_raw_med=139.195ms attention_per_step=27.339ms conv_per_step=78.398ms rank_per_step=0.271ms
```

Rank capture:

```text
decode_attention_rank lines=64
fused_score_in_attention=64/64
sum=16.107ms
mean=251.7us
median=244.0us
prep_sum=1.611ms
topk_init_sum=1.096ms
topk_sum=8.790ms
d2h_sum=4.069ms
top_m=32 for all profiled steps
```

Conclusion:

- The requested compact ranking design is already implemented on the current CUDA path.
- There is no full `[heads, kv_len]` attention matrix copy in this path.
- Reducing `top_m` below 32 can only save about `0.25 ms/step` at most and is not the main route to bring `x=1/3/5/7` close to `x=0`.
- Main optimization targets are compact dense/MLP, qtile attention, graph-level Raster/BinaryOp/While/Unary launch overhead.

## Graph profile: x=7

Run:

```text
run_id=decode_repair_graph_profile_x7_20260628_08
summary=.cache/mnn-pic-benchmark/decode_repair_graph_profile_x7_20260628_08/summary.csv
server_log=.cache/mnn-pic-benchmark/decode_repair_graph_profile_x7_20260628_08/remote/pic_server.log
server_env=MNN_PIC_DECODE_REPAIR_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=80
```

Profile-only result:

```text
x=7 lagged_attention_hkvd TPOT=213.852 ms
```

Graph summary:

```text
MNN_PIC_GRAPH_PROFILE_SUMMARY request=1 label=mode=full-reuse,ratio=0,score_layer=1,max_tokens=8 total_ms=1377.615 calls=10744 unique_ops=2381 unique_types=10
Convolution         788.956 ms, calls=1773
PicSparseAttention  268.379 ms, calls=234
Raster              141.019 ms, calls=4138
BinaryOp             58.927 ms, calls=1773
While                44.598 ms, calls=1260
UnaryOp              24.107 ms, calls=774
LayerNorm            16.269 ms, calls=513
PicScoreAttention    13.710 ms, calls=9
PagedAttention       11.945 ms, calls=9
Cast                  9.705 ms, calls=261
```

Top individual ops include one large first-layer K projection outlier in step0, then per-layer `PicSparseAttention`, `mlp/down_proj`, `mlp/gate_proj`, and `mlp/up_proj`. This reinforces that the next source optimization should target compact dense/MLP or graph-level elementwise/launch overhead.

## QTile-off A/B

Run:

```text
server_env=MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=0
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_off_20260628_08/summary.csv
```

Result:

```text
x=0 none                  TPOT=88.125 ms
x=1 lagged_attention_hkvd TPOT=101.057 ms
x=3 lagged_attention_hkvd TPOT=137.788 ms
x=5 lagged_attention_hkvd TPOT=150.136 ms
x=7 lagged_attention_hkvd TPOT=172.044 ms
```

Decision:

- Reject. Disabling qtile hurts all repair counts.
- Keep default `hd128_q8k16`.
- The earlier source comment that qtile is better than row-compressed for rows2/4 still holds for end-to-end TPOT even if individual profile medians look close.

## Attention rank pool/head A/B

Rationale:

- `decode_attention_rank` is small, but it is still invoked once per decode step.
- Current benchmark default uses `top_m=32` and all attention heads.
- Reducing `top_m` and selected heads changes the candidate-pool semantics, so this can only be a speed-profile config until output quality is checked.

### `top_m=8`, all heads

Run:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_topm8_20260628_08/summary.csv
```

Result:

```text
x=0 none                  TPOT=88.274 ms
x=1 lagged_attention_hkvd TPOT=98.374 ms
x=3 lagged_attention_hkvd TPOT=124.610 ms
x=5 lagged_attention_hkvd TPOT=125.719 ms
x=7 lagged_attention_hkvd TPOT=125.705 ms
```

Decision:

- Noise-level, not a clear default improvement.

### `top_m=8`, `attention_head_ids=0`

Run:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_topm8_head0_20260628_08/summary.csv
```

Result:

```text
x=0 none                  TPOT=86.255 ms
x=1 lagged_attention_hkvd TPOT=97.108 ms
x=3 lagged_attention_hkvd TPOT=123.215 ms
x=5 lagged_attention_hkvd TPOT=124.826 ms
x=7 lagged_attention_hkvd TPOT=125.641 ms
```

Raw repeat stability:

```text
x=1 96.183 / 97.444 / 97.699 ms
x=3 123.861 / 122.561 / 123.224 ms
x=5 125.471 / 124.365 / 124.641 ms
x=7 125.458 / 125.209 / 126.255 ms
```

Same-time default control had long-tail outliers and is not a clean average comparison:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_default_control_20260628_08_cont/summary.csv
x=1 repeat TPOT=97.614 / 173.384 / 126.317 ms
x=3 repeat TPOT=125.466 / 123.103 / 181.746 ms
x=5 repeat TPOT=130.960 / 124.626 / 563.437 ms
x=7 repeat TPOT=132.366 / 131.727 / 127.478 ms
```

Decision:

- `top_m=8, head0` is stable and slightly faster than the accepted 07:00 baseline, but the gain is only about `0.1-0.7 ms` and selector semantics change.
- Do not make it the default without an output-quality check.
- If the immediate goal is purely TPOT speed profiling, it is a reasonable optional config to include in the next benchmark table.

## 08:39 continuation checkpoint

User asked to record the optimization log before continuing.

Current source/default position:

- Keep the accepted rows4-8 MLP-like cuBLAS route in `ConvFpAIntBExecution.cu`.
- Keep the static decode-hot dequant cache.
- Do not reintroduce rejected square-projection cuBLAS admission or qtile variants.
- Treat `top_m=8, attention_head_ids=0` as optional speed-profile only because it changes selector semantics.

Current bottleneck reading:

- Compact attention rank capture is already fused and compact; it is not copying full attention matrices.
- The x=7 graph profile is dominated by `Convolution`, `PicSparseAttention`, `Raster`, `BinaryOp`, `While`, and `UnaryOp`.
- The next engineering pass should audit whether existing CUDA fusions (`PicSiluMul`, `PicPackedSiluMul`, `PicGateUpWeightOnly`) are present in the exported graph and selected by CUDA, before adding new kernels.

Planned next actions:

1. Inspect graph/export/runtime wiring for existing PIC fusion ops.
2. If fusion exists but is not selected, find the dispatch or shape reason.
3. If the graph does not emit fusion ops, consider a narrow export/runtime A/B that uses existing ops rather than adding a new compute operator.
4. Rebuild/sync only after a narrow source change, then run repeat3 decode repair matrix x=0/1/3/5/7 and append the result here.

## PicSiluMul export A/B setup

Audit result:

- The current benchmark model `.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary` has `pic_decode_tiny_fusion=false` and `pic_decode_gateup_fusion=false`.
- `rg -c "PicSiluMul|PicPackedSiluMul|PicGateUpWeightOnly"` over its `llm.mnn.json` returns no hits.
- Therefore the graph profile's per-layer MLP `UnaryOp + BinaryOp` is an export/model issue, not a CUDA dispatch failure.

Created a one-variable A/B model:

```text
dst=.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-silumul
log=.cache/logs/pic-llm-export/20260628_084229_AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-silumul.log
```

Command:

```text
MODELSCOPE_CACHE_ROOT=/root/nfs/l40s_ssd_7t/modelscope/hub/models \
MNN_LLM_EXPORTER=pic \
MNN_ARTIFACT_PLATFORM=x64 \
MNNCONVERT_PATH=.cache/output/mnn/artifacts/x64/bin/MNNConvert \
bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh \
  AI-ModelScope/Llama-3.2-3B-Instruct \
  AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-silumul \
  -- \
  --paged_kv_max_tokens 4096 \
  --pic_recompute_score_layer_idx 1 \
  --pic_decode_tiny_fusion \
  --lm_quant_bit 4 \
  --lm_quant_block 64
```

Checks:

```text
llm_config.json: paged_attention=true, pic_recompute_budget=true, score_layer=1, pic_decode_tiny_fusion=true, pic_decode_gateup_fusion=false
llm.mnn.weight: 1.9G
llm.mnn.json: contains PicSiluMul
```

This A/B does not change decode repair selection semantics and does not add a new operator implementation; it only exposes the existing CUDA `PicSiluMul` fusion in the graph.

### A/B export load failure

The first `pic-boundary-silumul` export was invalid even though it contained `PicSiluMul`:

```text
Jetson load: SIGSEGV in MNN::SizeComputerSuite::search(MNN::OpType)
gdb register w1: 0xffffffff
```

Inspection showed the A/B graph had 27 `type=-1` PagedAttention-family nodes:

```text
name=/layers.1/self_attn/PagedAttention main_type=AttentionParam layer_index=1 outputs=2 type=-1
name=/layers.2/self_attn/PagedAttention main_type=AttentionParam layer_index=2 outputs=1 type=-1
...
```

Root cause:

- `PicSiluMul` was correctly rewritten to `Extra`.
- `PicScoreAttention` / `PicSparseAttention` fell back to `type=-1` after JSON->MNN conversion with the stale x64 converter/lib combination.
- Jetson runtime then crashed because `SizeComputerSuite::search()` has no guard for invalid op type `-1`.

Source fix added in `transformers/pic_llm/export/utils/mnn_converter.py`:

- Rebuild `type=-1/main_type=AttentionParam` nodes as attention ops.
- Infer `PicScoreAttention` from `outputs=2` or `inputs>=5`.
- Infer `PicSparseAttention` for `layer_index > pic_recompute_score_layer_idx` when `pic_recompute_budget` is enabled.
- Add a final post-`removeDupOps` fallback rewrite because MNNConvert can reintroduce `type=-1`.

Next action:

- Rebuild current x64 `MNNConvert/libMNN`.
- Re-run the `pic-boundary-silumul` A/B export/conversion with the matching converter.
- Only then sync to Jetson and benchmark.
