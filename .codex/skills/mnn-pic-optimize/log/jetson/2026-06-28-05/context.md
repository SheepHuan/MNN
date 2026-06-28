# Jetson CUDA decode repair 05:00 context

## Goal

Reduce Jetson CUDA decode repair TPOT for:

```text
x=0 none
x=1/3/5/7 lagged_attention_hkvd
```

The target is to move `x=1/3/5/7` closer to `x=0` while keeping the compact attention-ranking design: only compact top-M PIC local indices are captured at one configured attention layer; full attention matrices are not copied or stored.

## Static Dequant Hot-Cache Change

Earlier 04:00 profile showed the 4.5 GiB cap improved MLP static cache use but still left rows4/6/8 attention projection Linears on runtime dequant:

```text
rows4/6/8 runtime-dequant counts ~= 550/request
runtime dequant time ~= 149-152 ms/profile request
static_cache_total ~= 4.743 GB
```

The next code change changed CUDA static dequant admission in:

```text
source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
```

Policy:

- High-memory devices (`totalGlobalMem >= 30 GiB`) get a decode-hot cap of `min(6 GiB, totalGlobalMem / 5)`.
- Low-memory/default behavior still uses the previous `min(4 GiB, totalGlobalMem / 8)` high path for decode-hot Linears and the base cap for non-hot Linears.
- Decode-hot Linears include:
  - MLP-like large Linears.
  - Attention projection-like Linears with `6 MiB <= dequantBytes <= 64 MiB`, `minDim >= 512`, `maxDim <= 4096`.
- Very large non-hot Linears, especially vocab/lm_head, remain on the base cap.

Non-profile A/B after this change:

```text
4.5 GiB cap:
x=0 88.770 ms
x=1 102.391 ms
x=3 163.131 ms
x=5 173.214 ms
x=7 195.221 ms

decode-hot 6 GiB cap:
x=0 87.358 ms
x=1 103.026 ms
x=3 145.990 ms
x=5 156.717 ms
x=7 177.508 ms
```

Graph profile after decode-hot cap confirmed:

```text
max_static_cache_total=5637144576 bytes
rows4 runtime_dequant=0 static_dequant=1559
rows6 runtime_dequant=0 static_dequant=1556
rows8 runtime_dequant=0 static_dequant=1554
```

Measured profile type totals for the measured x=3/5/7 requests:

```text
x=3 total=1368.998 ms Convolution=725.850 PicSparseAttention=324.020 Raster=131.685
x=5 total=1468.297 ms Convolution=733.685 PicSparseAttention=397.609 Raster=134.161
x=7 total=1629.108 ms Convolution=741.139 PicSparseAttention=544.162 Raster=133.689
```

This confirms runtime dequant was removed, and the remaining scaling gap moved to sparse attention plus compact dense/graph overhead.

## QTile A/B

Explicit qtile server env:

```text
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=2
MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q4k16
```

Non-profile result:

```text
x=0 87.763 ms
x=1 117.147 ms
x=3 148.674 ms
x=5 148.727 ms
x=7 155.003 ms
```

Conclusion:

- qtile hurts low active-row decode repair (`x=1`, sparse rows=2) and slightly hurts `x=3` (sparse rows=4).
- qtile helps `x=5` (sparse rows=6) and strongly helps `x=7` (sparse rows=8).

Implemented default adaptive qtile in:

```text
source/backend/cuda/execution/PagedAttentionExecution.cu
```

Policy:

- Without env overrides, `headDim=128 && attnLen>=6` selects `hd128_q4k16`.
- `headDim=64 && attnLen>=16` can select the HD64 qtile variant.
- Lower active-row counts remain on the existing row-compressed decode path.
- Env overrides still exist for explicit experiments.

Final non-profile smoke with no qtile env:

```text
x=0 none                  TPOT=87.887 ms  ok
x=1 lagged_attention_hkvd TPOT=102.506 ms ok
x=3 lagged_attention_hkvd TPOT=145.501 ms ok
x=5 lagged_attention_hkvd TPOT=150.995 ms ok
x=7 lagged_attention_hkvd TPOT=154.879 ms ok
```

Comparison:

```text
x,orig,cap45,hot6,qtile-q4k16,adaptive
0,87.163,88.770,87.358,87.763,87.887
1,158.793,102.391,103.026,117.147,102.506
3,185.317,163.131,145.990,148.674,145.501
5,184.994,173.214,156.717,148.727,150.995
7,207.958,195.221,177.508,155.003,154.879
```

## Artifacts And Commands

Build command:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Artifact:

```text
local=.cache/output/mnn/artifacts/jetson/
remote=/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

Final summary:

```text
.cache/mnn-pic-benchmark/decode_repair_static_hot6_adaptive_qtile_cuda_ab_20260628_0518/summary.csv
```

Notes:

- All valid CUDA runs used `.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json`.
- Do not merge these smoke/profile results into `benchmark_decode.csv` yet; formal reporting still needs the full Jetson + OrangePi decode matrix.
- The build script still incorrectly treats the Jetson cross build cache as stale (`cached: <empty>` for compiler/sysroot) and repeatedly wipes `.cache/build/mnn/jetson_cross`. This is a build-time overhead issue, not a runtime blocker.

## Next Work

- Run a higher-repeat Jetson smoke or formal Jetson decode matrix to reduce one-run noise.
- Run the OrangePi side before treating the result as formal.
- Continue reducing the remaining gap by profiling `PicSparseAttention` for active rows 4/6/8 and compact-row dense/activation launch overhead.
- Avoid default qtile for `attnLen=2/4` unless a future kernel variant proves it does not regress `x=1/3`.

## 2026-06-28 05:xx Continuation

User request: record the optimization log first, then continue optimizing the current `x=1/3/5/7` decode repair implementation so TPOT moves as close as possible to `x=0 none`.

Current baseline to improve:

```text
x=0 none                  TPOT=87.887 ms  ok
x=1 lagged_attention_hkvd TPOT=102.506 ms ok
x=3 lagged_attention_hkvd TPOT=145.501 ms ok
x=5 lagged_attention_hkvd TPOT=150.995 ms ok
x=7 lagged_attention_hkvd TPOT=154.879 ms ok
```

Working hypothesis for this continuation:

- qtile is already conditionally useful and should remain adaptive: disabled for active rows 2/4, enabled for rows 6/8 on `headDim=128`.
- The next useful reduction is likely from removing residual decode-repair overhead:
  - any extra CPU/GPU synchronization or host copy around compact attention rank capture;
  - selector work that can be deferred, cached, or bounded more tightly;
  - `PicSparseAttention` kernel cost for active rows 4/6/8;
  - compact-row dense/activation launch overhead that remains after runtime dequant was removed.
- The design constraint remains unchanged: capture only compact top-M PIC local indices at one configured attention layer, maintain compact rankings, and do not store/copy full attention weights.

## 2026-06-28 05:20 Rows4/5 cuBLAS Generic MLP Admission

Change under test:

```text
source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
```

`picRows45CublasMatches()` was relaxed from a Llama-1B-specific `2048 <-> 8192` check to MLP-like dimensions:

```text
minDim >= 1024
maxDim >= 2 * minDim
maxDim <= 8 * minDim
down:    ic > oc
gate/up: oc > ic
```

Reason: Llama3.2-3B uses `3072 <-> 8192`, so rows4/5 compact MLP shapes were missing the existing rows45 cuBLAS branch.

Build:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Build succeeded for `MNN_Cuda_Main`, `libMNN.so`, `libpic_llm.so`, and `pic_server`; artifact was checked as aarch64 and synced from:

```text
local=.cache/output/mnn/artifacts/jetson/
remote=/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

Non-profile smoke, context 1024, `max_tokens=8`, `attention_layer_idx=1`, `top_m=32`, repeat1:

```text
x=0 none                  TPOT=87.841 ms
x=1 lagged_attention_hkvd TPOT=103.753 ms
x=3 lagged_attention_hkvd TPOT=141.230 ms
x=5 lagged_attention_hkvd TPOT=151.411 ms
x=7 lagged_attention_hkvd TPOT=154.381 ms
```

Repeat3 summary:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_cublas_repeat3_20260628_0520/summary.csv
x=0 none                  TPOT=87.233 ms
x=1 lagged_attention_hkvd TPOT=102.155 ms
x=3 lagged_attention_hkvd TPOT=138.823 ms
x=5 lagged_attention_hkvd TPOT=149.857 ms
x=7 lagged_attention_hkvd TPOT=152.396 ms
```

Comparison against adaptive qtile:

```text
x,adaptive,rows45_generic
0,87.887,87.233
1,102.506,102.155
3,145.501,138.823
5,150.995,149.857
7,154.879,152.396
```

Conclusion: keep the generic MLP rows4/5 cuBLAS change. It clearly improves rows4 (`x=3`) and is neutral/slightly positive elsewhere.

Profile smoke with `MNN_PAGED_ATTENTION_PROFILE=1 MNN_PIC_DECODE_REPAIR_PROFILE=1` used `x=5/7`, `max_tokens=4`, so it is attribution-only rather than formal latency:

```text
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/pic_prefill_latency_sweep/decode_repair_rows45_profile_20260628_0525/pic_server.log
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_profile_20260628_0525/summary.csv
x=5 TPOT=234.453 ms
x=7 TPOT=264.603 ms
```

Profile observations:

- Fused decode attention rank capture is about `206-255 us`; not a bottleneck.
- Layer 1 `decode_causal_attention` ranges roughly `974-2881 us`.
- Later layers `sparse_flash_qtile_attention` for rows6/8 are about `1.4-1.6 ms` per layer.
- Compact CUTLASS dense at batch=8 is still material:
  - `3072->8192`: about `650-706 us`
  - `8192->3072`: about `597-678 us`
  - `3072->3072`: about `260-319 us`
  - `3072->1024`: about `122-156 us`

Next optimization target after this log entry:

- Verify rows4/5 cuBLAS fires in an `x=3` profile if needed.
- For `x=5/7`, focus on qtile variant A/B and compact dense/launch overhead; CPU selector and rank result handling are already down near microseconds.

## 2026-06-28 05:35 Capture-Layer QTile And Rank Split

The first qtile-rank experiment stopped excluding the configured attention-rank capture layer from qtile. It prepared the compact-rank workspace before qtile attention, then still finished rank capture as a separate post-step.

Repeat3 result:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_rank_split_repeat3_20260628_0535/summary.csv
x=0 none                  TPOT=85.947 ms
x=1 lagged_attention_hkvd TPOT=101.413 ms
x=3 lagged_attention_hkvd TPOT=138.181 ms
x=5 lagged_attention_hkvd TPOT=150.148 ms
x=7 lagged_attention_hkvd TPOT=150.359 ms
```

Interpretation: letting the capture layer use qtile is safe, but leaving rank scoring as a separate compact QK pass does not materially improve rows6/8.

## 2026-06-28 05:55 Fused QTile Rank

Change under test:

```text
source/backend/cuda/execution/PagedAttentionExecution.cu
```

The qtile kernel now accepts optional decode-rank capture arguments and atomically accumulates compact PIC-token scores for the last decode-repair row. When the fused path is valid, `finishDecodeAttentionRankCaptureCUDA()` only performs topK/result copy rather than launching the independent QK score kernel. The old `runDecodeAttentionRankCUDA()` path remains as fallback.

Repeat3 with the old default qtile variant still at `hd128_q4k16`:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_fused_rank_repeat3_20260628_0555/summary.csv
x=0 none                  TPOT=87.211 ms
x=1 lagged_attention_hkvd TPOT=103.046 ms
x=3 lagged_attention_hkvd TPOT=137.907 ms
x=5 lagged_attention_hkvd TPOT=148.777 ms
x=7 lagged_attention_hkvd TPOT=151.457 ms
```

Interpretation: fused rank removes a small independent kernel and keeps semantics unchanged, but rows6/8 still need a better qtile shape.

## 2026-06-28 06:00 QTile Variant A/B

Rows6/8 qtile variant A/B:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_q8k16_repeat3_20260628_0600/summary.csv
hd128_q8k16:
x=5 lagged_attention_hkvd TPOT=133.803 ms
x=7 lagged_attention_hkvd TPOT=135.078 ms

summary=.cache/mnn-pic-benchmark/decode_repair_qtile_q4k8_repeat3_20260628_0603/summary.csv
hd128_q4k8:
x=5 lagged_attention_hkvd TPOT=150.307 ms
x=7 lagged_attention_hkvd TPOT=155.156 ms
```

Conclusion: default `headDim=128 && attnLen>=6` should use `hd128_q8k16`, not `hd128_q4k16` or `hd128_q4k8`. Rows2/4 remain on the existing row-compressed path because prior qtile-on-small-rows regressed `x=1` and did not help `x=3`.

## 2026-06-28 06:10 q8/k16 Default Smoke

After switching the default qtile variant to `hd128_q8k16` for rows6/8, repeat3 produced the current best default:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_q8k16_default_repeat3_20260628_0610/summary.csv
x=0 none                  TPOT=88.367 ms
x=1 lagged_attention_hkvd TPOT=102.390 ms
x=3 lagged_attention_hkvd TPOT=138.141 ms
x=5 lagged_attention_hkvd TPOT=131.526 ms
x=7 lagged_attention_hkvd TPOT=133.656 ms
```

Comparison with the user-provided starting point:

```text
x,starting,current_best
0,87.163,88.367
1,158.793,102.390
3,185.317,138.141
5,184.994,131.526
7,207.958,133.656
```

Comparison with rows45 generic cuBLAS before q8/k16 default:

```text
x,rows45_generic,q8k16_default
0,87.233,88.367
1,102.155,102.390
3,138.823,138.141
5,149.857,131.526
7,152.396,133.656
```

This is a real improvement for rows6/8 and effectively neutral for rows2/4 within repeat3 noise.

## 2026-06-28 06:15 q8/k16 Profile Confirmation

Profile run:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_q8k16_default_profile_20260628_0615/summary.csv
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/pic_prefill_latency_sweep/decode_repair_q8k16_default_profile_20260628_0615/pic_server.log
x=5 lagged_attention_hkvd TPOT=213.263 ms
```

This is synchronized profile, not formal latency. Relevant profile facts:

- `op=sparse_flash_qtile_attention` reports `variant=hd128_q8k16`.
- `op=decode_attention_rank` reports `fused_score_in_attention=1`.
- After fusion, rank capture examples are around `us=205 prep_us=21 topk_us=127 d2h_us=37` to `us=257`, so the separate QK scan is gone.
- Later qtile attention layers for rows6 are about `0.86-0.95 ms`; layer1 is about `0.91-1.0 ms`.

Current conclusion:

- `x=5/7` are now much closer to `x=0`; remaining work there is mostly compact dense/graph overhead and the residual qtile attention cost.
- `x=1` and `x=3` remain about `102 ms` and `138 ms` because rows2/4 intentionally stay on the row-compressed path. Prior qtile-on-rows2 regressed `x=1`, so the next optimization for these must be a tiny-row-specific path rather than enabling the existing qtile.
- CPU selector and ranking bookkeeping are no longer meaningful targets; measured rank result handling is microsecond scale, and fused GPU rank topK/D2H is only a few tenths of a millisecond in profile.

## 2026-06-28 06:25 q8/k16 Rows4 And Rows2 A/B

The previous negative qtile experiment used q4/k16 for tiny rows and regressed `x=1`. Since q8/k16 was much faster for rows6/8, the next A/B tested q8/k16 on smaller decode-repair active row counts using only server environment overrides.

Rows4 A/B:

```text
server_env:
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=4
MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q8k16

summary=.cache/mnn-pic-benchmark/decode_repair_q8k16_min4_ab_repeat3_20260628_0625/summary.csv
x=0 none                  TPOT=86.927 ms
x=1 lagged_attention_hkvd TPOT=100.439 ms
x=3 lagged_attention_hkvd TPOT=124.263 ms
x=5 lagged_attention_hkvd TPOT=132.109 ms
x=7 lagged_attention_hkvd TPOT=133.327 ms
```

Rows2 A/B:

```text
server_env:
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=2
MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q8k16

summary=.cache/mnn-pic-benchmark/decode_repair_q8k16_min2_ab_repeat3_20260628_0630/summary.csv
x=0 none                  TPOT=85.839 ms
x=1 lagged_attention_hkvd TPOT=97.292 ms
x=3 lagged_attention_hkvd TPOT=123.065 ms

summary=.cache/mnn-pic-benchmark/decode_repair_q8k16_min2_ab_repeat3_20260628_0630_b/summary.csv
x=5 lagged_attention_hkvd TPOT=131.307 ms
x=7 lagged_attention_hkvd TPOT=133.656 ms
```

Conclusion: q8/k16 is safe and useful even for rows2/4 in the narrow decode-repair path. The earlier rows2 regression was specific to q4/k16/tiny qtile behavior, not qtile in general.

## 2026-06-28 06:40 q8/k16 Rows2 Default

Source change:

```text
source/backend/cuda/execution/PagedAttentionExecution.cu
selectCudaDecodeRepairQTileVariant():
  headDim == 128 && attnLen >= 2 -> hd128_q8k16
```

This is decode-repair-only and does not change `selectCudaSparseQTileVariant()` for generic sparse prefill.

Build:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Build succeeded for `MNN_Cuda_Main`, `libMNN.so`, `libpic_llm.so`, and `pic_server`; `file` confirmed the installed artifact is aarch64 and it was synced to:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

Default no-env repeat3:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_q8k16_min2_default_repeat3_20260628_0640/summary.csv
x=0 none                  TPOT=87.641 ms
x=1 lagged_attention_hkvd TPOT=97.924 ms
x=3 lagged_attention_hkvd TPOT=123.946 ms
x=5 lagged_attention_hkvd TPOT=131.463 ms
x=7 lagged_attention_hkvd TPOT=134.923 ms
```

Comparison with previous q8/k16 threshold rows6 default:

```text
x,previous_default,new_default
0,88.367,87.641
1,102.390,97.924
3,138.141,123.946
5,131.526,131.463
7,133.656,134.923
```

Comparison with the original user baseline:

```text
x,starting,new_default
0,87.163,87.641
1,158.793,97.924
3,185.317,123.946
5,184.994,131.463
7,207.958,134.923
```

Decision: keep the rows2 q8/k16 decode-repair default. Remaining gap to `x=0` is now about `10 ms` for `x=1`, `36 ms` for `x=3`, and `44-47 ms` for `x=5/7`; next work should focus on compact dense/MLP launch count or a proven fused MLP path, not attention-rank CPU bookkeeping.
