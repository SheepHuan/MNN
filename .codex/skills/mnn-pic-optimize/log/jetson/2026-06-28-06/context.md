# Jetson CUDA decode repair continuation context

## Required pasted text review

The required attachment was read before continuing:

```text
.codex/attachments/8587fef2-a80e-4e6a-b043-cbd327f2e050/pasted-text-1.txt
```

Its core argument was:

- `x=0` is normal decode and should remain the baseline.
- Decode repair only computes `x + 1` Q rows, so `x=1/3/5/7` maps to rows `2/4/6/8`.
- The old CUDA implementation routed decode repair through row-compressed decode attention and explicitly excluded qtile.
- Existing qtile bench cases were large sparse-prefill active rows, not tiny decode rows.
- Therefore qtile should not be the first default for the old implementation; the higher-value change is to fuse `lagged_attention_hkvd` rank capture into the decode attention QK loop and avoid saving/copying full attention weights.

Current source/results update that assessment:

- The fused-rank part is now already implemented for the qtile path: the configured layer accumulates compact PIC rank scores inside `sparse_flash_qtile_attention`, then only runs topK/D2H/result preparation.
- q8/k16 is materially different from the earlier tiny-row q4/k16 experiment. Server-env A/B and rebuilt default repeat3 showed `hd128_q8k16` improves rows2/4 and remains good for rows6/8.
- The old warning still applies to broad/generic qtile assumptions: this remains a decode-repair-only selector change and should not be generalized to sparse prefill without separate evidence.

## Current evidence

Previous-hour best default repeat3:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_q8k16_min2_default_repeat3_20260628_0640/summary.csv
x=0 none                  TPOT=87.641 ms
x=1 lagged_attention_hkvd TPOT=97.924 ms
x=3 lagged_attention_hkvd TPOT=123.946 ms
x=5 lagged_attention_hkvd TPOT=131.463 ms
x=7 lagged_attention_hkvd TPOT=134.923 ms
```

Improvement from the original user baseline:

```text
x=1: 158.793 -> 97.924 ms
x=3: 185.317 -> 123.946 ms
x=5: 184.994 -> 131.463 ms
x=7: 207.958 -> 134.923 ms
```

Remaining gap to x=0:

```text
x=1: +10.283 ms
x=3: +36.305 ms
x=5: +43.822 ms
x=7: +47.282 ms
```

## Next profile hypothesis

Rank capture and CPU-side selection are unlikely to explain the remaining gap. The next pass should profile:

- qtile attention total for rows2/4.
- `CUDAWeightOnlyConv` compact-row Linear cost for rows2/4.
- activation/BinaryOp/graph launch count if graph profile is available without making formal latency unusable.
- any hidden synchronization around rank D2H/result preparation.

The optimization target is to reduce the non-baseline overhead while preserving:

- `x=0 none` normal decode speed.
- one configured attention layer capture only.
- compact ranking only, not full attention weight storage.

## 2026-06-28 06:25 x=1/x=3 focused profile

Run:

```bash
MNN_PAGED_ATTENTION_PROFILE=1 MNN_PIC_DECODE_REPAIR_PROFILE=1 \
pic_server --config .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json \
  --host 0.0.0.0 --port 18131 \
  --kv-cache-dir .cache/pic_prefill_latency_sweep/shared_kv/llama3_2_3b \
  --model llama-pic

conda run -n kvshare-edge python .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://127.0.0.1:19131 \
  --device jetson --device-display Jetson --backend cuda \
  --model llama-pic --model-name 'Llama3.2 3B' \
  --model-config .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json \
  --contexts 1024 --budgets 0.00 \
  --decode-selectors lagged_attention_hkvd \
  --repair-tokens 1,3 \
  --max-tokens 4 --repeats 1 --warm-repeats 0 \
  --output-csv .cache/mnn-pic-benchmark/decode_repair_x13_profile_20260628_0620/summary.csv \
  --output-dir .cache/mnn-pic-benchmark/decode_repair_x13_profile_20260628_0620/client
```

Result, profile-only:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_x13_profile_20260628_0620/summary.csv
remote_log=.cache/mnn-pic-benchmark/decode_repair_x13_profile_20260628_0620/remote/pic_server.log
x=1 lagged_attention_hkvd TPOT=161.788 ms
x=3 lagged_attention_hkvd TPOT=262.223 ms
```

`MNN_PIC_DECODE_REPAIR_PROFILE`:

```text
x=1 steady steps:
forward_raw_ms ~= 102.590, 104.334, 106.308
rank_result_ms ~= 0.003

x=3 steady steps:
forward_raw_ms ~= 124.383, 133.979, 134.626
rank_result_ms ~= 0.003-0.004
```

CUDA attention aggregation:

```text
sparse_flash_qtile query=2: n=111 sum=93.895 ms mean=845.9 us median=839.0 us
sparse_flash_qtile query=4: n=111 sum=96.394 ms mean=868.4 us median=864.0 us
decode_attention_rank:      n=8   sum=2.053 ms  mean=256.6 us median=224.0 us
```

Weight-only dense aggregation:

```text
tiny_gemv batch=2:       n=781 sum=189.748 ms mean=243.0 us median=188.0 us
rows45_cublas batch=4:   n=554 sum=327.336 ms mean=590.9 us median=565.0 us
static CUTLASS batch=4:  n=223 sum=61.175 ms  mean=274.3 us median=267.0 us
```

Dimension breakdown:

```text
batch=2 tiny_gemv:
3072->1024 n=223 sum=22.436 ms median=99 us
3072->3072 n=223 sum=40.862 ms median=179 us
3072->8192 n=224 sum=85.444 ms median=378 us
8192->3072 n=111 sum=41.006 ms median=362 us

batch=4 rows45_cublas:
3072->1024 n=223 median=151 us, with first-step outlier up to 68.6 ms
3072->8192 n=220 median=575 us
8192->3072 n=111 median=632 us, with first-step outlier up to 25.7 ms

batch=4 static CUTLASS:
3072->3072 n=223 sum=61.175 ms median=267 us
```

Interpretation:

- The compact attention rank design is no longer a meaningful target.
- rows2/4 qtile attention is a fixed per-layer cost around `0.84-0.87 ms` in synchronized profile.
- rows2 dense is still dominated by packed tiny GEMV; this is now the most promising x=1 A/B.
- rows4 dense has useful rows45 cublas but still contains enough dense cost that additional attention-only changes are unlikely to close the gap to x=0.

Next action:

- Add or temporarily test a narrow decode-repair rows2 dense policy that bypasses `tiny_gemv` for `batch=2` and lets the static-dequant CUTLASS path handle it.
- Keep the default only if repeat3 non-profile `x=0/1/3/5/7` improves or stays neutral.

## 2026-06-28 06:35 rows2 dense A/B rejected

Temporary source A/B:

```text
MNN_CUDA_PIC_DECODE_REPAIR_INT4_GEMV_BATCH_LIMIT=1
```

This forced decode-repair batch2 rows to bypass `conv_fpa_intb_1x1_tiny_gemv` and fall through to the static-dequant CUTLASS path. Non-profile repeat3:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_gemvlimit1_ab_20260628_0635/summary.csv
x=0 none                  TPOT=88.011 ms
x=1 lagged_attention_hkvd TPOT=128.713 ms
x=3 lagged_attention_hkvd TPOT=125.216 ms
```

Comparison to current default:

```text
x=0: 87.641 -> 88.011 ms
x=1: 97.924 -> 128.713 ms
x=3: 123.946 -> 125.216 ms
```

Conclusion:

- rows2 decode repair must keep the existing packed tiny GEMV path.
- The temporary env A/B source change was removed after the test.
- This also weakens the case for batch2 cublas: profile medians already suggested cublas/CUTLASS-style dense would not beat tiny GEMV at rows2, and the end-to-end A/B confirmed it.

Next direction:

- Do not change the rows2 dense route unless adding a genuinely new tiny-row tensor-core/weight-only kernel.
- The remaining low-risk work is to reduce qtile attention overhead or graph launch/activation overhead without disrupting `x=0`.

## 2026-06-28 06:45 q4/k8 small-row qtile A/B rejected

Env-only A/B:

```text
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=2
MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q4k8
```

Non-profile repeat3:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_q4k8_min2_ab_20260628_0645/summary.csv
x=0 none                  TPOT=85.926 ms
x=1 lagged_attention_hkvd TPOT=118.555 ms
x=3 lagged_attention_hkvd TPOT=144.022 ms
```

Comparison to q8/k16 rows2 default:

```text
x=1: 97.924 -> 118.555 ms
x=3: 123.946 -> 144.022 ms
```

Conclusion:

- `hd128_q4k8` is not a small-row rescue variant.
- `hd128_q8k16` remains the best known decode-repair qtile default for rows2/4/6/8.
- Additional qtile work should be a new tiny-row kernel or launch/graph reduction, not another existing variant flip.

## 2026-06-28 06:50 default full smoke

After the rejected dense and qtile A/B runs, restarted the server without qtile/dense override env and ran repeat3 over all requested repair sizes:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_default_smoke_20260628_0650/summary.csv
x=0 none                  TPOT=86.443 ms
x=1 lagged_attention_hkvd TPOT=98.521 ms
x=3 lagged_attention_hkvd TPOT=123.406 ms
x=5 lagged_attention_hkvd TPOT=132.097 ms
x=7 lagged_attention_hkvd TPOT=133.897 ms
```

Comparison to previous default smoke:

```text
x=0: 87.641 -> 86.443 ms
x=1: 97.924 -> 98.521 ms
x=3: 123.946 -> 123.406 ms
x=5: 131.463 -> 132.097 ms
x=7: 134.923 -> 133.897 ms
```

Interpretation:

- No material regression from the A/B cycle.
- Current accepted default is still:
  - `hd128_q8k16` qtile for decode-repair active rows `>=2`.
  - existing packed tiny GEMV for rows2 dense.
  - rows45 cublas/static dense for rows4 where applicable.
- The remaining gap is likely inherent extra repair-row compute plus per-layer launch/graph overhead. Existing variant flips do not close it.

## 2026-06-28 06:45 continuation after log request

User request: log first, then continue optimizing the current `x=1/3/5/7` implementation.

Active target:

```text
x=0 none                  TPOT=86.443 ms
x=1 lagged_attention_hkvd TPOT=98.521 ms
x=3 lagged_attention_hkvd TPOT=123.406 ms
x=5 lagged_attention_hkvd TPOT=132.097 ms
x=7 lagged_attention_hkvd TPOT=133.897 ms
```

Relevant constraints before new edits:

- `x=0` normal decode remains the baseline and must not regress.
- `lagged_attention_hkvd` continues to use compact ranking only:
  - `lastAttentionRankedPicLocalIndices` style compact top-M local indices from one configured attention layer.
  - HKVD/delta-V global ranking remains separate.
  - no complete attention matrix is copied or stored.
- Existing profile says rank capture/result handling is already small:
  - `rank_result_ms ~= 0.003-0.005`.
  - fused `decode_attention_rank` topK/D2H/prep about `0.2-0.4 ms/step`.
- Rejected A/Bs:
  - rows2 tiny GEMV bypass to static CUTLASS: `x=1` regressed to `128.713 ms`.
  - qtile `hd128_q4k8`: `x=1/x=3` regressed to `118.555/144.022 ms`.

Next engineering pass:

1. Inspect current `PagedAttentionExecution.cu` q8/k16 decode-repair path for avoidable per-layer work specific to rows2/4/6/8.
2. Inspect current `ConvFpAIntBExecution.cu` small-row dense policy only for default-safe launch or initialization overhead; do not remove rows2 tiny GEMV.
3. Prefer a narrow optimization that can be validated by repeat3 `x=0,1,3,5,7` on Jetson before becoming the accepted default.

## 2026-06-28 06:55 output memset skip A/B rejected

Change tested:

- In `source/backend/cuda/execution/PagedAttentionExecution.cu`, temporarily skipped `cudaMemset(output)` for `ordinaryDecodeCausal` and `repairDecodeCausal` when `output->elementSize()` exactly matched `batch * attnLen * numHeads * headDim`.
- Rationale: these decode kernels write all compact output rows, and the memset is outside the attention profile timer, so it was a plausible launch/non-compute overhead target.

Build and sync:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/jetson/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

Non-profile repeat3:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_skip_output_memset_smoke_20260628_0655/summary.csv
x=0 none                  TPOT=87.079 ms
x=1 lagged_attention_hkvd TPOT=98.558 ms
x=3 lagged_attention_hkvd TPOT=123.686 ms
x=5 lagged_attention_hkvd TPOT=132.036 ms
x=7 lagged_attention_hkvd TPOT=134.931 ms
```

Comparison to accepted default smoke:

```text
x=0: 86.443 -> 87.079 ms
x=1: 98.521 -> 98.558 ms
x=3: 123.406 -> 123.686 ms
x=5: 132.097 -> 132.036 ms
x=7: 133.897 -> 134.931 ms
```

Conclusion:

- Reject. The change is neutral to slightly negative and does not close the decode-repair gap.
- The source change was reverted after the A/B.
- This suggests output clearing is not a meaningful non-compute overhead at the current compact output sizes.
