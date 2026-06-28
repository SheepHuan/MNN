# Jetson CUDA decode repair continuation

## Summary

- Continued the decode-repair optimization from the previous `2026-06-28-05` hour.
- Read the required pasted text attachment:
  - `.codex/attachments/8587fef2-a80e-4e6a-b043-cbd327f2e050/pasted-text-1.txt`
- The pasted text correctly warned that old qtile sparse attention was not an obvious default for tiny decode-repair rows and recommended prioritizing fused lagged-attention rank capture.
- Current source/results partially supersede that warning:
  - Fused rank capture is already implemented in the qtile path.
  - `hd128_q8k16` qtile is now proven useful for decode-repair rows `2/4/6/8`.
  - The previous rows2 regression was specific to the older q4/k16 tiny-row experiment, not q8/k16.

## Current Best Jetson Smoke

Default repeat3 after rebuilding and syncing the rows2 q8/k16 decode-repair default:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_q8k16_min2_default_repeat3_20260628_0640/summary.csv
x=0 none                  TPOT=87.641 ms
x=1 lagged_attention_hkvd TPOT=97.924 ms
x=3 lagged_attention_hkvd TPOT=123.946 ms
x=5 lagged_attention_hkvd TPOT=131.463 ms
x=7 lagged_attention_hkvd TPOT=134.923 ms
```

## Current Direction

- Keep the compact attention-ranking design: one configured layer captures top-M PIC local indices, not a full attention matrix.
- Keep decode-repair `headDim=128 && attnLen>=2` on `hd128_q8k16` unless a new profile contradicts it.
- Do not spend the next pass on CPU selector/ranking bookkeeping unless profile shows a regression; previous evidence says it is microsecond scale.
- Next step is focused x=1/x=3 profiling to attribute the remaining gap to qtile attention, compact dense/MLP, activation, graph launch, or hidden synchronization.

## x=1/x=3 Profile Snapshot

Attribution-only run with synchronized CUDA/profile timing:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_x13_profile_20260628_0620/summary.csv
remote_log=.cache/mnn-pic-benchmark/decode_repair_x13_profile_20260628_0620/remote/pic_server.log
x=1 lagged_attention_hkvd TPOT=161.788 ms profile-only
x=3 lagged_attention_hkvd TPOT=262.223 ms profile-only
```

Profile conclusion:

- `forward_raw` dominates decode repair; rank result handling is about `0.003-0.005 ms`.
- `decode_attention_rank` is fused and totals only about `0.2-0.4 ms` per step.
- rows2 qtile attention is about `0.84 ms/layer`; rows4 is about `0.87 ms/layer`.
- rows2 dense still uses `conv_fpa_intb_1x1_tiny_gemv` and costs about `47 ms/step` under synchronized profile.
- rows4 dense uses rows45 cuBLAS plus static CUTLASS and is also material; first-step cublas initialization causes outliers, but steady cublas/conv remains the larger target than rank capture.

Next A/B: test whether rows2 decode repair should bypass tiny GEMV and use the static-dequant CUTLASS/cublas-style path instead.

## Rows2 Dense A/B Result

Temporary env A/B:

```text
MNN_CUDA_PIC_DECODE_REPAIR_INT4_GEMV_BATCH_LIMIT=1
summary=.cache/mnn-pic-benchmark/decode_repair_gemvlimit1_ab_20260628_0635/summary.csv
x=0 none                  TPOT=88.011 ms
x=1 lagged_attention_hkvd TPOT=128.713 ms
x=3 lagged_attention_hkvd TPOT=125.216 ms
```

Decision: reject and remove the env A/B code. rows2 must keep the existing tiny GEMV path; static CUTLASS for batch2 is much slower for `x=1`.

## QTile Small-Row Variant A/B

Env-only A/B:

```text
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=2
MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q4k8
summary=.cache/mnn-pic-benchmark/decode_repair_q4k8_min2_ab_20260628_0645/summary.csv
x=0 none                  TPOT=85.926 ms
x=1 lagged_attention_hkvd TPOT=118.555 ms
x=3 lagged_attention_hkvd TPOT=144.022 ms
```

Decision: reject. `hd128_q8k16` remains the best known decode-repair qtile default for rows2/4/6/8.

## Default Smoke After A/B

No qtile/dense A/B env, repeat3:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_default_smoke_20260628_0650/summary.csv
x=0 none                  TPOT=86.443 ms
x=1 lagged_attention_hkvd TPOT=98.521 ms
x=3 lagged_attention_hkvd TPOT=123.406 ms
x=5 lagged_attention_hkvd TPOT=132.097 ms
x=7 lagged_attention_hkvd TPOT=133.897 ms
```

This confirms the accepted default remains the rows2 `hd128_q8k16` qtile path plus existing tiny GEMV for rows2 dense.

## 06:45 continuation entry

User request: record the log first, then continue optimizing current Jetson CUDA decode repair `x=1/3/5/7` so TPOT moves closer to `x=0 none`.

Current accepted non-profile smoke:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_default_smoke_20260628_0650/summary.csv
x=0 none                  TPOT=86.443 ms
x=1 lagged_attention_hkvd TPOT=98.521 ms
x=3 lagged_attention_hkvd TPOT=123.406 ms
x=5 lagged_attention_hkvd TPOT=132.097 ms
x=7 lagged_attention_hkvd TPOT=133.897 ms
```

Continuation direction:

- Keep compact attention ranking: one configured layer captures top-M PIC local indices, no full attention matrix.
- Keep `hd128_q8k16` qtile for decode-repair active rows `2/4/6/8`.
- Keep rows2 on packed tiny GEMV; the static CUTLASS bypass was rejected.
- Do not spend this pass on rank capture or CPU selector bookkeeping unless new evidence contradicts the existing profile.
- Inspect tiny-row qtile attention, compact-row dense/activation, and launch/graph overhead for a narrow default-safe optimization.

## 06:55 output memset skip A/B rejected

Tested skipping `cudaMemset(output)` for decode/repair causal attention when the kernel should fully overwrite the compact output.

```text
summary=.cache/mnn-pic-benchmark/decode_repair_skip_output_memset_smoke_20260628_0655/summary.csv
x=0 none                  TPOT=87.079 ms
x=1 lagged_attention_hkvd TPOT=98.558 ms
x=3 lagged_attention_hkvd TPOT=123.686 ms
x=5 lagged_attention_hkvd TPOT=132.036 ms
x=7 lagged_attention_hkvd TPOT=134.931 ms
```

Decision: reject and revert. It did not improve `x=1`, slightly slowed `x=0/x=3/x=7`, and only changed `x=5` within noise.
