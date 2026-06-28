# Jetson PIC Decode Repair Optimize 2026-06-28 12

## 12:10 Rows45 cuBLAS Skip Set-Math A/B Rejected

Continuation from `jetson/2026-06-28-11`: the env-gated `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_SKIP_SET_MATH=1` hook was tested against a same-build no-env control.

Control:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_skip_set_math_control_20260628_11/summary.csv
x=0  87.725 ms
x=1  98.784 ms
x=3 124.486 ms
x=5 126.112 ms
x=7 127.492 ms
```

Env:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_skip_set_math_env_20260628_11/summary.csv
x=0  88.188 ms
x=1  98.216 ms
x=3 123.801 ms
x=5 126.481 ms
x=7 126.734 ms
```

Delta:

```text
x=0 +0.463 ms
x=1 -0.568 ms
x=3 -0.685 ms
x=5 +0.370 ms
x=7 -0.758 ms
```

Decision: reject. The movement is sub-ms and mixed direction, with x=0/x=5 worse and x=3/x=7 only noise-level better. Repeated `cublasSetMathMode` is not a useful default target for this TPOT gap.

## 12:10 Default Restored

Cleanup:

```text
source check: no ROWS45_CUBLAS_SKIP_SET_MATH / picRows45CublasSkipSetMathMode matches
git diff -- ConvFpAIntBExecution.cu: empty
build=success
artifact=aarch64
rsync=jetson_cross_cuda complete
server=default/no profile/no rejected env
remote_pid=91733
/v1/models=HTTP 200
remote_log=.cache/logs/pic_server_decode_repair_default_after_skip_set_math_reject_20260628_11.log
```

Current standing best remains the accepted default-range result:

```text
x=0 none                  TPOT≈86-89 ms
x=1 lagged_attention_hkvd TPOT≈97-99 ms
x=3 lagged_attention_hkvd TPOT≈123-125 ms
x=5 lagged_attention_hkvd TPOT≈125-126 ms
x=7 lagged_attention_hkvd TPOT≈126-128 ms
```

Next useful work should target real compute reduction: rows4-8 MLP dense fusion/tensor-core path for x=3/5/7, and a batch=2 dense redesign for x=1. Attention rank capture and `cublasSetMathMode` are not the active bottlenecks.

## 12:22 Rows45 Policy Cache Micro-Cleanup

Source change:

```text
file=source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
change=cache rows45 cuBLAS env policy parsing in static locals
scope=MNN_CUDA_PIC_INT4_ROWS45_CUBLAS, *_MATH, *_COMPUTE, *_ALGO
```

Benchmark:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_policy_cache_20260628_12/summary.csv
x=0  87.926 ms
x=1  98.388 ms
x=3 124.054 ms
x=5 126.197 ms
x=7 126.497 ms
```

Delta vs same-hour default control:

```text
x=0 +0.200 ms
x=1 -0.396 ms
x=3 -0.432 ms
x=5 +0.086 ms
x=7 -0.995 ms
```

Decision: keep only as a low-risk hot-path cleanup, not as a claimed speedup. It removes repeated `getenv/strcmp/atoi` from rows45 cuBLAS calls and shows no material regression, but the TPOT movement is still run-to-run noise. Current Jetson server is this no-env policy-cache build and `/v1/models` returns HTTP 200.

## 12:30 Goal Reset

Current working goal is narrowed to Jetson CUDA decode repair TPOT optimization:

```text
Primary objective:
  Make lagged_attention_hkvd decode repair for x=1/3/5/7 approach x=0 none
  by reducing real compute and dispatch overhead, without weakening decode repair semantics.

Current accepted debug range:
  x=0 none                  TPOT=86-89 ms
  x=1 lagged_attention_hkvd TPOT=97-99 ms
  x=3 lagged_attention_hkvd TPOT=123-125 ms
  x=5 lagged_attention_hkvd TPOT=125-126 ms
  x=7 lagged_attention_hkvd TPOT=126-128 ms

Near-term target:
  Preserve x=0 normal decode speed.
  Pull x=1 closer to x=0 by improving batch=2 dense/PicSparseAttention cost.
  Pull x=3/5/7 down by reducing rows4-8 dense MLP projection cost.
```

Non-goals for the next optimization pass:

```text
Do not revisit qtile variants for x<=7.
Do not keep tuning rows45 env policy/state-call overhead.
Do not optimize attention-rank capture first; profile shows it is negligible.
Do not use shortcuts that change repair selection semantics or bypass PagedCache.
```

Next engineering focus: prototype and profile a substantive dense-compute path before another end-to-end A/B, using direct CUDA op benches where possible. Candidate directions are rows4-8 gate/up/down fusion, a dedicated small-N tensor-core path, or a redesigned batch=2 V14_MB path.

## 12:42 Rows45 cuBLAS Disable A/B Rejected

Direct-op setup:

```text
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda_test
build=current source, aarch64, run_test.out rebuilt at 12:34
bench=bench_ops/cuda/perf/PicDecodeMlp
shape=rows 2/4/6/8, hidden=3072, inter=8192
```

Direct MLP showed `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=0` faster for isolated rows4/6/8 MLP chain, but the end-to-end decode repair A/B contradicted that single-op result.

End-to-end with server env `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=0`:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_cublas0_20260628_12/summary.csv
x=0  86.944 ms
x=1  97.566 ms
x=3 130.534 ms
x=5 131.532 ms
x=7 133.769 ms
```

Decision: reject. Disabling rows45 cuBLAS preserves x=0/x=1 but regresses the important x=3/5/7 region by roughly 4-8 ms versus the current default policy-cache artifact. Current Jetson server has been restored to the default no-env build and `/v1/models` returns HTTP 200.
