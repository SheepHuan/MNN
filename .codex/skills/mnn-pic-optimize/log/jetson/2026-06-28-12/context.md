# Jetson PIC Decode Repair Optimize Context 2026-06-28 12

## Rows45 cuBLAS Skip Set-Math A/B Result

This continues the `11:58` experiment from `jetson/2026-06-28-11`.

Temporary source hook:

```text
file=source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
env=MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_SKIP_SET_MATH=1
scope=runRows45CublasFp16 helper only
default behavior=no env still calls cublasSetMathMode
```

Build and sync:

```text
build=success
outputs=libMNN.so, libMNN_Express.so, libMNN_Cuda_Main.so, libpic_llm.so, pic_server
artifact=file check aarch64
rsync=.cache/output/mnn/artifacts/jetson/ -> jetson_cross_cuda complete
```

Same-build no-env control:

```text
server_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_skip_set_math_control_20260628_11.log
summary=.cache/mnn-pic-benchmark/decode_repair_skip_set_math_control_20260628_11/summary.csv
x=0 none                  TPOT=87.72547619047619 ms
x=1 lagged_attention_hkvd TPOT=98.7837619047619 ms
x=3 lagged_attention_hkvd TPOT=124.48604761904761 ms
x=5 lagged_attention_hkvd TPOT=126.1115238095238 ms
x=7 lagged_attention_hkvd TPOT=127.492 ms
```

Env run:

```text
env=MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_SKIP_SET_MATH=1
server_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_skip_set_math_env_20260628_11.log
summary=.cache/mnn-pic-benchmark/decode_repair_skip_set_math_env_20260628_11/summary.csv
x=0 none                  TPOT=88.18809523809523 ms
x=1 lagged_attention_hkvd TPOT=98.21619047619049 ms
x=3 lagged_attention_hkvd TPOT=123.80119047619047 ms
x=5 lagged_attention_hkvd TPOT=126.48114285714286 ms
x=7 lagged_attention_hkvd TPOT=126.73428571428572 ms
```

Delta env vs same-build control:

```text
x=0 +0.463 ms
x=1 -0.568 ms
x=3 -0.685 ms
x=5 +0.370 ms
x=7 -0.758 ms
```

Decision:

- Reject and remove `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_SKIP_SET_MATH`.
- The deltas are sub-ms and mixed; the only target improvements are too small to distinguish from run-to-run variance.
- x=0 and x=5 regress slightly, so this is not safe as a default.
- This confirms repeated `cublasSetMathMode` is not a meaningful lever for the current x=3/5/7 TPOT gap.

## Default Artifact Restored After Rejected Skip Set-Math A/B

Cleanup:

```text
source check:
  rg -n "ROWS45_CUBLAS_SKIP_SET_MATH|picRows45CublasSkipSetMathMode" source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
  no matches
  git diff -- source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
  empty
```

Restore build:

```text
command=MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
result=success
outputs=libMNN.so, libMNN_Express.so, libMNN_Cuda_Main.so, libpic_llm.so, pic_server
artifact=file check aarch64
```

Restore sync/server:

```text
rsync .cache/output/mnn/artifacts/jetson/ -> jetson_cross_cuda complete
server=default/no profile/no rejected env
remote_pid=91733
remote_port=18131
local_tunnel=19131
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_default_after_skip_set_math_reject_20260628_11.log
/v1/models=HTTP 200
```

No default speedup landed in this segment.

## Current Optimization State

Best accepted debug range on Jetson:

```text
x=0 none                  TPOT≈86-89 ms
x=1 lagged_attention_hkvd TPOT≈97-99 ms
x=3 lagged_attention_hkvd TPOT≈123-125 ms
x=5 lagged_attention_hkvd TPOT≈125-126 ms
x=7 lagged_attention_hkvd TPOT≈126-128 ms
```

Rejected directions now include:

```text
rows2/3 cuBLAS
qtile-off
qtile hd128_q4k16 / hd128_q4k8 for x<=7
V14_MB OC=8
V14_MB OC=2
rows45 cuBLAS skip cublasSetMathMode
```

Profile-backed next direction:

- x=1 is compact batch=2 transformer compute: V14_MB tiny dense plus PicSparseAttention.
- x=3/5/7 are mostly rows4-8 dense, especially MLP-like rows45 cuBLAS projections, with qtile attention secondary.
- Attention-rank capture is negligible and should not be optimized next.
- Useful next engineering work is rows4-8 MLP dense fusion/tensor-core path or batch=2 dense redesign, not more env retuning around the existing cuBLAS helper.

## Rows45 Policy Cache Micro-Cleanup

Source change:

```text
file=source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
functions=picRows45CublasPolicy, picRows45CublasMathPolicy, picRows45CublasComputePolicy, picRows45CublasAlgoPolicy
change=wrap env parsing in static const int lambda caches
behavior=process-start env semantics remain; runtime setenv-after-first-call is no longer observed for these debug policies
```

Rationale:

- x=3 profile shows hundreds of rows45 cuBLAS dense calls per request.
- The previous `cublasSetMathMode` skip A/B showed that cuBLAS state calls are not the big lever.
- This micro-cleanup removes repeated `getenv/strcmp/atoi` from the hot rows45 helper without changing default cuBLAS math/compute/algo choices.

Build and sync:

```text
command=MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
result=success
artifact=file check aarch64
rsync=.cache/output/mnn/artifacts/jetson/ -> jetson_cross_cuda complete
server_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_rows45_policy_cache_20260628_12.log
remote_pid=91912
/v1/models=HTTP 200
```

Benchmark:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_policy_cache_20260628_12/summary.csv
x=0 none                  TPOT=87.92595238095238 ms
x=1 lagged_attention_hkvd TPOT=98.38819047619047 ms
x=3 lagged_attention_hkvd TPOT=124.05390476190478 ms
x=5 lagged_attention_hkvd TPOT=126.19709523809524 ms
x=7 lagged_attention_hkvd TPOT=126.4972380952381 ms
```

Delta vs same-hour default control:

```text
control=.cache/mnn-pic-benchmark/decode_repair_skip_set_math_control_20260628_11/summary.csv
x=0 +0.200 ms
x=1 -0.396 ms
x=3 -0.432 ms
x=5 +0.086 ms
x=7 -0.995 ms
```

Delta vs accepted rows48 baseline:

```text
baseline=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_smoke2_20260628_0710/summary.csv
x=0 +1.549 ms
x=1 +0.917 ms
x=3 +0.452 ms
x=5 +1.223 ms
x=7 +0.138 ms
```

Decision:

- Keep as a low-risk hot-path cleanup, but do not count it as a meaningful TPOT speedup.
- Same-hour deltas are mixed and sub-ms except x=7 `-0.995 ms`; this is still within the observed Jetson run variance.
- Current remote server is no-profile/no-env policy-cache build:

```text
remote_pid=91912
remote_port=18131
local_tunnel=19131
/v1/models=HTTP 200
```

Next:

- Stop spending time on rows45 env parsing/state-call overhead; the remaining gap is real dense/attention compute.
- The next useful attempt should be a substantive dense path, e.g. rows4-8 gate/up fusion, a dedicated small-N tensor-core GEMM, or a batch=2 V14_MB redesign.
- Separately, `build_artifacts.sh` now repeatedly sees `jetson_cross` cache compiler/sysroot as empty and deletes the build dir, causing full rebuilds. This is not a decode TPOT issue, but fixing it would speed up future A/B iterations.

## Goal Reset For Next Pass

The active goal is narrowed from the broad attachment/log-maintenance wording to a concrete decode-repair optimization objective:

```text
Optimize Jetson CUDA PIC decode repair TPOT for lagged_attention_hkvd at repair token
counts x=1/3/5/7 so that the measured latency approaches the x=0 none decode baseline,
while preserving the compact attention-rank + HKVD selection semantics and keeping
x=0 normal decode speed unchanged.
```

Primary measurements:

```text
baseline input:
  context=1024
  max_tokens=8
  repeats=3
  warm_repeats=1
  attention_layer_idx=1
  top_m=32
  repair_tokens=0,1,3,5,7

current best accepted debug result:
  summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_smoke2_20260628_0710/summary.csv
  x=0 none                  TPOT=86.377 ms
  x=1 lagged_attention_hkvd TPOT=97.471 ms
  x=3 lagged_attention_hkvd TPOT=123.602 ms
  x=5 lagged_attention_hkvd TPOT=124.975 ms
  x=7 lagged_attention_hkvd TPOT=126.360 ms

current policy-cache artifact:
  summary=.cache/mnn-pic-benchmark/decode_repair_rows45_policy_cache_20260628_12/summary.csv
  x=0 none                  TPOT=87.926 ms
  x=1 lagged_attention_hkvd TPOT=98.388 ms
  x=3 lagged_attention_hkvd TPOT=124.054 ms
  x=5 lagged_attention_hkvd TPOT=126.197 ms
  x=7 lagged_attention_hkvd TPOT=126.497 ms
```

Profile-backed bottleneck statement:

```text
x=1:
  Graph profile shows Convolution and PicSparseAttention dominate.
  Attention-rank capture is not the bottleneck.
  Batch=2 compact dense and sparse attention need real implementation work.

x=3/5/7:
  Kernel profile shows rows4 dense dominates.
  rows45 cuBLAS MLP-like projections are the largest remaining compute block.
  qtile sparse attention variants are not useful for x<=7 and have been rejected.
```

Constraints:

```text
Keep x=0 none in every benchmark run.
Do not bypass PagedCache or change lagged_attention_hkvd selection semantics.
Only one specified layer captures attention rank; do not copy full attention weights.
Maintain only compact rank indices on CPU.
Keep hourly logs under .codex/skills/mnn-pic-optimize/log/<device>/<hour>/.
```

Next implementation plan:

1. Inspect and use CUDA direct op benches before larger source changes.
2. Measure isolated rows4-8 MLP projection shapes, especially 3072->8192 and 8192->3072.
3. Prototype one dense-compute change at a time and run x=0/1/3/5/7 end-to-end after each candidate.
4. Keep rejected qtile and rows45 env micro-knobs out of the default path.

## Direct Dense Bench And Rows45 cuBLAS Disable A/B

Motivation:

- Profile showed x=3/5/7 dominated by rows4-8 dense MLP-like projections.
- Existing `test/bench_ops/cuda/CudaWeightOnlyConvPerf.cpp` has direct CUDA op cases:
  - `bench_ops/cuda/perf/PicDecodeMlp`
  - `bench_ops/cuda/perf/DecodeRepairMlpGemmFloor`
  - `bench_ops/cuda/perf/WeightOnlyConv`
- An old `jetson_cross_cuda_test` artifact existed, but profile smoke showed it was stale: rows4/6/8 still printed `op=conv_fpa_intb_1x1`, not `op=conv_fpa_intb_1x1_rows45_cublas`.

Rebuilt current test artifact:

```text
command=BUILD_DIR=$PWD/.cache/build/mnn/jetson_cross_cuda_test INSTALL_PREFIX=$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_test MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 CMAKE_ARGS="-DMNN_BUILD_TEST=ON" BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=48 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
result=success
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda_test/bin/run_test.out
file=aarch64
rsync=jetson_cross_cuda_test complete
```

Current artifact profile smoke:

```text
bench=bench_ops/cuda/perf/WeightOnlyConv
env=MNN_PAGED_ATTENTION_PROFILE=1
shape=hidden_to_inter, rows=2,4,6,8, hidden=3072, inter=8192

rows=2 -> op=conv_fpa_intb_1x1_tiny_gemv
rows=4 -> op=conv_fpa_intb_1x1_rows45_cublas
rows=6 -> op=conv_fpa_intb_1x1_rows45_cublas
rows=8 -> op=conv_fpa_intb_1x1_rows45_cublas
```

Direct MLP current default:

```text
log=remote .cache/bench_ops/jetson_pic_decode_mlp_3072x8192_rows2468_current_*.log
bench=bench_ops/cuda/perf/PicDecodeMlp
env=rows=2,4,6,8 hidden=3072 inter=8192 warmup=10 repeat=40

rows=2 chain=1.5564 ms sum=5.7235 ms
rows=4 chain=2.8007 ms sum=2.7512 ms
rows=6 chain=2.8445 ms sum=2.7866 ms
rows=8 chain=2.8889 ms sum=2.8325 ms
```

Direct MLP with rows45 cuBLAS disabled:

```text
log=remote .cache/bench_ops/jetson_pic_decode_mlp_3072x8192_rows2468_current_cublas0_*.log
env=MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=0

rows=2 chain=1.0858 ms sum=2.1861 ms
rows=4 chain=1.7742 ms sum=1.6459 ms
rows=6 chain=1.7499 ms sum=1.7387 ms
rows=8 chain=1.7461 ms sum=1.6933 ms
```

Direct-op interpretation:

- The direct MLP bench says cublas-off is faster for isolated rows4/6/8 MLP chain.
- It is not sufficient evidence for a default change because decode repair includes the full graph, attention, graph scheduling, and many projection shapes beyond the MLP chain.
- `PicDecodeMlp` chain values can also be noisy for rows=2 because single-op and chain event timings do not always sum cleanly.

End-to-end A/B:

```text
server_env=MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=0
server_pid=93486 during test
remote_log=.cache/logs/pic_server_decode_repair_rows45_cublas0_20260628_12.log
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_cublas0_20260628_12/summary.csv

x=0 none                  TPOT=86.94380952380952 ms
x=1 lagged_attention_hkvd TPOT=97.56566666666667 ms
x=3 lagged_attention_hkvd TPOT=130.53414285714285 ms
x=5 lagged_attention_hkvd TPOT=131.5319523809524 ms
x=7 lagged_attention_hkvd TPOT=133.76857142857142 ms
```

Delta vs current policy-cache default:

```text
default=.cache/mnn-pic-benchmark/decode_repair_rows45_policy_cache_20260628_12/summary.csv
x=0 -0.982 ms
x=1 -0.823 ms
x=3 +6.480 ms
x=5 +5.335 ms
x=7 +7.271 ms
```

Decision:

- Reject `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=0` for default decode repair.
- The direct-op isolated MLP result is not representative of the end-to-end x=3/5/7 region.
- Keep current rows45 cuBLAS default for now.
- Default Jetson server was restored:

```text
server=default/no env
remote_pid=94469
remote_port=18131
local_tunnel=19131
remote_log=.cache/logs/pic_server_decode_repair_default_after_cublas0_reject_20260628_12.log
/v1/models=HTTP 200
```

Next:

- Do not replace rows45 cuBLAS with a blanket off switch.
- If revisiting dense, target a shape-specific path that beats both rows45 cuBLAS and CUTLASS end-to-end, then test x=0/1/3/5/7 before accepting.
- The direct bench artifact is now current and can be reused for subsequent small-N dense experiments.
