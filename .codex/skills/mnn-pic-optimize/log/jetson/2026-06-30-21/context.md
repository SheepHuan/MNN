# Context

## Objective

Continue optimizing Jetson CUDA PIC decode-repair compact MLP after packed gate/up landed.

Accepted packed gate/up endpoint baseline:

| repair x | TPOT ms |
|---------:|--------:|
| 0 | 71.367 |
| 1 | 80.065 |
| 3 | 103.547 |
| 5 | 105.557 |
| 7 | 106.806 |

Remaining low-level packed MLP profile:

| repair x | rows | gate/up ms | down ms | MLP ms |
|---------:|-----:|-----------:|--------:|-------:|
| 0 | 1 | 17.742 | 8.634 | 26.376 |
| 1 | 2 | 19.601 | 9.780 | 29.381 |
| 3 | 4 | 29.157 | 17.083 | 46.240 |
| 5 | 6 | 26.495 | 15.490 | 41.985 |
| 7 | 8 | 27.787 | 15.653 | 43.440 |

Target shape for this hour:

```text
down_proj: rows=4/6/8, ic=8192, oc=3072, INT4 weight-only, fp16, static dequant
```

## Diagnostic Attempt

The tested source branch added an env-gated diagnostic path:

```text
MNN_CUDA_PIC_INT4_DOWN_SMALL_CUTLASS=1
```

Guard:

```text
pic_decode_repair_sparse_active
fp16
INT4
static dequant available
1x1 Linear
rows in {4,6,8}
ic=8192
oc=3072
```

It forced the existing SM70 compact CUTLASS N-wide tile (`64x128x64`) for down-proj and bypassed the rows45 cuBLAS helper only for that forced path.

This branch was only diagnostic. After the direct-op loss below, it was reverted from source.

## Build

Local build command:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON' \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda_down_small_test" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_down_small_test" \
JOBS=96 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Artifact:

```text
local:  .cache/output/mnn/artifacts/jetson_cross_cuda_down_small_test/
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_down_small_test/
```

Installed files:

```text
bin/run_test.out
lib/libMNN.so
lib/libMNN_Cuda_Main.so
lib/libMNN_Express.so
```

Remote benchmark logs:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_small_cutlass_20260630_2105/accuracy_rows45.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_small_cutlass_20260630_2105/perf_inter_to_hidden_default_r1.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_small_cutlass_20260630_2105/perf_inter_to_hidden_forced_r1.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_small_cutlass_20260630_2105/profile_default_dispatch.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_small_cutlass_20260630_2105/profile_forced_dispatch.log
```

## Accuracy

Remote command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && LOG=.cache/bench_ops/down_small_cutlass_20260630_2105 && ART=.cache/output/mnn/artifacts/jetson_cross_cuda_down_small_test && mkdir -p "$LOG" && export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && "$ART/bin/run_test.out" bench_ops/cuda/accuracy/Rows45Cublas 2 2 1 x 2 2>&1 | tee "$LOG/accuracy_rows45.log"'
```

Result:

```text
rows4/5 existing cuBLAS cases: bad=0
llama32_3b rows4/6/8 gate/up/down: bad=0
PicLinearNhwcWeightOnly rows1/2/4/6/8 gate/down: bad=0
all <bench_ops/cuda/accuracy/Rows45Cublas> tests passed
```

## Direct-Op Perf

Common settings:

```text
backend=2 CUDA
precision=2 fp16
memory=2 low
case=inter_to_hidden
rows=1,2,4,6,8
hidden=3072
inter=8192
kv=1024
warmup=30
repeat=120
```

Default command used no diagnostic env. Forced command added:

```text
MNN_CUDA_PIC_INT4_DOWN_SMALL_CUTLASS=1
```

Results:

| rows | default ms | forced ms | delta |
|-----:|-----------:|----------:|------:|
| 1 | 0.2563 | 0.2635 | +0.0072 |
| 2 | 0.3110 | 0.3074 | -0.0036 |
| 4 | 0.5091 | 0.5760 | +0.0669 |
| 6 | 0.5100 | 0.5771 | +0.0671 |
| 8 | 0.5109 | 0.5793 | +0.0684 |

The target rows4/6/8 all regress. This fails the direct-op gate, so endpoint TPOT was intentionally skipped.

## Dispatch Profile

One-shot profile confirmed the default path:

```text
CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1_rows45_cublas batch=4 ic=8192 oc=3072 policy=3 math=1 compute=0 algo=99
CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1_rows45_cublas batch=6 ic=8192 oc=3072 policy=3 math=1 compute=0 algo=99
CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1_rows45_cublas batch=8 ic=8192 oc=3072 policy=3 math=1 compute=0 algo=99
```

Forced diagnostic path:

```text
CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1 batch=4 ic=8192 oc=3072 static_dequant=1 pic_compact_sm70=1 pic_compact_tile=3
CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1 batch=6 ic=8192 oc=3072 static_dequant=1 pic_compact_sm70=1 pic_compact_tile=3
CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1 batch=8 ic=8192 oc=3072 static_dequant=1 pic_compact_sm70=1 pic_compact_tile=3
```

## Conclusion

The existing rows45 cuBLAS helper remains the better down-proj path for `M=4/6/8, K=8192, N=3072`. Reusing the already-available SM70 compact CUTLASS N-wide tile is not a viable P1 down optimization.

Next useful implementation work should move to one of:

1. a new compact dense kernel or cublasLt policy that beats rows45 cuBLAS for all rows4/6/8 in direct-op first;
2. full MLP fusion that streams packed gate/up activation tiles into down accumulation, reducing intermediate activation write/read and launch overhead.

Do not promote `MNN_CUDA_PIC_INT4_DOWN_SMALL_CUTLASS`; the branch has been removed.

## cuBLASLt Heuristic Sweep Follow-Up

After rejecting forced CUTLASS, the bench-only `DecodeRepairMlpGemmLtFloor` path was extended to sweep all cuBLASLt heuristic results returned for the same shape before touching production routing.

Local build command:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON' \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda_lt_sweep_test" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_lt_sweep_test" \
JOBS=96 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Artifact checks:

```text
.cache/output/mnn/artifacts/jetson_cross_cuda_lt_sweep_test/bin/run_test.out:        ELF 64-bit LSB executable, ARM aarch64
.cache/output/mnn/artifacts/jetson_cross_cuda_lt_sweep_test/lib/libMNN.so:           ELF 64-bit LSB shared object, ARM aarch64
.cache/output/mnn/artifacts/jetson_cross_cuda_lt_sweep_test/lib/libMNN_Cuda_Main.so: ELF 64-bit LSB shared object, ARM aarch64
CMakeCache: MNN_CUDA=ON, MNN_BUILD_TEST=ON, CUDA_ARCHS=7.2
```

Remote artifact:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_lt_sweep_test/
```

Remote logs:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/lt_sweep_20260630_2200/mlp_gemm_lt_sweep.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/lt_sweep_20260630_2200/weight_only_default_down_rows4_6_8_3072.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/lt_sweep_20260630_2200/pic_decode_mlp_rows4_6_8_3072.log
```

Sweep command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  LOG=.cache/bench_ops/lt_sweep_20260630_2200 && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_lt_sweep_test && \
  mkdir -p "$LOG" && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  env MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
      MNN_BENCH_MLP_HIDDEN=3072 \
      MNN_BENCH_MLP_INTER=8192 \
      MNN_BENCH_MLP_GEMM_WARMUP=30 \
      MNN_BENCH_MLP_GEMM_REPEAT=120 \
      MNN_BENCH_CUBLASLT_HEURISTICS=32 \
      MNN_BENCH_CUBLASLT_SWEEP=1 \
      "$ART/bin/run_test.out" bench_ops/cuda/perf/DecodeRepairMlpGemmLtFloor 2 2 1 x 2 \
      2>&1 | tee "$LOG/mlp_gemm_lt_sweep.log"'
```

cuBLASLt returned only 5 down heuristics despite requesting 32:

| rows | heuristic | cuBLASLt ms | workspace | waves |
|-----:|----------:|------------:|----------:|------:|
| 4 | 0 | 0.5042 | 49152 | 4.364 |
| 4 | 1 | 0.5076 | 768 | 4.364 |
| 4 | 2 | 0.5304 | 0 | 2.182 |
| 4 | 3 | 0.6594 | 0 | 2.400 |
| 4 | 4 | 0.7300 | 0 | 5.333 |
| 6 | 0 | 0.5106 | 73728 | 4.364 |
| 6 | 1 | 0.5103 | 768 | 4.364 |
| 6 | 2 | 0.5453 | 0 | 2.182 |
| 6 | 3 | 0.6540 | 0 | 2.400 |
| 6 | 4 | 0.9672 | 0 | 8.000 |
| 8 | 0 | 0.5049 | 98304 | 4.364 |
| 8 | 1 | 0.5063 | 768 | 4.364 |
| 8 | 2 | 0.5536 | 0 | 2.182 |
| 8 | 3 | 0.6512 | 0 | 2.400 |
| 8 | 4 | 1.3314 | 0 | 9.600 |

Same-artifact default rows45 cuBLAS command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  LOG=.cache/bench_ops/lt_sweep_20260630_2200 && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_lt_sweep_test && \
  mkdir -p "$LOG" && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  env MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
      MNN_BENCH_WEIGHT_ONLY_CASE=inter_to_hidden \
      MNN_BENCH_WEIGHT_ONLY_HIDDEN=3072 \
      MNN_BENCH_WEIGHT_ONLY_INTER=8192 \
      MNN_BENCH_WEIGHT_ONLY_WARMUP=30 \
      MNN_BENCH_WEIGHT_ONLY_REPEAT=120 \
      "$ART/bin/run_test.out" bench_ops/cuda/perf/WeightOnlyConv 2 2 1 x 2 \
      2>&1 | tee "$LOG/weight_only_default_down_rows4_6_8_3072.log"'
```

Comparison:

| rows | default rows45 cuBLAS ms | best cuBLASLt ms | delta |
|-----:|-------------------------:|-----------------:|------:|
| 4 | 0.5014 | 0.5042 | +0.0028 |
| 6 | 0.5008 | 0.5103 | +0.0095 |
| 8 | 0.5062 | 0.5049 | -0.0013 |

Decision: reject cuBLASLt as a down-proj production policy. It is not a clear win for all rows4/6/8, and rows6 regresses beyond the noise seen in this direct-op loop.

## Current MLP Direct-Op Anchor

Command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  LOG=.cache/bench_ops/lt_sweep_20260630_2200 && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_lt_sweep_test && \
  mkdir -p "$LOG" && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  env MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
      MNN_BENCH_MLP_HIDDEN=3072 \
      MNN_BENCH_MLP_INTER=8192 \
      MNN_BENCH_MLP_WARMUP=30 \
      MNN_BENCH_MLP_REPEAT=120 \
      "$ART/bin/run_test.out" bench_ops/cuda/perf/PicDecodeMlp 2 2 1 x 2 \
      2>&1 | tee "$LOG/pic_decode_mlp_rows4_6_8_3072.log"'
```

Packed path anchor:

| rows | packed gate/up concat ms | packed silu ms | down ms | packed chain ms |
|-----:|-------------------------:|---------------:|--------:|----------------:|
| 4 | 0.9434 | 0.0046 | 0.4973 | 1.5028 |
| 6 | 0.9676 | 0.0051 | 0.5123 | 1.5301 |
| 8 | 0.9839 | 0.0057 | 0.5320 | 1.5523 |

Existing scalar `PicGateUpSiluWeightOnly` path:

| rows | gateup_silu ms | down ms | chain ms | delta vs split chain |
|-----:|---------------:|--------:|---------:|---------------------:|
| 4 | 1.2415 | 0.5033 | 1.7457 | +0.2356 |
| 6 | 1.6939 | 0.5484 | 2.2441 | +0.7077 |
| 8 | 2.1292 | 0.5064 | 2.6308 | +1.0824 |

This rules out building P2 by extending the existing scalar gate/up INT4 kernels. They save an intermediate tensor but lose much more in gate/up projection throughput.

## P2 Direction

A credible full-MLP fusion should be treated as a new milestone and should start as direct-op only:

- inputs: compact hidden `[rows, 3072]`, packed gate/up INT4 weights, down INT4 or static FP16 weights;
- rows: one shared implementation for rows `1/2/4/6/8`, with schedule parameters only;
- semantic output: same compact NHWC hidden `[rows, 3072]`;
- graph/PagedCache semantics unchanged.

Design target:

1. keep packed gate/up projection tensor-core quality, either by using an internal packed GEMM tile path or by reusing the current packed `hidden -> 16384` schedule as the reference floor;
2. compute `SiLU(gate) * up` in tiles of the intermediate dimension;
3. accumulate those tiles into down-proj output without writing the full `[rows,8192]` activation to global memory;
4. only promote if direct-op beats current packed chain rows4/6/8 by at least about `0.10 ms/layer` without regressing rows1/2.

Do not use the current `PicGateUpSiluWeightOnly` scalar V14/GEMV family as the production base for this P2 work.

## P2 Tiled cuBLAS Floor Check

Added a bench-only lower-bound case:

```text
bench_ops/cuda/perf/DecodeRepairMlpTiledGemmFloor
```

It compares:

- full packed gate/up GEMM plus full down GEMM;
- tiled gate/up GEMMs plus tiled down GEMMs with beta accumulation.

This is only a floor for the possible P2 stream-activation-into-down idea. It
does not include real `SiLU(gate) * up` tile computation, tile materialization,
or a production fused-kernel schedule.

Build command:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON' \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda_tiled_mlp_floor_test" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_tiled_mlp_floor_test" \
JOBS=96 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Artifact:

```text
local:  .cache/output/mnn/artifacts/jetson_cross_cuda_tiled_mlp_floor_test/
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_tiled_mlp_floor_test/
```

Artifact checks:

```text
bin/run_test.out:        ELF 64-bit LSB executable, ARM aarch64
lib/libMNN.so:           ELF 64-bit LSB shared object, ARM aarch64
lib/libMNN_Cuda_Main.so: ELF 64-bit LSB shared object, ARM aarch64
CMakeCache: MNN_CUDA=ON, MNN_BUILD_TEST=ON, CUDA_ARCHS=7.2
```

Remote commands:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  LOG=.cache/bench_ops/tiled_mlp_floor_20260630_2300 && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_tiled_mlp_floor_test && \
  mkdir -p "$LOG" && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  env MNN_BENCH_WEIGHT_ONLY_ROWS=1,2,4,6,8 \
      MNN_BENCH_MLP_HIDDEN=3072 \
      MNN_BENCH_MLP_INTER=8192 \
      MNN_BENCH_MLP_TILE=1024 \
      MNN_BENCH_MLP_GEMM_WARMUP=30 \
      MNN_BENCH_MLP_GEMM_REPEAT=120 \
      "$ART/bin/run_test.out" bench_ops/cuda/perf/DecodeRepairMlpTiledGemmFloor 2 2 1 x 2 \
      2>&1 | tee "$LOG/tiled_floor_tile1024.log"'
```

The same command was repeated with `MNN_BENCH_MLP_TILE=2048` and `4096`.

Raw logs:

```text
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/tiled_mlp_floor_20260630_2300/tiled_floor_tile1024.log
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/tiled_mlp_floor_20260630_2300/tiled_floor_tile2048.log
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/tiled_mlp_floor_20260630_2300/tiled_floor_tile4096.log
local:  .cache/bench_ops/tiled_mlp_floor_20260630_2300/
```

Rows `4/6/8` target results:

| tile | rows | full gate/up ms | full down ms | full chain ms | tiled gate/up ms | tiled down accum ms | tiled floor ms | tiled delta | down delta |
|-----:|-----:|----------------:|-------------:|--------------:|-----------------:|--------------------:|---------------:|------------:|-----------:|
| 1024 | 4 | 0.9616 | 0.5098 | 1.5063 | 1.0679 | 0.6173 | 1.7047 | +0.1984 | +0.1074 |
| 1024 | 6 | 0.9669 | 0.5304 | 1.5330 | 1.0879 | 0.6378 | 1.7452 | +0.2122 | +0.1074 |
| 1024 | 8 | 1.0057 | 0.5260 | 1.5727 | 1.1111 | 0.6603 | 1.7898 | +0.2171 | +0.1342 |
| 2048 | 4 | 0.9888 | 0.5295 | 1.5453 | 1.0352 | 0.5521 | 1.5543 | +0.0090 | +0.0226 |
| 2048 | 6 | 0.9652 | 0.5028 | 1.5068 | 1.0269 | 0.5460 | 1.5667 | +0.0599 | +0.0432 |
| 2048 | 8 | 0.9633 | 0.5092 | 1.5280 | 1.0365 | 0.5547 | 1.5817 | +0.0537 | +0.0455 |
| 4096 | 4 | 0.9579 | 0.5131 | 1.5129 | 0.9724 | 0.5159 | 1.4874 | -0.0255 | +0.0028 |
| 4096 | 6 | 0.9605 | 0.5110 | 1.5030 | 0.9592 | 0.5151 | 1.4668 | -0.0361 | +0.0041 |
| 4096 | 8 | 0.9779 | 0.5127 | 1.5224 | 0.9780 | 0.5276 | 1.5050 | -0.0174 | +0.0149 |

Rows `1/2` sanity:

| tile | rows | full chain ms | tiled floor ms | tiled delta | down delta |
|-----:|-----:|--------------:|---------------:|------------:|-----------:|
| 1024 | 1 | 1.5742 | 1.7860 | +0.2117 | +0.0094 |
| 1024 | 2 | 1.4897 | 1.6537 | +0.1640 | +0.0919 |
| 2048 | 1 | 1.6098 | 1.5967 | -0.0130 | -0.0169 |
| 2048 | 2 | 1.5241 | 1.5858 | +0.0617 | +0.0345 |
| 4096 | 1 | 1.5527 | 1.4969 | -0.0558 | -0.0186 |
| 4096 | 2 | 1.4986 | 1.4627 | -0.0359 | -0.0036 |

Decision:

- Reject simple tiled cuBLAS P2. Tile `1024` is clearly slower. Tile `2048`
  is tied/slower. Tile `4096` has only a `0.017-0.036 ms/layer` chain-level
  lower-bound gain on rows `4/6/8`, while tiled down accumulation itself is
  `0.0028-0.0149 ms/layer` slower than the full down GEMM.
- The best theoretical rows4/6/8 endpoint impact from this floor is roughly
  `0.3-0.6 ms/token` across 16 layers before adding activation work, well below
  the target `>=3 ms` endpoint gain.
- This does not reject all full-MLP fusion. It rejects implementing the fused
  route as ordinary cuBLAS tile GEMMs. A viable next P2 must be a true fused
  CUDA kernel that keeps activation and partial sums close to compute and
  proves at least about `0.10 ms/layer` direct-op savings on rows `4/6/8`.

## rows45 cuBLAS compute=16f Repeat

The existing production code already has tune-only rows45 cuBLAS env knobs:

```text
MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_MATH
MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE
MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO
```

After the tiled floor failed, `compute=16f` was repeated because the first
isolated down run showed a small improvement.

Artifact reused:

```text
local:  .cache/output/mnn/artifacts/jetson_cross_cuda_tiled_mlp_floor_test/
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_tiled_mlp_floor_test/
```

Raw logs:

```text
local:  .cache/bench_ops/down_cublas_policy_20260701_0000/
local:  .cache/bench_ops/down_cublas_policy_20260701_0030/
local:  .cache/bench_ops/down_cublas_algo16f_20260701_0045/
local:  .cache/bench_ops/down_cublas_math16f_20260701_0055/
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_cublas_policy_20260701_0030/
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_cublas_algo16f_20260701_0045/
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_cublas_math16f_20260701_0055/
```

Representative isolated down command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  LOG=.cache/bench_ops/down_cublas_policy_20260701_0030 && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_tiled_mlp_floor_test && \
  mkdir -p "$LOG" && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  env MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=16f \
      MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
      MNN_BENCH_WEIGHT_ONLY_CASE=inter_to_hidden \
      MNN_BENCH_WEIGHT_ONLY_HIDDEN=3072 \
      MNN_BENCH_WEIGHT_ONLY_INTER=8192 \
      MNN_BENCH_WEIGHT_ONLY_WARMUP=40 \
      MNN_BENCH_WEIGHT_ONLY_REPEAT=160 \
      "$ART/bin/run_test.out" bench_ops/cuda/perf/WeightOnlyConv 2 2 1 x 2 \
      2>&1 | tee "$LOG/compute16f_down_r1.log"'
```

Representative packed chain command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  LOG=.cache/bench_ops/down_cublas_policy_20260701_0030 && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_tiled_mlp_floor_test && \
  mkdir -p "$LOG" && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  env MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=16f \
      MNN_BENCH_WEIGHT_ONLY_ROWS=1,2,4,6,8 \
      MNN_BENCH_MLP_HIDDEN=3072 \
      MNN_BENCH_MLP_INTER=8192 \
      MNN_BENCH_MLP_WARMUP=30 \
      MNN_BENCH_MLP_REPEAT=120 \
      "$ART/bin/run_test.out" bench_ops/cuda/perf/PicDecodeMlp 2 2 1 x 2 \
      2>&1 | tee "$LOG/compute16f_mlp_r1.log"'
```

True default vs `compute=16f` repeats, excluding other policy logs:

| rows | isolated default ms | isolated 16f ms | delta |
|-----:|--------------------:|----------------:|------:|
| 4 | 0.4988 | 0.4867 | -0.0121 |
| 6 | 0.5054 | 0.4921 | -0.0132 |
| 8 | 0.5034 | 0.4932 | -0.0102 |

Packed MLP repeat:

| rows | packed down default ms | packed down 16f ms | down delta | packed chain default ms | packed chain 16f ms | chain delta |
|-----:|-----------------------:|-------------------:|-----------:|------------------------:|--------------------:|------------:|
| 1 | 0.2607 | 0.2541 | -0.0066 | 0.8678 | 0.8451 | -0.0227 |
| 2 | 0.3148 | 0.3118 | -0.0030 | 0.9595 | 0.9539 | -0.0056 |
| 4 | 0.5235 | 0.4976 | -0.0258 | 1.5240 | 1.4962 | -0.0278 |
| 6 | 0.5051 | 0.4999 | -0.0052 | 1.5128 | 1.5097 | -0.0031 |
| 8 | 0.5271 | 0.5053 | -0.0218 | 1.5386 | 1.5166 | -0.0220 |

Accuracy:

```bash
env MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=16f \
  "$ART/bin/run_test.out" bench_ops/cuda/accuracy/Rows45Cublas 2 2 1 x 2
```

Result: passed. All displayed rows4/5, Llama3.2-3B rows4/6/8 gate/up/down, and
`PicLinearNhwcWeightOnly` rows1/2/4/6/8 cases had `bad=0`.

Algo and math cross-check:

- `compute=16f` with explicit `ALGO=default` and `ALGO=tensor` stayed in the
  same narrow range; numbered algorithms were slower or unsupported.
- `compute=16f` with `MATH=tensor/default/keep` did not produce a consistent
  larger win.

Decision:

- Keep `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=16f` as a diagnostic knob.
- Do not promote it as the default down policy for this objective. The best
  isolated gain is about `0.010-0.013 ms/layer`; even across all 28 layers of
  the 3B model this is well below the requested `>=3 ms/token` endpoint gain,
  and it changes accumulation precision.

## Temporary WMMA Down Kernel Diagnostic

A temporary source branch added an env-gated target-shape WMMA path:

```text
MNN_CUDA_PIC_INT4_DOWN_SMALL_WMMA=1
```

Guard:

```text
pic_decode_repair_sparse
fp16
static dequant available
no runtime dequant
rows in {4,6,8}
ic=8192
oc=3072
```

The kernel staged the small-M input tile to shared memory, used half WMMA
`16x16x16` fragments against the existing static FP16 dequantized weight
layout, and wrote compact NHWC output. This was a diagnostic branch only and
has been removed from source after the loss below.

Build command:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON' \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda_down_wmma_test" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_down_wmma_test" \
JOBS=96 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Artifact:

```text
local:  .cache/output/mnn/artifacts/jetson_cross_cuda_down_wmma_test/
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_down_wmma_test/
```

Artifact checks:

```text
bin/run_test.out:        ELF 64-bit LSB executable, ARM aarch64
lib/libMNN.so:           ELF 64-bit LSB shared object, ARM aarch64
lib/libMNN_Cuda_Main.so: ELF 64-bit LSB shared object, ARM aarch64
CMakeCache: MNN_CUDA=ON, MNN_BUILD_TEST=ON, CUDA_ARCHS=7.2
```

Raw logs:

```text
local:  .cache/bench_ops/down_wmma_20260701_0100/
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_wmma_20260701_0100/
```

Accuracy command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  LOG=.cache/bench_ops/down_wmma_20260701_0100 && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_down_wmma_test && \
  mkdir -p "$LOG" && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  env MNN_CUDA_PIC_INT4_DOWN_SMALL_WMMA=1 \
      "$ART/bin/run_test.out" bench_ops/cuda/accuracy/Rows45Cublas 2 2 1 x 2 \
      2>&1 | tee "$LOG/accuracy_wmma.log"'
```

Accuracy result: passed with `bad=0` for all displayed rows4/5, Llama3.2-3B
rows4/6/8 gate/up/down, and `PicLinearNhwcWeightOnly` rows1/2/4/6/8 cases.

Direct-op perf:

| rows | default cuBLAS ms | WMMA ms | delta |
|-----:|------------------:|--------:|------:|
| 4 | 0.5164 | 1.0593 | +0.5429 |
| 6 | 0.5055 | 1.0276 | +0.5221 |
| 8 | 0.5161 | 1.0542 | +0.5381 |

Profile confirmed the diagnostic branch:

```text
CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1_down_small_wmma batch=4 ic=8192 oc=3072 ...
CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1_down_small_wmma batch=6 ic=8192 oc=3072 ...
CUDAWeightOnlyConv profile op=conv_fpa_intb_1x1_down_small_wmma batch=8 ic=8192 oc=3072 ...
```

Decision:

- Reject the WMMA small-M down path. It is about 2x slower than rows45 cuBLAS.
- The diagnostic branch was removed from production source immediately after
  the direct-op loss.

## Transposed Static Layout Floor

After the WMMA diagnostic lost, a bench-only lower-bound case was added:

```text
bench_ops/cuda/perf/DecodeRepairDownTransposedLayoutFloor
```

It compares the current static FP16 down cache layout with cuBLAS `CUBLAS_OP_T`
against a synthetic transposed static cache layout using `CUBLAS_OP_N`.

Artifact:

```text
local:  .cache/output/mnn/artifacts/jetson_cross_cuda_down_transpose_test/
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_down_transpose_test/
```

Artifact checks:

```text
bin/run_test.out:        ELF 64-bit LSB executable, ARM aarch64
lib/libMNN.so:           ELF 64-bit LSB shared object, ARM aarch64
lib/libMNN_Cuda_Main.so: ELF 64-bit LSB shared object, ARM aarch64
CMakeCache: MNN_CUDA=ON, MNN_BUILD_TEST=ON, CUDA_ARCHS=7.2
```

Build command:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON' \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda_down_transpose_test" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_down_transpose_test" \
JOBS=96 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Run command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  LOG=.cache/bench_ops/down_transpose_layout_20260701_0120 && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_down_transpose_test && \
  mkdir -p "$LOG" && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  env MNN_BENCH_WEIGHT_ONLY_ROWS=1,2,4,6,8 \
      MNN_BENCH_MLP_HIDDEN=3072 \
      MNN_BENCH_MLP_INTER=8192 \
      MNN_BENCH_MLP_GEMM_WARMUP=40 \
      MNN_BENCH_MLP_GEMM_REPEAT=160 \
      "$ART/bin/run_test.out" bench_ops/cuda/perf/DecodeRepairDownTransposedLayoutFloor 2 2 1 x 2 \
      2>&1 | tee "$LOG/down_transposed_layout_floor_fast16.log"'
```

Raw logs:

```text
local:  .cache/bench_ops/down_transpose_layout_20260701_0120/
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_transpose_layout_20260701_0120/
```

Results:

| rows | production NT ms | transposed NN ms | transposed delta | transposed NN fast16 ms | fast16 delta | transposed NN 16f ms | 16f delta |
|-----:|-----------------:|-----------------:|-----------------:|------------------------:|-------------:|---------------------:|----------:|
| 1 | 0.6004 | 0.8435 | +0.2431 | 0.8328 | +0.2325 | 0.5043 | -0.0960 |
| 2 | 0.5361 | 0.5628 | +0.0267 | 0.5678 | +0.0317 | 0.4906 | -0.0455 |
| 4 | 0.5335 | 0.5635 | +0.0301 | 0.5636 | +0.0301 | 0.4935 | -0.0400 |
| 6 | 0.5328 | 0.5597 | +0.0270 | 0.5651 | +0.0324 | 0.4964 | -0.0364 |
| 8 | 0.5475 | 0.5672 | +0.0196 | 0.5715 | +0.0240 | 0.5057 | -0.0418 |

Decision:

- Reject transposed static layout as a production down cache. With 32F or
  `fast16` compute, the transposed `CUBLAS_OP_N` path regresses rows `4/6/8` by
  roughly `0.020-0.032 ms/layer`.
- The full `16f` variant improves rows `4/6/8` by about `0.036-0.042
  ms/layer`, but that is the same accumulation-policy knob already measured
  above, not a layout benefit.
- A production transposed cache would also need another large static FP16 copy
  of the down weight. There is no direct-op evidence to justify that memory
  cost.

## cuBLAS 32F Algorithm Sweep

As a final cheap P1 retune check, the existing rows45 cuBLAS path was swept
across non-Lt `cublasGemmEx` algorithm IDs using the default 32F accumulation
policy. This does not change layout or precision.

Artifact reused:

```text
local:  .cache/output/mnn/artifacts/jetson_cross_cuda_down_transpose_test/
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_down_transpose_test/
```

Run command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  LOG=.cache/bench_ops/down_cublas_algo32f_20260701_0225 && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_down_transpose_test && \
  mkdir -p "$LOG" && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  : > "$LOG/algo32f_sweep.log" && \
  for algo in default tensor 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115; do \
    echo "=== algo=$algo ===" | tee -a "$LOG/algo32f_sweep.log"; \
    env MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO="$algo" \
        MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
        MNN_BENCH_WEIGHT_ONLY_CASE=inter_to_hidden \
        MNN_BENCH_WEIGHT_ONLY_HIDDEN=3072 \
        MNN_BENCH_WEIGHT_ONLY_INTER=8192 \
        MNN_BENCH_WEIGHT_ONLY_WARMUP=20 \
        MNN_BENCH_WEIGHT_ONLY_REPEAT=80 \
        "$ART/bin/run_test.out" bench_ops/cuda/perf/WeightOnlyConv 2 2 1 x 2 \
        2>&1 | tee -a "$LOG/algo32f_sweep.log"; \
    status=${PIPESTATUS[0]}; \
    echo "=== algo=$algo status=$status ===" | tee -a "$LOG/algo32f_sweep.log"; \
  done'
```

Raw log:

```text
local:  .cache/bench_ops/down_cublas_algo32f_20260701_0225/algo32f_sweep.log
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_cublas_algo32f_20260701_0225/algo32f_sweep.log
```

Best supported algorithm per row:

| rows | best algo | best ms |
|-----:|:----------|--------:|
| 4 | 99 | 0.5007 |
| 6 | 99 | 0.5040 |
| 8 | 99 | 0.5029 |

Representative non-winners:

| algo | rows4 ms | rows6 ms | rows8 ms | note |
|:-----|---------:|---------:|---------:|:-----|
| default | 0.5386 | 0.5250 | 0.5276 | slower than tensor-op default |
| tensor | 0.5257 | 0.5197 | 0.5346 | alias path, slower in this run |
| 0 | 0.7444 | 0.9817 | 1.3479 | slower |
| 105 | 0.6202 | 0.6178 | 0.6046 | best non-99 tensor-op ID but still slower |
| 1/12/13/14/15/16/17 | unsupported | unsupported | unsupported | cublas status 15 |

Decision:

- Keep the existing `CUBLAS_GEMM_DEFAULT_TENSOR_OP` policy. Explicit algorithm
  IDs do not provide a hidden rows `4/6/8` down-proj win under 32F accumulation.
- This closes the practical P1 GEMM-policy search for the current static FP16
  down layout.

## Split-N Parallel Down GEMM Floor

As one last cheap lower-bound, a bench-only direct-op case was added:

```text
bench_ops/cuda/perf/DecodeRepairDownSplitNGemmFloor
```

It splits the down-proj output channel dimension `N=3072` into equal chunks and
compares:

- one full production-layout cuBLAS GEMM;
- sequential chunked cuBLAS GEMMs;
- parallel chunked cuBLAS GEMMs on separate streams and cuBLAS handles.

This was only a floor for a possible multi-stream down implementation. It does
not change production dispatch, graph layout, exporter behavior, or PagedCache
semantics.

Build command:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON' \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda_down_splitn_test" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_down_splitn_test" \
JOBS=96 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Artifact checks:

```text
.cache/output/mnn/artifacts/jetson_cross_cuda_down_splitn_test/bin/run_test.out:        ELF 64-bit LSB executable, ARM aarch64
.cache/output/mnn/artifacts/jetson_cross_cuda_down_splitn_test/lib/libMNN.so:           ELF 64-bit LSB shared object, ARM aarch64
.cache/output/mnn/artifacts/jetson_cross_cuda_down_splitn_test/lib/libMNN_Cuda_Main.so: ELF 64-bit LSB shared object, ARM aarch64
CMakeCache: MNN_CUDA=ON, MNN_BUILD_TEST=ON, CUDA_ARCHS=7.2
```

Artifact:

```text
local:  .cache/output/mnn/artifacts/jetson_cross_cuda_down_splitn_test/
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_down_splitn_test/
```

Remote logs:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_splitn_floor_20260630_1451/down_splitn_rows4_6_8.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_splitn_floor_20260630_1451/down_splitn_chunks4_rows1_2_4_6_8_repeat240.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_splitn_floor_20260630_1451/down_splitn_chunks8_rows1_2_4_6_8_repeat240.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/down_splitn_floor_20260630_1451/weight_only_default_down_rows4_6_8_repeat240.log
```

Local copies:

```text
.cache/bench_ops/down_splitn_floor_20260630_1451/
```

Initial sweep command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  LOG=.cache/bench_ops/down_splitn_floor_20260630_1451 && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_down_splitn_test && \
  mkdir -p "$LOG" && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  env MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
      MNN_BENCH_MLP_HIDDEN=3072 \
      MNN_BENCH_MLP_INTER=8192 \
      MNN_BENCH_MLP_GEMM_WARMUP=30 \
      MNN_BENCH_MLP_GEMM_REPEAT=120 \
      "$ART/bin/run_test.out" bench_ops/cuda/perf/DecodeRepairDownSplitNGemmFloor 2 2 1 x 2 \
      2>&1 | tee "$LOG/down_splitn_rows4_6_8.log"'
```

Initial sweep target rows:

| chunks | rows | single ms | split_seq ms | split_parallel ms | parallel delta |
|-------:|-----:|----------:|-------------:|------------------:|---------------:|
| 2 | 4 | 0.5162 | 0.6192 | 0.4950 | -0.0212 |
| 3 | 4 | 0.5169 | 0.4943 | 0.4903 | -0.0265 |
| 4 | 4 | 0.5023 | 0.5547 | 0.4821 | -0.0203 |
| 6 | 4 | 0.5138 | 0.5503 | 0.5062 | -0.0076 |
| 8 | 4 | 0.5139 | 0.5423 | 0.4810 | -0.0328 |
| 2 | 6 | 0.5136 | 0.6233 | 0.4995 | -0.0142 |
| 3 | 6 | 0.5091 | 0.4826 | 0.4881 | -0.0210 |
| 4 | 6 | 0.4998 | 0.5565 | 0.4778 | -0.0220 |
| 6 | 6 | 0.5186 | 0.5628 | 0.5206 | +0.0020 |
| 8 | 6 | 0.5052 | 0.5455 | 0.4859 | -0.0194 |
| 2 | 8 | 0.5090 | 0.6290 | 0.5132 | +0.0042 |
| 3 | 8 | 0.5061 | 0.4969 | 0.5058 | -0.0003 |
| 4 | 8 | 0.5198 | 0.5687 | 0.4928 | -0.0269 |
| 6 | 8 | 0.5297 | 0.5652 | 0.5273 | -0.0024 |
| 8 | 8 | 0.5219 | 0.5556 | 0.4986 | -0.0233 |

Longer repeat command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  LOG=.cache/bench_ops/down_splitn_floor_20260630_1451 && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_down_splitn_test && \
  mkdir -p "$LOG" && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  for chunks in 4 8; do \
    env MNN_BENCH_DOWN_NCHUNKS="$chunks" \
        MNN_BENCH_WEIGHT_ONLY_ROWS=1,2,4,6,8 \
        MNN_BENCH_MLP_HIDDEN=3072 \
        MNN_BENCH_MLP_INTER=8192 \
        MNN_BENCH_MLP_GEMM_WARMUP=50 \
        MNN_BENCH_MLP_GEMM_REPEAT=240 \
        "$ART/bin/run_test.out" bench_ops/cuda/perf/DecodeRepairDownSplitNGemmFloor 2 2 1 x 2 \
        2>&1 | tee "$LOG/down_splitn_chunks${chunks}_rows1_2_4_6_8_repeat240.log"; \
  done'
```

Longer repeat:

| chunks | rows | single ms | split_seq ms | split_parallel ms | parallel delta |
|-------:|-----:|----------:|-------------:|------------------:|---------------:|
| 4 | 1 | 0.5726 | 0.5996 | 0.5684 | -0.0042 |
| 4 | 2 | 0.5168 | 0.5496 | 0.4730 | -0.0439 |
| 4 | 4 | 0.5125 | 0.5563 | 0.4822 | -0.0303 |
| 4 | 6 | 0.5239 | 0.5613 | 0.4963 | -0.0276 |
| 4 | 8 | 0.5136 | 0.5624 | 0.4927 | -0.0209 |
| 8 | 1 | 0.5617 | 0.6207 | 0.5729 | +0.0112 |
| 8 | 2 | 0.4960 | 0.5326 | 0.4737 | -0.0222 |
| 8 | 4 | 0.4973 | 0.5366 | 0.4816 | -0.0156 |
| 8 | 6 | 0.5138 | 0.5485 | 0.4929 | -0.0208 |
| 8 | 8 | 0.5116 | 0.5381 | 0.4915 | -0.0201 |

Same-artifact production down anchor:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  LOG=.cache/bench_ops/down_splitn_floor_20260630_1451 && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_down_splitn_test && \
  mkdir -p "$LOG" && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  env MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
      MNN_BENCH_WEIGHT_ONLY_CASE=inter_to_hidden \
      MNN_BENCH_WEIGHT_ONLY_HIDDEN=3072 \
      MNN_BENCH_WEIGHT_ONLY_INTER=8192 \
      MNN_BENCH_WEIGHT_ONLY_WARMUP=50 \
      MNN_BENCH_WEIGHT_ONLY_REPEAT=240 \
      "$ART/bin/run_test.out" bench_ops/cuda/perf/WeightOnlyConv 2 2 1 x 2 \
      2>&1 | tee "$LOG/weight_only_default_down_rows4_6_8_repeat240.log"'
```

Result:

| rows | production `WeightOnlyConv` default ms |
|-----:|---------------------------------------:|
| 4 | 0.5071 |
| 6 | 0.5089 |
| 8 | 0.5177 |

Decision:

- Reject split-N as a production down-proj path for this objective. The repeat
  shows only about `0.016-0.030 ms/layer` lower-bound gain for rows `4/6/8`.
- Even if all 28 down projections hit the best observed delta, the endpoint
  gain would be below `1 ms/token` before accounting for additional scheduling,
  handle/stream ownership, and synchronization complexity in production.
- Chunk `8` can regress the synthetic rows1 floor, so a production policy would
  need more shape guards. That conflicts with the objective unless the gain is
  large enough to justify the added policy surface, which it is not.
- The default production down branch remains `conv_fpa_intb_1x1_rows45_cublas`
  on static dequantized FP16 weights.
