# Context

## Objective

Implement and validate a real Jetson CUDA compact dense / MLP optimization for `lagged_attention_hkvd` decode repair `x=3/5/7`.

Success criteria:

- endpoint decode path hit;
- correctness for `x=0/1/3/5/7`;
- at least one `x=3/5/7` endpoint latency win;
- no unacceptable `x=0/1` regression;
- profile proof.

## Code Scope

Modified implementation files:

```text
source/backend/cuda/core/CUDABackend.cpp
source/backend/cuda/execution/FuseExecution.cu
source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.hpp
source/shape/ShapePicExtra.cpp
test/bench_ops/cuda/CudaWeightOnlyConvPerf.cpp
transformers/pic_llm/export/utils/mnn_converter.py
```

Relevant pre-existing cuBLASLt rows48 support is still present in:

```text
source/backend/cuda/CMakeLists.txt
source/backend/cuda/core/runtime/CUDARuntime.cpp
source/backend/cuda/core/runtime/CUDARuntime.hpp
```

Implementation summary:

- `PicLinearNhwcWeightOnlyExecution` parses existing export attrs and builds a child weight-only Conv op.
- It validates raw NHWC dims `[d0,d1,d2,C]`, computes rows as `d0*d1*d2`, forces input/output descriptors to `MNN_DATA_FORMAT_NHWC`, and passes real MNN tensors directly to `ConvFpAIntBExecution`.
- `ShapePicExtra.cpp` marks `PicLinearNhwcWeightOnly` output as NHWC so the backend sees channels in dim 3.
- `CUDABackend.cpp` routes `PicLinearNhwcWeightOnly` as backend-native Extra, avoiding accidental `FuseExecutionV2` handling.
- The exporter only emits the NHWC linear rewrite when `ic` and `oc` are multiples of 8.
- `ConvFpAIntBExecution::Resource` now reuses static dequant tensors for cloned external INT4 1x1 Conv resources. The key is CUDA runtime pointer, external path/offset vector, padded shape, and byte size. Reused resources set `mStaticDequantFilter` and `mStaticDequantBytes` but do not charge `gPicStaticDequantBytes` again.

## Root Cause Found During Endpoint Validation

The first endpoint timing with NHWC was bad:

```text
x0 82.041 ms
x1 92.225 ms
x3 201.575 ms
x5 200.636 ms
x7 202.299 ms
```

Profile showed the op was on the endpoint decode path, but static dequant was missing for most decode repair clones:

```text
252 batch=1    static=1
 84 batch=1024 static=1
 68 batch=4    static=0
 16 batch=4    static=1
 84 batch=6    static=0
 84 batch=8    static=0
```

That meant rows4/6/8 paid full INT4 runtime dequant on every decode step. The first attempt keyed the share cache by CUDA backend pointer, but cloned modules use different backend objects, so it did not hit. Keying by `CUDARuntime*` fixed the clone reuse while keeping the cache scoped to one CUDA runtime/GPU.

After the runtime-key fix, profile shows all compact decode batches using static dequant:

```text
252 batch=1    static=1
 84 batch=1024 static=1
 84 batch=4    static=1
 84 batch=6    static=1
 84 batch=8    static=1
```

## Build

Build commands used from repo root:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda_nhwc_linear" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear" \
JOBS=96 BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS="-DMNN_BUILD_TEST=ON -DCUDA_HOST_COMPILER=$PWD/.cache/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++" \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda_nhwc_linear" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear" \
JOBS=96 BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS="-DMNN_BUILD_TEST=ON -DCUDA_HOST_COMPILER=$PWD/.cache/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++" \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

The build script repeatedly wiped the build directory because cached cross compiler/sysroot metadata was empty. The installed artifact remained:

```text
.cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear
```

Artifact sync:

```bash
rsync -a --delete .cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear/
```

Binary check:

```text
pic_server:          ELF 64-bit LSB executable, ARM aarch64
run_test.out:        ELF 64-bit LSB executable, ARM aarch64
libMNN.so:           ELF 64-bit LSB shared object, ARM aarch64
libMNN_Cuda_Main.so: ELF 64-bit LSB shared object, ARM aarch64
```

## Direct Accuracy

Remote command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN
ART=.cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear
CUDA_LIB=/usr/local/cuda-12.2/targets/aarch64-linux/lib
mkdir -p .cache/bench_ops/nhwc_linear_static_share
export LD_LIBRARY_PATH="$PWD/$ART/lib:$CUDA_LIB:${LD_LIBRARY_PATH:-}"
"$ART/bin/run_test.out" bench_ops/cuda/accuracy/Rows45Cublas 2 2 1 x 2 2>&1 | tee .cache/bench_ops/nhwc_linear_static_share/rows45_piclinear_accuracy.log'
```

Result:

```text
rows4/5 cuBLAS cases: bad=0
llama32_3b rows4/6/8 gate/up/down: bad=0
pic_linear_nhwc rows1/2/4/6/8 gate/down: bad=0
all tests passed
```

## Direct Perf

Remote log:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/nhwc_linear_static_share/linear_convert_chain.log
```

Command:

```bash
env MNN_BENCH_WEIGHT_ONLY_ROWS=1,2,4,6,8 \
    MNN_BENCH_WEIGHT_ONLY_CASE=mlp \
    MNN_BENCH_MLP_HIDDEN=3072 \
    MNN_BENCH_MLP_INTER=8192 \
    MNN_BENCH_LINEAR_CHAIN_WARMUP=20 \
    MNN_BENCH_LINEAR_CHAIN_REPEAT=80 \
    "$ART/bin/run_test.out" bench_ops/cuda/perf/LinearConvertChain 2 2 1 x 2
```

Representative direct results:

| rows | op | with convert ms | NHWC ms | NHWC ratio |
|-----:|----|----------------:|--------:|-----------:|
| 1 | gate/up | 0.3109 | 0.2970 | 0.955 |
| 1 | down | 0.2732 | 0.2651 | 0.970 |
| 2 | gate/up | 0.3382 | 0.3316 | 0.981 |
| 2 | down | 0.3246 | 0.3189 | 0.982 |
| 4 | gate/up | 0.5097 | 0.5000 | 0.981 |
| 4 | down | 0.5411 | 0.5233 | 0.967 |
| 6 | gate/up | 0.5015 | 0.4925 | 0.982 |
| 6 | down | 0.5218 | 0.5124 | 0.982 |
| 8 | gate/up | 0.4912 | 0.4858 | 0.989 |
| 8 | down | 0.5084 | 0.5124 | 1.008 |

## Profile Proof

Profile server:

```bash
env LD_LIBRARY_PATH="$PWD/$ART/lib:$CUDA_LIB:${LD_LIBRARY_PATH:-}" \
  MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=2000 \
  "$ART/bin/pic_server" --config "$CONFIG" --host 127.0.0.1 --port 18131 \
  --kv-cache-dir "$KV_DIR" --model llama-pic
```

Profile smoke:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://127.0.0.1:19131 \
  --output-csv .cache/mnn-pic-benchmark/nhwc_linear_runtime_key_profile_20260630_025220/summary.csv \
  --output-dir .cache/mnn-pic-benchmark/nhwc_linear_runtime_key_profile_20260630_025220/raw \
  --device jetson --device-display "Jetson Orin NX" \
  --backend cuda --frequency-profile max \
  --model llama-pic --model-name "Llama3.2 3B NHWC runtime-key profile" \
  --model-config .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-nhwc-linear/config_cuda_greedy.json \
  --mode full-reuse --contexts 1024 --budgets 0.00 \
  --repair-tokens 3,5,7 \
  --decode-selectors lagged_attention_hkvd \
  --max-tokens 1 --repeats 1 --warm-repeats 0 --no-update-cache \
  --timeout 900
```

Proof:

```text
remote log: .cache/logs/nhwc_linear_runtime_key_profile_20260630_025207.log
local copy: .cache/logs/jetson_nhwc_linear/nhwc_linear_runtime_key_profile_20260630_025207.log
PicLinearNhwcWeightOnly profile lines: 588
batch=4 static_dequant=1: 84
batch=6 static_dequant=1: 84
batch=8 static_dequant=1: 84
no ERROR / failed / Resize error / illegal signatures
```

Profile-mode latency is intentionally not used as formal latency because graph profiling synchronizes after ops.

Fresh baseline-vs-NHWC graph-profile attribution was collected after the endpoint run:

```text
baseline log: .cache/logs/jetson_nhwc_linear/nhwc_linear_baseline_profile_20260630_030553.log
baseline summary: .cache/mnn-pic-benchmark/nhwc_linear_baseline_profile_20260630_030553/summary.csv
NHWC log: .cache/logs/jetson_nhwc_linear/nhwc_linear_nhwc_profile_fresh_20260630_031000.log
NHWC summary: .cache/mnn-pic-benchmark/nhwc_linear_nhwc_profile_fresh_20260630_031000/summary.csv
```

The profile smoke used the same context 1024, `full-reuse`, `lagged_attention_hkvd`, repair tokens `3,5,7`, `max_tokens=1`, and one repeat. The graph-profile wall time is not comparable to formal latency; it is only used to prove shape routing and hotspot attribution.

Graph-profile category totals:

| x | run | attention kernel ms | attention dense ms | MLP dense ms | MLP silu+mul ms | raster/preprocess ms | norm ms | misc elem ms | lm_head ms |
|--:|-----|--------------------:|-------------------:|-------------:|----------------:|---------------------:|--------:|-------------:|-----------:|
| 3 | baseline | 81.006 | 140.184 | 85.599 | 3.773 | 16.392 | 4.257 | 14.844 | 8.172 |
| 3 | NHWC | 84.453 | 125.024 | 90.570 | 4.617 | 15.599 | 4.259 | 16.402 | 8.091 |
| 5 | baseline | 87.479 | 46.021 | 82.840 | 3.705 | 15.122 | 3.982 | 14.049 | 8.141 |
| 5 | NHWC | 88.059 | 45.478 | 89.061 | 3.293 | 16.986 | 3.067 | 12.893 | 8.160 |
| 7 | baseline | 75.593 | 45.610 | 82.747 | 3.513 | 15.272 | 3.806 | 14.741 | 8.074 |
| 7 | NHWC | 74.662 | 48.578 | 94.003 | 3.382 | 16.644 | 4.014 | 13.878 | 8.159 |

Shape/path evidence:

```text
x=3 baseline MLP graph types: Convolution+Raster; sample inputs [1x4x3072], [1x4x8192], [4x3072x1x1]
x=3 NHWC MLP graph types: Extra+Raster; sample inputs [1x4x3072], [1x4x8192], [4x1x1x3072]
x=5 baseline MLP graph types: Convolution+Raster; sample inputs [1x6x3072], [1x6x8192], [6x3072x1x1]
x=5 NHWC MLP graph types: Extra+Raster; sample inputs [1x6x3072], [1x6x8192], [6x1x1x3072]
x=7 baseline MLP graph types: Convolution+Raster; sample inputs [1x8x3072], [1x8x8192], [8x3072x1x1]
x=7 NHWC MLP graph types: Extra+Raster; sample inputs [1x8x3072], [1x8x8192], [8x1x1x3072]
```

Lower-level MLP compact dense profile:

| x | active rows | baseline child cuBLAS ms | NHWC child cuBLAS ms | NHWC wrapper total ms | NHWC wrapper calls | static wrappers |
|--:|------------:|-------------------------:|---------------------:|----------------------:|-------------------:|----------------:|
| 3 | 4 | 50.311 | 49.757 | 52.487 | 84 | 84 |
| 5 | 6 | 49.685 | 50.376 | 52.817 | 84 | 84 |
| 7 | 8 | 49.835 | 51.336 | 53.854 | 84 | 84 |

Interpretation:

- The endpoint path is definitely hit: `PicLinearNhwcWeightOnly` runs for all x=3/5/7 compact MLP rows, with static dequant in every wrapper call.
- The synchronized graph-profile MLP totals do not show a large category reduction; they are noisy and include profiler synchronization overhead.
- The isolated `LinearConvertChain` direct bench is the cleaner proof of the dense-side optimization: NHWC avoids the explicit pre/post convert chain and gives small per-linear wins for rows 4/6 and gate/up rows 8.
- Therefore the accepted claim is compact NHWC dense endpoint optimization plus clone-safe static dequant sharing, not full gate/up/SILU/down one-kernel MLP fusion.

## Endpoint Timing

Matched non-profile commands used the same artifact, shared KV root, context, sampler, and benchmark script. Baseline config:

```text
.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json
```

NHWC config:

```text
.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-nhwc-linear/config_cuda_greedy.json
```

Common command shape:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://127.0.0.1:19131 \
  --device jetson --device-display "Jetson Orin NX" \
  --backend cuda --frequency-profile max \
  --model llama-pic \
  --mode full-reuse --contexts 1024 --budgets 0.00 \
  --repair-tokens 0,1,3,5,7 \
  --decode-selectors lagged_attention_hkvd \
  --max-tokens 8 --repeats 3 --warm-repeats 1 \
  --timeout 900
```

Local summaries:

```text
.cache/mnn-pic-benchmark/nhwc_linear_static_share_baseline_20260630_025315/summary.csv
.cache/mnn-pic-benchmark/nhwc_linear_static_share_nhwc_20260630_025407/summary.csv
```

Result:

| repair tokens | baseline TPOT ms | NHWC TPOT ms | delta ms | delta |
|--------------:|-----------------:|-------------:|---------:|------:|
| 0 | 81.795762 | 81.381190 | -0.414571 | -0.507% |
| 1 | 92.157619 | 92.078048 | -0.079571 | -0.086% |
| 3 | 118.113905 | 118.126905 | +0.013000 | +0.011% |
| 5 | 120.137429 | 119.656190 | -0.481238 | -0.401% |
| 7 | 122.145667 | 120.848238 | -1.297429 | -1.062% |

Correctness:

```text
x=0: baseline and NHWC response text match; runtime=mnn_token_id_sparse_decode
x=1: baseline and NHWC response text match; runtime=mnn_token_id_sparse_decode
x=3: baseline and NHWC response text match; runtime=mnn_token_id_sparse_decode
x=5: baseline and NHWC response text match; runtime=mnn_token_id_sparse_decode
x=7: baseline and NHWC response text match; runtime=mnn_token_id_sparse_decode
sample text: " PIC cache source, current request Paged"
```

Raw response metadata also confirms the compact active rows:

| repair tokens | active_tokens_per_decode_step | refined_token_count | per-step repair token count |
|--------------:|------------------------------:|--------------------:|----------------------------:|
| 0 | 1 | 0 | 0 |
| 1 | 2 | 8 | 1 |
| 3 | 4 | 24 | 3 |
| 5 | 6 | 40 | 5 |
| 7 | 8 | 56 | 7 |

Formal server logs:

```text
.cache/logs/nhwc_linear_static_share_formal_baseline_20260630_025302.log
.cache/logs/nhwc_linear_static_share_formal_nhwc_20260630_025355.log
```

No error signatures were found in those logs.

## Conclusion

The NHWC compact-linear path is accepted for Jetson CUDA decode repair. The endpoint path and compact NHWC MLP routing are profile-proven, correctness matches baseline for all required repair settings, `x=5` and `x=7` improve, and `x=0/1` do not regress. `x=3` is effectively neutral at `+0.013 ms`. This is a compact dense optimization plus static dequant sharing, not a full one-kernel MLP fusion.
