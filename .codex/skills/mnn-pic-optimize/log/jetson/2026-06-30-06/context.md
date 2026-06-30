# Context

## Goal

Optimize Jetson CUDA `lagged_attention_hkvd` decode repair compact MLP so x=3/5/7 no longer sit at 110+ms TPOT. The implementation must be shared and parameterized; x=3/x=5/x=7 must not use separate graph paths, operators, or validation rules.

## Implementation

Changed the PIC exporter and CUDA Extra handling so MLP gate/up/silu can lower to:

```text
Reshape [-1,1,1,ic] NHWC
Extra PicLinearNhwcWeightOnly out_features = 2 * inter
Extra PicPackedSiluMul packed_input_nhwc=1 out_features = inter
```

For Llama3.2-3B this means:

```text
ic=3072
inter=8192
packed gate/up oc=16384
```

Files touched for this P0 path:

```text
transformers/pic_llm/export/utils/mnn_converter.py
transformers/pic_llm/export/llmexport.py
source/shape/ShapePicExtra.cpp
source/backend/cuda/core/CUDABackend.cpp
source/backend/cuda/execution/FuseExecution.cu
test/bench_ops/cuda/CudaWeightOnlyConvPerf.cpp
```

The exported graph confirms the intended lowering:

```text
PicLinearNhwcWeightOnly 56
PicPackedSiluMul        28
nhwc_gate_up_linear     84
packed_input_nhwc       28
concat_conv              0
```

`PicLinearNhwcWeightOnly=56` is 28 packed gate/up linears plus 28 down linears. `PicPackedSiluMul=28` is one per layer.

## Build And Export

Runtime artifact:

```text
local:  .cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear
```

Installed runtime timestamps:

```text
2026-06-30 13:41:21 .cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear/bin/pic_server
2026-06-30 13:41:21 .cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear/lib/libMNN.so
2026-06-30 13:41:21 .cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear/lib/libMNN_Cuda_Main.so
2026-06-30 13:41:21 .cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear/lib/libpic_llm.so
```

Test artifact for direct-op accuracy:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
  BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  CMAKE_ARGS='-DMNN_BUILD_TEST=ON' \
  BUILD_DIR=.cache/build/mnn/jetson_cross_cuda_nhwc_linear_test \
  INSTALL_PREFIX=.cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear_test \
  JOBS=24 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Exported real-weight PIC model:

```text
.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-nhwc-gateup
```

`llm_config.json` contains:

```text
paged_attention=true
pic_recompute_budget=true
pic_decode_gateup_fusion=true
pic_decode_nhwc_linear_fusion=true
pic_decode_nhwc_linear_scope=mlp
```

## Direct Op Accuracy

Remote command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  env LD_LIBRARY_PATH=.cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear_test/lib:${LD_LIBRARY_PATH:-} \
  .cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear_test/bin/run_test.out \
  bench_ops/cuda/accuracy/Rows45Cublas 2 2 1 empty 2'
```

Result:

```text
llama32_3b_gate_rows4_policy_all max_abs=0 max_rel=0 bad=0/32768
llama32_3b_up_rows4_policy_all   max_abs=0 max_rel=0 bad=0/32768
llama32_3b_down_rows4_policy_all max_abs=0 max_rel=0 bad=0/12288
llama32_3b_gate_rows6_policy_all max_abs=0 max_rel=0 bad=0/49152
llama32_3b_up_rows6_policy_all   max_abs=0 max_rel=0 bad=0/49152
llama32_3b_down_rows6_policy_all max_abs=0 max_rel=0 bad=0/18432
llama32_3b_gate_rows8_policy_all max_abs=0 max_rel=0 bad=0/65536
llama32_3b_up_rows8_policy_all   max_abs=0 max_rel=0 bad=0/65536
llama32_3b_down_rows8_policy_all max_abs=0 max_rel=0 bad=0/24576
pic_linear_nhwc_llama32_3b_gate_rows4 max_abs=0 max_rel=0 bad=0/32768
pic_linear_nhwc_llama32_3b_down_rows4 max_abs=0 max_rel=0 bad=0/12288
pic_linear_nhwc_llama32_3b_gate_rows6 max_abs=0 max_rel=0 bad=0/49152
pic_linear_nhwc_llama32_3b_down_rows6 max_abs=0 max_rel=0 bad=0/18432
pic_linear_nhwc_llama32_3b_gate_rows8 max_abs=0 max_rel=0 bad=0/65536
pic_linear_nhwc_llama32_3b_down_rows8 max_abs=0 max_rel=0 bad=0/24576
all <bench_ops/cuda/accuracy/Rows45Cublas> tests passed
```

The full remote log is:

```text
.cache/bench_ops/rows45_piclinear_accuracy_20260630_1416.log
```

## Formal Endpoint

Non-profile server used:

```bash
LD_LIBRARY_PATH=.cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear/lib:$LD_LIBRARY_PATH \
  .cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear/bin/pic_server \
  --config .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-nhwc-gateup/config_cuda_greedy.json \
  --host 0.0.0.0 --port 18131 \
  --kv-cache-dir .cache/pic_prefill_latency_sweep/shared_kv/llama-pic-nhwc-gateup \
  --model llama-pic-nhwc-gateup
```

Formal command shape:

```bash
NO_PROXY=192.168.101.192,localhost,127.0.0.1 \
no_proxy=192.168.101.192,localhost,127.0.0.1 \
conda run -n kvshare-edge python .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://192.168.101.192:18131 \
  --output-csv .cache/mnn-pic-benchmark/nhwc_gateup_x01357_repeat_20260630_135721/decode.csv \
  --output-dir .cache/mnn-pic-benchmark/nhwc_gateup_x01357_repeat_20260630_135721/raw \
  --device jetson --device-display 'Jetson AGX Xavier' \
  --backend cuda --frequency-profile max \
  --model llama-pic-nhwc-gateup \
  --model-name 'Llama3.2 3B NHWC gateup' \
  --model-config .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-nhwc-gateup/config_cuda_greedy.json \
  --mode full-reuse --contexts 1024 --budgets 0.00 \
  --repair-tokens 0,1,3,5,7 \
  --decode-selectors lagged_attention_hkvd \
  --score-layer-idx 1 --max-tokens 32 \
  --repeats 3 --warm-repeats 1 \
  --require-exact-context --timeout 900
```

Formal endpoint TPOT:

| x | active rows | old TPOT ms | new TPOT ms | delta |
|--:|------------:|------------:|------------:|------:|
| 0 | 1 | 81.381 | 71.367 | -10.014 |
| 1 | 2 | 92.078 | 80.065 | -12.013 |
| 3 | 4 | 118.127 | 103.547 | -14.580 |
| 5 | 6 | 119.656 | 105.557 | -14.099 |
| 7 | 8 | 120.848 | 106.806 | -14.042 |

Endpoint correctness metadata from raw repeat 1:

```text
x=0 active=1 measured=31 tpot=71.388 mode=native-full-reuse runtime=mnn_token_id_sparse_decode
x=1 active=2 measured=31 tpot=80.021 mode=native-full-reuse runtime=mnn_token_id_sparse_decode
x=3 active=4 measured=31 tpot=103.517 mode=native-full-reuse runtime=mnn_token_id_sparse_decode
x=5 active=6 measured=31 tpot=105.411 mode=native-full-reuse runtime=mnn_token_id_sparse_decode
x=7 active=8 measured=31 tpot=107.113 mode=native-full-reuse runtime=mnn_token_id_sparse_decode
```

## Profile Attribution

Profile server used only for attribution:

```bash
MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=2000 \
LD_LIBRARY_PATH=.cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear/lib:$LD_LIBRARY_PATH \
  .cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear/bin/pic_server \
  --config .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-nhwc-gateup/config_cuda_greedy.json \
  --host 0.0.0.0 --port 18131 \
  --kv-cache-dir .cache/pic_prefill_latency_sweep/shared_kv/llama-pic-nhwc-gateup \
  --model llama-pic-nhwc-gateup
```

Max1 profile CSV:

```text
.cache/mnn-pic-benchmark/nhwc_gateup_x01357_profile_max1_20260630_140845/decode.csv
```

Max1 profile log:

```text
.cache/logs/jetson_nhwc_gateup/profile_max1_20260630_140845.log
```

Low-level packed MLP profile, using `CUDAWeightOnlyConv profile` kernel rows before graph summary:

| x | active rows | new gate/up ms | new down ms | new MLP ms |
|--:|------------:|---------------:|------------:|-----------:|
| 0 | 1 | 17.742 | 8.634 | 26.376 |
| 1 | 2 | 19.601 | 9.780 | 29.381 |
| 3 | 4 | 29.157 | 17.083 | 46.240 |
| 5 | 6 | 26.495 | 15.490 | 41.985 |
| 7 | 8 | 27.787 | 15.653 | 43.440 |

Comparison to the previous NHWC static-share profile:

| x | old gate/up | new gate/up | delta | old down | new down | delta | old MLP | new MLP | delta |
|--:|------------:|------------:|------:|---------:|---------:|------:|--------:|--------:|------:|
| 0 | 19.49 | 17.74 | -1.75 | 8.74 | 8.63 | -0.11 | 28.23 | 26.38 | -1.85 |
| 1 | 21.48 | 19.60 | -1.88 | 10.25 | 9.78 | -0.47 | 31.73 | 29.38 | -2.35 |
| 3 | 32.08 | 29.16 | -2.92 | 17.82 | 17.08 | -0.74 | 49.90 | 46.24 | -3.66 |
| 5 | 32.25 | 26.50 | -5.75 | 17.85 | 15.49 | -2.36 | 50.10 | 41.99 | -8.12 |
| 7 | 32.29 | 27.79 | -4.50 | 17.92 | 15.65 | -2.27 | 50.21 | 43.44 | -6.77 |

This proves the endpoint is hitting the new packed MLP path and that `MLP gate/up` is lower than the previous ~32.3ms rows4/6/8 level. The remaining dense hotspot is `down_proj`, plus residual fixed graph/attention overhead outside MLP.

## Fixed Overhead Explanation

The reason x=3/5/7 previously clustered near 118-121ms is that once repair enters active rows 4/6/8, the graph executes almost the same number of kernels and the compact dense linears fall into the same small-M fixed-cost schedule bucket. The extra cost versus x=0/1 is mostly fixed scheduling and dense-kernel overhead, not a linear per-token increase.

Fixed items that remain visible:

```text
gate/up packed linear: one large 3072 -> 16384 compact weight-only GEMM per layer
down linear:           one 8192 -> 3072 compact weight-only GEMM per layer
attention q/o/k/v:     compact dense projection fixed launch/tile overhead
PicSparseAttention:    per-layer sparse attention launch and causal-K work
Raster/Binary/Unary:   graph glue around compact rows
```

The packed gate/up path removes the old two-projection graph shape and the duplicated gate/up scheduling/read path, but it still computes the same logical 2*inter output. That is why the endpoint cliff is removed, while profile still shows a non-trivial compact MLP base cost. The next useful optimization is not another x-specific branch; it is either:

1. a parameterized rows4/6/8 down compact GEMM schedule, or
2. a larger MLP fusion that streams `SiLU(gate) * up` tiles directly into down accumulation and avoids materializing the intermediate activation.

Both should remain one shared implementation with row-dependent schedule parameters only.
