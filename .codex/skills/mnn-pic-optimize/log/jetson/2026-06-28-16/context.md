# Context

## Starting Point

Formal no-profile baseline:

```text
.cache/mnn-pic-benchmark/decode_repair_default_policy3_confirm_20260628_150247/summary.csv
```

| x | TPOT ms |
|---|--------:|
| 0 | 81.22 |
| 1 | 89.56 |
| 3 | 116.78 |
| 5 | 118.45 |
| 7 | 121.13 |

Profile attribution from the previous hour showed that `x=3/5/7` are close because they share the same active rows=4/6/8 compact dense plateau. The added latency versus `x=0/1` is dominated by compact-row `Convolution`, especially MLP `gate_proj/up_proj/down_proj` and attention `q/k/v/o_proj`; selector/ranking and `PicSparseAttention` are not the source of the 100 ms gap.

## cuBLAS Policy Sweeps

Rejected:

- `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=32f_fast16`
- `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=16f`
- `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO=99..115`

The default tensor-op cuBLAS algorithm remained fastest or tied. Other algorithms were often slower, so this does not close the remaining `x=3/5/7` 17-21 ms gap.

## cublasLt Floor

Added a test-only direct-op bench:

```text
bench_ops/cuda/perf/DecodeRepairMlpGemmLtFloor
```

The bench links `run_test.out` against `cublasLt` when available. Jetson rows=4/6/8 MLP projection sums were:

| rows | cublasLt projection_sum ms |
|------|---------------------------:|
| 4 | 1.4977 |
| 6 | 1.4815 |
| 8 | 1.4917 |

This is effectively the same floor as the current rows45 cuBLAS/static-dequant direct path, so cublasLt is not a production switch.

## WMMA Experiment

An env-gated one-warp WMMA prototype was temporarily added to `ConvFpAIntBExecution.cu`, built for sm72, pushed to Jetson, and tested with:

```text
MNN_CUDA_PIC_INT4_ROWS45_WMMA=mlp
MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=all
MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8
MNN_BENCH_WEIGHT_ONLY_HIDDEN=3072
MNN_BENCH_WEIGHT_ONLY_INTER=8192
```

Baseline rows45 path:

| case | rows=4 ms | rows=6 ms | rows=8 ms |
|------|----------:|----------:|----------:|
| hidden_to_hidden | 0.2052 | 0.2048 | 0.2050 |
| hidden_to_kv | 0.0670 | 0.0685 | 0.0702 |
| hidden_to_inter | 0.4862 | 0.4861 | 0.4877 |
| inter_to_hidden | 0.5173 | 0.5101 | 0.5145 |

Naive WMMA `mlp`:

| case | rows=4 ms | rows=6 ms | rows=8 ms |
|------|----------:|----------:|----------:|
| hidden_to_hidden | 0.2036 | 0.2039 | 0.2054 |
| hidden_to_kv | 0.4072 | 0.4972 | 0.5741 |
| hidden_to_inter | 1.3248 | 1.5824 | 1.7834 |
| inter_to_hidden | 1.4681 | 1.7451 | 1.9514 |

The prototype was 2.7-3.8x slower for MLP rows and also misclassified the 3072->1024 KV projection as MLP-like under the broad shape matcher. It was removed from production code after the test. `source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu` has no remaining WMMA diff.

## Verification After Cleanup

Clean rebuild:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON' \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_test" \
JOBS=96 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Artifact synced to:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_test/
```

Post-cleanup smoke:

| case | rows=4 ms | rows=6 ms | rows=8 ms |
|------|----------:|----------:|----------:|
| hidden_to_inter | 0.4907 | 0.4941 | 0.4995 |

## Current Worktree Impact

Production CUDA dense code was restored to no diff. Remaining useful changes are test-only:

- `test/CMakeLists.txt`: link `run_test.out` with `cublasLt` if available.
- `test/bench_ops/cuda/CudaWeightOnlyConvPerf.cpp`: add `DecodeRepairMlpGemmLtFloor` to measure cublasLt small-M floor.

## Existing Gate/Up Fusion Evidence

Current direct-op `PicDecodeMlp` on the cleaned artifact:

```text
MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8
MNN_BENCH_MLP_HIDDEN=3072
MNN_BENCH_MLP_INTER=8192
MNN_BENCH_MLP_WARMUP=10
MNN_BENCH_MLP_REPEAT=50
bench_ops/cuda/perf/PicDecodeMlp
```

| rows | split chain ms | packed chain ms | packed delta ms |
|------|---------------:|----------------:|----------------:|
| 4 | 1.5268 | 1.5615 | +0.0347 |
| 6 | 1.5223 | 1.5223 | +0.0000 |
| 8 | 1.5434 | 1.5344 | -0.0089 |

The exported `PicGateUpSiluWeightOnly` concat-conv path maps to concat projection plus `PicPackedSiluMul`. This saves at most about 0.009 ms per MLP in the measured rows and can be slower, so it cannot close a 17-21 ms endpoint gap.

Older endpoint attempts with scalar fused gate/up (`v14mboc2` / `v14mboc8`) were also rejected. Representative summaries:

| run | x=3 ms | x=5 ms | x=7 ms |
|-----|-------:|-------:|-------:|
| `decode_repair_v14mboc8_control_20260628_11` | 124.01 | 125.25 | 126.66 |
| `decode_repair_v14mboc8_env_20260628_11` | 123.84 | 126.18 | 127.32 |

This confirms that packed/scalar gate-up fusion is not the missing small-M dense path.

## Attention q/k/v Concatenation Check

Direct-op `hidden_to_qkv_concat` was measured on the cleaned artifact:

| rows | qkv concat ms |
|------|--------------:|
| 4 | 0.3766 |
| 6 | 0.3693 |
| 8 | 0.3770 |

The current split projection estimate is `hidden_to_hidden + 2 * hidden_to_kv`, roughly `0.205 + 2 * 0.068 = 0.341 ms`. Simple q/k/v concatenation is therefore not a useful attention projection fusion for rows=4/6/8.

## cuBLAS Batched Projection Fusion Check

Added another test-only direct bench:

```text
bench_ops/cuda/perf/DecodeRepairGateUpBatchedGemmFloor
```

The bench uses the same production dequantized weight layout as rows45 cuBLAS (`weight[oc][icp]`, `CUBLAS_OP_T`) and compares:

- two separate `cublasGemmEx` calls;
- one `cublasGemmBatchedEx` call with device pointer arrays, matching the realistic case where gate/up resources are separate allocations;
- one `cublasGemmStridedBatchedEx` call with contiguous weights/outputs, matching the best-case packed layout.

Build and sync:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON' \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_test" \
JOBS=96 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/jetson_cross_cuda_test/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_test/
```

MLP gate/up (`hidden=3072`, `inter=8192`):

| rows | separate gate+up ms | batched pointer ms | pointer delta ms | batched strided ms | strided delta ms |
|------|--------------------:|-------------------:|-----------------:|-------------------:|-----------------:|
| 4 | 0.9651 | 1.2476 | +0.2826 | 1.2361 | +0.2710 |
| 6 | 0.9865 | 1.2452 | +0.2587 | 1.2392 | +0.2527 |
| 8 | 0.9980 | 1.2875 | +0.2896 | 1.2524 | +0.2545 |

Attention k/v-style pair (`hidden=3072`, `inter=1024` used as two same-shaped `3072->1024` projections):

| rows | separate two projections ms | batched pointer ms | pointer delta ms | batched strided ms | strided delta ms |
|------|----------------------------:|-------------------:|-----------------:|-------------------:|-----------------:|
| 4 | 0.1395 | 0.1708 | +0.0313 | 0.1708 | +0.0313 |
| 6 | 0.1391 | 0.1692 | +0.0301 | 0.1762 | +0.0372 |
| 8 | 0.1369 | 0.1634 | +0.0265 | 0.1679 | +0.0310 |

Conclusion: batched cuBLAS projection fusion is not a viable production direction. It is slower even in the contiguous-weight best case, so do not implement `PicGateUpWeightOnly` with cuBLAS batched pointer arrays or rework export solely to enable strided-batched cuBLAS.

## Current MLP Chain Refresh

Same artifact, `PicDecodeMlp` direct-op with `warmup=20`, `repeat=80`:

| rows | split chain ms | packed chain ms | packed delta ms | split gate/up/down ms |
|------|---------------:|----------------:|----------------:|----------------------:|
| 4 | 1.5403 | 1.5238 | -0.0164 | 0.4872 / 0.4837 / 0.5277 |
| 6 | 1.5388 | 1.5092 | -0.0296 | 0.4886 / 0.4839 / 0.5014 |
| 8 | 1.5602 | 1.5505 | -0.0096 | 0.4891 / 0.4897 / 0.5210 |

`PicPackedSiluMul` saves only a few hundredths of a millisecond per MLP layer. It can be kept as a small cleanup candidate, but it is not the 100 ms unlock by itself.

## V14 Row-Splitting Check

Existing batch<=3 V14 tiny GEMV is faster than old scalar experiments but still not useful as a split fallback for rows=4/6/8. Direct `WeightOnlyConv` on the same artifact:

| case | rows=1 ms | rows=2 ms | rows=3 ms | rows=4 ms | rows=6 ms | rows=8 ms |
|------|----------:|----------:|----------:|----------:|----------:|----------:|
| hidden_to_inter | 0.2887 | 0.3267 | 0.4572 | 0.4921 | 0.4888 | 0.4872 |
| inter_to_hidden | 0.2568 | 0.3127 | 0.4421 | 0.5207 | 0.5130 | 0.5211 |
| hidden_to_hidden | 0.1159 | 0.1273 | 0.1769 | 0.2049 | 0.2046 | 0.2078 |
| hidden_to_kv | 0.0449 | 0.0469 | 0.0640 | 0.0681 | 0.0678 | 0.0678 |

Estimated row splits are worse than the current rows45 plateau:

- rows=4 as `3+1`: `hidden_to_inter ~= 0.7459 ms` vs current `0.4921 ms`; `inter_to_hidden ~= 0.6989 ms` vs current `0.5207 ms`.
- rows=6 as `3+3`: `hidden_to_inter ~= 0.9144 ms` vs current `0.4888 ms`.
- rows=8 as `3+3+2`: `hidden_to_inter ~= 1.2411 ms` vs current `0.4872 ms`.

Conclusion: do not split rows=4/6/8 into smaller V14 GEMV launches. Despite the rows45 cuBLAS fixed-cost plateau, it is still better than multiple tiny GEMV launches for these 3B MLP and attention projection shapes.

## Next Step

The bottleneck is not another cuBLAS knob. To move `x=3/5/7` below 100 ms, the next implementation should directly reduce the number or fixed cost of rows=4/6/8 dense projection launches:

- true small-M tensor-core dense kernel that avoids repeated A reload and preserves current static-dequant/cublas semantics; or
- a new MLP fusion that beats the direct-op split chain by a meaningful margin, not the existing concat-conv / scalar gate-up variants; or
- an attention projection fusion better than naive q/k/v concatenation.

Any candidate must first beat the current direct-op rows=4/6/8 numbers above before running an endpoint x=0/1/3/5/7 sweep.
