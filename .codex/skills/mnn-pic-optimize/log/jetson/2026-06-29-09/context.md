# Context

## Objective

Continue optimizing Jetson CUDA PIC decode repair for `lagged_attention_hkvd`.

Accepted endpoint baseline:

```text
model: Llama3.2 3B PIC boundary
backend: CUDA
context: 1024
mode: full-reuse
max_tokens: 8
attention_layer_idx: 1
top_m: 32
```

| x / repair_tokens | active rows | TPOT ms |
|------------------:|------------:|--------:|
| 0 | 1 normal decode row | 81.22 |
| 1 | 2 | 89.56 |
| 3 | 4 | 116.78 |
| 5 | 6 | 118.45 |
| 7 | 8 | 121.13 |

`x` is `repair_tokens`; sparse decode rows are `x + 1`. The optimization target is to reduce rows=4/6/8 compact-row graph/kernel cost while preserving:

- `rows = repair_tokens + 1`
- HKVD / lagged attention selector semantics
- PagedCache as the only runtime KV exchange layer
- no scratch `.k/.v`
- no full attention matrix capture
- no selector/rank shortcut

## Prior Attribution

The accepted profile attribution remains:

- `rank_result_ms` is only about `0.003-0.005 ms`.
- fused `decode_attention_rank` is about `0.2-0.4 ms/step`.
- CPU-side preprocessing (`build_rows`, `begin_recompute`, `embedding`, `mask_pos`, validation, sparse cleanup, bookkeeping) is sub-ms.
- The `x=3/5/7` growth is inside `forwardRaw`, dominated by compact-row `Convolution` / weight-only Linear:
  - MLP `gate_proj`, `up_proj`, `down_proj`
  - attention `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `PicSparseAttention` is secondary, not the first target.

Historical graph-profile totals from `2026-06-28-15`:

| x | Convolution ms | PicSparseAttention ms | PagedAttention ms |
|---|---------------:|----------------------:|------------------:|
| 0 | 193.291 | 107.156 | 5.069 |
| 1 | 213.963 | 108.420 | 28.723 |
| 3 | 374.950 | 104.184 | 17.307 |
| 5 | 291.009 | 106.740 | 18.235 |
| 7 | 287.124 | 109.059 | 13.003 |

Stable grouped deltas vs `x=0`:

| group | x0 ms | x5 delta ms | x7 delta ms |
|-------|------:|------------:|------------:|
| MLP gate/up/down | 115.745 | +39.993 | +37.148 |
| attention q/o/k/v | 16.931 | +44.074 | +43.129 |
| lm_head | 15.939 | -0.244 | -0.357 |

## Artifact

Built locally using the cross Jetson CUDA test artifact path:

```text
.cache/output/mnn/artifacts/jetson_cross_cuda_test/bin/run_test.out
```

Verified locally:

```text
ELF 64-bit LSB executable, ARM aarch64, dynamically linked
```

Synced to Jetson:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_test/
```

Remote logs:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_hkvd_20260629/linear_convert_chain.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_hkvd_20260629/pic_decode_mlp.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_hkvd_20260629/weight_only_conv.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_hkvd_20260629/decode_repair_mlp_gemm_floor.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_hkvd_20260629/decode_repair_gateup_batched_gemm_floor.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_hkvd_20260629/decode_repair_gateup_parallel_gemm_floor.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_hkvd_20260629/decode_repair_kv_parallel_gemm_floor.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_hkvd_20260629/decode_repair_qkv_parallel_gemm_floor.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_hkvd_20260629/weight_only_conv_concat_focus.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_hkvd_20260629/pic_decode_mlp_concat_focus.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_hkvd_20260629/decode_repair_mlp_gemm_lt_floor.log
/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_hkvd_20260629/pic_decode_mlp_gateup_silu_weightonly.log
```

Common run settings:

```text
backend=2 CUDA
precision=2 fp16
thread=1
flag=0
memory=2 low
rows=4,6,8
hidden=3072
inter=8192
kv=1024
warmup=20
repeat=80
```

## Linear Convert Chain

`bench_ops/cuda/perf/LinearConvertChain`

| rows | case | with convert ms | no convert ms | delta ms | ratio |
|-----:|------|----------------:|--------------:|---------:|------:|
| 4 | mlp_gate_up | 0.4896 | 0.4777 | +0.0119 | 1.025 |
| 4 | mlp_down | 0.5278 | 0.5160 | +0.0118 | 1.023 |
| 4 | attn_q_o | 0.2088 | 0.1994 | +0.0094 | 1.047 |
| 4 | attn_k_v | 0.0751 | 0.0675 | +0.0076 | 1.113 |
| 6 | mlp_gate_up | 0.4841 | 0.4786 | +0.0056 | 1.012 |
| 6 | mlp_down | 0.5112 | 0.5053 | +0.0059 | 1.012 |
| 6 | attn_q_o | 0.2083 | 0.2005 | +0.0077 | 1.039 |
| 6 | attn_k_v | 0.0743 | 0.0684 | +0.0060 | 1.088 |
| 8 | mlp_gate_up | 0.4890 | 0.4754 | +0.0135 | 1.028 |
| 8 | mlp_down | 0.5262 | 0.5368 | -0.0107 | 0.980 |
| 8 | attn_q_o | 0.2130 | 0.2037 | +0.0093 | 1.046 |
| 8 | attn_k_v | 0.0736 | 0.0681 | +0.0055 | 1.080 |

Conclusion: layout convert chain cleanup alone is not a main endpoint lever. It can remove only several microseconds per projection.

## PicDecodeMlp Chain

`bench_ops/cuda/perf/PicDecodeMlp`

| rows | gate ms | up ms | silu ms | down ms | silu_down ms | split chain ms | packed chain ms | packed delta vs split |
|-----:|--------:|------:|--------:|--------:|-------------:|---------------:|----------------:|----------------------:|
| 4 | 0.4806 | 0.4750 | 0.0080 | 0.5047 | 0.5084 | 1.5149 | 1.5151 | +0.0001 |
| 6 | 0.4853 | 0.4824 | 0.0053 | 0.5280 | 0.5312 | 1.5422 | 1.5833 | +0.0412 |
| 8 | 0.4830 | 0.4840 | 0.0052 | 0.5176 | 0.5229 | 1.5254 | 1.5407 | +0.0153 |

Conclusion: packed gate/up is not useful for rows=4/6/8 in this artifact.

## WeightOnlyConv Components

`bench_ops/cuda/perf/WeightOnlyConv`

| rows | hidden_to_inter | gateup_concat | inter_to_hidden | hidden_to_hidden | hidden_to_kv | qkv_concat |
|-----:|----------------:|--------------:|----------------:|-----------------:|-------------:|-----------:|
| 4 | 0.4737 | 0.9539 | 0.5234 | 0.1998 | 0.0651 | 0.3619 |
| 6 | 0.4799 | 0.9487 | 0.5175 | 0.2031 | 0.0665 | 0.3739 |
| 8 | 0.4837 | 0.9615 | 0.5405 | 0.2021 | 0.0673 | 0.3674 |

Conclusion: rows=4/6/8 are on a compact dense plateau. MLP projections dominate; attention projections also contribute materially due to repeated layer count.

## FP16 GEMM Floor

`bench_ops/cuda/perf/DecodeRepairMlpGemmFloor`

| rows | layout | gate | up | down | gateup_concat | projection_sum | chain_no_silu |
|-----:|--------|-----:|---:|-----:|--------------:|---------------:|--------------:|
| 4 | oc_by_rows | 0.8685 | 0.8610 | 0.5495 | 2.3495 | 2.2791 | 2.3734 |
| 4 | rows_by_oc | 0.8495 | 0.8496 | 0.8078 | 1.6637 | 2.5068 | 2.5487 |
| 6 | oc_by_rows | 0.9022 | 0.9160 | 0.5494 | 2.4021 | 2.3676 | 2.5051 |
| 6 | rows_by_oc | 0.8585 | 0.8613 | 0.8147 | 1.6208 | 2.5345 | 2.5913 |
| 8 | oc_by_rows | 1.0277 | 0.9825 | 0.5566 | 2.4479 | 2.5668 | 2.5037 |
| 8 | rows_by_oc | 0.5585 | 0.5523 | 0.5525 | 1.0650 | 1.6632 | 1.6713 |

Conclusion: generic FP16 GEMM is not a strong replacement for current static-dequant/weight-only rows=4/6/8. Rows=8 rows_by_oc is closer, but still not enough to justify a production switch.

## Gate/Up Batched GEMM Floor

`bench_ops/cuda/perf/DecodeRepairGateUpBatchedGemmFloor`

| rows | gate | up | separate_sum | batched_ptr | ptr_delta | batched_strided | strided_delta |
|-----:|-----:|---:|-------------:|------------:|----------:|----------------:|--------------:|
| 4 | 0.4757 | 0.4759 | 0.9516 | 1.1748 | +0.2232 | 1.1831 | +0.2314 |
| 6 | 0.4840 | 0.4767 | 0.9606 | 1.1887 | +0.2280 | 1.1789 | +0.2183 |
| 8 | 0.4941 | 0.4833 | 0.9774 | 1.2078 | +0.2304 | 1.2189 | +0.2416 |

Conclusion: batched GEMM projection fusion is rejected for these rows.

## Gate/Up Parallel Stream Floor

Added a test-only direct bench:

```text
bench_ops/cuda/perf/DecodeRepairGateUpParallelGemmFloor
```

The bench measures one realistic pair at a time: gate and up are launched on separate CUDA streams and separate cuBLAS handles, and the timing stream waits for both projections before the next pair. This avoids turning the repeat loop into an unrealistic cross-layer pipeline.

Run command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  export MNN_ARTIFACT_ROOT="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_test" && \
  export LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
  MNN_BENCH_MLP_HIDDEN=3072 MNN_BENCH_MLP_INTER=8192 \
  MNN_BENCH_MLP_GEMM_WARMUP=20 MNN_BENCH_MLP_GEMM_REPEAT=80 \
  "$MNN_ARTIFACT_ROOT/bin/run_test.out" \
  bench_ops/cuda/perf/DecodeRepairGateUpParallelGemmFloor 2 2 1 0 2'
```

Results:

| rows | sequential pair ms | parallel pair ms | delta ms | speedup |
|-----:|-------------------:|-----------------:|---------:|--------:|
| 4 | 0.9848 | 0.9872 | +0.0024 | 0.998 |
| 6 | 0.9888 | 0.9874 | -0.0014 | 1.001 |
| 8 | 0.9808 | 0.9900 | +0.0092 | 0.991 |

The same bench was reused for attention k/v-style pairs by setting `inter=1024`:

| rows | sequential pair ms | parallel pair ms | delta ms | speedup |
|-----:|-------------------:|-----------------:|---------:|--------:|
| 4 | 0.1564 | 0.1635 | +0.0071 | 0.956 |
| 6 | 0.1531 | 0.1639 | +0.0108 | 0.934 |
| 8 | 0.1550 | 0.1626 | +0.0077 | 0.953 |

Conclusion: Jetson sm72 does not get useful overlap from running independent projection GEMMs on two streams. This is not a production direction for MLP gate/up or attention k/v pairs.

## Q/K/V Parallel Stream Floor

Added a test-only direct bench:

```text
bench_ops/cuda/perf/DecodeRepairQkvParallelGemmFloor
```

The bench measures attention q/k/v projections for the 3B shape as one realistic triple:

- q: `3072 -> 3072`
- k: `3072 -> 1024`
- v: `3072 -> 1024`

The parallel path launches q/k/v on three CUDA streams with three cuBLAS handles. A timing stream waits for all three projections before the next triple, so the repeat loop does not become an unrealistic cross-layer pipeline. Single q/k/v timings are also measured with CUDA events on the same cuBLAS stream.

Run command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  mkdir -p .cache/bench_ops/decode_repair_hkvd_20260629 && \
  export MNN_ARTIFACT_ROOT="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_test" && \
  export LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
  MNN_BENCH_MLP_HIDDEN=3072 MNN_BENCH_ATTN_KV=1024 \
  MNN_BENCH_MLP_GEMM_WARMUP=20 MNN_BENCH_MLP_GEMM_REPEAT=80 \
  "$MNN_ARTIFACT_ROOT/bin/run_test.out" \
  bench_ops/cuda/perf/DecodeRepairQkvParallelGemmFloor 2 2 1 0 2 \
  2>&1 | tee .cache/bench_ops/decode_repair_hkvd_20260629/decode_repair_qkv_parallel_gemm_floor.log'
```

Results:

| rows | q ms | k ms | v ms | individual sum ms | sequential triple ms | parallel triple ms | delta ms | speedup |
|-----:|-----:|-----:|-----:|------------------:|---------------------:|-------------------:|---------:|--------:|
| 4 | 0.2015 | 0.0665 | 0.0654 | 0.3334 | 0.3554 | 0.3517 | -0.0036 | 1.010 |
| 6 | 0.2002 | 0.0674 | 0.0656 | 0.3331 | 0.3523 | 0.3442 | -0.0080 | 1.023 |
| 8 | 0.2028 | 0.0686 | 0.0680 | 0.3395 | 0.3607 | 0.3482 | -0.0125 | 1.036 |

Conclusion: q/k/v three-stream overlap is too small to justify production complexity. Even the best rows=8 case saves only about `0.013 ms/layer`; this is far below the `0.2-0.3 ms/layer` direct-op target needed to materially reduce endpoint `x=3/5/7` TPOT. Reject this direction unless future production graph fusion can remove launches or memory traffic in a way this cuBLAS floor does not model.

## Simple Concat Fusion Floor

After multi-stream overlap was rejected, the remaining low-risk graph/export fusion question was whether simple concat projection can remove launches without increasing GEMM cost:

- MLP: replace separate `gate_proj` + `up_proj` with one `hidden_to_gateup_concat`, then run packed SwiGLU and `down_proj`.
- Attention: replace separate `q_proj` + `k_proj` + `v_proj` with one `hidden_to_qkv_concat`.

This is only a direct-op floor. Production graph/export work would still need shape/export changes and slicing/consumer updates, so the direct-op result must be clearly above noise before it is worth that work.

Focused `WeightOnlyConv` run:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  mkdir -p .cache/bench_ops/decode_repair_hkvd_20260629 && \
  export MNN_ARTIFACT_ROOT="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_test" && \
  export LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
  MNN_BENCH_WEIGHT_ONLY_HIDDEN=3072 MNN_BENCH_WEIGHT_ONLY_INTER=8192 MNN_BENCH_WEIGHT_ONLY_KV=1024 \
  MNN_BENCH_WEIGHT_ONLY_WARMUP=30 MNN_BENCH_WEIGHT_ONLY_REPEAT=200 \
  "$MNN_ARTIFACT_ROOT/bin/run_test.out" \
  bench_ops/cuda/perf/WeightOnlyConv 2 2 1 0 2 \
  2>&1 | tee .cache/bench_ops/decode_repair_hkvd_20260629/weight_only_conv_concat_focus.log'
```

Projection concat comparison:

| rows | gate+up separate ms | gateup concat ms | concat delta ms | q+2kv separate ms | qkv concat ms | concat delta ms |
|-----:|--------------------:|-----------------:|----------------:|------------------:|--------------:|----------------:|
| 4 | 0.9528 | 0.9506 | -0.0022 | 0.3324 | 0.3668 | +0.0344 |
| 6 | 0.9510 | 0.9504 | -0.0006 | 0.3408 | 0.3673 | +0.0265 |
| 8 | 0.9658 | 0.9589 | -0.0069 | 0.3453 | 0.3731 | +0.0278 |

Focused `PicDecodeMlp` run:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  mkdir -p .cache/bench_ops/decode_repair_hkvd_20260629 && \
  export MNN_ARTIFACT_ROOT="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_test" && \
  export LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
  MNN_BENCH_MLP_HIDDEN=3072 MNN_BENCH_MLP_INTER=8192 \
  MNN_BENCH_MLP_WARMUP=30 MNN_BENCH_MLP_REPEAT=200 \
  "$MNN_ARTIFACT_ROOT/bin/run_test.out" \
  bench_ops/cuda/perf/PicDecodeMlp 2 2 1 0 2 \
  2>&1 | tee .cache/bench_ops/decode_repair_hkvd_20260629/pic_decode_mlp_concat_focus.log'
```

MLP chain comparison:

| rows | split chain ms | packed chain ms | packed delta ms |
|-----:|---------------:|----------------:|----------------:|
| 4 | 1.5328 | 1.5077 | -0.0251 |
| 6 | 1.5434 | 1.5135 | -0.0299 |
| 8 | 1.5334 | 1.5420 | +0.0085 |

Conclusion:

- Attention q/k/v concat is slower than separate q/k/v projection for rows=4/6/8. Do not pursue simple qkv export fusion as the main Jetson CUDA decode repair optimization.
- MLP gate/up concat plus packed SwiGLU has only a marginal chain-level result: rows4/6 improve by about `0.025-0.030 ms/layer`, while rows8 regresses by `0.0085 ms/layer`. Even if rows4/6 generalized perfectly across all layers, this is far below the target needed to move endpoint `x=3/5/7` by `5-20 ms`. Do not spend production graph/export complexity on this simple concat form unless a later profile shows graph launch overhead, not GEMM work, is the dominant remaining cost.

## cuBLASLt Floor

`bench_ops/cuda/perf/DecodeRepairMlpGemmLtFloor`

| rows | gate | up | down | projection_sum |
|-----:|-----:|---:|-----:|---------------:|
| 4 | 0.4720 | 0.4827 | 0.5218 | 1.4765 |
| 6 | 0.4770 | 0.4881 | 0.5061 | 1.4713 |
| 8 | 0.4827 | 0.4763 | 0.5378 | 1.4968 |

Compared with `PicDecodeMlp` split chain:

| rows | split chain | cuBLASLt projection_sum | theoretical gap |
|-----:|------------:|------------------------:|----------------:|
| 4 | 1.5149 | 1.4765 | 0.0384 |
| 6 | 1.5422 | 1.4713 | 0.0709 |
| 8 | 1.5254 | 1.4968 | 0.0286 |

Conclusion: cuBLASLt is too close to the current chain to be a production endpoint. The gap is tens of microseconds per layer, not the `0.2-0.3 ms/layer` target.

## True-Case Fused MLP Candidate

Added a test-only CUDA Extra execution:

```text
PicGateUpSiluWeightOnly
```

It targets the actual decode repair MLP shape and fuses:

```text
input -> gate INT4 weight-only
input -> up INT4 weight-only
SiLU(gate) * up
```

The direct-op bench builds a real external INT4 `.weight` file in exporter-like format, then constructs Extra ops with the same external metadata style as production weight-only Conv. The candidate only takes the fused scalar INT4 path when `KVMeta::pic_decode_repair_sparse_active` is true, fp16 CUDA is active, weights are INT4, and rows are in `[2, 8]`; otherwise it falls back to child `ConvFpAIntBExecution` gate/up plus `PicSiluMul`.

Run command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  mkdir -p .cache/bench_ops/decode_repair_hkvd_20260629 && \
  export MNN_ARTIFACT_ROOT="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_test" && \
  export LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
  MNN_BENCH_MLP_HIDDEN=3072 MNN_BENCH_MLP_INTER=8192 \
  MNN_BENCH_MLP_WARMUP=30 MNN_BENCH_MLP_REPEAT=200 \
  "$MNN_ARTIFACT_ROOT/bin/run_test.out" bench_ops/cuda/perf/PicDecodeMlp 2 2 1 0 2 \
  2>&1 | tee .cache/bench_ops/decode_repair_hkvd_20260629/pic_decode_mlp_gateup_silu_weightonly.log'
```

Results:

| rows | split chain ms | packed chain ms | gateup weight-only chain ms | gateup silu weight-only chain ms | fused delta vs split |
|-----:|---------------:|----------------:|----------------------------:|---------------------------------:|---------------------:|
| 4 | 1.5210 | 1.5054 | 1.7254 | 1.7667 | +0.2457 |
| 6 | 1.5405 | 1.5031 | 2.1761 | 2.2112 | +0.6707 |
| 8 | 1.5290 | 1.5171 | 2.6188 | 2.6547 | +1.1257 |

Conclusion:

- The implementation proves a production-shaped MLP Extra op can be built and exercised with real external INT4 metadata.
- The scalar V14_MB-style fused INT4 kernel is much slower than the existing split chain for rows=4/6/8.
- Do not connect `PicGateUpSiluWeightOnly` to export or production decode repair. Keep it as a direct-op negative result unless a future tensor-core or fused-dequant implementation replaces the scalar kernel.

## Production Path Notes

`source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu` currently routes PIC decode repair sparse rows as follows:

- `picDecodeRepairSparse` comes from `KVMeta::pic_decode_repair_sparse_active`.
- `int4GemvBatchLimit = picDecodeRepairSparse ? 3 : 6`.
- Therefore rows=2/3 can use the tiny GEMV path, while rows=4/6/8 leave the tiny GEMV branch.
- The rows=4/6/8 path can match the static-dequant cuBLAS guard when fp16, static dequant cache, no runtime dequant, and `picRows45CublasMatches(...)` are true.
- If the guard does not match, it falls back to the prefill/batched GEMM path with dequantized FP16 weights.

This is consistent with the endpoint shape: `x=1` rows=2 remains near normal decode, while `x=3/5/7` share a compact dense plateau.

## Next Route Check

After rejecting q/k/v multi-stream overlap, the obvious remaining cuBLAS-level options were checked before starting another A/B:

- Jetson CUDA sysroot headers do not expose `cublasGemmGrouped*` / grouped GEMM APIs, so grouped q/k/v or gate/up/down GEMM is not available as a low-risk direct-op experiment in this toolchain.
- Prior Jetson logs already rejected rows45 cuBLAS compute/algo/policy retuning:
  - `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=16f`
  - `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=32f_fast16`
  - `rows45=down`
  - `rows45=0`
  - `algo=default`
- Therefore do not spend more iterations on cuBLAS env knobs. The next credible path must beat the existing static-dequant rows45 cuBLAS plateau with a new compact dense kernel/fusion, or remove projection launches through graph/export fusion without changing decode repair rows or PagedCache semantics.

## Decision

No source endpoint was changed from this run. Evidence is insufficient for a default production switch:

- layout convert cleanup is too small,
- packed gate/up is neutral or slower,
- batched gate/up is slower,
- parallel stream gate/up and k/v pairs do not overlap usefully,
- parallel stream q/k/v triples only save up to about `0.013 ms/layer`,
- simple attention q/k/v concat is slower than separate projections,
- simple MLP gate/up concat is only a marginal and unstable sub-`0.03 ms/layer` chain result,
- scalar `PicGateUpSiluWeightOnly` true-case MLP fusion is slower by `+0.2457 / +0.6707 / +1.1257 ms/layer` for rows=4/6/8,
- generic FP16 GEMM is usually slower,
- cuBLASLt only closes about `0.03-0.07 ms/layer`.

Next implementation should target one of:

1. a new INT4-native compact dense kernel for rows=4/6/8,
2. a true MLP fusion that reduces gate/up/down launch and memory traffic while preserving SwiGLU semantics,
3. a graph/export fusion that reduces projection launch count for decode repair compact rows,
4. or a validated attention projection fusion if it beats the current `hidden_to_hidden`, `hidden_to_kv`, and `hidden_to_qkv_concat` direct-op numbers by at least `0.2 ms/layer` aggregated over attention projections.

Any candidate must first beat the direct-op floor above before running a full endpoint `x=0/1/3/5/7` no-profile sweep.
