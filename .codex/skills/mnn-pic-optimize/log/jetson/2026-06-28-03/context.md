# Jetson CUDA decode repair tiny-row optimization context

## Starting State

The current implementation already matches the compact-ranking design for `lagged_attention_hkvd`:

- `lastAttentionRankedPicLocalIndices` stores compact PIC-local rank results from the previous decode step.
- CPU selection scans the existing HKVD logical ranking and filters through the attention candidate pool.
- If the attention-filtered result is short, selection falls back to the HKVD ranking.
- Selected PIC local tokens are marked repaired to avoid duplicate repair.
- CUDA captures attention ranking only on the configured layer through `PagedKVMeta::needsPicDecodeAttentionRankCapture(layerIndex)`.
- The fused capture path accumulates scores inside row-compressed decode attention and returns only compact top-M PIC local indices, not full attention matrices.

Recent smoke evidence:

```text
decode_repair_fused_smoke_20260628_024737:
  decode_attention_rank layer=1 count=4 avg_us=284.2 min=217 max=430
  fused_score_in_attention=1 count=4
  no rank lines on other layers
  no error lines

decode_repair_tpot_smoke_20260628_025031:
  x=0 none                  TPOT=87.163 ms  status=ok
  x=1 lagged_attention_hkvd TPOT=158.793 ms status=ok
  x=3 lagged_attention_hkvd TPOT=185.317 ms status=ok
  x=7 lagged_attention_hkvd TPOT=207.958 ms status=ok

decode_repair_tpot_x5_20260628_025625:
  x=5 lagged_attention_hkvd TPOT=184.994 ms status=ok
```

## Active Bottleneck Hypothesis

For `x=1/3/5/7`, decode repair computes `x + 1` rows per decode step: `2/4/6/8` rows. qtile sparse attention is not expected to help this default path because it is designed for larger active-row sparse prefill or batch-like work. It remains gated behind:

```text
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=16
```

The useful next split is:

- Non-compute/runtime overhead:
  - repair token selection
  - embedding preparation
  - `PagedKVMeta` decode-recompute row setup
  - rank finish/top-k/D2H
  - post-step repaired-token bookkeeping
- Compute overhead:
  - row-compressed `PicSparseAttention` for `attnLen<=8`
  - tiny-row weight-only Linear/MLP calls
  - residual/norm/activation tiny tensor graph fragments

The synchronized graph profile from the previous hour showed large `Convolution`, `PicSparseAttention`, and `Raster/BinaryOp/While` buckets, so attention-only changes cannot make `x>0` approach `x=0`.

## This-Hour Plan

1. Add a gated internal timer around `Llm::forwardVecWithPicDecodeRepair()` with low overhead when disabled.
2. Keep the timer out of formal results unless explicitly enabled by `MNN_PIC_DECODE_REPAIR_PROFILE=1`.
3. Use the timer to verify whether the next major gap is outside `forwardRaw` or inside the model graph.
4. Inspect CUDA attention and weight-only Conv code paths for a low-risk tiny-row specialization or more precise per-component profiler.
5. Rebuild and run Jetson smoke after any code change that affects runtime behavior.

## Logging Rule

Continue appending implementation notes, compile results, and benchmark runs under this hour directory until the next hour boundary. Do not merge any smoke-only results into `benchmark_decode.csv`; formal Jetson and OrangePi decode matrix remains a separate gate.

## Implemented Low-Overhead Runtime Timer

Source change:

```text
transformers/pic_llm/engine/src/llm.cpp
  MNN_PIC_DECODE_REPAIR_PROFILE
  picDecodeRepairProfileLog(...)
  Llm::forwardVecWithPicDecodeRepair(...)
```

The timer is disabled by default through a cached env flag. When enabled it prints one line per decode repair step:

```text
MNN_PIC_DECODE_REPAIR_PROFILE step=... selector=... repair_rows=... sparse_rows=...
  build_rows_ms=...
  begin_recompute_ms=...
  embedding_ms=...
  mask_pos_ms=...
  forward_raw_ms=...
  validate_ms=...
  rank_result_ms=...
  finish_sparse_ms=...
  bookkeeping_ms=...
  total_ms=...
```

Build:

```text
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=96
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

result: success
artifact: .cache/output/mnn/artifacts/jetson
synced to Jetson fixed benchmark artifact path:
  /home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda
```

The build script removed stale `.cache/build/mnn/jetson_cross` first because the cached compiler/sysroot fields were empty or mismatched. This matches the build-artifacts stale cross-build rule.

## Runtime Profile Smoke

Run:

```text
run_id=decode_repair_runtime_profile_20260628_031546
server_env=MNN_PIC_DECODE_REPAIR_PROFILE=1
context=1024
mode=full-reuse
selector=lagged_attention_hkvd
repair_tokens=0,1,3,5,7
max_tokens=8
repeats=1
warm=0
csv=.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_runtime_profile_20260628_031546/profile_decode.csv
log=.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_runtime_profile_20260628_031546/pic_server.log
```

CSV:

```text
x=0 none                  TPOT=100.419 ms status=ok
x=1 lagged_attention_hkvd TPOT=130.609 ms status=ok
x=3 lagged_attention_hkvd TPOT=202.247 ms status=ok
x=5 lagged_attention_hkvd TPOT=214.524 ms status=ok
x=7 lagged_attention_hkvd TPOT=236.831 ms status=ok
```

Runtime timer aggregation from 32 decode repair profile lines:

```text
repair_rows count avg_total_ms avg_forward_raw_ms avg_non_forward_ms post_step0_total_ms post_step0_forward_ms post_step0_non_forward_ms
1           8     114.122      113.951            0.171              90.668              90.557                0.111
3           8     176.803      176.682            0.121              154.004             153.883               0.121
5           8     187.544      187.418            0.127              167.279             167.160               0.119
7           8     207.048      206.922            0.125              187.436             187.314               0.122
```

Conclusion:

- CPU-side decode repair overhead is not the current main issue.
- `embedding`, `mask_pos`, rank result copy, `finishSparseQuery`, and next-step selection together are roughly `0.11-0.12 ms/step` after step 0.
- The latency gap is inside `forwardRaw`: model graph execution, CUDA attention, weight-only Linear/MLP, and tiny tensor graph fragments.

## Attention Experiment: Reuse Exp In Row-Compressed V Accumulation

Hypothesis:

- `pagedAttentionRowCompressedMaskKernel` recomputes `expf(score - tileMax)` in the V accumulation loop for every head-dim lane.
- Storing tile weights in `scoreTile` after softmax sum could reduce repeated exp work for `head_dim=128`.

Patch tested:

```text
source/backend/cuda/execution/PagedAttentionExecution.cu
  scoreTile[tid] = localSum
  tileAcc += scoreTile[kk] * V
```

Build:

```text
cmake --build .cache/build/mnn/jetson_cross --target MNN_Cuda_Main --parallel 1
result: success
```

Non-profile smoke:

```text
run_id=decode_repair_attention_exp_reuseexp_20260628_032040
server_env=<none>
context=1024
repair_tokens=0,1,3,5,7
max_tokens=8
repeats=3
warm=1

x=0 none                  TPOT=100.565 ms status=ok
x=1 lagged_attention_hkvd TPOT=127.873 ms status=ok
x=3 lagged_attention_hkvd TPOT=215.821 ms status=ok
x=5 lagged_attention_hkvd TPOT=210.142 ms status=ok
x=7 lagged_attention_hkvd TPOT=247.070 ms status=ok
```

Decision:

- Regressed `x=3` and `x=7`; not stable enough to keep.
- Patch was reverted.
- CUDA backend was rebuilt and re-synced after revert.

Interpretation:

- On Xavier, reducing expf did not translate into TPOT improvement, likely due to added shared-memory traffic or changed scheduling in the row-compressed kernel.
- Do not reapply this as a default path without a deeper kernel A/B.

## Dense Profile

Run:

```text
run_id=decode_repair_conv_profile_20260628_032440
server_env=MNN_PAGED_ATTENTION_PROFILE=1
context=1024
repair_tokens=1,3,5,7
max_tokens=4
repeats=1
warm=0
```

This run is synchronized profile only, not TPOT.

Tiny-row `CUDAWeightOnlyConv profile` aggregation for `batch<=8`:

```text
batch=2:
  ic=3072 oc=1024  tiny_gemv avg=104.8 us
  ic=3072 oc=3072  tiny_gemv avg=185.4 us
  ic=3072 oc=8192  tiny_gemv avg=385.4 us
  ic=8192 oc=3072  tiny_gemv avg=368.9 us
  lm_head ic=3072 oc=128256 tiny_gemv avg=3981.9 us

batch=4:
  ic=3072 oc=1024  CUTLASS/static-dequant avg=244.7 us
  ic=3072 oc=3072  CUTLASS/static-dequant avg=512.4 us
  ic=3072 oc=8192  CUTLASS/static-dequant avg=786.9 us
  ic=8192 oc=3072  CUTLASS/static-dequant avg=753.6 us

batch=6:
  ic=3072 oc=1024  CUTLASS/static-dequant avg=241.9 us
  ic=3072 oc=3072  CUTLASS/static-dequant avg=506.8 us
  ic=3072 oc=8192  CUTLASS/static-dequant avg=787.8 us
  ic=8192 oc=3072  CUTLASS/static-dequant avg=756.5 us

batch=8:
  ic=3072 oc=1024  CUTLASS/static-dequant avg=240.8 us
  ic=3072 oc=3072  CUTLASS/static-dequant avg=511.3 us
  ic=3072 oc=8192  CUTLASS/static-dequant avg=795.7 us
  ic=8192 oc=3072  CUTLASS/static-dequant avg=767.7 us
```

Findings:

- `x=1` sparse rows are `batch=2` and hit tiny GEMV.
- `x=3/5/7` sparse rows are `batch=4/6/8` and use CUTLASS with static dequantized FP16 weights.
- Runtime dequant is not the bottleneck in this run: static dequant is active and `runtime_dequant=0`.
- Existing rows4/5 cublas path did not hit Llama3.2-3B MLP because it only matched hidden dim `2048`, while this model is `3072<->8192`.

## Cublas 3B Shape Extension Experiment

Hypothesis:

- Extend existing rows4/5 cublas fast path to also match `3072<->8192` MLP shapes.
- Keep default max rows at 5; reserve `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_MAX_ROWS=8` for a later x=5/7 experiment.

Patch tested:

```text
source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
  picRows45CublasMatches:
    ic == 8192 && oc in {2048,3072}
    ic in {2048,3072} && oc == 8192
```

Build:

```text
cmake --build .cache/build/mnn/jetson_cross --target MNN_Cuda_Main --parallel 1
result: success
```

Non-profile smoke:

```text
run_id=decode_repair_cublas3b_default_20260628_032751
server_env=<none>
context=1024
repair_tokens=0,1,3,5,7
max_tokens=8
repeats=3
warm=1

x=0 none                  TPOT=99.112 ms  status=ok
x=1 lagged_attention_hkvd TPOT=127.841 ms status=ok
x=3 lagged_attention_hkvd TPOT=201.577 ms status=ok
x=5 lagged_attention_hkvd TPOT=228.298 ms status=ok
x=7 lagged_attention_hkvd TPOT=291.696 ms status=ok
```

Decision:

- Regressed compared with the prior non-profile smoke and did not make `x=3` approach `x=0`.
- Patch was reverted.
- CUDA backend was rebuilt and re-synced after revert.

## Current Direction After Negative A/B

Do not keep either default compute experiment from this hour.

More promising next steps:

1. Add or reuse a direct single-op bench for 3B tiny-row Linear shapes, not full decode TPOT, so candidate GEMM paths can be tested without decode noise.
2. Investigate graph-level MLP fusion for tiny rows: `gate_proj + up_proj`, `SiLU(gate) * up`, then `down_proj`.
3. Inspect why batch=4/6/8 CUTLASS per Linear is almost flat with row count; this suggests launch/algorithm overhead dominates at tiny M.
4. Keep `x=0` baseline in every TPOT run.
5. Leave qtile default-off for `x<=7`.

## Continuation: Direct Tiny-Row MLP Bench

Timestamp:

```text
2026-06-28 03:38 CST
```

Goal for the next optimization slice:

- Build `run_test.out` in a separate Jetson cross test build directory with `MNN_BUILD_TEST=ON`, leaving the existing runtime artifact untouched.
- Sync the test artifact to Jetson and run direct CUDA op benches for Llama3.2-3B tiny rows `2/4/6/8`.
- Compare current `PicDecodeMlp` / `WeightOnlyConv` behavior against `DecodeRepairMlpGemmFloor` so the next production change is guided by isolated kernel data rather than decode TPOT noise.

Planned build/test knobs:

```text
BUILD_DIR=.cache/build/mnn/jetson_cross_test
INSTALL_PREFIX=.cache/output/mnn/artifacts/jetson_cross_cuda_test
MNN_TARGET_DEVICE=jetson
CROSS_COMPILE=ON
ENABLE_CROSS_CUDA=ON
CUDA_ARCHS=72
CMAKE_ARGS=-DMNN_BUILD_TEST=ON
BUILD_TARGET=run_test.out
BUILD_MNNCONVERT=0
INSTALL_AFTER_BUILD=1

MNN_BENCH_MLP_HIDDEN=3072
MNN_BENCH_MLP_INTER=8192
MNN_BENCH_WEIGHT_ONLY_HIDDEN=3072
MNN_BENCH_WEIGHT_ONLY_INTER=8192
MNN_BENCH_WEIGHT_ONLY_ROWS=2,4,6,8
```

Expected interpretation:

- If direct GEMM floor is much faster than the current weight-only MLP chain, investigate graph/MLP fusion or a corrected tiny-row tensor-core path.
- If direct GEMM floor is not better, focus on graph fragmentation, activation/residual/norm overhead, and attention/Linear launch count rather than reapplying the reverted cublas extension.

## Direct Tiny-Row MLP Bench Results

Run:

```text
run_id=decode_repair_mlp_20260628_0342
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/bench_ops/decode_repair_mlp_20260628_0342
local_log=.cache/bench_ops/decode_repair_mlp_20260628_0342
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda_test
```

Build:

```text
MNN_TARGET_DEVICE=jetson
CROSS_COMPILE=ON
ENABLE_CROSS_CUDA=ON
CUDA_ARCHS=72
BUILD_DIR=.cache/build/mnn/jetson_cross_test
INSTALL_PREFIX=.cache/output/mnn/artifacts/jetson_cross_cuda_test
CMAKE_ARGS=-DMNN_BUILD_TEST=ON
BUILD_TARGET=run_test.out
BUILD_MNNCONVERT=0
INSTALL_AFTER_BUILD=1
JOBS=96

result: success
run_test.out: ELF 64-bit LSB executable, ARM aarch64
libMNN_Cuda_Main.so: ELF 64-bit LSB shared object, ARM aarch64
```

Main direct-op command shape:

```text
run_test.out bench_ops/cuda/perf/PicDecodeMlp 2 2 1 none 2
run_test.out bench_ops/cuda/perf/DecodeRepairMlpGemmFloor 2 2 1 none 2
run_test.out bench_ops/cuda/perf/WeightOnlyConv 2 2 1 none 2

MNN_BENCH_MLP_HIDDEN=3072
MNN_BENCH_MLP_INTER=8192
MNN_BENCH_WEIGHT_ONLY_HIDDEN=3072
MNN_BENCH_WEIGHT_ONLY_INTER=8192
MNN_BENCH_WEIGHT_ONLY_ROWS=2,4,6,8
```

Clean direct-op results were sensitive to Jetson DVFS and run order, so the most useful numbers are the hot repeat and the path classification, not single cold-run rows.

Hot `PicDecodeMlp` repeat:

```text
rows=2 gate=0.3247 ms up=0.3270 ms silu=0.0084 ms down=0.3109 ms chain=0.9651 ms
rows=4 gate=0.5743 ms up=0.5794 ms silu=0.0039 ms down=0.5351 ms chain=1.7441 ms
rows=6 gate=0.5734 ms up=0.5828 ms silu=0.0045 ms down=0.5396 ms chain=1.7229 ms
rows=8 gate=0.5730 ms up=0.5826 ms silu=0.0049 ms down=0.5276 ms chain=1.7463 ms
```

Row1 comparison run:

```text
rows=1 chain=1.4246 ms
rows=2 chain=0.9497 ms
rows=4 chain=1.7625 ms
rows=6 chain=1.7572 ms
rows=8 chain=1.7351 ms
```

The row1 individual gate/up/down numbers in this run were inconsistent with the chain number, which reinforces that direct benches on Jetson need warm state and should be read as approximate attribution. The stable signal is that rows4/6/8 MLP is around `1.7 ms/layer`, enough to explain much of the x>=3 TPOT gap when multiplied across layers.

Weight-only single Linear clean run:

```text
rows=2 hidden_to_inter=0.3361 ms inter_to_hidden=0.3164 ms hidden_to_gateup_concat=0.6568 ms
rows=4 hidden_to_inter=0.5938 ms inter_to_hidden=0.5263 ms hidden_to_gateup_concat=1.0961 ms
rows=6 hidden_to_inter=0.5689 ms inter_to_hidden=0.5238 ms hidden_to_gateup_concat=1.0887 ms
rows=8 hidden_to_inter=0.5831 ms inter_to_hidden=0.5351 ms hidden_to_gateup_concat=1.1081 ms
```

FP16 GEMM floor:

```text
rows=2 oc_by_rows chain_no_silu=2.4359 ms rows_by_oc chain_no_silu=2.6801 ms
rows=4 oc_by_rows chain_no_silu=2.5391 ms rows_by_oc chain_no_silu=2.6772 ms
rows=6 oc_by_rows chain_no_silu=2.5567 ms rows_by_oc chain_no_silu=2.7592 ms
rows=8 oc_by_rows chain_no_silu=2.5790 ms rows_by_oc chain_no_silu=1.7348 ms
```

Interpretation:

- Current weight-only/static-dequant MLP is already better than plain FP16 cublas for rows2/4/6.
- rows8 `rows_by_oc` cublas is comparable to current chain, but decode repair needs x=1/3/5/7 all improve; it does not justify a default cublas replacement.
- `PicSiluMul` is not the issue; it is only about `0.004-0.008 ms`.
- Gate/up concat is not a clear win. In single-op `WeightOnlyConv`, `hidden_to_gateup_concat` is close to the sum of two separate projections, but not enough to explain the full TPOT gap.

## Negative A/B: Decode-Repair GEMV Limit 6

Temporary patch tested:

```text
source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
  int4GemvBatchLimit = picDecodeRepairSparse ? 6 : 6
```

Purpose:

- Force decode repair rows4/6 through V14_MB GEMV instead of the default static-dequant CUTLASS path.

Direct result:

```text
rows=2 chain=1.9001 ms
rows=4 chain=1.9529 ms
rows=6 chain=2.4382 ms
rows=8 chain=1.7957 ms
```

Decision:

- Regressed rows4/6 and made rows2 unstable/slower.
- Patch was reverted immediately.
- `ConvFpAIntBExecution.cu` diff is empty after revert.
- Test artifact `libMNN_Cuda_Main.so` was rebuilt and re-synced with the default `picDecodeRepairSparse ? 3 : 6` policy.

Restore check:

```text
profile rows4/6:
  op=conv_fpa_intb_1x1
  runtime_dequant=0
  static_dequant=1
  pic_compact_sm70=0
  pic_compact_tile=0
```

Updated conclusion:

- Keep qtile default-off for `x<=7`.
- Keep rows4/6 decode repair on static-dequant CUTLASS.
- Do not reapply the 3B rows4/5 cublas extension or V14_MB expansion as a default path.
- Remaining optimization needs to reduce graph/kernel work inside `forwardRaw`: fewer tiny Linear launches, a proven fused MLP path, or a more effective compact tensor-core/weight-only GEMM. CPU selector/rank bookkeeping is already below `0.2 ms/step`.

## Negative A/B: V14_MB OC Group 8

Temporary patch tested:

```text
source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
  constexpr int V14_OC = 8
```

Purpose:

- Reduce rows1/2 tiny GEMV block count by processing 8 output channels per block instead of 4.
- This targets `x=1`, where decode repair has sparse rows2 and still uses V14_MB GEMV.

Direct result:

```text
rows=1 chain=3.9643 ms
rows=2 chain=1.4111 ms
```

Comparison:

```text
default hot rows2 chain ~= 0.95-0.97 ms
```

Decision:

- Regressed rows2.
- Patch was reverted immediately.
- `ConvFpAIntBExecution.cu` diff is empty after revert.
- Test artifact `libMNN_Cuda_Main.so` was rebuilt and re-synced with default `V14_OC=4`.

Updated next step:

- Do not tune V14_MB by simply increasing OC grouping.
- The remaining dense-side target is not a trivial GEMV threshold or OC grouping change. It likely needs either graph-level MLP fusion or a new compact tensor-core/weight-only kernel proven against rows1/2/4/6/8 before any decode TPOT run.

## Continuation: Logits/LmHead Waste Check

Timestamp:

```text
2026-06-28 03:58 CST
```

Goal:

- Continue from the direct tiny-row MLP conclusion and look for non-MLP work inside `forwardRaw` that scales with decode repair rows.
- Keep the required decode smoke matrix anchored by `x=0 none`, then compare `x=1/3/5/7 lagged_attention_hkvd`.

Immediate hypothesis to test:

- Decode repair may be paying for final logits / `lm_head` over all sparse rows even though only the normal decode row is used for sampling.
- The relevant runtime path is `Llm::forwardVecWithPicDecodeRepair()` -> `forwardRaw(inputEmbeds, repairMask, repairPos)`.
- `forwardRaw()` currently chooses `logits_index=-1` for normal decode unless all-logits/spec/pad mode is active.
- The exported model slices `hidden_states[:, logits_index_long:, :]` before `self.lm(hidden_states)`, so if `logits_index=-1` is preserved for decode repair, `lm_head` should only see the last row.

Decision criteria:

- If source inspection or `MNN_PIC_GRAPH_PROFILE=1` confirms `lm_head` input is one row, the optimization target stays tiny-row model body work, especially MLP/Linear and graph fragments.
- If `lm_head` is still full sparse rows in decode repair, add a scoped runtime/export fix so decode repair keeps only the last row before `lm_head` while preserving all-logits modes.

Constraints:

- Do not save full attention weights; compact attention rank remains the only decode rank state.
- Attention rank capture stays on the configured layer only.
- Do not reapply rejected A/Bs: row-compressed attention reuse, 3B rows4/5 cublas extension, GEMV limit 6, or V14 OC group 8.
- Keep qtile default-off for `x<=7`.
