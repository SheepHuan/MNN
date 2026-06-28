# Jetson CUDA decode repair attention-rank capture context

## User Requirements

The decode repair experiment should test TPOT for normal decode and small repair budgets:

- Keep `x=0` as the normal decode speed baseline.
- Test `x=1/3/7` for `lagged_attention_hkvd`.
- Do not copy full `[heads, kv_len]` attention matrices to CPU.
- Maintain a compact ranking only:
  - `lastAttentionRankedPicLocalIndices`
  - optional source step index
  - optional candidate pool size
  - existing HKVD global ranking
- Attention weight/rank capture is only needed at one specified layer.
- qtile sparse attention is not the default path for small decode repair budgets; keep it experimental for larger active rows or batch-like shapes.

## Current Code State

Relevant source points:

```text
source/core/PagedKVMeta.hpp
  beginPicDecodeAttentionRankCapture(...)
  needsPicDecodeAttentionRankCapture(layerIdx)
  setPicDecodeAttentionRankResult(...)

transformers/pic_llm/engine/src/llm.cpp
  preparePicDecodeRepair(...)
  selectPicDecodeRepairLogicalIndices()
  forwardVecWithPicDecodeRepair(...)

source/backend/cuda/execution/PagedAttentionExecution.cu
  DecodeAttentionRankCaptureCUDA
  finishDecodeAttentionRankCaptureCUDA(...)
  runDecodeAttentionRankCUDA(...)
  pagedAttentionRowCompressedMaskKernel(...)
```

Layer gating is explicit:

```text
needsPicDecodeAttentionRankCapture(layerIdx)
  => pic_decode_attention_rank_active
     && layerIdx == pic_decode_attention_layer_idx
     && top_m > 0
     && pic_token_count > 0
```

The runtime prepares capture only for `lagged_attention_hkvd`:

```text
if selector == lagged_attention_hkvd
   && attention_layer_idx >= 0
   && attention_candidate_pool_size > 0:
    beginPicDecodeAttentionRankCapture(attention_layer_idx, ...)
else:
    finish/clear rank capture
```

Selection logic:

1. If last step has `lastAttentionRankedPicLocalIndices`, build an attention candidate pool.
2. Scan existing `rankedLogicalIndices` in HKVD order.
3. Select only tokens whose PIC local index is in the attention pool.
4. If insufficient, fall back to normal HKVD order.
5. Mark selected PIC locals as repaired to avoid repeated repair.

This matches the compact-ranking plan and avoids a full attention-weight host copy.

## CUDA Rank Capture Implementation Notes

Current source has two CUDA rank paths:

- Standalone path: `runDecodeAttentionRankCUDA`, which runs a separate PIC score kernel then top-k.
- Fused path: row-compressed decode attention can receive `decodeAttentionScores` and `atomicAdd` selected head scores into the compact PIC-local score workspace while computing decode attention.

The fused path is only requested when:

```text
picDecodeRecompute
&& mQuerySeqLen == attnLen
&& meta->needsPicDecodeAttentionRankCapture(layerIndex)
&& last query row can attend the complete PIC span
&& topM / picStart / picTokenCount are valid
```

Other layers pass a null `decodeAttentionScores` pointer and therefore do not capture rank.

Fused rank score semantics:

- Standalone `decodeAttentionPicScoreKernel` averages over used head/batch contributions.
- Fused row-compressed attention now multiplies each accumulated score by `1 / (valid_heads * batch)` before `atomicAdd`.
- Empty `attention_head_ids` means all heads; non-empty head ids are counted only if they are valid for `mNumHead`.
- If no valid head id remains, fused capture is disabled and the standalone rank path is used instead.

## Build And Sync

The Jetson cross CUDA build used the MNN build-artifacts flow. The script detected a stale cross build directory and removed it before reconfigure:

```text
cached toolchain:   .cache/toolchains/jetson-aarch64.toolchain.cmake
expected toolchain: .cache/toolchains/jetson-gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu-aarch64-linux.toolchain.cmake
action: remove .cache/build/mnn/jetson_cross and reconfigure
```

This matches the build-artifacts skill rule for avoiding toolchain/sysroot mixups.

The first CUDA compile after adding normalization failed because the fused path referenced a nonexistent member:

```text
PagedAttentionExecution.cu(3954): identifier "mPicDecodeAttentionHeadIds" is undefined
```

Fix:

```text
use mMeta->pic_decode_attention_head_ids for host-side valid head counting
syncDecodeAttentionHeadIds() already uses the same metadata source for device head ids
```

Final build and install completed:

```text
cmake --build .cache/build/mnn/jetson_cross --target MNN_Cuda_Main --parallel 1
cmake --build .cache/build/mnn/jetson_cross --target pic_server --parallel <half-cpu>
cmake --install .cache/build/mnn/jetson_cross --prefix .cache/output/mnn/artifacts/jetson_cross_cuda
```

Artifact checks:

```text
.cache/output/mnn/artifacts/jetson_cross_cuda/bin/pic_server: aarch64
.cache/output/mnn/artifacts/jetson_cross_cuda/lib/libMNN_Cuda_Main.so: aarch64
.cache/output/mnn/artifacts/jetson_cross_cuda/lib/libMNN.so: aarch64
```

Because the install step did not refresh all runtime files in the benchmark artifact prefix, the fresh build outputs were copied explicitly:

```text
.cache/build/mnn/jetson_cross/pic_server
  -> .cache/output/mnn/artifacts/jetson_cross_cuda/bin/pic_server

.cache/build/mnn/jetson_cross/source/backend/cuda/libMNN_Cuda_Main.so
  -> .cache/output/mnn/artifacts/jetson_cross_cuda/lib/libMNN_Cuda_Main.so
```

Artifact was synced to:

```text
jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

## Initial Profile Evidence

Run:

```text
run_id=decode_repair_profile_20260628_022816
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_profile_20260628_022816/pic_server.log
local_log=.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_profile_20260628_022816/pic_server.log
csv=.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_profile_20260628_022816/profile_decode.csv
```

Command shape:

```text
context=1024
mode=full-reuse
selector=lagged_attention_hkvd
repair_tokens=0,1,3,7
max_tokens=8
repeats=1
warm=0
```

Profile-mode TPOT:

```text
x=0 selector=none                  latency=1.000324 s TPOT=142.903 ms status=ok
x=1 selector=lagged_attention_hkvd latency=1.234279 s TPOT=176.326 ms status=ok
x=3 selector=lagged_attention_hkvd latency=1.842168 s TPOT=263.167 ms status=ok
x=7 selector=lagged_attention_hkvd latency=2.128835 s TPOT=304.119 ms status=ok
```

Attention-rank profile:

```text
decode_attention_rank layer=1 count=24 avg=482.8 us min=452 us max=719 us
fused_score_in_attention field: missing in all 24 lines
```

Interpretation:

- The available profile confirms rank capture is limited to the configured layer 1.
- The available profile does not confirm the fused rank-capture print path because the log format is the standalone rank format.
- A fresh profile with the current source should show `fused_score_in_attention=1` if the fused path is active.

Sparse decode attention profile, aggregated across `x=1/3/7` profile requests:

```text
layer 0  avg=1555.2 us count=24
layer 1  avg=2072.9 us count=24
layer 2  avg=1551.4 us count=24
layers 3-25 roughly 1534-1560 us avg
layer 26 avg=1492.5 us count=24, one zero-us outlier
layer 27 avg=1536.9 us count=23
```

The profile mode includes callback synchronization and should not be used as official TPOT. It is useful only to split attribution between decode attention, attention-rank capture, and graph dense ops.

## Fused Rank Smoke

After normalizing fused rank accumulation and rebuilding the Jetson CUDA artifact, a minimal PagedAttention-profile-only smoke was run:

```text
run_id=decode_repair_fused_smoke_20260628_024737
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_fused_smoke_20260628_024737/pic_server.log
local_log=.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_fused_smoke_20260628_024737/pic_server.log
csv=.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_fused_smoke_20260628_024737/profile_decode.csv
env=MNN_PAGED_ATTENTION_PROFILE=1
context=1024
mode=full-reuse
selector=lagged_attention_hkvd
repair_tokens=1
max_tokens=4
repeats=1
warm=0
```

Result:

```text
x=1 selector=lagged_attention_hkvd latency=0.522256 s TPOT=174.085 ms status=ok
decode_attention_rank layer=1 count=4 avg=284.2 us min=217 us max=430 us
fused_score_in_attention=1 count=4
fused_score_in_attention=0 count=0
rank lines on layers other than 1: 0
error lines: 0
```

Interpretation:

- The current source confirms fused rank capture is active.
- Rank capture remains restricted to the configured layer 1.
- No full attention matrix is copied; only compact top-M PIC local ranking is returned through metadata.

## Non-Profile TPOT Smoke

The formal profile env was disabled and a Jetson-only smoke ran `x=0/1/3/7`:

```text
run_id=decode_repair_tpot_smoke_20260628_025031
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_tpot_smoke_20260628_025031/pic_server.log
local_log=.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_tpot_smoke_20260628_025031/pic_server.log
csv=.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_tpot_smoke_20260628_025031/profile_decode.csv
env=<no profile env>
context=1024
mode=full-reuse
selector=lagged_attention_hkvd
repair_tokens=0,1,3,7
max_tokens=8
repeats=3
warm_repeats=1
```

Results:

```text
x=0 selector=none                  latency=0.610140 s TPOT=87.163 ms  status=ok
x=1 selector=lagged_attention_hkvd latency=1.111550 s TPOT=158.793 ms status=ok
x=3 selector=lagged_attention_hkvd latency=1.297219 s TPOT=185.317 ms status=ok
x=7 selector=lagged_attention_hkvd latency=1.455705 s TPOT=207.958 ms status=ok
```

Additional `x=5` smoke:

```text
run_id=decode_repair_tpot_x5_20260628_025625
env=<no profile env>
context=1024
mode=full-reuse
selector=lagged_attention_hkvd
repair_tokens=5
max_tokens=8
repeats=3
warm_repeats=1

x=5 selector=lagged_attention_hkvd latency=1.294957 s TPOT=184.994 ms status=ok
ERROR/failed/unsupported/target unavailable/async persistent: none
```

Server log scan:

```text
ERROR/failed/unsupported/target unavailable/async persistent: none
```

This is not a formal decode matrix because it is Jetson-only and only covers one context, but it confirms the current fused-rank artifact runs without profile overhead and keeps the required `x=0` baseline row.

## Optimization Direction For x=1/3/5/7

The pasted analysis file correctly says qtile is not the default optimization for small decode repair rows:

```text
x=0: normal decode, attnLen=1, qtile is not applicable
x=1/3/5/7: repair attention sees x+1 Q rows, i.e. 2/4/6/8 rows
qtile is intended for sparse prefill or larger active rows, not this tiny-row decode path
```

Current accepted priority:

1. Keep `x=0` as normal decode baseline.
2. Keep `lagged_attention_hkvd` fused rank capture in row-compressed decode attention.
3. Reduce non-compute overhead in the decode repair runtime.
4. Optimize tiny-row compute kernels for Linear/MLP and row-compressed attention.
5. Only evaluate qtile behind `MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1` for `x>=16/32/64` or batch-like rows.

Existing graph profile is synchronized and therefore not formal timing, but it is useful for attribution. The four profile requests showed:

```text
request=1 total=867.905 ms:  Convolution=430.322 ms, PicSparseAttention=208.888 ms, Raster/BinaryOp/While=169.837 ms
request=2 total=1015.657 ms: Convolution=487.224 ms, PicSparseAttention=247.537 ms, Raster/BinaryOp/While=190.413 ms
request=3 total=1592.956 ms: Convolution=961.477 ms, PicSparseAttention=327.321 ms, Raster/BinaryOp/While=217.403 ms
request=4 total=1874.145 ms: Convolution=978.231 ms, PicSparseAttention=548.565 ms, Raster/BinaryOp/While=236.913 ms
```

Interpretation:

- Fused rank capture removed the standalone rank kernel from the critical path, but `x>0` still runs an extra sparse-row forward for every decode token.
- The main compute cost is many tiny-row weight-only Linear calls: Q/K/V/O projections, MLP gate/up/down, and `lm_head`.
- The main non-compute cost is graph/expression fragmentation: many `Raster`, `BinaryOp`, and `While` calls for tiny tensors.
- Attention optimization alone cannot bring `x=1/3/5/7` close to `x=0`; dense and graph overhead must be reduced too.

Concrete next engineering actions:

1. Add a non-profile internal timer around `forwardVecWithPicDecodeRepair`: split `select_repair`, `embeddingForPicDecodeRepair`, `beginPicDecodeRecomputeRows`, `forwardRaw`, `rank_finish/topk/d2h`, and post-step bookkeeping. This proves how much overhead is outside kernels.
2. Add CUDA weight-only Conv profile for decode repair tiny M: split runtime dequant/cache hit, GEMM, activation, and output copy for `M=2/4/6/8`.
3. Fuse `gate_proj` and `up_proj` for tiny decode repair rows, then fuse `SiLU(gate) * up` before `down_proj`; MLP dominates enough that this is higher value than qtile.
4. Add or select a tiny-M CUDA Linear path for `M<=8` that avoids heavyweight CUTLASS/runtime-dequant overhead. Candidate shapes are `M in {1,2,4,6,8}`, `K=3072/8192`, `N=1024/3072/8192/128256`.
5. Reduce `Raster`/`BinaryOp`/`While` fragmentation in the decode repair subgraph: pre-allocate/cache masks and position ids, avoid repeated tiny tensor graph plumbing, and fuse residual/add/norm/activation where the graph already emits separate ops.
6. Keep row-compressed decode attention as the default for `x<=7`; optimize it directly for tiny Q rows by specializing `attnLen<=8`, avoiding generic sparse-query overhead, and keeping compact rank capture inside the same kernel.

## qtile Decision For Decode Repair

Do not enable qtile by default for `x<=7`.

Current intended behavior:

- `MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1` gates the experiment.
- `MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS` defaults to 16.
- If decode attention rank capture is active, qtile experiment is disabled for that layer so rank capture stays on the row-compressed path.

Rationale:

- Small repair budgets have tiny active rows and are sensitive to occupancy/launch overhead.
- Prior Jetson qtile work helped larger sparse-prefill budgets, but small decode repair rows should keep the existing row-compressed decode causal path until a separate A/B proves otherwise.

## Remaining Work

1. Run the formal Jetson + OrangePi decode matrix before merging any result into `benchmark_decode.csv`.
2. Add low-overhead decode repair timers so we can separate non-compute runtime overhead from CUDA kernel time without graph-profile synchronization.
3. Implement and A/B tiny-M dense/MLP improvements for `M=2/4/6/8`.
4. Keep qtile decode-repair work as an experimental branch for `x>=16/32/64` or batch-like decode repair, not as the default for small `x`.
