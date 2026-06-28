# Jetson PIC Optimize 2026-06-28 09 Context

## Carry-over From 08:00

The current source/default position is:

- Keep the accepted rows4-8 MLP-like cuBLAS route and static decode-hot dequant cache.
- Keep rejected directions rejected: square-projection cuBLAS admission, qtile q4/q8k8 variants, qtile-off, and `top_m/head0` as a default.
- Treat `top_m=8, attention_head_ids=0` only as an optional speed-profile config because it changes attention-rank selector semantics.

Accepted repeat3 baseline:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_smoke2_20260628_0710/summary.csv
x=0 none                  TPOT=86.377 ms
x=1 lagged_attention_hkvd TPOT=97.471 ms
x=3 lagged_attention_hkvd TPOT=123.602 ms
x=5 lagged_attention_hkvd TPOT=124.975 ms
x=7 lagged_attention_hkvd TPOT=126.360 ms
```

Latest profile conclusion:

```text
x=1 forward_raw_med=104.873ms attention=23.747ms conv=52.393ms rank=0.238ms
x=3 forward_raw_med=135.382ms attention=24.946ms conv=79.078ms rank=0.227ms
x=5 forward_raw_med=138.214ms attention=25.900ms conv=78.443ms rank=0.241ms
x=7 forward_raw_med=139.195ms attention=27.339ms conv=78.398ms rank=0.271ms
```

Graph profile for `x=7`:

```text
Convolution         788.956 ms
PicSparseAttention  268.379 ms
Raster              141.019 ms
BinaryOp             58.927 ms
While                44.598 ms
UnaryOp              24.107 ms
```

This supersedes the older qtile/rank-capture suspicion: rank capture/top-k is now about `0.004 ms/step` CPU-visible and is not the primary TPOT gap.

## PicSiluMul A/B

Rationale:

- Current benchmark model `.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary` was exported with `pic_decode_tiny_fusion=false`.
- Its graph has no `PicSiluMul`, `PicPackedSiluMul`, or `PicGateUpWeightOnly`.
- The A/B keeps selector semantics unchanged and only exposes an existing activation fusion in the graph.

Current A/B model:

```text
.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-silumul
```

Known issue from the first A/B export:

- `PicSiluMul` was present.
- `PicScoreAttention` / `PicSparseAttention` were serialized as `type=-1`.
- Jetson crashed at load in `MNN::SizeComputerSuite::search(OpType=-1)`.

Exporter source fix already added:

- `transformers/pic_llm/export/utils/mnn_converter.py` now has fallback rewriting for `type=-1/main_type=AttentionParam`.
- It infers `PicScoreAttention` from two outputs or five inputs.
- It infers `PicSparseAttention` for layers after `pic_recompute_score_layer_idx` when `pic_recompute_budget` is enabled.
- It also runs a post-`removeDupOps` fallback rewrite.

## 09:10 Build State

Current x64 build finished:

```text
.cache/build/mnn/x64/MNNConvert: ELF x86-64, BuildID=8f5f8c92faacb611f1f4e94f0edf391593534610
.cache/build/mnn/x64/libMNN.so:  ELF x86-64, BuildID=29fee8aa66d59a7bb62dccd8885fd7410d750f4a
```

Installed artifact state:

```text
.cache/output/mnn/artifacts/x64/bin/MNNConvert: stale, BuildID=3ee038000556caaf11aeace064ed9c63bb8528be, timestamp May 26
.cache/output/mnn/artifacts/x64/lib/libMNN.so: current, BuildID=29fee8aa66d59a7bb62dccd8885fd7410d750f4a
```

Decision:

- Do not use `.cache/output/mnn/artifacts/x64/bin/MNNConvert` for the next export.
- Use `.cache/build/mnn/x64/MNNConvert` directly, which links to the current build-tree converter deps and `libMNN.so`.

Next commands:

```text
MNNCONVERT_PATH=.cache/build/mnn/x64/MNNConvert
LD_LIBRARY_PATH=.cache/build/mnn/x64:.cache/build/mnn/x64/express:.cache/build/mnn/x64/source/backend/cuda:.cache/build/mnn/x64/tools/converter
```

Then re-export `pic-boundary-silumul`, verify no `type=-1`, sync to Jetson, and run the same repeat3 x=0/1/3/5/7 decode repair matrix.

## 09:19 PicSiluMul A/B Graph Repair

The full re-export with current build-tree `MNNConvert` was started:

```text
MNNCONVERT_PATH=.cache/build/mnn/x64/MNNConvert
dst=.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-silumul
log=.cache/logs/pic-llm-export/20260628_091223_AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-silumul.log
```

It was interrupted after about four minutes because it had not written any new files or log output and was still in the model-load / NFS read phase. The existing A/B artifacts already had the full `PicSiluMul` graph and `1.9G` weight file, so the faster path was to repair only the attention op serialization.

Observed bad graph before repair:

```text
minus1_count 27
('/layers.1/self_attn/PagedAttention', layer=1, inputs=5, outputs=2)
('/layers.2/self_attn/PagedAttention', layer=2, inputs=4, outputs=1)
...
('/layers.27/self_attn/PagedAttention', layer=27, inputs=4, outputs=1)
```

Repair applied:

- Backup: `llm.mnn.json.pre_attention_fix_0917`
- `layer1` fallback -> `PicScoreAttention`
- `layer2..27` fallback -> `PicSparseAttention`
- Repacked with `.cache/build/mnn/x64/MNNConvert`
- Re-read `llm.mnn` back to JSON with the same converter

Validation after roundtrip:

```text
minus1_count      0
PagedAttention    1
PicScoreAttention 1
PicSparseAttention 26
PicSiluMul text hits 28
llm_config.json: paged_attention=true, pic_recompute_budget=true, score_layer=1, pic_decode_tiny_fusion=true
```

This is now safe to sync to Jetson for a load smoke and repeat3 decode-repair benchmark.

## 09:25 PicSiluMul A/B Benchmark

Remote sync note:

- The Jetson already had the complete `llm.mnn.weight` from the previous failed A/B load attempt.
- A first rsync began retransmitting the unchanged `1.9G` weight because mtime differed; it was interrupted after confirming the remote weight size matched.
- Final sync excluded `llm.mnn.weight` and only updated the repaired graph/config/tokenizer files.

Jetson load smoke:

```text
GET /v1/models -> 200
id=llama-pic
```

Benchmark:

```text
RUN_ID=decode_repair_silumul_20260628_09
summary=.cache/mnn-pic-benchmark/decode_repair_silumul_20260628_09/summary.csv
```

Result:

```text
x=0 none                  TPOT=87.545 ms
x=1 lagged_attention_hkvd TPOT=98.721 ms
x=3 lagged_attention_hkvd TPOT=123.097 ms
x=5 lagged_attention_hkvd TPOT=125.309 ms
x=7 lagged_attention_hkvd TPOT=127.290 ms
```

Delta vs accepted baseline:

```text
x=0 86.377 -> 87.545 ms  +1.168 ms  +1.35%
x=1 97.471 -> 98.721 ms  +1.250 ms  +1.28%
x=3 123.602 -> 123.097 ms -0.505 ms -0.41%
x=5 124.975 -> 125.309 ms +0.335 ms +0.27%
x=7 126.360 -> 127.290 ms +0.930 ms +0.74%
```

Decision:

- Reject `pic_decode_tiny_fusion/PicSiluMul` as a default performance change for decode repair.
- It fixes no selector or compute semantics and does not move TPOT toward x=0.
- Continue by confirming whether `PicSiluMul` dispatches to CUDA fusion; if yes, the remaining dominant overhead is elsewhere in compact dense/attention/Raster/graph launch.

## 09:34 GateUp A/B Benchmark

Rationale:

- `PicSiluMul` alone had no stable TPOT gain.
- Existing CUDA has `PicGateUpWeightOnlyExecution`, including a decode-repair active rows 2..8 scalar fused path.
- This A/B tested the existing path without re-exporting weights by rewriting the graph only.

Local artifact:

```text
.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-gateup
```

Construction:

- Started from the repaired `pic-boundary-silumul` graph and hardlinked the same `llm.mnn.weight`.
- For each of 28 layers, replaced the gate Conv op with `PicGateUpWeightOnly` Extra that outputs the original gate/up Conv tensors.
- Removed the standalone up Conv op.
- Kept existing post-convert/post-reshape and `PicSiluMul`.

Graph validation:

```text
changed_layers 28
ops 3388 -> 3360
minus1 0
PagedAttention 1
PicScoreAttention 1
PicSparseAttention 26
PicGateUpWeightOnly 28
PicSiluMul 28
mlp_convs 28
```

Remote sync:

- Reused the same Jetson weight via hardlink from the `silumul` directory.
- Synced only non-weight files.
- `/v1/models` load smoke passed.

Benchmark:

```text
RUN_ID=decode_repair_gateup_20260628_09
summary=.cache/mnn-pic-benchmark/decode_repair_gateup_20260628_09/summary.csv
```

Result:

```text
x=0 none                  TPOT=86.780 ms
x=1 lagged_attention_hkvd TPOT=98.442 ms
x=3 lagged_attention_hkvd TPOT=130.704 ms
x=5 lagged_attention_hkvd TPOT=146.296 ms
x=7 lagged_attention_hkvd TPOT=162.654 ms
```

Delta vs accepted baseline:

```text
x=0  +0.403 ms  +0.47%
x=1  +0.971 ms  +1.00%
x=3  +7.103 ms  +5.75%
x=5 +21.322 ms +17.06%
x=7 +36.294 ms +28.72%
```

Decision:

- Reject this `PicGateUpWeightOnly` path as a default optimization.
- The regression matches the existing Jetson note that V14/MB scalar GEMV-style low-batch compact MLP paths are not suitable as defaults.
- Do not pursue gate/up scalar fusion further unless a new tensor-core implementation replaces it.

## 09:43 Rows2 CuBLAS A/B

Rationale:

- Accepted default uses rows4-8 cuBLAS for decode repair MLP-like INT4 linears.
- `x=1` has active rows=2 and still uses V14_MB tiny GEMV.
- Tested whether rows=2 should also skip GEMV and use the static-dequant + cuBLAS path.

Implementation:

- Temporarily added `MNN_CUDA_PIC_INT4_ROWS2_CUBLAS=1`.
- When enabled, decode repair `int4GemvBatchLimit` became `1`, and `picRows45CublasMatches()` admitted batch>=2.
- Cross-built Jetson artifact and started server with the env.
- After the A/B result, reverted the local source experiment.

Build:

```text
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1
artifact=.cache/output/mnn/artifacts/jetson
remote artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
```

Build note:

- The script detected stale Jetson cross toolchain/sysroot cache and rebuilt from a clean `.cache/build/mnn/jetson_cross`.
- `pic_server`, `libMNN.so`, `libMNN_Cuda_Main.so`, and `libpic_llm.so` were aarch64.

Benchmark:

```text
RUN_ID=decode_repair_rows2_cublas_20260628_09
summary=.cache/mnn-pic-benchmark/decode_repair_rows2_cublas_20260628_09/summary.csv
```

Result:

```text
x=0 none                  TPOT=88.834 ms
x=1 lagged_attention_hkvd TPOT=136.646 ms
x=3 lagged_attention_hkvd TPOT=125.051 ms
x=5 lagged_attention_hkvd TPOT=127.432 ms
x=7 lagged_attention_hkvd TPOT=128.650 ms
```

Delta vs accepted baseline:

```text
x=0  +2.457 ms  +2.84%
x=1 +39.175 ms +40.19%
x=3  +1.450 ms  +1.17%
x=5  +2.458 ms  +1.97%
x=7  +2.291 ms  +1.81%
```

Decision:

- Reject rows=2 cuBLAS.
- Keep rows4-8 cuBLAS default.
- Do not add a rows2 env knob to production source; local experiment was reverted.

## 09:47 Post-Rebuild Default Control

After reverting the local rows2 source experiment, the already-synced Jetson artifact was restarted without `MNN_CUDA_PIC_INT4_ROWS2_CUBLAS`.

Runtime environment check:

```text
pic_server cmd: .cache/output/mnn/artifacts/jetson_cross_cuda/bin/pic_server --config .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json ...
environ: LD_LIBRARY_PATH=.cache/output/mnn/artifacts/jetson_cross_cuda/lib
```

Benchmark:

```text
RUN_ID=decode_repair_default_after_rebuild_20260628_09
summary=.cache/mnn-pic-benchmark/decode_repair_default_after_rebuild_20260628_09/summary.csv
```

Result:

```text
x=0 none                  TPOT=86.848 ms
x=1 lagged_attention_hkvd TPOT=98.709 ms
x=3 lagged_attention_hkvd TPOT=124.602 ms
x=5 lagged_attention_hkvd TPOT=126.723 ms
x=7 lagged_attention_hkvd TPOT=127.632 ms
```

Delta vs accepted baseline:

```text
x=0 +0.471 ms +0.55%
x=1 +1.237 ms +1.27%
x=3 +1.000 ms +0.81%
x=5 +1.748 ms +1.40%
x=7 +1.273 ms +1.01%
```

Decision:

- Default path is clean enough for further profile work.
- This confirms the rows2 failure was caused by the env-gated A/B, not by persistent server state.
- Next profile target remains default `x=7`: break down `Convolution`, `PicSparseAttention`, `Raster`, and graph launch/elementwise overhead after rejecting Silu/GateUp/rows2 cuBLAS.

## 09:55 Continue Optimization Checkpoint

User request:

```text
先记录日志，然后继续优化
```

The active goal was checked and still points to the pasted text file plus hourly optimization logging. The pasted qtile sparse-attention analysis was re-read before continuing. Key carry-over:

```text
qtile is not suitable as the main decode-repair optimization for x<=7.
x=0 has attnLen=1.
decode repair computes only x+1 Q rows: 2/4/6/8 rows for x=1/3/5/7.
Current CUDA routes picDecodeRecompute to decode_causal_attention row-compressed path.
The qtile selection path explicitly excludes picDecodeRecompute.
The priority is to profile decode_causal_attention, rank capture, and weight-only Conv/MLP.
```

Current accepted default remains:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_smoke2_20260628_0710/summary.csv
x=0 none                  TPOT=86.377 ms
x=1 lagged_attention_hkvd TPOT=97.471 ms
x=3 lagged_attention_hkvd TPOT=123.602 ms
x=5 lagged_attention_hkvd TPOT=124.975 ms
x=7 lagged_attention_hkvd TPOT=126.360 ms
```

The default post-rebuild control is close enough for new profiling:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_default_after_rebuild_20260628_09/summary.csv
x=0 none                  TPOT=86.848 ms
x=1 lagged_attention_hkvd TPOT=98.709 ms
x=3 lagged_attention_hkvd TPOT=124.602 ms
x=5 lagged_attention_hkvd TPOT=126.723 ms
x=7 lagged_attention_hkvd TPOT=127.632 ms
```

Profile run started from that default server:

```text
server_env=MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=1000
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_profile_x7_20260628_09.log
summary=.cache/mnn-pic-benchmark/decode_repair_profile_x7_20260628_09/summary.csv
x=7 lagged_attention_hkvd generated_tokens=4 decode_latency_s=0.80355 TPOT=267.850 ms status=ok
```

This TPOT is not comparable to formal latency because graph profile enables debug callbacks and op-level synchronization. It is only for attribution. Immediate next action is to parse `MNN_PIC_GRAPH_PROFILE_TYPE` / top `MNN_PIC_GRAPH_PROFILE_OP` rows, then restart the Jetson server without profile env before any formal benchmark.

## 09:58 x=7 Graph Profile Parse

Type summary from:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_profile_x7_20260628_09.log
```

```text
MNN_PIC_GRAPH_PROFILE_TYPE rank=1  type=Convolution        total_ms=464.357 max_ms=62.060 calls=985
MNN_PIC_GRAPH_PROFILE_TYPE rank=2  type=PicSparseAttention total_ms=151.778 max_ms=14.206 calls=130
MNN_PIC_GRAPH_PROFILE_TYPE rank=3  type=Raster             total_ms=72.880  max_ms=0.112  calls=2298
MNN_PIC_GRAPH_PROFILE_TYPE rank=4  type=BinaryOp           total_ms=30.914  max_ms=0.109  calls=985
MNN_PIC_GRAPH_PROFILE_TYPE rank=5  type=While              total_ms=22.600  max_ms=0.109  calls=700
MNN_PIC_GRAPH_PROFILE_TYPE rank=6  type=UnaryOp            total_ms=12.552  max_ms=0.080  calls=430
MNN_PIC_GRAPH_PROFILE_TYPE rank=7  type=LayerNorm          total_ms=8.588   max_ms=0.401  calls=285
MNN_PIC_GRAPH_PROFILE_TYPE rank=8  type=PicScoreAttention  total_ms=8.096   max_ms=3.143  calls=5
MNN_PIC_GRAPH_PROFILE_TYPE rank=9  type=PagedAttention     total_ms=7.559   max_ms=3.203  calls=5
MNN_PIC_GRAPH_PROFILE_TYPE rank=10 type=Cast               total_ms=5.106   max_ms=0.082  calls=145
```

Top operations:

```text
rank=1 /layers.0/self_attn/k_proj/Linear      Convolution total=62.607 ms max=62.060 calls=4 inputs=[8x3072x1x1] outputs=[8x1024x1x1]
rank=2 /layers.0/mlp/down_proj/Linear         Convolution total=26.387 ms max=24.404 calls=4 inputs=[8x8192x1x1] outputs=[8x3072x1x1]
rank=3 /lm/lm_head/Linear                     Convolution total=19.952 ms max=4.038  calls=5 inputs=[1x3072x1x1] outputs=[1x128256x1x1]
rank=4 /layers.2/self_attn/PagedAttention     PicSparseAttention total=14.206 ms max=14.206 calls=1 inputs=[1x1x24x128]|[1x1x8x128]|[1x1x8x128]|[1x1x1x1025]
rank=5 /layers.1/self_attn/PagedAttention     PicScoreAttention total=4.953 ms max=1.362 calls=4 inputs=[1x8x24x128]|[1x8x8x128]|[1x8x8x128]|[1x1x8x1]|[1]
rank=6 /layers.0/self_attn/PagedAttention     PagedAttention total=4.356 ms max=1.222 calls=4
rank=7..32 per-layer PicSparseAttention        roughly 3.96-4.33 ms total over 4 calls each
rank=35..117 per-layer MLP gate/up/down Conv   roughly 2.31-2.80 ms total over 4 calls each
```

Interpretation:

- The type order is stable and matches previous conclusions: `Convolution` is first, `PicSparseAttention` second, `Raster` third.
- The rank-1 layer0 `k_proj` and rank-2 layer0 `down_proj` include single-call cold spikes. They should not drive a default optimization unless reproduced in non-profile TPOT or warm profile.
- MLP linears for active rows=8 are otherwise around `0.58-0.75 ms/call` each under graph-profile synchronization; attention is around `1.0 ms/call/layer`.
- A second warm profile attempt using the same profile server failed with `/v1/prefill/text HTTP 502`, and `/v1/models` then timed out. Treat the profile server state as dirty and restart without profile env before any formal TPOT.

Next default-safe optimization choices:

- Re-check whether the current accepted rows4-8 cuBLAS path is entered for all MLP-like linears but not accidentally hurting q/k/v projections.
- Investigate layout `Raster` around `Linear_raster_0` and MLP `Mul_output_0_raster_0`, because there are 2298 calls and no individual call is large.
- Investigate `PicSparseAttention` tiny-row decode path only with event-level profiling or a very small kernel A/B; qtile remains rejected for x<=7.

## 09:59 Rows4-8 cuBLAS compute=16f A/B

Purpose:

- Test a no-source-change implementation knob in the existing rows4-8 cuBLAS path.
- Scope is only MLP-like INT4 rows4-8 linears in decode repair. It should not change selector semantics.

Server:

```text
env=LD_LIBRARY_PATH=.cache/output/mnn/artifacts/jetson_cross_cuda/lib
env+=MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=16f
log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_rows45_compute16f_20260628_09.log
```

Benchmark:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_compute16f_20260628_09/summary.csv
contexts=1024
selector=lagged_attention_hkvd
repair_tokens=0,1,3,5,7
max_tokens=8
repeats=3
warm_repeats=1
```

Result:

```text
x=0 none                  TPOT=85.890 ms
x=1 lagged_attention_hkvd TPOT=96.620 ms
x=3 lagged_attention_hkvd TPOT=122.585 ms
x=5 lagged_attention_hkvd TPOT=125.056 ms
x=7 lagged_attention_hkvd TPOT=127.378 ms
```

Delta vs accepted baseline:

```text
x=0 86.377 -> 85.890 ms  -0.487 ms  -0.56%
x=1 97.471 -> 96.620 ms  -0.852 ms  -0.87%
x=3 123.602 -> 122.585 ms -1.017 ms -0.82%
x=5 124.975 -> 125.056 ms +0.081 ms +0.07%
x=7 126.360 -> 127.378 ms +1.018 ms +0.81%
```

Decision:

- Not a default win. The direction is promising for low repair counts but fails the x=7 target.
- Continue with `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=fast16` before touching source.

## 09:59 Rows4-8 cuBLAS fast16 A/B

Server:

```text
env=LD_LIBRARY_PATH=.cache/output/mnn/artifacts/jetson_cross_cuda/lib
env+=MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=fast16
log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_rows45_fast16_20260628_09.log
```

Benchmark:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_fast16_20260628_09/summary.csv
contexts=1024
selector=lagged_attention_hkvd
repair_tokens=0,1,3,5,7
max_tokens=8
repeats=3
warm_repeats=1
```

Result:

```text
x=0 none                  TPOT=86.914 ms
x=1 lagged_attention_hkvd TPOT=97.692 ms
x=3 lagged_attention_hkvd TPOT=123.936 ms
x=5 lagged_attention_hkvd TPOT=125.363 ms
x=7 lagged_attention_hkvd TPOT=126.861 ms
```

Delta vs accepted baseline:

```text
x=0 +0.537 ms +0.62%
x=1 +0.221 ms +0.23%
x=3 +0.334 ms +0.27%
x=5 +0.389 ms +0.31%
x=7 +0.501 ms +0.40%
```

Delta vs post-rebuild control:

```text
x=0 +0.066 ms +0.08%
x=1 -1.016 ms -1.03%
x=3 -0.666 ms -0.53%
x=5 -1.360 ms -1.07%
x=7 -0.772 ms -0.60%
```

Decision:

- Not a default win versus accepted baseline.
- The difference versus post-rebuild control suggests some run-to-run noise; require a stronger margin before changing default.
- Next no-source A/B: `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=down`, to test whether keeping gate/up on rows4-8 cuBLAS hurts high repair counts.

## 09:59 Rows4-8 cuBLAS down-only A/B

Server:

```text
env=LD_LIBRARY_PATH=.cache/output/mnn/artifacts/jetson_cross_cuda/lib
env+=MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=down
log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_rows45_down_20260628_09.log
```

Result:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_down_20260628_09/summary.csv
x=0 none                  TPOT=86.198 ms
x=1 lagged_attention_hkvd TPOT=97.357 ms
x=3 lagged_attention_hkvd TPOT=129.307 ms
x=5 lagged_attention_hkvd TPOT=131.362 ms
x=7 lagged_attention_hkvd TPOT=132.969 ms
```

Delta vs accepted baseline:

```text
x=0 -0.179 ms -0.21%
x=1 -0.114 ms -0.12%
x=3 +5.705 ms +4.62%
x=5 +6.388 ms +5.11%
x=7 +6.610 ms +5.23%
```

Decision:

- Reject down-only. Gate/up rows4-8 cuBLAS is needed for x=3/5/7.
- New suspicion: the current dimension rule also admits Llama 3B attention k/v projection `3072 -> 1024` because `minDim=1024` and `maxDim=3072`.
- Next source A/B: add a temporary env threshold for rows4-8 cuBLAS `minDim`, test `minDim=2048` to exclude k/v while retaining MLP gate/up/down.

## 09:59 Rows4-8 cuBLAS minDim=2048 A/B

Temporary source change:

```text
source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_MIN_DIM default=1024
test env value=2048
```

Purpose:

- Exclude Llama 3B attention k/v projection `3072 -> 1024` from rows4-8 cuBLAS.
- Retain MLP gate/up/down because their min dimension is `3072`.

Build and sync:

```text
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1
artifact=.cache/output/mnn/artifacts/jetson
remote artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
file check: pic_server/libMNN/libMNN_Cuda_Main/libpic_llm are ARM aarch64
```

Result:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_mindim2048_20260628_09/summary.csv
x=0 none                  TPOT=88.632 ms
x=1 lagged_attention_hkvd TPOT=97.884 ms
x=3 lagged_attention_hkvd TPOT=124.003 ms
x=5 lagged_attention_hkvd TPOT=125.039 ms
x=7 lagged_attention_hkvd TPOT=127.005 ms
```

Delta vs accepted baseline:

```text
x=0 +2.255 ms +2.61%
x=1 +0.412 ms +0.42%
x=3 +0.401 ms +0.32%
x=5 +0.064 ms +0.05%
x=7 +0.645 ms +0.51%
```

Decision:

- Reject as a default. Excluding k/v projection from rows4-8 cuBLAS does not beat the accepted baseline.
- Remove the temporary env hook after this A/B unless a later combined test unexpectedly needs it.

## 09:59 Rows4-8 cuBLAS algo=default A/B

Server:

```text
env=LD_LIBRARY_PATH=.cache/output/mnn/artifacts/jetson_cross_cuda/lib
env+=MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO=default
log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_rows45_algo_default_20260628_09.log
```

Run 1:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_algo_default_20260628_09/summary.csv
x=0 none                  TPOT=86.027 ms
x=1 lagged_attention_hkvd TPOT=97.740 ms
x=3 lagged_attention_hkvd TPOT=123.310 ms
x=5 lagged_attention_hkvd TPOT=125.052 ms
x=7 lagged_attention_hkvd TPOT=125.372 ms
```

Run 2:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_algo_default_repeat2_20260628_09/summary.csv
x=0 none                  TPOT=85.202 ms
x=1 lagged_attention_hkvd TPOT=97.524 ms
x=3 lagged_attention_hkvd TPOT=122.894 ms
x=5 lagged_attention_hkvd TPOT=124.106 ms
x=7 lagged_attention_hkvd TPOT=126.189 ms
```

Mean vs accepted baseline:

```text
x=0 86.377 -> 85.615 ms  -0.762 ms -0.88%
x=1 97.471 -> 97.632 ms  +0.161 ms +0.16%
x=3 123.602 -> 123.102 ms -0.500 ms -0.40%
x=5 124.975 -> 124.579 ms -0.395 ms -0.32%
x=7 126.360 -> 125.780 ms -0.579 ms -0.46%
```

Decision:

- This is the best current rows4-8 cuBLAS knob: small but repeatable win on x=3/5/7 and x=0, tiny x=1 noise regression.
- Convert to source default by changing `picRows45CublasAlgoPolicy()` null/default behavior to `CUBLAS_GEMM_DEFAULT`.
- Remove the rejected `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_MIN_DIM` temporary hook.
- Rebuild/sync and verify no-env default TPOT.

## 09:59 algo=default Source Verification Rejected

Source default attempt:

```text
picRows45CublasAlgoPolicy():
  no env / "default" -> CUBLAS_GEMM_DEFAULT
  "tensor"           -> CUBLAS_GEMM_DEFAULT_TENSOR_OP

Rejected minDim env hook removed.
```

No-env verification run 1:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_algo_default_default_20260628_09/summary.csv
x=0 none                  TPOT=87.085 ms
x=1 lagged_attention_hkvd TPOT=98.631 ms
x=3 lagged_attention_hkvd TPOT=125.657 ms
x=5 lagged_attention_hkvd TPOT=126.729 ms
x=7 lagged_attention_hkvd TPOT=129.203 ms
```

No-env verification run 2:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_algo_default_default_repeat2_20260628_09/summary.csv
x=0 none                  TPOT=86.720 ms
x=1 lagged_attention_hkvd TPOT=97.619 ms
x=3 lagged_attention_hkvd TPOT=122.617 ms
x=5 lagged_attention_hkvd TPOT=125.228 ms
x=7 lagged_attention_hkvd TPOT=127.817 ms
```

Mean vs accepted baseline:

```text
x=0 +0.526 ms +0.61%
x=1 +0.653 ms +0.67%
x=3 +0.535 ms +0.43%
x=5 +1.004 ms +0.80%
x=7 +2.151 ms +1.70%
```

Decision:

- Reject source default change. It does not reproduce the env A/B and regresses the primary x=7 target.
- Restore tensor-op algo as source default.
- Keep `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO=default` as a diagnostic env only.
