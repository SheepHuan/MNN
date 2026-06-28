# Jetson PIC Optimize 2026-06-28 09

## Decode Repair Continuation

Goal: keep reducing Jetson CUDA decode repair TPOT for `x=1/3/5/7 lagged_attention_hkvd` toward the `x=0 none` baseline, without changing selector semantics unless the result is explicitly marked as an optional speed-profile variant.

Accepted current repeat3 baseline:

```text
x=0 none                  TPOT=86.377 ms
x=1 lagged_attention_hkvd TPOT=97.471 ms
x=3 lagged_attention_hkvd TPOT=123.602 ms
x=5 lagged_attention_hkvd TPOT=124.975 ms
x=7 lagged_attention_hkvd TPOT=126.360 ms
```

The 08:00 profile shows rank capture is already compact/fused and not the main gap. The remaining gap is dominated by compact dense/MLP, attention, Raster, BinaryOp/UnaryOp, and graph launch overhead.

## 09:10 Checkpoint

- x64 `MNNConvert` target finished building from current source.
- Installed x64 `libMNN.so` is current, but `.cache/output/mnn/artifacts/x64/bin/MNNConvert` is still the stale converter binary.
- Continue the PicSiluMul A/B by using `.cache/build/mnn/x64/MNNConvert` directly, or by repairing the artifact converter before export.
- Next step: re-run the `pic-boundary-silumul` export/conversion, verify no `type=-1` nodes, sync to Jetson, then benchmark x=0/1/3/5/7 repeat3.

## 09:19 A/B Graph Fixed

- Full re-export was interrupted after no file progress; used the existing A/B graph and weight.
- Rewrote only the 27 `type=-1/main_type=AttentionParam` nodes, then repacked with `.cache/build/mnn/x64/MNNConvert`.
- Roundtrip validation now passes:

```text
type=-1:          0
PagedAttention:   1
PicScoreAttention: 1
PicSparseAttention: 26
PicSiluMul text hits: 28
```

Next: sync the repaired A/B model to Jetson, start `pic_server`, and run repeat3 decode repair TPOT.

## 09:25 PicSiluMul A/B Result

Load smoke passed on Jetson; the previous `OpType=-1` crash is fixed.

Repeat3 result:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_silumul_20260628_09/summary.csv
x=0 none                  TPOT=87.545 ms
x=1 lagged_attention_hkvd TPOT=98.721 ms
x=3 lagged_attention_hkvd TPOT=123.097 ms
x=5 lagged_attention_hkvd TPOT=125.309 ms
x=7 lagged_attention_hkvd TPOT=127.290 ms
```

Against the accepted baseline, this is not a default win:

```text
x=0 +1.168 ms
x=1 +1.250 ms
x=3 -0.505 ms
x=5 +0.335 ms
x=7 +0.930 ms
```

Decision: reject `pic_decode_tiny_fusion/PicSiluMul` as the current default decode-repair optimization. Continue by checking whether it actually dispatches to CUDA fusion; if it does, activation fusion is not the main remaining bottleneck.

## 09:34 GateUp A/B Result

Built a local graph-only A/B that wraps each layer's gate/up projections in `PicGateUpWeightOnly` while reusing the same `llm.mnn.weight`.

Validation:

```text
type=-1: 0
PicGateUpWeightOnly: 28
PicSiluMul: 28
MLP Convolution ops: 84 -> 28
```

Repeat3 result:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_gateup_20260628_09/summary.csv
x=0 none                  TPOT=86.780 ms
x=1 lagged_attention_hkvd TPOT=98.442 ms
x=3 lagged_attention_hkvd TPOT=130.704 ms
x=5 lagged_attention_hkvd TPOT=146.296 ms
x=7 lagged_attention_hkvd TPOT=162.654 ms
```

Decision: reject this gate/up fusion path. It confirms the existing warning that the batch<=8 scalar gate/up GEMV path is not a default optimization; it regresses high repair counts sharply.

## 09:43 Rows2 CuBLAS A/B Result

Tested an env-gated source A/B that lets decode-repair active rows=2 skip V14_MB GEMV and try the static-dequant + cuBLAS path.

Repeat3 result:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows2_cublas_20260628_09/summary.csv
x=0 none                  TPOT=88.834 ms
x=1 lagged_attention_hkvd TPOT=136.646 ms
x=3 lagged_attention_hkvd TPOT=125.051 ms
x=5 lagged_attention_hkvd TPOT=127.432 ms
x=7 lagged_attention_hkvd TPOT=128.650 ms
```

Decision: reject. rows=2 cuBLAS is much worse for `x=1` and slightly worse for other repair counts. The local source experiment was reverted; keep rows4-8 cuBLAS as the accepted default.

## 09:47 Default Control After Rebuild

Restarted Jetson server without `MNN_CUDA_PIC_INT4_ROWS2_CUBLAS`; `/proc/<pid>/environ` only showed `LD_LIBRARY_PATH`.

Repeat3 control:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_default_after_rebuild_20260628_09/summary.csv
x=0 none                  TPOT=86.848 ms
x=1 lagged_attention_hkvd TPOT=98.709 ms
x=3 lagged_attention_hkvd TPOT=124.602 ms
x=5 lagged_attention_hkvd TPOT=126.723 ms
x=7 lagged_attention_hkvd TPOT=127.632 ms
```

Decision: current default server is usable for further profiling. It is within about `0.5-1.7 ms` of the accepted baseline, and does not retain the rows2 failure path.

## 09:55 Continue Decode Repair Optimization

User asked to record the hourly log first, then continue optimizing `x=1/3/5/7` decode repair toward the `x=0 none` TPOT baseline.

The pasted qtile analysis was re-read. It reinforces the current direction:

- qtile sparse attention is not a good default target for `x<=7`, because decode repair only computes `x+1` Q rows.
- rank capture is already compact/fused enough to be a minor cost in current profile.
- The next default-safe targets are tiny-row decode repair `Convolution`/MLP, `PicSparseAttention`, layout `Raster`, and graph/elementwise overhead.

An `x=7` graph-profile run has completed:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_profile_x7_20260628_09/summary.csv
x=7 lagged_attention_hkvd TPOT=267.850 ms with MNN_PIC_GRAPH_PROFILE=1
```

This is profile-only and not formal latency, because graph profile synchronizes after ops. Next step: parse the remote profile log and use it to choose a small, semantics-preserving CUDA A/B.

## 09:58 x=7 Profile Parsed

Profile type summary:

```text
Convolution         464.357 ms calls=985
PicSparseAttention  151.778 ms calls=130
Raster               72.880 ms calls=2298
BinaryOp             30.914 ms calls=985
While                22.600 ms calls=700
UnaryOp              12.552 ms calls=430
LayerNorm             8.588 ms calls=285
PicScoreAttention     8.096 ms calls=5
PagedAttention        7.559 ms calls=5
Cast                  5.106 ms calls=145
```

Top op has one cold spike:

```text
/layers.0/self_attn/k_proj/Linear total=62.607 ms max=62.060 ms calls=4
```

Do not optimize around that single cold spike. The stable gap still points at per-layer tiny-row MLP `Convolution`, `PicSparseAttention`, and layout/elementwise overhead. A second warm profile attempt returned 502 and left the profile server unresponsive, so switch back to the default server before formal timing.

## 09:59 Rows4-8 cuBLAS compute=16f A/B

Environment-only A/B:

```text
MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=16f
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_compute16f_20260628_09/summary.csv
x=0 85.890 ms
x=1 96.620 ms
x=3 122.585 ms
x=5 125.056 ms
x=7 127.378 ms
```

Against accepted baseline this is mixed: x=0/1/3 improve, x=5 is flat, x=7 regresses by about `+1.018 ms`. Do not accept as default yet; test `fast16` next.

## 09:59 Rows4-8 cuBLAS fast16 A/B

Environment-only A/B:

```text
MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=fast16
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_fast16_20260628_09/summary.csv
x=0 86.914 ms
x=1 97.692 ms
x=3 123.936 ms
x=5 125.363 ms
x=7 126.861 ms
```

Against the accepted baseline it regresses all rows by `+0.22..+0.54 ms`; against the noisy post-rebuild control it improves repair rows. Decision: not a default win. Next: test `rows45=down` to see whether gate/up cuBLAS is hurting high repair counts.

## 09:59 Rows4-8 cuBLAS down-only A/B

Environment-only A/B:

```text
MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=down
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_down_20260628_09/summary.csv
x=0 86.198 ms
x=1 97.357 ms
x=3 129.307 ms
x=5 131.362 ms
x=7 132.969 ms
```

Decision: reject. It regresses x=3/5/7 by about `+5.7..+6.6 ms` versus accepted baseline, so gate/up rows4-8 cuBLAS is useful. Next A/B should exclude only attention k/v projections from the rows4-8 cuBLAS rule while keeping MLP gate/up/down.

## 09:59 Rows4-8 cuBLAS minDim=2048 A/B

Temporary source A/B added `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_MIN_DIM`, then tested:

```text
MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_MIN_DIM=2048
summary=.cache/mnn-pic-benchmark/decode_repair_rows45_mindim2048_20260628_09/summary.csv
x=0 88.632 ms
x=1 97.884 ms
x=3 124.003 ms
x=5 125.039 ms
x=7 127.005 ms
```

Decision: reject as a default. It does not beat the accepted baseline; the temporary env hook should be removed after A/B.

## 09:59 Rows4-8 cuBLAS algo=default A/B

Two repeat3 runs:

```text
summary1=.cache/mnn-pic-benchmark/decode_repair_rows45_algo_default_20260628_09/summary.csv
summary2=.cache/mnn-pic-benchmark/decode_repair_rows45_algo_default_repeat2_20260628_09/summary.csv

x=0 mean 85.615 ms  delta vs accepted -0.762 ms
x=1 mean 97.632 ms  delta vs accepted +0.161 ms
x=3 mean 123.102 ms delta vs accepted -0.500 ms
x=5 mean 124.579 ms delta vs accepted -0.395 ms
x=7 mean 125.780 ms delta vs accepted -0.579 ms
```

Decision: accept as a source candidate. Change rows4-8 cuBLAS default algo from tensor-op default to cuBLAS default, then rebuild and verify with no env.

## 09:59 algo=default Source Verification Rejected

No-env verification after changing the source default did not reproduce the env A/B:

```text
summary1=.cache/mnn-pic-benchmark/decode_repair_algo_default_default_20260628_09/summary.csv
summary2=.cache/mnn-pic-benchmark/decode_repair_algo_default_default_repeat2_20260628_09/summary.csv

x=0 mean 86.903 ms  delta vs accepted +0.526 ms
x=1 mean 98.125 ms  delta vs accepted +0.653 ms
x=3 mean 124.137 ms delta vs accepted +0.535 ms
x=5 mean 125.979 ms delta vs accepted +1.004 ms
x=7 mean 128.510 ms delta vs accepted +2.151 ms
```

Decision: reject and restore tensor-op algo as the source default. Keep `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO=default` only as a diagnostic env.
