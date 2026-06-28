# Jetson CUDA decode repair 08:00 A/B

## Summary

- Tested cuBLAS rows4-8 runtime policies without source changes:
  - `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=32f_fast16`
  - `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=16f`
  - `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=down`
  - `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO=default`
- None beat the accepted default across `x=1/3/5/7`.
- Tested a source A/B in qtile attention: skip QK partial dot for causal-invalid `(logical > qLogical)` cells.
- The qtile causal-guard patch regressed TPOT and was reverted.

## Decision

Keep the accepted 07:00 rows4-8 cuBLAS default:

```text
x=0 none                  TPOT=86.377 ms
x=1 lagged_attention_hkvd TPOT=97.471 ms
x=3 lagged_attention_hkvd TPOT=123.602 ms
x=5 lagged_attention_hkvd TPOT=124.975 ms
x=7 lagged_attention_hkvd TPOT=126.360 ms
```

Rejected 08:00 candidates:

- `32f_fast16`: neutral/slightly worse.
- `16f`: hurts `x=3`.
- `rows45_cublas=down`: clearly worse for `x=3/5/7`.
- `algo=default`: small `x=3` noise win, not broad enough; `x=7` worse.
- qtile causal-guard source patch: worsened `x=5/x=7`.

Next useful direction is not more env tuning. It is either a real tiny-row dense kernel/fusion, attention kernel redesign that avoids mixed-logical tile waste without adding branch pressure, or graph-level launch/activation fusion.

## Continuation

- Started the next source A/B after the rejected qtile causal guard: extend rows4-8 cuBLAS matching to also cover large square projection Linears where `ic == oc` and `min(ic, oc) >= 1024`.
- Motivation: profile attribution shows batch4/6/8 `3072 -> 3072` projections still using CUTLASS at about `270 us` median per op, while MLP-like rows4-8 routes now use the cuBLAS path.
- Decision rule: keep the square-projection match only if repeat3 decode TPOT improves broadly for `x=3/5/7` without hurting `x=0/x=1`; otherwise revert just this A/B and restore the accepted 07:00 rows4-8 cuBLAS artifact.

## Square Projection A/B Result

Rejected. The route did not improve repeat3 TPOT and slightly hurt `x=0/x=1`:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_square_cublas_20260628_0848/summary.csv
x=0 none                  TPOT=87.419 ms
x=1 lagged_attention_hkvd TPOT=98.843 ms
x=3 lagged_attention_hkvd TPOT=123.998 ms
x=5 lagged_attention_hkvd TPOT=125.594 ms
x=7 lagged_attention_hkvd TPOT=126.822 ms
```

The square-projection match was reverted; accepted rows4-8 MLP-like cuBLAS remains the default.

## QTile Variant A/B Result

Rejected `hd128_q4k16` and `hd128_q4k8`; both were much slower than the default decode-repair `hd128_q8k16`.

```text
q4k16 x=1/3/5/7 TPOT=114.705 / 141.308 / 142.521 / 145.220 ms
q4k8  x=1/3/5/7 TPOT=116.506 / 143.489 / 145.505 / 149.690 ms
q8k8  x=1/3/5/7 TPOT=102.254 / 128.661 / 129.817 / 131.013 ms
```

Keep default `hd128_q8k16`. The next viable direction is not smaller K tiles; it is reducing launch/fusion overhead or building a dedicated tiny-row dense path.

## Restore

- Reverted the rejected square-projection cuBLAS and q8k8 source A/B changes.
- Rebuilt `MNN_Cuda_Main` / `pic_server` and synced the accepted artifact back to Jetson.
- Default server smoke after restore returned `/v1/models` HTTP 200.
- Remote 18131 server and local 19131 tunnel were cleaned after the smoke.

## Next Optimization Focus

Continue optimizing Jetson CUDA decode repair TPOT for `x=1/3/5/7 lagged_attention_hkvd` toward the `x=0 none` baseline while keeping the accepted rows4-8 MLP-like cuBLAS route.

Current accepted repeat3 baseline:

```text
x=0 none                  TPOT=86.377 ms
x=1 lagged_attention_hkvd TPOT=97.471 ms
x=3 lagged_attention_hkvd TPOT=123.602 ms
x=5 lagged_attention_hkvd TPOT=124.975 ms
x=7 lagged_attention_hkvd TPOT=126.360 ms
```

Direction for the next pass:

- Do not continue qtile shape/env tuning unless a profile shows a new specific issue; `q4k16`, `q4k8`, and `q8k8` all lost.
- Do not extend the rows4-8 cuBLAS admission to square projections; that A/B lost.
- Re-audit the current fused `decode_attention_rank` path, but treat it as secondary because prior profile shows only about 4 ms total over the run.
- Profile and optimize the remaining `forward_raw` gap: tiny-row attention, compact dense/MLP, activation/BinaryOp, and launch overhead.

## Continuation Profile

Profile-only run, not formal TPOT:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_profile_cont_20260628_08/summary.csv
x=0 none                  TPOT=91.944 ms
x=1 lagged_attention_hkvd TPOT=119.551 ms
x=3 lagged_attention_hkvd TPOT=152.461 ms
x=5 lagged_attention_hkvd TPOT=155.550 ms
x=7 lagged_attention_hkvd TPOT=156.818 ms
```

Attribution excluding step0:

```text
x=1 forward_raw_med=104.873ms attention=23.747ms conv=52.393ms rank=0.238ms
x=3 forward_raw_med=135.382ms attention=24.946ms conv=79.078ms rank=0.227ms
x=5 forward_raw_med=138.214ms attention=25.900ms conv=78.443ms rank=0.241ms
x=7 forward_raw_med=139.195ms attention=27.339ms conv=78.398ms rank=0.271ms
```

`decode_attention_rank` is fused in attention for all 64 profiled steps. The CPU-visible `rank_result_ms` is about `0.004 ms/step`, and rank capture/top-k is not the main TPOT gap.

Graph profile for `x=7` confirms the request is dominated by:

```text
Convolution         788.956 ms
PicSparseAttention  268.379 ms
Raster              141.019 ms
BinaryOp             58.927 ms
While                44.598 ms
UnaryOp              24.107 ms
```

This points to compact dense/MLP plus graph/elementwise launch overhead, not full attention-weight copy.

## QTile-Off A/B

Rejected. Forcing decode repair qtile off with `MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=0` regressed every repair count:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_off_20260628_08/summary.csv
x=0 none                  TPOT=88.125 ms
x=1 lagged_attention_hkvd TPOT=101.057 ms
x=3 lagged_attention_hkvd TPOT=137.788 ms
x=5 lagged_attention_hkvd TPOT=150.136 ms
x=7 lagged_attention_hkvd TPOT=172.044 ms
```

Keep default `hd128_q8k16` qtile for decode repair.

## Rank Pool Config A/B

`top_m=8` all-heads was noise-level and not a clear win:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_topm8_20260628_08/summary.csv
x=1/3/5/7 TPOT=98.374 / 124.610 / 125.719 / 125.705 ms
```

`top_m=8, attention_head_ids=0` was stable and slightly faster than the accepted 07:00 baseline, but it changes the attention-rank selector semantics:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_topm8_head0_20260628_08/summary.csv
x=0 none                  TPOT=86.255 ms
x=1 lagged_attention_hkvd TPOT=97.108 ms
x=3 lagged_attention_hkvd TPOT=123.215 ms
x=5 lagged_attention_hkvd TPOT=124.826 ms
x=7 lagged_attention_hkvd TPOT=125.641 ms
```

Treat `top_m=8, head0` as an optional speed-profile config, not a default source change, until output-quality impact is checked.

## 08:39 Continuation

- User asked to record the log first, then continue optimizing.
- Current accepted source baseline is still rows4-8 MLP-like cuBLAS plus static decode-hot dequant cache.
- Keep rejected directions rejected: square-projection cuBLAS admission, qtile q4/q8k8 variants, qtile-off, and selector-semantic `top_m/head0` as a default.
- Next pass focuses on graph-level CUDA overhead seen in x=7 profile: compact dense/MLP, existing `PicSiluMul` / `PicPackedSiluMul` / `PicGateUpWeightOnly` fusion availability, and small-row Raster/BinaryOp/Unary launch overhead.
- Decision criterion remains repeat3 decode TPOT for x=0/1/3/5/7, with no quality-changing selector semantics unless explicitly reported as an optional speed-profile config.

## PicSiluMul Export A/B Setup

- Current 3B `pic-boundary` model was exported with `pic_decode_tiny_fusion=false`, so the graph profile's MLP `UnaryOp + BinaryOp` overhead is expected.
- Created A/B model `.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-silumul` with only `--pic_decode_tiny_fusion` additionally enabled.
- Checks passed:
  - `llm_config.json`: `paged_attention=true`, `pic_recompute_budget=true`, `pic_recompute_score_layer_idx=1`, `pic_decode_tiny_fusion=true`.
  - `llm.mnn.weight`: `1.9G`.
  - `llm.mnn.json`: contains `PicSiluMul`.
- Next step: sync this model to Jetson and run the same repeat3 decode repair x=0/1/3/5/7 matrix.

## PicSiluMul Export Issue

- First A/B export used a stale x64 converter/lib combination.
- Result: `PicSiluMul` was present, but `PicScoreAttention` / `PicSparseAttention` were serialized as `type=-1`, and Jetson crashed at model load in `SizeComputerSuite::search(OpType=-1)`.
- Action: added a converter fallback for `type=-1/main_type=AttentionParam`, then identified that JSON->MNN still needs a current x64 `MNNConvert/libMNN` build; rebuilding x64 converter before rerunning the A/B.
