# Jetson Target-Model Decode Repair x3 Cliff Profile

## Summary

- Scope is explicitly limited to MiniCPM5-1B, Qwen3-4B, and Llama3.2-3B. Llama3.2-1B is not considered for optimization decisions.
- Ran Jetson CUDA endpoint decode-repair graph profile for ctx `1024`, `full-reuse`, selector `top_hkvd`, `max_tokens=32`, `x=1/3`.
- The `x=3` cliff is not primarily attention. It is the decode-repair active-row jump from 2 rows to 4 rows, which pushes CUDA weight-only 1x1 Linear from tiny GEMV into batch-4 dense routes.
- Across all target models, graph-profile `x=3-x=1` is dominated by `Convolution`; `PicSparseAttention` is flat or slightly lower.
- Qwen3-4B shows the largest cliff and additionally exposes generic runtime-dequant Conv in some batch-4 Linear shapes, likely because the static FP16 dequant cache does not cover all weights.

## Artifacts

- Run root: `.cache/bench_ops/decode_repair_x3_profile_20260701_102007`
- Target model logs:
  - `minicpm_server.log`, `minicpm_profile_analysis.txt`, `minicpm_profile_decode.csv`
  - `qwen3_4b_server.log`, `qwen3_4b_profile_analysis.txt`, `qwen3_4b_profile_decode.csv`
  - `3b_server.log`, `3b_profile_analysis.txt`, `3b_profile_decode.csv`

## Key Table

Graph profile is attribution-only and synchronizes after ops. The component columns below are normalized to single-TPOT contribution and are used only for attribution.

| model | x1 TPOT ms | x3 TPOT ms | TPOT delta ms | Conv contrib / TPOT | MLP contrib / TPOT | attn-proj contrib / TPOT | PicSparseAttention / TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM5-1B | 86.76 | 97.56 | +10.79 | +11.51 | +5.04 | +6.53 | -0.15 |
| Qwen3-4B | 170.47 | 225.59 | +55.12 | +51.67 | +28.93 | +22.87 | -0.22 |
| Llama3.2-3B | 128.82 | 160.07 | +31.26 | +28.20 | +17.72 | +10.59 | -0.23 |

## Decision

- Optimize CUDA active-row weight-only dense first, not `PicSparseAttention`, for the `x=1 -> x=3` decode-repair cliff.
- The important shapes are rows4/6/8 Linear for MLP gate/up/down and attention q/k/v/o projection.
- For Qwen3-4B, also inspect why some batch-4 Linear falls to generic runtime-dequant Conv instead of static-cache rows45 cublas.
- Gate/up packed remains insufficient by itself because attention projections and down_proj are a large part of the delta.
