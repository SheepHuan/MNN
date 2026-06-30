# Jetson CUDA Decode Repair MLP P0

## Summary

- Added same-selector NHWC graph-profile attribution for `lagged_attention_hkvd` decode repair `x=0/1/3/5/7`.
- Formal endpoint still shows the main jump at `x=1 -> x=3`: `92.08 ms -> 118.13 ms`, while `x=3 -> x=7` is only about `+2.72 ms`.
- Low-level dense attribution shows the actionable MLP delta:
  - `MLP gate/up`: `x=1 21.48 ms` to `x=7 32.29 ms`, delta `+10.81 ms`.
  - `MLP down`: `x=1 10.25 ms` to `x=7 17.92 ms`, delta `+7.66 ms`.
- Next optimization priority is MLP compact dense, not sparse attention. Attention is a large fixed base cost, but it does not explain the positive `x=1 -> x=7` latency delta in this run.

## Decision

Treat CUDA compact MLP as the next P0:

1. First target `gate_proj + up_proj` because both consume the same hidden rows and are independent before `SiLU(gate) * up`.
2. Then target `down_proj`, either by improving the rows4/6/8 compact GEMM path or by a larger MLP fusion that avoids writing/reading the intermediate activation.
3. Implement this as one generic compact MLP family, not separate x-specific operators. `x=3/5/7` may choose different schedule parameters such as tile shape, block size, split-K, or vector width, but the algorithm and graph/export path must be the same.
4. Success must be judged by formal endpoint TPOT for `x=0/1/3/5/7`, with graph profile used only for attribution.

## Candidate Goal

Optimize Jetson CUDA `lagged_attention_hkvd` decode repair compact MLP for active rows 4/6/8 with one parameterized implementation shared by `x=3/5/7`. Reduce `gate/up` and `down` dense fixed overhead while preserving `x=0/1` behavior and correctness. Primary target: remove at least `8 ms` of MLP dense delta versus current NHWC static-share endpoint and move `x=3/5/7` TPOT materially closer to or below `110 ms`.
