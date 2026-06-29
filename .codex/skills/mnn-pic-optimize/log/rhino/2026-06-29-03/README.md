# Rhino Decode Repair Tiny-Row Conv A/B

## Summary

- Fixed and verified `x=0` decode repair so it stays on `mnn_token_id_sparse_decode` instead of falling back to ordinary decode.
- Measured Rhino / Adreno `MiniCPM5-1B` full-reuse decode repair TPOT for `x=0/1/3/5/7`, contexts `512/1024`, `max_tokens=32`.
- Tested two tiny-row Conv/GEMV fusion directions:
  - direct C4NHW4 GEMV: rejected, warm/cold smoke regressed badly because input becomes strided by `bhw4`.
  - NHWC-input GEMV with C4NHW4 output: rejected as default; warm results were mixed and not stable.
- Restored the remote Rhino artifact to the original default three-stage batch-GEMV path after A/B.

## Stable Direction

The bottleneck for `x=1/3/5/7` is still `forwardRaw`, dominated by small-active-row dense graph work, especially weight-only 1x1 MLP linears plus surrounding layout/elementwise ops. The next useful implementation direction is MLP-level fusion for tiny active rows, not further bypassing the existing C4/NHWC layout conversion in isolation.
