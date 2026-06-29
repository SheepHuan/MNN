# Rhino Decode Repair FFN Fusion A/B

## Summary

- Confirmed the current MiniCPM5-1B Rhino model was exported with `pic_decode_tiny_fusion=false`; graph had no `PicSiluMul`.
- Exported and tested `PicSiluMul`-only model. It removed 24 `UnaryOp` and 24 `BinaryOp` nodes, but did not improve low-x TPOT.
- Implemented OpenCL buffer backend support for existing `PicPackedSiluMul` Extra op and exported `PicGateUpSiluWeightOnly` graph.
- Fixed an implementation mistake where `PicPackedSiluMul` used `buildKernelFromSource` inside `onEncode`; moving build to the execution constructor reduced smoke TPOT from about 519 ms/token to about 85 ms/token.
- Final gate+up packed fusion was still mixed: small wins at 1024/x1 and 1024/x7, but 512/x1/x3 regressed and x=1/3 remained farther from x=0.
- Added `--pic_decode_gateup_split_fusion`, which lowers gate/up to two shared-input Conv nodes plus `PicSiluMul` instead of the `2*OC` concat Conv. It improved only `ctx1024/x1` extra (`10.4ms -> 8.7ms`) and regressed the larger repair rows, so it is also not a default candidate.
- Added a row-aware Adreno tiny dense family heuristic for original Conv graphs:
  - sparse rows 2: `pic_quant_c4`
  - sparse rows 4: `pic_quant_incache64`
  - sparse rows >=6: `adreno_batch_gemv_c4out`
- On RhinoPiU full-reuse decode=32, this reduced average extra TPOT over x=0 from `+10.4/+27.5/+63.9/+67.8ms` to `+7.1/+22.8/+50.7/+54.4ms` for x=1/3/5/7.

## Decision

Do not enable `PicSiluMul`-only, concat `PicGateUpSiluWeightOnly`, or split gate/up fusion as default for Rhino decode repair yet.

The useful retained code changes are:

- OpenCL `PicPackedSiluMul` backend implementation, because it makes the existing exported op executable and avoids repeated OpenCL program compilation.
- Row-aware Adreno tiny dense family selection for rows 2/4/6/8 in `ConvBufLowMemoryExecution`, because it improves repair overhead without changing the exported graph.

Even with row-aware family selection, x=5/7 still remain about `50-54ms/token` above x=0 on the 512/1024 average. The stronger next direction is true fused `gate/up -> SiLU -> down` for rows `2/4/6/8`, or dedicated tiny-row attention projection kernels. Avoid spending more time only rearranging graph-level gate/up nodes.
