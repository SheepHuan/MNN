# Rhino Decode Repair MLP Chain Direct-Op Baseline

## Summary

- Added and built an AArch64 static OpenCL `run_test.out` testbench for Rhino/Aidlux.
- Verified the new `bench_ops/opencl/perf/DecodeRepairMlpChain` direct-op entry on Rhino Pi-X1 / Adreno.
- The bench preserves the current row-aware tiny dense family selection:
  - rows 2: `pic_quant_c4`
  - rows 4: `pic_quant_incache64`
  - rows 6/8: `adreno_batch_gemv_c4out`
- Captured MiniCPM5-1B decode repair MLP split-chain baseline for rows `2/4/6/8`, shape `hidden=1536`, `inter=4608`, int4 weight-only.

## Baseline

Repeat 160, warmup 40, OpenCL low precision / low memory:

| rows | gate family | gate ms | up ms | silu ms | down ms | split sum ms | chain ms |
|-----:|-------------|--------:|------:|--------:|--------:|-------------:|---------:|
| 2 | `pic_quant_c4` | 0.1668 | 0.1673 | 0.0164 | 0.1367 | 0.4871 | 0.4710 |
| 4 | `pic_quant_incache64` | 0.1706 | 0.1713 | 0.0143 | 0.1410 | 0.4972 | 0.4852 |
| 6 | `adreno_batch_gemv_c4out` | 0.2990 | 0.2997 | 0.0303 | 0.2453 | 0.8743 | 0.8464 |
| 8 | `adreno_batch_gemv_c4out` | 0.3010 | 0.3023 | 0.0454 | 0.2431 | 0.8917 | 0.8537 |

## Decision

This establishes the direct-op floor for the next fused tiny-row MLP kernel. A production route should not replace the current row-aware family selection until a fused MLP direct-op beats these chain times and passes accuracy for rows `2/4/6/8`.

