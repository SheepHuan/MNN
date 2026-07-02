# Rhino Resource-Backed Gate/Up Direct-Op A/B

## Summary

- Completed and built the bench-only OpenCL direct-op entry:
  - `bench_ops/opencl/perf/DecodeRepairMlpResourceGateUp`
- The path reuses `ConvBufLowMemoryExecution` resources for gate/up/down and runs one image-backed gate+up+SiLU kernel before the existing down projection.
- Accuracy passed for MiniCPM5-1B decode-repair MLP rows `2/4/6/8`.
- Performance failed the direct-op gate: every row count is slower than the current row-aware split chain.
- A second bench-only floor, `DecodeRepairMlpSiluDownC4`, kept row-aware gate/up and fused `SiLU(gate)*up + down` directly from C4 gate/up tensors. It also passed accuracy but regressed every row count.

## Result

Warmup 40, repeat 160, shape `hidden=1536`, `inter=4608`, `qblock=64`:

| rows | split chain ms | resource gate/up+SiLU ms | fused chain ms | delta ms | bad |
|-----:|---------------:|-------------------------:|---------------:|---------:|----:|
| 2 | 0.4681 | 0.4078 | 0.5427 | +0.0746 | 0/3072 |
| 4 | 0.4838 | 0.4583 | 0.5949 | +0.1111 | 0/6144 |
| 6 | 0.8438 | 0.8396 | 1.0823 | +0.2385 | 0/9216 |
| 8 | 0.8528 | 0.8892 | 1.1333 | +0.2805 | 0/12288 |

## Decision

Reject both bench-only fused floors as production routes. The resource-backed gate/up fusion proves correctness, but it does not preserve the accepted Rhino row-aware tiny dense family behavior:

```text
rows <= 2: pic_quant_c4
rows <= 4: pic_quant_incache64
rows >= 6: adreno_batch_gemv_c4out
```

The `SiluDownC4` floor confirms that directly consuming C4 gate/up activations for down is also not enough; the C4 stride read cost outweighs the saved activation/preconvert work. The next direct-op attempt should either add row-aware fused gate/up variants matching those families, or move to a tiled fused-through-down floor that preserves contiguous NHWC/input-cache behavior while reducing the `[rows, inter]` activation write/read.
