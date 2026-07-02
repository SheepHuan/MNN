# Rhino Decode Repair Tiny-Row MLP Direct-Op Follow-Up

## Summary

- Continued Rhino Pi-X1 / Adreno OpenCL `MiniCPM5-1B` decode repair MLP direct-op work for rows `2/4/6/8`, `hidden=1536`, `inter=4608`, int4 weight-only.
- Rechecked `DecodeRepairMlpSharedInputGateUp` with WGS sweep. Valid WGS candidates still do not make the old shared-input chain beat split for rows `6/8`.
- Added a bench-only `DecodeRepairMlpSiluNhwcDown` floor that fuses `SiLU(gate)*up` with C4-to-NHWC activation conversion, then calls the image-backed down GEMV directly.
- Extended `DecodeRepairMlpSharedInputGateUp` to also measure the combined path:

```text
shared hidden preconvert
-> gate/up direct image-backed projections
-> fused silu_c4_to_nhwc
-> direct image-backed down projection
```

## Key Result

Warmup 40, repeat 160, WGS 64:

| rows | split chain ms | shared NHWC chain ms | delta ms | bad |
|-----:|---------------:|---------------------:|---------:|----:|
| 2 | 0.4674 | 0.4459 | -0.0215 | 0/3072 |
| 4 | 0.4841 | 0.4520 | -0.0321 | 0/6144 |
| 6 | 0.8443 | 0.8359 | -0.0084 | 0/9216 |
| 8 | 0.8519 | 0.8343 | -0.0176 | 0/12288 |

This is the first bench-only direct-op candidate in this sequence where rows `2/4/6/8` all beat the row-aware split chain and pass accuracy.

## Decision

Do not route this to endpoint yet. The result is still a testbench-only composition, not a production graph/exporter path. The next engineering step is to decide whether to expose this as a narrow Rhino/Adreno decode-repair route, likely by preserving row-aware gate/up selection while replacing the `PicSiluMul + down preconvert` boundary with an NHWC-producing activation path consumed by down projection.

