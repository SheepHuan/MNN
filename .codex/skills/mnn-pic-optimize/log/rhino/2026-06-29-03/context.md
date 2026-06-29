# Rhino Decode Repair Tiny-Row Conv A/B Context

## Valid Baseline

Output CSV:

```text
.cache/tmp/rhino_decode_lagged_x0_x7_sparse_tpot_20260629_0225.csv
```

`MiniCPM5-1B`, OpenCL Adreno, full-reuse, `lagged_attention_hkvd`, `max_tokens=32`:

```text
ctx=512:  x0 66.219, x1 77.252, x3 88.989, x5 131.287, x7 139.411 ms/token
ctx=1024: x0 108.180, x1 102.279, x3 129.994, x5 157.917, x7 162.847 ms/token
```

`x=0` now reports `decode_refine_runtime=mnn_token_id_sparse_decode` and `repair_rows=0 sparse_rows=1 attention_rank_count=0`, so it is a valid sparse decode baseline.

## Profile Conclusion

`MNN_PIC_DECODE_REPAIR_PROFILE=1` showed decode repair time is almost entirely inside `forwardRaw`; embedding, mask/position ids, rank result, and bookkeeping are negligible. `lagged_attention_hkvd` rank readback is not the bottleneck. Graph profile points to compact-row dense/MLP work and layout/elementwise overhead.

## A/B 1: Direct C4NHW4 GEMV

Implementation idea:

- read C4NHW4 input directly;
- run one local-reduction GEMV kernel;
- write C4NHW4 output directly;
- avoid both `gemm_c4nhw4_to_nhwc` and `gemm_nhwc_to_c4nhw4`.

Smoke result:

```text
.cache/tmp/rhino_decode_direct_c4_smoke_20260629_0249.csv
ctx=512, max_tokens=8:
x0 78.916, x1 178.984, x3 167.615, x5 158.341, x7 168.900 ms/token
```

Conclusion: reject. The arithmetic is similar to batch GEMV, but direct C4 input makes the K loop read input with `bhw4` stride. The old input transpose is buying contiguous NHWC reads, and removing it hurts Adreno more than the launch it saves.

## A/B 2: NHWC GEMV With C4NHW4 Output

Implementation idea:

- keep `gemm_c4nhw4_to_nhwc`, preserving contiguous GEMV input;
- make `gemv_conv_c8_buf` optionally write C4NHW4 output;
- skip `gemm_nhwc_to_c4nhw4`.

Warm 32-token result:

```text
.cache/tmp/rhino_decode_c4out_warm32_20260629_0258.csv
ctx=512: x0 67.529, x1 79.125, x3 91.528, x5 131.435, x7 132.573 ms/token

.cache/tmp/rhino_decode_c4out_1024_warm32_20260629_0259.csv
ctx=1024: x0 78.159, x1 97.285, x3 133.475, x5 163.634, x7 166.420 ms/token
```

Conclusion: reject as default. The only clear win was `ctx=512/x7`; `x=1/3` regressed and `ctx=1024/x5/x7` was not better than the previous baseline. The remote artifact was restored to the original default three-stage batch-GEMV path.

Restore smoke:

```text
.cache/tmp/rhino_decode_default_restore_smoke_20260629_0305.csv
ctx=512, max_tokens=8: x0 76.465, x1 88.782 ms/token
```

## Next Implementation Direction

Do not optimize this by replacing the batch-GEMV layout path alone. The better target is a tiny-active-row MLP fusion:

1. Fuse `gate_proj + up_proj + SiLU(gate) * up` for rows `2/4/6/8`, keeping weights image-backed and avoiding separate Unary/Binary launches and intermediate writes.
2. Keep `down_proj` separate initially; it has different input channel width and can reuse the current tuned low-memory path.
3. Only after that consider a dedicated two-output weight-only kernel for `gate/up` so the two linears share input reads across the same active rows.
