# Jetson PIC Decode Repair Notes

## Summary

- Goal remains reducing `lagged_attention_hkvd` decode-repair TPOT for `x=3/5/7` below 100 ms.
- x=0 graph profile was used as the baseline; x=3/5/7 share the same extra bottleneck: compact-row dense `Convolution`, not selector/ranking or `PicSparseAttention`.
- Direct-op rows=4/6/8 tests show current INT4 weight-only dense already beats pure FP16 cuBLAS floor, so more env tuning is not enough.
- `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS` default was changed from policy 2 to policy 3, enabling the already validated cuBLAS route for attention projections as well as MLP rows=4..8.

## Current No-Profile TPOT

`decode_repair_default_policy3_confirm_20260628_150247`, Jetson CUDA, context 1024, full-reuse, max_tokens=8, repeats=3:

| x | TPOT ms |
|---|--------:|
| 0 | 81.22 |
| 1 | 89.56 |
| 3 | 116.78 |
| 5 | 118.45 |
| 7 | 121.13 |

