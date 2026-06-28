# Jetson CUDA MiniCPM 512 qtile/default fix

## Summary

- Fixed CUDA persistent PIC cache hydrate failure where async reads were scheduled before future layers had registered direct mapped PagedCache targets.
- Kept non-prefix CUDA requests from reusing same-shape non-mapped PagedCache when zero-copy mapped cache is required.
- Promoted head_dim=128 CUDA sparse qtile default to `hd128_q4k16`; explicit `MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=off/none` still disables it.
- Validated no-env default MiniCPM5-1B ctx512 10%-50% sweep on Jetson: cacheblend and epic are monotonic and 50% remains faster than normal by more than 1.4x.

## Accepted Run

```text
run_id=jetson_minicpm512_default_q4k16_sweep_20260627_1
normal=0.452172613s
cacheblend50=0.280231s speedup=1.614x
epic50=0.283709s speedup=1.594x
failures=none
server_env=[]
```

