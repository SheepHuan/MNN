# Jetson CUDA decode repair rows4-8 cuBLAS

## Summary

- Continued from the 06:00 accepted default:
  - `hd128_q8k16` qtile for decode-repair active rows `2/4/6/8`.
  - packed tiny GEMV for rows2.
  - rows4/5 MLP-like cuBLAS/static dense path.
- Tested extending the existing PIC decode-repair MLP-like cuBLAS admission from active rows `4/5` to `4..8`.
- The first smoke showed a strong `x=5/x=7` win but noisy `x=3`; a second smoke on the same server confirmed the high-row win and neutral low-row behavior.

## Accepted Debug Smoke

Rows4-8 cuBLAS repeat3, second run:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_smoke2_20260628_0710/summary.csv
x=0 none                  TPOT=86.377 ms
x=1 lagged_attention_hkvd TPOT=97.471 ms
x=3 lagged_attention_hkvd TPOT=123.602 ms
x=5 lagged_attention_hkvd TPOT=124.975 ms
x=7 lagged_attention_hkvd TPOT=126.360 ms
```

Compared with the previous accepted default smoke:

```text
x=0: 86.443 -> 86.377 ms
x=1: 98.521 -> 97.471 ms
x=3: 123.406 -> 123.602 ms
x=5: 132.097 -> 124.975 ms
x=7: 133.897 -> 126.360 ms
```

Decision: keep the rows4-8 cuBLAS admission. It materially reduces `x=5/x=7` and keeps `x=0/x=1/x=3` within noise.

## Current Best Jetson Debug Result

```text
x=0 none                  TPOT=86.377 ms
x=1 lagged_attention_hkvd TPOT=97.471 ms
x=3 lagged_attention_hkvd TPOT=123.602 ms
x=5 lagged_attention_hkvd TPOT=124.975 ms
x=7 lagged_attention_hkvd TPOT=126.360 ms
```

Remaining gap is now mostly rows2/4 attention+dense and general compact-row launch overhead. Rank capture remains a non-bottleneck.

## Profile Attribution

Profile-only run, not formal latency because the profile env adds synchronization:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_profile_20260628_0715/summary.csv
remote_log=.cache/mnn-pic-benchmark/decode_repair_rows48_profile_20260628_0715/remote/pic_server.log
x=0 none                  TPOT=123.901 ms
x=1 lagged_attention_hkvd TPOT=194.511 ms
x=3 lagged_attention_hkvd TPOT=263.907 ms
x=5 lagged_attention_hkvd TPOT=234.512 ms
x=7 lagged_attention_hkvd TPOT=231.010 ms
```

Key attribution:

- `decode_attention_rank`: `n=16`, total `4.019 ms`, median `235.5 us`; still not a bottleneck.
- qtile attention is about `0.85-0.95 ms` per layer for active rows `2/4/6/8`.
- rows4-8 cuBLAS reduced high-row dense totals; small-row `runtime_dequant` lines stayed at `0`.
- Remaining optimization targets are tiny-row qtile attention, dense/launch overhead, and graph-level activation/BinaryOp overhead. Do not revisit full attention-weight copy.
