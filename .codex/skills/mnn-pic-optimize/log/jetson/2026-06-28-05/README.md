# Jetson CUDA decode repair hot-cache and adaptive qtile

## Summary

- Continued optimizing `x=1/3/5/7 lagged_attention_hkvd` against the required `x=0 none` baseline.
- Implemented high-memory decode-hot static dequant admission for CUDA low-memory INT4 1x1 Linears:
  - MLP-like large Linears keep the high cap.
  - Attention projection-like Linears (`3072->3072`, `3072->1024` for Llama3.2-3B) also use the high-memory cap.
  - Very large non-decode-hot Linears such as vocab/lm_head still use the base cap.
- Implemented adaptive decode repair qtile for CUDA PagedAttention:
  - Default `headDim=128 && attnLen>=6` now uses `hd128_q8k16`.
  - `attnLen=2/4` stays on the row-compressed decode path to avoid the measured x=1/x=3 regression.
- Fused decode attention-rank score accumulation into the qtile path, so rows6/8 do not need a separate compact QK scan before top-M selection.

## Best Smoke Result

Non-profile Jetson CUDA, context 1024, `max_tokens=8`, `attention_layer_idx=1`, `top_m=32`:

```text
x=0 none                  TPOT=87.641 ms  ok
x=1 lagged_attention_hkvd TPOT=97.924 ms ok
x=3 lagged_attention_hkvd TPOT=123.946 ms ok
x=5 lagged_attention_hkvd TPOT=131.463 ms ok
x=7 lagged_attention_hkvd TPOT=134.923 ms ok
```

Compared with the original starting point:

```text
x=1: 158.793 -> 97.924 ms
x=3: 185.317 -> 123.946 ms
x=5: 184.994 -> 131.463 ms
x=7: 207.958 -> 134.923 ms
```

## Current Conclusion

- Compact attention-rank capture remains not the bottleneck.
- Runtime dequant for rows4/6/8 decode repair Linears was eliminated in profile; static cache total reached about `5.64 GB`, with Jetson still showing about `14 GiB` available memory.
- Decode-repair q8/k16 should stay enabled for all active row counts `>=2`; the earlier rows2 regression was specific to q4/k16, not q8/k16.
- Generic rows4/5 compact MLP cuBLAS admission should stay: it improves `x=3` and is neutral/slightly positive for `x=5/7`.
- Remaining gap to `x=0` is now mostly sparse attention plus compact-row dense/graph launch overhead, not CPU selector overhead.

## Rows4/5 cuBLAS Follow-Up

After changing the compact MLP rows4/5 cuBLAS guard from hardcoded `2048 <-> 8192` to generic MLP-like dimensions, repeat3 Jetson smoke produced:

```text
x=0 none                  TPOT=87.233 ms
x=1 lagged_attention_hkvd TPOT=102.155 ms
x=3 lagged_attention_hkvd TPOT=138.823 ms
x=5 lagged_attention_hkvd TPOT=149.857 ms
x=7 lagged_attention_hkvd TPOT=152.396 ms
```

Compared with adaptive qtile before this follow-up:

```text
x=1: 102.506 -> 102.155 ms
x=3: 145.501 -> 138.823 ms
x=5: 150.995 -> 149.857 ms
x=7: 154.879 -> 152.396 ms
```

Conclusion: keep this CUDA compact dense change. It specifically helps rows4 decode repair (`x=3`) and does not harm the `x=0` baseline.

## QTile Fused Rank And q8/k16 Follow-Up

The qtile path now accumulates decode attention-rank scores inside `sparse_flash_qtile_attention` at the configured capture layer. It then only runs topK/D2H/result preparation; it no longer launches the separate compact QK scan when fused capture is valid.

Best repeat3 default after switching `headDim=128 && attnLen>=6` to `hd128_q8k16`:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_q8k16_default_repeat3_20260628_0610/summary.csv
x=0 none                  TPOT=88.367 ms
x=1 lagged_attention_hkvd TPOT=102.390 ms
x=3 lagged_attention_hkvd TPOT=138.141 ms
x=5 lagged_attention_hkvd TPOT=131.526 ms
x=7 lagged_attention_hkvd TPOT=133.656 ms
```

The isolated qtile A/B showed `hd128_q8k16` is the right default for rows6/8:

```text
q8k16 x=5 133.803 ms, x=7 135.078 ms
q4k8  x=5 150.307 ms, x=7 155.156 ms
```

Profile confirmation:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_q8k16_default_profile_20260628_0615/summary.csv
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/pic_prefill_latency_sweep/decode_repair_q8k16_default_profile_20260628_0615/pic_server.log
```

Profile lines confirm `variant=hd128_q8k16` and `fused_score_in_attention=1`; after fusion, decode-rank capture is only about `0.20-0.26 ms` for topK/D2H/prep. Later qtile attention layers for rows6 are about `0.86-0.95 ms`, down from roughly `1.4-1.6 ms` on the earlier q4/k16 path.

## QTile Rows2/4 Threshold Follow-Up

Lowering the decode-repair-only q8/k16 threshold from rows6 to rows2 improved the remaining small-row gap. New default repeat3 after rebuilding and syncing Jetson artifact:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_q8k16_min2_default_repeat3_20260628_0640/summary.csv
x=0 none                  TPOT=87.641 ms
x=1 lagged_attention_hkvd TPOT=97.924 ms
x=3 lagged_attention_hkvd TPOT=123.946 ms
x=5 lagged_attention_hkvd TPOT=131.463 ms
x=7 lagged_attention_hkvd TPOT=134.923 ms
```

Against the original starting point:

```text
x=1: 158.793 -> 97.924 ms
x=3: 185.317 -> 123.946 ms
x=5: 184.994 -> 131.463 ms
x=7: 207.958 -> 134.923 ms
```

Against the previous q8/k16 default threshold rows6:

```text
x=1: 102.390 -> 97.924 ms
x=3: 138.141 -> 123.946 ms
x=5: 131.526 -> 131.463 ms
x=7: 133.656 -> 134.923 ms
```

Decision: keep decode-repair `headDim=128 && attnLen>=2` on `hd128_q8k16`. This does not change the generic sparse qtile selector.

## Continuation Goal

- Continue from the adaptive qtile + decode-hot static dequant result and reduce:
  - `x=1 lagged_attention_hkvd`: about `98 ms` toward `x=0 about 88 ms`.
  - `x=3 lagged_attention_hkvd`: about `124 ms`.
  - `x=5 lagged_attention_hkvd`: about `131 ms`.
  - `x=7 lagged_attention_hkvd`: about `135 ms`.
- Optimization focus:
  - Keep the compact attention ranking design: one configured layer captures only top-M PIC local indices, not full attention matrices.
  - Avoid broad graph or operator redesign unless profiling proves it is necessary.
  - First look for non-compute overhead in selector/rank transfer and unnecessary decode-repair setup.
  - Then profile and reduce `PicSparseAttention`, compact-row dense/activation, and graph launch overhead for active row counts 2/4/6/8.
