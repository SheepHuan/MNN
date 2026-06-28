# Jetson CUDA decode repair graph/kernel optimization

## Summary

- Continue optimizing decode repair `x=1/3/5/7 lagged_attention_hkvd` against the required `x=0 none` baseline.
- The 03:00 hour ruled out CPU selector/rank overhead as the bottleneck: non-`forwardRaw` decode repair overhead is only about `0.11-0.12 ms/step`.
- Direct tiny-row MLP benches showed rows4/6/8 are about `1.7 ms/layer`, while simple GEMV/GEMM threshold changes regressed.
- Source inspection shows decode repair keeps `logits_index=-1`; the exported graph slices to the last row before `lm_head`, so final logits are unlikely to be the row-scaling bottleneck.

## Current Direction

- Use Jetson graph profile to verify actual row shapes and op hotspots for `x=1/3/5/7`.
- If `lm_head` is confirmed one-row, focus on model-body tiny-row work: MLP/Linear, residual/norm/BinaryOp/Raster, and launch-heavy graph fragments.
- Keep rejected experiments reverted: attention V reuse, rows4/5 cublas extension, GEMV limit 6, and V14 OC group 8.
- Keep qtile default-off for `x<=7`.

## Graph Profile Finding

- Graph profile confirmed decode repair preserves `logits_index=-1`; `/lm/lm_head/Linear` receives one row (`inputs=[1x3072x1x1]`), so final logits are not the row-scaling bottleneck.
- Profile TPOT is inflated by per-op sync, but attribution is useful: `Convolution` dominates `x>=3`, with `PicSparseAttention` the second major bucket.
- End-to-end decode repair is falling back to runtime dequant for many rows4/6/8 MLP/Linear ops (`runtime_dequant=1 static_dequant=0`), unlike the direct hot bench. Static dequant cache total is about `4.039 GB`, effectively at the current cap.
- Next target is memory-safe static dequant cache selection/cap behavior for decode repair hot MLP shapes, not CPU-side rank selection or attention weight capture.

## Static Dequant Cap A/B

- Raised the CUDA low-memory INT4 static FP16 dequant cache cap for high-memory Jetson-class devices only. Low-memory devices keep the previous default behavior.
- Non-profile smoke with the CUDA config (`config_cuda_greedy.json`) improved the matrix substantially:

```text
baseline before cap:
x=0 none                  TPOT=87.163 ms
x=1 lagged_attention_hkvd TPOT=158.793 ms
x=3 lagged_attention_hkvd TPOT=185.317 ms
x=5 lagged_attention_hkvd TPOT=184.994 ms
x=7 lagged_attention_hkvd TPOT=207.958 ms

4.0 GiB cap:
x=0 none                  TPOT=87.379 ms
x=1 lagged_attention_hkvd TPOT=101.761 ms
x=3 lagged_attention_hkvd TPOT=169.789 ms
x=5 lagged_attention_hkvd TPOT=182.527 ms
x=7 lagged_attention_hkvd TPOT=205.131 ms

4.5 GiB high-memory cap:
x=0 none                  TPOT=88.770 ms
x=1 lagged_attention_hkvd TPOT=102.391 ms
x=3 lagged_attention_hkvd TPOT=163.131 ms
x=5 lagged_attention_hkvd TPOT=173.214 ms
x=7 lagged_attention_hkvd TPOT=195.221 ms
```

- Current best is the 4.5 GiB high-memory cap. It narrows `x=1` to about `+13.6 ms` over `x=0`, and reduces `x=7` by about `12.7 ms` versus the original baseline.
- A CPU-config smoke accidentally used `.cache/weight/.../config.json`; ignore those rows. CUDA decode repair tests must use `.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json` and verify `"backend": "mnn_cuda"`.
- Remaining gap is likely still split between residual runtime dequant/fallback Linears and sparse attention kernel work, not compact attention-rank capture.
