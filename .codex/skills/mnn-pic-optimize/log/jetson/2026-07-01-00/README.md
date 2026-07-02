# Jetson CUDA Decode Repair MLP Fusion Plan

## Summary

- Scope: analyze a bench-only MLP fusion path for Jetson CUDA decode repair after the accepted packed gate/up optimization.
- Attention optimization is explicitly out of scope for this pass.
- Current objective: for repair `x=3/5/7` (`rows=4/6/8`), squeeze about `0.25 ms/layer` from MLP execution before attempting endpoint work.
- This target is intentionally higher than the previously rejected down-only retuning gains. Across 28 layers, `0.25 ms/layer` is about `7 ms/token`; this is the minimum useful MLP-side contribution toward the `95 ms` endpoint target.
- Do not change the export graph first. Start with a single fused-op/direct bench harness that compares against the current packed chain:
  - packed gate/up projection: `3072 -> 16384`
  - packed SiLU/multiply: `16384 -> 8192`
  - down projection: `8192 -> 3072`
- Existing evidence says simple down-proj retuning is insufficient. The credible direction is a fused MLP operator that reduces intermediate activation materialization and graph/kernel overhead while preserving tensor-core-quality packed gate/up and current down accuracy policy.

## Decision

Proceed only as a bench-only investigation:

1. add a direct fused-chain test case under the CUDA bench-op framework;
2. measure rows `1/2/4/6/8`, with promotion gating on rows `4/6/8`;
3. require at least about `0.25 ms/layer` improvement on rows `4/6/8` versus the current packed chain before considering graph/export integration;
4. reject variants that improve one row count but regress another target row, or that depend on changing down accumulation precision without a separate accuracy decision.

## Bench Update

- Added bench-only CUDA Extra candidates:
  - `PicBenchFusedPackedSiluDown`: single-op materialized `PicPackedSiluMul + cuBLAS down` lower bound.
  - `PicBenchStreamedPackedSiluDown`: custom streaming activation/down micro-fusion that avoids the full activation tensor writeback.
- Added direct-op tests:
  - `bench_ops/cuda/accuracy/PicFusedSiluDown`
  - `bench_ops/cuda/perf/PicFusedSiluDown`
- Jetson accuracy passed for rows `1/2/4/6/8`; streaming max abs was `4.88e-4`.
- Jetson rows `4/6/8` repeat=80 perf did not pass the `-0.25 ms/layer` promotion gate:
  - materialized deltas: `-0.0015 / +0.0031 / -0.0017 ms`
  - streamed deltas: `+2.1478 / +3.4889 / +4.7494 ms`
- Conclusion: fused activation/down materialization removal alone does not have useful headroom on this shape, and the first scalar streaming micro-fusion is far too slow versus tensor-core cuBLAS down. Do not integrate this candidate into graph/export.

## WMMA Follow-Up

- Restored a bench-only tensor-core candidate, `PicBenchWmmaPackedSiluDown`, to test whether a custom WMMA down kernel can keep tensor-core execution while consuming packed gate/up directly.
- Accuracy passed for rows `1/2/4/6/8`; WMMA max abs was `4.88e-4`, `bad=0`.
- Jetson rows `4/6/8` repeat=80 perf still failed the promotion gate:
  - `wmma_fused`: `1.1484 / 1.1016 / 1.1942 ms`
  - `baseline_chain`: `1.5577 / 1.5094 / 1.5377 ms`
  - `wmma_chain`: `2.1369 / 2.0986 / 2.1902 ms`
  - `wmma_delta`: `+0.5793 / +0.5892 / +0.6526 ms`
- Guardrail rows `1/2` also regressed:
  - `wmma_delta`: `+0.4339 / +0.5617 ms`
- Conclusion: a naive WMMA 16x16 tile kernel is still much slower than the current cuBLAS/tensor-core down path. It preserves tensor-core math but loses cuBLAS scheduling quality and pays too much activation staging overhead. This is another rejected bench-only idea; do not route production graph/export to it.

## Production Baseline Retest

- Re-ran current production-shaped `PicDecodeMlp` direct-op baseline after the bench-only WMMA addition.
- Command used fp16 + `Memory_Low`: `run_test.out bench_ops/cuda/perf/PicDecodeMlp 2 2 1 x 2`.
- rows `1/2/4/6/8`, hidden `3072`, inter `8192`, repeat `80`:

| rows | split chain ms | packed chain ms | packed delta vs split |
|-----:|---------------:|----------------:|----------------------:|
| 1 | 0.8686 | 0.8570 | -0.0116 |
| 2 | 0.9740 | 0.9645 | -0.0095 |
| 4 | 1.5236 | 1.5257 | +0.0021 |
| 6 | 1.5359 | 1.5283 | -0.0077 |
| 8 | 1.5306 | 1.5467 | +0.0160 |

- Compared with the earlier fused-bench baseline (`1.5168 / 1.5209 / 1.5496 ms`), this is within direct-op noise and shows no production baseline regression.
- Production boundary remains unchanged: the fastest accepted path is the existing packed gate/up + `PicPackedSiluMul` + current rows4/6/8 cuBLAS down path. The materialized, scalar streamed, and WMMA candidates remain bench-only evidence and must not be exported or selected by default.

## End-to-End Decode Retest

- Rebuilt and synced `pic_server` plus CUDA runtime libs, then ran real `pic_server` chat decode with `max_tokens=32`.
- Fixed the decode benchmark client so `repair_tokens=0` is a true no-repair baseline and does not send `decode_refine`.
- The local client must set `NO_PROXY=192.168.101.192,127.0.0.1,localhost`; without it, local HTTP proxy returned a misleading `502` before the request reached Jetson.
- Focused Jetson run: Llama3.2 1B decode-repair silumul, CUDA, context `512`, `full-reuse`, `top_hkvd`, `tpd=0..4`, warm `1`, repeat `3`.

| tpd | decode TPOT ms | decode TPS | runtime check |
|----:|---------------:|-----------:|---------------|
| 0 | 29.1503 | 34.3057 | no `decode_refine` |
| 1 | 31.0386 | 32.2187 | `mnn_token_id_sparse_decode` |
| 2 | 35.0573 | 28.5258 | `mnn_token_id_sparse_decode` |
| 3 | 37.7080 | 26.5197 | `mnn_token_id_sparse_decode` |
| 4 | 36.9722 | 27.0485 | `mnn_token_id_sparse_decode` |

- This confirms the current production decode path is healthy in an end-to-end server request and did not regress from the bench-only fused MLP experiments.
- It also confirms the rejected MLP fusion ideas should remain bench-only. Direct-op data alone is not enough for a decode conclusion, but the real TPOT check still supports keeping the existing packed gate/up + `PicPackedSiluMul` + cuBLAS down baseline.

## Decode Matrix: Full-Reuse KV Cache

- Ran real `pic_server` decode TPOT for three Jetson CUDA decode-repair model directories, contexts `512/1024/1536`, and `x=0..7`.
- `x=0` is no-repair decode baseline; `x=1..7` all validated as `mnn_token_id_sparse_decode`.
- All rows use `selection_algorithm=full-reuse`, `budget=0.00`, `decode_selector=top_hkvd`, `max_tokens=32`, warm `1`, repeat `1`.
- Raw response validation: `72/72` rows passed; no server log `ERROR`, `unsupported`, `target unavailable`, `async persistent`, `PicBench`, or proxy `Bad Gateway`.

| model | ctx | KV cache | x=0 | x=1 | x=2 | x=3 | x=4 | x=5 | x=6 | x=7 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama3.2-1B silumul | 512 | full-reuse | 28.861 | 30.592 | 35.033 | 37.706 | 37.100 | 37.416 | 38.295 | 39.490 |
| Llama3.2-1B silumul | 1024 | full-reuse | 32.954 | 36.302 | 39.153 | 44.193 | 41.096 | 41.942 | 42.272 | 44.164 |
| Llama3.2-1B silumul | 1536 | full-reuse | 36.302 | 40.559 | 43.029 | 49.336 | 45.338 | 46.363 | 46.618 | 49.011 |
| Llama3.2-1B gateup-packed-silu | 512 | full-reuse | 28.456 | 29.944 | 34.525 | 37.114 | 36.438 | 37.330 | 38.096 | 39.614 |
| Llama3.2-1B gateup-packed-silu | 1024 | full-reuse | 32.432 | 35.248 | 38.683 | 43.129 | 40.477 | 41.643 | 42.140 | 44.074 |
| Llama3.2-1B gateup-packed-silu | 1536 | full-reuse | 35.964 | 40.397 | 42.540 | 48.666 | 45.058 | 45.515 | 46.201 | 48.403 |
| Llama3.2-1B decode-repair | 512 | full-reuse | 29.113 | 30.253 | 34.858 | 37.307 | 36.915 | 37.497 | 38.671 | 39.514 |
| Llama3.2-1B decode-repair | 1024 | full-reuse | 32.742 | 35.799 | 39.341 | 43.641 | 40.719 | 41.868 | 42.434 | 43.964 |
| Llama3.2-1B decode-repair | 1536 | full-reuse | 36.346 | 40.218 | 42.316 | 48.564 | 44.966 | 46.154 | 46.189 | 48.904 |
