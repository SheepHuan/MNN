# Jetson PIC Optimization 2026-07-01 02

## Summary

- Re-exported Llama3.2-1B CUDA decode-repair gate/up fused variants with fresh exporter output.
- Fresh CUDA gate/up exports are not production-ready: `PicLinearNhwcWeightOnly + PicPackedSiluMul`, direct concat `PicPackedSiluMul`, and fixed concat score1/score10 variants all fail during `/v1/prefill/text` with `forwardRaw outputs empty`.
- Historical `gateup-packed-silu-patch` still runs, but end-to-end decode repair x=`1/3/5/7` is slower than the current `silumul-score10` baseline by about `+2.0..+2.6 ms/token` on ctx `512/1024/1536`.
- Do not promote fresh CUDA gate/up graph rewrite or the historical packed patch into the production decode path until fresh export smoke passes and beats the current baseline in end-to-end decode.

## Artifacts

- Run root: `.cache/bench_ops/mlp_fusion_20260701/decode_gateup_packed_fresh_x1357_20260701_095423`
- Main CSV: `benchmark_decode_x1357_rerun.csv`
- Pivot: `pivot_x1357_tpot_ms.tsv`
- Delta: `delta_packed_vs_baseline_x1357.tsv`
