# Jetson CUDA HD128 qtile expansion

## Summary

- Added more HD128 MQTile sparse-attention parameter variants in the existing CUDA kernel family:
  `hd128_q2k16`, `hd128_q2k8`, `hd128_q16k16`, and `hd128_q16k8`.
- Kept the implementation as code-internal fixed heuristics. No env gate, tune cache, or online trial selector was added.
- Did not add K32 variants: the current kernel maps K scores through 16 `lane` values, so `K_TILE > 16` needs a different thread mapping and is not a safe parameter-only variant.
- Initial selector trial used `q2k16` for small fixed-plan rows and showed low-ratio Epic regressions, so small rows were restored to the previous `hd128_q4k8` default.
- Final selector only changes fixed-plan sparse rows with `attnLen >= 384` to `hd128_q16k16`; cacheblend remains on `hd128_q4k16`.

## Final Result

- Tested Jetson AGX Xavier PIC prefill against `benchmark.csv`.
- Excluded Llama3.2 1B and contexts `2048` / `2560`.
- Covered models: MiniCPM5-1B, Llama3.2 3B, Qwen3-4B.
- Covered contexts: `512`, `1024`, `1536`.
- Covered modes: `full-reuse`, `cacheblend`, `epic` where rows exist in `benchmark.csv`.
- Matched benchmark rows: `114`.
- Regressions `>5%`: `0`.
- Worst delta: `Llama3.2 3B / ctx=1536 / cacheblend=0.50`, `3.715375s -> 3.738033s`, `+0.610%`.
- Best delta: `Llama3.2 3B / ctx=1536 / full-reuse`, `1.831451s -> 0.163470s`, `-91.074%`.

## Artifacts

- Combined comparison CSV: `.cache/latency_budget_20260625/jetson_qtile_more_variants_20260704_0215_regression.csv`
- MiniCPM summary: `.cache/latency_budget_20260625/jetson_qtile_more_variants_20260704_0215_minicpm/summary.csv`
- Llama3.2 3B summary: `.cache/latency_budget_20260625/jetson_qtile_more_variants_20260704_0215_llama3b/summary.csv`
- Qwen3-4B summary: `.cache/latency_budget_20260625/jetson_qtile_more_variants_20260704_0215_qwen4b/summary.csv`

## Decision

Keep `hd128_q16k16` as the large fixed-plan sparse default. Keep `q2` variants compiled but not selected by default. Do not add K32 without redesigning the kernel K-lane mapping.
