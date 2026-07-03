# Jetson CUDA qtile variant sweep

## Summary

- Code change: kept Jetson CUDA on one fixed MQTile sparse-attention family and added small code-internal HD128 variants.
- No new env gate or online tuning path was added.
- Tested Jetson AGX Xavier PIC prefill against `benchmark.csv` on non-Llama1B models and contexts `512,1024,1536` only.
- Matched benchmark rows: `114`.
- Regressions `>5%`: `0`.
- Worst delta: `Llama3.2 3B / ctx=1536 / cacheblend=0.50`, `3.715375s -> 3.760861s`, `+1.224%`.
- Best delta: `Llama3.2 3B / ctx=1536 / full-reuse`, `1.831451s -> 0.162303s`, `-91.138%`.

## Artifacts

- Comparison CSV: `.cache/latency_budget_20260625/jetson_qtile_variants_20260704_0142_regression.csv`
- MiniCPM5-1B summary: `.cache/latency_budget_20260625/jetson_qtile_variants_minicpm5_20260704_0142/summary.csv`
- Llama3.2 3B summary: `.cache/latency_budget_20260625/jetson_qtile_variants_llama32_3b_20260704_0142/summary.csv`
- Qwen3-4B summary: `.cache/latency_budget_20260625/jetson_qtile_variants_qwen3_4b_20260704_0142/summary.csv`

## Decision

The added fixed-plan qtile variants do not cause a Jetson prefill regression in this covered matrix. Keep the implementation converged as fixed default code heuristics, not env-gated alternatives.
