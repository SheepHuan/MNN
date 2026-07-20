# Jetson CUDA 3072 Prefill Workspace Bound

## Summary

- Root cause: Qwen3-4B 3072 `prefill/text` exhausted Jetson memory while each PagedAttention layer retained 1024-row QK and softmax workspaces.
- Fix: cap Jetson prefill Q pieces at 256 rows in `PagedAttentionJetsonPolicy::prefillQSplitNum`.
- Jetson Qwen3-4B 3072 text-cache build, full-reuse, and all six cacheblend ratios completed successfully after the fix.
- Decode behavior is unchanged because decode uses `attnLen=1` and keeps one Q piece.

## Results

- normal-full-recompute: `44.4449666 s`
- full-reuse: `0.240277 s`
- cacheblend: `4.792358 / 8.020361 / 14.099130 / 20.523153 / 26.219778 / 30.845847 s` for `0.05 / 0.10 / 0.20 / 0.30 / 0.40 / 0.50`
- No failures or duplicate benchmark keys.

## Artifacts

- Build run: `.cache/build/mnn/jetson_cross`
- Installed artifact: `.cache/output/mnn/artifacts/jetson`
- Remote CUDA library synced to Jetson `jetson_cross_cuda` artifact root.
- Prefill summaries:
  - `.cache/latency_budget_20260625/qwen3_4b_jetson_3072_prefill_fix_20260720/summary.csv`
  - `.cache/latency_budget_20260625/qwen3_4b_jetson_3072_cacheblend_fix_20260720/summary.csv`

