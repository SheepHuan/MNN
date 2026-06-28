# 2026-06-28 06:00 OrangePi Mali CacheBlend Top-K Fix

本小时记录 Qwen3-8B `ctx2560 cacheblend 0.20/0.30` 在 OrangePi Mali 上 graph-boundary cacheblend warm 阶段 HTTP 500 的根因和修复。

## 结论

- 失败不是 OOM；MNN 返回 `code=5`，对应 `INVALID_VALUE`。
- Adreno 不失败而 Mali 失败的关键差异在 OpenCL cacheblend staged top-k：Mali-G610 虽然报告 32KiB local memory，但 16KiB 级动态 local allocation 的 staged top-k 会产生非法/重复 selected rows 或执行失败。
- 修复为 Mali 上把 cacheblend staged top-k 可用 local memory 保守裁剪到 8KiB；超过该阈值的高 topK ratio 直接使用 legacy top-k，并为 staged 失败/非法 selected rows 增加 fallback 与错误日志。
- 验证通过并已合并 `benchmark.csv`：Qwen3-8B `ctx2560 cacheblend 0.20 = 47.648407s`，`0.30 = 46.149394s`。

