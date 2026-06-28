# Jetson 2026-06-11 00

Jetson CUDA graph-boundary 适配、tile sparse flash 路径筛选和 compact dense 优化记录。

重点：

- 记录 CUDA graph-boundary 接入、accepted path 和多轮 negative A/B。
- 覆盖 tile sparse flash、static dequant cache、SM70 compact CUTLASS、half2 SiLU 等关键取舍。
- 用于约束 Jetson cacheblend/epic sweep 的正式默认路径。

收录章节：

- `2026-06-11 Jetson CUDA Graph-Boundary Adaptation`
- `2026-06-11 Jetson CUDA P0/P1 Negative A/B`
- `2026-06-11 Jetson CUDA Tile Sparse Flash Accepted Path`
- `2026-06-11 Jetson CUDA Causal K Limit for Tile Flash`
- `2026-06-11 Jetson CUDA Cacheblend Hybrid Sparse Flash`
- `2026-06-11 Jetson CUDA Further Optimization Potential After Hybrid`
- `2026-06-11 Jetson CUDA P0 Tail K/V Cut and P1 Dequant Profile`
- `2026-06-11 Jetson CUDA Further Potential Recheck`
- `2026-06-11 Jetson CUDA Cacheblend Score Source Slots`
- `2026-06-11 Jetson CUDA Static Dequant Cache For Compact Dense`
- `2026-06-11 Jetson CUDA Sparse Tile Full-Causal Mask Skip`
- `2026-06-11 Jetson CUDA High-Budget q16 Sparse Tile`
- `2026-06-11 Jetson CUDA High-Budget q32 Sparse Tile`
- `2026-06-11 Jetson CUDA Sparse Tile K-Slot Cache`
- `2026-06-11 Jetson CUDA q32 Threshold 384`
- `2026-06-11 Jetson CUDA SM70 Compact CUTLASS Dense Fast Path`
- `2026-06-11 Jetson CUDA half2 SiLU Unary Fast Path`
- `2026-06-11 Jetson CUDA Rejected Sparse Flash Register Accumulator`
- `2026-06-11 Jetson CUDA SM70 Compact CUTLASS Hybrid Tile`
- `2026-06-11 Jetson CUDA Rejected Mid-Budget q16 Sparse Tile`

文件：

- `README.md`: 简短摘要。
- `context.md`: 完整长日志。
