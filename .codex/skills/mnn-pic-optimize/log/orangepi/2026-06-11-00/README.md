# OrangePi 2026-06-11 00

OpenCL PIC graph-boundary 基线、profile detail 和 sparse flash 主路径收敛记录。

重点：

- 记录 1024-token cacheblend/epic baseline、full-reuse 对比和初始瓶颈定位。
- 覆盖 score-layer sparse flash、LWS tuning、direct-value prefill、compact dense GEMM 等阶段性决策。
- 作为后续 OrangePi OpenCL 性能回归和语义对齐的起始参考。

收录章节：

- `Current Baseline: 2026-06-11`
- `Profile Detail: 2026-06-11`
- `Optimization Candidates`
- `Future Update Template`
- `YYYY-MM-DD Experiment: <short name>`
- `2026-06-11 Experiment: Sectioned Attention Profile`
- `2026-06-11 Experiment: FlashMask Range Pieces + MNN Tune Cache`
- `2026-06-11 Experiment: Direct Value Prefill A/B`
- `2026-06-11 Design: Fused Sparse FlashAttention Mainline`
- `2026-06-11 Decision: Keep LWS Tuning in MNN Autotuning`
- `2026-06-11 Experiment: CacheBlend 50% Regression After LWS-Cache Retune`
- `2026-06-11 Update: Confirm Later SparseAttention Bottleneck`
- `2026-06-11 Experiment: Sparse FlashAttention Row64 V1`
- `2026-06-11 Implementation: Make Sparse FlashAttention the Only OpenCL Sparse Fast Path`
- `2026-06-11 Analysis: Current Layer-Type Bottleneck After Default Sparse Flash`
- `2026-06-11 Implementation: Sparse Flash Lane Tuning + Score-Layer Flash`
- `2026-06-11 P2: Graph-Level Dense Profile`
- `2026-06-11 Attention-Only: Why Sparse Flash Speedup Is Lower Than Expected`
- `2026-06-11 P0a Implementation: CacheBlend Single-Piece Sparse Flash`
- `2026-06-11 P0b Implementation: Private Sparse-Flash Output Accumulators`
- `2026-06-11 P2 Fix: PIC Compact Dense GEMM Fast Path`

文件：

- `README.md`: 简短摘要。
- `context.md`: 完整长日志。
