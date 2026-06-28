# OrangePi 2026-06-25 00

OrangePi headDim=128 稀疏路径恢复、可见 K tile 实验和 tune scaffold 调整记录。

重点：

- 围绕 MiniCPM5-1B / Llama3.2 3B / Qwen3-8B 的 OpenCL sparse slow path 展开排查。
- 记录 q-split 恢复、`COMPUTE_FLOAT4` 修复、cross-model 验证和多变体 tune scaffold。
- 用于后续 headDim=128 sparse flash 默认路径和 canary 决策。

收录章节：

- `2026-06-25 Investigation: MiniCPM5-1B / Llama3.2 3B / Qwen3-8B OpenCL Sparse Slow Path`
- `2026-06-25 Result: headDim-128 sparse q-split recovery on OrangePi`
- `2026-06-25 Experiment: visible-K tile skip on headDim=128 sparse q-split`
- `2026-06-25 Experiment: sparse q-split `COMPUTE_FLOAT4` fix repeat`
- `2026-06-25 Experiment: cross-model validation on OrangePi headDim=128 buckets`
- `2026-06-25 Change: shape-tuned sparse qsplit + OpenCL tune-level override`
- `2026-06-25 Change: headDim=128 later sparse flash canary`
- `2026-06-25 Change: multi-Q-tile sparse flash variants + variant tune scaffold`

文件：

- `README.md`: 简短摘要。
- `context.md`: 完整长日志。
