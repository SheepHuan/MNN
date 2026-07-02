# Rhino Decode x0 P0/P1 专门路径

## 摘要

本小时验证 MiniCPM5-1B / Rhino Pi-X1 / Adreno OpenCL 的 PIC full-reuse no-repair decode x0 专门路径。

结论：

- P0 `headDim=128, q=1, identity slot` fused decode path 已接入默认路由，确认不走 repair/sparse/rank。
- P0 没有把 x0 拉近 normal LLM：ctx512 仍约 `64.37 ms/token`，ctx1024 约 `105.87 ms/token`。
- P1 transposed-key mirror + normal-style QK/softmax/QKV 退化：ctx512 `100.11 ms/token`，ctx1024 `131.49 ms/token`，不能作为默认路径。
- 当前最大差异不在 slot_table identity lookup，也不是 decode_refine repair MLP；需要继续拆 normal LLM graph/runtime 与 PIC graph-boundary/PagedAttention 之外的阶段差异。

默认生产路由已退回 P0 fused identity。P1 transposed-key 只保留为 `MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_IDENTITY=1` 诊断开关。
