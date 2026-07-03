# 2026-07-03 17 Jetson CUDA PIC Path Convergence

- 收敛 CUDA `PagedAttentionExecution` 的 sparse qtile 路由：删除请求期 env variant、在线 tune cache、tune scratch output 和未使用的 `hd128_q4k8` CUDA 变体。
- Jetson 默认口径固定为代码内少量参数变体：普通 headDim=128 sparse rows 在 `attnLen>=64` 走 `hd128_q4k16`，更小 active rows 保持既有非 qtile sparse path；decode repair headDim=128 且 `attnLen>=2` 走 `hd128_q8k16`；headDim=64 高预算 sparse rows 走固定 q8/q32-k32 tile path。
- 收敛 CUDA weight-only compact dense 路由：rows45 cuBLAS 固定为现有默认参数，删除 rows48 cuBLASLt、V15 small-M、V16 dp4a 的 env-gated 可执行入口和无用状态。
- 这次只做路径收敛和静态验证；尚未产出新的 Jetson sweep 性能结论。
