# Jetson CUDA PIC Path Convergence

Context:

- 用户要求继续收敛成“Jetson 固定默认实现 + 少量参数变体”，避免 profiling 和回归判断被运行期 gate / tune 混淆。
- 本次范围先限于 CUDA `PagedAttentionExecution` sparse attention 路由，不改 OpenCL / Adreno 历史日志，不改 dense Conv 路由。

Code changes:

- `source/backend/cuda/execution/PagedAttentionExecution.cu`
  - 删除 `MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT` 解析路径、`auto` variant、显式 forced variant 和 support helper。
  - 删除 `MNN_CUDA_PAGED_ATTENTION_QTILE_TUNE` 在线 tune 相关的 shape/runtime key、candidate list、CUDA event timing、global tune cache 和 tune output buffer。
  - 删除 decode repair qtile experiment env gate；decode repair 现在按固定条件选择 `headDim == 128 && attnLen >= 2 -> hd128_q8k16`。
  - 删除未进入固定默认策略的 CUDA `hd128_q4k8` 分支。
  - 保留生产默认：普通 sparse headDim=128 且 `attnLen>=64` 选择 `hd128_q4k16`；tiny active rows 继续走既有非 qtile sparse path；headDim=64 高预算 sparse rows 保留固定 q8/q32-k32 tile path。
- `source/backend/cuda/execution/PagedAttentionExecution.hpp`
  - 删除 qtile tune scratch output 成员和 `ensureQTileTuneOutput()` 声明。
- `.codex/skills/mnn-pic-optimize/SKILL.md`
  - 增加 Jetson CUDA 固定默认实现与少量参数变体的口径，明确不允许请求期 env gate、在线 tune cache 或自动试跑候选作为生产路径。
- `source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu`
  - 固定 rows45 cuBLAS 策略为现有默认：policy=all、tensor math、32F compute、tensor-op default algo。
  - 删除 rows48 cuBLASLt opt-in route、plan cache 和 execution 成员。
  - 删除 V15 rows4/6/8 small-M packed INT4 route、V16 dp4a route、相关 env policy、dispatch、sum-nibble 预计算状态和无引用 kernel。
- `source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.hpp`
  - 删除 rows48 cuBLASLt plan 指针和 V16 `mSumNibble` 状态。

Validation:

- `rg` confirmed no remaining CUDA symbols for qtile env/tune/experiment paths:
  - `QTILE_TUNE`
  - `QTILE_VARIANT`
  - `DECODE_REPAIR_QTILE`
  - `cudaSparseQTileVariantFromEnv`
  - `kCudaSparseQTileAuto`
  - `ensureQTileTuneOutput`
  - `mQTileTune`
- `rg` confirmed no remaining CUDA dense env-gated symbols for:
  - `MNN_CUDA_PIC_*`
  - `ROWS48`
  - `SMALLM`
  - `V15`
  - `V16`
  - `DP4A`
  - `mSumNibble`
  - `mPicRowsCublasLtPlan`
- `git diff --check` passed for the touched CUDA files and Jetson benchmark frequency-lock docs/scripts.

Open work:

- Full Jetson build / device sweep was not run in this step because no existing local CMake build directory was present. Next verification should build the Jetson CUDA artifact and run prefill sweep without 2048/2560 cases, then compare cacheblend / epic / full-reuse against the current benchmark baseline.
- If future dense variants are needed, add them as fixed code heuristics after Jetson sweep validation instead of env-gated production routes.
