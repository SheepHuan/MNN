# Small-M Weight-Only Linear 优化思路总结

本文档系统梳理 Jetson AGX Xavier (sm_72) 上 decode-repair `x=1→x=3`（及 x=3/5/7）
TPOT 悬崖的所有优化思路：哪些已尝试且**无效**（证伪）、哪些**尚未做**但受 PLAN
硬约束排除、哪些是真正剩下的**潜在非-int4 杠杆**。

配套：`README.md`（Update 1-4 实测过程）、`context.md`（设计与带宽分析）、
`raw/jetson_directop_v15.txt`（原始 direct-op 日志）、`CUDA_SMALL_M_WEIGHT_ONLY_LINEAR_OPT_DESIGN.md`（仓库根设计文档）。

---

## 1. 悬崖根因（已确认）

decode-repair 下 `pic_decode_repair_sparse=true`，`int4GemvBatchLimit` 从 6 降到 3
（`ConvFpAIntBExecution.cu:2484`）。于是：

| x (repair_tokens) | batch=active_rows | 命中路径 |
|---|---|---|
| 0 | 1 | V14 tiny-GEMV (int4) |
| 1 | 2 | V14_MB\<2\> (int4) |
| 3 | 4 | **跳过 GEMV → cuBLAS/CUTLASS dense (FP16 权重)** |
| 5 | 6 | 同上 |
| 7 | 8 | 同上 |

`x=1→x=3` 悬崖 = batch=2 (V14_MB int4, 0.23ms/op) → batch=4 (cuBLAS FP16, 0.31ms/op)。
单 op 增量 ~0.08ms，但 28-36 层 × 多个 Linear 累积成 +10~55ms/TPOT（见
`jetson/2026-07-01-18` 归因表）。

---

## 2. 已尝试且无效的路线（全部 Jetson direct-op 实测证伪）

### 2.1 int4-native GEMV 路线（nibble ALU 瓶颈）— 无效

| 变体 | 思路 | batch=4 实测 | 失败原因 |
|---|---|---|---|
| V14_MB retune (SKILL 行 250-251) | 放宽 int4GemvBatchLimit 3→5/6，枚举 OC_PER_BLK=2/3/4/8/16 | 0.40ms (OC=4) | nibble ALU instruction-bound，随 batch 线性退化 |
| V15 v1 | shared-mem weight tile，4 线程 load | 2.07ms | weight load 串行化（仅 4/128 线程发 ldg） |
| V15 v2 | batch-pair 寄存器 load | 0.74ms | 每 batch-pair 重读整层权重 |
| V15 v3 | single-load + inner batch loop (V14_MB 结构) | 1.39ms | OC_PER_BLK=8 时 64 acc 寄存器溢出 |

**统一失败根因**：sm_72 上 int4 nibble 的 shift/mask/int2float/FMA 是
instruction-bound，不是 bandwidth-bound。int4 权重 4× 带宽红利被 per-nibble ALU
开销吃掉。V14_MB batch=4=0.40ms 已慢于 cuBLAS 0.31ms，任何同族 kernel 不可能更快。

### 2.2 INT8 `__dp4a` 路线（寄存器瓶颈）— 无效

| 变体 | 思路 | batch=4 实测 | 失败原因 |
|---|---|---|---|
| V16 dp4a v1 | per-(oc-chunk,group) grid | 21.48ms | block 数爆炸 (32k blocks → launch-bound) |
| V16 dp4a v2 | in-block group loop | 45.32ms | register spill 到 DRAM (单 launch 44ms) |

**数学正确**（accuracy `bad=0` bit-exact）：`acc = s_w·s_a·dp4a + s_w·o_a·sum_nibble +
adj_w·s_a·sum_aq + adj_w·o_a·K`，nibble 保持精确、A 运行时量化 int8。但 factored
per-(oc,group) combine 需在 32 group 循环里维持太多状态，spill。**瓶颈是寄存器，
不是数学。**

### 2.3 int4 fused-MLP 路线（SKILL 行 254-259）— 无效

| 变体 | 思路 | rows4/6/8 delta | 失败原因 |
|---|---|---|---|
| PicBenchFusedPackedSiluDown | cuBLAS down + Silu 融进 bench Extra | -0.0015/+0.0031/-0.0017ms | 无 headroom |
| PicBenchStreamedPackedSiluDown | 避免 [rows,8192] 写回，scalar 累加 | +2.15/+3.49/+4.75ms | scalar 累加远慢于 tensor-core |
| PicBenchWmmaPackedSiluDown | naive 16×16 WMMA down | +0.58/+0.59/+0.65ms | WMMA schedule 慢于 cuBLAS down |

### 2.4 fresh gate/up packed 图改写（SKILL 行 261）— 无效

`PicLinearNhwcWeightOnly + PicPackedSiluMul`、direct concat、历史 concat_conv layout：
`/v1/prefill/text` 失败（`forwardRaw outputs empty seq_len=512`），或端到端比 baseline
慢 +2.0~+2.6ms/token。

### 累计：5 大类、8 个具体 tile 全部证伪

后续**不应再**以 int4-native / int4-dp4a / int4 fused-MLP kernel 作为 Jetson
rows4/6/8 small-M 主线。

---

## 3. 尚未做但受 PLAN 硬约束排除的方向

PLAN (`CUDA_SMALL_M_WEIGHT_ONLY_LINEAR_OPT_PLAN.md` Non-Goals) 明确排除：

| 方向 | 为什么被排除 | 理论上是否有潜力 |
|---|---|---|
| Fresh gate/up packed exporter 集成 | 改图导出，破坏现有产物 | 有（减半 Linear 数量），但已证伪（§2.4） |
| `PicLinearNhwcWeightOnly` 图改写 | 要求新 op type / graph rewrite | 有，但端到端慢 +2ms |
| 放宽 V14 tiny GEMV 到 rows4/6/8 默认 | 已证伪（§2.1） | 无 |
| naive INT4 `PicGEMM` | Non-Goal | 无（同 nibble ALU 瓶颈） |
| 进一步调 PicSparseAttention | x=1→x=3 不是 attention 主导（归因表 PicSparseAttention -0.15~-0.23） | 无 |

这些方向即使做也多数已证伪或违反硬约束，不应投入。

---

## 4. 真正剩下的潜在非-int4 杠杆（未被排除、未尝试）

这些是 int4 路线穷尽后**真正还没做**的方向，但各有代价或风险：

### 4.1 减少 x=3 命中的 dense Linear 数量（杠杆最大，但触及图/调度）

`x=3` 的 batch=4 进 dense 路径是因为有 4 个 active rows。若能让部分 Linear 不在
batch=4 下全部走 dense——例如：
- **decode repair selector 选 fewer rows**：但 `tokens_per_decode_step` 是语义参数，
  x=3 本身是用户配置，不能为性能改语义。
- **shared/gate-up packed 融合**（不改导出，runtime 侧把 gate+up 两次 dense 融成一次
  [batch, 2×inter]）：权重读从 2 次变 1 次，省 ~1× 权重带宽。但要求两个 op 权重连续
  且 shape 对齐，runtime 调度复杂；且 PLAN 行 99 把 `PicPackedSiluMul` 作图改写排除。
  **runtime 侧融合（不改图）是否可行未验证**——这是少数还没试且有理论 2× 空间的方向。

### 4.2 INT8 tensor-core GEMM（非 dp4a，真 tensor core）

Explore 确认 MNN 有 CUTLASS INT8 tensor-core GEMM，但**硬门控 sm_75+**
（`ConvInt8CutlassExecution.cu:368`，sm_72 直接 return）。sm_72 (Volta) 的 INT8
tensor-core MMA (`mma.m8n8k16.s8`) MNN 未实例化。要启用需：
- 写 sm_70/72 的 INT8 `OpClassTensorOp` CUTLASS op（MNN 没有）
- 或手写 `mma.m8n8k16.s8` PTX kernel
- A 侧 INT8 量化 + 权重 int4→int8（或 int8 模型）

**理论潜力**：INT8 tensor core 算力远高于 dp4a，可能避开寄存器溢出。但工作量巨大，
且 weight-only 模型是 int4，需 int4→int8 上采样（2× 权重字节，抵消一半带宽红利）。
**未尝试，风险高，但未证伪。**

### 4.3 换设备（非 Jetson 杠杆）

V16 dp4a 数学 bit-exact 正确，瓶颈纯在 sm_72。sm_90+ (Hopper) 有原生 int4 tensor
core + INT8 tensor core，V15/V16 代码可直接复用跑 direct-op。**未尝试**（无该设备）。

### 4.4 cuBLASLt 替代 cuBLAS（微优化，非主线）

现 rows45 用 `cublasGemmEx`，rows48 用 cuBLASLt（仅 Qwen3 3072↔8192）。扩展
cuBLASLt 覆盖更多 shape 可能有小幅 headroom（~0.01-0.05ms/layer 量级，SKILL 行 246
tune 数据），但不足以闭合悬崖（缺口 ~0.6ms/token）。**未系统尝试，但上限低。**

---

## 5. 结论

| 维度 | 状态 |
|---|---|
| int4-based Linear kernel（V14/V15/V16 全族） | **已穷尽，全部证伪**，不再尝试 |
| 悬崖能否用 int4 闭合 | **不能**（nibble ALU + dp4a 寄存器双瓶颈） |
| 生产路径 | **不变**（rows45 cuBLAS static-dequant，env off） |
| 剩余潜在杠杆 | runtime gate/up 融合（未试，2× 理论）、INT8 tensor-core（未试，需新 CUTLASS op）、换设备（sm_90+）、cuBLASLt 扩展（上限低） |
| 资产保留 | V15/V16 env-gated 代码 + bit-exact accuracy harness（供未来设备复用） |

**Jetson sm_72 上 small-M weight-only Linear 的 int4 优化空间已关闭。** 任何后续优化
必须是非-int4 杠杆（§4），且多数触及 PLAN 硬约束或需重大新基础设施投入。
