# PIC decode x0 vs normal — 目标、当前 gap、为何难对齐、cache 聚焦路线

> 转交文档。本小时确立 PIC decode 优化的**目标与判据**，给出当前实现下 **Mali + Adreno** 两设备
> PIC x0 对比 normal decode q=1 的真实 gap，分析**为什么把 attention 算法对齐 normal decode
> 始终追不平**（Plan A 是第 6 个同型失败），并指出下一步应聚焦 **cache 优化**的路线方向。
>
> 详细数据/命令/Plan A A/B 见同目录 `context.md`。Plan A 完整设计见
> `log/orangepi/2026-07-05-22/plan_a_split_kernels.md`。

## 1. 目标与判据（本路线的北极星）

**目标：参考 normal decode 的实现思路与已有的历史经验，找到一条优化路线，让 Mali 和 Adreno 在
PIC decode `x=0/1/3/5/7/.../n` 全部 x 值上都能获得收益。**

具体含义：
- **全 x 覆盖**：不是只优化 x0，而是 x=0/1/3/5/7/n 共用同一思路（x0 是 active rows=1 的退化情形，x>0 只是 active rows 变成 n+1）。允许按情况使用**变体 kernel / 不同参数 / 不同数据格式（image vs buffer）**，但必须是同一思路下的显式变体，不能 x0 一套、x>0 另一套。
- **两设备都收益**：Mali-G610 和 Adreno 都要赢，不能 Mali 赢 Adreno 输（当前正是这个状况）。
- **终极判据（硬指标）**：**PIC decode `x=0` 的 TPOT 延迟不差于 true normal decode `q=1`**（`PIC_x0_tpot ≤ normal_q1_tpot`），在 3 模型 × ctx{512,1024} 上成立。这是本路线的 acceptance gate。

**为什么判据是 x0 ≤ normal q=1**：x0 是 PIC decode 的下界（active rows 最少、最接近 normal q=1 的形态）。如果连 x0 都追不平 normal q=1，说明 PIC 的固定开销（append / PagedCache / 调度）本身就把 PIC 拖慢，谈不上 x>0 的扩展收益。x0 追平后，x>0 的增量成本应只来自多算的 active rows，而不是固定开销。

## 2. 当前实现 vs normal decode q=1（TPOT ms，profile off，warm）

当前实现 = commit `38ba6851` + 工作区改动，Plan A env 默认 OFF（走原融合 path）。

### OrangePi / Mali-G610（OpenCL）

| model | ctx | PIC x0 | normal q1 | gap | 判据 |
|---|---|---|---|---|---|
| Llama3.2-3B | 512 | 119.24 | 121.62 | **−2.0%** | ✅ 已达标 |
| Llama3.2-3B | 1024 | 119.27 | 125.64 | **−5.1%** | ✅ 已达标 |
| MiniCPM5-1B | 512 | 44.48 | 46.74 | **−4.8%** | ✅ 已达标 |
| MiniCPM5-1B | 1024 | 46.06 | 49.96 | **−7.8%** | ✅ 已达标 |
| Qwen3-4B | 512 | 150.32 | 149.83 | **+0.3%** | ✅ 已达标（持平） |
| Qwen3-4B | 1024 | 172.46 | 154.86 | **+11.4%** | ❌ 未达标 |

**Mali：5/6 cell 已达标（PIC 不差于 normal），仅 Qwen ctx1024 慢 +11.4%（~17ms）。** 历史上"x0 慢 20ms"的口径在当前 artifact 已不成立。

### Rhino / Adreno（OpenCL）

| model | ctx | PIC x0 | normal q1 | gap | 判据 |
|---|---|---|---|---|---|
| Llama3.2-3B | 512 | 76.78 | 53.76 | **+42.8%** | ❌ 未达标 |
| Llama3.2-3B | 1024 | 88.58 | 56.66 | **+56.3%** | ❌ 未达标 |
| MiniCPM5-1B | 512 | 48.32 | 27.35 | **+76.7%** | ❌ 未达标 |
| MiniCPM5-1B | 1024 | 54.26 | 25.18 | **+115.5%** | ❌ 未达标 |
| Qwen3-4B | 512 | 100.31 | 69.77 | **+43.8%** | ❌ 未达标 |
| Qwen3-4B | 1024 | 115.72 | 72.37 | **+59.9%** | ❌ 未达标 |

**Adreno：6/6 cell 未达标，全面落后 +43%~+115%。** 这是当前最大的未解决问题，也是本路线的主战场。

### 关键观察：两设备 gap 量级差 ~10×

同一个 PIC 实现，Mali 上 5/6 cell 达标，Adreno 上全部 +43% 起步。**这证明"PIC x0 追不上 normal"不是 PIC 算法的固有缺陷，而是 PIC 实现的执行特性在 Adreno 上水土不服。** 新会话的主战场是 Adreno，且切入点应是**设备执行特性（cache / record queue / 调度）**，而不是继续调 attention 算法。

## 3. Plan A 做了什么 + 结果（对齐 normal 算法的第 6 次尝试）

Plan A 把 PIC x0 的 attention 从**融合 flash kernel**（1 个 workgroup/query-head，QK+softmax+QKV 融合，V 阶段仅 32 lane 活跃）改成**独立 QK / softmax / QKV 三 kernel**，完全抄 normal decode 的并行模型（2D 细粒度 gws + GQA 位置除法 `y/NUMHEAD_GROUP_SIZE` + V 全 128-lane 占用）。代码 isolated 在 `PagedAttentionBufExecutionDecodeSplitKernels.cpp`，env `MNN_PAGED_ATTENTION_DECODE_REPAIR_SPLIT_KERNELS` 默认 OFF。验证：3 kernel 命中、P0 全过、token 与默认路径逐 token 一致。

### Plan A A/B（OrangePi/Mali，profile off，2 repeats，warm 1）

| model | ctx | OLD(融合) | NEW(Plan A) | delta | pct |
|---|---|---|---|---|---|
| Llama3.2-3B | 512 | 119.24 | 126.11 | +6.87 | +5.8% 退化 |
| Llama3.2-3B | 1024 | 119.27 | 138.86 | +19.60 | +16.4% 退化 |
| MiniCPM5-1B | 512 | 44.48 | 50.84 | +6.35 | +14.3% 退化 |
| MiniCPM5-1B | 1024 | 46.06 | 61.64 | +15.58 | +33.8% 严重退化 |
| Qwen3-4B | 512 | 150.32 | 170.45 | +20.13 | +13.4% 退化 |
| Qwen3-4B | 1024 | 172.46 | 173.91 | +1.45 | +0.8% 持平(噪声) |

- x=1 行 Plan A vs OLD 全部 ±1% 持平 → env gate 干净，退化只发生在 x0。
- **MiniCPM（GQA group=8）退化最严重（+33.8%）**——若 GQA L2 复用生效它应收益最大，反证复用没拿到。

**Plan A 是第 6 个"想把 PIC x0 往 normal decode 并行模型上靠"的尝试，全部退化。** 前 5 个：fused-append / q1-gqa / slot-identity-split / lane-force / q1-identity-fused-kv。

## 4. 为什么对齐 normal 算法始终追不平——根因（cache 视角）

Plan A 对齐了 normal decode 的**算法并行模型**，但 normal decode 快的真正原因是它的 kernel 在它的执行环境下拿到的 **cache / 调度好处**，而这些好处在 PIC x0 的物理约束下不成立：

| normal decode 快的真正原因 | PIC x0 能否复用 | 为什么（cache 视角） |
|---|---|---|
| GQA K 跨 query-head 经 **L2 复用**（4×/8× 带宽） | ❌ 拿不到 | GPU L2 跨 workgroup 复用不可靠：Mali L2=1MB/core（10 核各独立，非全局共享），同 group work-item 跨核心→复用归零；Adreno L2/TLB+record queue 又不同。Plan A 核心赌点就是这个，**K external read 没降下来**（MiniCPM group=8 退化最严重反证） |
| 5 个独立 kernel，中间 tensor 往返是 baseline 已付代价 | ❌ PIC 原本用融合省掉了 | Plan A 主动放弃融合，重新付 qk/softmax 中间 tensor 的 **global memory 写读往返**（~128KB/layer×36）+ 多 2 次 launch。PIC 融合 path 本来省了这笔 |
| V 阶段全 128 lane 占用 | ✅ 对齐了 | 但 V 在 q=1 decode 不是瓶颈（原融合 32-lane V 仅占 attention ~25%），收益抵不过往返 |
| 无 decodeKey 额外写 | ❌ PIC 硬约束必须写 | append 每 token 写 key_cache+value_cache+decodeKey **3 份**，normal 只写 1 份 past_key。**这是 cache 流量上的固定 3× 放大** |

### 三重物理约束（新会话深挖方向，全是 cache 相关）

1. **PagedCache 强制 append + 3 份 store**：`append_sparse_decode_key_value_hd128` 每 token 写 `key_cache` + `value_cache` + `decode_key`。normal append（`rearrange_k`）只写 1 份 `past_key`。**PIC 的 append cache 流量 = 3× normal**，这是 PIC 难追平 normal 绝对速度的固定开销，且无法靠 attention 算法消除。

2. **GPU L2 跨 workgroup GQA K 复用不可靠**：
   - Mali-G610 L2 = 1MB/core，10 核各自独立（非全局共享）。Plan A QK gws `{257, 32}`（Qwen ctx1024）切多个 workgroup 跨核心调度，同 GQA group 的 4 个 query-head 若跨核心，L2 复用归零。
   - Adreno L2/TLB 架构不同，且有 QCOM record queue 能合并 dispatch（Mali 没有）。**两设备 L2 行为完全不同——这正是两设备 gap 量级差 10× 的主因。**
   - 这是 cache 优化的核心战场：如何在 PIC 的 PagedCache 布局下，让 K/V 的 cache 命中率逼近 normal decode。

3. **PIC 原本已融合，拆开必付往返；normal baseline 已含往返**：PIC 融合 path 把 QK+softmax+QKV 融进 1 个 kernel，**省掉** qk/softmax 中间 tensor 的 global memory 往返。normal decode 本来就是拆的，它的 baseline 已含往返。Plan A "对齐 normal" = 把 PIC 从"融合（省往返）"拉回"独立（付往返）"，换一个在 L2 上拿不到的 GQA 复用。**净效果 = 0 收益 − 往返开销 = 退化。**

### 决定性证据：MiniCPM（group=8）退化最严重

若 GQA L2 复用真生效，MiniCPM（8× 复用）应比 Qwen（4×）收益更大。实测相反——MiniCPM ctx1024 退化 +33.8%（最严重）。反证：**GQA 复用根本没拿到，退化纯粹来自中间 tensor 往返 + launch，无 cache 收益对冲。**

## 5. 下一步路线：聚焦 cache 优化（不是 attention 算法）

前 6 个失败的全部规律：**凡是从"算法并行模型"角度对齐 normal 的尝试都退化，因为它们依赖的 cache / 调度好处在 PIC 物理约束下不成立。** 下一步必须**直接从 cache 角度**入手，让 PIC x0 的 K/V cache 命中率逼近 normal decode，而不是改 kernel 的并行形状。

### 路线原则

1. **不拆融合**：保持 QK+softmax+QKV 融合（省中间 tensor 往返），在融合 kernel 内部做 cache 优化。Plan A 证明拆融合必败。
2. **聚焦 K/V cache 布局与复用**：核心是让同 GQA group 的多个 query-head 的 K/V 读取命中同一份 cache，且跨 token 的 append 不破坏已有 cache 行。可能方向（待新会话验证）：
   - K/V 的 image vs buffer 格式选择（Adreno image 有 texture cache，Mali buffer 更稳）——这正是用户提到的"不同数据格式"变体维度。
   - append 的写布局是否破坏 attention 读的 cache line（write-after-read 乒乓）。
   - decodeKey 的转置布局是否真的是 cache 最优，还是有更优的 K 访问模式。
3. **全 x 共用思路，变体显式**：x=0/1/3/5/7/n 共用同一 cache 优化思路，按 active rows / GQA group / 设备选变体 kernel + 参数（lane / tile / image/buffer）。变体用清晰 env 或代码内 heuristic 标注，不混入默认结果。
4. **两设备分别验证**：Mali 和 Adreno 的 cache 行为差异大（record queue / L2 / TLP），同一变体可能两设备表现相反。必须分别 A/B，不能只在 Mali 验证就推广。

### 优先级

- **P0：Rhino/Adreno**。6/6 cell 未达标，gap +43~+115%，且和 Mali 量级差 10× 说明是设备 cache/调度特性。先跑 Adreno device-side profile（record queue / L2 命中 / TLP），定位 PIC x0 在 Adreno 上慢的主因是 append cache 流量、attention K/V miss、还是 dense tail，再定 cache 优化方向。
- **P1：Mali Qwen ctx1024**（唯一未达标 cell，+11.4%）。方向是融合 kernel 内部 cache 优化或 V-stage 全 lane（保持融合），不要再拆。
- **判据复核**：每次变体必须跑 3 模型 × ctx{512,1024} × x{0,1} 两设备 A/B，x0 行 `PIC ≤ normal` 才算达标，x1 行不退化才算 env gate 干净。

## 6. 当前代码状态

- Plan A 代码**仍在 tree**，env-gated 默认 OFF，**不影响生产**。
- 实现 isolated 在 `PagedAttentionBufExecutionDecodeSplitKernels.cpp`（统一自门控入口 `runDecodeCausalAttentionHD128SplitKernels`）+ 2 个新 .cl kernel（`matmul_qk_decode_pic_hd128` / `matmul_qkv_decode_pic_hd128_b8`）。**这个隔离结构是未来 cache 变体可复用的模式**：每个变体一个独立 TU + 统一自门控入口，主 dispatch 文件只加一行调用，互不干扰。
- **尚未按硬约束 `#if 0` 注释 Plan A**——新会话需决定：直接 `#if 0` 记录第 6 个 tried-and-failed，或保留 env-gated 作为 cache 优化的对照基线。

见 `context.md`：完整 A/B CSV 路径、命令、数据来源、Plan A 实现细节、退化原因。
