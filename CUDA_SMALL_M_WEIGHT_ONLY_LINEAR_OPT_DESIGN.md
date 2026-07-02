# CUDA Small-M Weight-Only Linear — 优化设计与 Jetson 可行性分析

本文档是 `CUDA_SMALL_M_WEIGHT_ONLY_LINEAR_OPT_PLAN.md` 的工程落地分析与算子设计。
目标：在不破坏现有函数、不改图导出的前提下，新增一个 rows4/6/8 INT4-native
weight-only Linear 路由，分析在 Jetson AGX Xavier 上能否把 decode-repair 的
`x=3/5/7` TPOT 拉近到 `x=0/1`。

设计原则（来自 PLAN 的硬约束）：**只新增算子/路由，不改动现有 V14 / cuBLAS /
CUTLASS 路径**，旧路径保留为 fallback。

---

## 1. 现状路由回顾（为什么 x=1→x=3 有悬崖）

`source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu` 在
INT4 + 1x1 + fp16/mix 路径下的实际分派（`onExecute`, 行 2300-2618）：

```
batch = inputs[0]->batch()                         // = active_rows = repair_tokens + 1
int4GemvBatchLimit = picDecodeRepairSparse ? 3 : 6 // 行 2292

if (batch <= int4GemvBatchLimit)        → V14 / V14_MB tiny-GEMV   (行 2324)
else:                                   → rows48 cuBLASLt           (行 2462)
                                          rows45 cuBLAS              (行 2458)
                                          generic CUTLASS + runtime/static dequant (行 2527)
```

decode-repair 下 `picDecodeRepairSparse=true`，所以：

| x (repair_tokens) | active_rows=batch | 命中路径 |
|---|---|---|
| 0 | 1 | V14 (batch==1) |
| 1 | 2 | V14_MB\<2\> |
| **3** | **4** | **跳过 GEMV，进 rows48/rows45/generic** |
| 5 | 6 | 同上（batch=6） |
| 7 | 8 | 同上（batch=8） |

`x=1→x=3` 悬崖的根因：**batch=4 越过 GEMV 阈值（3），从 tiny-GEMV 切到 dense
GEMM 路径**。两条路径的代价差异不是算力，而是**权重读取量与读取格式**。

### 1.1 tiny-GEMV 为什么快（V14 的关键 trick）

V14 / V14_MB（行 1334 / 1442）用 `PrecomputeGemvParams`（行 552）把每量化组的
`{scale, offset}` 预算成 `float2{scale, adj_offset=offset-8*scale}`。于是反量化
`w_deq = (nibble-8)*scale + offset = scale*nibble + adj_offset` 可以**因式分解**：

```
acc += w_deq * in = scale * (Σ nibble*in) + adj_offset * (Σ in)
```

- 权重**保持 int4 packed**（每个权重 byte = 2 nibble），**不展开成 FP16**。
- 一个 block 处理一组 OC，权重只读 1 次，`MAX_BATCH` 个 batch 行共享同一份权重
  （V14_MB 行 1440-1441 注释明说 "reducing memory bandwidth by ~batch×"）。
- 热循环里没有 dequant，只有 `raw_dot += in * nibble`。

### 1.2 batch≥4 的 dense 路径为什么慢

rows45 cuBLAS / rows48 cuBLASLt / generic CUTLASS（行 2457-2614）都消费
**已反量化的 FP16 权重** `mDequantFilter [ocp, icp]`：

- 若 `staticDequant` 命中（行 2190-2196）：读预反量化的 FP16 静态 cache。
- 否则 `mNeedRuntimeDequant`（行 2197-2207）：每个 step 先跑
  `DequantizeInt4Weight`（行 481）把整层权重 int4→FP16 写进 DYNAMIC buffer，再
  CUTLASS。Qwen3-4B 的 `runtime_dequant=1 static_dequant=0`（PLAN 行 85、日志
  2026-07-01-18）正来自这里——某些 batch-4 Linear 形状没满足静态 cache 门控
  （`mlpLikeLargeLinear`/`attentionProjectionLikeLinear`，行 1952-1957）。

**核心代价对比**（per layer per op，权重矩阵 W∈R^{oc×ic}）：

| 路径 | 权重读取量 | 权重格式 | 备注 |
|---|---|---|---|
| V14_MB (batch≤3) | `oc·ic/2` bytes (int4) | packed int4 | batch 行共享权重 |
| cuBLAS/CUTLASS (batch≥4) | `oc·ic·2` bytes (FP16) | 已反量化 FP16 | 每 step 重读 |

**FP16 路径的权重读取量是 int4 路径的 4 倍**（2B/元素 vs 0.5B/元素）。在 Jetson
这种**带宽受限、算力相对过剩**的统一内存设备上，batch=4 时算力还没用满，瓶颈
直接转到权重带宽——这就是悬崖。

---

## 2. Jetson AGX Xavier 带宽可行性分析

设备：Jetson AGX Xavier，sm_72（Volta Tensor Core），32GB LPDDR4x 统一内存，
标称内存带宽 ~136.5 GB/s，实测 STREAM Triad 通常 ~80-95 GB/s。CUDA arch=`72`
（`mnn-opt-ops` SKILL 交叉编译 `CUDA_ARCHS=72`）。

### 2.1 单算子权重读取时间下界（bandwidth roofline）

权重读取时间 ≈ 权重字节数 / 有效带宽。对 decode-repair 的热 MLP/Linear，假设
有效带宽 80 GB/s（保守，统一内存 + sm_72 L2 命中率一般）：

**Llama3.2-3B** (hidden=3072, inter=8192, kv=1024, 28 layers)：

| op | oc×ic | int4 权重 bytes | FP16 权重 bytes | int4 读 @80GB/s | FP16 读 @80GB/s |
|---|---|---|---|---|---|
| gate/up (3072→8192) | 8192×3072 | 12.58 MB | 50.33 MB | 0.157 ms | 0.629 ms |
| down (8192→3072) | 3072×8192 | 12.58 MB | 50.33 MB | 0.157 ms | 0.629 ms |
| q_proj (3072→3072) | 3072×3072 | 4.72 MB | 18.87 MB | 0.059 ms | 0.236 ms |
| kv_proj (3072→1024) | 1024×3072 | 1.57 MB | 6.29 MB | 0.020 ms | 0.079 ms |
| o_proj (3072→3072) | 3072×3072 | 4.72 MB | 18.87 MB | 0.059 ms | 0.236 ms |

**Qwen3-4B** (hidden=2560, inter=9728, kv=1024, 36 layers)：

| op | oc×ic | int4 bytes | FP16 bytes | int4 读 @80GB/s | FP16 读 @80GB/s |
|---|---|---|---|---|---|
| gate/up (2560→9728) | 9728×2560 | 12.44 MB | 49.74 MB | 0.156 ms | 0.622 ms |
| down (9728→2560) | 2560×9728 | 12.44 MB | 49.74 MB | 0.156 ms | 0.622 ms |
| q_proj (2560→4096) | 4096×2560 | 5.24 MB | 20.97 MB | 0.065 ms | 0.262 ms |
| kv_proj (2560→1024) | 1024×2560 | 1.31 MB | 5.24 MB | 0.016 ms | 0.066 ms |
| o_proj (4096→2560) | 2560×4096 | 5.24 MB | 20.97 MB | 0.065 ms | 0.262 ms |

**MiniCPM5-1B** (hidden≈1024, inter≈4096 量级，24-28 layers)：权重约为 3B 的
1/6，单 op int4 读取 ~0.025-0.06ms 量级。

### 2.2 每层 MLP/Linear 的 per-TPOT 贡献与目标

从日志 2026-07-01-18 的 single-TPOT 归因（normalize 到单 token）：

| model | x3-x1 delta | Conv/TPOT | MLP/TPOT | attn-proj/TPOT |
|---|---|---|---|---|
| MiniCPM5-1B | +10.79ms | +11.51 | +5.04 | +6.53 |
| Qwen3-4B | +55.12ms | +51.67 | +28.93 | +22.87 |
| Llama3.2-3B | +31.26ms | +28.20 | +17.72 | +10.59 |

注意这是**单 TPOT**（每生成 1 token 的增量），不是一次请求总和。每 token 要跑
全部 layer 的所有 Linear 一次。`MLP/TPOT` 增量 = 每层 (gate+up+down) 增量 ×
num_layers / 1 token。

对 Llama3.2-3B（28 layers），`MLP/TPOT +17.72ms` ⇒ 每层 MLP 三连增量约
`17.72/28 ≈ 0.633ms/layer`。三连里 gate+up+down 各一个，batch=4 时三个 op 都
走 dense FP16 路径。

**理论可压缩空间**：把 batch≥4 的 dense 路径从 FP16 权重 (4×) 换回 int4 权重
(1×)，每个 op 的权重读取时间理论上能从 FP16 下界回到 int4 下界，即**节省约
3/4 的权重带宽**。

Llama3.2-3B 单层 MLP 三连：FP16 下界 `3 × 0.629 = 1.887ms`，int4 下界
`3 × 0.157 = 0.471ms`，**理论节省 ~1.42ms/layer**。但实际 dense 路径当前远高于
FP16 下界（因为还有 runtime dequant 开销、CUTLASS 启动、cuBLAS 数学模式等），
所以可压缩空间比理论下界更大。

### 2.3 能否让 x=3/5/7 接近 x=0/1？结论

`x=0/1` 的 tiny-GEMV 路径每 op 权重读 int4 = `0.157ms`（Llama3.2-3B gate）。
**若新 rows4/6/8 kernel 能把权重读回 int4 量级、并像 V14_MB 一样让 batch 行共享
权重，则 batch=4 的每 op 成本能回到 ~0.15-0.25ms 量级，与 batch=2 的 V14_MB
基本同档。**

带宽可行性结论：
- **算力不是瓶颈**。sm_72 上 batch=4、ic=3072 的 int4 dot-product 远低于 Tensor
  Core 峰值；问题是 cuBLAS/CUTLASS 用 FP16 权重把带宽放大 4×。
- **带宽是唯一瓶颈**，且 int4 路径的带宽下界就是 `x=0/1` 的下界。所以**理论上
  x=3/5/7 可以逼近 x=0/1**，差距应收敛到「batch 行带来的额外输入读取 + 归约
  开销 + kernel 启动」这点常数，而不是权重带宽量级差。
- **真实风险**：V14_MB 的 trick 依赖「一个 block 处理一组 OC、batch 行共享权重」。
  当 batch 从 2/3 增到 4/6/8 时，每个 OC 组要累加 `MAX_BATCH×OC_PER_BLK` 个
  accumulator，寄存器/shared-memory 压力上升；若 `OC_PER_BLK` 和 `MAX_BATCH`
  组合不当，occupancy 下降会吃掉带宽收益。这正是 PLAN Non-Goals 里「把 V14
  tiny GEMV 简单放宽到 rows4/6/8」被否定、`mnn-opt-ops` SKILL 行 250-252 也
  验证过会变慢的原因。**所以新 kernel 不能是 V14_MB 的简单调参，必须是重新设计
  的 batch 平铺结构**（见 §3）。

---

## 3. 新算子设计：`GEMV_FpAInt4B_V15_SmallM`（rows4/6/8 INT4-native）

设计目标：让 batch∈{4,6,8} 的 weight-only Linear 直接消费 **packed int4 权重**，
复用 `mGemvParams`（已预算的 `{scale, adj_offset}`）做因式分解反量化，**不走
FP16 权重、不走 cuBLAS/CUTLASS**。这是 PLAN 里 Phase 1 要求的 "rows4/6/8
INT4-native small-M weight-only kernel family"。

### 3.1 与 V14_MB 的关键差异（为什么不是简单调 MAX_BATCH）

V14_MB（行 1442）的设计：`grid(oc/OC_PER_BLK)`，每个 block 128 线程，把
`MAX_BATCH × OC_PER_BLK` 个 float accumulator 全放寄存器，热循环里所有 batch
行 × 所有 OC 同时累加。当 `MAX_BATCH=6, OC_PER_BLK=4` 时 = 24 个 float acc + 24
组权重寄存器，`__launch_bounds__(128,8)` 下 occupancy 已经很紧。把 MAX_BATCH
推到 8 会让寄存器溢出，直接退回 local memory → 变慢（SKILL 行 251 实测证实）。

V15 的重新设计：**把 batch 维从寄存器移到 shared memory + 分阶段累加**，让单个
block 只持有 `OC_PER_BLK` 个 acc，但分 `ceil(MAX_BATCH / B_TILE)` 个 micro-pass
串行处理 batch 行，权重只读 1 次留在 shared memory 复用给所有 micro-pass。

```
V15 tile:  Block = 128 threads
           每个 block 负责 OC_PER_BLK 个输出通道 (例如 4 或 8)
           权重 IC 维按 warp 切片，每个 warp 处理 IC 的一段
           batch 维分 B_TILE 个 micro-pass（B_TILE=2，即 batch=8 → 4 个 micro-pass，每个处理 2 行）

热循环 (per IC slice):
  1. 读 OC_PER_BLK 行 × IC_slice 的 packed int4 权重 → shared memory (一次)
  2. 读对应 gemv_params {scale, adj_offset} (一次)
  3. for b_tile in 0..B_TILE-1:        // micro-pass
       读 B_TILE 个 batch 行的 input (IC_slice)
       因式分解累加: acc[b_tile*OC_PER_BLK + oi] += s[oi]*raw_dot + adj[oi]*in_sum
  4. 权重 shared memory 在 micro-pass 间复用，不重读
```

这样：
- **权重读取量** = int4 = V14 量级（4× 优于 FP16 dense）。
- **寄存器压力** = `B_TILE × OC_PER_BLK` 个 acc（B_TILE=2, OC_PER_BLK=8 → 16
  float），远小于 V14_MB 的 `MAX_BATCH×OC_PER_BLK`。
- **batch 行共享权重**：micro-pass 复用同一份 shared-memory 权重，等效于 V14_MB
  的权重复用，但用 shared memory 而非寄存器承担 batch 维。
- **归约**：每个 (b, oi) 的 acc 跨 warp 归约（shfl）+ 跨 micro-pass 累加，最后
  写 `output[b*ocp + oc_base + oi]`。

### 3.2 OC_PER_BLK 与 tile 选择

按 op 形状分档（在 host 侧 dispatch 选模板实例）：

| 形状 | oc 量级 | OC_PER_BLK | 理由 |
|---|---|---|---|
| gate/up (→inter) | 8192/9728 | 8 | oc 大，多 OC/block 降 grid 数、提权重复用 |
| down (→hidden) | 3072/2560 | 8 | oc 中等 |
| q_proj (→hidden/q) | 3072/4096 | 8 | |
| kv_proj (→kv) | 1024 | 4 | oc 小，OC_PER_BLK=4 避免 tail 浪费 |
| o_proj (→hidden) | 3072/2560 | 8 | |

batch 维 micro-pass：`B_TILE=2` 固定，`num_micro_pass = ceil(batch/2)`，覆盖
batch=4(2 pass)/6(3 pass)/8(4 pass)。batch=1/2/3 不走 V15（仍走 V14/V14_MB）。

### 3.3 精度等价性

V15 与 V14/V14_MB 数学完全等价：都用 `w_deq = s*nibble + adj` 因式分解，acc 最终
是 `Σ w_deq·in = s·Σ(nibble·in) + adj·Σ in`。浮点累加顺序不同会引入极小
ULP 差异，与现有 V14/V14_MB 跨 batch 实例间的差异同量级。accuracy 用 direct-op
`runRows45CublasAccuracyCase` 同款对照（参考输出 vs V15 输出，`max_abs/max_rel/bad`
判据），与 PLAN Phase 1 "accuracy passes against the current backend route" 一致。

### 3.4 激活与 bias

V15 只做纯 Linear（`mActivationType==0`，与 rows45 cuBLAS 的 static-dequant 语义
一致）。bias 在最后归约后加：`result = clamp(acc + bias, minV, maxV)`。MLP 的
SwiGLU 在 down 之前由独立 op 完成，不在 V15 内融合（PLAN Non-Goals 已排除
naive fused SwiGLU+down 作为生产路线；V15 只优化单个 Linear）。

---

## 4. 集成方案（不破坏现有函数）

严格遵循 PLAN Phase 2 + "不要破坏原来的函数"：**所有新代码都是新增分支，旧路径
原样保留为 fallback**。

### 4.1 新增 kernel（ConvFpAIntBExecution.cu，V14_MB 之后插入）

```cpp
// V15: rows4/6/8 INT4-native weight-only GEMV.
// 与 V14_MB 相比：batch 维用 shared-memory + micro-pass 替代寄存器堆，
// 让 OC_PER_BLK 能开到 8 而不溢出寄存器；权重 int4 只读一次，micro-pass 间复用。
template<typename T, int OC_PER_BLK>
__global__ void GEMV_FpAInt4B_V15_SmallM(
    const T* __restrict__ input,
    const uint8_t* __restrict__ kernel,     // packed int4 [oc, ic_p/2]
    const float2* __restrict__ gemv_params, // [oc * num_qg] {scale, adj_offset}
    const T* __restrict__ bias,
    T* __restrict__ output,
    const float maxV, const float minV,
    const int batch, const int ic, const int ic_p,
    const int oc, const int oc_p, const int num_qg) {
    // ... 见 §3.1 结构 ...
}
```

模板实例化 `<half, 4>` 和 `<half, 8>`（fp16 主路径），mix 路径 `<float, ...>`。
**不替换 V14/V14_MB 的任何实例化**。

### 4.2 新增 dispatch 分支（onExecute，rows48/rows45 之前）

在 `onExecute` 行 2456 的 `else`（batch > int4GemvBatchLimit）里，**在 rows48 /
rows45 之前**插入一个新分支，由新 env 门控，命中则走 V15，否则**原样 fallthrough
到现有 rows48/rows45/generic**：

```cpp
} else {
    // === 新增：V15 rows4/6/8 INT4-native small-M 路由（env 门控，默认 off）===
    const bool useV15SmallM = picDecodeRepairSparse && mFp16Infer &&
        mResource->mGemvParams != nullptr &&
        picV15SmallMPolicy() > 0 &&            // 新 env MNN_CUDA_PIC_INT4_SMALLM_V15
        picV15SmallMMatches(batch, ic, oc);    // batch∈{4,6,8}, ic/oc 形状门控
    if (useV15SmallM) {
        // 选 OC_PER_BLK by shape，launch GEMV_FpAInt4B_V15_SmallM
        ... // profile tag: conv_fpa_intb_1x1_smallm_v15
    } else {
        // === 以下为现有代码，原样不动 ===
        const int rows45CublasPolicy = picRows45CublasPolicy();
        const bool useRows45Cublas = ...;
        ... rows48 / rows45 / generic ...
    }
}
```

关键点：
- **V15 不依赖 `staticDequant`**。它直接读 `mResource->mFilter`（packed int4）+
  `mResource->mGemvParams`，所以**同时解决 Qwen3-4B 的 `runtime_dequant=1`
  问题**——V15 路径下根本不分配/不读 FP16 dequant buffer，runtime dequant 自然
  不触发。这正好命中 PLAN 行 85/141 的 Qwen3-4B 痛点。
- 旧路径（rows48/rows45/generic + runtime/static dequant）**一行未动**，作为
  fallback 保留。V15 失败或形状不匹配时无缝回退。
- profile logging 新增 `conv_fpa_intb_1x1_smallm_v15` tag，与现有 tag 并列，便于
  PLAN 行 155-160 的路由证据区分。

### 4.3 不需要改的东西（满足硬约束）

- ❌ 不改 `transformers/pic_llm/export/*`，不改图导出。
- ❌ 不新增 op type / schema / 注册表（V15 是 ConvFpAIntBExecution 内部 kernel，
  复用现有 `Convolution2D` weight-only op，shape 匹配触发，符合 PLAN 行 91
  "backend shape matching, not graph rewriting"）。
- ❌ 不改 `onResize` 的 dequant 决策（V15 不走 dequant buffer，但 dequant 决策
  代码本身不动；V15 只是 `onExecute` 里多一个 if）。
- ✅ 现有 V14/V14_MB/rows45/rows48/generic 全部原样保留。

### 4.4 direct-op 验证（PLAN Phase 1）

在 `test/bench_ops/cuda/CudaWeightOnlyConvPerf.cpp` 新增（不替换现有 case）：

- **accuracy**：新 `CudaSmallMV15Accuracy`，对照 rows45 cuBLAS 输出，覆盖三模型
  rows4/6/8 × {gate/up/down/q/kv/o} 形状，`memory=2`（PLAN 行 142 硬要求）。
- **perf**：在现有 `CudaWeightOnlyConvPerf` 的 case 列表里，rows≥4 时**额外**跑
  V15 路径（通过 env `MNN_CUDA_PIC_INT4_SMALLM_V15=1` 切换），与 cuBLAS baseline
  并列打印，不破坏现有 baseline case。

验收（PLAN 行 137-143）：accuracy pass；rows4/6/8 V15 快于 cuBLAS baseline；
rows1/2 不受影响（V15 不命中）；Qwen3-4B batch-4 不再 runtime dequant。

---

## 5. 端到端 TPOT 预期（Jetson）

按 §2 带宽下界与 §3 设计，V15 把 batch≥4 权重读拉回 int4 量级：

**Llama3.2-3B / MLP 三连 / batch=4 (x=3)**：
- 当前 dense FP16 路径：每层 MLP 增量 ≈ `0.633ms`（来自 `17.72ms/28`）。
- V15 int4 下界：每层 MLP 三连 ≈ `3×0.157 = 0.471ms`。
- 但 `x=0/1` 的 V14 路径每层 MLP 也≈ `0.471ms` 量级（同样 int4 权重）。
- **预期 V15 后 x=3 的 MLP/TPOT 回到接近 x=1**，剩余 gap 主要是 batch=4 vs 2 的
  input 读取 + 归约 + kernel 启动常数（每层 < 0.05ms 量级）。

**Qwen3-4B**：额外收益——V15 消除 batch-4 的 `runtime_dequant`，省掉整层
int4→FP16 的 `DequantizeInt4Weight` kernel（约 0.1-0.2ms/层/op）+ 消除 4× 权重
读放大。`x=3` 的 `+55ms` delta 里大头（`+51.67 Conv/TPOT`）有望显著收窄。

**结论**：在 Jetson AGX Xavier 上，**V15 路线在带宽下界层面足以让 x=3/5/7 的
MLP/Linear 时间逼近 x=0/1**。决定成败的是 §3.2 的 tile/occupancy 调参——这必须
通过 direct-op 实测确认（PLAN Phase 1），不能只靠 roofline 断言。若实测
occupancy 不足导致 V15 仍慢于 cuBLAS，则按 PLAN Production Decision Rule 退回
fallback，不强行上默认。

---

## 6. 与历史负向结果的关系（避免重蹈覆辙）

PLAN Non-Goals 和 `mnn-opt-ops` SKILL 行 248-259 列出的已被否定的分支：

| 历史尝试 | 为什么被否定 | V15 是否回避 |
|---|---|---|
| 放宽 `int4GemvBatchLimit` 3→5 | 让 rows4/5 走 V14_MB 变慢（寄存器溢出） | ✅ V15 是全新 tile，不是 V14_MB 调参 |
| V14_MB 枚举 `OC_PER_BLK=2/3/4/8/16` | 寄存器/occupancy 不对 | ✅ V15 用 shared-mem micro-pass 解耦 batch 维 |
| naive fused SwiGLU+down | headroom 不足 | ✅ V15 只优化单 Linear，不融合 |
| naive WMMA down | 慢于 cuBLAS | ✅ V15 不用 WMMA，纯 int4 dot |
| fresh gate/up packed 导出 | `/v1/prefill/text` 失败 | ✅ V15 不改图、不改导出 |
| `PicLinearNhwcWeightOnly` 图改写 | 端到端慢 +2ms | ✅ V15 是 backend 内部路由 |

V15 的定位正是 PLAN 行 90 唯一还没被否定的方向：**"Add or tune a rows4/6/8
small-M weight-only kernel family only if direct-op tests show clear speedup"**，
且必须是 INT4-native（SKILL 行 224 "新的 rows=4/5 kernel 必须是 INT4-native 小
批量 weight-only kernel"）。

---

## 7. 落地顺序

1. **Phase 1a**：在 `CudaWeightOnlyConvPerf.cpp` 加 V15 accuracy + perf case
   （先空跑确认 harness 能创建 int4 weight-only op、能切 env）。
2. **Phase 1b**：在 `ConvFpAIntBExecution.cu` 实现 `GEMV_FpAInt4B_V15_SmallM`
   kernel + env 门控 dispatch（默认 off），交叉编译推 Jetson。
3. **Phase 1c**：Jetson direct-op 跑 rows4/6/8 × 三模型形状，确认 accuracy pass
   + 速度击败 cuBLAS baseline + rows1/2 不回归。调 OC_PER_BLK / B_TILE 直到达标。
4. **Phase 2**：direct-op 通过后才把 V15 env 默认开（或按 PLAN Decision Rule
   保留 opt-in），保留 fallback。
5. **Phase 3**：Jetson 端点 decode TPOT `x=0/1/3/5/7`，对照 baseline，确认
   `x=3/5/7` 逼近 `x=0/1` 且不回归。

任一阶段不达标，V15 保持 env-off，生产路径不变（满足 PLAN 行 197 "remains an
experiment"）。

---

## 8. Jetson direct-op 实验结论（2026-07-01，已实测）

V15 已实现、交叉编译、推到 Jetson AGX Xavier (sm_72) 跑 direct-op。**结论：
V15 int4-native 路线在 sm_72 上无法击败 static-dequant cuBLAS，保持 env-off。**

### 8.1 精度：bit-exact 通过

`bench_ops/cuda/accuracy/SmallMV15`，rows4/6/8 × Llama3.2-3B (gate/up/down/q/kv)
+ Qwen3-4B (gate/down/q)，全部 `max_abs=0 max_rel=0 bad=0`。V15 因式分解反量化
与现有 generic 路径位级一致。

### 8.2 性能：三种 tile 均慢于 cuBLAS

`hidden_to_inter` (2048→8192)，ms（越小越好）：

| rows | V15 v1 (smem 4线程load) | V15 v2 (batch-pair reg load) | V15 v3 (single-load 内层batch) | cuBLAS baseline | int4 roofline 下界 |
|---|---|---|---|---|---|
| 4 | 2.070 | 0.740 | 1.394 | 0.314 | 0.105 |
| 6 | 2.075 | 1.092 | 2.180 | 0.316 | 0.105 |
| 8 | 2.071 | 1.470 | 3.042 | 0.320 | 0.105 |

### 8.3 为什么 int4 路线在 sm_72 失败（带宽分析的反例）

- **cuBLAS baseline 已达峰值带宽**：0.31ms 读 33.6MB FP16 static-dequant 权重
  = ~108 GB/s ≈ Jetson LPDDR4x 峰值。它是 production rows45 cuBLAS 路径，**高效**。
- **V15 是 instruction-bound 不是 bandwidth-bound**：V15 随 batch 线性增长
  （0.74→1.09→1.47），且 rows=4 时仍 7× 高于 int4 下界（0.105ms）。sm_72（Volta，
  无高效 int4 tensor-core 路径）上 int4 nibble dot（每 nibble shift/mask/int2float/
  FMA）的 ALU 开销吃掉了 int4 比 FP16 省 4× 权重字节的红利。
- **非 spill 问题**：即使无 spill 的小 oc 形状（hidden_to_kv oc=512，OC_PER_BLK=4）
  V15 仍 2.6× 慢于 cuBLAS（0.076 vs 0.029ms）。

### 8.4 结论

- V15 保持 **env-off**，production 路径不变（rows45 cuBLAS static-dequant 仍是
  rows4/6/8 路由）。无图/导出/默认改动。
- 这与 `mnn-opt-ops` SKILL 行 248-259 已记录的负向结果一致：V14_MB retune、
  naive int4 small-M、WMMA down 都失败。V15 是第 4 个 int4-native 变体，同样失败。
- **Jetson sm_72 上 `x=1→x=3` 悬崖无法用 int4-native Linear kernel 闭合**，因为
  要替换的 dense 路径（static-dequant cuBLAS）已达峰值带宽，而 int4 GEMV 在 sm_72
  是 instruction-bound。闭合悬崖需要别的杠杆（减少 x=3 命中的 dense Linear 数量，
  或 packed-gate/up 图改写——PLAN 均列为 non-goal）。
- V15 代码 + accuracy harness 保留为 opt-in 实验，供未来有高效 int4 tensor-core
  路径的设备（sm_90+ Hopper int4 / Adreno）复用。

原始 direct-op 日志：`.codex/skills/mnn-pic-optimize/log/jetson/2026-07-01-19/raw/jetson_directop_v15.txt`。

