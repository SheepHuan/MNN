# Jetson Small-M Weight-Only Linear (V15) Design

> **最终结论（2026-07-01，Update 4）**：Jetson sm_72 上 decode-repair `x=1→x=3`
> 悬崖**不可用任何 int4-based Linear kernel 闭合**。已实测 5 个 int4 衍生变体
> 全部失败（V14_MB retune、V15 v1/v2/v3、V16 dp4a），两个独立瓶颈（nibble ALU
> instruction-bound、dp4a register-spill-bound）封死该路线。被替换的 dense 路径
> （static-dequant cuBLAS）已达峰值 DRAM 带宽。生产路径保持不变。详见末尾
> 「Update 4 — 不可优化定论」与 `OPTIMIZATION_SUMMARY.md`。

## Summary

- Drafted a rows4/6/8 INT4-native weight-only Linear kernel design (`V15_SmallM`) to
  close the decode-repair `x=1 -> x=3` cliff on Jetson AGX Xavier, without changing
  graph export or breaking existing CUDA routes.
- Root cause confirmed from source: under `pic_decode_repair_sparse`, `int4GemvBatchLimit`
  drops to 3, so `x=3 (batch=4)` leaves the fast V14/V14_MB tiny-GEMV path and enters the
  cuBLAS / CUTLASS dense path that consumes dequantized FP16 weights — a 4x weight-bandwidth
  blowup on a bandwidth-bound unified-memory device.
- Bandwidth roofline shows the int4 weight-read floor for batch=4 equals the batch=1/2
  floor, so `x=3/5/7` can in principle approach `x=0/1` if a new kernel keeps weights as
  packed int4 and reuses them across batch rows (like V14_MB) but without V14_MB's
  register-pressure collapse at `MAX_BATCH>=4`.
- V15 reorganizes the batch dimension out of registers into shared memory + micro-passes,
  decoupling `OC_PER_BLK` from `MAX_BATCH` so `OC_PER_BLK=8` is reachable. It also bypasses
  the FP16 dequant buffer entirely, fixing the Qwen3-4B `runtime_dequant=1` case.
- All existing routes (V14, V14_MB, rows45 cuBLAS, rows48 cuBLASLt, generic CUTLASS) are
  left untouched as fallback; V15 is env-gated, opt-in, default off.

## Artifacts

- Design / analysis doc: `CUDA_SMALL_M_WEIGHT_ONLY_LINEAR_OPT_DESIGN.md` (repo root).
- Source analyzed: `source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu`
  (routing map: `onResize` 2108-2234, `onExecute` 2236-2713; V14 1334, V14_MB 1442;
  rows45 cuBLAS 2457-2526; rows48 cuBLASLt 2462-2494; generic runtime-dequant 2527-2614).
- Bench harness to extend: `test/bench_ops/cuda/CudaWeightOnlyConvPerf.cpp`
  (`CudaWeightOnlyConvPerf` 2396, `CudaRows45CublasAccuracy` 2552).

## Key Bandwidth Table (Llama3.2-3B, @80 GB/s effective)

| op | int4 bytes | FP16 bytes | int4 read | FP16 read |
|---|---|---|---|---|
| gate/up 3072->8192 | 12.58 MB | 50.33 MB | 0.157 ms | 0.629 ms |
| down 8192->3072 | 12.58 MB | 50.33 MB | 0.157 ms | 0.629 ms |
| q/o_proj 3072->3072 | 4.72 MB | 18.87 MB | 0.059 ms | 0.236 ms |
| kv_proj 3072->1024 | 1.57 MB | 6.29 MB | 0.020 ms | 0.079 ms |

FP16 dense path reads 4x the weight bytes of the int4 GEMV path. On Jetson (bandwidth
bound, not compute bound) this is the cliff, not the extra batch rows.

## Decision

- Proceed to Phase 1 direct-op implementation: add `GEMV_FpAInt4B_V15_SmallM` kernel +
  env-gated dispatch + accuracy/perf cases. Default off until direct-op proves
  rows4/6/8 beats cuBLAS baseline and rows1/2 do not regress.
- V15 must NOT be a V14_MB `MAX_BATCH` retune (already negative, log jetson/2026-07-01-18
  and mnn-opt-ops SKILL rows 250-252). It must be a fresh tile using shared-memory
  micro-passes for the batch dimension.
- If direct-op does not beat cuBLAS, V15 stays env-off experiment per PLAN Production
  Decision Rule; production path unchanged.

## Next

- Phase 1a/1b: implement kernel + bench cases, cross-compile to Jetson, run direct-op
  rows4/6/8 x {gate/up/down/q/kv/o} for the three target models with `memory=2`.

## Update (same hour) — Phase 1a/1b implemented and cross-compiled

- Implemented `GEMV_FpAInt4B_V15_SmallM<T, OC_PER_BLK>` in
  `ConvFpAIntBExecution.cu` (after V14_MB). INT4-native, shared-memory weight tile
  + `B_TILE=2` micro-passes over batch; reuses `mGemvParams` factored dequant;
  does NOT touch the FP16 dequant buffer.
- Added env-gated dispatch (`MNN_CUDA_PIC_INT4_SMALLM_V15`, default off, values
  `0/off`, `down`, `mlp`, `all`) in `onExecute` `else` (batch > int4GemvBatchLimit),
  BEFORE rows48/rows45. Miss falls through to existing routes unchanged. All
  existing V14/V14_MB/rows45/rows48/generic routes untouched (fallback preserved).
- Dispatch picks `OC_PER_BLK = (oc>=2048)?8:4`; templates instantiated for
  `<half,4/8>` and `<float,4/8>`. Requires `mActivationType==0` and `mGemvParams`.
- New profile tag `conv_fpa_intb_1x1_smallm_v15`.
- Added bench cases: `bench_ops/cuda/accuracy/SmallMV15` (rows4/6/8 x
  Llama3.2-3B + Qwen3-4B shapes, ref=V15-off generic vs candidate=V15=all) and
  `bench_ops/cuda/perf/SmallMV15` (rows4/6/8 x MLP/attn shapes). Existing
  `WeightOnlyConv` / `Rows45Cublas` cases unchanged.
- Cross-compiled to Jetson sm_72 via `mnn-opt-ops` flow (`CUDA_ARCHS=72`):
  build OK, all 4 V15 template symbols present in `libMNN_Cuda_Main.so`, both
  SmallMV15 suites registered in `run_test.out`. Artifact root:
  `.cache/output/mnn/artifacts/jetson_cross_cuda_v15`.
- No graph export change, no op/schema/registry change, no `onResize` change.
  Production path unchanged when env unset.

## Next (pending)

- Push artifact to Jetson, run direct-op accuracy + perf rows4/6/8 with
  `memory=2`. Tune `OC_PER_BLK`/`B_TILE` if occupancy-bound. Gate default-on on
  PLAN Production Decision Rule. Requires Jetson SSH access (currently soft-blocked).

## Update 2 — direct-op run on Jetson: V15 is accuracy-clean but NOT faster than cuBLAS

After SSH access was authorized, pushed the V15 artifact and ran direct-op on
Jetson AGX Xavier (sm_72) with `memory=2`. Three V15 tile variants tested.

### Accuracy (all variants): PASS, bit-exact
`bench_ops/cuda/accuracy/SmallMV15` rows4/6/8 x Llama3.2-3B (gate/up/down/q/kv) +
Qwen3-4B (gate/down/q): `max_abs=0 max_rel=0 bad=0` for every case. V15 factored
dequant is bit-identical to the existing generic route. Math is correct.

### Perf (hidden_to_inter 2048->8192, rows4/6/8; cuBLAS = static-dequant FP16)

| rows | V15 v1 (smem 4-thr load) | V15 v2 (batch-pair register load) | V15 v3 (single-load inner batch) | cuBLAS baseline | int4 roofline floor |
|---|---|---|---|---|---|
| 4 | 2.070 | 0.740 | 1.394 | 0.314 | 0.105 |
| 6 | 2.075 | 1.092 | 2.180 | 0.316 | 0.105 |
| 8 | 2.071 | 1.470 | 3.042 | 0.320 | 0.105 |

(ms; lower is better)

### Verdict: V15 int4-native route CANNOT beat static-dequant cuBLAS on sm_72

- cuBLAS baseline is flat ~0.31ms across rows = bandwidth-bound on the FP16
  static-dequant weight (33.6MB @ ~108 GB/s, already at peak DRAM). It is the
  production rows45 cuBLAS path and it is efficient.
- V15 grows linearly with batch and is 7-14x above the int4 roofline floor even
  at rows=4. On sm_72 (Volta, no efficient int4 tensor-core path used here), the
  int4 nibble dot-product (shift/mask/int2float/FMA per nibble) is
  **instruction-bound**, not bandwidth-bound. The 4x weight-bytes saving of int4
  vs FP16 is eaten by per-nibble ALU overhead.
- Even the no-spill OC_PER_BLK=4 small-oc case (hidden_to_kv oc=512) is 2.6x
  slower than cuBLAS (0.076 vs 0.029ms), so this is not a spill problem — it is
  fundamental int4-GEMV instruction cost on this device.
- This matches the prior negative results in `mnn-opt-ops` SKILL (rows 248-259):
  V14_MB retunes, naive int4 small-M, and WMMA down all failed. V15 is a fourth
  int4-native variant that also fails to beat the static-dequant cuBLAS path.

### Decision

- V15 stays **env-off** (`MNN_CUDA_PIC_INT4_SMALLM_V15` unset). Production path
  unchanged: rows45 cuBLAS (static dequant) remains the rows4/6/8 route.
- No default-on, no graph/export change. V15 code retained as an opt-in
  experiment + accuracy harness for future devices that DO have efficient int4
  tensor-core paths (e.g. sm_90+ / Hopper int4, or Adreno).
- The `x=1->x=3` cliff on Jetson is NOT closeable by an int4-native Linear
  kernel, because the dense route it would replace (static-dequant cuBLAS) is
  already at peak bandwidth and int4 GEMV is instruction-bound on sm_72.
  Closing the cliff on Jetson would require a different lever (e.g. reducing the
  number of dense Linears hit at x=3, or a packed-gate/up graph rewrite — which
  PLAN excludes as non-goal). This is an experiment conclusion, not a production
  change.

### Direct-op raw logs

- accuracy: `.cache/bench_ops/v15_smoke/smallm_v15_accuracy.log`
- perf: `.cache/bench_ops/v15_smoke/` (SmallMV15 + WeightOnlyConv baseline)

## Update 3 — INT8 __dp4a route (V16): accuracy bit-exact, but register-spill-bound, ~100-1000x slower than cuBLAS

After re-calibration showed int4 factored-GEMV (V14_MB batch=4 = 0.40ms) loses to
cuBLAS (0.31ms) due to nibble-ALU instruction cost, tested a 5th variant: INT8
`__dp4a` with factored asymmetric math (no weight re-quant loss; nibble kept exact
0..15, A quantized to int8 per row at runtime). Math derived and verified:

```
acc = s_w*s_a * dp4a(nibble,a_q) + s_w*o_a*sum_nibble + adj_w*s_a*sum_aq + adj_w*o_a*K
```

### Accuracy: PASS, bit-exact
`bench_ops/cuda/accuracy/SmallMV16` rows4/6/8 x Llama3.2-3B (gate/down/q) +
Qwen3-4B (gate): `max_abs=0 max_rel=0 bad=0` for every case. The factored-dp4a
math is correct and matches the existing generic route bit-for-bit, including
runtime asymmetric A quantization. This is a real correctness result.

### Perf: 60-1000x SLOWER than cuBLAS (register-spill-bound)

`hidden_to_inter` (2048->8192), ms:

| rows | V16 dp4a (v1, per-group grid) | V16 dp4a (v2, in-block group loop) | cuBLAS |
|---|---|---|---|
| 4 | 21.48 | 45.32 | 0.314 |
| 6 | 27.90 | 58.71 | 0.316 |
| 8 | 34.29 | 72.52 | 0.320 |

Profile: single V16 kernel launch = 44ms for hidden_to_inter rows=4 (vs 0.31ms
cuBLAS). The kernel is spilling to local memory: the per-(oc,group) factored
combine requires keeping `totals[8*OC_PER_BLK]` + `sa/oa/saq[8]` + per-group
`accs[8*OC_PER_BLK]` + `w0/w1[OC_PER_BLK]` live across the 32-group in-block
loop. At OC_PER_BLK=8 that is ~168 register-resident variables -> massive spill
to DRAM. Even OC_PER_BLK=4 (hidden_to_kv oc=512) is 1.84ms vs cuBLAS 0.029ms
(60x slower) — still spill/scheduling bound, not compute bound.

### Verdict: dp4a int4 route also fails on sm_72

- The math is correct (bit-exact), so this is NOT a correctness issue.
- The failure is **register pressure**: the factored-dp4a combine needs too much
  per-block state to sustain across 32 quant groups in-block, and the per-group
  grid variant explodes block count (32768 blocks -> launch-bound). Both
  formulations lose by 2-3 orders of magnitude.
- V16 stays **env-off** (`MNN_CUDA_PIC_INT4_SMALLM_V15`/`_DP4A` unset).
- This is the **5th** int4-derivative variant that fails on Jetson sm_72
  (V14_MB retune, V15 v1/v2/v3, V16 dp4a). Combined with the measured V14_MB
  batch=4 = 0.40ms > cuBLAS 0.31ms, the conclusion is now firm:

### Firm conclusion: Jetson sm_72 x=1->x=3 cliff is NOT closeable by any int4-based Linear kernel

Two independent bottlenecks block every int4-derivative route on sm_72:
1. **nibble ALU** (V14/V15): per-nibble shift/mask/int2float/FMA is instruction-
   bound; V14_MB batch=4 (0.40ms) already loses to cuBLAS (0.31ms).
2. **dp4a register pressure** (V16): the factored per-group combine spills.

The target dense route (static-dequant cuBLAS) is already at peak DRAM bandwidth
(108 GB/s reading FP16 weights) and is efficient. There is no int4/dp4a lever on
sm_72 that beats it for batch=4/6/8 small-M. The cliff would need a non-int4
lever (e.g. reducing the count of dense Linears hit at x=3, or a graph rewrite —
both PLAN non-goals) or a different device with efficient int4/int8 tensor cores.

### Artifacts retained (env-off experiments)

- V16 kernel `GEMV_FpAInt4B_V16_Dp4a` + `QuantARowInt8` + `PrecomputeSumNibble`
  in ConvFpAIntBExecution.cu; `mSumNibble` Resource field. All env-gated, default off.
- `bench_ops/cuda/accuracy/SmallMV16` + `bench_ops/cuda/perf/SmallMV16` cases.
- Production path unchanged. V15 also stays env-off.

## Update 4 — 不可优化定论（2026-07-01 收尾）

本小时的工作以「尝试 INT8 `__dp4a` kernel 是否有加速效果」为最终实验，结论为
**否**。结合此前 V14_MB 校准与 V15 三种 tile，Jetson sm_72 上 small-M（batch=4/6/8）
weight-only Linear 的 int4-based 优化空间已被穷尽且证伪。此节作为正式的「不可优化」
记录写入日志，避免后续重复尝试同一类路线。

### 不可优化定论

**Jetson AGX Xavier (sm_72) 上，decode-repair `x=1→x=3`（及 x=3/5/7 整体）的 TPOT
悬崖，无法通过任何 int4-based Linear kernel 闭合。**

依据（全部为本小时 Jetson direct-op 实测，`memory=2`，target shapes）：

| 路线 | batch=4 hidden_to_inter (2048→8192) | 相对 cuBLAS | 瓶颈 | 失败类型 |
|---|---:|---:|---|---|
| cuBLAS static-dequant FP16 (生产 baseline) | 0.314 ms | 1.0× | 已达峰值带宽 (~108 GB/s) | — |
| V14_MB int4 (强制 limit=6 命中) | 0.401 ms | 1.28× 慢 | nibble ALU instruction-bound | 实测证伪 |
| V15 v1 int4 (shared-mem 4-thread load) | 2.070 ms | 6.6× 慢 | weight load 串行化 | 实测证伪 |
| V15 v2 int4 (batch-pair register load) | 0.740 ms | 2.4× 慢 | 每 pair 重读权重 | 实测证伪 |
| V15 v3 int4 (single-load inner batch) | 1.394 ms | 4.4× 慢 | register spill (OC_PER_BLK=8) | 实测证伪 |
| V16 dp4a v1 (per-group grid) | 21.48 ms | 68× 慢 | block 数爆炸 (32k blocks) | 实测证伪 |
| V16 dp4a v2 (in-block group loop) | 45.32 ms | 144× 慢 | register spill 到 DRAM | 实测证伪 |

两个独立瓶颈封死所有 int4 衍生路线：

1. **nibble ALU 瓶颈（V14/V15 全系）**：sm_72 上 int4 nibble 的
   shift/mask/int2float/FMA 是 instruction-bound。即便最成熟的 V14_MB，batch=4
   实测 0.40ms 也已慢于 cuBLAS 0.31ms，且随 batch 线性退化（0.23→0.40→0.59）。
   int4 权重 4× 带宽红利被 per-nibble ALU 开销吃掉。
2. **dp4a 寄存器瓶颈（V16）**：factored per-(oc,group) combine 需在 32 个 group 的
   in-block 循环里维持大量累加器/参数状态，OC_PER_BLK≥4 即 spill 到 DRAM，单次
   launch 44ms。group-grid 变体则 block 数爆炸（32768 blocks）。

被替换的 dense 路径（rows45 cuBLAS static-dequant）**已达峰值 DRAM 带宽**（108 GB/s
读 FP16 权重），本身高效，没有「慢路径」可优化——这是 int4 路线无法赢的根本原因：
要赢必须 4× 省权重字节且不付算力代价，但 sm_72 上 int4/dp4a 都要付算力或寄存器代价。

### 已累计 5 个 int4 衍生变体失败

本小时新增 2 类（V15 三 tile + V16 两 tile），叠加 `mnn-opt-ops` SKILL 行 248-259
历史 3 类（V14_MB retune、fused SwiGLU+down、WMMA down），共 5 大类、8 个具体 tile
全部在 Jetson direct-op 上证伪。**后续不应再以任何 int4-native / int4-dp4a / int4
fused-MLP kernel 作为 Jetson rows4/6/8 small-M 的主线优化方向。**

### 什么还有效（保留为资产）

- **V16 factored-dp4a 数学是 bit-exact 正确的**（accuracy `bad=0` 全过）。这套
  `acc = s_w·s_a·dp4a + s_w·o_a·sum_nibble + adj_w·s_a·sum_aq + adj_w·o_a·K` 的
  不对称量化 + dp4a 框架本身是可用的数学资产，瓶颈纯在 sm_72 寄存器/算力。换到有
  高效 int4/int8 tensor-core 的设备（sm_90+ Hopper int4、或 Adreno）可能翻身。
- **V15/V16 env-gated 代码 + accuracy harness 全保留**（默认 off），作为 opt-in
  实验供未来设备复用，不进入 Jetson 生产路径。

### 生产决策

- 生产路径**完全不变**：rows4/6/8 仍走 rows45 cuBLAS static-dequant（V15/V16 env off）。
- V15 (`MNN_CUDA_PIC_INT4_SMALLM_V15`)、V16 (`MNN_CUDA_PIC_INT4_SMALLM_DP4A`) 均默认 off。
- 无图导出改动、无 op/schema 改动、无 `onResize` 语义改动、无新模型产物依赖。

### 完整优化思路总结

见同目录 `OPTIMIZATION_SUMMARY.md`：系统梳理所有已尝试路线（有效/无效/未做），
明确哪些方向已被证伪、哪些尚未尝试但受 PLAN 硬约束排除、哪些是真正剩下的潜在
非-int4 杠杆。




