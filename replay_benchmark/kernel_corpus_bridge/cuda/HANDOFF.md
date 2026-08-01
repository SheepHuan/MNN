# CUDA Kernel Corpus — 交接文档

> **目标**: 将 MNN CUDA backend 的所有 `__global__` kernel（含历史版本 1.2.0-3.6.0）
> 忠实复制到 `replay_benchmark`，按 kernel-adapt skill 的"变体 vs 版本"分层规则实现。
>
> **当前状态**: 30 个 kernels/*.cu 文件，300+ 个 adapter 类，332 + 48 个 case（两条工作线合并）。
> 编译 0 error。所有 MNN `__global__` kernel 已 100% 覆盖（含 1.2.0-3.6.0 全版本 + BF16）。
> **两条工作线**：
> - 主线（remote）：25 个 legacy kernel + 28 个 legacy adapter + 37 个 weight_only_quant + prefill adapter（326 → 332）
> - 本地（HEAD）：raster 融合宏实例批量补齐 + BinaryMid4/Half2 向量化 + BINARY_INT8 完整覆盖 + SPLIT_FusedKV（261 → 309，+48）
>
> **核心原则**: 忠实复制 MNN kernel 源码。adapter 只负责参数转换 + buffer 打包 + 验证。

---

## 1. 文件结构

```
replay_benchmark/kernel_corpus_bridge/cuda/
├── CudaOpAdapter.hpp              ← CudaOpAdapter 接口 + CudaLaunchCtx
├── CudaOps.hpp                    ← 所有 adapter 类声明 + helper (227 个类)
├── CudaOps.cpp                    ← A类 fp32 adapter + registerCudaOps()
├── CudaOpsMisc.cpp                ← B/C/D 类 fp32 adapter + legacy kernel adapter
├── CudaOpsWoq.cpp                 ← weight_only_quant fp32 adapter + registerWoqAdapters()
├── CudaOpsFp16.cu                 ← fp16/int8/plugin/raster_fuse/woq adapter (nvcc)
├── CUDA_KERNEL_VERSIONING.md      ← 版本演进文档
├── HANDOFF.md                     ← 本文件
└── kernels/                       ← GPU kernel 定义层 (nvcc, 29 个文件)
    ├── corpus_common.cuh           ← 共享 helper: UP_DIV, DivModFast, blockReduceSum, CUDA_KERNEL_LOOP
    ├── unary_cast.cu               ← RELU/CLAMP/CAST/CASTMIDFLOAT/BF16转换 等
    ├── binary.cu                   ← ATAN2/MOD/LOGICALOR
    ├── range.cu                    ← RANGE
    ├── select.cu                   ← SELECT
    ├── softmax.cu                  ← SOFTMAX naive + WARP_32 + AXIS_REDUCE + DIVSUM/EXPSUB(1.2.7)
    ├── layernorm.cu                ← LAYERNORM + layernorm_c4 + binary_layernorm_c4 + input_layernorm_320/512/1024/2048/adaptive
    ├── prelu.cu                    ← PRELU
    ├── scale.cu                    ← SCALE
    ├── pool.cu                     ← MAXPOOL/AVGPOOL/GLOBAL + halfC16/floatC16(1.2.7) + BF16 pool
    ├── gatherv2_argmax.cu          ← GATHERV2 + ARGMAX + 两阶段
    ├── interp.cu                   ← INTERP nearest/bilinear/round + OPT (含 fp16)
    ├── transpose.cu                ← NHWC↔NCHW + TRANSPOSE + TRANSPOSE_LOCAL + PACKCOMMON/UNPACKCOMMON + blit
    ├── gridsample.cu               ← GRID_SAMPLE nearest/bilinear/3D (含 fp16)
    ├── reduction_naive.cu          ← SUM/MEAN/MAX/MIN/PROD + axis 变体 (含 fp16)
    ├── topkv2.cu                   ← TopKAllRows/GetResultAllRows (含 fp16, 4x 展开)
    ├── raster.cu                   ← blitRegion + fuseblit/fuseblitLimit/fuseblit_4 + pack/unpack + SCATTERND
    ├── convdw.cu                   ← CONV_DW + WeightPrepare/BiasPrepare/BiasZeroPrepare
    ├── convdw_extra.cu             ← CONV_DW_OPT/HALF2_OPT/3x3_HALF2_OPT/MULTI_WIDTH4/MULTI_WIDTH_CHANNEL
    ├── convdw_bf16.cu              ← CONV_DW_BF16/BF162_OPT/3x3_BF162_OPT/MULTI_WIDTH4 + WeightTransToBf16/BiasTransToBf16 (sm75 空 kernel, sm80+ 运行)
    ├── rope.cu                     ← ropeC4Kernel (含 fp16)
    ├── matmul.cu                   ← GENERAL_BATCH_MATMUL + matmul_gemv_kernel + GemmPacked/GemmPrearrange (1.2.7, wmma)
    ├── conv_base.cu                ← Float22Half2 + Im2Col_FilterC + Im2Col_FilterC_Vec4 + WeightPackFill + PackPadFill
    │                                + Im2Col1x1*/Im2Col_half*/Im2Col (1.2.7, 含 MATMULPACK)
    ├── deconv.cu                   ← DeconvKernelReorder + Col2Im + Col2Im_Vec4 + cutPad/DECONV_DW/DeconvInputRerange (1.2.0/1.2.7)
    ├── winograd.cu                 ← WinoWeightReorder + WinoInputTrans/half2 + WinoTrans2Output/half2
    ├── attention.cu                ← flash_decode*/splitk + combine_results + qk/qkv_tiled + conv1d_silu + short_conv* + gated_delta_rule_decode/prefill (含 fp16)
    ├── transpose_half.cu           ← PACKCOMMON_half_4 + REARRANGE_half_4 + fuseblit_half_4 + UNARY_HALF2_SIGMOID
    ├── plugins.cu                  ← GroupNorm(cub::BlockScan忠实复制) + SeqLen2Spatial + splitGeLU + SPLIT_FusedQKV + SPLIT_FusedKV(2.8.4)
    ├── int8.cu                     ← FLOAT_2_INT8/INT8_2_FLOAT + DequantizeInt8/Int4Weight + CONV_DW_INT8 + Im2Col + BinaryInt8
    ├── raster_fuse.cu              ← BinaryADD/MUL + FuseAdd + BinaryMid + BinaryMidLinear4 (4 宏族)
    └── weight_only_quant.cu        ← GEMV/GEMM/CONV FpAInt4B/Int8B + QuantA/DequantAndAcc/BiasAndActivation + Rearrange + Precompute (3.6.0)
```

---

## 2. 构建 & 验证

```bash
export PATH=/usr/local/cuda-13.2/bin:$PATH
cd build && cmake --build . --target replay_benchmark.out -j$(nproc)

LD_LIBRARY_PATH=.:source/backend/cuda:. ./replay_benchmark.out \
  --kernel-corpus-bench --kernel-corpus-root ../replay_benchmark/kernel_corpus \
  --kernel-corpus-runs 1 --perf-counter-output /tmp/kc_cuda.json

python3 -c "
import json
d=json.load(open('/tmp/kc_cuda.json'))
cuda=[c for c in d['cases'] if c['backend']=='cuda']
print(f'total={len(cuda)}, passed={sum(1 for c in cuda if c[\"validation_status\"]==\"validation_passed\")}')
fails=[c for c in cuda if c['validation_status']!='validation_passed']
for c in fails: print(f'  FAIL {c[\"case\"]}: {c[\"variant\"]}')"
```

---

## 3. 覆盖率指标

| 指标 | 数值 |
|------|------|
| .cu 文件 | 30 |
| `__global__` kernel | 240 |
| shim 函数 | 297 |
| operators.json entry | 297 |
| case (operator_cases.json) | 332 |
| adapter 类声明 | 233 |
| adapter 注册 | 242 |
| shim 无 entry | 0 ✅ |
| 声明未注册 | 0 ✅ |
| 编译 error | 0 ✅ |
| 测试通过 | 332/332 ✅ |

### 跨版本覆盖

所有 MNN CUDA backend 在 1.2.0-3.6.0 任何版本存在过的 `__global__` kernel 都有对应 corpus 实现。
经 grep 差集验证：0 个真实缺失（除调试用 `print_tensor_kernel`）。

---

## 4. 支持的 tag

```
1.2.0, 1.2.7, 1.2.8, 2.0.2, 2.0.4, 2.1.2, 2.2.2, 2.2.3, 2.4.1, 2.4.2,
2.5.0, 2.5.1, 2.5.3, 2.7.1, 2.7.2, 2.8.0, 2.8.4, 3.6.0
```

---

## 5. 任务进度总览

| 任务 | 状态 | 说明 |
|------|------|------|
| 基础 fp32 (#1-59) | ✅ | A/B/C/D 类全部完成 |
| P0 fp16 Attention/RoPE/TopKV2 | ✅ | 含忠实 cub::BlockScan |
| P1 fp16 Reduction/Interp/GridSample/LayerNormC4 | ✅ | 含 input_layernorm_* size-specialized |
| P2 插件 kernel | ✅ | GroupNorm 忠实 cub::BlockScan (-fexceptions) |
| P3 bf16 depthwise | ⏸️ 跳过 | 需 sm80+; BF16 pool 已复制(空 kernel) |
| P4 int8/weight_only_quant | ✅ | 12+20+37 adapter 全部完成 |
| P5 Raster 融合 | ✅ 代表性 8 个 | 剩余 66 个宏实例待后续 |
| §7 忠实性审计 P0-P14 | ✅ 全部完成 | 见 §7 |
| 跨版本历史 kernel (1.2.0-2.8.4) | ✅ 25 个 | cutPad/DECONV_DW/SCATTERND/GemmPacked/Im2Col1x1*/pool_C16/DIVSUM/EXPSUB/SPLIT_FusedKV |
| §8 Validator 质量升级 | ✅ +14 个 | woq GEMV/GEMM (9) + convdw_extra (5) 从 smoke→real；修复 GEMV launch geometry bug |
| §9.2 BF16 原生算子 | ✅ 6 个 | CONV_DW_BF16/BF162_OPT/3x3_BF162_OPT/MULTI_WIDTH4 + WeightTransToBf16/BiasTransToBf16；sm75 空 kernel smoke + 2 transpose real validator |
| 332/332 真机通过 | ✅ | sm75 RTX 2080 Ti |

---

## 6. 命名规则

- **变体(variant) = 不同实现策略** → 独立 adapter 类 + 唯一 variant 名 (`cuda_<op>_<strategy>_fp32`)
- **版本(tag) = 同一变体在不同 MNN 版本的函数体更新** → adapter 内部按 `spec.tag` 分流
- 类名: `Cuda<Op><Strategy>Fp32Kernel`（含变体标识，不含 tag）

---

## 7. 忠实性审计报告

### 7.1 总览（全部完成）

| 优先级 | 文件 | kernel | 状态 |
|--------|------|--------|------|
| 🔴 P0 | transpose.cu | NHWC↔NCHW + NHWC8/C4NHW4 系列 | ✅ 已修: 公开 kernel 恢复真实 DivMod 重排；packed 路径保留 C4 deinterleave + zero-fill |
| 🔴 P1 | transpose.cu | PACKCOMMON/UNPACKCOMMON | ✅ 已修: MNN [kh][kw][c_p] 布局 + zero-fill |
| 🔴 P2 | transpose.cu | PACKCOMMON_4/UNPACKCOMMON_4 | ✅ 已修: UP_DIV(axis,2)*2 + zero-fill |
| 🟡 P3 | convdw.cu | CONV_DW | ✅ 已修: 权重[kh][kw][c_p] + UP_DIV + real validator |
| 🟡 P4 | plugins.cu | groupNormNHWCSum/Scale | ✅ 已修: 忠实 cub::BlockScan (-fexceptions) |
| 🟡 P5 | int8.cu | Im2Col_packC_16 | ✅ 已修: DivModFast 从参数构造 |
| 🟡 P6 | conv_base.cu | Im2Col_FilterC_Vec4 | ✅ 已修: precision==2(int64) + precision==3(bf16) |
| 🟡 P7 | deconv.cu | Col2Im_Vec4 | ✅ 已修: DATA_CONVERT_COPY 宏 + precision 参数 |
| 🟡 P8 | topkv2.cu | TopKInThread | ✅ 已修: 4x 展开 + #pragma unroll |
| 🟢 P9 | binary.cu | MOD | ✅ 无需改: MNN 从未用 fmod (git 历史确认) |
| 🟢 P10 | layernorm.cu | input_layernorm_* | ✅ 已补: 5 个 size-specialized kernel |
| 🟢 P11 | pool.cu | BF16 pool | ✅ 已复制: #if __CUDA_ARCH__>=800 守卫 |
| 🟢 P12 | transpose_half.cu | PACKCOMMON_half_4 | ✅ 适配标注: runner 架构限制 (int2 4-half) |

### 7.2 核心教训（已加入 SKILL.md）

> MNN 能编译的 kernel，corpus 理论上也能编译。
> 1. 编译失败 → 查 MNN CMake flag 覆盖（如 `-fexceptions`）
> 2. 架构不支持 → 验证 `#if __CUDA_ARCH__` 守卫下类型是否可用
> 3. runner 不兼容 → 区分"编译限制"与"架构限制"
> 4. 描述与源码不符 → 以 `git log`/源码为准

### 7.3 P0 workload layout 与 transpose 方向审计（2026-08-01）

PMC Interpreter 的 workload 控制量必须描述 adapter 实际索引的逻辑张量，不能仅根据
op 名称猜测 layout。本轮对 P0 中容易混淆的三类进行了 adapter/source/case 三方核对：

| 类别 | 结构化语义 | 证据边界 |
|------|--------------|----------|
| `conv_dw` | 1.2.0 input/output 为 NCHW，weight 为 `[C, KH, KW]`；2.0.4 input/output 为 channel-last，weight 仍为 `[C, KH, KW]`；2.2.3+ prepared weight 为 `[KH, KW, packed_C]` | 来自精确 `cuda_conv_dw_fp32` adapter 和版本分流，不外推到其他 conv variant |
| `maxpool` / `avgpool` | 1.2.0 与独立 `_120` adapter 按 NCHW 索引；现代 C8 adapter 按 channel-last packed order 索引 | `CudaAvgPoolFp32Kernel` 保留真实 `ow = oh`；`cuda_avgpool_120_fp32` 独立用 `iw` 计算 `ow` |
| generic/local transpose | `[M, N] → [N, M]` | 只匹配精确 variant 白名单 |
| BDL→BLD | `[B, D, L] → [B, L, D]` | 来自 linear-attention adapter 契约 |
| NHWC↔NCHW | NHWC `[B, area, C]` 与 NCHW `[B, C, area]` 按声明方向重排 | FP32/FP16 validator 逐元素比较真实目标位置，不再验证 identity |
| packed format transpose | NCHW/C4NHW4 与 NHWC/NHWC8 按 variant 方向记录未 padding 的逻辑 extent | allocation padding 不计入 algorithmic shape/bytes |

源码忠实性证据：

- `NHWC_2_NCHW` 与 `NCHW_2_NHWC` 已恢复为公开
  `source/backend/cuda/execution/Transpose.cu` 的 `DivModFast` 重排函数体；
  `diff --ignore-all-space` 均为空。
- 2.1.2 历史 kernel 实际是 NCHW→NHWC，已分流为
  `mnn_corpus_nchw2nhwc_212_fp32` / `cuda_nchw2nhwc_fp32`；函数体与
  `git show 2.1.2:source/backend/cuda/execution/Transpose.cu` 忽略空白后一致。
- `replay_benchmark.out` 重新构建通过。FP32/FP16 NHWC→NCHW、FP32/FP16
  NCHW→NHWC、2.1.2 NCHW→NHWC、generic、local 和 BDL→BLD 共 8 个 case 均为
  `dispatched` + `validation_passed` + `valid=true`。

`enrich_pmc_workload_metadata.py` 在当前 manifest 上处理 122 个 P0 case：
`avgpool=6`、`conv_dw=14`、`matmul=11`、`maxpool=8`、`reduction=45`、
`softmax=15`、`transpose=23`。连续两次 `--in-place` 后 SHA256 均为
`55fe5369daa6ad3f4e011cd76372eb2f4772801e5f261ddb8f53c867978f2c5b`，证明本轮刷新幂等。
该 SHA 是当前工作树审计身份，不是永久常量；manifest 再变更后必须重新计算。

---

## 8. Validator 质量分析

### 8.1 真实验证 (101 个)

host 端重算 expected 值，与 kernel output 用 `fabs` 对比。涵盖：
- RELU/CLAMP/CAST/RANGE/SELECT/SCALE/PRELU
- SOFTMAX (naive/warp32/axis_reduce)
- LAYERNORM (含 input_layernorm_*)
- MAXPOOL/AVGPOOL/GLOBAL (3.6.0)
- REDUCTION (SUM/MEAN/MAX/MIN/PROD + axis)
- INTERP (nearest/bilinear/round)
- GRID_SAMPLE (nearest/bilinear/3D)
- GATHERV2/ARGMAX/ARGMIN + 两阶段
- TRANSPOSE（真实 NHWC↔NCHW 逐元素重排 + 格式转换 + fuseblit）
- PACKCOMMON/UNPACKCOMMON/Col2Im/DeconvKernelReorder
- MATMUL (GENERAL_BATCH_MATMUL + gemv)
- CONV_DW (3.6.0, real host recompute)
- ROPE (C4, real host recompute)
- TopKV2
- Raster 融合 (BinaryADD/MUL + FuseAdd + BinaryMid + Linear4)
- INT8 (FLOAT_2_INT8/INT8_2_FLOAT/DequantizeInt8/Int4Weight/CONV_DW_INT8/BinaryInt8)
- GroupNorm (Sum/Scale, cub::BlockScan)
- SeqLen2Spatial / splitGeLU / SPLIT_FusedQKV
- BiasAndActivation / PrecomputeGemvParams / Precompute_SumBq / GemmInt8
- TRANSPOSE_LOCAL / fuseblit_4
- CASTMIDFLOAT / float2int8 / int82float
- maxpool/avgpool 1.2.0 / reduction sum/mean 1.2.0
- NHWC8/C4NHW4 格式转换 (11 个)
- **weight_only_quant GEMV/GEMM (9 个)**：GEMM_FpAInt8B/Int4B + GEMV_FpAInt8B/Int4B/V5/V9/V14/V14_MB/V2 — 用 woqRefInt8/woqRefInt4/woqRefInt4V14 host 反量化+矩阵乘重算
- **convdw_extra (5 个)**：CONV_DW_OPT/HALF2_OPT/3x3_HALF2_OPT/MULTI_WIDTH4/MULTI_WIDTH_CHANNEL — 用 convdwExtraRefFp32 host depthwise conv 重算
- **BF16 transpose (2 个)**：WeightTransToBf16/BiasTransToBf16 — host 重算 transpose + bf16 精确匹配（sm75 可运行，无 arch 守卫）

### 8.2 Smoke-only validator (44 个) — 待改进

只检查 output 非全零，不验证具体值。按类别：

| 类别 | 数量 | 改进难度 | 说明 |
|------|------|---------|------|
| BF16 convdw (sm75 空 kernel) | ~4 | sm80+ | 需 sm80+ 运行真实 bf16 计算；sm75 smoke 合理 |
| weight_only_quant CONV | ~4 | 高 | 需 im2col+反量化+卷积 host 重算；woqRef helper 不适用（conv 权重布局不同） |
| attention/flash_decode | ~10 | 高 | 需在线 softmax host 重算 |
| winograd | ~5 | 中 | 需 Winograd 变换 host 重算 |
| conv_base/im2col | ~3 | 中 | 需 im2col host 重算 |
| raster/pack/rearrange | ~5 | 低 | 布局转换 host 重算 |
| BF16 pool/float22bfloat16 | ~3 | N/A | sm75 空 kernel，smoke 合理 |
| Precompute_SumBq/QuantA/DequantAndAcc | ~3 | 高 | 需 int8 量化+累加重算 |
| gated_delta_rule_prefill | ~1 | 高 | 线性 attention，复杂 |
| rope/quantA/packC4 等其他 | ~6 | 低-中 | 各需 host 重算 |

### 8.3 改进建议

**已完成（本轮）**：
- ✅ NHWC↔NCHW 公开 kernel 与 validator：原 corpus 的两个简化函数实际是
  identity copy，现已恢复 MNN `DivModFast` 重排逻辑；FP32/FP16 validator 均按声明方向
  计算 `src`/`dst`。2.1.2 case 从误标 NHWC→NCHW 修正为真实 NCHW→NHWC。
- ✅ weight_only_quant GEMV/GEMM (9 个 adapter)：`woqRefInt8`/`woqRefInt4`/`woqRefInt4V14` 提升到共享头 `CudaOps.hpp`，adapter 在 `validate()` 调用。同时修复了 GEMV 适配器预先存在的 launch geometry bug（原 gridX=(oc+15)/16、block=16/64 与 MNN `onExecute` 不一致，被 smoke validator 隐藏）
- ✅ convdw_extra (5 个 adapter)：新增 `convdwExtraRefFp32` host depthwise conv 重算，参考既有 `CudaConvDwFp32Kernel::validate` 公式（改进：支持 batch>1 + 从 validatorInputC 读 bias + clamp）
- ✅ §9.2 BF16 原生算子 (6 个 kernel + adapter)：`CONV_DW_BF16`/`BF162_OPT`/`3x3_BF162_OPT`/`MULTI_WIDTH4` + `WeightTransToBf16`/`BiasTransToBf16`。4 个 convdw sm75 smoke（空 kernel），2 个 transpose real validator（host 重算 + bf16 精确匹配）

**后续中优先级**：
- weight_only_quant CONV (FpAInt8B/Int4B)：需 im2col + 反量化 + 卷积 host 重算（权重布局 [oc, ic_p*kh*kw]，与 GEMV/GEMM 不同）
- convdw_extra fp16 variants：fp16 input+weight（非 fp32 input），需独立 half-path reference
- BF16 convdw (sm80+)：需 sm80+ 环境验证真实 bf16 计算

**低优先级**（smoke 合理）：
- BF16 pool/float22bfloat16: sm75 空 kernel，无法验证值
- SetZero: validator 检查全零 = 正确

---

## 9. 覆盖率缺口（剩余）

### 9.1 Raster 融合剩余实例 (~66 个)

4 个宏族 × 18 种运算 (ADD/SUB/MUL/DIV/...) − 已实现 8 个 = ~66 个。
通过添加 `BINARY_FUNC(SUB, x-y)` 等实例 + 注册 adapter 扩展。

### 9.2 BF16 原生算子 (6 个) — ✅ 已完成（见 §5/§8）

`CONV_DW_BF16`/`CONV_DW_BF162_OPT`/`CONV_DW3x3_BF162_OPT`/`CONV_DW_BF16_MULTI_WIDTH4`/
`WeightTransToBf16`/`BiasTransToBf16`

全部 6 个 kernel 忠实复制到 `kernels/convdw_bf16.cu`（含 `#if __CUDA_ARCH__>=800` 守卫）。
- 4 个 convdw：sm75 编译为空 kernel（smoke validator 合理），sm80+ 运行真实计算
- 2 个 transpose（WeightTransToBf16/BiasTransToBf16）：无 arch 守卫，sm75 可运行，real validator
  （host 重算 transpose + bf16 精确匹配）

maxpool/avgpool_C8_BF16 已在 pool.cu 中。需 sm80+ 环境验证 convdw 的真实计算。

`plugin/FmhaCommon/FmhaV2CommonExecution.cu:30`。与 SPLIT_FusedQKV 类似（2-way split）。

✅ **本次已补齐** (P4)：fp32 + fp16 两个 adapter + case，忠实复制 MNN 索引公式。

---

## 9. 本次扩展记录（merge kernel-agent-3.6.1 后）

> 基于 sm_89 RTX 4080 + CUDA 12.5 + g++-12 host compiler 环境。
> 起始状态 261/261 → 最终 309/309 (+48 cases, +18.4%)。

### 9.1 P1: Raster 融合宏实例批量补齐 (+34 cases)

| 类别 | 新增 op | 文件 |
|------|---------|------|
| raster_binary | SUB/DIV/MINIMUM/MAXIMUM/FLOORDIV/FLOORMOD/SquaredDifference/POW (8) | raster_fuse.cu |
| raster_binary_fuseadd | SUB/DIV/MINIMUM/MAXIMUM/FLOORDIV/FLOORMOD/SquaredDifference/POW (8) | raster_fuse.cu |
| raster_binarymid | SUB/MUL_SILU/DIV/MINIMUM/MAXIMUM/FLOORDIV/FLOORMOD/SquaredDifference/POW (9) | raster_fuse.cu |
| raster_binarymidlinear4 | SUB/MUL_SILU/DIV/MINIMUM/MAXIMUM/FLOORDIV/FLOORMOD/SquaredDifference/POW (9) | raster_fuse.cu |

**重构**：`raster_fuse.cu` 的 extern "C" shim 区从手写改为宏驱动（`SHIM_BINARY` / `SHIM_BINARY_FUSEADD` / `SHIM_BINARYMID` / `SHIM_BINARYMIDLINEAR4`)，便于批量实例化。
**注意**: `raster_binary_fuseadd` 是 `atomicAdd(output, OP(x,y))`，非幂等，case 必须 `warmup_runs=0` 且 `workload_runs=1`(CLI `--kernel-corpus-runs 1` 验证)。

### 9.2 P2: PACK_NUMBER 向量化变体 (+6 cases)

| 变体 | 数据类型 | PACK | 说明 |
|------|---------|------|------|
| BinaryMid4_<ADD/MUL> | fp32 | float4 | 3D stride,stride 不含 X 维（X 内嵌 `ix<<2`) |
| BinaryMidHalf2_<ADD/MUL> | fp16 | half2 | 同上，PACK_NUMBER=2 |
| BinaryMidLinearHalf4_<ADD/MUL> | fp16 | half2×2 | 1D linear，每个线程 4 half |

**关键 fp16 validator 修复**:`output.data()` 是 raw bytes，需 `reinterpret_cast<const __half*>` + `__half2float` 才能与 fp32 expected 比较。

### 9.3 P3: BINARY_INT8 完整覆盖 (+6 cases)

| 类别 | 新增 |
|------|------|
| BINARY_INT8 (single-scale) | SUB/DIV/MINIMUM/MAXIMUM |
| BINARY_INT8_CHANNELWISE (per-channel scale) | ADD/MUL |

**关键修复**:
1. `int8.cu` 的 `BINARY_INT8_ADD/MUL` 从手写重构为 `BINARY_INT8_FUNC` 宏驱动，便于批量实例化。
2. `host_float2int_rn` 从 `(int)roundf` 改为 `(int)rintf`(banker's rounding),匹配 device `__float2int_rn` 的 round-to-nearest-even 语义。原先 DIV `0.25*10=2.5` 时 host 期望 3、device 给 2 导致 false negative。
3. DIV 的输入数据 `in1[i] = 1 + (i % 5)` 避免 y=0 触发 inf/NaN 的 host-device 行为分歧。

### 9.4 P4: SPLIT_FusedKV (+2 cases)

`plugin/FmhaCommon/FmhaV2CommonExecution.cu:30`,fp32 + fp16 两 adapter,validator 检查 v 输出（k 不读回）。

### 9.5 未覆盖项（已记录）

| 类别 | 数量 | 状态 | 备注 |
|------|------|------|------|
| 权重量化 GEMV/GEMM int4/int8 | ~20 | ⏸️ 跳过 | cutlass 集成复杂，一次性投入大 |
| Raster 融合剩余实例 | ~22 | ⏸️ 跳过 | GREATER/LESS/EQUAL/NOTEQUAL/LOGICALOR 返回 int 类型，需独立 validator;REALDIV/ATAN2/MOD 已存在或冗余 |
| BF16 原生算子 | 8 | ⏸️ 跳过 | 需 `MNN_CUDA_BF16=ON` 重编 backend；当前 sm89 可跑但需配置切换 |
| `BinaryMid4_` SUB/DIV/... 8 个 op | 8 | ⏸️ 可选 | 已覆盖 ADD/MUL 两个代表性，剩余宏实例遵循同一模式可批量加 |

---

## 10. 提交历史

| commit | 描述 |
|--------|------|
| f1916a8fd | Complete 37 weight_only_quant + prefill adapters — 326/326 pass |
| c14a4aa00 | Split legacy_kernels.cu into per-op-type files + add 28 adapter cases |
| c351445e8 | Add 25 legacy kernels from deleted MNN versions (1.2.0-2.8.4) |
| d1e38d7f6 | Add weight_only_quant kernels: 100% __global__ coverage |
| 99214600c | Fix faithfulness gaps found in corpus-wide audit |
| c49a5210b | Re-evaluate P4/P11/P12: faithful cub::BlockScan + BF16 pool copy |
| 75b90581d | Complete §7 faithfulness audit (P0-P14) + add int8/plugins/raster_fuse/input_layernorm — 261/261 pass |
| 950432ed3 | Complete CUDA kernel corpus: fix 6 compile errors + add 29 kernel variants — 196/196 pass |

## 11. 测试基础设施审计：PMU multi-pass

本节记录对 `replay_benchmark/tests` 的补充审计。这里的问题不一定会导致 CUDA kernel 数值错误，但会让 PMU 回归结果出现“误通过”或未定义行为，因此不能把当前 PMU 测试结果直接视为有效覆盖。

### 11.1 multi-pass 状态被 single-pass 覆盖

位置：

- `replay_benchmark/tests/test_pmu_multipass.cpp:199-211`
- `replay_benchmark/tests/test_pmu_multipass.cpp:246-256`

`checkMultiVsSingle()` 先调用一次多指标 `runBench()`，随后针对每个指标调用 `runSingleMedian()`。`runBench()` 和 `getPmuStatus()` 都使用固定文件 `/tmp/_pmu_test.json`；因此 `runCombination()` 在 `checkMultiVsSingle()` 返回后读取到的 status 实际来自最后一次 single-metric 运行，而不是最初的 multi-metric 运行。对 `expectMultiPass` 为真的组，这会使 status 断言验证错误的采集。

**建议改法：**

1. 让一次 benchmark 返回结构化结果，例如 `BenchResult { metrics, pmuStatus, commandOk, parseError }`，在 `runBench()` 读文件的同一时刻解析并保存 `pmu_status`。
2. 让 `checkMultiVsSingle()` 返回 multi-pass 的 `BenchResult` 或至少返回 `multiPmuStatus`；`runCombination()` 只检查该快照，不再事后读取共享文件。
3. 更进一步，为每次运行生成唯一临时输出路径，并以 RAII 或显式清理管理路径，避免并行测试或残留文件造成串读。当前 `system()` 命令还应检查输出文件是否为本次进程生成，而不是只检查文件存在。
4. `AutoSplitStatus` 测试也应直接使用 `runBench()` 返回的 status，移除对 `getPmuStatus()` 的独立文件读取。

### 11.2 无效 PMU 指标被静默跳过

位置：

- `replay_benchmark/tests/test_pmu_multipass.cpp:203-211`
- 另有类似逻辑：`replay_benchmark/tests/test_pmu_multipass.cpp:341-347`

当 multi 或 single 结果为负值时，测试只打印 `SKIP` 并继续。若所有指标均不可用、采集失败、JSON 缺失或返回溢出哨兵值，测试函数仍可能没有任何失败断言，最终报告为 PASS。这把“硬件不支持”和“测试基础设施坏了”混成了同一种状态，也可能产生空通过。

**建议改法：**

1. 不要用 `double < 0` 表示所有错误。定义结果状态，例如 `Valid`、`Unsupported`、`Overflow`、`CommandFailed`、`MalformedOutput`，并保留错误文本、退出码和 `pmu_status`。
2. 在测试开始做一次 capability preflight：确认 CUDA 设备、权限、目标 metric 可用，并区分“平台不支持”与“采集过程失败”。固定要求的 metric 不可用时默认 FAIL；只有明确声明为 optional 的 metric 才允许跳过。
3. 对允许硬件差异的场景实现显式 `SKIPPED` 结果，而不是打印文字后按 PASS 计数。至少要求每个测试有 `validComparisons > 0`；否则必须失败或被测试框架统计为 skipped，不能按 passed 统计。
4. 对一个 metric 的 multi 和 single 任一侧无效时，应报告该 metric 的状态和原因，并将该比较计入失败/跳过计数，而不是静默 `continue`。`InstructionReproducibility_AllKernels` 的 `SKIP` 逻辑也应采用同一策略。
5. 对 PMU 不可用的机器，可通过明确的环境开关或独立的 capability test 选择整组 skipped；默认 CI 不应通过“全部不可用”的结果。环境开关必须在日志中打印，避免误把降级运行当成完整验证。

### 11.3 `setjmp/longjmp` 跨越 C++ 对象生命周期

位置：

- `replay_benchmark/tests/ReplayTest.hpp:54-63`
- `replay_benchmark/tests/ReplayTest.hpp:70-111`
- `replay_benchmark/tests/ReplayTest.hpp:142-146`

fatal assertion 从 `check*()` 调用 `longjmp()` 回到 `RunAll()` 的 `setjmp()`。如果失败发生在测试函数或其辅助函数中，跳转会绕过这些栈帧上已经构造的 `std::string`、`std::vector`、`std::function` 等非平凡 C++ 对象的析构。该用法不能提供 C++ 异常的栈展开保证，存在资源泄漏、锁未释放以及未定义行为风险；测试失败本身还可能污染后续测试。

**建议改法，按优先级：**

1. 首选结构化错误传播：`check*()` 只记录 failure 并返回 `bool`，`ASSERT_*` 宏在当前测试函数中显式 `return`；需要在辅助函数中使用断言的函数改为返回 `bool`，调用方检查后返回。当前 `ASSERT_EQ` 的使用点很少，迁移成本低。
2. 为测试结果增加 `fatalFailure`/`aborted` 标志，禁止失败后继续执行可能依赖前置条件的代码；这只是控制流状态，不使用跳转。`RunAll()` 在函数正常返回后根据该标志计失败。
3. 如果未来必须保留“任意嵌套层立即退出”的语义，应改用受控的错误对象沿调用链返回；不要在 C++ 对象活跃时使用 `longjmp`。在本项目禁用异常的构建约束下，不建议用异常替代。
4. 为回归测试增加一个带 `std::string`/`std::vector`/文件句柄的 fatal assertion 场景，确保修复后测试仍能继续运行下一个 test，并验证资源清理行为。

### 11.4 建议的最小实施顺序

1. 先将 `runBench()` 改为返回包含 metrics、status、退出码和解析状态的对象，消除共享文件状态覆盖。
2. 再把 invalid/unsupported/error 分类和“零有效比较失败”加入 PMU 测试，修复空通过。
3. 最后移除 `ReplayTest.hpp` 的 `setjmp/longjmp`，将 `ASSERT_*` 改为结构化返回，并补充失败后资源清理测试。

初始审计环境未将 `nvcc` 放入 `PATH`，因此当时只能做主机侧检查；后续发现 `/usr/local/cuda/bin/nvcc` 可用并完成了 CUDA corpus 构建。CUDA 设备上的 PMU 行为仍需在具备可用硬件计数器的环境复核。

## 12. PMU 测试基础设施改进实施记录

本轮已按上述方案修改：

- `test_pmu_multipass.cpp` 的 `runBench()` 现在返回包含 metric 结果、`pmu_status`、命令状态和错误信息的 `BenchResult`。
- 每次采集使用带 PID/序号的独立 JSON 路径，并在读取后清理；multi-pass status 从本次结果快照读取，不再被后续 single-pass 覆盖。
- metric 结果区分 `Valid`、`Unsupported`、`Overflow`、`CommandFailed`、`MalformedOutput`。PMU status 报告 `start_failed`/`error` 时，缺失 metric 归为 `CommandFailed`，不再误报为硬件不支持。
- multi-vs-single、指令复现和 AutoSplitStatus 测试都要求至少有一个有效 metric/比较；不可用结果会打印明确分类，采集错误会使测试失败。
- `ReplayTest.hpp` 已移除 `setjmp/longjmp` 和 `csetjmp` 依赖。`check*()` 返回 bool 并设置 `fatalFailure`，`ASSERT_*` 在当前 void 测试函数中结构化返回，保留析构路径。
- `test_pmu_availability.cpp` 新增 `CudaMetric.AllMetrics`：依赖 `list_cuda_metrics --no-submetrics` 发现的全部 metric，逐个启动单 metric benchmark，并分别报告 `VALID`、`NOT_FOUND`、`OVERFLOW`、`COMMAND_FAILED`；域测试仍只覆盖各域的 `.sum` 子集。

验证记录：

- `cmake -S . -B build-x86-cuda -DMNN_CUDA=ON`：通过。
- `cmake --build build-x86-cuda --target replay_pmu_test -j2`：通过。
- `cmake --build build-x86-cuda --target replay_benchmark.out -j2`：通过，包含 CUDA corpus 的 NVCC 编译。
- `cmake --build build-x86-cuda --target replay_pmu_availability -j2`：通过；保留该文件已有的 C++17 structured-binding 和 `fread` 警告。
- `./replay_pmu_test --filter PMU.ConvDw_Compute`：在当前 CUPTI 返回 `CUPTI_ERROR_HARDWARE_BUSY` 时明确失败，未空通过。
- `./replay_pmu_availability --filter CudaMetric.Reproducibility`：fatal assertion 正常结束当前测试并报告失败，未发生跨栈跳转。
- `./replay_pmu_availability --filter CudaMetric.Discovery`：通过，发现 7200 个 CUDA metric。

当前机器的 CUDA PMU 采集状态为 `cuda_event_fallback(start_failed: ... CUPTI_ERROR_HARDWARE_BUSY)`，因此尚未完成真实硬件上的 multi-pass 数值一致性验证。
全量 availability 扫描应单独运行 `./replay_pmu_availability --filter CudaMetric.AllMetrics`；该测试会执行约 7200 次单 metric benchmark，预计耗时很长，不应与其他 PMU 测试并发运行。

## 13. CUPTI multi-pass 语义修正

补充验证发现，CUPTI/NVPW 返回的 `numPasses > 1` 可能发生在单个 metric 上：derived metric 依赖多个 raw counter slot，并不表示请求了多个 metric。此前 replay 层把所有 `numPasses > 1` 配置手工拆成单 metric session，导致单 metric 也出现 `Split 1 metrics into 1`，并错误地把 pass 处理从 CUPTI Session 中移出。

当前已修正为：一次 benchmark 只创建一个 PMU Session；NVIDIA 使用
`CUPTI_AutoRange + CUPTI_KernelReplay`，由 CUPTI 自动识别 corpus workload
中的 kernel、完成 multi-pass replay 与聚合；`report.num_passes` 保留 Session
报告的真实 pass 数。AutoRange 下 `beginRange()/endRange()` 为兼容接口 no-op。
32 metric batch smoke 验证结果为 `pmu_status=sampled`、`num_passes=2`，32 个
metric 全部返回有限值。

旧 collector 验证中 `lts__t_sectors_srcnode_gpc_op_atom_dot_alu_lookup_hit.min` 因
double 强制转换为 uint64 而写出 `9223372036854775808` 哨兵。该值不是有效 PMU
计数；新 collector 必须使用每 metric 状态显式报告 `overflow`，不得将其计入
`VALID`。

## 14. P0 CUDA 三数据源当前状态与重采要求

本轮修改了 `operator_cases.json` 的 case 身份、layout/workload 元数据和 transpose
collector/validator。因此修改前产生的 availability、selection plan、PMC rows、latency 与
normalized dataset 全部作废，只能作为历史探索数据。不得继续使用旧 rows 的
`--resume`，也不得沿用旧的 case、metric 或 rows 数量作为验收常量。

当前正式 CUDA 原始输入固定为：

- `replay_benchmark/kernel_corpus/operator_cases.json`；
- `build-x86-cuda/cuda_kernel_latency.json`；
- `build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv`；
- rows 必须携带
  `build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv.selection.json`。

重采顺序必须是：

1. 对最终 manifest 再运行 workload enrichment，并验证连续两次 SHA 不变；
2. 重新构建 `replay_benchmark.out`、`list_cuda_metrics`、
   `replay_pmu_availability` 和 `replay_kernel_latency`；
3. 从新的空日志运行 availability，动态生成本轮 `cuda_pmc_valid.csv`；
4. 使用唯一正式策略 `--case-selection condition-balanced`、七个 P0 target 和每类
   5 个 condition，从空 rows 账本生成新 `CaseSelectionPlan` 与 PMC rows；
5. 显式关闭 PMU，独立采集全部 CUDA manifest case 的结构化 latency、真实
   launch/resource 和环境信息；
6. 使用 `skills/pmc-source-gate/scripts/validate_cuda_pmc_sources.py`，并启用
   `--require-complete-targets` 与 `--require-measured-environment`，完成 schema、
   manifest SHA、plan、有序笛卡尔积、session、环境和 latency join 严格验收；只有
   `valid=true` 才能构造正式 `mnn-pmc-dataset/v1`。

全部命令、字段和发布门禁以两份中文规范为准：

- `docs/superpowers/specs/2026-07-31-pmc-interpreter-cuda-data-sources.md`；
- `docs/superpowers/specs/2026-07-31-pmc-interpreter-module-physical-semantics.md`。

注意：历史文档中的 `--case-selection op-type` 别名已删除，不做兼容。
`all` 只用于调用方明确要求的全 case 扫描，不是正式 P0 分析的默认策略。
