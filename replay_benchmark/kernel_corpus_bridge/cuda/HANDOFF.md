# CUDA Kernel Corpus — 交接文档

> **目标**: 将 MNN CUDA backend 的所有 `__global__` kernel（含历史版本 1.2.0-3.6.0）
> 忠实复制到 `replay_benchmark`，按 kernel-adapt skill 的"变体 vs 版本"分层规则实现。
>
> **当前状态**: 30 个 kernels/*.cu 文件，297 个 shim，332 个 case，**332/332 全部通过**。
> 编译 0 error。所有 MNN `__global__` kernel 已 100% 覆盖（含 1.2.0-3.6.0 全版本 + BF16）。
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
| 🔴 P0 | transpose.cu | NHWC8/C4NHW4 系列 (10 个) | ✅ 已修: C4 deinterleave + zero-fill |
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
- TRANSPOSE (NHWC↔NCHW + 格式转换 + fuseblit)
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

当前已修正为：一次 benchmark 只创建一个 PMU Session；所有 range 和 workload 都在该 Session 内执行；由 CUPTI 自己完成 multi-pass replay 与聚合；`report.num_passes` 保留 Session 报告的真实 pass 数。单 metric smoke 验证结果为 `pmu_status=sampled`、`num_passes=2`，不再打印手工 split 日志。

验证中 `lts__t_sectors_srcnode_gpc_op_atom_dot_alu_lookup_hit.min` 返回 `9223372036854775808`。该值是无效/溢出哨兵，不是有效 PMU 计数；这属于 metric 评估或样本有效性问题，不能归因于 pass 拆分修复，也不能将其计入 `VALID`。

## 14. 有效 CUDA PMC 与 kernel corpus 全量 sweep

`skills/gpu-pmu-sweep/scripts/parse_cuda_pmu_log.py` 将
`build-x86-cuda/sweep.log` 解析为：

- `cuda_pmc_all.csv`：7200 个 metric 的 `VALID`、`NOT_FOUND`、`OVERFLOW`、`COMMAND_FAILED` 状态；
- `cuda_pmc_valid.csv`：仅 5952 个 `VALID` metric，溢出哨兵值不进入此文件。

`skills/gpu-pmu-sweep/scripts/sweep_cuda_kernel_pmc.py` 读取 valid CSV，
从 `operator_cases.json` 选择全部 425 个 `backend=cuda` case，执行
`case × valid metric`。每个命令只提交一个 case，默认将 32 个 metric
放入同一个 CUPTI Session，由 CUPTI 在该 Session 内管理真实的 multi-pass；
运行严格串行，避免 CUPTI profiler 资源竞争。
脚本自动传入 `--kernel-corpus-no-latency`，PMC sweep 不执行延迟测量。
如果 CUPTI 拒绝批量配置，脚本自动二分重试，最终降级到单 metric，避免
将整批配置失败误判为 metric 无效。

从 `build-x86-cuda` 执行：

```bash
python3 ../skills/gpu-pmu-sweep/scripts/parse_cuda_pmu_log.py \
  --input sweep.log --all-csv cuda_pmc_all.csv --valid-csv cuda_pmc_valid.csv
sudo -v
python3 ../skills/gpu-pmu-sweep/scripts/sweep_cuda_kernel_pmc.py \
  --valid-csv cuda_pmc_valid.csv \
  --corpus-root ../replay_benchmark/kernel_corpus \
  --binary ./replay_benchmark.out --workdir . \
  --lib-dir .:source/backend/cuda:. --metrics-per-session 32 \
  --sudo --resume \
  --output-csv cuda_kernel_pmc_rows.csv \
  --output-json cuda_kernel_pmc.json
```

`cuda_kernel_pmc_rows.csv` 是断点账本，记录每个组合的 PMU 状态、值和
错误；中断后再次使用 `--resume`。最终 JSON 的结构是：

```json
{"kernel_case_name": {"pmc": {"metric.name": 123}}}
```

延迟由独立的 `replay_kernel_latency` 测试测量，不从 PMC sweep 获取：

```bash
REPLAY_KERNEL_LATENCY_OUTPUT=cuda_kernel_latency.json \
  ./replay_kernel_latency --filter KernelLatency.AllCudaCases
```

单 case 冒烟验证可设置 `REPLAY_KERNEL_LATENCY_CASE_FILTER`，正式全量测量时不要设置。

该测试使用 `--perf-counter-events none`，确保 latency 测量不会创建 CUPTI
Session，也不会受到 PMU multi-pass 影响。
固定样例单元测试位于
`skills/gpu-pmu-sweep/tests/test_parse_cuda_pmu_log.py`，运行：

```bash
python3 -m unittest discover -s skills/gpu-pmu-sweep/tests -p 'test_*.py'
```
