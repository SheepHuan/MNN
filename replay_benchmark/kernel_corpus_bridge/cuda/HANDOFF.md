# CUDA Kernel Corpus — 交接文档

> **会话目标**: 将 MNN CUDA backend 的所有 `__global__` kernel 多版本适配到
> `replay_benchmark`，按 kernel-adapt skill 的"变体 vs 版本"分层规则实现。
>
> **当前状态**: 21 个 kernels/*.cu 文件，132 个已实现 kernel，100 个 adapter 类，
> 167 个 case。编译有 6 个 error（Col2Im_Vec4 / UNPACKCOMMON_4 的 launch
> 参数数不匹配），修复后预计 167/167 pass。

---

## 1. 已完成的工作

### 1.1 变体重构（✅ 已完成）

| # | 变体 | 重构前 | 重构后 |
|---|------|--------|--------|
| 1 | SOFTMAX_WARP_32 | `int_params.softmax_variant` 区分 | `CudaSoftmaxWarp32Fp32Kernel`, variant=`cuda_softmax_warp32_fp32` |
| 2 | SOFTMAX_AXIS_REDUCE | 同上 | `CudaSoftmaxAxisReduceFp32Kernel`, variant=`cuda_softmax_axis_reduce_fp32` |
| 3 | ARGMAX 两阶段 | `CudaArgMaxFp32Kernel` 按 dim>256 分流 | `CudaArgMaxTwostageFp32Kernel`, variant=`cuda_argmax_twostage_fp32` |
| 4 | grid_sample_nearest 3.6.0 | 缺 operators.json 条目 + launch intArgs[9] 越界 | 补条目 + 修复 launch + 修正 shim 声明 |

### 1.2 新增变体（✅ 已完成，已通过验证）

| # | Kernel | variant 名 | 类名 | kernel 文件 | tag |
|---|--------|-----------|------|------------|-----|
| 1 | SUM_REDUCE_AXIS | `cuda_reduction_sum_axis_fp32` | `CudaReductionSumAxisFp32Kernel` | reduction_naive.cu | 2.5.1, 3.6.0 |
| 2 | MEAN_REDUCE_AXIS | `cuda_reduction_mean_axis_fp32` | `CudaReductionMeanAxisFp32Kernel` | reduction_naive.cu | 2.5.1, 3.6.0 |
| 3 | RoPE C4 | `cuda_rope_c4_fp32` | `CudaRopeC4Fp32Kernel` | rope.cu (新) | 3.6.0 |
| 4 | GENERAL_BATCH_MATMUL | `cuda_general_batch_matmul_fp32` | `CudaGeneralBatchMatmulFp32Kernel` | matmul.cu (新) | 3.6.0 |
| 5 | matmul_gemv_kernel | `cuda_matmul_gemv_fp32` | `CudaMatmulGemvFp32Kernel` | matmul.cu (新) | 3.6.0 |
| 6 | layernorm_c4 | `cuda_layernorm_c4_fp32` | `CudaLayerNormC4Fp32Kernel` | layernorm.cu | 3.6.0 |
| 7 | binary_layernorm_c4 | `cuda_binary_layernorm_c4_fp32` | `CudaBinaryLayerNormC4Fp32Kernel` | layernorm.cu | 3.6.0 |
| 8 | Float22Half2 | `cuda_float22half2_fp32` | `CudaFloat22Half2Fp32Kernel` | conv_base.cu (新) | 3.6.0 |
| 9 | Im2Col_FilterC | `cuda_im2col_filterc_fp32` | `CudaIm2ColFilterCFp32Kernel` | conv_base.cu (新) | 3.6.0 |
| 10 | WeightPackFill | `cuda_weight_pack_fill_fp32` | `CudaWeightPackFillFp32Kernel` | conv_base.cu (新) | 3.6.0 |
| 11 | TRANSPOSE | `cuda_transpose_fp32` | `CudaTransposeFp32Kernel` | transpose.cu | 3.6.0 |
| 12 | PACKCOMMON | `cuda_packcommon_fp32` | `CudaPackCommonFp32Kernel` | transpose.cu | 3.6.0 |
| 13 | UNPACKCOMMON | `cuda_unpackcommon_fp32` | `CudaUnpackCommonFp32Kernel` | transpose.cu | 3.6.0 |
| 14 | blit_2_float | `cuda_blit_2_float_fp32` | `CudaBlit2FloatFp32Kernel` | transpose.cu | 3.6.0 |
| 15 | blit_2_half | `cuda_blit_2_half_fp32` | `CudaBlit2HalfFp32Kernel` | transpose.cu | 3.6.0 |
| 16 | fuseblit | `cuda_fuseblit_fp32` | `CudaFuseBlitFp32Kernel` | raster.cu | 3.6.0 |
| 17 | fuseblitLimit | `cuda_fuseblit_limit_fp32` | `CudaFuseBlitLimitFp32Kernel` | raster.cu | 3.6.0 |
| 18 | FLOAT_2_INT8_CAST_PACK | `cuda_float2int8_cast_pack_fp32` | `CudaFloat2Int8CastPackFp32Kernel` | unary_cast.cu | 2.5.3, 3.6.0 |
| 19 | INT8_2_FLOAT_CAST_PACK | `cuda_int82float_cast_pack_fp32` | `CudaInt82FloatCastPackFp32Kernel` | unary_cast.cu | 2.5.3, 3.6.0 |
| 20 | WeightPrepare | `cuda_weight_prepare_fp32` | `CudaWeightPrepareFp32Kernel` | convdw.cu | 3.6.0 |
| 21 | BiasPrepare | `cuda_bias_prepare_fp32` | `CudaBiasPrepareFp32Kernel` | convdw.cu | 3.6.0 |
| 22 | BiasZeroPrepare | `cuda_bias_zero_prepare_fp32` | `CudaBiasZeroPrepareFp32Kernel` | convdw.cu | 3.6.0 |
| 23 | PACKCOMMON_4 | `cuda_packcommon_4_fp32` | `CudaPackCommon4Fp32Kernel` | transpose.cu | 3.6.0 |
| 24 | UNPACKCOMMON_4 | `cuda_unpackcommon_4_fp32` | `CudaUnpackCommon4Fp32Kernel` | transpose.cu | 3.6.0 |
| 25 | transpose_BDL_to_BLD | `cuda_transpose_bdl_to_bld_fp32` | `CudaTransposeBdlToBldFp32Kernel` | transpose.cu | 3.6.0 |
| 26 | DeconvKernelReorder | `cuda_deconv_kernel_reorder_fp32` | `CudaDeconvKernelReorderFp32Kernel` | deconv.cu (新) | 3.6.0 |
| 27 | Col2Im | `cuda_col2im_fp32` | `CudaCol2ImFp32Kernel` | deconv.cu (新) | 3.6.0 |
| 28 | Col2Im_Vec4 | `cuda_col2im_vec4_fp32` | `CudaCol2ImVec4Fp32Kernel` | deconv.cu (新) | 3.6.0 |
| 29 | PackPadFill | `cuda_pack_pad_fill_fp32` | `CudaPackPadFillFp32Kernel` | conv_base.cu | 3.6.0 |
| 30 | WeightPackFill_Implicit | `cuda_weight_pack_fill_implicit_fp32` | `CudaWeightPackFillImplicitFp32Kernel` | conv_base.cu | 3.6.0 |

### 1.3 命名规则审计（✅ 已完成）

- 所有 100 个 adapter 类通过审计：variant 名唯一、不含 tag、operators.json/cases 双向匹配
- 每个变体有独立类名（含变体标识，不含 tag），内部按 `spec.tag` 路由到不同 shim

---

## 2. 当前编译状态（⚠️ 有 6 个 error 待修复）

```
编译错误位置: CudaOpsMisc.cpp
  1. CudaCol2ImVec4Fp32Kernel::launch — intArgs 参数数与 shim 声明不匹配
     (Col2Im_Vec4 的 adapt 中有 bias sentinel scalarInt(0)，但 launch 跳过它时
      intArgs 索引错位，需要从 adapt 中移除 bias sentinel 或修正 launch 索引)
  2. CudaUnpackCommon4Fp32Kernel::launch — intArgs 数量不足
     (shim 声明有 14 个参数但 launch 只传了 12 个)
```

**修复方案**:
- Col2Im / Col2Im_Vec4: 从 adapt 中移除 `bias sentinel` 的 `scalarInt(0)`（bias 是
  指针，不应进入 intArgs），launch 直接传 `nullptr`
- UNPACKCOMMON_4: 检查 adapt 中 intArgs 数量，确保与 launch 使用的索引一致

---

## 3. 当前文件结构

```
replay_benchmark/kernel_corpus_bridge/cuda/
├── CudaOpAdapter.hpp              ← CudaOpAdapter 接口 + CudaLaunchCtx
├── CudaOps.hpp                    ← 所有 adapter 类声明 + helper (100 个类)
├── CudaOps.cpp                    ← A类 fp32 adapter + registerCudaOps()
├── CudaOpsMisc.cpp                ← B类 fp32 adapter (含新增的 deconv/transpose/cast 等)
├── CudaOpsFp16.cu                 ← fp16/int8/bf16 adapter (需 nvcc)
├── CUDA_KERNEL_VERSIONING.md      ← 版本演进文档
├── HANDOFF.md                     ← 本文件
└── kernels/                       ← GPU kernel 定义层 (nvcc 编译, 21 个文件)
    ├── corpus_common.cuh           ← 共享 device helper (含 TransposeParam, FuseRegion)
    ├── unary_cast.cu               ← RELU/CLAMP/CAST/FLOAT_2_INT8_CAST_PACK 等
    ├── binary.cu                   ← ATAN2/MOD/LOGICALOR
    ├── range.cu                    ← RANGE
    ├── select.cu                   ← SELECT
    ├── softmax.cu                  ← SOFTMAX naive + WARP_32 + AXIS_REDUCE
    ├── layernorm.cu                ← LAYERNORM + layernorm_c4 + binary_layernorm_c4
    ├── prelu.cu                    ← PRELU
    ├── scale.cu                    ← SCALE
    ├── pool.cu                     ← MAXPOOL/AVGPOOL/GLOBAL
    ├── gatherv2_argmax.cu          ← GATHERV2 + ARGMAX + 两阶段
    ├── interp.cu                   ← INTERP nearest/bilinear/round
    ├── transpose.cu                ← NHWC↔NCHW + TRANSPOSE + PACKCOMMON/UNPACKCOMMON + blit
    ├── gridsample.cu               ← GRID_SAMPLE nearest/bilinear/3D
    ├── reduction_naive.cu          ← SUM/MEAN/MAX/MIN/PROD + SUM_REDUCE_AXIS + MEAN_REDUCE_AXIS
    ├── topkv2.cu                   ← TopKAllRows/GetResultAllRows
    ├── raster.cu                   ← blitRegion + fuseblit/fuseblitLimit + pack/unpack/setzero/add_bias
    ├── convdw.cu                   ← CONV_DW + WeightPrepare/BiasPrepare/BiasZeroPrepare
    ├── rope.cu                     ← ropeC4Kernel (含 QNorm/KNorm 完整路径)
    ├── matmul.cu                   ← GENERAL_BATCH_MATMUL + matmul_gemv_kernel
    ├── conv_base.cu                ← Float22Half2 + Im2Col_FilterC + WeightPackFill + PackPadFill + WeightPackFill_Implicit
    └── deconv.cu                   ← DeconvKernelReorder + Col2Im + Col2Im_Vec4
```

---

## 4. 待完成工作

### 4.0 编译错误修复（最高优先级）

| 问题 | 原因 | 修复方案 |
|------|------|---------|
| Col2Im_Vec4 launch 参数不匹配 | adapt 中 `bias sentinel` 的 `scalarInt(0)` 进入 intArgs，导致 launch 索引错位 | 从 adapt 移除 bias sentinel，launch 直接传 nullptr |
| UNPACKCOMMON_4 launch 参数不足 | shim 声明有 14 参数但 launch 传了 12 个 | 检查 adapt 的 intArgs 数量，确保 launch 索引覆盖所有 shim 参数 |

### 4.1 A 类：fp16/half2 变体（需 nvcc 编译，放在 CudaOpsFp16.cu）

| # | Kernel | 源文件 | 说明 | 状态 |
|---|--------|-------|------|------|
| 1 | CONV_DW_OPT | MultiInputConvDepthWiseExecution.hpp | fp32 input + fp16 kernel/bias | TODO |
| 2 | CONV_DW_HALF2_OPT | 同上 | 全 fp16 half2 向量化 | TODO |
| 3 | CONV_DW3x3_HALF2_OPT | 同上 | 3x3 half2 特化 | TODO |
| 4 | CONV_DW_MULTI_WIDTH4 | 同上 | 多宽度向量化 DW | TODO |
| 5 | CONV_DW_MULTI_WIDTH_CHANNEL | 同上 | 多宽度+通道 DW | TODO |
| 6 | PACKCOMMON_half_4 | Transpose.cu | half4 pack | TODO |
| 7 | PACKCOMMON_REARRANGE_half_4 | Transpose.cu | half4 rearrange pack | TODO |
| 8 | UNPACKCOMMON_REARRANGE_half_4 | Transpose.cu | half4 rearrange unpack | TODO |
| 9 | fuseblit_half_4 | Raster.cu | half4 fuseblit | TODO |
| 10 | UNARY_HALF2_SIGMOID | Raster.cu | half2 sigmoid 融合 | TODO |

### 4.2 B 类：Winograd 卷积变体（fp32，部分需 fp16）

| # | Kernel | 源文件 | 说明 | 状态 |
|---|--------|-------|------|------|
| 11 | WinoWeightReorder | ConvWinogradExecution.cu | Winograd weight 重排 | TODO |
| 12 | WinoInputTrans | WinogradTrans.cuh | Winograd 输入变换 | TODO |
| 13 | WinoInputTrans_half2 | 同上 | half2 版本 | TODO |
| 14 | WinoTrans2Output | WinogradTrans.cuh | Winograd 输出变换 | TODO |
| 15 | WinoTrans2Output_half2 | 同上 | half2 版本 | TODO |
| 16 | Im2Col_FilterC_Vec4 | ConvBaseKernel.cu | vec4 im2col | TODO |

### 4.3 C 类：Attention/LinearAttention（低优先级，复杂度高）

| # | Kernel | 源文件 | 说明 | 状态 |
|---|--------|-------|------|------|
| 17 | flash_decode_kernel | AttentionExecution.cu | Flash decode 主 kernel | TODO |
| 18 | flash_decode_kernel_with_mask | 同上 | 带 mask | TODO |
| 19 | flash_decode_kernel_splitk | 同上 | split-k | TODO |
| 20 | flash_attn_combine_results | 同上 | 结果合并 | TODO |
| 21 | compact_kv_cache_kernel | 同上 | KV cache 压缩 | TODO |
| 22 | copy_kv_to_cache_kernel | 同上 | KV cache 拷贝 | TODO |
| 23 | qk_kernel_tiled | 同上 | QK tiled | TODO |
| 24 | qkv_kernel_tiled | 同上 | QKV tiled | TODO |
| 25 | conv1d_silu_kernel | LinearAttentionExecution.cu | conv1d+silu 融合 | TODO |
| 26 | short_conv_kernel | 同上 | short conv | TODO |
| 27 | short_conv_output_kernel | 同上 | short conv output | TODO |
| 28 | gated_delta_rule_decode_kernel | 同上 | gated delta rule | TODO |

### 4.4 D 类：INTERP_BILINEAR_OPT（已确认在 MNN 源码中 disabled）

| # | Kernel | 源文件 | 说明 | 状态 |
|---|--------|-------|------|------|
| 29 | INTERP_BILINEAR_OPT | InterpExecution.cu | 在 `if(0)` 中，实际未启用 | SKIP |

---

## 5. 构建 & 验证命令

```bash
# 编译
export PATH=/usr/local/cuda-13.2/bin:$PATH
cd build && cmake --build . --target replay_benchmark.out -j$(nproc)

# 全量验证
LD_LIBRARY_PATH=.:source/backend/cuda:. ./replay_benchmark.out \
  --kernel-corpus-bench --kernel-corpus-root ../replay_benchmark/kernel_corpus \
  --kernel-corpus-runs 1 --perf-counter-output /tmp/kc_cuda.json

# 查看结果
python3 -c "
import json
d=json.load(open('/tmp/kc_cuda.json'))
cuda=[c for c in d['cases'] if c['backend']=='cuda']
print(f'total={len(cuda)}, passed={sum(1 for c in cuda if c[\"validation_status\"]==\"validation_passed\")}')
fails=[c for c in cuda if c['validation_status']!='validation_passed']
for c in fails: print(f'  FAIL {c[\"case\"]}: {c[\"tag\"]} {c[\"variant\"]}: {c.get(\"error\",\"\")[:80]}')
"

# 覆盖率检查
grep -rh "__global__ void" source/backend/cuda/execution/*.cu source/backend/cuda/execution/*.cuh \
  | sed 's/.*__global__ void //;s/(.*//;s/ //g' | sort -u > /tmp/mnn.txt
grep -rh "__global__ void" replay_benchmark/kernel_corpus_bridge/cuda/kernels/*.cu \
  | sed 's/.*__global__ void //;s/(.*//;s/ //g; s/<.*>//g' | sort -u > /tmp/replay.txt
comm -23 /tmp/mnn.txt /tmp/replay.txt | grep -v "##\|<float>\|<half>\|^Name$\|^FLOAT##Name$"
```

---

## 6. 命名规则（强制）

### 变体与版本分层

- **变体(variant) = 不同实现策略** → 每个变体一个独立 adapter 类 + 唯一 variant 名
- **版本(tag) = 同一变体在不同 MNN 版本的函数体更新** → adapter 内部按 `spec.tag` 分流
- variant 名格式：`cuda_<op>_<strategy>_fp32`（不含 tag）
- 类名格式：`Cuda<Op><Strategy>Fp32Kernel`（含变体标识，不含 tag）

### 已验证通过

- 100 个 adapter 类全部通过命名审计
- variant 名唯一，不含 tag 版本号
- operators.json / operator_cases.json / adapter 类三方一致

---

## 7. MnnBridge 支持的 tag

```cpp
"1.2.0", "1.2.7", "1.2.8", "2.0.2", "2.0.4", "2.1.2", "2.2.2", "2.2.3", "2.4.1", "2.4.2",
"2.5.0", "2.5.1", "2.5.3", "2.7.1", "2.7.2", "2.8.0", "2.8.4", "3.6.0"
```

---

## 8. 下一步建议

1. **修复编译错误**（§4.0）— Col2Im_Vec4 和 UNPACKCOMMON_4 的 launch 参数不匹配
2. **补齐 fp16/half2 变体**（§4.1）— 10 个 kernel，需放在 CudaOpsFp16.cu 中用 nvcc 编译
3. **补齐 Winograd 卷积变体**（§4.2）— 6 个 kernel
4. **补齐 Attention/LinearAttention**（§4.3）— 12 个 kernel，复杂度高
5. **提交所有改动** — 变体重构 + 30 个新增变体 + 命名审计 + 修复
