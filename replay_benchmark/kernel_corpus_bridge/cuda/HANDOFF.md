# CUDA Kernel Corpus — 交接文档

> **目标**: 将 MNN CUDA backend 的所有 `__global__` kernel 忠实复制到
> `replay_benchmark`，按 kernel-adapt skill 的"变体 vs 版本"分层规则实现。
>
> **当前状态**: 28 个 kernels/*.cu 文件，201 个 adapter 类，261 个 case，**261/261 全部通过**。
> 编译 0 error。P3 bf16 / P11 BF16 pool 因 sm75 环境限制跳过。
>
> **核心原则**: 忠实复制 MNN kernel 源码。adapter 只负责参数转换 + buffer 打包 + 验证。

---

## 1. 文件结构

```
replay_benchmark/kernel_corpus_bridge/cuda/
├── CudaOpAdapter.hpp              ← CudaOpAdapter 接口 + CudaLaunchCtx
├── CudaOps.hpp                    ← 所有 adapter 类声明 + helper (201 个类)
├── CudaOps.cpp                    ← A类 fp32 adapter + registerCudaOps()
├── CudaOpsMisc.cpp                ← B/C/D 类 fp32 adapter
├── CudaOpsFp16.cu                 ← fp16/int8/plugin/raster_fuse adapter (nvcc)
├── CUDA_KERNEL_VERSIONING.md      ← 版本演进文档
├── HANDOFF.md                     ← 本文件
└── kernels/                       ← GPU kernel 定义层 (nvcc, 28 个文件)
    ├── corpus_common.cuh           ← 共享 helper: UP_DIV, DivModFast, blockReduceSum, CUDA_KERNEL_LOOP
    ├── unary_cast.cu               ← RELU/CLAMP/CAST/FLOAT_2_INT8_CAST_PACK 等
    ├── binary.cu                   ← ATAN2/MOD/LOGICALOR
    ├── range.cu                    ← RANGE
    ├── select.cu                   ← SELECT
    ├── softmax.cu                  ← SOFTMAX naive + WARP_32 + AXIS_REDUCE
    ├── layernorm.cu                ← LAYERNORM + layernorm_c4 + binary_layernorm_c4 (含 fp16)
    ├── prelu.cu                    ← PRELU
    ├── scale.cu                    ← SCALE
    ├── pool.cu                     ← MAXPOOL/AVGPOOL/GLOBAL
    ├── gatherv2_argmax.cu          ← GATHERV2 + ARGMAX + 两阶段
    ├── interp.cu                   ← INTERP nearest/bilinear/round + OPT (含 fp16)
    ├── transpose.cu                ← NHWC↔NCHW + TRANSPOSE + PACKCOMMON/UNPACKCOMMON + blit
    ├── gridsample.cu               ← GRID_SAMPLE nearest/bilinear/3D (含 fp16)
    ├── reduction_naive.cu          ← SUM/MEAN/MAX/MIN/PROD + axis 变体 (含 fp16)
    ├── topkv2.cu                   ← TopKAllRows/GetResultAllRows (含 fp16)
    ├── raster.cu                   ← blitRegion + fuseblit/fuseblitLimit + pack/unpack
    ├── convdw.cu                   ← CONV_DW + WeightPrepare/BiasPrepare/BiasZeroPrepare
    ├── convdw_extra.cu             ← CONV_DW_OPT/HALF2_OPT/3x3_HALF2_OPT/MULTI_WIDTH4/MULTI_WIDTH_CHANNEL
    ├── rope.cu                     ← ropeC4Kernel (含 fp16)
    ├── matmul.cu                   ← GENERAL_BATCH_MATMUL + matmul_gemv_kernel
    ├── conv_base.cu                ← Float22Half2 + Im2Col_FilterC + WeightPackFill + PackPadFill
    ├── deconv.cu                   ← DeconvKernelReorder + Col2Im + Col2Im_Vec4
    ├── winograd.cu                 ← WinoWeightReorder + WinoInputTrans/half2 + WinoTrans2Output/half2
    ├── attention.cu                ← flash_decode*/splitk + combine_results + qk/qkv_tiled + conv1d_silu + short_conv* + gated_delta_rule (含 fp16)
    ├── transpose_half.cu           ← PACKCOMMON_half_4 + REARRANGE_half_4 + fuseblit_half_4 + UNARY_HALF2_SIGMOID
    ├── plugins.cu                  ← GroupNorm(Sum/Scale) + SeqLen2Spatial + splitGeLU + SPLIT_FusedQKV
    ├── int8.cu                     ← FLOAT_2_INT8/INT8_2_FLOAT + DequantizeInt8/Int4Weight + CONV_DW_INT8 + Im2Col + BinaryInt8
    └── raster_fuse.cu              ← BinaryADD/MUL + FuseAdd + BinaryMid + BinaryMidLinear4 (4 宏族)
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

## 3. 命名规则

- **变体(variant) = 不同实现策略** → 独立 adapter 类 + 唯一 variant 名 (`cuda_<op>_<strategy>_fp32`)
- **版本(tag) = 同一变体在不同 MNN 版本的函数体更新** → adapter 内部按 `spec.tag` 分流
- 类名: `Cuda<Op><Strategy>Fp32Kernel`（含变体标识，不含 tag）
- 196 个 adapter 类全部通过命名审计

---

## 4. 支持的 tag

```
1.2.0, 1.2.7, 1.2.8, 2.0.2, 2.0.4, 2.1.2, 2.2.2, 2.2.3, 2.4.1, 2.4.2,
2.5.0, 2.5.1, 2.5.3, 2.7.1, 2.7.2, 2.8.0, 2.8.4, 3.6.0
```

---

## 5. 已完成的适配器清单 (109 个)

### 基础 fp32 变体 (30 个, #1-30)

| # | Kernel | variant | 文件 | tag |
|---|--------|---------|------|-----|
| 1 | SUM_REDUCE_AXIS | `cuda_reduction_sum_axis_fp32` | reduction_naive.cu | 2.5.1, 3.6.0 |
| 2 | MEAN_REDUCE_AXIS | `cuda_reduction_mean_axis_fp32` | reduction_naive.cu | 2.5.1, 3.6.0 |
| 3 | RoPE C4 | `cuda_rope_c4_fp32` | rope.cu | 3.6.0 |
| 4 | GENERAL_BATCH_MATMUL | `cuda_general_batch_matmul_fp32` | matmul.cu | 3.6.0 |
| 5 | matmul_gemv | `cuda_matmul_gemv_fp32` | matmul.cu | 3.6.0 |
| 6 | layernorm_c4 | `cuda_layernorm_c4_fp32` | layernorm.cu | 3.6.0 |
| 7 | binary_layernorm_c4 | `cuda_binary_layernorm_c4_fp32` | layernorm.cu | 3.6.0 |
| 8-10 | Float22Half2/Im2Col_FilterC/WeightPackFill | `cuda_*_fp32` | conv_base.cu | 3.6.0 |
| 11-15 | TRANSPOSE/PACKCOMMON/UNPACKCOMMON/blit_2_* | `cuda_*_fp32` | transpose.cu | 3.6.0 |
| 16-17 | fuseblit/fuseblitLimit | `cuda_*_fp32` | raster.cu | 3.6.0 |
| 18-19 | FLOAT_2_INT8/INT8_2_FLOAT_CAST_PACK | `cuda_*_cast_pack_fp32` | unary_cast.cu | 2.5.3, 3.6.0 |
| 20-22 | WeightPrepare/BiasPrepare/BiasZeroPrepare | `cuda_*_prepare_fp32` | convdw.cu | 3.6.0 |
| 23-25 | PACKCOMMON_4/UNPACKCOMMON_4/transpose_BDL_to_BLD | `cuda_*_fp32` | transpose.cu | 3.6.0 |
| 26-28 | DeconvKernelReorder/Col2Im/Col2Im_Vec4 | `cuda_*_fp32` | deconv.cu | 3.6.0 |
| 29-30 | PackPadFill/WeightPackFill_Implicit | `cuda_*_fp32` | conv_base.cu | 3.6.0 |

### A类 fp16/half2 (10 个, #31-40)

| # | Kernel | variant | 文件 |
|---|--------|---------|------|
| 31-35 | CONV_DW_OPT/HALF2_OPT/3x3_HALF2_OPT/MULTI_WIDTH4/MULTI_WIDTH_CHANNEL | `cuda_conv_dw_*_fp32` | convdw_extra.cu |
| 36-40 | PACKCOMMON_half_4/REARRANGE/fuseblit_half_4/UNARY_HALF2_SIGMOID | `cuda_*_fp32` | transpose_half.cu |

### B类 Winograd (6 个, #41-46)

| # | Kernel | variant | 文件 |
|---|--------|---------|------|
| 41-45 | WinoWeightReorder/InputTrans/InputTrans_half2/Trans2Output/Trans2Output_half2 | `cuda_wino_*_fp32` | winograd.cu |
| 46 | Im2Col_FilterC_Vec4 | `cuda_im2col_filterc_vec4_fp32` | conv_base.cu |

### C类 Attention (12 个, #47-58)

| # | Kernel | variant | 文件 |
|---|--------|---------|------|
| 47-58 | flash_decode*/splitk/combine_results/compact_kv/copy_kv/qk_tiled/qkv_tiled/conv1d_silu/short_conv*/gated_delta_rule | `cuda_*_fp32` | attention.cu |

### D类 INTERP_BILINEAR_OPT (1 个, #59)

| # | Kernel | variant | 文件 | 说明 |
|---|--------|---------|------|------|
| 59 | INTERP_BILINEAR_OPT | `cuda_interp_bilinear_opt_fp32` | interp.cu | MNN `if(0)` 禁用，kernel 本身完整 |

### P0 fp16 Attention/LinearAttention/RoPE/TopKV2 (13 个, #60-72)

| # | Kernel | variant | 说明 |
|---|--------|---------|------|
| 60-72 | flash_decode*/copy_kv/qk/qkv/conv1d_silu/short_conv*/gated_delta/rope/topkv2 | `cuda_*_fp16` | T-typed I/O 改 half; float 累加参数保持 float* |

### P1 fp16 Reduction/Interp/GridSample/LayerNormC4 (17 个, #73-89)

| # | Kernel | variant | 说明 |
|---|--------|---------|------|
| 73-79 | SUM/MEAN/MAX/MIN/PROD_naive + SUM/MEAN_axis | `cuda_reduction_*_fp16` | 容差 1e-2~1e-1 |
| 80-83 | INTERP_NEAREST/BILINEAR/ROUND/OPT | `cuda_interp_*_fp16` | floor(x*sw) 无 +0.5 |
| 84-87 | GRID_SAMPLE_NEAREST/BILINEAR/3D×2 | `cuda_grid_sample_*_fp16` | 3D smoke-only |
| 88-89 | layernorm_c4/binary_layernorm_c4 | `cuda_*_fp16` | gamma/beta: float; 容差 1e-1 |

### P2 插件 kernel (8 个, #90-98 → 实际编号见 §5.6)

| Kernel | variant | 文件 | 说明 |
|--------|---------|------|------|
| GroupNorm Sum/Scale | `cuda_groupnorm_nhwc_*_fp16` | plugins.cu | half-only; blockReduceSum 替代 cub |
| SeqLen2Spatial fp32/fp16 | `cuda_seqlen2spatial_*` | plugins.cu | input+bias+residual |
| splitGeLU fp32/fp16 | `cuda_splitgelu_*` | plugins.cu | gelu(R)*L |
| SPLIT_FusedQKV fp32/fp16 | `cuda_split_fusedqkv_*` | plugins.cu | [B,S,H,3,D]→3×[B,S,H,D] |

### P4 int8 (12 个)

| Kernel | variant | 文件 | 说明 |
|--------|---------|------|------|
| FLOAT_2_INT8/INT8_2_FLOAT (per-channel + single) | `cuda_*_packed_fp32` | int8.cu | INT8_PACK=4 |
| DequantizeInt8/Int4Weight | `cuda_dequantize_*_weight_fp16` | int8.cu | 8-byte 向量化 (int4) |
| CONV_DW_INT8/CONV_DW3x3S1_INT8_OPT | `cuda_conv_dw*_int8_fp32` | int8.cu | dp4a; 2-x-per-thread (3x3) |
| Im2Col_packC_16/WeightInt8PackFill | `cuda_*_int8_fp32` | int8.cu | smoke-only (Im2Col) |
| BinaryInt8_ADD/MUL | `cuda_binary_int8_*_fp32` | int8.cu | 逐元素 |

### P5 Raster 融合 (8 个)

| Kernel | variant | 文件 | 宏族 |
|--------|---------|------|------|
| BinaryADD/MUL | `cuda_raster_binary_*_fp32` | raster_fuse.cu | BINARY_FUNC |
| BinaryFuseAddADD/MUL | `cuda_raster_fuseadd_*_fp32` | raster_fuse.cu | BINARY_FUSEADD_FUNC (atomicAdd, warmup_runs=0) |
| BinaryMidADD/MUL | `cuda_raster_binarymid_*_fp32` | raster_fuse.cu | BINARY_FUNC_FLOATMID (DivModFast) |
| BinaryMidLinear4ADD/MUL | `cuda_raster_binarymidlinear4_*_fp32` | raster_fuse.cu | BINARY_FUNC_FLOATMID4 (float4 向量化) |

---

## 6. 任务进度总览

| 任务 | 状态 | 说明 |
|------|------|------|
| 基础 fp32 (#1-59) | ✅ | A/B/C/D 类全部完成 |
| P0 fp16 Attention/RoPE/TopKV2 | ✅ | 14 cases (210/210) |
| P1 fp16 Reduction/Interp/GridSample/LayerNormC4 | ✅ | 17 cases (227/227) |
| P2 插件 kernel | ✅ | 9 cases (236/236) |
| P3 bf16 depthwise/pool | ⏸️ 跳过 | 需 sm80+ (当前 sm75 RTX 2080 Ti) |
| P4 int8 量化/反量化/DW conv | ✅ 第一批 12 个 | 248/248; GEMV/GEMM int4/int8 (~20) 待后续 |
| P5 Raster 融合 | ✅ 代表性 8 个 | 256/256; 剩余 66 个宏实例待后续 |
| §7 忠实性审计 P0-P14 | ✅ 全部完成 | 见 §7 各子节; 261/261 pass |

---

## 7. 忠实性审计报告 — 逐 kernel 行动计划

> 对全部 28 个 corpus kernel 与 MNN 源码逐函数对比。
> 新会话按优先级从高到低逐个修复。

### 7.1 总览

| 优先级 | 文件 | kernel | 状态 | 行动 |
|--------|------|--------|------|------|
| 🔴 P0 | transpose.cu | NHWC8/C4NHW4 系列 (10 个) | ✅ 已修 | §7.2 |
| 🔴 P1 | transpose.cu | PACKCOMMON/UNPACKCOMMON | ✅ 已修 | §7.3 |
| 🔴 P2 | transpose.cu | PACKCOMMON_4/UNPACKCOMMON_4 | ✅ 已修 | §7.4 |
| 🟡 P3 | convdw.cu | CONV_DW | ✅ 已修 | §7.5 |
| 🟡 P4 | plugins.cu | groupNormNHWCSum/Scale | ✅ 记录 | §7.6 |
| 🟡 P5 | int8.cu | Im2Col_packC_16 | ✅ 已修 | §7.7 |
| 🟡 P6 | conv_base.cu | Im2Col_FilterC_Vec4 | ✅ 已修 | §7.8 |
| 🟡 P7 | deconv.cu | Col2Im_Vec4 | ✅ 已修 | §7.9 |
| 🟡 P8 | topkv2.cu | TopKInThread | ✅ 已修 | §7.10 |
| 🟢 P9 | binary.cu | MOD | ✅ 无需改 | §7.11 |
| 🟢 P10 | layernorm.cu | input_layernorm_* | ✅ 已补 | §7.12 |
| 🟢 P11 | pool.cu | BF16 pool | ⏸️ 跳过 | §7.13 |
| 🟢 P12 | transpose_half.cu | PACKCOMMON_half_4 fp32 shim | ✅ 标注 | §7.14 |
| ✅ | 其余 16 个文件 | — | **YES** | 无需修改 |

### 7.2 🔴 transpose.cu — NHWC8/C4NHW4 格式转换 (10 个 kernel)

**问题**: corpus 用单一通用 body `(batch*area+area_idx)*inChannelPack+chnl_idx`，
没有 MNN 的 C4 lane deinterleave (`c4_idx=chnl_idx>>2; cL_idx=chnl_idx&3`)
和 zero-fill (`if(chnl_idx>=channel){output=0;continue;}`)。

**影响**: 对 NC4HW4 张量产生错误结果。

**涉及**: NHWC8_2_NCHW, C4NHW4_2_NCHW, NHWC8_2_NHWC, C4NHW4_2_NHWC,
NHWC_2_NHWC8, NCHW_2_NHWC8, C4NHW4_2_NHWC8, NHWC_2_C4NHW4, NCHW_2_C4NHW4, NHWC8_2_C4NHW4

**MNN 源码**: `source/backend/cuda/execution/Transpose.cu`

**修复**:
1. 读 MNN `Transpose.cu` 中对应 `__global__` 函数体
2. 忠实复制 C4 deinterleave/interleave + zero-fill 边界检查
3. 保留 corpus shim 名和签名不变（adapter 不需改）
4. adapter 测试数据需构造 NC4HW4 布局以触发 C4 路径

### 7.3 🔴 transpose.cu — PACKCOMMON / UNPACKCOMMON

**问题**: corpus 用 `DivModFast` 分解索引替代 MNN 的直接 `i/axisAlign` 算术，签名不同。

**MNN 源码**: `source/backend/cuda/execution/Transpose.cu`

**修复**:
1. 对比 MNN 的 `PACKCOMMON`/`UNPACKCOMMON` 签名和 body
2. 忠实复制 MNN 的索引公式（`i/axisAlign`、`i%inside` 等）
3. 如 MNN 签名更简单，改 shim 签名 + adapter

### 7.4 🔴 transpose.cu — PACKCOMMON_4 / UNPACKCOMMON_4

**问题**:
- `axisAlign` 用 `UP_DIV(axis,8)*8`（PACK_NUMBER=8），MNN 用 `UP_DIV(axis,2)*2`（PACK_NUMBER=4）
- 丢弃 `else { output[dstOffset]={0,0,0,0}; }` zero-fill

**修复**:
1. 改 `axisAlign` 为 `UP_DIV(axis,2)*2`
2. 加回 zero-fill else 分支
3. adapter 的 `c_p` 对齐逻辑同步修改

### 7.5 🟡 convdw.cu — CONV_DW

**问题**:
- 权重布局: corpus `kernel+(oz*kh+fy)*kw+fx` ([oc][kh][kw]) vs MNN `kernel[(fy*kw+fx)*c_p+oz]` ([kh][kw][c_p])
- `fxSta/fySta`: `(int)ceil(-(float)ix/dw)` vs MNN `UP_DIV(-ix,dw)`

**MNN 源码**: `source/backend/cuda/execution/ConvDepthWiseExecution.cu`

**修复**:
1. 忠实复制权重索引 `kernel[(fy*kw+fx)*c_p + oz]`
2. 忠实复制 `UP_DIV` 边界计算
3. adapter 的 kernel buffer 布局改为 [kh][kw][c_p]

### 7.6 🟡 plugins.cu — groupNormNHWCSum / groupNormNHWCScale

**问题**: 用 `blockReduceSum` + `atomicAdd` 替代 MNN 的 `cub::BlockScan`。数学等价但算法不同。

**MNN 源码**: `source/backend/cuda/execution/plugin/GroupNorm/groupNormKernel.cu`

**修复**:
1. 检查 `cub::BlockScan` 是否可在 sm75 + `-fno-exceptions` 下编译
2. 如可: 忠实复制 MNN 的 `GroupSums`/`GroupSumsOp`/`cub::BlockScan`
3. 如不可: 记录"算法等价替代"并标注原因

### 7.7 🟡 int8.cu — Im2Col_packC_16

**问题**: shim 中 `DivModFast d_ow(1), d_oh(1), d_fx(1)` 硬编码为 1（smoke-only）。

**MNN 源码**: `source/backend/cuda/execution/int8/ConvInt8CutlassExecution.cu`

**修复**:
1. shim 中从参数构造正确的 `DivModFast(ow)`、`DivModFast(oh)`、`DivModFast(kw)`
2. shim 签名新增 `ow`/`oh`/`kw` 参数
3. adapter 从 `iw/ih/kw/kh/sw/sh/pw/ph` 计算 `ow/oh`

### 7.8 🟡 conv_base.cu — Im2Col_FilterC_Vec4

**问题**: 只实现 `precision==0/1`，丢弃 `precision==2`(half4→half4 via int64 copy) 和 `precision==3`(bf16)。

**MNN 源码**: `source/backend/cuda/execution/ConvBaseKernel.cu`

**修复**:
1. 忠实复制 `precision==2` 路径（`int64` copy）
2. `precision==3`(bf16) 可跳过（sm75 限制），标注缺失

### 7.9 🟡 deconv.cu — Col2Im_Vec4

**问题**: hard-code fp32 路径，丢弃 `precision` 参数和 half/bf16 输出转换。

**MNN 源码**: `source/backend/cuda/execution/DeconvBaseKernel.cu`

**修复**:
1. 忠实复制 `DATA_CONVERT_COPY` 宏和 `if(precision==2)` 路径
2. shim 签名恢复 `precision` 参数

### 7.10 🟡 topkv2.cu — TopKInThread

**问题**: 丢弃 MNN 的 4x 循环展开（`for(; i+gridDim.x*blockDim.x*3 < numElePerRow; ...)` + `data[4]` + `#pragma unroll`），替换为单元素循环。功能等价但失去 ILP 优化。

**MNN 源码**: `source/backend/cuda/execution/TopKV2Execution.cu`

**修复**:
1. 忠实复制 4x 展开主循环 + `#pragma unroll`
2. 验证逻辑不需改（结果相同）

### 7.11 🟢 binary.cu — MOD

**问题**: corpus 用 `x - x/y`，MNN 当前用 `fmod(x,y)`(float) 或 `x%y`(int)。

**MNN 源码**: `source/backend/cuda/execution/BinaryExecution.cu`

**修复**:
1. 读 MNN `MOD` 当前实现
2. 如用 `fmod`，改 corpus 为 `fmodf(x,y)`
3. 如有 int/float 分支，忠实复制

### 7.12 🟢 layernorm.cu — input_layernorm_* (缺失)

**问题**: 只有通用 `LAYERNORM`，缺 MNN 的 size-specialized 快速 kernel:
`input_layernorm_320/512/1024/2048/adaptive`。

**MNN 源码**: `source/backend/cuda/execution/LayerNormExecution.cu`

**修复**:
1. 忠实复制各 `input_layernorm_*` kernel
2. 每个 kernel 配独立 shim + adapter (`cuda_layernorm_<size>_fp32`)
3. 这些是性能优化变体，数学结果与通用 `LAYERNORM` 相同

### 7.13 🟢 pool.cu — BF16 pool (缺失)

**问题**: 缺 `maxpool_C8_BF16`/`avgpool_C8_BF16`。MNN 在 `#ifdef ENABLE_CUDA_BF16` 下。

**MNN 源码**: `source/backend/cuda/execution/bf16/PoolBf16.cuh`

**修复**:
1. 复制 BF16 pool kernel 到 pool.cu
2. 注意 `#if (__CUDA_ARCH__ >= 800)` 守卫
3. 需 sm80+ 环境验证

### 7.14 🟢 transpose_half.cu — PACKCOMMON_half_4 fp32 shim

**问题**: fp32 shim 实例化 `<float,float>`（标量），MNN 实际用 `<int2*,int2*>`（half2 packed）。

**MNN 源码**: `source/backend/cuda/execution/Transpose.cu` 的 `PACKCOMMON_half_4`

**修复**:
1. 读 MNN `PACKCOMMON_half_4` 实际签名和类型
2. 如 MNN 只实例化 half2 版本，fp32 shim 标注"corpus-only fallback"
3. 或改为忠实复制 MNN 的 half2 packed 版本

### 7.15 ✅ 已忠实文件 (16 个，无需修改)

`unary_cast.cu`, `range.cu`, `select.cu`, `softmax.cu`, `prelu.cu`, `scale.cu`,
`gatherv2_argmax.cu`, `interp.cu`, `gridsample.cu`, `reduction_naive.cu`,
`raster.cu`, `rope.cu`, `matmul.cu`, `winograd.cu`, `attention.cu`,
`convdw_extra.cu`, `raster_fuse.cu`

---

## 8. 覆盖率缺口（MNN 有但 corpus 无）

### 8.1 权重量化 GEMV/GEMM int4/int8 (~20 个)

`GEMV_FpAInt4B/V5/V9/V14/V14_MB`, `GEMM_FpAInt4B/Int8B`, `GEMV_FpAInt8B_V2`,
`CONV_FpAInt4B/Int8B`, `Rearrange_Weight_Int4/Int8`, `Precompute*`, `QuantA`, `DequantAndAcc`, `BiasAndActivation`

**MNN 源码**: `source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu`

### 8.2 Raster 融合剩余实例 (~66 个)

4 个宏族 × 18 种运算 (ADD/SUB/MUL/DIV/...) − 已实现 8 个 = ~66 个。
通过添加 `BINARY_FUNC(SUB, x-y)` 等实例 + 注册 adapter 扩展。

### 8.3 BF16 原生算子 (8 个)

`CONV_DW_BF16`/`CONV_DW_BF162_OPT`/`CONV_DW3x3_BF162_OPT`/`CONV_DW_BF16_MULTI_WIDTH4`/
`WeightTransToBf16`/`BiasTransToBf16`/`maxpool_C8_BF16`/`avgpool_C8_BF16`

需 sm80+ 环境验证。

### 8.4 SPLIT_FusedKV

`plugin/FmhaCommon/FmhaV2CommonExecution.cu:30`。与 SPLIT_FusedQKV 类似（2-way split）。
