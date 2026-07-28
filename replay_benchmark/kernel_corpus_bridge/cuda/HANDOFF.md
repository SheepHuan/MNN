# CUDA Kernel 多版本适配 — 交接文档

> **会话目标**: 将 MNN 1.2.0→3.6.0 之间所有 CUDA kernel 的函数体变更点适配到
> `replay_benchmark` 的 kernel corpus,按 kernel-adapt skill 规范为每个 body-diff
> 版本写独立 kernel + shim + adapter 分支。
>
> **当前状态**: 部分完成,编译中断(重复定义+括号不匹配)。需要先修复编译,
> 再继续剩余 kernel 的适配,最后做文件拆分重构。

---

## 1. 已提交的 commits(按时间顺序)

| Commit | 说明 |
|--------|------|
| `a4d0e37dc` | [GPU:Bugfix] 修复 CUDA adapter 编译 + 1.2.0/3.6.0 双版本 89/89 validated |
| `43be496d2` | [Infra:Feature] CUPTI Range Profiler 后端 |
| `508e7c6d3` | [Doc:Feature] benchmark-model skill + kernel-adapt CUDA spec |
| `0e3ad473d` | [GPU:Feature] per-tag block/grid 策略 + CONV_DW 1.2.0/2.0.4 多版本 + bridge 支持 |
| `1dbb7a657` | [Doc:Feature] 全量 kernel body-diff 扫描 1.0.0→3.6.0 + skill 规则:函数体不一致就独立 shim |
| `185fc830c` | [GPU:Feature] LAYERNORM 1.2.7/2.2.2/2.8.4 + PRELU 1.2.7/1.2.8/2.0.4 多版本 |

## 2. 未提交的改动(工作区脏,4 个文件)

```
M  replay_benchmark/kernel_corpus/operator_cases.json   (+322 行,新增 19 个 case)
M  replay_benchmark/kernel_corpus/operators.json        (+209 行,新增 19 个 operator 条目)
M  replay_benchmark/kernel_corpus_bridge/cuda/CorpusKernels.cu       (+295 行,新增中间版本 kernel+shim)
M  replay_benchmark/kernel_corpus_bridge/cuda/CorpusKernelsMisc.cu   (+141 行,新增 blitRegion/NCHW/GridSample 2.7.2)
```

### 2.1 CorpusKernels.cu 新增内容(已编译通过)
- `SCALE_127` kernel + `mnn_corpus_scale_127_fp32` shim
- `CLAMP_127` (复用 CLAMP<float>) + `mnn_corpus_clamp_127_fp32` shim
- `ARGMAX_127` / `ARGMAX_128` / `ARGMAX_250` kernel + shim
- `ReduceParam_127` struct + `SUM_127`/`MEAN_127`/`MAXIMUM_127`/`MINIMUM_127`/`PROD_127` kernel + shim
- `SELECT_272` kernel + shim
- `SOFTMAX_222` kernel + shim
- `INTERP_NERAEST_127` / `INTERP_BILINEAR_127` kernel + shim

### 2.2 CorpusKernelsMisc.cu 新增内容(**编译失败**)
- `blitRegion_241` kernel + shim
- `NCHW_2_NHWC_212` kernel + shim
- `GRID_SAMPLE_NEAREST_272` / `GRID_SAMPLE_BILINEAR_272` kernel + shim

**编译错误**: `At end of source: error: expected a "}`
原因:之前删除重复定义块时,`} // namespace Corpus` / `} // namespace MNN` 的括号被误删。
需要检查 CorpusKernelsMisc.cu 的 namespace/extern "C" 闭合。

## 3. 当前编译状态

```
replay_cuda_corpus 静态库: 编译失败 (CorpusKernelsMisc.cu 括号不匹配)
replay_benchmark.out: 链接失败 (依赖 replay_cuda_corpus)
```

**修复步骤**:
1. 检查 `CorpusKernelsMisc.cu` 的 `namespace MNN { namespace Corpus {` 与 `} // namespace` 闭合
2. 检查 `extern "C" {` 与 `} // extern "C"` 闭合
3. 确认没有重复的 kernel 定义(blitRegion_241 等只定义一次)
4. 编译通过后,`git stash` 或单独提交 kernel+shim 部分,再继续 adapter 改动

## 4. 已完成的 kernel 多版本适配

| Kernel | 已适配的 tag | 状态 |
|--------|-------------|------|
| CONV_DW | 1.2.0, 2.0.4, 3.6.0 | ✅ validated |
| LAYERNORM | 1.2.0, 1.2.7, 2.2.2, 2.8.4, 3.6.0 | ✅ validated |
| PRELU | 1.2.0, 1.2.7, 1.2.8, 2.0.4, 3.6.0 | ✅ validated |
| Reduction 2.5.1 | 2.5.1 (SUM/MEAN/MAX/MIN/PROD) | ✅ validated (共用 3.6.0 shim,函数体一致) |
| INTERP 2.0.4 | 2.0.4 (nearest/bilinear) | ✅ validated (共用 3.6.0 shim,函数体一致) |

## 5. 待完成的 kernel 适配

以下 kernel 的 **kernel + shim 已写入 .cu 文件,但 adapter(CudaOps.cpp/CudaOpsMisc.cpp)还没加 tag 分支**:

| Kernel | 新增 tag | kernel+shim 位置 | adapter 位置 | 备注 |
|--------|---------|-----------------|-------------|------|
| SCALE | 1.2.7 | CorpusKernels.cu | CudaOps.cpp `CudaScaleFp32Kernel` | PACK_NUMBER 打包 |
| CLAMP | 1.2.7 | CorpusKernels.cu | CudaOps.cpp `CudaClampFp32Kernel` | 模板化,fp32 可共用 |
| ARGMAX | 1.2.7, 1.2.8, 2.1.2, 2.5.0 | CorpusKernels.cu | CudaOpsMisc.cpp `CudaArgMaxFp32Kernel` | 1.2.7=T*输出, 1.2.8=int*, 2.5.0=指针重写 |
| Reduction | 1.2.7 | CorpusKernels.cu | CudaOpsMisc.cpp REDUCTION_ADAPTER 宏 | ReduceParam 结构体 |
| SELECT | 2.7.2 | CorpusKernels.cu | CudaOps.cpp `CudaSelectFp32Kernel` | 无 stride |
| SOFTMAX | 2.2.2 | CorpusKernels.cu | CudaOps.cpp `CudaSoftmaxFp32Kernel` | ReduceParam |
| INTERP | 1.2.7 | CorpusKernels.cu | CudaOpsMisc.cpp `CudaInterpNearest/BilinearFp32Kernel` | PACK_NUMBER |
| blitRegion | 2.4.1 | CorpusKernelsMisc.cu | CudaOpsMisc.cpp `CudaBlitRegionFp32Kernel` | 无 count |
| NCHW_2_NHWC | 2.1.2 | CorpusKernelsMisc.cu | CudaOpsMisc.cpp `CudaNhwc2NchwFp32Kernel` | channel 而非 inChannelPack |
| GRID_SAMPLE | 2.7.2 | CorpusKernelsMisc.cu | CudaOpsMisc.cpp `CudaGridSampleNearest/BilinearFp32Kernel` | output[index] |

### 5.1 每个待完成 kernel 的适配步骤(模板)

参照 CONV_DW/LAYERNORM/PRELU 的已完成模式:

1. **adapter adapt()**: 加 `else if (spec.tag == "x.y.z")` 分支,设 `ac.entry = "mnn_corpus_xxx_tag_fp32"`,
   按 tag 版本的参数布局设 args/buffers
2. **adapter launch()**: 加 `if (ac.tag == "x.y.z")` 分支,调用对应 shim,传参顺序与 shim 签名匹配
3. **adapter validate()**: 加 `if (ac.tag == "x.y.z")` 分支,按 tag 版本的公式计算 expected 值
4. **extern "C" 声明**: 在 .cpp 顶部 extern "C" 块加 shim 前向声明
5. **block/grid**: 1.2.0 用 `mnnBlock120()`,1.2.1+ 用 `kBlock=128`
6. **验证**: `cmake --build && ./replay_benchmark.out --kernel-corpus-case <name> --kernel-corpus-runs 1`

## 6. operator_cases.json 新增的 case(19 个,已写入)

```
cuda_scale_127_fp32_smoke          (scale, 1.2.7)
cuda_clamp_127_fp32_smoke          (clamp, 1.2.7)
cuda_argmax_127_fp32_smoke         (argmax, 1.2.7)
cuda_argmax_128_fp32_smoke         (argmax, 1.2.8)
cuda_argmax_212_fp32_smoke         (argmax, 2.1.2)
cuda_argmax_250_fp32_smoke         (argmax, 2.5.0)
cuda_reduction_sum_127_fp32_smoke  (reduction, 1.2.7)
cuda_reduction_mean_127_fp32_smoke (reduction, 1.2.7)
cuda_reduction_max_127_fp32_smoke  (reduction, 1.2.7)
cuda_reduction_min_127_fp32_smoke  (reduction, 1.2.7)
cuda_reduction_prod_127_fp32_smoke (reduction, 1.2.7)
cuda_select_272_fp32_smoke         (select, 2.7.2)
cuda_softmax_222_fp32_smoke        (softmax, 2.2.2)
cuda_interp_nearest_127_fp32_smoke (interp_nearest, 1.2.7)
cuda_interp_bilinear_127_fp32_smoke(interp_bilinear, 1.2.7)
cuda_blitregion_241_fp32_smoke     (raster, 2.4.1)
cuda_nhwc2nchw_212_fp32_smoke      (transpose, 2.1.2)
cuda_grid_sample_nearest_272_fp32_smoke   (grid_sample, 2.7.2)
cuda_grid_sample_bilinear_272_fp32_smoke  (grid_sample_bilinear, 2.7.2)
```

## 7. 文件拆分计划(用户要求)

当前 6 个文件共 6704 行,用户要求按 kernel 类型拆分到子文件夹:

```
replay_benchmark/kernel_corpus_bridge/cuda/
├── CudaOpAdapter.hpp          (保留,51 行)
├── CudaOps.hpp                (保留,共享声明,637 行)
├── kernels/                   (新,.cu 文件,按算子分)
│   ├── unary.cu               (RELU/CLAMP 各版本)
│   ├── cast.cu                (CAST 各版本)
│   ├── binary.cu              (ATAN2/MOD/LOGICALOR)
│   ├── reduction.cu           (SUM/MEAN/MAX/MIN/PROD 各版本)
│   ├── softmax.cu             (SOFTMAX 各版本)
│   ├── layernorm.cu           (LAYERNORM 各版本)
│   ├── prelu.cu               (PRELU 各版本)
│   ├── scale.cu               (SCALE 各版本)
│   ├── pool.cu                (maxpool/avgpool/global 各版本)
│   ├── interp.cu              (INTERP nearest/bilinear/round 各版本)
│   ├── transpose.cu           (NCHW_2_NHWC/NHWC_2_NCHW 各版本)
│   ├── gridsample.cu          (GRID_SAMPLE nearest/bilinear/3D 各版本)
│   ├── raster.cu              (blitRegion/pack_c4/unpack_c4/setzero/add_bias)
│   ├── convdw.cu              (CONV_DW 各版本)
│   ├── gatherv2.cu            (GATHERV2)
│   ├── argmax.cu              (ARGMAX/ARGMIN 各版本)
│   ├── select.cu              (SELECT 各版本)
│   ├── range.cu               (RANGE)
│   ├── topkv2.cu              (TopKAllRows/GetResultAllRows)
│   └── corpus_common.cuh      (共享: DivModFast, ReduceParam, blockReduceSum 等)
├── adapters/                  (新,.cpp 文件,按算子分)
│   ├── unary.cpp
│   ├── cast.cpp
│   ├── ... (同上分类)
│   └── fp16.cu                (fp16/int8 adapter,需 nvcc)
└── CMakeLists 片段更新         (cuda_add_library 列出所有新 .cu)
```

拆分规则:
- 每个 `kernels/<op>.cu` 放 `namespace MNN::Corpus { __global__ ... }` + `extern "C" { shim }`
- 每个 `adapters/<op>.cpp` 放 `namespace MNN::Replay::KernelCorpus::MnnOps { adapt/launch/validate }`
- 共享的 device helper(DivModFast, warpReduceSum 等)放 `kernels/corpus_common.cuh`
- `CudaOps.hpp` 保留共享声明(inline helper, 类声明, extern "C" 声明)

## 8. 关键 skill 规则(已写入 skills/kernel-adapt/SKILL.md)

1. **函数体不一致就独立 shim**: 不仅看签名,更看公式/索引/累积方式。用 `git diff <tagA> <tagB> -- <file>.cu` 对比 `__global__` 函数体。仅改名(函数体一致)可共用 shim。
2. **block/grid 按版本分流**: 1.2.0 用 `mnnBlock120()` 自适应,1.2.1+ 固定 `kBlock=128`。`CudaLaunchCtx` 的 `grid`/`block` 无默认值,由 `ac.globalSize[0]`/`ac.localSize[0]` 设置。
3. **类名不含版本号**: 同一算子的所有 tag 共用一个 adapter 类,内部按 `spec.tag`/`ac.tag` 分流。
4. **shim 名可含 tag**: 如 `mnn_corpus_layernorm_127_fp32`。
5. **shim 声明位置**: 在 adapter `.cpp` 文件顶部的 `extern "C" {}` 块里(或在统一头文件中)。

## 9. 版本演进文档

`replay_benchmark/kernel_corpus_bridge/cuda/CUDA_KERNEL_VERSIONING.md` 记录了:
- 全量扫描结果(1.0.0→3.6.0 所有 tag 的 `__global__` 函数体 diff)
- 需要独立 shim 的变更点表
- 新增 kernel 列表
- 仅改名/稳定的 kernel 列表

## 10. MnnBridge 支持的 tag

`replay_benchmark/kernel_corpus_bridge/mnn/MnnBridge.cpp` 的 `supports()` 已支持:
```cpp
"1.2.0", "1.2.7", "1.2.8", "2.0.2", "2.0.4", "2.1.2", "2.2.2", "2.2.3", "2.4.2",
"2.5.1", "2.5.3", "2.7.1", "2.7.2", "2.8.0", "2.8.4", "3.6.0"
```

## 11. 下一步建议

1. **修复 CorpusKernelsMisc.cu 编译** — 检查 namespace/extern "C" 括号闭合
2. **完成 10 个待适配 kernel 的 adapter 分支** — 按 §5.1 模板逐个做
3. **全量验证** — `./replay_benchmark.out --kernel-corpus-bench --kernel-corpus-runs 1`,确认 123 个 case 全 pass
4. **提交** — `[GPU:Feature] Add intermediate version kernels (SCALE/ARGMAX/Reduction/SELECT/SOFTMAX/INTERP/blitRegion/NCHW/GridSample/CLAMP)`
5. **文件拆分重构** — 按 §7 计划拆分到 `kernels/` 和 `adapters/` 子文件夹

## 12. 构建 & 验证命令

```bash
# 编译
export PATH=/usr/local/cuda-13.2/bin:$PATH
cd build && cmake --build . --target replay_benchmark.out -j$(nproc)

# 单 case 验证
LD_LIBRARY_PATH=.:source/backend/cuda:. ./replay_benchmark.out \
  --kernel-corpus-bench --kernel-corpus-root ../replay_benchmark/kernel_corpus \
  --kernel-corpus-case <case_name> --kernel-corpus-runs 1

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
for c in fails: print(f'  FAIL {c[\"tag\"]} {c[\"operator\"]}/{c[\"variant\"]}: {c.get(\"error\",\"\")[:80]}')
"
```
