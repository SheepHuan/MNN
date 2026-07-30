---
name: kernel-adapt
description: 将历史 MNN/ncnn 的 OpenCL/Vulkan/CUDA kernel 适配到当前 MNN GPU runtime 运行。覆盖 kernel 提取、baked 文件生成、版本桥接器(OpAdapter)实现、case 注册、设备验证的完整 TDD 流程。
---

# 历史 GPU Kernel 适配 SKILL

> **触发条件**：当用户请求适配/运行历史 kernel、扩展算子变体覆盖、为 kernel corpus 新增 adapter 时触发。

## 核心原则（强制）

> **忠实复制 MNN 的 kernel 源码。** adapter 只负责参数转换 + buffer 打包 + 验证。
>
> 1. **kernel 实现 = 原样复制 MNN 源码的 `__global__` 函数体**——不简化、不替换逻辑、不改变算法。
>    如果 MNN 用 `cub::BlockScan`，corpus 也用 `cub::BlockScan`；如果 MNN 用 8-byte 向量化，corpus 也用 8-byte 向量化。
> 2. **adapter = 参数转换层**——从 `CaseSpec` 读参数，构造 input buffer（填充测试数据），设置 grid/block，
>    按 shim 签名打包 `ac.args`，调 shim launch。adapter 不包含任何 kernel 逻辑。
> 3. **验证 = adapter 在 host 端重新计算 expected 值**，与 kernel 输出对比。验证容差按数据类型调整
>    （fp32: 1e-3~1e-2，fp16: 1e-1~1e-2，int8: 精确匹配）。验证逻辑不依赖 kernel 内部实现细节。
>
> **违反此原则的典型错误**：简化 kernel 逻辑导致 corpus 与 MNN 行为不一致、验证逻辑与 kernel
> 内部数据布局不匹配导致 false negative。

## 概述

本 SKILL 指导 AI Agent 将 `replay_benchmark/kernel_corpus/sources/` 中的历史 MNN OpenCL `.cl` 和 ncnn Vulkan `.comp` kernel 适配为可在当前 MNN GPU runtime 上编译、dispatch、校验和采集 PMU 的独立 workload。

**不编译历史 backend，不链接历史 C++，不复用历史 allocator/scheduler。** 历史源码只作为 kernel 文本来源。

**bake 不跳过任何 kernel。** bake 的含义是：把历史 kernel 源码原样保留，只补齐编译期宏定义（FLOAT、GLOBAL_SIZE_DIMS、sfp/afp 等），使其单文件可独立编译。含 `image2d_t` 的 kernel 也保留——runner 遇到不支持的能力时标记 `unsupported`，而不是 bake 时丢掉。

## 核心架构

```
operator_cases.json (声明 case 语义)
        ↓
Bridge::adapt() → AdaptedCase (完全准备好的 args/buffers/dispatch/compileMacros)
        ↓
Runner (通用,只认 AdaptedCase)
  OpenCL: clBuildProgram(macros) → setArg → dispatch → readback → PMU
  Vulkan: SPIR-V → createPipeline → bindDescriptorSet → pushConstants → dispatch → readback → PMU
```

**分层**：
1. **OpAdapter**（算子级）：每个 `(op_type, variant)` 一个 adapter，知道参数顺序、数据填充、dispatch geometry、validator。tag 差异在 adapter 内部分支。
2. **FrameworkBridge**（框架级）：填 metadata + 加载 SPIR-V，按 `(opType, variant)` 找 OpAdapter 委托。
3. **Runner**（通用）：只消费 `AdaptedCase`，编译/dispatch/readback/PMU，零算子知识。

## 目录结构

```
kernel_corpus/
  operators/                          # baked 独立 kernel 文件
    opencl/<op_type>/mnn/<tag>/<variant>.cl
    vulkan/<op_type>/mnn/<tag>/<variant>.comp (+ .spv)
    vulkan/<op_type>/ncnn/<tag>/<variant>.comp (+ .spv)
  operators.json                      # 索引 (backend/op_type/framework/tag/variant/entry/file)
  operator_cases.json                 # 可执行 case
  bake_ncnn_shaders.py                # 自动 bake ncnn .comp
  bake_mnn_kernels.py                 # 自动 bake MNN .cl (OpenCL)
  bake_mnn_vulkan_shaders.py           # 自动 bake MNN .comp + 预编译 .spv (Vulkan)
  extract_operator_kernels.py         # 扫描 operators/ 生成 operators.json
  gen_adapter_scaffold.py             # 分析 kernel 签名，归类参数布局

kernel_corpus_bridge/
  Bridge.hpp                          # AdaptedCase + Bridge 接口 + validators
  Bridge.cpp                          # 注册表 + validator 实现
  OpAdapter.hpp                       # OpAdapter 接口 + FallbackAdapter + findAdapter
  OpAdapter.cpp                       # FallbackAdapter 实现
  mnn/
    MnnBridge.hpp/.cpp                # 顶层分发 + 注册 Kernel adapter + 加载 .spv
    ops/
      RasterOp.hpp/.cpp               # OpenCLRasterSetZeroKernel / VulkanRasterBlitC4Kernel
      UnaryOp.hpp/.cpp                # OpenCLUnaryExpKernel / VulkanUnaryExpKernel
      MatmulOp.hpp/.cpp               # OpenCLMatmulKernel
      ReductionOp.hpp/.cpp            # OpenCLReductionSumKernel / VulkanReductionSumKernel
      PoolingOp.hpp/.cpp              # OpenCLPoolingMaxKernel / VulkanPoolingMaxKernel / VulkanPoolingAvgKernel
      BinaryOp.hpp/.cpp               # OpenCLBinaryAddKernel / VulkanBinaryAddKernel
      ElementwiseOps.hpp/.cpp         # OpenCLCastKernel / OpenCLSelectKernel / ...
      NormConvOps.hpp/.cpp            # OpenCLScaleKernel / OpenCLGroupnormKernel / ...
      ConvOps.hpp/.cpp                # OpenCLConv2dKernel / OpenCLDepthwiseConv2dKernel / ...
      ArgmaxOp.hpp/.cpp               # OpenCLArgmaxKernel
      ComplexOps.hpp/.cpp             # OpenCLAttentionKernel / OpenCLSelfAttentionKernel / ...
  ncnn/
    NcnnBridge.hpp/.cpp
    ops/
      ElementwiseOp.hpp/.cpp          # VulkanSigmoidKernel / VulkanTanhKernel / VulkanPermuteKernel
      Pack4Op.hpp/.cpp                # VulkanSigmoidPack4Kernel / VulkanConcatKernel
      NcnnElementwiseOps.hpp/.cpp     # VulkanClipKernel / VulkanCeluKernel / ...
      NcnnNormOps.hpp/.cpp            # VulkanGroupnormCoeffsKernel / VulkanLayernormKernel / ...
      NcnnShapeOps.hpp/.cpp           # VulkanBatchnormKernel / VulkanReshapeKernel / ...
```

## 步骤 1: Bake Kernel 源码

### bake 的原则

1. **原样保留 kernel 源码**——不简化、不 fallback、不跳过任何 entry
2. **只补齐编译期宏定义**——使单文件可独立编译
3. **含 image2d_t 的 kernel 也保留**——runner 遇到不支持的能力时标记 `unsupported`
4. **cooperative-matrix-only shader 保留原样**——不篡改源码，设备不支持时标记 `unsupported`
5. **MNN Vulkan uniform→push_constant 转换**：MNN shader 用 `layout(set=0,binding=N) uniform constBuffer`，但 replay runner 用 push constant 传参。bake 时自动把 `uniform constBuffer` 声明替换为 `layout(push_constant) uniform constBuffer`，使 SPIR-V 与 runner 的 `vkCmdPushConstants` 路径匹配。push constant 上限 128 字节，MNN shader 的 constBuffer 都远小于此。

### ncnn Vulkan (.comp)

```bash
python3 replay_benchmark/kernel_corpus/bake_ncnn_shaders.py \
  --sources replay_benchmark/kernel_corpus/sources \
  --operators replay_benchmark/kernel_corpus/operators
```

脚本自动：
- `#undef` 所有 NCNN_fp16_*/NCNN_int8_*/ncnn_VK_* 宏（预处理器自然走 FP32 `#else` 分支）
- 内联 `#include "vulkan_activation.comp"` 内容
- 预置 FP32 macro preamble（sfp/afp/buffer_ld4/st4/psc + int8 宏 + int8 extension）
- 对含 subgroup op 的文件加 `#extension GL_KHR_shader_subgroup`
- 跳过 `vulkan_activation.comp`（include 头不是 compute shader，不单独 bake）

验证编译：
```bash
cd replay_benchmark/kernel_corpus/operators/vulkan
for f in $(find . -name "*.comp"); do
  glslangValidator -V --target-env vulkan1.1 --entry-point main "$f" -o /dev/null 2>/dev/null && echo "OK $f" || echo "FAIL $f"
done
```

### MNN Vulkan (.comp)

```bash
python3 replay_benchmark/kernel_corpus/bake_mnn_vulkan_shaders.py \
  --sources replay_benchmark/kernel_corpus/sources \
  --operators replay_benchmark/kernel_corpus/operators \
  --tags 3.6.0
```

脚本自动：
- 从 `sources/mnn/<tag>/source/backend/vulkan/buffer/execution/glsl/` 读取 `.comp`
- 预置 FP32 preamble（`#version 450` + `#define FLOAT float` / `#define FLOAT4 vec4` 等）
- 算子选择宏（`EXP`/`ADD`/`C4`/`SUM` 等）在 bake 时通过 `-D` 固化进 SPIR-V——**Vulkan runner 用预编译 .spv，无法在运行时传 `-D`**
- 调用 `glslangValidator` 预编译 `.spv`，输出到 `operators/vulkan/<op_type>/mnn/<tag>/<variant>.comp` + `.spv`
- **`uniform constBuffer` → `layout(push_constant) uniform constBuffer`**：MNN shader 用 UBO binding 声明常量，但 replay runner 用 `vkCmdPushConstants` 传参。bake 时自动替换声明，使 SPIR-V 与 runner 路径匹配。push constant 上限 128B，MNN shader 都远小于此。
- 输出到 `operators/vulkan/<op_type>/mnn/<tag>/<variant>.comp`

**OP_MAP 结构**：`{op_type: [(source_file, variant_stem, [macros]), ...]}`
- 同一 `op_type` 可有多个 variant（如 `pooling` 有 `vulkan_maxpool_fp32` 和 `vulkan_avgpool_fp32`），放在同一 key 下确保目录一致
- variant 名用 `vulkan_` 前缀，与 OpenCL variant 区分（`findAdapter` 按 `(opType, variant)` 匹配）

**Vulkan binding 顺序注意**：shader 的 `binding=N` 声明顺序与 `ac.buffers` 数组顺序不一定一致。`ac.vulkanBindings[i]` 表示 `buffers[i]` 对应的 binding 号。必须按 shader 声明的 binding 顺序映射，否则 kernel 写入的是 input buffer、读出的是 output buffer，output 全 0。例如 unary.comp 声明 `binding=0 output, binding=1 input`，则 `vulkanBindings = {1, 0}`（`buffers[0]=input→binding=1`，`buffers[1]=output→binding=0`）。

### MNN OpenCL (.cl)

```bash
python3 replay_benchmark/kernel_corpus/bake_mnn_kernels.py \
  --sources replay_benchmark/kernel_corpus/sources \
  --operators replay_benchmark/kernel_corpus/operators
```

处理所有 `*_buf.cl` / `*_subgroup_buf.cl`（含 image2d_t entry 的也保留）。preamble 包含：

| 宏/类型 | 定义 |
|---------|------|
| `FLOAT`/`FLOAT2`/`FLOAT4`/`FLOAT8`/`FLOAT16` | `float`/`float2`/`float4`/`float8`/`float16` |
| `INPUT_TYPE`/`OUTPUT_TYPE` | `float` |
| `INPUT_TYPE4`/`OUTPUT_TYPE4`/`INPUT_TYPE8`/`OUTPUT_TYPE8`/`INPUT_TYPE16`/`OUTPUT_TYPE16` | 对应 floatN |
| `COMPUTE_FLOAT`/`COMPUTE_FLOAT2`/.../`COMPUTE_FLOAT16` | 对应 floatN |
| `CONVERT_FLOAT4(x)`/`CONVERT_OUTPUT4(x)`/`CONVERT_INPUT4(x)`/... | `((floatN)(x))` |
| `CONVERT_COMPUTE_FLOAT4(x)` 等 | `((floatN)(x))` |
| `GLOBAL_SIZE_2_DIMS`/`GLOBAL_SIZE_3_DIMS`/`GLOBAL_SIZE_DIM2` | 展开为 `__private const int` 参数 |
| `DEAL_NON_UNIFORM_DIM2`/`DIM3` | boundary check 宏 |
| `RI_F`/`WI_F`/`SAMPLER` | image read/write 宏（保留 image entry 可编译） |
| `OPERATOR` | `in`（unary 默认，专用 adapter 覆盖） |
| `OPERATE` | `(num+in)`（reduction 默认） |

**注意**：OpenCL `-D` 不支持函数式宏（如 `-DCONVERT_FLOAT4(x)=...`），所有函数式宏必须直接 `#define` 在 preamble 里。

## 步骤 2: 重新生成 operators.json

```bash
python3 replay_benchmark/kernel_corpus/extract_operator_kernels.py \
  --root replay_benchmark/kernel_corpus \
  --output replay_benchmark/kernel_corpus/operators.json
```

## 步骤 3: 实现 Kernel adapter（强制：逐个变体写专用 adapter）

> **强制规则**：每个 `(op_type, variant)` 必须写专用 adapter，精确控制每个参数的类型、值和顺序。**禁止使用 GenericBufAdapter 自动解析签名的方式**——它低效且容易出错，每个都需要反复调试。

### 命名规则：Op vs Kernel

**核心概念区分**：
- **Op** = 一种计算语义（如 `unary`、`binary`、`pooling`）。一个 Op 只有一个逻辑计算。
- **Kernel** = Op 的具体变体实现（如 `unary_buf_exp_fp32`、`vulkan_unary_buf_exp_fp32`）。一个 Op 有多个 Kernel。

**类命名规则**（强制）：

| 层级 | 命名格式 | 示例 | 说明 |
|------|---------|------|------|
| 基类（接口） | `<Op>KernelBase` | `UnaryKernelBase` | 纯虚接口，`adapt() = 0`，**不提供任何共享实现** |
| OpenCL 子类 | `OpenCL<Op><Variant>Kernel` | `OpenCLUnaryExpKernel` | 完整实现 `adapt()`，含数据准备 + OpenCL 参数布局 |
| Vulkan 子类 | `Vulkan<Op><Variant>Kernel` | `VulkanUnaryExpKernel` | 完整实现 `adapt()`，含数据准备 + Vulkan 参数布局 |

**基类设计原则**：
- 基类只声明 `virtual bool adapt(...) = 0`，**不提供共享 `adapt()` 函数体**
- 每个 backend 子类**各自完整实现** `adapt()`——数据准备（填充 input/output buffer）和 backend-specific 参数布局（OpenCL `ac.args` / Vulkan `ac.vulkanBindings` + `ac.pushConstants`）都是子类自己的职责
- 基类不持有共享状态，不通过 `adaptBackend()` 钩子复用——这保证各 backend 完全自治，不会因为一个 backend 的数据准备改动影响另一个
- 同一 Op 的 OpenCL 和 Vulkan variant 用不同的 variant 名（Vulkan 用 `vulkan_` 前缀），`findAdapter` 按 `(opType, variant)` 自然分发到正确的子类

### 为什么不用通用 adapter

- kernel 参数顺序是交错的（buf, int2, buf, int, int4...），通用模板无法覆盖
- 每个参数的语义值不同（shape 的 w/h/c、stride 的 x/y、activationType 等）
- 通用 adapter 的 arg 顺序猜测导致 `CL_INVALID_ARG_SIZE`(-51) 和 `CL_INVALID_ARG_VALUE`(-49)
- 逐个写虽然代码量大，但每个都确定正确，不需要反复调试

### 专用 adapter 模板

每个变体一个 `.hpp` + `.cpp` 文件，放在 `mnn/ops/` 或 `ncnn/ops/` 下。

**BinaryOp.hpp**（示例——MNN 有 OpenCL + Vulkan 两个子类）：
```cpp
// 基类：纯虚接口，不提供 adapt() 共享实现
class BinaryKernelBase : public OpAdapter {
public:
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override = 0;
};

// OpenCL binary_buf kernel: 1.2.0 (int4 shape) 和 3.6.0 (int size, int actType)
class OpenCLBinaryAddKernel : public BinaryKernelBase {
public:
    const char* opType() const override { return "binary"; }
    const char* variant() const override { return "binary_buf_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

// MNN Vulkan binary.comp kernel (ADD baked into SPIR-V)
class VulkanBinaryAddKernel : public BinaryKernelBase {
public:
    const char* opType() const override { return "binary"; }
    const char* variant() const override { return "vulkan_binary_buf_add_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};

void registerBinaryOp();
```

**BinaryOp.cpp**（示例——Vulkan 子类，完整实现 adapt()）：
```cpp
bool VulkanBinaryAddKernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    if (spec.tag != "3.6.0") return false;
    ac.entry = "vulkan_binary_buf_add_fp32";
    const int elemCount = spec.intParam("size", 256);
    const int n4 = (elemCount + 3) / 4;
    const int totalFloats = n4 * 4;

    // 1. 准备 input data（子类自己负责，Vulkan 需要对齐到 vec4）
    std::vector<float> in0(totalFloats), in1(totalFloats);
    for (int i = 0; i < elemCount; ++i) {
        in0[i] = 0.1f * (i % 7);
        in1[i] = 0.1f * (i % 5);
    }
    for (int i = elemCount; i < totalFloats; ++i) { in0[i] = 0.0f; in1[i] = 0.0f; }

    // 2. 创建 buffer + 设 Vulkan binding 映射
    AdaptedBuffer buf0; buf0.setFp32(in0); buf0.isOutput = false;
    AdaptedBuffer buf1; buf1.setFp32(in1); buf1.isOutput = false;
    AdaptedBuffer outBuf;
    outBuf.sizeBytes = totalFloats * sizeof(float); outBuf.isOutput = true;
    ac.buffers.push_back(buf0);
    ac.buffers.push_back(buf1);
    ac.buffers.push_back(outBuf);
    ac.vulkanBindings = {0, 1, 2};  // buffers[i] -> binding index

    // 3. push constant（按 shader uniform block 布局精确填充）
    struct PushParam { int32_t stride00[4]; int32_t activationType; };
    PushParam pp;
    pp.stride00[0] = 1; pp.stride00[1] = 1; pp.stride00[2] = n4; pp.stride00[3] = n4;
    pp.activationType = 0;
    ac.pushConstants.resize(sizeof(pp));
    std::memcpy(ac.pushConstants.data(), &pp, sizeof(pp));

    // 4. dispatch geometry
    const uint32_t localX = 256;
    const uint32_t gx = ((static_cast<uint32_t>(n4) + localX - 1) / localX) * localX;
    ac.globalSize[0] = gx; ac.globalSize[1] = 1; ac.globalSize[2] = 1;
    ac.localSize[0] = localX; ac.localSize[1] = 1; ac.localSize[2] = 1;
    ac.dims = 1;

    ac.validatorInputA = in0;
    return true;
}
```

### 实现步骤

1. 读 kernel 源码签名（OpenCL `__kernel void NAME(...)` 或 Vulkan `layout(binding=N) buffer` + `uniform` block）
2. 定义基类 `<Op>KernelBase : public OpAdapter`，`adapt() = 0`（纯虚，不提供实现）
3. 为每个 backend 定义子类 `OpenCL<Variant>Kernel` / `Vulkan<Variant>Kernel`，各自完整实现 `adapt()`
4. OpenCL 子类按**精确顺序**逐个参数写 `ac.args.push_back(AdaptedArg::xxx(...))`
5. Vulkan 子类设 `ac.vulkanBindings`（buffer→binding 映射）+ `ac.pushConstants`（按 uniform block 布局填充）+ `ac.globalSize`/`ac.localSize`
6. 为每个 tag 版本写 `if (spec.tag == "x.y.z")` 分支
7. 在 `MnnBridge.cpp` / `NcnnBridge.cpp` 注册 `registerXxxOp()`
8. 更新 `CMakeLists.txt` 加入新 `.cpp` 文件
9. 本机构建测试 → 设备验证

### AdaptedArg 类型

| Kind | 用途 | runner 行为 |
|------|------|------------|
| `SizeConst(dim)` | OpenCL GLOBAL_SIZE_DIMS 展开的 `__private const int` | `setArg(i, globalSize[dim])` |
| `Buffer(idx)` | `__global FLOAT*` / `layout(binding) buffer` | `setArg(i, clBuffer)` |
| `Scalar(Int/Float)` | `__private const int`/`float` 标量 | `setArg(i, val)` |
| `Int2(x,y)` | `int2` 参数 | `setArg(i, sizeof(cl_int2), &v)` |
| `Int4(x,y,z,w)` | `int4` 参数 | `setArg(i, sizeof(cl_int4), &v)` |

### compileMacros

用于传 `-D` 宏给 `clBuildProgram`（仅简单宏，非函数式）。

### 注册 adapter

在 `MnnBridge.cpp` 的 `registerMnnBridge()` 里调用 `registerXxxOp()`（注册所有 backend 子类）：
```cpp
MnnOps::registerRasterOp();         // OpenCLRasterSetZeroKernel + VulkanRasterBlitC4Kernel
MnnOps::registerUnaryOp();         // OpenCLUnaryExpKernel + VulkanUnaryExpKernel
MnnOps::registerBinaryOp();        // OpenCLBinaryAddKernel + VulkanBinaryAddKernel
registerFallbackAdapter();          // 兜底
```

**MnnBridge 加载 .spv**：对 `spec.backend == "vulkan"` 的 case，`MnnBridge::adapt()` 自动把 `.comp` 路径替换成 `.spv` 读取预编译 SPIR-V（与 `NcnnBridge` 一致）。

### FallbackAdapter

**仅用于尚未写专用 adapter 的变体**，作为临时占位。每写一个专用 adapter 就从 fallback 覆盖中移除。最终目标是所有变体都有专用 adapter，FallbackAdapter 不再被触发。

## 步骤 4: 更新 operator_cases.json

每个 operator 至少一个 case。已有专用 adapter 的用专用参数，通用 adapter 的用默认值，fallback 用最小默认值。

```bash
python3 -c "
import json
ops = json.load(open('replay_benchmark/kernel_corpus/operators.json'))
existing = json.load(open('replay_benchmark/kernel_corpus/operator_cases.json'))
# merge: 保留已有 case 参数，新 operator 用默认值
..."
```

## 步骤 5: 验证

### 优先本机测试

本机有 AMD Radeon 核显 + Mesa RADV Vulkan 1.4 + rusticl OpenCL，可以快速验证编译和 dispatch。**优先在本机测试通过后再推送到设备。**

```bash
# 本机构建（需要 MNN_BUILD_BENCHMARK=ON）
rm -rf build-host && mkdir build-host && cd build-host
cmake .. -DMNN_OPENCL=ON -DMNN_VULKAN=ON -DMNN_REPLAY_ENABLE_PERFCOUNTER=OFF \
  -DMNN_SEP_BUILD=ON -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_BENCHMARK=ON \
  -DMNN_BUILD_TEST=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target replay_benchmark.out -j$(nproc)

# 本机运行
LD_LIBRARY_PATH=build-host/source/backend/opencl:build-host/source/backend/vulkan:build-host \
  ./build-host/replay_benchmark.out \
  --kernel-corpus-bench --kernel-corpus-root replay_benchmark/kernel_corpus \
  --kernel-corpus-runs 5 --perf-counter-output /tmp/kc_host.json

# 查看结果
cat /tmp/kc_host.json | python3 -c "
import sys,json
d=json.load(sys.stdin)
stats={}
for c in d['cases']:
    k=c['compile_status']
    stats[k]=stats.get(k,0)+1
print('compile:', stats)
disp=sum(1 for c in d['cases'] if c['dispatch_status']=='dispatched')
val=sum(1 for c in d['cases'] if c['validation_status']=='validation_passed')
print(f'dispatched: {disp}, validation_passed: {val}, total: {len(d[\"cases\"])}')
"
```

### Python 测试
```bash
python3 -m unittest discover -s replay_benchmark/kernel_corpus/tests
```

### Vulkan shader 编译验证
```bash
cd replay_benchmark/kernel_corpus/operators/vulkan
for f in $(find . -name "*.comp"); do
  glslangValidator -V --target-env vulkan1.1 --entry-point main "$f" -o /dev/null 2>/dev/null || echo "FAIL $f"
done
```

### 构建 + 设备验证
```bash
cmake --build build-aarch64-gnueabihf-vulkan --target replay_benchmark.out -j2
scp build-aarch64-gnueabihf-vulkan/replay_benchmark.out root@<device>:/mnt/.../bin/
rsync -az replay_benchmark/kernel_corpus/ root@<device>:/mnt/.../kernel_corpus/
ssh root@<device> "cd /mnt/... && LD_LIBRARY_PATH=lib:... ./bin/replay_benchmark.out \
  --kernel-corpus-bench --kernel-corpus-root kernel_corpus --kernel-corpus-runs 100"
```

### 批量设备验证脚本

```python
# /tmp/run_all_opencl.py — 逐个跑 OpenCL case 并收集结果
import json, subprocess, os
cases = json.load(open("kernel_corpus/operator_cases.json"))["cases"]
opencl = [c for c in cases if c["backend"]=="opencl"]
env = dict(os.environ); env["LD_LIBRARY_PATH"]="lib"
for c in opencl:
    name = c["name"]
    try:
        subprocess.run(["./bin/replay_benchmark.out","--kernel-corpus-bench",
            "--kernel-corpus-root","kernel_corpus","--kernel-corpus-case",name,
            "--kernel-corpus-runs","1","--perf-counter-output","/tmp/kc_one.json"],
            capture_output=True, timeout=10, env=env)
        with open("/tmp/kc_one.json") as f:
            d = json.load(f)
        if d["cases"]:
            cc = d["cases"][0]
            print(f"{name}: {cc.get('compile_status','?')} {cc.get('dispatch_status','?')} {cc.get('validation_status','?')} {cc.get('error','')[:60]}")
    except Exception as e:
        print(f"{name}: CRASH {type(e).__name__}")
```

## 关键约束

1. **不 fallback 简化实现**：baked kernel 必须功能上与原版一致，不自己简化或替换逻辑。cooperative-matrix-only shader 标记为 unsupported，不篡改源码。
2. **bake 不跳过任何 kernel**：含 image2d_t 的 kernel 也保留，runner 遇到不支持的能力时标记 `unsupported`。
3. **不编译历史 backend**：只提取 kernel 文本，用当前 MNN OpenCL/Vulkan runtime 编译。
4. **禁止访问** `schema/private/` 和 `source/internal/`。
5. **不批量格式化历史源码**，不修改尾随空格。
6. **C++ 遵循** C++11、4 空格缩进、禁止 namespace 作用域动态初始化对象。
7. **PMU 只包围 dispatch**：warmup 在 PMU session 外，workload dispatch 在 session 内。
8. **in-place shader**（sigmoid/tanh）每次 dispatch 前重写 initialData，保证 validation 对比原始输入。
9. **Op vs Kernel 命名**：一个 Op（计算语义）有多个 Kernel（变体实现）。类命名：基类 `<Op>KernelBase`（纯虚 `adapt()=0`，不提供共享实现），OpenCL 子类 `OpenCL<Variant>Kernel`，Vulkan 子类 `Vulkan<Variant>Kernel`。每个子类各自完整实现 `adapt()`（数据准备 + backend 参数布局），不通过 `adaptBackend()` 钩子复用。Vulkan variant 名用 `vulkan_` 前缀与 OpenCL variant 区分，`findAdapter` 按 `(opType, variant)` 自然分发。
10. **cooperative matrix shader**（gemm_cm/sdpa_fa_cm/sdpa_cross_cm）需 `#extension GL_KHR_cooperative_matrix` + `--target-env vulkan1.1`，无 fallback 时标记 unsupported。
11. **OpenCL `-D` 不支持函数式宏**：`CONVERT_FLOAT4(x)` 等必须 `#define` 在 preamble 里，不能用 `-D` 传。
12. **OpenCL runner 已支持 compileMacros**：`clBuildProgram` 接受 `ac.compileMacros` 拼成的 build options 字符串。
13. **backend-specific validator**：validator 需按 backend 数据布局适配。OpenCL 用 NC4HW4 FLOAT4 布局，Vulkan 可能用标量 FLOAT（如 reduce）。已有 `reduction_sum_fp32`（NC4HW4）和 `reduction_sum_scalar_fp32`（标量）两个版本。新增 backend 的 case 时需确认 validator 与该 backend 的实际数据布局匹配。
14. **忠实性审计方法论（强制）**：遇到"无法忠实复制 MNN kernel"的结论前，必须按以下顺序排查，**禁止仅凭环境限制就放弃**：
    1. **编译失败 → 查 MNN CMake flag 覆盖**：MNN 各 backend 子目录的 `CMakeLists.txt` 可能对顶层 flag 做局部覆盖。最典型的例子：`source/backend/cuda/CMakeLists.txt:110` 显式 `set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")` 覆盖顶层 `-fno-exceptions`，使 `cub::BlockScan`/thrust 能编译。corpus 必须镜像同一覆盖（在 `replay_benchmark/CMakeLists.txt` 里加对应 flag），而非声明"无法编译"。验证方法：`grep -rn "set(CMAKE_CXX_FLAGS\|target_compile_options\|add_compile_options" source/backend/<backend>/CMakeLists.txt`。
    2. **架构不支持 → 验证 `#if __CUDA_ARCH__` 守卫下类型是否仍可用**：BF16 等"高架构"类型在低架构下**类型定义仍可用**（通过 `#include <cuda_bf16.h>`），只是计算指令不可用。kernel body 用 `#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))` 守卫后，低架构编译为**空 kernel**（声明存在、body 不生成），仍可链接/调用（不产生输出）。这意味着"需 sm80+ 运行"≠"不能在 sm75 编译"——kernel 应复制，标注运行环境而非跳过。
    3. **runner 不兼容 → 区分"编译限制"与"架构限制"**：若 kernel 类型实例化（如 `int2*` = 8 字节 4-half 打包）与 corpus runner 的 buffer 模型（float 标量）不兼容，先确认这是**runner 架构限制**（需重构 buffer 管理）而非**编译限制**（换个实例化即可）。只有 runner 架构限制才允许保留数学等价的 corpus 适配，且必须标注为"corpus 适配"并说明限制根因。
    4. **HANDOFF/审计描述与源码不符 → 以 `git log`/源码为准**：审计文档可能过时或描述错误。例：曾标注"MNN MOD 用 fmod"，但 `git log -p -S "fmod" -- source/backend/cuda/execution/BinaryExecution.cu` 确认 MNN 从未用 fmod（一直是 `x - x/y`）。遇描述与实现不一致时，**以 MNN 源码为准**，修正文档而非"修复"已正确的 corpus。
    5. **黄金法则**：MNN 能编译的 kernel，corpus 理论上也能编译（同一 nvcc + 同一 CUDA toolkit）。遇编译失败先找 flag 差异，不要归因于"环境限制"。

## 已知编译失败分类及修复方向

| 失败原因 | 涉及 kernel | 修复方式 |
|----------|------------|---------|
| 缺 `CONVERT_FLOAT*` 等函数式宏 | 多个 | 已修：加到 bake preamble |
| `OPERATOR`/`OPERATE` 未定义 | unary/binary/reduction | 已修：bake 时自动 `#define` |
| `WI_F`/`RI_F` 未定义 | raster_buf (含 image entry) | 已修：加到 bake preamble |
| `Error: Program not built!` | subgroup kernel | 需 `-cl-std=CL2.0` build option |
| `OpenCL 2.0 built-in` | unary_subgroup | 同上 |
| `IN_C_BLOCK`/`LOCAL_SIZE`/`OPWM`/`STRIDE_X`/`VEC_H` | conv/matmul_local/pooling_subgroup/topkv2/strassen | kernel 特有宏，需在 adapter compileMacros 里设 `-D` |
| `float4→int4` 类型转换 | range_buf/conv_2d_int | 需要正确的 `CONVERT` 宏或 INT 路径定义 |
| `expected ')'` | gemm_conv1x1/gemv/grid_sample/raster | preamble 宏与 kernel 内 `#define` 冲突 |
| cooperative-matrix-only | gemm_cm/sdpa_fa_cm/sdpa_cross_cm | 标记 unsupported |

## 已验证通过的 case（host 验证）

### OpenCL / mnn (53/62 compiled, host)
- buffer_set_zero_fp32 (1.2.0/3.6.0) — OpenCLRasterSetZeroKernel
- unary_buf_exp_fp32 (1.2.0/3.6.0) — OpenCLUnaryExpKernel
- matmul_buf_nobias_fp32 (1.2.0) — OpenCLMatmulKernel
- reduct_buf_sum_fp32 (1.2.0) — OpenCLReductionSumKernel
- pooling_max_fp32 (1.2.0) — OpenCLPoolingMaxKernel
- cast_buf_fp32 (3.6.0) / select_buf_fp32 (3.6.0) — 通用 adapter

### Vulkan / mnn (6/6 compiled+dispatched+validation_passed, host) — 新增
- vulkan_unary_buf_exp_fp32 (3.6.0) — VulkanUnaryExpKernel
- vulkan_binary_buf_add_fp32 (3.6.0) — VulkanBinaryAddKernel
- vulkan_blit_c4_fp32 (3.6.0) — VulkanRasterBlitC4Kernel
- vulkan_reduce_buf_sum_fp32 (3.6.0) — VulkanReductionSumKernel
- vulkan_maxpool_fp32 (3.6.0) — VulkanPoolingMaxKernel
- vulkan_avgpool_fp32 (3.6.0) — VulkanPoolingAvgKernel

### Vulkan / ncnn (357/361 compiled, 48 validation_passed, host)
- sigmoid_fp32 / tanh_fp32 / permute_order0_fp32 (ncnn 20190611) — VulkanSigmoidKernel / VulkanTanhKernel / VulkanPermuteKernel
- sigmoid_pack4_fp32 / tanh_pack4_fp32 / absval_pack4_fp32 / relu_pack4_fp32_slope0 / concat_axis0_fp32 (ncnn 20260526) — VulkanSigmoidPack4Kernel 等

## 已知不支持的 kernel

| 文件 | 原因 | 处理 |
|------|------|------|
| gemm_cm.comp | cooperative-matrix-only，PAD/sum 在 coop matrix 分支内 | 标记 unsupported |
| sdpa_fa_cm.comp | 同上 | 标记 unsupported |
| sdpa_cross_cm.comp | 同上 | 标记 unsupported |

## 设备信息

- **Rhinopi-X1**：`root@192.168.101.227`，`/mnt/nvme/workspace/replay-benchmark`，Adreno 740
  - Vulkan lib: `/mnt/nvme/workspace/replay-benchmark-vulkan/lib/libvulkan.so`
- **OrangePi**：`root@192.168.101.113`，`/mnt/ssd/workspace`，Mali-G610
- 凭据只从环境读取，不写入仓库

---

## CUDA Backend 适配规范

### CUDA 与 OpenCL/Vulkan 的本质差异

| 维度 | OpenCL/Vulkan | CUDA |
|------|--------------|------|
| kernel 形态 | 源码字符串，runtime 编译 | nvcc 预编译 `__global__` 符号，编进 `libMNN_Cuda_Main.so` |
| 符号可见性 | cl::Kernel(program, name) 按名取 | `__global__` host stub 永远 internal，跨 TU 不可见 |
| corpus 策略 | bake 源码 → runtime 编译 → setArg → dispatch | **重实现 kernel** → `extern "C"` shim → `<<<>>>` 启动 |
| 多 tag | 源码按 tag 存不同文件（`1.2.0/xxx.cl` vs `3.6.0/xxx.cl`） | kernel 代码按 tag 内部分支（同 shim，不同参数/实现） |
| PMU | Adreno/Mali ioctl 或 Vulkan query | CUPTI Range Profiler（需 `RmProfilingAdminOnly=0` 或 sudo） |

### CUDA corpus 三层架构

```
replay_benchmark/kernel_corpus_bridge/cuda/
├── CudaOpAdapter.hpp              ← CudaOpAdapter 接口 + CudaLaunchCtx
├── CudaOps.hpp                    ← 所有 adapter 类声明（共享） + kBlock/gridFor/fillInputRamp/mnnBlock120 helper
├── CudaOps.cpp                    ← A类 fp32 adapter: relu/clamp/cast/binary/range/select/softmax/layernorm/prelu/scale/pool（g++ 编译，不拆分）
├── CudaOpsMisc.cpp                ← B类 fp32 adapter: gatherv2/argmax/interp/transpose/gridsample/reduction/topkv2/raster/convdw/pack_c4 等（g++ 编译，不拆分）
├── CudaOpsFp16.cu                 ← fp16/int8/bf16 adapter（nvcc 编译，__half 不可用于 g++）
└── kernels/                       ← GPU kernel 定义层（nvcc 编译，按算子类型分文件）
    ├── corpus_common.cuh           ← 共享 device helper: warpReduceSum/blockReduceMax/CUDA_KERNEL_LOOP/DivModFast/ReduceParam_127
    ├── unary_cast.cu               ← RELU/CLAMP/CAST 各版本
    ├── binary.cu                   ← ATAN2/MOD/LOGICALOR
    ├── range.cu                    ← RANGE
    ├── select.cu                   ← SELECT 3.6.0 + 2.7.2
    ├── softmax.cu                  ← SOFTMAX 3.6.0 + 2.2.2
    ├── layernorm.cu                ← LAYERNORM 3.6.0/1.2.7/2.2.2/1.2.0
    ├── prelu.cu                    ← PRELU 3.6.0/1.2.7/1.2.8/1.2.0
    ├── scale.cu                    ← SCALE 3.6.0/1.2.7/1.2.0
    ├── pool.cu                     ← MAXPOOL/AVGPOOL/GLOBAL 3.6.0/1.2.0
    ├── gatherv2_argmax.cu          ← GATHERV2 + ARGMAX/ARGMIN 3.6.0/1.2.0/1.2.7/1.2.8/2.1.2/2.5.0
    ├── interp.cu                   ← INTERP nearest/bilinear/round 3.6.0/1.2.0/1.2.7
    ├── transpose.cu                ← NHWC↔NCHW 3.6.0/2.1.2
    ├── gridsample.cu               ← GRID_SAMPLE nearest/bilinear/3D 3.6.0/2.7.2
    ├── reduction_naive.cu          ← SUM/MEAN/MAX/MIN/PROD 3.6.0/1.2.0/1.2.7
    ├── topkv2.cu                   ← TopKAllRows/GetResultAllRows
    ├── raster.cu                  ← blitRegion 3.6.0/2.4.1 + pack_c4/unpack_c4/setzero/add_bias 1.2.0
    └── convdw.cu                   ← CONV_DW 3.6.0/1.2.0/2.0.4
```

**拆分规则**（kernels/ 层）：
- 每个 `.cu` 文件按**算子类型**命名，同一算子的所有版本（3.6.0 + 1.2.7 + 1.2.0 等）放同一文件
- `__global__` kernel 和调用它的 `extern "C"` shim 必须在同一 `.cu` 文件（CUDA 要求 kernel host stub 同 TU 可见）
- 共享 device helper 放 `corpus_common.cuh`（inline/device-template，无 ODR 问题）
- **adapter 层（CudaOps.cpp/CudaOpsMisc.cpp/CudaOpsFp16.cu）暂不拆分**，保持按 A/B 类 + 数据类型分文件

**不拆 adapter 的原因**：
- `extern "C"` 前向声明分散在 4 处（CudaOps.cpp 顶部、CudaOpsMisc.cpp 中间 4 个块、CudaOpsFp16.cu 顶部），拆分后需在各子文件重复声明
- `REDUCTION_ADAPTER`/`REDUCTION_ADAPTER_REUSE` 宏定义在 CudaOpsMisc.cpp 中间，需整体搬到 reduction.cpp
- CudaOpsFp16.cu 需 nvcc 编译，拆分后各 .cu 要加入 `cuda_add_library`，收益不大
- CudaOps.hpp 是公共声明头，保留所有 adapter 类声明集中放置（类似 `<vector>`）

### CUDA Shim 模式（固定规则）

1. **每个 `__global__` kernel 必须在 corpus 的 `.cu` 文件里重实现**——因为 `libMNN_Cuda_Main.so` 是 SHARED 库，`__global__` 的 host stub 永远是 internal linkage，跨 TU 引用不到
2. **重实现的 kernel 放在 `namespace MNN::Corpus`**，与 MNN 推理后端的 `namespace MNN::CUDA` 隔离
3. **每个 kernel 配一个 `extern "C"` 启动 shim**（如 `mnn_corpus_relu_fp32`），签名只含 POD 类型（`const float*`/`int`/`cudaStream_t`），不含 C++ 类
4. **shim 定义在 `.cu` 文件里**（nvcc 编译），**shim 声明在 adapter `.cpp` 文件顶部的 `extern "C" {}` 块里**（g++ 编译）
5. **shim 名固定，不含 tag 版本号**——tag 分流在 adapter 内部
6. **`DivModFast` 等 device helper 放 `kernels/corpus_common.cuh`**（inline/device-template，各 `.cu` 文件 `#include` 即可，无 ODR 问题）

### CUDA Kernel 全覆盖约束（强制规则）

> **MNN CUDA backend 的每个 `__global__` kernel 都必须在 replay_benchmark 有对应重实现。**
> 缺失清单见 `replay_benchmark/kernel_corpus_bridge/cuda/CUDA_KERNEL_VERSIONING.md`
> 末尾"缺失 kernel 变体清单"，逐个版本补齐。

1. **新增 kernel 时必须检查覆盖率**：用以下命令对比 MNN 源码与 corpus：
   ```bash
   # MNN 有但 replay_benchmark 没有的 __global__ kernel
   grep -rh "__global__ void" source/backend/cuda/execution/*.cu \
     | sed 's/.*__global__ void //;s/(.*//;s/ //g' | sort -u | grep -v "##" > /tmp/mnn.txt
   grep -rh "__global__ void" replay_benchmark/kernel_corpus_bridge/cuda/kernels/*.cu \
     | sed 's/.*__global__ void //;s/(.*//;s/ //g' | sort -u > /tmp/replay.txt
   comm -23 /tmp/mnn.txt /tmp/replay.txt  # 差集 = 缺失
   ```
2. **按算子类型分文件**：新增 kernel 变体放入 `kernels/<op>.cu`（与已有同算子的 kernel 一起），不可新建 `intermediate_versions.cu` / `misc_*.cu` 等按版本分的文件
3. **函数体变更即独立 shim**：不仅看签名，更看公式/索引/累积方式。用 `git diff <tagA> <tagB> -- <file>.cu` 对比 `__global__` 函数体
4. **逐个版本推进**：补齐时先确认引入 tag，再做 body-diff，最后写 kernel+shim+adapter+case+validate。同一算子的不同实现策略（如 softmax naive/warp32/axis_reduce）按"变体"规则处理（见下"变体与版本的分层规则"）
5. **补齐后更新文档**：从 `CUDA_KERNEL_VERSIONING.md` 缺失清单中移除已补齐的条目

### CUDA Adapter 变体与版本的分层规则（强制规则）

> **变体(variant) ≠ 版本(tag)**。参照 OpenCL/Vulkan 的做法，两者是正交的两个维度。

#### 概念定义

| 概念 | 含义 | 对应什么 | 例子 |
|------|------|---------|------|
| **变体(variant)** | 同一算子的**不同实现策略** | 一个 adapter 类 + 一个 variant 名 | SOFTMAX 的 naive/warp32/axis_reduce；conv 的 conv_2d_buf / conv_2d_c16_subgroup_buf |
| **版本(tag)** | 同一变体在不同 MNN 版本的**函数体更新** | adapter 类内部按 `spec.tag` 分流 | SOFTMAX_WARP_32 在 2.4.2 无 exp cutoff，3.6.0 有 exp cutoff |

#### 变体划分规则

1. **每个变体一个独立 adapter 类**，类名含变体标识，`variant()` 返回独立名：
   ```cpp
   // 正确：naive 和 warp32 是不同变体，各自独立 adapter 类
   class CudaSoftmaxFp32Kernel : public CudaOpAdapter {  // naive 变体
       const char* variant() const override { return "cuda_softmax_fp32"; }
       // ...
   };
   class CudaSoftmaxWarp32Fp32Kernel : public CudaOpAdapter {  // warp32 变体
       const char* variant() const override { return "cuda_softmax_warp32_fp32"; }
       // ...
   };
   class CudaSoftmaxAxisReduceFp32Kernel : public CudaOpAdapter {  // axis_reduce 变体
       const char* variant() const override { return "cuda_softmax_axis_reduce_fp32"; }
       // ...
   };
   ```

2. **变体名 = `<op>_<策略>_fp32`**，如 `cuda_softmax_warp32_fp32`、`cuda_softmax_axis_reduce_fp32`

3. **operators.json 每个变体+tag 一条 entry**，entry 字段 = 该变体在该 tag 下的 shim 名：
   ```json
   {"backend":"cuda","op_type":"softmax","tag":"3.6.0","variant":"cuda_softmax_warp32_fp32","entry":"mnn_corpus_softmax_warp32_fp32"}
   {"backend":"cuda","op_type":"softmax","tag":"2.4.2","variant":"cuda_softmax_warp32_fp32","entry":"mnn_corpus_softmax_warp32_242_fp32"}
   ```

4. **operator_cases.json 每个变体+tag 一个 case**，不用 int_params 区分变体

#### 版本(tag)分流规则（同一变体内）

**类名不含版本号**——同一个变体的所有 tag 共用一个 adapter 类，内部按 `spec.tag`/`ac.tag` 分流：

```cpp
// 正确：类名固定（含变体标识但不含 tag），内部按 tag 分流
class CudaSoftmaxWarp32Fp32Kernel : public CudaOpAdapter {
    const char* variant() const override { return "cuda_softmax_warp32_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override {
        if (spec.tag == "2.4.2") {
            ac.entry = "mnn_corpus_softmax_warp32_242_fp32";  // 无 exp cutoff
        } else {
            ac.entry = "mnn_corpus_softmax_warp32_fp32";      // 有 exp cutoff (3.6.0)
        }
        // ... 相同的 buffer/grid 设置 ...
    }
    cudaError_t launch(const AdaptedCase& ac, const CudaLaunchCtx& ctx) const override {
        if (ac.entry == "mnn_corpus_softmax_warp32_242_fp32") {
            mnn_corpus_softmax_warp32_242_fp32(...);
        } else {
            mnn_corpus_softmax_warp32_fp32(...);
        }
    }
    bool validate(...) const override {
        // 2.4.2 和 3.6.0 的验证公式相同（都是 softmax），只是 kernel 实现有 exp cutoff 差异
        // 验证容差可适当放宽
    }
};
```

#### 反模式（禁止）

```cpp
// 错误：用 int_params 区分变体——变体应该用独立 adapter 类
bool CudaSoftmaxFp32Kernel::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    int variant = spec.intParam("softmax_variant", 0);  // 0=naive, 1=warp32 ← 错！
    if (variant == 1) { ... }
}
```

#### 变体识别方法

如何判断 MNN 的 kernel 是否属于不同变体：
1. **看 MNN `onExecute()` 的调度逻辑**：如果按 axis 大小 / 数据类型 / 布局选择不同 `__global__` kernel，则每个 kernel 是一个变体
2. **看 kernel 签名**：参数个数/类型不同 = 不同变体（如 SOFTMAX vs SOFTMAX_WARP_32 vs SOFTMAX_AXIS_REDUCE）
3. **看 kernel 函数体**：如果两个 kernel 公式完全不同（不是版本演进导致的微调），则是不同变体

### CUDA 多 Tag 分流规则（固定规则）

> **核心判定原则（强制）**：是否需要为某个 tag 单独写 kernel + shim，**只看
> `__global__` 函数体的源码是否完全一致**——不仅看签名，更要看公式/索引/累积方式。
> 用 `git diff <tagA> <tagB> -- <file>.cu` 对比 `__global__` 函数体：
> - **函数体字节级一致**（仅空白/注释/host 代码差异）→ 可复用相邻版本的 shim
> - **函数体有任何实质差异**（哪怕签名相同）→ **必须独立重实现 kernel + 独立 shim**
>
> 禁止仅因"签名相同"就共用 shim。例如 PROD 在 2.5.1 起循环从 `v=1` 改为 `v=0`、
> `res=basicInput[0]` 改为 `sumValue=1.0`——签名相同但公式不同，必须独立 shim。
> 调研中间版本时，对每个变更点都要 `git diff` 函数体确认，不能凭签名判断。

| tag | 含义 | 分流点 |
|-----|------|--------|
| `1.2.0` | MNN 1.2.0 发布版的 CUDA kernel | `adapt()` 里 `if (spec.tag == "1.2.0")` 选参数布局 + shim 名 |
| `1.2.7` ~ `2.8.4` | 中间版本变更点 | `adapt()` 里 `else if (spec.tag == "x.y.z")` 分支，独立 shim |
| `3.6.0` | MNN 3.6.0 当前版 | `adapt()` 的 `else` 分支 |

**block/grid 按版本分流（强制）**：
- `1.2.0`：自适应 block 分档，`adapt()` 内调 `mnnBlock120(total)` 设 `ac.localSize[0]`
- `1.2.1+` ~ `3.6.0`：固定 `kBlock = 128`
- `CudaLaunchCtx` 的 `grid`/`block` 无默认值，由 `ac.globalSize[0]`/`ac.localSize[0]` 设置

**中间版本变更点速查**（详见 `replay_benchmark/kernel_corpus_bridge/cuda/CUDA_KERNEL_VERSIONING.md`）：

| 算子 | 变更 tag | 是否需独立 shim | 原因 |
|------|---------|----------------|------|
| CONV_DW | 1.2.0, 1.2.7, 2.0.4, 2.2.3 | 是（各版本函数体不同） | 参数布局/索引/类型逐版变化 |
| Reduction SUM/MEAN | 1.2.7, 2.5.1, 2.8.2 | 是（累积类型/参数顺序/拆分变化） | T→float 累积、参数顺序翻转 |
| Reduction MAX/MIN/PROD | 1.2.7, 2.5.1 | 是（PROD 公式改、参数顺序改） | PROD 循环起点 0/`sumValue=1.0` |
| INTERP_NERAEST/ROUND/BILINEAR | 1.2.7, 2.0.4 | 是（PACK_NUMBER 重打包、加 c_p） | 索引方式变化 |
| LAYERNORM | 2.2.2, 2.8.4 | 是（gamma/beta 类型、加 RMSNorm） | 累积类型 + 新参数 |
| PRELU | 2.0.4 | 是（slope 类型、公式重写） | nhw/c_idx 布局 |
| SCALE | 2.0.4 | 是（scale/bias 类型、公式重写） | nhw/c_idx 布局 |
| ARGMAX | 2.0.4, 2.5.0 | 是（输出类型、指针算术重写） | 索引语义变化 |
| SELECT | 2.8.0 | 是（加 stride 参数 + 公式改） | 新增 s1/s2 |
| SOFTMAX | 2.2.3 | 是（ReduceParam→显式 int） | 参数布局变化 |
| blitRegion | 2.4.2 | 是（加 count + fuseIndex 循环） | 循环结构变化 |
| NCHW_2_NHWC | 2.7.1 | 是（src_offset 用 inChannelPack） | 索引公式变化 |
| GRID_SAMPLE_NEAREST/BILINEAR | 2.8.0 | 是（dst_offset 写入公式） | 输出索引变化 |
| CLAMP | 1.2.7 | 是（float→T 模板化） | 类型泛化 |
| RELU/ATAN2/MOD/LOGICALOR/GATHERV2 | 无 | 否（1.2.0→3.6.0 函数体一致） | 可共用 shim |

**1.2.0 vs 3.6.0 已知差异**（保留供参考，中间版本见上表）：

| 算子 | 1.2.0 | 3.6.0 | 差异类型 |
|------|-------|-------|---------|
| RELU/CLAMP/ATAN2/MOD/LOGICALOR | 签名相同 | 签名相同 | 无差异，共用 shim |
| CAST/CASTBOOL/GATHERV2 | 签名相同 | 签名相同 | 无差异，共用 shim |
| blitRegion | 签名相同 | 签名相同 | 无差异，共用 shim |
| MaxPool/AvgPool | NCHW 布局, `bc` 参数 | NC4HW4 布局, `ib+ic_p` | **独立 kernel + shim** |
| Reduction SUM/MEAN | T 累积, `(inside,axis,outside)` | float 累积, `(outside,axis,inside)` | **独立 kernel + shim** |
| Reduction MAX/MIN/PROD | 逻辑相同, 参数顺序不同 | 逻辑相同 | 复用 3.6.0 shim, launch 里 swap 参数 |
| ARGMAX | 输出 `T*`（float 存 index） | 输出 `int*` | **独立 kernel + shim** |
| SCALE | scale/bias 是 `T*` | scale/bias 是 `float*` | **独立 kernel + shim** |
| LAYERNORM | gamma/beta 是 `T*`, 无 RMSNorm | gamma/beta 是 `float*`, 有 RMSNorm | **独立 kernel + shim** |
| PRELU | slope 是 `T*`, `div_factor` | slope 是 `float*`, `share_factor` | **独立 kernel + shim** |
| INTERP | `INTERP(n, ...)` 无 c_p | `INTERP_NERAEST(total, c_p, ...)` | **独立 kernel + shim** |
| Softmax | cuDNN（无 `__global__`） | 自定义 kernel | 1.2.0 无 corpus case |
| pack_c4/unpack_c4 | 有 | 用 PACKCOMMON 替代 | 1.2.0 独有 shim |
| SETZERO/add_bias | 有 | 不存在 | 1.2.0 独有 shim |
| Cast/Range/Select/TopKV2/Transpose/GridSample/ArgMin/RoPE | 不存在 | 有 | 1.2.0 无 corpus case |

### CUDA PMU（CUPTI Range Profiler）

1. **PMU 后端**：`MNNPerfCounter` 的 `NvRangeProfiler.cpp`，通过 CUPTI Range Profiler API 做 per-kernel 计数
2. **权限**：需 `RmProfilingAdminOnly=0`（`/proc/driver/nvidia/params`），或 `sudo` 运行
3. **metric 查询**：`./build/list_cuda_metrics` 列举本机 GPU 支持的全部 metric
4. **手动指定**：`--perf-counter-events sm__cycles_elapsed.avg,sm__inst_executed.avg`
5. **runCuda() 流程**：`beginPmu()` → 每个 launch 包 `beginRange()/endRange()` → `endPmu()` → `nvEvaluateMetrics()`
6. **fallback**：PMU 不可用时 fallback 到 `cudaEvent` 计时

### CUDA 构建规则

1. `replay_cuda_corpus` 静态库（nvcc 编译）含 `kernels/*.cu`（17 个按算子分的文件）+ `CudaOpsFp16.cu`
2. `replay_benchmark.out` 链接 `replay_cuda_corpus` + `MNN_Cuda_Main` + `libcuda` + `libcupti` + `libnvperf_host/target`
3. `CudaOps.cpp` + `CudaOpsMisc.cpp` 用 g++ 编译（fp32 adapter，不需要 `__half`），直接编入 `replay_benchmark.out`
4. `CudaOpsFp16.cu` 用 nvcc 编译（fp16/int8 adapter，需要 `__half`/`__float2half`），编入 `replay_cuda_corpus`
5. CMake: `find_package(CUDA)` + `cuda_add_library` 编 kernels/*.cu + CudaOpsFp16.cu, `target_link_libraries` 链 MNN_Cuda_Main

### CUDA 验证流程

```bash
# 本机构建
export PATH=/usr/local/cuda-13.2/bin:$PATH
cd build && cmake .. -DMNN_CUDA=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_CUDA_NATIVE_ARCH=ON -DMNN_REPLAY_ENABLE_PERFCOUNTER=ON
make replay_benchmark.out list_cuda_metrics -j$(nproc)

# 列举 metric
./build/list_cuda_metrics --no-submetrics

# 运行 corpus（需 sudo 或 RmProfilingAdminOnly=0）
echo 'password' | sudo -S ./build/replay_benchmark.out \
  --kernel-corpus-bench --kernel-corpus-root replay_benchmark/kernel_corpus \
  --perf-counter-events sm__cycles_elapsed.avg \
  --perf-counter-output /tmp/result.json

# 查看
python3 -c "
import json
d=json.load(open('/tmp/result.json'))
cases=[c for c in d['cases'] if c['backend']=='cuda']
print(f'CUDA: {len(cases)}, valid: {sum(1 for c in cases if c[\"valid\"])}/{len(cases)}')
"
```
