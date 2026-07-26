---
name: kernel-adapt
description: 将历史 MNN/ncnn 的 OpenCL/Vulkan kernel 适配到当前 MNN GPU runtime 运行。覆盖 kernel 提取、baked 文件生成、版本桥接器(OpAdapter)实现、case 注册、设备验证的完整 TDD 流程。
---

# 历史 GPU Kernel 适配 SKILL

> **触发条件**：当用户请求适配/运行历史 kernel、扩展算子变体覆盖、为 kernel corpus 新增 adapter 时触发。

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
    vulkan/<op_type>/ncnn/<tag>/<variant>.comp (+ .spv)
  operators.json                      # 索引 (backend/op_type/framework/tag/variant/entry/file)
  operator_cases.json                 # 可执行 case (423 个)
  bake_ncnn_shaders.py                # 自动 bake ncnn .comp
  bake_mnn_kernels.py                 # 自动 bake MNN .cl
  extract_operator_kernels.py         # 扫描 operators/ 生成 operators.json
  gen_adapter_scaffold.py             # 分析 kernel 签名，归类参数布局

kernel_corpus_bridge/
  Bridge.hpp                          # AdaptedCase + Bridge 接口 + validators
  Bridge.cpp                          # 注册表 + validator 实现
  OpAdapter.hpp                       # OpAdapter 接口 + FallbackAdapter + findAdapter
  OpAdapter.cpp                       # FallbackAdapter 实现
  mnn/
    MnnBridge.hpp/.cpp                # 顶层分发 + 注册 OpAdapter
    ops/
      RasterOp.hpp/.cpp               # 专用 adapter (buffer_set_zero)
      UnaryOp.hpp/.cpp                # 专用 adapter (unary_buf_exp)
      MatmulOp.hpp/.cpp               # 专用 adapter (matmul_buf_nobias)
      ReductionOp.hpp/.cpp            # 专用 adapter (reduct_buf_sum)
      PoolingOp.hpp/.cpp              # 专用 adapter (pooling_max)
      GenericOps.hpp/.cpp             # 通用 adapter (覆盖 ~40 个变体)
  ncnn/
    NcnnBridge.hpp/.cpp
    ops/
      ElementwiseOp.hpp/.cpp          # 20190611 sigmoid/tanh (标量 sfp)
      Pack4Op.hpp/.cpp                # 20260526 sigmoid/tanh/absval/relu (pack4 sfpvec4)
```

## 步骤 1: Bake Kernel 源码

### bake 的原则

1. **原样保留 kernel 源码**——不简化、不 fallback、不跳过任何 entry
2. **只补齐编译期宏定义**——使单文件可独立编译
3. **含 image2d_t 的 kernel 也保留**——runner 遇到不支持的能力时标记 `unsupported`
4. **cooperative-matrix-only shader 保留原样**——不篡改源码，设备不支持时标记 `unsupported`

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

## 步骤 3: 实现 OpAdapter

### 专用 adapter vs 通用 adapter

**专用 adapter**：参数布局复杂或需要特定 validator 的算子。每个 `(op_type, variant)` 一个类。

**通用 adapter (GenericBufAdapter)**：参数布局相同的一类变体共用一个 adapter。通过 `TagSpec` 表声明每个 tag 下的参数布局（entry、buffer 数、scalar 数、sizeConst 数、int2 数、dispatch 维度、validator）。

```cpp
// GenericBufAdapter 用法示例
add({"cast", "cast_buf_fp32", {
    TS{"3.6.0", "cast_buf", 1,1, 1,0,2,0, 2, "identity_fp32"},
}});
// TagSpec: {tag, entry, numInputBufs, numOutputBufs, numScalarInts, numScalarFloats, numSizeConsts, numInt2s, globalDim, validator}
```

**优先用通用 adapter 覆盖多变体**，只有条件严格的才单独写专用 adapter。

### AdaptedArg 类型

| Kind | 用途 | runner 行为 |
|------|------|------------|
| `SizeConst(dim)` | OpenCL GLOBAL_SIZE_DIMS 展开的 `__private const int` | `setArg(i, globalSize[dim])` |
| `Buffer(idx)` | `__global FLOAT*` / `layout(binding) buffer` | `setArg(i, clBuffer)` / `writeBuffer(descSet)` |
| `Scalar(Int/Float)` | `__private const int` 等标量 | `setArg(i, val)` |
| `Int2(x,y)` | `int2` 参数（如 pooling shape） | `setArg(i, cl_int2{x,y})` |

### compileMacros

`AdaptedCase.compileMacros` 是 `vector<string>`，runner 拼成 build options 传给 `clBuildProgram`。用于传 `-D` 宏（仅简单宏，非函数式）。

```cpp
ac.compileMacros.push_back("-DBIAS");
ac.compileMacros.push_back("-DRELU");
```

### tag 版本差异

同一 variant 在不同 tag 下参数布局可能不同。`GenericBufAdapter` 用 `TagSpec` 表按 tag 选不同布局：

```cpp
add({"matmul", "matmul_buf_fp32", {
    TS{"1.2.0", "matmul_buf", 2,1, 3,0,2,0, 2, "identity_fp32"},
    TS{"3.6.0", "matmul_buf", 2,1, 3,0,2,0, 2, "identity_fp32"},
}});
```

### 注册 adapter

在 `MnnBridge.cpp` 的 `registerMnnBridge()` 里调用 `registerXxxOp()`：
```cpp
MnnOps::registerRasterOp();      // 专用
MnnOps::registerUnaryOp();        // 专用
MnnOps::registerGenericOps();     // 通用（覆盖 ~40 变体）
registerFallbackAdapter();        // 兜底
```

### FallbackAdapter

没有专用/通用 adapter 匹配的 operator 走 FallbackAdapter——单 buffer、1D dispatch、identity validator。至少验证编译。

### findAdapter 逻辑

```cpp
// 1. 精确匹配 (opType, variant)
// 2. 找不到则走 __fallback__
```

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
9. **每个 (op_type, variant) 一个 adapter 类**，同 opType 多变体共存。优先用通用 adapter。
10. **cooperative matrix shader**（gemm_cm/sdpa_fa_cm/sdpa_cross_cm）需 `#extension GL_KHR_cooperative_matrix` + `--target-env vulkan1.1`，无 fallback 时标记 unsupported。
11. **OpenCL `-D` 不支持函数式宏**：`CONVERT_FLOAT4(x)` 等必须 `#define` 在 preamble 里，不能用 `-D` 传。
12. **OpenCL runner 已支持 compileMacros**：`clBuildProgram` 接受 `ac.compileMacros` 拼成的 build options 字符串。

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

## 已验证通过的 case（设备 Rhinopi Adreno 740）

### OpenCL (15/62)
- buffer_set_zero_fp32 (1.2.0/3.6.0) — 专用 adapter
- unary_buf_exp_fp32 (1.2.0/3.6.0) — 专用 adapter
- matmul_buf_nobias_fp32 (1.2.0) — 专用 adapter
- reduct_buf_sum_fp32 (1.2.0) — 专用 adapter
- pooling_max_fp32 (1.2.0) — 专用 adapter
- cast_buf_fp32 (3.6.0) — 通用 adapter
- select_buf_fp32 (3.6.0) — 通用 adapter

### Vulkan (8/361)
- sigmoid_fp32 / tanh_fp32 / permute_order0_fp32 (ncnn 20190611) — 专用 adapter
- sigmoid_pack4_fp32 / tanh_pack4_fp32 / absval_pack4_fp32 / relu_pack4_fp32_slope0 / concat_axis0_fp32 (ncnn 20260526) — 专用 adapter

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
