---
name: kernel-adapt
description: 将历史 MNN/ncnn 的 OpenCL/Vulkan kernel 适配到当前 MNN GPU runtime 运行。覆盖 kernel 提取、baked 文件生成、版本桥接器(OpAdapter)实现、case 注册、设备验证的完整 TDD 流程。
---

# 历史 GPU Kernel 适配 SKILL

> **触发条件**：当用户请求适配/运行历史 kernel、扩展算子变体覆盖、为 kernel corpus 新增 adapter 时触发。

## 概述

本 SKILL 指导 AI Agent 将 `replay_benchmark/kernel_corpus/sources/` 中的历史 MNN OpenCL `.cl` 和 ncnn Vulkan `.comp` kernel 适配为可在当前 MNN GPU runtime 上编译、dispatch、校验和采集 PMU 的独立 workload。

**不编译历史 backend，不链接历史 C++，不复用历史 allocator/scheduler。** 历史源码只作为 kernel 文本来源。

## 核心架构

```
operator_cases.json (声明 case 语义)
        ↓
Bridge::adapt() → AdaptedCase (完全准备好的 args/buffers/dispatch)
        ↓
Runner (通用,只认 AdaptedCase)
  OpenCL: clBuildProgram → setArg → dispatch → readback → PMU
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

kernel_corpus_bridge/
  Bridge.hpp                          # AdaptedCase + Bridge 接口 + validators
  Bridge.cpp                          # 注册表 + validator 实现
  OpAdapter.hpp                       # OpAdapter 接口 + FallbackAdapter
  OpAdapter.cpp                       # FallbackAdapter 实现
  mnn/
    MnnBridge.hpp/.cpp                # 顶层分发 + 注册 OpAdapter
    ops/
      RasterOp.hpp/.cpp               # 每个 (op_type, variant) 一个类
      UnaryOp.hpp/.cpp
      MatmulOp.hpp/.cpp
      ReductionOp.hpp/.cpp
      PoolingOp.hpp/.cpp
  ncnn/
    NcnnBridge.hpp/.cpp
    ops/
      ElementwiseOp.hpp/.cpp          # 20190611 sigmoid/tanh (标量 sfp)
      Pack4Op.hpp/.cpp                # 20260526 sigmoid/tanh/absval/relu (pack4 sfpvec4)
```

## 步骤 1: Bake Kernel 源码

### ncnn Vulkan (.comp)

```bash
python3 replay_benchmark/kernel_corpus/bake_ncnn_shaders.py \
  --sources replay_benchmark/kernel_corpus/sources \
  --operators replay_benchmark/kernel_corpus/operators
```

脚本自动：
- `#undef` 所有 NCNN_fp16_*/NCNN_int8_*/ncnn_VK_* 宏（预处理器自然走 FP32 `#else` 分支）
- 内联 `#include "vulkan_activation.comp"`
- 预置 FP32 macro preamble（sfp/afp/buffer_ld4/st4/psc + int8）
- 跳过 `vulkan_activation.comp`（include 头不是 compute shader）

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

只处理 `*_buf.cl`/`*_subgroup_buf.cl`（buffer-only，无 image2d_t）。预置 FP32 macro preamble（FLOAT/GLOBAL_SIZE_DIMS/DEAL_NON_UNIFORM）。

## 步骤 2: 重新生成 operators.json

```bash
python3 replay_benchmark/kernel_corpus/extract_operator_kernels.py \
  --root replay_benchmark/kernel_corpus \
  --output replay_benchmark/kernel_corpus/operators.json
```

## 步骤 3: 实现 OpAdapter

### 新增一个变体 adapter

每个 `(op_type, variant)` 对一个类。例如 `unary_buf_exp_fp32`：

**mnn/ops/UnaryOp.hpp**:
```cpp
class UnaryBufExpOp : public OpAdapter {
public:
    const char* opType() const override { return "unary"; }
    const char* variant() const override { return "unary_buf_exp_fp32"; }
    bool adapt(const CaseSpec& spec, AdaptedCase& ac) const override;
};
void registerUnaryOp();
```

**mnn/ops/UnaryOp.cpp**:
```cpp
bool UnaryBufExpOp::adapt(const CaseSpec& spec, AdaptedCase& ac) const {
    ac.entry = "unary_buf";
    // 1. 填充 input data
    // 2. 创建 AdaptedBuffer (isOutput=true/false)
    // 3. 按 kernel 参数顺序填 ac.args (SizeConst/Buffer/Scalar/Int2)
    // 4. 设 globalSize/dims
    // 5. 设 validatorInputA 供 validator 用
    return true;
}
```

### AdaptedArg 类型

| Kind | 用途 | runner 行为 |
|------|------|------------|
| `SizeConst(dim)` | OpenCL GLOBAL_SIZE_DIMS 展开的 `__private const int` | `setArg(i, globalSize[dim])` |
| `Buffer(idx)` | `__global FLOAT*` / `layout(binding) buffer` | `setArg(i, clBuffer)` / `writeBuffer(descSet)` |
| `Scalar(Int/Float)` | `__private const int` 等标量 | `setArg(i, val)` |
| `Int2(x,y)` | `int2` 参数（如 pooling shape） | `setArg(i, cl_int2{x,y})` |

### tag 版本差异

同一 variant 在不同 tag 下参数布局可能不同。在 `adapt()` 内部用 `if (spec.tag == "1.2.0")` 分支：

```cpp
if (spec.tag == "1.2.0") {
    // 1.2.0: unary_buf(GLOBAL_SIZE_3_DIMS, input, output, height)
    ac.args.push_back(AdaptedArg::sizeConst(2));  // dim2
    ac.dims = 3;
} else {
    // 3.6.0: unary_buf(GLOBAL_SIZE_2_DIMS, input, output, size)
    ac.args.push_back(AdaptedArg::scalarInt(count));
    ac.dims = 2;
}
```

### 注册 adapter

在 `MnnBridge.cpp` 的 `registerMnnBridge()` 里调用 `registerXxxOp()`：
```cpp
MnnOps::registerRasterOp();
MnnOps::registerUnaryOp();
MnnOps::registerMatmulOp();
registerFallbackAdapter();  // 兜底
```

### FallbackAdapter

没有专用 adapter 的 operator 走 FallbackAdapter——单 buffer、1D dispatch、identity validator。至少验证编译。

## 步骤 4: 更新 operator_cases.json

每个 operator 至少一个 case。已有专用 adapter 的用专用参数，其余用 fallback 默认值。

```bash
python3 -c "
import json
ops = json.load(open('replay_benchmark/kernel_corpus/operators.json'))
existing = json.load(open('replay_benchmark/kernel_corpus/operator_cases.json'))
# merge: 保留已有 case 参数，新 operator 用默认值
...
"
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

## 关键约束

1. **不 fallback 简化实现**：baked kernel 必须功能上与原版一致，不自己简化或替换逻辑。cooperative-matrix-only shader（如 gemm_cm）标记为 unsupported，不篡改源码。
2. **不编译历史 backend**：只提取 kernel 文本，用当前 MNN OpenCL/Vulkan runtime 编译。
3. **禁止访问** `schema/private/` 和 `source/internal/`。
4. **不批量格式化历史源码**，不修改尾随空格。
5. **C++ 遵循** C++11、4 空格缩进、禁止 namespace 作用域动态初始化对象。
6. **PMU 只包围 dispatch**：warmup 在 PMU session 外，workload dispatch 在 session 内。
7. **in-place shader**（sigmoid/tanh）每次 dispatch 前重写 initialData，保证 validation 对比原始输入。
8. **每个 (op_type, variant) 一个 adapter 类**，同 opType 多变体共存。
9. **cooperative matrix shader**（gemm_cm/sdpa_fa_cm/sdpa_cross_cm）需 `#extension GL_KHR_cooperative_matrix` + `--target-env vulkan1.1`，无 fallback 时标记 unsupported。

## 已知不支持的 kernel

| 文件 | 原因 | 处理 |
|------|------|------|
| gemm_cm.comp | cooperative-matrix-only，PAD/sum 在 coop matrix 分支内 | 标记 unsupported |
| sdpa_fa_cm.comp | 同上 | 标记 unsupported |
| sdpa_cross_cm.comp | 同上 | 标记 unsupported |
| MNN *.cl (非 _buf) | 使用 image2d_t，runner 暂不支持 image | 不 bake |

## 设备信息

- **Rhinopi-X1**：`root@192.168.101.227`，`/mnt/nvme/workspace/replay-benchmark`，Adreno 740
  - Vulkan lib: `/mnt/nvme/workspace/replay-benchmark-vulkan/lib/libvulkan.so`
- **OrangePi**：`root@192.168.101.113`，`/mnt/ssd/workspace`，Mali-G610
- 凭据只从环境读取，不写入仓库
