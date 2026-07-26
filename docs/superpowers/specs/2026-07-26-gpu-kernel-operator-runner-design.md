# GPU Kernel Operator Runner Design

## Goal

将 `replay_benchmark/kernel_corpus/sources` 中的历史 MNN OpenCL 和 ncnn/MNN Vulkan
kernel 实现提取为可描述的单算子 workload，并使用当前 MNN 的 GPU runtime 通用地
编译、dispatch、校验和采集 PMU；不编译、不链接历史 backend。

## Scope

第一阶段覆盖：

- OpenCL C kernel 的源码发现、entry point 发现和执行描述生成；
- Vulkan GLSL compute shader 的源码发现、entry point/资源接口发现和执行描述生成；
- 当前 `replay_benchmark` 的通用 runner；
- 最小可靠 smoke case：MNN `buffer_set_zero`、`unary_buf`、`matmul_buf`，以及 ncnn
  `sigmoid`、`tanh`、`permute`；
- 输出 compile、dispatch、校验和可选 PMU 区间信息。

不在第一阶段范围内：

- 编译历史 MNN/ncnn backend C++；
- 自动推断所有 kernel 的业务参数、张量布局和正确答案；
- 直接复用历史 backend 的 allocator、scheduler 或 shader registry；
- 在没有执行描述的情况下批量运行任意历史 kernel。

## Architecture

### 1. Source extractor

`replay_benchmark/kernel_corpus/extract_operator_kernels.py` 扫描 manifest 指向的
源码快照，生成 `operators.json`。每条记录包含：

```json
{
  "framework": "mnn",
  "tag": "1.2.0",
  "backend": "opencl",
  "operator": "unary_buf",
  "source": "sources/mnn/1.2.0/source/backend/opencl/execution/cl/unary_buf.cl",
  "entry": "unary_buf",
  "language": "opencl",
  "required_extensions": [],
  "execution_spec": "manual"
}
```

提取器只负责发现事实，不猜测参数。OpenCL 的 `__kernel` entry point 通过源码扫描
发现；Vulkan 的 `void main()` shader 以文件名和相对路径作为稳定 operator key，并
额外记录 descriptor binding、push constant 和 specialization constant 的声明文本。

### 2. Execution descriptors

自动发现记录和可运行记录分离。`operators.json` 保存全部候选 kernel；
`operator_cases.json` 只保存有明确输入布局、参数、dispatch geometry 和 reference
output 的 smoke case。这样没有适配信息的历史 kernel 仍可检索，但不会被误执行。

OpenCL case 使用 buffer/image binding、scalar 参数、源码前缀宏和 global/local size。
Vulkan case 使用 storage buffer binding、push constant bytes、specialization values、
SPIR-V 编译选项和 dispatch size。描述中的数据类型和布局使用固定枚举，不允许把任意
用户输入拼接到 shader 编译选项中。

### 3. Generic runner

runner 放在 `replay_benchmark`，只依赖当前构建中的 MNN OpenCL/Vulkan runtime：

- OpenCL：通过当前 wrapper 创建 context、program、kernel、buffer/image，设置描述中
  的参数并 enqueue；源码前缀只提供当前 case 声明的宏。
- Vulkan：通过当前 Vulkan device/pipeline abstraction 编译 GLSL 得到 SPIR-V，创建
  descriptor set、pipeline layout、compute pipeline，写入 buffer/push constants 后
  dispatch。
- runner 统一执行 warmup → control（可选）→ workload → finish/readback，并将结果
  校验和 PMU 采样输出为 JSON；后续 sweep 可以将 JSON 合并到现有 CSV。

历史版本的 wrapper、allocator、execution C++ 不进入 runner；历史源码只作为 kernel
文本和元数据来源。

## Error handling

- 缺少 source、entry、descriptor 或 execution spec：记录 `not_runnable`，不启动 GPU；
- 编译失败：记录完整 compiler log 和 `compile_failed`；
- runtime capability 不满足：记录 `unsupported`；
- dispatch/readback/reference 不匹配：记录 `validation_failed`；
- 单 case 失败不终止同一版本下其他 case。

## Verification

- Python 单元测试验证 OpenCL/Vulkan entry 提取、路径约束、manifest 关联和 descriptor
  schema；
- host 上验证 fixture shader 的编译描述和缺失能力分类；
- OpenCL 真机验证至少一个 buffer kernel；Vulkan 真机验证至少一个 storage-buffer
  shader；
- 真机输出必须包含 `framework/tag/backend/operator/status/dispatch/validation`，并能
  被现有 PMU sweep 读取；
- 不宣称历史 kernel 与当前 runtime 的数值语义一致，除非 reference 校验通过。
