# GPU Kernel Corpus 适配完成记录 — OpenCL / Vulkan

> 本文档记录接手 `HANDOFF.md` 后，在本机（有 glslangValidator + OpenCL/Vulkan runtime）完成的全部适配工作。
> 参考 skill：`skills/kernel-adapt/SKILL.md`
> 推送目标：`origin/tmp`，最终 commit `92d5b64af`

## 完成情况总览

| Backend / Framework | Tags | Cases | Compiled | Unsupported | Validation Passed | HANDOFF 基线 | 备注 |
|---------------------|------|-------|----------|-------------|-------------------|--------------|------|
| OpenCL / mnn        | 1.2.0 + 3.6.0 | 62 | 55 | 7 | **53** | 51 | +2 (reduction_buf 3.6.0 修复) |
| Vulkan / mnn        | **1.2.0 + 3.6.0** | 23 | 23 | 0 | **23** | 6 (3.6.0) | 新增 12 个 3.6.0 + 5 个 1.2.0 |
| Vulkan / ncnn       | 20190611 + 20260526 | 361 | 358 | 0 | **358** | 14 | compile_failed 353→3 (cm shader) |
| CUDA / mnn          | 多 tag | 196 | not_found | 196 | 0 | 0 | host 无 nvcc，预期 |
| **总计** | — | **642** | — | — | **434** | — | — |

## 完成的工作

### 任务 1：Rebake 全部 ncnn Vulkan .spv（358/361）

**问题**：HANDOFF 撰写时 host 无 `glslangValidator`，导致 ncnn 361 个 `.comp` 只有 14 个预编译 `.spv`，运行时 353 个 `compile_failed`。

**做法**：
1. 安装 `glslang-tools`（`/usr/bin/glslangValidator` 15.1.0）
2. 跑 `bake_ncnn_shaders.py` 生成全部 `.comp`
3. 用 `xargs -P 32` 并行调 `glslangValidator -V --target-env vulkan1.1` 批量编译 `.spv`

**结果**：358/361 `.spv` 生成成功。3 个失败均为 cooperative-matrix shader（`gemm_cm` / `sdpa_fa_cm` / `sdpa_cross_cm`），SKILL 明确标记为 unsupported，预期内。

**验证**：host benchmark 显示 Vulkan `compile_failed` 从 353 → 3。

### 任务 2：扩展 Vulkan/mnn 3.6.0 adapter（6 → 18 variants）

**问题**：HANDOFF 时 `bake_mnn_vulkan_shaders.py` 的 `OP_MAP` 只覆盖 5 个 op / 6 个 variant（unary/binary/raster/reduction/pooling），但 3.6.0 有 93 个 buffer-based `.comp` 可用。

**做法**：新增 `VulkanMiscOps.hpp/.cpp`，实现 12 个新 adapter：

| Adapter | Shader | 数据布局 |
|---------|--------|----------|
| `VulkanSelectKernel` | `select.comp` | 1 out + 2 in + 1 sel (int) + ivec4 size |
| `VulkanRangeKernel` | `range.comp` | 1 out + 1 start + 1 delta + ivec4 size |
| `VulkanCastFloatIntKernel` | `cast_float_int.comp` | vec4 → ivec4（输出按 int32 校验） |
| `VulkanScaleKernel` | `scale.comp` | NC4HW4, scale/bias per channelC4 |
| `VulkanPreluKernel` | `preluWithChannel.comp` | NC4HW4, slope per channelC4 |
| `VulkanArgmaxKernel` | `argmax.comp` | out int32, inside/axis/outside/reduceAxis |
| `VulkanSoftmaxHeightKernel` | `softmaxHeight_NHWC.comp` | softmax along H, NHWC |
| `VulkanNormKernel` | `norm.comp` | LayerNorm, USE_RMS=0, no gamma/beta |
| `VulkanResizeNearestKernel` | `resizeNearest.comp` | NC4HW4, 2D dispatch |
| `VulkanResizeBilinearKernel` | `resizeBilinear.comp` | NC4HW4, 双线性 |
| `VulkanGridSampleNearestKernel` | `gridSampleNearest.comp` | NC4HW4 + grid |
| `VulkanNc4hw4ToNchwKernel` | `nc4hw4Tonchw.comp` | 布局转换 |

**关键修复**：
1. **bake 脚本 push_constant 正则**：原只匹配 `uniform constBuffer`，现兼容 `readonly uniform constBuffer` 和自定义 block 名（`reluBuffer` / `gridSampleBuffer`）。
   ```python
   # 原：r'layout\(set=0,binding=N\) uniform constBuffer'
   # 新：r'layout\(set=0,binding=N\) (readonly )?uniform \w+'
   ```
2. **CMake**：把 `CudaOps.cpp/CudaOpsMisc.cpp` 包到 `MNN_CUDA` 条件块，使 host 无 nvcc 时也能 link。
3. **binding 顺序**：按 shader 声明顺序映射 `ac.vulkanBindings`，如 select.comp 是 `0=output, 1=select, 2=in0, 3=in1`，不能按"输入在前"的习惯。

**验证**：18/18 `validation_passed`。

### 任务 3：修复 OpenCL reduction_buf 3.6.0 compile_failed

**问题**：HANDOFF 基线 51/62，其中 `reduction_buf_fp32 (3.6.0)` compile_failed。

**根因**：3.6.0 用函数式宏 `OPERATE(out, in)`，OpenCL `-D` 不支持函数式宏。bake 的 1.2.0 用 `#define OPERATE (num+in)`（标量表达式，可 `-D` 传），但 3.6.0 需要 `OPERATE(a,b)` 函数式。

**修复**：
1. `bake_mnn_kernels.py`：对 3.6.0 的 `reduction_buf.cl` 在 preamble 加 `#define OPERATE(a,b) ((a)+(b))`
2. `ElementwiseOps.cpp` adapter：3.6.0 分支加 `-DREDUCT_LOCAL_SIZE=256 -DVALUE=0`；1.2.0 分支保留 `-DOPERATE=(num+in)`

**结果**：OpenCL validation_passed 51 → 53。

### 任务 4：补完 Vulkan/mnn 1.2.0（HANDOFF 误标为"不可适配"）

**HANDOFF 错误**：任务 4 声称 1.2.0 Vulkan 全是 image-based shader，不可适配。

**实际验证**：`sources/mnn/1.2.0/source/backend/vulkan/execution/glsl/` 共 41 个 `.comp`，其中 6 个是 buffer-based（5 个纯 buffer + 1 个 image+buffer 混合）：

| Shader | 类型 | 状态 |
|--------|------|------|
| `blit.comp` | 纯 buffer | ✅ 适配 `vulkan_blit_120_fp32` |
| `nc4hw4Tonchw.comp` | 纯 buffer | ✅ 适配 `vulkan_nc4hw4_to_nchw_120_fp32` |
| `nchwTonc4hw4.comp` | 纯 buffer | ✅ 适配 `vulkan_nchw_to_nc4hw4_120_fp32` |
| `reduce.comp` | 纯 buffer | ✅ 适配 `vulkan_reduce_sum_120_fp32`（SUM 变体） |
| `softmaxHeight_NHWC.comp` | 纯 buffer | ✅ 适配 `vulkan_softmax_height_120_fp32` |
| `buffer2Image1D.comp` | imageStore | ❌ 跳过（runner buffer-only） |

**做法**：新增 `VulkanTag120Ops.hpp/.cpp`，5 个 adapter 类。

**关键适配点**：
1. **glsl 目录差异**：1.2.0 用 `execution/glsl/`，3.6.0 用 `buffer/execution/glsl/`。bake 脚本按 tag 选择路径。
2. **重复 #version**：1.2.0 shader 第一行有 `#version 440 core`，与 bake preamble 的 `#version 450` 冲突。bake 时用 `re.sub(r'^\s*#version[^\n]*\n', '', raw, count=1)` 去掉。
3. **OP_MAP 加 tag 维度**：每个 variant tuple 改为 `(src_file, variant_name, macros, [tags])`。
4. **1.2.0 数据类型**：1.2.0 shader 直接用 `float`/`vec4`，不用 `FLOAT` 宏，但 bake preamble 仍然兼容（`#define FLOAT float` 是 no-op）。

**验证**：5/5 `validation_passed`。

## 不可适配项验证（HANDOFF 是否高估）

| 项目 | HANDOFF 判断 | 实际验证 | 结论 |
|------|--------------|----------|------|
| OpenCL subgroup 9 个不可适配 | 正确 | 7 个确实不可适配（用 `INTEL_SUB_GROUP_READ*` Intel 专用扩展），1 个文件名带 subgroup 但源码无 subgroup API 实际通过，1 个数字错误 | ✅ 判断正确，数字错 |
| OpenCL image-based `scale_nobias_fp32` (1.2.0) | 不可适配 | 确实不可适配（kernel 用 `image2d_t`，runner 只支持 buffer），但状态是 `compiled/not_run` 而非 `unsupported` | ✅ 判断正确 |
| Vulkan/mnn 1.2.0 全不可适配 | **错误** | 41 个中 6 个 buffer-based，5 个已适配 | ❌ **过度估计** |
| CUDA host 196 unsupported | 预期 | 无 nvcc，host build 不含 CUDA，预期 | ✅ 正确 |

## 关键技术点

### 1. Vulkan push_constant 布局

MNN shader 用 `layout(set=0,binding=N) uniform constBuffer` 声明常量，runner 用 `vkCmdPushConstants` 传参。bake 时必须把声明替换为 `layout(push_constant) uniform constBuffer`，否则 SPIR-V 与 runner 不匹配。

### 2. Vulkan binding 顺序 ≠ buffers 顺序

shader 的 `binding=N` 声明顺序与 `ac.buffers` 数组顺序不一定一致。`ac.vulkanBindings[i]` 表示 `buffers[i]` 对应的 binding 号。必须按 shader 声明的 binding 顺序映射，否则 kernel 写入 input、output 全 0。

### 3. OpenCL `-D` 不支持函数式宏

`CONVERT_FLOAT4(x)`、`OPERATE(a,b)` 等函数式宏必须 `#define` 在 bake preamble 里，不能用 `-D` 传。1.2.0 的 `OPERATE` 是标量表达式 `(num+in)` 可以 `-D` 传，3.6.0 的 `OPERATE(a,b)` 必须 bake 进 `.cl`。

### 4. 1.2.0 与 3.6.0 shader 差异

| 维度 | 1.2.0 | 3.6.0 |
|------|-------|-------|
| 数据类型 | 直接 `float`/`vec4` | `FLOAT`/`FLOAT4` 宏 |
| `#version` | 首行 `#version 440 core` | 无（bake 加 `#version 450`） |
| 函数体 | 部分与 3.6.0 略不同（如 softmax maxValue 初始 -1000） | 用第一个元素初始化 |
| glsl 路径 | `execution/glsl/` | `buffer/execution/glsl/` |

### 5. 函数体差异即独立 kernel

SKILL 规则：即使签名相同，`__global__` 函数体有实质差异就必须独立 kernel + 独立 shim。1.2.0 的 softmax 与 3.6.0 的差异（maxValue 初始值）足够小，可以共用 validator，但 adapter 和 `.spv` 必须独立。

## 关键文件变更

| 用途 | 路径 | 变更 |
|------|------|------|
| Bake MNN Vulkan | `replay_benchmark/kernel_corpus/bake_mnn_vulkan_shaders.py` | OP_MAP 加 tag 维度、支持 1.2.0 路径、去重 #version、push_constant 正则兼容 readonly |
| Bake MNN OpenCL | `replay_benchmark/kernel_corpus/bake_mnn_kernels.py` | 3.6.0 reduction_buf 加 `OPERATE(a,b)` preamble |
| 3.6.0 adapter | `replay_benchmark/kernel_corpus_bridge/mnn/ops/VulkanMiscOps.hpp/.cpp` | **新增**，12 个 adapter |
| 1.2.0 adapter | `replay_benchmark/kernel_corpus_bridge/mnn/ops/VulkanTag120Ops.hpp/.cpp` | **新增**，5 个 adapter |
| OpenCL reduction | `replay_benchmark/kernel_corpus_bridge/mnn/ops/ElementwiseOps.cpp` | 1.2.0/3.6.0 compileMacros 分支修复 |
| Bridge 注册 | `replay_benchmark/kernel_corpus_bridge/mnn/MnnBridge.cpp` | 注册 `registerVulkanMiscOps()` + `registerVulkanTag120Ops()` |
| CMake | `replay_benchmark/CMakeLists.txt` | 加新 .cpp、CudaOps 包到 MNN_CUDA 块 |
| cases | `replay_benchmark/kernel_corpus/operator_cases.json` | +17 case (12 个 3.6.0 + 5 个 1.2.0) |
| operators.json | `replay_benchmark/kernel_corpus/operators.json` | +17 entry |

## 验证命令

```bash
# Host build
rm -rf build-host && mkdir build-host && cd build-host
cmake .. -DMNN_OPENCL=ON -DMNN_VULKAN=ON -DMNN_REPLAY_ENABLE_PERFCOUNTER=OFF \
  -DMNN_SEP_BUILD=ON -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_BENCHMARK=ON \
  -DMNN_BUILD_TEST=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target replay_benchmark.out -j$(nproc)

# Host benchmark
export LD_LIBRARY_PATH=$(pwd)/build-host/source/backend/opencl:$(pwd)/build-host/source/backend/vulkan:$(pwd)/build-host
./build-host/replay_benchmark.out \
  --kernel-corpus-bench --kernel-corpus-root replay_benchmark/kernel_corpus \
  --kernel-corpus-runs 5 --perf-counter-output /tmp/kc.json

# 统计
python3 -c "
import json
from collections import Counter
d=json.load(open('/tmp/kc.json'))
cs=d['cases']
for b in ['opencl','vulkan','cuda']:
    bc=[c for c in cs if c['backend']==b]
    if not bc: continue
    comp=Counter(c['compile_status'] for c in bc)
    disp=sum(1 for c in bc if c['dispatch_status']=='dispatched')
    val=sum(1 for c in bc if c['validation_status']=='validation_passed')
    print(f'  {b}: {len(bc)} | compile={dict(comp)} dispatched={disp} validation_passed={val}')
"
```

## 提交记录

```
92d5b64af [GPU:Feature] Add Vulkan/mnn 1.2.0 adapter coverage (5 buffer-based variants)
14d5c3aa3 [GPU:Chore] Add pre-compiled ncnn source .spv artifacts for runtime loading
234ff746b [GPU:Feature] Complete Vulkan/mnn 3.6.0 adapter coverage + rebake ncnn .spv + fix OpenCL reduction 3.6.0
b7d9fee1b [Doc:Chore] Add GPU kernel corpus adapter handoff doc for OpenCL/Vulkan remaining work
950432ed3 [GPU:Feature] Complete CUDA kernel corpus: fix 6 compile errors + add 29 kernel variants — 196/196 pass
```

## 后续工作建议

1. **OpenCL image-based kernel 支持**：若需要让 `scale_nobias_fp32` (1.2.0) 等 image-based kernel 跑起来，runner 需要加 `clCreateImage2D` 支持。这超出 HANDOFF 范围。
2. **Vulkan/mnn 1.2.0 `buffer2Image1D.comp`**：同上，需要 image 支持。
3. **CUDA host 验证**：当前 host 无 nvcc，196 个 case 预期 unsupported。在 CUDA 设备上跑通后可确认 196/196。
4. **更多 3.6.0 shader**：HANDOFF 标记"低优先级"的 `topkv2.comp` / `gemm_m8n4.comp` / `matmulunit.comp` 因签名复杂未适配，可按需补做。
