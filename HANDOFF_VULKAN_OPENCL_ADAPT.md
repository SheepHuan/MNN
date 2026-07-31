# GPU Kernel Corpus 适配完成记录 — OpenCL / Vulkan / CUDA

> 本文档记录接手 `HANDOFF.md` 后，在本机（有 glslangValidator + OpenCL/Vulkan runtime + nvcc）完成的全部适配工作。
> 参考 skill：`skills/kernel-adapt/SKILL.md`
> 推送目标：`origin/tmp`，最新 commit 含 CUDA 扩展（merge `kernel-agent-3.6.1` 后）

## 完成情况总览

| Backend / Framework | Tags | Cases | Compiled | Unsupported | Validation Passed | 备注 |
|---------------------|------|-------|----------|-------------|-------------------|------|
| OpenCL / mnn        | 1.2.0 + 3.6.0 | 62 | 55 | 7 | **53** | +2 (reduction_buf 3.6.0 修复) |
| Vulkan / mnn        | **1.2.0 + 3.6.0** | 23 | 23 | 0 | **23** | 12 个 3.6.0 + 5 个 1.2.0 |
| Vulkan / ncnn       | 20190611 + 20260526 | 361 | 358 | 0 | **358** | 3 个 cm shader unsupported |
| CUDA / mnn          | 1.2.0 ~ 3.6.0 | **309** | precompiled | 0 | **309** | **本次扩展 261 → 309 (+48, +18.4%)** |
| **总计** | — | **755** | — | — | **743** | — |

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

### 任务 5：CUDA 覆盖率扩展（本次，261 → 309，+48 cases）

**前置**：merge `kernel-agent-3.6.1` 后，CUDA 适配层已存在 28 个 `kernels/*.cu`、201 个 adapter、261 个 case 全部通过。本次在 sm_89 RTX 4080 + CUDA 12.5 + g++-12 host compiler 环境下扩展覆盖率。

#### P1: Raster 融合宏实例批量补齐 (+34 cases)

| 类别 | 新增 op | 数量 |
|------|---------|------|
| `raster_binary` | SUB / DIV / MINIMUM / MAXIMUM / FLOORDIV / FLOORMOD / SquaredDifference / POW | 8 |
| `raster_binary_fuseadd` | 同上 8 个 | 8 |
| `raster_binarymid` | 同上 8 个 + MUL_SILU | 9 |
| `raster_binarymidlinear4` | 同上 8 个 + MUL_SILU | 9 |

**重构**：`raster_fuse.cu` 的 extern "C" shim 区从手写改为宏驱动（`SHIM_BINARY` / `SHIM_BINARY_FUSEADD` / `SHIM_BINARYMID` / `SHIM_BINARYMIDLINEAR4`)，便于批量实例化。
**注意**: `raster_binary_fuseadd` 是 `atomicAdd(output, OP(x,y))`，非幂等，case 必须 `warmup_runs=0` 且用 `--kernel-corpus-runs 1` 运行（CLI 参数会覆盖 case 内 `workload_runs` 字段）。

#### P2: PACK_NUMBER 向量化变体 (+6 cases)

| 变体 | 数据类型 | PACK | 说明 |
|------|---------|------|------|
| `BinaryMid4_<ADD/MUL>` | fp32 | float4 | 3D stride,stride 不含 X 维（X 内嵌 `ix<<2`) |
| `BinaryMidHalf2_<ADD/MUL>` | fp16 | half2 | 同上，PACK_NUMBER=2 |
| `BinaryMidLinearHalf4_<ADD/MUL>` | fp16 | half2×2 | 1D linear，每线程 4 half |

**关键 fp16 validator 修复**：`output.data()` 是 raw bytes，需 `reinterpret_cast<const __half*>` + `__half2float` 才能与 fp32 expected 比较。

#### P3: BINARY_INT8 完整覆盖 (+6 cases)

| 类别 | 新增 |
|------|------|
| `BINARY_INT8` (single-scale) | SUB / DIV / MINIMUM / MAXIMUM |
| `BINARY_INT8_CHANNELWISE` (per-channel scale) | ADD / MUL |

**关键修复**：
1. `int8.cu` 的 `BINARY_INT8_ADD/MUL` 从手写重构为 `BINARY_INT8_FUNC` 宏驱动，便于批量实例化。
2. **`host_float2int_rn` 语义修正**：从 `(int)roundf` 改为 `(int)rintf`(banker's rounding),匹配 device `__float2int_rn` 的 round-to-nearest-even。原先 DIV `0.25*10=2.5` 时 host 期望 3、device 给 2 导致 false negative。
3. DIV 的输入数据 `in1[i] = 1 + (i % 5)` 避免 y=0 触发 inf/NaN 的 host-device 行为分歧。

#### P4: SPLIT_FusedKV (+2 cases)

`plugin/FmhaCommon/FmhaV2CommonExecution.cu:30`,fp32 + fp16 两 adapter,validator 检查 v 输出（k 不读回）。

#### 未覆盖项（已记录，留后续）

| 类别 | 数量 | 状态 | 备注 |
|------|------|------|------|
| 权重量化 GEMV/GEMM int4/int8 | ~20 | ⏸️ 跳过 | cutlass 集成复杂，一次性投入大 |
| Raster 融合剩余实例 | ~22 | ⏸️ 跳过 | GREATER/LESS/EQUAL/NOTEQUAL/LOGICALOR 返回 int 类型，需独立 validator;REALDIV/ATAN2/MOD 已存在或冗余 |
| BF16 原生算子 | 8 | ⏸️ 跳过 | 需 `MNN_CUDA_BF16=ON` 重编 backend；当前 sm89 可跑但需配置切换 |
| `BinaryMid4_` SUB/DIV/... 8 个 op | 8 | ⏸️ 可选 | 已覆盖 ADD/MUL 两个代表性，剩余宏实例遵循同一模式可批量加 |

**CUDA 验证命令**：
```bash
cd build && cmake --build . --target replay_benchmark.out -j$(nproc)
LD_LIBRARY_PATH=.:source/backend/cuda:. ./replay_benchmark.out \
  --kernel-corpus-bench --kernel-corpus-root ../replay_benchmark/kernel_corpus \
  --kernel-corpus-runs 1 --perf-counter-output /tmp/kc_cuda.json
# 309/309 passed
```

## 不可适配项验证（HANDOFF 是否高估）

| 项目 | HANDOFF 判断 | 实际验证 | 结论 |
|------|--------------|----------|------|
| OpenCL subgroup 9 个不可适配 | 正确 | 7 个确实不可适配（用 `INTEL_SUB_GROUP_READ*` Intel 专用扩展），1 个文件名带 subgroup 但源码无 subgroup API 实际通过，1 个数字错误 | ✅ 判断正确，数字错 |
| OpenCL image-based `scale_nobias_fp32` (1.2.0) | 不可适配 | 确实不可适配（kernel 用 `image2d_t`，runner 只支持 buffer），但状态是 `compiled/not_run` 而非 `unsupported` | ✅ 判断正确 |
| Vulkan/mnn 1.2.0 全不可适配 | **错误** | 41 个中 6 个 buffer-based，5 个已适配 | ❌ **过度估计** |
| CUDA host 196 unsupported | 预期 | 有 nvcc 后实际 261/261 通过，本次扩到 309/309 | ❌ **过度估计**（本机有 CUDA） |

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

### 6. CUDA validator 的 round-to-nearest-even

device 端 `__float2int_rn` 用 banker's rounding（如 `2.5 → 2`),host 端若用 `(int)roundf`（如 `2.5 → 3`）会导致 false negative。统一用 `(int)rintf`。

### 7. fp16 validator 的 output 解码

`output.data()` 是 raw bytes，需 `reinterpret_cast<const __half*>` + `__half2float` 转换回 fp32 才能与 expected 比较。

## 关键文件变更

| 用途 | 路径 | 变更 |
|------|------|------|
| Bake MNN Vulkan | `replay_benchmark/kernel_corpus/bake_mnn_vulkan_shaders.py` | OP_MAP 加 tag 维度、支持 1.2.0 路径、去重 #version、push_constant 正则兼容 readonly |
| Bake MNN OpenCL | `replay_benchmark/kernel_corpus/bake_mnn_kernels.py` | 3.6.0 reduction_buf 加 `OPERATE(a,b)` preamble |
| 3.6.0 adapter | `replay_benchmark/kernel_corpus_bridge/mnn/ops/VulkanMiscOps.hpp/.cpp` | **新增**，12 个 adapter |
| 1.2.0 adapter | `replay_benchmark/kernel_corpus_bridge/mnn/ops/VulkanTag120Ops.hpp/.cpp` | **新增**，5 个 adapter |
| OpenCL reduction | `replay_benchmark/kernel_corpus_bridge/mnn/ops/ElementwiseOps.cpp` | 1.2.0/3.6.0 compileMacros 分支修复 |
| Bridge 注册 | `replay_benchmark/kernel_corpus_bridge/mnn/MnnBridge.cpp` | 注册 `registerVulkanMiscOps()` + `registerVulkanTag120Ops()` |
| CMake | `replay_benchmark/CMakeLists.txt` | 加新 .cpp、CudaOps 包到 MNN_CUDA 块、CUDA 12.5 + GCC 13 host compiler 兼容、`-Xcompiler -fexceptions` 镜像 MNN backend 覆盖、传播 `CUDA_ARCH_FLAGS` 启用 `__hfma2` |
| **CUDA kernel 扩展** | `replay_benchmark/kernel_corpus_bridge/cuda/kernels/raster_fuse.cu` | extern "C" shim 宏重构（`SHIM_BINARY`/`SHIM_BINARY_FUSEADD`/`SHIM_BINARYMID`/`SHIM_BINARYMIDLINEAR4`)，新增 36 个 shim；新增 BinaryMid4/Half2/LinearHalf4 6 个 shim |
| **CUDA kernel 扩展** | `replay_benchmark/kernel_corpus_bridge/cuda/kernels/int8.cu` | `BINARY_INT8_ADD/MUL` 重构为 `BINARY_INT8_FUNC` 宏驱动，新增 SUB/DIV/MINIMUM/MAXIMUM；新增 CHANNELWISE ADD/MUL |
| **CUDA kernel 扩展** | `replay_benchmark/kernel_corpus_bridge/cuda/kernels/plugins.cu` | 新增 `SPLIT_FusedKV` kernel + fp32/fp16 shim |
| **CUDA adapter** | `replay_benchmark/kernel_corpus_bridge/cuda/CudaOps.hpp` | 新增 5 个 `DECL_*_ADAPTER` 宏（批量声明类）;48 个新 adapter 类声明 |
| **CUDA adapter** | `replay_benchmark/kernel_corpus_bridge/cuda/CudaOpsFp16.cu` | 48 个 adapter 实现（宏驱动）+ 注册；**`host_float2int_rn` 改为 `(int)rintf`** |
| cases | `replay_benchmark/kernel_corpus/operator_cases.json` | 755 cases（其中 CUDA 309) |
| operators.json | `replay_benchmark/kernel_corpus/operators.json` | 751 entries（其中 CUDA 309 + 历史 442) |

## 验证命令

### OpenCL / Vulkan (host, 无 nvcc)

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
```

### CUDA (host with nvcc)

```bash
cd build
cmake .. -DMNN_CUDA=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_REPLAY_ENABLE_PERFCOUNTER=ON
cmake --build . --target replay_benchmark.out -j$(nproc)

LD_LIBRARY_PATH=.:source/backend/cuda:. ./replay_benchmark.out \
  --kernel-corpus-bench --kernel-corpus-root ../replay_benchmark/kernel_corpus \
  --kernel-corpus-runs 1 --perf-counter-output /tmp/kc_cuda.json
# 309/309 passed (runs=1 因 fuseadd 非幂等)
```

### 统计脚本

```bash
python3 -c "
import json
from collections import Counter
d=json.load(open('/tmp/kc_cuda.json'))
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

## 提交记录（含本次）

```
[本次] [GPU:Feature] CUDA kernel corpus 扩展 261→309 (+48 cases) — Raster 宏实例批量补齐 + BinaryMid4/Half2/LinearHalf4 + BINARY_INT8 完整覆盖 + SPLIT_FusedKV
7cf11202b Merge branch 'kernel-agent-3.6.1' into tmp
92d5b64af [GPU:Feature] Add Vulkan/mnn 1.2.0 adapter coverage (5 buffer-based variants)
14d5c3aa3 [GPU:Chore] Add pre-compiled ncnn source .spv artifacts for runtime loading
234ff746b [GPU:Feature] Complete Vulkan/mnn 3.6.0 adapter coverage + rebake ncnn .spv + fix OpenCL reduction 3.6.0
b7d9fee1b [Doc:Chore] Add GPU kernel corpus adapter handoff doc for OpenCL/Vulkan remaining work
950432ed3 [GPU:Feature] Complete CUDA kernel corpus: fix 6 compile errors + add 29 kernel variants — 196/196 pass
```

## 后续工作建议

1. **OpenCL image-based kernel 支持**：若需要让 `scale_nobias_fp32` (1.2.0) 等 image-based kernel 跑起来，runner 需要加 `clCreateImage2D` 支持。这超出 HANDOFF 范围。
2. **Vulkan/mnn 1.2.0 `buffer2Image1D.comp`**：同上，需要 image 支持。
3. **更多 3.6.0 shader**：HANDOFF 标记"低优先级"的 `topkv2.comp` / `gemm_m8n4.comp` / `matmulunit.comp` 因签名复杂未适配，可按需补做。
4. **CUDA BF16 原生算子**：8 个 (`CONV_DW_BF16` / `WeightTransToBf16` / `maxpool_C8_BF16` 等），需 `MNN_CUDA_BF16=ON` 重编 backend;sm_89 已可跑。
5. **CUDA 权重量化 GEMV/GEMM int4/int8** (~20 个）:`GEMV_FpAInt4B/V5/V9` / `GEMM_FpAInt4B/Int8B` / `Rearrange_Weight_Int4/Int8` / `Precompute*` / `QuantA` / `DequantAndAcc` / `BiasAndActivation`,cutlass 集成复杂，一次性投入大。
6. **CUDA Raster 剩余宏实例** (~22 个）：返回 int 类型的 GREATER/LESS/EQUAL/NOTEQUAL/LOGICALOR，需独立 validator 处理 int 输出；以及 `BinaryMid4_` 的 SUB/DIV/MIN/MAX 等扩展（已覆盖 ADD/MUL 两个代表性，剩余遵循同一模式可批量加）。
7. **fuseadd runner framework 改造**：当前 `raster_binary_fuseadd` 因 `atomicAdd` 非幂等只能在 `--kernel-corpus-runs 1` 下验证。若要让这类 case 在 runs>1 下也能验证，需在 runner 中支持"每次 dispatch 前重置 output buffer 为 initialData"(in-place 模式）。
