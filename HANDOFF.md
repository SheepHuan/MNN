# GPU Kernel Corpus 适配接手文档

> 本文档由上一轮 agent 在**无法安装 glslangValidator / 无 CUDA 编译环境**的设备上盘点后撰写。
> 目标：让接手者在**有完整工具链的设备**上完成 OpenCL / Vulkan 剩余适配工作。
> 参考 skill：`skills/kernel-adapt/SKILL.md`（完整架构 + bake + adapter + 验证流程）

## 当前分支与基线

- 分支：`kernel-agent-3.6.1`（推送目标：`origin/tmp`）
- 最后提交：`950432ed3 [GPU:Feature] Complete CUDA kernel corpus: fix 6 compile errors + add 29 kernel variants (A/B/C/D class) — 196/196 pass`
- 工作树干净（无未提交改动）

## 上次 host 验证基线（NVIDIA L40S，无 glslangValidator，host build 不含 CUDA）

```
total: 625 cases
  opencl:  62 cases | compiled=52, unsupported=10 | dispatched=51 | validation_passed=51
  vulkan: 367 cases | compiled=14, compile_failed=353 | dispatched=14 | validation_passed=14
  cuda:   196 cases | unsupported=196 (host build 不含 CUDA，预期) | dispatched=0
```

**注意**：Vulkan 的 353 个 `compile_failed` **全部是 `glslangValidator: not found` 导致**，不是代码问题。装好 `glslang-tools` 后，已有 `.spv` 的 14 个会继续通过，其余 353 个 ncnn shader 需要重新 bake `.spv`（见下"任务 1"）。

---

## 未完成工作清单

### 任务 1（前置，必做）：安装工具链 + 重新 bake 全部 ncnn Vulkan .spv

**问题**：`replay_benchmark/kernel_corpus/operators/vulkan/**/*.spv` 只有 14 个（已验证通过的），其余 353 个 ncnn shader 没有 `.spv`，运行时调 `glslangValidator` 失败。

**步骤**：

```bash
# 1. 装 glslangValidator
sudo apt-get install -y glslang-tools
which glslangValidator  # 确认 /usr/bin/glslangValidator

# 2. 重新 bake ncnn Vulkan shader（生成全部 .comp + .spv）
cd /path/to/MNN
python3 replay_benchmark/kernel_corpus/bake_ncnn_shaders.py \
  --sources replay_benchmark/kernel_corpus/sources \
  --operators replay_benchmark/kernel_corpus/operators

# 3. 重新 bake MNN Vulkan shader（已有 6 个，重跑确保 .spv 都在）
python3 replay_benchmark/kernel_corpus/bake_mnn_vulkan_shaders.py \
  --sources replay_benchmark/kernel_corpus/sources \
  --operators replay_benchmark/kernel_corpus/operators \
  --tags 3.6.0

# 4. 验证全部 .spv 都生成
find replay_benchmark/kernel_corpus/operators/vulkan -name "*.spv" | wc -l  # 应该 ~367
```

**验证**：跑一次 host benchmark，Vulkan 的 `compile_failed` 应降到 0（或接近 0，仅剩真正不支持的 cooperative-matrix shader）。

### 任务 2（主任务）：扩展 Vulkan/mnn 3.6.0 adapter 覆盖

**现状**：Vulkan/mnn 只适配了 **5 个 op / 6 个 variant**（unary/binary/raster/reduction/pooling），但有 93 个 buffer-based `.comp` 可用。`bake_mnn_vulkan_shaders.py` 的 `OP_MAP`（脚本第 58-67 行）只列了这 5 个 op。

**目标**：扩展 `OP_MAP` + 新增专用 adapter，覆盖 3.6.0 的**基础算子**类（排除 LLM/量化专用 shader，它们不在 corpus replay 范围）。

**3.6.0 可适配的基础算子 shader**（路径：`sources/mnn/3.6.0/source/backend/vulkan/buffer/execution/glsl/`）：

| shader 文件 | op_type | 适配优先级 | 说明 |
|------------|---------|-----------|------|
| `select.comp` | select | 高 | 1 output + 2 input + 1 push(ivec4 size)，对应 OpenCL `select_buf_fp32` |
| `range.comp` | range | 高 | 1 output + 2 input(start/delta) + 1 push(ivec4 size)，对应 OpenCL `range_buf_fp32` |
| `cast_float_int.comp` | cast | 高 | buffer→buffer cast，对应 OpenCL `cast_buf_fp32` |
| `nc4hw4Tonchw.comp` | raster | 中 | 布局转换，可并入 RasterOp |
| `nchwTonc4hw4.comp` | raster | 中 | 同上 |
| `scale.comp` | scale | 高 | 对应 OpenCL `scale_buf_fp32` |
| `preluWithChannel.comp` | prelu | 中 | channel-wise prelu |
| `topkv2.comp` | topkv2 | 低 | 对应 OpenCL `topkv2_buf_fp32`，但签名复杂 |
| `argmax.comp` | argmax | 中 | 对应 OpenCL `argmax_buf_fp32` |
| `softmaxHeight_NHWC.comp` | softmax | 中 | 对应 OpenCL `softmax_buf_fp32` |
| `gridSampleNearest.comp` | grid_sample | 低 | 对应 OpenCL `grid_sample_buf_fp32` |
| `gridSampleBilinear.comp` | grid_sample | 低 | 同上，双线性变体 |
| `resizeNearest.comp` | interp | 低 | 对应 OpenCL `interp_buf_fp32` nearest 变体 |
| `resizeBilinear.comp` | interp | 低 | 同上，双线性变体 |
| `gemm_m8n4.comp` | gemm | 低 | matmul 变体，签名复杂 |
| `matmulunit.comp` | matmul | 低 | 另一 matmul 变体 |
| `matmulunit_HAS_BIAS.comp` | matmul | 低 | 带 bias 的 matmul |
| `norm.comp` / `norm_opt.comp` | layernorm | 中 | 对应 OpenCL `layernorm_buf_fp32` |

**不适配的 shader**（确认排除）：
- `attention_*.comp`（LLM 专用，cooperative-matrix / subgroup，超出 corpus 范围）
- `gemv_dequant_int*.comp`（量化 LLM）
- `int[2348]_weight_to_*.comp` / `conv1x1_int*_weight_*.comp`（量化权重预处理）
- `dynamic_*.comp`（动态量化）
- `linear_attn_*.comp`（线性注意力，LLM 专用）
- `pack_a_k4m4_to_m64k4.comp`（pack 辅助）
- `binary_blit*.comp` / `binary_int.comp`（image blit 变体，push_constant 布局不同）
- `convolution.comp` / `convolutionDepthwise.comp`（参数过多，需大量权重数据准备，优先级低）

**适配步骤**（每个新 op）：

1. **读 shader 签名**：
   ```bash
   grep -E "(layout\(binding|uniform constBuffer|void main)" \
     sources/mnn/3.6.0/source/backend/vulkan/buffer/execution/glsl/<shader>.comp
   ```
   记录：binding 顺序（哪个是 output/input）、constBuffer 结构（push constant 布局）、local_size。

2. **扩展 `OP_MAP`**（`bake_mnn_vulkan_shaders.py` 第 58-67 行）：
   ```python
   OP_MAP = {
       # ... 已有 5 个 ...
       "select": [("select.comp", "vulkan_select_fp32", [])],
       "range": [("range.comp", "vulkan_range_fp32", [])],
       # 注意：range.comp 有 #ifdef USE_INT，FP32 路径不需要传宏
       "cast": [("cast_float_int.comp", "vulkan_cast_float_int_fp32", [])],
       # ...
   }
   ```
   - variant 名必须用 `vulkan_` 前缀（与 OpenCL variant 区分，`findAdapter` 按 `(opType, variant)` 分发）
   - macros 列表：shader 内 `#ifdef` 选择的算子变体（如 unary 的 `EXP`）必须 bake 时固化

3. **重新 bake + 生成 .spv**：
   ```bash
   python3 replay_benchmark/kernel_corpus/bake_mnn_vulkan_shaders.py \
     --sources replay_benchmark/kernel_corpus/sources \
     --operators replay_benchmark/kernel_corpus/operators --tags 3.6.0
   ```

4. **重新生成 operators.json**：
   ```bash
   python3 replay_benchmark/kernel_corpus/extract_operator_kernels.py \
     --root replay_benchmark/kernel_corpus \
     --output replay_benchmark/kernel_corpus/operators.json
   ```

5. **实现专用 adapter**（按 SKILL.md 步骤 3 的强制规则）：
   - 每个 `(op_type, variant)` 一个 adapter 类，放在 `replay_benchmark/kernel_corpus_bridge/mnn/ops/`
   - **Vulkan binding 顺序注意**：shader 的 `binding=N` 声明顺序与 `ac.buffers` 数组顺序不一定一致。`ac.vulkanBindings[i]` 表示 `buffers[i]` 对应的 binding 号。**必须按 shader 声明的 binding 顺序映射**，否则 kernel 写入 input、output 全 0。（SKILL.md 第 134 行有详细说明，参考 `VulkanBinaryAddKernel::adapt()` 在 `BinaryOp.cpp`）
   - push constant 按 shader `uniform constBuffer` 结构精确填充，上限 128B
   - 在已有 op 文件里加新子类（如 `select` 加到 `ElementwiseOps.hpp/.cpp`），或新建文件（需在 `replay_benchmark/CMakeLists.txt` 第 12-22 行加 `.cpp`）

6. **更新 operator_cases.json**：每个新 variant 加一个 smoke case

7. **本机验证**（见下"验证流程"）

### 任务 3（确认项）：OpenCL/mnn 已基本完成

**现状**：62 个 variant，52 compiled / 10 unsupported / 51 dispatched / 51 validation_passed。

**10 个 unsupported 的原因**（全部预期内，无需修复）：
- 9 个 `*_subgroup_buf_fp32`：subgroup kernel，runner buffer-only 不支持，标记 unsupported 正确
- 1 个 `scale_nobias_fp32`（1.2.0）：image-based kernel（`image2d_t`），runner 不支持

**SKILL.md 明确规定**："含 image2d_t 的 kernel 也保留——runner 遇到不支持的能力时标记 `unsupported`，而不是 bake 时丢掉"（第 16、87 行）。

**唯一可能补的**：如果想让 `scale_nobias_fp32` 也能跑，需要看是否有 buffer-based 版本（但 1.2.0 源码里它是 image 的，无法转）。**结论：OpenCL 不需要再投入工作**。

### 任务 4（确认项）：Vulkan/mnn 1.2.0 tag 技术上不可行，跳过

**原因**：MNN 1.2.0 的 Vulkan 后端**全是 image-based shader**（`texture2d` / `image2d_t`），路径在 `sources/mnn/1.2.0/source/backend/vulkan/execution/glsl/`（注意：没有 `buffer/` 子目录）。replay runner 只支持 buffer binding + push constant，不支持 image。

**对比**：
- 3.6.0 有 `buffer/execution/glsl/`（93 个 buffer-based `.comp`）→ 可适配
- 1.2.0 只有 `execution/glsl/`（48 个 image-based）→ 不可适配

**结论**：Vulkan/mnn 维持只做 3.6.0。`operators.json` 里 `('vulkan','mnn')` 只有 `3.6.0` 是正确的。

### 任务 5（确认项）：CUDA/mnn 已完成

**现状**：14 个 tag、194 variant，在 CUDA 设备上 196/196 pass（见最后提交 `950432ed3`）。host build 不含 CUDA（无 nvcc），所以显示 196 unsupported 是预期。**不需要投入工作**。

---

## 验证流程

### 本机构建（有 glslangValidator 后）

```bash
cd /path/to/MNN
rm -rf build-host && mkdir build-host && cd build-host
cmake .. -DMNN_OPENCL=ON -DMNN_VULKAN=ON -DMNN_REPLAY_ENABLE_PERFCOUNTER=OFF \
  -DMNN_SEP_BUILD=ON -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_BENCHMARK=ON \
  -DMNN_BUILD_TEST=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target replay_benchmark.out -j$(nproc)
```

如果有 CUDA 且想验证 CUDA corpus：
```bash
cmake .. -DMNN_OPENCL=ON -DMNN_VULKAN=ON -DMNN_CUDA=ON \
  -DMNN_REPLAY_ENABLE_PERFCOUNTER=OFF -DMNN_SEP_BUILD=ON \
  -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TEST=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --target replay_benchmark.out -j$(nproc)
```

### 本机运行全量验证

```bash
# 找 vulkan lib（不同发行版路径不同）
VULKAN_LIB=$(find / -name "libvulkan.so.1" 2>/dev/null | head -1)
export LD_LIBRARY_PATH=$(dirname $VULKAN_LIB):build-host/source/backend/opencl:build-host/source/backend/vulkan:build-host

cd /path/to/MNN
./build-host/replay_benchmark.out \
  --kernel-corpus-bench --kernel-corpus-root replay_benchmark/kernel_corpus \
  --kernel-corpus-runs 5 --perf-counter-output /tmp/kc_host.json

# 查看统计
python3 -c "
import json
from collections import Counter
d=json.load(open('/tmp/kc_host.json'))
cs=d['cases']
print('total:', len(cs))
print('compile:', Counter(c['compile_status'] for c in cs))
print('dispatch:', Counter(c['dispatch_status'] for c in cs))
print('validation:', Counter(c['validation_status'] for c in cs))
for b in ['opencl','vulkan','cuda']:
    bc=[c for c in cs if c['backend']==b]
    if not bc: continue
    comp=sum(1 for c in bc if c['compile_status']=='compiled')
    disp=sum(1 for c in bc if c['dispatch_status']=='dispatched')
    val=sum(1 for c in bc if c['validation_status']=='validation_passed')
    print(f'  {b}: {len(bc)} | compiled={comp} dispatched={disp} validation_passed={val}')
"
```

### 完成判据

- **OpenCL**: 62 cases，52 compiled / 10 unsupported / 51 dispatched / 51 validation_passed（维持现状即合格）
- **Vulkan/mnn**: 任务 2 完成后，compiled + validation_passed 数应从 6 增长（每加一个 op 至少 +1 smoke case）
- **Vulkan/ncnn**: 任务 1 完成后，compile_failed 从 353 降到 0，validation_passed 从 48 增长（更多 shader 能编译就能 dispatch）
- **CUDA**: 在 CUDA 设备上 196/196 pass（已合格）

### Vulkan shader 单独编译验证

```bash
cd replay_benchmark/kernel_corpus/operators/vulkan
for f in $(find . -name "*.comp"); do
  glslangValidator -V --target-env vulkan1.1 --entry-point main "$f" -o /dev/null 2>/dev/null \
    && echo "OK $f" || echo "FAIL $f"
done
```

---

## 关键文件索引

| 用途 | 路径 |
|------|------|
| Bake 脚本（OpenCL） | `replay_benchmark/kernel_corpus/bake_mnn_kernels.py` |
| Bake 脚本（MNN Vulkan） | `replay_benchmark/kernel_corpus/bake_mnn_vulkan_shaders.py`（OP_MAP 在第 58 行） |
| Bake 脚本（ncnn Vulkan） | `replay_benchmark/kernel_corpus/bake_ncnn_shaders.py` |
| operators.json 索引 | `replay_benchmark/kernel_corpus/operators.json` |
| operator_cases.json | `replay_benchmark/kernel_corpus/operator_cases.json` |
| MNN Bridge 注册 | `replay_benchmark/kernel_corpus_bridge/mnn/MnnBridge.cpp`（`registerMnnBridge()` 第 88 行） |
| MNN adapter 实现 | `replay_benchmark/kernel_corpus_bridge/mnn/ops/*.cpp` |
| ncnn Bridge | `replay_benchmark/kernel_corpus_bridge/ncnn/NcnnBridge.cpp` |
| CMake 源文件列表 | `replay_benchmark/CMakeLists.txt` 第 12-30 行 |
| SKILL 完整规范 | `skills/kernel-adapt/SKILL.md` |

## 适配参考（已有 Vulkan adapter 作为模板）

已实现的 6 个 Vulkan adapter 分布在：

| adapter 类 | 文件 | variant | 学什么 |
|-----------|------|---------|-------|
| `VulkanUnaryExpKernel` | `mnn/ops/UnaryOp.cpp` | `vulkan_unary_buf_exp_fp32` | 1 output + 1 input + push(int4 size) 的最简形态 |
| `VulkanBinaryAddKernel` | `mnn/ops/BinaryOp.cpp` | `vulkan_binary_buf_add_fp32` | 2 input + 1 output + push，binding 顺序注意 |
| `VulkanRasterBlitC4Kernel` | `mnn/ops/RasterOp.cpp` | `vulkan_blit_c4_fp32` | blit 布局转换 |
| `VulkanReductionSumKernel` | `mnn/ops/ReductionOp.cpp` | `vulkan_reduce_buf_sum_fp32` | reduce，标量 validator |
| `VulkanPoolingMaxKernel` | `mnn/ops/PoolingOp.cpp` | `vulkan_maxpool_fp32` | 1 input + 1 output + push(shape+kernel+stride+pad) |
| `VulkanPoolingAvgKernel` | `mnn/ops/PoolingOp.cpp` | `vulkan_avgpool_fp32` | 同上，avg 变体 |

**新 adapter 模板**：参考 SKILL.md 第 205-277 行的 `BinaryOp.hpp` / `BinaryOp.cpp` 完整示例（含基类设计、Vulkan 子类完整 `adapt()` 实现、push constant 布局、vulkanBindings 映射）。

## 推送

完成所有适配 + 验证后：
```bash
git add -A
git commit -m "[GPU:Feature] Complete Vulkan/mnn adapter coverage + rebake ncnn .spv"
git push origin HEAD:tmp
```
