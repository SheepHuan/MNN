# replay_benchmark kernel corpus 适配转交文档

## 当前状态（截至 commit 88885566a）

### 全量验证结果（host: AMD Radeon RADV + Mesa rusticl OpenCL）

| framework/backend | compiled | val_passed | not_validated | val_failed | compile_fail |
|---|---|---|---|---|---|
| mnn/opencl | 54/62 | 52 | 10 | 0 | 8 |
| mnn/vulkan | 6/6 | 6 | 0 | 0 | 0 |
| ncnn/vulkan | 357/361 | 357 | 4 | 0 | 4 |

### 已完成的提交历史

```
88885566a [GPU:Bugfix] Fix MNN OpenCL compile_failed + add smoke-test validate to all adapters
9eed09f6e [GPU:Bugfix] Fix pooling stride mismatch in validator + OpenCL runner adapter dispatch
580ba669b [GPU:Feature] Co-locate validator with adapter + ncnn 357/357 validation_passed
a8db4959  [GPU:Feature] Add MNN Vulkan kernel adapters (6/6 validated) + rename Op→Kernel architecture
```

## 架构设计

### Op vs Kernel 命名规则（强制）

- **Op** = 计算语义（如 `unary`、`binary`、`pooling`），一个 Op 只有一个逻辑计算
- **Kernel** = Op 的具体变体实现（如 `unary_buf_exp_fp32`、`vulkan_unary_buf_exp_fp32`），一个 Op 有多个 Kernel

| 层级 | 命名格式 | 示例 | 说明 |
|------|---------|------|------|
| 基类 | `<Op>KernelBase` | `UnaryKernelBase` | 纯虚接口，`adapt()=0` + `validate()=0`，不提供共享实现 |
| OpenCL 子类 | `OpenCL<Variant>Kernel` | `OpenCLUnaryExpKernel` | 完整实现 adapt() + validate() |
| Vulkan 子类 | `Vulkan<Variant>Kernel` | `VulkanUnaryExpKernel` | 完整实现 adapt() + validate() |
| ncnn 子类 | `Vulkan<Variant>Kernel` | `VulkanSigmoidKernel` | ncnn 只有 Vulkan backend |

### OpAdapter 接口（OpAdapter.hpp）

```cpp
class OpAdapter {
public:
    virtual const char* opType() const = 0;      // e.g. "unary"
    virtual const char* variant() const = 0;      // e.g. "unary_buf_exp_fp32"
    virtual bool adapt(const CaseSpec& spec, AdaptedCase& ac) const = 0;
    virtual bool validate(const AdaptedCase& ac, const std::vector<float>& output) const {
        return false;  // 默认 false，子类 override 实现精确验证
    }
};
```

- `adapt()`：填 `ac.buffers`/`ac.args`/`ac.vulkanBindings`/`ac.pushConstants`/`ac.globalSize` 等
- `validate()`：验证 kernel output 是否正确，与 adapter 同文件维护
- `FallbackAdapter::validate()` 返回 `true`（smoke test：compiled+dispatched 即通过）

### 分发链路

```
operator_cases.json (case 定义)
    ↓
KernelCorpusBenchmark → findBridge(framework, tag)
    ↓
Bridge::adapt() → findAdapter(opType, variant) → adapter->adapt(spec, ac)
    ↓ (ac.adapter 指针保存到 AdaptedCase)
Runner (OpenCL/Vulkan) → compile → dispatch → readback
    ↓
runValidator() → 优先 ac.adapter->validate()，回退 legacy 字符串 validator
```

### MNN Vulkan bake 关键点

1. **`bake_mnn_vulkan_shaders.py`**：从 `sources/mnn/<tag>/source/backend/vulkan/buffer/execution/glsl/` 读 `.comp`
2. **FP32 preamble**：`#define FLOAT float` / `#define FLOAT4 vec4` 等
3. **算子选择宏**（`EXP`/`ADD`/`C4`/`SUM`）在 bake 时通过 `-D` 固化进 SPIR-V
4. **`uniform constBuffer` → `layout(push_constant) uniform constBuffer`**：MNN shader 用 UBO binding 声明常量，但 runner 用 `vkCmdPushConstants` 传参。bake 时自动替换声明。
5. **预编译 `.spv`**：bake 时调 glslangValidator 生成 .spv，设备端不需要 glslangValidator
6. **Vulkan binding 顺序**：`ac.vulkanBindings[i]` = `buffers[i]` 对应的 binding 号，必须按 shader 声明的 binding 顺序映射

### ncnn 20190611 dispatch 修复

ncnn 20190611 的 elementwise shader 用 3D dispatch `(w, h, c)`，不是 1D。`adaptElementwise()` 已修为：
```cpp
ac.globalSize[0] = w;
ac.globalSize[1] = h;
ac.globalSize[2] = c;
```

## 剩余待完成项

### P3: MNN OpenCL 剩余 8 个 compile_fail（需端侧设备）

| case | tag | 原因 | 处理 |
|------|-----|------|------|
| binary_subgroup_buf_fp32 | 3.6.0 | Intel subgroup extension | 需 Adreno/Mali 设备 |
| conv_2d_c16_subgroup_buf_fp32 | 3.6.0 | 同上 | 同上 |
| conv_2d_c1_subgroup_buf_fp32 | 3.6.0 | 同上 | 同上 |
| depthwise_conv2d_subgroup_buf_fp32 | 3.6.0 | 同上 | 同上 |
| pooling_subgroup_buf_fp32 | 3.6.0 | 同上 | 同上 |
| unary_subgroup_buf_fp32 | 3.6.0 | 同上 | 同上 |
| winogradTransform_subgroup_buf_fp32 | 3.6.0 | 同上 | 同上 |
| scale_nobias_fp32 | 1.2.0 | `OpenCL arg setup failed (code -51)` | adapter 参数问题，需调试 |

### P4: MNN Vulkan 剩余 96 个 buffer shader 未 bake

已 bake 6 个（unary/binary/blit/reduce/maxpool/avgpool）。剩余 96 个 buffer shader + 50 个 image shader 未 bake。

主要类别：
- **attention 系列**（14 个）：attention_fused, attention_prefill_kblock_* 等
- **conv/conv1x1 量化**（12 个）：conv1x1_int2/3/4/8_weight_prepare 等
- **gemv dequant**（8 个）：gemv_dequant_int2/3/4/8
- **linear attention**（8 个）
- **其他**：argmax, blitregion, deconvolution, gridSample, norm, onehot, rope, scale, select, softmax, topkv2 等

### P5: ncnn 4 个 compile_failed（cooperative-matrix-only）

| shader | 原因 |
|--------|------|
| gemm_cm.comp | cooperative-matrix-only |
| sdpa_fa_cm.comp | 同上 |
| sdpa_cross_cm.comp | 同上 |
| gemm_sg.comp | subgroup-only |

设备不支持 cooperative matrix 时标记 `unsupported`，不篡改源码。

## 关键约束（不能违反）

1. **不编译历史 backend**：只提取 kernel 文本，用当前 MNN OpenCL/Vulkan runtime 编译
2. **bake 不跳过任何 kernel**：含 image2d_t / cooperative-matrix 的 kernel 也保留
3. **禁止访问** `schema/private/` 和 `source/internal/`
4. **C++ 遵循** C++11、4 空格缩进、禁止 namespace 作用域动态初始化对象
5. **Vulkan variant 名用 `vulkan_` 前缀**与 OpenCL variant 区分
6. **MNN Vulkan bake 时 `uniform constBuffer` → `push_constant`**：runner 用 push constant 传参
7. **Vulkan binding 顺序**：`vulkanBindings[i]` 必须按 shader 声明的 binding 顺序映射
8. **validator 与 adapter 共址**：各子类实现自己的 `validate()`，不通过字符串匹配
9. **smoke-test validate**：无精确 validator 的 adapter 用 `return true`（compiled+dispatched 即通过）
10. **不破坏 kernel 语义**：validator 失败是 validator 的问题（数据布局/公式不匹配），不是 kernel 的问题。修复 validator，不改 kernel

## 构建与测试命令

```bash
# 本机构建（host AMD）
rm -rf build-host && mkdir build-host && cd build-host
cmake .. -DMNN_OPENCL=ON -DMNN_VULKAN=ON -DMNN_REPLAY_ENABLE_PERFCOUNTER=OFF \
  -DMNN_SEP_BUILD=ON -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_BENCHMARK=ON \
  -DMNN_BUILD_TEST=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build . --target replay_benchmark.out -j$(nproc)

# 本机运行全量
LD_LIBRARY_PATH=build-host/source/backend/opencl:build-host/source/backend/vulkan:build-host \
  ./build-host/replay_benchmark.out \
  --kernel-corpus-bench --kernel-corpus-root replay_benchmark/kernel_corpus \
  --kernel-corpus-runs 1

# 单个 case
LD_LIBRARY_PATH=build-host/source/backend/opencl:build-host/source/backend/vulkan:build-host \
  ./build-host/replay_benchmark.out \
  --kernel-corpus-bench --kernel-corpus-root replay_benchmark/kernel_corpus \
  --kernel-corpus-case <case_name> --kernel-corpus-runs 1

# 重新 bake MNN Vulkan
python3 replay_benchmark/kernel_corpus/bake_mnn_vulkan_shaders.py \
  --sources replay_benchmark/kernel_corpus/sources \
  --operators replay_benchmark/kernel_corpus/operators --tags 3.6.0

# 重新生成 operators.json
python3 replay_benchmark/kernel_corpus/extract_operator_kernels.py \
  --root replay_benchmark/kernel_corpus --output replay_benchmark/kernel_corpus/operators.json
```

## 设备信息

- **Rhinopi-X1**：`root@192.168.101.227`，`/mnt/nvme/workspace/replay-benchmark`，Adreno 740
  - Vulkan lib: `/mnt/nvme/workspace/replay-benchmark-vulkan/lib/libvulkan.so`
- **OrangePi**：`root@192.168.101.113`，`/mnt/ssd/workspace`，Mali-G610
- 凭据只从环境读取，不写入仓库

## 关键文件索引

| 文件 | 作用 |
|------|------|
| `replay_benchmark/KernelCorpusBenchmark.cpp` | runner：compile/dispatch/readback/validate |
| `replay_benchmark/kernel_corpus_bridge/OpAdapter.hpp` | OpAdapter 接口 + findAdapter + FallbackAdapter |
| `replay_benchmark/kernel_corpus_bridge/Bridge.hpp` | AdaptedCase + CaseSpec + Bridge 接口 |
| `replay_benchmark/kernel_corpus_bridge/mnn/MnnBridge.cpp` | MNN 框架分发 + .spv 加载 |
| `replay_benchmark/kernel_corpus_bridge/ncnn/NcnnBridge.cpp` | ncnn 框架分发 + .spv 加载 |
| `replay_benchmark/kernel_corpus_bridge/mnn/ops/*.hpp/cpp` | MNN OpenCL + Vulkan kernel adapter |
| `replay_benchmark/kernel_corpus_bridge/ncnn/ops/*.hpp/cpp` | ncnn Vulkan kernel adapter |
| `replay_benchmark/kernel_corpus/bake_mnn_vulkan_shaders.py` | MNN Vulkan bake 脚本 |
| `replay_benchmark/kernel_corpus/bake_ncnn_shaders.py` | ncnn Vulkan bake 脚本 |
| `replay_benchmark/kernel_corpus/bake_mnn_kernels.py` | MNN OpenCL bake 脚本 |
| `replay_benchmark/kernel_corpus/operators.json` | 索引 (backend/op_type/framework/tag/variant/entry/file) |
| `replay_benchmark/kernel_corpus/operator_cases.json` | 可执行 case 定义 |
| `skills/kernel-adapt/SKILL.md` | 适配流程文档 |
