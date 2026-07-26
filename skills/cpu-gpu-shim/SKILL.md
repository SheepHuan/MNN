---
name: cpu-gpu-shim
description: 在 x86_64 Linux 宿主机上为 MNN 的 replay_benchmark / kernel_corpus_benchmark 配置 OpenCL 与 Vulkan 运行时。覆盖两条路径：(A) 真实核显驱动（Intel/AMD Mesa Vulkan + Mesa Clover/Rusticl OpenCL 或 Intel NEO compute runtime），(B) CPU 软件实现（PoCL 对应 OpenCL、SwiftShader 对应 Vulkan）用于无 GPU 的宿主机。用于把 MNN 的 `MNN_FORWARD_OPENCL`（forward=3）与 `MNN_FORWARD_VULKAN`（forward=7）后端跑在宿主机上；不做交叉编译。
---

# 宿主机 OpenCL / Vulkan 运行时（真实核显驱动 + CPU 软件实现）

MNN 的 OpenCL 后端（`MNN_FORWARD_OPENCL`，forward=3）和 Vulkan 后端
（`MNN_FORWARD_VULKAN`，forward=7）通过标准 Khronos ICD 机制发现运行时。在
x86_64 Linux 宿主机上有两种方式满足该机制：

- **路径 A —— 真实核显驱动。** 使用宿主机核显（Intel HD/Iris、AMD Radeon
  Vega/RDNA）。Vulkan 走 Mesa RADV/ANV，OpenCL 走 Mesa Clover/Rusticl 或 Intel
  NEO compute runtime。当宿主机有核显、且希望跑真实硬件（虽然不是移动 GPU，
  但比 CPU 软件实现更接近，且能覆盖 FP16/subgroup 等可选特性）时用这条路径。
- **路径 B —— CPU 软件实现。** PoCL（OpenCL 3.0）+ SwiftShader（Vulkan
  1.3）。用于无 GPU 的 CI runner，或需要一份确定性参考实现做正确性回归时。

两条路径都**不需要改 MNN 代码**。区别仅在于装了哪些 ICD 文件、以及
`VK_ICD_FILENAMES` / `OCL_ICD_VENDORS` 环境变量指向哪里。

本 skill 范围：在 x86_64 宿主机上安装上述任意一条或两条路径，然后编译 MNN
对接它们。**不做交叉编译**，只面向宿主机；AArch64 / 真机 replay 用
`skills/replay-benchmark-cross-compile`。

## 适用范围与不适用范围

- **适用**：大规模并行正确性回归 `replay_benchmark`（record + replay）、
  `kernel_corpus_benchmark`、op 级 snapshot 校验。路径 B 可跨 CI 核数并行；
  路径 A 在核显上跑。
- **适用（仅路径 A）**：覆盖比 CPU 软件实现更多的可选特性（FP16 storage、
  subgroup、部分核显上的 cooperative matrix）。
- **不适用**：与移动 Adreno / Mali 可比的性能数据。路径 A 反映 x86 核显；
  路径 B 是纯 CPU。两者都不要并入真机 perf CSV。
- **不适用**：PMU 计数器。`--opencl-pmu-bench` / `--model-pmu-bench` 依赖
  移动 GPU 的真实 PMU，必须在真机上跑。
- **不适用（路径 B）**：完整特性对齐。`VK_KHR_cooperative_matrix`、subgroup
  shuffle、FP16 arithmetic 在 CPU 软件实现上通常缺失。MNN 会自动降级到 FP32
  / 非 coopmat 路径，因此*基础*路径被覆盖，但 coopmat / FP16 路径不被覆盖。

## 关键约束

- **仅限宿主机。** 本 skill 面向 x86_64 Linux。不要在 Adreno / Mali 设备上
  装；设备本身已有真实 GPU。
- **不改 MNN 代码。** MNN 的 `OpenCLWrapper.cpp` Linux 搜索列表已包含 PoCL
  路径（`/usr/local/lib/libpocl.so`）；Vulkan 后端走标准
  `dlopen("libvulkan.so")` + loader ICD 协商，SwiftShader 或 Mesa RADV 通过
  ICD manifest 注入。
- **MNN 两个后端都是渐进降级设计。** 缺失可选特性时 op 会 fallback 而非直接
  失败。任意路径下 replay "pass" 是有效信号；"fail" 可能是真实 op bug *也可能
  是* 缺某个可选能力——查 stderr 看 MNN 报的具体 extension/feature 来区分。
- **SwiftShader 没有 apt 包。** 必须源码编译。PoCL、Mesa、Intel NEO 都有 apt
  包；除非需要更新版本，否则优先用包。
- **两条路径都不要跑 perf benchmark** 并把结果与真机 perf 数据合并；不可比。
- 跑 MNN 前先用 `clinfo`（OpenCL）和 `vulkaninfo --summary`（Vulkan）验证 ICD
  注册。ICD 文件缺失是最常见的失败原因。
- **Ubuntu 24.04 上 Mesa OpenCL 的 ICD 文件不会自动注册。** `mesa-opencl-icd`
  包只装 runtime，默认**不**在 `/etc/OpenCL/vendors/` 下放 ICD 文件；`clinfo`
  会是空的，直到手动补一个（见路径 A 第 3 步）。

## 工作流 —— 路径 A：真实核显驱动

当宿主机有 Intel 或 AMD 核显、希望 MNN 跑在真实（x86）GPU 硬件上时用这条
路径。Ubuntu 24.04 上 Mesa 栈已经通过 `mesa-vulkan-drivers` 为 Intel 和 AMD
核显提供 Vulkan 1.4 驱动；OpenCL 由 `mesa-opencl-icd`（Clover，较老）或
`intel-opencl-icd`（NEO compute runtime，仅 Intel）提供。

### A.1 识别核显并选择驱动组合

```bash
lspci | grep -iE 'vga|3d|display'
# 示例：
#   Intel:  "Intel Corporation Iris Xe Graphics"
#   AMD:    "Advanced Micro Devices ... Renoir [Radeon Vega]"
ls /dev/dri  # 必须有 card0 和 renderD128 才表示核显可用
```

| 核显厂商 | Vulkan 驱动 | OpenCL 驱动（推荐） |
|---------|-------------|---------------------|
| Intel (Gen9+) | Mesa ANV (`intel_icd.json`) | Intel NEO (`intel-opencl-icd`) |
| AMD (Vega/RDNA) | Mesa RADV (`radeon_icd.json`) | Mesa Clover/Rusticl (`mesa-opencl-icd`) |
| Intel 或 AMD（Mesa 内的 CPU 回退） | llvmpipe (`lvp_icd.json`) | （不适用） |

### A.2 安装核显的 Vulkan + OpenCL 驱动

```bash
sudo apt update
sudo apt install -y \
    mesa-vulkan-drivers \
    libvulkan1 libvulkan-dev vulkan-tools \
    ocl-icd-opencl-dev ocl-icd-libopencl1 clinfo
```

再按核显厂商补 OpenCL 驱动：

```bash
# Intel 核显：
sudo apt install -y intel-opencl-icd

# AMD 核显（或没有 NEO runtime 的 Intel 核显）：
sudo apt install -y mesa-opencl-icd
```

### A.3 注册 OpenCL ICD 文件（Ubuntu 24.04 专用）

Ubuntu 24.04 上 `mesa-opencl-icd` 包只装 runtime，不会自动注册 ICD 文件；
`intel-opencl-icd` 会自动注册一个。手动检查并补齐：

```bash
ls /etc/OpenCL/vendors/
# Intel NEO 期望：  intel.icd
# Mesa Clover 期望：mesa.icd  （可能缺失 —— 需要手动创建）

# 如果 mesa.icd 缺失，创建一个指向 Mesa OpenCL runtime：
ls /usr/lib/x86_64-linux-gnu/gallium-pipe/*.pipe  # 查找 clover 的 .so
sudo tee /etc/OpenCL/vendors/mesa.icd >/dev/null <<'EOF'
/usr/lib/x86_64-linux-gnu/gallium-pipe/clover_pipe.so
EOF
```

`mesa-opencl-icd` 在 Mesa 24.x 上优先用 Rusticl（较新的 Rust 实现），
Clover 是 legacy。包可能装其中一个或两个都装：

```bash
ls /usr/lib/x86_64-linux-gnu/gallium-pipe/  # clover_pipe.so / rusticl_pipe.so
```

如果 Rusticl 存在，优先把 `mesa.icd` 指向 `rusticl_pipe.so` 而不是
`clover_pipe.so`。

### A.4 验证核显可见

```bash
vulkaninfo --summary | grep -A 4 'GPU0'
# 期望：deviceName "AMD Radeon Graphics (RADV ...)" 或
#       "Intel(R) Iris(R) Xe Graphics (ANV)"
# apiVersion 必须 >= 1.3 才能和 MNN Vulkan 后端特性对齐。

clinfo | grep -E 'Platform Name|Device Name|Device Type|Device Version'
# 期望：某个平台（Mesa 或 Intel）下有一个 GPU 设备，而不是 CPU。
```

路径 A 下 `VK_ICD_FILENAMES` **保持未设置**，让 Vulkan loader 自动从
`/usr/share/vulkan/icd.d/` 拾取系统 ICD 文件。仅当需要把 MNN 限制到单一设备
（例如排除 llvmpipe）时才设置它。

### A.5 核显 vs 移动 GPU 的注意点

- Mesa RADV/ANV 报告 Vulkan 1.4（实测 `1.4.318`），远高于 MNN 需求（core
  特性需 1.2，1.3 头文件有但代码未用）。
- FP16 storage 和 subgroup 在 Intel Gen9+ 与 AMD RDNA 上通常支持；
  cooperative matrix 不一定有——MNN 运行时逐项检查并降级。
- 核显共享系统内存；时间与移动 dGPU 不可比，但正确性是有效信号。

## 工作流 —— 路径 B：CPU 软件实现（PoCL + SwiftShader）

用于无 GPU 的 CI runner，或作为确定性参考实现。PoCL 是合规的 CPU OpenCL
3.0 实现；SwiftShader 是合规的 CPU Vulkan 1.3 实现。

### B.1 通过 apt 安装 PoCL（CPU OpenCL）

在 Ubuntu 24.04（noble）上验证。包名 `pocl-opencl-icd`，会拉入
`libpocl2t64`（runtime）并在 `/etc/OpenCL/vendors/pocl.icd` 注册 ICD。

```bash
sudo apt update
sudo apt install -y pocl-opencl-icd ocl-icd-opencl-dev clinfo
```

验证：

```bash
# clinfo 必须列出 pocl 平台，且至少有一个 CPU 设备。
clinfo | grep -E "Platform Name|Device Name|Device Type" | head -12
# 期望："Portable Computing Language" 平台，设备 "pthread-xxx"。
```

MNN 的 `OpenCLWrapper.cpp` Linux 搜索列表已包含 `/usr/local/lib/libpocl.so`
和 `libOpenCL.so`。apt 安装会把 ICD loader（`libOpenCL.so.1`）放到
`/usr/lib/x86_64-linux-gnu/`，在默认 loader 路径上，所以 OpenCL 不需要
`LD_LIBRARY_PATH`。

编译并运行任意 OpenCL 冒烟测试（或 record 模式 forward=3 的
replay_benchmark）验证 MNN 能看到 PoCL。如果 MNN 报 "Invalide device for
support opencl"，说明 ICD 文件缺失或不可读。

### B.2 （可选）从源码安装更新的 PoCL

仅当 apt 版本（noble 上是 5.0）缺某个修复时才做。源码编译需要
LLVM/Clang dev 头：

```bash
sudo apt install -y build-essential cmake ninja-build \
    llvm-dev libclang-dev libclang-cpp-dev zlib1g-dev libtinfo-dev \
    libhwloc-dev pkg-config python3 llvm-spirv-15 spirv-tools

git clone --depth 1 https://github.com/pocl/pocl.git /tmp/pocl
cmake -S /tmp/pocl -B /tmp/pocl/build -G Ninja \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DENABLE_ICD=ON
cmake --build /tmp/pocl/build --parallel
sudo cmake --install /tmp/pocl/build
# ICD 文件落在 /etc/OpenCL/vendors/pocl.icd；验证：
clinfo | grep "Platform Name"
```

### B.3 从源码编译 SwiftShader（CPU Vulkan）

没有 apt 包。SwiftShader 编译产出 `libvk_swiftshader.so` + ICD manifest
`vk_swiftshader_icd.json`。

```bash
sudo apt install -y build-essential cmake git
git clone --depth 1 https://github.com/google/swiftshader.git /tmp/SwiftShader
cmake -S /tmp/SwiftShader -B /tmp/SwiftShader/build \
    -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/SwiftShader/build --parallel "$(nproc)"
```

产物在 `/tmp/SwiftShader/build/Linux/` 下：

- `libvk_swiftshader.so`
- `vk_swiftshader_icd.json`

拷到稳定路径，方便 MNN 和 `vulkaninfo` 找到：

```bash
sudo mkdir -p /usr/local/lib/swiftshader
sudo cp /tmp/SwiftShader/build/Linux/libvk_swiftshader.so /usr/local/lib/swiftshader/
sudo cp /tmp/SwiftShader/build/Linux/vk_swiftshader_icd.json /usr/local/lib/swiftshader/
```

如果没装 `vulkaninfo`：

```bash
sudo apt install -y vulkan-tools
```

### B.4 把 SwiftShader 注册为 Vulkan ICD

把 `VK_ICD_FILENAMES` 指向 SwiftShader 的 ICD manifest。在跑 MNN 的 shell
（或 CI 环境块）里设置：

```bash
export VK_ICD_FILENAMES=/usr/local/lib/swiftshader/vk_swiftshader_icd.json
export LD_LIBRARY_PATH=/usr/local/lib/swiftshader:${LD_LIBRARY_PATH:-}
vulkaninfo --summary | head -30
# 期望：deviceName "SwiftShader Device"、driverName "SwiftShader"。
```

`VK_ICD_FILENAMES` 会覆盖系统 loader 的搜索；如果宿主机有真实 GPU，这个变量
确保 MNN 走 SwiftShader 而不是宿主 GPU。取消设置
（`unset VK_ICD_FILENAMES`）即恢复宿主 GPU。

持久化（开发机，非 CI）：在
`/usr/local/share/vulkan/icd.d/swiftshader_icd.x86_64.json` 放一个含绝对库
路径的文件，然后保持 `VK_ICD_FILENAMES` 未设置。CI 里优先用环境变量，显式
且可复现。

## 编译 MNN（OpenCL + Vulkan 后端，宿主机）

```bash
cd /home/yanghuan/code/MNN
cmake -S . -B build-host \
    -DMNN_OPENCL=ON \
    -DMNN_VULKAN=ON \
    -DMNN_BUILD_BENCHMARK=ON \
    -DMNN_SEP_BUILD=ON \
    -DMNN_BUILD_SHARED_LIBS=ON \
    -DMNN_BUILD_TOOLS=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-host --parallel "$(nproc)"
```

注意：

- `MNN_SEP_BUILD=ON` + `MNN_BUILD_SHARED_LIBS=ON` 把 `libMNN_CL.so` 和
  `libMNN_Vulkan.so` 保持为独立动态库，与真机部署布局一致，并避免
  `OpenCLWrapper.cpp` 静态链接重复符号。
- `MNN_BUILD_BENCHMARK=ON` 编译 `replay_benchmark.out` 和
  `kernel_corpus_benchmark`。
- `MNN_OPENCL` 默认用 ICD loader（`libOpenCL.so`）；PoCL 或 Mesa Clover 的
  ICD 通过 `/etc/OpenCL/vendors/` 自动拾取。
- `MNN_VULKAN` 默认用系统 `libvulkan.so`；SwiftShader 通过
  `VK_ICD_FILENAMES` 拾取，Mesa RADV/ANV 通过 `/usr/share/vulkan/icd.d/`
  自动拾取。

## 冒烟测试

路径 A（真实核显，不设 `VK_ICD_FILENAMES`）：

```bash
cd /home/yanghuan/code/MNN
export LD_LIBRARY_PATH=build-host:build-host/source/backend/opencl:build-host/source/backend/vulkan

# OpenCL (forward=3)：Mesa Clover/Rusticl 或 Intel NEO
./build-host/replay_benchmark.out models 2 2 3 4 2 --record records/opencl-igpu
# Vulkan (forward=7)：Mesa RADV/ANV
./build-host/replay_benchmark.out models 2 2 7 0 2 --record records/vulkan-igpu
```

路径 B（CPU 软件实现）：

```bash
cd /home/yanghuan/code/MNN
export VK_ICD_FILENAMES=/usr/local/lib/swiftshader/vk_swiftshader_icd.json
export LD_LIBRARY_PATH=/usr/local/lib/swiftshader:build-host:build-host/source/backend/opencl:build-host/source/backend/vulkan

# OpenCL (forward=3)：PoCL
./build-host/replay_benchmark.out models 2 2 3 4 2 --record records/opencl-pocl
# Vulkan (forward=7)：SwiftShader
./build-host/replay_benchmark.out models 2 2 7 0 2 --record records/vulkan-swiftshader
```

record 步骤成功（`records/<backend>/*.op.fb` 文件产出且无 "Invalide device"
报错）即确认链路打通。逐 op replay 跑完整正确性矩阵：

```bash
for op in records/opencl-pocl/ops/*.op.fb; do
    ./build-host/replay_benchmark.out --record records/opencl-pocl --op-id "$(basename "$op" .op.fb)"
done
```

## 并行化正确性矩阵

CPU 软件实现是纯 CPU，矩阵可跨核并行。一个进程跑一个 op 是天然粒度：

```bash
find records/opencl-pocl/ops -name '*.op.fb' -printf '%f\n' \
    | sed 's/\.op.fb$//' \
    | xargs -P "$(nproc)" -I {} \
        ./build-host/replay_benchmark.out --record records/opencl-pocl --op-id {}
```

汇总退出码和每 op 日志；非零退出说明有 mismatch，用
`skills/general-debug` 排查。

## 已知限制与排错

| 症状 | 原因 | 修复 |
|------|------|------|
| `clinfo` 无平台 | `/etc/OpenCL/vendors/` 下缺 ICD 文件 | `sudo apt reinstall pocl-opencl-icd`（路径 B）或手动创建 `mesa.icd`（路径 A 第 3 步）；确认文件可读 |
| MNN 报 "Invalide device for support opencl" | `OpenCLWrapper` 无法 `dlopen` 到库 | 确认 `libOpenCL.so.1` 在 loader 路径上；`ldconfig -p \| grep libOpenCL` |
| `vulkaninfo` 选了宿主 GPU 而非 SwiftShader | `VK_ICD_FILENAMES` 未设或被覆盖 | 在同一 shell 里 `export VK_ICD_FILENAMES=...`；清掉其他 `VK_*` 环境变量 |
| MNN Vulkan：coopmat / FP16 路径被静默禁用 | SwiftShader 不实现这些扩展 | 预期行为；FP32 基础路径仍可跑 |
| 路径 B 下 replay mismatch 但真机 pass | CPU 软件实现缺可选特性，MNN 选了不同代码路径 | 标记该 op 为 "shim-incompatible"；不要为此 block CI |
| `--opencl-pmu-bench` 产生零 delta | PoCL 无 GPU PMU 寄存器 | 预期；不要在软件实现下跑 PMU 模式 |
| 路径 A 下 OpenCL 仍为空 | Mesa ICD 文件未注册（Ubuntu 24.04 已知问题） | 按路径 A 第 3 步手动创建 `/etc/OpenCL/vendors/mesa.icd`，指向 `rusticl_pipe.so` 或 `clover_pipe.so` |

## 与其他 skill 的关系

- `skills/replay-benchmark-cross-compile`：AArch64 真机 replay；本 skill 是
  它的宿主机对应物。
- `skills/general-debug`：replay 出现 mismatch 时，按调试 skill 区分是 op
  bug 还是软件实现限制。
- `skills/test-ci`：把软件实现跑的正确性矩阵接入 CI stage 时使用。
