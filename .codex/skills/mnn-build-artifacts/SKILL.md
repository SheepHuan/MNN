---
name: mnn-build-artifacts
description: 当用户要求在本 MNN 仓库中编译、重编或诊断 MNN 产物，尤其是 Jetson/aarch64 CUDA、CUDA_ARCHS、KleidiAI、构建残留进程、build/install/output 产物检查时使用这个 Skill。
metadata:
  short-description: 编译和检查 MNN CUDA/LLM 产物
---

# MNN Build Artifacts

本 Skill 用于本 MNN 仓库内的构建和产物检查。构建目录、install 目录和临时产物放在本仓库 `.cache/` 下，运行时发布产物统一放到 `.cache/output/mnn/artifacts/<platform>/`，不要提交这些本地产物。

## 基本约束

1. 默认从本 MNN 仓库根目录执行命令。
2. 修改或构建前先看本仓库状态：

```bash
git status --short
```

3. 不要覆盖用户已有改动，不要提交 `.cache/`、`output/`、模型缓存、`3rd_party/cutlass/` 等本地产物。
4. 构建日志里必须确认关键开关，而不是只看最后是否链接成功。

## 平台产物路径

默认按当前平台名查找和安装产物：

```bash
detect_mnn_artifact_platform() {
  if [[ -n "${MNN_ARTIFACT_PLATFORM:-}" ]]; then
    printf '%s\n' "${MNN_ARTIFACT_PLATFORM}"
    return
  fi
  if [[ -r /proc/device-tree/model ]] && tr -d '\0' </proc/device-tree/model | grep -qiE 'Jetson|NVIDIA'; then
    printf 'jetson\n'
    return
  fi
  case "$(uname -m)" in
    x86_64|amd64) printf 'x64\n' ;;
    aarch64|arm64) printf 'jetson\n' ;;
    *) uname -m ;;
  esac
}

MNN_ARTIFACT_PLATFORM="$(detect_mnn_artifact_platform)"
MNN_ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-$PWD/.cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM}"
```

Jetson 默认平台名是 `jetson`，默认安装目录是：

```text
.cache/output/mnn/artifacts/jetson/
```

需要覆盖时只设置 `MNN_ARTIFACT_PLATFORM` 或 `MNN_ARTIFACT_ROOT`，不要在各个 skill 里写死 `x64`、`jetson_cuda` 或旧的 `output/mnn/<name>`。

## 统一构建入口

设备产物构建脚本由本 Skill 维护：

```text
.codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

`project/linux/build_on_jetson.sh` 只保留为兼容入口，内部转发到上面的通用脚本；不要再把构建逻辑维护到 `project/linux`。

不显式设置 `JOBS` 时，脚本默认使用当前设备在线 CPU 总数的一半，最少为 1；这个默认值对 Jetson、Orange Pi 5 Plus 和 OnePlus 13T 三类目标都一致生效。本机编译、交叉编译和构建验证都遵守同一约定，除非用户明确给出 `JOBS` 或 `--parallel`。

优先显式把 build/install 放到本仓库：

```bash
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cuda" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson" \
JOBS=6 CUDA_ARCHS=72 CLEAN=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

脚本会自动判断当前脚本运行平台：

- Jetson 本机：走原有 CUDA 构建流程，使用 `/usr/local/cuda`、`nvcc`、自动探测或使用 `CUDA_ARCHS`。
- Linux x86_64：自动下载并校验 AArch64 Linux 交叉工具链到 `.cache/toolchains/`，再交叉编译 Jetson Linux 可运行产物。默认交叉编译不启用 CUDA，因为 x86 主机通常没有 Jetson CUDA sysroot；需要自行提供 CUDA/sysroot 时设置 `ENABLE_CROSS_CUDA=ON` 或 `MNN_CROSS_CUDA=ON`。
- `MNN_TARGET_DEVICE=orangepi5plus`：Linux x86_64 主机下载 Arm GNU 11.3 AArch64 Linux toolchain，交叉编译 Orange Pi 5 Plus Linux 产物，并默认打开 OpenCL/Vulkan。
- `MNN_TARGET_DEVICE=oneplus13t`：Linux x86_64 主机下载 Android NDK r29，交叉编译 Android arm64-v8a 产物，并默认打开 OpenCL/Vulkan。

Jetson 本机预期配置日志包含：

```text
CUDA architectures: 7.2
KleidiAI: OFF
Install after build: 1
Build MNNConvert: 1
Enabling CUDA support (... archs: sm_72)
-gencode arch=compute_72,code=sm_72
```

`.codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh` 会显式配置 `-DMNN_BUILD_CONVERTER=ON`，完整构建和快速目标构建后都会尝试把已有运行时产物同步到 `INSTALL_PREFIX`，其中必须包含可用于导出校验的 `MNNConvert`。

`CUDA_ARCHS=72`、`sm_72`、`compute_72` 都应该被归一化为 `7.2`。如果 CMake 报 `Unknown CUDA Architecture Name 72`，优先检查 `.codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh` 和 `source/backend/cuda/SelectCudaComputeArch.cmake` 的归一化逻辑。

## Linux x86_64 交叉编译 Jetson 产物

在 Linux x86_64 主机上直接运行同一个脚本即可自动进入交叉编译：

```bash
CLEAN=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

默认工具链：

```text
https://developer.arm.com/-/media/files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
```

脚本会下载压缩包到：

```text
.cache/toolchains/
```

缓存复用条件是本地压缩包存在，并且 `TOOLCHAIN_HASH_ALG` / `TOOLCHAIN_HASH` 校验通过；否则会重新下载。旧的 `.asc` 校验路径仍保留为兼容逻辑，但默认使用显式 checksum。

默认 Jetson x86_64 交叉工具链 checksum：

```text
gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz
md5 23ecc1dc528253c43e43365c6d923ec3
```

可覆盖项：

```bash
TOOLCHAIN_URL=<aarch64-linux-gnu-toolchain.tar.xz> \
TOOLCHAIN_HASH_ALG=sha256 \
TOOLCHAIN_HASH=<expected-checksum> \
TOOLCHAIN_CACHE_DIR="$PWD/.cache/toolchains" \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

注意不要使用 `gcc-arm-none-eabi` 这类裸机工具链。Jetson 是 Linux 用户态目标，必须使用 `aarch64-*-linux-gnu` 工具链；脚本解压后会检查 `aarch64*linux*gcc/g++`，不匹配就报错。

交叉编译默认输出：

```text
.cache/build/mnn/jetson_cross/
.cache/output/mnn/artifacts/jetson/
```

## Linux x86_64 交叉编译 Orange Pi 5 Plus 产物

目标是 Orange Pi 5 Plus 时使用：

```bash
MNN_TARGET_DEVICE=orangepi5plus CLEAN=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

默认工具链：

```text
arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
https://developer.arm.com/-/media/files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz
sha256 50cdef6c5baddaa00f60502cc8b59cc11065306ae575ad2f51e412a9b2a90364
```

Orange Pi 5 Plus 默认打开：

```text
MNN_OPENCL=ON
MNN_VULKAN=ON
MNN_CUDA=OFF
```

默认输出：

```text
.cache/build/mnn/orangepi5plus/
.cache/output/mnn/artifacts/orangepi5plus/
```

Orange Pi 5 Plus 设备默认登录方式：

```bash
ssh orangepi@192.168.101.113
```

需要推送产物或远端验证时，优先使用这个 SSH 目标；不要把它写成 Jetson 用户名或旧平台名。

## Linux x86_64 交叉编译 OnePlus 13T Android 产物

目标是 OnePlus 13T 时使用：

```bash
MNN_TARGET_DEVICE=oneplus13t CLEAN=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

默认 Android NDK：

```text
android-ndk-r29-linux.zip
https://dl.google.com/android/repository/android-ndk-r29-linux.zip
sha1 87e2bb7e9be5d6a1c6cdf5ec40dd4e0c6d07c30b
```

默认 Android 配置：

```text
ANDROID_ABI=arm64-v8a
ANDROID_PLATFORM=android-29
MNN_OPENCL=ON
MNN_VULKAN=ON
MNN_CUDA=OFF
```

默认输出：

```text
.cache/build/mnn/oneplus13t_android/
.cache/output/mnn/artifacts/oneplus13t/
```

## 暂停和清理残留构建

用户要求暂停或重新编译时，先停掉同仓库 MNN 相关构建进程，再重新启动。先查看：

```bash
ps -eo pid,ppid,pgid,stat,cmd | rg -n "build_artifacts|build_on_jetson|cmake --build|ninja|make|nvcc|cicc|ptxas" -S
```

只杀属于当前 MNN 仓库构建的进程。不要误杀无关系统进程；确认命令行里包含本仓库内的构建目录或脚本路径，例如：

```text
.cache/build/mnn
.codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
project/linux/build_on_jetson.sh
```

重编译使用 `CLEAN=1`；增量继续则不设置 `CLEAN`。

## CUDA Arch 和 KleidiAI

Jetson Xavier 用 `CUDA_ARCHS=72` 或 `7.2`；Orin 用 `87` 或 `8.7`。CUDA backend CMake 必须只追加 `CUDA_SELECT_NVCC_ARCH_FLAGS` 选出的 gencode，不要再无条件塞 `compute_60/61/62/70/72/75/80/86/87/89`。这对 CUDA 12/13 尤其重要。

Jetson CUDA LLM 构建默认关闭 KleidiAI：

```text
MNN_KLEIDIAI=OFF
```

如果看到 KleidiAI 的 `i8mm`、SME2 或 ARM feature 编译错误，先确认 configure 输出和 `CMakeCache.txt`：

```bash
rg -n "^(CUDA_ARCHS|MNN_KLEIDIAI|MNN_BUILD_CONVERTER):" .cache/build/mnn/jetson_cuda/CMakeCache.txt
```

## 产物检查

Jetson 构建成功后，优先检查 build 目录里的新鲜产物：

```text
.cache/build/mnn/jetson_cuda/libMNN.so
.cache/build/mnn/jetson_cuda/express/libMNN_Express.so
.cache/build/mnn/jetson_cuda/source/backend/cuda/libMNN_Cuda_Main.so
.cache/build/mnn/jetson_cuda/libllm.so
.cache/build/mnn/jetson_cuda/libpic_llm.so
.cache/build/mnn/jetson_cuda/tools/converter/libMNNConvertDeps.so
.cache/build/mnn/jetson_cuda/llm_demo
.cache/build/mnn/jetson_cuda/pic_llm_demo
.cache/build/mnn/jetson_cuda/llm_bench
.cache/build/mnn/jetson_cuda/pic_llm_bench
.cache/build/mnn/jetson_cuda/mls
.cache/build/mnn/jetson_cuda/pic_server
.cache/build/mnn/jetson_cuda/run_test.out
.cache/build/mnn/jetson_cuda/MNNConvert
```

最小检查：

```bash
file .cache/build/mnn/jetson_cuda/source/backend/cuda/libMNN_Cuda_Main.so \
     .cache/build/mnn/jetson_cuda/libllm.so \
     .cache/build/mnn/jetson_cuda/libpic_llm.so \
     .cache/build/mnn/jetson_cuda/MNNConvert \
     .cache/build/mnn/jetson_cuda/tools/converter/libMNNConvertDeps.so
```

`.codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh` 默认 `INSTALL_AFTER_BUILD=1`，完整构建后会先运行 `cmake --install "${BUILD_DIR}"`，再把运行时产物同步到 `INSTALL_PREFIX`：

```text
.cache/output/mnn/artifacts/jetson/lib/libMNN.so
.cache/output/mnn/artifacts/jetson/lib/libMNN_Express.so
.cache/output/mnn/artifacts/jetson/lib/libMNN_Cuda_Main.so
.cache/output/mnn/artifacts/jetson/lib/libllm.so
.cache/output/mnn/artifacts/jetson/lib/libpic_llm.so
.cache/output/mnn/artifacts/jetson/lib/libMNNConvertDeps.so
.cache/output/mnn/artifacts/jetson/bin/llm_demo
.cache/output/mnn/artifacts/jetson/bin/pic_llm_demo
.cache/output/mnn/artifacts/jetson/bin/llm_bench
.cache/output/mnn/artifacts/jetson/bin/pic_llm_bench
.cache/output/mnn/artifacts/jetson/bin/mls
.cache/output/mnn/artifacts/jetson/bin/pic_server
.cache/output/mnn/artifacts/jetson/bin/run_test.out
.cache/output/mnn/artifacts/jetson/bin/MNNConvert
```

如果只用 `BUILD_TARGET=<target>` 做快速增量构建，脚本会跳过 `cmake --install`，但仍会把 build 目录中已经存在的运行时产物同步到 `INSTALL_PREFIX`。需要完全跳过同步时显式设置：

```bash
INSTALL_AFTER_BUILD=0 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

## 常见失败点

- `Unknown CUDA Architecture Name 72`: 传 `7.2` 或修复 arch 归一化。
- CUDA 12/13 编译旧 arch 失败: CUDA CMake 仍在硬编码多架构 gencode，改成只用选中 arch。
- KleidiAI ARM feature 报错: Jetson CUDA 构建强制 `MNN_KLEIDIAI=OFF`。
- `CUDART_INF_F` 未定义: CUDA 源里优先用 `<float.h>` 的 `FLT_MAX`。
- `pic_llm` 报 `Llm::beginPagedRequestIfNeeded` 无声明: `transformers/pic_llm/engine/CMakeLists.txt` 的 include 目录要优先于普通 `transformers/llm/engine/include`。

## x64 构建

Linux x64 可直接用 CMake 在本仓库 `.cache` 下构建：

```bash
cmake -S . -B .cache/build/mnn/x64 \
  -DMNN_CUDA=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_BUILD_TEST=ON \
  -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_BUILD_SHARED_LIBS=ON
TOTAL_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
JOBS="${JOBS:-$((TOTAL_CPUS > 1 ? TOTAL_CPUS / 2 : 1))}"
cmake --build .cache/build/mnn/x64 --parallel "${JOBS}"
cmake --install .cache/build/mnn/x64 --prefix .cache/output/mnn/artifacts/x64
mkdir -p .cache/output/mnn/artifacts/x64/bin .cache/output/mnn/artifacts/x64/lib
install -m 755 .cache/build/mnn/x64/libMNN.so .cache/output/mnn/artifacts/x64/lib/ 2>/dev/null || true
install -m 755 .cache/build/mnn/x64/express/libMNN_Express.so .cache/output/mnn/artifacts/x64/lib/ 2>/dev/null || true
install -m 755 .cache/build/mnn/x64/source/backend/cuda/libMNN_Cuda_Main.so .cache/output/mnn/artifacts/x64/lib/ 2>/dev/null || true
install -m 755 .cache/build/mnn/x64/libllm.so .cache/output/mnn/artifacts/x64/lib/ 2>/dev/null || true
install -m 755 .cache/build/mnn/x64/libpic_llm.so .cache/output/mnn/artifacts/x64/lib/ 2>/dev/null || true
install -m 755 .cache/build/mnn/x64/MNNConvert .cache/output/mnn/artifacts/x64/bin/
install -m 755 .cache/build/mnn/x64/tools/converter/libMNNConvertDeps.so .cache/output/mnn/artifacts/x64/lib/
install -m 755 .cache/build/mnn/x64/llm_bench .cache/output/mnn/artifacts/x64/bin/ 2>/dev/null || true
install -m 755 .cache/build/mnn/x64/pic_llm_bench .cache/output/mnn/artifacts/x64/bin/ 2>/dev/null || true
install -m 755 .cache/build/mnn/x64/llm_demo .cache/output/mnn/artifacts/x64/bin/ 2>/dev/null || true
install -m 755 .cache/build/mnn/x64/pic_llm_demo .cache/output/mnn/artifacts/x64/bin/ 2>/dev/null || true
install -m 755 .cache/build/mnn/x64/mls .cache/output/mnn/artifacts/x64/bin/ 2>/dev/null || true
install -m 755 .cache/build/mnn/x64/run_test.out .cache/output/mnn/artifacts/x64/bin/ 2>/dev/null || true
```

只重编某个 target 时指定 target：

```bash
TOTAL_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
JOBS="${JOBS:-$((TOTAL_CPUS > 1 ? TOTAL_CPUS / 2 : 1))}"
cmake --build .cache/build/mnn/x64 --target <target> --parallel "${JOBS}"
```
