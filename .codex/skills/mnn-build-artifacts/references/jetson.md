# Jetson

适用场景：

- Jetson 本机 CUDA 构建
- x64 主机交叉编译 Jetson Linux 用户态产物

## 默认入口

Jetson 本机：

```bash
MNN_TARGET_DEVICE=jetson \
CUDA_ARCHS=72 \
CLEAN=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

x64 主机交叉编译 Jetson：

```bash
MNN_TARGET_DEVICE=jetson \
CROSS_COMPILE=ON \
CLEAN=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

x64 主机交叉编译 Jetson CUDA 产物时，需要本仓库的 Jetson CUDA
sysroot：

```text
.cache/sysroots/jetson_cuda/targets/aarch64-linux/include
.cache/sysroots/jetson_cuda/targets/aarch64-linux/lib/libcudart.so
.cache/sysroots/jetson_cuda/targets/aarch64-linux/lib/libcublas.so
```

`build_artifacts.sh` 会在 `ENABLE_CROSS_CUDA=ON` 时自动导出
`CUDA_TOOLKIT_ROOT=.cache/sysroots/jetson_cuda`，让 CMake legacy
`FindCUDA` 解析到 `targets/aarch64-linux`。如果 sysroot 在别处，显式传：

```bash
MNN_JETSON_CUDA_SYSROOT=/path/to/jetson_cuda/targets/aarch64-linux
```

快速增量只编运行时：

```bash
MNN_TARGET_DEVICE=jetson \
CROSS_COMPILE=ON \
ENABLE_CROSS_CUDA=ON \
CUDA_ARCHS=72 \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

## 预期产物

- build: `.cache/build/mnn/jetson_cuda/`、`.cache/build/mnn/jetson_cross/` 或显式 `BUILD_DIR`
- install: `.cache/output/mnn/artifacts/jetson/` 或显式 `INSTALL_PREFIX`

至少检查：

```bash
file .cache/output/mnn/artifacts/jetson/bin/pic_server \
     .cache/output/mnn/artifacts/jetson/lib/libMNN.so \
     .cache/output/mnn/artifacts/jetson/lib/libMNN_Cuda_Main.so
```

## 必查日志

Jetson CUDA configure 输出至少包含：

- `CUDA support: ON`
- `CUDA architectures: 7.2` 或目标实际架构
- `KleidiAI: OFF`
- `MNN_BUILD_CONVERTER=ON`

## 常见失败点

- `Unknown CUDA Architecture Name 72`
  - 只传 `72` / `7.2` / `sm_72` / `compute_72`
- CUDA 12/13 旧 arch 报错
  - 先检查 `source/backend/cuda/SelectCudaComputeArch.cmake`
- KleidiAI ARM feature 报错
  - Jetson CUDA 默认保持 `MNN_KLEIDIAI=OFF`
- `run_test.out` 缺失
  - 默认 release build 不带测试；需要时额外加 `CMAKE_ARGS='-DMNN_BUILD_TEST=ON'`

## 残留进程

只清当前仓库构建进程：

```bash
ps -eo pid,ppid,pgid,stat,cmd | rg -n "build_artifacts|cmake --build|ninja|make|nvcc|cicc|ptxas" -S
```

命令行里必须能看到本仓库 `.cache/build/mnn` 路径，再决定是否 kill。
