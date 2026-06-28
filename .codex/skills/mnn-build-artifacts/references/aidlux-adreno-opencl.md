# Rhino Pi-X1 / Aidlux Adreno OpenCL

适用场景：

- x64 主机交叉编译 Rhino Pi-X1 / Aidlux Linux AArch64 OpenCL 产物

## 默认入口

完整构建：

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
CLEAN=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

快速增量：

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

## 预期配置

- toolchain: `arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu`
- `MNN_OPENCL=ON`
- `MNN_VULKAN=OFF`
- `MNN_CUDA=OFF`

默认目录：

- build: `.cache/build/mnn/aidlux_adreno_opencl/`
- install: `.cache/output/mnn/artifacts/aidlux_adreno_opencl/`

## 这条链的核心防错点

Rhino 最近反复踩到的不是代码问题，而是交叉编译缓存被污染：

- 编译器切到了 Arm GNU 11.3
- 但 `CMAKE_SYSROOT` 还残留在 Jetson 的 `gcc-arm-9.2`

脚本现在必须保证两件事：

1. Rhino 使用自己独立的 toolchain file 路径，不再和 Jetson 共用。
2. 如果已有 build dir 的 `CMAKE_TOOLCHAIN_FILE / CMAKE_C_COMPILER / CMAKE_CXX_COMPILER / CMAKE_SYSROOT` 与当前 Rhino 目标不一致，脚本先删掉 build dir 再重新 configure。

如果你手工直接跑过 `cmake -S/-B` 或 `cmake --build`，不要继续沿用那个 build dir；回到脚本并显式 `CLEAN=1`。

## 产物检查

```bash
file .cache/output/mnn/artifacts/aidlux_adreno_opencl/bin/pic_server \
     .cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN.so \
     .cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN_CL.so
```

同步设备时只推 artifact：

```bash
rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

## 运行时提示

- `LD_LIBRARY_PATH` 一般需要带上：
  - `/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/lib`
  - `/usr/lib`
  - `/usr/lib/aarch64-linux-gnu`
- 如果 OpenCL loader 没选中 Adreno，可在设备侧附加：
  - `LD_PRELOAD=/usr/lib/libOpenCL_adreno.so`

## 常见失败点

- build 期直接炸在 libc/pthread/bits 头文件
  - 先怀疑 `sysroot` 和 `compiler` 混配，而不是源码本身
- `run_test.out` 缺失
  - 默认 release build 不带测试；需要 bench test 时单独开 testbench 配置
- 远端验证端口被占
  - 直接杀旧 `pic_server`，不要换端口逃避
