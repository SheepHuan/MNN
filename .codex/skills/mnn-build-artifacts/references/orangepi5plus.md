# Orange Pi 5 Plus

适用场景：

- x64 主机交叉编译 Orange Pi 5 Plus Linux OpenCL/Vulkan 产物

## 默认入口

完整构建：

```bash
MNN_TARGET_DEVICE=orangepi5plus \
CLEAN=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

快速增量：

```bash
MNN_TARGET_DEVICE=orangepi5plus \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

## 预期配置

- toolchain: `arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu`
- `MNN_OPENCL=ON`
- `MNN_VULKAN=ON`
- `MNN_CUDA=OFF`

默认目录：

- build: `.cache/build/mnn/orangepi5plus/`
- install: `.cache/output/mnn/artifacts/orangepi5plus/`

## 交叉编译防错

OrangePi 和 Rhino 都用 Arm GNU 11.3，但它们不能再和 Jetson 共用一个 toolchain file 路径。脚本现在会：

- 为 OrangePi 生成独立 toolchain file
- 如果发现 build dir 缓存的是别的 `compiler/sysroot/toolchain`，先删目录再重配

如果之前手工跑过 `cmake --build .cache/build/mnn/orangepi5plus`，先回到脚本并显式加一次 `CLEAN=1`。

## 产物检查

```bash
file .cache/output/mnn/artifacts/orangepi5plus/bin/pic_server \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN.so \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN_CL.so
```

推设备时只同步 artifact，不同步源码：

```bash
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

## 常见失败点

- OpenCL 产物缺失：检查 `CMakeCache.txt` 里的 `MNN_OPENCL:BOOL=ON`
- `run_test.out` 缺失：默认 release build 不带测试
- 设备验证时端口占用：直接 kill 旧进程，不要反复换端口
