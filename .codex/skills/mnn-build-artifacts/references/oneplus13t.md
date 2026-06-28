# OnePlus 13T

适用场景：

- x64 主机交叉编译 OnePlus 13T Android arm64-v8a 产物

## 默认入口

```bash
MNN_TARGET_DEVICE=oneplus13t \
CLEAN=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

快速增量：

```bash
MNN_TARGET_DEVICE=oneplus13t \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

## 预期配置

- NDK: `android-ndk-r29`
- `ANDROID_ABI=arm64-v8a`
- `ANDROID_PLATFORM=android-29`
- `MNN_OPENCL=ON`
- `MNN_VULKAN=ON`
- `MNN_CUDA=OFF`

默认目录：

- build: `.cache/build/mnn/oneplus13t_android/`
- install: `.cache/output/mnn/artifacts/oneplus13t/`

## 产物检查

```bash
file .cache/output/mnn/artifacts/oneplus13t/lib/libMNN.so
```

如果需要 adb 推送或 Android 侧装载，再按用户实际设备路径处理；本 skill 只负责本地产物构建与 install。

## 常见失败点

- NDK 路径错：优先让脚本自己下载并缓存
- ABI/Platform 手工改错：先回到默认 `arm64-v8a + android-29`
- `run_test.out` 缺失：默认 release build 不带测试
