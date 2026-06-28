---
name: mnn-build-artifacts
description: 当用户要求在本 MNN 仓库中构建、重编、安装、检查或诊断 MNN 产物时使用，覆盖 Jetson CUDA、x64 主机交叉编译 Jetson、Orange Pi 5 Plus OpenCL/Vulkan、Rhino Pi-X1/Aidlux Adreno OpenCL、OnePlus 13T Android 以及本地 x64 构建；也用于排查 build dir 残留、toolchain/sysroot 混用、CUDA_ARCHS、KleidiAI、pic_server/libpic_llm/MNNConvert 等产物问题。
metadata:
  short-description: 编译和检查多设备 MNN 产物
---

# MNN Build Artifacts

本 skill 只负责本仓库构建和安装产物。`SKILL.md` 只保留分流和通用约束；每个设备的构建入口单独放在 `references/`。

## 通用约束

1. 从本 MNN 仓库根目录执行，修改或构建前先运行：

```bash
git status --short
```

2. 构建目录、install 目录和临时产物只放在本仓库 `.cache/` 下；不要提交这些文件。
3. 默认使用 `.codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh`，不要把设备构建命令散落到别处。
4. 没有用户显式指定时，`JOBS` 默认使用一半 CPU；不要默认跑满机器。
5. 需要设备验证时，优先本地交叉编译再同步 artifact；不要把源码推到设备上重新编译，除非用户明确要求。
6. `BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1` 是默认的快速运行时增量构建口径；完整产物安装再去掉 `BUILD_TARGET`。
7. `run_test.out` 不是默认 release 产物。只有显式附加 `-DMNN_BUILD_TEST=ON` 或单独 testbench build 时才应期待它存在。
8. 只看“链接成功”不够；至少再检查一次 `CMakeCache.txt`、`file` 输出、目标时间戳和 `INSTALL_PREFIX` 下的实际产物。

## 交叉编译防错规则

Linux AArch64 交叉构建现在必须遵守这两条：

1. 不同目标设备使用独立的 toolchain file 路径，不能再共用一个 `jetson-aarch64.toolchain.cmake`。
2. 脚本会在重配前检查已有 build dir 的 `CMAKE_TOOLCHAIN_FILE`、`CMAKE_C_COMPILER`、`CMAKE_CXX_COMPILER`、`CMAKE_SYSROOT`；如果和当前目标不一致，会自动删掉 stale build dir 再重新 configure。

这解决的是之前 Rhino/Adreno build 目录里出现的“编译器已经切到 Arm GNU 11.3，但 sysroot 还停在 Jetson gcc-arm-9.2”这类混配错误。

即便有这层保护，下面两种情况第一次重跑仍优先显式加 `CLEAN=1`：

- 刚切换目标设备。
- 最近手工直接跑过 `cmake -S/-B` 或 `cmake --build`，绕开了脚本。

## 分流

根据目标设备或构建环境，只读对应文档：

- Jetson 本机 CUDA，或 x64 主机交叉编译 Jetson：读 [references/jetson.md](references/jetson.md)
- Orange Pi 5 Plus OpenCL/Vulkan：读 [references/orangepi5plus.md](references/orangepi5plus.md)
- Rhino Pi-X1 / Aidlux Adreno OpenCL：读 [references/aidlux-adreno-opencl.md](references/aidlux-adreno-opencl.md)
- OnePlus 13T Android：读 [references/oneplus13t.md](references/oneplus13t.md)
- 本地 x64 Linux：读 [references/x64.md](references/x64.md)

如果任务是在改 `build_artifacts.sh` 本身，除了本页，还要读你要影响的那个设备文档；不要只改脚本不更新对应入口说明。

## 过程检查

编辑 skill 或脚本后，至少做这两步：

```bash
bash -n .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
python .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/mnn-build-artifacts
```

如果改动涉及某个设备的默认构建路径，再按对应设备文档做一次最小构建验证。
