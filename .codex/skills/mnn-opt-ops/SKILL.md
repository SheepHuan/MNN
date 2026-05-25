---
name: mnn-opt-ops
description: 当用户要求优化本 MNN 仓库中的 CPU/CUDA 算子性能、分析算子慢因、在本机修改编译并同步到 Jetson 运行验证，或需要记录 Jetson 设备与 Attention/PagedAttention bench 流程时使用这个 Skill。
metadata:
  short-description: MNN 算子性能优化和 Jetson 验证流程
---

# MNN Opt Ops

本 Skill 用于优化 MNN CPU/CUDA 算子并闭环验证性能。它只记录优化流程和设备信息；构建细节引用 `$mnn-build-artifacts`，单算子 bench 细节引用 `$mnn-ops-bench`。

## 入口约束

1. 从本 MNN 仓库根目录工作。
2. 修改前先运行：

```bash
git status --short
```

3. 不要读取或修改 `schema/private/`、`source/internal/`。
4. 构建产物、日志、临时文件放到 `.cache/`、`output/` 或用户指定本地目录，不要提交。

## 相关 Skills

- 构建、安装、产物检查：读 `.codex/skills/mnn-build-artifacts/SKILL.md`。
- CUDA direct-op 精度/性能测试：读 `.codex/skills/mnn-ops-bench/SKILL.md`。
- 如果优化需要新增 op/schema/backend 注册：再读 `.codex/skills/mnn-add-new-op/SKILL.md`。

## Jetson 设备

Jetson 可免密登录：

```text
jetson@192.168.101.192
```

远端 MNN 仓库根目录：

```text
/home/jetson/code/kvshare-edge/impl/MNN
```

远端 `.cache` 工作目录：

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache
```

## 本机修改与交叉编译

优先在本机完成源码修改和编译。若需要快速发现普通 CPU/C++ 语法问题，先构建本机 `run_test.out`；若目标是 Jetson CUDA 性能，使用本机 `nvcc` + AArch64 GNU Linux toolchain + Jetson CUDA target sysroot 交叉编译 Jetson 产物，然后只把产物推到 Jetson 执行。

AArch64 GNU toolchain 自带 glibc/libstdc++ sysroot，但不带 CUDA headers/libs。CUDA cross 编译还需要 Jetson 的 CUDA target sysroot，例如：

```text
/usr/local/cuda-12.2/targets/aarch64-linux
```

首次准备本机 CUDA target sysroot：

```bash
mkdir -p .cache/sysroots/jetson_cuda/targets
rsync -a --delete \
  jetson@192.168.101.192:/usr/local/cuda-12.2/targets/aarch64-linux/ \
  .cache/sysroots/jetson_cuda/targets/aarch64-linux/
```

本机交叉编译 Jetson CUDA `run_test.out`：

```bash
CUDA_TOOLKIT_ROOT="$PWD/.cache/sysroots/jetson_cuda" \
CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda" \
JOBS=8 CUDA_ARCHS=72 ENABLE_CROSS_CUDA=ON \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS="-DMNN_BUILD_TEST=ON -DCUDA_HOST_COMPILER=$PWD/.cache/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++" \
bash .codex/skills/mnn-build-artifacts/scripts/build_jetson_artifacts.sh
```

期望配置日志包含：

```text
Native Jetson build: 0
CUDA support: ON
CUDA architectures: 7.2
Found CUDA: .../.cache/sysroots/jetson_cuda
Enabling CUDA support (... archs: sm_72)
```

推送交叉编译产物到 Jetson：

```bash
rsync -a --delete \
  .cache/output/mnn/artifacts/jetson_cross_cuda/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

Jetson 原生构建只作为 fallback：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  BUILD_DIR="$PWD/.cache/build/mnn/jetson_cuda" \
  INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson" \
  JOBS=6 CUDA_ARCHS=72 CMAKE_ARGS="-DMNN_BUILD_TEST=ON" \
  BUILD_TARGET=run_test.out INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_jetson_artifacts.sh'
```

## 同步到 Jetson

默认只同步交叉编译产物。如果确实需要同步源码改动到远端同路径仓库，避免同步 `.cache/`、`output/`、构建下载目录：

```bash
rsync -a --delete \
  --exclude .git \
  --exclude .cache \
  --exclude output \
  --exclude 3rd_party/cutlass \
  ./ jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/
```

如果远端有用户未保存改动，先查看远端 `git status --short`，不要覆盖。

## Attention/PagedAttention Bench

远端运行普通 Attention perf：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  export ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  mkdir -p .cache/bench_ops/cross_cuda_latest && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  "$ART/bin/run_test.out" bench_ops/cuda/perf/Attention 2 1 2>&1 | tee .cache/bench_ops/cross_cuda_latest/attention_perf.log'
```

远端运行 PagedAttention perf：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  export ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  mkdir -p .cache/bench_ops/cross_cuda_latest && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  "$ART/bin/run_test.out" bench_ops/cuda/perf/PagedAttention 2 1 2>&1 | tee .cache/bench_ops/cross_cuda_latest/paged_attention_perf.log'
```

远端运行 PagedAttention 精度对比：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  export ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  mkdir -p .cache/bench_ops/cross_cuda_latest && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  "$ART/bin/run_test.out" bench_ops/cuda/accuracy/PagedAttention/CompareAttention 2 1 2>&1 | tee .cache/bench_ops/cross_cuda_latest/paged_vs_attention_accuracy.log'
```

汇总日志：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  python3 .codex/skills/mnn-ops-bench/scripts/summarize_attention_logs.py .cache/bench_ops/cross_cuda_latest'
```

## 优化判断

报告结果时至少说明：

- 修改了 CPU 还是 CUDA 路径，核心瓶颈是什么。
- 精度是否通过。
- 性能表包含 `stage | model | ctx | qH | kvH | D | op | latency_ms | Attention/PagedAttention`，其中 `Attention/PagedAttention > 1` 表示普通 `Attention` 比 `PagedAttention` 慢。
- 与优化前的 Jetson 数据相比，PagedAttention 的 prefill/decode 延迟和 `Attention/PagedAttention` 倍数是否改善。
