---
name: mnn-ops-bench
description: 当用户要求在本 MNN 仓库的 test/bench_ops/cuda 下新增、运行或诊断 MNN CUDA 单算子精度/性能测试，或要求用 CUDA event 统计直接算子 kernel 耗时时使用这个 Skill。
metadata:
  short-description: MNN CUDA 单算子精度和性能测试
---

# MNN Ops Bench

本 Skill 用于在本 MNN 仓库中维护 CUDA 单算子测试。测试入口在：

```text
test/bench_ops/cuda/
```

目标是直接调用 MNN backend 的算子实现：

```text
Runtime -> Backend -> Backend::onCreate(inputs, outputs, op) -> Execution::onResize -> Execution::onExecute
```

不要用 Express/Module 包裹性能测试；性能数据必须由 CUDA event 包住重复 `onExecute` 得到。

## 测试拆分

精度测试和性能测试分开维护：

```text
CudaOpBenchUtils.hpp          公共 direct-op runtime/backend/tensor/op 工具
CudaAttentionAccuracy.cpp     CPU direct op vs CUDA direct op 输出对比
CudaAttentionPerf.cpp         CUDA event 统计 onExecute 平均耗时
```

注册名按功能分组：

```text
bench_ops/cuda/accuracy/LinearAttention
bench_ops/cuda/accuracy/PagedAttention
bench_ops/cuda/accuracy/PagedAttention/CompareAttention
bench_ops/cuda/perf/LinearAttention
bench_ops/cuda/perf/LinearAttention/Prefill
bench_ops/cuda/perf/LinearAttention/Decode
bench_ops/cuda/perf/Attention
bench_ops/cuda/perf/Attention/Prefill
bench_ops/cuda/perf/Attention/Decode
bench_ops/cuda/perf/PagedAttention
bench_ops/cuda/perf/PagedAttention/Prefill
bench_ops/cuda/perf/PagedAttention/Decode
```

当前 CUDA direct-op attention bench 覆盖：

```text
Attention        MNN 自身 softmax attention，用作 PagedAttention 同语义 baseline
PagedAttention   本仓库新增 paged KV softmax attention
LinearAttention  MNN 自身 linear attention，语义不同，只做 CPU/CUDA 精度和速度单列展示
```

`PagedAttention` 与 `Attention` 做逐元素精度对比；`LinearAttention` 不和 `PagedAttention` 做精度对比，因为二者 attention 语义不同。

## 精度测试原则

1. 使用同一个 FlatBuffer `Op` 参数，分别创建 CPU backend 和 CUDA backend 的 `Execution`。
2. 输入使用确定性 host 数据，写入 CPU tensor 和 CUDA tensor。
3. CUDA 输出通过 `cudaMemcpy(device->host)` 读回。
4. 对比 CPU 输出和 CUDA 输出，打印 `max_abs`、`max_rel` 和 bad element 数。
5. 精度测试只跑 fp32 tensor，对应 `run_test.out ... 2 1` 或 `... 2 0`；fp16/BF16 不做这个 direct float copy 对比。
6. 对带状态的算子必须覆盖状态延续路径，例如 `prefill -> decode`。
7. `PagedAttention/CompareAttention` 使用同一组 Llama 形状，CUDA `PagedAttention` 对比 CUDA `Attention`。`Attention` prefill 需要显式 causal additive mask，`PagedAttention` 的 causal 规则在算子内部完成。

## 性能测试原则

1. 只创建 CUDA backend。
2. `onResize` 在计时前完成。
3. warmup 不计时。
4. 用 `cudaEventRecord(start)` 和 `cudaEventRecord(stop)` 包住重复 `onExecute`，最终输出平均 ms 和 us/token。
5. 不把 host timer、Express 调度或 Module 构建计入结果。
6. 如果要测 decode with cache，计时循环里要保持 logical KV 长度稳定，避免每轮无限增长导致 case 漂移。
7. speed case 拆成 `Prefill` 和 `Decode` 两类；decode 的 `past` 等于 context，`add=1`。
8. Llama 形状固定为：

```text
llama3.2-1B  qH=32 kvH=8 D=64
llama3.2-3B  qH=24 kvH=8 D=128
llama3.2-8B  qH=32 kvH=8 D=128
context       512, 1024, 2048
```

## 新增一个 CUDA 单算子测试

如果算子语义还没有进 MNN，先使用 `$mnn-add-new-op` 完成 schema、shape、CPU/CUDA backend 和注册表。已有算子只需要补测试。

新增测试时：

1. 在 `CudaOpBenchUtils.hpp` 中复用 `DirectOpBench`、`OpHolder`、`CudaEventPair`、`compareVectors`。
2. 如果新算子有专属参数，新增一个 `makeXxxOp(...)` 工厂，或在本算子的 `.cpp` 中局部构造 `OpT`。
3. 在 `CudaXxxAccuracy.cpp` 中创建 CPU/CUDA 两条 direct execution 路径，喂相同输入并对比输出。
4. 在 `CudaXxxPerf.cpp` 中创建 CUDA execution，`onResize` 后用 CUDA event 统计重复 `onExecute`。
5. 注册名使用：

```text
bench_ops/cuda/accuracy/<OpName>
bench_ops/cuda/perf/<OpName>
```

6. `test/CMakeLists.txt` 已经在 `MNN_CUDA=OFF` 时排除 `test/bench_ops/cuda/*.cpp`；不要把 CUDA-only 测试塞进普通 `test/op`。

## 构建测试产物

构建和产物路径遵循 `$mnn-build-artifacts`。Jetson 上从本 MNN 仓库根目录运行：

```bash
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cuda" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson" \
JOBS=6 CUDA_ARCHS=72 CMAKE_ARGS="-DMNN_BUILD_TEST=ON" \
BUILD_TARGET=run_test.out INSTALL_AFTER_BUILD=1 \
bash project/linux/build_on_jetson.sh
```

需要全量产物时按 `$mnn-build-artifacts` 使用默认构建；只验证单算子测试时优先构建 `run_test.out` target。

## 产物自动定位

运行测试时按当前平台名从 output artifacts 找产物：

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
RUN_TEST="$MNN_ARTIFACT_ROOT/bin/run_test.out"
```

## 执行测试

从本 MNN 仓库根目录运行，确保 CUDA backend so 能被 loader 找到：

```bash
export LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

精度测试：

```bash
"$RUN_TEST" bench_ops/cuda/accuracy 2 1
"$RUN_TEST" bench_ops/cuda/accuracy/LinearAttention 2 1
"$RUN_TEST" bench_ops/cuda/accuracy/PagedAttention 2 1
"$RUN_TEST" bench_ops/cuda/accuracy/PagedAttention/CompareAttention 2 1
```

性能测试：

```bash
"$RUN_TEST" bench_ops/cuda/perf 2 1
"$RUN_TEST" bench_ops/cuda/perf/LinearAttention 2 1
"$RUN_TEST" bench_ops/cuda/perf/LinearAttention/Prefill 2 1
"$RUN_TEST" bench_ops/cuda/perf/LinearAttention/Decode 2 1
"$RUN_TEST" bench_ops/cuda/perf/Attention 2 1
"$RUN_TEST" bench_ops/cuda/perf/Attention/Prefill 2 1
"$RUN_TEST" bench_ops/cuda/perf/Attention/Decode 2 1
"$RUN_TEST" bench_ops/cuda/perf/PagedAttention 2 1
"$RUN_TEST" bench_ops/cuda/perf/PagedAttention/Prefill 2 1
"$RUN_TEST" bench_ops/cuda/perf/PagedAttention/Decode 2 1
```

参数含义：

```text
2 = MNN_FORWARD_CUDA
1 = BackendConfig::Precision_High
```

## 结果判断

精度测试通过时会打印每个子路径的误差摘要，例如：

```text
[bench_ops/cuda/accuracy] LinearAttention/prefill      max_abs=... max_rel=... bad=0/N
```

性能测试通过时会打印每个 case 的 CUDA event 平均耗时：

```text
[bench_ops/cuda/perf/PagedAttention] decode_ctx2048... avg=... ms ... us/token
```

汇总结果时，性能表必须包含这些列：

```text
stage | model | ctx | qH | kvH | D | op | avg_ms | us/token | vs_Attention
```

精度表必须包含这些列：

```text
op | ref_op | stage | model | ctx | qH | kvH | D | max_abs | max_rel | bad
```

从日志生成 Markdown 表：

```bash
python3 .codex/skills/mnn-ops-bench/scripts/summarize_attention_logs.py .cache/bench_ops
```

如果 `$RUN_TEST` 不存在，先按本 Skill 的构建命令打开 `MNN_BUILD_TEST=ON`。如果 CUDA backend 未注册或动态库找不到，先按 `$mnn-build-artifacts` 检查 `$MNN_ARTIFACT_ROOT/lib/libMNN_Cuda_Main.so` 和 `LD_LIBRARY_PATH`。
