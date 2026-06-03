---
name: mnn-llm-bench
description: 当用户要求运行 MNN llm_bench、测试 CUDA/OpenCL/Vulkan/CPU LLM 推理、确认 GPU 后端是否注册、检查 CUDA Execution class 日志、诊断后端回退到 CPU 时使用这个 Skill，帮助用当前平台 output artifacts 产物完成验证。
metadata:
  short-description: 运行和诊断 MNN llm_bench
---

# MNN LLM Bench

本 Skill 由本 MNN 仓库维护，用于使用本仓库 `.cache/output/mnn/artifacts/<platform>/bin/llm_bench` 或 `pic_llm_bench` 测试 MNN LLM 模型，也用于启动和验证同一 artifact root 下的 `mls` OpenAI-compatible server。PIC LLM / PagedAttention 导出模型优先使用 `pic_llm_bench`，不要混用链接 `libllm.so` 的 `llm_bench` 做最终稳定性判断。

## 工作流

1. 自动解析 `MNN_ARTIFACT_ROOT`，确认 `$MNN_ARTIFACT_ROOT/bin/llm_bench`、需要时的 `$MNN_ARTIFACT_ROOT/bin/mls` 和目标 backend 动态库存在。
2. 组装一条从仓库根目录可直接复制执行的完整命令。
3. 执行前把完整命令发给用户；如果命令很长，用 fenced `bash` 代码块。
4. 执行命令并保存关键输出。
5. 检查是否出现 backend fallback 日志。
6. 检查是否出现对应 backend 的 execution class-name 日志：
   `[CPUExecution]`、`[CUDAExecution]`、`[OpenCLExecution]` 或 `[VulkanExecution]`。
7. 结束时再次给出可复现命令、执行目录、模型路径、backend、关键结果和失败/回退判断。
8. OpenCL/Vulkan loader、ICD 或动态库注册问题按 `docs/mnn_environment.md` 继续排查。

## 补编/重编约定

`mnn-llm-bench` 主要使用 `.cache/output/mnn/artifacts/<platform>/` 下已有产物运行验证。只有当 `llm_bench`、`mls` 或 backend 动态库缺失，且用户同意或明确要求补编时，才启动编译。

编译目录必须集中在本仓库固定根目录 `.cache/build/` 下，不要散落到 `build/`、`cmake-build-*`、临时目录或多个互不相关的位置。不同平台和编译模式只用子目录区分：

```text
.cache/build/jetson_cuda
.cache/build/jetson_opencl
.cache/build/jetson_vulkan
.cache/build/x64_cuda
.cache/build/x64_cpu
```

生成补编命令时必须显式传 `BUILD_DIR`，不要依赖构建脚本默认值。

Jetson CUDA 补编示例：

```bash
BUILD_DIR="$PWD/.cache/build/jetson_cuda" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson" \
JOBS=6 CUDA_ARCHS=72 CLEAN=1 \
bash project/linux/build_on_jetson.sh
```

补编后仍只从 `MNN_ARTIFACT_ROOT="$PWD/.cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM"` 运行 bench；不要直接混用 build 目录里的二进制做最终验证，除非是在定位构建产物同步问题，并且最终回复中明确说明。

## Codex 执行约定

当 Codex 通过本 Skill 帮用户执行 `llm_bench` 时，必须让用户能复制并复现结果。

执行前回复：

```text
我将从 <repo_root> 执行下面命令：
```

随后给出完整命令。命令必须包含：

```text
LD_LIBRARY_PATH=...
LD_PRELOAD=...        仅 OpenCL/Vulkan 需要
$MNN_ARTIFACT_ROOT/bin/llm_bench
-m <config.json>
-a <backend>
完整 bench 参数
```

不要只说“运行 llm_bench”或只给参数片段。不要依赖用户当前 shell 已经 export 过的变量；用于复现的命令里直接内联 `LD_LIBRARY_PATH` 和必要的 `LD_PRELOAD`。
如果命令中使用 `$MNN_ARTIFACT_ROOT`，要么在同一个代码块里包含平台解析片段，要么先把它解析成具体绝对路径后再给用户。

执行后回复必须包含：

```text
复现命令：
<同一条完整命令>

结果摘要：
backend: <cuda|opencl|vulkan|cpu>
model: <config.json>
fallback: <yes|no|unknown>
execution_class_log: <yes|no|not_applicable>
exit_code: <code>
```

如果命令失败，保留同样的复现命令，并摘要关键错误日志。不要把超长日志全文贴出；优先摘出 backend 注册、loader、fallback、execution class 和最终错误相关行。

## 基础环境

从仓库根目录执行：

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
CUDA_LIB_DIR=${CUDA_LIB_DIR:-$(dirname "$(dirname "$(command -v nvcc)")")/lib64}
export LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
```

给用户复现时优先使用内联环境变量形式：

```bash
CUDA_LIB_DIR=${CUDA_LIB_DIR:-$(dirname "$(dirname "$(command -v nvcc)")")/lib64}
LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
"$MNN_ARTIFACT_ROOT/bin/llm_bench" ...
```

检查产物：

```text
$MNN_ARTIFACT_ROOT/bin/llm_bench
$MNN_ARTIFACT_ROOT/bin/mls
$MNN_ARTIFACT_ROOT/lib/libMNN.so
$MNN_ARTIFACT_ROOT/lib/libMNN_Cuda_Main.so
$MNN_ARTIFACT_ROOT/lib/libMNN_CL.so
$MNN_ARTIFACT_ROOT/lib/libMNN_Vulkan.so
```

MNN LLM 模型目录通常包含：

```text
config.json
llm_config.json
llm.mnn
llm.mnn.weight
```

示例模型配置路径用仓库内相对变量承载；实际运行时按用户给定模型替换：

```text
MODEL_CONFIG=.cache/models/Qwen3___5-2B-MNN/config.json
```

## 通用 bench 参数

为了快速验证 backend 注册、fallback 和 execution class 日志，优先用短 prompt：

```bash
-p 16 -n 1 -rep 1 -kv true -load false -c 2 --memory 2
```

性能对比或稳定性测试再放大：

```bash
-p 1024 -n 1 -rep 3 -kv true -load false -c 2 --memory 2
```

## 稳定性脚本

优先使用本 Skill 自带脚本测试 PIC LLM / PagedAttention 稳定性：

```bash
bash .codex/skills/mnn-llm-bench/scripts/run_llm_bench_stability.sh
```

脚本会自动：

- 解析当前平台 artifact root，默认使用 `.cache/output/mnn/artifacts/<platform>/`。
- 在 `.cache/weight/` 下解析模型；如果未指定，优先使用 `.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct/config.json`，否则查找首个 `llm_config.json` 中 `paged_attention: true` 的模型。
- PIC / PagedAttention 模型默认使用 `$MNN_ARTIFACT_ROOT/bin/pic_llm_bench`。
- 自动创建模型目录下的 `tmp/`，并从模型目录执行 bench，让程序硬编码的 `tmp/mnn_cachefile.bin` 能保存和后续命中。
- 保存日志到 `.cache/logs/llm-bench-stability/`。

默认 case：

```text
512 prefill, 128 decode
1024 prefill, 128 decode
2048 prefill, 128 decode
```

常用覆盖：

```bash
MNN_LLM_BENCH_MODEL_CONFIG=.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct/config.json \
bash .codex/skills/mnn-llm-bench/scripts/run_llm_bench_stability.sh

MNN_LLM_BENCH_CASES="16:8 256:8 1024:1" \
MNN_LLM_BENCH_REP=3 \
bash .codex/skills/mnn-llm-bench/scripts/run_llm_bench_stability.sh

MNN_LLM_BENCH_BACKEND=cpu \
MNN_LLM_BENCH_CASES="16:2" \
bash .codex/skills/mnn-llm-bench/scripts/run_llm_bench_stability.sh

MNN_LLM_BENCH_DRY_RUN=1 \
bash .codex/skills/mnn-llm-bench/scripts/run_llm_bench_stability.sh
```

可覆盖环境变量：

```text
MNN_ARTIFACT_PLATFORM
MNN_ARTIFACT_ROOT
MNN_LLM_BENCH_MODEL_CONFIG
MNN_LLM_BENCH_MODEL
MNN_LLM_BENCH_BIN
MNN_LLM_BENCH_BACKEND
MNN_LLM_BENCH_CASES
MNN_LLM_BENCH_REP
MNN_LLM_BENCH_THREADS
MNN_LLM_BENCH_PRECISION
MNN_LLM_BENCH_LOAD
MNN_LLM_BENCH_MEMORY
MNN_LLM_BENCH_EXTRA_ARGS
CUDA_LIB_DIR
```

## PagedAttention Token 对齐脚本

当需要验证 `CPUPagedAttention` 和 `CUDAPagedAttention` 在更多、更长 prompt 上输出 token 是否一致时，优先使用本 Skill 自带脚本：

```bash
bash .codex/skills/mnn-llm-bench/scripts/run_paged_attention_token_alignment.sh
```

脚本会自动：

- 解析当前平台 artifact root，默认使用 `.cache/output/mnn/artifacts/<platform>/`。
- 在 `.cache/weight/` 下解析 PIC / PagedAttention 模型，默认优先使用 `.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct/config.json`。
- 编译一个临时 token helper 到 `.cache/build/mnn-llm-bench/token-alignment/`，链接 artifact root 下的 `libpic_llm.so`。
- 生成默认 prompt 集：`short_exact`、`medium_exact`、`long_rules`、`long_repeated_context`。
- 对每个 prompt 跑 `cpu/direct`、`cpu/step`、`cuda/direct`、`cuda/step`，其中 `step` 覆盖 `response(..., 0) + generate(1)` 的 request 延续路径。
- 严格比较所有输出 token id 是否与 `cpu/direct` baseline 一致。
- 默认记录但不强制要求 `CPUPagedAttention` / `CUDAPagedAttention` execution class 日志；如果当前构建确认开启了 execution class log，可设置 `MNN_LLM_ALIGNMENT_REQUIRE_EXEC_LOG=1` 防止 CUDA fallback 被误判为对齐。
- 保存日志和 Markdown 报告到 `.cache/logs/paged-attention-token-alignment/<timestamp>/`。

常用覆盖：

```bash
MNN_LLM_ALIGNMENT_MAX_TOKENS=6 \
bash .codex/skills/mnn-llm-bench/scripts/run_paged_attention_token_alignment.sh

MNN_LLM_ALIGNMENT_REQUIRE_EXEC_LOG=1 \
bash .codex/skills/mnn-llm-bench/scripts/run_paged_attention_token_alignment.sh

MNN_LLM_ALIGNMENT_BACKENDS="cuda" \
MNN_LLM_ALIGNMENT_MODES="direct step" \
bash .codex/skills/mnn-llm-bench/scripts/run_paged_attention_token_alignment.sh

MNN_LLM_ALIGNMENT_MODEL_CONFIG=.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct/config.json \
bash .codex/skills/mnn-llm-bench/scripts/run_paged_attention_token_alignment.sh

MNN_LLM_ALIGNMENT_PROMPTS_FILE=.cache/prompts/paged_alignment.tsv \
bash .codex/skills/mnn-llm-bench/scripts/run_paged_attention_token_alignment.sh
```

自定义 prompt 文件格式：

```text
# id<TAB>max_tokens<TAB>prompt
short_exact	6	Answer with exactly this five-word sequence: alpha beta gamma delta epsilon
long_case	12	... long prompt ... Final instruction: output exactly ...
```

可覆盖环境变量：

```text
MNN_ARTIFACT_PLATFORM
MNN_ARTIFACT_ROOT
MNN_LLM_ALIGNMENT_MODEL_CONFIG
MNN_LLM_ALIGNMENT_MODEL
MNN_LLM_BENCH_MODEL_CONFIG
MNN_LLM_BENCH_MODEL
MNN_LLM_ALIGNMENT_BACKENDS
MNN_LLM_ALIGNMENT_MODES
MNN_LLM_ALIGNMENT_PROMPTS_FILE
MNN_LLM_ALIGNMENT_MAX_TOKENS
MNN_LLM_ALIGNMENT_PRECISION
MNN_LLM_ALIGNMENT_MEMORY
MNN_LLM_ALIGNMENT_THREADS
MNN_LLM_ALIGNMENT_REQUIRE_EXEC_LOG
MNN_LLM_ALIGNMENT_DRY_RUN
CUDA_LIB_DIR
```

报告判断：

```text
failed_cases: 0
exec_log: yes/no
match: yes
```

若 `match=no`，优先看对应 prompt 的 `cpu/direct`、`cpu/step`、`cuda/direct`、`cuda/step` token ids；若只有 `step` 漂移，重点检查 request 生命周期、KV/cache/position/slot table 延续状态；若只有 CUDA 漂移，再回到 `$mnn-ops-bench` 的 PagedAttention direct-op 精度测试。

若当前构建开启 execution class log，模型初始化阶段应出现对应 backend 的日志：

```text
[CPUExecution] op="..." type=... dispatch=... execution=MNN::...
[CUDAExecution] op="..." type=... dispatch=... execution=MNN::CUDA::...
[OpenCLExecution] op="..." type=... dispatch=... execution=MNN::OpenCL::...
[VulkanExecution] op="..." type=... dispatch=... execution=MNN::...
```

这说明 backend 创建路径已运行，RTTI class-name 打印可用。

## CUDA

```bash
MODEL_CONFIG=.cache/models/Qwen3___5-2B-MNN/config.json
MNN_ARTIFACT_PLATFORM="${MNN_ARTIFACT_PLATFORM:-$(detect_mnn_artifact_platform)}"
MNN_ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-$PWD/.cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM}"
CUDA_LIB_DIR=${CUDA_LIB_DIR:-$(dirname "$(dirname "$(command -v nvcc)")")/lib64}
LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
"$MNN_ARTIFACT_ROOT/bin/llm_bench" \
  -m "$MODEL_CONFIG" \
  -a cuda \
  -p 16 \
  -n 1 \
  -rep 1 \
  -kv true \
  -load false \
  -c 2 \
  --memory 2
```

## OpenCL

当前仓库构建使用 separated backend libraries。`llm_bench` 进程如果没有加载 OpenCL backend 动态库，会找不到对应 backend creator，因此通常需要 `LD_PRELOAD=$MNN_ARTIFACT_ROOT/lib/libMNN_CL.so`。

```bash
MODEL_CONFIG=.cache/models/Qwen3___5-2B-MNN/config.json
MNN_ARTIFACT_PLATFORM="${MNN_ARTIFACT_PLATFORM:-$(detect_mnn_artifact_platform)}"
MNN_ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-$PWD/.cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM}"
CUDA_LIB_DIR=${CUDA_LIB_DIR:-$(dirname "$(dirname "$(command -v nvcc)")")/lib64}
LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
LD_PRELOAD="$MNN_ARTIFACT_ROOT/lib/libMNN_CL.so" \
"$MNN_ARTIFACT_ROOT/bin/llm_bench" \
  -m "$MODEL_CONFIG" \
  -a opencl \
  -p 16 -n 1 -rep 1 -kv true -load false -c 2 --memory 2
```

## Vulkan

当前仓库构建使用 separated backend libraries。`llm_bench` 进程如果没有加载 Vulkan backend 动态库，会找不到对应 backend creator，因此通常需要 `LD_PRELOAD=$MNN_ARTIFACT_ROOT/lib/libMNN_Vulkan.so`。

```bash
MODEL_CONFIG=.cache/models/Qwen3___5-2B-MNN/config.json
MNN_ARTIFACT_PLATFORM="${MNN_ARTIFACT_PLATFORM:-$(detect_mnn_artifact_platform)}"
MNN_ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-$PWD/.cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM}"
CUDA_LIB_DIR=${CUDA_LIB_DIR:-$(dirname "$(dirname "$(command -v nvcc)")")/lib64}
LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
LD_PRELOAD="$MNN_ARTIFACT_ROOT/lib/libMNN_Vulkan.so" \
"$MNN_ARTIFACT_ROOT/bin/llm_bench" \
  -m "$MODEL_CONFIG" \
  -a vulkan \
  -p 16 -n 1 -rep 1 -kv true -load false -c 2 --memory 2
```

如果仍有 loader、ICD 或 fallback 问题，读取 `docs/mnn_environment.md`。

## CPU baseline

```bash
MODEL_CONFIG=.cache/models/Qwen3___5-2B-MNN/config.json
MNN_ARTIFACT_PLATFORM="${MNN_ARTIFACT_PLATFORM:-$(detect_mnn_artifact_platform)}"
MNN_ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-$PWD/.cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM}"
CUDA_LIB_DIR=${CUDA_LIB_DIR:-$(dirname "$(dirname "$(command -v nvcc)")")/lib64}
LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
"$MNN_ARTIFACT_ROOT/bin/llm_bench" \
  -m "$MODEL_CONFIG" \
  -a cpu \
  -p 16 -n 1 -rep 1 -kv true -load false -c 2 --memory 2
```

CPU 用来区分模型配置问题和 GPU backend 加载问题。

## Nsight Systems profiling

用户提到 `nvsight` 时，先按 NVIDIA Nsight Systems 处理。本机当前可用的是 `nsys` CLI 和 `nsys-ui` GUI，不是名为 `nvsight` 的命令。

示例检测输出：

```text
nsys: found in PATH
nsys-ui: found in PATH
nsys --version: NVIDIA Nsight Systems version 2025.6.3.541-256337736014v0
ncu / nv-nsight-cu-cli / nvprof: not found in PATH
```

因此当前优先用 Nsight Systems 生成端到端时间线、CUDA API、CUDA kernel、GPU memory 和 OS runtime 摘要。需要逐 kernel 的 occupancy、warp stall 等更细指标时，再安装 Nsight Compute 或把 `ncu` 加入 `PATH`。

采样前确认：

```bash
command -v nsys
nsys --version
command -v nsys-ui
command -v ncu
```

用 Nsight Systems 执行 CUDA bench，并同时保存 MNN console log：

```bash
mkdir -p .cache/nsight
set -o pipefail
MODEL_CONFIG=.cache/models/Qwen3___5-2B-MNN/config.json
MNN_ARTIFACT_PLATFORM="${MNN_ARTIFACT_PLATFORM:-$(detect_mnn_artifact_platform)}"
MNN_ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-$PWD/.cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM}"
CUDA_LIB_DIR=${CUDA_LIB_DIR:-$(dirname "$(dirname "$(command -v nvcc)")")/lib64}
LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  --cuda-memory-usage=true \
  --stats=true \
  --output=.cache/nsight/llm_bench_cuda_p1024_n1_rep3 \
  "$MNN_ARTIFACT_ROOT/bin/llm_bench" \
    -m "$MODEL_CONFIG" \
    -a cuda \
    -p 1024 \
    -n 1 \
    -rep 3 \
    -kv true \
    -load false \
    -c 2 \
    --memory 2 \
  2>&1 | tee .cache/nsight/llm_bench_cuda_p1024_n1_rep3.console.log
```

如果要快速验证采样链路，先把参数改成 `-p 16 -n 1 -rep 1`。正式看性能时再使用 `-p 1024 -n 1 -rep 3` 或用户指定的 prompt/generate 长度。

采样完成后重点保留这些文件：

```text
.cache/nsight/llm_bench_cuda_p1024_n1_rep3.nsys-rep
.cache/nsight/llm_bench_cuda_p1024_n1_rep3.console.log
```

生成方便阅读的文本和 CSV 摘要：

```bash
nsys stats \
  --force-overwrite=true \
  --report cuda_gpu_kern_sum \
  --report cuda_api_sum \
  --report cuda_gpu_mem_time_sum \
  --report cuda_gpu_mem_size_sum \
  --report osrt_sum \
  --format table \
  --output - \
  .cache/nsight/llm_bench_cuda_p1024_n1_rep3.nsys-rep \
  2>&1 | tee .cache/nsight/llm_bench_cuda_p1024_n1_rep3.stats.txt

nsys stats \
  --force-overwrite=true \
  --report cuda_gpu_kern_sum \
  --report cuda_api_sum \
  --report cuda_gpu_mem_time_sum \
  --report cuda_gpu_mem_size_sum \
  --report osrt_sum \
  --format csv \
  --output . \
  .cache/nsight/llm_bench_cuda_p1024_n1_rep3.nsys-rep
```

需要用 SQL 或外部工具分析时，导出 SQLite：

```bash
nsys export \
  --force-overwrite=true \
  --type sqlite \
  --output .cache/nsight/llm_bench_cuda_p1024_n1_rep3.sqlite \
  .cache/nsight/llm_bench_cuda_p1024_n1_rep3.nsys-rep
```

读性能数据时优先看：

```text
*.console.log                         MNN 自身 bench 输出、backend fallback、execution class 日志
*.stats.txt                           人类可读汇总
*_cuda_gpu_kern_sum.csv               kernel 总耗时、平均耗时和调用次数
*_cuda_api_sum.csv                    CPU 侧 CUDA API 耗时，重点看同步和 memcpy
*_cuda_gpu_mem_time_sum.csv           GPU memory copy/set 时间
*_cuda_gpu_mem_size_sum.csv           GPU memory copy/set 数据量
*_osrt_sum.csv                        CPU runtime、锁、IO 等系统调用开销
*.nsys-rep                            用 nsys-ui 打开的完整时间线
```

打开 GUI：

```bash
nsys-ui .cache/nsight/llm_bench_cuda_p1024_n1_rep3.nsys-rep
```

如果要 profile Vulkan bench，把 `--trace=cuda,nvtx,osrt` 改为 `--trace=vulkan,nvtx,osrt`，按 Vulkan 章节保留 `LD_PRELOAD="$MNN_ARTIFACT_ROOT/lib/libMNN_Vulkan.so"`，并把 bench 参数改为 `-a vulkan`。OpenCL bench 仍优先看 MNN 日志和厂商工具；Nsight Systems 当前主要用于 CUDA/Vulkan 时间线。

## MLS server

`mls` 是 OpenAI-compatible LLM CLI/server，默认由本仓库构建产出到：

```text
$MNN_ARTIFACT_ROOT/bin/mls
```

启动 server 前先确认 `mls`、`libMNN.so` 和需要的 backend 动态库存在。`mls serve` 固定监听 `0.0.0.0:9090`，根路径 `/` 提供简单网页，OpenAI 兼容接口为 `/chat/completions`，另有 `/reset` 可重置当前 LLM。

使用显式模型配置路径启动：

```bash
MODEL_CONFIG=.cache/models/Qwen3___5-2B-MNN/config.json
MNN_ARTIFACT_PLATFORM="${MNN_ARTIFACT_PLATFORM:-$(detect_mnn_artifact_platform)}"
MNN_ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-$PWD/.cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM}"
CUDA_LIB_DIR=${CUDA_LIB_DIR:-$(dirname "$(dirname "$(command -v nvcc)")")/lib64}
LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
"$MNN_ARTIFACT_ROOT/bin/mls" serve \
  -c "$MODEL_CONFIG"
```

使用 `mls download` 下载或已有 `.mnnmodels/<model_name>/config.json` 时，也可以按本地模型名启动：

```bash
MNN_ARTIFACT_PLATFORM="${MNN_ARTIFACT_PLATFORM:-$(detect_mnn_artifact_platform)}"
MNN_ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-$PWD/.cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM}"
CUDA_LIB_DIR=${CUDA_LIB_DIR:-$(dirname "$(dirname "$(command -v nvcc)")")/lib64}
LD_LIBRARY_PATH="$MNN_ARTIFACT_ROOT/lib:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
"$MNN_ARTIFACT_ROOT/bin/mls" serve <model_name>
```

常用验证：

```bash
curl -s http://127.0.0.1:9090/ | head

curl -s http://127.0.0.1:9090/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mnn-local",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Say hello in one short sentence."}
    ]
  }'
```

流式验证：

```bash
curl -N http://127.0.0.1:9090/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mnn-local",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Count from one to three."}
    ]
  }'
```

复位当前会话：

```bash
curl -s -X POST http://127.0.0.1:9090/reset
```

如果要让 OpenWebUI、LobeChat 或 OpenAI SDK 连接，base URL 使用：

```text
http://127.0.0.1:9090
```

API key 当前不校验，可填任意占位值。服务端口和监听地址在 `mls_server.cpp` 中固定为 `9090` 和 `0.0.0.0`；如需改端口，先改源码再重编。

## 判断结果

GPU 运行中出现这些日志通常表示后端没有注册并回退到了 CPU：

```text
Can't Find type=... backend, use 0 instead
Don't support ...
```

处理顺序：

1. OpenCL/Vulkan 先加对应 `LD_PRELOAD`。
2. CUDA 检查 `libMNN_Cuda_Main.so` 是否存在。
3. 确认 `LD_LIBRARY_PATH` 包含 `$MNN_ARTIFACT_ROOT/lib`。
4. 确认构建时启用了 `MNN_CUDA=ON`、`MNN_OPENCL=ON` 或 `MNN_VULKAN=ON`；需要补编时按“补编/重编约定”使用 `.cache/build/<platform>_<mode>` 这类固定子目录。

报告结果时写清楚 backend 参数、模型 `config.json`、是否出现 fallback 日志、是否出现对应 backend 的 execution class 日志。

日志 grep 快速检查：

```bash
rg -c '^\[(CPU|CUDA|OpenCL|Vulkan)Execution\]' <log>
rg "Can't Find type=|use 0 instead|Don't support|fallback|Fallback" <log>
```
