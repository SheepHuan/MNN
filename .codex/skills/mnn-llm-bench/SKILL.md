---
name: mnn-llm-bench
description: 当用户要求运行 MNN llm_bench、llm_demo、pic_llm_bench 或 pic_llm_demo，测试 CUDA/OpenCL/Vulkan/CPU LLM 推理，对比普通 LLM 与 PIC/PagedAttention 输出，确认 GPU 后端注册、检查 CUDA Execution class 日志、诊断后端回退，或围绕 LLM bench 采集 DF power / Jetson tegrastats 功耗时使用。
metadata:
  short-description: 运行和诊断 MNN llm_bench
---

# MNN LLM Bench

本 Skill 由本 MNN 仓库维护，用于使用本仓库 `.cache/output/mnn/artifacts/<platform>/bin/llm_bench`、`llm_demo`、`pic_llm_bench` 或 `pic_llm_demo` 测试 MNN LLM 模型，也用于启动和验证同一 artifact root 下的 `mls` 或独立 `pic_server` HTTP server。PIC LLM / PagedAttention 导出模型优先使用 `pic_llm_bench`、`pic_llm_demo` 或 `pic_server`，不要混用链接 `libllm.so` 的 `llm_bench` / `llm_demo` 做最终稳定性或输出判断。

功耗采集是本 Skill 的可选 bench 能力：优先使用 DF power API 做跨设备外部功耗采样；需要 Jetson rail 细分时，使用本 Skill 内置的 `tegrastats` 采集脚本。不要再新增或使用独立的 Jetson power skill。

## 工作流

1. 自动解析 `MNN_ARTIFACT_ROOT`，确认 `$MNN_ARTIFACT_ROOT/bin/llm_bench`、需要时的 `$MNN_ARTIFACT_ROOT/bin/mls` / `pic_server` 和目标 backend 动态库存在。
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
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
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
$MNN_ARTIFACT_ROOT/bin/pic_server
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

## 普通 LLM 与 PIC/PagedAttention 输出对比

需要比较普通 MNN LLM 和 PIC/PagedAttention LLM 的 prompt 输出精度时，必须比较两个独立导出的模型目录，不要修改 `llm_config.json` 的 `paged_attention` 字段伪装另一种模式：

```text
普通 MNN LLM:       .cache/mnn-llm-export/<model>/config.json  用 llm_demo 跑
PIC/PagedAttention: .cache/weight/<model>/config.json           用 pic_llm_demo 跑
```

对比要求：

- 普通导出目录来自 `.cache/mnn-llm-export/`，使用 `$MNN_ARTIFACT_ROOT/bin/llm_demo`。
- PIC / PagedAttention 导出目录来自 `.cache/weight/`，使用 `$MNN_ARTIFACT_ROOT/bin/pic_llm_demo`。
- 不要把 `.cache/weight/` 下的 PIC 模型复制成 normal view 来跑普通 `llm_demo`；这不能代表普通导出。
- 两边使用同一个 prompt 文件、相同 backend / precision / memory / sampler 配置，建议 greedy：`sampler_type=greedy`、`temperature=0.0`、`top_k=1`、`top_p=1.0`。
- 从各自模型目录执行 demo，确保程序使用相对路径访问 `llm.mnn`、weight、tokenizer 和 `tmp/mnn_cachefile.bin`。
- 日志至少保存生成文本、退出码、backend fallback 相关行，以及 `[CUDAExecution]` / `CUDAPagedAttention` / `CPUPagedAttention` 等 execution class 线索。

Jetson 上已有 Llama 示例通常按下面配对：

```text
normal_config=.cache/mnn-llm-export/AI-ModelScope__Llama-3___2-3B-Instruct/config.json
paged_config=.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct/config.json
```

Jetson 远端复现模板：

```bash
cd /home/jetson/code/kvshare-edge/impl/MNN
REPO=$PWD
ART=$REPO/.cache/output/mnn/artifacts/jetson_cross_cuda
NORMAL=$REPO/.cache/mnn-llm-export/AI-ModelScope__Llama-3___2-3B-Instruct
PAGED=$REPO/.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct
PROMPT=$REPO/.cache/prompts/normal_vs_paged_prompt.txt
LOG=$REPO/.cache/logs/llm-output-compare
mkdir -p "$NORMAL/tmp" "$PAGED/tmp" "$LOG"

cd "$NORMAL"
LD_LIBRARY_PATH="$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" \
  "$ART/bin/llm_demo" config_cuda_greedy.json "$PROMPT" \
  > "$LOG/normal_llm_demo_cuda_greedy.log" 2>&1

cd "$PAGED"
LD_LIBRARY_PATH="$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" \
  "$ART/bin/pic_llm_demo" config_cuda_greedy.json "$PROMPT" \
  > "$LOG/paged_pic_llm_demo_cuda_greedy.log" 2>&1
```

## 独立 PIC Server

MNN PIC server 由 `transformers/pic_llm/engine/app/pic_server.cpp` 维护，产物名是 `pic_server`，不依赖 `mls`。首个对齐 `hf_pic_runtime` 的能力是文档 prefill：

```text
POST /v1/prefill/text
POST /v1/kv/pic_caches
POST /v1/chat/completions
POST /chat/completions
```

请求体必须发送真实内联文本，不接受路径：

```json
{
  "id": "doc-1",
  "type": "text",
  "content": "full document text",
  "force": false
}
```

Jetson 远端启动和 smoke 模板：

```bash
cd /home/jetson/code/kvshare-edge/impl/MNN
REPO=$PWD
ART=$REPO/.cache/output/mnn/artifacts/jetson_cross_cuda
MODEL=$REPO/.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct
CONFIG=$MODEL/config_cuda_greedy.json
KV_DIR=$REPO/.cache/kvshare/pic_server_smoke
LD_LIBRARY_PATH="$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" \
  "$ART/bin/pic_server" --config "$CONFIG" --host 127.0.0.1 --port 18091 \
  --kv-cache-dir "$KV_DIR" --model llama-pic
```

另一个 shell 发请求：

```bash
curl -s http://127.0.0.1:18091/healthz
curl -s -X POST http://127.0.0.1:18091/v1/prefill/text \
  -H 'Content-Type: application/json' \
  -d '{"id":"smoke_doc","type":"text","content":"Jetson PIC cache smoke text.","force":true}'
curl -s -X POST http://127.0.0.1:18091/v1/kv/pic_caches \
  -H 'Content-Type: application/json' \
  -d '{"id":"smoke_pic","text_cache_refs":[{"id":"smoke_doc"}],"selection_algorithm":"full-reuse"}'
curl -s -X POST http://127.0.0.1:18091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-pic","messages":[{"role":"user","content":"{{pic_cache}}\nQuestion: what text was cached?"}],"max_tokens":8,"temperature":0,"pic_cache":{"id":"smoke_pic","text_cache_refs":[{"id":"smoke_doc"}],"selection_algorithm":"full-reuse"}}'
```

`/v1/prefill/text` 会把每层 KV 分 key/value 导出到 `KV_DIR/objects/<backend>/<cache_name>/layers/` 下的 `.k` / `.v` raw 文件，并为同层写 `.json` shape sidecar，最后写 `meta.json` / `tokens.json`。导出前 CPU/CUDA PagedAttention 会对 key cache 做 inverse RoPE，因此 metadata 应标注：

```text
format=kvshare-prefix-cache-meta-v1
kv_layout.layout=mnn_paged_attention_raw_v1
kv_layout.key_rope_state=canonical_no_rope
source.content_sha256=<sha256>
```

每层 shape sidecar 使用 `mnn-paged-attention-kv-shape-v1`，至少包含 `batch`、`kv_heads`、`head_dim`、`token_count`、`dtype_bytes`、`key_rope_state=canonical_no_rope`、`rope_pairing=half`、`rope_theta` 和 `rope_dim`；`meta.json` 的 `kv_layout.key_shape` / `value_shape` / RoPE 字段必须从 sidecar 的真实执行参数填充。

PIC chat smoke 重点检查：

- `/` 的 `endpoints` 包含 `/v1/kv/pic_caches` 与 `/v1/chat/completions`。
- `/v1/kv/pic_caches` 返回 `type=pic_cache`、非零 `token_count` 和 `text_caches`。
- `/v1/chat/completions` 返回 OpenAI 风格 `choices[0].message.content` 与 `usage`。
- `pic_cache.precision_recovery.execution_mode`：`full-reuse` 应为 `native-full-reuse`，`full-compute` 应为 `native-full-compute`，`epic` 应为 `native-epic-sparse-recompute`，`cacheblend` 应为 `native-cacheblend-sparse-recompute`，`kvshare` 应为 `native-kvshare-sparse-recompute`。
- `epic/cacheblend/kvshare` 应返回 `metadata.native_sparse_recompute_scope=python_prefill_layer_plan`；当 `pic_recompute_score_layer_idx > 0` 时，还应有 `metadata.pre_score_kv_source=full_prompt_reference` 和 `metadata.pre_score_compute_layers=<score_layer_idx>`，表示 score layer 前保持 token 正常计算语义。从 score layer 开始，选中的 PIC token 和 suffix/非复用 KV token 参与计算，其他 PIC 位置直接复用磁盘 KV。
- `cacheblend` / `delta-v` 的评分来自 score layer full-vs-cached value mean-abs delta；`kvshare` / `delta-a` 当前 MNN C++ 使用 K/V delta influence proxy，metadata 会标注它与 HF Python autograd `attention_output` gradient influence 的差异。

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
MNN_LLM_BENCH_POWER
MNN_POWER_DEVICE
MNN_POWER_API_URL
MNN_POWER_SAMPLE_RATE_HZ
MNN_POWER_VOLTAGE_MV
CUDA_LIB_DIR
```

## Jetson CUDA 多模型 Power Suite

Jetson 上测试普通 MNN LLM CUDA 在不同 prefill 长度下的 DF power 时，使用本 Skill 的多模型 suite：

```bash
bash .codex/skills/mnn-llm-bench/scripts/run_jetson_cuda_power_suite.sh
```

固定测试模型和普通 MNN 导出目录：

```text
AI-ModelScope/Llama-3.2-1B-Instruct -> .cache/mnn-llm-export/AI-ModelScope__Llama-3.2-1B-Instruct
AI-ModelScope/Llama-3.2-3B-Instruct -> .cache/mnn-llm-export/AI-ModelScope__Llama-3.2-3B-Instruct
LLM-Research/Meta-Llama-3-8B-Instruct -> .cache/mnn-llm-export/LLM-Research__Meta-Llama-3-8B-Instruct
ZhipuAI/glm-edge-4b-chat            -> .cache/mnn-llm-export/ZhipuAI__glm-edge-4b-chat
Qwen/Qwen2.5-7B-Instruct            -> .cache/mnn-llm-export/Qwen__Qwen2.5-7B-Instruct
```

suite 默认：

```text
MNN_JETSON_REMOTE=jetson@192.168.101.192
MNN_JETSON_REPO=/home/jetson/code/kvshare-edge/impl/MNN
MNN_JETSON_ARTIFACT_ROOT=$MNN_JETSON_REPO/.cache/output/mnn/artifacts/jetson
MNN_JETSON_MODEL_ROOT=$MNN_JETSON_REPO/.cache/mnn-llm-export
MNN_LLM_EXPORT_ROOT=$PWD/.cache/mnn-llm-export
MODELSCOPE_CACHE_ROOT=$HOME/.cache/modelscope/hub/models
MNN_LLM_BENCH_CASES="256:1 512:1 1024:1 2048:1 4096:1"
MNN_POWER_API_URL=http://192.168.101.14:8000
MNN_POWER_SERIAL=1A5D43
MNN_POWER_WARMUP_SEC=10
MNN_POWER_COOLDOWN_SEC=10
```

脚本会先检查默认 ModelScope cache `~/.cache/modelscope/hub/models` 或 Hugging Face cache `~/.cache/huggingface/hub` 中对应模型 source 是否存在，再检查本地 `.cache/mnn-llm-export/` 下 4bit 普通 MNN 导出是否完整。缺失时只打印精确导出命令并退出，不自动导出。检查通过后，脚本用 `rsync -a --delete` 增量同步 artifact root 和全部模型目录到 Jetson，再逐个运行：

```bash
LD_LIBRARY_PATH="$MNN_JETSON_ARTIFACT_ROOT/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" \
  "$MNN_JETSON_ARTIFACT_ROOT/bin/llm_bench" \
  -m config.json -a cuda -p <prefill> -n 1 -rep 1 -kv true -load true -c 2 -t 4 --memory 2
```

每个 case 的 power 窗口是：启动 DF capture，等待 10 秒，执行远端 bench，bench 结束后再等待 10 秒，然后停止 capture 并保存 CSV。产物保存在：

```text
.cache/logs/llm-bench-power/<timestamp>/manifest.tsv
.cache/logs/llm-bench-power/<timestamp>/<model>_p<prompt>_n1.log
.cache/logs/llm-bench-power/<timestamp>/<model>_p<prompt>_n1_power.csv
```

只看命令不触发 rsync、SSH 或 power API：

```bash
MNN_LLM_BENCH_DRY_RUN=1 \
bash .codex/skills/mnn-llm-bench/scripts/run_jetson_cuda_power_suite.sh
```

## DF Power 可选采集

当需要测试 `llm_bench` / `pic_llm_bench` 在不同上下文长度下的功耗时，使用 DF power API 作为 bench 可选项。默认 API 和设备序列号固定记录如下：

```text
MNN_POWER_API_URL=http://192.168.101.14:8000
jetson:        1A5D43
orangepi5plus: C071EB951330
oneplus13t:    F96FBDBA05B0
```

测量顺序必须是：

1. 先把本次要测的 MNN 产物、模型和脚本推送到目标设备，并确认远端命令可执行。
2. 再启动 DF power capture 计时。
3. 立刻执行目标 `llm_bench` / `pic_llm_bench` 推理命令。
4. 推理命令结束后立刻停止 capture 并下载 CSV。

不要把产物 push、模型同步、首次环境准备、手动等待时间混进功耗窗口。

通用命令包装器：

```bash
MNN_POWER_DEVICE=orangepi5plus \
MNN_POWER_OUTPUT_CSV=.cache/logs/llm-bench-power/orangepi5plus_p1024_n128.csv \
.codex/skills/mnn-llm-bench/scripts/df_power_capture.sh -- \
  bash -lc 'cd /path/to/model && LD_LIBRARY_PATH=/path/to/artifacts/lib:$LD_LIBRARY_PATH /path/to/artifacts/bin/pic_llm_bench -m config.json -a opencl -p 1024 -n 128 -rep 1 -kv true -load true -c 2 --memory 2'
```

稳定性脚本可直接为每个 `prompt:decode` case 采集一份 CSV：

```bash
MNN_LLM_BENCH_POWER=1 \
MNN_POWER_DEVICE=orangepi5plus \
MNN_LLM_BENCH_BACKEND=opencl \
MNN_LLM_BENCH_CASES="512:128 1024:128 2048:128" \
bash .codex/skills/mnn-llm-bench/scripts/run_llm_bench_stability.sh
```

输出位置：

```text
.cache/logs/llm-bench-stability/<timestamp>_<model>_<backend>_p<prompt>_n<decode>.log
.cache/logs/llm-bench-stability/<timestamp>_<model>_<backend>_p<prompt>_n<decode>_power.csv
```

设备名也可直接换成 `jetson` 或 `oneplus13t`；如需临时指定未登记设备，使用 `MNN_POWER_SERIAL=<df-serial>`。

### 渲染 DF Power CSV

DF power CSV 可以直接渲染成 PNG，图像 x 轴是时间秒，y 轴是功耗 W：

```bash
.codex/skills/mnn-llm-bench/scripts/render_power_csv.sh \
  .cache/logs/llm-bench-power/<timestamp>/<case>_power.csv
```

默认在 CSV 同目录生成同名 `.png`。常用批量和竖排总览图：

```bash
# 每个 CSV 各生成一张 PNG
.codex/skills/mnn-llm-bench/scripts/render_power_csv.sh \
  --out-dir .cache/logs/llm-bench-power/<timestamp>/plots \
  .cache/logs/llm-bench-power/<timestamp>/*_power.csv

# 多个 CSV 渲染到一张 PNG；每个 CSV 一行子图，竖直排布
.codex/skills/mnn-llm-bench/scripts/render_power_csv.sh \
  --overlay \
  -o .cache/logs/llm-bench-power/<timestamp>/power_stacked.png \
  .cache/logs/llm-bench-power/<timestamp>/*_power.csv

# 裁掉采样窗口前后各 10 秒 warmup/cooldown 再画
.codex/skills/mnn-llm-bench/scripts/render_power_csv.sh \
  --trim-sec 10 \
  .cache/logs/llm-bench-power/<timestamp>/*_power.csv
```

脚本使用当前 Python 环境的 `matplotlib`，缺依赖时先在当前 bench 环境安装或切到有 `matplotlib` 的 Python。

### 计算 Bench 窗口能耗

DF power CSV 采集窗口通常包含前后 warmup/cooldown。计算 `llm_bench` 真实运行时能耗时，先裁掉前后没有运行 bench 的采样段，再对功率积分；单次迭代能耗为：

```text
energy_per_iter_j = bench_energy_j / repeat
```

从 suite 的 `manifest.tsv` 计算：

```bash
.codex/skills/mnn-llm-bench/scripts/summarize_power_energy.sh \
  --trim-sec 10 \
  -o .cache/logs/llm-bench-power/<timestamp>/energy_summary.tsv \
  .cache/logs/llm-bench-power/<timestamp>/manifest.tsv
```

也可以直接传目录或 CSV：

```bash
.codex/skills/mnn-llm-bench/scripts/summarize_power_energy.sh \
  --trim-sec 10 \
  .cache/logs/llm-bench-power/<timestamp>/
```

## Jetson tegrastats 可选采集

需要 Jetson CPU/GPU/DDR rail 细分功耗时，使用本 Skill 内置采集脚本：

```text
.codex/skills/mnn-llm-bench/scripts/jetson_power_collect.sh
```

采集字段来自 `tegrastats` 当前功耗：

```text
CPU ...mW/...mW   -> cpu_w
GPU ...mW/...mW   -> gpu_w
VDDRQ ...mW/...mW -> ddr_w
```

CSV 列：

```text
timestamp_unix_ms,elapsed_ms,cpu_w,gpu_w,ddr_w
```

围绕远端 bench 的基本流程：

```bash
JETSON=jetson@192.168.101.192
RUN_ID=$(date +%Y%m%d_%H%M%S)
REMOTE_DIR=/tmp/mnn_llm_bench_power/$RUN_ID
LOCAL_DIR=$PWD/.cache/logs/llm-bench-tegrastats/$RUN_ID
INTERVAL_MS=50
mkdir -p "$LOCAL_DIR"

ssh "$JETSON" "mkdir -p '$REMOTE_DIR'"
scp .codex/skills/mnn-llm-bench/scripts/jetson_power_collect.sh \
  "$JETSON:$REMOTE_DIR/jetson_power_collect.sh"
ssh "$JETSON" "chmod +x '$REMOTE_DIR/jetson_power_collect.sh'"

REMOTE_CSV=$REMOTE_DIR/power.csv
REMOTE_LOG=$REMOTE_DIR/power.log
ssh "$JETSON" "nohup '$REMOTE_DIR/jetson_power_collect.sh' '$REMOTE_CSV' '$INTERVAL_MS' > '$REMOTE_LOG' 2>&1 & echo \$! > '$REMOTE_DIR/power.pid'"

# run llm_bench / pic_llm_bench workload here

ssh "$JETSON" "if [ -f '$REMOTE_DIR/power.pid' ]; then kill \$(cat '$REMOTE_DIR/power.pid') 2>/dev/null || true; fi; sleep 1"
scp "$JETSON:$REMOTE_CSV" "$LOCAL_DIR/power.csv"
scp "$JETSON:$REMOTE_LOG" "$LOCAL_DIR/power.log" 2>/dev/null || true
```

如果目标 Jetson 的 `tegrastats` 不输出 `CPU`、`GPU` 或 `VDDRQ` 独立 rail，对应列会为空；不要把 `SOC`、`CV` 或 `SYS5V` 强行当成 CPU/GPU/DDR。

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

Jetson 设备上也可能已经安装 Nsight Compute，例如：

```text
nsys: /usr/local/bin/nsys
nsys-ui: /usr/local/bin/nsys-ui
ncu: /opt/nvidia/nsight-compute/2022.2.1/ncu
nsys --version: NVIDIA Nsight Systems version 2023.2.4.44-33011852v0
```

优先用 Nsight Systems 生成端到端时间线、CUDA API、CUDA kernel、GPU memory 和 OS runtime 摘要；这能对比 normal LLM、PIC full-compute、full-reuse 的 kernel 序列和阶段耗时。需要逐 kernel 的 occupancy、warp stall、memory throughput 等更细指标时，再用 Nsight Compute `ncu` 对少量 kernel 或短 prompt 做二次采样。

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

### Jetson 四模式 Nsight 对比

当用户要比较“正常 LLM、普通 full-compute、PIC LLM full-compute、full-reuse”的计算图/时间线时，在 Jetson 上用同一套 artifact 和模型长度采集，输出统一放到 `.cache/nsight/<run_id>/`。建议先用 `max_tokens=1` 和固定 prompt/token 长度，避免 decode 循环把 prefill 差异淹没。

严格对比时必须记录每个响应的 `usage.prompt_tokens`。`normal_llm` 的 `-p <N>`、`pic_full_compute` / `pic_full_reuse` 的 `prelude + PIC + suffix`、以及 `server_full_compute` 的真实文本 prompt 长度应尽量一致；如果没有原始全文，只能把 `server_full_compute` 当作 no-PIC server 路径 smoke，不要直接和 PIC 594/2312-token 路径下结论。

四类模式约定：

```text
normal_llm              普通导出模型，.cache/mnn-llm-export/...，用 llm_bench
server_full_compute     pic_server 无 pic_cache 的普通 chat/full-compute 请求
pic_full_compute        pic_server 带 pic_cache，selection_algorithm=full-compute
pic_full_reuse          pic_server 带 pic_cache，selection_algorithm=full-reuse
```

先确认工具和路径：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  command -v nsys && nsys --version && command -v ncu || true && \
  ART=$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda && \
  test -x "$ART/bin/llm_bench" && test -x "$ART/bin/pic_server" && \
  test -f "$ART/lib/libMNN_Cuda_Main.so"'
```

采集 normal LLM：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  REPO=$PWD && ART=$REPO/.cache/output/mnn/artifacts/jetson_cross_cuda && \
  RUN_ID=${RUN_ID:-jetson_nsight_llm_pic_compare} && OUT=$REPO/.cache/nsight/$RUN_ID && \
  NORMAL=$REPO/.cache/mnn-llm-export/AI-ModelScope__Llama-3___2-3B-Instruct/config_cuda_greedy.json && \
  mkdir -p "$OUT" && \
  LD_LIBRARY_PATH="$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" \
  nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=process-tree \
    --cuda-memory-usage=true --stats=true \
    --output="$OUT/normal_llm_p2369_n1" \
    "$ART/bin/llm_bench" -m "$NORMAL" -a cuda -p 2369 -n 1 -rep 1 -kv true -load false \
    2>&1 | tee "$OUT/normal_llm_p2369_n1.console.log"'
```

采集 `pic_server` 三种请求时，用 `nsys profile` 包住 server 进程，再发送一次 HTTP 请求。`TEXT_META` 指向已经由 `/v1/prefill/text` 生成的 text cache `meta.json`；如果没有现成 cache，先启动一次普通 server 调 `/v1/prefill/text` 构建。

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && bash -s' <<'REMOTE'
set -euo pipefail
REPO=$PWD
ART=$REPO/.cache/output/mnn/artifacts/jetson_cross_cuda
CONFIG=$REPO/.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct/config_cuda_greedy.json
TEXT_META=${TEXT_META:-$REPO/.cache/kvshare/jetson_full_reuse_final/objects/mnn_cuda/doc_jetson-full-reuse-doc/meta.json}
RUN_ID=${RUN_ID:-jetson_nsight_llm_pic_compare}
OUT=$REPO/.cache/nsight/$RUN_ID
mkdir -p "$OUT"

profile_pic_server_mode() {
  local mode="$1"
  local port="$2"
  local out="$OUT/${mode}"
  local kv_dir="$REPO/.cache/kvshare/nsight_${mode}"
  rm -f "${out}.nsys-rep" "${out}.sqlite" "${out}.console.log" "${out}.response.json"
  mkdir -p "$kv_dir"
  fuser -k "${port}/tcp" >/dev/null 2>&1 || true

  LD_LIBRARY_PATH="$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" \
  nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=process-tree \
    --cuda-memory-usage=true --stats=true --output="$out" \
    "$ART/bin/pic_server" --config "$CONFIG" --host 127.0.0.1 --port "$port" \
      --kv-cache-dir "$kv_dir" --model llama-pic \
    > "${out}.console.log" 2>&1 &
  local nsys_pid=$!

  for _ in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  MODE="$mode" PORT="$port" TEXT_META="$TEXT_META" RESPONSE_PATH="${out}.response.json" python3 - <<'PY'
import http.client, json, os
mode = os.environ["MODE"]
port = int(os.environ["PORT"])
text_meta = os.environ["TEXT_META"]
response_path = os.environ["RESPONSE_PATH"]
messages = [
    {"role": "system", "content": "You are concise. Use cached context if present.\n{{pic_cache}}"},
    {"role": "user", "content": "Answer with exactly one token: KV"},
]
payload = {"model": "llama-pic", "messages": messages, "max_tokens": 1, "temperature": 0.0}
if mode == "pic_full_compute":
    payload["pic_cache"] = {"id": "nsight-pic-full-compute", "text_cache_refs": [{"meta_path": text_meta}], "selection_algorithm": "full-compute"}
elif mode == "pic_full_reuse":
    payload["pic_cache"] = {"id": "nsight-pic-full-reuse", "text_cache_refs": [{"meta_path": text_meta}], "selection_algorithm": "full-reuse"}
conn = http.client.HTTPConnection("127.0.0.1", port, timeout=240)
conn.request("POST", "/v1/chat/completions", body=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
resp = conn.getresponse()
raw = resp.read()
conn.close()
open(response_path, "wb").write(raw)
print(resp.status)
PY

  kill -INT "$nsys_pid" >/dev/null 2>&1 || true
  wait "$nsys_pid" || true
}

profile_pic_server_mode server_full_compute 18151
profile_pic_server_mode pic_full_compute 18152
profile_pic_server_mode pic_full_reuse 18153
REMOTE
```

生成四种模式的摘要：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  RUN_ID=${RUN_ID:-jetson_nsight_llm_pic_compare} && OUT=$PWD/.cache/nsight/$RUN_ID && \
  for rep in "$OUT"/*.nsys-rep; do \
    base=${rep%.nsys-rep}; \
    nsys stats --force-overwrite=true \
      --report cuda_gpu_kern_sum --report cuda_api_sum \
      --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum \
      --format table --output - "$rep" 2>&1 | tee "${base}.stats.txt"; \
    nsys export --force-overwrite=true --type sqlite --output "${base}.sqlite" "$rep" >/dev/null 2>&1 || true; \
  done'
```

对比时至少交付：

```text
*.nsys-rep          Nsight Systems GUI 时间线，可看 kernel 序列/并发/空洞
*.stats.txt         kernel、CUDA API、memcpy/memset 汇总
*.sqlite            可脚本化抽取 kernel 顺序和阶段
*.console.log       MNN backend / execution mode / fallback 日志
*.response.json     pic_server 返回的 precision_recovery.execution_mode 和 token usage
```

`server_full_compute` 的响应不应包含 `pic_cache`；`pic_full_compute` 应为 `precision_recovery.execution_mode=native-full-compute`；`pic_full_reuse` 应为 `native-full-reuse` 且 `recompute_token_count=0`。如果响应 metadata 不满足这些条件，该 profile 不能用于四模式对比。

需要逐 kernel 计算指标时，用 Nsight Compute 对短 prompt 或少量 kernel 采样，避免完整 LLM 运行过慢：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  REPO=$PWD && ART=$REPO/.cache/output/mnn/artifacts/jetson_cross_cuda && \
  NORMAL=$REPO/.cache/mnn-llm-export/AI-ModelScope__Llama-3___2-3B-Instruct/config_cuda_greedy.json && \
  OUT=$REPO/.cache/nsight/ncu_smoke && mkdir -p "$OUT" && \
  LD_LIBRARY_PATH="$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" \
  ncu --target-processes all --set default --launch-count 20 \
    --export "$OUT/normal_llm_p128_ncu" --force-overwrite \
    "$ART/bin/llm_bench" -m "$NORMAL" -a cuda -p 128 -n 1 -rep 1 -kv true -load false'
```

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
