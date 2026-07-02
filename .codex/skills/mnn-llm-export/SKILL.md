---
name: mnn-llm-export
description: 当用户要求把 ModelScope 或 Hugging Face/Transformers 格式 LLM 导出为普通 MNN LLM、PIC LLM PagedAttention 模型或 PrefixLLM/PrefixAttention 模型，并区分普通 transformers/llm 导出和 .cache/weight 特化导出路径时使用这个 Skill。
metadata:
  short-description: 导出普通 MNN、PIC 或 PrefixLLM 模型
---

# MNN LLM Export

本 Skill 用于在本 MNN 仓库中导出 LLM 模型。导出输入是 Hugging Face/Transformers 格式模型目录；普通 `transformers/llm` 导出默认保存到 `.cache/mnn-llm-export/<model-name>/`，PIC/Prefix 特化导出保存到 `.cache/weight/<model-name>/`。不要把导出结果写进源码目录或 `output/`。

## 导出链路选择

| 用途 | Exporter | 开关 | 默认输出 | 运行侧 |
| --- | --- | --- | --- | --- |
| 普通/正常 MNN LLM | `transformers/llm/export/llmexport.py` | `MNN_LLM_EXPORTER=mnn`，也是默认值 | `.cache/mnn-llm-export/<model-name>/` | `llm_demo` / `llm_bench` |
| PIC PagedAttention LLM | `transformers/pic_llm/export/llmexport.py` | `MNN_LLM_EXPORTER=pic` | `.cache/weight/<model-name>/` | `pic_llm_demo` / `pic_llm_bench` |
| PrefixAttention / prefix cache | 通过 `MNN_LLM_EXPORT_SCRIPT` 指定的 prefix exporter | `MNN_LLM_EXPORTER=prefix` | `.cache/weight/<model-name>/` | 对应 PrefixLLM runtime |

## 两类导出的区别和限制

普通导出使用 `transformers/llm`，目标是得到可用的基线 MNN LLM 模型。它不写 `paged_attention: true` 或 `prefix_attention: true`，也不会验证本仓库新增的 `PagedAttention`/`PrefixAttention` op；这类模型用于普通 LLM runtime、功能回归和常规量化验证。普通导出的产物不要放进 `.cache/weight`，避免和 PagedAttention/PrefixAttention 测试模型混淆。

PIC/Prefix 特化导出用于验证 paged KV cache 或 prefix cache。PIC exporter 默认导出 `PagedAttention`，并在 `llm_config.json` 写入 `paged_attention: true`；Prefix exporter 用于 `PrefixAttention`/prefix cache 专用图。它们依赖本仓库同一份 schema 构建出的 `MNNConvert` 和 runtime，不要依赖 PyPI `MNN.tools.mnnconvert` fallback；产物不能当作普通 `transformers/llm` 基线模型来对比。

`dualgraph` / `--pic_export_decode_graph` 只允许出现在 PIC/PagedAttention 导出路径中，输出目录必须是 `.cache/weight/<model>/`，运行侧只允许 `pic_llm_demo` / `pic_llm_bench` / `pic_server`。普通 MNN LLM 导出不能启用 dualgraph，也不能把 dualgraph 产物放进 `.cache/mnn-llm-export/` 或用于 `llm_bench` / `llm_demo` 的 normal baseline。

PIC dualgraph 导出由两个独立轴组成：

- 设备轴：只使用规范设备名 `jetson`、`rhinopi`、`orangepi`。
- 图角色轴：`llm.mnn` 是 `prefill` 角色，必须按普通 PIC/PagedAttention prefill / graph-boundary 逻辑导出；`llm_decode.mnn` 是 `decode` 角色，才允许使用 decode-only rewrite。

dualgraph 的 prefill 图和 decode 图必须隔离：`llm.mnn` 不能因为启用 dualgraph 或 decode-only fusion flag 而改写 gate/up、SiluMul、NHWC linear、PicScore/PicSparse 等 prefill 结构；只有 `llm_decode.mnn` 可以按 decode-only 逻辑去掉 `pic_recompute_budget`、`PicScoreAttention` / `PicSparseAttention` 和不需要的 decode 小算子。两个图必须共享同一个 `llm.mnn.weight`，配置中应写 `llm_decode_shared_weight=true`，运行时只维护一份权重内存和同一套当前请求 PagedCache。

PIC 需要固定预分配 KV token 上限时追加：

```text
--paged_kv_max_tokens <N>
```

## PIC 导出两轴契约

PIC 导出必须明确审定目标设备，避免 Jetson / RhinoPi / OrangePi 的实验参数互相污染。设备名只能使用：

```text
--pic_export_device jetson
--pic_export_device rhinopi
--pic_export_device orangepi
```

脚本入口也支持：

```bash
MNN_PIC_EXPORT_DEVICE=jetson  MNN_LLM_EXPORTER=pic bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh <model> -- --paged_kv_max_tokens 4096
MNN_PIC_EXPORT_DEVICE=rhinopi MNN_LLM_EXPORTER=pic bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh <model> -- --paged_kv_max_tokens 4096
MNN_PIC_EXPORT_DEVICE=orangepi MNN_LLM_EXPORTER=pic bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh <model> -- --paged_kv_max_tokens 4096
```

设备契约规则：

- `jetson`：CUDA family，只允许 CUDA/Jetson decode fusion；禁止 `pic_decode_tiny_mlp_fusion`、`pic_decode_silu_nhwc_down_fusion` 这类 RhinoPi/Adreno-only flag。
- `rhinopi`：Adreno family，允许 `PicAdreno*` 导出；不要和 Jetson CUDA packed/gateup 实验参数共用同一个导出命令或产物目录。
- `orangepi`：generic/Mali family，不启用 CUDA/Adreno 专属 decode fusion；传入 `pic_decode_tiny_fusion`、`pic_decode_gateup_fusion`、`pic_decode_nhwc_linear_fusion` 等 backend-specific flag 必须 fail-fast。
- `pic_decode_gateup_direct_fusion`、`pic_decode_gateup_split_fusion`、`pic_decode_silu_nhwc_down_fusion` 必须和 `pic_decode_gateup_fusion` 一起使用，否则导出图结构不明确。

导出后的 `export_args.json` 和 `llm_config.json` 必须记录：

```text
pic_export_contract_version
pic_export_device
pic_export_device_family
pic_export_graph_role
pic_decode_rewrites_enabled
pic_decode_fusion_backend
pic_decode_fusion_family
```

dualgraph 还必须记录 `pic_decode_graph_config`，用于描述 `llm_decode.mnn` 的 decode 角色配置；主层 `pic_export_graph_role` 应对应 `llm.mnn`，即 `prefill`。不要用 decode graph 的配置解释 prefill 图结构。

同一个 `.cache/weight/<model>/` 目录不要在不同设备契约之间反复覆盖；需要 A/B 时把设备和实验名写进 `MNN_LLM_EXPORT_DST` 的目录名。

`--skip_weight` 是导出流程/图结构 smoke，不是可用于正确性或性能的模型产物。skip-weight 模型可以缺少真实 embedding / lm_head 数值；测试输出、PIC server 精度、full-reuse/cacheblend/epic 延迟时必须使用真实权重导出，或者明确标注只是结构诊断。对 GLM / GLM-Edge 等 `tie_word_embeddings=false` 的模型，导出后必须确认 `export_args.json` 中 `tie_word_embeddings` 仍为 false，`llm_config.json` 不应出现错误的 `tie_embeddings`；否则输入 embedding 可能读到 lm_head 或 EOF 后占位数据，典型现象是 NUL、`APP` 或无意义重复 token。

## 基本工作流

1. 所有命令默认从本 MNN 仓库根目录执行。

2. 普通导出只读源码；如果要改 exporter 或 MNN schema/runtime，先检查：

```bash
git status --short
```

3. Python 固定使用 `kvshare-edge` conda 环境。不要用系统 `python`、`python3` 或 base 环境直接跑 exporter。
4. 优先使用本 Skill 自带脚本。它会解析 ModelScope cache、规范化输出目录、保存日志并检查关键产物。
5. 输出目录按 exporter 固定：普通 `mnn` 只能写 `.cache/mnn-llm-export/<model-name>/`；`pic` 和 `prefix` 只能写 `.cache/weight/<model-name>/`。日志默认在 `.cache/logs/<exporter>-llm-export/`。

## 平台产物解析

导出脚本会按当前平台自动查找本仓库构建出的 converter：

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
MNNCONVERT_PATH="${MNNCONVERT_PATH:-$MNN_ARTIFACT_ROOT/bin/MNNConvert}"
```

不要在命令里写死旧路径。需要覆盖时优先设置 `MNN_ARTIFACT_PLATFORM` 或 `MNN_ARTIFACT_ROOT`；只有调试特殊 converter 时才直接设置 `MNNCONVERT_PATH`。

## ModelScope 路径解析

优先使用用户给的模型目录。否则脚本按 `MODELSCOPE_CACHE_ROOT` 和常见 ModelScope cache 位置查找模型：

```text
MODELSCOPE_CACHE_ROOT
$HOME/.cache/modelscope/hub/models
```

ModelScope cache 常把 `.` 编码成 `___`：

```text
Qwen/Qwen3.5-2B              -> Qwen/Qwen3___5-2B
AI-ModelScope/Llama-3.2-3B   -> AI-ModelScope/Llama-3___2-3B
```

源模型目录至少要有 `config.json` 和 `*.safetensors`、`pytorch_model*.bin` 或 `model*.bin`。`MNN/*-MNN` 这类目录通常已经是导出产物，不要作为 `--path` 输入。

## 推荐脚本

无参数默认用 `transformers/llm` 导出 Llama 3.2 3B Instruct，并保存到 `.cache/mnn-llm-export/<model-name>/`：

```bash
bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh
```

默认模型：

```text
AI-ModelScope/Llama-3___2-3B-Instruct -> .cache/mnn-llm-export/AI-ModelScope__Llama-3___2-3B-Instruct
```

导出一组普通模型时，显式循环模型名，仍使用默认 `MNN_LLM_EXPORTER=mnn`：

```bash
for model in \
  AI-ModelScope/Llama-3.2-3B-Instruct \
  Qwen/Qwen3.5-2B
do
  bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh "$model"
done
```

普通 MNN 导出：

```bash
bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh \
  AI-ModelScope/Llama-3.2-3B-Instruct
```

PIC PagedAttention 导出：

```bash
MNN_LLM_EXPORTER=pic \
bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh \
  AI-ModelScope/Llama-3.2-3B-Instruct \
  -- --paged_kv_max_tokens 4096
```

PrefixAttention 导出：

```bash
MNN_LLM_EXPORTER=prefix \
MNN_LLM_EXPORT_SCRIPT=path/to/prefixllm/export/llmexport.py \
bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh \
  AI-ModelScope/Llama-3.2-1B-Instruct
```

常用覆盖：

```bash
MNN_QUANT_BIT=8 bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh AI-ModelScope/Llama-3.2-3B-Instruct
MNN_QUANT_BLOCK=128 bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh AI-ModelScope/Llama-3.2-3B-Instruct
MNN_LLM_EXPORT_DST=.cache/mnn-llm-export/AI-ModelScope__Llama-3___2-3B-Instruct-test bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh AI-ModelScope/Llama-3.2-3B-Instruct
MNN_EXPORT_DRY_RUN=1 MNN_LLM_EXPORTER=pic bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh AI-ModelScope/Llama-3.2-3B-Instruct
```

`MNN_LLM_EXPORT_ROOT` 如果设置，只允许解析为当前 exporter 的默认根：普通 `mnn` 是 `.cache/mnn-llm-export`，`pic`/`prefix` 是 `.cache/weight`。`MNN_LLM_EXPORT_DST` 如果设置，只允许解析为该默认根的直接子目录。

额外 exporter 参数放在 `--` 后面：

```bash
bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh \
  AI-ModelScope/Llama-3.2-3B-Instruct \
  -- --sym --lm_quant_bit 8
```

## Converter 要求

普通 MNN 导出可以使用 exporter 默认的 converter 解析逻辑。PIC PagedAttention 和 PrefixAttention 导出必须使用同一份 schema 构建出的 `MNNConvert`，不能依赖 PyPI `MNN.tools.mnnconvert` fallback；否则本地新增 op 可能被写成 `type=-1`。

脚本会自动查找：

```text
$MNN_ARTIFACT_ROOT/bin/MNNConvert
.cache/build/<platform>_cuda/MNNConvert
.cache/build/<platform>/MNNConvert
.cache/build/mnn/<platform>_cuda/MNNConvert
.cache/build/mnn/<platform>/MNNConvert
```

如果 converter 不存在，先构建：

```bash
BUILD_TARGET=MNNConvert \
JOBS=6 CUDA_ARCHS=72 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

## 产物检查

导出后必须检查对应输出目录。普通导出使用 `.cache/mnn-llm-export/<model>/`，PIC/Prefix 使用 `.cache/weight/<model>/`：

```bash
MODEL_DIR=.cache/mnn-llm-export/<model>
# or:
MODEL_DIR=.cache/weight/<model>
```

```text
config.json
llm_config.json
llm.mnn
llm.mnn.weight
```

PIC 还要确认：

```bash
rg '"paged_attention"[[:space:]]*:[[:space:]]*true' "$MODEL_DIR/llm_config.json"
```

Prefix 还要确认：

```bash
rg '"prefix_attention"[[:space:]]*:[[:space:]]*true' "$MODEL_DIR/llm_config.json"
```

需要检查图 op 时，用当前仓库构建出的 `MNNConvert` 转 JSON：

```bash
MNN_ARTIFACT_PLATFORM="${MNN_ARTIFACT_PLATFORM:-jetson}"
MNN_ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-$PWD/.cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM}"
"$MNN_ARTIFACT_ROOT/bin/MNNConvert" \
  -f MNN \
  --modelFile "$MODEL_DIR/llm.mnn" \
  --JsonFile "$MODEL_DIR/llm.mnn.json"

rg '"type": "PagedAttention"|PagedAttention|"type": -1' \
  "$MODEL_DIR/llm.mnn.json"
```

Prefix 对应检查 `PrefixAttention`。

## 手动命令

普通 MNN：

```bash
conda run -n kvshare-edge python transformers/llm/export/llmexport.py \
  --path "$MODELSCOPE_CACHE_ROOT/AI-ModelScope/Llama-3___2-3B-Instruct" \
  --dst_path .cache/mnn-llm-export/AI-ModelScope__Llama-3___2-3B-Instruct \
  --export mnn \
  --quant_bit 4 \
  --quant_block 64 \
  --embed_bit 16
```

PIC PagedAttention：

```bash
MNN_ARTIFACT_PLATFORM="${MNN_ARTIFACT_PLATFORM:-jetson}"
MNN_ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-$PWD/.cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM}"
conda run -n kvshare-edge python transformers/pic_llm/export/llmexport.py \
  --path "$MODELSCOPE_CACHE_ROOT/AI-ModelScope/Llama-3___2-3B-Instruct" \
  --dst_path .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct \
  --export mnn \
  --quant_bit 4 \
  --quant_block 64 \
  --embed_bit 16 \
  --mnnconvert "$MNN_ARTIFACT_ROOT/bin/MNNConvert" \
  --pic_export_device jetson \
  --paged_kv_max_tokens 4096
```

PrefixAttention：

```bash
MNN_ARTIFACT_PLATFORM="${MNN_ARTIFACT_PLATFORM:-jetson}"
MNN_ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-$PWD/.cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM}"
conda run -n kvshare-edge python path/to/prefixllm/export/llmexport.py \
  --path "$MODELSCOPE_CACHE_ROOT/AI-ModelScope/Llama-3___2-1B-Instruct" \
  --dst_path .cache/weight/AI-ModelScope__Llama-3___2-1B-Instruct \
  --export mnn \
  --quant_bit 4 \
  --quant_block 64 \
  --embed_bit 16 \
  --mnnconvert "$MNN_ARTIFACT_ROOT/bin/MNNConvert"
```

缺 Python 依赖时，在当前环境安装：

```bash
conda run -n kvshare-edge python -m pip install -r transformers/llm/export/requirements.txt
```

不要把依赖缓存、导出模型、ONNX 临时文件或日志加入 git。
