# MNN Agent Guide

## 范围

本目录是本 MNN 仓库。Codex 只使用本仓库 `.codex/skills/` 下的 MNN 专属 skills。

MNN 专属 Codex skills 由本目录维护：

```text
.codex/skills/mnn-add-new-op/SKILL.md  新增或扩展 MNN 算子
.codex/skills/mnn-build-artifacts/SKILL.md  MNN Jetson/x64 CUDA/LLM 产物构建与检查
.codex/skills/mnn-llm-bench/SKILL.md  运行和诊断 MNN llm_bench / demo，比较普通 LLM 与 PIC/PagedAttention 输出
.codex/skills/mnn-llm-export/SKILL.md  导出 MNN LLM 或 prefixllm 模型
.codex/skills/mnn-ops-bench/SKILL.md  CUDA 单算子精度/性能测试
.codex/skills/mnn-opt-ops/SKILL.md  CPU/CUDA 算子优化和 Jetson 验证流程
.codex/skills/mnn-pic-benchmark/SKILL.md  本机交叉编译 MNN PIC server 产物、推到 Jetson 启服务，并从本机跑数据集 benchmark
.codex/skills/mnn-support-new-llm/SKILL.md  在 transformers/pic_llm 中适配新 LLM 模型
```

上游 MNN 原生 skills 仍保留在 `skills/`，不要和 Codex skills 混放。Codex 不打开、不阅读、不引用 `skills/` 下的内容；即使任务看起来匹配，也不要读取 `skills/*/SKILL.md` 或把其中内容作为依据。需要新增或调整工作流时，在 `.codex/skills/` 中维护对应 Codex skill。

## 进入本目录后

1. 先读本 MNN 仓库根目录 `AGENTS.md`。
2. 修改 MNN 源码前先在本仓库运行：

```bash
git status --short
```

不要覆盖用户已有改动。

3. 仓库级构建、输出同步和测试命令默认从本 MNN 仓库根目录执行。
4. 构建产物、模型缓存、导出结果和临时文件放在本仓库 `.cache/`、`output/` 或用户指定的本地目录，不要提交到本仓库。

## 何时读 skill

- MNN 构建、重编 CUDA MNN、CUDA arch、Jetson/x64 MNN CUDA/LLM 产物构建、install/output 检查或残留构建进程诊断：读 `.codex/skills/mnn-build-artifacts/SKILL.md`。
- 新增或扩展 MNN 算子、schema/op type/shape/backend/注册表、`PrefixAttention`：读 `.codex/skills/mnn-add-new-op/SKILL.md`。
- CUDA 单算子 direct-op 精度/性能测试、`test/bench_ops/cuda` 或 CUDA event kernel 耗时统计：读 `.codex/skills/mnn-ops-bench/SKILL.md`。
- CPU/CUDA 算子性能优化、分析算子慢因，或在本机交叉编译产物并推到 Jetson 验证：读 `.codex/skills/mnn-opt-ops/SKILL.md`。
- 从 ModelScope 或 Hugging Face/Transformers 格式模型导出 MNN LLM、PIC LLM PagedAttention 或 PrefixLLM/PrefixAttention 模型：读 `.codex/skills/mnn-llm-export/SKILL.md`。
- 运行 `llm_bench` / `llm_demo` / `pic_llm_bench` / `pic_llm_demo`、测试 CUDA/OpenCL/Vulkan/CPU LLM 推理、比较普通 MNN LLM 与 PIC/PagedAttention LLM 输出、确认 GPU 后端注册、检查 execution class 日志、排查后端回退，或围绕 LLM bench 采集 DF power / Jetson `tegrastats` 功耗：读 `.codex/skills/mnn-llm-bench/SKILL.md`。
- 本机交叉编译 MNN `pic_server` / `libpic_llm` / CUDA artifact，推送到 Jetson，远端启动 MNN PIC server，并从本机运行 `impl/pic_bench/cli.py` 数据集 benchmark 向 Jetson 发请求：读 `.codex/skills/mnn-pic-benchmark/SKILL.md`。
- 新增、适配或诊断一个尚未支持的 Hugging Face/ModelScope LLM 或多模态 LLM 模型，并修改 `transformers/pic_llm` 的 mapper/config/model/vision/audio/export 流程：读 `.codex/skills/mnn-support-new-llm/SKILL.md`。

如果任务没有对应的 `.codex/skills/` 覆盖，按本文件、仓库源码和用户上下文处理，不要读取 `skills/` 下的内容。不要批量阅读无关文档；按当前任务打开需要的文件即可。

## LLM 输出对比约定

比较普通 MNN LLM 和 PIC/PagedAttention LLM 的 prompt 输出精度时，必须使用两个模式各自真实导出的模型目录：

```text
普通 MNN LLM:       .cache/mnn-llm-export/<model>/config.json  用 llm_demo 跑
PIC/PagedAttention: .cache/weight/<model>/config.json           用 pic_llm_demo 跑
```

不要把 `.cache/weight/` 下的 PIC 模型复制或改配置当作普通模型对比；普通模型以 `.cache/mnn-llm-export/` 下的正常导出为准。两边用同一个 prompt、相同 backend / precision / memory / sampler 配置，并分别从各自模型目录执行 demo。

## PIC Server 约定

`transformers/pic_llm/engine/app/pic_server.cpp` 是 MNN PIC 自维护的独立 HTTP server，不依赖 `mls`。构建产物名固定为 `pic_server`，由 `.codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh` 安装到 artifact root 的 `bin/` 下；旧的 `build_jetson_artifacts.sh` 只作为兼容入口转发。

首个协议入口是：

```text
POST /v1/prefill/text
POST /v1/kv/pic_caches
POST /v1/chat/completions
POST /chat/completions
```

它接受内联文本 `{"id":"doc-1","type":"text","content":"...","force":false}`，用 PIC/PagedAttention 模型 prefill 文档，并按 layer 导出 raw KV cache：key 和 value 分开写到 `.k` / `.v` 文件，真实 `batch/kv_heads/head_dim/dtype_bytes` 写到同层 `.json` sidecar。导出前 CPU/CUDA PagedAttention 必须对 key cache 做 inverse RoPE，metadata 标注 `kv_layout.layout=mnn_paged_attention_raw_v1` 和 `kv_layout.key_rope_state=canonical_no_rope`；value cache 保持原布局。

`POST /v1/kv/pic_caches` 接受 `{"id":"pic-1","text_cache_refs":[{"id":"doc-1"}],"selection_algorithm":"full-reuse"}`，解析一个或多个 text cache 并返回 PIC metadata。`POST /v1/chat/completions` / `/chat/completions` 兼容 OpenAI 风格 `messages`，可在 `pic_cache` 中内联同一份 spec；prompt 中的 `{{pic_cache}}` 是 PIC 注入位置。当前 MNN 原生执行语义：

- `full-reuse`：不重算 PIC token，CPU/CUDA PagedAttention 从磁盘 `.k/.v` 读取 canonical_no_rope key，按当前 logical slot 重新施加 RoPE，并写入 paged KV slots。
- `full-compute`：不读取磁盘 PIC KV，按 prelude + PIC tokens + suffix 完整计算。
- `epic`：参考 `hf_pic_runtime` 的 EpicPlanner，按 `pic_recompute_ratio` 选择 PIC 开头连续 token；`score_layer_idx` 之前的层保持 token 正常计算语义，使用 full-prompt reference PIC K/V hydrate slots；从 `score_layer_idx` 开始，选中的 PIC token 和 suffix/非复用 KV token 参与计算，其他 PIC 位置直接复用磁盘 KV。
- `cacheblend` / `delta-v`：参考 Python CacheBlend/DeltaV planner，用 score layer 上 full-prompt reference 与 cached PIC value 的 mean-abs delta 选 top-ratio token；执行模式为 `native-cacheblend-sparse-recompute`。
- `kvshare` / `delta-a`：执行模式为 `native-kvshare-sparse-recompute`，复用同一套 sparse prefill layer plan；当前 MNN C++ 没有 HF autograd，因此评分使用 full-vs-cached K/V delta influence proxy，metadata 中必须标注与 Python `attention_output` gradient influence 的差异。

## MNN 修改约束

- 只在必要范围内修改本仓库源码。
- 不要读取或修改 `schema/private/` 和 `source/internal/`。
- 不要把 `3rd_party/cutlass/` 等构建时下载目录加入 git。
- 不要用旧路径 `impl/mnn` 新增说明或脚本默认值。
- 修改 schema 后使用 `bash schema/generate.sh`，并检查生成文件 diff。
- 更新注册表时使用 `python3 tools/script/register.py .`，并确认生成变化合理。
