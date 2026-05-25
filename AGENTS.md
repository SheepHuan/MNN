# MNN Agent Guide

## 范围

本目录是本 MNN 仓库。Codex 只使用本仓库 `.codex/skills/` 下的 MNN 专属 skills。

MNN 专属 Codex skills 由本目录维护：

```text
.codex/skills/mnn-add-new-op/SKILL.md  新增或扩展 MNN 算子
.codex/skills/mnn-build-artifacts/SKILL.md  MNN Jetson/x64 CUDA/LLM 产物构建与检查
.codex/skills/mnn-llm-bench/SKILL.md  运行和诊断 MNN llm_bench
.codex/skills/mnn-llm-export/SKILL.md  导出 MNN LLM 或 prefixllm 模型
.codex/skills/mnn-ops-bench/SKILL.md  CUDA 单算子精度/性能测试
.codex/skills/mnn-opt-ops/SKILL.md  CPU/CUDA 算子优化和 Jetson 验证流程
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
- 运行 `llm_bench`、测试 CUDA/OpenCL/Vulkan/CPU LLM 推理、确认 GPU 后端注册、检查 execution class 日志或排查后端回退：读 `.codex/skills/mnn-llm-bench/SKILL.md`。
- 新增、适配或诊断一个尚未支持的 Hugging Face/ModelScope LLM 或多模态 LLM 模型，并修改 `transformers/pic_llm` 的 mapper/config/model/vision/audio/export 流程：读 `.codex/skills/mnn-support-new-llm/SKILL.md`。

如果任务没有对应的 `.codex/skills/` 覆盖，按本文件、仓库源码和用户上下文处理，不要读取 `skills/` 下的内容。不要批量阅读无关文档；按当前任务打开需要的文件即可。

## MNN 修改约束

- 只在必要范围内修改本仓库源码。
- 不要读取或修改 `schema/private/` 和 `source/internal/`。
- 不要把 `3rd_party/cutlass/` 等构建时下载目录加入 git。
- 不要用旧路径 `impl/mnn` 新增说明或脚本默认值。
- 修改 schema 后使用 `bash schema/generate.sh`，并检查生成文件 diff。
- 更新注册表时使用 `python3 tools/script/register.py .`，并确认生成变化合理。
