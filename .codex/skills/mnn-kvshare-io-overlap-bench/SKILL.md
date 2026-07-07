---
name: mnn-kvshare-io-overlap-bench
description: 当用户要求在 Jetson、Orange Pi 5 Plus、Rhino Pi-X1/Aidlux Adreno 等设备上设计、运行或汇总 KVShare / PIC PagedAttention 的异步 layer-wise hydrate、磁盘 KV cache 读取与 GPU/OpenCL/CUDA sparse compute overlap 实验，绘制 IO timeline / latency breakdown，计算 overlap ratio、hidden IO latency、future 阻塞时间、prefill 上升阈值或不同模型/设备/上下文的 KV cache 加载阈值表时使用。
---

# MNN KVShare IO Overlap Bench

本 skill 专门用于验证 KVShare 的设备侧 IO pipeline：把持久 PIC cache 的磁盘 K/V 读取从串行 prefill 关键路径中拆出，按 layer 做 just-in-time hydrate，并测量多少 IO 被当前层 sparse compute 隐藏。

## 相关 skills

- 构建和同步 artifact：读 `.codex/skills/mnn-build-artifacts/SKILL.md`
- 启动设备侧 `pic_server` 并跑 prefill sweep：读 `.codex/skills/mnn-pic-benchmark/SKILL.md`
- 分析或修改 PagedAttention/OpenCL/CUDA 实现：读 `.codex/skills/mnn-pic-optimize/SKILL.md`
- 采集磁盘/DRAM 基线：读 `.codex/skills/mnn-device-io-bench/SKILL.md`

## 硬约束

1. 从 MNN 仓库根目录工作，先读 `AGENTS.md`；修改源码前运行 `git status --short`。
2. 只使用 `.codex/skills/` 下的 MNN 专属 skills；不要读取上游 `skills/`。
3. 正式实验必须只统计 prefill-only：PIC chat 请求使用 `max_tokens=0`，不要把 decode、采样、decode1 或 next-logits forward 算入 latency。
4. 只允许 `/v1/prefill/text` 构建持久 `.k/.v` text cache；chat、full-reference、scoring、sparse recompute 不得生成 scratch `.k/.v`。
5. 每个 ratio / context / variant 必须作为独立请求测量；不得跨请求复用 score vector、top-k indices 或 recompute plan。
6. OpenCL 正式结果必须 warm 目标 shape / ratio，并调用 `/v1/tune/update_cache` 写回 autotune cache；冷启动 kernel build、LWS tuning 和首轮 prefix-cache 探测不计入正式数据。
7. 报告中不要写成“磁盘直接 DMA 到 GPU”。准确表述是 UMA 上 CPU 线程将磁盘 `.k/.v` 读入 mapped PagedCache/source-slot buffer，GPU/OpenCL/CUDA kernel 随后在当前请求 PagedCache 上 re-apply RoPE 和 hydrate value。

## 实验入口

当任务是设计或执行 IO overlap 实验时，继续读取：

- [references/experiment.md](references/experiment.md)

该 reference 定义 ablation、指标、日志字段、阈值计算和推荐表格。不要在没有 per-layer IO/hydrate instrumentation 的情况下只用端到端 TTFT 伪造 overlap 数据；若缺少字段，先补 instrumentation 或明确标为 unavailable。

## 结果口径

正式输出至少包含：

- 每个设备/模型/context/ratio/variant 的 prefill latency。
- `total_disk_read_ms`、`exposed_io_ms`、`hidden_io_ms`、`overlap_ratio`。
- `async_read_layer_ratio` 与每层 `future_wait_ms` 分布。
- `hydrate_ms` / re-RoPE kernel 时间、sparse compute 时间、suffix compute 时间。
- 重计算阈值：在哪个 context 或 KV cache byte size 之后，hydrate/read 让 full-reuse 或低 ratio KVShare prefill latency 显著上升。

## 编辑后校验

更新本 skill 后至少运行：

```bash
python3 .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/mnn-kvshare-io-overlap-bench
```
