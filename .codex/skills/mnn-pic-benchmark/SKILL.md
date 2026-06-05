---
name: mnn-pic-benchmark
description: 当用户要求把本 MNN 仓库本机交叉编译出的 pic_server/libpic_llm/CUDA 产物推送到 Jetson，远端启动 MNN PIC server，并在本机用 impl/pic_bench 数据集 benchmark 向 Jetson 发 /v1/prefill/text、/v1/chat/completions 请求，测试 HotpotQA/AmbigQA/2Wiki/MuSiQue/SAMSum/MultiNews 上 full-reuse/full-compute/cacheblend/epic/kvshare 精度、稳定性或延迟时使用。
---

# MNN PIC Benchmark

本 Skill 负责 MNN PIC server 的端到端数据集验证闭环：本机编译 MNN 产物，推送到 Jetson 运行服务，本机启动数据集 bench，通过 SSH tunnel 或远端地址向 Jetson 发请求。

## 相关 Skills

- 构建和 artifact 检查：读 `.codex/skills/mnn-build-artifacts/SKILL.md`。
- Jetson 设备、rsync 和 PIC server smoke：读 `.codex/skills/mnn-opt-ops/SKILL.md`。
- LLM/PIC server 输出判断：读 `.codex/skills/mnn-llm-bench/SKILL.md`。

参考实现来源是 kvshare-edge 顶层 skills：

```text
/root/code/kvshare-edge/.codex/skills/pic-benchmark
/root/code/kvshare-edge/.codex/skills/pic-kvcache-server
/root/code/kvshare-edge/.codex/skills/pic-precision-recovery
```

这里只记录 MNN 仓库专用流程，不把顶层 HF runtime 的启动命令搬进 MNN。

## 固定设备

Jetson：

```text
jetson@192.168.101.192
```

远端 MNN 仓库：

```text
/home/jetson/code/kvshare-edge/impl/MNN
```

本机 MNN 仓库：

```text
/root/code/kvshare-edge/impl/MNN
```

本机数据集 bench 仓库根目录：

```text
/root/code/kvshare-edge
```

默认 artifact：

```text
.cache/output/mnn/artifacts/jetson_cross_cuda
```

默认 PIC 模型配置：

```text
.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct/config_cuda_greedy.json
```

## 一键脚本

优先使用本 Skill 的脚本：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_mnn_pic_dataset_bench.sh \
  --port 18096 \
  --local-port 18096 \
  --run-id mnn_pic_hotpotqa20_kvshare \
  -- \
  --dataset hotpotqa \
  --mode pic_cache_reuse \
  --phase both \
  --cases 20 \
  --context-len 1500 \
  --max-tokens 64 \
  --temperature 0.0 \
  --local-files-only \
  --min-doc-tokens 0 \
  --force-cache \
  --reset-before-each-infer \
  --pic-selection-algorithm kvshare \
  --pic-recompute-ratio 0.20 \
  --pic-recompute-score-layer-idx 1
```

脚本会：

1. 从本 MNN 仓库交叉编译 `pic_server`。
2. 安装并补拷贝 `pic_server`、`libpic_llm.so`、`libMNN_Cuda_Main.so` 到 artifact root。
3. `rsync` artifact 到 Jetson。
4. 在 Jetson 上启动 `$ART/bin/pic_server --host 127.0.0.1 --port <port>`。
5. 建立本机 `127.0.0.1:<local-port>` 到 Jetson `127.0.0.1:<port>` 的 SSH tunnel。
6. 从 `/root/code/kvshare-edge` 运行 `impl/pic_bench/cli.py`，`--base-url` 指向本地 tunnel。
7. 结束时清理 tunnel 和远端同端口 `pic_server`。

默认 bench 参数未传时使用 HotpotQA 20 case、`kvshare`、20% recompute、score layer 1。

常用覆盖：

```bash
# 跳过编译，只推送并启动服务
--skip-build

# 跳过 rsync，只重启远端已有 artifact
--skip-rsync

# 切换算法
-- --pic-selection-algorithm cacheblend
-- --pic-selection-algorithm epic
-- --pic-selection-algorithm full-reuse

# 复用已有 manifest 做 infer
-- --phase infer --manifest /path/to/manifest.jsonl --pic-selection-algorithm kvshare
```

输出默认写到：

```text
/root/code/kvshare-edge/.cache/mnn-pic-benchmark/runs/<run_id>/
/root/code/kvshare-edge/.cache/mnn-pic-benchmark/logs/<run_id>/
```

## Prefill-only 延迟对比

当前做 PIC 延迟分析时暂不比较 decode 性能，只测模型拿到请求后完成 prefill 的时间。PIC server `/v1/chat/completions` 请求必须显式设置：

```json
{"max_tokens": 0}
```

`max_tokens=0` 是 prefill-only 请求：服务端应完成 prelude / PIC hydrate / sparse recompute / suffix prefill，但不生成第一个 decode token，也不把 decode 或采样时间计入结果。不要把 `max_tokens=1` 的首 token输出或额外 next-logits forward 混入这组数据。

请求隔离由客户端显式 `/reset` 完成，不要在 CUDA/OpenCL PagedAttention backend 里按请求自动清整块 PagedCache；backend 级清理会影响 decode/continuous 状态和性能。数据集 bench、ratio sweep、手工 smoke 在每个独立 infer 前调用：

```bash
curl -fsS -X POST "$BASE_URL/reset" -H 'Content-Type: application/json' -d '{}'
```

Jetson CUDA 服务刚启动后的第一轮 prefix-cache `PendingWrite` 可能包含 CUDA/session warm-up 噪声，而且这个问题可能和 token length / shape bucket 有关；泛化短文本 warm-up 不一定覆盖正式样本。正式构建 text cache 或统计延迟前，客户端应对同目标长度/同 shape bucket 做 disposable `/v1/prefill/text` warm-up，或对首个 text cache 做重建/验证；warm-up 和重建探测都不进入正确性或性能报告。

磁盘 K/V 边界必须一起验证：只有 `/v1/prefill/text` 可以生成持久 text cache `.k/.v`。`/v1/chat/completions` 里的 `cacheblend`、`kvshare`、`epic`、`full-compute`、full-reference 和 scoring 都不得调用 prefix-cache `PendingWrite`、`setPrefixCacheFile`，不得写 scratch reference `.k/.v`，也不得把 full-reference K/V 读回 CPU 做 `std::vector` delta 扫描。cacheblend 的 score 应由 CUDA/OpenCL scoring 分支在 score layer 直接读 PagedCache / device buffer 得到，并在 GPU/CL/CUDA 上完成 top-k；不得拷回完整 score vector 到 CPU 排序。

多 ratio 测试不共享 scoring 结果：`cacheblend 1%/5%/10%/20%/30%` 可以共享同一份已构建 text cache、同一模型和同一 suffix，但每个 ratio 必须发送独立 `/v1/chat/completions` 请求，并在该请求内重新完成 full-reference / scoring / GPU top-k 选择。不得为了加速报告跨 ratio 复用 score vector、排序结果或 recompute logical indices；每个 ratio 的 latency 都应包含本次请求自己的 scoring 成本。

历史调试记录：Jetson CUDA 曾复现过首轮 `/v1/prefill/text` prefix-cache `PendingWrite` 后层 `.k/.v` 写成 `ff7f...`，导致后续 full-reuse / sparse reuse 输出大量 `!`。这不是 decode 性能问题，而是 legacy prefix-cache 写盘路径与当前 PagedCache 语义的交界问题。不要用 backend 按请求自动清整块 PagedCache 兜底；这会影响 decode/continuous 状态和性能。也不要依赖临时 clone prefill module 隔离 prefix-write，因为 PagedAttention `onClone()` 仍可能共享 PagedCache/session 资源。长期方向是最小化 legacy prefix-cache 兼容：只有 `/v1/prefill/text` 可以持久导出 `.k/.v`；chat、full-reference、scoring、top-k、sparse recompute 全部维护当前请求 PagedCache，不写 scratch `.k/.v`，不绕过 slot table，不把磁盘 KV 拷回 CPU 做 scoring。

比较 cacheblend 重计算预算时，同一台设备、同一模型、同一份 text cache、同一 suffix 下至少跑这些模式：

```text
full-reuse              selection_algorithm=full-reuse, recompute_token_count=0
full-compute            selection_algorithm=full-compute
cacheblend 1%           selection_algorithm=cacheblend, pic_recompute_ratio=0.01
cacheblend 5%           selection_algorithm=cacheblend, pic_recompute_ratio=0.05
cacheblend 10%          selection_algorithm=cacheblend, pic_recompute_ratio=0.10
cacheblend 20%          selection_algorithm=cacheblend, pic_recompute_ratio=0.20
cacheblend 30%          selection_algorithm=cacheblend, pic_recompute_ratio=0.30
```

报告时不要只列 cacheblend 自身耗时；必须同时给出：

```text
normal LLM full-compute prefill baseline
PIC full-reuse prefill latency
PIC full-compute prefill latency
cacheblend 各 ratio prefill latency
cacheblend / full-reuse 倍数
cacheblend / PIC full-compute 倍数
cacheblend / normal LLM full-compute 倍数
```

报告里如果写“同一份 text cache / suffix”，只表示输入条件对齐；不表示多个 ratio 共享一次 scoring。若脚本做了多 ratio sweep，必须确认每个 ratio 的 HTTP 请求、metadata 和计时都是独立记录。

普通 MNN LLM baseline 用真实普通导出模型目录跑 `llm_bench`，建议用同等 prompt token 长度并设 `-n 0`：

```bash
"$ART/bin/llm_bench" -m "$NORMAL_CONFIG" -a cuda -p <prompt_tokens> -n 0 -rep 3 -load false -j normal_prefill.json
```

只读取 JSON 中 `results[].type == "prefill"` 的 `prompt_len` 和 `tps`，计算 `prefill_s = prompt_len / tps`；不要把 decode 或 `ttft_est` 加进这张对比表。

## 手动流程

脚本不适合当前调试时，按下面顺序手动执行。

1. 本机交叉编译：

```bash
CUDA_TOOLKIT_ROOT="$PWD/.cache/sysroots/jetson_cuda" \
CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda" \
JOBS=8 CUDA_ARCHS=72 ENABLE_CROSS_CUDA=ON \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

2. 补齐 runtime artifact：

```bash
install -m 755 .cache/build/mnn/jetson_cross_cuda/pic_server \
  .cache/output/mnn/artifacts/jetson_cross_cuda/bin/pic_server
install -m 755 .cache/build/mnn/jetson_cross_cuda/libpic_llm.so \
  .cache/output/mnn/artifacts/jetson_cross_cuda/lib/libpic_llm.so
install -m 755 .cache/build/mnn/jetson_cross_cuda/source/backend/cuda/libMNN_Cuda_Main.so \
  .cache/output/mnn/artifacts/jetson_cross_cuda/lib/libMNN_Cuda_Main.so
```

3. 推到 Jetson：

```bash
rsync -a --delete .cache/output/mnn/artifacts/jetson_cross_cuda/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

4. 远端启动服务：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  REPO=$PWD && ART=$REPO/.cache/output/mnn/artifacts/jetson_cross_cuda && \
  CONFIG=$REPO/.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct/config_cuda_greedy.json && \
  KV_DIR=$REPO/.cache/kvshare/mnn_pic_dataset_bench && \
  LOG=$REPO/.cache/logs/mnn_pic_dataset_bench.log && \
  mkdir -p "$REPO/.cache/logs" && \
  LD_LIBRARY_PATH="$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" \
    "$ART/bin/pic_server" --config "$CONFIG" --host 127.0.0.1 --port 18096 \
    --kv-cache-dir "$KV_DIR" --model llama-pic > "$LOG" 2>&1'
```

5. 本机开 tunnel：

```bash
ssh -N -L 127.0.0.1:18096:127.0.0.1:18096 jetson@192.168.101.192
```

6. 本机运行数据集 bench：

```bash
cd /root/code/kvshare-edge
/root/miniconda3/envs/kvshare-edge/bin/python impl/pic_bench/cli.py run \
  --base-url http://127.0.0.1:18096 \
  --output-dir .cache/mnn-pic-benchmark/runs/manual_hotpotqa20 \
  --dataset hotpotqa \
  --mode pic_cache_reuse \
  --phase both \
  --cases 20 \
  --context-len 1500 \
  --max-tokens 64 \
  --temperature 0.0 \
  --local-files-only \
  --min-doc-tokens 0 \
  --force-cache \
  --reset-before-each-infer \
  --pic-selection-algorithm kvshare \
  --pic-recompute-ratio 0.20 \
  --pic-recompute-score-layer-idx 1
```

## 结果判断

成功标准：

- 本机脚本退出码为 `0`。
- 远端 `pic_server` ready，`/healthz` 返回 `{"status":"ok"}`。
- bench `summary.json.errors == 0`。
- `manifest.jsonl` 中有非空 text cache refs / segments。
- `predictions.jsonl` 中 PIC 请求响应带 `pic_cache.precision_recovery.execution_mode`。

自然语言正确性 smoke：

- 性能报告之外还要跑一组 `max_tokens>0` 的小问题，覆盖 `full-reuse`、`cacheblend`、`epic`。判定标准不是只看 HTTP 200，而是输出应包含文档中的可核验答案，且文本应是正常自然语言或短答案。
- 带 `pic_cache` 的 chat 请求必须在 `messages.content` 中显式包含 `{{pic_cache}}` 或 `pic_cache.placeholder` 指定的自定义 placeholder；不要再用无 placeholder 的 legacy 隐式插入路径做测试。
- 推荐固定文档里放一个唯一答案，例如 `BLUE-17`，请求 `{{pic_cache}}` 后问“secret launch code”。`temperature=0`、`top_k=1`、`top_p=1.0`，每种模式生成 16-32 token 即可。
- 解析响应时同时检查 `choices[0].message.content`、`data[0].message.content`、`data[0].choices[0].message.content` 和常见 `text/content/response/generated_text` 字段。MNN PIC server 常把 OpenAI 风格结果包在 `data[0]` 下，漏掉这一层会误判。
- 结果报告至少列出：设备、模式、HTTP status、`precision_recovery.execution_mode`、`recompute_token_count`、`reuse_token_count`、输出文本、是否包含期望答案。若 sparse 模式输出重复标点或乱码，即使 latency 正常也视为正确性失败。
- `cacheblend/epic/kvshare` 不应静默退化为 `full-compute-fallback`。
- `cacheblend` / `delta-v` 不应通过 prefix-cache `PendingWrite`、scratch `.k/.v` 或 CPU 读盘 delta 扫描冒充 native scoring；如果 GPU/CL/CUDA score + top-k 分支缺失，结果必须标为 unsupported/fallback。

精度恢复 metadata 重点看：

```text
native-epic-sparse-recompute
native-cacheblend-sparse-recompute
native-kvshare-sparse-recompute
metadata.pre_score_kv_source=request_pagedcache_full_compute
metadata.post_score_reuse_kv_source=cached_pic_kv
```

`kvshare` 当前 MNN C++ 使用 K/V delta influence proxy；如果用户要求和 HF Python autograd `delta-a` 精确一致，需要单独实现 query/attention-output gradient scoring。

## 常见问题

- 本机无法连 Jetson 服务：确认 SSH tunnel 进程还在，或改用远端 `--host 0.0.0.0` 并把 `--base-url` 指向 `http://192.168.101.192:<port>`。
- 端口冲突：换 `--port` 和 `--local-port`，或远端 `pgrep -af "pic_server.*<port>"` 后清理。
- bench 找不到数据：优先加 `--local-files-only` 使用已有 HF cache；确实缺数据再回到 kvshare-edge 顶层数据准备流程。
- CUDA backend 回退：检查远端 artifact 的 `lib/libMNN_Cuda_Main.so` 和 `LD_LIBRARY_PATH`。
- 远端服务崩溃：先看 `.cache/logs/mnn_pic_dataset_bench_<run_id>.log`，摘出 CUDA illegal access/OOM/backend fallback。
