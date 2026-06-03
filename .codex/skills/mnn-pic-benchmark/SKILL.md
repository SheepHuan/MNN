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
- `cacheblend/epic/kvshare` 不应静默退化为 `full-compute-fallback`。

精度恢复 metadata 重点看：

```text
native-epic-sparse-recompute
native-cacheblend-sparse-recompute
native-kvshare-sparse-recompute
metadata.pre_score_kv_source=full_prompt_reference
metadata.post_score_reuse_kv_source=cached_pic_kv
```

`kvshare` 当前 MNN C++ 使用 K/V delta influence proxy；如果用户要求和 HF Python autograd `delta-a` 精确一致，需要单独实现 query/attention-output gradient scoring。

## 常见问题

- 本机无法连 Jetson 服务：确认 SSH tunnel 进程还在，或改用远端 `--host 0.0.0.0` 并把 `--base-url` 指向 `http://192.168.101.192:<port>`。
- 端口冲突：换 `--port` 和 `--local-port`，或远端 `pgrep -af "pic_server.*<port>"` 后清理。
- bench 找不到数据：优先加 `--local-files-only` 使用已有 HF cache；确实缺数据再回到 kvshare-edge 顶层数据准备流程。
- CUDA backend 回退：检查远端 artifact 的 `lib/libMNN_Cuda_Main.so` 和 `LD_LIBRARY_PATH`。
- 远端服务崩溃：先看 `.cache/logs/mnn_pic_dataset_bench_<run_id>.log`，摘出 CUDA illegal access/OOM/backend fallback。
