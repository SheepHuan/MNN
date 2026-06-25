---
name: mnn-pic-benchmark
description: 当用户要求把本 MNN 仓库本机交叉编译出的 pic_server/libpic_llm/CUDA/OpenCL 产物推送到 Jetson、Orange Pi 5 Plus 或 Rhino Pi-X1/Adreno，远端启动 MNN PIC server，并在本机用 impl/pic_bench 数据集 benchmark 发 /v1/prefill/text、/v1/chat/completions 请求，测试 HotpotQA/AmbigQA/2Wiki/MuSiQue/SAMSum/MultiNews 上 full-reuse/full-compute/cacheblend/epic/kvshare 精度、稳定性或延迟时使用。
---

# MNN PIC Benchmark

本 Skill 负责 MNN PIC server 的端到端数据集验证闭环：本机编译 MNN 产物，推送到 Jetson CUDA、Orange Pi 5 Plus OpenCL 或 Rhino Pi-X1/Adreno OpenCL 设备运行服务，本机启动数据集 bench，通过 SSH tunnel 或远端地址向目标设备发请求。

## 相关 Skills

- 构建和 artifact 检查：读 `.codex/skills/mnn-build-artifacts/SKILL.md`。
- Jetson/Orange Pi/Rhino Pi-X1 设备、rsync 和 PIC server smoke：读 `.codex/skills/mnn-opt-ops/SKILL.md`。
- LLM/PIC server 输出判断：读 `.codex/skills/mnn-llm-bench/SKILL.md`。

参考实现来源是 kvshare-edge 顶层 skills：

```text
/root/code/kvshare-edge/.codex/skills/pic-benchmark
/root/code/kvshare-edge/.codex/skills/pic-kvcache-server
/root/code/kvshare-edge/.codex/skills/pic-precision-recovery
```

这里只记录 MNN 仓库专用流程，不把顶层 HF runtime 的启动命令搬进 MNN。

## 固定设备

设备速查（本 Skill 的目标设备入口；Orange Pi 5 Plus / Rhino Pi-X1 信息必须保留在这里，不只写到其它 skill）：

```text
Jetson CUDA:
  ssh:          jetson@192.168.101.192
  remote repo:  /home/jetson/code/kvshare-edge/impl/MNN
  artifact:     .cache/output/mnn/artifacts/jetson_cross_cuda

Orange Pi 5 Plus OpenCL:
  ssh:          orangepi@192.168.101.113
  auth:         key-based login works from this host
  target env:   MNN_TARGET_DEVICE=orangepi5plus
  build dir:    .cache/build/mnn/orangepi5plus
  remote repo:  /home/orangepi/code/kvshare-edge/impl/MNN
  artifact:     .cache/output/mnn/artifacts/orangepi5plus
  cache root:   /mnt/ssd/code/.cache/mnn_opencl_pic
  backend:      OpenCL GPU

Rhino Pi-X1 / Adreno OpenCL:
  ssh:          aidlux@192.168.101.227
  password:     aidlux
  remote work:  /mnt/nvme/mnn_pic_opencl
  artifact:     /mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl
  models:       /mnt/nvme/mnn_pic_opencl/models/{pic,normal}/...
  cache root:   /mnt/nvme/mnn_pic_opencl/cache
  logs:         /mnt/nvme/mnn_pic_opencl/logs
  backend:      OpenCL / Adreno GPU
```

当前本机到这三台设备均按 SSH 登录流程使用；Orange Pi 已验证可从本机 key-based 登录。Rhino Pi-X1 如需密码登录，密码为 `aidlux`，若后续配置免密登录也按同一套 `--remote` / `--remote-art-rel` 参数跑。

Jetson：

```text
jetson@192.168.101.192
```

Orange Pi 5 Plus / OpenCL：

```text
orangepi@192.168.101.113
ssh: key-based login works from this host
```

Orange Pi 5 Plus 是 Linux AArch64 目标，默认使用 Arm GNU 11.3 AArch64 Linux toolchain 从本机交叉编译；PIC benchmark 主要验证 OpenCL GPU 路径，不使用 Jetson CUDA artifact、CUDA sysroot 或 `libMNN_Cuda_Main.so`。构建时使用：

```text
MNN_TARGET_DEVICE=orangepi5plus
```

默认 build / artifact：

```text
.cache/build/mnn/orangepi5plus
.cache/output/mnn/artifacts/orangepi5plus
```

远端 MNN 仓库默认：

```text
/home/orangepi/code/kvshare-edge/impl/MNN
```

默认同步目标：

```bash
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/home/orangepi/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/orangepi5plus/
```

Orange Pi 实验 run/log/KV cache 优先放到 SSD-backed cache root，避免写满远端仓库所在根分区；当前可写路径优先使用：

```text
/mnt/ssd/code/.cache/mnn_opencl_pic
```

OpenCL 正式性能测试前需要 warm 目标 shape / ratio，并写回 MNN OpenCL autotune cache；冷启动 kernel build、LWS tuning 和 cachefile 生成不计入正式 latency。

Rhino Pi-X1 / Adreno OpenCL：

```text
aidlux@192.168.101.227
password: aidlux
```

Rhino Pi-X1 也是 Linux AArch64 目标，默认使用 Arm GNU 11.3 AArch64 Linux toolchain 从本机交叉编译；性能测试走 OpenCL backend，目标 GPU 是 Adreno。不要沿用 Jetson CUDA artifact、CUDA sysroot 或 `libMNN_Cuda_Main.so`；Rhino Pi-X1 artifact 应单独放到设备专用目录，例如：

```text
.cache/output/mnn/artifacts/aidlux_adreno_opencl
```

Rhino Pi-X1 OpenCL benchmark 只验证 GPU/OpenCL 路径，启动前检查远端 `LD_LIBRARY_PATH` 包含对应 artifact `lib/`，并确认运行日志没有 CUDA backend、CPU fallback 或 OpenCL target unavailable。

## 固定模型权重位置

性能 benchmark 不在测试流程中反复同步模型权重。模型导出在本机完成后，只把每个模型固定放到对应设备 SSD/NVMe 模型目录；后续 `pic_server` 和 `llm_bench` 只引用这些固定路径：

```text
Jetson AGX Xavier:
  normal: /home/jetson/code/kvshare-edge/impl/MNN/.cache/mnn-llm-export/<model>/
  pic:    /home/jetson/code/kvshare-edge/impl/MNN/.cache/weight/<model-pic-boundary>/

Orange Pi 5 Plus:
  normal: /mnt/ssd/code/.cache/mnn_opencl_pic/models/normal/<model>/
  pic:    /mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/<model-pic-boundary>/

Rhino Pi X1:
  normal: /mnt/nvme/mnn_pic_opencl/models/normal/<model>/
  pic:    /mnt/nvme/mnn_pic_opencl/models/pic/<model-pic-boundary>/
```

验证一致性优先用 dry-run itemize，不直接传输：

```bash
rsync -anic --delete --out-format='%i %n%L' LOCAL_MODEL_DIR/ user@host:REMOTE_MODEL_DIR/
```

只有 dry-run 显示缺失或差异时，才针对具体模型和设备执行一次实际 `rsync -a --delete --partial --inplace`。不要在 benchmark run 脚本中自动同步权重；benchmark 脚本最多检查固定路径存在和 graph/config 能力。

## Prefill Benchmark Cache 复用

同一模型、设备、context 和 token span 的 `/v1/prefill/text` 持久 text cache 要复用固定 KV cache 目录，不要每个 run-id 重建一份：

```text
Jetson:    /home/jetson/code/kvshare-edge/impl/MNN/.cache/pic_prefill_latency_sweep/shared_kv/<model-key>/
OrangePi:  /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/shared_kv/<model-key>/
Rhino:     /mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/shared_kv/<model-key>/
```

benchmark 客户端应为文档 cache 使用稳定 id，例如 `bench-<model-key>-<backend>-ctx<context>-<token_hash>`，并默认发送 `force=false`。第二次及后续测试应直接命中已有 persistent text cache；只有显式要求重建或改变模型/token span 时才设置 `force=true`。run-id 只用于日志、summary 和临时服务目录，不应进入 text cache id，也不应作为 KV cache 存储目录的一部分。

正式 prefill latency sweep 不要使用按 run-id 命名的 KV cache 目录，例如 `<run_id>/kv` 或 `mnn_pic_dataset_bench_<run_id>`。这类目录只适合一次性数据集正确性实验；MiniCPM5-1B、Qwen3-8B 和 Llama3.2 3B 的重复性能测试必须使用上面的 `shared_kv/<model-key>/`。脚本需要在 run metadata 里记录 shared cache root、stable doc id、`cache_hit` 和 `cache_status`；若第二次跑同一模型/设备/context/token span 仍显示 `cache_status=built`，应先查为什么没有命中缓存，不要继续把该轮当正式性能数据。

`--force-cache-build` 只用于模型权重、KV layout、RoPE metadata 或 token span 确认改变后的主动重建；正常补测、重复测、换 budget 或换 `cacheblend`/`epic` ratio 都不应打开它。改变 budget/ratio 只影响 `/v1/chat/completions` 内的 sparse recompute/scoring，不改变 `/v1/prefill/text` 的持久 text cache。

为了避免连续构建不同 context 时污染 LLM 请求状态，prefill benchmark 可以按 context 重启 `pic_server`；但服务启动和 cache-hit 检查不计入 `prefill_latency_s`。正式计时仍只取 `/v1/chat/completions max_tokens=0` 的独立请求耗时。

Jetson 远端 MNN 仓库：

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

Jetson 默认 artifact：

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

注意：当前 `run_mnn_pic_dataset_bench.sh` 的默认构建路径仍是 Jetson CUDA。Orange Pi 5 Plus / Rhino Pi-X1 的设备信息属于本 Skill，并记录在上面的“固定设备”段；但用这个脚本跑 OpenCL 设备时，不要直接沿用默认构建步骤。先按目标设备单独构建 OpenCL artifact，然后用 `--skip-build` 配合对应 remote / artifact / config 参数启动服务和 bench。

Orange Pi 5 Plus 常用方式：

```bash
MNN_TARGET_DEVICE=orangepi5plus \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

bash .codex/skills/mnn-pic-benchmark/scripts/run_mnn_pic_dataset_bench.sh \
  --skip-build \
  --remote orangepi@192.168.101.113 \
  --remote-repo /home/orangepi/code/kvshare-edge/impl/MNN \
  --build-dir "$PWD/.cache/build/mnn/orangepi5plus" \
  --install-prefix "$PWD/.cache/output/mnn/artifacts/orangepi5plus" \
  --remote-art-rel .cache/output/mnn/artifacts/orangepi5plus \
  --remote-cuda-lib "" \
  --remote-kv-dir /mnt/ssd/code/.cache/mnn_opencl_pic/mnn_pic_dataset_bench_<run_id> \
  --remote-config /home/orangepi/code/kvshare-edge/impl/MNN/.cache/weight/<opencl-model>/config_opencl_greedy.json \
  --port 18096 \
  --local-port 18096 \
  --run-id <run_id> \
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

Rhino Pi-X1 / Adreno 也按同样原则处理：本机 artifact 使用 `.cache/output/mnn/artifacts/aidlux_adreno_opencl`，远端统一使用 `/mnt/nvme/mnn_pic_opencl` 专用工作目录，启动日志必须确认走 OpenCL/Adreno GPU，不能出现 CUDA backend、CPU fallback 或 OpenCL target unavailable。

Rhino Pi-X1 已按 Linux AArch64/OpenCL 目标接入构建脚本；不要使用 Jetson CUDA 或 Android OnePlus artifact。默认构建：

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Rhino Pi-X1 远端是 Ubuntu/glibc 用户态跑在 Android kernel 上，Adreno OpenCL 通常通过 `/usr/lib/libOpenCL.so` / `/usr/lib/libOpenCL_adreno.so` 暴露，不一定有标准 `/etc/OpenCL/vendors`。启动服务时显式设置 OpenCL library path；必要时加 `LD_PRELOAD=/usr/lib/libOpenCL_adreno.so`：

```bash
REMOTE_WORK=/mnt/nvme/mnn_pic_opencl

bash .codex/skills/mnn-pic-benchmark/scripts/run_mnn_pic_dataset_bench.sh \
  --skip-build \
  --remote aidlux@192.168.101.227 \
  --remote-repo "${REMOTE_WORK}" \
  --install-prefix "$PWD/.cache/output/mnn/artifacts/aidlux_adreno_opencl" \
  --remote-art-rel artifacts/aidlux_adreno_opencl \
  --remote-cuda-lib "" \
  --remote-ld-library-path "${REMOTE_WORK}/artifacts/aidlux_adreno_opencl/lib:/usr/lib:/usr/lib/aarch64-linux-gnu" \
  --remote-server-env "LD_PRELOAD=/usr/lib/libOpenCL_adreno.so" \
  --remote-config "${REMOTE_WORK}/models/pic/<opencl-model>/config_opencl_greedy.json" \
  --remote-kv-dir "${REMOTE_WORK}/cache/mnn_pic_dataset_bench_<run_id>" \
  --remote-log-dir "${REMOTE_WORK}/logs" \
  --line-buffer \
  --port 18096 \
  --local-port 18096 \
  --run-id <run_id> \
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
  --pic-selection-algorithm cacheblend \
  --pic-recompute-ratio 0.20 \
  --pic-recompute-score-layer-idx 1
```

正式 Rhino Pi-X1 OpenCL 性能测试前必须先做 backend sanity：`/dev/kgsl-3d0` 存在、`/sys/class/kgsl/kgsl-3d0/gpu_model` 为 Adreno、服务日志没有 `target unavailable` / CPU fallback，并且 warm 目标 shape 后 `/v1/tune/update_cache` 成功。

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

full-compute baseline 的命名必须严格区分：加速比的 `full-compute baseline` 只能是普通非 PIC 导出模型，用 `.cache/mnn-llm-export/<model>/config*.json` 配 `$ART/bin/llm_bench -n 0` 测出来的 normal LLM full-compute prefill。PIC server 的 `selection_algorithm=full-compute` 不是加速比 baseline；它只作为 PIC/PagedAttention full-compute 参考，用来衡量当前 PagedAttention / PagedCache 路径相对 normal LLM 的实现开销。报告中不要把 PIC server full-compute 简写成 normal full-compute 或 full-compute baseline。

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
normal LLM full-compute prefill baseline  仅此项作为加速比 baseline
PIC full-reuse prefill latency
PIC full-compute prefill latency          仅作为 PagedAttention / PagedCache 实现效率参考
cacheblend 各 ratio prefill latency
normal full-compute / full-reuse speedup
normal full-compute / cacheblend speedup
PIC full-compute / full-reuse speedup     仅作为 PagedAttention / PagedCache 参考
PIC full-compute / cacheblend speedup     仅作为 PagedAttention / PagedCache 参考
```

输出表格的加速比列一律按 `full_compute_latency / algo_latency` 计算，让大于 1 的数表示更快。主加速比必须使用普通 LLM full-compute baseline：`speedup_vs_normal_full_compute = normal_full_compute_s / algo_s`。可额外给出 PIC 参考加速比：`speedup_vs_pic_full_compute_ref = pic_full_compute_s / algo_s`，但它只能说明相对 PIC/PagedAttention full-compute 路径节省了多少，不能替代 normal baseline。不要在主表中输出 `algo_s / full_compute_s` 这种小于 1 的反向倍数并称为 speedup。

推荐表头：

```text
context_tokens, algorithm, ratio, algo_s, normal_full_compute_s, speedup_vs_normal_full_compute, pic_full_compute_s, speedup_vs_pic_full_compute_ref
```

正式 sweep 已下沉到 skill 脚本，不要再依赖 `.cache/` 里的临时脚本。统一入口：

```bash
python .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --model-key minicpm5-1b \
  --devices orangepi,rhino \
  --benchmark-csv benchmark.csv \
  --only-missing-contexts \
  --run-id prefill_minicpm5_missing_$(date +%Y%m%d_%H%M%S)
```

关键约定：

- `--benchmark-csv benchmark.csv --only-missing-contexts` 会直接以 `benchmark.csv` 中 `Llama3.2 1B` 为模板，只重跑目标模型当前缺口 context。
- 脚本固定使用设备约定端口，并在启动前直接杀掉旧端口监听进程；不要换端口规避旧进程。
- `summary.csv` 只写本次真实成功返回的行；失败/超时/500 会写到 `failures.jsonl`，不要把失败行直接合进 `benchmark.csv`。
- OpenCL 默认会先做本次 run 的 warm，再测正式请求；这属于当前实跑流程，不是复用历史 warm 数据。
- 当前内置模型 key：`minicpm5-1b`、`llama3.2-3b`、`qwen3-8b`。

如果只想看缺口，不立刻开跑：

```bash
python .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --model-key qwen3-8b \
  --devices jetson,orangepi,rhino \
  --benchmark-csv benchmark.csv \
  --only-missing-contexts \
  --print-gap-only
```

把成功 sweep 结果增量合并回 `benchmark.csv`：

```bash
python .codex/skills/mnn-pic-benchmark/scripts/merge_prefill_benchmark_csv.py \
  benchmark.csv \
  .cache/latency_budget_20260625/<run_id>/summary.csv
```

报告里如果写“同一份 text cache / suffix”，只表示输入条件对齐；不表示多个 ratio 共享一次 scoring。若脚本做了多 ratio sweep，必须确认每个 ratio 的 HTTP 请求、metadata 和计时都是独立记录。

普通 MNN LLM baseline 用真实普通导出模型目录跑 `llm_bench`，建议用同等 prompt token 长度并设 `-n 0`。`NORMAL_CONFIG` 必须来自 `.cache/mnn-llm-export/<model>/`，不能来自 `.cache/weight/<model>/` 的 PIC/PagedAttention 导出目录：

```bash
"$ART/bin/llm_bench" -m "$NORMAL_CONFIG" -a cuda -p <prompt_tokens> -n 0 -rep 3 -load false -j normal_prefill.json
```

只读取 JSON 中 `results[].type == "prefill"` 的 `prompt_len` 和 `tps`，计算 `prefill_s = prompt_len / tps`；不要把 decode 或 `ttft_est` 加进这张对比表。

## Decode Repair TPOT/TPS Benchmark

Decode repair benchmark 与 prefill-only benchmark 是两张表。正式输出文件使用 `benchmark_decode.csv`，保持和 `benchmark.csv` 类似的精简主表，只包含：

```text
device,device_display,model,backend,frequency_profile,context_tokens,mode,budget,decode_selector,repair_tokens,generated_tokens,decode_latency_s,decode_tpot_ms,decode_tps,benchmark_status
```

`repair_tokens=0` 行就是 normal/no-repair decode baseline；其它 repair 行和 baseline 的对比由同一 `device/context/mode/budget` 下的 `decode_tpot_ms` / `decode_tps` 直接计算，不在主 CSV 里冗余写 baseline 或 overhead 列。`model_config`、runtime、execution mode、unsupported 详细错误、payload/response 等调试信息保留在 run 的 raw JSON / log 中，不进入正式主表。

正式矩阵：

```text
devices:        Jetson AGX Xavier, Orange Pi 5 Plus, Rhino Pi X1
model:          Llama3.2 3B
mode:           epic
budgets:        0.05, 0.10, 0.20
contexts:       512, 1024, 1536, 2048, 2560, 3072
max_tokens:     32
repair_tokens:  0, 1, 2, 3, 4, 5, 6, 7
```

`mode` / `budget` 只描述 prefill sparse recompute，`decode_selector` / `repair_tokens` 只描述 decode repair。`repair_tokens=0` 是 normal/no-repair decode baseline，请求不得启用 `decode_refine`；`repair_tokens>=1` 时才在 `pic_cache.decode_refine` 内设置 `enabled=true`、`selector` 和 `tokens_per_decode_step`。

decode selector 是 decode runtime 的选择器，必须由 decode 侧状态和调度实现，例如 `Llm::selectPicDecodeRepairLogicalIndices()` 及其 runtime state。不得通过修改 prefill sparse recompute 的 `buildExecutionPlan()`、`plan.recomputeLogicalIndices`、`nativeSelectedLocalIndices`、`sparseTokenIds` 或 `recomputeTokenCount` 来实现或冒充 decode selector；这些字段只属于 prefill 的重算计划和本次 prefill metadata。

`lagged_attention_hkvd` 是 decode selector，不是 prefill selection algorithm。若 MNN runtime 尚未真正实现它，正式结果必须写 `benchmark_status=unsupported` 或明确失败，不能静默退化为 `top_hkvd`，也不能把 prefill selected indices 复用后命名为 lagged attention。需要临时做 token-id sparse decode smoke 时，可以显式传 `--decode-selector top_hkvd`，但这类结果不能标成 `lagged_attention_hkvd`。

OpenCL 设备正式计时前必须 warm 每个目标 context/budget/repair shape，并调用 `/v1/tune/update_cache` 写回 MNN OpenCL autotune cache。冷启动 kernel build、LWS tuning、cachefile 生成或首轮 text cache 探测不计入正式 TPOT/TPS。run log 需确认没有 `Cache invalid`、`target unavailable`、`async persistent PIC cache read failed` 或 `ERROR`。

采集脚本入口：

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://127.0.0.1:18096 \
  --output-csv benchmark_decode.csv \
  --append \
  --device jetson \
  --device-display "Jetson AGX Xavier" \
  --backend CUDA \
  --model-name "Llama3.2 3B" \
  --model-config "<remote config path>" \
  --mode epic \
  --budgets 0.05,0.10,0.20 \
  --contexts 512,1024,1536,2048,2560,3072 \
  --repair-tokens 0,1,2,3,4,5,6,7 \
  --decode-selector lagged_attention_hkvd \
  --attention-layer-idx 1 \
  --max-tokens 32 \
  --repeats 3 \
  --warm-repeats 1
```

设备参数替换：

```text
Jetson:    --device jetson   --device-display "Jetson AGX Xavier" --backend CUDA
OrangePi:  --device orangepi --device-display "Orange Pi 5 Plus"  --backend OpenCL
Rhino:     --device rhino    --device-display "Rhino Pi X1"       --backend OpenCL
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
