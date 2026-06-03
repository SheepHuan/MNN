---
name: mnn-opt-ops
description: 当用户要求优化本 MNN 仓库中的 CPU/CUDA 算子性能、分析算子慢因、在本机修改编译并同步到 Jetson 运行验证，或需要记录 Jetson 设备与 Attention/PagedAttention bench 流程时使用这个 Skill。
metadata:
  short-description: MNN 算子性能优化和 Jetson 验证流程
---

# MNN Opt Ops

本 Skill 用于优化 MNN CPU/CUDA 算子并闭环验证性能。它只记录优化流程和设备信息；构建细节引用 `$mnn-build-artifacts`，单算子 bench 细节引用 `$mnn-ops-bench`。

## 入口约束

1. 从本 MNN 仓库根目录工作。
2. 修改前先运行：

```bash
git status --short
```

3. 不要读取或修改 `schema/private/`、`source/internal/`。
4. 构建产物、日志、临时文件放到 `.cache/`、`output/` 或用户指定本地目录，不要提交。

## 相关 Skills

- 构建、安装、产物检查：读 `.codex/skills/mnn-build-artifacts/SKILL.md`。
- CUDA direct-op 精度/性能测试：读 `.codex/skills/mnn-ops-bench/SKILL.md`。
- 如果优化需要新增 op/schema/backend 注册：再读 `.codex/skills/mnn-add-new-op/SKILL.md`。

## Jetson 设备

Jetson 可免密登录：

```text
jetson@192.168.101.192
```

远端 MNN 仓库根目录：

```text
/home/jetson/code/kvshare-edge/impl/MNN
```

远端 `.cache` 工作目录：

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache
```

## 本机修改与交叉编译

优先在本机完成源码修改和编译。若需要快速发现普通 CPU/C++ 语法问题，先构建本机 `run_test.out`；若目标是 Jetson CUDA 性能，使用本机 `nvcc` + AArch64 GNU Linux toolchain + Jetson CUDA target sysroot 交叉编译 Jetson 产物，然后只把产物推到 Jetson 执行。

AArch64 GNU toolchain 自带 glibc/libstdc++ sysroot，但不带 CUDA headers/libs。CUDA cross 编译还需要 Jetson 的 CUDA target sysroot，例如：

```text
/usr/local/cuda-12.2/targets/aarch64-linux
```

首次准备本机 CUDA target sysroot：

```bash
mkdir -p .cache/sysroots/jetson_cuda/targets
rsync -a --delete \
  jetson@192.168.101.192:/usr/local/cuda-12.2/targets/aarch64-linux/ \
  .cache/sysroots/jetson_cuda/targets/aarch64-linux/
```

本机交叉编译 Jetson CUDA `run_test.out`：

```bash
CUDA_TOOLKIT_ROOT="$PWD/.cache/sysroots/jetson_cuda" \
CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda" \
JOBS=8 CUDA_ARCHS=72 ENABLE_CROSS_CUDA=ON \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS="-DMNN_BUILD_TEST=ON -DCUDA_HOST_COMPILER=$PWD/.cache/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++" \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

期望配置日志包含：

```text
Native Jetson build: 0
CUDA support: ON
CUDA architectures: 7.2
Found CUDA: .../.cache/sysroots/jetson_cuda
Enabling CUDA support (... archs: sm_72)
```

推送交叉编译产物到 Jetson：

```bash
rsync -a --delete \
  .cache/output/mnn/artifacts/jetson_cross_cuda/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

LLM / PIC server 产物也走同一个 artifact root；`pic_server` 应出现在：

```text
.cache/output/mnn/artifacts/jetson_cross_cuda/bin/pic_server
```

Jetson 原生构建只作为 fallback：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  BUILD_DIR="$PWD/.cache/build/mnn/jetson_cuda" \
  INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson" \
  JOBS=6 CUDA_ARCHS=72 CMAKE_ARGS="-DMNN_BUILD_TEST=ON" \
  BUILD_TARGET=run_test.out INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh'
```

## 同步到 Jetson

默认只同步交叉编译产物。如果确实需要同步源码改动到远端同路径仓库，避免同步 `.cache/`、`output/`、构建下载目录：

```bash
rsync -a --delete \
  --exclude .git \
  --exclude .cache \
  --exclude output \
  --exclude 3rd_party/cutlass \
  ./ jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/
```

如果远端有用户未保存改动，先查看远端 `git status --short`，不要覆盖。

## Attention/PagedAttention Bench

远端运行普通 Attention perf：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  export ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  mkdir -p .cache/bench_ops/cross_cuda_latest && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  "$ART/bin/run_test.out" bench_ops/cuda/perf/Attention 2 1 2>&1 | tee .cache/bench_ops/cross_cuda_latest/attention_perf.log'
```

远端运行 PagedAttention perf：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  export ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  mkdir -p .cache/bench_ops/cross_cuda_latest && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  "$ART/bin/run_test.out" bench_ops/cuda/perf/PagedAttention 2 1 2>&1 | tee .cache/bench_ops/cross_cuda_latest/paged_attention_perf.log'
```

远端运行 PagedAttention 精度对比：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  export ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  mkdir -p .cache/bench_ops/cross_cuda_latest && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" && \
  "$ART/bin/run_test.out" bench_ops/cuda/accuracy/PagedAttention/CompareAttention 2 1 2>&1 | tee .cache/bench_ops/cross_cuda_latest/paged_vs_attention_accuracy.log'
```

汇总日志：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  python3 .codex/skills/mnn-ops-bench/scripts/summarize_attention_logs.py .cache/bench_ops/cross_cuda_latest'
```

## PagedAttention / PIC 优化约束

优化 PagedAttention、PIC full-reuse、cacheblend、epic 或 kvshare 时，先区分执行语义，再选择专用路径；不要把所有模式都塞进一个通用慢 kernel 或 CPU fallback。

框线草图：

```text
+==================================================================================+
| ChatCompletionBatchRequest                                                       |
| batch must be homogeneous: all full-compute OR all PIC-cache requests             |
+===========================+==========================+===========================+
                            |                          |
                            v                          v

FULL-COMPUTE PREFILL
+--------------------------+
| prompt = prelude+PIC+suffix|
+-------------+------------+
              |
              v
+--------------------------+
| Q/K/V projection kernels  |
| produce current K/V       |
+-------------+------------+
              |
              v
+--------------------------+
| copy/write current K/V    |
| into PagedCache CL Buffer |
+-------------+------------+
              |
              v
+--------------------------+
| pack/read from PagedCache |
| tiled Attention prefill   |
+-------------+------------+
              |
              v
+--------------------------+        +--------------------------+
| suffix/output logits      +------->| normal decode            |
| no disk PIC KV involved   |        | PagedAttention reads     |
+--------------------------+        | existing PagedCache KV   |
                                    +--------------------------+

FULL-REUSE PIC PREFILL
+--------------------------+
| prompt = prelude +        |
| {{pic_cache}} + suffix    |
+-------------+------------+
              |
              v
+--------------------------+        +--------------------------+
| per-layer disk .k/.v      |------->| ordered async loader     |
| cached PIC KV             |        | following layer queue    |
+-------------+------------+        +--------------------------+
              |
              v
+--------------------------+
| mapped source CL buffer   |
| CPU reads into host ptr   |
+-------------+------------+
              |
              v
+--------------------------+
| PICPageAttention series   |
| hydrate path: slot write  |
| + RoPE in GPU             |
+-------------+------------+
              |
              v
+--------------------------+        +--------------------------+
| PICPageAttention          +------->| normal decode            |
| suffix prefill reuses KV  |        | PagedAttention reads     |
+--------------------------+        | existing PagedCache KV   |
                                    +--------------------------+

CACHEBLEND / EPIC / KVSHARE PIC PREFILL
+--------------------------+
| prompt = prelude +        |
| {{pic_cache}} + suffix    |
+-------------+------------+
              |
              v
+--------------------------+
| ScorePageAttention /      |
| planner branch            |
| score layer only          |
+-------------+------------+
              |
              v
+--------------------------+
| recompute logical indices |
| selected PIC tokens only  |
+-------------+------------+
              |
              v
+--------------------------+        +--------------------------+
| per-layer disk .k/.v      |------->| ordered async loader     |
| bulk reused PIC KV        |        | following layer queue    |
+-------------+------------+        +--------------------------+
              |
              v
+--------------------------+
| mapped source CL buffer   |
| CPU reads into host ptr   |
+-------------+------------+
              |
              v
+--------------------------+
| PICPageAttention series   |
| hydrate reused KV + RoPE  |
| into PagedCache CL Buffer |
+-------------+------------+
              |
              v
+--------------------------+
| sparse recompute selected |
| PIC K/V in PagedCache     |
+-------------+------------+
              |
              v
+--------------------------+        +--------------------------+
| PICPageAttention          +------->| normal decode            |
| suffix prefill uses mixed |        | PagedAttention reads     |
| reused + recomputed KV    |        | existing PagedCache KV   |
+--------------------------+        +--------------------------+
```

统一内存 / OpenCL Buffer 约束：

- PagedCache 本体就是 backend 侧的 device buffer / CL Buffer。对 OpenCL UMA 设备，优化目标是直接按 slot 写入这个 CL Buffer。
- 禁止为了 hydrate 外部 PIC KV，把完整 PagedCache `enqueueReadBuffer` 到 CPU、在 CPU 上 patch 后再整块 `enqueueWriteBuffer` 回 GPU。这会破坏 UMA/CL Buffer 路径，也会把每层 cache 访问放大成全 cache 往返。
- 磁盘 KV 读取只允许进入小的 mapped source CL buffer。OpenCL UMA 设备上优先用 host-visible / mapped buffer，CPU 直接拿 host pointer，把 `.k/.v` 文件内容读进这个 mapped pointer；unmap 后由 GPU kernel 消费。避免 `std::vector` 暂存后再 `enqueueWriteBuffer` 的二次拷贝。
- 如果某个 OpenCL runtime 不能高效 map host-visible buffer，才允许 fallback 到普通 source buffer + `enqueueWriteBuffer`，并在性能报告中明确标注这个 fallback。
- PIC token RoPE 必须融合进 `PICPageAttention` 系列 GPU kernel 中，包括 full-reuse hydrate、cacheblend/epic/kvshare reused-KV hydrate、sparse recompute 这些变体。磁盘里的 key cache 保持 `canonical_no_rope`，CPU 不做逐 token/head/dim 的 RoPE sin/cos 计算，也不要增加独立 RoPE 预处理 pass。
- 对 full-reuse 这类纯 hydrate 路径，目标数据流是：`disk .k/.v -> mapped source CL buffer host pointer -> PICPageAttention hydrate path -> PagedCache CL Buffer -> PICPageAttention suffix prefill`。
- OpenCL buffer 实现应使用专门的 PIC hydrate kernel（例如 `pic_page_attention_hydrate_kv`）承接 `mapped source CL buffer -> PagedCache CL Buffer + GPU RoPE`，不要退回到 CPU patch PagedCache。
- `cacheblend` / `epic` / `kvshare` 仍然会复用大量 PIC KV；这些复用 KV 同样来自磁盘 `.k/.v`，也必须走 `mapped source CL buffer -> PICPageAttention hydrate path -> PagedCache CL Buffer`，并在 PICPageAttention 系列 kernel 中按当前 logical slot 重新施加 RoPE。
- 对按层加载外部 KV，允许 CPU 后台线程按 layer 顺序持续异步预读后续层 `.k/.v` 文件；当前层 attention 计算时，loader 自动推进后面层的 KV 队列。预取窗口大小只是实现参数，不能把语义限制成只提前读一层或两层。预取只缓存 source KV 文件内容或可上传的 source buffer，不缓存完整 PagedCache 镜像。
- 异步预取不能改变语义：每层真正可见的 KV 仍以当前 request 的 `slot_table`、`logicalStart`、`sourceTokenOffset` 和 `sourceTokenCount` 为准。
- 正常 decode 阶段不再重新 hydrate 或重算已经存在的旧 KV；decode 直接使用当前 PagedCache 中已施加 RoPE 的 K/V，走普通 PagedAttention。

PagedAttention 专用分支约定：

- `full-compute` prefill：不读取磁盘 PIC KV。应使用与普通 Attention 对齐的 tiled/prefill 计算路径，但仍然先写入 PagedCache，再从 PagedCache/packed PagedCache 读出计算输出；不能绕过 PagedCache 直接用原始 K/V “作假”。
- `full-reuse`：特殊 PIC 计算路径，不重算 PIC token。应启动专门的 PIC hydrate + PICPageAttention 路径，先把磁盘 PIC KV 按 slot 注入 PagedCache，suffix token 作为真实 query 从同一份 PagedCache 读取 PIC KV。
- `cacheblend`：参考 `/root/code/kvshare-edge/impl/pic_server` 的语义，不应在通用 PagedAttention 中隐式完成全部工作。应拆成两个明确阶段：
  - `CacheblendScorePageAttention` / score 分支：在指定 score layer 上计算或读取评分所需的 full-reference/cached delta，产出需要重算的 PIC logical indices。
  - `PICPageAttention` / sparse recompute 分支：后续层从磁盘 `.k/.v` hydrate 大量复用 KV 到 PagedCache，只对选中的 PIC token 重新计算 K/V，其余 PIC token 继续复用 PagedCache 中的 cached KV。
- `epic` / `kvshare` 同理应有各自明确的 scoring/planning 分支，再把 logical indices 传给 PIC sparse recompute，不要让每个 layer 重复做规划。
- `cacheblend` / `epic` / `kvshare` 的 sparse recompute 只发生在 PIC prefill/reuse 阶段；进入 decode 后，所有模式都使用普通 PagedAttention 读现有 PagedCache，不对旧 PIC KV 做二次 RoPE 或二次重算。
- batch 请求必须保持同质：要么全是 full-compute，要么全带 PIC cache；混合 PIC/no-PIC 拒绝。单请求也按 batch=1 走同一套 batch scheduler 和 PagedAttention 逻辑。

瓶颈定位要求：

- 报告 cacheblend/kvshare 性能时必须拆开：`text_cache_build`、`full_reference/scoring`、`disk_read/hydrate`、`sparse_recompute`、`suffix_generate/TTFT`。
- 如果 full-reuse 慢，优先检查 disk read + hydrate + PagedCache CL Buffer 写入；如果 cacheblend 比 full-reuse 多出大段延迟，优先检查 full-reference/scoring 是否每次请求重跑和是否落盘。
- `precision_recovery.execution_mode` 必须确认是 `native-full-reuse`、`native-cacheblend-sparse-recompute`、`native-epic-sparse-recompute` 或 `native-kvshare-sparse-recompute`，不能只看 HTTP 成功。
- 对 OpenCL UMA 优化，验证时至少对比优化前后的 full-reuse latency；full-reuse 是隔离 PagedCache hydrate/cache 访问成本的最直接用例。

Full-reuse / TTFT 经验：

- `full-reuse` 不是“默认重算 PIC 最后一个 token”。正确语义是：hydrate 全量 PIC KV 到 PagedCache 后，suffix token 作为当前上下文里的真实 query 去 attend 已 hydrate 的 PIC KV；`precision_recovery.recompute_token_count` 应为 0，metadata 应体现 `reuse_token_count=pic_token_count`。只有 `cacheblend` / `epic` / `kvshare` / explicit 计划选中的 PIC token 才 sparse recompute。
- 统计 TTFT 时，口径是“模型拿到请求到输出第一个 token”。不应把输出第一个 token 之后、为了准备下一 token logits 而执行的 decode forward 算进去。高层一次性 `generate(input, max_tokens)` 可以在达到 `max_tokens` 后跳过 next-logits forward；底层可连续调用的 `generate(1)` 仍应保留 next-logits forward，避免破坏连续 decode。
- `max_tokens=0` 不能作为 chat prefill-only benchmark；当前 chat completion 会拒绝 `max_tokens <= 0`。需要 prefill-only 拆分时，应加专门 timing 或使用 server 内部阶段计时，不要把这个快速错误响应当作 prefill 延迟。
- OpenCL PagedAttention profiling 可用 `MNN_PAGED_ATTENTION_PROFILE=1` 打开。profile 日志应至少区分 `hydrate`、`fast_prefill`、`row`、`generic`。注意 profile 中会 `queue.finish()`，只用于定位瓶颈，不作为最终性能数。
- 如果 profile 中 full-reuse 出现大量 `row` / `generic` 且 `sparse=1`，先检查是否错误触发了 PIC sparse recompute。一次错误的 1-token sparse recompute 会过全部层，KV 长度接近 PIC+prelude，延迟会远高于 hydrate 本身。
- 如果 `max_tokens=1` 后还出现一轮 `row` / `generic`、`sparse=0` 的 decode forward，通常是在输出第一个 token 后又为下一 token 准备 logits；它不属于 TTFT，应通过高层生成参数跳过。
- 2026-06-03 Orange Pi 5 Plus / GLM Edge 4B / OpenCL 参考数据：594 PIC tokens + 15 suffix tokens + `max_tokens=1`，修正后 `native-full-reuse`、`recompute_token_count=0`，热态约 3.0s，首轮偏冷态约 4.6s。此前约 13s 的主要原因是错误默认重算最后一个 PIC token，并且 `max_tokens=1` 后额外跑了 next-logits forward。
- 同一组 profile 中，错误版本约可见：40 层 hydrate 约 0.46s，suffix fast-prefill 约 0.52s，错误 sparse recompute generic 约 3.44s，额外 decode generic 约 1.44s。修正后 profile 应没有 PIC sparse recompute，剩余延迟主要来自 KV hydrate、15 个 suffix token 的真实模型计算，以及 HTTP/调度/采样开销。

## PIC Server Smoke

需要验证 MNN 自维护的独立 PIC server 时，仍然在本机交叉编译产物并推到 Jetson，再从远端 artifact root 启动，不依赖 `mls`：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  REPO=$PWD && \
  ART=$REPO/.cache/output/mnn/artifacts/jetson_cross_cuda && \
  MODEL=$REPO/.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct && \
  CONFIG=$MODEL/config_cuda_greedy.json && \
  KV_DIR=$REPO/.cache/kvshare/pic_server_smoke && \
  LD_LIBRARY_PATH="$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" \
    "$ART/bin/pic_server" --config "$CONFIG" --host 127.0.0.1 --port 18091 \
    --kv-cache-dir "$KV_DIR" --model llama-pic'
```

另一个远端命令发文档 prefill：

```bash
ssh jetson@192.168.101.192 'curl -s -X POST http://127.0.0.1:18091/v1/prefill/text \
  -H "Content-Type: application/json" \
  -d "{\"id\":\"smoke_doc\",\"type\":\"text\",\"content\":\"Jetson PIC cache smoke text.\",\"force\":true}"'
```

期望响应包含 `format=kvshare-prefix-cache-meta-v1`、`cache_status=built`、非零 `token_count`、`layer_count`，并在 `.cache/kvshare/pic_server_smoke/objects/<backend>/<cache_name>/layers/` 下生成每层分离的 `.k` / `.v` raw KV 文件和同层 `.json` shape sidecar。确认 `kv_layout.kv_heads`、`kv_layout.head_dim`、`key_shape`、`value_shape` 来自真实 sidecar，而不是 0 或猜测值。

继续验证 PIC 复用与重算模式：

```bash
ssh jetson@192.168.101.192 'curl -s -X POST http://127.0.0.1:18091/v1/kv/pic_caches \
  -H "Content-Type: application/json" \
  -d "{\"id\":\"smoke_pic\",\"text_cache_refs\":[{\"id\":\"smoke_doc\"}],\"selection_algorithm\":\"full-reuse\"}"'

ssh jetson@192.168.101.192 'curl -s -X POST http://127.0.0.1:18091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"llama-pic\",\"messages\":[{\"role\":\"user\",\"content\":\"{{pic_cache}}\\nQuestion: what text was cached?\"}],\"max_tokens\":8,\"temperature\":0,\"pic_cache\":{\"id\":\"smoke_pic\",\"text_cache_refs\":[{\"id\":\"smoke_doc\"}],\"selection_algorithm\":\"full-reuse\"}}"'
```

`precision_recovery.execution_mode` 是判断实际执行路径的关键：`native-full-reuse` 表示磁盘 PIC KV 已写入 paged slots，`native-full-compute` 表示完整重算，`native-epic-sparse-recompute` 表示 EPIC 头部 token sparse recompute + 其余 PIC token 复用，`native-cacheblend-sparse-recompute` 表示按 score layer value delta 选 top-ratio token，`native-kvshare-sparse-recompute` 表示按 MNN C++ K/V delta influence proxy 选 top-ratio token。`epic/cacheblend/kvshare` 应有 `metadata.native_sparse_recompute_scope=python_prefill_layer_plan`；当 `pic_recompute_score_layer_idx > 0` 时，`metadata.pre_score_kv_source=full_prompt_reference` 和 `metadata.pre_score_compute_layers=<score_layer_idx>` 表示 score layer 前保持 token 正常计算语义。从 score layer 开始，选中的 PIC token 和 suffix/非复用 KV token 参与计算，其他 PIC 位置直接复用磁盘 KV。

## 优化判断

报告结果时至少说明：

- 修改了 CPU 还是 CUDA 路径，核心瓶颈是什么。
- 精度是否通过。
- 性能表包含 `stage | model | ctx | qH | kvH | D | op | latency_ms | Attention/PagedAttention`，其中 `Attention/PagedAttention > 1` 表示普通 `Attention` 比 `PagedAttention` 慢。
- 与优化前的 Jetson 数据相比，PagedAttention 的 prefill/decode 延迟和 `Attention/PagedAttention` 倍数是否改善。
