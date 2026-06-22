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

## Decode Repair Tune Utilities

PIC decode repair 做正式 TPOT 前，先把算子级 tune 和 server 启动配置 warm 分开：

- 算子级 CUDA event tune：用 `.codex/skills/mnn-opt-ops/scripts/run_remote_weight_only_conv_tune.sh` 在 Jetson 上跑 `bench_ops/cuda/perf/WeightOnlyConv`，筛 rows=2..8 的 weight-only / fused kernel shape policy。
- Server warm + cache 固化：用 `.codex/skills/mnn-opt-ops/scripts/run_pic_decode_repair_config_tune.sh` 枚举候选启动配置；每个配置会在独立 workdir 启动 `pic_server`，再调用 `.codex/skills/mnn-opt-ops/scripts/run_pic_decode_repair_tune_warm.sh` 发 `tpd=0..4` warm 请求和 `/v1/tune/update_cache`，把 `tmp/mnn_cachefile.bin` 固化到该候选目录。正式 TPOT 需要重启同一配置并复用该 cache 文件，最后比较各配置的热态 TPOT / smoke 结果。

聚焦 rows=4/5 的 MLP/Linear direct-op tune：

```bash
MNN_TUNE_CASES="hidden_to_inter inter_to_hidden hidden_to_gateup_concat" \
MNN_TUNE_ROWS=4-5 \
MNN_TUNE_WARMUP=20 \
MNN_TUNE_REPEAT=80 \
MNN_TUNE_MEMORY=2 \
bash .codex/skills/mnn-opt-ops/scripts/run_remote_weight_only_conv_tune.sh
```

汇总 `WeightOnlyConv` log 并计算 gate/up concat 理论收益：

```bash
python3 .codex/skills/mnn-opt-ops/scripts/summarize_weight_only_conv_tune.py \
  .cache/bench_ops/decode_repair_tune/weight_only_rows4_5_20260622_134205.log
```

`tpd=3/4` 到 50ms 目标通常还缺约 5ms/token；以 16 层 Llama-3.2-1B 粗略折算，rows=4/5 MLP/Linear 候选需要接近 0.31ms/layer 的真实节省才值得作为主线进入 strict TPOT。汇总器会输出 `decision`：`strong_candidate_for_strict_tpot` 可以进长测，`partial_candidate_needs_end_to_end_check` 需要结合风险判断，`cleanup_only_do_not_use_as_main_tpot_candidate` 只作为小 cleanup。

`WeightOnlyConv` direct-op 必须使用 `run_test.out ... 2 <precision> 1 x 2`，即 `memory=2 (Memory_Low)`；否则 CUDA 会创建普通 Conv/CUTLASS execution，而不是 `ConvFpAIntBExecution`，rows policy、GEMV profile 或任何 weight-only 策略验证都不会生效。

评估 rows=4/5 新 weight-only kernel family 前，先用 dense FP16 GEMM floor 判断普通 GEMM 路线是否有足够理论空间：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  CUDA_LIB=/usr/local/cuda-12.2/targets/aarch64-linux/lib && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:$CUDA_LIB:${LD_LIBRARY_PATH:-}" && \
  env MNN_BENCH_WEIGHT_ONLY_ROWS=4-5 MNN_BENCH_MLP_GEMM_WARMUP=20 MNN_BENCH_MLP_GEMM_REPEAT=80 \
  "$ART/bin/run_test.out" bench_ops/cuda/perf/DecodeRepairMlpGemmFloor 2 2 1 x 2'
```

若 FP16 GEMM floor 仍慢于当前 `PicDecodeMlp` weight-only chain，则不要把主线放到 generic dense GEMM、cuBLAS batched 或继续调现有 GEMV/CUTLASS 阈值上；新的 rows=4/5 kernel 必须是 INT4-native 小批量 weight-only kernel，或真正融合 `SwiGLU + down` 的输入读取/累加路径。

例外是已验证的 real-layout static-dequant cuBLAS shape policy：`MNN_CUDA_PIC_INT4_ROWS45_CUBLAS` 默认 `all`，`0` 禁用，`down` 只启用 `8192->2048` down。该分支只在 PIC decode repair sparse、fp16、static dequant cache 已存在、无 runtime dequant 时命中 rows4/5 的 `2048->8192` gate/up 和 `8192->2048` down；它复用 MNN 的 `mDequantFilter [ocp, icp]`，不是上面 synthetic floor 的 layout。

rows4/5 cuBLAS policy 的最小验证命令：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  CUDA_LIB=/usr/local/cuda-12.2/targets/aarch64-linux/lib && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:$CUDA_LIB:${LD_LIBRARY_PATH:-}" && \
  "$ART/bin/run_test.out" bench_ops/cuda/accuracy/Rows45Cublas 2 2 1 x 2 && \
  env MNN_BENCH_WEIGHT_ONLY_ROWS=4-5 MNN_BENCH_MLP_WARMUP=10 MNN_BENCH_MLP_REPEAT=30 \
  "$ART/bin/run_test.out" bench_ops/cuda/perf/PicDecodeMlp 2 2 1 x 2 && \
  env MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=0 MNN_BENCH_WEIGHT_ONLY_ROWS=4-5 MNN_BENCH_MLP_WARMUP=10 MNN_BENCH_MLP_REPEAT=30 \
  "$ART/bin/run_test.out" bench_ops/cuda/perf/PicDecodeMlp 2 2 1 x 2'
```

Jetson 参考结果：accuracy rows4/5 gate/up/down/down_all 全部 `bad=0`；default cuBLAS rows4/5 chain `1.2768/1.2819ms`，禁用回退 chain `1.5966/1.6076ms`，节省约 `0.32ms/layer`，按 16 层约 `5.1ms/token`。这类结果可标记为 `strong_candidate_for_strict_tpot`，但必须再跑 strict `tpd=0..4` 和自然语言 smoke；若端到端回归或 server path 未稳定命中 static dequant cache，默认应收窄为 `down` 或退回 opt-in。

2026-06-22 strict follow-up：在 warm + `/v1/tune/update_cache` 后重启同一 Jetson server，rows4/5 cuBLAS 默认策略跑 1B decode-repair silumul strict `tpd=0..4` 得到 `37.54 / 42.02 / 49.54 / 50.71 / 50.64 ms/token`，`tpd>=1` runtime 均为 `mnn_token_id_sparse_decode`，failures=0。它比上一组 gateup-packed-silu 的 `55.10/54.79ms` 明显改善，但 `tpd=3/4` 仍高于 50ms 约 `0.6-0.7ms/token`，因此目标仍未完成；后续需要再找约 `0.04-0.05ms/layer` 的真实节省，或优化 rows3/4 周边小算子/调度。

同日继续 tune rows4/5 cuBLAS 启动策略：`MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=all/down/0` 的 direct-op chain 分别为 rows4 `1.2747/1.4741/1.5957ms`、rows5 `1.2804/1.4798/1.6019ms`，因此 `all` 仍是默认最优，不能收窄为 `down`。新增 tune-only env：`MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_MATH=keep/default/tensor`、`MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=32f/fast16/16f`、`MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO=default/tensor/<cublas algo id>`；默认保持 `tensor + 32f + CUBLAS_GEMM_DEFAULT_TENSOR_OP`。direct-op 上 `math_keep`、`compute_fast16`、`algo_default` 只有约 `0.0-0.01ms/layer` 波动，`compute_16f` 的 down 变慢。`math_keep` 经过 warm + `/v1/tune/update_cache` 后短 strict `tpd=3/4` 为 `51.76/50.49ms/token`，failures=0 但没有稳定进 50ms，且 `tpd=3` 回退；这些 cuBLAS 参数只保留为诊断/tune 开关，不进入默认主线。

已验证过的负向 rows=4/5 weight-only 分支不要重复作为主线：

- 把 PIC decode repair 的 `int4GemvBatchLimit` 从 3 放宽到 5 会让 rows4/5 MLP chain 变慢。
- 强制 rows4/5 回到现有 `GEMV_FpAInt4B_V14_MB` 并枚举 `OC_PER_BLK=2/3/4/8/16` 也变慢；代表数据为 default rows4/5 chain `1.5962/1.6088ms`，`OC_PER_BLK=2` 退到 `2.1377/2.6379ms`，`OC_PER_BLK=16` 退到 `6.7765/8.3407ms`。
- 相关临时 env 分支 `MNN_CUDA_PIC_INT4_ROWS45_PROTO_OC` 已从 CUDA path 移除；后续 rows4/5 主线必须是新的 INT4-native kernel family，或真正融合 `SwiGLU + down` 的累加路径。

汇总 config tune warm 结果：

```bash
python3 .codex/skills/mnn-opt-ops/scripts/summarize_pic_decode_repair_config_tune.py \
  .cache/decode_repair_config_tune/<run_id> \
  --output-tsv .cache/decode_repair_config_tune/<run_id>/warm_summary.tsv
```

对已启动 PIC server 做 decode-repair warm 和 runtime cache 写回：

```bash
PIC_TUNE_BASE_URL=http://127.0.0.1:18091 \
PIC_TUNE_SELECTION_ALGORITHM=full-reuse \
PIC_TUNE_TPDS="0 1 2 3 4" \
PIC_TUNE_MAX_TOKENS=32 \
bash .codex/skills/mnn-opt-ops/scripts/run_pic_decode_repair_tune_warm.sh
```

`tpd=0` 是 baseline，请求不启用 `decode_refine`；`tpd>=1` 的 warm 请求会把 `decode_refine.enabled=true` 放在 `pic_cache` 内，触发 `mnn_token_id_sparse_decode`。`/v1/tune/update_cache` 的成功响应表示当前已 warm 配置的 runtime cache 已提交；候选搜索是否完成，要看外层配置矩阵的重启后热态 TPOT 和正确性 smoke。

枚举候选启动配置并为每个候选执行 warm/update_cache：

```bash
PIC_CONFIG_TUNE_CONFIG=.cache/weight/AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-silumul/config_cuda_greedy.json \
PIC_CONFIG_TUNE_ENV_DIR=.cache/decode_repair_candidates \
PIC_CONFIG_TUNE_CANDIDATES="baseline candidate_a candidate_b" \
PIC_CONFIG_TUNE_TPDS="0 1 2 3 4" \
PIC_CONFIG_TUNE_MAX_TOKENS=32 \
bash .codex/skills/mnn-opt-ops/scripts/run_pic_decode_repair_config_tune.sh
```

每个候选可以有可选 env 文件：`${PIC_CONFIG_TUNE_ENV_DIR}/<name>.server.env` 放 `MNN_PIC_*`、`MNN_PAGED_ATTENTION_*`、CUDA policy env 等启动参数；`<name>.warm.env` 放 `PIC_TUNE_*` 请求矩阵覆盖。脚本会输出 `summary.tsv` 和 `warm_summary.tsv`，其中 `summary.tsv.cache_file` 指向该候选的 `server_work/tmp/mnn_cachefile.bin`，`warm_summary.tsv` 会检查 `update_cache`、cache 文件、`tpd>=1` 的 `mnn_token_id_sparse_decode` runtime。若 strict TPOT 需要在另一套脚本中执行，必须用同一候选 workdir 和同一组 server env 重启；`warm_summary.tsv` 有 `fail` 的候选不能进入正式计时。

`warm_summary.tsv.cache_policy` 默认为 `auto` 推断：OpenCL/OrangePi 候选缺 `tmp/mnn_cachefile.bin` 是 `fail`，因为正式 OpenCL 测试必须复用 warm 后的 MNN autotune cache；CUDA/Jetson 候选缺该文件只记为 `warn/missing_optional`，因为当前 CUDA decode repair shape 通常不会产生可写 runtime cache 条目。无论 backend，`/v1/tune/update_cache` 必须成功，`tpd>=1` 必须进入 `mnn_token_id_sparse_decode`，否则候选不能进入正式 TPOT。

在 Jetson 上跑同一流程时，从本机通过 SSH 进入远端 MNN 根目录执行：

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  PIC_CONFIG_TUNE_CONFIG=.cache/weight/AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-silumul/config_cuda_greedy.json \
  PIC_CONFIG_TUNE_CANDIDATES="baseline sched_a" \
  PIC_CONFIG_TUNE_TPDS="0 1 2 3 4" \
  PIC_CONFIG_TUNE_MAX_TOKENS=32 \
  bash .codex/skills/mnn-opt-ops/scripts/run_pic_decode_repair_config_tune.sh'
```

当前 PIC server 已禁用 placeholder 分段 tokenization，带 PIC 的 chat 请求必须显式提供 full prompt token span。`run_pic_decode_repair_tune_warm.sh` 默认从 `/v1/kv/pic_caches` 响应读取 `token_ids`，写入 `pic_cache.full_prompt_token_ids`，设置 `pic_token_start=0` / `token_count=<cache tokens>`，并追加 1 个 cache token 作为 suffix 以满足 suffix prefill 要求。若要用真实 fixture，可通过 `PIC_TUNE_FULL_PROMPT_TOKEN_IDS`、`PIC_TUNE_PIC_TOKEN_START`、`PIC_TUNE_PIC_TOKEN_COUNT` 和 `PIC_TUNE_SUFFIX_FROM_CACHE_TOKENS=0` 覆盖。

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
- hydrate kernel 做 RoPE 时，同一 pair 的前半维度线程应一次写出前后两个 key 维度，避免后一半维度重复计算同一组 `sin/cos`。value 仍按全 head_dim 拷贝到 PagedCache。
- 对 full-reuse 这类纯 hydrate 路径，目标数据流是：`disk .k/.v -> mapped source CL buffer host pointer -> PICPageAttention hydrate path -> PagedCache CL Buffer -> PICPageAttention suffix prefill`。
- OpenCL buffer 实现应使用专门的 PIC hydrate kernel（例如 `pic_page_attention_hydrate_kv`）承接 `mapped source CL buffer -> PagedCache CL Buffer + GPU RoPE`，不要退回到 CPU patch PagedCache。
- `cacheblend` / `epic` / `kvshare` 仍然会复用大量 PIC KV；这些复用 KV 同样来自磁盘 `.k/.v`，也必须走 `mapped source CL buffer -> PICPageAttention hydrate path -> PagedCache CL Buffer`，并在 PICPageAttention 系列 kernel 中按当前 logical slot 重新施加 RoPE。
- 对按层加载外部 KV，允许 CPU 后台线程按 layer 顺序持续异步预读后续层 `.k/.v` 文件；当前层 attention 计算时，loader 自动推进后面层的 KV 队列。预取窗口大小只是实现参数，不能把语义限制成只提前读一层或两层。预取只缓存 source KV 文件内容或可上传的 source buffer，不缓存完整 PagedCache 镜像。
- 异步预取不能改变语义：每层真正可见的 KV 仍以当前 request 的 `slot_table`、`logicalStart`、`sourceTokenOffset` 和 `sourceTokenCount` 为准。
- 正常 decode 阶段不再重新 hydrate 或重算已经存在的旧 KV；decode 直接使用当前 PagedCache 中已施加 RoPE 的 K/V，走普通 PagedAttention。

PagedAttention 专用分支约定：

- `full-compute` prefill：不读取磁盘 PIC KV。应使用与普通 Attention 对齐的 tiled/prefill 计算路径，但仍然先写入 PagedCache，再从 PagedCache/packed PagedCache 读出计算输出；不能绕过 PagedCache 直接用原始 K/V “作假”。
- `/v1/prefill/text` 持久化 text cache 时会用 prefix-cache `PendingWrite` 写 `.k/.v`，但这不应强制 attention 退到逐行 `row` kernel；当前 K/V 已先写入 PagedCache，写盘后仍可走 fast/tiled PagedCache prefill。
- `full-reuse`：特殊 PIC 计算路径，不重算 PIC token。应启动专门的 PIC hydrate + PICPageAttention 路径，先把磁盘 PIC KV 按 slot 注入 PagedCache，suffix token 作为真实 query 从同一份 PagedCache 读取 PIC KV。
- `cacheblend`：参考 `/root/code/kvshare-edge/impl/pic_server` 的语义，不应在通用 PagedAttention 中隐式完成全部工作。应拆成两个明确阶段：
  - `CacheblendScorePageAttention` / score 分支：在指定 score layer 上计算或读取评分所需的 full-reference/cached delta，并在 GPU/CL/CUDA 上完成 top-k，产出需要重算的 PIC logical indices。不得把完整 score vector 拷回 CPU 后排序。
  - `PICPageAttention` / sparse recompute 分支：后续层从磁盘 `.k/.v` hydrate 大量复用 KV 到 PagedCache，只对选中的 PIC token 重新计算 K/V，其余 PIC token 继续复用 PagedCache 中的 cached KV。
- cacheblend / kvshare 的磁盘边界必须严格遵守：只有 `/v1/prefill/text` 可以把 text cache K/V 持久写成 `.k/.v`。`/v1/chat/completions` 的 full-reference / scoring 只是请求内的正常计算，不得通过 prefix-cache `PendingWrite`、`setPrefixCacheFile`、scratch `.k/.v` 或 CPU 读盘扫描来实现。
- score layer 之前的 Attention 是普通 full-prompt PagedAttention：prelude + PIC + suffix 正常计算，并把当前请求 K/V 写入 PagedCache；不写 full-reference `.k/.v`，也不从磁盘 hydrate pre-score full-reference KV。
- cacheblend / delta-v 的 score 分支必须读内存态 backend buffer：full-reference 正常 forward 到 score layer 后，reference K/V 留在 PagedCache / device buffer；CUDA/OpenCL score kernel 直接比较 reference K/V 与 cached PIC source/PagedCache 中的 K/V，输出每个 PIC token 的 score，并在设备侧按 `pic_recompute_ratio` 选 top-k token。不应改变 reference 计算、触发重复落盘或重复读盘；如果当前 C++ 调度还需要 CPU vector，最多只拷回最终 top-k logical indices，不拷完整 score vector 做 CPU 排序。
- CPU / CUDA / OpenCL 三套 PagedAttention backend 的语义能力要对齐。CPU 也必须支持 full-compute PagedCache 读写、full-reuse 从持久 PIC cache 源 hydrate + canonical_no_rope key 重新 RoPE、sparse recompute 按 logical slot 写回、任意 `score_layer_idx` 的 cacheblend/kvshare scoring。CPU 上 delta-v score 和 top-k 可以在 host 执行，但仍必须读当前请求 PagedCache / slot table 与持久 cached `.v`，并通过 `PagedKVMeta::setCacheBlendScoringResult` 返回 local indices；不得用 chat/scoring scratch `.k/.v`、临时 tensor 对比或绕过 PagedCache 来冒充对齐。
- CacheBlend 的执行顺序必须保持自然语义：score layer 先按完整 token 长度正常计算 Q/K/V 和该层 PagedAttention，随后基于本次 request 的 reference V 与 cached PIC V 做 Delta-V 评分和设备侧 top-k；选出 token 后才进入 PIC sparse recompute，把选中 logical slot 的 K/V 覆盖回 PagedCache。不要在 score layer attention 之前先切 Q，也不要把未选中的 PIC token 当 query 重算。
- 当前实现可以先用 score-layer-only Module 完成“跑到 score layer 后立即评分，然后 reset 进入 sparse recompute”的两阶段路径。下一步更优目标是 fused score-to-sparse 路径：score layer 的 reference Q/K/V 和 PagedCache 保留在同一请求内，设备侧 top-k 后直接 compact/slice selected PIC query，覆盖 selected logical slots，并从 score layer 之后继续 forward，避免 reset 后重复跑 pre-score 层。
- CacheBlend score 的数学定义必须在 metadata 和报告中写清楚：若实现是 `mean(abs(V_ref - V_cached))`，标注为 `layer_value_delta_mean_abs`；若切到二次项误差，标注为 `layer_value_delta_mean_square` 或等价名称。不要把 mean-abs 实现说成二次项 Delta-V。
- 多 ratio 测试不共享 scoring 结果：`1%/5%/10%/20%/30%` 可以共享同一份 text cache、模型和 suffix，但每个 ratio 都必须是一次独立 `/v1/chat/completions` 请求，独立执行 full-reference / scoring / top-k 选择。不要跨 ratio 或跨请求缓存 score vector、排序结果或 recompute logical indices；每个 ratio 的 latency 都要包含本次请求自己的 scoring 成本。
- 如果 CUDA/OpenCL scoring + top-k 分支尚未实现，报告为 unsupported/fallback；不能把 `std::vector` + `readBinaryFile(.k/.v)` 的 CPU delta 扫描标成 `native-cacheblend-sparse-recompute`。
- `epic` / `kvshare` 同理应有各自明确的 scoring/planning 分支，再把 logical indices 传给 PIC sparse recompute，不要让每个 layer 重复做规划。
- `cacheblend` / `epic` / `kvshare` 的 sparse recompute 只发生在 PIC prefill/reuse 阶段；进入 decode 后，所有模式都使用普通 PagedAttention 读现有 PagedCache，不对旧 PIC KV 做二次 RoPE 或二次重算。
- batch 请求必须保持同质：要么全是 full-compute，要么全带 PIC cache；混合 PIC/no-PIC 拒绝。单请求也按 batch=1 走同一套 batch scheduler 和 PagedAttention 逻辑。

自然语言正确性门槛：

- 性能优化后必须跑 `max_tokens>0` 的功能 smoke，确认 full-reuse、cacheblend、epic 输出的是正常自然语言或可解析答案，而不是重复标点、空串或乱码。正式延迟表仍只用 `max_tokens=0`，这个 smoke 只验证 hydrate / sparse recompute 后 decode 能正确读取 PagedCache。
- 带 `pic_cache` 的 chat smoke 必须显式包含 `{{pic_cache}}` 或 `pic_cache.placeholder` 指定的自定义 placeholder；无 placeholder 的 legacy 隐式 PIC 插入路径已删除，不再作为兼容入口或测试路径。
- 正确性 smoke 的响应解析要覆盖 OpenAI 形状和 MNN PIC server 包装形状，尤其是 `choices[0].message.content` 与 `data[0].choices[0].message.content`。不能因为解析器漏字段而把正常输出误报失败。
- sparse 模式若返回 HTTP 200 但输出重复标点或无意义文本，应优先排查：score pass 后 reset 是否破坏 position/history，selected logical slot 是否覆盖到正确 physical slot，sparse recompute 后 `logical_length` / `slotTable` 是否仍包含全量 PIC + suffix 上下文，以及 decode 是否直接读混合后的 PagedCache。

瓶颈定位要求：

- 报告 cacheblend/kvshare 性能时必须拆开：`text_cache_build`、`full_reference/scoring`、`disk_read/hydrate`、`sparse_recompute`、`suffix_prefill`。当前优化分析暂不比较 decode 性能。
- 如果 full-reuse 慢，优先检查 disk read + hydrate + PagedCache CL Buffer 写入；如果 cacheblend 比 full-reuse 多出大段延迟，优先检查 full-reference/scoring 是否错误走了 prefix-cache 写盘、scratch `.k/.v` 读回或 CPU `std::vector` 扫描。
- `precision_recovery.execution_mode` 必须确认是 `native-full-reuse`、`native-cacheblend-sparse-recompute`、`native-epic-sparse-recompute` 或 `native-kvshare-sparse-recompute`，不能只看 HTTP 成功。
- 对 OpenCL UMA 优化，验证时至少对比优化前后的 full-reuse latency；full-reuse 是隔离 PagedCache hydrate/cache 访问成本的最直接用例。

Full-compute / CUDA PagedAttention 经验：

- 如果普通 MNN LLM prefill 明显快于 PIC `selection_algorithm=full-compute`，优先检查 CUDA `PagedAttention` 是否因为 float causal mask 掉进逐 query row 的 generic kernel。full-compute 长 prefill 通常带 mask；fast/tiled path 必须支持 mask，不能只在 `!useMask` 时启用。
- CUDA `PagedAttention` 的优化边界是：仍然先把当前 K/V 写入 PagedCache，再通过 `slotTable + PagedCache key/value` 读回做 QK/QKV；不要为了追普通 Attention 性能绕过 PagedCache 直接吃原始 K/V。
- mask 在 CUDA PagedAttention 中按 float additive mask 读取。fp16 Q/K/V 时也不能把 float mask 指针转成 half；否则既慢又可能在 fallback path 上读错 mask。
- 长 prefill 应像普通 CUDA Attention 一样按 query piece 分片，例如 256/1024 token 阈值，而不是一次性分配完整 `B*H*Q*K` 的 QK/Softmax 临时缓冲。对 2k+ prompt，这能显著降低显存压力和 cache 抖动。
- Direct-op 验证要覆盖带 mask 的 PagedAttention prefill。`bench_ops/cuda/perf/PagedAttention/Prefill` 和 `bench_ops/cuda/accuracy/PagedAttention/CompareAttention` 中的 PagedAttention prefill 应传 causal mask，否则测试不到线上 full-compute 的慢路径。
- 2026-06-03 Jetson / Llama-3.2-3B / CUDA 旧参考数据：修正 mask fast path 和 query split 后，direct-op `PagedAttention/Prefill` 与普通 `Attention/Prefill` 同量级；ctx2048 的 3B 形状约 `PagedAttention 480ms` vs `Attention 550ms`。旧端到端 PIC full-compute 数据使用 2312 PIC tokens + 39 prelude + 18 suffix + `max_tokens=1`，从约 59.5s 降到首轮约 26.2s、热态约 21.8s；同长度普通 LLM `llm_bench -p 2369 -n 1 -rep 3` 约 94.3 tok/s，即约 25.1s。当前 prefill-only 对比不要沿用这组含首 token 的口径。

Full-reuse / prefill-only 经验：

- `full-reuse` 不是“默认重算 PIC 最后一个 token”。正确语义是：hydrate 全量 PIC KV 到 PagedCache 后，suffix token 作为当前上下文里的真实 query 去 attend 已 hydrate 的 PIC KV；`precision_recovery.recompute_token_count` 应为 0，metadata 应体现 `reuse_token_count=pic_token_count`。只有 `cacheblend` / `epic` / `kvshare` / explicit 计划选中的 PIC token 才 sparse recompute。
- 当前 cacheblend / kvshare / full-reuse / full-compute 延迟分析只测 prefill。PIC server 请求应设置 `max_tokens=0`，完成 prelude / hydrate / sparse recompute / suffix prefill 后直接返回；不要把 decode token、采样或 next-logits forward 算入这组结果。
- 对比 cacheblend 不同重计算预算时，必须在同一份 text cache 和 suffix 下同时跑 `full-reuse`、`full-compute`、`cacheblend 1%/5%/10%/20%/30%`。每个 ratio 是独立请求，独立 scoring，不共享 score 或 top-k 计划。报告要列出各 ratio 相对 `full-reuse`、PIC `full-compute`、普通 LLM full-compute prefill 的倍数。
- 普通 LLM baseline 用真实普通导出模型跑 `llm_bench -n 0`，只读 `results[type=prefill]` 计算 `prefill_s = prompt_len / tps`；不要使用 `prefill + decode1` 或进程 wall time 代表普通 full-compute prefill。
- CUDA/OpenCL PagedAttention profiling 可用 `MNN_PAGED_ATTENTION_PROFILE=1` 打开。profile 日志应至少区分 `hydrate`、`fast_prefill`、`row` 或 `generic`。注意 profile 中会触发 `cudaDeviceSynchronize()` / `queue.finish()`，只用于定位瓶颈，不作为最终性能数。
- 远端 `pic_server` profile 写文件时 stdout 可能块缓冲；需要统计完整层数时用 `stdbuf -oL -eL env MNN_PAGED_ATTENTION_PROFILE=1 ...` 启动，或等 server 正常 flush 后再杀进程。不要把半截 profile 日志误判为缺层。
- 如果 profile 中 full-reuse 出现大量 `row` / `generic` 且 `sparse=1`，先检查是否错误触发了 PIC sparse recompute。一次错误的 1-token sparse recompute 会过全部层，KV 长度接近 PIC+prelude，延迟会远高于 hydrate 本身。
- 复查旧 `max_tokens=1` 数据时，如果出现一轮 `row` / `generic`、`sparse=0` 的 decode forward，通常是在输出第一个 token 后又为下一 token 准备 logits；它不属于当前 prefill-only 口径。
- 2026-06-03 Orange Pi 5 Plus / GLM Edge 4B / OpenCL 旧参考数据：594 PIC tokens + 15 suffix tokens + `max_tokens=1`，修正后 `native-full-reuse`、`recompute_token_count=0`，热态约 3.0s，首轮偏冷态约 4.6s。此前约 13s 的主要原因是错误默认重算最后一个 PIC token，并且 `max_tokens=1` 后额外跑了 next-logits forward。当前 prefill-only 对比应改用 `max_tokens=0` 重测。
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

`precision_recovery.execution_mode` 是判断实际执行路径的关键：`native-full-reuse` 表示磁盘 PIC KV 已写入 paged slots，`native-full-compute` 表示完整重算，`native-epic-sparse-recompute` 表示 EPIC 头部 token sparse recompute + 其余 PIC token 复用，`native-cacheblend-sparse-recompute` 表示按 score layer value delta 选 top-ratio token，`native-kvshare-sparse-recompute` 表示按 MNN C++ K/V delta influence proxy 选 top-ratio token。`epic/cacheblend/kvshare` 应有 `metadata.native_sparse_recompute_scope=python_prefill_layer_plan`；当 `pic_recompute_score_layer_idx > 0` 时，`metadata.pre_score_kv_source=request_pagedcache_full_compute` 和 `metadata.pre_score_compute_layers=<score_layer_idx>` 表示 score layer 前保持 token 正常计算语义，正常 Attention 只把 KV 写入当前请求 PagedCache，不写 scratch `.k/.v`。从 score layer 开始，选中的 PIC token 和 suffix/非复用 KV token 参与计算，其他 PIC 位置直接复用磁盘 KV。

## 优化判断

报告结果时至少说明：

- 修改了 CPU 还是 CUDA 路径，核心瓶颈是什么。
- 精度是否通过。
- 性能表包含 `stage | model | ctx | qH | kvH | D | op | latency_ms | Attention/PagedAttention`，其中 `Attention/PagedAttention > 1` 表示普通 `Attention` 比 `PagedAttention` 慢。
- 与优化前的 Jetson 数据相比，PagedAttention 的 prefill/decode 延迟和 `Attention/PagedAttention` 倍数是否改善。
