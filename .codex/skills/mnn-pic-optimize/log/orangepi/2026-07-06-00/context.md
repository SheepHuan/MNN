# Context — PIC decode x0 vs normal + Plan A A/B

> 配套 `README.md`。README 给目标、判据、cache 聚焦路线；本文件给完整数据、命令、Plan A 实现、退化原因细节。
>
> **目标（北极星）**：参考 normal decode + 历史经验，找一条让 **Mali + Adreno** 在 PIC decode
> `x=0/1/3/5/7/.../n` 全 x 都收益的 **cache 聚焦**路线（允许变体 kernel / 不同参数 / image/buffer
> 不同格式）。**终极判据：PIC decode x=0 TPOT ≤ true normal decode q=1**，3 模型 × ctx{512,1024} 成立。
> 当前：Mali 5/6 达标，Adreno 6/6 未达标（+43~+115%），主战场是 Adreno。

## Artifact

```text
commit: 38ba6851 (pic: split paged attention backend families) + working-tree mods
branch: kvshare-edge
Plan A env: MNN_PAGED_ATTENTION_DECODE_REPAIR_SPLIT_KERNELS (default OFF)
OrangePi artifact: .cache/output/mnn/artifacts/orangepi5plus/ (built 2026-07-06 00:00)
  remote: orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
Rhino data: 7/5 同期 artifact (decode_adreno_q1_identity_fused_x0_20260705_1225 + adreno_true_normal_current_20260705_cmp)
```

## 当前实现 PIC x0 vs true normal decode q=1

### 数据来源 CSV

```text
OrangePi PIC x0 (本轮 OLD, env off = 当前实现):
  .cache/mnn-pic-benchmark/decode_experiment/decode_ab_old_splitkernels_orangepi_20260706/decode_tpot_long.csv
OrangePi normal ctx1024:
  .cache/mnn-pic-benchmark/decode_experiment/cmp_normal3_x0_claude/decode_tpot_long.csv
OrangePi normal ctx512 (多 run 稳定值, decode_wall_formal_orangepi.../decode_q1_sparse_flash...):
  MiniCPM5-1B=46.74, Llama3.2-3B=121.62, Qwen3-4B=149.83  (各 run 一致, ±0.3ms)
Rhino PIC x0:
  .cache/mnn-pic-benchmark/decode_experiment/decode_adreno_q1_identity_fused_x0_20260705_1225/decode_tpot_long.csv
Rhino normal:
  .cache/mnn-pic-benchmark/decode_experiment/adreno_true_normal_current_20260705_cmp/decode_tpot_long.csv
  (source: "llm_bench true normal export -kv true -n 16; TPOT is autoregressive q=1 decode")
```

### OrangePi / Mali-G610 (OpenCL, TPOT ms)

```
model         ctx   PIC_x0    normal    gap_ms    gap_pct
Llama3.2-3B   512   119.24    121.62    -2.38     -2.0%   PIC 更快
Llama3.2-3B   1024  119.27    125.64    -6.38     -5.1%   PIC 更快
MiniCPM5-1B   512   44.48     46.74     -2.26     -4.8%   PIC 更快
MiniCPM5-1B   1024  46.06     49.96     -3.90     -7.8%   PIC 更快
Qwen3-4B      512   150.32    149.83    +0.49     +0.3%   持平
Qwen3-4B      1024  172.46    154.86    +17.60    +11.4%  PIC 慢 (唯一明显落后)
```

### Rhino / Adreno (OpenCL, TPOT ms)

```
model         ctx   PIC_x0    normal    gap_ms    gap_pct
Llama3.2-3B   512   76.78     53.76     +23.01    +42.8%
Llama3.2-3B   1024  88.58     56.66     +31.93    +56.3%
MiniCPM5-1B   512   48.32     27.35     +20.97    +76.7%
MiniCPM5-1B   1024  54.26     25.18     +29.08    +115.5%
Qwen3-4B      512   100.31    69.77     +30.53    +43.8%
Qwen3-4B      1024  115.72    72.37     +43.36    +59.9%
```

关键：两设备 gap 量级差 ~10×。Mali 5/6 cell 追平/超过 normal；Adreno 全部 +43% 起步。

## Plan A 实现

### 新增文件

- `source/backend/opencl/execution/buffer/PagedAttentionBufExecutionDecodeSplitKernels.cpp`
  - 自包含 TU（本地拷贝 `_envFlagEnabled`/`_nowUs`/`_profilePagedAttention`/`_picOpenCLDebug`/`_makeOpenCLPmcMeta`/`_run3DKernelDefaultPmc`，因主 TU 的这些 static 不可见）。
  - 统一自门控入口 `runDecodeCausalAttentionHD128SplitKernels`：env off / attnLen!=1 / 任何前置不满足 → return INVALID_VALUE → 主 dispatch fall through 到默认融合 path。
  - `ensureDecodeCausalKernelHD128Split`：build QK kernel (`-DNUMHEAD_GROUP_SIZE=`) + QKV kernel (`-DNUMHEAD_GROUP_SIZE= -DLOOP_UNROLL_8`)。softmax 复用 `mDecodeSoftmaxKernel`（`softmax_buf`/`softmax_in1_buf`，`-DSOFTMAX_LOCAL_SIZE=64`，已在 onResize build）。
  - dispatch 顺序：append（复用 `mDecodeKeyAppendSparseKernel`）→ QK → softmax → QKV → rank capture。
  - 中间 tensor 复用 `mTempQK`/`mTempSoftmax`（`ensureDecodeTransposedTemps(kvLen, 1)`，布局 `[batch*head_num, ROUND_UP(kvLen,8)]`）。

### 修改文件

- `source/backend/opencl/execution/cl/paged_decode_attention_buf.cl`：新增 `matmul_qk_decode_pic_hd128` + `matmul_qkv_decode_pic_hd128_b8`（接在 sparse row128 宏实例化后）。已 `opencl_codegen.py` + `git diff --check`（source 干净；benchmark.csv 的 trailing whitespace 是无关数据文件）。
- `PagedAttentionBufExecution.hpp`：声明 2 个成员函数 + `mDecodeQKSplitKernel`/`mDecodeQKVSplitKernel`/`mDecodeSplitKernelGroupSize`。
- `PagedAttentionBufExecution.cpp`：`decodeRepairSparseQTile` 分支里加 1 行 `runDecodeCausalAttentionHD128SplitKernels(...)` 调用（自门控，失败 fall through）。
- `PagedAttentionBufExecutionDecode.cpp`：**未改**（Plan A 完全 isolated 到新 TU）。

### kernel 设计要点

- QK kernel gws `{UP_DIV(kvLen,4), batch*numHead}`，K 源 `decode_key[((b*kvh)*128+d)*key_max_len + k]`（转置 stride-1），causal: future 位置写 `-FLT_MAX`（softmax max-subtract 数值稳定，已确认 `softmax_in1_buf` 实现）。x0 q_index 固定 0。
- QKV kernel gws `{16, batch*numHead}`，V 源 `value_cache[((b*kvh)*key_max_len + slot)*128 + x8]`，slot_identity 硬假设（slot==logical，与 append kernel + 现有 sparse path 一致；无运行时检查函数，`mLastDecodeKeySlotIdentity` 仅 profile 标记）。
- slot_identity / decodeKey / append 全复用现有 path，未引入第二份 KV cache（符合 PagedCache 硬约束）。

## 验证流程

### Step 1: build + sync (OrangePi)

```bash
cd source/backend/opencl/execution/cl && python3 opencl_codegen.py .
git diff --check -- source/   # 干净
JOBS=$(( $(nproc) / 2 )) MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server \
  BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
file .cache/output/mnn/artifacts/orangepi5plus/bin/pic_server  # aarch64 ELF
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

### Step 2: profile smoke (Plan A 命中 + P0)

```bash
RUN_ID=planA_profile_smoke_orangepi_qwen_ctx1024_x0_20260706
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_REPAIR_SPLIT_KERNELS=1 \
MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=200 MNN_PAGED_ATTENTION_PROFILE=1 \
MNN_PIC_DECODE_DEBUG=1 MNN_PIC_DECODE_REPAIR_PROFILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models qwen3-4b --contexts 1024 \
  --pic-repair-tokens 0 --pic-max-tokens 2 --pic-suffix-from-cache-tokens 1 \
  --repeats 1 --warm-repeats 1 --skip-normal --run-id "${RUN_ID}"
```

结果：72 条 `plan_a_split` 日志行（36 层 × 2 token），全部 `decode_prepare_inside_decode=0` `ordinary_q1=0` `slot_identity=1`。`Cache invalid, will be reset` 仅启动时 1 次（binary 变更后的 tune-cache MD5 重置，非运行时失败，decode 中 0 复发）。无 ERROR / target unavailable / async failed。

### Step 3: 正确性 (token 一致性)

```bash
# default (env off) vs Plan A (env on), Qwen3-4B ctx512 + ctx1024, max-tokens 16
# 两路径生成 content 均为 '!!!!!!!!!!!!!!!!' (harness token_ids_16 prompt 特性)
# MATCH=True (逐 token 一致, Plan A 无回归)
```

### Step 4: formal TPOT A/B (profile off, 有效性判据)

```bash
# OLD (env off = 当前实现)
RUN_ID=decode_ab_old_splitkernels_orangepi_20260706
PIC_SWEEP_SERVER_ENV_EXTRA='' python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models llama3.2-3b,minicpm5-1b,qwen3-4b --contexts 512,1024 \
  --pic-repair-tokens 0,1 --pic-max-tokens 16 --pic-suffix-from-cache-tokens 1 \
  --repeats 2 --warm-repeats 1 --skip-normal --run-id "${RUN_ID}"

# NEW (Plan A env on)
RUN_ID=planA_new_orangepi_20260706
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_REPAIR_SPLIT_KERNELS=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models llama3.2-3b,minicpm5-1b,qwen3-4b --contexts 512,1024 \
  --pic-repair-tokens 0,1 --pic-max-tokens 16 --pic-suffix-from-cache-tokens 1 \
  --repeats 2 --warm-repeats 1 --skip-normal --run-id "${RUN_ID}"
```

## Plan A A/B 结果 (OrangePi/Mali, profile off, 2 repeats, warm 1)

```
model         ctx   x  OLD(融合)  NEW(Plan A)  delta     pct
Llama3.2-3B   512   0  119.243    126.109      +6.866    +5.8%  退化
Llama3.2-3B   1024  0  119.268    138.863      +19.595   +16.4% 退化
MiniCPM5-1B   512   0  44.485     50.838       +6.353    +14.3% 退化
MiniCPM5-1B   1024  0  46.059     61.642       +15.584   +33.8% 严重退化
Qwen3-4B      512   0  150.316    170.447      +20.131   +13.4% 退化
Qwen3-4B      1024  0  172.460    173.910      +1.450    +0.8%  持平(噪声)
Llama3.2-3B   512   1  162.971    163.332      +0.361    +0.2%  持平 (env gate 干净)
Llama3.2-3B   1024  1  178.899    180.310      +1.411    +0.8%  持平
MiniCPM5-1B   512   1  70.762     71.610       +0.848    +1.2%  持平
MiniCPM5-1B   1024  1  91.034     89.885       -1.149    -1.3%  持平
Qwen3-4B      512   1  213.433    217.813      +4.380    +2.1%  持平
Qwen3-4B      1024  1  237.048    234.948      -2.099    -0.9%  持平
```

- failures.csv 空，所有行 status=ok。
- x0 全退化（+6~+34%），x1 全 ±1% 持平 → env gate 干净。
- MiniCPM ctx1024 退化最严重（+33.8%），反证 GQA L2 复用没拿到（否则 group=8 应收益最大）。

## 退化原因（详细）

### 1. 中间 tensor 往返 + 多 launch（主因）

PIC 融合 path = 1 个 attention kernel（QK+softmax+QKV 融合，省 qk/softmax 中间 tensor 往返）。
Plan A = 3 个独立 kernel，多 2 次 launch + 2 次中间 tensor global memory 写读：
- 中间 tensor = `kvLen × numHead × batch × 2B`，Qwen ctx1024 = 64KB(qk)+64KB(softmax) = 128KB/layer × 36 层。
- 多 2 次 launch × 36 层 ≈ 1.5–4ms（Mali per-launch+finish ~20–60µs）。
- 中间 tensor DRAM 往返 ~128KB×36×2(写+读) ≈ 9MB，Mali ~10GB/s ≈ 0.9ms。
- 合计 ~2–5ms，与实测 +6~+20ms 退化同量级。

### 2. GQA K L2 复用没拿到（核心赌点失败）

Mali-G610 L2 = 1MB/core（10 核各独立，非全局共享）。Plan A QK gws `{257, 32}`（Qwen ctx1024）切多个 workgroup 跨核心调度，同 GQA group 4 个 query-head 若跨核心，L2 复用归零。从 MiniCPM（group=8）退化最严重反推，K external read 没降下来。本轮未跑 PMC 验证（`orangepi_opencl_pmc_profile.md` 流程可补）。

### 3. V 全占用收益 < 往返开销

V 阶段在 q=1 decode 不是瓶颈：原融合 path 的 32-lane V（128 lane workgroup 里 32 活跃）只占 attention 总时间 ~25%。Plan A 把 V 拆成全 128-lane 占用，收益 ~25%×attention，但被中间 tensor 往返 + launch 吃掉。

### 4. append 3 份写（PIC 固有，Plan A 无法改善）

`append_sparse_decode_key_value_hd128` 每 token 写 key_cache + value_cache + decodeKey 3 个 store。normal append（`rearrange_k`）只写 1 份 past_key。PIC append bandwidth 恒为 normal 3×。Plan A 复用同一 append，此项不退化但解释了 PIC 难追平 normal 绝对速度。

## 与前 5 个 tried-and-failed 的关系

Plan A 是第 6 个"想把 PIC x0 往 normal decode 并行模型上靠"的尝试，全部退化：
1. fused-append（07-05-13）：x0 +14.7ms
2. q1-gqa fused（07-02/05）：Mali Qwen 1.96→2.4-3.0ms/layer
3. slot-identity-split：Qwen x0 243ms vs 162ms
4. lane-force 32/64/128：无稳定收益
5. q1-identity-fused-kv（07-05-00 Adreno）：全退化
6. **Plan A split-kernels（本小时）：x0 +6~+34%**

规律：normal decode 的优势依赖（无 append / L2 跨 workgroup 复用 / 稳态调度亲和性），PIC x0 在 PagedCache 约束下都不满足。

## 待办（新会话）

- [ ] 按硬约束决定：Plan A `#if 0` 注释 + 记录第 6 个 tried-and-failed，或保留 env-gated 等深挖。
- [ ] Rhino/Adreno device-side profile：gap +43~+115% 是最大未解决问题，且和 Mali 量级差 10× 说明是设备执行特性（Adreno record queue / L2 / TLP）。优先级高于继续调 attention 算法。
- [ ] 若继续优化 Mali Qwen ctx1024（唯一 +11.4% cell）：方向是融合 kernel 内部解决 V 阶段 32-lane 空闲（`log/orangepi/2026-07-05-22/context.md` 末尾"唯一未试 lever"），不要再拆融合。
- [ ] 把"两设备 gap 量级差 10×"结论回收进 SKILL.md `Decode 算子优化约束`。
