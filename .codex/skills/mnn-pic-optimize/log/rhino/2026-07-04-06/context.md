# Rhino 2026-07-04 06 Context — native full-reuse warm 重启

## 目标

检查 Rhino Pi-X1 是否在线，并解释新 native prefill 实现下 `cacheblend` / `epic` 是否退化。

本文件前半记录初始故障分析，后半记录同日源码修复和 Rhino smoke 验证。

## 运行记录

命令：

```bash
RUN_ID=pic_prefill_regression_rhino_minicpm_ctx512_current_20260704_142649
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_rhino.sh \
  --model-key minicpm5-1b \
  --contexts 512 \
  --modes normal-full-recompute,full-reuse,cacheblend,epic \
  --ratios 0.05,0.10,0.20,0.30,0.40,0.50 \
  --run-id "$RUN_ID"
```

本地结果目录：

```text
.cache/latency_budget_20260625/pic_prefill_regression_rhino_minicpm_ctx512_current_20260704_142649/
```

远端结果目录：

```text
/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/pic_prefill_regression_rhino_minicpm_ctx512_current_20260704_142649/
```

## 已确认事实

### 设备状态

初始 `ping` / `ssh` 可用；失败后 Rhino 一度 `No route to host` / ping 100% loss。随后设备恢复，`last -x` 显示：

```text
reboot system boot 5.15.148-android Sat Jul 4 14:43
```

恢复后：

```text
HOST=kalama
UPTIME≈544s
```

没有 `pic_server` 监听 `18133`。

### normal baseline 健康

`normal-full-recompute` 成功：

```text
MiniCPM5-1B ctx=512 normal-full-recompute = 1.081499178s
TPS = 473.4169107
```

warm normal 首轮生成了 OpenCL runtime cache：

```text
Can't open file:tmp/mnn_cachefile.bin
Load Cache file error.
Update cache to tmp/mnn_cachefile.bin, size = 7775632
```

本次 run 启动前 runtime cache 记录为不存在；失败后：

```text
/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/runtime_cache/opencl/mnn_cachefile.bin
size = 7775632
```

### text cache 健康

`/v1/prefill/text` 成功且为 cache hit：

```text
latency = 0.034621785s
cache_status = reused
token_count = 498
layer_count = 24
per-layer key_bytes/value_bytes = 254976
key_rope_state = canonical_no_rope
```

请求 token span：

```text
full_prompt_token_count = 512
prompt_start = 7
source_start = 0
token_count = 498
prelude_token_count = 7
suffix_token_count = 7
```

PIC config 是正确的 graph-boundary 路径：

```text
/mnt/nvme/mnn_pic_opencl/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary/config_opencl_greedy.json
```

不是历史上会污染 benchmark 的非-boundary 导出。

### PIC warm 失败

第一条 PIC chat warm 是 `full-reuse`：

```json
{
  "status": 0,
  "latency_s": 585.6372359970119,
  "payload_meta": {
    "mode": "full-reuse",
    "budget": "0"
  },
  "response": {
    "error_body": "curl: (52) Empty reply from server\n\nHTTP_STATUS:000",
    "curl_returncode": 52
  }
}
```

远端 `pic_server_ctx512.log` 只有：

```text
CPU Group: [ 0  1  2 ], 307200 - 2016000
CPU Group: [ 3  4 ], 499200 - 2803200
CPU Group: [ 5 ], 595200 - 3187200
The device supports: i8sdot:1, fp16:1, i8mm: 1, sve2: 0, sme2: 0
Starting PIC server on http://127.0.0.1:18133
Warning: module need new clone, cloning now.
```

后续所有 `cacheblend` / `epic` failure 都是：

```text
curl: (7) Failed to connect to 127.0.0.1 port 19133
```

因此这些行没有进入 cacheblend/epic scoring、top-k、sparse recompute 或 suffix prefill，不能作为性能退化样本。

## 代码路径定位

`pic_server.cpp` 的 full-reuse 分支：

```text
handleChatCompletions
  mLlm->reset()
  mLlm->generate_init(...)
  mLlm->prefillFullReuseExternalPagedKV(...)
```

`Llm::prefillFullReuseExternalPagedKV()` 当前语义：

```text
full_prompt_len = 512
pic_start = 7
pic_token_count = 498
active rows = prelude 7 + suffix 7 = 14
```

它会：

1. `beginPagedRequestIfNeeded(fullPromptLen)`
2. `reserveExternalSourceSlots(pic_token_count)`
3. `bindExternalSegments(...)`
4. `beginSparseQuery(activeLogicalIndices, 0)`
5. 设置 `sparse_query_force_plain_attention = true`
6. 只对 14 个 active token 调用 `prefill(activeTokenIds)`

OpenCL PagedAttention 的相关边界：

- `ensureCache()` 会为 request capacity + source slots 分配 PagedCache，并注册 direct mapped target。
- `hydrateExternalSegments()` 会读取持久 PIC cache 源到 source slots，再用 `pic_page_attention_hydrate_kv_inplace` 将 canonical_no_rope key 重新施加 RoPE 写回 logical slots。
- Adreno 默认 `useAdrenoSourceSlotValueHydrate()` 为 true，value 也先放到 source slots，再由 inplace hydrate 复制。

ctx512 active rows 只有 14 行，正常 compute 不可能解释 `585s` + 整机重启。故障边界集中在第一次 PIC chat forward 的 OpenCL module/session clone、首次 PagedCache capacity、direct source-slot hydrate 或首个 hydrate kernel 上。

## 当前判定

这是 native full-reuse 稳定性 bug，不是 cacheblend/epic 性能结论。

更具体地说：新实现把 full-reuse 从旧路径切到“持久 PIC cache 源 hydrate 到当前请求 PagedCache + 只算 prelude/suffix”的 native 路径后，Rhino/Adreno 在第一条 full-reuse warm 进入 `forwardRaw()` module clone 之后挂死/重启。cacheblend/epic 只是因为 server 已死而连接失败。

该形态和 2026-07-03 slot-table removal/request-generation smoke 中的 Rhino full-reuse failure 一致，远端历史日志也停在同一行 `Warning: module need new clone, cloning now.`。

## 修复/排查方案

### P0：先建立可观察的最小复现

不要跑全量 sweep。只跑：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_rhino.sh \
  --model-key minicpm5-1b \
  --contexts 512 \
  --modes full-reuse \
  --ratios 0.05 \
  --run-id <run_id>
```

服务端环境至少加：

```text
MNN_PIC_REQUEST_PROFILE=1
MNN_PAGED_ATTENTION_PROFILE=1
MNN_PAGED_ATTENTION_PROFILE_DETAIL=1
```

如果需要图级 op 边界，再加：

```text
MNN_PIC_GRAPH_PROFILE=1
MNN_PIC_GRAPH_PROFILE_TOP=1000
```

目标是确认日志最后停在：

1. `forward_raw_select_module` 之前：优先看 module clone/session 创建；
2. layer0 `hydrate` 之前/期间：优先看 direct source-slot hydrate；
3. `copy_paged_kv` / attention 之后：再看普通 PagedAttention active rows。

### P1：module clone 路径修复

如果日志仍只到 `Warning: module need new clone, cloning now.`，先处理 `forwardRaw()` 的 prefill module pool：

- full-reuse prefill 的 `seqLenKey` 应命中预期 prefill module key，而不是在第一条 PIC chat warm 才 clone 新 OpenCL module。
- 可先确认 `(mPrefillKey, all_logits)` 是否已在 `mModulePool` 中；若缺失，修复初始化/预克隆逻辑，或让 full-reuse prefill 复用已构建的 `mModule`。
- 修复后必须看到 profile 打出 `forward_raw_select_module`，再进入 layer profile。

### P2：Adreno source-slot hydrate A/B

如果能进入 layer hydrate 后才失败，用 debug fallback 做一次 A/B：

```text
MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK=1
```

这个开关只用于判断 direct mapped PagedCache source-slot 读写是否触发 Adreno driver 问题，不能作为正式性能结果。若 fallback 稳定而默认重启，重点查：

- `_readExternalValueSegmentToPagedCacheOpenCL()`
- `pic_page_attention_hydrate_kv_inplace`
- `valueSourceStart = sourceSlotStart`
- `valueSourceStride = target->maxSlots`
- source slots 与 logical slots 是否在同一 `value_cache` buffer 中出现非法重叠或 Adreno map/unmap 同步问题。

### P3：重测顺序

只有 full-reuse ctx512 连续稳定后，才进入 cacheblend/epic：

1. full-reuse ctx512 warm/measure；
2. cacheblend/epic 低 ratio smoke；
3. 1%-50% sweep；
4. 写回 `/v1/tune/update_cache` 后再正式计时。

## 同日修复记录

### 源码改动

修复点：

```text
transformers/pic_llm/engine/include/llm/llm.hpp
transformers/pic_llm/engine/src/llm.cpp
```

改动内容：

- 新增 `mForcePrefillForward`。
- `prefillFullReuseExternalPagedKV()` 用作用域 guard 包住 `prefill(activeTokenIds)`。
- `forwardRaw()` 计算 `inDecode` 时在 `mForcePrefillForward=true` 下忽略 `mDecodeForwardActive || mContext->gen_seq_len > 0`，强制按 prefill graph/module key 选择。
- `forward_raw_select_module` profile 增加 `force_prefill_forward`，用于设备侧确认。

目的：

```text
full-reuse active rows 只有 prelude + suffix，属于请求内 prefill。
它不能因为已有生成状态或 decode-ish 状态被当成 14-token decode/prefill clone key。
Rhino/Adreno 上首次 OpenCL module clone 曾在这个位置挂住并触发设备重启。
```

### 构建与同步

构建命令：

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

构建结果：

```text
pic_server, libMNN.so, libMNN_CL.so, libpic_llm.so 均为 ARM aarch64 ELF
```

同步到：

```text
/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

### Rhino smoke 验证

命令：

```bash
RUN_ID=fullreuse_rhino_fix_ctx512_20260704_153815
PIC_SWEEP_SERVER_ENV_EXTRA="MNN_PIC_REQUEST_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=1000" \
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_rhino.sh \
  --model-key minicpm5-1b \
  --contexts 512 \
  --modes full-reuse \
  --ratios 0.05 \
  --run-id "$RUN_ID"
```

本地结果：

```text
.cache/latency_budget_20260625/fullreuse_rhino_fix_ctx512_20260704_153815/summary.csv
```

summary：

```text
rhino,MiniCPM5-1B,OpenCL,cpu=max,gpu=max,ddr=max,ctx512,full-reuse,0,0.573422s,892.885 tok/s
```

HTTP 结果：

```text
warm:    status=200, prefill_latency_s=2.832662, execution_mode=native-full-reuse-hydrate-suffix
measure: status=200, prefill_latency_s=0.573422, execution_mode=native-full-reuse-hydrate-suffix
```

远端关键日志：

```text
MNN_PIC_REQUEST_PROFILE stage=forward_raw_select_module ... seq_len=14 seq_len_key=100 all_logits=0 use_decode_graph=0 force_prefill_forward=1 module_pool=2 decode_pool=0
OpenCLPagedAttention profile op=hydrate layer=0 tokens=498 kv_len=512 async_read=1 direct_segments=1 direct_tokens=498 fallback_tokens=0
MNN_PIC_REQUEST_PROFILE stage=prefill_full_reuse_external_pagedkv_total cost_ms=573.782 ...
```

确认：

- 远端 log 没有 `Warning: module need new clone, cloning now.`。
- 24 层 hydrate 都完成。
- Rhino smoke 后仍在线，未重启。

### 后续判断

full-reuse 崩溃 gate 已解除。此前同一 run 里的 cacheblend/epic failure 仍然只代表 server 已崩后的连接失败，不能用于性能结论。

### Sparse smoke

修复后又跑了低 ratio smoke：

```bash
RUN_ID=sparse_rhino_fix_ctx512_20260704_154150
PIC_SWEEP_SERVER_ENV_EXTRA="MNN_PIC_REQUEST_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE=1" \
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_rhino.sh \
  --model-key minicpm5-1b \
  --contexts 512 \
  --modes cacheblend,epic \
  --ratios 0.05 \
  --run-id "$RUN_ID"
```

summary：

```text
cacheblend 5% ctx512 = 0.374792s, 1366.091 tok/s
epic       5% ctx512 = 0.340086s, 1505.502 tok/s
```

确认：

- warm/measure 都 HTTP 200。
- `execution_mode` 分别为 `native-cacheblend-graph-boundary` 和 `native-epic-graph-boundary`。
- 远端 log 无 `Warning: module need new clone`、`ERROR`、`Empty reply` 或 `Connection refused`。
- Rhino smoke 后仍在线，未重启。

该 smoke 只证明修复后的 artifact 能继续进入 cacheblend/epic 低 ratio 路径，不替代正式 ratio sweep。完整性能结论仍需独立跑 `1%/5%/10%/20%/30%/40%/50%`，并按 warm 后计时口径报告。

## Benchmark Regression Check

目的只是判断修复后的当前 artifact 相比仓库现有 `benchmark.csv` 是否变慢；不合并、不覆盖 `benchmark.csv`。

### ctx512 / ctx1024 结果

运行：

```bash
RUN_ID=prefill_perf_rhino_minicpm_ctx512_fixed_20260704_155146
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_rhino.sh \
  --model-key minicpm5-1b \
  --contexts 512 \
  --modes normal-full-recompute,full-reuse,cacheblend,epic \
  --ratios 0.05,0.10,0.20,0.30,0.40,0.50 \
  --run-id "$RUN_ID"

RUN_ID=prefill_perf_rhino_minicpm_ctx1024_fixed_20260704_155351
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_rhino.sh \
  --model-key minicpm5-1b \
  --contexts 1024 \
  --modes normal-full-recompute,full-reuse,cacheblend,epic \
  --ratios 0.05,0.10,0.20,0.30,0.40,0.50 \
  --run-id "$RUN_ID"
```

对比 `benchmark.csv`：

```text
ctx512:
full-reuse       0.289987 -> 0.203718  -29.7%
cacheblend 5%    0.378844 -> 0.340877  -10.0%
cacheblend 10%   0.427261 -> 0.383401  -10.3%
cacheblend 20%   0.531958 -> 0.508174  -4.5%
cacheblend 30%   0.656011 -> 0.635697  -3.1%
cacheblend 40%   0.782775 -> 0.777336  -0.7%
cacheblend 50%   0.895200 -> 0.816908  -8.7%
epic 5%          0.333902 -> 0.318240  -4.7%
epic 10%         0.389586 -> 0.343992  -11.7%
epic 20%         0.490962 -> 0.459756  -6.4%
epic 30%         0.609684 -> 0.603469  -1.0%
epic 40%         0.712800 -> 0.750759  +5.3% initial, repeated below
epic 50%         0.856555 -> 0.805313  -6.0%

ctx1024:
full-reuse       0.328981 -> 0.256685  -22.0%
cacheblend 5%    0.586074 -> 0.591650  +1.0%
cacheblend 10%   0.632900 -> 0.633386  +0.1%
cacheblend 20%   0.911618 -> 0.913716  +0.2%
cacheblend 30%   1.230344 -> 1.146618  -6.8%
cacheblend 40%   1.520313 -> 1.421919  -6.5%
cacheblend 50%   1.809501 -> 1.661471  -8.2%
epic 5%          0.525158 -> 0.493258  -6.1%
epic 10%         0.563803 -> 0.556658  -1.3%
epic 20%         0.804555 -> 0.808262  +0.5%
epic 30%         1.096019 -> 1.050318  -4.2%
epic 40%         1.366975 -> 1.325820  -3.0%
epic 50%         1.696423 -> 1.647240  -2.9%
```

唯一初测超过 `+3%` 的 PIC 点是 `ctx512 epic 40%`。针对性复测：

```bash
RUN_ID=prefill_regression_check_rhino_minicpm_ctx512_epic40_20260704_160030
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_rhino.sh \
  --model-key minicpm5-1b \
  --contexts 512 \
  --modes epic \
  --ratios 0.40 \
  --run-id "$RUN_ID"
```

复测结果：

```text
ctx512 epic 40%: benchmark 0.712800s -> repeat 0.718947s, +0.9%
```

结论：

- 修复后的 native full-reuse / cacheblend / epic 在 `ctx512` 和 `ctx1024` 上没有系统性性能退化。
- `ctx512 epic 40%` 初测 `+5.3%` 不稳定，复测回到 `+0.9%`，判为运行噪声。
- `ctx1536/2048/2560` 扩展 sweep 曾被启动，但用户确认只需要测试对比、不需要覆盖 benchmark；该扩展在 `ctx1536 normal warm` 阶段中断，不用于结论。

本次失败不合并到 `benchmark.csv`。若需要记录，应只作为 failure/raw log，不作为 cacheblend/epic latency 行。
