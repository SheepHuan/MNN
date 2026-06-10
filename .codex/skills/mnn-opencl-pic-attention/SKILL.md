---
name: mnn-opencl-pic-attention
description: 当用户要求分析或优化 MNN PIC/PagedAttention 的 OpenCL Attention 计算、OrangePi OpenCL cacheblend/epic/kvshare sparse prefill、score layer 前后 full compute 与 sparse compute 边界、PagedAttentionBufExecution.cpp/attention_buf.cl/softmax_buf.cl 热点和逐层延迟时使用。
---

# MNN OpenCL PIC Attention Optimization

本 Skill 用于优化 MNN PIC/PagedAttention 在 OpenCL backend 上的 attention 计算，尤其是 OrangePi/Mali 上的 cacheblend、epic、kvshare sparse prefill。核心目标是先保证执行语义正确，再做 kernel 优化。

## 入口约束

1. 从 MNN 仓库根目录工作，先读 `AGENTS.md`。
2. 修改源码前运行：

```bash
git status --short
```

3. 不要读取或修改 `schema/private/`、`source/internal/`。
4. 构建产物、日志和实验输出放到 `.cache/`、`output/` 或用户指定目录。
5. 构建、同步和跑 PIC server benchmark 时，按需读取：

- `.codex/skills/mnn-build-artifacts/SKILL.md`
- `.codex/skills/mnn-llm-bench/SKILL.md`
- `.codex/skills/mnn-pic-benchmark/SKILL.md`
- `.codex/skills/mnn-opt-ops/SKILL.md`

## 核心执行边界

PIC sparse 优化必须按 score layer 分层，不能把整张模型都用选中 token 从第 0 层重跑。

对 `cacheblend` / `delta-v` / `kvshare` / `epic`：

- `layer < score_layer_idx`：必须保持正常 full compute 语义。执行 full prompt 的 prelude + PIC tokens + suffix，使用普通 full-prompt PagedAttention，把当前请求 K/V 写入 PagedCache。这里只能优化 full prefill fast path，不能启用 sparse query、不能只算 selected PIC tokens。
- `layer == score_layer_idx`：这是 sparse 起点，即 `sparse_start_layer = score_layer_idx`。本层先用 full prompt 计算/写入评分所需的 K/V，并完成 delta-v / delta-k / top-k；拿到 active token 后立刻裁剪 hidden states / query / residual 等后续必要变量，从本层 attention 和后续计算开始进入 sparse。
- `layer >= score_layer_idx`：可以考虑 sparse attention 优化。未选中的 PIC token 从持久 PIC cache 源 hydrate 到当前请求 PagedCache；选中的 PIC token 按真实 logical position sparse recompute；suffix 继续读同一份 PagedCache。

不需要维护一套完整的新 PIC LLM engine；正确目标是维护一个轻量 score-layer boundary。这个 boundary 优先做成 backend kernel 级的 `pic_active_rows` 模式，而不是切独立 module：score layer 的 PagedAttention 先 full-K/V scoring/top-k，然后切到 compact active rows；同一层后续需要适配的 kernels（residual/add、o_proj、norm、MLP、下一层 QKV projection 等）都识别 `pic_active_rows`，只计算 active token rows，并按 active logical index 从原 full hidden/residual 读取必要输入。

`pic_active_rows` 模式下 tensor 的静态 shape 可以暂时仍是 full prompt 长度，但有效 seq_len 是 `active_count`，有效数据放在前 `active_count` 行。所有尚未适配的 kernel 如果继续按 full seq_len 读写，会把未裁剪的行重新带进计算，既不正确也不省时；因此不要只改 PagedAttention 一个 kernel 就声称完成 sparse prefill。

如果当前调度只能调用 `forwardVec(selected_tokens)` 从模型输入开始重跑，导致 `layer 0..score_layer_idx-1` 也走 sparse query，这是错误路径。应改成 score-layer boundary：先 full prompt 产生 score 所需 K/V，再 gather/crop active hidden states，并从 `score_layer_idx` 对应的 attention/后续子图继续 sparse。做不到时要标注 unsupported/fallback，不能用错误语义换速度。

`full-reuse` 和 `full-compute` 不属于 sparse 优化：

- `full-reuse`：不做 scoring，不做 sparse recompute，只把持久 PIC cache 源 hydrate 到当前请求 PagedCache 并计算 suffix。
- `full-compute`：不读取磁盘 PIC KV，完整计算 prelude + PIC tokens + suffix，但仍通过 PagedCache/PagedAttention 读写。

## 图级 sparse 边界设计

当前正确方向是把 PIC sparse recompute 做成“图输入 budget + score layer 产出 active indices + 图中显式 gather”的边界，而不是让 runtime 从 layer 0 用 selected tokens 重新 forward。

图输入只应该新增 `pic_recompute_budget`，shape 为 `[1]`，类型为 int32。它是一个标量，表示 score layer 之后 compact active rows 的长度，不是 indices 向量。具体 token 坐标不能作为 graph input；它们必须在 score layer 的 PagedAttention backend 中由 scoring/top-k 实时产生。

`budget` 的语义是 score layer 之后继续参与计算的总 active row 数，而不是“只选中的 PIC token 数”。通常应包括：

- prelude / 非复用前缀 rows。
- top-k 选中的 PIC rows。
- suffix / 当前请求非复用 rows。

score layer 的 PagedAttention 应支持 5 输入、2 输出：

```text
inputs:
  query, key, value, attention_mask, pic_recompute_budget

outputs:
  attention_output  [batch, budget, hidden]
  active_indices    [budget] int32 logical positions
```

导出图的层级语义：

- `layer < score_layer_idx`：不插入 hidden/query/residual gather，Q/K/V/mask/rotary 都保持 full prompt shape。
- `layer == score_layer_idx`：PagedAttention 输入的 Q/K/V 仍来自 full hidden；K/V 必须按 full prompt 写入 PagedCache，并在同一层完成 full-length scoring/top-k。PagedAttention 输出 compact attention rows 和 `active_indices`。本层 attention 后的 residual、norm hidden、gate、PLE per-layer input 等必须用同一个 `active_indices` gather，避免 full rows 被重新带入 FFN。
- `layer > score_layer_idx`：hidden 已经是 compact rows；后续 attention 的 Q/K/V 自然是 compact 长度，但 PagedAttention 写入和读取 PagedCache 时仍通过 `active_indices` / metadata 使用真实 logical position。

导出端重点：

- `transformers/pic_llm/export/llmexport.py` 暴露 `pic_recompute_budget`，不要暴露 `pic_recompute_indices`。
- `transformers/pic_llm/export/utils/custom_op.py` 的 score-layer PagedAttention symbolic 返回两个输出，第二输出是 `[budget]` 的 int32 active indices。
- `transformers/pic_llm/export/utils/model.py` 在 score layer 之前不 gather；score layer 返回 active indices 后，再 gather rotary、后续 mask query axis、PLE input 等。
- `transformers/pic_llm/export/utils/transformers.py` 的 Decoder 在 score layer 内 gather residual / norm hidden / gate / per-layer input 后再进入 FFN。

shape 端重点：

- `source/shape/ShapePagedAttention.cpp` 需要把第 4 个输入标记为 shape content dependency。
- 当 PagedAttention 有 budget 输入时，output 0 的 query 维度来自 `pic_recompute_budget[0]`，output 1 shape 为 `[budget]` int32。
- 普通 PagedAttention 没有第 5 输入时保持原 shape 语义。

runtime 端重点：

- 不需要新增公开 forward API。现有 `forwardRaw` / `forwardVec` / `prefill` 在模型配置包含 `pic_recompute_budget` 时，多喂一个 `{budget}` 的 int32 VARP 即可。
- full-compute / 非 sparse 请求可以传 `budget = seq_len`，score layer 产出 identity active indices。
- cacheblend / epic / kvshare 请求应在一次 full-prompt forward 前把 PagedKVMeta 准备好：pic start、pic token count、score layer、topK、持久 PIC cache 源 segment 等。score layer backend 完成 scoring/top-k 后，把 active logical indices 写入 PagedKVMeta 和 PagedAttention 第二输出。
- 旧的 `forwardVec(selected_tokens)` sparse recompute 只能作为 legacy/unsupported fallback；不能作为 native PIC sparse 生产路径。

PagedAttention 后端必须拆分两个长度：

```text
kvWriteLen = newKvLen / meta.add    // 本层新 K/V 写入 PagedCache 的长度
attnLen    = output query length    // attention 真正计算/输出的 compact rows
```

禁止再用 `insertLen = min(newKvLen, queryLen)` 同时控制 K/V 写入和 Q 计算。score layer 的典型形态是 full Q input + full K/V write + compact output；后续层通常是 compact Q/K/V input + compact output，但 logical position 仍由 active indices 决定。

## OpenCL PagedCache 零拷贝边界

文档、日志和新增代码不要使用容易暗示“另一份运行时 KV cache”的旧英文命名；统一使用“持久 PIC cache 源”和“当前请求 PagedCache”。持久 PIC cache 源只表示 `/v1/prefill/text` 已经构建好的可复用 cache 数据来源，不表示另一份运行时 KV cache。新增变量名也要按这个语义命名；历史 API 名称只允许作为待收敛 legacy 代码的定位线索，不进入新的说明、日志或生产路径设计。

OpenCL 生产路径的硬约束：

- 当前请求唯一真实 KV 工作区是 PagedCache。
- PagedCache 在 OpenCL backend 中必须以可 map 的 zero-copy OpenCL buffer 为目标；hydrate 直接写入当前请求 PagedCache 对应 logical slot，不允许绕到另一份运行时 KV cache。
- 持久 PIC cache 源需要在 GPU/CL kernel 中参与 score 或 hydrate 时，必须先写入同一份 PagedCache 的保留 physical source slots；score kernel 从 PagedCache 的 reference slots 和 source slots 做比较，hydrate kernel 从 source slots 读取 canonical key 并写回真实 logical/physical slot。
- 不允许新增 chat/scoring/sparse recompute 的中间 `.k/.v` 文件。
- 不允许新增完整 K/V host vector、中间 CL tensor 或旧式中转缓存来承接整层 PIC KV；历史 staging buffer 只能在显式 fallback/export 场景存在，不能扩展为 native OpenCL PIC 路径。
- OpenCL 默认要求 mapped PagedCache 路径；只有显式调试开关 `MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK=1` 可以进入 staging fallback。正式 OrangePi profile 不允许打开这个开关。
- 若设备或 driver 无法提供 mapped/zero-copy 路径，必须明确标注为 unsupported/fallback；不要把额外 KV 中转伪装成 native OpenCL PIC 路径。

OpenCL 先跑通时的保守策略：

- score layer 的 input query 仍是 full rows，但 output 是 compact rows，因此不能直接走当前 sparse fast prefill kernel，除非 kernel 明确按 `active_indices[q]` 去读取 full query row。
- 当前 `runSparseFastPrefill` / `rearrange_q` 假设输入 query 已经是 compact rows。score layer full-Q/compact-output 场景应先走 row/generic correctness path，或新增专门的 gather/rearrange-Q kernel 后再启用 fast path。
- `copy_paged_kv` / hydrate / scoring 使用 `kvWriteLen` 和 full KV length；attention row/generic kernel 使用 `attnLen`，并在 score layer 用 active logical index 读取 full query row。
- score layer backend 在 scoring/top-k 后需要把 active logical indices 写到 `outputs[1]`，同时同步到 `PagedKVMeta::sparse_query_logical_indices`，让后续层 PagedAttention 使用真实 logical slot。
- 后续层输入 hidden 已被图中 gather 成 compact rows，OpenCL sparse fast path 才可以按 compact Q 继续优化。

## 关键源码位置

- `transformers/pic_llm/engine/app/pic_server.cpp`
  - `buildExecutionPlan`
  - cacheblend/epic/kvshare 请求调度
  - `pic_recompute_score_layer_idx`、`sparseLogicalIndices`
- `transformers/pic_llm/engine/src/llm.cpp`
  - 持久 PIC cache 源绑定、score-layer graph boundary prefill、legacy sparse recompute fallback
  - `runCacheBlendScorePrefill`
  - `gen_attention_mask` / `gen_position_ids` 的 sparse query 分支
- `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
  - cacheblend scoring
  - 持久 PIC cache 源 hydrate 到当前请求 PagedCache
  - `runFastPrefill`
  - `runSparseFastPrefill`
  - `canUseSparseFastPrefill`
- `source/backend/opencl/execution/cl/attention_buf.cl`
  - sparse/full QK and QKV kernels
- `source/backend/opencl/execution/cl/softmax_buf.cl`
  - sparse/full softmax kernels

## 逐层 Profile 口径

OpenCL attention profile：

```bash
MNN_PAGED_ATTENTION_PROFILE=1
MNN_PAGED_ATTENTION_PROFILE_DETAIL=1
```

重点看这些日志：

- `op=prefill_attention_fast_qk_softmax_qkv`：full prefill attention。
- `op=cacheblend_score layer=N`：score/top-k 所在层。
- `op=hydrate layer=N`：持久 PIC cache 源直接 hydrate 到当前请求 PagedCache。
- `op=sparse_prefill_attention_fast_qk_softmax_qkv layer=N`：sparse recompute attention。

解释日志时要注意：

- `layer=N` 是真实 PagedAttention layer index，不是 profile parser 的序号。
- 若 `cacheblend_score layer=1` 后马上出现 `sparse_prefill... layer=0`，这是语义红灯：sparse recompute 被错误地从 layer 0 开始执行。
- profile detail 会在每个子 op 后 `queue.finish()`，只用于拆瓶颈，不作为正式 latency。
- `qk_active_tiles / qk_rect_tiles` 接近 1 时，说明 selected Q 的 logical position 很散，虽然 Q 行少了，但每个 Q chunk 的 K 方向仍接近满长。

## 优化优先级

1. 先修正执行语义：`sparse_start_layer = score_layer_idx`。score layer 之前 full compute；score layer 当层先 full-K/V scoring，然后立刻裁剪 active hidden states，从本层开始 sparse。不要让 `forwardVec(selected_tokens)` 从 layer 0 开始触发 sparse query。
2. 再看 sparse attention 热点。通常 QK 是主热点，QKV 次之，hydrate/disk 往往不是第一瓶颈。
3. 如果第一个 sparse layer 的 QK 比后续层慢数秒，优先怀疑 OpenCL `localWS3DDefault` 调参或首次执行开销。只在 PagedAttention sparse fast path 内处理：固定/禁用 request-critical LWS autotune，或在正式计时前预热对应 shape。
4. 如果 `qk_active_tiles / qk_rect_tiles` 很高，优化方向是让 sparse Q 按 logical position 分组、缩小 K 有效范围，或设计按 Q range 的分段 QK，而不是继续优化 hydrate。
5. direct PagedCache value prefill 主要影响 QKV 和 pack，不会解决 QK 主瓶颈；只有在 QK 已经降下来后再评估。
6. 不要用 `MNN_PAGED_ATTENTION_IMPL` 或 V2 路由影响生产路径；bench ops 可以保留 V1/V2 对比入口。

## 验证要求

每次改动后至少确认：

- `cacheblend` / `epic` 的 score layer 之前没有 `sparse_prefill... layer < score_layer_idx`。
- `normal LLM full-compute` baseline 只在普通模型或实现变化时重测；不要为了每个 ratio 反复测 normal。
- `PIC full-compute` 仍接近 normal full-compute。
- `full-reuse` 明显快于 PIC full-compute。
- `cacheblend` / `epic` 每个 ratio 是独立请求，延迟包含本次 scoring/top-k、hydrate、sparse recompute、suffix prefill。
- 报告 OrangePi TPS 时保持简单 CSV 格式：

```text
model_name,mode,buget,latency_s,effective_tps,speedup_vs_normal
```

## 判断热点

优先用逐层 breakdown 判断优化点：

- `hydrate` 总计只有几百 ms：不要优先优化磁盘或 direct hydrate。
- `sparse_total` 中 `qk_us` 占大头：优化 sparse QK kernel、LWS、Q/K 分块和 K range。
- `layer 0 qk_us` 远大于其他层：先排查首次 sparse shape 调参/编译/driver overhead，再看是否错误从 layer 0 sparse。
- 所有层 QK 都慢且随 ratio 线性增长：需要真正减少 QK 计算量，而不是只减少请求 token 列表长度。

## 2026-06-10 x64 CPU 稳定性检查

CPU backend 是当前 score-layer graph boundary 的 reference 语义。调 OpenCL 前先保证 CPU 路径满足：

- `PicScoreAttention`：输入 Q/K/V 仍是 full prompt rows；先把 full K/V 写入当前请求 PagedCache，再用 score layer 完成 cacheblend scoring/top-k 或固定 epic active rows；输出为 compact active rows，并通过第二输出写出 `active_indices`。
- `PicSparseAttention`：输入 hidden/Q/K/V 已被图中 gather 成 compact rows；K/V 只更新 `active_indices` 对应 logical slots；attention 读 `kvLen = PagedKVMeta::logical_length` 的 full KV。
- `/v1/prefill/text` 即使图里存在 `PicScoreAttention` 第二输出，也不能激活 sparse state；只有真实 PIC runtime active 的 score layer 才调用 `PagedKVMeta::activatePicRows(...)`。
- CPU `queryRowsAreFull` 只用于 score layer 的 full-Q/compact-output 形态，此时 Q row 和 mask query row 都按 active logical index 读取；后续 sparse layers 使用 compact query row。

已定位过一次 CPU 崩溃：`/v1/prefill/text` 在 PIC graph-boundary 模型上触发 `SIGFPE`，gdb 栈顶为 `MNN::execute<int, int, BinaryModInt<...>> -> MNN::CPUBinary::onExecute -> Transformer::Llm::forwardRaw -> PicServer::buildTextCache`。根因不是 PagedAttention KV 本身，而是 CPU scalar attention mask fast path 给 `attention_mask` 传 rank-0 tensor；导出图在 score layer 后对 `attention_mask` 做 `_pic_gather_rows(..., -2)`，形状规范化路径里对 `Rank(attention_mask)` 做 modulo，rank 为 0 时除零。修复口径：当 `config.has_pic_recompute_budget()` 为 true 时，CPU 也使用正常 full attention mask，不走 rank-0 scalar mask fast path。

本地 x64 CPU 验证命令：

```bash
cmake --build .cache/build/mnn/x64_cpu_pic --target pic_server --parallel 48
.cache/build/mnn/x64_cpu_pic/pic_server \
  --config .cache/weight/AI-ModelScope__Llama-3___2-1B-Instruct-pic-boundary/config.json \
  --host 127.0.0.1 --port 18120 \
  --kv-cache-dir .cache/kvshare/cpu_pic_local \
  --model llama-pic-cpu
```

验证结果：

```text
/v1/prefill/text cpu-doc-small: HTTP 200, 0.5405s, layer_count=16, key=[23,1,8,64], value=[1,8,23,64], key_rope_state=canonical_no_rope
/v1/chat/completions full-reuse max_tokens=0: HTTP 200, 0.4990s, execution_mode=native-full-reuse
/v1/chat/completions epic ratio=0.1 max_tokens=0: HTTP 200, 0.5225s, execution_mode=native-epic-graph-boundary, selected local=[0,1,2]
/v1/chat/completions cacheblend ratio=0.1 max_tokens=0: HTTP 200, 0.5395s, execution_mode=native-cacheblend-graph-boundary, selected local=[0,6,10]
```

注意：`POST /reset` 本地用 `Content-Type: application/json` 和 body `{}`，空 body 会被 httplib 请求层返回 400。

## 2026-06-10 OrangePi OpenCL 实测

模型：`AI-ModelScope/Llama-3.2-1B-Instruct`，PIC graph-boundary 导出，`score_layer_idx=1`，OpenCL low precision。普通 LLM baseline 使用真实普通导出模型 `llm_bench -p 1491 -n 0 -rep 3`，结果 `143.06 tok/s`，即 `normal_full_compute_s=10.4222s`。

PIC server prefill-only 使用同一份持久 PIC cache 源和 suffix，每个 ratio 都独立 `/reset` 后发 `max_tokens=0` 请求，延迟包含本次 score/top-k、hydrate、sparse recompute、suffix prefill。

当前默认 sparse fast path（`MNN_PAGED_ATTENTION_OPENCL_SPARSE_FAST_Q_CHUNK=64`）best 结果：

```text
context_tokens,algorithm,ratio,best_s,speedup_vs_normal
1488,full-reuse,,1.5350,6.790
1488,epic,0.01,2.0160,5.170
1488,epic,0.05,2.9070,3.590
1488,epic,0.10,2.7570,3.780
1488,epic,0.20,10.5160,0.990
1488,epic,0.30,14.7420,0.710
1488,cacheblend,0.01,2.2000,4.740
1488,cacheblend,0.05,2.8740,3.630
1488,cacheblend,0.10,3.4040,3.060
1488,cacheblend,0.20,12.4930,0.830
1488,cacheblend,0.30,18.5640,0.560
```

可选调参结果（`MNN_PAGED_ATTENTION_OPENCL_SPARSE_FAST_Q_CHUNK=256` + `MNN_PAGED_ATTENTION_OPENCL_DIRECT_VALUE_PREFILL=1`）：

```text
context_tokens,algorithm,ratio,best_s,speedup_vs_normal
1488,full-reuse,,1.5530,6.710
1488,epic,0.01,3.1330,3.330
1488,epic,0.05,3.2440,3.210
1488,epic,0.10,3.6450,2.860
1488,epic,0.20,9.8830,1.050
1488,epic,0.30,12.5540,0.830
1488,cacheblend,0.01,1.9590,5.320
1488,cacheblend,0.05,2.3600,4.420
1488,cacheblend,0.10,3.4710,3.000
1488,cacheblend,0.20,11.7730,0.890
1488,cacheblend,0.30,15.0210,0.690
```

结论：

- 默认 q64 对 epic 低/中预算更稳，cacheblend `10%` 也更好；q256 + direct value 对 cacheblend `1%/5%` 和高预算有收益，但会伤 epic 低/中预算。默认保持 q64，把 q256/direct-value 作为可调项或后续按 budget 做启发式选择。
- 低预算 `1%/5%/10%` 已经比普通 OpenCL full compute 快；`20%` epic 接近或略快，cacheblend `20%/30%` 仍低于 normal baseline。
- profile 确认 `layer 0` 是普通 `PagedAttention` full prefill；`layer 1` 是 `PicScoreAttention`，形态为 full K/V write + compact output；`layer >= 2` 是 `PicSparseAttention` compact rows，并已进入 `op=sparse_prefill_attention_fast_qk_softmax_qkv`。
- hydrate 走 mapped PagedCache direct path，profile 中 `direct_tokens=1441 fallback_tokens=0`。不要再引入中间 KV staging。
- 当前主要优化点是 `PicScoreAttention` 仍走 row/generic correctness path，第一层 `PicSparseAttention` 可能有 OpenCL 编译/调参开销，高预算下 QK active tiles 接近满 K 范围。
