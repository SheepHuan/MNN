---
name: mnn-pic-optimize
description: 当用户要求分析或优化 MNN PIC/PagedAttention 的 OpenCL/CUDA/Adreno 稀疏 prefill、score layer 前后 full compute 与 sparse compute 边界、PagedAttention/attention kernel 热点、跨设备性能回归或优化日志维护时使用。
---

# MNN PIC Optimization

本 Skill 用于维护 MNN PIC/PagedAttention 的优化工作流，覆盖 OpenCL、CUDA、Adreno/Rhino 等后端。重点仍然是先保证 graph-boundary / PagedCache 语义正确，再做 attention、dense、layout、tuning 和跨设备回归优化。

## 日志约定

优化日志不再维护单个 `OPTIMIZATION_LOG.md`。统一按 `log/<device>/<hour>/` 存储，`hour` 目录名格式固定为 `YYYY-MM-DD-HH`：

- `.codex/skills/mnn-pic-optimize/log/README.md`
  - 已迁移日志目录索引。
- `.codex/skills/mnn-pic-optimize/log/<device>/<hour>/README.md`
  - 当小时摘要、重点结论和章节入口。
- `.codex/skills/mnn-pic-optimize/log/<device>/<hour>/context.md`
  - 长上下文、实验细节和原始结论。

当前设备目录至少覆盖：

- `orangepi`
- `jetson`
- `rhino`

## 设备参考

除按日期归档的 `log/` 外，稳定的设备结论放在 `references/`：

- Rhino / Adreno 正式默认口径、当前错误路由、端到端目标和下一步工程动作：
  - 读取 `.codex/skills/mnn-pic-optimize/references/rhino.md`
  - 详细实验明细再回看 `.codex/skills/mnn-pic-optimize/log/rhino/2026-06-27-00/context.md`

新增优化记录时：

1. 先按设备选择目录，再按小时创建 `log/<device>/<YYYY-MM-DD-HH>/`。
2. 把短摘要写进当小时目录的 `README.md`。
3. 把长实验上下文、命令、profile 和结论写进当小时目录的 `context.md`。
4. 稳定结论再回收进本 Skill；不要重新恢复单文件 `OPTIMIZATION_LOG.md`。

## 入口约束

1. 从 MNN 仓库根目录工作，先读 `AGENTS.md`。
2. 修改源码前运行：

```bash
git status --short
```

3. 不要读取或修改 `schema/private/`、`source/internal/`。
4. 构建产物、日志和实验输出放到 `.cache/`、`output/` 或用户指定目录。
5. OpenCL 远端实验的 cache 路径必须固定，不要为同一设备维护多个 cache 根路径，也不要通过环境变量在运行时切换 cache 目录。当前设备固定路径如下：

```text
OrangePi: /mnt/ssd/code/.cache/mnn_opencl_pic
Rhino:    /mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep
```

不要再增加备用 cache 路径、不要在正式流程里引入 path override。OrangePi 当前也不再使用 `/mnt/ssd/.cache/...` 这类第二路径。

OrangePi 设备侧所有生成数据都必须落到 SSD `/mnt/ssd/code/.cache/mnn_opencl_pic` 下，包括 artifact、prefill sweep raw/summary、OpenCL runtime cache、shared KV、dataset bench 输出和日志。不要把产物或 benchmark 数据写到 OrangePi 根分区 `/home/orangepi/...`；根分区只允许保留已有源码/配置引用，不作为测试输出目录。

OpenCL 的 MNN runtime/autotune cache 是设备级共享文件，不按模型、context、ratio 或 normal/PIC 测试情况分叉。普通 `llm_bench` baseline、PIC server warm、cacheblend、epic 和 full-reuse 都必须落到同一个 `<remote_cache_root>/runtime_cache/opencl/mnn_cachefile.bin`。normal bench 通过 `MNN_LLM_RUNTIME_CACHE_DIR` 指向该目录；为了兼容旧 `llm_bench`，模型目录下的 `tmp` 只能作为指向该目录的 symlink。已有 model-local `tmp/mnn_cachefile.bin` 只能在统一 cache 缺失时复制过去并备份旧目录，不要清理或新建第二份 cache。

6. 构建、同步和跑 PIC server benchmark 时，按需读取：

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

## Decode 算子优化约束

OpenCL PIC decode 的优化目标是形成统一的新算子实现思路，而不是用按模型、head 数、设备或 q 长度的路由回退把端到端 TPOT 凑到不回退。`old` / identity PagedCache attention 只能作为 baseline 或显式 A/B 对照；生产默认不得自动根据 `llama3.2-3b`、`minicpm5-1b`、`qwen3-4b`、Mali/Adreno、`num_heads/kv_heads` 或单个 q 值切回 old。

允许的收敛手段只有三类：

- 优化新算子实现：例如 q=1 的 fused append+attention、record queue / fixed dispatch、decodeKey 写入合并、lane/local memory、QK/QKV 拆测后针对瓶颈改 kernel。
- 调整参数或 autotune：例如 lane 32/64/128、q_tile、workgroup、local memory 布局、设备级 tune cache；参数可以按设备或 shape tune，但不能表达成“这个模型走 old、那个模型走 new”。
- 提供显式实验变体：例如 fused、append+readonly、QK-only、QKV-only、split profile。变体必须用清晰 env 或 benchmark 标签运行并单独报告，不能作为默认自动 fallback。

q=1 和 q>1 可以有不同 kernel 家族，因为 workload 不同：q=1 优先解决固定开销、append+attention 融合和 launch/record queue；q>1 必须做真正 qtile，让一个 workgroup/tile 共享同一段 K load，不能退化成多个 row 独立扫 K。若 `new_attention_only` 赢但端到端输，继续拆 `prepare/append/rank/launch/QK/QKV/V` 并修实现细节；若 qtile 在某设备或模型上回归，继续调 q_tile/lane/local memory 或新增 qtile 变体，而不是自动切回 old。

decode-transposed K 的 prepare 边界是硬约束：prefill/full-reuse 后显式 prepare 历史 K，decode 计时内只允许 append 当前新 token 的 decodeKey。profile 中出现 `decode_prepare_inside_decode=1` 时，该实现直接判为不合格。

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

PIC server 图级 op profile：

```bash
MNN_PIC_GRAPH_PROFILE=1
MNN_PIC_GRAPH_PROFILE_TOP=1000
```

这个 profile 只用于 attribution，不作为正式 latency：server 会开启 MNN debug callback，并在每个 op 后等待输出完成，打印 `MNN_PIC_GRAPH_PROFILE_TYPE` 和 `MNN_PIC_GRAPH_PROFILE_OP`。它适合确认 score layer 后 dense 图是否按 active rows 运行，例如 `inputs=[519x2048x1x1]` 表示 compact active rows，`inputs=[1024x2048x1x1]` 表示 full prompt rows。正式性能报告不要开启这个开关。

重点看这些日志：

- `op=prefill_attention_fast_qk_softmax_qkv`：full prefill attention。
- `op=cacheblend_score layer=N`：score/top-k 所在层。
- `op=hydrate layer=N`：持久 PIC cache 源直接 hydrate 到当前请求 PagedCache。
- `op=score_flash_attention layer=1`：当前默认 score layer full-Q/compact-output flash-style attention；`full_q=1` 时按 active logical index 读 full query row。
- `op=sparse_flash_attention layer=N`：当前默认 later `PicSparseAttention` compact-Q sparse recompute attention。
- 历史日志里的 `op=sparse_prefill_attention_fast_qk_softmax_qkv` 是已移除的三段式 sparse QK / sparse softmax / sparse QKV 旧路径；不要再围绕它新增调参。

解释日志时要注意：

- `layer=N` 是真实 PagedAttention layer index，不是 profile parser 的序号。
- 若 `cacheblend_score layer=1` 后马上出现 `sparse_prefill... layer=0`，这是语义红灯：sparse recompute 被错误地从 layer 0 开始执行。
- profile detail 会在每个子 op 后 `queue.finish()`，只用于拆瓶颈，不作为正式 latency。
- `qk_active_tiles / qk_rect_tiles` 接近 1 时，说明 selected Q 的 logical position 很散，虽然 Q 行少了，但每个 Q chunk 的 K 方向仍接近满长。

## 优化优先级

1. 先修正执行语义：`sparse_start_layer = score_layer_idx`。score layer 之前 full compute；score layer 当层先 full-K/V scoring，然后立刻裁剪 active hidden states，从本层开始 sparse。不要让 `forwardVec(selected_tokens)` 从 layer 0 开始触发 sparse query。
2. 再把 OpenCL LWS tuning 移出正式计时请求，而不是禁用第一次 tuning。MNN 已有原生 OpenCL Autotuning cache：`RuntimeManager::setCache(...)` 加载 `mnn_cachefile.bin`，`RuntimeManager::updateCache()` 写回 `AutotuningT`。第一次遇到未覆盖的新 kernel / shape / local work size 时，warm 阶段必须允许 OpenCL 编译 program、尝试 LWS 候选并写回 cache；不要为了缩短 warm 而跳过必要的首次编译或伪造固定 LWS。不要在 PIC 正式请求路径里维护另一套在线 tuner。新增 sparse/PIC OpenCL kernel 默认接 `localWS2DDefault` / `localWS3DDefault` 和 MNN cache；固定 LWS 或禁用 tune 只作为 A/B 诊断开关。正式 OrangePi sweep 的正确流程是 normal baseline 先 warm 目标 prompt shape 并由 `llm_bench` 写回 runtime cache，PIC server 再 warm 目标 sparse shapes 并调用 `/v1/tune/update_cache`，然后在正式计时中复用同一份 MNN cache；冷 shape tuning 不能算作 cacheblend/epic request latency。OpenCL cache 根路径按设备固定，但 PIC 手写 tune key 必须带设备族 namespace，至少区分 `mali` 与 `adreno`，避免两类 GPU 互相回放 sparse family / schedule / variant 选择。PIC server 的 MNN OpenCL runtime/autotune cache 不按模型分叉；OrangePi 固定使用 `/mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/runtime_cache/opencl/mnn_cachefile.bin`，Rhino 固定使用 `/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/runtime_cache/opencl/mnn_cachefile.bin`，`shared_kv/<model>/` 只用于持久 text KV cache。warm 成功后写回，后续同设备 / 同 driver / 同 artifact 的 MNN OpenCL 运行应加载这份 cache，只有未覆盖的新 model/context/ratio shape 才需要新增 warm/tune。
3. 再看 sparse attention 热点。通常 QK 是主热点，QKV 次之，hydrate/disk 往往不是第一瓶颈。
4. 如果第一个 sparse layer 的 QK 比后续层慢数秒，优先怀疑 OpenCL `localWS3DDefault` 调参或首次执行开销。生产口径用 MNN tune cache/prewarm 解决；固定/禁用 LWS 只作为 A/B 实验开关，不作为长期生产路径。
5. 如果 `qk_active_tiles / qk_rect_tiles` 很高，优化方向是让 sparse Q 按 logical position 分组、缩小 K 有效范围，或设计按 Q range 的分段 QK，而不是继续优化 hydrate。
6. OpenCL sparse attention 当前生产路径是 fused sparse FlashAttention：
   - `PicScoreAttention layer=1` 使用专门的 full-Q/compact-output `score_flash_attention` 路径：先完成 full prompt K/V 写入和 score/top-k，再用 active logical indices 读取 full query row 并输出 compact rows。不要把它理解成 compact-Q later-layer kernel。
   - `PicSparseAttention layer>=2` 使用 compact-Q `sparse_flash_attention` 路径。
   - cacheblend/score-ready sparse flash 默认使用 single-piece 调度，`q_chunk=activeLen`、`q_split=1`，减少 fused row flash 的 per-layer launch 数；epic / 非 score-ready sparse flash 保留 range-aware q64 pieces，避免低预算路径回退。
   - sparse flash row32/row64 的 output accumulator 默认放在 private `COMPUTE_FLOAT8 o0..o7` 中，K loop 结束后再写入 `local_o` 做跨 lane reduction，避免在热循环里反复读写 local memory。
   - sparse flash 同时保留 row32/row64 两个内核，默认由内置形状/plan 启发式选择；不要重新增加请求期 env fallback。
   - cacheblend selected PIC ratio `>=50%` 默认使用 row64，因为 row32 在 1024-token cb50 正式 repeat 中会让 cacheblend 低于 normal baseline；其他高预算 sparse layers 可使用 row32。
   - 不保留 row64/row32 的 `local_q` Q-cache 变体。实测它在 detail profile 中偶尔降低 flash 子耗时，但会让 cacheblend50 端到端退化；不要重新引入。
   - OpenCL kernel 不要硬编码 `half4` / `float4` 来表达精度策略；用 `FLOAT*` 表示存储类型，用 `COMPUTE_FLOAT*` 和 `CONVERT_COMPUTE_FLOAT*` 表示计算类型。MNN runtime 会按 precisionLevel 决定 fp16-storage/fp32-compute 或 fp16-compute。
   - 旧的三段式 sparse QK / sparse softmax / sparse QKV OpenCL kernel 和 env-gated fused softmax-QKV 实验路径已移除；不要再新增开关回退到旧路径。
   - 若进一步限制 cacheblend 选择窗口或重新排序 active rows 会改变算法语义，必须作为 `cacheblend-windowed` / `cacheblend-prefix-regularized` 这类命名变体报告，不能算普通 cacheblend。
7. score layer 后的 compact-row dense Linear 在 OpenCL low-memory 1x1 Conv 路径中默认使用 PIC 专属 fast path：
   - kernel 名称固定为 `pic_gemm_b4_c8_int4_buf` / `pic_gemm_b4_c8_int8_buf`，只用于大通道、小 M compact rows，例如 1024-token sparse recompute 的 1%-50% budget。
   - 这个 fast path 的主要价值是给 PIC compact dense shapes 独立 MNN tune namespace：`pic_m<active_rows>` 会进入 LWS key，避免普通 `gemm_b4_c8_*` / `convBufLowMemory_*` 历史 tune 让 216-row 这类 shape 掉到慢路径。
   - MLP 是后续优化的关键线索：score layer 后每层 `mlp/gate_proj`、`mlp/up_proj`、`mlp/down_proj` 都是 compact-row 大 Linear，1%-50% 不同重算比例下都必须确认它们被优化，而不能只看 `PicSparseAttention` 是否变快。
   - 1024-token 当前 active rows 约为 prelude/suffix 加选中 PIC token；1% 也通常超过 16 rows，因此应进入 `pic_gemm_b4_c8_*` 而不是 tiny GEMV 分支。若未来模型/模板导致 active rows <=16，需要单独记录为 tiny-MLP fast path 问题。
   - 不要新增 env fallback 或请求期在线 tuner；正式测试仍然先 warm 目标 ratios，再 `/v1/tune/update_cache`，计时请求复用 MNN cache。
8. Jetson CUDA 按同一套 graph-boundary 语义适配，但 dense fast path 不能照搬 OpenCL：
   - CUDA `PagedAttentionExecution` 必须注册 `PicScoreAttention` / `PicSparseAttention`，拆分 `kvWriteLen` 和 `attnLen`，score layer 写出 `active_indices`，后续层按 compact active rows 计算并用真实 logical slot 写 PagedCache。
   - 不要新增朴素 packed INT4 CUDA `PicGEMM` 作为 PIC compact MLP 快路径。2026-06-11 Jetson A/B 证明它会让 cacheblend20 从 CUTLASS 路径的 `0.816942s` 退化到 `3.527971s`，216-row compact MLP 变成第一瓶颈。CUDA 目前默认保留 tensor-core CUTLASS + runtime dequant，直到有真正的 compact tensor-core GEMM 并在 1%-50% 全部更快。
   - 不要把现有 `v2_row_compressed_mask` 或 sparse single-piece qSplit 当作 CUDA sparse flash 替代品。Jetson A/B 证明 row-compressed fused kernel 会让 cacheblend20/50 退化到 `1.548174s`/`2.303546s`；single-piece qSplit 会让 cacheblend50 退化到 `1.860186s`。CUDA P0 必须是 tile-based sparse flash，而不是单 row flash 或单纯减少 split。
   - CUDA 当前默认使用 q8/k32 tile sparse flash，`attnLen >= 384` 的高预算 sparse tile 使用 q32/k32 wide-Q 变体：epic / fixed active-plan sparse rows 始终走 tile flash；cacheblend sparse rows 只有 `attnLen > 256` 才走 tile flash，`attnLen <= 256` 保持现有 QK/softmax/QKV 路径。tile flash 必须按每个 Q tile 的 `max_q_logical + 1` 限制 K/V streaming，跳过 causal 之外的整块 K/V；不要保留全长 K loop 的旧 tile-flash 路径。
   - cacheblend 预算不能只按 row count 估算。Jetson 1024-token cacheblend 10%-50% 的 selected rows 虽然是 101-505 行，但 max logical position 到 1006-1009，平均 causal K 仍是 full 的约 65%-75%；epic 同预算只有约 6%-25%。后续 cacheblend 优化要看 active logical distribution 和 causal K work，而不是只看 recompute ratio。
   - 不要把 cacheblend high-budget tile flash 改成 q4。2026-06-11 Jetson A/B 证明 q4 会让 cacheblend30/50 退化到 `1.289018s`/`1.887966s`；当前默认是 q8/k32 + 高预算 q32/k32 hybrid。
   - 不要把 V14_MB packed GEMV 扩到 8/16/32 行作为低预算默认路径。Jetson A/B 证明 batch<=32 GEMV 会让 cacheblend1/epic1 从 `0.354654s`/`0.318615s` 退化到 `0.543940s`/`0.521939s`。低预算 compact dense 仍默认走现有 CUTLASS 路径。
   - CUDA 下一步可做两条并行：其一是更细的 cacheblend 专用 sparse flash，内部按 logical position bucket active rows 并 scatter 回原 compact row order；其二是继续做 compact dense/activation 融合，因为 q32+k-slot、SM70 compact CUTLASS 和 half2 SiLU 后 cacheblend50 仍由 `PicSparseAttention` 与 compact `Convolution` 共同决定，epic50 仍偏 dense-bound。继续调 epic/fixed tile flash 的 `Q_TILE/K_TILE`、barrier、score staging 和 half/half2 load 只能算 P1/P2 微优化。任何默认替换必须覆盖 1/5/10/20/30/40/50 并保持 cacheblend/epic 都快于 normal。
   - CUDA compact dense 下一步先拆 profile，不要直接上 dequant-cache。当前 low-memory weight-only 1x1 Conv 会在每次 execute 中把 INT4 weight dequant 到 DYNAMIC FP16 buffer 再跑 CUTLASS；这可能解释不同重算比例下 MLP latency 不按 active rows 线性下降。必须先分别计时 `DequantizeInt4Weight` 与 `runCutlassGemmFunc()`，确认收益后才能默认替换，并覆盖 1%-50% sweep。
   - 2026-06-11 Jetson profile 已确认 compact MLP dequant 是真成本但不是最大项：cb50 dequant 约 `60.9 ms`、epic50 约 `50.9 ms`，占 Conv profile `11%-14%`；M=519 的 gate/up/down 每个 Linear 大约 `0.85-0.93 ms` dequant + `6.3-7.0 ms` CUTLASS。下一步 P1 不应盲目全静态 dequant cache，而应做内存可控 cache 或 fused dequant+tensor-core/weight-only tensor-core GEMM，目标是同时减少 dequant 和 compact GEMM。
   - CUDA compact dense 默认使用 memory-capped static FP16 dequant cache：`ConvFpAIntBExecution::Resource` 对 low-memory INT4 1x1 Linear 在构造期预反量化一次，cap 为 `min(2 GiB, totalGlobalMem / 16)` 且不走 env 开关；超过 cap 或非目标 precision 仍保留 DYNAMIC runtime-dequant 作为内存安全路径。2026-06-11 Jetson 1024-token profile 显示 `runtime_dequant=1` 行数为 `0`、`static_cache_total=1946157056 bytes`，cb50 `Convolution` 从 `471.301 ms` 降到 `409.255 ms`。1/5/10/20/30/40/50 sweep 中 cacheblend 和 epic 全部快于 normal。
   - CUDA compact dense 高预算默认使用 SM70 compact CUTLASS hybrid fast path：在 static FP16 dequant cache、fp16、无 activation 的 low-memory INT4 1x1 Linear 上，当 compact row `M in [384,768]` 且 `K >= 1024` 时使用 tensor-core Linear 变体；`384 <= M < 512` 走 `128x64x64`，`M >= 512` 走 `64x128x64`。2026-06-11 Jetson 1024-token sweep 显示 `M>=512` hybrid 后 cacheblend50 `0.753021s -> 0.721868s`、epic50 `0.635977s -> 0.606127s`，同时 cacheblend/epic 1%-40% 保持基本不退且全部快于 normal。不要把 `64x128x64` 扩到 `M < 512`：实测全量使用该 tile 会让 cacheblend40 从约 `0.636s` 退到约 `0.696s`。不要把 compact path 扩到 `M < 384`，除非重新覆盖 1/5/10/20/30/40/50 并确认低预算不退化。
   - CUDA FP16 contiguous `UnaryOpOperation_SILU` 默认使用 half2 fast path。2026-06-11 Jetson 1024-token sweep 显示 cacheblend50 `0.818838s -> 0.753021s`、epic50 `0.705095s -> 0.635977s`，profile 中 `UnaryOp` 从约 `74 ms` 降到约 `9 ms`，compact per-layer SiLU 从约 `4.16 ms` 降到约 `0.34 ms`。这不是完整 `SiLU(gate) * up` fusion；后续若继续做 activation，应聚焦 graph-level fusion 去掉 BinaryOp/中间 tensor。
   - Jetson sm72 上不要把真正的 INT4 tensor-core GEMM 当成默认可落地方向。更现实的 CUDA dense 路线是内存可控 FP16 dequant cache、fused dequant+FP16 GEMM、或 graph/export 级 gate_proj+up_proj 合并来改善现有 CUTLASS compact-row shape。
   - CUDA sparse flash tile 默认保留 per-Q-tile causal K limit，并且最后一个 K tile 也必须避免加载 `logical >= causalKLimit` 的 K/V。该 tail K/V cut 在 cb50 profile 中把 `PicSparseAttention` 从约 `464.7 ms` 降到约 `454.7 ms`，属于可保留的小收益。
   - CUDA sparse flash tile 默认把每个 K tile 的 logical slot 解析到 shared `kSlotShared[]`，K/V 两段加载复用同一份 slot，避免按 head_dim 重复读 `slotTable`。2026-06-11 q32 + k-slot-cache + `attnLen >= 384` sweep 显示 cacheblend40 `0.859711s -> 0.755525s`、cacheblend50 `0.871012s -> 0.872416s`、epic40 `0.666852s -> 0.636039s`、epic50 `0.757372s -> 0.757277s`，1%-50% cacheblend/epic 全部快于 normal。
   - 不要把 CUDA sparse flash tile 的 `384 <= attnLen < 512` 中间预算改成 q16/k32。2026-06-11 Jetson A/B 显示该变体让 cacheblend40 `0.634798s -> 0.720326s`、epic40 `0.514860s -> 0.522904s`，同时 50% 也有噪声/退化；当前默认仍是 `attnLen >= 384` 用 q32/k32。
   - 不要把 CUDA sparse flash tile 的 output accumulator 从 shared `outShared[Q_TILE][HEAD_DIM]` 改成每线程 `outAcc0/outAcc1` register 变体。2026-06-11 Jetson cb50/epic50 profile 显示该变体把 `sparse_flash_tile_attention` 合计从已接受口径约 `362 ms` 拉高到 `441 ms`，端到端 cacheblend50 `0.807s -> 0.982s`、epic50 `0.664s -> 0.776s`；推测寄存器压力/occupancy 退化超过 shared-memory 节省。
   - CUDA sparse flash tile 在 `llm_config.attention_mask=float` 且 `attention_type=full` 的 PagedAttention 模型上默认跳过 additive causal mask 读取：kernel 已经用 `active_indices` 的真实 logical position 做 `logical <= qLogical` 和 per-Q-tile causal K limit。不要把这条推广到 sliding/mix attention；那些 mask 不是冗余的。
   - CUDA cacheblend active logical indices 在 graph-boundary active row 构建时已经排序；不要假设简单 sort/bucket/scatter 会带来大收益。若继续优化 cacheblend sparse flash，优先做 adaptive q-piece：只在 q8 tile 跨大 logical gap 时拆小 piece，保持输出顺序不变；不要全局切 q4。
   - CUDA cacheblend score 默认使用当前请求 PagedCache source slots：在 mapped PagedCache 上把持久 PIC value 段直接读入 `mCache->mappedValue` 的保留 source slots，score kernel 在同一份 `mCache->value` buffer 中比较 reference logical slots 和 source slots，不再使用 `std::vector<int8_t>` + H2D value workspace。这里的“zero-copy”指不再额外拷到 CUDA workspace；磁盘读仍落到 CPU mapped pinned memory，不是 GPUDirect Storage。
   - CUDA cacheblend score/top-k profile 已拆分：`alloc/read/h2d/score_kernel/topk_init/topk/d2h/metadata`。1024-token cb50 source-slot path 已确认 `source_slot_tokens=1010`、`h2d_us=0`，score 子路径约 `3.5 ms`；它是小收益清理，不是主要瓶颈。
   - 当前 CUDA `cacheBlendTopKKernel` 使用 device `used` bitmap 保持确定性 top-k，不再按 `prev < k` 扫 selected；cb50 `topK ~=505` 时 `topk_us` 仍约 `1.8 ms`，后续可优化但优先级低于 sparse attention/compact dense。
   - CUDA q8/q32 k-slot-cache tile flash 之后，epic50 的第一瓶颈已经是 compact dense/MLP；cacheblend50 是 sparse attention 与 compact dense 并列，且 `PicScoreAttention` 还有非 attention 成本。后续汇报不要只给 attention 数字，必须同时列 `PicSparseAttention`、`PicScoreAttention`、`Convolution`、`UnaryOp`。
   - CUDA fresh gate/up packed 不能只凭图上少一个 `gate_proj/up_proj` launch 判断可进生产。2026-07-01 Jetson A/B 证明，修复 `/v1/prefill/text` 后的 fresh `PicLinearNhwcWeightOnly + PicPackedSiluMul` 在 Llama3.2 1B full-reuse decode `x=0/1/3/5/7`、contexts `512/1024/1536` 上仍比旧 `silumul-score10` 慢约 `+0.36/+2.65/+2.57/+2.46/+2.38 ms/token`，且 `x=1 -> x=3` cliff 仍约 `+7.7 ms/token`。任何 fresh CUDA gate/up packed 导出必须先通过 `/v1/prefill/text` 持久 text cache 构建，再用真实 endpoint decode TPOT 同条件打赢当前 baseline，才能替换生产路径。
   - CUDA decode repair 的 `x=1 -> x=3` cliff 先查 weight-only dense 路由，不要默认归因到 attention。2026-07-01 Jetson graph profile 在 MiniCPM5-1B、Qwen3-4B、Llama3.2-3B 上都显示，`x=1` 的 active rows 是 2，走 `conv_fpa_intb_1x1_tiny_gemv batch=2`；`x=3` 的 active rows 是 4，触发 `conv_fpa_intb_1x1_rows45_cublas batch=4` 或相近 batch-4 Conv 路径。按单 TPOT 归因，三模型 `x=3-x=1` 的组件增量几乎都由 `Convolution` 贡献，MLP gate/up/down 与 attention q/k/v/o projection 是主因，`PicSparseAttention` 不增加或略降。`x=3/5/7` 接近平台期也符合 rows4/6/8 已在同一 dense 路由族内的现象；Qwen3-4B 还暴露出部分 batch-4 Linear 因 static dequant cache 覆盖不足而落到 runtime-dequant generic Conv。
   - CUDA fresh PIC 导出还必须用当前 schema/converter 验证自定义 op type，防止 `PicScoreAttention` / `PicSparseAttention` 在 JSON-to-MNN roundtrip 后变成 `type=-1`。这类错误发生在 text cache prefill 阶段，不是 decode kernel 问题；不得用手工修过的 `llm.mnn` 作为生产推广依据。
   - CUDA 正式 sweep 至少覆盖 10/20/30/40/50；如果修改 compact MLP 或 graph-boundary row shape，补 1/5 低预算 smoke。
9. direct PagedCache value prefill 现在只保留为内置窄 heuristic：cacheblend 高预算且 slot table identity 时才直接读 PagedCache value，避免 packed V staging；不再提供 `MNN_PAGED_ATTENTION_OPENCL_DIRECT_VALUE_PREFILL` 强制开关作为生产路径。
10. 不要用 `MNN_PAGED_ATTENTION_IMPL` 或 V2 路由影响生产路径；bench ops 可以保留 V1/V2 对比入口。

## 验证要求

每次改动后至少确认：

- 要进入默认路径、正式性能结论或 `benchmark.csv` 正式更新的改动，至少补 `jetson + orangepi` 两台设备的 prefill sweep。OrangePi 用来验证 OpenCL 路径，Jetson 用来验证共享 PIC/PagedCache 语义和 CUDA 路径没有被连带打坏；Rhino 只能作为额外 profile，不替代这两个正式设备。脚本层面同样按这个规则执行：formal prefill sweep 默认要求 `--devices jetson,orangepi`，Rhino 只有显式 `--allow-extra-device` 才允许加入；单设备 dataset/decode helper 的 run metadata 不应标成 `formal_matrix_complete=true`；`merge_prefill_benchmark_csv.py` 默认也要求这次新 summary 已覆盖 `jetson,orangepi`，单设备或 Rhino-only 合并必须显式 `--allow-nonformal`。
- `cacheblend` / `epic` 的 score layer 之前没有 `sparse_prefill... layer < score_layer_idx`。
- `normal LLM full-compute` baseline 只在普通模型或实现变化时重测；不要为了每个 ratio 反复测 normal。
- 不再把 `PIC full-compute` / `pic-full-recompute` 作为正式补测目标；历史数据可保留作参考，但缺口补测不再为了补齐它而重跑。
- `full-reuse` 和 sparse 模式优先对比 normal LLM full-compute baseline；如果已有历史 PIC full-compute，可作为辅助参考但不是必填指标。
- `cacheblend` / `epic` 每个 ratio 是独立请求，延迟包含本次 scoring/top-k、hydrate、sparse recompute、suffix prefill。
- 对 1%-50% ratio sweep，必须把 MLP 作为单独检查项：至少 spot-check 1/10/20/30/40/50 或当前报告覆盖的全部 ratios，确认 score layer 后 `/layers.* /mlp/{gate_proj,up_proj,down_proj}/Linear` 输入是 compact rows，并且这些 low-memory 1x1 Conv 使用 PIC compact dense fast path / 独立 tune cache。
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
- 若某个较低 budget 反而比更高 budget 慢，先用 `MNN_PIC_GRAPH_PROFILE=1` 看 `Convolution` dense Linear，而不是默认归因到 attention。历史 20% cliff 是 `[216x2048]` / `[216x8192]` compact-row low-memory 1x1 Conv 形状触发的 dense GEMM/LWS 问题，已用 `pic_gemm_b4_c8_*` fast path 修复。
- 若 `Convolution` 在 graph profile 里仍是第一热点，优先拆 MLP gate/up/down，而不是继续微调 sparse flash；MLP compact dense 的性能必须随 1%-50% active rows 平滑增长，不能再出现低 budget 比高 budget 更慢的形状 cliff。

## 2026-06-11 OpenCL 1024-token 当前结论

OrangePi OpenCL PIC graph-boundary 路径在 1024-token prompt 下已经稳定运行。默认 `score_layer_idx=1`，不要改成 layer 0：`layer 0` 是 score 前 full compute，`layer 1` 是 `PicScoreAttention` scoring/top-k 和 compact 输出，`layer >= 2` 是后续 `PicSparseAttention` compact sparse rows。

当前最快默认路径已经包含 score-layer flash、layer>=2 sparse flash private accumulators，以及 compact-row dense `pic_gemm_b4_c8_*` fast path：

```text
tag=opencl_pic_1024_picgemm_formal_20260611_155453
mode,budget,latency_s,speedup_vs_normal
normal,full,6.968234,1.000
full-reuse,full,0.817891,8.520
cacheblend,0.10,2.212856,3.149
cacheblend,0.20,3.290629,2.118
cacheblend,0.30,3.889923,1.791
cacheblend,0.40,5.281809,1.319
cacheblend,0.50,5.769052,1.208
epic,0.10,2.185259,3.189
epic,0.20,3.199813,2.178
epic,0.30,3.829921,1.819
epic,0.40,4.779629,1.458
epic,0.50,5.119875,1.361
```

对应 cb50 profile 口径：

```text
score_flash_attention layer=1 total=186.431 ms flash=110.222 ms full_q=1
later_sparse_flash_attention_total=1783.178 ms
later_sparse_flash_compute=1326.024 ms
hydrate/async/score-topk remain non-bottlenecks
```

下一步优先 P2：补非 attention dense 图 profile，确认 score layer 后的 QKV projection、o_proj、MLP、norm、residual 是否全部只跑 compact active rows。若仍有 full 1024-row dense kernel，这是 cb50 剩余 4s 级差额的首要优化点。

source-slot reserve 修复后的 fresh repeat：

```text
mode,budget,latency_s,speedup_vs_normal
normal,full,6.968234,1.000
full-reuse,full,1.035310,6.731
cacheblend,0.10,4.187917,1.664
cacheblend,0.20,6.196139,1.125
cacheblend,0.30,4.680493,1.489
cacheblend,0.40,5.594163,1.246
cacheblend,0.50,7.188658,0.969
epic,0.10,3.761944,1.852
epic,0.20,5.863076,1.188
epic,0.30,3.706188,1.880
epic,0.40,4.979558,1.399
epic,0.50,6.012580,1.159
```

cacheblend 50% profile-detail 拆分：

```text
PagedAttention_layer0_ms=1720.863
ScoreAttention_scoring_ms=5.783
ScoreAttention_topk_ms=0.863
ScoreAttention_layer1_attention_ms=183.203
SparseAttention_layers_gt1_ms=2958.786
SparseAttention_layers_gt1_qk_ms=1285.347
SparseAttention_layers_gt1_qkv_ms=836.536
Hydrate_ms=63.654
```

结论：

- 稳定性问题已修：`log_scan.json` 中 `target_unavailable=false`、`async_failed=false`，source-slot reserve 后 full-reuse 和后续 cacheblend/epic 不再隐藏 forward error。
- `PagedAttention` / async hydrate / PageCache zero-copy 不是当前主瓶颈。cacheblend 50% hydrate 只有约 64ms，score/top-k 只有约 5.8ms/0.9ms。
- 当前主瓶颈明确是 `PicSparseAttention layers > 1` 的每层 sparse attention，尤其 QK 与 QKV；cacheblend 50% 需要至少再省约 220ms 才能稳定快于 normal full-compute。
- FlashMask-style range-aware pieces 已经默认打开，但 50% active rows 下 `qk_active_tiles/qk_rect_tiles ~= 0.928`，selected rows 覆盖到 1023 附近，4-row group 内 causal K 上界浪费不到 1%。因此单纯“把每段 K 上界收紧”已经不够，后续要优化 sparse QK/QKV kernel 本身，或做会改变选择分布的命名变体。
- 2026-06-11 已否定的快速 heuristic：手动 direct-value、`q_chunk=16`、`q_chunk=128`、`q_chunk=256 + direct-value` 都没有让 cacheblend 50% 稳定快于 normal；不要恢复这些可调开关。env-gated fused softmax+QKV V0 在 50% active rows 上触发过 OpenCL `CL_OUT_OF_RESOURCES (-14)`，该旧实验路径已被移除，不作为性能结论。

已实现并默认启用的 sparse FlashAttention：

- 只覆盖 `layer >= 2` 的 `PicSparseAttention` compact-Q；`layer=1` score layer full-Q/compact-output 仍走现有 sparse fast correctness path。
- 新 kernel `sparse_flash_attention_row64` 用 64-lane workgroup 处理一个 active row/head，在线合并各 lane 的 softmax `(m,l,o)`，每个 K 的 QK 只计算一次；输出 accumulator 使用 `float8` 向量 local storage，避免 V0 scalar local-memory 循环。
- profile op 单独打印为 `op=sparse_flash_attention`；不再需要设置 sparse flash 环境变量。
- 旧的普通 OpenCL sparse kernel 已从生产代码移除：不再构建 `rearrange_sparse_q`、`matmul_qk_sparse_prefill_piece`、`matmul_qkv_sparse_prefill_piece` 或 `matmul_softmax_qkv_sparse_prefill_piece`。
- direct PagedCache value prefill 只保留窄 heuristic：cacheblend 高预算 sparse layers 且 slot table identity 时自动启用。

正式验证：

```text
tag=opencl_pic_1024_sparse_flash_vec8_formal_20260611_130036
note=当时通过 env 开启；当前源码已默认使用同一 sparse flash 逻辑
normal,full,6.968234,1.000
full-reuse,full,0.764264,9.118
cacheblend,0.10,3.780810,1.843
cacheblend,0.20,6.638145,1.050
cacheblend,0.30,3.903327,1.785
cacheblend,0.40,4.756370,1.465
cacheblend,0.50,5.834781,1.194
epic,0.10,4.015643,1.735
epic,0.20,5.884426,1.184
epic,0.30,3.889955,1.791
epic,0.40,4.398280,1.584
epic,0.50,5.346845,1.303
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
```

默认路径验证：

```text
tag=opencl_pic_1024_sparse_flash_default_repeat_20260611_052347
server_env=<none for sparse flash/direct-value>
normal,full,6.968234,1.000
full-reuse,full,0.854002,8.160
cacheblend,0.10,4.116973,1.693
cacheblend,0.20,6.318665,1.103
cacheblend,0.30,4.174995,1.669
cacheblend,0.40,5.634095,1.237
cacheblend,0.50,6.424953,1.085
epic,0.10,3.795829,1.836
epic,0.20,5.948663,1.171
epic,0.30,3.831606,1.819
epic,0.40,4.720560,1.476
epic,0.50,5.842008,1.193
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
```

默认路径 profile smoke：

```text
tag=opencl_pic_1024_sparse_flash_default_cb10_profile_20260611_052724
old_sparse_prefill_rows=0
sparse_flash_layers=2..15
score_layer_row: op=row layer=1 full_q=1 us=99893
sparse_flash_total_ms=192.641
```

当前 TODO：

1. 继续优化 score layer `layer=1` 的 full-Q/compact-output attention；不要把它混入 compact-Q sparse flash 路径。
2. 继续缩小默认 repeat 与早先 env-gated sparse-flash best run 的波动差距，优先看 tune cache、score-layer row path 和 full/suffix prefill timing。
3. 继续记录 selected logical rows 的覆盖范围、`qk_active_tiles/qk_rect_tiles`、每层 flash/QK/QKV 时间；只有在不改变 active rows 的情况下减少真实 QK 工作量，才算普通 cacheblend 优化。
4. 如果要限制 cacheblend selected rows 的窗口、前缀正则或重排以降低 K range，必须命名为 `cacheblend-windowed` / `cacheblend-prefix-regularized` 等语义变体，不能报作普通 cacheblend。

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

## 2026-06-11 CPU 精度恢复：显式 token 边界

CPU cacheblend/full-reuse 曾出现输出碎片、拒答和 full-compute 与 no-PIC 不一致。根因不是 CPUPagedAttention 数学本身，而是 `pic_server` 旧的 placeholder 文本路径把 prompt 切成 `preludeText + docText + suffixText` 分别 tokenize，再拼 persistent doc token；BPE 边界和 BOS trimming 会让 PIC full-compute/reuse/cacheblend 实际输入 token 序列不同于普通完整 prompt。这个错误会让 full-compute 也坏，因此不能用它判断 Attention 精度。

修复边界改为显式 token 协议：

- `/v1/prefill/text` 可以直接接受 `token_ids`，并用这段 token 构建持久 text cache；不要强制由 server 重新 tokenize 原始文档文本。
- `/v1/chat/completions` 可以直接接受 `full_prompt_token_ids` / `prompt_token_ids` / `input_token_ids`，有显式 token 时 `messages` 不再必需。
- PIC 请求必须显式标记 doc cache token span：`doc_cache_spans: [{"prompt_start": P, "source_start": S, "token_count": N}]`，或兼容别名 `pic_token_start/token_count`、连续 `pic_token_indices`。
- server 只做硬校验：`full_prompt_token_ids[P+i] == text_cache.token_ids[S+i]`。不相等立即报错，不再猜 placeholder 边界，不再 decode cache token 拼文本，不再悄悄 fallback 到错误 CPU 路径。
- no-PIC baseline、PIC full-compute、full-reuse、epic/cacheblend 都应使用同一份 `full_prompt_token_ids`。full-compute 不读取磁盘 KV，但 prefill 输入 token 必须完全相同；full-reuse/sparse 只把显式 span 对应 token 替换成持久 PagedCache KV。

本地 x64 CPU 验证结果：

```text
explicit span: prompt_start=5, source_start=0, token_count=36, full_prompt_token_count=59
no-pic:         The project codename is Cobalt Lantern and the reusable object is the persistent PIC
full-compute:   The project codename is Cobalt Lantern and the reusable object is the persistent PIC
full-reuse:     The project codename is cobalt lantern. The reusable object is the persistent PIC
epic 0.1:       The project codename is cobalt lantern and the reusable object is the persistent PIC
cacheblend 0.1: The project codename is cobalt lantern and the reusable object is the persistent PIC
```

负向校验：故意改坏 span 第一个 prompt token，server 返回：

```text
Doc cache span token mismatch at prompt index 5: prompt token 32716 != cache source token 32715
```

## 2026-06-11 持久 KV RoPE metadata 硬标准

CPU / CUDA / OpenCL 导出的每层 `.json` sidecar 和 `meta.json.kv_layout` 必须包含 `rope_attention_scaling`。hydrate canonical_no_rope key 时，CPU/CUDA/OpenCL 都会按当前 logical slot 重新施加 RoPE，并乘以这个 attention scaling；导出 canonical key 时会除回同一个 scaling。旧 cache 如果没有该字段，`pic_server` 读取 text cache 时必须硬错误并要求重建，不能默认当作 1.0 静默运行，因为这会让非 1.0 RoPE scaling 的模型在 full-reuse/cacheblend/epic 上恢复错精度。

这条 metadata 修复不改变异步加载 KV 的设计：OpenCL 仍先把持久 PIC cache 源读入当前请求 PagedCache 的保留 physical source slots，score 从 reference slots/source slots 比较，hydrate 再从 source slots 写回真实 logical slots。异步预取窗口、zero-copy PagedCache target 和 `MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK` 调试开关语义保持不变。

## 2026-06-11 OrangePi OpenCL cacheblend 稳定性修复

OrangePi OpenCL `pic_server` 已用本机交叉编译验证：

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=48 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

远端 OpenCL 服务使用：

```text
artifact: /mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus
config:   .cache/weight/AI-ModelScope__Llama-3___2-1B-Instruct-pic-boundary/config_opencl_greedy.json
kv dir:   .cache/kvshare/pic_opencl_explicit_20260611_v3
log:      .cache/logs/pic_opencl_explicit_20260611_v3.log
```

显式 token/span smoke 结果：

```text
/v1/prefill/text token_ids: HTTP 200, token_count=36, cache_status=built
full-reuse max_tokens=0:    HTTP 200, protocol=explicit-full-prompt-token-span-v1, token_alignment=explicit_request_verified, execution_mode=native-full-reuse
cacheblend 0.1 max_tokens=0: HTTP 200, protocol=explicit-full-prompt-token-span-v1, token_alignment=explicit_request_verified, execution_mode=native-cacheblend-graph-boundary
```

OpenCL prefetch 正确性要求更新为硬约束：

- 异步读取持久 PIC cache 源必须严格按 layer index 升序组织队列，hydrate 当前 layer 时只能消费同一 `layerIndex` 的任务结果。
- 不需要等待所有 layer 的 PagedCache target 全部注册后才开始预取；只要队首 layer 的 target 已注册且 `maxSlots` 覆盖真实 KV 长度和保留 physical source slots，就可以启动该 layer 的 async read。这样 full-reuse 的 layer 0 计算时，layer 1 可以并行加载；cacheblend/epic 从 `external_hydrate_start_layer_idx = score_layer_idx + 1` 开始同理流水。
- 队列推进时遇到第一个 target 未注册或容量不足的 layer 必须停止，不能跳过它去调度更后面的 layer；后续 layer 注册 target 时再重新推进队列。
- async task 创建时要捕获已注册 PagedCache target 的强引用，不能在线程里重新 lookup 弱引用后失败。
- 不允许为了处理 “direct PagedCache target unavailable” 做失败后同步重试兜底；正确做法是未就绪时不建 async task，让当前 layer 的 hydrate 走已有同步 direct read，后续注册 target 后继续有序预取。
- OrangePi v3 日志确认没有 `target unavailable` / `async persistent PIC cache read failed` / `ERROR`；full-reuse 首次容量变化时 layer0/layer1 是同步 direct hydrate，后续 layer `async_read=1 direct_segments=1 fallback_tokens=0`；cacheblend 在 `cacheblend_score layer=1` 后进入 `sparse_prefill_attention_fast_qk_softmax_qkv`，后续 hydrate 同样为 direct PagedCache。

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
