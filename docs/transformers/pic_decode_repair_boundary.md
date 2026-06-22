# PIC Decode Repair Boundary

本文档约束 PIC decode repair 的实现边界。目标是在不改变 prefill 语义和性能的前提下，优化 decode 阶段 `tokens_per_decode_step` 为 1 到 7 时的延迟。

## 核心原则

1. Decode repair 不做 PIC recompute scoring。

   Decode 阶段只消费 prefill 阶段已经产生的 HKVD/attention rank 或 selected PIC token 顺序。decode 阶段不能重新执行 cacheblend/delta-v/epic/kvshare scoring，不能重新 top-k，不能把 full score vector 拷回 CPU 排序。

2. Decode repair 的算子不能影响 prefill。

   为 decode 新增的 fused op、shape cache、mask 表示、kernel heuristic 和临时 tensor 构造，只能在 decode repair 的 token-id sparse decode 路径生效。prefill 的 graph-boundary sparse recompute、cacheblend scoring、full-reuse hydrate、full-compute prefill 均不得因为 decode 优化改变 graph、op type、score layer、kernel 参数或结果。

3. Prefill 的 score layer 不能作为 decode 执行边界。

   `pic_recompute_score_layer_idx` 是 prefill sparse/cacheblend 的 scoring 边界，也可以作为 prefill 产出 HKVD rank 的来源标记。decode 阶段直接输入 token id，不从 score-layer hidden checkpoint 继续执行，所以它不应影响 decode repair TPOT。若改变 `pic_recompute_score_layer_idx` 会改变 decode 延迟，说明实现仍错误地复用了 prefill scoring 边界作为 decode hidden continuation 起点。

4. Decode 阶段直接输入 token ids。

   每步 decode 的输入应是本步待 repair 的 PIC token ids 加上当前 decode token id。runtime 同时绑定这些 token 对应的 logical positions：repair token 写回原 PIC logical slots，当前 decode token append 到当前末尾 logical slot。kernel 根据 logical slot 更新 selected token 的 K/V，并只把当前 decode token 的 logits 交给 sampler。

   仅调用 `forwardVec(input_ids)` 不够，因为普通 decode 会把所有输入 token 当作连续 append 到当前末尾；repair token 实际属于历史 PIC logical slots。正确输入形态是 `token_ids + logical_indices + per-row visible end`，而不是 hidden concat 或 score-hidden continuation。

## Decode HKVD 选择语义

Decode repair 的 token 选择参考 vLLM PIC server 的思路，但在 MNN 中必须保持 decode-only：

1. Prefill 阶段保存选择依据到内存。

   在 prefill 的指定 layer/head 上保存每一步需要的 attention weight 摘要或 high-attention 区间，同时保存 prefill score layer 得到的 HKVD token rank。这里的 rank 是 decode 后续消费的静态排序依据，不要求 decode 重新生成。

2. Runtime 维护已重算状态。

   对每个 PIC logical token 维护 `recomputed` 标记。prefill sparse/cacheblend 已经重算过的 token 应作为初始已重算集合；decode repair 每步 token-id sparse decode 成功后，再把本步 repair rows 对应的 token 标记为已重算。

3. Decode 每步只做区间内 top-HKVD 选择。

   Decode 阶段根据上一 decode step 或上一 token 对应的 high-attention 区间，从 prefill 保存的 HKVD rank 中筛选落在该区间、且尚未重算的 PIC token，取最多 `tokens_per_decode_step` 个加入本步 token-id sparse decode。若该区间候选不足，可以按预定义策略扩展到相邻区间或全局 HKVD rank，但仍只能消费 prefill 已保存的 rank/weight 信息。

4. Decode 不重新评分。

   上述选择不属于 cacheblend/delta-v/epic/kvshare scoring；decode 不计算新的 attention score vector，不做新的 top-k 排序，不读写 scratch `.k/.v`，也不改变 prefill score layer。

5. Decode 只更新被选 token 的 K/V。

   本步选出的 HKVD token ids 与当前 decode token id 一起进入 token-id sparse decode；kernel 按 logical slot 覆盖这些 selected token 的 K/V。未选中 token 保持原 PagedCache 中已有 K/V。

## 允许的优化

允许做 decode-only 的通用优化：

- 新增 decode-only HKVD selection state，例如 high-attention interval buffer、HKVD rank view、recomputed bitmap 和 per-step selected logical indices。该 state 只能由 decode repair 消费，不能驱动 prefill sparse recompute。
- 新增 decode-only token-id sparse decode 入口。该入口接收 `token_ids`、`logical_indices`、当前 decode row index 和 visible-end 信息；不接收 score hidden，不拼接 hidden rows。
- 新增 decode-only fused op，例如 tiny-row `SiluMul`、RMSNorm、MLP 局部融合、gate/up 合并、mask/position 轻量输入。
- 在 CUDA/OpenCL decode kernel 中使用 `q_max_visible_position` 或 per-row visible end 代替完整 float attention mask，只要该表示只接入 decode repair token-id sparse decode，不改 prefill attention mask 语义。
- 为不同 `tokens_per_decode_step` 设置不同通用 heuristic 参数，例如 row bucket、tile size、CUTLASS/GEMV threshold、static dequant 缓存策略。
- 缓存 decode repair 的 small VAR 构造结果，例如 position ids、budget scalar、row index buffer、visible position buffer。缓存必须按 shape/backend/request 安全失效，不能污染 prefill session。

## 禁止的优化

以下做法禁止作为正式实现：

- 通过提高 `pic_recompute_score_layer_idx` 来降低 decode hidden continuation 层数，因为 decode 不应使用 hidden continuation，且这会同时改变 prefill score layer 和 prefill sparse/cacheblend 行为。
- 在 decode repair 中使用 prefill 保存的 score hidden、hidden checkpoint 或 `_Concat` hidden rows 作为正式路径。
- 在 decode 阶段重新计算 cacheblend/delta-v score、重新 top-k 或重新生成 rank。
- 在 decode 阶段重新生成 high-attention 区间或 HKVD rank。decode 只能读取 prefill 已保存到内存中的 attention weight 摘要、区间和 rank。
- 为了 decode 提速修改 prefill graph 中 `PicScoreAttention` 的层位置、输出、mask 语义或 selected indices 语义。
- 让 decode-only fused op 在 prefill full-compute、full-reuse、cacheblend、epic、kvshare 路径无条件替换原算子。
- 在 decode repair 中生成或读取 scratch `.k/.v` 文件来模拟评分或补充 rank。
- 把 `tokens_per_decode_step=x` 写成独立特殊分支。可以按 tpd 设置参数，但实现必须解决一类 shape/row 的通用问题。

## 通用计算图与参数边界

Decode repair 的正式优化必须保持同一套 token-id sparse decode 计算图：

```text
selected_pic_token_ids + current_decode_token_id
        + logical_indices + causal-visible metadata
        -> embedding
        -> all decoder layers
        -> final_layernorm/logits_index=-1/lm_head
```

`tokens_per_decode_step=1..7` 只改变 active rows 数量：`active_rows = tpd + 1`，其中最后一行永远是当前 decode token。允许的差异只能是 kernel 参数或调度参数，例如 row bucket、tile size、OpenCL row32/row64、CUDA GEMV/CUTLASS threshold、静态 workspace 大小、shared sparse-index buffer 容量等。禁止把 `tpd=3`、`tpd=5` 这类值写成独立代码路径，也禁止为某个 tpd 导出不同 graph。

当前正确的通用入口是 token-id sparse decode：

- runtime 直接构造 `token_ids` 与 `logical_indices`。
- `beginPicDecodeRecomputeRows(logical_indices, 0, 1)` 绑定 sparse rows 并只 append 当前 decode row。
- repair rows 写回原 PIC logical slots，decode row 写到当前末尾 slot。
- final logits 通过 `logits_index=-1` 只计算最后 decode row，不应为 repair rows 计算 vocab logits。

注意：为 decode repair 新增的 backend flag 或参数也必须按“路径类型 + shape”解释，而不是按具体 tpd 解释。例如 `pic_decode_repair_sparse_active` 可以让同一个 weight-only Conv 根据 active rows 选择不同 row bucket；它不能变成 `if tpd == 5` 这种分支。

## 当前性能结论

旧 score-layer=1 的 CUDA decode repair 数据来自 hidden-continuation 实现，只能作为问题定位参考。它显示 `tpd=5` 受 tiny-row MLP/attention 和变量构造开销影响很大。一次临时 A/B 将导出图的 `pic_recompute_score_layer_idx` 改到 10，`tpd=5` 可到约 47.8 ms/token。这个实验不能作为正式方向，因为它依赖 hidden-continuation 路径，同时把 prefill-only 延迟从约 134 ms 拉到约 300 ms，并改变了 prefill scoring 边界。

正式方向已经切到 token-id sparse decode：prefill scoring 继续使用原 `pic_recompute_score_layer_idx=1`，decode repair 每步直接输入 selected HKVD token ids 和当前 decode token id，并用 logical indices 控制 PagedCache 写回位置。

2026-06-21 Jetson CUDA / Llama-3.2-1B / token-id sparse decode / `max_tokens=32` / repeats=5 数据：

```text
implementation                         tpd0    tpd1    tpd2    tpd3    tpd4    tpd5    tpd6    tpd7
token-id sparse baseline              38.34   44.20   50.38   58.29   67.29   74.65      -       -
rows5/6 CUTLASS heuristic             36.76   43.70   49.83   57.85   56.63   57.52      -       -
rows>=4 CUTLASS heuristic             36.78   43.68   50.02   56.48   56.71   57.71   58.93   62.98
fresh retest, rows>=4 CUTLASS          38.45   44.65   50.50   57.57   57.21   57.98   59.01   63.39
```

结论：

- token-id sparse decode 已经消除了旧 hidden-continuation 的结构性错误，`tpd=5` 从旧问题数据的 116.7 ms/token 下降到约 57.7 ms/token。
- 调整 INT4 tiny-row GEMV/CUTLASS threshold 属于通用 row-bucket 参数，能修掉 rows=5/6 的明显 cliff，但不能把 `tpd<=5` 全部压到 50 ms/token 内。
- `tpd=3..7` 现在形成约 56-63 ms/token 的平台期，说明剩余主开销不再是某个单独 tpd 的分支问题，而是 active rows 穿过全 decoder 层时的通用 tiny-row 计算图开销。
- `tpd=0->1->2` 仍有约 6 ms/token 的阶梯：fresh retest 中 `tpd=0` 为 38.45 ms/token，`tpd=1` 为 44.65 ms/token，`tpd=2` 为 50.50 ms/token。这个 gap 不是 score 层或 lm_head 引起，而是进入 token-id sparse decode 后，每步额外打开完整 small-row sparse forward graph：logical/token row 构造、causal visible metadata、每层 PagedAttention sparse bookkeeping、MLP tiny-row weight-only Conv、小 tensor Raster/Binary/LayerNorm 调度都会出现。
- `tpd=4/5/6` 非常接近是合理现象：active rows 分别为 5/6/7，已经落入同一类 batched/static-dequant/CUTLASS row bucket，新增 row 的算术量被更好的 kernel occupancy 和固定 launch/调度成本摊薄。不能把这解释成 `tpd=5` 需要特殊分支；它说明应继续优化 rows=2..8 的通用 tiny-row kernel family 和小算子融合。

已排除的非主因：

- final lm_head 不应是主因：export graph 在 lm_head 前使用 `logits_index=-1`，只对最后 decode row 计算 vocab logits。
- 完整 float mask 不应是正式快路径主因：full causal PagedAttention 可以通过 `full_causal_attention_mask` 和 sparse logical indices 在 kernel 内用 `q_logical + 1` 控制可见 KV。后续仍应把这一点固定成小 metadata/buffer，不回退到 `queryLen * kvLen` mask。
- score layer 不应影响 decode：若改变 `pic_recompute_score_layer_idx` 仍改变正式 token-id sparse decode TPOT，应视为 bug。

## `tpd=0/1/2` gap 的优化判断

`tpd=0` 是 normal/no repair decode，只计算当前 decode row。`tpd>=1` 会把 selected PIC token rows 与当前 decode row 放到同一个 sparse batch 中：`active_rows = tpd + 1`。因此 `tpd=1` 不是在 normal decode 上只增加一个小 token 的算术量，而是从普通 single-row decode 进入 decode repair 的 sparse graph。

当前 `tpd=0/1/2` 的 gap 应按三类通用成本处理：

1. Sparse repair 进入成本。

   每步要从 HKVD selector 取 selected logical rows，构造 `logical_indices` 和 `token_ids`，绑定 `beginPicDecodeRecomputeRows(logical_indices, 0, 1)`，并生成 position/visible metadata。优化方向是把 per-step selection 计划预排成 chunk view，复用 host/device scratch buffer；每步只写入当前 decode token 和有效 row count。这里的 cache key 是 request/backend/max_active_rows，不是具体 tpd。

2. Rows=2/3 tiny GEMV 成本。

   CUDA weight-only Conv 当前可按 `pic_decode_repair_sparse_active` 和 active rows 选择 GEMV 或 CUTLASS/static-dequant row bucket。rows=2/3 仍然容易受 tiny GEMV launch、OC 分块和权重读取效率限制；rows>=5 已基本进入更平的平台。优化方向是 row-bucket autotune：让 active rows、hidden size、intermediate size、dtype、INT4 group 信息选择 `GEMV_MB`、`CUTLASS`、OC-per-block 和 static-dequant workspace，而不是写 `if tpd == 1/2`。

3. 小 tensor 图调度成本。

   对 rows=2..8，MLP 的 gate/up/down、SiluMul、RMSNorm、residual add、Raster/Cast 等小算子会在所有层重复。只优化 PagedAttention mask 不能消除这部分阶梯。优化方向是通用 fused op：满足 shape/precision/backend 条件时走 fused kernel，不满足时 fused execution 内部调用原子算子 fallback。导出层可以统一使用 fused op，但 prefill 路径不得被 decode-only fused kernel 替换。

因此，降低 `tpd=0/1/2` gap 的正确目标不是让 `tpd=1` 回到 normal decode 的计算图，而是把 token-id sparse decode 的固定进入成本和 rows=2/3 tiny-row kernel 成本做小。这个优化一旦有效，也应该让 `tpd=3..7` 的平台值同步下降；如果只改善某一个 tpd，则说明实现很可能变成了 tpd 特化。

## 通用优化路线

下一步优化应按下面顺序推进，均要求同一图、同一类 kernel 覆盖 `tpd=1..7`，只允许参数不同。

1. MLP weight-only 投影融合。

   Llama dense MLP 每层包含 `gate_proj`、`up_proj`、`SiluMul`、`down_proj`。对 `active_rows=2..8`，每层多次 tiny-row weight-only Conv 的 kernel launch、activation 读取和中间结果读写是最可疑的剩余主开销。应新增通用 fused MLP family，而不是 tpd 特化：

   - `PicGateUpWeightOnly`: 同一个 kernel/执行类同时计算 gate/up 两个投影，输出 packed gate/up。tpd 只选择 tile/row bucket。
   - `PicSwiGLUDownWeightOnly`: 把 `silu(gate) * up` 与 down projection 的输入阶段融合，减少 PicSiluMul 的独立 kernel 和中间 tensor 往返。
   - 不满足 dense SiLU MLP、INT4/FP16、row bucket、shape 对齐等条件时，fallback 到原 gate/up/PicSiluMul/down 子算子。fallback 只能在 fused op 内部发生，导出层仍可统一使用 fused op。

2. RMSNorm/residual 小算子融合。

   每层 attention 前后通常有 residual add、RMSNorm、MLP 前后 add。对 rows=2..8，单个 kernel 的 launch 成本和小 tensor 往返占比高。应做通用 `PicResidualRMSNorm` / `PicAddResidual` 类 op，覆盖 active rows 小批量；tpd 只影响 row count/block size。该优化只允许在 decode repair fused graph 或 fused op fallback 内生效，不替换 prefill 的普通 LayerNorm 路径。

3. 共享 sparse logical index device buffer。

   当前每层 PagedAttention 都会同步 active sparse logical indices 到本层 cache buffer。数据很小，但在 32 层、每 token 重复时会放大 CPU 调度和 H2D/CL enqueue 开销。应把 `logical_indices` / per-row visible-end 作为每 decode step 的 backend shared buffer 上传一次，所有层 PagedAttention 读取同一个 device/CL buffer。容量按 `max_active_rows` 分配，tpd 只改变有效长度。

4. 固化 causal visible metadata。

   对 decode repair full-causal attention，正式表示应是 `logical_indices` 加可选 `visible_end`/`q_max_visible_position` 小 buffer。CUDA/OpenCL kernel 内根据 `visible_end[row]` 或默认 `logical_indices[row] + 1` 控制 KV 读取。完整 float mask 只允许作为 fallback，不作为 CUDA/OpenCL decode repair 快路径。

5. Row-bucket autotune，而不是 tpd 分支。

   CUDA weight-only Conv、OpenCL decode causal attention、OpenCL sparse flash attention 都应按 active rows/work 选择 row bucket：

   ```text
   active_rows = tpd + 1
   bucket = tune(active_rows, hidden_size, intermediate_size, head_dim, backend)
   ```

   bucket 可以选择 GEMV/CUTLASS、row32/row64、block size、tile size、是否用 static dequant workspace，但实现仍是一套通用 kernel family。正式报告必须列出每个 tpd 使用的参数，不能把参数选择藏成 tpd 特殊逻辑。

   建议把调参表写成 shape policy，而不是 tpd policy：

   ```text
   rows=2..3:  tiny-row GEMV_MB / compact OC grouping candidate
   rows=4..8:  CUTLASS or static-dequant batched candidate
   rows>8:     normal batched/prefill candidate
   ```

   这些阈值需要通过 Jetson CUDA / OpenCL 实测确认，且必须允许不同 backend 使用不同默认参数。它们是 active-row shape 超参数，不是 `tpd=1`、`tpd=5` 的独立代码路径。

## 通用 fused op 合约

Decode repair 可以新增 fused op，但 fused op 的行为必须是“同一导出图 + 条件化执行”，而不是“为某个 tpd 导出另一张图”。推荐合约如下：

1. Export 侧。

   导出层可以统一使用 decode repair fused op 表示，例如 `PicSiluMul`、`PicGateUpWeightOnly`、`PicDecodeRepairMLP` 或后续等价命名。这个 fused op 必须保留足够的子算子语义信息，使 backend 在不满足条件时能按原始顺序执行子算子。若 fused op 被放入可能由 prefill 触达的 graph，则 prefill 条件下必须直接走 fallback 子算子，且不申请额外大 workspace、不改变原 op 参数。

2. `onResize`。

   `onResize` 只根据 backend、dtype、active rows、hidden/intermediate size、weight layout、INT4 group、alignment、workspace 可用性等 shape 条件选择 fused plan。选择失败时，必须创建或复用原始子 execution，并调用子 execution 的 `onResize`。这里不能读取 `tokens_per_decode_step`，只能读取 active rows 或 tensor shape。

3. `onExecute`。

   `onExecute` 必须与 `onResize` 的 plan 一致。fused plan 执行单个或少数 fused kernel；fallback plan 按原始子算子执行。fallback 的数值、buffer lifetime、side effect 要与没有 fused op 的图一致。对 decode-only fused op，若 runtime metadata 表明当前不是 `pic_decode_repair_sparse_active`，必须走 fallback。

4. 参数策略。

   fused kernel 可以有多套 tile/block 参数，但参数表必须以 shape key 表示：

   ```text
   key = backend + dtype + active_rows + hidden + intermediate + head_dim + int4_group + weight_layout
   ```

   允许 `active_rows=2` 和 `active_rows=6` 选择不同 tile；不允许 `tpd=1` 和 `tpd=5` 进入不同代码语义。

5. 性能回退保护。

   每个 fused op 的正式启用必须同时报告：

   - fused hit rate: `tpd=1..7` 每档命中 fused kernel 的层数/调用数。
   - fallback hit rate: 不满足条件时实际走原始子算子的调用数。
   - prefill impact: full-compute/full-reuse/cacheblend/epic/kvshare prefill-only 延迟对比。
   - decode impact: `tpd=0..7` TPOT 对比。

## Kernel policy 草案

当前最需要避免的是把 `0/1/2` gap 优化成局部 tpd 特判。实现上应拆成下面几类通用 kernel policy。

1. MLP policy。

   - `PicGateUpWeightOnly`: rows=2..8 时，一个 kernel 同时消费同一份 hidden input 和两份 weight-only gate/up weight，输出 packed gate/up 或两个连续 output view。目标是少读一次 input，少一次 kernel launch，并让 rows=2/3 也能使用更合适的 OC grouping。
   - `PicSwiGLUDownWeightOnly`: 把 `silu(gate) * up` 融入 down projection 的输入阶段。第一阶段可以只融合 SiluMul + down input transform，保守保持 down projection kernel 不变；第二阶段再把 down weight-only matmul 合进去。
   - 对 rows>8、非 SiLU MLP、非 INT4/FP16、weight layout 不匹配或 workspace 不足时，内部 fallback 到 gate/up/PicSiluMul/down 原子算子。

2. Attention policy。

   - CUDA decode repair 应固定走 causal sparse attention fast path：`logical_indices` / shared sparse query buffer 决定每个 row 的 logical q，kernel 内用 `q_logical + 1` 或 `visible_end[row]` 限制 KV 读取。
   - OpenCL buffer 路径已有 row32/row64 decode causal attention 形态，正式 policy 应把 lane 选择写成 active rows/head_dim/kvLen 的 autotune 结果，而不是 tpd 分支。
   - 完整 float mask 只能是 fallback。正式 CUDA/OpenCL decode repair 路径不应每步构造 `queryLen * kvLen` mask。

3. Metadata policy。

   - 每 decode step 只上传一次 compact metadata：`logical_indices`、可选 `visible_end`、当前 decode row index、active row count。
   - 所有层 PagedAttention 共享同一份 device/CL buffer，只读有效长度，不在每层重复构造或上传 sparse logical indices。
   - CPU 侧 selection 结果应以 chunk view 复用，不在每步重新 sort 大 vector。若需要稳定顺序，预处理时把 HKVD rank 切成按区间排序的不可变 view；decode step 只跳过已 recomputed token。

4. Small-op policy。

   - RMSNorm/residual/Binary/Raster 类小 op 的融合只按 active rows 和 tensor layout 生效。
   - 优先融合一层内固定相邻的小 op，不跨 layer，不改变 PagedCache 写回语义。
   - 如果 fusion 需要额外临时 tensor，临时 tensor 必须按 `max_active_rows` 缓存并在 request/backend 结束时释放。

## 落地文件边界

现有 `PicSiluMul` 已经提供了正确的实现模板：export 侧用自定义 `LlmExporter::PicSiluMul`，converter rebuild 成 MNN `Extra(type=PicSiluMul)`，CUDA/OpenCL backend 在 `onResize` 中按 shape 选择 fused 或 fallback，fallback 内部继续调用原 Unary/Binary execution。后续通用 fused op 应沿用这个模式。

1. Export / converter。

   - `transformers/pic_llm/export/utils/custom_op.py`: 新增 torch symbolic wrapper 时，只表达数学语义和 shape，不放 tpd 逻辑。
   - `transformers/pic_llm/export/utils/transformers.py`: MLP export 只根据模型结构和 `pic_decode_tiny_fusion` 这类 feature flag 选择是否导出统一 fused op；不能按 `tokens_per_decode_step` 导出不同图。
   - `transformers/pic_llm/export/utils/mnn_converter.py`: 把 exporter custom op rebuild 成 `Extra`，保留必要 attr，例如 layer id、hidden/intermediate size、activation type、是否有 bias、子算子名。不要在 converter 中做 backend 或 tpd 分支。

2. CUDA backend。

   - `source/backend/cuda/execution/FuseExecution.cu`: 适合承接 elementwise / small-op fused execution，例如 `PicSiluMul`、后续 `PicResidualRMSNorm`。这里的 plan 只能看 tensor shape、dtype 和 metadata。
   - `source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu`: 适合承接 weight-only row-bucket policy 与 gate/up/down 投影融合。这里可以按 `pic_decode_repair_sparse_active` + active rows 选择 GEMV_MB / CUTLASS / static-dequant 参数，但不能读取或推断具体 tpd。
   - `source/backend/cuda/execution/PagedAttentionExecution.cu`: 只负责 sparse decode causal attention 与 shared metadata buffer。这里的优化目标是少传 mask、少重复上传 sparse indices，不改变 PagedCache slot 写回语义。

3. OpenCL backend。

   - `source/backend/opencl/execution/buffer/FuseBufExecution.cpp` 和 `source/backend/opencl/execution/image/FuseExecution.cpp`: 保持 buffer/image 两套 fused op fallback 语义一致。
   - `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`: row32/row64 选择应由 shape policy/autotune 决定。
   - `source/backend/opencl/execution/cl/attention_buf.cl`: decode causal kernels 应继续用 `q_logical + 1` 或后续 `visible_end[row]` 控制 KV 可见范围，不引入完整 float mask。

4. Runtime metadata。

   - `source/core/PagedKVMeta.hpp`: decode repair metadata 只表达 active rows、logical indices、append count、visible-end/shared-buffer 状态。
   - `transformers/pic_llm/engine/src/llm.cpp`: 每 step 构造 token ids/logical indices 的逻辑应向预排 chunk view 和 scratch buffer 复用演进；不能回到 score hidden continuation，也不能在这里为某个 tpd 改 graph。

## 建议实施顺序

为了尽快判断哪些优化能整体下移曲线，建议按以下顺序推进。每一步都保持同一 graph、同一 kernel family，并把参数选择记录成 shape policy。

1. Metadata / mask first。

   把 decode repair 的正式快路径固定为 `logical_indices + visible_end`，避免每步构造 `queryLen * kvLen` mask。这个改动风险最低，能直接验证 `0/1/2` gap 中的 host/VAR 部分，但预期不会单独把 `tpd=5` 压到 50 ms 内。

2. Weight-only row policy。

   对 rows=2..8 做 CUDA event/direct-op 小矩阵，比较 GEMV_MB、CUTLASS/static-dequant、OC-per-block、tile size。产出一张 shape policy 表后再进主线。该阶段不新增图，只调同一个 Conv execution 的参数策略。

3. `PicGateUpWeightOnly`。

   在确认 MLP Conv 仍是主项后实现 gate/up 投影融合。先只覆盖 Llama dense SiLU MLP、INT4/FP16、rows=2..8；其它情况 fallback。这个阶段是最可能同时降低 `tpd=1/2` gap 和 `tpd=3..7` 平台值的优化。

4. `PicSwiGLUDownWeightOnly` / small-op fusion。

   若 gate/up 后 Raster/Binary/LayerNorm 占比上升，再推进 SiluMul+down input transform、RMSNorm/residual 融合。该阶段要严控临时 tensor 生命周期，避免为了小 rows 额外申请大 buffer。

5. Shared sparse metadata buffer。

   把每层重复的 sparse logical index 上传改成每 step 一次上传、所有层共享。该阶段主要降低 host enqueue/H2D/CL enqueue 固定成本，对 `tpd=1/2` 更敏感。

## 实验矩阵与通过条件

每次优化都必须使用同一 artifact、同一模型、同一 token fixture 做矩阵测试，避免把随机输出长度、冷启动或 graph profile 开销混进结论。

```text
decode TPOT:
  tpd=0,1,2,3,4,5,6,7
  max_tokens=32
  repeats>=5
  warmup>=1

prefill-only:
  full-compute
  full-reuse
  cacheblend
  epic
  kvshare
  max_tokens=0

function smoke:
  representative tpd=1,3,5,7
  max_tokens>0
  output must be parseable text, not empty/repeated punctuation/garbled bytes
```

通过条件：

- `tpd=0` normal/no repair 不应变慢；它是 decode baseline。
- `tpd=1/2` gap 应下降，且不能通过改变 selection 语义、减少 repair token 数或跳过 layer 达成。
- `tpd=3..7` 平台值应同步下降；如果只改善某个单点 tpd，需要回查是否引入了 tpd 特化。
- prefill-only 的 full-compute/full-reuse/cacheblend/epic/kvshare 不应回退、变慢或改变 execution mode。
- graph profile / CUDA event profile 只用于归因，正式 TPOT 不使用 profile 环境变量。

## Tune / warm-up 测试协议

PIC server 已有 `POST /v1/tune/update_cache`，其语义是把当前 MNN runtime 在 warm 过程中生成的 cache 写回 `tmp_path/mnn_cachefile.bin`。该接口调用 `Llm::updateRuntimeCache()`，再调用主 runtime manager 和 cacheblend score runtime manager 的 `updateCache()`；它本身不是候选枚举器。decode repair 的 tune/search 应理解成“外层枚举启动配置 -> warm 目标 shape -> 调用 `update_cache` 固化该配置的 runtime cache -> 重启同一配置做热态 TPOT -> 汇总比较最佳启动参数”。rows=4/5 CUDA INT4 MLP 的新 kernel family / shape policy 仍需要 direct-op 和代码实现验证；server warm 负责把每个候选配置放到可比较的热态。

候选配置必须完整记录，至少包含：

```text
artifact_id / git diff id
model config path
tmp_path / mnn_cachefile.bin path
backend / precision / memory / power
export flags: pic_decode_tiny_fusion, pic_decode_gateup_fusion
server env: MNN_PIC_* / MNN_PAGED_ATTENTION_* / CUDA policy env
decode_refine: enabled, selector, tokens_per_decode_step, top_m, attention_layer_idx/head_ids
prompt fixture: text cache id, suffix, max_tokens, sampler config
```

正式 decode repair TPOT 前的 warm 顺序：

1. 使用独立 `tmp_path` 启动 `pic_server`；如果是比较新候选，先删除该候选旧的 `mnn_cachefile.bin`，避免复用其它配置的 cache。
2. 构建同一份 text cache 和 PIC cache。`tpd=0` baseline 请求不带 `decode_refine`，或显式 `decode_refine.enabled=false`；`tpd>=1` 的 decode repair warm 必须发带 `decode_refine.enabled=true` 的 `/v1/chat/completions` 请求，并设置 `max_tokens>0`，否则不会进入 token-id sparse decode。
3. 按目标矩阵 warm `tpd=0` 和 `tokens_per_decode_step=1,2,3,4`；若要扩展目标，再 warm `5,6,7`。每档至少跑 1 次 warm 请求，建议请求 shape 与正式 `max_tokens=32` 一致；为节省时间可以先用短 `max_tokens` 做 cold-shape 探测，再用正式 shape warm 一轮。
4. 如果本次候选会影响 prefill 或 PIC sparse path，再 warm `full-compute/full-reuse/cacheblend/epic/kvshare` 的 `max_tokens=0` 请求；decode-only weight-only / fused op 候选仍需至少抽样确认 prefill execution mode 不变。
5. 调用 `POST /v1/tune/update_cache`，确认返回 `{"scope":"mnn_runtime_cache","status":"ok"}`。
6. 重启 server，复用同一 `tmp_path/mnn_cachefile.bin`，再跑正式 strict TPOT。正式计时不打开 `MNN_PIC_GRAPH_PROFILE`、CUDA event profile 或其它会同步每个 op 的诊断开关。

多候选对比时，`update_cache` 只能固化该候选 warm 过的 runtime cache；不能把一个候选的 cache 文件拿给另一个候选复用。若某个候选只在 cold run 正向、warm+update_cache 后不稳定或重启后退化，则不能进入正式结论。

当前 PIC server 没有暴露 `Llm::tuning(OP_ENCODER_NUMBER, ...)` 的 HTTP 入口；该旧 tuning 逻辑只在 `mls.cpp` 中使用，用来试 `OP_ENCODER_NUMBER_FOR_COMMIT` 候选。decode repair 的启动参数搜索应先由外层脚本枚举候选配置、触发 warm 请求和 `/v1/tune/update_cache`，或者后续新增明确的 PIC tune endpoint；`/v1/tune/update_cache` 的成功响应只表示“当前已 warm 配置的 runtime cache 已提交”，外层仍要根据后续热态 TPOT 和 smoke 结果判断该配置是否是最佳启动参数。

直接算子 tune 与 server warm 要分层使用：

```text
direct-op CUDA event:
  目标: 筛 rows=2..8 的 weight-only / fused kernel shape policy
  工具: bench_ops/cuda/perf/WeightOnlyConv
  参数: run_test.out ... 2 <precision> 1 x 2, 即 memory=2(Memory_Low)
  输出: hidden_to_inter / inter_to_hidden / gateup / vocab 等 shape 的 kernel ms

server warm + update_cache:
  目标: warm 候选启动配置、固化 runtime/kernel cache，比较热态端到端 TPOT
  工具: pic_server + /v1/chat/completions + /v1/tune/update_cache
  输出: tpd=0..4 strict TPOT、runtime=mnn_token_id_sparse_decode、正确性 smoke
```

已有辅助脚本分两层：

- `.codex/skills/mnn-opt-ops/scripts/run_pic_decode_repair_tune_warm.sh`：假设 `pic_server` 已按候选配置启动，只负责发 `/v1/prefill/text`、`/v1/kv/pic_caches`、`tpd=0..4` warm chat 请求和 `/v1/tune/update_cache`。
- `.codex/skills/mnn-opt-ops/scripts/run_pic_decode_repair_config_tune.sh`：枚举候选启动配置，在每个候选独立 workdir 启动 `pic_server`，调用上一层 warm 脚本，然后可重启同一 workdir 验证 `tmp/mnn_cachefile.bin` 可复用。候选启动 env 放 `<name>.server.env`，请求矩阵覆盖放 `<name>.warm.env`，输出 `summary.tsv` 和 `warm_summary.tsv` 供后续 strict TPOT 接同一配置。
- `.codex/skills/mnn-opt-ops/scripts/summarize_pic_decode_repair_config_tune.py`：解析 config tune 输出，检查 `/v1/tune/update_cache` 响应、cache 文件、`tpd>=1` 是否进入 `mnn_token_id_sparse_decode`。cache 文件策略默认 `auto`：OpenCL/OrangePi 候选缺 `tmp/mnn_cachefile.bin` 是 `fail`，CUDA/Jetson 候选缺 cache 文件只记为 `warn/missing_optional`，因为当前 CUDA decode repair shape 可能没有可写 runtime cache 条目。`warm_summary.tsv` 出现 `candidate_status=fail` 的候选不能进入正式 TPOT；`warn` 候选可以进入后续 strict TPOT，但报告必须说明没有可复用 runtime cache 文件。

正式 TPOT 仍用独立 strict 测试流程；config tune runner 的作用是把候选参数先放到可比较的热态，不负责替代最终计时。

2026-06-22 continuation 用 Jetson CUDA / 1B silumul decode-repair 模型验证该 warm/update_cache 链路。服务启动在独立工作目录 `.cache/decode_repair_tune_warm/server_smoke_18131`，避免污染仓库根目录 `tmp/`；warm 脚本使用 `/v1/kv/pic_caches` 响应里的 `token_ids` 构造显式 `pic_cache.full_prompt_token_ids`，并追加 1 个 cache token 作为 suffix，因为当前 PIC server 已禁用 placeholder 分段 tokenization，chat 请求必须显式提供 full prompt token span。验证矩阵为 `tpd=0 1`、`max_tokens=4`：

```text
tpd=0: precision_recovery.execution_mode=native-full-reuse, decode_refine.disabled_reason=not_requested
tpd=1: decode_refine.runtime=mnn_token_id_sparse_decode, candidate_count=19, refined_token_count=4
POST /v1/tune/update_cache: {"scope":"mnn_runtime_cache","status":"ok"}
```

该 smoke 证明 PIC server warm API 链路和 token-id sparse decode warm 请求可用。远端该 CUDA run 未生成 `tmp/mnn_cachefile.bin`，说明这组 CUDA decode repair shape 没有通过 runtime cache 产生可写 autotune 条目；这不改变当前优化判断：server warm/update_cache 仍可用于比较不同启动配置的热态表现，但 CUDA weight-only rows=4/5 的主调度若要产生新收益，仍要靠 direct-op tune 和代码里的 shape policy / kernel family，而不是期待 `/v1/tune/update_cache` 自动生成新的 weight-only policy。

同日把 config-tune runner 同步到 Jetson 远端后，又用真实 artifact 跑通一次端到端候选 warm smoke：

```text
run_id: decode_repair_config_tune_smoke_20260622_144923
candidate: baseline
tpds: 0 1
max_tokens: 4
restart_check: ok

summary.tsv:
baseline  port=18140  status=ok  cache_exists=0

warm_summary.tsv:
tpd=0  candidate_status=warn  cache_policy=optional  cache_status=missing_optional  update_status=ok  execution_mode=native-full-reuse  decode_enabled=False  refined_token_count=0  completion_tokens=4
tpd=1  candidate_status=warn  cache_policy=optional  cache_status=missing_optional  update_status=ok  execution_mode=native-full-reuse  decode_runtime=mnn_token_id_sparse_decode  decode_enabled=True  refined_token_count=4  completion_tokens=4

POST /v1/tune/update_cache:
{"note":"runtime manager cache updated through MNN native cache flow","scope":"mnn_runtime_cache","status":"ok"}
```

这证明外层“候选启动配置 -> warm 目标 decode repair shape -> `/v1/tune/update_cache` -> restart check -> `warm_summary.tsv` 汇总”的调度配置 tune 链路可用。CUDA 候选因为缺 runtime cache 文件只进入 `warn`，后续可以进入 strict TPOT 但报告必须说明 `mnn_cachefile.bin` 缺失；OpenCL/OrangePi 候选不能套用这个宽松口径，缺 cache 文件仍是 `fail`，正式 OpenCL 计时必须复用 warm 后的 MNN autotune cache。

## 实施优先级

1. 先用 `MNN_PIC_GRAPH_PROFILE=1` 或 CUDA event profile 拆 `tpd=0/1/3/5/7` 的 per-op 时间，确认 MLP Conv、LayerNorm/Binary、PagedAttention、VAR/host 调度各自占比。
2. 若 MLP Conv 占主导，优先实现 `PicGateUpWeightOnly`，因为它减少两次 gate/up 投影的输入读取和 kernel launch，是最通用的收益点。
3. 若小算子/launch 占比高，再实现 residual/RMSNorm 与 `PicSwiGLUDownWeightOnly`。
4. 若 PagedAttention 或 host enqueue 占比高，优先共享 sparse logical buffer，并确保 CUDA/OpenCL 都走 decode causal attention 快路径。
5. 每一步都按同一 artifact 测 `tpd=0..7`，并同时验证 prefill-only `full-compute/full-reuse/cacheblend/epic/kvshare` 不变慢。

## TODO: 50 ms/token decode repair 压缩目标

当前最新 Jetson CUDA / Llama-3.2-1B / token-id sparse decode / `max_tokens=32` / repeats=7 的正式参考数据：

```text
tpd=0  38.81 ms/token
tpd=1  42.71 ms/token
tpd=2  49.71 ms/token
tpd=3  57.07 ms/token
tpd=4  56.94 ms/token
```

撤回 gate/up rows3-only fallback 后，用同一 silumul baseline 模型和当前 artifact 复测 `tpd=0..4`，路径仍回到同一平台期：

```text
run                                             tpd0 mean/median    tpd1    tpd2    tpd3    tpd4
cuda_current_after_gateup_revert_tpd0_4_r7     44.80 / 38.99      42.37   49.65   56.67   56.50
cuda_current_post_merged_revert_tpd0_4_r7      40.09 / 38.26      41.98   49.61   56.47   56.36
cuda_waitcleanup_tpd0_4_r7                     46.65 / 46.93      42.37   49.63   56.67   56.62
cuda_waitcleanup_tpd0_r7_recheck               42.12 / 38.74        -       -       -       -
cuda_embedcache_tpd0_4_r7                      38.76 / 38.28      42.16   49.71   56.36   56.45
cuda_argmax_sampler_tpd0_4_r7                  38.53 / 38.36      42.08   49.72   56.67   56.71
cuda_policy_limit3_tpd0_4_r7                   40.03 / 39.71      41.82   49.52   56.55   56.61
cuda_vocab_oc2_tpd0_4_r7                       38.84 / 37.85      41.33   48.90   56.29   56.35
cuda_no_double_sync_tpd0_4_r7_mt32             38.30 / 38.22      41.20   48.81   56.12   56.06
cuda_vocab_oc2_smalloc_tpd0_4_r7_mt32          45.19 / 37.77      41.43   48.92   55.54   59.23
cuda_vocab_oc2_attn_smalloc_tpd0_4_r7          38.43 / 37.95      41.38   48.97   56.28   58.15
cuda_async_noop_raster_tpd0_4_r7_mt32          38.48 / 37.64      41.59   48.96   55.51   55.49
cuda_async_raster_fullcopy_tpd0_4_r7_mt32      37.95 / 37.44      41.35   49.02   55.80   55.86
cuda_async_noop_raster_recheck_tpd0_4_r7_mt32  38.36 / 37.89      41.60   49.21   55.86   55.92
```

`tpd=0` mean 易被单次 outlier 拉高；wait-cleanup 首轮 `tpd=0` median 也被普通 decode 抖动拉到 46.93 ms/token，但单独复测回到 38.74 ms/token。`tpd=1..4` mean 稳定，去掉 `forwardRaw()` 后重复 logits `readMap/wait`、缓存 PIC token embedding、或把 greedy sampler 改成 `_ArgMax` 只带来噪声级变化，不能解决 `tpd=3/4` 约 56 ms/token 的平台期。`cuda_no_double_sync_tpd0_4_r7_mt32` 进一步去掉 sparse decode 结束前重复的 logits `readMap/wait` 和第二次 `syncPaged()`，strict 无 failures，能带来约 0.2-0.4 ms/token 的固定成本改善；这是可以保留的 decode-only cleanup，但仍不能把 `tpd=3/4` 压到 50 ms/token。当前核心问题仍是 rows=4/5 的通用小批量 forward，而不是 gate/up fallback、merged-CUTLASS 实验污染、embedding 构造、sampler readback 或 sparse 结束同步。

Raster 最新实验说明：只把 NC4HW4/NCHW `area=1 && channel%8==0` 的 no-op layout convert 从同步 `cudaMemcpy` 改为 runtime stream 异步 D2D copy，可以让 `tpd=3/4` 小幅下降到约 `55.51/55.49 ms/token`，但对 `tpd=1/2` 基本是噪声级。继续把 generic single-region contiguous `RasterBlit` 替换成异步 D2D full-copy 反而让 `tpd=3/4` 回退到 `55.80/55.86 ms/token`；该 full-copy 泛化已从源码移除，只保留 no-op layout convert 的 decode-only async copy。

2026-06-22 用移除 full-copy 泛化后的 artifact 复测 strict `tpd=0..4`，输出目录为 `.cache/decode_repair_smoke/cuda_async_noop_raster_recheck_tpd0_4_r7_mt32`。结果为 `38.36/41.60/49.21/55.86/55.92 ms/token`，strict 无 failures，runtime 仍为 `mnn_token_id_sparse_decode`，`old_decode_name_hits=[]`。结论不变：当前已把 `tpd=1/2` 压到 50ms 内，但 `tpd=3/4` 仍高约 5.9ms/token，必须继续做 MLP/Linear 主热点。

近期目标是先把 `tpd=0..4` 全部压到 50 ms/token 以内，并且不能通过改变 selection 语义、减少 repair token 数、跳过 decoder layer、改变 `pic_recompute_score_layer_idx` 或使用 hidden continuation 达成。所有优化必须仍然满足同一 token-id sparse decode 图、decode-only 生效、prefill 不回退。

已验证的负收益实验不能进入正式路径：

```text
experiment                                  tpd0    tpd1    tpd2    tpd3    tpd4
shared sparse metadata buffer              38.55   42.88   50.46   57.60   57.56
static CUTLASS/compact for tiny/vocab rows 46.19   50.07   57.45   77.67   77.63
PicGateUpWeightOnly fused rows=2..8        38.61   43.41   48.51   61.45   64.85
PicGateUpWeightOnly rows=3 fused fallback  39.22   42.39   50.25   86.58   86.28
PicGateUpWeightOnly merged CUTLASS rows4+  38.97   43.30   48.83   61.82   65.32
vocab batch1 GEMV OC_PER_BLK=8             53.98   56.55   62.64   70.55   70.49
vocab batch1 GEMV OC_PER_BLK=2             38.84   41.33   48.90   56.29   56.35
PicGateUpWeightOnly cuBLAS batched FP16     38.43   44.97   49.95   81.96   81.58
PicGateUpWeightOnly OC policy rows4/5      38.46   43.32   50.49   63.61   72.43
decode-only skip physicalSlots vector      39.43   42.31   49.73   58.04   57.95
small stack active_indices for decode      41.14   42.87   49.71   57.10   57.01
PIC token embedding cache                  38.76   42.16   49.71   56.36   56.45
greedy sampler _ArgMax fast path           38.53   42.08   49.72   56.67   56.71
row policy limit=2 screening r3              -       -     55.95   56.55   56.71
row policy limit=4 screening r3              -       -       -     59.46   56.58
row policy limit=5 screening r3              -       -       -       -     67.05
small-OC batch-limit oc<=2048=6            45.19   41.43   48.92   55.54   59.23
attention-only small-OC ic<=2048,oc<=2048  38.43   41.38   48.97   56.28   58.15
```

结论：

- Shared sparse metadata buffer 没有降低 `tpd=1/2` 固定成本，反而让 `tpd=2..4` 小幅回退，当前不保留。
- 把 decode tiny rows 或 final vocab row 强行切到 static-dequant CUTLASS/compact 会显著变慢，说明 rows=2/3 的 INT4 tiny GEMV 和 rows>=4 的现有 static CUTLASS 桶仍是当前更优 policy。
- 当前 `PicGateUpWeightOnly` CUDA fused kernel 只在 `tpd=2` 看到局部收益，但 rows=4/5 明显变慢；把 rows>=4 交给 fused op 内部 fallback 更差，说明当前 fallback 并不等价于原始图的两路 Conv 调度，`Extra` 包装和子 execution 路径本身会引入额外成本。
- 合并 gate/up 的 static-dequant CUTLASS 输出再 split 的方案同样不成立：strict `tpd=0..4` / repeats=7 无 failures，runtime 仍是 `mnn_token_id_sparse_decode`，但 `tpd=3/4` 为 `61.82/65.32 ms/token`，比 token-id sparse baseline 的约 `57 ms/token` 更慢。该 merged CUTLASS path 已从源码回退，不作为后续正式候选。
- 使用 static-dequant weight 走 cuBLAS `cublasGemmBatchedEx` 同时计算 gate/up 也不成立：`tpd=1/2` 没有足够收益，`tpd=3/4` 退化到 `81.96/81.58 ms/token`。这个结果说明对 rows=4/5 来说，额外 static-dequant/GEMM 调度与输出 bias pass 比原 weight-only tiny-row policy 更重；该实验已从源码回退，不能进入正式路径。
- 只调整当前 `PicGateUpInt4V14MBKernel` 的 OC policy 也不成立：把 rows=4 拆到 `MAX_BATCH=4/OC_PER_BLK=4`、rows=5..8 改成 `OC_PER_BLK=3` 后，opt-in gateup 模型 strict `tpd=0..4` 为 `38.46/43.32/50.49/63.61/72.43 ms/token`，比第一版 gateup rows=2..8 更差。该实验已回退；当前 naive gate/up fused kernel 的问题不是单个 OC-per-block 参数能解决。
- final vocab `batch=1, oc=128256` 的 INT4 GEMV 是所有 tpd 共享的固定成本。简单把 `OC_PER_BLK` 从 4 调到 8 是明显负收益：`tpd=0` 被拉到 `53.98 ms/token`，`tpd=3/4` 到 `70.55/70.49 ms/token`；但反向把 vocab-only `OC_PER_BLK` 调到 2 是正收益，strict `tpd=0..4` 为 `38.84/41.33/48.90/56.29/56.35 ms/token`。该优化按 shape 生效，仅限 `batch=1 && oc>=32768` 的 vocab GEMV，不是 tpd 分支；它可以保留，但只解决了 `tpd=1/2` 和小幅平台下移，`tpd=3/4` 仍需 MLP/attention rows=4/5 优化。
- 继续把 vocab-only `OC_PER_BLK` 从 2 降到 1 没有 batch=1 收益：direct-op `hidden_to_vocab rows=1` 为 `8.0359 ms`，与 OC=2 的 `8.0335 ms` 持平；`rows=2` 虽从 `8.4441 ms` 到 `8.1394 ms`，但正式 graph 通过 `logits_index=-1` 只对最后 decode row 做 lm_head，等价 batch=1。该实验未进入 strict TPOT，源码已回退到 OC=2。
- 在 CUDA PagedAttention decode 阶段跳过未使用的 `physicalSlots` host vector 构造也不是有效热点：`tpd=0` 可回到 39.43 ms/token，但 `tpd=3/4` 退到 `58.04/57.95 ms/token`。说明这类 host 小 vector 不是 rows=4/5 平台期主因，该实验已从源码回退。
- 把 decode recompute 输出的 `active_indices` 从 `std::vector` 改成栈上小数组也没有收益：`tpd=3/4` 仍为 `57.10/57.01 ms/token`，`tpd=1` 还小幅回退。说明每层几个 int 的 indices host 构造不是主瓶颈，该实验已从源码回退。
- 在 `preparePicDecodeRepair()` 缓存 PIC token embedding 后，每步只对当前 decode token 查 embedding，strict `tpd=0..4` 无 failures，`tpd=3/4` 仍为 `56.36/56.45 ms/token`。这说明 token id sparse decode 的 embedding/VAR 构造不是主瓶颈；该优化可以保留为 decode-only 辅助项，但不能替代 MLP/Linear 主优化。
- greedy sampler `_ArgMax` 快路径同样只是噪声级：`tpd=3/4` 为 `56.67/56.71 ms/token`，没有降低平台期。sampler readback 不是当前核心痛点；该全局 sampler 改动已从源码回退，不作为 50 ms/token 目标的主要证据。
- INT4 GEMV/CUTLASS active-row 分界复测后，当前最佳默认仍是 PIC decode repair 下 `rows<=3` 走 GEMV_MB、`rows>=4` 走 static-dequant/CUTLASS。把 limit 降到 2 会让 `tpd=2` 从约 49.5ms 退到约 56.0ms；把 limit 升到 4 会让 `tpd=3` 退到约 59.5ms；把 limit 升到 5 会让 `tpd=4` 退到约 67.1ms。该 override 已从源码回退，不能作为正式路径。
- 让 `oc<=2048` 的小输出投影在 decode repair 下继续使用 `int4GemvBatchLimit=6` 只在 direct-op 上看起来正向，端到端 strict `tpd=0..4` 为 `45.19/41.43/48.92/55.54/59.23 ms/token`，其中 `tpd=4` 比 `vocab OC_PER_BLK=2` 单独策略退化约 2.9ms/token。该 small-OC policy 已从源码回退；当前只保留 `batch=1 && oc>=32768` 的 vocab `OC_PER_BLK=2` shape policy。
- 把 small-OC policy 收窄成 attention-only `ic<=2048 && oc<=2048` 也不能保留。direct-op 中 `hidden_to_hidden` / `hidden_to_kv` 变快，但 strict `tpd=0..4` 为 `38.43/41.38/48.97/56.28/58.15 ms/token`，`tpd=3` 基本不变、`tpd=4` 仍退化约 1.8ms/token。该 refined policy 已回退到 `picDecodeRepairSparse ? 3 : 6`；attention projection 不是足够大的独立杠杆。
- 后续不应再用“更大 static-dequant 覆盖范围”或当前 `PicGateUpWeightOnly` kernel 作为主优化方向；应先回到原始 token-id sparse baseline，再用 CUDA event/direct-op 证明新的 MLP 融合实现真正减少 gate/up/down 投影次数、权重读取和小算子 launch。
- 当前 profile 归因仍指向 MLP/Linear：`MNN_PIC_GRAPH_PROFILE` 下 `max_tokens=8` 的 decode 请求中，`Convolution` 约 63% 总同步时间，`Raster` 约 12%，`PicSparseAttention` 约 9%，`BinaryOp/LayerNorm/UnaryOp` 更低；CUDA attention 已显示 `mask_elements=0`、`causal_mask_skipped=1`，不是当前主要瓶颈。
- 2026-06-22 用 clean `vocab OC_PER_BLK=2` artifact 再跑 `MNN_PIC_GRAPH_PROFILE=1` / `max_tokens=8` / `tpd=3,4`，profile 同步开销不能作为正式 TPOT，但热点排序一致：`tpd=4` 正式请求聚合约 `total_ms=1028.5`，其中 `Convolution=643.6ms`、`Raster=125.5ms`、`PicSparseAttention=85.4ms`、`BinaryOp=46.9ms`、`While=40.7ms`、`PagedAttention=25.5ms`。Top op 仍是 `/lm/lm_head/Linear` 和每层 `mlp/down_proj/gate_proj/up_proj/Linear`；因此下一步主线仍是 MLP/Linear 融合或更强的 weight-only kernel，而不是继续调 sampler、embedding、metadata 小 vector 或 causal mask。

2026-06-22 direct-op `hidden_to_qkv_concat` 补测，Jetson CUDA `WeightOnlyConv` CUDA event，rows=1..8，`ic=2048, oc=3072, qblock=64`：

```text
rows  hidden_to_qkv_concat_ms
1     1.5386
2     1.5394
3     1.5422
4     0.9557
5     0.4260
6     0.4268
7     0.5610
8     0.5630
```

与当前独立 attention projection direct-op 参考相比，这条路不成立。以 Llama-3.2-1B 形状估算，独立 `q_proj + k_proj + v_proj` 约等于 `hidden_to_hidden(2048->2048) + 2 * hidden_to_kv(2048->512)`：rows=4 约 `0.1452 + 2*0.0513 = 0.2478ms`，rows=5 约 `0.1450 + 2*0.0525 = 0.2500ms`。concat qkv rows=4/5 分别是 `0.9557/0.4260ms`，明显更慢。因此不要把 q/k/v concat projection 导入正式 graph；attention projection 当前不是可用的大杠杆。

同日临时验证过另一个 weight-only shape policy：把 PIC decode repair 的 `int4GemvBatchLimit` 从 3 放宽到 5，并把 V14 multi-batch `OC_PER_BLK` 从 4/3 降到 2。该实验只用于 direct-op 判断，源码随后已回退。结果负向：

```text
case                  rows4_ms  rows5_ms  结论
hidden_to_inter       1.1143    0.9029    慢于当前默认参考约 0.696/0.528
hidden_to_gateup      2.1885    1.7673    慢于当前默认参考约 1.146/1.028
inter_to_hidden       0.9577    0.9508    慢于当前默认参考约 0.580/0.576
```

结论：rows=4/5 回到 V14_MB 并缩小 OC grouping 不能解决平台期，反而会放大 MLP 三个主要 projection 的耗时。当前默认仍应保持 PIC decode repair 下 `rows<=3` 走 tiny GEMV_MB，`rows>=4` 走现有 static-dequant/CUTLASS 桶；后续要找的是新的 kernel family 或更深融合，而不是继续调整这个分界。

下一步 TODO 更新：

1. 保留 `PicSiluMul` 3D feature-dim 修正和 vocab `batch=1 && oc>=32768` 的 `OC_PER_BLK=2` shape policy。
2. 不再推进 q/k/v concat、NC4 swiglu-down 输入绕行、shared sparse metadata buffer、small-OC GEMV 扩大覆盖、当前 naive `PicGateUpWeightOnly` kernel 或 cuBLAS batched gate/up。
3. 继续围绕 MLP/Linear 主热点做通用实现：优先证明新的 rows=4/5 MLP weight-only kernel 或真正的 `PicSwiGLUDownWeightOnly` 能在 direct-op 上减少 `gate/up/down` 总成本；再进入端到端 strict `tpd=0..4`。
4. 若 MLP direct-op 无法给出足够空间，转向 residual/RMSNorm/Raster 小算子融合，但仍必须按 active rows/shape policy 生效，不能写 tpd 分支。

2026-06-22 新增 direct-op CUDA event 基准 `bench_ops/cuda/perf/WeightOnlyConv`，专门测 decode repair active rows 下的 INT4 weight-only Conv 原子成本；Jetson log 为 `.cache/bench_ops/decode_repair/weight_only_conv_20260622_014351.log`，运行参数应为 `run_test.out bench_ops/cuda/perf/WeightOnlyConv 2 2 1 x 2`，最后的 `2` 是 `Memory_Low`。不含 vocab 的 MLP shape 数据：

```text
rows    hidden->inter 2048x8192    inter->hidden 8192x2048
1       0.9146 ms                  0.5883 ms
2       0.6078 ms                  0.5830 ms
3       0.5727 ms                  0.5836 ms
4       0.5287 ms                  0.5839 ms
5       0.5289 ms                  0.5910 ms
6       0.5277 ms                  0.5899 ms
7       0.5318 ms                  0.5856 ms
8       0.5337 ms                  0.5899 ms
```

这组 direct-op 数据给出两个判断：

- rows=4..8 的单个 MLP projection 已经平台化，继续只调 GEMV/CUTLASS threshold 或 OC-per-block 很难拿到 7ms/token 级收益。
- 但 gate/up 两个 `2048->8192` projection 每层仍各约 0.53ms；如果下一版 gate/up fusion 能真正省掉一次 projection 级成本，16 层理论可释放约 8ms/token，刚好覆盖 `tpd=3/4` 到 50ms 的缺口。因此 gate/up 方向仍是优先级最高，但必须换实现方式，不能复用当前 naive `PicGateUpInt4V14MBKernel` 或 Extra fallback。

同日新增 direct-op `hidden_to_gateup_concat` 形状，验证“标准 weight-only Conv 一次输出 gate+up 拼接 OC=16384，再 split”的理论空间；这不是端到端 TPOT，未计入后续 `Slice/Raster` split 成本：

```text
rows    hidden->inter 2048x8192    gateup-concat 2048x16384    concat vs 2x inter
1       0.8147 ms                  1.9579 ms                   slower
2       0.7404 ms                  1.1219 ms                   faster by ~0.36 ms
3       0.5924 ms                  1.1213 ms                   faster by ~0.06 ms
4       0.5779 ms                  1.1259 ms                   faster by ~0.03 ms
5       0.6071 ms                  1.0285 ms                   faster by ~0.19 ms
6       0.5788 ms                  1.0305 ms                   faster by ~0.13 ms
7       0.7355 ms                  1.0319 ms                   faster by ~0.44 ms
8       0.5725 ms                  1.0350 ms                   faster by ~0.11 ms
```

这条标准 concat Conv 路线比旧 Extra custom gateup kernel 更保守，因为它复用 `ConvFpAIntBExecution` 的原生 weight-only policy，不再在 CUDA backend 引入新的 gate/up scalar kernel。当前 converter 已把 opt-in `PicGateUpWeightOnly` 重建成 `Convolution(outputCount=2*intermediate) -> Slice(axis=1)`，小型 fake Linear 单元验证通过，生成 op 类型为 `Reshape, ConvertTensor, Convolution, ConvertTensor, Slice, Reshape, Reshape`。完整 1B patch 模型已完成 MNNConvert 和 Jetson strict TPOT，但收益仍不足以达标：

```text
run                                      tpd0    tpd1    tpd2    tpd3    tpd4
cuda_gateup_concat_patch_tpd0_4_r7      36.28   41.05   48.67   55.24   54.88
```

结论：标准 concat Conv 比旧 Extra gate/up fused kernel 安全，能小幅降低 `tpd=3/4`，但仍只能把平台值从约 `55.9ms` 拉到约 `55ms`，不足以把 `tpd=3/4` 压到 50ms 内。它可以作为 opt-in 实验继续研究，但不能作为完成 50ms 目标的证据。

Jetson 重启后第一次 rows=4/5 direct-op 曾在 GPU `114750000 Hz`、EMC `800000000 Hz` 的低频状态下误跑，得到 `hidden_to_inter rows4/5 = 4.0523/4.0481 ms`、`inter_to_hidden rows4/5 = 1.6171/1.6164 ms`、`hidden_to_gateup_concat rows4/5 = 2.8946/2.1961 ms`。这组数据只说明功耗/频率状态错误，不能进入优化判断。执行 `sudo -n jetson_clocks` 后，GPU 固定到 `905250000 Hz`、EMC 到 `1600000000 Hz`，同一 artifact、同一 filtered direct-op、`repeat=50/warmup=10` 的有效数据为：

```text
case                         rows4_ms  rows5_ms
hidden_to_inter              0.5244    0.5273
inter_to_hidden              0.5704    0.5659
hidden_to_gateup_concat      1.0262    1.0268
```

有效数据与历史 direct-op 量级一致，也进一步证明标准 gate/up concat 的理论空间很小：rows=4 时 `2 * hidden_to_inter = 1.0488 ms`，concat 为 `1.0262 ms`，每层只省约 `0.0226 ms`；rows=5 时每层只省约 `0.0278 ms`。即使按 16 层估算，也只有约 `0.36-0.45 ms/token` 的投影级空间，不可能单独覆盖 `tpd=3/4` 到 50ms 还缺的约 5ms。因此 gate/up concat + packed silu 只能作为小幅 opt-in cleanup；后续主优化必须找更深的 `SwiGLU+down` 累加融合、RMSNorm/residual/Raster 调度融合，或新的 weight-only kernel family。

2026-06-22 continuation 用新增脚本 `.codex/skills/mnn-opt-ops/scripts/run_remote_weight_only_conv_tune.sh` 复跑 Jetson CUDA focused direct-op tune，log 为 `.cache/bench_ops/decode_repair_tune/weight_only_rows4_5_20260622_134205.log`，参数为 rows=4-5、warmup=20、repeat=80、precision=2：

```text
case                         rows4_ms  rows5_ms
hidden_to_inter              0.5253    0.5268
inter_to_hidden              0.5647    0.5609
hidden_to_gateup_concat      1.0241    1.0266
```

该结果再次确认当前调度配置下 rows=4/5 单 projection 已平台化，concat gate/up 只比两个独立 hidden_to_inter 少约 `0.026-0.029ms/layer`；继续调同一条 concat Conv / split / packed silu 路线无法提供端到端约 5ms 的缺口。后续 direct-op tune 应优先比较新的 weight-only kernel family、真正融合 `SwiGLU + down` 累加路径，或 residual/RMSNorm/Raster 调度融合，而不是继续扩大现有 GEMV/CUTLASS threshold。

同日用新增远端 tune 脚本做轻量复核，命令参数为 rows=4-5、warmup=10、repeat=30，输出本地镜像日志 `.cache/bench_ops/decode_repair_tune/local_weight_only_rows4_5_latest.log`，远端原始日志 `.cache/bench_ops/decode_repair_tune/weight_only_rows_4-5_20260622_143637.log`。结果与 repeat=80 长测一致：rows=4/5 的 `hidden_to_inter` 为 `0.5265/0.5285ms`，`inter_to_hidden` 为 `0.5561/0.5646ms`，`hidden_to_gateup_concat` 为 `1.0257/1.0264ms`；汇总器给出的 `best_estimated_total_saving_ms=0.4896`，decision 仍是 `cleanup_only_do_not_use_as_main_tpot_candidate`。因此当前已有 tune 测试没有支持把 gate/up concat 或单纯调度配置作为主优化进入 strict TPOT。

同日修正 direct-op 协议：`WeightOnlyConv` / `PicDecodeMlp` 必须用 `memory=2(Memory_Low)`，否则 CUDA 会创建普通 Conv/CUTLASS execution，而不是 `ConvFpAIntBExecution`，rows policy / GEMV profile / weight-only 策略验证不会生效。bench 现已在 `memory!=2` 时直接 fail；远端 tune 脚本默认 `MNN_TUNE_MEMORY=2`。用正确 memory-low 参数重新验证 Jetson rows=4/5，log 为 `.cache/bench_ops/decode_repair_mlp/mlp_rows4_5_memory_low_20260622_152031.log`：

```text
case                                  rows4_ms  rows5_ms
hidden_to_inter default               0.5242    0.5247
inter_to_hidden default               0.5684    0.5682
PicDecodeMlp chain default            1.5959    1.5985
hidden_to_inter rows45_gemv opt-in    0.6046    0.7548
inter_to_hidden rows45_gemv opt-in    0.5697    0.6963
PicDecodeMlp chain rows45_gemv        1.7795    2.2096
```

结论更明确：把 PIC decode repair rows=4/5 强行拉回现有 V14_MB GEMV family 是负收益，rows4 MLP chain 慢约 `0.18ms/layer`，rows5 慢约 `0.61ms/layer`。该临时实验开关已从生产 CUDA path 移除，不进入 strict TPOT；后续 rows=4/5 主线必须是新的 weight-only kernel family 或真正减少 `SwiGLU + down` / MLP 总成本的融合，而不是继续调整当前 GEMV/CUTLASS 分界。

同日扩展 `bench_ops/cuda/perf/PicDecodeMlp` 输出 `silu_down` 和 `silu_down_over_down`，用于估算“只把 `PicSiluMul` 合到 down 输入阶段”的理论上限。Jetson memory-low log 为 `.cache/bench_ops/decode_repair_mlp/mlp_swiglu_down_bound_rows4_5_20260622_153845.log`：

```text
rows  gate_ms  up_ms   silu_ms  down_ms  silu_down_ms  silu_down_over_down_ms  chain_ms
4     0.5236   0.5256  0.0151   0.5562   0.5693        0.0131                  1.5962
5     0.5259   0.5264  0.0080   0.5683   0.5782        0.0099                  1.5991
```

因此，`SwiGLU+down` 如果只删除独立 `PicSiluMul` kernel / 中间 tensor 往返，rows4/5 上限只有约 `0.01ms/layer`，按 16 层也只有约 `0.16-0.21ms/token`，远小于 `tpd=3/4` 到 50ms 还缺的约 5ms。下一版 `PicSwiGLUDownWeightOnly` 必须真正改变 down projection 的输入读取/累加路径，或者转向新的 rows=4/5 weight-only matmul kernel；不能只做 NC4 输入绕行或 SiluMul 合并。

同日新增 bench-only direct-op `bench_ops/cuda/perf/DecodeRepairMlpGemmFloor`，直接用 cuBLAS FP16 GEMM 测 rows=4/5 的 dense GEMM floor，并同时比较 `oc_by_rows` 与 `rows_by_oc` 两种 cuBLAS 维度组织。Jetson log 为 `.cache/bench_ops/decode_repair_mlp/mlp_fp16_gemm_floor_rows4_5_20260622_160719.log`，同 artifact 的 weight-only recheck log 为 `.cache/bench_ops/decode_repair_mlp/mlp_weight_only_chain_recheck_rows4_5_20260622_160748.log`：

```text
case                                  rows4_ms  rows5_ms
FP16 GEMM oc_by_rows gate             0.7090    0.7048
FP16 GEMM oc_by_rows up               0.6891    0.7044
FP16 GEMM oc_by_rows down             0.4804    0.4757
FP16 GEMM oc_by_rows chain_no_silu    1.8489    1.8796
FP16 GEMM rows_by_oc chain_no_silu    2.5365    4.0550
current INT4 weight-only gate         0.5255    0.5276
current INT4 weight-only up           0.5252    0.5279
current INT4 weight-only down         0.5761    0.5607
current INT4 weight-only chain        1.6023    1.5987
```

结论：普通 dense FP16 GEMM / cuBLAS 路线没有给 rows4/5 MLP 提供足够空间；更快的 `oc_by_rows` chain 仍比当前 INT4 weight-only chain 慢约 `0.25-0.27ms/layer`，`rows_by_oc` 更差。单个 down projection 的 FP16 GEMM 比当前 weight-only down 快约 `0.09-0.10ms/layer`，但 gate/up 已经明显慢于现有 INT4 path，单靠 dense GEMM 或 gate/up concat 无法覆盖端到端约 5ms/token 缺口。后续如果做 rows4/5 kernel family，必须是 INT4-native、面向小 batch 的 weight-only 累加/解包设计；如果做 `PicSwiGLUDownWeightOnly`，也必须把 activation 与 down weight-only 累加真正合并，而不是把中间结果交给普通 dense GEMM。

同日又临时验证了 `MNN_CUDA_PIC_INT4_ROWS45_PROTO_OC`：在 PIC decode repair rows=4/5 下强制回到 `GEMV_FpAInt4B_V14_MB`，并枚举更大的 `OC_PER_BLK=2/3/4/8/16`。该分支只用于 direct-op 判断，源码已移除。Jetson/local mirror log 目录为 `.cache/bench_ops/decode_repair_mlp/rows45_proto_20260622_161611/`：

```text
case          rows4_chain_ms  rows5_chain_ms
default       1.5962          1.6088
proto_oc=2    2.1377          2.6379
proto_oc=3    1.7494          2.2103
proto_oc=4    1.7804          2.2691
proto_oc=8    3.4080          4.5196
proto_oc=16   6.7765          8.3407
```

结论：基于现有 V14_MB 的 rows4/5 GEMV 扩展和 OC grouping 调整都明显负收益，不能作为正式路径或继续调参主线。该实验只排除了“继续调当前 GEMV family”的路线；rows4/5 后续默认只能接受经过 direct-op 和 strict TPOT 证明的 shape-specific path。下一步要做的是新的 INT4-native rows4/5 weight-only kernel，或真正融合 `SwiGLU + down` 累加路径。

移除该临时 CUDA env 分支后重新交叉编译并同步 Jetson，memory-low direct-op 复核 log 为 `.cache/bench_ops/decode_repair_mlp/mlp_rows4_5_cleanup_recheck_20260622_162311.log`：rows4 `gate=0.5263/up=0.5247/down=0.5749/chain=1.6011ms`，rows5 `gate=0.5276/up=0.5277/down=0.5742/chain=1.6164ms`。这证明 cleanup 后默认 rows4/5 仍保持在约 `1.60ms/layer`，没有落入已排除的 GEMV prototype。

随后新增 rows4/5 real-layout cuBLAS shape policy：`MNN_CUDA_PIC_INT4_ROWS45_CUBLAS` 默认等价于 `all`，`0` 可禁用回到 static-dequant/CUTLASS 对照，`down` 只作用于 `ic=8192,oc=2048`。该分支只在 `pic_decode_repair_sparse_active && fp16 && staticDequant && !runtimeDequant` 下命中 rows4/5 的 MLP 形状：`2048->8192` gate/up 与 `8192->2048` down。它不是前面 `DecodeRepairMlpGemmFloor` 的 generic synthetic layout；它直接复用 MNN 已静态反量化的 `mDequantFilter` `[ocp, icp]`，并用 `cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_N)` 计算当前真实 layout。

Jetson direct-op accuracy/perf log 目录为 `.cache/bench_ops/decode_repair_mlp/rows45_cublas_default_20260622_164156/`，本地镜像为 `.cache/bench_ops/decode_repair_mlp/rows45_cublas_default_latest/`。`bench_ops/cuda/accuracy/Rows45Cublas 2 2 1 x 2` 覆盖 rows4/5 的 gate/up/down/down_all，全部 `max_abs=0`、`bad=0`：

```text
case                         rows4_bad  rows5_bad
gate policy=all              0/32768    0/40960
up policy=all                0/32768    0/40960
down policy=down             0/8192     0/10240
down policy=all              0/8192     0/10240
```

同一 artifact 的 `bench_ops/cuda/perf/PicDecodeMlp 2 2 1 x 2` 对照：

```text
policy                         rows4_gate  rows4_up  rows4_silu  rows4_down  rows4_chain  rows5_gate  rows5_up  rows5_silu  rows5_down  rows5_chain
disabled static-dequant/CUTLASS 0.5253      0.5263    0.0155      0.5750      1.5966       0.5282      0.5254    0.0131      0.5682      1.6076
default rows45 cuBLAS           0.4243      0.4233    0.0129      0.4239      1.2768       0.4257      0.4257    0.0156      0.4259      1.2819
```

该候选 direct-op 节省 rows4 `0.3198ms/layer`、rows5 `0.3257ms/layer`，按 Llama-3.2-1B 16 层粗略折算约 `5.1-5.2ms/token`，首次达到 `tpd=3/4` 到 50ms 所需的数量级。因此它可以进入 strict `tpd=0..4` 端到端 TPOT 和自然语言 smoke；但在 strict TPOT 通过前，只能标记为 `strong_candidate_for_strict_tpot`，不能把 50ms 目标视为完成。如果 TPOT 回归或发现 server path 没有稳定命中 static-dequant cache，应把默认收窄为 `down` 或退回 opt-in。

修复 `Rows45Cublas` accuracy case 的 env 恢复后重新交叉编译并同步 Jetson，复核 log 为 `.cache/bench_ops/decode_repair_mlp/rows45_cublas_recheck_20260622_165059/`：accuracy 仍全部 `bad=0`；default cuBLAS rows4/5 chain 为 `1.2762/1.2814ms`，禁用回退为 `1.5938/1.5975ms`。这说明测试 env 修复没有改变性能结论。

正式 strict TPOT 前按 autotune/warm 约定启动当前 Jetson `pic_server`，先运行 `.codex/skills/mnn-opt-ops/scripts/run_pic_decode_repair_tune_warm.sh` 对 `tpd=0..4` 做 warm，并调用 `/v1/tune/update_cache`；日志目录为 `.cache/decode_repair_tune_warm/rows45_cublas_warm_20260622_1655/`，`update_cache.response.json` 返回 `{"status":"ok","scope":"mnn_runtime_cache"}`。随后重启同一 server config 再跑 strict timing，server log 只出现 CUDA 当前可接受的启动期 `Load Cache file error`，没有 `Cache invalid`、`target unavailable`、`async persistent PIC cache read failed`、CUDA illegal access 或 HTTP error。

端到端 strict `tpd=0..4` 使用当前默认 rows4/5 cuBLAS policy、1B decode-repair silumul PIC 模型、320-token fixture、`cacheblend ratio=0.10`、`score_layer=1`、`max_tokens=32`、每档 warmup=1/repeats=7，输出目录 `.cache/decode_repair_smoke/cuda_rows45_cublas_tpd0_4_r7_mt32/`：

```text
tpd  mean_ms/token  median_ms/token  p90_ms/token  runtime                     failures
0    37.54          37.32            37.90         -                           0
1    42.02          41.96            42.38         mnn_token_id_sparse_decode  0
2    49.54          49.49            49.73         mnn_token_id_sparse_decode  0
3    50.71          50.65            50.80         mnn_token_id_sparse_decode  0
4    50.64          50.63            50.67         mnn_token_id_sparse_decode  0
```

相对上一组 `cuda_gateup_packed_silu_tpd0_4_r7_mt32` 的 `37.10/39.45/48.30/55.10/54.79ms`，rows4/5 cuBLAS 对 `tpd=3/4` 分别改善约 `4.39/4.15ms/token`，但 `tpd=3/4` 仍高于 50ms 约 `0.6-0.7ms/token`。因此该候选应保留为当前最强正向实现，但 50ms 目标仍未完成；下一步只需要再找约 `0.04-0.05ms/layer` 量级的真实节省，或降低 rows3/4 调度/小算子开销。

随后做 rows4/5 cuBLAS policy 与 cuBLAS 参数小范围 tune。`MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=all/down/0` 的 direct-op log 为 `.cache/bench_ops/decode_repair_mlp/rows45_cublas_policy_tune_latest/`：

```text
policy  rows4_chain_ms  rows5_chain_ms  conclusion
all     1.2747          1.2804          仍是最优，gate/up/down 全走 real-layout cuBLAS
down    1.4741          1.4798          只优化 down，少了 gate/up 收益，不能替代 all
0       1.5957          1.6019          回退 static-dequant/CUTLASS 对照
```

因此默认 policy 不应从 `all` 收窄到 `down`。同轮新增 tune-only env：`MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_MATH=keep/default/tensor`、`MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_COMPUTE=32f/fast16/16f`、`MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO=default/tensor/<cublas algo id>`；默认仍保持原来的 `tensor + 32f + CUBLAS_GEMM_DEFAULT_TENSOR_OP`。Jetson direct-op log 为 `.cache/bench_ops/decode_repair_mlp/rows45_cublas_algo_tune_latest/`：

```text
variant             rows4_chain_ms  rows5_chain_ms  accuracy
default_current     1.2799          1.2833          bad=0
math_keep           1.2738          1.2817          bad=0
math_default        1.2739          1.2822          not promoted
algo_default        1.2749          1.2809          not promoted
compute_fast16      1.2734          1.2812          bad=0
compute_16f         1.2801          1.2913          bad=0, down 变慢
math_keep_fast16    1.2744          1.2833          bad=0
```

这些 direct-op 差异都低于当前剩余缺口的数量级；`math_keep` 主要尝试去掉每次 execute 的 host-side `cublasSetMathMode`，CUDA event 本身不能完整统计这个 host 开销，所以只对它做短 strict。按正式流程用 `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_MATH=keep` 启 Jetson server，先 warm `tpd=0..4` 并调用 `/v1/tune/update_cache`，`update_cache.response.json` 返回 `{"status":"ok","scope":"mnn_runtime_cache"}`；重启同一 env 后 strict `tpd=3,4`，输出目录 `.cache/decode_repair_smoke/cuda_rows45_cublas_math_keep_tpd3_4_r7_mt32/`：

```text
tpd  mean_ms/token  median_ms/token  p90_ms/token  runtime                     failures
3    51.76          51.71            52.25         mnn_token_id_sparse_decode  0
4    50.49          50.39            50.80         mnn_token_id_sparse_decode  0
```

server log `.cache/logs/decode_repair_rows45_cublas_math_keep_strict_20260622_1718.log` 未见 `Cache invalid`、`target unavailable`、`async persistent PIC cache read failed`、CUDA illegal access 或 HTTP error。结论：`math_keep` 没有让 `tpd=3/4` 稳定进入 50ms 内，且 `tpd=3` 明显回退；不扩大到 `tpd=0..4`，不提升为默认。cuBLAS 参数 tune 只作为诊断开关保留，主线仍应转向新的 rows4/5 INT4-native weight-only kernel family、真正融合 `SwiGLU+down` 累加路径，或 rows3/4 周边调度/小算子固定开销。

扩展 `tpd=5..7` 用同一 server、同一 fixture、repeats=3，输出目录 `.cache/decode_repair_smoke/cuda_rows45_cublas_tpd5_7_r3_mt32/`：

```text
tpd  mean_ms/token  median_ms/token  p90_ms/token  runtime                     failures
5    58.89          56.99            62.87         mnn_token_id_sparse_decode  0
6    57.74          57.65            57.93         mnn_token_id_sparse_decode  0
7    59.08          59.07            59.17         mnn_token_id_sparse_decode  0
```

这说明 rows4/5 cuBLAS 主要解决了当前 `tpd=3/4` 边界，不能让更高 `tpd=5..7` 自动达标；若后续目标扩到 0..7，需要新的 rows6/7 策略或真正减少每步 sparse decode 固定开销。

后续候选进入 strict TPOT 前仍应先过 direct-op 数量级门槛：在 rows4/5 cuBLAS 之前，`tpd=3/4` 距离 50ms 约差 `5ms/token`，按 Llama-3.2-1B 的 16 层粗略折算，rows=4/5 的 MLP/Linear 大候选至少需要接近 `0.31ms/layer` 的真实节省，才可能单独完成目标；rows4/5 cuBLAS 已经满足这个入场条件并完成 strict 复核。当前剩余缺口约 `0.6-0.7ms/token`，下一步候选至少要能解释约 `0.04-0.05ms/layer` 的真实节省，或直接降低 decode repair 的调度/小算子固定开销。`.codex/skills/mnn-opt-ops/scripts/summarize_weight_only_conv_tune.py` 会把 `WeightOnlyConv` log 汇总成 raw table、MLP projection cost、gate/up concat delta 和 verdict；旧 gate/up concat log 的 verdict 是 `cleanup_only_do_not_use_as_main_tpot_candidate`，因为 `best_estimated_total_saving_ms=0.4320` 低于 `strict_tpot_entry_threshold_ms=1.0`。如果新候选仍低于当前剩余缺口，该候选只能算小 cleanup，不应进入长时间端到端 TPOT 作为主线。

2026-06-22 继续实现 `PicPackedSiluMul`：让 opt-in `PicGateUpSiluWeightOnly` 在 converter 中重建为 `concat Conv -> PicPackedSiluMul`，去掉 concat 后的 `Slice` 和双路 post-reshape；CUDA backend 增加 `PicPackedSiluMul` native Extra，`ShapePicExtra` 支持该 op；同时修正 CUDA/OpenCL `PicSiluMul` 对 3D tensor 使用 `Tensor::channel()` 的问题，改为 3D 取最后一维、4D 取 channel，确保 tiny-row `PicSiluMul` fused kernel 能按真实 feature dim 命中。

后续 continuation 又把 opt-in `PicGateUpSiluWeightOnly` 的重建进一步收窄为 `pre_reshape -> pre_convert -> concat Conv -> PicPackedSiluMul`：`PicPackedSiluMul` 可在 `packed_input_nc4=1` 时直接线性读取 concat Conv 的 `area=1 && channel%8==0` NC4 输出，并仍写回 `[1, rows, intermediate]` 的 3D NCHW 结果。这只删除 gateup+silu opt-in 路径中的 no-op layout copy 和 reshape，不改变默认 `pic_decode_tiny_fusion`，也不改变 prefill graph。该改动尚未在 Jetson 上复测，不能作为 50ms 目标证据；设备恢复后需要和 `cuda_gateup_packed_silu_tpd0_4_r7` 做严格对比。

同一 Jetson CUDA artifact 下的 strict `tpd=0..4` 对照：

```text
run                                      tpd0    tpd1    tpd2    tpd3    tpd4
cuda_silumul_featuredim_tpd0_4_r7       36.48   40.83   48.51   55.28   55.25
cuda_gateup_packed_silu_tpd0_4_r7       37.10   39.45   48.30   55.10   54.79
cuda_gateup_swiglu_down_nc4_tpd0_4_r7   43.14   44.32   48.69   57.32   56.77
```

三组均 strict 无 failures，`tpd>=1` runtime 仍为 `mnn_token_id_sparse_decode`，`old_decode_name_hits=[]`。`cuda_gateup_packed_silu` 的 `tpd=1` completion token 均值为 27，其他档为 32；因此 `tpd=1` 数字需和 completion length 一起看。

结论：

- `PicSiluMul` 3D feature-dim 修正是可保留的通用 shape 修正，能让默认 silumul 模型稳定在 `36.48/40.83/48.51/55.28/55.25 ms/token`，但 `tpd=3/4` 仍未达标。
- `PicGateUpSiluWeightOnly -> concat Conv + PicPackedSiluMul` 是正向 opt-in：相比 silumul 对照，`tpd=1/2/3/4` 分别约改善 `1.38/0.21/0.18/0.46 ms/token`，但幅度远小于需要的约 5ms。
- `PicSwiGLUDown` 的 NC4HW4 第一阶段实验是负收益：直接让 packed SiLU 输出 NC4HW4 并跳过 down pre-convert 后，`tpd=3/4` 退到 `57.32/56.77 ms/token`。当前实现不能进入正式路径；后续若继续做 `PicSwiGLUDownWeightOnly`，必须先用 direct-op/CUDA event 证明 down input transform 或 fused down matmul 本身能减少 projection 级成本，而不是只删 Convert/Reshape。
- 目前最好的 strict 数据仍只把 `tpd=0/1/2` 压到 50ms 内，`tpd=3/4` 仍在约 `54.8-55.3ms`。50ms 目标未完成，下一步需要更强的 MLP projection fusion 或 weight-only kernel family，而不是继续只删小 Raster/Reshape。

2026-06-22 又验证了 CUDA decode-only sparse flash tile attention。该方向只改 PagedAttention decode repair 的 causal sparse attention，不改 prefill；但 correctness 不成立，不能保留为默认路径：

```text
run                                                       tpd0    tpd1    tpd2    tpd3    tpd4    correctness
cuda_sparse_flash_tile_1b_silumul_tpd0_4_r7_mt32          38.01   41.05   49.26   54.51   54.68   fail: tpd>=1 输出 repeated "!"
cuda_sparse_flash_tile_fix_1b_silumul_tpd0_4_r7_mt32      38.85   40.29   48.38   53.83   54.09   fail: tpd>=1 输出 repeated "!...."
cuda_force_v2_1b_silumul_tpd0_4_r7_mt32                   38.97   42.59   50.60   57.26   58.01   pass
cuda_tile_revert_broadcastfix_1b_silumul_tpd0_4_r7_mt32   38.08   41.35   48.98   55.53   55.60   pass
```

第一版 tile kernel 的 warp reduction 只在 lane0 得到正确 max/sum，其它 lane 用后缀 reduction 结果计算 softmax probability，导致 attention 输出错误；已修正为 lane0 广播。广播修正后输出仍退化为重复标点，说明该 `Q_TILE x K_TILE` decode kernel 还有更深的语义差异。由于收益不足以完成 50ms 目标且破坏自然语言 smoke，decode-only flash tile 分支已从默认路径移除；只保留通用 warp broadcast 修正给已有 sparse tile kernel。后续不要继续把 attention tile 当作主线，除非先用 direct-op accuracy 证明与 row-compressed v2 完全一致。

按优先级推进下面三项 TODO：

2026-06-22 continuation 更新：已读取附件建议，结论与本 TODO 保持一致：优先级仍是
`PicGateUpWeightOnly -> PicSwiGLUDownWeightOnly -> metadata/mask`，但前两项必须先用 direct-op
证明 rows=4/5 真实降本，不能重复当前已验证负收益的 naive gate/up、concat+split、NC4 swiglu-down
或 shared sparse metadata 路线。为减少 Jetson direct-op 长跑风险，`bench_ops/cuda/perf/WeightOnlyConv`
已增加测试控制 env：`MNN_BENCH_WEIGHT_ONLY_CASE`、`MNN_BENCH_WEIGHT_ONLY_ROWS`、`MNN_BENCH_WEIGHT_ONLY_REPEAT`、
`MNN_BENCH_WEIGHT_ONLY_WARMUP`，下一次只跑 `hidden_to_inter/inter_to_hidden/hidden_to_gateup_concat`
的 `rows=4-5` 小矩阵。当前 Jetson 在旧 `pic_server` 和一次 direct-op 进入 D 状态后已发起重启；
这次 direct-op 未产出有效数据，不能作为性能结论。

2026-06-22 true-fusion 复核：按“普通 dense FP16 GEMM/cuBLAS 不是 rows4/5 主线，新的候选必须是
INT4-native 小批量 weight-only 累加/解包，或真正融合 `SwiGLU + down` 累加路径”的判断，曾临时验证
`SwiGLUDownInt4Proto` standalone CUDA event 原型。该原型已从源码测试分支和 `.cache/decode_repair_fusion`
临时目录删除，不进入默认 `run_test.out`，也不改变正式 decode/prefill 路径；后续判断只保留本文档里的
负向数据和当前最佳实现。默认 Jetson cross CUDA `run_test.out` 已重新构建通过，`Rows45Cublas`
精度 rows4/5 全部 `bad=0`。

当前正式 rows4/5 MLP direct-op 基线，Jetson CUDA / Llama-3.2-1B shape / `memory=2`：

```text
case                         rows4_ms  rows5_ms
PicDecodeMlp chain           1.2839    1.2844
hidden_to_inter              0.4240    0.4253
inter_to_hidden              0.4254    0.4241
hidden_to_gateup_concat      1.0242    1.0256
```

dense FP16 GEMM floor 复核：

```text
layout=oc_by_rows            rows4_ms  rows5_ms
gate_fp16_gemm               0.7170    0.7142
up_fp16_gemm                 0.6809    0.7438
down_fp16_gemm               0.4751    0.4752
projection_sum               1.8730    1.9331
```

结论与原判断一致：普通 dense FP16 GEMM/cuBLAS projection sum 比当前 INT4/static-dequant chain 慢
约 `0.59-0.65 ms/layer`，不能作为 rows4/5 主线；`down` 单投影的 FP16 floor 只比当前 down
约快 `~0.05 ms/layer`，但 gate/up 明显更慢，整体没有空间。

`SwiGLU + INT4 down` 真 fused-down 原型结果：

```text
rows  oc_per_block  fused_int4_swiglu_down_ms
4     2             3.2723
4     4             2.1361
4     8             1.9893
5     2             4.2388
5     4             2.8338
5     8             2.6137
```

该原型把 `silu(gate) * up` 放进 down 的 INT4 累加热循环，避免单独写 `swiglu` 中间 tensor；但最好结果
rows4/5 仍为 `1.99/2.61 ms/layer`，明显慢于当前 `PicSiluMul + down` 的约 `0.47/0.56 ms/layer`，
也慢于完整 MLP chain 的 `1.28 ms/layer`。根因是 Xavier sm72 上当前 down/gate/up 已经走
static-dequant FP16 tensor-core/cuBLAS/CUTLASS 级路径；INT4-native fused-down scalar 累加虽然减少
一个小中间 tensor 和一个 tiny activation kernel，但失去 tensor-core GEMM 形态，并在每个 output
tile 反复读取 gate/up，代价远大于省掉的 `PicSiluMul`/memory traffic。`PicDecodeMlp` direct-op 也显示
`PicSiluMul + down` 相比 down 单投影的额外成本只有噪声到小几十微秒量级，不能解释 `tpd=3/4`
仍缺的约 `5 ms/token`。

因此当前已删除 `SwiGLUDownInt4Proto` 测试分支，也不把普通 FP16 dense GEMM 路线推进到正式实现。
后续如果继续做“真正融合”，必须是更深的 gate/up/down tile pipeline：在同一 tile 内计算 gate/up
的中间片段、应用 SwiGLU，并直接累加 down，同时保持或替代 tensor-core 级吞吐；单独把
`PicSiluMul` 融入 down 不能提供足够收益。

1. `PicGateUpWeightOnly`。

   - 合并 MLP 的 `gate_proj + up_proj` 两个 weight-only projection。
   - 只按 `active_rows / hidden / intermediate / dtype / int4 group / weight layout` 选择参数，不写 `if tpd == x`。
   - 当前第一版 fused kernel、`Extra` 内部 fallback、merged CUTLASS split 和 cuBLAS batched FP16 方案都不能进入正式默认路径：fused rows=2..8 会伤害 `tpd=3/4`，rows=3-only fallback 会把 `tpd=3/4` 拉到约 86 ms/token，merged CUTLASS split 会把 `tpd=3/4` 拉到约 `61.82/65.32 ms/token`，cuBLAS batched FP16 会把 `tpd=3/4` 拉到约 `81.96/81.58 ms/token`。
   - 标准 concat Conv + `PicPackedSiluMul` 是当前最安全的 opt-in gate/up 路线，strict `tpd=0..4` 为 `37.10/39.45/48.30/55.10/54.79 ms/token`，但它只带来小幅收益，不能作为默认完成目标。
   - Export 侧已将该实验拆到独立 `--pic_decode_gateup_fusion`；正式 `--pic_decode_tiny_fusion` 只默认启用已验证的 `PicSiluMul` 等小 elementwise fusion，不能无意导出当前负收益的 gate/up fused graph。
   - 下一版必须先解决“projection 级真实少做一次主成本”的问题；仅删除 `Slice/Reshape/Convert` 或用旧 naive gate/up scalar kernel 都不足以覆盖 `tpd=3/4` 的约 5ms 缺口。
   - 这是仍值得继续研究的 MLP 主优化点，但验收不能只看 `tpd=2`；必须同时压低 `tpd=1..4`，尤其不能让 rows=4/5 平台值高于 token-id sparse baseline。实现边界主要在 `transformers/pic_llm/export/utils/custom_op.py`、`transformers/pic_llm/export/utils/transformers.py`、`transformers/pic_llm/export/utils/mnn_converter.py`、CUDA weight-only execution / fused execution，以及必要的权重导出或图重写逻辑。

2. `PicSwiGLUDownWeightOnly`。

   - 把 `silu(gate) * up` 和 down projection 的输入阶段融合。
   - 目标是减少独立 `PicSiluMul` kernel 和中间 tensor 往返。
   - 已测的保守 NC4HW4 输入阶段融合为负收益，`tpd=3/4` 退到 `57.32/56.77 ms/token`，不能保留。
   - 下一版不应只删 down pre-convert；需要把 activation 与 down weight-only matmul 的输入读取/累加路径真正合并，并先用 direct-op/CUDA event 证明 rows=4/5 的 down projection 成本下降。

3. Metadata / mask 轻量化。

   - Shared sparse logical-index device buffer 已在 CUDA 上实测为负收益，不能按当前实现保留。
   - 后续若继续优化 metadata，只做更小范围的 mask/visible-end 构造优化，例如避免完整 `queryLen * kvLen` mask、缓存 per-step host scratch，不再重复尝试每层 buffer 共享。
   - 这类优化只能作为辅助项；它不足以把 `tpd=3/4` 平台值压到 50 ms/token 内。

验收要求：

- 正式 TPOT 测试必须覆盖 `tpd=0,1,2,3,4`，目标全部 `< 50 ms/token`；后续再扩展到 `tpd=5,6,7`。
- 每次优化要报告相对上表的 delta，并说明是否修改 CUDA、OpenCL、exporter 或 runtime metadata。
- `tpd=0` normal/no repair 不应变慢；它仍是 decode baseline。
- 若 `tpd=3/4` 仍高于 50 ms/token，则不能宣称目标完成，下一步继续推进 MLP projection fusion。
- prefill-only 路径不得因 decode-only fused op 或 shared metadata buffer 改变 graph、execution mode 或 kernel 参数。

## 推荐实现边界

1. Exporter 不新增 decode hidden checkpoint。

   - `pic_recompute_score_layer_idx`: 只用于 prefill sparse/cacheblend scoring。
   - `pic_decode_repair_outputs`: 不应输出 score hidden 作为 decode 正式输入；若现有导出仍包含该输出，只能视为旧实现兼容字段。
   - decode repair 需要的持久信息是 token ids、HKVD rank、high-attention interval/summary 和 recomputed bitmap。

2. Runtime 新增 token-id sparse decode repair 入口。

   `preparePicDecodeRepair` 接收 prefill rank、PIC token ids、seed recomputed token 集合和 decode selection metadata，不能读取 score hidden，不能加载 continuation module。每步 decode 选择 repair logical indices 后，构造与其一一对应的 token ids，并把当前 decode token id 追加到同一个 sparse batch。

3. Runtime 新增 decode-only HKVD selector。

   `preparePicDecodeRepair` 应接收或绑定 prefill 已保存的 high-attention interval、指定 layer/head attention weight 摘要、HKVD rank 和 seed recomputed token 集合。`selectPicDecodeRepairLogicalIndices` 只从这些内存态数据中选择尚未重算 token；recomputed bitmap 应在本步 sparse decode 成功后更新。

4. Decode attention 输入轻量化。

   对 token-id sparse decode，优先传 `logical_indices` 和 `q_max_visible_position` 或 row visible-end buffer。kernel 内按 visible end 控制 K/V 读取。完整 float mask 只作为 fallback，不作为 CUDA/OpenCL decode 快路径。

5. Decode tiny op 融合保持条件化。

   融合算子满足 shape/backend/precision 条件时走 fused kernel；不满足时内部 fallback 到原有子算子执行。fallback 不能改变输出和 prefill 性能。

6. 验证必须分开报告。

   - Prefill-only: full-compute/full-reuse/cacheblend/epic/kvshare 不回退、不变慢。
   - Decode-only: normal tpd=0 与 tpd=1..7 的 TPOT。
   - 功能正确性: `max_tokens>0` 输出必须是自然语言或可解析内容，不能只看 HTTP 200。
