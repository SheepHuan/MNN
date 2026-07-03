# Rhino Pi-X1 OpenCL PIC 优化记录

本文收敛 Rhino Pi-X1 / Adreno 上 PIC OpenCL 的当前稳定判断。更细的实验过程、A/B 记录和对应小时目录上下文在：

- `.codex/skills/mnn-pic-optimize/log/rhino/2026-06-27-00/README.md`
- `.codex/skills/mnn-pic-optimize/log/rhino/2026-06-27-00/context.md`

## 范围

本文记录 Rhino Pi-X1 / Adreno 上 PIC OpenCL 的当前判断，重点是 `headDim=128` 的 3 个模型：

- `MiniCPM5-1B`
- `Llama3.2 3B`
- `Qwen3-8B`

目标不是单看算子绝对更快，而是让 `cacheblend` / `epic` 在正式实验口径下，相对 `normal-full-recompute` 获得稳定加速。

## 正式实验口径

当前 Rhino 的正式默认对比口径固定为：

- 频率：`cpu=max,gpu=max,ddr=max`
- 上下文：`512,1024,1536,2048,2560`
- 暂不纳入：`3072`

原因：

- `3072` 在 Rhino 上更容易触发内存和整机稳定性问题，不适合作为当前默认 formal sweep。
- 低 CPU 或低 GPU 频率下得到的结果，只能用于定位趋势，不能直接覆盖正式 `benchmark.csv` 的默认结论。

## 现在确认的问题

Rhino 上不是 attention 全面失效，当前已经确认有 4 个独立问题：

1. `run_pic_prefill_latency_sweep.py` 之前把 Rhino `frequency_profile=max` 当成“查询当前状态”，没有真正下发 `cpu=max,gpu=max,ddr=max`。
2. normal `llm_bench` 之前没有 OpenCL warmup，`Llama3.2 3B` 在 `ctx=1024` 首轮 cache rebuild / shape-tune 会被直接算进 baseline。
3. PIC config 解析之前错误优先了“带旧 `tmp/mnn_cachefile.bin` 的非 boundary 导出”，导致 `cacheblend` / `epic` 落到不支持 `pic_recompute_budget` 的旧模型。
4. 在真正 graph-boundary 路径跑通之后，算子级默认路由仍有两处需要继续优化：
   - score layer 的 full-Q sparse family 仍会默认选到 `qsplit`
   - later sparse layer 的 `headDim=128` 变体仍会默认选到 `mqtile_hd128_q4k16`

前 3 个问题会直接污染端到端 benchmark；后 1 个问题才是当前真正的算子优化方向。

## 算子级实测结论

### 1. score layer 默认选 `qsplit` 是错误路由

稳定 bench 入口：

- `bench_ops/opencl/perf/PagedAttention/SparseRuntime`
- `run_test.out`

热 shape：

- case: `MiniCPM5-1B_score_scatter_ctx1024_q519`

Rhino 实测：

- 当前 auto/default:
  - `family=qsplit`
  - `avg≈4320.6 ms`
- 强制 flash:
  - `mqtile_hd128_q8k16 + single_piece` -> `avg≈1551.2 ms`
  - `mqtile_hd128_q4k8_kimg + range_q128` -> `avg≈1550.9 ms`

结论：

- Rhino 的 score layer full-Q hot shape 上，flash 家族约比默认 `qsplit` 快 `2.8x`。
- 当前正式路径不应该继续让这类 shape 默认落回 `qsplit`。

### 2. later sparse layer 默认选 `q4k16` 不是最优

热 shape：

- case: `MiniCPM5-1B_later_compact_scatter_ctx1024_q519`

Rhino 实测：

- 当前 auto/default:
  - `variant=mqtile_hd128_q4k16`
  - `schedule=range_q64`
  - `avg≈1502.6 ms`
- 候选对比：
  - `mqtile_hd128_q4k8 + range_q64` -> `avg≈1405.7 ms`
  - `mqtile_hd128_q4k8 + range_q128` -> `avg≈1507.4 ms`
  - `mqtile_hd128_q4k8_kimg + range_q64` -> `avg≈1491.4 ms`
  - `mqtile_hd128_q4k8_kimg + range_q128` -> `avg≈1464.4 ms`
  - `mqtile_hd128_q8k16_kimg + range_q64` -> `avg≈1500.3 ms`
  - `row64 + range_q64` -> `avg≈1789.0 ms`

结论：

- Rhino 当前 hot band 上，later sparse 更应优先 `mqtile_hd128_q4k8 + range_q64`。
- 这不是一个大数量级收益，但它是稳定收益，而且是正式路径上的默认选型错误。

### 3. compact dense 不是当前第一阻塞点

单入口 bench：

- `bench_ops/opencl/perf/WeightOnlyConv`

热 shape：

- `minicpm_hidden_to_ffn`, rows=`519`, `1536 -> 4608`
- `minicpm_ffn_to_hidden`, rows=`519`, `4608 -> 1536`

Rhino 实测：

- `hidden_to_ffn`
  - auto/image: `11.2236 ms`
  - forced `generic_quant`: `11.1789 ms`
  - forced `pic_quant`: `11.1377 ms`
  - forced `fp_weight`: `13.6078 ms`
- `ffn_to_hidden`
  - auto/image: `12.0376 ms`
  - forced `pic_quant`: `12.0450 ms`
  - forced `fp_weight`: `14.6966 ms`

结论：

- `rows=519` 这一档 compact dense 已经没有“灾难性误路由”。
- Adreno 上 image-backed weight 方向仍然成立，但 dense 这里当前不是最值得优先修的点。
- 现阶段更大的问题仍是 sparse attention family / variant 路由错误。

## 为什么 OrangePi 的 tune 经验不能直接套到 Rhino

不是 tune 框架不同，而是设备侧代价模型不同，且 Rhino 之前受过冷启动测量和旧 cache 影响：

1. Adreno 上部分 flash 候选首次执行的额外开销更重
   - 尤其是混合 `buffer + image` 候选时，第一次跑 flash 候选比 `qsplit` 更容易带入额外初始化成本。
   - 如果 warmup/tune 直接记录第一次执行时间，就会把本应更快的 steady-state flash 错判成慢路径。

2. Rhino 曾经存在旧 tune cache / 旧 program binary 干扰
   - 这会让修复后的候选继续回放旧选择。
   - PIC 手写 tune key 必须带设备族 namespace。固定 cache 路径不变，但 `paged_sparse_flash_schedule_*`、`paged_sparse_flash_variant_*`、`paged_sparse_qsplit_chunk_*`、`paged_cacheblend_topk_family_*` 这类条目必须区分 `mali` / `adreno`，否则 OrangePi/Mali 与 Rhino/Adreno 会互相继承错误路由。score-layer sparse family 现在按设备固定，不再通过 `paged_score_sparse_family_*` 选择。

3. Mali 和 Adreno 的最优点本来就不一样
   - OrangePi 上跑得好的变体，不代表 Adreno 上也会是默认最优。
   - Rhino 需要单独的 Adreno 路由和 tune namespace，而不是复用 Mali 的经验值。

因此，Rhino 问题的本质不是“tune 逻辑没生效”，而是“tune 在错误测量条件下，把错误 family/variant 固化了”。

## 端到端效果结论

当前已经确认，Rhino 上 `Llama3.2 3B / ctx=1024 / cpu=max,gpu=max,ddr=max` 在正确口径下已经过线。

关键修正有两个：

1. normal baseline 先 warm 再测。
   - `Llama3.2 3B` 在 Rhino 上 `ctx=1024` 首轮 OpenCL warm 为 `3m38s`，steady-state normal 约 `8.99s`。
   - 之前把首轮 rebuild/tune 直接算进 normal，是 benchmark 过慢的直接原因。
2. PIC 请求必须优先 graph-boundary 模型目录。
   - `AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_opencl_greedy.json`
   - 不能因为旧 `tmp/mnn_cachefile.bin` 存在，就退回非 boundary 导出。

### Llama3.2 3B, ctx=1024, true max

- normal: `8.9903 s`
- pic-full-recompute: `8.1639 s`
- full-reuse: `0.7251 s`
- cacheblend:
  - `0.05` -> `1.3380 s` -> `6.719x`
  - `0.10` -> `1.7286 s` -> `5.201x`
  - `0.20` -> `2.5885 s` -> `3.473x`
  - `0.30` -> `3.4502 s` -> `2.606x`
  - `0.40` -> `4.3262 s` -> `2.078x`
  - `0.50` -> `5.1483 s` -> `1.746x`
- epic:
  - `0.05` -> `1.2728 s` -> `7.064x`
  - `0.10` -> `1.6142 s` -> `5.569x`
  - `0.20` -> `2.4497 s` -> `3.670x`
  - `0.30` -> `3.2881 s` -> `2.734x`
  - `0.40` -> `4.2141 s` -> `2.133x`
  - `0.50` -> `5.1529 s` -> `1.745x`

结论：

- `cacheblend` / `epic` 的 `5%-50%` 全预算都已经 `>=1.2x`。
- `cacheblend` / `epic` 的 TTFT 随 budget 单调上升，没有异常反转。
- 当前 Rhino `benchmark.csv` 里的 `Llama3.2 3B` 旧行已经不应再作为判断依据；要以这组 canonical `cpu=max,gpu=max,ddr=max` 新结果为准。

### 其他模型

`MiniCPM5-1B` 和 `Qwen3-8B` 仍需要按同一条 canonical 口径补重跑，才能更新正式结论。它们历史结果只能作为旧趋势参考，不能和这次 `Llama3.2 3B` 的新口径混用。

## 当前明确的工程动作

下一步不再发散，按这三件事执行：

1. 用同一套 canonical 口径补 `MiniCPM5-1B` 和 `Qwen3-8B`
   - 真 `cpu=max,gpu=max,ddr=max`
   - normal 先 warm，再测 steady-state
   - PIC config 固定优先 graph-boundary 导出

2. 继续做 Rhino / Adreno 的算子路由优化
   - 对 Rhino / Adreno 的 `headDim=128` full-Q hot shape，默认优先 flash。
   - later sparse hot band 默认优先 `mqtile_hd128_q4k8 + range_q64`。
   - 需要同时处理旧 tune cache 继续回放旧 family/variant 的问题；当前约束是固定同一个 OpenCL cache 根路径，但 tune key 内必须带 `adreno` namespace。

3. 保持 benchmark 口径收敛
   - Rhino 只保留 `frequency_note=cpu=max,gpu=max,ddr=max` 的结果。
   - OpenCL cache 路径固定单一路径，normal 与 PIC 都必须先 warm 目标 shape，再进入正式计时。

## 本文对应的判断

当前最重要的结论只有三条：

1. Rhino 之前“bench 太慢”和 `Llama3.2 3B` 不过线，首先是 benchmark 口径和模型路径问题，不是单纯 attention 算子慢。
2. 在 true max + normal warm + graph-boundary PIC 模型下，`Llama3.2 3B / ctx=1024` 的 `cacheblend` / `epic` 已经全预算过线且单调。
3. Rhino 后续真正要优化的重点，重新回到了 Adreno sparse attention family / variant 默认路由，而不是继续怀疑 dense MLP 或 benchmark 框架本身。
