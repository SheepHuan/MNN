# Rhino 2026-07-02 01 Context — x0 decode 慢因复盘

> Superseded note (2026-07-02 03): 本文件中的 aux decode-module route 方向和
> x0 `family=fp_weight` 归因是旧复盘结论。当前源码已无
> `pic_decode_model` / `pic_decode_weight` route；当前 x0 batch=1 Conv 日志显示
> `use_fp_weight=0`，且源码 batch=1 直接走 `tuneGemvLowMemory`。继续定位时以
> `log/rhino/2026-07-02-03/` 和 `RHINOPI_DECODE_REPAIR_HANDOFF.md` 为准。

## 目标

复盘 `RHINOPI_DECODE_REPAIR_HANDOFF.md` 中 MiniCPM5-1B / Rhino Pi-X1 / Adreno OpenCL
PIC x0 decode（58 ms/token）比 normal LLM（23 ms/token）慢 2.5× 的根因，对齐 CUDA
decode-repair 执行流程，找出误差点。

## 环境

- device: Rhino Pi-X1 / Adreno OpenCL，`aidlux@192.168.101.227:18133`
- model: `OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp`（PIC）+ 辅助
  `OpenBMB__MiniCPM5-1B-pic-decode-normal`（normal-style decode 模块）
- normal LLM baseline: `OpenBMB__MiniCPM5-1B`（`.cache/mnn-llm-export/`）
- runtime cache: `/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/runtime_cache/opencl`

## 方法

1. `MNNConvert -f MNN --modelFile X --JsonFile Y` 把 4 个图 dump 成 JSON：
   normal、PIC tiny-mlp、PIC split、pic-decode-normal（辅助模块）。
2. Python 脚本统计 op type 数量（`.cache/tmp/analyze_graph.py`）。
3. 从 device 拉 4 个 profile log（`.cache/tmp/rhino_logs/`）。
4. 解析 decode attention kernel us、forward_raw 总时（`.cache/tmp/sum_decode.py`）。
5. 读 `llm.cpp` patch（staged）+ `StaticModule.cpp`/`PipelineModule.cpp` 错误路径。

## 关键数据

### 算子数量对比（见 README 表）

normal 2716 op vs PIC tiny-mlp 2699 op，差 17。PIC：Conv 169→97（−72，融成 24 Extra），
Attention 24→0，PicSparseAttention 0→22，Extra 0→24。

### x0 decode attention kernel（device profile）

`pic_x0_attn_profile.log`，ctx512 x=0：

```text
op=decode_causal_attention_hd128_identity layer=0..23 query=1 kv_len=513..530
  lane=128 identity_slot=1 us=320..1189（多数 320..820）
  24 层合计 8.7..18.5 ms/step（profile-detail 放大）
```

238 次 dispatch（10 step × ~24 层），0 次 sparse_flash / prefill_attention_fast。
**attention 占 forward_raw 的 5.8%（冷）..29%（热），不是主瓶颈**。

### graph profile type-level（pic_x0_graph_profile，profile-detail 放大）

```
Raster 317.0ms(37%) > Convolution 179.8ms(21%) > BinaryOp 106.2ms(12%)
> While 82.1ms(10%) > UnaryOp 56.6ms(7%) > PagedAttention 39.7ms(5%)
> LayerNorm 38.8ms(5%) > PicSparseAttention 24.7ms(3%) > Cast 7.0ms(1%)
> PicScoreAttention 2.6ms(0.3%)
total_ms=854.5 calls=2872 unique_ops=1052 unique_types=10（max_tokens=2）
```

attention 三类合计 ~67ms（8%）。normal LLM `llm_bench --profile` 也是 Raster 31% +
Conv 41%，所以这部分不是 PIC 独有；PIC 多的是 `PagedAttention_raster_0` layout 税
+ decode 形状下 `OpenCLConvBufLowMemory family=fp_weight` 慢路径。

### 辅助 decode 模块（pic-decode-normal）图

dump `pic_decode_normal.json`：2716 op，24 PagedAttention（4 输入 1 输出，无 budget），
169 Convolution（72 MLP + 24 o_proj + 72 qkv + 1 lm_head），0 PicSparseAttention，
0 Extra。`outputName=['logits','hidden_states']`。4 个 Input：input_ids /
attention_mask / position_ids / logits_index（无 pic_recompute_budget）。
**和 normal LLM 图完全等价**，是 handoff 想要的正确产物。

### usePicNormalDecodeModule 从未触发

`forwardRaw`（llm.cpp:1652）：

```cpp
const bool usePicNormalDecodeModule =
    inDecode && mPicDecodeModule != nullptr && mConfig->paged_attention() && mValidBlockSize.empty();
```

- `mPicDecodeModule` 已加载（日志 `Loaded PIC normal-style decode module`）。
- `paged_attention()` = true（llm_config.json）。
- `mValidBlockSize.empty()` = true（config 无 `chunk_limits`，已确认）。
- 但真实 TPOT run（`pic_x0_attn_profile` / `pic_x0_graph_profile`）里
  `Warning: module need new clone ... for PIC normal-style decode` 出现 **0 次**。
- 反证：decode forwardRaw 日志 `has_pic_budget=1` ⇒ `picBudget` 非空 ⇒
  `has_pic_recompute_budget() && !usePicNormalDecodeModule` 为 true ⇒
  `usePicNormalDecodeModule` 为 false ⇒ **`inDecode` 在 decode forwardRaw 时为 false**。
- `inDecode = mDecodeForwardActive || mContext->gen_seq_len > 0`。
  `mDecodeForwardActive` 由 `decode()` 里的 `ScopedDecodeForwardFlag` 设置（llm.cpp:2284），
  但真实 decode forwardRaw 走的是 `forwardVec`（prefill block 路径，llm.cpp:1903 记录
  forward_raw_begin/end），不是 `decode()` 路径。**需要确认 generate() decode 实际走
  哪条 forwardRaw，以及 `mDecodeForwardActive` 是否覆盖到它**。

注：`forward_raw_begin/end` 日志只在 `forwardVec`（llm.cpp:1896）打印，不在
`forwardVecWithPicDecodeRepair`（llm.cpp:1530）或 `generate()` decode 路径打印。
所以 `pic_x0_attn_profile` 的 10 条 `forward_raw_end` 是 **prefill block forward**
（seq_len=1 suffix prefill），不是 decode step。decode step 只在
`decode_causal_attention_hd128_identity` dispatch 里可见。

### p0_fused 的 StaticModule forward failed

`p0_fused_decode_profile.log` line 337：

```text
PIC server prefill debug step=bind_pic_source status=0 all_seq=512 tokens=512
PIC module debug StaticModule forward failed code=2 inputs=3 outputs=7 run_resize=1 run_compute=1
PIC module debug PipelineModule child failed index=0 type=StaticModule name= inputs=3 outputs=0 expected=7
PIC forward debug outputs empty seq_len=1 add=1 all_seq=512 gen_seq=0
```

- `code=2` = NOT_SUPPORT（ErrorCode.hpp），来自 `mSession->resize()`。
- `gen_seq=0` ⇒ prefill 阶段，`inDecode=false`，`usePicNormalDecodeModule=false`，
  走主模块。**这是主模块 suffix prefill（seq_len=1）的 StaticModule resize 失败**，
  不是辅助 decode 模块。
- 该 run 是 prefill debug，0 次 `decode_causal_attention`，未到 decode。
- `inputs=3` 与 `outputs=7` 是 StaticModule 内部 pipeline 的 input tensor 数 / output
  slot 数，不是命名输入输出数；主模块 5 命名输入、1..2 命名输出，但 PipelineModule
  内部 mInputSize/mOutputNumbers 另算。**此失败不影响真实 TPOT run**（真实 run 走
  `pic_x0_attn_profile`，无此错误，prefill 正常完成）。

## 对齐 CUDA decode-repair

### 共享 C++ 构造（非后端差异）

`llm.cpp::forwardVecWithPicDecodeRepair`（1461）+ `embeddingForPicDecodeRepair`
（1345）+ `genDecodeRepairAttentionMask`（1398）+ `genDecodeRepairPositionIds`（1434）：

- `logicalIndices` = 选中 PIC logical（repairLogical）+ decode logical（all_seq_len），
  长度 x+1。x=0 时 repairLogical 空，仅 [decodeLogical]，长度 1。
- `sparseTokenIds` = 对应 token id：PIC token 从 `picTokenEmbeddings` cache memcpy，
  decode token 走 `mDiskEmbedding`。
- `embeddingForPicDecodeRepair` 产 `[seqLen,1,hidden]`，前 repairRows 行是 PIC cache
  embedding，最后一行是 decode token embedding。
- `mMeta->add = logicalIndices.size()`（= x+1）。
- `beginPicDecodeRecomputeRows`（PagedKVMeta.hpp:268）置
  `pic_decode_recompute_active=true`、`pic_active_count=x+1`、
  `sparse_query_logical_indices=logicalIndices`。
- `genDecodeRepairAttentionMask`：`pic_decode_recompute_active` 时
  queryLen=pic_active_count，kvLen=logical_length；`full_causal_attention_mask` 时
  返回 `[1,1,queryLen,1]` 全 0，否则 `[1,1,queryLen,kvLen]` causal float mask
  （`k > logicalQ ? -inf : 0`）。
- `genDecodeRepairPositionIds`：`[1,seqLen]`（mrope 时 `[3,seqLen]`），position =
  logical index。
- `forwardRaw(inputEmbeds, repairMask, repairPos)`：**x+1 行 causal prefill，只取
  最后一行 logits**。

### 后端差异点

- **CUDA（有效，见 SKILL 278-279 + memory v15-smallm-weight-only-kernel）**：
  - x=1 → active rows=2 → `conv_fpa_intb_1x1_tiny_gemv batch=2`
  - x=3 → active rows=4 → `conv_fpa_intb_1x1_rows45_cublas batch=4`
  - 即 small-batch compact dense GEMM；attention 走 CUDA sparse flash tile。
  - Jetson x=1→x=3 cliff 根因：`int4GemvBatchLimit=3`，batch=4 离开 V14/V14_MB
    int4 GEMV 进 cuBLAS dense（读 dequant FP16 = 4× int4 带宽）。5 个 int4 变体
    （V14_MB retune / V15 v1-3 / V16 dp4a）全失败，dense 路径已达带宽峰值。

- **OpenCL/Adreno（我们的）**：
  - attention：`decode_repair_causal_attention_hd128_row32/row64`
    （`PagedAttentionBufExecution::ensureDecodeRepairKernel`，2901 行）。
  - dense Conv：`OpenCLConvBufLowMemory`，family 选择由
    `decision_source=adreno_family_cache` 决定，prefill 形状选 `family=fp_weight`。
  - **待补**：x>0 decode-repair 的 device profile，确认 x=1/3/5/7 的 dense Conv
    family 是 `pic_gemm_b4_c8` / image-backed 还是 `fp_weight`，以及 attention
    kernel 是否 `decode_repair_causal_attention_hd128`。这是 CUDA cliff 的 OpenCL
    对应判断点。

## 待办

1. 抓 x>0 decode-repair device profile（`MNN_PIC_GRAPH_PROFILE=1` + x=1/3/5/7）。
2. 修 `usePicNormalDecodeModule` 路由：确认 `mDecodeForwardActive` 是否覆盖真实
   decode forwardRaw；若不覆盖，调整 `inDecode` 判定或 clone 时机。
3. 修通后重测 x0 ctx512/1024，确认是否逼近 normal 23 ms/token。
4. x>0 跑通后再对齐 CUDA 判断 dense Conv family 误差。

## 命令备忘

```bash
# dump 图
.cache/output/mnn/artifacts/x64/bin/MNNConvert -f MNN \
  --modelFile .cache/weight/OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp/llm.mnn \
  --JsonFile .cache/tmp/graph_dump/pic_tinymlp.json

# 拉 device profile
ssh aidlux@192.168.101.227 'cat /mnt/nvme/mnn_pic_opencl/logs/pic_decode_generic_x0_graph_profile_20260701_214051.log' \
  > .cache/tmp/rhino_logs/pic_x0_graph_profile.log

# 分析
conda run -n kvshare-edge python3 .cache/tmp/analyze_graph.py
conda run -n kvshare-edge python3 .cache/tmp/sum_decode.py
```
