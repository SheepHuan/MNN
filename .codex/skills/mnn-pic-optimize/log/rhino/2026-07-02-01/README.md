# Rhino 2026-07-02 01 — x0 decode 慢因复盘 + CUDA decode-repair 对齐

> Superseded note (2026-07-02 03): 本小时记录保留为历史复盘，但其中
> `batch=1 family=fp_weight` 和“修 aux normal-style decode route 作为 P0”的结论
> 已被当前源码和 x0 日志推翻。当前权威结论见
> `log/rhino/2026-07-02-03/` 和仓库根目录 `RHINOPI_DECODE_REPAIR_HANDOFF.md`。

## 摘要

复盘 `RHINOPI_DECODE_REPAIR_HANDOFF.md` 里 x0 比 normal LLM 慢 2.5× 的根因，
修正 handoff "graph-boundary overhead" 的猜测，并对齐 CUDA decode-repair 执行流程。

核心结论：

1. **算子数量不是瓶颈**：normal LLM 2716 op vs PIC tiny-mlp 2699 op，几乎相同。
   PIC 把 72 个 MLP Conv 融成 24 个 `PicAdrenoTinyMlpWeightOnly` Extra，结构等价。
2. **x0 decode 的 attention 不是瓶颈**：每层 backend 实际 dispatch
   `decode_causal_attention_hd128_identity`（q=1 identity），不是 PicSparseAttention
   sparse kernel；24 层仅 8–18 ms/token。
3. **真瓶颈是 Raster(37%) + Convolution(21%) + BinaryOp/While/Cast(23%)**：
   PIC graph decode 形状下 dense Conv 落到 `OpenCLConvBufLowMemory family=fp_weight`
   慢路径，attention 后每个 `PagedAttention_raster_0` 有 layout 转换税。
4. **辅助 normal-style decode 模块已正确导出但从未被选中**：该模块图 dump 出来
   是 2716 op、24 PagedAttention、169 Conv、0 PicSparseAttention，和 normal LLM 等价；
   但 `usePicNormalDecodeModule` 在所有真实 TPOT run 里从未为 true（`for PIC
   normal-style decode` clone 标记出现 0 次）。P0 是修通它的路由条件。
5. **`p0_fused_decode_profile` 里的 `StaticModule forward failed code=2`** 是
   prefill suffix 阶段（`seq_len=1 add=1 all_seq=512 gen_seq=0`）的 resize
   NOT_SUPPORT，不是 decode 模块失败，与辅助 decode 模块无关。

## 对齐 CUDA decode-repair

用户指示：decode-repair 直接输入 token id + position id（按 logical 顺序），输出只看
最后一个 token logits；对比 CUDA 有效路径找误差。

decode-repair C++ 构造在 `llm.cpp::forwardVecWithPicDecodeRepair`（1461 行），
**CUDA 和 OpenCL 共享同一套 engine 代码**，不是后端差异点。x+1 行 forward 构造：

- `logicalIndices` = 选中 PIC logical + decode logical（长度 x+1）
- `mMeta->add = x+1`，`beginPicDecodeRecomputeRows` 置 `pic_decode_recompute_active=true`
- mask = `[1,1,x+1,kvLen]` full causal，position = `[1,x+1]` = logical index
- `forwardRaw(embeds, mask, pos)` = **x+1 行 causal prefill，只取最后一行 logits**

后端差异点：

- **CUDA（有效）**：x+1 行 compact dense → `conv_fpa_intb_1x1_tiny_gemv batch=x+1`
  或 `rows45_cublas`；attention 走 sparse flash tile。
- **OpenCL/Adreno（我们的）**：attention 走 `decode_repair_causal_attention_hd128`
  （`PagedAttentionBufExecution::ensureDecodeRepairKernel` 2901 行）；dense Conv 走
  `OpenCLConvBufLowMemory`。待补 x>0 profile 确认 dense Conv family 是否落到
  `pic_gemm_b4_c8` / image fast path 还是退化 `family=fp_weight` 慢路径。

## 待补

- 抓 x>0 decode-repair 的 device profile（当前 device 只有 x=0）。
- 修 `usePicNormalDecodeModule` 路由条件（`inDecode` / `mDecodeForwardActive` 在
  decode forwardRaw 时为何仍 false）。

## 产物

- 图 dump：`.cache/tmp/graph_dump/{normal,pic_tinymlp,pic_split,pic_decode_normal}.json`
- 脚本：`.cache/tmp/{analyze_graph,sum_decode,inspect_graph}.py`
- device profile：`.cache/tmp/rhino_logs/{pic_x0_graph_profile,pic_x0_attn_profile,p0_fused_decode_profile,generic_decode_request_profile}.log`
- 完整结论已写入 `RHINOPI_DECODE_REPAIR_HANDOFF.md` "2026-07-02 调查结论" 节。

详见 `context.md`。
