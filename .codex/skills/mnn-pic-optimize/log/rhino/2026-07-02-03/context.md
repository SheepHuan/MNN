# Rhino 2026-07-02 03 Context — x0 decode 定位校正

## 目标

继续 `RHINOPI_DECODE_REPAIR_HANDOFF.md` 的 x0-only 定位：解释为什么 Rhino Pi-X1 /
Adreno OpenCL / MiniCPM5-1B PIC `full-reuse x=0` decode 比 normal LLM 慢，并把过期
结论替换为当前源码和日志能证明的结论。

## 当前证据

### attention 不是完整根因

`.cache/tmp/sum_decode.py` 解析 `.cache/tmp/rhino_logs/pic_x0_attn_profile.log`：

```text
decode steps reconstructed: 10
attention total per step: 8.668 ms, 17.295 ms, 18.337 ms, 18.386 ms,
18.536 ms, 11.428 ms, 26.748 ms, 26.180 ms, 26.608 ms, 24.002 ms
forward_raw_end steady-ish: 62.22 ms .. 85.71 ms, with polluted/cold entries
```

PagedAttention x0 kernel is `decode_causal_attention_hd128_identity` with q=1,
identity slot, lane=128. Kernel build entry:
`source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
`ensureDecodeCausalKernelHD128Identity()` around lines 2854-2898.

### x0 batch=1 Conv is not currently `fp_weight`

`source/backend/opencl/execution/buffer/ConvBufLowMemoryExecution.cpp`:

- `onResize()` lines around 1799-1802: if `batch == 1`, it calls
  `tuneGemvLowMemory(input, output)` directly.
- The `OpenCLConvBufLowMemory profile ... family=...` print around lines 2090-2094
  is gated by `profileTinyRows = batch > 1 && batch <= 16`, so it cannot prove
  x0 batch=1 family selection.

Local log count:

```text
.cache/tmp/rhino_logs/pic_x0_graph_profile.log:
  batch1_begin=499, batch1_not_fp=499
.cache/tmp/rhino_logs/pic_x0_attn_profile.log:
  batch1_begin=1676, batch1_not_fp=1674
.cache/tmp/rhino_logs/generic_decode_request_profile.log:
  batch1_begin=1512, batch1_not_fp=1510
.cache/tmp/rhino_logs/p0_fused_decode_profile.log:
  family_profile=120, all batch=512, all family=fp_weight
```

Conclusion: `family=fp_weight` is prefill/full-prompt evidence in the current logs, not
x0 decode evidence.

### graph/Raster/shape overhead is the strongest current attribution

`.cache/tmp/rhino_logs/pic_x0_graph_profile.log` type totals:

```text
Raster 317.011 ms
Convolution 179.815 ms
BinaryOp 106.223 ms
While 82.125 ms
UnaryOp 56.584 ms
PagedAttention 39.731 ms
LayerNorm 38.789 ms
PicSparseAttention 24.660 ms
Cast 6.977 ms
PicScoreAttention 2.624 ms
```

`MNN_PIC_GRAPH_PROFILE` is finish-polluted and should only be used for attribution ratios,
but the ordering is still useful: x0 is dominated by Raster/graph/elementwise plus normal
dense work, not by a single sparse attention kernel.

Graph diff from `.cache/tmp/analyze_graph.py`:

```text
normal ops: 2716
PIC tinymlp ops: 2699
PIC deltas: BinaryOp +38, Unsqueeze +52, GatherV2 +26, Squeeze +26, Cast +24,
PicSparseAttention +22, PicScoreAttention +1, PagedAttention +1, Convolution -72,
Attention -24
```

This is consistent with x0 still paying graph-boundary identity machinery:
`active_indices`, `_pic_gather_rows`, Cast/Shape/While/Rank mask plumbing, and layout
conversions after PagedAttention.

Graph provenance from `.cache/tmp/graph_dump/pic_tinymlp.json`:

```text
PIC graph inputs include: position_ids, pic_recompute_budget, logits_index, input_ids, ...
PicScoreAttention op#234:
  inputs  = [..., attention_mask, pic_recompute_budget]
  outputs = [attention_output, active_indices]
active_indices direct consumers by type:
  Cast: 24
```

The exporter source explains why this happens:

- `llmexport.py` exports `pic_recompute_budget` as a normal model input.
- `custom_op.py::PagedAttentionWithBudgetOp.symbolic()` emits a 2-output
  `PicScoreAttention`; output 1 is `active_indices`.
- `transformers.py::_pic_gather_rows()` uses `torch.index_select`, and the decoder
  applies it to gate, residual, norm hidden, and per-layer inputs.
- `model.py` applies the same `active_indices` to later masks/rotary and switches
  later attention layers to `PicSparseAttention`.

So the extra `Cast/GatherV2/Shape/Rank/BinaryOp/Unsqueeze/Squeeze` nodes are not
caused by the KV cache storage format itself. They are the graph-boundary sparse-row
plumbing that remains in the x0 identity case.

Additional graph-profile grouping from the same finish-polluted log:

```text
PagedAttention_raster_0: 19.203 ms over 24 records / 71 calls
self_attn Raster:       178.740 ms
linear_raster:           76.459 ms
mlp_raster:              76.984 ms
While:                   78.903 ms
Pic attention ops:       27.284 ms
PagedAttention ops:      39.731 ms
```

These numbers are attribution only, but they bound the identity/layout tax and match
the root-cause direction: the x0 gap is graph-boundary/layout dominated, not a pure
PagedAttention-kernel issue.

### normal reference profile exists but is not same server surface

`.codex/skills/mnn-pic-optimize/log/rhino/2026-07-01-15/context.md` records the
normal MiniCPM5-1B `llm_bench` reference:

```text
ctx512: 42.6709 tok/s = 23.44 ms/token
ctx1024: 40.1293 tok/s = 24.92 ms/token
ctx512 --profile type share:
  Convolution 40.66%
  Raster 31.27%
  BinaryOp 8.50%
  Attention 6.32%
  While 5.90%
  UnaryOp 4.35%
  LayerNorm 2.95%
```

This supports the conclusion that normal decode is also dense/Raster heavy and that
PIC's extra gap is not solely attention. It does not fully close the timing-caliber
issue because normal is `llm_bench` while PIC x0 evidence comes from server/generate
profile logs.

## Corrected P0

1. Align normal and PIC timing/attribution surfaces before reporting the ratio. Current
   normal is `llm_bench`; current PIC is HTTP server/generate path.
2. Add an x0 identity fast path at the graph/export/backend level for
   `pic_recompute_budget=seq_len=1` and identity `active_indices`. It must keep
   PagedCache/slot-table semantics and only eliminate redundant identity graph work.
3. Remove per-layer PagedAttention layout/Raster tax by making the x0 identity kernel
   produce the layout consumed by the following projection where possible.
4. Only after same-shape evidence shows batch=1 Conv is slower than normal should
   `tuneGemvLowMemory` become a performance target. Do not route x0 through the
   compact-family `fp_weight` hypothesis.

## Localization closure

The root-cause localization is closed for x0: the current PIC graph-boundary model is
not equivalent to "normal decode plus a different PagedAttention KV storage layout".
It still carries sparse-boundary `pic_recompute_budget` / `active_indices` machinery,
and the x0 identity value is not folded away. The remaining work is optimization and
formal same-caliber reporting:

1. Same-caliber normal-vs-PIC attribution is still needed before publishing a final
   ratio.
2. The first optimization prototype should remove identity active-index users and
   PagedAttention layout Raster without bypassing PagedCache or slot-table semantics.
3. x>0 decode-repair profiling remains deferred and must not be mixed into this x0
   conclusion.

## Re-run commands

```bash
conda run -n kvshare-edge python .cache/tmp/sum_decode.py
conda run -n kvshare-edge python .cache/tmp/analyze_graph.py
```

Useful remote profile env:

```text
MNN_PAGED_ATTENTION_PROFILE=1
MNN_PIC_REQUEST_PROFILE=1
MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=1000
```

Do not use `MNN_PAGED_ATTENTION_PROFILE_DETAIL=1` or graph profile numbers as formal
latency; they are for attribution only.
