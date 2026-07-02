# RhinoPi Decode-Repair Handoff

Date: 2026-07-02 (rewritten as a clean memory point; supersedes the earlier
"Current Patch State" / aux decode-module P0).

This handoff is for continuing the Rhino Pi-X1 / Adreno OpenCL / MiniCPM5-1B
decode-repair optimization. The goal of the present pass is **x=0 only**:
locate why PIC `full-reuse x=0` decode is ~2.5× slower than normal LLM decode,
and give reproducible evidence + code-level root cause + fix direction.

## Hard constraints (do not violate)

- x=0 means `repair_tokens=0` / no-repair decode; do NOT send `decode_refine`;
  do NOT analyze x>0 sparse repair here.
- Make PIC x=0 decode TPOT approach normal LLM decode **under correct
  PagedCache / slot-table / PagedAttention semantics** — no fake normal path.
- Do NOT restore `pic_decode_model` / `pic_decode_weight` auxiliary normal-decode
  module. (Already removed from source — see "Aux decode module is gone" below.)
- Do NOT bypass PagedCache for x=0, and do NOT introduce an x=0 special path
  that skips PagedCache.
- Normal baseline MUST use the real normal export:
  `/mnt/nvme/mnn_pic_opencl/models/normal/OpenBMB__MiniCPM5-1B/config_opencl_greedy.json`.
- PIC baseline MUST use the graph-boundary PIC model:
  `/mnt/nvme/mnn_pic_opencl/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary/config_opencl_greedy.json`.
- Both sides MUST use the same OpenCL runtime cache:
  `/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/runtime_cache/opencl/mnn_cachefile.bin`.

## Known baseline (current reference; timing surfaces still differ)

| | normal LLM | PIC full-reuse x=0 | gap |
| --- | ---: | ---: | ---: |
| decode TPOT | 25.96 ms/tok (`llm_bench -p 513 -n 8`, decode_tps=38.52) | ~60.90 ms/tok | +34.94 ms/tok |

(x=0 log confirms the old `decode_repair_causal_attention` helper hit count = 0;
PIC x=0 q=1 PagedAttention single layer ~0.74 ms, 24 layers ~18 ms/token, so
attention is NOT the main cost.)

Existing normal `llm_bench --profile` attribution from
`.codex/skills/mnn-pic-optimize/log/rhino/2026-07-01-15/context.md` shows
ctx512 normal decode is also dense/Raster heavy:
`Convolution 40.66%`, `Raster 31.27%`, `BinaryOp 8.50%`, `Attention 6.32%`,
`While 5.90%`, `UnaryOp 4.35%`, `LayerNorm 2.95%`. This is useful type-level
evidence, but it is still not the same execution surface as PIC HTTP server
decode; keep that caveat when reporting ratios.

## Aux decode module is GONE from source (do not chase it)

The earlier handoff described a `pic_decode_model` / `pic_decode_weight`
auxiliary normal-style decode module with a `usePicNormalDecodeModule` route,
and proposed "P0 = fix the route condition". **That direction is dead.**

Verified against current `impl/MNN` source: grep for `usePicNormalDecodeModule`,
`mPicDecodeModule`, `pic_decode_model`, `pic_decode_weight`, `pic-decode-normal`
in `.cpp/.h/.hpp` returns **zero hits**. These symbols survive only in this
handoff and in `log/rhino/2026-07-02-01/` markdown. Current `forwardRaw`
(`transformers/pic_llm/engine/src/llm.cpp:1647+`) has only the plain
`mModulePool` clone-from-`mModule` path; there is no decode-time module swap
and no `pic_decode_model` loading.

Therefore x=0 is a **clean-slate in-graph optimization of the PIC
graph-boundary model**. There is no routing bug to fix. The stale
`.cache/tmp/graph_dump/pic_decode_normal.json` is a dump from the removed aux
module — do not treat it as a current selectable route; it is at most a
reference for what a normal-equivalent graph looks like.

## Root cause (current best evidence, code-level)

Not attention, not op count, and **not proven to be batch=1 `fp_weight` Conv**.
The current localization points at PIC graph-boundary overhead plus a
normal-vs-PIC execution-caliber gap:

### Primary: x=0 still pays PIC graph-boundary / layout / shape machinery

- PIC graph-boundary op counts are close to normal: 2699 vs 2716 (diff 17).
  The count alone is not the issue, but PIC changes the mix: `BinaryOp +38`,
  `Unsqueeze +52`, `GatherV2 +26`, `Squeeze +26`, `Cast +24`,
  `PicSparseAttention +22`, `PicScoreAttention +1`, `PagedAttention +1`,
  `Convolution -72`, `Attention -24`.
- The graph still has `PicScoreAttention`(layer=1) + `PicSparseAttention`(layer>=2)
  nodes with the extra `pic_recompute_budget` input and `active_indices` output.
  At x=0 the effective active row is identity (`budget=seq_len=1`), but the
  exported graph still executes downstream `_pic_gather_rows(..., -2)`,
  Cast/Shape/While/Rank-related mask plumbing, and per-attention layout/Raster
  conversions that normal LLM decode does not need.
- Code-level provenance for these ops:
  - `llmexport.py` exports `pic_recompute_budget` as a real graph input.
  - `custom_op.py::PagedAttentionWithBudgetOp.symbolic()` emits
    `PicScoreAttention` with two outputs: compact attention output and
    `active_indices`.
  - `transformers.py::_pic_gather_rows()` lowers to `torch.index_select`; it is
    applied to gate, residual, norm hidden, and per-layer inputs.
  - `model.py` applies the same `active_indices` to later-layer masks, rotary
    embeddings, and switches later layers to `PicSparseAttention`.
  - Therefore x=0 still materializes a dynamic sparse-boundary graph. The
    runtime value is identity, but the converter cannot fold it because
    `active_indices` is produced by a custom op at runtime.
- Current graph dump proof (`.cache/tmp/graph_dump/pic_tinymlp.json`):
  `PicScoreAttention` op#234 consumes `pic_recompute_budget` and produces
  `PicScoreAttention_output_1`; that tensor has 24 direct `Cast` consumers.
  The graph-level deltas vs normal are exactly the expected dynamic-index
  lowering: `GatherV2 +26`, `Cast +24`, `BinaryOp +38`, `Unsqueeze +52`,
  `Squeeze +26`, plus `PicScoreAttention +1` and `PicSparseAttention +22`.
- Graph-profile type share from `.cache/tmp/rhino_logs/pic_x0_graph_profile.log`
  (ratio only; absolute time is amplified by `queue.finish`) is:
  **Raster 317.011 ms > Convolution 179.815 ms > BinaryOp 106.223 ms >
  While 82.125 ms > UnaryOp 56.584 ms > PagedAttention 39.731 ms >
  LayerNorm 38.789 ms > PicSparseAttention 24.660 ms > Cast 6.977 ms >
  PicScoreAttention 2.624 ms**.
- Within the same finish-polluted graph profile, the top-1000 op records give a
  useful bound for the identity/layout tax: `PagedAttention_raster_0` records
  sum to 19.203 ms over 24 layers, all `self_attn` Raster records sum to
  178.740 ms, `linear_raster` records sum to 76.459 ms, `mlp_raster` records sum
  to 76.984 ms, and `While` records sum to 78.903 ms. These are attribution
  numbers, not formal latency.
- This distribution says the x0 residual is dominated by graph/Raster/elementwise
  and normal dense work, not by a single sparse attention kernel. The next
  useful breakdown is a same-caliber normal-vs-PIC graph profile or an x0 server
  path that reports the same type totals on both models.

### Attention is a bounded secondary cost

- x=0 decode per-layer attention dispatches `decode_causal_attention_hd128_identity`
  (q=1, identity slot, lane=128), not the old decode-repair helper and not the
  sparse prefill kernel. The kernels are built in
  `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
  `ensureDecodeCausalKernelHD128Identity()` around lines 2854-2898.
- `.cache/tmp/sum_decode.py` reconstructs 10 x0 steps from
  `.cache/tmp/rhino_logs/pic_x0_attn_profile.log`: attention totals are about
  8.7-26.7 ms/step, while `forward_raw_end` is about 62-85 ms for the steadier
  steps. Attention can matter, especially around the 18-26 ms band, but it does
  not explain the full ~35 ms/token gap alone.
- Hydrate in that same log is prefill/setup cost (`31.73 ms` over 48 layer calls)
  and is not the decode TPOT root cause.

### The previous `fp_weight batch=1` conclusion is contradicted by current evidence

- Source: in `ConvBufLowMemoryExecution::onResize`, `batch == 1` directly calls
  `tuneGemvLowMemory(input, output)` and skips the compact-family selection that
  can print/choose `family=fp_weight`.
- Source: the `OpenCLConvBufLowMemory profile ... family=...` print is gated by
  `profileTinyRows = batch > 1 && batch <= 16`, so it does not cover x0
  `batch=1` Conv at all.
- Logs: `.cache/tmp/rhino_logs/pic_x0_graph_profile.log` has 499 batch=1
  `conv_lowmem_begin` records and all show `use_fp_weight=0`. The current
  x0 attention profile similarly shows batch=1 `use_fp_weight=0` for the
  overwhelming majority of records.
- Logs: the `family=fp_weight` records found in
  `.cache/tmp/rhino_logs/p0_fused_decode_profile.log` are all `batch=512`, i.e.
  prefill/full-prompt shape, not x0 decode.
- Therefore do not spend the x0 pass widening `fp_weight`/compact-family
  admission. That may still matter for x>0 decode-repair or prefill, but it is
  not the current x0 localization.

### Red herrings already closed

- `StaticModule forward failed code=2` in an older `p0_fused_decode_profile` was
  a prefill suffix-stage resize NOT_SUPPORT (`seq_len=1 add=1 all_seq=512
  gen_seq=0`) on the main module, not decode; that run had 0
  `decode_causal_attention` dispatches. Real x0 TPOT/profile logs do not show
  that failure.
- The old `pic_decode_model` / `pic_decode_weight` auxiliary module route is
  gone from current source and must not be restored. The stale
  `.cache/tmp/graph_dump/pic_decode_normal.json` can only be used as a reference
  for a normal-equivalent graph shape, not as a route to revive.

## Shared decode-repair engine path (context for x>0, not analyzed here)

`forwardVecWithPicDecodeRepair` (`llm.cpp:1440`) is shared C++ engine code, not a
backend diff. It builds an x+1-row causal prefill (`logicalIndices` + decode
logical, mask `[1,1,x+1,kvLen]`, position=logical), `mMeta->add=x+1`,
`pic_decode_recompute_active=true`; only the last row's logits matter. OpenCL
attention uses `decode_repair_causal_attention_hd128` (`ensureDecodeRepairKernel`
~:2901) + `OpenCLConvBufLowMemory` dense. CUDA analog uses
`conv_fpa_intb_1x1_tiny_gemv` (batch=x+1) or `rows45_cublas` (batch=4) — see
memory `v15-smallm-weight-only-kernel` for the Jetson x=3 cliff.

## Fix direction (compliant with hard constraints)

1. **P0 — same-caliber attribution**: collect normal and PIC x0 under matching
   timing surfaces before reporting ratios. Best is same server/generate path if
   available; otherwise pair `llm_bench --profile` normal decode with PIC server
   graph profile and explicitly mark the surface mismatch.
2. **P0 — decode-specialized graph, shared weights**: x0 decode should not run a
   graph that contains score-layer boundary outputs at all. Export or derive a
   decode-only graph with no `pic_recompute_budget`, no 2-output
   `PicScoreAttention`, no `active_indices`, and no downstream
   `_pic_gather_rows` / Cast / Shape / GatherV2 chain. This graph must still use
   PagedAttention, the same current-request PagedCache and slot table, and the
   same external weight file as the PIC graph-boundary model.
3. **P0/P1 — per-attention Raster/layout tax**: make the OpenCL x0 identity
   PagedAttention path emit the layout consumed by the following projection so
   `PagedAttention_raster_0`-style conversions do not appear 24 times.
4. **P1 — dense GEMV parity check**: compare batch=1 normal and PIC Conv/GEMV
   kernel timings under the same profile sync. Optimize `tuneGemvLowMemory` only
   if same-shape PIC batch=1 Conv is slower than normal. Do not route this through
   the compact-family `fp_weight` hypothesis unless new batch=1 evidence appears.

### Decode-specialized graph design note

The preferred fix is not a runtime identity fold inside `GatherV2` or related
small ops. If the static graph still contains
`PicScoreAttention_output_1 -> Cast -> Shape/Rank/BinaryOp -> GatherV2`, MNN
will still pay graph scheduling, shape propagation, possible layout barriers,
debug/profile callbacks, and op dispatch even when every op returns an identity
result.

Instead, decode should select a graph whose topology never contains the
score-layer side branch:

```text
prefill/sparse graph:
  layer 1 PicScoreAttention -> active_indices -> gather residual/mask/rotary/...
  layer 2..23 PicSparseAttention

x0 decode graph:
  layer 0..23 PagedAttention decode
  no pic_recompute_budget input
  no active_indices output
  no Cast/GatherV2/Shape chain derived from active_indices
```

This is different from restoring the old removed `pic_decode_model` /
`pic_decode_weight` route. A new decode graph, if added, should be an explicit
current design with the following constraints:

- It reads the same `llm.mnn.weight` / external weight storage as the PIC model;
  do not copy weights into a second `.weight` file.
- It shares the same runtime manager, current request PagedCache, slot table,
  tokenizer state, logits selection, sampler, and KV lifecycle.
- It only changes graph topology for decode: remove score-layer sparse-boundary
  bookkeeping that is meaningful for prefill/cacheblend/epic but redundant for
  q=1 no-repair decode.
- Memory expectation: the large model weights should not be duplicated if both
  graphs reference the same external weight resource. There will still be small
  extra cost for the decode graph metadata, module/session state, compiled kernel
  cache entries, and temporary tensors; verify with Rhino RSS/PSS and MNN memory
  logging before reporting "no memory increase".
- Correctness gate: generated tokens and logits for x0 must match the current
  PagedCache/PagedAttention semantics; this is not a bypass to normal LLM KV and
  must not skip PagedCache.

Implementation should live in `transformers/pic_llm/export`, not as an ad-hoc
runtime graph rewrite:

- A normal PIC export should produce two graph files in the same model directory:

  ```text
  llm.mnn          # prefill / cacheblend / epic / sparse graph-boundary graph
  llm_decode.mnn   # x0 autoregressive decode graph
  llm.mnn.weight   # single shared external weight file
  ```

- `llm.mnn` keeps the current graph-boundary behavior: `pic_recompute_budget`
  input, score-layer `PicScoreAttention` with `active_indices`, and later
  `PicSparseAttention`.
- `llm_decode.mnn` should be exported from the same HF/PIC wrapper with
  `pic_recompute_budget` disabled, equivalent to the existing
  `--no_pic_recompute_budget` graph shape: every layer remains PagedAttention,
  but there is no score-layer side output and no `_pic_gather_rows` lowering.
- `config.json` / `llm_config.json` should record the decode graph explicitly,
  for example `llm_decode_model: "llm_decode.mnn"`. Do not add a second decode
  weight field unless the loader truly needs one; prefer reusing the existing
  `llm_weight: "llm.mnn.weight"`.
- Export flow sketch:

  ```text
  1. Export main PIC graph as today:
     dst_name=llm, pic_recompute_budget=true.
  2. Export decode graph in the same run:
     dst_name=llm_decode, pic_recompute_budget=false,
     pic_decode_repair_outputs=false.
  3. Convert both with the same quantization / fusion / device contract.
  4. Verify the decode graph contains PagedAttention only and no
     PicScoreAttention/PicSparseAttention/Gather chain from active_indices.
  5. Verify weight sharing: if converter emits llm_decode.mnn.weight, compare it
     byte-for-byte / hash against llm.mnn.weight before deleting or ignoring it.
     If the files differ, do not silently claim shared weights; add converter
     support for deterministic external-weight reuse first.
  ```

- Runtime selection should be narrow: normal prefill, full-reuse setup,
  cacheblend, epic, and x>0 decode-repair continue to use `llm.mnn` unless a
  separate repair-specialized graph is deliberately designed later. Only x0
  normal autoregressive decode should use `llm_decode.mnn`.

## Dual-graph export smoke after 2026-07-02 prototype

Prototype exporter change exists in `transformers/pic_llm/export/llmexport.py`:

- `--pic_export_decode_graph` exports a second graph, `llm_decode.mnn`, in the
  same model directory after the normal PIC graph.
- Main graph keeps `pic_recompute_budget=true`.
- Decode graph temporarily sets `pic_recompute_budget=false` and
  `pic_decode_repair_outputs=false`, then exports through the same converter and
  unloaded weight map.
- If `llm_decode.mnn.weight` is byte-identical to `llm.mnn.weight`, the decode
  weight file is deleted and `config.json` records:

  ```json
  {
      "llm_decode_model": "llm_decode.mnn",
      "llm_decode_weight": "llm.mnn.weight",
      "llm_decode_shared_weight": true
  }
  ```

Local prototype output:

```text
.cache/weight/OpenBMB__MiniCPM5-1B-pic-boundary-rhino-dualgraph/
  llm.mnn                 445136 bytes
  llm_decode.mnn          419768 bytes
  llm.mnn.weight       550616714 bytes
  llm_decode.mnn.weight   absent, intentionally shared
```

Structural verification from MNNConvert JSON:

```text
llm.mnn:
  Input=5, includes pic_recompute_budget
  PagedAttention=1, PicScoreAttention=1, PicSparseAttention=22

llm_decode.mnn:
  Input=4, no pic_recompute_budget
  PagedAttention=24, PicScoreAttention=0, PicSparseAttention=0
```

Important converter note: the successful export used
`.cache/build/mnn/x64_pic_opencl/MNNConvert` with
`LD_LIBRARY_PATH=$PWD/.cache/build/mnn/x64:$LD_LIBRARY_PATH`. An earlier export
with `.cache/output/mnn/artifacts/x64/bin/MNNConvert` wrote unknown PIC ops as
`type=-1` in the flatbuffer; Rhino then failed shape inference with
`PIC shape failure ... type=-1`.

Temporary standalone decode configs were generated only for smoke testing:

- `config_opencl_greedy.json`: loads `llm.mnn`.
- `config_decode_opencl_greedy.json`: loads `llm_decode.mnn`.
- `llm_config_decode.json`: same as `llm_config.json` but with
  `pic_recompute_budget=false`, otherwise runtime tries to feed a fifth input to
  `llm_decode.mnn`.

Rhino smoke result, target `aidlux@192.168.101.227`, port `18133`:

```text
prefill graph:
  config: config_opencl_greedy.json
  log: /mnt/nvme/mnn_pic_opencl/logs/dualgraph_prefill_fixed_20260702_120435.log
  request: POST /v1/prefill/text id=dualgraph-smoke-doc-fixed
  result: HTTP 200, cache_status=built, token_count=19, layer_count=24
  grep: no type=-1, no shape failure, no forwardRaw outputs empty, no ERROR

decode graph:
  config: config_decode_opencl_greedy.json
  log: /mnt/nvme/mnn_pic_opencl/logs/dualgraph_decode_20260702_120709.log
  request: POST /v1/chat/completions, no pic_cache, max_tokens=8
  result: HTTP 200, prompt_tokens=13, completion_tokens=8
  timing: prefill_latency_s=1.0001, decode_tpot_ms=52.8729
  log proof: 24 layers report PagedAttention decode=1, q=1, new_kv=1;
             no shape failure / type=-1 / forwardRaw empty / ERROR
```

This is a graph/backend smoke, not a formal quality or performance result. The
current C++ runtime still does not automatically load and route
`llm_decode_model`; the decode graph was tested as the primary `llm_model` via
`config_decode_opencl_greedy.json`. Next implementation step is a narrow runtime
route that uses `llm_decode.mnn` only for x0 autoregressive decode while keeping
prefill/full-reuse/cacheblend/epic/repair on `llm.mnn`.

## Formal-device dual-graph integration after 2026-07-02 runtime route

Runtime route implemented in `transformers/pic_llm/engine/src/llm.cpp`:

- `config.json` may now contain `llm_decode_model`, `llm_decode_weight`, and
  `llm_decode_shared_weight`.
- Runtime loads both modules with the same `RuntimeManager` and the same
  `PagedKVMeta`, so PagedCache / slot-table state is shared.
- `forwardRaw()` routes to `llm_decode.mnn` only for x0 autoregressive decode:
  decode active, `seqLen == 1`, not all-logits, and
  `mPicDecodeRepair.enabled == false`.
- Decode graph route does not append `pic_recompute_budget`; expected graph
  inputs are the normal four decode inputs.
- `MNN_PIC_DECODE_DEBUG=1` prints `PIC decode graph loaded ... inputs=4` and
  `PIC decode graph route ... inputs=4`.

Device-contract dualgraph exports:

```text
OrangePi/Mali:
  local:  .cache/weight/OpenBMB__MiniCPM5-1B-pic-boundary-orangpi-dualgraph
  remote: /mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary-orangpi-dualgraph
  config: config_opencl_greedy.json
  contract: pic_export_device=orangepi, pic_decode_fusion_family=generic

Jetson/CUDA:
  local:  .cache/weight/OpenBMB__MiniCPM5-1B-pic-boundary-jetson-dualgraph
  remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/weight/OpenBMB__MiniCPM5-1B-pic-boundary-jetson-dualgraph
  config: config_cuda_greedy.json
  contract: pic_export_device=jetson, pic_decode_fusion_family=cuda
```

Both exports used the known-good converter path:

```bash
LD_LIBRARY_PATH=$PWD/.cache/build/mnn/x64:$LD_LIBRARY_PATH \
MNNCONVERT_PATH=.cache/build/mnn/x64_pic_opencl/MNNConvert \
MNN_LLM_EXPORTER=pic \
bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh \
  /root/.cache/modelscope/hub/models/OpenBMB/MiniCPM5-1B \
  -- --paged_kv_max_tokens 4096 --pic_export_decode_graph
```

Local graph structure check for both OrangePi and Jetson exports:

```text
llm.mnn:
  Input=5, PagedAttention=1, PicScoreAttention=1, PicSparseAttention=22,
  type=-1=0, contains pic_recompute_budget

llm_decode.mnn:
  Input=4, PagedAttention=24, PicScoreAttention=0, PicSparseAttention=0,
  type=-1=0, no pic_recompute_budget

weights:
  llm_decode_weight=llm.mnn.weight
  llm_decode_shared_weight=true
```

Runtime artifacts built and synced:

```text
OrangePi artifact:
  local:  .cache/output/mnn/artifacts/orangepi5plus
  remote: /mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus

Jetson artifact:
  local:  .cache/output/mnn/artifacts/jetson
  remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda
```

Route smoke with `MNN_PIC_DECODE_DEBUG=1`:

```text
OrangePi, 192.168.101.113:18132:
  log: /mnt/ssd/code/.cache/mnn_opencl_pic/logs/dualgraph_route_orangepi_20260702_1239.log
  result: HTTP 200, prompt_tokens=13, completion_tokens=8
  proof: "PIC decode graph loaded ... inputs=4" and route lines for
         gen_seq=1..7, each with inputs=4.
  grep: no type=-1, no forwardRaw empty, no ERROR. First run had
        "Cache invalid, will be reset" from OpenCL runtime cache warm-up.

Jetson, 192.168.101.192:18131:
  log: /home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/dualgraph_route_jetson_20260702_1239.log
  result: HTTP 200, prompt_tokens=13, completion_tokens=8
  proof: "PIC decode graph loaded ... inputs=4" and route lines for
         gen_seq=1..7, each with inputs=4.
  grep: no type=-1, no forwardRaw empty, no ERROR.
```

Decode performance check, 13-token prompt and 32 generated tokens:

```text
Jetson/CUDA:
  PIC dualgraph pic_server no-debug:
    files: .cache/logs/dualgraph-perf/jetson_pic_dualgraph_decode32_{1,2,3}.json
    mean decode_tpot_ms = 19.41
  normal LLM llm_bench:
    remote json: .cache/logs/normal_minicpm5_decode32_jetson_20260702_1252.json
    decode_tps = 55.47, tpot ~= 18.03 ms/token
  conclusion: recovered to normal-LLM granularity, about 1.08x slower than
              normal in this short decode test.

OrangePi/OpenCL:
  PIC dualgraph pic_server no-debug:
    files: .cache/logs/dualgraph-perf/orangepi_pic_dualgraph_decode32_{1,2,3}.json
    mean decode_tpot_ms = 254.52
  normal LLM llm_bench:
    remote json: /mnt/ssd/code/.cache/mnn_opencl_pic/logs/normal_minicpm5_decode32_orangepi_20260702_1252.json
    decode_tps = 24.23, tpot ~= 41.26 ms/token
  conclusion: dualgraph switching is correct, but OrangePi decode performance is
              not recovered in this first run. This line was superseded by the
              later hd128-dispatch rebuild/profile below: the generic
              PagedAttention fallback was fixed, but the remaining gap is still
              not normal-LLM parity because non-attention graph/runtime work
              dominates.
```

## Reproduction & artifacts

- Normal baseline:
  `/mnt/nvme/mnn_pic_opencl/models/normal/OpenBMB__MiniCPM5-1B/config_opencl_greedy.json`
- PIC baseline:
  `/mnt/nvme/mnn_pic_opencl/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary/config_opencl_greedy.json`
- Shared OpenCL cache:
  `/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/runtime_cache/opencl/mnn_cachefile.bin`
- Remote target: `aidlux@192.168.101.227`, fixed port `18133`.
- Graph dumps (local): `.cache/tmp/graph_dump/{normal,pic_tinymlp,pic_split}.json`
  (`pic_decode_normal.json` is a STALE dump from the removed aux module —
  reference only, not a current route).
- Analysis scripts (local): `.cache/tmp/{analyze_graph,sum_decode,inspect_graph}.py`.
- Device profiles (local, x=0 only):
  `.cache/tmp/rhino_logs/{pic_x0_graph_profile,pic_x0_attn_profile,p0_fused_decode_profile,generic_decode_request_profile}.log`

Profile switches (see `mnn-pic-optimize` SKILL "逐层 Profile 口径"):
- OpenCL attention profile (no finish pollution, use for real per-layer us):
  `MNN_PAGED_ATTENTION_PROFILE=1` (do NOT add `_DETAIL` for real latency).
- PIC server graph op profile (ratio only, not formal latency):
  `MNN_PIC_GRAPH_PROFILE=1  MNN_PIC_GRAPH_PROFILE_TOP=1000`.
- Debug trace: `MNN_PIC_DECODE_DEBUG=1` (already wired in `forwardRaw` /
  `decode`).

## Localization status after 2026-07-02-03

Closed for root-cause localization:

1. x0 request/attention profile exists and confirms `has_pic_budget=1`, q=1
   identity PagedAttention, no `decode_repair_causal_attention` hits, attention
   bounded below full-token time, and batch=1 Conv `use_fp_weight=0`.
2. x0 graph attribution exists and confirms graph/Raster/elementwise/layout
   dominates over `PicScoreAttention`/`PicSparseAttention` alone.
3. Graph dump plus exporter source proves why the PIC-only Cast/Gather/Shape
   machinery exists: it is the sparse-boundary `active_indices` path that remains
   present even when x0 makes it an identity selection.
4. The stale aux normal-decode route is disproven in current source and must not
   be restored.

Remaining before a formal performance report or optimization patch:

1. Produce same-caliber normal-vs-PIC attribution if a final numeric ratio is
   required. Current normal evidence is `llm_bench --profile`; current PIC
   evidence is server/generate plus graph-profile.
2. Prototype a decode-specialized graph that shares the existing weight file and
   PagedCache state but removes score-layer `active_indices` topology from x0
   decode. Runtime identity noops inside `GatherV2` are not the target because
   they keep the unwanted ops in the scheduled graph.
3. (x>0, deferred) capture x=1/3/5/7 device profile to confirm Adreno dense Conv
   family and decode-repair attention route. Do not mix that with the x0
   conclusion.

## Dualgraph decode-only small-op attribution (2026-07-02 post hd128 dispatch)

This section supersedes the earlier OrangePi conclusion that blamed a missing
Mali hd128 ordinary-decode kernel. That was a real intermediate bug, but it is
not the current remaining root cause after the OrangePi artifact was rebuilt
from the source that ungates `ordinaryDecodeHD128Identity`.

Current confirmed facts:

- Dualgraph route is active: decode logs show `PIC decode graph route
  seq_len=1 ... inputs=4`. The decode graph no longer receives
  `pic_recompute_budget`, and its flatbuffer structure is `PagedAttention=24`,
  `PicScoreAttention=0`, `PicSparseAttention=0`.
- The old `active_indices -> Cast/Gather/Shape` sparse-boundary chain is gone
  from x0 decode. Do not continue to attribute current dualgraph x0 latency to
  that chain.
- OrangePi/Mali now dispatches
  `decode_causal_attention_hd128_identity_fused_kv` for x0 decode. The profile
  log `/mnt/ssd/code/.cache/mnn_opencl_pic/logs/dualgraph_decode_profile_orangepi_hd128_dispatch_20260702.log`
  shows `identity_slot=1 fused=1` and per-layer decode attention mostly around
  0.5-0.7 ms after warm-up. PagedAttention profile totals are about
  `calls=164 total_ms=92.193 mean_us=562`.
- End-to-end is still slow on OrangePi: PIC dualgraph `pic_llm_bench` reports
  `decode=9.05 tok/s`, while normal OpenCL reports `decode=22.70 tok/s`.
  So the generic PagedAttention fallback was fixed, but normal parity did not
  return.

Current graph-profile attribution says the remaining gap is mostly non-attention
graph/runtime work:

```text
OrangePi dualgraph, hd128 dispatch, graph profile:
  Convolution   total_ms=1654.993  calls=1352
  Raster        total_ms=787.249   calls=3126
  BinaryOp      total_ms=249.625   calls=1016
  While         total_ms=185.747   calls=792
  UnaryOp       total_ms=136.818   calls=592
  PagedAttention total_ms=114.790  calls=169
  LayerNorm     total_ms=102.825   calls=392

Rhino/Adreno dualgraph graph profile:
  Raster        total_ms=950.679
  Convolution   total_ms=563.572
  BinaryOp      total_ms=294.303
  While         total_ms=232.212
  UnaryOp       total_ms=166.619
  LayerNorm     total_ms=116.861
  PagedAttention total_ms=67.871
```

These graph-profile numbers use debug callbacks / synchronization and are
attribution, not formal latency. The ordering is still useful: once hd128
PagedAttention is fixed, `PagedAttention` is no longer the largest type. The
hot types are the base transformer decode graph plus OpenCL layout/runtime
overhead.

Answer to "are these small ops absent from normal LLM?":

- The `active_indices` family was PIC sparse-boundary-only. Normal LLM does not
  have `PicScoreAttention -> active_indices -> gather residual/mask/rotary`, and
  the new `llm_decode.mnn` x0 graph also should not have it.
- `Raster`, `Convolution`, `BinaryOp`, `While`, `UnaryOp`, and `LayerNorm` are
  not PIC-only. Normal LLM decode also has them because a one-token decode still
  runs the full transformer block: QKV projection, attention output projection,
  MLP gate/up/down, residual adds, RMSNorm/LayerNorm, activation, shape/mask and
  OpenCL layout conversions. OrangePi normal `llm_bench --profile` shows the
  same families: `Convolution 50.63%`, `Raster 25.38%`, `BinaryOp 7.53%`,
  `While 5.64%`, `UnaryOp 4.40%`, `LayerNorm 3.28%`.
- What is different now is not the existence of these ops, but their count,
  layout barriers, backend selection, and runtime overhead under PIC
  PagedAttention/PagedCache and `pic_server`/`StaticModule` execution.

Sketch of the three graphs:

```text
Normal LLM decode graph (non-PIC):

  token hidden
    |
    v
  [layer i]
    RMSNorm / shape plumbing                 -> LayerNorm, While/BinaryOp
    QKV Linear                               -> Convolution
    reshape / transpose / backend layout     -> Raster
    standard Attention                       -> Attention
    o_proj Linear                            -> Convolution
    residual add                             -> BinaryOp
    RMSNorm                                  -> LayerNorm
    gate/up/down MLP + activation            -> Convolution + UnaryOp/BinaryOp
    residual add                             -> BinaryOp
    |
    v
  lm_head Linear                             -> Convolution
```

```text
Old PIC sparse-boundary graph used even for x0 before dualgraph:

  full/prefill path
    |
    v
  layer 1 PicScoreAttention
      outputs:
        attention_output
        active_indices  <---- dynamic runtime tensor
             |
             v
        Cast / Shape / Rank / Binary / While
             |
             v
        GatherV2 residual, norm hidden, rotary, mask, gate input
             |
             v
  layer 2..23 PicSparseAttention

  For x0, active_indices == [current row], but the graph still scheduled this
  dynamic-index branch. Runtime identity folding inside Gather would not remove
  the scheduling/shape/layout cost because the ops still exist in the graph.
```

```text
Current PIC dualgraph x0 decode graph:

  token hidden
    |
    v
  [layer i]
    RMSNorm / shape plumbing                 -> LayerNorm, While/BinaryOp
    QKV Linear                               -> Convolution
    reshape / transpose / backend layout     -> Raster
    PagedAttention(q=1)
      - writes current K/V into PagedCache
      - reads historical K/V through slot table
      - hd128 identity fused kernel on OpenCL
    o_proj Linear                            -> Convolution
    residual add                             -> BinaryOp
    MLP / activation / residual              -> Convolution + UnaryOp/BinaryOp
    |
    v
  lm_head Linear                             -> Convolution

  Removed:
    pic_recompute_budget input
    active_indices output
    PicScoreAttention/PicSparseAttention decode nodes
    active_indices-derived Cast/Gather/Shape chain

  Still present:
    normal transformer dense ops, elementwise ops, layout Raster,
    dynamic shape/mask plumbing, StaticModule resize/session overhead,
    and PIC-specific PagedAttention/PagedCache layout boundaries.
```

Why "decode-only graph" still has small ops:

1. "Decode-only" means `seq_len=1` autoregressive execution and no sparse
   recompute boundary. It does not mean "only run one attention kernel".
2. Each generated token still passes through all 24 transformer layers and the
   lm_head. That necessarily schedules the dense Linear/MLP, norm, activation,
   residual, shape/mask, and layout conversion ops.
3. Normal LLM and PIC dualgraph both pay this base graph. PIC can pay extra when
   PagedAttention output format or PagedCache slot-table semantics create layout
   transitions that normal `Attention` does not need.
4. Graph-profile itself forces synchronization after ops, so it amplifies
   small-op costs. It is only for ranking. The next latency-grade split should
   use `MNN_PIC_MODULE_PROFILE=1` without graph profile to separate
   `resize_ms` from `execute_ms`.

Next localization steps:

1. Run OrangePi and Rhino with `MNN_PIC_MODULE_PROFILE=1` and no graph profile.
   If `resize_ms` dominates, optimize `StaticModule`/session shape path for
   fixed x0 decode. If `execute_ms` dominates, continue with same-surface normal
   vs PIC graph execution attribution.
2. Compare normal and PIC batch=1 `Convolution` / `Raster` / `While` under the
   same profiling surface. Optimize only if the same op family is measurably
   worse in PIC; do not return to active-indices identity folding as the fix.
3. Check PagedAttention output layout and following `o_proj` input format. A
   per-layer `Raster` between PagedAttention and Linear is a plausible PIC-only
   tax even after the active-indices chain is gone.

## Related memory

- `rhino-x0-decode-slow-rootcause` — same conclusion, condensed; tracks that
  the aux module was removed and the OrangePi residual is the missing
  head_dim=128 Mali ordinary-decode kernel.
- `v15-smallm-weight-only-kernel` — Jetson CUDA decode-repair x=3 cliff (int4
  dense route), for cross-backend comparison.
