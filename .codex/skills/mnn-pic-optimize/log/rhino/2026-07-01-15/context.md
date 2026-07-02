# Rhino Decode x0 PagedCache-Native Context

## Objective

Locate why MiniCPM5-1B PIC/PagedAttention x=0 no-repair decode on Rhino Pi-X1 / Adreno OpenCL is much slower than normal LLM decode, without using a transposed-K mirror as the default OpenCL solution.

The target path is:

- `execution_mode = native-full-reuse`
- `decode_refine.enabled = false`
- `decode_refine_runtime = None`
- `mLlm->decode(maxTokens)`
- current request PagedCache, identity slot table, `q=1`, `headDim=128`

## Source Changes

Files touched in this slice:

- `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
  - Removed the env-triggered Adreno identity route to `runDecodeNormalStyleAttentionHD128Identity`.
  - The Adreno identity x0 path now stays on `runDecodeCausalAttentionHD128Identity`.
- `source/backend/opencl/execution/cl/attention_buf.cl`
  - Reworked `decode_causal_attention_hd128_identity_row32/row64`.
  - Old style: each lane owned a subset of K positions and accumulated all 128 output dims as `COMPUTE_FLOAT8 o[16]`, then reduced full output vectors through local memory.
  - New style: tile online softmax, one score per lane, local max/sum per tile, and private `COMPUTE_FLOAT4` V accumulation per output lane.
  - Reads current PagedCache K/V buffers directly. No K transpose mirror and no disk/hydrate in decode.
- Regenerated OpenCL source map:
  - `source/backend/opencl/execution/cl/attention_buf_mnn_cl.cpp`
  - `source/backend/opencl/execution/cl/opencl_source_map.hpp`

`git diff --check` passed for these files.

## Build And Sync

Build command:

```bash
env MNN_TARGET_DEVICE=aidlux_adreno_opencl \
  BUILD_TARGET=pic_server \
  BUILD_MNNCONVERT=0 \
  INSTALL_AFTER_BUILD=1 \
  JOBS=96 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

The build script detected empty compiler/sysroot cache fields and removed the stale Rhino build dir before configuring. Output artifacts were checked as AArch64 ELF files and synced with:

```bash
rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Server paths:

- current non-profile server: `http://192.168.101.227:18133`
- model config: `/mnt/nvme/mnn_pic_opencl/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp/config_opencl_greedy.json`
- runtime cache: `/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/runtime_cache/opencl`
- shared KV: `/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/shared_kv/OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp`

## TPOT Results

Formal-ish warmed x0, profile off:

| path | ctx512 ms/token | ctx1024 ms/token |
| --- | ---: | ---: |
| prior P0 fused identity | 64.373 | 105.873 |
| transposed-K mirror A/B | 100.110 | 131.486 |
| tile online-softmax | 58.798 | 62.712 |
| tile + private `float4/vload4` | 58.183 | 62.084 |
| force row32 A/B | 59.508 | 64.445 |
| normal LLM x0 | 23.436 | 24.920 |

Current ratio:

- ctx512: `58.183 / 23.436 = 2.48x`
- ctx1024: `62.084 / 24.920 = 2.49x`

Row32 A/B was worse than the current row64 heuristic:

- ctx512: `+1.32 ms/token`
- ctx1024: `+2.36 ms/token`

Keep the current lane heuristic: row64 for average causal KV work `>=512`, row32 below that.

## Attention Profile

PagedAttention detail profile after the kernel rewrite:

- op: `decode_causal_attention_hd128_identity`
- `query=1`
- `input_query=1`
- `full_q=0`
- `identity_slot=1`
- `lane=64` at ctx512/1024
- most layers around `0.49-0.56 ms/layer` in warmed profile; first few layers can be higher.

Attention total is roughly `13-15 ms/token`. This is still slower than normal attention, but it is not enough to explain the remaining `~34-37 ms/token` end-to-end gap alone.

## Normal LLM Reference

Command shape:

```bash
artifacts/aidlux_adreno_opencl/bin/llm_bench \
  -m /mnt/nvme/mnn_pic_opencl/models/normal/OpenBMB__MiniCPM5-1B/config_opencl_greedy.json \
  -a opencl -c 2 -p <ctx> -n 32 -rep 3 -kv true -load false \
  -j logs/decode_x0_compare_20260701/normal_ctx<ctx>.json
```

Results:

- ctx512: `42.6709 tok/s` = `23.44 ms/token`
- ctx1024: `40.1293 tok/s` = `24.92 ms/token`

Normal `llm_bench --profile` only provides type-level attribution, not named graph ops. For ctx512/profile:

| type | percent |
| --- | ---: |
| Convolution | 40.66% |
| Raster | 31.27% |
| BinaryOp | 8.50% |
| Attention | 6.32% |
| While | 5.90% |
| UnaryOp | 4.35% |
| LayerNorm | 2.95% |

This confirms normal decode is also dense/raster heavy, but its attention type is only a small fraction.

## PIC Graph Profile

Profile server env:

```bash
MNN_PAGED_ATTENTION_PROFILE=1
MNN_PAGED_ATTENTION_PROFILE_DETAIL=1
MNN_PIC_GRAPH_PROFILE=1
MNN_PIC_GRAPH_PROFILE_TOP=1000
```

Short profile request:

- ctx512
- `max_tokens=4`
- x0/no-repair
- profile only, not TPOT

Summary:

```text
MNN_PIC_GRAPH_PROFILE_SUMMARY label=mode=full-reuse,ratio=0,score_layer=1,max_tokens=4 total_ms=1342.195 calls=4280 unique_ops=1040 unique_types=11
```

Type totals:

| type | total_ms | calls | share |
| --- | ---: | ---: | ---: |
| Raster | 457.499 | 1610 | 34.1% |
| BinaryOp | 192.406 | 725 | 14.3% |
| While | 163.656 | 600 | 12.2% |
| Convolution | 159.833 | 485 | 11.9% |
| Extra | 94.782 | 120 | 7.1% |
| PicSparseAttention | 93.566 | 110 | 7.0% |
| LayerNorm | 67.203 | 245 | 5.0% |
| UnaryOp | 65.768 | 250 | 4.9% |
| Cast | 33.222 | 125 | 2.5% |
| PicScoreAttention | 8.749 | 5 | 0.7% |
| PagedAttention | 5.511 | 5 | 0.4% |

Top named ops:

- `/lm/lm_head/Linear`: `13.324 ms`, calls=5, max=`2.676 ms`
- `PicAdrenoTinyMlpWeightOnly/fused`: roughly `3.6-4.4 ms` per layer across 5 calls for many layers.
- PagedAttention family still appears as:
  - layer 0: `PagedAttention`, calls=5
  - layer 1: `PicScoreAttention`, calls=5
  - layers 2-23: `PicSparseAttention`, calls=110

Important interpretation:

- Backend attention dispatch is correct for x0: `decode_causal_attention_hd128_identity`, q=1, identity slot, no sparse repair.
- Graph-level op types are still PIC graph-boundary types even in x0/no-repair decode.
- The big profile mass is now graph/runtime overhead and dense/raster work, not disk, hydrate, or PagedCache discontinuity.

## Root Cause Statement

The first-order bug was the old Adreno identity x0 route experimenting with normal-style transposed K. That path is removed from default and was slower.

The remaining `~2.5x` gap vs normal LLM is not explained by PagedCache storage continuity or disk access. Decode does not load KV from disk. The current PagedCache-native attention kernel is still a cost, but profile shows the larger gap is that no-repair x0 decode is still executed through the PIC graph-boundary graph shape:

- `PicScoreAttention` / `PicSparseAttention` remain in the graph.
- Extra `active_indices` / budget / gather-related shape machinery remains alive.
- Many tiny `Raster`, `While`, `BinaryOp`, `Cast`, rotary reshape/gather ops are launched per token.
- MLP/lm_head/QKV/O projection still dominate substantial time.

So the next target is a no-repair decode runtime/graph route that preserves PagedCache semantics but removes PIC sparse boundary mechanics for x0 decode.

## Next Patch Direction

P0:

- Add an explicit `no_repair_decode` graph/runtime mode for full-reuse decode.
- In this mode, score-layer and sparse-layer PagedAttention should execute as ordinary single-output PagedAttention for decode:
  - no `pic_recompute_budget` dependency,
  - no active-indices output,
  - no gather/crop/shape side path,
  - still read/write the current request PagedCache by logical position.
- Keep backend attention on PagedCache-native hd128 q=1 identity decode. Do not add transposed-K mirror as the default.

P1:

- Reduce launch count around decode rotary and reshape:
  - fold q/k rotary small ops if possible,
  - eliminate repeated scalar `Cast/BinaryOp/While` shape ops in decode,
  - specialize q=1 decode graph shapes to avoid full graph-boundary bookkeeping.

P2:

- Only after P0/P1, A/B V-cache image reads for the identity decode kernel. K image or K transpose mirror is not the current priority because CUDA does not need that mirror and the current bottleneck is larger outside attention.

## Validation Required After P0

Run:

- normal LLM x0 ctx512/1024
- generic PIC x0 ctx512/1024
- optimized PIC x0 ctx512/1024
- optimized x=1/3/5/7 smoke

Acceptance target:

- PIC x0 should move toward normal x0 same order of magnitude.
- Attention profile should stay on `decode_causal_attention_hd128_identity`.
- Graph profile should no longer show x0 decode dominated by `PicScoreAttention`, `PicSparseAttention`, and active-index gather machinery.
