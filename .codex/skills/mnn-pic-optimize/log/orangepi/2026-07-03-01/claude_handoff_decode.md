# OpenCL PIC Decode Transposed-K Handoff

Created: 2026-07-03
Scope: MNN PIC/PagedAttention OpenCL decode optimization, primarily OrangePi/Mali now, Rhino/Adreno next.

## Background

The problem is slow endpoint TPOT in PIC LLM decode for:

- `llama3.2-3b`
- `minicpm5-1b`
- `qwen3-4b`

The decode repair benchmark uses `repair_tokens = 0/1/3/5/7`, which maps to effective decode attention `q = 1/2/4/6/8`.

The new implementation idea is `decode-transposed K`: after prefill/full-reuse, historical K in PagedCache is explicitly transposed into a decode-friendly K view. Decode then only appends the new token K into both normal PagedCache and transposed decodeKey, and attention reads historical K contiguously by head_dim/k.

The key distinction is:

- Prefill can use `Q tile x K tile` because Q is large.
- Decode cannot blindly reuse prefill strategy because Q is tiny (`q=1/2/4/6/8`) and launch/fixed overhead, V-side reads, local memory, and lane width dominate.

PagedCache itself is not the conceptual problem. The old and new paths both still scan all causal K for exact attention. The difference is the memory view used to read K:

- old identity path reads K from normal paged layout;
- new transposed-K path reads historical K from `decodeKey` layout;
- V still comes from normal paged value cache.

## Hard Constraints

Do not use old-path routing to hide regressions.

Specifically, do not automatically route by:

- model name
- device
- GPU type
- head count / kv head count
- q value

Allowed convergence mechanisms:

- optimize the new kernel implementation;
- tune implementation parameters such as lane, qtile, local memory layout;
- add explicit benchmark variants with clear env labels.

Old identity PagedCache attention is only a baseline/A-B reference, not a production fallback.

Decode prepare is outside timed decode. After prefill/full-reuse, `decode_prepare_transpose_k` must run explicitly. During timed decode, only append of the current token decodeKey is allowed. If profile shows:

```text
decode_prepare_inside_decode=1
```

the implementation is invalid.

PagedCache should preallocate decode-stage KV/decodeKey storage at creation time. Do not allocate temporary KV storage during decode. Decode append should write normal K/V and transposed decodeKey together where appropriate.

## Current Code State

Important files:

- `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
- `source/backend/opencl/execution/cl/paged_decode_attention_buf.cl`
- `source/backend/opencl/execution/cl/paged_decode_attention_buf_mnn_cl.cpp`
- `source/backend/opencl/execution/cl/opencl_source_map.hpp`
- `.codex/skills/mnn-pic-optimize/SKILL.md`

Already done:

- Removed q1 route enum/env auto fallback:
  - no `DecodeTransposedKQ1Route`
  - no `MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_Q1_ROUTE`
  - q1 with `MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1` uses new transposed fused path.
- Removed q>1 model/head/device sparse route fallback:
  - `_decodeTransposedKSparseShapeSupported(runtime, attnLen)` now only gates non-legacy OpenCL and `2 <= attnLen <= 8`.
- Added qtile V-side identity-slot optimization:
  - qtile kernel has `identity_slot`;
  - if identity slot table, V side uses `value_slot = value_logical`;
  - otherwise reads `slot_table[value_logical]`.
- Added optimization constraint section in `.codex/skills/mnn-pic-optimize/SKILL.md`.
- Added current q>1 parameter rule:
  - `Mali && q > 1` defaults to lane64 via `_decodeHD128SparseLaneWidth`;
  - q1 still uses original `_decodeHD128LaneWidth`;
  - `MNN_PAGED_ATTENTION_DECODE_QTILE_FORCE=2|4|8` is available as an explicit benchmark variant.

Important failed experiment:

- q1 fused kernel experiment that only let one query head per kv head write current K/V/decodeKey and read current token directly from input was tested and reverted.
- It made q1 worse:
  - Llama: `218.032 -> 225.018 ms`
  - MiniCPM: `56.498 -> 62.032 ms`
  - Qwen: `263.114 -> 276.412 ms`
- Do not revive that exact current-token direct-read branch without a different kernel design.

Final artifact state:

- Latest local OrangePi build completed after reverting the failed q1 experiment and adding Mali q>1 lane64.
- Latest artifact has been rsynced to:

```text
/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

This final synced artifact has not yet been benchmarked after the last rebuild.

## Key OrangePi Results

Baseline old path run:

```text
.cache/mnn-pic-benchmark/decode_experiment/decode_ab_old_current_orangepi_20260703
```

Unified new before identity-V:

```text
.cache/mnn-pic-benchmark/decode_experiment/decode_unified_new_orangepi_nodebug_20260703
```

Unified new with identity-V:

```text
.cache/mnn-pic-benchmark/decode_experiment/decode_unified_new_identity_v_orangepi_20260703
```

Best q/lane diagnostic summary from explicit variants:

| model | q | old | default new | lane64 | lane32 | best-new | best vs old |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama3.2 3B | 1 | 235.143 | 218.032 | 218.022 | 214.883 | 214.883 | 1.094x |
| Llama3.2 3B | 4 | 373.776 | 394.873 | 364.021 | 377.558 | 364.021 | 1.027x |
| Llama3.2 3B | 6 | 814.644 | 749.300 | 769.690 | 806.496 | 749.300 | 1.087x |
| Llama3.2 3B | 8 | 844.873 | 837.000 | 822.413 | 843.457 | 822.413 | 1.027x |
| MiniCPM5-1B | 1 | 61.675 | 56.498 | 60.691 | 59.856 | 56.498 | 1.092x |
| MiniCPM5-1B | 4 | 240.754 | 260.175 | 255.997 | 248.325 | 248.325 | 0.970x |
| MiniCPM5-1B | 6 | 285.706 | 309.890 | 274.235 | 319.113 | 274.235 | 1.042x |
| MiniCPM5-1B | 8 | 298.453 | 342.810 | 281.499 | 315.621 | 281.499 | 1.060x |
| Qwen3-4B | 1 | 249.352 | 263.114 | 267.361 | 285.579 | 263.114 | 0.948x |
| Qwen3-4B | 4 | 826.331 | 724.360 | 704.953 | 737.997 | 704.953 | 1.172x |
| Qwen3-4B | 6 | 1107.961 | 1051.024 | 1003.290 | 1082.252 | 1003.290 | 1.104x |
| Qwen3-4B | 8 | 1114.840 | 1121.160 | 1065.079 | 1062.963 | 1062.963 | 1.049x |

Interpretation:

- q1 is not solved for Qwen.
- q>1 is strongly sensitive to lane width on Mali.
- lane64 fixes most high-q regressions.
- MiniCPM q4 remains the main q>1 regression.
- A single fixed lane for every shape is not optimal, but Mali q>1 lane64 is a safer current default than old work-per-row lane128.

## q1 Diagnosis

q1 transposed-K attention-only is conceptually strong.

Profile split on Qwen q1:

- identity split:
  - 144 rows
  - QK sum: `1384.703 ms`
  - QKV sum: `165.884 ms`
- transposed split:
  - 108 rows
  - QK sum: `134.699 ms`
  - QKV sum: `93.827 ms`

So transposed K improves QK heavily when measured as split attention-only. Qwen q1 endpoint regression is not because transposed-K reads are bad in principle.

q1 endpoint issue is likely in fixed overhead and fused implementation details:

- append/write cost;
- record queue coverage;
- layers that require rank capture do not use record;
- fused kernel still writes current K/V/decodeKey per query head, not per kv head, but the attempted direct-current-token branch was worse;
- launch/setArg/record and whole decode graph overhead can hide attention-only wins.

Explicit append+readonly q1 variant was worse:

| model | old q1 | fused | append+readonly |
|---|---:|---:|---:|
| Llama3.2 3B | 235.143 | 218.032 | 215.724 |
| MiniCPM5-1B | 61.675 | 56.498 | 72.775 |
| Qwen3-4B | 249.352 | 263.114 | 290.118 |

Do not make append+readonly the default.

## q>1 Diagnosis

q>1 must be true qtile:

- one workgroup/tile should process multiple Q rows;
- K load should be shared across those Q rows;
- do not degenerate into independent q=1 rows.

Current qtile kernel does share the K loop across Q rows, but it has pressure from:

- `local_q[Q_TILE * 128]`;
- `score_tile[Q_TILE * LANES]`;
- `reduce[Q_TILE * LANES]`;
- `out4[Q_TILE]`, `running_m/l[Q_TILE]`;
- V-side loop still runs per tile and per Q.

Lane results imply Mali dislikes qtile + lane128 for several shapes. q>1 lane64 is currently a better default on OrangePi.

Remaining q>1 problem:

- MiniCPM q4 still regresses even with lane32/lane64 variants.
- Next likely axis is qtile structure, not old fallback.

Potential qtile variants to test:

- force q4 to qtile2 (`MNN_PAGED_ATTENTION_DECODE_QTILE_FORCE=2`);
- force q6/q8 to qtile4;
- keep transposed K and same semantics.

These are explicit variants first; only promote a rule if it is shape/device-parameter based and validated.

## Commands

Build OrangePi artifact:

```bash
JOBS=$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 )) \
MNN_TARGET_DEVICE=orangepi5plus \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Sync OrangePi artifact:

```bash
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

Run unified new matrix:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models llama3.2-3b,minicpm5-1b,qwen3-4b \
  --contexts 512 \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --repeats 2 \
  --warm-repeats 1 \
  --skip-normal \
  --run-id <run_id>
```

Run explicit lane variant:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1 MNN_PAGED_ATTENTION_DECODE_HD128_FORCE_LANE=64' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models llama3.2-3b,minicpm5-1b,qwen3-4b \
  --contexts 512 \
  --pic-repair-tokens 0,3,5,7 \
  --pic-max-tokens 16 \
  --repeats 2 \
  --warm-repeats 1 \
  --skip-normal \
  --run-id <run_id>
```

Run explicit qtile variant:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1 MNN_PAGED_ATTENTION_DECODE_QTILE_FORCE=2' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b \
  --contexts 512 \
  --pic-repair-tokens 3 \
  --pic-max-tokens 16 \
  --repeats 2 \
  --warm-repeats 1 \
  --skip-normal \
  --run-id <run_id>
```

Check invalid prepare/errors:

```bash
ssh orangepi@192.168.101.113 \
  'grep -R "decode_prepare_inside_decode=1" -n /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/<run_id>/pic_server_*.log || true; \
   grep -R "ERROR\\|target unavailable\\|async persistent PIC cache read failed" -n /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/<run_id>/pic_server_*.log || true'
```

## Immediate Next Steps

1. Run the final synced artifact once:

```text
run_id = decode_unified_new_mali_lane64_default_orangepi_20260703
env    = MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1
q      = 1/2/4/6/8
models = llama3.2-3b,minicpm5-1b,qwen3-4b
```

Expected:

- q1 should return to pre failed-experiment behavior:
  - Llama and Mini OK;
  - Qwen q1 still regresses.
- q>1 should look close to previous explicit lane64 run where lane64 applies.

2. For MiniCPM q4, run qtile force variants:

- `MNN_PAGED_ATTENTION_DECODE_QTILE_FORCE=2`
- `MNN_PAGED_ATTENTION_DECODE_QTILE_FORCE=4`
- possibly with lane32/lane64.

3. For Qwen q1, do not split append+readonly and do not revive current-token direct branch. Better directions:

- optimize fused q1 kernel without adding per-k branch in hot QK/V loops;
- examine record queue coverage and rank-capture layers;
- profile endpoint with graph/op timing to see non-attention fixed cost;
- consider a dedicated GQA-aware fused q1 kernel where cache write is separated inside the kernel without making hot loop branch on `k == q_logical`.

4. After OrangePi is acceptable, repeat on Rhino/Adreno. Rhino may prefer different lanes; do not hard-code Mali result for Adreno without A/B.

