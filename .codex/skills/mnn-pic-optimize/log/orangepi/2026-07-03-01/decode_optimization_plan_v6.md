# Decode Optimization Plan V6 — Unified Mali/Adreno OpenCL Decode

Created: 2026-07-03
Scope: MNN PIC/PagedAttention OpenCL decode, OrangePi/Mali + Rhino/Adreno.
Predecessors: V2–V5 in `context.md`; handoff in `claude_handoff_decode.md`.

## 0. Goal

For `llama3.2-3b`, `minicpm5-1b`, `qwen3-4b` on OrangePi/Mali and Rhino/Adreno:

```text
q = 1/2/4/6/8 new DecodeK TPOT <= old identity TPOT, every cell,
without routing any cell back to old by model / device / GPU / head / q.
```

Old identity PagedCache attention is a baseline / A-B reference only
(`MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=0`). It is never a production fallback.

## 1. Unified Design Philosophy (the one rule)

One transposed-K decode kernel family. Mali and Adreno differ only by
**parameter axes**, never by routing:

```text
lane        in {32, 64, 128}
q_tile      in {2, 4, 8}            (q=6 -> q8 + valid-row mask)
k_tile      in {16, 32, 64}         (new; decoupled from lane)
gqa_shared  in {off, on}            (share K/V stream across GQA group)
k_format    in {buffer, image}      (image = Adreno-first)
v_format    in {buffer, image}      (image = Adreno-first; identity-slot fast path otherwise)
```

Selection is by `device_family + head_dim + kv_len_bucket + q + num_heads/kv_heads`,
expressed as **env variants + tune cache** (key namespaced `mali` / `adreno`).
No `if (model == ...)`, no `if (device == ...) -> old`.

Allowed convergence mechanisms (from SKILL `Decode 算子优化约束`):

- optimize the new kernel implementation;
- tune parameters (lane, q_tile, k_tile, local memory layout, buffer/image);
- add explicit benchmark variants with clear env labels.

## 2. Current Kernel Family Inventory (baseline)

`source/backend/opencl/execution/cl/paged_decode_attention_buf.cl`, all HD128:

| family | use | K source | V source | shares K |
|---|---|---|---|---|
| `identity` / `identity_fused_kv` | old baseline, q=1 old | paged key `[slot,B,H_kv,D]` | paged value | no |
| `identity_fused_kv_gqa` | old q=1 GQA-shared | paged key | paged value | yes (per kv_head) |
| `transposed_k_fused_kv` | q=1 new | decodeKey `[B,H_kv,D,slots]` | paged value | no |
| `transposed_k_readonly` | q=1 micro (append split) | decodeKey | paged value | no |
| `transposed_k_sparse` (qtile q2/q4/q8) | q>1 decode-repair | decodeKey | paged value | yes (Q_TILE rows) |

State of the art:

- q>1 qtile (`DEFINE_DECODE_CAUSAL_TRANSPOSED_K_QTILE_HD128`, line 973) is already
  a true shared-K tile: one workgroup processes Q_TILE rows, loads each K tile once.
- q=1 transposed (`DEFINE_DECODE_CAUSAL_TRANSPOSED_K_FUSED_KV_HD128`, line 560) is
  still row-softmax: every query head re-reads the full decodeKey K stream. This is
  the Qwen-q1 / MiniCPM-q1 regression root cause (matches handoff diagnosis).
- V is read from paged value for every family. `identity_slot` fast path exists in
  qtile (`PagedAttentionBufExecution.cpp:7140`) but not yet in q=1 families.
- decodeKey/value/key are all `__global FLOAT *` buffer. Image format exists only
  for prefill sparse flash (`mqtile_*_kimg` / `_kvimg`, `PagedAttentionBufExecution.cpp:235`),
  Adreno only. Decode has no image variant. This is the largest untapped axis.

GQA group sizes (num_heads / kv_heads):

```text
Llama3.2-3B   24 / 8  -> group = 3
Qwen3-4B      32 / 8  -> group = 4
MiniCPM5-1B   16 / 2  -> group = 8
```

## 3. Hard Constraints (must hold throughout)

- P0 prepare boundary: historical K transpose runs only in
  `prepareDecodePrefix` (`llm.cpp:1625` -> `PagedAttentionBufExecution.cpp:3959`).
  Timed decode may only append the current token K/V/decodeKey and run
  attention/rank. Any profile with `decode_prepare_inside_decode=1` is invalid.
- Image mirror allocation and fill also happen in prepare, not in decode.
  The image is a derived view of PagedCache (same status as decodeKey), not a
  second KV owner. append writes buffer + image together.
- PagedCache is the only runtime KV owner; decodeKey/image are decode-only views.
- No routing back to old by model/device/head/q. Old is A/B only.
- Tune keys must carry `mali` / `adreno` namespace (`_openCLTuneDeviceKey`,
  line 383) so the two device families do not replay each other's selections.
- Formal matrix: OrangePi + Rhino x 3 models x q=1/2/4/6/8, profile off.

## 4. q=1 Direction

### 4.1 GQA-cooperative transposed-K q=1 (highest priority)

Port the GQA-shared structure of `identity_fused_kv_gqa` (line 1193) onto the
transposed-K data path. One workgroup handles one kv_head's whole GQA group
(`NUMHEAD_GROUP_SIZE` query heads), shares the decodeKey K stream, keeps
independent softmax per query head.

```text
new kernel family:
  decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row{32,64,128}

K read change (vs identity_fused_kv_gqa):
  old: key_cache[((k*batch+b)*kv_head_num+kvh)*128 + d]    // paged key, slot-major
  new: decode_key[decode_key_base + d*key_max_len + k]      // transposed, D-major per slot

decode_key_base = ((b * kv_head_num + kvh) * 128) * key_max_len   // same as fused_kv
```

Why this is the right first step:

- directly removes the "each query head re-reads full K stream" cost;
- Llama group=3, Qwen group=4, MiniCPM group=8 -> K reads drop by 1/group;
- V tile is also shared across the GQA group (current row kernel re-reads V per head);
- no new routing — just a new kernel variant selected by lane/group autotune;
- does NOT revive the reverted "current-token direct-read" branch (handoff forbids it).

Guard: `NUMHEAD_GROUP_SIZE <= 1` falls back to the existing `transposed_k_fused_kv`
(the macro already early-returns when `NUMHEAD_GROUP_SIZE <= 1`, see line 1216).

### 4.2 identity-slot V fast path for all q=1 families

`_slotTableIsIdentity` (line 1518) is already used by qtile. Extend it to
`transposed_k_fused_kv` / `_readonly` / the new `_gqa` family:

```text
add __private const int identity_slot arg
q_slot = identity_slot ? q_logical : slot_table[q_logical]
value_slot (V read) = identity_slot ? value_logical : slot_table[value_logical]
skip the slot_table load + bounds branch when identity_slot=1
```

full-reuse requests are almost always identity, so this is a stable small win
across both devices and does not touch routing.

### 4.3 K-vectorized q=1 (kernel-internal, lower priority)

decodeKey layout `[B,H_kv,D,slots]` makes D contiguous per slot. Current K read
in `transposed_k_fused_kv` (line 642) steps `d4` by 4 but each step crosses a
slot (`d*key_max_len`), so the D-contiguous advantage is not fully used.

Variant: load a contiguous D segment with `vload4`/`vload8` along D, then stride
across slots. This is purely inside the K-load loop; no routing, no new API.
Defer until 4.1 is measured.

## 5. q>1 Direction

### 5.1 qtile local-memory pressure and V-side serialization

Current qtile (`DEFINE_DECODE_CAUSAL_TRANSPOSED_K_QTILE_HD128`, line 973) local
memory:

```text
local_q     Q_TILE * 128
score_tile  Q_TILE * LANES
reduce      Q_TILE * LANES
```

At q8 + lane128, `score_tile`/`reduce` alone are 8*128 = 1024 floats each. V-side
accumulation runs only on `lid < 32` (line 1140), so lanes 32..127 idle during the
V stage. This explains the handoff's "MiniCPM q4 regresses, lane-sensitive"
result on Mali.

Variants (all parameter-axis, no routing):

```text
V-side parallel: split the 128-D output across all lanes (not just lid<32).
  e.g. out_d4 = lid << 2 covers 32 lanes for 128 D;
       let lid in [32,64) handle a second Q_TILE row's V, etc.

K_TILE decoupled from LANES:
  today K tile size == LANES. Introduce an independent K_TILE in {16,32,64}
  so K load granularity and softmax-reduce width are independent.
  This is the prefill qtile pattern; decode can reuse it.

local-memory layout: when lane128 + q8 over-occupies, prefer lane64 + q8 or
  lane128 + q4 by tune, not by hardcoded model rule.
```

### 5.2 qtile x GQA shared

Mirror 4.1 for q>1: change gws z from `batch*head_num` to `batch*kv_head_num`
and process the GQA group inside the workgroup, sharing the K tile across both
Q_TILE rows and GQA heads. Largest benefit at long kv_len.

### 5.3 q=6

Already handled by q8 kernel + `q_valid[]`/`active_len[]` mask (lines 1013-1032).
Keep; do not add a separate q6 kernel.

## 6. Buffer vs Image Cache Format (the new axis)

### 6.1 Why

- Adreno texture L1 bandwidth is materially higher than buffer L1 (rhino logs and
  `references/rhino.md` confirm image-backed weight wins on Adreno). decode K/V
  read is the attention hot path, so image directly buys bandwidth.
- decodeKey `[B,H_kv,D,slots]` maps naturally to image2d: `(x,y) = (D, slot)`,
  D contiguous = x contiguous, one `read_imagef` fetches 4 fp16. This also fixes
  the 4.3 K-stride problem for free.
- Mali: image and buffer share the same cache on Astec, and fp16 image format
  support is more restricted. So image is an **Adreno-first** variant; Mali stays
  buffer + GQA-shared + identity-slot V. This is a variant choice, not a route
  to old.

### 6.2 Implementation (no second KV owner)

```text
SharedPagedCache gains optional image mirrors (Adreno only):
  decodeKeyImage   image2d_t, (width=D, height=slots) per (b,kvh)
  valueImage       image2d_t, (width=D, height=slots) per (b,kvh)

allocation + first fill: in prepareDecodePrefix, only when
  runtime->getGpuType() == ADRENO && _supportsAdrenoSparseFlashKImage(...)
  (reuse the existing prefill predicates, PagedAttentionBufExecution.cpp:235+).

append (q=1 fused / q>1 sparse append):
  write buffer (decodeKey + value) as today
  if image enabled: also write image (append kernel gains an image arg, or a
  tiny buffer->image copy kernel runs right after append).
  This stays in the append stage, not in historical-K prepare.

new kernel variants (Adreno):
  decode_causal_attention_hd128_transposed_k_fused_kv_gqa_kimg_row{64,128}
  decode_causal_attention_hd128_transposed_k_qtile_q4_kimg_row64
  decode_causal_attention_hd128_transposed_k_qtile_q4_kvimg_row64
  ... (kimg = K image, V buffer; kvimg = K+V image; match prefill naming)
```

Hard rule: the image is a derived view of PagedCache, allocated with it, filled
in prepare/append. It is never a second runtime KV cache and never a staging
buffer (SKILL `OpenCL PagedCache 零拷贝边界`).

### 6.3 V-side image

`_kvimg` variants read value from `valueImage`. Pair with identity-slot fast path:
when `identity_slot=1` and image enabled, V read is a single `read_imagef` at
`(D, logical)`. This is the Adreno equivalent of the 4.2 buffer fast path.

## 7. Unified Parameter / Tune Axes

All axes are env variants + tune cache, never route if-else:

| axis | candidates | default lean |
|---|---|---|
| lane | 32/64/128 | Mali q>1 -> 64; Adreno -> 64 (buffer) / 128 (image) |
| q_tile | 2/4/8 | q2->2, q4->4, q6->8(mask), q8->8 |
| k_tile | 16/32/64 (new) | long kv_len -> larger k_tile |
| gqa_shared | off/on | q=1 group>1 -> on; q>1 optional |
| k_format | buffer/image | Adreno image; Mali buffer |
| v_format | buffer/image | Adreno kvimg; identity-slot -> buffer direct |
| v_side_parallel | lid<32 / full lane | q_tile>=4 -> full lane |

New env variant names (additive, all default off):

```text
MNN_PAGED_ATTENTION_DECODE_K_TILE          = 16|32|64
MNN_PAGED_ATTENTION_DECODE_GQA_SHARED      = 1
MNN_PAGED_ATTENTION_DECODE_KIMG            = 1     # K image (Adreno)
MNN_PAGED_ATTENTION_DECODE_VIMG            = 1     # V image (Adreno)
MNN_PAGED_ATTENTION_DECODE_V_SIDE_PARALLEL = 1
```

Existing env kept: `MNN_PAGED_ATTENTION_DECODE_HD128_FORCE_LANE`,
`MNN_PAGED_ATTENTION_DECODE_QTILE_FORCE`, `MNN_PAGED_ATTENTION_PROFILE[_DETAIL]`,
`MNN_PIC_DECODE_DEBUG`.

## 8. Implementation Order

Following V5 "attention-only must win first, then overhead":

```text
P1 q=1 GQA-cooperative transposed-K kernel (4.1)
   - new macro decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row{32,64,128}
   - host: build + dispatch when NUMHEAD_GROUP_SIZE > 1 and gqa_shared on
   - measure attention-only via _decodeTransposedKAttentionOnlyBenchEnabled path
   - target: Qwen q1, MiniCPM q1 stop regressing vs old

P2 identity-slot V fast path on all q=1 families (4.2)
   - add identity_slot arg to fused_kv / readonly / new gqa macros
   - host: pass _slotTableIsIdentity(mMeta, kvLen) (already computed for qtile)
   - stable small win, full-reuse identity case

P3 Adreno decode image variants (6.2)
   - SharedPagedCache decodeKeyImage/valueImage (Adreno only, prepared in prepareDecodePrefix)
   - append writes image alongside buffer
   - new _kimg / _kvimg kernel variants, reuse _supportsAdrenoSparseFlashKImage predicates
   - target: Adreno decode K/V bandwidth, biggest Adreno-specific win

P4 q>1 qtile K_TILE decoupling + V-side parallel (5.1)
   - independent K_TILE, full-lane V accumulation
   - target: MiniCPM q4 regression, local-mem occupancy

P5 q>1 qtile x GQA shared (5.2)
   - gws z = batch*kv_head_num, share K tile across GQA group
   - target: long kv_len duplicate-K removal

P6 autotune key landing
   - write all axes into paged_* tune keys, namespaced mali/adreno
   - warm -> /v1/tune/update_cache -> formal measure with profile off
```

## 9. Validation Ladder

```text
Step 1  correctness + profile (profile on, profile_detail on):
        - new op= lines appear (transposed_k_fused_kv_gqa, *_kimg, *_kvimg, qtile_k_tile)
        - decode_prepare_inside_decode=0 for every layer
        - no ERROR / target unavailable / async read failed

Step 2  attention-only A/B (profile_detail, record queue off for attribution):
        - old_identity_attention_only vs new_attention_only per q/device/model
        - promotion rule Step 1: new_attention_only must beat old_attention_only

Step 3  if Step 2 passes: restore record queue + append, measure formal TPOT
        if Step 2 fails: tune lane/k_tile/local-mem/image before routing

Step 4  formal TPOT matrix (profile off):
        devices: OrangePi, Rhino
        models : llama3.2-3b, minicpm5-1b, qwen3-4b
        q      : 1,2,4,6,8   (repair_tokens 0,1,3,5,7)
        ctx    : 512 first, then 1024/1536/2048/2560
        pass   : new median TPOT <= old median TPOT, every cell
        no cell may pass only because q=1 silently fell back to old

Step 5  extend contexts 1024/1536/2048/2560 once ctx512 is clean
```

Promotion rule (from V4):

```text
Step 1: new_attention_only beats old_attention_only for same q/device/model
Step 2: if Step 1 passes, remove fixed overhead: prepare, append, rank, launch/record
Step 3: if Step 1 fails, tune kernel internals first: lane, k_tile, local mem, image
Step 4: promote a default route only when formal TPOT is non-regressing for all
        llama3.2-3b, minicpm5-1b, qwen3-4b x OrangePi/Rhino x q=1/2/4/6/8
```

## 10. File Touchpoints

```text
source/backend/opencl/execution/cl/paged_decode_attention_buf.cl
  - new macro DEFINE_DECODE_CAUSAL_TRANSPOSED_K_FUSED_KV_GQA_HD128 (port line 1193 to decodeKey)
  - add identity_slot arg to fused_kv (560) / readonly (707) macros
  - new _kimg / _kvimg variants for fused_gqa and qtile
  - qtile: independent K_TILE, full-lane V accumulation

source/backend/opencl/execution/cl/paged_decode_attention_buf_mnn_cl.cpp
  - register new kernel names

source/backend/opencl/execution/cl/opencl_source_map.hpp
  - add new sources if split

source/backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp
  - SharedPagedCache: optional decodeKeyImage / valueImage (Adreno)
  - new kernel members, new env readers

source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
  - _decodeHD128QTile / new _decodeHD128KTile / _decodeGqaShared / _decodeKImg / _decodeVImg readers
  - ensureDecodeTransposedKKernel: build new variants with -DNUMHEAD_GROUP_SIZE
  - runDecodeCausalAttentionHD128TransposedKFusedKV: dispatch gqa variant when group>1
  - runDecodeCausalAttentionHD128TransposedKSparse: k_tile + v_side_parallel + gqa-shared gws
  - prepareDecodePrefix: allocate + fill image mirrors (Adreno only)
  - append kernels: write image alongside buffer when image enabled
  - onExecute dispatch (line ~8440-8780): keep single decision tree, no model/device->old

source/core/PagedKVMeta.hpp
  - no structural change expected; reuse pic_decode_repair_tokens_per_step,
    sparse_query_logical_indices, slot_table_host, request_base

transformers/pic_llm/engine/src/llm.cpp
  - preparePagedDecode (1625): ensure image mirror prepared per layer (no decode-time work)
```

## 11. What This Plan Does NOT Do

- Does not route any cell to old by model/device/head/q.
- Does not revive the reverted q=1 current-token direct-read branch.
- Does not introduce a second runtime KV cache; image/decodeKey are derived views.
- Does not move historical K prepare (or image fill) into timed decode.
- Does not add a request-time online tuner; uses MNN tune cache + warm.
- Does not change PIC score-layer / graph-boundary semantics; decode-only.

## 12. One-line Summary

```text
One transposed-K decode kernel family (q=1 fused-GQA / q>1 qtile), tuned across
five axes (lane, q_tile, k_tile, gqa_shared, buffer-vs-image) by device_family.
Mali: buffer + GQA K-share + identity-slot V.
Adreno: image-backed decodeKey/value + large lane.
No model/device routing; old is A/B only.
First two steps: q=1 GQA-cooperative transposed kernel, then Adreno image variants.
```
