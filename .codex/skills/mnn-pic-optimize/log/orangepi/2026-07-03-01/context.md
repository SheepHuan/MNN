# OrangePi Decode Transposed-K View For PIC Decode

## Source Requirement

Attachment:

```text
.codex/attachments/14e8bd15-1b21-40ce-8dc8-3c06d039ec4c/pasted-text-1.txt
```

Key requirements extracted:

- Preserve PagedCache as the semantic owner of KV state.
- Add a decode-friendly K view after prefill:
  - PagedCacheK: `[slot][batch][kv_head][dim]`
  - DecodeK: `[batch][kv_head][dim][logical_pos]`
- Use `slot_table` when preparing DecodeK.
- Exclude `decode_prepare_transpose_k` from decode TPOT, but profile it separately.
- During decode append, write both normal PagedCache K/V and DecodeK.
- Cover q=1/2/4/6/8. The target design calls for q1/q2/q4/q8 small-Q kernels and eventually GQA cooperative K-load sharing without serializing query heads.

Follow-up after Step 2': the "Use `slot_table`" line above is the original requirement before
identity-slot analysis. Current PagedCache requests store tokens sequentially and `request_base`
was always zero, so `slot_table[logical] == logical` for all live rows. The implemented removal
keeps PagedCache as the semantic owner and replaces slot lookup with direct `slot = logical`.
Correctness is now guarded by sparse-row validation, not by restoring a logical-to-physical table.

## Implementation

Modified files:

```text
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
source/backend/opencl/execution/cl/paged_decode_attention_buf.cl
source/backend/opencl/execution/cl/paged_decode_attention_buf_mnn_cl.cpp
source/backend/opencl/execution/cl/opencl_source_map.hpp
```

Main code points:

```text
SharedPagedCache::decodeKey
ensureDecodeKeyReady(requiredLen)
runDecodeCausalAttentionHD128TransposedKFusedKV(...)
runDecodeCausalAttentionHD128TransposedKSparse(...)
append_sparse_decode_key_value_hd128
decode_causal_attention_hd128_transposed_k_sparse_row{32,64,128}
```

Runtime gate:

```text
MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1
```

PagedCache allocation now creates all decode-side storage up front:

```text
key       [max_slots, B, H_kv, D]
decodeKey [B, H_kv, D, max_slots]
value     [B, H_kv, max_slots, D]
```

Cache compatibility now accepts `cache->maxSlots >= requestedMaxSlots`. This avoids request-internal cache recreation and prevents losing `decodeKeyReadyLength`.

## q=1 Versus q>1

q=1 is a normal no-repair decode step:

```text
append logical T:
  write PagedCache K/V[T]
  write DecodeK[T]
  Q[T] attends K[0..T]
```

This can be fused into one kernel because only one new row is written and then read by the same workgroup.

q>1 here is decode-repair, not contiguous multi-token decode:

```text
q=4 means:
  repair logical rows: r1, r2, r3
  append logical row:  T
```

The logical rows are sparse, may overwrite old PIC slots, and later rows can need the K/V written by earlier repair rows. OpenCL has no global synchronization inside one kernel across workgroups, so the q>1 path is intentionally two-stage:

```text
Stage A:
  append_sparse_decode_key_value_hd128
  writes all active rows to PagedCache K/V and DecodeK

Stage B:
  decode_causal_attention_hd128_transposed_k_sparse
  reads K from DecodeK and V from PagedCache value via slot_table
```

This preserves PagedCache semantics and avoids temporary decode KV allocation.

## Build And Device Validation

Build command:

```bash
MNN_TARGET_DEVICE=orangepi5plus \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Artifact check:

```text
.cache/output/mnn/artifacts/orangepi5plus/bin/pic_server:    ELF 64-bit LSB executable, ARM aarch64
.cache/output/mnn/artifacts/orangepi5plus/lib/libMNN.so:     ELF 64-bit LSB shared object, ARM aarch64
.cache/output/mnn/artifacts/orangepi5plus/lib/libMNN_CL.so:  ELF 64-bit LSB shared object, ARM aarch64
.cache/output/mnn/artifacts/orangepi5plus/lib/libpic_llm.so: ELF 64-bit LSB shared object, ARM aarch64
```

Sync:

```bash
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

Local checks:

```text
git diff --check: pass
OpenCL source generation: python3 opencl_codegen.py .
```

Runtime smoke:

```text
remote log:
/mnt/ssd/code/.cache/mnn_opencl_pic/runs/decode_transposed_k_qmulti_profile_20260703_014424_qwen_new_sparse/pic_server.log

decode_causal_attention_hd128_transposed_k_sparse lines: 144
```

Representative profile line:

```text
OpenCLPagedAttention profile op=decode_causal_attention_hd128_transposed_k_sparse layer=2 query=8 input_query=8 kv_len=514 lane=32 append_count=1 prepare_len=513 dense_kv_work=4112 causal_kv_work=542 us=5000 append_us=1316 attention_us=3666
```

This proves q>1 uses `prepareLen = kvLen - appendCount`, hits existing DecodeK for old rows, and writes the new active rows through the sparse append kernel.

No remaining local tunnel or remote `pic_server` was present after the run.

## A/B Setup

Common setup:

```text
device: OrangePi 5 Plus
backend: OpenCL
context: 512
mode: full-reuse
selector: lagged_attention_hkvd
attention_layer_idx: 1
repair_tokens: 0,1,3,5,7  => q=1,2,4,6,8
max_tokens: 8
warm_repeats: 1
repeats: 3
```

CSV outputs:

```text
.cache/mnn-pic-benchmark/decode_transposed_k/qwen3-4b_ctx512_qall_ab_old.csv
.cache/mnn-pic-benchmark/decode_transposed_k/qwen3-4b_ctx512_qall_ab_new.csv
.cache/mnn-pic-benchmark/decode_transposed_k/llama3.2-3b_ctx512_qall_ab_old.csv
.cache/mnn-pic-benchmark/decode_transposed_k/llama3.2-3b_ctx512_qall_ab_new.csv
.cache/mnn-pic-benchmark/decode_transposed_k/minicpm5-1b_ctx512_qall_ab_old.csv
.cache/mnn-pic-benchmark/decode_transposed_k/minicpm5-1b_ctx512_qall_ab_new.csv
```

`old` means default row-layout PagedCache path. `new` means `MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1`.

## Highest-Performance Table

TPOT in milliseconds:

| model | q | old row-PagedCache | new DecodeK | highest path | highest TPOT | DecodeK delta vs old |
|---|---:|---:|---:|---|---:|---:|
| qwen3-4b | 1 | 236.609 | 274.340 | old row-PagedCache | 236.609 | -13.8% |
| qwen3-4b | 2 | 709.640 | 721.961 | old row-PagedCache | 709.640 | -1.7% |
| qwen3-4b | 4 | 687.281 | 624.993 | new DecodeK | 624.993 | +10.0% |
| qwen3-4b | 6 | 1073.939 | 1065.913 | new DecodeK | 1065.913 | +0.8% |
| qwen3-4b | 8 | 1079.274 | 1098.561 | old row-PagedCache | 1079.274 | -1.8% |
| llama3.2-3b | 1 | 208.744 | 205.392 | new DecodeK | 205.392 | +1.6% |
| llama3.2-3b | 2 | 405.262 | 374.795 | new DecodeK | 374.795 | +8.1% |
| llama3.2-3b | 4 | 412.698 | 342.693 | new DecodeK | 342.693 | +20.4% |
| llama3.2-3b | 6 | 953.183 | 688.659 | new DecodeK | 688.659 | +38.4% |
| llama3.2-3b | 8 | 897.927 | 803.935 | new DecodeK | 803.935 | +11.7% |
| minicpm5-1b | 1 | 137.054 | 162.790 | old row-PagedCache | 137.054 | -15.8% |
| minicpm5-1b | 2 | 253.511 | 234.265 | new DecodeK | 234.265 | +8.2% |
| minicpm5-1b | 4 | 277.385 | 209.310 | new DecodeK | 209.310 | +32.5% |
| minicpm5-1b | 6 | 275.303 | 247.350 | new DecodeK | 247.350 | +11.3% |
| minicpm5-1b | 8 | 277.665 | 277.786 | old row-PagedCache | 277.665 | ~0.0% |

Highest path by tested shape:

```text
Qwen3-4B:
  q=1 old
  q=2 old
  q=4 DecodeK
  q=6 DecodeK
  q=8 old

Llama3.2-3B:
  q=1/2/4/6/8 DecodeK

MiniCPM5-1B:
  q=1 old
  q=2/4/6 DecodeK
  q=8 old/tie
```

## Interpretation

The transposed DecodeK view removes the worst row-layout K scan pressure for several shapes, especially Llama3.2-3B q>1 and MiniCPM5-1B q=2/4/6. It is not safe as a global default because Qwen q=1/q=2/q=8 and MiniCPM q=1 regress.

The current q>1 implementation is intentionally conservative: it keeps one attention workgroup per output row/head, so it still reloads DecodeK per sparse row. This satisfies the PagedCache correctness boundary and validates the transposed-K data path, but it is not the final small-Q tiled kernel from the target design. A true q2/q4/q8 kernel should load each K tile once per head group and update multiple q rows in the same workgroup or cooperative group.

Likely next step:

```text
decode_causal_attention_hd128_transposed_k_qtile_q2
decode_causal_attention_hd128_transposed_k_qtile_q4
decode_causal_attention_hd128_transposed_k_qtile_q8
```

The qtile version should preserve query-head parallelism and avoid the failed GQA serial grouping experiment from `orangepi/2026-07-02-22`.

## Recommendation

Keep `MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1` as an experimental/benchmark gate for now. Do not promote it as a universal default. For highest measured performance on this OrangePi ctx512 matrix, select the tested winner per model/q as listed above.

If adding production auto-selection, the heuristic must be model/shape aware and must be backed by the same A/B table. A blanket `head_dim=128 && q<=8` rule would regress Qwen and MiniCPM q=1.

## Follow-up P2: q=1 Append + Readonly Attention Split

Reason:

```text
q=1 still regressed on OrangePi for Qwen3-4B and MiniCPM5-1B.
The fused DecodeK q=1 kernel hides append/write cost and readonly attention cost in one number.
Old identity q=1 has record-queue advantages that DecodeK does not yet match.
```

Implementation additions:

```text
MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_ATTENTION_ONLY=1

decode path:
  ensureDecodeKeyReady(baseLogical, insideDecode=true)
  append_decode_key_value_hd128
  decode_causal_attention_hd128_transposed_k_readonly_row{32,64,128}
```

This path is intentionally env-gated and defaults off. It does not replace the fused q=1 DecodeK kernel in production. It is a micro profile path to isolate:

```text
decode_prepare_transpose_k   historical K transpose, should be outside timed decode
append_us                    current-token K/V + DecodeK write
attention_us                 readonly QK/softmax/QKV over DecodeK + PagedCache V
rank_us                      optional decode-repair rank capture, only active when metadata requests it
```

The q=1 profile line now reports:

```text
op=decode_causal_attention_hd128_transposed_k_readonly
append_us=...
attention_us=...
rank_us=...
append_count=1
prepare_len=...
attention_only=1
append_fused=0
decode_key_ready_hit=...
decode_key_prepared=...
decode_prepare_inside_decode=...
decode_prepare_us=...
decode_key_required=...
slot_identity=...
record_queue=0
```

The existing fused path also reports `decode_key_*` and `append_fused=1`, so P0 can reject any decode profile where `decode_prepare_inside_decode=1` and `decode_key_prepared=1` in the timed decode loop.

P1 hardening after the first q=1 split:

```text
SharedPagedCache::decodeKeyReadyPrefixSlots records the physical slot prefix that DecodeK was prepared from.
ensureDecodeKeyReady(..., insideDecode=true) now only allows ready-hit cases:
  same slot table version
  identity slot table
  exact prefix slot match
If none holds, it returns failure and does not launch decode_prepare_transpose_k inside timed decode.
```

This specifically addresses the coarse `decodeKeySlotTableVersion` issue: an append-only slot table version change no longer forces historical K to be transposed again when the prefix physical slots are unchanged.

The q=1 readonly micro path was also split into two gates:

```text
MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_READONLY_ONLY=1
  Requires DecodeK to already cover kv_len.
  Does not append K/V and does not prepare in decode.
  Intended for pure attention-only kernel comparison.

MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_ATTENTION_ONLY=1
  Prepares only historical prefix via ready-hit.
  Runs append_decode_key_value_hd128 as a separate timed append_us.
  Runs the readonly attention kernel as attention_us.
```

Local validation:

```bash
(cd source/backend/opencl/execution/cl && python3 opencl_codegen.py .)
git diff --check

JOBS=$(( $(nproc) / 2 )); [ "$JOBS" -lt 1 ] && JOBS=1
cmake --build .cache/build/mnn/aidlux_adreno_opencl --target MNN_CL -- -j"$JOBS"
cmake --build .cache/build/mnn/orangepi5plus --target MNN_CL -- -j"$JOBS"
```

Both `MNN_CL` builds passed after the q=1 split.

## Consolidated Optimization Plan

Current validated matrix, TPOT in milliseconds:

```text
+----------+-------------+-----+----------+--------------+----------------+---------+-----------+
| Device   | Model       | q   | old TPOT | DecodeK TPOT | DecodeK vs old | Winner  | Best TPOT |
+----------+-------------+-----+----------+--------------+----------------+---------+-----------+
| OrangePi | llama3.2-3b | q=1 | 208.7    | 205.4        | +1.6%          | DecodeK | 205.4     |
| OrangePi | llama3.2-3b | q=2 | 405.3    | 374.8        | +7.5%          | DecodeK | 374.8     |
| OrangePi | llama3.2-3b | q=4 | 412.7    | 342.7        | +17.0%         | DecodeK | 342.7     |
| OrangePi | llama3.2-3b | q=6 | 953.2    | 688.7        | +27.8%         | DecodeK | 688.7     |
| OrangePi | llama3.2-3b | q=8 | 897.9    | 803.9        | +10.5%         | DecodeK | 803.9     |
| OrangePi | minicpm5-1b | q=1 | 137.1    | 162.8        | -18.8%         | old     | 137.1     |
| OrangePi | minicpm5-1b | q=2 | 253.5    | 234.3        | +7.6%          | DecodeK | 234.3     |
| OrangePi | minicpm5-1b | q=4 | 277.4    | 209.3        | +24.5%         | DecodeK | 209.3     |
| OrangePi | minicpm5-1b | q=6 | 275.3    | 247.4        | +10.2%         | DecodeK | 247.4     |
| OrangePi | minicpm5-1b | q=8 | 277.7    | 277.8        | -0.0%          | old     | 277.7     |
| OrangePi | qwen3-4b    | q=1 | 236.6    | 274.3        | -15.9%         | old     | 236.6     |
| OrangePi | qwen3-4b    | q=2 | 709.6    | 722.0        | -1.7%          | old     | 709.6     |
| OrangePi | qwen3-4b    | q=4 | 687.3    | 625.0        | +9.1%          | DecodeK | 625.0     |
| OrangePi | qwen3-4b    | q=6 | 1073.9   | 1065.9       | +0.7%          | DecodeK | 1065.9    |
| OrangePi | qwen3-4b    | q=8 | 1079.3   | 1098.6       | -1.8%          | old     | 1079.3    |
| RhinoPi  | llama3.2-3b | q=1 | 111.3    | 77.4         | +30.5%         | DecodeK | 77.4      |
| RhinoPi  | llama3.2-3b | q=2 | 189.2    | 158.2        | +16.4%         | DecodeK | 158.2     |
| RhinoPi  | llama3.2-3b | q=4 | 192.7    | 175.8        | +8.8%          | DecodeK | 175.8     |
| RhinoPi  | llama3.2-3b | q=6 | 255.5    | 236.7        | +7.4%          | DecodeK | 236.7     |
| RhinoPi  | llama3.2-3b | q=8 | 265.3    | 241.6        | +8.9%          | DecodeK | 241.6     |
| RhinoPi  | minicpm5-1b | q=1 | 33.3     | 33.2         | +0.3%          | DecodeK | 33.2      |
| RhinoPi  | minicpm5-1b | q=2 | 80.6     | 69.2         | +14.1%         | DecodeK | 69.2      |
| RhinoPi  | minicpm5-1b | q=4 | 82.7     | 78.7         | +4.9%          | DecodeK | 78.7      |
| RhinoPi  | minicpm5-1b | q=6 | 116.0    | 112.1        | +3.4%          | DecodeK | 112.1     |
| RhinoPi  | minicpm5-1b | q=8 | 119.5    | 115.4        | +3.4%          | DecodeK | 115.4     |
| RhinoPi  | qwen3-4b    | q=1 | 114.0    | 95.9         | +15.9%         | DecodeK | 95.9      |
| RhinoPi  | qwen3-4b    | q=2 | 242.6    | 189.4        | +21.9%         | DecodeK | 189.4     |
| RhinoPi  | qwen3-4b    | q=4 | 250.4    | 205.5        | +17.9%         | DecodeK | 205.5     |
| RhinoPi  | qwen3-4b    | q=6 | 321.4    | 271.1        | +15.7%         | DecodeK | 271.1     |
| RhinoPi  | qwen3-4b    | q=8 | 323.4    | 272.0        | +15.9%         | DecodeK | 272.0     |
+----------+-------------+-----+----------+--------------+----------------+---------+-----------+
```

OrangePi q=1 record-queue micro data:

```text
+------------------+----------------+-------+----------------------+--------------------------------+
| Orange q=1 Model | Shape          | old   | DecodeK fused+record | DecodeK append+readonly+record |
+------------------+----------------+-------+----------------------+--------------------------------+
| llama3.2-3b      | L28 H24/KV8 G3 | -     | -                    | 211.8                          |
| minicpm5-1b      | L24 H16/KV2 G8 | 165.5 | 154.5                | 156.4                          |
| qwen3-4b         | L36 H32/KV8 G4 | 253.1 | 262.3                | 244.1                          |
+------------------+----------------+-------+----------------------+--------------------------------+
```

Rhino q=1 warmed-cache record-queue micro data:

```text
+----------------+-------+----------------------+--------------------------------+------------------------+
| Rhino q=1 Model | old   | DecodeK fused+record | DecodeK append+readonly+record | Best                   |
+----------------+-------+----------------------+--------------------------------+------------------------+
| llama3.2-3b    | 109.2 | 110.1                | 109.2                          | old / append tie       |
| minicpm5-1b    | 32.7  | 32.8                 | 33.2                           | old                    |
| qwen3-4b       | 114.4 | 114.0                | 113.7                          | append+readonly        |
+----------------+-------+----------------------+--------------------------------+------------------------+
```

Rhino command shape:

```text
device=rhino backend=OpenCL frequency=max ctx=512 repair_tokens=0 max_tokens=8 warm=1 repeats=3
server_env:
  old:                  LD_PRELOAD=/usr/lib/libOpenCL_adreno.so
  DecodeK fused:         LD_PRELOAD=/usr/lib/libOpenCL_adreno.so MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1
  DecodeK append+ro:     LD_PRELOAD=/usr/lib/libOpenCL_adreno.so MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1 MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_ATTENTION_ONLY=1
output:
  .cache/mnn-pic-benchmark/decode_transposed_k_record_q1/rhino_*_{old,new_record,attention_only_record}.csv
remote log:
  /mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/decode_transposed_k_record_q1_warm_20260703_035142/
```

The first Rhino pass after syncing the new artifact printed one expected runtime-cache reset line:

```text
Cache invalid, will be reset
```

That cold-cache pass was not used for the table above. The warmed second pass log scan was clean:

```text
no ERROR
no Cache invalid
no target unavailable
no async persistent PIC cache read failed
no decode_prepare_transpose_k
no decode_prepare_inside_decode=1
```

Interpretation:

```text
OrangePi/Mali:
  q=1 append+readonly is useful for Qwen but not for MiniCPM, and Llama record data is incomplete.

Rhino/Adreno:
  q=1 old/fused/append+readonly are near noise for MiniCPM and Llama.
  Qwen slightly prefers append+readonly, but only by about 0.6% over old and 0.3% over fused.

Conclusion:
  Do not promote append+readonly as a global q=1 default.
  If a q=1 heuristic is added, it must be device-family and shape scoped.
  For Adreno, keep q=1 conservative until attention-only kernel data shows a real margin.
```

Execution plan:

1. P0 contract: keep `decodeKey` preallocated with PagedCache and treat it as a decode-only view, not a second KV owner. `decode_prepare_transpose_k` must run only after prefill/full-reuse hydration or other non-decode setup. Any timed decode profile with `decode_prepare_inside_decode=1` and `decode_key_prepared=1` is invalid.
2. P0 q=1 launch parity: retain record queue for both DecodeK fused and DecodeK append+readonly paths. Profile mode intentionally disables record queue so attribution remains visible; formal TPOT must run with profile off.
3. P1 q=1 routing: do not make append+readonly the global default. On OrangePi, Qwen benefits from append+readonly, MiniCPM prefers fused, and Llama regresses with append+readonly. The likely shape signal is per-request duplicated append work in fused q=1:
   `layer_nums * num_heads` is 1152 for Qwen, 672 for Llama, and 384 for MiniCPM. Candidate heuristic for Mali only: when record queue is available and `layer_nums * num_heads >= 1024`, try append+readonly first; otherwise keep fused DecodeK. This must be validated on Rhino before production use because Adreno already shows DecodeK wins broadly with the current fused path.
4. P1 q=1 validation result: Rhino q=1 warmed-cache record smoke shows no broad append+readonly win. Keep any q=1 append+readonly route scoped to `device_family + layer_nums * num_heads` and require a minimum margin before enabling it. Never route only by `groupSize`.
5. P2 q>1 kernel direction: the current q>1 sparse DecodeK path is correct but still row-independent. The real optimization is qtile:
   `decode_causal_attention_hd128_transposed_k_qtile_q2/q4/q8`, where one workgroup handles multiple Q rows for one head, loads each K tile once, maintains per-Q softmax accumulators, and shares the same K stream across q rows. This is expected to help Qwen q=2/q8 and reduce duplicate K reads for all models.
6. P2 q>1 safety: keep the two-stage append boundary. Repair rows can overwrite sparse logical slots and later active rows can depend on earlier repair K/V. Without a kernel boundary there is no global synchronization. qtile should start after `append_sparse_decode_key_value_hd128`, not fuse sparse append into the attention kernel.
7. P2 V-side and slot-table work: DecodeK improves QK only. QKV still reads PagedCache value. Add QK-only and QKV-only detail profile to decide whether to add identity-slot V kernels or value transpose. For full-reuse identity slot tables, add a value identity fast path only if QKV is measurable.
8. P3 autotune: lane 32/64/128 and qtile shape should be keyed by `device_family + head_dim + kv_len_bucket + q + num_heads/kv_heads`. OrangePi and Rhino differ enough that a fixed heuristic is not acceptable.

Promotion rule:

```text
Step 1: new_attention_only must beat old_attention_only for the same q/device/model.
Step 2: if Step 1 passes, remove fixed overhead: prepare, append, rank, launch/record.
Step 3: if Step 1 fails, tune kernel internals first: lane, local memory, K/V load shape.
Step 4: only promote a default route when formal TPOT is non-regressing for all:
        llama3.2-3b, minicpm5-1b, qwen3-4b x OrangePi/Rhino x q=1/2/4/6/8.
```

## q=1 route hardening after route4-route6

Problem found after the first q=1 auto route:

```text
Global MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1 was still too risky for q=1.
Qwen append+readonly was not stable across refreshed artifacts, and q=1 old-route shapes
could still pay control-plane overhead if the code entered transposed-K route checks.
```

Code decisions:

```text
1. q=1 auto now resolves to old for all devices/models.
2. q=1 DecodeK fused / append+readonly are explicit opt-in only:
   MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_Q1_ROUTE=fused|append_readonly
   or MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_ATTENTION_ONLY=1 /
      MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_READONLY_ONLY=1.
3. q=1 old route bypasses the entire transposed-K q=1 branch tree.
4. predecode DecodeK prepare is gated by either an explicit q=1 DecodeK route
   or actual q>1 decode repair, tracked by PagedKVMeta::pic_decode_repair_tokens_per_step.
```

Debug proof:

```text
run_id:
  decode_transposed_k_q1_debug_rhino_llama_route6_20260703

server env:
  LD_PRELOAD=/usr/lib/libOpenCL_adreno.so
  MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1
  MNN_PIC_DECODE_DEBUG=1

log result:
  no decode_key prepare
  no decode_prepare_transpose_k
  no transposed_k q1 route print
  only old copy_paged_kv invalidation appears
```

route6 q=1 median smoke:

```text
+==========+=============+==========+==========+===============+======+
| device   | model       | old_med  | route6   | speedup       | pass |
+==========+=============+==========+==========+===============+======+
| RhinoPi  | llama3.2-3b |  107.341 |  111.540 |         0.962 | no   |
| RhinoPi  | minicpm5-1b |   26.552 |   26.550 |         1.000 | yes  |
| RhinoPi  | qwen3-4b    |  112.876 |  112.534 |         1.003 | yes  |
| OrangePi | llama3.2-3b |  226.494 |  204.491 |         1.108 | yes  |
| OrangePi | minicpm5-1b |   60.662 |   56.086 |         1.082 | yes  |
| OrangePi | qwen3-4b    |  256.194 |  266.322 |         0.962 | no   |
+==========+=============+==========+==========+===============+======+
```

Interpretation:

```text
The remaining q=1 differences are not from DecodeK prepare or DecodeK q1 kernels;
route6 bypasses them. OrangePi raw repeats still show large device variance
(for example 187-232 ms range on Llama and 247-341 ms range on Qwen old).
Therefore q=1 should stay production-old, with DecodeK q=1 kept as an explicit
microbenchmark path only. The next real optimization target is q>1 qtile.
```

## Decode Optimization Plan V2

Target:

```text
For llama3.2-3b, minicpm5-1b, qwen3-4b on OrangePi and Rhino:
  q=1/2/4/6/8 new DecodeK TPOT must be <= old identity TPOT.

This target is not satisfied by routing q=1 auto to old. Old fallback is only a
guardrail while the DecodeK implementation is being fixed.
```

Current diagnosis:

```text
q=1:
  The current DecodeK q=1 kernels use the same row-softmax shape as old identity:
    one workgroup per output row/head
    one lane owns one K position
    each lane scans D=128

  This does not fully exploit DecodeK layout. DecodeK makes K contiguous along
  logical K for each D, but the current row kernel still computes one K per lane
  and reloads the same K stream separately for every query head. It can win on
  some Adreno/Mali shapes, but it is not a robust new path yet.

q>1:
  The current sparse DecodeK path is correct but row-independent:
    append_sparse_decode_key_value_hd128
    decode_causal_attention_hd128_transposed_k_sparse_row{32,64,128}

  The attention kernel launches gws=[lanes, attnLen, batch*heads], so q=2/4/6/8
  still reloads the same DecodeK range once per Q row and per query head.
```

Plan:

```text
P0. Measurement contract
  - Keep DecodeK preallocated with PagedCache.
  - Keep decode_prepare_transpose_k outside timed decode.
  - Any timed profile with decode_prepare_inside_decode=1 and decode_key_prepared=1 is invalid.
  - Split metrics into prepare_us, append_us, attention_us, rank_us, lane,
    record_queue, slot_identity, prefix_stable, decode_key_ready_hit.
  - Rename the mental model:
      ATTENTION_ONLY=1 today is append+readonly, not pure attention-only.
      READONLY_ONLY=1 is pure readonly only if DecodeK already covers kv_len.

P1. q=1 apples-to-apples microbench
  - Add an old identity attention-only profile path:
      K/V already written, then run decode_causal_attention_hd128_identity_row* only.
  - Add a real new DecodeK attention-only profile path:
      DecodeK already covers current kv_len, then run
      decode_causal_attention_hd128_transposed_k_readonly_row* only.
  - Compare old_attention_only vs new_attention_only before optimizing append,
    rank, launch, or record queue. If new attention-only does not win, fix the
    kernel body first.

P2. q=1 kernel variants that can actually use DecodeK
  - K-vectorized DecodeK q=1:
      process multiple adjacent K positions per lane using DecodeK's contiguous
      K dimension for each D, then reduce a wider score tile.
  - GQA cooperative q=1:
      one workgroup handles a small query-head tile for the same kv_head, shares
      the K stream, and keeps independent softmax accumulators per query head.
      Do not serialize the whole GQA group in one worker; test H_TILE=2/4.
  - Identity-slot V fast path:
      when slot_table is identity, read value_cache by logical position directly
      and skip slot_table load/branch.
  - Lane/tile autotune:
      key by device_family + kv_len_bucket + num_heads/kv_heads + model head shape.

P3. q=1 end-to-end route
  - Only after P1/P2 attention-only wins, decide fused append+attention versus
    append+readonly.
  - Match old record-queue/fixed-dispatch behavior for the chosen q=1 new path.
  - Promotion requires no-regression TPOT across both devices and all 3 models.

P4. q>1 qtile
  - Keep the two-stage append boundary:
      append_sparse_decode_key_value_hd128
      qtile attention
    Sparse repair rows need a kernel-launch boundary before attention reads them.
  - Implement qtile kernels:
      decode_causal_attention_hd128_transposed_k_qtile_q2
      decode_causal_attention_hd128_transposed_k_qtile_q4
      decode_causal_attention_hd128_transposed_k_qtile_q8
    Use q8 with masks for q=6.
  - One workgroup handles multiple Q rows for one head/kv_head tile, loads each
    K tile once, and maintains per-Q softmax m/l/o accumulators.
  - Stage V tile locally only if profile shows QKV/V is a bottleneck and local
    memory pressure does not hurt occupancy.

P5. Formal acceptance
  - Warm OpenCL runtime/autotune cache first.
  - Run with profile off for formal TPOT; profile detail only for attribution.
  - Matrix:
      devices: OrangePi, Rhino
      models: llama3.2-3b, minicpm5-1b, qwen3-4b
      q: 1,2,4,6,8
  - Pass condition:
      new median TPOT <= old median TPOT for every cell.
      No cell is allowed to pass only because q=1 silently fell back to old.
```

Immediate implementation order:

```text
1. Add true q=1 old/new attention-only profile paths.
2. Run q=1 attention-only on OrangePi/Rhino for all 3 models.
3. If new q=1 attention-only loses, implement K-vectorized and GQA-cooperative
   DecodeK q=1 variants before touching append/fused routing.
4. Once q=1 attention wins, restore append/fused/record queue and test TPOT.
5. Then implement q>1 qtile, starting with q4/q8 because current data shows
   most DecodeK benefit there and the duplicate K-load cost is largest.
```

## Decode Optimization Plan V3

New root cause from q=1 attention-only smoke:

```text
run_id:
  decode_q1_attention_only_new_debug_20260703

observed sequence:
  text-cache/prefill creates layer cache with max_slots=4096, source_segments=0
  decode request binds one 512-token persistent PIC source segment
  first decode resize now needs max_slots=4608
  ensureCache recreates layer cache and resets decodeKeyReadyLength
  copy_paged_kv then invalidates DecodeK again

critical log shape:
  prefill: q=512 new_kv=512 capacity=4096 max_slots=4096 source_segments=0
  decode:  q=1   new_kv=1   capacity=4096 max_slots=4608 source_segments=1
```

Interpretation:

```text
The q=1 new attention-only path is not yet a valid measurement because the
decode request still performs full-reuse source hydration / cache expansion
inside the first decode step. Even if decode_prepare_transpose_k is not printed,
the first decode layer can still pay old-K work through source hydrate and cache
reallocation. This violates the intended contract:

  before timed decode:
    PagedCache is already allocated for request slots + source slots
    persistent PIC source has already been hydrated to current request slots
    DecodeK already covers logical prefix [0, baseLogical)

  inside timed decode:
    only append current K/V/DecodeK
    only run attention/rank/sample
```

Correct P1 boundary:

```text
PIC full-reuse with max_tokens > 0 must insert an explicit decode prepare stage
between prefill/full-reuse and decode.

prepare stage responsibilities:
  1. reserve request logical capacity:
       logical_length + max_tokens + max(tokens_per_decode_step, 1)
  2. reserve physical PagedCache capacity:
       request_capacity + aligned(persistent_source_slot_count)
     This is not a temporary decode allocation; it is part of SharedPagedCache.
  3. hydrate all persistent PIC source segments for all layers into PagedCache.
  4. run decode_prepare_transpose_k for all layers for kv_len=logical_length.
  5. capture slot prefix so decode append does not invalidate old DecodeK.
  6. optionally detach hydrated external segments before decode so decode does
     not schedule lazy source reads.

decode stage responsibilities:
  q=1:
    append_decode_key_value_hd128 for current token only
    readonly/attention kernel reads DecodeK prefix + appended current K

  q>1:
    append_sparse_decode_key_value_hd128 for active repair rows + current token
    qtile attention reads DecodeK, value still from PagedCache
```

Implementation implication:

```text
Do not rely on suffix prefill to trigger prepare. In full-reuse requests the
suffix can be empty, so no PagedAttention op may run before decode. The server
needs a real prepare call after appendExternalPagedKV()/suffix prefill and before
Llm::decode(maxTokens).

API shape:
  Llm::preparePagedDecode(int maxTokens)

OpenCL backend shape:
  prepare current request PagedCache by layer:
    ensure SharedPagedCache capacity is large enough
    hydrateExternalSegments(layer, logical_length)
    ensureDecodeKeyReady(logical_length, insideDecode=false)

Acceptance log:
  before "PIC server decode debug branch=pic":
    op=hydrate layer=0..N-1
    op=decode_prepare_transpose_k layer=0..N-1 inside_decode=0

  after decode starts:
    no "PIC OpenCL PA cache create" for existing layers
    no op=hydrate
    no op=decode_prepare_transpose_k with decode_key_prepared=1
    q=1 attention-only setup has decode_key_ready_hit=1, decode_key_prepared=0
```

Revised immediate order:

```text
1. Implement explicit full-reuse decode prepare:
     server: call Llm::preparePagedDecode(maxTokens) before Llm::decode().
     backend: prepare every layer cache, not just layers reached by suffix prefill.

2. Re-run q=1 new attention-only smoke:
     must hit decode_transposed_k_q1_attention_only_setup
     must hit decode_causal_attention_hd128_transposed_k_readonly
     must not hydrate/recreate/prepare historical K after decode starts.

3. Only then compare old_identity_attention_only vs new_decodek_attention_only.

4. If new attention-only loses, implement kernel-internal q=1 variants:
     K-vectorized DecodeK row kernel
     GQA cooperative H_TILE=2/4 kernel
     identity-slot V fast path

5. If new attention-only wins, optimize overhead:
     append current K/V/DecodeK
     record queue / fixed dispatch parity
     rank split or fused rank, depending on profile

6. After q=1 passes, implement q>1 qtile after the existing sparse append
   boundary. q=2/4/8 get native kernels; q=6 uses q8 with masks.
```

## Final q=1 Default Route Hardening

After moving prepare out of timed decode, the remaining q=1 question is not
whether DecodeK is theoretically better. It is whether the actual q=1 route can
beat the old identity path after launch, setArg, append, record queue, rank, and
V-side slot-table costs are included.

Current production default is therefore conservative:

```text
Mali / OrangePi:
  Llama3.2-3B  heads=24 kv_heads=8 -> append_readonly DecodeK
  MiniCPM5-1B  heads=16 kv_heads=2 -> old identity
  Qwen3-4B     heads=32 kv_heads=8 -> old identity

Adreno / Rhino:
  Llama3.2-3B  heads=24 kv_heads=8 -> fused DecodeK
  MiniCPM5-1B  heads=16 kv_heads=2 -> old identity
  Qwen3-4B     heads=32 kv_heads=8 -> fused DecodeK
```

The final route selector keeps `MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1`
enabled without forcing every q=1 shape through DecodeK. Explicit override
remains available:

```text
MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_Q1_ROUTE=old|fused|append_readonly|auto
```

Build and sync after route hardening:

```bash
MNN_TARGET_DEVICE=orangepi5plus \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/

MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Both builds succeeded. `file` confirmed AArch64 `pic_server`, `libMNN.so`,
`libMNN_CL.so`, and `libpic_llm.so` for OrangePi and Rhino. Both build dirs were
reconfigured because the script detected empty cached cross-compiler/sysroot
fields.

Final q=1 run:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi,rhino \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512 \
  --pic-repair-tokens 0 \
  --pic-max-tokens 16 \
  --repeats 3 \
  --warm-repeats 1 \
  --skip-normal \
  --run-id decode_q1_orangepi_rhino_all_auto_final_20260703
```

Final `ctx512`, q=1 TPOT:

```text
+----------+-------------+-----+-------------+-------------+------------+-----------+----------+-------------+
| device   | model       | ctx | final route | old_current | auto_final | old/final | delta ms | auto_before |
+----------+-------------+-----+-------------+-------------+------------+-----------+----------+-------------+
| orangepi | Llama3.2 3B | 512 | append      | 225.126     | 207.574    | 1.085     | -17.552  | 211.783     |
| orangepi | MiniCPM5-1B | 512 | old         | 56.007      | 50.267     | 1.114     | -5.740   | 59.694      |
| orangepi | Qwen3-4B    | 512 | old         | 259.488     | 237.219    | 1.094     | -22.269  | 265.064     |
| rhino    | Llama3.2 3B | 512 | fused       | 101.179     | 75.948     | 1.332     | -25.231  | 75.675      |
| rhino    | MiniCPM5-1B | 512 | old         | 26.569      | 26.466     | 1.004     | -0.103   | 26.531      |
| rhino    | Qwen3-4B    | 512 | fused       | 112.965     | 96.190     | 1.174     | -16.776  | 95.701      |
+----------+-------------+-----+-------------+-------------+------------+-----------+----------+-------------+
```

Notes:

- `old_current` is the same-day forced-old run:
  `decode_q1_orangepi_rhino_all_current_old_20260703`.
- `auto_before` is the earlier auto selector before reverting OrangePi
  MiniCPM5-1B to old identity.
- No `failures.csv` was produced for the final run; all six rows have
  `status=ok`.
- The old-routed final rows can still move with thermal / frequency / cache
  state. Treat the route decision as the stable conclusion, not every
  old-vs-old millisecond delta as kernel speedup.

## Decode Optimization Plan V4

Target invariant:

```text
decode timed path must never prepare historical K
decode timed path may only append new active token K/V/DecodeK and run attention/rank
PagedCache remains the only runtime KV owner
DecodeK is a derived transposed K view allocated with PagedCache, not a temporary cache
```

P0: keep the control-plane contract enforced.

```text
prefill/full-reuse:
  build/hydrate current request PagedCache
  reserve decode logical capacity and persistent source physical slots
  run preparePagedDecode(max_tokens)
  optionally run decode_prepare_transpose_k only if route/q requires DecodeK

decode:
  no cache recreation
  no lazy external hydrate
  no decode_prepare_transpose_k for old K
  q=1 appends current token only
  q>1 appends active repair rows + current token only
```

Acceptance profile:

```text
before "PIC server decode debug branch=pic":
  op=prepare_paged_decode
  op=hydrate layer=...
  op=decode_prepare_transpose_k inside_decode=0 when DecodeK is required

after decode branch starts:
  no op=hydrate
  no op=decode_prepare_transpose_k with inside_decode=1
  no "cache create" for existing layer cache
```

P1: finish q=1 from evidence, not intuition.

```text
1. Keep old identity as q=1 fallback wherever final TPOT does not prove DecodeK wins.
2. For q=1 DecodeK winners, keep record queue parity:
     fused DecodeK record path
     append+readonly two-kernel record path
3. Split q=1 profile into:
     append current K/V/DecodeK
     QK-only
     QKV-only
     rank
     launch/record overhead
4. If QK-only wins but full QKV loses:
     optimize V side first
     add identity slot-table fast path
     remove avoidable int load / branch in value read
5. If QK-only loses:
     implement K-vectorized DecodeK row kernel
     implement GQA cooperative head tile so grouped query heads share K load
     autotune lane=32/64/128 per device+heads+kv_len
```

P2: q>1 must become true multi-Q decode attention.

Current q>1 is correct but incomplete:

```text
Stage A:
  append_sparse_decode_key_value_hd128
  writes repair rows + current token to PagedCache K/V and DecodeK

Stage B:
  decode_causal_attention_hd128_transposed_k_sparse_row{32,64,128}
  each Q row scans historical K mostly independently
```

Next kernel family:

```text
decode_causal_attention_hd128_transposed_k_qtile_q2
decode_causal_attention_hd128_transposed_k_qtile_q4
decode_causal_attention_hd128_transposed_k_qtile_q8
q=6 routes to q8 kernel with row mask
```

Kernel shape:

```text
workgroup handles:
  Q_TILE = 2 / 4 / 8 active rows
  one query head or one GQA head group tile
  K_TILE = 16 / 32 / 64 logical positions

loop:
  load DecodeK[B, kv_head, dim, k:k+K_TILE] once
  for each active q row:
    compute q dot shared K tile
    apply per-row causal limit
    update row-local softmax state
  load V tile once when possible, or use identity-slot fast path
  accumulate O for all Q rows
```

This is different from prefill qtile:

```text
prefill qtile:
  many contiguous Q rows x many K rows
  good reuse is natural because Q rows are dense and causal rectangle is large

decode qtile:
  Q is tiny and may be sparse repair rows
  value comes from PagedCache value, not DecodeK
  benefit comes from sharing old K loads across q=2/4/8 rows
  append/update must happen before attention because later rows may read earlier repaired K/V
```

P3: route and autotune by measured shape.

```text
route key:
  device family: mali / adreno
  num_heads / kv_heads / head_dim
  kv_len bucket
  q: 1 / 2 / 4 / 6 / 8
  repair rows sparse distribution if available

lane candidates:
  32 / 64 / 128 for old identity, q1 DecodeK, and qtile DecodeK

hard rule:
  no shape becomes default DecodeK unless final TPOT beats old or is neutral within noise
```

P4: benchmark matrix before promoting q>1.

```text
devices:
  OrangePi formal
  Rhino extra-profile

models:
  llama3.2-3b
  minicpm5-1b
  qwen3-4b

contexts:
  start 512
  then extend to 1024/1536/2048/2560

q:
  repair_tokens=0/1/3/5/7
  effective q=1/2/4/6/8

tables:
  old identity row PagedCache
  DecodeK current sparse-row
  DecodeK qtile
  best route
  route reason
```

## P0 Profile Boundary and Field Parity Smoke V2

After adding request-level `preparePagedDecode()` and conservative q=1 routing, one profile gap remained: the old `decode_causal_attention_hd128_identity_fused_kv` profile line did not expose the required decode attribution fields. This affected old-routed MiniCPM5-1B and some Qwen3-4B q=1 runs.

Patch scope:

```text
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp

runDecodeCausalAttentionHD128IdentityFusedKV:
  add profile/detail split for attention_us and rank_us
  print append_us=0, attention_us, rank_us
  print record_queue=0, slot_identity=1
  print decode_key_ready_hit=0, decode_prepare_inside_decode=0, prepare_us=0

runDecodeCausalAttentionHD128IdentityFusedKVGQA:
  same field parity, preserving group_size and kv_heads
```

No kernel dispatch, route, or math path was changed by this patch.

Build and sync:

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

MNN_TARGET_DEVICE=aidlux_adreno_opencl BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/

rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Both artifacts were checked with `file`; `pic_server`, `libMNN.so`, `libMNN_CL.so`, and `libpic_llm.so` are AArch64 ELF outputs for both OrangePi and Rhino.

Smoke command:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1 MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi,rhino \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512 \
  --pic-repair-tokens 0,1 \
  --pic-max-tokens 2 \
  --repeats 1 \
  --warm-repeats 0 \
  --skip-normal \
  --run-id decode_profile_fields_prepare_smoke_v2_20260703
```

Result CSV:

```text
decode_tpot_long.csv: 12 ok rows
failures.csv: not generated
```

Profile timings in this run are not performance-valid because profile/detail mode inserts synchronization. The run only validates attribution fields and the decode/prepare boundary.

Remote log parse:

```text
device,model,branches,prepare_total,prepare_in_timed_decode,prepare_inside_decode,attention_lines,rank_lines,q1_old,q1_new,qgt1_sparse,q_values,missing_required_fields
orangepi,llama3.2-3b,2,84,0,0,112,1,56,28,28,1/2,0
orangepi,minicpm5-1b,2,24,0,0,96,1,72,0,24,1/2,0
orangepi,qwen3-4b,2,36,0,0,144,1,108,0,36,1/2,0
rhino,llama3.2-3b,2,84,0,0,112,1,56,28,28,1/2,0
rhino,minicpm5-1b,2,24,0,0,96,1,72,0,24,1/2,0
rhino,qwen3-4b,2,108,0,0,144,1,72,36,36,1/2,0
```

Observed op families:

```text
orangepi llama3.2-3b:
  identity=56, transposed_k_readonly=28, transposed_k_sparse=28
orangepi minicpm5-1b:
  identity=48, identity_fused_kv=24, transposed_k_sparse=24
orangepi qwen3-4b:
  identity=72, identity_fused_kv=36, transposed_k_sparse=36

rhino llama3.2-3b:
  identity=56, transposed_k_fused_kv=28, transposed_k_sparse=28
rhino minicpm5-1b:
  identity=48, identity_fused_kv=24, transposed_k_sparse=24
rhino qwen3-4b:
  identity=72, transposed_k_fused_kv=36, transposed_k_sparse=36
```

Conclusion:

```text
P0 is now locked:
  full historical K transpose is outside timed decode
  decode_prepare_inside_decode is always 0
  all decode attention/rank profile lines expose the required fields
  q=2 path reaches decode_causal_attention_hd128_transposed_k_sparse

Remaining performance work is not another prepare-boundary fix.
The next bottleneck is q>1 attention shape: current sparse-row DecodeK still scans old K per Q row.
```

## Decode Optimization Plan V5

P1 stays conservative for q=1:

```text
Default q=1 route remains evidence-driven:
  Mali:
    Llama3.2-3B -> append+readonly DecodeK
    MiniCPM5-1B -> old identity
    Qwen3-4B -> old identity
  Adreno:
    Llama3.2-3B -> fused DecodeK
    MiniCPM5-1B -> old identity
    Qwen3-4B -> fused DecodeK

Reason:
  q=1 has too little Q reuse.
  DecodeK only helps if its better K load pattern beats append/write/launch/rank overhead.
  Old identity remains valid where final TPOT does not prove DecodeK wins.
```

P2 is the real next implementation:

```text
Keep current Stage A:
  append_sparse_decode_key_value_hd128
  writes repair rows/current token K/V into PagedCache
  writes K into DecodeK at the same time

Replace current Stage B for q>1:
  current: one sparse-row attention workgroup per Q row/head
  target: one qtile attention workgroup per Q_TILE/head

Kernel family:
  decode_causal_attention_hd128_transposed_k_qtile_q2
  decode_causal_attention_hd128_transposed_k_qtile_q4
  decode_causal_attention_hd128_transposed_k_qtile_q8
  q=6 uses q8 with valid-row mask
```

Workgroup sketch:

```text
for each batch/head/q_tile:
  rows = active Q logical rows in this tile
  initialize per-row softmax state:
    m[row], l[row], o[row][128]

  for k_block in old K:
    load DecodeK[k_block, dim] once for the kv head
    for row in rows:
      apply row-specific causal limit
      compute q[row] dot shared K block
      update row-local online softmax

    load V block from PagedCache value
    for row in rows:
      accumulate o[row] with the same softmax weights

  write O for each valid q row
```

Implementation constraints:

```text
1. Do not move historical prepare into decode.
2. Do not merge Stage A and Stage B for q>1 unless row dependency is proven safe.
   Later decode-repair rows may need earlier repaired K/V.
3. Slot-table identity fast path should be a dedicated qtile variant or branch-free path.
4. Lane must remain tunable: 32/64/128 by device + heads + kv_len + q.
5. The selector only promotes qtile when final TPOT beats old/current for the shape.
```

Validation ladder:

```text
Step 1: q=2 qtile correctness and profile on ctx512, all three models, OrangePi + Rhino.
Step 2: q=4/q8 kernels; q=6 through q8 mask.
Step 3: compare old identity, current DecodeK sparse-row, and qtile DecodeK:
  attention_us
  append_us
  rank_us
  TPOT
Step 4: extend contexts to 1024/1536/2048/2560.
Step 5: enable route table only for no-regression shapes.
```

## Final Route-V2 And Endpoint TPOT

Implementation state after the final patch:

```text
DecodeK storage:
  SharedPagedCache::decodeKey is allocated with PagedCache.
  Layout is [B, H_kv, D, max_slots].

Historical prepare:
  Llm::preparePagedDecode(maxTokens) runs before server decode.
  prepareDecodePrefix() only calls ensureDecodeKeyReady() when the actual route needs DecodeK.
  Timed decode treats decode_prepare_inside_decode=1 as invalid.

Decode append:
  q=1 DecodeK variants append current K/V and DecodeK together.
  q>1 decode-repair keeps Stage A append_sparse_decode_key_value_hd128, then Stage B attention.

Mali route-v2:
  q=1: old identity.
  Llama3.2-3B 24/8 q=2/q=4: qtile candidate remains enabled.
  Qwen3-4B 32/8: qtile disabled after q4 endpoint regression.
  MiniCPM5-1B 16/2: qtile disabled by shape.

Adreno route:
  q=1 fused DecodeK for 24/8 and 32/8.
  q>1 qtile remains enabled.
```

The Qwen q4 route-off smoke used:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=0 MNN_PIC_REQUEST_PROFILE=1 MNN_PIC_DECODE_DEBUG=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models qwen3-4b \
  --contexts 512 --pic-repair-tokens 3 --pic-max-tokens 2 \
  --repeats 1 --warm-repeats 0 --skip-normal \
  --run-id decode_route_profile_orangepi_qwen_q4_routeoff_20260703
```

Remote log scan:

```text
run_id=decode_route_profile_orangepi_qwen_q4_routeoff_20260703
qtile=0
decode_prepare_inside_decode=0
prepare_paged_decode=1
```

Build and sync after route-v2:

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

The build completed and installed `pic_server`, `libMNN.so`, `libMNN_CL.so`, `libMNN_Vulkan.so`, and `libpic_llm.so`. It did a full reconfigure because the cross build cache had empty compiler/sysroot entries. The only warnings were the existing `perfetto_c.cc` `#pragma system_header` warnings.

OrangePi same-window A/B used the patched artifact:

```text
old_current:
  run_id=decode_ab_old_current_orangepi_20260703
  env: MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=0

new_route_v2:
  run_id=decode_ab_new_mali_route_v2_orangepi_20260703
  env: MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1

common:
  device=orangepi
  context=512
  repair_tokens=0,1,3,5,7 => q=1,2,4,6,8
  max_tokens=16
  warm_repeats=1
  repeats=2
```

OrangePi same-window result:

```text
device    model        ctx  q  old_ms    new_ms    delta_ms  old/new  status
--------  -----------  ---  -  --------  --------  --------  -------  ------
orangepi  Llama3.2 3B  512  1  235.143   232.314   -2.829    1.012x   OK
orangepi  Llama3.2 3B  512  2  408.689   351.285   -57.403   1.163x   OK
orangepi  Llama3.2 3B  512  4  373.776   388.522   +14.746   0.962x   REG
orangepi  Llama3.2 3B  512  6  814.644   723.384   -91.260   1.126x   OK
orangepi  Llama3.2 3B  512  8  844.873   770.136   -74.738   1.097x   OK
orangepi  MiniCPM5-1B  512  1  61.675    60.243    -1.433    1.024x   OK
orangepi  MiniCPM5-1B  512  2  280.510   244.134   -36.376   1.149x   OK
orangepi  MiniCPM5-1B  512  4  240.754   285.436   +44.682   0.843x   REG
orangepi  MiniCPM5-1B  512  6  285.706   270.160   -15.546   1.058x   OK
orangepi  MiniCPM5-1B  512  8  298.453   292.401   -6.052    1.021x   OK
orangepi  Qwen3-4B     512  1  249.352   269.525   +20.173   0.925x   REG
orangepi  Qwen3-4B     512  2  805.201   740.569   -64.632   1.087x   OK
orangepi  Qwen3-4B     512  4  826.331   843.580   +17.249   0.980x   REG
orangepi  Qwen3-4B     512  6  1107.961  978.142   -129.819  1.133x   OK
orangepi  Qwen3-4B     512  8  1114.840  1010.031  -104.808  1.104x   OK
```

Combined report uses OrangePi same-window old and Rhino historical final old:

```text
device    model        ctx  q  old_ms    new_ms    delta_ms  old/new  status
--------  -----------  ---  -  --------  --------  --------  -------  ------
orangepi  Llama3.2 3B  512  1  235.143   232.314   -2.829    1.012x   OK
orangepi  Llama3.2 3B  512  2  408.689   351.285   -57.403   1.163x   OK
orangepi  Llama3.2 3B  512  4  373.776   388.522   +14.746   0.962x   REG
orangepi  Llama3.2 3B  512  6  814.644   723.384   -91.260   1.126x   OK
orangepi  Llama3.2 3B  512  8  844.873   770.136   -74.738   1.097x   OK
orangepi  MiniCPM5-1B  512  1  61.675    60.243    -1.433    1.024x   OK
orangepi  MiniCPM5-1B  512  2  280.510   244.134   -36.376   1.149x   OK
orangepi  MiniCPM5-1B  512  4  240.754   285.436   +44.682   0.843x   REG
orangepi  MiniCPM5-1B  512  6  285.706   270.160   -15.546   1.058x   OK
orangepi  MiniCPM5-1B  512  8  298.453   292.401   -6.052    1.021x   OK
orangepi  Qwen3-4B     512  1  249.352   269.525   +20.173   0.925x   REG
orangepi  Qwen3-4B     512  2  805.201   740.569   -64.632   1.087x   OK
orangepi  Qwen3-4B     512  4  826.331   843.580   +17.249   0.980x   REG
orangepi  Qwen3-4B     512  6  1107.961  978.142   -129.819  1.133x   OK
orangepi  Qwen3-4B     512  8  1114.840  1010.031  -104.808  1.104x   OK
rhino     Llama3.2 3B  512  1  97.874    78.726    -19.148   1.243x   OK
rhino     Llama3.2 3B  512  2  177.129   150.505   -26.624   1.177x   OK
rhino     Llama3.2 3B  512  4  191.076   174.219   -16.857   1.097x   OK
rhino     Llama3.2 3B  512  6  250.149   241.620   -8.529    1.035x   OK
rhino     Llama3.2 3B  512  8  258.723   255.406   -3.318    1.013x   OK
rhino     MiniCPM5-1B  512  1  26.576    29.847    +3.272    0.890x   REG
rhino     MiniCPM5-1B  512  2  91.001    66.598    -24.403   1.366x   OK
rhino     MiniCPM5-1B  512  4  92.542    75.160    -17.382   1.231x   OK
rhino     MiniCPM5-1B  512  6  113.330   102.124   -11.206   1.110x   OK
rhino     MiniCPM5-1B  512  8  117.574   107.364   -10.210   1.095x   OK
rhino     Qwen3-4B     512  1  114.242   95.166    -19.075   1.200x   OK
rhino     Qwen3-4B     512  2  240.338   185.887   -54.451   1.293x   OK
rhino     Qwen3-4B     512  4  248.866   199.940   -48.927   1.245x   OK
rhino     Qwen3-4B     512  6  317.594   290.654   -26.939   1.093x   OK
rhino     Qwen3-4B     512  8  323.453   301.085   -22.368   1.074x   OK
```

No `failures.csv` was produced for:

```text
decode_ab_old_current_orangepi_20260703
decode_ab_new_mali_route_v2_orangepi_20260703
decode_ab_new_final_orangepi_rhino_20260703
```

Final interpretation:

```text
What is solved:
  1. Full historical DecodeK prepare is outside timed decode.
  2. Fallback-old shapes no longer request DecodeK prepare.
  3. Qwen Mali q4 is no longer routed to qtile.
  4. DecodeK storage is preallocated with PagedCache.
  5. q>1 has a real qtile kernel family available, but routing remains conservative.

What is not solved:
  1. q1 is still launch/fixed-overhead sensitive. Old identity can win.
  2. q>1 qtile is not yet universally faster on Mali.
  3. Fallback rows still show endpoint variance; a stronger interleaved A/B harness is needed.
  4. QKV still reads PagedCache value, so DecodeK only fixes QK.

Next implementation details:
  1. Add q1 attention-only old-vs-new profile with record queue enabled.
  2. For q1 DecodeK, reduce append/write overhead or make K load cooperative across GQA heads.
  3. For q>1, make qtile choose lane/q_tile by device+shape and keep q6 on q8 mask only if it wins.
  4. Split QK-only and QKV-only timing so V-side cost does not hide K-side wins.
  5. Add identity-slot V fast path for full-reuse when slot_table is identity.
  6. Promote route only after no-regression endpoint TPOT across OrangePi and Rhino.
```

## V6 Step 0 — Baseline rebuild + TPOT A/B confirmation (2026-07-03)

Plan: `/root/.claude/plans/refactored-munching-wren.md` (approved).
Workflow reference: `references/decode_opencl_workflow.md`.

Rebuilt the route-v2 OrangePi artifact from commit `ca9fd3e` (clean working tree
except doc edits — `decode_optimization_plan_v6.md`, `references/decode_opencl_workflow.md`,
`CLAUDE.md` pointer). `opencl_codegen.py` produced no `.cl` diffs (source already
committed). Build OK (only the pre-existing `perfetto_c.cc` `#pragma system_header`
warnings); `file` confirmed AArch64 `pic_server`/`libMNN.so`/`libMNN_CL.so`/`libpic_llm.so`;
rsynced to `orangepi@192.168.101.113`.

Profile smoke (`step0_profile_smoke_orangepi_20260703`, llama3.2-3b q1/q4, profile on):
2 ok rows, no `failures.csv`, **P0 validity scan clean** (no `decode_prepare_inside_decode=1`,
no `ERROR`/`target unavailable`/`async read failed`). Harness + P0 boundary confirmed.

Formal TPOT A/B (profile off, ctx512, q=1/2/4/6/8, 3 models, repeats=2/warm=1) —
reproduces the handoff same-window table exactly (same committed source, identical
binary). This is the Step 1+ reference:

```text
model         q    old_ms     new_ms    delta  old/new  status
Llama3.2 3B   1    235.14     232.31    -2.83   1.012   OK
Llama3.2 3B   2    408.69     351.29   -57.40   1.163   OK
Llama3.2 3B   4    373.78     388.52   +14.75   0.962   REG   <- Step 3 target
Llama3.2 3B   6    814.64     723.38   -91.26   1.126   OK
Llama3.2 3B   8    844.87     770.14   -74.74   1.097   OK
MiniCPM5-1B   1     61.68      60.24    -1.43   1.024   OK
MiniCPM5-1B   2    280.51     244.13   -36.38   1.149   OK
MiniCPM5-1B   4    240.75     285.44   +44.68   0.843   REG   <- Step 3 target
MiniCPM5-1B   6    285.71     270.16   -15.55   1.058   OK
MiniCPM5-1B   8    298.45     292.40    -6.05   1.021   OK
Qwen3-4B      1    249.35     269.53   +20.17   0.925   REG   <- Step 1 target
Qwen3-4B      2    805.20     740.57   -64.63   1.087   OK
Qwen3-4B      4    826.33     843.58   +17.25   0.980   REG   <- Step 3 target
Qwen3-4B      6   1107.96     978.14  -129.82   1.133   OK
Qwen3-4B      8   1114.84    1010.03  -104.81   1.104   OK
```

Baseline runs (reused, same committed source):
- old: `decode_ab_old_current_orangepi_20260703` (`MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=0`)
- new: `decode_ab_new_mali_route_v2_orangepi_20260703` (`=1`, route-v2 default)

Conclusion: Step 0 gate passed. Reference established. Next: Step 1 (q=1 GQA-cooperative
transposed-K kernel) targets Qwen q1 (+20.17ms REG). Step 3 (qtile K_TILE + V-side
parallel) targets Llama q4, MiniCPM q4, Qwen q4.

## V6 Step 1 — q=1 GQA-cooperative transposed-K kernel (2026-07-03)

### What was implemented

New OpenCL kernel family `decode_causal_attention_hd128_transposed_k_fused_kv_gqa_row{32,64,128}`
in `paged_decode_attention_buf.cl`: ports `DEFINE_DECODE_CAUSAL_IDENTITY_FUSED_KV_GQA_HD128` onto
the transposed-K (decodeKey) data path. One workgroup per `(b, kvh)` shares the decodeKey K stream
across the whole GQA group (`NUMHEAD_GROUP_SIZE` query heads), fuses the current-token append
(key_cache + value_cache + decode_key), and keeps independent online-softmax per group head. Host:
`ensureDecodeTransposedKKernel` builds it (guarded `groupSize > 1`), new
`runDecodeCausalAttentionHD128TransposedKFusedKVGQA` dispatch with `gws z = mKvNumHead * mBatch`
(not `mNumHead*mBatch`), routed in `onExecute` inside `if (decodeTransposedKQ1NeedsDecodeKey)`
before the non-GQA fused path, gated on `_decodeGqaFusedKVEnabled()` (env
`MNN_PAGED_ATTENTION_DECODE_GQA_FUSED=1`, default off). Build OK, codegen OK, AArch64 confirmed.

### Rank-capture mechanism (recorded for future sessions)

`runDecodeAttentionRankCaptureOpenCL` (`PagedAttentionBufExecution.cpp:7491`) is the PIC
decode-repair attribution path. When `mMeta->needsPicDecodeAttentionRankCapture(layerIndex)` is
true, after the attention kernel it runs `mDecodeAttentionRankScoreKernelHD128` to score the
current query against the PIC token span and write a top-M candidate list — this is what picks
which tokens get recompute-repaired on later steps. It is **separate** from the attention kernel
and reuses `mCache->key`/`value`/`slotTable`/`sparseQuery` (no second KV). The profile `rank_us`
field times it. Rank capture is why some layers show a non-zero `rank_us`; it does not change the
attention dispatch. The identity profile lines seen in the smoke are NOT rank-capture — they are
the non-fused identity decode path that fires when `ordinaryDecodeFusedKV` is false (see below).

### TPOT already excludes the first token (warmup is not the issue)

`pic_server.cpp:2011` computes `measuredDecodeTokens = max(1, completionTokens - 1)` and
`decode_tpot_ms = decode_us / 1000 / measuredDecodeTokens`. ArGeneration reuses the prefill logits
for the **first** generated token, so `decode_us` covers only tokens 2..N. The first decode step
(which can hit the identity path when `external_loaded_layers` is still being filled) is the TTFT
token and is excluded from TPOT. So TPOT is already a steady-state number; no harness change
needed for warmup exclusion. `--warm-repeats` additionally discards whole warm requests before the
measured ones.

### Lane (recorded for future sessions)

`lane` = `LANES` = the OpenCL workgroup size in dimension 0 (the K-reduction width). One workgroup
of `LANES` lanes processes one output row/head: each lane owns one K position in the current K
tile, computes one QK score, and the tile reduces across `LANES` lanes (tree reduction) for
softmax max/sum. Larger lane = wider K tile per step = fewer K-loop iterations but more
local-memory (`score_tile[LANES]`, `reduce[LANES]`, or `[Q_TILE*LANES]`/`[GROUP*LANES]`) and
more register pressure. `_decodeHD128LaneWidth` picks 32/64/128 by `causalWorkPerRow` (kv_len
bucket); `MNN_PAGED_ATTENTION_DECODE_HD128_FORCE_LANE` overrides. For GQA, local mem scales with
`NUMHEAD_GROUP_SIZE*LANES`, so large groups + large lane blow up occupancy.

### Results (OrangePi, ctx512, profile off, repeats=2/warm=1)

```text
model         q    old      v2     gqa128  gqa/old  gate
Llama3.2 3B   1   235.14  232.31   204.14   1.152   OK   <- fixed
Llama3.2 3B   2   408.69  351.29   356.75   1.146   OK
Llama3.2 3B   4   373.78  388.52   341.53   1.094   OK
Llama3.2 3B   6   814.64  723.38   686.97   1.186   OK
Llama3.2 3B   8   844.87  770.14   783.33   1.079   OK
MiniCPM5-1B   1    61.68   60.24    75.57   0.816   REG
MiniCPM5-1B   2   280.51  244.13   232.50   1.206   OK
MiniCPM5-1B   4   240.75  285.44   237.59   1.013   OK
MiniCPM5-1B   6   285.71  270.16   285.68   1.000   OK
MiniCPM5-1B   8   298.45  292.40   323.37   0.923   REG
Qwen3-4B      1   249.35  269.53   262.39   0.950   REG   <- not fixed
Qwen3-4B      2   805.20  740.57   679.51   1.185   OK
Qwen3-4B      4   826.33  843.58   692.16   1.194   OK
Qwen3-4B      6  1107.96  978.14  1047.87   1.057   OK
Qwen3-4B      8  1114.84 1010.03  1093.61   1.019   OK
```

q1 lane sweep (GQA, force lane):
```text
MiniCPM q1: lane128=75.57  lane64=77.75  lane32=100.82  (old=61.68)  -> all REG
Qwen3  q1:  lane128=262.39 lane64=274.47 lane32=317.99  (old=249.35) -> all REG
```

P0 clean for every run (no `decode_prepare_inside_decode=1`, no ERROR/target unavailable/async
failed). GQA op confirmed firing for q1 (group>1): Llama=56, MiniCPM=48, Qwen=72 profile lines;
q>1 still uses qtile (unchanged).

### Conclusion

Step 1 is **partially effective**: GQA fixes Llama q1 (the largest q1 win, 1.152x) and is neutral
or better on most q>1, but it **regresses MiniCPM q1 (0.816x) and Qwen q1 (0.950x)** and the lane
sweep cannot recover them. Root cause:

- MiniCPM has only `kv_heads=2` -> GQA gws z = `kv_head*batch` = 2 workgroups, catastrophic
  occupancy on Mali; the K-share benefit cannot offset the lost parallelism.
- Qwen q1 (group=4, kv_heads=8, z=8) has enough workgroups but the per-workgroup local-memory
  pressure (`local_q[4*128]`, `score_tile[4*LANES]`, `reduce[4*LANES]`) and the serialized
  per-group softmax reduce outweigh the saved K loads at q1 (kv_len~512, tiny Q reuse).

This matches the handoff diagnosis: q1 has too little Q reuse; K-share only pays when K reload
cost dominates, which is true for q>1 (qtile) but not q1. The reverted q1 direct-read branch and
now this GQA q1 both confirm: **q1 needs a different axis** — likely V-side / append / launch-
record overhead reduction, not K-share.

### Decision

- Do **not** promote `MNN_PAGED_ATTENTION_DECODE_GQA_FUSED=1` to default (it regresses 2 of 3
  models at q1). Keep it env-gated as an explicit variant (SKILL-compliant).
- Keep the GQA kernel in source — it is the right building block for q>1 qtile×GQA (Step 4) where
  K-share across Q_TILE rows × GQA group is the real win.
- Step 1 q1 target (Qwen/MiniCPM q1) is **not met**. Per plan, q1 work pivots to V-side
  identity-slot fast path (Step 2) and overhead reduction, not more K-share variants.

Runs: `step1_gqa_profile_smoke_orangepi_20260703`, `step1_gqa_new_orangepi_20260703`,
`step1_gqa_minicpm_q1_lane{64,32}_orangepi_20260703`, `step1_gqa_qwen_q1_lane{64,32}_orangepi_20260703`.

## V6 Step 2' — slot_table removal smoke + prefill benchmark gate (2026-07-03)

### Implementation note

Removed identity `slot_table` logical->physical indirection from CPU, CUDA, and OpenCL
PagedAttention paths. The runtime mapping is now direct:

```text
logical slot k -> physical slot k
```

This matches the actual invariant for current PIC requests: request tokens occupy sequential
PagedCache slots, while persistent PIC cache source data uses separate source-slot kernel args
(`_picCacheSourceSlotBase` / source slot start) and not the request slot table.

Important bug found during smoke: the old `slot_table_version` was also part of CUDA/OpenCL async
persistent-cache read task keys. Removing it together with the table made the key too broad across
requests. This is not a token-ordering bug; PagedCache request tokens are still sequential. It is a
request-lifetime invalidation issue. The fix keeps slot removal but adds:

```text
PagedKVMeta::request_generation
```

`beginRequest()` increments it, and CUDA/OpenCL `externalLayerRequestKey()` includes it. This
keeps async task keys request-scoped without restoring logical->physical indirection.

### Export graph check

Static grep:

```bash
rg -n "slot_table|slotTable|request_base|physicalSlot|slot_table_host|slot_table_version" \
  transformers/pic_llm source/shape -S
```

Result: no matches. Exporter and shape graph do not expose `slot_table`, so no exporter graph
change is required for this removal.

### Build / sync

All builds used `JOBS=96` on the 192-core host.

```bash
JOBS=96 MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

JOBS=96 MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
  BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

JOBS=96 MNN_TARGET_DEVICE=aidlux_adreno_opencl BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

`file` checks passed for:

```text
.cache/output/mnn/artifacts/orangepi5plus/bin/pic_server
.cache/output/mnn/artifacts/orangepi5plus/lib/libMNN.so
.cache/output/mnn/artifacts/orangepi5plus/lib/libMNN_CL.so
.cache/output/mnn/artifacts/jetson/bin/pic_server
.cache/output/mnn/artifacts/jetson/lib/libMNN.so
.cache/output/mnn/artifacts/jetson/lib/libMNN_Cuda_Main.so
.cache/output/mnn/artifacts/aidlux_adreno_opencl/bin/pic_server
.cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN.so
.cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN_CL.so
```

Artifacts synced to fixed benchmark paths:

```bash
rsync -a --delete .cache/output/mnn/artifacts/jetson/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/

rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Static checks:

```bash
git diff --check
rg -n "slot_table|slotTable|slot_table_host|slot_table_version|physicalSlot|physicalSlots|request_base|_slotTable|syncSlotTable|decodeKeyReadyPrefixSlots|decodeKeySlotTableVersion|slotTableVersion|slotTableLength" \
  source/core/PagedKVMeta.hpp source/backend/cpu source/backend/cuda source/backend/opencl source/shape transformers/pic_llm -S
```

`git diff --check` passed. Residual slot-table grep returned no matches.

### Benchmark gate

Minimal prefill benchmark: `MiniCPM5-1B`, ctx512, `full-reuse`, profile off, only rows already in
`benchmark.csv`, no merge.

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_jetson.sh \
  --model-key minicpm5-1b --contexts 512 --modes full-reuse \
  --benchmark-csv benchmark.csv --only-benchmark-csv-rows \
  --run-id slot_table_request_generation_jetson_fullreuse_ctx512_20260703

bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_orangepi.sh \
  --model-key minicpm5-1b --contexts 512 --modes full-reuse \
  --benchmark-csv benchmark.csv --only-benchmark-csv-rows \
  --run-id slot_table_request_generation_orangepi_fullreuse_ctx512_20260703

bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_rhino.sh \
  --model-key minicpm5-1b --contexts 512 --modes full-reuse \
  --benchmark-csv benchmark.csv --only-benchmark-csv-rows \
  --run-id slot_table_request_generation_rhino_fullreuse_ctx512_20260703
```

Repeat for Jetson/OrangePi:

```bash
run_id=slot_table_request_generation_jetson_fullreuse_ctx512_repeat2_20260703
run_id=slot_table_request_generation_orangepi_fullreuse_ctx512_repeat2_20260703
```

Results vs `benchmark.csv` baseline:

```text
device     benchmark.csv baseline    runs after slot removal      best ratio
jetson     0.313210591 s           0.481185 / 0.472853 s          1.510x slower
orangepi   0.429257000 s           0.624368 / 0.721490 s          1.455x slower
rhino      0.289987000 s           FAIL warm: Empty reply / SSH timeout log
```

Successful Jetson/OrangePi measure responses were native full-reuse with cache hit:

```text
jetson:   status=200 prefill=0.481185 / 0.472853, execution_mode=native-full-reuse
orangepi: status=200 prefill=0.624368 / 0.721490, execution_mode=native-full-reuse
```

OrangePi diagnostic profile smoke:

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_orangepi.sh \
  --model-key minicpm5-1b --contexts 512 --modes full-reuse \
  --benchmark-csv benchmark.csv --only-benchmark-csv-rows \
  --server-env MNN_PAGED_ATTENTION_PROFILE=1 \
  --run-id slot_table_request_generation_orangepi_fullreuse_ctx512_profile_20260703
```

Key profile finding:

```text
OpenCLPagedAttention profile op=hydrate layer=0..23 tokens=498 kv_len=512
async_read=1 direct_segments=1 direct_tokens=498 fallback_tokens=0
```

So the request-generation fix did not break async direct hydrate. The performance regression is
not explained by prefetch falling back to sync read.

### Decision

Do not merge these summaries into `benchmark.csv`. Functional and build smoke passed, and export
graph needs no change, but the prefill latency gate failed on both formal devices:

```text
Jetson best:   1.51x slower than current benchmark.csv
OrangePi best: 1.455x slower than current benchmark.csv
Rhino:         no valid measurement due device/SSH failure
```

Next debugging should isolate pure slot-table removal from the other same-window decode/hydrate
experiments, then profile full-reuse by component (`hydrate`, suffix attention, graph dense) on
Jetson and OrangePi. The current result supports the user's semantic point: request tokens are
sequential in PagedCache, so slot removal should be correct. The unresolved issue is performance
regression, not a need to restore `slot_table`.

### Step 2' follow-up: full-reuse root-cause attribution

The goal of this follow-up is root-cause attribution, not switching `full-reuse` to a different
fast path. Slow `full-reuse` is the correct canary because every PIC mode pays its common floor:
persistent PIC cache source hydrate, current-request PagedCache write/read, suffix/prompt graph
execution, and backend PagedAttention. If `full-reuse` is high, `cacheblend` / `epic` / `kvshare`
inherit that floor before adding scoring and sparse recompute.

Formal-profile evidence splits by backend:

```text
MiniCPM5-1B ctx512 full-reuse:
Jetson legacy native-full-reuse:      0.481185 / 0.472853 s
Jetson graph-boundary evidence run:   0.081381 s, native-full-reuse-graph-boundary
OrangePi legacy native-full-reuse:    0.624368 / 0.721490 s
OrangePi graph-boundary evidence run: 0.976856 s
```

Jetson root cause:

- Current `pic_server.cpp` only enables graph-boundary prefill for `cacheblend` / `delta-v` /
  `epic`, not `full-reuse`.
- Therefore `full-reuse` falls through the legacy split path:
  `prefill_pic_prefix` -> `appendExternalPagedKV` -> `prefill_pic_suffix`.
- Request profile proves the measured request spends two large chunks in tiny LLM forwards:

```text
prefill_pic_prefix: forward_raw_end cost_ms=232.299 seq_len=7
prefill_pic_suffix: forward_raw_end cost_ms=238.490 seq_len=7
```

Warm in the same log showed the same shape with `296.708ms + 241.372ms`. This is not a
PagedCache lookup problem. CUDA PagedAttention profile sums for the same shape:

```text
hydrate:   24 calls, 14.668 ms
attention: 47 calls, 13.176 ms
```

The graph-op profile also points away from PagedAttention as the dominant Jetson cost:

```text
total graph profile: 146.132 ms
Convolution:         72.144 ms
PicSparseAttention:  26.860 ms
Raster:              18.881 ms
BinaryOp:             9.708 ms
While:                7.763 ms
PagedAttention:       1.186 ms
PicScoreAttention:    1.527 ms
```

Interpretation: Jetson's current full-reuse latency is dominated by low-M dense graph execution
and per-forward graph/runtime overhead from the legacy split scheduling. The existing
`native-full-reuse-graph-boundary` 0.081381s run is useful evidence because it keeps
`recompute_token_count=0` and `reuse_token_count=498` while avoiding the two tiny legacy forwards.
It proves continuous PagedCache and slot-table removal are not inherently the Jetson bottleneck.
It is evidence for the root cause, not the accepted fix target for this step.

OrangePi root cause is different:

- Legacy full-reuse is still slow even before considering scoring.
- Non-detail OpenCL profile already accounts for most of the latency inside PagedAttention paths:

```text
hydrate:   24 calls, 201.652 ms
attention: 48 calls, 417.017 ms
```

- Detail profile is not a formal latency number because it inserts queue finishes, but its
  attribution is clear:

```text
attention total: 729.184 ms
rearrange:       280.428 ms
qkv:             236.573 ms
qk:              132.718 ms
pack:             33.636 ms
mask:             15.667 ms
softmax:          18.312 ms
```

Representative detail lines show two regimes:

```text
prefix tiny forward: query=7 kv_len=7, many layers spend ~12ms in rearrange
suffix against cache: query=7 kv_len=512, qk ~= 4.8-5.1ms and qkv ~= 9.1-9.5ms per layer
```

Interpretation: on OrangePi, the full-reuse floor is genuinely operator-side: hydrate plus
OpenCL prefill attention for small `query=7` and long `kv_len=512`, with `qkv`, `qk`, and
unfavorable rearrange overhead. The graph-boundary evidence run is slower on OrangePi
(`0.976856s`), so simply changing full-reuse scheduling is not sufficient there. The next
root-cause work should split OrangePi full-reuse into:

```text
1. persistent PIC cache source read/hydrate cost
2. suffix q=7 attention QK / softmax / QKV cost at kv_len=512
3. query/rearrange/pack overhead for q=7
4. dense graph overhead outside attention
```

This is the actionable conclusion: the rejected Step 2' benchmark is not explained by slot-table
removal. Jetson is mostly legacy split-forward dense/graph overhead; OrangePi is mostly OpenCL
PagedAttention/hydrate operator cost. Both are common PIC-floor costs, so optimizing
cacheblend/epic ratios before lowering this floor will hide the real bottleneck.

### Legacy split-forward path removal

The follow-up code cleanup removes the server-side legacy chat scheduler instead of treating
Jetson `full-reuse` as an isolated fast-path fix.

Removed runtime path:

```text
prefill_pic_prefix
appendExternalPagedKV
recomputeExternalPagedKV
prefill_pic_suffix
```

Removed public helpers:

```text
Llm::appendExternalPagedKV
Llm::recomputeExternalPagedKV
Llm::selectCacheBlendExternalPagedKV
PagedKVMeta::appendExternalSegments
```

The score-layer module loader used only by the removed non-graph-boundary cacheblend scorer was
also deleted:

```text
Llm::getCacheBlendScoreModule
Llm::runCacheBlendScorePrefill
mCacheBlendScoreModulePool
mCacheBlendScoreRuntimeManager
```

New server behavior:

```text
full-compute:
  unchanged full prompt execution

full-reuse:
  requires a graph-boundary PIC model with pic_recompute_budget
  execution_mode = native-full-reuse-graph-boundary
  selected PIC rows = empty
  recompute_token_count = 0

cacheblend / delta-v / epic:
  require the same graph-boundary model
  use exported score-layer boundary instead of layer-0 selected-token forward

non-graph-boundary PIC reuse/sparse request:
  fails early if it would need the removed split scheduler
  falls back only through explicit full-compute metadata when scoring/top-k is unavailable
```

This deliberately removes the accidental fallback where non-graph-boundary chat could still run
by stitching multiple forwards around a persistent PIC cache source. It also removes the stale
non-graph-boundary cacheblend score-prefill helper, which would run an old truncated scoring pass
only to fall back later. These fallbacks were the Jetson root cause and also hid common-floor costs
from every PIC mode. The remaining model contract is clearer: request-time reuse and sparse
recompute must be expressed through the exported graph boundary and the current request PagedCache.

Important semantic note: the current exported graph-boundary is a score-layer boundary. It is the
correct production path for cacheblend/epic/kvshare because rows are compacted only after the
score layer. For `full-reuse`, the selected PIC row set is empty, so no PIC rows continue past the
boundary; however the graph-boundary implementation still follows the exported boundary shape.
If strict "hydrate plus suffix only from layer 0" is required later, it should be implemented as a
separate graph contract or by teaching the graph-budget model to accept active-row logical
position/mask input from layer 0. The removed legacy split path should not be reintroduced for
that.

Jetson validation after syncing the rebuilt CUDA artifact:

```text
run_id: fullreuse_path_removed_jetson_ctx512_synced_20260703
model:  MiniCPM5-1B
ctx:    512
mode:   full-reuse
latency: 0.080379 s
```

The remote fixed artifact path was updated before the validation:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/bin/pic_server
/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/lib/libpic_llm.so
/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/lib/libMNN_Cuda_Main.so
```

Request metadata from `raw/jetson/ctx512_chat_full-reuse_0.json`:

```text
execution_mode: native-full-reuse-graph-boundary
graph_boundary: fixed_active_rows_full_reuse
native_sparse_recompute_scope: exported_graph_score_layer_boundary
score_pass: not_required_full_reuse_graph_boundary
recompute_token_count: 0
reuse_token_count: 498
```

Request profile from `raw/jetson/pic_server_ctx512.log` now uses the new single graph-boundary
stage:

```text
prefill_fixed_graph_external_pagedkv selected_count=0
forward_raw_end seq_len=512
build_execution_plan recompute_tokens=0 reuse_tokens=498
```

The same log has no `prefill_pic_prefix`, `prefill_pic_suffix`, `select_cacheblend_external`, or
`score_pool_*` fields, confirming that both the legacy split scheduler and the stale
non-graph-boundary score-prefill helper are out of the runtime path.

### Sparse Q row hardening after slot removal

The functional issue to guard after deleting slot-table indirection is not logical-to-physical
addressing. It is making sure sparse prefill changes Q-related lengths at the score-layer boundary:

```text
layer < score_layer_idx:
  full prompt rows, no sparse query

layer == score_layer_idx:
  input queryLen is full context length
  output/activeLen is selected-token length
  qRow = sparse_query[q] for score-layer full-Q/compact-output
  K/V write uses full prompt K/V before active indices are emitted

layer > score_layer_idx:
  queryLen == activeLen
  qRow = q because graph hidden/Q/K/V are already compact rows
  qLogical = sparse_query[q] only identifies the true logical position
```

Code hardening added:

```text
PagedKVMeta::beginSparseQuery:
  requires strictly increasing logical indices within [0, logical_length)

PagedKVMeta::validateSparseQueryRows(activeLen, queryLen, allowFullQueryRows):
  active index vector size must equal activeLen
  pic_active_count must equal activeLen
  queryLen > activeLen is rejected unless allowFullQueryRows is true
  every active logical index must be increasing and < logical_length
  when full-Q is allowed, every active logical index must also be < queryLen
```

CPU/CUDA/OpenCL call this validation twice:

```text
before sparse K/V write:
  allowFullQueryRows = false
  later sparse layers must already be compact-Q

after score-layer active-index emit:
  allowFullQueryRows = scoreAttention && queryLen > activeLen
  only PicScoreAttention can use full-Q/compact-output
```

The kernels then use the same rule across CPU/CUDA/OpenCL:

```text
qLogical = sparse_query[q]
qRow     = queryRowsAreFull ? qLogical : q
slot     = logical
```

This is the explicit guarantee that Q moves from full context rows to selected-token rows at the
score layer, while K/V is still read and written by the true logical position in the sequential
PagedCache.

### Current-artifact prefill regression check after deleting legacy split/suffix

Question: after deleting the identity `slot_table` indirection and deleting the legacy PIC
split/suffix chat path, does PIC prefill recompute show an obvious regression versus
`benchmark.csv` across algorithms?

Artifacts and runs:

```text
Jetson slot-removal sparse smoke:
  .cache/latency_budget_20260625/slot_table_smoke_jetson_minicpm5_ctx512_20260703/summary.csv

Jetson full-reuse after path deletion:
  .cache/latency_budget_20260625/fullreuse_path_removed_jetson_ctx512_synced_20260703/summary.csv

OrangePi old slot-removal smoke:
  .cache/latency_budget_20260625/slot_table_smoke_orangepi_minicpm5_ctx512_20260703/summary.csv

OrangePi current artifact after deleting legacy split/suffix:
  .cache/latency_budget_20260625/slot_table_removed_path_removed_orangepi_ctx512_20260703/summary.csv
```

The OrangePi artifact was rebuilt and synced before the current run. `libpic_llm.so` no longer
exports the deleted split/suffix helpers (`appendExternalPagedKV`,
`recomputeExternalPagedKV`, `runCacheBlendScorePrefill`, `getCacheBlendScoreModule`). The remaining
PIC helper symbols are graph-boundary helpers.

The current OrangePi server log has only the expected OpenCL cache update:

```text
Update cache to /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/runtime_cache/opencl/mnn_cachefile.bin
```

There were no matches for:

```text
ERROR
Cache invalid
target unavailable
async persistent PIC cache read failed
prefill_pic_prefix
prefill_pic_suffix
```

Current OrangePi raw metadata confirms graph-boundary native execution:

```text
full-reuse:
  execution_mode = native-full-reuse-graph-boundary
  recompute_token_count = 0

cacheblend:
  execution_mode = native-cacheblend-graph-boundary
  native_sparse_recompute_scope = exported_graph_score_layer_boundary
  graph_boundary = score_layer_pic_score_attention
  pic_recompute_score_layer_idx = 1

epic:
  execution_mode = native-epic-graph-boundary
  native_sparse_recompute_scope = exported_graph_score_layer_boundary
  pic_recompute_score_layer_idx = 1
```

Joined comparison versus `benchmark.csv`:

```text
Jetson slot-removal sparse smoke
mode                   budget  current_s  benchmark_s  ratio  delta
normal-full-recompute  full    0.448997   0.450046     0.998  -0.2%
full-reuse             0       0.442611   0.313211     1.413  +41.3%
cacheblend             0.05    0.112536   0.127653     0.882  -11.8%
cacheblend             0.10    0.114071   0.128897     0.885  -11.5%
cacheblend             0.20    0.151300   0.169279     0.894  -10.6%
cacheblend             0.30    0.186417   0.207742     0.897  -10.3%
cacheblend             0.40    0.234237   0.259385     0.903  -9.7%
cacheblend             0.50    0.274125   0.300547     0.912  -8.8%
epic                   0.05    0.111039   0.113568     0.978  -2.2%
epic                   0.10    0.107184   0.115133     0.931  -6.9%
epic                   0.20    0.142285   0.146945     0.968  -3.2%
epic                   0.30    0.175980   0.183102     0.961  -3.9%
epic                   0.40    0.223118   0.232526     0.960  -4.0%
epic                   0.50    0.272529   0.279597     0.975  -2.5%

Jetson full-reuse after legacy path deletion
mode                   budget  current_s  benchmark_s  ratio  delta
full-reuse             0       0.080379   0.313211     0.257  -74.3%

OrangePi current artifact after legacy path deletion
mode                   budget  current_s  benchmark_s  ratio  delta
normal-full-recompute  full    2.329689   2.087799     1.116  +11.6%
full-reuse             0       0.724233   0.429257     1.687  +68.7%
cacheblend             0.05    1.131507   0.622859     1.817  +81.7%
cacheblend             0.10    1.209630   0.788489     1.534  +53.4%
cacheblend             0.20    1.507391   1.122164     1.343  +34.3%
cacheblend             0.30    2.150900   1.268948     1.695  +69.5%
cacheblend             0.40    2.080987   1.494961     1.392  +39.2%
cacheblend             0.50    2.396816   1.712549     1.400  +40.0%
epic                   0.05    1.311376   0.617986     2.122  +112.2%
epic                   0.10    1.108858   0.731855     1.515  +51.5%
epic                   0.20    1.328631   1.060809     1.252  +25.2%
epic                   0.30    1.610699   1.218384     1.322  +32.2%
epic                   0.40    2.056601   1.440272     1.428  +42.8%
epic                   0.50    2.055337   1.655416     1.242  +24.2%
```

OrangePi current artifact versus the older slot-removal smoke is mixed rather than uniformly
slower:

```text
mode                   budget  current_s  old_s     ratio  delta
normal-full-recompute  full    2.329689   2.371690  0.982  -1.8%
full-reuse             0       0.724233   0.787570  0.920  -8.0%
cacheblend             0.05    1.131507   1.023714  1.105  +10.5%
cacheblend             0.10    1.209630   1.397316  0.866  -13.4%
cacheblend             0.20    1.507391   1.336417  1.128  +12.8%
cacheblend             0.30    2.150900   1.644752  1.308  +30.8%
cacheblend             0.40    2.080987   2.115182  0.984  -1.6%
cacheblend             0.50    2.396816   2.547934  0.941  -5.9%
epic                   0.05    1.311376   0.784736  1.671  +67.1%
epic                   0.10    1.108858   0.868082  1.277  +27.7%
epic                   0.20    1.328631   1.563216  0.850  -15.0%
epic                   0.30    1.610699   2.031200  0.793  -20.7%
epic                   0.40    2.056601   1.805454  1.139  +13.9%
epic                   0.50    2.055337   2.221829  0.925  -7.5%
```

Conclusion:

1. Jetson does not show a PIC recompute regression after slot-table deletion. The sparse
   algorithms are equal or faster than `benchmark.csv` (`cacheblend` 0.882x to 0.912x,
   `epic` 0.931x to 0.978x). The previous Jetson `full-reuse` regression was the removed
   legacy split/suffix routing, not slot-table deletion; the path-deleted run is 0.257x the
   benchmark latency.
2. OrangePi does show a current-artifact performance regression versus `benchmark.csv`, including
   `full-reuse`, `cacheblend`, and `epic`.
3. The OrangePi regression cannot be attributed solely to slot-table deletion from the available
   evidence. `normal-full-recompute` also regresses by 11.6%, and current-vs-old slot-removal
   results are mixed rather than a consistent slot-deletion penalty.
4. The likely bucket is OpenCL/operator/runtime/tuning drift around the full compute floor plus
   graph-boundary sparse kernels. Next attribution should profile the current OrangePi artifact
   with `MNN_PAGED_ATTENTION_PROFILE=1` and compare:
   full-prompt attention, score layer `score_flash_attention`, later `sparse_flash_attention`,
   hydrate, and dense graph compact-row work.

These results were not merged into `benchmark.csv`.

### OrangePi obvious-regression profile after path deletion

Targeted debug profile runs:

```text
regression_profile_orangepi_minicpm5_ctx512_fr_cb05_epic05_20260703
regression_profile_orangepi_minicpm5_ctx512_cb30_20260703
```

Both runs used:

```text
MNN_PAGED_ATTENTION_PROFILE=1
MNN_PAGED_ATTENTION_PROFILE_DETAIL=1
MNN_PIC_GRAPH_PROFILE=1
MNN_PIC_GRAPH_PROFILE_TOP=400
```

These latencies are debug/profile latencies only. Graph profile waits after ops, so they must not
be compared to `benchmark.csv` as formal latency. The purpose is attribution.

Formal current-artifact rows that motivated the profile:

```text
normal-full-recompute  2.329689s  vs benchmark 2.087799s  1.116x
full-reuse             0.724233s  vs benchmark 0.429257s  1.687x
cacheblend 0.05        1.131507s  vs benchmark 0.622859s  1.817x
cacheblend 0.30        2.150900s  vs benchmark 1.268948s  1.695x
epic 0.05              1.311376s  vs benchmark 0.617986s  2.122x
```

Profile measure breakdown:

```text
case              graph_total  Convolution  full PA  PicSparseAttention  score/topk  hydrate  active_rows
full-reuse 0      1502.4 ms     672.7 ms     415.7    117.1 ms            n/a        16.0 ms  14
cacheblend 0.05   2463.4 ms    1358.0 ms     327.0    308.2 ms            2.3 ms     27.4 ms  39
epic 0.05         1793.0 ms     883.1 ms     405.5    136.0 ms            n/a        16.3 ms  39
cacheblend 0.30   3521.3 ms    2069.2 ms     293.4    530.3 ms            3.2 ms     19.0 ms  164
```

More detailed measure sums:

```text
full-reuse:
  OpenCL attention:
    prefill_attention_fast_qk_softmax_qkv = 415.4 ms
    sparse_flash_attention               = 87.3 ms across 22 layers
    score_qsplit_attention               = 18.1 ms
    hydrate                              = 16.0 ms across 22 layers
  Graph:
    Convolution                          = 672.7 ms across 169 ops
    full-row Convolution                 = 251.9 ms across 10 ops
    compact-row Convolution              = 387.7 ms across 158 ops

cacheblend 0.05:
  OpenCL attention:
    prefill_attention_fast_qk_softmax_qkv = 326.8 ms
    sparse_flash_attention               = 251.5 ms across 22 layers
    cacheblend_score                     = 2.3 ms
    hydrate                              = 27.4 ms across 22 layers
  Graph:
    Convolution                          = 1358.0 ms across 169 ops
    full-row Convolution                 = 115.3 ms across 10 ops
    compact-row Convolution              = 1201.0 ms across 158 ops
  Sparse flash:
    flash_us                             = 201.9 ms
    qk_active_tiles / qk_rect_tiles      = 15466 / 20064 = 0.771

epic 0.05:
  OpenCL attention:
    prefill_attention_fast_qk_softmax_qkv = 405.3 ms
    sparse_flash_attention               = 104.7 ms across 22 layers
    hydrate                              = 16.3 ms across 22 layers
  Graph:
    Convolution                          = 883.1 ms across 169 ops
    full-row Convolution                 = 117.3 ms across 10 ops
    compact-row Convolution              = 739.2 ms across 158 ops
  Sparse flash:
    flash_us                             = 76.8 ms
    qk_active_tiles / qk_rect_tiles      = 6424 / 7040 = 0.912

cacheblend 0.30:
  OpenCL attention:
    prefill_attention_fast_qk_softmax_qkv = 293.2 ms
    sparse_flash_attention               = 496.1 ms across 22 layers
    score_qsplit_attention               = 24.0 ms
    cacheblend_score                     = 3.2 ms
    hydrate                              = 19.0 ms across 22 layers
  Graph:
    Convolution                          = 2069.2 ms across 169 ops
    full-row Convolution                 = 107.5 ms across 10 ops
    compact-row Convolution              = 1935.1 ms across 158 ops
  Sparse flash:
    flash_us                             = 452.5 ms
    qk_active_tiles / qk_rect_tiles      = 62128 / 71522 = 0.869
```

Interpretation:

1. The obvious OrangePi regressions are not explained by persistent PIC cache hydrate. Hydrate is
   tens of milliseconds in debug profile, not the dominant term.
2. They are not explained by cacheblend score/top-k either. `cacheblend_score` is `2-3 ms` in
   these ctx512 profiles.
3. `full-reuse` is still expensive because the current graph-boundary implementation calls
   `prefill(full_prompt_token_ids)`. With `score_layer_idx=1`, layer 0 is a full prompt layer and
   score layer Q/K/V are still full rows. Active rows only take effect at the score-layer boundary.
   This is one forward, but it has a full-compute region before the boundary and a compact region
   after it. It is not a slot-table cost.
4. `cacheblend` / `epic` inherit the same full-compute floor before the boundary. After the
   boundary, OrangePi is dominated by compact-row `Convolution` / MLP plus `PicSparseAttention`.
5. Cacheblend sparse attention is worse than epic at the same selected count because selected
   logical positions are scattered. The profile confirms larger effective sparse QK work:
   `cacheblend 0.05 active/rect = 0.771` with `20064` rect tiles, while `epic 0.05` has only
   `7040` rect tiles because its selected PIC rows are prefix-contiguous.

Source correspondence:

```text
transformers/pic_llm/engine/src/llm.cpp:
  prefillFixedGraphExternalPagedKV(...)
    paged->external_hydrate_start_layer_idx = score_layer_idx + 1
    paged->beginPicGraphActivePlan(...)
    prefill(full_prompt_token_ids)

source/core/PagedKVMeta.hpp:
  graphActiveBudget(seqLen)
    active = seqLen - pic_token_count + selected_local_indices.size()
```

For MiniCPM5-1B ctx512, `pic_token_count=498`, so:

```text
full-reuse active rows: 512 - 498 + 0   = 14
5% active rows:         512 - 498 + 25  = 39
30% active rows:        512 - 498 + 150 = 164
```

Next actions:

1. Decide whether `full-reuse` should get a separate true hydrate+suffix graph contract. The
   deleted split/suffix scheduler should not return, but the current score-layer graph-boundary
   contract is inherently too heavy for ideal full-reuse because it still computes layer 0 full
   prompt rows.
2. Profile and fix OpenCL compact-row `Convolution` / MLP routing for active row sizes 14, 39, and
   164. This is currently the largest post-boundary cost.
3. For cacheblend, optimize sparse flash around selected logical distribution, not score/top-k.
   Low ratio still has large K work because selected rows are scattered across the PIC span.
4. Treat the 11.6% normal-full-recompute regression as a separate OpenCL normal baseline drift.
   It is not caused by slot-table deletion and should be checked with normal `llm_bench` / runtime
   cache / artifact A/B if this becomes the blocking regression.

## 2026-07-03 Full-Reuse Active Non-PIC Resolution

Previous section identified the right root cause: current `full-reuse` was not ideal
hydrate+suffix-only because it was routed through the score-layer graph-boundary contract. That
forced one full-prompt forward until `score_layer_idx=1`, so layer0 and score-layer inputs were
still full rows before compaction. The fix is not to restore the removed split/suffix scheduler.
Instead, `full-reuse` now has a dedicated active-non-PIC prefill path:

```text
logical prompt length = 512
PIC span              = [7, 505) = 498 tokens
active rows           = prelude [0, 7) + suffix [505, 512) = 14 tokens
```

Implementation:

- `prefillFullReuseExternalPagedKV(...)` builds `activeLogicalIndices` and `activeTokenIds` for
  non-PIC rows only.
- Persistent PIC cache source segments are bound with `logicalLength=fullPromptLen`; no scratch
  `.k/.v` is created.
- `PagedKVMeta::beginSparseQuery(activeLogicalIndices, 0)` makes every PagedAttention layer use
  real logical positions while the graph input has only active rows.
- `sparse_query_force_plain_attention` forces CPU/CUDA/OpenCL PagedAttention to avoid
  `PicScoreAttention` / `PicSparseAttention` scoring behavior for this full-reuse request.
- For graph-boundary models with `pic_recompute_budget`, forced sparse query now creates a real
  `[active_rows, logical_length]` causal mask. The initial scalar-mask attempt failed because graph
  gather nodes saw a rank/shape mismatch.

The first active-row attempt failed in warm:

```text
Broad cast error, dim1 = 0, dim2 = 14
Compute Shape Error for BinaryOp523
forwardRaw outputs empty seq_len=14 add=14 all_seq=0 prompt=14 logical_length=512
```

Root cause: the graph-boundary export still gathers `attention_mask` after the score-layer-shaped
PagedAttention op. A rank-0 scalar mask is valid for plain PagedAttention kernels, but invalid for
the exported graph's gather path. The mask fix was to materialize query rows by sparse logical
position:

```text
mask shape = [1, 1, active_rows, logical_length]
mask[i, j] = -inf when j > sparseLogicalIndex(i)
```

Build and sync:

```bash
JOBS=$(( ($(nproc) + 1) / 2 )) \
MNN_TARGET_DEVICE=orangepi5plus \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

Artifact check:

```text
pic_server    ELF 64-bit LSB executable, ARM aarch64
libMNN.so     ELF 64-bit LSB shared object, ARM aarch64
libMNN_CL.so  ELF 64-bit LSB shared object, ARM aarch64
libpic_llm.so ELF 64-bit LSB shared object, ARM aarch64
MNN_OPENCL=ON, MNN_CUDA=OFF, MNN_VULKAN=ON
```

Formal prefill-only debug row:

```bash
NO_PROXY=192.168.101.113,127.0.0.1,localhost \
no_proxy=192.168.101.113,127.0.0.1,localhost \
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_orangepi.sh \
  --model-key minicpm5-1b \
  --contexts 512 \
  --modes full-reuse \
  --run-id fullreuse_active_nonpic_mask_orangepi_ctx512_20260703 \
  --benchmark-csv benchmark.csv \
  --only-benchmark-csv-rows \
  --allow-subset-devices \
  --ssh-command-timeout-sec 960 \
  --remote-bench-timeout-sec 900 \
  --server-runtime-timeout-sec 1200
```

Result:

```text
summary.csv:
orangepi,MiniCPM5-1B,OpenCL,ctx512,full-reuse,0,0.406900s,1258.29 tok/s

benchmark.csv reference:
orangepi,MiniCPM5-1B,OpenCL,ctx512,full-reuse,0,0.429257s
```

Metadata:

```text
execution_mode      = native-full-reuse-hydrate-suffix
graph_boundary      = none_full_reuse_hydrate_suffix
graph_level_boundary= false
score_pass          = not_required_full_reuse_hydrate_suffix
recompute_token_count = 0
pic_recompute_logical_indices = []
```

Profile run:

```bash
NO_PROXY=192.168.101.113,127.0.0.1,localhost \
no_proxy=192.168.101.113,127.0.0.1,localhost \
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_orangepi.sh \
  --model-key minicpm5-1b \
  --contexts 512 \
  --modes full-reuse \
  --run-id fullreuse_active_nonpic_mask_orangepi_ctx512_profile_20260703 \
  --benchmark-csv benchmark.csv \
  --only-benchmark-csv-rows \
  --allow-subset-devices \
  --ssh-command-timeout-sec 960 \
  --remote-bench-timeout-sec 900 \
  --server-runtime-timeout-sec 1200 \
  --server-env MNN_PIC_REQUEST_PROFILE=1 \
  --server-env MNN_PAGED_ATTENTION_PROFILE=1
```

Profile result:

```text
summary.csv latency under profiling overhead: 0.598636s
forward_raw_on_forward: seq_len=14, outputs=1, cost=582.060 ms
forward_raw_total:      seq_len=14, outputs=1, cost=598.144 ms
full_reuse_prefill_active_non_pic:
  active_tokens=14 prelude_tokens=7 suffix_tokens=7 full_prompt_tokens=512

cacheblend_score count:      0
score_flash_attention count: 0
sparse_flash_attention count: 44 profile lines
hydrate count:               22 profile lines
full-prompt query=512 lines: 0
```

Manual decode smoke:

```text
request: same explicit tokens/span, full-reuse, max_tokens=1
status: 200
content: "<"
execution_mode: native-full-reuse-hydrate-suffix
graph_boundary: none_full_reuse_hydrate_suffix
recompute_token_count: 0
```

Conclusion:

`full-reuse` is no longer dragged through score-layer full-prompt graph-boundary compute for this
case. The remaining production cost is now active-row graph execution plus persistent PIC cache
source hydrate into current request PagedCache. The deleted legacy split/suffix path stays deleted;
this fix uses one active-row forward over non-PIC logical rows and keeps PagedCache as the KV
exchange layer.

### Cacheblend / epic check after full-reuse hydrate+suffix fix

Question: with the slot-table indirection removed and with `full-reuse` no longer routed through
the score-layer graph-boundary, do `cacheblend` and `epic` prefill still regress versus
`benchmark.csv`?

Current-artifact debug sweep:

```bash
NO_PROXY=192.168.101.113,127.0.0.1,localhost \
no_proxy=192.168.101.113,127.0.0.1,localhost \
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_orangepi.sh \
  --model-key minicpm5-1b \
  --contexts 512 \
  --modes cacheblend,epic \
  --run-id slotless_cb_epic_orangepi_ctx512_after_fullreusefix_20260703 \
  --benchmark-csv benchmark.csv \
  --only-benchmark-csv-rows \
  --allow-subset-devices \
  --ssh-command-timeout-sec 960 \
  --remote-bench-timeout-sec 900 \
  --server-runtime-timeout-sec 1200
```

Output:

```text
.cache/latency_budget_20260625/slotless_cb_epic_orangepi_ctx512_after_fullreusefix_20260703/summary.csv
```

Joined comparison versus `benchmark.csv`:

```text
mode          budget  current_s  benchmark_s  current/benchmark
cacheblend    0.05     1.269443    0.622859        2.038
cacheblend    0.10     1.114407    0.788489        1.413
cacheblend    0.20     1.356220    1.122164        1.209
cacheblend    0.30     1.878969    1.268948        1.481
cacheblend    0.40     2.060806    1.494961        1.379
cacheblend    0.50     2.321475    1.712549        1.356
epic          0.05     0.794243    0.617986        1.285
epic          0.10     1.068844    0.731855        1.460
epic          0.20     1.325192    1.060809        1.249
epic          0.30     1.789788    1.218384        1.469
epic          0.40     2.054780    1.440272        1.427
epic          0.50     2.109459    1.655416        1.274
```

Comparison against the earlier slotless sweep before the dedicated `full-reuse` fix:

```text
mode          budget  after_fix_s  old_slotless_s  after/old
cacheblend    0.05     1.269443      1.131507        1.122
cacheblend    0.10     1.114407      1.209630        0.921
cacheblend    0.20     1.356220      1.507391        0.900
cacheblend    0.30     1.878969      2.150900        0.874
cacheblend    0.40     2.060806      2.080987        0.990
cacheblend    0.50     2.321475      2.396816        0.969
epic          0.05     0.794243      1.311376        0.606
epic          0.10     1.068844      1.108858        0.964
epic          0.20     1.325192      1.328631        0.997
epic          0.30     1.789788      1.610699        1.111
epic          0.40     2.054780      2.056601        0.999
epic          0.50     2.109459      2.055337        1.026
```

Request metadata:

```text
completion_tokens=0
decode_latency_s=0
pic_recompute_score_layer_idx=1
doc span: prompt_start=7, token_count=498, full_prompt_token_count=512
cacheblend selected PIC counts: 25/50/100/150/200/249
epic selected PIC counts:       25/50/100/150/200/249
cacheblend active logical span at 30%: first=16 last=501, scattered rows
epic active logical span at 30%:       first=7  last=156, contiguous prefix rows
```

Run log scan:

```text
ERROR / Cache invalid / target unavailable / async persistent PIC cache read failed: none
old sparse_prefill path lines: none
sparse_flash_attention at layer 0 or layer 1: none
```

Existing attribution profile remains the useful breakdown for this code path:

```text
profile run:
  .cache/latency_budget_20260625/regression_profile_orangepi_minicpm5_ctx512_cb30_20260703

OpenCLPagedAttention profile, cacheblend 30%, profile overhead enabled:
  sparse_flash_attention total = 1120.978 ms, count=44
  prefill_attention_fast_qk_softmax_qkv = 723.870 ms, count=2
  score_qsplit_attention = 57.817 ms, count=2
  hydrate = 41.242 ms, count=44
  cacheblend_score = 5.828 ms, count=2

Graph profile type totals:
  Convolution        = 4124.155 ms, calls=338
    MLP Convolution  = 2817.794 ms
    attn projections = 1259.018 ms
  PicSparseAttention = 1210.386 ms, calls=44
  PagedAttention     = 724.297 ms, calls=2
  PicScoreAttention  = 64.944 ms, calls=2
```

Interpretation:

1. `cacheblend` / `epic` still show a real OrangePi current-artifact regression versus
   `benchmark.csv` after slot-table removal and after the dedicated full-reuse fix.
2. The regression is not explained by the deleted slot table itself. The current run is mixed
   against the earlier slotless sweep and has no old-path or layer-0 sparse execution signal.
3. It is also not a `full-reuse` hydrate+suffix problem anymore: fixed full-reuse is now
   `0.406900s` versus benchmark `0.429257s`, while cacheblend/epic still regress.
4. The intended graph-boundary floor remains for sparse algorithms: layer 0 is full-prompt
   compute, layer 1 is the score layer, and layer >= 2 is compact sparse. This is the correct
   cacheblend/epic contract, not the full-reuse bug.
5. The measured bottlenecks are OpenCL operators after and around that boundary: compact-row
   Convolution/MLP dominates the graph profile, followed by PicSparseAttention and the score-before
   full PagedAttention floor. `cacheblend_score` and hydrate are small.
6. Cacheblend is intrinsically harder than epic at the same selected count because selected rows are
   scattered deep into the PIC span; at 30% it reaches logical 501, so sparse attention still sees a
   near-full causal K range. Epic's prefix rows keep the causal range much smaller.

Next root-cause direction:

- Do not restore the removed slot table or the deleted legacy split/suffix path.
- First compare current normal-full-recompute OpenCL tuning/artifact drift against the benchmark
  normal row, because the earlier slotless run already showed normal `2.329689s` versus benchmark
  `2.087799s` (1.116x).
- Then profile current cacheblend/epic without changing semantics, focusing on compact-row
  `Convolution` / MLP tune keys and `PicSparseAttention` active logical distribution. For
  cacheblend-specific improvements that change row distribution, report a named algorithm variant
  rather than ordinary cacheblend.

These debug/profile results were not merged into `benchmark.csv`.
