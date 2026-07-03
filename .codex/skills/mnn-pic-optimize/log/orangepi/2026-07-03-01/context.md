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
