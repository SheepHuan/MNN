# OrangePi x0 decode gap re-assessment (current artifact)

Goal from the user: continue optimizing PIC dualgraph OpenCL decode /
decode-repair attention so x=0/1/3/5/7 approaches normal LLM decode, removing
fixed PIC overhead. Focus OrangePi/Mali first. The user's brief cited a
historical "x0 still ~20 ms slower than normal" split as append ~10 ms +
attention scan ~17 ms + rank ~0.1-0.2 ms.

## Headline finding

The ~20 ms x0 gap is already largely closed on the current OrangePi artifact
(commit `38ba6851` + the 07-04-16 decode-graph route + qtile heuristic + Mali
direct C4 GEMV). On the same artifact, same runtime cache, profile off, ctx1024:

```text
model         normal_x0   pic_x0    gap
MiniCPM5-1B    49.96      43.09    -6.87   (PIC FASTER)
Qwen3-4B      154.86     157.32    +2.46
Llama3.2-3B   125.64      (harness failure, rerun pending)
```

The historical "20 ms" number is stale on this artifact. x0 is no longer
consistently +20 ms vs true normal. Remaining Qwen ctx1024 gap is ~2.5 ms.

## Profile P0 validation (current artifact, Qwen3-4B ctx1024 x0/x1/x3/x5/x7)

```text
use_decode_graph=1
pic_decode_repair_decode_graph=1
ordinary_q1=0
repair_qtile_route=1
q1_row=1 lane=128 q_tile=1            (x0)
q1_row=0 lane=64  q_tile=2            (x1)
q1_row=0 lane=64  q_tile=4            (x3, x5)
q1_row=0 lane=64  q_tile=2            (x7)
decode_prepare_inside_decode=0
slot_identity=1 identity_v=1
record_queue=0   (Mali: QCOM cl_recording_qcom not supported -> never available)
no ERROR / target unavailable / async persistent PIC cache read failed / Cache invalid
```

x0 attention kernel: `decode_causal_attention_hd128_transposed_k_sparse_row128`
(`DEFINE_DECODE_CAUSAL_TRANSPOSED_K_SPARSE_HD128`, `paged_decode_attention_buf.cl:1014`).
Per-layer warm attention ~611 us. 36 layers -> ~22 ms attention scan. append ~0
(in fused prepare, `inside_decode=0`), rank ~0 (not armed for x0).

## Why the obvious levers are all already tried-and-failed

The user's brief and V6 plan both point at the same levers. Every one was A/B'd
and regressed on Mali and/or Adreno:

1. **fused append + attention single-kernel** (07-05-13): x0 +14.7 ms, x1 +17.6 ms
   on Qwen ctx1024. Code commented out (`#if 0`), `_decodeRepairFusedAppendEnabled`
   hardcoded false. Saved one launch but moved append into the wide attention
   workgroup shape -> register pressure + write-after-read fencing on Mali.

2. **lane32/64/128 force** (07-05-13): no stable x0 benefit (repeat=3 rejected).
   Kept as explicit A/B env only.

3. **slot-identity-split** (`MNN_PAGED_ATTENTION_DECODE_REPAIR_SLOT_IDENTITY_SPLIT`,
   `runDecodeCausalAttentionHD128SlotIdentitySplit`): skips decodeKey, reads
   key_cache directly. **Regressed** on Qwen ctx1024 (x0 243 ms vs default 162 ms).
   Reason: key_cache layout `[slot, batch, kv_head, head_dim]` gives token-to-token
   stride `H_kv*128` (strided gathers), whereas transposed decodeKey
   `[batch, kv_head, head_dim, max_len]` gives stride-1 vload4 along the sequence
   axis. decodeKey is NOT removable overhead — it is the same layout transform
   as normal decode's `rearrange_k` -> `past_key`. Both normal and PIC pay it.

4. **q1 GQA-shared fused kernel** (`fused_kv_gqa`, gws z = kv_head_num):
   - Adreno (07-05-00): MiniCPM 0.71-0.80x, Qwen 0.93-0.97x — regressed.
   - Mali (07-02-22): Qwen 1.96 -> 2.4-3.0 ms/layer — regressed.
   - Root cause: GQA group is serialized in a `for (gh=0; gh<NUMHEAD_GROUP_SIZE; ++gh)`
     loop inside one workgroup; workgroup count drops 4x/8x (MiniCPM only 2 kv_heads
     -> 2 workgroups -> occupancy collapse). `_decodeRepairQ1GqaEnabled` hardcoded
     false.

5. **q1 identity-fused-kv, keycache-qtile-v2, q1 GQA** (07-05-00 Adreno): all
   regressed, all hardcoded false in `PagedAttentionBufExecution.cpp:276-294`.

## Normal vs PIC x0 stage alignment

```text
true normal q1 (AttentionBufExecution):
  5 recorded kernels: rearrange_k, matmul_qk_decode, softmax_in1_buf,
                      rearrange_v, matmul_qkv_decode_b8
  K source: past_key [kv_head, head_dim, max_len] (transposed, stride-1 seq)
  V source: past_value [kv_head, max_len, head_dim] (stride-1 head_dim)
  GQA: positional division y/NUMHEAD_GROUP_SIZE, K/V read once per kv_head,
       L2 absorbs the group-4 reuse. 5 separate kernels each full-occupancy.
  Record queue: yes on Adreno, NO on Mali (cl_recording_qcom unsupported).

PIC x0 q1 (PagedAttentionBufExecution):
  2 kernels: append_sparse_decode_key_value_hd128 + decode_causal_attention_hd128_transposed_k_sparse_row128
  K source: decode_key [batch, kv_head, head_dim, max_len] (transposed, stride-1 seq)
  V source: value_cache [batch, kv_head, max_len, head_dim] (stride-1 head_dim)
  GQA: gws z = batch*num_head (one workgroup per query head) -> each query head
       workgroup reads full K stream independently. Same-GQA-group query heads
       do NOT share K reads (no cross-workgroup L2 guarantee). PMC ~2.17 MB/layer.
  Fused: QK+softmax+V in one kernel (avoids intermediate qk/softmax tensor).
  Record queue: NO on Mali.
```

The structural difference is NOT decodeKey (both pay the same transpose). It is:
(a) PIC has no GQA K/V reuse across query-head workgroups; normal does (via L2).
(b) PIC fuses QK+softmax+V -> V stage runs with only 32 of 128 lanes active
    (`if (lid < 32)`, `paged_decode_attention_buf.cl:1116,1140`); normal's V is a
    separate full-occupancy 2D kernel.

## The one untried lever: V-stage full-lane utilization

Every PIC x0/x>0 attention kernel gates the V stage with `if (lid < 32)` — only
32 of 128 lanes read V and accumulate (32 lanes * 4 floats = head_dim=128). The
other 96 lanes idle during V, which is the memory-bound stage. Normal decode's
`matmul_qkv_decode_b8` is a separate 2D kernel with full utilization.

This is the only structurally-novel lever in the design space (every existing
variant has the `if (lid < 32)` V gate). But it is a non-trivial kernel-internal
restructure: the fusion (QK+softmax+V in one workgroup) is what forces the V
stage into the QK workgroup shape. Splitting V across all 128 lanes needs either
per-element reduction or a sub-group tile reorganization.

Risk: the V-stage idle is a side effect of the fusion that SAVES the intermediate
qk/softmax tensor round-trip. Splitting V might regain V utilization but lose the
fusion benefit, net-neutral or negative (similar to how fused-append regressed).

## Decision this hour

Do NOT implement a new kernel this hour. The x0 gap is already ~2.5 ms (Qwen) /
PIC-faster (MiniCPM) on the current artifact; the user's "20 ms" premise is
stale. The remaining levers are all kernel-internal and high-risk (every prior
kernel-internal A/B regressed). Implementing a V-stage restructure or a
parallel-lane GQA variant would need a full A/B cycle and is not justified by a
2.5 ms gap that is within run-to-run noise on some cells.

This hour is recorded as: re-validated current state, confirmed all obvious
levers are tried-and-failed, localized the one untried lever (V-stage full-lane),
and declined to implement it without a stronger signal (the gap is too small to
justify the regression risk given the history).

See `context.md` for commands, CSV paths, and the per-model TPOT table.
