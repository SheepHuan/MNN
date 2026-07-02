# Rhino Decode x0 P0/P1 专门路径上下文

## 环境

- device: Rhino Pi-X1 / Adreno OpenCL
- model: MiniCPM5-1B optimized PIC (`OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp`)
- mode: full-reuse
- repair_tokens: 0
- generated tokens: 32 for formal x0, 4 for x>0 smoke
- contexts: 512, 1024
- runtime cache: `/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/runtime_cache/opencl`
- artifact: `/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl`

## Implemented paths

### P0 default

Added a dedicated Adreno no-repair decode route for:

- `ordinaryDecodeCausal`
- `headDim == 128`
- `q == 1`
- `new_kv == 1`
- identity slot table
- no sparse query / no decode repair

The route calls:

```text
runDecodeCausalAttentionHD128Identity
  -> decode_causal_attention_hd128_identity_row32/row64
```

This removes slot-table lookup, sparse query indexing, repair rank capture, and decode_refine runtime work from x0 attention.

### P1 diagnostic only

Added a transposed-key mirror experiment:

```text
update_decode_transposed_key_hd128_identity
matmul_qk_decode
softmax_in1_buf
matmul_qkv_decode_b8
```

This is gated behind:

```text
MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_IDENTITY=1
```

It is not enabled by default because measured TPOT regressed.

## Results

Pre-change reference:

| path | ctx512 ms/token | ctx1024 ms/token |
| --- | ---: | ---: |
| normal LLM x0 | ~22.9 | ~23.6 |
| PIC x0 before P0/P1 optimized | ~64.06 | ~110.01 |

P1 transposed-key default experiment:

| path | ctx512 ms/token | ctx1024 ms/token |
| --- | ---: | ---: |
| PIC x0 transposed-key P1 | 100.110 | 131.486 |

P0 restored as default:

| path | ctx512 ms/token | ctx1024 ms/token |
| --- | ---: | ---: |
| PIC x0 P0 fused identity | 64.373 | 105.873 |

Repair path smoke after dispatch change:

| ctx | x | generated tokens | status |
| ---: | ---: | ---: | --- |
| 512 | 1 | 4 | ok |
| 512 | 3 | 4 | ok |
| 512 | 5 | 4 | ok |
| 512 | 7 | 4 | ok |

The 4-token smoke TPOT is not a formal metric; it only verifies no immediate repair-path breakage.

## Profile confirmation

Short profile run with P0 default:

```text
op=decode_causal_attention_hd128_identity
query=1
input_query=1
kv_len=513..515
lane=64
identity_slot=1
```

Parsed profile:

```text
count=62
avg_us=752.18
min_us=648
max_us=856
```

No `decode_transposed` line appeared in the default run.

## Interpretation

P0 proved that identity-slot no-repair routing is correct and removes the obvious PIC-specific decode repair/sparse/rank machinery from x0. It did not materially reduce TPOT. Therefore the 3x gap versus normal LLM is not primarily caused by slot table lookup or decode_refine.

P1 proved that copying PagedCache keys into a normal-style transposed mirror at decode time is worse on Adreno. The additional update kernel and split QK/softmax/QKV launches dominate any locality benefit.

Next useful work is to compare normal LLM decode graph/runtime against PIC x0 outside this attention kernel:

- graph-boundary extra ops around PagedAttention
- QKV projection / o_proj / MLP low-memory kernel selection differences
- runtime allocation/barrier/session state differences
- normal Attention decode key/value layout and whether transposed K must be produced at prefill/hydrate time, not rebuilt during decode
