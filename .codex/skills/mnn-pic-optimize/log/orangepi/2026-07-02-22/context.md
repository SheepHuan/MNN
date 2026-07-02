# OrangePi Qwen3-4B Decode GQA Single-Kernel A/B

## Code Change

Added an experimental grouped-GQA OpenCL decode attention path for `head_dim=128`:

```text
source/backend/opencl/execution/cl/paged_decode_attention_buf.cl
source/backend/opencl/execution/cl/paged_decode_attention_buf_mnn_cl.cpp
source/backend/opencl/execution/cl/opencl_source_map.hpp
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp
```

New profile op:

```text
decode_causal_attention_hd128_identity_fused_kv_gqa
```

Runtime gate:

```text
MNN_PAGED_ATTENTION_DECODE_GQA_FUSED=1
```

Default path remains:

```text
decode_causal_attention_hd128_identity_fused_kv
```

## Build And Sync

Build:

```bash
MNN_TARGET_DEVICE=orangepi5plus \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Artifact check:

```text
.cache/output/mnn/artifacts/orangepi5plus/bin/pic_server:
  ELF 64-bit LSB executable, ARM aarch64
.cache/output/mnn/artifacts/orangepi5plus/lib/libMNN_CL.so:
  ELF 64-bit LSB shared object, ARM aarch64
```

Sync:

```bash
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

## Test Setup

Device paths:

```text
artifact: /mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus
config:   /mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/Qwen__Qwen3-4B-pic-boundary/config_opencl_greedy.json
kv dir:   /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/shared_kv/qwen3-4b
runtime:  /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/runtime_cache/opencl
port:     remote 18132, local tunnel 19132
```

Client shape:

```text
model=Qwen3-4B
context=512
mode=full-reuse
repair_tokens=0
decode_selector=lagged_attention_hkvd
max_tokens=2
repeats=1
warm_repeats=0
```

`repair_tokens=0` is the no-repair decode baseline. It does not send `decode_refine`; `max_tokens=2` is used so the second generated token enters decode attention.

Client command template:

```bash
NO_PROXY=127.0.0.1,localhost,192.168.101.113 \
no_proxy=127.0.0.1,localhost,192.168.101.113 \
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://127.0.0.1:19132 \
  --device orangepi \
  --device-display OrangePi \
  --backend OpenCL \
  --frequency-profile profile \
  --model qwen3-4b \
  --model-name Qwen3-4B \
  --model-config /mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/Qwen__Qwen3-4B-pic-boundary/config_opencl_greedy.json \
  --mode full-reuse \
  --contexts 512 \
  --budgets 0.00 \
  --repair-tokens 0 \
  --decode-selectors lagged_attention_hkvd \
  --attention-layer-idx 1 \
  --max-tokens 2 \
  --repeats 1 \
  --warm-repeats 0 \
  --suffix-from-cache-tokens 1 \
  --force-text-cache \
  --reset-before-each \
  --no-update-cache \
  --require-exact-context \
  --timeout 900
```

## Results

CSV outputs:

```text
.cache/mnn-pic-benchmark/decode_gqa_ab/default_ctx512.csv
.cache/mnn-pic-benchmark/decode_gqa_ab/gqa_ctx512.csv
.cache/mnn-pic-benchmark/decode_gqa_ab/gqa_ctx512_warm2.csv
.cache/mnn-pic-benchmark/decode_gqa_ab/default_ctx512_clean.csv
.cache/mnn-pic-benchmark/decode_gqa_ab/gqa_ctx512_clean.csv
```

End-to-end decode TPOT rows:

```text
run,env,decode_tpot_ms,decode_tps,status
default_debug,default,1089.719,0.917668,ok
gqa_debug_first,MNN_PAGED_ATTENTION_DECODE_GQA_FUSED=1,1178.980,0.848191,ok
gqa_debug_warm2,MNN_PAGED_ATTENTION_DECODE_GQA_FUSED=1,580.023,1.724070,ok
default_clean,default,1002.544,0.997462,ok
gqa_clean,MNN_PAGED_ATTENTION_DECODE_GQA_FUSED=1,1169.826,0.854828,ok
```

Profile aggregation from landed decode attention lines:

```text
run,op,count,total_ms,mean_ms,min_ms,max_ms
default_debug,fused_kv,34,66.672,1.961,1.750,2.355
gqa_debug_first,fused_kv_gqa,33,100.327,3.040,2.171,24.293
gqa_debug_warm2_partial,fused_kv_gqa,31,83.826,2.704,0.970,30.526
default_clean_partial,fused_kv,25,47.646,1.906,1.560,2.117
gqa_clean_partial,fused_kv_gqa,22,57.069,2.594,2.181,3.451
```

The debug logs had interleaved stdout from decode debug/profile. The clean logs still showed truncated tail lines in `nohup` output, so the `count` column is not treated as exact layer count. The per-line landed kernel timing is still useful because it consistently shows the same direction.

Representative lines:

```text
default:
OpenCLPagedAttention profile op=decode_causal_attention_hd128_identity_fused_kv layer=20 query=1 kv_len=514 lane=128 us=1787
OpenCLPagedAttention profile op=decode_causal_attention_hd128_identity_fused_kv layer=24 query=1 kv_len=514 lane=128 us=1798

gqa:
OpenCLPagedAttention profile op=decode_causal_attention_hd128_identity_fused_kv_gqa layer=12 query=1 kv_len=514 lane=128 group_size=4 kv_heads=8 us=2392
OpenCLPagedAttention profile op=decode_causal_attention_hd128_identity_fused_kv_gqa layer=18 query=1 kv_len=514 lane=128 group_size=4 kv_heads=8 us=3451
```

Log scan:

```text
ERROR / CL_OUT_OF_RESOURCES / target unavailable / async persistent PIC cache read failed: 0
```

## Interpretation

The grouped kernel reduces workgroups from query heads to KV heads, but each workgroup serializes the GQA group (`group_size=4`) internally. On Mali this loses too much parallelism and does not offset the saved K/V reads or launch work. For Qwen3-4B `ctx512`, the old per-query-head fused-KV kernel is about `1.9-2.0ms/layer`, while the grouped implementation is usually `2.5-3.0ms/layer` after excluding obvious first-use outliers.

This means the current single-op GQA implementation is not the fix for `pic_x0` being slower than `normal_x0` on OrangePi/Rhino. It should stay behind the env gate.

Next viable directions:

1. Keep parallelism close to one workgroup per query head, but reduce duplicated K/V loads using local memory or subgroup cooperation across the GQA group.
2. Split a KV head into multiple cooperative workgroups over query-head group and K tiles, then reduce outputs without serializing all heads in one workgroup.
3. Profile dense Linear/Conv around decode again with graph profile, because end-to-end TPOT variance is larger than the attention-kernel delta and may still hide major non-attention cost.
