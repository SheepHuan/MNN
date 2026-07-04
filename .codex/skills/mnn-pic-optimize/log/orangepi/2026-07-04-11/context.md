# Context

## Build

Command:

```bash
JOBS=$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 )) \
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Checks:

```text
file .cache/output/mnn/artifacts/orangepi5plus/bin/pic_server
file .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN.so
file .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN_CL.so
file .cache/output/mnn/artifacts/orangepi5plus/lib/libpic_llm.so
git diff --check
```

All artifacts were aarch64; `git diff --check` was clean.

Synced to:

```text
orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

## Profile Smoke

Run:

```text
decode_repair_q1row_smoke_orangepi_minicpm_ctx512_20260704_110045
```

Env:

```text
MNN_PIC_REQUEST_PROFILE=1
MNN_PAGED_ATTENTION_PROFILE=1
MNN_PAGED_ATTENTION_PROFILE_DETAIL=1
MNN_PIC_DECODE_DEBUG=1
MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_ROW=1
```

P0 grep was clean:

```text
ERROR|target unavailable|async persistent PIC cache read failed|Cache invalid|decode_prepare_inside_decode=1|ordinary_q1=1
```

Routing facts:

- `pic_decode_repair_decode_graph=1` appeared for decode.
- x0/query=1 logged `q1_row=1`, `q_tile=1`, `decode_prepare_inside_decode=0`.
- x1/query=2 logged `q1_row=0`, so q1-row did not take over x>0.

Smoke latency is not formal because profile detail inserts queue finishes.

## Formal Q1-Row A/B

Run:

```text
decode_repair_q1row_orangepi_ctx512_1024_pic_20260704_110137
```

Env:

```text
MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_ROW=1
```

Command shape:

```text
devices=orangepi
models=minicpm5-1b,llama3.2-3b,qwen3-4b
contexts=512,1024
pic-repair-tokens=0,1,3,5,7
pic-max-tokens=16
pic-suffix-from-cache-tokens=1
repeats=2
warm-repeats=1
skip-normal
```

P0 grep and `failures.csv` were empty.

Comparison baseline:

```text
default decodegraph: decode_repair_decodegraph_orangepi_ctx512_1024_pic_20260704_1749
true normal n16:     decode_gate_orangepi_ctx512_1024_20260704_0001
```

### x0 vs default and true normal

```text
model,ctx,true_normal_n16,default_x0,q1row_x0,default_gap,q1row_gap,q1row_vs_default
Llama3.2 3B,512,121.866,121.958,116.587,+0.091,-5.279,-5.370
Llama3.2 3B,1024,126.138,136.278,135.954,+10.141,+9.816,-0.325
MiniCPM5-1B,512,46.760,50.479,41.033,+3.719,-5.727,-9.446
MiniCPM5-1B,1024,49.809,58.618,43.640,+8.809,-6.169,-14.978
Qwen3-4B,512,149.916,162.153,153.001,+12.238,+3.086,-9.152
Qwen3-4B,1024,155.065,178.487,171.800,+23.422,+16.734,-6.687
```

### Variant summary vs default

```text
variant,x,avg_delta_ms,wins
qtile1,0,-0.172,3/6
qtile1,1,-0.893,4/6
qtile1,3,-2.712,5/6
qtile1,5,-1.356,4/6
qtile1,7,-0.256,3/6
fusedappend,0,+0.393,4/6
fusedappend,1,+10.645,0/6
fusedappend,3,+15.532,0/6
fusedappend,5,+9.895,1/6
fusedappend,7,+14.415,0/6
q1row,0,-7.660,6/6
q1row,1,-1.336,6/6
q1row,3,-2.296,4/6
q1row,5,-3.827,5/6
q1row,7,-0.788,4/6
```

Only x0 is the intended q1-row target. x>0 movement is treated as run-to-run variance unless a separate profile proves otherwise.

## Hidden-Cost Attribution

Graph profile run:

```text
decode_repair_q1row_graphprofile_orangepi_qwen_ctx1024_x0_20260704_111249
```

This is attribution only. It used `MNN_PIC_GRAPH_PROFILE=1` and `MNN_PAGED_ATTENTION_PROFILE_DETAIL=1`.

Important interpretation:

- `force_prefill_forward=1` appears for full-reuse suffix/active prefill before decode. It is per request, not per decode token, so it mostly explains `request_wall_s` / `wall_tpot_ms`, not steady decode TPOT.
- The actual decode-repair forward selected the decode graph with `pic_decode_repair_decode_graph=1`.
- PagedAttention detail showed `q1_row=1`, `decode_prepare_inside_decode=0`, and no ordinary q1 path.

Light profile run:

```text
decode_repair_q1row_lightprofile_orangepi_qwen_ctx1024_x0_20260704_111647
```

Env:

```text
MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_ROW=1
MNN_PIC_DECODE_REPAIR_PROFILE=1
MNN_PIC_REQUEST_PROFILE=1
```

Result:

```text
Qwen3-4B ctx1024 x0: tpot_ms=159.632, wall_tpot_ms=188.549
```

Decode-repair profile showed:

- Stable `forward_raw_ms` after the first profiled token: roughly `158-160 ms`.
- build rows / embedding / mask / position / bookkeeping are all microsecond-scale and not the problem.
- `sample_device_argmax` was about `13.2 ms` per token for Qwen3-4B vocab. It is not part of `decode_tpot_ms`, but it is part of wall time.

Current conclusion:

- q1-row removes a real x0 fixed overhead, especially MiniCPM and Qwen512.
- Remaining x0 TPOT overhead is mostly inside decode graph execution, not C++ row construction.
- Remaining wall overhead must be split separately: suffix active prefill once per request, then sampler/argmax per token.
