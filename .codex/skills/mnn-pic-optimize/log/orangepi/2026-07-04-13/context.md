# Context

## Problem

The remaining OrangePi decode-repair issue was hidden fixed overhead in the PIC
dualgraph `x=0` path. `x=0` must remain the active-rows=1 degenerate member of the
same decode-repair family; it cannot gate to ordinary q=1 identity/normal decode.

Current attribution before this step:

- PIC decode-repair already uses the PIC `llm_decode.mnn` graph.
- Full logits readback was removed from `forwardRaw`; greedy uses device `_ArgMax`
  and reads one int.
- Qwen3-4B ctx1024 still showed a residual x0 gap of about 20 ms/token in an earlier
  same-run check.
- Decode-specific PA q1-row profile showed about 36 layers / 22 ms total.

## Code change

Touched:

```text
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp
```

Added:

- `_decodeRepairRecordQueueEnabled()`, default true, env off switch:
  `MNN_PAGED_ATTENTION_DECODE_REPAIR_RECORD_QUEUE=0`.
- `runDecodeCausalAttentionHD128TransposedKSparseRecord(...)`.
- Record state for sparse append + sparse/q1 attention in the OpenCL execution object.

The recorded command contains two kernels:

```text
append_sparse_decode_key_value_hd128
decode_causal_attention_hd128_transposed_k_sparse/q1/qtile
```

The record path is used only when:

- OpenCL record queue is enabled.
- Profile is off.
- Fused append is off.
- The layer is not capturing lagged-attention rank.
- The same repair sparse/q1 family route is already selected.

If any condition fails, execution falls back to the existing non-record append +
attention path. This keeps profile/debug attribution stable and avoids changing
the algorithm family.

## Build

Command:

```text
JOBS=<half-cpus> MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Result:

```text
pic_server:    ARM aarch64 ELF
libMNN.so:     ARM aarch64 ELF
libMNN_CL.so:  ARM aarch64 ELF
libpic_llm.so: ARM aarch64 ELF
```

Synced to:

```text
orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

`git diff --check` passed.

## Runs

### MiniCPM x0 smoke

Run:

```text
decode_repair_record_smoke_orangepi_minicpm_ctx512_20260704_133619
```

Result:

```text
benchmark_status=ok
runtime=mnn_token_id_sparse_decode
ctx=512 x0 decode_tpot_ms=43.209 wall_tpot_ms=72.293
failures.csv absent
P0 grep clean
```

### Qwen ctx1024 record off/on A/B

Record off:

```text
run_id=decode_repair_record_off_qwen_ctx1024_x0_20260704_133700
env=MNN_PAGED_ATTENTION_DECODE_REPAIR_RECORD_QUEUE=0
ctx=1024 x0 decode_tpot_ms=160.255 wall_tpot_ms=188.983
runtime=mnn_token_id_sparse_decode
failures.csv absent
P0 grep clean
```

Record on:

```text
run_id=decode_repair_record_on_qwen_ctx1024_x0_20260704_133901
ctx=1024 x0 decode_tpot_ms=156.809 wall_tpot_ms=185.561
runtime=mnn_token_id_sparse_decode
failures.csv absent
P0 grep clean
```

Delta:

```text
decode_tpot_ms: -3.446
wall_tpot_ms:   -3.422
```

### Qwen ctx1024 same-run true normal vs PIC x0

Run:

```text
decode_repair_record_on_same_run_qwen_ctx1024_x0_20260704_134101
```

Wide CSV:

```text
true_normal_x0_tpot_ms=155.0753006321117
pic_dualgraph_x0_tpot_ms=159.43306666666666
pic_dualgraph_x0_wall_tpot_ms=187.698875
```

Gap:

```text
PIC - normal = +4.358 ms/token
PIC / normal = 1.028x
```

This confirms `x=0` is no longer materially slower than true normal in the
previous Qwen ctx1024 problem case. It remains PIC dualgraph x0, not normal x0.

### MiniCPM x0/x1/x3 smoke

Run:

```text
decode_repair_record_smoke_orangepi_minicpm_ctx512_x013_20260704_134851
```

Rows:

```text
x0 decode_tpot_ms=43.834 wall_tpot_ms=82.745 runtime=mnn_token_id_sparse_decode
x1 decode_tpot_ms=75.723 wall_tpot_ms=106.088 runtime=mnn_token_id_sparse_decode
x3 decode_tpot_ms=81.210 wall_tpot_ms=101.516 runtime=mnn_token_id_sparse_decode
```

No `failures.csv`; P0 grep clean.

## Interpretation

The record-queue change attacks fixed OpenCL driver/dispatch overhead in the
repair-family PagedAttention path. It is not a logits optimization and not a
normal decode fallback. The win on Qwen ctx1024 x0 is about 3.4 ms/token over the
same artifact with record disabled. The remaining difference to true normal x0 is
about 4.4 ms/token in the same run.

Remaining generic optimization targets:

1. Terminal vocab production: greedy decode still materializes `[1, vocab]` before
   top1. A fused/exported `lm_head + top1` path should avoid the full logits tensor.
2. Repair q1-row attention kernel body: record removes host/driver overhead, but
   the q1 sparse attention still scans K/V through the repair family.
3. x>0 slope: keep tuning qtile/lane/shared K load and record coverage for layers
   that do not need rank capture.

## Formal OrangePi ctx512/1024 matrix

Run:

```bash
RUN_ID=decode_repair_record_formal_orangepi_ctx512_1024_20260704_135242
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512,1024 \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 2 \
  --warm-repeats 1 \
  --run-id "$RUN_ID"
```

Outputs:

```text
.cache/mnn-pic-benchmark/decode_experiment/decode_repair_record_formal_orangepi_ctx512_1024_20260704_135242/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_repair_record_formal_orangepi_ctx512_1024_20260704_135242/decode_tpot_wide.csv
```

Validation:

```text
decode_tpot_long.csv rows: 60 data rows
decode_tpot_wide.csv rows: 6 data rows
normal_decode_long.csv rows: 30 data rows
pic_decode rows: 10 rows/model
failures.csv absent
P0 grep clean:
  decode_prepare_inside_decode=1
  ERROR
  target unavailable
  async persistent PIC cache read failed
  Cache invalid
  Empty reply from server
  Connection refused
residual process check clean
```

Formal decode TPOT, measure rows only:

```text
model,ctx,true_normal_x0,pic_x0,x1,x3,x5,x7,pic_x0_minus_true_normal
MiniCPM5-1B,512,46.640,41.714,76.602,83.698,111.743,111.191,-4.926
MiniCPM5-1B,1024,49.722,45.936,95.994,107.326,132.482,129.632,-3.786
Llama3.2-3B,512,121.909,116.925,175.278,183.409,326.270,325.311,-4.984
Llama3.2-3B,1024,126.075,135.705,194.378,208.374,351.985,356.557,+9.630
Qwen3-4B,512,149.708,151.098,234.132,242.419,423.271,427.223,+1.390
Qwen3-4B,1024,155.142,173.639,256.586,272.607,458.451,461.541,+18.497
```

Same-family extra cost:

```text
model,ctx,extra_x1,extra_x3,extra_x5,extra_x7
MiniCPM5-1B,512,34.888,41.984,70.029,69.478
MiniCPM5-1B,1024,50.058,61.390,86.545,83.695
Llama3.2-3B,512,58.353,66.483,209.345,208.386
Llama3.2-3B,1024,58.673,72.669,216.280,220.851
Qwen3-4B,512,83.035,91.321,272.174,276.126
Qwen3-4B,1024,82.947,98.968,284.812,287.902
```

Wall TPOT shows a stable non-decode/sample tail:

```text
MiniCPM5 x0 wall_minus_decode_sample_tpot_ms: ctx512 7.320, ctx1024 9.350
Llama3.2 x0 wall_minus_decode_sample_tpot_ms: ctx512 15.755, ctx1024 20.127
Qwen3 x0 wall_minus_decode_sample_tpot_ms: ctx512 19.945, ctx1024 25.711
```

Interpretation:

- MiniCPM5 x0 and Llama3.2 ctx512 x0 are already faster than true normal x0.
- Qwen3 ctx512 x0 is near parity, only +1.390 ms/token.
- Llama3.2 ctx1024 and Qwen3 ctx1024 remain the real x0 gaps. They should be
  profiled with graph profile and PA detail, not fixed by routing x0 into the
  ordinary normal decode family.
- Qwen and Llama normal baseline had very slow warm/initialization phases before
  writing JSON/log output. This is a runner/autotune warm issue and should be
  handled like RhinoPi: keep warm outside formal reporting and prewarm/write
  OpenCL cache for target shapes before measure.
