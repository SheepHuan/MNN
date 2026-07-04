# Context

## Code changes

Touched:

```text
.codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py
.cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py
```

The PIC decode client already extracted `performance.request_wall_s`, but did not write it into the raw CSV. It now writes:

```text
completion_tokens
decode_measured_tokens
request_wall_s
request_wall_tpot_ms
```

`request_wall_tpot_ms` is computed as:

```text
request_wall_s * 1000 / (completion_tokens or decode_measured_tokens)
```

The decode experiment combiner now propagates:

```text
request_wall_s
wall_tpot_ms
```

into `decode_tpot_long.csv`, and emits corresponding PIC wall TPOT / wall extra-ms columns in `decode_tpot_wide.csv`.

## Validation

Syntax and skill validation:

```bash
python3 -m py_compile \
  .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py

python3 .codex/skills/.system/skill-creator/scripts/quick_validate.py \
  .codex/skills/mnn-pic-benchmark
```

Both passed.

## Smoke command

```bash
RUN_ID=decode_wall_smoke_orangepi_minicpm_ctx512_20260704_162059
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b \
  --contexts 512 \
  --pic-repair-tokens 0,1 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --skip-normal \
  --run-id "$RUN_ID"
```

## Smoke CSV check

Raw PIC rows:

```text
x0 completion_tokens=16 decode_measured_tokens=15 decode_tpot_ms=66.283933 request_wall_s=1.310290 request_wall_tpot_ms=81.893125 runtime=mnn_token_id_sparse_decode
x1 completion_tokens=16 decode_measured_tokens=15 decode_tpot_ms=99.582800 request_wall_s=1.809382 request_wall_tpot_ms=113.086375 runtime=mnn_token_id_sparse_decode
```

Long CSV rows:

```text
x0 tpot_ms=66.283933 request_wall_s=1.310290 wall_tpot_ms=81.893125
x1 tpot_ms=99.582800 request_wall_s=1.809382 wall_tpot_ms=113.086375
```

Wide CSV row:

```text
pic_dualgraph_x0_wall_tpot_ms=81.893125
pic_repair_x1_wall_tpot_ms=113.086375
pic_repair_x1_wall_extra_ms=31.193250
```

No `failures.csv` was produced for the smoke run. Remote P0 grep returned no matches for:

```text
decode_prepare_inside_decode=1
ERROR
target unavailable
async persistent PIC cache read failed
Cache invalid
```

## Formal OrangePi ctx512/1024

Command:

```bash
RUN_ID=decode_wall_formal_orangepi_ctx512_1024_20260704_162430
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
.cache/mnn-pic-benchmark/decode_experiment/decode_wall_formal_orangepi_ctx512_1024_20260704_162430/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_wall_formal_orangepi_ctx512_1024_20260704_162430/decode_tpot_wide.csv
```

`failures.csv` was not generated. PIC raw CSV validation:

```text
pic_decode_orangepi_llama3.2-3b.csv rows=10 statuses=[ok] runtimes=[mnn_token_id_sparse_decode] wall_missing=0
pic_decode_orangepi_minicpm5-1b.csv rows=10 statuses=[ok] runtimes=[mnn_token_id_sparse_decode] wall_missing=0
pic_decode_orangepi_qwen3-4b.csv rows=10 statuses=[ok] runtimes=[mnn_token_id_sparse_decode] wall_missing=0
```

The remote P0 grep returned no matches for:

```text
decode_prepare_inside_decode=1
ERROR
target unavailable
async persistent PIC cache read failed
Cache invalid
Empty reply from server
Connection refused
```

OrangePi did print normal first-use module clone warnings:

```text
Warning: module need new clone, cloning now.
```

These warnings occurred during the initial repair-token shape sequence and the same server then completed all warm and measure rows for ctx512 and ctx1024. This differs from the Rhino failure: Rhino stopped after the clone warning, returned an empty reply after about 585s, and later requests saw connection refused / device reboot.

### Decode TPOT Table

```text
device,model,ctx,true_normal_x0,pic_x0,x1,x3,x5,x7,x0_abs_gap,extra_x1,extra_x3,extra_x5,extra_x7
orangepi,Llama3.2 3B,512,121.751,148.821,208.780,215.298,372.313,376.035,27.069,59.959,66.477,223.492,227.214
orangepi,Llama3.2 3B,1024,125.632,163.133,227.098,238.410,413.843,424.841,37.501,63.964,75.277,250.709,261.708
orangepi,MiniCPM5-1B,512,46.729,70.088,100.141,107.824,159.588,158.913,23.359,30.054,37.737,89.500,88.825
orangepi,MiniCPM5-1B,1024,49.768,78.318,118.778,127.180,194.953,195.519,28.550,40.461,48.862,116.635,117.201
orangepi,Qwen3-4B,512,149.809,196.455,268.902,281.061,487.305,494.116,46.646,72.447,84.606,290.850,297.661
orangepi,Qwen3-4B,1024,155.111,215.387,299.755,314.360,539.230,558.090,60.276,84.368,98.973,323.844,342.703
```

### PIC Endpoint Wall TPOT

```text
device,model,ctx,pic_x0_wall,x1_wall,x3_wall,x5_wall,x7_wall,wall_extra_x1,wall_extra_x3,wall_extra_x5,wall_extra_x7
orangepi,Llama3.2 3B,512,172.659,228.504,233.327,380.312,383.676,55.845,60.668,207.653,211.017
orangepi,Llama3.2 3B,1024,189.188,249.429,259.944,424.345,434.579,60.241,70.756,235.157,245.391
orangepi,MiniCPM5-1B,512,85.775,116.470,121.116,169.709,169.122,30.695,35.341,83.934,83.347
orangepi,MiniCPM5-1B,1024,95.311,133.351,141.099,204.854,205.229,38.040,45.789,109.544,109.918
orangepi,Qwen3-4B,512,216.853,285.830,296.141,489.351,495.769,68.977,79.288,272.498,278.916
orangepi,Qwen3-4B,1024,240.310,319.777,333.530,544.222,561.440,79.467,93.220,303.912,321.130
```

Average endpoint overhead relative to `decode_tpot_ms`:

```text
x0: 21.316ms
x1: 18.318ms
x3: 16.838ms
x5: 7.594ms
x7: 7.050ms
```

Interpretation:

- The OrangePi run does not reproduce the Rhino native full-reuse warm hang.
- PIC x0 remains slower than true-normal x0 by 23-60ms depending on model/context.
- The endpoint wall gap is largest at small active rows, so next attribution should split logits/sample/readback and fixed request overhead in addition to attention and active-row graph compute.

## Greedy ArgMax / logits row slice

### Code change

Touched:

```text
transformers/pic_llm/engine/src/llm.cpp
```

Change:

- `forwardVecWithPicDecodeRepair()` now sets `mGenerateParam->validLogitStart` and
  `validLogitSize` to the last logits row produced by the sparse decode-repair
  forward. This makes the sampled row explicit for x0/x1/x3/x5/x7.
- `Llm::sample()` no longer implements `offset/size` by calling
  `logits->readMap<float>() + offset` before slicing. It now flattens logits and
  creates an Express `_Slice`, so the slice is part of the graph/runtime path.
- When `sampler_type=greedy`, `Llm::sample()` uses Express `_ArgMax(logits, -1)` and
  only reads back one `int` token. Non-greedy sampling continues through the existing
  `Sampler` pipeline.
- With `MNN_PIC_REQUEST_PROFILE=1`, this path logs `sample_device_argmax`.

This is not a new decode-repair family and does not change x routing. It only changes
the terminal logits/sample path shared by the same PIC dualgraph decode-repair family.

### Build validation

Local x64 compile check:

```text
cmake --build .cache/build/mnn/x64_cpu_pic --target pic_server --parallel <half-cpus>
```

Result:

- `llm.cpp` compiled and `libpic_llm.so` linked.
- `pic_server` final link failed on an existing unrelated x64 symbol:
  `MNN::___PicExtraSizeComputer__OpType_Extra__()`.

OrangePi target build:

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 \
  INSTALL_AFTER_BUILD=1 JOBS=<half-cpus> \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Result:

```text
pic_server:    ARM aarch64 ELF
libMNN.so:     ARM aarch64 ELF
libMNN_CL.so:  ARM aarch64 ELF
libpic_llm.so: ARM aarch64 ELF
```

The artifact was synced to:

```text
/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

### Profile smoke

Run:

```bash
RUN_ID=decode_argmax_smoke_orangepi_minicpm_ctx512_20260704_1718
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_REQUEST_PROFILE=1 MNN_PIC_DECODE_DEBUG=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b \
  --contexts 512 \
  --pic-repair-tokens 0,1 \
  --pic-max-tokens 4 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --skip-normal \
  --run-id "$RUN_ID"
```

Result:

```text
x0 decode_tpot_ms=67.175 wall_tpot_ms=90.182 runtime=mnn_token_id_sparse_decode
x1 decode_tpot_ms=95.666 wall_tpot_ms=111.594 runtime=mnn_token_id_sparse_decode
```

The remote log contains `sample_device_argmax`, e.g.:

```text
MNN_PIC_REQUEST_PROFILE stage=sample_device_argmax cost_ms=13.270 offset=0 size=130560 token=343 logits_size=130560
```

P0 grep found no `decode_prepare_inside_decode=1`, `ERROR`, `target unavailable`,
`async persistent PIC cache read failed`, `Cache invalid`, `Empty reply from server`,
`Connection refused`, `LLM greedy sampler failed`, or `LLM sampler failed`.

### PIC matrix vs previous formal

Run:

```bash
RUN_ID=decode_argmax_orangepi_ctx512_1024_pic_20260704_1726
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512,1024 \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 2 \
  --warm-repeats 1 \
  --skip-normal \
  --run-id "$RUN_ID"
```

No `failures.csv` was produced, and remote P0 grep was clean.

Compared to the previous formal PIC rows from
`decode_wall_formal_orangepi_ctx512_1024_20260704_162430`:

```text
x,avg_delta_decode_ms,avg_delta_wall_ms
0,-1.949,-2.033
1,-1.654,-1.687
3,-0.856,-0.829
5,-2.080,-1.895
7,-2.340,-2.227
overall,-1.776,-1.734
```

Latest decode TPOT table, using true-normal x0 from the previous formal baseline:

```text
model,ctx,true_normal_x0,new_pic_x0,new_x1,new_x3,new_x5,new_x7,new_x0_gap,extra_x1,extra_x3,extra_x5,extra_x7
Llama3.2 3B,512,121.751,147.353,205.622,210.874,370.070,373.092,25.601,58.270,63.521,222.718,225.740
Llama3.2 3B,1024,125.632,160.086,228.374,239.753,410.445,423.686,34.454,68.288,79.667,250.359,263.600
MiniCPM5-1B,512,46.729,66.407,97.722,109.848,159.009,158.991,19.678,31.315,43.441,92.602,92.584
MiniCPM5-1B,1024,49.768,80.563,119.312,128.658,196.664,195.033,30.795,38.750,48.096,116.101,114.470
Qwen3-4B,512,149.809,193.453,267.463,278.846,481.900,490.157,43.644,74.010,85.394,288.447,296.704
Qwen3-4B,1024,155.111,212.647,295.033,311.016,536.666,552.513,57.536,82.386,98.369,324.019,339.866
```

Latest endpoint wall TPOT table:

```text
model,ctx,pic_x0_wall,x1_wall,x3_wall,x5_wall,x7_wall,wall_extra_x1,wall_extra_x3,wall_extra_x5,wall_extra_x7
Llama3.2 3B,512,170.725,225.038,229.157,378.255,380.737,54.312,58.432,207.530,210.012
Llama3.2 3B,1024,186.017,250.575,261.431,420.934,433.414,64.558,75.414,234.917,247.397
MiniCPM5-1B,512,81.966,112.820,123.132,170.380,169.130,30.855,41.167,88.415,87.164
MiniCPM5-1B,1024,97.547,134.010,142.615,206.479,204.576,36.463,45.068,108.931,107.028
Qwen3-4B,512,213.616,285.639,293.554,483.776,492.162,72.023,79.938,270.160,278.547
Qwen3-4B,1024,238.026,315.156,330.297,541.603,556.433,77.130,92.272,303.577,318.407
```

Interpretation:

- This optimization is a measurable but small endpoint win: average wall TPOT improves
  by about `1.7 ms/token`.
- The win is broad enough to keep as default for greedy, but it does not remove the
  main PIC x0 gap to true-normal x0.
- Some individual MiniCPM 1024 and x3/x5 points move within noise or slightly regress.
  This confirms the next larger target remains graph-side fixed overhead, `lm_head`
  vocab production, Raster/elementwise/dense active-row cost, and possibly a more
  specialized OpenCL logits/top1 kernel rather than generic ArgMax alone.

## Decode-repair through PIC decode graph

### Code change

Touched:

```text
transformers/pic_llm/engine/src/llm.cpp
```

`forwardRaw()` now allows decode-repair to use the PIC package decode module when:

```text
mDecodeModule != nullptr
mPicDecodeRepair.enabled
inDecode
seqLen >= 1
seqLen == seqLenKey
!isAllLogists
```

The ordinary q=1 decode graph condition is kept separate and still requires
`!mPicDecodeRepair.enabled && seqLen == 1 && seqLenKey == 1`.

This is a runtime convergence of the PIC dualgraph decode-repair family, not a
renaming of x0 to normal. `x=0` is still the repair family's active-row-1 case, and
`x=n` remains active rows/token budget `n+1`. The route is considered valid only when
the PagedAttention metadata/logs still show sparse decode repair behavior.

The request profile now prints:

```text
pic_decode_repair_decode_graph=1
```

### Profile smoke

Run:

```bash
RUN_ID=decode_repair_decodegraph_smoke_orangepi_minicpm_ctx512_20260704_1746
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_REQUEST_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PIC_DECODE_DEBUG=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b \
  --contexts 512 \
  --pic-repair-tokens 0,1,3 \
  --pic-max-tokens 4 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --skip-normal \
  --run-id "$RUN_ID"
```

Smoke result:

```text
x0 tpot_ms=58.630 wall_tpot_ms=88.121
x1 tpot_ms=87.004 wall_tpot_ms=117.832
x3 tpot_ms=94.706 wall_tpot_ms=115.054
```

Route validation from remote logs:

```text
use_decode_graph=1
pic_decode_repair_decode_graph=1
ordinary_q1=0
repair=1
repair_qtile_route=1
decode_prepare_inside_decode=0
```

The same log also confirms `sample_device_argmax` still costs about 13ms when graph
profile is not forcing extra materialization, so terminal vocab/top1 remains a real
hidden cost.

### Formal PIC-only matrix

Run:

```bash
RUN_ID=decode_repair_decodegraph_orangepi_ctx512_1024_pic_20260704_1749
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512,1024 \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 2 \
  --warm-repeats 1 \
  --skip-normal \
  --run-id "$RUN_ID"
```

Outputs:

```text
.cache/mnn-pic-benchmark/decode_experiment/decode_repair_decodegraph_orangepi_ctx512_1024_pic_20260704_1749/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_repair_decodegraph_orangepi_ctx512_1024_pic_20260704_1749/decode_tpot_wide.csv
```

Validation:

- `failures.csv` was not generated.
- All 30 PIC rows are `status=ok`.
- Remote P0 grep returned no matches for:

```text
ERROR
target unavailable
async persistent PIC cache read failed
Cache invalid
decode_prepare_inside_decode=1
ordinary_q1=1
```

### x0 vs previous PIC and true normal

True-normal values are from the previous formal baseline
`decode_wall_formal_orangepi_ctx512_1024_20260704_162430`; this run only measured PIC
to isolate the route change.

```text
model,ctx,true_normal_x0_ms,old_pic_x0_ms,new_pic_x0_ms,delta_vs_old_ms,pct_vs_old,gap_vs_normal_ms,ratio_vs_normal
MiniCPM5-1B,512,46.729,66.407,50.479,-15.928,-24.0%,3.750,1.080
MiniCPM5-1B,1024,49.768,80.563,58.618,-21.945,-27.2%,8.850,1.178
Llama3.2 3B,512,121.751,147.353,121.958,-25.395,-17.2%,0.206,1.002
Llama3.2 3B,1024,125.632,160.086,136.278,-23.808,-14.9%,10.646,1.085
Qwen3-4B,512,149.809,193.453,162.153,-31.299,-16.2%,12.344,1.082
Qwen3-4B,1024,155.111,212.647,178.487,-34.160,-16.1%,23.376,1.151
```

### Full per-x table

```text
model,ctx,x,tpot_ms,extra_ms,old_tpot_ms,delta_vs_old_ms,pct_vs_old
MiniCPM5-1B,512,0,50.479,+0.000,66.407,-15.928,-24.0%
MiniCPM5-1B,512,1,77.696,+27.217,97.722,-20.026,-20.5%
MiniCPM5-1B,512,3,84.711,+34.232,109.848,-25.137,-22.9%
MiniCPM5-1B,512,5,136.792,+86.313,159.009,-22.216,-14.0%
MiniCPM5-1B,512,7,137.996,+87.517,158.991,-20.995,-13.2%
MiniCPM5-1B,1024,0,58.618,+0.000,80.563,-21.945,-27.2%
MiniCPM5-1B,1024,1,97.466,+38.848,119.312,-21.846,-18.3%
MiniCPM5-1B,1024,3,110.973,+52.355,128.658,-17.686,-13.7%
MiniCPM5-1B,1024,5,168.225,+109.607,196.664,-28.439,-14.5%
MiniCPM5-1B,1024,7,177.803,+119.185,195.033,-17.230,-8.8%
Llama3.2 3B,512,0,121.958,+0.000,147.353,-25.395,-17.2%
Llama3.2 3B,512,1,181.074,+59.116,205.622,-24.548,-11.9%
Llama3.2 3B,512,3,188.299,+66.341,210.874,-22.575,-10.7%
Llama3.2 3B,512,5,345.700,+223.742,370.070,-24.371,-6.6%
Llama3.2 3B,512,7,349.264,+227.306,373.092,-23.828,-6.4%
Llama3.2 3B,1024,0,136.278,+0.000,160.086,-23.808,-14.9%
Llama3.2 3B,1024,1,198.022,+61.743,228.374,-30.353,-13.3%
Llama3.2 3B,1024,3,213.695,+77.416,239.753,-26.058,-10.9%
Llama3.2 3B,1024,5,386.097,+249.818,410.445,-24.348,-5.9%
Llama3.2 3B,1024,7,395.485,+259.207,423.686,-28.201,-6.7%
Qwen3-4B,512,0,162.153,+0.000,193.453,-31.299,-16.2%
Qwen3-4B,512,1,234.124,+71.970,267.463,-33.340,-12.5%
Qwen3-4B,512,3,245.001,+82.848,278.846,-33.845,-12.1%
Qwen3-4B,512,5,450.606,+288.453,481.900,-31.294,-6.5%
Qwen3-4B,512,7,458.422,+296.268,490.157,-31.736,-6.5%
Qwen3-4B,1024,0,178.487,+0.000,212.647,-34.160,-16.1%
Qwen3-4B,1024,1,258.114,+79.627,295.033,-36.919,-12.5%
Qwen3-4B,1024,3,273.461,+94.974,311.016,-37.556,-12.1%
Qwen3-4B,1024,5,508.036,+329.549,536.666,-28.630,-5.3%
Qwen3-4B,1024,7,513.870,+335.383,552.513,-38.643,-7.0%
```

### Additional attribution profile

Run:

```text
profile_decodegraph_x0_hidden_orangepi_minicpm_ctx512_20260704_1758
```

This run used `MNN_PIC_GRAPH_PROFILE=1`, `MNN_PAGED_ATTENTION_PROFILE=1`,
`MNN_PAGED_ATTENTION_PROFILE_DETAIL=1`, and request profiling. It is only for
attribution, not TPOT.

Key route lines:

```text
forward_raw_select_module ... use_decode_graph=1 pic_decode_repair_decode_graph=1
PIC OpenCL PA decode hd128 route layer=0 ordinary_q1=0 repair=1 repair_qtile_route=1
decode_causal_attention_hd128_transposed_k_sparse_qtile ... decode_prepare_inside_decode=0
```

Caveat: graph profile includes the full-reuse suffix forward in the same request, so
its type summary still contains `PicScoreAttention` / `PicSparseAttention` from the
suffix step. The actual measured decode token uses the decode graph and repair qtile
route.

Interpretation:

- The major hidden x0 overhead before this change was not PagedKVMeta sync or logits
  readback; it was forcing decode-repair through the larger PIC main graph family.
- The remaining formal x0 gap is now small on MiniCPM/Llama and still visible on Qwen
  ctx1024. The next common optimization should target the terminal vocabulary path:
  fuse `lm_head + top1` or export a greedy top1 decode output to avoid materializing
  and reducing the full vocab tensor through generic ops.
- After terminal vocab/top1, the next repair-family work is fixed decode attention
  overhead: append/decodeKey write plus causal K scan. A q=1 active-row case should
  have a specialized append+attention fast path within the same repair qtile family,
  not a fallback to ordinary q1.

## QTile heuristic default validation

Code state:

- `MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_ROW` defaults to enabled, so x0 uses the
  repair-family active-rows=1 path.
- Mali qtile heuristic:
  - `attnLen >= 8 -> q_tile=2`
  - `attnLen >= 5 -> q_tile=4`
  - otherwise the previous `<=2 -> 2`, `<=4 -> 4`, else `8` rule, with x0
    overridden to `q1_row=1`.
- `MNN_PAGED_ATTENTION_DECODE_QTILE_FORCE` remains an explicit A/B override.

Route smoke:

```text
decode_repair_qtile_heuristic_route_smoke_orangepi_qwen_ctx1024_20260704_125548
```

Result:

```text
x5 active rows=6: q_tile=4, ordinary_q1=0, decode_prepare_inside_decode=0
x7 active rows=8: q_tile=2, ordinary_q1=0, decode_prepare_inside_decode=0
```

P0 grep found no `ERROR`, `target unavailable`, `async persistent PIC cache read
failed`, `Cache invalid`, `decode_prepare_inside_decode=1`, or `ordinary_q1=1`.

Formal PIC-only matrix:

```bash
RUN_ID=decode_repair_qtile_heuristic_default_orangepi_ctx512_1024_pic_20260704_125913
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512,1024 \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --skip-normal \
  --run-id "$RUN_ID"
```

No `failures.csv` was produced, and remote P0 grep was clean. Compared to
`decode_repair_q1row_default_orangepi_ctx512_1024_pic_20260704_122350`:

```text
model,ctx,dx0,dx1,dx3,dx5,dx7
Llama3.2 3B,512,+3.879,-0.840,-0.165,-18.073,-29.452
Llama3.2 3B,1024,-6.427,+0.406,+1.669,-29.823,-38.176
MiniCPM5-1B,512,+1.606,+0.983,-0.706,-26.557,-26.853
MiniCPM5-1B,1024,+2.728,-5.247,-3.714,-47.226,-45.008
Qwen3-4B,512,+17.374,+0.888,-5.549,-31.095,-29.721
Qwen3-4B,1024,+2.421,+10.291,+4.415,-44.227,-45.849
```

Family extra-ms after the qtile heuristic:

```text
model,ctx,true_normal_ref,pic_x0,gap,extra_x1,extra_x3,extra_x5,extra_x7
Llama3.2 3B,512,121.866,117.135,-4.731,58.968,71.340,208.319,204.665
Llama3.2 3B,1024,126.138,134.591,+8.453,64.074,80.328,222.028,225.158
MiniCPM5-1B,512,46.760,44.303,-2.457,32.455,38.467,63.760,67.645
MiniCPM5-1B,1024,49.809,45.723,-4.086,45.583,59.116,81.574,84.318
Qwen3-4B,512,149.916,170.259,+20.343,65.057,70.234,249.662,257.122
Qwen3-4B,1024,155.065,173.717,+18.651,89.509,98.529,280.250,293.905
```

The qtile change itself targets high active-row cases; x0 deltas are mostly run
variance and must be checked with a same-run true-normal baseline.

## Current Qwen x0 same-run baseline

Run:

```text
decode_x0_same_run_default_orangepi_qwen_ctx512_1024_20260704_130801
```

Scope: OrangePi, Qwen3-4B, ctx512/1024, true-normal x0 plus PIC dualgraph x0 only,
profile off, `repeats=2`, warm excluded for PIC.

Validity:

- no `failures.csv`
- P0 grep clean for `ERROR`, `target unavailable`, `async persistent PIC cache read
  failed`, `Cache invalid`, `decode_prepare_inside_decode=1`, `ordinary_q1=1`,
  `Empty reply from server`, and `Connection refused`
- no residual `pic_server`, `llm_bench`, or curl process on port 18132

Result:

```text
ctx,true_normal_x0_ms,pic_dualgraph_x0_ms,gap_ms,ratio_pic_over_normal,pic_wall_tpot_ms,sample_tpot_ms
512,149.854,148.146,-1.709,0.989,170.854,12.362
1024,154.648,175.235,+20.587,1.133,203.538,12.379
```

Interpretation:

- Qwen ctx512 x0 is no longer slower than true normal under the same-run baseline.
- Qwen ctx1024 still has a stable positive gap around 20.6 ms/token. This is the
  current best residual target for OrangePi x0.

## Current Qwen ctx1024 x0 attribution

Run:

```bash
RUN_ID=profile_current_default_qwen_ctx1024_x0_orangepi_20260704_131856
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=1000 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PIC_REQUEST_PROFILE=1 MNN_PIC_DECODE_DEBUG=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models qwen3-4b \
  --contexts 1024 \
  --pic-repair-tokens 0 \
  --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 0 \
  --skip-normal \
  --run-id "$RUN_ID"
```

This is profile/debug only. Its `tpot_ms=544.254` is inflated by graph/profile
synchronization and is not a formal latency result.

Route evidence:

```text
forward_raw_select_module ... use_decode_graph=1 pic_decode_repair_decode_graph=1
PIC decode graph route seq_len=1 add=1 all_seq=1025 gen_seq=1 inputs=4
PIC OpenCL PA decode hd128 route ... ordinary_q1=0 repair=1 repair_qtile_route=1
decode_causal_attention_hd128_transposed_k_sparse_qtile ... q_tile=1 q1_row=1 decode_prepare_inside_decode=0
```

Decode-specific PagedAttention q1-row profile:

```text
layers=36
sum_ms=22.034
mean_us_per_layer=612.1
max_layer=1
max_us=1190
variant=(query=1,lane=128,q_tile=1,q1_row=1)
```

Request-profile tail for the decode forward:

```text
forward_raw_on_forward cost_ms=544.138 seq_len=1 inputs=4 outputs=1
forward_raw_logits_defer_map cost_ms=0.000 seq_len=1 logits_deferred=1 logits_size=151936
sample_device_argmax cost_ms=0.613 offset=0 size=151936 token=0 logits_size=151936
```

Graph-profile type summary for the whole request, noting that it includes the
full-reuse suffix forward as well as the decode token:

```text
Convolution       351.532 ms calls=506
PicSparseAttention 211.809 ms calls=34
PagedAttention    174.139 ms calls=37
Raster             95.571 ms calls=1168
BinaryOp           33.093 ms calls=434
LayerNorm          23.299 ms calls=290
While              23.299 ms calls=324
UnaryOp            15.415 ms calls=220
PicScoreAttention  13.195 ms calls=1
Cast                2.878 ms calls=38
ArgMax              0.280 ms calls=2
```

Top graph op:

```text
/lm/lm_head/Linear total_ms=34.019 max_ms=21.436 calls=2 inputs=[1x2560x1x1] outputs=[1x151936x1x1]
```

Conclusions for the next engineering step:

1. The remaining Qwen ctx1024 x0 gap is not caused by full logits readback. The
   eager full-logits `readMap` has been removed from decode forward, and greedy
   sampling only reads back one argmax token.
2. The two largest generic targets are now:
   - terminal vocab production: avoid producing full `[1, vocab]` logits when greedy
     decode only needs top1, by exporting or fusing `lm_head + top1` in the PIC
     decode graph;
   - repair-family q1-row PagedAttention: reduce the 36-layer fixed scan/append
     cost without routing x0 to ordinary q=1.
3. For x>0, qtile heuristic is already a broad win, but the long-tail extra-ms still
   needs attention kernel work: shared K load across q_tile, record queue/fixed
   dispatch, or a better fused append+attention variant. The earlier fused-append
   A/B regressed and should remain explicit A/B only.
