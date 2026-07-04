# Context

## Why decode-repair does not use ordinary q1 decode

`llm_decode.mnn` ordinary decode is a q=1 path for one new contiguous token. PIC decode-repair feeds `repair_rows + current_decode_row` into one forward. The repair rows are logical positions inside the persistent PIC document, not new trailing tokens. They must be bound with `PagedKVMeta::beginPicDecodeRecomputeRows()` so PagedAttention writes and reads the real sparse logical slots. Therefore x0 is the active-rows=1 degenerate decode-repair case, but x>0 cannot use the ordinary q1 decode graph without losing repair-row PagedCache updates.

## Profile finding before edit

Run:

```bash
RUN_ID=profile_orangepi_minicpm_ctx512_x0_x1_x5_20260704_153341
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1 MNN_PIC_DECODE_REPAIR_PROFILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models minicpm5-1b --contexts 512 \
  --pic-repair-tokens 0,1,5 --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 --repeats 1 --warm-repeats 1 \
  --skip-normal --run-id "$RUN_ID"
```

Warm profile rows:

```text
repair=0 sparse=1 total=96.295 forward=96.261 on_forward=83.072 logits_map=13.130 attn_total=16.111
repair=1 sparse=2 total=137.190 forward=137.146 on_forward=123.934 logits_map=13.149 attn_total=26.048
repair=5 sparse=6 total=184.626 forward=184.580 on_forward=171.375 logits_map=13.136 attn_total=46.262
```

Important correction: `seq_len=6 logits_size=130560` means the exported graph already returns one vocab row, not 6 rows. The row clipping is already done by `logitsLastIdx`; vocab columns cannot be clipped without changing sampler semantics.

## Edit

- `transformers/pic_llm/engine/src/llm.cpp`
  - For decode-repair forward only, skip eager `outputs[0]->readMap<float>()`.
  - Emit `forward_raw_logits_defer_map`.
- `transformers/pic_llm/engine/src/sampler.cpp`
  - Add null/empty logits guard before sampling, so deferred materialization failures become `INTERNAL_ERROR` instead of a null dereference.

No `.cl` files changed; `opencl_codegen.py` was not required.

## Build and sync

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

file .cache/output/mnn/artifacts/orangepi5plus/bin/pic_server \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN.so \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN_CL.so \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libpic_llm.so

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

The first build failed on `VARP != nullptr`; fixed to the existing `nullptr != VARP` style. The final build and artifact file checks passed.

## Profile smoke after edit

Run:

```bash
RUN_ID=profile_orangepi_defer_logits_minicpm_ctx512_x0_x1_x5_20260704_154745
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1 MNN_PIC_DECODE_REPAIR_PROFILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models minicpm5-1b --contexts 512 \
  --pic-repair-tokens 0,1,5 --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 --repeats 1 --warm-repeats 1 \
  --skip-normal --run-id "$RUN_ID"
```

Log check:

- `forward_raw_logits_defer_map` appeared for decode-repair q=1/2/6.
- P0 grep printed nothing for `decode_prepare_inside_decode=1`, `ERROR`, `target unavailable`, `async persistent PIC cache read failed`, `Cache invalid`.

Warm profile rows:

```text
repair=0 sparse=1 total=82.277 forward=82.240 on_forward=82.184 deferred=1 attn_total=16.136
repair=1 sparse=2 total=115.437 forward=115.394 on_forward=115.331 deferred=1 attn_total=26.435
repair=5 sparse=6 total=167.906 forward=167.857 on_forward=167.793 deferred=1 attn_total=46.265
```

## Formal run

```bash
RUN_ID=decode_defer_logits_orangepi_ctx512_1024_suffix1_20260704_154948
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512,1024 --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 --pic-suffix-from-cache-tokens 1 \
  --repeats 1 --warm-repeats 1 \
  --run-id "$RUN_ID"
```

Validity:

- `failures.csv`: empty.
- P0 grep printed nothing.
- `decode_tpot_long.csv`: 60 rows.
- PIC raw rows: 30/30 `benchmark_status=ok`, `decode_runtime=mnn_token_id_sparse_decode`.

Wide result:

```text
device,model,ctx,true_normal_x0,pic_x0,x1,x3,x5,x7,x0/true,extra_x1,extra_x3,extra_x5,extra_x7
orangepi,Llama3.2 3B,512,121.729,144.373,203.047,214.944,366.218,372.000,0.843,58.675,70.572,221.845,227.628
orangepi,Llama3.2 3B,1024,126.147,162.099,221.081,235.960,415.052,419.435,0.778,58.981,73.861,252.953,257.336
orangepi,MiniCPM5-1B,512,46.748,69.964,102.008,109.013,159.813,159.186,0.668,32.044,39.048,89.848,89.222
orangepi,MiniCPM5-1B,1024,50.067,78.109,116.820,126.048,192.872,193.092,0.641,38.711,47.939,114.763,114.983
orangepi,Qwen3-4B,512,149.885,193.645,271.890,282.515,480.298,486.637,0.774,78.245,88.870,286.653,292.993
orangepi,Qwen3-4B,1024,154.748,218.814,295.044,313.675,543.348,554.966,0.707,76.231,94.861,324.534,336.153
```

PIC deltas vs `decode_unified_family_orangepi_ctx512_1024_suffix1_20260704_145211`:

```text
model,ctx,x0,x1,x3,x5,x7
Llama3.2 3B,512,-19.9 (-12.1%),-23.7 (-10.5%),-14.3 (-6.2%),-16.8 (-4.4%),-16.5 (-4.2%)
Llama3.2 3B,1024,-15.9 (-8.9%),-21.7 (-8.9%),-21.1 (-8.2%),-18.8 (-4.3%),-19.7 (-4.5%)
MiniCPM5-1B,512,-13.2 (-15.9%),-14.7 (-12.6%),-13.7 (-11.2%),-14.7 (-8.4%),-14.7 (-8.5%)
MiniCPM5-1B,1024,-11.3 (-12.7%),-12.8 (-9.9%),-17.9 (-12.4%),-19.2 (-9.1%),-18.1 (-8.6%)
Qwen3-4B,512,-11.3 (-5.5%),-13.4 (-4.7%),-12.5 (-4.2%),-15.6 (-3.1%),-13.2 (-2.6%)
Qwen3-4B,1024,-7.7 (-3.4%),-9.3 (-3.1%),-12.9 (-4.0%),-16.0 (-2.9%),-20.7 (-3.6%)
```

Averages:

```text
x0 old=157.722 new=144.501 delta=-13.221 pct=-8.4%
x1 old=217.584 new=201.648 delta=-15.936 pct=-7.3%
x3 old=229.104 new=213.692 delta=-15.412 pct=-6.7%
x5 old=376.450 new=359.600 delta=-16.850 pct=-4.5%
x7 old=381.369 new=364.220 delta=-17.150 pct=-4.5%
```

## Next bottleneck

After removing eager logits readback from `decode_us`, warm profile still shows:

- q=1: attention total about 16 ms, other forward cost about 64 ms.
- q=2: attention total about 26 ms, other forward cost about 85 ms.
- q=6: attention total about 46 ms, other forward cost about 118 ms.

Next real optimization should target active-row graph cost / dense-Raster-elementwise path and qtile attention scaling. Endpoint wall-time should be added to the benchmark CSV before treating deferred readback as end-to-end latency improvement.

