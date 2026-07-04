# OrangePi decode-repair wall TPOT harness

## Summary

- Added request wall timing propagation for PIC dualgraph decode-repair benchmark CSVs.
- PIC raw CSV now keeps `request_wall_s`, `request_wall_tpot_ms`, `completion_tokens`, and `decode_measured_tokens`.
- Decode experiment `decode_tpot_long.csv` now keeps `request_wall_s` and `wall_tpot_ms`.
- Decode experiment `decode_tpot_wide.csv` now reports PIC wall TPOT and wall extra-ms columns.

## Smoke

Run:

```text
decode_wall_smoke_orangepi_minicpm_ctx512_20260704_162059
```

Scope: OrangePi, MiniCPM5-1B, ctx512, PIC only, x0/x1, 1 warm, 1 measure. This is a schema/runtime smoke, not a formal performance row.

Result:

```text
x0 decode_tpot_ms=66.284 wall_tpot_ms=81.893 runtime=mnn_token_id_sparse_decode
x1 decode_tpot_ms=99.583 wall_tpot_ms=113.086 runtime=mnn_token_id_sparse_decode
```

P0 log scan found no `decode_prepare_inside_decode=1`, `ERROR`, `target unavailable`, `async persistent PIC cache read failed`, or `Cache invalid`.

## Next

Rerun the OrangePi formal ctx512/1024 matrix with true-normal x0 enabled and use both `decode_tpot_ms` and `wall_tpot_ms` in the report. The previous defer-logits optimization improves `decode_tpot_ms`; the wall TPOT columns are needed to prove endpoint latency improvement instead of only moving logits materialization outside `decode_us`.

## Formal Matrix

Run:

```text
decode_wall_formal_orangepi_ctx512_1024_20260704_162430
```

Scope: OrangePi, MiniCPM5-1B / Llama3.2 3B / Qwen3-4B, ctx512/1024, true-normal x0 plus PIC dualgraph decode-repair x0/x1/x3/x5/x7, profile off, warm excluded.

Validity:

- `failures.csv` was not produced.
- All 30 PIC rows are `benchmark_status=ok`.
- All PIC rows use `decode_runtime=mnn_token_id_sparse_decode`.
- All PIC rows have `request_wall_s` and `request_wall_tpot_ms`.
- P0 grep found no `decode_prepare_inside_decode=1`, `ERROR`, `target unavailable`, `async persistent PIC cache read failed`, `Cache invalid`, `Empty reply from server`, or `Connection refused`.
- OrangePi logs contain normal first-use `Warning: module need new clone` lines, but each warning is followed by completed decode-repair rows; this is not the Rhino failure mode where the server stopped after clone and returned empty replies.

Decode TPOT formal table:

```text
device,model,ctx,true_normal_x0,pic_x0,x1,x3,x5,x7,x0_abs_gap,extra_x1,extra_x3,extra_x5,extra_x7
orangepi,Llama3.2 3B,512,121.751,148.821,208.780,215.298,372.313,376.035,27.069,59.959,66.477,223.492,227.214
orangepi,Llama3.2 3B,1024,125.632,163.133,227.098,238.410,413.843,424.841,37.501,63.964,75.277,250.709,261.708
orangepi,MiniCPM5-1B,512,46.729,70.088,100.141,107.824,159.588,158.913,23.359,30.054,37.737,89.500,88.825
orangepi,MiniCPM5-1B,1024,49.768,78.318,118.778,127.180,194.953,195.519,28.550,40.461,48.862,116.635,117.201
orangepi,Qwen3-4B,512,149.809,196.455,268.902,281.061,487.305,494.116,46.646,72.447,84.606,290.850,297.661
orangepi,Qwen3-4B,1024,155.111,215.387,299.755,314.360,539.230,558.090,60.276,84.368,98.973,323.844,342.703
```

PIC endpoint wall TPOT table:

```text
device,model,ctx,pic_x0_wall,x1_wall,x3_wall,x5_wall,x7_wall,wall_extra_x1,wall_extra_x3,wall_extra_x5,wall_extra_x7
orangepi,Llama3.2 3B,512,172.659,228.504,233.327,380.312,383.676,55.845,60.668,207.653,211.017
orangepi,Llama3.2 3B,1024,189.188,249.429,259.944,424.345,434.579,60.241,70.756,235.157,245.391
orangepi,MiniCPM5-1B,512,85.775,116.470,121.116,169.709,169.122,30.695,35.341,83.934,83.347
orangepi,MiniCPM5-1B,1024,95.311,133.351,141.099,204.854,205.229,38.040,45.789,109.544,109.918
orangepi,Qwen3-4B,512,216.853,285.830,296.141,489.351,495.769,68.977,79.288,272.498,278.916
orangepi,Qwen3-4B,1024,240.310,319.777,333.530,544.222,561.440,79.467,93.220,303.912,321.130
```

Average `wall_tpot_ms - decode_tpot_ms` by x:

```text
x0=21.316ms x1=18.318ms x3=16.838ms x5=7.594ms x7=7.050ms
```

## Greedy ArgMax / Logits Slice

Implemented a scoped sampler-side logits optimization:

- PIC decode-repair now records the last valid logits row in `validLogitStart` / `validLogitSize`.
- `Llm::sample()` no longer creates the offset slice by first `readMap()`-ing the full logits tensor. It builds an Express `_Slice` for the requested row.
- Greedy sampling uses Express `_ArgMax(logits, -1)` and reads back one int token instead of running the C++ sampler over a copied float vocab row.
- Non-greedy samplers still use the existing sampler pipeline.

Validation:

```text
build: MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1
artifact: aarch64 pic_server / libMNN.so / libMNN_CL.so / libpic_llm.so
smoke: decode_argmax_smoke_orangepi_minicpm_ctx512_20260704_1718
matrix: decode_argmax_orangepi_ctx512_1024_pic_20260704_1726
```

The profile smoke confirmed the new `sample_device_argmax` path fires. P0 grep for the
matrix found no `decode_prepare_inside_decode=1`, `ERROR`, `target unavailable`,
`async persistent PIC cache read failed`, `Cache invalid`, `Empty reply from server`,
`Connection refused`, or sampler failures.

Against `decode_wall_formal_orangepi_ctx512_1024_20260704_162430`, same PIC rows only:

```text
x,avg_delta_decode_ms,avg_delta_wall_ms
0,-1.949,-2.033
1,-1.654,-1.687
3,-0.856,-0.829
5,-2.080,-1.895
7,-2.340,-2.227
overall,-1.776,-1.734
```

This is a small endpoint win, not a main bottleneck fix. Some individual points still
move within noise or slightly regress, so the next larger optimization remains
active-row graph cost and lm_head/vocab production rather than sampler readback alone.

## Decode-Repair Decode Graph Route

Implemented and validated a runtime route that lets PIC decode-repair use the PIC
package `llm_decode.mnn` when `mPicDecodeRepair.enabled`, `inDecode`,
`seqLen == seqLenKey`, and only the last logits row is requested.

This keeps the same PIC dualgraph decode-repair family: x0 is still active rows=1,
x>0 is active rows=x+1, and OpenCL still reports `repair=1`, `ordinary_q1=0`, and
`repair_qtile_route=1`. It is not a true-normal LLM path.

Validation:

```text
smoke/profile: decode_repair_decodegraph_smoke_orangepi_minicpm_ctx512_20260704_1746
formal PIC:    decode_repair_decodegraph_orangepi_ctx512_1024_pic_20260704_1749
profile note:  profile_decodegraph_x0_hidden_orangepi_minicpm_ctx512_20260704_1758
```

Formal PIC-only run had no `failures.csv`; P0 grep found no `ERROR`,
`target unavailable`, `async persistent PIC cache read failed`, `Cache invalid`,
`decode_prepare_inside_decode=1`, or `ordinary_q1=1`.

x0 vs previous PIC and true-normal x0:

```text
model,ctx,true_normal_x0,old_pic_x0,new_pic_x0,delta_vs_old,gap_vs_normal,ratio_vs_normal
MiniCPM5-1B,512,46.729,66.407,50.479,-15.928,3.750,1.080
MiniCPM5-1B,1024,49.768,80.563,58.618,-21.945,8.850,1.178
Llama3.2 3B,512,121.751,147.353,121.958,-25.395,0.206,1.002
Llama3.2 3B,1024,125.632,160.086,136.278,-23.808,10.646,1.085
Qwen3-4B,512,149.809,193.453,162.153,-31.299,12.344,1.082
Qwen3-4B,1024,155.111,212.647,178.487,-34.160,23.376,1.151
```

Per-x TPOT and family extra-ms:

```text
model,ctx,x,tpot_ms,extra_ms,delta_vs_old
MiniCPM5-1B,512,0,50.479,+0.000,-15.928
MiniCPM5-1B,512,1,77.696,+27.217,-20.026
MiniCPM5-1B,512,3,84.711,+34.232,-25.137
MiniCPM5-1B,512,5,136.792,+86.313,-22.216
MiniCPM5-1B,512,7,137.996,+87.517,-20.995
MiniCPM5-1B,1024,0,58.618,+0.000,-21.945
MiniCPM5-1B,1024,1,97.466,+38.848,-21.846
MiniCPM5-1B,1024,3,110.973,+52.355,-17.686
MiniCPM5-1B,1024,5,168.225,+109.607,-28.439
MiniCPM5-1B,1024,7,177.803,+119.185,-17.230
Llama3.2 3B,512,0,121.958,+0.000,-25.395
Llama3.2 3B,512,1,181.074,+59.116,-24.548
Llama3.2 3B,512,3,188.299,+66.341,-22.575
Llama3.2 3B,512,5,345.700,+223.742,-24.371
Llama3.2 3B,512,7,349.264,+227.306,-23.828
Llama3.2 3B,1024,0,136.278,+0.000,-23.808
Llama3.2 3B,1024,1,198.022,+61.743,-30.353
Llama3.2 3B,1024,3,213.695,+77.416,-26.058
Llama3.2 3B,1024,5,386.097,+249.818,-24.348
Llama3.2 3B,1024,7,395.485,+259.207,-28.201
Qwen3-4B,512,0,162.153,+0.000,-31.299
Qwen3-4B,512,1,234.124,+71.970,-33.340
Qwen3-4B,512,3,245.001,+82.848,-33.845
Qwen3-4B,512,5,450.606,+288.453,-31.294
Qwen3-4B,512,7,458.422,+296.268,-31.736
Qwen3-4B,1024,0,178.487,+0.000,-34.160
Qwen3-4B,1024,1,258.114,+79.627,-36.919
Qwen3-4B,1024,3,273.461,+94.974,-37.556
Qwen3-4B,1024,5,508.036,+329.549,-28.630
Qwen3-4B,1024,7,513.870,+335.383,-38.643
```

Conclusion:

- This is the first large OrangePi decode-repair x0 fix: x0 is now close to true
  normal on MiniCPM/Llama and materially closer on Qwen.
- x1/x3/x5/x7 all improve in the same run, so the change behaves like removal of a
  shared fixed graph overhead rather than an x0-only gate.
- Remaining hidden overhead should be attacked in this order: terminal vocab
  production (`lm_head + top1`), decode attention append/scan fixed cost, then
  active-row dense/Raster/elementwise cleanup for x>0.

## QTile Heuristic And Current Residual

After keeping x0 in the repair family, the OpenCL qtile choice was promoted as a
same-family parameter heuristic: Mali uses `q_tile=4` for active rows 5-7 and
`q_tile=2` for active rows >=8; x0 remains `q1_row=1`. This is not an x/model/head
gate and does not route any case to the ordinary q=1 family.

Formal OrangePi PIC-only validation:

```text
decode_repair_qtile_heuristic_default_orangepi_ctx512_1024_pic_20260704_125913
```

P0 grep was clean and no `failures.csv` was produced. Against the prior q1-row
default, x5/x7 improved on all 6 model/context cases:

```text
model,ctx,dx0,dx1,dx3,dx5,dx7
Llama3.2 3B,512,+3.879,-0.840,-0.165,-18.073,-29.452
Llama3.2 3B,1024,-6.427,+0.406,+1.669,-29.823,-38.176
MiniCPM5-1B,512,+1.606,+0.983,-0.706,-26.557,-26.853
MiniCPM5-1B,1024,+2.728,-5.247,-3.714,-47.226,-45.008
Qwen3-4B,512,+17.374,+0.888,-5.549,-31.095,-29.721
Qwen3-4B,1024,+2.421,+10.291,+4.415,-44.227,-45.849
```

Same-run Qwen x0 check with true-normal enabled:

```text
decode_x0_same_run_default_orangepi_qwen_ctx512_1024_20260704_130801
ctx512:  true-normal 149.854 ms, PIC x0 148.146 ms, gap -1.709 ms
ctx1024: true-normal 154.648 ms, PIC x0 175.235 ms, gap +20.587 ms
```

Current Qwen ctx1024 profile:

```text
profile_current_default_qwen_ctx1024_x0_orangepi_20260704_131856
```

Route is valid: `pic_decode_repair_decode_graph=1`, `ordinary_q1=0`,
`repair_qtile_route=1`, `q1_row=1`, `decode_prepare_inside_decode=0`.
Decode-specific q1-row PagedAttention totals 36 layers / 22.034 ms, mean
612 us/layer. `forward_raw_logits_defer_map` is 0 and `sample_device_argmax` is
about 0.6 ms in this profile, so the remaining Qwen ctx1024 x0 gap is not logits
readback. The next generic targets are still `lm_head + top1` production and
decode attention fixed scan/append cost inside the same repair family.
