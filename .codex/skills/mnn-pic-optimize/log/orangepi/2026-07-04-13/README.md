# OrangePi decode-repair record queue

## Summary

- Added a same-family record-queue path for OpenCL PIC decode-repair sparse/q1-row attention.
- The path records the per-layer `append_sparse_decode_key_value_hd128` + selected sparse/q1-row attention kernel pair and updates only dynamic buffers and length arguments per decode step.
- It is still PIC dualgraph decode-repair: `x=0` remains active rows=1, `x>0` remains active rows=x+1. It does not route to ordinary q=1 / identity decode.
- Default is enabled through `MNN_PAGED_ATTENTION_DECODE_REPAIR_RECORD_QUEUE=1`; `=0` is an explicit A/B off switch.
- Profile mode still uses the non-record path so attribution logs remain readable.

## Validation

Build:

```text
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1
```

Result: OrangePi aarch64 `pic_server`, `libMNN.so`, `libMNN_CL.so`, and `libpic_llm.so` built and synced.

Smoke:

```text
decode_repair_record_smoke_orangepi_minicpm_ctx512_20260704_133619
decode_repair_record_smoke_orangepi_minicpm_ctx512_x013_20260704_134851
```

Both had no `failures.csv`; P0 grep found no `decode_prepare_inside_decode=1`,
`ERROR`, `target unavailable`, `async persistent PIC cache read failed`,
`Cache invalid`, `Empty reply from server`, or `Connection refused`.

## Effect

Qwen3-4B ctx1024 PIC x0 A/B, profile off, warm excluded:

```text
record_off: 160.255 ms/token, wall 188.983 ms/token
record_on:  156.809 ms/token, wall 185.561 ms/token
delta:       -3.446 ms/token, wall -3.422 ms/token
```

Same-run true normal vs PIC x0 with record on:

```text
true_normal_x0: 155.075 ms/token
PIC_dualgraph_x0: 159.433 ms/token
gap: +4.358 ms/token, 1.028x normal
```

This removes the previous 20 ms class Qwen ctx1024 x0 gap as a "clearly slower"
case. The residual target is now small enough that the next main work should move
to terminal vocab/top1 production and deeper PA q1 kernel cost.

## Formal Matrix

Run:

```text
decode_repair_record_formal_orangepi_ctx512_1024_20260704_135242
```

Scope: OrangePi OpenCL, MiniCPM5-1B / Llama3.2-3B / Qwen3-4B,
ctx512/1024, x0/x1/x3/x5/x7, `warm_repeats=1`, `repeats=2`.

Validity:

```text
failures.csv absent
P0 grep clean
no residual pic_server / llm_bench process
```

Measure-only decode TPOT:

```text
model,ctx,true_normal_x0,pic_x0,x1,x3,x5,x7,pic_x0_gap
MiniCPM5-1B,512,46.640,41.714,76.602,83.698,111.743,111.191,-4.926
MiniCPM5-1B,1024,49.722,45.936,95.994,107.326,132.482,129.632,-3.786
Llama3.2-3B,512,121.909,116.925,175.278,183.409,326.270,325.311,-4.984
Llama3.2-3B,1024,126.075,135.705,194.378,208.374,351.985,356.557,+9.630
Qwen3-4B,512,149.708,151.098,234.132,242.419,423.271,427.223,+1.390
Qwen3-4B,1024,155.142,173.639,256.586,272.607,458.451,461.541,+18.497
```

`extra_ms[x] = tpot[x] - tpot[x0]`:

```text
model,ctx,extra_x1,extra_x3,extra_x5,extra_x7
MiniCPM5-1B,512,34.888,41.984,70.029,69.478
MiniCPM5-1B,1024,50.058,61.390,86.545,83.695
Llama3.2-3B,512,58.353,66.483,209.345,208.386
Llama3.2-3B,1024,58.673,72.669,216.280,220.851
Qwen3-4B,512,83.035,91.321,272.174,276.126
Qwen3-4B,1024,82.947,98.968,284.812,287.902
```

Conclusion: record queue plus PIC decode graph routing makes x0 acceptable for
MiniCPM and Llama ctx512, and near-normal for Qwen ctx512. Llama ctx1024 and
Qwen ctx1024 still have a real x0 gap versus true normal, so the next work should
profile graph tail / PagedAttention / logits production at ctx1024 rather than
changing x0 into an ordinary decode family.

## Next

- Run graph/profile attribution for Llama ctx1024 and Qwen ctx1024 x0 against
  true normal x0, focusing on PagedAttention, Raster/dense tail, and logits/top1.
- Then run the Jetson audit/matrix to confirm no decode-repair family split remains there.
- Continue the larger graph-tail work: avoid materializing full `[1, vocab]`
  logits for greedy top1, or fuse `lm_head + top1` in the PIC decode graph.
