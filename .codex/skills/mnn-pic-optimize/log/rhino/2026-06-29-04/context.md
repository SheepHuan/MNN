# Rhino Decode Repair FFN Fusion Context

## Baseline And Profile Motivation

Earlier graph profile for `lagged_attention_hkvd` showed the extra x cost is dominated by request `forwardRaw`, not by rank selection/readback. The main deltas are tiny-row `Convolution` in attention projections and FFN (`gate_proj`, `up_proj`, `down_proj`), with `PicSparseAttention` secondary.

Current Rhino production model:

```text
remote: /mnt/nvme/mnn_pic_opencl/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary
export_args.json: pic_decode_tiny_fusion=false, pic_decode_gateup_fusion=false
graph: PicSiluMul=0
```

Same-window default lagged baseline:

```text
.cache/tmp/rhino_decode_default_ab_lagged_x0_x7_20260629_033513.csv
```

## A/B 1: PicSiluMul Only

Export:

```text
.cache/weight/OpenBMB__MiniCPM5-1B-pic-boundary-tiny-fusion
flags: --pic_decode_tiny_fusion
graph: ops 2915, Conv 169, UnaryOp 50, BinaryOp 620, Extra PicSiluMul 24
```

Result:

```text
.cache/tmp/rhino_decode_tiny_fusion_lagged_x0_x7_20260629_033011.csv
```

Compared to same-window default:

```text
ctx,x,default_ms,silu_only_ms,delta
512,1,76.3,81.0,+6.2%
512,3,87.5,93.0,+6.2%
512,5,131.9,132.0,+0.1%
512,7,137.1,134.1,-2.2%
1024,1,109.6,98.1,-10.5%
1024,3,132.5,133.9,+1.0%
1024,5,160.9,161.1,+0.2%
1024,7,163.6,165.8,+1.3%
```

Conclusion: reject as default. It removes elementwise ops but does not consistently reduce low-x TPOT; `x=1/3` at ctx512 regresses.

## A/B 2: Gate+Up Concat Conv + PicPackedSiluMul

Code:

```text
source/backend/opencl/execution/buffer/FuseBufExecution.cpp
```

Implemented `PicPackedSiluMul` for OpenCL buffer backend. It reads one packed row-major NC4HW4 input with gate channels followed by up channels, computes `SiLU(gate) * up`, and writes contiguous `[rows, channels]` output for the following down projection.

Initial smoke was very slow:

```text
.cache/tmp/rhino_decode_gateup_fusion_smoke2_20260629_035051.csv
ctx512 x1 max_tokens=8: 519.5 ms/token
```

Root cause: the first implementation built the OpenCL source inside `onEncode`, so decode dynamic resize repeatedly compiled the program. Moving `buildKernelFromSource` to the execution constructor fixed the smoke:

```text
.cache/tmp/rhino_decode_gateup_fusion_reuse_kernel_smoke_20260629_035435.csv
ctx512 x1 max_tokens=8: 85.2 ms/token
```

Export:

```text
.cache/weight/OpenBMB__MiniCPM5-1B-pic-boundary-gateup-fusion
flags: --pic_decode_tiny_fusion --pic_decode_gateup_fusion
graph: ops 2795, Conv 145, UnaryOp 50, BinaryOp 620, Extra PicPackedSiluMul 24
```

Graph structure for layer 0:

```text
pre_reshape -> pre_convert -> concat_conv outputCount=9216 -> PicPackedSiluMul -> down_proj
```

Full result:

```text
.cache/tmp/rhino_decode_gateup_fusion_lagged_x0_x7_20260629_035500.csv
```

Compared to same-window default:

```text
ctx,x,default_ms,gateup_ms,delta
512,0,65.8,65.4,-0.7%
512,1,76.3,78.8,+3.2%
512,3,87.5,89.2,+1.9%
512,5,131.9,131.4,-0.4%
512,7,137.1,136.1,-0.7%
1024,0,99.3,77.2,-22.2%
1024,1,109.6,96.1,-12.4%
1024,3,132.5,131.7,-0.6%
1024,5,160.9,158.7,-1.4%
1024,7,163.6,153.5,-6.2%
```

Extra TPOT over each model's own x=0:

```text
ctx,x,default_extra,gateup_extra
512,1,10.5,13.4
512,3,21.7,23.9
512,5,66.1,66.1
512,7,71.3,70.8
1024,1,10.4,18.9
1024,3,33.2,54.5
1024,5,61.6,81.5
1024,7,64.4,76.3
```

Conclusion: gateup graph improves some absolute TPOT points, especially 1024/x7, but it does not make x=1/3/5/7 closer to x=0. Low-x repair overhead is worse when normalized to the model's own x=0. Do not default this export route.

## A/B 3: Shared-Input Split Gate/Up Conv + PicSiluMul

Code:

```text
transformers/pic_llm/export/llmexport.py
transformers/pic_llm/export/utils/model.py
transformers/pic_llm/export/utils/custom_op.py
transformers/pic_llm/export/utils/transformers.py
transformers/pic_llm/export/utils/mnn_converter.py
```

Added export flag:

```text
--pic_decode_gateup_split_fusion
```

This path keeps the `PicGateUpSiluWeightOnly` ONNX marker but lowers it to two separate weight-only Conv nodes sharing one `pre_reshape -> pre_convert`, then applies `PicSiluMul`:

```text
pre_reshape -> pre_convert -> gate_conv -> gate_post_convert -> gate_post_reshape
                         \-> up_conv   -> up_post_convert   -> up_post_reshape
gate_post_reshape + up_post_reshape -> PicSiluMul -> down_proj
```

It is not the final fused kernel; it isolates whether the concat `outputCount=9216` path was the primary regression.

Export:

```text
.cache/weight/OpenBMB__MiniCPM5-1B-pic-boundary-gateup-split-fusion
flags: --pic_decode_tiny_fusion --pic_decode_gateup_fusion --pic_decode_gateup_split_fusion
graph: ops 2915, Conv 169, UnaryOp 50, BinaryOp 620, Extra PicSiluMul 24
graph check: concat_conv=0, gate_conv=24, up_conv=24, split_silu=24
```

Smoke:

```text
.cache/tmp/rhino_decode_gateup_split_smoke_20260629_0412.csv
ctx512 x1 max_tokens=8: 90.9 ms/token
```

Full result:

```text
.cache/tmp/rhino_decode_gateup_split_lagged_x0_x7_20260629_0413.csv
```

Absolute TPOT compared to default and concat gateup:

```text
ctx,x,default,concat_gateup,split_gateup
512,0,65.8,65.4,66.2
512,1,76.3,78.8,78.6
512,3,87.5,89.2,89.8
512,5,131.9,131.4,131.7
512,7,137.1,136.1,135.3
1024,0,99.3,77.2,88.0
1024,1,109.6,96.1,96.7
1024,3,132.5,131.7,133.0
1024,5,160.9,158.7,161.6
1024,7,163.6,153.5,166.0
```

Extra TPOT over each model's own x=0:

```text
ctx,x,default,concat_gateup,split_gateup
512,1,10.5,13.4,12.4
512,3,21.7,23.9,23.6
512,5,66.1,66.1,65.6
512,7,71.3,70.8,69.2
1024,1,10.4,18.9,8.7
1024,3,33.2,54.5,45.0
1024,5,61.6,81.5,73.6
1024,7,64.4,76.3,78.0
```

Conclusion: the shared-input split path gives one useful signal: `ctx1024/x1` extra improves from `10.4ms` to `8.7ms`, so avoiding the generic `2*OC` concat path helps the smallest repair case. It still fails the actual objective: `ctx512/x1/x3` regress, and `ctx1024/x3/x5/x7` are farther from x=0 than the default model. Do not default this export route.

The result narrows the bottleneck: once repair rows grow to 4/6/8, the remaining extra is not dominated by SiLU/Mul or concat layout alone. The hot chain is still per-layer tiny-row dense work across attention q/o projections and MLP down/gate/up. The next implementation should target one of:

1. True fused `gate/up -> SiLU -> down` for rows 2/4/6/8, so the 4608-wide intermediate is not written/read as a full tensor before down_proj.
2. Dedicated tiny-row attention projection family for `1536->2048` q/o and `1536->256` k/v, because graph profile shows these are comparable to MLP per-layer cost.
3. A direct rows=4/6/8 weight-only kernel benchmark for `1536->2048`, `2048->1536`, `1536->4608`, and `4608->1536`, so end-to-end changes are not driven by stale OpenCL tune keys.

## Next Direction

The concat Conv path reduces graph op count but forces a generic `outputCount=9216` weight-only Conv; the split path avoids concat but still leaves rows=4/6/8 dense work unsolved. Neither shares enough of the hot input/dequant/accumulation work to solve low-x repair overhead.

Next useful implementation target:

1. A dedicated tiny-row `SwiGLU + down` or gate/up weight-only kernel that computes both projections in one kernel while sharing input loads and dequant scale reads.
2. Keep output in the layout needed by the following down projection, or fuse through down, without adding new per-token program compilation or extra ConvertTensor hops.
3. Measure with the same x=0/1/3/5/7, contexts 512/1024, 32 decode, and judge primarily by `extra_ms = TPOT(x) - TPOT(x=0)`.

## A/B 4: Row-Aware Adreno Tiny Dense Family

Motivation: the original Conv low-memory graph preserves the fast x=0 path, but rows=2/4/6/8 repair decode still pays large per-layer tiny-row dense cost in `forwardRaw`. Full NHWC Linear reduces x=5/x=7 but hurts x=0/x=1. The better target is to keep the original graph and only choose a better tiny-row Conv family for batch>1.

Screening with `MNN_BENCH_OPENCL_COMPACT_DENSE_FORCE_FAMILY` showed different sparse row counts prefer different families:

```text
ctx512, max_tokens=16, repeats=1
family                 x1      x3       x5       x7
default                81.5    104.5    132.9    134.0
pic_quant              78.3     98.5    135.0    135.7
pic_quant_b2           82.3     98.4    132.9    132.9
pic_quant_c4           76.4     98.6    135.7    134.0
adreno_direct_c4_gemv 101.3    118.0    137.8    148.2
adreno_batch_gemv_c4out 100.5  104.4    130.4    130.3
```

Additional rows=4 screening with `ctx512/x3/max_tokens=32/repeats=1`:

```text
family                 x3 TPOT ms/token
adreno_batch_gemv      108.1
pic_quant_wg64         109.3
pic_quant_wg128        109.8
pic_quant_wg4x32        87.3
pic_quant_wg8x16        97.1
pic_quant_incache32     90.9
pic_quant_incache64     87.1
```

Implemented row-aware heuristic in:

```text
source/backend/opencl/execution/buffer/ConvBufLowMemoryExecution.cpp
```

Default family selection for Adreno int4 tiny rows now is:

```text
rows <= 2:  pic_quant_c4
rows <= 4:  pic_quant_incache64
rows >= 6:  adreno_batch_gemv_c4out
```

First version used `pic_quant_c4` for rows<=4. It helped x=1 and high x but regressed x=3:

```text
.cache/tmp/rhino_decode_rowheur_lagged_x0_x7_20260629_1008.csv

x0 avg 87.500
x1 avg 90.307  extra +2.807
x3 avg 115.859 extra +28.359
x5 avg 140.699 extra +53.199
x7 avg 143.502 extra +56.002
```

Second version switched rows=4 to `pic_quant_incache64`:

```text
.cache/tmp/rhino_decode_rowheur2_lagged_x0_x7_20260629_1019.csv

ctx,x,TPOT_ms
512,0,73.437
512,1,77.481
512,3,95.045
512,5,129.896
512,7,130.734
1024,0,104.494
1024,1,114.729
1024,3,128.417
1024,5,149.506
1024,7,156.022
```

512/1024 average versus old same-window default:

```text
x,baseline_avg,rowheur2_avg,absolute_delta,baseline_extra,rowheur2_extra,extra_reduction
0,82.539,88.966,+6.427,0.000,0.000,0.000
1,92.977,96.105,+3.128,10.438,7.139,+3.299
3,110.005,111.731,+1.726,27.467,22.765,+4.701
5,146.394,139.701,-6.692,63.855,50.735,+13.119
7,150.372,143.378,-6.994,67.833,54.412,+13.421
```

Short `MNN_PIC_DECODE_REPAIR_PROFILE=1` run confirmed the residual cost is still almost entirely `forwardRaw`; embedding/mask/rank/bookkeeping are negligible:

```text
ctx512, max_tokens=8, excluding step0
repair_rows,sparse_rows,forward_raw_ms,total_ms
0,1,75.731,75.761
1,2,73.135,73.160
3,4,99.334,99.388
5,6,123.705,123.784
7,8,125.615,125.680
```

Conclusion: row-aware family selection is a real improvement for repair overhead, especially x=5/x=7, and keeps the original graph rather than using global NHWC Linear. It is not sufficient: high repair rows still sit about `50-54ms/token` above x=0 on the 512/1024 average. The next bottleneck is still inside `forwardRaw` dense Linear work, so the next useful implementation is a fused tiny-row MLP path or targeted attention projection tiny-row kernels.
