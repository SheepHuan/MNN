# Context — Jetson Small-M Weight-Only Linear (V15) Design

## Scope

Continuation of `jetson/2026-07-01-18` (x3 cliff profile). Target models: MiniCPM5-1B,
Qwen3-4B, Llama3.2-3B. Llama3.2-1B excluded.

Device: Jetson AGX Xavier, sm_72 (CUDA arch 7.2), 32GB LPDDR4x unified memory, nominal
~136.5 GB/s, effective STREAM ~80-95 GB/s. Cross-compile via `mnn-opt-ops` SKILL
(`CUDA_ARCHS=72`).

## Root Cause (from source)

`ConvFpAIntBExecution.cu` INT4 + 1x1 + fp16/mix dispatch (`onExecute`):

```
batch = inputs[0]->batch()                         // active_rows = repair_tokens + 1
int4GemvBatchLimit = picDecodeRepairSparse ? 3 : 6   // line 2292

batch <= int4GemvBatchLimit  -> V14 / V14_MB tiny-GEMV   (line 2324)
else                         -> rows48 cuBLASLt (2462) / rows45 cuBLAS (2458)
                                / generic CUTLASS + dequant (2527)
```

decode-repair sets `picDecodeRepairSparse=true`, so GEMV limit = 3:
- x=0 (batch=1) -> V14
- x=1 (batch=2) -> V14_MB<2>
- x=3 (batch=4) -> leaves GEMV -> dense FP16 weight path  <- cliff
- x=5 (batch=6), x=7 (batch=8) -> same dense path

### Why GEMV is fast (V14 factored dequant)

`PrecomputeGemvParams` (line 552) bakes `{scale, adj_offset = offset - 8*scale}`.
Dequant `w = (nibble-8)*scale + offset = scale*nibble + adj_offset`, so:
`acc += w*in = scale * sum(nibble*in) + adj_offset * sum(in)`.
- Weights stay packed int4 (0.5 B/element), never expanded to FP16.
- One block per OC group; `MAX_BATCH` rows share the single weight load (V14_MB line 1440).
- No dequant in hot loop.

### Why dense path is slow

rows45/rows48/generic consume `mDequantFilter [ocp, icp]` = dequantized FP16 weights
(2 B/element, 4x int4). If static cache misses (`mlpLikeLargeLinear`/
`attentionProjectionLikeLinear` gates, lines 1952-1957), `mNeedRuntimeDequant` runs
`DequantizeInt4Weight` every step (lines 2537-2577) — this is the Qwen3-4B
`runtime_dequant=1 static_dequant=0` case (PLAN line 85, log 2026-07-01-18).

## Bandwidth Roofline (effective 80 GB/s)

Weights = `oc*ic` bytes (int4) / `*4` (FP16). Read time = bytes / 80 GB/s.

Llama3.2-3B (hidden=3072, inter=8192, kv=1024, 28 layers):
| op | int4 bytes | FP16 bytes | int4 ms | FP16 ms |
|---|---|---|---|---|
| gate/up 8192x3072 | 12.58M | 50.33M | 0.157 | 0.629 |
| down 3072x8192 | 12.58M | 50.33M | 0.157 | 0.629 |
| q/o 3072x3072 | 4.72M | 18.87M | 0.059 | 0.236 |
| kv 1024x3072 | 1.57M | 6.29M | 0.020 | 0.079 |

Qwen3-4B (hidden=2560, inter=9728, kv=1024, 36 layers):
| op | int4 bytes | FP16 bytes | int4 ms | FP16 ms |
|---|---|---|---|---|
| gate/up 9728x2560 | 12.44M | 49.74M | 0.156 | 0.622 |
| down 2560x9728 | 12.44M | 49.74M | 0.156 | 0.622 |
| q 4096x2560 | 5.24M | 20.97M | 0.065 | 0.262 |
| kv 1024x2560 | 1.31M | 5.24M | 0.016 | 0.066 |
| o 2560x4096 | 5.24M | 20.97M | 0.065 | 0.262 |

MiniCPM5-1B (hidden~1024, inter~4096): ~1/6 of 3B weights, per-op int4 ~0.025-0.06 ms.

Conclusion: int4 weight-read floor for batch=4 == batch=1/2 floor. The cliff is the 4x
weight bandwidth of the FP16 dense path, not the extra batch rows. So a batch>=4 kernel
that keeps weights int4 and reuses them across rows can in principle reach the x=0/1 floor.

## V15 Design (see CUDA_SMALL_M_WEIGHT_ONLY_LINEAR_OPT_DESIGN.md for full)

`GEMV_FpAInt4B_V15_SmallM<T, OC_PER_BLK>`:
- Block 128 threads, grid `ceil(oc/OC_PER_BLK)`.
- Batch dimension via shared-memory + micro-passes (`B_TILE=2`,
  `num_micro_pass = ceil(batch/2)`), NOT registers. Weight loaded once into shared mem,
  reused across micro-passes.
- Register pressure = `B_TILE*OC_PER_BLK` (e.g. 2*8=16 floats), vs V14_MB's
  `MAX_BATCH*OC_PER_BLK` (e.g. 6*4=24 or 8*4=32 -> spills). Decouples OC_PER_BLK from
  MAX_BATCH, so OC_PER_BLK=8 is reachable.
- Same factored-dequant math as V14 (bit-exact equivalence modulo FMA order; same ULP
  class as existing V14/V14_MB cross-batch variants).
- Reads `mResource->mFilter` (packed int4) + `mResource->mGemvParams` directly. Does NOT
  touch `mDequantFilter` / runtime dequant -> fixes Qwen3-4B runtime_dequant.

OC_PER_BLK by shape: 8 for gate/up/down/q/o (large oc), 4 for kv_proj (oc=1024).
batch coverage: 4/6/8 via 2/3/4 micro-passes. batch 1/2/3 stay on V14/V14_MB.

## Integration (no breakage)

- New kernel inserted after V14_MB (line ~1548). No existing kernel touched.
- New dispatch branch in `onExecute` `else` (batch > limit), BEFORE rows48/rows45,
  env-gated `MNN_CUDA_PIC_INT4_SMALLM_V15` (default off). On miss -> fallthrough to
  existing rows48/rows45/generic unchanged.
- No `onResize` change. No op/schema/registry change. No export change.
- New profile tag `conv_fpa_intb_1x1_smallm_v15`.
- New bench: `CudaSmallMV15Accuracy` (vs rows45 cuBLAS ref, rows4/6/8 x 3 models,
  `memory=2`); `CudaWeightOnlyConvPerf` rows>=4 additionally runs V15 path via env,
  printed alongside cuBLAS baseline (baseline cases untouched).

## Negative-result avoidance (PLAN Non-Goals + mnn-opt-ops SKILL 248-259)

| prior attempt | rejected because | V15 avoids |
|---|---|---|
| int4GemvBatchLimit 3->5 | V14_MB slower on rows4/5 (register spill) | fresh tile, not V14_MB retune |
| V14_MB OC_PER_BLK sweep 2/3/4/8/16 | occupancy wrong | shared-mem micro-pass decouples |
| naive fused SwiGLU+down | no headroom | V15 = single Linear, no fusion |
| naive WMMA down | slower than cuBLAS | V15 no WMMA |
| fresh gate/up packed export | /v1/prefill/text fails + slower e2e | no graph/export change |
| PicLinearNhwcWeightOnly graph rewrite | e2e +2ms | backend-internal route only |

## Open risks

- Occupancy: V15 tile (OC_PER_BLK=8, shared-mem weight tile) must be direct-op tuned on
  Jetson sm_72; roofline is necessary but not sufficient (PLAN line 90).
- If V15 does not beat cuBLAS on direct-op, stays env-off; production unchanged.

## End-to-end expectation

Llama3.2-3B MLP/TPOT delta x3-x1 = +17.72ms -> /28 layers = ~0.633ms/layer (dense FP16).
V15 int4 floor ~3*0.157 = 0.471ms/layer, same order as x=0/1 V14 floor. Expect x=3 MLP
to approach x=1; residual gap = batch input read + reduction + launch constant (<0.05ms/
layer). Qwen3-4B gains extra from eliminating runtime_dequant. Verdict: bandwidth floor
supports closing x=3/5/7 toward x=0/1; final gate is Phase 1 direct-op on Jetson.
