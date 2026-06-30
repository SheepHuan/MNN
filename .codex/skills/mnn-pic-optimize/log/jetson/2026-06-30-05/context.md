# Context

## Evidence

Same-selector profile was collected for the current NHWC artifact:

```text
summary: .cache/mnn-pic-benchmark/nhwc_linear_x01357_profile_20260630_125240/summary.csv
server log: .cache/logs/jetson_nhwc_linear/nhwc_linear_x01357_profile_20260630_125240.log
artifact: .cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear
model config: .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-nhwc-linear/config_cuda_greedy.json
selector: lagged_attention_hkvd
context: 1024
mode: full-reuse
max_tokens: 1 for profile attribution
```

Formal endpoint TPOT remains from the non-profile run:

| x | active rows | TPOT ms | delta vs x=1 |
|--:|------------:|--------:|-------------:|
| 0 | 1 | 81.381 | -10.697 |
| 1 | 2 | 92.078 | 0.000 |
| 3 | 4 | 118.127 | +26.049 |
| 5 | 6 | 119.656 | +27.578 |
| 7 | 8 | 120.848 | +28.770 |

The shape/call profile explains why `x=3/5/7` cluster together: the graph call counts are almost fixed after entering sparse repair. For `x=1/3/5/7`:

```text
attention kernel calls: 56
PicLinearNhwcWeightOnly Extra calls: 168
Convolution calls: 226
Raster+Cast calls: 976
Binary+Unary calls: 566
LayerNorm calls: 114
```

So the extra latency is not a per-row-linear increase. It is mostly the fixed cost of the compact dense path used once active rows reach rows4/6/8.

## MLP Bottleneck

Low-level `CUDAWeightOnlyConv profile` totals by dense family:

| dense family | x=0 | x=1 | x=3 adjusted | x=5 | x=7 | x7-x1 |
|-------------|----:|----:|--------------:|----:|----:|------:|
| attn q/o | 9.49 | 10.25 | 15.84 | 16.45 | 16.57 | +6.31 |
| attn k/v | 5.29 | 5.41 | 7.98 | 7.99 | 7.79 | +2.38 |
| MLP gate/up | 19.49 | 21.48 | 32.08 | 32.25 | 32.29 | +10.81 |
| MLP down | 8.74 | 10.25 | 17.82 | 17.85 | 17.92 | +7.66 |
| lm_head | 3.99 | 3.97 | 3.96 | 3.96 | 3.95 | -0.02 |
| total | 47.00 | 51.36 | 77.68 | 78.50 | 78.51 | +27.15 |

`x=3 adjusted` excludes one profile-only first-shape spike:

```text
conv_fpa_intb_1x1_rows45_cublas batch=4 3072->3072 87.286 ms
```

That spike should not be used as formal endpoint attribution; it is a profile/synchronization artifact. The stable observation is that rows4/6/8 dense totals are around `78.5 ms`, versus rows2 `51.36 ms`.

## Optimization Direction

P0 should be a generic parameterized compact MLP path, starting with `gate_proj + up_proj` fusion at the graph/export or CUDA execution level:

- `gate_proj` and `up_proj` have the same input `[rows, hidden=3072]` and output width `8192`.
- They are independent until `SiLU(gate) * up`, so they can be computed as one logical `gate_up` operation.
- A first implementation can concatenate the two output matrices into one effective `OC=16384` compact dense, then run fused split + `SiLU(gate) * up` to produce the intermediate `[rows, 8192]`.
- This halves the number of gate/up dense launches and can share input reads and scheduling overhead.
- The implementation should be guarded to Llama3.2 3B PIC compact NHWC decode shapes first: `rows in {4,6,8}`, `ic=3072`, `inter=8192`, INT4 weight-only, static dequant available.

This must not become three separate `x=3`, `x=5`, and `x=7` operators. The intended design is one compact MLP execution with:

```text
runtime parameters:
  rows = active rows, e.g. 4/6/8
  hidden = 3072
  inter = 8192
  quant = int4 weight-only
  layout = compact NHWC rows

schedule parameters:
  M tile / rows per CTA
  N tile
  K tile
  num warps / block size
  vector width
  optional split-K or staging depth
```

`x=3/5/7` can select different schedule parameters from a small policy table or autotuned profile, but the graph rewrite, CUDA execution class, math, memory layout, output semantics, and validation path must be shared.

The current code already has related pieces:

- `PicLinearNhwcWeightOnly` keeps compact NHWC rows and delegates to existing weight-only Conv. This is the accepted endpoint path but still runs gate/up/down as separate linears.
- `PicGateUpSiluWeightOnly` already expresses a gate/up/silu logical fusion, but its current direct scalar fused path is a narrow rows2..8 GEMV-style path and is not the desired general tensor-core compact MLP solution.
- `PicPackedSiluMul` can consume a packed gate/up output. That makes a staged implementation practical: first make one compact `gate_up` dense producing packed `[rows, 2 * inter]`, then use a shared packed SiLU/mul output path.

P1 should target `down_proj`:

- Current down is `8192 -> 3072` for rows4/6/8 and accounts for `+7.66 ms` of `x7-x1`.
- Simple packed GEMV expansion to batch <=32 has been rejected in earlier Jetson A/B, so do not default to that path without fresh proof.
- Safer options:
  - tune/add a rows4/6/8 down-specific tensor-core compact GEMM only if direct op bench proves it beats current rows45/cuBLAS for all rows4/6/8;
  - or fold down into a larger MLP fusion that streams `SiLU(gate) * up` tiles directly into down accumulation, avoiding the intermediate activation write/read. This is more complex and should be P1/P2 after `gate_up` fusion proof.

## Generic Implementation Sketch

Use a single new CUDA execution family, conceptually `PicCompactMlpWeightOnly`, with staged milestones:

1. Export-level fusion:
   - Detect the MLP pattern `gate_proj(hidden)`, `up_proj(hidden)`, `SiLU(gate) * up`, `down_proj(...)`.
   - Rewrite only the gate/up/silu portion first.
   - Keep `down_proj` as the existing `PicLinearNhwcWeightOnly` until gate/up is proven.
   - Keep fallback to the current separate NHWC linears if quant/layout/model guards fail.

2. Gate/up packed dense:
   - Build one logical packed weight-only linear with output width `2 * inter`.
   - Keep input/output in compact NHWC rows.
   - Reuse the existing static dequant sharing/resource-key mechanism so cloned decode modules do not pay runtime dequant.
   - Feed the packed result to a fused packed SiLU/mul kernel.

3. Parameterized CUDA schedule:
   - One algorithm for rows4/6/8: load compact rows, stream K, compute gate/up accumulators, apply SiLU/mul.
   - Different rows can select only schedule parameters, not different code semantics.
   - Candidate schedule knobs are `M_TILE`, `N_TILE`, `K_TILE`, `num_warps`, vectorized load width, and optional split-K.
   - A direct bench should sweep these knobs for rows4/6/8 and choose a small default policy.

4. Down projection:
   - Keep current path for the first milestone.
   - After gate/up win is proven, add a parameterized down compact GEMM schedule under the same policy framework.
   - Only then consider full MLP fusion that streams activation tiles into down accumulation.

Avoid:

- one-off `x3`, `x5`, `x7` operators;
- separate exporter branches per repair token count;
- env-only production paths;
- resurrecting batch<=32 packed GEMV as the default without fresh rows4/6/8 proof.

## Goal Candidate

Optimize Jetson CUDA compact MLP for `lagged_attention_hkvd` decode repair by replacing the separate rows4/6/8 `gate_proj` and `up_proj` small-M dense calls with one shared parameterized `gate_up` path, then evaluating a down-specific compact dense improvement under the same parameterized schedule policy. The goal is complete only when:

- correctness passes for `x=0/1/3/5/7`;
- formal endpoint TPOT shows no `x=0/1` regression beyond noise;
- `x=3/5/7` each improve versus current NHWC static-share baseline;
- low-level profile shows `MLP gate/up` total reduced from the current `~32.3 ms` rows4/6/8 level;
- graph/profile evidence confirms the endpoint decode path hits the new MLP path.
- the implementation has one graph/export path and one CUDA execution family for rows4/6/8, with row-specific differences limited to schedule parameters.

Stretch target: bring `x=3/5/7` TPOT to `<=110 ms`. First acceptable milestone: remove at least `8 ms` from the combined MLP dense overhead and get `x=7` below `115 ms`.
