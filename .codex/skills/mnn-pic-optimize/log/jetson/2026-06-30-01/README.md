# Jetson CUDA NHWC Linear Decode Repair

## Summary

- Implemented the exported `PicLinearNhwcWeightOnly` CUDA Extra op for Llama3.2-3B PIC MLP linears.
- The op keeps compact active rows in raw NHWC `[rows,1,1,C]`, forces NHWC tensor descriptors at the CUDA boundary, and delegates to the existing INT4 weight-only Conv execution without transient alias tensors.
- Added `PicLinearNhwcWeightOnly` shape format propagation and CUDA native Extra routing.
- Added exporter guard: NHWC linear rewrite only emits when `ic % 8 == 0` and `oc % 8 == 0`.
- Added clone-safe static dequant sharing for external INT4 1x1 Conv resources, keyed by CUDA runtime + external weight offsets + padded shape. This was required for endpoint wins because decode repair clones otherwise exhausted the static dequant cache and fell back to runtime dequant.

## Validation

- Build artifact:
  - local: `.cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear`
  - remote: `/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_nhwc_linear`
- Direct accuracy on Jetson passed:
  - log: `.cache/bench_ops/nhwc_linear_static_share/rows45_piclinear_accuracy.log`
  - rows4/5 cuBLAS cases and NHWC rows1/2/4/6/8 all report `bad=0`.
- Profile proof after static-share fix:
  - remote log: `.cache/logs/nhwc_linear_runtime_key_profile_20260630_025207.log`
  - local copy: `.cache/logs/jetson_nhwc_linear/nhwc_linear_runtime_key_profile_20260630_025207.log`
  - `PicLinearNhwcWeightOnly` endpoint hits: `588`
  - decode repair batches hit with static dequant:
    - `batch=4 static_dequant=1`: `84`
    - `batch=6 static_dequant=1`: `84`
    - `batch=8 static_dequant=1`: `84`
  - no `ERROR`, resize failure, empty output, or illegal access signatures.
- Fresh graph-profile attribution logs:
  - baseline: `.cache/logs/jetson_nhwc_linear/nhwc_linear_baseline_profile_20260630_030553.log`
  - NHWC: `.cache/logs/jetson_nhwc_linear/nhwc_linear_nhwc_profile_fresh_20260630_031000.log`
  - These logs are only for attribution because `MNN_PIC_GRAPH_PROFILE=1` synchronizes after ops. Formal latency below uses non-profile runs.
- Correctness: baseline and NHWC raw responses match for `x=0/1/3/5/7`; runtime is `mnn_token_id_sparse_decode`.

Active row mapping from raw responses:

| repair tokens | active rows | refined tokens over 8 decode steps |
|--------------:|------------:|-----------------------------------:|
| 0 | 1 | 0 |
| 1 | 2 | 8 |
| 3 | 4 | 24 |
| 5 | 6 | 40 |
| 7 | 8 | 56 |

## Endpoint Result

Matched non-profile endpoint timing, Jetson CUDA, context 1024, `full-reuse`, `lagged_attention_hkvd`, `max_tokens=8`, `repeats=3`, `warm_repeats=1`.

| repair tokens | baseline TPOT ms | NHWC TPOT ms | delta |
|--------------:|-----------------:|-------------:|------:|
| 0 | 81.7958 | 81.3812 | -0.51% |
| 1 | 92.1576 | 92.0780 | -0.09% |
| 3 | 118.1139 | 118.1269 | +0.01% |
| 5 | 120.1374 | 119.6562 | -0.40% |
| 7 | 122.1457 | 120.8482 | -1.06% |

## Profile Breakdown

Fresh graph-profile smoke, x=3/5/7, `max_tokens=1`, `repeats=1`, same artifact and shared KV. Units are synchronized graph-profile milliseconds, not formal latency.

| x | run | attention | attn dense | MLP dense | MLP act+mul | raster/preprocess |
|--:|-----|----------:|-----------:|----------:|------------:|------------------:|
| 3 | baseline | 81.006 | 140.184 | 85.599 | 3.773 | 16.392 |
| 3 | NHWC | 84.453 | 125.024 | 90.570 | 4.617 | 15.599 |
| 5 | baseline | 87.479 | 46.021 | 82.840 | 3.705 | 15.122 |
| 5 | NHWC | 88.059 | 45.478 | 89.061 | 3.293 | 16.986 |
| 7 | baseline | 75.593 | 45.610 | 82.747 | 3.513 | 15.272 |
| 7 | NHWC | 74.662 | 48.578 | 94.003 | 3.382 | 16.644 |

The profile proves endpoint path hit and compact shape routing rather than a large graph-profile MLP category win: baseline MLP linears show as `Convolution` with NCHW-like compact inputs, while NHWC shows `Extra` `PicLinearNhwcWeightOnly` with `[rows,1,1,C]` inputs. Each x=3/5/7 profile has 84 NHWC MLP wrapper calls and all 84 use static dequant.

Direct `LinearConvertChain` bench isolates the intended dense-side change. For rows 4/6/8, NHWC removes the explicit convert chain and is faster for gate/up and rows 4/6 down, while rows 8 down is neutral/slightly slower:

| rows | op | with convert ms | NHWC ms | ratio |
|-----:|----|----------------:|--------:|------:|
| 4 | gate/up | 0.5097 | 0.5000 | 0.981 |
| 4 | down | 0.5411 | 0.5233 | 0.967 |
| 6 | gate/up | 0.5015 | 0.4925 | 0.982 |
| 6 | down | 0.5218 | 0.5124 | 0.982 |
| 8 | gate/up | 0.4912 | 0.4858 | 0.989 |
| 8 | down | 0.5084 | 0.5124 | 1.008 |

## Decision

Accept the NHWC linear path with static dequant sharing as a real compact-dense endpoint optimization for Jetson CUDA decode repair. It proves endpoint path hit and correctness, gives wins for `x=5` and `x=7`, keeps `x=0/1` non-regressing, and leaves `x=3` effectively neutral.

This is not a full gate/up/SILU/down one-kernel MLP fusion. The accepted scope is a guarded compact NHWC dense path that removes the old explicit convert-chain pattern for validated small-row MLP linears and keeps cloned decode repair linears on static dequant. Remaining risk is that synchronized graph-profile MLP totals are noisy and do not show a broad category reduction; future work should target true gate/up/SILU/down fusion or a tensor-core weight-only compact GEMM if larger endpoint gains are required.
