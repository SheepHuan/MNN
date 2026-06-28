# Context

## Profile Attribution

Profile input:

```text
.cache/mnn-pic-benchmark/decode_repair_current_graph_profile_x0x1x3x5x7_20260628_1412/
```

Request mapping:

```text
request 2 = x0
request 3 = x1
request 4 = x3
request 5 = x5
request 6 = x7
```

Type totals:

| x | Convolution ms | PicSparseAttention ms | PagedAttention ms |
|---|---------------:|----------------------:|------------------:|
| 0 | 193.291 | 107.156 | 5.069 |
| 1 | 213.963 | 108.420 | 28.723 |
| 3 | 374.950 | 104.184 | 17.307 |
| 5 | 291.009 | 106.740 | 18.235 |
| 7 | 287.124 | 109.059 | 13.003 |

Grouped Convolution totals and deltas vs x0:

| group | x0 ms | x5 delta ms | x7 delta ms |
|-------|------:|------------:|------------:|
| MLP gate/up/down | 115.745 | +39.993 | +37.148 |
| attention q/o/k/v | 16.931 | +44.074 | +43.129 |
| lm_head | 15.939 | -0.244 | -0.357 |

x=3 had an `attn_k` outlier; x=5/x=7 are the more stable shape representatives. The stable conclusion is that x=3/5/7 are close because rows=4/6/8 compact dense has a mostly fixed cost plateau.

## Direct-Op Evidence

Current direct-op artifact:

```text
remote: /home/jetson/code/kvshare-edge/impl/MNN/.cache/build/mnn/jetson_cross_current/
logs:
  .cache/bench_ops/weightonly_rows468_policies_20260628_144758.log
  .cache/bench_ops/pic_decode_mlp_rows1468_20260628_144852.log
  .cache/bench_ops/weightonly_rows12468_policy3_20260628_144934.log
```

Policy 3 `WeightOnlyConv` selected values:

| rows | hidden->inter ms | inter->hidden ms | hidden->hidden ms | hidden->kv ms |
|------|-----------------:|-----------------:|------------------:|--------------:|
| 1 | 0.287 | 0.258 | 0.117 | 0.046 |
| 4 | 0.483 | 0.533 | 0.208 | 0.068 |
| 6 | 0.480 | 0.508 | 0.198 | 0.063 |
| 8 | 0.487 | 0.512 | 0.201 | 0.068 |

`PicDecodeMlp` chain with policy 3:

| rows | split chain ms | packed chain ms |
|------|---------------:|----------------:|
| 1 | 0.855 | 0.851 |
| 4 | 1.556 | 1.537 |
| 6 | 1.578 | 1.553 |
| 8 | 1.575 | 1.522 |

Pure FP16 cuBLAS MLP floor is not lower for these rows; the current INT4 static-dequant path is already competitive or faster. Existing packed gate/up shape saves only tens of microseconds per layer, not enough to close the 17-21 ms remaining gap.

## Code Change

Changed `source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu`:

- Default `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS` policy is now `3`.
- Added explicit `mlp` alias for policy `2`.
- `all` now maps to policy `3`, covering MLP plus attention projection-like rows=4..8 linears.

This does not change PagedCache, selector, scoring, or sparse attention semantics.

## Verification

Build:

```bash
JOBS=96 MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda" \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Synced to Jetson artifact:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

No-profile decode repair confirm:

```text
.cache/mnn-pic-benchmark/decode_repair_default_policy3_confirm_20260628_150247/summary.csv
```

| x | TPOT ms |
|---|--------:|
| 0 | 81.22 |
| 1 | 89.56 |
| 3 | 116.78 |
| 5 | 118.45 |
| 7 | 121.13 |

## Next Optimization Direction

To get x=3/5/7 under 100 ms, the next real work should target a lower fixed-cost rows=4/6/8 dense path:

- MLP gate/up/down rows=4/6/8.
- attention q/o/k/v rows=4/6/8.

Further cuBLAS algo/compute-type tuning is unlikely to provide the missing 17-21 ms. A meaningful next step would need either a better compact tensor-core path for these tiny-M linears or a graph-level reduction in projection count that is proven on direct-op first.

