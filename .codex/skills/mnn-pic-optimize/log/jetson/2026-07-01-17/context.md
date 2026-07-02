# Context

## Models

Fresh shapefix model:

```text
.cache/weight/AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-gateup_packed_cuda_fresh_shapefix_20260701
```

Baseline model:

```text
.cache/weight/AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-silumul-score10
```

Both were served by the same Jetson CUDA artifact:

```text
/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda
```

The server was restarted without graph/profile env flags for each side. The benchmark client used:

```text
device=jetson
backend=cuda
frequency_profile=max
mode=full-reuse
contexts=512,1024,1536
repair_tokens=0,1,3,5,7
decode_selector=top_hkvd
max_tokens=32
warm_repeats=1
repeats=3
require_exact_context=true
NO_PROXY=192.168.101.192,localhost,127.0.0.1
```

## Prefill Failure Repair

The earlier fresh gate/up packed failure happened during the `/v1/prefill/text` prefill stage that builds the persistent text cache, before chat decode.

Two issues were isolated:

1. CUDA shape contract:
   - `PicLinearNhwcWeightOnly` was inferred to a 3D tensor in the fresh chain.
   - CUDA execution expects the packed NHWC path to keep 4D NHWC layout.
   - Shape inference was adjusted so this op preserves the input rank and rewrites only the last dimension to `oc`.

2. Invalid MNN binary op types:
   - The fresh `llm.mnn` contained `type=-1` for PIC attention ops at runtime.
   - Current JSON-to-MNN conversion preserves `PicScoreAttention` and `PicSparseAttention`.
   - Rebuilding the fresh model's `llm.mnn` from its JSON using the current converter fixed the invalid op types.

After these repairs, a 512-token `/v1/prefill/text` smoke returned HTTP 200 with:

```text
cache_status=built
backend=mnn_cuda
```

## End-To-End Decode A/B

Run root:

```text
.cache/bench_ops/mlp_fusion_20260701/fresh_gateup_shapefix_ab_20260701_091744
```

Raw CSVs:

```text
fresh_decode.csv
silumul_decode.csv
```

All rows succeeded:

```text
fresh:   15 ok rows
silumul: 15 ok rows
```

No server `ERROR`, `failed`, `unsupported`, `type=-1`, or `MNN_PIC_GRAPH_PROFILE` lines were found in the local run logs.

### Per-Context TPOT

| ctx | x | fresh ms | silumul ms | fresh - baseline |
|---:|---:|---:|---:|---:|
| 512 | 0 | 29.842 | 28.847 | +0.995 |
| 512 | 1 | 30.738 | 27.724 | +3.014 |
| 512 | 3 | 37.754 | 34.943 | +2.812 |
| 512 | 5 | 38.211 | 35.191 | +3.021 |
| 512 | 7 | 39.554 | 36.933 | +2.621 |
| 1024 | 0 | 32.827 | 32.577 | +0.251 |
| 1024 | 1 | 35.501 | 32.827 | +2.674 |
| 1024 | 3 | 43.332 | 40.608 | +2.723 |
| 1024 | 5 | 41.667 | 39.431 | +2.236 |
| 1024 | 7 | 44.205 | 41.662 | +2.542 |
| 1536 | 0 | 36.205 | 36.370 | -0.165 |
| 1536 | 1 | 40.309 | 38.037 | +2.271 |
| 1536 | 3 | 48.494 | 46.329 | +2.165 |
| 1536 | 5 | 45.883 | 43.774 | +2.108 |
| 1536 | 7 | 48.641 | 46.681 | +1.961 |

### Interpretation

The fresh packed gate/up path is not a production win:

- `x=0` is roughly tied, with small context-dependent noise.
- For repair rows `x>0`, fresh is consistently slower by about `+2.0..+3.0 ms/token`.
- The `x=1 -> x=3` jump stays around `+7.7 ms/token` for both paths.

Packed gate/up can reduce one graph shape boundary, but it does not remove the main decode repair cost. Once x increases, every decode step carries more active rows through attention, output projection, MLP gate/up, activation, and down projection across layers. The current fresh implementation also does not show a gate/up projection saving large enough to offset its layout/op overhead.

## Lesson

Do not reason from "one fewer gate/up Linear launch" to a production decode improvement. For CUDA/Jetson, gate/up packed must be treated as a graph/export change with two mandatory gates:

1. fresh export must build persistent text cache through `/v1/prefill/text`;
2. endpoint TPOT must beat the current verified decode path for the same model, context, selector, `max_tokens`, and `x`.

Until both gates pass, the production path remains `silumul-score10`.
