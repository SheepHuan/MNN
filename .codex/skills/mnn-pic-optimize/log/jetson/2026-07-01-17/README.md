# Jetson Fresh Gate/Up Packed Shapefix A/B

## Summary

- Fixed the fresh CUDA gate/up packed `/v1/prefill/text` failure enough to build persistent text cache on Jetson.
- The failure had two separate causes:
  - `PicLinearNhwcWeightOnly` shape inference must preserve CUDA's 4D NHWC execution contract.
  - The fresh `llm.mnn` binary had invalid PIC attention op types after an old JSON-to-MNN roundtrip; rebuilding `llm.mnn` from the current schema/converter restored `PicScoreAttention` and `PicSparseAttention`.
- The repaired fresh model completed end-to-end `full-reuse` decode TPOT for Llama3.2 1B, contexts `512/1024/1536`, `x=0/1/3/5/7`.
- Production decision: do not promote fresh gate/up packed. It is slower than the verified `silumul-score10` path at every averaged `x`, and it does not reduce the `x=1 -> x=3` cliff.

## Artifacts

- Run root: `.cache/bench_ops/mlp_fusion_20260701/fresh_gateup_shapefix_ab_20260701_091744`
- Fresh CSV: `fresh_decode.csv`
- Baseline CSV: `silumul_decode.csv`
- Summary: `ab_summary_by_x.tsv`
- Context pivot: `ab_pivot_tpot_ms.tsv`
- Cliff detail: `ab_x1_x3_cliff.tsv`

## Result

Average TPOT across contexts `512/1024/1536`:

| x | fresh gate/up packed ms | silumul-score10 ms | fresh - baseline | baseline/fresh |
|---:|---:|---:|---:|---:|
| 0 | 32.958 | 32.598 | +0.361 | 0.989x |
| 1 | 35.516 | 32.863 | +2.653 | 0.925x |
| 3 | 43.193 | 40.627 | +2.567 | 0.941x |
| 5 | 41.920 | 39.465 | +2.455 | 0.941x |
| 7 | 44.133 | 41.759 | +2.375 | 0.946x |

The averaged `x=1 -> x=3` jump is essentially unchanged:

```text
fresh gate/up packed: +7.677 ms
silumul-score10:      +7.764 ms
```

This means packed gate/up does not address the decode repair cliff. The cliff still tracks the extra active repair rows through the per-layer MLP/attention path, not just the separate gate/up projection launch count.

## Guardrail

Fresh CUDA gate/up packed is allowed to remain an experimental/export path only. A future attempt must pass both gates before production:

1. `/v1/prefill/text` text-cache build must pass on a fresh export without manual binary repair.
2. Real endpoint decode TPOT must beat the current verified baseline for `x=0/1/3/5/7` across multiple contexts.
