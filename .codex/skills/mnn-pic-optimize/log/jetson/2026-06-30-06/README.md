# Jetson CUDA Decode Repair Packed Gate/Up

## Summary

- Implemented and validated the shared packed NHWC gate/up path for Jetson CUDA `lagged_attention_hkvd` decode repair.
- Export now lowers MLP gate/up/silu to one `PicLinearNhwcWeightOnly` with `out_features=16384`, followed by `PicPackedSiluMul packed_input_nhwc=1`; `down_proj` stays on the existing `PicLinearNhwcWeightOnly` path.
- This is one shared graph/export/runtime path for all active row counts. There are no x=3/x=5/x=7 one-off operators or per-x exporter branches.
- Warm formal endpoint TPOT improved for every requested x and hit the stretch target:

| x | old ms | new ms | delta |
|--:|-------:|-------:|------:|
| 0 | 81.381 | 71.367 | -10.014 |
| 1 | 92.078 | 80.065 | -12.013 |
| 3 | 118.127 | 103.547 | -14.580 |
| 5 | 119.656 | 105.557 | -14.099 |
| 7 | 120.848 | 106.806 | -14.042 |

## Evidence

- Direct op accuracy on Jetson passed for rows4/6/8 Llama3.2-3B gate/up/down and `PicLinearNhwcWeightOnly` rows1/2/4/6/8:
  - remote log: `.cache/bench_ops/rows45_piclinear_accuracy_20260630_1416.log`
- Formal non-profile endpoint run:
  - local CSV: `.cache/mnn-pic-benchmark/nhwc_gateup_x01357_repeat_20260630_135721/decode.csv`
- Graph/profile attribution run:
  - local CSV: `.cache/mnn-pic-benchmark/nhwc_gateup_x01357_profile_max1_20260630_140845/decode.csv`
  - local log copy: `.cache/logs/jetson_nhwc_gateup/profile_max1_20260630_140845.log`
- Exported model graph check:
  - `PicLinearNhwcWeightOnly=56`
  - `PicPackedSiluMul=28`
  - `packed_input_nhwc=28`
  - `concat_conv=0`

## Decision

Packed gate/up is accepted as the P0 endpoint fix: it removes the 110+ms cliff for x=3/5/7 under the warm formal endpoint protocol. The remaining MLP work is P1: `down_proj` still costs about `15.7-17.1 ms` in the max1 low-level profile for rows4/6/8, so the next optimization should target down-specific compact dense scheduling or a larger MLP fusion that streams the activation into down accumulation.
