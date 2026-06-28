# Jetson PIC Decode Repair Optimize 2026-06-28 13

## Packed Gate/Up Direct-Op Check

Added a measurement-only path in `test/bench_ops/cuda/CudaWeightOnlyConvPerf.cpp`:

```text
bench=bench_ops/cuda/perf/PicDecodeMlp
new row=PicDecodeMlpPacked
path=hidden -> 2*inter concat Conv -> PicPackedSiluMul -> down
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda_test
remote logs=.cache/bench_ops/decode_repair_dense/
```

Longer-repeat result:

```text
rows=4 split chain=2.6012 ms packed chain=2.5848 ms delta=-0.0164 ms
rows=6 split chain=1.5541 ms packed chain=1.4878 ms delta=-0.0663 ms
rows=8 split chain=1.5025 ms packed chain=1.5155 ms delta=+0.0130 ms
```

Decision: reject as a production/default path. The signal is only tens of microseconds per layer and not consistent across rows. Do not switch exporter or `PicGateUpWeightOnly` to packed gate/up based on this result.

## Rows4-8 Dense Policy Diagnostics

Additional direct-op diagnostics were run for default / `down` / `0` rows45 cuBLAS policies and individual projection families.

Important interpretation:

```text
direct-op PicDecodeMlp numbers are diagnostic only
server x=3 profile remains authoritative for production path attribution
down-only and minDim=2048 were already rejected in jetson/2026-06-28-09
blanket rows45 cublas=0 was rejected in jetson/2026-06-28-12
```

The direct-op process does not reproduce server cuBLAS timings: direct `PicDecodeMlp` profile shows rows45 cuBLAS at multi-ms, while server x=3 profile shows the same MLP shapes at roughly `0.56..0.73 ms` with full model static-dequant cache already resident. Therefore these direct-op policy numbers are not sufficient for a default change.

Current standing result remains:

```text
x=0 none                  TPOT=86-89 ms
x=1 lagged_attention_hkvd TPOT=97-99 ms
x=3 lagged_attention_hkvd TPOT=123-125 ms
x=5 lagged_attention_hkvd TPOT=125-126 ms
x=7 lagged_attention_hkvd TPOT=126-128 ms
```

Next useful direction: stop retuning rows45 env policies and avoid packed gate/up as mainline. The remaining gap needs a real production-path reduction in compact dense or graph overhead, measured by server profile and validated by x=0/1/3/5/7 end-to-end TPOT.
