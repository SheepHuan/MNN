# Jetson PIC Decode Repair Optimize 2026-06-28 10

## 10:20 Restore Default Build Complete

Continuation from `jetson/2026-06-28-09`: the source-default attempt for `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO=default` was rejected because no-env verification regressed x=0/1/3/5/7 versus the accepted rows48 baseline. Source was restored so rows4-8 cuBLAS uses tensor-op default again unless the diagnostic env explicitly asks for cuBLAS default.

The restore cross-build completed successfully:

```text
target=pic_server
artifact=.cache/output/mnn/artifacts/jetson
outputs=libMNN.so, libMNN_Express.so, libMNN_Cuda_Main.so, libpic_llm.so, pic_server
```

Next: sync the restored artifact to Jetson, restart a default server without `MNN_PIC_GRAPH_PROFILE` and without rows45 algo env, then run the x=0/1/3/5/7 decode repair smoke to confirm the default path is back to the accepted baseline range.

## 10:26 Restore Default TPOT Smoke

Restored no-env default server:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_restored_default_20260628_10/summary.csv
x=0 none                  TPOT=87.561 ms
x=1 lagged_attention_hkvd TPOT=98.376 ms
x=3 lagged_attention_hkvd TPOT=123.945 ms
x=5 lagged_attention_hkvd TPOT=124.635 ms
x=7 lagged_attention_hkvd TPOT=126.489 ms
```

Decision: restore is good enough to continue optimizing. x=5 is slightly faster than accepted baseline, x=7 is within `+0.13 ms`, and x=3 is within `+0.34 ms`; x=0/x=1 are about `+1 ms` slower, likely run-to-run noise or cold shape variance. Continue with attribution for fixed decode-repair overhead and the x>=3 plateau.

## 10:34 Short Profile: x=1/x=7

Profile env:

```text
MNN_PIC_DECODE_REPAIR_PROFILE=1
MNN_PAGED_ATTENTION_PROFILE=1
```

Short run:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_profile_x1_x7_20260628_10/summary.csv
x=1 profile TPOT=190.347 ms
x=7 profile TPOT=330.377 ms
```

Profile-only conclusion:

- Host-side decode repair bookkeeping is not the bottleneck. Per step, build rows / rank-result / finish sparse / bookkeeping are all sub-ms; `forward_raw` dominates.
- Fused attention rank capture is small: about `0.18-0.44 ms` per step with `top_m=32`.
- x=1 steady profile steps spend about `103-105 ms` in `forward_raw`; x=7 steady steps spend about `129-135 ms`.
- Dense is a primary target. For x=1 batch=2, `tiny_gemv` totals across the short run were dominated by `3072->8192`, `8192->3072`, `3072->3072`, and `3072->1024` linears. For x=7 batch=8, rows4-8 cuBLAS is already active.

Next A/B: add a temporary env-gated path for decode repair batch=2/3 MLP-like static-dequant INT4 Linear to use cuBLAS instead of tiny GEMV, then rebuild and test x=1/3/5/7.

## 10:50 Rows2-3 cuBLAS A/B Rejected

Temporary source A/B:

```text
MNN_CUDA_PIC_INT4_ROWS23_CUBLAS=1
```

No-env control after rebuild:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows23_control_20260628_10/summary.csv
x=0 87.392 ms
x=1 99.125 ms
x=3 124.625 ms
x=5 126.411 ms
x=7 127.805 ms
```

Rows2-3 cuBLAS env:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows23_cublas_20260628_10/summary.csv
x=0 87.624 ms
x=1 118.332 ms
x=3 122.658 ms
x=5 125.149 ms
x=7 127.543 ms
```

Decision: reject and remove the temporary env hook. x=1 is the target for batch=2, and it regresses by `+19.2 ms` versus the no-env control. The apparent x=3 improvement is not attributable to rows2-3 cuBLAS because x=3 uses sparseRows=4, which already takes the existing rows4-8 path.

## 10:46 Default Artifact Restored

The temporary `ROWS23_CUBLAS` hook was removed and a clean default Jetson artifact was rebuilt, synced, and started without profile/env knobs.

Final smoke:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_default_final_20260628_10/summary.csv
x=0 none                  TPOT=88.478 ms
x=1 lagged_attention_hkvd TPOT=99.515 ms
x=3 lagged_attention_hkvd TPOT=124.977 ms
x=5 lagged_attention_hkvd TPOT=125.020 ms
x=7 lagged_attention_hkvd TPOT=128.037 ms
```

Delta vs accepted baseline:

```text
x=0 +2.101 ms
x=1 +2.044 ms
x=3 +1.376 ms
x=5 +0.046 ms
x=7 +1.678 ms
```

Decision: no default optimization landed in this hour. The current code/artifact is back on the default path, and the rejected rows2/3 cuBLAS experiment is only documented in the log.

## 10:53 Decode Repair QTile-Off A/B Rejected

Environment-only A/B:

```text
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=0
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_off_20260628_10/summary.csv
x=0 87.003 ms
x=1 101.716 ms
x=3 136.681 ms
x=5 149.679 ms
x=7 171.018 ms
```

Decision: reject. Disabling decode-repair qtile makes repair rows much slower, especially x=3/5/7. The current qtile default is necessary for this path.

## 10:55 Continuation: QTile Variant A/B Scope

Current state before continuing:

```text
server=Jetson default/no profile/no rejected env
remote_port=18131
local_tunnel=19131
/v1/models=HTTP 200
current_default_summary=.cache/mnn-pic-benchmark/decode_repair_default_final_20260628_10/summary.csv
```

Next action is environment-only tuning of the decode-repair qtile kernel variant while keeping `x=0 none` in every run as the normal decode control:

```text
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=2
MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q4k16
MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q4k8
```

Note: the decode repair path otherwise forces `hd128_q8k16`, so the experiment/min-rows env is required for the variant env to take effect on rows2/4/6/8.

The rejected paths remain rejected: rows2/3 cuBLAS and qtile-off should not be revisited unless a new profile shows a different bottleneck.

## 10:59 QTile Variant A/B Rejected

Both candidate variants were forced with:

```text
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=2
```

Results:

```text
hd128_q4k16 summary=.cache/mnn-pic-benchmark/decode_repair_qtile_hd128_q4k16_20260628_10/summary.csv
x=0  86.511 ms
x=1 113.313 ms
x=3 139.738 ms
x=5 141.565 ms
x=7 143.698 ms

hd128_q4k8 summary=.cache/mnn-pic-benchmark/decode_repair_qtile_hd128_q4k8_20260628_10/summary.csv
x=0  86.396 ms
x=1 116.947 ms
x=3 143.097 ms
x=5 144.282 ms
x=7 147.954 ms
```

Decision: reject both. `x=0` is normal decode noise, while every repair case regresses strongly versus current default q8k16. Keep decode repair default as `hd128_q8k16`.
