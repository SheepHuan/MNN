# Jetson PIC Decode Repair Optimize Context 2026-06-28 10

## Starting Point

User goal remains: reduce Jetson CUDA decode repair TPOT for `x=1/3/5/7 lagged_attention_hkvd` toward `x=0 none`, while preserving an `x=0` normal decode control and writing optimization logs hourly under `.codex/skills/mnn-pic-optimize/log/jetson/<YYYY-MM-DD-HH>/`.

Accepted current baseline:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_smoke2_20260628_0710/summary.csv
x=0 none                  TPOT=86.377 ms
x=1 lagged_attention_hkvd TPOT=97.471 ms
x=3 lagged_attention_hkvd TPOT=123.602 ms
x=5 lagged_attention_hkvd TPOT=124.975 ms
x=7 lagged_attention_hkvd TPOT=126.360 ms
```

The 09:00 hour tested rows4-8 cuBLAS policy, compute precision, k/v exclusion, and algo policy. The only initially promising diagnostic env was:

```text
MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO=default
```

But converting that into a no-env source default did not reproduce the gain:

```text
source-default no-env mean vs accepted baseline:
x=0 +0.526 ms
x=1 +0.653 ms
x=3 +0.535 ms
x=5 +1.004 ms
x=7 +2.151 ms
```

Decision carried into this hour: reject the source default, keep `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_ALGO=default` as diagnostics only, and restore tensor-op default behavior.

## Restore Build

Restored source state:

```text
picRows45CublasAlgoPolicy():
  no env / "tensor" -> CUBLAS_GEMM_DEFAULT_TENSOR_OP
  "default"         -> CUBLAS_GEMM_DEFAULT
```

Temporary `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_MIN_DIM` hook is removed.

Build command:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Build result:

```text
[93%] Built target MNN_Cuda_Main
[93%] Built target MNN
[95%] Built target MNN_Express
[100%] Built target pic_llm
[100%] Built target pic_server
[build_artifacts] Installed libMNN.so
[build_artifacts] Installed libMNN_Express.so
[build_artifacts] Installed libMNN_Cuda_Main.so
[build_artifacts] Installed libpic_llm.so
[build_artifacts] Installed pic_server
[build_artifacts] Done
```

Immediate next verification:

1. `file` check aarch64 artifacts.
2. `rsync` `.cache/output/mnn/artifacts/jetson/` to Jetson `jetson_cross_cuda`.
3. Restart remote server on port `18131` with no graph profile and no rows45 algo env.
4. Use the local `19131` tunnel.
5. Run decode repair smoke:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://127.0.0.1:19131 \
  --output-csv .cache/mnn-pic-benchmark/decode_repair_restored_default_20260628_10/summary.csv \
  --output-dir .cache/mnn-pic-benchmark/decode_repair_restored_default_20260628_10/client \
  --device jetson --device-display Jetson --backend cuda --frequency-profile max \
  --model llama-pic --model-name 'Llama3.2 3B' \
  --model-config .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json \
  --contexts 1024 --budgets 0.00 --decode-selectors lagged_attention_hkvd \
  --repair-tokens 0,1,3,5,7 --attention-layer-idx 1 --top-m 32 \
  --max-tokens 8 --repeats 3 --warm-repeats 1 --no-update-cache
```

If restore is clean, continue with kernel-level attribution. Current likely targets are stable `Convolution`/MLP, `PicSparseAttention`, and `Raster` overhead; qtile sparse attention remains unsuitable as a default for tiny decode repair rows.

## Restore Default TPOT Smoke Result

Artifact check:

```text
.cache/output/mnn/artifacts/jetson/bin/pic_server:          ELF 64-bit LSB executable, ARM aarch64
.cache/output/mnn/artifacts/jetson/lib/libMNN.so:           ELF 64-bit LSB shared object, ARM aarch64
.cache/output/mnn/artifacts/jetson/lib/libMNN_Cuda_Main.so: ELF 64-bit LSB shared object, ARM aarch64
.cache/output/mnn/artifacts/jetson/lib/libpic_llm.so:       ELF 64-bit LSB shared object, ARM aarch64
```

Server:

```text
remote artifact=/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda
remote log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_restored_default_20260628_10.log
server env=LD_LIBRARY_PATH only
profile env=off
rows45 algo env=off
/v1/models HTTP 200
```

Benchmark:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_restored_default_20260628_10/summary.csv
x=0 none                  TPOT=87.56104761904761 ms
x=1 lagged_attention_hkvd TPOT=98.37571428571430 ms
x=3 lagged_attention_hkvd TPOT=123.94528571428572 ms
x=5 lagged_attention_hkvd TPOT=124.63547619047620 ms
x=7 lagged_attention_hkvd TPOT=126.48890476190475 ms
```

Delta vs accepted baseline:

```text
x=0 +1.184 ms
x=1 +0.905 ms
x=3 +0.343 ms
x=5 -0.340 ms
x=7 +0.129 ms
```

Decision:

- Restore is accepted as the current working default.
- Do not keep the rejected source-default `algo=default` change.
- Continue optimizing from this source state.

Next focus:

- x=1 adds about `10.8 ms` over normal decode; this is the fixed repair overhead to reduce first.
- x=3/5/7 are clustered around `124-126 ms`, so the current path likely pays a shape/kernal-family overhead after repair becomes active rather than purely linear per repaired token.
- Re-profile default source if needed, but avoid treating single cold `Linear` spikes as optimization targets.

## Short x=1/x=7 Profile

Profile server:

```text
env=MNN_PIC_DECODE_REPAIR_PROFILE=1
env=MNN_PAGED_ATTENTION_PROFILE=1
log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_profile_20260628_10.log
local copy=.cache/mnn-pic-benchmark/decode_repair_profile_x1_x7_20260628_10/server.log
```

Benchmark:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_profile_x1_x7_20260628_10/summary.csv
x=1 lagged_attention_hkvd generated=3 TPOT=190.347 ms ok
x=7 lagged_attention_hkvd generated=3 TPOT=330.377 ms ok
```

`MNN_PIC_DECODE_REPAIR_PROFILE` rows:

```text
x=1 step0 total=173.039 ms forward_raw=172.575 ms
x=1 step1 total=102.328 ms forward_raw=102.249 ms
x=1 step2 total=105.005 ms forward_raw=104.846 ms

x=7 step0 total=395.585 ms forward_raw=395.421 ms
x=7 step1 total=129.574 ms forward_raw=129.416 ms
x=7 step2 total=135.326 ms forward_raw=135.211 ms
```

Observations:

- The first profiled step includes cold/profile noise; steady steps matter more.
- Host-side rows/mask/embedding/rank/bookkeeping are not the main bottleneck. For steady steps, non-`forward_raw` stages are around `0.1 ms`.
- Fused `decode_attention_rank` is small: 6 matches, average `~0.252 ms`, `topk_avg ~0.141 ms`, `d2h_avg ~0.049 ms`.

Parsed dense profile across the short run:

```text
x=1-like batch=2 tiny_gemv:
  3072->8192 total=63.467 ms calls=167 avg=380 us
  8192->3072 total=30.711 ms calls=84  avg=366 us
  3072->3072 total=30.380 ms calls=167 avg=182 us
  3072->1024 total=16.617 ms calls=167 avg=100 us

x=7-like batch=8 rows45_cublas:
  3072->8192 total=96.651 ms calls=167 avg=579 us
  8192->3072 total=76.995 ms calls=84  avg=917 us (contains cold spikes; many steady lines are ~600 us)
  3072->1024 total=87.560 ms calls=168 avg=521 us (contains a cold spike; many steady lines are ~130 us)
```

Parsed attention profile:

```text
sparse_flash_qtile_attention query=2 attn=2 total=71.149 ms calls=84 avg=847 us
sparse_flash_qtile_attention query=8 attn=8 total=78.091 ms calls=83 avg=941 us
decode_causal_attention query=1 attn=1 total=37.918 ms calls=56 avg=677 us
```

Conclusion:

- x=1 is not dominated by attention-rank CPU/D2H. It is dominated by full model `forwardRaw`, with tiny-row dense and sparse attention as the largest visible pieces.
- x=7 already benefits from rows4-8 cuBLAS. The remaining plateau is a mix of rows4-8 cuBLAS dense, sparse qtile attention, and normal decode/lm_head.
- Next test should target x=1 by trying cuBLAS for batch=2/3 MLP-like static-dequant linears only. This needs an env gate first because non-MLP projections should stay on the current tiny GEMV unless proven faster.

## Rows2-3 cuBLAS A/B

Temporary source change:

```text
MNN_CUDA_PIC_INT4_ROWS23_CUBLAS=1 allows picRows45CublasMatches() to match batch=2/3.
Only decode repair sparse, fp16, static-dequant, MLP-like INT4 1x1 Linear can take it.
Default no-env path remains batch>=4.
```

Build:

```text
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1
result=success
outputs=libMNN.so, libMNN_Express.so, libMNN_Cuda_Main.so, libpic_llm.so, pic_server
```

No-env control:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows23_control_20260628_10/summary.csv
x=0 none                  TPOT=87.3915238095238 ms
x=1 lagged_attention_hkvd TPOT=99.12485714285714 ms
x=3 lagged_attention_hkvd TPOT=124.62457142857143 ms
x=5 lagged_attention_hkvd TPOT=126.41138095238095 ms
x=7 lagged_attention_hkvd TPOT=127.80500000000000 ms
```

Rows2-3 cuBLAS env:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows23_cublas_20260628_10/summary.csv
x=0 none                  TPOT=87.62361904761906 ms
x=1 lagged_attention_hkvd TPOT=118.33190476190475 ms
x=3 lagged_attention_hkvd TPOT=122.65823809523810 ms
x=5 lagged_attention_hkvd TPOT=125.14866666666667 ms
x=7 lagged_attention_hkvd TPOT=127.54328571428572 ms
```

Delta:

```text
x=0 rows23-control +0.232 ms; rows23-accepted +1.247 ms
x=1 rows23-control +19.207 ms; rows23-accepted +20.860 ms
x=3 rows23-control -1.966 ms; rows23-accepted -0.943 ms
x=5 rows23-control -1.263 ms; rows23-accepted +0.174 ms
x=7 rows23-control -0.262 ms; rows23-accepted +1.184 ms
```

Decision:

- Reject. The target case x=1/sparseRows=2 becomes much worse.
- Do not interpret x=3 as a win: sparseRows=4 was already covered by existing rows4-8 cuBLAS, so this is run-to-run noise or secondary effects.
- Remove `MNN_CUDA_PIC_INT4_ROWS23_CUBLAS` hook and rebuild/sync a clean default artifact.

## Default Artifact Restored After Rejected A/B

Cleanup:

```text
source check: no `ROWS23_CUBLAS` / `picRows23` matches in ConvFpAIntBExecution.cu
build: success
artifact file: aarch64 pic_server / libMNN_Cuda_Main.so / libpic_llm.so
rsync: .cache/output/mnn/artifacts/jetson/ -> Jetson jetson_cross_cuda
server env: LD_LIBRARY_PATH only
profile env: off
rows45/rows23 env: off
```

Final default smoke:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_default_final_20260628_10/summary.csv
x=0 none                  TPOT=88.47790476190477 ms
x=1 lagged_attention_hkvd TPOT=99.51514285714286 ms
x=3 lagged_attention_hkvd TPOT=124.97728571428571 ms
x=5 lagged_attention_hkvd TPOT=125.02028571428572 ms
x=7 lagged_attention_hkvd TPOT=128.03728571428573 ms
```

Delta vs accepted baseline:

```text
x=0 +2.101 ms
x=1 +2.044 ms
x=3 +1.376 ms
x=5 +0.046 ms
x=7 +1.678 ms
```

Conclusion:

- This hour did not land a default speedup.
- The rejected batch2/3 cuBLAS path confirms x=1 should stay on tiny GEMV for now.
- Remaining plausible targets are:
  - sparse attention qtile for tiny repair rows, especially reducing the fixed ~0.85-0.94ms/layer attention cost;
  - non-attention graph overhead/Raster for sparseRows=2;
  - a true small-M fused dense path, not cuBLAS-on-dequant for M=2/3.

## Decode Repair QTile-Off A/B

Environment-only A/B:

```text
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=0
server log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_qtile_off_20260628_10.log
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_off_20260628_10/summary.csv
```

Results:

```text
x=0 none                  TPOT=87.00342857142857 ms
x=1 lagged_attention_hkvd TPOT=101.71633333333334 ms
x=3 lagged_attention_hkvd TPOT=136.68052380952380 ms
x=5 lagged_attention_hkvd TPOT=149.67938095238097 ms
x=7 lagged_attention_hkvd TPOT=171.01771428571430 ms
```

Delta vs default-final:

```text
x=0 -1.474 ms
x=1 +2.201 ms
x=3 +11.703 ms
x=5 +24.659 ms
x=7 +42.980 ms
```

Decision:

- Reject. The x=0 change is irrelevant noise because no repair qtile is active.
- Decode repair qtile is mandatory for x>=3 and still beneficial for x=1.
- Future attention work should tune qtile internals or variants, not fall back to row-compressed attention.

## Continuation State Before QTile Variant A/B

Timestamp:

```text
2026-06-28 10:55 CST
```

Working state:

```text
server=Jetson default/no profile/no rejected env
remote=jetson@192.168.101.192
remote_artifact=/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda
remote_port=18131
local_tunnel=19131
health=/v1/models HTTP 200
server_log=.cache/logs/pic_server_decode_repair_default_restored_after_qtile_20260628_10.log
```

Baseline to compare against remains:

```text
accepted=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_smoke2_20260628_0710/summary.csv
x=0  86.377 ms
x=1  97.471 ms
x=3 123.602 ms
x=5 124.975 ms
x=7 126.360 ms
```

Most recent default smoke:

```text
current=.cache/mnn-pic-benchmark/decode_repair_default_final_20260628_10/summary.csv
x=0  88.478 ms
x=1  99.515 ms
x=3 124.977 ms
x=5 125.020 ms
x=7 128.037 ms
```

Continuation plan:

1. Run `MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1`, `MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=2`, `MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q4k16` with x=0/1/3/5/7.
2. Run `MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1`, `MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=2`, `MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q4k8` with x=0/1/3/5/7.
3. Compare against both current default and accepted baseline.
4. Only if a variant improves x=1/3/5/7 without hurting x=0 beyond normal noise should it become a source-default candidate. Otherwise keep the current default and move to x=1 graph-overhead attribution.

Source note: `selectCudaDecodeRepairQTileVariant()` defaults decode repair to `hd128_q8k16` for `headDim=128, attnLen>=2`; the variant env is only honored in this path when `MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT` is present.

## QTile Variant A/B Result

Forced q4k16 env:

```text
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=2
MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q4k16
server_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_qtile_hd128_q4k16_20260628_10.log
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_hd128_q4k16_20260628_10/summary.csv
```

Result:

```text
x=0 none                  TPOT=86.51076190476190 ms
x=1 lagged_attention_hkvd TPOT=113.31347619047620 ms
x=3 lagged_attention_hkvd TPOT=139.73819047619048 ms
x=5 lagged_attention_hkvd TPOT=141.56528571428570 ms
x=7 lagged_attention_hkvd TPOT=143.69795238095240 ms
```

Delta vs current default-final:

```text
x=0  -1.967 ms
x=1 +13.798 ms
x=3 +14.761 ms
x=5 +16.545 ms
x=7 +15.661 ms
```

Forced q4k8 env:

```text
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_EXPERIMENT=1
MNN_CUDA_PAGED_ATTENTION_DECODE_REPAIR_QTILE_MIN_ROWS=2
MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=hd128_q4k8
server_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_qtile_hd128_q4k8_20260628_10.log
summary=.cache/mnn-pic-benchmark/decode_repair_qtile_hd128_q4k8_20260628_10/summary.csv
```

Result:

```text
x=0 none                  TPOT=86.39633333333335 ms
x=1 lagged_attention_hkvd TPOT=116.94747619047620 ms
x=3 lagged_attention_hkvd TPOT=143.09747619047620 ms
x=5 lagged_attention_hkvd TPOT=144.28238095238095 ms
x=7 lagged_attention_hkvd TPOT=147.95423809523808 ms
```

Delta vs current default-final:

```text
x=0  -2.082 ms
x=1 +17.432 ms
x=3 +18.120 ms
x=5 +19.262 ms
x=7 +19.917 ms
```

Decision:

- Reject both q4 variants.
- The `x=0` improvement is not meaningful because normal decode does not execute repair qtile.
- All repair cases strongly prefer the current decode repair default `hd128_q8k16`.
- Next optimization should not retune only Q/K tile sizes. Move to x=1 graph-level attribution to see whether compact-row dense, Raster/gather, or other non-attention kernels explain the remaining `~11-13 ms` gap over `x=0`.


Benchmark command template:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://127.0.0.1:19131 \
  --output-csv <summary.csv> \
  --output-dir <client_dir> \
  --device jetson --device-display Jetson --backend cuda --frequency-profile max \
  --model llama-pic --model-name 'Llama3.2 3B' \
  --model-config .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json \
  --contexts 1024 --budgets 0.00 --decode-selectors lagged_attention_hkvd \
  --repair-tokens 0,1,3,5,7 --attention-layer-idx 1 --top-m 32 \
  --max-tokens 8 --repeats 3 --warm-repeats 1 --no-update-cache
```
