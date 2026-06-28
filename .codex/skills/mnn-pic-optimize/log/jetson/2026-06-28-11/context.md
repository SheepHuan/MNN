# Jetson PIC Decode Repair Optimize Context 2026-06-28 11

## Starting Point

Continuation from `jetson/2026-06-28-10`.

Accepted baseline:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_smoke2_20260628_0710/summary.csv
x=0 none                  TPOT=86.377 ms
x=1 lagged_attention_hkvd TPOT=97.471 ms
x=3 lagged_attention_hkvd TPOT=123.602 ms
x=5 lagged_attention_hkvd TPOT=124.975 ms
x=7 lagged_attention_hkvd TPOT=126.360 ms
```

Rejected in the previous hour:

```text
rows2/3 cuBLAS: x=1 regressed to 118.332 ms
qtile-off:      x=7 regressed to 171.018 ms
hd128_q4k16:    x=1/x3/x5/x7 regressed to 113.313/139.738/141.565/143.698 ms
hd128_q4k8:     x=1/x3/x5/x7 regressed to 116.947/143.097/144.282/147.954 ms
```

## x=1 Graph Profile Attribution

Profile server:

```text
env=MNN_PIC_GRAPH_PROFILE=1
env=MNN_PIC_GRAPH_PROFILE_TOP=1000
env=MNN_PIC_DECODE_REPAIR_PROFILE=1
server_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_graph_profile_x1_20260628_10.log
local_copy=.cache/mnn-pic-benchmark/decode_repair_graph_profile_x1_20260628_10/server.log
```

Profile benchmark:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://127.0.0.1:19131 \
  --output-csv .cache/mnn-pic-benchmark/decode_repair_graph_profile_x1_20260628_10/summary.csv \
  --output-dir .cache/mnn-pic-benchmark/decode_repair_graph_profile_x1_20260628_10/client \
  --device jetson --device-display Jetson --backend cuda --frequency-profile max \
  --model llama-pic --model-name 'Llama3.2 3B' \
  --model-config .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json \
  --contexts 1024 --budgets 0.00 --decode-selectors lagged_attention_hkvd \
  --repair-tokens 1 --attention-layer-idx 1 --top-m 32 \
  --max-tokens 3 --repeats 1 --warm-repeats 0 --no-update-cache
```

Profile latency, for attribution only:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_graph_profile_x1_20260628_10/summary.csv
x=1 lagged_attention_hkvd generated=3 TPOT=222.2095 ms
```

Decode-repair stage profile:

```text
step0 forward_raw=195.946 ms total=196.072 ms
step1 forward_raw=123.902 ms total=123.973 ms
step2 forward_raw=123.916 ms total=124.005 ms
```

Graph type profile:

```text
MNN_PIC_GRAPH_PROFILE_SUMMARY request=1 label=mode=full-reuse,ratio=0,score_layer=1,max_tokens=3 total_ms=448.094 calls=4774 unique_ops=2381 unique_types=10
Convolution        total_ms=210.781 max_ms=3.918 calls=788
PicSparseAttention total_ms=117.170 max_ms=13.902 calls=104
Raster             total_ms=49.873  max_ms=0.092 calls=1838
BinaryOp           total_ms=22.187  max_ms=0.131 calls=788
While              total_ms=16.970  max_ms=0.119 calls=560
UnaryOp            total_ms=9.244   max_ms=0.061 calls=344
LayerNorm          total_ms=6.681   max_ms=0.373 calls=228
PicScoreAttention  total_ms=5.935   max_ms=2.719 calls=4
PagedAttention     total_ms=5.673   max_ms=2.713 calls=4
Cast               total_ms=3.580   max_ms=0.329 calls=116
```

Top single ops:

```text
rank=1 Convolution /lm/lm_head/Linear total=15.522 ms calls=4
rank=2 PicSparseAttention /layers.2/self_attn/PagedAttention total=13.902 ms calls=1
rank=3 PicScoreAttention /layers.1/self_attn/PagedAttention total=3.216 ms calls=3
rank=4 PagedAttention /layers.0/self_attn/PagedAttention total=2.960 ms calls=3
rank=5 PicSparseAttention /layers.2/self_attn/PagedAttention total=2.758 ms calls=3
```

The `rank=2` one-off spike is profile/cold-shape noise; the repeated sparse attention calls are around `0.90-0.94 ms` per layer for batch=2.

Tiny GEMV aggregate from `CUDAWeightOnlyConv profile`:

```text
conv_fpa_intb_1x1_tiny_gemv batch=2 3072->8192   total=62.078 ms calls=164 avg=378.5 us
conv_fpa_intb_1x1_tiny_gemv batch=2 8192->3072   total=29.802 ms calls=82  avg=363.4 us
conv_fpa_intb_1x1_tiny_gemv batch=2 3072->3072   total=29.758 ms calls=165 avg=180.4 us
conv_fpa_intb_1x1_tiny_gemv batch=2 3072->1024   total=16.216 ms calls=165 avg=98.3 us
conv_fpa_intb_1x1_tiny_gemv batch=1 3072->128256 total=7.748 ms  calls=2   avg=3874.0 us
```

Conclusion:

- CPU-side decode repair scheduling is not the bottleneck; steady non-`forward_raw` work is about `0.1 ms`.
- The attention-rank compact output path is not the bottleneck.
- Remaining x=1 gap over x=0 is primarily real batch=2 transformer compute: tiny-M INT4 dense plus per-layer `PicSparseAttention`.
- `Raster` is visible but below dense/attention. It may be worth trimming after compute kernels, not before.
- Since q4k16/q4k8 and rows2/3 cuBLAS both regress, the next source candidate needs to be a true small-M fused dense path or an internal improvement to `hd128_q8k16` that preserves the current variant.

## Default Server Restored

After graph profile:

```text
server=Jetson default/no profile/no rejected env
remote_port=18131
local_tunnel=19131
/v1/models=HTTP 200
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_default_after_graph_profile_20260628_10.log
```

Next direction: inspect small-M INT4 dense and current `hd128_q8k16` internals. Do not revisit qtile-off, q4k16/q4k8, or rows2/3 cuBLAS.

## V14 MB OC=8 A/B Scope

Temporary source change:

```text
file=source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
env=MNN_CUDA_PIC_INT4_V14_MB_OC=8
default behavior=no env returns OC_PER_BLK=4
active only when pic_decode_repair_sparse && batch<=2 && V14_MB tiny GEMV path
```

Rationale:

- x=1 profile shows batch=2 tiny GEMV dominates dense cost.
- Existing rows2/3 cuBLAS A/B regressed x=1, so this should stay inside the current V14_MB kernel family.
- `OC_PER_BLK=8` halves OC blocks versus OC=4 but increases per-thread accumulator/register pressure. It is not safe as a default without Jetson data.

Required verification:

1. Cross-build `pic_server` artifact for Jetson.
2. Sync to Jetson.
3. Start no-env server and run x=0/1/3/5/7 as control to ensure source default unchanged.
4. Start `MNN_CUDA_PIC_INT4_V14_MB_OC=8` server and run x=0/1/3/5/7.
5. Reject unless x=1 improves versus same-build no-env control and x=0 remains noise-only.

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

Artifact check:

```text
.cache/output/mnn/artifacts/jetson/bin/pic_server:          ELF 64-bit LSB executable, ARM aarch64
.cache/output/mnn/artifacts/jetson/lib/libMNN.so:           ELF 64-bit LSB shared object, ARM aarch64
.cache/output/mnn/artifacts/jetson/lib/libMNN_Cuda_Main.so: ELF 64-bit LSB shared object, ARM aarch64
.cache/output/mnn/artifacts/jetson/lib/libpic_llm.so:       ELF 64-bit LSB shared object, ARM aarch64
```

Sync:

```text
rsync .cache/output/mnn/artifacts/jetson/ -> jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
result=success
```

## V14 MB OC=8 A/B Result

Same-build no-env control:

```text
server_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_v14mboc8_control_20260628_11.log
summary=.cache/mnn-pic-benchmark/decode_repair_v14mboc8_control_20260628_11/summary.csv
x=0 none                  TPOT=87.58471428571427 ms
x=1 lagged_attention_hkvd TPOT=97.97085714285714 ms
x=3 lagged_attention_hkvd TPOT=124.01161904761905 ms
x=5 lagged_attention_hkvd TPOT=125.25404761904763 ms
x=7 lagged_attention_hkvd TPOT=126.65790476190477 ms
```

OC=8 env:

```text
env=MNN_CUDA_PIC_INT4_V14_MB_OC=8
server_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_v14mboc8_env_20260628_11.log
summary=.cache/mnn-pic-benchmark/decode_repair_v14mboc8_env_20260628_11/summary.csv
x=0 none                  TPOT=86.92838095238095 ms
x=1 lagged_attention_hkvd TPOT=115.93909523809525 ms
x=3 lagged_attention_hkvd TPOT=123.83538095238096 ms
x=5 lagged_attention_hkvd TPOT=126.17538095238096 ms
x=7 lagged_attention_hkvd TPOT=127.32223809523809 ms
```

Delta env vs same-build control:

```text
x=0  -0.656 ms
x=1 +17.968 ms
x=3  -0.176 ms
x=5  +0.921 ms
x=7  +0.664 ms
```

Decision:

- Reject and remove `MNN_CUDA_PIC_INT4_V14_MB_OC`.
- The target batch=2 / x=1 case regresses badly, likely from register pressure/occupancy loss.
- The apparent x=3 movement is noise and not attributable to the batch=2-only env.
- Keep current V14_MB OC=4 for batch=2.

## Default Artifact Restored After Rejected OC=8 A/B

Cleanup:

```text
source check:
  rg -n "V14_MB_OC|V14_OC8|usePicV14Oc8|picV14Mb" source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
  no matches
  git diff -- source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
  empty
```

Restore build:

```text
result=success
outputs=libMNN.so, libMNN_Express.so, libMNN_Cuda_Main.so, libpic_llm.so, pic_server
artifact timestamps=2026-06-28 11:18 CST
file check=aarch64
```

Restore sync/server:

```text
rsync .cache/output/mnn/artifacts/jetson/ -> jetson_cross_cuda complete
server=default/no profile/no rejected env
remote_port=18131
local_tunnel=19131
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_default_after_v14mboc8_reject_20260628_11.log
/v1/models=HTTP 200
```

Conclusion:

- No default speedup landed in this segment.
- The strongest current data remains the same-build control:

```text
x=0  87.585 ms
x=1  97.971 ms
x=3 124.012 ms
x=5 125.254 ms
x=7 126.658 ms
```

- Further work should target a new small-M dense design or q8k16 internal work reduction, not cuBLAS rows2/3, q4 qtile variants, qtile-off, or V14_MB OC=8.

## V14 MB OC=2 A/B Scope

Temporary source change:

```text
file=source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
env=MNN_CUDA_PIC_INT4_V14_MB_OC=2
default behavior=no env returns OC_PER_BLK=4
active only when pic_decode_repair_sparse && batch<=2 && V14_MB tiny GEMV path
```

Rationale:

- OC=8 regressed x=1, likely from register pressure/occupancy loss.
- OC=2 is the opposite tradeoff: fewer accumulators per thread, more OC blocks and more weight reads.
- This should be a short x=0/1 A/B first. If x=1 is not better, reject immediately and restore default.

Build attempt:

```text
command=MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
result=not completed
```

Blocker:

```text
The build script removed stale jetson_cross and entered full configure/build.
CMake cutlass-populate removed 3rd_party/cutlass/v4.0.0 and attempted to clone https://github.com/NVIDIA/cutlass.git.
The clone hung; manual network check failed:
fatal: unable to access 'https://github.com/NVIDIA/cutlass.git/': gnutls_handshake() failed: The TLS connection was non-properly terminated.
```

Cleanup:

```text
terminated stuck build process group
source hook removed
rg -n "V14_MB_OC|V14_OC2|usePicV14Oc2|picV14Mb" source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
  no matches
git diff -- source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
  empty
build processes
  none
```

Current server state after cleanup:

```text
server=default/no profile/no rejected env
remote_pid=88858
remote_port=18131
local_tunnel=19131
/v1/models=HTTP 200
```

Conclusion:

- OC=2 A/B was not run; do not treat it as accepted or rejected on performance.
- Re-running it requires restoring/caching cutlass locally or waiting for GitHub TLS access to work.
- The working Jetson server remains default and healthy from the restored artifact before the failed build attempt.

## V14 MB OC=2 Build Resumed

GitHub access recovered:

```text
git ls-remote https://github.com/NVIDIA/cutlass.git HEAD
e8ecfad75b44d1ad56264f5001d877e9e47fe080 HEAD
```

The temporary OC=2 hook was re-applied and the Jetson cross-build was retried.

Build result:

```text
[100%] Built target pic_llm
[100%] Built target pic_server
[build_artifacts] Installed libMNN.so
[build_artifacts] Installed libMNN_Express.so
[build_artifacts] Installed libMNN_Cuda_Main.so
[build_artifacts] Installed libpic_llm.so
[build_artifacts] Installed pic_server
[build_artifacts] Done
```

Artifact check:

```text
.cache/output/mnn/artifacts/jetson/bin/pic_server:          ELF 64-bit LSB executable, ARM aarch64
.cache/output/mnn/artifacts/jetson/lib/libMNN_Cuda_Main.so: ELF 64-bit LSB shared object, ARM aarch64
.cache/output/mnn/artifacts/jetson/lib/libpic_llm.so:       ELF 64-bit LSB shared object, ARM aarch64
```

Sync:

```text
rsync .cache/output/mnn/artifacts/jetson/ -> jetson_cross_cuda complete
```

Same-build no-env control:

```text
server_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/decode_repair_v14mboc2_control_20260628_11.log
summary=.cache/mnn-pic-benchmark/decode_repair_v14mboc2_control_20260628_11/summary.csv
x=0 none                  TPOT=88.95157142857143 ms
x=1 lagged_attention_hkvd TPOT=98.80628571428572 ms
```

OC=2 env:

```text
env=MNN_CUDA_PIC_INT4_V14_MB_OC=2
server_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/decode_repair_v14mboc2_env_20260628_11.log
summary=.cache/mnn-pic-benchmark/decode_repair_v14mboc2_env_20260628_11/summary.csv
x=0 none                  TPOT=87.68671428571429 ms
x=1 lagged_attention_hkvd TPOT=104.73409523809524 ms
```

Delta env vs same-build control:

```text
x=0 -1.265 ms
x=1 +5.928 ms
```

Decision:

- Reject and remove `MNN_CUDA_PIC_INT4_V14_MB_OC`.
- OC=2 regresses the target batch=2 / x=1 path, likely because extra OC blocks and rereads dominate any register-pressure benefit.
- Together with rejected OC=8, this says simple `OC_PER_BLK` retuning inside V14_MB is not the next useful direction.
- Next dense work should profile/decompose V14_MB memory traffic or design a real batch=2 tensor-core/fused dense path, not continue OC-only sweeps.

## Default Artifact Restored After Rejected OC=2 A/B

Cleanup:

```text
source check:
  rg -n "V14_MB_OC|V14_OC2|usePicV14Oc2|picV14Mb" source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
  no matches
  git diff -- source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
  empty
```

Restore build:

```text
result=success
outputs=libMNN.so, libMNN_Express.so, libMNN_Cuda_Main.so, libpic_llm.so, pic_server
artifact timestamps=2026-06-28 11:42 CST
file check=aarch64
```

Restore sync/server:

```text
rsync .cache/output/mnn/artifacts/jetson/ -> jetson_cross_cuda complete
server=default/no profile/no rejected env
remote_pid=89724
remote_port=18131
local_tunnel=19131
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_default_after_v14mboc2_reject_20260628_11.log
/v1/models=HTTP 200
```

Conclusion:

- No default speedup landed from OC retuning.
- Keep current default V14_MB `OC_PER_BLK=4`.
- OC=8 and OC=2 both regressed the target x=1 case, so the next dense candidate should reduce actual V14_MB work/memory traffic or switch to a real batch=2 tensor-core/fused implementation.

## x=3 Kernel Profile Attribution

Profile server:

```text
env=MNN_PAGED_ATTENTION_PROFILE=1
env=MNN_PIC_DECODE_REPAIR_PROFILE=1
server_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_kernel_profile_x3_20260628_11.log
local_copy=.cache/mnn-pic-benchmark/decode_repair_kernel_profile_x3_20260628_11/server.log
```

Profile benchmark:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://127.0.0.1:19131 \
  --output-csv .cache/mnn-pic-benchmark/decode_repair_kernel_profile_x3_20260628_11/summary.csv \
  --output-dir .cache/mnn-pic-benchmark/decode_repair_kernel_profile_x3_20260628_11/client \
  --device jetson --device-display Jetson --backend cuda --frequency-profile max \
  --model llama-pic --model-name 'Llama3.2 3B' \
  --model-config .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json \
  --contexts 1024 --budgets 0.00 --decode-selectors lagged_attention_hkvd \
  --repair-tokens 3 --attention-layer-idx 1 --top-m 32 \
  --max-tokens 3 --repeats 1 --warm-repeats 0 --no-update-cache
```

Profile latency, for attribution only:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_kernel_profile_x3_20260628_11/summary.csv
x=3 lagged_attention_hkvd generated=3 TPOT=281.9995 ms
```

Decode-repair stage profile:

```text
step0 forward_raw=287.742 ms total=288.234 ms
step1 forward_raw=133.958 ms total=134.044 ms
step2 forward_raw=141.362 ms total=141.445 ms
```

Regex aggregate across the three decode steps:

```text
sparse_flash_qtile_attention:
  count=82
  sum=71.204 ms
  avg=868.3 us
  variant=hd128_q8k16

decode_attention_rank:
  count=3
  sum=0.883 ms
  avg=294.3 us
  parts: prep=0.133 ms topk_init=0.053 ms topk=0.502 ms d2h=0.143 ms

rows4 CUTLASS dense:
  3072->3072 count=163 sum=44.560 ms avg=273.4 us

rows4 cuBLAS dense:
  total count=409 sum=252.825 ms
  3072->1024 count=164 sum=81.423 ms avg=496.5 us
  3072->8192 count=163 sum=94.674 ms avg=580.8 us
  8192->3072 count=82  sum=76.728 ms avg=935.7 us
```

Notes:

- The profile has synchronized kernel timing, so the TPOT is not formal latency.
- Server logging can interleave lines, but the aggregate is stable enough for attribution.
- Host-side decode repair scheduling remains near microseconds; `forward_raw` dominates.
- x=3 is now primarily rows4 dense, especially rows45 cuBLAS MLP-like projections.
- qtile attention remains material at about `24 ms/step`, but dense is larger at about `99 ms/step` in the synchronized profile.
- Existing rows45 envs were already explored earlier: `rows45=down` regressed x=3/5/7, `compute=16f/fast16` and `algo=default` did not produce a stable default win. Do not keep repeating those as the main path.

Current next direction:

- x=1: V14_MB needs a real batch=2 dense redesign; OC-only sweeps are rejected.
- x=3/5/7: rows4-8 MLP dense needs fewer launches or fused work, e.g. gate/up fusion or a dedicated batch4-8 tensor-core/fused dense path. Attention-rank capture is not worth optimizing further.

After this profile the server was restored to default/no profile/no rejected env:

```text
remote_pid=90021
remote_port=18131
local_tunnel=19131
remote_log=/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs/pic_server_decode_repair_default_after_x3_profile_20260628_11.log
/v1/models=HTTP 200
```

## Rows45 cuBLAS Skip Set-Math A/B Scope

Temporary source change:

```text
file=source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
env=MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_SKIP_SET_MATH=1
default behavior=no env preserves the existing cublasSetMathMode calls
active scope=runRows45CublasFp16 helper, used by rows4-8 cuBLAS dense path
```

Rationale:

- x=3 profile shows rows4 cuBLAS dense is the largest synchronized kernel aggregate:

```text
rows4 cuBLAS dense total count=409 sum=252.825 ms
```

- Prior rows45 algorithm and compute-mode env sweeps did not produce a stable default win, and OC-only V14_MB retuning was rejected for x=1.
- This A/B tests a smaller non-compute hypothesis: avoid repeated `cublasSetMathMode` calls inside the hot helper and rely on the handle's already configured mode for the server process.

Required verification:

1. Cross-build `pic_server` artifact for Jetson with the env-gated hook.
2. Sync to Jetson.
3. Start same-build no-env server and run x=0/1/3/5/7 as control.
4. Start `MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_SKIP_SET_MATH=1` server and run the same x=0/1/3/5/7 matrix.
5. Reject unless x=3/5/7 improve versus same-build no-env control without harming x=0/x=1.

Build command:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```
