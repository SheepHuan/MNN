# Jetson PIC Decode Repair Optimize 2026-06-28 11

## 11:02 x=1 Graph Profile Attribution

Profile-only run:

```text
env=MNN_PIC_GRAPH_PROFILE=1,MNN_PIC_DECODE_REPAIR_PROFILE=1
summary=.cache/mnn-pic-benchmark/decode_repair_graph_profile_x1_20260628_10/summary.csv
server_log=.cache/mnn-pic-benchmark/decode_repair_graph_profile_x1_20260628_10/server.log
x=1 profile TPOT=222.210 ms
```

Steady decode-repair profile:

```text
step1 forward_raw=123.902 ms total=123.973 ms
step2 forward_raw=123.916 ms total=124.005 ms
```

Graph type totals:

```text
Convolution=210.781 ms
PicSparseAttention=117.170 ms
Raster=49.873 ms
BinaryOp=22.187 ms
While=16.970 ms
```

Tiny GEMV aggregate:

```text
3072->8192 total=62.078 ms calls=164 avg=378.5 us
8192->3072 total=29.802 ms calls=82  avg=363.4 us
3072->3072 total=29.758 ms calls=165 avg=180.4 us
3072->1024 total=16.216 ms calls=165 avg=98.3 us
```

Conclusion: remaining x=1 overhead is compute-bound inside compact batch=2 forward: small-M INT4 dense and `PicSparseAttention`. Host ranking/attention-rank capture remains negligible, and qtile variant retuning did not help. Default server was restored after the profile and `/v1/models` returned HTTP 200.

## 11:08 V14 MB OC=8 A/B Started

Temporary source A/B:

```text
env=MNN_CUDA_PIC_INT4_V14_MB_OC=8
scope=PIC decode repair sparse, batch<=2, INT4 V14_MB tiny GEMV only
default=no env remains OC_PER_BLK=4
```

Hypothesis: x=1 is dominated by batch=2 tiny GEMV; processing 8 output channels per block may reduce blocks and weight rereads enough to beat the current OC=4 path. This is env-gated and must be rejected if x=1 regresses or if x=0 changes beyond noise.

Build and sync:

```text
build=success
artifact=.cache/output/mnn/artifacts/jetson
outputs=pic_server, libMNN.so, libMNN_Express.so, libMNN_Cuda_Main.so, libpic_llm.so
file=aarch64
rsync=jetson_cross_cuda complete
```

## 11:17 V14 MB OC=8 A/B Rejected

Same-build no-env control:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_v14mboc8_control_20260628_11/summary.csv
x=0  87.585 ms
x=1  97.971 ms
x=3 124.012 ms
x=5 125.254 ms
x=7 126.658 ms
```

OC=8 env:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_v14mboc8_env_20260628_11/summary.csv
x=0  86.928 ms
x=1 115.939 ms
x=3 123.835 ms
x=5 126.175 ms
x=7 127.322 ms
```

Decision: reject and remove the temporary hook. The target x=1 regresses by `+17.968 ms`; x=3 movement is noise and not attributable to the batch=2-only env.

## 11:20 Default Restored

Cleanup:

```text
source check: no V14_MB_OC / V14_OC8 / usePicV14Oc8 / picV14Mb matches
build=success
artifact=aarch64
rsync=jetson_cross_cuda complete
server=default/no profile/no rejected env
/v1/models=HTTP 200
remote_log=.cache/logs/pic_server_decode_repair_default_after_v14mboc8_reject_20260628_11.log
```

No default speedup landed in this segment.

## 11:22 V14 MB OC=2 A/B Started

Temporary source A/B:

```text
env=MNN_CUDA_PIC_INT4_V14_MB_OC=2
scope=PIC decode repair sparse, batch<=2, INT4 V14_MB tiny GEMV only
default=no env remains OC_PER_BLK=4
```

This tests the opposite of rejected OC=8: reduce per-thread accumulators/register pressure at the cost of more OC blocks. First run will be x=0/1 only; full x=0/1/3/5/7 only if x=1 improves.

## 11:29 V14 MB OC=2 A/B Not Run

The temporary OC=2 hook was added but the build could not complete because the full rebuild re-entered `cutlass-populate` and GitHub clone failed:

```text
git ls-remote https://github.com/NVIDIA/cutlass.git HEAD
fatal: unable to access 'https://github.com/NVIDIA/cutlass.git/': gnutls_handshake() failed
```

Cleanup:

```text
source check: no V14_MB_OC / V14_OC2 / usePicV14Oc2 / picV14Mb matches
git diff -- ConvFpAIntBExecution.cu: empty
build processes: none
server=default/no profile/no rejected env
/v1/models=HTTP 200
```

No OC=2 performance conclusion was produced.

## 11:31 V14 MB OC=2 Build Resumed

GitHub access recovered:

```text
git ls-remote https://github.com/NVIDIA/cutlass.git HEAD
e8ecfad75b44d1ad56264f5001d877e9e47fe080 HEAD
```

OC=2 hook was re-applied and the build was retried.

## 11:37 V14 MB OC=2 A/B Rejected

Build and sync completed:

```text
build=success
artifact=aarch64
rsync=jetson_cross_cuda complete
```

Same-build no-env control:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_v14mboc2_control_20260628_11/summary.csv
x=0 88.952 ms
x=1 98.806 ms
```

OC=2 env:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_v14mboc2_env_20260628_11/summary.csv
x=0  87.687 ms
x=1 104.734 ms
```

Decision: reject. The target x=1 case regressed by `+5.928 ms`; x=0 movement is normal noise. The temporary hook is being removed and the default artifact will be rebuilt/restored.

## 11:45 Default Restored After OC=2 Reject

Cleanup:

```text
source check: no V14_MB_OC / V14_OC2 / usePicV14Oc2 / picV14Mb matches
git diff -- ConvFpAIntBExecution.cu: empty
build=success
artifact=aarch64
rsync=jetson_cross_cuda complete
server=default/no profile/no rejected env
/v1/models=HTTP 200
remote_log=.cache/logs/pic_server_decode_repair_default_after_v14mboc2_reject_20260628_11.log
```

No default speedup landed from OC retuning. Both OC=8 and OC=2 regressed x=1, so the next useful dense direction is not `OC_PER_BLK` sweep; it needs either V14_MB internal memory-traffic reduction or a real batch=2 tensor-core/fused path.

## 11:55 x=3 Kernel Profile Attribution

Profile-only run:

```text
env=MNN_PAGED_ATTENTION_PROFILE=1,MNN_PIC_DECODE_REPAIR_PROFILE=1
summary=.cache/mnn-pic-benchmark/decode_repair_kernel_profile_x3_20260628_11/summary.csv
server_log=.cache/mnn-pic-benchmark/decode_repair_kernel_profile_x3_20260628_11/server.log
x=3 profile TPOT=281.999 ms
```

Steady decode-repair profile:

```text
step1 forward_raw=133.958 ms total=134.044 ms
step2 forward_raw=141.362 ms total=141.445 ms
```

Kernel aggregate across three decode steps:

```text
sparse_flash_qtile_attention query=4 count=82 sum=71.204 ms avg=868 us
decode_attention_rank count=3 sum=0.883 ms avg=294 us
rows4 cuBLAS dense count=409 sum=252.825 ms
rows4 CUTLASS 3072->3072 count=163 sum=44.560 ms
```

Conclusion: x=3 is dominated by rows4 dense, especially rows45 cuBLAS MLP-like projections. Attention rank capture is still negligible; qtile attention is material but secondary. Since prior `rows45=down`, `compute=16f/fast16`, and `algo=default` A/B did not land, next useful work is a new rows4 dense design, not another existing cuBLAS env sweep.

## 11:58 Rows45 cuBLAS Skip Set-Math A/B Started

Temporary source A/B:

```text
env=MNN_CUDA_PIC_INT4_ROWS45_CUBLAS_SKIP_SET_MATH=1
scope=rows4-8 cuBLAS dense helper only
default=no env still calls cublasSetMathMode according to existing policy
```

Hypothesis: x=3/5/7 synchronized profile shows hundreds of rows4 cuBLAS calls per request. If `cublasSetMathMode` is contributing measurable per-call host overhead, skipping it after handle setup could reduce non-compute overhead without changing kernels. This is a narrow A/B hook; reject it if same-build env does not improve x=3/5/7 or if x=0/x=1 regresses beyond noise.
