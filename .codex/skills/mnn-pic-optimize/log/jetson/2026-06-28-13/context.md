# Jetson PIC Decode Repair Optimize Context 2026-06-28 13

## Starting Point

Carry-over from `jetson/2026-06-28-12`:

```text
objective=optimize Jetson CUDA PIC decode repair TPOT for lagged_attention_hkvd x=1/3/5/7
accepted range:
  x=0 none                  TPOT=86-89 ms
  x=1 lagged_attention_hkvd TPOT=97-99 ms
  x=3 lagged_attention_hkvd TPOT=123-125 ms
  x=5 lagged_attention_hkvd TPOT=125-126 ms
  x=7 lagged_attention_hkvd TPOT=126-128 ms
```

Do not change decode selector semantics:

```text
selector=lagged_attention_hkvd
attention rank capture remains compact and CPU-visible only as rank indices
no qtile shortcuts
no PagedCache bypass
x=0 none must be included in every end-to-end validation
```

Known rejected rows45/cuBLAS policy directions before this hour:

```text
rows45 cublas=down     rejected in 2026-06-28-09, x=3/5/7 +5.7..+6.6 ms
rows45 minDim=2048     rejected in 2026-06-28-09, no win over accepted baseline
rows45 algo default    env looked weakly good, source default verification rejected
rows45 cublas=0        rejected in 2026-06-28-12, x=3/5/7 +4..+8 ms
skip cublasSetMathMode rejected in 2026-06-28-12, sub-ms mixed noise
```

## Build And Sync For Direct-Op

The previous cross build had been interrupted and was terminated before this continuation:

```text
stale process group 491274 killed with TERM
stale wait helper 503283 killed
```

Rebuilt test artifact:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda_test" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda_test" \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
JOBS=16 CMAKE_ARGS="-DMNN_BUILD_TEST=ON" \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Verification:

```text
run_test.out: ELF 64-bit ARM aarch64
libMNN_Cuda_Main.so: ELF 64-bit ARM aarch64
timestamp=2026-06-28 13:40
```

Synced to Jetson:

```bash
rsync -a --delete .cache/output/mnn/artifacts/jetson_cross_cuda_test/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_test/
```

## Packed Gate/Up Measurement

Source hook:

```text
file=test/bench_ops/cuda/CudaWeightOnlyConvPerf.cpp
change=add PicDecodeMlpPacked measurement
path=hidden -> 2*inter concat Conv -> PicPackedSiluMul -> down
scope=direct-op benchmark only
```

Remote command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  mkdir -p .cache/bench_ops/decode_repair_dense && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_test && \
  CUDA_LIB=/usr/local/cuda-12.2/targets/aarch64-linux/lib && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:$CUDA_LIB:${LD_LIBRARY_PATH:-}" && \
  env MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 MNN_BENCH_MLP_HIDDEN=3072 MNN_BENCH_MLP_INTER=8192 \
      MNN_BENCH_MLP_WARMUP=20 MNN_BENCH_MLP_REPEAT=80 \
      "$ART/bin/run_test.out" bench_ops/cuda/perf/PicDecodeMlp 2 2 1 x 2 \
      2>&1 | tee .cache/bench_ops/decode_repair_dense/pic_decode_mlp_packed_repeat_20260628_13.log'
```

Longer-repeat result:

```text
rows=4 split chain=2.6012 ms packed chain=2.5848 ms delta=-0.0164 ms
rows=6 split chain=1.5541 ms packed chain=1.4878 ms delta=-0.0663 ms
rows=8 split chain=1.5025 ms packed chain=1.5155 ms delta=+0.0130 ms
```

Earlier shorter-repeat result was noisier:

```text
rows=4 delta=-2.8106 ms
rows=6 delta=-0.0208 ms
rows=8 delta=-0.1986 ms
```

Decision:

- Keep the direct-op measurement hook for diagnostics.
- Do not change production graph/export defaults.
- Do not revive `PicGateUpWeightOnly` or packed gate/up as a mainline based on this weak signal.

## Projection-Family Diagnostics

Ran `WeightOnlyConv` by projection family in separate processes to avoid a single process creating many large static-dequant resources:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  mkdir -p .cache/bench_ops/decode_repair_dense && \
  ART=.cache/output/mnn/artifacts/jetson_cross_cuda_test && \
  CUDA_LIB=/usr/local/cuda-12.2/targets/aarch64-linux/lib && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:$CUDA_LIB:${LD_LIBRARY_PATH:-}" && \
  for case in hidden_to_inter inter_to_hidden hidden_to_kv hidden_to_qkv_concat hidden_to_gateup_concat hidden_to_hidden; do \
    env MNN_BENCH_WEIGHT_ONLY_CASE=$case MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
        MNN_BENCH_WEIGHT_ONLY_HIDDEN=3072 MNN_BENCH_WEIGHT_ONLY_INTER=8192 \
        MNN_BENCH_WEIGHT_ONLY_WARMUP=20 MNN_BENCH_WEIGHT_ONLY_REPEAT=80 \
        "$ART/bin/run_test.out" bench_ops/cuda/perf/WeightOnlyConv 2 2 1 x 2; \
    env MNN_CUDA_PIC_INT4_ROWS45_CUBLAS=0 MNN_BENCH_WEIGHT_ONLY_CASE=$case \
        MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 MNN_BENCH_WEIGHT_ONLY_HIDDEN=3072 \
        MNN_BENCH_WEIGHT_ONLY_INTER=8192 MNN_BENCH_WEIGHT_ONLY_WARMUP=20 \
        MNN_BENCH_WEIGHT_ONLY_REPEAT=80 "$ART/bin/run_test.out" \
        bench_ops/cuda/perf/WeightOnlyConv 2 2 1 x 2; \
  done 2>&1 | tee .cache/bench_ops/decode_repair_dense/weight_only_shapes_by_case_default_vs_cublas0_20260628_13.log'
```

Selected rows4-8 results:

```text
hidden_to_inter default: rows4 1.7586, rows6 1.7864, rows8 1.3768 ms
hidden_to_inter cublas0: rows4 3.2408, rows6 1.1860, rows8 0.8106 ms

inter_to_hidden default: rows4 1.1130, rows6 1.1335, rows8 1.1543 ms
inter_to_hidden cublas0: rows4 2.1052, rows6 1.5863, rows8 1.2946 ms

hidden_to_kv default: rows4 0.0675, rows6 0.0674, rows8 0.0657 ms
hidden_to_kv cublas0: rows4 0.0739, rows6 0.0739, rows8 0.0753 ms

hidden_to_gateup_concat default: rows4 0.9419, rows6 0.9512, rows8 0.9623 ms
hidden_to_gateup_concat cublas0: rows4 1.0609, rows6 1.0585, rows8 1.0724 ms
```

Interpretation:

- The by-family direct-op result is useful for diagnosing shapes, but it still does not supersede end-to-end results.
- `rows45=down` was already rejected in the real server path; it regressed x=3/5/7 by about 5.7-6.6 ms.
- `rows45=0` was already rejected in the real server path; it regressed x=3/5/7 by about 4-8 ms.
- The current server x=3 profile shows rows4 rows45 cuBLAS MLP shapes around `0.56..0.73 ms`, while direct-op profile in the test process can report multi-ms cuBLAS calls. That mismatch is why no default policy change is made from these direct-op numbers.

## Policy Repeat Diagnostic

Ran `PicDecodeMlp` with default, `down`, and `0` policies:

```text
log=.cache/bench_ops/decode_repair_dense/pic_decode_mlp_policy_repeat_20260628_13.log
rows=4,6,8
hidden=3072
inter=8192
warmup=30
repeat=120
```

Unprofiled policy repeat:

```text
default:
  rows4 split=7.9620 packed=7.9172 delta=-0.0449
  rows6 split=5.5270 packed=5.2635 delta=-0.2636
  rows8 split=2.8947 packed=1.5612 delta=-1.3335

down:
  rows4 split=1.8164 packed=1.7023 delta=-0.1141
  rows6 split=1.7174 packed=1.6851 delta=-0.0322
  rows8 split=1.7326 packed=1.6361 delta=-0.0965

cublas0:
  rows4 split=1.6900 packed=1.6314 delta=-0.0585
  rows6 split=1.7376 packed=1.6641 delta=-0.0736
  rows8 split=1.7503 packed=1.6690 delta=-0.0813
```

Decision:

- Treat this as diagnostic only.
- The direct-op default path is not representative of the production server profile, likely because the direct process and full model server differ in cuBLAS warm state, full-model static-dequant residency, and launch/profile conditions.
- No end-to-end A/B is justified from this alone because the relevant policy variants have already been rejected in server TPOT runs.

## Current Next Direction

Stop retuning rows45 env policy knobs and stop trying packed gate/up as a default. The next credible production optimization should come from one of:

```text
1. compact dense implementation work that beats current server-profile rows4-8 cuBLAS in the full model path
2. graph/elementwise/layout overhead reduction visible in MNN_PIC_GRAPH_PROFILE x=7
3. batch=2 dense redesign for x=1, where V14_MB and PicSparseAttention dominate
```

Any candidate must be validated by:

```text
direct-op accuracy/perf if it touches a kernel
server x=0/1/3/5/7 repeat3 TPOT
x=0 no regression
lagged_attention_hkvd runtime remains mnn_token_id_sparse_decode
no selector or PagedCache semantic changes
```
