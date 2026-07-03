# Jetson CUDA HD128 qtile expansion context

## Code shape

Changed `source/backend/cuda/execution/PagedAttentionExecution.cu` inside the existing
`pagedSparseFlashMQTileKernel<HEAD_DIM, Q_TILE, K_TILE>` family only.

Added enum/name/launch support:

```text
hd128_q2k16
hd128_q2k8
hd128_q16k16
hd128_q16k8
```

Final production selector:

```text
headDim != 128: no HD128 MQTile variant
headDim=128, attnLen < 64: non-qtile sparse path
cacheblend sparse rows: hd128_q4k16
fixed-plan sparse rows, 64 <= attnLen < 128: hd128_q4k8
fixed-plan sparse rows, 128 <= attnLen < 256: hd128_q8k8
fixed-plan sparse rows, 256 <= attnLen < 384: hd128_q8k16
fixed-plan sparse rows, attnLen >= 384: hd128_q16k16
decode repair: unchanged hd128_q8k16 for headDim=128 and attnLen>=2
```

No CUDA qtile env gate or tune path was added. The implementation remains a fixed
Jetson selector with a small set of parameter variants.

## Rejected variants and selector trials

K32 was not added. The current MQTile kernel uses `DLANES=16` and computes beta scores
with `lane < K_TILE`; with `K_TILE > 16`, the upper K rows would not have a lane to
compute their score. K32 therefore requires a kernel mapping change, not just a new
template instantiation.

Initial trial selected `hd128_q2k16` for `64 <= attnLen < 96`. MiniCPM5-1B showed
low-ratio Epic regressions:

```text
MiniCPM5-1B ctx=1536 epic 0.05: +20.723%
MiniCPM5-1B ctx=1024 epic 0.05: +17.135%
MiniCPM5-1B ctx=512  epic 0.10: +11.732%
```

Decision: keep q2 variants compiled as candidates, but do not select them by default.
Small fixed-plan rows stay on the previous `hd128_q4k8` path.

## Build and sync

Build command:

```bash
MNN_TARGET_DEVICE=jetson \
CROSS_COMPILE=ON \
ENABLE_CROSS_CUDA=ON \
CUDA_ARCHS=72 \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Artifact check:

```text
.cache/output/mnn/artifacts/jetson/bin/pic_server: ARM aarch64 ELF
.cache/output/mnn/artifacts/jetson/lib/libMNN.so: ARM aarch64 ELF
.cache/output/mnn/artifacts/jetson/lib/libMNN_Cuda_Main.so: ARM aarch64 ELF
.cache/output/mnn/artifacts/jetson/lib/libpic_llm.so: ARM aarch64 ELF
```

Sync command:

```bash
rsync -a --delete .cache/output/mnn/artifacts/jetson/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

## Test matrix

Excluded by design:

```text
Llama3.2 1B
contexts 2048, 2560, 3072
```

Final runs:

```bash
MNN_PIC_CONTEXTS=512,1024,1536 \
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_jetson.sh \
  --model-key minicpm5-1b \
  --benchmark-csv benchmark.csv \
  --only-benchmark-csv-rows \
  --modes full-reuse,cacheblend,epic \
  --ratios 0.05,0.10,0.20,0.30,0.40,0.50 \
  --run-id jetson_qtile_more_variants_20260704_0215_minicpm

MNN_PIC_CONTEXTS=512,1024,1536 \
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_jetson.sh \
  --model-key llama3.2-3b \
  --benchmark-csv benchmark.csv \
  --only-benchmark-csv-rows \
  --modes full-reuse,cacheblend,epic \
  --ratios 0.05,0.10,0.20,0.30,0.40,0.50 \
  --run-id jetson_qtile_more_variants_20260704_0215_llama3b

MNN_PIC_CONTEXTS=512,1024,1536 \
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_jetson.sh \
  --model-key qwen3-4b \
  --benchmark-csv benchmark.csv \
  --only-benchmark-csv-rows \
  --modes full-reuse,cacheblend,epic \
  --ratios 0.05,0.10,0.20,0.30,0.40,0.50 \
  --run-id jetson_qtile_more_variants_20260704_0215_qwen4b
```

The wrapper reports `frequency_note=cpu=max,gpu=max,ddr=max` for the final rows.

## Regression summary vs benchmark.csv

Combined output:

```text
.cache/latency_budget_20260625/jetson_qtile_more_variants_20260704_0215_regression.csv
matched_rows=114
regress_gt5=0
```

Per run:

```text
MiniCPM5-1B matched=39 regress_gt5=0 worst=-1.365%
Llama3.2 3B matched=39 regress_gt5=0 worst=+0.610%
Qwen3-4B    matched=36 regress_gt5=0 worst=+0.145%
```

Combined slowest deltas:

```text
Llama3.2 3B ctx=1536 cacheblend 0.50 base=3.715375s new=3.738033s delta=+0.610%
Qwen3-4B    ctx=1536 cacheblend 0.40 base=6.420928s new=6.430236s delta=+0.145%
Qwen3-4B    ctx=512  cacheblend 0.50 base=1.301597s new=1.301912s delta=+0.024%
Qwen3-4B    ctx=1536 cacheblend 0.05 base=1.488172s new=1.483283s delta=-0.329%
Qwen3-4B    ctx=512  cacheblend 0.40 base=1.076873s new=1.072805s delta=-0.378%
```

Best deltas:

```text
Llama3.2 3B ctx=1536 full-reuse base=1.831451s new=0.163470s delta=-91.074%
Llama3.2 3B ctx=512  full-reuse base=1.046799s new=0.125536s delta=-88.008%
Llama3.2 3B ctx=1024 full-reuse base=1.119425s new=0.144377s delta=-87.103%
MiniCPM5-1B ctx=512  full-reuse base=0.313211s new=0.058614s delta=-81.286%
MiniCPM5-1B ctx=1024 full-reuse base=0.330579s new=0.067224s delta=-79.665%
```

## Conclusion

The q16 large fixed-plan variant is safe in this covered Jetson matrix and improves
high-ratio Epic substantially. q2 is not a default candidate after the low-ratio
Epic regression trial. Cacheblend is intentionally unchanged because selected rows
are scattered and the accepted `hd128_q4k16` path remains the stable default.
