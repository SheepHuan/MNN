# Jetson CUDA qtile variant sweep context

## Code shape

Changed `source/backend/cuda/execution/PagedAttentionExecution.cu` inside the existing `pagedSparseFlashMQTileKernel` family:

- Restored code-internal `hd128_q4k8`.
- Added code-internal `hd128_q8k8`.
- Kept `hd128_q4k16` and `hd128_q8k16`.
- Removed/kept absent CUDA qtile env gates and tune paths; scan found no `MNN_CUDA_PAGED_ATTENTION_QTILE_*`, qtile tune, or decode-repair qtile experiment symbols.

Fixed selection:

```text
headDim != 128: no HD128 MQTile variant
headDim=128, attnLen < 64: non-qtile sparse path
cacheblend sparse rows: hd128_q4k16
fixed-plan sparse rows, 64 <= attnLen < 128: hd128_q4k8
fixed-plan sparse rows, 128 <= attnLen < 256: hd128_q8k8
fixed-plan sparse rows, attnLen >= 256: hd128_q8k16
decode repair: unchanged hd128_q8k16 for headDim=128 and attnLen>=2
```

Also fixed `.codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py` Jetson frequency verification:

- The remote Python snippet now imports `re`, `shutil`, and `subprocess`.
- Jetson frequency query now uses `sudo -n jetson_clocks --show` / `sudo -n nvpmodel -q` when normal user access is insufficient, so GPU/EMC lock verification is not falsely marked failed.

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

Build details:

```text
HEAD=ca9fd3e5 plus dirty worktree
CUDA support: ON
CUDA architectures: sm_72
KleidiAI: OFF
local artifact: .cache/output/mnn/artifacts/jetson
remote artifact: jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda
```

Sync command:

```bash
rsync -a --delete .cache/output/mnn/artifacts/jetson/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

Artifact check:

```text
pic_server: ARM aarch64 ELF
libMNN.so: ARM aarch64 ELF
libMNN_Cuda_Main.so: ARM aarch64 ELF
libpic_llm.so: ARM aarch64 ELF
```

## Test matrix

Excluded by design:

- Llama3.2 1B.
- Contexts `2048` and `2560`.
- Context `3072`.

Ran only benchmark-existing PIC rows:

```text
device: jetson
frequency: max
contexts: 512,1024,1536
modes: full-reuse, cacheblend, epic
ratios: 0.05,0.10,0.20,0.30,0.40,0.50
models: MiniCPM5-1B, Llama3.2 3B, Qwen3-4B
```

Commands:

```bash
MNN_PIC_CONTEXTS=512,1024,1536 \
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_jetson.sh \
  --model-key minicpm5-1b \
  --benchmark-csv benchmark.csv \
  --only-benchmark-csv-rows \
  --modes full-reuse,cacheblend,epic \
  --ratios 0.05,0.10,0.20,0.30,0.40,0.50 \
  --run-id jetson_qtile_variants_minicpm5_20260704_0142

MNN_PIC_CONTEXTS=512,1024,1536 \
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_jetson.sh \
  --model-key llama3.2-3b \
  --benchmark-csv benchmark.csv \
  --only-benchmark-csv-rows \
  --modes full-reuse,cacheblend,epic \
  --ratios 0.05,0.10,0.20,0.30,0.40,0.50 \
  --run-id jetson_qtile_variants_llama32_3b_20260704_0142

MNN_PIC_CONTEXTS=512,1024,1536 \
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_jetson.sh \
  --model-key qwen3-4b \
  --benchmark-csv benchmark.csv \
  --only-benchmark-csv-rows \
  --modes full-reuse,cacheblend,epic \
  --ratios 0.05,0.10,0.20,0.30,0.40,0.50 \
  --run-id jetson_qtile_variants_qwen3_4b_20260704_0142
```

Qwen3-4B had no `full-reuse` rows in `benchmark.csv`, so `--only-benchmark-csv-rows` ran only `cacheblend` and `epic` for that model.

Frequency records in all three runs:

```text
CPU: 2265600 min/max/current
GPU: 1377000000 min/max/current
EMC: 2133000000 current, FreqOverride=1
```

Log scan:

```text
No ERROR / target unavailable / async persistent PIC cache read failed / Cache invalid / HTTP 500 / Traceback matches in local run logs.
```

## Regression summary vs benchmark.csv

Comparison output:

```text
.cache/latency_budget_20260625/jetson_qtile_variants_20260704_0142_regression.csv
matched_rows=114
regress_gt5=0
status_counts: improve_gt5=83, flat=31
```

Per model/mode:

```text
Llama3.2 3B  cacheblend n=18 avg=-19.43% worst= +1.22% best=-77.71%
Llama3.2 3B  epic       n=18 avg=-34.95% worst=-21.51% best=-48.29%
Llama3.2 3B  full-reuse n= 3 avg=-88.79% worst=-87.11% best=-91.14%
MiniCPM5-1B  cacheblend n=18 avg=-10.34% worst= -1.72% best=-27.45%
MiniCPM5-1B  epic       n=18 avg=-23.30% worst=-12.88% best=-33.09%
MiniCPM5-1B  full-reuse n= 3 avg=-79.60% worst=-78.22% best=-81.31%
Qwen3-4B     cacheblend n=18 avg= -1.52% worst= +0.04% best= -3.94%
Qwen3-4B     epic       n=18 avg=-15.74% worst= -3.83% best=-27.38%
```

Top slowest deltas:

```text
Llama3.2 3B ctx=1536 cacheblend 0.50 base=3.715375s new=3.760861s delta=+1.22%
Qwen3-4B    ctx=1024 cacheblend 0.40 base=3.135856s new=3.137125s delta=+0.04%
Qwen3-4B    ctx=1024 cacheblend 0.50 base=3.715716s new=3.702658s delta=-0.35%
Qwen3-4B    ctx=1536 cacheblend 0.40 base=6.420928s new=6.377461s delta=-0.68%
Qwen3-4B    ctx=1536 cacheblend 0.30 base=5.166968s new=5.130748s delta=-0.70%
```

## Conclusion

The fixed-plan HD128 qtile variants do not introduce a Jetson PIC prefill regression in the covered benchmark matrix. Epic benefits most from the fixed-plan q8 variants. Cacheblend remains effectively unchanged or faster because it stays on the accepted `hd128_q4k16` default.
