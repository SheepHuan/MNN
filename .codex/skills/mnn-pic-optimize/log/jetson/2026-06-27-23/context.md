# Jetson CUDA MiniCPM5-1B ctx512 sparse prefill optimization

## Problem

The rebuilt Jetson CUDA artifact initially failed MiniCPM5-1B ctx512 sparse warmup:

```text
CUDAPagedAttention async external PIC KV read failed at layer 1: CUDA external PIC KV requires direct mapped PagedCache at layer 1
Native graph-level cacheblend score/top-k failed on backend mnn_cuda
Native graph-level epic prefill failed on backend mnn_cuda
```

The failing run was:

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_jetson.sh \
  --model-key minicpm5-1b \
  --contexts 512 \
  --modes normal-full-recompute,cacheblend,epic \
  --ratios 0.50 \
  --run-id jetson_minicpm512_default_ab_20260627_3 \
  --output-dir .cache/jetson_qtile_ab
```

## Root Cause

CUDA `scheduleExternalLayerReadsFrom` created async read tasks for future layers even when those layers had not registered direct mapped PagedCache targets yet. The async task then stored a hard failure, and the later layer consumed that stale failed future instead of doing a synchronous direct read after its own target registration.

This differs from the OpenCL direct-target rule, where prefetch stops at the first missing target and only schedules layers with ready mapped targets captured by strong references.

A second correctness issue was that `ensureCache` could reuse a same-shape non-mapped cache for a later non-prefix request. Non-prefix CUDA requests on Xavier need mapped PagedCache whenever direct persistent PIC source hydrate/scoring is enabled.

## Code Changes

- `ensureCache` now refuses to reuse a non-mapped cache when `shouldUseMappedPagedCache()` is true for a non-prefix request.
- CUDA async external reads now:
  - compute required slots including reserved PIC source slots,
  - check target shape/capacity/readiness before scheduling,
  - stop scheduling at the first unready future layer,
  - capture `ExternalLayerMappedTarget` strong refs into the async task.
- CUDA qtile selection now defaults head_dim=128 sparse rows to `hd128_q4k16` with no env var. Explicit `off` / `none` still disables qtile, and explicit variants remain available.

## Build And Sync

Used existing compatible Jetson cross build directory:

```bash
cmake --build .cache/build/mnn/jetson_cross_cuda_proto --target MNN_Cuda_Main/fast --parallel 96
cp -a .cache/build/mnn/jetson_cross_cuda_proto/source/backend/cuda/libMNN_Cuda_Main.so \
  .cache/output/mnn/artifacts/jetson_cross_cuda_gcc9/lib/
rsync -a --delete .cache/output/mnn/artifacts/jetson_cross_cuda_gcc9/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

Remote ABI smoke remained compatible:

```text
libMNN.so GLIBC_2.29
libMNN_Cuda_Main.so GLIBC_2.29
pic_server GLIBC_2.17
llm_bench GLIBC_2.17
```

## Experiments

After the async scheduling fix but before default qtile:

```text
run_id=jetson_minicpm512_default_schedfix_20260627_1
normal=0.964411533s
cacheblend50=0.329401s
epic50=0.328695s
failures=none
```

The normal row in this run was an outlier versus the stable historical/current normal near 0.452s, so sparse acceptance was judged against the later no-env default sweep.

Explicit qtile variants at ctx512 ratio 50:

```text
run_id=jetson_minicpm512_qtile_auto_20260627_1
cacheblend50=1.373783s
epic50=1.364781s

run_id=jetson_minicpm512_qtile_hd128_q4k8_20260627_1
cacheblend50=0.445927s
epic50=0.444315s

run_id=jetson_minicpm512_qtile_hd128_q4k16_20260627_1
cacheblend50=0.292236s
epic50=0.291614s

run_id=jetson_minicpm512_qtile_hd128_q8k16_20260627_1
cacheblend50=0.308855s
epic50=0.237949s
```

`hd128_q4k16` was selected as default because it improves cacheblend and epic together. `hd128_q8k16` is faster for epic50 but worse for cacheblend50.

Explicit q4k16 ratio sweep:

```text
run_id=jetson_minicpm512_qtile_hd128_q4k16_sweep_20260627_1
cacheblend: 0.10=0.121289 0.20=0.159729 0.30=0.192236 0.40=0.242216 0.50=0.284649
epic:       0.10=0.112699 0.20=0.148891 0.30=0.183898 0.40=0.233901 0.50=0.280411
monotonic=true
failures=none
```

Accepted no-env default sweep after code change:

```text
run_id=jetson_minicpm512_default_q4k16_sweep_20260627_1
normal=0.452172613s
cacheblend: 0.10=0.119594 0.20=0.159081 0.30=0.191862 0.40=0.236773 0.50=0.280231
epic:       0.10=0.111904 0.20=0.147736 0.30=0.182191 0.40=0.228757 0.50=0.283709
cacheblend speedups: 3.781x 2.842x 2.357x 1.910x 1.614x
epic speedups:       4.041x 3.061x 2.482x 1.977x 1.594x
monotonic=true
failures=none
server_env=[]
```

No `direct mapped`, `async external`, `ERROR`, or `failed` lines were present in the accepted server log.

Broader Jetson-only default-context refresh:

```text
run_id=jetson_minicpm_formal_contexts_q4k16_20260627_1
contexts=512,1024,1536,2048,2560
modes=normal-full-recompute,cacheblend,epic
ratios=0.10,0.20,0.30,0.40,0.50
server_env=[]
summary_rows=55
failures=none
```

All cacheblend and epic latency series were monotonic for every context.
Minimum sparse speedups versus same-context normal full recompute:

```text
ctx512:  cacheblend min=1.497x, epic min=1.610x
ctx1024: cacheblend min=2.478x, epic min=2.550x
ctx1536: cacheblend min=1.870x, epic min=2.902x
ctx2048: cacheblend min=1.783x, epic min=3.056x
ctx2560: cacheblend min=1.615x, epic min=3.166x
```

The run was merged into `benchmark.csv` with:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/merge_prefill_benchmark_csv.py \
  --allow-nonformal \
  benchmark.csv \
  .cache/jetson_qtile_ab/jetson_minicpm_formal_contexts_q4k16_20260627_1/summary.csv
```

Merge impact relative to the pre-merge local CSV copy:

```text
changed_rows=55
added_rows=0
removed_rows=0
unexpected_changed_rows=0
```

## 5% budget and ctx3072 high-risk refresh

After the 10%-50% default-context refresh, old `benchmark.csv` rows still existed for
5% budget and ctx3072. The initial 5%-only run exposed a real small-active-row
anomaly:

```text
run_id=jetson_minicpm_budget05_contexts_q4k16_20260627_1
ctx512 cacheblend05=0.178721, cacheblend10 previous=0.126830
ctx512 epic05=0.162824, epic10 previous=0.111596
ctx1024 cacheblend05=0.456610, cacheblend10 previous=0.242442
ctx1024 epic05=0.437782, epic10 previous=0.218097
```

This violated the expected budget monotonicity for small contexts, even though
all rows remained faster than normal full-compute. A repeat run showed the issue
was concentrated in ctx512 cacheblend 5%:

```text
run_id=jetson_minicpm_budget05_repeat_small_q4k16_20260627_1
ctx512 cacheblend05=0.188410, cacheblend10=0.150895, monotonic=false
ctx512 epic05=0.124334, epic10=0.136548, monotonic=true
ctx1024 cacheblend05=0.208365, cacheblend10=0.234523, monotonic=true
ctx1024 epic05=0.201057, epic10=0.211131, monotonic=true
```

The ctx512 cacheblend metadata showed selected PIC rows of 25 for 5% and 50 for
10%, both with max logical position 256. That means 5% was not doing more true
causal K work; the regression was a tiny-active-row kernel/occupancy issue.

The qtile A/B isolated the cause:

```text
run_id=jetson_minicpm512_budget05_qtile_off_20260627_1
server_env=MNN_CUDA_PAGED_ATTENTION_QTILE_VARIANT=off
ctx512 cacheblend05=0.122267, cacheblend10=0.130057
ctx512 epic05=0.111278, epic10=0.126526
```

Fix: CUDA `selectCudaSparseQTileVariant` now keeps the legacy split path for
default/auto `head_dim=128 && attnLen < 64`, while still honoring explicit
developer-forced qtile variants. This preserves qtile for high-budget sparse
attention but avoids the tiny-row occupancy cliff.

Build and sync:

```bash
cmake --build .cache/build/mnn/jetson_cross_cuda_proto \
  --target MNN_Cuda_Main/fast \
  --parallel 96
cp -a .cache/build/mnn/jetson_cross_cuda_proto/source/backend/cuda/libMNN_Cuda_Main.so \
  .cache/output/mnn/artifacts/jetson_cross_cuda_gcc9/lib/
rsync -a --delete .cache/output/mnn/artifacts/jetson_cross_cuda_gcc9/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

Default no-env validation after the guard:

```text
run_id=jetson_minicpm512_tiny_qtile_guard_20260627_1
ctx512 cacheblend05=0.127653, cacheblend10=0.128897, monotonic=true
ctx512 epic05=0.113568, epic10=0.115133, monotonic=true
failures=none
```

High-risk ctx3072 coverage:

```text
run_id=jetson_minicpm3072_q4k16_20260627_1
normal=14.171284891s
cacheblend: 0.10=2.553866 0.20=4.429644 0.30=6.282745 0.40=7.976254 0.50=9.322885
epic:       0.10=0.958321 0.20=1.415365 0.30=2.208434 0.40=3.194247 0.50=4.373687
failures=none
```

Post-guard ctx3072 50% smoke kept the high-budget qtile path healthy:

```text
run_id=jetson_minicpm3072_tiny_qtile_guard_smoke_20260627_1
cacheblend50=9.288044, speedup_vs_normal=1.526x
epic50=4.395712, speedup_vs_normal=3.224x
failures=none
```

The validated merged summary was:

```text
.cache/jetson_qtile_ab/jetson_minicpm_validated_q4k16_20260627_1/summary.csv
rows=25
```

After merging those rows into `benchmark.csv`, all Jetson MiniCPM5-1B CUDA
budgets 5%-50% for contexts 512,1024,1536,2048,2560,3072 are monotonic and
faster than same-context normal full-compute by at least 1.4x:

```text
ctx512:  cacheblend min=1.497x, epic min=1.610x
ctx1024: cacheblend min=2.478x, epic min=2.550x
ctx1536: cacheblend min=1.870x, epic min=2.902x
ctx2048: cacheblend min=1.783x, epic min=3.056x
ctx2560: cacheblend min=1.615x, epic min=3.166x
ctx3072: cacheblend min=1.526x, epic min=3.224x
violations=0
```

## Follow-Up

- The ctx512 debug-only sweeps above remain diagnostic. The broader Jetson-only default-context run was merged only because the user explicitly requested a benchmark CSV refresh and the merge used `--allow-nonformal`.
- A fully formal regression still requires the matching OrangePi sweep before treating the CSV as a complete formal matrix update.
