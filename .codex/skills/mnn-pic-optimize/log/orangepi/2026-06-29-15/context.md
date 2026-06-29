# OrangePi Qwen3-4B Cacheblend Top-K Mismatch

## Original Failure Shape

Run:

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_orangepi.sh \
  --model-key qwen3-4b \
  --contexts 512,1024,1536,2048 \
  --benchmark-csv benchmark.csv \
  --only-missing-contexts \
  --run-id orangepi_qwen3_4b_ctx2048_20260629
```

Observed:

```text
ctx1536: normal row only; first cacheblend 0.05 warm failed with curl 52.
ctx2048: cacheblend 0.05/0.10/0.20 warm succeeded, cacheblend 0.30 failed with curl 52.
After ctx2048 failure: subsequent requests saw connection refused, and SSH later showed a fresh boot.
```

Important server log from the old artifact:

```text
OpenCLPagedAttention cacheblend staged top-k produced invalid selected rows,
retrying legacy path layer=1 pic_tokens=2034 top_k=407 family=stage1024
reason=out_of_range at=256 index=1069198121
```

## Top-K Comparison Diagnostic

Added diagnostic env:

```text
MNN_PAGED_ATTENTION_COMPARE_CACHEBLEND_TOPK=1
```

When the selected path is staged, `runCacheBlendScoring()` now saves staged indices, runs legacy top-k on the same `mCacheBlendScores`, validates both lists, and prints ordered/set comparison.

Diagnostic command:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices \
  --model-key qwen3-4b \
  --contexts 2048 \
  --modes cacheblend --ratios 0.20 \
  --benchmark-csv benchmark.csv --only-missing-contexts \
  --no-warm --restart-server-each-spec \
  --server-env MNN_PAGED_ATTENTION_COMPARE_CACHEBLEND_TOPK=1 \
  --run-id orangepi_qwen3_4b_topk_compare_ctx2048_cb20_20260629
```

Result:

```text
OpenCLPagedAttention cacheblend top-k compare mismatch layer=1 pic_tokens=2034 top_k=407
staged=stage1024 staged_err=0 staged_reason=
legacy_err=0 legacy_reason=
ordered_equal=0 set_equal=0 first_mismatch=0
staged_sample=[1854,1468,700,1614,766,616,552,2030,238,1244,1548,414,308,1788,1738,886,...]
legacy_sample=[1879,1468,1880,700,1472,1881,766,1473,616,1474,552,1475,1882,238,1476,1883,...]
```

Conclusion: staged top-k is not only occasionally out-of-range on Mali; it can return a valid but different top-k set from legacy for the same scores.

## Fix

In `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`:

```cpp
if (runtime->getGpuType() == GpuType::MALI) {
    // Mali-G610 returns different or even invalid indices from the staged
    // path at high context lengths. Keep Mali on the deterministic legacy
    // top-k until the staged kernel is fixed.
    return false;
}
```

This leaves staged top-k available for non-Mali devices, but makes OrangePi/Mali use the legacy deterministic top-k path.

## Build And Sync

```bash
MNN_TARGET_DEVICE=orangepi5plus \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

The script reconfigured the OrangePi build directory because cached cross compiler/sysroot fields were empty. The rebuilt `pic_server`, `libMNN_CL.so`, and `libpic_llm.so` were synced to the fixed OrangePi artifact path.

## Validation

### ctx2048 cacheblend 0.20 profile

Command used `MNN_PAGED_ATTENTION_PROFILE=1`:

```text
OpenCLPagedAttention profile op=cacheblend_score layer=1 pic_tokens=2034 top_k=407
topk_path=legacy stage_candidates=0 stage_block=0 stage_sort=0
```

Summary:

```text
context=2048 mode=cacheblend budget=0.20 prefill_latency_s=95.588108 status=200
```

This run did not enable `MNN_PAGED_ATTENTION_PROFILE_DETAIL=1`, so `topk_us=0` in that log is not a breakdown. It only proved the fixed path selected `topk_path=legacy`.

### ctx1536 cacheblend 0.05 smoke

Old path failed immediately with empty reply. Fixed path:

```text
context=1536 mode=cacheblend budget=0.05 prefill_latency_s=92.770805 status=200
```

No `staged top-k`, `invalid`, `ERROR`, `Empty reply`, or `connection refused` lines were found in the saved logs.

### ctx2048 cacheblend 0.30 smoke

Old path failed around this request and left subsequent requests refused. Fixed path:

```text
context=2048 mode=cacheblend budget=0.30 prefill_latency_s=128.091522 status=200
```

After the request:

```text
SSH_OK
up 46 min
no pic_server or llm_bench process left on port 18132
```

## Formal Benchmark Fill

After the Mali staged top-k guard was rebuilt and synced, OrangePi Qwen3-4B `ctx1536/2048` cacheblend/epic/full-reuse rows were rerun with the warmed benchmark flow and merged into `benchmark.csv`.

Source summary:

```text
.cache/latency_budget_20260625/orangepi_qwen3_4b_ctx2048_pic_fill_after_topk_fix_20260629/summary.csv
```

Key `ctx2048` rows:

```text
full-reuse: 3.563025s
cacheblend 0.05: 16.068888s
cacheblend 0.10: 23.327970s
cacheblend 0.20: 33.997339s
cacheblend 0.30: 42.794835s
cacheblend 0.40: 54.396225s
cacheblend 0.50: 70.185120s
epic 0.05: 11.947038s
epic 0.10: 15.713916s
epic 0.20: 21.069722s
epic 0.30: 29.288296s
epic 0.40: 38.195540s
epic 0.50: 50.159099s
```

## Current Bottleneck Profile

The follow-up profile was added specifically to answer whether the slower post-fix path is actually bottlenecked by top-k.

Command:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices \
  --model-key qwen3-4b \
  --contexts 2048 \
  --modes cacheblend --ratios 0.20 \
  --benchmark-csv benchmark.csv \
  --restart-server-each-spec \
  --server-env MNN_PAGED_ATTENTION_PROFILE=1 \
  --server-env MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 \
  --server-env MNN_PIC_GRAPH_PROFILE=1 \
  --server-env MNN_PIC_GRAPH_PROFILE_TOP=120 \
  --run-id orangepi_qwen3_4b_ctx2048_cb20_detail_graph_20260629
```

Artifacts:

```text
.cache/latency_budget_20260625/orangepi_qwen3_4b_ctx2048_cb20_detail_graph_20260629/summary.csv
.cache/latency_budget_20260625/orangepi_qwen3_4b_ctx2048_cb20_detail_graph_20260629/raw/orangepi/pic_server_ctx2048_cacheblend_0p20.full.log
```

Measure request result:

```text
ctx2048 cacheblend 0.20 prefill_latency_s=35.098518 status=200
```

PagedAttention detail, measure request:

```text
cacheblend_score: 0.125770s
  read_us=0.005902s
  score_kernel_us=0.002141s
  topk_us=0.117482s
  readback_us=0.000171s
  topk_path=legacy

layer0 prefill_attention_fast_qk_softmax_qkv: 4.560301s
score_qsplit_attention layer1: 0.423093s
  qk_us=0.221687s
  softmax_us=0.026102s
  qkv_us=0.167065s

layer>=2 sparse_flash_attention: 18.278s across 34 calls
  flash_us=18.068s
  rearrange_us=0.061s
  pack_us=0.120s
  qk_active_tiles/qk_rect_tiles=969782/1068314=0.908

hydrate: 0.080s across 34 calls
```

Graph profile, measure request:

```text
MNN_PIC_GRAPH_PROFILE_SUMMARY request=2 total_ms=34668.226 calls=1602
PicSparseAttention: 18410.169ms
Convolution:         9586.675ms
PagedAttention:       4561.005ms
Raster:                937.390ms
PicScoreAttention:     554.795ms
BinaryOp:              208.288ms
While:                 171.672ms
LayerNorm:             116.313ms
UnaryOp:                88.788ms
Cast:                   33.131ms
```

Top graph ops were one full layer0 `PagedAttention` at `4561.005ms`, then `PicSparseAttention` layers at roughly `0.49s-0.75s` each. Compact dense is the second major bucket: graph profile reports `Convolution=9.587s`; within the top-120 ops, compact MLP contributes at least:

```text
compact mlp/gate_proj: 1183.692ms
compact mlp/up_proj:   1157.974ms
compact mlp/down_proj: 2096.603ms
layer0 Convolution:    1258.060ms
```

No `ERROR`, `invalid`, `target unavailable`, `async persistent`, `Empty reply`, or `connection refused` lines were present in the full log.

Conclusion:

- Top-k definitely caused the earlier crash/incorrectness, because staged top-k returned out-of-range rows and also disagreed with legacy on the selected set.
- Current `ctx2048/cacheblend0.20` latency is not top-k-bound. Legacy top-k is about `0.117s`, roughly `0.33%` of the `35.099s` prefill request.
- The primary bottleneck is later `PicSparseAttention` (`18.410s` graph total). The second bottleneck is OpenCL `Convolution`/compact dense (`9.587s` graph total), especially compact MLP.
- Cacheblend selected rows still produce a high K-work footprint: `qk_active_tiles/qk_rect_tiles=0.908`. This means active rows are sparse in row count but broad in logical coverage, so sparse attention still streams nearly full K ranges.
- Hydrate and cacheblend score/top-k are not first-order optimization targets for this case.

### ctx2048 cacheblend 0.50 profile

The matching high-budget profile was added to cover the original `0.20/0.50` bottleneck attribution target:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices \
  --model-key qwen3-4b \
  --contexts 2048 \
  --modes cacheblend --ratios 0.50 \
  --benchmark-csv benchmark.csv \
  --restart-server-each-spec \
  --server-env MNN_PAGED_ATTENTION_PROFILE=1 \
  --server-env MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 \
  --server-env MNN_PIC_GRAPH_PROFILE=1 \
  --server-env MNN_PIC_GRAPH_PROFILE_TOP=200 \
  --run-id orangepi_qwen3_4b_ctx2048_cb50_detail_graph_20260629
```

Artifacts:

```text
.cache/latency_budget_20260625/orangepi_qwen3_4b_ctx2048_cb50_detail_graph_20260629/summary.csv
.cache/latency_budget_20260625/orangepi_qwen3_4b_ctx2048_cb50_detail_graph_20260629/raw/orangepi/pic_server_ctx2048_cacheblend_0p50.full.log
```

Measure request result:

```text
ctx2048 cacheblend 0.50 prefill_latency_s=46.286698 status=200
```

PagedAttention detail, measure request:

```text
layer0 prefill_attention_fast_qk_softmax_qkv: 4.451136s

cacheblend_score: 0.741065s
  read_us=0.010370s
  score_kernel_us=0.002665s
  topk_us=0.727736s
  readback_us=0.000189s
  topk_path=legacy

layer>=2 sparse_flash_attention: 13.490870s across 34 calls
  variant=mqtile_hd128_q8k16
  flash_us=12.973580s
  rearrange_us=0.192087s
  pack_us=0.148912s
  qk_active_tiles/qk_rect_tiles=2239478/2351100=0.953

hydrate: 0.094701s across 34 calls
```

Graph profile, measure request:

```text
MNN_PIC_GRAPH_PROFILE_SUMMARY request=2 total_ms=45843.640 calls=1602
Convolution:        22530.618ms
PicSparseAttention: 13696.305ms
PagedAttention:      4451.515ms
Raster:              1969.992ms
PicScoreAttention:   1712.864ms
BinaryOp:             607.854ms
While:                316.168ms
UnaryOp:              267.603ms
LayerNorm:            253.432ms
Cast:                  37.289ms
```

Top-200 graph op bucket sums:

```text
mlp/gate_proj:   5600.954ms
mlp/up_proj:     5597.723ms
mlp/down_proj:   5303.176ms
self_attn/q_proj 1706.439ms
self_attn/o_proj 2148.118ms
```

The full remote log contained two profiled requests; the measure request is the second one. No `ERROR`, `invalid`, `target unavailable`, `async persistent`, `CL_OUT`, `Empty reply`, `HTTP_STATUS:000`, or `curl:` lines were present in the saved local run directory.

Conclusion:

- `ctx2048/cacheblend0.50` is also not top-k-bound. Legacy top-k is `0.728s`, about `1.57%` of the `46.287s` prefill request.
- At high budget, compact dense becomes the largest bucket: `Convolution=22.531s` versus `PicSparseAttention=13.696s`.
- The selected-token distribution is even more K-heavy than cb20: `qk_active_tiles/qk_rect_tiles=0.953`, so sparse attention still scans almost full K coverage.
- The next useful optimization target for cb50 is compact dense / MLP routing first, then sparse attention K-work reduction. Top-k remains a correctness guardrail, not the primary performance lever.

## Mali qtile Sparse Attention A/B

The first A/B kept production routing on row32, then used the explicit bench override to test Mali qtile variants:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices \
  --model-key qwen3-4b \
  --contexts 2048 \
  --modes cacheblend --ratios 0.20 \
  --benchmark-csv benchmark.csv \
  --restart-server-each-spec \
  --server-env MNN_PAGED_ATTENTION_PROFILE=1 \
  --server-env MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 \
  --server-env MNN_PAGED_ATTENTION_BENCH_FORCE_SPARSE_FLASH_VARIANT=<variant> \
  --run-id <run_id>
```

Measured request, `ctx2048/cacheblend0.20`:

```text
variant            end_to_end_s  later_sparse_s  flash_s    topk_s   pieces
row32              33.949952     17.833224       17.609850  0.117385 row32=272
mqtile_hd128_q4k16 24.656505      7.596648        7.327586  0.117199 q4k16=272
mqtile_hd128_q4k8  25.523433      7.833833        7.499404  0.128628 q4k8=272
mqtile_hd128_q8k16 22.880143      5.906905        5.625730  0.117336 q8k16=272
```

All qtile runs had:

```text
errors=0
qk_active_tiles/qk_rect_tiles=0.906959
```

So the speedup is from the qtile kernel implementation, not from changing cacheblend selected rows or K-range coverage.

Follow-up forced `mqtile_hd128_q8k16` validation covered the requested OrangePi Qwen3-4B cacheblend contexts and a high-budget case:

```text
context  budget  end_to_end_s  later_sparse_s  flash_s    score_s   topk_s   hydrate_s  active/rect  errors
1536     0.20    15.759642      3.357829       3.157059   0.216864  0.052064 0.073994   0.874054     0
1536     0.50    30.500866      6.879258       6.458710   0.464888  0.311791 0.084228   0.939281     0
2048     0.20    23.442063      6.021091       5.704141   0.405020  0.117443 0.097873   0.906959     0
2048     0.50    45.253128     13.413220      12.901442   0.950558  0.727585 0.093971   0.952111     0
```

Device remained reachable after the qtile runs. `dmesg` showed Mali activity/frequency warnings but no reset/OOM/OpenCL crash signature.

## Routing Change

The production route was changed in `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp` with explicit Mali/Adreno isolation:

- `MNN_PAGED_ATTENTION_BENCH_FORCE_SPARSE_FLASH_VARIANT` now prints `variant_source=bench_override`, so forced experiments are not misreported as default.
- Mali `headDim=128` later sparse (`queryRowsAreFull=false`) defaults to `mqtile_hd128_q8k16` only when `kvLen<=2048`.
- Mali `headDim=128` score-layer full-Q was left on the existing score-family route after A/B; if forced to score flash, the flash variant is row32. Contexts larger than 2048 remain on row32.
- Mali sparse variant candidate/cache replay for this bucket only accepts `mqtile_hd128_q8k16`; stale row32 tune cache cannot override the new default back to the slow path.
- Adreno route remains in the existing Adreno helpers (`_preferAdrenoScoreSparseFlash`, `_preferAdrenoLaterSparseQ4K8`, image/KV-image fallbacks) and was not coupled to the Mali helper.

Default-route smoke after rebuild and sync:

```text
run_id=orangepi_qwen3_4b_ctx2048_cb20_mali_default_q8k16_20260629
ctx2048/cacheblend0.20 prefill_latency_s=24.121404
variant=mqtile_hd128_q8k16
variant_source=default
row32_pieces=0
mqtile_hd128_q8k16_pieces=272
later_sparse_attention=6.025989s
errors=0
```

### Mali score-layer family A/B

Because the repository target mentions score-layer flash, a separate A/B forced only the score-layer sparse family to `flash` while leaving later sparse on the default Mali q8k16 route:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices \
  --model-key qwen3-4b \
  --contexts 2048 \
  --modes cacheblend --ratios 0.20,0.50 \
  --benchmark-csv benchmark.csv \
  --restart-server-each-spec \
  --server-env MNN_PAGED_ATTENTION_PROFILE=1 \
  --server-env MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 \
  --server-env MNN_PAGED_ATTENTION_BENCH_FORCE_SCORE_SPARSE_FAMILY=flash \
  --run-id orangepi_qwen3_4b_ctx2048_score_flash_ab_20260629
```

Result:

```text
forced score_flash cb20: 23.889652s
forced score_flash cb50: 46.093726s
```

Remote full-log measure details:

```text
cb20 score_flash_attention layer=1 variant=row32 us=543953 flash_us=525167
cb50 score_flash_attention layer=1 variant=row32 us=1889099 flash_us=1870495
```

Compared with the earlier default score-family detail-profile q8k16 validation:

```text
default score qsplit cb20: 23.442063s
default score qsplit cb50: 45.253128s
```

Conclusion:

- For Mali Qwen3-4B `ctx2048`, forcing score-layer flash is not a win under the same detail-profile style.
- The production change therefore stays limited to later sparse `mqtile_hd128_q8k16`; Mali score-layer family is not changed.
- This keeps Mali and Adreno isolated: Adreno can continue using its score-flash preference helper, while Mali does not inherit that route without evidence.

## Benchmark Refresh

No-profile default-route run:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices \
  --model-key qwen3-4b \
  --contexts 1536,2048 \
  --modes cacheblend --ratios 0.20,0.50 \
  --benchmark-csv benchmark.csv \
  --restart-server-each-spec \
  --run-id orangepi_qwen3_4b_ctx1536_2048_cb20_cb50_mali_default_q8k16_formal_20260629
```

Merged into `benchmark.csv`:

```text
1536 cacheblend 0.20: 16.480467s, 93.201242 tok/s
1536 cacheblend 0.50: 29.126481s, 52.735516 tok/s
2048 cacheblend 0.20: 23.219433s, 88.201982 tok/s
2048 cacheblend 0.50: 44.708272s, 45.808078 tok/s
```

Remaining no-profile cacheblend ratios were then rerun on the same default q8k16 route:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices \
  --model-key qwen3-4b \
  --contexts 1536,2048 \
  --modes cacheblend --ratios 0.05,0.10,0.30,0.40 \
  --benchmark-csv benchmark.csv \
  --restart-server-each-spec \
  --run-id orangepi_qwen3_4b_ctx1536_2048_cacheblend_remaining_mali_default_q8k16_formal_20260629
```

Final `benchmark.csv` OrangePi Qwen3-4B cacheblend rows for these contexts:

```text
1536 cacheblend 0.05:  7.993061s, 192.166681 tok/s
1536 cacheblend 0.10: 11.077847s, 138.655101 tok/s
1536 cacheblend 0.20: 16.480467s,  93.201242 tok/s
1536 cacheblend 0.30: 20.651151s,  74.378421 tok/s
1536 cacheblend 0.40: 24.587187s,  62.471563 tok/s
1536 cacheblend 0.50: 29.126481s,  52.735516 tok/s
2048 cacheblend 0.05: 12.147464s, 168.594861 tok/s
2048 cacheblend 0.10: 17.124267s, 119.596360 tok/s
2048 cacheblend 0.20: 23.219433s,  88.201982 tok/s
2048 cacheblend 0.30: 29.247740s,  70.022504 tok/s
2048 cacheblend 0.40: 34.940372s,  58.614144 tok/s
2048 cacheblend 0.50: 44.708272s,  45.808078 tok/s
```

Compared with the previous row32 rows:

```text
1536 cacheblend 0.20: 20.833902s -> 16.480467s
1536 cacheblend 0.50: 43.481204s -> 29.126481s
2048 cacheblend 0.20: 33.997339s -> 23.219433s
2048 cacheblend 0.50: 70.185120s -> 44.708272s
```

The raw ssh tunnel logs contain short startup `Connection refused` retries before the server is ready; the benchmark requests succeeded, server logs did not contain `ERROR` / `invalid` / `CL_OUT` / `Empty reply`, and OrangePi uptime did not reset.

## Epic vs Cacheblend Comparison

The pasted `benchmark.csv` excerpt made OrangePi Qwen3-4B `ctx2048` look like epic was slower than cacheblend at high budgets:

```text
cacheblend 0.40: 34.940372s
cacheblend 0.50: 44.708272s
epic      0.40: 38.195540s
epic      0.50: 50.159099s
```

Root cause: these rows were from different routing generations.

- Current cacheblend rows came from the Mali default q8k16 refresh:
  - `orangepi_qwen3_4b_ctx1536_2048_cb20_cb50_mali_default_q8k16_formal_20260629`
  - `orangepi_qwen3_4b_ctx1536_2048_cacheblend_remaining_mali_default_q8k16_formal_20260629`
- The epic rows were still from the older run:
  - `orangepi_qwen3_4b_ctx2048_pic_fill_after_topk_fix_20260629`

Within the older run, epic was not slower than cacheblend:

```text
old cacheblend 0.40: 54.396225s
old cacheblend 0.50: 70.185120s
old epic      0.40: 38.195540s
old epic      0.50: 50.159099s
```

To close the loop, current default epic was rerun with the same production route used after the q8k16 Mali change:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices \
  --model-key qwen3-4b \
  --contexts 2048 \
  --modes epic --ratios 0.05,0.10,0.20,0.30,0.40,0.50 \
  --benchmark-csv benchmark.csv \
  --restart-server-each-spec \
  --run-id orangepi_qwen3_4b_ctx2048_epic_mali_default_q8k16_formal_20260629
```

Summary:

```text
epic 0.05: 11.315142s, 180.996403 tok/s
epic 0.10: 14.396486s, 142.256937 tok/s
epic 0.20: 17.765212s, 115.281484 tok/s
epic 0.30: 24.081450s,  85.044713 tok/s
epic 0.40: 29.603160s,  69.181804 tok/s
epic 0.50: 36.830896s,  55.605489 tok/s
```

Error scan only matched timeout configuration fields in `device.json`; no request or server error lines were found:

```bash
rg -n "ERROR|invalid|target unavailable|async persistent|CL_OUT|Empty reply|HTTP_STATUS:000|curl:|Cache invalid|timeout|killed|Traceback|Exception|failed" \
  .cache/latency_budget_20260625/orangepi_qwen3_4b_ctx2048_epic_mali_default_q8k16_formal_20260629
```

After merging the new summary into `benchmark.csv`, current `ctx2048` high-budget comparison is:

```text
cacheblend 0.40: 34.940372s
epic      0.40: 29.603160s
cacheblend 0.50: 44.708272s
epic      0.50: 36.830896s
```

Conclusion: there is no evidence that current epic is slower than current cacheblend. The apparent slowdown was a stale `benchmark.csv` composition problem: q8k16-refreshed cacheblend was compared against older epic data.

## Row32 / Row8 / Row16 Rationale

`row32` does not mean 32 token rows. In the legacy OpenCL sparse flash family, it is the fixed workgroup K-lane width:

```text
sparse_flash_attention_row32: local size {32, 1, 1}, each lane scans k += 32
sparse_flash_attention_row64: local size {64, 1, 1}, each lane scans k += 64
```

The current implementation has no `row8` or `row16` sparse flash kernel to route to. Adding them would require new OpenCL kernels, local reduction logic, dispatch cases, tune keys, and per-device validation. It is not just adding more candidate names to `_sparseFlashVariantCandidates`.

For the Qwen3-4B `headDim=128` hot shape, reducing the legacy row lane width is also not the first likely win. The legacy row kernel computes one query row per workgroup, so smaller lane width would reduce parallelism over K and usually makes long-context K scans slower unless occupancy/local-memory pressure was the actual bottleneck. The profile showed the issue was kernel efficiency at high K coverage (`qk_active_tiles/qk_rect_tiles≈0.907`), and the existing qtile kernels directly attack that by computing multiple Q rows per workgroup with a fixed K tile:

```text
mqtile_hd128_q4k16: local size {16, 4, 1}
mqtile_hd128_q4k8:  local size {16, 4, 1}
mqtile_hd128_q8k16: local size {16, 8, 1}
```

That is why the production change uses the already-implemented and measured `mqtile_hd128_q8k16` for the validated Mali bucket instead of inventing `row8/row16`. The conservative gate remains `Mali && headDim=128 && later sparse && kvLen<=2048`; larger Mali contexts and score-layer full-Q still stay on row32 until separately validated.

## Rewritten Goal

Restore OrangePi/Mali Qwen3-4B `ctx1536/2048` cacheblend stability and performance without assuming top-k is the performance bottleneck.

Required details:

1. Keep the Mali staged top-k path disabled or guarded until its kernel is proven deterministic against legacy. The old staged path caused crashes because it could return out-of-range local row indices such as `1069198121`; graph-boundary PIC then treated those indices as active logical rows, which could make `PagedKVMeta` validation fail, return `INVALID_VALUE`, produce empty HTTP replies, and destabilize the service. It also sometimes returned legal but different top-k sets, so correctness was already broken even when it did not crash.
2. Treat current performance as a separate bottleneck problem. The latest profile proves `topk_us=117.482ms` is too small to explain `35.098518s`; optimize only after attributing the dominant buckets.
3. First optimize `PicSparseAttention layer>=2`: reduce true QK/QKV work or improve the qtile/range schedule. Because `qk_active_tiles/qk_rect_tiles=0.908`, simply counting active rows is misleading; cacheblend row distribution keeps K coverage high.
4. In parallel, inspect compact dense/MLP. `Convolution=9.587s` is the second largest graph bucket, and compact `mlp/{gate,up,down}_proj` rows at `421x...` appear repeatedly in the top ops.
5. Only revisit a Mali-safe fast top-k after sparse attention and compact dense are no longer dominant, or if a later profile shows top-k has grown to a meaningful fraction of request time.

## Remaining Work

- Do not make top-k the next optimization target for this case unless new evidence changes the profile.
- Keep the Mali q8k16 default gate at `kvLen<=2048` until Qwen3-8B and/or `ctx2560` high-budget runs are explicitly validated. The older Qwen3-8B hang was later attributed to static sparse scratch memory, but this does not automatically prove all larger Mali qtile shapes are production-safe.
- For `PicSparseAttention`, inspect whether q-range grouping or cacheblend-specific active row ordering can reduce actual K work without changing ordinary cacheblend semantics; qtile improved kernel efficiency but did not change `qk_active_tiles/qk_rect_tiles`.
- For compact dense, add a focused graph/profile pass that sums all `layers.* mlp/{gate,up,down}_proj` and self-attn projection `Convolution` rows, not only top-120, then check whether `pic_gemm_b4_c8_*` is choosing good tune keys for `M=421`.
- Keep the staged-vs-legacy top-k comparison diagnostic available for any future Mali-safe top-k candidate.
