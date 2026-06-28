# Rhino 2026-06-27 00 Long Context

以下内容从旧的 `OPTIMIZATION_LOG.md` 迁移，按本设备/日期归档。

## 2026-06-27 Experiment: Rhino / Adreno compact dense b2 family bring-up

Goal:

- Test whether a lower-register-pressure Adreno compact dense variant (`pic_gemm_b2_c8_int4_buf`) can beat the existing compact dense families on Rhino Pi X1 for headDim=128 sparse PIC shapes.
- Keep the routing inside warmup/tune; do not add request-time env branches.

Code changes:

- Added `pic_gemm_b2_c8_int4_buf` and `kCompactDenseFamilyPicQuantB2`.
- Added Adreno family candidate comparison:
  - `generic_quant`
  - `pic_quant`
  - `pic_quant_b2`

## 2026-06-27 Experiment: Adreno tune selection should measure steady state

Goal:

- Fix a likely Rhino / Adreno tune-selection bug where score-layer family tuning prefers `qsplit` because the first `flash` candidate run also pays nested schedule/variant/image setup cost.

Code changes:

- `PagedAttentionBufExecution.cpp` now warms a candidate once and measures the second run when recording tune cache for:
  - `paged_score_sparse_family_*`
  - `paged_sparse_flash_schedule_*`
  - `paged_sparse_flash_variant_*`
  - `paged_sparse_qsplit_chunk_*`
  - `paged_cacheblend_topk_family_*`

Reasoning:

- On Adreno, `flash` candidates can trigger deeper first-use work than `qsplit`, especially once mixed `buffer + image` sparse-flash variants are in the candidate set.
- Measuring the very first candidate execution poisons the tune cache toward the cheaper cold-start path instead of the faster steady-state path.
- This does not change formal execution semantics; it only changes how warmup chooses the cached implementation.

Next:

- Re-run Rhino warmup/profile and inspect whether score layer stops sticking to `qsplit`.
- If score layer still prefers `qsplit`, then the remaining work is the kernel path itself, not tune selection fairness.

Continuation of the earlier compact-dense bring-up notes:

- `fp_weight`
- Fixed `opencl_codegen.py` to always write `opencl_source_map.hpp` into `source/backend/opencl/execution/cl/` instead of the caller cwd.
- Hardened `ConvBufLowMemoryExecution::tuneGemmLowMemory()` so unavailable candidate kernels are skipped or downgraded instead of dereferencing a null kernel.

Important bug found:

- The first Rhino run crashed during `cacheblend 0.10` only on `/v1/chat/completions`; `/v1/prefill/text` was fine.
- gdb backtrace on Rhino showed:
  - `getKernel: pic_gemm_b2_c8_int4_buf error, res:-46`
  - then `OpenCLRuntime::getMaxWorkGroupSize(nullptr)` from `ConvBufLowMemoryExecution::tuneGemmLowMemory()`
- Root cause was not the b2 kernel body itself. The earlier manual codegen invocation had written a new `opencl_source_map.hpp` into the repo root, leaving `source/backend/opencl/execution/cl/opencl_source_map.hpp` stale.
- Result:
  - compiled library still carried the old `gemm_conv1x1_buf` md5/program cache identity
  - Rhino loaded/reused an old OpenCL program binary without `pic_gemm_b2_c8_int4_buf`
  - `clCreateKernel` returned `CL_INVALID_KERNEL_NAME (-46)`
  - tune path crashed because kernel creation failure was not checked

Validation:

- Rebuilt `aidlux_adreno_opencl_compactb2`, rsynced to Rhino fixed artifact path, and reran:

```text
device: Rhino Pi X1
model: MiniCPM5-1B
context: 2560
mode: cacheblend
budget: 0.10
frequency: cpu=1670400/2323200/2592000,gpu=680000000,ddr=max
```

- New run:

```text
.cache/latency_budget_20260625/rhino_minicpm_compactb2_retest_20260627_022820/summary.csv
prefill_latency_s = 2.573437
```

- Previous reference:

```text
.cache/latency_budget_20260625/rhino_minicpm_topk_family_profile_20260627_014433/summary.csv
prefill_latency_s = 2.605505
```

- End-to-end improvement is small but real:

```text
2.605505 -> 2.573437  (about 1.25% faster)
```

What the family tune actually picked:

- `pic_quant_b2` is now buildable and benchmarked correctly on Rhino.
- For the repeated compact dense hot shapes at `batch=269`, Adreno heavy tune measured:

```text
269 x 2048 -> 1536 : generic 2376, pic 2378, b2 3230, fp_weight 2701
269 x 1536 -> 4608 : generic 4912, pic 4908, b2 7055, fp_weight 5760
269 x 4608 -> 1536 : generic 5394, pic 5395, b2 7303, fp_weight 6043
269 x 1536 -> 2048 : generic 2336, pic 2332, b2 3234, fp_weight 2778
```

Interpretation:

- The b2 family is valid on Adreno after the codegen/cache fix, but it is consistently slower than both existing quant families on the current hot compact-row shapes.
- Lowering the batch tile from `b4` to `b2` increased total time enough that reduced register pressure did not compensate for extra input/output overhead.
- This means the current Rhino dense hotspot is not primarily a `b4` register-pressure problem.

Current Rhino hotspot split after the fix:

```text
Convolution      total_ms = 781.583
PicSparseAttention total_ms = 615.755
Raster           total_ms = 488.797
PagedAttention   total_ms = 152.672
PicScoreAttention total_ms = 40.731
```

Conclusions:

- Keep `pic_quant_b2` only as an experimental family candidate for future shapes; do not expect it to win on the current `batch≈269` MiniCPM sparse shapes.
- The crash itself is fixed by:
  - correcting OpenCL source-map generation path
  - making dense family tune robust to unavailable kernels
- For Rhino / Adreno, the next dense direction should reduce output-channel accumulation pressure without halving the Q-row tile, for example a `pic_gemm_b4_c4_*` family or other wider-LWS / narrower-N variants.
- The next major optimization buckets on Rhino remain:
  - compact dense family expansion beyond `b2`
  - `PicSparseAttention`
  - `Raster`

## 2026-06-27 Experiment: Rhino / Adreno compact dense c4 family validation

Goal:

- Validate whether lowering the compact dense output-channel accumulation tile from `c8` to `c4`
  can beat the existing Adreno compact dense families on the current Rhino hot sparse shape.
- Keep the decision inside heavy warmup/tune; do not add request-time switches.

Code path under test:

- Added `pic_gemm_b4_c4_int4_buf` and `kCompactDenseFamilyPicQuantC4`.
- Rebuilt OpenCL embedded source map with:

```bash
python3 source/backend/opencl/execution/cl/opencl_codegen.py source/backend/opencl/execution/cl
```

- Local sanity compile passed:

```bash
cmake --build .cache/build/mnn/x64_opencl_check --target MNN_CL --parallel <half-cpus>
```

- Rebuilt Rhino artifact in the production Adreno build/install path and resynced to:

```text
/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl
```

Focused validation run:

```text
device: Rhino Pi X1
model: MiniCPM5-1B
context: 2560
mode: cacheblend
budget: 0.10
frequency: cpu=1670400/2323200/2592000,gpu=680000000,ddr=max
run: rhino_minicpm_compactc4_retest_20260627_025318
summary: .cache/latency_budget_20260625/rhino_minicpm_compactc4_retest_20260627_025318/summary.csv
prefill_latency_s = 2.550989
```

Reference row from the previous `b2`-based bring-up:

```text
run: rhino_minicpm_compactb2_retest_20260627_022820
prefill_latency_s = 2.573437
```

Important interpretation:

- The new end-to-end latency is slightly lower (`2.573437 -> 2.550989`), but this is **not** because
  `pic_quant_c4` won the dense family tune.
- Rhino heavy tune measured `pic_quant_c4` and rejected it on every repeated hot `batch=269` compact shape:

```text
269 x 2048 -> 1536 : generic 2380, pic 2377, c4 3718, b2 3230, fp_weight 2695
269 x 1536 -> 4608 : generic 4908, pic 4911, c4 8102, b2 7055, fp_weight 5756
269 x 4608 -> 1536 : generic 5373, pic 5377, c4 8360, b2 7296, fp_weight 6046
269 x 1536 -> 2048 : generic 2331, pic 2333, c4 3720, b2 3239, fp_weight 2775
```

What Rhino actually selected:

- `2048 -> 1536` kept `family=pic_quant`, `compact_mode=pic_b4c8`
- `1536 -> 4608`, `4608 -> 1536`, `1536 -> 2048` kept `family=generic_quant`, `compact_mode=generic_b4c8`
- `weight_image=1` stayed enabled for these hot shapes, so this is not a “forgot to use image reads” issue

Current hotspot split on the measured request:

```text
Convolution        total_ms = 776.075
PicSparseAttention total_ms = 618.614
Raster             total_ms = 481.142
PagedAttention     total_ms = 156.805
PicScoreAttention  total_ms = 37.868
```

Additional runtime evidence:

- score layer full-Q flash path selected `variant=mqtile_hd128_q8k16`
- later sparse layers selected `variant=mqtile_hd128_q4k8`
- staged cacheblend top-k remained healthy:

```text
topk_path=stage2048 topk_us≈0.77-0.78 ms
```

Conclusions:

- `pic_quant_c4` is a valid Adreno candidate but it is decisively low-efficiency on the current Rhino hot shape family.
- The present Rhino bottleneck is still not “output-channel accumulation pressure inside `pic_b4_c8`”.
- The meaningful remaining Rhino work is:
  - keep dense routing on `pic_quant/generic_quant/fp_weight` for the current hot bucket
  - focus next on compact-row `Raster` reduction and layout fusion
  - continue sparse attention tuning as a secondary bucket, not as the only hotspot
- The fact that `weight_image=1` is already active means the next Adreno optimization should target
  buffer-native layout flow, raster elimination, or family-specific workgroup/vector strategies,
  not another “just switch to image reads” iteration.

## 2026-06-27 Experiment: Rhino / Adreno cacheblend top-k dynamic stage family

Goal:

- Address the confirmed Rhino-specific `cacheblend_score` hotspot without
  rewriting the current sparse-attention routing again.
- Make top-k selection route through warmup/tune, consistent with the existing
  score-family and sparse-flash tune flow.

Code/env:

- Code change:
  - `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
  - `source/backend/opencl/execution/cl/paged_attention_buf.cl`
- Added a new warmup-time family route:
  - `paged_cacheblend_topk_family_*`
  - current families:
    - `stage1024`
    - `stage2048`
- The old iterative `legacy` kernel remains only as fallback when staged top-k
  is unsupported.
- Reworked the staged kernels so they no longer hard-code:
  - stage1 local array size `2048`
  - stage2 local array size `4096`
- New staged top-k behavior:
  - family decides stage1 `block_size`
  - stage2 `sort_size = next_pow2(block_count * top_k)`
  - local memory is allocated dynamically per run via kernel args
  - profile log now prints:
    - `topk_path=<family>`
    - `stage_candidates=<n>`
    - `stage_block=<block_size>`
    - `stage_sort=<sort_size>`

Why this change:

- Rhino already showed that `cacheblend_score` was dominated by top-k, not the
  score kernel itself:
  - `score_kernel ~= 0.9 ms`
  - `topk ~= 63.7 ms`
- For shapes like Rhino `ctx2560/cacheblend0.10`:
  - `pic_tokens ~= 2546`
  - `top_k ~= 255`
  - old stage path only produced `~510-765` real candidates but still sorted a
    fixed `4096`-entry local buffer in stage2
- That is a structural mismatch for Adreno local memory / barrier cost, so this
  is the right place to spend the next optimization step.

Sanity:

- Regenerated embedded OpenCL sources with:

```bash
cd source/backend/opencl/execution/cl
python3 opencl_codegen.py .
```

- x64 OpenCL syntax sanity passed:

```bash
make -C .cache/build/mnn/x64_opencl_check -j8 MNN_CL
```

  - `PagedAttentionBufExecution.cpp.o` compiled
  - `paged_attention_buf_mnn_cl.cpp.o` compiled
  - `libMNN_CL.so` linked

- Local `aidlux_adreno_opencl` cross build is currently blocked by an existing
  toolchain/sysroot mismatch in that build directory:
  - missing `bits/types/struct___jmp_buf_tag.h`
  - mixed `bits/endian.h` include failure
  - this failure occurs while compiling unrelated OpenCL objects too, so it is
    not evidence against the top-k patch itself

Expected effect on Rhino:

- Directly reduce the measured Adreno cacheblend tax inside
  `cacheblend_score topk_us`.
- Preserve cacheblend semantics.
- Help most when the staged candidate set is far smaller than `4096`, which is
  exactly the current Rhino low/mid-budget pattern.

Next:

1. Produce a clean `aidlux_adreno_opencl` artifact from a non-broken build dir
   and deploy it to Rhino.
2. Re-run a focused Rhino profile:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices rhino \
  --model-key minicpm5-1b \
  --contexts 2560 \
  --modes cacheblend \
  --ratios 0.10,0.20,0.40,0.50 \
  --server-env MNN_OPENCL_TUNE_LEVEL=heavy \
  --remote-memory-limit-percent 95 \
  --frequency-profile cpu-low-gpu-max \
  --run-id rhino_minicpm_topk_family_$(date +%Y%m%d_%H%M%S)
```

3. Verify in Rhino profile logs:
  - `cacheblend_score topk_us` drops
  - tuned family is recorded in `topk_path`
  - `Convolution` / `Raster` remain the next bottlenecks after top-k shrinks

### 2026-06-27 Validation: Rhino top-k family landed, dense/image path verified

Artifact / deployment:

- Produced a clean Rhino build from:
  - `.cache/build/mnn/aidlux_adreno_opencl_topk_family`
  - install: `.cache/output/mnn/artifacts/aidlux_adreno_opencl_topk_family`
- Synced it to Rhino production artifact path:
  - `/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl`

Focused runtime validation:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices rhino --allow-subset-devices \
  --model-key minicpm5-1b \
  --contexts 2560 \
  --modes cacheblend --ratios 0.10 \
  --server-env MNN_OPENCL_TUNE_LEVEL=heavy \
  --server-env MNN_PAGED_ATTENTION_PROFILE=1 \
  --server-env MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 \
  --server-env MNN_PIC_GRAPH_PROFILE=1 \
  --server-env MNN_PIC_GRAPH_PROFILE_TOP=200 \
  --remote-memory-limit-percent 95 \
  --frequency-profile max
```

Measured row:

- run: `rhino_minicpm_topk_family_profile_20260627_014433`
- `cacheblend 0.10`: `2.605505 s`
- frequency note from remote query:
  - `cpu=1670400/2323200/2592000,gpu=680000000,ddr=max`

Key full-log evidence on Rhino:

- warm request:
  - `op=cacheblend_score ... topk_us=813 ... topk_path=stage2048 stage_candidates=510 stage_sort=512`
- measured request:
  - `op=cacheblend_score ... topk_us=1029 ... topk_path=stage2048 stage_candidates=510 stage_sort=512`

Interpretation:

- The new staged top-k family worked exactly as intended.
- Rhino old cacheblend top-k hotspot (`~63.7 ms`) is gone; current measured
  top-k is about `0.8-1.0 ms`, roughly a `60x+` reduction.
- Score kernel itself stayed small (`~1.0 ms`), so the remaining cacheblend
  cost is no longer score/top-k bound.

Current measured bottleneck split on the warmed measured request:

- `Convolution total_ms=777.643`
- `PicSparseAttention total_ms=618.880`
- `Raster total_ms=487.420`
- `PagedAttention total_ms=150.845`
- `PicScoreAttention total_ms=39.303`

Conclusion after the top-k fix:

- Rhino is now decisively not blocked by `cacheblend_score`.
- The main remaining work is compact dense `Convolution` plus `Raster`.
- `PicSparseAttention` still matters, but it is now clearly behind dense +
  raster combined.

Runtime image-path verification:

- Added `weight_image=%d` to `OpenCLConvBufLowMemory profile` in:
  - `source/backend/opencl/execution/buffer/ConvBufLowMemoryExecution.cpp`
- Re-ran focused profile:
  - run: `rhino_minicpm_topk_family_profile_weightimg_20260627_015001`
- Rhino full log now shows for compact sparse rows `batch=269`:
  - `use_fp_weight=0 weight_image=1 selected_pic_kernel=1 family=pic_quant`

This removes the ambiguity around Adreno image reads:

- The current Rhino compact dense path is already using image-backed weight
  storage.
- Therefore, “switch dense to image read” is not the missing optimization for
  the present hotspot.
- The next dense gains must come from kernel-family shape efficiency and raster
  reduction, not from simply enabling weight-image mode.

Updated next-step priority:

1. Keep the new cacheblend top-k family route as the default Rhino path.
2. Focus next on Adreno compact dense kernel families:
   - add richer candidates than current `generic_quant/pic_quant/fp_weight`
   - keep warmup/tune routing by `M/N/K` bucket
3. Reduce compact-row `Raster` around attention/MLP layout transitions.

## 2026-06-27 Experiment: Rhino / Adreno routing provenance + official guidance

Goal:

- Stop inferring Rhino warmup/tune routing indirectly from “non-default
  outcome”.
- Make future Adreno profile logs say explicitly whether the chosen path came
  from:
  - default heuristic
  - warm cache replay
  - online heavy/wide tune during warmup
- Record the external Adreno guidance that matters for the next kernel work, so
  we do not keep circling on already-rejected ideas such as “maybe weight image
  just was not enabled”.

Code changes:

- `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
  - added profile-only tune provenance for:
    - `score_sparse_family`
    - `cacheblend_topk_family`
    - sparse flash `schedule`
    - sparse flash `variant`
  - new log fields:
    - `source=default|cache|online_tuned`
    - `schedule_source=...`
    - `variant_source=...`
- `source/backend/opencl/execution/buffer/ConvBufLowMemoryExecution.cpp`
  - added profile-only compact dense routing provenance:
    - `decision_source=adreno_family_cache`
    - `decision_source=adreno_family_online_tuned`
    - `decision_source=legacy_binary_cache`
    - `decision_source=legacy_binary_online_tuned`
    - `decision_source=legacy_heuristic_fp_weight`
    - `decision_source=legacy_default`
  - added `execute_fallback=1` when the selected quant compact kernel cannot be
    built/executed and the runtime falls back to FP-weight.

Why this matters:

- Rhino already proved that warmup+tune cache can work:
  - current MiniCPM `ctx2560/cacheblend0.10` profile is on
    `score_flash_attention + sparse_flash_attention`, not on the old row
    fallback.
- But without explicit provenance we still had to reverse-infer whether a bad
  route was:
  - stale cache replay
  - missing warmup
  - unsupported candidate
  - default heuristic
- The new logs make that distinction direct, which is necessary before adding
  more Adreno families.

Local verification:

- Rebuilt the local x64 OpenCL check target after the logging patch:

```bash
make -C .cache/build/mnn/x64_opencl_check -j48 MNN_CL
```

- Result:
  - `ConvBufLowMemoryExecution.cpp.o` compiled
  - `PagedAttentionBufExecution.cpp.o` compiled
  - `libMNN_CL.so` linked

Adreno guidance from Qualcomm docs that is relevant to the next step:

- Qualcomm’s OpenCL optimization guide says image objects are useful when the
  workload has spatial locality and can benefit from texture/L1 caching, but it
  also says buffer accesses can be better for non-spatial patterns or when the
  extra image conversion cost outweighs cache benefit.
- The same guide recommends storing frequently reused values in local memory and
  reducing the number of shared-memory reads/writes and barriers, because local
  memory is optimized for low-latency cooperative sharing, not for gratuitous
  traffic.
- Qualcomm also recommends subgroup/broadcast/shuffle-style data exchange for
  reductions because this avoids explicit local-memory staging in many cases.
- Workgroup size should be tuned and not left to `NULL`; the best shape depends
  on register pressure, memory pattern, and occupancy.

Interpretation for current Rhino hotspots:

1. Compact dense:
   - We already verified `weight_image=1` on the hot compact-row shapes.
   - Therefore, image objects are not the missing lever by themselves.
   - The real next move is to add richer Adreno compact-dense families whose
     tradeoffs differ in tile shape / accumulation pressure / workgroup shape,
     then let heavy warmup route by `M/N/K` bucket.

2. Raster:
   - The large repeated `Mul_1_output_0_raster_0` / `Neg_output_0_raster_0`
     cost suggests layout churn around RoPE/self-attention and compact MLP
     epilogues.
   - This matches the Qualcomm guidance: the next win is likely fewer layout
     conversions or subgroup/local-memory-friendly fused epilogues, not more
     attention-family proliferation.

3. Sparse attention:
   - Current Rhino sparse flash is no longer on the catastrophic fallback path.
   - If we optimize it again, the right Adreno-specific direction is to reduce
     barrier/local-memory pressure or use subgroup-style exchange inside the
     flash reduction, not to reopen the old qsplit-vs-row debate.

Updated execution order for Rhino:

1. Re-run a focused Rhino profile with the new provenance logging to confirm:
   - compact dense hot rows are really `cache`-routed or `online_tuned`
   - sparse flash schedule/variant are replaying from cache after warmup
2. Add at least one new Adreno compact-dense family aimed at the current hot
   wide-N shapes, then let heavy warmup choose it by bucket.
3. Profile the same shape again and only then decide whether the next kernel
   dollar should go to dense or raster fusion.

## 2026-06-27 Validation: Rhino operator-level compact dense hotspot bench at rows=269

Goal:

- Stop inferring compact-dense behavior only from end-to-end PIC graph profile.
- Run the exact Rhino hot sparse-row dense shapes as standalone OpenCL ops and
  compare family candidates directly on device.

Code changes:

- Added the exact Rhino hot sparse-row case `rows=269` to:
  - `test/bench_ops/opencl/OpenCLWeightOnlyConvExprBench.cpp`
- Extended compact dense profile logging in
  `source/backend/opencl/execution/buffer/ConvBufLowMemoryExecution.cpp` to
  print:
  - `gws`
  - `lws`

Build / deploy:

```bash
cmake --build .cache/build/mnn/aidlux_adreno_opencl_test \
  --target MNN_CL opencl_weight_only_expr_bench.out -j 96
```

Remote bench bundle:

```text
/mnt/nvme/mnn_pic_opencl/bench_ops/opencl_weight_only_expr_bench_20260627
```

Remote run:

```bash
export LD_LIBRARY_PATH=$REMOTE_DIR:$REMOTE_DIR/express:$REMOTE_DIR/opencl:/usr/lib:/usr/lib/aarch64-linux-gnu
export LD_PRELOAD=/usr/lib/libOpenCL_adreno.so
export MNN_OPENCL_TUNE_LEVEL=heavy
export MNN_PAGED_ATTENTION_PROFILE=1
export MNN_BENCH_OPENCL_WEIGHT_ONLY_ROWS=269
./opencl_weight_only_expr_bench.out
```

Direct Adreno operator results:

```text
1536 -> 4608:
  generic_quant   4818-4819
  pic_quant       4815-4857
  pic_quant_c4    5752
  pic_quant_b2    6920-6921
  fp_weight       5659-5665
  selected        pic_quant
  gws/lws         68,576 / 1,64
  avg end-to-end  7.684-8.297 ms

4608 -> 1536:
  generic_quant   5291-5294
  pic_quant       5278-5299
  pic_quant_c4    6510-6530
  pic_quant_b2    7158-7160
  fp_weight       6019-6040
  selected        pic_quant
  gws/lws         68,192 / 1,128
  avg end-to-end  7.971-8.597 ms

1536 -> 2048:
  generic_quant   2287
  pic_quant       2289-2290
  pic_quant_c4    2758-2761
  pic_quant_b2    3175-3177
  fp_weight       2755-2769
  selected        generic_quant
  gws/lws         68,256 / 1,128
  avg end-to-end  4.215-4.512 ms

2048 -> 1536:
  generic_quant   2335-2340
  pic_quant       2339-2341
  pic_quant_c4    2855-2857
  pic_quant_b2    3166
  fp_weight       2675-2683
  selected        generic_quant or pic_quant by noise-level delta
  gws/lws         68,192 / 1,128
  avg end-to-end  4.130-4.695 ms
```

Important source-level fact confirmed while reading the kernel code:

- `gemm_b4_c8_int4_buf` and `pic_gemm_b4_c8_int4_buf` currently call the same
  `gemm_b4_c8_int4_buf_impl(...)` body in
  `source/backend/opencl/execution/cl/gemm_conv1x1_buf.cl`.
- So for the current Rhino hot bucket, `generic_quant` vs `pic_quant` is not a
  real implementation family difference; it is effectively:
  - the same kernel body
  - separate kernel names / tune namespaces
  - sometimes different chosen `lws`

Interpretation:

1. Current Adreno compact-dense routing is not selecting a catastrophically
   wrong family.
   - The measured winner stays on the existing `b4_c8` body.
   - `pic_quant_b2` and `pic_quant_c4` are decisively worse on the hot
     `rows=269` bucket.

2. The remaining Rhino dense problem is not “forgot to use images”.
   - All hot cases still showed `weight_image=1`.
   - Therefore, “switch to image reads” is already true on the current hotspot.

3. The current family-tune space is structurally too narrow.
   - Two of the top candidates (`generic_quant` / `pic_quant`) are the same
     kernel body.
   - The actual performance delta between them is close to noise and mainly
     reflects LWS / namespace effects.
   - This means future warmup routing needs at least one genuinely different
     Adreno dense kernel family, not just more cache keys around `b4_c8`.

Next:

1. Add one real Adreno dense family that changes the kernel body while keeping
   accumulator pressure roughly comparable to the current winner, for example a
   `b8 x c4` or similar 32-accumulator shape rather than another obvious
   “halve M” or “halve N” variant.
2. Keep operator-level bench as the first gate for that family on Rhino before
   re-running full PIC server profile.
3. If the new family still loses, shift the next main effort from dense-family
   tuning to Raster/layout fusion, because the current `b4_c8` route is already
   near the best of the implemented dense candidates.

## 2026-06-27 Adreno sparse-attention image-hybrid bench scaffold

Added a bench-first `buffer + image` path for Rhino / Adreno sparse attention
without touching production PIC routing:

- `test/bench_ops/opencl/OpenCLAttentionPerf.cpp`
  - added case filter:
    - `MNN_BENCH_OPENCL_SPARSE_FLASH_CASE=<substring>`
  - added real Rhino hot cases:
    - `MiniCPM5-1B_later_scatter_ctx2560_q269`
    - `MiniCPM5-1B_score_scatter_ctx2560_q269`
  - added bench-only image prep that converts packed sparse-flash `K/V` buffers
    into linear RGBA images
  - added image-hybrid comparison entry:
    - `mqtile_hd128_q4k8_kvimg`

- `source/backend/opencl/execution/cl/attention_buf.cl`
  - added helper readers for linear scalar-stream reads from image:
    - `sparse_flash_read_linear4_image`
    - `sparse_flash_read_linear8_image`
  - added new bench-only kernel:
    - `mqtile_sparse_flash_hd128_q4k8_kvimg`

Design choice:

- keep `query`, `sparse_query`, `output` on buffer
- move only packed `K/V` to image
- this matches the current access pattern and isolates whether Adreno texture/L1
  cache helps on the real sparse-attention hotspot before we redesign runtime
  storage

Build validation:

- regenerated OpenCL source map via `opencl_codegen.py`
- cross-build target `MNN_CL` succeeded, so the new OpenCL kernel source and
  generated `*_mnn_cl.cpp` compile cleanly
- `run_test.out` reached link after recompiling
  `test/bench_ops/opencl/OpenCLAttentionPerf.cpp`
- final `run_test.out` failure is still the old cross-test OpenCL link issue
  (missing OpenCL linkage in this test target), not a new C++ compile error

Next device-side check on Rhino:

```bash
MNN_BENCH_OPENCL_SPARSE_FLASH_CASE=MiniCPM5-1B_later_scatter_ctx2560_q269 \
./run_test.out op 1 2 0 0.0 1.0 0 0 bench_ops/opencl/perf/SparseFlash/MultiQTile
```

What to compare first:

- `current_row32`
- `current_row64`
- `mqtile_hd128_q4k16`
- `mqtile_hd128_q4k8`
- `mqtile_hd128_q4k8_kvimg`
- `mqtile_hd128_q8k16`

If `kvimg` wins on Rhino for the real hot shape, the correct production follow-up
is not per-request buffer->image conversion. It is a persistent packed `K/V`
image shadow or image-backed paged-cache source path, routed by warmup/tune on
Adreno only.

## 2026-06-27 Adreno sparse-flash compile crash narrowing

Follow-up to the image-hybrid scaffold on Rhino / Adreno:

- Built a bench-only aarch64 OpenCL test binary without `LLM` to avoid the
  current duplicate-symbol issue in the user's worktree:
  - build dir:
    - `.cache/build/mnn/aidlux_adreno_opencl_test_nosep_nollm`
- Replaced the direct `clEnqueueWriteBuffer` / `clEnqueueReadBuffer` test-harness
  path in `OpenCLAttentionPerf.cpp` with official tensor host/device copy APIs:
  - `Tensor::copyFromHostTensor`
  - `Tensor::createHostTensorFromDevice`
- That moved the Rhino crash point from host buffer upload to OpenCL program
  creation.

Then split sparse-flash bench programs to rule out “large monolithic
`attention_buf` source” as the cause:

- `sparse_flash_row_bench`
  - only `sparse_flash_attention_row32/row64`
  - no qtile kernels
  - no image helper functions
- `sparse_flash_qtile_bench`
  - only buffer qtile kernels
- `sparse_flash_qtile_kvimg_bench`
  - only the `K/V image + Q/index/output buffer` hybrid kernel

Also added bench-only impl filter:

```bash
MNN_BENCH_OPENCL_SPARSE_FLASH_IMPL=<label-substring>
```

Current Rhino result:

- even with:
  - `MNN_BENCH_OPENCL_SPARSE_FLASH_CASE=MiniCPM5-1B_later_scatter_ctx2560_q269`
  - `MNN_BENCH_OPENCL_SPARSE_FLASH_IMPL=current_row32`
- the process still segfaults inside:
  - `clCreateProgramWithSource`
- gdb stack:
  - `clCreateProgramWithSource`
  - `OpenCLRuntime::loadProgram`
  - `OpenCLRuntime::buildKernel`
  - `runSparseFlashImpl`

Meaning:

- the failure is not caused by:
  - the large production `attention_buf` source blob
  - the `K/V image` hybrid kernel specifically
  - the old direct host buffer write path in the bench harness
- the failure already reproduces on a much smaller row32-only sparse-flash
  source for the real Rhino hot shape

Practical conclusion for Adreno:

- Before discussing production routing of `buffer + image` paged-cache / sparse
  attention, we need an Adreno-compilable sparse-flash kernel form.
- Added a `sparse_flash_attention_row32_probe` bench-only kernel with:
  - the same sparse-flash row32 argument list
  - the same `gws/lws`
  - no local arrays
  - no score loop
  - no softmax
  - no image reads
  - only boundary checks plus zero write to output
- Rhino still segfaults in `clCreateProgramWithSource` even when running only:

```bash
MNN_BENCH_OPENCL_SPARSE_FLASH_IMPL=current_row32_probe
```

- So the compile-time crash is narrower than “complex sparse-flash math body”.
  It is already triggered by the current row32 sparse-attention kernel form even
  after removing the heavy body. The likely next binary-search axes are:
  1. reduce kernel argument count / signature shape
  2. remove `FLOAT8` / `vstore8`
  3. remove `NUMHEAD_GROUP_SIZE` / `kvh` path
  4. reduce global/local-id structure toward a minimal 3D kernel

## 2026-06-27 Rhino direct-op chain vs Expr chain

New narrowing on Rhino / Adreno:

- In `test/bench_ops/opencl/OpenCLAttentionPerf.cpp`, even these probes crash in
  `clCreateProgramWithSource`:
  - `inline_build_only_probe`
    - completely trivial custom source
    - zero-arg kernel body
    - no sparse buffers required by the kernel itself
  - `standard_argmax_build_only_probe`
    - standard existing MNN program `argmax_buf`
    - built through `runtime->buildKernel(...)`

gdb confirms both still die at:

- `clCreateProgramWithSource`
- from either:
  - `OpenCLRuntime::buildKernelFromSource`
  - or `OpenCLRuntime::loadProgram`

This means the problem is broader than sparse-flash source complexity:

- Rhino's current `DirectOpBenchOpenCL` chain is not a reliable place to bench
  custom OpenCL kernels.
- Continuing to debug Adreno sparse/image kernels on top of that chain will mix
  runtime instability with kernel issues.

To test whether Rhino can still compile custom OpenCL source through a more
production-like path, added a separate Expr/image probe:

- `test/bench_ops/opencl/OpenCLFuseExprProbeBench.cpp`
- target:
  - `opencl_fuse_expr_probe_bench.out`
- path:
  - `Executor/Expr`
  - `MNN_GPU_MEMORY_IMAGE`
  - `OpType_Extra`
  - OpenCL image `FuseExecution`
  - `runtime->buildKernelFromSource(...)`

Result on Rhino:

- after fixing one local source-signature bug, `opencl_fuse_expr_probe_bench.out`
  runs successfully:
  - no segfault
  - custom source compiles
  - kernel executes

Conclusion:

- Rhino / Adreno can compile and execute custom OpenCL source through the
  `Executor/Expr + image + Extra/FuseExecution` chain.
- The unstable piece is the current direct-op sparse-flash bench harness, not
  custom source support in general.

Practical next step for Adreno work:

1. Stop using `DirectOpBenchOpenCL` as the primary Rhino custom-kernel bench
   path.
2. Move Adreno-specific image/buffer custom-kernel probes to a dedicated
   Expr/image bench first.
3. Once that path is stable, start with image-friendly kernels there:
   - K/V image shadow read probes
   - image-based score-layer probes
   - later hybrid image/buffer attention microbenches
4. Keep Mali/OrangePi paths unchanged; Adreno custom probes stay in separate
   bench files and later separate runtime files if promoted.

Only after that compile blocker is removed does it make sense to compare:

- pure buffer row/qtile
- `K/V image + buffer query/output`
- possible paged-cache image shadow designs for Adreno

## 2026-06-27 Rhino Expr probe: image vs buffer on hot attention-like shapes

Follow-up on the stable Rhino / Adreno `Expr + Extra` bench path.

Bench infrastructure changes:

- Added generic custom-source buffer Extra execution in:
  - `source/backend/opencl/execution/buffer/FuseBufExecution.cpp`
- Extended:
  - `test/bench_ops/opencl/OpenCLFuseExprProbeBench.cpp`
  so the same bench can run both:
  - `MNN_GPU_MEMORY_IMAGE` via `FuseExecution`
  - `MNN_GPU_MEMORY_BUFFER` via `FuseBufExecution`
- The bench now covers:
  - build-only
  - contiguous copy
  - contiguous repeated dual-read (`r8`)
  - pseudo-random/scrambled repeated dual-read (`r8`)

Rhino results (`MNN_OPENCL_TUNE_LEVEL=heavy`, repeat=30):

- 269 x 2048
  - image build-only: `0.9592 ms`
  - image copy: `1.2335 ms`
  - image dual-read-r8: `1.9727 ms`
  - image scramble-dual-read-r8: `2.3642 ms`
  - buffer build-only: `1.1980 ms`
  - buffer copy: `1.2447 ms`
  - buffer dual-read-r8: `1.4781 ms`
  - buffer scramble-dual-read-r8: `1.6126 ms`
- 519 x 2048
  - image copy: `1.7600 ms`
  - image dual-read-r8: `2.7949 ms`
  - image scramble-dual-read-r8: `3.5417 ms`
  - buffer copy: `1.6872 ms`
  - buffer dual-read-r8: `2.1389 ms`
  - buffer scramble-dual-read-r8: `2.9678 ms`
- 269 x 1536
  - image copy: `0.8771 ms`
  - image dual-read-r8: `1.7471 ms`
  - buffer copy: `0.9079 ms`
  - buffer dual-read-r8: `1.1338 ms`

What this means for Adreno:

1. `image` is not the missing default lever for the current attention hotspot.
   - For simple contiguous copy, image and buffer are roughly tied.
   - For repeated dual-read and scrambled gather-like dual-read, buffer is
     consistently faster on Rhino in this bench.

2. We should not blindly convert the OpenCL paged-cache or sparse-attention main
   path from buffer to image on Adreno.
   - The current evidence says the canonical runtime KV store should remain
     buffer-backed.
   - Query / sparse indices / slot table / output should also stay buffer-backed.

3. If we later introduce an Adreno-specific `buffer + image` hybrid, the image
   side should be narrow and explicitly justified by a realistic gather pattern.
   - Candidate scope:
     - persistent source-K/V image shadow
     - score-layer source-slot shadow
     - offline packed-K/V image view reused across requests
   - Non-candidate scope:
     - wholesale replacement of the current paged-cache buffer path
     - default conversion of all sparse-attention K/V streaming to image

4. The next useful hybrid bench is not another contiguous image-read probe.
   - It should be a more realistic mixed-argument probe:
     - `Q / indices / slot_table / output` in buffer
     - `K/V` optionally in image shadow
   - Route that later by warmup/tune only if a real gather-heavy pattern beats
     the buffer baseline.

Current practical conclusion:

- On Rhino / Adreno, keep optimizing the main paged-attention and sparse
  attention path around buffer-native layouts, buffer-native gather/streaming,
  and raster reduction first.
- Treat image-backed K/V shadow as a secondary, shape-specific candidate, not as
  the default answer.

## 2026-06-27 Rhino raw mixed probe: buffer Q/slot/output + image-or-buffer K/V

Goal:

- Stop guessing from all-image or all-buffer microbenches.
- Measure a more realistic Adreno mixed path:
  - `query` in buffer
  - `slot_idx` in buffer
  - `output` in buffer
  - `K/V` compared as either buffer-backed or image-backed
- Sweep several `lws` candidates to see whether the access pattern itself or
  only workgroup choice explains the difference.

Bench scaffolding:

- Added:
  - `test/bench_ops/opencl/OpenCLRawHybridProbeBench.cpp`
  - target: `opencl_raw_hybrid_probe_bench.out`
- This bench uses plain OpenCL kernels on Rhino, with explicit wrapper-symbol
  init so it does not depend on the unstable direct-op sparse bench path.
- Access pattern:
  - `qRows` compact active rows
  - `kvRows` larger source-slot space
  - per thread:
    - read one `query` buffer vector
    - read one `slot_idx`
    - stream `span=8` K/V vectors from either
      - contiguous source-slot range (`span`)
      - pseudo-random slot range (`scramble`)

Measured Rhino results (`warmup=2`, `repeat=10`):

- `q=269, kv_rows=2560, c=2048, span=8`
  - contiguous span:
    - `K/V image best = 0.1642 ms` at `lws=64x4`
    - `K/V buffer best = 0.1955 ms` at `lws=64x4`
    - image wins by about `16.0%`
  - scrambled slot reads:
    - `K/V image best = 4.0935 ms`
    - `K/V buffer best = 4.0987 ms`
    - effectively tied
- `q=519, kv_rows=2560, c=2048, span=8`
  - contiguous span:
    - `K/V image best = 0.3483 ms` at `lws=8x8`
    - `K/V buffer best = 0.3726 ms` at `lws=64x4`
    - image wins by about `6.5%`
  - scrambled slot reads:
    - `K/V image best = 7.8881 ms`
    - `K/V buffer best = 7.8913 ms`
    - effectively tied

Interpretation:

1. The earlier “all-image is not better” result was incomplete for the mixed
   path question.
   - In a realistic mixed arrangement where only `K/V` move to image and
     `Q/slot/output` remain buffer-backed, Adreno *can* benefit from image
     reads on contiguous/prefix-like source-slot access.

2. The benefit disappears once the source-slot access becomes scattered.
   - For scramble-like access, image and buffer are essentially equal on Rhino.
   - Therefore, later sparse attention with highly scattered logical indices is
     not a strong candidate for default image-shadow routing.

3. If we introduce an Adreno-specific `buffer + image` runtime path, the first
   safe candidate is narrow:
   - contiguous or near-contiguous source-slot K/V access only
   - examples:
     - score-layer source-slot shadow
     - full-Q score/reference span over source slots
     - hydrate/source reuse stages with sequential source slot ranges
   - not:
     - generic later sparse attention over scattered logical rows

4. Workgroup preference is pattern-dependent even when the kernel body is the
   same.
   - `q=269` contiguous span favored `64x4`
   - `q=519` contiguous span with image K/V favored `8x8`
   - This is a good fit for Adreno-specific warmup/tune keyed by
     `qRows/kvRows/channelBlocks/access_pattern`, not a single fixed LWS.

Practical next step:

- Keep later sparse attention buffer-native on Adreno.
- If we want to exploit image cache on Rhino, do it first in an Adreno-only
  score/source-slot path where source access is sequential enough for the image
  win to appear.

## 2026-06-27 Rhino runtime integration: Adreno-only cacheblend cached-value image shadow

Implemented a narrow runtime path in `PagedAttentionBufExecution` for Adreno:

- keep reference `PagedCache` value reads on the existing buffer path
- load cached source-value segments into the existing contiguous temp buffer
- copy that contiguous temp buffer into an Adreno-only image shadow
- run a new score kernel:
  - `pic_cacheblend_value_score_cached_image`
  - reference side = buffer + slot indirection
  - cached side = linear image read

Key constraints preserved:

- no change to Mali / OrangePi production path
- no change to later sparse-attention default kernels
- no change to `PagedCache` as the runtime truth
- image route is only attempted when the cached source span is large enough and
  the image shape fits the device limits; otherwise it stays on the old buffer
  route

Build status:

- regenerated OpenCL sources with `opencl_codegen.py`
- aarch64 Adreno build target `opencl_raw_hybrid_probe_bench.out` rebuilt
  successfully, which also rebuilt `MNN_CL` and the modified
  `PagedAttentionBufExecution`

Next validation step:

- run Rhino cacheblend/epic profile again and compare the score-layer share;
  this change is expected to help the contiguous cached-source scoring slice,
  but not the scattered later sparse-attention slice.

## 2026-06-27 Rhino runtime integration: Adreno sparse-flash kvimg candidate

Integrated the existing `mqtile_hd128_q4k8_kvimg` kernel into production
`PagedAttentionBufExecution` as an Adreno-only sparse-flash candidate:

- only enabled for Adreno
- only exposed to the headDim=128 sparse-flash variant tune space
- keeps `Q`, `sparse_query`, `output` on buffer
- copies packed `K/V` scratch buffers into linear images and runs the hybrid
  image-read kernel only when that candidate is selected
- Mali / OrangePi candidate space and default path are unchanged

Implementation notes:

- added `kSparseFlashVariantMQTileHD128Q4K8KVImage`
- extended `ensureSparseFlashKernel()` to build
  `mqtile_sparse_flash_hd128_q4k8_kvimg`
- added packed `K/V` image-shadow allocation in
  `ensureAdrenoSparseFlashPackedKVImages()`
- added `image_copy_us` and
  `mqtile_hd128_q4k8_kvimg_pieces` to sparse-flash profile output

Rhino validation on `MiniCPM5-1B ctx=1536 cacheblend 0.20`:

- first profiled run after sync still showed the old default:
  - `variant=mqtile_hd128_q4k16`
  - `schedule=range_q64`
  - this was expected because sparse-flash variant tuning is disabled when
    `MNN_PAGED_ATTENTION_PROFILE=1`
- after one non-profile heavy warm, the same case improved from about
  `2.22 s` TTFT to `1.53 s`
- a follow-up profiled run then showed the warmed cache-selected route:
  - `schedule=range_q128`
  - `variant=mqtile_hd128_q8k16`
  - `variant_source=cache`
  - `schedule_source=cache`
- the new `kvimg` candidate was not selected on this hot shape:
  - `mqtile_hd128_q4k8_kvimg_pieces=0`
  - `image_copy_us=0`

Profile delta versus the earlier profiled run on the same case:

- `PicSparseAttention` total:
  - about `590-596 ms` -> `385.493 ms`
- per-layer sparse attention:
  - about `25-28 ms` -> `16-19 ms`
- end-to-end profiled graph total:
  - about `2142 ms` -> `1923 ms`

Interpretation:

1. The Adreno-only mixed buffer+image route is now safely in the tune space.
2. On this real Rhino hot shape, it is not the best current choice.
3. The large practical win came from letting tune pick the better buffer-native
   sparse-flash family and schedule (`q8k16 + range_q128`), not from forcing
   image-backed `K/V`.
4. For Rhino, the next main bottlenecks after this attention improvement remain:
   - compact dense `Convolution`
   - `Raster`
   - score layer cost is smaller and later sparse attention is no longer the
     dominant gap it was before tuning.

## 2026-06-27 Rhino mixed K/V storage split probe and Adreno key-image sparse-flash candidate

Goal:

- Stop treating `K/V image` as a single indivisible choice.
- Measure whether Adreno benefits more from:
  - `K=image, V=buffer`
  - `K=buffer, V=image`
  - `K/V=image`
  - `K/V=buffer`
- Use that result to narrow the runtime sparse-flash image candidate instead of
  forcing the broader `kvimg` path.

Bench changes:

- Extended `test/bench_ops/opencl/OpenCLRawHybridProbeBench.cpp` to cover four
  storage combinations while keeping `Q / slot_idx / output` buffer-backed:
  - `imgkey_imgvalue`
  - `imgkey_bufvalue`
  - `bufkey_imgvalue`
  - `bufkey_bufvalue`
- Fixed the standalone bench target so it cross-builds under the Linux OpenCL
  wrapper configuration:
  - compile `source/backend/opencl/core/runtime/OpenCLWrapper.cpp` directly
    into `opencl_raw_hybrid_probe_bench.out`
  - add target-level `MNN_USE_LIB_WRAPPER`
  - link `dl`

Rhino probe results (`warmup=2`, `repeat=10`, `LD_PRELOAD=/usr/lib/libOpenCL_adreno.so`):

- `q=269, kv_rows=2560, c=2048, access=span`
  - `imgkey_bufvalue = 0.1587 ms`
  - `bufkey_imgvalue = 0.1601 ms`
  - `imgkey_imgvalue = 0.1624 ms`
  - `bufkey_bufvalue = 0.1957 ms`
- `q=519, kv_rows=2560, c=2048, access=span`
  - `imgkey_bufvalue = 0.3300 ms`
  - `bufkey_imgvalue = 0.3401 ms`
  - `imgkey_imgvalue = 0.3457 ms`
  - `bufkey_bufvalue = 0.3721 ms`
- `q=269, kv_rows=2048, c=1536, access=span`
  - `imgkey_bufvalue = 0.1092 ms`
  - `bufkey_imgvalue = 0.1093 ms`
  - `imgkey_imgvalue = 0.1134 ms`
  - `bufkey_bufvalue = 0.1444 ms`

Scramble results stayed effectively tied:

- `q=269, kv_rows=2560, c=2048, access=scramble`
  - all four modes cluster at `4.0933-4.0977 ms`
- `q=519, kv_rows=2560, c=2048, access=scramble`
  - all four modes cluster at `7.8898-7.8964 ms`
- `q=269, kv_rows=2048, c=1536, access=scramble`
  - all four modes cluster at `3.0172-3.0254 ms`

Interpretation:

1. The mixed-path win on Adreno is real, but it is narrower than `kvimg`.
   - For contiguous or prefix-like source-slot access, moving only `K` to image
     while keeping `V` in buffer is the best observed split.
   - `K/V=image` is better than all-buffer, but it is not the best of the four.

2. The scatter case still does not justify image-backed later sparse attention
   as a default.
   - Once logical access becomes scattered, `K=image`, `V=image`, and buffer
     variants all collapse to the same cost band.

3. The most practical runtime follow-up is therefore:
   - keep the canonical `PagedCache` and sparse runtime path buffer-native
   - add an Adreno-only `q8k16` sparse-flash candidate with
     `K=image + V=buffer`
   - leave it in tune space only

Runtime implementation:

- Added a new sparse-flash variant:
  - `mqtile_hd128_q8k16_kimg`
- Runtime changes:
  - `PagedAttentionBufExecution` now exposes this Adreno-only candidate in the
    headDim=128 sparse-flash variant tune space
  - only packed `K` is copied to image; `V` stays on the existing buffer path
  - profile output now reports `mqtile_hd128_q8k16_kimg_pieces`
- Kernel changes:
  - new OpenCL kernel in `attention_buf.cl`:
    - `mqtile_sparse_flash_hd128_q8k16_kimg`
- Re-ran `opencl_codegen.py`
- Rebuilt successfully:
  - `opencl_raw_hybrid_probe_bench.out`
  - production `aidlux_adreno_opencl` `pic_server`

Expected next validation:

- Run Rhino `MiniCPM5-1B ctx=1536 cacheblend 0.20` warm + profile again and
  check whether tune selects `mqtile_hd128_q8k16_kimg`.
- The candidate is only expected to help contiguous score/full-Q or
  prefix-like later sparse pieces; if the hot shape remains dominated by
  scattered logical rows, buffer-native `mqtile_hd128_q8k16` should still win.

## 2026-06-27 Rhino runtime validation: mixed `K=image + V=buffer` is not the later-sparse winner

Focused runtime validation on Rhino Pi X1 / Adreno:

- model: `MiniCPM5-1B`
- context: `1536`
- mode: `cacheblend 0.20`
- frequency: `cpu-low-gpu-max`
  - `cpu=1670400/2323200/2592000,gpu=680000000,ddr=max`

Important execution note:

- `MNN_PAGED_ATTENTION_PROFILE=1` disables score-family / sparse-flash tune in
  `PagedAttentionBufExecution.cpp`.
- Therefore, a profile-first run only shows the profile-mode default route, not
  the final warmed route.
- Real validation has to be two-stage:
  1. run once with `MNN_OPENCL_TUNE_LEVEL=heavy` and no profile, so family /
     schedule / variant are written into cache
  2. run again with profile enabled and read the cached route

### Stage A: profile-first run is misleading

Run id:

```text
rhino_minicpm5_ctx1536_cb20_kimgprobe_20260627
```

Observed latency:

```text
cacheblend 0.20 = 1.893260 s
```

Profile showed:

- score layer family:
  - `score_sparse_family layer=1 family=qsplit source=default`
- score layer kernel:
  - `score_qsplit_attention ... us=73933`
- later sparse layers:
  - `schedule=range_q64 schedule_source=default`
  - `variant=mqtile_hd128_q4k16 variant_source=default`
  - `mqtile_hd128_q8k16_kimg_pieces=0`

This is not the tuned outcome. It is only the profile-mode default outcome.

### Stage B: real warm+tune run

Run id:

```text
rhino_minicpm5_ctx1536_cb20_kimgtune_20260627
```

Observed latency:

```text
cacheblend 0.20 = 1.556334 s
```

This is the meaningful TTFT number for the current code path. It is
substantially better than the profile-first default-route run.

### Stage C: profile-after-tune, reading cache

Run id:

```text
rhino_minicpm5_ctx1536_cb20_kimgcacheprof_20260627
```

Observed latency:

```text
cacheblend 0.20 = 1.669734 s
```

Expectedly a little slower than Stage B because profile detail forces extra
queue synchronization.

Cached route observed from server log:

- later sparse attention:
  - `schedule=range_q128 schedule_source=cache`
  - `variant=mqtile_hd128_q8k16 variant_source=cache`
  - `mqtile_hd128_q8k16_pieces=6`
  - `mqtile_hd128_q8k16_kimg_pieces=0`
  - typical layer flash cost: `flash_us ~= 12.7-15.6 ms`
- score layer:
  - `score_sparse_family layer=1 family=qsplit source=cache`
  - `score_qsplit_attention ... us=74845`
  - `cacheblend_score ... score_kernel_us=1253 topk_us=967`

### Interpretation

1. The new Adreno-only mixed-storage candidate is valid, but it is not the
   winner for this real later-sparse shape.
   - After real tune, later sparse moved from default `q4k16 + range_q64` to
     cached `q8k16 + range_q128`.
   - It did not choose `q8k16_kimg`.
   - This matches the synthetic probe conclusion: once later sparse access is
     dominated by scattered logical rows, image-backed `K` does not produce a
     strong enough win to beat the best buffer-native tile route.

2. The mixed-storage opportunity on Adreno is more likely in the score/full-Q
   path than in later sparse layers.
   - Later sparse already chose buffer-native `q8k16`.
   - Score layer still chose `qsplit` from cache, not flash.
   - `score_qsplit_attention` remains a concrete hotspot at roughly `75 ms`
     for this shape.
   - This path has much more contiguous / prefix-like K access than later
     sparse, which matches the synthetic `imgkey_bufvalue` advantage.

3. The practical next move is not to force `kimg` into later sparse.
   - Keep `mqtile_hd128_q8k16_kimg` in tune space only.
   - Do not make it the default route.
   - Next Adreno experiment should target score layer specifically:
     either a score-flash family that can use `K=image + V=buffer`, or a
     score-qsplit family with the same storage split.

4. There was one workflow pitfall during validation:
   - local build output and local artifact `libMNN_CL.so` were briefly out of
     sync, so Rhino was initially running an older OpenCL runtime library
   - resolved by explicitly copying
     `.cache/build/mnn/aidlux_adreno_opencl/source/backend/opencl/libMNN_CL.so`
     into
     `.cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN_CL.so`
     before rsync
   - future Rhino validation must verify runtime library hashes, not just the
     `pic_server` binary hash

## 2026-06-27 Rhino Adreno sparse-flash bench bring-up status

Goal of this branch was to stop reasoning only from end-to-end PIC timing and
directly bench exact score-layer / later-sparse flash shapes on Rhino from
`test/bench_ops/opencl`.

### What was added on host side

- `test/bench_ops/opencl/OpenCLAttentionPerf.cpp`
  - added score-shape cases:
    - `MiniCPM5-1B_score_scatter_ctx1536_q319`
    - `llama3.2-3B_score_scatter_ctx1536_q319`
    - `MiniCPM5-1B_score_scatter_ctx2560_q269`
  - added image-key / image-kv bench execution helpers
  - added optional env cache load:
    - `MNN_BENCH_OPENCL_CACHE_FILE=<path>`
  - stopped default-running probe-only impl labels; probe variants now require
    explicit `MNN_BENCH_OPENCL_SPARSE_FLASH_IMPL=*probe*`
  - sparse-flash bench no longer incorrectly inherits the old
    `supportPerfPrecision()` gate from `PagedAttention/V1V2`; low-precision
    OpenCL bench is now allowed
  - bench kernels are now requested from production program `attention_buf`
    for real row32/row64/qtile/kimg/kvimg comparisons, instead of preferring
    separate bench-only program copies
- `test/CMakeLists.txt`
  - `run_test.out` cross-build for Adreno needed two fixes:
    1. with `MNN_SEP_BUILD=ON`, hidden `MNN_CL` symbols caused unresolved
       references from `OpenCLAttentionPerf.cpp`
    2. with `MNN_USE_SYSTEM_LIB=OFF`, wrapper symbols had to be handled
       carefully to avoid missing OpenCL entry points in sep-build and duplicate
       `OpenCLWrapper.cpp` symbols in no-sep builds
  - current workable host build recipe for this bench is:
    - use the dedicated Adreno testbench build dir
    - reconfigure it with `MNN_SEP_BUILD=OFF`
    - keep the wrapper extra-source only for the `MNN_SEP_BUILD=ON` fallback
      case

### Rhino run-time result so far

Deployed:

```text
/mnt/nvme/mnn_pic_opencl/bench/run_test_nosep_20260627/run_test.out
/mnt/nvme/mnn_pic_opencl/bench/run_test_nosep_20260627/lib/libMNN.so
```

Target case:

```text
MiniCPM5-1B_score_scatter_ctx1536_q319
```

Observed blockers:

1. First crash was self-inflicted by bench probes:
   - with no impl filter, `inline_*probe` variants also ran by default
   - Rhino gdb stack:

```text
SIGSEGV at clCreateProgramWithSource
buildSparseFlashBenchKernel -> buildKernelFromSource
```

   - fixed by default-disabling probe labels unless
     `MNN_BENCH_OPENCL_SPARSE_FLASH_IMPL` explicitly asks for them

2. After removing default probes, Rhino still crashes in the real sparse-flash
   bench path before any per-impl timing is printed:

```text
RC=139
stdout log empty
```

   gdb without cache:

```text
SIGSEGV at clCreateProgramWithSource
OpenCLRuntime::loadProgram
OpenCLRuntime::buildKernel(buildProgram=attention_buf)
buildSparseFlashBenchKernel
runSparseFlashImpl
```

   Interpretation:
   - on Rhino, this standalone bench path currently crashes while trying to
     create an OpenCL program from source for `attention_buf`
   - this is not yet evidence about `row32` vs `q8k16` vs `q8k16_kimg`
     winner/loser; the bench runtime itself is not stable enough yet

3. Tried to bypass source compile via existing device cache:

```text
MNN_BENCH_OPENCL_CACHE_FILE=/mnt/nvme/mnn_pic_opencl/tmp/mnn_cachefile.bin
```

   gdb with cache:

```text
SIGSEGV at clRetainProgram
OpenCLRuntime::buildKernelWithCache
```

   Interpretation:
   - the optional cache load path is active, but this generic Rhino cache file
     is not a safe drop-in for the current standalone sparse-flash bench
   - likely causes are:
     - cached binary/build-info mismatch for this exact bench build
     - or standalone bench/runtime linkage differs enough from the warmed PIC
       artifact path that cache reuse is not safe here

### Current conclusion

The Rhino operator-bench direction is still the right direction, but the
current `run_test.out` path is blocked by Adreno OpenCL program bring-up
stability:

- source compile path: `clCreateProgramWithSource` crash
- borrowed cache path: `clRetainProgram` crash

This means we should not yet draw any performance conclusion from Rhino
operator bench about:

- `current_row32`
- `current_row64`
- `mqtile_hd128_q8k16`
- `mqtile_hd128_q8k16_kimg`
- `mqtile_hd128_q8k16_kvimg`

### Recommended next engineering move

Do not keep iterating blindly on end-to-end PIC latency while this bench path
is unstable. Next focused options are:

1. build a dedicated Adreno sparse-flash bench binary with a cleaner linkage
   model than the current `run_test.out + libMNN.so` path, or
2. reuse the already-stable PIC/OpenCL artifact runtime path to host a
   kernel-timing hook, so Rhino bench no longer needs to source-compile or
   rehydrate OpenCL programs through this fragile standalone path

Until one of those is done, Rhino score-layer `buffer + image` decisions should
still be guided by the earlier stable evidence:

- end-to-end tuned runtime picked buffer-native `q8k16` for later sparse
- score layer remains the more plausible Adreno target for mixed storage
  (`K=image + V=buffer`) than later sparse layers

## 2026-06-27 Experiment: Rhino profile/tune visibility and Adreno mixed default

Goal:

- stop blind Rhino end-to-end runs where `MNN_PAGED_ATTENTION_PROFILE=1`
  showed dense logs but never exposed which score-layer sparse family /
  schedule / variant was actually selected
- move Adreno score-layer default a step closer to `buffer + image` mixed
  storage without changing Mali/OrangePi behavior

Code/env:

- `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
  - added profile-only candidate begin/end logs for:
    - `score_sparse_family_tune`
    - `score_flash_schedule_tune`
    - `score_flash_variant_tune`
    - `score_qsplit_chunk_tune`
    - later sparse counterparts
  - added explicit `MNN_PAGED_ATTENTION_PROFILE_TUNE=1`
    - before this, `_shouldTuneScoreSparseFamily()` /
      `_shouldTuneSparseFlashSchedule()` /
      `_shouldTuneSparseFlashVariant()` /
      `_shouldTuneSparseQSplitChunk()` all hard-disabled tune whenever
      `MNN_PAGED_ATTENTION_PROFILE=1`
    - this was the main reason Rhino profile runs could never print tune
      candidate decisions even after line-buffered logging was added
  - changed Adreno score-layer default `headDim=128` sparse-flash variant:
    - prefer `mqtile_hd128_q8k16_kimg` when packed key image fits
    - otherwise prefer `mqtile_hd128_q8k16_kvimg` if full KV image fits
    - otherwise fall back to buffer-native `mqtile_hd128_q8k16`
- rebuilt and resynced `aidlux_adreno_opencl` artifact after each change

Runtime observation:

- Rhino `MiniCPM5-1B ctx=512 cacheblend 0.20` profile runs still spent tens of
  seconds before any score-layer sparse-attention trace appeared
- the only early stable log line remained the compact dense family decision:

```text
OpenCLConvBufLowMemory profile batch=512 ic=1536 oc=2048 ... family=fp_weight ...
```

Interpretation:

- `fp_weight + weight_image=1` is already the preferred compact dense route for
  at least one hot Adreno shape
- the missing sparse-attention decision was not just “log buffering”; the
  stronger issue was that profile mode had disabled tune selection entirely
- for low/mid budgets where `activeLen < 128`, the default score-layer variant
  matters more because online sparse-flash variant tuning may not trigger

Current direction:

- keep later sparse default conservative
- bias only Adreno score-layer `headDim=128` toward mixed `K=image + V=buffer`
  first, because score-layer QK is the more plausible texture-cache win and it
  avoids paying the full KV image-copy cost up front
- next Rhino step should use:

```text
MNN_PAGED_ATTENTION_PROFILE=1
MNN_PAGED_ATTENTION_PROFILE_DETAIL=1
MNN_PAGED_ATTENTION_PROFILE_TUNE=1
```

  and confirm whether score-layer sparse family actually reaches:
  - `flash`
  - `mqtile_hd128_q8k16_kimg`
  - or still falls back to buffer `q8k16` / `qsplit`

## 2026-06-27 Experiment: Adreno `headDim=128` direct-value sparse flash + raw hybrid probe

Goal:

- let Adreno `headDim=128` high-budget sparse flash actually use a mixed
  `K=image + V=buffer` path instead of forcing both K/V through packed temp
  buffers first
- verify on Rhino whether this mixed storage direction is materially better than
  pure buffer for paged sparse-attention-like access patterns

Code:

- `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
  - `_useDirectValuePrefillForSparse(...)` now accepts `OpenCLRuntime*`
  - `headDim=128` no longer hard-disables direct-value sparse prefill on
    Adreno; non-Adreno behavior stays unchanged
  - Adreno high-budget `headDim=128` default sparse-flash variant now prefers
    `mqtile_hd128_q8k16_kimg` when direct-value sparse prefill is allowed
  - image sparse-flash setup still rejects `directValuePrefill` for `KV=image`,
    but no longer rejects it for `K=image / V=buffer`

Reasoning:

- PagedCache remains buffer-backed because it is still the random-write /
  slot-table source of truth
- only the read-mostly K stream is mirrored to image
- V stays on the existing paged buffer path, avoiding another runtime KV copy

Rhino end-to-end observation:

- `MiniCPM5-1B ctx=512 cacheblend 0.50` still spent its visible front half in
  compact dense tuning before first sparse-attention begin log
- request log advanced through:
  - `forward_raw_begin`
  - `OpenCLConvBufLowMemory family-candidate batch=512 ic=1536 oc=2048 ...`
  - `OpenCLConvBufLowMemory family-candidate batch=512 ic=2048 oc=1536 ...`
- so the current Rhino end-to-end bottleneck is still the dense chain before
  first score/later sparse attention launch; the new sparse-flash mixed route
  did not surface as the first blocking issue

Direct Rhino probe (`opencl_raw_hybrid_probe_bench.out`) results:

- shape `q=519 kv=2560 c=2048 span=8`
  - `buf+buf`: `0.3702 ms`
  - `K=image + V=buffer`: `0.3308 ms`
  - `K=buffer + V=image`: `0.3292 ms`
  - `image+image`: `0.3322 ms`
  - takeaway: for contiguous / span-style access, image-assisted paths beat
    pure buffer by about `10%-12%`; `K=image + V=buffer` is already in the
    winning band
- same shape, scramble access
  - all four variants converge around `7.89 ms`
  - takeaway: when slot access is highly scrambled, texture-cache help mostly
    disappears
- shape `q=269 kv=2048 c=1536 span=8`
  - `buf+buf`: `0.1438 ms`
  - `K=image + V=buffer`: `0.1105 ms`
  - `K=buffer + V=image`: `0.1104 ms`
  - `image+image`: `0.1134 ms`
  - takeaway: on a smaller, more sparse-like span shape, mixed
    `K=image + V=buffer` is about `23%` faster than pure buffer
- same shape, scramble access
  - all four variants converge around `3.02 ms`
  - takeaway: the win again depends on contiguous/tile-friendly reads, not
    random slot patterns

Current conclusion:

- Adreno should keep the production PagedCache as buffer-backed storage
- the useful hybrid optimization surface is the read-mostly sparse-flash working
  set, especially K/QK-dominant span-friendly access
- `K=image + V=buffer` is the right mixed default to keep pushing on Adreno:
  it captures the texture-cache win on K without forcing an extra V image path
  when access is already reasonably contiguous
- if Rhino cacheblend/epic still underperform end to end after this change, the
  next bottleneck to attack remains:
  1. dense compact family/tune before first attention
  2. then only the span-friendly sparse-flash pieces

2026-06-27 Adreno mixed-path routing tightening
-----------------------------------------------

Code change:

- `PagedAttentionBufExecution.cpp` now treats Adreno `KV=image` sparse-flash as
  a fallback-only route.
- Production/tune candidate selection for Adreno `headDim=128` score-layer and
  later sparse attention now prefers `K=image + V=buffer` first, and only keeps
  `KV=image` if key-only image materialization is not available.

Why:

- Rhino raw probe still shows the same pattern after the latest sync:
  - `q=519 kv=2560 c=2048 span=8`
    - `buf+buf`: `0.3784 ms`
    - `K=image + V=buffer`: `0.3294 ms`
    - `image+image`: `0.3372 ms`
  - `q=269 kv=2048 c=1536 span=8`
    - `buf+buf`: `0.1473 ms`
    - `K=image + V=buffer`: `0.1228 ms`
    - `image+image`: `0.1257 ms`
- On scramble access, all four layouts stay effectively flat:
  - `q=519 kv=2560 c=2048`: about `7.89-7.93 ms`
  - `q=269 kv=2048 c=1536`: about `3.03-3.04 ms`

Conclusion:

- Adreno still benefits from image-assisted reads only when sparse-flash access
  has enough locality/span.
- The stable production direction is:
  - pagedcache stays buffer-backed
  - sparse-flash mirrors packed K into image when possible
  - V stays buffer-backed unless a bench/diagnostic path explicitly forces
    `KV=image`
- This keeps the texture-cache/L1 benefit on QK without paying an unnecessary V
  image copy in the common span-friendly path.
- Implementation detail:
  - old tuned sparse-flash cache entries that still point at `KV=image` are now
    rejected on Adreno when `K=image` is available, so the device can retune
    onto the new mixed-path candidate set instead of getting stuck on a stale
    selection.

2026-06-27 Rhino release-path validation and helper split
---------------------------------------------------------

Release-path fix:

- `aidlux_adreno_opencl` release `pic_server` initially failed to link in
  `MNN_SEP_BUILD=ON` mode with:
  - `libMNN_CL.so: undefined reference to clGetKernelInfo`
- Root cause: `OpenCLWrapper` already wrapped many `cl*` entry points, but
  `clGetKernelInfo` was missing from the wrapper declarations/loader/forwarder.
- Fix:
  - added `clGetKernelInfo` pointer declaration and load path in
    `source/backend/opencl/core/runtime/OpenCLWrapper.hpp/.cpp`
  - rebuilt `aidlux_adreno_opencl`; `pic_server` now links successfully again

Maintainability split:

- moved Adreno-specific paged-attention routing heuristics into:
  - `source/backend/opencl/execution/buffer/PagedAttentionAdrenoUtils.hpp`
  - `source/backend/opencl/execution/buffer/PagedAttentionAdrenoUtils.cpp`
- `PagedAttentionBufExecution.cpp` now keeps only thin local wrappers and the
  variant-selection logic.
- This isolates the Adreno image/buffer policy from the shared Mali/OpenCL
  scheduling code and makes the next Rhino-specific routing changes cheaper.

Rhino minimal release-path profile:

- ran a targeted formal-path debug request:
  - device: `rhino`
  - model: `MiniCPM5-1B`
  - context: `512`
  - mode: `cacheblend`
  - ratio: `0.50`
  - frequency profile: `cpu-high-gpu-max`
  - server env:
    - `MNN_OPENCL_TUNE_LEVEL=heavy`
    - `MNN_PAGED_ATTENTION_PROFILE=1`
    - `MNN_PAGED_ATTENTION_PROFILE_DETAIL=1`
    - `MNN_PAGED_ATTENTION_PROFILE_TUNE=1`
    - `MNN_PIC_REQUEST_PROFILE=1`
    - `MNN_PIC_REQUEST_PROFILE_SYNC=1`
- request reached:
  - `prepare_pic_cache`
  - `graph_cacheblend_begin_scoring`
  - `prefill_embedding`
  - `forward_raw_begin`
- it still did **not** reach any:
  - `score_flash_attention`
  - `sparse_flash_attention`
  - `OpenCLPagedAttention profile op=... phase=begin`
- the log stopped at the first dense compact family tune:
  - `OpenCLConvBufLowMemory family-candidate batch=512 ic=1536 oc=2048 ...`
  - selected family:
    - `family=fp_weight`
    - `decision_source=adreno_family_online_tuned`
    - `gws=64,512 lws=4,8`

Current conclusion:

- the new Adreno sparse-flash mixed path is now correctly wired into release
  artifacts, but it is still not the first end-to-end blocker on Rhino
- the immediate end-to-end hotspot remains the pre-attention compact dense chain
  and/or its family-tune execution latency
- next Rhino optimization priority should stay:
  1. compact dense Adreno family/tile/workgroup path before first attention
  2. then re-check whether the request finally enters `score_flash_attention`
     and later sparse-flash with the new mixed-path routing

2026-06-27 Adreno mixed sparse-flash: direct key-image pack
-----------------------------------------------------------

Code change:

- added `pack_paged_k_prefill_to_image` in
  `source/backend/opencl/execution/cl/paged_attention_buf.cl`
- `PagedAttentionBufExecution.cpp` now uses that kernel for the Adreno
  production mixed sparse-flash route:
  - `directValuePrefill=1`
  - sparse variant = `mqtile_hd128_q8k16_kimg`

What changed in the data path:

- old mixed route:
  1. pack paged key -> `tempK` buffer
  2. copy `tempK` buffer -> key image
  3. sparse flash reads `K=image`, `V=buffer`
- new mixed route:
  1. pack paged key -> key image directly
  2. sparse flash reads `K=image`, `V=buffer`

Why:

- on Adreno the value stream is already intentionally left on the paged buffer
  path for the production mixed route
- after that decision, `tempK -> image` became pure extra traffic on the hot
  `K=image + V=buffer` path
- direct pack-to-image keeps the same packed layout expected by the flash kernel
  while removing one intermediate buffer write + one image-copy pass

Validation:

- regenerated OpenCL sources with `opencl_codegen.py`
- rebuilt `aidlux_adreno_opencl` release `pic_server`
- synced new artifact to Rhino successfully

Focused Rhino run:

- device: `rhino`
- model: `MiniCPM5-1B`
- context: `256`
- mode: `cacheblend`
- ratio: `0.50`
- env:
  - `MNN_OPENCL_TUNE_LEVEL=heavy`
  - `MNN_PAGED_ATTENTION_PROFILE=1`
  - `MNN_PAGED_ATTENTION_PROFILE_DETAIL=1`
  - `MNN_PAGED_ATTENTION_PROFILE_TUNE=1`
  - `MNN_PIC_REQUEST_PROFILE=1`
  - `MNN_PIC_REQUEST_PROFILE_SYNC=1`

Observed log:

- request still did not advance to `score_flash_attention` / `sparse_flash_attention`
- it again spent its visible front half in the compact dense chain before first
  attention
- but the first hot dense shape changed to a better family choice:
  - `batch=242 ic=1536 oc=2048`
  - candidates:
    - `generic_quant`: `2106`
    - `pic_quant`: `2098`
    - `pic_quant_wg64`: `2140`
    - `pic_quant_wg128`: `2097`
    - `fp_weight`: `2154`
  - selected:
    - `family=pic_quant_wg128`
    - `decision_source=adreno_family_online_tuned`

Current takeaway:

- the mixed sparse-flash route is now cheaper in principle on Adreno because the
  common `K=image + V=buffer` path no longer pays `tempK -> image`
- end-to-end Rhino is still blocked earlier by compact dense, so the next real
  performance work remains:
  1. keep improving Adreno compact dense family routing / kernels
  2. then re-profile sparse attention once requests reliably reach the score
     layer

2026-06-27 Adreno mixed sparse-flash: q4k8_kimg candidate + tune override fix
------------------------------------------------------------------------------

Code change:

- added a new sparse-flash variant:
  - `mqtile_hd128_q4k8_kimg`
  - kernel: `mqtile_sparse_flash_hd128_q4k8_kimg`
- wired it into `PagedAttentionBufExecution` variant enum / kernel build /
  candidate list / profile counters
- fixed sparse-flash variant override selection so image-backed candidates are
  actually measurable during tune:
  - `_defaultSparseFlashVariant(...)` now checks support with
    `runtime + batch + kvHeads + kvLen`
  - old code only checked `headDim`, so `kimg/kvimg` override candidates could
    silently fail to override the default route
- extended `OpenCLAttentionPerf.cpp` bench wiring so `mqtile_hd128_q4k8_kimg`
  can be invoked from bench

Why this matters:

- before this change, Adreno had:
  - `q4k8` buffer/buffer
  - `q4k8_kvimg` key-image/value-image
  - `q8k16_kimg` key-image/value-buffer
- it did *not* have the lighter `q4k8 + K=image + V=buffer` option
- that left tune with an incomplete choice set for headDim=128 sparse layers:
  either cheaper arithmetic with pure buffers, or mixed image/buffer only on
  the heavier `q8k16` tile

Measured effect (Rhino raw hybrid probe, device-local OpenCL microbench):

- bench binary:
  - `opencl_raw_hybrid_probe_bench.out`
- note:
  - `run_test.out bench_ops/opencl/perf/SparseFlash/MultiQTile` is still
    unstable on Rhino and currently segfaults even for single-impl runs, so the
    stable evidence for this round is the raw KV-access probe

Best LWS results:

- `q=269 kv=2560 c=2048 span=8`:
  - `bufbuf`: `0.1938 ms`
  - `imgkey_bufvalue`: `0.1575 ms`
  - `imgimg`: `0.1619 ms`
  - `bufkey_imgvalue`: `0.1588 ms`
- `q=519 kv=2560 c=2048 span=8`:
  - `bufbuf`: `0.3704 ms`
  - `imgkey_bufvalue`: `0.3258 ms`
  - `imgimg`: `0.3316 ms`
  - `bufkey_imgvalue`: `0.3258 ms`
- `q=269 kv=2048 c=1536 span=8`:
  - `bufbuf`: `0.1441 ms`
  - `imgkey_bufvalue`: `0.1081 ms`
  - `imgimg`: `0.1122 ms`
  - `bufkey_imgvalue`: `0.1131 ms`

Interpretation:

- on Rhino/Adreno, for the cache-friendly span-like access pattern that sparse
  flash wants, `K=image + V=buffer` is consistently better than pure
  `buffer+buffer`
- the gain is material:
  - about `1.23x` at `269x2560x2048`
  - about `1.14x` at `519x2560x2048`
  - about `1.33x` at `269x2048x1536`
- `K=image + V=buffer` is also slightly better than `K=image + V=image` on
  these spans, which matches the current Adreno production policy

Current takeaway:

- the new `q4k8_kimg` candidate is worth keeping because the raw probe confirms
  the underlying Adreno memory choice is good
- the next missing piece is still stable sparse-flash direct-op measurement on
  Rhino:
  - `SparseFlash/MultiQTile` bench itself is unstable
  - once that bench is stable, we can quantify whether `q4k8_kimg` beats
    `q4k8` / `q4k8_kvimg` / `q8k16_kimg` for the real sparse kernel

## 2026-06-27 Rhino compact-dense direction and runtime-cache fix

What was fixed:

- `transformers/pic_llm/engine/app/pic_server.cpp`
  now stops using relative `tmp/` as the OpenCL runtime cache root.
- `tmp_path` is redirected to:
  - `<kv_cache_dir>/runtime_cache`
- the directory is created before `llm->set_config(...)`.

Why this matters:

- Rhino real-request profiling had already shown:
  - server logs printing `Can't open file:tmp/mnn_cachefile.bin`
  - first profiled requests spending time in
    `OpenCLConvBufLowMemory ... decision_source=adreno_family_online_tuned`
    before even reaching score/sparse attention attribution
- with the old relative `tmp/`, warm+tune results were not reliably persisted
  into a stable per-model cache directory, so restart-based profile runs could
  re-enter online tuning noise

Current compact-dense conclusion from stable `run_test.out` evidence:

- `run_test.out bench_ops/opencl/perf/WeightOnlyConv` remains the preferred
  single synthetic entrance for compact-dense work; standalone extra probe bins
  stay disabled by default
- on Rhino hot compact shape:
  - `rows=519 ic=1536 oc=4608`
  - forcing `storage=buffer` vs `storage=image`
  - result was effectively identical (`~17.37 ms`)
- on a relevant full-row shape:
  - `rows=1031 ic=1536 oc=2048`
  - `generic_quant` / `pic_quant` / `fp_weight`
  - all landed around `~24.7-24.9 ms`

Interpretation:

- the missing Rhino `cacheblend` / `epic` gain is not mainly explained by
  weight storage buffer-vs-image
- it is also not mainly explained by the final compact-dense family choice
  among the currently implemented kernels
- the higher-ROI next targets are therefore:
  - make warm/tune cache reuse deterministic across server restarts
  - then profile real requests again and attribute remaining cost between:
    - `Convolution`
    - `Raster`
    - `PicScoreAttention`
    - `PicSparseAttention`
- for compact dense specifically, the next worthwhile Adreno experiment is not
  another weight-image rewrite; it is graph-side layout / activation movement,
  and if image is revisited, prioritize activation-read / layout-path changes
  over weight storage alone

## 2026-06-27 Rhino unified run_test.out compact-dense findings

What changed in the bench surface:

- `test/bench_ops/opencl/OpenCLWeightOnlyConvExprBench.cpp`
  was converted from a standalone `main()` binary into a registered
  `MNNTestCase`, so `bench_ops/opencl/perf/WeightOnlyConvExpr` now runs through
  the same `run_test.out` entrance as `WeightOnlyConv`
- `test/CMakeLists.txt`
  no longer strips that file out of `run_test.out`
- `test/bench_ops/opencl/OpenCLAttentionPerf.cpp`
  now includes `rows=269` and `rows=1287` direct-op cases, matching the expr
  bench rows more closely

Why this matters:

- the user requirement for Rhino profiling was to stop proliferating
  measurement entrances
- with both direct-op and expr-level compact dense tests inside `run_test.out`,
  we can compare:
  - bare low-memory weight-only Conv kernel cost
  - graph-level cost including extra layout / raster / module overhead

Rhino measurements (Adreno, `LD_PRELOAD=libOpenCL_adreno.so`,
`MNN_OPENCL_TUNE_LEVEL=none`, forced `storage=buffer`):

- `minicpm_hidden_to_ffn`, `rows=519`, `1536 -> 4608`, `pic_quant`
  - direct `WeightOnlyConv`: `17.3356 ms`
  - expr   `WeightOnlyConvExpr`: `21.3876 ms`
  - extra graph/layout overhead: `~4.05 ms` (`~23%`)
- `minicpm_hidden_to_attn`, `rows=519`, `1536 -> 2048`, `pic_quant`
  - direct: `7.8962 ms`
  - expr:   `10.1185 ms`
  - extra graph/layout overhead: `~2.22 ms` (`~28%`)
- `minicpm_hidden_to_ffn`, `rows=317`, `1536 -> 4608`, `pic_quant`
  - direct: `16.0273 ms`
  - expr:   `18.3214 ms`
  - extra graph/layout overhead: `~2.29 ms` (`~14%`)

Interpretation:

- compact dense graph/layout overhead on Rhino is real and material; it is not
  just a tiny tail after GEMM
- for the shapes above, removing or shrinking layout/raster movement can still
  recover a meaningful part of end-to-end sparse-request TTFT

More important anomaly:

- low-M compact dense on Rhino is badly non-monotonic
- `minicpm_hidden_to_attn`, `1536 -> 2048`, `pic_quant`, direct path:
  - `rows=317`: `7.6672 ms`
  - `rows=269`: `15.1299 ms`
- `minicpm_hidden_to_ffn`, `1536 -> 4608`, `pic_quant`, direct path:
  - `rows=317`: `16.0273 ms`
  - `rows=269`: `32.8545 ms`

What was ruled out for `rows=269`:

- changing dense family did not fix it:
  - `generic_quant`, `pic_quant`, `fp_weight` for
    `269 x 1536 -> 4608` all stayed around `~33 ms`
- Adreno-specific compact family variants did not fix it for
  `269 x 1536 -> 2048`:
  - `pic_quant_b2`: `15.0361 ms`
  - `pic_quant_wg64`: `15.1475 ms`
  - `pic_quant_wg128`: `15.1213 ms`
  - `pic_quant_c4`: `15.0827 ms`
- forcing weight storage image also did not help:
  - `269 x 1536 -> 4608`, `pic_quant`, `storage=image`: `33.0221 ms`
- enabling `MNN_OPENCL_TUNE_LEVEL=heavy` also did not materially improve
  `269 x 1536 -> 2048`:
  - direct remained about `15.2750 ms`

Current conclusion:

- Rhino compact dense has a true low-M kernel-path anomaly around the
  `rows ~= 269` band
- this is not primarily a `buffer vs image` issue
- it is not primarily a `generic vs pic vs fp_weight` routing issue
- it is not fixed by current Adreno family variants or by ordinary heavy LWS
  tune
- therefore the next Adreno-only dense optimization should target:
  - the low-M kernel path itself, or
  - graph/layout movement around it
  rather than adding more weight-storage or family-selection heuristics

## 2026-06-27 Experiment: Rhino `adreno_batch_gemv` family via `run_test.out`

Goal:

- keep compact-dense attribution on one entrance: `run_test.out`
- test a genuinely different Adreno compact-dense family rather than another
  minor direct-kernel variant
- check whether image reads or padded low-row launch tricks explain the bad
  `rows=216/269` band

Code:

- `ConvBufLowMemoryExecution.cpp`
  - new compact-dense family candidate: `adreno_batch_gemv`
  - family tune key bumped to `convBufLowMemory_adreno_family_exact_v4_*`
- `OpenCLAttentionPerf.cpp`
  - direct bench understands the new family name
- a padded-launch experiment for low-row `adreno_batch_gemv` was tried and then
  removed because it did not improve the bad band

Rhino measurements, same single entrance (`run_test.out`):

- forced `adreno_batch_gemv`, `storage=image`, `tune=none`
  - `minicpm_hidden_to_attn`
    - `rows=269`: direct `15.1188 ms`, expr `16.7337 ms`
    - `rows=317`: direct `7.7075 ms`, expr `9.0667 ms`
    - `rows=519`: direct `7.9708 ms`, expr `10.0986 ms`
  - `minicpm_hidden_to_ffn`
    - `rows=269`: direct `32.9743 ms`, expr `35.2583 ms`
    - `rows=317`: direct `16.1040 ms`, expr `18.5907 ms`
    - `rows=519`: direct `17.5257 ms`, expr `21.7701 ms`

- forced `adreno_batch_gemv`, `storage=image`, `tune=heavy`
  - `minicpm_hidden_to_attn`
    - `rows=216`: direct `11.5534 ms`, expr `13.4634 ms`
    - `rows=269`: direct `14.8752 ms`, expr `16.2679 ms`
    - `rows=317`: direct `4.1141 ms`, expr `6.0027 ms`
    - `rows=418`: direct `5.3931 ms`, expr `7.4916 ms`
    - `rows=519`: direct `6.5121 ms`, expr `8.7654 ms`
  - `minicpm_hidden_to_ffn`
    - `rows=519`: direct `14.3395 ms`, expr `18.1788 ms`

- forced `adreno_batch_gemv`, `rows=269`, `minicpm_hidden_to_attn`
  - `storage=image`: `15.0852 ms`
  - `storage=buffer`: `15.0585 ms`

Conclusions:

- the single-entry measurement chain is good enough for compact-dense direct
  vs expr attribution on Rhino
- `adreno_batch_gemv` is a real useful family for the higher compact-row band:
  - `rows=317`: direct `7.71 -> 4.11 ms`
  - `rows=519`: direct `7.97 -> 6.51 ms`
  - `rows=519`, FFN: direct `17.53 -> 14.34 ms`
- image weight reads are not the main lever on the bad shape; `rows=269`
  stayed essentially identical on image vs buffer
- the bad Rhino compact-dense band is still specifically `rows=216/269`
- a simple padded launch to a larger row bucket did not fix it, so the problem
  is not just a trivial workgroup-count bucket issue

Next:

- keep `adreno_batch_gemv` in the family tune candidate set
- use heavy warm/cache reuse to pick it for the higher compact-row band that
  maps to long-context high-budget PIC requests
- treat `rows=216/269` as a separate unresolved Adreno compact-dense problem
  that likely needs either:
  - a different low-M compute path, or
  - less graph/layout movement around that band

## 2026-06-27 Experiment: Rhino single-entry cleanup and compact-dense `rows=269` recheck

What changed:

- `run_test.out` remains the only operator-measurement entrance.
- `OpenCLRawHybridProbeBench.cpp` no longer touches direct raw CL on Adreno:
  it now prints a skip message instead of crashing the wrapped Rhino runtime.
  The direct raw path was crashing inside wrapper-facing `clGetPlatformIDs`,
  then `clFinish`, then `clCreateBuffer`; keeping it enabled on Adreno was not
  a productive profiling path.
- compact-dense tuning namespace bumped again:
  - `convBufLowMemory_adreno_family_exact_v5_*`
- added two Adreno trial variants for compact dense:
  - `pic_quant_wg4x32`
  - `pic_quant_wg8x16`

Stable `run_test.out` result:

- Rhino now handles:
  - `bench_ops/opencl/perf/RawHybridProbe`
  without tearing down the SSH session; on Adreno it prints an explicit skip.

Rhino compact-dense measurements, `storage=buffer`, `tune=heavy`:

- `minicpm_hidden_to_attn`, `1536 -> 2048`
  - `rows=216`
    - `pic_quant`: `11.9578 ms`
    - `fp_weight`: `11.8698 ms`
    - `adreno_batch_gemv`: `16.7295 ms`
  - `rows=269`
    - `pic_quant`: `15.1420 ms`
    - `fp_weight`: `16.1940 ms`
    - `adreno_batch_gemv`: `15.2105 ms`
  - `rows=317`
    - `pic_quant`: `4.1097 ms`
    - `fp_weight`: `4.1605 ms`
    - `adreno_batch_gemv`: `4.1120 ms`
  - `rows=418`
    - `pic_quant`: `5.4645 ms`
    - `fp_weight`: `6.8542 ms`
    - `adreno_batch_gemv`: `5.5097 ms`
  - `rows=519`
    - `pic_quant`: `6.6312 ms`
    - `fp_weight`: `6.6147 ms`
    - `adreno_batch_gemv`: `12.9628 ms`

- `minicpm_hidden_to_ffn`, `1536 -> 4608`
  - `rows=216`
    - `pic_quant`: `27.1145 ms`
    - `fp_weight`: `25.8458 ms`
    - `adreno_batch_gemv`: `27.3005 ms`
  - `rows=269`
    - `pic_quant`: `34.8800 ms`
    - `fp_weight`: `35.4195 ms`
    - `adreno_batch_gemv`: `35.4230 ms`
  - `rows=317`
    - `pic_quant`: `8.4040 ms`
    - `fp_weight`: `8.5125 ms`
    - `adreno_batch_gemv`: `9.2753 ms`
  - `rows=418`
    - `pic_quant`: `11.4712 ms`
    - `fp_weight`: `11.8167 ms`
    - `adreno_batch_gemv`: `11.9440 ms`
  - `rows=519`
    - `pic_quant`: `14.4830 ms`
    - `fp_weight`: `14.7280 ms`
    - `adreno_batch_gemv`: `14.4873 ms`

Weight-storage check, `rows=269`, `minicpm_hidden_to_attn`:

- `generic_quant`
  - `storage=image`: `32.5895 ms`
  - `storage=buffer`: `15.2627 ms`
- `adreno_batch_gemv`
  - `storage=image`: `15.5855 ms`
  - `storage=buffer`: `15.5498 ms`
- `fp_weight`
  - `storage=image`: `15.4190 ms`
  - `storage=buffer`: `15.2362 ms`

New fixed-workgroup trials at `rows=269`, `storage=buffer`:

- `minicpm_hidden_to_attn`, `1536 -> 2048`
  - `pic_quant_wg4x32`: `14.9260 ms`
  - `pic_quant_wg8x16`: `15.2748 ms`
- `minicpm_hidden_to_ffn`, `1536 -> 4608`
  - `pic_quant_wg4x32`: `33.6154 ms`
  - `pic_quant_wg8x16`: `34.5966 ms`
- `rows=317`, `minicpm_hidden_to_attn`
  - `pic_quant_wg4x32`: `4.4104 ms`
  - `pic_quant_wg8x16`: `4.5044 ms`

Conclusion:

- the `rows ~= 269` anomaly is real and survives:
  - family changes
  - image vs buffer changes
  - simple fixed-workgroup changes
- `image` is not the compact-dense lever on Rhino:
  - in the one clear case where it matters, it makes `generic_quant` much
    worse
  - for the faster families it is flat to slightly worse than buffer
- the tiny `wg4x32` gain (`15.14 -> 14.93 ms`) is not enough to explain the
  `269 -> 317` cliff
- therefore the next meaningful Adreno dense work should move away from
  family/LWS micro-routing and toward a different compact-row dataflow:
  - buffer-native compact hidden layout reused across several linears, or
  - a new low-/mid-M kernel path that changes the shared `k * bhw4` strided
    access pattern rather than only workgroup shape

## 2026-06-27 Change: Adreno compact-dense weight storage default

Code:

- added `ConvAdreno::preferCompactDenseWeightBuffer(...)`
- `ConvBufLowMemoryExecution::set1x1WeightLowMemory()` now keeps Mali behavior
  unchanged but, on Adreno, defaults large int4 1x1 linear weights to buffer
  storage before any bench override is applied

Reason:

- compact dense on Rhino does not show a consistent image-weight win
- for the large PIC-style linear projections, buffer is flat to slightly better
  and avoids the very slow image-only generic path

Verification on Rhino (`run_test.out`, `MNN_OPENCL_TUNE_LEVEL=heavy`):

- forced `pic_quant`, `1536 -> 2048`
  - `rows=269`
    - image: `15.1838 ms`
    - buffer: `15.1737 ms`
  - `rows=317`
    - image: `4.1064 ms`
    - buffer: `3.9609 ms`
  - `rows=519`
    - image: `6.6706 ms`
    - buffer: `6.6232 ms`

- forced `fp_weight`, `1536 -> 2048`
  - `rows=269`
    - image: `15.3690 ms`
    - buffer: `14.9730 ms`
  - `rows=317`
    - image: `4.1658 ms`
    - buffer: `4.1525 ms`
  - `rows=519`
    - image: `7.2298 ms`
    - buffer: `6.6405 ms`

Interpretation:

- on current Rhino data, Adreno compact dense does not justify an image-first
  default
- attention/hydrate/sparse-flash can still keep their own Adreno image logic;
  this change only narrows the low-memory compact-dense weight-storage default

## 2026-06-27 Change: stabilize Rhino `run_test.out` and narrow the next dense target

What changed:

- `bench_ops/opencl/perf/WeightOnlyConv` no longer accidentally runs
  `WeightOnlyConvExpr` through prefix matching:
  - `WeightOnlyConvExpr` moved to `bench_ops/opencl/dev/WeightOnlyConvExpr`
- `WeightOnlyConv` bench now reports the actual default storage decision on
  Adreno (`auto_buffer` after the new compact-dense storage policy), instead of
  the stale static `auto_image` heuristic
- `bench_ops/opencl/perf/SparseFlash/MultiQTile` now explicitly skips on
  Adreno inside `run_test.out`
  - reason: Rhino still crashes in `clCreateProgramWithSource` for this bench,
    even on tiny build-only probes
  - this keeps the single-entry bench runner stable instead of killing SSH

Verification on Rhino:

- `run_test.out bench_ops/opencl/perf/WeightOnlyConv ...`
  - only runs `WeightOnlyConv`
  - `rows=269, 1536 -> 2048` now reports `storage=auto_buffer`
- `run_test.out bench_ops/opencl/perf/SparseFlash/MultiQTile ...`
  - prints a skip line on Adreno
  - exits cleanly instead of crashing

Dense-cliff conclusion tightened with the now-clean single-entry runner:

- `minicpm_hidden_to_attn`, `1536 -> 2048`
  - `rows=269`
    - direct `WeightOnlyConv`: `14.61 ms`
    - expr/module path: `16.80-16.95 ms`
  - `rows=317`
    - direct `WeightOnlyConv`: `3.86-3.87 ms`
    - expr/module path: `5.95-6.06 ms`
  - `rows=519`
    - direct `WeightOnlyConv`: `6.24 ms`
    - expr/module path: `8.78 ms`

- `minicpm_hidden_to_ffn`, `1536 -> 4608`
  - `rows=269`
    - direct `WeightOnlyConv`: `32.61 ms`
    - expr/module path: `35.13 ms`
  - `rows=317`
    - direct `WeightOnlyConv`: `7.86 ms`
    - expr/module path: `10.68 ms`

Interpretation:

- the `rows ~= 269` cliff is dominated by the underlying dense kernel/dataflow,
  not by a graph/raster-only tax
- the module/layout overhead is roughly a small near-constant add-on
  (`~2.1-2.8 ms`) across these rows, while the direct kernel cost is what jumps
- therefore the next compact-dense optimization should focus on the low-/mid-M
  compute path itself:
  - change the compact-row dataflow / memory access pattern
  - only then come back to raster/layout fusion

## 2026-06-27 Rhino compact-dense storage correction: Adreno low-memory path wants image, not buffer

What changed:

- rebuilt the Rhino single-entry testbench so `run_test.out
  bench_ops/opencl/perf/WeightOnlyConv` actually compiles with
  `MNN_LOW_MEMORY=ON`
- verified creator selection on Rhino:
  - `creator selected=ConvBufLowMemoryExecution`
- fixed Adreno kernel source legality for the experimental input-cache family:
  - moved the `__local` scratch array from the inline helper into the actual
    `__kernel` entry in `gemm_conv1x1_buf.cl`
- removed the Adreno-only `preferCompactDenseWeightBuffer(...)` override in
  `ConvBufAdrenoUtils.cpp`
  - this lets low-memory INT4 1x1 compact dense return to the normal image
    default instead of forcibly choosing buffer storage

Why:

- the previous "Adreno compact dense prefers buffer" rule was derived from a
  different generic path
- once the real low-memory weight-only 1x1 path was benchable on Rhino, the
  result was the opposite: image-backed weights were consistently faster across
  the MiniCPM compact dense hot shapes

Single-entry Rhino evidence (`run_test.out`, heavy tune, 10 warmup / 40 repeat):

- `minicpm_hidden_to_attn`, `1536 -> 2048`
  - `rows=269`
    - old `auto_buffer`: `3.2938 ms`
    - new `auto_image`: `2.6291 ms`
    - `1.253x` faster
  - `rows=519`
    - old `auto_buffer`: `5.8204 ms`
    - new `auto_image`: `4.7158 ms`
    - `1.234x` faster

- `minicpm_hidden_to_ffn`, `1536 -> 4608`
  - `rows=269`
    - old `auto_buffer`: `6.6921 ms`
    - new `auto_image`: `5.5250 ms`
    - `1.211x` faster
  - `rows=519`
    - old `auto_buffer`: `12.3144 ms`
    - new `auto_image`: `10.3288 ms`
    - `1.192x` faster

- `minicpm_ffn_to_hidden`, `4608 -> 1536`
  - `rows=269`
    - old `auto_buffer`: `8.0746 ms`
    - new `auto_image`: `6.0898 ms`
    - `1.326x` faster
  - `rows=519`
    - old `auto_buffer`: `14.4006 ms`
    - new `auto_image`: `11.1295 ms`
    - `1.294x` faster

What this means for PIC:

- the immediate Rhino compact-dense fix is not "invent another buffer kernel"
- the first production correction is:
  - let Adreno low-memory compact dense use image storage again
  - then rebuild the real `aidlux_adreno_opencl` PIC server artifact and check
    whether `Convolution` time drops inside `cacheblend` / `epic`
- a second-stage improvement can still do joint family+storage tuning, because
  family selection and storage selection are currently independent

## 2026-06-27 Rhino single-entry bench script and hot-shape recheck

What changed:

- Added a single-entry remote bench helper:
  - `.codex/skills/mnn-opencl-pic-attention/scripts/run_opencl_remote_op_bench.sh`
- It always uses `run_test.out` and passes the required OpenCL low-memory test
  args:
  - `./run_test.out <test> 3 0 1 "" 2`
- Rhino and OrangePi keep separate remote roots / library env. This keeps the
  bench entrance unified without mixing Adreno and Mali logic.

Verified Rhino command:

```bash
bash .codex/skills/mnn-opencl-pic-attention/scripts/run_opencl_remote_op_bench.sh \
  --device rhino \
  --case minicpm_hidden_to_attn \
  --rows 519 \
  --warmup 4 \
  --repeat 12
```

Measured Rhino hot-shape results from the single entrance:

- `minicpm_hidden_to_attn`, `1536 -> 2048`, `rows=519`
  - auto/image: `6.0873 ms`
  - forced `pic_quant`, image: `6.0262 ms`
  - forced `pic_quant`, buffer: `6.2135 ms`
  - forced `fp_weight`, image: `7.8731 ms`
  - forced `pic_quant_wg4x32`, image: `7.0867 ms`
  - forced `pic_quant_incache64`, image: `11.4344 ms`

- `minicpm_hidden_to_attn`, `1536 -> 2048`, `rows=269`
  - auto/image: `29.4428 ms`
  - forced `fp_weight`, image: `29.0975 ms`
  - forced `fp_weight`, buffer: `44.6762 ms`
  - forced `pic_quant`, image: `43.6367 ms`
  - forced `pic_quant_wg4x32`, image: `44.0539 ms`

Conclusion:

- Rhino compact dense clearly splits into two buckets:
  - `rows ~= 519`: normal compact-row band; `pic_quant + image` is the right
    direction and beats `fp_weight`, fixed-LWS, and input-cache variants.
  - `rows ~= 269`: separate bad band; family swaps do not solve it, but image
    still beats buffer decisively.
- Immediate Rhino production direction:
  - keep Adreno compact dense on image-backed weights
  - keep `pic_quant` as the primary family for the higher compact-row band
  - treat `rows ~= 216/269` as a dedicated low-/mid-M Adreno kernel problem,
    not a tune-key or storage-choice problem

## 2026-06-27 Rhino status checkpoint: default experiment policy and current effect

Experiment policy to keep fixed:

- Rhino default comparison policy stays:
  - `cpu=max,gpu=max,ddr=max`
  - contexts only: `512,1024,1536,2048,2560`
  - do not include `3072` in the default sweep for now
- Any Rhino result produced under lower CPU / lower GPU profiles is useful for
  hotspot isolation, but it is not the formal default conclusion and must not be
  mixed with the `max/max/max` rows in `benchmark.csv`.

Operator-level findings from the stable single-entry Rhino `run_test.out` bench:

- Score-layer full-Q sparse attention:
  - current auto/default on Rhino still selected `qsplit`
  - measured:
    - `qsplit`: `4320.6 ms`
    - forced flash:
      - `mqtile_hd128_q8k16 + single_piece`: `1551.2 ms`
      - `mqtile_hd128_q4k8_kimg + range_q128`: `1550.9 ms`
  - conclusion:
    - Rhino score-layer routing is still wrong today
    - flash family is about `2.8x` faster than qsplit on the real hot shape
      `MiniCPM5-1B_score_scatter_ctx1024_q519`

- Later sparse attention:
  - current auto/default on Rhino used `mqtile_hd128_q4k16`
  - measured on `MiniCPM5-1B_later_compact_scatter_ctx1024_q519`:
    - auto `mqtile_hd128_q4k16 + range_q64`: `1502.6 ms`
    - forced `mqtile_hd128_q4k8 + range_q64`: `1405.7 ms`
    - forced `row64 + range_q64`: `1789.0 ms`
  - conclusion:
    - later sparse routing also still has headroom
    - the practical Rhino winner on this hot shape is currently
      `mqtile_hd128_q4k8 + range_q64`

- Compact dense MLP hot rows:
  - `minicpm_hidden_to_ffn`, rows=`519`, `1536 -> 4608`
    - auto/image: `11.2236 ms`
    - forced `generic_quant`: `11.1789 ms`
    - forced `pic_quant`: `11.1377 ms`
    - forced `fp_weight`: `13.6078 ms`
  - `minicpm_ffn_to_hidden`, rows=`519`, `4608 -> 1536`
    - auto/image: `12.0376 ms`
    - forced `pic_quant`: `12.0450 ms`
    - forced `fp_weight`: `14.6966 ms`
  - conclusion:
    - dense MLP is no longer choosing a catastrophic family on this 519-row
      band
    - but the remaining room here is small compared with the score-layer
      attention misroute

End-to-end effect conclusion as of this checkpoint:

- Under the formal Rhino default comparison policy
  `cpu=max,gpu=max,ddr=max`, the three target headDim=128 models have not yet
  all reached the requirement that `cacheblend` and `epic` beat
  `normal-full-recompute` by at least `1.2x` across the target budgets.

- Current `benchmark.csv` status for Rhino default-frequency rows:
  - `Llama3.2 3B`, `ctx=1024`:
    - normal: `7.7436 s`
    - best cacheblend: `8.8885 s` -> `0.871x`
    - best epic: `8.1925 s` -> `0.945x`
    - not yet meeting target
  - `Qwen3-8B`, `ctx=1024`:
    - normal: `18.9012 s`
    - best cacheblend: `17.0273 s` -> `1.110x`
    - best epic: `12.5120 s` -> `1.511x`
    - only low-budget epic is clearly over target; cacheblend is still short
  - `MiniCPM5-1B`:
    - old Rhino `max/max/max` rows in `benchmark.csv` are still bad and should
      not be treated as fixed
    - a newer non-default-frequency rerun at `ctx=2048`,
      `cpu=1670400/2323200/2592000,gpu=680000000,ddr=max` already showed:
      - cacheblend `0.40`: `1.469x`
      - cacheblend `0.50`: `1.229x`
      - epic `0.40`: `1.723x`
      - epic `0.50`: `1.354x`
    - this proves the pipeline can be pulled over the target on Rhino, but it
      is not yet the formal default-frequency conclusion

Immediate next implication:

- Do not spend more time on speculative dense-family churn first.
- First fix the Rhino score-layer auto-routing bug so default chooses flash
  instead of qsplit.
- Then fix later sparse default routing to prefer
  `mqtile_hd128_q4k8 + range_q64` on the real Rhino hot band.
- After those two routing corrections, rerun Rhino under
  `cpu=max,gpu=max,ddr=max` and only compare contexts up to `2560`.

## 2026-06-27 Change: Rhino attention routing correction in release path

Goal:

- Turn the confirmed Rhino / Adreno operator-level routing conclusion into the
  default release path instead of leaving it only in bench overrides.

Code changes:

- Added Adreno-only routing helpers in
  `source/backend/opencl/execution/buffer/PagedAttentionAdrenoUtils.*`:
  - `preferAdrenoScoreSparseFlash(...)`
  - `preferAdrenoLaterSparseQ4K8(...)`
- `PagedAttentionBufExecution.cpp` now:
  - defaults score-layer full-Q sparse family to `flash` instead of `qsplit`
    on the Rhino `headDim=128` hot band
  - rejects stale tuned `qsplit` cache replay for that bucket
  - defaults later sparse `headDim=128` hot-band variant to
    `mqtile_hd128_q4k8`
  - rejects stale tuned variant replay for that bucket unless it is already
    `mqtile_hd128_q4k8`

Why:

- Bench evidence already showed:
  - score layer default `qsplit` on Rhino was about `4320.6 ms`
  - flash candidates were about `1551 ms`
  - later sparse default `q4k16` was about `1502.6 ms`
  - later sparse `q4k8 + range_q64` was about `1405.7 ms`
- So the remaining problem was no longer “can the kernels be fast”, but
  “release path is still replaying the wrong family / variant”.

Expected effect:

- Rhino warm/default path should stop falling back to stale `qsplit` on the
  score layer.
- Later sparse high-budget compact scatter should stop defaulting to
  `q4k16` on the same Adreno bucket.
- The next validation step is not more microbench invention; it is rerunning
  Rhino TTFT under `cpu=max,gpu=max,ddr=max`, `ctx<=2560`, and checking whether
  the formal `benchmark.csv` rows move toward the `>=1.2x` target.
