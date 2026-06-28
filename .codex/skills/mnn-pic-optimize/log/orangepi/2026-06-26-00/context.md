# OrangePi 2026-06-26 00 Long Context

以下内容从旧的 `OPTIMIZATION_LOG.md` 迁移，按本设备/日期归档。

## 2026-06-26 Experiment: MiniCPM5-1B later-sparse half-tile skip

Goal:

- Revisit later sparse layer performance first, not dense path.
- Test whether skipping invisible `q·k` dot-products inside mqtile sparse flash half-full causal tiles improves real OpenCL PIC latency on OrangePi.

Code/env:

- Code change:
  - `source/backend/opencl/execution/cl/attention_buf.cl`
  - For `mqtile_sparse_flash_hd64_q4k16`, `mqtile_sparse_flash_hd128_q4k16`, `mqtile_sparse_flash_hd128_q4k8`, `mqtile_sparse_flash_hd128_q8k16`, avoid computing `dot(q, k)` when `k_index > q_logical` in half-full tiles.
- Regenerated embedded OpenCL sources with:

```bash
cd source/backend/opencl/execution/cl
python3 opencl_codegen.py .
```

- Rebuilt and synced OrangePi artifact:

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

Formal latency:

- 1024 / cacheblend 0.50
  - previous stable row in `orangepi_minicpm_ctx1024_cbepic_full_current_20260626_1/summary.csv`:
    - `3.956519 s`
  - new formal run `orangepi_minicpm1024_cb50_halfmask_formal_20260626_1/summary.csv`:
    - `4.098852 s`
- 2048 / cacheblend 0.50
  - previous stable row in `orangepi_minicpm_ctx2048_cbepic1020304050_profile_current2_20260626_1/summary.csv`:
    - `12.000538 s`
  - new formal run `orangepi_minicpm2048_cb50_halfmask_formal_20260626_1/summary.csv`:
    - `11.656823 s`

Profile detail:

- Warning:
  - `MNN_PAGED_ATTENTION_PROFILE=1` disables sparse variant/schedule online tuning in code.
  - Right after kernel source changes, the first profiled run may fall back to default `mqtile_hd128_q4k16`, which is not representative of warmed production routing.
  - Correct procedure is: formal non-profile heavy warm first, then profile again using the updated tune cache.

- 1024 / cacheblend 0.50 / after warm+tune:
  - run: `orangepi_minicpm1024_cb50_halfmask_profile_aftertune_20260626_1`
  - score layer:
    - `family=qsplit`
    - `score_qsplit_attention us≈102 ms`
  - later sparse layers:
    - `variant=mqtile_hd128_q8k16`
    - `schedule=range_q128`
    - `q_chunk=128`, `q_split=10`
    - measured `flash_us≈46.7-47.9 ms / layer`
  - comparison against older tuned log `orangepi_minicpm1024_cacheblend4050_split_profile_20260626_1`:
    - older tuned `flash_us≈46.7-47.6 ms / layer`
    - conclusion: half-tile dot skip is effectively neutral on this shape once the best tuned variant is restored.

- 2048 / cacheblend 0.50 / after warm+tune:
  - run: `orangepi_minicpm2048_cb50_halfmask_profile_aftertune_20260626_1`
  - score layer:
    - `family=qsplit`
    - `score_qsplit_attention us≈426 ms`
  - later sparse layers:
    - `variant=mqtile_hd128_q8k16`
    - `schedule=range_q128`
    - `q_chunk=128`, `q_split=18`
    - measured `flash_us≈181-183 ms / layer`
    - `qk_rect_tiles=64827`, `qk_active_tiles=62092`, `qk_row_tiles=247087`
  - graph profile, measured request:
    - `PicSparseAttention total_ms=4098.742`
    - `Convolution total_ms=3917.054`
    - `PagedAttention total_ms=2070.718`
    - `PicScoreAttention total_ms=1145.355`
    - `Raster total_ms=419.585`

Conclusion:

- The half-full tile white-compute hypothesis was valid semantically, but on OrangePi/Mali the implemented dot-skip does not produce a meaningful tuned end-to-end gain.
- After warm+tune, later sparse routing is already landing on the expected family:
  - `score layer -> qsplit`
  - `later sparse -> mqtile_hd128_q8k16 + range_q128`
- On 2048 / cacheblend 0.50, later sparse attention is still a major hotspot, but it is no longer the only one; compact dense `Convolution` is essentially co-first.
- Therefore, continuing to micro-tune only the current later sparse kernel will not deliver the user target by itself. The next real gain must come from either:
  - a more structural later-sparse change such as logical-range bucketed Q grouping / a stronger hd128 q4-style variant, or
  - parallel dense MLP / compact Conv optimization.

Next:

- Treat current mqtile half-tile skip as experimental, not yet justified as a production performance win.
- If staying on the later sparse line, prioritize structural work:
  - reduce intra-Q-tile logical spread rather than only skipping masked lanes after load/reduction.
- In parallel, keep dense compact MLP optimization active because 2048/cb50 shows `PicSparseAttention` and `Convolution` are now joint bottlenecks.

## OrangePi Follow-up: 2026-06-26

Accepted dense-side change was broadened from the original `M > 384` pocket to
the full high-arithmetic compact-row band:

```text
source/backend/opencl/execution/buffer/ConvBufLowMemoryExecution.cpp
  useFPWeightGemmLowMemory():
    if M > 192 && K >= 1024 && N >= 1024 && ratio >= 0.35
      mAlignM = 64
```

Reason:

- The MiniCPM/Llama3.2-3B follow-up anomaly was another FP-weight Xgemm bucket:
  `M~=216` stayed in the 32-row align regime and rounded to `alignM=224`.
- `M~=317` also stays in the 32-row regime but rounds to `alignM=320`, which
  is already fine on OrangePi Mali.
- Promoting the `M~=216` class to 64-row alignment moves it to `alignM=256`
  without touching sparse attention semantics or `/v1/prefill/text`.

Validated device-side runs:

```text
Llama3.2-3B:
  .cache/latency_budget_20260625/orangepi_llama32_3b_ctx1024_alignm64b_profileoff_20260626_1/summary.csv

MiniCPM5-1B:
  .cache/latency_budget_20260625/orangepi_minicpm_ctx512_1024_alignm64b_profileoff_20260626_1/summary.csv
```

Validated deltas:

- `Llama3.2 3B`, ctx1024
  - `cacheblend 0.20: 16.096029s -> 5.864512s`
  - `epic 0.20: 15.796590s -> 5.584404s`
  - `cacheblend/epic 0.30/0.40` stay essentially unchanged
  - budget scaling is restored:
    - `0.20 < 0.30 < 0.40`

- `MiniCPM5-1B`, ctx512
  - `cacheblend 0.40: 3.308097s -> 1.487589s`
  - `cacheblend 0.50: 4.075209s -> 1.737152s`
  - `epic 0.40: 3.264958s -> 1.456020s`
  - `epic 0.50: 4.026324s -> 1.652854s`

- `MiniCPM5-1B`, ctx1024
  - `cacheblend 0.20: 3.936728s -> 2.142356s`
  - `epic 0.20: 3.777856s -> 1.958362s`
  - `0.30/0.40/0.50` are essentially unchanged, which means the `M~=216`
    bucket was the real culprit there, not the whole dense path.

Rejected follow-up:

- A second experiment forced some `512 < M <= 576` spill-row compact dense
  shapes back onto the PIC quant path using an extra arithmetic-ratio filter.
- Target rerun `orangepi_minicpm_ctx1024_cbepic50_ratio6spill_20260626_1`
  showed no real gain:
  - `cacheblend 0.50: 4.625373s -> 4.667668s`
  - `epic 0.50: 4.276527s -> 4.252858s`
- That branch was reverted and is not part of the accepted source state.

Updated interpretation:

- There were at least two separate OrangePi dense FP-weight bucket cliffs:
  - `M~=471 -> alignM=480`
  - `M~=216 -> alignM=224`
- Both are fixed by the accepted `mAlignM=64` promotion for
  high-arithmetic large-channel compact-row shapes above `M > 192`.
- Remaining slow cases such as `MiniCPM5-1B ctx1024 0.50` are not explained by
  the `216/471` bucket anymore and need fresh attribution before more routing
  changes are accepted.

2026-06-26 follow-up on OrangePi MiniCPM5-1B / Llama3.2-3B ctx1024 `0.50`:

Accepted change:

```text
source/backend/opencl/execution/buffer/ConvBufLowMemoryExecution.cpp
  onResize():
    remove the hard clamp that forced 512<M<=576 PIC compact rows onto
    generic quant after tune lookup
    add a dedicated tune namespace suffix for that spill-row band:
      convBufLowMemory_<ic>_<oc>_picspill_v2
```

Reason:

- Fresh profile confirmed the remaining MiniCPM ctx1024 `0.50` cliff was still
  a routing bug, not a general dense-shape problem:
  - `batch=418` compact rows: `use_fp_weight=1`
  - `batch=519` compact rows: `use_fp_weight=0`
- That difference came from the spill-row clamp in `onResize()`, which overrode
  tuned info and kept replaying the old generic-quant decision.
- The new `_picspill_v2` tune namespace forces a real re-benchmark for this
  narrow 512<M<=576 band without disturbing the already-validated buckets.

Validated device-side probe:

```text
.cache/latency_budget_20260625/orangepi_minicpm1024_cbepic50_spillv2_profile_20260626_1/summary.csv
```

Key profile deltas vs the older `orangepi_minicpm1024_epic50_profileprobe_20260626_1`:

- `batch=519` now routes to `use_fp_weight=1`
- `epic 0.50`
  - `Convolution total_ms: 2821.449 -> 2164.255`
  - `PicSparseAttention total_ms: 826.228 -> 617.097`
  - profile latency: `4.575798s -> 3.736992s`
- `cacheblend 0.50`
  - `Convolution total_ms: 2157.547`
  - `PicSparseAttention total_ms: 936.731`
  - profile latency: `4.068520s`

Validated formal rerun:

```text
.cache/latency_budget_20260625/orangepi_minicpm1024_cbepic50_spillv2_profileoff_20260626_1/summary.csv
```

- `MiniCPM5-1B`, ctx1024
  - `cacheblend 0.50: 4.627990s -> 3.984845s`
  - `epic 0.50: 4.248407s -> 3.612556s`

Important follow-up finding:

- After the spill-row fix, MiniCPM ctx1024 `0.50` no longer has a dense-route
  anomaly. The remaining dense compact MLP cost scales roughly linearly:
  - `418x4608 -> 1536 down_proj` about `16.7 ms`
  - `519x4608 -> 1536 down_proj` about `21.9 ms`
- The remaining MiniCPM `0.50` gap is now shared between:
  - full `layer0` `PagedAttention` (~`0.51s`)
  - normal compact-row `Convolution` growth at `M=519`
  - later `PicSparseAttention`

Llama3.2-3B follow-up:

- Profiled `orangepi_llama32_3b_ctx1024_cbepic50_spillv2_profile_20260626_1`
  and confirmed its `batch=519` compact rows also route to `use_fp_weight=1`.
- However formal `0.50` latency stayed essentially unchanged:
  - `epic 0.50: 10.638186s -> 10.727524s`
  - `cacheblend 0.50: 11.130283s -> 11.237782s`
- Interpretation:
  - Llama3.2-3B was already sufficiently dense-bound that the spill-row reroute
    does not produce a reportable end-to-end gain.
  - It still satisfies the `>=1.4x` target vs normal full recompute at ctx1024,
    so keep the old `benchmark.csv` rows rather than overwriting with noise.

Rejected follow-up:

```text
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
  runSparseFastPrefill():
    force cacheblend activeLen>=384 to abandon single-piece sparse flash and
    use multi-piece range-aware splitting
```

Measured result:

- `orangepi_minicpm1024_cacheblend4050_split_profile_20260626_1`
- This regressed badly:
  - `cacheblend 0.40 profile total_ms: 3301.037 -> 3324.684`, but
    `PicSparseAttention: 744.132 -> 782.265`
  - `cacheblend 0.50 profile total_ms: 4068.520 -> 4167.835`
  - `PicSparseAttention: 936.731 -> 1048.811`
- The extra piece/launch overhead outweighed any causal-range savings, so that
  change was reverted and is not part of the accepted source state.

Llama3.2-3B small-row routing follow-up:

- The `0.10 > 0.20` anomaly on OrangePi `Llama3.2-3B ctx1024` had the same
  symptom as MiniCPM, but not the same safe threshold.
- A first broad heuristic (`batch * 8 <= minChannel`) fixed `batch=115`, but
  it also pushed `batch=216` `3072->3072/8192/1024` compact rows onto the
  generic path and regressed `0.20` profile totals:
  - `Convolution total_ms` rose from about `4149-4212 ms` to about `5080 ms`.
- The accepted fix is narrower and shape-aware:

```text
compactTinyArithmeticRows :=
  usePicCompactKernel &&
  int4 &&
  batch^2 <= 5 * maxChannel
```

- This keeps truly tiny compact rows on the generic quant path while allowing
  mid rows like `batch=216` to retune normally.
- Verified route on OrangePi:
  - `batch=115`:
    - `3072->3072`, `3072->8192`, `8192->3072`, `3072->1024`
    - all `use_fp_weight=0`
  - `batch=216`:
    - `3072->3072`, `3072->8192`, `8192->3072`, `3072->1024`
    - all `use_fp_weight=1`
- Formal rerun:

```text
.cache/latency_budget_20260625/orangepi_llama32_3b_ctx1024_cbepic_full_smallratiofix2_heavy_20260626_1/summary.csv
```

- New OrangePi `Llama3.2-3B`, ctx1024 formal latency:
  - `cacheblend`: `0.05 3.380340`, `0.10 4.372394`, `0.20 5.854538`,
    `0.30 7.268154`, `0.40 9.322362`, `0.50 11.118220`
  - `epic`: `0.05 3.312802`, `0.10 4.199468`, `0.20 5.558722`,
    `0.30 6.841970`, `0.40 8.852828`, `0.50 10.627023`
- Effect:
  - `cacheblend 0.10: 8.901100s -> 4.372394s`
  - `epic 0.10: 8.735235s -> 4.199468s`
  - `0.20+` stays essentially aligned with the previous good data.
- Result:
  - `cacheblend` and `epic` are both monotonic at ctx1024.
  - Both modes are now `>1.4x` faster than `normal-full-recompute` across
    `0.05-0.50`.

Score-layer sparse family tune follow-up:

- Removed the old `headDim=128 + queryRowsAreFull` hard gate in
  `PagedAttentionBufExecution::canUseSparseFastPrefill()`.
- Wired two new tune layers into
  `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`:
  - top-level score-layer family tune:
    - `score_qsplit_attention`
    - `score_flash_attention`
  - sparse-flash schedule tune:
    - `single_piece`
    - `range_q64`
    - `range_q128`
- Also made sparse-flash variant tune schedule-aware by including the chosen
  schedule in its tune key/shape.

Targeted OrangePi validation:

```text
.cache/latency_budget_20260625/orangepi_minicpm_ctx2048_scorefamily_tuned_20260626_1/summary.csv
```

- `MiniCPM5-1B`, ctx2048:
  - `cacheblend 0.40: 9.620878s -> 9.682685s`
  - `cacheblend 0.50: 11.892315s -> 11.610874s`
  - `epic 0.40: 7.119305s -> 7.076168s`
  - `epic 0.50: 8.922163s -> 9.013160s`
- Interpretation:
  - score-layer family tuning is live, but it does not unlock a material
    end-to-end gain on these shapes.
  - The post-change results are effectively flat against the current
    `benchmark.csv` rows, so the table should not be overwritten with noise.

Cached-family verification with profile:

```text
.cache/latency_budget_20260625/orangepi_minicpm_ctx2048_scorefamily_profile_aftertune_20260626_1/
```

- Heavy-tuned cache still resolves the score layer to `qsplit` on all tested
  `MiniCPM5-1B ctx2048` high-budget shapes:
  - `cacheblend 0.40`: `family=qsplit`, `score_qsplit_attention us≈351ms`
  - `cacheblend 0.50`: `family=qsplit`, `score_qsplit_attention us≈418ms`
  - `epic 0.40`: `family=qsplit`, `score_qsplit_attention us≈155ms`
  - `epic 0.50`: `family=qsplit`, `score_qsplit_attention us≈231ms`
- Later sparse layers still use sparse flash with:
  - `schedule=range_q64`
  - `variant=mqtile_hd128_q4k16`

Conclusion:

- The problem on OrangePi `MiniCPM5-1B ctx2048 0.40-0.50` is no longer
  “score layer was artificially pinned to qsplit because family tune did not
  exist”.
- After enabling the tune, the runtime still prefers `qsplit`, which means the
  next optimization target should move to:
  - score-layer `qsplit` itself, or
  - `cacheblend_score` cost before sparse recompute,
  rather than assuming `score_flash_attention` is the missing fast path.

Rhino Pi X1 / Adreno current-code rebaseline (MiniCPM5-1B, ctx2560, safe-low):

- Old Rhino rows in `benchmark.csv` for ctx2560 were stale and came from an
  earlier artifact where later sparse layers fell back to `row`.
- Current rebuilt artifact + remote rerun:
  - `normal-full-recompute`: `6.795897s`
  - `cacheblend 0.10`: `1.820802s`
  - `epic 0.10`: `1.551523s`
- Relative speedup vs current Rhino normal:
  - `cacheblend 0.10`: `3.73x`
  - `epic 0.10`: `4.38x`
- This means Rhino is no longer in the old “cacheblend/epic slower than
  normal” failure mode for this shape.

Current Rhino profile evidence:

```text
.cache/latency_budget_20260625/rhino_minicpm_ctx2560_cacheblend010_currprofile_20260626_1/summary.csv
.cache/latency_budget_20260625/rhino_minicpm_ctx2560_epic010_currprofile_20260626_1/summary.csv
remote:
/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/rhino_minicpm_ctx2560_cacheblend010_currprofile_20260626_1/pic_server_ctx2560.log
/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/rhino_minicpm_ctx2560_epic010_currprofile_20260626_1/pic_server_ctx2560.log
```

- Full prefill layer0 is on the Adreno-specific fast path:
  - `prefill_attention_adreno_gemm`
- Score layer now routes through family tune and resolves to flash:
  - `score_sparse_family -> flash`
  - `score_flash_attention variant=mqtile_hd128_q8k16 schedule=range_q128`
- Later sparse layers are no longer `row`; they are:
  - `sparse_flash_attention variant=mqtile_hd128_q4k8 schedule=range_q128`
- The relevant source-side tune keys are already distribution-aware:
  - score family tune includes `sparseLogicalWorkPermille` and selected ratio
    in `paged_score_sparse_family_*`
  - sparse flash schedule / variant tune also include
    `sparseLogicalWorkPermille` and selected ratio in
    `paged_sparse_flash_schedule_*` / `paged_sparse_flash_variant_*`

Rhino cacheblend vs epic bottleneck split:

- `cacheblend 0.10` graph profile total (`~2533.8 ms` with profile overhead):
  - `Convolution`: `~785.0 ms`
  - `PicSparseAttention`: `~624.1 ms`
  - `Raster`: `~487.9 ms`
  - `PagedAttention`: `~166.7 ms`
  - `PicScoreAttention`: `~109.3 ms`
- `epic 0.10` graph profile total (`~2297.9 ms` with profile overhead):
  - `Convolution`: `~794.5 ms`
  - `Raster`: `~514.9 ms`
  - `PicSparseAttention`: `~406.6 ms`
  - `PagedAttention`: `~162.2 ms`
  - `PicScoreAttention`: `~30.5 ms`

Interpretation:

- On current Rhino, attention fallback is no longer the primary problem.
- The first bottleneck has moved to compact dense `Convolution` and `Raster`.
- `cacheblend` is still slower than `epic` at the same ratio mainly because:
  - later sparse attention sees a much larger visible-K span
  - `cacheblend_score` adds a non-trivial extra stage

Cacheblend-specific Rhino findings:

- `cacheblend_score` itself took about `66 ms`, where:
  - score kernel only about `0.9 ms`
  - top-k about `63.7 ms`
- Current OpenCL top-k kernel is structurally weak for `token_count > 1024`:
  - `pic_cacheblend_topk` falls back to an iterative
    `for k in top_k -> scan token_count -> check prev selected`
    scheme
  - that is effectively `O(top_k * token_count)` and is a real Adreno
    hotspot for `pic_tokens ~= 2546, top_k ~= 255`
- This is now a concrete optimization target on Rhino, not a speculative one.

Compact dense / image-path findings on Rhino:

- For current compact sparse rows (`batch=269`) the profile shows:
  - `pic_compact=1`
  - `use_fp_weight=0`
- For full rows (`batch=2560`) it shows:
  - `pic_compact=0`
  - `use_fp_weight=1`
- The 1x1 low-memory int4 path already enables image-backed weight reads when
  the weight shape fits the image limits, so “just switch Adreno to image read”
  is not the missing lever here.
- The real issue is that compact dense still has only a narrow family:
  - generic `pic_gemm_b4_c8_*` with LWS tuning
  - versus FP-weight/Xgemm
  - but no richer family tune on Adreno comparable to the attention-side
    schedule/variant routing

Recommended Adreno next steps:

1. Keep current attention routing as-is for Rhino.
   - It already fixed the catastrophic later-sparse row fallback.
   - Do not spend the next cycle rewriting sparse attention again before dense
     and top-k are addressed.
2. Add an Adreno-specific compact-dense family tune, not just LWS tuning.
   - Family candidates should at least distinguish:
     - `pic_gemm_b4_c8_*`
     - FP-weight/Xgemm
     - future wider compact tiles if added
   - Key it by `M/N/K` aspect bucket, not only raw `batch`.
3. Reduce `Raster` around compact Linear/MLP.
   - Current Rhino profile shows nearly `0.5s` of raster on the same request.
   - The likely win is a buffer-native compact Linear path or fused
     layout/epilogue, not more attention tuning.
4. Replace `pic_cacheblend_topk` with a two-stage GPU select/sort kernel for
   `token_count > 1024`.
   - The current iterative scan kernel is an actual cacheblend tax on Adreno.
5. If further routing is needed, separate `epic`-like prefix distributions from
   scattered `cacheblend` distributions using the existing logical-work signal,
   rather than adding new env switches.

### Rhino / Adreno compact-dense family retune on high-budget `batch=1287`

Code changes:

- Extended Adreno compact-dense family tuning eligibility to
  `M <= min(1536, 2 * min(N, K))` instead of stopping around the old PIC
  compact-row gate.
- Switched discrete family lookup away from `getTunedInfo()` nearest-shape
  reuse to an exact-shape helper:
  - `source/backend/opencl/execution/buffer/ConvBufLowMemoryExecution.cpp`
  - exact lookup is now used for Adreno dense family routing and the legacy
    dense binary route cache
- Versioned the Adreno family tune namespace to
  `convBufLowMemory_adreno_family_exact_v1_*` so current runs do not replay
  older approximate decisions.
- Added profile-only logging of candidate family execute times.

Rhino targeted rerun:

- Device: `Rhino Pi X1`
- Model: `MiniCPM5-1B`
- Context: `2560`
- Frequency: `cpu=1670400/2323200/2592000,gpu=475000000,ddr=max`
- Run: `rhino_minicpm_ctx2560_cbepic050_profile_20260626_4`

Measured 0.50 latency after the retune changes:

- `normal-full-recompute`: `6.771868 s`
- `cacheblend 0.50`: `6.175386 s`
- `epic 0.50`: `6.128734 s`

Key result:

- The dense route was not a wrong cached choice. After exact retune, the same
  high-budget compact-dense shapes still preferred `fp_weight`.
- Candidate family times logged on Rhino for `batch=1287`:
  - `2048 -> 1536`:
    - `generic_quant ~= 10.262 ms`
    - `pic_quant ~= 10.209 ms`
    - `fp_weight ~= 9.201 ms`
  - `1536 -> 4608`:
    - `generic_quant ~= 22.434 ms`
    - `pic_quant ~= 22.342 ms`
    - `fp_weight ~= 20.431 ms`
  - `4608 -> 1536`:
    - `generic_quant ~= 22.974 ms`
    - `pic_quant ~= 23.019 ms`
    - `fp_weight ~= 20.253 ms`
  - `1536 -> 2048`:
    - `generic_quant ~= 10.096 ms`
    - `pic_quant ~= 10.036 ms`
    - `fp_weight ~= 9.303 ms`

Interpretation:

- On Adreno high-budget compact rows, current quant dense families are
  genuinely slower than `fp_weight` by about `9%-13%`.
- This is no longer a tune-key bug or nearest-shape cache bug.
- The next dense optimization should target kernel shape/epilogue/layout
  efficiency for medium/high `M`, not more family-key tweaking.

Image-path note:

- The low-memory 1x1 int4 dense path already enables image-backed weight
  storage when the weight shape fits:
  - `UP_DIV(IC, actual_packCin) <= 16384`
  - `ROUND_UP(OC, 8) <= 16384`
- The Rhino high-budget `1536/2048/4608` channel shapes satisfy this, so these
  dense ops are already on the weight-image path.
- Therefore, “switch dense to image read on Adreno” is not the missing lever
  for this hotspot. The problem is the current quant GEMM family itself.
