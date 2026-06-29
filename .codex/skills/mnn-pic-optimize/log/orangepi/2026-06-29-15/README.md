# OrangePi 2026-06-29 15

## Summary

- Issue: Qwen3-4B OrangePi OpenCL PIC high-context cacheblend requests failed before completing the missing benchmark rows. `ctx1536/cacheblend0.05` returned an empty reply, and `ctx2048/cacheblend0.30` later killed the service/device state.
- Diagnostic correction: before disabling the suspected path, compare staged cacheblend top-k against legacy top-k on the same device scores.
- Result: `ctx2048/cacheblend0.20` showed staged and legacy both returned valid indices, but the ordered list and set differed. First mismatch was at index 0.
- Fix: disable staged cacheblend top-k on Mali and keep Mali on the deterministic legacy top-k path until the staged kernel is fixed.
- Validation: rebuilt and synced OrangePi artifacts. `ctx1536/cacheblend0.05`, `ctx2048/cacheblend0.20`, and `ctx2048/cacheblend0.30` all returned HTTP 200 after the fix. OrangePi SSH remained responsive and uptime did not reset.
- Benchmark fill: warmed OrangePi Qwen3-4B `ctx1536/2048` rows were later generated and merged into `benchmark.csv`.
- Performance correction: profile now shows top-k was the crash root cause, but it is not the current latency bottleneck. `ctx2048/cacheblend0.20` has `topk_us=117.482ms`, `PicSparseAttention=18.410s`, `Convolution=9.587s`; `ctx2048/cacheblend0.50` has `topk_us=727.736ms`, `Convolution=22.531s`, `PicSparseAttention=13.696s`, and layer0 `PagedAttention=4.452s`.
- qtile sparse attention A/B: forced `mqtile_hd128_q8k16` reduced Qwen3-4B `ctx2048/cacheblend0.20` later sparse attention from `17.833s` to about `5.9-6.0s` and passed `ctx1536/2048` `cacheblend0.20/0.50` stability checks.
- Routing change: Mali headDim128 later sparse now defaults to `mqtile_hd128_q8k16` only for `kvLen<=2048`; score-layer full-Q and larger contexts remain on row32. Adreno routing remains isolated in the existing Adreno helpers.
- Score-layer A/B: forced Mali `score_flash_attention` on `ctx2048 cb20/cb50` was not faster than the existing score qsplit route under the same detail-profile style, so Mali score-layer family was not changed.
- Benchmark refresh: no-profile OrangePi Qwen3-4B cacheblend rows were merged into `benchmark.csv` for `ctx1536/2048` budgets `0.05/0.10/0.20/0.30/0.40/0.50`.
- Epic/cacheblend comparison fix: the apparent `ctx2048` `epic0.40/0.50` slowdown came from mixed benchmark generations. Current cacheblend rows used the Mali q8k16 route, while epic rows were still from the older `pic_fill_after_topk_fix` run. In that old run epic was faster than old cacheblend; after rerunning current default epic, `epic0.40/0.50` became `29.603160s/36.830896s`, faster than current cacheblend `34.940372s/44.708272s`.
- Row variant note: `row32` is the legacy sparse flash workgroup K-lane width, not a token row count. The current OpenCL kernels expose `row32/row64` plus qtile variants (`q4k16/q4k8/q8k16`); `row8/row16` would be new kernels and tuning work, not a cheap routing change.
- Current priority: keep Mali staged top-k disabled/guarded for stability, keep qtile routing shape-gated, then validate broader Mali headDim128 coverage separately before increasing the `kvLen` gate.

## Files

- `context.md`: commands, logs, and latency observations.
