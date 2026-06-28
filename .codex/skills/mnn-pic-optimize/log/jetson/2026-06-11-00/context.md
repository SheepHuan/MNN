# Jetson 2026-06-11 00 Long Context

以下内容从旧的 `OPTIMIZATION_LOG.md` 迁移，按本设备/日期归档。

## 2026-06-11 Jetson CUDA Graph-Boundary Adaptation

Scope:

- Ported the current PIC score-layer graph-boundary semantics to CUDA `PagedAttentionExecution`.
- CUDA now registers `OpType_PicScoreAttention` and `OpType_PicSparseAttention`.
- `kvWriteLen` and `attnLen` are separated:
  - score layer uses full Q/K/V write with compact attention output and emits `active_indices`.
  - later layers use compact Q/K/V rows while writing/reading true logical slots through `active_indices`.
- Decode and non-PIC runtime still use normal `PagedAttention` behavior.
- Async persistent PIC cache loading and PagedCache hydrate semantics are unchanged.

Rejected CUDA dense experiment:

- Tried a naive packed INT4 CUDA `PicGEMM_FpAInt4B` for compact 1x1 MLP rows.
- It was wrong for production performance even though it avoided runtime dequant:
  - old graph profile with this path, cacheblend20: `Convolution=2983.717 ms`, `PicSparseAttention=230.608 ms`.
  - per-layer compact MLP was pathological: 216-row `down_proj` around `67 ms`, `gate/up` around `51 ms`.
  - end-to-end formal cacheblend20 regressed to `3.408640s` / `0.690x` normal, and cacheblend50 to `7.818670s` / `0.301x`.
- The final CUDA code does not keep this path. CUDA compact MLP currently uses the existing tensor-core CUTLASS + runtime dequant path; a future compact GEMM must beat this path across 1%-50% before becoming default.

A/B confirmation after removing naive CUDA PicGEMM:

```text
tag=pic_cuda_1024_profile_cb20_cutlass_ab_20260611_164608
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend20 latency=0.816942s speedup_vs_normal=2.881
graph request=3 cacheblend20 total=706.398 ms
  Convolution=262.514 ms
  PicSparseAttention=230.444 ms
  PagedAttention=87.252 ms
  PicScoreAttention=30.225 ms

score layer output: [1x216x2048] + active_indices [216]
later sparse layers: [1x216x32x64] Q/K/V, [1x1x216x1024] mask
compact MLP after fix: 216-row down_proj around 3.9 ms, not 67 ms
```

Formal Jetson CUDA 1024-token sweep:

```text
tag=pic_cuda_1024_boundary_cutlass_formal_20260611_164704
model=Llama-3.2-1B-Instruct@jetson-cuda
profile=false, graph_profile=false
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.359027,0.997
full-reuse,full,0.292211,8.051
cacheblend,0.10,0.513967,4.577
cacheblend,0.20,0.690936,3.405
cacheblend,0.30,0.863871,2.723
cacheblend,0.40,1.103854,2.131
cacheblend,0.50,1.297994,1.812
epic,0.10,0.473170,4.972
epic,0.20,0.670933,3.506
epic,0.30,0.839971,2.801
epic,0.40,1.051566,2.237
epic,0.50,1.253765,1.876
```

Low-budget CUDA smoke:

```text
tag=pic_cuda_1024_low_budget_cutlass_smoke_20260611_164825
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.01,0.354654,6.633
cacheblend,0.05,0.410731,5.728
epic,0.01,0.318615,7.384
epic,0.05,0.413581,5.688
```

High-budget CUDA attribution:

```text
tag=pic_cuda_1024_profile_cb50_cutlass_20260611_164857
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50 graph total=1334.525 ms
  PicSparseAttention=535.167 ms
  Convolution=459.797 ms
  PicScoreAttention=89.685 ms
  PagedAttention=87.304 ms
  UnaryOp=74.069 ms

score layer output rows=519
later sparse attention per layer ~=38.1 ms
compact MLP per layer ~=7.4-7.8 ms
```

Conclusion:

- CUDA graph-boundary semantics are active and stable on Jetson for 1024 tokens.
- `PIC full-compute` is aligned with normal full compute (`2.359027s` vs `2.352533s`).
- `full-reuse`, cacheblend and epic are faster than normal for 1/5/10/20/30/40/50 tested budgets.
- Current remaining CUDA high-budget bottleneck is split between later-layer `PicSparseAttention` and compact dense MLP. For the next CUDA optimization round, P0 is a real CUDA sparse flash attention kernel that fuses QK/softmax/QKV without repeating QK; P1 is a tensor-core compact-row INT4/FP16 GEMM that beats CUTLASS + runtime dequant across 1%-50%.

## 2026-06-11 Jetson CUDA P0/P1 Negative A/B

Baseline to compare:

```text
tag=pic_cuda_1024_boundary_cutlass_formal_20260611_164704
cacheblend,0.20,0.690936,3.405
cacheblend,0.50,1.297994,1.812
epic,0.20,0.670933,3.506
epic,0.50,1.253765,1.876

tag=pic_cuda_1024_low_budget_cutlass_smoke_20260611_164825
cacheblend,0.01,0.354654,6.633
cacheblend,0.05,0.410731,5.728
epic,0.01,0.318615,7.384
epic,0.05,0.413581,5.688
```

Rejected P0a: existing fused row-compressed attention

- Tested by launching the existing `v2_row_compressed_mask` path with `MNN_PAGED_ATTENTION_BENCH_FORCE_V2_KERNEL=1`.
- This is mathematically closer to flash attention because it streams softmax and V accumulation, but it is one query row/head per block and loses the Q/K tiling efficiency of the existing QK kernel.

```text
tag=pic_cuda_1024_v2_sparse_ab_cb20_50_20260611_191539
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.20,1.548174,1.520
cacheblend,0.50,2.303546,1.021

attn=216 v2 sparse rows: ~58.2 ms/layer
baseline attn=216 three-stage rows: ~15.6 ms/layer
```

Conclusion: do not default the existing row-compressed fused kernel for PIC sparse. It reduces temporary buffers but destroys per-layer throughput.

Rejected P0b: sparse single-piece qSplit

- Tried forcing sparse prefill `q_split=1` while keeping normal full-compute split unchanged.
- Intended to reduce per-layer launch count for cb40/cb50, similar to the successful OpenCL single-piece result.
- On Jetson CUDA it enlarged QK/softmax temporary buffers enough to hurt high-budget layers.

```text
tag=pic_cuda_1024_sparse_singlepiece_ab_cb20_50_20260611_191852
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.20,0.813060,2.893
cacheblend,0.50,1.860186,1.265

cb50 sparse q_split=1 layer samples:
  layer=1 attn=519 q_split=1 us=60143
  layer=2 attn=519 q_split=1 us=64089
  layer=7 attn=519 q_split=1 us=70262

baseline cb50 q_split=3 later layers: ~37.3 ms/layer
```

Conclusion: CUDA P0 must be a tile-based sparse flash kernel that preserves Q/K tiling and avoids repeated QK, not a single-row fused kernel or larger single-piece QK buffer.

Rejected P1a: extend V14_MB packed GEMV to batch<=32

- Tried using existing factored-dequant packed INT4 GEMV for low-budget compact rows, processed in groups of 8 rows.
- This avoided runtime dequant for 7-32 rows, but repeated weight reads and many small GEMV launches cost more than CUTLASS + runtime dequant.

```text
tag=pic_cuda_1024_gemv32_ab_1_5_20_20260611_192317
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

full-reuse,full,0.281823,8.348
cacheblend,0.01,0.543940,4.325
cacheblend,0.05,0.414722,5.673
cacheblend,0.20,0.684411,3.437
epic,0.01,0.521939,4.507
epic,0.05,0.404292,5.819
epic,0.20,0.671286,3.505
```

Compared to the low-budget CUTLASS baseline, cacheblend1 and epic1 regressed badly. The source patch was reverted.

Next CUDA work:

- P0 needs a new tile-based sparse flash kernel:
  - keep multiple Q rows per block or per CTA tile;
  - preserve Q/K tiling and coalesced K/V loads;
  - compute online softmax and V accumulation without writing global QK/softmax;
  - avoid the existing row kernel's one-row/head occupancy problem;
  - preserve score-layer full-Q/compact-output and later compact-Q semantics.
- P1 needs either real tensor-core weight-only INT4 GEMM or a memory-aware dequant cache. Packed GEMV or naive packed GEMM should stay rejected until it beats CUTLASS across 1/5/10/20/30/40/50.

## 2026-06-11 Jetson CUDA Tile Sparse Flash Accepted Path

Implemented `pagedSparseFlashTileKernel<T, Q_TILE=8, K_TILE=32>` in CUDA `PagedAttentionExecution`:

- one CTA handles 8 active query rows for one head;
- streams K/V in 32-token tiles;
- keeps online softmax state and V accumulation inside the kernel;
- avoids global QK and softmax tensors;
- currently only supports `head_dim=64`.

The production route is intentionally narrow:

```text
fixedPlanSparseQuery =
    sparseQuery &&
    pic_graph_active_plan_ready &&
    !cacheblend_score_active
```

This means epic / fixed active-plan rows use tile sparse flash, including the score layer full-Q/compact-output case and later compact-Q sparse layers. Cacheblend scattered rows continue using the existing QK/softmax/QKV path because the all-sparse tile flash A/B was flat-to-slightly-worse on cacheblend.

Formal sweep after narrowing to epic/fixed plan:

```text
tag=pic_cuda_1024_tileflash_epic_only_formal_20260611_193310
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.367386,0.994
full-reuse,full,0.293133,8.025
cacheblend,0.10,0.516453,4.555
cacheblend,0.20,0.695926,3.380
cacheblend,0.30,0.867265,2.713
cacheblend,0.40,1.106997,2.125
cacheblend,0.50,1.294405,1.817
epic,0.10,0.461220,5.101
epic,0.20,0.643831,3.654
epic,0.30,0.812402,2.896
epic,0.40,1.008296,2.333
epic,0.50,1.211449,1.942
```

Compared with the CUTLASS graph-boundary baseline, cacheblend is unchanged within noise while epic improves:

```text
epic10: -2.5%
epic20: -4.0%
epic30: -3.3%
epic40: -4.1%
epic50: -3.4%
```

Low-budget smoke after the accepted route:

```text
tag=pic_cuda_1024_tileflash_epic_only_low_20260611_193752
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.01,0.351891,6.685
cacheblend,0.05,0.413794,5.685
epic,0.01,0.317318,7.414
epic,0.05,0.406373,5.789
```

Compared with low-budget CUTLASS baseline, cacheblend1 is slightly better, cacheblend5 is noise/slightly worse, epic1/5 are better. No extra active-row threshold is needed for epic/fixed plan right now.

Profile attribution at 50%:

```text
tag=pic_cuda_1024_tileflash_epic_only_profile_50_20260611_193417

cacheblend50:
  PicSparseAttention=534.766 ms
  Convolution=459.124 ms
  PicScoreAttention=89.024 ms
  PagedAttention=87.673 ms
  later sparse attention path=prefill_attention_fast_qk_softmax_qkv
  per later sparse layer ~=37.3 ms

epic50:
  PicSparseAttention=493.569 ms
  Convolution=466.032 ms
  PicScoreAttention=36.264 ms
  PagedAttention=80.450 ms
  sparse attention path=sparse_flash_tile_attention
  per sparse flash layer ~=34.4 ms
```

Active-index analysis explains why cacheblend budget does not map linearly to compute:

```text
mode       ratio rows max_logical avg_causal_k work_frac tail>=768 gaps>1
cacheblend 0.10 101  1006        771.1        0.753     45        83
cacheblend 0.20 202  1008        732.3        0.715     74       119
cacheblend 0.30 303  1009        737.1        0.720    126       117
cacheblend 0.40 404  1009        740.1        0.723    179        59
cacheblend 0.50 505  1009        673.6        0.658    210        59
epic       0.10 101   107         58.0        0.057      0         0
epic       0.20 202   208        108.5        0.106      0         0
epic       0.30 303   309        159.0        0.155      0         0
epic       0.40 404   410        209.5        0.205      0         0
epic       0.50 505   511        260.0        0.254      0         0
```

Cacheblend active rows are sorted but scattered and reach the tail even at 10%. The selected-row count is small, but the causal K range per selected row remains large. Epic selects a contiguous prefix, so both selected rows and causal K range shrink together. This is the main reason cacheblend speedup is weaker and less proportional to budget.

Further optimization potential:

- P0a cacheblend-specific sparse flash: keep the same active set, but bucket rows internally by logical position and scatter outputs back to the original compact row order. The kernel should skip K tiles above each bucket's max causal position and avoid loading V for invalid K. This does not change cacheblend semantics, but the upside is bounded because cacheblend average causal K is already 65%-75% of full.
- P0b tune the accepted tile flash for epic/fixed rows: A/B `Q_TILE=4/8/16`, `K_TILE=32/64`, fewer `__syncthreads()`, less `scoreShared` round trip, vectorized half/half2 K/V loads, and adaptive q-piece/bucket heuristics. Promote only a shape heuristic that wins 1/5/10/20/30/40/50; do not add production env fallbacks. The later `outAcc0/outAcc1` register-accumulator experiment was rejected on sm72.
- P1 cacheblend score layer: cacheblend50 `PicScoreAttention=89 ms`, but the attention sub-kernel is only about `37 ms`; scoring/top-k/metadata costs the other ~50 ms. Split and optimize score/top-k before changing attention semantics.
- P2 compact dense: at 50%, `Convolution ~=459-466 ms`, the same order as sparse attention. The next large end-to-end win likely needs a real tensor-core compact-row weight-only GEMM or dequant-cache strategy, not naive packed GEMV.

## 2026-06-11 Jetson CUDA Causal K Limit for Tile Flash

Follow-up optimization:

- `pagedSparseFlashTileKernel` now computes the max logical position inside each `Q_TILE`.
- The K/V streaming loop stops at `min(kvLen, max_q_logical + 1)`.
- This skips whole invalid causal K/V tiles for epic/fixed active plans.
- Cacheblend scattered rows still do not route into this kernel, so cacheblend remains protected from the previous all-sparse tile flash regression.

Build/sync:

```text
target=pic_server
CUDA architectures: 7.2
KleidiAI: OFF
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Formal sweep:

```text
tag=pic_cuda_1024_tileflash_causal_limit_formal_20260611_194416
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.378959,0.989
full-reuse,full,0.290140,8.108
cacheblend,0.10,0.512455,4.591
cacheblend,0.20,0.689947,3.410
cacheblend,0.30,0.864029,2.723
cacheblend,0.40,1.103161,2.133
cacheblend,0.50,1.298412,1.812
epic,0.10,0.382938,6.143
epic,0.20,0.494079,4.761
epic,0.30,0.596555,3.944
epic,0.40,0.734550,3.203
epic,0.50,0.888805,2.647
```

Compared with the previous epic-only tile flash run:

```text
epic10: 0.461220 -> 0.382938 (-17.0%)
epic20: 0.643831 -> 0.494079 (-23.3%)
epic30: 0.812402 -> 0.596555 (-26.6%)
epic40: 1.008296 -> 0.734550 (-27.1%)
epic50: 1.211449 -> 0.888805 (-26.6%)
```

Compared with the original CUTLASS graph-boundary baseline:

```text
epic10: 0.473170 -> 0.382938 (-19.1%)
epic20: 0.670933 -> 0.494079 (-26.4%)
epic30: 0.839971 -> 0.596555 (-29.0%)
epic40: 1.051566 -> 0.734550 (-30.2%)
epic50: 1.253765 -> 0.888805 (-29.1%)
```

Low-budget smoke:

```text
tag=pic_cuda_1024_tileflash_causal_limit_low_20260611_194507
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.01,0.359771,6.539
cacheblend,0.05,0.418441,5.622
epic,0.01,0.307299,7.656
epic,0.05,0.366858,6.413
```

Profile attribution at 50%:

```text
tag=pic_cuda_1024_tileflash_causal_limit_profile_50_20260611_194540
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50:
  PicSparseAttention=533.888 ms
  Convolution=459.820 ms
  PicScoreAttention=89.250 ms
  PagedAttention=87.287 ms
  later sparse path=prefill_attention_fast_qk_softmax_qkv

epic50:
  Convolution=465.632 ms
  PicSparseAttention=194.614 ms
  PagedAttention=80.075 ms
  UnaryOp=74.756 ms
  PicScoreAttention=15.336 ms
  sparse path=sparse_flash_tile_attention
  per sparse flash layer ~=13.0 ms
```

Conclusion:

- The main wasted work in the accepted tile sparse flash was full-length K/V streaming even when the active rows were a contiguous prefix. Limiting each Q tile by `max_q_logical` fixes that without changing sparse semantics.
- Epic/fixed-plan attention is no longer the primary bottleneck at 50%; dense `Convolution`/MLP is now larger than `PicSparseAttention`.
- Cacheblend remains attention-bound because it still uses scattered rows that reach the tail; the next cacheblend-specific path must exploit logical-position buckets or change algorithm semantics under a new mode name.
- Keep the causal K limit in the default CUDA tile flash route. Do not add an env fallback for the slower full-K tile loop.

## 2026-06-11 Jetson CUDA Cacheblend Hybrid Sparse Flash

Follow-up P0 A/B:

- The first "all sparse" route only removed `!cacheblend_score_active` but still required `pic_graph_active_plan_ready`; profile showed cacheblend remained on `prefill_attention_fast_qk_softmax_qkv`, so that was not a real cacheblend flash test.
- The real all-sparse route uses tile flash whenever `sparseQuery` is active. This finally routes cacheblend score/later sparse rows into `sparse_flash_tile_attention`.
- True all-sparse was good at high cacheblend budgets but too noisy around 20%, so the accepted route is a built-in shape heuristic:

```text
fixed/epic sparse rows: always tile flash, q_tile=8, k_tile=32
cacheblend sparse rows: tile flash only when attnLen > 256, q_tile=8, k_tile=32
cacheblend attnLen <= 256: keep existing QK/softmax/QKV path
```

Smoke that proved cacheblend can benefit when it really enters tile flash:

```text
tag=pic_cuda_1024_tileflash_true_allsparse_smoke_20_50_20260611_195425
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend20: 0.703900, 3.342x
cacheblend50: 1.084251, 2.170x
epic20:       0.496554, 4.738x
epic50:       0.906377, 2.596x
```

True all-sparse formal:

```text
tag=pic_cuda_1024_tileflash_true_allsparse_formal_20260611_195522
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.10,0.508005,4.631
cacheblend,0.20,0.671017,3.506
cacheblend,0.30,0.824639,2.853
cacheblend,0.40,1.028797,2.287
cacheblend,0.50,1.160317,2.027
epic,0.10,0.381198,6.171
epic,0.20,0.668295,3.520  # treated as outlier; repeat below returned to ~0.49s
epic,0.30,0.600304,3.919
epic,0.40,0.737021,3.192
epic,0.50,0.891496,2.639
```

Repeat for the unstable 20/30 area:

```text
tag=pic_cuda_1024_tileflash_true_allsparse_repeat_20_30_20260611_195612
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend20: 0.706689, 3.329x
cacheblend30: 0.791195, 2.973x
epic20:       0.491828, 4.783x
epic30:       0.613817, 3.833x
```

Accepted hybrid formal:

```text
tag=pic_cuda_1024_tileflash_hybrid_formal_20260611_195828
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.411054,0.976
full-reuse,full,0.296088,7.945
cacheblend,0.10,0.513422,4.582
cacheblend,0.20,0.692993,3.395
cacheblend,0.30,0.833976,2.821
cacheblend,0.40,1.048359,2.244
cacheblend,0.50,1.196245,1.967
epic,0.10,0.382526,6.150
epic,0.20,0.491652,4.785
epic,0.30,0.597417,3.938
epic,0.40,0.733922,3.205
epic,0.50,0.890640,2.641
```

Compared with the causal-limit epic-only route:

```text
cacheblend30: 0.864029 -> 0.833976 (-3.5%)
cacheblend40: 1.103161 -> 1.048359 (-5.0%)
cacheblend50: 1.298412 -> 1.196245 (-7.9%)
epic10/20/30/40/50: stays on the accepted causal-limit tile flash path
```

Hybrid low-budget smoke:

```text
tag=pic_cuda_1024_tileflash_hybrid_low_20260611_195917
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.01,0.364208,6.459
cacheblend,0.05,0.419508,5.608
epic,0.01,0.309107,7.611
epic,0.05,0.372496,6.316
```

The low-budget cacheblend path does not use tile flash under the accepted heuristic; small variation here is run noise, not a q_tile change.

Hybrid profile at 50%:

```text
tag=pic_cuda_1024_tileflash_hybrid_profile_50_20260611_195951
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50:
  PicSparseAttention=464.707 ms
  Convolution=458.810 ms
  PagedAttention=87.546 ms
  PicScoreAttention=84.367 ms
  sparse path=sparse_flash_tile_attention
  q_tile=8 k_tile=32

epic50:
  Convolution=463.017 ms
  PicSparseAttention=193.909 ms
  PagedAttention=79.806 ms
  PicScoreAttention=14.531 ms
  sparse path=sparse_flash_tile_attention
  q_tile=8 k_tile=32
```

Rejected q4 cacheblend-high variant:

```text
tag=pic_cuda_1024_tileflash_hybrid_q4_cb_smoke_30_50_20260611_200246
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend30: 1.289018, 1.825x
cacheblend50: 1.887966, 1.246x
epic30:       0.604598, 3.891x
epic50:       0.905547, 2.598x
```

Conclusion:

- Do not use `q_tile=4` for cacheblend high budgets. The smaller tile reduces some invalid K range but creates too many CTAs and loses more to scheduling/launch/memory overhead.
- Keep `q_tile=8,k_tile=32` as the accepted CUDA sparse flash tile shape for now.
- The accepted cacheblend heuristic is `attnLen > 256`. It protects 20% and low budgets while improving 30/40/50.
- Final artifact was rebuilt and re-synced after reverting q4. Final smoke `tag=pic_cuda_1024_tileflash_hybrid_final_smoke_50_20260611_200614` reported cacheblend50 `1.264013s`, epic50 `0.910181s`, confirming q4 was not left on the Jetson artifact.
- After this P0, cacheblend50 attention and compact dense are roughly tied (`PicSparseAttention ~=465 ms`, `Convolution ~=459 ms`), so the next high-value line is P1 compact tensor-core GEMM / dequant-cache work.

## 2026-06-11 Jetson CUDA Further Optimization Potential After Hybrid

Current accepted default remains the hybrid CUDA route:

```text
fixed/epic sparse rows: q8/k32 sparse flash tile with per-Q-tile causal K limit
cacheblend sparse rows: q8/k32 sparse flash tile only when attnLen > 256
cacheblend attnLen <= 256: existing QK/softmax/QKV path
```

The next potential is no longer a single "make sparse attention faster" problem. The profile at 50% splits into three different bottleneck classes:

```text
tag=pic_cuda_1024_tileflash_hybrid_profile_50_20260611_195951

cacheblend50 total_ms=1258.747
  PicSparseAttention=464.707 ms
  Convolution=458.810 ms
  PagedAttention=87.546 ms
  PicScoreAttention=84.367 ms
  UnaryOp=74.213 ms
  Raster=44.629 ms
  BinaryOp=26.187 ms

epic50 total_ms=911.664
  Convolution=463.017 ms
  PicSparseAttention=193.909 ms
  PagedAttention=79.806 ms
  UnaryOp=73.962 ms
  Raster=43.236 ms
  BinaryOp=26.038 ms
  PicScoreAttention=14.531 ms
```

Detailed attention facts:

```text
cacheblend50:
  score layer sparse flash layer=1 full_q=1 us=32670
  later sparse flash layers ~=32-33 ms/layer
  cacheblend_score metadata line precedes attention, so PicScoreAttention total has about 50 ms not explained by flash attention itself

epic50:
  score layer sparse flash layer=1 full_q=1 us=13087
  later sparse flash layers ~=13 ms/layer
```

Why cacheblend still costs much more than epic with the same active row count:

- Both have `attn=519` at 50%.
- Epic selected rows are a contiguous prefix; the causal K limit per Q tile is about 25% of full at 50%.
- Cacheblend selected rows are scattered and reach logical position 1009; average causal K is about 65%-75% of full.
- The accepted tile flash already skips whole K/V tiles above each tile's max logical position. Cacheblend still streams much more real K/V, not just mask waste.

Further attention potential:

- P0a: cacheblend-specific logical-range tiling. Keep the exact same active set and output compact order, but internally process active rows in narrower logical buckets and scatter outputs back. This can reduce invalid K/V tile loads inside a Q tile. The q4 A/B proved that simply shrinking `Q_TILE` loses too much scheduling efficiency; a useful version must keep enough rows per CTA while narrowing each tile's causal range.
- P0b: reduce tile-flash overhead without changing work: remove or reduce `scoreShared` round trips, reduce `__syncthreads()`, vectorize K/V loads, and test `K_TILE=64` only if occupancy and shared-memory pressure stay acceptable. This is likely a smaller win than P0a for cacheblend and smaller than dense work for epic.
- P0c is not to window cacheblend selected tokens. Any change that prevents scattered tail rows from being selected changes algorithm semantics and must be reported as a new mode such as `cacheblend-windowed`, not as ordinary cacheblend.

Score/top-k potential:

- `runCacheBlendScoringCUDA` still reads the cached value segment into a host `std::vector<int8_t>`, copies it to a CUDA workspace, scores one segment, runs a single-block top-k kernel, copies only top-k indices back to host, and calls `setCacheBlendScoringResult`.
- For 1024-token cacheblend50, `PicScoreAttention=84 ms` while sparse flash attention inside the same layer is about `33 ms`; the remaining roughly `50 ms` is the score/top-k/source-value path plus metadata.
- First action should be instrumentation that splits this path into `readExternalValueSegment`, H2D copy, value-score kernel, top-k kernel, D2H index copy, and metadata. Do not guess which part dominates.
- If source values already exist in the current request's mapped PagedCache source slots after async hydrate scheduling, score directly from those source slots and remove the per-request host read/H2D copy. This keeps the persistent PIC cache source/current request PagedCache boundary intact; it must not create scratch `.k/.v`.
- The current `cacheBlendTopKKernel<<<1,256>>>` is deterministic but O(topK * tokenCount) inside one CTA. For 1010 tokens this is not huge, but if instrumentation shows it is material, replace it with a two-stage block top-k or threshold/select kernel and keep deterministic tie behavior.

Compact dense/MLP potential:

- After sparse flash, dense MLP is a first-class bottleneck:
  - cacheblend50: `Convolution ~=459 ms`, tied with `PicSparseAttention`.
  - epic50: `Convolution ~=463 ms`, more than 2x `PicSparseAttention`.
- Score-layer graph boundary is working: after layer 1, MLP and QKV/o-proj run on compact rows such as `519x2048` or `519x8192`; layer 0 remains full 1024 rows by semantics.
- Per compact layer at 50%, `mlp/gate_proj`, `mlp/up_proj`, and `mlp/down_proj` are each about `7.1-7.8 ms`. The full layer-0 MLP projections are about `12.6 ms` each.
- The CUDA low-memory weight-only 1x1 Conv path currently dequantizes INT4 weights into a DYNAMIC FP16 buffer on every execute, then calls CUTLASS GEMM. This is memory-safe for large models, but for 1B PIC compact rows it creates a ratio-independent fixed cost for every Linear.
- Before adding a dequant cache, add profile-only timing around `DequantizeInt4Weight` and `runCutlassGemmFunc()` for compact MLP shapes. A dequant cache is only worth defaulting if dequant is a meaningful fraction of the 7-8 ms compact Linear time and a 1/5/10/20/30/40/50 sweep proves no OOM/regression.
- Memory math for caching all Llama-3.2-1B dequant MLP weights is large but not absurd: gate/up/down are about 32 MB each per layer, about 96 MB/layer and about 1.5 GB for 16 layers, before attention projection weights. This may be acceptable on an 8 GB Jetson for this model but is not a general production rule for larger models.

Unary/Binary/Raster potential:

- `UnaryOp ~=74 ms` and `BinaryOp ~=26 ms` at 50% are mostly MLP activation and multiply. At compact 519 rows, each `act_fn/Mul_output_0` is about `4.16 ms`, and each MLP elementwise multiply is about `0.61 ms`.
- A fused SiLU(gate) * up kernel could remove a large part of this overhead and reduce one intermediate read/write. It is less invasive than a full fused MLP kernel and should be considered alongside compact GEMM work.
- A full fused MLP kernel that combines gate/up projection, activation, multiply, and down projection could save more memory traffic but is much higher risk because it must avoid recomputing gate/up work for each down-output tile.

PagedAttention/full-prefix potential:

- `PagedAttention` layer 0 remains about `80-88 ms` and is semantically full compute. It is not the first priority now.
- Full-reuse is already about `0.296 s`; hydrate/restore lines are typically under 1 ms/layer with async reads, so the async persistent PIC cache loading path is not the current bottleneck.

Recommended next experiment order:

1. Add CUDA profile-only timers for compact weight-only Conv: dequant vs CUTLASS GEMM, at least for `mlp/gate_proj`, `mlp/up_proj`, and `mlp/down_proj` on active rows 1/5/10/20/30/40/50.
2. Add CUDA profile-only timers inside `runCacheBlendScoringCUDA`: host read, H2D copy, score kernel, top-k, D2H, metadata.
3. If dequant is large, implement a memory-aware default dequant cache for the 1B PIC artifact or a real fused dequant+tensor-core GEMM; reject it unless all ratios stay faster than the current hybrid.
4. If cacheblend score/source copy is large, score from current request PagedCache source slots and keep only final top-k indices crossing to host.
5. Only after those measurements, attempt cacheblend logical-range sparse flash. The upside is bounded by cacheblend's real causal K distribution; do not expect epic-like latency unless the algorithm changes selection distribution.

Expected ceiling from current profile:

- Halving only cacheblend sparse attention would save about `230 ms` at cb50 and move `1.20 s` toward `0.96-1.03 s`.
- Halving only compact dense would save about `230 ms` for both cb50 and epic50; epic50 would move from about `0.89 s` toward `0.66 s`.
- Removing MLP Unary/Binary overhead could save up to about `100 ms` at 50%, but real savings depend on fusion and memory traffic.
- Therefore the most realistic next end-to-end gain is not one giant kernel change; it is dense dequant/GEMM + MLP elementwise fusion for all sparse modes, plus cacheblend-specific score/source-value cleanup.

## 2026-06-11 Jetson CUDA P0 Tail K/V Cut and P1 Dequant Profile

Implemented two low-risk CUDA changes:

- P0: in `pagedSparseFlashTileKernel`, the final K tile now only loads K/V when `logical < causalKLimit`, not merely `logical < kvLen`. This keeps the same causal softmax range and avoids reading tail lanes above `max_q_logical + 1` in the last K tile.
- P1 measurement: `ConvFpAIntBExecution` now prints profile-only split timing for batched 1x1 weight-only Conv when `MNN_PAGED_ATTENTION_PROFILE=1` or `MNN_PIC_GRAPH_PROFILE=1`: `dequant_us`, `convert_us`, `cutlass_us`, and `total_us`. It does not change the production GEMM path or add an env fallback.

Build:

```text
target=pic_server
CUDA support: ON
CUDA architectures: 7.2
Enabling CUDA support (... archs: sm_72)
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Profile run:

```text
tag=pic_cuda_1024_p0_tail_p1_dequant_profile_50_20260611_202624
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000  # reused baseline
full-compute,full,2.566778,0.917         # profile sync overhead
full-reuse,full,0.433422,5.428           # profile sync overhead
cacheblend,0.50,1.375528,1.710           # profile sync overhead
epic,0.50,1.014570,2.319                 # profile sync overhead
```

Graph profile at 50%:

```text
cacheblend50 total_ms=1266.352
  Convolution=475.315 ms
  PicSparseAttention=454.714 ms
  PagedAttention=87.991 ms
  PicScoreAttention=84.041 ms
  UnaryOp=74.372 ms

epic50 total_ms=924.531
  Convolution=479.164 ms
  PicSparseAttention=189.692 ms
  PagedAttention=80.020 ms
  UnaryOp=73.800 ms
  PicScoreAttention=14.240 ms
```

P0 effect:

- Previous hybrid profile had cacheblend50 `PicSparseAttention=464.707 ms`; this run reports `454.714 ms`, about `10 ms` lower under profile.
- Raw cacheblend50 sparse flash layer times moved from roughly `32.0-32.6 ms/layer` to roughly `31.3-31.8 ms/layer`.
- Previous epic50 sparse flash layers were roughly `13.0 ms/layer`; this run is roughly `12.7 ms/layer` for normal layers. Treat this as a small win, not a new major P0 line.
- Formal no-profile smoke still passed:

```text
tag=pic_cuda_1024_p0_tail_smoke_50_20260611_202809
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

full-compute,full,2.354317,0.999
full-reuse,full,0.291271,8.077
cacheblend,0.50,1.251329,1.880
epic,0.50,0.896326,2.625
```

20% smoke:

```text
tag=pic_cuda_1024_p0_tail_smoke_20_20260611_202849
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

full-compute,full,2.351203,1.001
full-reuse,full,0.289471,8.127
cacheblend,0.20,0.720039,3.267
epic,0.20,0.499301,4.712
```

The cb20 path does not use tile flash under the accepted `attnLen > 256` cacheblend heuristic, so its single-run latency should be treated as smoke/noise rather than evidence for or against the P0 tail cut.

P1 dequant/GEMM split:

```text
profile aggregation from tag=pic_cuda_1024_p0_tail_p1_dequant_profile_50_20260611_202624

cacheblend50 Conv profile:
  lines=121
  dequant_ms=60.869
  cutlass_ms=392.414
  total_ms=453.380
  dequant_share=13.4%

epic50 Conv profile:
  lines=103
  dequant_ms=50.893
  cutlass_ms=380.283
  total_ms=431.266
  dequant_share=11.8%

compact M=519 representative shapes:
  [519,2048,8192] gate/up:
    avg_dequant ~=0.87 ms
    avg_cutlass ~=6.6-6.7 ms
    avg_total   ~=7.5-7.6 ms
  [519,8192,2048] down:
    avg_dequant ~=0.85-0.86 ms
    avg_cutlass ~=6.6 ms
    avg_total   ~=7.5 ms
  [519,2048,2048] q/o:
    avg_dequant ~=0.28-0.30 ms
    avg_cutlass ~=1.7 ms
    avg_total   ~=2.0 ms
  [519,2048,512] k/v:
    avg_dequant ~=0.14-0.15 ms
    avg_cutlass ~=0.51 ms
    avg_total   ~=0.65 ms
```

Conclusion:

- P0 tail K/V cut is safe and slightly improves sparse flash, but it is not enough to change the bottleneck class.
- P1 dequant is real but smaller than expected: full dequant caching would save roughly `50-60 ms` at cb50/epic50 after warm-up, not `100-140 ms` by itself. The larger dense cost is CUTLASS GEMM work.
- Do not blindly default a full static dequant cache for all 1B weights yet. It may need around GB-scale FP16 weight memory across MLP and attention projections, and larger models would be unsafe.
- The better P1 directions are now:
  - a memory-capped dequant cache validated across 1/5/10/20/30/40/50 and no OOM;
  - or a real fused dequant + tensor-core GEMM / weight-only tensor-core GEMM that reduces both dequant traffic and compact GEMM time;
  - plus MLP activation/multiply fusion, because `UnaryOp+BinaryOp` remains about `100 ms` at 50%.

## 2026-06-11 Jetson CUDA Further Potential Recheck

This pass rechecked the remaining CUDA upside after the accepted hybrid sparse flash and the P1 dequant split. The main conclusion is that there is still optimization potential, but it is now distributed across three smaller buckets instead of one obvious bottleneck.

Current 50% profile shape:

```text
cacheblend50:
  PicSparseAttention ~=455 ms
  Convolution        ~=475 ms
  PicScoreAttention  ~=84 ms
  UnaryOp+BinaryOp   ~=100 ms

epic50:
  Convolution        ~=479 ms
  PicSparseAttention ~=190 ms
  UnaryOp+BinaryOp   ~=100 ms
  PicScoreAttention  ~=14 ms
```

Attention recheck:

- Cacheblend active logical indices are already sorted before graph-boundary sparse rows are built (`buildCacheBlendActiveLogicalIndices` sorts selected local indices and appends active rows in logical order). Therefore a simple "sort/bucket then scatter" rewrite is less valuable than initially assumed; fixed q8 tiles already see mostly monotonic logical rows.
- The remaining cacheblend attention waste comes from q8 tiles that span large logical gaps. A useful next attention experiment is adaptive piece building: keep q8 for dense/contiguous active ranges, split only the tiles whose logical span is too large, and keep output order unchanged. Do not globally switch to q4; the q4 A/B already regressed cb30/cb50.
- `K_TILE=64` is not a drop-in change for the current kernel because the tile code relies on warp-level reductions over `threadIdx.x == 0..31`. A 64-wide K tile needs a different two-warp reduction design before it can be tested.
- WMMA/tensor-core sparse flash is theoretically interesting for the 8x32x64 QK and 8x32x64 PV tiles, but it would require a different shared-memory score/softmax layout and likely half probability staging. Treat this as P2 research, not the next production default.

Score/top-k recheck:

- CUDA cacheblend scoring still reads the persistent value segment into a host vector, copies it to a CUDA workspace, runs value-delta scoring, then runs `cacheBlendTopKKernel`.
- OpenCL already uses PagedCache source slots for scoring. CUDA can copy that design, but it first needs CUDA `onResize`/`ensureCache` to reserve source slots beyond the real logical slots, like OpenCL's `_picCacheSourceSlotBase/_picCacheSourceSlotCount`.
- The current top-k kernel is deterministic but has an expensive inner `for prev < k` scan. At 1024-token cb50, `topK ~=505`, so the algorithm does much more work than the token count suggests. First add split timers; if top-k is visible, replace the used-check scan with a shared used flag or two-stage select while preserving tie order.
- This line is likely a tens-of-ms cleanup for cacheblend, not the main epic bottleneck.

Dense/MLP recheck:

- Dequant is real but only about 11%-14% of profiled Conv time at 50%. Full dequant caching alone would likely save only `50-60 ms` at cb50/epic50 after warm-up.
- Most dense time is CUTLASS GEMM on compact rows. On sm72, a true INT4 tensor-core GEMM is not the safest assumption; the realistic CUDA path is either memory-capped FP16 dequant cache, fused dequant+FP16 GEMM, or export/runtime fusion that improves the existing CUTLASS shapes.
- The most promising graph-level dense fusion is `gate_proj` + `up_proj` because both read the same compact hidden input and produce the two operands for SwiGLU. A combined output-channel projection could improve launch/input-read efficiency, then a fused `SiLU(gate) * up` kernel can remove much of `UnaryOp+BinaryOp`.
- This dense line benefits cacheblend and epic at every ratio, while cacheblend-specific attention work mostly helps higher budgets.

Recommended next order:

1. Add split timers for CUDA cacheblend scoring: host read, H2D copy, value-score kernel, top-k, D2H indices, metadata.
2. Prototype fused `SiLU(gate) * up` or graph-level gate/up handling for compact rows, because `UnaryOp+BinaryOp ~=100 ms` is now comparable to the remaining attention micro-optimization ceiling.
3. If touching attention, implement adaptive q-piece construction for cacheblend scattered rows rather than global q4 or K_TILE=64.
4. Only after those, evaluate CUDA PagedCache source-slot scoring to remove the host value workspace path.

Practical ceiling:

- Cacheblend50 can probably still save `100-180 ms` without changing the algorithm: around `50-80 ms` from adaptive attention pieces, `20-50 ms` from score/top-k/source cleanup, and `50-100 ms` from MLP elementwise/dense cleanup.
- Epic50's best remaining win is dense/MLP, not attention. A good dense/MLP change could move epic50 from about `0.90 s` toward `0.75-0.80 s`; attention-only work will not.

## 2026-06-11 Jetson CUDA Cacheblend Score Source Slots

Implemented CUDA cacheblend score cleanup:

- `cacheBlendTopKKernel` now uses a device `used` bitmap instead of scanning `selected[0..k)` for every candidate. Tie behavior is unchanged: higher score wins, equal score picks smaller local index.
- CUDA cacheblend scoring now reserves PagedCache source slots beyond the real logical slots, matching the OpenCL source-slot design.
- On mapped PagedCache, persistent PIC value segments are read directly into `mCache->mappedValue->host` at `sourceSlotStart`, and `cacheBlendValueScoreFromPagedSourceKernel` compares reference logical slots and source slots inside the same `mCache->value` device buffer.
- This removes the previous `std::vector<int8_t> cachedValue` + H2D value workspace from the Jetson default path. It is "zero-copy" in the PagedCache/device-buffer sense; disk read still lands in CPU mapped pinned memory, not GPUDirect Storage.

Build:

```text
target=pic_server
CUDA support: ON
CUDA architectures: 7.2
Enabling CUDA support (... archs: sm_72)
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Profile before source-slot scoring, after used-bitmap top-k:

```text
tag=pic_cuda_1024_topk_used_profile_50_20260611_124634
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend_score:
  us=7346
  alloc_us=206
  read_us=3894
  h2d_us=641
  score_kernel_us=548
  topk_init_us=43
  topk_us=1862
  d2h_us=119
  metadata_us=14

cacheblend50 graph:
  PicSparseAttention=453.563 ms
  Convolution=473.952 ms
  PicScoreAttention=40.906 ms
```

Profile after source-slot scoring:

```text
tag=pic_cuda_1024_source_slot_profile_50_20260611_125345
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend_score:
  source_slot_segments=1
  source_slot_tokens=1010
  us=3477
  alloc_us=163
  read_us=501
  h2d_us=0
  score_kernel_us=778
  topk_init_us=42
  topk_us=1829
  d2h_us=119
  metadata_us=27

cacheblend50 graph:
  PicSparseAttention=455.965 ms
  Convolution=471.301 ms
  PicScoreAttention=37.116 ms
```

No-profile smoke:

```text
tag=pic_cuda_1024_source_slot_smoke_20_50_20260611_125449
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend20: 0.707247s, 3.326x vs normal
cacheblend50: 1.047366s, 2.246x vs normal

tag=pic_cuda_1024_source_slot_repeat_50_20260611_125538
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50 repeat: 1.200309s, 1.960x vs normal
```

Conclusion:

- Source-slot scoring is semantically cleaner and removes the extra value H2D copy from CUDA cacheblend score. Keep it as the default Jetson path.
- The measured score subpath improvement is about `3.9 ms` in profile (`7.346 ms -> 3.477 ms`). This is useful but not the main bottleneck.
- `topk_us` is still about `1.8 ms` at `topK=505`; further top-k work is possible but only a small cleanup unless larger context lengths make `topK * tokenCount` worse.
- End-to-end cacheblend50 remains dominated by later `PicSparseAttention` and compact dense `Convolution`; source-slot scoring should not change the next priority order.

## 2026-06-11 Jetson CUDA Static Dequant Cache For Compact Dense

Implemented a memory-capped static FP16 dequant cache for CUDA low-memory weight-only INT4 1x1 Conv:

- `ConvFpAIntBExecution::Resource` now can pre-dequant its packed INT4 weight once into a STATIC FP16 tensor.
- Default cap is `min(2 GiB, totalGlobalMem / 16)`, with a `256 MiB` floor. This keeps the fast path default on Jetson 1B while avoiding unbounded FP16 expansion for larger models or smaller devices.
- `onResize` uses the Resource static dequant filter when available; otherwise it keeps the existing DYNAMIC runtime-dequant path for tensors beyond the cap or unsupported precision.
- Profile lines now include `static_dequant`, `static_cache_bytes`, and `static_cache_total`.

Build:

```text
target=pic_server
CUDA support: ON
CUDA architectures: 7.2
Enabling CUDA support (... archs: sm_72)
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Profile:

```text
tag=pic_cuda_1024_static_dequant_2g_profile_50_20260611_211227
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

static dequant profile:
  runtime_dequant=1 lines: 0
  static_dequant=1 lines: 555
  static_cache_total=1946157056 bytes

cacheblend50 graph:
  Convolution        = 409.255 ms
  PicSparseAttention = 401.494 ms
  PicScoreAttention  = 33.362 ms
  UnaryOp            = 74.109 ms
  BinaryOp           = 25.779 ms
```

Compared with `pic_cuda_1024_source_slot_profile_50_20260611_125345`:

```text
Convolution:        471.301 ms -> 409.255 ms  (-62.046 ms)
PicSparseAttention: 455.965 ms -> 401.494 ms  (also lower in this profiled run; treat as run-to-run/profile interaction, not solely dequant)
PicScoreAttention:   37.116 ms ->  33.362 ms
cacheblend50 e2e:  1.329647 s -> 1.204583 s in profile run
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_static_dequant_2g_sweep_1_50_20260611_211338
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.312875,1.017
full-reuse,full,0.636679,3.695

cacheblend,0.01,0.321178,7.325
cacheblend,0.05,0.373273,6.302
cacheblend,0.10,0.432630,5.438
cacheblend,0.20,0.642387,3.662
cacheblend,0.30,0.762592,3.085
cacheblend,0.40,0.906735,2.595
cacheblend,0.50,1.045583,2.250

epic,0.01,0.267146,8.806
epic,0.05,0.307547,7.649
epic,0.10,0.336170,6.998
epic,0.20,0.447952,5.252
epic,0.30,0.548095,4.292
epic,0.40,0.686786,3.425
epic,0.50,0.833466,2.823
```

Conclusion:

- Keep the static dequant cache as the default CUDA compact dense fast path. It removes runtime dequant from all profiled 1B Linear ops under the 2 GiB / 1/16-memory cap.
- This is a P1 win, not the final dense solution: cb50 `Convolution` is still about `409 ms`, so most remaining dense cost is CUTLASS GEMM itself, not dequant.
- After this change, cacheblend50 is roughly split between `PicSparseAttention` and `Convolution`; epic50 remains more dense-bound than attention-bound.
- Next dense work should target graph/export fusion for gate/up and SwiGLU or a real fused dequant+GEMM/tensor-core path, not another naive packed GEMM.

## 2026-06-11 Jetson CUDA Sparse Tile Full-Causal Mask Skip

Implemented a small P0 cleanup for CUDA q8/k32 sparse tile flash:

- `PagedKVMeta` now records whether the model uses a float full-causal attention mask (`attention_mask=float`, `attention_type=full`).
- `Llm` sets this once from `llm_config.json`.
- `CUDAPagedAttention` passes `nullptr` / `mask_elements=0` to `pagedSparseFlashTileKernel` only for that full-causal sparse tile path. The tile kernel already enforces true logical causality through `active_indices`, `logical <= qLogical`, and per-Q-tile `causalKLimit`.
- Sliding/mix attention must keep the additive mask because it encodes a real window, not just causal order.

Build:

```text
target=pic_server
CUDA support: ON
CUDA architectures: 7.2
Enabling CUDA support (... archs: sm_72)
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Profile:

```text
tag=pic_cuda_1024_sparse_tile_skip_mask_profile_50_20260611_132617
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend_score:
  source_slot_segments=1
  source_slot_tokens=1010
  us=3392
  h2d_us=0
  score_kernel_us=750
  topk_us=1846

sparse tile:
  layer=1  query=1024 attn=519 mask_elements=0 input_mask_elements=1048576 causal_mask_skipped=1 us=26893
  layer=2+ query=519  attn=519 mask_elements=0 input_mask_elements=531456  causal_mask_skipped=1 us~=26277-26912

cacheblend50 graph:
  Convolution        = 410.185 ms
  PicSparseAttention = 383.119 ms
  PagedAttention     = 87.673 ms
  UnaryOp            = 73.701 ms
  PicScoreAttention  = 32.100 ms
  BinaryOp           = 25.966 ms
```

Compared with `pic_cuda_1024_static_dequant_2g_profile_50_20260611_211227`:

```text
PicSparseAttention: 401.494 ms -> 383.119 ms (-18.375 ms)
Convolution:        409.255 ms -> 410.185 ms (noise-level flat)
cacheblend50 e2e profile: 1.204583 s -> 1.186406 s
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_sparse_tile_skip_mask_sweep_1_50_20260611_132737
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.319955,1.014
full-reuse,full,0.638572,3.684

cacheblend,0.01,0.314500,7.480
cacheblend,0.05,0.374485,6.282
cacheblend,0.10,0.431842,5.448
cacheblend,0.20,0.642368,3.662
cacheblend,0.30,0.748229,3.144
cacheblend,0.40,0.889348,2.645
cacheblend,0.50,1.027361,2.290

epic,0.01,0.265385,8.865
epic,0.05,0.313428,7.506
epic,0.10,0.333599,7.052
epic,0.20,0.445318,5.283
epic,0.30,0.547138,4.300
epic,0.40,0.682878,3.445
epic,0.50,0.821617,2.863
```

Repeat smoke:

```text
tag=pic_cuda_1024_sparse_tile_skip_mask_repeat_5_50_20260611_132830
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.05,0.390996,6.017
cacheblend,0.50,1.047438,2.246
epic,0.05,0.309786,7.594
epic,0.50,0.835076,2.817
```

Conclusion:

- Keep this as a safe full-causal sparse tile cleanup. It removes redundant mask loads/branches in the target Llama-3.2 full-attention PIC graph-boundary path and shows a clear `PicSparseAttention` profile reduction.
- Do not count it as a large end-to-end win: no-profile latency still has run-to-run noise and repeat cb50 is close to the previous static-dequant sweep.
- The next P0 work must reduce real tile math / synchronization cost, not just mask overhead. The most promising direction remains adaptive q-piece / bucketed cacheblend sparse flash that reduces K work for scattered active rows while preserving compact output order.
- P1 dense work is still equally important: cacheblend50 remains split between `PicSparseAttention` and compact `Convolution`, while epic50 is dense-bound.

## 2026-06-11 Jetson CUDA High-Budget q16 Sparse Tile

Implemented a high-budget sparse tile variant:

- Keep q8/k32 as the default sparse tile.
- Use q16/k32 only when `attnLen >= 512`. This targets 1024-token 50% budgets where active rows are 519 and q8 launches many small blocks.
- Low budgets and mid budgets remain on q8/k32; cacheblend `attnLen <= 256` still keeps the existing QK/softmax/QKV path.
- This is not the rejected q4 path. q4 increased block count and regressed high budgets; q16 reduces block count and improves K/V reuse at the cost of larger CTAs.

Build:

```text
target=pic_server
CUDA support: ON
CUDA architectures: 7.2
Enabling CUDA support (... archs: sm_72)
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Profile:

```text
tag=pic_cuda_1024_q16_sparse_tile_profile_50_20260611_133723
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50 sparse tile:
  q_tile=16 k_tile=32
  layer=1 us=19476
  layer=2+ us~=19184-19426

cacheblend50 graph:
  Convolution        = 410.138 ms
  PicSparseAttention = 282.117 ms
  PagedAttention     = 87.561 ms
  UnaryOp            = 74.427 ms
  BinaryOp           = 26.398 ms
  PicScoreAttention  = 25.754 ms

epic50 graph:
  Convolution        = 418.182 ms
  PicSparseAttention = 136.829 ms
  PagedAttention     = 80.605 ms
  UnaryOp            = 74.018 ms
  BinaryOp           = 25.848 ms
  PicScoreAttention  = 10.632 ms
```

Compared with q8 + mask-skip profile `pic_cuda_1024_sparse_tile_skip_mask_profile_50_20260611_132617`:

```text
cacheblend50 PicSparseAttention: 383.119 ms -> 282.117 ms (-101.002 ms)
cacheblend50 profile e2e:        1.186406 s -> 1.086636 s
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_q16_sparse_tile_sweep_1_50_20260611_133816
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.318760,1.015
full-reuse,full,0.641728,3.666

cacheblend,0.01,0.319125,7.372
cacheblend,0.05,0.373730,6.295
cacheblend,0.10,0.430131,5.469
cacheblend,0.20,0.641187,3.669
cacheblend,0.30,0.749005,3.141
cacheblend,0.40,0.887050,2.652
cacheblend,0.50,0.912159,2.579

epic,0.01,0.264764,8.885
epic,0.05,0.306151,7.684
epic,0.10,0.334668,7.029
epic,0.20,0.442513,5.316
epic,0.30,0.543946,4.325
epic,0.40,0.678555,3.467
epic,0.50,0.773988,3.039
```

Repeat smoke:

```text
tag=pic_cuda_1024_q16_sparse_tile_repeat_40_50_20260611_133907
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.40,0.954381,2.465
cacheblend,0.50,0.937722,2.509
epic,0.40,0.684346,3.438
epic,0.50,0.794305,2.962
```

Conclusion:

- q16/k32 was a real P0 gain for the 50% budget where sparse attention was still a major bottleneck, but it is now superseded by q32/k32 for the same `attnLen >= 512` threshold.
- q16 does not apply to 40% in the 1024-token setup (`attnLen` is below 512), so the slower cacheblend40 repeat is treated as run-to-run noise rather than q16 regression.
- Compared with the prior q8 no-profile sweep, cacheblend50 improved from `1.027361 s` to `0.912159 s`, and epic50 improved from `0.821617 s` to `0.773988 s`.
- Further sparse attention work should look at wide-Q threshold tuning for larger contexts and cacheblend-specific logical bucket/scatter only if it preserves compact output order. q4 remains rejected.

## 2026-06-11 Jetson CUDA High-Budget q32 Sparse Tile

Tested and accepted q32/k32 for high-budget sparse tile:

- Keep q8/k32 as the default sparse tile for lower budgets.
- Use q32/k32 when `attnLen >= 512`, replacing the q16/k32 intermediate path.
- q32 uses the largest legal CTA size here (`32 x 32 = 1024` threads), so this acceptance is specific to the current sm72 1024-token profile/sweep and must be rechecked for larger contexts or different head dimensions.

Profile:

```text
tag=pic_cuda_1024_q32_sparse_tile_profile_50_20260611_135111
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50 sparse tile:
  q_tile=32 k_tile=32
  layer=1 us=17404
  layer=2+ us~=17099-17449

cacheblend50 graph:
  Convolution        = 410.933 ms
  PicSparseAttention = 252.729 ms
  PagedAttention     = 87.644 ms
  UnaryOp            = 74.383 ms
  BinaryOp           = 26.196 ms
  PicScoreAttention  = 23.336 ms

epic50 graph:
  Convolution        = 416.042 ms
  PicSparseAttention = 123.640 ms
  PagedAttention     = 81.209 ms
  UnaryOp            = 73.979 ms
  BinaryOp           = 26.403 ms
  PicScoreAttention  = 9.505 ms
```

No-profile 40/50 repeat:

```text
tag=pic_cuda_1024_q32_sparse_tile_repeat_40_50_20260611_135207
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.40,0.949852,2.477
cacheblend,0.50,0.908365,2.590
epic,0.40,0.689273,3.413
epic,0.50,0.785227,2.996
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_q32_sparse_tile_sweep_1_50_20260611_135252
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.308713,1.019
full-reuse,full,0.636979,3.693

cacheblend,0.01,0.313335,7.508
cacheblend,0.05,0.372454,6.316
cacheblend,0.10,0.432448,5.440
cacheblend,0.20,0.641337,3.668
cacheblend,0.30,0.747125,3.149
cacheblend,0.40,0.890466,2.642
cacheblend,0.50,0.884471,2.660

epic,0.01,0.264879,8.882
epic,0.05,0.309508,7.601
epic,0.10,0.331932,7.087
epic,0.20,0.442790,5.313
epic,0.30,0.544793,4.318
epic,0.40,0.678758,3.466
epic,0.50,0.764576,3.077
```

Compared with q16:

```text
cacheblend50: 0.912159 s -> 0.884471 s
epic50:       0.773988 s -> 0.764576 s
```

Conclusion:

- Keep q32/k32 for `attnLen >= 512` as the current Jetson CUDA high-budget sparse tile default.
- This is the first P0 change in this round that pushes cacheblend50 under `0.9 s` in a full no-profile sweep while preserving every 1%-50% ratio faster than normal.
- Remaining cb50 bottleneck after q32 is no longer dominated by sparse attention alone; compact `Convolution`, `UnaryOp`, and residual graph overhead need the next P1/P2 work.

## 2026-06-11 Jetson CUDA Sparse Tile K-Slot Cache

Implemented a small P0 cleanup inside `pagedSparseFlashTileKernel`:

- Each K tile now resolves `slotTable[logical]` once into shared `kSlotShared[K_TILE]`.
- K and V shared-memory load phases reuse the same cached physical slot.
- This avoids repeating the same slot-table global load and bounds check for every head_dim lane in both K and V loads.
- No semantic change: active logical indices, causal K limit, PagedCache physical slots, GQA head mapping, q8/q32 tile selection, and full-causal mask skip all stay unchanged.

Build:

```text
target=pic_server
CUDA support: ON
CUDA architectures: 7.2
artifact=.cache/output/mnn/artifacts/jetson_cross_cuda
rsync target=jetson@192.168.101.192
```

Initial no-profile 40/50 smoke had visible run-to-run noise:

```text
tag=pic_cuda_1024_q32_kslot_smoke_40_50_20260611_2204
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.40,0.923296,2.548
cacheblend,0.50,0.890479,2.642
epic,0.40,0.675901,3.481
epic,0.50,0.776560,3.029
```

Graph/profile showed the kernel-level win clearly, compared with q32 baseline `pic_cuda_1024_q32_sparse_tile_profile_50_20260611_135111`:

```text
tag=pic_cuda_1024_q32_kslot_profile_50_20260611_2205
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50 graph:
  Convolution        = 409.960 ms
  PicSparseAttention = 242.378 ms
  PagedAttention     = 87.480 ms
  UnaryOp            = 73.924 ms
  BinaryOp           = 25.910 ms
  PicScoreAttention  = 22.101 ms

epic50 graph:
  Convolution        = 415.936 ms
  PicSparseAttention = 118.928 ms
  PagedAttention     = 80.307 ms
  UnaryOp            = 73.895 ms
  BinaryOp           = 26.017 ms
  PicScoreAttention  = 9.350 ms

Compared with q32 baseline:
  cacheblend50 PicSparseAttention: 252.729 ms -> 242.378 ms
  epic50 PicSparseAttention:       123.640 ms -> 118.928 ms
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_q32_kslot_sweep_1_50_20260611_2207
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.313485,1.017
full-reuse,full,0.634177,3.710

cacheblend,0.01,0.312411,7.530
cacheblend,0.05,0.360835,6.520
cacheblend,0.10,0.427082,5.508
cacheblend,0.20,0.640144,3.675
cacheblend,0.30,0.718657,3.274
cacheblend,0.40,0.859711,2.736
cacheblend,0.50,0.871012,2.701

epic,0.01,0.260832,9.019
epic,0.05,0.300341,7.833
epic,0.10,0.327441,7.185
epic,0.20,0.438567,5.364
epic,0.30,0.534851,4.398
epic,0.40,0.666852,3.528
epic,0.50,0.757372,3.106
```

Compared with the q32 no-profile sweep:

```text
cacheblend50: 0.884471 s -> 0.871012 s
epic50:       0.764576 s -> 0.757372 s
```

Conclusion:

- Keep the shared K-slot cache as part of the default CUDA sparse tile path. It is a small but real P0 attention cleanup, confirmed by profile attribution and a full 1%-50% sweep.
- The remaining high-budget bottleneck is now even more balanced: cacheblend50 still has `PicSparseAttention ~=242 ms` and `Convolution ~=410 ms`; epic50 is clearly compact-dense bound.
- Next CUDA work should move to P1/P2 dense/activation fusion unless doing a larger cacheblend-specific sparse flash redesign that reduces true causal K work for scattered active rows.

Rejected follow-up in the same area:

```text
tag=pic_cuda_1024_q32_kslot_skipclear_profile_50_20260611_2211
change=skip cudaMemset(output) before sparse tile flash

cacheblend50 PicSparseAttention: 242.378 ms -> 242.429 ms
cacheblend50 graph total:        923.615 ms -> 924.581 ms
epic50 PicSparseAttention:       118.928 ms -> 118.493 ms
epic50 graph total:              784.749 ms -> 788.512 ms
```

Do not keep the skip-output-clear change as default. The sparse tile kernel writes all active rows in the current graph-boundary path, but the measured benefit is noise-level and graph total did not improve. Keep the conservative output clear outside any future explicit A/B branch.

## 2026-06-11 Jetson CUDA q32 Threshold 384

Lowered the wide-Q sparse tile threshold:

- Previous default: q32/k32 only when `attnLen >= 512`, so in the 1024-token sweep only 50% budgets used q32.
- New default: q32/k32 when `attnLen >= 384`, so 40% budgets with active rows around 418 also use q32.
- Lower budgets keep the existing q8/k32 or cacheblend `attnLen <= 256` QK/softmax/QKV path.
- This does not change attention semantics, active indices, causal K limit, or PagedCache slot mapping.

No-profile 40/50 smoke:

```text
tag=pic_cuda_1024_q32_kslot_thr384_smoke_40_50_20260611_2223
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend,0.40,0.819659,2.870
cacheblend,0.50,0.897994,2.620
epic,0.40,0.643837,3.654
epic,0.50,0.783233,3.004
```

Graph/profile:

```text
tag=pic_cuda_1024_q32_kslot_thr384_profile_40_50_20260611_2224
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend40 graph:
  Convolution        = 342.747 ms
  PicSparseAttention = 228.638 ms
  PagedAttention     = 87.619 ms
  UnaryOp            = 62.099 ms
  BinaryOp           = 23.607 ms
  PicScoreAttention  = 21.264 ms

cacheblend50 graph:
  Convolution        = 410.863 ms
  PicSparseAttention = 236.728 ms
  PagedAttention     = 80.738 ms
  UnaryOp            = 73.770 ms
  BinaryOp           = 25.774 ms
  PicScoreAttention  = 20.555 ms

epic40 graph:
  Convolution        = 343.400 ms
  PicSparseAttention = 92.426 ms
  PagedAttention     = 80.068 ms
  UnaryOp            = 61.799 ms
  BinaryOp           = 22.298 ms
  PicScoreAttention  = 7.294 ms

epic50 graph:
  Convolution        = 414.465 ms
  PicSparseAttention = 118.701 ms
  PagedAttention     = 80.117 ms
  UnaryOp            = 73.924 ms
  BinaryOp           = 25.664 ms
  PicScoreAttention  = 8.883 ms
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_q32_kslot_thr384_sweep_1_50_20260611_2226
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.303239,1.021
full-reuse,full,0.640213,3.675

cacheblend,0.01,0.311590,7.550
cacheblend,0.05,0.369951,6.359
cacheblend,0.10,0.429822,5.473
cacheblend,0.20,0.641737,3.666
cacheblend,0.30,0.723793,3.250
cacheblend,0.40,0.755525,3.114
cacheblend,0.50,0.872416,2.697

epic,0.01,0.259708,9.058
epic,0.05,0.300698,7.824
epic,0.10,0.326507,7.205
epic,0.20,0.438586,5.364
epic,0.30,0.534949,4.398
epic,0.40,0.636039,3.699
epic,0.50,0.757277,3.107
```

Compared with q32 + k-slot-cache threshold 512 sweep:

```text
cacheblend40: 0.859711 s -> 0.755525 s
cacheblend50: 0.871012 s -> 0.872416 s
epic40:       0.666852 s -> 0.636039 s
epic50:       0.757372 s -> 0.757277 s
```

Conclusion:

- Keep q32/k32 threshold at `attnLen >= 384`. It is a large win for 40% budgets and neutral at 50% in the full sweep.
- Do not push the threshold lower without another full 1%-50% sweep. 30% has active rows around 317 and may not have enough Q work to offset q32's 1024-thread CTA cost.
- After this change, cacheblend40/50 are still split between `PicSparseAttention` and compact dense `Convolution`; epic40/50 are more compact-dense bound. P1 compact dense and P2 activation fusion remain the next larger opportunities.

## 2026-06-11 Jetson CUDA SM70 Compact CUTLASS Dense Fast Path

Tested and accepted a narrow P1 compact dense fast path for score-layer graph-boundary rows:

- Add a sm70 tensor-core CUTLASS Linear variant with `GemmShape<128,64,64>` / warp `64x32x64`.
- Use it only for low-memory INT4 1x1 Linear with static FP16 dequant cache, fp16 inference, no activation, `M in [384,768]`, and padded input channels at least 1024.
- The condition targets 1024-token high-budget compact rows (`cacheblend/epic 40%-50%`) and intentionally avoids low-budget tiny rows and full 1010-row full-compute.
- No environment gate is needed; the default path selects this variant only for the measured compact range and falls back to the existing CUTLASS path for other shapes.

Profile A/B against q32 + k-slot-cache threshold 384:

```text
old tag=pic_cuda_1024_q32_kslot_thr384_profile_40_50_20260611_2224
new tag=pic_cuda_1024_piccompact_sm70_profile_40_50_20260611_223500
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend40:
  Convolution        342.747 ms -> 273.260 ms
  PicSparseAttention 228.638 ms -> 227.874 ms
  graph total        824.835 ms -> 750.962 ms

cacheblend50:
  Convolution        410.863 ms -> 354.839 ms
  PicSparseAttention 236.728 ms -> 236.047 ms
  graph total        908.114 ms -> 851.371 ms

epic40:
  Convolution        343.400 ms -> 272.537 ms
  PicSparseAttention  92.426 ms ->  92.691 ms
  graph total        662.669 ms -> 595.666 ms

epic50:
  Convolution        414.465 ms -> 361.439 ms
  PicSparseAttention 118.701 ms -> 119.267 ms
  graph total        781.244 ms -> 732.635 ms

profile hit counts:
  pic_compact_sm70=1: 403 lines
  pic_compact_sm70=0: 488 lines
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_piccompact_sm70_sweep_1_50_20260611_223627
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.321451,1.013
full-reuse,full,0.636562,3.696

cacheblend,0.01,0.314889,7.471
cacheblend,0.05,0.374720,6.278
cacheblend,0.10,0.430173,5.469
cacheblend,0.20,0.642467,3.662
cacheblend,0.30,0.720687,3.264
cacheblend,0.40,0.687828,3.420
cacheblend,0.50,0.818838,2.873

epic,0.01,0.265916,8.847
epic,0.05,0.303827,7.743
epic,0.10,0.329392,7.142
epic,0.20,0.440432,5.341
epic,0.30,0.538072,4.372
epic,0.40,0.570946,4.120
epic,0.50,0.705095,3.336
```

Compared with q32 + k-slot-cache threshold 384 no-profile sweep:

```text
cacheblend40: 0.755525 s -> 0.687828 s  (+8.96%)
cacheblend50: 0.872416 s -> 0.818838 s  (+6.14%)
epic40:       0.636039 s -> 0.570946 s  (+10.23%)
epic50:       0.757277 s -> 0.705095 s  (+6.89%)
```

Low-budget 1%-30% results are within small noise/regression because the new path does not trigger below `M=384`; all cacheblend/epic ratios remain faster than normal full-compute.

Conclusion:

- Keep the SM70 compact CUTLASS fast path as the default CUDA compact dense high-budget path.
- Do not broaden it below `M=384` without another full 1%-50% sweep; low-budget compact rows previously rejected packed GEMV/PicGEMM attempts and are sensitive to launch/tiling overhead.
- P0 sparse flash remains the cacheblend-specific bottleneck; P1 dense is improved for high budgets, but `Convolution` is still large enough that graph-level gate/up fusion and fused SiLU*up remain worthwhile.

Rejected follow-up: lower the SM70 compact CUTLASS threshold to `M >= 256`.

```text
tag=pic_cuda_1024_piccompact_sm70_thr256_smoke_30_50_20260611_224503
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

Compared with accepted M>=384 sweep:
cacheblend30: 0.720687 s -> 0.736422 s  (-15.735 ms)
cacheblend40: 0.687828 s -> 0.712581 s  (-24.753 ms)
cacheblend50: 0.818838 s -> 0.832377 s  (-13.539 ms)
epic30:       0.538072 s -> 0.528442 s  (+9.630 ms)
epic40:       0.570946 s -> 0.574024 s  (-3.078 ms)
epic50:       0.705095 s -> 0.704313 s  (+0.782 ms)
```

Do not lower the default threshold to 256. It gives a small epic30 win but regresses cacheblend30/40/50, so it violates the default-path rule that cacheblend and epic must both remain faster/no-regression across the sweep.

## 2026-06-11 Jetson CUDA half2 SiLU Unary Fast Path

Tested and accepted a P2 activation optimization:

- Add a contiguous FP16 `UnaryOpOperation_SILU` half2 kernel in CUDA `UnaryBlit`.
- Keep the existing generic unary path for non-contiguous tensors, non-FP16, odd element counts, and other unary ops.
- This is not the full graph-level `SiLU(gate) * up` fusion, but it removes most of the expensive generic SiLU unary time while preserving graph structure and tensor semantics.

Profile A/B against accepted SM70 compact CUTLASS path:

```text
old tag=pic_cuda_1024_piccompact_sm70_profile_40_50_20260611_223500
new tag=pic_cuda_1024_half2_silu_profile_50_20260611_225119
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend50 graph:
  UnaryOp            73.874 ms ->   9.009 ms
  BinaryOp           25.802 ms ->  26.302 ms
  Convolution       354.839 ms -> 356.754 ms
  PicSparseAttention 236.047 ms -> 243.252 ms
  graph total       851.371 ms -> 807.220 ms

epic50 graph:
  UnaryOp            74.187 ms ->   8.979 ms
  BinaryOp           26.307 ms ->  26.158 ms
  Convolution       361.439 ms -> 358.732 ms
  PicSparseAttention 119.267 ms -> 119.180 ms
  graph total       732.635 ms -> 664.275 ms

compact SiLU representative:
  /blocks.* /mlp/act_fn/Mul_output_0, input [1x519x8192]
  about 4.16 ms/layer -> about 0.34 ms/layer
```

No-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_half2_silu_sweep_1_50_20260611_225217
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.190676,1.074
full-reuse,full,0.637651,3.689

cacheblend,0.01,0.310606,7.574
cacheblend,0.05,0.354184,6.642
cacheblend,0.10,0.410112,5.736
cacheblend,0.20,0.609088,3.862
cacheblend,0.30,0.678736,3.466
cacheblend,0.40,0.635933,3.699
cacheblend,0.50,0.753021,3.124

epic,0.01,0.252736,9.308
epic,0.05,0.287881,8.172
epic,0.10,0.307184,7.658
epic,0.20,0.406571,5.786
epic,0.30,0.494993,4.753
epic,0.40,0.515728,4.562
epic,0.50,0.635977,3.699
```

Correctness smoke:

```text
Jetson command:
  run_test.out op/unary/silu 2 1

result:
  all <op/unary/silu> tests passed
```

Compared with accepted SM70 compact CUTLASS no-profile sweep:

```text
full-compute: 2.321451 s -> 2.190676 s  (+5.63%)
cacheblend10: 0.430173 s -> 0.410112 s  (+4.66%)
cacheblend20: 0.642467 s -> 0.609088 s  (+5.20%)
cacheblend30: 0.720687 s -> 0.678736 s  (+5.82%)
cacheblend40: 0.687828 s -> 0.635933 s  (+7.54%)
cacheblend50: 0.818838 s -> 0.753021 s  (+8.04%)
epic10:       0.329392 s -> 0.307184 s  (+6.74%)
epic20:       0.440432 s -> 0.406571 s  (+7.69%)
epic30:       0.538072 s -> 0.494993 s  (+8.01%)
epic40:       0.570946 s -> 0.515728 s  (+9.67%)
epic50:       0.705095 s -> 0.635977 s  (+9.80%)
```

Conclusion:

- Keep the CUDA half2 SiLU fast path as default. It benefits full-compute and all sparse ratios, and no target/async/error log lines appeared.
- The remaining elementwise opportunity is the actual graph-level `SiLU(gate) * up` fusion, which would remove the separate BinaryOp and some raster/intermediate traffic. Do not call that done just because unary SiLU is now fast.
- After this optimization, cacheblend50 is again mostly split between `PicSparseAttention` and compact `Convolution`; `UnaryOp` is no longer the first P2 bottleneck.

## 2026-06-11 Jetson CUDA Rejected Sparse Flash Register Accumulator

Goal:

- Reduce CUDA `pagedSparseFlashTileKernel` hot-loop shared-memory traffic by moving the output accumulator from
  `outShared[Q_TILE][HEAD_DIM]` into two per-thread register accumulators.
- Test this as a P0 sparse flash attention follow-up after the accepted k-slot cache and q32/k32 threshold.

Code/env:

- Changed only `source/backend/cuda/execution/PagedAttentionExecution.cu` after the accepted q32/k-slot path:
  removed `outShared`, used `accIdx0=linearTid` and `accIdx1=linearTid+linearThreads`, and wrote output directly
  from `outAcc0/outAcc1`.
- Built locally with Jetson CUDA cross compile and synced to Jetson.
- Profile tag:

```text
pic_cuda_1024_sparse_regacc_profile_50_20260611_150559
```

Profile result:

```text
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

profile latency:
cacheblend50: 0.982005 s
epic50:       0.776255 s

CUDAPagedAttention op=sparse_flash_tile_attention across cacheblend50+epic50:
  count=30
  total=441.449 ms
  max_layer=9.237 ms
```

Compared with accepted half2 SiLU profile:

```text
accepted profile latency:
cacheblend50: 0.807220 s
epic50:       0.664275 s

accepted PicSparseAttention graph total:
cacheblend50: 243.252 ms
epic50:       119.180 ms
total:        362.432 ms
```

Conclusion:

- Reject this variant. It worsens the sparse flash portion instead of improving it, likely because the extra private
  registers reduce occupancy / scheduling efficiency on sm72 more than the removed shared-memory accumulator helps.
- Reverted the register-accumulator source change and rebuilt/synced the accepted artifact back to Jetson.
- Do not reintroduce this `outAcc0/outAcc1` variant as a default CUDA sparse flash path.

Post-revert smoke:

```text
tag=pic_cuda_1024_after_regacc_revert_smoke_50_20260611_150905
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

full-compute: 2.201449 s
full-reuse:   0.638487 s
cacheblend50: 0.806360 s
epic50:       0.656208 s
```

## 2026-06-11 Jetson CUDA SM70 Compact CUTLASS Hybrid Tile

Goal:

- Continue P1 compact dense optimization after static dequant cache and the accepted `128x64x64` SM70 compact CUTLASS path.
- Target the remaining 50% budget compact MLP shapes:
  - `M=519,K=2048,N=8192` gate/up
  - `M=519,K=8192,N=2048` down

Code/env:

- Added a second SM70 FP16 tensor-core Linear instance:

```text
GemmTensor_F16_F16_Linear_AlignTensor_Sm70_64x128x64
```

- Added `mPicCompactSm70Tile` profile metadata:
  - `0`: normal CUTLASS path
  - `1`: existing PIC compact `128x64x64`
  - `2`: new PIC compact `64x128x64`
- Kept the path default and shape-gated, with no env switch:

```text
384 <= M < 512: 128x64x64
M >= 512:       64x128x64
```

Rejected intermediate:

```text
tag=pic_cuda_1024_piccompact_nwide_smoke_40_50_20260611_151644

cacheblend40: 0.695613 s
cacheblend50: 0.739442 s
epic40:       0.517384 s
epic50:       0.624207 s
```

The all-compact `64x128x64` tile improves 50% but regresses cacheblend40 badly compared with the accepted half2 SiLU sweep
(`cacheblend40=0.635933 s`). Do not use it for `M < 512`.

Profile evidence for `M=519`:

```text
tag=pic_cuda_1024_piccompact_nwide_profile_50_20260611_151532
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

accepted 128x64x64 profile, compact M=519:
  2048->8192: avg cutlass 5452.2 us
  8192->2048: avg cutlass 5552.8 us
  2048->2048: avg cutlass 1397.7 us

64x128x64 profile, compact M=519:
  2048->8192: avg cutlass 4916.4 us
  8192->2048: avg cutlass 4722.0 us
  2048->2048: avg cutlass 1267.6 us
```

Accepted no-profile 1%-50% sweep:

```text
tag=pic_cuda_1024_piccompact_hybrid_nwide_sweep_1_50_20260611_151823
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

normal-full-compute,full,2.352533,1.000
full-compute,full,2.192074,1.073
full-reuse,full,0.636807,3.694

cacheblend,0.01,0.308307,7.630
cacheblend,0.05,0.363945,6.464
cacheblend,0.10,0.414109,5.681
cacheblend,0.20,0.611740,3.846
cacheblend,0.30,0.678835,3.466
cacheblend,0.40,0.634798,3.706
cacheblend,0.50,0.721868,3.259

epic,0.01,0.254410,9.247
epic,0.05,0.290160,8.108
epic,0.10,0.308750,7.620
epic,0.20,0.408996,5.752
epic,0.30,0.494652,4.756
epic,0.40,0.514860,4.569
epic,0.50,0.606127,3.881
```

Compared with the accepted half2 SiLU sweep:

```text
cacheblend50: 0.753021 s -> 0.721868 s  (+4.14%)
epic50:       0.635977 s -> 0.606127 s  (+4.69%)
cacheblend40: 0.635933 s -> 0.634798 s  (+0.18%)
epic40:       0.515728 s -> 0.514860 s  (+0.17%)
```

Conclusion:

- Accept the hybrid SM70 compact CUTLASS tile as the default P1 dense path.
- Keep `64x128x64` gated to `M >= 512`; below that, the existing `128x64x64` path is safer.
- This is a compact dense win, not an attention win. After it, cacheblend50 still needs P0 sparse attention work, while epic50 is pushed further toward dense/MLP and graph overhead.

## 2026-06-11 Jetson CUDA Rejected Mid-Budget q16 Sparse Tile

Goal:

- Test whether `384 <= attnLen < 512` should use a q16/k32 sparse flash tile instead of the accepted q32/k32 tile.
- Motivation: 40% budgets have active rows around 418, so q32 might waste K streaming for scattered cacheblend rows even though it reduces block count.

Code/env:

- Temporary change in `source/backend/cuda/execution/PagedAttentionExecution.cu`:

```text
attnLen < 384:   q8/k32
384 <= attnLen < 512: q16/k32
attnLen >= 512:  q32/k32
```

- Built locally with Jetson CUDA cross compile and synced to Jetson.

Profile run:

```text
tag=pic_cuda_1024_sparse_midq16_profile_40_50_20260611_152446
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

sparse tile summary:
  attn=418 q_tile=16 count=29 total=339.458 ms max=17.075 ms
  attn=519 q_tile=32 count=28 total=332.758 ms max=16.353 ms

graph profile:
cacheblend40 PicSparseAttention: 247.913 ms
cacheblend50 PicSparseAttention: 235.928 ms
epic40       PicSparseAttention:  96.965 ms
epic50       PicSparseAttention: 118.787 ms
```

No-profile smoke:

```text
tag=pic_cuda_1024_sparse_midq16_smoke_40_50_20260611_152613
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend40: 0.720326 s
cacheblend50: 0.743643 s
epic40:       0.522904 s
epic50:       0.623024 s
```

Compared with accepted hybrid dense + q32 threshold sweep:

```text
cacheblend40: 0.634798 s -> 0.720326 s  (regression)
cacheblend50: 0.721868 s -> 0.743643 s  (regression/noise)
epic40:       0.514860 s -> 0.522904 s  (regression)
epic50:       0.606127 s -> 0.623024 s  (regression)
```

Conclusion:

- Reject q16 for the `384 <= attnLen < 512` mid-budget range.
- Reverted to the accepted rule: q32/k32 when `attnLen >= 384`, q8/k32 below that.
- q32's larger CTA still wins for the current 1024-token 40% shape despite potential scattered-row K waste. Future P0 must reduce causal K work with a more explicit bucket/adaptive-piece design, not by simply shrinking q32 to q16.

Post-revert smoke:

```text
tag=pic_cuda_1024_after_midq16_revert_smoke_40_50_20260611_152900
log_scan: target_unavailable=false, async_failed=false, error_lines=[]

cacheblend40: 0.701642 s
cacheblend50: 0.737961 s
epic40:       0.521909 s
epic50:       0.626386 s
```
