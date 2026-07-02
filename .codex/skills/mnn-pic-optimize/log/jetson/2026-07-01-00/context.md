# Jetson CUDA Decode Repair MLP Fusion Plan

## Objective

The endpoint target is to move repair `x=3/5/7` below `95 ms` on top of the current accepted packed gate/up implementation. This pass excludes attention work. The MLP-side target is:

```text
rows4/6/8 MLP savings target: about 0.25 ms/layer
28 layers endpoint equivalent: about 7.0 ms/token
```

This target is not arbitrary. Current post-packed endpoint gaps to `95 ms` are:

| x | active rows | current TPOT ms | target ms | gap ms | gap / 28 layers |
|--:|------------:|----------------:|----------:|-------:|----------------:|
| 3 | 4 | 103.547 | 95.000 | 8.547 | 0.305 |
| 5 | 6 | 105.557 | 95.000 | 10.557 | 0.377 |
| 7 | 8 | 106.806 | 95.000 | 11.806 | 0.422 |

MLP cannot realistically cover all of the x=7 gap by itself, but it needs to contribute roughly `7 ms/token` for the combined endpoint plan to be credible. That means a bench-only MLP variant that saves only `0.02-0.05 ms/layer` is not worth endpoint integration for this objective.

## Current MLP Baseline

Accepted packed gate/up profile:

| x | rows | gate/up ms | down ms | MLP ms |
|--:|-----:|-----------:|--------:|-------:|
| 3 | 4 | 29.157 | 17.083 | 46.240 |
| 5 | 6 | 26.495 | 15.490 | 41.985 |
| 7 | 8 | 27.787 | 15.653 | 43.440 |

Per-layer rough scale:

| rows | gate/up ms/layer | down ms/layer | MLP ms/layer |
|-----:|-----------------:|--------------:|-------------:|
| 4 | 1.041 | 0.610 | 1.651 |
| 6 | 0.946 | 0.553 | 1.499 |
| 8 | 0.993 | 0.559 | 1.551 |

Required `0.25 ms/layer` is therefore about `15%-17%` of total packed MLP time. A down-only change would need to remove roughly `40%-45%` of down time, which is not supported by the prior direct-op data. A full MLP fusion can plausibly target both:

- intermediate `[rows, 8192]` activation write/read after packed SiLU/multiply;
- launches and graph glue around packed gate/up, activation, down input raster, and down projection.

## Prior Negative Evidence

The following down-proj-only or tiled-cuBLAS directions are not sufficient for this objective:

- forced N-wide CUTLASS regressed rows `4/6/8` by about `0.067 ms/layer`;
- cuBLASLt heuristic sweep did not beat the current rows45 cuBLAS path;
- tiled cuBLAS floor with beta-accumulated down showed only `0.017-0.036 ms/layer` chain-level win at best, and tiled down itself was flat/slower;
- rows45 `compute=16f` improved isolated down by only about `0.010-0.013 ms/layer` and changes accumulation policy;
- WMMA small-M down was about 2x slower than current cuBLAS;
- transposed static FP16 down layout did not help under the default 32F/fast16 policies;
- split-N output chunks only showed about `0.016-0.030 ms/layer` lower-bound win.

Conclusion: do not spend the next pass on another down-only dispatch policy. The bench target must be a true fused MLP chain or a strong lower-bound for one.

## Bench-Only Design

Do not modify the exported graph first. Add a direct CUDA bench case that exercises a single logical fused MLP operator against the current packed chain.

Inputs:

```text
hidden:      [rows, 3072] fp16
gate weight: packed/current INT4 or static FP16 equivalent
up weight:   packed/current INT4 or static FP16 equivalent
down weight: current static FP16 down cache or equivalent bench weight
rows:        1, 2, 4, 6, 8
```

Reference chain:

```text
gate_up = PicLinearNhwcWeightOnly(hidden, packed_gate_up_weight)  // [rows, 16384]
act     = PicPackedSiluMul(gate_up)                               // [rows, 8192]
out     = rows45 cuBLAS down(act, down_weight)                     // [rows, 3072]
```

Candidate fused-chain semantics:

```text
out = FusedDecodeRepairMlp(hidden, packed_gate_up_weight, down_weight)
```

The fused candidate must produce the same output as the reference chain within the same tolerance currently used by the CUDA bench-op accuracy tests for rows4/6/8 gate/up/down.

## Candidate Variants

### Variant A: fused activation plus down-input staging

Keep current packed gate/up projection unchanged, but replace `PicPackedSiluMul + down input raster + down` with a bench-only fused activation/down entry point.

Expected value:

- smallest implementation risk;
- directly tests whether eliminating activation materialization and down input layout glue has enough headroom;
- preserves current gate/up quality.

Concern:

- if down remains one ordinary cuBLAS call over a fully materialized activation, this becomes close to the already rejected tiled floor and probably cannot reach `0.25 ms/layer`;
- to clear the target, activation must feed down accumulation through a more efficient staging path than global `[rows,8192]` write/read.

Promotion gate:

```text
rows4/6/8 chain delta <= -0.25 ms/layer
rows1/2 no material regression
accuracy pass
```

### Variant B: tile-streamed fused MLP lower bound

Split intermediate dim `8192` into large tiles, compute gate/up tile, apply SiLU/multiply, and accumulate into down output tile-by-tile.

This is the conceptually correct fusion because it can avoid writing the full intermediate activation to global memory:

```text
for inter_tile in 0..8192:
    gate_tile = hidden @ W_gate_tile
    up_tile   = hidden @ W_up_tile
    act_tile  = silu(gate_tile) * up_tile
    out      += act_tile @ W_down_tile
```

Implementation warning:

- the previous ordinary tiled cuBLAS floor was too weak because it split work into separate GEMMs and beta accumulations;
- a useful variant must preserve tensor-core-quality gate/up and avoid excessive per-tile launch overhead;
- this may need a custom persistent/threadblock schedule or a grouped-GEMM style bench path, not a simple loop of cuBLAS calls.

Promotion gate:

```text
at least -0.25 ms/layer on rows4/6/8
no target row regression
numerical tolerance matches reference
no graph/export dependency
```

### Variant C: fused custom kernel only for activation/down micro-part

If full tile-streamed MLP is too large, create a narrower custom bench kernel:

```text
input:  packed gate_up output [rows, 16384]
output: final hidden [rows, 3072]
work:   SiLU(gate) * up + down accumulation
```

This isolates the exact value of removing the `[rows,8192]` activation tensor and down-input raster. It is a useful floor even if the first implementation is not production-ready.

Decision rule:

- if this micro-fusion cannot approach `0.25 ms/layer`, full graph integration is unlikely to pay off;
- if it clears the gate, then graph integration can be considered later with the packed gate/up op feeding the fused activation/down op.

## Measurement Plan

Use a single-op CUDA bench rather than endpoint:

1. add bench-only cases under the existing CUDA bench-op test file or a dedicated CUDA MLP bench file;
2. load or synthesize the same target shapes as Llama3.2-3B decode repair;
3. run accuracy for rows `1/2/4/6/8`;
4. run perf for rows `1/2/4/6/8`, with enough repeats to reduce per-launch noise;
5. report:

```text
rows
reference packed gate/up ms
reference packed silu ms
reference down ms
reference chain ms
candidate fused ms
delta ms/layer
accuracy max abs / max rel
```

Promotion threshold:

| rows | required delta |
|-----:|---------------:|
| 4 | <= -0.25 ms/layer |
| 6 | <= -0.25 ms/layer |
| 8 | <= -0.25 ms/layer |

Rows `1/2` are guardrails. They do not need to hit `0.25 ms/layer`, but they must not regress enough to hurt x=0/1 endpoint behavior.

## Execution Logic

Phase 1: establish a cleaner bench baseline.

- Re-run current packed chain rows `1/2/4/6/8` in direct bench mode.
- Confirm the direct bench numbers agree with the accepted profile scale:
  - rows4/6/8 packed chain around `1.5 ms/layer`;
  - down around `0.50-0.60 ms/layer`;
  - packed gate/up around `0.95-1.10 ms/layer`.

Phase 2: implement bench-only micro-fusion.

- First implement Variant C or A because it isolates activation/down materialization without touching graph export.
- Keep all code behind a bench case or non-production test path.
- Do not route `PicLinearNhwcWeightOnly` production execution to the new path.

Phase 3: decide whether a full fused MLP is worth engineering.

- If the micro-fusion saves much less than `0.25 ms/layer`, stop and do not integrate.
- If it saves near or above `0.25 ms/layer`, prototype Variant B as a better production-shaped fused op.
- Only after direct-op evidence passes should we discuss graph/export integration.

Phase 4: endpoint validation only after direct-op pass.

- Build a temporary graph/export path or manual graph replacement only after direct bench passes.
- Validate x=`0/1/3/5/7`, with endpoint focus on x=`3/5/7`.
- Accept only if formal endpoint shows a real `>=6-7 ms/token` MLP-side improvement without x=0/1 regression.

## Current Hypothesis

The hypothesis that a small custom fused activation/down kernel can beat the current packed MLP baseline is rejected for this pass.

Negative data now covers three bench-only variants:

- `PicBenchFusedPackedSiluDown`: materialized `PicPackedSiluMul + cuBLAS down` in one Extra, effectively a launch/glue lower bound.
- `PicBenchStreamedPackedSiluDown`: scalar streaming activation/down that avoids the full `[rows,8192]` activation writeback.
- `PicBenchWmmaPackedSiluDown`: custom WMMA 16x16 down candidate that keeps tensor-core math but uses a simple custom schedule.

All three failed the rows `4/6/8` promotion gate. The custom WMMA candidate is the most important lesson: using tensor cores is not sufficient if the schedule is worse than cuBLAS and still stages activation tiles inefficiently.

## Follow-Up Commands

WMMA accuracy:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  mkdir -p .cache/bench_ops/mlp_fusion_20260701 && \
  export ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  export CUDA_LIB=/usr/local/cuda-12.2/targets/aarch64-linux/lib && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:$CUDA_LIB:${LD_LIBRARY_PATH:-}" && \
  MNN_BENCH_WEIGHT_ONLY_ROWS=1,2,4,6,8 \
  "$ART/bin/run_test.out" bench_ops/cuda/accuracy/PicFusedSiluDown 2 2 1 x 2 \
    2>&1 | tee .cache/bench_ops/mlp_fusion_20260701/pic_fused_silu_down_wmma_accuracy.log'
```

WMMA rows `4/6/8` performance:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  mkdir -p .cache/bench_ops/mlp_fusion_20260701 && \
  export ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  export CUDA_LIB=/usr/local/cuda-12.2/targets/aarch64-linux/lib && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:$CUDA_LIB:${LD_LIBRARY_PATH:-}" && \
  MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
  MNN_BENCH_FUSED_MLP_WARMUP=20 \
  MNN_BENCH_FUSED_MLP_REPEAT=80 \
  "$ART/bin/run_test.out" bench_ops/cuda/perf/PicFusedSiluDown 2 2 1 x 2 \
    2>&1 | tee .cache/bench_ops/mlp_fusion_20260701/pic_fused_silu_down_wmma_perf_rows468_repeat80.log'
```

Production-shaped baseline retest:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  mkdir -p .cache/bench_ops/mlp_fusion_20260701 && \
  export ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  export CUDA_LIB=/usr/local/cuda-12.2/targets/aarch64-linux/lib && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:$CUDA_LIB:${LD_LIBRARY_PATH:-}" && \
  MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
  MNN_BENCH_MLP_HIDDEN=3072 \
  MNN_BENCH_MLP_INTER=8192 \
  MNN_BENCH_MLP_WARMUP=20 \
  MNN_BENCH_MLP_REPEAT=80 \
  "$ART/bin/run_test.out" bench_ops/cuda/perf/PicDecodeMlp 2 2 1 x 2 \
    2>&1 | tee .cache/bench_ops/mlp_fusion_20260701/pic_decode_mlp_baseline_retest_rows468_h3072_repeat80.log'
```

Guardrail rows `1/2` used the same command with `MNN_BENCH_WEIGHT_ONLY_ROWS=1,2` and log path:

```text
.cache/bench_ops/mlp_fusion_20260701/pic_decode_mlp_baseline_retest_rows12_h3072_repeat80.log
```

## Final Decision Boundary

Do not integrate these fused/down candidates into exporter, production graph rewrite, or default runtime selection. The production path should remain the current packed gate/up + packed SiLU + rows4/6/8 cuBLAS down baseline unless a future direct-op candidate beats it by at least `0.25 ms/layer` on all of rows `4/6/8` with accuracy passing.
The most credible MLP-only path is not a faster standalone down GEMM. It is a fused chain that avoids writing and rereading the full `[rows,8192]` SwiGLU activation while also reducing compact-row graph glue. The bench-only test should therefore answer one question before any graph work:

```text
Can rows4/6/8 MLP chain save about 0.25 ms/layer while preserving current accuracy?
```

If yes, the path is worth turning into a real graph op. If no, the remaining `95 ms` work must come from outside MLP or from a more invasive tensor-core fused MLP design.

## 2026-07-01 Bench-Only Fused Activation/Down Result

Implemented a bench-only CUDA Extra pair, not wired to exporter or production graph:

```text
PicBenchFusedPackedSiluDown
  input:  packed gate/up [rows, 16384]
  weight: down FP16 production layout [3072, 8192]
  work:   PicPackedSiluMul into a temporary activation, then cuBLAS down

PicBenchStreamedPackedSiluDown
  input:  packed gate/up [rows, 16384]
  weight: down FP16 production layout [3072, 8192]
  work:   compute activation tiles in shared memory and scalar-accumulate down output
```

New direct-op entries:

```text
bench_ops/cuda/accuracy/PicFusedSiluDown
bench_ops/cuda/perf/PicFusedSiluDown
```

Build and sync:

```bash
CUDA_TOOLKIT_ROOT="$PWD/.cache/sysroots/jetson_cuda" \
CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc \
BUILD_DIR="$PWD/.cache/build/mnn/jetson_cross_cuda" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/jetson_cross_cuda" \
JOBS=96 CUDA_ARCHS=72 ENABLE_CROSS_CUDA=ON \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS="-DMNN_BUILD_TEST=ON -DCUDA_HOST_COMPILER=$PWD/.cache/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++" \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/jetson_cross_cuda/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

Accuracy command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  export ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  export CUDA_LIB=/usr/local/cuda-12.2/targets/aarch64-linux/lib && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:$CUDA_LIB:${LD_LIBRARY_PATH:-}" && \
  MNN_BENCH_WEIGHT_ONLY_ROWS=1,2,4,6,8 \
  "$ART/bin/run_test.out" bench_ops/cuda/accuracy/PicFusedSiluDown 2 2 1 x 2'
```

Accuracy result:

| rows | materialized max_abs | materialized bad | streamed max_abs | streamed bad |
|-----:|---------------------:|-----------------:|-----------------:|-------------:|
| 1 | 0 | 0/3072 | 0.000244 | 0/3072 |
| 2 | 0 | 0/6144 | 0.000488 | 0/6144 |
| 4 | 0 | 0/12288 | 0.000488 | 0/12288 |
| 6 | 0 | 0/18432 | 0.000488 | 0/18432 |
| 8 | 0 | 0/24576 | 0.000488 | 0/24576 |

Perf command:

```bash
ssh jetson@192.168.101.192 'cd /home/jetson/code/kvshare-edge/impl/MNN && \
  export ART=.cache/output/mnn/artifacts/jetson_cross_cuda && \
  export CUDA_LIB=/usr/local/cuda-12.2/targets/aarch64-linux/lib && \
  export LD_LIBRARY_PATH="$PWD/$ART/lib:$CUDA_LIB:${LD_LIBRARY_PATH:-}" && \
  MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
  MNN_BENCH_FUSED_MLP_WARMUP=20 \
  MNN_BENCH_FUSED_MLP_REPEAT=80 \
  "$ART/bin/run_test.out" bench_ops/cuda/perf/PicFusedSiluDown 2 2 1 x 2'
```

Repeat=80 result:

| rows | gateup ms | packed_silu_down ms | baseline_chain ms | materialized_chain ms | materialized delta | streamed_chain ms | streamed delta |
|-----:|----------:|--------------------:|------------------:|----------------------:|-------------------:|------------------:|---------------:|
| 4 | 0.9649 | 0.5178 | 1.5168 | 1.5154 | -0.0015 | 3.6646 | +2.1478 |
| 6 | 0.9616 | 0.5086 | 1.5209 | 1.5240 | +0.0031 | 5.0098 | +3.4889 |
| 8 | 0.9750 | 0.5363 | 1.5496 | 1.5479 | -0.0017 | 6.2990 | +4.7494 |

Current packed-chain cross-check at target hidden size:

```bash
MNN_BENCH_WEIGHT_ONLY_ROWS=4,6,8 \
MNN_BENCH_MLP_HIDDEN=3072 \
MNN_BENCH_MLP_INTER=8192 \
MNN_BENCH_MLP_WARMUP=20 \
MNN_BENCH_MLP_REPEAT=80 \
"$ART/bin/run_test.out" bench_ops/cuda/perf/PicDecodeMlp 2 2 1 x 2
```

`PicDecodeMlpPacked` chain:

| rows | concat ms | packed_silu ms | down ms | chain ms |
|-----:|----------:|---------------:|--------:|---------:|
| 4 | 0.9620 | 0.0050 | 0.5079 | 1.5237 |
| 6 | 0.9586 | 0.0051 | 0.5137 | 1.5270 |
| 8 | 0.9680 | 0.0057 | 0.5175 | 1.5350 |

Conclusion:

- The materialized single-op lower bound shows graph glue around `PicPackedSiluMul + down` is essentially zero on this direct-op path.
- The full activation tensor write/read is too small relative to tensor-core down to provide the required `0.25 ms/layer` headroom.
- The first streaming scalar micro-fusion preserves accuracy but is much slower because it gives up tensor-core down throughput.
- This candidate must not be promoted to graph/export integration.
- If MLP remains the target, the next credible bench-only candidate must be a tensor-core-quality fused MLP/down design, such as a CUTLASS-style custom A iterator/epilogue or a grouped persistent schedule that does not recompute activation per output tile. A scalar streaming kernel is not a viable path.

## 2026-07-01 Full-Reuse Decode Matrix

User requested true end-to-end decode latency for `x=1..7`, multiple contexts,
and three model directories. This run used real `pic_server` chat requests, not
direct-op timing:

```text
device: Jetson Orin NX
artifact: .cache/output/mnn/artifacts/jetson_cross_cuda
server port: 18131
mode: full-reuse
budget: 0.00
selector: top_hkvd
contexts: 512,1024,1536
repair_tokens: 0,1,2,3,4,5,6,7
max_tokens: 32
warm_repeats: 1
repeats: 1
```

Important harness fixes / lessons:

- Local benchmark requests must set `NO_PROXY/no_proxy=192.168.101.192,127.0.0.1,localhost`; otherwise the host HTTP proxy can return `502 Bad Gateway` before the request reaches Jetson.
- `repair_tokens=0` / `tpd=0` is the no-repair decode baseline and must not send `decode_refine`.
- `repair_tokens>0` must send `decode_refine` and must validate `mnn_token_id_sparse_decode`.

Command shape:

```bash
NO_PROXY=192.168.101.192,127.0.0.1,localhost \
no_proxy=192.168.101.192,127.0.0.1,localhost \
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://192.168.101.192:18131 \
  --device jetson \
  --device-display 'Jetson Orin NX' \
  --backend cuda \
  --frequency-profile max \
  --model llama-pic \
  --mode full-reuse \
  --contexts 512,1024,1536 \
  --budgets 0.00 \
  --repair-tokens 0,1,2,3,4,5,6,7 \
  --decode-selectors top_hkvd \
  --max-tokens 32 \
  --repeats 1 \
  --warm-repeats 1 \
  --require-exact-context \
  --require-decode-runtime mnn_token_id_sparse_decode
```

Output:

```text
.cache/bench_ops/mlp_fusion_20260701/decode_matrix_20260701_012219/benchmark_decode_matrix.csv
.cache/bench_ops/mlp_fusion_20260701/decode_matrix_20260701_012219/pivot_tpot_ms.tsv
```

Raw validation:

```text
rows: 72
runtime_bad_count: 0
x=0: decode_refine_enabled=false, runtime=None
x=1..7: decode_refine_enabled=true, runtime=mnn_token_id_sparse_decode
server log scan: no ERROR / unsupported / target unavailable / async persistent / PicBench / Bad Gateway
```

Pivot TPOT in ms/token:

| model | ctx | KV cache | x=0 | x=1 | x=2 | x=3 | x=4 | x=5 | x=6 | x=7 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama3.2-1B silumul | 512 | full-reuse | 28.861 | 30.592 | 35.033 | 37.706 | 37.100 | 37.416 | 38.295 | 39.490 |
| Llama3.2-1B silumul | 1024 | full-reuse | 32.954 | 36.302 | 39.153 | 44.193 | 41.096 | 41.942 | 42.272 | 44.164 |
| Llama3.2-1B silumul | 1536 | full-reuse | 36.302 | 40.559 | 43.029 | 49.336 | 45.338 | 46.363 | 46.618 | 49.011 |
| Llama3.2-1B gateup-packed-silu | 512 | full-reuse | 28.456 | 29.944 | 34.525 | 37.114 | 36.438 | 37.330 | 38.096 | 39.614 |
| Llama3.2-1B gateup-packed-silu | 1024 | full-reuse | 32.432 | 35.248 | 38.683 | 43.129 | 40.477 | 41.643 | 42.140 | 44.074 |
| Llama3.2-1B gateup-packed-silu | 1536 | full-reuse | 35.964 | 40.397 | 42.540 | 48.666 | 45.058 | 45.515 | 46.201 | 48.403 |
| Llama3.2-1B decode-repair | 512 | full-reuse | 29.113 | 30.253 | 34.858 | 37.307 | 36.915 | 37.497 | 38.671 | 39.514 |
| Llama3.2-1B decode-repair | 1024 | full-reuse | 32.742 | 35.799 | 39.341 | 43.641 | 40.719 | 41.868 | 42.434 | 43.964 |
| Llama3.2-1B decode-repair | 1536 | full-reuse | 36.346 | 40.218 | 42.316 | 48.564 | 44.966 | 46.154 | 46.189 | 48.904 |

Conclusion:

- All measured rows are full-reuse KV cache decode requests.
- The gateup-packed-silu model directory is consistently the fastest or tied in most contexts.
- The gap between silumul / gateup-packed-silu / decode-repair is small at this context range; no bench-only `PicBench*` route is involved in production decode.
- For a larger formal table, repeat with more contexts and repeats, but keep the same runtime validation: `x=0` no repair, `x>0` token-id sparse decode.
