# CUDA Small-M Weight-Only Linear Optimization Plan

## Background

Jetson CUDA decode-repair profiling for `MiniCPM5-1B`, `Qwen3-4B`, and `Llama3.2-3B` shows that the `x=1 -> x=3` TPOT cliff is dominated by dense `Convolution`/Linear work, not by `PicSparseAttention`.

Decode repair uses `active_rows = repair_tokens + 1`:

```text
x=1 -> active rows = 2 -> tiny GEMV route
x=3 -> active rows = 4 -> rows4/5 cuBLAS or generic Conv route
```

The single-TPOT attribution from the current Jetson profile is:

| model | x1 TPOT ms | x3 TPOT ms | TPOT delta ms | Conv / TPOT | MLP / TPOT | attn-proj / TPOT | PicSparseAttention / TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| MiniCPM5-1B | 86.76 | 97.56 | +10.79 | +11.51 | +5.04 | +6.53 | -0.15 |
| Qwen3-4B | 170.47 | 225.59 | +55.12 | +51.67 | +28.93 | +22.87 | -0.22 |
| Llama3.2-3B | 128.82 | 160.07 | +31.26 | +28.20 | +17.72 | +10.59 | -0.23 |

MLP is a major contributor, but attention projections are also large. Therefore the right problem statement is not only "MLP fusion"; it is a unified small-M CUDA weight-only Linear optimization that benefits:

- MLP `gate_proj`, `up_proj`, `down_proj`.
- Attention `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- Existing MNN weight-only `1x1 Conv` / Linear backend paths.

## Hard Constraints

Do not change graph export for this optimization.

This means:

- Do not require new exporter flags.
- Do not change `transformers/pic_llm/export/*` to pack gate/up or q/k/v.
- Do not introduce a production dependency on fresh gate/up packed export.
- Do not rely on new graph topology to get speedup.
- Do not replace the production model with a hand-edited `.mnn`.

The production candidate must accelerate the existing exported graph by improving CUDA backend execution for the existing weight-only Conv/Linear ops.

## Relevant Skills

Use these Codex skills for this work:

- `.codex/skills/mnn-pic-optimize/SKILL.md`
  - Maintains the PIC/decode-repair optimization constraints, TPOT attribution, production-path lessons, and Jetson CUDA negative results.
- `.codex/skills/mnn-opt-ops/SKILL.md`
  - Defines the Jetson CUDA operator optimization workflow, cross-build/sync flow, direct-op benchmark rules, and decode-repair validation path.
- `.codex/skills/mnn-ops-bench/SKILL.md`
  - Use when adding or running CUDA direct-op accuracy/performance tests under `test/bench_ops/cuda`.
- `.codex/skills/mnn-build-artifacts/SKILL.md`
  - Use when rebuilding or syncing Jetson CUDA artifacts.
- `.codex/skills/mnn-pic-benchmark/SKILL.md`
  - Use when validating endpoint decode TPOT after a backend candidate passes single-op testing.

## Goal

Reduce decode-repair TPOT for `x=3/5/7`, especially the `x=1 -> x=3` cliff, while keeping `x=0/1` and normal decode from regressing.

Primary target models:

```text
MiniCPM5-1B
Qwen3-4B
Llama3.2-3B
```

Do not use `Llama3.2-1B` as the main optimization decision target.

## Optimization Idea

Build a better CUDA small-M weight-only Linear route for rows `4/6/8`.

The target backend path is:

```text
source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
```

Current behavior:

- `batch <= 3` in decode repair uses tiny INT4 GEMV.
- `batch >= 4` enters rows4/5 cuBLAS or generic CUTLASS/Conv paths.
- Qwen3-4B additionally shows `runtime_dequant=1 static_dequant=0` for some batch-4 Linear ops, which makes the cliff worse.

The candidate should improve the existing op execution without changing graph structure:

1. First ensure all hot rows4/6/8 target Linears hit static dequant where memory allows.
2. Add or tune a rows4/6/8 small-M weight-only kernel family only if direct-op tests show clear speedup.
3. Apply the same route to MLP and attention projection shapes through backend shape matching, not graph rewriting.
4. Keep existing safe routes for unsupported shapes or memory pressure.

## Non-Goals

These are not the production route for this task:

- Fresh gate/up packed exporter integration.
- `PicLinearNhwcWeightOnly + PicPackedSiluMul` as a required graph rewrite.
- Extending old V14 tiny GEMV to rows4/6/8 by default.
- Naive INT4 `PicGEMM`.
- Naive fused `SwiGLU + down`.
- Naive WMMA down kernel.
- Further tuning `PicSparseAttention` for the `x=1 -> x=3` cliff.

Previous experiments showed these routes either failed `/v1/prefill/text`, did not beat endpoint TPOT, or regressed direct-op rows4/6/8.

## Work Plan

### Phase 1: Single-Op Evidence

Start with direct CUDA operator tests before touching production routing.

Use or extend:

```text
test/bench_ops/cuda/CudaWeightOnlyConvPerf.cpp
```

Required single-op coverage:

- rows: `1/2/4/6/8`
- MLP shapes:
  - `hidden -> inter`
  - `inter -> hidden`
  - optional `hidden -> gateup_concat` as a benchmark-only theoretical floor
- attention projection shapes:
  - `hidden -> hidden`
  - `hidden -> kv`
  - optional `hidden -> qkv_concat` as a benchmark-only theoretical floor
- models:
  - MiniCPM5-1B shape set
  - Qwen3-4B shape set
  - Llama3.2-3B shape set

Single-op acceptance criteria:

- accuracy passes against the current backend route.
- rows4/6/8 improve enough to matter at endpoint TPOT.
- rows1/2 do not regress materially.
- Qwen3-4B no longer falls into avoidable runtime dequant on hot rows4/6/8 shapes.
- benchmark uses `Memory_Low`, otherwise it is not testing the production weight-only route.

### Phase 2: Backend Integration

Only after Phase 1 passes, integrate the candidate into `ConvFpAIntBExecution.cu`.

Integration rules:

- Match existing weight-only `1x1 Conv` / Linear ops by shape and metadata.
- Do not require exporter changes.
- Do not require new graph op types.
- Guard the route by precision, quantization type, row count, static-dequant availability, and supported dimensions.
- Keep fallback to current production path.
- Add profile logging that clearly distinguishes:
  - tiny GEMV
  - rows4/6/8 candidate route
  - rows45 cuBLAS
  - generic CUTLASS/Conv
  - runtime dequant vs static dequant

### Phase 3: Endpoint Decode Validation

After backend integration passes direct-op tests, run endpoint decode TPOT.

Required decode matrix:

- `x=0/1/3/5/7`
- target models:
  - MiniCPM5-1B
  - Qwen3-4B
  - Llama3.2-3B
- multiple context lengths.
- compare against current production baseline and normal LLM decode baseline where relevant.

Endpoint acceptance criteria:

- `x=3/5/7` TPOT improves versus current production baseline.
- `x=0/1` does not regress.
- normal decode path does not regress.
- `/v1/prefill/text` still succeeds before decode tests.
- endpoint logs show the intended backend route is hit.
- no production report uses `MNN_PIC_GRAPH_PROFILE=1` latency as final latency; graph profile is attribution only.

## Production Decision Rule

A backend candidate can become the default production route only if all are true:

1. Direct-op accuracy passes.
2. Direct-op rows4/6/8 speedup is clear on the three target model shape families.
3. Endpoint decode TPOT improves on `x=3/5/7`.
4. `x=0/1` and normal decode do not regress.
5. No graph export change is required.
6. No new model export artifact is required for the speedup.
7. The current fastest production baseline remains the fallback for unsupported shapes.

If a candidate only improves a benchmark-only packed shape such as `gateup_concat` or `qkv_concat`, it is not enough. It must speed up the existing exported graph, or it remains an experiment.

## Reporting Format

Use single-TPOT deltas, not request-total deltas.

Preferred table:

| model | context | x | baseline TPOT ms | candidate TPOT ms | delta ms | route evidence |
|---|---:|---:|---:|---:|---:|---|

For attribution, report:

| model | x1 TPOT ms | x3 TPOT ms | TPOT delta ms | Conv / TPOT | MLP / TPOT | attn-proj / TPOT | PicSparseAttention / TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|

Do not use request-total delta or `request delta / N` as the primary optimization metric.
