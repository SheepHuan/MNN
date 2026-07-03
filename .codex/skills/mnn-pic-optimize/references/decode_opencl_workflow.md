# OpenCL PIC Decode — Build / Benchmark / Validation Workflow

This is the reusable quick-reference for OpenCL PIC decode optimization on
OrangePi/Mali and Rhino/Adreno. It captures the build+sync commands, the TPOT A/B
harness, the P0 validity scan, and the skill-routing rules so a fresh session does
not need to re-explore them.

For the optimization *plan* and kernel design, see
`log/orangepi/2026-07-03-01/decode_optimization_plan_v6.md`. For the handoff state,
see `log/orangepi/2026-07-03-01/claude_handoff_decode.md`. For Rhino/Adreno
device-specific conclusions, see `references/rhino.md`.

## Skill routing (per AGENTS.md / CLAUDE.md)

- Build + sync + artifact check: `.codex/skills/mnn-build-artifacts/SKILL.md` +
  `scripts/build_artifacts.sh`.
- TPOT / decode benchmark: `.codex/skills/mnn-pic-benchmark/SKILL.md`; harness
  `.cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py`.
- Optimization log + hard constraints: `.codex/skills/mnn-pic-optimize/SKILL.md`
  (the `Decode 算子优化约束` section is authoritative).
- `mnn-opt-ops` is Jetson/CUDA-only — **not** for OpenCL decode on OrangePi/Rhino.

## Hard rules (must hold for every step)

- No routing back to `old` identity by model / device / GPU / head / q. `old` is
  A/B only (`MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=0`).
- Historical K transpose (`decode_prepare_transpose_k`) runs only in
  `prepareDecodePrefix` (`transformers/pic_llm/engine/src/llm.cpp:1625` →
  `PagedAttentionBufExecution.cpp:3959`). Timed decode may only append current-token
  K/V/decodeKey. Any profile line `decode_prepare_inside_decode=1` invalidates the run.
- Image mirrors (when used) are derived views of PagedCache (same status as
  `decodeKey`), allocated/filled in prepare, written alongside buffer in append —
  never a second runtime KV cache, never a staging buffer.
- Tune keys must carry `mali` / `adreno` namespace (`_openCLTuneDeviceKey`).
- Storage/compute types: `FLOAT*` = storage, `COMPUTE_FLOAT*` / `CONVERT_COMPUTE_FLOAT*`
  = compute; images use `RI_F`/`WI_F` + `CL_RGBA` + `fpType()` (CL_HALF_FLOAT unless
  BackendConfig::Precision_High).

## #1 gotcha: opencl_codegen.py is manual

`build_artifacts.sh` does **NOT** run `opencl_codegen.py` (the CMake hook is
commented out in `source/backend/opencl/CMakeLists.txt`). After editing any `.cl`
under `source/backend/opencl/execution/cl/`, run from repo root:

```bash
( cd source/backend/opencl/execution/cl && python3 opencl_codegen.py . )
git diff --check
```

This regenerates each `<name>_mnn_cl.cpp` plus `opencl_source_map.hpp`. Forgetting
this = "build succeeded but my .cl change had no effect."

## Build + sync (OrangePi / Mali)

```bash
JOBS=$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 )) \
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

file .cache/output/mnn/artifacts/orangepi5plus/bin/pic_server \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN.so \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN_CL.so

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

## Build + sync (Rhino / Adreno) — only for Adreno-relevant steps

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

file .cache/output/mnn/artifacts/aidlux_adreno_opencl/bin/pic_server \
     .cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN.so \
     .cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN_CL.so

rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Rhino frequency **must** be set to max BEFORE measuring (the decode harness does NOT
set it; only the prefill-sweep rhino wrapper does):

```bash
ssh aidlux@192.168.101.227 'sudo bash -c "
echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo performance > /sys/devices/system/cpu/cpufreq/policy3/scaling_governor
echo performance > /sys/devices/system/cpu/cpufreq/policy5/scaling_governor
echo 124800000 > /sys/class/devfreq/3d00000.qcom,kgsl-3d0/min_freq
echo 680000000 > /sys/class/devfreq/3d00000.qcom,kgsl-3d0/max_freq"'
```

OrangePi/Mali has no frequency step.

## Correctness + profile smoke (profile ON — attribution only, NOT a TPOT number)

`MNN_PAGED_ATTENTION_PROFILE_DETAIL=1` inserts a `queue.finish()` after every sub-op;
use it only to confirm op routing and the prepare boundary, never as official latency.

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1 MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models llama3.2-3b,minicpm5-1b,qwen3-4b \
  --contexts 512 --pic-repair-tokens 0,1,3 --pic-max-tokens 2 \
  --repeats 1 --warm-repeats 0 --skip-normal \
  --run-id <step>_profile_smoke_orangepi_20260703
```

### P0 validity scan (must print NOTHING)

```bash
RUN_ID=<step>_profile_smoke_orangepi_20260703
ssh orangepi@192.168.101.113 \
  "grep -R 'decode_prepare_inside_decode=1' -n /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/${RUN_ID}/pic_server_*.log || true; \
   grep -R 'ERROR\\|target unavailable\\|async persistent PIC cache read failed' -n /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/${RUN_ID}/pic_server_*.log || true"
```

A clean run prints nothing from both greps. To confirm a new kernel actually fires,
count its profile line (replace the `op=` name):

```bash
ssh orangepi@192.168.101.113 "grep -R 'op=decode_causal_attention_hd128_transposed_k_fused_kv_gqa' -c /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/${RUN_ID}/pic_server_*.log || true"
```

Rhino log root: `/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/<run_id>/pic_server_*.log`.

## PMC scoped diagnostic run (PMC ON — hardware counters, NOT a TPOT number)

Use this only after the profile smoke has identified candidate decode kernels.
PMC controls are passed to the remote `pic_server` through
`PIC_SWEEP_SERVER_ENV_EXTRA`, the same server env mechanism used by the normal
decode experiment harness. Values with `|` must be quoted inside the env string.

```bash
RUN_ID=pmc_decode_repair_sparse_qtile_orangepi_20260703
PMC_TAG=sparse_qtile_rank
PMC_REMOTE_OUT=/mnt/ssd/code/.cache/mnn_opencl_pic/pmc/${RUN_ID}_${PMC_TAG}.jsonl
PMC_KERNEL_REGEX='decode_causal_attention_hd128_transposed_k_sparse_qtile|append_sparse_decode|decode_attention_rank'
PMC_PHASE_REGEX='attention|append|rank'

PIC_SWEEP_SERVER_ENV_EXTRA="MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1 \
MNN_PIC_PMC_PROFILE=1 \
MNN_PIC_PMC_KERNEL_REGEX='${PMC_KERNEL_REGEX}' \
MNN_PIC_PMC_PHASE_REGEX='${PMC_PHASE_REGEX}' \
MNN_PIC_PMC_LAYER='all' \
MNN_PIC_PMC_COUNTERS='default' \
MNN_PIC_PMC_STRICT_FINISH=1 \
MNN_PIC_PMC_OUTPUT='${PMC_REMOTE_OUT}' \
MNN_PIC_PMC_MAX_RECORDS=2000 \
MNN_PIC_PMC_WARMUP_SKIP=1" \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models minicpm5-1b \
  --contexts 512 --pic-repair-tokens 0,1,3,5,7 --pic-max-tokens 16 \
  --repeats 1 --warm-repeats 1 --skip-normal \
  --run-id "${RUN_ID}"
```

After the run, inspect the remote JSONL:

```bash
ssh orangepi@192.168.101.113 \
  "test -s ${PMC_REMOTE_OUT} && tail -n 5 ${PMC_REMOTE_OUT}"
```

Rules:

- Keep PMC raw JSONL under `/mnt/ssd/code/.cache/mnn_opencl_pic/pmc/`.
- Do not merge PMC-scoped latency into formal TPOT or `benchmark.csv`.
- Record the exact `PIC_SWEEP_SERVER_ENV_EXTRA` in the hour's `context.md`, because
  current `run_decode_experiment.py` does not persist it in `run_config.json`.
- If PMC support is unavailable, the run should emit `pmc_status=unsupported` and
  remain functionally valid.

## Formal TPOT A/B (profile OFF — this is the per-step effectiveness number)

The harness forwards `PIC_SWEEP_SERVER_ENV_EXTRA` into the remote `pic_server`
process. `run_config.json` does **not** record this env, so encode old-vs-new in the
`--run-id` name (matching the `decode_ab_old_*` / `decode_ab_new_*` convention).

```bash
# OLD baseline (once per device; reuse across steps if artifact unchanged):
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=0' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models llama3.2-3b,minicpm5-1b,qwen3-4b \
  --contexts 512 --pic-repair-tokens 0,1,3,5,7 --pic-max-tokens 16 \
  --repeats 2 --warm-repeats 1 --skip-normal \
  --run-id decode_ab_old_current_orangepi_20260703

# NEW (this step), with step-specific env appended:
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1 <step_env>' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models llama3.2-3b,minicpm5-1b,qwen3-4b \
  --contexts 512 --pic-repair-tokens 0,1,3,5,7 --pic-max-tokens 16 \
  --repeats 2 --warm-repeats 1 --skip-normal \
  --run-id <step>_new_orangepi_20260703
```

### Reading the result

- TPOT column: `tpot_ms` in `decode_tpot_long.csv`.
- Wide CSV `decode_tpot_wide.csv` has per-q columns `pic_r{1,3,5,7}_tpot_ms` and
  `normal_x{1,2,4,6,8}_tpot_ms`. `pic-repair-tokens` maps to effective decode
  `q = repair_tokens + 1` (0→1, 1→2, 3→4, 5→6, 7→8).
- `failures.csv` rows (`device,model,stage,error`) must be empty / not generated.
- Output dir: `.cache/mnn-pic-benchmark/decode_experiment/<run_id>/`.
- Compare `<step>_new` vs `decode_ab_old_current` per `(device, model, q)`.
  A step is **effective** iff `new_ms <= old_ms` for the targeted shapes AND the P0
  validity scan is clean.

### CLI args of run_decode_experiment.py (key ones)

`--devices` (jetson,orangepi,rhino), `--models` (minicpm5-1b,llama3.2-3b,qwen3-4b),
`--contexts` (512,1024,1536), `--pic-repair-tokens` (1,3,5,7 → q=2,4,6,8; add 0 for
q=1), `--pic-max-tokens` (16), `--repeats`, `--warm-repeats`, `--skip-normal`,
`--run-id`, `--attention-layer-idx` (1 = score_layer_idx).

## Fixed device cache paths (do not fork)

- OrangePi runtime/autotune cache:
  `/mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/runtime_cache/opencl/mnn_cachefile.bin`
- Rhino runtime/autotune cache:
  `/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/runtime_cache/opencl/mnn_cachefile.bin`
- Both are device-level shared, not per-model/per-context. Warm first, then formal
  measure with profile off.

## Per-step effectiveness gate (summary)

1. `opencl_codegen.py` → `git diff --check` → `build_artifacts.sh` → `file` check → `rsync`.
2. Profile smoke (profile on) → P0 validity scan prints nothing → new `op=` line fires.
3. Optional PMC diagnostic run: `pmc_*` run id, hardware-counter JSONL under the
   remote `.cache`, no formal TPOT conclusion from this run.
4. Formal TPOT A/B (profile off): `<step>_new` `tpot_ms` ≤ `decode_ab_old_current`
   `tpot_ms` for targeted shapes; record the table in the hour's `context.md`.
5. Rhino-only steps: set `cpu=max,gpu=max,ddr=max` before measuring.
6. A step that does not pass its gate is **not promoted**; tune its parameters
   (lane / k_tile / format) as explicit env variants before declaring the step done.
