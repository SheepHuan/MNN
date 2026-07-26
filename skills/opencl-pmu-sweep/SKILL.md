---
name: opencl-pmu-sweep
description: Run reproducible single-event GPU PMU sweeps on MNN replay_benchmark, including OpenCL synthetic cases and complete OpenCL/Vulkan DNN Session workloads on Rhinopi/OrangePi, repeated measurements, raw JSON preservation, and fixed-schema CSV output.
---

# OpenCL / Vulkan PMU sweep

Use `scripts/sweep_opencl_pmu.py` to run one benchmark process for every
event/measurement combination. Synthetic cases are OpenCL-only. Complete MNN
model sessions can select OpenCL or Vulkan with `--backend opencl|vulkan`; the
selected backend is passed to `replay_benchmark` as forward 3 or 7. By default,
selected cases are passed together to that process; the benchmark still
creates a separate control/workload PMU session for each case, and the script
expands the result into one CSV row per case/event/measurement combination. The
default `--events all` first asks the selected device to enumerate its complete
supported PMU event set, then tests every discovered event against every
selected case. Never pass a
comma-separated event list to one benchmark invocation: the script enforces
one `--perf-counter-events` value per process so physical PMU slot scheduling
and session boundaries stay unambiguous.

Rhinopi and OrangePi are separate experiments. Run the script once per
device, use separate detailed/raw/summary paths, and never union their event
lists or combine their validity counts. A device's summary contains every
event discovered on that device, including events that remain invalid for all
cases.

Each invocation writes both the detailed CSV and a metric summary CSV. The
summary has `pmu_metric_name,valid,valid_cases`; `valid` is true when at least
one case has a nonzero `delta_metric`, and `valid_cases` lists those cases
separated by semicolons. An event with no nonzero delta in any case is invalid.

To get the valid-metric union for one device across OpenCL and Vulkan, run:

```bash
python3 skills/opencl-pmu-sweep/scripts/merge_backend_valid_metrics.py \
  --opencl-metrics records/opencl-pmu-sweep/orangepi-all-metrics.csv \
  --vulkan-metrics records/opencl-pmu-sweep/orangepi-vulkan-model-all-metrics.csv \
  --output-csv records/opencl-pmu-sweep/orangepi-opencl-vulkan-valid-metrics.csv
```

Run the same command with the Rhinopi paths for the separate Rhinopi report.
The output contains `category` (`compute`, `cache`, `memory`, or `other`),
per-backend validity and valid cases, and `valid_backends`. Cache markers take
precedence over memory markers, and memory markers take precedence over compute
markers; this keeps explicit L1/L2/cache metrics classified as cache.

To maintain one device-level report containing both synthetic PMU cases and
model workloads, merge the detailed CSVs after each sweep:

```bash
python3 skills/opencl-pmu-sweep/scripts/merge_opencl_pmu_csv.py \
  --input-csv records/opencl-pmu-sweep/rhinopi-all.csv \
  --input-csv records/opencl-pmu-sweep/rhinopi-model-smoke.csv \
  --input-csv records/opencl-pmu-sweep/rhinopi-llm-smoke.csv \
  --output-csv records/opencl-pmu-sweep/rhinopi-all.csv \
  --summary-csv records/opencl-pmu-sweep/rhinopi-all-metrics.csv
```

Repeat independently for OrangePi. The merged detailed report keeps the
five-column `all.csv` contract; model path, status, and error metadata are
appended to `case_args`, while the original model CSV remains the authoritative
status/error audit record. The merged `valid_cases` therefore includes both
synthetic PMU case names and model workload case names, without combining
Rhinopi and OrangePi results.

Use `--model-sweep` for complete MNN model Sessions. Every discovered `.mnn`
model is run against every selected PMU event, again with exactly one event per
process. Ordinary models use MNN `Interpreter`/`Session`; a model package with
`llm_config.json` uses the MNN LLM engine and measures a fixed prefill plus one
decode token. Model loading, graph compilation, input preparation, tokenizer,
and warmup happen outside the PMU interval. The model CSV adds
`model_path,status,error` to the normal case columns:

```text
case_name,model_path,case_args,case_runs,pmu_metric_name,delta_metric,status,error
```

`--backend vulkan` currently applies to ordinary DNN `.mnn` models. LLM
packages remain OpenCL-only because the MNN LLM runtime still constructs its
runtime configuration with `backend_type=opencl`; Vulkan LLM rows are retained
as unavailable rather than silently falling back to CPU or OpenCL.

Before a device sweep, verify that `bin/replay_benchmark.out`,
`bin/opencl_pmu_benchmark_test`, `lib/libMNN.so`, `lib/libMNN_CL.so`, and
`lib/libMNN_Express.so` belong to the same build. Record SHA-256 hashes and the
device kernel/GPU identity with the sweep evidence.

## Quick start

From the MNN repository:

```bash
python3 skills/opencl-pmu-sweep/scripts/sweep_opencl_pmu.py \
  --device rhinopi \
  --cases buffer_stride,buffer_pchase,buffer_vec4,fp32_throughput,fp16_throughput \
  --events all \
  --measurements 3 \
  --workload-runs 5 \
  --warmup-runs 2 \
  --output-csv records/opencl-pmu-sweep/rhinopi-all.csv \
  --keep-json-dir records/opencl-pmu-sweep/rhinopi-all-raw
```

OrangePi uses `--device orangepi`. The built-in device roots are:

- Rhinopi: `root@192.168.101.227`, `/mnt/nvme/workspace/replay-benchmark`
- OrangePi: `root@192.168.101.113`, `/mnt/ssd/workspace`

Use `--remote-root` when the binary is deployed elsewhere. The remote layout
must contain `bin/replay_benchmark.out` and `lib/`.

The discovery command used by the script is equivalent to
`replay_benchmark.out --opencl-pmu-list-events`. Its JSON result is
device-local and must not be cached from one device and reused for the other.

## Measurement rules

- Keep `--measurements >= 3` for comparisons. Each measurement starts a new
  benchmark process and therefore new control/workload PMU sessions.
- Keep `--workload-runs >= 3` so one workload session contains repeated
  dispatches. `--iterations` is the inner kernel loop and is independent of
  `--workload-runs`.
- Use `--warmup-runs` before sampling. Warmups are outside both PMU sessions.
- With `--events all`, execute the full Cartesian product:
  `selected_cases × discovered_events × measurements`. Do not reduce this to
  the representative profile unless the user explicitly passes an event list.
  Keep one event per benchmark process. Use `--no-batch-cases` only when a
  separate process per case is needed for debugging.
- Sweep one variable at a time: case, event, size, or local size. Record those
  values in the command line or output directory.
- Preserve the generated raw JSON directory. Inspect `status`, `error`,
  `pmu_measured`, `sampled_workload_dispatches`, `global_size`, and
  `local_size` before interpreting deltas.
- Save CSV, raw JSON, command parameters, hashes, and failed-row diagnostics
  under `records/opencl-pmu-sweep/` when running on a device.

## CSV contract

The CSV header is exactly:

```text
case_name,case_args,case_runs,pmu_metric_name,delta_metric
```

The summary CSV header is exactly:

```text
pmu_metric_name,valid,valid_cases
```

Pass `--summary-csv` to choose its path; otherwise it is written beside the
detailed output as `<output-stem>-metrics.csv`.

Use separate output families for the two devices:

```text
records/opencl-pmu-sweep/rhinopi-all.csv
records/opencl-pmu-sweep/rhinopi-all-metrics.csv
records/opencl-pmu-sweep/orangepi-all.csv
records/opencl-pmu-sweep/orangepi-all-metrics.csv
```

Do not create a combined `all-devices.csv`; device identity is an experiment
boundary, not a CSV grouping field.

`delta_metric` is the signed value:

```text
workload_delta - control_delta
```

An empty `delta_metric` means the report did not contain a readable matching
counter; keep that row for auditability rather than converting it to zero.
`case_runs` is the `workload_runs` value. `case_args` records size, local size,
iterations, workload runs, warmups, and measurement number.

## Local execution

For a locally running binary, omit `--device`, pass `--binary`, and optionally
pass `--lib-dir`:

```bash
python3 skills/opencl-pmu-sweep/scripts/sweep_opencl_pmu.py \
  --binary ./build/replay_benchmark.out \
  --lib-dir ./build \
  --cases buffer_fp32 \
  --events gpu_active_cycles \
  --output-csv records/local.csv
```

Do not put device passwords in commands, scripts, CSV, or logs. The script
uses `ssh` in `BatchMode` and relies on the existing SSH key setup.

## Model execution

The repository's model roots can be swept together; the output must remain
device-specific:

```bash
python3 skills/opencl-pmu-sweep/scripts/sweep_opencl_pmu.py \
  --model-sweep \
  --backend opencl \
  --device rhinopi \
  --model-roots benchmark/models,/path/to/prebuilts/models/DNNs,/path/to/models/bench_genais/mnn \
  --events all \
  --measurements 3 \
  --workload-runs 5 \
  --warmup-runs 2 \
  --control-runs 1 \
  --model-output-csv records/opencl-pmu-sweep/rhinopi-model-all.csv \
  --model-raw-dir records/opencl-pmu-sweep/rhinopi-model-all-raw
```

For a remote device, copy ordinary DNN models and LLM packages to their actual
remote roots. Use `--remote-dnn-root` for the directory containing ordinary
`.mnn` files and `--remote-llm-root` for the directory containing LLM package
subdirectories. Package files that need LLM weights must preserve their `.mnn`,
`.mnn.weight`, tokenizer, and `llm_config.json` siblings. The older
`--remote-model-root` remains as a common-root fallback. Use separate model
output/raw/summary paths for OrangePi.

For a Vulkan model sweep, use a separate output family and retain the same
full event Cartesian product:

```bash
python3 skills/opencl-pmu-sweep/scripts/sweep_opencl_pmu.py \
  --model-sweep \
  --backend vulkan \
  --device orangepi \
  --model-roots /path/to/DNNs \
  --events all \
  --measurements 3 \
  --workload-runs 5 \
  --warmup-runs 2 \
  --control-runs 1 \
  --model-output-csv records/opencl-pmu-sweep/orangepi-vulkan-model-all.csv \
  --model-summary-csv records/opencl-pmu-sweep/orangepi-vulkan-model-all-metrics.csv \
  --model-raw-dir records/opencl-pmu-sweep/orangepi-vulkan-model-all-raw
```

Do not merge Vulkan results into the OpenCL synthetic-case CSV. If a combined
device report is needed later, merge only reports with the same backend and
preserve the backend in the output directory and file name.

## Validation

Run the bundled unit tests after changing the script:

```bash
python3 -m unittest discover -s skills/opencl-pmu-sweep/tests -p 'test_*.py'
python3 /home/yanghuan/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/opencl-pmu-sweep
```
