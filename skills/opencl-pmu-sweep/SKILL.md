---
name: opencl-pmu-sweep
description: Run reproducible single-event OpenCL GPU PMU sweeps on MNN replay_benchmark, including Rhinopi/OrangePi SSH execution, repeated workload measurements, raw JSON preservation, and fixed-schema CSV output. Use when comparing PMU metrics across OpenCL cases, local sizes, working sets, or Android GPU devices.
---

# OpenCL PMU sweep

Use `scripts/sweep_opencl_pmu.py` to run one benchmark process for every
case/event/measurement combination. Never pass a comma-separated event list to
one benchmark invocation: the script enforces one `--perf-counter-events`
value per process so physical PMU slot scheduling and session boundaries stay
unambiguous.

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
  --events gpu_active_cycles \
  --measurements 3 \
  --workload-runs 5 \
  --warmup-runs 2 \
  --output-csv records/rhinopi-opencl-pmu.csv
```

OrangePi uses `--device orangepi`. The built-in device roots are:

- Rhinopi: `root@192.168.101.227`, `/mnt/nvme/workspace/replay-benchmark`
- OrangePi: `root@192.168.101.113`, `/mnt/ssd/workspace`

Use `--remote-root` when the binary is deployed elsewhere. The remote layout
must contain `bin/replay_benchmark.out` and `lib/`.

## Measurement rules

- Keep `--measurements >= 3` for comparisons. Each measurement starts a new
  benchmark process and therefore new control/workload PMU sessions.
- Keep `--workload-runs >= 3` so one workload session contains repeated
  dispatches. `--iterations` is the inner kernel loop and is independent of
  `--workload-runs`.
- Use `--warmup-runs` before sampling. Warmups are outside both PMU sessions.
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

## Validation

Run the bundled unit tests after changing the script:

```bash
python3 -m unittest discover -s skills/opencl-pmu-sweep/tests -p 'test_*.py'
python3 /home/yanghuan/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/opencl-pmu-sweep
```
