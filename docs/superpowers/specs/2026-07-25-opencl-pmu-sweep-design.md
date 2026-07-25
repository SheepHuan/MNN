# OpenCL PMU Single-Event Sweep Design

## Goal

Measure one GPU PMU event at a time around a prepared OpenCL workload and emit
a reproducible CSV. The same workload and measurement model applies to the
Adreno A7xx Rhinopi and Mali OrangePi devices; event availability is reported
per device rather than inferred from another architecture.

## Measurement model

The control and workload values come from separate PMU sessions.
`control_delta` is the count during matched control work;
`workload_delta` is the count during the prepared workload;
`delta_metric = workload_delta - control_delta` is signed. Inner
`iterations` controls the kernel loop, outer `workload_runs` controls
consecutive dispatches inside one workload session, and `measurements` repeats
independent one-event sessions.

Warmup dispatches execute before both sampled sessions. Program, kernel,
buffer, image, sampler, constant-table, pointer-chain, and local-memory
preparation also occur before sampling. A workload session contains only
prepared dispatches and queue completion.

## Single-event invariant

The sweep invokes `replay_benchmark.out` once for each case, event, and
measurement, passing exactly one name to `--perf-counter-events`. It does not
use physical-slot batching for sweep conclusions. The benchmark itself may
batch a manually requested profile, but the sweep never passes a comma-separated
event list.

## CSV contract

The header is exactly:

```text
case_name,case_args,case_runs,pmu_metric_name,delta_metric
```

`case_runs` is `workload_runs`. `case_args` records size, local size, inner
iterations, workload runs, warmups, and independent measurement number.
Failed or unsupported combinations retain a row with an empty `delta_metric`
and a diagnostic in stderr/raw JSON. An empty value is not converted to zero.

## Workload preparation

Every case prepares its OpenCL objects before PMU sampling. Image inputs are
deterministically initialized. Buffer/compute global sizes are rounded to the
selected local size with guarded writes; image workloads report explicit
two-dimensional width and height. Pointer chase tables, constant tables,
local allocations, and atomic target counts are included in the case metadata.

## ArchProbe relationship

ArchProbe supplies timing-based cache, vector-width, bandwidth, local-memory,
and throughput hints. MNN cases map those dimensions as follows:

```text
buffer_stride / buffer_pchase -> BufferCachelineSize / BufferCacheHierarchyPChase
buffer_vec4                  -> BufferVecWidth
fp32/fp16_throughput         -> Gflops
constant_bandwidth           -> ConstMemBandwidth
local_bandwidth              -> LocalMemBandwidth
image_stride_x/y             -> ImageCachelineSize / ImageCacheHierarchyPChase
atomic_contended/distributed -> atomic contention control
```

ArchProbe output is used to choose workload parameters; it is not treated as a
PMU count. MNNPerfCounter validates whether a selected event responds to the
corresponding workload. Architectural conclusions require repeatable deltas,
not merely a readable counter.

## Analysis and evidence

Use at least three independent measurements for comparisons. Report per-case
medians and median absolute deviations, along with the number of empty or
failed rows. Preserve raw JSON, device identity, binary hashes, command line,
and event availability under the device `records/opencl-pmu-sweep/` directory.
Do not call an event discriminative when rows are missing or unstable.
