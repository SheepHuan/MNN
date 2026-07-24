# OpenCL PMU Baseline Benchmark Design

## Goal

Add a self-contained OpenCL microbenchmark mode to `replay_benchmark`. The
mode provides deterministic workloads for validating whether GPU PMU metrics
respond to known buffer, image, texture, memory, and arithmetic activity. It
is a diagnostic mode for `MNNPerfCounter`; it does not change normal model
recording or single-op replay behavior.

The benchmark must cover both OpenCL buffer and image paths, including texture
sampling, and must exercise L1/L2 locality, external memory traffic, cache
filling, integer arithmetic, FP16/FP32 arithmetic, and supported OpenCL
extensions.

## Scope

The feature is enabled explicitly with a new benchmark command. Normal
`replay_benchmark` execution remains unchanged unless this mode is selected.
The first implementation targets OpenCL devices and uses the existing
`MNNPerfCounter` backend for Adreno and Mali when available.

Supported workload families:

- Buffer: sequential read, sequential write, read-modify-write, small-set
  reuse, strided access, random access, and large streaming access.
- Image: 2D local reads, row/column stride, tiled reads, random reads, image
  writes, and read-modify-write for supported image formats.
- Texture: `sampler_t` reads with nearest and linear filtering, normalized and
  unnormalized coordinates, local sampling, cross-row sampling, and random
  sampling.
- Compute: integer, FP16, FP32, vector arithmetic, local/private/constant
  memory, barriers, and atomics when supported.
- Extensions: detect and run extension-specific cases only when the device
  advertises the required extension; skipped cases include the extension and
  reason in the report.

The benchmark does not claim that a vendor counter name maps exactly to a
physical L1 or L2 cache. It reports the observed PMU signal and the expected
workload relationship, allowing the caller to determine which counters are
responsive and discriminative on a particular driver/GPU combination.

## Command-line interface

Add an explicit mode such as:

```text
replay_benchmark.out --opencl-pmu-bench
    [--opencl-pmu-case name|all]
    [--opencl-pmu-iterations N]
    [--opencl-pmu-size bytes]
    [--perf-counter-events name1,name2,...|auto]
    [--perf-counter-output report.json]
```

`all` runs the supported built-in cases. A single case is useful when
debugging one event or one cache path. `auto` selects the normalized default
profile for the detected GPU; explicit event lists remain available for A7xx
event validation. The mode requires OpenCL and does not require an MNN model or
an execution record.

## Components

Add a focused benchmark component under `replay_benchmark/`:

```text
replay_benchmark/
├── README.md
├── OpenCLPmuBenchmark.hpp
├── OpenCLPmuBenchmark.cpp
└── OpenCLPmuKernels.hpp
```

`replay_benchmark/README.md` documents the OpenCL concepts exercised by the
suite: global, local, private, and constant address spaces; work-items,
work-groups, sub-groups, barriers, atomics, memory ordering, buffer/image
objects, samplers, texture filtering, cache locality, and optional
extensions. Each benchmark case links its workload to the concept and the
PMU signal it is intended to probe.

`OpenCLPmuBenchmark` owns case discovery, device capability checks, buffer and
image allocation, kernel compilation, execution, synchronization, PMU
session lifecycle, and report assembly. Kernel source is compiled into the
executable as constant character data so the diagnostic mode does not depend
on external files.

The component reuses the existing OpenCL symbol-loading path and
`MNNPerfCounter::Session`. It must not add a dependency from the normal MNN
runtime library to the benchmark code.

## Workload semantics

Each case declares:

- data path: buffer, image, or texture;
- element format and required extension;
- working-set size and access pattern;
- read/write ratio;
- arithmetic operation and type;
- expected PMU signal direction;
- repeat count and synchronization policy.

The baseline cases use at least three working-set regimes:

1. `reuse_small`: repeatedly access a small region to favor L1 reuse;
2. `reuse_l2`: exceed the likely L1 capacity while fitting the selected L2
   working set as far as practical;
3. `stream_large`: exceed cache capacity and force sustained external traffic.

The report names these as workload hypotheses, not hard-coded cache sizes.
Cache sizes are not assumed because they vary by product, driver, and image or
buffer path.

## PMU measurement and validity analysis

For every case:

1. Query device capabilities and mark unsupported formats/extensions as
   `skipped`.
2. Run a synchronized empty-kernel control to estimate interval noise.
3. Start one PMU session for a physical-slot-sized event batch.
4. Execute the fixed workload and call `queue.finish()` before stopping PMU.
5. Record counter deltas and duration.
6. Repeat the control/workload pair when requested to reduce scheduling noise.

Events are classified independently:

- `readable`: session creation, activation, synchronization, and read succeed;
- `responsive`: workload delta exceeds the control noise threshold;
- `discriminative`: workload delta is measurably different from the control;
- `valid`: the signal direction matches the case hypothesis;
- `unavailable`: driver, permission, or ioctl failure;
- `skipped`: OpenCL capability or required extension is absent.

Zero delta is retained as data. It is not automatically treated as an ioctl
failure. A report also records whether the result is a device-wide interval;
it does not claim process-exclusive or kernel-exclusive attribution.

When `auto` or an event profile contains more events than the hardware has
slots, the benchmark runs multiple synchronized passes. Every pass records
the exact event names, product, driver, case, iteration count, and runtime
capabilities so results from different passes remain comparable.

## Report format

Use a versioned `mnn-opencl-pmu-benchmark` JSON document containing:

```json
{
  "format": "mnn-opencl-pmu-benchmark",
  "version": 1,
  "status": "ok",
  "runtime": {
    "backend": "OPENCL",
    "vendor": "Adreno",
    "product_id": 740,
    "extensions": [],
    "synchronized": true
  },
  "cases": [
    {
      "name": "buffer_stream_large",
      "path": "buffer",
      "status": "ok",
      "iterations": 100,
      "duration_ns": 0,
      "counters": [
        {
          "name": "uche_vbif_read_beats_sp",
          "delta": 0,
          "control_delta": 0,
          "signal": "responsive",
          "valid": true
        }
      ]
    }
  ],
  "errors": []
}
```

The implementation may extend the schema, but must preserve the distinction
between case status, counter readability, counter signal, and OpenCL support
status.

## Testing

Host tests cover command parsing, case metadata, JSON schema construction,
capability filtering, and validity classification using synthetic counter
samples. Existing `MNNPerfCounter` tests remain unchanged.

On Rhinopi, validation runs at least:

- buffer reuse, L2-sized reuse, and large streaming cases;
- image and texture cases for every supported image format;
- integer, FP16, and FP32 cases;
- each advertised OpenCL extension case;
- explicit A7xx event batches, including CP/RBBM/HLSQ/UCHE/TP/SP/RB groups;
- repeated runs to separate stable signals from scheduling noise.

The validation summary reports total, readable, responsive, discriminative,
valid, skipped, and unavailable event results. A successful report generation
does not imply that every event is valid.

## Constraints

- Keep the feature inside `replay_benchmark`; do not modify restricted MNN
  directories or introduce a normal-runtime dependency.
- Preserve C++11 compatibility, no RTTI, no exceptions, and no namespace-scope
  dynamically initialized STL objects.
- Keep the existing default replay path and output format unchanged.
- Do not infer physical cache sizes or declare a counter valid solely because
  its ioctl call returned success.
