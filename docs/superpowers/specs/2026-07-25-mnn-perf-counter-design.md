# MNNPerfCounter Design

## Goal

Add an optional `MNNPerfCounter` library under `3rd_party` and expose one
uniform API to `replay_benchmark`. The library must automatically detect the
GPU platform and product, select the matching counter definitions, and measure
only an explicitly selected replayed operator. Existing benchmark and replay
flows must remain unchanged when counter output is not requested.

## Scope

The first implementation targets Android/Linux AArch64 GPU devices using the
Google `HardwarePerfCounter` low-level C implementation as the runtime source.
It supports the Adreno and Mali paths that the upstream library can identify.
The ARM `libGPUCounters` project is used as the reference for Mali counter
semantics, naming, and future model coverage, but its full HWCpipe/device C++
runtime is not vendored in the first version. This keeps the optional binary
small and avoids adding a second Mali driver implementation before a concrete
device requires it.

The following are out of scope for version 1:

- CPU PMU counters;
- modifying MNN core or backend execution scheduling;
- recording counter data in `record.json`;
- per-kernel OpenCL event counters;
- exposing vendor enum values in the replay benchmark command line;
- treating GPU counters as process-exclusive or kernel-exclusive values.

## Design alternatives

### Selected: thin MNN wrapper over Google HardwarePerfCounter

Vendor the required Google C sources and headers, then put all platform and
product selection behind `MNNPerfCounter`. The replay benchmark depends only on
the wrapper. This gives one C++11-facing API, supports both requested GPU
families, preserves MNN's no-exception/no-RTTI constraints, and keeps the
feature link-optional.

### Not selected: ARM HWCpipe for Mali plus Google HPC for Adreno

This offers a newer Mali driver path and richer derived counter metadata, but
adds the full C++14 HWCpipe/device backend, a second driver implementation, and
a larger maintenance surface. It remains a planned backend extension when a
target device needs it.

### Not selected: expose both upstream APIs directly

This would couple `replay_benchmark` to vendor-specific APIs and counter
identifiers. It also makes output and error handling inconsistent across GPU
families.

## Components and boundaries

### `3rd_party/MNNPerfCounter`

This directory contains the MNN wrapper and the vendored Google implementation
needed by the wrapper:

```text
3rd_party/MNNPerfCounter/
├── CMakeLists.txt
├── README.md
├── LICENSE                       # MNN wrapper license
├── NOTICE                        # upstream attribution and licenses
├── include/MNNPerfCounter.hpp   # stable wrapper API
├── src/MNNPerfCounter.cpp       # detection, profiles, lifecycle, errors
└── upstream/hardware_perfcounter/
    ├── include/
    ├── lib/gpu/adreno/
    ├── lib/gpu/mali/
    └── third_party/linux/
```

The vendored files retain their upstream copyright headers. A version file or
README entry records the exact upstream commit and the update procedure. The
wrapper must not define namespace-scope dynamically initialized STL objects.

The CMake target is a static library named `MNNPerfCounter`. It is created only
when `MNN_REPLAY_ENABLE_PERFCOUNTER=ON`; it is linked only by
`replay_benchmark.out`, never by the normal MNN library.

### Unified API

The wrapper owns all vendor-specific details. The API is intentionally narrow:

```cpp
namespace MNN {
namespace PerfCounter {

enum class GpuVendor {
    Unknown,
    Mali,
    Adreno,
};

struct DeviceInfo {
    GpuVendor vendor;
    uint32_t productId;
    uint32_t series;
    const char* productName;
    const char* driverName;
};

struct CounterSpec {
    const char* name;
};

struct CounterValue {
    const char* name;
    uint64_t value;
};

class Session {
public:
    static Session* create(const CounterSpec* counters, size_t count,
                           DeviceInfo* device, const char** error);
    ~Session();

    bool start();
    bool stop(CounterValue* values, size_t count);
    const char* error() const;
};

} // namespace PerfCounter
} // namespace MNN
```

The implementation may use private `std::vector`/`std::string` storage, but
the public lifecycle has explicit status returns and does not throw. The
caller owns the counter name and output arrays; the session owns device handles
and all vendor contexts. `create` performs GPU detection and model-specific
counter translation before returning a usable session.

### Platform and product selection

`Session::create` runs a deterministic auto-detection flow:

1. Probe the GPU driver path supported by the requested runtime. For the
   current replay integration, only an OpenCL replay can request GPU counters.
2. Identify the vendor: Adreno or Mali.
3. Read the product identifier through the vendor driver interface.
4. Map the identifier to a Google HPC series/layout and a MNN counter profile.
5. Translate normalized counter names into the corresponding upstream enum
   values and create the vendor context.

The profile table is the only place that maps normalized names to vendor
counter IDs. It must include a generic profile for each recognized family and
can add product-specific overrides when a GPU model changes counter layout.
Unknown models return a clear `unsupported device/profile` error rather than
silently interpreting a counter with the wrong layout. Device information is
written to the result JSON so a counter result is reproducible.

The initial normalized profiles use names based on the common semantics shared
by the two references, such as GPU activity, shader/compute activity,
instruction counts, and external/L2 memory traffic. The wrapper may report a
counter as unsupported when the selected GPU family does not provide that
semantic.

## Replay integration

Add `perfCounterOutput` and the associated internal counter options to
`replay_benchmark/ReplayRecord.hpp`. Extend argument parsing with:

```text
--perf-counter-output <path.json>
```

This option is valid only for replay mode with a selected operator. It does not
enable recording, does not affect directory discovery, and does not activate
the library for normal model benchmark runs.

The execution flow in `ReplayRunner.cpp` is:

```text
select one op record
  -> create isolated Runtime / Backend / Execution
  -> if perf output requested: create MNNPerfCounter session
  -> start counters
  -> backend onExecuteBegin
  -> execution onExecute
  -> backend onExecuteEnd
  -> synchronize the GPU output/queue
  -> stop and query counters
  -> compare output tensor
  -> write separate perf-counter JSON
```

The counter session is created only for the selected op and is destroyed on all
success and failure paths. OpenCL synchronization must complete the queued GPU
work before `stop` reads the counters. Tensor comparison remains the source of
correctness status; counter collection is an independent measurement concern.
The measured scope is explicitly reported as `gpu_device_interval`, because
other GPU work may be present on the same device.

If session creation or sampling is unavailable, the output file is still
written with an `unavailable` or `error` status and the diagnostic. The replay
operation's correctness result is not converted into a pass because counters
were unavailable, and normal replay without the option has no new failure mode.

## Perf-counter JSON

The output is independent of `record.json` and uses a versioned format:

```json
{
  "format": "mnn-perf-counter",
  "version": 1,
  "status": "ok",
  "model": "model.mnn",
  "op": {
    "op_id": 108,
    "name": "conv_3",
    "op_type": "Convolution",
    "execution": "ConvBufExecution",
    "variant": "conv_1x1"
  },
  "runtime": {
    "backend": "OPENCL",
    "vendor": "Adreno",
    "product_id": 630,
    "product_name": "Adreno",
    "driver": "google-hardware-perfcounter",
    "scope": "gpu_device_interval",
    "synchronized": true
  },
  "measurement": {
    "start_ns": 0,
    "end_ns": 0,
    "duration_ns": 0
  },
  "counters": [
    {"name": "gpu_active_cycles", "value": 1234}
  ],
  "error": ""
}
```

`start_ns` and `end_ns` use a monotonic clock and are diagnostic metadata, not
GPU timestamps. Counter values are deltas from the session start. For
unsupported CPU/driver/device cases, `counters` is empty and `error` explains
the reason. JSON writing must use the existing RapidJSON dependency and create
parent directories only for the requested output path.

## Build and configuration

Add:

```cmake
option(MNN_REPLAY_ENABLE_PERFCOUNTER
       "Enable optional GPU performance counters for replay_benchmark" OFF)
```

When enabled together with `MNN_BUILD_BENCHMARK`, CMake adds
`3rd_party/MNNPerfCounter`, defines `MNN_REPLAY_HAS_PERFCOUNTER` for the replay
target, and links `MNNPerfCounter` only into `replay_benchmark.out`. The normal
default remains off, so host builds and deployments that do not need counters
do not gain the dependency or binary size.

The vendored source is compiled with MNN-compatible C11/C++11 settings, no
exceptions, no RTTI, and position-independent code where required by the
parent build. Unsupported operating systems compile the wrapper's unavailable
backend only when the target is explicitly enabled; they do not attempt to
access Linux GPU device nodes.

## Testing and validation

Tests are layered:

1. Host unit tests cover normalized counter profile lookup, unknown vendor or
   product errors, JSON status output, and lifecycle error handling using a
   fake backend seam; no GPU device is required.
2. A host CMake build verifies that the feature is absent when disabled and
   that the optional target links when enabled.
3. A cross-compiled AArch64 build verifies that `replay_benchmark.out` and the
   static library contain the expected symbols and architecture.
4. On a Mali and an Adreno device, run one known replayed op with
   `--perf-counter-output`, verify `status: ok`, device/vendor identification,
   non-empty values, and output correctness.
5. Run the same replay without the option and verify that no perf-counter file
   is created and no counter device is opened.
6. Run unsupported CPU, unknown model, and insufficient-permission cases and
   verify explicit JSON diagnostics without changing ordinary replay behavior.

The final device validation must report the exact GPU product, selected
profile, op ID, output status, and any driver limitation. Counter results must
not be presented as process-exclusive measurements.

## Risks and mitigations

- **Kernel/driver variation:** detect product and layout before enabling any
  counter; reject unknown combinations.
- **Permissions:** preserve the driver error in JSON and keep ordinary replay
  usable without the feature.
- **Asynchronous OpenCL execution:** force queue/output synchronization before
  reading the final counter delta.
- **Global GPU activity:** label the measurement scope and avoid claiming
  kernel exclusivity.
- **Upstream drift:** record exact Google HPC commit, retain notices, and keep
  all vendor mappings in the wrapper profile table.
