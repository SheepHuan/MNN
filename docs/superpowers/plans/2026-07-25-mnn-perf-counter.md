# MNNPerfCounter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional MNNPerfCounter library that auto-detects Mali/Adreno GPU models and lets replay_benchmark write independent per-op counter JSON on request.

**Architecture:** Vendor the required Google HardwarePerfCounter C sources under `3rd_party/MNNPerfCounter/upstream`, wrap them with a small C++11 no-exception API that selects normalized counter profiles from the detected vendor/product, and link the target only to replay_benchmark when explicitly enabled. Keep JSON serialization and replay lifecycle orchestration in replay_benchmark so the library remains independent of MNN record format.

**Tech Stack:** C11/C++11, CMake, Google HardwarePerfCounter Apache-2.0 sources, MNN Backend/Tensor APIs, RapidJSON, assert-based host tests, AArch64 cross-build.

---

## File map

Create:

- `3rd_party/MNNPerfCounter/CMakeLists.txt`: optional static target and source lists.
- `3rd_party/MNNPerfCounter/README.md`: upstream commit, supported drivers, profile names, update procedure.
- `3rd_party/MNNPerfCounter/LICENSE`: MNN wrapper license text.
- `3rd_party/MNNPerfCounter/NOTICE`: Google and Linux ioctl attribution.
- `3rd_party/MNNPerfCounter/include/MNNPerfCounter.hpp`: stable unified API.
- `3rd_party/MNNPerfCounter/src/MNNPerfCounter.cpp`: detection, product mapping, profile translation, lifecycle, errors.
- `3rd_party/MNNPerfCounter/test/MNNPerfCounterTest.cpp`: deterministic profile and unavailable-backend tests.
- `3rd_party/MNNPerfCounter/upstream/hardware_perfcounter/...`: exact required Google HPC snapshot.
- `replay_benchmark/PerfCounterReport.hpp`: report data and serializer declaration.
- `replay_benchmark/PerfCounterReport.cpp`: versioned JSON writer and directory creation.
- `test/replay_benchmark/PerfCounterReportTest.cpp`: report schema test without a GPU.

Modify:

- `CMakeLists.txt`: add `MNN_REPLAY_ENABLE_PERFCOUNTER` and optional subdirectory wiring.
- `replay_benchmark/CMakeLists.txt`: link `MNNPerfCounter`, add report source, and define the feature macro.
- `replay_benchmark/ReplayRecord.hpp`: add `Options::perfCounterOutput` and perf-counter report inputs.
- `replay_benchmark/replay_benchmark.cpp`: parse `--perf-counter-output`, validate replay-only usage, and update help text.
- `replay_benchmark/ReplayRunner.cpp`: create the selected session, measure the isolated execution, synchronize OpenCL, and write the report on all requested paths.

Do not modify `schema/private/` or `source/internal/`.

### Task 1: Vendor the low-level source and create an optional target

**Files:**

- Create: `3rd_party/MNNPerfCounter/upstream/hardware_perfcounter/` from Google commit `7949f047e58e44996236e6ec6153cfb60951c37a`.
- Create: `3rd_party/MNNPerfCounter/CMakeLists.txt`.
- Create: `3rd_party/MNNPerfCounter/README.md`, `LICENSE`, `NOTICE`.
- Modify: `CMakeLists.txt`.
- Modify: `replay_benchmark/CMakeLists.txt`.

- [ ] **Step 1: Copy only the required Google sources and preserve notices**

Copy these upstream files without changing their headers: all files under
`include/hpc/gpu/{base_utilities.h,adreno/*.h,mali/*.h}`, all `.c`/`.h` files
under `lib/gpu/{adreno,mali}`, and `third_party/linux/{adreno_driver_ioctl.h,LICENSE,Linux-syscall-note,README.md}`. Record the exact commit and source URL in `README.md`; put Apache-2.0 and Linux ioctl attribution in `NOTICE`.

- [ ] **Step 2: Add the static target with no global feature enablement**

Use a target whose public include directory is only
`3rd_party/MNNPerfCounter/include`; keep upstream include paths private:

```cmake
add_library(MNNPerfCounter STATIC
    src/MNNPerfCounter.cpp
    upstream/hardware_perfcounter/lib/gpu/adreno/a5xx.c
    upstream/hardware_perfcounter/lib/gpu/adreno/a6xx.c
    upstream/hardware_perfcounter/lib/gpu/adreno/common.c
    upstream/hardware_perfcounter/lib/gpu/adreno/context.c
    upstream/hardware_perfcounter/lib/gpu/adreno/driver_ioctl.c
    upstream/hardware_perfcounter/lib/gpu/mali/bifrost.c
    upstream/hardware_perfcounter/lib/gpu/mali/common.c
    upstream/hardware_perfcounter/lib/gpu/mali/context.c
    upstream/hardware_perfcounter/lib/gpu/mali/driver_ioctl.c
    upstream/hardware_perfcounter/lib/gpu/mali/valhall.c)
target_include_directories(MNNPerfCounter
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include
    PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}/upstream/hardware_perfcounter/include
      ${CMAKE_CURRENT_LIST_DIR}/upstream/hardware_perfcounter/lib/gpu/adreno
      ${CMAKE_CURRENT_LIST_DIR}/upstream/hardware_perfcounter/lib/gpu/mali
      ${CMAKE_CURRENT_LIST_DIR}/upstream/hardware_perfcounter/third_party)
set_target_properties(MNNPerfCounter PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(MNN_PERFCOUNTER_BUILD_TESTS "Build MNNPerfCounter host tests" OFF)
if(MNN_PERFCOUNTER_BUILD_TESTS)
    add_executable(mnn_perfcounter_test test/MNNPerfCounterTest.cpp)
    target_link_libraries(mnn_perfcounter_test PRIVATE MNNPerfCounter)
    enable_testing()
    add_test(NAME mnn_perfcounter_test COMMAND mnn_perfcounter_test)
endif()
```

- [ ] **Step 3: Add the parent option and conditional target wiring**

Add `option(MNN_REPLAY_ENABLE_PERFCOUNTER "Enable optional GPU performance counters for replay_benchmark" OFF)` beside the other build options. Under `MNN_BUILD_BENCHMARK`, add the third-party subdirectory only when that option is on, and in `replay_benchmark/CMakeLists.txt` add `MNNPerfCounter` and `MNN_REPLAY_HAS_PERFCOUNTER` only under the same condition.

- [ ] **Step 4: Run the disabled-feature configure check**

Run:

```bash
cmake -S . -B /tmp/mnn-perfcounter-disabled -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_TEST=OFF
cmake --build /tmp/mnn-perfcounter-disabled --target replay_benchmark.out -j2
```

Expected: configure and build complete without compiling or linking `MNNPerfCounter`.

### Task 2: Implement unified auto-detection and model-aware profiles

**Files:**

- Create: `3rd_party/MNNPerfCounter/include/MNNPerfCounter.hpp`.
- Create: `3rd_party/MNNPerfCounter/src/MNNPerfCounter.cpp`.
- Create: `3rd_party/MNNPerfCounter/test/MNNPerfCounterTest.cpp`.
- Modify: `3rd_party/MNNPerfCounter/CMakeLists.txt`.

- [ ] **Step 1: Write deterministic profile tests first**

Test that the normalized names `gpu_active_cycles`, `compute_active_cycles`, `compute_tasks`, `l2_any_lookup`, `l2_ext_read`, and `l2_ext_write` map to Mali common counters, that `sp_busy_cycles`, `sp_vs_instructions`, `sp_fs_instructions`, `sp_cs_instructions`, `sp_gm_load_instructions`, and `sp_gm_store_instructions` map to Adreno common counters, and that an unknown product returns an error without creating a context. The test must use the wrapper's injected probe/context seam and must not open `/dev/mali0` or `/dev/kgsl-3d0`.

- [ ] **Step 2: Run the tests to verify the missing API fails**

Run:

```bash
cmake -S . -B /tmp/mnn-perfcounter-test -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_TEST=OFF -DMNN_REPLAY_ENABLE_PERFCOUNTER=ON -DMNN_PERFCOUNTER_BUILD_TESTS=ON
cmake --build /tmp/mnn-perfcounter-test --target mnn_perfcounter_test -j2
```

Expected: compilation fails because the wrapper API and profile seam do not exist yet.

- [ ] **Step 3: Define the no-exception unified API**

Expose the following C++11 interface. `Session::create` auto-probes the platform and product unless a test probe is supplied internally; callers only provide normalized names and receive product metadata and values:

```cpp
namespace MNN {
namespace PerfCounter {

enum class GpuVendor { Unknown, Mali, Adreno };

struct DeviceInfo {
    GpuVendor vendor = GpuVendor::Unknown;
    uint32_t productId = 0;
    uint32_t family = 0;
    const char* productName = nullptr;
    const char* driverName = nullptr;
};

struct CounterSpec { const char* name = nullptr; };
struct CounterValue { const char* name = nullptr; uint64_t value = 0; };

class Session {
public:
    static Session* create(const CounterSpec* specs, size_t count,
                           DeviceInfo* device, const char** error);
    ~Session();
    bool start();
    bool stop(CounterValue* values, size_t count);
    const char* error() const;
};

} // namespace PerfCounter
} // namespace MNN
```

- [ ] **Step 4: Implement the profile table and vendor selection**

Use Google common APIs for both families. Probe Adreno through
`hpc_gpu_adreno_ioctl_open_gpu_device` and `hpc_gpu_adreno_ioctl_get_gpu_device_id`, classify with `hpc_gpu_adreno_get_series`, then create the common Adreno context. Probe Mali through the API version/context/property calls, read `gpu_product_id`, classify with `hpc_gpu_mali_get_counter_layout`, then create the common Mali context. Close probe descriptors on every failure path. Reject unknown product IDs/layouts before activating counters.

Keep normalized names and vendor enum values in function-local/static POD tables only; do not create namespace-scope `std::string` or STL containers. `start` activates and takes the baseline, `stop` deactivates and queries deltas, and all upstream status values are copied into a stable error string stored by the session.

- [ ] **Step 5: Run the profile and host-unavailable tests to verify green**

Run the same CMake build plus `ctest --test-dir /tmp/mnn-perfcounter-test --output-on-failure`. Expected: all profile tests pass and the host without GPU device nodes reports a controlled unavailable error rather than crashing.

### Task 3: Add independent perf-counter JSON reporting

**Files:**

- Create: `replay_benchmark/PerfCounterReport.hpp`.
- Create: `replay_benchmark/PerfCounterReport.cpp`.
- Create: `test/replay_benchmark/PerfCounterReportTest.cpp`.
- Modify: `replay_benchmark/CMakeLists.txt`.

- [ ] **Step 1: Write the report schema test first**

Construct an `MNN::Replay::PerfCounterReport` with one op and one counter, write it to a temporary JSON path, parse it with RapidJSON, and assert `format == "mnn-perf-counter"`, `version == 1`, `status == "ok"`, the op ID is preserved, and the counter value is a JSON integer. Add a second case for `status == "unavailable"` with an empty counter array and a non-empty error.

- [ ] **Step 2: Run the report test to verify it fails**

Run:

```bash
cmake -S . -B /tmp/mnn-perfcounter-report-test -DMNN_BUILD_TEST=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TOOLS=OFF
cmake --build /tmp/mnn-perfcounter-report-test --target replay_perfcounter_report_test -j2
```

Expected: target or report symbols are missing.

- [ ] **Step 3: Implement the report types and serializer**

Define report fields for model, op metadata, runtime/device metadata, monotonic start/end nanoseconds, synchronization state, counter values, status, and error. Serialize exactly the version-1 schema from the design document through RapidJSON. Create only the parent directories needed for the requested output path; do not write anything when no output path is supplied.

- [ ] **Step 4: Register the test target and verify green**

Add `replay_perfcounter_report_test` directly in `replay_benchmark/CMakeLists.txt` under `MNN_PERFCOUNTER_BUILD_TESTS`; link it to `${MNN_DEPS}`, include `replay_benchmark/` and `3rd_party/`, register it with `add_test`, and do not add it to the default `run_test.out` source glob. Then rerun the command above and `ctest --test-dir /tmp/mnn-perfcounter-report-test -R replay_perfcounter_report --output-on-failure`. Expected: both `ok` and `unavailable` schema cases pass.

### Task 4: Add replay CLI and measure exactly one isolated operator

**Files:**

- Modify: `replay_benchmark/ReplayRecord.hpp`.
- Modify: `replay_benchmark/replay_benchmark.cpp`.
- Modify: `replay_benchmark/ReplayRunner.cpp`.
- Modify: `replay_benchmark/CMakeLists.txt`.

- [ ] **Step 1: Add a parser regression test or compile-only assertion for the new option**

Exercise `--perf-counter-output perf/op.json` with `--model`, `--record`, and `--op-id`; assert that `Options::perfCounterOutput` contains the path and replay mode is selected. Assert that the option without an op selector is rejected with the usage path.

- [ ] **Step 2: Run the parser test to verify it fails**

Run the existing replay benchmark test target or its focused executable with the new arguments. Expected: the option is currently treated as an unknown positional argument and validation fails.

- [ ] **Step 3: Implement option parsing and conditional compilation**

Add `std::string perfCounterOutput` to `Options`. Parse `--perf-counter-output <path>`; require a non-empty `recordDir`, replay selector, and output path. Update help text with the exact replay invocation. When `MNN_REPLAY_HAS_PERFCOUNTER` is absent, keep parsing valid but make the requested replay write an `unavailable` report instead of referencing missing symbols.

- [ ] **Step 4: Instrument the replay lifecycle**

After the `Execution` passes resize/variant validation and immediately before `backend->onExecuteBegin()`, create a session with the built-in normalized profile for the detected GPU. Record monotonic `start_ns`, call `start`, execute the op, call `backend->onExecuteEnd`, then call `output->wait(Tensor::MAP_TENSOR_READ, true)` for every output before `stop`. Record `end_ns`, collect values, compare tensors, and write the independent report. Use a small local cleanup guard or explicit cleanup branches so counters are stopped and the report is written if execution or comparison fails. Do not create a session in record mode or when `perfCounterOutput` is empty.

- [ ] **Step 5: Verify no-option behavior and host fallback**

Build and run a CPU replay without `--perf-counter-output`; verify stdout and exit status match the existing behavior and no report file is created. Run an OpenCL replay on a host without an OpenCL device with the option; verify the output JSON reports `unavailable` and the process reports the underlying replay failure accurately.

### Task 5: Build and device validation

**Files:**

- Modify: `3rd_party/MNNPerfCounter/README.md` with validated device/profile notes only after testing.
- Modify: `docs/agent/replay.md` with the final command and JSON output documentation.

- [ ] **Step 1: Run a clean host configure/build with the feature disabled**

Run:

```bash
cmake -S . -B /tmp/mnn-perfcounter-host-off -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_TEST=OFF -DMNN_REPLAY_ENABLE_PERFCOUNTER=OFF
cmake --build /tmp/mnn-perfcounter-host-off --target replay_benchmark.out -j2
```

Expected: exit code 0 and no `MNNPerfCounter` target in the build graph.

- [ ] **Step 2: Run a clean host configure/build with the feature enabled**

Run:

```bash
cmake -S . -B /tmp/mnn-perfcounter-host-on -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_TEST=OFF -DMNN_REPLAY_ENABLE_PERFCOUNTER=ON
cmake --build /tmp/mnn-perfcounter-host-on --target replay_benchmark.out -j2
```

Expected: exit code 0, `libMNNPerfCounter.a` exists, and replay_benchmark links it.

- [ ] **Step 3: Cross-compile AArch64 using the repository replay workflow**

Reuse the existing `skills/replay-benchmark-cross-compile/SKILL.md` workflow, seed `MNN_BUILD_BENCHMARK=ON`, `MNN_REPLAY_ENABLE_PERFCOUNTER=ON`, and the documented shared OpenCL configuration, then verify `file`, `readelf -h`, and `readelf -d` for an AArch64 executable and the expected shared libraries.

- [ ] **Step 4: Run one device replay per GPU family**

For each Mali and Adreno device, run:

```bash
./replay_benchmark.out --model model.mnn --record records/model.mnn --op-id 108 --perf-counter-output perf/op-108.json
```

Verify `status: ok`, the detected vendor/product/profile, `synchronized: true`, non-empty counter values, and the existing replay correctness result. Also run CPU and permission-denied cases and retain their explicit JSON diagnostics.

- [ ] **Step 5: Run final verification before claiming completion**

Run the focused unit tests, both host builds, and the available replay smoke test in one fresh command sequence; inspect exit codes and JSON status values. Only then report the changed files and verified results.
