# OpenCL PMU Baseline Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit OpenCL PMU benchmark mode to `replay_benchmark` that exercises buffer, image, texture, memory, arithmetic, and extension paths and classifies which GPU PMU metrics are readable and responsive.

**Architecture:** Keep normal replay unchanged. Add a dedicated `OpenCLPmuBenchmark` component that obtains the existing OpenCL runtime, compiles embedded kernels, runs capability-filtered cases, and uses `MNNPerfCounter::Session` in physical-slot-sized batches. Emit a separate versioned benchmark report with workload deltas, control deltas, and validity classifications.

**Tech Stack:** C++11, existing MNN OpenCL wrapper/runtime, OpenCL C kernels embedded as constant strings, RapidJSON, `MNNPerfCounter`, existing CMake and replay test infrastructure.

---

## File map

- Create: `replay_benchmark/OpenCLPmuBenchmark.hpp` — benchmark options, case metadata, result/report types, and runner entry point.
- Create: `replay_benchmark/OpenCLPmuBenchmark.cpp` — OpenCL device setup, capability detection, kernel execution, PMU sampling, classification, and JSON output.
- Create: `replay_benchmark/OpenCLPmuKernels.hpp` — embedded OpenCL C source and kernel names for buffer/image/texture/compute cases.
- Create: `replay_benchmark/README.md` — OpenCL concepts, benchmark matrix, PMU interpretation, and device validation commands.
- Create: `replay_benchmark/OpenCLPmuBenchmarkTest.cpp` — host-only metadata, parser, and synthetic classification tests.
- Modify: `replay_benchmark/ReplayRecord.hpp` — add explicit PMU benchmark options without coupling them to model replay selectors.
- Modify: `replay_benchmark/replay_benchmark.cpp` — parse benchmark flags, print usage, dispatch before model/record validation.
- Modify: `replay_benchmark/CMakeLists.txt` — compile benchmark files and test target; link OpenCL/perf-counter dependencies only when enabled.
- Modify: `3rd_party/MNNPerfCounter/CMakeLists.txt` only if a reusable helper is required for event profile expansion; otherwise keep event policy in the benchmark component.
- Test on device: `/mnt/nvme/workspace/replay-benchmark/` — deploy the AArch64 executable and libraries and retain JSON/CSV validation output.

### Task 1: Add failing host tests for benchmark metadata and validity classification

**Files:**
- Create: `replay_benchmark/OpenCLPmuBenchmarkTest.cpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmark.hpp`

- [ ] **Step 1: Define testable pure interfaces.**

Declare these functions in `OpenCLPmuBenchmark.hpp` so they do not require a
GPU: `openclPmuCaseNames()`, `findOpenclPmuCase(name)`,
`classifyPmuSignal(controlDelta, workloadDelta, expectedIncrease, threshold)`,
and `parseOpenclPmuCaseList(text)`.

- [ ] **Step 2: Write failing tests.**

Cover the following exact expectations:

```cpp
assert(findOpenclPmuCase("buffer_reuse_small") != nullptr);
assert(findOpenclPmuCase("image_texture_random") != nullptr);
assert(findOpenclPmuCase("does_not_exist") == nullptr);
assert(classifyPmuSignal(10, 100, true, 20).valid);
assert(!classifyPmuSignal(10, 12, true, 20).responsive);
assert(parseOpenclPmuCaseList("buffer_reuse_small,image_texture_random").size() == 2);
```

- [ ] **Step 3: Build the test target and verify the expected failure.**

Run:

```bash
cmake -S . -B /tmp/mnn-opencl-pmu-test \
  -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TEST=OFF \
  -DMNN_OPENCL=OFF -DMNN_REPLAY_ENABLE_PERFCOUNTER=OFF \
  -DMNN_OPENCL_PMU_BENCH_BUILD_TESTS=ON
cmake --build /tmp/mnn-opencl-pmu-test --target opencl_pmu_benchmark_test -j2
```

Expected: compile or link failure because the new interfaces are not yet
implemented.

### Task 2: Implement benchmark metadata, options, and synthetic classification

**Files:**
- Modify: `replay_benchmark/OpenCLPmuBenchmark.hpp`
- Create: `replay_benchmark/OpenCLPmuBenchmark.cpp`
- Modify: `replay_benchmark/ReplayRecord.hpp`
- Modify: `replay_benchmark/replay_benchmark.cpp`

- [ ] **Step 1: Add options without changing normal replay.**

Add fields for `openclPmuBench`, case selector, iteration count, byte size,
event profile, and output path. Parse:

```text
--opencl-pmu-bench
--opencl-pmu-case <name|all>
--opencl-pmu-iterations <N>
--opencl-pmu-size <bytes>
--perf-counter-events <names|auto>
--perf-counter-output <path>
```

Reject the benchmark mode when OpenCL support was not compiled in. Require an
output path only when the caller requests report persistence; stdout must
still show a concise summary.

- [ ] **Step 2: Implement pure metadata and classification.**

Define case metadata with path, data type, extension requirement, working-set
regime, expected signal direction, and default iterations. Implement case list
parsing and classify each event independently as `readable`, `responsive`,
`discriminative`, `valid`, `unavailable`, or `skipped`.

- [ ] **Step 3: Make the host tests pass.**

Run:

```bash
cmake --build /tmp/mnn-opencl-pmu-test --target opencl_pmu_benchmark_test -j2
/tmp/mnn-opencl-pmu-test/replay_benchmark/opencl_pmu_benchmark_test
```

Expected: exit code 0 and all metadata/classification assertions pass.

### Task 3: Add embedded OpenCL kernels and capability-filtered case definitions

**Files:**
- Create: `replay_benchmark/OpenCLPmuKernels.hpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmark.cpp`

- [ ] **Step 1: Add buffer kernels.**

Embed kernels for sequential read, sequential write, read-modify-write,
small-set reuse, L2-sized reuse, strided access, random access, and large
streaming. Each kernel accepts a loop count and uses an output sink so the
compiler cannot remove the memory operation.

- [ ] **Step 2: Add image and texture kernels.**

Embed `image2d_t` read/write kernels and sampler kernels for nearest/linear,
normalized/unnormalized, local, cross-row, and random coordinates. Use only
formats reported by `CL_DEVICE_IMAGE_SUPPORT` and the queried image format
list; unsupported cases become `skipped` with a reason.

- [ ] **Step 3: Add compute and extension kernels.**

Add integer, FP32, FP16, vector, local-memory, private-register, constant-memory,
barrier, and atomic kernels. Document and cover the OpenCL address spaces used
by each kernel: global, local, private, and constant. Add explicit cases for
work-item/work-group indexing, sub-groups when advertised, synchronization,
and memory ordering.
Guard FP16 and extension-specific source with the corresponding pragma and
capability check. Do not compile unsupported extension cases as ordinary
FP32 cases because that would invalidate the PMU comparison.

- [ ] **Step 4: Add deterministic case sizing.**

Use caller-provided size when present, otherwise select bounded defaults for
small reuse, L2 reuse, and large streaming. Clamp allocations to device
`CL_DEVICE_MAX_MEM_ALLOC_SIZE` and report a case as skipped if the minimum
working set cannot be allocated.

### Task 4: Implement OpenCL execution and PMU batch collection

**Files:**
- Modify: `replay_benchmark/OpenCLPmuBenchmark.cpp`
- Modify: `replay_benchmark/CMakeLists.txt`

- [ ] **Step 1: Reuse the existing OpenCL loading/runtime path.**

Initialize the same OpenCL symbol operator used by replay, obtain a device,
context, and command queue, and fail with a structured `unavailable` report
when symbols or a device are absent. Keep all OpenCL objects local to the
benchmark runner.

- [ ] **Step 2: Add control and workload execution.**

Compile one embedded program per compatible kernel family, allocate buffers or
images, enqueue the empty control and workload, and call `queue.finish()`
before every PMU stop. Run control and workload with the same synchronization
and iteration policy.

- [ ] **Step 3: Batch counters by physical slots.**

Resolve requested event names through `MNNPerfCounter`. Group events by GPU
group and split each group at `slotsAvailable`; run a separate synchronized
pass for every batch. Preserve event order in the report and record creation,
activation, read, synchronization, and deactivation errors separately.

- [ ] **Step 4: Add robust cleanup.**

Release kernels, programs, memory objects, queues, and contexts on every
return path. Ensure a partially activated PMU batch is deactivated even when a
later activation or OpenCL enqueue fails.

- [ ] **Step 5: Enable the test target and OpenCL source files in CMake.**

Add benchmark sources to `replay_benchmark.out` and add
`opencl_pmu_benchmark_test` behind `MNN_OPENCL_PMU_BENCH_BUILD_TESTS`. Link
`MNNPerfCounter` only when `MNN_REPLAY_ENABLE_PERFCOUNTER=ON`; preserve the
existing no-perf-counter build.

### Task 5: Add versioned JSON report and CLI dispatch

**Files:**
- Modify: `replay_benchmark/OpenCLPmuBenchmark.cpp`
- Modify: `replay_benchmark/replay_benchmark.cpp`

- [ ] **Step 1: Serialize runtime and capability metadata.**

Write `mnn-opencl-pmu-benchmark` version 1 JSON containing vendor/product,
driver, OpenCL version, extensions, case metadata, status, duration, counter
delta, control delta, signal classification, and errors.

- [ ] **Step 2: Dispatch benchmark mode before model discovery.**

When `--opencl-pmu-bench` is present, run the benchmark and return its status
without requiring `--model`, `--record`, or an execution selector. Normal
record/replay paths must remain byte-for-byte compatible in their CLI behavior.

- [ ] **Step 3: Verify JSON parent-directory creation and parseability.**

Use the existing RapidJSON dependency and test a nested output path. Verify
that unsupported cases remain in the report rather than disappearing.

### Task 6: Host verification and cross-build verification

**Files:**
- Test: `3rd_party/MNNPerfCounter/test/MNNPerfCounterTest.cpp`
- Test: `replay_benchmark/OpenCLPmuBenchmarkTest.cpp`

- [ ] **Step 1: Run host tests and static checks.**

```bash
cmake --build /tmp/mnn-opencl-pmu-test --target opencl_pmu_benchmark_test -j2
/tmp/mnn-opencl-pmu-test/replay_benchmark/opencl_pmu_benchmark_test
cmake -S . -B /tmp/mnn-opencl-pmu-host \
  -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TEST=OFF -DMNN_OPENCL=OFF \
  -DMNN_REPLAY_ENABLE_PERFCOUNTER=ON
cmake --build /tmp/mnn-opencl-pmu-host --target replay_benchmark.out -j2
```

Expected: metadata tests pass; non-OpenCL benchmark binary builds and reports
that the benchmark mode is unavailable rather than crashing.

- [ ] **Step 2: Cross-compile the OpenCL benchmark.**

Reuse `build-aarch64-gnueabihf`, configure with
`MNN_BUILD_BENCHMARK=ON`, `MNN_OPENCL=ON`, shared MNN libraries,
`MNN_SEP_BUILD=ON`, and `MNN_REPLAY_ENABLE_PERFCOUNTER=ON`; verify the output
is an AArch64 ELF and contains the benchmark symbols.

### Task 7: Rhinopi validation and event effectiveness report

**Files:**
- Runtime output: `/mnt/nvme/workspace/replay-benchmark/perf/`
- Optional helper: `3rd_party/MNNPerfCounter/tools/analyze_opencl_pmu.py`

- [ ] **Step 1: Deploy the rebuilt executable and libraries.**

Deploy `replay_benchmark.out`, `libMNN.so`, `libMNN_Express.so`, and
`libMNN_CL.so` with `LD_LIBRARY_PATH` pointing to the device `lib` directory.

- [ ] **Step 2: Run representative cases.**

Run buffer reuse/L2 reuse/streaming, image, texture, integer, FP16, FP32, and
all advertised extension cases. Use explicit A7xx event batches covering
CP/RBBM/HLSQ/UCHE/TP/SP/RB and then run `auto`.

- [ ] **Step 3: Summarize validity.**

Report total/readable/responsive/discriminative/valid/skipped/unavailable
counts. Keep zero deltas as observations and identify cases where a readable
event is not responsive or where the workload hypothesis is not supported by
the device driver.

### Task 8: Documentation and final review

**Files:**
- Create: `replay_benchmark/README.md`
- Modify: `3rd_party/MNNPerfCounter/README.md`
- Modify: `replay_benchmark/replay_benchmark.cpp` usage text

- [ ] **Step 1: Document the command and report schema.**

Include one-device smoke commands and explain that benchmark PMU values are
device-wide intervals and that zero does not mean ioctl failure. In
`replay_benchmark/README.md`, include a concept-to-case table covering global,
local, private, and constant memory; buffer/image/texture objects and samplers;
work-item/work-group/sub-group execution; barrier/atomic/memory ordering;
L1/L2 locality, cache fill, miss, and streaming hypotheses; and INT/FP16/FP32
and extension-specific instructions.

- [ ] **Step 2: Run formatting and final verification.**

Format new C++ files with the repository clang-format, rebuild host and
AArch64 targets, run host tests, and inspect `git diff` for restricted paths,
namespace-scope dynamic initialization, and unintended changes to normal
replay behavior.
