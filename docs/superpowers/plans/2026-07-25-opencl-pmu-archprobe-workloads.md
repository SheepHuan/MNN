# OpenCL PMU ArchProbe-Guided Workloads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve MNN's OpenCL PMU micro-workloads using ArchProbe's cache, vectorization, occupancy, and bandwidth probing ideas, then add a single-event sweep skill that emits reliable CSV measurements on Rhinopi.

**Architecture:** First make the PMU measurement window strict: prepare OpenCL resources and warm up outside the sampled interval, then run a matched control interval and a workload interval containing configurable repeated dispatches. Next add explicit launch geometry and architecture-oriented workload families without removing the existing 14 compatibility cases. Finally provide a device-oriented Python sweep skill that launches exactly one PMU event per process, computes `workload_delta - control_delta`, and writes the requested CSV schema.

**Tech Stack:** C++11, OpenCL C, MNN `MNN_CL` wrapper, MNNPerfCounter/KGSL, Python 3 standard library, SSH/SCP, CMake, AArch64 Rhinopi validation.

---

## Scope and file map

The implementation is split into independently testable stages:

- PMU measurement core: `replay_benchmark/ReplayRecord.hpp`, `replay_benchmark/replay_benchmark.cpp`, `replay_benchmark/OpenCLPmuBenchmark.cpp`, `replay_benchmark/OpenCLPmuBenchmark.hpp`.
- Kernel families: `replay_benchmark/OpenCLPmuBufferKernels.hpp`, `replay_benchmark/OpenCLPmuComputeKernels.hpp`, `replay_benchmark/OpenCLPmuImageKernels.hpp`.
- Regression coverage: `replay_benchmark/OpenCLPmuBenchmarkTest.cpp` and host/device command checks.
- Sweep skill: create `skills/opencl-pmu-sweep/SKILL.md`, `skills/opencl-pmu-sweep/agents/openai.yaml`, `skills/opencl-pmu-sweep/scripts/sweep_opencl_pmu.py`, and its Python tests/fixtures.
- User documentation: `replay_benchmark/README.md`.

Do not read or modify `schema/private/` or `source/internal/`. Preserve the existing MNN wrapper reuse: `replay_benchmark.out` and the sweep tool must use the single `MNN_CL` OpenCL wrapper.

## Device validation matrix

Keep both device targets in every device-validation stage:

| Target | SSH | Work directory | GPU family |
|---|---|---|---|
| Rhinopi-X1 | `root@192.168.101.227` | `/mnt/nvme/workspace/replay-benchmark` | Adreno 740 / A7xx |
| OrangePi | `root@192.168.101.113` | `/mnt/ssd/workspace` | Mali |

The sweep must accept `--device` and `--remote-root` so the same command shape can run against both targets. Do not put either device password in a command, script, plan, or log.

### Task 1: Add repeated workload runs and strict option handling

**Files:**
- Modify: `replay_benchmark/ReplayRecord.hpp`
- Modify: `replay_benchmark/replay_benchmark.cpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmark.cpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmarkTest.cpp`
- Modify: `replay_benchmark/README.md`

- [x] **Step 1: Add a failing runtime assertion for the new JSON field.**

Extend the opt-in device test to run `image_reuse` with `options.openclPmuWorkloadRuns = 3`, write a temporary JSON report, and return failure unless the selected case contains numeric `workload_runs == 3`. Use explicit return checks instead of `assert`, so the test remains active in Release builds.

- [x] **Step 2: Run the new test before implementation.**

Run:

```bash
MNN_RUN_OPENCL_PMU_RUNTIME_TEST=1 \
LD_LIBRARY_PATH="$PWD/build-aarch64-gnueabihf" \
./build-aarch64-gnueabihf/opencl_pmu_benchmark_test
```

Expected: compilation or test failure because `Options` has no `openclPmuWorkloadRuns` field and the JSON has no `workload_runs` field.

- [x] **Step 3: Add the option and CLI parser.**

Add `int openclPmuWorkloadRuns = 1;` to `MNN::Replay::Options`. Parse:

```text
--opencl-pmu-workload-runs N
```

Clamp values below one to one, and add the option to `--help`. Keep `--opencl-pmu-iterations` as the inner loop count; `workload-runs` is the number of consecutive kernel dispatches in one PMU workload session.

- [x] **Step 4: Make the workload PMU interval execute N consecutive runs.**

In `samplePmuInterval`, keep one `Session::start()` and one `Session::stop()` around the entire interval. Execute the control workload once. For a workload interval, call the prepared-case enqueue function exactly `options.openclPmuWorkloadRuns` times and finish every dispatch. Do not create or destroy benchmark buffers inside this loop.

- [x] **Step 5: Add measurement metadata.**

Add `workload_runs` to every case object in the PMU JSON and document the distinction between inner kernel `iterations` and outer workload `workload_runs` in `replay_benchmark/README.md`.

- [x] **Step 6: Rebuild and run the test.**

Run:

```bash
cmake --build build-aarch64-gnueabihf \
  --target replay_benchmark.out opencl_pmu_benchmark_test -j"$(nproc)"
```

Expected: both targets build successfully. On Rhinopi, the JSON test must report `workload_runs=3` and `status=ok`.

- [ ] **Step 7: Commit the isolated option change.**

```bash
git add replay_benchmark/ReplayRecord.hpp replay_benchmark/replay_benchmark.cpp \
  replay_benchmark/OpenCLPmuBenchmark.cpp replay_benchmark/OpenCLPmuBenchmarkTest.cpp \
  replay_benchmark/README.md
git commit --only replay_benchmark/ReplayRecord.hpp replay_benchmark/replay_benchmark.cpp \
  replay_benchmark/OpenCLPmuBenchmark.cpp replay_benchmark/OpenCLPmuBenchmarkTest.cpp \
  replay_benchmark/README.md -m "[OpenCL:Feature] Repeat PMU workload dispatches"
```

### Task 2: Move all preparation and warmup outside the PMU session

**Files:**
- Modify: `replay_benchmark/OpenCLPmuBenchmark.cpp`
- Modify: `replay_benchmark/OpenCLPmuBufferKernels.hpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmarkTest.cpp`
- Modify: `replay_benchmark/README.md`

- [x] **Step 1: Add a regression check for the prepared-case invariant.**

Add a test-only counter or debug mode that reports `prepare_count` and `sampled_enqueue_count` for one case. The expected invariant is: preparation occurs once per case/batch, while sampled enqueue count is `1 + workload_runs` (one control and N workload dispatches). The test must fail against the current implementation because `enqueueCase()` currently allocates resources on every call.

- [x] **Step 2: Introduce an internal prepared-case object.**

Create an internal `PreparedOpenCLPmuCase` in `OpenCLPmuBenchmark.cpp` containing the kernel, buffers, image/sampler handles, global range, local range, and argument state needed by one case. Give raw `cl_mem` and `cl_sampler` members explicit RAII cleanup. Preparation must:

1. allocate buffers/images;
2. write deterministic input data;
3. create the kernel and set all arguments;
4. select global/local ranges;
5. leave only `enqueueNDRangeKernel()` and `finish()` for the sampled path.

- [x] **Step 3: Split preparation from enqueue.**

Replace the current allocation-heavy `enqueueCase()` path with:

```cpp
bool prepareCase(const OpenCLPmuCaseInfo&, const Options&, OpenCLPmuRuntime*,
                 cl::Program&, cl::Program*, cl::Program*, PreparedOpenCLPmuCase*,
                 std::string* error);
bool enqueuePreparedCase(OpenCLPmuRuntime*, PreparedOpenCLPmuCase*, std::string* error);
```

`samplePmuInterval()` may call only `enqueuePreparedCase()` after `Session::start()`.

- [x] **Step 4: Add a matched control kernel.**

Extend `OpenCLPmuBufferKernels.hpp` with a control kernel that uses the same global/local geometry as the prepared workload, performs only a cheap private operation, and writes one result from work-item zero. Keep the existing one-work-item `pmu_empty` for compatibility, but use the matched control for new PMU measurements so CP/RBBM/launch overhead is comparable.

- [x] **Step 5: Add warmup runs.**

Add `int openclPmuWarmupRuns = 2;` and CLI option `--opencl-pmu-warmup-runs N`. Run the prepared case and matched control before `Session::start()`. Warmup must be outside both sampled sessions and must not be included in `control_duration_ns` or `workload_duration_ns`.

- [x] **Step 6: Initialize image contents before sampling.**

Use `clEnqueueWriteImage` through the existing MNN wrapper before warmup for image and texture read cases. Fill the image with deterministic `float4` data. Do not perform image initialization after PMU sampling starts.

- [x] **Step 7: Run the regression and device tests.**

Run the host build and then on Rhinopi:

```bash
MNN_RUN_OPENCL_PMU_RUNTIME_TEST=1 \
LD_LIBRARY_PATH="$PWD/lib" ./bin/opencl_pmu_benchmark_test

LD_LIBRARY_PATH="$PWD/lib" ./bin/replay_benchmark.out \
  --opencl-pmu-bench --opencl-pmu-case buffer_fp16 \
  --opencl-pmu-iterations 20 --opencl-pmu-workload-runs 5 \
  --perf-counter-events gpu_active_cycles
```

Expected: no crash, `workload_runs=5`, and no buffer/image allocation occurs in the sampled interval according to the debug invariant.

- [ ] **Step 8: Commit the strict measurement window.**

```bash
git add replay_benchmark/OpenCLPmuBenchmark.cpp \
  replay_benchmark/OpenCLPmuBufferKernels.hpp \
  replay_benchmark/OpenCLPmuBenchmarkTest.cpp \
  replay_benchmark/README.md
git commit --only replay_benchmark/OpenCLPmuBenchmark.cpp \
  replay_benchmark/OpenCLPmuBufferKernels.hpp \
  replay_benchmark/OpenCLPmuBenchmarkTest.cpp \
  replay_benchmark/README.md -m "[OpenCL:Perf] Isolate PMU workload measurement window"
```

### Task 3: Add explicit launch geometry and common workload parameters

**Files:**
- Modify: `replay_benchmark/ReplayRecord.hpp`
- Modify: `replay_benchmark/replay_benchmark.cpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmark.cpp`
- Modify: `replay_benchmark/OpenCLPmuBufferKernels.hpp`
- Modify: `replay_benchmark/OpenCLPmuComputeKernels.hpp`
- Modify: `replay_benchmark/OpenCLPmuImageKernels.hpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmarkTest.cpp`

- [x] **Step 1: Add a failing non-multiple-size test.**

Run the current benchmark with a size that is not divisible by 64 and add a test expecting explicit geometry metadata. The test must fail because current buffer/compute dispatches use `CL_NULL_RANGE` and do not report local size.

- [x] **Step 2: Add `--opencl-pmu-local-size N`.**

Use zero as the default meaning “device-selected legacy behavior” for existing cases. For new architecture-guided cases, require one of `32, 64, 128, 256`; reject other values with a JSON error before dispatch. Round buffer/compute global size up to a multiple of local size and add `gid < count` guards to every kernel that writes by `gid`.

- [x] **Step 3: Add geometry metadata.**

For every case, report `global_size`, `local_size`, and `dispatches`. Keep image geometry as two-dimensional `width × height`; report it as an object rather than flattening it into the buffer fields.

- [x] **Step 4: Validate behavior.**

Run CLI byte sizes `262144`, `262145`, and `1048576`, plus byte sizes that produce
element counts `262144`, `262145`, and `1048576`, with local sizes `32`, `64`, and
`128`. Verify no out-of-bounds writes, no OpenCL enqueue error, and exact
`dispatches == workload_runs` for the workload interval on Rhinopi and OrangePi.

- [ ] **Step 5: Commit explicit geometry support.**

```bash
git add replay_benchmark/ReplayRecord.hpp replay_benchmark/replay_benchmark.cpp \
  replay_benchmark/OpenCLPmuBenchmark.cpp replay_benchmark/OpenCLPmuBufferKernels.hpp \
  replay_benchmark/OpenCLPmuComputeKernels.hpp replay_benchmark/OpenCLPmuImageKernels.hpp \
  replay_benchmark/OpenCLPmuBenchmarkTest.cpp
git commit --only replay_benchmark/ReplayRecord.hpp replay_benchmark/replay_benchmark.cpp \
  replay_benchmark/OpenCLPmuBenchmark.cpp replay_benchmark/OpenCLPmuBufferKernels.hpp \
  replay_benchmark/OpenCLPmuComputeKernels.hpp replay_benchmark/OpenCLPmuImageKernels.hpp \
  replay_benchmark/OpenCLPmuBenchmarkTest.cpp -m "[OpenCL:Feature] Expose PMU workload launch geometry"
```

### Task 4: Add ArchProbe-guided buffer and compute workloads

**Files:**
- Modify: `replay_benchmark/OpenCLPmuBufferKernels.hpp`
- Modify: `replay_benchmark/OpenCLPmuComputeKernels.hpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmark.cpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmark.hpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmarkTest.cpp`
- Modify: `replay_benchmark/README.md`

- [x] **Step 1: Add metadata tests for the new cases.**

Require these case names to resolve and expose the intended concept:

```text
buffer_stride
buffer_pchase
buffer_vec4
fp32_throughput
fp16_throughput
```

- [x] **Step 2: Implement `buffer_stride`.**

Use a power-of-two working set and bit-mask addressing instead of `%`. Expose `working_set_bytes` and `stride_bytes`. Keep the global access pattern coalesced or uniformly strided across work-items, and write one reduced value per work-item so the loads cannot be removed.

- [x] **Step 3: Implement `buffer_pchase`.**

Build a pointer-chain index buffer outside the PMU interval. Run one or a small fixed number of work-items with a serial dependent load chain. Sweep power-of-two working sets to identify cache-level latency transitions. Report the working set and stride in JSON so the sweep CSV can distinguish the configurations.

- [x] **Step 4: Implement `buffer_vec4`.**

Use `float4` loads and stores, with the same transferred-byte count as scalar baseline where possible. Add `vector_width` to the prepared-case configuration and guard alignment. Keep this separate from `buffer_stream` so vectorization effects are not confused with working-set effects.

- [x] **Step 5: Implement independent-accumulator throughput kernels.**

Change the new FP32/FP16 throughput cases to use multiple independent accumulators, unroll the arithmetic loop, and reduce once at the end. This follows ArchProbe’s `gflops` aspect and avoids measuring only a single dependency chain. Build FP16 with the existing `cl_khr_fp16` check.

- [x] **Step 6: Add a device smoke matrix.**

On Rhinopi and OrangePi run each new case with one event at a time:

```bash
for case in buffer_stride buffer_pchase buffer_vec4 fp32_throughput fp16_throughput; do
  LD_LIBRARY_PATH="$PWD/lib" ./bin/replay_benchmark.out \
    --opencl-pmu-bench --opencl-pmu-case "$case" \
    --opencl-pmu-iterations 100 --opencl-pmu-workload-runs 5 \
    --perf-counter-events gpu_active_cycles
done
```

Expected: every supported case returns JSON `status=ok`; FP16 is `skipped` only when the device lacks `cl_khr_fp16`.

- [ ] **Step 7: Commit the buffer/compute extensions.**

```bash
git add replay_benchmark/OpenCLPmuBufferKernels.hpp \
  replay_benchmark/OpenCLPmuComputeKernels.hpp replay_benchmark/OpenCLPmuBenchmark.cpp \
  replay_benchmark/OpenCLPmuBenchmark.hpp replay_benchmark/OpenCLPmuBenchmarkTest.cpp \
  replay_benchmark/README.md
git commit --only replay_benchmark/OpenCLPmuBufferKernels.hpp \
  replay_benchmark/OpenCLPmuComputeKernels.hpp replay_benchmark/OpenCLPmuBenchmark.cpp \
  replay_benchmark/OpenCLPmuBenchmark.hpp replay_benchmark/OpenCLPmuBenchmarkTest.cpp \
  replay_benchmark/README.md -m "[OpenCL:Feature] Add architecture-guided PMU workloads"
```

### Task 5: Separate image, constant, local-memory, and atomic effects

**Files:**
- Modify: `replay_benchmark/OpenCLPmuImageKernels.hpp`
- Modify: `replay_benchmark/OpenCLPmuComputeKernels.hpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmark.cpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmark.hpp`
- Modify: `replay_benchmark/OpenCLPmuBenchmarkTest.cpp`
- Modify: `replay_benchmark/README.md`

- [x] **Step 1: Add metadata tests for separated effects.**

Require these cases:

```text
image_stride_x
image_stride_y
constant_bandwidth
local_bandwidth
local_barrier
atomic_contended
atomic_distributed
```

- [x] **Step 2: Implement image X/Y and working-set controls.**

Expose image width, height, and stride. Keep nearest and linear filtering separate. Use deterministic initialized image data and report the two-dimensional dispatch. Ensure `--opencl-pmu-size` either changes image dimensions or is rejected with a clear incompatible-argument error; it must not silently leave image size fixed.

- [x] **Step 3: Implement constant-memory working-set sweep.**

Replace the fixed 16-float table in the new `constant_bandwidth` case with a power-of-two table size argument. Use a mask for indexing and keep the table allocation outside the PMU interval.

- [x] **Step 4: Separate local bandwidth from barrier cost.**

Implement `local_bandwidth` with one initialization barrier followed by repeated local loads. Implement `local_barrier` with minimal private work and repeated barriers. Use explicit local sizes and local allocation sizes in the JSON.

- [x] **Step 5: Separate atomic contention modes.**

Keep `atomic_contended` writing all work-items to one location. Implement `atomic_distributed` with one counter per work-item or per workgroup. Report the target counter count so the two cases are not mistaken for the same workload.

- [x] **Step 6: Verify expected event separation.**

For each new case, collect at least:

```text
gpu_active_cycles
uche_busy_cycles
tp_busy_cycles
sp_busy_cycles
sp_gm_load_instructions
sp_lm_load_instructions
sp_lm_atomics
```

Use one event per process. The acceptance criterion is not that every event is nonzero; it is that the case executes successfully and the expected subsystem events show repeatable differences across at least three measurements.

- [ ] **Step 7: Commit the special-memory extensions.**

```bash
git add replay_benchmark/OpenCLPmuImageKernels.hpp replay_benchmark/OpenCLPmuComputeKernels.hpp \
  replay_benchmark/OpenCLPmuBenchmark.cpp replay_benchmark/OpenCLPmuBenchmark.hpp \
  replay_benchmark/OpenCLPmuBenchmarkTest.cpp replay_benchmark/README.md
git commit --only replay_benchmark/OpenCLPmuImageKernels.hpp \
  replay_benchmark/OpenCLPmuComputeKernels.hpp replay_benchmark/OpenCLPmuBenchmark.cpp \
  replay_benchmark/OpenCLPmuBenchmark.hpp replay_benchmark/OpenCLPmuBenchmarkTest.cpp \
  replay_benchmark/README.md -m "[OpenCL:Feature] Split PMU memory subsystem workloads"
```

### Task 6: Create the single-event PMU sweep skill

**Files:**
- Create: `skills/opencl-pmu-sweep/SKILL.md`
- Create: `skills/opencl-pmu-sweep/agents/openai.yaml`
- Create: `skills/opencl-pmu-sweep/scripts/sweep_opencl_pmu.py`
- Create: `skills/opencl-pmu-sweep/tests/test_sweep_opencl_pmu.py`
- Create: `skills/opencl-pmu-sweep/tests/fixtures/sample_pmu_report.json`

- [x] **Step 1: Write failing Python tests for CSV flattening.**

Cover these behaviors:

```python
def test_delta_metric_subtracts_control_from_workload():
    assert delta_metric(100, 850) == 750

def test_negative_delta_is_preserved():
    assert delta_metric(850, 100) == -750

def test_one_event_is_present_in_each_command():
    command = build_command(case="buffer_fp32", event="gpu_active_cycles", workload_runs=5)
    assert command.count("--perf-counter-events") == 1
    assert "gpu_active_cycles" in command
    assert "," not in command[command.index("--perf-counter-events"):]

def test_csv_schema_is_exact():
    assert fieldnames == [
        "case_name", "case_args", "case_runs",
        "pmu_metric_name", "delta_metric"
    ]
```

- [x] **Step 2: Run the Python tests to verify RED.**

Run:

```bash
python3 -m unittest discover -s skills/opencl-pmu-sweep/tests -v
```

Expected: failure because the skill directory and sweep functions do not exist.

- [x] **Step 3: Initialize the skill structure.**

Run the skill creator initializer with the repository skill path and script resource:

```bash
python3 /home/yanghuan/.codex/skills/.system/skill-creator/scripts/init_skill.py \
  opencl-pmu-sweep --path "$PWD/skills" --resources scripts
```

Keep only `SKILL.md`, `agents/openai.yaml`, and the required script; do not add README or unrelated helper files.

- [x] **Step 4: Implement the deterministic sweep script.**

Provide:

```text
--device root@host                 SSH target; omit for local execution
--remote-root PATH                 device benchmark workspace
--binary PATH                      replay_benchmark.out relative to remote-root
--cases CASE[,CASE...]             default all registered cases
--events EVENT[,EVENT...]          default MNN-supported A7xx profile
--iterations N
--size BYTES
--workload-runs N                  consecutive dispatches in one workload session
--warmup-runs N
--measurements N                   independent one-event PMU sessions
--timeout SECONDS
--output CSV
--keep-json-dir DIR                optional raw report retention
```

For every `case × event × measurement`, invoke the benchmark with exactly one event in `--perf-counter-events`. Parse the case’s `control_delta` and `workload_delta`, compute the signed integer `workload_delta - control_delta`, and write exactly:

```text
case_name,case_args,case_runs,pmu_metric_name,delta_metric
```

Use `csv.DictWriter`, `subprocess.run(..., check=False)`, `shlex.quote` for remote command construction, and no password handling. Continue after an unsupported event or failed case by writing an empty `delta_metric` and printing the diagnostic to stderr.

- [x] **Step 5: Implement Python tests and fixture parsing.**

Use the fixture JSON to verify one case, one counter, `control_delta=100`, `workload_delta=850`, and output `delta_metric=750`. Inject a fake runner instead of contacting SSH. Verify that a failed command does not create a false numeric delta.

- [x] **Step 6: Write the skill instructions.**

`SKILL.md` must instruct an agent to:

1. verify the AArch64 binary and `libMNN_CL.so` on the device;
2. run one event per process, never pass a comma-separated event list for the sweep;
3. use `workload_runs >= 3` and warmup before interpreting small deltas;
4. preserve raw JSON when a case fails;
5. treat empty/zero deltas as observations, not automatic PMU failure;
6. report CSV row counts, failed rows, device identity, and command parameters;
7. never put the device password in commands or logs.

`agents/openai.yaml` should advertise the skill as “OpenCL PMU Single-Event Sweep” and trigger on requests to sweep PMU events, compare control/workload deltas, or generate PMU CSVs.

- [x] **Step 7: Run the Python tests to verify GREEN.**

Run:

```bash
python3 -m unittest discover -s skills/opencl-pmu-sweep/tests -v
```

Expected: all tests pass and the CSV header is exactly the five requested columns.

- [ ] **Step 8: Commit the skill and script.**

```bash
git add skills/opencl-pmu-sweep
git commit --only skills/opencl-pmu-sweep -m "[OpenCL:Tool] Add single-event PMU sweep skill"
```

### Task 7: Integrate documentation and run the staged validation matrix

**Files:**
- Modify: `replay_benchmark/README.md`
- Modify: `skills/opencl-pmu-sweep/SKILL.md`
- Create: `docs/superpowers/specs/2026-07-25-opencl-pmu-sweep-design.md`

- [x] **Step 1: Document the measurement model.**

Create `docs/superpowers/specs/2026-07-25-opencl-pmu-sweep-design.md` with this exact structure and decisions:

```markdown
# OpenCL PMU Single-Event Sweep Design

## Goal
Measure one A7xx PMU event at a time around a prepared OpenCL workload and emit a reproducible CSV.

## Measurement model
The control and workload values come from separate PMU sessions. `control_delta` is the count during matched control work; `workload_delta` is the count during the prepared workload. `delta_metric = workload_delta - control_delta` is signed. Inner `iterations` controls the kernel loop, outer `workload_runs` controls consecutive dispatches inside one workload session, and `measurements` repeats independent one-event sessions.

## Single-event invariant
The sweep invokes `replay_benchmark.out` once for each case, event, and measurement, passing exactly one name to `--perf-counter-events`. It never relies on physical-slot batching for sweep conclusions.

## CSV contract
The header is exactly `case_name,case_args,case_runs,pmu_metric_name,delta_metric`. Failed or unsupported combinations retain a row with an empty delta and a diagnostic in the run log.

## Workload preparation
Programs, kernels, buffers, images, samplers, and deterministic input data are prepared before PMU sampling. Warmup runs are outside control/workload sessions. The workload session performs repeated prepared dispatches only.

## ArchProbe relationship
ArchProbe supplies timing-based cache, vector-width, bandwidth, local-memory, and occupancy hints. MNNPerfCounter validates whether selected A7xx events respond to the corresponding workload; ArchProbe output is not treated as a PMU count.
```

Explain that `control_delta` and `workload_delta` are measurements from separate PMU sessions, while `delta_metric` is the signed baseline-subtracted value. Distinguish inner `iterations`, outer `workload_runs`, and independent `measurements`.

- [x] **Step 2: Document the ArchProbe mapping.**

Map MNN cases to ArchProbe concepts:

```text
buffer_stride / buffer_pchase -> BufferCachelineSize / BufferCacheHierarchyPChase
buffer_vec4                  -> BufferVecWidth
fp32/fp16_throughput         -> Gflops
constant_bandwidth           -> ConstMemBandwidth
local_bandwidth              -> LocalMemBandwidth
image_stride                 -> ImageCachelineSize / ImageCacheHierarchyPChase
```

State that ArchProbe is timing-based and is used to choose sweep parameters, while MNNPerfCounter validates hardware event response.

- [x] **Step 3: Run host verification.**

Run:

```bash
cmake --build build-aarch64-gnueabihf \
  --target replay_benchmark.out opencl_pmu_benchmark_test -j"$(nproc)"
python3 -m unittest discover -s skills/opencl-pmu-sweep/tests -v
git diff --check
```

Expected: both binaries build, Python tests pass, and `git diff --check` is clean.

- [x] **Step 4: Run the small Rhinopi sweep first.**

Deploy only the rebuilt executable and MNN shared libraries, then run:

```bash
python3 skills/opencl-pmu-sweep/scripts/sweep_opencl_pmu.py \
  --device root@192.168.101.227 \
  --remote-root /mnt/nvme/workspace/replay-benchmark \
  --cases buffer_reuse_small,buffer_fp32,image_reuse \
  --events gpu_active_cycles,uche_busy_cycles \
  --iterations 20 --size 1048576 \
  --workload-runs 5 --warmup-runs 2 --measurements 3 \
  --output /tmp/rhinopi-opencl-pmu-smoke.csv
```

Expected: `3 cases × 2 events × 3 measurements = 18` data rows, excluding no rows for unsupported/failed combinations; every successful row contains a signed integer delta.

- [x] **Step 5: Run the representative full sweep.**

Use the 14 baseline cases plus the new cases, and the representative A7xx event set:

```text
gpu_active_cycles,cp_busy_cycles,rbbm_busy_cycles,hlsq_busy_cycles,
uche_busy_cycles,tp_busy_cycles,sp_busy_cycles,sp_gm_load_instructions,
sp_gm_store_instructions,sp_lm_load_instructions,sp_lm_store_instructions,
sp_lm_atomics,rb_busy_cycles
```

Run with `workload_runs=5`, `measurements=3`, and preserve raw JSON reports. Analyze per-case medians and median absolute deviations before declaring an event discriminative.

- [x] **Step 6: Record validation evidence.**

Save the CSV, raw JSON directory, device identity, binary hashes, command line, row counts, failed combinations, and the first diagnostic for every failed combination under the device workspace’s `records/opencl-pmu-sweep/` directory. Do not claim architectural conclusions from a run with missing or unstable rows.

- [ ] **Step 7: Commit documentation and spec.**

```bash
git add replay_benchmark/README.md skills/opencl-pmu-sweep/SKILL.md \
  docs/superpowers/specs/2026-07-25-opencl-pmu-sweep-design.md
git commit --only replay_benchmark/README.md skills/opencl-pmu-sweep/SKILL.md \
  docs/superpowers/specs/2026-07-25-opencl-pmu-sweep-design.md \
  -m "[OpenCL:Doc] Document PMU sweep methodology"
```

## Final acceptance criteria

- Existing 14 PMU cases remain executable on Adreno 740.
- Preparation, image initialization, and warmup occur outside the sampled PMU intervals.
- `workload_runs` executes multiple prepared dispatches inside one workload session.
- Control and workload geometry is reported and reproducible.
- At least one cache-oriented, vector-oriented, throughput-oriented, local-memory, image, constant, and atomic workload is available.
- The sweep invokes exactly one PMU event per benchmark process.
- CSV columns are exactly `case_name,case_args,case_runs,pmu_metric_name,delta_metric`.
- Host build, Python tests, `git diff --check`, and the Rhinopi smoke sweep pass with recorded evidence.
