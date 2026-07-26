# Model PMU Workload Implementation Plan

> **For agentic workers:** Execute the tasks in order with TDD checkpoints.

**Goal:** Add a replay_benchmark mode that measures complete MNN model Session execution with one PMU event per process, suitable for DNN model sweeps on both devices.

**Architecture:** Keep the existing synthetic OpenCL PMU mode unchanged. Add a separate model-PMU adapter that loads one `.mnn`, creates one native `Interpreter`/`Session`, fills inputs, performs warmup outside PMU, then samples a control interval and repeated full `runSession()` workload intervals. Emit one JSON report per model/event; the Python skill converts reports to per-device CSV. LLM paths are represented by an explicit adapter boundary and are rejected with a diagnostic until the existing LLM engine entry point is wired.

**Tech Stack:** C++11, MNN Interpreter/Session API, MNNPerfCounter, RapidJSON, Python sweep script, CMake.

---

### Task 1: Define model-PMU data contract and CLI mode

**Files:**
- Create: `replay_benchmark/ModelPmuBenchmark.hpp`
- Create: `replay_benchmark/ModelPmuBenchmark.cpp`
- Modify: `replay_benchmark/ReplayRecord.hpp`
- Modify: `replay_benchmark/replay_benchmark.cpp`
- Test: `replay_benchmark/OpenCLPmuBenchmarkTest.cpp`

- [x] Add pure helpers for model workload defaults, signed delta, and CSV/JSON field names.
- [x] Add `--model-pmu-bench`, `--model-pmu-workload-runs`, `--model-pmu-warmup-runs`, `--model-pmu-control-runs`, and `--model-pmu-precision` parsing.
- [x] Make model-PMU mode require a single model path, OpenCL forward, one event, and output path.
- [x] Write failing unit assertions for defaults, signed delta, and mode dispatch; run the test to observe failure.
- [x] Implement the minimum helpers and parser changes; rerun the test.

### Task 2: Implement complete DNN Session measurement

**Files:**
- Modify: `replay_benchmark/ModelPmuBenchmark.cpp`
- Modify: `replay_benchmark/CMakeLists.txt`
- Test: `replay_benchmark/OpenCLPmuBenchmarkTest.cpp`

- [x] Create/load an MNN `Interpreter`, create the requested OpenCL Session, fill every input, and finish all warmup before PMU.
- [x] Create a PMU session for exactly one event, run a control interval, then run the complete model Session repeatedly while PMU is active.
- [x] Synchronize outputs before stopping PMU and report status, `control_delta`, `workload_delta`, signed `delta_metric`, run counts, and errors.
- [x] Add and execute opt-in/device smoke coverage for DNN and LLM Session report shapes.

### Task 3: Integrate model sweep into the existing skill

**Files:**
- Modify: `skills/opencl-pmu-sweep/scripts/sweep_opencl_pmu.py`
- Modify: `skills/opencl-pmu-sweep/SKILL.md`
- Modify: `skills/opencl-pmu-sweep/tests/test_sweep_opencl_pmu.py`

- [x] Add model discovery for `benchmark/models`, DNN prebuilts, and GenAI `.mnn` files without merging device results.
- [x] Add model command construction and model report extraction while preserving one event per process.
- [x] Add per-device model raw/summary CSV output and valid-case aggregation.
- [x] Add tests for model command arguments, model row extraction, and separate device output names.

### Task 4: Build and device validation

**Files:**
- Modify: `.gitignore` only if new sweep output paths are not already ignored.

- [x] Build AArch64/unit targets and run Python tests.
- [x] Cross-build and deploy to Rhinopi and OrangePi.
- [ ] Run the full model/event sweep on both devices; smoke tests are complete.
- [x] Verify reports contain real Session execution intervals and keep all generated records ignored.
