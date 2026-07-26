# GPU Kernel Operator Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract historical OpenCL/Vulkan kernel units into normalized case metadata and run selected cases through the current MNN GPU runtime.

**Architecture:** A Python extractor discovers immutable source facts and writes `operators.json`; a hand-authored case registry supplies only the input/dispatch/reference data needed for reliable execution. `replay_benchmark` then runs those cases through the current OpenCL/Vulkan runtime and reports compile, dispatch, validation, and PMU status without linking historical backends.

**Tech Stack:** Python 3, JSON, C++11, current MNN OpenCL/Vulkan runtime, OpenCL C, GLSL compute shaders, existing replay benchmark JSON reporting.

---

### Task 1: Add failing extractor tests

**Files:**
- Create: `replay_benchmark/kernel_corpus/tests/test_operator_extractor.py`
- Test fixtures: inline temporary OpenCL and GLSL files

- [ ] **Step 1: Write tests for entry discovery and path safety**

  Test that `__kernel void unary_buf(...)` yields one OpenCL record, a GLSL file with
  `void main()` yields one Vulkan record, restricted paths are rejected, and duplicate
  `(framework, tag, backend, source, entry)` keys are rejected.

- [ ] **Step 2: Run the focused test and verify the expected import/attribute failure**

  Run:

  ```bash
  python3 -m unittest replay_benchmark.kernel_corpus.tests.test_operator_extractor
  ```

  Expected: FAIL because `extract_operator_kernels.py` does not exist yet.

### Task 2: Implement source extraction and schema validation

**Files:**
- Create: `replay_benchmark/kernel_corpus/extract_operator_kernels.py`
- Modify: `replay_benchmark/kernel_corpus/tests/test_operator_extractor.py`

- [ ] **Step 1: Implement deterministic manifest scanning**

  Read `manifest.json`, select paths under MNN OpenCL/Vulkan and ncnn Vulkan, scan only
  `.cl` and `.comp`, preserve source-relative paths, and emit sorted records with
  `execution_spec: "manual"`.

- [ ] **Step 2: Implement minimal declaration scanners**

  Use anchored regular expressions for OpenCL `__kernel` declarations and Vulkan
  `void main()` plus `layout(binding = N)`, push-constant, and specialization declarations.
  The scanner must report malformed declarations as errors instead of guessing.

- [ ] **Step 3: Add CLI and generated artifact**

  Support `--manifest`, `--root`, `--output`, and optional `--framework`/`--tag` filters;
  write stable JSON and no files outside the output path.

- [ ] **Step 4: Run the focused tests and corpus validation**

  ```bash
  python3 -m unittest replay_benchmark.kernel_corpus.tests.test_operator_extractor
  python3 replay_benchmark/kernel_corpus/extract_operator_kernels.py \
    --manifest replay_benchmark/kernel_corpus/manifest.json \
    --root replay_benchmark/kernel_corpus \
    --output replay_benchmark/kernel_corpus/operators.json
  ```

  Expected: all extractor tests pass and `operators.json` is deterministic.

### Task 3: Add explicit runnable smoke case registry

**Files:**
- Create: `replay_benchmark/kernel_corpus/operator_cases.json`
- Create: `replay_benchmark/kernel_corpus/tests/test_operator_cases.py`
- Modify: `replay_benchmark/kernel_corpus/README.md`

- [ ] **Step 1: Write schema tests**

  Verify every case references an extractor record, has one backend, positive dispatch
  dimensions, bounded buffer sizes, and a reference validator name.

- [ ] **Step 2: Add initial cases**

  Add OpenCL `buffer_set_zero`, `unary_buf`, `matmul_buf` and Vulkan `sigmoid`, `tanh`,
  `permute` for representative early/latest tags, with explicit bindings and expected
  outputs. Keep unsupported or layout-dependent kernels out of the runnable registry.

- [ ] **Step 3: Run tests and reject invalid references**

  ```bash
  python3 -m unittest replay_benchmark.kernel_corpus.tests.test_operator_cases
  ```

### Task 4: Add generic OpenCL corpus runner

**Files:**
- Create: `replay_benchmark/KernelCorpusBenchmark.hpp`
- Create: `replay_benchmark/KernelCorpusBenchmark.cpp`
- Modify: `replay_benchmark/replay_benchmark.cpp`
- Modify: `replay_benchmark/CMakeLists.txt`
- Create: `replay_benchmark/KernelCorpusBenchmarkTest.cpp`

- [ ] **Step 1: Write metadata and failure-mode tests**

  Test option parsing, missing source classification, compile failure classification, and
  the JSON status fields before adding GPU code.

- [ ] **Step 2: Implement current-runtime OpenCL execution**

  Load one case, build the source with only case-declared preamble macros, create the
  described buffers, set scalar arguments in declared order, enqueue warmups and dispatches,
  read back output, and run the named reference validator.

- [ ] **Step 3: Add CLI and report contract**

  Add `--kernel-corpus-bench`, `--kernel-corpus-case`, `--kernel-corpus-root`, and
  `--kernel-corpus-runs`; emit `compile_status`, `dispatch_status`, `validation_status`,
  and `pmu_status` without aborting other cases.

- [ ] **Step 4: Build and run host metadata tests**

  ```bash
  cmake --build build-aarch64-gnueabihf-vulkan --target replay_benchmark.out kernel_corpus_benchmark_test -j2
  build-aarch64-gnueabihf-vulkan/replay_benchmark/kernel_corpus_benchmark_test
  ```

### Task 5: Add generic Vulkan corpus runner

**Files:**
- Modify: `replay_benchmark/KernelCorpusBenchmark.hpp`
- Modify: `replay_benchmark/KernelCorpusBenchmark.cpp`
- Modify: `replay_benchmark/CMakeLists.txt`
- Modify: `replay_benchmark/replay_benchmark.cpp`

- [ ] **Step 1: Add failing Vulkan descriptor tests**

  Verify binding/push-constant/specialization parsing and unsupported capability reporting.

- [ ] **Step 2: Implement current-runtime Vulkan execution**

  Compile GLSL to SPIR-V using the configured shader compiler path, create a compute
  pipeline from explicit descriptor metadata, upload storage buffers and push constants,
  dispatch, synchronize, read back, and validate.

- [ ] **Step 3: Run host Vulkan compile smoke and device dispatch smoke**

  Host command must classify missing Vulkan device as `unsupported`, while an Android
  device run must produce `validation_passed` for at least `sigmoid`.

### Task 6: Integrate PMU and documentation

**Files:**
- Modify: `replay_benchmark/KernelCorpusBenchmark.cpp`
- Modify: `replay_benchmark/README.md`
- Modify: `replay_benchmark/kernel_corpus/README.md`
- Modify: `skills/opencl-pmu-sweep/SKILL.md`

- [ ] **Step 1: Add control/workload PMU intervals around generic dispatch**

  Reuse the existing one-session workload-runs and warmup semantics; do not sample source
  preparation or compilation.

- [ ] **Step 2: Add sweep-compatible JSON/CSV fields**

  Preserve `framework`, `tag`, `backend`, `operator`, `case`, `status`, `validation`, and
  PMU deltas so the existing sweep can merge historical operator rows.

- [ ] **Step 3: Document supported and non-runnable cases**

  Explain that discovery is broader than executable coverage and that each runnable case
  has an explicit adapter contract.

- [ ] **Step 4: Run final verification**

  ```bash
  python3 -m unittest discover -s replay_benchmark/kernel_corpus/tests
  python3 replay_benchmark/kernel_corpus/validate_kernel_sources.py \
    --manifest replay_benchmark/kernel_corpus/manifest.json \
    --root replay_benchmark/kernel_corpus
  git diff --check
  ```
