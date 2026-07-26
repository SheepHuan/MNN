# Vulkan PMU Workload Implementation Plan

## Goal

Run MNN DNN workloads through the Vulkan backend on the target devices and
measure the same device GPU PMU counters around the workload. Keep OpenCL and
Vulkan results in separate CSV families while preserving the one-event,
control-versus-workload measurement contract.

## Stages

1. [x] Add explicit `model-PMU` backend selection for OpenCL (`3`) and Vulkan (`7`),
   with unit coverage and no change to the OpenCL default.
2. [x] Make the complete DNN Session PMU adapter backend-neutral. Remove OpenCL-only
   diagnostics, verify the requested Session backend, and synchronize Vulkan
   outputs before stopping the PMU session.
3. [x] Build and deploy an `MNN_VULKAN=ON` replay benchmark. Run a short OrangePi
   MobileNet Vulkan smoke and prove the report says `forward=7`/`VULKAN` rather
   than falling back to CPU.
4. [~] Add Vulkan model/isolated-operator sweep support with one event per process,
   preserving the existing five-column detail CSV and `valid_cases` summary.
   Complete DNN model sweep support is implemented and smoke-tested; isolated
   operator replay remains the next sub-step.
5. Verify Rhinopi Vulkan loader/driver availability. Run device-specific
   Vulkan sweeps only on devices whose runtime and MNN backend are confirmed.

## Verification checkpoints

- Python and C++ unit tests pass after each code stage.
- `git diff --check` passes.
- The Vulkan smoke must reject CPU fallback; a successful timing alone is not
  evidence that Vulkan executed.
- PMU reports must contain one event, non-empty control/workload intervals, and
  a backend value of `VULKAN`.
- OpenCL records remain unchanged and Vulkan records are device-specific.
