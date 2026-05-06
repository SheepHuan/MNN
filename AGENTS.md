# Agent Instructions for MNN

These instructions apply to the whole repository unless a deeper `AGENTS.md`
overrides them.

MNN is a lightweight deep learning inference engine, not a training framework.
It targets mobile and server platforms and supports CNN, Transformer, LLM, and
Diffusion workloads. Changes should prioritize runtime performance, memory use,
and binary size.

## Restricted Access

Do not read, modify, summarize, or reference files under these internal
proprietary directories:

- `schema/private/`
- `source/internal/`

If a task appears to require those paths, stop and ask the user for direction.

## Architecture Overview

MNN uses graph optimization plus heterogeneous backend scheduling.

Primary inference APIs:

- Session API: low-level `Interpreter -> createSession -> runSession`, operating
  directly on `Tensor`.
- Module API: high-level and recommended for modern workloads,
  `Module::load -> onForward(VARP)`, Express-based dynamic graph used by LLM,
  Diffusion, and related features.

Key abstractions:

- `Interpreter` / `Session`: model loading and inference session management.
- `Backend` / `Execution`: hardware backend abstraction and per-op
  implementations for CPU, Metal, CUDA, OpenCL, Vulkan, and others.
- `Tensor`: data container; internal layouts often use NC4HW4 packing for SIMD.
- `Op` / Schema: FlatBuffers operator descriptors in `schema/default/*.fbs`.

Operator implementation flow:

1. Add or update schema definition.
2. Add shape inference under `source/shape/`.
3. Add geometry decomposition under `source/geometry/` when needed.
4. Add backend `Execution` implementations for relevant backends.
5. Add focused tests or model/converter coverage according to the change.

## LLM Subsystem

MNN supports end-to-end LLM export and inference.

- Python export lives under `transformers/llm/export/`.
  Important files include `llmexport.py`, `utils/model_mapper.py`,
  `utils/model.py`, and `utils/transformers.py`.
- C++ inference lives under `transformers/llm/engine/`.
  Important files include `llm.cpp` and `omni.cpp`, with KV cache management and
  sampling logic nearby.

## Repository Map

- `include/MNN/`: public C++ headers.
- `source/core/`: inference core such as `Interpreter`, `Session`, `Pipeline`,
  and `Backend`.
- `source/backend/`: hardware backend implementations.
- `source/shape/`: shape inference.
- `source/geometry/`: geometry computation and op decomposition.
- `express/`: Express API and `VARP` dynamic graph support.
- `schema/default/`: public FlatBuffers schema definitions.
- `tools/converter/`: model converter for ONNX, TensorFlow, Caffe, and related
  formats.
- `transformers/llm/`: LLM export and inference engine.
- `transformers/diffusion/`: Diffusion model support.
- `pymnn/`: Python bindings.
- `test/`: test cases.
- `skills/`: repository-local task guides for complex agent workflows.

## Coding Style

- C++ follows the repository `.clang-format` style: 4-space indentation,
  120-column line width, attached braces, class names in `PascalCase`,
  functions in `camelCase`, and member variables like `mCamelCase`.
- C++ defaults to C++11 unless a target explicitly requires otherwise.
- RTTI and exceptions are disabled in normal builds; do not introduce code that
  depends on them.
- Python follows standard Python conventions and the local style of nearby code.
- Keep changes narrowly scoped. Avoid unrelated refactors and metadata churn.
- Use structured parsers or existing project utilities instead of ad hoc string
  manipulation when practical.
- Format edited C++ files with:

```bash
clang-format -i -style=file <file>
```

## Build and Test

Common C++ build with LLM support:

```bash
mkdir -p build
cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
make -j$(nproc)
```

Useful CMake options include `MNN_BUILD_TEST`, `MNN_BUILD_CONVERTER`,
`MNN_METAL`, `MNN_OPENCL`, `MNN_VULKAN`, `MNN_CUDA`, `MNN_ARM82`,
`MNN_BUILD_QUANTOOLS`, and `MNN_SUPPORT_TRANSFORMER_FUSE`. Check
`CMakeLists.txt` for the current authoritative option list.

Unit tests:

```bash
cd build
./run_test.out
```

LLM export:

```bash
cd transformers/llm/export
python llmexport.py --path /path/to/model --export mnn --hqq --dst_path ./MODEL
```

LLM inference test:

```bash
cd build
./llm_demo /path/to/MODEL/config.json prompt.txt
```

LLM benchmark:

```bash
cd build
./llm_bench -m /path/to/MODEL/config.json
```

The broader test surface includes unit tests, model tests, conversion tests,
quantization tests, LLM tests, and PyMNN tests. See `test.sh` and `test/` for
task-specific coverage.

## Repository Skills

For these task types, read the listed skill file before changing code and follow
its workflow. Run the relevant tests after each meaningful step.

- Support a new LLM model: `skills/support-new-llm/SKILL.md`
- Add a new operator: `skills/add-new-op/SKILL.md`
- Optimize ARM CPU performance: `skills/arm-cpu-optimize/SKILL.md`
- Retrospective after non-trivial skill work: `skills/retrospective/SKILL.md`

If a skill conflicts with this file, follow the more specific skill instructions
for that task while still respecting the restricted-access directories above.

## Agent Workflow Notes

- Prefer `rg` / `rg --files` for repository search.
- Read nearby code before editing and match existing patterns.
- Preserve user changes in the working tree. Do not revert unrelated edits.
- Do not run destructive git commands unless the user explicitly asks.
- When working under a subdirectory with its own `AGENTS.md`, follow that file
  for the files in that subtree.
