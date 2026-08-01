# MNN Project Instructions

MNN is a lightweight deep learning **inference engine** (not a training framework), targeting mobile and server platforms. Supports CNN / Transformer / LLM / Diffusion models. Code must prioritize **performance and binary size**.

## Restricted Access

> The following directories contain internal proprietary code. **Do NOT read, modify, or reference** any files within:
> - `schema/private/`
> - `source/internal/`

## Architecture Overview

MNN uses a **graph optimization + heterogeneous backend scheduling** architecture.

Two inference APIs are available:
- **Session API** (low-level): `Interpreter → createSession → runSession`, operates on Tensor directly
- **Module API** (high-level, recommended): `Module::load → onForward(VARP)`, Express-based dynamic graph. Used by LLM / Diffusion and most modern workloads

**Key abstractions** (see corresponding headers under `source/core/`):
- **Interpreter** / **Session**: model loading and inference session management
- **Backend** / **Execution**: hardware backend abstraction and per-op implementation (CPU/Metal/CUDA/OpenCL/Vulkan/...)
- **Tensor**: data container; internally uses NC4HW4 format (channels packed by 4 for SIMD)
- **Op / Schema**: FlatBuffers-defined operator descriptors (`schema/default/*.fbs`)

**Op registration pattern**: Schema definition → shape inference (`source/shape/`) → Geometry decomposition (optional) → Backend Execution implementation

## LLM Subsystem

MNN supports end-to-end LLM export and inference:

- **Python export** (`transformers/llm/export/`): HuggingFace model → MNN format. Core modules: `llmexport.py` entry point, `utils/model_mapper.py` (model field mapping), `utils/model.py` (unified LlmModel class), `utils/transformers.py` (Attention/Decoder/RoPE export)
- **C++ inference** (`transformers/llm/engine/`): `llm.cpp` (text inference), `omni.cpp` (multimodal: vision/audio), includes KVCache management and sampling strategies

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `include/MNN/` | Public C++ headers |
| `source/core/` | Inference core (Interpreter, Session, Pipeline, Backend) |
| `source/backend/` | Hardware backend implementations (cpu, arm82, metal, cuda, opencl, vulkan, ...) |
| `source/shape/` | Shape inference |
| `source/geometry/` | Geometry computation (op decomposition) |
| `express/` | Express API (high-level dynamic graph, VARP) |
| `schema/default/` | FlatBuffers schema (op definitions) |
| `tools/converter/` | Model converter (ONNX/TF/Caffe → MNN) |
| `transformers/llm/` | LLM export (Python) + inference engine (C++) |
| `transformers/diffusion/` | Diffusion model support |
| `pymnn/` | Python bindings |
| `test/` | Test cases |
| `skills/` | AI Agent Skills |
| `kernel_agent/` | Kernel 级分析代理与平台无关 PMC Interpreter |

## Coding Style

- **C++**: Google Style variant, see `.clang-format`. 4-space indent, 120-char line width, attached braces. Class names `PascalCase`, functions `camelCase`, member variables `mCamelCase`. RTTI and exceptions disabled (`-fno-rtti -fno-exceptions`). Default standard: C++11.
- **Python**: Standard Python conventions
- **Formatting**: `clang-format -i -style=file <file>`

> ⚠️ **禁止 C++ 全局对象（动态初始化）**:不允许在命名空间作用域定义需要动态初始化的非 POD 对象（如 `static const std::string` / `static const json` / 全局 STL 容器，**头文件中同样禁止**,每个包含它的 TU 都会生成一份）。这类对象产生 `__GLOBAL__sub_I_*` 启动初始化函数，会被手淘等打包静态检查拦截（代码安全风险 + 启动性能下降）。替代方案:`constexpr`/POD 常量（直接进 .rodata)、函数内 `static`（懒初始化）、或按需构造返回。

## Build & Test

```bash
# Build C++ (with LLM)
mkdir build && cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON && make -j$(nproc)

# Common CMake options: MNN_BUILD_TEST, MNN_BUILD_CONVERTER, MNN_METAL, MNN_OPENCL,
# MNN_VULKAN, MNN_CUDA, MNN_ARM82, MNN_BUILD_QUANTOOLS, MNN_SUPPORT_TRANSFORMER_FUSE
# Full list: see option() declarations at the top of CMakeLists.txt

# Unit tests
cd build && ./run_test.out

# LLM export
cd transformers/llm/export
python llmexport.py --path /path/to/model --export mnn --hqq --dst_path ./MODEL

# LLM test
cd build
./llm_demo /path/to/MODEL/config.json prompt.txt

# LLM benchmark
./llm_bench -m /path/to/MODEL/config.json
```

Test suite includes: unit tests (`run_test.out`), model tests, conversion tests (ONNX/TF/TFLite/Torch), quantization tests, LLM tests, PyMNN tests. See `test.sh`, `test_stages.json`, and `test/` directory for details.

## Development Workflow

- **Branch**: Use `feature/<short-description>`, for example `feature/llm-streaming`.
- **Commit**: Use a concise, one-line English message: `[Module:Type] Summary`, for example `[LLM:Feature] Add streaming support`.
  - Modules: `LLM`, `CPU`, `Metal`, `CUDA`, `OpenCL`, `Vulkan`, `Core`, `Infra`, `Doc`, etc.
  - Types: `Feature`, `Bugfix`, `Perf`, `Refact`, `Style`, `Doc`, `Test`, `Chore`, `Release`.

## Skills

For the following tasks, **read the Skill entry file first** and execute step by step. Each step must pass its tests before proceeding.

**After non-trivial skill-driven tasks, run Retrospective only when there are reusable lessons.**

上表可能未覆盖全部 skill。遇到打包/发布/摩天轮/MTL/crash/集成等关键词时，**先 `Glob skills/**/SKILL.md` 确认是否有对应 skill**，不要仅凭表格判断。

Public skills are listed below. Environment-dependent skills may exist under `skills/*/SKILL.md`.

### PMC Interpreter 数据源规范

涉及 CUDA PMC rows、kernel latency、CUDA selection plan、source validation 或
`operator_cases.json` 刷新时，必须先阅读：

`docs/superpowers/specs/2026-07-31-pmc-interpreter-cuda-data-sources.md`

分析 `mnn-pmc-dataset/v1` bundle，或修改 `kernel_agent/pmc_interpreter/` 的 dataset 对象、
数据源加载器、语义映射、分析模型、分析层或知识输出时，必须阅读：

`docs/superpowers/specs/2026-07-31-pmc-interpreter-module-physical-semantics.md`

CUDA 原始数据进入 Interpreter 时必须同时遵循两份规范；只分析 Adreno/Mali 或其他已标准化
bundle 时，不强制进入 CUDA 数据源流程。

固定职责链为：
`kernel-adapt -> corpus-audit -> gpu-pmu-sweep -> pmc-source-gate -> pmc-interpreter`。
采集完成不等于可发布，source gate 通过也不等于得到统计或因果结论。

CUDA 分析固定使用以下三份原始数据源：

- `build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv`
- `build-x86-cuda/cuda_kernel_latency.json`
- `replay_benchmark/kernel_corpus/operator_cases.json`

三份文件必须来自同一份 case manifest。修改 `operator_cases.json` 后，必须重新生成
PMC rows 与 latency；不得对来源已变化的旧 CSV 继续使用 `--resume`。

| Skill | Entry File | Trigger |
|-------|-----------|---------|
| Support new LLM | `skills/support-new-llm/SKILL.md` | Add / adapt a new LLM model |
| Add new op | `skills/add-new-op/SKILL.md` | Add a new operator |
| ARM CPU optimization | `skills/arm-cpu-optimize/SKILL.md` | Optimize op performance on ARM CPU |
| OpenCL optimization | `skills/opencl-optimize/SKILL.md` | Optimize op performance on OpenCL |
| Vulkan optimization | `skills/vulkan-optimize/SKILL.md` | Optimize op performance on Vulkan |
| Metal optimization | `skills/metal-optimize/SKILL.md` | Optimize op performance on Metal |
| Bugfix / debugging | `skills/general-debug/SKILL.md` | Diagnose correctness bugs / regressions in MNN — organized by bug category. |
| Run tests / CI | `skills/test-ci/SKILL.md` | Run the regression / CI suite (host or on-device), or add / select / retune a test stage |
| Replay benchmark cross-compile | `skills/replay-benchmark-cross-compile/SKILL.md` | Cross-compile `replay_benchmark` with the ARM GNU Toolchain 11.3 archive |
| Kernel adapt | `skills/kernel-adapt/SKILL.md` | 新增或修复 replay_benchmark kernel、shim、adapter、validator、manifest 与 workload metadata。 |
| Corpus audit | `skills/corpus-audit/SKILL.md` | 独立审计 kernel 忠实性、adapter/launch、validator 和 workload metadata，冻结采集前 manifest。 |
| GPU PMU sweep | `skills/gpu-pmu-sweep/SKILL.md` | 采集或刷新 Adreno/Mali/NVIDIA PMC，生成 availability、selection plan、rows 与独立 latency。 |
| PMC source gate | `skills/pmc-source-gate/SKILL.md` | 严格验收 manifest、plan、rows、latency、session/environment 和发布身份。 |
| PMC Interpreter | `skills/pmc-interpreter/SKILL.md` | 将已验收 PMC 数据转换为相关性分类、Canonical PMC、Kernel Signature、Delta Rulebook 和中文报告。 |
| Retrospective | `skills/retrospective/SKILL.md` | After non-trivial tasks with reusable lessons |
