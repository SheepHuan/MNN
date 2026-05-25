---
name: mnn-add-new-op
description: 当用户要求在本 MNN 仓库中新增或扩展 MNN 算子、更新 schema/op type/shape/CPU 后端/注册表、实现 PrefixAttention 这类自定义算子，或把 skills/add-new-op 的流程用于本仓库时使用这个 Skill。
metadata:
  short-description: 在本 MNN 仓库中新增或扩展 MNN 算子
---

# MNN Add New Op

本 Skill 由本 MNN 仓库维护，是 `skills/add-new-op` 的 Codex 适配版本。它把 MNN 新增算子的 TDD 流程整理成本仓库可执行的流程；构建和生成产物放在本仓库 `.cache/` 与 `output/`。

## 工作流

1. 确认当前目录是本 MNN 仓库根目录，先读根目录 `AGENTS.md`。
2. 修改源码前先运行：

```bash
git status --short
```

不要覆盖用户已有改动。不要读取或修改 `schema/private/` 和 `source/internal/`。

3. 先确认是否已有可复用算子、参数或后端实现。只有确实需要新语义时才新增 `OpType`；能复用现有 `AttentionParam`、shape computer 或 CPU creator 时优先复用。
4. 按顺序做：Schema → shape → compute/backend → 注册生成 → 构建 → 测试。几何计算优先；无法拆解或性能敏感时再写 CPU 后端。

## Schema

在 `schema/default/MNN.fbs` 里只向枚举末尾追加新 `OpType`，不要插入已有枚举中间。如果需要新参数 table，也只追加到 `OpParameter` union 末尾。

生成 schema：

```bash
bash schema/generate.sh
```

检查 `schema/current/MNN_generated.h` 中是否出现新 `OpType` 或新参数结构。

## Shape 和注册

输出形状不同或需要 transformer fuse 特殊输入时，更新 `source/shape/Shape*.cpp`。普通算子使用 `REGISTER_SHAPE` / `REGISTER_SHAPE_INPUTS`；Transformer attention 类算子优先复用现有 `ShapeAttention.cpp` 和 `REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE`。

更新注册表：

```bash
python3 tools/script/register.py .
```

`register.py` 可能更新多个 `*OPRegister.cpp` 文件；检查 diff，确认只是注册表生成变化。

## Compute

优先判断能否在 `source/geometry/` 拆解为已有算子组合。不能拆解时，在对应 backend 中实现；本仓库默认先完成 CPU：

```text
source/backend/cpu/CPU*.cpp
source/backend/cpu/CPU*.hpp
```

CPU creator 注册常用：

```cpp
REGISTER_CPU_OP_CREATOR(CPUMyOpCreator, OpType_MyOp);
```

Transformer fused attention 类算子常用：

```cpp
REGISTER_CPU_OP_CREATOR_TRANSFORMER(CPUAttentionCreator, OpType_PrefixAttention);
```

如果新 op 需要参与 KV cache、session clone、算子类型归类或 transformer fuse，检查并按需更新：

```text
source/core/OpCommonUtils.cpp
source/core/Pipeline.cpp
source/core/Session.cpp
```

## PrefixAttention 特例

当前 prefix LLM 方向优先走“新 op type + runtime KVMeta metadata”，不要把 `prefix_key/prefix_value/prefix_len` 强行做成模型图普通 tensor 输入，除非用户明确要求。

维护 `PrefixAttention` 时重点检查：

```text
schema/default/MNN.fbs
schema/current/MNN_generated.h
source/shape/ShapeAttention.cpp
source/backend/cpu/CPUAttention.cpp
source/core/OpCommonUtils.cpp
source/core/Pipeline.cpp
source/core/Session.cpp
transformers/pic_llm/export/utils/custom_op.py
transformers/pic_llm/export/utils/mnn_converter.py
transformers/pic_llm/export/llmexport.py
```

第一阶段可以让 `PrefixAttention` 复用 `AttentionParam`、shape computer 和 CPUAttention creator；后续再把 prefix cache 的 RoPE 读取/合并逻辑从通用 KV manager 收敛到 PrefixAttention 专属路径。

## 构建和测试

完整构建使用 `$mnn-build-artifacts` 中的本仓库构建入口。需要本地 converter 时：

```bash
cmake -S . -B .cache/build/mnn/x64 -DMNN_BUILD_CONVERTER=ON
cmake --build .cache/build/mnn/x64 --target MNNConvert -j "${JOBS:-$(nproc)}"
```

普通算子应补 `test/op/*Test.cpp` 并运行对应测试。LLM attention 类算子还要用 `$mnn-llm-export` 导出含新 op 的模型，并用 `$mnn-llm-bench` 或 `$pic-kvcache-server` 做 CPU 冒烟验证。

## 输出约束

不要把 `.cache/`、`output/`、模型缓存、ONNX 临时文件、构建目录或导出结果加入 git。MNN 专属 Codex Skill 放在 `.codex/skills/`，由 MNN 本仓库自己维护。
