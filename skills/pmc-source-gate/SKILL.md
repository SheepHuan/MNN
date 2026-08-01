---
name: pmc-source-gate
description: 对 PMC 原始数据执行独立、严格、可审计的发布验收。用于检查 CUDA manifest、CaseSelectionPlan、availability、PMC long rows、独立 latency、session/environment 和笛卡尔积完整性，生成 mnn-cuda-pmc-source-validation/v1；位于 gpu-pmu-sweep 与 pmc-interpreter 之间，不负责设备采集、kernel 正确性或统计解释。
---

# PMC Source Gate：原始数据发布门禁

把本 Skill 作为设备采集与语义解释之间的独立门禁：

~~~text
kernel-adapt
  -> corpus-audit
  -> gpu-pmu-sweep
  -> pmc-source-gate
  -> pmc-interpreter
~~~

只在门禁输出 `valid=true` 后，把本轮原始数据交给 `pmc-interpreter`。loader 能读取文件、
CSV 行数看似完整或 Markdown 能生成，都不能代替发布验收。

按副作用划分边界：`gpu-pmu-sweep` 可以接触设备、`sudo` 和长时间 profiler 状态；本 Skill
必须离线、只读、确定性执行，不能启动采集，也不能修改输入来让验证通过。

## 1. 必读规范

开始验收或修改本 Skill 前，完整阅读：

- `docs/superpowers/specs/2026-07-31-pmc-interpreter-cuda-data-sources.md`

如果同时修改共享 `CaseSelectionPlan`、Pydantic 数据契约或 Interpreter source adapter，还要
完整阅读：

- `docs/superpowers/specs/2026-07-31-pmc-interpreter-module-physical-semantics.md`

遵守仓库 `AGENTS.md`，禁止读取或修改 `schema/private/`、`source/internal/`。

## 2. Ownership

本 Skill 拥有：

- 原始数据 schema、集合、顺序、数值状态和引用完整性校验；
- manifest、selection plan、availability、rows 与 latency 的身份一致性校验；
- PMC session 连续性、共享字段和 environment 完整性校验；
- P0 target/K、显式 shape/workload/provenance 和 latency 全覆盖要求；
- 结构化门禁结果及失败诊断；
- 当前 CUDA source gate 的发布语义。

本 Skill 不拥有：

- kernel、shim、adapter、validator 或 manifest 的实现与正确性审计；
- metric discovery、PMC rows、latency 或环境信息的采集；
- 修改原始文件来“修复”验收失败；
- dataset 统计层、相关性、Kernel Signature 或 Delta Rulebook；
- 用 loader 的兼容模式、忽略 issue 或删行来绕过失败。

实现或 metadata 有问题时回到 `kernel-adapt`，正确性证据有问题时回到 `corpus-audit`，
采集缺失或身份陈旧时回到 `gpu-pmu-sweep`。

## 3. 固定输入与输出

当前 CUDA gate 接受同一轮实验的五份输入：

| 输入 | 固定角色 |
| --- | --- |
| `operator_cases.json` | 全部 CUDA condition、shape、workload 和 provenance |
| `cuda_kernel_latency.json` | 关闭 PMU 后的全部 CUDA case latency、launch/resource 和 environment |
| `rows.csv.selection.json` | `mnn-pmc-case-selection/v1` 采样设计 |
| `cuda_pmc_valid.csv` | 本设备、本轮 availability 发布的有序 metric 清单 |
| `cuda_kernel_pmc_kernelreplay_rows.csv` | plan-selected case × ordered metric 的完整状态账本 |

采集交接还应提供 manifest SHA、设备/driver/profiler 指纹、collector binary 身份、原始日志和
sweep 配置。当前脚本不读取 `corpus-audit` 放行记录；它是开始采集和验收的前置证据，不能因
脚本返回 0 而省略。

固定输出：

~~~text
schema_version = mnn-cuda-pmc-source-validation/v1
valid = true | false
errors
warnings
动态 case / metric / row / session 数量
source hashes 与 plan/sweep identity
~~~

输出 JSON 只证明原始数据可发布，不是 `mnn-pmc-dataset/v1`，也不包含统计解释。

## 4. CaseSelectionPlan 的共享边界

`CaseSelectionPlan` 定义在：

~~~text
kernel_agent/pmc_interpreter/dataset/selection.py
~~~

把它视为采集、门禁和解释共同使用的实验设计契约：

- `gpu-pmu-sweep` 从冻结 manifest 生成计划并按计划采集；
- `pmc-source-gate` 使用当前算法完整重建计划并验收 rows；
- `pmc-interpreter` 把已验收计划写入 dataset provenance，限定 PMC condition 范围。

模型位于 Interpreter 包中不表示 Interpreter 拥有设备采集。不要在三个阶段分别实现名称相似
但规则不同的 selection 算法，也不要恢复旧 `op-type` 选择别名。

## 5. CUDA 严格验收

确认采集已结束、输入文件不再写入后执行：

~~~bash
cd build-x86-cuda

python3 ../skills/pmc-source-gate/scripts/validate_cuda_pmc_sources.py \
  --operator-cases ../replay_benchmark/kernel_corpus/operator_cases.json \
  --latency-json <staging-dir>/cuda_kernel_latency.json \
  --selection-plan <staging-dir>/cuda_kernel_pmc_kernelreplay_rows.csv.selection.json \
  --valid-csv <staging-dir>/cuda_pmc_valid.csv \
  --pmc-rows <staging-dir>/cuda_kernel_pmc_kernelreplay_rows.csv \
  --require-complete-targets \
  --require-measured-environment \
  --output-json <staging-dir>/cuda_pmc_source_validation.json
~~~

P0 发布必须同时满足：

- 命令返回 0，输出 schema 正确且 `valid=true`；
- manifest SHA 与 latency、plan 完全一致；
- plan 使用当前 `condition-balanced` policy version、固定 K=5 和精确七类 P0 target；
- 七类 group 全部满足 condition 要求，selected case 的语义和 provenance 与 manifest 一致；
- latency 覆盖当前 manifest 的全部 CUDA case，值为正数有限值；
- selected latency 与所有 PMC session 都具有实测正 clock/temperature 和非空来源；
- rows 表头、plan ID、sweep config ID、case/metric 顺序和状态合法；
- rows 是 plan-selected case × ordered valid metric 的完整有序笛卡尔积；
- 同 session rows 连续，且共享 case、order、PMU 结果字段和 environment；
- CUDA `VALID` metric 值有限且非负，非 `VALID` 状态完整保留。

不要在 Skill 中写死最终 case 数、metric 数、rows 数、plan ID 或 manifest SHA；这些只能由当前
输入动态产生。

## 6. 失败分流

按失败事实回到唯一上游：

| 失败类型 | 回到 |
| --- | --- |
| manifest case、shape/workload、provenance 不正确 | `kernel-adapt`，修复后重新 `corpus-audit` |
| kernel/adapter/validator 正确性证据不足 | `corpus-audit` |
| availability、latency、rows 缺失或环境未采到 | `gpu-pmu-sweep` |
| manifest、metric、collector、设备或 sweep identity 混用 | 新 staging 从空账本重采 |
| gate schema 或校验代码缺陷 | 本 Skill |
| 相关性、signature 或 delta 证据不足 | `pmc-interpreter`，不得放宽 source gate |

失败时保留原始输入和 `valid=false` 诊断 JSON，不覆盖固定正式路径，不删除失败行，不生成新的
正式 signature 或 rulebook。

## 7. 交接给 PMC Interpreter

交接记录至少包含：

~~~text
cuda_pmc_source_validation.json: valid=true
operator_cases.json + manifest_sha256
cuda_kernel_latency.json
cuda_kernel_pmc_kernelreplay_rows.csv
cuda_kernel_pmc_kernelreplay_rows.csv.selection.json + plan_id
cuda_pmc_valid.csv
device/driver/profiler fingerprint
collector binary identity
~~~

随后使用 `pmc-interpreter` 构造 `mnn-pmc-dataset/v1`。门禁不会自动运行 Interpreter，也不会
把 source validity 提升为相关性或因果证据。

## 8. 修改与验证

修改 gate 脚本或共享 selection 契约后运行：

~~~bash
git diff --check

.venv/bin/python -m unittest discover \
  -s skills/pmc-source-gate/tests \
  -p 'test_*.py'

.venv/bin/python -m unittest discover \
  -s kernel_agent/tests/pmc_interpreter \
  -p 'test_*.py'
~~~

这些测试不替代真实设备采集，只验证发布契约和拒绝路径。
