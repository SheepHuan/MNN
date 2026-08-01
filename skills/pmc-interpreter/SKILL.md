---
name: pmc-interpreter
description: 将已验收的 CUDA、Adreno、Mali PMC 数据或 mnn-pmc-dataset/v1 bundle 转换为平台无关的相关性分类、Canonical PMC、Kernel Type × PMC Signature、Delta Rulebook 和中文 Markdown。用于构造或检查 PmcDataset、MetricRegistry、KernelTaxonomy、串行分析 layers、mnn-pmc-analysis-report/v1，或修改 kernel_agent/pmc_interpreter/；不负责 kernel 适配、corpus 正确性放行、设备采集或 CUDA source gate。
---

# PMC Interpreter：平台无关语义解释

本 Skill 只消费已经结构化并通过来源门禁的实验事实，回答 PMC 的三个语义问题：

1. 哪些 PMC 在控制工作量后与 latency 存在稳定关联，它们属于哪类用途；
2. 每个 Kernel Type 应重点观察哪些 PMC signature；
3. 在合法 implementation pair 中，`Delta PMC` 与 `Delta latency` 有什么条件关系。

设备采集由 `gpu-pmu-sweep` 负责，原始数据发布由 `pmc-source-gate` 负责；kernel 实现和
manifest 修改由 `kernel-adapt` 负责；独立正确性放行由 `corpus-audit` 负责。

开始前遵守仓库 `AGENTS.md`，禁止读取或修改 `schema/private/`、`source/internal/`。

## 1. 必读资料

分析标准化 bundle 或修改 `kernel_agent/pmc_interpreter/` 前完整阅读：

- `docs/superpowers/specs/2026-07-31-pmc-interpreter-module-physical-semantics.md`
- `kernel_agent/pmc_interpreter/README.md`

如果输入来自 CUDA 三原始数据源，还必须完整阅读：

- `docs/superpowers/specs/2026-07-31-pmc-interpreter-cuda-data-sources.md`

## 2. Ownership

本 Skill 拥有：

- `mnn-pmc-dataset/v1` 的 Pydantic 数据契约和平台 source adapter；
- native metric 到平台无关 `concept_id` 的 catalog 映射；
- Projection、Quality、Redundancy、Correlation、KernelSignature、DeltaRelation 固定串行层；
- `mnn-pmc-analysis-report/v1` 的证据、warning 和 readiness 编译；
- `mnn-pmc-markdown-document/v1` 中文 Markdown 渲染；
- 分析配置、统计实现、解释边界和相关测试。

本 Skill 不拥有：

- kernel、shim、adapter、kernel correctness validator 或 manifest 的实现；
- metric availability、PMC rows、独立 latency 或 profiler session 的采集；
- CUDA manifest/plan/rows/latency 的 strict source gate；
- 把多个设备的 native 数值或 effect slope 合并建模；
- 在没有合法 implementation pair 时制造 delta 规则；
- 把跨 kernel correlation 表述为优化因果关系。

## 3. 固定输入与交接门槛

优先输入已经标准化的：

~~~text
mnn-pmc-dataset/v1
~~~

它必须保存 device、condition、run、metric descriptor、PMC/latency observation 和 comparison
pair，而不是只有一个无语义数值矩阵。

CUDA raw-source 模式必须先从 `pmc-source-gate` 收到：

~~~text
cuda_pmc_source_validation.json
  schema_version = mnn-cuda-pmc-source-validation/v1
  valid = true

operator_cases.json + manifest_sha256
cuda_kernel_pmc_kernelreplay_rows.csv
cuda_kernel_latency.json
cuda_kernel_pmc_kernelreplay_rows.csv.selection.json + plan_id
cuda_pmc_valid.csv
platform/backend/namespace
device/driver/profiler fingerprint
~~~

strict gate 未通过时只能保留诊断数据，不得生成或发布新的正式 signature、rulebook 或报告。
一次分析只能包含一个 `device_id`；跨设备只在报告完成后按 `concept_id` 比较语义。

## 4. 固定输出

| 产物 | Schema | 物理意义 |
| --- | --- | --- |
| 标准化数据集 | `mnn-pmc-dataset/v1` | 不可丢失实验身份的统一事实 |
| 结构化报告 | `mnn-pmc-analysis-report/v1` | 权威、可机读的分析知识 |
| 中文文档 | `mnn-pmc-markdown-document/v1` | 面向人的报告视图，不替代 JSON |

报告至少保留：

- metric quality、用途分类、相关证据和冗余簇；
- descriptive canonical 与 latency-association 集合；
- Kernel Type × PMC Signature 及来源层级；
- delta rulebook、有效区间、证据等级和配对样本；
- warnings、data readiness 和不可回答的问题。

## 5. 串行组件契约

平台差异只允许位于 `dataset/sources/` 和 `dataset/catalog/`。进入统一数据集后，固定执行：

~~~text
ProjectionOutput
  -> QualityOutput
  -> RedundancyOutput
  -> CorrelationOutput
  -> KernelSignatureOutput
  -> DeltaRelationOutput
  -> semantics compiler
  -> Markdown renderer
~~~

每个 layer 只接受上一层的具体 Pydantic 输出，不接收松散 dict，不跳层、不重排：

- Projection：从 raw observation 构造可解释 FeatureSpec；
- Quality：处理覆盖率、常量、稳定性和无效值；
- Redundancy：按公式、统计和机制聚类，选择代表指标；
- Correlation：控制 workload/shape/launch/environment 后输出观察性关联分类；
- KernelSignature：生成 semantic family/op type 的诊断指标集合；
- DeltaRelation：只分析同设备、同语义、合法配对的实现差分；
- `semantics/`：只编译既有结果和渲染 Markdown，不做新统计。

## 6. 三个问题的证据边界

### 6.1 相关性与分类

主要解释 residual latency，区分表征、诊断、优化、target-coupled、redundant 和 unstable
指标。输出是条件关联，不是因果结论。

### 6.2 Kernel Type × PMC Signature

signature 是类别诊断面板。必须注明它来自 op-type 直接实证、semantic-family 直接结果，还是
family 先验继承；不能把 inheritance 写成该 op type 的直接统计证据。

### 6.3 Delta PMC × Delta latency

只使用同设备、同计算语义、baseline/candidate 明确且 paired run 合法的 comparison pair。
样本或干预证据不足时，空 `delta_rulebook` 是正确结果。不得用第一问或第二问补造第三问。

## 7. 标准执行

分析标准化 bundle：

~~~bash
python3 -m kernel_agent.pmc_interpreter \
  --input-bundle <pmc-dataset-v1.json> \
  --output-json <pmc-analysis-report-v1.json> \
  --output-md <pmc-analysis-report-v1.md>
~~~

分析已通过 strict gate 的 CUDA 三数据源：

~~~bash
python3 -m kernel_agent.pmc_interpreter \
  --pmc-csv <staging-dir>/cuda_kernel_pmc_kernelreplay_rows.csv \
  --latency-json <staging-dir>/cuda_kernel_latency.json \
  --operator-cases replay_benchmark/kernel_corpus/operator_cases.json \
  --case-selection-plan <staging-dir>/cuda_kernel_pmc_kernelreplay_rows.csv.selection.json \
  --platform cuda \
  --backend cuda \
  --namespace cupti \
  --normalized-output-json <staging-dir>/cuda_pmc_dataset_v1.json \
  --output-json <staging-dir>/cuda_pmc_analysis_report_v1.json \
  --output-md <staging-dir>/cuda_pmc_analysis_report_v1.md
~~~

完成后必须检查 schema、layer history、warnings、data readiness、signature 来源和 delta 证据，
不能只检查命令返回 0 或 Markdown 是否存在。

## 8. 修改与验证

修改 dataset、catalog、layer、pipeline、semantics 或 CLI 后运行：

~~~bash
git diff --check

.venv/bin/python -m unittest discover \
  -s kernel_agent/tests/pmc_interpreter \
  -p 'test_*.py'
~~~

若同时修改 CUDA source adapter 或 selection plan，再运行 `pmc-source-gate` 与
`gpu-pmu-sweep` 测试。gate 逻辑缺陷回到 `pmc-source-gate`，采集缺失回到
`gpu-pmu-sweep`；不得在 Interpreter 中放宽模型来绕过。manifest/workload/validator 问题
分别回到 `kernel-adapt` 和 `corpus-audit`。
