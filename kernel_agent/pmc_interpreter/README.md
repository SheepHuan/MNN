# PMC Interpreter

`PMC Interpreter` 将 CUDA、Adreno、Mali 等 profiler 的原生 PMC 观测转换为可审计的
结构化分析报告和中文 Markdown。平台差异只允许存在于 `dataset/sources/` 与
`dataset/catalog/`；投影、质量过滤、去冗余、相关性、kernel signature 和实现差分分析
不按平台分支。

工作流由三个独立 Skill 分工：`gpu-pmu-sweep` 负责设备采集，`pmc-source-gate` 负责 CUDA
原始数据发布门禁，`pmc-interpreter` 负责标准化数据集、固定分析层和语义输出。CUDA
raw-source 模式只有在 `cuda_pmc_source_validation.json` 为 `valid=true` 后才能用于正式分析。

## 数据源规范

CUDA 数据的产生、刷新和一致性验收见：

[`docs/superpowers/specs/2026-07-31-pmc-interpreter-cuda-data-sources.md`](../../docs/superpowers/specs/2026-07-31-pmc-interpreter-cuda-data-sources.md)

模块的物理语义、固定输入输出与证据边界见：

[`docs/superpowers/specs/2026-07-31-pmc-interpreter-module-physical-semantics.md`](../../docs/superpowers/specs/2026-07-31-pmc-interpreter-module-physical-semantics.md)

CUDA 固定使用三份原始输入：

- `build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv`
- `build-x86-cuda/cuda_kernel_latency.json`
- `replay_benchmark/kernel_corpus/operator_cases.json`

三份文件必须来自同一份 case manifest。修改 `operator_cases.json` 后，必须重新生成
PMC rows 和 latency；不能把不同 manifest 时点的数据合并。

单次 `PmcInterpreter` 分析严格以一个 device 为边界：所有参与分析的
`KernelCondition.device_id` 必须相同。包含多个设备的 bundle 必须先按 device 拆分并分别
运行；跨设备只能在报告产生后按 `concept_id` 比较语义，不能合并 native 数值或效应系数。

## 组件边界

```text
原生 profiler 输出与 kernel manifest
              │
              ▼
dataset/sources + dataset/catalog
              │
              ▼
PmcDataset（mnn-pmc-dataset/v1）
              │
              ▼
ProjectionOutput
              │
              ▼
QualityOutput
              │
              ▼
RedundancyOutput
              │
              ▼
CorrelationOutput
              │
              ▼
KernelSignatureOutput
              │
              ▼
DeltaRelationOutput
              │
              ▼
semantics/compiler.py
              │
              ▼
PmcAnalysisReport（mnn-pmc-analysis-report/v1）
              │
              ▼
semantics/markdown.py
              │
              ▼
MarkdownDocument
```

各目录的职责固定如下：

| 位置 | 职责 |
| --- | --- |
| `dataset/model.py` | 定义设备、kernel condition、采集 run、PMC/latency 观测、实现配对和 `PmcDataset` |
| `dataset/reports.py` | 定义特征、配置、每一层的严格输入输出、各层报告和最终 `PmcAnalysisReport` |
| `dataset/document.py` | 定义 `MarkdownSection` 与 `MarkdownDocument` |
| `dataset/catalog/` | 定义 kernel taxonomy 和 native metric 到平台无关概念的语义映射 |
| `dataset/sources/` | 将平台相关原始文件转换为统一实验事实 |
| `layers/` | 严格按上一层输出类型执行统计分析并产生下一层报告 |
| `semantics/compiler.py` | 只汇总已经完成的层报告，不执行新统计 |
| `semantics/markdown.py` | 只把结构化报告渲染为固定章节的中文 Markdown |
| `pipeline.py` | 固化层顺序并串接报告编译与 Markdown 渲染 |
| `cli.py` / `__main__.py` | 提供命令行输入输出 |

不保留旧目录、旧模块路径或旧调用别名。

## 严格数据模型

所有持久化和层间模型都继承 `PmcModel`，统一采用 Pydantic 严格约束：

- `frozen=True`：只提供浅冻结，禁止模型字段重新赋值；内含映射和列表仍可能被 Python
  调用方修改，必须按只读约定消费，不能把它当作深不可变或完整审计保证；
- `extra="forbid"`：未知字段直接报错；
- `allow_inf_nan=False`：不允许非有限浮点数进入结构化结果。

`PmcDataset` 保存只读实验事实：

- `KernelCondition`：device、kernel 语义、shape、工作量、实现、launch，以及 taxonomy 给出的 `expected_mechanisms`；
- `MeasurementRun`：独立 PMC 或 latency 采集，包含 repeat、session 和环境信息；
- `MetricDescriptor`：native metric 身份、硬件机制、数量类型和跨平台概念映射；
- `MetricObservation` / `LatencyObservation`：单条测量结果；
- `ComparisonPair`：同设备、同计算语义的 baseline/candidate 实现配对。

以下概念必须分开保存：

- workload 减 empty-control 的 PMC delta；
- candidate 减 baseline 的 implementation delta；
- 独立 `repeat_id`；
- 单次测量内部的 workload runs；
- profiler replay pass 数。

最小输入 bundle schema 为 `mnn-pmc-dataset/v1`：

```json
{
  "schema_version": "mnn-pmc-dataset/v1",
  "manifest": {
    "dataset_id": "experiment-1",
    "platform": "cuda",
    "backend": "cuda",
    "collector": "cupti",
    "metadata": {}
  },
  "devices": {},
  "conditions": {},
  "runs": {},
  "metric_catalog": {},
  "metric_observations": [],
  "latency_observations": [],
  "comparisons": [],
  "issues": []
}
```

输入语义不是分析时额外传入的松散映射。`KernelTaxonomy` 和 `MetricRegistry` 在 dataset
构造阶段将语义写入 `KernelCondition` 与 `MetricDescriptor`；其中
`KernelCondition.expected_mechanisms` 是 Signature 层的显式输入，Signature 层不能再按
平台或类别硬编码预期机制。后续 layer 只读取这些结构化字段。

## 固定串行层契约

流水线不能跳层或重排：

```text
ProjectionInput
  → ProjectionOutput
  → QualityOutput
  → RedundancyOutput
  → CorrelationOutput
  → KernelSignatureOutput
  → DeltaRelationOutput
```

每个输出模型继承上一层输出模型，因此后续层能读取此前全部报告，但不能使用无类型的
`annotations` 或 `artifacts` 传递隐式状态。每层 `run()` 会校验输入的具体类型。

1. `ProjectionLayer`
   - 将合法观测投影为 condition × feature 的固定坐标系；
   - 选择数值变换、汇总 latency、计算数据与三个核心问题的 readiness；
   - 输出 `ProjectionOutput` 和 `ProjectionReport`。
2. `QualityLayer`
   - 计算 coverage、常量/全零、独立 repeat MAD、状态分布与 canonical rollup；
   - 不使用 latency 做质量筛选；
   - 输出 `QualityOutput` 和 `QualityReport`。
3. `RedundancyLayer`
   - 记录相等观测向量；
   - 只在相同平台无关 concept 内强制去重；
   - 输出 `RedundancyOutput` 和 `RedundancyReport`。
4. `CorrelationLayer`
   - 对 latency 和 PMC feature 分别进行工作量/launch/类别/环境控制；
   - 将 condition → residual feature 的固定坐标写入 `CorrelationReport`，供后续类别分析消费；
   - 输出 raw 与 residual Spearman、FDR、distance correlation 和用途分类；
   - 输出 `CorrelationOutput` 和 `CorrelationReport`。
5. `KernelSignatureLayer`
   - 使用 Correlation 层输出的 residual feature 与 residual latency，按 semantic family、
     execution role、op type 和 performance regime 选择 PMC set；
   - 分开解释型指标与 `diagnostic_only` 指标；
   - 输出 `KernelSignatureOutput` 和 `KernelSignatureReport`。
6. `DeltaRelationLayer`
   - 只使用合法 implementation pair 计算 `ΔPMC` 与 `Δlatency`；
   - 先按 `semantic_equivalence_key` 将重复 pair 行折叠为独立语义组，最低样本门槛和统计
     样本数按独立语义组计算，而不是按 pair 行数计算；
   - 只有 baseline/candidate 共享至少两个非空 `paired_run_group_id`，且每组都同时连接有效
     PMC 与 latency 重复，才可能升级为因果表述；独立 `repeat_id` 只在这些共享组中统计，
     未配对的额外重复不能补足门槛；
   - 每条受控 pair 都必须填写同一个非空 `controlled_mechanism`，并与 feature 机制或 concept
     匹配；
   - 输出相关方向、FDR、斜率、方向一致率、共线组和证据等级；
   - 输出 `DeltaRelationOutput` 和 `DeltaRelationReport`。

## 最终结构化报告

最终 JSON schema 固定为 `mnn-pmc-analysis-report/v1`。`PmcAnalysisReport` 包含：

- `data_readiness`：三个核心问题目前能回答到什么程度；
- `layer_history` 与 `layer_reports`：完整的逐层输入输出证据；
- `metric_knowledge`：每个 feature 的语义、质量、冗余、相关和 delta 注释；
- `canonical_metric_sets`：描述性 canonical set 与 latency association set；
- `kernel_signatures`：kernel class 到 PMC set，以及反向索引；
- `delta_rulebook`：只由合法实现配对产生的条件关系规则；
- `evidence_index`、`warnings` 和 `provenance`：证据等级、限制与分析配置。

结构化报告分别回答：

| 核心问题 | 主要报告 |
| --- | --- |
| 哪些 PMC 与 latency 相关，属于什么类型 | `CorrelationReport` + 质量和冗余报告 |
| 每类 kernel 最相关的 PMC set 是什么 | `KernelSignatureReport` |
| 哪些 `ΔPMC` 显著伴随 `Δlatency` | `DeltaRelationReport` + `delta_rulebook` |

第三问不能从跨 kernel 相关性推出。没有足够、合法的 implementation pair 时，空的
`delta_rulebook` 是正确输出。

`delta_rulebook` 中的规则只能用于其 `scope_selector` 指定的设备和 kernel 类别，并且只能
在贡献配对实际覆盖的 `ΔPMC`/`Δlatency` 观测区间与受控干预语义内解释。不得外推到其他
设备、类别、未观测数值区间、不同实现变换或不同干预机制。

每条 `DeltaRule` 是自包含的：`scope_selector.conditions` 保存 baseline/candidate 的结构化
condition 范围，`intervention_ids` 和 `controlled_mechanisms` 保存干预边界，
`observed_support` 保存原始/变换后 PMC 范围、`ΔPMC`、latency ratio 与 `Δlog latency` 的
实际最小值和最大值，`statistics.contributing_pair_ids` 提供实验回溯索引。

## Markdown 语义文档

`semantics/` 不负责数据接入、metric 映射、特征选择或统计计算：

- `compile_analysis_report()` 只接受完整 `DeltaRelationOutput`，编译为
  `PmcAnalysisReport`；
- `render_markdown_document()` 只接受 `PmcAnalysisReport`，渲染为
  `MarkdownDocument`。

Markdown 固定包含以下章节：

```text
# PMC 语义分析报告

## 数据集与证据完整性
## 三个核心问题结论
## PMC 质量和过滤结果
## PMC 相关性分类
## 重复指标与 Canonical Set
## Kernel Type × PMC Signature
## ΔPMC 与 ΔLatency 规则
## 下一步完成路径
## 证据限制和不可回答问题
```

Markdown 会展示 association PMC 的 mechanism、phenomenon role、单位、normalizer、语义映射
置信度和解释边界，并严格区分 semantic-family 直接结果、op-type 直接实证和 family 先验继承。
JSON 是完整、可机读的分析结果；Markdown 是面向人的语义化视图，不能替代 JSON。

## Python 调用

```python
from kernel_agent.pmc_interpreter import AnalysisConfig, PmcInterpreter
from kernel_agent.pmc_interpreter.dataset import load_normalized_bundle

dataset = load_normalized_bundle("pmc-dataset-v1.json")
interpreter = PmcInterpreter(config=AnalysisConfig(global_top_k=15))

report = interpreter.analyze(dataset)
payload = report.model_dump(mode="json")

report, document = interpreter.analyze_with_document(dataset)
markdown = document.markdown
```

## CLI

分析标准化 bundle，并同时输出结构化 JSON 与中文 Markdown：

```bash
python3 -m kernel_agent.pmc_interpreter \
  --input-bundle <pmc-dataset-v1.json> \
  --output-json pmc_analysis_report.json \
  --output-md pmc_analysis_report.md
```

从 CUDA 三数据源构造统一数据集并分析：

```bash
python3 -m kernel_agent.pmc_interpreter \
  --pmc-csv build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv \
  --latency-json build-x86-cuda/cuda_kernel_latency.json \
  --operator-cases replay_benchmark/kernel_corpus/operator_cases.json \
  --case-selection-plan build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv.selection.json \
  --platform cuda \
  --backend cuda \
  --namespace cupti \
  --normalized-output-json build-x86-cuda/cuda_pmc_dataset_v1.json \
  --output-json build-x86-cuda/cuda_pmc_analysis_report_v1.json \
  --output-md build-x86-cuda/cuda_pmc_analysis_report_v1.md
```

`--output-json` 必填，保存 `mnn-pmc-analysis-report/v1`；`--output-md` 可选，保存固定章节的
中文报告。`--normalized-output-json` 可选，保存可复用的 `mnn-pmc-dataset/v1`。

## 验证

```bash
.venv/bin/python -m unittest discover -s kernel_agent/tests/pmc_interpreter -p 'test_*.py'
```

若同时修改 CUDA 原始数据交接或共享 selection 契约，还要运行：

```bash
.venv/bin/python -m unittest discover -s skills/pmc-source-gate/tests -p 'test_*.py'
```
