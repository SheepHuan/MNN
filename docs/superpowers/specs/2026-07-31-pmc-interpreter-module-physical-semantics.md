# PMC Interpreter 模块物理语义规范

## 一、目的

本文规定 `kernel_agent/pmc_interpreter/` 中每个组件在真实 GPU 性能实验中的物理意义、
固定输入输出、串行依赖和证据边界。

本文讨论“代码对象如何对应实验对象”，不替代 CUDA、Adreno 或 Mali 的原生 PMC 手册。
原生 metric 的精确公式、单位、依赖关系和采集限制必须由设备专用目录提供。

CUDA 三数据源的产生、刷新和一致性要求见：

`docs/superpowers/specs/2026-07-31-pmc-interpreter-cuda-data-sources.md`

## 二、总体物理模型

`PMC Interpreter` 不是“输入 PMC 矩阵，输出 latency Top-K”的单模型，而是一条严格的
证据升级链：

```text
真实设备与 kernel condition
        │
        ├─ PMC measurement run ─────────> MetricObservation
        ├─ latency measurement run ─────> LatencyObservation
        └─ baseline/candidate 设计 ─────> ComparisonPair
                                            │
                                            ▼
dataset/sources：平台原始记录 → 实验事实
dataset/catalog：写入 kernel 与 metric 语义
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
                              PmcAnalysisReport
                         （mnn-pmc-analysis-report/v1）
                                            │
                                            ▼
                              semantics/markdown.py
                                            │
                                            ▼
                                 MarkdownDocument
```

六个分析层回答不同问题：

| 层 | 物理问题 | 是否使用 latency | 能否解释优化方向 |
| --- | --- | --- | --- |
| `ProjectionLayer` | 原始观测如何形成可比较的 condition × feature 坐标系 | 只汇总目标 | 否 |
| `QualityLayer` | 仪表是否激活、覆盖充分、可重复、属于 kernel 核心路径 | 否 | 否 |
| `RedundancyLayer` | 是否重复观察了同一概念 | 否 | 否 |
| `CorrelationLayer` | 控制现有工作量等因素后，PMC 是否仍伴随额外 latency | 是 | 只能给观察性方向 |
| `KernelSignatureLayer` | 某类 kernel 应重点观察哪些 PMC | 是 | 不能直接给优化规则 |
| `DeltaRelationLayer` | 同语义实现变化时，`ΔPMC` 是否稳定伴随 `Δlatency` | 是 | 可给配对关系；严格门槛后才允许局部因果表述 |

## 三、统一 Pydantic 契约

### 3.1 `PmcModel`

`dataset/model.py` 中的 `PmcModel` 是全部实验事实、分析报告和文档对象的共同基类。
统一配置为：

```text
frozen=True
extra="forbid"
allow_inf_nan=False
```

物理意义是：

- layer 不能原地改写已经完成的实验事实或前层报告；
- 未声明字段不能通过松散字典悄悄进入分析；
- 非有限值必须在数据源边界被识别，不能污染统计结果。

`frozen=True` 只禁止模型字段重新赋值，不会深度冻结内含的 Python 映射；所有 layer 必须把
这些容器视为只读，并通过新的输出模型传递变化。它只是浅冻结约束，不是深不可变存储、
防篡改日志或完整审计保证；结构严格也不保证采集设计正确、单位正确或结论可识别。

### 3.2 固定层间类型

流水线输入输出严格为：

```text
ProjectionInput
  → ProjectionOutput
  → QualityOutput
  → RedundancyOutput
  → CorrelationOutput
  → KernelSignatureOutput
  → DeltaRelationOutput
```

每个后续输出模型继承前一个输出模型并增加本层报告。例如
`CorrelationOutput` 同时包含 projection、quality、redundancy 和 correlation 报告。

每层 `run()` 只接受直接上一层的具体类型；不能跳层、重排，也不能以无类型的
`annotations` 或 `artifacts` 传递隐式状态。这条约束表达的是认识顺序：

1. 先固定观测与变换；
2. 再判断可用性；
3. 再保守去重；
4. 再查看 latency；
5. 再构造类别 signature；
6. 最后分析实现配对。

## 四、`dataset/model.py`：只读实验事实

`dataset/model.py` 规定“一批 GPU 性能实验由哪些物理对象组成”，不执行文件解析、
kernel 分类、metric 推断或统计分析。

### 4.1 数据集、设备与实验条件

| 对象 | 物理意义 | 不能误解为 |
| --- | --- | --- |
| `DatasetManifest` | 一批采集结果的数据集身份、平台、后端与 collector | 同一 CUPTI session 或同环境的证明 |
| `DeviceSpec` | 被测物理 GPU 或必须独立分析的逻辑设备 | 只写 `platform=cuda` 就足以标识设备 |
| `KernelCondition` | 固定设备、语义、shape、参数、实现和 launch 的被测条件 | 一次采集 run 或一个 PMC 值 |

`KernelCondition` 的实验单位是：

```text
device
+ semantic family / op type / execution role
+ dtype / shape / params / workload
+ implementation
+ launch configuration
= condition
```

其中：

- `semantic_family` 是算法类别；
- `execution_role` 是该 kernel 在实现流水中的角色；
- `expected_mechanisms` 是 taxonomy 在 dataset 构造阶段写入的类别机制先验，Signature
  layer 只能读取这个显式字段，不能在 layer 内硬编码平台或类别语义表；自定义 dataset
  也必须为每个 `KernelCondition` 明确提供该字段，未知时使用空 tuple；
- `performance_regime` 是 compute-bound、memory-bound 等瓶颈标签，未知时必须保留
  `unknown`，不能由名称猜成事实；
- `semantic_equivalence_key` 用于筛选实现配对候选，不代替真实正确性验证。

### 4.2 测量过程与观测

| 对象 | 物理意义 | 关键边界 |
| --- | --- | --- |
| `MeasurementRun` | 一次独立安排的 PMC 或 latency 采集过程 | 不等于一次 kernel launch |
| `MetricObservation` | profiler 对一个硬件事件给出的读数 | 不等于瓶颈原因或优化建议 |
| `LatencyObservation` | 一次 latency 读数 | 不保证与 PMC 同频率、同温度或同 cache 状态 |

必须区分：

| 字段 | 实验意义 |
| --- | --- |
| `repeat_id` | 独立重复实验身份 |
| `workload_runs` | 一次 measurement 内部执行的 kernel 次数 |
| `collection_id` | profiler 或采集 session 身份 |
| `paired_run_group_id` | PMC、latency 或其他测量的配套组 |
| `profiler_pass_count` | profiler 为收集 metric 执行的 replay pass 数 |

`workload_runs`、重复账本行和 replay pass 都不能替代独立 `repeat_id`。

`MetricObservation.value_semantics` 至少区分：

- `absolute_workload`：workload 本身的计数；
- `workload_minus_control`：workload 与 empty-control 的背景校正差值。

`workload_minus_control` 可以为负，但它不是 candidate-baseline 的实现差值。

### 4.3 Metric 说明书

`MetricDescriptor` 是原生 PMC 的结构化语义说明书：

| 字段 | 物理含义 |
| --- | --- |
| `native_name` / `namespace` | 厂商原生 metric 身份 |
| `native_domain` / `hardware_scope` | 事件发生的硬件域与观察范围 |
| `basename` / `aggregation` / `normalizer` | 基础事件、rollup 与归一化方式 |
| `mechanism` | compute、cache、DRAM、scheduler、同步等机制 |
| `phenomenon_role` | work、utilization、symptom、capacity、timing 等角色 |
| `quantity_kind` / `unit` | counter、ratio、throughput 及其单位 |
| `duration_coupled` | 公式是否机械使用 elapsed time/cycle |
| `target_equivalent` | 是否近似就是 latency 或 elapsed cycles 的表达 |
| `dependency_metric_ids` | 派生 metric 的底层依赖 |
| `concept_id` | 跨平台对齐的概念槽 |
| `mapping_level` / `mapping_confidence` | 语义映射粒度和可信度 |

`concept_id` 相同只允许概念层对齐，不表示不同设备的 native 数值、峰值、单位或效应系数
可以直接合并。

### 4.4 实现对照

`ComparisonPair` 表达：

```text
同设备、同计算语义下的 candidate implementation - baseline implementation
```

它记录 intervention、受控机制、匹配字段、环境匹配和运行顺序随机化。有 pair 不等于有
因果实验；`paired_observational` 仍只支持配对条件关联。

### 4.5 `PmcDataset`

`PmcDataset` 是 `mnn-pmc-dataset/v1` 的唯一 Python 模型，聚合：

- manifest、devices 和 conditions；
- measurement runs；
- metric catalog；
- PMC 与 latency observations；
- implementation comparisons；
- 数据源发现的问题。

Pydantic validator 验证 schema、ID 唯一性和引用完整性，但不验证单位、独立重复、
同 session 采集、实现正确性或因果识别能力。

### 4.6 单设备分析边界

一次 `PmcInterpreter` 调用中，所有参与分析的 `KernelCondition.device_id` 必须相同。
Projection 层会拒绝含多个已使用 device 的 dataset；调用方必须先按 device 拆分。

这是统计和物理边界，而不仅是实现限制：不同 GPU 型号、架构、driver、频率域和 native
counter 定义不能共享同一组相关系数、mRMR 排名或 delta slope。跨设备只能对齐
`concept_id` 和证据类别，不能在一次分析中合并原生数值。

## 五、`dataset/reports.py`：特征、逐层报告与最终结果

`dataset/reports.py` 是全部派生结构的唯一模型定义位置，不读取原始文件，也不执行统计。

### 5.1 特征和分析上下文

| 对象 | 统计意义 |
| --- | --- |
| `AnalysisConfig` | coverage、样本数、FDR、Top-K 等工程阈值；不是 GPU 物理常数 |
| `FeatureSpec` | raw PMC 如何形成分析 feature，包括来源、表达式、变换和可比范围 |
| `FeatureStore` | feature 在各 condition 下的全部观测值 |
| `DataReadiness` | 三个核心问题的数据资产与证据准备度 |
| `AnalysisContext` | dataset、配置、特征和 condition 范围的只读上下文 |
| `LayerRecord` | 每层输入/输出 feature 数和说明的审计记录 |

`FeatureSpec` 与 `MetricDescriptor` 必须分开：前者是分析变量，后者是原生仪表说明书。
未来一个 feature 可以依赖多个原生 metric。

Delta readiness 同时报告原始合法 pair 行数和独立 pair group 数。最低样本门槛只使用不同
`semantic_equivalence_key` 的独立组数；同一语义组中的多个 pair 行不能虚增可识别样本。

### 5.2 各层报告

| 输出类型 | 新增报告 | 回答的问题 |
| --- | --- | --- |
| `ProjectionOutput` | `ProjectionReport` | 有多少有效观测被投影，使用了哪些变换，数据能回答什么 |
| `QualityOutput` | `QualityReport` | 哪些 PMC 因覆盖、激活、重复性或语义范围被过滤 |
| `RedundancyOutput` | `RedundancyReport` | 哪些 PMC 是同 concept 重复，哪些跨 concept 相等只做注释 |
| `CorrelationOutput` | `CorrelationReport` | 哪些 PMC 在现有控制下仍与 latency 有条件关联 |
| `KernelSignatureOutput` | `KernelSignatureReport` | 每类 kernel 的解释型与诊断型 PMC set |
| `DeltaRelationOutput` | `DeltaRelationReport` | 合法配对中 `ΔPMC` 与 `Δlatency` 的关系 |

每个报告都保留固定字段，不允许 layer 临时发明未声明的结果键。

### 5.3 最终结构化报告

`PmcAnalysisReport` 的 schema 固定为：

```text
mnn-pmc-analysis-report/v1
```

它包含：

- `scope`：数据集、平台、后端、设备与跨平台策略；
- `data_readiness`：三个核心问题的可回答程度；
- `layer_history` 和 `layer_reports`：完整逐层报告；
- `metric_knowledge`：每个 feature 的质量、冗余、相关、delta 和类别索引；
- `canonical_metric_sets`：描述性 canonical set 与 latency association set；
- `kernel_signatures`：分类到 PMC set 的正向与反向索引；
- `delta_rulebook`：通过门槛的配对关系规则；
- `evidence_index`、`warnings` 与 `provenance`。

最终报告只是对既有 layer report 的结构化汇总，不提高任何证据等级。

## 六、`dataset/document.py`：Markdown 输出模型

`dataset/document.py` 定义：

- `MarkdownSection`：固定标题与正文；
- `MarkdownDocument`：schema、标题、章节、警告和完整 Markdown 字符串。

Markdown 文档 schema 为 `mnn-pmc-markdown-document/v1`。它是
`PmcAnalysisReport` 的面向人视图，不是新的统计结果，也不能作为标准化 dataset 使用。

## 七、`dataset/catalog/`：输入语义目录

语义映射属于 dataset 构造阶段，因此目录固定放在 `dataset/catalog/`。

### 7.1 `default_kernel_taxonomy.json`

该文件是人工维护的 kernel 算法知识：

- `semantic_families`：`op_type` 到语义大类；
- `exact_variant_roles`：已知 variant 的精确执行角色；
- `ordered_role_rules`：按 variant 名称 token 推断执行角色。

它不读取 kernel 源码或 PMC，也不判断 compute-bound、memory-bound、同步瓶颈等
微架构状态。规则顺序会影响分类结果。

### 7.2 `registry.py::KernelTaxonomy`

`KernelTaxonomy` 把 `op_type + variant` 转为：

```text
semantic_family + execution_role + expected_mechanisms + 映射来源 + 映射可信度
```

这是业务语义和执行阶段分类，不是瓶颈分类。默认 `core_compute` 只是低置信度回退。
`expected_mechanisms` 会写入每个 `KernelCondition`，是后续 Signature 层唯一允许使用的类别
机制先验。

### 7.3 `registry.py::MetricRegistry`

`MetricRegistry` 把 native metric 转为完整 `MetricDescriptor`：

```text
CUPTI / KGSL / Mali native metric
              │
              ▼
hardware domain + mechanism + quantity + concept_id
```

通用 fallback 依靠名称 token 推断，通常只达到 `mechanism_proxy`，不等于硬件手册中的精确
数量定义。设备专用 override 应补充 unit、公式依赖、bounded range、collection cost 和更
可信的 concept 映射。

以下标记必须在 dataset 构造时确定：

- `duration_coupled`：`.per_second`、`.per_cycle_elapsed`、`%peak_elapsed` 等目标耦合；
- `target_equivalent`：`cycles_elapsed`、`time_duration` 等目标等价量；
- `phenomenon_role`：work、utilization、symptom 等现象角色；
- `dependency_metric_ids`：已知公式依赖；空值表示尚未录入，不证明没有依赖。

分析 layer 不再接收额外“语义映射”参数，只读取 `PmcDataset.metric_catalog`。

## 八、`dataset/` 的其他模块

### 8.1 `dataset/parsing.py`

该模块是字段解析边界：

- `parse_number()` 只保证数值有限；
- `parse_optional_int()` / `parse_optional_bool()` 解析可选元数据；
- `semantic_equivalence_key()` 构造实现配对使用的语义键。

它没有性能推断能力。manifest 字段不完整时，语义等价键也会不可靠。

### 8.2 `dataset/sources/mnn_replay.py`

该模块把 CUDA 三数据源连接为一个 `PmcDataset`：

| 输入 | 标准化对象 | 物理角色 |
| --- | --- | --- |
| `operator_cases.json` | `KernelCondition` | 被测条件与输入语义 |
| PMC rows CSV | `MetricDescriptor`、PMC run、`MetricObservation` | 硬件事件读数 |
| latency JSON | latency run、`LatencyObservation` | 独立目标测量 |
| 可选 pairs CSV | `ComparisonPair` | 实现对照设计 |

legacy CUDA rows 没有真实 `collection_session_id`。合成的 run 容器不能证明不同 metric
同进程、同 CUPTI session 或同时采集，因此不能盲目构造跨 metric 比值。

当前 CUDA rows 不提供独立 `repeat_id`；`num_passes` 不能代替 repeat。PMC 与 latency 也是
独立采集，只按 condition 对齐。

没有显式 pairs CSV 时，loader 可按 `semantic_equivalence_key` 发现候选 pair。自动发现的
pair 必须视为待审计的观察性配对，不是受控优化实验。

### 8.3 `dataset/sources/control_delta.py`

该模块把 Adreno、Mali、OpenCL 或 Vulkan 风格的 control-delta CSV 追加到已有 dataset。
每行表示：

```text
workload PMU 变化 - empty-control PMU 变化
```

负 delta 是合法观测，使用保留符号的数值变换。它不是实现优化的 `ΔPMC`。

### 8.4 `dataset/bundle.py`

该模块只负责 `PmcDataset` 与 `mnn-pmc-dataset/v1` JSON 的读写和结构校验，不创造物理语义
或统计证据。

## 九、`layers/base.py`：Projection 与统一坐标系

`ProjectionLayer` 接受 `ProjectionInput`，输出 `ProjectionOutput`。它负责：

- 验证所有参与分析的 condition 属于同一个 device，遇到多设备输入直接失败；
- 只投影状态为 `VALID` 且数值有效的 PMC；
- 保留 condition 内全部观测，后续主要使用中位数；
- 为每个 native metric 构造独立 `FeatureSpec`；
- 非负计数通常用 `log1p`，有界比例用 `logit`，control delta 用固定尺度的
  `signed_asinh`；
- 汇总 condition 的中位 latency 并使用 `log latency`；
- 统计 workload、launch、环境、repeat 和合法 pair 的 readiness；
- 发现同一 feature 混合不同 `value_semantics` 时写入 issue。

本层没有瓶颈或相关性结论。当前只构造 raw 或 control-adjusted feature，尚未自动构造
`PMC/output_elements`、`PMC/FLOPs` 或 `PMC/algorithmic_bytes` 等单位工作量特征。

## 十、`layers/quality.py`：质量与 canonical rollup

`QualityLayer` 只接受 `ProjectionOutput`，输出 `QualityOutput`。它回答：

> 这个 PMC 在当前 corpus 中是否值得作为传感器保留？

它计算：

- condition coverage；
- nonzero rate 和 unique values；
- 跨 condition 均值、标准差和 CV；
- 至少两个非空独立 `repeat_id` 时的 relative MAD；
- 原始 observation status 分布；
- 同 basename、现象角色和 normalizer 下的 canonical rollup。

它排除：

- 低覆盖、全零、近常量；
- target-equivalent timing；
- 不属于 kernel core 的 metric；
- 混合不同 `value_semantics` 的 feature；
- 有足够独立重复且 MAD 过高的 feature；
- 同一语义表达组中的非 canonical rollup。

`cross_condition_cv` 大表示不同 condition 差异大，不等于采集噪声大；
`repeat_relative_mad=None` 表示不可估计，不表示稳定。

## 十一、`layers/redundancy.py`：保守去重

`RedundancyLayer` 只接受 `QualityOutput`，输出 `RedundancyOutput`。它按各 condition 中位数
构造完整观测向量。

强制去重条件固定为：

```text
观测向量完全相同 + concept_id 相同
```

跨 concept 的相等向量只记录、不删除。这样能保留同一性能链上的原因、结果和调度症状。

本层证明的是“当前数据快照上的观测冗余”，不是公式等价。当前仍未覆盖高相关但不完全
相同、固定比例、完整 dependency graph 或机制级近似冗余聚类。

## 十二、`layers/correlation.py`：条件相关与用途分类

`CorrelationLayer` 只接受 `RedundancyOutput`，输出 `CorrelationOutput`。它回答第一个
核心问题：

> 在当前可用的工作量、shape、参数、launch、kernel 类别和环境控制下，该 PMC 是否仍伴随
> kernel 相对基线额外变慢或变快？

### 12.1 Latency baseline

目标为：

```text
log(median latency)
```

控制变量包括覆盖充分的：

- `workload`、`shapes`、`params` 和 `launch` 数值字段；
- `semantic_family`、`execution_role`、`op_type` 和 `dtype`；
- GPU clock 和 temperature。

代码使用按 `condition_id` 分折的 cross-fitted ridge。若 out-of-fold `R² <= 0`，退化为
全局均值并把证据降级为 `descriptive_cross_kernel`。

残差定义为：

```text
residual log latency
= actual log latency - baseline predicted log latency
```

它表示“相对当前基线额外慢”，不是 Roofline gap，也不证明所有混杂已消除。

### 12.2 PMC feature residualization

当 baseline 有效时，同一设计矩阵也用于 cross-fit 每个变换后 PMC feature：

```text
residual feature
= transformed feature - feature baseline prediction
```

随后计算 residual feature 与 residual latency 的 Spearman 和 distance correlation。
这比只 residualize latency 更接近条件关联，但仍不是条件互信息或因果效应。

`CorrelationReport.residual_feature_values` 固定保存每个 feature 的 condition → residual
坐标。后续 Signature 层必须消费这份前层输出计算类别内 relevance 与特征间冗余，不能重新
退回原始 PMC 值；当 baseline 证据降级为 `descriptive_cross_kernel` 时，该坐标就是变换后的
原始 feature，并随报告一起明确降级。

### 12.3 关联证据

每个 feature 输出：

- raw latency Spearman；
- residual latency Spearman；
- feature 是否 residualize 及其 baseline R²；
- distance correlation 非线性筛查；
- event presence correlation；
- Fisher-z 近似 p-value 与 Benjamini-Hochberg FDR；
- relation class、usage class 和解释文本。

relation class 区分：

- target-equivalent / target-coupled；
- workload indicator；
- sparse activation；
- residual monotonic；
- residual nonlinear candidate；
- moderate context candidate；
- weak、descriptive 或 insufficient evidence。

正相关不能解释为“PMC 越大越坏”。throughput 高可能表示效率提高、资源饱和，也可能由
elapsed time 进入分母而机械升高。

### 12.4 当前统计边界

- condition 哈希分折不是 leave-one-kernel、leave-one-shape 或
  leave-one-implementation-out；
- Fisher-z 用于有 ties 的整数 counter，只是近似显著性；
- distance correlation 当前没有 permutation p-value 和 FDR；
- 缺少 FLOPs、algorithmic bytes、output elements、clock 或 temperature 会降低证据等级；
- residualization 只控制已进入结构化 dataset 且覆盖充分的字段。

因此本层输出是条件关联和诊断分类，不是优化因果规则。

## 十三、`layers/signatures.py`：Kernel Type × PMC Set

`KernelSignatureLayer` 只接受 `CorrelationOutput`，输出 `KernelSignatureOutput`。它回答
第二个核心问题：

> 对某类 kernel，当前设备上哪些 native PMC 最适合作为联合诊断面板？

它按以下轴建立 signature：

- `semantic_family`；
- `execution_role`；
- `semantic_family × execution_role`；
- `op_type`；
- 已知 `performance_regime`。

选择分数使用 group 内 residual-feature 与 residual-latency 的 Spearman 及 activation 作为
relevance，减去已选 residual feature 之间的平均绝对 Spearman 冗余，并加入
`KernelCondition.expected_mechanisms` 的匹配 bonus
和机制多样性的工程 bonus。layer 不允许内置按平台或 semantic family 分支的机制表。
每个 concept 最多两个 feature，每个 mechanism 最多四个。

输出严格分开：

- `resolved_features`：可用于 latency 解释探索的 feature；
- `diagnostic_features`：target-coupled、稀疏触发或 workload indicator 等诊断 feature；
- `concept_slots`：平台无关传感器槽；
- `metric_to_kernel_classes`：PMC 到适用 kernel 类别的反向索引。

样本不足的 op type 可以继承 family feature 作为先验回退，但不能冒充 op-type 实证。

当前选择器是工程约束 mRMR，不是互信息意义的标准 mRMR，也尚未实现 stability selection、
分层 GAM 或 group-aware 验证。signature 表示“值得联合观察的传感器集合”，不表示这些
PMC 越高越好，也不表示它们就是瓶颈原因。

## 十四、`layers/delta_relation.py`：实现配对中的 ΔPMC–Δlatency

`DeltaRelationLayer` 只接受 `KernelSignatureOutput`，输出 `DeltaRelationOutput`。它回答
第三个核心问题：

> 对同设备、同计算语义的 baseline/candidate 实现，PMC 的变化是否稳定伴随 latency 变化？

### 14.1 合法配对

合法 pair 至少要求：

- baseline 和 candidate condition 存在；
- 同一 device；
- `semantic_equivalence_key` 相同；
- 双方都有有效 latency。

合法 pair 行仍不是独立统计样本。每个 pair 由 baseline condition 的
`semantic_equivalence_key` 归入一个独立语义组；同一组中的多个 baseline/candidate 行先以
中位数折叠成一个 `Δfeature` 和一个 `Δlatency` 样本。

对每个 feature 计算：

```text
Δfeature
= transform(candidate PMC) - transform(baseline PMC)

Δlatency
= log(candidate median latency / baseline median latency)
```

零 `ΔPMC` 样本保留，因为它能证明 latency 变化未被该 feature 的变化解释。

### 14.2 规则证据

每个 feature 输出：

- 原始 pair 行数、独立 `semantic_equivalence_key` 组数、贡献 pair ID 和零 delta 数；
- delta Spearman 与 FDR；
- 过原点 transformed slope 和普通 95% 区间；
- 正向、反向和最终方向一致率；
- effect score；
- delta 共线组；
- relation class 与 exclusion reason。

稳定关系的最低样本门槛按独立语义组数计算，而不是按 pair 行数计算；还要求至少三个不同
`ΔPMC`、FDR 达标、相关强度与方向一致率达标。配置字段虽然仍名为 `min_delta_pairs`，其物理
含义是“最少独立 `semantic_equivalence_key` 组数”。

以下 feature 不能进入优化 effect rule：

- `diagnostic_only`；
- duration-coupled；
- target-equivalent。

`increase_associated_with_higher_latency` 只表示 candidate 中 PMC 增大时通常更慢，不表示主动
调高该 PMC 必然导致变慢。`effect_score` 是排序分数，不是物理效应幅度。

### 14.3 共线与类别范围

全局和 semantic-family 范围分别建 rule。精确 rank 共线的多个 feature 只选一个代表，
避免全部升级为独立原因；但单变量模型仍不能分离同时变化的非共线机制。

共线 group ID 必须由“统计 scope + delta fingerprint”共同决定；同一 family 中两个不同
fingerprint 是两个独立共线簇，各自可以保留一个代表。报告 `status` 按全局与 family-scoped
effect rule 的并集判断，不能因为全局方向抵消就隐藏已经成立的类别规则。

### 14.4 因果表述门槛

只有同时满足以下条件，`causal_claim_allowed` 才可能为真：

- pair design 为 `controlled_intervention`；
- 环境匹配且运行顺序随机；
- 每个共享配套组都记录 clock/temperature，且 baseline/candidate 的组内实测差异在容差内；
- 每条 pair 都必须填写非空 `controlled_mechanism`，所有 pair 控制同一机制，且与 feature 的
  mechanism 或 concept 匹配；
- 每个 pair 有 intervention ID；
- feature 不属于目标耦合或 target-equivalent；
- feature 不在 delta 共线组；
- baseline 和 candidate 的 PMC 与 latency 都至少有两个不同、非空 `repeat_id`；这些 repeat
  必须来自双方共享的配套组，未配对的额外 repeat 不参与因果门槛。
- 对每个实现配对，baseline 与 candidate 至少共享两个非空 `paired_run_group_id`；每个共享
  group 都必须在双方各自同时连接该 feature 的有效 PMC 与有效 latency 观测。

即便通过这些门槛，规则仍只适用于报告中的设备、kernel 类别、数值范围和实验设计。
当前 slope 是单变量变换尺度近似，置信区间不是 paired bootstrap 或 cluster-robust 区间，
也没有 GAM/ALE 的阈值和饱和曲线。

### 14.5 Rulebook 适用范围与禁止外推

每条 Delta rule 必须按以下四个维度解释：

- 设备：只适用于 `scope_selector.device_ids` 指定的设备环境；
- 类别：只适用于 `scope_selector.semantic_families` 和贡献 pair 覆盖的 kernel 语义；
- 观测区间：只适用于贡献 pair 实际覆盖的 `ΔPMC`、`Δlatency` 和变换尺度范围；
- 干预：只适用于记录的 implementation 变化、`controlled_mechanism` 和 intervention 设计。

结构化 `DeltaRule` 直接携带这些边界，不要求消费方回读原始 dataset 才能知道规则范围：

- `scope_selector.conditions` 保存 baseline/candidate 角色、condition ID、op type、execution
  role、dtype、implementation、shape、参数、工作量和 launch；
- `scope_selector.intervention_ids` 与 `controlled_mechanisms` 保存实际贡献规则的干预语义；
- `observed_support` 保存 baseline/candidate 原始 feature、原始和变换后 `Δfeature`、latency
  ratio 与 `Δlog latency` 的最小/最大观测范围；
- `statistics.contributing_pair_ids` 保存产生该统计量的 pair 行，用于回溯实验设计。

不能把局部 slope 外推到其他 GPU、其他 semantic family、未观测的大幅 PMC 变化、不同
baseline、不同优化组合或不同干预机制。结构化结果中的 `contributing_pair_ids` 是审计这些
范围的依据；rulebook 不是跨平台、跨类别的通用硬件定律。

## 十五、`layers/statistics.py`：数学原语

该模块没有 GPU 硬件物理意义，只为 layer 提供可审计的数学工具。

| 工具 | 统计意义 | 不能声称 |
| --- | --- | --- |
| `median` | 鲁棒中心 | 描述抖动、多峰或漂移 |
| `rankdata` / `spearman` | 单调顺序共变 | 因果、阈值或绝对效应 |
| `distance_correlation` | 非线性依赖筛查 | 方向或当前实现中的显著性 |
| `approximate_correlation_pvalue` | Fisher-z 近似 | ties-aware 精确 Spearman 检验 |
| `benjamini_hochberg` | 一批检验的 FDR 控制 | 每条规则都正确或效应足够大 |
| 数值变换工具 | 将计数、比例和负 delta 放到合适尺度 | 自动区分工作量与效率 |
| ridge 工具 | 构造 latency/feature baseline | ridge 系数是硬件因果效应 |
| `cross_fitted_ridge` | condition 分折的 out-of-fold 预测 | group-aware 泛化验证 |
| `through_origin_effect` | 单变量 `Δlatency = slope × Δfeature` | 独立机制因果效应 |

`signed_asinh` 的 scale 在 Projection 阶段固定，保证层间可比；不同 dataset 或 device 的 slope
仍不能直接横向比较。

当前尚未实现 ICC、条件互信息、HSIC permutation test、CLR/ILR、GAM/ALE、stability
selection、paired bootstrap 或 group-aware CV。

## 十六、`pipeline.py`：不可重排的编排

`PmcInterpreter.run_layers()` 固定执行：

```text
Projection
→ Quality
→ Redundancy
→ Correlation
→ KernelSignature
→ DeltaRelation
```

它不允许注入自定义 layer 顺序。`analyze()` 在完整 `DeltaRelationOutput` 上编译
`PmcAnalysisReport`；`analyze_with_document()` 再返回结构化报告与 `MarkdownDocument`。

pipeline 不采集 PMC、不增加样本、不验证 CUDA 三数据源来自同一 manifest，也不会把
观察性证据自动升级为因果。

## 十七、`semantics/`：只编译和渲染

`semantics/` 固定只包含两个职责模块，不保存输入 taxonomy，不执行统计。

### 17.1 `semantics/compiler.py`

`compile_analysis_report()` 只接受完整 `DeltaRelationOutput`。它把已有层报告汇总为：

- 两套 canonical metric set；
- `metric_knowledge`；
- kernel signatures；
- delta rulebook；
- evidence index、warnings 和 provenance。

编译器不重新计算相关系数、不修改 feature、不补造 pair，也不提高证据等级。
编译器保留 rule 的 device/category scope 和贡献配对证据；消费方仍必须据此限制观测区间与
intervention 范围，不能只读取方向字段后外推。

### 17.2 `semantics/markdown.py`

`render_markdown_document()` 只接受 `PmcAnalysisReport`，渲染固定中文章节：

```markdown
# PMC 语义分析报告

## 数据集与证据完整性
## PMC 质量和过滤结果
## PMC 相关性分类
## 重复指标与 Canonical Set
## Kernel Type × PMC Signature
## ΔPMC 与 ΔLatency 规则
## 证据限制和不可回答问题
```

Markdown 只展示结构化报告的重要部分。完整 feature 明细、全部 layer report 和 provenance
必须从 JSON 获取。

## 十八、`cli.py` 与 `__main__.py`

CLI 负责：

1. 读取 `mnn-pmc-dataset/v1`，或从 CUDA 三数据源构造 dataset；
2. 可选追加 control-delta CSV 和 explicit pairs；
3. 可选保存 normalized dataset；
4. 运行固定串行 interpreter；
5. 将 `mnn-pmc-analysis-report/v1` 写入 `--output-json`；
6. 在提供 `--output-md` 时写入中文 Markdown。

示例：

```bash
python3 -m kernel_agent.pmc_interpreter \
  --input-bundle pmc_dataset.json \
  --output-json pmc_analysis_report.json \
  --output-md pmc_analysis_report.md
```

`--output-json` 必填；`--output-md` 可选。Markdown 不是 JSON 的替代品，也不能作为
下一次分析的输入。

`--pmc-workload-runs` 和 `--latency-workload-runs` 是 measurement 内部运行次数，不是独立
repeat。CLI 的默认设备信息可能为 `unknown`，也不会自动证明三数据源来自同一 manifest
和同一设备环境。

`__main__.py` 只把 `python3 -m kernel_agent.pmc_interpreter` 转发到 CLI。

## 十九、三个核心问题如何落到输出

| 核心问题 | 主要层 | 固定输出 | 正确证据表述 |
| --- | --- | --- | --- |
| 哪些 PMC 与 latency 相关，属于什么类型 | Projection + Quality + Redundancy + Correlation | `CorrelationReport`、`metric_knowledge`、两套 canonical set | 控制程度明确的观察性关联 |
| 每类 kernel 最相关的 PMC set 是什么 | KernelSignature | `KernelSignatureReport`、`metric_to_kernel_classes` | 类别诊断面板，不是瓶颈因果集合 |
| 哪些 `ΔPMC` 会显著伴随 `Δlatency` | ComparisonPair + DeltaRelation | `DeltaRelationReport`、`delta_rulebook` | 配对条件关联；严格受控后才允许局部因果表述 |

第一问必须区分：

- 描述性 canonical PMC：稳定、非重复，不使用 latency；
- latency association PMC：使用 latency 的监督式探索集合；
- diagnostic-only PMC：可判断状态，但存在目标耦合或稀疏触发；
- workload indicator：主要反映做了多少工作；
- latency-informative candidate：控制现有工作量后仍保留信息。

第三问绝不能由第一问的跨 kernel 相关性推出。没有足够 implementation pair 时，空的
`delta_rulebook` 是正确结果。

## 二十、跨平台边界

平台差异只能存在于：

- `dataset/sources/`：读取 CUDA、Adreno、Mali 等不同采集格式；
- `dataset/catalog/`：把 native metric 和 kernel 元数据写入共享结构。

所有 layer 只读取 `PmcDataset` 和 `MetricDescriptor`，不得出现 CUDA、Adreno、Mali
专用分支。

“平台无关”表示同一套 schema 和 layer 算法可用于不同平台，不表示一次分析可以跨设备。
单次 Interpreter 必须 device-scoped；多设备结果只能分别分析后，在概念层汇总证据。

跨平台只能通过 `concept_id`、`mapping_level` 和 `mapping_confidence` 对齐。native counter
数值、峰值、变换斜率和效应规则默认保持 device-scoped；不能因名称或机制相似直接合并。

## 二十一、当前能力边界

当前包最准确的定位是：

> 一个平台无关、严格串行、可审计的 PMC 语义与探索性证据解释器。

它已经把实验事实、分析报告和文档输出分离，并能有边界地回答三个核心问题，但还不能被
描述为完整的因果性能诊断器。特别是：

- Correlation 与 KernelSignature 只能产生条件关联；
- DeltaRelation 依赖真实 implementation pair；
- 独立重复、环境匹配和受控干预必须来自实验设计，代码不能凭空创造；
- Delta 样本数按独立语义组而非 pair 账本行计算；
- 因果升级要求 baseline/candidate 共享至少两个真正连接 PMC 与 latency 的采集组；
- 缺少 workload、shape、launch 或环境字段时，报告必须显式降级；
- “PMC 变大”只有在确定 metric 类型、适用 kernel 类别、有效范围、目标耦合和配对证据后
  才能解释。
