# PMC Interpreter CUDA 三数据源规范

## 一、目的

本文规定 `kernel_agent/pmc_interpreter` 分析 CUDA kernel 时使用的三份权威原始数据源，
以及它们从同一份 case manifest 产生、刷新、连接和验收的方式。

对应 Skill 的职责固定分流为：`kernel-adapt` 构建 kernel 与 manifest，`corpus-audit` 做独立
正确性放行，`gpu-pmu-sweep` 采集原始事实，`pmc-source-gate` 执行 strict source gate，
`pmc-interpreter` 只消费 `valid=true` 的交接数据并生成语义知识。不得用 Interpreter 的
loader 或报告输出替代数据源门禁。

固定路径如下：

| 数据源 | 固定路径 | 物理角色 |
| --- | --- | --- |
| Kernel case manifest | `replay_benchmark/kernel_corpus/operator_cases.json` | 定义可执行 kernel condition、shape、算法 workload 和语义来源 |
| CUDA PMC 明细 | `build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv` | 保存 selection plan 选中 condition 与设备有效 PMC 的完整长表账本 |
| CUDA latency | `build-x86-cuda/cuda_kernel_latency.json` | 在关闭 PMU 后独立保存全部 CUDA case 的延迟、launch/resource 和环境记录 |

正式 PMC rows 还必须配套保存：

```text
build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv.selection.json
```

它是 schema 为 `mnn-pmc-case-selection/v1` 的 `CaseSelectionPlan`。它不是第四份原始测量，
而是 rows 的采样设计与连接凭据。`cuda_pmc_valid.csv` 是设备 availability 中间产物，也必须
保留用于审计，但不属于 Interpreter 的三份最终原始输入。

数据流固定为：

```text
operator_cases.json
  ├─ 全部 CUDA case ──────────────────────> cuda_kernel_latency.json
  │                                         replay_kernel_latency 产生
  │
  └─ condition-balanced 选择 ─────────────> CaseSelectionPlan
                                             │
cuda_pmc_valid.csv ──────────────────────────┤
                                             ▼
                           cuda_kernel_pmc_kernelreplay_rows.csv
                           sweep_cuda_kernel_pmc.py 产生
```

三份原始输入与 selection plan 通过严格验收后，
`kernel_agent/pmc_interpreter/dataset/sources/mnn_replay.py` 才能构造
`mnn-pmc-dataset/v1`。固定串行 layer 随后生成 `mnn-pmc-analysis-report/v1`，并可选渲染
中文 Markdown。数据产生、dataset 构造、统计分析和文档渲染是四个独立阶段。

## 二、同一实验身份

一次正式 CUDA 分析至少由以下身份共同确定：

```text
operator_cases.json 内容
+ CaseSelectionPlan 与选择算法版本
+ target op types 与每类 condition 目标数
+ cuda_pmc_valid.csv 内容和 metric 顺序
+ collector binary 与 collector code
+ runs、metrics-per-session、timeout 等 sweep 配置
+ GPU、driver、CUDA、CUPTI 和设备指纹
```

三份原始数据不得来自不同 manifest 快照或不同设备环境。一次 Interpreter 分析还必须保持
device-scoped：所有 `KernelCondition.device_id` 必须指向同一块设备。多 GPU 或跨机器结果
必须先拆成多个 dataset，分别分析后再在 `concept_id` 层比较。

以下任一身份变化后都必须从空账本重采，禁止继续使用旧 rows 的 `--resume`：

- `operator_cases.json` 内容变化；
- `CaseSelectionPlan`、选择算法或 policy version 变化；
- `--target-op-type` 或 `--conditions-per-op-type` 变化；
- `cuda_pmc_valid.csv`、metric 集合或 metric 顺序变化；
- collector binary、collector code 变化；
- `--runs`、`--metrics-per-session`、timeout 等 sweep 配置变化；
- device、GPU、driver、CUDA、CUPTI 或传入的 device fingerprint 变化。

## 三、统一构建前提

CUDA 数据采集使用 `build-x86-cuda`。首次配置至少需要：

```bash
cmake -S . -B build-x86-cuda \
  -DMNN_BUILD_BENCHMARK=ON \
  -DMNN_CUDA=ON \
  -DMNN_REPLAY_ENABLE_PERFCOUNTER=ON \
  -DCMAKE_BUILD_TYPE=Release
```

构建生产三份数据所需的目标：

```bash
cmake --build build-x86-cuda \
  --target replay_benchmark.out list_cuda_metrics \
           replay_pmu_availability replay_kernel_latency \
  -j2
```

对应源码入口：

| 目标 | 源码入口 |
| --- | --- |
| `replay_benchmark.out` | `replay_benchmark/KernelCorpusBenchmark.cpp` |
| `replay_pmu_availability` | `replay_benchmark/tests/test_pmu_availability.cpp` |
| `replay_kernel_latency` | `replay_benchmark/tests/test_kernel_latency.cpp` |
| `list_cuda_metrics` | `replay_benchmark/list_cuda_metrics.cu` |

CUDA CUPTI Range Profiler 通常需要管理员权限。一次只能运行一个完整 availability 或 sweep
实例，避免多个进程竞争 profiler 资源。

## 四、数据源一：`operator_cases.json`

### 4.1 权威来源

`replay_benchmark/kernel_corpus/operator_cases.json` 是仓库内维护的可执行 case manifest，
不是运行 `replay_benchmark/tests` 后自动生成的文件。

它被以下生产者读取：

- `replay_kernel_latency`：枚举全部 `backend == "cuda"` 的 case；
- `sweep_cuda_kernel_pmc.py`：构造或验证 `CaseSelectionPlan`；
- `replay_benchmark.out --kernel-corpus-bench`：查找 kernel、构造参数、运行并校验结果；
- `dataset/sources/mnn_replay.py`：构造 `KernelCondition` 并写入语义和 provenance。

`operators.json` 与它不是同一个数据源：

- `operators.json` 是 kernel 文件与 entry 的索引，可以由 extractor 重新生成；
- `operator_cases.json` 是可执行 workload 定义，需要与 `OpAdapter`、validator 和 launch
  参数一起维护。

### 4.2 文件与字段契约

顶层必须满足：

```json
{
  "format": "mnn-kernel-operator-cases",
  "version": 1,
  "cases": []
}
```

CUDA case 的核心字段：

| 字段 | 要求 | 含义 |
| --- | --- | --- |
| `name` | 必填且 CUDA 子集唯一 | latency、rows 和 plan 的 join key |
| `backend` | 必填 | CUDA case 固定为 `cuda` |
| `op_type` | 必填 | 语义算子类型和 selection 分组键 |
| `framework`、`tag`、`variant` | 必填 | 实现来源和执行变体，不计入 workload 多样性 |
| `int_params`、`float_params` | 应提供 | adapter 参数；显式 workload 不足时可作为 proxy |
| `dtype` | 应提供 | condition 身份的一部分 |
| `shapes` | P0 目标类别应提供 | 逻辑输入输出 shape，不是 allocation 或 launch geometry |
| `workload` | P0 目标类别应提供 | `algorithmic_flops`、`algorithmic_bytes`、`output_elements` 等算法工作量 |
| `shape_provenance` | 与 `shapes` 配套 | 每个 shape 事实的来源、方法、公式、可信度和说明 |
| `workload_provenance` | 与 `workload` 配套 | 每个 workload 事实的来源、方法、公式、可信度和说明 |
| `validator` | 应提供 | 正确性验证器，也是实现语义等价审计的一部分 |
| `warmup_runs` | 可选 | 缺省值由 benchmark 决定；不是独立 repeat |
| `workload_runs` | 可选 | 单次 measurement 内运行次数；不计入 workload 多样性 |

`variant`、`tag` 和 `workload_runs` 不得参与 workload condition fingerprint。否则同一 shape 与
算法工作量仅因实现名或测量次数不同就会被错误视为多个工作负载。

### 4.3 P0 workload 类别

当前显式补充 shape、workload 和 provenance 的 P0 op type 固定为：

- `matmul`
- `conv_dw`
- `reduction`
- `softmax`
- `transpose`
- `maxpool`
- `avgpool`

权威列表位于：

```text
replay_benchmark/kernel_corpus/enrich_pmc_workload_metadata.py::SUPPORTED_OP_TYPES
```

对应测试位于：

```text
replay_benchmark/kernel_corpus/tests/test_pmc_workload_metadata.py
```

更新元数据时运行：

```bash
python3 replay_benchmark/kernel_corpus/enrich_pmc_workload_metadata.py \
  --input replay_benchmark/kernel_corpus/operator_cases.json \
  --in-place

python3 -m unittest discover \
  -s replay_benchmark/kernel_corpus/tests \
  -p 'test_pmc_workload_metadata.py'
```

脚本只能写入可由 adapter 契约证明的算法事实，不得把 allocation bytes、launch geometry、
设备实测值或无法证明的 FLOPs 填成 workload。无法适用的字段应在 provenance 中标为
`not_applicable`，而不是伪造 0。

脚本生成的算法估算使用 `source=replay_benchmark_op_adapter_contract`、
`method=semantic_formula`；可用数值的 status 为 `theoretical_estimate`，比较或搬运类不适用
FLOPs 明确标为 `not_applicable`。它只覆盖白名单中的精确 adapter variant。

刷新必须是字段级、可重复且事务式的：脚本可以清理并重建自身 provenance 管理的字段，用户
额外 key 必须保留；若同一个 shape/workload key 已有非脚本所有的用户值，只有与新公式结果
完全一致时才能保留，不一致必须抛错且不得部分写回。对同一输入连续运行两次，输出内容哈希
必须一致。

这七类只是当前 P0 明确 enrichment 的 cohort，不代表最终 selection plan 的 case 数。
不得预写最终 selected-case 数、plan ID 或 rows 数。

## 五、`CaseSelectionPlan`：PMC condition 采样设计

### 5.1 固定策略

正式分析默认且唯一的选择策略是：

```text
--case-selection condition-balanced
```

不得保留或描述旧 `op-type` 别名。`--case-selection all` 只用于调用方明确要求的全 case
扫描，不是默认分析策略。

`CaseSelectionPlan` 定义在：

```text
kernel_agent/pmc_interpreter/dataset/selection.py
```

schema 固定为：

```text
mnn-pmc-case-selection/v1
```

### 5.2 选择算法

对每个目标 `op_type`，算法固定执行：

1. 只保留目标 backend 的合法 case，并要求 case name 唯一；
2. condition 身份由 dtype、`shapes`、`workload` 和 adapter params 构造；
3. 优先在同时具有显式 `shapes + workload` 的 distinct condition 中做确定性覆盖选择；
4. 显式 condition 不足 K 时，才按 `params_proxy`、缺失元数据的顺序补位；
5. distinct condition 总数少于 K 时，全选该 `op_type` 的 case，并标记
   `insufficient_unique_conditions`；
6. 选择结果按 target op type 和 selection rank 固定顺序输出，不依赖 manifest 行顺序选择
   “第一条”。

默认 K 为 5，对应：

```text
--conditions-per-op-type 5
```

可以重复使用以下参数限制目标类别：

```text
--target-op-type <精确 op_type>
```

### 5.3 计划固定内容

计划至少持有：

| 字段 | 物理意义 |
| --- | --- |
| `source_manifest_sha256` | 产生计划的 manifest 内容身份 |
| `policy`、`policy_version` | 选择规则及版本 |
| `minimum_conditions_per_op_type` | 每类请求的 distinct condition 目标数 |
| `target_op_types` | 本轮目标类别，排序且唯一 |
| `groups` | 每类 available、distinct、selected 数和满足状态 |
| `selected_cases` | case、shape/workload/condition fingerprint、元数据等级、rank 和原因 |
| `issues` | proxy 补位或 condition 不足等显式问题 |
| `plan_id` | 除自身外完整计划内容的确定性哈希 |

计划加载时必须同时验证 schema、plan ID、manifest SHA，并使用计划保存的 backend、policy、
policy version、K 和 target op types 调用当前选择算法重建完整计划。重建结果必须与保存计划
逐字段相等；仅让 `plan_id` 自洽而未由当前算法产生的计划不得通过。manifest 内容或算法结果
不一致时必须失败。

## 六、数据源二：`cuda_kernel_latency.json`

### 6.1 唯一生产者

正式 latency 由以下测试产生：

```text
replay_benchmark/tests/test_kernel_latency.cpp
目标：replay_kernel_latency
测试：KernelLatency.AllCudaCases
```

测试读取当前 `operator_cases.json`，枚举全部 CUDA case，并为每个 case 独立运行：

```text
replay_benchmark.out
  --kernel-corpus-case <case>
  --kernel-corpus-runs 1
  --perf-counter-events none
```

测试要求 `pmu_status == "disabled"`。latency 由 CUDA event 测量，单位为微秒，不包含 CUPTI
replay 开销。launch/resource metadata 可以通过额外 metadata launch 采样，但它不是 latency
样本，也不能与 PMC session 混为一个 run。

### 6.2 正式生成命令

```bash
cd build-x86-cuda

env -u REPLAY_KERNEL_LATENCY_CASE_FILTER \
  REPLAY_KERNEL_CORPUS_ROOT=../replay_benchmark/kernel_corpus \
  REPLAY_KERNEL_LATENCY_OUTPUT=cuda_kernel_latency.json \
  LD_LIBRARY_PATH=.:source/backend/cuda:. \
  ./replay_kernel_latency --filter KernelLatency.AllCudaCases
```

正式采集不得设置 `REPLAY_KERNEL_LATENCY_CASE_FILTER`。该变量只允许用于单 case 冒烟检查。

### 6.3 结构化 latency 契约

顶层 schema 固定为：

```text
format = "mnn-kernel-latency"
version = 1
source_manifest_sha256 = operator_cases.json 原始字节的 SHA256
cases = case name 到结构化记录的映射
```

完整示例：

```json
{
  "format": "mnn-kernel-latency",
  "version": 1,
  "source_manifest_sha256": "<64位小写十六进制SHA256>",
  "cases": {
    "cuda_relu_fp32_smoke": {
      "latency_us": 6.144,
      "launch_sampling_status": "sampled",
      "launch_records": [
        {
          "kernel_name": "relu_kernel",
          "grid": [64, 1, 1],
          "block": [256, 1, 1],
          "registers_per_thread": 16,
          "static_shared_memory_bytes": 0,
          "dynamic_shared_memory_bytes": 0,
          "local_memory_per_thread_bytes": 0,
          "local_memory_total_bytes": 0
        }
      ],
      "environment": {
        "sampling_source": "nvml",
        "sampling_status": "sampled",
        "gpu_clock_hz_before": 1350000000,
        "gpu_clock_hz_after": 1365000000,
        "temperature_c_before": 51.0,
        "temperature_c_after": 52.0
      }
    }
  }
}
```

字段语义：

| 字段 | 要求 |
| --- | --- |
| `format`、`version` | 必须分别为 `mnn-kernel-latency` 和 `1` |
| `source_manifest_sha256` | 必须等于本轮 `operator_cases.json` 原始字节 SHA256；任意字节变化都失配 |
| `cases` | 必须是按 case name 索引的结构化对象 |
| `latency_us` | 正式数据必须是大于 0 的有限数 |
| `launch_sampling_status` | launch metadata 采样状态；成功时必须有非空记录。只有原始状态明确为 `no_launch` 才表示没有 device launch；`unavailable:no kernel activity records` 表示采集器未获得记录，必须归为 `unavailable` |
| `launch_records` | 一个 case 可能触发多个真实 device-kernel，必须逐条保留 |
| `kernel_name`、`grid`、`block` | 真实 launch 名称与几何，不得用 manifest 参数猜测 |
| resource 字段 | 每线程寄存器、静态/动态 shared memory、CUPTI 报告的每线程/总 local-memory reservation；0 是合法实测值，不能直接等同于实际 spill 流量 |
| `environment.sampling_source/status` | 环境采样来源与状态 |
| clock/temperature 前后值 | 只有实测可用时才出现；缺失不得填 0 |
| `error` | 失败诊断；出现失败时正式测试必须非零 |

顶层不保存 `collector`、`run_id`、`collection_id` 或 `workload_runs`；每个 launch record 也不
保存 `stage_index` 或 `repeat_index`。loader 按 `launch_records` 数组顺序补出
`stage_index`，并为每个 case 构造独立 latency run。不能从这些合成 ID 推断独立 repeat。

最终文件不输出 `pmu_status`；`replay_kernel_latency` 在读取中间 benchmark report 时已经强制
验证 PMU 为 disabled。

`dataset/sources/mnn_replay.py` 将每个 latency case 构造成独立 `MeasurementRun`，保存
`KernelLaunchRecord` 和采集前后 `EnvironmentSample`。该 run 与 PMC collection session
独立，只能按 condition 对齐。当前单次 latency 文件没有独立 repeat 设计，不能用 case 数、
workload runs 或 launch records 代替 `repeat_id`。

正式验收要求：

- key 与当前 manifest 的全部 CUDA case name 完全一致；
- `source_manifest_sha256` 与当前 manifest 原始字节 SHA256 完全一致；
- 所有 `latency_us` 有限且大于 0；
- 所有 case 都通过 PMU disabled、launch metadata 和 environment 结构校验；
- 任何 case 失败时测试失败；`null` 或错误对象只能用于诊断，不能进入正式数据集。

## 七、数据源三：`cuda_kernel_pmc_kernelreplay_rows.csv`

PMC rows 分两阶段产生：先发现当前设备的有效 metric，再执行 plan-selected case × metric
sweep。

### 7.1 阶段一：设备 availability

```bash
cd build-x86-cuda
set -o pipefail

sudo env LD_LIBRARY_PATH=.:source/backend/cuda:. \
  ./replay_pmu_availability --filter CudaMetric.AllMetrics \
  2>&1 | tee sweep.log

python3 ../skills/gpu-pmu-sweep/scripts/parse_cuda_pmu_log.py \
  --input sweep.log \
  --all-csv cuda_pmc_all.csv \
  --valid-csv cuda_pmc_valid.csv
```

`parse_cuda_pmu_log.py` 默认只发布完整且通过的 `AllMetrics` 结果。逐行 `index` 必须从 1
连续到统一的 `total`，结果行数必须等于 `total`，汇总状态计数必须与逐行状态一致，日志还
必须包含 `CudaMetric.AllMetrics` 的 `OK` 和测试套件的 `PASSED` 结束标记。中断、截断、缺号、
重复追加或测试未完成时，解析器必须非零退出，并且不得创建或覆盖目标 CSV。正式运行必须从
新的空日志和新的暂存 CSV 开始，不能把部分 `sweep.log.next` 解析为本轮有效 metric 清单。

`cuda_pmc_all.csv` 按发现请求位置保留完整状态。非 `VALID` 的无后缀名称可能重复，不能据此
删除请求。`cuda_pmc_valid.csv` 与正式 rows 中的有效 metric 必须唯一，并保持确定性顺序。

availability 必须由本轮重新构建的 collector 实测，不能把历史有效 metric 数写死为验收
目标。出现 `OVERFLOW -> VALID` 等状态迁移时，先对至少一个迁移 metric 做端到端数值冒烟，
确认 C++ report、CSV 和 JSON 没有整数溢出或浮点截断，再开始全量 sweep。

Kernel corpus JSON 使用 `pmu_metric_statuses` 保存每个请求 metric 的 `valid`、`invalid` 或
`overflow`，并仅在 `pmu_metrics` 中写入有限、非负的有效数值。CUDA/NVPW 的 double 不得转换
为整数；Adreno/Mali 的原生 uint64 counter 仍按 JSON 整数写出，避免超过 `2^53` 后失真。

### 7.2 首次正式采集

首次采集必须使用空账本，不带 `--resume`：

```bash
cd build-x86-cuda
sudo -v

python3 ../skills/gpu-pmu-sweep/scripts/sweep_cuda_kernel_pmc.py \
  --valid-csv cuda_pmc_valid.csv \
  --corpus-root ../replay_benchmark/kernel_corpus \
  --binary ./replay_benchmark.out \
  --workdir . \
  --lib-dir .:source/backend/cuda:. \
  --case-selection condition-balanced \
  --conditions-per-op-type 5 \
  --target-op-type matmul \
  --target-op-type conv_dw \
  --target-op-type reduction \
  --target-op-type softmax \
  --target-op-type transpose \
  --target-op-type maxpool \
  --target-op-type avgpool \
  --selection-plan-output cuda_kernel_pmc_kernelreplay_rows.csv.selection.json \
  --device-fingerprint '<GPU型号|驱动|CUDA|CUPTI身份>' \
  --metrics-per-session 32 \
  --runs 1 \
  --sudo \
  --output-csv cuda_kernel_pmc_kernelreplay_rows.csv \
  --output-json cuda_kernel_pmc_kernelreplay.json
```

采集语义：

- 每行是一个 `(case, metric)`；
- 每个 benchmark 子进程只运行一个 case；
- 每个真实 benchmark/CUPTI session 获得唯一 `collection_session_id`；
- 同 session 的多个 metric 共享 `collection_session_id` 和 `order_index`；
- CUPTI replay pass 由该 session 内部管理；
- `--kernel-corpus-no-latency` 由脚本自动传入；
- `--runs 1` 是单次 measurement 内 launch 数，不是独立 repeat；
- 批量 metric 配置失败时脚本二分重试，最终可降级到单 metric；
- CSV 是完整状态与断点账本；紧凑 JSON 只保存 `VALID` 值，不能替代 CSV。

可以重复 `--target-op-type <名称>` 只采集指定类别。显式全 case 扫描使用
`--case-selection all`。

### 7.3 CSV 固定字段

字段顺序以 `sweep_cuda_kernel_pmc.py::ROW_COLUMNS` 为准：

```text
sweep_plan_id
sweep_config_id
case
metric
status
value
pmu_status
num_passes
returncode
error
collection_session_id
order_index
gpu_clock_hz_before
gpu_clock_hz_after
temperature_c_before
temperature_c_after
environment_status
environment_source
```

| 字段 | 物理含义 |
| --- | --- |
| `sweep_plan_id` | 本行所属 `CaseSelectionPlan.plan_id` |
| `sweep_config_id` | metric 来源、顺序、binary、运行参数和设备指纹等 sweep 身份哈希 |
| `case`、`metric` | 当前笛卡尔积单元 |
| `status` | `VALID`、`NOT_FOUND`、`OVERFLOW`、`COMMAND_FAILED` 或 `MALFORMED_OUTPUT` |
| `value` | 原始 metric 值；CUDA `VALID` 值必须为非负有限数，只有 `VALID` 才进入分析特征 |
| `pmu_status` | benchmark 返回的 profiler 状态 |
| `num_passes` | CUPTI replay pass 数；不是 repeat |
| `returncode`、`error` | 子进程结果与诊断 |
| `collection_session_id` | 一次真实 benchmark/CUPTI session 身份 |
| `order_index` | session 的实际串行采集顺序；同 session metric 相同 |
| clock/temperature 前后列 | benchmark 环境采样值；未采到时必须为空，绝不能填 0 |
| `environment_status/source` | benchmark JSON 中环境采样状态和来源 |

launch geometry 与 resource usage 不写入 PMC rows，由结构化 latency JSON 独立保存。
不得只保留 `VALID` 行后覆盖正式 rows CSV；完整状态是质量评估和失败审计的一部分。

### 7.4 严格断点续跑

续跑必须显式复用首次保存的 plan，并保持所有 sweep 参数一致：

```bash
python3 ../skills/gpu-pmu-sweep/scripts/sweep_cuda_kernel_pmc.py \
  --valid-csv cuda_pmc_valid.csv \
  --corpus-root ../replay_benchmark/kernel_corpus \
  --binary ./replay_benchmark.out \
  --workdir . \
  --lib-dir .:source/backend/cuda:. \
  --selection-plan-input cuda_kernel_pmc_kernelreplay_rows.csv.selection.json \
  --selection-plan-output cuda_kernel_pmc_kernelreplay_rows.csv.selection.json \
  --device-fingerprint '<与首次采集完全相同>' \
  --metrics-per-session 32 \
  --runs 1 \
  --sudo --resume \
  --output-csv cuda_kernel_pmc_kernelreplay_rows.csv \
  --output-json cuda_kernel_pmc_kernelreplay.json
```

断点续跑固定校验：

- plan 文件存在、plan ID 正确、manifest SHA 正确；
- 当前选择算法重建的完整 plan 与保存计划相同；
- sweep config ID 与每一行相同；
- CSV 表头与 `ROW_COLUMNS` 完全相同；
- 行顺序是 selection plan case 顺序 × `cuda_pmc_valid.csv` metric 顺序；
- 旧账本恰好是该确定性笛卡尔积的合法前缀；
- 不存在额外行、重复 pair、旧 schema 或混合身份。

断点续跑校验通过只表示账本可以继续，不表示旧数据已经完成正式发布验收。

## 八、三数据源严格一致性验收

正式生成顺序固定为：

1. 更新并验证 `operator_cases.json` 的 workload metadata 与 provenance；
2. 重新构建四个 CUDA 目标；
3. 运行 availability，生成新的 `cuda_pmc_valid.csv`；
4. 从空账本生成 `CaseSelectionPlan` 与 PMC rows；
5. 独立运行 `replay_kernel_latency`，生成结构化 latency JSON；
6. 完成以下集合、schema、顺序和数值校验；
7. 最后生成 `mnn-pmc-dataset/v1`。

硬性验收规则：

| 规则 | 要求 |
| --- | --- |
| Manifest 唯一性 | CUDA case name 唯一，所有引用可执行 |
| Plan 来源 | manifest SHA、schema、policy version、plan ID 和算法重建全部一致 |
| Plan 状态 | 每个 target op type 都有 group；不足 K 的类别明确标记并全选 |
| Latency 覆盖 | latency key 集合等于全部 CUDA manifest case 集合 |
| 结构化 latency | 每个 case 有正 latency、launch/resource 和环境结构；失败时整体不发布 |
| PMC case 覆盖 | rows case 集合及顺序等于 plan 的 `selected_cases` |
| PMC metric 覆盖 | metric 集合及顺序等于本轮 `cuda_pmc_valid.csv` |
| Rows schema | 表头与 18 个 `ROW_COLUMNS` 完全相同 |
| Rows 身份 | 全部行使用唯一且正确的 plan ID 和 sweep config ID |
| PMC 唯一性 | `(case, metric)` 不重复 |
| 笛卡尔完整性 | 完整采集后 rows 等于 plan-selected case × ordered metric |
| 行顺序 | rows 是确定性笛卡尔积的完整有序序列 |
| 状态与数值 | `VALID` PMC 有有限 value；非 `VALID` 状态仍完整保留 |
| Session 语义 | 同 session 的 rows 必须连续，且共享 case、order、PMU 结果字段和环境值；关闭后不得再次重开 |
| 环境缺失 | 未采到 clock/temperature 时为空，不以 0 代替；`sampled` 必须有非空来源，实测 clock 与 temperature 都必须大于 0 |
| Join 完整性 | 每个 plan-selected case 都存在有效 latency |

`load_mnn_kernelreplay_dataset()` 可以把不完整输入加载为带 `issues` 的探索 dataset，但它不是
正式数据发布门禁。正式替换三数据源前必须运行：

```bash
cd build-x86-cuda

python3 ../skills/pmc-source-gate/scripts/validate_cuda_pmc_sources.py \
  --operator-cases ../replay_benchmark/kernel_corpus/operator_cases.json \
  --latency-json cuda_kernel_latency.json \
  --selection-plan cuda_kernel_pmc_kernelreplay_rows.csv.selection.json \
  --valid-csv cuda_pmc_valid.csv \
  --pmc-rows cuda_kernel_pmc_kernelreplay_rows.csv \
  --require-complete-targets \
  --require-measured-environment \
  --output-json cuda_pmc_source_validation.json
```

发布门禁的结构化输出 schema 为 `mnn-cuda-pmc-source-validation/v1`。它检查 manifest、
结构化 latency、plan、有效 metric 清单和 rows；任一 mismatch 时 `valid=false` 且进程
非零退出。

P0 正式验收必须同时使用两个加强开关：

- `--require-complete-targets`：plan 必须使用当前 policy version 的 `condition-balanced` 和固定
  `K=5`；target 集合必须精确等于 `avgpool`、`conv_dw`、`matmul`、`maxpool`、`reduction`、
  `softmax`、`transpose`，缺少或多出任何类别都失败；七个 group 都必须为 `satisfied`，
  `requested_condition_count=5`、选中数不少于 5，且每个 selected case 的 op type、dtype、
  shape、workload 和 params 必须与 manifest 一致并具有显式 provenance；
- `--require-measured-environment`：P0 selected latency case 与所有 PMC session 都必须有实测
  前后 clock/temperature 和非空采样来源，不能以 missing、not_collected 或 0 通过。

只有验收 JSON 中 `valid=true`，才允许执行下一节的 normalized dataset 和语义报告生成。

## 九、生成 PMC Interpreter 统一数据集

严格验收通过后运行：

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

统一产物：

| 产物 | Schema | 用途 |
| --- | --- | --- |
| 标准化数据集 | `mnn-pmc-dataset/v1` | 保存实验事实、provenance、run/session、launch/resource 和环境信息 |
| 结构化分析报告 | `mnn-pmc-analysis-report/v1` | 保存全部 layer report、PMC 知识、signature 和 delta rulebook |
| Markdown 文档 | `mnn-pmc-markdown-document/v1` | 保存面向人的固定章节中文解释 |

`dataset/sources/mnn_replay.py` 的职责是连接已给出的事实：

- manifest -> `KernelCondition`、shape/workload provenance；
- selection plan -> PMC condition 范围与采样身份；
- rows session -> 独立 PMC `MeasurementRun` 和 `MetricObservation`；
- 结构化 latency -> 独立 latency `MeasurementRun`、`KernelLaunchRecord`、
  `EnvironmentSample` 和 `LatencyObservation`。

它不会把 `num_passes` 变成 repeat，不会把 PMC 与 latency 合并成同一 run，也不会把缺失的
环境或 workload 信息补造成实测事实。

## 十、当前正式数据状态（2026-08-01）

manifest、collector、selection policy、latency schema 和 rows schema 已变化。当前正式三数据源
必须从空账本全量重采。

在以下验收全部完成前，旧文件只能作为历史探索数据，不得称为正式有效输入：

- 新 `CaseSelectionPlan` 的来源、plan ID、group status 和 selected cases 已验证；
- 新 availability 与 collector 数值链路已验证；
- 结构化 latency 覆盖全部 CUDA manifest case；
- rows 满足新 18 列 schema、唯一身份和完整有序笛卡尔积；
- plan-selected rows 与 latency 严格连接；
- `validate_cuda_pmc_sources.py` 使用两个 P0 加强开关并输出 `valid=true`；
- normalized dataset 的 provenance、session、launch/resource 和环境 readiness 已检查。

因此本文不记录新的最终 selected-case 数、metric 数、rows 数、plan ID 或 manifest SHA。它们
只能由下一轮完整采集和严格验收实际产生，不能从旧数据或 P0 cohort 数量推断。
