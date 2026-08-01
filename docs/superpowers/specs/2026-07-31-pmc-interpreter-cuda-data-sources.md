# PMC Interpreter CUDA 三数据源规范

## 一、目的

本文规定 `kernel_agent/pmc_interpreter` 在分析 CUDA kernel 时使用的三份权威数据源，
以及它们从 `replay_benchmark`、`replay_benchmark/tests` 和 GPU PMU sweep 工具中产生、
刷新与验收的方式。

固定路径如下：

| 数据源 | 固定路径 | 角色 |
| --- | --- | --- |
| Kernel case 定义 | `replay_benchmark/kernel_corpus/operator_cases.json` | 定义有哪些 kernel condition 可以运行 |
| CUDA PMC 明细 | `build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv` | 记录每个代表 case 与每个设备有效 PMC 的采集结果 |
| CUDA 延迟 | `build-x86-cuda/cuda_kernel_latency.json` | 记录每个 CUDA case 在关闭 PMU 后的独立延迟 |

三份文件共同组成 PMC Interpreter 的 CUDA 原始输入，但职责不同：

```text
operator_cases.json
  ├─ 选择 CUDA case ────────────────┐
  │                                │
  │                                ▼
  │                    cuda_kernel_latency.json
  │                    replay_kernel_latency 产生
  │
  └─ 按 op_type 选择代表 case ─────┐
                                   │
cuda_pmc_valid.csv ────────────────┤
设备有效 metric 清单               ▼
                    cuda_kernel_pmc_kernelreplay_rows.csv
                    sweep_cuda_kernel_pmc.py 产生
```

三份原始输入验收通过后，由 `dataset/sources/mnn_replay.py` 构造
`mnn-pmc-dataset/v1`；固定串行 layer 再生成 `mnn-pmc-analysis-report/v1`，并可选渲染中文
Markdown。数据源加载、统计分析和文档渲染是三个独立阶段。

`operator_cases.json` 是受版本控制的源数据；另外两份是当前 GPU、driver、CUDA、
CUPTI 和代码版本下的测量结果。不得把三份来自不同 manifest 版本或不同设备环境的文件
拼成同一个数据集。

单次 Interpreter 分析还必须保持 device-scoped：三数据源构造出的所有
`KernelCondition.device_id` 必须指向同一块设备。多 GPU 或跨机器结果必须先按 device
拆分成多个 dataset，分别分析后再在概念层比较。

## 二、统一构建前提

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

CUDA CUPTI Range Profiler 通常需要管理员权限。一次只能运行一个完整 availability 或
sweep 实例，避免多个进程竞争 profiler 资源。

## 三、数据源一：`operator_cases.json`

### 3.1 权威来源

`replay_benchmark/kernel_corpus/operator_cases.json` 是仓库内维护的可执行 case
manifest，不是运行 `replay_benchmark/tests` 后自动生成的文件。

它由 kernel corpus 适配过程维护，并被以下生产者读取：

- `replay_kernel_latency`：枚举全部 `backend == "cuda"` 的 case；
- `sweep_cuda_kernel_pmc.py`：枚举全部 CUDA case，或按 `op_type` 选代表 case；
- `replay_benchmark.out --kernel-corpus-bench`：查找 kernel、构造参数、运行并校验结果；
- `dataset/sources/mnn_replay.py`：构造 `PmcDataset`，补充 `op_type`、dtype、variant、
  参数和语义分类。

`operators.json` 与它不是同一个数据源：

- `operators.json` 是 kernel 文件与 entry 的索引，可以由
  `extract_operator_kernels.py` 重新生成；
- `operator_cases.json` 是可执行 workload 定义，需要与对应 `OpAdapter`、validator
  和 launch 参数一起人工维护。

### 3.2 文件格式

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
| `name` | 必填且全局唯一 | 后续 CSV/JSON 的 join key |
| `backend` | 必填 | CUDA case 固定为 `cuda` |
| `op_type` | 必填 | 语义算子类型，也是代表 case 分组键 |
| `framework` | 必填 | kernel 来源框架 |
| `tag` | 必填 | kernel 来源版本 |
| `variant` | 必填 | 对应 `replay_benchmark` 的 `OpAdapter`/kernel 变体 |
| `int_params` | 应提供 | `OpAdapter` 使用的整数 workload 参数 |
| `float_params` | 可选 | `OpAdapter` 使用的浮点参数 |
| `dtype` | 可选 | 数据类型；缺失时 PMC Interpreter 标为 `unknown` |
| `validator` | 可选 | 正确性验证器 |
| `warmup_runs` | 可选 | 缺省值为 2 |
| `workload_runs` | 可选 | 缺省值为 5；不等于独立 repeat |

### 3.3 代表 case 规则

规范的 PMC rows 使用：

```text
--case-selection op-type
```

选择规则由 `sweep_cuda_kernel_pmc.py` 固定：

1. 只保留 `backend == "cuda"`；
2. 对每个 `op_type` 保留 manifest 中出现的第一个 case；
3. 最终按 `op_type` 字典序输出代表 case。

因此，调整 `operator_cases.json` 的顺序也可能改变代表 case。若增加、删除、重排
CUDA case，必须重新生成 PMC rows 和 latency，不能继续复用旧输出。

### 3.4 获取与基础校验

该文件直接从仓库获取，不从 build 目录复制：

```bash
python3 -m json.tool \
  replay_benchmark/kernel_corpus/operator_cases.json >/dev/null
```

修改 kernel corpus 后至少运行：

```bash
python3 -m unittest discover \
  -s replay_benchmark/kernel_corpus/tests -p 'test_*.py'
```

还需要通过 `replay_benchmark.out --kernel-corpus-bench` 验证新增 case 可以被
`operators.json` 和对应 `OpAdapter` 解析；Python extractor 测试不能代替真机执行验证。

## 四、数据源二：`cuda_kernel_latency.json`

### 4.1 唯一生产者

正式 latency 由以下测试产生：

```text
replay_benchmark/tests/test_kernel_latency.cpp
目标：replay_kernel_latency
测试：KernelLatency.AllCudaCases
```

该测试读取当前 `operator_cases.json`，枚举所有 CUDA case，并为每个 case 启动一次：

```text
replay_benchmark.out
  --kernel-corpus-runs 1
  --perf-counter-events none
```

测试会确认 `pmu_status == "disabled"`。延迟由 CUDA event 测量，单位为微秒，
不包含 CUPTI replay 开销。

### 4.2 正式生成命令

在仓库根目录构建后，从 build 目录执行：

```bash
cd build-x86-cuda

env -u REPLAY_KERNEL_LATENCY_CASE_FILTER \
  REPLAY_KERNEL_CORPUS_ROOT=../replay_benchmark/kernel_corpus \
  REPLAY_KERNEL_LATENCY_OUTPUT=cuda_kernel_latency.json \
  LD_LIBRARY_PATH=.:source/backend/cuda:. \
  ./replay_kernel_latency --filter KernelLatency.AllCudaCases
```

正式采集不得设置 `REPLAY_KERNEL_LATENCY_CASE_FILTER`。该变量只允许用于单 case
冒烟检查。

### 4.3 输出契约

输出是 `case name -> latency_us` 的 JSON 对象：

```json
{
  "cuda_relu_fp32_smoke": 6.144,
  "cuda_softmax_fp32_smoke": 12.288
}
```

验收条件：

- key 必须与当前 manifest 中全部 CUDA case 完全一致；
- value 必须是大于 0 的有限数；
- 任何 case 失败时测试必须失败；
- `null` 只能作为失败诊断，不得进入正式数据集；
- latency 测量与 PMC sweep 必须独立运行。

## 五、数据源三：`cuda_kernel_pmc_kernelreplay_rows.csv`

PMC rows 分两阶段产生：先发现当前设备的有效 metric，再执行 case × metric sweep。

### 5.1 阶段一：产生设备有效 metric 清单

在 `build-x86-cuda` 中运行：

```bash
cd build-x86-cuda

sudo LD_LIBRARY_PATH=.:source/backend/cuda:. \
  ./replay_pmu_availability --filter CudaMetric.AllMetrics \
  2>&1 | tee sweep.log
```

`CudaMetric.AllMetrics` 由 `test_pmu_availability.cpp` 实现。它通过
`list_cuda_metrics --no-submetrics` 发现 metric，并在固定 CUDA smoke kernel 上逐个
采集，输出 `VALID`、`NOT_FOUND`、`OVERFLOW` 或 `COMMAND_FAILED`。

将日志转换成完整清单和有效清单：

```bash
python3 ../skills/gpu-pmu-sweep/scripts/parse_cuda_pmu_log.py \
  --input sweep.log \
  --all-csv cuda_pmc_all.csv \
  --valid-csv cuda_pmc_valid.csv
```

`cuda_pmc_valid.csv` 是 PMC rows 的直接前置输入，但不是 PMC Interpreter 的三份
最终数据源之一。它是设备、driver 和 profiler 版本相关的中间产物。

### 5.2 阶段二：产生正式 rows CSV

仍在 `build-x86-cuda` 中执行：

```bash
sudo -v

python3 ../skills/gpu-pmu-sweep/scripts/sweep_cuda_kernel_pmc.py \
  --valid-csv cuda_pmc_valid.csv \
  --corpus-root ../replay_benchmark/kernel_corpus \
  --binary ./replay_benchmark.out \
  --workdir . \
  --lib-dir .:source/backend/cuda:. \
  --case-selection op-type \
  --metrics-per-session 32 \
  --runs 1 \
  --sudo \
  --resume \
  --output-csv cuda_kernel_pmc_kernelreplay_rows.csv \
  --output-json cuda_kernel_pmc_kernelreplay.json
```

采集语义：

- 每行是一个 `(case, metric)`；
- 每个 benchmark 子进程只运行一个 case；
- 每个 CUPTI session 最多请求 32 个 metric；
- CUPTI 内部 replay pass 由同一个 session 管理；
- `--kernel-corpus-no-latency` 由脚本自动传入；
- `--runs 1` 是单次测量内 launch 数，不是独立 repeat；
- 批量 metric 配置失败时脚本会二分，最终降级到单 metric；
- CSV 是断点账本，JSON 只保存 `VALID` 值。

只有在 `operator_cases.json`、`cuda_pmc_valid.csv`、GPU、driver、CUDA 和 collector
均未变化时，才允许对原 CSV 使用 `--resume`。任一前置条件变化时，应先归档旧文件，
从新的空账本开始采集。

### 5.3 CSV 字段契约

字段顺序固定为：

```text
case,metric,status,value,pmu_status,num_passes,returncode,error
```

| 字段 | 含义 |
| --- | --- |
| `case` | `operator_cases.json` 中的 case name |
| `metric` | 当前设备的 native CUPTI metric |
| `status` | 账本状态：`VALID`、`NOT_FOUND`、`OVERFLOW`、`COMMAND_FAILED`、`MALFORMED_OUTPUT` |
| `value` | 原始 metric 值；只有 `VALID` 才进入分析特征 |
| `pmu_status` | `replay_benchmark.out` 返回的 profiler 状态 |
| `num_passes` | CUPTI replay pass 数；不是 repeat，也不能用于稳定性估计 |
| `returncode` | benchmark 子进程返回码 |
| `error` | 失败诊断文本 |

不得只保留 `VALID` 行后覆盖正式 rows CSV。完整状态是质量评估和失败审计的一部分。

## 六、三数据源一致性要求

正式生成顺序固定为：

1. 更新并验证 `operator_cases.json`；
2. 重新构建四个 CUDA 目标；
3. 运行 availability，生成新的 `cuda_pmc_valid.csv`；
4. 从空账本生成 `cuda_kernel_pmc_kernelreplay_rows.csv`；
5. 独立运行 `replay_kernel_latency`，生成 `cuda_kernel_latency.json`；
6. 完成以下集合与数值校验；
7. 最后生成 `mnn-pmc-dataset/v1`。

硬性验收规则：

| 规则 | 要求 |
| --- | --- |
| Manifest 唯一性 | 所有 case name 唯一 |
| Latency 覆盖 | latency keys 等于全部 CUDA case names |
| PMC case 覆盖 | rows case 集合等于当前每个 `op_type` 的代表 case 集合 |
| Join 完整性 | 每个 rows case 都存在 latency |
| PMC 唯一性 | `(case, metric)` 不重复 |
| 笛卡尔完整性 | rows 数等于代表 case 数乘以 metric 数 |
| 数值有效性 | latency 全部大于 0；`VALID` PMC 必须有有限 value |
| 状态保留 | 非 `VALID` 行仍保留在 rows CSV |

## 七、生成 PMC Interpreter 统一数据集

三份数据通过验收后运行：

```bash
python3 -m kernel_agent.pmc_interpreter \
  --pmc-csv build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv \
  --latency-json build-x86-cuda/cuda_kernel_latency.json \
  --operator-cases replay_benchmark/kernel_corpus/operator_cases.json \
  --platform cuda \
  --backend cuda \
  --namespace cupti \
  --normalized-output-json build-x86-cuda/cuda_pmc_dataset_v1.json \
  --output-json build-x86-cuda/cuda_pmc_analysis_report_v1.json \
  --output-md build-x86-cuda/cuda_pmc_analysis_report_v1.md
```

统一输入、分析报告和 Markdown 的 schema 分别为：

| 产物 | Schema | 用途 |
| --- | --- | --- |
| 标准化数据集 | `mnn-pmc-dataset/v1` | 保存按只读约定消费的实验事实，供不同平台共用分析层 |
| 结构化分析报告 | `mnn-pmc-analysis-report/v1` | 保存全部 layer report、PMC 知识、signature 和 delta rulebook |
| Markdown 文档 | `mnn-pmc-markdown-document/v1` | 保存面向人的固定章节中文解释 |

代码的固定职责如下：

| 模块 | 职责 |
| --- | --- |
| `kernel_agent/pmc_interpreter/dataset/model.py` | 定义 `PmcDataset` 及全部实验事实对象 |
| `kernel_agent/pmc_interpreter/dataset/reports.py` | 定义特征、分析配置、逐层输入输出和 `PmcAnalysisReport` |
| `kernel_agent/pmc_interpreter/dataset/document.py` | 定义 `MarkdownSection` 与 `MarkdownDocument` |
| `kernel_agent/pmc_interpreter/dataset/catalog/` | 在 dataset 构造阶段写入 kernel 分类和 metric 语义 |
| `kernel_agent/pmc_interpreter/dataset/sources/mnn_replay.py` | 将本规范三数据源连接为 `PmcDataset` |
| `kernel_agent/pmc_interpreter/dataset/bundle.py` | 读写 `mnn-pmc-dataset/v1` JSON |
| `kernel_agent/pmc_interpreter/layers/` | 严格串行产生带类型的逐层报告，不读取平台专用原始文件 |
| `kernel_agent/pmc_interpreter/semantics/compiler.py` | 只将完整层报告编译为 `mnn-pmc-analysis-report/v1` |
| `kernel_agent/pmc_interpreter/semantics/markdown.py` | 只将结构化报告渲染为中文 Markdown |

`cuda_pmc_dataset_v1.json` 是可复用的标准化输入；三份原始数据仍必须保留，用于审计、
重新映射 metric 语义和验证数据源加载行为。

`dataset/catalog/default_kernel_taxonomy.json` 中的机制先验必须在构造时物化到
`KernelCondition.expected_mechanisms`。它是 Signature 层的显式输入；分析层不得根据 CUDA
平台或 `semantic_family` 再维护一份隐藏的预期机制表。

层间契约严格固定为：

```text
ProjectionOutput
  → QualityOutput
  → RedundancyOutput
  → CorrelationOutput
  → KernelSignatureOutput
  → DeltaRelationOutput
```

每层只接受上一层的具体 Pydantic 输出类型。最终编译器只接受
`DeltaRelationOutput`，不得跳过前置质量、冗余或相关性报告。

`--output-json` 必填，保存完整结构化分析报告；`--output-md` 可选，保存面向人的固定章节
中文文档。Markdown 不能替代 JSON，也不能作为下一轮分析输入。

若额外提供 implementation pairs，Delta 最低样本门槛按不同
`semantic_equivalence_key` 的独立语义组计数，不按 CSV 中的 pair 行数计数；同一语义组的
多行先做组内汇总。只有 baseline/candidate 至少共享两个非空 `paired_run_group_id`，并且
这些组都能同时连接有效 PMC 与 latency 重复，才具备因果升级的必要条件。

生成的 rulebook 仍是 device、kernel 类别、贡献配对观测区间和 intervention-scoped；不能
外推到其他设备、类别、未观测 `ΔPMC` 区间或不同干预机制。

## 八、当前工作区状态（2026-08-01）

当前 CUDA 三数据源已经按第六节顺序从同一份 manifest 快照重新生成，并通过严格集合、
笛卡尔积、协议字段和数值验收：

| 检查项 | 当前结果 |
| --- | --- |
| `operator_cases.json` | 919 个总 case；473 个 CUDA case；70 个 CUDA op_type；CUDA 子集 name 唯一 |
| Availability | 7,200 条请求；6,316 `VALID`；712 `NOT_FOUND`；172 `OVERFLOW`；0 `COMMAND_FAILED` |
| PMC rows | 70 个代表 case；6,316 个唯一有效 metric；442,120 行；442,120 个唯一 `(case, metric)`；全部 `VALID` |
| Latency JSON | 473 个有限正数 latency；key 与 CUDA manifest 完全一致 |
| Rows 与 latency | 70 个代表 case 全部可以连接 latency |
| 标准化数据集 | 70 个 condition；6,316 个 metric；442,120 个 PMC observation；70 个 latency observation |

本次采集使用的 manifest SHA256 为：

```text
f1987cc8f9644b608c3b6316b8d89662f686e399ffd3b1599db8e142182dc09d
```

旧的 60-case PMC、425-case latency 和旧 availability 文件已归档到：

```text
build-x86-cuda/pmc-archive/20260801-070449-pre-refresh/
```

当前还有两个不阻塞本轮 CUDA 数据的独立警告：

- manifest 全局存在 98 个非 CUDA 重名 case；CUDA 473 个 case 内没有重名；
- availability 完整清单中的 151 个非 `VALID` basename 各出现四次，但
  `cuda_pmc_valid.csv` 的 6,316 个 metric 全部唯一。

这两项不能解释为 CUDA 三数据源缺失，也不得在已有采集过程中修改 manifest 后继续
`--resume`；后续清理 manifest 时仍需重新执行完整刷新流程。
