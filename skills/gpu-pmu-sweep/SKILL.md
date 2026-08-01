---
name: gpu-pmu-sweep
description: 在 Adreno/Mali/NVIDIA GPU 上运行 replay_benchmark 的 kernel corpus case，采集每个 kernel 执行前后的 PMU（Performance Monitoring Unit）metric 值，计算 delta 增量并输出 CSV。覆盖 OpenCL（Adreno/Mali）、Vulkan（Mali/Adreno）、CUDA（NVIDIA CUPTI）三种后端，支持单 metric 单进程扫描、多次重复测量、raw JSON 保存。
---

# GPU PMU Sweep — Adreno / Mali / NVIDIA

> **触发条件**：需要对 GPU kernel 做 PMU 性能计数器采集时触发。包括：kernel corpus 的 per-kernel metric 采集、特定 case 的多 metric 扫描、不同形状参数下的性能对比。

## 概述

本 skill 通过 `replay_benchmark.out` 的 PMU 采集能力，对 kernel corpus 中每个 case 执行**前（control）**和**后（workload）**的 PMU 采样，输出 delta 增量：

```
delta_metric = workload_delta - control_delta
```

每个 GPU 后端有自己的 PMU 机制和 metric 列表：

| 后端 | GPU 厂商 | PMU 机制 | metric 发现方式 |
|------|---------|---------|----------------|
| OpenCL | Adreno (Rhinopi) | KGSL ioctl | `replay_benchmark.out --opencl-pmu-list-events` |
| OpenCL | Mali (OrangePi) | Mali Performance Counter | 同上 |
| Vulkan | Adreno/Mali | Vulkan extension query | 同上 |
| CUDA | NVIDIA | CUPTI Range Profiler | `list_cuda_metrics` 或 `--perf-counter-events` |

### 语义分析前置约束

PMC Interpreter 规范文档位于：

- 数据产生与同步：`docs/superpowers/specs/2026-07-31-pmc-interpreter-cuda-data-sources.md`
- 模块物理语义：`docs/superpowers/specs/2026-07-31-pmc-interpreter-module-physical-semantics.md`

涉及以下文件的产生、刷新、字段解释或完整性校验时，必须以数据产生与同步规范为准：

- `build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv`
- `build-x86-cuda/cuda_kernel_latency.json`
- `replay_benchmark/kernel_corpus/operator_cases.json`

- `operator_cases.json` 是受版本控制的权威 case manifest，不是 test 运行产物；
- manifest 变化后，PMC rows 与 latency 必须重新生成；
- 前置 case 或 metric 集合发生变化时，不得对旧 rows CSV 继续 `--resume`。

- `repeat_id`、一次测量内的 workload runs、profiler replay passes、
  workload-control delta、implementation baseline/candidate delta 是五个不同概念；
  不得互相代替。
- legacy CUDA rows 没有 `collection_session_id`。不同 metric 可能来自不同进程或
  CUPTI session，不能据此假设 numerator/denominator 同时采集，也不能盲目构造比值。
- `--case-selection op-type` 只取 `operator_cases.json` 中该 op_type 的第一条记录，
  不是语义代表选择；分析时还必须区分 semantic family、execution role 和 variant。
- 当前 CUDA collector/ledger 主要验证了整数 raw rollup。采集 ratio、throughput、
  per-cycle 等浮点 metric 前，必须确认 C++ report 和 Python ledger 全链路保留
  `double`，不能把小数强制转换为整数。
- 当前数据中两个 metric 向量完全相等，不等于公式或机制相同。只有同 dependency
  group/canonical concept 才能强制去重；跨机制相等仅作为冗余注释和选择惩罚。
- 通用名称推断只能生成低置信度 mechanism proxy。atomic byte/sector/request/wavefront 是原子
  操作工作量，不是竞争症状；只有 conflict/serialization/retry/stall 等事件才能标为 contention。
- 语义报告必须把类别直接 `resolved_features` 与样本不足时的 family 继承分开。继承项不能写成
  op-type 实证，也不能把未归一化 `.sum` 工作量 counter 写成效率指标或优化方向。
- availability 必须以本轮重新构建的 collector 实测结果为准，不能把旧的有效 metric 数量
  写死为验收目标。即使 GPU、driver 和 manifest 未换，旧 `OVERFLOW` 也可能在 collector
  重建后变成 `VALID`；正式 rows 行数必须按新的 `cuda_pmc_valid.csv` 动态计算。
- 如果 availability 出现 `OVERFLOW → VALID` 等状态迁移，先在一个代表 case 上直接采集至少
  一个迁移 metric，确认 report、CSV ledger 和 JSON 全链路能保存其数值，再启动全量 sweep。
- `cuda_pmc_all.csv` 是按发现请求位置保存的完整状态账本，非 `VALID` 的无后缀 metric 名称
  可能重复，不能据此删除请求行或要求 metric name 全局唯一；必须要求
  `cuda_pmc_valid.csv` 和正式 rows 中的有效 metric 唯一。
- `load_mnn_kernelreplay_dataset()` 可以把不完整旧数据加载为带 `issues` 的探索数据，不是正式
  数据发布 gate。替换正式三数据源前仍需用独立严格验收检查精确集合、笛卡尔积、状态、数值、
  行顺序和 latency join，并在任何 mismatch 时非零退出。

## PMU Metric 分类

### NVIDIA CUDA（CUPTI）

NVIDIA GPU（本机 TU102 / RTX 2080 Ti，sm75）共 **7200 个 metric**（1802 个去后缀 basename），按硬件域分类：

| 域前缀 | 数量 | 类别 | 说明 |
|--------|------|------|------|
| `lts__` | 814 | 缓存 | L2 Cache Slice（L2 缓存切片，命中率/sector/throughput） |
| `smsp__` | 335 | 计算 | SM Sub-Partition（SM 子分区，warp stall/指令/寄存器） |
| `l1tex__` | 307 | 缓存 | L1/Texture Cache（L1 纹理缓存，sector/hit_rate） |
| `sm__` | 183 | 计算 | Streaming Multiprocessor（SM 核心，指令/周期/占用率） |
| `gcc__` | 30 | 缓存 | Graphics Command Cache（图形命令缓存） |
| `tpc__` | 21 | 计算 | Texture Processing Cluster（纹理处理簇） |
| `gpu__` | 17 | 全局 | GPU 全局（时间/功耗/温度） |
| `fbpa__` | 13 | 显存 | Frame Buffer Partition（帧缓冲分区，DRAM 子单元） |
| `dram__` | 13 | 显存 | DRAM 控制器（读写字节/sector/throughput） |
| `idc__` | 12 | 显存 | Inter-DRAM Channel（DRAM 通道间） |
| `gr__` | 10 | 计算 | Graphics Engine（图形引擎） |
| `pcie__` | 7 | 总线 | PCIe 总线 |
| `fe__` | 5 | 前端 | Front End（前端命令处理器） |
| `sys__` | 4 | 系统 | 系统级 |
| `gpc__` | 4 | 计算 | Graphics Processing Cluster |
| `nvltx__/nvlrx__` | 各13 | 总线 | NVLink TX/RX |

**后缀**：每个 basename 有 4 种聚合方式 — `.avg`（平均）、`.sum`（总和）、`.max`（最大）、`.min`（最小）。少量 ratio 类 metric 无后缀。

#### 常用 CUDA metric 推荐

**计算性能**（compute bound 分析）：
- `sm__cycles_elapsed.avg` — SM 周期数（执行时间）
- `sm__inst_executed.avg` — 执行指令数
- `sm__warps_active.avg.per_cycle_active` — 活跃 warp 数（占用率）
- `sm__sass_thread_inst_executed_op_fadd_pred_on.sum` — FP32 加法指令数
- `sm__sass_thread_inst_executed_op_fmul_pred_on.sum` — FP32 乘法指令数
- `sm__sass_thread_inst_executed_op_ffma_pred_on.sum` — FP32 FMA 指令数

**显存带宽**（memory bound 分析）：
- `dram__bytes.sum` — DRAM 读写总字节
- `dram__bytes_read.sum` — DRAM 读字节
- `dram__bytes_write.sum` — DRAM 写字节
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — DRAM 带宽利用率
- `lts__t_bytes.sum` — L2 缓存总字节

**缓存命中**（cache 分析）：
- `l1tex__t_sector_hit_rate.pct` — L1 缓存命中率
- `lts__t_sector_hit_rate.pct` — L2 缓存命中率
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` — L1 全局加载 sector 数

**Warp 停顿**（stall 分析）：
- `smsp__warp_issue_stalled_long_scoreboard.avg.per_warp_active` — 长延迟等待（显存）
- `smsp__warp_issue_stalled_short_scoreboard.avg.per_warp_active` — 短延迟等待
- `smsp__warp_issue_stalled_mio_throttle.avg.per_warp_active` — MIO 管线拥塞

### Adreno（OpenCL/KGSL）

Adreno GPU 通过 KGSL ioctl 暴露 PMU 计数器，典型事件：

| 类别 | 事件名 | 说明 |
|------|--------|------|
| **计算** | `cp_busy_cycles` | CP 繁忙周期 |
| | `sp_busy_cycles` | SP 繁忙周期 |
| | `sp_cs_invocations` | Compute shader 调用次数 |
| **显存** | `rbbm_vbif_busy` | VBIF 总线繁忙 |
| | `uche_busy_cycles` | UCHE 繁忙周期 |
| **缓存** | `tp_l1_cacheline_requests` | L1 缓存请求 |
| | `tp_l1_cacheline_misses` | L1 缓存 miss |
| | `uche_busy_cycles` | UCHE（统一 L2）繁忙 |
| **指令** | `sp_gm_load_instructions` | 全局内存加载指令 |
| | `sp_gm_store_instructions` | 全局内存存储指令 |
| | `sp_lm_load_instructions` | 本地内存加载指令 |
| **线程** | `sp_working_eu_cs_stage` | 活跃执行单元 |

使用 `replay_benchmark.out --opencl-pmu-list-events` 发现设备支持的全部事件。

### Mali（OpenCL/Vulkan）

Mali GPU 通过 `MaliDebugInterface` 或 Vulkan 扩展暴露 PMU 计数器，典型事件：

| 类别 | 事件名 | 说明 |
|------|--------|------|
| **计算** | `gpu_active_cycles` | GPU 活跃周期 |
| | `compute_active_cycles` | Compute 活跃周期 |
| | `compute_tasks` | Compute 任务数 |
| **缓存** | `l2_any_lookup` | L2 查找总数 |
| | `l2_ext_read` | L2 外部读 |
| | `l2_ext_write` | L2 外部写 |

使用 `replay_benchmark.out --opencl-pmu-list-events` 发现设备支持的全部事件。

## 快速开始

### NVIDIA CUDA（本机）

```bash
# 单次采集（所有 case × 指定 metric）
sudo LD_LIBRARY_PATH=build/:build/source/backend/cuda:. build/replay_benchmark.out \
  --kernel-corpus-bench \
  --kernel-corpus-root replay_benchmark/kernel_corpus \
  --kernel-corpus-runs 1 \
  --perf-counter-events sm__cycles_elapsed.avg,dram__bytes.sum,sm__inst_executed.avg \
  --perf-counter-output /tmp/kc_pmu.json

# 查看
python3 -c "
import json
d=json.load(open('/tmp/kc_pmu.json'))
for c in d['cases'][:3]:
    print(c['case'], c.get('pmu_metrics',{}))
"
```

**权限**：CUDA CUPTI Range Profiler 需要 `sudo` 运行（或设置 `RmProfilingAdminOnly=0`）。

### PMU 进程清理

PMU 扫描被 `Ctrl-Z` 挂起后，进程仍可能持有 CUPTI/NVIDIA profiler 资源；再次启动扫描可能得到 `CUPTI_ERROR_HARDWARE_BUSY`。`replay_pmu_availability` 不要并发启动多个实例，它们还可能竞争固定的临时 JSON 文件。

通过进程名查找 PID 并清理测试进程：

```bash
pattern='(^|/| )replay_pmu_(test|availability)( |$)|(^|/| )replay_benchmark\.out.*--kernel-corpus-bench'
pids=$(pgrep -f "$pattern" || true)
if [ -n "$pids" ]; then
    # 先恢复被 Ctrl-Z 挂起的进程，否则 TERM 可能只会排队而不会退出。
    sudo kill -CONT $pids 2>/dev/null || true
    sudo kill -TERM $pids 2>/dev/null || true
    sleep 2
    pids=$(pgrep -f "$pattern" || true)
    [ -z "$pids" ] || sudo kill -KILL $pids
fi

pgrep -af 'replay_pmu_(test|availability)|replay_benchmark\.out.*--kernel-corpus-bench' || true
```

优先使用 `Ctrl-C` 结束前台扫描，不要使用 `Ctrl-Z`。如果只需清理当前 shell 的挂起任务，可先执行 `jobs -l`，再用 `fg %<job>` 恢复到前台并按 `Ctrl-C`。清理后确认没有残留 `replay_pmu_*` 或 `replay_benchmark.out --kernel-corpus-bench`，再启动下一轮采集。

### CUDA PMU metric availability 测试

`replay_benchmark/tests/test_pmu_availability.cpp` 会发现 CUDA metric，并按硬件域逐个执行单 metric 采集。一次只运行一个 availability 实例：

```bash
# 构建发现器和测试目标
cmake --build build-x86-cuda --target list_cuda_metrics replay_pmu_availability -j2

# 运行完整测试
cd build-x86-cuda
sudo LD_LIBRARY_PATH=.:source/backend/cuda:. ./replay_pmu_availability

# 运行单个测试过滤器
sudo LD_LIBRARY_PATH=.:source/backend/cuda:. ./replay_pmu_availability \
  --filter CudaMetric.SmDomain

# 逐个检查发现的全部 metric（7200 个 metric，预计耗时很长）
sudo LD_LIBRARY_PATH=.:source/backend/cuda:. ./replay_pmu_availability \
  --filter CudaMetric.AllMetrics
```

常用过滤器包括 `CudaMetric.Discovery`、`CudaMetric.AllMetrics`、`CudaMetric.SmDomain`、`CudaMetric.DramDomain`、`CudaMetric.L1texDomain`、`CudaMetric.LtsDomain`、`CudaMetric.SmspDomain`、`CudaMetric.KeyMetrics` 和 `CudaMetric.Reproducibility`。`AllMetrics` 每个 metric 启动一个单 metric benchmark，预计比域测试耗时很多；如果输出出现大量 `invalid` 或 `CUPTI_ERROR_HARDWARE_BUSY`，先按上面的进程清理流程停止残留扫描，再重试；不要把 metric 枚举成功当作 metric 采集成功。

### Adreno / Mali（远程设备）

```bash
# 单 metric 单进程扫描（脚本自动遍历所有 event × case × measurement）
python3 skills/gpu-pmu-sweep/scripts/sweep_opencl_pmu.py \
  --device rhinopi \
  --events all \
  --measurements 3 \
  --workload-runs 5 \
  --output-csv records/rhinopi-all.csv \
  --keep-json-dir records/rhinopi-all-raw
```

### CUDA kernel corpus sweep

```bash
# 扫描所有 CUDA case × 指定 metric 集合
sudo LD_LIBRARY_PATH=build/:build/source/backend/cuda:. build/replay_benchmark.out \
  --kernel-corpus-bench \
  --kernel-corpus-root replay_benchmark/kernel_corpus \
  --kernel-corpus-runs 3 \
  --perf-counter-events sm__cycles_elapsed.avg,sm__inst_executed.avg,dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct \
  --perf-counter-output /tmp/kc_sweep.json

# 输出 CSV
python3 -c "
import json, csv
d=json.load(open('/tmp/kc_sweep.json'))
with open('/tmp/kc_sweep.csv','w') as f:
    w=csv.writer(f)
    w.writerow(['case','variant','tag'] + [k for k in d['cases'][0].get('pmu_metrics',{})])
    for c in d['cases']:
        m=c.get('pmu_metrics',{})
        w.writerow([c['case'],c['variant'],c['tag']] + [m.get(k,'') for k in d['cases'][0].get('pmu_metrics',{})])
"
```

## 测量规则

- **CUDA**：`--kernel-corpus-runs N` 控制 workload 重复次数。PMU 在所有 N 次 launch 外包一个 range，输出的是 N 次聚合值。
- **OpenCL/Vulkan**：`--measurements M` 控制独立进程数，`--workload-runs N` 控制每进程 dispatch 数。一次只测一个 event（单 metric 单进程）。
- **delta 计算**：`delta = workload_delta - control_delta`。control 是空跑（无 kernel launch），workload 是实际执行。
- **raw JSON 保存**：始终保留原始 JSON 输出用于审计。

## CSV 契约

### CUDA 全量有效 PMC sweep

先把 `CudaMetric.AllMetrics` 的日志拆成两个 CSV。完整 CSV 保留所有状态，
`cuda_pmc_valid.csv` 只包含状态为 `VALID` 的 metric；`OVERFLOW` 的
`9223372036854775808` 哨兵值不会进入有效清单：

```bash
cd build-x86-cuda
python3 ../skills/gpu-pmu-sweep/scripts/parse_cuda_pmu_log.py \
  --input sweep.log \
  --all-csv cuda_pmc_all.csv \
  --valid-csv cuda_pmc_valid.csv
```

默认从 CUDA kernel corpus 中按 `op_type` 选择一个代表 case（数量随当前
`operator_cases.json` 动态变化，每个保留文件中的第一个 variant/case），再和
有效 metric 自动扫描。每个 benchmark 进程只请求一个 case，默认将 32 个
metric 放入同一个 CUPTI Session；CUPTI 负责该 Session 内部的 replay pass。
扫描严格串行，`sweep_rows.csv` 是断点账本，使用 `--resume` 可从中断处继续：

脚本会自动传入 `--kernel-corpus-no-latency`，PMC sweep 不执行额外的
latency event workload。若 CUPTI 拒绝一个批量配置，脚本会自动二分重试，
最终降级到单 metric，不会把整批误记为无效。

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
  --sudo --resume \
  --output-csv cuda_kernel_pmc_kernelreplay_rows.csv \
  --output-json cuda_kernel_pmc_kernelreplay.json
```

输出 CSV 每行记录一个 `(case, metric)` 的 PMU 状态、值和错误；输出 JSON
只写入有效 PMC，结构为：

```json
{
  "cuda_relu_fp32_smoke": {
    "pmc": {"sm__cycles_elapsed.avg": 123}
  }
}
```

如需扫描当前 corpus 中的全部 CUDA case，显式指定：

```bash
--case-selection all
```

延迟不从 PMC sweep 获取。独立延迟测试会显式关闭 PMU，并输出单独的
`case -> latency_us` JSON：

```bash
cmake --build build-x86-cuda --target replay_kernel_latency replay_benchmark.out -j2

REPLAY_KERNEL_LATENCY_OUTPUT=cuda_kernel_latency.json \
  ./replay_kernel_latency --filter KernelLatency.AllCudaCases
```

单 case 冒烟验证可设置 `REPLAY_KERNEL_LATENCY_CASE_FILTER`；正式测量时不要设置：

```bash
REPLAY_KERNEL_LATENCY_CASE_FILTER=cuda_relu_fp32_smoke \
  ./replay_kernel_latency --filter KernelLatency.AllCudaCases
```

没有权限或 CUPTI 启动失败会记录为 `COMMAND_FAILED`，不会伪造 PMC 值。

### CUDA kernel corpus sweep CSV

```text
case,variant,tag,sm__cycles_elapsed.avg,sm__inst_executed.avg,dram__bytes.sum,...
```

每行一个 case，每列一个 metric 值（workload 采样值）。

### OpenCL/Vulkan sweep CSV

```text
case_name,case_args,case_runs,pmu_metric_name,delta_metric
```

每行一个 (case, event, measurement) 组合，`delta_metric` 是 workload-control 增量。

### Summary CSV

```text
pmu_metric_name,valid,valid_cases
```

`valid=true` 表示至少一个 case 有非零 delta；`valid_cases` 列出这些 case（分号分隔）。

## 平台无关 PMC Interpreter

语义解释核心位于 `kernel_agent/pmc_interpreter/`。CUDA、Adreno、Mali
的差异只允许出现在 `dataset/sources/` 和 `dataset/catalog/`；投影、质量、去冗余、
相关性、kernel signature 与 implementation delta layer 不按平台分支。

统一输入 schema 为 `mnn-pmc-dataset/v1`，核心实体必须区分：

- `KernelCondition`：device + kernel 语义 + shape/workload + implementation；
- `MeasurementRun`：独立的 PMC 或 latency run，含 repeat/session/environment；
- `MetricObservation`：原生 metric、状态、值语义和 profiler pass；
- `ComparisonPair`：implementation baseline/candidate 配对；
- `FeatureSpec`：由 raw PMC 构造出的分析特征，和原生 metric 分开保存。

特别注意以下概念不能合并：

- workload-control PMC delta；
- candidate-baseline implementation delta；
- 独立 measurement repeat；
- 单次测量内部的 workload runs；
- profiler replay passes。

结构化类的固定位置为：

- `dataset/model.py`：设备、condition、run、PMC/latency 观测、配对和 `PmcDataset`；
- `dataset/reports.py`：特征、分析配置、逐层输入输出和 `PmcAnalysisReport`；
- `dataset/document.py`：`MarkdownSection` 与 `MarkdownDocument`；
- `dataset/catalog/`：kernel taxonomy 与 native metric 语义目录。

所有这些模型都继承严格 Pydantic 基类，禁止未知字段和非有限数值；模型字段禁止重新赋值，
内含容器由 layer 按只读约定消费。`frozen=True` 只是浅冻结，不是深不可变或完整审计保证。
输入语义必须在 dataset 构造阶段写入 `KernelCondition` 和 `MetricDescriptor`，不能在分析时
再传入松散的语义映射。`KernelCondition.expected_mechanisms` 是 Signature 层的显式输入；
未知时使用空 tuple，layer 不得按平台或 kernel 类别硬编码另一份机制表。

一次 Interpreter 分析必须 device-scoped：所有参与分析的 `KernelCondition.device_id` 必须
相同。多设备 bundle 要先拆分并分别运行；“平台无关”不代表可以在一次统计中合并多设备
native counter。

分析层必须严格串行，不允许跳层或重排：

```text
ProjectionOutput
  → QualityOutput
  → RedundancyOutput
  → CorrelationOutput
  → KernelSignatureOutput
  → DeltaRelationOutput
```

每个 layer 只接受上一层的具体 Pydantic 输出类型，并在输出中累积此前所有固定报告；不得以
无类型的 annotations/artifacts 传递隐式状态。

Correlation 层必须把控制后的 condition → residual feature 坐标写入固定报告，Signature
层据此计算类别 relevance 和冗余；不能只 residualize latency 后又用原始 PMC 做类别选择。

统一结构化输出 schema 为 `mnn-pmc-analysis-report/v1`，包含：

- `metric_knowledge`：语义目录、质量、用途分类、相关证据和冗余簇；
- `canonical_metric_sets.descriptive_canonical`：不使用 latency 的稳定描述候选；
- `canonical_metric_sets.latency_association`：单独的监督式探索集合；
- `kernel_signatures`：platform-neutral concept slot 到当前设备 native metric 的解析；
- `delta_rulebook`：只由合法 implementation pairs 生成的条件关系规则；
- `evidence_index` 和 `warnings`：显式标注能回答与不能回答的问题。

Delta 的最低样本门槛按不同 `semantic_equivalence_key` 的独立语义组计算，不按 pair CSV
行数计算；同一组内的多个 pair 先汇总。因果升级还要求 baseline/candidate 共享至少两个
非空 `paired_run_group_id`，且每个共享组在双方都同时连接有效 PMC 与 latency 重复。独立
`repeat_id` 只在这些共享组内统计，未配对重复不能补足门槛；所有 pair 还必须填写一致、非空
且与 feature 语义匹配的 `controlled_mechanism`。clock/temperature 的实测匹配也必须在每个
共享组内成立，不能用未配对 run 的环境值补足。

Delta 同时生成全局和 family-scoped rule：报告可用状态必须按两者并集判断。每个统计 scope
内，不同 delta 共线 fingerprint 必须分配不同 group ID；不能给整个 family 共用一个
`collinear` ID，否则选择器会把多个独立共线簇误删到只剩一条规则。

rulebook 只能在其设备、kernel 类别、贡献配对观测区间和 intervention 设计内解释；禁止
外推到其他设备、类别、未观测数值范围或不同干预机制。

`semantics/` 只允许包含结果编译和文档渲染职责：

- `semantics/compiler.py`：完整 `DeltaRelationOutput` → `PmcAnalysisReport`，不做新统计；
- `semantics/markdown.py`：`PmcAnalysisReport` → 固定章节中文 `MarkdownDocument`。

当前 CUDA long rows 可通过 `dataset/sources/mnn_replay.py` 直接构造统一数据集并分析：

```bash
python3 -m kernel_agent.pmc_interpreter \
  --pmc-csv build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv \
  --latency-json build-x86-cuda/cuda_kernel_latency.json \
  --operator-cases replay_benchmark/kernel_corpus/operator_cases.json \
  --platform cuda --backend cuda --namespace cupti \
  --normalized-output-json build-x86-cuda/cuda_pmc_dataset_v1.json \
  --output-json build-x86-cuda/cuda_pmc_analysis_report_v1.json \
  --output-md build-x86-cuda/cuda_pmc_analysis_report_v1.md
```

`--output-json` 必填并保存完整 `mnn-pmc-analysis-report/v1`；`--output-md` 可选并保存面向人的
中文解释。Markdown 不能替代结构化 JSON。

CUDA 辅助入口 `skills/gpu-pmu-sweep/scripts/analyze_cuda_pmc_semantics.py` 调用上述
统一 CLI，本身不包含分析逻辑。

Adreno/Mali 的 control-delta CSV 使用 `dataset/sources/control_delta.py` 中的
`append_control_delta_csv()`
追加到同一 normalized dataset；负 delta 是合法观测，会使用 signed-asinh
变换，不能按 CUDA raw counter 的非负约束过滤：

```python
from kernel_agent.pmc_interpreter import analyze
from kernel_agent.pmc_interpreter.dataset import load_normalized_bundle
from kernel_agent.pmc_interpreter.dataset.sources import append_control_delta_csv

dataset = load_normalized_bundle("device_conditions.json")
dataset = append_control_delta_csv(
    dataset, "records/device-pmu.csv", namespace="kgsl"  # 或 mali
)
report = analyze(dataset)
```

跨平台只能通过 `MetricDescriptor.concept_id`、`mapping_level` 和
`mapping_confidence` 对齐。不同 GPU 的 native counter 数值和效应系数默认仍是
device-scoped，不能因名称或机制相似就直接合并。

## metric 选择策略

### 按 op 类型选择

| Op 类型 | 推荐 metric | 分析目标 |
|---------|-------------|---------|
| conv_dw / matmul / gemm | `sm__cycles_elapsed.avg`, `dram__bytes.sum`, `sm__sass_thread_inst_executed_op_fadd_pred_on.sum` | 计算 vs 带宽比 |
| reduction / softmax | `sm__warps_active.avg.per_cycle_active`, `smsp__warp_issue_stalled_long_scoreboard.avg.per_warp_active` | 占用率 + 显存停顿 |
| pooling / interp | `dram__bytes.sum`, `l1tex__t_sector_hit_rate.pct` | 带宽 + L1 命中 |
| cast / raster | `dram__bytes_read.sum`, `dram__bytes_write.sum` | 纯带宽 |

### 按 memory-bound / compute-bound 选择

- **memory-bound case**（大 spatial、小 kernel）：看 `dram__throughput.avg.pct_of_peak_sustained_elapsed`（带宽利用率是否接近峰值）
- **compute-bound case**（大 kernel、高 MAC）：看 `sm__inst_executed.avg` / `sm__cycles_elapsed.avg`（IPC = inst/cycles）
- **latency-bound case**：看 `smsp__warp_issue_stalled_long_scoreboard.avg.per_warp_active`（长延迟等待占比）

## 设备信息

- **本机**：NVIDIA GeForce RTX 2080 Ti (TU102, sm75)，CUDA 13.2
- **Rhinopi**：`${RHINO_PI_USER}@${RHINO_PI_HOST}`，`$RHINO_PI_WORKSPACE/replay-benchmark`，Adreno 740
- **OrangePi**：`${ORANGE_PI_USER}@${ORANGE_PI_HOST}`，`$ORANGE_PI_WORKSPACE`，Mali-G610
- 凭据（IP/用户/密码/workspace）统一记录在仓库根 `.env`，使用前 `source ./.env`；不写入本 skill

## 验证

修改脚本后运行单元测试：

```bash
python3 -m unittest discover -s kernel_agent/tests -p 'test_*.py'
python3 -m unittest discover -s skills/gpu-pmu-sweep/tests -p 'test_*.py'
```

## 合并报告

跨后端合并 valid metric（同一设备）：

```bash
python3 skills/gpu-pmu-sweep/scripts/merge_backend_valid_metrics.py \
  --opencl-metrics records/orangepi-all-metrics.csv \
  --vulkan-metrics records/orangepi-vulkan-model-all-metrics.csv \
  --output-csv records/orangepi-opencl-vulkan-valid-metrics.csv
```

合并多轮 sweep 的详细 CSV：

```bash
python3 skills/gpu-pmu-sweep/scripts/merge_opencl_pmu_csv.py \
  --input-csv records/rhinopi-all.csv \
  --input-csv records/rhinopi-model-smoke.csv \
  --output-csv records/rhinopi-all.csv \
  --summary-csv records/rhinopi-all-metrics.csv
```

不同设备的结果**不合并**——设备身份是实验边界。
