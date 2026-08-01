---
name: gpu-pmu-sweep
description: 在 NVIDIA、Adreno 或 Mali GPU 上发现可用 metric，并采集 replay_benchmark kernel 的 PMC、CaseSelectionPlan、完整状态账本、独立 latency、launch/resource 和 environment 原始事实。用于刷新设备数据、验证 collector 数值链路或安全断点续跑；不负责 kernel 正确性、原始数据发布门禁或 PMC 统计解释。
---

# GPU PMU Sweep：设备发现与原始数据采集

本 Skill 只负责产生测量事实，并把不可变采集产物交给独立发布门禁：

~~~text
kernel-adapt
  -> corpus-audit
  -> gpu-pmu-sweep
  -> pmc-source-gate
  -> pmc-interpreter
~~~

采集完成不等于数据可发布。不要在本 Skill 中运行统计解释，也不要因为 rows 数量看似完整就
自行宣称数据有效。

开始前必须完整阅读：

- `docs/superpowers/specs/2026-07-31-pmc-interpreter-cuda-data-sources.md`

## 1. Ownership 与禁止跨界

本 Skill 拥有：

- 当前设备 metric discovery/availability；
- PMC collector 数值状态链路验证；
- case selection plan 与 PMC long rows 采集；
- 关闭 PMU 后的独立 latency/launch/resource/environment 采集；
- Adreno/Mali control-delta sweep 的原始数据保存；
- 采集脚本的 session、顺序、resume ledger 和 raw 日志维护。

本 Skill 不拥有：

- kernel、shim、adapter、validator 或 `operator_cases.json` 的实现与修改；
- kernel 正确性放行；
- 为满足 target/K 而增加、复制或重命名 case；
- CUDA 三数据源严格发布门禁；
- 把 CUDA NVPW double 转成整数；
- 用历史固定 metric 数、case 数或 rows 数作为验收目标；
- 构造 `mnn-pmc-dataset/v1`、统计相关性、signature 或 delta rulebook。

实现或 manifest 有问题时回到 `kernel-adapt`，正确性未放行时回到 `corpus-audit`。采集完成后
交给 `pmc-source-gate`，门禁通过后再交给 `pmc-interpreter`。

## 2. 固定输入与输出

### CUDA 采集输入与输出

| 数据 | 固定路径 | 本 Skill 中的角色 |
| --- | --- | --- |
| Manifest | `replay_benchmark/kernel_corpus/operator_cases.json` | 冻结的采集输入 |
| PMC rows | `build-x86-cuda/cuda_kernel_pmc_kernelreplay_rows.csv` | plan-selected case × metric 完整状态账本 |
| Latency | `build-x86-cuda/cuda_kernel_latency.json` | 全部 CUDA case 的独立 latency 与 launch/environment 输出 |

PMC rows 必须同时产生并保留：

- `cuda_kernel_pmc_kernelreplay_rows.csv.selection.json`；
- 本轮 `cuda_pmc_valid.csv`；
- availability 原始日志与 `cuda_pmc_all.csv`；
- metric smoke、设备/driver/profiler、collector binary 和 sweep 配置证据。

本 Skill 不产生 `cuda_pmc_source_validation.json`、normalized dataset 或语义报告。

## 3. Collection 入口条件

开始正式采集前必须获得 `corpus-audit` 放行记录：

~~~text
audit_status=passed
manifest_sha256
audited_case_names
unsupported cases and reasons
test evidence
~~~

同时满足：

1. manifest 已冻结；
2. collector 源码和 binary 版本已确定；
3. GPU、driver、CUDA/CUPTI 或移动 GPU 驱动身份已记录；
4. 没有 availability、latency 或 PMC sweep 进程运行；
5. 使用新的 staging 路径，不覆盖或 resume 已作废数据；
6. 整轮采集期间不重建 binary、不修改 manifest、不切换设备或 metric 顺序。

availability、latency 和 PMC rows 必须严格串行，不能并发竞争 profiler 或环境。

## 4. 实验身份与 resume

一次 CUDA sweep 身份至少包含：

~~~text
manifest bytes
selection policy/version、target op types、K
ordered valid metrics
collector binary/code
runs、metrics-per-session、timeout
device fingerprint、driver、CUDA、CUPTI
~~~

任一项变化后：

- 创建新的 selection plan；
- 从不存在的 rows CSV 开始；
- 首次运行禁止 `--resume`；
- 旧 normalized dataset/report 只能作为历史探索数据。

另外：

- manifest、kernel/runner binary 或被测设备变化时，重新采 latency；
- collector、metric discovery 条件、driver/CUDA/CUPTI 或设备变化时，重新跑 availability；
- 只改变 target/K、runs、session 大小或 timeout 时，也必须重采 rows，但不把无关旧文件自动
  宣称为同一轮实验；最终仍由 strict gate 校验所有身份。

只有旧 rows 恰好是当前确定性笛卡尔积的合法前缀，且 plan ID、sweep config ID、表头、metric
顺序、binary 和设备身份全部一致时，才允许显式 `--resume`。resume 通过不等于正式发布通过。

禁止恢复旧 `op-type` 选择别名；正式默认策略是 `condition-balanced`，`all` 只用于调用方
明确要求的全 case sweep。

## 5. CUDA Collection

### 5.1 构建

~~~bash
cmake -S . -B build-x86-cuda \
  -DMNN_BUILD_BENCHMARK=ON \
  -DMNN_CUDA=ON \
  -DMNN_REPLAY_ENABLE_PERFCOUNTER=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-x86-cuda \
  --target replay_benchmark.out list_cuda_metrics \
           replay_pmu_availability replay_kernel_latency \
  -j2
~~~

构建完成后冻结 binary；扫描期间不得再次构建。

### 5.2 数值链路冒烟

CUDA NVPW metric 的权威数值类型是 `double`。链路必须保持：

~~~text
NVPW double
  -> collector report double
  -> benchmark JSON number
  -> Python float
  -> CSV decimal
  -> Pydantic finite float
~~~

禁止经过 `uint64_t`、`static_cast<uint64_t>` 或 `int(raw)`。Adreno/Mali 原生 uint64 counter
则保持 JSON integer，二者不能用同一强制转换策略。

collector 或 ledger 变更后，在 full availability 前完成两类单 metric 冒烟：

1. 选择当前设备上确实产生小数的 avg/ratio/throughput metric，确认 report、JSON 和 CSV
   保留小数，而不是只验证状态为 VALID；
2. 选择历史 overflow/non-finite 或本轮发生状态迁移的 metric，确认非有限值不会进入 VALID，
   `OVERFLOW -> VALID` 时确认新有限值可端到端保存。

示例命令模板：

~~~bash
cd build-x86-cuda

sudo env LD_LIBRARY_PATH=.:source/backend/cuda:. \
  ./replay_benchmark.out \
  --kernel-corpus-bench \
  --kernel-corpus-root ../replay_benchmark/kernel_corpus \
  --kernel-corpus-case <audited-representative-case> \
  --kernel-corpus-runs 1 \
  --kernel-corpus-no-latency \
  --perf-counter-events <fractional-or-migration-metric> \
  --perf-counter-output <new-staging-dir>/metric-smoke.json
~~~

不要把某个历史 metric 名或历史 status 当作所有设备的固定断言；以本轮 discovery 与原始报告
为准。

### 5.3 Availability

每轮使用新日志和新 CSV：

~~~bash
cd build-x86-cuda
set -o pipefail

sudo env LD_LIBRARY_PATH=.:source/backend/cuda:. \
  ./replay_pmu_availability --filter CudaMetric.AllMetrics \
  2>&1 | tee <new-staging-dir>/sweep.log

python3 ../skills/gpu-pmu-sweep/scripts/parse_cuda_pmu_log.py \
  --input <new-staging-dir>/sweep.log \
  --all-csv <new-staging-dir>/cuda_pmc_all.csv \
  --valid-csv <new-staging-dir>/cuda_pmc_valid.csv
~~~

解析器必须拒绝中断、缺号、重复、汇总不一致、没有 OK/PASSED 结束标记的日志。有效 metric 数
由本轮 `cuda_pmc_valid.csv` 动态决定，不得写死。`cuda_pmc_all.csv` 可保留重复的非 VALID
请求名；`cuda_pmc_valid.csv` 中 VALID metric 必须唯一、有序且数值状态合法。

### 5.4 CaseSelectionPlan 与 PMC rows

P0 固定 target 为 `matmul`、`conv_dw`、`reduction`、`softmax`、`transpose`、
`maxpool`、`avgpool`，每类请求 K=5 个 distinct condition。最终 selected case 数只能由
当前 manifest 和 plan 产生。

首次正式采集：

~~~bash
cd build-x86-cuda
sudo -v

python3 ../skills/gpu-pmu-sweep/scripts/sweep_cuda_kernel_pmc.py \
  --valid-csv <new-staging-dir>/cuda_pmc_valid.csv \
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
  --selection-plan-output <new-staging-dir>/cuda_kernel_pmc_kernelreplay_rows.csv.selection.json \
  --device-fingerprint '<GPU|driver|CUDA|CUPTI>' \
  --metrics-per-session 32 \
  --runs 1 \
  --sudo \
  --output-csv <new-staging-dir>/cuda_kernel_pmc_kernelreplay_rows.csv \
  --output-json <new-staging-dir>/cuda_kernel_pmc_kernelreplay.json
~~~

语义约束：

- 每行是一个 `(case, metric)`；
- 一个 benchmark/CUPTI session 有唯一 `collection_session_id`；
- 同 session metric 共享 `order_index` 和环境事实；
- `num_passes` 是 replay pass，不是 repeat；
- `--runs` 是 measurement 内 launch 数，不是 repeat；
- compact JSON 不能替代完整 rows ledger；
- 完整 rows 数动态等于 plan-selected case 数 × 本轮 ordered valid metric 数。

合法续跑必须复用保存的 plan：

~~~bash
python3 ../skills/gpu-pmu-sweep/scripts/sweep_cuda_kernel_pmc.py \
  --valid-csv <same-staging-dir>/cuda_pmc_valid.csv \
  --corpus-root ../replay_benchmark/kernel_corpus \
  --binary ./replay_benchmark.out \
  --workdir . \
  --lib-dir .:source/backend/cuda:. \
  --selection-plan-input <same-staging-dir>/cuda_kernel_pmc_kernelreplay_rows.csv.selection.json \
  --selection-plan-output <same-staging-dir>/cuda_kernel_pmc_kernelreplay_rows.csv.selection.json \
  --device-fingerprint '<与首次完全相同>' \
  --metrics-per-session 32 \
  --runs 1 \
  --sudo --resume \
  --output-csv <same-staging-dir>/cuda_kernel_pmc_kernelreplay_rows.csv \
  --output-json <same-staging-dir>/cuda_kernel_pmc_kernelreplay.json
~~~

### 5.5 独立 latency

latency 必须关闭 PMU，并覆盖当前 manifest 的全部 CUDA case。正式运行不得设置
`REPLAY_KERNEL_LATENCY_CASE_FILTER`：

~~~bash
cd build-x86-cuda

env -u REPLAY_KERNEL_LATENCY_CASE_FILTER \
  REPLAY_KERNEL_CORPUS_ROOT=../replay_benchmark/kernel_corpus \
  REPLAY_KERNEL_LATENCY_OUTPUT=<new-staging-dir>/cuda_kernel_latency.json \
  LD_LIBRARY_PATH=.:source/backend/cuda:. \
  ./replay_kernel_latency --filter KernelLatency.AllCudaCases
~~~

latency producer 只能接受中间 benchmark report 中 `valid=true` 且
`validation_status=validation_passed` 的 case；正式 latency 记录必须有正数有限 latency、
真实 launch records 和 environment。最终 schema 不需要重复保存这两个中间状态字段。
launch stage、workload runs 或 profiler passes 都不是独立 repeat。latency run 与 PMC
session 永远分开。

## 6. 交给 PMC Source Gate

availability、latency 和 PMC rows 全部结束后，先确认没有进程继续写 staging，再把以下原始
产物交给 `pmc-source-gate`：

~~~text
corpus-audit 放行记录
operator_cases.json + manifest_sha256
availability 原始日志
cuda_pmc_all.csv
cuda_pmc_valid.csv
cuda_kernel_pmc_kernelreplay_rows.csv.selection.json + plan_id
cuda_kernel_pmc_kernelreplay_rows.csv
cuda_kernel_latency.json
device/driver/CUDA/CUPTI fingerprint
collector binary/code identity
runs、metrics-per-session、timeout 和 sweep_config_id
fractional/overflow smoke evidence
~~~

本 Skill 不生成 `cuda_pmc_source_validation.json`，也不把采集目录复制到正式固定路径。只有
`pmc-source-gate` 输出 `valid=true` 后，数据才能交给 `pmc-interpreter`。

## 7. Adreno / Mali Collection

OpenCL/Vulkan 使用单 event、独立 measurement、保留 raw JSON 的扫描：

~~~bash
python3 skills/gpu-pmu-sweep/scripts/sweep_opencl_pmu.py \
  --device <device-name> \
  --events all \
  --measurements 3 \
  --workload-runs 5 \
  --output-csv <new-output>.csv \
  --keep-json-dir <new-raw-json-dir>
~~~

必须区分：

- `measurements`：独立进程测量；
- `workload-runs`：一次测量内 dispatch 数；
- `workload_minus_control`：workload 与 empty control 的差；
- implementation candidate-baseline delta：另一个实验设计。

control delta 可以为负，使用 signed-asinh 等保留符号变换；不能套用 CUDA absolute counter
的非负过滤。不同设备的结果不直接合并。

## 8. 退出条件与交接

Collection 阶段退出产物：

~~~text
manifest_sha256
device/driver/profiler fingerprint
collector binary identity
availability all/valid ledger
selection plan
complete PMC rows
independent latency JSON
raw logs and smoke evidence
~~~

采集退出只表示原始账本已冻结。若 manifest、collector、metric 清单、设备或采样策略发生变化，
旧 staging 立即失效，不能跨身份 resume，也不能交给 Source Gate 冒充同一轮实验。

## 9. 静态与单元验证

修改采集脚本后运行：

~~~bash
git diff --check

.venv/bin/python -m unittest discover \
  -s skills/gpu-pmu-sweep/tests \
  -p 'test_*.py'

.venv/bin/python -m unittest discover \
  -s replay_benchmark/kernel_corpus/tests \
  -p 'test_pmc_workload_metadata.py'
~~~

这些测试不替代真实设备的 fractional/overflow smoke、availability、latency 和 rows。发布门禁
及其测试属于 `pmc-source-gate`。
