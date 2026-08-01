---
name: corpus-audit
description: 独立审计 replay_benchmark kernel corpus 的源码忠实性、adapter/launch 正确性、validator 有效性以及 operator_cases.json 的 shape/workload/provenance 一致性。用于新增或修复 kernel 后的放行、validator fallback 排查、多 shape/版本审计和 PMC 采集前冻结 manifest；不负责实现新 kernel、构造采样计划、采集 PMC/latency 或解释性能计数器。
---

# Corpus Audit：Kernel 正确性放行

本 Skill 是 `kernel-adapt` 与设备采集之间的独立门禁。它回答：

> 当前 manifest 中声明的 case 是否忠实执行了来源 kernel，并且 validator 真正验证了正确输出？

默认以只读方式审计。发现缺陷后回交 `kernel-adapt` 修复；若用户明确要求同时修复，也必须
按 `kernel-adapt` 的实现流程完成后，从头重新审计。

开始前遵守仓库 `AGENTS.md`，禁止读取或修改 `schema/private/`、`source/internal/`。

## 1. Ownership

本 Skill 拥有：

- kernel body、签名、模板实例化和版本分流审计；
- adapter 参数、buffer layout、grid/block/shared memory 审计；
- validator 真实性、失败传播和 fallback 隔离审计；
- 输入值域、输出覆盖和多 shape/边界 case 审计；
- manifest shape/workload/provenance 与 adapter 契约的一致性审计；
- 审计通过后的 manifest 身份冻结与下游放行记录。

本 Skill 不拥有：

- 新 kernel、shim、adapter、case 或 workload metadata 的设计与实现；
- 为满足每类 K 个 condition 而添加或复制 case；
- `CaseSelectionPlan` 的生成；
- PMC availability、PMC rows 或 latency 采集；
- CUDA 三数据源发布门禁；
- metric 相关性、signature 或 delta 规则。

## 2. 输入、输出与状态

| 项目 | 内容 |
| --- | --- |
| 输入 | 冻结候选 `operator_cases.json`、变更 case 列表、来源映射、构建产物 |
| 审计范围 | 指定变更 cohort；P0 发布时必须覆盖所有目标 case |
| 输出 | pass/fail/unsupported 逐 case 结论、发现项、测试证据 |
| 放行身份 | manifest SHA256、case 集合、审计时 binary/config 身份 |
| 下游 | `gpu-pmu-sweep` 的采集前置交接 |

状态只允许：

- `pass`：忠实性、执行和 real validation 均通过；
- `unsupported`：来源能力在当前设备不可执行，原因明确且未 fallback；
- `fail`：任一证据不满足。

“能 launch”“输出非零”“smoke validator 通过”都不能单独成为 `pass`。

## 3. 入口条件

开始审计前必须收到 `kernel-adapt` 的交接：

~~~text
manifest_path
manifest_sha256
changed_case_names
source -> corpus/shim/adapter 映射
validator 类型与容差
已运行的基础自测
已知 unsupported 限制
~~~

先重新计算 manifest SHA。若与交接值不同，停止审计并要求重新冻结；不得猜测变化是否“无关”。

~~~bash
sha256sum replay_benchmark/kernel_corpus/operator_cases.json
~~~

确认没有 availability、latency 或 PMC sweep 正在使用该 manifest。审计期间允许重建和运行
`--perf-counter-events none` 的 correctness case，但不启动 profiler 采集。

## 4. 审计顺序

按以下顺序执行；前一步失败时先记录并停止该 case 的放行。

### 4.1 来源忠实性

逐 kernel 核对：

1. 来源文件、tag、entry 和 corpus 文件映射正确；
2. 函数签名、参数顺序、const、模板类型和 arch guard 一致；
3. 函数体未简化、未替换算法、未改变循环、索引或 shared-memory 布局；
4. host 编译所需改写只涉及 namespace、include、宏/binding 和 shim；
5. 每个实质函数体版本有正确的独立 shim；
6. variant 表示实现策略，tag 表示版本，二者未混用。

CUDA 版本审计同时参考：

`replay_benchmark/kernel_corpus_bridge/cuda/CUDA_KERNEL_VERSIONING.md`

审计说明必须记录“相同”“允许的适配差异”或“实质差异”，不能只写“看起来一致”。

### 4.2 Adapter 与 launch

对照来源 host `onResize/onExecute` 或 launch 点核对：

- 参数顺序、标量类型、vector packing 和 buffer 读写角色；
- logical shape、allocation shape 和布局转换；
- grid X/Y/Z 与 block X/Y/Z；
- 非方阵、非整除、最小 shape 和边界线程；
- dynamic shared memory、静态 shared memory 和版本相关 block 策略；
- 一个 case 触发多个 device-kernel 时的 stage 顺序和最终输出；
- CUDA shim 与 kernel 位于同一 translation unit。

不得从 kernel 名称或经验公式替代 host 侧证据。典型高风险项包括 transpose 的 X/Y 交换、
reduction 的参数顺序、packed channel 尾部、shared-memory padding 和多个版本共用错误 shim。

### 4.3 Validator 与失败传播

每个可执行 case 必须使用 real validator：

1. expected 由输入和算子语义独立计算；
2. 实际输出按真实 layout 解码；
3. 比较覆盖全部逻辑输出，而非只检查首元素或“任一非零”；
4. 容差与 dtype 和累积误差相符；
5. validator 返回失败后，最终报告必须保持失败；
6. identity、smoke 或 fallback validator 不得救回失败；
7. benchmark/latency harness 只能接受
   `valid=true` 且 `validation_status=validation_passed`。

至少增加或运行一个负向测试：人为提供错误输出/expected 后，case 必须变为
`validation_failed`。优先使用测试 fixture 或最小测试 patch；不得只审阅成功路径。

### 4.4 输入值域与输出覆盖

测试数据必须能够暴露错误：

- 有正有负，并覆盖 clamp、cast、量化边界；
- 不因过小值域导致 cast/quantize 全零；
- reduction min/prod 不因固定 0 输入失去区分力；
- 输出包含多个不同值；
- 非整齐 shape 覆盖尾部和边界；
- 明确区分语义本应全零、arch-guarded 空 kernel 与意外全零。

意外全零、只有首元素正确或只有一个 channel 被写入，一律 `fail`。

### 4.5 Manifest 与 workload metadata

先阅读：

- `docs/superpowers/specs/2026-07-31-pmc-interpreter-cuda-data-sources.md`
- `replay_benchmark/kernel_corpus/enrich_pmc_workload_metadata.py::SUPPORTED_OP_TYPES`

逐 case 核对：

- case name 唯一，`op_type/tag/variant/dtype` 与 adapter 一致；
- `shapes` 是逻辑 shape；
- `algorithmic_flops`、`algorithmic_bytes`、`output_elements` 由算法语义证明；
- allocation bytes、grid/block、寄存器和设备实测值没有混入 workload；
- provenance 的 source、method、formula、confidence/status 与事实一致；
- `not_applicable` 与缺失分开，均未伪造成 0；
- variant、tag 和 workload runs 没有被计为不同 workload condition。

以临时文件做幂等和所有权审计，不要在审计阶段重写候选 manifest：

~~~bash
audit_dir=$(mktemp -d)

python3 replay_benchmark/kernel_corpus/enrich_pmc_workload_metadata.py \
  --input replay_benchmark/kernel_corpus/operator_cases.json \
  --output "$audit_dir/once.json"

python3 replay_benchmark/kernel_corpus/enrich_pmc_workload_metadata.py \
  --input "$audit_dir/once.json" \
  --output "$audit_dir/twice.json"

cmp replay_benchmark/kernel_corpus/operator_cases.json "$audit_dir/once.json"
cmp "$audit_dir/once.json" "$audit_dir/twice.json"
~~~

任一 `cmp` 失败都必须解释；不得在 audit 中直接 `--in-place` 掩盖差异。

### 4.6 多 condition 覆盖

覆盖是正确性问题，不是“case 数越多越好”：

- 每个变体至少有满足其约束的有效 condition；
- 关键算子覆盖小/中/大、整齐/非整齐、边界参数；
- 不同 variant 可在相同语义输入上比较输出一致性；
- 不满足专用 kernel 约束的 shape 不得硬塞给该 variant；
- case 数、CUDA case 数和每类 distinct condition 数全部从当前 manifest 动态计算。

不得在 Skill 中保存历史固定数量或通过复制 case 满足下游 K。

## 5. 执行验证

先运行静态和 Python 测试：

~~~bash
python3 -m unittest discover \
  -s replay_benchmark/kernel_corpus/tests \
  -p 'test_*.py'

git diff --check
~~~

CUDA correctness 构建与单 case 运行示例：

~~~bash
cmake --build build-x86-cuda --target replay_benchmark.out -j2

LD_LIBRARY_PATH=build-x86-cuda:build-x86-cuda/source/backend/cuda:. \
  build-x86-cuda/replay_benchmark.out \
  --kernel-corpus-bench \
  --kernel-corpus-root replay_benchmark/kernel_corpus \
  --kernel-corpus-case <audited-case> \
  --kernel-corpus-runs 1 \
  --perf-counter-events none \
  --perf-counter-output /tmp/corpus-audit-case.json
~~~

检查实际 JSON 字段，不以进程返回 0 代替：

~~~text
compile_status
dispatch_status
valid
validation_status
error
~~~

若修改了 runner、adapter 或 validator，按 `test-ci` 选择相应回归。若定位 bug，使用
`general-debug`；修复实现时回到 `kernel-adapt`。

## 6. 退出条件

只有同时满足以下条件才可放行：

- manifest SHA 在整个审计期间未变化；
- 审计范围内每个 case 都有 pass/unsupported/fail 明确结论；
- 所有 pass case 来源忠实、launch 正确、real validator 有效；
- validator 失败不能被 fallback 改写；
- shape/workload/provenance 与 adapter 契约一致且 enrichment 幂等；
- 所有相关测试通过；
- known unsupported 和剩余风险已记录。

任一 fail 存在时，不得启动正式 PMC 或 latency 采集。

## 7. 交接产物

交给 `gpu-pmu-sweep` 的记录必须包含：

~~~text
audit_status=passed
manifest_path
manifest_sha256
audited_case_names
unsupported_case_names_and_reasons
source_fidelity_summary
validator_negative_test_summary
workload_metadata_summary
test_commands_and_results
~~~

可以把长期有效结论更新到现有 corpus `HANDOFF.md`，但不要写入瞬时 case 数、metric 数、
plan ID、rows 数或设备 availability 结果；这些属于后续具体实验。

manifest 交接后必须冻结。任何 kernel、adapter、validator、case 或 metadata 修改都会使本次
审计身份和全部下游采集失效，必须重新执行本 Skill。

固定后续顺序是
`corpus-audit -> gpu-pmu-sweep -> pmc-source-gate -> pmc-interpreter`。本 Skill 不能把仅通过
correctness audit 的 manifest 直接交给 Interpreter；中间仍必须完成真实设备采集和 source gate。
