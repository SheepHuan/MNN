---
name: kernel-adapt
description: 将 MNN/ncnn 的 OpenCL、Vulkan 或 CUDA kernel 忠实接入 replay_benchmark kernel corpus，并构建可执行 case manifest、shape/workload metadata 与 provenance。用于新增或修复 kernel、shim、OpAdapter、validator、operators.json、operator_cases.json；不负责独立正确性放行、PMC/latency 采集、数据源发布验收或 PMC Interpreter 分析。
---

# Kernel Adapt：实现与 Manifest 构建

本 Skill 只负责把“源码中的 kernel 实现”变成“可被 corpus 执行和验证的 condition”，并产出
下游可审计的 manifest。独立正确性放行交给 `corpus-audit`；PMC 数据采集交给
`gpu-pmu-sweep`，发布门禁交给 `pmc-source-gate`；平台无关语义解释交给
`pmc-interpreter`。

开始前遵守仓库 `AGENTS.md`，禁止读取或修改 `schema/private/`、`source/internal/`。

## 1. Ownership

本 Skill 拥有：

- kernel 源码提取或 bake；
- CUDA kernel 副本与 `extern "C"` launch shim；
- 每个 `(op_type, variant)` 的专用 adapter；
- 参数、buffer、grid/block、dynamic shared memory 和数据类型打包；
- host 端 expected 计算与 validator 实现；
- `operators.json` 索引更新；
- `operator_cases.json` 中 case、shape、算法 workload 和 provenance 的构建；
- 变更 case 的编译、dispatch 和基础 validation 自测。

本 Skill 不拥有：

- “实现已忠实且正确”的独立审计结论；
- GPU PMC availability、PMC rows 或 latency 的采集；
- CUDA 三数据源严格发布门禁；
- metric 语义分类、相关性、kernel signature 或 delta rulebook；
- 为满足采样数量而虚构 case、shape、FLOPs、bytes 或 provenance。

## 2. 输入、输出与阶段边界

| 项目 | 固定内容 |
| --- | --- |
| 输入源码 | 当前或历史 MNN/ncnn kernel、host launch/onExecute 逻辑、版本 tag |
| 输入语义 | op type、variant、dtype、逻辑 shape、算法参数、预期输出 |
| 主输出 | kernel/baked 文件、shim、adapter、validator、CMake 注册 |
| Manifest 输出 | `operators.json`、`operator_cases.json` |
| 元数据输出 | `shapes`、`workload`、`shape_provenance`、`workload_provenance` |
| 交接对象 | 冻结后的 manifest SHA、变更 case 列表、源码映射和自测结果 |

`operator_cases.json` 是受版本控制的权威 manifest，不是测试运行产物。下游采集一旦以某个
manifest SHA 开始，本 Skill 不得继续修改该文件；需要修改时必须宣布旧的 plan、PMC rows 和
latency 全部失效。

## 3. 入口条件

开始适配前必须明确：

1. 来源 framework、backend、tag、源文件和 kernel entry；
2. 语义 op type 与实现 variant；
3. 目标 dtype 和逻辑输入输出；
4. MNN host 侧真实参数布局、launch geometry 和资源配置；
5. validator 可独立重算的数学语义；
6. shape/workload 中哪些事实可由 adapter 契约证明。

缺少其中任一项时先补证据，不得用名称猜参数或工作量。

## 4. 强制实现原则

### 4.1 忠实性

- kernel 函数体必须忠实于来源实现，不简化算法、不替换数据布局、不删循环或 arch guard。
- 允许的变化仅限独立编译所需的 namespace、宏展开、include、binding 或 launch shim。
- 不同 tag 的函数体有实质差异时，使用独立 kernel/shim；签名相同不代表实现相同。
- variant 表示实现策略，tag 表示版本演进。禁止用普通参数把不同实现策略塞入同一 variant。
- unsupported 能力必须显式标记，不能替换成容易运行的 fallback kernel。

### 4.2 Adapter

- 每个 `(op_type, variant)` 使用专用 adapter；不靠通用签名猜测器发布新 case。
- adapter 只负责数据构造、参数转换、buffer 打包、launch 和 validator 所需事实，不包含
  kernel 算法。
- grid、block、dynamic shared memory、参数顺序、vector packing 和版本分流必须来自 host
  侧真实调度逻辑。
- fallback adapter 只能用于开发占位，不能作为新 case 的交接完成状态。
- validator 失败必须是终态；禁止由 identity、smoke 或另一 validator 把失败改写成成功。

### 4.3 Validator

- validator 必须从输入与语义独立计算 expected，再逐元素或按结构比较实际输出。
- 测试输入应覆盖正负值、边界、非整齐 shape 和非零输出，避免“全零也通过”。
- fp32、fp16、int8 等容差按数值误差确定，不能用过宽容差掩盖索引或布局错误。
- smoke validator 只适用于设备确实无法执行该路径的显式 unsupported/arch-guarded 情况，
  并必须记录限制。

## 5. 平台实现路径

### OpenCL / Vulkan

1. 从来源目录 bake 独立 kernel/shader，只补齐编译所需宏和 binding 转换。
2. 重新生成 `operators.json`。
3. 为变体实现专用 adapter：
   - OpenCL：精确构造 `ac.args` 和 compile macros；
   - Vulkan：精确构造 buffer-to-binding 映射、push constants 与 dispatch。
4. 保留设备不支持的 entry，并在运行时报告 unsupported。

常用入口：

~~~bash
python3 replay_benchmark/kernel_corpus/bake_mnn_kernels.py \
  --sources replay_benchmark/kernel_corpus/sources \
  --operators replay_benchmark/kernel_corpus/operators

python3 replay_benchmark/kernel_corpus/bake_mnn_vulkan_shaders.py \
  --sources replay_benchmark/kernel_corpus/sources \
  --operators replay_benchmark/kernel_corpus/operators

python3 replay_benchmark/kernel_corpus/bake_ncnn_shaders.py \
  --sources replay_benchmark/kernel_corpus/sources \
  --operators replay_benchmark/kernel_corpus/operators

python3 replay_benchmark/kernel_corpus/extract_operator_kernels.py \
  --root replay_benchmark/kernel_corpus \
  --output replay_benchmark/kernel_corpus/operators.json
~~~

### CUDA

1. 将忠实 kernel 副本放入
   `replay_benchmark/kernel_corpus_bridge/cuda/kernels/<op>.cu`。
2. kernel 与调用它的 `extern "C"` shim 保持在同一 CUDA translation unit。
3. adapter 根据 tag 选择正确 shim、参数布局和 launch 配置。
4. 共享 device helper 只放入 `corpus_common.cuh`；不要复制 host backend 对象。
5. 对照
   `replay_benchmark/kernel_corpus_bridge/cuda/CUDA_KERNEL_VERSIONING.md`
   检查版本和变体覆盖。

CUDA 适配不能顺带运行 PMC。基础执行自测必须显式使用 `--perf-counter-events none`。

## 6. Manifest 与 workload metadata 构建

### 6.1 Manifest 规则

每个 case name 在 backend 范围内唯一，并至少保存：

- `backend`、`framework`、`op_type`、`tag`、`variant`、`dtype`；
- adapter 所需 `int_params` / `float_params`；
- 逻辑 `shapes`，不是 allocation 或 launch shape；
- 可证明的算法 `workload`；
- shape/workload 的字段级 provenance；
- validator 标识。

`variant`、`tag` 和 `workload_runs` 不得制造新的 workload condition。不能因为需要更多
condition 就复制同一语义 case 改名。

### 6.2 元数据 enrichment

先阅读：

- `docs/superpowers/specs/2026-07-31-pmc-interpreter-cuda-data-sources.md`
- `replay_benchmark/kernel_corpus/enrich_pmc_workload_metadata.py::SUPPORTED_OP_TYPES`

只对脚本明确支持且能由 adapter 契约证明的 variant 写入理论事实：

~~~bash
python3 replay_benchmark/kernel_corpus/enrich_pmc_workload_metadata.py \
  --input replay_benchmark/kernel_corpus/operator_cases.json \
  --in-place

python3 -m unittest discover \
  -s replay_benchmark/kernel_corpus/tests \
  -p 'test_pmc_workload_metadata.py'
~~~

强制语义：

- `algorithmic_flops`、`algorithmic_bytes`、`output_elements` 是算法量；
- allocation bytes、grid/block、寄存器、shared memory 和设备实测值不能写入算法 workload；
- 不适用使用 provenance `not_applicable`，未知保持缺失，二者都不能伪造为 0；
- enrichment 必须字段级保留人工 metadata、事务式失败并可重复执行；
- 不能在 Skill 中写死 case 数、manifest SHA、plan ID、metric 数或最终 rows 数。

## 7. 自测顺序

每一步通过后再继续：

1. 运行相关 Python manifest/adapter tests；
2. 构建对应 replay target；
3. 只运行变更 case，显式关闭 PMU；
4. 检查 compile、dispatch、`valid=true` 和
   `validation_status=validation_passed`；
5. 运行非整齐 shape、边界值和至少一个非 smoke case；
6. 连续运行 enrichment 两次，确认第二次不改变字节内容；
7. 记录 manifest SHA 和变更 case 清单。

示例 CUDA 自测：

~~~bash
cmake --build build-x86-cuda --target replay_benchmark.out -j2

LD_LIBRARY_PATH=build-x86-cuda:build-x86-cuda/source/backend/cuda:. \
  build-x86-cuda/replay_benchmark.out \
  --kernel-corpus-bench \
  --kernel-corpus-root replay_benchmark/kernel_corpus \
  --kernel-corpus-case <changed-case> \
  --kernel-corpus-runs 1 \
  --perf-counter-events none \
  --perf-counter-output /tmp/kernel-adapt-smoke.json
~~~

自测证明“新实现可执行”，不等于独立正确性放行。

## 8. 退出条件与交接

只有同时满足以下条件才可交给 `corpus-audit`：

- 来源映射、版本分流和变体身份明确；
- 新 case 使用专用 adapter，无成功 fallback；
- changed cases 编译、dispatch、real validation 通过；
- manifest schema、case name、shape/workload/provenance 测试通过；
- enrichment 幂等；
- manifest 已冻结并计算内容 SHA。

交接记录必须包含：

~~~text
manifest_path
manifest_sha256
changed_case_names
source_file/tag/entry -> corpus_file/shim/adapter 映射
validator 类型与容差
已运行的命令及结果
unsupported/arch-guarded 限制
~~~

后续流转固定为：

~~~text
kernel-adapt
  -> corpus-audit
  -> gpu-pmu-sweep
  -> pmc-source-gate
  -> pmc-interpreter
~~~

任何下游阶段发现实现或 metadata 问题，都回到本 Skill 修复；修复后生成新的 manifest SHA，
旧采集产物不得 resume 或发布。
