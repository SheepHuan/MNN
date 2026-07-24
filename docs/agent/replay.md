# MNN Execution Record and Replay

## 1. 目标

本文定义 MNN 计算图执行记录（Execution Record）的格式、采集方式和单算子回放方式，覆盖 CPU 和 OpenCL。

记录由 benchmark 生成，由外部单算子测试消费，用于复现静态计算图中实际执行过的每个算子，包括：

- 算子的精确 FlatBuffer 参数；
- 算子执行时的输入和输出 Tensor；
- 实际使用的 Backend 配置；
- 具体的 `Execution` 类和 variant；
- 一个 OpenCL `Execution` 内部实际 dispatch 的 kernel 信息；CPU 记录该字段为空；
- 必要时用于 raw kernel 调试的 scalar 参数和 Tensor 引用。

记录的核心目标是：外部测试不需要重新推断算子参数、Tensor layout 或 Execution 分派条件，即可复现 benchmark 中的单个算子执行。

### 1.1 代码组织

`replay_benchmark` 按职责拆分为以下模块：

```text
replay_benchmark.cpp       CLI、benchmark 兼容流程、record 采集编排
ReplayRecord.hpp/.cpp      record 数据结构、JSON/FlatBuffer、Tensor 快照和比较
ReplayRunner.hpp/.cpp      指定 op 的 Runtime / Backend / Execution 回放
```

CPU 和 OpenCL 共用 record 与 Tensor 逻辑；Backend 差异集中在 `ReplayRunner`，避免主程序继续堆叠后端细节。

## 2. 测试粒度

主要测试对象是一个具体的 `Execution` 类及其 variant，而不是单独的 `.cl` 函数。

一个测试单元由以下信息共同决定：

```text
OpType
+ concrete Execution class
+ Execution variant
+ memory object
+ precision
+ device feature
+ shape / operator attributes
```

例如，同一个 `OpType_Convolution` 可能实际选择：

```text
ConvBufExecution
ConvBufLowMemoryExecution
ConvBufWinograd
ConvSubgroupBuf
```

同一个 Execution 类内部也可能选择不同 kernel、build options 或 work size。因此记录同时保存 `execution` 和 `variant`。

## 3. 采集时机

静态计算图的执行过程可以分成两部分：

```text
Session resize / Pipeline allocation
    - 创建 Execution
    - 选择 Execution variant
    - 计算 Tensor shape
    - onResize / onEncode
    - 创建 kernel

Session run
    - 产生实际输入 Tensor
    - 执行 Execution
    - 产生输出 Tensor
```

Execution 和 variant 信息应在 Session resize 阶段记录，Tensor 数据应在实际执行阶段记录。

当前 MNN 已有逐算子回调能力：

- `Interpreter::runSessionWithCallBackInfo()` 提供算子执行前后的回调；
- `Pipeline::executeCallBack()` 按实际 `Command` 执行并触发回调；
- `Command` 保存 `Op`、`Execution`、输入 Tensor 和输出 Tensor。

但是，公开的 `OperatorInfo` 只包含算子名称、类型和 FLOPs。要获得具体 Execution 类、variant 和精确 Op 参数，需要增加 Backend execution record 的内部采集器。

采集器必须是 Session/Pipeline 级别的对象，不使用命名空间作用域的动态初始化全局对象。

## 4. 记录目录

单模型记录目录结构如下：

```text
record/
├── record.json
├── ops/
│   ├── 000012.op.fb
│   └── 000013.op.fb
└── tensors/
    ├── op_000012_input_0.logical.bin
    ├── op_000012_input_0.storage.bin
    ├── op_000012_output_0.logical.bin
    └── op_000012_output_0.storage.bin
```

当输入是模型目录时，工具会为每个模型建立一个子目录：

```text
records/
└── model.mnn/
    ├── record.json
    ├── ops/
    └── tensors/
```

回放时传入 `--record records` 和 `--model model.mnn` 即可自动定位该子目录。

大规模 Tensor 数据不直接写入 JSON。JSON 只保存元数据、文件路径、字节数和校验和。

## 5. record.json

### 5.1 顶层结构

```json
{
  "format": "mnn-execution-record",
  "version": 1,
  "model": {
    "path": "model.mnn",
    "sha256": "...",
    "mnn_version": "3.6.1"
  },
  "runtime": {
    "backend": "CPU",
    "forward": 0,
    "gpu_mode": 4,
    "precision": 2,
    "memory": 0
  },
  "capture": {
    "iteration": 0,
    "dump_logical_tensor": true,
    "dump_storage_tensor": true,
    "dump_op_flatbuffer": true,
    "dump_dispatch": true
  },
  "ops": [],
  "tensors": []
}
```

`format` 和 `version` 是必填字段。当前回放器只接受 `mnn-execution-record`、版本 `1`；其他格式或版本会直接拒绝，不维护旧的 OpenCL 专用 record 兼容路径。

### 5.2 Op 记录

```json
{
  "op_id": 12,
  "origin_op_id": 12,
  "command_id": 0,
  "pipeline_id": 0,
  "name": "conv_3",
  "op_type": "Convolution",
  "op_type_id": 42,
  "op_file": "ops/000012.op.fb",

  "backend": "CPU",
  "precision": 2,
  "execution": "ConvBufExecution",
  "variant": "conv_1x1_local",

  "inputs": [
    {
      "id": "op_000012_input_0",
      "logical_file": "tensors/op_000012_input_0.logical.bin"
    }
  ],
  "outputs": [
    {
      "id": "op_000012_output_0",
      "logical_file": "tensors/op_000012_output_0.logical.bin"
    }
  ],

  "kernels": []
}
```

字段含义：

- `op_id`：记录中的唯一 Op ID；
- `origin_op_id`：原始静态图 Op ID；
- `command_id`：经过 geometry、copy、wrap 等处理后的实际 Command ID；
- `execution`：实际创建的具体 Execution 类名；
- `variant`：该 Execution 根据 shape、属性和设备能力选择的实现路径；
- `inputs` / `outputs`：指向 Tensor 快照 ID；
- `kernels`：OpenCL Execution 内部的 kernel dispatch 列表；CPU record 通常为空。

一个原始 Op 可能对应多个实际 Command，必须通过 `origin_op_id` 和 `command_id` 保留这种关系。

### 5.3 Op 参数文件

`op_file` 保存一个独立的 MNN `Op` FlatBuffer，例如：

```text
ops/000012.op.fb
```

不能直接保存原始模型中的 `Op*` 内存，因为其中的字符串、向量和子表依赖原始 FlatBuffer 的偏移地址。

实现时应将原始 Op `UnPack` 到 `OpT`，再重新 `Pack` 成独立 FlatBuffer。外部测试读取该文件后，可以直接通过 `flatbuffers::GetRoot<MNN::Op>()` 得到可执行的 Op。

JSON 中可以保存一份可读的算子参数摘要，但 `op.fb` 是回放的权威数据。

当前 `replay_benchmark` V1 实际写入的核心字段是 `op_id`、`name`、`op_type`、`op_file`、`backend`、`execution`、`variant`、`inputs`、`outputs` 和 `kernels`；`origin_op_id`、`command_id`、`pipeline_id`、设备信息和 checksum 是后续扩展字段。

## 6. Tensor 数据

### 6.1 Tensor 元数据

```json
{
  "id": "op_000012_input_0",
  "role": "work_input",
  "dtype": "float32",
  "logical_shape": [1, 64, 56, 56],
  "dimension_type": "CAFFE",
  "dimension_format": "NC4HW4",
  "channel_pack": 4,
  "memory_object": "BUFFER",
  "pads": {
    "left": 0,
    "right": 0,
    "top": 0,
    "bottom": 0
  },
  "quant": null,
  "logical_file": "tensors/op_000012_input_0.logical.bin",
  "storage_file": "tensors/op_000012_input_0.storage.bin",
  "logical_bytes": 802816,
  "storage_bytes": 802816,
  "sha256": "..."
}
```

必须同时保存逻辑数据和实际存储数据：

### logical data

用于外部 Execution 级测试。数据按照逻辑 Tensor layout 保存，外部测试可以通过 MNN 的 Tensor copy/convert 逻辑重新上传到目标 Backend。

### storage data

用于精确调试 OpenCL kernel。数据按照被测 Execution 实际使用的 layout 保存，包括：

- `NC4HW4` / `NC16HW16`；
- padding channel；
- Tensor pads；
- FP16、FP32 或整型实际存储；
- Buffer 或 Image 的存储信息；
- 临时 Tensor 和预处理后的权重 Tensor。

对于 Image，还需要额外保存 image width、height、channel type 和 row pitch 等信息。

外部测试优先使用 logical data 进行 Execution 回放；storage data 主要用于 raw kernel 或 layout 问题定位。

### 6.2 数据采集规则

- 输入数据在该 Command 执行前保存；
- 输出数据在该 Command 执行后立即保存；
- OpenCL 数据读取前必须同步 command queue；CPU 直接读取 host buffer；
- 每个执行事件使用独立快照，不能依赖后续 Tensor 是否复用；
- 常量、权重和临时 Tensor 如果作为该 Execution 的输入，也必须保存；
- Tensor 数据采用固定端序的二进制格式，建议 V1 使用 little-endian；
- 每个文件保存字节数和 SHA-256，防止记录不完整。

## 7. Backend dispatch 信息

`kernels` 是 OpenCL 的可选调试信息，不是普通 Execution 回放的唯一依据。CPU record 不依赖 kernel dispatch，通常只记录 `execution`（如果该 CPU Execution 暴露了名称）和输出快照。

```json
{
  "program": "conv_2d_buf",
  "kernel": "conv_2d_1x1_local",
  "build_options": [
    "-DMNN_SUPPORT_FP16"
  ],
  "gws": [56, 56, 32],
  "lws": [1, 1, 1],
  "args": [
    {
      "index": 0,
      "kind": "scalar",
      "dtype": "int32",
      "value": 56
    },
    {
      "index": 1,
      "kind": "tensor",
      "tensor": "op_000012_input_0"
    }
  ]
}
```

Kernel 参数中的 `cl_mem`、Image 对象和设备地址不能直接保存。必须用 Tensor ID 或临时资源 ID 表示，外部测试重新分配资源后建立映射。

Scalar 参数必须保存精确类型和位宽，例如：

- `int32`；
- `uint32`；
- `float32`；
- `int4` 或固定字节数组。

如果只做 Execution 级回放，可以不保存所有 kernel args，让 Execution 根据 Op 和 Tensor 重新生成参数。当前 V1 不保存 scalar kernel args，保存 `kernels` 的主要目的是确认 OpenCL kernel 路径和辅助定位问题。

## 8. replay_benchmark 参数

位置参数保持与 `benchmark.out` 一致，记录模式通过 `--record` 开启：

```bash
./replay_benchmark.out model.mnn 10 10 0 4 2 --record ./mnn_execution_record

# CPU record
./replay_benchmark.out model.mnn 10 10 0 4 2 --record ./mnn_cpu_record

# OpenCL record
./replay_benchmark.out model.mnn 10 10 3 4 2 --record ./mnn_opencl_record
```

也可以通过 `--args` 传入兼容配置：

```bash
./replay_benchmark.out model.mnn 10 10 3 4 2 --args record_args.json
```

配置文件格式：

```json
{
  "execution_record": {
    "enabled": true,
    "output_dir": "./mnn_execution_record",
    "capture_once": true,
    "dump_logical_tensor": true,
    "dump_storage_tensor": true,
    "dump_op_flatbuffer": true,
    "dump_dispatch": true,
    "backend_only": "CPU",
    "op_filter": []
  }
}
```

字段说明：

- `enabled`：是否开启记录；
- `output_dir`：记录输出目录；
- `capture_once`：只保存一次正式执行，避免 warmup 和 benchmark 循环产生大量重复数据；
- `dump_logical_tensor`：保存逻辑 Tensor；
- `dump_storage_tensor`：保存 Backend 实际存储；
- `dump_op_flatbuffer`：保存独立 Op FlatBuffer；
- `dump_dispatch`：保留字段，用于声明需要保存 kernel dispatch 信息；当前 V1 在 `record.json` 中保存 `program`、`kernel`、`gws` 和 `lws`；
- `backend_only`：只记录指定 Backend；
- `op_filter`：可选的 Op ID、Op name 或 Op type 过滤器。

记录应在 warmup 和首次 tuning 完成后采集。输入数据必须保存实际执行时使用的数据，不能只保存随机 seed。

## 9. 外部单算子回放

外部测试至少支持两种模式。

### 9.1 Execution replay

```text
读取 record.json
读取 op.fb
创建 Backend 和 Tensor
读取 logical input
恢复 Tensor shape / dtype / layout / quant 信息
创建目标 Execution
执行 onResize / onExecute
读取输出
与 logical output 比较
```

该模式用于验证具体 `Execution` 类和 variant 的正确性。

外部测试应该检查：

```text
实际 execution == record.execution
实际 variant   == record.variant
输出 Tensor    == record 输出快照
```

### 9.2 Raw dispatch replay

```text
读取 kernels
创建 OpenCL program/kernel
根据 Tensor ID 创建 Buffer/Image
加载 storage data
必要时恢复 scalar args（V1 Execution replay 由 Execution 重新生成）
恢复 gws/lws
执行 kernel
与 storage output 比较
```

该模式主要用于定位 kernel、padding、向量化和 OpenCL 参数问题。

## 10. Execution 标识

当前 MNN 关闭 RTTI，不能使用 `dynamic_cast` 可靠判断具体 Execution 类。因此支持名称导出的 Backend 需要显式记录：

```text
execution = "ConvBufExecution"
variant   = "conv_1x1_local"
```

当前实现通过 `Execution::getExecutionName()` 和 OpenCL backend 的 resize hook 记录 Execution 名称及 kernel dispatch；scalar kernel args 仍由真实 Execution 在回放时重新生成。CPU Execution 尚未统一导出名称时，`execution` 和 `variant` 可以为空，但仍通过 CPU Backend 创建、执行和输出快照完成回放校验。

不能仅凭 `OpType`、kernel 名称或日志文本推断 Execution 类，因为同一个 OpType 可能对应多个 Execution 实现，同一个 Execution 也可能包含多个内部 kernel。

## 11. V1 验收标准

V1 记录应满足：

1. 可以定位静态图中的每个实际 Command；
2. 可以读取每个 Command 的完整 Op 参数；
3. 可以恢复所有输入、输出、权重和临时 Tensor；
4. 可以确认实际 Backend、Execution 类和 variant；
5. 可以在外部测试中重新执行该 Execution；
6. 输出可以与 benchmark 的实际输出逐元素比较；
7. 记录不依赖原模型进程中的指针、句柄或地址；
8. 相同输入和配置可以稳定生成相同格式的记录；
9. 记录包含版本号、模型校验和、数据字节数及文件校验和。

## 12. 单算子测试架构

### 12.1 总体流程

外部单算子测试不需要为每个 Op 手写一个新的测试程序，而是使用一个通用的 Replay Runner：

```text
Replay Runner
    |
    +-- Record Loader
    |      - 读取 record.json
    |      - 校验记录版本和文件 checksum
    |
    +-- Case Selector
    |      - 按 op_id 选择
    |      - 或按 op_type / execution / variant 选择
    |
    +-- Op Materializer
    |      - 读取独立 op.fb
    |      - 恢复 Op 参数
    |
    +-- Tensor Materializer
    |      - 恢复 shape / dtype / format / quant
    |      - 加载 logical 或 storage 数据
    |
    +-- Backend Execution Runner
    |      - 创建 CPU 或 OpenCL Runtime / Backend
    |      - 创建目标 Execution
    |      - onResize
    |      - onExecute
    |
    +-- Variant Verifier
    |      - 验证实际 Execution 类
    |      - 验证实际 variant
    |
    +-- Output Comparator
           - 比较输出 shape
           - 比较 logical output
           - 必要时比较 storage output
```

单算子测试的输入不是重新构造的随机数据，而是直接来自记录。这样测试复现的是 benchmark 中实际经过的 Tensor 和算子参数。

### 12.2 指定某个 Op 的选择方式

优先使用唯一的 `op_id`：

```bash
execution_replay.out \
    --record ./mnn_execution_record \
    --op-id 12
```

当需要测试某个变体集合时，可以使用过滤条件：

```bash
execution_replay.out \
    --record ./mnn_execution_record \
    --op-type Convolution \
    --execution ConvBufExecution \
    --variant conv_1x1_local
```

过滤条件的优先级建议为：

```text
op_id
  > op_type + execution + variant
  > op_type + execution
  > op_type
```

如果过滤结果不是唯一记录，测试程序应列出候选项并失败，不能随机选择一条记录。

如果指定了 `execution` 或 `variant`，但记录中的实际记录不匹配，也必须失败。不能自动 fallback 到其他 Execution，否则会把“没有复现目标实现”误判为测试通过。

### 12.3 Execution 级回放

Execution 级回放必须调用真实的目标 Backend 创建流程，CPU 和 OpenCL 共用以下步骤：

```text
1. 读取 record.json 中的 runtime
2. 按 `runtime.forward` 创建 CPU 或 OpenCL Runtime
3. 创建对应 Backend
4. 读取 op.fb
5. 根据 Tensor metadata 创建输入和输出 Tensor
6. 加载 input logical data
7. 通过 Backend copy 逻辑数据到 Backend Tensor；CPU 回放的输入和输出使用独立的静态 buffer，确保直接执行期间 host 指针有效
8. Backend::onCreate(inputs, outputs, op)
9. 验证实际 Execution 类
10. Execution::onResize(inputs, outputs)
11. Backend::onExecuteBegin()
12. Execution::onExecute(inputs, outputs)
13. Backend::onExecuteEnd()
14. 读取输出 Tensor
15. 与记录中的 output logical data 比较
```

这里不能直接根据 `execution` 字符串手工调用某个类的构造函数作为默认路径。默认路径必须使用对应 `Backend::onCreate()`，这样可以同时验证 Creator 分派逻辑。OpenCL 额外验证 kernel variant；CPU 以 execution 结果和输出快照为主。

直接构造具体类只作为调试模式使用，因为它会绕过：

- `OpType + Backend memory object` 查找；
- Creator 内部的 shape/属性判断；
- OpenCL fallback 判断；
- 实际 Execution variant 选择。

### 12.4 Tensor 重建策略

普通 Execution 回放使用 `logical_file`：

```text
logical data
    ↓
host Tensor
    ↓ Backend::onCopyBuffer
Backend Tensor
```

这样可以让被测 Execution 自己完成 NC4HW4、NC16HW16、padding 和其他 layout 转换。

如果需要验证已经完成 layout 转换后的 kernel，则使用 `storage_file`：

```text
storage data
    ↓
按 record 中的 storage metadata 创建 Buffer/Image
    ↓
直接加载实际存储数据
```

两种模式应明确区分：

```text
execution-replay : logical input + 真实 Execution
storage-replay   : storage input + 指定 kernel/dispatch
```

Execution replay 是单算子正确性测试的默认模式，storage replay 是底层 kernel 调试模式。

### 12.5 Op 输入输出和原始计算图的关系

记录中的 `inputs` 和 `outputs` 应优先记录实际 Command 使用的 `workInputs` 和 `workOutputs`，而不是只记录原始图上的逻辑输入输出。

原因是静态图经过 geometry、Raster、Copy、Wrap 或 Tensor format conversion 后，Execution 看到的 Tensor 可能已经不是原始 Tensor。

对于每条记录，同时保存：

```text
origin_op_id       原始静态图 Op
command_id         实际执行 Command
op.fb              该 Command 使用的 Op 参数
workInputs         Execution 实际输入
workOutputs        Execution 实际输出
```

如果一个原始 Op 生成了多个 Command，应为每个 Command 生成一个独立回放 case。外部测试默认回放 `command_id`，而不是尝试重新执行整个原始计算图。

### 12.6 Variant 校验

回放前后都要校验 variant：

```json
{
  "expected": {
    "backend": "OPENCL",
    "memory": "BUFFER",
    "precision": "HIGH",
    "execution": "ConvBufExecution",
    "variant": "conv_1x1_local"
  }
}
```

回放结果至少包含：

```text
requested execution
actual execution
requested variant
actual variant
fallback backend
```

以下情况都应判定为失败：

- 目标 Runtime 创建失败；
- 对应 Backend 的 `onCreate()` 返回空；
- 实际创建了 CPU backup Execution；
- 实际 Execution 类不匹配；
- OpenCL 实际 variant 不匹配；CPU record 没有 variant 时不应因为该字段为空失败；
- 输出 Tensor shape 不匹配；
- 输出数据超过容差。

### 12.7 测试用例组织

外部单算子测试可以使用一个通用注册入口：

```text
ExecutionReplayTest
    ├── record/op_000012
    ├── record/op_000013
    └── record/op_000014
```

每个记录 Op 自动生成一个测试名称：

```text
replay/op_000012/Convolution/ConvBufExecution/conv_1x1_local
```

测试框架只负责：

1. 读取记录；
2. 选择目标记录；
3. 调用通用 Replay Runner；
4. 校验 Execution、variant 和输出。

不建议把每个记录转换成一份手写 C++ 测试代码。这样可以让 benchmark 采集的新 case 直接进入单算子回归测试。

## 13. 记录到单算子测试的映射

一个完整映射如下：

```text
benchmark static graph
    |
    | 记录 op / Execution / variant / Tensor snapshots
    v
replay record
    |
    | --op-id 12
    v
single-op case
    |
    | op.fb + Tensor metadata + input data
    v
Backend::onCreate
    |
    | actual Execution / variant check
    v
Execution::onResize + Execution::onExecute
    |
    v
output comparison
```

该设计保证测试既能复现某一个具体 `OpType`，也能复现这个 `OpType` 在指定 shape、precision、memory mode 和设备能力下实际选择的 Execution 实现。
