# replay_benchmark 算子适配与 kernel corpus 总结

## 一、replay_benchmark 算子适配逻辑

`replay_benchmark` 的算子适配逻辑集中在 `replay_benchmark/ReplayRunner.cpp` 的
`replayOp()` 函数。它把 MNN 正常推理流水线中 Backend 与 Execution 的协作过程
剥离到单算子粒度重放，用于验证某个算子在特定后端下的实现是否与录制时的结果一致。

核心思路：从 `record.json` 还原算子的输入/输出张量规格与 Op 描述，手动调用
`Backend` 和 `Execution` 基类的各个生命周期函数，复现一次完整推理。

### 1. 基类函数作用

**Runtime 层**（`ReplayRunner.cpp:149-158`）
- `MNNGetExtraRuntimeCreator(type)`：按 forward 类型获取 Runtime 工厂。
- `Runtime::onCreate(BackendConfig*, Backend*)`：创建运行时，初始化设备上下文
  （如 OpenCL context/queue）。
- `Runtime::onCreate(&backendConfig, nullptr)`：在 Runtime 上创建一个具体的
  `Backend` 实例。

**Backend 层**
- `Backend::onCreate(inputs, outputs, op)`（ReplayRunner.cpp:210）：算子适配
  核心入口。根据 op 类型 + 输入输出形状，在 backend 上构造对应的 `Execution`。
  返回 nullptr 或 `valid()==false` 表示该 backend 不支持此 op。
- `Backend::onAcquireBuffer(tensor, storageType)`（ReplayRunner.cpp:229,235）：
  为输入/输出张量分配设备内存。`STATIC` 不复用；`DYNAMIC_SEPERATE` 不复用且
  release 时不释放；`DYNAMIC` 可复用池。
- `Backend::onResizeBegin()`（ReplayRunner.cpp:244）：进入 resize 阶段的开头
  （GPU backend 用于准备命令缓冲池等）。
- `Backend::onExecutionResizeBegin(op, execution)`（ReplayRunner.cpp:245）：
  可选钩子，per-execution resize 前通知（默认空实现，OpenCL backend 用于录制
  回放诊断信息）。
- `Backend::onExecutionResizeEnd(op, execution)`（ReplayRunner.cpp:247）：
  per-execution resize 后通知。
- `Backend::onResizeEnd()`（ReplayRunner.cpp:248）：resize 阶段收尾，返回
  `ErrorCode`。GPU 上会触发 kernel 编译完成 / buffer 真正分配。
- `Backend::onExecuteBegin()`（ReplayRunner.cpp:320）：执行阶段开始
  （GPU 上 enqueue 命令）。
- `Backend::onExecuteEnd()`（ReplayRunner.cpp:322）：执行阶段结束
  （GPU 上 flush/sync）。

**Execution 层**
- `Execution::valid()`（ReplayRunner.cpp:211,223）：检查创建出的 execution
  是否有效。`mValid=false` 通常意味着该 op 在当前精度/配置下不可用。
- `Execution::onResize(inputs, outputs)`（ReplayRunner.cpp:246）：形状推理 +
  资源准备。计算输出形状、选择 kernel 变体、预分配 workspace。返回
  `ErrorCode`，非 `NO_ERROR` 即失败。
- `Execution::onExecute(inputs, outputs)`（ReplayRunner.cpp:321）：真正计算。
  执行算子逻辑，把结果写入 output 张量。返回 `ErrorCode`。
- `Execution::getExecutionName()`（ReplayRunner.cpp:284）：返回实现名
  （如 `ConvDirect`），用于与 record 中记录的实现名做一致性校验。

### 2. 通过状态含义

`replayOp` 的通过判定是多阶段串联的，每一阶段失败都会提前 return false。只有
全部通过才会打印 `Replay succeeded`。

1. **Record 加载**：`reader.load()==true`，否则 record.json 不存在/格式不对。
2. **Op 选择**：`reader.select()!=nullptr`，否则 opId/opType/execution/variant
   过滤后未唯一命中。
3. **Op 描述还原**：`op != nullptr` 且 `modelOp->type()==op->type()`，否则
   op.fb 读不到，或与模型中算子类型不一致。
4. **Runtime 创建**：`creator!=nullptr && runtime!=nullptr`，否则该后端未编译/
   不可用。
5. **Backend 创建**：`backend!=nullptr`，否则后端初始化失败。
6. **Execution 创建**：`execution!=nullptr && execution->valid()==true`（CPU_EXTENSION
   时允许 fallback 到 CPU）。失败表示算子未适配：该 backend 不支持此 op 类型/
   精度。
7. **内存分配**：`onAcquireBuffer()==true && copyFromHostTensor()==true`，否则
   设备显存不足或 H2D 拷贝失败。
8. **Resize**：`resizeCode==NO_ERROR && resizeEndCode==NO_ERROR`，否则形状推理
   失败、kernel 编译失败。
9. **实现名校验**：`opSpec->execution == execution->getExecutionName()`，否则
   选到了不同实现（如预期 `ConvWinograd` 实际 `ConvDirect`）。
10. **Variant 校验**（OpenCL）：`traces.front().kernelName == opSpec->variant`，
    否则选到了不同的 OpenCL kernel 变体。
11. **Execute**：`executeCode==NO_ERROR`，否则计算过程出错。
12. **输出对比**：`compareTensor()==true`（atol=1e-4, rtol=1e-3），否则结果数值
    不一致：实现正确性回归。
13. **PerfCounter**（可选）：`perfReport.status=="ok"`，性能计数器采集失败不
    影响 replay 通过，但会写 error。

最终全部通过 → `std::cout << "Replay succeeded: ..."` → `return true`；
任一失败 → `std::cerr` 打印原因 → `return false`。

`PerfCounterReport.status` 三态：
- `"unavailable"`：未请求 perf 或未编译 PerfCounter 模块。
- `"ok"`：perf 采集+同步成功。
- 仍为 `"unavailable"` 且 `error` 非空：perf 采集失败。

### 3. 流程草图

```
   record.json  +  model.mnn  +  tensors/*.bin
        |              |              |
        v              v              v
   RecordReader    ModelOpDatabase   host input bytes
        |              |              |
        |   select(opId/opType/       |
        |     execution/variant)     |
        v                             |
   ReplayOpSpec ----------------------+
   (inputs/outputs spec, op)         |
        |                             |
        v                             |
   ┌───────────── Runtime 层 ──────────────────┐
   │ MNNGetExtraRuntimeCreator(forward)         │
   │   -> Runtime::onCreate()                   │
   │       -> Backend = Runtime::onCreate()      │
   └─────────────────────────────────────────────┘
        |
        |  构造 device tensors (shape/dtype/storageType/channelPack)
        |  restoreTensorRegions() 恢复 region 映射
        v
   ┌───────────── Execution 适配 ────────────────┐
   │ Backend::onCreate(inputs, outputs, op)       │  <- 算子适配核心入口
   │   |                                          │
   │   ├─ nullptr / !valid() ?                    │
   │   │   └─ CPU_EXTENSION: fallback 到 CPU      │
   │   │       backend, 重新 onCreate()           │
   │   └─ execution (具体算子实现)                │
   └──────────────────────────────────────────────┘
        |
        |  onAcquireBuffer(inputs/outputs, STATIC|DYNAMIC_SEPERATE)
        |  copyFromHostTensor()  (H2D 输入)
        v
   ┌───────────── Resize 阶段 ───────────────────┐
   │ Backend::onResizeBegin()                     │
   │ Backend::onExecutionResizeBegin(op, exec)    │
   │ Execution::onResize(inputs, outputs)         │  <- 形状推理/kernel 编译
   │ Backend::onExecutionResizeEnd(op, exec)      │
   │ Backend::onResizeEnd()                        │
   │   != NO_ERROR ? -> FAIL "resize failed"       │
   └───────────────────────────────────────────────┘
        |
        |  校验: execution name == record.execution ?
        |  校验: OpenCL kernel variant == record.variant ?
        v
   ┌───────────── Execute 阶段 ──────────────────┐
   │ [可选] PerfCounter::Session::start()         │
   │ Backend::onExecuteBegin()                    │
   │ Execution::onExecute(inputs, outputs)        │  <- 真正计算
   │ Backend::onExecuteEnd()                      │
   │   != NO_ERROR ? -> FAIL "Execution failed"   │
   │ [可选] output->wait(MAP_TENSOR_READ) 同步     │
   │ [可选] PerfCounter::Session::stop()           │
   └───────────────────────────────────────────────┘
        |
        v
   ┌───────────── 结果校验 ───────────────────────┐
   │ for each output:                             │
   │   compareTensor(actual, record.logical.bin)  │
   │     atol=1e-4, rtol=1e-3                     │
   │   != true ? -> FAIL "Output mismatch"        │
   └───────────────────────────────────────────────┘
        |
        v
   全部通过 -> "Replay succeeded: op_id=.. type=.. execution=.."
              return true

   任一阶段失败 -> stderr 打印原因, return false
   (perfReport.status: "unavailable" / "ok" / error 字符串)
```

关键点：`Backend::onCreate` 是算子适配的入口——它根据 op 类型在 backend 的
`Execution` 工厂表里查找具体实现；`Execution::onResize` 是形状/资源适配阶段；
`Execution::onExecute` 是数值适配验证。三层串联，任何一层与录制时不一致都会
被判失败。

## 二、kernel corpus 的作用

`kernel_corpus` 是一个 GPU kernel 源码语料库，归档了 MNN（73 个 tag，
1.2.0→3.6.0）和 ncnn（37 个 tag）历史上发布的 OpenCL/Vulkan kernel 源码快照。
它不参与 MNN 的正常编译和推理，而是作为 `replay_benchmark` 的离线评测素材库
存在。

### 1. 解决的三个问题

**跨版本 kernel 回归对比**：同一算子（如 `binary_buf_fp32`）在不同 MNN 版本
（1.2.0 vs 3.6.0）下的 kernel 实现可能差异很大。corpus 把这些源码快照都收齐，
让 `KernelCorpusBenchmark` 可以把同一个 kernel 在不同 GPU 驱动/不同版本实现下
跑同一组 workload，用 PMU 计数器对比性能差异。

**kernel 级（而非算子级）隔离测试**：`replay_benchmark` 里的 `replayOp` 是算子
粒度重放——它依赖 MNN 的 `Backend/Execution` 框架。而 kernel corpus 更进一步：
通过 `kernel_corpus_bridge`（`Bridge` + `OpAdapter`）把 kernel 源码从 backend
框架中剥离出来，直接编译源码 → 构造 buffer → setArg → dispatch → readback →
validate。这样能测裸 kernel 的编译是否通过、dispatch 是否响应、数值是否正确，
不经过 MNN 的算子调度层。

**多框架对齐**：同时归档 MNN 和 ncnn 的 Vulkan kernel，可以在同一 GPU 上对比
两个框架的同类算子 kernel 性能。

### 2. 数据组织

```
kernel_corpus/
├── sources/<framework>/<tag>/   # 源码快照（按 git tag 切片）
├── operators.json               # kernel → (op_type, variant, tag, entry, file) 映射表
├── operator_cases.json          # 具体测试用例（shape/参数/validator）
├── arg_layouts.json             # kernel 参数布局（setArg 顺序）
├── manifest.json                # 每个快照的 commit/SHA-256 校验
├── config.json                  # tag 选择与 backend 状态
└── operators/<backend>/<op>/<framework>/<tag>/<file>  # 按 op 归类的 kernel 文件
```

### 3. 与 replay_benchmark 其它模块的关系

```
                    ┌─────────────────────────────┐
                    │   kernel_corpus (源码语料库) │
                    │   - operators.json          │
                    │   - operator_cases.json     │
                    │   - sources/ 快照            │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  kernel_corpus_bridge        │
                    │  Bridge: 按 (framework,tag)  │
                    │   选择 MnnBridge/NcnnBridge  │
                    │  OpAdapter: 把 CaseSpec 适配 │
                    │   成 AdaptedCase             │
                    │   (source/args/buffers/gws)  │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────▼────────────────────┐
              │  KernelCorpusBenchmark                   │
              │  per case:                               │
              │   compile  -> compileStatus              │
              │   dispatch -> dispatchStatus (responsive)│
              │   validate -> validationStatus (valid)   │
              │   PMU      -> pmuStatus + delta          │
              └──────────────────────────────────────────┘
```

### 4. CaseReport 四个状态字段

定义于 `replay_benchmark/KernelCorpusBenchmark.hpp:30-33`：

- `compileStatus`：kernel 源码能否在当前 GPU 驱动上编译通过。
- `dispatchStatus`：dispatch 是否能在超时内返回（`responsive`）。
- `validationStatus`：readback 结果是否通过 validator
  （如 `identity_fp32`/`matmul_fp32`）。
- `pmuStatus`：PMU 计数器采集是否成功，`controlDelta`/`workloadDelta` 是控制组
  与负载组的计数差值。

### 5. 总结

kernel corpus 是 GPU kernel 的"版本化石库"，配合 `KernelCorpusBenchmark` 实现
脱离 MNN 算子框架的裸 kernel 编译/调度/数值/性能回归测试，用于发现"同一算子
在不同版本/不同框架实现下的 GPU 行为差异"。

它和 `replayOp` 的区别：`replayOp` 测整个 Execution 在 backend 中的适配；
kernel corpus 测单个 kernel 文件能否独立编译并通过 PMU 评测。
