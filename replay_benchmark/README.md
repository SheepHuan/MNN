# replay_benchmark OpenCL PMU 基线

`replay_benchmark.out --opencl-pmu-bench` 内置一组最小 OpenCL C workload，用于验证
MNNPerfCounter 是否能够在一个明确的 GPU 执行区间内读到 PMU 变化。该模式不需要
model、record 或 replay 文件，只在显式传入 `--opencl-pmu-bench` 时运行。

## 覆盖的 OpenCL 概念

- `__global` / buffer：小工作集重复访问用于观察 L1/L1.5/L2 locality；较大工作集
  顺序或跨步访问用于观察 L2 miss、外部内存读；独立 write 和 read-modify-write
  用于观察写流量。
- `image2d_t`：覆盖 image read 和 image write。image 使用 RGBA/FP32 格式，适合
  检查图像对象访问、TP/纹理路径以及 UCHE 相关流量。
- sampler/texture cache：nearest 与 linear filtering 分开测试。OpenCL 的 image
  sampler 路径不等同于普通 buffer cache，具体是否命中由 GPU、尺寸、地址模式和
  驱动决定，不能只根据源码推断。
- `__constant`：小的只读常量表，观察 constant address space 的访问行为。
- `__local`：工作组共享的片上 local memory；kernel 使用 local scratch、
  `barrier(CLK_LOCAL_MEM_FENCE)` 和 work-group 内复用，验证 local load/store、
  bank conflict 与同步开销。`__local` 不是设备全局 L2，也不是 host 可直接访问的
  buffer。
- private/register：循环中的标量累加和地址计算，主要用于区分 memory workload
  与 ALU workload。
- ArchProbe 风格的扩展 workload：`buffer_stride` 使用 power-of-two 工作集和
  固定 stride，`buffer_pchase` 在 PMU 区间外准备 pointer chain 并用单 work-item
  串行追踪，`buffer_vec4` 使用 `float4` 访问，`fp32_throughput` / `fp16_throughput`
  使用多个独立累加器。
- 特殊内存 workload：`image_stride_x` / `image_stride_y` 分离二维 image 方向，
  `constant_bandwidth` 扫描 constant working set，`local_bandwidth` 与
  `local_barrier` 分离 local load 和同步，`atomic_contended` / `atomic_distributed`
  分离原子竞争与分散计数器。
- work-item / work-group：global size 表示 work-item 总数，local size 表示一个
  work-group 的大小；local memory 和 barrier 只能在同一个 work-group 内协作。
  可通过 `--opencl-pmu-local-size 32|64|128|256` 显式选择一维 buffer/compute
  work-group，global size 会向上取整；默认仍使用 64。image/texture 保持二维
  `256 × 256` dispatch，不套用一维 local size。
- atomic / memory ordering：global atomic case 使用 `atomic_add` 测量原子更新和
  全局内存排序带来的额外活动；它不是普通 global write 的替代品。
- INT32 / FP16 / FP32：分别使用整数递推、FP32 FMA 和 `cl_khr_fp16` half 运算。
  FP16 case 只有在设备报告 `cl_khr_fp16` 且 kernel 能成功编译时才执行。

## Kernel 文件组织

为了便于增加新的基线，OpenCL C 字符串按功能嵌入到独立头文件，而不是集中在
`.cpp`：

- `OpenCLPmuBufferKernels.hpp`：buffer reuse、stream、write 和空 control kernel；
- `OpenCLPmuComputeKernels.hpp`：INT32、FP32、constant、local/barrier、atomic；
- `OpenCLPmuImageKernels.hpp`：image read/write、nearest/linear texture read。

`.cpp` 只负责 OpenCL 对象、参数、dispatch、同步和 PMU 结果组织。

## 运行方式

```bash
replay_benchmark.out --opencl-pmu-bench
replay_benchmark.out --opencl-pmu-bench \
  --opencl-pmu-case buffer_reuse_small,buffer_stream_large,local_memory
replay_benchmark.out --opencl-pmu-bench \
  --opencl-pmu-iterations 200 --opencl-pmu-size 16777216 \
  --perf-counter-output perf/opencl-pmu.json

# Keep one PMU workload session around five consecutive dispatches. Warmup is
# outside both the control and workload sessions.
replay_benchmark.out --opencl-pmu-bench \
  --opencl-pmu-case buffer_fp32 --opencl-pmu-iterations 100 \
  --opencl-pmu-workload-runs 5 --opencl-pmu-warmup-runs 2 \
  --opencl-pmu-local-size 128 \
  --perf-counter-events gpu_active_cycles
```

默认 PMU profile 会根据 GPU 自动选择：A7xx 使用覆盖 SP、TP、UCHE、HLSQ、CP、RBBM
和 RB 的代表性事件；Mali 使用 MNNPerfCounter 的跨架构活动、compute 和 L2 别名。
通过 `--perf-counter-events a,b,c` 可以指定事件。事件超过硬件物理槽位时会自动
按 group/slot 分批，每批都重复 control/workload 测量。

## 报告解释

每个 case 会先在 PMU 采样前创建并初始化 program、kernel、buffer、image 和 sampler，
然后执行 warmup。之后先执行一个最小 `pmu_empty` control 区间，再执行目标 workload
区间，且每个区间都调用 `finish()` 后才读取计数器。报告中的：

- `control_delta`：control 区间计数器变化；
- `workload_delta`：目标 kernel 区间计数器变化；
- `iterations`：单次 kernel dispatch 内部的循环次数；
- `workload_runs`：同一个 workload PMU session 内连续 enqueue 的 dispatch 次数；
- `warmup_runs`：PMU session 开始前执行的预热 dispatch 次数；
- `prepare_count`：case 资源准备次数，正常应为 1；
- `sampled_workload_dispatches`：实际落在 workload PMU session 内的 dispatch 数量；
- `dispatches`：本次 workload 实际执行的 dispatch 数量；PMU 可用时与
  `sampled_workload_dispatches` 相同；
- `working_set_bytes` / `stride_bytes`：cache/stride workload 的实际配置；
- `vector_width`：vector workload 的元素宽度，`buffer_vec4` 为 4；
- `image_width` / `image_height`：image 实际二维尺寸；`--opencl-pmu-size` 会按
  RGBA FP32 像素数调整尺寸；
- `local_bytes`：local workload 的动态 local allocation 大小；
- `counter_count`：atomic workload 的目标计数器数量。
- `global_size` / `local_size`：实际 dispatch geometry 对象，包含 `x` 和 `y`；image/texture
  的 `local_size` 为零对象，表示使用 OpenCL 默认二维 local 选择；
- `responsive`：workload 与 control 有可观察差异；
- `discriminative`：当前基线中等同于 responsive，表示该事件能区分 workload；
- `valid`：符合该 case 预期方向的响应；
- `unsupported_counters`：当前 GPU/驱动没有该事件，未进行猜测式读取。

`readable` 只表示 ioctl/驱动读取成功。`workload_delta == 0` 不等于 PMU 失败：
事件可能属于未被该 kernel 触发的硬件模块、只统计 graphics 路径、被驱动屏蔽，或
在当前 dispatch 规模下没有形成可见活动。判断事件是否有效应同时看 readable、
control/workload 差异、workload 类型以及重复运行的一致性。

`workload_runs` 用于提高单次 PMU 区间的计数规模；它和 `iterations` 不同，前者会
增加真实 kernel dispatch 次数，后者只增加一个 dispatch 内的循环。`warmup_runs`
用于降低首次 OpenCL 执行、频率切换和冷启动缓存对采样的影响。

在 Android 真机上才有意义地评价 Adreno/Mali PMU。桌面 host 构建只能验证命令行、
JSON 和 metadata 测试，不能证明移动 GPU 的 cache 或 PMU 语义。

单事件重复 sweep 使用 `skills/opencl-pmu-sweep/scripts/sweep_opencl_pmu.py`，
测量模型和 CSV 契约见
`docs/superpowers/specs/2026-07-25-opencl-pmu-sweep-design.md`。

## 设备验证目标

- Rhinopi-X1：`root@192.168.101.227`，工作目录 `/mnt/nvme/workspace/replay-benchmark`，Adreno 740/A7xx。
- OrangePi：`root@192.168.101.113`，工作目录 `/mnt/ssd/workspace`，Mali-G610。

两台设备都使用同一套 `replay_benchmark.out`、`libMNN.so`、`libMNN_Express.so` 和
`libMNN_CL.so`，通过 `LD_LIBRARY_PATH` 指向设备工作目录的 `lib/`。

多个 MNN/ncnn tag 的 GPU kernel 源码快照位于
[`kernel_corpus/`](kernel_corpus/)，归档范围、版本矩阵和校验方式见其 README；该
目录当前只作为源码参考库，不参与 replay_benchmark 编译或运行时链接。
