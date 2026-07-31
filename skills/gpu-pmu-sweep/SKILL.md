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

默认从 CUDA kernel corpus 中按 `op_type` 选择一个代表 case（当前 60 个
`op_type`，每个保留 `operator_cases.json` 中的第一个 variant/case），再和
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
  --output-csv cuda_kernel_pmc_rows.csv \
  --output-json cuda_kernel_pmc.json
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

如需恢复原来的 425 个 CUDA case 全量扫描，显式指定：

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

## 本地执行

```bash
# CUDA 本机
sudo LD_LIBRARY_PATH=build/:build/source/backend/cuda:. build/replay_benchmark.out \
  --kernel-corpus-bench \
  --kernel-corpus-root replay_benchmark/kernel_corpus \
  --kernel-corpus-runs 1 \
  --perf-counter-events sm__cycles_elapsed.avg \
  --perf-counter-output /tmp/local_pmu.json

# OpenCL 本机（需要 OpenCL GPU）
python3 skills/gpu-pmu-sweep/scripts/sweep_opencl_pmu.py \
  --binary ./build/replay_benchmark.out \
  --lib-dir ./build \
  --cases buffer_fp32 \
  --events gpu_active_cycles \
  --output-csv records/local.csv
```

## 验证

修改脚本后运行单元测试：

```bash
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
