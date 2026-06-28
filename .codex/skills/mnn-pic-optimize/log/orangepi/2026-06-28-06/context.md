# Context

## 现象

OrangePi 5 Plus / Mali-G610 上 Qwen3-8B 的 graph-boundary `cacheblend` 在 `ctx2560` 高 ratio warm 阶段失败：

```text
Native graph-level cacheblend score/top-k failed on backend mnn_opencl
last_error=forwardRaw outputs empty seq_len=2560 add=2560 all_seq=0 gen_seq=0 output_tokens=0
paged_max=4096 request_capacity=4096 logical_length=2560 previous=0 remove=0
```

服务端日志只打印：

```text
code=5 in onForward, 637
PIC forward failed: outputs empty, seq_len=2560 add=2560 all_seq=0 gen_seq=0 paged_max=4096 request_capacity=4096 logical_length=2560
```

`code=5` 是 MNN `INVALID_VALUE`，不是 `OUT_OF_MEMORY`。同样 context 的 `epic 0.20/0.30/0.40/0.50` 能跑通，说明不是单纯 token 长度或 PagedCache 容量问题。

## 根因分析

`epic` 和 `cacheblend` 的 graph-boundary 差异：

- `epic` 在 C++ 层提前给出固定 selected local indices，然后 `beginPicGraphActivePlan()` 让图按已知 active rows 执行。
- `cacheblend` 必须在 score layer 的 OpenCL backend 内完成 value delta score 和 top-k，再把 selected local indices 写回 `PagedKVMeta`，随后图继续 compact active rows。

失败集中在 OpenCL `runCacheBlendScoring()` 的 top-k 路径。`picTokenCount > 1024` 时会优先使用 staged top-k：

- `stage1024` 使用 `1024 * (sizeof(float) + sizeof(int)) = 8192` bytes local memory。
- `stage2048` 使用 `2048 * (sizeof(float) + sizeof(int)) = 16384` bytes local memory。
- `ctx2560 cacheblend 0.20/0.30` 的 topK 大约为 512/768，会进入 staged top-k；之前 0.05/0.10 成功，符合问题从较高 topK/staged local memory 压力开始暴露。

OrangePi `clinfo` 显示 Mali-G610:

```text
Max work group size 1024
Local memory size 32768 (32KiB)
```

虽然设备声明 local memory 足够，Mali driver 在这个 staged bitonic top-k 的动态 local memory 分配和 barrier 密集路径上不稳定，最终表现为 top-k 返回非法 selected rows 或 OpenCL enqueue/readback 失败，C++ selected 校验返回 `INVALID_VALUE`，MNN graph `onForward()` 返回空输出。

Adreno 没有失败的原因不是上层语义不同，而是 backend 路径和驱动特性不同：

- Adreno 走专门的 cacheblend value image score 路径，减少对同一 PagedCache source slots 的 buffer 写入压力。
- Adreno 对 staged top-k 的 local memory / workgroup 调度更宽容。
- 现有 tune key 已按 `mali` / `adreno` 设备族分 namespace，不会互相复用 top-k family 选择。

## 修复

修改文件：

```text
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
```

关键改动：

- `_cacheBlendTopKDispatchForFamily()` 中，Mali runtime 把 staged top-k 可用 local memory 上限保守裁剪到 `8 * 1024` bytes。
- 这样 Mali 仍可使用 `stage1024` 小 staged path，但会拒绝需要 16KiB local memory 的 `stage2048`，高 ratio 自动回到 legacy top-k。
- `runTopKFamily()` 对 `setArg` / `enqueueNDRangeKernel` 的非 `CL_SUCCESS` 立即返回 `INVALID_VALUE`，不再假装成功继续读回。
- staged top-k 若执行失败或读回 selected rows 后发现 out-of-range / duplicate，会打印具体原因并 retry legacy top-k。

这不是语义 fallback：score 仍由当前请求 OpenCL PagedCache / 持久 PIC cache 源完成，top-k 仍在 OpenCL kernel 上执行，只是 Mali 高 ratio 使用更保守的 legacy top-k kernel，避免 staged local-memory 不稳定。

## 构建与同步

本地交叉编译 OrangePi artifact：

```bash
MNN_TARGET_DEVICE=orangepi5plus \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

脚本检测到旧 build cache 的 compiler/sysroot 为空并清理重配。产物检查为 AArch64，`MNN_OPENCL=ON`、`MNN_VULKAN=ON`、`MNN_CUDA=OFF`。

同步到 OrangePi SSD：

```bash
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

远端 artifact 和测试数据均放在：

```text
/mnt/ssd/code/.cache/mnn_opencl_pic
```

没有向 OrangePi 根分区写 benchmark 数据。

## 验证

固定 OpenCL runtime cache 路径：

```text
/mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/runtime_cache/opencl/mnn_cachefile.bin
```

验证命令：

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices --frequency-profile max \
  --contexts 2560 --model-key qwen3-8b \
  --modes cacheblend --ratios 0.20 \
  --benchmark-csv benchmark.csv --only-missing-contexts \
  --restart-server-each-spec \
  --run-id orangepi_qwen3_8b_ctx2560_cb020_mali_topkfix_20260628_0621
```

结果：

```text
orangepi,Orange Pi 5 Plus,Qwen3-8B,OpenCL,"cpu=max,gpu=max,ddr=max",2560,cacheblend,0.20,47.648407,53.72687485648786
```

继续验证：

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices --frequency-profile max \
  --contexts 2560 --model-key qwen3-8b \
  --modes cacheblend --ratios 0.30 \
  --benchmark-csv benchmark.csv --only-missing-contexts \
  --restart-server-each-spec \
  --run-id orangepi_qwen3_8b_ctx2560_cb030_mali_topkfix_20260628_0624
```

结果：

```text
orangepi,Orange Pi 5 Plus,Qwen3-8B,OpenCL,"cpu=max,gpu=max,ddr=max",2560,cacheblend,0.30,46.149394,55.472017682399034
```

两次 `/v1/tune/update_cache` 都返回 200：

```text
status=ok
scope=mnn_runtime_cache
note=runtime manager cache updated through MNN native cache flow
```

两行已立即合并到 `benchmark.csv`。

## 后续注意

`0.30` 比 `0.20` 略快，不满足高 ratio 单调变慢的理论预期。这已不是本次 HTTP 500 bug，而是性能稳定性/测量噪声或 cache/tune 形状差异问题；后续补测 `0.40/0.50` 时需要继续观察单调性，并必要时做 repeat/profile。

