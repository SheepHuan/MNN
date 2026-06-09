# PagedAttention V1/V2 direct-op 性能扫点

## 测试入口

- 日期：2026-06-09
- 设备：Jetson `jetson@192.168.101.192`
- 测试：`test/bench_ops/cuda/CudaAttentionPerf.cpp`
- 注册名：`bench_ops/cuda/perf/PagedAttention/V1V2`
- 命令：

```bash
ART=.cache/output/mnn/artifacts/jetson_cross_cuda
LD_LIBRARY_PATH="$PWD/$ART/lib:/usr/local/cuda-12.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}" \
  "$ART/bin/run_test.out" bench_ops/cuda/perf/PagedAttention/V1V2 2 1
```

每个 case 输出三行：

- `v1`：`MNN_PAGED_ATTENTION_IMPL=v1`
- `v2_kernel`：`MNN_PAGED_ATTENTION_IMPL=v2` 且 `MNN_PAGED_ATTENTION_BENCH_FORCE_V2_KERNEL=1`
- `v2_route`：`MNN_PAGED_ATTENTION_IMPL=v2`，使用当前生产路由

`MNN_PAGED_ATTENTION_BENCH_FORCE_V2_KERNEL` 只用于 direct-op threshold 扫点，不作为请求路由接口。

## insertLen / Qlen 定义

`insertLen` 是当前 PagedAttention 实际参与 attention 计算的 query rows：

```text
insertLen = min(meta.add, new_kv_seq_len, query_seq_len)
```

因此本文里的 `Qlen` 就是 `insertLen`。它不是输入 tensor 的 `mQuerySeqLen` 概念；kernel grid、V1 fast path、V2 row-compressed kernel 都按 `insertLen` 执行。

在 PIC sparse recompute 中，score layer 之前通常是 full/reference 语义，`Qlen` 接近完整 prefill，应走 V1；score layer 之后 selected query rows 会变少，但是否走 V2 需要按 direct-op 性能阈值决定。

## 3B / KVLen=2048 sweep

形状：

```text
B=1, qH=24, kvH=8, D=128, KVLen=2048
```

| Qlen | Q/KV | V1 ms | raw V2 kernel ms | current V2 route ms | raw V2 speed vs V1 | route decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.0005 | 5.453 | 1.896 | 1.893 | 2.877x | V2 |
| 2 | 0.0010 | 2.475 | 2.834 | 2.473 | 0.873x | V1 |
| 4 | 0.0020 | 2.551 | 7.023 | 2.363 | 0.363x | V1 |
| 8 | 0.0039 | 2.626 | 9.266 | 2.572 | 0.283x | V1 |
| 16 | 0.0078 | 3.580 | 17.084 | 3.584 | 0.210x | V1 |
| 32 | 0.0156 | 7.115 | 33.526 | 7.112 | 0.212x | V1 |
| 64 | 0.0312 | 14.170 | 66.047 | 14.179 | 0.215x | V1 |
| 128 | 0.0625 | 28.311 | 129.522 | 28.325 | 0.219x | V1 |
| 256 | 0.1250 | 56.676 | 250.193 | 56.651 | 0.227x | V1 |
| 512 | 0.2500 | 113.338 | 466.289 | 113.313 | 0.243x | V1 |
| 1024 | 0.5000 | 226.687 | 799.006 | 226.636 | 0.284x | V1 |
| 1536 | 0.7500 | 340.100 | 996.891 | 340.156 | 0.341x | V1 |

## 结论

当前 V2 tiled online kernel 只有 `Qlen == 1` 明确快于 V1。`Qlen=2` 已经慢于 V1，`Qlen>=4` 的差距迅速扩大。

因此当前生产路由不应使用 `Qlen < 2/3 KVLen`。这个条件描述的是“语义上 short query / long KV”的目标区间，但不是当前 kernel 的性能阈值。

当前 production threshold：

```text
MNN_PAGED_ATTENTION_IMPL=v2:
  use V2 kernel iff Qlen == 1
  otherwise use V1 generic/fast path
```

后续如果要把阈值扩大到 `Qlen < 2/3 KVLen`，需要先优化 V2 kernel 对 multi-query 的并行方式。当前 raw V2 是每个 query/head 一个 block，Qlen 增加时调度和 per-row K scan 成本线性堆叠；V1 fast path 的 QK/softmax/QKV tiled pipeline 对 multi-query 明显更强。
