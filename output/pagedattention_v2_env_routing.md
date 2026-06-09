# PagedAttention V2 环境变量路由设计

## 核心修正

PagedAttention V2 不实现“近似选择一部分 KV block”的 row sparse。那会改变 attention 语义，除非上层明确证明被跳过的 KV 本来就被 mask 掉。

V2 的正确方向是 **row-compressed exact sparse mask attention**：

- FlashMask 参考点：按 column 压缩复杂 mask 信息，让 FlashAttention 跳过无效 block。
- 我们的场景：PIC/PagedAttention 推理里 query 很短、KV 很长，更适合按 query row 压缩每行可见 KV 范围。
- V2 只跳过 mask 语义上确定不可见的 KV，不做近似 KV 选择。

也就是说，对每个 query row `q`：

```text
dense V1:
  scan k = [0, kvLen)
  if k > qLogical: score = -inf

row-compressed V2:
  rowEnd = min(kvLen, qLogical + 1)
  scan k = [0, rowEnd)
```

输出应等价于 dense masked attention。

## 外部一致性

V2 不改变外部定义：

- 不新增 op type。
- 不修改 schema。
- 不修改导出模型中的 `PagedAttention` op 参数。
- 不从 `pic_server` request、`pic_cache` metadata 或 HTTP body 读取实现选择。
- CPU/CUDA/OpenCL backend 仍注册同一个 `OpType_PagedAttention`。

V1/V2 只在 backend 内部通过环境变量选择。

## 环境变量

### 主开关

```text
MNN_PAGED_ATTENTION_IMPL=v1|v2|auto
```

- 未设置：保持当前 V1 行为。
- `v1`：强制当前 V1 通用实现。
- `v2`：启用 V2 优化实现；当前按 direct-op perf 结果，只在 `Qlen == 1` 的 profitable short-query 场景走 V2 row-compressed kernel，其余场景仍走 V1 通用 fast path。
- `auto`：当前等价于 `v2`，保留给后续策略扩展。

### 日志与 profile

沿用：

```text
MNN_PAGED_ATTENTION_PROFILE=1
MNN_PAGED_ATTENTION_NVTX=1
```

新增：

```text
MNN_PAGED_ATTENTION_ROUTE_LOG=1
```

示例日志：

```text
CUDAPagedAttention route layer=12 impl=v2 mode=row_compressed_mask reason=forced_v2 query=1 insert=1 kv_len=8192 sparse=1 mask=0
CUDAPagedAttention route layer=12 impl=v1 reason=v2_qlen_above_profitable_threshold query=512 insert=512 kv_len=512 sparse=0 mask=262144
```

## 路由位置

不在 `pic_server` 或 `Llm` 层控制。路由在 backend PagedAttention 内部：

- CUDA：`source/backend/cuda/execution/PagedAttentionExecution.cu`
- OpenCL：`source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
- CPU：可保持 V1，作为 correctness reference

CUDA 路由顺序保持：

1. 计算 `baseLogical/insertLen/kvLen`。
2. 同步 slot table / sparse query。
3. hydrate external segments。
4. 写当前 K/V 到 PagedCache。
5. cacheblend score path。
6. attention kernel route。

只替换最后的 attention 计算，不改变 external KV hydrate、sparse recompute 写 slot、PendingWrite export 等语义。V1 仍是通用 fast path；V2 当前只接管 direct-op perf 已证明有收益的 `Qlen == 1` 分支。

## V2 exact row-compressed mask 语义

V2 支持的第一类 exact sparse mask：

```text
每个 query row 的合法 KV 范围是 [0, rowEnd)
rowEnd = min(kvLen, qLogical + 1)
```

其中：

- 普通 prefill：`qLogical = baseLogical + q`
- sparse recompute：`qLogical = sparse_query_logical_indices[q]`

这正好覆盖当前 PagedAttention 的 causal mask 和 sparse recompute 场景。它的收益来自跳过 `k > qLogical` 的未来 token，而不是近似跳过历史 token。

CUDA V2 kernel 现在不 materialize 整行 `kvLen` scores。每个 `(batch, query row, q head)` block：

1. 根据 `qLogical` 得到 `rowEnd`。
2. 按 K tile 只遍历 `[0, rowEnd)`。
3. 每个 tile 内 QK score 只计算一次，暂存在 shared `scoreTile`。
4. 用 tile max / tile sum 做 online softmax 合并，把 tile 的 `prob * V` 累积到 running output。

这相当于把 dense causal mask 的“未来列全不可见”压缩成每行一个 `rowEnd` 元数据。当前不是近似 top-k / block-k 选择，也不改变 attention 概率分布。

注意：普通 decode 通常是最后一个 query row，`qLogical + 1` 接近 `kvLen`，这种情况下 rowEnd 压缩不会跳过历史 KV；真正有 rowEnd 压缩收益的主要是 sparse recompute / sparse query 这类 row logical position 明显小于 `kvLen` 的场景。但这只是性能分析结论，不作为 V2 路由条件。

## 当前实现状态

已落地：

- CUDA backend env parser。
- CUDA backend `pagedAttentionRowCompressedMaskKernel`。
- `onExecute` 末端 attention kernel route。
- `MNN_PAGED_ATTENTION_ROUTE_LOG` 路由日志。
- V2 row-compressed kernel 内部使用 tiled online softmax，避免旧验证实现中 max / sum / output 三阶段重复计算 QK。
- V2 mode 下 dense multi-token prefill 使用 V1 通用 fast path；当前只有 `Qlen == 1` 使用 V2 kernel。
- OpenCL backend env parser。
- OpenCL backend 把 `paged_attention` 作为 V1 generic kernel，把 `paged_attention_row` 作为 V2 row kernel。
- OpenCL 默认不再在 `head_dim <= 256` 时无条件走 row kernel；只有 `MNN_PAGED_ATTENTION_IMPL=v2|auto` 且 `Qlen == 1`，或 bench 强制 raw V2 时，才走 row kernel。
- OpenCL direct-op bench：`test/bench_ops/opencl/OpenCLAttentionPerf.cpp`，注册名 `bench_ops/opencl/perf/PagedAttention/V1V2`。

当前限制：

- CPU 仍作为 V1 correctness reference。
- `MNN_PAGED_ATTENTION_IMPL=v2|auto` 不再表示“所有形状都替换成 V2 单 kernel”，而是启用 V2 优化分支；非目标 dense prefill / multi-query sparse recompute 仍由 V1 通用实现承接。
- 显式 float mask 会按 V1 同样方式加到 score 上；当前压缩收益主要来自 causal/paged rowEnd，不会从任意复杂 mask 中解析多个 row intervals。
- 默认未设置环境变量时仍走 V1；V1 是当前通用实现，V2 是显式选择的优化路径。

这里的 `Qlen` 使用实际参与 PagedAttention 计算的 `insertLen`，不是输入 tensor 的 `mQuerySeqLen`。CUDA grid、V1 prefill fast path 和 V2 row-compressed kernel 都按 `insertLen` 作为 query row 数执行。

`Qlen < 2/3 KVLen` 是后续 V2 目标语义区间，但当前 tiled online V2 kernel 在 `Qlen >= 2` 时仍慢于 V1 fast path。`test/bench_ops/cuda` 的 `bench_ops/cuda/perf/PagedAttention/V1V2` 会输出 V1、raw V2 kernel 和当前 V2 route 三列，用于继续校准阈值。该测试使用 bench-only 环境变量：

```text
MNN_PAGED_ATTENTION_BENCH_FORCE_V2_KERNEL=1
```

它只用于 direct-op 性能扫点，不作为请求路由接口。

OpenCL 对应 bench 输出同样的三列：

```bash
LD_LIBRARY_PATH="$PWD/.cache/build/mnn/x64_opencl_bench:$PWD/.cache/build/mnn/x64_opencl_bench/source/backend/opencl:${LD_LIBRARY_PATH:-}" \
  .cache/build/mnn/x64_opencl_bench/run_test.out bench_ops/opencl/perf/PagedAttention/V1V2 3 1
```

其中：

- `impl=v1`：强制 OpenCL V1 generic kernel。
- `impl=v2_kernel`：`MNN_PAGED_ATTENTION_IMPL=v2` + `MNN_PAGED_ATTENTION_BENCH_FORCE_V2_KERNEL=1`，强制 raw V2 row kernel。
- `impl=v2_route`：`MNN_PAGED_ATTENTION_IMPL=v2`，按当前生产路由选择，`Qlen == 1` 走 V2，其余走 V1。

## 后续扩展

如果要支持更复杂 mask，可以扩展为 row-wise intervals：

```text
row q -> [begin0, end0), [begin1, end1), ...
```

但这些 intervals 必须来自真实 mask 语义，而不是为了性能近似挑选 KV。

## 验证方式

direct op / PIC server 测试只切环境变量，不改 request：

```bash
export MNN_PAGED_ATTENTION_IMPL=v2
export MNN_PAGED_ATTENTION_ROUTE_LOG=1
export MNN_PAGED_ATTENTION_PROFILE=1
```

期望看到：

```text
CUDAPagedAttention route ... impl=v2 mode=row_compressed_mask ...
CUDAPagedAttention profile op=v2_row_compressed_mask ...
```
