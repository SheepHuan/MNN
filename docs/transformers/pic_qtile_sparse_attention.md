# PIC QTile Sparse Attention 实现逻辑

本文总结当前 CUDA / OpenCL PIC sparse prefill 中 qtile sparse attention 的实现逻辑，重点说明 active query rows 的计算方式、FlashAttention 式 online softmax，以及如何跳过 causal 不可见的 K tile。

相关源码位置：

- CUDA: `source/backend/cuda/execution/PagedAttentionExecution.cu`
- OpenCL dispatch: `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
- OpenCL kernels: `source/backend/opencl/execution/cl/attention_buf.cl`

## 基本语义

当前 qtile sparse attention 是 Q 维 sparse：

- 只对 `sparse_query` 指定的 active query rows 计算 attention。
- K/V 仍来自当前请求的 PagedCache。
- 每个 active query row 通过 `sparse_query[q_index]` 找到真实 logical position。
- causal 可见范围由 logical position 决定，即 query row `q_logical` 只能看 `K[0, q_logical]`。

它不是任意 K/V block sparse。K 方向只做 causal 范围裁剪和整 tile 跳过，仍按连续 K tile 扫描可见前缀。

## Query Row 映射

每个 active row 有两个坐标：

```text
q_index   compact active row index, 也是输出 row index
q_logical 原 prompt / PagedCache 中的真实 logical position
```

kernel 中的 query 读取位置由 `query_rows_are_full` 决定：

```text
q_row = query_rows_are_full ? q_logical : q_index
```

- `query_rows_are_full = true`：score layer 场景。输入 Q 仍是 full prompt rows，输出是 compact rows，因此按 `q_logical` 读取 Q。
- `query_rows_are_full = false`：score layer 后续 sparse layer。图中 hidden/Q 已经 gather 成 compact rows，因此按 `q_index` 读取 Q。

输出始终写到 compact row `q_index`。

## CUDA QTile Kernel

CUDA qtile sparse attention 主 kernel 是：

```cpp
pagedSparseFlashMQTileKernel<T, HEAD_DIM, Q_TILE, K_TILE, DLANES>
```

当前主要 variant：

```text
HD64  Q4 K16
HD128 Q4 K16
HD128 Q4 K8
HD128 Q8 K16
```

CUDA block 组织：

```text
threadIdx.x = dim lane，通常 16
threadIdx.y = q lane，4 或 8 个 active Q rows
grid.x      = ceil(attnLen / Q_TILE)
grid.y      = numHeads
grid.z      = batch
```

一个 block 处理一个 batch、一个 attention head、一组 active Q rows，并在 K 方向循环多个 `K_TILE`。

## OpenCL MQTile Kernel

OpenCL mqtile sparse flash 与 CUDA 语义一致，但 dispatch 前会先做布局预处理：

1. `rearrange_q` 把 Q 转成 sparse flash kernel 需要的 packed layout。
2. `pack_paged_kv_prefill` 或 `pack_paged_k_prefill` 按 slot table 把 PagedCache K/V pack 成连续 buffer。
3. Adreno 上部分 variant 会把 K 或 K/V 放入 image，利用 texture cache。
4. mqtile kernel 按 `Q_TILE x K_TILE` 计算 sparse flash attention。

OpenCL 典型 variant：

```text
mqtile_hd64_q4k16
mqtile_hd128_q4k16
mqtile_hd128_q4k8
mqtile_hd128_q8k16
mqtile_hd128_q4k8_kimg / kvimg
mqtile_hd128_q8k16_kimg / kvimg
```

## Causal 空 Tile 跳过

qtile kernel 对每个 Q tile 先统计 active rows 的 logical 范围：

```text
q_min_logical = min(q_logical in current Q tile)
q_max_logical = max(q_logical in current Q tile)
```

然后得到当前 Q tile 的 K 方向循环上限：

```text
causal_k_limit = min(kvLen, q_max_logical + 1)
```

K tile 循环只执行：

```text
for k_base in [0, causal_k_limit) step K_TILE
```

因此所有满足下面条件的 K tile 会被整块跳过：

```text
k_base >= q_max_logical + 1
```

这些 tile 对当前 Q tile 内所有 query 都 causal 不可见，因为最大的 `q_logical` 也看不到它们。

剩下的 K tile 分两类。

### 全可见 Tile

如果一个 K tile 的结束位置不超过 Q tile 内最小 logical position：

```text
k_tile_end <= q_min_logical
```

则这个 K tile 对当前 Q tile 内所有 query 都可见。kernel 用 `k_tile_full` 标记该情况，后续 score 计算不需要逐个 `(q, k)` 判断 causal 可见性。

### 部分可见 Tile

如果 K tile 落在：

```text
q_min_logical < k <= q_max_logical
```

则它不能整块跳过，因为 Q tile 里至少有部分 query row 可以看到其中的 K。kernel 对每个 `(q, k)` 做判断：

```text
k_visible = k_tile_full || k_index <= q_logical
```

不可见位置的 score 保持 `-inf`，softmax 后权重为 0。

举例：

```text
Q tile logical rows = [100, 130, 135, 500]
```

- `K > 500` 的 tile 整块跳过。
- `K tile end <= 100` 的 tile 全可见。
- `K = 101..500` 范围内的 tile 只能部分可见，不能整块跳过。

## FlashAttention 式计算

每个 K tile 内部执行：

1. 加载 K/V tile。
2. 每个 dim lane 计算一段 head_dim 的 partial dot。
3. 对 dim lanes 做 reduce，得到当前 `(q, k)` score。
4. 对每个 Q row 做 online softmax：

```text
tile_max  = max(score in current K tile)
merged_m  = max(old_m, tile_max)
alpha     = exp(old_m - merged_m)
beta      = exp(score - merged_m)
row_l     = old_l * alpha + sum(beta)
out       = out * alpha + beta @ V_tile
```

5. K 循环结束后写：

```text
output[q_index] = out / row_l
```

因此 qtile sparse attention 不需要落完整 QK 矩阵。

## OpenCL Piece 级裁剪

OpenCL dispatch 还有一层 piece 级优化。`_buildFixedSparsePieces` / `_buildRangeAwareSparsePieces` 会把 active Q rows 切成若干 piece，并为每个 piece 计算：

```text
activeKvLen = min(kvLen, maxLogicalInPiece + 1)
```

这个 `activeKvLen` 会作为 kernel 的 `key_seq_len` 传入，先在 dispatch 层减少 K 方向工作量。kernel 内部仍会按当前 Q tile 的 `q_tile_max_logical + 1` 再收紧一次。

`_buildRangeAwareSparsePieces` 会按 4-row group 的 max logical 做动态规划切分，在增加有限 piece 数的前提下，尽量减少：

```text
piece_q_groups * ceil(activeKvLen / 4)
```

这对 active rows logical position 分布很散的 cacheblend 场景有帮助。

## 性能含义

qtile sparse attention 的实际省算量不只取决于 selected row 数，还取决于 active rows 的 logical 分布：

- 如果 active rows 都在较靠前的位置，`q_max_logical` 小，右侧大量 K tile 会整块跳过。
- 如果 active rows 很散，且每个 Q tile 中混入靠后的 logical row，`q_max_logical` 会很大，K 方向仍接近 full scan。
- 如果一个 tile 的 `q_min_logical` 和 `q_max_logical` 差距大，中间大量 K tile 是部分可见，不能整块跳过，只能靠 `k_index <= q_logical` mask 掉不可见 score。

因此看 sparse attention 性能时，不能只看 active row count / recompute ratio，还要看：

```text
active logical distribution
q_tile 内 q_min/q_max 差距
qk_active_tiles / qk_rect_tiles
```

低 ratio 不一定很快；如果 selected rows 集中在靠后的 logical position，平均 causal K work 仍可能很高。

## 正确性约束

qtile sparse attention 必须保持以下约束：

- `sparse_query[q_index]` 是真实 logical position。
- PagedCache slot table 是唯一 K/V 访问入口。
- score layer full-Q / compact-output 和 later layer compact-Q / compact-output 都必须支持。
- causal 规则按 logical position 判断，不能按 compact index 判断。
- 不可见 score 必须等价于 `-inf`。
- 输出 row 是 compact `q_index`，不是 logical position。

只满足 Q 行减少而不满足上述逻辑，会导致 PIC sparse recompute 与 full prompt attention 语义不一致。
