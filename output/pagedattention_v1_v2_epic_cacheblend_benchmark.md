# PagedAttention V1/V2 epic/cacheblend 同 case 测速

## 测试设置

- 日期：2026-06-09
- 设备：Jetson `jetson@192.168.101.192`
- artifact：`.cache/output/mnn/artifacts/jetson_cross_cuda`
- 模型：`.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct/config_cuda_greedy.json`
- 数据集：HotpotQA，3 cases
- manifest：`/root/code/kvshare-edge/.cache/mnn-pic-benchmark/runs/pa_v1v2_epic_cacheblend_samecase_manifest/manifest.jsonl`
- 请求：`max_tokens=0`，prefill-only
- PIC 参数：`pic_recompute_ratio=0.20`，`pic_recompute_score_layer_idx=1`
- V1：`MNN_PAGED_ATTENTION_IMPL=v1`
- V2：`MNN_PAGED_ATTENTION_IMPL=v2`

V2 已按当前设计移除额外支持条件：`v2` 和 `auto` 都直接选择 CUDA `row_compressed_mask` kernel。远端 V2 server 进程环境确认包含：

```text
MNN_PAGED_ATTENTION_IMPL=v2
```

Jetson direct-op accuracy 复跑通过，日志确认强制 V2 直接进入 row-compressed mask：

```text
CUDAPagedAttention route layer=0 impl=v2 mode=row_compressed_mask reason=forced_v2 ...
all <bench_ops/cuda/accuracy/PagedAttention> tests passed.
TEST_CASE={"name":"单元测试","failed":0,"passed":3}
```

## 汇总结果

### 初始 all-V2 结果

| algorithm | impl | avg_s | p50_s | p90_s | max_s | cases_ok | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epic | V1 | 2.662 | 2.240 | 4.153 | 4.153 | 3 | 0 |
| epic | V2 | 4.467 | 2.897 | 8.299 | 8.299 | 3 | 0 |
| cacheblend | V1 | 2.210 | 2.313 | 2.605 | 2.605 | 3 | 0 |
| cacheblend | V2 | 4.772 | 5.243 | 5.345 | 5.345 | 3 | 0 |

按 `V1_avg / V2_avg` 作为 V2 相对 V1 的速度倍率：

| algorithm | V1_avg_s | V2_avg_s | V2_speed_vs_V1 | latency_ratio_V2_over_V1 |
| --- | ---: | ---: | ---: | ---: |
| epic | 2.662 | 4.467 | 0.596x | 1.678x slower |
| cacheblend | 2.210 | 4.772 | 0.463x | 2.159x slower |

### targeted online V2 优化后

当前源码的 V2 mode 不再把 dense multi-token prefill 全部替换成 V2 单 kernel，而是：

- dense multi-token prefill：V1 通用 fast path。
- `Qlen == 1`：V2 row-compressed online-softmax kernel。

| algorithm | impl | avg_s | p50_s | p90_s | max_s | cases_ok | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epic | V1 | 2.662 | 2.240 | 4.153 | 4.153 | 3 | 0 |
| epic | old all-V2 | 4.467 | 2.897 | 8.299 | 8.299 | 3 | 0 |
| epic | targeted online V2 | 2.635 | 2.014 | 4.279 | 4.279 | 3 | 0 |
| cacheblend | V1 | 2.210 | 2.313 | 2.605 | 2.605 | 3 | 0 |
| cacheblend | old all-V2 | 4.772 | 5.243 | 5.345 | 5.345 | 3 | 0 |
| cacheblend | targeted online V2 | 2.746 | 3.065 | 3.095 | 3.095 | 3 | 0 |

| algorithm | old_all_V2_avg_s | targeted_online_V2_avg_s | speedup_vs_old_all_V2 | speed_vs_V1 |
| --- | ---: | ---: | ---: | ---: |
| epic | 4.467 | 2.635 | 1.695x | 1.010x |
| cacheblend | 4.772 | 2.746 | 1.738x | 0.805x |

## per-case 延迟

| algorithm | impl | case_id | prompt_tokens | completion_tokens | latency_s |
| --- | --- | --- | ---: | ---: | ---: |
| epic | V1 | `5a8b57f25542995d1e6f1371` | 638 | 0 | 4.153 |
| epic | V1 | `5a8c7595554299585d9e36b6` | 594 | 0 | 1.594 |
| epic | V1 | `5a85ea095542994775f606a8` | 866 | 0 | 2.240 |
| epic | V2 | `5a8b57f25542995d1e6f1371` | 638 | 0 | 8.299 |
| epic | V2 | `5a8c7595554299585d9e36b6` | 594 | 0 | 2.204 |
| epic | V2 | `5a85ea095542994775f606a8` | 866 | 0 | 2.897 |
| cacheblend | V1 | `5a8b57f25542995d1e6f1371` | 638 | 0 | 2.605 |
| cacheblend | V1 | `5a8c7595554299585d9e36b6` | 594 | 0 | 1.713 |
| cacheblend | V1 | `5a85ea095542994775f606a8` | 866 | 0 | 2.313 |
| cacheblend | V2 | `5a8b57f25542995d1e6f1371` | 638 | 0 | 5.345 |
| cacheblend | V2 | `5a8c7595554299585d9e36b6` | 594 | 0 | 3.728 |
| cacheblend | V2 | `5a85ea095542994775f606a8` | 866 | 0 | 5.243 |

### targeted online V2 per-case

| algorithm | impl | case_id | prompt_tokens | completion_tokens | latency_s |
| --- | --- | --- | ---: | ---: | ---: |
| epic | targeted online V2 | `5a8b57f25542995d1e6f1371` | 638 | 0 | 4.279 |
| epic | targeted online V2 | `5a8c7595554299585d9e36b6` | 594 | 0 | 1.614 |
| epic | targeted online V2 | `5a85ea095542994775f606a8` | 866 | 0 | 2.014 |
| cacheblend | targeted online V2 | `5a8b57f25542995d1e6f1371` | 638 | 0 | 3.095 |
| cacheblend | targeted online V2 | `5a8c7595554299585d9e36b6` | 594 | 0 | 2.076 |
| cacheblend | targeted online V2 | `5a85ea095542994775f606a8` | 866 | 0 | 3.065 |

## 结论

当前 V2 的语义路线是对的：它是 exact row-compressed causal/mask attention，不是近似 row sparse，也没有 request 变量或额外 shape 条件。

这组数据对应的是旧版 CUDA V2 验证型 kernel，性能还没吃到收益：

- V2 每个 query row/head 使用一个 block，QK 在 max/sum/output 阶段重复计算，整体比 V1 的 prefill 优化路径重。
- 当前 case 的 query 很短、KV 较长，但 rowEnd 压缩不一定能覆盖 cacheblend/full-reference 和 suffix 中的大量接近全长 attention 行。
- 低层 full-reference/scoring/hydrate/sparse recompute 的端到端开销都计入 latency，V2 attention kernel 本身的节省被调度和 kernel 内重复计算抵消。

后续优化不应恢复 unsupported/fallback 报错，而应优化 V2 kernel 本体和路由语义。当前源码已把 V2 kernel 改成 K tile 内 QK 只算一次，再通过 online softmax 合并 tile 输出；同时 V2 mode 不再把 dense multi-token prefill 替换成 V2 单 kernel，而是保留 V1 通用 fast path。direct-op sweep 显示当前 V2 kernel 只有 `Qlen == 1` 稳定快于 V1，因此生产路由暂时只在这个 case 启用 V2。

这轮结果说明：

- targeted online V2 已经修复 all-V2 把 V1 fast path 替掉导致的大幅回退。
- epic 在同 case 上从 4.467s 降到 2.635s，基本追平/略快于 V1。
- cacheblend 从 4.772s 降到 2.746s，但仍慢于 V1，下一步应重点拆 cacheblend 的 scoring/full-reference 与 sparse recompute 内部 V2 调用占比。
