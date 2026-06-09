# PagedAttention V1 Long Context TPS Tables

Date: 2026-06-09

TPS 口径：`effective_tps = prompt_tokens / latency_s`。这里的 TPS 是 prefill-only 的等效吞吐，用来和 normal LLM full-compute prefill 对比。

## Jetson CUDA 强制 V1

证据等级：严格确认。`pic_server` 以 `MNN_PAGED_ATTENTION_IMPL=v1` 启动。

prompt tokens: `1669`

normal LLM full-compute: `13.953s`, `119.61 TPS`

| mode | ratio | latency_s | effective_tps | speedup_vs_normal |
| --- | ---: | ---: | ---: | ---: |
| normal LLM full-compute | - | 13.953 | 119.61 | 1.00x |
| PIC full-reuse | - | 0.793 | 2103.78 | 17.59x |
| PIC full-compute | - | 13.121 | 127.20 | 1.06x |
| cacheblend | 1% | 2.450 | 681.21 | 5.70x |
| cacheblend | 5% | 2.394 | 697.20 | 5.83x |
| cacheblend | 10% | 3.096 | 539.10 | 4.51x |
| cacheblend | 20% | 4.104 | 406.65 | 3.40x |
| cacheblend | 30% | 5.187 | 321.76 | 2.69x |
| epic | 1% | 1.093 | 1527.30 | 12.77x |
| epic | 5% | 1.586 | 1052.03 | 8.80x |
| epic | 10% | 2.080 | 802.48 | 6.71x |
| epic | 20% | 3.292 | 507.01 | 4.24x |
| epic | 30% | 4.391 | 380.10 | 3.18x |

结论：Jetson 上强制 V1 后，cacheblend / epic 的 1%-30% 全部高于 normal `119.61 TPS`。

## OrangePi OpenCL 强制 V1

证据等级：本轮新测。已重新交叉编译并同步 OrangePi artifact，远端 `libMNN_CL.so` 已确认包含 `MNN_PAGED_ATTENTION_IMPL` 路由字符串；`pic_server` 以 `MNN_PAGED_ATTENTION_IMPL=v1` 启动。

target: `1024`

prompt tokens: `949`

normal LLM full-compute: `20.998s`, `45.20 TPS`

raw summary: `.cache/v1_long_context_orangepi_current_1024/summary_orangepi_v1_1024_clean.json`

| mode | ratio | latency_s | effective_tps | speedup_vs_normal |
| --- | ---: | ---: | ---: | ---: |
| normal LLM full-compute | - | 20.998 | 45.20 | 1.00x |
| PIC full-reuse | - | 3.732 | 254.28 | 5.63x |
| cacheblend | 1% | 7.894 | 120.22 | 2.66x |
| cacheblend | 5% | 33.147 | 28.63 | 0.63x |
| cacheblend | 10% | 98.791 | 9.61 | 0.21x |
| cacheblend | 20% | 198.548 | 4.78 | 0.11x |
| epic | 1% | 4.885 | 194.26 | 4.30x |
| epic | 5% | 14.339 | 66.18 | 1.46x |
| epic | 10% | 27.320 | 34.74 | 0.77x |
| epic | 20% | 72.772 | 13.04 | 0.29x |

简化结论：Jetson 强制 V1 的 cacheblend / epic TPS 明确高于 normal；OrangePi 强制 V1 下只有 full-reuse、cacheblend 1%、epic 1/5% 快于 normal，cacheblend 5% 起和 epic 10% 起都慢于 normal。
