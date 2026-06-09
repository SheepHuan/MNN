# PagedAttention V1 Long Context Confirmation

Date: 2026-06-09

## Scope

Question: with `PagedAttention` forced to V1 for the whole PIC path, are long-context `cacheblend` and `epic` prefill-only latencies still faster than normal LLM full-compute?

Main baseline is normal non-PIC LLM full-compute prefill from `.cache/mnn-llm-export/...` with `llm_bench -n 0`. PIC server requests use `max_tokens=0`, so decode and sampling are not included.

## V1 Routing Proof

For CUDA, the active route is controlled only by environment:

```cpp
const char* impl = ::getenv("MNN_PAGED_ATTENTION_IMPL");
if (impl == nullptr || impl[0] == '\0' || envValueEquals(impl, "v1")) {
    route.reason = "impl_v1";
    return route;
}
```

Then V2 can only run when `v2Route.useV2` is true:

```cpp
const bool useV2Kernel = v2Route.useV2 && (benchForceV2Kernel || shortQueryProfitable);
```

The Jetson `pic_server` process was launched with:

```text
MNN_PAGED_ATTENTION_IMPL=v1
MNN_PAGED_ATTENTION_ROUTE_LOG=1
```

So this run forces V1 without request-level routing. Route logs are empty because the current CUDA code intentionally does not print the common `reason=impl_v1` path.

## Jetson CUDA Result

Device: Jetson at `192.168.101.192`

Model: `AI-ModelScope__Llama-3___2-3B-Instruct`

PIC config: `.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct/config_cuda_greedy.json`

Normal config: `.cache/mnn-llm-export/AI-ModelScope__Llama-3___2-3B-Instruct/config_cuda_greedy.json`

Actual normal prompt length: 1669 tokens

Normal LLM full-compute prefill: `13.953s` (`llm_bench -a cuda -p 1669 -n 0 -rep 3`)

PIC prompt composition: prelude `36`, PIC `1621`, suffix `12`

| mode | ratio | recompute | reuse | latency_s | speedup_vs_normal |
| --- | ---: | ---: | ---: | ---: | ---: |
| full-reuse | - | 0 | 1621 | 0.793 | 17.59x |
| PIC full-compute | - | 1621 | 0 | 13.121 | 1.06x |
| cacheblend | 1% | 17 | 1604 | 2.450 | 5.70x |
| cacheblend | 5% | 82 | 1539 | 2.394 | 5.83x |
| cacheblend | 10% | 163 | 1458 | 3.096 | 4.51x |
| cacheblend | 20% | 325 | 1296 | 4.104 | 3.40x |
| cacheblend | 30% | 487 | 1134 | 5.187 | 2.69x |
| epic | 1% | 17 | 1604 | 1.093 | 12.77x |
| epic | 5% | 82 | 1539 | 1.586 | 8.80x |
| epic | 10% | 163 | 1458 | 2.080 | 6.71x |
| epic | 20% | 325 | 1296 | 3.292 | 4.24x |
| epic | 30% | 487 | 1134 | 4.391 | 3.18x |

Conclusion for Jetson: confirmed. With `MNN_PAGED_ATTENTION_IMPL=v1`, all tested long-context `cacheblend` and `epic` ratios from 1% to 30% are faster than normal LLM full-compute. The slowest tested sparse case is still `2.69x` faster than normal.

Raw result: `.cache/v1_long_context_confirm/summary_jetson_v1_2048.json`

## OrangePi OpenCL Status

Strict V1 confirmation is not complete for OrangePi in this run.

Observed state:

- `192.168.101.113` responds to ping.
- SSH repeatedly fails with `Connection timed out during banner exchange`.
- Existing HTTP server port `18108` accepts TCP but `/v1/models` times out.
- Local OrangePi OpenCL artifact did not show the new `MNN_PAGED_ATTENTION_IMPL` route string, so I cannot claim the deployed OpenCL binary was forced to V1.

Historical non-forced-V1 data is mixed:

| source | model/case | normal_s | mode | ratio | latency_s | speedup_vs_normal |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `.cache/pic_latency_long_context/summary.json` | OrangePi target 2048 | 38.519 | cacheblend | 10% | 10.958 | 3.52x |
| `.cache/pic_latency_long_context/summary.json` | OrangePi target 2048 | 38.519 | cacheblend | 20% | 19.989 | 1.93x |
| `.cache/pic_latency_long_context/summary.json` | OrangePi target 2048 | 38.519 | epic | 10% | 9.634 | 4.00x |
| `.cache/pic_latency_long_context/summary.json` | OrangePi target 2048 | 38.519 | epic | 20% | 19.552 | 1.97x |
| `.cache/standard_prefill_latency_long_gpu_20260605_081004/combined_summary.json` | Llama3 OpenCL long_1024 | 24.509 | cacheblend | 20% | 34.417 | 0.71x |
| `.cache/standard_prefill_latency_long_gpu_20260605_081004/combined_summary.json` | Llama3 OpenCL long_1024 | 24.509 | cacheblend | 30% | 47.739 | 0.51x |
| `.cache/standard_prefill_latency_long_gpu_20260605_081004/combined_summary.json` | Llama3 OpenCL long_1024 | 24.509 | epic | 20% | 18.684 | 1.31x |
| `.cache/standard_prefill_latency_long_gpu_20260605_081004/combined_summary.json` | Llama3 OpenCL long_1024 | 24.509 | epic | 30% | 22.778 | 1.08x |

OrangePi conclusion: cannot strictly confirm forced-V1 behavior until SSH/server access is restored and the updated OpenCL artifact is deployed. The historical data suggests `epic` is usually still faster, but `cacheblend` is not guaranteed faster at higher ratios on OrangePi.
