# PagedAttention V1 Long Context Detailed Tables

Date: 2026-06-09

## Evidence Level

| device | backend | status | note |
| --- | --- | --- | --- |
| Jetson | CUDA | strict forced-V1 confirmed | `pic_server` launched with `MNN_PAGED_ATTENTION_IMPL=v1`; results below are from this run. |
| OrangePi | OpenCL | not strictly confirmed in this run | SSH banner times out and current deployed OpenCL artifact could not be verified as forced V1. OrangePi tables below are historical, non-forced-V1 references. |

## Jetson CUDA Forced V1

Normal baseline: real normal LLM export, `llm_bench -a cuda -p 1669 -n 0 -rep 3`, `normal_s = 13.953`.

PIC config: `.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct/config_cuda_greedy.json`

Normal config: `.cache/mnn-llm-export/AI-ModelScope__Llama-3___2-3B-Instruct/config_cuda_greedy.json`

Raw JSON: `.cache/v1_long_context_confirm/summary_jetson_v1_2048.json`

| device | backend | prompt_tokens | pic_tokens | mode | ratio | score_layer | execution_mode | recompute | reuse | latency_s | normal_s | speedup_vs_normal | pic_full_compute_s | speedup_vs_pic_full_compute | full_reuse_s | latency_over_full_reuse |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jetson | cuda | 1669 | 1621 | full-reuse | - | 0 | native-full-reuse | 0 | 1621 | 0.793 | 13.953 | 17.59x | 13.121 | 16.54x | 0.793 | 1.00x |
| jetson | cuda | 1669 | 1621 | pic-full-compute | - | 28 | native-full-compute | 1621 | 0 | 13.121 | 13.953 | 1.06x | 13.121 | 1.00x | 0.793 | 16.54x |
| jetson | cuda | 1669 | 1621 | cacheblend | 1% | 1 | native-cacheblend-sparse-recompute | 17 | 1604 | 2.450 | 13.953 | 5.70x | 13.121 | 5.36x | 0.793 | 3.09x |
| jetson | cuda | 1669 | 1621 | cacheblend | 5% | 1 | native-cacheblend-sparse-recompute | 82 | 1539 | 2.394 | 13.953 | 5.83x | 13.121 | 5.48x | 0.793 | 3.02x |
| jetson | cuda | 1669 | 1621 | cacheblend | 10% | 1 | native-cacheblend-sparse-recompute | 163 | 1458 | 3.096 | 13.953 | 4.51x | 13.121 | 4.24x | 0.793 | 3.90x |
| jetson | cuda | 1669 | 1621 | cacheblend | 20% | 1 | native-cacheblend-sparse-recompute | 325 | 1296 | 4.104 | 13.953 | 3.40x | 13.121 | 3.20x | 0.793 | 5.17x |
| jetson | cuda | 1669 | 1621 | cacheblend | 30% | 1 | native-cacheblend-sparse-recompute | 487 | 1134 | 5.187 | 13.953 | 2.69x | 13.121 | 2.53x | 0.793 | 6.54x |
| jetson | cuda | 1669 | 1621 | epic | 1% | 1 | native-epic-sparse-recompute | 17 | 1604 | 1.093 | 13.953 | 12.77x | 13.121 | 12.01x | 0.793 | 1.38x |
| jetson | cuda | 1669 | 1621 | epic | 5% | 1 | native-epic-sparse-recompute | 82 | 1539 | 1.586 | 13.953 | 8.80x | 13.121 | 8.27x | 0.793 | 2.00x |
| jetson | cuda | 1669 | 1621 | epic | 10% | 1 | native-epic-sparse-recompute | 163 | 1458 | 2.080 | 13.953 | 6.71x | 13.121 | 6.31x | 0.793 | 2.62x |
| jetson | cuda | 1669 | 1621 | epic | 20% | 1 | native-epic-sparse-recompute | 325 | 1296 | 3.292 | 13.953 | 4.24x | 13.121 | 3.99x | 0.793 | 4.15x |
| jetson | cuda | 1669 | 1621 | epic | 30% | 1 | native-epic-sparse-recompute | 487 | 1134 | 4.391 | 13.953 | 3.18x | 13.121 | 2.99x | 0.793 | 5.53x |

## OrangePi OpenCL Historical Reference: Synthetic Long Context

Source: `.cache/pic_latency_long_context/summary.json`

Important: this is historical and was not explicitly run with `MNN_PAGED_ATTENTION_IMPL=v1`.

| source | device | backend | target | prompt_tokens | pic_tokens | mode | ratio | execution_mode | recompute | reuse | latency_s | normal_s | speedup_vs_normal | pic_full_compute_s | speedup_vs_pic_full_compute | full_reuse_s | latency_over_full_reuse |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pic_latency_long_context | orangepi | opencl | 512 | 305 | 289 | full-reuse | - | native-full-reuse | 0 | 289 | 3.944 | 10.194 | 2.58x | 10.927 | 2.77x | 3.944 | 1.00x |
| pic_latency_long_context | orangepi | opencl | 512 | 305 | 289 | pic-full-compute | - | native-full-compute | 289 | 0 | 10.927 | 10.194 | 0.93x | 10.927 | 1.00x | 3.944 | 2.77x |
| pic_latency_long_context | orangepi | opencl | 512 | 305 | 289 | cacheblend | 1% | native-cacheblend-sparse-recompute | 3 | 286 | 5.421 | 10.194 | 1.88x | 10.927 | 2.02x | 3.944 | 1.37x |
| pic_latency_long_context | orangepi | opencl | 512 | 305 | 289 | cacheblend | 5% | native-cacheblend-sparse-recompute | 15 | 274 | 4.947 | 10.194 | 2.06x | 10.927 | 2.21x | 3.944 | 1.25x |
| pic_latency_long_context | orangepi | opencl | 512 | 305 | 289 | cacheblend | 10% | native-cacheblend-sparse-recompute | 29 | 260 | 5.860 | 10.194 | 1.74x | 10.927 | 1.86x | 3.944 | 1.49x |
| pic_latency_long_context | orangepi | opencl | 512 | 305 | 289 | cacheblend | 20% | native-cacheblend-sparse-recompute | 58 | 231 | 6.749 | 10.194 | 1.51x | 10.927 | 1.62x | 3.944 | 1.71x |
| pic_latency_long_context | orangepi | opencl | 512 | 305 | 289 | epic | 1% | native-epic-sparse-recompute | 3 | 286 | 4.047 | 10.194 | 2.52x | 10.927 | 2.70x | 3.944 | 1.03x |
| pic_latency_long_context | orangepi | opencl | 512 | 305 | 289 | epic | 5% | native-epic-sparse-recompute | 15 | 274 | 4.642 | 10.194 | 2.20x | 10.927 | 2.35x | 3.944 | 1.18x |
| pic_latency_long_context | orangepi | opencl | 512 | 305 | 289 | epic | 10% | native-epic-sparse-recompute | 29 | 260 | 5.718 | 10.194 | 1.78x | 10.927 | 1.91x | 3.944 | 1.45x |
| pic_latency_long_context | orangepi | opencl | 512 | 305 | 289 | epic | 20% | native-epic-sparse-recompute | 58 | 231 | 6.152 | 10.194 | 1.66x | 10.927 | 1.78x | 3.944 | 1.56x |
| pic_latency_long_context | orangepi | opencl | 1024 | 593 | 577 | full-reuse | - | native-full-reuse | 0 | 577 | 3.929 | 18.971 | 4.83x | 22.267 | 5.67x | 3.929 | 1.00x |
| pic_latency_long_context | orangepi | opencl | 1024 | 593 | 577 | pic-full-compute | - | native-full-compute | 577 | 0 | 22.267 | 18.971 | 0.85x | 22.267 | 1.00x | 3.929 | 5.67x |
| pic_latency_long_context | orangepi | opencl | 1024 | 593 | 577 | cacheblend | 1% | native-cacheblend-sparse-recompute | 6 | 571 | 7.111 | 18.971 | 2.67x | 22.267 | 3.13x | 3.929 | 1.81x |
| pic_latency_long_context | orangepi | opencl | 1024 | 593 | 577 | cacheblend | 5% | native-cacheblend-sparse-recompute | 29 | 548 | 6.958 | 18.971 | 2.73x | 22.267 | 3.20x | 3.929 | 1.77x |
| pic_latency_long_context | orangepi | opencl | 1024 | 593 | 577 | cacheblend | 10% | native-cacheblend-sparse-recompute | 58 | 519 | 7.440 | 18.971 | 2.55x | 22.267 | 2.99x | 3.929 | 1.89x |
| pic_latency_long_context | orangepi | opencl | 1024 | 593 | 577 | cacheblend | 20% | native-cacheblend-sparse-recompute | 116 | 461 | 10.453 | 18.971 | 1.81x | 22.267 | 2.13x | 3.929 | 2.66x |
| pic_latency_long_context | orangepi | opencl | 1024 | 593 | 577 | epic | 1% | native-epic-sparse-recompute | 6 | 571 | 4.917 | 18.971 | 3.86x | 22.267 | 4.53x | 3.929 | 1.25x |
| pic_latency_long_context | orangepi | opencl | 1024 | 593 | 577 | epic | 5% | native-epic-sparse-recompute | 29 | 548 | 6.567 | 18.971 | 2.89x | 22.267 | 3.39x | 3.929 | 1.67x |
| pic_latency_long_context | orangepi | opencl | 1024 | 593 | 577 | epic | 10% | native-epic-sparse-recompute | 58 | 519 | 6.727 | 18.971 | 2.82x | 22.267 | 3.31x | 3.929 | 1.71x |
| pic_latency_long_context | orangepi | opencl | 1024 | 593 | 577 | epic | 20% | native-epic-sparse-recompute | 116 | 461 | 9.800 | 18.971 | 1.94x | 22.267 | 2.27x | 3.929 | 2.49x |
| pic_latency_long_context | orangepi | opencl | 2048 | 1169 | 1153 | full-reuse | - | native-full-reuse | 0 | 1153 | 5.641 | 38.519 | 6.83x | 163.389 | 28.97x | 5.641 | 1.00x |
| pic_latency_long_context | orangepi | opencl | 2048 | 1169 | 1153 | pic-full-compute | - | native-full-compute | 1153 | 0 | 163.389 | 38.519 | 0.24x | 163.389 | 1.00x | 5.641 | 28.97x |
| pic_latency_long_context | orangepi | opencl | 2048 | 1169 | 1153 | cacheblend | 1% | native-cacheblend-sparse-recompute | 12 | 1141 | 9.114 | 38.519 | 4.23x | 163.389 | 17.93x | 5.641 | 1.62x |
| pic_latency_long_context | orangepi | opencl | 2048 | 1169 | 1153 | cacheblend | 5% | native-cacheblend-sparse-recompute | 58 | 1095 | 8.754 | 38.519 | 4.40x | 163.389 | 18.66x | 5.641 | 1.55x |
| pic_latency_long_context | orangepi | opencl | 2048 | 1169 | 1153 | cacheblend | 10% | native-cacheblend-sparse-recompute | 116 | 1037 | 10.958 | 38.519 | 3.52x | 163.389 | 14.91x | 5.641 | 1.94x |
| pic_latency_long_context | orangepi | opencl | 2048 | 1169 | 1153 | cacheblend | 20% | native-cacheblend-sparse-recompute | 231 | 922 | 19.989 | 38.519 | 1.93x | 163.389 | 8.17x | 5.641 | 3.54x |
| pic_latency_long_context | orangepi | opencl | 2048 | 1169 | 1153 | epic | 1% | native-epic-sparse-recompute | 12 | 1141 | 5.672 | 38.519 | 6.79x | 163.389 | 28.81x | 5.641 | 1.01x |
| pic_latency_long_context | orangepi | opencl | 2048 | 1169 | 1153 | epic | 5% | native-epic-sparse-recompute | 58 | 1095 | 7.997 | 38.519 | 4.82x | 163.389 | 20.43x | 5.641 | 1.42x |
| pic_latency_long_context | orangepi | opencl | 2048 | 1169 | 1153 | epic | 10% | native-epic-sparse-recompute | 116 | 1037 | 9.634 | 38.519 | 4.00x | 163.389 | 16.96x | 5.641 | 1.71x |
| pic_latency_long_context | orangepi | opencl | 2048 | 1169 | 1153 | epic | 20% | native-epic-sparse-recompute | 231 | 922 | 19.552 | 38.519 | 1.97x | 163.389 | 8.36x | 5.641 | 3.47x |

## OrangePi OpenCL Historical Reference: Mixed Long GPU Cases

Source: `.cache/standard_prefill_latency_long_gpu_20260605_081004/combined_summary.json`

Important: this is also historical and was not explicitly run with `MNN_PAGED_ATTENTION_IMPL=v1`. It is included because it shows the cases where `cacheblend` high ratios become slower than normal full-compute on OrangePi OpenCL.

| source | device | backend | model | case | prompt_tokens | pic_tokens | mode | ratio | execution_mode | recompute | reuse | latency_s | normal_s | speedup_vs_normal | pic_full_compute_s | speedup_vs_pic_full_compute | full_reuse_s | latency_over_full_reuse |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_512 | 617 | 561 | full-compute | - | native-full-compute | 561 | 0 | 15.114 | 13.273 | 0.88x | 15.114 | 1.00x | 4.546 | 3.32x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_512 | 617 | 561 | full-reuse | - | native-full-reuse | 0 | 561 | 4.546 | 13.273 | 2.92x | 15.114 | 3.32x | 4.546 | 1.00x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_512 | 617 | 561 | cacheblend | 1% | native-cacheblend-sparse-recompute | 6 | 555 | 9.914 | 13.273 | 1.34x | 15.114 | 1.52x | 4.546 | 2.18x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_512 | 617 | 561 | cacheblend | 5% | native-cacheblend-sparse-recompute | 29 | 532 | 9.558 | 13.273 | 1.39x | 15.114 | 1.58x | 4.546 | 2.10x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_512 | 617 | 561 | cacheblend | 10% | native-cacheblend-sparse-recompute | 57 | 504 | 10.014 | 13.273 | 1.33x | 15.114 | 1.51x | 4.546 | 2.20x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_512 | 617 | 561 | cacheblend | 20% | native-cacheblend-sparse-recompute | 113 | 448 | 13.615 | 13.273 | 0.97x | 15.114 | 1.11x | 4.546 | 2.99x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_512 | 617 | 561 | cacheblend | 30% | native-cacheblend-sparse-recompute | 169 | 392 | 15.592 | 13.273 | 0.85x | 15.114 | 0.97x | 4.546 | 3.43x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_512 | 617 | 561 | epic | 1% | native-epic-sparse-recompute | 6 | 555 | 5.807 | 13.273 | 2.29x | 15.114 | 2.60x | 4.546 | 1.28x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_512 | 617 | 561 | epic | 5% | native-epic-sparse-recompute | 29 | 532 | 6.599 | 13.273 | 2.01x | 15.114 | 2.29x | 4.546 | 1.45x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_512 | 617 | 561 | epic | 10% | native-epic-sparse-recompute | 57 | 504 | 6.938 | 13.273 | 1.91x | 15.114 | 2.18x | 4.546 | 1.53x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_512 | 617 | 561 | epic | 20% | native-epic-sparse-recompute | 113 | 448 | 9.221 | 13.273 | 1.44x | 15.114 | 1.64x | 4.546 | 2.03x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_512 | 617 | 561 | epic | 30% | native-epic-sparse-recompute | 169 | 392 | 11.751 | 13.273 | 1.13x | 15.114 | 1.29x | 4.546 | 2.58x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_1024 | 1121 | 1065 | full-compute | - | native-full-compute | 1065 | 0 | 28.142 | 24.509 | 0.87x | 28.142 | 1.00x | 5.836 | 4.82x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_1024 | 1121 | 1065 | full-reuse | - | native-full-reuse | 0 | 1065 | 5.836 | 24.509 | 4.20x | 28.142 | 4.82x | 5.836 | 1.00x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_1024 | 1121 | 1065 | cacheblend | 1% | native-cacheblend-sparse-recompute | 11 | 1054 | 11.131 | 24.509 | 2.20x | 28.142 | 2.53x | 5.836 | 1.91x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_1024 | 1121 | 1065 | cacheblend | 5% | native-cacheblend-sparse-recompute | 54 | 1011 | 12.151 | 24.509 | 2.02x | 28.142 | 2.32x | 5.836 | 2.08x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_1024 | 1121 | 1065 | cacheblend | 10% | native-cacheblend-sparse-recompute | 107 | 958 | 17.750 | 24.509 | 1.38x | 28.142 | 1.59x | 5.836 | 3.04x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_1024 | 1121 | 1065 | cacheblend | 20% | native-cacheblend-sparse-recompute | 213 | 852 | 34.417 | 24.509 | 0.71x | 28.142 | 0.82x | 5.836 | 5.90x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_1024 | 1121 | 1065 | cacheblend | 30% | native-cacheblend-sparse-recompute | 320 | 745 | 47.739 | 24.509 | 0.51x | 28.142 | 0.59x | 5.836 | 8.18x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_1024 | 1121 | 1065 | epic | 1% | native-epic-sparse-recompute | 11 | 1054 | 6.556 | 24.509 | 3.74x | 28.142 | 4.29x | 5.836 | 1.12x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_1024 | 1121 | 1065 | epic | 5% | native-epic-sparse-recompute | 54 | 1011 | 7.101 | 24.509 | 3.45x | 28.142 | 3.96x | 5.836 | 1.22x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_1024 | 1121 | 1065 | epic | 10% | native-epic-sparse-recompute | 107 | 958 | 9.395 | 24.509 | 2.61x | 28.142 | 3.00x | 5.836 | 1.61x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_1024 | 1121 | 1065 | epic | 20% | native-epic-sparse-recompute | 213 | 852 | 18.684 | 24.509 | 1.31x | 28.142 | 1.51x | 5.836 | 3.20x |
| standard_prefill_latency_long_gpu | orangepi | opencl | llama3_3b | long_1024 | 1121 | 1065 | epic | 30% | native-epic-sparse-recompute | 320 | 745 | 22.778 | 24.509 | 1.08x | 28.142 | 1.24x | 5.836 | 3.90x |
| standard_prefill_latency_long_gpu | orangepi | opencl | glm_edge_4b | long_512 | 695 | 671 | full-compute | - | native-full-compute | 671 | 0 | 24.902 | 22.168 | 0.89x | 24.902 | 1.00x | 4.969 | 5.01x |
| standard_prefill_latency_long_gpu | orangepi | opencl | glm_edge_4b | long_512 | 695 | 671 | full-reuse | - | native-full-reuse | 0 | 671 | 4.969 | 22.168 | 4.46x | 24.902 | 5.01x | 4.969 | 1.00x |
| standard_prefill_latency_long_gpu | orangepi | opencl | glm_edge_4b | long_512 | 695 | 671 | cacheblend | 1% | native-cacheblend-sparse-recompute | 7 | 664 | 7.006 | 22.168 | 3.16x | 24.902 | 3.55x | 4.969 | 1.41x |
| standard_prefill_latency_long_gpu | orangepi | opencl | glm_edge_4b | long_512 | 695 | 671 | cacheblend | 5% | native-cacheblend-sparse-recompute | 34 | 637 | 10.168 | 22.168 | 2.18x | 24.902 | 2.45x | 4.969 | 2.05x |
| standard_prefill_latency_long_gpu | orangepi | opencl | glm_edge_4b | long_512 | 695 | 671 | cacheblend | 10% | native-cacheblend-sparse-recompute | 68 | 603 | 12.584 | 22.168 | 1.76x | 24.902 | 1.98x | 4.969 | 2.53x |
| standard_prefill_latency_long_gpu | orangepi | opencl | glm_edge_4b | long_512 | 695 | 671 | cacheblend | 20% | native-cacheblend-sparse-recompute | 135 | 536 | 21.163 | 22.168 | 1.05x | 24.902 | 1.18x | 4.969 | 4.26x |
| standard_prefill_latency_long_gpu | orangepi | opencl | glm_edge_4b | long_512 | 695 | 671 | cacheblend | 30% | native-cacheblend-sparse-recompute | 202 | 469 | 26.615 | 22.168 | 0.83x | 24.902 | 0.94x | 4.969 | 5.36x |
| standard_prefill_latency_long_gpu | orangepi | opencl | glm_edge_4b | long_512 | 695 | 671 | epic | 1% | native-epic-sparse-recompute | 7 | 664 | 5.499 | 22.168 | 4.03x | 24.902 | 4.53x | 4.969 | 1.11x |
| standard_prefill_latency_long_gpu | orangepi | opencl | glm_edge_4b | long_512 | 695 | 671 | epic | 5% | native-epic-sparse-recompute | 34 | 637 | 7.719 | 22.168 | 2.87x | 24.902 | 3.23x | 4.969 | 1.55x |
| standard_prefill_latency_long_gpu | orangepi | opencl | glm_edge_4b | long_512 | 695 | 671 | epic | 10% | native-epic-sparse-recompute | 68 | 603 | 9.402 | 22.168 | 2.36x | 24.902 | 2.65x | 4.969 | 1.89x |
| standard_prefill_latency_long_gpu | orangepi | opencl | glm_edge_4b | long_512 | 695 | 671 | epic | 20% | native-epic-sparse-recompute | 135 | 536 | 14.914 | 22.168 | 1.49x | 24.902 | 1.67x | 4.969 | 3.00x |
| standard_prefill_latency_long_gpu | orangepi | opencl | glm_edge_4b | long_512 | 695 | 671 | epic | 30% | native-epic-sparse-recompute | 202 | 469 | 21.377 | 22.168 | 1.04x | 24.902 | 1.16x | 4.969 | 4.30x |
