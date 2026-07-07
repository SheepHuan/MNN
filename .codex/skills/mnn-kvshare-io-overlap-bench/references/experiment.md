# KVShare IO Overlap Experiment

Use this reference when running device experiments for KVShare asynchronous layer-wise hydrate, exposed IO, hidden IO, and cache-size thresholds.

## Implementation Model

Describe the implementation as a layer-wise pipeline:

```text
Naive:
Layer i:   disk read KV_i -> hydrate KV_i -> attention/MLP compute_i
Layer i+1: disk read KV_i+1 -> hydrate KV_i+1 -> attention/MLP compute_i+1

KVShare:
Async loader: read KV_i+1 / KV_i+2 / ...
Accelerator: hydrate KV_i -> sparse attention/MLP compute_i
Next layer:  consume prefetched KV_i+1 -> hydrate -> compute_i+1
```

Persistent PIC cache is stored per layer as `.k/.v` files. Key is `canonical_no_rope`; value keeps the original layout. At request time the runtime binds those files as `PagedKVExternalSegment` entries in `PagedKVMeta::external_segments`.

The OpenCL path schedules future layer reads via `_scheduleExternalLayerReadsFrom(...)`, using `std::async(std::launch::async, ...)` for later layers. When a PagedAttention layer needs external KV, `hydrateExternalSegments(layerIndex, kvLen)` consumes a ready future. If the future is not ready, the main path blocks and the wait is counted as exposed IO.

On UMA devices, frame this as mapped memory, not disk-to-GPU DMA: CPU reads persistent `.k/.v` ranges into mapped PagedCache or source slots; GPU/OpenCL/CUDA kernels then re-apply RoPE for canonical keys and hydrate value into the current request PagedCache.

## Ablation Matrix

Run these variants when supported by the current code. If an env switch does not exist yet, either add it with clear logging or mark the variant unsupported.

| Variant | Suggested setting | Purpose |
|---|---|---|
| Sync hydrate | `MNN_PAGED_ATTENTION_PREFETCH_WINDOW=0` | Serial baseline: every layer reads when needed. |
| Async hydrate | Default prefetch window | Show disk read and sparse compute overlap. |
| Async + direct mapped PagedCache | `ZERO_COPY_CACHE=1` or current native mapped path | Show direct UMA mapped target contribution. |
| Async + staging fallback | Disable direct mapped path or set explicit fallback env | Quantify staging copy overhead. |
| Full recompute | No persistent PIC cache read | Compute-heavy reference. |
| Full reuse | Hydrate + suffix prefill only | Most sensitive IO-loading case. |
| KVShare sparse | `kvshare` at 10% and 20% | Show overlap when sparse compute provides hiding window. |

Do not mix unsupported fallback rows into default results. Label them as `fallback` or `unsupported`.

## Sweep Dimensions

Default matrix:

| Dimension | Values |
|---|---|
| Device | Jetson CUDA, OrangePi OpenCL, Rhino/Adreno OpenCL |
| Context tokens | 512, 1024, 1536, 2048 |
| Recompute ratio | 0%, 10%, 20% |
| Prefetch window | 0, 1, 2, 4, unlimited |
| Mode | full-reuse, kvshare, full-recompute |

Use the same persistent text cache for rows that share device/model/context. Each ratio still sends an independent chat request and recomputes its own scoring/top-k when applicable.

## Required Instrumentation

Add or verify per-layer logs before collecting final numbers. A machine-readable JSONL row is preferred.

Required request-level fields:

```text
run_id
device
model
backend
precision
context_tokens
pic_cache_bytes
mode
recompute_ratio
prefetch_window
direct_mapped
staging_fallback
prefill_ms
ttft_ms
total_disk_read_ms
exposed_io_ms
hidden_io_ms
overlap_ratio
async_read_layers
total_external_layers
async_read_layer_ratio
hydrate_ms
rerope_ms
sparse_compute_ms
suffix_compute_ms
status
```

Required layer-level fields:

```text
layer
kv_bytes
read_start_ms
read_end_ms
read_ms
future_ready_on_consume
future_wait_ms
hydrate_start_ms
hydrate_end_ms
hydrate_ms
compute_start_ms
compute_end_ms
compute_ms
source_path_k
source_path_v
```

If GPU kernel timing is not available, use explicit `unavailable` and still report CPU-visible request timing. Do not infer kernel timing from unrelated total latency.

## Metrics

Compute:

```text
total_disk_read_ms = sum(layer.read_ms)
exposed_io_ms = sum(layer.future_wait_ms)
hidden_io_ms = max(total_disk_read_ms - exposed_io_ms, 0)
overlap_ratio = 1 - exposed_io_ms / total_disk_read_ms
async_read_layer_ratio = async_read_layers / total_external_layers
```

Clamp `overlap_ratio` to `0..1` only after preserving raw values for debugging. If `total_disk_read_ms == 0`, report overlap as `NA`.

For threshold analysis, compute these per device/model/mode:

```text
latency_growth_vs_prev = prefill_ms(ctx_n) / prefill_ms(ctx_prev)
latency_delta_vs_sync = prefill_ms(async) - prefill_ms(sync)
io_exposed_share = exposed_io_ms / prefill_ms
hydrate_share = hydrate_ms / prefill_ms
```

Define the significant prefill-rise threshold as the first context or `pic_cache_bytes` where either:

- `prefill_ms` grows by at least 20% versus the previous context at the same variant, or
- `io_exposed_share >= 0.15`, or
- async latency is within 5% of sync latency for two consecutive contexts, meaning overlap is saturated.

If the paper needs a stricter rule, report both the default threshold and the raw curve.

## Recommended Tables

Main cross-device table:

| Device | Backend | Model | Context | PIC cache MB | Mode | Window | Hidden IO ms | Exposed IO ms | Overlap % | Prefill ms | Threshold note |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|

Threshold table:

| Device | Model | Mode | First significant context | KV cache MB | Exposed IO share | Overlap % | Recommendation |
|---|---|---|---:|---:|---:|---:|---|

Layer aggregation table:

| Device | Model | Context | Mode | Avg read ms/layer | Avg wait ms/layer | P95 wait ms | Async layer ratio | Avg hydrate ms/layer |
|---|---|---|---|---:|---:|---:|---:|---:|

## Recommended Figures

Use a two-part figure for paper section 5.3:

- Fig. 5.3(a), timeline: compare sync read-hydrate-compute against async layer-wise read overlapped with current layer hydrate+compute.
- Fig. 5.3(b), latency breakdown: stacked bars for exposed disk read, hidden disk read, hydrate/re-RoPE, sparse compute, and suffix compute.

Do not mix this figure with the global TTFT speedup figure. This experiment explains why disk-backed KV reuse is practical on real edge devices.

## Validity Checks

Before accepting a run:

1. Confirm logs do not contain `Cache invalid`, `target unavailable`, `async persistent PIC cache read failed`, or `ERROR`.
2. Confirm OpenCL tuning is warm for the tested shape and ratio.
3. Confirm `prefetch_window=0` disables async overlap and produces nonzero exposed IO for nontrivial contexts.
4. Confirm async variants show at least some ready futures on contexts where compute time can hide IO.
5. Confirm full-reuse has no scoring or sparse recompute cost.
6. Confirm KVShare ratios are independent requests and include request-local scoring/top-k when enabled.

## Reporting Language

Use this phrasing:

```text
KVShare does not merely reduce the number of recomputed tokens; it restructures the cache-loading path so that persistent KV movement is overlapped with sparse layer execution, turning disk-backed KV reuse into a practical edge-side runtime rather than a serialized IO bottleneck.
```

Avoid claiming disk-to-GPU DMA unless the device path actually implements and verifies that transport.
