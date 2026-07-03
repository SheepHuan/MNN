# OrangePi OpenCL PMC Profiling

Status: implemented as an optional MNN OpenCL build path.
Scope: OrangePi 5 Plus / Mali OpenCL `pic_server`, focused on PIC decode and
lagged-attention decode repair bottleneck analysis.

## Goal

Use the optional hardware performance-counter profiling mechanism for selected
OpenCL kernels inside MNN PIC server. The mechanism must answer:

- Which decode-repair sub-kernel is slow after graph/op profile has already
  identified the hot path.
- Whether the kernel is memory-bandwidth bound, cache/local-memory bound,
  compute/occupancy bound, launch/CPU bound, or readback/synchronization bound.
- Whether lagged-attention decode repair loses time in append/decodeKey update,
  transposed-K attention, sparse qtile attention, rank/top-k capture, or prepare.

This is a diagnosis tool only. It is not a formal TPOT benchmark path.

## Non-goals

- Do not hook the OpenCL driver as the first implementation.
- Do not use Android `bhook` on OrangePi Linux. `bhook` is relevant for Android
  PLT hook experiments, not for the OrangePi/Mali production diagnosis path.
- Do not add PMC profiling to formal latency runs.
- Do not add new PIC cache files, staging K/V buffers, scratch `.k/.v`, or a
  second runtime KV cache.
- Do not change decode routing to make PMC easier to collect.

## Data source

Use ARM `libGPUCounters` / `hwcpipe` for Mali counters. Keep it optional:

- Current implementation is a CMake opt-in path: `MNN_OPENCL_PMC_PROFILE=ON`.
- If PMC is disabled at build time, all profiler calls are no-ops and normal
  OpenCL artifacts have no `libGPUCounters` dependency.
- If PMC is enabled but the target GPU/counters are unavailable at runtime, the
  profiler writes a single `unsupported` status record and continues normally.
- Counter names must be discovered at runtime. Do not hardcode a single Mali
  counter database name; select counters by requested name or regex and skip
  missing counters with a warning.

The counter stream is GPU-wide, not per OpenCL event. Kernel attribution is valid
only when the queue is isolated with `finish()` around the target dispatch and no
other process is using the GPU heavily.

## Build / link integration

`libGPUCounters` is designed to be embedded into a CMake build and produces two
libraries: `device` (hardware sampling backend) and `hwcpipe` (query/sampling
frontend). It is valid for Arm Mali / Immortalis GPUs with the Arm commercial
driver.

For MNN artifacts:

- Do not vendor a downloaded `libGPUCounters` checkout into this repository.
  Put any checkout/build output under `.cache/` or an external user-specified
  directory.
- Enable with `MNN_OPENCL_PMC_PROFILE=ON`.
- Use `LIBGPUCOUNTERS_ROOT=/path/to/libGPUCounters` for a source/install root
  containing `hwcpipe/include` and `backend/device/include`.
- If libraries live in a separate CMake build tree, also set
  `LIBGPUCOUNTERS_BUILD_ROOT=/path/to/libGPUCounters/build`.
- Build failure to find `libGPUCounters` must leave normal OpenCL artifacts
  buildable with PMC disabled.
- If linked dynamically, sync `libdevice.so` / `libhwcpipe.so` with the
  OrangePi artifact and extend remote `LD_LIBRARY_PATH` through the existing
  server environment.

The skill should not run `git clone` into tracked directories. If it needs to
prepare `libGPUCounters`, use a cache path such as:

```text
.cache/libGPUCounters/
.cache/output/libGPUCounters/orangepi5plus/
```

Minimal local check pattern:

```bash
cmake -S .cache/libGPUCounters -B .cache/build/libGPUCounters-x64 \
  -DHWCPIPE_WERROR=OFF -DHWCPIPE_BUILD_EXAMPLES=OFF \
  -DHWCPIPE_FRONTEND_ENABLE_TESTS=OFF
cmake --build .cache/build/libGPUCounters-x64 --target hwcpipe --parallel "${JOBS}"

cmake -S . -B .cache/build/mnn/x64_pic_opencl_pmc_check \
  -DMNN_OPENCL=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_BUILD_LLM=ON \
  -DMNN_SEP_BUILD=ON -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_OPENCL_PMC_PROFILE=ON \
  -DLIBGPUCOUNTERS_ROOT=.cache/libGPUCounters \
  -DLIBGPUCOUNTERS_BUILD_ROOT=.cache/build/libGPUCounters-x64
cmake --build .cache/build/mnn/x64_pic_opencl_pmc_check --target MNN_CL --parallel "${JOBS}"
```

## Instrumentation order

1. Run existing MNN attribution first:

```bash
MNN_PIC_GRAPH_PROFILE=1
MNN_PIC_GRAPH_PROFILE_TOP=1000
MNN_PAGED_ATTENTION_PROFILE=1
MNN_PAGED_ATTENTION_PROFILE_DETAIL=1
MNN_PIC_DECODE_REPAIR_PROFILE=1
```

Use this only to find candidate ops. It is not a latency number.

2. Pick one to three implemented scoped kernels / sub-ops from
`PagedAttentionBufExecution.cpp`:

- `decode_prepare_transpose_k`
- `decode_causal_attention`
- `append_decode_key_value_hd128`
- `append_sparse_decode_key_value_hd128`
- `decode_causal_attention_hd128_identity*`
- `decode_causal_attention_hd128_transposed_k_*`
- `decode_causal_attention_hd128_transposed_k_readonly`
- `decode_causal_attention_hd128_transposed_k_sparse_qtile`
- `decode_attention_pic_rank_score_hd128`
- `decode_attention_rank_topk`
- `decode_attention_rank_readback`
- `copy_paged_kv`

3. Enable scoped PMC only for those kernels.

4. If graph profile later shows a hot non-PagedAttention OpenCL kernel, add the
   generic OpenCL wrapper/TLS label path as a second stage. Do not start there.

## Implemented architecture

### `OpenCLPmcProfiler`

The optional profiler lives under `source/backend/opencl/core/`:

```text
OpenCLPmcProfiler.hpp/.cpp
```

Responsibilities:

- Parse environment configuration once.
- Initialize `libGPUCounters` through the optional build flag.
- Enumerate device and available counters.
- Select counter set.
- Own a thread-safe JSONL sink.
- Provide a low-overhead `enabledFor(op, layer, phase)` fast path.
- Provide `beginScope(...)` / `endScope(...)`.

Pseudo API:

```cpp
struct OpenCLPmcScopeMeta {
    const char* op = nullptr;
    const char* phase = nullptr;      // append, attention, rank, prepare, copy, graph_op
    int layer = -1;
    int query = -1;
    int inputQuery = -1;
    int kvLen = -1;
    int baseLogical = -1;
    int lane = 0;
    int qTile = 0;
    int heads = 0;
    int kvHeads = 0;
    int headDim = 0;
    std::vector<uint32_t> gws;
    std::vector<uint32_t> lws;
};

class OpenCLPmcProfiler {
public:
    static OpenCLPmcProfiler& get();
    bool enabledFor(const OpenCLPmcScopeMeta& meta) const;
    uint64_t begin(OpenCLRuntime* runtime, cl::CommandQueue& queue,
                   const OpenCLPmcScopeMeta& meta);
    void end(uint64_t token, OpenCLRuntime* runtime, cl::CommandQueue& queue,
             const OpenCLPmcScopeMeta& meta, cl::Event* event);
};
```

`begin()` and `end()` are no-ops when disabled or not compiled with
`MNN_OPENCL_PMC_PROFILE`.

### Scoped dispatch rule

For explicit PagedAttention sites, the implementation uses strict isolated
sampling:

```cpp
if (pmc.enabledFor(meta)) {
    queue.finish();
    token = pmc.begin(runtime, queue, meta);
}
run3DKernelDefault(kernel, gws, lws, runtime, eventPtr);
if (token) {
    queue.finish();
    pmc.end(token, runtime, queue, meta, eventPtr);
}
```

Keep `finish()` inside PMC scope only. Formal TPOT runs must not enable PMC.

For q=1 record-queue paths, PMC should either:

- Disable record queue for profiled kernels and take the normal explicit dispatch
  path; or
- Record only a coarse `record_queue_replay` scope and mark it as coarse.

Do not claim per-kernel PMC if the actual dispatch is hidden inside record queue
replay and cannot be isolated.

### Event timing

Prefer OpenCL event duration when profiling queue supports it. Always include
wall-clock time as fallback:

```json
"event_us": 830,
"wall_us": 910,
"event_status": "ok|unavailable"
```

PMC deltas should be paired with wall/event timing in the same record.

## Environment interface

Recommended initial variables:

```bash
MNN_PIC_PMC_PROFILE=1
MNN_PIC_PMC_KERNEL_REGEX='decode_causal_attention_hd128_transposed_k_sparse_qtile|append_sparse_decode'
MNN_PIC_PMC_PHASE_REGEX='attention|append|rank|prepare'
MNN_PIC_PMC_LAYER='all'              # all, comma list, or range such as 0-27
MNN_PIC_PMC_COUNTERS='default'       # default, all, or regex/list
MNN_PIC_PMC_STRICT_FINISH=1          # default 1
MNN_PIC_PMC_OUTPUT='/mnt/ssd/code/.cache/mnn_opencl_pic/pmc/decode_repair.jsonl'
MNN_PIC_PMC_MAX_RECORDS=2000
MNN_PIC_PMC_WARMUP_SKIP=1
```

`MNN_PIC_PMC_PROFILE=1` is enough to initialize the profiler, but it should not
record anything if regex/layer filters reject the scope.

Default output path on OrangePi should stay under:

```text
/mnt/ssd/code/.cache/mnn_opencl_pic/pmc/
```

Do not write PMC output to the OrangePi root partition.

## Benchmark command parameterization

The decode and prefill benchmark launchers already forward
`PIC_SWEEP_SERVER_ENV_EXTRA` into the remote `pic_server` process. The skill
should use that variable as the first PMC parameterization mechanism.

Use a small set of shell variables instead of hardcoding PMC options into every
command:

```bash
RUN_ID=pmc_decode_repair_sparse_qtile_orangepi_20260703
PMC_TAG=sparse_qtile_rank
PMC_REMOTE_OUT=/mnt/ssd/code/.cache/mnn_opencl_pic/pmc/${RUN_ID}_${PMC_TAG}.jsonl
PMC_KERNEL_REGEX='decode_causal_attention_hd128_transposed_k_sparse_qtile|append_sparse_decode|decode_attention_rank'
PMC_PHASE_REGEX='attention|append|rank'
PMC_COUNTERS=default
PMC_LAYER=all
```

Then append the PMC controls to the server env:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA="MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1 \
MNN_PIC_PMC_PROFILE=1 \
MNN_PIC_PMC_KERNEL_REGEX='${PMC_KERNEL_REGEX}' \
MNN_PIC_PMC_PHASE_REGEX='${PMC_PHASE_REGEX}' \
MNN_PIC_PMC_LAYER='${PMC_LAYER}' \
MNN_PIC_PMC_COUNTERS='${PMC_COUNTERS}' \
MNN_PIC_PMC_STRICT_FINISH=1 \
MNN_PIC_PMC_OUTPUT='${PMC_REMOTE_OUT}' \
MNN_PIC_PMC_MAX_RECORDS=2000 \
MNN_PIC_PMC_WARMUP_SKIP=1" \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models minicpm5-1b \
  --contexts 512 --pic-repair-tokens 0,1,3,5,7 --pic-max-tokens 16 \
  --repeats 1 --warm-repeats 1 --skip-normal \
  --run-id "${RUN_ID}"
```

Important shell rule: values containing `|`, spaces, or commas that the remote
shell could reinterpret must be quoted inside `PIC_SWEEP_SERVER_ENV_EXTRA`.
Use a double-quoted local assignment and single-quote the remote env values, as
shown above.

PMC runs are separate from formal TPOT runs:

- PMC run id must include `pmc` and the kernel family, for example
  `pmc_decode_repair_sparse_qtile_orangepi_YYYYMMDD`.
- Do not merge PMC run latency into `benchmark.csv`.
- Record the exact `PIC_SWEEP_SERVER_ENV_EXTRA` in the hour's `context.md`
  because current decode experiment `run_config.json` does not record this env.
- After the PMC run, fetch or inspect `${PMC_REMOTE_OUT}` from the OrangePi cache
  path. Keep large raw JSONL files under `.cache/`, not in git.

When the benchmark scripts are updated, expose these as first-class CLI flags
and translate them into the same server env:

```text
--pmc-profile
--pmc-kernel-regex
--pmc-phase-regex
--pmc-layer
--pmc-counters
--pmc-output
--pmc-max-records
--pmc-warmup-skip
```

The CLI implementation must also write the resolved PMC config into
`run_config.json`. Until those flags exist, the skill should use
`PIC_SWEEP_SERVER_ENV_EXTRA`.

## JSONL record shape

Each scoped record is one JSON line. Current implementation uses a `counters`
array so raw counter names and units stay unambiguous:

```json
{
  "pmc_status": "ok",
  "token": 12,
  "op": "decode_causal_attention_hd128_transposed_k_sparse_qtile",
  "phase": "attention",
  "layer": 17,
  "query": 5,
  "input_query": 5,
  "kv_len": 2048,
  "base_logical": 2044,
  "lane": 64,
  "q_tile": 4,
  "heads": 32,
  "kv_heads": 8,
  "head_dim": 128,
  "gws": [64, 2, 32],
  "lws": [64, 1, 1],
  "wall_us": 910,
  "event_us": 830,
  "event_status": "ok",
  "dense_kv_work": 10240,
  "causal_kv_work": 9732,
  "append_count": 5,
  "prepare_len": 2043,
  "decode_prepare_inside_decode": 0,
  "counters": [
    {
      "name": "GPU active cycles",
      "units": "cycles",
      "begin": 1000,
      "end": 124456,
      "delta": 123456
    }
  ]
}
```

Status lines use the same JSONL sink:

```json
{"pmc_status":"ready","reason":"selected_counter_count=8"}
{"pmc_status":"unsupported","reason":"Mali GPU device 0 is missing or unsupported by libGPUCounters"}
```

## Counter selection

`default` should try to include these classes, using runtime counter discovery:

- GPU active / cycles / shader cycles.
- External memory read/write beats or bytes.
- L2 lookup/miss/read/write where available.
- Load/store or varying/instruction counters if exposed by the device.

Do not fail if one class is unavailable. The profiler writes a `ready` status
with the selected counter count at init; inspect the first `ok` record for exact
raw counter names and units.

Derived values are computed offline from the JSONL record and are useful only
when their inputs exist:

- `external_read_bytes_per_kv = read_bytes / max(1, causal_kv_work)`.
- `cycles_per_kv = shader_cycles / max(1, causal_kv_work)`.
- `write_bytes_per_new_token = write_bytes / max(1, query)`.
- `l2_miss_rate = l2_miss / max(1, l2_lookup)`.

For decode repair, PMC records include these software metadata fields when
available:

- `dense_kv_work`
- `causal_kv_work`
- `append_count`
- `prepare_len`
- `decode_prepare_inside_decode`

## Analysis rules

Use PMC after the existing profile has already identified the hot region.

- High `attention_us`, high external reads per KV, high GPU active:
  K/V scan or decodeKey layout is memory-bound. Focus on transposed K, V layout,
  qtile, lane width, and K reuse.
- High `append_us`, high writes:
  append K/V/decodeKey update is expensive. Focus on fused append, coalesced
  stores, and avoiding redundant PagedCache writes.
- High `rank_us`, high readback or low GPU active:
  rank/top-k or score readback is the issue. Focus on GPU-side top-k and compact
  result transfer.
- High wall time, low GPU active:
  launch/CPU/queue/record/readback/synchronization overhead dominates. Look at
  record queue, fixed dispatch, and avoiding per-layer readback.
- High GPU cycles, low external memory:
  kernel math/reduction/local memory is likely the bottleneck. Inspect lane
  utilization, local memory pressure, and register pressure.

If `decode_prepare_inside_decode=1`, the run is invalid regardless of PMC.

## Hooking decision tree

1. **Explicit PagedAttention scoped PMC**: default implementation. Highest signal
   because it has layer/query/kv_len/mode metadata.
2. **MNN OpenCL wrapper scoped PMC**: second stage. Add a thread-local dispatch
   label around graph/op or selected execution sites, then let
   `OpenCLWrapper.cpp::clEnqueueNDRangeKernel` sample matching labels. Use only
   when a hot kernel is outside PagedAttention and explicit instrumentation would
   be too scattered.
3. **LD_PRELOAD OpenCL API hook**: diagnostic escape hatch for black-box audit.
   It can see every `clEnqueueNDRangeKernel` but lacks reliable MNN op/layer
   context unless paired with labels.
4. **Driver hook / Android bhook**: not for OrangePi first pass. Use only for
   Android-specific experiments or when the OpenCL wrapper path cannot observe
   the dispatch.

## Validation workflow

1. Build/run without `MNN_PIC_PMC_PROFILE`; behavior and TPOT must be unchanged.
2. Run profile smoke to choose kernel filters.
3. Run one short PMC profile with `MNN_PIC_PMC_MAX_RECORDS` set, confirm JSONL
   records contain counter deltas and existing decode metadata.
4. Repeat for a stable `(model, context, repair_tokens)` case and compare median
   counter deltas per op/layer.
5. Turn PMC off and run formal TPOT A/B; never report PMC-scoped latency as
   official performance.

## Recommended first use for lagged-attention decode repair

Start with OrangePi, context 512, one model at a time. Use q values mapped from
repair tokens:

```text
repair_tokens=0 -> q=1
repair_tokens=1 -> q=2
repair_tokens=3 -> q=4
repair_tokens=5 -> q=6
repair_tokens=7 -> q=8
```

First PMC pass:

```bash
RUN_ID=pmc_decode_repair_ctx512_orangepi_20260703
PMC_REMOTE_OUT=/mnt/ssd/code/.cache/mnn_opencl_pic/pmc/${RUN_ID}.jsonl
PIC_SWEEP_SERVER_ENV_EXTRA="MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1 \
MNN_PIC_PMC_PROFILE=1 \
MNN_PIC_PMC_KERNEL_REGEX='decode_causal_attention_hd128_transposed_k_sparse_qtile|append_sparse_decode|decode_attention_rank' \
MNN_PIC_PMC_PHASE_REGEX='attention|append|rank' \
MNN_PIC_PMC_LAYER='all' \
MNN_PIC_PMC_COUNTERS='default' \
MNN_PIC_PMC_OUTPUT='${PMC_REMOTE_OUT}' \
MNN_PIC_PMC_MAX_RECORDS=2000" \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models minicpm5-1b \
  --contexts 512 --pic-repair-tokens 0,1,3,5,7 --pic-max-tokens 16 \
  --repeats 1 --warm-repeats 1 --skip-normal \
  --run-id "${RUN_ID}"
```

Then split by one kernel at a time if the first pass shows noisy deltas.
