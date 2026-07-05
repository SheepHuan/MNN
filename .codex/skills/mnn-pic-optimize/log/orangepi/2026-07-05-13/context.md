# Context

User request: try a single fused OpenCL kernel that can optimize PIC dualgraph
decode-repair x=0/1/3/5/7, test the effect, and if ineffective comment out the
new code rather than deleting it.

## Implementation Attempt

The attempted path fused sparse decode K/V append into the existing
headDim=128 transposed-K qtile decode attention family:

- `source/backend/opencl/execution/cl/paged_decode_attention_buf.cl`
  - Used existing `decode_causal_attention_hd128_transposed_k_qtile_q*_row*_fused_append`
    kernel family.
  - Fixed a correctness bug in the final output store: `output_offset` already
    includes `out_d4`, so the final `vstore4` must use `output + output_offset`
    rather than `output + output_offset + out_d4`.
- `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
  - Added a record-queue path for fused sparse qtile decode attention.
  - Tried routing through it only when
    `MNN_PAGED_ATTENTION_DECODE_REPAIR_FUSED_APPEND=1`.
- `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp`
  - Added fused sparse record state for the experiment.

The route preserved the same PIC dualgraph decode-repair family: x0 is active
rows=1, x1 is active rows=2, and nonzero x values remain the same sparse qtile
family. It did not route x0 to true normal LLM decode.

## Build And Sync

Build command:

```bash
JOBS=$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 )) \
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

After the negative result, the fused record route was disabled/commented and the
same build command was run again. Outputs were installed under:

```text
.cache/output/mnn/artifacts/orangepi5plus/
```

Artifact sync:

```bash
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

Verified `pic_server`, `libMNN.so`, `libMNN_CL.so`, and `libpic_llm.so` are
ARM aarch64 binaries.

## Benchmark

Initial MiniCPM/Qwen runs failed before decode because the harness defaulted to
`--suffix-from-cache-tokens 0`, which produced:

```text
PIC prompt suffix tokenization produced no tokens
```

The working runs used `--pic-suffix-from-cache-tokens 1`.

Default A/B command:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_REPAIR_FUSED_APPEND=0' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models qwen3-4b --contexts 1024 \
  --pic-repair-tokens 0,1 --pic-max-tokens 8 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 --warm-repeats 1 --skip-normal \
  --run-id decode_repair_fused_ab_default_orangepi_qwen_ctx1024_x01_20260705_codex
```

Fused A/B command:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_REPAIR_FUSED_APPEND=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models qwen3-4b --contexts 1024 \
  --pic-repair-tokens 0,1 --pic-max-tokens 8 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 --warm-repeats 1 --skip-normal \
  --run-id decode_repair_fused_ab_on_orangepi_qwen_ctx1024_x01_20260705_codex
```

CSV paths:

```text
.cache/mnn-pic-benchmark/decode_experiment/decode_repair_fused_ab_default_orangepi_qwen_ctx1024_x01_20260705_codex/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_repair_fused_ab_on_orangepi_qwen_ctx1024_x01_20260705_codex/decode_tpot_long.csv
```

Result:

```text
default:
  x0 162.86928571428572 ms
  x1 237.751 ms

fused:
  x0 177.58542857142857 ms
  x1 255.33142857142857 ms
```

P0 scans were clean for both default and fused runs: no
`decode_prepare_inside_decode=1`, `ERROR`, `target unavailable`,
`async persistent PIC cache read failed`, or `Cache invalid`.

## Why It Was Not Effective

The fused implementation saved one append dispatch/record entry, but the append
work was moved into the attention workgroup shape. That is unfavorable here:

- The original append kernel is mostly a narrow current-token K/V write.
- The attention kernel is a wide head/query/K scan kernel. Adding append there
  increases register pressure and memory instructions in the hot kernel.
- A correct same-kernel write-then-read path needs global memory visibility
  ordering, which is expensive on Mali OpenCL.
- With GQA, append is naturally per KV head, while attention is per query head.
  A naive fused attention-shaped dispatch can duplicate or serialize work that
  should remain KV-head scoped.
- The original record-queue path already hides part of the launch overhead, so
  the saved launch was smaller than expected.

Net result: fusing append into attention made the dominant attention scan slower
by more than the append launch it removed.

## Current Code Status

Disabled/commented code:

- `_decodeRepairFusedAppendEnabled()` returns `false`; the env branch is inside
  `#if 0`.
- The fused-record dispatch block is inside `#if 0`.
- `runDecodeCausalAttentionHD128TransposedKSparseFusedRecord(...)` is inside
  `#if 0`.
- Fused sparse record state in the header is inside `#if 0`.

Active retained fix:

- The fused kernel output-offset fix remains active in `.cl` and regenerated
  `paged_decode_attention_buf_mnn_cl.cpp`. It only fixes dormant kernel
  correctness and does not enable the failed route.

## Forward Plan For New Session

Do not continue the fused append + attention route as implemented. It regresses
despite removing the separate append dispatch.

Next work should target why default PIC dualgraph x0 is still slower than true
normal decode:

1. Run detail profile and scoped PMC on default two-kernel record path for
   Qwen3-4B ctx1024 x0/x1, with kernels:
   `append_sparse_decode_key_value_hd128`,
   `decode_causal_attention_hd128_transposed_k_sparse_qtile`,
   `decode_attention_pic_rank_score_hd128`, and rank/top-k readback.
2. Confirm whether the remaining delta versus normal decode is GPU memory scan,
   append write traffic, queue/record replay, rank/readback, or dense/logits
   tail. Do not assume attention until profile proves it.
3. Optimize append independently instead of fusing it into attention:
   keep append KV-head scoped, reduce duplicate stores under GQA, vectorize /
   coalesce decodeKey writes, and consider a current-token direct-read path that
   avoids global write-read fencing.
4. Optimize the transposed-K qtile attention kernel directly:
   compare row32/row64/row128 and q1/q2/q4/q8 variants with PMC counters,
   focus on K/V external reads, local barriers, register pressure, and value
   load coalescing.
5. Keep x0/x1/x3/x5/x7 one decode-repair implementation family. x0 may use the
   q1/rows=1 member, while x>0 uses q2/q4/q8 members, but no hidden route to
   true normal decode or old identity attention should enter default results.
6. Fix or override the benchmark harness default: current server requires
   `--pic-suffix-from-cache-tokens 1`; `0` fails before decode with empty suffix
   tokenization.

## Follow-up Default-Path Software Profile

After the fused path was disabled, a default two-kernel path profile was run.
The first run used no warm repeat and showed a cold/debug outlier
(`PicSparseAttention layer=2 max=200 ms`), so attribution was based on the second
run with `--warm-repeats 1`.

Warm profile command:

```bash
RUN_ID=decode_repair_default_profile_warm_orangepi_qwen_ctx1024_x01_20260705_codex2
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_REPAIR_FUSED_APPEND=0 MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=1000 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PIC_DECODE_REPAIR_PROFILE=1 MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models qwen3-4b --contexts 1024 \
  --pic-repair-tokens 0,1 --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 --warm-repeats 1 --skip-normal \
  --run-id "${RUN_ID}"
```

P0 scan was clean: no `decode_prepare_inside_decode=1`, `ERROR`,
`target unavailable`, `async persistent PIC cache read failed`, or
`Cache invalid`.

The formal TPOT columns from this profile run are not official latency because
profile/detail adds synchronization. The useful measure segment was:

```text
x0 / repair_rows=0 / sparse_rows=1:
  forward_raw_ms=257.849
  PagedAttention sub-op total=20.959 ms
  append_us total=2.742 ms
  attention_us total=18.128 ms
  rank_us total=0.034 ms

x1 / repair_rows=1 / sparse_rows=2:
  forward_raw_ms=388.815
  PagedAttention sub-op total=67.836 ms
  append_us total=4.117 ms
  attention_us total=62.052 ms
  rank_us total=1.600 ms
```

Graph callback profile still shows large `Convolution` / graph totals, but those
totals are inflated by debug callback synchronization and can exceed
`forward_raw_ms`. They are useful for spotting unexpected first-run outliers, not
for exact latency accounting. The direct PagedAttention sub-op timers are the
better signal for append/attention/rank split.

## Scoped PMC Diagnostic

The regular OrangePi artifact had `MNN_OPENCL_PMC_PROFILE=OFF`, so a separate
diagnostic artifact was built:

```bash
cmake -S .cache/libGPUCounters -B .cache/build/libGPUCounters-orangepi5plus \
  -DCMAKE_TOOLCHAIN_FILE=$PWD/.cache/toolchains/orangepi5plus-arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu-aarch64-linux.toolchain.cmake \
  -DHWCPIPE_WERROR=OFF -DHWCPIPE_BUILD_EXAMPLES=OFF \
  -DHWCPIPE_FRONTEND_ENABLE_TESTS=OFF
cmake --build .cache/build/libGPUCounters-orangepi5plus --target hwcpipe device --parallel "${JOBS}"
```

Because the OrangePi cross toolchain uses `CMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY`,
the aarch64 libGPUCounters include/lib outputs were symlinked under the toolchain
cache root:

```text
.cache/toolchains/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu/libgpu_pmc/
```

PMC MNN artifact build:

```bash
BUILD_DIR=.cache/build/mnn/orangepi5plus_pmc \
INSTALL_PREFIX=.cache/output/mnn/artifacts/orangepi5plus_pmc \
CLEAN=1 JOBS="${JOBS}" MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS="-DMNN_OPENCL_PMC_PROFILE=ON -DLIBGPUCOUNTERS_ROOT=${PMCPREFIX} -DLIBGPUCOUNTERS_BUILD_ROOT=${PMCPREFIX}" \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

The PMC artifact was temporarily rsynced to the normal OrangePi artifact path,
the PMC run was executed, and then the normal no-PMC artifact was restored from
`.cache/output/mnn/artifacts/orangepi5plus/`.

PMC run:

```bash
RUN_ID=pmc_decode_repair_default_orangepi_qwen_ctx1024_x01_20260705_codex2
PMC_REMOTE_OUT=/mnt/ssd/code/.cache/mnn_opencl_pic/pmc/${RUN_ID}_attention_append_rank.jsonl
PIC_SWEEP_SERVER_ENV_EXTRA="MNN_PAGED_ATTENTION_DECODE_REPAIR_FUSED_APPEND=0 \
MNN_PIC_DECODE_DEBUG=1 \
MNN_PIC_DECODE_REPAIR_PROFILE=1 \
MNN_PIC_PMC_PROFILE=1 \
MNN_PIC_PMC_KERNEL_REGEX='decode_causal_attention_hd128_transposed_k_sparse_qtile|append_sparse_decode|decode_attention_pic_rank_score_hd128|decode_attention_rank_topk|decode_attention_rank_readback' \
MNN_PIC_PMC_PHASE_REGEX='attention|append|rank|rank_score|rank_topk|readback' \
MNN_PIC_PMC_LAYER='all' \
MNN_PIC_PMC_COUNTERS='default' \
MNN_PIC_PMC_STRICT_FINISH=1 \
MNN_PIC_PMC_OUTPUT='${PMC_REMOTE_OUT}' \
MNN_PIC_PMC_MAX_RECORDS=2000 \
MNN_PIC_PMC_WARMUP_SKIP=1" \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models qwen3-4b --contexts 1024 \
  --pic-repair-tokens 0,1 --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 --warm-repeats 1 --skip-normal \
  --run-id "${RUN_ID}"
```

PMC status:

```text
pmc_status=ready selected_counter_count=16
ok records=293
```

Attention PMC medians:

```text
q1/x0 decode_causal_attention_hd128_transposed_k_sparse_qtile:
  n=72
  wall_us median=644
  Load/store external read median=2,171,072 bytes/layer
  Output external read median=2,207,024 bytes/layer
  Any workload active cycles median=1,683,072
  Execution engine starvation cycles median=1,701,311
  Arithmetic issue cycles median=488,608
  GPU active cycles median=465,916

q2/x1 decode_causal_attention_hd128_transposed_k_sparse_qtile:
  n=72
  wall_us median=1,622.5
  Load/store external read median=4,280,416 bytes/layer
  Output external read median=4,324,016 bytes/layer
  Any workload active cycles median=5,605,037.5
  Execution engine starvation cycles median=4,665,204
  Arithmetic issue cycles median=1,379,488
  GPU active cycles median=1,457,802
```

Rank PMC for x1:

```text
decode_attention_pic_rank_score_hd128:
  wall_us median=1269.5
  external read median=2,105,344 bytes
decode_attention_rank_topk:
  wall_us median=372
decode_attention_rank_readback:
  wall_us median=69.5
```

Append PMC wall medians were small but the selected hardware counters showed
negative deltas on very short append scopes, likely due counter reset/wrap or
sampling instability. Use software profile for append timing:

```text
append q1 wall_us median=208 under PMC, but counters invalid
append q2 wall_us median=153 under PMC, but counters invalid
```

## Updated Diagnosis

The current default-path bottleneck for x0 is not rank and not append. The
remaining PIC-over-normal delta is primarily the q1 transposed-K qtile attention
scan over PagedCache/decodeKey:

- warm software profile x0: attention scan `18.128 ms`, append `2.742 ms`,
  rank `0.034 ms`;
- PMC q1 attention: about `2.17 MB` load/store external read per layer and high
  engine-starvation cycles;
- x1 roughly doubles attention external reads and increases attention wall time
  from about `644 us/layer` to `1622 us/layer`.

Next optimization should focus on the attention kernel family itself:

1. For x0/q1, reduce per-layer K/V external read and starvation in
   `decode_causal_attention_hd128_transposed_k_sparse_qtile`.
2. Compare row64/row128 for q1 and q2 using the PMC path, because current q1
   uses lane128/q_tile1 while q2 uses lane64/q_tile2.
3. Investigate K/V reuse across query heads / GQA: q1 currently reads about
   `2.17 MB/layer`, which is larger than the theoretical minimal current
   K/V scan if KV-head reuse were ideal.
4. Keep append as an independent small kernel for now; optimize its stores only
   after attention scan is reduced.
5. Keep x0/x1/x3/x5/x7 in the same transposed-K qtile decode-repair family; do
   not route x0 to true normal decode or old identity attention.

## Lane Force A/B

Existing env controls were used to test whether the lane width alone explains
the x0 overhead:

```text
MNN_PAGED_ATTENTION_DECODE_QTILE_LANE_FORCE=32|64|128
```

Single-run x0/x1 TPOT:

```text
variant,x0_ms,x1_ms
default_old,162.869,237.751
default_repeat2,159.776,235.736
lane64,161.176,237.411
lane32,155.950,237.902
lane128,159.766,232.871
fused_bad,177.585,255.331
```

P0 scans were clean for all lane-force runs.

Because lane32 looked better for x0 in the single run, x0-only `repeats=3` was
run:

```text
default x0 repeat=3: 156.940 ms
lane32  x0 repeat=3: 159.115 ms
```

Conclusion: lane forcing is not a stable optimization. The single-run lane32
improvement was within run-to-run/cache/tune noise. Do not change the default
lane heuristic based on this data.

Rejected/unsupported paths from this hour:

- fused append + attention record path: regressed x0/x1, code commented out.
- lane32/64/128 force: no stable x0 benefit, keep as explicit A/B env only.
- append-focused optimization as P0: append is only a few ms in warm software
  profile; attention scan dominates.

Current target:

- redesign or improve the q1/q2 transposed-K qtile attention scan so it reads
  less external K/V data or reuses K/V across query heads/GQA more effectively.

## True Normal x0 Follow-up

Normal-only run, using the ordinary non-PIC Qwen3-4B export:

```bash
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models qwen3-4b --contexts 1024 \
  --normal-decode-tokens 1,8 \
  --pic-repair-tokens 0 --pic-max-tokens 1 \
  --repeats 3 --warm-repeats 0 --skip-pic \
  --run-id decode_normal_only_orangepi_qwen_ctx1024_n1n8_20260705_codex3
```

Output:

```text
.cache/mnn-pic-benchmark/decode_experiment/decode_normal_only_orangepi_qwen_ctx1024_n1n8_20260705_codex3/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_normal_only_orangepi_qwen_ctx1024_n1n8_20260705_codex3/decode_tpot_wide.csv
```

Result:

```text
true normal x0, generated_tokens=1: 154.962311 ms/token
true normal x0, generated_tokens=8: 155.264327 ms/token
```

This changes the interpretation of the earlier “extra 20 ms” statement. The
current default PIC x0 formal-style repeat from this hour was `156.940 ms`, so
PIC x0 is not consistently +20 ms/token versus true normal on this artifact and
runtime cache. The single-run `162.869 ms` default row is about `+7.6 ms` versus
the true normal `155.264 ms`, while the repeat=3 row is about `+1.7 ms`.

The ~20 ms figure is real, but it is a synchronized debug/profile subpath cost
inside PIC PagedAttention:

```text
PIC x0 / q=1 / 36-layer measure segment:
  total_ms=20.959
  append_ms=2.742
  attention_ms=18.128
  rank_ms=0.034
  lane=128
  q_tile=1
```

Re-parsing the profile log confirmed warm and measure split:

```text
q=1 warm:    total=28.253 ms, append=6.477 ms, attention=21.657 ms, rank=0.033 ms
q=1 measure: total=20.959 ms, append=2.742 ms, attention=18.128 ms, rank=0.034 ms
q=2 warm:    total=56.488 ms, append=4.710 ms, attention=49.961 ms, rank=1.718 ms
q=2 measure: total=67.836 ms, append=4.117 ms, attention=62.052 ms, rank=1.600 ms
```

Normal PagedAttention profile attempt:

```bash
RUN_ID=decode_normal_profile_orangepi_qwen_ctx1024_n1_20260705_codex3
ROOT=/mnt/ssd/code/.cache/mnn_opencl_pic
OUT=$ROOT/pic_prefill_latency_sweep/$RUN_ID/decode_normal/qwen3-4b
cd $ROOT/models/normal/Qwen__Qwen3-4B
env LD_PRELOAD=$ROOT/artifacts/orangepi5plus/lib/libMNN_CL.so \
    LD_LIBRARY_PATH=$ROOT/artifacts/orangepi5plus/lib:${LD_LIBRARY_PATH:-} \
    MNN_LLM_RUNTIME_CACHE_DIR=$ROOT/pic_prefill_latency_sweep/runtime_cache/opencl \
    MNN_PAGED_ATTENTION_PROFILE=1 \
    MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 \
    MNN_PIC_DECODE_DEBUG=1 \
    $ROOT/artifacts/orangepi5plus/bin/llm_bench \
    -m $ROOT/models/normal/Qwen__Qwen3-4B/config_opencl_greedy.json \
    -a opencl -c 2 -t 4 -p 1024 -n 1 -rep 1 -kv true -load false \
    -j "$OUT/normal_decode_profile.json" \
    > "$OUT/normal_decode_profile.log" 2>&1
```

It produced no `OpenCLPagedAttention profile` rows. That is expected after
checking the code: the ordinary normal export uses `OpType_Attention` /
`AttentionBufExecution`, not `PagedAttentionBufExecution`.

A whole-response normal `llm_bench --profile` was also run:

```bash
RUN_ID=decode_normal_graph_profile_all_orangepi_qwen_ctx1024_n1_20260705_codex3
ROOT=/mnt/ssd/code/.cache/mnn_opencl_pic
OUT=$ROOT/pic_prefill_latency_sweep/$RUN_ID/decode_normal/qwen3-4b
cd $ROOT/models/normal/Qwen__Qwen3-4B
env LD_PRELOAD=$ROOT/artifacts/orangepi5plus/lib/libMNN_CL.so \
    LD_LIBRARY_PATH=$ROOT/artifacts/orangepi5plus/lib:${LD_LIBRARY_PATH:-} \
    MNN_LLM_RUNTIME_CACHE_DIR=$ROOT/pic_prefill_latency_sweep/runtime_cache/opencl \
    $ROOT/artifacts/orangepi5plus/bin/llm_bench \
    -m $ROOT/models/normal/Qwen__Qwen3-4B/config_opencl_greedy.json \
    -a opencl -c 2 -t 4 -p 1024 -n 1 -rep 1 -kv true -load false \
    --profile \
    -j "$OUT/normal_decode_graph_profile.json" \
    > "$OUT/normal_decode_graph_profile.log" 2>&1
```

This profile includes the 1024-token prefill, so it cannot isolate decode-only
attention. It does confirm the ordinary normal graph route:

```text
Attention    6088.027832 ms, 144 calls
Convolution 33032.679688 ms, 1012 calls
total time  43325.457031 ms
```

Source-level comparison:

- True normal OpenCL decode: `AttentionBufExecution::decodeResize` builds and
  records `rearrange_k`, `matmul_qk_decode`, `softmax_in1_buf`, `rearrange_v`,
  and `matmul_qkv_decode_b4/b8`. Runtime updates only the KV cache args and
  lengths when record queue is enabled.
- PIC x0 decode-repair: `PagedAttentionBufExecution` sets
  `picDecodeRecompute+sparseQuery`, writes current K/V through
  `append_sparse_decode_key_value_hd128`, then scans with
  `decode_causal_attention_hd128_transposed_k_sparse_qtile` or the q1 row member
  over `decodeKey`, `value_cache`, and `sparseQuery`.
- Therefore true normal has no PagedCache slot/sparse-query/decode-rank
  semantics; PIC x0 carries them because x0 is the active-rows=1 member of the
  x=0/1/3/5/7 decode-repair family.

Updated plan:

1. Do not spend more time on fused append+attention as previously implemented;
   it is already commented out and regressed.
2. Do not promote lane forcing; repeat=3 rejected it.
3. If more evidence is needed, add a runtime-gated detail profile to ordinary
   `AttentionBufExecution` decode, or build a temporary
   `ENABLE_OPENCL_TIME_PROFILER` artifact. That is only for measurement, not an
   optimization.
4. The real optimization target is a new PagedAttention q1/q2 decode-repair
   kernel family that keeps PagedCache/sparseQuery semantics but narrows the
   gap to ordinary `Attention`:
   - avoid or reduce the `decodeKey` transposed copy when it causes extra
     external reads;
   - reuse K/V across GQA groups without the low-occupancy q1 GQA route that was
     already rejected;
   - consider a split QK/softmax/QKV measurement variant for q1 first, then
     design a fused online-softmax kernel only if it reduces external reads;
   - keep x0/x1/x3/x5/x7 in the same explicit decode-repair family and report
     any variant by name.

## Device Policy Split Cleanup

Follow-up cleanup split device-specific selection logic out of the large
PagedAttention execution files without changing runtime policy:

- `source/backend/opencl/execution/buffer/PagedAttentionMaliUtils.{hpp,cpp}`
  now owns Mali-specific OpenCL decisions:
  decode-repair qtile default enablement, HD128 sparse lane choice, HD128 qtile
  choice, later sparse q8/k16 preference, and Mali sparse-flash variant cache
  rejection.
- `source/backend/opencl/execution/buffer/PagedAttentionAdrenoUtils.{hpp,cpp}`
  now also owns Adreno sparse-flash cache/schedule rejection and static sparse
  flash workspace disablement. Existing Adreno image/GEMM/persistent-source
  helpers remain there.
- `source/backend/cuda/execution/PagedAttentionJetsonPolicy.{hpp,cpp}` now owns
  Jetson CUDA sparse qtile policy: variant enum/name, ordinary sparse variant
  selection, decode-repair variant selection, large sparse tile thresholds, wide
  qtile threshold, and prefill q-split threshold.

Rationale:

- The main OpenCL execution file should route through policy helpers instead of
  embedding Adreno/Mali branches around the compute body.
- The CUDA Jetson file should keep kernel launch code local but not carry the
  tuning policy table inline.
- This prepares the next split where prefill-stage compute can move into its
  own compilation unit while keeping per-device choices stable and auditable.

No performance result is attached to this cleanup; it is intended to be
behavior-preserving. A build/reconfigure is still required because new `.cpp`
files were added under OpenCL and CUDA globbed source directories.
