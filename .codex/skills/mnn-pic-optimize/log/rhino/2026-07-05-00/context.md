# Context

## Commands

Synced local OpenCL dualgraph exports to Rhino:

```bash
rsync -a .cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-orangepi-dualgraph/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/models/pic/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-rhino-dualgraph/

rsync -a .cache/weight/Qwen__Qwen3-4B-pic-boundary-orangepi-dualgraph/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/models/pic/Qwen__Qwen3-4B-pic-boundary-rhino-dualgraph/
```

Patched the local decode harness candidate paths so Rhino resolves the new
`*-rhino-dualgraph/config_opencl_greedy.json` before legacy non-dualgraph PIC
boundary configs.

Benchmark:

```bash
MNN_RHINO_SUDO_PASSWORD=<set> \
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512,1024 \
  --normal-decode-tokens 16 \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 2 \
  --warm-repeats 1 \
  --run-id decode_adreno_three_models_vs_normal_ctx512_1024_20260704_161944
```

## TPOT

All values below are `tpot_ms` from `decode_tpot_long.csv`.

```text
model,ctx,normal_x0,pic_x0,pic_x1,pic_x3,pic_x5,pic_x7,pic_x0/normal
MiniCPM5-1B,512,23.219,79.247,87.985,88.419,109.570,112.707,3.41x slower
MiniCPM5-1B,1024,29.214,118.497,129.223,128.662,161.062,163.661,4.06x slower
Llama3.2-3B,512,53.256,156.851,179.459,182.678,235.421,239.763,2.95x slower
Llama3.2-3B,1024,55.824,240.879,266.173,269.229,360.011,360.752,4.32x slower
Qwen3-4B,512,70.256,227.291,258.660,262.987,336.053,340.885,3.24x slower
Qwen3-4B,1024,71.710,362.245,394.323,398.947,512.008,491.333,5.05x slower
```

Family-internal extra over PIC x0:

```text
model,ctx,x1_extra,x3_extra,x5_extra,x7_extra
MiniCPM5-1B,512,+8.738,+9.172,+30.323,+33.460
MiniCPM5-1B,1024,+10.726,+10.164,+42.564,+45.163
Llama3.2-3B,512,+22.607,+25.827,+78.570,+82.911
Llama3.2-3B,1024,+25.295,+28.350,+119.133,+119.873
Qwen3-4B,512,+31.369,+35.696,+108.762,+113.595
Qwen3-4B,1024,+32.078,+36.702,+149.764,+129.088
```

## Interpretation

This is a valid Adreno decode-repair run, but it is not a speedup result. True
normal LLM decode is still much faster than PIC dualgraph x0. The current Adreno
default also does not justify promoting direct C4 GEMV: prior MiniCPM A/B showed
forced `adreno_direct_c4_gemv` regressed x3/x5/x7 versus the default Adreno tiny
family.

Next optimization target is x0 first. If x0 remains slower than normal by 3x-5x,
rows>1 dense work cannot make end-to-end decode-repair acceptable. Profile x0
with graph and PagedAttention detail, then split:

- q1 PagedAttention append/scan and PagedCache access
- graph tail including lm_head/top1 and greedy sampling waits
- Raster/reshape and elementwise overhead in decode-only graph
- launch/record queue overhead

## Adreno x0 qtile default fix

Hypothesis:

The initial Adreno x0 run was valid PIC dualgraph decode-repair, but it did not
enter the transposed-K sparse qtile q1-row route. `PagedAttentionBufExecution.cpp`
defaulted `_decodeRepairSparseQTileDefaultEnabled()` to Mali only. Therefore
Adreno x0 active rows=1 fell back to sparse qsplit attention, adding fixed
PagedAttention overhead that Mali avoided.

### A/B before source change

Default profiled MiniCPM5-1B ctx512 x0:

```bash
RUN_ID=decode_adreno_minicpm_ctx512_x0_default_profile_20260705_0100
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1' \
conda run -n kvshare-edge python .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino --models minicpm5-1b --contexts 512 --skip-normal \
  --pic-repair-tokens 0 --pic-max-tokens 4 --pic-suffix-from-cache-tokens 1 \
  --repeats 1 --warm-repeats 1 --chat-timeout 900 --remote-timeout 9000 \
  --run-id "${RUN_ID}"
```

Result:

```text
tpot_ms=128.253, generated_tokens=4
route: repair_qtile_enabled=0 repair_qtile_candidate=0 repair_qtile_route=0 repair_policy=0
op: sparse_qsplit_attention
decode graph: use_decode_graph=1 pic_decode_repair_decode_graph=1
```

Forced qtile profiled MiniCPM5-1B ctx512 x0:

```bash
RUN_ID=decode_adreno_minicpm_ctx512_x0_force_qtile_profile_20260705_0105
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE=1' \
conda run -n kvshare-edge python .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino --models minicpm5-1b --contexts 512 --skip-normal \
  --pic-repair-tokens 0 --pic-max-tokens 4 --pic-suffix-from-cache-tokens 1 \
  --repeats 1 --warm-repeats 1 --chat-timeout 900 --remote-timeout 9000 \
  --run-id "${RUN_ID}"
```

Result:

```text
tpot_ms=73.382, generated_tokens=4
route: repair_qtile_enabled=1 repair_qtile_candidate=1 repair_qtile_prefix_ready=1 repair_qtile_route=1 repair_policy=2
op: decode_causal_attention_hd128_transposed_k_sparse_qtile
q_tile=1 q1_row=1 decode_prepare_inside_decode=0
```

Forced qtile 16-token TPOT smoke:

```bash
RUN_ID=decode_adreno_minicpm_ctx512_x0_force_qtile_tpot16_20260705_0110
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1 MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE=1' \
conda run -n kvshare-edge python .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino --models minicpm5-1b --contexts 512 --skip-normal \
  --pic-repair-tokens 0 --pic-max-tokens 16 --pic-suffix-from-cache-tokens 1 \
  --repeats 2 --warm-repeats 1 --chat-timeout 900 --remote-timeout 9000 \
  --run-id "${RUN_ID}"
```

Result:

```text
old default full-matrix ctx512 x0: 79.247 ms
forced qtile ctx512 x0:           54.993 ms
sample_tpot_ms:                   3.384 ms
wall_tpot_ms:                     63.418 ms
log scan: no ERROR, target unavailable, async persistent PIC cache read failed, or decode_prepare_inside_decode=1
```

### Source change

Narrow default change:

```text
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp

_decodeRepairSparseQTileDefaultEnabled(OpenCLRuntime* runtime, int attnLen):
  - keep MALI default true for supported rows.
  - add ADRENO default true only for attnLen == 1.
```

This preserves the current design boundary:

- x0 is still PIC dualgraph decode-repair active rows=1, not true normal LLM.
- Adreno x0 uses the q1/rows=1 special implementation.
- Adreno x1/x3/x5/x7 rows>1 remain on the existing default path until separately
  profiled and validated.

Build and sync:

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Build note: the script detected an empty/stale cached C/CXX compiler and sysroot
for `.cache/build/mnn/aidlux_adreno_opencl`, removed that build directory, and
reconfigured with Arm GNU 11.3. The installed outputs were verified as AArch64:

```text
.cache/output/mnn/artifacts/aidlux_adreno_opencl/bin/pic_server
.cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN.so
.cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN_CL.so
.cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libpic_llm.so
```

### Rebuilt default validation

MiniCPM5-1B ctx512 x0, no qtile env override:

```bash
RUN_ID=decode_adreno_minicpm_ctx512_x0_default_after_qtile_gate_20260705_0115
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1' \
conda run -n kvshare-edge python .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino --models minicpm5-1b --contexts 512 --skip-normal \
  --pic-repair-tokens 0 --pic-max-tokens 16 --pic-suffix-from-cache-tokens 1 \
  --repeats 2 --warm-repeats 1 --chat-timeout 900 --remote-timeout 9000 \
  --run-id "${RUN_ID}"
```

MiniCPM5-1B ctx1024 x0, no qtile env override:

```bash
RUN_ID=decode_adreno_minicpm_ctx1024_x0_default_after_qtile_gate_20260705_0120
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1' \
conda run -n kvshare-edge python .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino --models minicpm5-1b --contexts 1024 --skip-normal \
  --pic-repair-tokens 0 --pic-max-tokens 16 --pic-suffix-from-cache-tokens 1 \
  --repeats 2 --warm-repeats 1 --chat-timeout 900 --remote-timeout 9000 \
  --run-id "${RUN_ID}"
```

Results:

```text
ctx,old_default_ms,new_default_ms,sample_ms,wall_ms
512,79.247,53.708,3.368,62.189
1024,118.497,64.038,3.402,75.305
```

Route evidence from rebuilt default logs:

```text
ctx512:
ordinary_q1=0 repair=1 q=1 kv_len=514
repair_qtile_enabled=1 repair_qtile_candidate=1 repair_qtile_prefix_ready=1 repair_qtile_route=1
repair_policy=0 decode_key_ready=513 sparse_prepare_len=513 sparse_append=1

ctx1024:
ordinary_q1=0 repair=1 q=1 kv_len=1026
repair_qtile_enabled=1 repair_qtile_candidate=1 repair_qtile_prefix_ready=1 repair_qtile_route=1
repair_policy=0 decode_key_ready=1025 sparse_prepare_len=1025 sparse_append=1
```

Both rebuilt default runs had no `ERROR`, `target unavailable`,
`async persistent PIC cache read failed`, or `decode_prepare_inside_decode=1` in
the remote server logs.

### Remaining model x0 check

After the MiniCPM smoke, the same rebuilt default was checked on the remaining
two models, x0 only:

```bash
RUN_ID=decode_adreno_llama_qwen_ctx512_1024_x0_default_after_qtile_gate_20260705_0128
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1' \
conda run -n kvshare-edge python .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino --models llama3.2-3b,qwen3-4b --contexts 512,1024 --skip-normal \
  --pic-repair-tokens 0 --pic-max-tokens 16 --pic-suffix-from-cache-tokens 1 \
  --repeats 2 --warm-repeats 1 --chat-timeout 900 --remote-timeout 9000 \
  --run-id "${RUN_ID}"
```

Combined with the MiniCPM rebuilt-default runs:

```text
model,ctx,normal,old_pic_x0,new_pic_x0,old_pic/normal,new_pic/normal,new_vs_old
MiniCPM5-1B,512,23.219,79.247,53.708,3.41x,2.31x,1.48x
MiniCPM5-1B,1024,29.214,118.497,64.038,4.06x,2.19x,1.85x
Llama3.2-3B,512,53.256,156.851,86.500,2.95x,1.62x,1.81x
Llama3.2-3B,1024,55.824,240.879,99.730,4.31x,1.79x,2.42x
Qwen3-4B,512,70.256,227.291,110.759,3.24x,1.58x,2.05x
Qwen3-4B,1024,71.710,362.245,122.108,5.05x,1.70x,2.97x
```

Llama/Qwen route evidence:

```text
ordinary_q1=0 repair=1 q=1
repair_qtile_enabled=1 repair_qtile_candidate=1 repair_qtile_prefix_ready=1 repair_qtile_route=1
repair_policy=0
```

The remaining-model run had no `failures.csv` rows and no `ERROR`,
`target unavailable`, `async persistent PIC cache read failed`, or
`decode_prepare_inside_decode=1` in the remote logs.

### Remaining work

This is a confirmed x0 improvement, not full objective completion:

- keep x1/x3/x5/x7 on their current rows>1 path until separate Adreno evidence
  justifies another default change;
- if x0 is still meaningfully above true normal after this qtile fix, the next
  bottlenecks are graph tail/lm_head+top1, Raster/reshape, and launch/record
  overhead rather than the old sparse qsplit attention route.

## Adreno all-supported-rows qtile alignment

User direction:

```text
rhinopi的路径应该对齐orangpi的mali的实现路径
```

Interpretation: Adreno should not keep a production default where x0 enters the
qtile family but x1/x3/x5/x7 stay on another rows>1 default path. `x=0/1/3/5/7`
are the same PIC dualgraph decode-repair implementation family; x0 is just
active rows=1. For Rhino/Adreno, align the default gate with OrangePi/Mali for
the same supported rows.

### Source change

Current source diff:

```diff
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp

-    return runtime->getGpuType() == GpuType::MALI;
+    return runtime->getGpuType() == GpuType::MALI || runtime->getGpuType() == GpuType::ADRENO;
```

The existing guard remains:

```text
_decodeRepairSparseQTileShapeSupported(runtime, attnLen):
  runtime != nullptr
  0 < attnLen <= 8
  !legacy B863976 OpenCL
```

Therefore Adreno now defaults to qtile for the same decode-repair active rows
covered by Mali: q=1/2/4/6/8, corresponding to x0/x1/x3/x5/x7.

### Build and sync

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Build note: the script detected stale/empty cross compiler and sysroot cache in
`.cache/build/mnn/aidlux_adreno_opencl`, removed the build directory, and
reconfigured with the Aidlux-specific Arm GNU 11.3 toolchain file.

Checked config:

```text
CMAKE_INSTALL_PREFIX=.cache/output/mnn/artifacts/aidlux_adreno_opencl
CMAKE_TOOLCHAIN_FILE=.cache/toolchains/aidlux_adreno_opencl-arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu-aarch64-linux.toolchain.cmake
MNN_OPENCL=ON
MNN_VULKAN=OFF
MNN_CUDA=OFF
toolchain compiler=aarch64-none-linux-gnu-gcc/g++
sysroot=arm-gnu-toolchain-11.3.../aarch64-none-linux-gnu/libc
```

Installed artifact check:

```text
pic_server:    ELF 64-bit LSB executable, ARM aarch64
libMNN.so:     ELF 64-bit LSB shared object, ARM aarch64
libMNN_CL.so:  ELF 64-bit LSB shared object, ARM aarch64
libpic_llm.so: ELF 64-bit LSB shared object, ARM aarch64
```

Synced:

```bash
rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Remote artifact check confirmed AArch64 `pic_server` and `libMNN_CL.so` with the
new timestamp.

### MiniCPM all-row smoke

```bash
RUN_ID=decode_adreno_minicpm_ctx512_qtile_allrows_default_20260704_172439
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1' \
conda run -n kvshare-edge python .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino --models minicpm5-1b --contexts 512 --skip-normal \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 2 --warm-repeats 1 \
  --chat-timeout 900 --remote-timeout 9000 \
  --run-id "$RUN_ID"
```

Result:

```text
x0=54.165 ms
x1=61.663 ms
x3=70.242 ms
x5=98.886 ms
x7=103.101 ms
failures.csv absent
```

Route summary:

```text
q=1 total=1080 route1=1080 route0=0 enabled1=1080 ordinary1=0 record1=0
q=2 total=1080 route1=1080 route0=0 enabled1=1080 ordinary1=0 record1=0
q=4 total=1080 route1=1080 route0=0 enabled1=1080 ordinary1=0 record1=0
q=6 total=1080 route1=1080 route0=0 enabled1=1080 ordinary1=0 record1=0
q=8 total=1080 route1=1080 route0=0 enabled1=1080 ordinary1=0 record1=0
seq_len=1/2/4/6/8 graph_route_count=45 each
```

No `ERROR`, `target unavailable`, `async persistent PIC cache read failed`,
`Cache invalid`, `CL_OUT_OF_RESOURCES`, `decode_prepare_inside_decode=1`,
`repair_qtile_route=0`, or `ordinary_q1=1`.

### Three-model all-row validation

```bash
RUN_ID=decode_adreno_allmodels_ctx512_1024_qtile_allrows_default_20260704_172439
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1' \
conda run -n kvshare-edge python .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino --models minicpm5-1b,llama3.2-3b,qwen3-4b --contexts 512,1024 --skip-normal \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 2 --warm-repeats 1 \
  --chat-timeout 900 --remote-timeout 9000 \
  --run-id "$RUN_ID"
```

Outputs:

```text
.cache/mnn-pic-benchmark/decode_experiment/decode_adreno_allmodels_ctx512_1024_qtile_allrows_default_20260704_172439/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_adreno_allmodels_ctx512_1024_qtile_allrows_default_20260704_172439/decode_tpot_wide.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_adreno_allmodels_ctx512_1024_qtile_allrows_default_20260704_172439/logs/pic_server_decode_minicpm5-1b.log
.cache/mnn-pic-benchmark/decode_experiment/decode_adreno_allmodels_ctx512_1024_qtile_allrows_default_20260704_172439/logs/pic_server_decode_llama3.2-3b.log
.cache/mnn-pic-benchmark/decode_experiment/decode_adreno_allmodels_ctx512_1024_qtile_allrows_default_20260704_172439/logs/pic_server_decode_qwen3-4b.log
```

P0:

```text
long rows: 30 + header
wide rows: 6 + header
failures.csv: absent
error scan: empty
```

Current TPOT:

```text
model,ctx,x0,x1,x3,x5,x7
MiniCPM5-1B,512,54.020,62.074,70.713,99.051,104.077
MiniCPM5-1B,1024,64.640,68.763,86.797,124.069,132.243
Llama3.2-3B,512,87.014,107.552,118.707,189.927,195.877
Llama3.2-3B,1024,98.548,115.965,143.425,219.060,229.175
Qwen3-4B,512,111.503,139.731,154.459,248.829,255.691
Qwen3-4B,1024,121.775,149.947,187.898,296.221,305.274
```

Old default to new default speedup:

```text
model,ctx,x0,x1,x3,x5,x7
MiniCPM5-1B,512,1.47x,1.42x,1.25x,1.11x,1.08x
MiniCPM5-1B,1024,1.83x,1.88x,1.48x,1.30x,1.24x
Llama3.2-3B,512,1.80x,1.67x,1.54x,1.24x,1.22x
Llama3.2-3B,1024,2.44x,2.30x,1.88x,1.64x,1.57x
Qwen3-4B,512,2.04x,1.85x,1.70x,1.35x,1.33x
Qwen3-4B,1024,2.97x,2.63x,2.12x,1.73x,1.61x
```

Route summary:

```text
MiniCPM:
q=1/2/4/6/8 total=2160 each, route1=2160 each, route0=0, ordinary1=0
decode graph seq_len=1/2/4/6/8 count=90 each

Llama3.2-3B:
q=1/2/4/6/8 total=2520 each, route1=2520 each, route0=0, ordinary1=0
decode graph seq_len=1/2/4/6/8 count=90 each

Qwen3-4B:
q=1/2/4/6/8 total=3240 each, route1=3240 each, route0=0, ordinary1=0
decode graph seq_len=1/2/4/6/8 count=90 each
```

Conclusion:

The user-directed path alignment is validated on Rhino for the supported
decode-repair row family. Adreno now follows the same qtile default family as
Mali for x0/x1/x3/x5/x7. This improves every measured row versus the old default,
with the largest wins at ctx1024 and on Qwen/Llama. The remaining performance
gap should be profiled inside this unified family, especially q=6/q=8 plus graph
tail/dense overhead, rather than solved by automatically routing rows>1 to a
different default implementation.

## Adreno record queue A/B and default promotion

After Adreno was aligned with Mali's qtile family for all supported decode-repair
rows, the remaining MiniCPM5-1B ctx512 x0 profile still showed a large fixed
overhead gap:

```text
profile_adreno_x0_residual_minicpm_ctx512_20260705_171339
tpot_ms=321.627 with graph/profile sync enabled; not formal latency
route: use_decode_graph=1, pic_decode_repair_decode_graph=1
route: repair_qtile_route=1, ordinary_q1=0, decode_prepare_inside_decode=0

graph profile measured tokens=3:
Raster        340.991 ms / 1170 calls
Convolution   214.611 ms / 507 calls
PagedAttention 102.593 ms / 72 calls
BinaryOp      100.270 ms / 363 calls
While          80.601 ms / 288 calls
UnaryOp        59.934 ms / 222 calls
LayerNorm      41.539 ms / 147 calls
```

Interpretation: the qtile route was fixed; remaining cost was PagedAttention
append/attention plus graph tail, small-op dispatch, queue, and readback
overhead.

### Diagnostic env hook

Added a PIC runtime hook in `transformers/pic_llm/engine/src/llm.cpp`:

```text
MNN_PIC_OPENCL_RECORD_QUEUE=op     -> MNN_GPU_RECORD_OP
MNN_PIC_OPENCL_RECORD_QUEUE=batch  -> MNN_GPU_RECORD_BATCH
MNN_PIC_OPENCL_RECORD_QUEUE=off    -> clear record queue bits
unset                              -> MNN_GPU_RECORD_OP after default promotion
```

The OpenCL backend only uses this on devices that support Qualcomm recordable
queues, so Mali keeps the same functional route.

### MiniCPM ctx512 x0 A/B

Baseline qtile default:

```text
RUN_ID=decode_adreno_minicpm_ctx512_x0_record_baseline_20260705_0200
server_env_extra=MNN_PIC_DECODE_DEBUG=1
tpot_ms=54.024
sample_tpot_ms=3.378
wall_tpot_ms=62.607
record_queue=0
repair_qtile_route=1
failures.csv absent
```

Record op:

```text
RUN_ID=decode_adreno_minicpm_ctx512_x0_record_op_20260705_0202
server_env_extra=MNN_PIC_DECODE_DEBUG=1 MNN_PIC_OPENCL_RECORD_QUEUE=op
tpot_ms=47.468
sample_tpot_ms=0.978
wall_tpot_ms=55.488
record_queue=1
repair_qtile_route=1
failures.csv absent
speedup vs qtile default: 1.14x
```

Record batch:

```text
RUN_ID=decode_adreno_minicpm_ctx512_x0_record_batch_20260705_0204
server_env_extra=MNN_PIC_DECODE_DEBUG=1 MNN_PIC_OPENCL_RECORD_QUEUE=batch
```

This failed with a closed HTTP response before producing a successful benchmark
row. The server did enter `record_queue=1` and `repair_qtile_route=1`, but this
variant is marked unstable and is not performance data.

### Full record-op matrix

Because the x0 A/B was positive, record-op was tested on the whole Adreno qtile
decode-repair matrix:

```text
RUN_ID=decode_20260705_015956
server_env_extra=MNN_PIC_DECODE_DEBUG=1 MNN_PIC_OPENCL_RECORD_QUEUE=op
devices=rhino
models=minicpm5-1b,llama3.2-3b,qwen3-4b
contexts=512,1024
pic_repair_tokens=0,1,3,5,7
pic_max_tokens=16
pic_suffix_from_cache_tokens=1
repeats=2
warm_repeats=1
```

Note: the first command attempt used `RUN_ID=...` as a one-shot environment
assignment before the command and passed `--run-id "$RUN_ID"` in the same shell
line. Shell expansion happened before that temporary assignment, so the harness
used its automatic run id `decode_20260705_015956`. The run config records the
intended model/context/repair/env matrix correctly.

P0:

```text
long rows: 30 + header
wide rows: 6 + header
failures.csv: absent

MiniCPM log:
record_queue=1: 10800
repair_qtile_route=1: 10800
repair_qtile_route=0 / ordinary_q1=1 / decode_prepare_inside_decode=1: 0
errors: 0

Llama log:
record_queue=1: 12600
repair_qtile_route=1: 12600
repair_qtile_route=0 / ordinary_q1=1 / decode_prepare_inside_decode=1: 0
errors: 0

Qwen log:
record_queue=1: 16200
repair_qtile_route=1: 16200
repair_qtile_route=0 / ordinary_q1=1 / decode_prepare_inside_decode=1: 0
errors: 0

pic_decode_rhino_*.csv:
execution_mode={'native-full-reuse-hydrate-suffix': 10 each}
decode_runtime={'mnn_token_id_sparse_decode': 10 each}
benchmark_status={'ok': 10 each}
```

Current best TPOT:

```text
model,ctx,x0,x1,x3,x5,x7
MiniCPM5-1B,512,47.764,52.282,58.526,87.317,90.177
MiniCPM5-1B,1024,56.439,58.193,76.751,117.582,122.399
Llama3.2-3B,512,73.981,93.201,108.505,174.666,180.143
Llama3.2-3B,1024,83.540,100.285,129.203,212.656,221.757
Qwen3-4B,512,96.612,123.788,139.742,233.287,233.203
Qwen3-4B,1024,108.852,135.159,175.784,279.865,286.755
```

Remaining PIC x0 gap to true normal:

```text
model,ctx,true_normal_x0_ms,record_op_pic_x0_ms,pic_x0_over_true_normal
MiniCPM5-1B,512,23.219,47.764,2.057x
MiniCPM5-1B,1024,29.214,56.439,1.932x
Llama3.2-3B,512,53.256,73.981,1.389x
Llama3.2-3B,1024,55.824,83.540,1.497x
Qwen3-4B,512,70.256,96.612,1.375x
Qwen3-4B,1024,71.710,108.852,1.518x
```

### Default promotion and rebuilt smoke

`transformers/pic_llm/engine/src/llm.cpp` now defaults PIC OpenCL runtime config
to `MNN_GPU_RECORD_OP` when `MNN_PIC_OPENCL_RECORD_QUEUE` is unset. Explicit env
still supports:

```text
off/0/false/none -> clear record queue bits
op/1/true        -> MNN_GPU_RECORD_OP
batch            -> MNN_GPU_RECORD_BATCH
```

Build and sync:

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

The build reconfigured because cached C/CXX compiler and sysroot entries were
empty, then produced AArch64 artifacts with OpenCL ON, Vulkan/CUDA OFF.

Default smoke without `MNN_PIC_OPENCL_RECORD_QUEUE`:

```text
RUN_ID=decode_adreno_minicpm_ctx512_x0_record_default_20260705_0215
server_env_extra=MNN_PIC_DECODE_DEBUG=1
MiniCPM5-1B ctx512 x0 tpot_ms=48.043
sample_tpot_ms=1.054
wall_tpot_ms=56.224
record_queue=1: 1080
record_queue=0: 0
repair_qtile_route=1: 1080
ordinary_q1=1 / decode_prepare_inside_decode=1 / errors: 0
```

Conclusion: Adreno now defaults to the Mali-aligned qtile decode-repair family
and Qualcomm record-op queue. This is the current best validated Rhino/Adreno
decode-repair path, but it does not complete true-normal parity: MiniCPM x0 is
still about 2x normal, while Llama/Qwen x0 are about 1.4x-1.5x normal.

## q1 identity fused-KV candidate

Question:

PIC x0 and true normal q=1 are close in algorithm shape: append current K/V,
scan historical K, softmax, then QV. The important difference is K layout:

```text
true normal identity fused-KV:
  key_cache + k * 128 + d4
  each lane reads a contiguous 4-float vector for one historical token.

PIC transposed-K qtile:
  decode_key + d * key_max_len + k
  one lane reading one fixed d walks contiguous k, but a lane's 128-dim vector
  for the same token is across key_max_len stride.
```

Transposed K exists for rows>1 / q_tile sharing: multiple active Q rows can reuse
the same K stream. For x0 (`attnLen == 1`), that reuse does not exist, so Adreno
may prefer the normal identity fused-KV contiguous K pattern.

### Implementation

Added an explicit env-gated candidate:

```text
MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_IDENTITY_FUSED_KV=1
```

Files:

```text
source/backend/opencl/execution/cl/paged_decode_attention_buf.cl
source/backend/opencl/execution/cl/paged_decode_attention_buf_mnn_cl.cpp
source/backend/opencl/execution/cl/opencl_source_map.hpp
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
```

Kernel family:

```text
decode_causal_attention_hd128_sparse_identity_fused_kv_row32
decode_causal_attention_hd128_sparse_identity_fused_kv_row64
decode_causal_attention_hd128_sparse_identity_fused_kv_row128
```

The kernel stays in PIC decode-repair semantics:

- requires `attnLen == 1`
- uses `sparse_query[0]` as the real logical slot
- appends current K/V into `key_cache` and `value_cache`
- also writes current K into `decode_key`, so later repair family state remains
  valid
- computes QK from contiguous `key_cache + k * 128`
- reads V from the existing `value_cache`
- supports the same record-op path as qtile

After editing `.cl`:

```bash
( cd source/backend/opencl/execution/cl && python3 opencl_codegen.py . )
```

Build and sync:

```bash
JOBS=96 \
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

### Adreno A/B

Candidate ctx512:

```text
RUN_ID=decode_adreno_minicpm_ctx512_x0_sparse_identity_fused_seq_20260705_0256
server_env_extra=MNN_PIC_DECODE_DEBUG=1 MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_IDENTITY_FUSED_KV=1
tpot_ms=47.3017666667
wall_tpot_ms=54.7373125
wall_minus_decode_tpot_ms=10.39190625
failures.csv=absent
route counts:
  ordinary_q1=0: 1080
  record_queue=1: 1080
  repair_q1_identity_fused_kv=1: 1080
  repair_qtile_route=1: 1080
```

Default ctx512, same artifact:

```text
RUN_ID=decode_adreno_minicpm_ctx512_x0_default_seq_20260705_0259
server_env_extra=MNN_PIC_DECODE_DEBUG=1
tpot_ms=48.4792666667
wall_tpot_ms=55.96071875
wall_minus_decode_tpot_ms=10.51140625
failures.csv=absent
route counts:
  ordinary_q1=0: 1080
  record_queue=1: 1080
  repair_q1_identity_fused_kv=0: 1080
  repair_qtile_route=1: 1080
```

Candidate ctx1024:

```text
RUN_ID=decode_adreno_minicpm_ctx1024_x0_sparse_identity_fused_seq_20260705_0302
server_env_extra=MNN_PIC_DECODE_DEBUG=1 MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_IDENTITY_FUSED_KV=1
tpot_ms=53.1802333333
wall_tpot_ms=63.756875
wall_minus_decode_tpot_ms=13.90040625
failures.csv=absent
route counts:
  ordinary_q1=0: 1080
  record_queue=1: 1080
  repair_q1_identity_fused_kv=1: 1080
  repair_qtile_route=1: 1080
```

Default ctx1024, same artifact:

```text
RUN_ID=decode_adreno_minicpm_ctx1024_x0_default_seq_20260705_0304
server_env_extra=MNN_PIC_DECODE_DEBUG=1
tpot_ms=56.3080666667
wall_tpot_ms=66.45646875
wall_minus_decode_tpot_ms=13.66765625
failures.csv=absent
route counts:
  ordinary_q1=0: 1080
  record_queue=1: 1080
  repair_q1_identity_fused_kv=0: 1080
  repair_qtile_route=1: 1080
```

Summary:

```text
ctx,default_qtile_ms,q1_identity_fused_kv_ms,delta_ms,speedup
512,48.479,47.302,-1.178,1.025x
1024,56.308,53.180,-3.128,1.059x
```

Two intermediate 02:49 runs are intentionally ignored because they were started
in parallel on the same server/local port pair:

```text
decode_adreno_minicpm_ctx512_x0_sparse_identity_fused_warm_20260705_0249
decode_adreno_minicpm_ctx512_x0_default_same_artifact_20260705_0249
```

The earlier 02:42 candidate run reported `48.3786 ms` but had a `Cache invalid,
will be reset` line, so the clean sequential 02:56/02:59 pair above is the
comparison point.

### Negative or bounded alternatives

Record queue A/B on PIC x0:

```text
run_id,variant,tpot_ms
decode_adreno_minicpm_ctx512_x0_record_baseline_20260705_0200,record off,54.0244666667
decode_adreno_minicpm_ctx512_x0_record_op_20260705_0202,record op,47.4677333333
```

Record-op helps PIC x0, so it remains default. A separate true normal A/B showed
the opposite direction:

```text
true normal llm_bench -t 4:   about 24.03 ms/token
true normal llm_bench -t 260: about 46.94 ms/token
```

So the remaining PIC gap is not explained by normal lacking record queue.

Fused append A/B:

```text
RUN_ID=decode_adreno_minicpm_ctx512_x0_fused_append_20260705_0358
server_env_extra=MNN_PIC_DECODE_DEBUG=1 MNN_PIC_OPENCL_RECORD_QUEUE=op MNN_PAGED_ATTENTION_DECODE_REPAIR_FUSED_APPEND=1
tpot_ms=66.8258333333
```

This is a regression versus split append + attention. Keep fused append as
negative evidence for Adreno x0.

Lane A/B:

```text
run_id,lane,tpot_ms
decode_adreno_minicpm_ctx512_x0_lane32_20260705_0408,32,53.6825666667
decode_adreno_minicpm_ctx512_x0_lane64_20260705_0408,64,49.7038
default lane128-ish,128,about 48 ms
```

Lane width tuning is not the main remaining lever for x0.

Graph profile with record-op and PagedAttention detail disabled:

```text
RUN_ID=profile_adreno_x0_record_op_nodeetail_minicpm_ctx512_20260705_0342
tpot_ms=84.2433333333
generated_tokens=4

MNN_PIC_GRAPH_PROFILE_TYPE:
PagedAttention total_ms=71.160 calls=72
Convolution   total_ms=6.523  calls=507
Raster        total_ms=3.807  calls=1170
ArgMax        total_ms=3.251  calls=4
BinaryOp      total_ms=1.471  calls=363
While         total_ms=1.116  calls=288
UnaryOp       total_ms=0.696  calls=222
LayerNorm     total_ms=0.419  calls=147
```

This profile is not formal latency because graph profile perturbs timing, but it
does attribute the remaining cost: PagedAttention dominates, while fixed graph
tail is secondary but still visible in wall-minus-decode.

### Decision

The q1 identity fused-KV candidate is a real Adreno x0 improvement but too small
to explain the full gap:

```text
MiniCPM ctx512 true normal x0: 23.219 ms
MiniCPM ctx512 best candidate: 47.302 ms

MiniCPM ctx1024 true normal x0: 29.214 ms
MiniCPM ctx1024 best candidate: 53.180 ms
```

Interpretation:

- contiguous K read helps Adreno more at longer context, which supports the
  memory-layout hypothesis
- most remaining gap is still PIC dualgraph/PagedAttention family overhead:
  sparse query metadata, current request PagedCache/decodeKey maintenance,
  repair route, and scanning hydrated PIC KV through the PIC kernel family
- graph tail remains around 10-14 ms/token in the CSV wall-minus-decode columns,
  but profile shows PagedAttention as the largest component

Do not promote this env-gated candidate to default yet. It needs at least:

1. three-model Adreno x0 ctx512/1024 A/B
2. OrangePi/Mali x0 ctx512/1024 A/B with the same env gate
3. a decision that x0-only q1 identity is acceptable as a named implementation
   variant while x1/x3/x5/x7 remain transposed-K qtile

Full rows>1 adaptation is not this kernel. It would require a separate
`identity_qtile` family:

- active row loop over `attnLen` 2/4/6/8
- K read from contiguous `key_cache`
- either duplicate K loads per active row or introduce local/shared K staging
  inside a q_tile
- preserve sparse logical indices and current-request PagedCache writes
- still maintain `decode_key` if subsequent kernels or prepares depend on it
- record-op update state and profile/PMC metadata for the new family

This is likely a larger tradeoff: rows>1 is exactly where transposed K was meant
to amortize K loads across Q rows. A full identity rows>1 path may improve one
device/shape but regress the repair family unless it is separately tuned.

For Mali, do not assume the Adreno x0 result transfers. Mali may benefit from
the transposed layout because lanes walking the same `d` over consecutive `k`
are more naturally coalesced. Identity K gives each lane a contiguous vector for
one token, but lanes are separated by the 128-dim token stride. The right next
step is an OrangePi env-gated A/B, not a default route change.

## q1 GQA fused A/B

Implemented explicit env-gated variant:

```text
MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_GQA=1
MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_GQA_FUSED=1
```

Kernel family:

```text
decode_causal_attention_hd128_transposed_k_sparse_gqa_row32
decode_causal_attention_hd128_transposed_k_sparse_gqa_row64
decode_causal_attention_hd128_transposed_k_sparse_gqa_row128
```

The route keeps PIC dualgraph decode-repair semantics: append still writes
`key_cache`, `value_cache`, and `decode_key`; attention still reads transposed
`decode_key`; only the workgroup z dimension changes from `batch * num_heads` to
`batch * kv_heads`, computing the GQA group inside one workgroup.

Smoke:

```text
decode_adreno_q1_gqa_smoke_minicpm_ctx512_suffix1_20260705_0320
```

P0:

```text
ordinary_q1=0
repair_qtile_route=1
repair_q1_gqa=1
decode_prepare_inside_decode=0
op=decode_causal_attention_hd128_transposed_k_sparse_gqa fired
no ERROR / CL error
```

Formal-ish x0 A/B, profile off:

```text
default: decode_adreno_three_models_ctx512_1024_x0_default_suffix1_20260705_0322
gqa:     decode_adreno_three_models_ctx512_1024_x0_gqa_suffix1_20260705_0326
```

Result:

```text
model,ctx,default_ms,gqa_ms,gqa_minus_default_ms,gqa_speedup_vs_default
Llama3.2 3B,512,72.297,73.580,+1.283,0.983x
Llama3.2 3B,1024,81.910,85.277,+3.366,0.961x
MiniCPM5-1B,512,46.967,58.558,+11.591,0.802x
MiniCPM5-1B,1024,54.473,75.927,+21.454,0.717x
Qwen3-4B,512,95.448,98.695,+3.247,0.967x
Qwen3-4B,1024,107.799,115.606,+7.807,0.932x
```

PagedAttention profile detail, MiniCPM ctx512:

```text
default qtile: calls=360 total=431.948 ms append=162.003 ms attention=266.064 ms rank=2.130 ms per_attention_call=0.739 ms
gqa fused:     calls=360 total=607.076 ms append=173.011 ms attention=429.443 ms rank=2.594 ms per_attention_call=1.193 ms
```

Conclusion:

GQA fused reuse is not the Adreno x0 root cause. The regression is in the
attention body, not append or rank. Reducing z workgroups from `num_heads` to
`kv_heads` hurts occupancy/parallelism enough to dominate any K reuse, especially
for MiniCPM where the GQA group is large and KV heads are few.

## Current root-cause framing

PIC x0 should be compared against true normal decode semantically, but the
implementation is not identical:

```text
true normal q1:
  ordinaryDecodeFusedKV -> decode_causal_attention_hd128_identity_fused_kv
  fused current K/V write
  QK reads key_cache as contiguous 128-dim rows

PIC x0:
  picDecodeRecompute + sparseQuery -> repair_qtile_route
  append_sparse_decode_key_value_hd128
  decode_causal_attention_hd128_transposed_k_sparse_qtile
  optional decode_attention_rank metadata path
  QK reads decode_key as [batch, kv_head, dim, logical]
```

Both use the current request PagedCache, but PIC x0 additionally keeps sparse
query/repair metadata, writes `decode_key`, and scans K through a layout designed
for rows>1/q-tile reuse. On Adreno q=1 this transposed-K read path appears to be
the most plausible remaining PagedAttention gap. The q1 identity fused-KV A/B
supports the direction only weakly because it improves MiniCPM by a few ms but
regresses other models in the three-model run.

Decision:

- keep qtile + record queue as the current Adreno default
- keep q1 identity fused-KV and GQA as explicit A/B only
- remove half-implemented identity attention-only routing before further work
- next experiment should be an explicit Adreno image/K-read A/B for the PIC
  repair qtile family, not more default identity-buffer routing

## Adreno no-transposed-K qtile family

User direction:

```text
考虑为adreno做这个no transposed k的kernel famliy
```

Implemented as an explicit env-gated Adreno-only family:

```text
MNN_PAGED_ATTENTION_DECODE_REPAIR_IDENTITY_QTILE=1
MNN_PAGED_ATTENTION_DECODE_REPAIR_NO_TRANSPOSED_K=1
```

Implementation scope:

- C++ gate `_decodeRepairIdentityQTileEnabled(runtime, attnLen)` only returns
  true for Adreno and `1 <= attnLen <= 8`.
- `runDecodeCausalAttentionHD128TransposedKSparse()` keeps the normal
  decode-repair qtile route but selects the new identity-qtiler kernel and passes
  `mCache->key` as the attention K buffer.
- record queue caches `identityKeyCache` as part of the record key, so the
  default transposed path and the identity K path cannot reuse the wrong recorded
  kernel args.
- OpenCL source adds q1/q2/q4/q8 x row32/64/128 kernels:

```text
decode_causal_attention_hd128_identity_qtile_q1_row32
decode_causal_attention_hd128_identity_qtile_q1_row64
decode_causal_attention_hd128_identity_qtile_q1_row128
decode_causal_attention_hd128_identity_qtile_q2_row32
decode_causal_attention_hd128_identity_qtile_q2_row64
decode_causal_attention_hd128_identity_qtile_q2_row128
decode_causal_attention_hd128_identity_qtile_q4_row32
decode_causal_attention_hd128_identity_qtile_q4_row64
decode_causal_attention_hd128_identity_qtile_q4_row128
decode_causal_attention_hd128_identity_qtile_q8_row32
decode_causal_attention_hd128_identity_qtile_q8_row64
decode_causal_attention_hd128_identity_qtile_q8_row128
```

The kernel does not route to true normal decode. It still runs:

```text
append_sparse_decode_key_value_hd128
decode_causal_attention_hd128_identity_qtile_*
```

and append still writes `key_cache`, `value_cache`, and `decode_key`. The only
changed part is attention QK K-read layout:

```c
const int key_offset = ((k * batch + b) * kv_head_num + kvh) * 128;
COMPUTE_FLOAT4 kv = CONVERT_COMPUTE_FLOAT4(vload4(0, key_cache + key_offset + d4));
```

Build:

```bash
( cd source/backend/opencl/execution/cl && python3 opencl_codegen.py . )
git diff --check

JOBS=$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 )) \
MNN_TARGET_DEVICE=aidlux_adreno_opencl BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Build note: the Rhino build script found stale/empty cross compiler/sysroot
cache again, removed `.cache/build/mnn/aidlux_adreno_opencl`, reconfigured with
Arm GNU 11.3, and installed AArch64 `pic_server`, `libMNN.so`, `libMNN_CL.so`,
and `libpic_llm.so`.

Manual Rhino max-frequency setting failed before this smoke because remote sudo
requires an interactive password:

```text
sudo: a terminal is required to read the password
```

Therefore the TPOT below is useful as same-run A/B evidence, not as a formal
max-frequency Rhino report.

### Profile smoke

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
MNN_PIC_DECODE_DEBUG=1 \
MNN_PAGED_ATTENTION_PROFILE=1 \
MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 \
MNN_PAGED_ATTENTION_DECODE_REPAIR_IDENTITY_QTILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino --models minicpm5-1b --contexts 512 \
  --pic-repair-tokens 0 --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 --repeats 1 --warm-repeats 0 \
  --skip-normal
```

Run id:

```text
decode_20260705_035943
```

P0:

```text
ordinary_q1=0
repair_qtile_route=1
identity_qtile=1
op=decode_causal_attention_hd128_identity_qtile
decode_prepare_inside_decode=0
```

No remote log hits for:

```text
decode_prepare_inside_decode=1
ERROR
target unavailable
async persistent PIC cache read failed
CL_OUT_OF_RESOURCES
Build Program failed
```

### MiniCPM x0 quick A/B

Default:

```text
adreno_identity_qtile_ab_default_20260705_0405
PIC_SWEEP_SERVER_ENV_EXTRA='LD_PRELOAD=/usr/lib/libOpenCL_adreno.so'
```

Candidate:

```text
adreno_identity_qtile_ab_candidate_20260705_0406
PIC_SWEEP_SERVER_ENV_EXTRA='LD_PRELOAD=/usr/lib/libOpenCL_adreno.so MNN_PAGED_ATTENTION_DECODE_REPAIR_IDENTITY_QTILE=1'
```

Result:

```text
model,ctx,x,default_ms,identity_qtile_ms,delta_ms,ratio_identity/default
MiniCPM5-1B,512,0,48.168,46.432,-1.736,0.9640
MiniCPM5-1B,1024,0,55.764,51.750,-4.014,0.9280
```

This looked promising enough to run the three-model x0 A/B.

### Three-model x0 A/B

Default:

```text
adreno_identity_qtile_ab3_default_20260705_0410
```

Candidate:

```text
adreno_identity_qtile_ab3_candidate_20260705_0414
```

Both:

```bash
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino \
  --models llama3.2-3b,minicpm5-1b,qwen3-4b \
  --contexts 512,1024 \
  --pic-repair-tokens 0 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 2 \
  --warm-repeats 1 \
  --skip-normal
```

Candidate added:

```text
MNN_PAGED_ATTENTION_DECODE_REPAIR_IDENTITY_QTILE=1
```

Result:

```text
model,ctx,x,default_ms,identity_qtile_ms,delta_ms,ratio_identity/default
Llama3.2 3B,512,0,72.524,74.588,+2.064,1.0285
Llama3.2 3B,1024,0,81.941,86.699,+4.758,1.0581
MiniCPM5-1B,512,0,46.813,46.683,-0.130,0.9972
MiniCPM5-1B,1024,0,54.121,52.316,-1.806,0.9666
Qwen3-4B,512,0,95.559,98.251,+2.692,1.0282
Qwen3-4B,1024,0,108.206,113.718,+5.511,1.0509
```

Profile-off CSV component deltas show that the change is in decode TPOT, not in
sampling or non-decode wall time:

```text
model,ctx,delta_tpot_ms,delta_sample_tpot_ms,delta_wall_tpot_ms,delta_wall_minus_decode_tpot_ms
Llama3.2 3B,512,+2.064,-0.029,+1.824,-0.111
Llama3.2 3B,1024,+4.758,-0.012,+4.157,-0.304
MiniCPM5-1B,512,-0.130,-0.065,-0.204,-0.082
MiniCPM5-1B,1024,-1.806,-0.001,-1.913,-0.220
Qwen3-4B,512,+2.692,-0.009,+2.806,+0.282
Qwen3-4B,1024,+5.511,+0.083,+3.368,-1.798
```

Because the first conclusion was still too inferential, a paired profile-on
attribution run was added. These numbers are not formal TPOT because
`MNN_PAGED_ATTENTION_PROFILE_DETAIL=1` synchronizes sub-ops; they are only used
to split decode-forward cost.

Default profile:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='LD_PRELOAD=/usr/lib/libOpenCL_adreno.so MNN_PIC_DECODE_DEBUG=1 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino --models llama3.2-3b,minicpm5-1b,qwen3-4b \
  --contexts 512,1024 --pic-repair-tokens 0 --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 --repeats 1 --warm-repeats 0 \
  --skip-normal --run-id adreno_identity_qtile_profile3_default_20260705_cmp
```

Identity-qtiler profile:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='LD_PRELOAD=/usr/lib/libOpenCL_adreno.so MNN_PIC_DECODE_DEBUG=1 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PAGED_ATTENTION_DECODE_REPAIR_IDENTITY_QTILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino --models llama3.2-3b,minicpm5-1b,qwen3-4b \
  --contexts 512,1024 --pic-repair-tokens 0 --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 --repeats 1 --warm-repeats 0 \
  --skip-normal --run-id adreno_identity_qtile_profile3_identity_20260705_cmp
```

Parser boundary:

- start a segment at `Prepared PIC decode repair ... pic_tokens=512/1024`
- count only `OpenCLPagedAttention profile op=decode_causal_attention...`
- ignore earlier `/v1/prefill/text`, `hydrate`, `prefill_attention`, and
  full-reuse suffix prefill rows
- this profile shows one decode-forward step per context; CSV `generated_tokens=2`
  includes the first token produced by the request's prefill/logits path

Profile P0:

```text
both runs:
  errors=0
  decode_prepare_inside_decode=0
  ordinary_q1=0
  repair_qtile_route=1

default op:
  decode_causal_attention_hd128_transposed_k_sparse_qtile

identity qtile op:
  decode_causal_attention_hd128_identity_qtile
```

Profile component totals:

```text
model,ctx,variant,calls,total_ms,append_ms,attention_ms,rank_ms,avg_us
Llama3.2 3B,512,default,28,30.610,11.572,18.937,0.053,1093.2
Llama3.2 3B,512,identity_qtile,28,32.777,12.073,20.612,0.043,1170.6
Llama3.2 3B,1024,default,28,47.915,14.281,33.256,0.203,1711.2
Llama3.2 3B,1024,identity_qtile,28,51.222,14.740,35.991,0.311,1829.4
MiniCPM5-1B,512,default,24,25.572,9.355,16.121,0.053,1065.5
MiniCPM5-1B,512,identity_qtile,24,22.259,9.470,12.721,0.027,927.5
MiniCPM5-1B,1024,default,24,40.255,11.387,28.543,0.180,1677.3
MiniCPM5-1B,1024,identity_qtile,24,33.572,11.284,21.927,0.223,1398.8
Qwen3-4B,512,default,36,37.889,14.246,23.538,0.048,1052.5
Qwen3-4B,512,identity_qtile,36,38.492,14.048,24.333,0.046,1069.2
Qwen3-4B,1024,default,36,61.204,18.029,42.731,0.239,1700.1
Qwen3-4B,1024,identity_qtile,36,67.426,20.411,45.037,0.319,1872.9
```

Identity-qtiler minus default:

```text
model,ctx,delta_total_ms,delta_append_ms,delta_attention_ms,delta_rank_ms,delta_avg_us
Llama3.2 3B,512,+2.167,+0.501,+1.675,-0.010,+77.4
Llama3.2 3B,1024,+3.307,+0.459,+2.735,+0.108,+118.1
MiniCPM5-1B,512,-3.313,+0.115,-3.400,-0.026,-138.0
MiniCPM5-1B,1024,-6.683,-0.103,-6.616,+0.043,-278.5
Qwen3-4B,512,+0.603,-0.198,+0.795,-0.002,+16.8
Qwen3-4B,1024,+6.222,+2.382,+2.306,+0.080,+172.8
```

P0:

- both runs have no `failures.csv`
- remote logs have no `decode_prepare_inside_decode=1`
- remote logs have no `ERROR`, `target unavailable`, async PIC cache read
  failure, `CL_OUT_OF_RESOURCES`, or OpenCL build failure

Decision:

No-transposed-K buffer qtile is not a production default for Adreno. It is a
useful diagnostic family and may help MiniCPM, but it regresses Llama and Qwen.
The measured profile says the gain/loss is mostly in the decode attention body:
MiniCPM's `attention_ms` drops, while Llama/Qwen's `attention_ms` increases.
Qwen ctx1024 also has a visible `append_ms` increase. Therefore the data does
not support "transposed K is the Adreno PIC-series root cause" as a general
statement; it only supports keeping no-transposed-K as a named A/B family.

Keep this env-gated only. Do not transfer it to Mali by default. If continuing
this direction, the next experiment should be a named Adreno image/K-read
variant or an append-no-decodeKey split, each reported separately.

## PIC x0 vs true normal decode: context, graph, and kernel timing

User request: do not guess. Compare context length, compare the two computation
graphs, and identify what PIC x0 computes extra and which operators are slow.

### Source data

TPOT CSVs:

```text
.cache/mnn-pic-benchmark/decode_experiment/adreno_identity_qtile_ab3_default_20260705_0410/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/adreno_true_normal_current_20260705_cmp/decode_tpot_long.csv
```

PIC graph/profile log:

```text
.cache/mnn-pic-benchmark/decode_experiment/adreno_pic_x0_graph_profile_minicpm512_20260705/logs/pic_server_decode_minicpm5-1b.log
```

Normal debug count log:

```text
.cache/mnn-pic-benchmark/decode_experiment/adreno_normal_decode_profile_minicpm512_20260705/logs/normal_llm_bench_minicpm512_decode_profile.log
```

Normal OpenCL time-profile diagnostic:

```text
.cache/mnn-pic-benchmark/decode_experiment/adreno_normal_timeprof_minicpm512_20260705/logs/normal_llm_bench_minicpm512_timeprof.log
```

The normal debug profile uses MNN callbacks and distorts wall-time; use it only
for op counts. The OpenCL time-profile diagnostic keeps normal decode speed
reasonable and is used for kernel-event attribution.

### Current TPOT gap and context caveat

```text
model,ctx_col,normal_ms,pic_x0_ms,gap_ms,ratio
Llama3.2 3B,512,53.763,72.524,+18.762,1.349
Llama3.2 3B,1024,56.658,81.941,+25.283,1.446
MiniCPM5-1B,512,27.350,46.813,+19.463,1.712
MiniCPM5-1B,1024,25.181,54.121,+28.941,2.149
Qwen3-4B,512,69.774,95.559,+25.785,1.370
Qwen3-4B,1024,72.365,108.206,+35.841,1.495
```

The context column is not a strict same-context comparison. Both run configs set
`pic_suffix_from_cache_tokens=1`, and the PIC profile confirms:

```text
PIC MiniCPM ctx512 decode: kv_len=514, prepare_len=513
normal diagnostic: llm_bench -p 512
```

Therefore the current conclusion is near-context attribution. Do not label it
as exact same-context until a normal p514/p1026 run or a PIC suffix=0 rerun is
available.

### Route evidence

PIC x0 is a valid PIC dualgraph decode-repair x0 route, not true normal:

```text
PIC decode graph loaded ... llm_decode.mnn
Prepared PIC decode repair ... budget_per_step=0
ordinary_q1=0 repair=1 q=1 kv_len=514
repair_qtile_enabled=1 repair_qtile_candidate=1 repair_qtile_prefix_ready=1 repair_qtile_route=1
op=decode_causal_attention_hd128_transposed_k_sparse_qtile
decode_prepare_inside_decode=0
```

### Graph op count comparison

Normal decode-only debug profile for 2 generated tokens gives the following
per-token counts:

```text
op type,normal calls per token
Attention,24
Convolution,169
Raster,390
BinaryOp,121
While,96
UnaryOp,74
LayerNorm,49
Cast,1
```

PIC graph profile decode blocks:

```text
block,type,total_ms,calls
1,PagedAttention,30.571,24
1,Raster,0.609,390
1,ArgMax,0.436,2
1,Convolution,0.266,169
1,BinaryOp,0.251,121
1,While,0.196,96
1,UnaryOp,0.127,74
1,LayerNorm,0.083,49
1,Cast,0.002,1
2,PagedAttention,32.363,24
2,Raster,2.703,390
2,Convolution,1.180,169
2,BinaryOp,1.058,121
2,While,0.848,96
2,UnaryOp,0.522,74
2,ArgMax,0.397,2
2,LayerNorm,0.301,49
2,Cast,0.006,1
```

The graph bodies match for the major model ops. PIC adds two `ArgMax` calls per
decode block, but the measured cost is about `0.4 ms` in callback-profile mode.
This is not the `+19 ms` MiniCPM ctx512 TPOT gap.

### Normal q1 OpenCL kernel timing

Normal time-profile decode candidate blocks with 24 attention layers:

```text
block,total_ms,attention_ms,qk_ms,softmax_ms,qkv_ms,rearr_k_ms,rearr_v_ms,conv_ms,raster_ms
4,16.264,3.198,1.395,0.218,1.456,0.067,0.062,9.963,1.814
7,16.128,3.208,1.392,0.220,1.468,0.062,0.066,9.863,1.786
13,16.219,3.191,1.390,0.220,1.454,0.061,0.066,9.913,1.812
16,16.149,3.228,1.401,0.232,1.466,0.063,0.066,9.851,1.786
```

Normal q1 attention cost is about `3.2 ms/token` across 24 layers. The rest of
normal decode is dominated by `Convolution0`, about `9.85-9.96 ms/token`.

### PIC x0 PagedAttention timing

PIC MiniCPM ctx512 PagedAttention decode groups:

```text
group,layers,kv_len,prepare_len,total_ms,append_ms,attention_ms,rank_ms,avg_us,min_us,max_us,prepare_inside
1,24,514,513,24.976,9.099,15.814,0.028,1040.7,1009,1083,0
2,24,514,513,29.502,11.158,18.007,0.207,1229.2,1138,1421,0
```

Prepare-transpose groups:

```text
group,total_ms,avg_us_per_layer,inside_decode
1,11.310,471.2,0
2,5.730,238.8,0
```

Because `decode_prepare_inside_decode=0`, prepare transpose is not counted as a
decode token cost in this run. The slow decode-time work is:

- append current token K/V/decodeKey: `9.1-11.2 ms/token`
- scan/attention body in `decode_causal_attention_hd128_transposed_k_sparse_qtile`:
  `15.8-18.0 ms/token`
- rank: negligible here

### Attribution conclusion

For MiniCPM ctx512 near-context:

```text
normal q1 Attention kernels: ~3.2 ms/token
PIC x0 PagedAttention:       25.0-29.5 ms/token
attention delta:             about +21.8 to +26.3 ms/token
observed TPOT gap:           +19.46 ms/token
```

This measured attention delta is enough to explain the TPOT gap. The graph count
comparison rules out a large hidden MLP/Raster/body over-compute explanation.
The current slow operator family is the PIC decode PagedAttention route,
especially append plus the transposed-K sparse-qtil attention scan.

This does not by itself prove that the transposed-K layout alone is the root
cause, because the no-transposed-K qtile A/B was mixed: it improved MiniCPM but
regressed Llama/Qwen. The correct next step is a same-context confirmation and
then a named append/attention split or image/K-read variant inside the PIC
decode family.

## Strict MiniCPM p514 Alignment And Final x0 Gap Split

The context mismatch was checked directly on Rhino with the true normal MiniCPM
model. PIC ctx512 with `pic_suffix_from_cache_tokens=1` enters decode with
`kv_len=514`; the matched normal command was rerun at `-p 514`.

Official TPOT logs:

```text
normal p512: decode=44.88 tok/s -> 22.28 ms/token
normal p514: decode=44.35 tok/s -> 22.55 ms/token
```

The strict same-command context delta is only about `+0.27 ms/token`. The
OpenCL time-profile q1 blocks agree: normal p512 attention is `3.206 ms/token`,
normal p514 attention is `3.299 ms/token`, and the full q1 block changes from
`16.190 ms` to `16.293 ms`. Therefore the `512 vs 514` difference is not the
`20+ ms` PIC x0 gap.

Record queue helps fixed launch/update overhead but does not explain the
remaining gap:

```text
run,server_env,tpot_ms
record_baseline,MNN_PIC_DECODE_DEBUG=1,54.024
record_op,MNN_PIC_DECODE_DEBUG=1 MNN_PIC_OPENCL_RECORD_QUEUE=op,47.468
record_default_after_llm_default,MNN_PIC_DECODE_DEBUG=1,48.043
three_model_same_artifact_default_op,MiniCPM ctx512,49.021
```

So record queue saves roughly `6 ms/token` versus the older baseline, but PIC
x0 remains about `25-26 ms/token` slower than the strict normal p514 command.

The remaining measured delta lines up with PIC PagedAttention itself:

```text
normal p514 q1 full block:        16.293 ms/token
normal p514 q1 Attention only:     3.299 ms/token
PIC x0 PagedAttention group 1:    24.976 ms/token = 9.099 append + 15.814 attention + 0.028 rank
PIC x0 PagedAttention group 2:    29.502 ms/token = 11.158 append + 18.007 attention + 0.207 rank
```

Source-level mapping:

- normal q1 `AttentionBufExecution` runs `rearrange_k`, `matmul_qk_div_mask`,
  `softmax`, `rearrange_v`, `matmul_qkv`; profile shows `rearrange_k+v` is only
  about `0.13 ms/token` across 24 layers.
- PIC x0 qtile runs `append_sparse_decode_key_value_hd128` before attention.
  That append kernel reads `sparse_query[l]` and writes `key_cache`,
  `value_cache`, and `decode_key`. It costs `9.1-11.2 ms/token` across
  24 layers in the profile-detail run.
- PIC qtile attention then scans the prepared K/V with the fused
  `decode_causal_attention_hd128_transposed_k_sparse_qtile` family, costing
  `15.8-18.0 ms/token`.
- `decode_prepare_transpose_k` was measured separately at `11.310 ms` and
  `5.730 ms` for two prepare groups, but `inside_decode=0`, so it is not part
  of decode TPOT.

Conclusion after strict alignment: the extra delay is not context length, not
extra dense/MLP/Raster graph body, not rank/top-k, and not prepare-transpose
inside decode. It is the PIC x0 PagedAttention implementation: current-token
append to PagedCache/decodeKey plus the qtile attention scan. The next
optimization should split or replace those two pieces, with any no-transposed-K
or image-backed K-read variant reported as an explicit PIC decode family A/B.

## Key-cache QTile V2 A/B

### Implementation summary

Added an explicit Adreno-only diagnostic gate:

```text
MNN_PAGED_ATTENTION_DECODE_REPAIR_KEYCACHE_QTILE_V2=1
```

The route is limited to `1 <= attnLen <= 8` and disabled when fused append is
enabled. It keeps PIC dualgraph decode-repair semantics: the append kernel still
uses `sparse_query[l]` as the logical slot, but only writes `key_cache` and
`value_cache`. It does not write `decode_key`, bypasses
`ensureDecodeKeyReady()`, and passes `decode_key_required=0` /
`decode_key_prepared=0`. Attention then uses the existing identity qtile kernel
family reading contiguous `key_cache`.

New OpenCL kernel:

```text
append_sparse_decode_key_value_hd128_keycache_v2
```

C++ profile/debug labels:

```text
append_sparse_decode_key_value_hd128_keycache_v2
decode_causal_attention_hd128_keycache_qtile_v2
repair_keycache_qtile_v2=1
```

The generated OpenCL source was regenerated with:

```bash
( cd source/backend/opencl/execution/cl && python3 opencl_codegen.py . )
```

The Adreno artifact was rebuilt and synced to Rhino:

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Artifact check: `pic_server`, `libMNN.so`, `libMNN_CL.so`, and `libpic_llm.so`
were AArch64 ELF files with timestamps around `2026-07-05 11:25`.

### P0 smoke

The first profile smoke after adding the kernel invalidated the old OpenCL
runtime cache, so it was not used for a performance conclusion. The clean
profile smoke was:

```text
decode_adreno_keycache_qtile_v2_profile_smoke_clean_20260705_1130
```

It had no `decode_prepare_inside_decode=1`, no `ERROR`, no target unavailable,
no async persistent PIC cache read failure, no cache invalid, no
`CL_OUT_OF_RESOURCES`, and no failures file. For q=1/2/4/6/8 it produced v2
route/profile lines; `ordinary_q1=1` count was zero.

The request-profile P0 smoke was:

```text
decode_adreno_keycache_qtile_v2_p0_profile_20260705_1133
```

Checks:

```text
decode_graph_loaded: 1
decode_graph_route: 5
use_decode_graph=1: 5
pic_decode_repair_decode_graph=1: 5
ordinary_q1=1: 0
q=1/2/4/6/8 route_lines: 24 each
q=1/2/4/6/8 profile_lines: 24 each
decode_runtime: mnn_token_id_sparse_decode
```

Profile-detail timing is attribution only; it adds queue synchronizations and
must not be used as TPOT.

### Profile-off matrix

Default profile-off run:

```text
decode_adreno_keycache_qtile_v2_default_matrix_20260705_1136
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1'
```

V2 profile-off run:

```text
decode_adreno_keycache_qtile_v2_matrix_20260705_1200
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1 MNN_PAGED_ATTENTION_DECODE_REPAIR_KEYCACHE_QTILE_V2=1'
```

Both runs used:

```text
--devices rhino
--models minicpm5-1b,llama3.2-3b,qwen3-4b
--contexts 512,1024
--skip-normal
--pic-repair-tokens 0,1,3,5,7
--pic-max-tokens 16
--pic-suffix-from-cache-tokens 1
--repeats 2
--warm-repeats 1
--chat-timeout 900
--remote-timeout 9000
```

Default had no `failures.csv`; the hard error scan for
`decode_prepare_inside_decode=1|ERROR|target unavailable|async persistent PIC cache read failed|Cache invalid|CL_OUT_OF_RESOURCES`
printed nothing.

V2 had no `failures.csv`; the same hard error scan printed nothing. Route scan:

```text
repair_keycache_qtile_v2=1 39600
ordinary_q1=1 0
decode_prepare_inside_decode=1 0
```

The v2 profile-off run does not print profile op labels, so the op-name counts
are expected to be zero in that run.

### x0 result

```text
model,ctx,true_normal,default_x0,v2_x0,default_extra,v2_extra,v2_minus_default
MiniCPM5-1B,512,23.219,48.526,47.796,+25.307,+24.577,-0.730
MiniCPM5-1B,1024,29.214,56.701,53.535,+27.487,+24.321,-3.167
Llama3.2 3B,512,53.256,73.945,75.107,+20.689,+21.851,+1.162
Llama3.2 3B,1024,55.824,83.696,87.466,+27.872,+31.642,+3.770
Qwen3-4B,512,70.256,96.914,100.540,+26.658,+30.284,+3.627
Qwen3-4B,1024,71.710,109.734,115.672,+38.024,+43.962,+5.938
```

V2 did not eliminate the x0 gap. It improved MiniCPM x0 slightly, but made
Llama/Qwen x0 worse. The remaining v2 x0 gap versus true normal is still about
`24-44 ms/token`.

### Full matrix

Cells show `v2_tpot_ms (v2 - default)`:

```text
model,ctx,x0,x1,x3,x5,x7
MiniCPM5-1B,512,47.796 (-0.730),50.786 (-1.887),63.502 (-3.314),84.126 (-3.105),87.838 (-2.741)
MiniCPM5-1B,1024,53.535 (-3.167),58.151 (-1.929),75.333 (-3.367),112.376 (-5.551),117.351 (-5.692)
Llama3.2 3B,512,75.107 (+1.162),92.369 (-1.041),114.613 (-4.081),170.619 (-3.834),176.184 (-3.805)
Llama3.2 3B,1024,87.466 (+3.770),109.425 (+5.511),131.382 (-1.257),206.979 (-5.612),218.237 (-0.900)
Qwen3-4B,512,100.540 (+3.627),123.486 (-1.253),149.486 (-3.665),229.151 (-4.275),229.461 (-4.680)
Qwen3-4B,1024,115.672 (+5.938),149.010 (+9.248),185.816 (+4.243),276.041 (-3.732),282.435 (-5.705)
```

V2 is mixed: most x3/x5/x7 rows improve by a few ms/token, but the targeted x0
row regresses on Llama and Qwen, and x1 also regresses for Llama ctx1024 and
Qwen ctx1024. This is not default-promotable.

### Conclusion

The current key-cache qtile v2 route proves that removing the decodeKey write
dependency is not sufficient to remove the PIC x0 penalty on Adreno. The slow
path remains the PIC PagedAttention decode family: current-token append plus
the key/value scan. Keep v2 as an explicit env-gated diagnostic variant. The
next useful experiments should target the append+attention body directly, such
as an image-backed K read or a fused append/attention q1 variant, and must still
report x0/x1/x3/x5/x7 as the same PIC dualgraph decode-repair family.

## Code Cleanup: Disabled Negative Kernel Paths

User direction on 2026-07-05: ineffective code should be commented first, not
deleted. The cleanup therefore keeps all negative Adreno experiments as `#if 0`
source blocks and removes them only from the active default dispatch.

Active default after cleanup:

```text
PIC decode-repair default:
  transposed-K qtile family
  record queue enabled
  append kernel: append_sparse_decode_key_value_hd128
  attention key source: decodeKey
  attention heads: num_heads * batch
```

Disabled but retained as commented code:

```text
gate/env path,kernel/path,reason
MNN_PAGED_ATTENTION_DECODE_REPAIR_KEYCACHE_QTILE_V2,append_sparse_decode_key_value_hd128_keycache_v2 + key_cache qtile,MiniCPM small gain only; Llama/Qwen x0 regression; x0 gap still +24..44 ms/token
MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_IDENTITY_FUSED_KV,decode_causal_attention_hd128_sparse_identity_fused_kv_*,x0 still +23..44 ms/token vs true normal; Llama/Qwen regression
MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_GQA / Q1_GQA_FUSED,decode_causal_attention_hd128_transposed_k_sparse_gqa_*,regressed every tested x0 model/context pair
MNN_PAGED_ATTENTION_DECODE_REPAIR_IDENTITY_QTILE / NO_TRANSPOSED_K,decode_causal_attention_hd128_identity_qtile_*,helped MiniCPM but regressed Llama/Qwen
```

Files touched for this cleanup:

```text
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp
source/backend/opencl/execution/cl/paged_decode_attention_buf.cl
source/backend/opencl/execution/cl/paged_decode_attention_buf_mnn_cl.cpp
source/backend/opencl/execution/cl/opencl_source_map.hpp
```

The `.cl` kernels are wrapped in `#if 0` and `opencl_codegen.py` was rerun so the
generated embedded OpenCL source matches. The C++ side also keeps the old env
gates, kernel build calls, record queue overload/state, and legacy dispatch
snippets as `#if 0` comments. The active path no longer builds or checks the
disabled kernels.

Current conclusion for the user's "did we eliminate PIC x=0 extra 20ms delay?"
question: no. Strict normal p514 alignment showed context mismatch is not the
cause. The graph body is aligned closely enough with true normal; rank/top-k and
decode prepare are negligible or outside timed decode. The remaining x0 gap is
in PIC PagedAttention itself:

```text
path,cost
normal p514 q1 Attention,about 3.3 ms/token
PIC x0 PagedAttention,about 25.0-29.5 ms/token
PIC x0 append,about 9.1-11.2 ms/token
PIC x0 attention scan,about 15.8-18.0 ms/token
```

Do not keep spending default-path effort on the four disabled variants above.
The next useful analysis should split or replace the PagedAttention append and
attention scan while preserving the same PIC dualgraph decode-repair family for
x0/x1/x3/x5/x7.

### Post-cleanup build and route smoke

Local build:

```bash
JOBS=96 MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

The build completed and installed:

```text
.cache/output/mnn/artifacts/aidlux_adreno_opencl/bin/pic_server
.cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN.so
.cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN_CL.so
.cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libpic_llm.so
```

All four outputs are AArch64 ELF files. The build rebuilt
`PagedAttentionBufExecution.cpp` and `paged_decode_attention_buf_mnn_cl.cpp`
successfully.

Artifact sync:

```bash
rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Route smoke:

```bash
RUN_ID=decode_adreno_minicpm_ctx512_x0_after_cleanup_20260705_041937
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1' \
conda run -n kvshare-edge python .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino \
  --models minicpm5-1b \
  --contexts 512 \
  --skip-normal \
  --pic-repair-tokens 0 \
  --pic-max-tokens 4 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --chat-timeout 900 \
  --remote-timeout 9000 \
  --run-id "${RUN_ID}"
```

Result:

```text
device,model,ctx,family,x,tpot_ms,status
rhino,MiniCPM5-1B,512,pic-dualgraph-decode-repair,0,48.726,ok
```

`failures.csv` was absent. Route log evidence:

```text
ordinary_q1=0 repair=1 q=1
repair_qtile_route=1
record_queue=1
repair_keycache_qtile_v2=0
repair_q1_identity_fused_kv=0
repair_q1_gqa=0
decode_prepare_inside_decode=0
```

The server log had one startup `Cache invalid, will be reset`, so this row is
only a build/route smoke after source cleanup. It is not a formal TPOT data
point and should not replace the earlier warmed A/B tables.

## Post-cleanup x0 attribution runs

### PagedAttention detail profile

Command:

```bash
RUN_ID=decode_adreno_minicpm_ctx512_x0_profile_attribution_20260705_043604
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1' \
conda run -n kvshare-edge python .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino \
  --models minicpm5-1b \
  --contexts 512 \
  --skip-normal \
  --pic-repair-tokens 0 \
  --pic-max-tokens 4 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --chat-timeout 900 \
  --remote-timeout 9000 \
  --run-id "${RUN_ID}"
```

Result:

```text
device,model,ctx,family,x,generated_tokens,tpot_ms,status
rhino,MiniCPM5-1B,512,pic-dualgraph-decode-repair,0,4,79.458,ok
```

Frequency note: an interactive sudo frequency write was not available from this
shell, but the benchmark frequency snapshot showed CPU governors already on
`performance` at max frequencies and GPU `cur_freq=max_freq=680000000`.

P0:

```text
failures.csv: absent/empty
decode_prepare_inside_decode=1: 0
ERROR / target unavailable / async persistent PIC cache read failed / Cache invalid / CL_OUT_OF_RESOURCES: 0
route: ordinary_q1=0 repair=1 repair_qtile_route=1 repair_keycache_qtile_v2=0 repair_q1_identity_fused_kv=0 repair_q1_gqa=0
```

`MNN_PAGED_ATTENTION_PROFILE_DETAIL=1` disables the record replay path inside
the PagedAttention profile body, so these numbers are attribution only. The
configured route still reports `record_queue=1`.

Aggregated profile lines:

```text
kv_len,layers,total_ms,append_ms,attention_ms,rank_ms,other_ms,lane,q_tile,prepare_inside_sum
514,48,53.903,20.146,33.407,0.181,0.169,128,1,0
515,48,59.664,22.661,36.383,0.345,0.275,128,1,0
516,48,57.775,21.800,35.429,0.296,0.250,128,1,0
```

Because this includes warm + measure passes, the per decode-pass split is:

```text
kv_len,total_ms,append_ms,attention_ms,rank_ms
514,26.952,10.073,16.704,0.091
515,29.832,11.331,18.192,0.173
516,28.888,10.900,17.715,0.148
```

This confirms the earlier attribution after cleanup: the active x0 qtile path
still spends about `10-11 ms/token` appending K/V/decodeKey and about
`17-18 ms/token` scanning K/V in attention. Rank is noise, and prepare-transpose
does not run inside timed decode.

### Decode-scope graph profile

Command:

```bash
RUN_ID=decode_adreno_minicpm_ctx512_x0_graph_profile_20260705_0440
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1 MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_SCOPE=decode MNN_PIC_GRAPH_PROFILE_TOP=1000' \
conda run -n kvshare-edge python .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices rhino \
  --models minicpm5-1b \
  --contexts 512 \
  --skip-normal \
  --pic-repair-tokens 0 \
  --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --chat-timeout 900 \
  --remote-timeout 9000 \
  --run-id "${RUN_ID}"
```

Result:

```text
device,model,ctx,family,x,generated_tokens,tpot_ms,status
rhino,MiniCPM5-1B,512,pic-dualgraph-decode-repair,0,2,52.697,ok
```

P0 was clean again: no failure rows, no `decode_prepare_inside_decode=1`, and
no server/CL error lines. Decode graph route evidence remained:

```text
use_decode_graph=1
pic_decode_repair_decode_graph=1
ordinary_q1=0
repair_qtile_route=1
```

Graph-profile summaries:

```text
request=1 total_ms=11.361 calls=926 unique_ops=925
  PagedAttention 8.425 ms
  Convolution    1.040 ms
  Raster         0.721 ms
  ArgMax         0.409 ms
  BinaryOp       0.294 ms
  While          0.245 ms
  UnaryOp        0.131 ms
  LayerNorm      0.094 ms
  Cast           0.002 ms

request=2 total_ms=2.397 calls=926 unique_ops=925
  Convolution    0.720 ms
  Raster         0.591 ms
  PagedAttention 0.368 ms
  BinaryOp       0.218 ms
  While          0.185 ms
  ArgMax         0.133 ms
  UnaryOp        0.104 ms
  LayerNorm      0.076 ms
  Cast           0.002 ms
```

`MNN_PIC_REQUEST_PROFILE` still showed `forward_raw_on_forward cost_ms=52.619`
for one decode graph call. Therefore the graph callback totals do not explain
the full server-side TPOT on OpenCL; they are useful for ruling out a large
dense/Raster/ArgMax graph-body regression, but not for assigning all GPU wait
time. The missing time is outside the non-attention graph body and should be
investigated as OpenCL queue execution/synchronization/record replay timing and
PagedAttention event timing mismatch.

### Current conclusion

The x0 extra 20ms versus true normal decode has not been removed. The current
best cleaned default is still around `48 ms/token` for MiniCPM ctx512 x0, while
true normal p512/p514 is about `22-23 ms/token`. The useful attribution is:

```text
component,status
context mismatch,rejected; p512->p514 normal delta only about 0.27 ms/token
decode graph body,rejected as primary cause; Convolution/Raster/ArgMax are small
rank/top-k,rejected; rank is about 0.1-0.2 ms/token in detail profile
decode prepare,rejected; decode_prepare_inside_decode=0
disabled kernel variants,rejected for default; all kept behind #if 0 comments
PagedAttention append,still significant at about 10-11 ms/token
PagedAttention attention scan,still dominant at about 17-18 ms/token
OpenCL queue/record timing,needs more instrumentation because graph callback totals under-report forward wall time
```

Do not spend more default-path effort on the disabled keycache-v2,
q1-identity-fused-KV, q1-GQA, or identity-qtiler paths without new full-matrix
evidence. The next implementation change should be instrumentation or a
targeted replacement of the active PagedAttention append/attention scan path,
with all experimental code commented or `#if 0`-guarded if it fails.
