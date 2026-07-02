# Context

## Objective

Continue Rhino Pi-X1 / Adreno OpenCL PIC decode repair tiny-row MLP optimization for `MiniCPM5-1B`, `lagged_attention_hkvd`, `full-reuse`, repair x=`0/1/3/5/7`.

The hard gate remains:

1. direct-op rows `2/4/6/8` fused MLP must be faster than the current row-aware split Conv/MLP chain and pass accuracy;
2. only then should endpoint decode repair be tested at ctx `512/1024`, max_tokens `32`, repeat `>=3`, with success judged by reduced `extra_ms = TPOT(x)-TPOT(x=0)` for x=`3/5/7` and no meaningful x=`0/1` regression.

## Source Change

Completed the existing bench-only resource-backed direct-op path in:

```text
test/bench_ops/opencl/OpenCLAttentionPerf.cpp
```

New registered entry:

```text
bench_ops/opencl/perf/DecodeRepairMlpResourceGateUp
```

It uses these envs:

```text
MNN_BENCH_OPENCL_MLP_RESOURCE_WARMUP
MNN_BENCH_OPENCL_MLP_RESOURCE_REPEAT
MNN_BENCH_OPENCL_MLP_RESOURCE_HIDDEN
MNN_BENCH_OPENCL_MLP_RESOURCE_INTER
MNN_BENCH_OPENCL_MLP_RESOURCE_QBLOCK
MNN_BENCH_OPENCL_MLP_RESOURCE_CASE
MNN_BENCH_OPENCL_MLP_RESOURCE_ROWS
MNN_BENCH_OPENCL_MLP_RESOURCE_ABS_TOL_MILLI
MNN_BENCH_OPENCL_MLP_RESOURCE_REL_TOL_MILLI
```

The bench constructs normal gate/up/down `ConvBufLowMemoryExecution` instances, extracts gate/up image-backed resources, runs a custom `resource_gateup_silu_int4_image` kernel, then sends the fused activation through the existing down Conv. It compares final output against the normal split chain.

This is not production routing.

The same source file now also contains a second bench-only entry:

```text
bench_ops/opencl/perf/DecodeRepairMlpSiluDownC4
```

This path keeps the current row-aware gate/up Conv executions, then replaces `PicSiluMul + down_proj` with a custom image-backed `resource_silu_down_c4_int4_image` kernel. The kernel reads gate/up in C4NHW4 layout, applies `SiLU(gate)*up`, multiplies by the down_proj int4 image weights, and writes final C4 output. It is a lower-bound test for avoiding the `[rows, inter]` activation tensor and down preconvert, not a production route.

## Build

Command:

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl CLEAN=0 \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_SEP_BUILD=OFF -DMNN_BUILD_LLM=OFF -DMNN_BUILD_CONVERTER=OFF -DMNN_LLM_BUILD_DEMO=OFF' \
BUILD_DIR="$PWD/.cache/build/mnn/aidlux_adreno_opencl_testbench_static" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/aidlux_adreno_opencl_testbench_static" \
JOBS=96 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

The script removed the stale build directory because cached compiler/sysroot fields were empty, then rebuilt successfully.

Artifact check:

```text
.cache/output/mnn/artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out:
ELF 64-bit LSB executable, ARM aarch64, dynamically linked, interpreter /lib/ld-linux-aarch64.so.1
```

CMake cache:

```text
CMAKE_TOOLCHAIN_FILE=.../aidlux_adreno_opencl-arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu-aarch64-linux.toolchain.cmake
MNN_BUILD_SHARED_LIBS=OFF
MNN_BUILD_TEST=ON
MNN_CUDA=OFF
MNN_LOW_MEMORY=ON
MNN_OPENCL=ON
MNN_VULKAN=OFF
```

Synced to:

```text
/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl_testbench_static/
```

## Rhino Commands

Rows=2 smoke:

```bash
ssh aidlux@192.168.101.227 'cd /mnt/nvme/mnn_pic_opencl && mkdir -p cache/pic_prefill_latency_sweep/bench_ops && \
  env LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
      MNN_BENCH_OPENCL_MLP_RESOURCE_ROWS=2 \
      MNN_BENCH_OPENCL_MLP_RESOURCE_WARMUP=2 \
      MNN_BENCH_OPENCL_MLP_RESOURCE_REPEAT=3 \
      artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out \
      bench_ops/opencl/perf/DecodeRepairMlpResourceGateUp 3 2 1 empty 2 \
      2>&1 | tee cache/pic_prefill_latency_sweep/bench_ops/decode_mlp_resource_gateup_smoke_rows2_20260630_2323.log'
```

Rows `2/4/6/8`, repeat 30:

```bash
ssh aidlux@192.168.101.227 'cd /mnt/nvme/mnn_pic_opencl && mkdir -p cache/pic_prefill_latency_sweep/bench_ops && \
  env LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
      MNN_BENCH_OPENCL_MLP_RESOURCE_ROWS=2,4,6,8 \
      MNN_BENCH_OPENCL_MLP_RESOURCE_WARMUP=10 \
      MNN_BENCH_OPENCL_MLP_RESOURCE_REPEAT=30 \
      artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out \
      bench_ops/opencl/perf/DecodeRepairMlpResourceGateUp 3 2 1 empty 2 \
      2>&1 | tee cache/pic_prefill_latency_sweep/bench_ops/decode_mlp_resource_gateup_rows2468_repeat30_20260630_2324.log'
```

Rows `2/4/6/8`, repeat 160:

```bash
ssh aidlux@192.168.101.227 'cd /mnt/nvme/mnn_pic_opencl && mkdir -p cache/pic_prefill_latency_sweep/bench_ops && \
  env LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
      MNN_BENCH_OPENCL_MLP_RESOURCE_ROWS=2,4,6,8 \
      MNN_BENCH_OPENCL_MLP_RESOURCE_WARMUP=40 \
      MNN_BENCH_OPENCL_MLP_RESOURCE_REPEAT=160 \
      artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out \
      bench_ops/opencl/perf/DecodeRepairMlpResourceGateUp 3 2 1 empty 2 \
      2>&1 | tee cache/pic_prefill_latency_sweep/bench_ops/decode_mlp_resource_gateup_rows2468_repeat160_20260630_2325.log'
```

SiluDownC4 rows `2/4/6/8`, repeat 160:

```bash
ssh aidlux@192.168.101.227 'cd /mnt/nvme/mnn_pic_opencl && mkdir -p cache/pic_prefill_latency_sweep/bench_ops && \
  env LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
      MNN_BENCH_OPENCL_MLP_SILU_DOWN_C4_ROWS=2,4,6,8 \
      MNN_BENCH_OPENCL_MLP_SILU_DOWN_C4_WARMUP=40 \
      MNN_BENCH_OPENCL_MLP_SILU_DOWN_C4_REPEAT=160 \
      artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out \
      bench_ops/opencl/perf/DecodeRepairMlpSiluDownC4 3 2 1 empty 2 \
      2>&1 | tee cache/pic_prefill_latency_sweep/bench_ops/decode_mlp_silu_down_c4_rows2468_repeat160_20260630_2333.log'
```

## Results

Smoke rows=2:

```text
rows=2 split_chain=0.6643 gateup_silu=0.4923 fused_chain=0.7317 delta=+0.0673 ms bad=0/3072
```

Repeat 30:

```text
rows=2 split_chain=0.4893 gateup_silu=0.4232 fused_chain=0.5618 delta=+0.0725 ms bad=0/3072
rows=4 split_chain=0.5031 gateup_silu=0.4792 fused_chain=0.6111 delta=+0.1080 ms bad=0/6144
rows=6 split_chain=0.8693 gateup_silu=0.8551 fused_chain=1.1256 delta=+0.2563 ms bad=0/9216
rows=8 split_chain=0.8749 gateup_silu=0.9053 fused_chain=1.1610 delta=+0.2861 ms bad=0/12288
```

Repeat 160:

```text
rows=2 split_chain=0.4681 gateup_silu=0.4078 fused_chain=0.5427 delta=+0.0746 ms bad=0/3072
rows=4 split_chain=0.4838 gateup_silu=0.4583 fused_chain=0.5949 delta=+0.1111 ms bad=0/6144
rows=6 split_chain=0.8438 gateup_silu=0.8396 fused_chain=1.0823 delta=+0.2385 ms bad=0/9216
rows=8 split_chain=0.8528 gateup_silu=0.8892 fused_chain=1.1333 delta=+0.2805 ms bad=0/12288
```

SiluDownC4 repeat 160:

```text
rows=2 split_chain=0.4677 silu_down=0.1817 fused_chain=0.5065 delta=+0.0388 ms bad=0/3072
rows=4 split_chain=0.4839 silu_down=0.2952 fused_chain=0.6332 delta=+0.1493 ms bad=0/6144
rows=6 split_chain=0.8437 silu_down=0.4584 fused_chain=1.0538 delta=+0.2100 ms bad=0/9216
rows=8 split_chain=0.8508 silu_down=0.5465 fused_chain=1.1426 delta=+0.2917 ms bad=0/12288
```

## Interpretation

The fused resource path is numerically correct but slower at every target row count. It likely loses because it uses one generic gate/up batch-GEMV-style kernel and does not preserve the accepted row-aware dense family:

```text
rows <= 2: pic_quant_c4
rows <= 4: pic_quant_incache64
rows >= 6: adreno_batch_gemv_c4out
```

The key evidence is that `gateup_silu` alone is already close to or slower than the full split gate+up+SiLU portion. Adding the existing down projection then pushes the full chain above the split baseline:

- rows 2/4 regress by about `0.07-0.11 ms/layer`;
- rows 6/8 regress by about `0.24-0.28 ms/layer`.

This direct-op result blocks endpoint testing for this path.

The `SiluDownC4` path is also correct but slower. It keeps the row-aware gate/up side intact, so the regression points specifically at direct C4 activation consumption in down:

- rows 2 regresses by about `0.039 ms/layer`;
- rows 4 regresses by about `0.149 ms/layer`;
- rows 6/8 regress by about `0.210-0.292 ms/layer`.

This matches the earlier 6/29 direct-C4 conclusion: avoiding conversion is not enough on Adreno when the K loop then reads row data with C4/BHW stride.

## Decision

Reject `DecodeRepairMlpResourceGateUp` and `DecodeRepairMlpSiluDownC4` as production routes. Keep them as diagnostic direct-op benches only.

Next implementation should not use a single global fused gate/up kernel. The next credible direct-op attempts are:

1. row-aware fused gate/up variants that match the existing selected family for rows `2/4/6/8`;
2. a tiled fused-through-down floor that keeps contiguous NHWC/input-cache behavior for the down accumulation instead of directly streaming C4 gate/up activations.
