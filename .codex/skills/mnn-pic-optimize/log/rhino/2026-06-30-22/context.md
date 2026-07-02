# Context

## Objective

Continue Rhino Pi-X1 / Adreno OpenCL decode repair optimization for `MiniCPM5-1B`, `lagged_attention_hkvd`, full-reuse, repair tokens `x=0/1/3/5/7`.

The current objective is not to copy Jetson CUDA packed gate/up concat Conv to Rhino. Rhino 2026-06-29 logs showed concat/split gateup graph routes were mixed and often worse by `extra_ms = TPOT(x) - TPOT(x=0)`. The preserved baseline is the row-aware tiny dense family selection:

```text
rows <= 2: pic_quant_c4
rows <= 4: pic_quant_incache64
rows >= 6: adreno_batch_gemv_c4out
```

The next gate is a direct-op fused tiny-row MLP benchmark:

```text
gate_proj + up_proj -> SiLU(gate) * up -> down_proj
```

It should reduce duplicate input/dequant/weight reads and avoid materializing the full `[rows, inter]` activation where possible.

## Source Change In Scope

The OpenCL testbench now has:

```text
bench_ops/opencl/perf/DecodeRepairMlpChain
```

Default shape:

```text
rows:   2,4,6,8
hidden: 1536
inter:  4608
qblock: 64
```

Useful env overrides:

```text
MNN_BENCH_OPENCL_MLP_CHAIN_WARMUP
MNN_BENCH_OPENCL_MLP_CHAIN_REPEAT
MNN_BENCH_OPENCL_MLP_CHAIN_HIDDEN
MNN_BENCH_OPENCL_MLP_CHAIN_INTER
MNN_BENCH_OPENCL_MLP_CHAIN_QBLOCK
MNN_BENCH_OPENCL_MLP_CHAIN_CASE
MNN_BENCH_OPENCL_MLP_CHAIN_ROWS
```

It prints `gate_family`, `up_family`, `down_family`, individual op times, `split_sum`, and full `chain`.

## Build

The normal shared Aidlux testbench build compiled `OpenCLAttentionPerf.cpp`, but failed final link because Release shared builds hide backend C++ symbols that the OpenCL bench calls directly:

```text
undefined reference to MNN::OpenCL::OpenCLBackend::getOpenCLRuntime()
undefined reference to MNN::OpenCLRuntime::buildKernel(...)
undefined reference to MNN::OpenCL::run3DKernelDefault(...)
```

This is a testbench linking issue, not a front-end compile failure in `DecodeRepairMlpChain`.

The working bench artifact was built statically:

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl CLEAN=1 \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_SEP_BUILD=OFF -DMNN_BUILD_LLM=OFF -DMNN_BUILD_CONVERTER=OFF -DMNN_LLM_BUILD_DEMO=OFF' \
BUILD_DIR="$PWD/.cache/build/mnn/aidlux_adreno_opencl_testbench_static" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/aidlux_adreno_opencl_testbench_static" \
JOBS=96 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Artifact:

```text
local:  .cache/output/mnn/artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out
remote: /mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out
```

Artifact check:

```text
ELF 64-bit LSB executable, ARM aarch64, dynamically linked, interpreter /lib/ld-linux-aarch64.so.1
NEEDED: libstdc++.so.6, libm.so.6, libgcc_s.so.1, libc.so.6
```

CMake cache check:

```text
CMAKE_TOOLCHAIN_FILE=.../aidlux_adreno_opencl-arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu-aarch64-linux.toolchain.cmake
MNN_BUILD_SHARED_LIBS=OFF
MNN_SEP_BUILD=OFF
MNN_BUILD_LLM=OFF
MNN_BUILD_CONVERTER=OFF
MNN_BUILD_TEST=ON
MNN_LOW_MEMORY=ON
MNN_OPENCL=ON
```

## Rhino Smoke

Remote smoke command:

```bash
ssh aidlux@192.168.101.227 'cd /mnt/nvme/mnn_pic_opencl && \
  env LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
      MNN_BENCH_OPENCL_MLP_CHAIN_ROWS=2 \
      MNN_BENCH_OPENCL_MLP_CHAIN_WARMUP=2 \
      MNN_BENCH_OPENCL_MLP_CHAIN_REPEAT=3 \
      artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out \
      bench_ops/opencl/perf/DecodeRepairMlpChain 3 2 1 empty 2'
```

Result:

```text
rows=2 gate_family=pic_quant_c4 up_family=pic_quant_c4 down_family=pic_quant_c4
gate=0.3517 up=0.2883 silu=0.0953 down=0.2590 split_sum=0.9943 chain=0.6910 ms
all <bench_ops/opencl/perf/DecodeRepairMlpChain> tests passed
```

Remote smoke log:

```text
/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/bench_ops/decode_mlp_chain_smoke_20260630_224718.log
```

## Baseline Run

Command:

```bash
ssh aidlux@192.168.101.227 'cd /mnt/nvme/mnn_pic_opencl && \
  env LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
      MNN_BENCH_OPENCL_MLP_CHAIN_ROWS=2,4,6,8 \
      MNN_BENCH_OPENCL_MLP_CHAIN_WARMUP=40 \
      MNN_BENCH_OPENCL_MLP_CHAIN_REPEAT=160 \
      artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out \
      bench_ops/opencl/perf/DecodeRepairMlpChain 3 2 1 empty 2'
```

Result:

```text
rows=2 hidden=1536 inter=4608 qblock=64 gate_family=pic_quant_c4            up_family=pic_quant_c4            down_family=pic_quant_c4            gate=0.1668 up=0.1673 silu=0.0164 down=0.1367 split_sum=0.4871 chain=0.4710 ms
rows=4 hidden=1536 inter=4608 qblock=64 gate_family=pic_quant_incache64     up_family=pic_quant_incache64     down_family=pic_quant_incache64     gate=0.1706 up=0.1713 silu=0.0143 down=0.1410 split_sum=0.4972 chain=0.4852 ms
rows=6 hidden=1536 inter=4608 qblock=64 gate_family=adreno_batch_gemv_c4out up_family=adreno_batch_gemv_c4out down_family=adreno_batch_gemv_c4out gate=0.2990 up=0.2997 silu=0.0303 down=0.2453 split_sum=0.8743 chain=0.8464 ms
rows=8 hidden=1536 inter=4608 qblock=64 gate_family=adreno_batch_gemv_c4out up_family=adreno_batch_gemv_c4out down_family=adreno_batch_gemv_c4out gate=0.3010 up=0.3023 silu=0.0454 down=0.2431 split_sum=0.8917 chain=0.8537 ms
```

Remote log:

```text
/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/bench_ops/decode_mlp_chain_rows2468_repeat160_20260630_224750.log
```

## Interpretation

The bench proves the current split MLP chain baseline and confirms the row-aware family stays active:

- rows 2/4 are around `0.47-0.49 ms` per MLP chain.
- rows 6/8 are around `0.85 ms` per MLP chain.
- The primary direct-op target for a fused MLP kernel is rows 6/8 first, because they pay about `0.60 ms` for gate+up alone and about `0.24 ms` for down.

The next fused kernel should use this as the gate. Do not route it into the exported model unless it beats these chain times and passes accuracy for rows `2/4/6/8`.

