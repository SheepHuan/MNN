# Context

## Objective

Continue Rhino Pi-X1 / Adreno OpenCL PIC decode repair tiny-row MLP optimization for `MiniCPM5-1B`, `lagged_attention_hkvd`, `full-reuse`, repair x=`0/1/3/5/7`.

The hard direct-op gate from the objective remains:

```text
rows:   2/4/6/8
hidden: 1536
inter:  4608
qblock: 64
dtype:  int4 weight-only
```

The production constraint remains: preserve the current Rhino row-aware tiny dense family selection:

```text
rows <= 2: pic_quant_c4
rows <= 4: pic_quant_incache64
rows >= 6: adreno_batch_gemv_c4out
```

Do not promote Jetson CUDA packed gate/up concat Conv as Rhino default.

## Baseline

The accepted split-chain baseline from `DecodeRepairMlpChain`, warmup 40, repeat 160:

```text
rows=2 chain=0.4710 ms
rows=4 chain=0.4852 ms
rows=6 chain=0.8464 ms
rows=8 chain=0.8537 ms
```

Latest same-run baselines in this hour are close:

```text
rows=2 split_chain=0.4674 ms
rows=4 split_chain=0.4841 ms
rows=6 split_chain=0.8443 ms
rows=8 split_chain=0.8519 ms
```

## Shared-Input WGS Sweep

The existing `DecodeRepairMlpSharedInputGateUp` path was rebuilt with:

```text
MNN_BENCH_OPENCL_MLP_SHARED_INPUT_WGS
```

Valid power-of-two WGS sweep for rows `6/8`, warmup 20, repeat 80:

```text
WGS=16
rows=6 split=0.8460 shared_gateup=1.5640 shared_chain=1.8023 delta=+0.9562 bad=0
rows=8 split=0.8550 shared_gateup=1.5495 shared_chain=1.8175 delta=+0.9625 bad=0

WGS=32
rows=6 split=0.8459 shared_gateup=0.8780 shared_chain=1.1174 delta=+0.2715 bad=0
rows=8 split=0.8541 shared_gateup=0.8668 shared_chain=1.1211 delta=+0.2670 bad=0

WGS=64
rows=6 split=0.8456 shared_gateup=0.6063 shared_chain=0.8696 delta=+0.0239 bad=0
rows=8 split=0.8547 shared_gateup=0.6105 shared_chain=0.8583 delta=+0.0036 bad=0

WGS=128
rows=6 split=0.8459 shared_gateup=0.8378 shared_chain=1.0851 delta=+0.2392 bad=0
rows=8 split=0.8548 shared_gateup=0.8262 shared_chain=1.0860 delta=+0.2312 bad=0

WGS=256
rows=6 split=0.8467 shared_gateup=1.4074 shared_chain=1.6693 delta=+0.8226 bad=0
rows=8 split=0.8547 shared_gateup=1.4107 shared_chain=1.6686 delta=+0.8138 bad=0
```

Neighbor WGS values `48/56/72/80/96` produced nonzero bad counts, so this reduction kernel should not treat arbitrary local sizes as valid candidates.

Full WGS 64 repeat 160:

```text
rows=2 split=0.4675 shared_gateup=0.3215 shared_chain=0.4591 delta=-0.0084 bad=0
rows=4 split=0.4834 shared_gateup=0.3234 shared_chain=0.4651 delta=-0.0184 bad=0
rows=6 split=0.8448 shared_gateup=0.5988 shared_chain=0.8476 delta=+0.0027 bad=0
rows=8 split=0.8520 shared_gateup=0.6032 shared_chain=0.8552 delta=+0.0031 bad=0
```

Decision: old shared-input gate/up chain still fails the direct-op gate because rows `6/8` are slightly slower.

## New Bench-Only SiluNhwcDown Floor

Added in `test/bench_ops/opencl/OpenCLAttentionPerf.cpp`:

```text
bench_ops/opencl/perf/DecodeRepairMlpSiluNhwcDown
```

This keeps the existing row-aware gate/up executions, but replaces:

```text
PicSiluMul(C4) -> down Conv internal C4-to-NHWC preconvert -> down GEMV
```

with:

```text
silu_c4_to_nhwc -> direct image-backed down GEMV
```

It is not a production route yet.

Rows `2/4/6/8`, warmup 40, repeat 160:

```text
rows=2 split=0.4669 silu_nhwc=0.0425 direct_down=0.1271 fused_chain=0.4596 delta=-0.0073 bad=0
rows=4 split=0.4840 silu_nhwc=0.0365 direct_down=0.1285 fused_chain=0.4697 delta=-0.0143 bad=0
rows=6 split=0.8444 silu_nhwc=0.0398 direct_down=0.2318 fused_chain=0.8371 delta=-0.0072 bad=0
rows=8 split=0.8521 silu_nhwc=0.0335 direct_down=0.2305 fused_chain=0.8424 delta=-0.0096 bad=0
```

Interpretation:

- Directly consuming C4 gate/up inside down was previously slower (`SiluDownC4`).
- Producing NHWC activation for down is better on Adreno because down keeps contiguous K reads.
- The useful boundary is not "avoid all conversion"; it is "fuse activation with the conversion that down already needs".

## Combined Shared-Input + NHWC Down Result

Extended `DecodeRepairMlpSharedInputGateUp` to also print:

```text
shared_silu_nhwc
shared_direct_down
shared_nhwc_chain
nhwc_delta
nhwc_bad
```

Command:

```bash
ssh aidlux@192.168.101.227 'cd /mnt/nvme/mnn_pic_opencl && mkdir -p cache/pic_prefill_latency_sweep/bench_ops && \
  env LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
      MNN_BENCH_OPENCL_MLP_SHARED_INPUT_ROWS=2,4,6,8 \
      MNN_BENCH_OPENCL_MLP_SHARED_INPUT_WGS=64 \
      MNN_BENCH_OPENCL_MLP_SHARED_INPUT_WARMUP=40 \
      MNN_BENCH_OPENCL_MLP_SHARED_INPUT_REPEAT=160 \
      artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out \
      bench_ops/opencl/perf/DecodeRepairMlpSharedInputGateUp 3 2 1 empty 2 \
      2>&1 | tee cache/pic_prefill_latency_sweep/bench_ops/decode_mlp_shared_input_nhwc_down_wgs64_rows2468_repeat160_20260630.log'
```

Result:

```text
rows=2 split=0.4674 shared_gateup=0.3169 shared_chain=0.4580 delta=-0.0094 shared_silu_nhwc=0.0320 shared_direct_down=0.1273 shared_nhwc_chain=0.4459 nhwc_delta=-0.0215 bad=0 nhwc_bad=0
rows=4 split=0.4841 shared_gateup=0.3177 shared_chain=0.4658 delta=-0.0183 shared_silu_nhwc=0.0436 shared_direct_down=0.1273 shared_nhwc_chain=0.4520 nhwc_delta=-0.0321 bad=0 nhwc_bad=0
rows=6 split=0.8443 shared_gateup=0.6018 shared_chain=0.8511 delta=+0.0068 shared_silu_nhwc=0.0364 shared_direct_down=0.2333 shared_nhwc_chain=0.8359 nhwc_delta=-0.0084 bad=0 nhwc_bad=0
rows=8 split=0.8519 shared_gateup=0.6028 shared_chain=0.8574 delta=+0.0055 shared_silu_nhwc=0.0293 shared_direct_down=0.2303 shared_nhwc_chain=0.8343 nhwc_delta=-0.0176 bad=0 nhwc_bad=0
```

This is the first bench-only direct-op candidate in this sequence where all target rows beat the row-aware split chain and pass accuracy.

## Build / Artifact

Build command:

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl CLEAN=0 \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_SEP_BUILD=OFF -DMNN_BUILD_LLM=OFF -DMNN_BUILD_CONVERTER=OFF -DMNN_LLM_BUILD_DEMO=OFF' \
BUILD_DIR="$PWD/.cache/build/mnn/aidlux_adreno_opencl_testbench_static" \
INSTALL_PREFIX="$PWD/.cache/output/mnn/artifacts/aidlux_adreno_opencl_testbench_static" \
JOBS=96 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

The script still removes the build dir because cached compiler/sysroot fields are empty, then rebuilds successfully.

Artifact:

```text
local:  .cache/output/mnn/artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out
remote: /mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out
```

## Decision And Next Step

Do not run endpoint yet. The current winning result is direct-op/testbench-only.

The next useful production route is narrow:

1. Preserve current row-aware gate/up projection selection.
2. Replace the `PicSiluMul(C4) -> down internal C4-to-NHWC preconvert` boundary with an activation path that directly produces NHWC for down.
3. If also using shared-input gate/up, keep it guarded to Rhino/Adreno tiny rows and verify rows `6/8` do not regress under endpoint TPOT.
4. Only after production routing exists, run endpoint decode repair:
   - ctx `512/1024`
   - max_tokens `32`
   - x `0/1/3/5/7`
   - repeat `>=3`
   - judge `extra_ms = TPOT(x)-TPOT(x=0)`, especially x `3/5/7`.

