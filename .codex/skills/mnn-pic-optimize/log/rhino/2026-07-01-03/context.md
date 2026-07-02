# Rhino Adreno NHWC Down Real-Graph Test

## Change

Runtime crash was traced to `PicAdrenoLinearNhwcWeightOnly` not being recognized as an Extra op with external weights. `OpCommonUtils::_isPicExtraWithExternalWeights` only allowed:

- `PicGateUpWeightOnly`
- `PicGateUpSiluWeightOnly`
- `PicLinearNhwcWeightOnly`

The generated Adreno graph uses `PicAdrenoLinearNhwcWeightOnly`, so `Pipeline` did not rebuild the op with `externalPath`. `PicLinearNhwcWeightOnlyBufExecution` then constructed its child low-memory Conv without a valid external weight file and crashed in `ConvolutionCommon::load`.

Fix:

- Add `PicAdrenoLinearNhwcWeightOnly` to `_isPicExtraWithExternalWeights`.

## Build And Smoke

Built Rhino artifact with:

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Synced to:

```text
aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Profile smoke:

- model: `OpenBMB__MiniCPM5-1B-pic-boundary-adreno-silu-nhwc-down`
- context: 512
- mode: full-reuse
- selector: `lagged_attention_hkvd`
- repair_tokens: 0
- max_tokens: 1
- result: ok

Profile logs confirmed:

- `PicAdrenoSiluMulNhwc` created.
- `PicAdrenoLinearNhwcWeightOnly` created.
- down projection selected `PicLinearNhwcWeightOnly resize_nhwc_kernel`.

## Decode Repair Matrix

Clean service, no `MNN_PIC_GRAPH_PROFILE`, no `MNN_PIC_DECODE_DEBUG`.

Output:

```text
.cache/mnn-pic-benchmark/rhino_adreno_silu_nhwc_down_externalfix_perf_clean/decode.csv
```

Results, TPOT ms:

| ctx | x | new | rowheur2 baseline | delta |
| --- | ---: | ---: | ---: | ---: |
| 512 | 0 | 1555.090 | 73.437 | +1481.653 |
| 512 | 1 | 97.979 | 77.481 | +20.498 |
| 512 | 3 | 108.242 | 95.045 | +13.197 |
| 512 | 5 | 131.309 | 129.896 | +1.413 |
| 512 | 7 | 134.401 | 130.734 | +3.667 |
| 1024 | 0 | 2935.879 | 104.494 | +2831.385 |
| 1024 | 1 | 110.817 | 114.729 | -3.912 |
| 1024 | 3 | 123.201 | 128.417 | -5.216 |
| 1024 | 5 | 151.551 | 149.506 | +2.045 |
| 1024 | 7 | 157.274 | 156.022 | +1.252 |

Separate no-repair rerun:

```text
.cache/mnn-pic-benchmark/rhino_adreno_silu_nhwc_down_externalfix_x0_rerun/decode.csv
```

Confirmed reproducible:

- ctx512 x=0: 1555.166 ms
- ctx1024 x=0: 2935.648 ms

## Interpretation

The Adreno real graph now runs and the repair path is valid. For `repair_tokens > 0`, ctx1024 is slightly better than rowheur2 at x=1/x=3 and near-neutral at x=5/x=7; ctx512 is neutral only at x=5/x=7 and worse at x=1/x=3.

The `repair_tokens=0` path is a separate regression. It does not send `decode_refine`, so it exercises the no-repair decode execution route, not the sparse decode-repair route. Because x=0 regresses by 21x-28x and reproduces after warm/cache, OrangePi replication should wait until no-repair decode is separated from this Adreno graph experiment or fixed.

## Follow-Up: Adreno hd128 No-Repair Decode Fix

The x=0 profile showed that no-repair full-reuse did not call `decode_refine`, but the graph-boundary model still contains `PicScoreAttention` / `PicSparseAttention` op types. OpenCL already reduced their effective PIC mode to normal attention when no sparse runtime state was active, but `headDim=128` ordinary decode had no causal fast path. It fell through to the slow generic path, causing the earlier `PicSparseAttention total_ms=2536.837 calls=44` profile.

Fix:

- Reuse the existing hd128 causal decode kernel for ordinary decode by allowing `sparse_query_active=0`.
- Keep repair decode behavior unchanged: sparse repair still passes `sparseQuery=1`.
- Enable ordinary hd128 fast path only on Adreno, so this remains a Rhino/Adreno OpenCL runtime fix.
- Keep graph export isolation unchanged: `PicAdreno*` ops are still emitted only with `--pic_decode_fusion_backend adreno/rhino`; CUDA/Jetson export and execution are not changed by this fix.

Changed files for this follow-up:

```text
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.hpp
source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp
```

Build:

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

Note: the build script still reports Rhino cross build cache C/CXX/sysroot as `<empty>` and deletes the build dir before configure. This is the known build-cache diagnosis issue, not a source regression.

Profile smoke service:

```bash
env LD_LIBRARY_PATH=/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/lib:/usr/lib:/usr/lib/aarch64-linux-gnu \
    LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
    MNN_LLM_RUNTIME_CACHE_DIR=/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/runtime_cache/opencl \
    MNN_PAGED_ATTENTION_PROFILE=1 \
    artifacts/aidlux_adreno_opencl/bin/pic_server \
    --config /mnt/nvme/mnn_pic_opencl/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp/config_opencl_greedy.json \
    --host 0.0.0.0 --port 18133 \
    --kv-cache-dir /mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/shared_kv/OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp \
    --runtime-cache-dir /mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/runtime_cache/opencl \
    --model minicpm5-adreno-tiny-mlp
```

Profile evidence for ctx512 x=0:

```text
OpenCLPagedAttention profile op=decode_causal_attention_hd128 layer=0  query=1 kv_len=513 lane=64 us=992
OpenCLPagedAttention profile op=decode_causal_attention_hd128 layer=1  query=1 kv_len=513 lane=64 us=875
...
OpenCLPagedAttention profile op=decode_causal_attention_hd128 layer=23 query=1 kv_len=513 lane=64 us=833
```

No `ERROR`, `target unavailable`, or `async persistent PIC cache read failed` was found in the no-profile service log.

## Tiny-MLP Real-Graph Matrix After hd128 Fix

Model:

```text
/mnt/nvme/mnn_pic_opencl/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp/config_opencl_greedy.json
model id: minicpm5-adreno-tiny-mlp
```

Command shape:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://192.168.101.227:18133 \
  --output-csv .cache/mnn-pic-benchmark/rhino_adreno_tiny_mlp_hd128_matrix/decode.csv \
  --output-dir .cache/mnn-pic-benchmark/rhino_adreno_tiny_mlp_hd128_matrix/raw \
  --device rhino --allow-extra-device --device-display 'Rhino Pi-X1' \
  --backend opencl-adreno --frequency-profile max \
  --model minicpm5-adreno-tiny-mlp --model-name MiniCPM5-1B \
  --model-config /mnt/nvme/mnn_pic_opencl/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp/config_opencl_greedy.json \
  --mode full-reuse --contexts 512,1024 --budgets 0.00 \
  --repair-tokens 0,1,3,5,7 \
  --decode-selectors lagged_attention_hkvd --attention-layer-idx 1 \
  --max-tokens 32 --repeats 3 --warm-repeats 1 \
  --require-exact-context --timeout 600
```

Results, TPOT ms:

| ctx | x | tiny-MLP graph | rowheur2 baseline | speedup vs rowheur2 | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 | 0 | 66.150 | 73.437 | 1.110x | -7.287 |
| 512 | 1 | 82.382 | 77.481 | 0.941x | +4.901 |
| 512 | 3 | 97.167 | 95.045 | 0.978x | +2.122 |
| 512 | 5 | 130.286 | 129.896 | 0.997x | +0.390 |
| 512 | 7 | 134.790 | 130.734 | 0.970x | +4.056 |
| 1024 | 0 | 98.284 | 104.494 | 1.063x | -6.210 |
| 1024 | 1 | 103.838 | 114.729 | 1.105x | -10.891 |
| 1024 | 3 | 125.849 | 128.417 | 1.020x | -2.568 |
| 1024 | 5 | 154.560 | 149.506 | 0.967x | +5.054 |
| 1024 | 7 | 158.041 | 156.022 | 0.987x | +2.019 |

Separate x=0 spot checks:

```text
profile on, ctx512 x=0 max_tokens=1: 708.295 ms TPOT, with per-layer queue-finish profiling
profile off, ctx512 x=0 max_tokens=1: 63.260 ms TPOT
profile off, ctx512 x=0 max_tokens=32: 65.001 ms TPOT
```

Interpretation:

- The catastrophic x=0 regression is fixed. No-repair decode now uses the hd128 causal decode kernel rather than generic `PicSparseAttention` execution.
- The real tiny-MLP graph is now safe to benchmark, but it is not yet a clear promotion candidate. It is healthy at x=0 and helpful for ctx1024 x=1/x=3, but still slightly worse than rowheur2 for ctx512 x=1/x=3/x=7 and ctx1024 x=5/x=7.
- Do not replicate this exact tiny-MLP graph to OrangePi yet. First decide whether to preserve row-aware split-chain gate/up and only inherit the useful NHWC activation/down boundary, or add a true fused tiny-row MLP kernel that avoids the current child Conv overhead without losing the row-aware Adreno dense family selection.

## Follow-Up: Backend-Isolated Adreno Export Ops

Constraint from follow-up review:

- Adreno graph export experiments must not share graph op definitions with CUDA/Jetson paths.
- Export should choose the backend family explicitly, and Adreno/Rhino-only rewrites must stay behind `--pic_decode_fusion_backend adreno/rhino/rhinopi/aidlux_adreno_opencl`.
- CUDA/Jetson export and execution should continue using their own existing op names and runtime classes.

Tightening applied:

- Added Python symbolic/module `PicAdrenoGateUpSiluWeightOnly`.
- Changed Adreno MLP gate/up+silu export wrapper to emit `LlmExporter::PicAdrenoGateUpSiluWeightOnly` instead of the CUDA-recognized `LlmExporter::PicGateUpSiluWeightOnly`.
- Changed converter direct gate/up+silu output to final MNN Extra type `PicAdrenoGateUpSiluWeightOnly`.
- Kept the existing CUDA `PicGateUpSiluWeightOnly` execution and CUDA bench untouched.
- Added shape and external-weight rebuild support for `PicAdrenoGateUpSiluWeightOnly`.
- Added OpenCL creator guard: `PicAdrenoSiluMulNhwc`, `PicAdrenoGateUpSiluWeightOnly`, `PicAdrenoTinyMlpWeightOnly`, and `PicAdrenoLinearNhwcWeightOnly` are only created when `OpenCLRuntime::getGpuType() == ADRENO`; non-Adreno OpenCL runtimes return unsupported instead of silently consuming Adreno experiment ops.

Verification:

```bash
conda run -n kvshare-edge python -m py_compile \
  transformers/pic_llm/export/llmexport.py \
  transformers/pic_llm/export/utils/model.py \
  transformers/pic_llm/export/utils/transformers.py \
  transformers/pic_llm/export/utils/custom_op.py \
  transformers/pic_llm/export/utils/mnn_converter.py

git diff --check -- \
  transformers/pic_llm/export/utils/custom_op.py \
  transformers/pic_llm/export/utils/transformers.py \
  transformers/pic_llm/export/utils/mnn_converter.py \
  source/shape/ShapePicExtra.cpp \
  source/core/OpCommonUtils.cpp \
  source/backend/opencl/execution/buffer/FuseBufExecution.cpp
```

Both checks passed.

Rhino cross-build:

```bash
JOBS=96 \
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Build result:

```text
[100%] Built target pic_server
[build_artifacts] Installed libMNN.so
[build_artifacts] Installed libMNN_Express.so
[build_artifacts] Installed libMNN_CL.so
[build_artifacts] Installed libpic_llm.so
[build_artifacts] Installed pic_server
```

Artifact check:

```text
.cache/output/mnn/artifacts/aidlux_adreno_opencl/bin/pic_server:   ELF 64-bit LSB executable, ARM aarch64
.cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN.so:    ELF 64-bit LSB shared object, ARM aarch64
.cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN_CL.so: ELF 64-bit LSB shared object, ARM aarch64
```

Conclusion:

- Adreno direct gate/up+silu experiment is now graph-name isolated from CUDA/Jetson.
- `--pic_decode_fusion_backend=generic/cuda/jetson` still leaves Adreno rewrites disabled.
- OrangePi should get its own backend family if the same idea is replicated; do not reuse `PicAdreno*` op names as the Mali/OpenCL production interface.

## Follow-Up: Shared-NHWC Tiny-MLP Runtime In Real Graph

Goal:

- Move the bench-only `shared_nhwc_chain` idea into real generated Adreno tiny-MLP graph execution.
- Preserve CUDA/Jetson isolation: graph export still requires `--pic_decode_fusion_backend adreno/rhino/rhinopi/aidlux_adreno_opencl`, and OpenCL creation of `PicAdreno*` ops is still rejected on non-Adreno GPUs.
- Avoid replacing the row-aware tiny dense conclusion globally. Rows `<=2`, `<=4`, and `>=6` had different Adreno family choices in the valid baseline, so any production fast path must be judged by endpoint TPOT, not only by direct-op rows.

Runtime implementation tested:

- `PicAdrenoTinyMlpWeightOnlyBufExecution` gained a shared-NHWC recorded chain:
  - `gemm_c4nhw4_to_nhwc`
  - gate `gemv_conv_c8_buf OUTPUT_C4NHW4`
  - up `gemv_conv_c8_buf OUTPUT_C4NHW4`
  - `pic_silu_mul_c4_to_nhwc_buf`
  - down `gemv_conv_c8_buf OUTPUT_C4NHW4`
- Fallback remains the row-aware child Conv chain:
  - gate child `ConvBufLowMemoryExecution`
  - up child `ConvBufLowMemoryExecution`
  - `PicSiluMul`
  - down child `ConvBufLowMemoryExecution`
- The final guard is intentionally narrow:
  - Adreno only
  - MiniCPM5-1B shape `1536 -> 4608 -> 1536`
  - int4 image-backed weights
  - rows `5..8`
- Earlier rows `1..8` and rows `2..8` tests were rejected:
  - rows `1..8` made x=0 regress because ordinary decode row=1 used the shared-NHWC chain.
  - rows `2..8` fixed x=0 but made ctx512 x=3 worse; row3 was not covered by the direct-op rows `2/4/6/8` evidence.

Additional isolation fix:

- Converter now requires Adreno backend only for `PicAdrenoGateUpSiluWeightOnly`.
- Legacy `PicGateUpSiluWeightOnly` keeps the previous CUDA/generic lowering behavior, so existing Jetson direct-op experiments are not blocked by the Adreno guard.

Backend-family isolation follow-up:

- `--pic_decode_fusion_backend` is now treated as a family selector:
  - `generic`: no backend-native gate/up or NHWC linear graph rewrite is enabled by the flags.
  - `cuda` / `jetson`: keep the Jetson CUDA family and emit `PicGateUpSiluWeightOnly`, `PicLinearNhwcWeightOnly`, and `PicPackedSiluMul`.
  - `adreno` / `rhino` / `rhinopi` / `aidlux_adreno_opencl`: emit Adreno-only graph ops such as `PicAdrenoGateUpSiluWeightOnly`, `PicAdrenoLinearNhwcWeightOnly`, and `PicAdrenoTinyMlpWeightOnly`.
- This fixes a review issue where `cuda/jetson` appeared in the argparse choices but the export model/config path only enabled Adreno family rewrites, which would have disabled the Jetson NHWC gate/up path from the 2026-06-30 Jetson log.
- Adreno-only `silu_nhwc_down_fusion` and tiny-MLP fusion remain gated to the Adreno family.
- OpenCL creator still rejects `PicAdreno*` ops on non-Adreno runtimes; CUDA execution continues to recognize only the CUDA/generic `Pic*` ops.

Static checks:

```bash
git diff --check

conda run -n kvshare-edge python -m py_compile \
  transformers/pic_llm/export/llmexport.py \
  transformers/pic_llm/export/utils/custom_op.py \
  transformers/pic_llm/export/utils/model.py \
  transformers/pic_llm/export/utils/transformers.py \
  transformers/pic_llm/export/utils/mnn_converter.py
```

Both passed.

Build / sync:

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

Final artifact check:

```text
.cache/output/mnn/artifacts/aidlux_adreno_opencl/bin/pic_server:   ELF 64-bit LSB executable, ARM aarch64
.cache/output/mnn/artifacts/aidlux_adreno_opencl/lib/libMNN_CL.so: ELF 64-bit LSB shared object, ARM aarch64
timestamp: 2026-07-01 06:08:22 +0800
```

Direct-op confirmation:

```bash
ssh aidlux@192.168.101.227 'cd /mnt/nvme/mnn_pic_opencl && \
  env LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
      MNN_BENCH_OPENCL_MLP_SHARED_INPUT_ROWS=1,2,3,4,5,6,7,8 \
      MNN_BENCH_OPENCL_MLP_SHARED_INPUT_WGS=64 \
      MNN_BENCH_OPENCL_MLP_SHARED_INPUT_WARMUP=40 \
      MNN_BENCH_OPENCL_MLP_SHARED_INPUT_REPEAT=160 \
      artifacts/aidlux_adreno_opencl_testbench_static/bin/run_test.out \
      bench_ops/opencl/perf/DecodeRepairMlpSharedInputGateUp 3 2 1 empty 2'
```

The static testbench was extended so this run covers rows `1..8`. `shared_nhwc_chain` is accurate for all rows, but row `1` is slower than split-chain while rows `2..8` are faster:

```text
rows=1 split_chain=0.2862 shared_nhwc_chain=0.4563 nhwc_delta=+0.1701 bad=0 nhwc_bad=0
rows=2 split_chain=0.4712 shared_nhwc_chain=0.4542 nhwc_delta=-0.0170 bad=0 nhwc_bad=0
rows=3 split_chain=0.4783 shared_nhwc_chain=0.4522 nhwc_delta=-0.0261 bad=0 nhwc_bad=0
rows=4 split_chain=0.4881 shared_nhwc_chain=0.4529 nhwc_delta=-0.0352 bad=0 nhwc_bad=0
rows=5 split_chain=0.8390 shared_nhwc_chain=0.8348 nhwc_delta=-0.0042 bad=0 nhwc_bad=0
rows=6 split_chain=0.8465 shared_nhwc_chain=0.8373 nhwc_delta=-0.0092 bad=0 nhwc_bad=0
rows=7 split_chain=0.8528 shared_nhwc_chain=0.8393 nhwc_delta=-0.0135 bad=0 nhwc_bad=0
rows=8 split_chain=0.8541 shared_nhwc_chain=0.8390 nhwc_delta=-0.0151 bad=0 nhwc_bad=0
```

Rows `2..8` endpoint retry:

```bash
NO_PROXY=192.168.101.227,127.0.0.1,localhost \
no_proxy=192.168.101.227,127.0.0.1,localhost \
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://192.168.101.227:18133 \
  --output-csv .cache/mnn-pic-benchmark/rhino_adreno_tiny_mlp_rows2_8_shared_nhwc_runtime_retry/decode.csv \
  --output-dir .cache/mnn-pic-benchmark/rhino_adreno_tiny_mlp_rows2_8_shared_nhwc_runtime_retry/raw \
  --device rhino --allow-extra-device --device-display 'Rhino Pi-X1' \
  --backend opencl-adreno --frequency-profile max \
  --model minicpm5-adreno-tiny-mlp --model-name MiniCPM5-1B \
  --model-config /mnt/nvme/mnn_pic_opencl/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp/config_opencl_greedy.json \
  --mode full-reuse --contexts 512,1024 --budgets 0.00 \
  --repair-tokens 0,1,3,5,7 \
  --decode-selectors lagged_attention_hkvd --attention-layer-idx 1 \
  --max-tokens 32 --repeats 3 --warm-repeats 1 \
  --require-exact-context --timeout 600
```

Rows `2..8` endpoint TPOT / extra-ms:

| ctx | x | rows2..8 TPOT | rows2..8 extra | rowheur2 TPOT | rowheur2 extra | delta TPOT | delta extra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0 | 64.476 | 0.000 | 73.437 | 0.000 | -8.961 | +0.000 |
| 512 | 1 | 85.444 | 20.969 | 77.481 | 4.044 | +7.963 | +16.925 |
| 512 | 3 | 103.956 | 39.480 | 95.045 | 21.608 | +8.911 | +17.872 |
| 512 | 5 | 130.198 | 65.722 | 129.896 | 56.459 | +0.302 | +9.263 |
| 512 | 7 | 134.412 | 69.936 | 130.734 | 57.297 | +3.678 | +12.639 |
| 1024 | 0 | 86.205 | 0.000 | 104.494 | 0.000 | -18.289 | +0.000 |
| 1024 | 1 | 109.581 | 23.376 | 114.729 | 10.235 | -5.148 | +13.141 |
| 1024 | 3 | 122.762 | 36.558 | 128.417 | 23.923 | -5.655 | +12.635 |
| 1024 | 5 | 150.648 | 64.443 | 149.506 | 45.012 | +1.142 | +19.431 |
| 1024 | 7 | 156.235 | 70.030 | 156.022 | 51.528 | +0.213 | +18.502 |

This rejects rows `2..8` as a production default even though direct-op rows `2..8` are locally faster. The endpoint objective is based on repair overhead over x=0, and `extra_ms` worsens at every repair setting.

Rows `5..8` endpoint command:

```bash
NO_PROXY=192.168.101.227,127.0.0.1,localhost \
no_proxy=192.168.101.227,127.0.0.1,localhost \
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py \
  --base-url http://192.168.101.227:18133 \
  --output-csv .cache/mnn-pic-benchmark/rhino_adreno_tiny_mlp_rows5_shared_nhwc_runtime/decode.csv \
  --output-dir .cache/mnn-pic-benchmark/rhino_adreno_tiny_mlp_rows5_shared_nhwc_runtime/raw \
  --device rhino --allow-extra-device --device-display 'Rhino Pi-X1' \
  --backend opencl-adreno --frequency-profile max \
  --model minicpm5-adreno-tiny-mlp --model-name MiniCPM5-1B \
  --model-config /mnt/nvme/mnn_pic_opencl/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp/config_opencl_greedy.json \
  --mode full-reuse --contexts 512,1024 --budgets 0.00 \
  --repair-tokens 0,1,3,5,7 \
  --decode-selectors lagged_attention_hkvd --attention-layer-idx 1 \
  --max-tokens 32 --repeats 3 --warm-repeats 1 \
  --require-exact-context --timeout 600
```

Final rows `5..8` TPOT:

| ctx | x | rows5 shared | extra vs x0 | rowheur2 | delta vs rowheur2 | previous tiny | delta vs previous tiny |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0 | 63.922 | 0.000 | 73.437 | -9.515 | 66.150 | -2.228 |
| 512 | 1 | 71.783 | +7.861 | 77.481 | -5.698 | 82.382 | -10.599 |
| 512 | 3 | 94.238 | +30.316 | 95.045 | -0.807 | 97.167 | -2.929 |
| 512 | 5 | 130.116 | +66.194 | 129.896 | +0.220 | 130.286 | -0.170 |
| 512 | 7 | 132.560 | +68.638 | 130.734 | +1.826 | 134.790 | -2.230 |
| 1024 | 0 | 109.782 | 0.000 | 104.494 | +5.288 | 98.284 | +11.498 |
| 1024 | 1 | 105.211 | -4.571 | 114.729 | -9.518 | 103.838 | +1.373 |
| 1024 | 3 | 124.737 | +14.954 | 128.417 | -3.680 | 125.849 | -1.112 |
| 1024 | 5 | 151.408 | +41.626 | 149.506 | +1.902 | 154.560 | -3.152 |
| 1024 | 7 | 156.039 | +46.256 | 156.022 | +0.017 | 158.041 | -2.002 |

The ctx1024 x=0 row was noisy. A separate x=0 rerun under the same service produced:

```text
ctx1024 x=0 TPOT=96.741 ms
```

Log check:

```bash
grep -Ei "error|failed|invalid|target unavailable|async persistent|build program|cl_" \
  /mnt/nvme/mnn_pic_opencl/logs/pic_server_minicpm5_adreno_tiny_mlp_rows5_shared_nhwc.log
```

No matches.

Conclusion:

- Direct-op evidence is valid for rows `2/4/6/8`, but it is not sufficient to promote a broad production route.
- The broad rows `2..8` runtime route is rejected because ctx512 x=3 regresses.
- The narrow rows `5..8` route is safer and improves the previous tiny-MLP graph for most repair cases, but it still does not clearly reduce x=5/x=7 against rowheur2.
- This does not satisfy the original endpoint goal. The next credible Rhino step remains a real fused tiny-row MLP design that preserves row-aware family selection, or a graph route that keeps row-aware gate/up/down and only removes the proven layout boundary cost.
- Do not replicate this exact Adreno path on OrangePi. If Mali is tested, use a separate `pic_decode_fusion_backend` family and Mali-specific `PicMali*` or generic OpenCL op names, then rerun direct-op and endpoint from scratch.

Final source / remote cleanup after the rows `2..8` retry:

- `PicAdrenoTinyMlpWeightOnlyBufExecution::canUseFast` is rows `5..8`.
- Ordinary `PicGateUpWeightOnlyBufExecution::canUseFused` keeps the original rows `1..8` guard, so the rejected Adreno tiny-MLP retry does not alter the generic OpenCL gate/up op.
- Rebuilt Rhino artifact with:

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Build result:

```text
[100%] Built target pic_server
[build_artifacts] Installed libMNN.so
[build_artifacts] Installed libMNN_Express.so
[build_artifacts] Installed libMNN_CL.so
[build_artifacts] Installed libpic_llm.so
[build_artifacts] Installed pic_server
```

Local and remote artifact timestamp:

```text
2026-07-01 07:32:33 +0800 pic_server
2026-07-01 07:32:33 +0800 libMNN_CL.so
```

Synced artifact:

```bash
rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Restarted Rhino service on port `18133` with:

```text
log: /mnt/nvme/mnn_pic_opencl/logs/pic_server_minicpm5_adreno_tiny_mlp_rows5_shared_nhwc_final.log
pid: 540891
```

Smoke checks:

```text
curl /health: HTTP 404, expected for this server but confirms the port responds.
error grep: no matches for error/failed/invalid/target unavailable/async persistent/build program/cl_
```

## Follow-Up: Packed Silu Alias Isolation

Review constraint:

- Adreno/Rhino graph export experiments should not reuse CUDA/Jetson op names for backend-specific graph rewrites.
- The backend family selector must make the generated op family obvious from the graph, so CUDA/Jetson changes can be reviewed independently from Adreno experiments.

Tightening applied:

- Added converter helper `_pic_packed_silu_op_type()`.
- `--pic_decode_fusion_backend cuda/jetson` keeps `PicPackedSiluMul`.
- `--pic_decode_fusion_backend adreno/rhino/rhinopi/aidlux_adreno_opencl` emits `PicAdrenoPackedSiluMul`.
- `ShapePicExtra` treats `PicAdrenoPackedSiluMul` as a shape alias of the existing packed-silu op.
- OpenCL `FuseBufCreator` recognizes `PicAdrenoPackedSiluMul` only after the existing `PicAdreno*` runtime guard, so non-Adreno OpenCL runtimes do not silently consume this experiment op.
- CUDA registration remains unchanged and still only accepts `PicPackedSiluMul`.

Verification:

```bash
conda run -n kvshare-edge python -m py_compile \
  transformers/pic_llm/export/llmexport.py \
  transformers/pic_llm/export/utils/model.py \
  transformers/pic_llm/export/utils/transformers.py \
  transformers/pic_llm/export/utils/custom_op.py \
  transformers/pic_llm/export/utils/mnn_converter.py

git diff --check -- \
  transformers/pic_llm/export/llmexport.py \
  transformers/pic_llm/export/utils/model.py \
  transformers/pic_llm/export/utils/transformers.py \
  transformers/pic_llm/export/utils/custom_op.py \
  transformers/pic_llm/export/utils/mnn_converter.py \
  source/backend/opencl/execution/buffer/FuseBufExecution.cpp \
  source/shape/ShapePicExtra.cpp \
  source/core/OpCommonUtils.cpp
```

Both checks passed.

Backend selector smoke:

```text
generic: family=generic linear=PicLinearNhwcWeightOnly packed_silu=PicPackedSiluMul
cuda:    family=cuda    linear=PicLinearNhwcWeightOnly packed_silu=PicPackedSiluMul
jetson:  family=cuda    linear=PicLinearNhwcWeightOnly packed_silu=PicPackedSiluMul
adreno:  family=adreno  linear=PicAdrenoLinearNhwcWeightOnly packed_silu=PicAdrenoPackedSiluMul
rhino:   family=adreno  linear=PicAdrenoLinearNhwcWeightOnly packed_silu=PicAdrenoPackedSiluMul
```

## Follow-Up: True Gate/Up Pair + Silu-Down Direct-Op Candidate

Goal:

- Test a stronger direct-op direction before touching real generated graph code:
  - shared input preconvert
  - fused gate/up weight-only projection with pair output
  - fused `SiLU(gate) * up -> down_proj`
- Preserve the existing row-aware dense family decision:
  - rows `<=2`: `pic_quant_c4`
  - rows `<=4`: `pic_quant_incache64`
  - rows `>=6`: `adreno_batch_gemv_c4out`
- Only consider production wiring if rows `2/4/6/8` are both accurate and faster than the row-aware split-chain baseline.

Bench-only implementation:

- Extended `resource_gateup_silu_int4_image` with `RESOURCE_GATEUP_PAIR_OUTPUT` to share input load/dequant while writing separate C4 gate/up tensors.
- Added `resource_silu_down_c4_int4_image` direct-op to apply `SiLU(gate) * up` inside down projection.
- Added `bench_ops/opencl/perf/DecodeRepairMlpGateUpSiluDownC4`.

Testbench build:

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_DIR=.cache/build/mnn/aidlux_adreno_opencl_test_static \
INSTALL_PREFIX=.cache/output/mnn/artifacts/aidlux_adreno_opencl_test_static \
BUILD_TARGET=run_test.out \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_LLM=OFF -DMNN_LLM_BUILD_DEMO=OFF' \
JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Notes:

- The normal shared Rhino `run_test.out` build failed because shared sep-build hides internal `MNN_CL` symbols that OpenCL direct-op benches call.
- A static testbench with LLM disabled links cleanly and keeps this experiment out of the production shared `pic_server/libMNN_CL.so` artifact.

Rhino command:

```bash
ssh aidlux@192.168.101.227 'cd /mnt/nvme/mnn_pic_opencl && \
  env LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
      MNN_BENCH_OPENCL_MLP_GATEUP_SILU_DOWN_ROWS=2,4,6,8 \
      MNN_BENCH_OPENCL_MLP_GATEUP_SILU_DOWN_WARMUP=40 \
      MNN_BENCH_OPENCL_MLP_GATEUP_SILU_DOWN_REPEAT=160 \
      artifacts/aidlux_adreno_opencl_test_static/bin/run_test.out \
      bench_ops/opencl/perf/DecodeRepairMlpGateUpSiluDownC4 3 2 1 empty 2'
```

Results:

| rows | gate family | down family | split chain ms | gateup pair ms | silu-down ms | fused chain ms | delta ms | bad |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | `pic_quant_c4` | `pic_quant_c4` | 0.4703 | 0.4067 | 0.1801 | 0.5799 | +0.1096 | 0/3072 |
| 4 | `pic_quant_incache64` | `pic_quant_incache64` | 0.4902 | 0.4459 | 0.2951 | 0.7360 | +0.2458 | 0/6144 |
| 6 | `adreno_batch_gemv_c4out` | `adreno_batch_gemv_c4out` | 0.8468 | 0.8244 | 0.4609 | 1.2727 | +0.4259 | 0/9216 |
| 8 | `adreno_batch_gemv_c4out` | `adreno_batch_gemv_c4out` | 0.8550 | 0.8699 | 0.5456 | 1.4135 | +0.5585 | 0/12288 |

Conclusion:

- Accuracy passed, but performance failed for every tested row count.
- `gateup_pair` alone saves some time for rows `2/4/6`, but the `silu_down` kernel costs more than the saved gate/up work. The full fused chain is slower than the existing row-aware split chain on rows `2/4/6/8`.
- Do not wire this candidate into real generated code.
- Do not carry this candidate to OrangePi. OrangePi/Mali should only be tested after a Rhino candidate wins direct-op and endpoint metrics, and it should use a separate Mali/generic backend family rather than `PicAdreno*`.
- The next viable direction is not "pair-output then silu-down" in two kernels. It needs either:
  - a true single-kernel tiny-row MLP that keeps gate/up partials in registers/local memory and streams down weights before writing final hidden rows, or
  - a narrower real-graph route that preserves row-aware gate/up/down child Conv family selection and removes only proven layout/activation overhead.

## Follow-Up: Streamed-Tile Single-Kernel Direct-Op Candidate

Goal:

- Test a more aggressive bench-only direction before touching real generated graph code:
  - preconvert hidden C4 input to NHWC once
  - launch one streamed MLP kernel
  - compute gate/up projection, `SiLU(gate) * up`, and down accumulation without writing a full `[rows, inter]` intermediate tensor
- Preserve the same row-aware split-chain baseline for comparison:
  - rows `<=2`: `pic_quant_c4`
  - rows `<=4`: `pic_quant_incache64`
  - rows `>=6`: `adreno_batch_gemv_c4out`

Important limitation:

- This prototype computes gate/up again for every hidden output tile. It removes the large intermediate write/read, but multiplies gate/up projection work by the number of output tiles. The test is useful as evidence for rejecting this schedule, not as a production candidate.

Bench-only implementation:

- Added `bench_ops/opencl/perf/DecodeRepairMlpStreamedTile`.
- Added OpenCL kernel `resource_streamed_tile_mlp_int4_image` in the opencl bench source.
- Kept it in the static `run_test.out` direct-op testbench only; it is not wired into graph export, `pic_server`, or runtime op creation.

Testbench build:

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_DIR=.cache/build/mnn/aidlux_adreno_opencl_test_static \
INSTALL_PREFIX=.cache/output/mnn/artifacts/aidlux_adreno_opencl_test_static \
BUILD_TARGET=run_test.out \
BUILD_MNNCONVERT=0 \
INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_LLM=OFF -DMNN_LLM_BUILD_DEMO=OFF' \
JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Build result:

```text
[100%] Built target run_test.out
[build_artifacts] Installed run_test.out -> .cache/output/mnn/artifacts/aidlux_adreno_opencl_test_static/bin
```

Artifact check:

```text
.cache/output/mnn/artifacts/aidlux_adreno_opencl_test_static/bin/run_test.out:
ELF 64-bit LSB executable, ARM aarch64
```

Rhino command:

```bash
ssh aidlux@192.168.101.227 'cd /mnt/nvme/mnn_pic_opencl && \
  env LD_LIBRARY_PATH=/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl_test_static/lib:/usr/lib:/usr/lib/aarch64-linux-gnu \
      LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
      MNN_BENCH_OPENCL_MLP_STREAMED_TILE_ROWS=2,4,6,8 \
      MNN_BENCH_OPENCL_MLP_STREAMED_TILE_WARMUP=2 \
      MNN_BENCH_OPENCL_MLP_STREAMED_TILE_REPEAT=5 \
      MNN_BENCH_OPENCL_MLP_STREAMED_TILE_OC=4 \
      artifacts/aidlux_adreno_opencl_test_static/bin/run_test.out \
      bench_ops/opencl/perf/DecodeRepairMlpStreamedTile 3 2 1 empty 2'
```

Results:

| rows | gate family | down family | split chain ms | stream kernel ms | stream chain ms | delta ms | bad |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | `pic_quant_c4` | `pic_quant_c4` | 0.6142 | 176.8694 | 177.4538 | +176.8396 | 0/3072 |
| 4 | `pic_quant_incache64` | `pic_quant_incache64` | 0.6296 | 178.7460 | 177.7310 | +177.1014 | 0/6144 |
| 6 | `adreno_batch_gemv_c4out` | `adreno_batch_gemv_c4out` | 0.9580 | 432.9022 | 432.9456 | +431.9876 | 0/9216 |
| 8 | `adreno_batch_gemv_c4out` | `adreno_batch_gemv_c4out` | 0.9644 | 437.3450 | 439.1252 | +438.1608 | 0/12288 |

Conclusion:

- Accuracy passed, but performance failed by two to three orders of magnitude.
- The schedule is fundamentally wrong for Adreno decode repair because it repeats gate/up projection per hidden output tile.
- Do not wire this direct-op into real generated code.
- Do not carry this candidate to OrangePi. If Mali is explored later, use a separate Mali/generic backend family and start with direct-op plus endpoint validation from scratch.
- The next credible Rhino path must keep gate/up work reusable across down accumulation, or avoid fused MLP entirely and only remove already-proven layout/activation overhead while preserving row-aware child Conv family selection.
