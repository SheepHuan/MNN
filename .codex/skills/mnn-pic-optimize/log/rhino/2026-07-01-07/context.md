# Rhino Adreno Backend-Isolated Decode Repair Follow-Up

## Export And Runtime Isolation Rule

Adreno graph export experiments must stay isolated from CUDA/Jetson export and runtime execution.

The exporter now treats `--pic_decode_fusion_backend` as the family selector:

```text
generic                         -> no backend-native decode repair rewrites
cuda, jetson                    -> CUDA/Jetson Pic* family
adreno, rhino, rhinopi,
aidlux_adreno_opencl            -> Adreno-only PicAdreno* family
```

Expected op families:

```text
CUDA/Jetson:
  PicGateUpSiluWeightOnly
  PicLinearNhwcWeightOnly
  PicPackedSiluMul

Adreno/Rhino:
  PicAdrenoGateUpSiluWeightOnly
  PicAdrenoLinearNhwcWeightOnly
  PicAdrenoPackedSiluMul
  PicAdrenoSiluMulNhwc
  PicAdrenoTinyMlpWeightOnly
```

Runtime isolation:

- CUDA creator only recognizes the CUDA/Jetson `Pic*` family.
- OpenCL creator checks `OpenCLRuntime::getGpuType() == ADRENO` before creating any `PicAdreno*` Extra execution.
- OrangePi/Mali must not inherit `PicAdreno*`. If the same idea is tested on Mali, add a separate Mali/generic family and validate from direct-op through endpoint again.

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
  source/core/OpCommonUtils.cpp \
  source/backend/cuda/execution/FuseExecution.cu \
  source/backend/cuda/core/CUDABackend.cpp
```

Both checks passed.

## Direct-Op: SiluNhwcDown Boundary

Candidate:

```text
DecodeRepairMlpSiluNhwcDown
```

This candidate keeps gate/up as split projections, computes SiLU(gate) * up into NHWC, then runs down projection from NHWC. It does not fuse gate/up/down into one true tiny-row MLP kernel.

Direct-op Rhino rows `1..8`, hidden `1536`, inter `4608`, int4 weight-only:

| rows | split_chain ms | silu_nhwc ms | direct_down ms | fused_chain ms | delta ms | bad |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2836 | 0.0194 | 0.1239 | 0.3244 | +0.0408 | 0/1536 |
| 2 | 0.4672 | 0.0246 | 0.1265 | 0.4578 | -0.0094 | 0/3072 |
| 3 | 0.4751 | 0.0259 | 0.1305 | 0.4652 | -0.0100 | 0/4608 |
| 4 | 0.4837 | 0.0462 | 0.1256 | 0.4679 | -0.0158 | 0/6144 |
| 5 | 0.8362 | 0.0350 | 0.2356 | 0.8296 | -0.0067 | 0/7680 |
| 6 | 0.8425 | 0.0343 | 0.2349 | 0.8328 | -0.0096 | 0/9216 |
| 7 | 0.8508 | 0.0330 | 0.2360 | 0.8375 | -0.0133 | 0/10752 |
| 8 | 0.8515 | 0.0435 | 0.2333 | 0.8408 | -0.0106 | 0/12288 |

Direct-op interpretation:

- Accuracy passed for all tested rows.
- Rows `2..8` are slightly faster than row-aware split-chain.
- Row `1` regresses and cannot be enabled by default.
- The win is narrow: the candidate removes part of the activation/down layout overhead but does not reduce the two gate/up projection costs or the down projection cost.

## Endpoint Retry After hd128 Fix

Service/model:

```text
model: OpenBMB__MiniCPM5-1B-pic-boundary-adreno-silu-nhwc-down
port: 18133
selector: lagged_attention_hkvd
mode: full-reuse
max_tokens: 32
```

CSV:

```text
.cache/mnn-pic-benchmark/rhino_adreno_silu_nhwc_down_hd128_retry/decode.csv
```

Results:

| ctx | x | tpot ms | extra ms | rowheur2 ms | rowheur2 extra ms | delta tpot | delta extra |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0 | 64.620 | 0.000 | 73.437 | 0.000 | -8.817 | +0.000 |
| 512 | 1 | 81.528 | 16.908 | 77.481 | 4.044 | +4.047 | +12.864 |
| 512 | 3 | 89.352 | 24.732 | 95.045 | 21.608 | -5.693 | +3.124 |
| 512 | 5 | 130.098 | 65.478 | 129.896 | 56.459 | +0.202 | +9.019 |
| 512 | 7 | 133.460 | 68.840 | 130.734 | 57.297 | +2.726 | +11.543 |
| 1024 | 0 | 96.434 | 0.000 | 104.494 | 0.000 | -8.060 | +0.000 |
| 1024 | 1 | 106.858 | 10.424 | 114.729 | 10.235 | -7.871 | +0.189 |
| 1024 | 3 | 122.009 | 25.575 | 128.417 | 23.923 | -6.408 | +1.652 |
| 1024 | 5 | 152.032 | 55.598 | 149.506 | 45.012 | +2.526 | +10.586 |
| 1024 | 7 | 157.025 | 60.590 | 156.022 | 51.528 | +1.003 | +9.062 |

Endpoint interpretation:

- The x=0 no-repair path is healthy after the hd128 decode fix.
- The direct-op row win does not translate into endpoint `extra_ms` improvement.
- x=5/x=7 are worse than rowheur2 at both context lengths.
- Reject this candidate for production graph wiring.

## Current Conclusion

Do not promote `adreno-silu-nhwc-down`.

The useful permanent result is the backend isolation mechanism:

- Keep row-aware tiny dense family selection for Rhino:
  - rows `<=2`: `pic_quant_c4`
  - rows `<=4`: `pic_quant_incache64`
  - rows `>=6`: `adreno_batch_gemv_c4out`
- Keep Adreno graph rewrites behind the backend family selector.
- Do not replicate Adreno op names or runtime paths to OrangePi.
- Treat `--pic_decode_fusion_backend generic` as a no-native-decode-rewrite export path. `pic_decode_tiny_fusion` is now masked by the backend family too, so setting the flag without `cuda/jetson/adreno/rhino` does not emit `PicSiluMul` decode-repair graph rewrites by accident.

## Direct-Op: StreamedTile OC Tile Sweep

Question:

```text
Can increasing the streamed-tile down output tile reduce gate/up recompute enough to make the
single-kernel streamed MLP candidate competitive with row-aware split-chain?
```

Bench-only code change:

```text
MNN_BENCH_OPENCL_MLP_STREAMED_TILE_OC now accepts values up to 64.
```

Build:

```bash
MNN_TARGET_DEVICE=aidlux_adreno_opencl \
BUILD_TARGET=run_test.out BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
CMAKE_ARGS='-DMNN_BUILD_TEST=ON -DMNN_BUILD_SHARED_LIBS=OFF -DMNN_BUILD_LLM=OFF -DMNN_LLM_BUILD_DEMO=OFF' \
BUILD_DIR=.cache/build/mnn/aidlux_adreno_opencl_test_static \
INSTALL_PREFIX=.cache/output/mnn/artifacts/aidlux_adreno_opencl_test_static \
JOBS=96 bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

Artifact check:

```text
run_test.out: ELF 64-bit LSB executable, ARM aarch64
MNN_BUILD_SHARED_LIBS=OFF
MNN_BUILD_TEST=ON
MNN_OPENCL=ON
MNN_SEP_BUILD=OFF
```

Rhino command pattern:

```bash
ssh aidlux@192.168.101.227 'cd /mnt/nvme/mnn_pic_opencl && \
  env LD_LIBRARY_PATH=/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl_test_static/lib:/usr/lib:/usr/lib/aarch64-linux-gnu \
      LD_PRELOAD=/usr/lib/libOpenCL_adreno.so \
      MNN_BENCH_OPENCL_MLP_STREAMED_TILE_OC=<16|32|64> \
      MNN_BENCH_OPENCL_MLP_STREAMED_TILE_ROWS=2,4,6,8 \
      artifacts/aidlux_adreno_opencl_test_static/bin/run_test.out \
      bench_ops/opencl/perf/DecodeRepairMlpStreamedTile 3 0 1 x 2'
```

Results:

| oc_tile | rows | split_chain ms | stream_kernel ms | stream_chain ms | delta ms | bad |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2 | 0.7633 | 32.1769 | 32.2284 | +31.4651 | 0/3072 |
| 16 | 4 | 0.7757 | 33.3517 | 33.4788 | +32.7032 | 0/6144 |
| 16 | 6 | 1.4342 | 32.9594 | 33.0378 | +31.6036 | 0/9216 |
| 16 | 8 | 1.4478 | 33.9659 | 34.0381 | +32.5903 | 0/12288 |
| 32 | 2 | 0.7930 | 49.9878 | 49.9944 | +49.2014 | 0/3072 |
| 32 | 4 | 0.8171 | 51.1725 | 51.2372 | +50.4201 | 0/6144 |
| 32 | 6 | 1.4469 | 50.3650 | 50.3209 | +48.8740 | 0/9216 |
| 32 | 8 | 1.4810 | 51.5241 | 51.6944 | +50.2134 | 0/12288 |
| 64 | 2 | 0.7952 | 85.6787 | 85.8780 | +85.0828 | 0/3072 |
| 64 | 4 | 0.8108 | 87.3074 | 87.3358 | +86.5250 | 0/6144 |
| 64 | 6 | 1.4574 | 85.8729 | 86.0393 | +84.5819 | 0/9216 |
| 64 | 8 | 1.4581 | 87.0554 | 87.0599 | +85.6018 | 0/12288 |

Interpretation:

- Accuracy is acceptable for all rows and OC tile values.
- Increasing `OC_TILE` from the previous tiny default reduces the original streamed-tile catastrophe, but the best case is still about `32-34 ms`.
- Row-aware split-chain remains about `0.76-1.45 ms`, so the best streamed-tile result is still roughly `22x-42x` slower.
- Larger `OC_TILE=32/64` is worse, likely from register pressure and occupancy loss.
- Reject this direction at direct-op gate; do not wire it into exporter or endpoint models.

Next Rhino optimization should target one of:

- a true fused tiny-row MLP kernel:
  `gate/up weight-only projection -> SiLU(gate)*up -> down`, sharing input load, dequant scale, image weight reads, and avoiding full `[rows, inter]` writeback/readback;
- targeted tiny-row attention projection kernels if profile shows q/o/k/v projections dominate a specific decode-repair step.

## Service State

After rejecting the `adreno-silu-nhwc-down` endpoint result, Rhino port `18133` was restored to the last guarded tiny-MLP model:

```text
model: minicpm5-adreno-tiny-mlp
config: /mnt/nvme/mnn_pic_opencl/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp/config_opencl_greedy.json
kv-cache-dir: /mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/shared_kv/OpenBMB__MiniCPM5-1B-pic-boundary-adreno-tiny-mlp
runtime-cache-dir: /mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/runtime_cache/opencl
log: /mnt/nvme/mnn_pic_opencl/logs/pic_server_minicpm5_adreno_tiny_mlp_rows5_shared_nhwc_final.log
```

Check:

```text
/health returned HTTP 404, which is the current pic_server behavior for that route, but the port responded.
fuser/ps confirmed pic_server PID 559262 on port 18133 with the tiny-MLP config.
error grep for error|failed|invalid|target unavailable|async persistent|build program|cl_ returned no matches.
```
