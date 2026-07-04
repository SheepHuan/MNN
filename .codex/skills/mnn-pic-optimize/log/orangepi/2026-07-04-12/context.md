# Context

## Code Change

Changed `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`:

```cpp
static bool _decodeRepairQ1RowEnabled() {
    // active_rows=1 is the degenerate decode-repair case; keep it in the same
    // transposed-K sparse family and allow env=0 only as an explicit A/B variant.
    static const bool enabled = _envFlagEnabled("MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_ROW", true);
    return enabled;
}
```

This does not route `x=0` to ordinary q=1 decode. The route remains:

- `forwardRaw`: PIC decode-repair enabled selects PIC `llm_decode.mnn`.
- PagedAttention: `pic_decode_recompute_active && sparseQuery` enters decode-repair sparse qtile route.
- `attnLen==1` selects the row kernel as a family-internal parameter.
- `MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_ROW=0` is preserved as an explicit A/B disable.

## Build And Sync

```bash
git diff --check

JOBS=$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 )) \
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

file .cache/output/mnn/artifacts/orangepi5plus/bin/pic_server \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN.so \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN_CL.so \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libpic_llm.so

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

Build result:

- `pic_server`: ARM aarch64 ELF executable.
- `libMNN.so`, `libMNN_CL.so`, `libpic_llm.so`: ARM aarch64 shared libraries.

## Formal Measure Command

No `PIC_SWEEP_SERVER_ENV_EXTRA` was set for q1-row.

```bash
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512,1024 \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --skip-normal \
  --run-id decode_repair_q1row_default_orangepi_ctx512_1024_pic_20260704_122350
```

Outputs:

```text
.cache/mnn-pic-benchmark/decode_experiment/decode_repair_q1row_default_orangepi_ctx512_1024_pic_20260704_122350/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_repair_q1row_default_orangepi_ctx512_1024_pic_20260704_122350/decode_tpot_wide.csv
```

P0 grep:

```bash
ssh orangepi@192.168.101.113 \
  "grep -R 'ERROR\|target unavailable\|async persistent PIC cache read failed\|Cache invalid\|decode_prepare_inside_decode=1\|ordinary_q1=1' -n /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/decode_repair_q1row_default_orangepi_ctx512_1024_pic_20260704_122350/pic_server_*.log || true"
```

Output was empty. `failures.csv` was not generated.

## Route Smoke

Profile-only run ID:

```text
decode_repair_q1row_default_profile_smoke_orangepi_minicpm_ctx512_x0_20260704_123206
```

Command:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b \
  --contexts 512 \
  --pic-repair-tokens 0 \
  --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 0 \
  --skip-normal \
  --run-id decode_repair_q1row_default_profile_smoke_orangepi_minicpm_ctx512_x0_20260704_123206
```

Relevant route lines:

```text
forward_raw_select_module ... use_decode_graph=1 pic_decode_repair_decode_graph=1
PIC OpenCL PA decode hd128 route ... ordinary_q1=0 repair=1 q=1 ... repair_qtile_route=1
OpenCLPagedAttention profile op=decode_causal_attention_hd128_transposed_k_sparse_qtile ... q_tile=1 ... q1_row=1 ... decode_prepare_inside_decode=0
```

This confirms default `x=0` is active rows=1 inside decode-repair family, not true normal LLM decode.

## Detailed Table

| model | ctx | true normal x0 | PIC x0 | gap | x1 extra | x3 extra | x5 extra | x7 extra | sample x0 | wall-decode-sample x0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama3.2 3B | 512 | 121.866 | 113.256 | -8.610 | 63.688 | 75.384 | 230.271 | 237.997 | 15.584 | 15.339 |
| Llama3.2 3B | 1024 | 126.138 | 141.018 | +14.880 | 57.241 | 72.232 | 245.423 | 256.907 | 15.968 | 22.924 |
| MiniCPM5-1B | 512 | 46.760 | 42.697 | -4.063 | 33.078 | 40.779 | 91.923 | 96.105 | 12.485 | 7.344 |
| MiniCPM5-1B | 1024 | 49.809 | 42.995 | -6.814 | 53.559 | 65.558 | 131.528 | 132.054 | 12.465 | 9.251 |
| Qwen3-4B | 512 | 149.916 | 152.885 | +2.969 | 81.542 | 93.156 | 298.131 | 304.218 | 12.429 | 20.333 |
| Qwen3-4B | 1024 | 155.065 | 171.296 | +16.231 | 81.640 | 96.535 | 326.898 | 342.175 | 12.377 | 25.667 |

## Interpretation

`x=0` is no longer obviously slower than true normal on most cases. The remaining positive gaps are context-sensitive:

- Qwen ctx1024: `+16.231 ms`
- Llama ctx1024: `+14.880 ms`
- Qwen ctx512: `+2.969 ms`

Existing decode-only graph profile before this default promotion showed Qwen ctx1024 PIC-vs-normal deltas under profile:

```text
total +341.501 ms over profiled decode call set
PagedAttention +153.361
Raster +70.663
Convolution +38.272
Binary/LayerNorm/While/Unary combined +79.034
ArgMax/Cast negligible
```

After promoting q1-row as default, Qwen ctx1024 `x0` decode-only graph profile was rerun without q1-row env:

```text
pic_x0_decodeonly_graphprofile_default_orangepi_qwen_ctx1024_20260704_123418
```

P0 grep was clean. Profile summary:

```text
total 1059.462 ms calls=4360
Convolution 485.455
PagedAttention 211.487
Raster 187.984
BinaryOp 53.855
LayerNorm 47.136
While 40.429
UnaryOp 32.296
ArgMax 0.565
Cast 0.255
lm_head 37.830 over 3 calls
```

Compared with the earlier true normal decode-only profile:

```text
normal total 792.129
Convolution 465.608
Raster 144.864
Attention 46.759
BinaryOp 40.681
LayerNorm 37.167
While 32.302
UnaryOp 23.970
ArgMax 0.558
Cast 0.220
```

Approximate profile delta after q1-row default:

```text
total +267.333
PagedAttention/Attention +164.728
Raster +43.120
Convolution +19.847
BinaryOp +13.174
LayerNorm +9.969
While +8.127
UnaryOp +8.326
ArgMax/Cast ~0
```

So the remaining hidden overhead is not one kernel:

- PagedAttention still pays transposed-K sparse repair machinery even for active rows=1.
- Graph-side small ops still behave like the PIC decode graph, with extra Raster/While/elementwise overhead.
- Sampler reads one token through `_ArgMax`, but sample TPOT is still 12-16 ms, so fused lm_head top1/topk is the next sampler-level target.

## Next Work Items

1. PA detail on Qwen ctx1024 `x0` and `x5` without mixing into formal data, checking append, attention, rank, prepare and decode-key readiness.
2. Implement same-family active-row graph trimming:
   - preserve logical active rows in metadata after decode-repair attention;
   - ensure post-attention residual/add/norm/MLP/Raster kernels operate on active rows, not stale full rows;
   - keep static tensor shape if required, but make effective rows explicit.
3. Test `MNN_PAGED_ATTENTION_DECODE_REPAIR_FUSED_APPEND=1` as an explicit variant across all x, not as hidden auto gate.
4. Evaluate fixed dispatch / record queue for decode-repair qtile kernels and adjacent append/rank kernels.
5. Design fused lm_head top1/topk path to eliminate logits materialization and sampler sync beyond the current `_ArgMax` device-side reduction.
