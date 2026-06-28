# OrangePi Qwen3-8B 2048/2560 High-Budget Hang

## Failing Case

Target missing row:

```text
orangepi,Orange Pi 5 Plus,Qwen3-8B,OpenCL,"cpu=max,gpu=max,ddr=max",2048,cacheblend,0.50
```

Repro command before the memory fix:

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices --frequency-profile max \
  --contexts 2048 --model-key qwen3-8b \
  --modes cacheblend --ratios 0.50 \
  --benchmark-csv benchmark.csv --only-missing-contexts \
  --restart-server-each-spec \
  --run-id orangepi_qwen3_8b_ctx2048_cb050_fill_20260628_0638
```

Result:

```text
phase=warm
error=curl: (52) Empty reply from server
HTTP_STATUS:000
```

`prefill_text` succeeded and was a cache hit:

```text
cache_status=reused
cache_hit=true
doc_token_count=2034
```

After the failure, OrangePi became unreachable via SSH banner exchange. `last -x` after recovery showed a new boot around `2026-06-28 14:46 CST`, while the run started around `14:38 CST`. Persistent journal for the previous boot was unavailable, so there was no OOM killer line to preserve.

## Row32 Narrowing Attempt

Initial suspicion was Mali sparse flash variant tuning: `headDim=128` could try mqtile/row64 candidates during warm. I changed Mali `headDim=128` production sparse flash to only allow row32:

```text
_sparseFlashVariantSupported(): MALI && headDim == 128 => row32 only
_sparseFlashVariantCandidates(): MALI && headDim == 128 => {row32}
_defaultSparseFlashVariant(): MALI && headDim == 128 => row32
```

The protected repro used:

```bash
PIC_SWEEP_CHAT_TIMEOUT_SECONDS=1200 python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices --frequency-profile max \
  --contexts 2048 --model-key qwen3-8b \
  --modes cacheblend --ratios 0.50 \
  --benchmark-csv benchmark.csv --only-missing-contexts \
  --restart-server-each-spec \
  --remote-memory-limit-percent 80 \
  --ssh-command-timeout-sec 90 \
  --server-runtime-timeout-sec 1800 \
  --remote-bench-timeout-sec 1800 \
  --server-env MNN_PAGED_ATTENTION_TRACE_PROGRESS=1 \
  --run-id orangepi_qwen3_8b_ctx2048_cb050_mali_row32_fix_20260628_0657
```

This showed progress through later layers with row32 only:

```text
OpenCLPagedAttention trace phase=begin op=sparse_flash_attention layer=27 query=1031 input_query=1031 full_q=0 kv_len=2048 q_chunk=64 q_split=19 schedule=range_q64 schedule_source=cache static=1 direct_value=0 variant=row32 variant_source=default
...
OpenCLPagedAttention trace phase=begin op=sparse_flash_attention layer=31 query=1031 input_query=1031 full_q=0 kv_len=2048 q_chunk=64 q_split=19 schedule=range_q64 schedule_source=cache static=1 direct_value=0 variant=row32 variant_source=default
```

But during that run `free -h` showed only about `602 MiB` available memory while warm was still running:

```text
Mem: 15Gi used=14Gi free=343Mi available=602Mi
```

The run again ended with warm `curl: (52) Empty reply from server`, and SSH banner timed out afterwards. Therefore mqtile/row64 was not the only root cause.

## Root Cause

`runSparseFastPrefill()` called:

```cpp
ensureFastPrefillTemps(qStorageLen, kvLen, qChunkLen, staticWorkspace)
```

Inside `ensureFastPrefillTemps()`:

```cpp
qChunkLen = staticWorkspace ? seqLen : std::min(qChunkLen, seqLen);
mTempQK = Tensor::createDevice<float>({qChunkPack * kvPack * mNumHead * mBatch});
mTempSoftmax = Tensor::createDevice<float>({qChunkPack * kvPack * mNumHead * mBatch});
```

For Qwen3-8B `ctx2048/cacheblend50`, later sparse layers have roughly:

```text
activeLen ~= 1031
kvLen     = 2048
heads     = 32
```

Each of `mTempQK` and `mTempSoftmax` is roughly:

```text
ROUND_UP(1031,4) * ROUND_UP(2048,32) * 32 * sizeof(float)
~= 270 MiB
```

Together they are about `540 MiB` per layer. Qwen3-8B has 32 layers, so static per-layer scratch can consume more than available system memory. Sparse flash does not use `mTempQK`, `mTempSoftmax`, or `mTempMask`; it only needs packed/rearranged Q/K/V buffers.

## Implemented Fix

Added:

```cpp
ErrorCode PagedAttentionBufExecution::ensureSparseFlashTemps(int seqLen, int kvLen, bool staticWorkspace)
```

It allocates only:

```text
mTempQ
mTempK
mTempV
```

and resets:

```text
mTempMask
mTempQK
mTempSoftmax
```

`runSparseFastPrefill()` now calls:

```cpp
ensureSparseFlashTemps(qStorageLen, kvLen, staticWorkspace)
```

instead of `ensureFastPrefillTemps(...)`.

Expected effect: remove the per-layer full QK/softmax static scratch from sparse flash, reducing `ctx2048/cacheblend50` memory by roughly tens of GiB at process level and avoiding device-level memory pressure/hang.

## Runner Hardening

The sweep runner was also hardened so future failures do not hang the host-side process:

- Local SSH/curl/power/render commands are wrapped in `timeout -k`.
- SSH commands have a default command timeout.
- Remote `llm_bench` can be wrapped in remote `timeout`.
- Remote `pic_server` systemd units get `RuntimeMaxSec`, `TimeoutStopSec`, `KillMode=control-group`, and `SendSIGKILL=yes`.

This does not solve Mali kernel hangs by itself, because GPU/kernel memory pressure can make sshd unresponsive before systemd can cleanly kill user processes. The real fix is the sparse flash memory allocation change above.

## Current Status

The memory fix was built locally for OrangePi:

```bash
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=96 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
```

But OrangePi stayed SSH-unresponsive after the pre-fix repro, so the fixed artifact had not yet been synced or validated at the end of this hour.

Next validation once SSH recovers or the board is power-cycled:

```bash
rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/

PIC_SWEEP_CHAT_TIMEOUT_SECONDS=1200 python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py \
  --devices orangepi --allow-subset-devices --frequency-profile max \
  --contexts 2048 --model-key qwen3-8b \
  --modes cacheblend --ratios 0.50 \
  --benchmark-csv benchmark.csv --only-missing-contexts \
  --restart-server-each-spec \
  --remote-memory-limit-percent 80 \
  --ssh-command-timeout-sec 90 \
  --server-runtime-timeout-sec 1800 \
  --remote-bench-timeout-sec 1800 \
  --server-env MNN_PAGED_ATTENTION_TRACE_PROGRESS=1 \
  --run-id orangepi_qwen3_8b_ctx2048_cb050_sparse_flash_temp_fix_<timestamp>
```
