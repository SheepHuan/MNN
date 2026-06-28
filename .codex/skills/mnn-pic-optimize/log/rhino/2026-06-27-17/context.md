# Rhino 2026-06-27 17 Long Context

## OpenCL warmup tune: separate Mali and Adreno route caches

User concern:

- Rhino / Adreno warmup is very long.
- Mali and Adreno should not run each other's tuned paths.
- OpenCL cache path should remain fixed; do not add multiple cache roots.

Analysis:

- The benchmark-level warmup is mandatory because formal OpenCL latency must exclude kernel build, LWS tuning, program binary creation, PIC text-cache probing and first-use shape tuning.
- Existing code already measures PIC tune candidates in steady state by warming each candidate once and timing the second run. That prevents a flash candidate from losing only because its first execution paid image setup or nested tune cost.
- However, the PIC-specific tune entries are stored through `getTunedInfo` / `setTunedInfo` using artificial keys such as:
  - `paged_score_sparse_family_*`
  - `paged_sparse_flash_schedule_*`
  - `paged_sparse_flash_variant_*`
  - `paged_sparse_qsplit_chunk_*`
  - `paged_cacheblend_topk_family_*`
- Those keys previously encoded shape and sparse distribution, but not GPU family. If the same cache file or copied cache contents contained Mali-tuned choices, Rhino / Adreno could replay them. The reverse is also unsafe because Adreno image/K-image candidates and Mali row/qsplit choices have different cost models.

Code change:

- `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
  - Added `_openCLTuneDeviceKey(OpenCLRuntime*)`.
  - Namespaces PIC hand-written tune keys by GPU family:
    - `mali`
    - `adreno`
    - `adreno_legacy`
    - `radeon`
    - `intel`
    - `other`
  - Updated all five PIC tune key families listed above to include that device key.

Important boundary:

- This does not change the fixed OpenCL cache root. Rhino still uses:

```text
/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep
```

- The change only isolates entries inside the cache namespace. Old unqualified tune entries remain in the file but are no longer looked up by the new keys.

Expected impact:

- First warm after this change may still be slow because it has to retune the new Adreno-namespaced keys.
- Subsequent warm/formal runs on Rhino should not inherit OrangePi/Mali sparse family, schedule or variant choices.
- This should reduce unexplained wrong routing such as score-layer `qsplit` on Adreno headDim=128 hot shapes after the Adreno default route says flash is preferred.

Next validation:

- Rebuild and sync the Rhino OpenCL artifact.
- Run a small warm-required Rhino profile with `MNN_PAGED_ATTENTION_PROFILE=1`.
- Confirm logs show `score_sparse_family ... source=default/cache/online_tuned` with Adreno-appropriate family and `score_flash_attention` / `sparse_flash_attention` paths, not stale Mali `qsplit` replay.
- Do not use `--no-warm` for formal data. Merge only successful post-warm measured rows into `benchmark.csv`.
