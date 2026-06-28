# Rhino 2026-06-23 00 Long Context

以下内容从旧的 `OPTIMIZATION_LOG.md` 迁移，按本设备/日期归档。

## 2026-06-23 Rhino Pi-X1 / Adreno q-split Full-Prefill Fix

Goal:

- Verify whether Rhino Pi-X1 normal `llm_bench` and PIC server prefill are using chunk forward.
- Fix the Adreno OpenCL full-prefill path without adding a temporary branch.
- Re-run formal `max_tokens=0` latency for 1024/2048 prompt lengths at cacheblend/epic budgets `0.05/0.40/0.50`, plus PIC full-compute/full-reuse and normal `llm_bench` baseline.

Chunk-forward conclusion:

- `Llm::set_config` only enables chunk forward when `config["chunk"]` or `config["chunk_limits"]` is present.
- Rhino normal and PIC OpenCL configs both have `chunk=None` and `chunk_limits=None`.
- Therefore the current Rhino normal baseline and PIC graph-boundary prefill are not chunk forward; they run a single prefill forward for these prompt lengths.

Code change:

- `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`
- In the Adreno GEMM full-prefill path, `q_split > 1` wrote QK for each piece at an offset, while softmax/QKV consumed from offset 0. The production fix reuses the same per-piece QK/softmax scratch instead of carrying a stale piece offset.
- The QK/softmax scratch allocation was also corrected from `qChunkPack * kvLen` to `qChunkPack * kvPack`, matching the 32-aligned K dimension used by the GEMM path.

Build/deploy:

```text
MNN_TARGET_DEVICE=aidlux_adreno_opencl BUILD_TARGET=MNN_CL BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

MNN_TARGET_DEVICE=aidlux_adreno_opencl BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
  bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/aidlux_adreno_opencl/ \
  aidlux@192.168.101.227:/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl/
```

Remote sanity:

```text
device: Rhino Pi-X1 / Adreno740v2
kgsl: /dev/kgsl-3d0 present
normal_config: /mnt/nvme/mnn_pic_opencl/models/normal/AI-ModelScope__Llama-3.2-1B-Instruct/config_opencl_greedy.json
pic_config:    /mnt/nvme/mnn_pic_opencl/models/pic/AI-ModelScope__Llama-3___2-1B-Instruct-pic-boundary/config_opencl_greedy.json
configs: chunk=None, chunk_limits=None
artifact: /mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl
```

Formal run:

```text
run_id=rhino_adreno_qsplit_fix_clean_ctx1024_2048_cbepic_005_040_050_20260623
local_summary=.cache/latency_budget_20260622/rhino_adreno_qsplit_fix_clean_ctx1024_2048_cbepic_005_040_050_20260623/summary.csv
remote_cache=/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/rhino_adreno_qsplit_fix_clean_ctx1024_2048_cbepic_005_040_050_20260623
warm: each context/mode/ratio before formal measure
/v1/tune/update_cache: status=200 for ctx1024 and ctx2048
log_scan: target_unavailable=false, async_failed=false, error_lines=[]
```

Results:

```text
context,mode,budget,latency_s,recompute,reuse,selected,speedup_vs_normal,speedup_vs_pic_full
1024,normal-full-compute,full,3.014913,,,,1.000,
1024,PIC full-compute,full,3.209506,1010,0,,0.939,1.000
1024,full-reuse,full,0.364463,0,1010,,8.272,8.806
1024,cacheblend,0.05,0.657118,51,959,51,4.588,4.884
1024,cacheblend,0.40,1.767268,404,606,404,1.706,1.816
1024,cacheblend,0.50,2.218288,505,505,505,1.359,1.447
1024,epic,0.05,0.612807,51,959,,4.920,5.237
1024,epic,0.40,1.651901,404,606,,1.825,1.943
1024,epic,0.50,2.007737,505,505,,1.502,1.599
2048,normal-full-compute,full,6.323234,,,,1.000,
2048,PIC full-compute,full,6.696720,2034,0,,0.944,1.000
2048,full-reuse,full,0.442758,0,2034,,14.281,15.125
2048,cacheblend,0.05,1.230309,102,1932,102,5.140,5.443
2048,cacheblend,0.40,4.228442,814,1220,814,1.495,1.584
2048,cacheblend,0.50,5.207544,1017,1017,1017,1.214,1.286
2048,epic,0.05,1.058596,102,1932,,5.973,6.326
2048,epic,0.40,3.279326,814,1220,,1.928,2.042
2048,epic,0.50,3.902066,1017,1017,,1.620,1.716
```

Conclusion:

- The Rhino performance issue was not caused by LLM chunk forward; both configs are non-chunked.
- PIC full-compute is now close to normal full-compute on Adreno: about 6% overhead at 1024/2048.
- full-reuse remains dominated by hydrate/suffix work and is 8.27x faster than normal at 1024, 14.28x at 2048.
- cacheblend and epic remain faster than normal even at 50% recompute on Rhino/Adreno. Epic is consistently faster than cacheblend at high budgets because its selected rows are prefix-contiguous rather than cacheblend's scattered top-k rows.
- No production env fallback or temporary branch is required; keep the Adreno GEMM q-split scratch fix as the default path.
