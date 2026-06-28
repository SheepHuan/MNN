# Jetson CUDA decode repair continuation

## 04:00 CST Status

Starting point:

```text
x=0 none                  TPOT=87.163 ms  status=ok
x=1 lagged_attention_hkvd TPOT=158.793 ms status=ok
x=3 lagged_attention_hkvd TPOT=185.317 ms status=ok
x=5 lagged_attention_hkvd TPOT=184.994 ms status=ok
x=7 lagged_attention_hkvd TPOT=207.958 ms status=ok
```

Known from the previous hour:

- Rank capture is already compact and restricted to the configured attention layer.
- `MNN_PIC_DECODE_REPAIR_PROFILE=1` showed post-step0 non-`forwardRaw` overhead around `0.11-0.12 ms/step`.
- Direct `PicDecodeMlp` hot repeat on Llama3.2-3B shapes:

```text
rows=2 chain ~= 0.965 ms
rows=4 chain ~= 1.744 ms
rows=6 chain ~= 1.723 ms
rows=8 chain ~= 1.746 ms
```

Source inspection:

- `ArGeneration::generate()` samples from the previous logits, calls `updateContext(0, 1)`, then calls `forwardVecWithPicDecodeRepair({current_token})`.
- `forwardVecWithPicDecodeRepair()` builds `repair_tokens + 1` sparse rows and calls `forwardRaw(inputEmbeds, repairMask, repairPos)`.
- `forwardRaw()` sees `gen_seq_len > 0`, sets `logitsIndex = logitsLastIdx`, and only falls back to all logits when `mMeta->add != seqLen`.
- For decode repair, `mMeta->add` is set to `logicalIndices.size()`, matching `seqLen`, so `logits_index=-1` is preserved.
- Export code slices `hidden_states[:, logits_index_long:, :]` before `self.lm(hidden_states)`, so `lm_head` should receive one row.

Next profile:

- Run Jetson `pic_server` with `MNN_PIC_GRAPH_PROFILE=1` and `MNN_PIC_GRAPH_PROFILE_TOP=1000`.
- Profile a small decode repair matrix with `x=0/1/3/5/7` and inspect:
  - whether `lm_head` / final Linear input is one row,
  - whether MLP gate/up/down rows match `x+1`,
  - top op categories and per-op shape cliffs.

Constraints:

- Do not merge graph-profile numbers into formal `benchmark_decode.csv`.
- Keep `x=0` in every TPOT/profile smoke.
- Do not change algorithm semantics or store full attention weights.

## Graph Profile Result

Remote run:

```text
host=jetson@192.168.101.192
server_port=18131
local_tunnel_port=19131
remote_run_dir=.cache/pic_prefill_latency_sweep/decode_repair_graph_profile_20260628_0404
local_copy=.cache/mnn-pic-benchmark/decode_repair_graph_profile_20260628_0404
```

Server env:

```text
MNN_PIC_GRAPH_PROFILE=1
MNN_PIC_GRAPH_PROFILE_TOP=1000
MNN_PIC_DECODE_REPAIR_PROFILE=1
MNN_PIC_SERVER_PAGED_KV_MAX_TOKENS=4096
```

Benchmark shape:

```text
context=1024
max_tokens=8
repair_tokens=0,1,3,5,7
decode_selectors=lagged_attention_hkvd
attention_layer_idx=1
top_m=32
mode=full-reuse
```

Profile TPOT is inflated by graph sync and is not a formal latency result:

```text
x=0 none                  TPOT=146.901 ms
x=1 lagged_attention_hkvd TPOT=209.473 ms
x=3 lagged_attention_hkvd TPOT=301.535 ms
x=5 lagged_attention_hkvd TPOT=316.171 ms
x=7 lagged_attention_hkvd TPOT=333.601 ms
```

Request mapping:

```text
request 1 = x0
request 2 = x1
request 3 = x3
request 4 = x5
request 5 = x7
```

Type totals:

```text
x,total,Convolution,PicSparseAttention,PicScoreAttention,PagedAttention,Raster,BinaryOp,LayerNorm
0,873.033,432.507,204.207,9.124,11.443,83.550,50.902,15.347
1,1034.315,483.563,241.851,11.704,35.243,128.320,51.380,14.900
3,1627.211,972.643,325.296,14.329,26.075,137.036,58.669,16.576
5,1731.917,988.579,400.991,17.731,29.716,137.475,60.818,17.544
7,1884.087,992.078,542.033,22.545,31.417,137.816,61.159,17.113
```

Decode repair `forward_raw` excluding step0:

```text
repair_rows=1 sparse=2 avg_forward_raw=132.217 ms
repair_rows=3 sparse=4 avg_forward_raw=212.614 ms
repair_rows=5 sparse=6 avg_forward_raw=228.836 ms
repair_rows=7 sparse=8 avg_forward_raw=249.760 ms
```

Conv profile counts:

```text
(1, gemv) 31
(2, gemv) 1561
(4, rt0_st1) 896
(4, rt1_st0) 668
(6, rt0_st1) 896
(6, rt1_st0) 669
(8, rt0_st1) 896
(8, rt1_st0) 658
(1024, rt0_st1) 112
(1024, rt1_st0) 83
```

Key attribution:

- `/lm/lm_head/Linear` input is one row: `inputs=[1x3072x1x1]`. Decode repair preserves `logits_index=-1`; final logits are not the source of the x scaling gap.
- CPU-side rank/selector cost is not the bottleneck. Previous `MNN_PIC_DECODE_REPAIR_PROFILE=1` showed post-step0 non-`forwardRaw` overhead around `0.11-0.12 ms/step`.
- `Convolution` dominates x>=3 and many rows4/6/8 low-memory INT4 1x1 Linear ops fall back to runtime dequant (`runtime_dequant=1 static_dequant=0`).
- Static dequant cache total is about `4039114752` bytes, near the current cap in `ConvFpAIntBExecution.cu`; direct microbench does not reproduce the same fallback mix.

Next target:

- Inspect static dequant cap and selection in `source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu`.
- Prefer a memory-safe improvement that keeps hot decode repair MLP gate/up/down linears in static FP16 dequant cache before less useful full-prefill/other linears.
- Validate with non-profile TPOT smoke for `x=0/1/3/5/7`, then graph profile only if the smoke improves.

## Static Dequant Cap A/B

Change under test:

```text
file=source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
function=picStaticDequantMlpLimitBytes()
policy=high-memory devices (>=30 GiB) may use a larger static FP16 dequant cache
```

Rationale:

- The graph-profile run showed `static_cache_total ~= 4039114752` bytes, near the old effective cap.
- Rows4/6/8 decode repair MLP/Linear ops still had many `runtime_dequant=1 static_dequant=0` entries.
- The direct hot MLP bench was faster because it did not reproduce the end-to-end static-cache pressure and fallback mix.

Non-profile CUDA A/B, same 1024 context and `max_tokens=8` decode matrix:

```text
original known baseline:
x=0 none                  TPOT=87.163 ms  ok
x=1 lagged_attention_hkvd TPOT=158.793 ms ok
x=3 lagged_attention_hkvd TPOT=185.317 ms ok
x=5 lagged_attention_hkvd TPOT=184.994 ms ok
x=7 lagged_attention_hkvd TPOT=207.958 ms ok

first cap raise, about 4.0 GiB:
x=0 none                  TPOT=87.379 ms  ok
x=1 lagged_attention_hkvd TPOT=101.761 ms ok
x=3 lagged_attention_hkvd TPOT=169.789 ms ok
x=5 lagged_attention_hkvd TPOT=182.527 ms ok
x=7 lagged_attention_hkvd TPOT=205.131 ms ok

current high-memory 4.5 GiB cap:
x=0 none                  TPOT=88.770 ms  ok
x=1 lagged_attention_hkvd TPOT=102.391 ms ok
x=3 lagged_attention_hkvd TPOT=163.131 ms ok
x=5 lagged_attention_hkvd TPOT=173.214 ms ok
x=7 lagged_attention_hkvd TPOT=195.221 ms ok
```

Current 4.5 GiB policy:

```text
defaultMaxLimit = 4096 MiB
highMemMaxLimit = 4608 MiB
highMemThreshold = 30 GiB
highMemDevice ? totalGlobalMem / 6 : totalGlobalMem / 8
```

Observed impact:

- `x=1` improved from `158.793 ms` to `102.391 ms`; the lagged-attention path is now about `+13.6 ms` over `x=0`.
- `x=3` improved from `185.317 ms` to `163.131 ms`.
- `x=5` improved from `184.994 ms` to `173.214 ms`.
- `x=7` improved from `207.958 ms` to `195.221 ms`.
- `x=0` stayed essentially stable within smoke noise (`87.163 -> 88.770 ms`).

Artifact and run notes:

```text
build=jetson cross CUDA
targets=MNN_Cuda_Main, libMNN.so, libpic_llm.so, pic_server
local_artifact=.cache/output/mnn/artifacts/jetson/
remote_artifact=/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
remote_port=18131
summary=.cache/mnn-pic-benchmark/decode_repair_static_cap45_cuda_ab_20260628_0440/summary.csv
```

CUDA config pitfall:

- `.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config.json` is CPU and produces `"backend": "mnn_cpu"`.
- CUDA decode repair tests must use `.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json` and confirm response metadata reports `"backend": "mnn_cuda"`.

Remaining work:

- Run a current 4.5 GiB graph profile to confirm whether residual `runtime_dequant=1` rows4/6/8 counts fell further.
- If runtime dequant remains, prefer cache admission/prioritization for decode-hot MLP shapes over blindly raising the global cap again.
- Re-check `PicSparseAttention` contribution for `x=3/5/7`; qtile sparse attention stays default-off for `x<=7` unless an explicit A/B proves a win.
- Do not merge these debug/profiling CSVs into `benchmark_decode.csv`; formal results still need the Jetson + OrangePi matrix.
