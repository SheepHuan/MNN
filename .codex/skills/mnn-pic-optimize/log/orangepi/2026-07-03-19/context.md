# OrangePi Decode Path Audit Context

Post-audit update: this context records the pre-unification OrangePi path audit.
The current code no longer treats x0 as "no decode_refine / identity fused-KV";
`repair_tokens=0` now sends `decode_refine.enabled=true,
tokens_per_decode_step=0`, validates `mnn_token_id_sparse_decode`, and routes as
active rows=1 in the same decode-repair family as x=1/3/5/7.

## 2026-07-04 validated dualgraph rerun

The original run in this file is retained as a path-audit sample, but its
Llama3.2-3B and Qwen3-4B `pic_dualgraph_x0` rows were not true dualgraph rows.
OrangePi only had `llm_decode.mnn` for MiniCPM:

```text
/mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/OpenBMB__MiniCPM5-1B-pic-boundary-orangpi-dualgraph/llm_decode.mnn
```

The Llama and Qwen OrangePi PIC directories used by the first run had no
`llm_decode_model` in `config_opencl_greedy.json` and no `llm_decode.mnn`. Those
rows were therefore main-graph PIC decode, not `PIC dualgraph x0`.

Fixes applied before rerun:

- `pic_server /` now reports `pic_dualgraph_decode`, `llm_decode_model`,
  `llm_decode_weight`, and `llm_decode_shared_weight`.
- `run_pic_decode_repair_benchmark.py` defaults to requiring
  `pic_dualgraph_decode=true`.
- `run_decode_experiment.py` validates remote `config_opencl_greedy.json`,
  `llm_config.json`, `llm_decode.mnn`, and decode weight before it can label a
  row as `pic-dualgraph-x0`.
- Exported real OrangePi/generic dualgraph models:

```bash
MNNCONVERT_PATH=$PWD/.cache/build/mnn/x64_pic_opencl/MNNConvert \
MNN_LLM_EXPORTER=pic \
MNN_PIC_EXPORT_DEVICE=orangepi \
MNN_LLM_EXPORT_DST=.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-orangepi-dualgraph \
bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh \
  AI-ModelScope/Llama-3.2-3B-Instruct -- \
  --paged_kv_max_tokens 4096 --pic_export_decode_graph

MNNCONVERT_PATH=$PWD/.cache/build/mnn/x64_pic_opencl/MNNConvert \
MNN_LLM_EXPORTER=pic \
MNN_PIC_EXPORT_DEVICE=orangepi \
MNN_LLM_EXPORT_DST=.cache/weight/Qwen__Qwen3-4B-pic-boundary-orangepi-dualgraph \
bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh \
  Qwen/Qwen3-4B -- \
  --paged_kv_max_tokens 4096 --pic_export_decode_graph
```

The x64 artifact `MNNConvert` under `.cache/output/mnn/artifacts/x64/bin` was not
usable here because it depended on `libMNN_Cuda_Main.so`; the non-CUDA converter
from `.cache/build/mnn/x64_pic_opencl/MNNConvert` was used.

Synced model roots:

```text
/mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary-orangepi-dualgraph/
/mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/Qwen__Qwen3-4B-pic-boundary-orangepi-dualgraph/
```

Both contain:

```text
config_opencl_greedy.json
llm.mnn
llm.mnn.weight
llm_decode.mnn
llm_config.json
```

Validated run command:

```bash
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512,1024 \
  --normal-decode-tokens 16 \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --run-id decode_dualgraph_validated_orangepi_ctx512_1024_20260704_0001
```

Result files:

```text
.cache/mnn-pic-benchmark/decode_experiment/decode_dualgraph_validated_orangepi_ctx512_1024_20260704_0001/decode_tpot_wide.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_dualgraph_validated_orangepi_ctx512_1024_20260704_0001/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_dualgraph_validated_orangepi_ctx512_1024_20260704_0001/pic_dualgraph_orangepi_*.json
```

No `failures.csv` was generated. Log scan for `decode_prepare_inside_decode=1`,
`requires sparse flash production path`, `ERROR`, `target unavailable`, async read
failure, and decode-module load failure printed nothing.

Frequency was max for both validated and A/B runs:

```text
CPU little 1.8 GHz min=max, performance
CPU mid/big 2.304 GHz min=max, performance
GPU 1.0 GHz min=max, performance
DDR 2.112 GHz min=max, performance
```

Validated TPOT in ms/token:

```text
model,ctx,true_normal_x0,pic_dualgraph_x0,pic_repair_x1,pic_repair_x3,pic_repair_x5,pic_repair_x7,pic_x0_vs_true_normal
MiniCPM5-1B,512,46.637,46.778,122.330,120.113,155.766,155.741,0.9970
MiniCPM5-1B,1024,50.116,49.894,143.521,144.380,182.779,181.971,1.0044
Llama3.2 3B,512,121.645,124.253,260.553,257.711,404.882,404.882,0.9790
Llama3.2 3B,1024,126.032,142.155,310.542,313.329,455.551,457.611,0.8866
Qwen3-4B,512,149.789,151.603,329.573,332.351,512.853,520.691,0.9880
Qwen3-4B,1024,155.065,176.901,388.424,393.066,594.836,590.635,0.8766
```

Compared with the old mislabeled run, validated x0 improved:

```text
Llama3.2 3B ctx512: 151.020 -> 124.253 ms/token, 1.215x faster
Llama3.2 3B ctx1024:170.592 -> 142.155 ms/token, 1.200x faster
Qwen3-4B ctx512:    187.282 -> 151.603 ms/token, 1.235x faster
Qwen3-4B ctx1024:   215.077 -> 176.901 ms/token, 1.216x faster
```

Repair extra ms vs validated PIC dualgraph x0:

```text
MiniCPM5-1B ctx512:  x1 +75.552,  x3 +73.335,  x5 +108.988, x7 +108.963
MiniCPM5-1B ctx1024: x1 +93.627,  x3 +94.486,  x5 +132.884, x7 +132.076
Llama3.2 3B ctx512: x1 +136.300, x3 +133.458, x5 +280.629, x7 +280.629
Llama3.2 3B ctx1024:x1 +168.388, x3 +171.175, x5 +313.396, x7 +315.456
Qwen3-4B ctx512:    x1 +177.970, x3 +180.748, x5 +361.250, x7 +369.087
Qwen3-4B ctx1024:   x1 +211.524, x3 +216.165, x5 +417.935, x7 +413.734
```

Transposed-K x0 A/B on the same validated dualgraph model roots:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512,1024 \
  --skip-normal \
  --pic-repair-tokens 0 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --run-id decode_dualgraph_x0_transposed_k_orangepi_ctx512_1024_20260704_0001
```

```text
model,ctx,true_normal,default_x0,transposed_x0,default/normal,transposed/normal,transposed/default
MiniCPM5-1B,512,46.637,46.778,47.870,1.003,1.026,1.023
MiniCPM5-1B,1024,50.116,49.894,51.223,0.996,1.022,1.027
Llama3.2 3B,512,121.645,124.253,123.721,1.021,1.017,0.996
Llama3.2 3B,1024,126.032,142.155,129.910,1.128,1.031,0.914
Qwen3-4B,512,149.789,151.603,152.310,1.012,1.017,1.005
Qwen3-4B,1024,155.065,176.901,171.454,1.141,1.106,0.969
```

Interpretation:

- The biggest root cause of Llama/Qwen x0 overhead was not q1 kernel choice; it
  was that the benchmark pointed at non-dualgraph OrangePi PIC model roots.
- After real OrangePi dualgraph exports, ctx512 x0 is close to true normal for
  all three models. ctx1024 still has Llama/Qwen overhead.
- `MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1` is a mixed result: it helps Llama
  ctx1024 and Qwen ctx1024, but regresses MiniCPM and Qwen ctx512. Do not promote
  it as a global default. It remains an explicit variant until there is a fixed
  per-device, parameter-based heuristic with no request-time env gate.
- Decode repair overhead is still large and has plateaus: x1/x3 are close, x5/x7
  are close. Next profile should focus on `mnn_token_id_sparse_decode`, especially
  sparse decode attention and compact-row dense/MLP thresholds.

Default path profile smoke:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b \
  --contexts 512 \
  --skip-normal \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 0 \
  --run-id decode_path_profile_default_orangepi_minicpm_ctx512_20260704_0001
```

This profile is not a latency number because `MNN_PAGED_ATTENTION_PROFILE_DETAIL=1`
adds `queue.finish()` calls. It only validates routing. Remote log scan for
`decode_prepare_inside_decode=1`, sparse flash unsupported, `ERROR`, target
unavailable, and async read failure printed nothing.

Profile op counts from the default path:

```text
sparse_flash_attention 404
hydrate 120
sparse_qsplit_attention 28
prefill_attention_fast_qk_softmax_qkv 24
decode_causal_attention_hd128_identity_fused_kv 24
decode_prepare_request 5
decode_attention_rank 4
```

Query breakdown:

```text
decode_causal_attention_hd128_identity_fused_kv q=1: 24
sparse_flash_attention q=1: 220
sparse_flash_attention q=2/4/6/8: 46 each
sparse_qsplit_attention q=1: 20
sparse_qsplit_attention q=2/4/6/8: 2 each
decode_attention_rank q=2/4/6/8: 1 each
```

Interpretation:

- `PIC dualgraph x0` reaches the q=1 `decode_causal_attention_hd128_identity_fused_kv`
  path, with `fused=1`, `kv_write=1`, `decode_prepare_inside_decode=0`.
- `PIC decode-repair x>0` in the default run does not use the transposed-K qtile
  decode kernel. It uses sparse flash / sparse qsplit attention plus
  `decode_attention_rank`.
- The transposed-K qtile decode-repair family is therefore still hidden behind
  `MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1`, not a fixed OrangePi default.

Transposed-K repair A/B:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512,1024 \
  --skip-normal \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --run-id decode_dualgraph_transposed_k_repair_orangepi_ctx512_1024_20260704_0001
```

No `failures.csv` was generated. Remote log scan for `decode_prepare_inside_decode=1`,
`ERROR`, target unavailable, async read failure, decode module load failure, and
sparse unsupported printed nothing.

Transposed-K TPOT in ms/token:

```text
model,ctx,x0,x1,x3,x5,x7
MiniCPM5-1B,512,47.359,117.320,119.062,170.279,174.355
MiniCPM5-1B,1024,51.318,129.703,142.006,208.693,209.536
Llama3.2 3B,512,123.885,224.472,232.294,387.725,395.511
Llama3.2 3B,1024,142.833,242.795,255.743,432.443,438.032
Qwen3-4B,512,152.217,286.993,293.337,495.611,505.345
Qwen3-4B,1024,172.493,311.722,334.205,554.853,569.176
```

Transposed/default ratio:

```text
model,ctx,x0,x1,x3,x5,x7
MiniCPM5-1B,512,1.012,0.959,0.991,1.093,1.120
MiniCPM5-1B,1024,1.029,0.904,0.984,1.142,1.151
Llama3.2 3B,512,0.997,0.862,0.901,0.958,0.977
Llama3.2 3B,1024,1.005,0.782,0.816,0.949,0.957
Qwen3-4B,512,1.004,0.871,0.883,0.966,0.971
Qwen3-4B,1024,0.975,0.803,0.850,0.933,0.964
```

Repair interpretation:

- qtile/transposed-K is materially faster than default sparse flash for Llama/Qwen
  repair, especially x1/x3.
- qtile/transposed-K is mixed for MiniCPM: x1 improves, x3 is roughly neutral,
  and x5/x7 regress.
- This is not acceptable as another ad hoc runtime env gate. It should be turned
  into a fixed OrangePi shape/parameter variant, likely keyed by active rows and
  shape work rather than model name. Until that is implemented, keep the default
  production path unchanged and report qtile as an explicit variant.

## Run

Command:

```bash
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512,1024 \
  --normal-decode-tokens 16 \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --run-id decode_q1_sparse_flash_orangepi_ctx512_1024_20260704_0001
```

Result files:

```text
.cache/mnn-pic-benchmark/decode_experiment/decode_q1_sparse_flash_orangepi_ctx512_1024_20260704_0001/decode_tpot_wide.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_q1_sparse_flash_orangepi_ctx512_1024_20260704_0001/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_q1_sparse_flash_orangepi_ctx512_1024_20260704_0001/frequency_orangepi.json
```

Artifact was built with the OrangePi artifact script path and synced to:

```text
/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

The run used current default PIC server env:

```text
server_env_extra=""
```

## 2026-07-04 auto-route repair default

Code change:

- The old global `MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K` policy was later
  split. Current code uses `MNN_PAGED_ATTENTION_DECODE_Q1_TRANSPOSED_K=1` for
  q1 A/B and `MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE=0/1` for repair
  qtile A/B.
- q=1 no-repair remains on `decode_causal_attention_hd128_identity_fused_kv`
  unless an explicit q1 experimental env is set.
- q>1 decode-repair on Mali uses transposed-K qtile by default only for
  `kv_heads>=8` or `attnLen<=4`. This promotes qtile for Llama/Qwen repair and
  for MiniCPM x1/x3, while keeping MiniCPM x5/x7 on the existing sparse
  flash/qsplit repair path.

Build and sync:

```bash
JOBS=$(( $(nproc) / 2 > 0 ? $(nproc) / 2 : 1 )) \
MNN_TARGET_DEVICE=orangepi5plus BUILD_TARGET=pic_server \
BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

file .cache/output/mnn/artifacts/orangepi5plus/bin/pic_server \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN.so \
     .cache/output/mnn/artifacts/orangepi5plus/lib/libMNN_CL.so

rsync -a --delete .cache/output/mnn/artifacts/orangepi5plus/ \
  orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
```

Artifact check:

```text
pic_server:   ELF 64-bit LSB executable, ARM aarch64
libMNN.so:    ELF 64-bit LSB shared object, ARM aarch64
libMNN_CL.so: ELF 64-bit LSB shared object, ARM aarch64
```

Route profile command:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512 \
  --skip-normal \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 0 \
  --run-id decode_auto_route_profile_orangepi_ctx512_20260704_0001
```

Route profile P0 scan printed nothing for:

```text
decode_prepare_inside_decode=1
ERROR
target unavailable
async persistent PIC cache read failed
```

Route profile op counts:

```text
pic_server_decode_minicpm5-1b.log
  identity_fused=24
  transposed_qtile=48
  sparse_flash=312
  sparse_qsplit=24
  decode_rank=4

pic_server_decode_llama3.2-3b.log
  identity_fused=28
  transposed_qtile=112
  sparse_flash=260
  sparse_qsplit=20
  decode_rank=4

pic_server_decode_qwen3-4b.log
  identity_fused=36
  transposed_qtile=144
  sparse_flash=340
  sparse_qsplit=20
  decode_rank=4
```

Interpretation:

- MiniCPM has 24 layers and uses qtile for x1/x3 only: `24 * 2 = 48`.
- Llama has 28 layers and uses qtile for x1/x3/x5/x7: `28 * 4 = 112`.
- Qwen has 36 layers and uses qtile for x1/x3/x5/x7: `36 * 4 = 144`.
- `PIC dualgraph x0` still reaches q=1 identity fused KV. The auto-route does
  not mix qtile into no-repair x0.

Formal command:

```bash
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models minicpm5-1b,llama3.2-3b,qwen3-4b \
  --contexts 512,1024 \
  --normal-decode-tokens 16 \
  --pic-repair-tokens 0,1,3,5,7 \
  --pic-max-tokens 16 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 1 \
  --run-id decode_dualgraph_auto_route_orangepi_ctx512_1024_20260704_0001
```

Result files:

```text
.cache/mnn-pic-benchmark/decode_experiment/decode_dualgraph_auto_route_orangepi_ctx512_1024_20260704_0001/decode_tpot_wide.csv
.cache/mnn-pic-benchmark/decode_experiment/decode_dualgraph_auto_route_orangepi_ctx512_1024_20260704_0001/decode_tpot_long.csv
```

`failures.csv` was empty. Remote P0 scan again printed nothing for
`decode_prepare_inside_decode=1`, `ERROR`, target unavailable, and async read
failure.

Auto-route TPOT in ms/token:

```text
model,ctx,true_normal_x0,pic_dualgraph_x0,pic_repair_x1,pic_repair_x3,pic_repair_x5,pic_repair_x7
MiniCPM5-1B,512,46.569,47.344,111.385,117.680,157.980,159.653
MiniCPM5-1B,1024,50.041,49.961,129.773,145.492,186.168,183.379
Llama3.2 3B,512,122.154,124.025,223.906,232.785,386.902,388.529
Llama3.2 3B,1024,125.637,141.984,243.976,260.397,425.027,439.663
Qwen3-4B,512,149.922,151.659,287.284,294.319,499.941,500.262
Qwen3-4B,1024,155.188,177.036,304.178,332.883,563.022,570.106
```

Auto-route / validated default ratio, lower is better:

```text
model,ctx,normal,pic_x0,x1,x3,x5,x7
MiniCPM5-1B,512,0.999,1.012,0.911,0.980,1.014,1.025
MiniCPM5-1B,1024,0.999,1.001,0.904,1.008,1.019,1.008
Llama3.2 3B,512,1.004,0.998,0.859,0.903,0.956,0.960
Llama3.2 3B,1024,0.997,0.999,0.786,0.831,0.933,0.961
Qwen3-4B,512,1.001,1.000,0.872,0.886,0.975,0.961
Qwen3-4B,1024,1.001,1.001,0.783,0.847,0.947,0.965
```

PIC x0 / true normal and repair extra ms over PIC x0:

```text
model,ctx,pic_x0/normal,x1_extra,x3_extra,x5_extra,x7_extra
MiniCPM5-1B,512,1.017,64.041,70.336,110.636,112.309
MiniCPM5-1B,1024,0.998,79.812,95.531,136.206,133.417
Llama3.2 3B,512,1.015,99.881,108.759,262.877,264.504
Llama3.2 3B,1024,1.130,101.991,118.413,283.043,297.679
Qwen3-4B,512,1.012,135.625,142.660,348.282,348.602
Qwen3-4B,1024,1.141,127.142,155.847,385.986,393.070
```

Conclusion:

- The repair-path gate confusion is reduced: the fast qtile repair family is no
  longer hidden behind a q1/repair mixed global gate for Llama/Qwen and small-q
  MiniCPM. It is a fixed OrangePi default variant, with explicit repair A/B via
  `MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE=0/1`.
- q=1 no-repair remains fixed to identity fused KV. This avoids the q1
  transposed-K mixed result that regressed MiniCPM and Qwen ctx512.
- Llama/Qwen repair improves materially, especially x1/x3. MiniCPM x1 improves,
  x3 is neutral, and x5/x7 remain on sparse flash with only small repeat noise.
- The x0 problem is not solved by this change. Llama/Qwen ctx1024 still show
  `PIC dualgraph x0 / true normal` around 1.13-1.14. That overhead should be
  analyzed separately in the no-repair decode path rather than attributed to
  repair qtile routing.

## 2026-07-04 x0 ctx1024 profile

Run id:

```text
decode_x0_profile_pic_orangepi_ctx1024_20260704_0002
```

Command:

```bash
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=120 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PAGED_ATTENTION_PROFILE_DETAIL=1 MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi \
  --models llama3.2-3b,qwen3-4b \
  --contexts 1024 \
  --skip-normal \
  --pic-repair-tokens 0 \
  --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 \
  --warm-repeats 0 \
  --run-id decode_x0_profile_pic_orangepi_ctx1024_20260704_0002
```

This run is attribution-only. `MNN_PAGED_ATTENTION_PROFILE_DETAIL=1` and
`MNN_PIC_GRAPH_PROFILE=1` add synchronization/debug callback overhead, so these
TPOT numbers are not formal latency:

```text
Llama3.2 3B ctx1024 PIC x0 profile TPOT: 421.292 ms/token
Qwen3-4B ctx1024 PIC x0 profile TPOT:    553.675 ms/token
```

P0 scan was clean:

```text
decode_prepare_inside_decode=1: none
ERROR / target unavailable / async persistent read failure: none
```

Path evidence:

```text
Llama3.2 3B:
  PIC decode graph route count: 1
  mnn_token_id_sparse_decode count: 0
  transposed/sparse qtile decode op count: 0
  q=1 identity_fused_kv decode ops: 28

Qwen3-4B:
  PIC decode graph route count: 1
  mnn_token_id_sparse_decode count: 0
  transposed/sparse qtile decode op count: 0
  q=1 identity_fused_kv decode ops: 36
```

Measured decode forward selection:

```text
Llama3.2 3B:
  suffix prefill: seq_len=1, seq_len_key=100, use_decode_graph=0, inputs=5, forward_raw_total=740.971 ms
  measured decode: seq_len=1, seq_len_key=1, use_decode_graph=1, inputs=4, forward_raw_total=421.243 ms

Qwen3-4B:
  suffix prefill: seq_len=1, seq_len_key=100, use_decode_graph=0, inputs=5, forward_raw_total=1068.470 ms
  measured decode: seq_len=1, seq_len_key=1, use_decode_graph=1, inputs=4, forward_raw_total=553.555 ms
```

This confirms PIC `decode_tpot_ms` is not full-reuse hydrate/suffix time. It is
the timed q=1 decode forward after the first sampled token, matching
`pic_server.cpp::performanceSummary` and `speculative_decoding/generate.cpp`.

PagedAttention detail inside measured decode:

```text
Llama3.2 3B: identity_fused_kv count=28, total=17.557 ms, avg=627.0 us/layer
Qwen3-4B:    identity_fused_kv count=36, total=19.124 ms, avg=531.2 us/layer
```

Graph-level profile for the full chat request still shows `PicScoreAttention`
and `PicSparseAttention` op types:

```text
Llama3.2 3B graph profile:
  Convolution        265.014 ms
  PicSparseAttention 149.049 ms
  Raster              71.026 ms
  PagedAttention      50.475 ms
  PicScoreAttention   11.487 ms

Qwen3-4B graph profile:
  Convolution        351.684 ms
  PicSparseAttention 208.650 ms
  Raster             115.984 ms
  PagedAttention      81.411 ms
  PicScoreAttention   18.494 ms
```

Interpretation:

- The x0 path is not currently gate-confused at runtime: no `decode_refine`, no
  `mnn_token_id_sparse_decode`, no qtile/transposed repair kernel, and no
  historical K prepare inside measured decode.
- The graph-level `PicScoreAttention` / `PicSparseAttention` names are an export
  artifact of graph-boundary PIC models and should not be used alone as evidence
  that decode-repair is active.
- The remaining ctx1024 x0 slowdown versus true normal is a real q=1 PIC decode
  overhead. It should be investigated as PIC q=1 PagedAttention plus decode graph
  dense/Raster/elementwise overhead versus true normal LLM, not as repair-route
  qtile/sparse gate confusion.

## Frequency

`frequency_orangepi.json` records:

```text
CPU little: performance, 1.8 GHz min=max
CPU mid:    performance, 2.304 GHz min=max
CPU big:    performance, 2.304 GHz min=max
GPU:        performance, 1.0 GHz min=max
DDR:        performance, 2.112 GHz min=max
```

## Path issue found

The first attempt to measure pure suffix-free PIC decode failed at the server protocol layer:

```text
PIC prompt suffix tokenization produced no tokens
```

So the formal measurement used `--pic-suffix-from-cache-tokens 1`. With a one-token suffix, full-reuse must perform a q=1 suffix prefill before timed decode. That q=1 prefill enters later-layer `PicSparseAttention` (`mPicAttentionMode == 2`, compact query rows). The old sparse support check rejected `attnLen <= 1`, producing:

```text
OpenCL PicSparseAttention layer 2 requires sparse flash production path,
but sparse flash is unsupported for query=1 kv_len=513 head_dim=128
```

The local source change only expands the existing sparse flash family for this later-layer q=1 PIC sparse shape:

```cpp
const bool allowSingleRowLaterPicSparse = !queryRowsAreFull && mPicAttentionMode == 2 && attnLen == 1;
if (attnLen <= 0 || (!allowSingleRowLaterPicSparse && attnLen <= 1) || kvLen <= 0 || mQuerySeqLen <= 0) {
    return false;
}
```

This is not a new request-time env gate.

## Decode gate cleanup state

The old global `MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K` gate has been removed
from `PagedAttentionBufExecution.cpp`. Current decode controls are split by
path:

```text
PIC dualgraph x0 q=1 default:
  decode_causal_attention_hd128_identity_fused_kv

q=1 transposed-K explicit A/B/profile:
  MNN_PAGED_ATTENTION_DECODE_Q1_TRANSPOSED_K=1
  MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_ATTENTION_ONLY=1
  MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_READONLY_ONLY=1
  MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K_Q1_ATTENTION_ONLY_BENCH=1

PIC decode-repair q>1 qtile A/B:
  unset MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE -> fixed Mali default
  MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE=0 -> force old sparse repair baseline
  MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE=1 -> force sparse qtile when shape-supported

parameter/profile-only variants:
  MNN_PAGED_ATTENTION_DECODE_HD128_FORCE_LANE
  MNN_PAGED_ATTENTION_DECODE_QTILE_FORCE
  MNN_PAGED_ATTENTION_DECODE_GQA_FUSED
```

Route debug now prints separate `q1_policy` and `repair_policy` plus
`q1_transposed_enabled` / `q1_transposed_route` / `repair_qtile_route`, so x0
and repair no longer share an ambiguous gate.

## Validity checks

Remote log scan:

```bash
ssh orangepi@192.168.101.113 \
  'grep -R "decode_prepare_inside_decode=1\|requires sparse flash production path\|ERROR\|target unavailable\|async persistent PIC cache read failed" -n /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/decode_q1_sparse_flash_orangepi_ctx512_1024_20260704_0001/pic_server_*.log || true'
```

The scan printed nothing.

Local checks:

```bash
python3 -m py_compile .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py
python3 .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/mnn-pic-benchmark
git diff --check
```

All passed.

## Data

TPOT in ms/token:

```text
model,ctx,true_normal_x0,pic_dualgraph_x0,pic_repair_x1,pic_repair_x3,pic_repair_x5,pic_repair_x7,pic_x0_vs_true_normal
MiniCPM5-1B,512,46.654,46.666,125.753,121.386,167.619,166.929,0.9998
MiniCPM5-1B,1024,50.187,50.032,144.587,149.922,187.590,187.852,1.0031
Llama3.2 3B,512,121.953,151.020,259.363,263.382,403.922,404.636,0.8075
Llama3.2 3B,1024,125.870,170.592,308.516,312.215,483.845,458.547,0.7378
Qwen3-4B,512,150.272,187.282,338.616,329.706,518.315,521.920,0.8024
Qwen3-4B,1024,154.974,215.077,392.983,396.910,584.674,586.243,0.7206
```

Repair extra ms vs PIC dualgraph x0:

```text
MiniCPM5-1B ctx512:  x1 +79.087,  x3 +74.720,  x5 +120.953, x7 +120.263
MiniCPM5-1B ctx1024: x1 +94.554,  x3 +99.889,  x5 +137.558, x7 +137.820
Llama3.2 3B ctx512: x1 +108.342, x3 +112.362, x5 +252.901, x7 +253.615
Llama3.2 3B ctx1024:x1 +137.924, x3 +141.622, x5 +313.253, x7 +287.955
Qwen3-4B ctx512:    x1 +151.334, x3 +142.425, x5 +331.033, x7 +334.638
Qwen3-4B ctx1024:   x1 +177.906, x3 +181.833, x5 +369.597, x7 +371.166
```

Interpretation:

- `true_normal_x0` is ordinary MNN LLM q=1 autoregressive decode.
- `pic_dualgraph_x0` is PIC full-reuse dualgraph decode without `decode_refine`.
- `pic_repair_x*` is PIC dualgraph decode-repair, with `decode_runtime=mnn_token_id_sparse_decode` and `active_tokens_per_decode_step=x+1`.
- MiniCPM5 x0 is aligned with true normal. Llama3.2/Qwen x0 remain slower than true normal, so PIC dualgraph x0 itself still has decode overhead.
- x1/x3 and x5/x7 form two cost plateaus. This points to route or dense/attention shape thresholds in the sparse decode path rather than a smooth per-extra-token cost.

## 2026-07-04 q1 transposed-K gate cleanup

Problem:

- A q=1 auto selector was briefly tested for Mali ctx>=1024 GQA shapes.
- The selector made the profile smoke route to `decode_causal_attention_hd128_transposed_k_fused_kv`, but the formal TPOT did not produce a reliable x0 recovery.
- This mixed two concerns in one gate: whether historical K needs a transposed decodeKey prepared, and whether q=1 production attention should route to transposed-K.

Profile smoke run:

```text
run_id=decode_x0_auto_q1_transposed_profile_orangepi_ctx1024_20260704_0001
Llama/Qwen ctx1024:
  PIC decode graph route = 1
  mnn_token_id_sparse_decode = 0
  decode_prepare_inside_decode = 0
  transposed_k_fused_kv profile ops = 28 / 36
  identity_fused_kv profile ops = 0
```

Formal TPOT run:

```text
run_id=decode_auto_q1_transposed_orangepi_ctx512_1024_20260704_0001
model,ctx,true_normal_x0,pic_x0,pic_x0_over_normal,x1,x3,x5,x7
Llama3.2 3B,512,121.746,123.499,1.014x,225.228,231.494,386.594,392.682
Llama3.2 3B,1024,125.555,142.926,1.138x,242.744,256.373,426.779,442.899
MiniCPM5-1B,512,46.765,47.822,1.023x,119.168,118.702,154.436,161.763
MiniCPM5-1B,1024,50.089,49.880,0.996x,134.329,140.040,184.633,181.211
Qwen3-4B,512,149.770,151.570,1.012x,282.795,292.859,501.355,505.648
Qwen3-4B,1024,154.818,175.509,1.134x,309.628,332.841,563.011,572.372
```

Compared with `decode_dualgraph_auto_route_orangepi_ctx512_1024_20260704_0001`,
the q1 auto selector did not materially improve x0:

```text
Llama ctx1024 PIC x0: 141.984 -> 142.926 ms (+0.66%)
Qwen ctx1024 PIC x0: 177.036 -> 175.509 ms (-0.86%)
MiniCPM ctx512 PIC x0: 47.344 -> 47.822 ms (+1.01%)
```

Decision implemented in `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`:

- Remove the q=1 automatic transposed-K selector based on `kvLen`, `num_heads/kv_heads` or GQA.
- Split q=1 and q>1 policy:
  - q=1 production default: fixed identity fused-KV family.
  - q=1 transposed-K: explicit A/B/profile variant only (`MNN_PAGED_ATTENTION_DECODE_Q1_TRANSPOSED_K=1` or q1 attention-only envs).
  - q>1 PIC decode-repair: transposed-K sparse qtile family remains the Mali default when supported; A/B uses `MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE=0/1`.
- Add debug-only route logging:

```text
PIC OpenCL PA decode hd128 route layer=... ordinary_q1=... repair=... q=...
  q1_transposed_enabled=... q1_transposed_route=...
  repair_qtile_enabled=... repair_qtile_candidate=...
  repair_qtile_route=... q1_attention_only_bench=...
  record_queue=... q1_policy=... repair_policy=...
```

This resolves the gate problem before further performance work: future x0 analysis
should compare true normal q=1 against PIC q=1 identity fused-KV plus decode graph
dense/Raster/elementwise overhead, not a shape/head-dependent q1 transposed route.

## 2026-07-04 gate split same-server validation

After splitting the decode gates, the first validation separated path routing from
same-server request lifetime:

```text
run_id=decode_gate_split_x0_only_orangepi_ctx1024_20260704_0001
  Llama3.2 3B ctx1024 x0: 264.682 ms/token, ok

run_id=decode_gate_split_x1_only_orangepi_ctx1024_20260704_0001
  Llama3.2 3B ctx1024 x1: 489.861 ms/token, ok

run_id=decode_gate_split_combined_orangepi_ctx1024_20260704_0001
  x0 followed by x1 in one server process: failed
```

The failed combined run completed the x0 request, then crashed during the second
request's `full-reuse` suffix prefill:

```text
MNN_PIC_REQUEST_PROFILE phase=begin stage=prefill_full_reuse_external_pagedkv ...
PIC OpenCL PA exec layer=0 ... decode=0 q=1 ... source_segments=1 ...
PIC OpenCL PA cache reuse layer=1 ...
PIC OpenCL PA resize layer=1 ...
```

There was no repair route log before the server closed the connection, so the
failure was not caused by `decode_causal_attention_hd128_transposed_k_sparse_qtile`.
The path evidence was:

- x0-only passed.
- x1-only passed.
- x0 then x1 failed in the same server process.
- The failure happened in the second full-reuse prefill layer 1, before
  `mnn_token_id_sparse_decode` / repair qtile execution.

Root cause:

```cpp
static std::string _externalLayerMappedTargetKey(...)
```

previously keyed the mapped target registry by `meta pointer + layer + shape`,
but not by `meta->request_generation`. The async external-layer read request key
already included `request_generation`, so after `/reset` the async task namespace
and mapped target namespace disagreed. A later request could therefore pick up
a stale mapped PagedCache target from the previous request/decode graph.

Fix implemented in `source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp`:

```cpp
os << reinterpret_cast<uintptr_t>(meta) << ":" << (meta != nullptr ? meta->request_generation : 0) << ":"
   << layerIndex << ":" << batch << ":" << kvHeads << ":" << headDim << ":" << bytes;
```

Post-fix validation:

```text
run_id=decode_gate_split_combined_orangepi_ctx1024_20260704_0002
  x0: 277.117 ms/token
  x1: 465.226 ms/token

run_id=decode_gate_split_combined_x013_orangepi_ctx1024_20260704_0001
  x0: 264.475 ms/token
  x1: 458.228 ms/token
  x3: 494.326 ms/token
```

Remote route scan for
`decode_gate_split_combined_x013_orangepi_ctx1024_20260704_0001`:

```text
q1_transposed_route=1:                 0
repair_qtile_route=1:                 56
ordinary_q1=1 repair=0:               28
ordinary_q1=0 repair=1 q=2:           28
ordinary_q1=0 repair=1 q=4:           28
decode_prepare_inside_decode=1:        0
target unavailable / async read error: 0
ERROR:                                 0
```

Final gate interpretation:

- q1 x0 production path is fixed identity fused-KV by default.
- q1 transposed-K exists only as explicit A/B or profile path.
- q>1 repair sparse qtile is a separate repair-family default on Mali-supported
  shapes, with `MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE=0/1` for A/B.
- The same-server crash was a request-generation scoping bug in mapped target
  lookup, not a compute-path gate bug.

## 2026-07-04 q1 gate performance recheck

Question checked: removing the old global
`MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K` env made old scripts silently fall back
to the default q1 route. The q1 transposed-K implementation itself was not
deleted; current code uses `MNN_PAGED_ATTENTION_DECODE_Q1_TRANSPOSED_K=1`.

Important measurement caveat:

- `decode_gate_split_combined_x013_orangepi_ctx1024_20260704_0001` used
  `MNN_PIC_DECODE_DEBUG=1 MNN_PIC_REQUEST_PROFILE=1`, `pic_max_tokens=2`, and
  `warm_repeats=0`; its x0 `264.475 ms/token` is a debug-path number.
- Formal-ish q1 comparisons must use profile off, 16 generated tokens, and a
  warm request.

Llama3.2-3B ctx1024 x0 recheck:

```text
old identity formal:
  run_id=decode_dualgraph_validated_orangepi_ctx512_1024_20260704_0001
  x0=142.155 ms/token

old global transposed-K:
  run_id=decode_dualgraph_x0_transposed_k_orangepi_ctx512_1024_20260704_0001
  server_env=MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1
  x0=129.910 ms/token

current default identity, no warm:
  run_id=decode_gate_split_identity_nowarm_llama_ctx1024_20260704_0001
  x0=144.939 ms/token

current default identity, warm + profile off:
  run_id=decode_gate_split_identity_formal_llama_ctx1024_20260704_0002
  x0=129.326 ms/token
  warm request x0=145.267 ms/token

current default identity, warm + route debug only:
  run_id=decode_gate_split_identity_debugroute_llama_ctx1024_20260704_0001
  x0=129.497 ms/token
  q1_transposed_route=0
  ordinary_q1=840

current explicit q1 transposed-K:
  run_id=decode_gate_split_q1_transposed_formal_llama_ctx1024_20260704_0001
  server_env=MNN_PAGED_ATTENTION_DECODE_Q1_TRANSPOSED_K=1
  x0=130.138 ms/token
```

Interpretation:

- The old global env name being removed is a real usability hazard: any command
  still setting `MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1` no longer enables q1
  transposed-K. Update scripts to `MNN_PAGED_ATTENTION_DECODE_Q1_TRANSPOSED_K=1`
  or add a deprecated compatibility alias.
- The large apparent regression to `264 ms/token` was measurement mode, not the
  production route. It used heavy request/debug logging, only two generated
  tokens, and no warm.
- For current warmed Llama3.2-3B ctx1024, the default identity route and explicit
  q1 transposed route are both about `130 ms/token`. This means the x0 recovery
  seen in past logs is not solely the transposed-K route; warm/runtime-cache state
  and first-request overhead are also material.
- Old 3-model A/B still shows q1 transposed-K is not uniformly faster:

```text
MiniCPM5-1B ctx512:  identity 46.778, transposed 47.870 (+2.34%)
MiniCPM5-1B ctx1024: identity 49.894, transposed 51.223 (+2.66%)
Llama3.2 3B ctx512: identity 124.253, transposed 123.721 (-0.43%)
Llama3.2 3B ctx1024:identity 142.155, transposed 129.910 (-8.61%)
Qwen3-4B ctx512:    identity 151.603, transposed 152.310 (+0.47%)
Qwen3-4B ctx1024:   identity 176.901, transposed 171.454 (-3.08%)
```

Next gate action should be conservative:

1. Do not restore a single ambiguous global gate controlling both q1 and repair.
2. Add/update benchmark scripts so old `MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K`
   is not used silently.
3. If backward compatibility is needed, map the old env only as a deprecated q1
   alias, not as a repair gate.
4. Decide the Mali q1 production default only after a warm, profile-off matrix
   across MiniCPM/Llama/Qwen and ctx512/1024, because cold/no-warm results and
   debug-profile results tell a different story.
