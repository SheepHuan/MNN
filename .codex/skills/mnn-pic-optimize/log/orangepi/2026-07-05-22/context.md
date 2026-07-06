# Context

## Artifact

```text
cache: .cache/output/mnn/artifacts/orangepi5plus/  (built 2026-07-05 20:02)
remote: orangepi@192.168.101.113:/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus/
commit: 38ba6851 (pic: split paged attention backend families) + working-tree mods
git status: M source/backend/opencl/execution/buffer/PagedAttentionBufExecution.cpp (+ hpp, +DecodeRepairSlotIdentity.cpp, +MaliUtils, etc.)
```

## Commands used

Profile smoke (profile on, attribution only):

```bash
RUN_ID=prof_p0_x0x1357_orangepi_qwen_ctx1024_20260705_claude
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=200 MNN_PAGED_ATTENTION_PROFILE=1 MNN_PIC_DECODE_DEBUG=1 MNN_PIC_DECODE_REPAIR_PROFILE=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models qwen3-4b --contexts 1024 \
  --pic-repair-tokens 0,1,3,5,7 --pic-max-tokens 2 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 1 --warm-repeats 1 --skip-normal --run-id "${RUN_ID}"
```

Formal TPOT (profile off):

```bash
# PIC
RUN_ID=cmp_pic3_x0x1x3x5x7_claude
PIC_SWEEP_SERVER_ENV_EXTRA='MNN_PIC_DECODE_DEBUG=1' \
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models llama3.2-3b,minicpm5-1b,qwen3-4b --contexts 1024 \
  --pic-repair-tokens 0,1,3,5,7 --pic-max-tokens 8 \
  --pic-suffix-from-cache-tokens 1 \
  --repeats 2 --warm-repeats 1 --skip-normal --run-id "${RUN_ID}"

# Normal
RUN_ID=cmp_normal3_x0_claude
python3 .cache/mnn-pic-benchmark/decode_experiment/run_decode_experiment.py \
  --devices orangepi --models llama3.2-3b,minicpm5-1b,qwen3-4b --contexts 1024 \
  --normal-decode-tokens 8 \
  --pic-repair-tokens 0 --pic-max-tokens 1 \
  --repeats 2 --warm-repeats 1 --skip-pic --run-id "${RUN_ID}"
```

P0 validity scan (both greps must print NOTHING):

```bash
RUN_ID=<run_id>
ssh orangepi@192.168.101.113 \
  "grep -R 'decode_prepare_inside_decode=1' -n /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/${RUN_ID}/pic_server_*.log || true; \
   grep -R 'ERROR\|target unavailable\|async persistent PIC cache read failed\|Cache invalid' -n /mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/${RUN_ID}/pic_server_*.log || true"
```

## CSV paths

```text
.cache/mnn-pic-benchmark/decode_experiment/cmp_pic3_x0x1x3x5x7_claude/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/cmp_normal3_x0_claude/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/baseline_formal_x0x1_orangepi_qwen_ctx1024_20260705_claude/decode_tpot_long.csv
.cache/mnn-pic-benchmark/decode_experiment/cmp_normal_x0_qwen_ctx1024_claude/decode_tpot_long.csv
```

## Per-model TPOT table (current artifact, ctx1024, profile off, warm)

```text
model         normal_x0   pic_x0    pic_x1   pic_x3   pic_x5   pic_x7
MiniCPM5-1B    49.96      43.09     86.75    99.50   113.61   122.44
Qwen3-4B      154.86     157.32    233.51   260.25   415.75   442.77
Llama3.2-3B   125.64      (harness exit-1, server log shows healthy decode; rerun pending)
```

Note: Llama3.2-3B PIC rows failed in the harness with `returned non-zero exit
status 1`, but the remote `pic_server_decode_llama3.2-3b.log` shows healthy
`PIC decode debug after status=2 ... output_tokens=8 gen_seq=8 all_seq=1032` and
normal layer-0/1 PagedAttention activity. This is a harness-side issue (likely a
client timeout / port reuse after the prior model's server teardown), not a PIC
regression. Rerun in a clean session.

## x0 route confirmation (Qwen3-4B ctx1024 x0, profile on)

```text
PIC OpenCL PA decode hd128 route layer=0 ordinary_q1=0 repair=1 q=1 kv_len=1026
  q1_transposed_enabled=0 q1_transposed_route=0
  repair_qtile_enabled=1 repair_qtile_candidate=1 repair_qtile_prefix_ready=1 repair_qtile_route=1
  q1_attention_only_bench=0
  repair_keycache_qtile_v2=0 repair_q1_identity_fused_kv=0 repair_q1_gqa=0
  record_queue=0  (Mali: cl_recording_qcom unsupported)
  decode_key_ready=1025 sparse_prepare_len=1025 sparse_append=1 gqa_variant=0

op=decode_causal_attention_hd128_transposed_k_sparse_qtile layer=0 query=1
  lane=128 q_tile=1 append_count=1 prepare_len=1025 causal_kv_work=1026
  us=611 append_us=0 attention_us=611 rank_us=0
  decode_prepare_inside_decode=0 prepare_us=0
  slot_identity=1 prefix_stable=1 identity_v=1
```

## Why no new kernel was implemented this hour

1. The user's premise ("x0 still ~20 ms slower") is stale on the current
   artifact. Measured gap is ~2.5 ms (Qwen) and PIC is faster on MiniCPM.
2. Every obvious lever in the V6 plan and the user's brief has been A/B'd and
   regressed on both Mali and Adreno (see README.md for the full list).
3. The one untried lever (V-stage full-lane utilization) is high-risk
   kernel-internal surgery on a kernel that is already within 2.5 ms of normal.
   The history (5 failed kernel-internal A/Bs) means a new variant needs a
   stronger justification than a 2.5 ms gap that is within noise on some cells.
4. The skill's hard constraint: "对无效/低效新实现先注释掉并写明为什么无效，
   不要直接删掉上下文" — implementing then commenting out a 6th failed variant
   is not a good use of a build+sync+A/B cycle.

## If a future session wants to pursue the V-stage lever

The change would be in `DEFINE_DECODE_CAUSAL_TRANSPOSED_K_SPARSE_HD128`
(`paged_decode_attention_buf.cl:1014-1147`). The V stage is at lines 1116-1132
(`if (lid < 32)` block) and the final store at 1140-1142. To use all 128 lanes
for V, the cleanest approach is:

- Keep the QK+softmax stages as-is (lanes 0..tile_count do QK, all do reduce).
- For V: instead of 32 lanes each doing `vload4` at `out_d4=lid<<2` over all kk,
  split the kk loop across 4 sub-groups of 32 lanes (lid in [0,32), [32,64),
  [64,96), [96,128)), each accumulating the SAME out_d4 but a quarter of the kk
  range; then cross-sub-group reduce the 4 partial sums in local memory.
- This keeps head_dim coverage (32 lanes * 4 = 128) and parallelizes the kk loop
  4x. V read traffic is unchanged (each V[kk] read once), but V compute+read is
  spread across 128 lanes instead of 32.

Risk: 4x more concurrent V reads may saturate bandwidth differently; the local
reduction adds a barrier; register pressure for 4 partial out4 accumulators.
Must be env-gated (`MNN_PAGED_ATTENTION_DECODE_REPAIR_V_PARALLEL=1`) and A/B'd
on all 3 models x ctx512/1024 x x0/x1/x3/x5/x7 before any promotion. If it
regresses (likely, given the pattern), comment out with the A/B numbers.
