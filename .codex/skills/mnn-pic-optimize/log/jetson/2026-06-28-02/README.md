# Jetson CUDA decode repair attention-rank capture

## Summary

- Decode repair benchmark must keep `x=0` as normal decode TPOT baseline.
- `lagged_attention_hkvd` is implemented as compact attention-rank filtering of the existing HKVD ranking: keep only PIC local token ranks, not full attention matrices.
- Attention-rank capture is gated by `PagedKVMeta::needsPicDecodeAttentionRankCapture(layerIndex)`, so only the configured `attention_layer_idx` may capture rank.
- Existing Jetson profile confirms `decode_attention_rank` appears only on layer 1 for this run.
- Fresh Jetson fused smoke confirms CUDA row-compressed decode attention emits `fused_score_in_attention=1` only on layer 1.
- Fused rank score accumulation now normalizes by `valid_heads * batch`, matching standalone rank-kernel average semantics.
- qtile for decode repair remains experimental and default-off for small repair rows; it is not a default optimization for `x<=7`.

## Profile Run

```text
run_id=decode_repair_profile_20260628_022816
device=Jetson AGX Xavier
model=Llama3.2 3B PIC
backend=CUDA
context=1024
mode=full-reuse
selector=lagged_attention_hkvd
repair_tokens=0,1,3,7
max_tokens=8
profile_env=MNN_PAGED_ATTENTION_PROFILE=1 MNN_PIC_GRAPH_PROFILE=1 MNN_PIC_GRAPH_PROFILE_TOP=1000
csv=.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_profile_20260628_022816/profile_decode.csv
log=.cache/mnn-pic-benchmark/decode_repair_profile/decode_repair_profile_20260628_022816/pic_server.log
```

Profile-mode TPOT, not formal latency:

```text
x=0 none                  TPOT=142.903 ms status=ok
x=1 lagged_attention_hkvd TPOT=176.326 ms status=ok
x=3 lagged_attention_hkvd TPOT=263.167 ms status=ok
x=7 lagged_attention_hkvd TPOT=304.119 ms status=ok
```

Accepted interpretation:

- `x=0` is the required normal decode baseline for this benchmark family.
- The profile run is useful for attribution only; graph/profile callbacks add synchronization overhead.
- Next formal TPOT must rerun without `MNN_PAGED_ATTENTION_PROFILE` and `MNN_PIC_GRAPH_PROFILE`.

## Fused Smoke

```text
run_id=decode_repair_fused_smoke_20260628_024737
env=MNN_PAGED_ATTENTION_PROFILE=1
repair_tokens=1
max_tokens=4
decode_attention_rank layer=1 count=4 avg=284.2 us min=217 us max=430 us
fused_score_in_attention=1 count=4
other_rank_layers=0
error_lines=0
```

## Non-Profile TPOT Smoke

This is a Jetson-only smoke run, not a formal decode matrix.

```text
run_id=decode_repair_tpot_smoke_20260628_025031
env=<no profile env>
context=1024
mode=full-reuse
max_tokens=8
repeats=3
warm_repeats=1

x=0 none                  TPOT=87.163 ms  status=ok
x=1 lagged_attention_hkvd TPOT=158.793 ms status=ok
x=3 lagged_attention_hkvd TPOT=185.317 ms status=ok
x=5 lagged_attention_hkvd TPOT=184.994 ms status=ok
x=7 lagged_attention_hkvd TPOT=207.958 ms status=ok
```

Optimization direction after the smoke:

- Rank capture overhead is no longer the primary target after fusion.
- Small-budget decode repair is dominated by tiny-row dense Linear/MLP plus row-compressed `PicSparseAttention`.
- Non-compute overhead is also visible: `Raster` / `BinaryOp` / `While` account for roughly 13%-19% in graph-profile requests.
- qtile remains a non-default experiment for larger active rows; it is not the right default lever for `x<=7`.
