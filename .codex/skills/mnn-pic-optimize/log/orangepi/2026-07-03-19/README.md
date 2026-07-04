# OrangePi Decode Path Audit

## Summary

Post-audit update: the measurements below were taken before the x0/x>0
decode-repair unification change. Current code treats `repair_tokens=0` as
`decode_refine.enabled=true, tokens_per_decode_step=0`, requires
`mnn_token_id_sparse_decode`, and routes x0 as active rows=1 in the same
decode-repair family; the older "no decode_refine / identity fused-KV x0"
conclusions below are historical baseline evidence, not the current default.

- Primary run id: `decode_dualgraph_validated_orangepi_ctx512_1024_20260704_0001`
- A/B run id: `decode_dualgraph_x0_transposed_k_orangepi_ctx512_1024_20260704_0001`
- Repair A/B run id: `decode_dualgraph_transposed_k_repair_orangepi_ctx512_1024_20260704_0001`
- Path profile run id: `decode_path_profile_default_orangepi_minicpm_ctx512_20260704_0001`
- Auto-route profile run id: `decode_auto_route_profile_orangepi_ctx512_20260704_0001`
- Auto-route formal run id: `decode_dualgraph_auto_route_orangepi_ctx512_1024_20260704_0001`
- X0 profile run id: `decode_x0_profile_pic_orangepi_ctx1024_20260704_0002`
- Gate split validation run ids: `decode_gate_split_combined_orangepi_ctx1024_20260704_0002`,
  `decode_gate_split_combined_x013_orangepi_ctx1024_20260704_0001`
- Device: Orange Pi 5 Plus / Mali OpenCL
- Frequency: locked max (`cpu=max,gpu=max,ddr=max`)
- Scope: true normal q=1 decode TPOT vs PIC dualgraph x0 and PIC dualgraph decode-repair x=1/3/5/7 at context 512/1024.

## Findings

- `true-normal-llm-x0` uses normal export and `llm_bench -kv true -n 16`; it is not PIC normal.
- `repair_tokens=0` is `PIC dualgraph x0`, with no `decode_refine`.
- `repair_tokens>0` uses `mnn_token_id_sparse_decode`, with active rows `repair_tokens + 1`.
- The earlier run `decode_q1_sparse_flash_orangepi_ctx512_1024_20260704_0001` had a label/path issue: OrangePi MiniCPM used a real dualgraph model, but Llama3.2-3B and Qwen3-4B pointed at non-dualgraph PIC directories without `llm_decode.mnn`. Those Llama/Qwen `pic x0` numbers must not be used as PIC dualgraph x0.
- Llama3.2-3B and Qwen3-4B were exported again with `--pic_export_device orangepi --pic_export_decode_graph`, synced to OrangePi, and the decode harness now validates `llm_decode_model` plus `llm_decode.mnn` before labeling any row as `pic-dualgraph-x0`.
- `pic_server /` now exposes `pic_dualgraph_decode`, and the standalone decode client defaults to requiring it.
- The q=1 suffix prefill blocker was real: full-reuse uses one suffix token in this benchmark, and later-layer `PicSparseAttention` q=1 was previously rejected by sparse flash support. The OpenCL check now allows later-layer PIC sparse q=1.
- Pure zero-suffix decode is still unsupported by `pic_server`: it returns `PIC prompt suffix tokenization produced no tokens`.
- Decode routing now has a fixed OrangePi default for repair q>1: q=1 no-repair stays on `decode_causal_attention_hd128_identity_fused_kv`; repair q>1 uses transposed-K qtile on Mali for `kv_heads>=8` or `attnLen<=4`, otherwise remains on the existing sparse flash/qsplit repair path. q1 transposed-K A/B uses `MNN_PAGED_ATTENTION_DECODE_Q1_TRANSPOSED_K=1`; repair qtile A/B uses `MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE=0/1`.
- Auto-route profile confirms MiniCPM only uses qtile for x1/x3 (24 layers x 2 = 48 qtile ops), while Llama and Qwen use qtile for x1/x3/x5/x7. P0 log scan remains clean.
- X0 profile at ctx1024 confirms no repair gate is active: Llama/Qwen each have one `PIC decode graph route`, zero `mnn_token_id_sparse_decode`, zero transposed/sparse qtile decode ops, and only q=1 `decode_causal_attention_hd128_identity_fused_kv` in measured decode. The graph-level op names still include `PicScoreAttention` / `PicSparseAttention`, so future analysis must not treat those names alone as decode-repair evidence.
- A later q=1 auto transposed-K experiment tried to enable transposed-K on Mali for ctx>=1024 GQA shapes. Profile confirmed the kernel could fire, but formal TPOT did not recover Llama/Qwen x0 and slightly regressed/noised other rows. That automatic q1 gate was removed: production q1 x0 stays fixed on identity fused-KV; transposed-K q1 remains an explicit A/B/profile variant only.
- The post-split x0-only and x1-only runs passed, but the first same-server x0->x1 run crashed during the second full-reuse suffix prefill, before decode repair routing started. Root cause was not the repair qtile kernel: `_externalLayerMappedTargetKey` omitted `meta->request_generation`, while the async read request key already included it. After `/reset`, the registry could reuse the previous request/decode graph mapped target for the next request.
- The mapped target key now includes `request_generation`. Same-server x0,x1 and x0,x1,x3 validation passes. Route scan for `decode_gate_split_combined_x013_orangepi_ctx1024_20260704_0001`: q1 transposed route = 0, repair qtile route = 56, ordinary q1 rows = 28, q=2 repair rows = 28, q=4 repair rows = 28, with no `decode_prepare_inside_decode=1`, `target unavailable`, async read failure, or `ERROR`.

## TPOT ms

| model | ctx | true normal x0 | PIC dualgraph x0 | PIC repair x1 | x3 | x5 | x7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniCPM5-1B | 512 | 46.637 | 46.778 | 122.330 | 120.113 | 155.766 | 155.741 |
| MiniCPM5-1B | 1024 | 50.116 | 49.894 | 143.521 | 144.380 | 182.779 | 181.971 |
| Llama3.2 3B | 512 | 121.645 | 124.253 | 260.553 | 257.711 | 404.882 | 404.882 |
| Llama3.2 3B | 1024 | 126.032 | 142.155 | 310.542 | 313.329 | 455.551 | 457.611 |
| Qwen3-4B | 512 | 149.789 | 151.603 | 329.573 | 332.351 | 512.853 | 520.691 |
| Qwen3-4B | 1024 | 155.065 | 176.901 | 388.424 | 393.066 | 594.836 | 590.635 |

## X0 A/B

| model | ctx | default x0 | transposed-K x0 | transposed/default |
| --- | ---: | ---: | ---: | ---: |
| MiniCPM5-1B | 512 | 46.778 | 47.870 | 1.023 |
| MiniCPM5-1B | 1024 | 49.894 | 51.223 | 1.027 |
| Llama3.2 3B | 512 | 124.253 | 123.721 | 0.996 |
| Llama3.2 3B | 1024 | 142.155 | 129.910 | 0.914 |
| Qwen3-4B | 512 | 151.603 | 152.310 | 1.005 |
| Qwen3-4B | 1024 | 176.901 | 171.454 | 0.969 |

## Repair A/B

`transposed/default` below is the older global-gate A/B data. Current code splits that gate: use `MNN_PAGED_ATTENTION_DECODE_Q1_TRANSPOSED_K=1` for q1 and `MNN_PAGED_ATTENTION_DECODE_REPAIR_SPARSE_QTILE=0/1` for repair qtile.

| model | ctx | x1 | x3 | x5 | x7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniCPM5-1B | 512 | 0.959 | 0.991 | 1.093 | 1.120 |
| MiniCPM5-1B | 1024 | 0.904 | 0.984 | 1.142 | 1.151 |
| Llama3.2 3B | 512 | 0.862 | 0.901 | 0.958 | 0.977 |
| Llama3.2 3B | 1024 | 0.782 | 0.816 | 0.949 | 0.957 |
| Qwen3-4B | 512 | 0.871 | 0.883 | 0.966 | 0.971 |
| Qwen3-4B | 1024 | 0.803 | 0.850 | 0.933 | 0.964 |

## Conclusion

Validated dualgraph x0 is now close to true normal at ctx512 for all three models. At ctx1024, MiniCPM still matches true normal, but Llama is ~13.0% slower and Qwen is ~14.1% slower; the auto-route change does not solve that x0 overhead, so x0 still needs a separate no-repair decode investigation. For repair x>0, the fixed auto-route promotes the qtile family where it was clearly faster: Llama/Qwen improve by 4%-22% depending on x/context, MiniCPM x1 improves by about 9%-10%, and MiniCPM x5/x7 stay on the existing sparse repair family with only small run-noise deltas. This removes the main repair-path env gate from the production default without changing q=1 no-repair.

X0 profile conclusion: PIC x0 is not slowed by suffix/hydrate being counted as decode TPOT, and not by qtile repair routing. The measured decode forward selects `llm_decode.mnn` (`use_decode_graph=1`, `inputs=4`) and its q=1 PagedAttention detail cost is much smaller than full decode forward time. Remaining ctx1024 x0 overhead should be investigated in the PIC q=1 PagedAttention implementation plus decode graph dense/Raster/elementwise overhead versus true normal LLM.

Gate cleanup conclusion: do not promote q=1 transposed-K through a shape/head automatic selector. Keep q=1 identity fused-KV as the fixed OrangePi production route and use debug logs (`PIC OpenCL PA decode hd128 route ...`) to distinguish production q1, explicit q1 variants, and q>1 sparse qtile repair.

Same-server stability conclusion: the gate split is now validated independently of the mapped target lifetime bug. The crash pattern was cross-request PagedCache target reuse, fixed by request-generation scoping; future gate analysis should check route fields first, then inspect request-generation scoped external target lookup if a failure happens before any repair route log.

Follow-up q1 gate/perf check: the apparent `264 ms/token` x0 was from a heavy
debug, 2-token, no-warm validation run and is not a formal regression signal.
Current profile-off warm Llama3.2-3B ctx1024 x0 is back to `129.326 ms/token`
on the default q1 identity route, and explicit
`MNN_PAGED_ATTENTION_DECODE_Q1_TRANSPOSED_K=1` is `130.138 ms/token`. The old
`MNN_PAGED_ATTENTION_DECODE_TRANSPOSED_K=1` command name no longer enables the
q1 variant, so old A/B scripts must be updated or a deprecated alias added.

## Auto-Route TPOT ms

| model | ctx | true normal x0 | PIC dualgraph x0 | PIC repair x1 | x3 | x5 | x7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniCPM5-1B | 512 | 46.569 | 47.344 | 111.385 | 117.680 | 157.980 | 159.653 |
| MiniCPM5-1B | 1024 | 50.041 | 49.961 | 129.773 | 145.492 | 186.168 | 183.379 |
| Llama3.2 3B | 512 | 122.154 | 124.025 | 223.906 | 232.785 | 386.902 | 388.529 |
| Llama3.2 3B | 1024 | 125.637 | 141.984 | 243.976 | 260.397 | 425.027 | 439.663 |
| Qwen3-4B | 512 | 149.922 | 151.659 | 287.284 | 294.319 | 499.941 | 500.262 |
| Qwen3-4B | 1024 | 155.188 | 177.036 | 304.178 | 332.883 | 563.022 | 570.106 |

## Auto-Route / Default

Lower is better. Normal is included only to show run drift.

| model | ctx | normal | PIC x0 | x1 | x3 | x5 | x7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniCPM5-1B | 512 | 0.999 | 1.012 | 0.911 | 0.980 | 1.014 | 1.025 |
| MiniCPM5-1B | 1024 | 0.999 | 1.001 | 0.904 | 1.008 | 1.019 | 1.008 |
| Llama3.2 3B | 512 | 1.004 | 0.998 | 0.859 | 0.903 | 0.956 | 0.960 |
| Llama3.2 3B | 1024 | 0.997 | 0.999 | 0.786 | 0.831 | 0.933 | 0.961 |
| Qwen3-4B | 512 | 1.001 | 1.000 | 0.872 | 0.886 | 0.975 | 0.961 |
| Qwen3-4B | 1024 | 1.001 | 1.001 | 0.783 | 0.847 | 0.947 | 0.965 |
