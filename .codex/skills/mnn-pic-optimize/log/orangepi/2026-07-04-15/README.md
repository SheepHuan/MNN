# OrangePi Decode-Repair Defer Logits Readback

## Summary

- Scope: OrangePi OpenCL PIC dualgraph decode-repair, ctx 512/1024, MiniCPM5-1B, Llama3.2 3B, Qwen3-4B.
- Change: defer decode-repair `forwardRaw` logits `readMap` until sampler consumption; keep graph output at one vocab row via existing `logitsLastIdx`.
- Validity: no failures, no P0 log hits, all 30 PIC rows used `mnn_token_id_sparse_decode`.
- Official `decode_tpot_ms` improved on all 30 PIC points. Average PIC deltas: x0 -8.4%, x1 -7.3%, x3 -6.7%, x5 -4.5%, x7 -4.5%.
- Caveat: current `decode_tpot_ms` is based on `context->decode_us`, which excludes sampling. This change removes eager logits readback from next-logits forward timing; endpoint wall-time should be measured separately before claiming full wall improvement.

## Artifacts

- Baseline run: `.cache/mnn-pic-benchmark/decode_experiment/decode_unified_family_orangepi_ctx512_1024_suffix1_20260704_145211/`
- New run: `.cache/mnn-pic-benchmark/decode_experiment/decode_defer_logits_orangepi_ctx512_1024_suffix1_20260704_154948/`
- Profile smoke: `.cache/mnn-pic-benchmark/decode_experiment/profile_orangepi_defer_logits_minicpm_ctx512_x0_x1_x5_20260704_154745/`

