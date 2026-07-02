# Rhino Decode x0 PagedCache-Native Follow-up

## Summary

- Target: MiniCPM5-1B, Rhino Pi-X1 / Adreno OpenCL, PIC full-reuse x=0 no-repair decode.
- Change: kept decode on PagedCache-native identity path; disabled the old env-triggered transposed-K mirror route; rewrote `headDim=128,q=1,identity slot` decode kernel toward CUDA-style tile online softmax.
- Result: ctx512 improved modestly to `58.18 ms/token`; ctx1024 improved to `62.08 ms/token`.
- Interpretation: PagedAttention attention is no longer the only large gap. Current attention profile is about `13-15 ms/token`; the remaining gap is spread across MLP, lm_head, QKV/O projection, rotary/raster/norm and graph/runtime overhead.
- Row32 A/B: forcing row32 for identity decode was worse (`59.51/64.44 ms/token` for ctx512/1024), so the current row64 heuristic is kept.
- Normal LLM reference on the same Rhino artifact/cache is still `23.44 ms/token` at ctx512 and `24.92 ms/token` at ctx1024; PIC x0 remains about `2.5x` slower end to end.
- Current root cause direction: x0 reaches the PagedCache-native q=1 attention kernel, but the exported/runtime graph still executes decode through PIC graph-boundary ops (`PicScoreAttention`, `PicSparseAttention`, many gather/shape/raster/while ops). The next fix should provide a no-repair decode graph/runtime route that behaves like ordinary decode after PagedCache lookup.

## Image KV Judgment

Adreno image reads may still be worth A/B for V-cache, but it is not the first fix. Maintaining an image mirror for K/V would add update/copy complexity and does not match the CUDA PagedCache-native direction. The next high-confidence work is graph/dense decode cleanup and per-op launch reduction, then optional V image A/B if attention remains a limiting share.

See `context.md` for commands, profile attribution, and next patch list.
