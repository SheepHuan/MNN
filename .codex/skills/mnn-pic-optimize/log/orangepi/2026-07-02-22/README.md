# OrangePi 2026-07-02 22

## Summary

- Question: whether a single OpenCL decode attention kernel can reduce the OrangePi/Rhino `normal_x0` vs `pic_x0` gap, focusing first on Qwen3-4B PIC no-repair decode at `ctx512`.
- Implemented an experimental grouped-GQA fused decode kernel for `head_dim=128`: `decode_causal_attention_hd128_identity_fused_kv_gqa`.
- The experimental path is guarded by `MNN_PAGED_ATTENTION_DECODE_GQA_FUSED=1`. Default production remains `decode_causal_attention_hd128_identity_fused_kv`.
- Build and device smoke passed on OrangePi. No `ERROR`, `CL_OUT_OF_RESOURCES`, `target unavailable`, or async PIC cache read failures were seen in the tested logs.
- Default clean profile, Qwen3-4B `ctx512/max_tokens=2/repair_tokens=0`: `decode_tpot_ms=1002.544`; landed decode fused-KV profile lines averaged about `1.906ms` per attention layer.
- GQA clean profile, same shape with `MNN_PAGED_ATTENTION_DECODE_GQA_FUSED=1`: `decode_tpot_ms=1169.826`; landed GQA fused-KV profile lines averaged about `2.594ms` per attention layer.
- Earlier debug-profile A/B also showed the same direction: default `fused_kv` around `1.96ms/layer`, GQA grouped kernel around `2.4-3.0ms/layer` depending on first-use/tune outliers.
- Conclusion: the first grouped-GQA single-kernel implementation does not speed up OrangePi Qwen3-4B no-repair decode. Reducing workgroups by grouping query heads is outweighed by serial per-group work and reduced parallelism/occupancy on Mali.
- Keep the GQA kernel as an env-gated experiment only. Do not promote it to default without a new implementation and same-shape A/B proving a win.
- More promising next optimization: keep one workgroup per query head or split by head groups while reducing duplicated K/V loads with local/subgroup cooperation, rather than serializing all GQA heads in one workgroup.

## Files

- `context.md`: commands, logs, CSV outputs, and interpretation.
