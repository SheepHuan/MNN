# Jetson 2026-07-01 11

## Summary

- Added a PIC export device contract for `jetson` / `rhinopi` / `orangpi` to prevent backend-specific decode fusion flags from crossing device families.
- Re-exported Llama 3.2 1B Jetson baseline with `--pic_export_device jetson`, `--pic_decode_tiny_fusion`, `score_layer_idx=10`.
- Fresh Jetson export still fails during `/v1/prefill/text` persistent text cache construction with `forwardRaw outputs empty`.
- Old `silumul-score10` baseline remains the only verified production path in this test; it completed Jetson CUDA decode x=0/1/3/5/7 at context 512.

## Entries

- [context.md](context.md)
