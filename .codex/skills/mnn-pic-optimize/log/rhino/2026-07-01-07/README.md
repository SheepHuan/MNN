# Rhino 2026-07-01 07

## Summary

- Backend-family export isolation is a hard rule for the Adreno/Rhino decode-repair graph experiments.
- `--pic_decode_fusion_backend generic` leaves native decode-repair graph rewrites disabled.
- `--pic_decode_fusion_backend cuda/jetson` keeps the CUDA/Jetson family: `PicGateUpSiluWeightOnly`, `PicLinearNhwcWeightOnly`, `PicPackedSiluMul`.
- `--pic_decode_fusion_backend adreno/rhino/rhinopi/aidlux_adreno_opencl` uses Adreno-only op names: `PicAdrenoGateUpSiluWeightOnly`, `PicAdrenoLinearNhwcWeightOnly`, `PicAdrenoPackedSiluMul`, `PicAdrenoSiluMulNhwc`, `PicAdrenoTinyMlpWeightOnly`.
- OpenCL runtime creation refuses `PicAdreno*` Extra ops unless the runtime GPU type is ADRENO. OrangePi/Mali must not consume these ops.
- Export generic backend now also masks `--pic_decode_tiny_fusion`, so `--pic_decode_fusion_backend generic` does not emit native decode-repair tiny ops by accident.
- Rhino `DecodeRepairMlpSiluNhwcDown` direct-op rows `2..8` passed accuracy and was slightly faster than split-chain, but row `1` regressed.
- The real generated `adreno-silu-nhwc-down` endpoint retry did not improve `extra_ms`; x=5/x=7 were worse than rowheur2 at both ctx512 and ctx1024. Do not promote this graph path.
- `DecodeRepairMlpStreamedTile` with larger down `OC_TILE` was rejected at direct-op level. Best tested tile was `OC_TILE=16`, but rows `2/4/6/8` still took about `32-34 ms` versus row-aware split-chain `0.76-1.45 ms`.
- The narrow activation/down boundary is insufficient. The next credible Rhino route remains a true fused tiny-row MLP kernel, or a targeted projection optimization that preserves row-aware tiny dense family selection.
- Rhino service on port `18133` was restored to the guarded `minicpm5-adreno-tiny-mlp` rows `5..8` model after rejecting `adreno-silu-nhwc-down`; startup error grep is clean.
