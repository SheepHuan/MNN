# Rhino 2026-07-01 03

## Summary

- Real generated Adreno/Rhino `PicAdrenoSiluMulNhwc` + `PicAdrenoLinearNhwcWeightOnly` graph now initializes on Rhino after adding the Adreno NHWC linear Extra type to the external-weight rebuild whitelist.
- Smoke passed on Rhino with the generated `OpenBMB__MiniCPM5-1B-pic-boundary-adreno-silu-nhwc-down` model.
- Decode repair `repair_tokens > 0` is usable and roughly matches rowheur2 baseline; `ctx=1024` improves x=1/x=3 and is near-neutral at x=5/x=7.
- `repair_tokens=0` no-repair decode path regresses badly and is reproducible; do not carry this graph to OrangePi until that path is separated or fixed.
- Follow-up fixed the x=0 regression for the Adreno tiny-MLP real graph by routing ordinary `headDim=128` decode on Adreno through the hd128 causal decode kernel when graph-boundary `PicScoreAttention/PicSparseAttention` is inactive.
- New tiny-MLP real-graph matrix is now healthy for x=0 and roughly neutral for x=1/3/5/7. It is not yet a clear replacement for rowheur2 because ctx512 x=1/3/7 and ctx1024 x=5/7 remain slightly slower than rowheur2.
- Export isolation rule is preserved: `PicAdreno*` graph rewrites stay behind `--pic_decode_fusion_backend adreno/rhino`; the runtime fix touched only OpenCL Adreno ordinary decode and did not change CUDA/Jetson execution.
- Export isolation was tightened further: Adreno gate/up+silu direct export now uses `PicAdrenoGateUpSiluWeightOnly` instead of the CUDA-recognized `PicGateUpSiluWeightOnly`, and OpenCL refuses `PicAdreno*` Extra ops on non-Adreno runtimes. Rhino `pic_server` cross-build passed.
- Backend-family isolation was corrected after review: `--pic_decode_fusion_backend cuda/jetson` still emits the Jetson CUDA family (`PicGateUpSiluWeightOnly`, `PicLinearNhwcWeightOnly`, `PicPackedSiluMul`), while `adreno/rhino` emits `PicAdreno*`; `generic` does not enable these graph rewrites.
- The last shared packed-silu edge was split as well: Adreno/Rhino packed gate/up export now emits `PicAdrenoPackedSiluMul`, while CUDA/Jetson keep `PicPackedSiluMul`.
- Production `PicAdrenoTinyMlpWeightOnly` was tested with a shared-NHWC chain fast path. Direct-op rows `2..8` beat split-chain, but row `1` is slower and endpoint rows `2..8` worsened repair `extra_ms`; the runtime guard was narrowed to rows `5..8`.
- Final rows `5..8` endpoint is better than the previous tiny-MLP graph for most repair cases, but x=5/x=7 are still not clearly better than rowheur2. Treat this as a measured partial result, not a promotion candidate.
- OrangePi should not inherit this Adreno graph/op path. If replicated on Mali, add a separate backend family and op names, then first validate row-aware family selection plus endpoint extra_ms.
- A stronger bench-only `gate/up pair output + silu-down` direct-op candidate was tested on Rhino rows `2/4/6/8`. Accuracy passed, but fused-chain latency was slower than the row-aware split-chain for every row count, so it was rejected and not wired into real generated code.
- A bench-only streamed-tile MLP candidate was also tested. It avoids the full `[rows, inter]` gate/up writeback, but recomputes gate/up for each hidden output tile; rows `2/4/6/8` were accurate but 177-439 ms versus 0.61-0.96 ms split-chain, so it is rejected and must not be production-wired.
- The next Rhino MLP route should either be a true single-kernel tiny-row MLP that keeps intermediate gate/up values on chip until down projection without recomputing gate/up per output tile, or a narrower graph/runtime path that preserves row-aware child Conv selection and removes only proven layout/activation overhead. Do not carry rejected Adreno candidates to OrangePi.
- Final source/remote state was resynced after the rejected rows `2..8` retry: `PicAdrenoTinyMlpWeightOnly` fast path is rows `5..8`, ordinary `PicGateUpWeightOnly` keeps its original rows `1..8` guard, and Rhino service on port `18133` is running the 2026-07-01 07:32:33 artifact.
