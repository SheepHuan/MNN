# Jetson CUDA Decode Repair HKVD Dense Check

## Summary

- Goal remains reducing Llama3.2-3B Jetson CUDA `lagged_attention_hkvd` decode repair `x=3/5/7` TPOT toward the `x=0` normal decode baseline without changing repair rows, HKVD selection, PagedCache semantics, or token-id sparse decode behavior.
- Rebuilt and synced the current test-only `run_test.out` artifact to Jetson:
  - local: `.cache/output/mnn/artifacts/jetson_cross_cuda_test/bin/run_test.out`
  - remote: `/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda_test/bin/run_test.out`
- Re-ran rows=4/6/8 direct-op benches for 3B shapes (`hidden=3072`, `inter=8192`, `kv=1024`, CUDA fp16, memory low).
- `LinearConvertChain` shows layout convert overhead is small, usually about `0.006-0.014 ms` per projection, so it is not the main explanation for the `x=3/5/7` TPOT growth.
- `PicDecodeMlp` split chain is already close to the cuBLASLt projection floor: rows=4/6/8 are `1.5149 / 1.5422 / 1.5254 ms`; cuBLASLt projection sums are `1.4765 / 1.4713 / 1.4968 ms`.
- Packed gate/up and batched gate/up do not provide a useful production candidate for rows=4/6/8. Batched GEMM is about `0.22-0.24 ms` slower than two separate projections.
- New test-only `DecodeRepairGateUpParallelGemmFloor` shows two-stream cuBLAS does not overlap usefully on Jetson sm72:
  - MLP gate/up `3072->8192`: rows=4/6/8 parallel pair delta is `+0.0024 / -0.0014 / +0.0092 ms`.
  - attention k/v-style `3072->1024`: rows=4/6/8 parallel pair delta is `+0.0071 / +0.0108 / +0.0077 ms`.
- New test-only `DecodeRepairQkvParallelGemmFloor` shows three-stream cuBLAS for attention q/k/v also has only tiny overlap:
  - q/k/v `3072->3072,1024,1024`: rows=4/6/8 parallel triple delta is `-0.0036 / -0.0080 / -0.0125 ms`.
  - The gain is at most about `0.013 ms/layer`, well below the target needed to move `x=3/5/7` endpoint TPOT.
- Jetson CUDA sysroot does not expose a `cublasGemmGrouped*` API, and earlier rows45 cuBLAS compute/algo/policy sweeps were already rejected as unstable or regressive, so there is no remaining low-risk cuBLAS policy knob to retune.
- Focused concat-fusion floor:
  - `hidden_to_qkv_concat` is slower than separate q+k+v by `+0.0344 / +0.0265 / +0.0278 ms` for rows=4/6/8.
  - packed MLP gate/up chain is only `-0.0251 / -0.0299 / +0.0085 ms` vs split chain for rows=4/6/8.
  - Simple graph/export concat fusion is therefore too small and not stable enough to be the main production route.
- Added a test-only true-case CUDA MLP candidate, `PicGateUpSiluWeightOnly`, that fuses `gate/up INT4 weight-only + SiLU(gate) * up` and drives it from exporter-like external INT4 weights in the direct-op bench. It runs correctly in the bench but is slower than the current split chain:
  - rows=4: `+0.2457 ms/layer`
  - rows=6: `+0.6707 ms/layer`
  - rows=8: `+1.1257 ms/layer`
  - This scalar INT4 fused MLP route is rejected for production.

## Decision

Do not default any new rows=4/6/8 endpoint from this evidence. The current direct-op results show only tens of microseconds per layer of plausible savings from cuBLASLt or layout cleanup, below the `0.2-0.3 ms/layer` target needed to move endpoint TPOT materially.

Also reject multi-stream overlap for MLP gate/up, attention k/v pairs, and attention q/k/v triples. Simple concat fusion for attention q/k/v is slower, and MLP gate/up packing is only a sub-`0.03 ms/layer` marginal result. The next useful implementation direction remains a real compact dense kernel/fusion that materially reduces rows=4/6/8 projection cost while preserving the existing decode repair semantics.

Also reject the new scalar `PicGateUpSiluWeightOnly` fused MLP candidate. It proves the graph/backend hook shape, but the direct-op data shows it loses badly to the existing static-dequant/cuBLAS/CUTLASS split MLP path for rows=4/6/8.
