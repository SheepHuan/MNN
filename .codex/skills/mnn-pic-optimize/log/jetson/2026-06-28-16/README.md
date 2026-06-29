# Jetson PIC Decode Repair Dense Notes

## Summary

- Goal remains reducing Jetson CUDA Llama3.2-3B `lagged_attention_hkvd` decode-repair `x=3/5/7` TPOT below 100 ms without changing selector, PagedCache, repair token count, or decode-only token-id sparse semantics.
- This hour focused on the shared rows=4/6/8 compact dense bottleneck identified from x=0/1/3/5/7 profile.
- cuBLAS compute/algo sweeps did not improve the small-M floor; `CUBLAS_GEMM_DEFAULT_TENSOR_OP` remained fastest or tied.
- cublasLt did not provide a lower floor for rows=4/6/8 MLP projection chains.
- A naive WMMA rows=4..8 experiment compiled and was tested, but it was much slower than the current rows45 cuBLAS/static-dequant path and was removed from production code.
- New test-only `DecodeRepairGateUpBatchedGemmFloor` showed cuBLAS pointer-array and strided-batched GEMM are slower than two separate rows45 GEMMs for both MLP gate/up (`3072->8192`) and attention k/v (`3072->1024`), so batched cuBLAS projection fusion is rejected.
- Existing batch<=3 V14 tiny GEMV is not a useful row-splitting fallback: estimated `3+1`, `3+3`, and `3+3+2` splits are slower than the current rows=4/6/8 cuBLAS plateau.

## Decision

Do not continue with env tuning, cublasLt replacement, batched cuBLAS projection fusion, V14 row splitting, or one-warp-per-OC-tile WMMA for the current bottleneck. The next useful direction is a real compact dense kernel/fusion that reduces fixed launch/GEMM overhead for rows=4/6/8, or a graph/export fusion that reduces the number of projection launches while preserving semantics.
