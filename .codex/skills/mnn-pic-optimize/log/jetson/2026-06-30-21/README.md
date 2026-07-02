# Jetson CUDA Decode Repair Down-Proj A/B

## Summary

- Continued the P1 down-proj investigation after the accepted packed gate/up endpoint.
- Tested a diagnostic CUDA path that forced the existing SM70 compact CUTLASS N-wide tile for Llama3.2-3B PIC decode-repair down-proj rows `4/6/8`, shape `8192 -> 3072`.
- Accuracy passed for existing rows4/5, Llama3.2-3B rows4/6/8 gate/up/down, and `PicLinearNhwcWeightOnly` rows1/2/4/6/8.
- Direct-op perf rejected the forced N-wide CUTLASS path:

| rows | default cuBLAS ms | forced CUTLASS ms | delta |
|-----:|------------------:|------------------:|------:|
| 1 | 0.2563 | 0.2635 | +0.0072 |
| 2 | 0.3110 | 0.3074 | -0.0036 |
| 4 | 0.5091 | 0.5760 | +0.0669 |
| 6 | 0.5100 | 0.5771 | +0.0671 |
| 8 | 0.5109 | 0.5793 | +0.0684 |

- Follow-up cuBLASLt heuristic sweep for Llama3.2-3B down-proj rows `4/6/8`
  also failed the direct-op gate:

| rows | default rows45 cuBLAS ms | best cuBLASLt ms | best heuristic | decision |
|-----:|-------------------------:|-----------------:|---------------:|:---------|
| 4 | 0.5014 | 0.5042 | 0 | reject |
| 6 | 0.5008 | 0.5103 | 1 | reject |
| 8 | 0.5062 | 0.5049 | 0 | noise only |

- P2 tiled cuBLAS lower-bound test for streaming `SiLU(gate) * up` tiles into
  down accumulation was also not strong enough to justify production work. The
  only chain-level win came from tile `4096`, and it was just `0.017-0.036
  ms/layer` for rows `4/6/8` while tiled down accumulation itself was flat or
  slower.
- Repeated the existing rows45 cuBLAS `compute=16f` diagnostic. It consistently
  improved isolated rows `4/6/8` down by only about `0.010-0.013 ms/layer`.
  Repeated packed-chain runs were neutral to mildly positive, but the expected
  endpoint impact is far below the `>=3 ms/token` target, so this remains a
  diagnostic knob and was not promoted.
- Tested a temporary target-shape WMMA down kernel gated by
  `MNN_CUDA_PIC_INT4_DOWN_SMALL_WMMA=1`. Accuracy passed, but direct-op perf
  regressed to roughly `1.03-1.06 ms` versus the current cuBLAS `0.51 ms`. The
  diagnostic source branch was removed.
- Tested a bench-only transposed static FP16 down weight layout floor. The
  normal and `fast16` transposed layouts both regressed rows `4/6/8`; only full
  `16f` compute improved, confirming this is another accumulation-policy effect
  rather than a useful layout change.
- Swept the existing non-Lt cuBLAS `cublasGemmEx` algorithm IDs under the
  default 32F compute policy. The current default tensor-op algorithm `99`
  remained best for rows `4/6/8`.
- Added a bench-only output-channel split-N lower-bound test for the down GEMM.
  Parallel chunked cuBLAS gave only about `0.016-0.030 ms/layer` on the target
  rows in the repeat run, below the endpoint target and not worth promoting.

| tile | rows | full chain ms | tiled floor ms | tiled delta | down delta |
|-----:|-----:|--------------:|---------------:|------------:|-----------:|
| 1024 | 4 | 1.5063 | 1.7047 | +0.1984 | +0.1074 |
| 1024 | 6 | 1.5330 | 1.7452 | +0.2122 | +0.1074 |
| 1024 | 8 | 1.5727 | 1.7898 | +0.2171 | +0.1342 |
| 2048 | 4 | 1.5453 | 1.5543 | +0.0090 | +0.0226 |
| 2048 | 6 | 1.5068 | 1.5667 | +0.0599 | +0.0432 |
| 2048 | 8 | 1.5280 | 1.5817 | +0.0537 | +0.0455 |
| 4096 | 4 | 1.5129 | 1.4874 | -0.0255 | +0.0028 |
| 4096 | 6 | 1.5030 | 1.4668 | -0.0361 | +0.0041 |
| 4096 | 8 | 1.5224 | 1.5050 | -0.0174 | +0.0149 |

Rows45 cuBLAS `compute=16f` repeat summary:

| rows | isolated default ms | isolated 16f ms | delta | packed chain default ms | packed chain 16f ms | delta |
|-----:|--------------------:|----------------:|------:|------------------------:|--------------------:|------:|
| 4 | 0.4988 | 0.4867 | -0.0121 | 1.5240 | 1.4962 | -0.0278 |
| 6 | 0.5054 | 0.4921 | -0.0132 | 1.5128 | 1.5097 | -0.0031 |
| 8 | 0.5034 | 0.4932 | -0.0102 | 1.5386 | 1.5166 | -0.0220 |

WMMA diagnostic:

| rows | default cuBLAS ms | WMMA ms | delta |
|-----:|------------------:|--------:|------:|
| 4 | 0.5164 | 1.0593 | +0.5429 |
| 6 | 0.5055 | 1.0276 | +0.5221 |
| 8 | 0.5161 | 1.0542 | +0.5381 |

Transposed static layout floor:

| rows | production NT ms | transposed NN ms | transposed NN fast16 ms | transposed NN 16f ms |
|-----:|-----------------:|-----------------:|------------------------:|---------------------:|
| 4 | 0.5335 | 0.5635 | 0.5636 | 0.4935 |
| 6 | 0.5328 | 0.5597 | 0.5651 | 0.4964 |
| 8 | 0.5475 | 0.5672 | 0.5715 | 0.5057 |

cuBLAS 32F algorithm sweep bests:

| rows | best algo | best ms |
|-----:|:----------|--------:|
| 4 | 99 | 0.5007 |
| 6 | 99 | 0.5040 |
| 8 | 99 | 0.5029 |

Split-N parallel down GEMM repeat:

| chunks | rows | single ms | split_parallel ms | delta |
|-------:|-----:|----------:|------------------:|------:|
| 4 | 4 | 0.5125 | 0.4822 | -0.0303 |
| 4 | 6 | 0.5239 | 0.4963 | -0.0276 |
| 4 | 8 | 0.5136 | 0.4927 | -0.0209 |
| 8 | 4 | 0.4973 | 0.4816 | -0.0156 |
| 8 | 6 | 0.5138 | 0.4929 | -0.0208 |
| 8 | 8 | 0.5116 | 0.4915 | -0.0201 |

## Decision

Reject the tested P1 down-proj retuning attempts and the simple P2 tiled cuBLAS
floor:

1. forced N-wide CUTLASS loses rows4/6/8 by about `0.067 ms/layer`;
2. cuBLASLt non-top1 heuristic sweep is tied/noisy and does not beat the current rows45 cuBLAS path across all rows4/6/8;
3. tiled cuBLAS gate/up plus beta-accumulated down has no down-proj win and the best chain lower bound is far below the required endpoint-level gain.
4. rows45 `compute=16f` is real but too small to matter for endpoint TPOT and changes accumulation precision, so it stays diagnostic.
5. the temporary WMMA small-M down kernel is about 2x slower than rows45 cuBLAS.
6. transposing the static FP16 down cache is not useful with 32F or fast16
   compute; the only win again comes from full `16f` compute and is not enough
   for the endpoint target.
7. explicit cuBLAS algo selection has no hidden 32F win; the existing tensor-op
   default remains the fastest supported option in the sweep.
8. split-N parallel output-channel chunks produce only a small lower-bound win
   and add multi-stream/cuBLAS scheduling complexity inside every down op; the
   maximum plausible endpoint gain is far below the requested `>=3 ms/token`.

The default down-proj branch is already `conv_fpa_intb_1x1_rows45_cublas` on static dequantized FP16 weights and is effectively at the same level as the cuBLASLt floor. The next credible direction is still a true fused MLP kernel, but not by splitting the work into ordinary cuBLAS tiles or by extending the existing scalar `PicGateUpSiluWeightOnly` kernels. A viable P2 design must preserve tensor-core-quality packed gate/up work, compute activation in registers/shared memory, and reduce down accumulation overhead enough to clear at least about `0.10 ms/layer` on rows4/6/8 before endpoint testing.
