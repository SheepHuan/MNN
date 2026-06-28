# Jetson CUDA decode repair 07:00 context

## Starting point

Previous accepted non-profile smoke:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_default_smoke_20260628_0650/summary.csv
x=0 none                  TPOT=86.443 ms
x=1 lagged_attention_hkvd TPOT=98.521 ms
x=3 lagged_attention_hkvd TPOT=123.406 ms
x=5 lagged_attention_hkvd TPOT=132.097 ms
x=7 lagged_attention_hkvd TPOT=133.897 ms
```

Rejected immediately before this hour:

- `PagedAttentionExecution.cu` output memset skip for decode/repair causal attention.
- Result: no improvement; source reverted.

## Rows4-8 cuBLAS A/B

Change:

```text
source/backend/cuda/execution/weight_only_quant/ConvFpAIntBExecution.cu
```

The existing PIC decode-repair MLP-like cuBLAS route was extended from active rows `4/5` to active rows `4..8`:

```text
picRows45CublasMatches(): batch < 4 || batch > 8 returns false
```

Rationale:

- `x=5/x=7` correspond to active rows `6/8`.
- Before this change those high-row MLP-like compact Linears mainly used the static CUTLASS path.
- The existing rows4/5 cuBLAS route had already helped `x=3`; extending only the batch range is a narrow dense-path A/B and does not change attention or rank capture.

Build and sync:

```bash
MNN_TARGET_DEVICE=jetson CROSS_COMPILE=ON ENABLE_CROSS_CUDA=ON CUDA_ARCHS=72 \
BUILD_TARGET=pic_server BUILD_MNNCONVERT=0 INSTALL_AFTER_BUILD=1 JOBS=96 \
bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh

rsync -a --delete .cache/output/mnn/artifacts/jetson/ \
  jetson@192.168.101.192:/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda/
```

Build note:

- The build script still reported stale cross cache with cached compiler/sysroot as `<empty>` and rebuilt `.cache/build/mnn/jetson_cross`.
- This remains a build-time overhead issue, not a runtime blocker.

## First smoke

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_smoke_20260628_0705/summary.csv
x=0 none                  TPOT=86.493 ms
x=1 lagged_attention_hkvd TPOT=99.145 ms
x=3 lagged_attention_hkvd TPOT=126.213 ms
x=5 lagged_attention_hkvd TPOT=127.864 ms
x=7 lagged_attention_hkvd TPOT=127.798 ms
```

Interpretation:

- `x=5/x=7` clearly improved.
- `x=3` regressed in this run, but the source change does not alter batch4 matching relative to the previous accepted rows4/5 cuBLAS path; suspected run noise.

## Second smoke

Same server, same artifact, repeat3:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_cublas_smoke2_20260628_0710/summary.csv
x=0 none                  TPOT=86.377 ms
x=1 lagged_attention_hkvd TPOT=97.471 ms
x=3 lagged_attention_hkvd TPOT=123.602 ms
x=5 lagged_attention_hkvd TPOT=124.975 ms
x=7 lagged_attention_hkvd TPOT=126.360 ms
```

Comparison to previous accepted default:

```text
x=0: 86.443 -> 86.377 ms
x=1: 98.521 -> 97.471 ms
x=3: 123.406 -> 123.602 ms
x=5: 132.097 -> 124.975 ms
x=7: 133.897 -> 126.360 ms
```

Decision:

- Keep the rows4-8 cuBLAS admission.
- It improves high repair counts by about `7 ms` while keeping `x=0/x=1/x=3` neutral within smoke noise.

## Current direction

- Do not revisit full attention-weight capture; compact rank is still the right design and profile says it is small.
- Do not bypass rows2 tiny GEMV; that A/B regressed `x=1`.
- Next useful work is likely:
  - rows2/4 qtile attention specialization beyond existing `hd128_q8k16`,
  - a genuinely better tiny-row weight-only tensor-core/fused dense kernel,
  - or graph-level activation/BinaryOp launch reduction.

## Profile attribution after rows4-8 cuBLAS

Profile command used the accepted rows4-8 cuBLAS artifact with:

```text
MNN_PAGED_ATTENTION_PROFILE=1
MNN_PIC_DECODE_REPAIR_PROFILE=1
```

Profile result:

```text
summary=.cache/mnn-pic-benchmark/decode_repair_rows48_profile_20260628_0715/summary.csv
remote_log=.cache/mnn-pic-benchmark/decode_repair_rows48_profile_20260628_0715/remote/pic_server.log
x=0 none                  TPOT=123.901 ms
x=1 lagged_attention_hkvd TPOT=194.511 ms
x=3 lagged_attention_hkvd TPOT=263.907 ms
x=5 lagged_attention_hkvd TPOT=234.512 ms
x=7 lagged_attention_hkvd TPOT=231.010 ms
```

This run is attribution-only. The profile env forces extra synchronization and should not be compared directly with the non-profile TPOT.

Parsed attribution:

```text
forward median:
rows2 ~= 103.6 ms
rows4 ~= 135.4 ms
rows6 ~= 137.8 ms
rows8 ~= 135.4 ms

qtile attention:
rows2 n=111 sum=94.769ms mean=853.8us med=849us
rows4 n=112 sum=98.368ms mean=878.3us med=874us
rows6 n=110 sum=100.271ms mean=911.6us med=926us
rows8 n=111 sum=104.598ms mean=942.3us med=949us

decode attention rank:
n=16 sum=4.019ms mean=251us med=235.5us

dense totals:
batch2 tiny_gemv n=781 sum=193.100ms med=192us
batch4 CUTLASS 3072->3072 n=223 sum=61.949ms med=272us
batch4 rows45_cublas n=557 sum=332.608ms med=572us, first-step outliers max=73.5ms
batch6 CUTLASS 3072->3072 n=224 sum=62.620ms med=272.5us
batch6 rows45_cublas n=559 sum=235.942ms med=570us
batch8 CUTLASS 3072->3072 n=220 sum=60.661ms med=269us
batch8 rows45_cublas n=555 sum=230.928ms med=564us

small-row runtime_dequant lines: 0
```

Interpretation:

- Rank capture is already cheap enough for the current compact-rank design.
- The accepted rows4-8 cuBLAS route reduced high-row dense cost versus the previous profile, especially rows6/8.
- Remaining gap is mainly tiny-row qtile attention at about `0.85-0.95 ms` per layer plus compact-row dense and launch overhead.
- Rows2 is still dominated by packed tiny GEMV and qtile attention; the earlier rows2 static-CUTLASS bypass is still rejected.
