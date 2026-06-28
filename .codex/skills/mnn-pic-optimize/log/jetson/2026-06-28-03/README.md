# Jetson CUDA decode repair tiny-row optimization

## Summary

- This hour continues the Jetson CUDA decode repair optimization after fused compact attention-rank capture landed.
- Current required baseline remains `x=0 none` normal decode TPOT.
- Current small repair budgets remain `x=1/3/5/7 lagged_attention_hkvd`.
- The latest non-profile smoke before this hour was:

```text
x=0 none                  TPOT=87.163 ms  status=ok
x=1 lagged_attention_hkvd TPOT=158.793 ms status=ok
x=3 lagged_attention_hkvd TPOT=185.317 ms status=ok
x=5 lagged_attention_hkvd TPOT=184.994 ms status=ok
x=7 lagged_attention_hkvd TPOT=207.958 ms status=ok
```

## Current Conclusion

- Rank capture has already been fused into the row-compressed decode attention path and is no longer the first optimization target.
- Attention rank capture remains restricted to the configured layer.
- qtile stays default-off for `x<=7`; it is only an experiment for larger decode repair row counts or batch-like shapes.
- `MNN_PIC_DECODE_REPAIR_PROFILE=1` now gives low-overhead per-step runtime attribution outside graph profile.
- Runtime attribution shows CPU-side decode repair overhead is not the current bottleneck: after step 0, non-`forwardRaw` cost is about `0.11-0.12 ms/step`.
- Two compute experiments were tested and reverted because they regressed TPOT:
  - row-compressed attention exp reuse inside V accumulation.
  - extending rows4/5 cublas to Llama3.2-3B MLP shapes.
- Next useful work should target graph/kernel compute inside `forwardRaw`, especially tiny-row Linear/MLP and graph fragments, not selector/rank CPU overhead.

## Work Items For This Hour

1. Add low-overhead decode repair timers gated by an env var so we can split selector/runtime overhead from `forwardRaw` without graph-profile synchronization.
2. Inspect the CUDA row-compressed decode attention path for `attnLen<=8` overhead that can be skipped for repair rows.
3. Inspect tiny-row weight-only Linear/MLP execution and identify the lowest-risk profiling or fast-path hook for `M=2/4/6/8`.
4. Rebuild and run Jetson smoke only after code changes compile cleanly.

## Runs

```text
decode_repair_runtime_profile_20260628_031546
  x=0 TPOT=100.419 ms
  x=1 TPOT=130.609 ms
  x=3 TPOT=202.247 ms
  x=5 TPOT=214.524 ms
  x=7 TPOT=236.831 ms

decode_repair_attention_exp_reuseexp_20260628_032040
  result: regressed; experiment reverted

decode_repair_conv_profile_20260628_032440
  purpose: CUDAWeightOnlyConv tiny-row attribution

decode_repair_cublas3b_default_20260628_032751
  result: regressed; experiment reverted

decode_repair_mlp_20260628_0342
  build: separate Jetson cross testbench run_test.out, MNN_BUILD_TEST=ON
  result: direct-op benches show MLP/Linear compute is the dominant x>0 target
  PicDecodeMlp hot repeat:
    rows=2 chain=0.965 ms
    rows=4 chain=1.744 ms
    rows=6 chain=1.723 ms
    rows=8 chain=1.746 ms
  A/B: raising decode-repair INT4 GEMV limit from 3 to 6 regressed rows4/6; reverted
  A/B: raising V14_MB OC_PER_BLK from 4 to 8 regressed rows2; reverted
```

## Continuation

- `2026-06-28 03:38 CST`: Continue from the runtime profile conclusion. The next step is a separate Jetson cross testbench build with `MNN_BUILD_TEST=ON`, so direct CUDA op benches can compare the current weight-only MLP path against GEMM floor baselines for Llama3.2-3B tiny rows `2/4/6/8`.
- Keep `x=0` in every decode TPOT smoke; do not merge smoke-only decode numbers into formal CSV.

## Updated Direction

- Do not expand decode-repair V14_MB GEMV to rows4/6 as default; it regressed direct `PicDecodeMlp`.
- Plain FP16 cublas GEMM floor is not better for rows2/4/6, so the reverted rows4/5 cublas extension should stay reverted.
- The next viable optimization is graph/MLP structure level: reduce the number or overhead of tiny-row Linear launches, or introduce a proven fused gate/up/SwiGLU/down path. Selector/rank CPU work is already below `0.2 ms/step`.

## 03:58 CST Continuation

- Continue optimizing Jetson CUDA decode repair `x=1/3/5/7` against the required `x=0 none` TPOT baseline.
- Current measured overhead outside `forwardRaw` is already negligible, so the next investigation is inside the graph/kernel work.
- First check whether decode repair still runs `lm_head` / logits materialization for all sparse repair rows. If `logits_index=-1` already slices to the last row before `lm_head`, do not spend time on this path.
- If `lm_head` is already limited to one row, use graph/profile evidence to target tiny-row graph fragments around MLP/Linear, residual/norm/BinaryOp, and output materialization.
