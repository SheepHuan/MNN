# OrangePi PIC Decode-Repair q1-row Default

## Summary

- Promoted `MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_ROW` to default-on for OpenCL decode-repair.
- Semantics stay inside the same PIC dualgraph decode-repair family: `x=0` still runs `mnn_token_id_sparse_decode` with active rows=1, not ordinary q=1 LLM decode.
- Built and synced OrangePi artifact, then ran profile-off measure for MiniCPM5-1B, Llama3.2-3B and Qwen3-4B at ctx512/1024 with `x=0/1/3/5/7`.
- P0 scan was clean: no `ERROR`, `target unavailable`, `Cache invalid`, `async persistent PIC cache read failed`, `decode_prepare_inside_decode=1`, or `ordinary_q1=1`.

## Measure Result

Run ID:

```text
decode_repair_q1row_default_orangepi_ctx512_1024_pic_20260704_122350
```

Formal table uses measure rows only:

| model | ctx | true normal x0 | PIC x0 | gap | x1 extra | x3 extra | x5 extra | x7 extra |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama3.2 3B | 512 | 121.866 | 113.256 | -8.610 | 63.688 | 75.384 | 230.271 | 237.997 |
| Llama3.2 3B | 1024 | 126.138 | 141.018 | +14.880 | 57.241 | 72.232 | 245.423 | 256.907 |
| MiniCPM5-1B | 512 | 46.760 | 42.697 | -4.063 | 33.078 | 40.779 | 91.923 | 96.105 |
| MiniCPM5-1B | 1024 | 49.809 | 42.995 | -6.814 | 53.559 | 65.558 | 131.528 | 132.054 |
| Qwen3-4B | 512 | 149.916 | 152.885 | +2.969 | 81.542 | 93.156 | 298.131 | 304.218 |
| Qwen3-4B | 1024 | 155.065 | 171.296 | +16.231 | 81.640 | 96.535 | 326.898 | 342.175 |

## Current Read

- `PIC x0` is now close to true normal x0 on 4/6 cases, with remaining slow cases at long context: Llama ctx1024 `+14.880 ms`, Qwen ctx1024 `+16.231 ms`.
- `sample_tpot_ms` is stable around 12-16 ms and ArgMax A/B did not materially reduce it; the remaining sampler cost is likely logits materialization/sync or lm_head top1 path, not ArgMax local size.
- `x5/x7` are still the obvious broken slope. The next work should optimize hidden row-count amplification in the same family, not add an x-specific fallback.

## Next Direction

1. Use decode-only graph profile and PA detail on Qwen ctx1024 x0 to split the remaining +16 ms into PagedAttention, Raster, Convolution, Binary/Unary/LayerNorm/While and sampler.
2. Make active-row metadata visible to more post-attention graph ops so decode-repair rows do not re-expand through Raster/While/elementwise.
3. A/B `fused_append` and record-queue/fixed-dispatch as explicit variants for all `x=0/1/3/5/7`; promote only if they reduce the same-family curve without routing by x/model.
4. Investigate fused lm_head top1/topk as a larger sampler fix; current `_ArgMax` only avoids full host logits read but still pays roughly 12-16 ms/sample.
