# OrangePi Decode-Repair x0 Hidden-Cost Pass

## Summary

- Built and synced OrangePi artifact with explicit `MNN_PAGED_ATTENTION_DECODE_REPAIR_Q1_ROW=1` A/B support.
- Profile smoke was P0 clean: no `ordinary_q1=1`, no `decode_prepare_inside_decode=1`, no `ERROR` / cache invalid lines. Smoke confirmed `pic_decode_repair_decode_graph=1`; x0/query=1 used `q1_row=1`, x1/query=2 used `q1_row=0`.
- Formal profile-off run:
  - `decode_repair_q1row_orangepi_ctx512_1024_pic_20260704_110137`
  - 3 models x 2 contexts x x=0/1/3/5/7, repeats=2, warm=1, PIC-only.
- q1-row x0 improved over default decodegraph on all 6 model/context pairs, average `-7.660 ms`.
- q1-row is still not enough to declare x0 fully solved: Qwen3-4B ctx1024 formal x0 remained above true normal x0, though light profile showed steady-state forwardRaw closer than the formal aggregate.

## Next

- Keep q1-row as explicit A/B until the team decides whether single-row sparse-kernel specialization is acceptable as same-family default.
- Continue with x0 hidden-cost split:
  - attention append/attention fixed cost,
  - graph-level 1-row Raster / BinaryOp / LayerNorm launch overhead,
  - request-wall sample / suffix-prefill overhead separately from `decode_tpot_ms`.
