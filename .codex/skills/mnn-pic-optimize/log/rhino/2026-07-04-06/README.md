# Rhino 2026-07-04 06 — native full-reuse warm 重启与修复

## 摘要

本小时定位并修复 Rhino Pi-X1 / Adreno OpenCL 在新 native prefill 实现下的 `MiniCPM5-1B / ctx=512` 失败：

1. Rhino 当前在线，主机 `kalama`，但本次失败后设备发生过重启。`last -x` 显示 `2026-07-04 14:43 CST` boot，失败窗口之后 `ping/ssh` 一度不可达。
2. normal OpenCL baseline 成功：`ctx=512 normal-full-recompute = 1.081499178s`，`473.4169 tok/s`。`/v1/prefill/text` 也成功并复用 text cache，latency `0.0346s`。
3. 第一条 PIC chat warm 是 `full-reuse`，等待 `585.637s` 后返回 `curl: (52) Empty reply from server`。后续 cacheblend/epic 全部只是 `127.0.0.1:19133 Connection refused`，没有真正进入算法路径。
4. 远端 `pic_server_ctx512.log` 停在 `Starting PIC server...` 和 `Warning: module need new clone, cloning now.`；这与 2026-07-03 slot-table removal 后的 Rhino full-reuse smoke 失败形态一致。
5. 结论：这不是 cacheblend/epic 的数值性能退化，而是 native full-reuse hydrate+suffix 路径在 Rhino/Adreno 上的稳定性 gate。
6. 修复：`prefillFullReuseExternalPagedKV()` 用 `mForcePrefillForward` 作用域标志包住 active-row `prefill()`，`forwardRaw()` 在该标志下不再因为 decode-ish 状态选择 14-token module key，而是复用 prefill graph key。
7. 验证 run `fullreuse_rhino_fix_ctx512_20260704_153815` 成功：warm/measure 都 HTTP 200；measure `full-reuse ctx512 = 0.573422s`，`892.885 tok/s`；远端 log 无 module clone warning，显示 `seq_len=14 seq_len_key=100 force_prefill_forward=1`，24 层 hydrate 完成，Rhino 仍在线。
8. 同 artifact 上 `sparse_rhino_fix_ctx512_20260704_154150` 低 ratio smoke 成功：`cacheblend 5% = 0.374792s`、`epic 5% = 0.340086s`，均 HTTP 200，Rhino 未重启。该结果只说明崩溃 gate 已解除，不替代完整 ratio sweep。
9. 回归测试只用于和当前 `benchmark.csv` 对比，不覆盖/合并 benchmark。`ctx512/1024` 的 PIC prefill 算法没有系统性变慢；唯一初测慢点 `ctx512 epic 40%` 复测为 `0.718947s`，相对 benchmark `+0.9%`，判为噪声。

详见 `context.md`。

## 回归处理顺序

1. 不要直接跑全量 sweep；先只跑 `full-reuse` ctx512 smoke。
2. 打开 `MNN_PIC_REQUEST_PROFILE=1`、`MNN_PAGED_ATTENTION_PROFILE=1`、`MNN_PAGED_ATTENTION_PROFILE_DETAIL=1` 和必要时 `MNN_PIC_GRAPH_PROFILE=1`。
3. 若日志仍停在 module clone，先确认 `forward_raw_select_module` 是否带 `force_prefill_forward=1` 且 `seq_len_key=100`。
4. 若进入 layer hydrate 后重启，用 `MNN_PAGED_ATTENTION_OPENCL_ALLOW_KV_STAGING_FALLBACK=1` 只做 A/B 诊断，判断 Adreno direct source-slot value hydrate / inplace hydrate kernel 是否触发。
5. full-reuse 稳定前，不合并本次 cacheblend/epic failure 到 `benchmark.csv`，也不要报告为 sparse prefill 性能退化。
