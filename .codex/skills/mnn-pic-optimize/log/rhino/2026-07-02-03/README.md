# Rhino 2026-07-02 03 — x0 decode 定位校正

## 摘要

本小时校正 `RHINOPI_DECODE_REPAIR_HANDOFF.md` 和 2026-07-02-01 日志中两个过期方向：

1. x0 batch=1 Conv 当前没有证据显示走 `family=fp_weight`。源码中 batch=1 直接走
   `tuneGemvLowMemory`，现有 x0 日志的 batch=1 Conv 均为 `use_fp_weight=0`。
2. `pic_decode_model` / `pic_decode_weight` 辅助 normal decode route 已从当前源码消失，
   不应作为修复方向恢复。
3. 当前 x0 慢因定位转为：PIC graph-boundary identity 场景仍支付
   Gather/Cast/Shape/While/Raster/layout 税，加上 PagedAttention identity decode 和
   normal-vs-PIC 计时口径差异。
4. 图级来源已闭合：`pic_recompute_budget` 让 score layer 导出 2-output
   `PicScoreAttention`，其 `active_indices` 即使在 x0 为 identity，也继续驱动每层
   `_pic_gather_rows` / Cast / GatherV2 / Shape lowering。当前真实图不是
   “normal decode 只替换 PagedAttention KV 存储”。

当前建议 P0：

- 对齐 normal 和 PIC 的计时/attribution 口径。
- 对 `pic_recompute_budget=seq_len=1` 且 `active_indices` identity 的 x0 图路径做
  identity noop/fold，去掉冗余 gather/shape/cast。
- 优先消除 24 层 PagedAttention 后的 Raster/layout 转换税。
- x0 根因定位已完成；剩余是正式同口径报告和优化 patch。

详见 `context.md`。
