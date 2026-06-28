# Rhino 2026-06-27 00

Rhino / Adreno 稠密家族调参、mixed K/V storage 探索、sparse-flash 候选和 release 路由修正总记录。

重点：

- 覆盖 compact-dense b2/c4 候选、steady-state tune 选择、公认热点和 bench 脚手架。
- 记录 image/buffer 混合存储、cached-value image shadow、kvimg 候选和 compile/runtime 验证。
- 收敛到 Rhino 当前 release path 的注意事项和默认实验策略。

收录章节：

- `2026-06-27 Experiment: Rhino / Adreno compact dense b2 family bring-up`
- `2026-06-27 Experiment: Adreno tune selection should measure steady state`
- `2026-06-27 Experiment: Rhino / Adreno compact dense c4 family validation`
- `2026-06-27 Experiment: Rhino / Adreno cacheblend top-k dynamic stage family`
- `2026-06-27 Experiment: Rhino / Adreno routing provenance + official guidance`
- `2026-06-27 Validation: Rhino operator-level compact dense hotspot bench at rows=269`
- `2026-06-27 Adreno sparse-attention image-hybrid bench scaffold`
- `2026-06-27 Adreno sparse-flash compile crash narrowing`
- `2026-06-27 Rhino direct-op chain vs Expr chain`
- `2026-06-27 Rhino Expr probe: image vs buffer on hot attention-like shapes`
- `2026-06-27 Rhino raw mixed probe: buffer Q/slot/output + image-or-buffer K/V`
- `2026-06-27 Rhino runtime integration: Adreno-only cacheblend cached-value image shadow`
- `2026-06-27 Rhino runtime integration: Adreno sparse-flash kvimg candidate`
- `2026-06-27 Rhino mixed K/V storage split probe and Adreno key-image sparse-flash candidate`
- `2026-06-27 Rhino runtime validation: mixed `K=image + V=buffer` is not the later-sparse winner`
- `2026-06-27 Rhino Adreno sparse-flash bench bring-up status`
- `2026-06-27 Experiment: Rhino profile/tune visibility and Adreno mixed default`
- `2026-06-27 Experiment: Adreno `headDim=128` direct-value sparse flash + raw hybrid probe`
- `2026-06-27 Rhino compact-dense direction and runtime-cache fix`
- `2026-06-27 Rhino unified run_test.out compact-dense findings`
- `2026-06-27 Experiment: Rhino `adreno_batch_gemv` family via `run_test.out``
- `2026-06-27 Experiment: Rhino single-entry cleanup and compact-dense `rows=269` recheck`
- `2026-06-27 Change: Adreno compact-dense weight storage default`
- `2026-06-27 Change: stabilize Rhino `run_test.out` and narrow the next dense target`
- `2026-06-27 Rhino compact-dense storage correction: Adreno low-memory path wants image, not buffer`
- `2026-06-27 Rhino single-entry bench script and hot-shape recheck`
- `2026-06-27 Rhino status checkpoint: default experiment policy and current effect`
- `2026-06-27 Change: Rhino attention routing correction in release path`

文件：

- `README.md`: 简短摘要。
- `context.md`: 完整长日志。
