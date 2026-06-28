# Rhino 2026-06-27 17

Rhino / Adreno OpenCL warmup tune 设备族隔离记录。

重点：

- warm 时间长的直接原因不是正式请求退化，而是首轮会触发 kernel build、LWS tune、PIC sparse family / schedule / variant 选择和候选 steady-state 计时。
- 固定 OpenCL cache 根路径不变，但 PIC 手写 tune key 必须区分 `mali` / `adreno` 等设备族，避免 OrangePi/Mali 与 Rhino/Adreno 互相回放路径选择。
- 本次代码把五类 PIC tune key 加入设备族 namespace：`score_sparse_family`、`sparse_flash_schedule`、`sparse_flash_variant`、`sparse_qsplit_chunk`、`cacheblend_topk_family`。

文件：

- `README.md`: 简短摘要。
- `context.md`: 长上下文和代码结论。
