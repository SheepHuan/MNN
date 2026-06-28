# 2026-06-28 04:00 OrangePi OpenCL Runtime Cache Unification

本小时记录 normal `llm_bench` 与 PIC server 共用设备级 MNN OpenCL runtime/autotune cache 的约束和实现。

## 结论

- OpenCL runtime cache 固定为设备级 `<remote_cache_root>/runtime_cache/opencl/mnn_cachefile.bin`。
- normal baseline 不再使用模型目录私有 `tmp/mnn_cachefile.bin`；通过 `MNN_LLM_RUNTIME_CACHE_DIR` 和兼容 symlink 落到同一文件。
- warm 阶段允许首次 kernel build / LWS tuning，并在正式计时前写回 cache，后续 context / ratio / mode 增量复用。
