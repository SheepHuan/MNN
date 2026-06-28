# Context

目标是避免 OrangePi OpenCL benchmark 在 normal baseline、PIC server、不同 context、cacheblend/epic ratio 之间生成多份 `mnn_cachefile.bin`，导致 warm 后正式 run 仍重复编译或重新 tune。

实现口径：

- `run_pic_prefill_latency_sweep.py` 对 OpenCL 设备统一使用 `runtime_cache_dir(device) = <remote_cache_root>/runtime_cache/opencl`。
- normal `llm_bench` 运行时注入 `MNN_LLM_RUNTIME_CACHE_DIR=<runtime_cache_dir>`。
- 为兼容还未同步新二进制的设备，normal 模型目录下的 `tmp` 会绑定成指向统一 runtime cache 目录的 symlink。
- 如果旧 `tmp/mnn_cachefile.bin` 存在且统一 cachefile 尚不存在，先复制旧 cache；随后把旧 `tmp` 目录移动到 `tmp.model_cache_<timestamp>`，不直接删除。
- 普通 LLM `Llm::updateRuntimeCache()` 暴露 `RuntimeManager::updateCache()`，`llm_bench` 每个实例结束后显式写回 cache。
- PIC server 继续使用 `--runtime-cache-dir` 和 `/v1/tune/update_cache` 写回同一 cachefile。

正式测试约束：

- warm 必须执行，不能因为 warm 时间长而跳过首次 LWS 编译/tune。
- 成功数据应立即 merge；不要为了 cache 重新跑已经成功的 rows。
- OrangePi 数据和 cache 都放在 SSD `/mnt/ssd/code/.cache/mnn_opencl_pic`，不要写根分区。
