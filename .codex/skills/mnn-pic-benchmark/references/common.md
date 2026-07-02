# Common Rules

## 设备角色

- formal devices: `jetson`, `orangepi`
- extra-profile devices: `rhino`

Rhino 不能替代 formal 双设备回归。要把 Rhino 加进 sweep，必须显式 `--allow-extra-device`。

## 共享 text cache

同一模型、设备、context、token span 的 `/v1/prefill/text` 持久 text cache 要复用固定 shared KV 路径：

- Jetson: `/home/jetson/code/kvshare-edge/impl/MNN/.cache/pic_prefill_latency_sweep/shared_kv/<model-key>/`
- OrangePi: `/mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep/shared_kv/<model-key>/`
- Rhino: `/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/shared_kv/<model-key>/`

OpenCL cache 路径按设备固定为单一路径，不要为同一设备设置多个 cache 根目录，也不要在正式 benchmark 中临时切换 cache path。

MNN OpenCL runtime/autotune cache 要和 text KV cache 分开：同一设备只使用固定的 `<remote_cache_root>/runtime_cache/opencl/mnn_cachefile.bin`，不要放到 `shared_kv/<model>/runtime_cache/` 这种按模型分叉的目录。第一次遇到未覆盖的新 shape 时，warm 阶段必须允许 OpenCL 编译 program、尝试 local work size 候选并生成 cache。warm 成功后必须调用 `/v1/tune/update_cache` 写回这个设备级 cachefile，后续同设备、同 driver、同 artifact 的 PIC server 运行应加载并复用它。

普通 `llm_bench` baseline 也必须复用同一个设备级 `mnn_cachefile.bin`，不能再使用各模型目录自己的 `tmp/mnn_cachefile.bin` 作为独立 cache。prefill sweep 脚本必须给 normal bench 注入 `MNN_LLM_RUNTIME_CACHE_DIR=<remote_cache_root>/runtime_cache/opencl`，并为兼容旧二进制把 normal 模型目录下的 `tmp` 绑定到同一 runtime cache 目录；已有 model-local cache 只能迁移/备份，不能删除。

不要把 run-id 写进 text cache id 或 KV cache 目录。

## Prefill formal 口径

- 默认 contexts: `512,1024,1536,2048,2560`
- `3072` 不是默认上下文
- 正式端口固定，启动前直接清理旧进程
- OpenCL 正式计时前必须 warm 并 `/v1/tune/update_cache`
- PIC full-compute / `pic-full-recompute` 不再作为测试缺口；补测只关注 normal baseline、full-reuse、cacheblend 和 epic

正式 prefill sweep 通用主入口：

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep.py
```

## Dataset / Decode

单设备 dataset helper：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_mnn_pic_dataset_bench.sh
```

decode repair 主入口：

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/run_pic_decode_repair_benchmark.py
```

Decode TPOT 正式表格改为双族对比：

- `optimized-decode-repair`：使用当前要验证的优化导出 / server，测 `x=0/1/3/5/7`。
- `normal/generic-decode-repair`：使用同一模型、同一 prompt/token、同一 selector、同一 context/max_tokens/repeats，但导出时关闭后端专属 decode-repair fusion，例如 `--pic_decode_fusion_backend generic`；同样测 `x=0/1/3/5/7`。
- `normal-llm-x0`：可额外用普通 normal export `.cache/mnn-llm-export/<model>/config.json` 跑普通 MNN decode，作为绝对 baseline；普通 LLM 没有 `x=1/3/5/7` 的 decode-repair workload，不能把这 4 列伪造出来。
- `repair_tokens=0` / `tpd=0` 是各自族内的 no-repair 诊断行，列名可以是 `x0`，但必须带上族名，不能把 `optimized x0`、`generic x0` 和 `normal-llm-x0` 混成同一列。
- `dualgraph` 只用于 PIC/PagedAttention LLM：模型目录必须来自 `.cache/weight/<model>/`，运行入口是 `pic_llm` / `pic_server`，语义是 PIC prefill graph 和 PIC decode graph 分离。`true normal` 必须来自 `.cache/mnn-llm-export/<model>/` 并走普通 normal 计算图；不要用 `llm_bench` / `llm_demo` 把 dualgraph 产物跑成 normal baseline，也不要把 dualgraph 产物、dualgraph 日志或 PIC no-repair 行写成 `normal-llm-x0`。
- 报告 speedup 时优先按同一 `x` 对比：`speedup_opt_vs_normal_same_x = normal_tpot[x] / optimized_tpot[x]`。同时报告族内增量：`extra_ms[x] = tpot[x] - tpot[0]`。
- OpenCL 两个族都要使用同一个设备级 runtime/autotune cache，正式计时前分别 warm 目标 context / generated token / repair-token shape 并写回 cache；kernel build、LWS tuning、旧 cache rebuild 不计入正式 TPOT。

推荐表格：

```text
device,model,ctx,family,x0,x1,x3,x5,x7
device,model,ctx,optimized-decode-repair,...
device,model,ctx,normal/generic-decode-repair,...
device,model,ctx,normal-llm-x0,<value>,,,,
```

推荐派生表：

```text
device,model,ctx,x,optimized_tpot_ms,normal_tpot_ms,speedup_opt_vs_normal_same_x,optimized_extra_ms,normal_extra_ms
```

formal dataset/decode 结论仍要求 Jetson 和 OrangePi 各自独立覆盖一轮；Rhino 只记 profile。

## benchmark.csv 合并

正式结果合并：

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/merge_prefill_benchmark_csv.py \
  benchmark.csv \
  .cache/latency_budget_20260625/<run_id>/summary.csv
```

单设备 debug/profile 合并必须显式：

```bash
python3 .codex/skills/mnn-pic-benchmark/scripts/merge_prefill_benchmark_csv.py \
  --allow-nonformal \
  benchmark.csv \
  .cache/latency_budget_20260625/<run_id>/summary.csv
```

## Power 采集

PIC power 采集用 eperf Power API，不走 LLM bench skill 的脚本。默认 API：

```text
http://192.168.101.14:8766
```

设备映射：

```text
jetson:         monitor_type=df   serial=1A5D43
rhino/rhinopi:  monitor_type=df   serial=1A5D43
orangepi-mali:  device=orangepi5plus / monitor_type=blu serial=F96FBDBA05B0
```

正式 power 数据只采集每个 benchmark row 的 measure 窗口：不包含 artifact 推送、服务启动、`/v1/prefill/text` text cache 构建、warm 请求、OpenCL kernel build/tune 或 `/v1/tune/update_cache`。Power 模式默认 `--power-rep 3`，每个 benchmark case 只启动一次 Power API capture、生成一个 CSV，并在这个 CSV 内连续执行 3 次测量；每次执行之间固定间隔 3 秒。Power 模式下 normal baseline 的内部 `llm_bench -rep` 固定为 1，避免外层 rep 和 llm_bench 内部 rep 叠乘。Power API `max_duration_sec` 默认 1800 秒，只作为兜底，正常仍在 benchmark case 结束后立即 stop。采集输出必须放在本 skill 的 `power/` 子树下，按测量进程建目录；不要为每个 CSV 单独建 case 子目录：

```text
.codex/skills/mnn-pic-benchmark/power/<device>/<YYYY-MM-DD>/<HH>/<MM>/<process-id>/
  manifest.tsv
  <uuid>.csv
  <uuid>.png
  <uuid>.txt
  <uuid>.start.json
  <uuid>.start_response.json
  <uuid>.stop.json
  <uuid>.stop_response.json
```

功耗测试不跑 `Llama3.2 1B` / `llama3.2-1b`；从 `benchmark.csv` 规划 Rhino、Jetson 或 OrangePi-Mali power 覆盖时直接排除这个模型。

`<uuid>.png` 不是 Power API 或远端设备返回的 artifact。流程必须是：本机从 `power.stop` 的 `artifacts.csv.raw_url` 下载 `<uuid>.csv` 后，立即在本机调用 `.codex/skills/mnn-pic-benchmark/scripts/render_power_csv.py` 渲染同目录 `<uuid>.png`。

`<uuid>.txt` 至少记录：

```text
device
backend
frequency_profile / frequency_note
run_id
power_uuid
power_rep_total
power_rep_interval_sec
power_repetition_results
model_key / model
context_tokens
algorithm
budget
pic_recompute_ratio
pic_recompute_score_layer_idx
power monitor type / serial
<uuid>.csv path
<uuid>.png path
```

单设备 sweep 加 power：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_rhino.sh \
  --model-key minicpm5-1b \
  --power-capture
```

已有 CSV 只用于选择要覆盖的配置时，按 CSV 中的 model/context/mode/budget 分组调用 sweep；不要把 `--only-missing-contexts` 用在 power 覆盖，因为已有 benchmark 行不是缺口。

## 本地构建约束

构建 artifact 时只走本地 build 或交叉编译，再 rsync artifact 到设备。构建入口按设备去读 `mnn-build-artifacts` 对应 reference，不要把设备源码拿去远端编译。
