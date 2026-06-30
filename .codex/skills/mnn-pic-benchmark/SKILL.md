---
name: mnn-pic-benchmark
description: 当用户要求把本 MNN 仓库本地构建或交叉编译出的 pic_server / libpic_llm / CUDA / OpenCL artifact 推送到 Jetson、Orange Pi 5 Plus 或 Rhino Pi-X1/Aidlux Adreno 设备上运行，并从本机发起 PIC benchmark、prefill latency sweep、decode repair benchmark、dataset bench、benchmark.csv 合并或缺口补测时使用；也用于区分 formal 设备矩阵与 Rhino extra-profile 流程、固定端口清理旧进程、共享 text cache 和每设备独立测试方案。
---

# MNN PIC Benchmark

本 skill 只保留通用规则和设备分流。每个设备的测试方案、路径、端口、频率和脚本入口分别写在 `references/`。

## 相关 skills

- 构建 artifact：读 `.codex/skills/mnn-build-artifacts/SKILL.md`，并继续读对应设备 reference
- 设备 rsync / smoke：按需读 `.codex/skills/mnn-opt-ops/SKILL.md`
- 输出判定：按需读 `.codex/skills/mnn-llm-bench/SKILL.md`

## 通用硬约束

1. 只允许本地构建或交叉编译，再把 artifact 推到设备；不要把源码推到设备上重新编译。
2. 启服务前直接 kill 旧端口监听进程；不要靠反复换端口避开脏进程。
3. `/v1/prefill/text` 持久 text cache 必须走 shared KV cache 路径；不要按 run-id 重建新目录。
4. `cacheblend` / `epic` / `kvshare` 每个 ratio 都必须发独立请求，不能跨 ratio 复用 scoring/top-k 结果。
5. 正式 `prefill` / `decode` formal 设备矩阵默认是 `jetson + orangepi`；`rhino` 只属于 extra-profile。
6. 正式默认上下文是 `512,1024,1536,2048,2560`；`3072` 只有显式接受风险时才单独打开。
7. 正式 OpenCL 结果必须先 warm 并写回 autotune cache；冷启动 kernel build/LWS tuning 不计入正式 latency。
8. `summary.csv` 只合并成功行；失败、超时、500 进入 failure/raw 日志，不直接写入 `benchmark.csv`。
9. 不再把 PIC full-compute / `pic-full-recompute` 作为补测目标；默认 prefill sweep 和缺口报告只覆盖 `normal-full-recompute`、`full-reuse`、`cacheblend`、`epic`。历史 CSV 中已有的 `pic-full-recompute` 行保留，但后续不要为了补齐它重跑。
10. PIC 功耗采集走本 skill 下的 power 流程；只包正式 measure 请求，不把 artifact 同步、cache build、warm、OpenCL tune 写回或手动等待计入 power CSV。
11. PIC 功耗测试不跑 `Llama3.2 1B` / `llama3.2-1b`；从 `benchmark.csv` 选 power 配置时直接排除它。

## Power 采集

优先使用 `scripts/run_pic_prefill_latency_sweep.py --power-capture`，它会为每个实际测量 row 单独启动 `POST /v1/power/start`、先 idle sleep 5 秒作为前置 base window、连续执行 `--power-rep` 次正式 PIC/normal measure、最后一次 rep 后再 idle sleep 5 秒作为尾部回落 window，然后 `POST /v1/power/stop` 并下载一个 CSV。Power 模式默认 `--power-rep 3`，即每个 benchmark case 在同一个 power CSV 内连续执行 3 次；每次执行之间固定间隔 3 秒。前后 idle 时长可用 `--power-warmup-sec` / `--power-cooldown-sec` 或 `MNN_POWER_WARMUP_SEC` / `MNN_POWER_COOLDOWN_SEC` 覆盖。

默认 Power API：

```text
MNN_POWER_API_URL=http://192.168.101.14:8766
jetson:        df  serial 1A5D43
rhino/rhinopi: df  serial 1A5D43
orangepi-mali: blu serial F96FBDBA05B0
```

输出目录按测量进程分流；不要为每个 CSV 单独建 case 子目录：

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

Power API 只返回 CSV；每个 `<uuid>.csv` 下载到本机后，必须由本仓库本地脚本 `scripts/render_power_csv.py` 在同目录渲染 `<uuid>.png`，不要等待远端返回 PNG。`<uuid>.txt` 至少必须包含 device、backend、frequency、run_id、model、context_tokens、algorithm、budget、rep total、rep interval、每次执行的结果、pic_recompute_ratio、score_layer、Power API 请求、monitor、session/run id、CSV 路径和 PNG 路径。

Power CSV 渲染和能耗积分脚本也在本 skill 下：

```bash
.codex/skills/mnn-pic-benchmark/scripts/render_power_csv.sh <power.csv>
.codex/skills/mnn-pic-benchmark/scripts/summarize_power_energy.sh --trim-sec 0 <power-dir>
.codex/skills/mnn-pic-benchmark/scripts/summarize_pic_power_reps.py \
  .codex/skills/mnn-pic-benchmark/power/orangepi/<YYYY-MM-DD> \
  .codex/skills/mnn-pic-benchmark/power/rhino/<YYYY-MM-DD> \
  -o .codex/skills/mnn-pic-benchmark/power/middle_rep_energy_<YYYY-MM-DD>.csv \
  --plot-output .codex/skills/mnn-pic-benchmark/power/middle_rep_energy_<YYYY-MM-DD>.png
```

`summarize_pic_power_reps.py` 是最终批量汇总脚本，不挂到采集流程里，也不要每生成一个 CSV 就跑一次。它一次接收一个或多个 power 根目录 / 日期目录 / `manifest.tsv`，递归发现每个测量进程目录里的 `manifest.tsv`，最后生成一个汇总 CSV，每个 benchmark case 一行。它专门用于 `--power-rep` 连续多次采集后的正式能耗口径：默认丢掉第 1 次和最后 1 次 rep，只保留中间 rep；从 power CSV 曲线识别 active rep window；用 rep window 外的 idle 样本识别 base power；对选中 rep 的 `max(power - base_power, 0)` 积分，输出每次推理增量能耗和按 `context_tokens` 归一的 `mJ/token`。带 `--plot-output` 时还会输出一个总览 PNG：每个子图对应一个 case，显示完整 power 曲线，红色虚线/浅红区间标记被丢弃的首尾 rep，绿色虚线/浅绿区间标记用于能耗计算的中间 rep。如果 CSV 全 0 或 active window 数不足，会在 `analysis_status` / `error` 中标出，不生成伪有效能耗。

独立包装任意命令时使用：

```bash
MNN_POWER_DEVICE=rhino \
MNN_POWER_OUTPUT_CSV=.codex/skills/mnn-pic-benchmark/power/manual/rhino/power.csv \
.codex/skills/mnn-pic-benchmark/scripts/pic_power_capture.sh -- <command>
```

## 分流

先读通用规则：

- [references/common.md](references/common.md)

再按设备只读对应文档：

- Jetson： [references/jetson.md](references/jetson.md)
- Orange Pi 5 Plus： [references/orangepi5plus.md](references/orangepi5plus.md)
- Rhino Pi-X1 / Adreno： [references/rhino-adreno.md](references/rhino-adreno.md)

如果任务是改 formal 双设备流程，再额外读：

- [references/formal-matrix.md](references/formal-matrix.md)

## 推荐脚本入口

通用主脚本保留：

- `scripts/run_pic_prefill_latency_sweep.py`
- `scripts/run_mnn_pic_dataset_bench.sh`
- `scripts/run_pic_decode_repair_benchmark.py`
- `scripts/merge_prefill_benchmark_csv.py`
- `scripts/pic_power_capture.sh`
- `scripts/render_power_csv.sh`
- `scripts/summarize_power_energy.sh`
- `scripts/summarize_pic_power_reps.py`

设备预设 wrapper：

- `scripts/run_pic_prefill_latency_sweep_formal.sh`
- `scripts/run_pic_prefill_latency_sweep_jetson.sh`
- `scripts/run_pic_prefill_latency_sweep_orangepi.sh`
- `scripts/run_pic_prefill_latency_sweep_rhino.sh`

这些 wrapper 只负责设备 preset 和 formal/debug 口径，不替代主脚本能力。

## 编辑后校验

至少跑：

```bash
bash -n .codex/skills/mnn-pic-benchmark/scripts/run_mnn_pic_dataset_bench.sh
bash -n .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_formal.sh
bash -n .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_jetson.sh
bash -n .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_orangepi.sh
bash -n .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_rhino.sh
bash -n .codex/skills/mnn-pic-benchmark/scripts/pic_power_capture.sh
python3 .codex/skills/.system/skill-creator/scripts/quick_validate.py .codex/skills/mnn-pic-benchmark
```
