# Orange Pi 5 Plus

先读 [common.md](common.md)。

## 角色

- formal device
- backend: OpenCL / Mali

## 固定路径

- ssh: `orangepi@192.168.101.113`
- remote work: `/mnt/ssd/code/.cache/mnn_opencl_pic`
- remote repo/work root: `/mnt/ssd/code/.cache/mnn_opencl_pic`
- artifact: `/mnt/ssd/code/.cache/mnn_opencl_pic/artifacts/orangepi5plus`
- shared cache root: `/mnt/ssd/code/.cache/mnn_opencl_pic/pic_prefill_latency_sweep`

OrangePi 根分区 `/home/orangepi/...` 空间紧张，不作为 benchmark 数据目录。设备侧 artifact、OpenCL runtime cache、shared KV、raw log、dataset 输出和临时 run 目录都必须放在 SSD `/mnt/ssd/code/.cache/mnn_opencl_pic` 下。

## 固定端口

- remote port: `18132`
- local tunnel port: `19132`

## Prefill sweep

单设备 debug/profile：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_orangepi.sh \
  --model-key llama3.2-3b \
  --benchmark-csv benchmark.csv \
  --only-missing-contexts
```

OrangePi OpenCL 正式结果默认需要 warm，并由脚本写回 autotune cache。
`--frequency-profile max` 会在正式 benchmark 前锁定 CPU policy、Mali GPU
`/sys/class/devfreq/fb000000.gpu` 和 DDR DMC `/sys/class/devfreq/dmc` 到最高档；
默认使用 `root@192.168.101.113` 执行 sysfs 写入。如果 root SSH 不可用，需要启用
passwordless sudo 或设置 `MNN_ORANGEPI_SUDO_PASSWORD`。锁频校验失败时脚本必须直接退出，
不能把未锁频结果写成 `cpu=max,gpu=max,ddr=max`。

OrangePi 补测不再跑 PIC full-compute / `pic-full-recompute`。Qwen3-8B 在 ctx1536 的 PIC full-compute 曾触发全局 OOM，后续缺口补测只跑 normal baseline、full-reuse、cacheblend 和 epic。

## Power 采集

OrangePi 5 Plus / Mali 的外部功耗计使用 eperf API 设备绑定：

```text
device=orangepi5plus
monitor_type=blu
serial=F96FBDBA05B0
sample_rate_hz=10000
voltage_mv=4800
```

单设备 power sweep：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_orangepi.sh \
  --model-key llama3.2-3b \
  --benchmark-csv benchmark.csv \
  --power-capture
```

OrangePi-Mali power 覆盖不跑 `llama3.2-1b` / `Llama3.2 1B`。

输出在 `.codex/skills/mnn-pic-benchmark/power/orangepi/<YYYY-MM-DD>/<HH>/<MM>/<process-id>/`，每个 case/rep 用随机 UUID 文件组保存：`<uuid>.csv`、同目录 `<uuid>.png` 和 `<uuid>.txt`，并追加到 `manifest.tsv`。

## Dataset bench

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_mnn_pic_dataset_bench.sh \
  --device-key orangepi \
  --run-id <run_id> \
  -- \
  --dataset hotpotqa \
  --mode pic_cache_reuse \
  --phase both
```

## Decode repair

OrangePi decode repair 是 formal 设备的一半，不能被 Rhino 替代。

## 构建入口

构建 artifact 先读：

- `.codex/skills/mnn-build-artifacts/references/orangepi5plus.md`
