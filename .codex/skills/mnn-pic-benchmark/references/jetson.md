# Jetson

先读 [common.md](common.md)。

## 角色

- formal device
- backend: CUDA

## 固定路径

- ssh: `jetson@192.168.101.192`
- remote repo/work: `/home/jetson/code/kvshare-edge/impl/MNN`
- artifact: `/home/jetson/code/kvshare-edge/impl/MNN/.cache/output/mnn/artifacts/jetson_cross_cuda`
- shared cache root: `/home/jetson/code/kvshare-edge/impl/MNN/.cache/pic_prefill_latency_sweep`

## 固定端口

- remote port: `18131`
- local tunnel port: `19131`

## Prefill sweep

单设备 debug/profile：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_jetson.sh \
  --model-key minicpm5-1b \
  --benchmark-csv benchmark.csv \
  --only-missing-contexts
```

如果只是查缺口：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_jetson.sh \
  --model-key qwen3-8b \
  --benchmark-csv benchmark.csv \
  --only-missing-contexts \
      --print-gap-only
```

## Power 采集

Jetson 的外部功耗计按用户约定使用：

```text
monitor_type=df
serial=1A5D43
```

单设备 power sweep：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_jetson.sh \
  --model-key minicpm5-1b \
  --benchmark-csv benchmark.csv \
  --power-capture
```

Jetson power 覆盖不跑 `llama3.2-1b` / `Llama3.2 1B`。

输出在 `.codex/skills/mnn-pic-benchmark/power/jetson/<YYYY-MM-DD>/<HH>/<MM>/<process-id>/`，每个 case/rep 用随机 UUID 文件组保存：`<uuid>.csv`、同目录 `<uuid>.png` 和 `<uuid>.txt`，并追加到 `manifest.tsv`。

## Dataset bench

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_mnn_pic_dataset_bench.sh \
  --device-key jetson \
  --run-id <run_id> \
  -- \
  --dataset hotpotqa \
  --mode pic_cache_reuse \
  --phase both
```

## Decode repair

Jetson decode repair 仍然按单设备采集，但 formal 结论要和 OrangePi 各跑一轮。

## 构建入口

构建 artifact 先读：

- `.codex/skills/mnn-build-artifacts/references/jetson.md`
