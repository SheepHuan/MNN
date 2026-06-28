# Formal Matrix

## Prefill formal wrapper

正式双设备 prefill sweep：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_formal.sh \
  --model-key minicpm5-1b \
  --benchmark-csv benchmark.csv \
  --only-missing-contexts
```

这个 wrapper 固定：

- `--devices jetson,orangepi`
- `--remote-memory-limit-percent 95`
- formal 默认 contexts：`512,1024,1536,2048,2560`

## Dataset formal

dataset formal 继续用：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_formal_mnn_pic_dataset_bench.sh
```

## 合并规则

formal summary 合并到 `benchmark.csv` 不需要 `--allow-nonformal`。

Rhino、单设备 debug、临时补测如果要并表，必须显式 `--allow-nonformal`。
