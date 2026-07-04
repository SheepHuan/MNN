# Rhino Pi-X1 / Adreno

先读 [common.md](common.md)。

## 角色

- extra-profile device
- backend: OpenCL / Adreno

Rhino 不进入 formal 默认矩阵。任何 sweep 都必须显式 `--allow-extra-device`。

## 正式默认口径

- 频率：`cpu=max,gpu=max,ddr=max`
- contexts：`512,1024,1536,2048,2560`
- 暂不纳入：`3072`

## 固定路径

- ssh: `aidlux@192.168.101.227`
- remote work: `/mnt/nvme/mnn_pic_opencl`
- artifact: `/mnt/nvme/mnn_pic_opencl/artifacts/aidlux_adreno_opencl`
- shared cache root: `/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep`
- OpenCL runtime/autotune cache: `/mnt/nvme/mnn_pic_opencl/cache/pic_prefill_latency_sweep/runtime_cache/opencl/mnn_cachefile.bin`
- logs: `/mnt/nvme/mnn_pic_opencl/logs`

## 固定端口

- remote port: `18133`
- local tunnel port: `19133`

## Prefill sweep

单设备 Rhino profile：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_rhino.sh \
  --model-key minicpm5-1b \
  --modes cacheblend,epic \
  --contexts 1024,2048
```

Rhino wrapper 默认会带：

- `--devices rhino`
- `--allow-extra-device`
- `--frequency-profile max`
- `--remote-memory-limit-percent 95`
- `--server-env LD_PRELOAD=/usr/lib/libOpenCL_adreno.so`

## Power 采集

Rhino / rhinopi 的外部功耗计按用户约定使用：

```text
monitor_type=df
serial=1A5D43
```

测试 `benchmark.csv` 中 Rhino 已有配置时，按模型分组调用 wrapper，并打开 `--power-capture`。不要加 `--only-missing-contexts`：

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_pic_prefill_latency_sweep_rhino.sh \
  --model-key minicpm5-1b \
  --benchmark-csv benchmark.csv \
  --power-capture \
  --power-rep 3
```

Rhino power 覆盖不跑 `llama3.2-1b` / `Llama3.2 1B`；按 `benchmark.csv` 规划时只覆盖需要的非 1B 模型。

输出在：

```text
.codex/skills/mnn-pic-benchmark/power/rhino/<YYYY-MM-DD>/<HH>/<MM>/<process-id>/manifest.tsv
.codex/skills/mnn-pic-benchmark/power/rhino/<YYYY-MM-DD>/<HH>/<MM>/<process-id>/<uuid>.csv
.codex/skills/mnn-pic-benchmark/power/rhino/<YYYY-MM-DD>/<HH>/<MM>/<process-id>/<uuid>.png
.codex/skills/mnn-pic-benchmark/power/rhino/<YYYY-MM-DD>/<HH>/<MM>/<process-id>/<uuid>.txt
```

每个 `<uuid>.txt` 必须能对应回 benchmark row：model、context_tokens、mode/algorithm、budget、rep total、rep interval、每次执行结果、frequency、backend 和 power monitor 都要完整记录。
`--power-rep 3` 表示同一个 `<uuid>.csv` 内连续执行 3 次测量，每次执行之间固定间隔 3 秒；不要为 rep 1/2/3 分别生成 3 个 CSV。

## Dataset bench

```bash
bash .codex/skills/mnn-pic-benchmark/scripts/run_mnn_pic_dataset_bench.sh \
  --device-key rhino \
  --allow-extra-device \
  --run-id <run_id> \
  -- \
  --dataset hotpotqa \
  --mode pic_cache_reuse \
  --phase both
```

## Decode repair

Rhino decode repair 只能作为额外 profile。正式表格仍以 Jetson + OrangePi 为准。

Rhino / Adreno 的 Decode TPOT 也按 PIC dualgraph 双族组织；这里的 `x0`
是 PIC dualgraph decode-repair family 内 active rows=1 的退化行，不是普通 LLM：

- `optimized-decode-repair`：Adreno/Rhino 专属优化导出，例如包含 `PicAdreno*` / guarded tiny-row MLP 的模型。
- `normal/generic-decode-repair`：同一 PIC graph-boundary 模型语义，但导出时使用 `--pic_decode_fusion_backend generic`，不生成 `PicAdreno*` 后端专属 rewrite。
- `true-normal-llm-x0`：普通 `.cache/mnn-llm-export/<model>/config.json` 的 MNN decode 绝对 baseline，只作为 `x0` 参考。

两族都测 `x=0/1/3/5/7`，再按相同 `x` 计算 `generic_tpot[x] / optimized_tpot[x]`。若看到某个族的 `x0` 比 `x>0` 慢很多，先排查 cold OpenCL tuning、旧 runtime cache rebuild、模型路径选错或把另一族的 x0 行混入；不要直接报告 speedup。

## 构建入口

构建 artifact 先读：

- `.codex/skills/mnn-build-artifacts/references/aidlux-adreno-opencl.md`

## 运行时提示

设备侧通常需要：

- `LD_PRELOAD=/usr/lib/libOpenCL_adreno.so`
- `LD_LIBRARY_PATH=<artifact>/lib:/usr/lib:/usr/lib/aarch64-linux-gnu`
