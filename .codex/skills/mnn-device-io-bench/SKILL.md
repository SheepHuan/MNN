---
name: mnn-device-io-bench
description: 当用户要求测试或报告 Jetson、Orange Pi 5 Plus、Rhino Pi-X1/Adreno 等 MNN 目标设备的 NVMe/SSD/UFS/块设备型号、fio 磁盘读取速度、lmbench bw_mem rd 内存读取带宽、STREAM 持续内存带宽，或为 PIC/OpenCL/CUDA benchmark 采集存储和 DRAM 基线时使用。
---

# MNN Device IO Bench

本 Skill 用于给 MNN PIC / LLM 性能实验采集设备侧存储和内存基线。主实验使用 `fio` 测存储读取，使用 `lmbench bw_mem rd` 测 CPU 侧 DRAM read bandwidth，可选用 STREAM 报告持续内存带宽。

## 固定设备

Jetson：

```text
jetson@192.168.101.192
```

Orange Pi 5 Plus / OpenCL：

```text
orangepi@192.168.101.113
```

Rhino Pi-X1 / Adreno OpenCL：

```text
aidlux@192.168.101.227
```

Rhino Pi-X1 默认密码是 `aidlux`，但不要把密码写进脚本、日志或提交内容。自动化脚本默认使用 SSH key 或已配置的 SSH agent；如果没有免密登录，先手动建立登录方式。

## 推荐指标

最终报告至少给出：

```text
device
storage model / medium
fio sequential read MiB/s
fio random 4KB read IOPS
fio random 128KB read MiB/s
lmbench bw_mem rd MB/s
STREAM Triad GB/s          可选
test path and backing block device
```

不要只用 `dd` 或 `hdparm` 作为论文主结果；它们只能作为 sanity check。`fio` 必须记录 `--direct=1`、block size、iodepth、runtime、测试文件路径和输出中的 `READ` 指标。

## 一键脚本

优先从本 MNN 仓库根目录运行：

```bash
bash .codex/skills/mnn-device-io-bench/scripts/run_remote_device_io_bench.sh \
  --devices jetson,orangepi,aidlux \
  --fio-size 4G \
  --fio-runtime 60 \
  --mem-size 1024M
```

脚本会：

1. SSH 到目标设备。
2. 采集 `uname`、`lscpu`、`free`、`lsblk`、`df`、`findmnt`、`nvme list`。
3. 在目标测试目录创建或复用 `fio-test.bin`。
4. 运行 `fio` 顺序读、4KB 随机读、128KB 随机读。
5. 运行 `lmbench bw_mem <mem-size> rd`。
6. 把远端原始结果拉回本仓库 `.cache/device-io-bench/<run-id>/raw/<device>/`。
7. 生成 `.cache/device-io-bench/<run-id>/summary.tsv`。

默认测试目录：

```text
Jetson:     /home/jetson/code/kvshare-edge/impl/MNN/.cache/device-io-bench
Orange Pi:  /mnt/ssd/code/.cache/device-io-bench
Rhino Pi-X1:     /mnt/nvme/mnn_pic_opencl/cache/device-io-bench
```

当前设备可测性快照：

```text
Jetson:
  SSD/NVMe IO: 可测。默认测试目录所在挂载点是 /，source=/dev/nvme0n1p1，ext4，
               磁盘型号 KIOXIA-EXCERIA G2 SSD，约 916G。
  内存 IO:     可测，但当前 bw_mem 未安装；需要安装 lmbench，或编译/运行 STREAM。
  依赖状态:    fio 已安装；lsblk/findmnt/df 已安装；nvme-cli 当前未安装，但 lsblk 足够确认块设备。

Orange Pi 5 Plus:
  SSD/NVMe IO: 可测，但必须使用 /mnt/ssd 或其子目录。/mnt/ssd source=/dev/nvme0n1p1，
               ext4，磁盘型号 SSSTC CA5-8D512，约 469G。
  默认根分区:  /home/orangepi/... 落在 /dev/mmcblk1p2，且当前根分区约 95% 已用；
               不要用它作为 SSD IO 结论。
  内存 IO:     可测，但当前 bw_mem 未安装；需要安装 lmbench，或编译/运行 STREAM。
  依赖状态:    fio 当前未安装；nvme-cli/lsblk/findmnt/df 已安装。需要安装 fio + lmbench。

Rhino Pi-X1 / Adreno:
  SSD/NVMe IO: 可测。推荐测试目录 /mnt/nvme/mnn_pic_opencl/cache/device-io-bench，
               source=/dev/nvme0n1p1，ext4，磁盘型号 GVL512，约 469G。
  内存 IO:     可测，但当前 bw_mem 未安装；需要安装 lmbench，或编译/运行 STREAM。
  依赖状态:    fio/nvme-cli 当前未安装；lsblk/findmnt/df 已安装。需要安装 fio + lmbench；
               需要 sudo 时用交互式 sudo，不把密码写进脚本或日志。
```

三台设备都可以采集 SSD/NVMe 与内存基线，但正式报告必须满足两个条件：测试目录经 `findmnt -T` 证明在目标
SSD/NVMe 挂载点上，且依赖缺失时先安装 `fio` / `lmbench`，不要把缺依赖或落在 eMMC/rootfs 的结果写成 SSD
性能。

如果要测某块指定 SSD/NVMe/UFS，必须把 `--test-dir` 指向该存储挂载点下的目录，并用结果里的 `df-test-dir.txt`、`findmnt-test-dir.txt`、`lsblk.json` 确认测试路径确实落在目标块设备上。例如：

```bash
bash .codex/skills/mnn-device-io-bench/scripts/run_remote_device_io_bench.sh \
  --devices orangepi \
  --test-dir /mnt/ssd/code/.cache/device-io-bench \
  --fio-size 8G \
  --fio-runtime 60 \
  --mem-size 1024M
```

Rhino Pi-X1 使用 NVMe 专用工作区：

```bash
bash .codex/skills/mnn-device-io-bench/scripts/run_remote_device_io_bench.sh \
  --devices aidlux \
  --test-dir /mnt/nvme/mnn_pic_opencl/cache/device-io-bench \
  --fio-size 8G \
  --fio-runtime 60 \
  --mem-size 1024M
```

依赖未安装时，优先手动在设备上安装：

```bash
sudo apt update
sudo apt install -y fio lmbench nvme-cli build-essential
```

也可以让脚本尝试非交互 sudo 安装：

```bash
bash .codex/skills/mnn-device-io-bench/scripts/run_remote_device_io_bench.sh \
  --devices orangepi \
  --install-deps
```

如果 `sudo -n` 不可用，脚本会跳过安装并在日志中报告缺失命令。

## 频率固定

正式实验前固定频率，否则内存带宽和随机读延迟容易波动。Jetson CUDA / PIC 正式性能测试前还必须强制风扇最高转速；实测默认 `nvfancontrol` 在低温时会保持 `pwm=0/rpm=0`，Llama-3.2-1B CUDA `pp512` 曾触发 SSH 断开和整机重启，而同一 workload 在 `pwm=255/rpm≈3650` 下跑通且温度远低于 thermal trip。脚本默认只采集状态，不改设备配置；需要固定频率时显式传：

```bash
bash .codex/skills/mnn-device-io-bench/scripts/run_remote_device_io_bench.sh \
  --devices jetson,orangepi,aidlux \
  --set-performance
```

`--set-performance` 只做通用 best-effort。正式 PIC / OpenCL / CUDA 性能测试前优先使用本 skill 的三台设备专用调频脚本，分别在目标设备本机执行，或通过 SSH 把脚本 stdin 传过去执行：

```bash
ssh jetson@192.168.101.192 \
  'bash -s -- --show-only' \
  < .codex/skills/mnn-device-io-bench/scripts/set_jetson_max_perf.sh

ssh orangepi@192.168.101.113 \
  'bash -s -- --show-only' \
  < .codex/skills/mnn-device-io-bench/scripts/set_orangepi_max_perf.sh

ssh aidlux@192.168.101.227 \
  'bash -s -- --show-only' \
  < .codex/skills/mnn-device-io-bench/scripts/set_aidlux_max_perf.sh
```

确认状态后去掉 `--show-only` 应用最高性能设置：

```bash
ssh jetson@192.168.101.192 \
  'bash -s' \
  < .codex/skills/mnn-device-io-bench/scripts/set_jetson_max_perf.sh

ssh orangepi@192.168.101.113 \
  'bash -s' \
  < .codex/skills/mnn-device-io-bench/scripts/set_orangepi_max_perf.sh

ssh aidlux@192.168.101.227 \
  'bash -s' \
  < .codex/skills/mnn-device-io-bench/scripts/set_aidlux_max_perf.sh
```

三个脚本都支持：

```text
--dry-run             只打印将写入的 sysfs / root 命令
--show-only           只采集并打印当前频率/governor 状态
--allow-sudo-prompt   允许交互式 sudo 密码提示；自动化默认不启用
```

设备覆盖范围：

```text
Jetson:
  .codex/skills/mnn-device-io-bench/scripts/set_jetson_max_perf.sh
  - nvpmodel mode 0，可用 JETSON_NVP_MODEL 覆盖
  - jetson_clocks
  - CPU cpufreq governor/per-core min=max
  - GPU/media devfreq max
  - EMC debug clk best-effort lock to max_rate
  - 默认停止 nvfancontrol 并强制 pwmfan/pwm1=255；日志必须确认 fan rpm 非 0
    如需临时跳过风扇强制，可设置 MNN_JETSON_FORCE_MAX_FAN=0，但不得用于正式 CUDA/PIC 性能报告

Orange Pi 5 Plus:
  .codex/skills/mnn-device-io-bench/scripts/set_orangepi_max_perf.sh
  - CPU cpufreq governor/per-core min=max
  - Mali GPU /sys/class/devfreq/fb000000.gpu governor=performance, min=max
  - RK3588 DMC /sys/class/devfreq/dmc governor=performance, min=max

Rhino Pi-X1 / Adreno:
  .codex/skills/mnn-device-io-bench/scripts/set_aidlux_max_perf.sh
  - CPU cpufreq governor/per-core min=max
  - KGSL force_clk_on / force_bus_on / force_rail_on
  - KGSL pwrlevel 固定到 0，GPU devfreq min=max
  - Qualcomm bus_dcvs DDR / DDRQOS / LLCC boost_freq=max，并提高子节点 min_freq
```

如果没有 root 或 sudo 权限，不要伪造固定频率结果；报告中写明 governor / 当前频率来自脚本 `--show-only` 输出。Rhino Pi-X1 默认密码是 `aidlux`，但脚本、日志和报告中不要写入密码；需要交互 sudo 时由人工在终端输入。

## 手工命令

内存读取：

```bash
bw_mem 1024M rd
```

STREAM：

```bash
wget https://www.cs.virginia.edu/stream/FTP/Code/stream.c
gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=100000000 stream.c -o stream
export OMP_NUM_THREADS=$(nproc)
./stream
```

顺序读：

```bash
fio --name=seqread \
  --filename=/path/on/target-storage/fio-test.bin \
  --size=4G \
  --rw=read \
  --bs=1M \
  --iodepth=32 \
  --numjobs=1 \
  --direct=1 \
  --ioengine=libaio \
  --runtime=60 \
  --time_based \
  --group_reporting
```

4KB 随机读：

```bash
fio --name=randread4k \
  --filename=/path/on/target-storage/fio-test.bin \
  --size=4G \
  --rw=randread \
  --bs=4k \
  --iodepth=32 \
  --numjobs=1 \
  --direct=1 \
  --ioengine=libaio \
  --runtime=60 \
  --time_based \
  --group_reporting
```

128KB 随机读：

```bash
fio --name=randread128k \
  --filename=/path/on/target-storage/fio-test.bin \
  --size=4G \
  --rw=randread \
  --bs=128k \
  --iodepth=16 \
  --numjobs=1 \
  --direct=1 \
  --ioengine=libaio \
  --runtime=60 \
  --time_based \
  --group_reporting
```

## 报告约束

- `nvme list` 只能说明系统识别到 NVMe；最终 storage medium 以测试目录所在挂载点和块设备为准。
- Rhino Pi-X1 常见目标可能是 UFS 或其它内置存储，不要强行写成 NVMe；无 NVMe 时报告 `nvme: none`，并使用 `lsblk` / `findmnt` 的块设备型号。
- `fio` 测试文件大小要明显大于 page cache；正式结果建议 4G 或更大，并使用 `--direct=1`。
- `bw_mem rd` 的第二列按 MB/s 报告；如果转 GB/s，明确使用 1024 还是 1000 换算。
- STREAM 的 Triad 是持续内存带宽代表值，不等同于纯 read bandwidth；论文里可同时写 `bw_mem rd` 和 `STREAM Triad`。
