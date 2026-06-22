---
name: mnn-device-io-bench
description: 当用户要求测试或报告 Jetson、Orange Pi 5 Plus、AidLux/Adreno 等 MNN 目标设备的 NVMe/SSD/UFS/块设备型号、fio 磁盘读取速度、lmbench bw_mem rd 内存读取带宽、STREAM 持续内存带宽，或为 PIC/OpenCL/CUDA benchmark 采集存储和 DRAM 基线时使用。
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

AidLux / Adreno OpenCL：

```text
aidlux@192.168.101.227
```

AidLux 默认密码是 `aidlux`，但不要把密码写进脚本、日志或提交内容。自动化脚本默认使用 SSH key 或已配置的 SSH agent；如果没有免密登录，先手动建立登录方式。

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
Orange Pi:  /home/orangepi/code/kvshare-edge/impl/MNN/.cache/device-io-bench
AidLux:     /home/aidlux/.cache/mnn-device-io-bench
```

如果要测某块指定 SSD/NVMe/UFS，必须把 `--test-dir` 指向该存储挂载点下的目录，并用结果里的 `df-test-dir.txt`、`findmnt-test-dir.txt`、`lsblk.json` 确认测试路径确实落在目标块设备上。例如：

```bash
bash .codex/skills/mnn-device-io-bench/scripts/run_remote_device_io_bench.sh \
  --devices orangepi \
  --test-dir /mnt/nvme/mnn-device-io-bench \
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

正式实验前固定频率，否则内存带宽和随机读延迟容易波动。脚本默认只采集状态，不改设备配置；需要固定频率时显式传：

```bash
bash .codex/skills/mnn-device-io-bench/scripts/run_remote_device_io_bench.sh \
  --devices jetson,orangepi,aidlux \
  --set-performance
```

`--set-performance` 会尽量执行：

```text
Jetson: sudo nvpmodel -m 0; sudo jetson_clocks
Linux AArch64: /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor = performance
```

如果没有 sudo 权限，不要伪造固定频率结果；报告中写明 governor / 当前频率来自采集日志。

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
- AidLux 常见目标可能是 UFS 或其它内置存储，不要强行写成 NVMe；无 NVMe 时报告 `nvme: none`，并使用 `lsblk` / `findmnt` 的块设备型号。
- `fio` 测试文件大小要明显大于 page cache；正式结果建议 4G 或更大，并使用 `--direct=1`。
- `bw_mem rd` 的第二列按 MB/s 报告；如果转 GB/s，明确使用 1024 还是 1000 换算。
- STREAM 的 Triad 是持续内存带宽代表值，不等同于纯 read bandwidth；论文里可同时写 `bw_mem rd` 和 `STREAM Triad`。
