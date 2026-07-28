#!/usr/bin/env bash
# Run MNN benchmark on Rhino Pi-X1, Orange Pi 5+, and x86 CUDA host.
# Output CSV with full config for reproducibility: device, backend, model,
# input_shape, warmup, loop, cpu_freq, gpu_freq, ddr_freq, avg/min/max ms.
#
# Prerequisites:
#   - .env with RHINO_PI_PASSWORD / ORANGE_PI_PASSWORD (sourced by caller)
#   - sshpass installed
#   - ARM benchmark.out + libs deployed to devices
#   - schedule_rhino.sh / schedule_orangepi.sh on devices
#   - Models deployed to <remote>/models/
#   - x86 CUDA build at build-x86-cuda/

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
set -a && source .env 2>/dev/null && set +a || true

LOOP=10; WARMUP=5; THREAD=4; PREC=2
RESULT_DIR="$PWD/benchmark-model-result"
mkdir -p "$RESULT_DIR"
CSV="$RESULT_DIR/bench_summary.csv"

# Model input shapes (from GetMNNInfo)
declare -A MODEL_SHAPE=(
  [whisper.mnn]="1x80x3000"
  [yolo_world.mnn]="1x3x640x640"
  [depth_anything_v3.mnn]="1x3x518x518"
  [clip.mnn]="1x3x224x224"
  [metric3d.mnn]="1x3x616x1064"
  [mask2former.mnn]="1x3x256x256"
  [vitpose.mnn]="1x3x256x192"
)

# CSV header
echo "device,backend,model,input_shape,warmup,loop,cpu_freq_mhz,gpu_freq_mhz,ddr_freq_mhz,avg_ms,min_ms,max_ms" > "$CSV"

# Parse benchmark.out output lines (busybox-compatible):
#   [ - ] whisper.mnn   max = 333.283 ms  min = 333.283 ms  avg = 333.283 ms
parse_and_append() {
    local device="$1" backend="$2" cpu_f="$3" gpu_f="$4" ddr_f="$5" log="$6"
    sed -n 's/^[[:space:]]*\[ - \][[:space:]]*\([^[:space:]]*\).*max[[:space:]]*=[[:space:]]*\([0-9.]*\).*min[[:space:]]*=[[:space:]]*\([0-9.]*\).*avg[[:space:]]*=[[:space:]]*\([0-9.]*\).*/\1|\2|\3|\4/p' \
        "$log" | while IFS='|' read -r model mx mn av; do
            local shape="${MODEL_SHAPE[$model]:-unknown}"
            printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
                "$device" "$backend" "$model" "$shape" "$WARMUP" "$LOOP" \
                "$cpu_f" "$gpu_f" "$ddr_f" "$av" "$mn" "$mx" >> "$CSV"
        done
}

# Extract freq summary from schedule_*.sh status output.
# CPU: take max across clusters (kHz -> MHz)
get_rhino_freqs() {
    local log="$1"
    # cpu: max of cur values across policies
    local cpu_mhz
    cpu_mhz=$(grep -oE 'cur=[0-9]+' "$log" | grep -oE '[0-9]+' | sort -n | tail -1)
    cpu_mhz=$((cpu_mhz / 1000))
    local gpu_mhz
    gpu_mhz=$(grep -oE 'gpuclk=[0-9]+' "$log" | grep -oE '[0-9]+' | head -1)
    gpu_mhz=$((gpu_mhz / 1000000))
    echo "${cpu_mhz}|${gpu_mhz}|n/a"
}

get_orangepi_freqs() {
    local log="$1"
    local cpu_mhz
    cpu_mhz=$(grep -oE 'cur=[0-9]+' "$log" | grep -oE '[0-9]+' | sort -n | tail -1)
    cpu_mhz=$((cpu_mhz / 1000))
    local gpu_mhz
    gpu_mhz=$(grep 'GPU(Mali)' "$log" | grep -oE 'cur=[0-9]+' | grep -oE '[0-9]+')
    gpu_mhz=$((gpu_mhz / 1000000))
    local ddr_mhz
    ddr_mhz=$(grep 'DDR(DMC)' "$log" | grep -oE 'cur=[0-9]+' | grep -oE '[0-9]+')
    ddr_mhz=$((ddr_mhz / 1000000))
    echo "${cpu_mhz}|${gpu_mhz}|${ddr_mhz}"
}

run_remote() {
    local device_name="$1" host="$2" pass_var="$3" remote="$4" backend_name="$5" forward="$6" freqs="$7"
    local pass="${!pass_var}"
    echo ">>> [$device_name] $backend_name (forward=$forward)"
    local log; log="$RESULT_DIR/${device_name}_${backend_name}.log"
    SSHPASS="$pass" sshpass -e ssh -o StrictHostKeyChecking=no "root@$host" \
        "cd $remote && LD_LIBRARY_PATH=\$PWD/lib \
         bin/benchmark.out models $LOOP $WARMUP $forward $THREAD $PREC" 2>&1 \
        | tee "$log" | tail -15
    local cpu_f gpu_f ddr_f
    IFS='|' read -r cpu_f gpu_f ddr_f <<< "$freqs"
    parse_and_append "$device_name" "$backend_name" "$cpu_f" "$gpu_f" "$ddr_f" "$log"
}

# ---------------------------------------------------------------------------
# 1. Rhino Pi-X1: OpenCL / Vulkan / CPU
# ---------------------------------------------------------------------------
RHINO_HOST=192.168.101.227
RHINO_REMOTE=/mnt/nvme/workspace/benchmark-model
RHINO_NAME=rhino-pi-x1

echo "=== Locking freq on $RHINO_NAME ==="
freq_log="$RESULT_DIR/${RHINO_NAME}_freq.log"
SSHPASS="$RHINO_PI_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no \
    "root@$RHINO_HOST" "bash /data/local/tmp/schedule_rhino.sh set; bash /data/local/tmp/schedule_rhino.sh status" \
    2>&1 | tee "$freq_log" | tail -10
RHINO_FREQS=$(get_rhino_freqs "$freq_log")
echo "  freqs: $RHINO_FREQS"

run_remote "$RHINO_NAME" "$RHINO_HOST" RHINO_PI_PASSWORD "$RHINO_REMOTE" "opencl" 3 "$RHINO_FREQS"

# Vulkan: Rhino Pi-X1 (Ubuntu, Adreno) needs libvulkan.so symlink in lib/
# (system only has libvulkan.so.1; MNN dlopens "libvulkan.so")
SSHPASS="$RHINO_PI_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no \
    "root@$RHINO_HOST" \
    "ln -sf /usr/lib/aarch64-linux-gnu/libvulkan.so.1 $RHINO_REMOTE/lib/libvulkan.so 2>/dev/null; \
     ls /usr/lib/libvulkan.so.1 /lib/aarch64-linux-gnu/libvulkan.so.1 2>/dev/null | head -1" || true
run_remote "$RHINO_NAME" "$RHINO_HOST" RHINO_PI_PASSWORD "$RHINO_REMOTE" "vulkan" 7 "$RHINO_FREQS"
run_remote "$RHINO_NAME" "$RHINO_HOST" RHINO_PI_PASSWORD "$RHINO_REMOTE" "cpu" 0 "$RHINO_FREQS"

# ---------------------------------------------------------------------------
# 2. Orange pi 5+: OpenCL / CPU
# ---------------------------------------------------------------------------
ORANGE_HOST=192.168.101.113
ORANGE_REMOTE=/root/benchmark-model
ORANGE_NAME=orange-pi-5plus

echo "=== Locking freq on $ORANGE_NAME ==="
freq_log="$RESULT_DIR/${ORANGE_NAME}_freq.log"
SSHPASS="$ORANGE_PI_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no \
    "root@$ORANGE_HOST" "bash /tmp/schedule_orangepi.sh set; bash /tmp/schedule_orangepi.sh status" \
    2>&1 | tee "$freq_log" | tail -10
ORANGE_FREQS=$(get_orangepi_freqs "$freq_log")
echo "  freqs: $ORANGE_FREQS"

run_remote "$ORANGE_NAME" "$ORANGE_HOST" ORANGE_PI_PASSWORD "$ORANGE_REMOTE" "opencl" 3 "$ORANGE_FREQS"
run_remote "$ORANGE_NAME" "$ORANGE_HOST" ORANGE_PI_PASSWORD "$ORANGE_REMOTE" "cpu" 0 "$ORANGE_FREQS"

# ---------------------------------------------------------------------------
# 3. x86 CUDA host (localhost)
# ---------------------------------------------------------------------------
CUDA_BUILD="$PWD/build-x86-cuda"
X86_NAME=x86-cuda
if [ -x "$CUDA_BUILD/benchmark.out" ]; then
    echo "=== Locking freq on $X86_NAME ==="
    # x86: read CPU max freq from sysfs; GPU via nvidia-smi
    cpu_max_khz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)
    cpu_mhz=$((cpu_max_khz / 1000))
    gpu_mhz=$(nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    [ -z "$gpu_mhz" ] && gpu_mhz=0
    X86_FREQS="${cpu_mhz}|${gpu_mhz}|n/a"
    echo "  freqs: $X86_FREQS"

    # Governor: performance
    for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance 2>/dev/null > "$c" 2>/dev/null || true
    done

    echo ">>> [$X86_NAME] cuda (forward=2)"
    flat=/tmp/opencode/mnn-bench-flat
    rm -rf "$flat" && mkdir -p "$flat"
    for m in \
        /mnt/hdd_4tb/kernflow-models/artifacts/whisper/pytorch-mnn/whisper.mnn \
        /mnt/hdd_4tb/kernflow-models/artifacts/yolo-world/yolo_world.mnn \
        /mnt/hdd_4tb/kernflow-models/artifacts/depth-anything-v3/depth_anything_v3.mnn \
        /mnt/hdd_4tb/kernflow-models/artifacts/clip/pytorch-mnn/clip.mnn \
        /mnt/hdd_4tb/kernflow-models/artifacts/metric3d/pytorch-mnn/metric3d.mnn \
        /mnt/hdd_4tb/kernflow-models/artifacts/mask2former/pytorch-mnn/mask2former.mnn \
        /mnt/hdd_4tb/kernflow-models/artifacts/vitpose/pytorch-mnn/vitpose.mnn; do
        ln -sf "$m" "$flat/"
    done
    log="$RESULT_DIR/${X86_NAME}_cuda.log"
    LD_LIBRARY_PATH="$CUDA_BUILD:$CUDA_BUILD/source/backend/cuda" \
    "$CUDA_BUILD/benchmark.out" "$flat" $LOOP $WARMUP 2 $THREAD $PREC 2>&1 \
        | tee "$log" | tail -15
    local cpu_f gpu_f ddr_f
    IFS='|' read -r cpu_f gpu_f ddr_f <<< "$X86_FREQS"
    parse_and_append "$X86_NAME" "cuda" "$cpu_f" "$gpu_f" "$ddr_f" "$log"

    # Also run CPU baseline on x86 for comparison
    echo ">>> [$X86_NAME] cpu (forward=0)"
    log="$RESULT_DIR/${X86_NAME}_cpu.log"
    LD_LIBRARY_PATH="$CUDA_BUILD:$CUDA_BUILD/express" \
    "$CUDA_BUILD/benchmark.out" "$flat" $LOOP $WARMUP 0 $THREAD $PREC 2>&1 \
        | tee "$log" | tail -15
    IFS='|' read -r cpu_f gpu_f ddr_f <<< "$X86_FREQS"
    parse_and_append "$X86_NAME" "cpu" "$cpu_f" "$gpu_f" "$ddr_f" "$log"
fi

echo
echo "=== CSV saved: $CSV ==="
echo
echo "=== Results (latency in ms) ==="
column -t -s',' "$CSV"
