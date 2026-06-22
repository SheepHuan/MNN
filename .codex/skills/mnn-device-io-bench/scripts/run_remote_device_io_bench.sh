#!/usr/bin/env bash
set -euo pipefail

DEVICES="jetson,orangepi,aidlux"
FIO_SIZE="4G"
FIO_RUNTIME="60"
MEM_SIZE="1024M"
RUN_ID="device-io-$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR=""
GLOBAL_TEST_DIR=""
INSTALL_DEPS=0
SET_PERFORMANCE=0
SKIP_FIO=0
SKIP_BW_MEM=0
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=15 -o StrictHostKeyChecking=accept-new)

usage() {
  cat <<'EOF'
Usage:
  run_remote_device_io_bench.sh [options]

Options:
  --devices LIST        Comma-separated aliases: jetson,orangepi,aidlux,all
  --test-dir PATH      Override remote test directory for all devices
  --fio-size SIZE      fio test file size, default: 4G
  --fio-runtime SEC    fio runtime per read workload, default: 60
  --mem-size SIZE      lmbench bw_mem size, default: 1024M
  --output-dir DIR     Local output directory, default: .cache/device-io-bench/<run-id>
  --run-id ID          Run id used under remote and local result dirs
  --install-deps       Try sudo -n apt-get install fio lmbench nvme-cli build-essential
  --set-performance    Try to set Jetson clocks / Linux CPU governors to performance
  --skip-fio           Skip fio storage tests
  --skip-bw-mem        Skip lmbench bw_mem rd
  --ssh-option OPT     Append one ssh/scp option, repeatable
  -h, --help           Show this help

Default devices:
  jetson    jetson@192.168.101.192
  orangepi  orangepi@192.168.101.113
  aidlux    aidlux@192.168.101.227
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --devices)
      DEVICES="$2"
      shift 2
      ;;
    --test-dir)
      GLOBAL_TEST_DIR="$2"
      shift 2
      ;;
    --fio-size)
      FIO_SIZE="$2"
      shift 2
      ;;
    --fio-runtime)
      FIO_RUNTIME="$2"
      shift 2
      ;;
    --mem-size)
      MEM_SIZE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --install-deps)
      INSTALL_DEPS=1
      shift
      ;;
    --set-performance)
      SET_PERFORMANCE=1
      shift
      ;;
    --skip-fio)
      SKIP_FIO=1
      shift
      ;;
    --skip-bw-mem)
      SKIP_BW_MEM=1
      shift
      ;;
    --ssh-option)
      SSH_OPTS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$PWD/.cache/device-io-bench/$RUN_ID"
fi
mkdir -p "$OUTPUT_DIR/raw"

shell_quote() {
  printf "%q" "$1"
}

resolve_device() {
  local alias="$1"
  case "$alias" in
    jetson)
      printf '%s\t%s\n' "jetson@192.168.101.192" "/home/jetson/code/kvshare-edge/impl/MNN/.cache/device-io-bench"
      ;;
    orangepi)
      printf '%s\t%s\n' "orangepi@192.168.101.113" "/home/orangepi/code/kvshare-edge/impl/MNN/.cache/device-io-bench"
      ;;
    aidlux)
      printf '%s\t%s\n' "aidlux@192.168.101.227" "/home/aidlux/.cache/mnn-device-io-bench"
      ;;
    *)
      echo "Unknown device alias: $alias" >&2
      return 1
      ;;
  esac
}

expand_devices() {
  local list="$1"
  if [[ "$list" == "all" ]]; then
    printf '%s\n' "jetson" "orangepi" "aidlux"
    return
  fi
  local item
  IFS=',' read -ra parts <<< "$list"
  for item in "${parts[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -n "$item" ]] && printf '%s\n' "$item"
  done
}

generate_summary() {
  local out_dir="$1"
  python3 - "$out_dir" <<'PY'
import json
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
raw_dir = out_dir / "raw"
summary = out_dir / "summary.tsv"

headers = [
    "device",
    "host",
    "test_dir",
    "mount_source",
    "mount_fstype",
    "nvme_models",
    "disk_models",
    "bw_mem_rd_MB_s",
    "fio_seqread_MiB_s",
    "fio_randread4k_IOPS",
    "fio_randread4k_MiB_s",
    "fio_randread4k_p99_us",
    "fio_randread128k_IOPS",
    "fio_randread128k_MiB_s",
    "fio_randread128k_p99_us",
]

def clean(value):
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\n", " ").strip()

def load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def parse_env(path):
    data = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return data

def walk_blockdevices(devices):
    for dev in devices or []:
        yield dev
        yield from walk_blockdevices(dev.get("children") or [])

def nvme_models(device_dir):
    data = load_json(device_dir / "nvme-list.json")
    models = []
    if isinstance(data, dict):
        devices = data.get("Devices") or data.get("devices") or []
        for dev in devices:
            model = (
                dev.get("ModelNumber")
                or dev.get("Model")
                or dev.get("model")
                or dev.get("mn")
                or dev.get("NameSpace")
            )
            serial = dev.get("SerialNumber") or dev.get("Serial") or dev.get("sn")
            path = dev.get("DevicePath") or dev.get("NameSpace") or dev.get("Path")
            if model:
                parts = [str(model).strip()]
                if serial:
                    parts.append(f"SN={serial}")
                if path:
                    parts.append(f"path={path}")
                models.append(" ".join(parts))
    if models:
        return "; ".join(models)

    text_path = device_dir / "nvme-list.txt"
    if text_path.exists():
        lines = [
            line.strip()
            for line in text_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip().startswith("/dev/nvme")
        ]
        return "; ".join(lines)
    return ""

def disk_models(device_dir):
    data = load_json(device_dir / "lsblk.json")
    if not isinstance(data, dict):
        return ""
    models = []
    for dev in walk_blockdevices(data.get("blockdevices") or []):
        if dev.get("type") == "disk":
            name = dev.get("name", "")
            size = dev.get("size", "")
            model = dev.get("model") or ""
            rota = dev.get("rota")
            medium = "rotational" if rota is True else "nonrotational" if rota is False else ""
            models.append(" ".join(x for x in [name, size, model, medium] if x))
    return "; ".join(models)

def mount_info(device_dir):
    data = load_json(device_dir / "findmnt-test-dir.json")
    if isinstance(data, dict):
        filesystems = data.get("filesystems") or []
        if filesystems:
            fs = filesystems[0]
            return fs.get("source", ""), fs.get("fstype", "")
    return "", ""

def bw_mem(device_dir):
    path = device_dir / "bw_mem-rd.txt"
    if not path.exists():
        return ""
    value = ""
    pattern = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s*$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            value = match.group(2)
    return value

def fio_read_metrics(device_dir, name):
    data = load_json(device_dir / f"fio-{name}.json")
    if not isinstance(data, dict):
        return "", "", ""
    jobs = data.get("jobs") or []
    if not jobs:
        return "", "", ""
    read = jobs[0].get("read") or {}
    bw_bytes = read.get("bw_bytes")
    if bw_bytes is None:
        bw = read.get("bw")
        bw_mib = float(bw) / 1024.0 if bw is not None else None
    else:
        bw_mib = float(bw_bytes) / 1048576.0
    iops = read.get("iops")
    clat = read.get("clat_ns") or read.get("lat_ns") or {}
    pct = clat.get("percentile") or {}
    p99 = pct.get("99.000000") or pct.get("99.000000%") or pct.get("99.00")
    p99_us = float(p99) / 1000.0 if p99 is not None else None
    return (
        f"{float(iops):.2f}" if iops is not None else "",
        f"{bw_mib:.2f}" if bw_mib is not None else "",
        f"{p99_us:.2f}" if p99_us is not None else "",
    )

rows = []
if raw_dir.exists():
    for device_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        meta = parse_env(device_dir / "meta.env")
        mount_source, mount_fstype = mount_info(device_dir)
        seq_iops, seq_bw, seq_p99 = fio_read_metrics(device_dir, "seqread")
        r4k_iops, r4k_bw, r4k_p99 = fio_read_metrics(device_dir, "randread4k")
        r128_iops, r128_bw, r128_p99 = fio_read_metrics(device_dir, "randread128k")
        rows.append([
            device_dir.name,
            meta.get("ssh_host", ""),
            meta.get("test_dir", ""),
            mount_source,
            mount_fstype,
            nvme_models(device_dir),
            disk_models(device_dir),
            bw_mem(device_dir),
            seq_bw,
            r4k_iops,
            r4k_bw,
            r4k_p99,
            r128_iops,
            r128_bw,
            r128_p99,
        ])

with summary.open("w", encoding="utf-8") as f:
    f.write("\t".join(headers) + "\n")
    for row in rows:
        f.write("\t".join(clean(x) for x in row) + "\n")

print(summary)
PY
}

remote_payload() {
  cat <<'REMOTE'
set -u

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

sudo_run() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return $?
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo -n "$@"
    return $?
  fi
  return 127
}

write_root_file() {
  local path="$1"
  local value="$2"
  if [[ "$(id -u)" -eq 0 ]]; then
    printf '%s\n' "$value" > "$path"
    return $?
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    printf '%s\n' "$value" | sudo -n tee "$path" >/dev/null
    return $?
  fi
  return 127
}

run_to_file() {
  local output="$1"
  shift
  log "run: $*"
  "$@" > "$RESULT_DIR/$output" 2>&1
  local status=$?
  printf '%s\n' "$status" > "$RESULT_DIR/$output.exit"
  return 0
}

find_bw_mem() {
  if command -v bw_mem >/dev/null 2>&1; then
    command -v bw_mem
    return 0
  fi
  find /usr/lib/lmbench /usr/local /opt -type f -name bw_mem -perm -111 2>/dev/null | head -n 1
}

run_fio_json() {
  local name="$1"
  local rw="$2"
  local bs="$3"
  local iodepth="$4"
  log "fio $name rw=$rw bs=$bs iodepth=$iodepth runtime=${FIO_RUNTIME}s"
  fio \
    --name="$name" \
    --filename="$FIO_FILE" \
    --size="$FIO_SIZE" \
    --rw="$rw" \
    --bs="$bs" \
    --iodepth="$iodepth" \
    --numjobs=1 \
    --direct=1 \
    --ioengine=libaio \
    --runtime="$FIO_RUNTIME" \
    --time_based \
    --group_reporting \
    --invalidate=1 \
    --output-format=json \
    --output="$RESULT_DIR/fio-${name}.json" \
    > "$RESULT_DIR/fio-${name}.stdout.txt" \
    2> "$RESULT_DIR/fio-${name}.stderr.txt"
  local status=$?
  printf '%s\n' "$status" > "$RESULT_DIR/fio-${name}.exit"
  return 0
}

mkdir -p "$BENCH_TEST_DIR" "$RESULT_DIR"
cd "$BENCH_TEST_DIR" || exit 1
FIO_FILE="$BENCH_TEST_DIR/fio-test.bin"

cat > "$RESULT_DIR/meta.env" <<EOF
device_alias=$DEVICE_ALIAS
ssh_host=$SSH_HOST
test_dir=$BENCH_TEST_DIR
result_dir=$RESULT_DIR
fio_file=$FIO_FILE
fio_size=$FIO_SIZE
fio_runtime=$FIO_RUNTIME
mem_size=$MEM_SIZE
EOF

log "device=$DEVICE_ALIAS host=$SSH_HOST test_dir=$BENCH_TEST_DIR result_dir=$RESULT_DIR"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    log "installing dependencies with apt-get via non-interactive sudo if available"
    sudo_run apt-get update > "$RESULT_DIR/install-deps.log" 2>&1 || true
    sudo_run apt-get install -y fio lmbench nvme-cli build-essential >> "$RESULT_DIR/install-deps.log" 2>&1 || true
  else
    printf 'apt-get not found\n' > "$RESULT_DIR/install-deps.log"
  fi
fi

if [[ "$SET_PERFORMANCE" == "1" ]]; then
  {
    if command -v nvpmodel >/dev/null 2>&1; then
      sudo_run nvpmodel -m 0 || true
    fi
    if command -v jetson_clocks >/dev/null 2>&1; then
      sudo_run jetson_clocks || true
    fi
    for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      [[ -e "$governor" ]] || continue
      write_root_file "$governor" performance || true
    done
  } > "$RESULT_DIR/set-performance.log" 2>&1
fi

run_to_file date.txt date -Iseconds
run_to_file uname.txt uname -a
run_to_file lscpu.txt lscpu
run_to_file free.txt free -h
run_to_file cpu-governors.txt sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do [ -e "$g" ] && printf "%s " "$g" && cat "$g"; done'
run_to_file cpu-freqs.txt sh -c 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do [ -e "$f" ] && printf "%s " "$f" && cat "$f"; done'

if command -v lsblk >/dev/null 2>&1; then
  run_to_file lsblk.txt lsblk -o NAME,SIZE,TYPE,MODEL,ROTA,MOUNTPOINT
  run_to_file lsblk.json lsblk -J -o NAME,SIZE,TYPE,MODEL,ROTA,MOUNTPOINT,PKNAME
else
  printf 'lsblk missing\n' > "$RESULT_DIR/lsblk.txt"
fi

run_to_file df-test-dir.txt df -h "$BENCH_TEST_DIR"
if command -v findmnt >/dev/null 2>&1; then
  run_to_file findmnt-test-dir.txt findmnt -T "$BENCH_TEST_DIR" -o SOURCE,TARGET,FSTYPE,OPTIONS
  run_to_file findmnt-test-dir.json findmnt -J -T "$BENCH_TEST_DIR" -o SOURCE,TARGET,FSTYPE,OPTIONS
else
  printf 'findmnt missing\n' > "$RESULT_DIR/findmnt-test-dir.txt"
fi

if command -v nvme >/dev/null 2>&1; then
  run_to_file nvme-list.txt nvme list
  nvme list -o json > "$RESULT_DIR/nvme-list.json" 2> "$RESULT_DIR/nvme-list-json.stderr.txt" || true
else
  printf 'nvme command missing; install nvme-cli if this target has NVMe\n' > "$RESULT_DIR/nvme-list.txt"
fi

if [[ "$SKIP_BW_MEM" != "1" ]]; then
  BW_MEM_BIN="$(find_bw_mem || true)"
  if [[ -n "$BW_MEM_BIN" && -x "$BW_MEM_BIN" ]]; then
    log "bw_mem rd size=$MEM_SIZE bin=$BW_MEM_BIN"
    "$BW_MEM_BIN" "$MEM_SIZE" rd > "$RESULT_DIR/bw_mem-rd.txt" 2> "$RESULT_DIR/bw_mem-rd.stderr.txt"
    printf '%s\n' "$?" > "$RESULT_DIR/bw_mem-rd.exit"
  else
    printf 'bw_mem missing; install lmbench\n' > "$RESULT_DIR/bw_mem-rd.txt"
  fi
fi

if [[ "$SKIP_FIO" != "1" ]]; then
  if command -v fio >/dev/null 2>&1; then
    if [[ ! -f "$FIO_FILE" ]]; then
      log "preparing fio file $FIO_FILE size=$FIO_SIZE"
      fio \
        --name=prepare \
        --filename="$FIO_FILE" \
        --size="$FIO_SIZE" \
        --rw=write \
        --bs=1M \
        --iodepth=16 \
        --numjobs=1 \
        --direct=1 \
        --ioengine=libaio \
        --group_reporting \
        --end_fsync=1 \
        --output-format=json \
        --output="$RESULT_DIR/fio-prepare.json" \
        > "$RESULT_DIR/fio-prepare.stdout.txt" \
        2> "$RESULT_DIR/fio-prepare.stderr.txt"
      printf '%s\n' "$?" > "$RESULT_DIR/fio-prepare.exit"
    else
      log "reusing existing fio file $FIO_FILE"
      ls -lh "$FIO_FILE" > "$RESULT_DIR/fio-existing-file.txt" 2>&1 || true
    fi
    run_fio_json seqread read 1M 32
    run_fio_json randread4k randread 4k 32
    run_fio_json randread128k randread 128k 16
  else
    printf 'fio missing; install fio\n' > "$RESULT_DIR/fio-missing.txt"
  fi
fi

log "done"
REMOTE
}

failures=0
mapfile -t DEVICE_LIST < <(expand_devices "$DEVICES")

for alias in "${DEVICE_LIST[@]}"; do
  device_info="$(resolve_device "$alias")" || {
    failures=$((failures + 1))
    continue
  }
  ssh_host="${device_info%%$'\t'*}"
  test_dir="${device_info#*$'\t'}"
  if [[ -n "$GLOBAL_TEST_DIR" ]]; then
    test_dir="$GLOBAL_TEST_DIR"
  fi
  remote_result_dir="$test_dir/results/$RUN_ID"
  local_raw="$OUTPUT_DIR/raw/$alias"
  mkdir -p "$local_raw"

  env_cmd=(
    "DEVICE_ALIAS=$(shell_quote "$alias")"
    "SSH_HOST=$(shell_quote "$ssh_host")"
    "BENCH_TEST_DIR=$(shell_quote "$test_dir")"
    "RESULT_DIR=$(shell_quote "$remote_result_dir")"
    "FIO_SIZE=$(shell_quote "$FIO_SIZE")"
    "FIO_RUNTIME=$(shell_quote "$FIO_RUNTIME")"
    "MEM_SIZE=$(shell_quote "$MEM_SIZE")"
    "INSTALL_DEPS=$(shell_quote "$INSTALL_DEPS")"
    "SET_PERFORMANCE=$(shell_quote "$SET_PERFORMANCE")"
    "SKIP_FIO=$(shell_quote "$SKIP_FIO")"
    "SKIP_BW_MEM=$(shell_quote "$SKIP_BW_MEM")"
  )
  remote_cmd="${env_cmd[*]} bash -s"

  echo "==> [$alias] $ssh_host test_dir=$test_dir"
  set +e
  remote_payload | ssh "${SSH_OPTS[@]}" "$ssh_host" "$remote_cmd" 2>&1 | tee "$local_raw/ssh-run.log"
  ssh_status=${PIPESTATUS[1]}
  set -e
  printf '%s\n' "$ssh_status" > "$local_raw/ssh.exit"
  if [[ "$ssh_status" -ne 0 ]]; then
    echo "WARN: [$alias] ssh run failed with exit code $ssh_status" >&2
    failures=$((failures + 1))
    continue
  fi

  set +e
  scp -r "${SSH_OPTS[@]}" "$ssh_host:$remote_result_dir/." "$local_raw/" > "$local_raw/scp.log" 2>&1
  scp_status=$?
  set -e
  printf '%s\n' "$scp_status" > "$local_raw/scp.exit"
  if [[ "$scp_status" -ne 0 ]]; then
    echo "WARN: [$alias] scp failed with exit code $scp_status" >&2
    failures=$((failures + 1))
  fi
done

summary_path="$(generate_summary "$OUTPUT_DIR")"
echo "Summary: $summary_path"
echo "Raw results: $OUTPUT_DIR/raw"

if [[ "$failures" -ne 0 ]]; then
  echo "Completed with $failures device failure(s)." >&2
  exit 1
fi
