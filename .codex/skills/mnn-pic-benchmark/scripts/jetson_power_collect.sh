#!/bin/sh
set -eu

CSV_PATH=${1:-}
INTERVAL_MS=${2:-50}

if [ -z "$CSV_PATH" ]; then
  echo "usage: $0 <csv_path> [interval_ms]" >&2
  exit 2
fi

case "$INTERVAL_MS" in
  ''|*[!0-9]*)
    echo "interval_ms must be an integer, got: $INTERVAL_MS" >&2
    exit 2
    ;;
esac

if ! command -v tegrastats >/dev/null 2>&1; then
  echo "tegrastats not found in PATH" >&2
  exit 127
fi

mkdir -p "$(dirname "$CSV_PATH")"
FIFO_PATH="${CSV_PATH}.fifo.$$"
mkfifo "$FIFO_PATH"

TEGRA_PID=
START_MS=$(date +%s%3N 2>/dev/null || awk 'BEGIN{srand(); printf("%d000\n", srand())}')

cleanup() {
  if [ -n "${TEGRA_PID:-}" ]; then
    kill "$TEGRA_PID" 2>/dev/null || true
    wait "$TEGRA_PID" 2>/dev/null || true
  fi
  rm -f "$FIFO_PATH"
}

trap 'cleanup; exit 0' INT TERM
trap 'cleanup' EXIT

printf 'timestamp_unix_ms,elapsed_ms,cpu_w,gpu_w,ddr_w\n' > "$CSV_PATH"

tegrastats --interval "$INTERVAL_MS" > "$FIFO_PATH" 2>&1 &
TEGRA_PID=$!

while IFS= read -r line; do
  now_ms=$(date +%s%3N 2>/dev/null || awk 'BEGIN{srand(); printf("%d000\n", srand())}')
  elapsed_ms=$((now_ms - START_MS))
  printf '%s\n' "$line" | awk -v now_ms="$now_ms" -v elapsed_ms="$elapsed_ms" '
    function watts(token, value) {
      gsub(/mW.*/, "", value)
      if (value ~ /^[0-9]+$/) {
        return sprintf("%.6f", value / 1000.0)
      }
      return ""
    }
    {
      cpu = ""; gpu = ""; ddr = ""
      for (i = 1; i < NF; i++) {
        if ($i == "CPU") {
          cpu = watts($i, $(i + 1))
        } else if ($i == "GPU") {
          gpu = watts($i, $(i + 1))
        } else if ($i == "VDDRQ") {
          ddr = watts($i, $(i + 1))
        }
      }
      printf("%s,%s,%s,%s,%s\n", now_ms, elapsed_ms, cpu, gpu, ddr)
    }' >> "$CSV_PATH"
done < "$FIFO_PATH"
