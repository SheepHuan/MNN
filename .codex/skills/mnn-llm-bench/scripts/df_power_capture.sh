#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  df_power_capture.sh -- <command> [args...]

Environment:
  MNN_POWER_API_URL        eperf power API. Defaults to http://192.168.101.14:8766.
  MNN_POWER_DEVICE         jetson, orangepi5plus, oneplus13t, or explicit serial.
  MNN_POWER_SERIAL         Explicit power monitor serial. Overrides MNN_POWER_DEVICE.
  MNN_POWER_MONITOR_TYPE   Explicit monitor type, e.g. df or blu.
  MNN_POWER_PORT           Explicit monitor port.
  MNN_POWER_SAMPLE_RATE_HZ Optional sample rate. Defaults to API binding/device defaults.
  MNN_POWER_VOLTAGE_MV     Optional voltage. Defaults to API binding/device defaults.
  MNN_POWER_MAX_DURATION_SEC Capture safety limit. Defaults to 3600.
  MNN_POWER_OUTPUT_CSV     Output CSV path.

Known device bindings from /v1/power/list:
  jetson:        df  serial 1A5D43
  orangepi5plus: blu serial F96FBDBA05B0
  oneplus13t:    blu serial C071EB951330
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi

API_URL="${MNN_POWER_API_URL:-${API_URL:-http://192.168.101.14:8766}}"
SAMPLE_RATE_HZ="${MNN_POWER_SAMPLE_RATE_HZ:-${SAMPLE_RATE_HZ:-}}"
VOLTAGE_MV="${MNN_POWER_VOLTAGE_MV:-${VOLTAGE_MV:-}}"
MAX_DURATION_SEC="${MNN_POWER_MAX_DURATION_SEC:-${MAX_DURATION_SEC:-3600}}"
OUTPUT_CSV="${MNN_POWER_OUTPUT_CSV:-${OUTPUT_CSV:-power_$(date +%Y%m%d_%H%M%S).csv}}"
POWER_DEVICE_RAW="${MNN_POWER_DEVICE:-${TARGET_DEVICE:-}}"
POWER_DEVICE_NORM="$(printf '%s' "${POWER_DEVICE_RAW}" | tr '[:upper:]' '[:lower:]')"
POWER_DEVICE_NORM="${POWER_DEVICE_NORM//-/_}"

START_DEVICE=""
DEFAULT_SERIAL=""
DEFAULT_MONITOR_TYPE=""
case "${POWER_DEVICE_NORM}" in
  jetson|nvidia_jetson)
    START_DEVICE=""
    DEFAULT_MONITOR_TYPE="df"
    DEFAULT_SERIAL="1A5D43"
    ;;
  orangepi5plus|orange_pi_5_plus|orange_pi5_plus|opi5plus|opi_5_plus)
    START_DEVICE="orangepi5plus"
    ;;
  oneplus13t|oneplus_13t)
    START_DEVICE="oneplus13t"
    ;;
  "")
    ;;
  *)
    DEFAULT_SERIAL="${POWER_DEVICE_RAW}"
    ;;
esac

EXPLICIT_SERIAL="${MNN_POWER_SERIAL:-${DF_SERIAL:-}}"
POWER_SERIAL="${EXPLICIT_SERIAL:-${DEFAULT_SERIAL}}"
if [[ -n "${EXPLICIT_SERIAL}" ]]; then
  START_DEVICE=""
fi
MONITOR_TYPE="${MNN_POWER_MONITOR_TYPE:-${DEFAULT_MONITOR_TYPE}}"
POWER_PORT="${MNN_POWER_PORT:-}"

tmp_dir="$(mktemp -d)"
start_body="${tmp_dir}/start_body.json"
start_response="${tmp_dir}/start_response.json"
stop_body="${tmp_dir}/stop_body.json"
stop_response="${tmp_dir}/stop_response.json"
capture_started=0
SESSION_ID=""

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

write_start_json() {
  python3 - \
    "${START_DEVICE}" \
    "${POWER_SERIAL}" \
    "${MONITOR_TYPE}" \
    "${POWER_PORT}" \
    "${SAMPLE_RATE_HZ}" \
    "${VOLTAGE_MV}" \
    "${MAX_DURATION_SEC}" > "${start_body}" <<'PY'
import json
import sys

device, serial, monitor_type, port, sample_rate_hz, voltage_mv, max_duration_sec = sys.argv[1:8]
payload = {}
if device:
    payload["device"] = device
if serial:
    payload["serial"] = serial
if monitor_type:
    payload["monitor_type"] = monitor_type
if port:
    payload["port"] = port
if sample_rate_hz:
    payload["sample_rate_hz"] = int(sample_rate_hz)
if voltage_mv:
    payload["voltage_mv"] = int(voltage_mv)
if max_duration_sec:
    payload["max_duration_sec"] = float(max_duration_sec)
print(json.dumps(payload))
PY
}

write_stop_json() {
  python3 - "${SESSION_ID}" "${MNN_POWER_STOP_TIMEOUT_SEC:-30}" > "${stop_body}" <<'PY'
import json
import sys

print(json.dumps({
    "session_id": sys.argv[1],
    "timeout_sec": float(sys.argv[2]),
}))
PY
}

json_get() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

path, dotted_key = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as fp:
    value = json.load(fp)

for part in dotted_key.split("."):
    if isinstance(value, dict):
        value = value.get(part)
    else:
        value = None
    if value is None:
        break

if value is None:
    print("", end="")
elif isinstance(value, (dict, list)):
    print(json.dumps(value, ensure_ascii=False), end="")
else:
    print(value, end="")
PY
}

absolute_url() {
  python3 - "${API_URL}" "$1" <<'PY'
import sys
from urllib.parse import urljoin

base, path = sys.argv[1:3]
print(urljoin(base.rstrip("/") + "/", path), end="")
PY
}

post_json() {
  local url="$1"
  local body_path="$2"
  local out_path="$3"
  local http_code

  http_code="$(
    curl -sS \
      -X POST "${url}" \
      -H "Content-Type: application/json" \
      --data-binary "@${body_path}" \
      -o "${out_path}" \
      -w "%{http_code}"
  )"
  if [[ "${http_code}" != 2* ]]; then
    echo "[power-api] POST ${url} failed with HTTP ${http_code}" >&2
    cat "${out_path}" >&2 2>/dev/null || true
    return 1
  fi
}

stop_capture() {
  local raw_url download_url run_id csv_url monitor

  if [[ "${capture_started}" != "1" ]]; then
    return 0
  fi

  write_stop_json
  mkdir -p "$(dirname "${OUTPUT_CSV}")"
  echo "[power-api] stopping capture: session_id=${SESSION_ID}"
  post_json "${API_URL}/v1/power/stop" "${stop_body}" "${stop_response}" || return 1

  capture_started=0

  raw_url="$(json_get "${stop_response}" "artifacts.csv.raw_url")"
  download_url="$(json_get "${stop_response}" "artifacts.csv.download_url")"
  run_id="$(json_get "${stop_response}" "run_id")"
  if [[ -n "${raw_url}" ]]; then
    csv_url="$(absolute_url "${raw_url}")"
  elif [[ -n "${download_url}" ]]; then
    csv_url="$(absolute_url "${download_url}")"
  elif [[ -n "${run_id}" ]]; then
    csv_url="$(absolute_url "/v1/artifact/raw?run_id=${run_id}&path=power.csv")"
  else
    echo "[power-api] stop response did not include a CSV artifact URL" >&2
    cat "${stop_response}" >&2 2>/dev/null || true
    return 1
  fi

  echo "[power-api] downloading CSV: ${OUTPUT_CSV}"
  if ! curl -fsS "${csv_url}" -o "${OUTPUT_CSV}"; then
    echo "[power-api] CSV download failed: ${csv_url}" >&2
    return 1
  fi

  monitor="$(json_get "${stop_response}" "monitor")"
  if [[ -n "${monitor}" ]]; then
    echo "[power-api] monitor: ${monitor}"
  fi
}

trap 'echo; echo "[power-api] interrupted, stopping capture..."; stop_capture || true; exit 130' INT TERM

echo "[power-api] API: ${API_URL}"
write_start_json

echo "[power-api] start request:"
python3 -m json.tool "${start_body}" || cat "${start_body}"
post_json "${API_URL}/v1/power/start" "${start_body}" "${start_response}" || exit 1
SESSION_ID="$(json_get "${start_response}" "session_id")"
if [[ -z "${SESSION_ID}" ]]; then
  echo "[power-api] start response did not include session_id" >&2
  cat "${start_response}" >&2 2>/dev/null || true
  exit 1
fi
capture_started=1
echo "[power-api] started: session_id=${SESSION_ID} run_id=$(json_get "${start_response}" "run_id")"
echo "[power-api] monitor: $(json_get "${start_response}" "monitor")"

set +e
"$@"
cmd_status=$?
set -e

stop_status=0
stop_capture || stop_status=$?

if [[ "${stop_status}" -eq 0 ]]; then
  echo "[power-api] CSV saved: ${OUTPUT_CSV}"
fi

if [[ "${cmd_status}" -ne 0 ]]; then
  exit "${cmd_status}"
fi
exit "${stop_status}"
