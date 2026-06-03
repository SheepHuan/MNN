#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  df_power_capture.sh -- <command> [args...]

Environment:
  MNN_POWER_API_URL        DF power API. Defaults to http://192.168.101.14:8000.
  MNN_POWER_DEVICE         jetson, orangepi5plus, oneplus13t, or explicit serial.
  MNN_POWER_SERIAL         Explicit DF power monitor serial. Overrides MNN_POWER_DEVICE.
  MNN_POWER_SAMPLE_RATE_HZ Defaults to 2000.
  MNN_POWER_VOLTAGE_MV     Defaults to 4200.
  MNN_POWER_MAX_DURATION_SEC Capture safety limit. Defaults to 3600.
  MNN_POWER_OUTPUT_CSV     Output CSV path.

Known device serials:
  jetson:        1A5D43
  orangepi5plus: C071EB951330
  oneplus13t:    F96FBDBA05B0
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

API_URL="${MNN_POWER_API_URL:-${API_URL:-http://192.168.101.14:8000}}"
SAMPLE_RATE_HZ="${MNN_POWER_SAMPLE_RATE_HZ:-${SAMPLE_RATE_HZ:-2000}}"
VOLTAGE_MV="${MNN_POWER_VOLTAGE_MV:-${VOLTAGE_MV:-4200}}"
MAX_DURATION_SEC="${MNN_POWER_MAX_DURATION_SEC:-${MAX_DURATION_SEC:-3600}}"
OUTPUT_CSV="${MNN_POWER_OUTPUT_CSV:-${OUTPUT_CSV:-df_pm_$(date +%Y%m%d_%H%M%S).csv}}"
POWER_DEVICE="${MNN_POWER_DEVICE:-${TARGET_DEVICE:-}}"
POWER_DEVICE="${POWER_DEVICE,,}"
POWER_DEVICE="${POWER_DEVICE//-/_}"

case "${POWER_DEVICE}" in
  jetson|nvidia_jetson)
    DEFAULT_SERIAL="1A5D43"
    ;;
  orangepi5plus|orange_pi_5_plus|orange_pi5_plus|opi5plus|opi_5_plus)
    DEFAULT_SERIAL="C071EB951330"
    ;;
  oneplus13t|oneplus_13t)
    DEFAULT_SERIAL="F96FBDBA05B0"
    ;;
  "")
    DEFAULT_SERIAL=""
    ;;
  *)
    DEFAULT_SERIAL="${POWER_DEVICE}"
    ;;
esac

DF_SERIAL="${MNN_POWER_SERIAL:-${DF_SERIAL:-${DEFAULT_SERIAL}}}"

tmp_dir="$(mktemp -d)"
serials_json="${tmp_dir}/serials.json"
start_json="${tmp_dir}/start.json"
stop_body="${tmp_dir}/stop.json"
stop_response="${tmp_dir}/stop_response.txt"
capture_started=0

cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

select_serial() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

path, requested = sys.argv[1:3]
with open(path, "r", encoding="utf-8") as fp:
    data = json.load(fp)

devices = data.get("devices", [])
available = [
    item for item in devices
    if item.get("type") == "df" and item.get("available") and item.get("serial")
]

if requested:
    for item in available:
        if item.get("serial") == requested:
            print(requested, end="")
            sys.exit(0)
    print("", end="")
    sys.exit(0)

print(available[0]["serial"] if available else "", end="")
PY
}

write_start_json() {
  python3 - "${DF_SERIAL}" "${SAMPLE_RATE_HZ}" "${VOLTAGE_MV}" "${MAX_DURATION_SEC}" > "${start_json}" <<'PY'
import json
import sys

serial, sample_rate_hz, voltage_mv, max_duration_sec = sys.argv[1:5]
print(json.dumps({
    "type": "df",
    "serial": serial,
    "sample_rate_hz": int(sample_rate_hz),
    "voltage_mv": int(voltage_mv),
    "max_duration_sec": float(max_duration_sec),
}))
PY
}

write_stop_json() {
  python3 - "${DF_SERIAL}" > "${stop_body}" <<'PY'
import json
import sys

print(json.dumps({
    "type": "df",
    "serial": sys.argv[1],
}))
PY
}

stop_capture() {
  local http_code

  if [[ "${capture_started}" != "1" ]]; then
    return 0
  fi

  write_stop_json
  mkdir -p "$(dirname "${OUTPUT_CSV}")"
  echo "[df-pm] stopping capture and downloading CSV: ${OUTPUT_CSV}"
  http_code="$(
    curl -sS \
      -X POST "${API_URL}/eperf/pm/stop" \
      -H "Content-Type: application/json" \
      --data-binary "@${stop_body}" \
      -o "${OUTPUT_CSV}" \
      -w "%{http_code}"
  )"

  capture_started=0

  if [[ "${http_code}" != "200" ]]; then
    mv "${OUTPUT_CSV}" "${stop_response}" 2>/dev/null || true
    echo "[df-pm] stop failed with HTTP ${http_code}" >&2
    cat "${stop_response}" >&2 2>/dev/null || true
    return 1
  fi
}

trap 'echo; echo "[df-pm] interrupted, stopping capture..."; stop_capture || true; exit 130' INT TERM

echo "[df-pm] querying devices: ${API_URL}/eperf/pm/serials"
curl -fsS "${API_URL}/eperf/pm/serials" -o "${serials_json}"

selected_serial="$(select_serial "${serials_json}" "${DF_SERIAL}")"
if [[ -z "${selected_serial}" ]]; then
  if [[ -n "${DF_SERIAL}" ]]; then
    echo "[df-pm] requested DF serial is not available: ${DF_SERIAL}" >&2
  else
    echo "[df-pm] no available DF power monitor found" >&2
  fi
  echo "[df-pm] raw devices:" >&2
  python3 -m json.tool "${serials_json}" >&2 || cat "${serials_json}" >&2
  exit 1
fi
DF_SERIAL="${selected_serial}"

write_start_json

echo "[df-pm] selected DF serial: ${DF_SERIAL}"
echo "[df-pm] starting capture: ${SAMPLE_RATE_HZ}Hz ${VOLTAGE_MV}mV max=${MAX_DURATION_SEC}s"
curl -fsS \
  -X POST "${API_URL}/eperf/pm/start" \
  -H "Content-Type: application/json" \
  --data-binary "@${start_json}"
echo
capture_started=1

set +e
"$@"
cmd_status=$?
set -e

stop_status=0
stop_capture || stop_status=$?

if [[ "${stop_status}" -eq 0 ]]; then
  echo "[df-pm] CSV saved: ${OUTPUT_CSV}"
fi

if [[ "${cmd_status}" -ne 0 ]]; then
  exit "${cmd_status}"
fi
exit "${stop_status}"
