#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MNN_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
KVSHARE_ROOT_DEFAULT="$(cd "${MNN_ROOT}/../.." && pwd)"

timestamp() {
  date +%Y%m%d_%H%M%S
}

log() {
  printf '[mnn-pic-formal] %s\n' "$*"
}

die() {
  printf '[mnn-pic-formal] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash .codex/skills/mnn-pic-benchmark/scripts/run_formal_mnn_pic_dataset_bench.sh [options] -- [pic_bench args]

Formal dataset bench wrapper. By default it runs both formal devices:
  jetson,orangepi

Options:
  --devices CSV               Default: jetson,orangepi
  --run-id-prefix ID          Default: mnn_pic_formal_<timestamp>
  --allow-subset-devices      Allow targeted non-formal subsets.
  --allow-extra-device        Allow extra-profile devices such as rhino.
  --skip-build                Pass through to each single-device run.
  --skip-rsync                Pass through to each single-device run.
  --line-buffer               Pass through to each single-device run.
  --ready-timeout-sec SEC     Pass through to each single-device run.
  --kvshare-root PATH         Default: from KVSHARE_ROOT or repo root.
  --conda-env NAME            Pass through to each single-device run.
  --python-bin PATH           Pass through to each single-device run.
  --bench-command CMD         Pass through to each single-device run.
EOF
}

FORMAL_REQUIRED_DEVICES=(jetson orangepi)
EXTRA_PROFILE_DEVICES=(rhino)
FORMAL_REQUIRED_DEVICES_CSV="jetson,orangepi"
EXTRA_PROFILE_DEVICES_CSV="rhino"

DEVICE_SET="${FORMAL_REQUIRED_DEVICES_CSV}"
RUN_ID_PREFIX="mnn_pic_formal_$(timestamp)"
ALLOW_SUBSET_DEVICES="0"
ALLOW_EXTRA_DEVICE="0"
SKIP_BUILD="0"
SKIP_RSYNC="0"
LINE_BUFFER="0"
READY_TIMEOUT_SEC=""
KVSHARE_ROOT="${KVSHARE_ROOT:-${KVSHARE_ROOT_DEFAULT}}"
CONDA_ENV=""
PYTHON_BIN=""
BENCH_COMMAND=""
BENCH_ARGS=()

while (($#)); do
  case "$1" in
    --devices)
      DEVICE_SET="$2"; shift 2 ;;
    --run-id-prefix)
      RUN_ID_PREFIX="$2"; shift 2 ;;
    --allow-subset-devices)
      ALLOW_SUBSET_DEVICES="1"; shift ;;
    --allow-extra-device)
      ALLOW_EXTRA_DEVICE="1"; shift ;;
    --skip-build)
      SKIP_BUILD="1"; shift ;;
    --skip-rsync)
      SKIP_RSYNC="1"; shift ;;
    --line-buffer)
      LINE_BUFFER="1"; shift ;;
    --ready-timeout-sec)
      READY_TIMEOUT_SEC="$2"; shift 2 ;;
    --kvshare-root)
      KVSHARE_ROOT="$2"; shift 2 ;;
    --conda-env)
      CONDA_ENV="$2"; shift 2 ;;
    --python-bin)
      PYTHON_BIN="$2"; shift 2 ;;
    --bench-command)
      BENCH_COMMAND="$2"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    --)
      shift
      BENCH_ARGS=("$@")
      break ;;
    *)
      die "unknown argument: $1" ;;
  esac
done

parse_devices() {
  local text="$1"
  local seen=" "
  local parsed=()
  local device=""
  IFS=',' read -r -a raw_devices <<< "${text}"
  for device in "${raw_devices[@]}"; do
    device="${device// /}"
    [[ -n "${device}" ]] || continue
    case "${device}" in
      jetson|orangepi|rhino) ;;
      *) die "unknown device ${device}; expected one of: ${FORMAL_REQUIRED_DEVICES_CSV},${EXTRA_PROFILE_DEVICES_CSV}" ;;
    esac
    if [[ "${seen}" != *" ${device} "* ]]; then
      parsed+=("${device}")
      seen="${seen}${device} "
    fi
  done
  ((${#parsed[@]} > 0)) || die "no devices selected"
  printf '%s\n' "${parsed[@]}"
}

mapfile -t DEVICES < <(parse_devices "${DEVICE_SET}")

if [[ "${ALLOW_EXTRA_DEVICE}" != "1" ]]; then
  for device in "${DEVICES[@]}"; do
    if [[ "${device}" == "rhino" ]]; then
      die "extra-profile devices require --allow-extra-device: rhino"
    fi
  done
fi

if [[ "${ALLOW_SUBSET_DEVICES}" != "1" ]]; then
  for required in "${FORMAL_REQUIRED_DEVICES[@]}"; do
    found="0"
    for device in "${DEVICES[@]}"; do
      if [[ "${device}" == "${required}" ]]; then
        found="1"
        break
      fi
    done
    [[ "${found}" == "1" ]] || die "formal dataset bench requires devices ${FORMAL_REQUIRED_DEVICES_CSV}; use --allow-subset-devices only for targeted debug/profile runs"
  done
fi

LOG_DIR="${KVSHARE_ROOT}/.cache/mnn-pic-benchmark/formal_runs/${RUN_ID_PREFIX}"
RUN_ENV="${LOG_DIR}/run.env"
mkdir -p "${LOG_DIR}"

overall_exit=0
completed_devices=()

for device in "${DEVICES[@]}"; do
  run_id="${RUN_ID_PREFIX}_${device}"
  cmd=(
    bash "${SCRIPT_DIR}/run_mnn_pic_dataset_bench.sh"
    --device-key "${device}"
    --run-id "${run_id}"
  )
  [[ "${SKIP_BUILD}" == "1" ]] && cmd+=(--skip-build)
  [[ "${SKIP_RSYNC}" == "1" ]] && cmd+=(--skip-rsync)
  [[ "${LINE_BUFFER}" == "1" ]] && cmd+=(--line-buffer)
  [[ -n "${READY_TIMEOUT_SEC}" ]] && cmd+=(--ready-timeout-sec "${READY_TIMEOUT_SEC}")
  [[ -n "${KVSHARE_ROOT}" ]] && cmd+=(--kvshare-root "${KVSHARE_ROOT}")
  [[ -n "${CONDA_ENV}" ]] && cmd+=(--conda-env "${CONDA_ENV}")
  [[ -n "${PYTHON_BIN}" ]] && cmd+=(--python-bin "${PYTHON_BIN}")
  [[ -n "${BENCH_COMMAND}" ]] && cmd+=(--bench-command "${BENCH_COMMAND}")
  if [[ "${device}" == "rhino" ]]; then
    cmd+=(--allow-extra-device)
  fi
  if ((${#BENCH_ARGS[@]} > 0)); then
    cmd+=(-- "${BENCH_ARGS[@]}")
  fi
  log "running ${device} with run_id=${run_id}"
  if "${cmd[@]}"; then
    completed_devices+=("${device}")
  else
    overall_exit=1
    log "device ${device} failed"
  fi
done

formal_matrix_complete="0"
formal_covered="1"
for required in "${FORMAL_REQUIRED_DEVICES[@]}"; do
  found="0"
  for device in "${completed_devices[@]}"; do
    if [[ "${device}" == "${required}" ]]; then
      found="1"
      break
    fi
  done
  if [[ "${found}" != "1" ]]; then
    formal_covered="0"
    break
  fi
done
if [[ "${formal_covered}" == "1" && "${overall_exit}" == "0" ]]; then
  formal_matrix_complete="1"
fi

{
  printf 'run_id_prefix=%s\n' "${RUN_ID_PREFIX}"
  printf 'requested_devices=%s\n' "$(IFS=,; printf '%s' "${DEVICES[*]}")"
  printf 'completed_devices=%s\n' "$(IFS=,; printf '%s' "${completed_devices[*]}")"
  printf 'formal_required_devices=%s\n' "${FORMAL_REQUIRED_DEVICES_CSV}"
  printf 'extra_profile_devices=%s\n' "${EXTRA_PROFILE_DEVICES_CSV}"
  printf 'allow_subset_devices=%s\n' "${ALLOW_SUBSET_DEVICES}"
  printf 'allow_extra_device=%s\n' "${ALLOW_EXTRA_DEVICE}"
  printf 'formal_matrix_complete=%s\n' "${formal_matrix_complete}"
  printf 'exit_code=%s\n' "${overall_exit}"
} > "${RUN_ENV}"

exit "${overall_exit}"
