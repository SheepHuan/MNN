#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MNN_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
KVSHARE_ROOT_DEFAULT="$(cd "${MNN_ROOT}/../.." && pwd)"
TOTAL_CPUS="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 1)"
DEFAULT_JOBS="$(( TOTAL_CPUS > 1 ? TOTAL_CPUS / 2 : 1 ))"

timestamp() {
  date +%Y%m%d_%H%M%S
}

log() {
  printf '[mnn-pic-bench] %s\n' "$*"
}

die() {
  printf '[mnn-pic-bench] ERROR: %s\n' "$*" >&2
  exit 1
}

kill_local_port_users() {
  local port="$1"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -tiTCP:${port} -sTCP:LISTEN 2>/dev/null | tr '\n' ' ')"
  elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser -n tcp "${port}" 2>/dev/null | tr '\n' ' ')"
  elif command -v ss >/dev/null 2>&1; then
    pids="$(ss -ltnpH "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | tr '\n' ' ')"
  fi
  if [[ -n "${pids// /}" ]]; then
    log "killing local tcp/${port} listeners: ${pids}"
    kill ${pids} >/dev/null 2>&1 || true
    sleep 1
    kill -9 ${pids} >/dev/null 2>&1 || true
    sleep 1
  fi
}

kill_remote_port_users() {
  local port="$1"
  ssh "${REMOTE}" "bash -lc '
set +e
port=${port}
collect_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:\$port -sTCP:LISTEN 2>/dev/null
    return 0
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -n tcp \$port 2>/dev/null | tr \" \" \"\n\" | sed \"/^$/d\"
    return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltnpH \"sport = :\$port\" 2>/dev/null | sed -n \"s/.*pid=\\([0-9]\\+\\).*/\\1/p\"
    return 0
  fi
  pgrep -f \"pic_server.*--port \$port\" 2>/dev/null
}
pids=\"\$(collect_pids | sort -u | tr \"\n\" \" \")\"
if [[ -n \"\${pids// /}\" ]]; then
  kill \$pids >/dev/null 2>&1 || true
  sleep 1
  pids=\"\$(collect_pids | sort -u | tr \"\n\" \" \")\"
  if [[ -n \"\${pids// /}\" ]]; then
    kill -9 \$pids >/dev/null 2>&1 || true
    sleep 1
  fi
fi
' " >/dev/null 2>&1 || true
}

usage() {
  cat <<'EOF'
Usage:
  bash .codex/skills/mnn-pic-benchmark/scripts/run_mnn_pic_dataset_bench.sh [options] -- [pic_bench args]

Single-device helper for MNN PIC dataset bench. Formal regression still requires
running it once for Jetson and once for OrangePi. Rhino is extra-profile only.

Common:
  --device-key jetson|orangepi|rhino
                                Built-in device preset. Default: jetson
  --allow-extra-device          Allow extra-profile preset rhino.
  --skip-build                  Do not build locally.
  --skip-rsync                  Do not rsync artifact root.
  --remote USER@HOST            Override preset remote host.
  --remote-repo PATH            Override preset remote work/repo path.
  --port PORT                   Remote pic_server port. Default: fixed per device preset
  --local-port PORT             Local SSH tunnel port. Default: same as --port
  --run-id ID                   Run id for logs/output.
  --remote-config PATH          Override preset remote model config path.
  --remote-log-dir PATH         Override preset remote log/PID directory.
  --remote-ld-library-path PATH Exact remote LD_LIBRARY_PATH for pic_server.
  --remote-server-env ENV       Extra env assignments before pic_server, e.g. LD_PRELOAD=/usr/lib/libOpenCL_adreno.so.
  --line-buffer                 Start remote server through stdbuf when available.
  --bench-command run|compare   impl/pic_bench/cli.py subcommand. Default: run

Everything after -- is forwarded to impl/pic_bench/cli.py.
EOF
}

FORMAL_REQUIRED_DEVICES=(jetson orangepi)
EXTRA_PROFILE_DEVICES=(rhino)
FORMAL_REQUIRED_DEVICES_CSV="jetson,orangepi"
EXTRA_PROFILE_DEVICES_CSV="rhino"

default_remote_for_device() {
  case "$1" in
    jetson) printf '%s' 'jetson@192.168.101.192' ;;
    orangepi) printf '%s' 'orangepi@192.168.101.113' ;;
    rhino) printf '%s' 'aidlux@192.168.101.227' ;;
    *) die "unsupported device preset: $1" ;;
  esac
}

default_remote_repo_for_device() {
  case "$1" in
    jetson) printf '%s' '/home/jetson/code/kvshare-edge/impl/MNN' ;;
    orangepi) printf '%s' '/mnt/ssd/code/.cache/mnn_opencl_pic' ;;
    rhino) printf '%s' '/mnt/nvme/mnn_pic_opencl' ;;
    *) die "unsupported device preset: $1" ;;
  esac
}

default_remote_cuda_lib_for_device() {
  case "$1" in
    jetson) printf '%s' '/usr/local/cuda-12.2/targets/aarch64-linux/lib' ;;
    orangepi|rhino) printf '%s' '' ;;
    *) die "unsupported device preset: $1" ;;
  esac
}

default_port_for_device() {
  case "$1" in
    jetson) printf '%s' '18096' ;;
    orangepi) printf '%s' '18097' ;;
    rhino) printf '%s' '18098' ;;
    *) die "unsupported device preset: $1" ;;
  esac
}

default_build_dir_for_device() {
  case "$1" in
    jetson) printf '%s' "${MNN_ROOT}/.cache/build/mnn/jetson_cross_cuda" ;;
    orangepi) printf '%s' "${MNN_ROOT}/.cache/build/mnn/orangepi5plus" ;;
    rhino) printf '%s' "${MNN_ROOT}/.cache/build/mnn/aidlux_adreno_opencl" ;;
    *) die "unsupported device preset: $1" ;;
  esac
}

default_install_prefix_for_device() {
  case "$1" in
    jetson) printf '%s' "${MNN_ROOT}/.cache/output/mnn/artifacts/jetson_cross_cuda" ;;
    orangepi) printf '%s' "${MNN_ROOT}/.cache/output/mnn/artifacts/orangepi5plus" ;;
    rhino) printf '%s' "${MNN_ROOT}/.cache/output/mnn/artifacts/aidlux_adreno_opencl" ;;
    *) die "unsupported device preset: $1" ;;
  esac
}

default_remote_art_rel_for_device() {
  case "$1" in
    jetson) printf '%s' '.cache/output/mnn/artifacts/jetson_cross_cuda' ;;
    orangepi) printf '%s' 'artifacts/orangepi5plus' ;;
    rhino) printf '%s' 'artifacts/aidlux_adreno_opencl' ;;
    *) die "unsupported device preset: $1" ;;
  esac
}

default_remote_config_for_device() {
  case "$1" in
    jetson) printf '%s' '/home/jetson/code/kvshare-edge/impl/MNN/.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_cuda_greedy.json' ;;
    orangepi) printf '%s' '/mnt/ssd/code/.cache/mnn_opencl_pic/models/pic/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_opencl_greedy.json' ;;
    rhino) printf '%s' '/mnt/nvme/mnn_pic_opencl/models/pic/AI-ModelScope__Llama-3___2-3B-Instruct-pic-boundary/config_opencl_greedy.json' ;;
    *) die "unsupported device preset: $1" ;;
  esac
}

default_remote_kv_dir_for_device() {
  local device="$1"
  local run_id="$2"
  case "${device}" in
    jetson) printf '%s' "/home/jetson/code/kvshare-edge/impl/MNN/.cache/kvshare/mnn_pic_dataset_bench_${run_id}" ;;
    orangepi) printf '%s' "/mnt/ssd/code/.cache/mnn_opencl_pic/mnn_pic_dataset_bench_${run_id}" ;;
    rhino) printf '%s' "/mnt/nvme/mnn_pic_opencl/cache/mnn_pic_dataset_bench_${run_id}" ;;
    *) die "unsupported device preset: ${device}" ;;
  esac
}

default_remote_log_dir_for_device() {
  case "$1" in
    jetson) printf '%s' '/home/jetson/code/kvshare-edge/impl/MNN/.cache/logs' ;;
    orangepi) printf '%s' '/mnt/ssd/code/.cache/mnn_opencl_pic/logs' ;;
    rhino) printf '%s' '/mnt/nvme/mnn_pic_opencl/logs' ;;
    *) die "unsupported device preset: $1" ;;
  esac
}

default_remote_server_env_for_device() {
  case "$1" in
    rhino) printf '%s' 'LD_PRELOAD=/usr/lib/libOpenCL_adreno.so' ;;
    jetson|orangepi) printf '%s' '' ;;
    *) die "unsupported device preset: $1" ;;
  esac
}

default_remote_ld_library_path_for_device() {
  local device="$1"
  local remote_art="$2"
  local remote_cuda_lib="$3"
  case "${device}" in
    jetson) printf '%s' "${remote_art}/lib${remote_cuda_lib:+:${remote_cuda_lib}}" ;;
    orangepi) printf '%s' "${remote_art}/lib" ;;
    rhino) printf '%s' "${remote_art}/lib:/usr/lib:/usr/lib/aarch64-linux-gnu" ;;
    *) die "unsupported device preset: ${device}" ;;
  esac
}

device_role_for_key() {
  case "$1" in
    jetson|orangepi) printf '%s' 'formal' ;;
    rhino) printf '%s' 'extra-profile' ;;
    *) die "unsupported device preset: $1" ;;
  esac
}

validate_device_key() {
  case "$1" in
    jetson|orangepi) return 0 ;;
    rhino)
      [[ "${ALLOW_EXTRA_DEVICE}" == "1" ]] || die "rhino is extra-profile only; use --allow-extra-device for targeted debug/profile runs"
      return 0
      ;;
    *)
      die "unknown --device-key: $1 (expected one of: ${FORMAL_REQUIRED_DEVICES_CSV},${EXTRA_PROFILE_DEVICES_CSV})"
      ;;
  esac
}

DEVICE_KEY="jetson"
ALLOW_EXTRA_DEVICE="0"
REMOTE=""
REMOTE_REPO=""
REMOTE_CUDA_LIB=""
REMOTE_HOST="127.0.0.1"
REMOTE_PORT=""
LOCAL_HOST="127.0.0.1"
LOCAL_PORT=""
SERVED_MODEL="llama-pic"
RUN_ID="mnn_pic_dataset_$(timestamp)"
BUILD_DIR=""
INSTALL_PREFIX=""
REMOTE_ART_REL=""
REMOTE_CONFIG=""
REMOTE_KV_DIR=""
REMOTE_LOG_DIR=""
REMOTE_LD_LIBRARY_PATH=""
REMOTE_SERVER_ENV=""
REMOTE_LINE_BUFFER="0"
SKIP_BUILD="0"
SKIP_RSYNC="0"
READY_TIMEOUT_SEC="180"
KVSHARE_ROOT="${KVSHARE_ROOT:-${KVSHARE_ROOT_DEFAULT}}"
CONDA_ENV="kvshare-edge"
PYTHON_BIN="${PYTHON_BIN:-}"
BENCH_COMMAND="run"
BENCH_ARGS=()

while (($#)); do
  case "$1" in
    --device-key)
      DEVICE_KEY="$2"; shift 2 ;;
    --allow-extra-device)
      ALLOW_EXTRA_DEVICE="1"; shift ;;
    --remote)
      REMOTE="$2"; shift 2 ;;
    --remote-repo)
      REMOTE_REPO="$2"; shift 2 ;;
    --remote-cuda-lib)
      REMOTE_CUDA_LIB="$2"; shift 2 ;;
    --remote-host)
      REMOTE_HOST="$2"; shift 2 ;;
    --port)
      REMOTE_PORT="$2"; shift 2 ;;
    --local-host)
      LOCAL_HOST="$2"; shift 2 ;;
    --local-port)
      LOCAL_PORT="$2"; shift 2 ;;
    --served-model)
      SERVED_MODEL="$2"; shift 2 ;;
    --run-id)
      RUN_ID="$2"; shift 2 ;;
    --build-dir)
      BUILD_DIR="$2"; shift 2 ;;
    --install-prefix)
      INSTALL_PREFIX="$2"; shift 2 ;;
    --remote-art-rel)
      REMOTE_ART_REL="$2"; shift 2 ;;
    --remote-config)
      REMOTE_CONFIG="$2"; shift 2 ;;
    --remote-kv-dir)
      REMOTE_KV_DIR="$2"; shift 2 ;;
    --remote-log-dir)
      REMOTE_LOG_DIR="$2"; shift 2 ;;
    --remote-ld-library-path)
      REMOTE_LD_LIBRARY_PATH="$2"; shift 2 ;;
    --remote-server-env)
      REMOTE_SERVER_ENV="$2"; shift 2 ;;
    --line-buffer)
      REMOTE_LINE_BUFFER="1"; shift ;;
    --skip-build)
      SKIP_BUILD="1"; shift ;;
    --skip-rsync)
      SKIP_RSYNC="1"; shift ;;
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

validate_device_key "${DEVICE_KEY}"
DEVICE_ROLE="$(device_role_for_key "${DEVICE_KEY}")"
REMOTE="${REMOTE:-$(default_remote_for_device "${DEVICE_KEY}")}"
REMOTE_REPO="${REMOTE_REPO:-$(default_remote_repo_for_device "${DEVICE_KEY}")}"
REMOTE_CUDA_LIB="${REMOTE_CUDA_LIB:-$(default_remote_cuda_lib_for_device "${DEVICE_KEY}")}"
REMOTE_PORT="${REMOTE_PORT:-$(default_port_for_device "${DEVICE_KEY}")}"
LOCAL_PORT="${LOCAL_PORT:-${REMOTE_PORT}}"
BUILD_DIR="${BUILD_DIR:-$(default_build_dir_for_device "${DEVICE_KEY}")}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$(default_install_prefix_for_device "${DEVICE_KEY}")}"
REMOTE_ART_REL="${REMOTE_ART_REL:-$(default_remote_art_rel_for_device "${DEVICE_KEY}")}"
REMOTE_ART="${REMOTE_REPO}/${REMOTE_ART_REL}"
REMOTE_CONFIG="${REMOTE_CONFIG:-$(default_remote_config_for_device "${DEVICE_KEY}")}"
REMOTE_KV_DIR="${REMOTE_KV_DIR:-$(default_remote_kv_dir_for_device "${DEVICE_KEY}" "${RUN_ID}")}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-$(default_remote_log_dir_for_device "${DEVICE_KEY}")}"
REMOTE_SERVER_ENV="${REMOTE_SERVER_ENV:-$(default_remote_server_env_for_device "${DEVICE_KEY}")}"
REMOTE_LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH:-$(default_remote_ld_library_path_for_device "${DEVICE_KEY}" "${REMOTE_ART}" "${REMOTE_CUDA_LIB}")}"
REMOTE_LOG="${REMOTE_LOG_DIR}/mnn_pic_dataset_bench_${RUN_ID}.log"
REMOTE_PID_FILE="${REMOTE_LOG_DIR}/mnn_pic_dataset_bench_${RUN_ID}.pid"
LOG_DIR="${KVSHARE_ROOT}/.cache/mnn-pic-benchmark/logs/${RUN_ID}"
RUN_OUTPUT_DIR="${KVSHARE_ROOT}/.cache/mnn-pic-benchmark/runs/${RUN_ID}"
BENCH_LOG="${LOG_DIR}/bench.log"
TUNNEL_LOG="${LOG_DIR}/ssh_tunnel.log"
RUN_ENV="${LOG_DIR}/run.env"
BASE_URL="http://${LOCAL_HOST}:${LOCAL_PORT}"

mkdir -p "${LOG_DIR}" "${RUN_OUTPUT_DIR}"

REMOTE_SERVER_STARTED="0"
TUNNEL_PID=""

log "dataset bench preset: ${DEVICE_KEY} (${DEVICE_ROLE}); formal required devices: ${FORMAL_REQUIRED_DEVICES_CSV}"

cleanup() {
  local exit_code=$?
  if [[ -n "${TUNNEL_PID}" ]]; then
    kill "${TUNNEL_PID}" >/dev/null 2>&1 || true
    wait "${TUNNEL_PID}" >/dev/null 2>&1 || true
  fi
  kill_local_port_users "${LOCAL_PORT}"
  if [[ "${REMOTE_SERVER_STARTED}" == "1" ]]; then
    ssh "${REMOTE}" "if [[ -f '${REMOTE_PID_FILE}' ]]; then kill \$(cat '${REMOTE_PID_FILE}') >/dev/null 2>&1 || true; fi" >/dev/null 2>&1 || true
    kill_remote_port_users "${REMOTE_PORT}"
  fi
  {
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'device_key=%s\n' "${DEVICE_KEY}"
    printf 'device_role=%s\n' "${DEVICE_ROLE}"
    printf 'formal_required_devices=%s\n' "${FORMAL_REQUIRED_DEVICES_CSV}"
    printf 'extra_profile_devices=%s\n' "${EXTRA_PROFILE_DEVICES_CSV}"
    printf 'formal_matrix_device_covered=%s\n' "$([[ "${DEVICE_ROLE}" == "formal" ]] && printf '1' || printf '0')"
    printf 'formal_matrix_complete=0\n'
    printf 'mnn_root=%s\n' "${MNN_ROOT}"
    printf 'kvshare_root=%s\n' "${KVSHARE_ROOT}"
    printf 'remote=%s\n' "${REMOTE}"
    printf 'remote_repo=%s\n' "${REMOTE_REPO}"
    printf 'remote_art=%s\n' "${REMOTE_ART}"
    printf 'remote_config=%s\n' "${REMOTE_CONFIG}"
    printf 'remote_kv_dir=%s\n' "${REMOTE_KV_DIR}"
    printf 'remote_log_dir=%s\n' "${REMOTE_LOG_DIR}"
    printf 'remote_ld_library_path=%s\n' "${REMOTE_LD_LIBRARY_PATH}"
    printf 'remote_server_env=%s\n' "${REMOTE_SERVER_ENV}"
    printf 'remote_log=%s\n' "${REMOTE_LOG}"
    printf 'base_url=%s\n' "${BASE_URL}"
    printf 'run_output_dir=%s\n' "${RUN_OUTPUT_DIR}"
    printf 'bench_log=%s\n' "${BENCH_LOG}"
    printf 'exit_code=%s\n' "${exit_code}"
  } > "${RUN_ENV}"
}
trap cleanup EXIT

if [[ "${SKIP_BUILD}" != "1" ]]; then
  log "building local ${DEVICE_KEY} pic_server artifacts"
  (
    cd "${MNN_ROOT}"
    case "${DEVICE_KEY}" in
      jetson)
        export MNN_TARGET_DEVICE="${MNN_TARGET_DEVICE:-jetson}"
        export CUDA_ARCHS="${CUDA_ARCHS:-72}"
        export ENABLE_CROSS_CUDA="${ENABLE_CROSS_CUDA:-ON}"
        if [[ -z "${CUDA_TOOLKIT_ROOT:-}" && -d "${MNN_ROOT}/.cache/sysroots/jetson_cuda" ]]; then
          export CUDA_TOOLKIT_ROOT="${MNN_ROOT}/.cache/sysroots/jetson_cuda"
        fi
        if [[ -z "${CUDA_NVCC_EXECUTABLE:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
          export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc
        fi
        if [[ -z "${CMAKE_ARGS:-}" && -x "${MNN_ROOT}/.cache/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++" ]]; then
          export CMAKE_ARGS="-DCUDA_HOST_COMPILER=${MNN_ROOT}/.cache/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++"
        fi
        ;;
      orangepi)
        export MNN_TARGET_DEVICE="${MNN_TARGET_DEVICE:-orangepi5plus}"
        ;;
      rhino)
        export MNN_TARGET_DEVICE="${MNN_TARGET_DEVICE:-aidlux_adreno_opencl}"
        ;;
      *)
        die "unsupported device preset: ${DEVICE_KEY}"
        ;;
    esac
    export BUILD_DIR
    export INSTALL_PREFIX
    export JOBS="${JOBS:-${DEFAULT_JOBS}}"
    export BUILD_TARGET="${BUILD_TARGET:-pic_server}"
    export BUILD_MNNCONVERT="${BUILD_MNNCONVERT:-0}"
    export INSTALL_AFTER_BUILD="${INSTALL_AFTER_BUILD:-1}"
    bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
  )
fi

if [[ "${SKIP_RSYNC}" != "1" ]]; then
  log "rsync artifact root to ${REMOTE}:${REMOTE_REPO}/${REMOTE_ART_REL}/"
  rsync -a --delete "${INSTALL_PREFIX}/" "${REMOTE}:${REMOTE_REPO}/${REMOTE_ART_REL}/"
fi

kill_local_port_users "${LOCAL_PORT}"
kill_remote_port_users "${REMOTE_PORT}"

log "starting remote MNN pic_server on ${REMOTE}:${REMOTE_PORT}"
ssh "${REMOTE}" "cd '${REMOTE_REPO}' && \
  mkdir -p '${REMOTE_LOG_DIR}' && \
  rm -rf '${REMOTE_KV_DIR}' && \
  LAUNCHER='' && \
  if [[ '${REMOTE_LINE_BUFFER}' == '1' ]] && command -v stdbuf >/dev/null 2>&1; then LAUNCHER='stdbuf -oL -eL'; fi && \
  nohup \${LAUNCHER} env ${REMOTE_SERVER_ENV} LD_LIBRARY_PATH='${REMOTE_LD_LIBRARY_PATH}' \
    '${REMOTE_ART}/bin/pic_server' --config '${REMOTE_CONFIG}' --host '${REMOTE_HOST}' --port '${REMOTE_PORT}' \
    --kv-cache-dir '${REMOTE_KV_DIR}' --model '${SERVED_MODEL}' > '${REMOTE_LOG}' 2>&1 & \
  echo \$! > '${REMOTE_PID_FILE}'"
REMOTE_SERVER_STARTED="1"

log "waiting for remote /healthz"
ready="0"
for _ in $(seq 1 "${READY_TIMEOUT_SEC}"); do
  if ssh "${REMOTE}" "curl -fsS 'http://${REMOTE_HOST}:${REMOTE_PORT}/healthz' >/dev/null" >/dev/null 2>&1; then
    ready="1"
    break
  fi
  sleep 1
done
if [[ "${ready}" != "1" ]]; then
  ssh "${REMOTE}" "tail -n 100 '${REMOTE_LOG}'" || true
  die "remote pic_server did not become ready within ${READY_TIMEOUT_SEC}s"
fi

log "opening SSH tunnel ${LOCAL_HOST}:${LOCAL_PORT} -> ${REMOTE_HOST}:${REMOTE_PORT}"
ssh -N -L "${LOCAL_HOST}:${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT}" \
  -o ExitOnForwardFailure=yes "${REMOTE}" >"${TUNNEL_LOG}" 2>&1 &
TUNNEL_PID="$!"
sleep 1
if ! curl -fsS "${BASE_URL}/healthz" >/dev/null; then
  tail -n 80 "${TUNNEL_LOG}" || true
  die "local SSH tunnel is not ready: ${BASE_URL}"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="/root/miniconda3/envs/${CONDA_ENV}/bin/python"
fi
BENCH_CMD=()
if [[ -x "${PYTHON_BIN}" ]]; then
  BENCH_CMD=("${PYTHON_BIN}" impl/pic_bench/cli.py "${BENCH_COMMAND}")
else
  BENCH_CMD=(conda run -n "${CONDA_ENV}" python impl/pic_bench/cli.py "${BENCH_COMMAND}")
fi
BENCH_CMD+=(--base-url "${BASE_URL}" --output-dir "${RUN_OUTPUT_DIR}")

if ((${#BENCH_ARGS[@]} == 0)); then
  BENCH_ARGS=(
    --dataset hotpotqa
    --mode pic_cache_reuse
    --phase both
    --cases 20
    --context-len 1500
    --max-tokens 64
    --temperature 0.0
    --local-files-only
    --min-doc-tokens 0
    --force-cache
    --reset-before-each-infer
    --pic-selection-algorithm kvshare
    --pic-recompute-ratio 0.20
    --pic-recompute-score-layer-idx 1
  )
fi
BENCH_CMD+=("${BENCH_ARGS[@]}")

log "running local dataset bench against ${BASE_URL}"
set +e
(
  cd "${KVSHARE_ROOT}"
  "${BENCH_CMD[@]}"
) 2>&1 | tee "${BENCH_LOG}"
bench_exit="${PIPESTATUS[0]}"
set -e
if [[ "${bench_exit}" != "0" ]]; then
  die "dataset bench failed with exit code ${bench_exit}"
fi

log "dataset bench finished"
find "${RUN_OUTPUT_DIR}" -maxdepth 2 -type f | sort | sed "s#^#  #"
