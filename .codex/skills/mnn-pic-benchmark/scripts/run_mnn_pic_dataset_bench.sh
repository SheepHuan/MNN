#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MNN_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
KVSHARE_ROOT_DEFAULT="$(cd "${MNN_ROOT}/../.." && pwd)"

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

usage() {
  cat <<'EOF'
Usage:
  bash .codex/skills/mnn-pic-benchmark/scripts/run_mnn_pic_dataset_bench.sh [options] -- [pic_bench args]

Build local MNN Jetson-cross pic_server artifacts, rsync them to Jetson, start
remote MNN pic_server, open an SSH tunnel, and run local impl/pic_bench/cli.py.

Common:
  --skip-build                  Do not build locally.
  --skip-rsync                  Do not rsync artifact root.
  --remote USER@HOST            Default: jetson@192.168.101.192
  --remote-repo PATH            Default: /home/jetson/code/kvshare-edge/impl/MNN
  --port PORT                   Remote pic_server port. Default: 18096
  --local-port PORT             Local SSH tunnel port. Default: same as --port
  --run-id ID                   Run id for logs/output.
  --remote-config PATH          Remote model config path.
  --remote-log-dir PATH         Remote log/PID directory. Default: <remote-repo>/.cache/logs
  --remote-ld-library-path PATH Exact remote LD_LIBRARY_PATH for pic_server.
  --remote-server-env ENV       Extra env assignments before pic_server, e.g. LD_PRELOAD=/usr/lib/libOpenCL_adreno.so.
  --line-buffer                 Start remote server through stdbuf when available.
  --bench-command run|compare   impl/pic_bench/cli.py subcommand. Default: run

Everything after -- is forwarded to impl/pic_bench/cli.py.
EOF
}

REMOTE="jetson@192.168.101.192"
REMOTE_REPO="/home/jetson/code/kvshare-edge/impl/MNN"
REMOTE_CUDA_LIB="/usr/local/cuda-12.2/targets/aarch64-linux/lib"
REMOTE_HOST="127.0.0.1"
REMOTE_PORT="18096"
LOCAL_HOST="127.0.0.1"
LOCAL_PORT=""
SERVED_MODEL="llama-pic"
RUN_ID="mnn_pic_dataset_$(timestamp)"
BUILD_DIR="${MNN_ROOT}/.cache/build/mnn/jetson_cross_cuda"
INSTALL_PREFIX="${MNN_ROOT}/.cache/output/mnn/artifacts/jetson_cross_cuda"
REMOTE_ART_REL=".cache/output/mnn/artifacts/jetson_cross_cuda"
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

LOCAL_PORT="${LOCAL_PORT:-${REMOTE_PORT}}"
REMOTE_CONFIG="${REMOTE_CONFIG:-${REMOTE_REPO}/.cache/weight/AI-ModelScope__Llama-3___2-3B-Instruct/config_cuda_greedy.json}"
REMOTE_KV_DIR="${REMOTE_KV_DIR:-${REMOTE_REPO}/.cache/kvshare/mnn_pic_dataset_bench_${RUN_ID}}"
REMOTE_ART="${REMOTE_REPO}/${REMOTE_ART_REL}"
REMOTE_LD_LIBRARY_PATH="${REMOTE_LD_LIBRARY_PATH:-${REMOTE_ART}/lib${REMOTE_CUDA_LIB:+:${REMOTE_CUDA_LIB}}}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-${REMOTE_REPO}/.cache/logs}"
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

cleanup() {
  local exit_code=$?
  if [[ -n "${TUNNEL_PID}" ]]; then
    kill "${TUNNEL_PID}" >/dev/null 2>&1 || true
    wait "${TUNNEL_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${REMOTE_SERVER_STARTED}" == "1" ]]; then
    ssh "${REMOTE}" "if [[ -f '${REMOTE_PID_FILE}' ]]; then kill \$(cat '${REMOTE_PID_FILE}') >/dev/null 2>&1 || true; fi; pkill -f 'pic_server.*--port ${REMOTE_PORT}' >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
  fi
  {
    printf 'run_id=%s\n' "${RUN_ID}"
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
  log "building local Jetson-cross pic_server artifacts"
  (
    cd "${MNN_ROOT}"
    export BUILD_DIR
    export INSTALL_PREFIX
    export JOBS="${JOBS:-8}"
    export CUDA_ARCHS="${CUDA_ARCHS:-72}"
    export ENABLE_CROSS_CUDA="${ENABLE_CROSS_CUDA:-ON}"
    export BUILD_TARGET="${BUILD_TARGET:-pic_server}"
    export BUILD_MNNCONVERT="${BUILD_MNNCONVERT:-0}"
    export INSTALL_AFTER_BUILD="${INSTALL_AFTER_BUILD:-1}"
    if [[ -z "${CUDA_TOOLKIT_ROOT:-}" && -d "${MNN_ROOT}/.cache/sysroots/jetson_cuda" ]]; then
      export CUDA_TOOLKIT_ROOT="${MNN_ROOT}/.cache/sysroots/jetson_cuda"
    fi
    if [[ -z "${CUDA_NVCC_EXECUTABLE:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
      export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc
    fi
    if [[ -z "${CMAKE_ARGS:-}" && -x "${MNN_ROOT}/.cache/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++" ]]; then
      export CMAKE_ARGS="-DCUDA_HOST_COMPILER=${MNN_ROOT}/.cache/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++"
    fi
    bash .codex/skills/mnn-build-artifacts/scripts/build_artifacts.sh
    cmake --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX}"
    mkdir -p "${INSTALL_PREFIX}/bin" "${INSTALL_PREFIX}/lib"
    install -m 755 "${BUILD_DIR}/pic_server" "${INSTALL_PREFIX}/bin/pic_server"
    install -m 755 "${BUILD_DIR}/libpic_llm.so" "${INSTALL_PREFIX}/lib/libpic_llm.so"
    install -m 755 "${BUILD_DIR}/source/backend/cuda/libMNN_Cuda_Main.so" "${INSTALL_PREFIX}/lib/libMNN_Cuda_Main.so"
  )
fi

if [[ "${SKIP_RSYNC}" != "1" ]]; then
  log "rsync artifact root to ${REMOTE}:${REMOTE_REPO}/${REMOTE_ART_REL}/"
  rsync -a --delete "${INSTALL_PREFIX}/" "${REMOTE}:${REMOTE_REPO}/${REMOTE_ART_REL}/"
fi

log "starting remote MNN pic_server on ${REMOTE}:${REMOTE_PORT}"
ssh "${REMOTE}" "cd '${REMOTE_REPO}' && \
  mkdir -p '${REMOTE_LOG_DIR}' && \
  { pkill -f 'pic_server.*--port ${REMOTE_PORT}' >/dev/null 2>&1 || true; } && \
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
ssh -N -L "${LOCAL_HOST}:${LOCAL_PORT}:${REMOTE_HOST}:${REMOTE_PORT}" "${REMOTE}" >"${TUNNEL_LOG}" 2>&1 &
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
