#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_pic_decode_repair_config_tune.sh

Enumerate PIC decode-repair server startup candidates on the current machine.
For each candidate, this script starts pic_server in an isolated workdir, warms
target decode-repair shapes through run_pic_decode_repair_tune_warm.sh, calls
/v1/tune/update_cache, then optionally restarts the same candidate to verify the
cache file can be reused before strict TPOT.

This is a server/runtime-cache tune runner. It does not replace direct-op CUDA
event screening for rows=4/5 MLP kernels.

Environment:
  PIC_CONFIG_TUNE_ART             Artifact root. Default: .cache/output/mnn/artifacts/jetson_cross_cuda
  PIC_CONFIG_TUNE_CONFIG          Model config path. Required unless default exists.
  PIC_CONFIG_TUNE_CUDA_LIB        CUDA lib dir. Default: /usr/local/cuda-12.2/targets/aarch64-linux/lib
  PIC_CONFIG_TUNE_MODEL           Served model name. Default: llama-pic
  PIC_CONFIG_TUNE_HOST            Server host. Default: 127.0.0.1
  PIC_CONFIG_TUNE_PORT_BASE       First candidate port. Default: 18140
  PIC_CONFIG_TUNE_RUN_ID          Run id. Default: decode_repair_config_tune_<timestamp>
  PIC_CONFIG_TUNE_OUTPUT_ROOT     Output root. Default: .cache/decode_repair_config_tune/<run_id>
  PIC_CONFIG_TUNE_CANDIDATES      Space-separated candidate names. Default: baseline
  PIC_CONFIG_TUNE_ENV_DIR         Candidate env dir. Default: <output_root>/candidate_env
                                  Optional files: <name>.server.env and <name>.warm.env
  PIC_CONFIG_TUNE_TPDS            Warm tpd list. Default: "0 1 2 3 4"
  PIC_CONFIG_TUNE_MAX_TOKENS      Warm max_tokens. Default: 32
  PIC_CONFIG_TUNE_WARM_REPEATS    Warm repeats per tpd. Default: 1
  PIC_CONFIG_TUNE_SELECTION_ALGORITHM  Default warm selection. Default: full-reuse
  PIC_CONFIG_TUNE_RESTART_CHECK   Restart after update_cache and check /healthz. Default: 1
  PIC_CONFIG_TUNE_KEEP_LAST       Keep final candidate server running. Default: 0
  PIC_CONFIG_TUNE_DRY_RUN         Write manifests/payloads without starting server. Default: 0

Candidate env files are shell snippets and are sourced in trusted CI/dev runs.
Use *.server.env for startup env such as MNN_PIC_* or CUDA policy toggles, and
*.warm.env for PIC_TUNE_* request matrix overrides.

Example:
  mkdir -p .cache/decode_repair_candidates
  cat > .cache/decode_repair_candidates/baseline.server.env <<'EOF_CAND'
  # no overrides
EOF_CAND
  cat > .cache/decode_repair_candidates/gateup.warm.env <<'EOF_CAND'
  PIC_TUNE_TPDS="0 1 2 3 4"
EOF_CAND
  PIC_CONFIG_TUNE_CONFIG=.cache/weight/AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-silumul/config_cuda_greedy.json \
  PIC_CONFIG_TUNE_ENV_DIR=.cache/decode_repair_candidates \
  PIC_CONFIG_TUNE_CANDIDATES="baseline gateup" \
  bash .codex/skills/mnn-opt-ops/scripts/run_pic_decode_repair_config_tune.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MNN_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

timestamp() {
  date +%Y%m%d_%H%M%S
}

log() {
  printf '[pic-config-tune] %s\n' "$*"
}

die() {
  printf '[pic-config-tune] ERROR: %s\n' "$*" >&2
  exit 1
}

need_tool() {
  command -v "$1" >/dev/null 2>&1 || die "required tool not found: $1"
}

abs_path() {
  local path="$1"
  case "${path}" in
    /*) printf '%s\n' "${path}" ;;
    *) printf '%s/%s\n' "${MNN_ROOT}" "${path}" ;;
  esac
}

sanitize_name() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

source_env_if_exists() {
  local file="$1"
  if [[ -f "${file}" ]]; then
    # shellcheck disable=SC1090
    source "${file}"
  fi
}

ART="$(abs_path "${PIC_CONFIG_TUNE_ART:-.cache/output/mnn/artifacts/jetson_cross_cuda}")"
CONFIG_DEFAULT=".cache/weight/AI-ModelScope__Llama-3___2-1B-Instruct-decode-repair-silumul/config_cuda_greedy.json"
CONFIG="$(abs_path "${PIC_CONFIG_TUNE_CONFIG:-${CONFIG_DEFAULT}}")"
CUDA_LIB="${PIC_CONFIG_TUNE_CUDA_LIB:-/usr/local/cuda-12.2/targets/aarch64-linux/lib}"
MODEL="${PIC_CONFIG_TUNE_MODEL:-llama-pic}"
HOST="${PIC_CONFIG_TUNE_HOST:-127.0.0.1}"
PORT_BASE="${PIC_CONFIG_TUNE_PORT_BASE:-18140}"
RUN_ID="${PIC_CONFIG_TUNE_RUN_ID:-decode_repair_config_tune_$(timestamp)}"
OUTPUT_ROOT="$(abs_path "${PIC_CONFIG_TUNE_OUTPUT_ROOT:-.cache/decode_repair_config_tune/${RUN_ID}}")"
ENV_DIR="$(abs_path "${PIC_CONFIG_TUNE_ENV_DIR:-${OUTPUT_ROOT}/candidate_env}")"
CANDIDATES="${PIC_CONFIG_TUNE_CANDIDATES:-baseline}"
TPDS="${PIC_CONFIG_TUNE_TPDS:-0 1 2 3 4}"
MAX_TOKENS="${PIC_CONFIG_TUNE_MAX_TOKENS:-32}"
WARM_REPEATS="${PIC_CONFIG_TUNE_WARM_REPEATS:-1}"
SELECTION_ALGORITHM="${PIC_CONFIG_TUNE_SELECTION_ALGORITHM:-full-reuse}"
RESTART_CHECK="${PIC_CONFIG_TUNE_RESTART_CHECK:-1}"
KEEP_LAST="${PIC_CONFIG_TUNE_KEEP_LAST:-0}"
DRY_RUN="${PIC_CONFIG_TUNE_DRY_RUN:-0}"
READY_TIMEOUT_SEC="${PIC_CONFIG_TUNE_READY_TIMEOUT_SEC:-180}"
KILL_EXISTING="${PIC_CONFIG_TUNE_KILL_EXISTING:-1}"

PIC_SERVER="${ART}/bin/pic_server"
WARM_SCRIPT="${MNN_ROOT}/.codex/skills/mnn-opt-ops/scripts/run_pic_decode_repair_tune_warm.sh"
SUMMARIZE_SCRIPT="${MNN_ROOT}/.codex/skills/mnn-opt-ops/scripts/summarize_pic_decode_repair_config_tune.py"
SUMMARY="${OUTPUT_ROOT}/summary.tsv"
WARM_SUMMARY="${OUTPUT_ROOT}/warm_summary.tsv"

need_tool bash
need_tool python3
if [[ "${DRY_RUN}" != "1" ]]; then
  need_tool curl
  [[ -x "${PIC_SERVER}" ]] || die "pic_server not found or not executable: ${PIC_SERVER}"
  [[ -f "${CONFIG}" ]] || die "model config not found: ${CONFIG}"
fi
[[ -f "${WARM_SCRIPT}" ]] || die "warm script not found: ${WARM_SCRIPT}"

mkdir -p "${OUTPUT_ROOT}" "${ENV_DIR}"

CURRENT_PID_FILE=""
CURRENT_PORT=""

stop_current_server() {
  if [[ -n "${CURRENT_PID_FILE}" && -f "${CURRENT_PID_FILE}" ]]; then
    kill "$(cat "${CURRENT_PID_FILE}")" >/dev/null 2>&1 || true
    wait "$(cat "${CURRENT_PID_FILE}")" >/dev/null 2>&1 || true
    rm -f "${CURRENT_PID_FILE}"
  fi
  if [[ -n "${CURRENT_PORT}" && "${KILL_EXISTING}" == "1" ]]; then
    pkill -f "pic_server.*--port ${CURRENT_PORT}" >/dev/null 2>&1 || true
  fi
  CURRENT_PID_FILE=""
  CURRENT_PORT=""
}

cleanup() {
  local exit_code=$?
  if [[ "${KEEP_LAST}" != "1" || "${exit_code}" != "0" ]]; then
    stop_current_server
  fi
}
trap cleanup EXIT

wait_ready() {
  local base_url="$1"
  local server_log="$2"
  local ready="0"
  for _ in $(seq 1 "${READY_TIMEOUT_SEC}"); do
    if curl -fsS "${base_url}/healthz" >/dev/null 2>&1; then
      ready="1"
      break
    fi
    sleep 1
  done
  if [[ "${ready}" != "1" ]]; then
    tail -n 120 "${server_log}" >&2 || true
    die "pic_server did not become ready: ${base_url}"
  fi
}

start_server() {
  local candidate="$1"
  local port="$2"
  local work_dir="$3"
  local kv_dir="$4"
  local server_log="$5"
  local pid_file="$6"
  local server_env_file="$7"

  mkdir -p "${work_dir}" "${kv_dir}" "$(dirname "${server_log}")"
  if [[ "${KILL_EXISTING}" == "1" ]]; then
    pkill -f "pic_server.*--port ${port}" >/dev/null 2>&1 || true
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      printf 'dry_run=1\n'
      printf 'candidate=%s\n' "${candidate}"
      printf 'server_env_file=%s\n' "${server_env_file}"
      printf 'work_dir=%s\n' "${work_dir}"
      printf 'command=%s --config %s --host %s --port %s --kv-cache-dir %s --model %s\n' \
        "${PIC_SERVER}" "${CONFIG}" "${HOST}" "${port}" "${kv_dir}" "${MODEL}"
    } > "${server_log}"
    return
  fi

  (
    set -a
    source_env_if_exists "${server_env_file}"
    set +a
    export LD_LIBRARY_PATH="${ART}/lib:${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
    cd "${work_dir}"
    nohup "${PIC_SERVER}" --config "${CONFIG}" --host "${HOST}" --port "${port}" \
      --kv-cache-dir "${kv_dir}" --model "${MODEL}" > "${server_log}" 2>&1 &
    echo $! > "${pid_file}"
  )
  CURRENT_PID_FILE="${pid_file}"
  CURRENT_PORT="${port}"
  wait_ready "http://${HOST}:${port}" "${server_log}"
}

run_warm() {
  local candidate="$1"
  local port="$2"
  local warm_dir="$3"
  local warm_env_file="$4"

  mkdir -p "${warm_dir}"
  (
    set -a
    source_env_if_exists "${warm_env_file}"
    set +a
    export PIC_TUNE_BASE_URL="http://${HOST}:${port}"
    export PIC_TUNE_MODEL="${PIC_TUNE_MODEL:-${MODEL}}"
    export PIC_TUNE_OUTPUT_DIR="${warm_dir}"
    export PIC_TUNE_RUN_ID="${candidate}"
    export PIC_TUNE_SELECTION_ALGORITHM="${PIC_TUNE_SELECTION_ALGORITHM:-${SELECTION_ALGORITHM}}"
    export PIC_TUNE_TPDS="${PIC_TUNE_TPDS:-${TPDS}}"
    export PIC_TUNE_MAX_TOKENS="${PIC_TUNE_MAX_TOKENS:-${MAX_TOKENS}}"
    export PIC_TUNE_WARM_REPEATS="${PIC_TUNE_WARM_REPEATS:-${WARM_REPEATS}}"
    export PIC_TUNE_DRY_RUN="${DRY_RUN}"
    cd "${MNN_ROOT}"
    bash "${WARM_SCRIPT}"
  ) 2>&1 | tee "${warm_dir}/warm.log"
}

write_manifest() {
  local file="$1"
  shift
  {
    printf 'run_id=%s\n' "${RUN_ID}"
    printf 'artifact=%s\n' "${ART}"
    printf 'config=%s\n' "${CONFIG}"
    printf 'model=%s\n' "${MODEL}"
    printf 'host=%s\n' "${HOST}"
    printf 'output_root=%s\n' "${OUTPUT_ROOT}"
    printf 'env_dir=%s\n' "${ENV_DIR}"
    for kv in "$@"; do
      printf '%s\n' "${kv}"
    done
  } > "${file}"
}

printf 'candidate\tport\tstatus\tcache_exists\tcache_file\twork_dir\twarm_dir\n' > "${SUMMARY}"

read -r -a CANDIDATE_ARRAY <<<"${CANDIDATES}"
candidate_count="${#CANDIDATE_ARRAY[@]}"
[[ "${candidate_count}" -gt 0 ]] || die "no candidates configured"

log "artifact=${ART}"
log "config=${CONFIG}"
log "output_root=${OUTPUT_ROOT}"
log "env_dir=${ENV_DIR}"
log "candidates=${CANDIDATES}"

index=0
for raw_candidate in "${CANDIDATE_ARRAY[@]}"; do
  [[ -n "${raw_candidate}" ]] || continue
  candidate="$(sanitize_name "${raw_candidate}")"
  port="$((PORT_BASE + index))"
  candidate_dir="${OUTPUT_ROOT}/${candidate}"
  work_dir="${candidate_dir}/server_work"
  kv_dir="${candidate_dir}/kv_cache"
  warm_dir="${candidate_dir}/warm"
  server_log="${candidate_dir}/server.log"
  pid_file="${candidate_dir}/server.pid"
  cache_file="${work_dir}/tmp/mnn_cachefile.bin"
  server_env_file="${ENV_DIR}/${candidate}.server.env"
  warm_env_file="${ENV_DIR}/${candidate}.warm.env"
  status="ok"

  log "candidate=${candidate} port=${port}"
  mkdir -p "${candidate_dir}"
  if [[ "${PIC_CONFIG_TUNE_CLEAR_CACHE:-1}" == "1" ]]; then
    rm -rf "${work_dir}/tmp" "${kv_dir}"
  fi

  write_manifest "${candidate_dir}/manifest.env" \
    "candidate=${candidate}" \
    "port=${port}" \
    "work_dir=${work_dir}" \
    "kv_dir=${kv_dir}" \
    "warm_dir=${warm_dir}" \
    "server_log=${server_log}" \
    "server_env_file=${server_env_file}" \
    "warm_env_file=${warm_env_file}" \
    "cache_file=${cache_file}"

  if ! start_server "${candidate}" "${port}" "${work_dir}" "${kv_dir}" "${server_log}" "${pid_file}" "${server_env_file}"; then
    status="server_start_failed"
  elif ! run_warm "${candidate}" "${port}" "${warm_dir}" "${warm_env_file}"; then
    status="warm_failed"
  fi

  cache_exists="0"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cache_exists="dry_run"
  elif [[ -f "${cache_file}" ]]; then
    cache_exists="1"
  fi

  if [[ "${status}" == "ok" && "${RESTART_CHECK}" == "1" ]]; then
    stop_current_server
    log "restart check candidate=${candidate}"
    start_server "${candidate}" "${port}" "${work_dir}" "${kv_dir}" "${candidate_dir}/server_restart.log" "${pid_file}" "${server_env_file}"
  fi

  if [[ "${KEEP_LAST}" != "1" || "$((index + 1))" -lt "${candidate_count}" ]]; then
    stop_current_server
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${candidate}" "${port}" "${status}" "${cache_exists}" "${cache_file}" "${work_dir}" "${warm_dir}" >> "${SUMMARY}"

  if [[ "${status}" != "ok" ]]; then
    die "candidate ${candidate} failed: ${status}"
  fi
  index="$((index + 1))"
done

log "config tune warm complete"
log "summary=${SUMMARY}"
cat "${SUMMARY}"
if [[ -f "${SUMMARIZE_SCRIPT}" ]]; then
  log "warm_summary=${WARM_SUMMARY}"
  python3 "${SUMMARIZE_SCRIPT}" "${OUTPUT_ROOT}" --output-tsv "${WARM_SUMMARY}"
fi
