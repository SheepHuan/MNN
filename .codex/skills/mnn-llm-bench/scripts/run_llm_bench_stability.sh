#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_llm_bench_stability.sh

Environment:
  MNN_ARTIFACT_PLATFORM        Artifact platform name. Defaults to jetson on Jetson/aarch64, x64 on x86_64.
  MNN_ARTIFACT_ROOT            Artifact root. Defaults to .cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM.
  MNN_LLM_BENCH_MODEL_CONFIG   Exact model config.json path.
  MNN_LLM_BENCH_MODEL          Model dir, config path, or .cache/weight/<name> model name.
  MNN_LLM_BENCH_BIN            Exact bench executable path. Defaults to pic_llm_bench for PIC models.
  MNN_LLM_BENCH_BACKEND        Backend. Defaults to cuda when CUDA artifact exists, otherwise cpu.
  MNN_LLM_BENCH_CASES          Case list in prompt:decode form. Defaults to "512:128 1024:128 2048:128".
  MNN_LLM_BENCH_REP            Repeat count. Defaults to 1.
  MNN_LLM_BENCH_THREADS        Thread count. Defaults to 4.
  MNN_LLM_BENCH_PRECISION      Precision option passed to -c. Defaults to 2.
  MNN_LLM_BENCH_LOAD           loading-time option passed to -load. Defaults to true.
  MNN_LLM_BENCH_MEMORY         Optional --memory value. Defaults unset.
  MNN_LLM_BENCH_EXTRA_ARGS     Extra arguments appended to every bench invocation.
  CUDA_LIB_DIR                 CUDA lib64 directory. Auto-detected from nvcc when unset.
  MNN_LLM_BENCH_DRY_RUN        Print resolved commands without running when set to 1.
  MNN_EXPORT_DRY_RUN           Also accepted as a dry-run switch when MNN_LLM_BENCH_DRY_RUN is unset.

Default cases:
  512 prefill, 128 decode
  1024 prefill, 128 decode
  2048 prefill, 128 decode
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
WEIGHT_ROOT="${REPO_ROOT}/.cache/weight"

detect_mnn_artifact_platform() {
  if [[ -n "${MNN_ARTIFACT_PLATFORM:-}" ]]; then
    printf '%s\n' "${MNN_ARTIFACT_PLATFORM}"
    return
  fi
  if [[ -r /proc/device-tree/model ]] && tr -d '\0' </proc/device-tree/model | grep -qiE 'Jetson|NVIDIA'; then
    printf 'jetson\n'
    return
  fi
  case "$(uname -m)" in
    x86_64|amd64) printf 'x64\n' ;;
    aarch64|arm64) printf 'jetson\n' ;;
    *) uname -m ;;
  esac
}

normalize_path() {
  local path="$1"
  if [[ "${path}" != /* ]]; then
    path="${REPO_ROOT}/${path}"
  fi
  realpath -m "${path}"
}

resolve_model_config() {
  local model="${MNN_LLM_BENCH_MODEL_CONFIG:-${MNN_LLM_BENCH_MODEL:-}}"
  local candidate

  if [[ -n "${model}" ]]; then
    candidate="$(normalize_path "${model}")"
    if [[ -d "${candidate}" && -f "${candidate}/config.json" ]]; then
      realpath "${candidate}/config.json"
      return 0
    fi
    if [[ -f "${candidate}" ]]; then
      realpath "${candidate}"
      return 0
    fi
    candidate="${WEIGHT_ROOT}/${model}/config.json"
    if [[ -f "${candidate}" ]]; then
      realpath "${candidate}"
      return 0
    fi
  fi

  candidate="${WEIGHT_ROOT}/AI-ModelScope__Llama-3___2-3B-Instruct/config.json"
  if [[ -f "${candidate}" ]]; then
    realpath "${candidate}"
    return 0
  fi

  if [[ -d "${WEIGHT_ROOT}" ]]; then
    while IFS= read -r candidate; do
      local dir
      dir="$(dirname "${candidate}")"
      if [[ -f "${dir}/llm_config.json" ]] && grep -q '"paged_attention"[[:space:]]*:[[:space:]]*true' "${dir}/llm_config.json"; then
        realpath "${candidate}"
        return 0
      fi
    done < <(find "${WEIGHT_ROOT}" -mindepth 2 -maxdepth 2 -name config.json | sort)
  fi

  return 1
}

ARTIFACT_PLATFORM="$(detect_mnn_artifact_platform)"
ARTIFACT_ROOT="$(normalize_path "${MNN_ARTIFACT_ROOT:-${REPO_ROOT}/.cache/output/mnn/artifacts/${ARTIFACT_PLATFORM}}")"
MODEL_CONFIG="$(resolve_model_config || true)"

if [[ -z "${MODEL_CONFIG}" || ! -f "${MODEL_CONFIG}" ]]; then
  echo "Could not resolve model config. Set MNN_LLM_BENCH_MODEL_CONFIG or MNN_LLM_BENCH_MODEL." >&2
  exit 2
fi

MODEL_DIR="$(dirname "${MODEL_CONFIG}")"
MODEL_NAME="$(basename "${MODEL_DIR}")"
LLM_CONFIG="${MODEL_DIR}/llm_config.json"

if [[ ! -f "${MODEL_DIR}/llm.mnn" || ! -f "${MODEL_DIR}/llm.mnn.weight" || ! -f "${LLM_CONFIG}" ]]; then
  echo "Model directory is missing required files: ${MODEL_DIR}" >&2
  echo "Expected config.json, llm_config.json, llm.mnn and llm.mnn.weight." >&2
  exit 2
fi

IS_PIC_MODEL=0
if grep -q '"paged_attention"[[:space:]]*:[[:space:]]*true' "${LLM_CONFIG}"; then
  IS_PIC_MODEL=1
fi

if [[ -n "${MNN_LLM_BENCH_BIN:-}" ]]; then
  BENCH_BIN="$(normalize_path "${MNN_LLM_BENCH_BIN}")"
elif [[ "${IS_PIC_MODEL}" -eq 1 && -x "${ARTIFACT_ROOT}/bin/pic_llm_bench" ]]; then
  BENCH_BIN="${ARTIFACT_ROOT}/bin/pic_llm_bench"
else
  BENCH_BIN="${ARTIFACT_ROOT}/bin/llm_bench"
fi

if [[ ! -x "${BENCH_BIN}" ]]; then
  echo "Bench executable not found or not executable: ${BENCH_BIN}" >&2
  exit 2
fi

BACKEND="${MNN_LLM_BENCH_BACKEND:-}"
if [[ -z "${BACKEND}" ]]; then
  if [[ -f "${ARTIFACT_ROOT}/lib/libMNN_Cuda_Main.so" ]]; then
    BACKEND="cuda"
  else
    BACKEND="cpu"
  fi
fi

case "${BACKEND}" in
  cuda)
    if [[ ! -f "${ARTIFACT_ROOT}/lib/libMNN_Cuda_Main.so" ]]; then
      echo "CUDA backend library not found: ${ARTIFACT_ROOT}/lib/libMNN_Cuda_Main.so" >&2
      exit 2
    fi
    ;;
  opencl)
    if [[ ! -f "${ARTIFACT_ROOT}/lib/libMNN_CL.so" ]]; then
      echo "OpenCL backend library not found: ${ARTIFACT_ROOT}/lib/libMNN_CL.so" >&2
      exit 2
    fi
    ;;
  vulkan)
    if [[ ! -f "${ARTIFACT_ROOT}/lib/libMNN_Vulkan.so" ]]; then
      echo "Vulkan backend library not found: ${ARTIFACT_ROOT}/lib/libMNN_Vulkan.so" >&2
      exit 2
    fi
    ;;
  cpu) ;;
  *)
    echo "Unsupported MNN_LLM_BENCH_BACKEND: ${BACKEND}" >&2
    exit 2
    ;;
esac

if [[ -z "${CUDA_LIB_DIR:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_LIB_DIR="$(dirname "$(dirname "$(command -v nvcc)")")/lib64"
  elif [[ -d /usr/local/cuda/lib64 ]]; then
    CUDA_LIB_DIR="/usr/local/cuda/lib64"
  else
    CUDA_LIB_DIR=""
  fi
fi

LD_PATH=""
append_ld_path() {
  local path="$1"
  [[ -n "${path}" ]] || return
  case ":${LD_PATH}:" in
    *":${path}:"*) ;;
    *) LD_PATH="${LD_PATH:+${LD_PATH}:}${path}" ;;
  esac
}

append_ld_path "${ARTIFACT_ROOT}/lib"
append_ld_path "${CUDA_LIB_DIR}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  IFS=':' read -r -a current_ld_paths <<< "${LD_LIBRARY_PATH}"
  for ld_path in "${current_ld_paths[@]}"; do
    append_ld_path "${ld_path}"
  done
fi

PRELOAD=()
case "${BACKEND}" in
  opencl) PRELOAD=(LD_PRELOAD="${ARTIFACT_ROOT}/lib/libMNN_CL.so") ;;
  vulkan) PRELOAD=(LD_PRELOAD="${ARTIFACT_ROOT}/lib/libMNN_Vulkan.so") ;;
esac

CASES="${MNN_LLM_BENCH_CASES:-512:128 1024:128 2048:128}"
REP="${MNN_LLM_BENCH_REP:-1}"
THREADS="${MNN_LLM_BENCH_THREADS:-4}"
PRECISION="${MNN_LLM_BENCH_PRECISION:-2}"
LOAD_TIME="${MNN_LLM_BENCH_LOAD:-true}"
MEMORY="${MNN_LLM_BENCH_MEMORY:-}"
DRY_RUN="${MNN_LLM_BENCH_DRY_RUN:-${MNN_EXPORT_DRY_RUN:-0}}"
EXTRA_ARGS=()
if [[ -n "${MNN_LLM_BENCH_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${MNN_LLM_BENCH_EXTRA_ARGS})
fi

mkdir -p "${MODEL_DIR}/tmp"
LOG_DIR="${REPO_ROOT}/.cache/logs/llm-bench-stability"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${LOG_DIR}/${STAMP}_${MODEL_NAME}_${BACKEND}_summary.log"

{
  echo "repo: ${REPO_ROOT}"
  echo "artifact_platform: ${ARTIFACT_PLATFORM}"
  echo "artifact_root: ${ARTIFACT_ROOT}"
  echo "bench_bin: ${BENCH_BIN}"
  echo "model_config: ${MODEL_CONFIG}"
  echo "model_dir: ${MODEL_DIR}"
  echo "tmp_dir: ${MODEL_DIR}/tmp"
  echo "backend: ${BACKEND}"
  echo "cases: ${CASES}"
  echo "rep: ${REP}"
  echo "threads: ${THREADS}"
  echo "precision: ${PRECISION}"
  echo "load_time: ${LOAD_TIME}"
  echo "summary_log: ${SUMMARY_LOG}"
} | tee "${SUMMARY_LOG}"

status=0
for case_spec in ${CASES}; do
  case_spec="${case_spec//,/}"
  if [[ "${case_spec}" != *:* ]]; then
    echo "Invalid case '${case_spec}', expected prompt:decode" | tee -a "${SUMMARY_LOG}" >&2
    status=2
    continue
  fi
  prompt="${case_spec%%:*}"
  decode="${case_spec##*:}"
  if ! [[ "${prompt}" =~ ^[0-9]+$ && "${decode}" =~ ^[0-9]+$ ]]; then
    echo "Invalid numeric case '${case_spec}', expected prompt:decode" | tee -a "${SUMMARY_LOG}" >&2
    status=2
    continue
  fi

  case_log="${LOG_DIR}/${STAMP}_${MODEL_NAME}_${BACKEND}_p${prompt}_n${decode}.log"
  cmd=(
    "${BENCH_BIN}"
    -m "${MODEL_CONFIG}"
    -a "${BACKEND}"
    -p "${prompt}"
    -n "${decode}"
    -rep "${REP}"
    -kv true
    -load "${LOAD_TIME}"
    -c "${PRECISION}"
    -t "${THREADS}"
  )
  if [[ -n "${MEMORY}" ]]; then
    cmd+=(--memory "${MEMORY}")
  fi
  cmd+=("${EXTRA_ARGS[@]}")

  {
    echo
    echo "==> case p=${prompt} n=${decode}"
    echo "workdir: ${MODEL_DIR}"
    printf 'command: LD_LIBRARY_PATH=%q ' "${LD_PATH}"
    for kv in "${PRELOAD[@]}"; do
      printf '%q ' "${kv}"
    done
    printf '%q ' "${cmd[@]}"
    echo
    echo "log: ${case_log}"
  } | tee -a "${SUMMARY_LOG}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  set +e
  (
    cd "${MODEL_DIR}"
    env LD_LIBRARY_PATH="${LD_PATH}" "${PRELOAD[@]}" "${cmd[@]}"
  ) 2>&1 | tee "${case_log}"
  case_status=${PIPESTATUS[0]}
  set -e

  echo "case p=${prompt} n=${decode} exit_code=${case_status}" | tee -a "${SUMMARY_LOG}"
  if [[ "${case_status}" -ne 0 ]]; then
    status="${case_status}"
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete." | tee -a "${SUMMARY_LOG}"
else
  if [[ -f "${MODEL_DIR}/tmp/mnn_cachefile.bin" ]]; then
    echo "cachefile: ${MODEL_DIR}/tmp/mnn_cachefile.bin" | tee -a "${SUMMARY_LOG}"
  else
    echo "cachefile was not created: ${MODEL_DIR}/tmp/mnn_cachefile.bin" | tee -a "${SUMMARY_LOG}"
  fi
fi

exit "${status}"
