#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_jetson_cuda_power_suite.sh

Environment:
  MNN_JETSON_REMOTE          SSH target. Defaults to jetson@192.168.101.192.
  MNN_JETSON_REPO            Remote repo root. Defaults to /home/jetson/code/kvshare-edge/impl/MNN.
  MNN_JETSON_ARTIFACT_ROOT   Remote artifact root. Defaults to $MNN_JETSON_REPO/.cache/output/mnn/artifacts/jetson.
  MNN_JETSON_MODEL_ROOT      Remote model root. Defaults to $MNN_JETSON_REPO/.cache/mnn-llm-export.
  MNN_LLM_EXPORT_ROOT        Local normal export root. Defaults to .cache/mnn-llm-export.
  MODELSCOPE_CACHE_ROOT      ModelScope cache root. Defaults to ~/.cache/modelscope/hub/models.
  MNN_LLM_BENCH_MODEL_FILTER Optional model id/export-dir substring filter.
  MNN_LLM_BENCH_CASES        prompt:decode[:rep] cases. Defaults to "256:1 512:1 1024:1 2048:1 4096:1".
  MNN_LLM_BENCH_REP          Default repeat count when a case omits :rep. Defaults to 1.
  MNN_LLM_BENCH_THREADS      Thread count. Defaults to 4.
  MNN_LLM_BENCH_PRECISION    Precision option passed to -c. Defaults to 2.
  MNN_LLM_BENCH_LOAD         loading-time option passed to -load. Defaults to true.
  MNN_LLM_BENCH_MEMORY       --memory value. Defaults to 2.
  MNN_POWER_API_URL          DF power API. Defaults to http://192.168.101.14:8000.
  MNN_POWER_SERIAL           DF serial. Defaults to Jetson serial 1A5D43.
  MNN_POWER_WARMUP_SEC       Seconds before bench inside power window. Defaults to 10.
  MNN_POWER_COOLDOWN_SEC     Seconds after bench inside power window. Defaults to 10.
  MNN_LLM_BENCH_DRY_RUN      Print checks/commands without rsync, ssh, or power API calls when set to 1.

This suite expects normal 4-bit MNN exports to already exist in .cache/mnn-llm-export.
It does not export models automatically.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
POWER_SCRIPT="${SCRIPT_DIR}/df_power_capture.sh"
HF_CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
MODELSCOPE_CACHE_ROOT="${MODELSCOPE_CACHE_ROOT:-${HOME}/.cache/modelscope/hub/models}"
LOCAL_EXPORT_ROOT="${MNN_LLM_EXPORT_ROOT:-${REPO_ROOT}/.cache/mnn-llm-export}"
LOCAL_ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-${REPO_ROOT}/.cache/output/mnn/artifacts/jetson}"
if [[ "${LOCAL_EXPORT_ROOT}" != /* ]]; then
  LOCAL_EXPORT_ROOT="${REPO_ROOT}/${LOCAL_EXPORT_ROOT}"
fi
if [[ "${LOCAL_ARTIFACT_ROOT}" != /* ]]; then
  LOCAL_ARTIFACT_ROOT="${REPO_ROOT}/${LOCAL_ARTIFACT_ROOT}"
fi
if [[ -z "${MNN_ARTIFACT_ROOT:-}" && ! -f "${LOCAL_ARTIFACT_ROOT}/lib/libMNN_Cuda_Main.so" ]]; then
  ALT_LOCAL_ARTIFACT_ROOT="${REPO_ROOT}/.cache/output/mnn/artifacts/jetson_cross_cuda"
  if [[ -f "${ALT_LOCAL_ARTIFACT_ROOT}/lib/libMNN_Cuda_Main.so" ]]; then
    LOCAL_ARTIFACT_ROOT="${ALT_LOCAL_ARTIFACT_ROOT}"
  fi
fi

REMOTE="${MNN_JETSON_REMOTE:-jetson@192.168.101.192}"
REMOTE_REPO="${MNN_JETSON_REPO:-/home/jetson/code/kvshare-edge/impl/MNN}"
REMOTE_ARTIFACT_ROOT="${MNN_JETSON_ARTIFACT_ROOT:-${REMOTE_REPO}/.cache/output/mnn/artifacts/jetson}"
REMOTE_MODEL_ROOT="${MNN_JETSON_MODEL_ROOT:-${REMOTE_REPO}/.cache/mnn-llm-export}"
REMOTE_CUDA_LIB_DIR="${MNN_JETSON_CUDA_LIB_DIR:-/usr/local/cuda-12.2/targets/aarch64-linux/lib}"

CASES="${MNN_LLM_BENCH_CASES:-256:1 512:1 1024:1 2048:1 4096:1}"
MODEL_FILTER="${MNN_LLM_BENCH_MODEL_FILTER:-}"
REP="${MNN_LLM_BENCH_REP:-1}"
THREADS="${MNN_LLM_BENCH_THREADS:-4}"
PRECISION="${MNN_LLM_BENCH_PRECISION:-2}"
LOAD_TIME="${MNN_LLM_BENCH_LOAD:-true}"
MEMORY="${MNN_LLM_BENCH_MEMORY:-2}"
POWER_API_URL="${MNN_POWER_API_URL:-http://192.168.101.14:8000}"
POWER_SERIAL="${MNN_POWER_SERIAL:-1A5D43}"
POWER_WARMUP_SEC="${MNN_POWER_WARMUP_SEC:-10}"
POWER_COOLDOWN_SEC="${MNN_POWER_COOLDOWN_SEC:-10}"
DRY_RUN="${MNN_LLM_BENCH_DRY_RUN:-${MNN_EXPORT_DRY_RUN:-0}}"

MODEL_REFS=(
  "AI-ModelScope/Llama-3.2-1B-Instruct|AI-ModelScope__Llama-3.2-1B-Instruct"
  "AI-ModelScope/Llama-3.2-3B-Instruct|AI-ModelScope__Llama-3.2-3B-Instruct"
  "LLM-Research/Meta-Llama-3-8B-Instruct|LLM-Research__Meta-Llama-3-8B-Instruct"
  "ZhipuAI/glm-edge-4b-chat|ZhipuAI__glm-edge-4b-chat"
  "Qwen/Qwen2.5-7B-Instruct|Qwen__Qwen2.5-7B-Instruct"
)

log() {
  printf '[jetson-power-suite] %s\n' "$*"
}

die() {
  printf '[jetson-power-suite] ERROR: %s\n' "$*" >&2
  exit 1
}

quote_args() {
  printf '%q ' "$@"
}

model_selected() {
  local model_ref="$1"
  local dst_name="$2"

  [[ -z "${MODEL_FILTER}" ]] && return 0
  [[ "${model_ref}" == *"${MODEL_FILTER}"* || "${dst_name}" == *"${MODEL_FILTER}"* ]]
}

resolve_hf_snapshot() {
  local model_ref="$1"
  local repo_dir refs_main snapshot

  repo_dir="${HF_CACHE_ROOT}/models--${model_ref//\//--}"
  if [[ ! -d "${repo_dir}" ]]; then
    return 1
  fi

  if [[ -f "${repo_dir}/refs/main" ]]; then
    refs_main="$(<"${repo_dir}/refs/main")"
    snapshot="${repo_dir}/snapshots/${refs_main}"
    if [[ -d "${snapshot}" ]]; then
      printf '%s\n' "${snapshot}"
      return 0
    fi
  fi

  snapshot="$(find "${repo_dir}/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
  [[ -n "${snapshot}" ]] || return 1
  printf '%s\n' "${snapshot}"
}

modelscope_encoded_ref() {
  local model_ref="$1"
  local namespace="${model_ref%%/*}"
  local name="${model_ref#*/}"
  printf '%s/%s\n' "${namespace}" "${name//./___}"
}

resolve_model_source() {
  local model_ref="$1"
  local source_dir encoded_ref

  source_dir="${MODELSCOPE_CACHE_ROOT}/${model_ref}"
  if [[ -d "${source_dir}" ]]; then
    printf '%s\n' "${source_dir}"
    return 0
  fi

  encoded_ref="$(modelscope_encoded_ref "${model_ref}")"
  source_dir="${MODELSCOPE_CACHE_ROOT}/${encoded_ref}"
  if [[ -d "${source_dir}" ]]; then
    printf '%s\n' "${source_dir}"
    return 0
  fi

  resolve_hf_snapshot "${model_ref}"
}

has_model_weights() {
  local model_dir="$1"
  find -L "${model_dir}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'model*.bin' \) -print -quit 2>/dev/null | grep -q .
}

check_local_export() {
  local export_dir="$1"
  local missing=()
  local file

  for file in config.json llm_config.json llm.mnn llm.mnn.weight; do
    if [[ ! -f "${export_dir}/${file}" ]]; then
      missing+=("${file}")
    fi
  done

  if [[ "${#missing[@]}" -gt 0 ]]; then
    printf '%s\n' "${missing[*]}"
    return 1
  fi
}

print_export_command() {
  local model_ref="$1"
  local source_path="$2"
  local dst_name="$3"
  local dst_path="${LOCAL_EXPORT_ROOT}/${dst_name}"

  printf '  MNN_QUANT_BIT=4 MNN_QUANT_BLOCK=64 MNN_EMBED_BIT=16 conda run -n kvshare-edge python transformers/llm/export/llmexport.py \\\n'
  printf '    --path %q \\\n' "${source_path}"
  printf '    --dst_path %q \\\n' "${dst_path#${REPO_ROOT}/}"
  printf '    --export mnn \\\n'
  printf '    --quant_bit 4 \\\n'
  printf '    --quant_block 64 \\\n'
  printf '    --embed_bit 16\n'
  printf '  # model: %s\n' "${model_ref}"
}

validate_case() {
  local case_spec="$1"
  local prompt decode rep rest

  if [[ "${case_spec}" != *:* ]]; then
    die "invalid case '${case_spec}', expected prompt:decode[:rep]"
  fi
  prompt="${case_spec%%:*}"
  rest="${case_spec#*:}"
  decode="${rest%%:*}"
  rep="${rest#*:}"
  if [[ "${rep}" == "${rest}" ]]; then
    rep="${REP}"
  fi
  if ! [[ "${prompt}" =~ ^[0-9]+$ && "${decode}" =~ ^[0-9]+$ && "${rep}" =~ ^[0-9]+$ ]]; then
    die "invalid numeric case '${case_spec}', expected prompt:decode[:rep]"
  fi
}

case_prompt() {
  local case_spec="$1"
  printf '%s\n' "${case_spec%%:*}"
}

case_decode() {
  local rest="${1#*:}"
  printf '%s\n' "${rest%%:*}"
}

case_rep() {
  local rest="${1#*:}"
  local rep="${rest#*:}"
  if [[ "${rep}" == "${rest}" ]]; then
    rep="${REP}"
  fi
  printf '%s\n' "${rep}"
}

remote_probe_script() {
  cat <<EOF
set -euo pipefail
test -x "$(printf '%q' "${REMOTE_ARTIFACT_ROOT}")/bin/llm_bench"
test -f "$(printf '%q' "${REMOTE_ARTIFACT_ROOT}")/lib/libMNN_Cuda_Main.so"
test -f "$(printf '%q' "${REMOTE_ARTIFACT_ROOT}")/lib/libMNN.so"
test -d "$(printf '%q' "${REMOTE_MODEL_ROOT}")"
EOF
}

build_remote_bench_script() {
  local remote_model_dir="$1"
  local prompt="$2"
  local decode="$3"
  local rep="$4"

  cat <<EOF
set -euo pipefail
cd "$(printf '%q' "${remote_model_dir}")"
mkdir -p tmp
LD_LIBRARY_PATH="$(printf '%q' "${REMOTE_ARTIFACT_ROOT}")/lib:$(printf '%q' "${REMOTE_CUDA_LIB_DIR}"):\${LD_LIBRARY_PATH:-}" \\
  "$(printf '%q' "${REMOTE_ARTIFACT_ROOT}")/bin/llm_bench" \\
  -m config.json \\
  -a cuda \\
  -p "${prompt}" \\
  -n "${decode}" \\
  -rep "$(printf '%q' "${rep}")" \\
  -kv true \\
  -load "$(printf '%q' "${LOAD_TIME}")" \\
  -c "$(printf '%q' "${PRECISION}")" \\
  -t "$(printf '%q' "${THREADS}")" \\
  --memory "$(printf '%q' "${MEMORY}")"
EOF
}

extract_bench_summary() {
  local case_log="$1"

  awk -F'|' '
    /prompt=/ {
      speed=$8
      loading=$9
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", speed)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", loading)
      print speed "\t" loading
      found=1
      exit
    }
    END {
      if (!found) {
        print "\t"
      }
    }
  ' "${case_log}"
}

run_power_wrapped_remote_bench() {
  local remote_script="$1"
  local power_csv="$2"
  local remote_script_file status

  remote_script_file="$(mktemp)"
  printf '%s\n' "${remote_script}" > "${remote_script_file}"

  set +e
  MNN_POWER_API_URL="${POWER_API_URL}" \
  MNN_POWER_SERIAL="${POWER_SERIAL}" \
  MNN_POWER_OUTPUT_CSV="${power_csv}" \
  "${POWER_SCRIPT}" -- bash -c '
set -euo pipefail
warmup="$1"
cooldown="$2"
remote="$3"
remote_script_file="$4"

sleep "${warmup}"
set +e
ssh "${remote}" bash -s < "${remote_script_file}"
bench_status=$?
set -e
sleep "${cooldown}"
exit "${bench_status}"
' _ "${POWER_WARMUP_SEC}" "${POWER_COOLDOWN_SEC}" "${REMOTE}" "${remote_script_file}"
  status=$?
  set -e

  rm -f "${remote_script_file}"
  return "${status}"
}

for case_spec in ${CASES}; do
  validate_case "${case_spec//,/}"
done

[[ -x "${POWER_SCRIPT}" ]] || die "power script is missing or not executable: ${POWER_SCRIPT}"

missing_any=0
selected_count=0
log "checking fixed ModelScope/HF sources and local normal MNN exports"
for item in "${MODEL_REFS[@]}"; do
  model_ref="${item%%|*}"
  dst_name="${item##*|}"
  if ! model_selected "${model_ref}" "${dst_name}"; then
    continue
  fi
  selected_count=$((selected_count + 1))
  source_path="$(resolve_model_source "${model_ref}" || true)"
  export_dir="${LOCAL_EXPORT_ROOT}/${dst_name}"

  if [[ -z "${source_path}" ]]; then
    missing_any=1
    printf '[jetson-power-suite] missing model cache source: %s\n' "${model_ref}" >&2
    printf '[jetson-power-suite] expected ModelScope under: %s/%s or %s/%s\n' \
      "${MODELSCOPE_CACHE_ROOT}" "${model_ref}" \
      "${MODELSCOPE_CACHE_ROOT}" "$(modelscope_encoded_ref "${model_ref}")" >&2
    printf '[jetson-power-suite] expected HF under: %s\n' "${HF_CACHE_ROOT}/models--${model_ref//\//--}/snapshots/<revision>" >&2
  elif [[ ! -f "${source_path}/config.json" ]] || ! has_model_weights "${source_path}"; then
    missing_any=1
    printf '[jetson-power-suite] incomplete model cache source: %s -> %s\n' "${model_ref}" "${source_path}" >&2
    printf '[jetson-power-suite] expected config.json and weight files\n' >&2
  fi

  if ! missing_files="$(check_local_export "${export_dir}")"; then
    missing_any=1
    printf '[jetson-power-suite] missing MNN export files for %s: %s\n' "${model_ref}" "${missing_files}" >&2
    if [[ -n "${source_path}" ]]; then
      printf '[jetson-power-suite] export command:\n' >&2
      print_export_command "${model_ref}" "${source_path}" "${dst_name}" >&2
    fi
  fi
done

if [[ "${selected_count}" -eq 0 ]]; then
  die "no fixed models matched MNN_LLM_BENCH_MODEL_FILTER='${MODEL_FILTER}'"
fi

if [[ "${missing_any}" -ne 0 ]]; then
  die "local prerequisites are missing; no rsync, ssh, or power capture was started"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ ! -d "${LOCAL_ARTIFACT_ROOT}" ]]; then
    log "dry run: local artifact root is not present yet: ${LOCAL_ARTIFACT_ROOT}"
  elif [[ ! -x "${LOCAL_ARTIFACT_ROOT}/bin/llm_bench" ]]; then
    log "dry run: local llm_bench is not present yet: ${LOCAL_ARTIFACT_ROOT}/bin/llm_bench"
  elif [[ ! -f "${LOCAL_ARTIFACT_ROOT}/lib/libMNN_Cuda_Main.so" ]]; then
    log "dry run: local CUDA backend is not present yet: ${LOCAL_ARTIFACT_ROOT}/lib/libMNN_Cuda_Main.so"
  fi
else
  if [[ ! -d "${LOCAL_ARTIFACT_ROOT}" ]]; then
    die "local artifact root is missing: ${LOCAL_ARTIFACT_ROOT}"
  fi
  if [[ ! -x "${LOCAL_ARTIFACT_ROOT}/bin/llm_bench" ]]; then
    die "local llm_bench is missing: ${LOCAL_ARTIFACT_ROOT}/bin/llm_bench"
  fi
  if [[ ! -f "${LOCAL_ARTIFACT_ROOT}/lib/libMNN_Cuda_Main.so" ]]; then
    die "local CUDA backend is missing: ${LOCAL_ARTIFACT_ROOT}/lib/libMNN_Cuda_Main.so"
  fi
fi

LOG_ROOT="${REPO_ROOT}/.cache/logs/llm-bench-power/$(date +%Y%m%d_%H%M%S)"
MANIFEST="${LOG_ROOT}/manifest.tsv"
mkdir -p "${LOG_ROOT}"
printf 'model_ref\tmodel_dir\tprompt\tdecode\trep\texit_code\tprefill_speed_tok_s\tdecode_speed_tok_s\tloading_time_s\tremote_command\tlog_path\tpower_csv\n' > "${MANIFEST}"

log "repo: ${REPO_ROOT}"
log "remote: ${REMOTE}"
log "remote_repo: ${REMOTE_REPO}"
log "remote_artifact_root: ${REMOTE_ARTIFACT_ROOT}"
log "remote_model_root: ${REMOTE_MODEL_ROOT}"
if [[ -n "${MODEL_FILTER}" ]]; then
  log "model_filter: ${MODEL_FILTER}"
fi
log "cases: ${CASES}"
log "logs: ${LOG_ROOT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "dry run: skipping rsync, ssh probes, and power API calls"
else
  log "rsync artifact root -> ${REMOTE}:${REMOTE_ARTIFACT_ROOT}/"
  ssh "${REMOTE}" "mkdir -p $(printf '%q' "$(dirname "${REMOTE_ARTIFACT_ROOT}")") $(printf '%q' "${REMOTE_MODEL_ROOT}")"
  rsync -a --delete "${LOCAL_ARTIFACT_ROOT}/" "${REMOTE}:${REMOTE_ARTIFACT_ROOT}/"

  for item in "${MODEL_REFS[@]}"; do
    model_ref="${item%%|*}"
    dst_name="${item##*|}"
    if ! model_selected "${model_ref}" "${dst_name}"; then
      continue
    fi
    log "rsync model ${dst_name} -> ${REMOTE}:${REMOTE_MODEL_ROOT}/${dst_name}/"
    rsync -a --delete "${LOCAL_EXPORT_ROOT}/${dst_name}/" "${REMOTE}:${REMOTE_MODEL_ROOT}/${dst_name}/"
  done

  log "checking remote llm_bench and CUDA backend"
  ssh "${REMOTE}" bash -s <<< "$(remote_probe_script)"
fi

suite_status=0
for item in "${MODEL_REFS[@]}"; do
  model_ref="${item%%|*}"
  dst_name="${item##*|}"
  if ! model_selected "${model_ref}" "${dst_name}"; then
    continue
  fi
  remote_model_dir="${REMOTE_MODEL_ROOT}/${dst_name}"
  safe_name="$(printf '%s' "${dst_name}" | tr -c 'A-Za-z0-9._-' '_')"

  for case_spec in ${CASES}; do
    case_spec="${case_spec//,/}"
    prompt="$(case_prompt "${case_spec}")"
    decode="$(case_decode "${case_spec}")"
    rep="$(case_rep "${case_spec}")"
    case_base="${safe_name}_p${prompt}_n${decode}_rep${rep}"
    case_log="${LOG_ROOT}/${case_base}.log"
    power_csv="${LOG_ROOT}/${case_base}_power.csv"
    remote_script="$(build_remote_bench_script "${remote_model_dir}" "${prompt}" "${decode}" "${rep}")"
    remote_command="ssh ${REMOTE} bash -s < generated-remote-bench-script"

    {
      echo "==> model=${model_ref} case=${prompt}:${decode}:rep${rep}"
      echo "remote_model_dir: ${remote_model_dir}"
      echo "power_csv: ${power_csv}"
      echo "remote_command: ${remote_command}"
    } | tee "${case_log}"

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${model_ref}" "${dst_name}" "${prompt}" "${decode}" "${rep}" "DRY_RUN" "" "" "" "${remote_command}" "${case_log}" "${power_csv}" >> "${MANIFEST}"
      continue
    fi

    set +e
    run_power_wrapped_remote_bench "${remote_script}" "${power_csv}" 2>&1 | tee -a "${case_log}"
    case_status=${PIPESTATUS[0]}
    set -e
    summary="$(extract_bench_summary "${case_log}")"
    speed_field="${summary%%$'\t'*}"
    loading_time_s="${summary##*$'\t'}"
    prefill_speed_tok_s="${speed_field%%<br>*}"
    decode_speed_tok_s="${speed_field##*<br>}"
    [[ "${decode_speed_tok_s}" != "${speed_field}" ]] || decode_speed_tok_s=""

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${model_ref}" "${dst_name}" "${prompt}" "${decode}" "${rep}" "${case_status}" \
      "${prefill_speed_tok_s}" "${decode_speed_tok_s}" "${loading_time_s}" \
      "${remote_command}" "${case_log}" "${power_csv}" >> "${MANIFEST}"

    if [[ "${case_status}" -ne 0 ]]; then
      suite_status="${case_status}"
      log "case failed: model=${model_ref} case=${case_spec} exit=${case_status}"
    fi
  done
done

log "manifest: ${MANIFEST}"
log "logs: ${LOG_ROOT}"
exit "${suite_status}"
