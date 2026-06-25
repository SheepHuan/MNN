#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  export_modelscope_llm.sh
  export_modelscope_llm.sh MODEL_REF_OR_PATH [DST_NAME] [-- EXTRA_LLMEXPORT_ARGS...]

Examples:
  bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh
  bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh Qwen/Qwen3.5-2B
  MNN_LLM_EXPORTER=pic bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh AI-ModelScope/Llama-3.2-3B-Instruct -- --paged_kv_max_tokens 4096
  MNN_LLM_EXPORTER=prefix MNN_LLM_EXPORT_SCRIPT=path/to/prefixllm/export/llmexport.py bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh AI-ModelScope/Llama-3.2-1B-Instruct
  MNN_QUANT_BIT=8 bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh Qwen/Qwen3.5-2B
  bash .codex/skills/mnn-llm-export/scripts/export_modelscope_llm.sh "$MODELSCOPE_CACHE_ROOT/Qwen/Qwen3___5-2B"

Environment:
  MODELSCOPE_CACHE_ROOT  Override the ModelScope models cache root.
  MNN_LLM_EXPORTER       Exporter to use: mnn, pic or prefix. Defaults to mnn.
  MNN_LLM_EXPORT_SCRIPT   Exact llmexport.py path. Required for prefix exporter.
  MNN_LLM_EXPORT_ROOT    Output root. Defaults to .cache/mnn-llm-export for mnn and .cache/weight for pic/prefix.
                          When set, it must resolve to the selected exporter's default root.
  MNN_LLM_EXPORT_DST     Exact output directory. Overrides the generated destination and must be a direct child of the selected root.
  MNN_ARTIFACT_PLATFORM  Output artifact platform name. Defaults to jetson on aarch64/Jetson, x64 on x86_64.
  MNN_ARTIFACT_ROOT      Built artifact root. Defaults to .cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM.
  MNN_QUANT_BIT          MNN weight quant bit. Defaults to 4.
  MNN_QUANT_BLOCK        MNN weight quant block. Defaults to 64.
  MNN_EMBED_BIT          Embedding bit. Defaults to 16.
  MNN_PAGED_KV_MAX_TOKENS  Default PIC export paged KV token limit. Defaults to 4096.
  MNNCONVERT_PATH        Optional MNNConvert executable path. Auto-detected under MNN_ARTIFACT_ROOT when unset.
  MNN_EXPORT_DRY_RUN     Print resolved command without running export when set to 1.
EOF
}

DEFAULT_MODEL_REFS=(
  "AI-ModelScope/Llama-3___2-3B-Instruct"
)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  default_exporter="${MNN_LLM_EXPORTER:-mnn}"
  default_extra_args=()
  case "${default_exporter}" in
    pic|pic_llm|pic-llm|paged|paged-attention)
      default_extra_args=(-- --paged_kv_max_tokens "${MNN_PAGED_KV_MAX_TOKENS:-4096}")
      ;;
  esac

  status=0
  for default_model_ref in "${DEFAULT_MODEL_REFS[@]}"; do
    echo "==> default export: ${default_model_ref}"
    if ! MNN_LLM_EXPORTER="${default_exporter}" bash "${BASH_SOURCE[0]}" "${default_model_ref}" "${default_extra_args[@]}"; then
      status=1
    fi
  done
  exit "${status}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
WEIGHT_ROOT="${REPO_ROOT}/.cache/weight"
STANDARD_ROOT="${REPO_ROOT}/.cache/mnn-llm-export"

normalize_repo_path() {
  local path="$1"
  if [[ "${path}" != /* ]]; then
    path="${REPO_ROOT}/${path}"
  fi
  realpath -m "${path}"
}

sanitize_model_name() {
  local raw="$1"
  raw="${raw//\//__}"
  printf '%s' "${raw}" | tr -c 'A-Za-z0-9._-' '_'
}

require_export_root() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [[ "${path}" != "${expected}" ]]; then
    echo "MNN_LLM_EXPORT_ROOT must be exactly under this repo as ${label}" >&2
    echo "resolved root: ${path}" >&2
    echo "required root: ${expected}" >&2
    exit 2
  fi
}

require_export_dst() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [[ "$(dirname "${path}")" != "${expected}" ]]; then
    echo "Export destination must be a direct child of ${label}: ${label}/<model-name>" >&2
    echo "resolved dst: ${path}" >&2
    echo "required parent: ${expected}" >&2
    exit 2
  fi
}

detect_artifact_platform() {
  local model=""

  if [[ -r /proc/device-tree/model ]]; then
    model="$(tr -d '\0' </proc/device-tree/model)"
  elif [[ -r /sys/firmware/devicetree/base/model ]]; then
    model="$(tr -d '\0' </sys/firmware/devicetree/base/model)"
  fi

  case "${model}" in
    *Jetson*|*NVIDIA*)
      printf 'jetson'
      return 0
      ;;
  esac

  case "$(uname -m)" in
    x86_64|amd64)
      printf 'x64'
      ;;
    aarch64|arm64)
      printf 'jetson'
      ;;
    *)
      uname -m
      ;;
  esac
}

ARTIFACT_PLATFORM="${MNN_ARTIFACT_PLATFORM:-$(detect_artifact_platform)}"
ARTIFACT_ROOT="${MNN_ARTIFACT_ROOT:-${REPO_ROOT}/.cache/output/mnn/artifacts/${ARTIFACT_PLATFORM}}"
if [[ "${ARTIFACT_ROOT}" != /* ]]; then
  ARTIFACT_ROOT="${REPO_ROOT}/${ARTIFACT_ROOT}"
fi

MODEL_REF="$1"
shift

DST_NAME=""
if [[ $# -gt 0 && "${1:-}" != "--" ]]; then
  DST_NAME="$1"
  shift
fi
if [[ $# -gt 0 && "${1:-}" == "--" ]]; then
  shift
fi
EXTRA_ARGS=("$@")

QUANT_BIT="${MNN_QUANT_BIT:-4}"
QUANT_BLOCK="${MNN_QUANT_BLOCK:-64}"
EMBED_BIT="${MNN_EMBED_BIT:-16}"
EXPORTER="${MNN_LLM_EXPORTER:-mnn}"
case "${EXPORTER}" in
  mnn|default|standard)
    EXPORTER="mnn"
    DEFAULT_EXPORT_ROOT="${STANDARD_ROOT}"
    DEFAULT_EXPORT_ROOT_LABEL=".cache/mnn-llm-export"
    DEFAULT_EXPORT_SCRIPT="${REPO_ROOT}/transformers/llm/export/llmexport.py"
    ;;
  pic|pic_llm|pic-llm|paged|paged-attention)
    EXPORTER="pic"
    DEFAULT_EXPORT_ROOT="${WEIGHT_ROOT}"
    DEFAULT_EXPORT_ROOT_LABEL=".cache/weight"
    DEFAULT_EXPORT_SCRIPT="${REPO_ROOT}/transformers/pic_llm/export/llmexport.py"
    ;;
  prefix|prefixllm|prefix-llm)
    EXPORTER="prefix"
    DEFAULT_EXPORT_ROOT="${WEIGHT_ROOT}"
    DEFAULT_EXPORT_ROOT_LABEL=".cache/weight"
    DEFAULT_EXPORT_SCRIPT=""
    ;;
  *)
    echo "Unsupported MNN_LLM_EXPORTER: ${EXPORTER}" >&2
    echo "Expected one of: mnn, pic, prefix" >&2
    exit 2
    ;;
esac
EXPORT_ROOT="$(normalize_repo_path "${MNN_LLM_EXPORT_ROOT:-${DEFAULT_EXPORT_ROOT}}")"
require_export_root "${EXPORT_ROOT}" "${DEFAULT_EXPORT_ROOT}" "${DEFAULT_EXPORT_ROOT_LABEL}"
if [[ -n "${MNN_LLM_EXPORT_SCRIPT:-}" ]]; then
  EXPORT_SCRIPT="${MNN_LLM_EXPORT_SCRIPT}"
elif [[ -n "${DEFAULT_EXPORT_SCRIPT}" ]]; then
  EXPORT_SCRIPT="${DEFAULT_EXPORT_SCRIPT}"
else
  echo "MNN_LLM_EXPORT_SCRIPT is required when MNN_LLM_EXPORTER=prefix" >&2
  exit 2
fi
if [[ "${EXPORT_SCRIPT}" != /* ]]; then
  EXPORT_SCRIPT="${REPO_ROOT}/${EXPORT_SCRIPT}"
fi
if [[ ! -f "${EXPORT_SCRIPT}" ]]; then
  echo "llmexport.py not found: ${EXPORT_SCRIPT}" >&2
  exit 2
fi

cache_roots=()
if [[ -n "${MODELSCOPE_CACHE_ROOT:-}" ]]; then
  cache_roots+=("${MODELSCOPE_CACHE_ROOT}")
else
  cache_roots+=("${HOME}/.cache/modelscope/hub/models")
fi

encoded_ref="${MODEL_REF//./___}"

resolve_model_path() {
  if [[ -d "${MODEL_REF}" ]]; then
    realpath "${MODEL_REF}"
    return 0
  fi

  local root candidate
  for root in "${cache_roots[@]}"; do
    [[ -d "${root}" ]] || continue
    for candidate in "${root}/${MODEL_REF}" "${root}/${encoded_ref}"; do
      if [[ -d "${candidate}" ]]; then
        realpath "${candidate}"
        return 0
      fi
    done
  done

  local name encoded_name found
  name="$(basename "${MODEL_REF}")"
  encoded_name="${name//./___}"
  for root in "${cache_roots[@]}"; do
    [[ -d "${root}" ]] || continue
    found="$(find "${root}" -maxdepth 3 -type d \( -name "${name}" -o -name "${encoded_name}" \) -print -quit 2>/dev/null || true)"
    if [[ -n "${found}" ]]; then
      realpath "${found}"
      return 0
    fi
  done

  return 1
}

MODEL_PATH="$(resolve_model_path || true)"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "Could not find ModelScope model cache for: ${MODEL_REF}" >&2
  echo "Searched roots:" >&2
  printf '  %s\n' "${cache_roots[@]}" >&2
  exit 2
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model path is missing config.json: ${MODEL_PATH}" >&2
  exit 2
fi

has_hf_weight=0
if find "${MODEL_PATH}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'model*.bin' \) -print -quit | grep -q .; then
  has_hf_weight=1
fi
if [[ "${has_hf_weight}" -ne 1 ]]; then
  if [[ -f "${MODEL_PATH}/llm.mnn" ]]; then
    echo "This looks like an already-exported MNN model, not a Hugging Face source model: ${MODEL_PATH}" >&2
  else
    echo "No Hugging Face weight file found in: ${MODEL_PATH}" >&2
  fi
  exit 2
fi

if [[ -n "${MNN_LLM_EXPORT_DST:-}" ]]; then
  DST_PATH="$(normalize_repo_path "${MNN_LLM_EXPORT_DST}")"
elif [[ -n "${DST_NAME}" ]]; then
  DST_NAME="$(sanitize_model_name "${DST_NAME}")"
  DST_PATH="${EXPORT_ROOT}/${DST_NAME}"
else
  model_base="$(basename "${MODEL_PATH}")"
  owner_base="$(basename "$(dirname "${MODEL_PATH}")")"
  DST_PATH="${EXPORT_ROOT}/$(sanitize_model_name "${owner_base}__${model_base}")"
fi
require_export_dst "${DST_PATH}" "${EXPORT_ROOT}" "${DEFAULT_EXPORT_ROOT_LABEL}"

LOG_DIR="${REPO_ROOT}/.cache/logs/${EXPORTER}-llm-export"
mkdir -p "${DST_PATH}" "${LOG_DIR}"

safe_name="$(printf '%s' "$(basename "${DST_PATH}")" | tr -c 'A-Za-z0-9._-' '_')"
stamp="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/${stamp}_${safe_name}.log"

resolve_mnnconvert() {
  local candidate
  local candidates=()

  if [[ -n "${MNNCONVERT_PATH:-}" ]]; then
    candidates+=("${MNNCONVERT_PATH}")
  else
    candidates+=(
      "${ARTIFACT_ROOT}/bin/MNNConvert"
      "${ARTIFACT_ROOT}/MNNConvert"
      "${REPO_ROOT}/.cache/build/${ARTIFACT_PLATFORM}_cuda/MNNConvert"
      "${REPO_ROOT}/.cache/build/${ARTIFACT_PLATFORM}/MNNConvert"
      "${REPO_ROOT}/.cache/build/mnn/${ARTIFACT_PLATFORM}_cuda/MNNConvert"
      "${REPO_ROOT}/.cache/build/mnn/${ARTIFACT_PLATFORM}/MNNConvert"
      "${REPO_ROOT}/.cache/build/mnn/jetson_cuda/MNNConvert"
      "${REPO_ROOT}/.cache/build/mnn/x64/MNNConvert"
    )
  fi

  for candidate in "${candidates[@]}"; do
    if [[ "${candidate}" != /* ]]; then
      candidate="${REPO_ROOT}/${candidate}"
    fi
    if [[ -x "${candidate}" ]]; then
      realpath "${candidate}"
      return 0
    fi
  done

  return 1
}

mnnconvert_args=()
RESOLVED_MNNCONVERT_PATH="$(resolve_mnnconvert || true)"
if [[ -n "${RESOLVED_MNNCONVERT_PATH}" ]]; then
  mnnconvert_args=(--mnnconvert "${RESOLVED_MNNCONVERT_PATH}")
  converter_bin_dir="$(dirname "${RESOLVED_MNNCONVERT_PATH}")"
  converter_lib_dirs=("${converter_bin_dir}")
  for candidate in \
    "${converter_bin_dir}/../lib" \
    "${converter_bin_dir}/lib" \
    "${converter_bin_dir}/express" \
    "${converter_bin_dir}/tools/converter"
  do
    if [[ -d "${candidate}" ]]; then
      converter_lib_dirs+=("$(cd "${candidate}" && pwd)")
    fi
  done
  converter_ld_path="$(IFS=:; printf '%s' "${converter_lib_dirs[*]}")"
  export LD_LIBRARY_PATH="${converter_ld_path}:${LD_LIBRARY_PATH:-}"
elif [[ "${EXPORTER}" == "pic" || "${EXPORTER}" == "prefix" ]]; then
  echo "MNNConvert not found for ${EXPORTER} export." >&2
  echo "Build artifacts first, or set MNNCONVERT_PATH explicitly." >&2
  echo "Searched artifact root: ${ARTIFACT_ROOT}" >&2
  exit 2
fi

python_cmd=(conda run -n kvshare-edge python)

cmd=(
  "${python_cmd[@]}"
  "${EXPORT_SCRIPT}"
  --path "${MODEL_PATH}"
  --dst_path "${DST_PATH}"
  --export mnn
  --quant_bit "${QUANT_BIT}"
  --quant_block "${QUANT_BLOCK}"
  --embed_bit "${EMBED_BIT}"
  "${mnnconvert_args[@]}"
  "${EXTRA_ARGS[@]}"
)

echo "repo: ${REPO_ROOT}"
echo "exporter: ${EXPORTER}"
echo "script: ${EXPORT_SCRIPT}"
echo "artifact_platform: ${ARTIFACT_PLATFORM}"
echo "artifact_root: ${ARTIFACT_ROOT}"
echo "mnnconvert: ${RESOLVED_MNNCONVERT_PATH:-<pymnn fallback>}"
echo "model: ${MODEL_PATH}"
echo "dst: ${DST_PATH}"
echo "log: ${LOG_PATH}"
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${MNN_EXPORT_DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run: export command was not executed"
  exit 0
fi

(
  cd "${REPO_ROOT}"
  "${cmd[@]}"
) 2>&1 | tee "${LOG_PATH}"

missing=()
for file in config.json llm_config.json llm.mnn llm.mnn.weight; do
  if [[ ! -f "${DST_PATH}/${file}" ]]; then
    missing+=("${file}")
  fi
done

if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "Export finished but missing expected artifact(s): ${missing[*]}" >&2
  echo "Check log: ${LOG_PATH}" >&2
  exit 3
fi

if [[ "${EXPORTER}" == "prefix" ]]; then
  if ! grep -Eq '"prefix_attention"[[:space:]]*:[[:space:]]*true' "${DST_PATH}/llm_config.json"; then
    echo "Prefix export finished but llm_config.json does not contain prefix_attention: true" >&2
    echo "Check log: ${LOG_PATH}" >&2
    exit 3
  fi
fi

if [[ "${EXPORTER}" == "pic" ]]; then
  if ! grep -Eq '"paged_attention"[[:space:]]*:[[:space:]]*true' "${DST_PATH}/llm_config.json"; then
    echo "PIC export finished but llm_config.json does not contain paged_attention: true" >&2
    echo "Check log: ${LOG_PATH}" >&2
    exit 3
  fi
fi

echo "Export complete: ${DST_PATH}/config.json"
echo "Artifacts:"
find "${DST_PATH}" -maxdepth 1 -type f | sort
