#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_paged_attention_token_alignment.sh

Environment:
  MNN_ARTIFACT_PLATFORM             Artifact platform name. Defaults to jetson on Jetson/aarch64, x64 on x86_64.
  MNN_ARTIFACT_ROOT                 Artifact root. Defaults to .cache/output/mnn/artifacts/$MNN_ARTIFACT_PLATFORM.
  MNN_LLM_ALIGNMENT_MODEL_CONFIG    Exact PIC/PagedAttention model config.json path.
  MNN_LLM_ALIGNMENT_MODEL           Model dir, config path, or .cache/weight/<name> model name.
  MNN_LLM_BENCH_MODEL_CONFIG        Also accepted as model config when alignment-specific vars are unset.
  MNN_LLM_BENCH_MODEL               Also accepted as model when alignment-specific vars are unset.
  MNN_LLM_ALIGNMENT_BACKENDS        Backend list. Defaults to "cpu cuda" when CUDA artifact exists, otherwise "cpu".
  MNN_LLM_ALIGNMENT_MODES           Generation modes. Defaults to "direct step".
  MNN_LLM_ALIGNMENT_PROMPTS_FILE    TSV prompt file: id<TAB>max_tokens<TAB>prompt.
  MNN_LLM_ALIGNMENT_MAX_TOKENS      Override every prompt's max_tokens.
  MNN_LLM_ALIGNMENT_PRECISION       Runtime precision. Defaults to high.
  MNN_LLM_ALIGNMENT_MEMORY          Runtime memory mode. Defaults to low.
  MNN_LLM_ALIGNMENT_THREADS         Runtime thread count. Defaults to 4.
  MNN_LLM_ALIGNMENT_REQUIRE_EXEC_LOG
                                      Require CPUPagedAttention/CUDAPagedAttention execution logs. Defaults to 1.
  MNN_LLM_ALIGNMENT_DRY_RUN         Print resolved work without running when set to 1.
  CUDA_LIB_DIR                      CUDA lib64 directory. Auto-detected from nvcc when unset.

Default prompt set:
  short_exact, medium_exact, long_rules, long_repeated_context.

The script compiles a small PIC LLM token helper into .cache/build/mnn-llm-bench/token-alignment
and writes logs plus report.md into .cache/logs/paged-attention-token-alignment/<timestamp>/.
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
  local model="${MNN_LLM_ALIGNMENT_MODEL_CONFIG:-${MNN_LLM_ALIGNMENT_MODEL:-${MNN_LLM_BENCH_MODEL_CONFIG:-${MNN_LLM_BENCH_MODEL:-}}}}"
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
  echo "Could not resolve model config. Set MNN_LLM_ALIGNMENT_MODEL_CONFIG or MNN_LLM_ALIGNMENT_MODEL." >&2
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

if ! grep -q '"paged_attention"[[:space:]]*:[[:space:]]*true' "${LLM_CONFIG}"; then
  echo "Model does not look like a PIC/PagedAttention model: ${LLM_CONFIG}" >&2
  echo "Set a model exported with paged_attention=true." >&2
  exit 2
fi

for required in "${ARTIFACT_ROOT}/lib/libMNN.so" "${ARTIFACT_ROOT}/lib/libMNN_Express.so" "${ARTIFACT_ROOT}/lib/libpic_llm.so"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required artifact missing: ${required}" >&2
    exit 2
  fi
done

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

if [[ -n "${MNN_LLM_ALIGNMENT_BACKENDS:-}" ]]; then
  BACKENDS="${MNN_LLM_ALIGNMENT_BACKENDS}"
elif [[ -f "${ARTIFACT_ROOT}/lib/libMNN_Cuda_Main.so" ]]; then
  BACKENDS="cpu cuda"
else
  BACKENDS="cpu"
fi

for backend in ${BACKENDS}; do
  case "${backend}" in
    cpu) ;;
    cuda)
      if [[ ! -f "${ARTIFACT_ROOT}/lib/libMNN_Cuda_Main.so" ]]; then
        echo "CUDA backend requested but missing: ${ARTIFACT_ROOT}/lib/libMNN_Cuda_Main.so" >&2
        exit 2
      fi
      ;;
    *)
      echo "Unsupported backend for token alignment: ${backend}" >&2
      exit 2
      ;;
  esac
done

MODES="${MNN_LLM_ALIGNMENT_MODES:-direct step}"
for mode in ${MODES}; do
  case "${mode}" in
    direct|step) ;;
    *)
      echo "Unsupported generation mode: ${mode}" >&2
      exit 2
      ;;
  esac
done

PRECISION="${MNN_LLM_ALIGNMENT_PRECISION:-high}"
MEMORY="${MNN_LLM_ALIGNMENT_MEMORY:-low}"
THREADS="${MNN_LLM_ALIGNMENT_THREADS:-4}"
REQUIRE_EXEC_LOG="${MNN_LLM_ALIGNMENT_REQUIRE_EXEC_LOG:-1}"
DRY_RUN="${MNN_LLM_ALIGNMENT_DRY_RUN:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${REPO_ROOT}/.cache/logs/paged-attention-token-alignment/${STAMP}"
BUILD_ROOT="${REPO_ROOT}/.cache/build/mnn-llm-bench/token-alignment"
HELPER_SRC="${BUILD_ROOT}/pic_llm_token_alignment.cpp"
HELPER_BIN="${BUILD_ROOT}/pic_llm_token_alignment"
PROMPTS_TSV="${LOG_ROOT}/prompts.tsv"
MANIFEST="${LOG_ROOT}/manifest.tsv"
REPORT="${LOG_ROOT}/report.md"

mkdir -p "${LOG_ROOT}" "${BUILD_ROOT}"

if [[ -n "${MNN_LLM_ALIGNMENT_PROMPTS_FILE:-}" ]]; then
  PROMPTS_FILE="$(normalize_path "${MNN_LLM_ALIGNMENT_PROMPTS_FILE}")"
  if [[ ! -f "${PROMPTS_FILE}" ]]; then
    echo "Prompt file not found: ${PROMPTS_FILE}" >&2
    exit 2
  fi
  cp "${PROMPTS_FILE}" "${PROMPTS_TSV}"
else
  python3 - "${PROMPTS_TSV}" <<'PY'
import sys

out = sys.argv[1]

context_rules = " ".join(
    f"Rule {i:02d}: keep the final answer short, deterministic, and copied from the requested sequence."
    for i in range(1, 22)
)
repeated_context = " ".join(
    f"Segment {i:03d} says the alignment test is about preserving KV cache order across prefill and decode."
    for i in range(1, 29)
)

prompts = [
    (
        "short_exact",
        6,
        "Answer with exactly this five-word sequence: alpha beta gamma delta epsilon",
    ),
    (
        "medium_exact",
        10,
        "This is a deterministic decoding check. Do not explain. Do not add punctuation. "
        "Return exactly this eight-word sequence: red blue green yellow black white orange purple",
    ),
    (
        "long_rules",
        12,
        context_rules
        + " Final instruction: output exactly these six words and nothing else: spring summer autumn winter dawn dusk",
    ),
    (
        "long_repeated_context",
        12,
        repeated_context
        + " Final instruction: output exactly these six words and nothing else: mercury venus earth mars jupiter saturn",
    ),
]

with open(out, "w", encoding="utf-8") as f:
    f.write("# id\tmax_tokens\tprompt\n")
    for name, max_tokens, prompt in prompts:
        prompt = prompt.replace("\t", " ").replace("\n", " ")
        f.write(f"{name}\t{max_tokens}\t{prompt}\n")
PY
fi

cat > "${HELPER_SRC}" <<'CPP'
#include "llm/llm.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MNN::Transformer;

static std::string readFile(const std::string& path) {
    std::ifstream file(path);
    std::ostringstream os;
    os << file.rdbuf();
    return os.str();
}

static std::string jsonEscape(const std::string& text) {
    std::ostringstream os;
    os << '"';
    for (unsigned char c : text) {
        switch (c) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (c < 0x20) {
                    const char* hex = "0123456789abcdef";
                    os << "\\u00" << hex[(c >> 4) & 0x0f] << hex[c & 0x0f];
                } else {
                    os << static_cast<char>(c);
                }
                break;
        }
    }
    os << '"';
    return os.str();
}

static void printTokens(const std::vector<int>& tokens) {
    std::cout << "OUTPUT_TOKEN_IDS=[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << tokens[i];
    }
    std::cout << "]\n";
}

static std::string runtimeConfig(const std::string& backend, const std::string& precision,
                                 const std::string& memory, int threads, int maxTokens) {
    std::ostringstream os;
    os << "{";
    os << "\"backend_type\":\"" << backend << "\",";
    os << "\"precision\":\"" << precision << "\",";
    os << "\"memory\":\"" << memory << "\",";
    os << "\"thread_num\":" << threads << ",";
    os << "\"sampler_type\":\"greedy\",";
    os << "\"temperature\":0.0,";
    os << "\"top_k\":1,";
    os << "\"top_p\":1.0,";
    os << "\"repetition_penalty\":1.0,";
    os << "\"penalty_window\":0,";
    os << "\"max_new_tokens\":" << maxTokens << ",";
    os << "\"tmp_path\":\"tmp\",";
    os << "\"async\":false";
    os << "}";
    return os.str();
}

int main(int argc, const char* argv[]) {
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0]
                  << " config.json prompt.txt backend precision memory threads max_tokens direct|step\n";
        return 2;
    }

    const std::string configPath = argv[1];
    const std::string promptPath = argv[2];
    const std::string backend = argv[3];
    const std::string precision = argv[4];
    const std::string memory = argv[5];
    const int threads = std::atoi(argv[6]);
    const int maxTokens = std::atoi(argv[7]);
    const std::string mode = argv[8];
    const std::string prompt = readFile(promptPath);

    if (prompt.empty()) {
        std::cerr << "Prompt is empty: " << promptPath << "\n";
        return 2;
    }
    if (mode != "direct" && mode != "step") {
        std::cerr << "Unsupported mode: " << mode << "\n";
        return 2;
    }

    std::unique_ptr<Llm> llm(Llm::createLLM(configPath));
    llm->set_config(runtimeConfig(backend, precision, memory, threads, maxTokens));
    if (!llm->load()) {
        std::cerr << "LLM load failed\n";
        return 1;
    }

    std::ostringstream generated;
    if (mode == "direct") {
        llm->response(prompt, &generated, nullptr, maxTokens);
    } else {
        llm->set_config("{\"max_new_tokens\":1}");
        llm->response(prompt, &generated, nullptr, 0);
        auto context = llm->getContext();
        while (!llm->stoped() && context->gen_seq_len < maxTokens) {
            llm->generate(1);
            context = llm->getContext();
            if (context->status == LlmStatus::INTERNAL_ERROR || context->status == LlmStatus::TIMEOUT ||
                context->status == LlmStatus::USER_CANCEL) {
                std::cerr << "Generation interrupted with status " << static_cast<int>(context->status) << "\n";
                return 1;
            }
        }
    }

    auto context = llm->getContext();
    std::cout << "BACKEND=" << backend << "\n";
    std::cout << "MODE=" << mode << "\n";
    std::cout << "PROMPT_TOKEN_COUNT=" << context->prompt_len << "\n";
    std::cout << "DECODE_TOKEN_COUNT=" << context->gen_seq_len << "\n";
    std::cout << "STATUS=" << static_cast<int>(context->status) << "\n";
    printTokens(context->output_tokens);
    std::cout << "GENERATED_TEXT_JSON=" << jsonEscape(generated.str()) << "\n";
    return 0;
}
CPP

compile_helper() {
  local cxx="${CXX:-g++}"
  "${cxx}" -std=c++17 -O2 \
    -I"${REPO_ROOT}/transformers/pic_llm/engine/include" \
    -I"${REPO_ROOT}/transformers/pic_llm/engine/src" \
    -I"${REPO_ROOT}/include" \
    -I"${REPO_ROOT}/express" \
    "${HELPER_SRC}" \
    -L"${ARTIFACT_ROOT}/lib" \
    -Wl,-rpath,"${ARTIFACT_ROOT}/lib" \
    -lpic_llm -lMNN_Express -lMNN -ldl -lpthread \
    -o "${HELPER_BIN}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  compile_helper
fi

{
  echo "repo: ${REPO_ROOT}"
  echo "artifact_platform: ${ARTIFACT_PLATFORM}"
  echo "artifact_root: ${ARTIFACT_ROOT}"
  echo "model_config: ${MODEL_CONFIG}"
  echo "model_dir: ${MODEL_DIR}"
  echo "helper: ${HELPER_BIN}"
  echo "backends: ${BACKENDS}"
  echo "modes: ${MODES}"
  echo "precision: ${PRECISION}"
  echo "memory: ${MEMORY}"
  echo "threads: ${THREADS}"
  echo "require_execution_log: ${REQUIRE_EXEC_LOG}"
  echo "prompts_tsv: ${PROMPTS_TSV}"
  echo "report: ${REPORT}"
} | tee "${LOG_ROOT}/summary.log"

printf 'prompt_id\tmax_tokens\tbackend\tmode\texit_code\tlog_path\n' > "${MANIFEST}"

while IFS=$'\t' read -r prompt_id prompt_max_tokens prompt_text; do
  [[ -n "${prompt_id}" ]] || continue
  [[ "${prompt_id}" != \#* ]] || continue
  if [[ -n "${MNN_LLM_ALIGNMENT_MAX_TOKENS:-}" ]]; then
    prompt_max_tokens="${MNN_LLM_ALIGNMENT_MAX_TOKENS}"
  fi
  prompt_dir="${LOG_ROOT}/${prompt_id}"
  mkdir -p "${prompt_dir}"
  prompt_file="${prompt_dir}/prompt.txt"
  printf '%s' "${prompt_text}" > "${prompt_file}"

  for backend in ${BACKENDS}; do
    for mode in ${MODES}; do
      run_dir="${prompt_dir}/${backend}_${mode}"
      mkdir -p "${run_dir}/tmp"
      log_file="${run_dir}/run.log"
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY_RUN ${backend}/${mode} ${prompt_id}: ${HELPER_BIN} ${MODEL_CONFIG} ${prompt_file} ${backend} ${PRECISION} ${MEMORY} ${THREADS} ${prompt_max_tokens} ${mode}" | tee "${log_file}"
        exit_code=0
      else
        set +e
        (
          cd "${run_dir}"
          LD_LIBRARY_PATH="${LD_PATH}" \
            "${HELPER_BIN}" "${MODEL_CONFIG}" "${prompt_file}" "${backend}" "${PRECISION}" "${MEMORY}" \
            "${THREADS}" "${prompt_max_tokens}" "${mode}"
        ) > "${log_file}" 2>&1
        exit_code=$?
        set -e
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${prompt_id}" "${prompt_max_tokens}" "${backend}" "${mode}" "${exit_code}" "${log_file}" >> "${MANIFEST}"
      echo "[${exit_code}] ${prompt_id} ${backend}/${mode} -> ${log_file}"
    done
  done
done < "${PROMPTS_TSV}"

python3 - "${MANIFEST}" "${REPORT}" "${REQUIRE_EXEC_LOG}" <<'PY'
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

manifest = Path(sys.argv[1])
report = Path(sys.argv[2])
require_exec_log = sys.argv[3] == "1"

rows = []
with manifest.open("r", encoding="utf-8") as f:
    header = f.readline().rstrip("\n").split("\t")
    for line in f:
        if not line.strip():
            continue
        item = dict(zip(header, line.rstrip("\n").split("\t")))
        text = Path(item["log_path"]).read_text(encoding="utf-8", errors="replace")
        item["log_text"] = text
        token_match = re.search(r"^OUTPUT_TOKEN_IDS=(\[.*\])$", text, re.M)
        item["tokens"] = ast.literal_eval(token_match.group(1)) if token_match else None
        decode_match = re.search(r"^DECODE_TOKEN_COUNT=([0-9]+)$", text, re.M)
        prompt_match = re.search(r"^PROMPT_TOKEN_COUNT=([0-9]+)$", text, re.M)
        status_match = re.search(r"^STATUS=(-?[0-9]+)$", text, re.M)
        text_match = re.search(r"^GENERATED_TEXT_JSON=(.*)$", text, re.M)
        item["decode_tokens"] = int(decode_match.group(1)) if decode_match else -1
        item["prompt_tokens"] = int(prompt_match.group(1)) if prompt_match else -1
        item["status"] = int(status_match.group(1)) if status_match else None
        item["generated_text"] = json.loads(text_match.group(1)) if text_match else ""
        if item["backend"] == "cuda":
            item["exec_log_ok"] = ("CUDAPagedAttention" in text) or ("[CUDAExecution]" in text and "PagedAttention" in text)
        elif item["backend"] == "cpu":
            item["exec_log_ok"] = ("CPUPagedAttention" in text) or ("[CPUExecution]" in text and "PagedAttention" in text)
        else:
            item["exec_log_ok"] = True
        rows.append(item)

by_prompt = defaultdict(list)
for row in rows:
    by_prompt[row["prompt_id"]].append(row)

failures = []
for prompt_id, cases in by_prompt.items():
    baseline = None
    for row in cases:
        if row["backend"] == "cpu" and row["mode"] == "direct" and row["tokens"] is not None and row["exit_code"] == "0":
            baseline = row
            break
    if baseline is None:
        for row in cases:
            if row["tokens"] is not None and row["exit_code"] == "0":
                baseline = row
                break
    for row in cases:
        row["baseline"] = baseline
        row["matches_baseline"] = bool(baseline and row["tokens"] == baseline["tokens"])
        if row["exit_code"] != "0":
            failures.append(f"{prompt_id} {row['backend']}/{row['mode']} exited {row['exit_code']}")
        elif row["tokens"] is None:
            failures.append(f"{prompt_id} {row['backend']}/{row['mode']} did not print token ids")
        elif not row["matches_baseline"]:
            failures.append(f"{prompt_id} {row['backend']}/{row['mode']} token ids differ from baseline")
        elif require_exec_log and not row["exec_log_ok"]:
            failures.append(f"{prompt_id} {row['backend']}/{row['mode']} missing expected PagedAttention execution log")

lines = []
lines.append("# PagedAttention Token Alignment Report")
lines.append("")
lines.append(f"- manifest: `{manifest}`")
lines.append(f"- require_execution_log: `{int(require_exec_log)}`")
lines.append(f"- total_cases: `{len(rows)}`")
lines.append(f"- failed_cases: `{len(failures)}`")
lines.append("")
lines.append("| prompt | backend | mode | exit | prompt_tokens | decode_tokens | status | exec_log | match | token_ids | text_preview |")
lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
for row in rows:
    tokens = row["tokens"] if row["tokens"] is not None else []
    token_preview = "[" + ",".join(str(x) for x in tokens[:16]) + (",..." if len(tokens) > 16 else "") + "]"
    text_preview = row["generated_text"].replace("\n", "\\n")
    if len(text_preview) > 80:
        text_preview = text_preview[:77] + "..."
    lines.append(
        f"| {row['prompt_id']} | {row['backend']} | {row['mode']} | {row['exit_code']} | "
        f"{row['prompt_tokens']} | {row['decode_tokens']} | {row['status']} | "
        f"{'yes' if row['exec_log_ok'] else 'no'} | {'yes' if row['matches_baseline'] else 'no'} | "
        f"`{token_preview}` | `{text_preview}` |"
    )

if failures:
    lines.append("")
    lines.append("## Failures")
    lines.extend(f"- {failure}" for failure in failures)

report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"report: {report}")
if failures:
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    sys.exit(1)
PY

echo "PagedAttention token alignment passed."
echo "Report: ${REPORT}"
