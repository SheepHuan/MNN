#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_pic_decode_repair_tune_warm.sh

Warm an already running MNN PIC server for decode-repair target shapes, then
persist the MNN runtime cache through POST /v1/tune/update_cache. This script
does not run formal TPOT; use it before strict timing runs.

Environment:
  PIC_TUNE_BASE_URL              Server URL. Default: http://127.0.0.1:18091
  PIC_TUNE_MODEL                 Served model name. Default: llama-pic
  PIC_TUNE_OUTPUT_DIR            Log/payload directory. Default: .cache/decode_repair_tune_warm/<timestamp>
  PIC_TUNE_DOC_ID                Text cache id. Default: tune_doc
  PIC_TUNE_PIC_ID                PIC cache id. Default: tune_pic
  PIC_TUNE_DOC_CONTENT           Text used for /v1/prefill/text.
  PIC_TUNE_FORCE_TEXT_CACHE      true/false. Default: true
  PIC_TUNE_SELECTION_ALGORITHM   full-reuse/full-compute/cacheblend/epic/kvshare. Default: full-reuse
  PIC_TUNE_RECOMPUTE_RATIO       PIC recompute ratio for sparse modes. Default: 0.20
  PIC_TUNE_SCORE_LAYER_IDX       PIC score layer. Default: 1
  PIC_TUNE_TPDS                  Space-separated tpd list. Default: "0 1 2 3 4"
  PIC_TUNE_MAX_TOKENS            Decode warm max_tokens. Default: 32
  PIC_TUNE_WARM_REPEATS          Repeats per tpd. Default: 1
  PIC_TUNE_QUESTION              User suffix after {{pic_cache}}.
  PIC_TUNE_DECODE_SELECTOR       decode_refine selector. Default: top_hkvd
  PIC_TUNE_DECODE_TOP_M          decode_refine top_m. Default: 32
  PIC_TUNE_ATTENTION_LAYER_IDX   decode_refine attention_layer_idx. Default: -1
  PIC_TUNE_ATTENTION_HEAD_IDS    Comma-separated head ids. Default: empty
  PIC_TUNE_EXPLICIT_PROMPT_FROM_CACHE  Use PIC cache token_ids as full_prompt_token_ids. Default: 1
  PIC_TUNE_SUFFIX_FROM_CACHE_TOKENS    Append this many cache tokens as explicit suffix. Default: 1
  PIC_TUNE_DRY_RUN               Print payloads without sending when set to 1.
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
  printf '[pic-decode-tune-warm] %s\n' "$*"
}

die() {
  printf '[pic-decode-tune-warm] ERROR: %s\n' "$*" >&2
  exit 1
}

need_tool() {
  command -v "$1" >/dev/null 2>&1 || die "required tool not found: $1"
}

BASE_URL="${PIC_TUNE_BASE_URL:-http://127.0.0.1:18091}"
MODEL="${PIC_TUNE_MODEL:-llama-pic}"
RUN_ID="${PIC_TUNE_RUN_ID:-decode_repair_tune_warm_$(timestamp)}"
OUTPUT_DIR="${PIC_TUNE_OUTPUT_DIR:-${MNN_ROOT}/.cache/decode_repair_tune_warm/${RUN_ID}}"
DOC_ID="${PIC_TUNE_DOC_ID:-tune_doc}"
PIC_ID="${PIC_TUNE_PIC_ID:-tune_pic}"
DOC_CONTENT="${PIC_TUNE_DOC_CONTENT:-MNN PIC decode repair tune warm document. The cache text is intentionally short but stable.}"
FORCE_TEXT_CACHE="${PIC_TUNE_FORCE_TEXT_CACHE:-true}"
SELECTION_ALGORITHM="${PIC_TUNE_SELECTION_ALGORITHM:-full-reuse}"
RECOMPUTE_RATIO="${PIC_TUNE_RECOMPUTE_RATIO:-0.20}"
SCORE_LAYER_IDX="${PIC_TUNE_SCORE_LAYER_IDX:-1}"
TPDS="${PIC_TUNE_TPDS:-0 1 2 3 4}"
MAX_TOKENS="${PIC_TUNE_MAX_TOKENS:-32}"
WARM_REPEATS="${PIC_TUNE_WARM_REPEATS:-1}"
QUESTION="${PIC_TUNE_QUESTION:-Answer briefly using the cached text.}"
DECODE_SELECTOR="${PIC_TUNE_DECODE_SELECTOR:-top_hkvd}"
DECODE_TOP_M="${PIC_TUNE_DECODE_TOP_M:-32}"
ATTENTION_LAYER_IDX="${PIC_TUNE_ATTENTION_LAYER_IDX:--1}"
ATTENTION_HEAD_IDS="${PIC_TUNE_ATTENTION_HEAD_IDS:-}"
EXPLICIT_PROMPT_FROM_CACHE="${PIC_TUNE_EXPLICIT_PROMPT_FROM_CACHE:-1}"
SUFFIX_FROM_CACHE_TOKENS="${PIC_TUNE_SUFFIX_FROM_CACHE_TOKENS:-1}"
DRY_RUN="${PIC_TUNE_DRY_RUN:-0}"

export PIC_TUNE_MODEL="${MODEL}"
export PIC_TUNE_DOC_ID="${DOC_ID}"
export PIC_TUNE_PIC_ID="${PIC_ID}"
export PIC_TUNE_DOC_CONTENT="${DOC_CONTENT}"
export PIC_TUNE_FORCE_TEXT_CACHE="${FORCE_TEXT_CACHE}"
export PIC_TUNE_SELECTION_ALGORITHM="${SELECTION_ALGORITHM}"
export PIC_TUNE_RECOMPUTE_RATIO="${RECOMPUTE_RATIO}"
export PIC_TUNE_SCORE_LAYER_IDX="${SCORE_LAYER_IDX}"
export PIC_TUNE_MAX_TOKENS="${MAX_TOKENS}"
export PIC_TUNE_QUESTION="${QUESTION}"
export PIC_TUNE_DECODE_SELECTOR="${DECODE_SELECTOR}"
export PIC_TUNE_DECODE_TOP_M="${DECODE_TOP_M}"
export PIC_TUNE_ATTENTION_LAYER_IDX="${ATTENTION_LAYER_IDX}"
export PIC_TUNE_ATTENTION_HEAD_IDS="${ATTENTION_HEAD_IDS}"
export PIC_TUNE_FULL_PROMPT_TOKEN_IDS="${PIC_TUNE_FULL_PROMPT_TOKEN_IDS:-}"
export PIC_TUNE_PIC_TOKEN_START="${PIC_TUNE_PIC_TOKEN_START:-0}"
export PIC_TUNE_PIC_TOKEN_COUNT="${PIC_TUNE_PIC_TOKEN_COUNT:-0}"

need_tool python3
if [[ "${DRY_RUN}" != "1" ]]; then
  need_tool curl
fi

mkdir -p "${OUTPUT_DIR}"

write_payload() {
  local kind="$1"
  local tpd="${2:-0}"
  local out="$3"
  PIC_TUNE_PAYLOAD_KIND="${kind}" PIC_TUNE_PAYLOAD_TPD="${tpd}" python3 - >"${out}" <<'PY'
import json
import os

kind = os.environ["PIC_TUNE_PAYLOAD_KIND"]
tpd = int(os.environ.get("PIC_TUNE_PAYLOAD_TPD", "0"))

def env(name, default=""):
    return os.environ.get(name, default)

def env_int(name, default):
    try:
        return int(env(name, str(default)))
    except ValueError:
        return default

def env_float(name, default):
    try:
        return float(env(name, str(default)))
    except ValueError:
        return default

def env_bool(name, default):
    value = env(name, str(default)).strip().lower()
    return value in ("1", "true", "yes", "on")

def head_ids():
    value = env("PIC_TUNE_ATTENTION_HEAD_IDS", "").strip()
    if not value:
        return []
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out

doc_id = env("PIC_TUNE_DOC_ID", "tune_doc")
pic_id = env("PIC_TUNE_PIC_ID", "tune_pic")
pic_spec = {
    "id": pic_id,
    "text_cache_refs": [{"id": doc_id}],
    "selection_algorithm": env("PIC_TUNE_SELECTION_ALGORITHM", "full-reuse"),
    "pic_recompute_ratio": env_float("PIC_TUNE_RECOMPUTE_RATIO", 0.20),
    "pic_recompute_score_layer_idx": env_int("PIC_TUNE_SCORE_LAYER_IDX", 1),
}
full_prompt = env("PIC_TUNE_FULL_PROMPT_TOKEN_IDS", "").strip()
if full_prompt:
    tokens = json.loads(full_prompt)
    pic_spec["full_prompt_token_ids"] = tokens
    pic_spec["pic_token_start"] = env_int("PIC_TUNE_PIC_TOKEN_START", 0)
    pic_spec["token_count"] = env_int("PIC_TUNE_PIC_TOKEN_COUNT", len(tokens))

if kind == "prefill_text":
    payload = {
        "id": doc_id,
        "type": "text",
        "content": env("PIC_TUNE_DOC_CONTENT", ""),
        "force": env_bool("PIC_TUNE_FORCE_TEXT_CACHE", True),
    }
elif kind == "pic_cache":
    payload = pic_spec
elif kind == "chat":
    if tpd > 0:
        pic_spec["decode_refine"] = {
            "enabled": True,
            "tokens_per_decode_step": tpd,
            "top_m": env_int("PIC_TUNE_DECODE_TOP_M", 32),
            "selector": env("PIC_TUNE_DECODE_SELECTOR", "top_hkvd"),
            "attention_layer_idx": env_int("PIC_TUNE_ATTENTION_LAYER_IDX", -1),
            "attention_head_ids": head_ids(),
        }
    payload = {
        "model": env("PIC_TUNE_MODEL", "llama-pic"),
        "messages": [{
            "role": "user",
            "content": "{{pic_cache}}\n" + env("PIC_TUNE_QUESTION", "Answer briefly using the cached text."),
        }],
        "max_tokens": env_int("PIC_TUNE_MAX_TOKENS", 32),
        "temperature": 0,
        "pic_cache": pic_spec,
    }
elif kind == "update_cache":
    payload = {}
else:
    raise SystemExit(f"unknown payload kind: {kind}")

print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
}

post_json() {
  local endpoint="$1"
  local payload="$2"
  local output="$3"
  log "POST ${endpoint}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    cat "${payload}"
    printf '\n' | tee "${output}" >/dev/null
    return 0
  fi
  local status
  status="$(curl -sS -o "${output}" -w '%{http_code}' -X POST "${BASE_URL}${endpoint}" \
    -H 'Content-Type: application/json' \
    --data-binary "@${payload}")"
  if [[ "${status}" -lt 200 || "${status}" -ge 300 ]]; then
    printf '[pic-decode-tune-warm] ERROR: %s returned HTTP %s\n' "${endpoint}" "${status}" >&2
    cat "${output}" >&2 || true
    return 22
  fi
}

log "base_url=${BASE_URL}"
log "output_dir=${OUTPUT_DIR}"
log "selection_algorithm=${SELECTION_ALGORITHM} score_layer=${SCORE_LAYER_IDX} ratio=${RECOMPUTE_RATIO}"
log "tpds=${TPDS} max_tokens=${MAX_TOKENS} repeats=${WARM_REPEATS}"

prefill_payload="${OUTPUT_DIR}/prefill_text.json"
prefill_response="${OUTPUT_DIR}/prefill_text.response.json"
write_payload prefill_text 0 "${prefill_payload}"
post_json /v1/prefill/text "${prefill_payload}" "${prefill_response}"

pic_payload="${OUTPUT_DIR}/pic_cache.json"
pic_response="${OUTPUT_DIR}/pic_cache.response.json"
write_payload pic_cache 0 "${pic_payload}"
post_json /v1/kv/pic_caches "${pic_payload}" "${pic_response}"

if [[ "${DRY_RUN}" != "1" && "${EXPLICIT_PROMPT_FROM_CACHE}" == "1" && -z "${PIC_TUNE_FULL_PROMPT_TOKEN_IDS:-}" ]]; then
  prompt_info="$(python3 - "${pic_response}" "${SUFFIX_FROM_CACHE_TOKENS}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
suffix_count = max(0, int(sys.argv[2]))
tokens = data.get("token_ids") or []
if not tokens:
    raise SystemExit("pic cache response does not contain token_ids")
suffix = tokens[-suffix_count:] if suffix_count > 0 else []
full_prompt = list(tokens) + list(suffix)
print(json.dumps(full_prompt, separators=(",", ":")))
print(len(tokens))
PY
)"
  export PIC_TUNE_FULL_PROMPT_TOKEN_IDS="$(printf '%s\n' "${prompt_info}" | sed -n '1p')"
  export PIC_TUNE_PIC_TOKEN_START="${PIC_TUNE_PIC_TOKEN_START:-0}"
  export PIC_TUNE_PIC_TOKEN_COUNT="$(printf '%s\n' "${prompt_info}" | sed -n '2p')"
  log "explicit full_prompt_token_ids from PIC cache token_count=${PIC_TUNE_PIC_TOKEN_COUNT}"
fi

read -r -a TPD_ARRAY <<<"${TPDS}"
for repeat in $(seq 1 "${WARM_REPEATS}"); do
  for tpd in "${TPD_ARRAY[@]}"; do
    [[ -n "${tpd}" ]] || continue
    payload="${OUTPUT_DIR}/chat_tpd${tpd}_r${repeat}.json"
    response="${OUTPUT_DIR}/chat_tpd${tpd}_r${repeat}.response.json"
    write_payload chat "${tpd}" "${payload}"
    post_json /v1/chat/completions "${payload}" "${response}"
  done
done

update_payload="${OUTPUT_DIR}/update_cache.json"
update_response="${OUTPUT_DIR}/update_cache.response.json"
write_payload update_cache 0 "${update_payload}"
post_json /v1/tune/update_cache "${update_payload}" "${update_response}"

log "warm complete; restart the same server config with the same tmp_path before strict TPOT"
