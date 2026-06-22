#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_remote_weight_only_conv_tune.sh

Run focused CUDA WeightOnlyConv direct-op sweeps on Jetson. This is for
decode-repair active-row shape policy screening before end-to-end TPOT.

Environment:
  MNN_TUNE_REMOTE          SSH target. Default: jetson@192.168.101.192
  MNN_TUNE_REMOTE_REPO     Remote MNN repo. Default: /home/jetson/code/kvshare-edge/impl/MNN
  MNN_TUNE_ART_REL         Artifact root relative to repo. Default: .cache/output/mnn/artifacts/jetson_cross_cuda
  MNN_TUNE_CUDA_LIB        Remote CUDA lib dir. Default: /usr/local/cuda-12.2/targets/aarch64-linux/lib
  MNN_TUNE_CASES           Space-separated WeightOnlyConv cases.
                           Default: "hidden_to_inter inter_to_hidden hidden_to_gateup_concat"
  MNN_TUNE_ROWS            Row filter passed to MNN_BENCH_WEIGHT_ONLY_ROWS. Default: 4-5
  MNN_TUNE_WARMUP          Warmup iterations. Default: 20
  MNN_TUNE_REPEAT          Timed iterations. Default: 80
  MNN_TUNE_PRECISION       run_test precision arg. Default: 2 (fp16)
  MNN_TUNE_MEMORY          run_test memory arg. Default: 2 (Memory_Low, required for ConvFpAIntBExecution)
  MNN_TUNE_LOG_DIR         Remote log dir. Default: .cache/bench_ops/decode_repair_tune
  MNN_TUNE_JETSON_CLOCKS   Run sudo -n jetson_clocks first. Default: 1
  MNN_TUNE_EXTRA_ENV       Extra env prefix, for diagnostic policy toggles.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REMOTE="${MNN_TUNE_REMOTE:-jetson@192.168.101.192}"
REMOTE_REPO="${MNN_TUNE_REMOTE_REPO:-/home/jetson/code/kvshare-edge/impl/MNN}"
ART_REL="${MNN_TUNE_ART_REL:-.cache/output/mnn/artifacts/jetson_cross_cuda}"
CUDA_LIB="${MNN_TUNE_CUDA_LIB:-/usr/local/cuda-12.2/targets/aarch64-linux/lib}"
CASES="${MNN_TUNE_CASES:-hidden_to_inter inter_to_hidden hidden_to_gateup_concat}"
ROWS="${MNN_TUNE_ROWS:-4-5}"
WARMUP="${MNN_TUNE_WARMUP:-20}"
REPEAT="${MNN_TUNE_REPEAT:-80}"
PRECISION="${MNN_TUNE_PRECISION:-2}"
MEMORY="${MNN_TUNE_MEMORY:-2}"
LOG_DIR="${MNN_TUNE_LOG_DIR:-.cache/bench_ops/decode_repair_tune}"
JETSON_CLOCKS="${MNN_TUNE_JETSON_CLOCKS:-1}"
EXTRA_ENV="${MNN_TUNE_EXTRA_ENV:-}"

printf '[weight-only-tune] remote=%s repo=%s\n' "${REMOTE}" "${REMOTE_REPO}"
printf '[weight-only-tune] cases=%s rows=%s warmup=%s repeat=%s precision=%s memory=%s\n' \
  "${CASES}" "${ROWS}" "${WARMUP}" "${REPEAT}" "${PRECISION}" "${MEMORY}"

ssh "${REMOTE}" \
  "REMOTE_REPO='${REMOTE_REPO}' ART_REL='${ART_REL}' CUDA_LIB='${CUDA_LIB}' CASES='${CASES}' ROWS='${ROWS}' WARMUP='${WARMUP}' REPEAT='${REPEAT}' PRECISION='${PRECISION}' MEMORY='${MEMORY}' LOG_DIR='${LOG_DIR}' JETSON_CLOCKS='${JETSON_CLOCKS}' EXTRA_ENV='${EXTRA_ENV}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${REMOTE_REPO}"
if [[ "${JETSON_CLOCKS}" == "1" ]]; then
  sudo -n jetson_clocks >/dev/null 2>&1 || true
fi
ART="${ART_REL}"
RUN_TEST="${ART}/bin/run_test.out"
if [[ ! -x "${RUN_TEST}" ]]; then
  echo "run_test.out not found or not executable: ${RUN_TEST}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/weight_only_rows_${ROWS//[^A-Za-z0-9_,-]/_}_$(date +%Y%m%d_%H%M%S).log"
export LD_LIBRARY_PATH="${PWD}/${ART}/lib:${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
{
  printf 'remote_repo=%s\n' "${REMOTE_REPO}"
  printf 'artifact=%s\n' "${ART}"
  printf 'rows=%s\n' "${ROWS}"
  printf 'warmup=%s\n' "${WARMUP}"
  printf 'repeat=%s\n' "${REPEAT}"
  printf 'precision=%s\n' "${PRECISION}"
  printf 'memory=%s\n' "${MEMORY}"
  printf 'extra_env=%s\n' "${EXTRA_ENV}"
} | tee "${LOG}" >/dev/null
for CASE in ${CASES}; do
  echo "### CASE=${CASE}" | tee -a "${LOG}"
  env ${EXTRA_ENV} \
    MNN_BENCH_WEIGHT_ONLY_CASE="${CASE}" \
    MNN_BENCH_WEIGHT_ONLY_ROWS="${ROWS}" \
    MNN_BENCH_WEIGHT_ONLY_WARMUP="${WARMUP}" \
    MNN_BENCH_WEIGHT_ONLY_REPEAT="${REPEAT}" \
    "${RUN_TEST}" bench_ops/cuda/perf/WeightOnlyConv 2 "${PRECISION}" 1 x "${MEMORY}" 2>&1 | tee -a "${LOG}"
done
echo "LOG=${LOG}"
REMOTE_SCRIPT
