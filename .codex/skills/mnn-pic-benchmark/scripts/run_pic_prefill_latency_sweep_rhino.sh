#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/run_pic_prefill_latency_sweep.py" \
  --devices rhino \
  --allow-subset-devices \
  --allow-extra-device \
  --contexts "${MNN_PIC_CONTEXTS:-512,1024,1536,2048,2560}" \
  --frequency-profile "${MNN_PIC_FREQUENCY_PROFILE:-max}" \
  --remote-memory-limit-percent "${MNN_PIC_REMOTE_MEM_LIMIT:-95}" \
  --server-env "LD_PRELOAD=/usr/lib/libOpenCL_adreno.so" \
  "$@"
