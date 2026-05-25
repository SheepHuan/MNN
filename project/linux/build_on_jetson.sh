#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec bash "${ROOT_DIR}/.codex/skills/mnn-build-artifacts/scripts/build_jetson_artifacts.sh" "$@"
