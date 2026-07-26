#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARCHIVE_DIR="${ARCHIVE_DIR:-${REPO_ROOT}/prebuilts/archives}"
SELF_PATH="${SCRIPT_DIR}/download_archives.sh"

DEFAULT_JOBS=2
JOBS="${JOBS:-${DEFAULT_JOBS}}"
FORCE_DOWNLOAD=0
LIST_ONLY=0

# Git-managed manifest:
# filename|url|checksum_algorithm|checksum
ARCHIVE_MANIFEST=(
    "android-ndk-r29-linux.zip|https://dl.google.com/android/repository/android-ndk-r29-linux.zip|sha1|87e2bb7e9be5d6a1c6cdf5ec40dd4e0c6d07c30b"
    "arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz|https://developer.arm.com/-/media/files/downloads/gnu/11.3.rel1/binrel/arm-gnu-toolchain-11.3.rel1-x86_64-aarch64-none-linux-gnu.tar.xz|sha256|50cdef6c5baddaa00f60502cc8b59cc11065306ae575ad2f51e412a9b2a90364"
)

usage() {
    cat <<'EOF'
Usage:
  tools/download_archives.sh [--jobs N] [--force] [--list]

Options:
  --jobs N    Parallel download workers. Default: 2
  --force     Re-download files even if checksum passes
  --list      Print the download manifest and exit
  -h, --help  Show this help

Behavior:
  1. Download into prebuilts/archives/
  2. Verify existing files with the embedded checksums
  3. Re-download missing or invalid files
  4. Verify every downloaded file before reporting success
EOF
}

log() {
    printf '[download_archives] %s\n' "$*"
}

require_command() {
    local command_name="$1"
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Error: required command not found: ${command_name}" >&2
        exit 1
    fi
}

hash_command() {
    case "$1" in
        sha256) printf 'sha256sum\n' ;;
        sha1) printf 'sha1sum\n' ;;
        *)
            echo "Error: unsupported checksum algorithm: $1" >&2
            exit 1
            ;;
    esac
}

calc_hash() {
    local file="$1"
    local algorithm="$2"
    "$(hash_command "${algorithm}")" "${file}" | awk '{print $1}'
}

verify_file() {
    local file="$1"
    local algorithm="$2"
    local expected="$3"
    [ -f "${file}" ] && [ "$(calc_hash "${file}" "${algorithm}")" = "${expected}" ]
}

download_file() {
    local url="$1"
    local destination="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --retry-delay 2 -C - -o "${destination}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget --continue --tries=3 --output-document="${destination}" "${url}"
    else
        echo "Error: neither curl nor wget is available" >&2
        exit 1
    fi
}

download_one() {
    local name="$1"
    local url="$2"
    local algorithm="$3"
    local expected="$4"
    local destination="${ARCHIVE_DIR}/${name}"
    local temporary_file="${destination}.part"

    if [ "${FORCE_DOWNLOAD}" -eq 0 ] && verify_file "${destination}" "${algorithm}" "${expected}"; then
        log "skip ${name} (${algorithm} ok)"
        return 0
    fi

    if [ -f "${destination}" ] && [ "${FORCE_DOWNLOAD}" -eq 0 ]; then
        log "checksum mismatch, re-download ${name}"
        rm -f "${destination}"
    fi

    log "start ${name}"
    mkdir -p "$(dirname "${destination}")"
    download_file "${url}" "${temporary_file}"

    if ! verify_file "${temporary_file}" "${algorithm}" "${expected}"; then
        echo "Error: checksum verify failed after download: ${name}" >&2
        echo "Expected ${algorithm}: ${expected}" >&2
        echo "Actual   ${algorithm}: $(calc_hash "${temporary_file}" "${algorithm}")" >&2
        exit 1
    fi

    mv -f "${temporary_file}" "${destination}"
    log "done ${name}"
}

print_manifest() {
    local entry
    for entry in "${ARCHIVE_MANIFEST[@]}"; do
        printf '%s\n' "${entry}"
    done
}

verify_all() {
    local entry name url algorithm expected
    for entry in "${ARCHIVE_MANIFEST[@]}"; do
        IFS='|' read -r name url algorithm expected <<< "${entry}"
        if ! verify_file "${ARCHIVE_DIR}/${name}" "${algorithm}" "${expected}"; then
            echo "Error: final verification failed: ${name}" >&2
            return 1
        fi
    done
}

run_parallel_downloads() {
    local -a selected_entries=()
    local entry

    mkdir -p "${ARCHIVE_DIR}"
    export ARCHIVE_DIR FORCE_DOWNLOAD

    for entry in "${ARCHIVE_MANIFEST[@]}"; do
        selected_entries+=("${entry}")
    done

    printf '%s\0' "${selected_entries[@]}" | \
        xargs -0 -n 1 -P "${JOBS}" "${SELF_PATH}" __download_one
}

if [ "${1:-}" = "__download_one" ]; then
    shift
    entry="$1"
    IFS='|' read -r name url algorithm expected <<< "${entry}"
    download_one "${name}" "${url}" "${algorithm}" "${expected}"
    exit 0
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --jobs)
            if [ $# -lt 2 ]; then
                echo "Error: --jobs requires a value" >&2
                exit 1
            fi
            JOBS="$2"
            shift 2
            ;;
        --force)
            FORCE_DOWNLOAD=1
            shift
            ;;
        --list)
            LIST_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --jobs must be a positive integer" >&2
    exit 1
fi

if [ "${LIST_ONLY}" -eq 1 ]; then
    print_manifest
    exit 0
fi

require_command xargs
require_command sha1sum
require_command sha256sum
run_parallel_downloads
verify_all
log "all archives are ready under ${ARCHIVE_DIR}"
