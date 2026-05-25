#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/.cache/build/mnn/${ARTIFACT_PLATFORM}_cuda}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${ROOT_DIR}/.cache/output/mnn/artifacts/${ARTIFACT_PLATFORM}}"
JOBS="${JOBS:-$(nproc)}"
CLEAN="${CLEAN:-0}"
BUILD_TARGET="${BUILD_TARGET:-}"
BUILD_MNNCONVERT="${BUILD_MNNCONVERT:-1}"
INSTALL_AFTER_BUILD="${INSTALL_AFTER_BUILD:-1}"
CUDA_ARCHS="${CUDA_ARCHS:-}"
MNN_KLEIDIAI="${MNN_KLEIDIAI:-OFF}"
MNN_CUDA_PROFILE="${MNN_CUDA_PROFILE:-OFF}"
MNN_CUDA_BF16="${MNN_CUDA_BF16:-OFF}"
MNN_CUDA_QUANT="${MNN_CUDA_QUANT:-OFF}"

log() {
    printf '[build_on_jetson] %s\n' "$*"
}

die() {
    printf '[build_on_jetson] ERROR: %s\n' "$*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

install_if_exists() {
    local src="$1"
    local dst_dir="$2"

    if [[ ! -f "${src}" ]]; then
        return 0
    fi
    mkdir -p "${dst_dir}"
    install -m 755 "${src}" "${dst_dir}/$(basename "${src}")"
    log "Installed $(basename "${src}") -> ${dst_dir}"
}

install_runtime_artifacts() {
    local lib_dir="${INSTALL_PREFIX}/lib"
    local bin_dir="${INSTALL_PREFIX}/bin"

    install_if_exists "${BUILD_DIR}/libMNN.so" "${lib_dir}"
    install_if_exists "${BUILD_DIR}/express/libMNN_Express.so" "${lib_dir}"
    install_if_exists "${BUILD_DIR}/source/backend/cuda/libMNN_Cuda_Main.so" "${lib_dir}"
    install_if_exists "${BUILD_DIR}/libllm.so" "${lib_dir}"
    install_if_exists "${BUILD_DIR}/libpic_llm.so" "${lib_dir}"
    install_if_exists "${BUILD_DIR}/tools/converter/libMNNConvertDeps.so" "${lib_dir}"

    install_if_exists "${BUILD_DIR}/llm_demo" "${bin_dir}"
    install_if_exists "${BUILD_DIR}/pic_llm_demo" "${bin_dir}"
    install_if_exists "${BUILD_DIR}/llm_bench" "${bin_dir}"
    install_if_exists "${BUILD_DIR}/pic_llm_bench" "${bin_dir}"
    install_if_exists "${BUILD_DIR}/mls" "${bin_dir}"
    install_if_exists "${BUILD_DIR}/run_test.out" "${bin_dir}"
    install_if_exists "${BUILD_DIR}/MNNConvert" "${bin_dir}"
}

detect_jetson_arch_from_model() {
    local model=""

    if [[ -r /proc/device-tree/model ]]; then
        model="$(tr -d '\0' </proc/device-tree/model)"
    elif [[ -r /sys/firmware/devicetree/base/model ]]; then
        model="$(tr -d '\0' </sys/firmware/devicetree/base/model)"
    fi

    case "${model}" in
        *Orin*|*orin*)
            printf '87'
            ;;
        *Xavier*|*xavier*)
            printf '72'
            ;;
        *TX2*|*tx2*)
            printf '62'
            ;;
        *)
            return 1
            ;;
    esac
}

detect_cuda_arch() {
    local capability=""

    if command -v nvidia-smi >/dev/null 2>&1; then
        capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:].' || true)"
        if [[ "${capability}" =~ ^[0-9]+$ ]]; then
            printf '%s' "${capability}"
            return 0
        fi
    fi

    if command -v deviceQuery >/dev/null 2>&1; then
        capability="$(deviceQuery 2>/dev/null | awk -F: '/CUDA Capability Major\/Minor version number/ {gsub(/[ .]/, "", $2); print $2; exit}' || true)"
        if [[ "${capability}" =~ ^[0-9]+$ ]]; then
            printf '%s' "${capability}"
            return 0
        fi
    fi

    detect_jetson_arch_from_model
}

normalize_cuda_archs() {
    local raw="$1"
    local normalized=()
    local item=""

    raw="${raw//,/ }"
    for item in ${raw}; do
        case "${item}" in
            Auto|Common|All)
                normalized+=("${item}")
                ;;
            sm_[0-9][0-9])
                normalized+=("${item:3:1}.${item:4:1}")
                ;;
            compute_[0-9][0-9])
                normalized+=("${item:8:1}.${item:9:1}")
                ;;
            [0-9].[0-9])
                normalized+=("${item}")
                ;;
            [0-9][0-9])
                normalized+=("${item:0:1}.${item:1:1}")
                ;;
            *)
                die "unsupported CUDA_ARCHS entry: ${item}. Use values like 72, 7.2, sm_72, Auto, Common or All."
                ;;
        esac
    done

    printf '%s' "${normalized[*]}"
}

if [[ -d /usr/local/cuda/bin ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
fi

if [[ -d /usr/local/cuda/lib64 ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

require_cmd cmake
require_cmd nvcc
require_cmd install

if [[ -z "${CUDA_ARCHS}" ]]; then
    CUDA_ARCHS="$(detect_cuda_arch || true)"
fi

if [[ -z "${CUDA_ARCHS}" ]]; then
    CUDA_ARCHS="Auto"
    log "CUDA_ARCHS not detected; falling back to CMake CUDA_ARCHS=Auto"
fi

CUDA_ARCHS="$(normalize_cuda_archs "${CUDA_ARCHS}")"

case "${CUDA_ARCHS}" in
    5.3)
        die "Jetson Nano/TX1 compute capability 5.3 is not supported by this MNN CUDA backend; supported single arch values include 60, 61, 62, 70, 72, 75, 80, 86, 87, 89."
        ;;
esac

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
    GENERATOR_ARGS=(-G Ninja)
fi

if [[ "${CLEAN}" == "1" ]]; then
    log "Removing build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

log "Root: ${ROOT_DIR}"
log "Artifact platform: ${ARTIFACT_PLATFORM}"
log "Build directory: ${BUILD_DIR}"
log "Install prefix: ${INSTALL_PREFIX}"
log "Build type: ${BUILD_TYPE}"
log "CUDA architectures: ${CUDA_ARCHS}"
log "KleidiAI: ${MNN_KLEIDIAI}"
log "Parallel jobs: ${JOBS}"
log "Build MNNConvert: ${BUILD_MNNCONVERT}"
log "Install after build: ${INSTALL_AFTER_BUILD}"

CMAKE_CONFIGURE_ARGS=(
    -S "${ROOT_DIR}"
    -B "${BUILD_DIR}"
    "${GENERATOR_ARGS[@]}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
    -DCUDA_ARCHS="${CUDA_ARCHS}"
    -DMNN_CUDA=ON
    -DMNN_BUILD_LLM=ON
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
    -DMNN_LOW_MEMORY=ON
    -DMNN_BUILD_SHARED_LIBS=ON
    -DMNN_SEP_BUILD=ON
    -DMNN_BUILD_TOOLS=ON
    -DMNN_BUILD_CONVERTER=ON
    -DMNN_LLM_BUILD_DEMO=ON
    -DMNN_CUDA_PROFILE="${MNN_CUDA_PROFILE}"
    -DMNN_CUDA_BF16="${MNN_CUDA_BF16}"
    -DMNN_CUDA_QUANT="${MNN_CUDA_QUANT}"
    -DMNN_KLEIDIAI="${MNN_KLEIDIAI}"
)

if [[ -n "${CMAKE_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_CMAKE_ARGS=(${CMAKE_ARGS})
    CMAKE_CONFIGURE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")
fi

log "Configuring CMake"
cmake "${CMAKE_CONFIGURE_ARGS[@]}"

log "Building MNN with CUDA and LLM support"
if [[ -n "${BUILD_TARGET}" ]]; then
    cmake --build "${BUILD_DIR}" --target "${BUILD_TARGET}" --parallel "${JOBS}"
else
    cmake --build "${BUILD_DIR}" --parallel "${JOBS}"
fi

if [[ "${BUILD_MNNCONVERT}" == "1" && "${BUILD_TARGET}" != "MNNConvert" ]]; then
    log "Building MNNConvert"
    cmake --build "${BUILD_DIR}" --target MNNConvert --parallel "${JOBS}"
fi

if [[ "${INSTALL_AFTER_BUILD}" == "1" ]]; then
    if [[ -n "${BUILD_TARGET}" ]]; then
        log "Skipping cmake --install because BUILD_TARGET=${BUILD_TARGET}; syncing available runtime artifacts"
        install_runtime_artifacts
    else
        log "Installing to: ${INSTALL_PREFIX}"
        cmake --install "${BUILD_DIR}"
        install_runtime_artifacts
    fi
else
    log "Skipping install because INSTALL_AFTER_BUILD=${INSTALL_AFTER_BUILD}"
fi

log "Done"
log "Main outputs are under: ${BUILD_DIR}"
log "Installed outputs are under: ${INSTALL_PREFIX}"
