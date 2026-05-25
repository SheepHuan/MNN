#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${MNN_ROOT_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

DEFAULT_AARCH64_TOOLCHAIN_URL="https://developer.arm.com/-/media/files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"

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
TOTAL_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
DEFAULT_JOBS="$((TOTAL_CPUS > 1 ? TOTAL_CPUS / 2 : 1))"
JOBS="${JOBS:-${DEFAULT_JOBS}}"
CLEAN="${CLEAN:-0}"
BUILD_TARGET="${BUILD_TARGET:-}"
BUILD_MNNCONVERT="${BUILD_MNNCONVERT:-1}"
INSTALL_AFTER_BUILD="${INSTALL_AFTER_BUILD:-1}"
CUDA_ARCHS="${CUDA_ARCHS:-}"
MNN_KLEIDIAI="${MNN_KLEIDIAI:-OFF}"
MNN_CUDA_PROFILE="${MNN_CUDA_PROFILE:-OFF}"
MNN_CUDA_BF16="${MNN_CUDA_BF16:-OFF}"
MNN_CUDA_QUANT="${MNN_CUDA_QUANT:-OFF}"
TOOLCHAIN_CACHE_DIR="${TOOLCHAIN_CACHE_DIR:-${ROOT_DIR}/.cache/toolchains}"
TOOLCHAIN_URL="${TOOLCHAIN_URL:-${MNN_JETSON_TOOLCHAIN_URL:-${DEFAULT_AARCH64_TOOLCHAIN_URL}}}"
TOOLCHAIN_SIG_URL="${TOOLCHAIN_SIG_URL:-${MNN_JETSON_TOOLCHAIN_SIG_URL:-}}"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-}"
CROSS_TRIPLE="${CROSS_TRIPLE:-}"
SYSROOT="${SYSROOT:-}"
ENABLE_CROSS_CUDA="${ENABLE_CROSS_CUDA:-${MNN_CROSS_CUDA:-OFF}}"
CROSS_COMPILE="${CROSS_COMPILE:-auto}"
GPG_HOME="${GPG_HOME:-${TOOLCHAIN_CACHE_DIR}/gnupg}"
GPG_KEYSERVER="${GPG_KEYSERVER:-hkps://keyserver.ubuntu.com}"
IS_NATIVE_JETSON=0
REQUESTED_BUILD_DIR="${BUILD_DIR:-}"
REQUESTED_INSTALL_PREFIX="${INSTALL_PREFIX:-}"

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

detect_is_native_jetson() {
    local model=""

    if [[ -r /proc/device-tree/model ]]; then
        model="$(tr -d '\0' </proc/device-tree/model)"
    elif [[ -r /sys/firmware/devicetree/base/model ]]; then
        model="$(tr -d '\0' </sys/firmware/devicetree/base/model)"
    fi

    if [[ "${model}" == *Jetson* ]] || [[ "${model}" == *NVIDIA* ]]; then
        return 0
    fi

    [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]] && [[ -d /usr/local/cuda ]]
}

download_url() {
    local url="$1"
    local dst="$2"

    mkdir -p "$(dirname "${dst}")"
    log "Downloading: ${url}"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 -o "${dst}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${dst}" "${url}"
    else
        die "missing curl or wget for downloading toolchain"
    fi
}

verify_checksum_signature() {
    local archive_path="$1"
    local sig_path="$2"
    local expected actual hash_len hash_tool

    expected="$(grep -Eom 1 '^[[:space:]]*([A-Fa-f0-9]{32}|[A-Fa-f0-9]{40}|[A-Fa-f0-9]{64})' "${sig_path}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]' || true)"
    [[ -n "${expected}" ]] || return 1

    hash_len="${#expected}"
    case "${hash_len}" in
        32) hash_tool="md5sum" ;;
        40) hash_tool="sha1sum" ;;
        64) hash_tool="sha256sum" ;;
        *) return 1 ;;
    esac

    require_cmd "${hash_tool}"
    actual="$("${hash_tool}" "${archive_path}" | awk '{print tolower($1)}')"
    if [[ "${actual}" != "${expected}" ]]; then
        printf 'expected %s checksum: %s\nactual %s checksum:   %s\n' "${hash_tool}" "${expected}" "${hash_tool}" "${actual}" >&2
        return 1
    fi
}

verify_openpgp_signature() {
    local archive_path="$1"
    local sig_path="$2"
    local status_file missing_key

    require_cmd gpg
    mkdir -p "${GPG_HOME}"
    chmod 700 "${GPG_HOME}"

    status_file="$(mktemp)"
    if GNUPGHOME="${GPG_HOME}" gpg --batch --status-fd 1 --verify "${sig_path}" "${archive_path}" >"${status_file}" 2>&1; then
        rm -f "${status_file}"
        return 0
    fi

    missing_key="$(awk '/^\[GNUPG:\] NO_PUBKEY / {print $3; exit}' "${status_file}")"
    if [[ -n "${missing_key}" ]]; then
        log "Importing missing Arm toolchain signing key: ${missing_key}"
        GNUPGHOME="${GPG_HOME}" gpg --batch --keyserver "${GPG_KEYSERVER}" --recv-keys "${missing_key}" >/dev/null 2>&1 || true
        if GNUPGHOME="${GPG_HOME}" gpg --batch --status-fd 1 --verify "${sig_path}" "${archive_path}" >"${status_file}" 2>&1; then
            rm -f "${status_file}"
            return 0
        fi
    fi

    cat "${status_file}" >&2
    rm -f "${status_file}"
    return 1
}

verify_toolchain_signature() {
    local archive_path="$1"
    local sig_path="$2"

    if grep -q "BEGIN PGP SIGNATURE" "${sig_path}"; then
        verify_openpgp_signature "${archive_path}" "${sig_path}"
        return $?
    fi

    verify_checksum_signature "${archive_path}" "${sig_path}"
}

ensure_verified_toolchain_archive() {
    local url="$1"
    local archive_path="$2"
    local sig_path="$3"
    local sig_url="$4"

    if [[ -f "${archive_path}" && -f "${sig_path}" ]] && verify_toolchain_signature "${archive_path}" "${sig_path}"; then
        log "Using cached verified toolchain archive: ${archive_path}"
        return 0
    fi

    rm -f "${archive_path}" "${sig_path}"
    download_url "${url}" "${archive_path}"
    download_url "${sig_url}" "${sig_path}"

    if ! verify_toolchain_signature "${archive_path}" "${sig_path}"; then
        rm -f "${archive_path}" "${sig_path}"
        die "toolchain signature verification failed"
    fi
}

find_toolchain_root() {
    local search_dir="$1"
    local gcc_path=""

    gcc_path="$(find "${search_dir}" -type f -path '*/bin/*gcc' 2>/dev/null | grep -E '/bin/aarch64.*linux.*-gcc$' | head -n 1 || true)"
    if [[ -z "${gcc_path}" ]]; then
        return 1
    fi
    dirname "$(dirname "${gcc_path}")"
}

prepare_cross_toolchain() {
    local archive_url archive_name archive_path sig_url sig_path extract_dir found_root triple_prefix libc_dir toolchain_file

    require_cmd tar

    if [[ -n "${TOOLCHAIN_ROOT}" ]]; then
        found_root="${TOOLCHAIN_ROOT}"
    else
        archive_url="${TOOLCHAIN_URL%%\?*}"
        archive_name="$(basename "${archive_url}")"
        sig_url="${TOOLCHAIN_SIG_URL:-${archive_url}.asc}"
        extract_dir="${TOOLCHAIN_CACHE_DIR}/${archive_name%.tar.*}"
        archive_path="${TOOLCHAIN_CACHE_DIR}/${archive_name}"
        sig_path="${archive_path}.asc"
        ensure_verified_toolchain_archive "${TOOLCHAIN_URL}" "${archive_path}" "${sig_path}" "${sig_url}"
        if [[ ! -d "${extract_dir}" ]]; then
            mkdir -p "${extract_dir}"
            log "Extracting toolchain to: ${extract_dir}"
            tar -xf "${archive_path}" -C "${extract_dir}" --strip-components=1
        fi
        found_root="$(find_toolchain_root "${extract_dir}" || true)"
        [[ -n "${found_root}" ]] || die "downloaded toolchain is not an AArch64 Linux toolchain. Jetson requires aarch64-*-linux-gnu, not arm-none-eabi."
    fi

    [[ -d "${found_root}/bin" ]] || die "toolchain root has no bin directory: ${found_root}"

    if [[ -z "${CROSS_TRIPLE}" ]]; then
        triple_prefix="$(find "${found_root}/bin" -maxdepth 1 -type f -name 'aarch64*linux*gcc' | head -n 1 || true)"
        [[ -n "${triple_prefix}" ]] || die "cannot find aarch64 Linux gcc under ${found_root}/bin"
        CROSS_TRIPLE="$(basename "${triple_prefix}")"
        CROSS_TRIPLE="${CROSS_TRIPLE%gcc}"
    fi

    [[ "${CROSS_TRIPLE}" == aarch64*linux*- ]] || die "CROSS_TRIPLE=${CROSS_TRIPLE} is not a Jetson Linux target triple"
    [[ -x "${found_root}/bin/${CROSS_TRIPLE}gcc" ]] || die "missing ${found_root}/bin/${CROSS_TRIPLE}gcc"
    [[ -x "${found_root}/bin/${CROSS_TRIPLE}g++" ]] || die "missing ${found_root}/bin/${CROSS_TRIPLE}g++"

    if [[ -z "${SYSROOT}" ]]; then
        libc_dir="${found_root}/${CROSS_TRIPLE%-}/libc"
        if [[ -d "${libc_dir}" ]]; then
            SYSROOT="${libc_dir}"
        fi
    fi

    toolchain_file="${TOOLCHAIN_CACHE_DIR}/jetson-aarch64.toolchain.cmake"
    mkdir -p "${TOOLCHAIN_CACHE_DIR}"
    {
        printf 'set(CMAKE_SYSTEM_NAME Linux)\n'
        printf 'set(CMAKE_SYSTEM_PROCESSOR aarch64)\n'
        printf 'set(CMAKE_C_COMPILER "%s")\n' "${found_root}/bin/${CROSS_TRIPLE}gcc"
        printf 'set(CMAKE_CXX_COMPILER "%s")\n' "${found_root}/bin/${CROSS_TRIPLE}g++"
        if [[ -n "${SYSROOT}" ]]; then
            printf 'set(CMAKE_SYSROOT "%s")\n' "${SYSROOT}"
            printf 'set(CMAKE_FIND_ROOT_PATH "%s" "%s")\n' "${SYSROOT}" "${found_root}"
        else
            printf 'set(CMAKE_FIND_ROOT_PATH "%s")\n' "${found_root}"
        fi
        printf 'set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)\n'
        printf 'set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)\n'
        printf 'set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)\n'
        printf 'set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)\n'
    } >"${toolchain_file}"

    export PATH="${found_root}/bin:${PATH}"
    CMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE:-${toolchain_file}}"
    log "Cross toolchain root: ${found_root}"
    log "Cross triple: ${CROSS_TRIPLE}"
    log "Cross sysroot: ${SYSROOT:-<toolchain default>}"
    log "CMake toolchain file: ${CMAKE_TOOLCHAIN_FILE}"
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

if detect_is_native_jetson; then
    IS_NATIVE_JETSON=1
fi

if [[ "${CROSS_COMPILE}" == "1" || "${CROSS_COMPILE}" == "ON" ]]; then
    IS_NATIVE_JETSON=0
elif [[ "${CROSS_COMPILE}" == "0" || "${CROSS_COMPILE}" == "OFF" ]]; then
    IS_NATIVE_JETSON=1
fi

if [[ "${IS_NATIVE_JETSON}" == "1" && -d /usr/local/cuda/bin ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
fi

if [[ "${IS_NATIVE_JETSON}" == "1" && -d /usr/local/cuda/lib64 ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

require_cmd cmake
require_cmd install

if [[ "${IS_NATIVE_JETSON}" == "1" ]]; then
    if [[ -n "${REQUESTED_BUILD_DIR}" ]]; then
        BUILD_DIR="${REQUESTED_BUILD_DIR}"
    elif [[ "${ARTIFACT_PLATFORM}" == "jetson" ]]; then
        BUILD_DIR="${ROOT_DIR}/.cache/build/mnn/jetson_cuda"
    else
        BUILD_DIR="${ROOT_DIR}/.cache/build/mnn/${ARTIFACT_PLATFORM}_cuda"
    fi
    INSTALL_PREFIX="${REQUESTED_INSTALL_PREFIX:-${ROOT_DIR}/.cache/output/mnn/artifacts/${ARTIFACT_PLATFORM}}"

    require_cmd nvcc

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
else
    ARTIFACT_PLATFORM="jetson"
    BUILD_DIR="${REQUESTED_BUILD_DIR:-${ROOT_DIR}/.cache/build/mnn/jetson_cross}"
    INSTALL_PREFIX="${REQUESTED_INSTALL_PREFIX:-${ROOT_DIR}/.cache/output/mnn/artifacts/jetson}"
    prepare_cross_toolchain
    if [[ "${ENABLE_CROSS_CUDA}" == "ON" || "${ENABLE_CROSS_CUDA}" == "1" ]]; then
        require_cmd nvcc
        if [[ -z "${CUDA_ARCHS}" ]]; then
            CUDA_ARCHS="7.2"
        fi
        CUDA_ARCHS="$(normalize_cuda_archs "${CUDA_ARCHS}")"
    else
        CUDA_ARCHS=""
    fi
fi

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
log "Native Jetson build: ${IS_NATIVE_JETSON}"
log "Artifact platform: ${ARTIFACT_PLATFORM}"
log "Build directory: ${BUILD_DIR}"
log "Install prefix: ${INSTALL_PREFIX}"
log "Build type: ${BUILD_TYPE}"
log "CUDA support: $([[ "${IS_NATIVE_JETSON}" == "1" || "${ENABLE_CROSS_CUDA}" == "ON" || "${ENABLE_CROSS_CUDA}" == "1" ]] && printf 'ON' || printf 'OFF')"
log "CUDA architectures: ${CUDA_ARCHS:-<disabled>}"
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

if [[ "${IS_NATIVE_JETSON}" == "1" || "${ENABLE_CROSS_CUDA}" == "ON" || "${ENABLE_CROSS_CUDA}" == "1" ]]; then
    CMAKE_CONFIGURE_ARGS+=(
        -DCUDA_ARCHS="${CUDA_ARCHS}"
        -DMNN_CUDA=ON
    )
else
    CMAKE_CONFIGURE_ARGS+=(
        -DMNN_CUDA=OFF
    )
fi

if [[ "${IS_NATIVE_JETSON}" != "1" ]]; then
    CMAKE_CONFIGURE_ARGS+=(
        -DCMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE}"
    )
fi

if [[ -n "${CMAKE_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_CMAKE_ARGS=(${CMAKE_ARGS})
    CMAKE_CONFIGURE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")
fi

log "Configuring CMake"
cmake "${CMAKE_CONFIGURE_ARGS[@]}"

if [[ "${IS_NATIVE_JETSON}" == "1" || "${ENABLE_CROSS_CUDA}" == "ON" || "${ENABLE_CROSS_CUDA}" == "1" ]]; then
    log "Building MNN with CUDA and LLM support"
else
    log "Building MNN Jetson/aarch64 artifacts with LLM support"
fi
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
