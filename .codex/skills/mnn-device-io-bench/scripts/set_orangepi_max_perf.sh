#!/usr/bin/env bash

set -uo pipefail

DRY_RUN="${MNN_PERF_DRY_RUN:-0}"
SHOW_ONLY="0"
ALLOW_SUDO_PROMPT="${MNN_PERF_ALLOW_SUDO_PROMPT:-0}"
SUDO_CMD=()

log() {
    printf '[mnn-max-perf][orangepi] %s\n' "$*"
}

warn() {
    printf '[mnn-max-perf][orangepi] WARN: %s\n' "$*" >&2
}

usage() {
    cat <<'EOF'
Usage:
  bash set_orangepi_max_perf.sh [--dry-run] [--show-only] [--allow-sudo-prompt]

Set Orange Pi 5 Plus CPU, Mali GPU, and RK3588 DMC/DDR devfreq to maximum
performance for MNN OpenCL benchmarks. Run on the Orange Pi device.
Root or passwordless sudo is required for writes.
EOF
}

while (($#)); do
    case "$1" in
        --dry-run)
            DRY_RUN="1"; shift ;;
        --show-only)
            SHOW_ONLY="1"; shift ;;
        --allow-sudo-prompt)
            ALLOW_SUDO_PROMPT="1"; shift ;;
        --help|-h)
            usage; exit 0 ;;
        *)
            warn "unknown argument: $1"; usage; exit 2 ;;
    esac
done

init_sudo() {
    if [[ "${EUID}" -eq 0 ]]; then
        return 0
    fi
    if ! command -v sudo >/dev/null 2>&1; then
        warn "not root and sudo is unavailable; writes will be skipped"
        return 0
    fi
    if [[ "${ALLOW_SUDO_PROMPT}" == "1" ]]; then
        SUDO_CMD=(sudo)
    elif sudo -n true >/dev/null 2>&1; then
        SUDO_CMD=(sudo -n)
    else
        warn "not root and passwordless sudo is unavailable; pass --allow-sudo-prompt or run as root"
    fi
}

run_root() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "dry-run root command: $*"
        return 0
    fi
    if [[ "${EUID}" -eq 0 ]]; then
        "$@"
    elif ((${#SUDO_CMD[@]})); then
        "${SUDO_CMD[@]}" "$@"
    else
        warn "need root for: $*"
        return 1
    fi
}

write_sysfs() {
    local path="$1"
    local value="$2"
    [[ -e "${path}" ]] || return 0
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "dry-run write ${path}=${value}"
        return 0
    fi
    if [[ -w "${path}" ]]; then
        if printf '%s\n' "${value}" >"${path}" 2>/dev/null; then
            log "set ${path}=${value}"
            return 0
        fi
    fi
    if run_root sh -c 'printf "%s\n" "$1" > "$2"' _ "${value}" "${path}"; then
        log "set ${path}=${value}"
    else
        warn "failed to set ${path}=${value}"
    fi
}

max_number_from_file() {
    local path="$1"
    [[ -r "${path}" ]] || return 0
    tr '[:space:]' '\n' <"${path}" | awk '/^[0-9]+$/ { if ($1 > max) max = $1 } END { if (max != "") print max }'
}

set_cpu_max() {
    local seen=" "
    local gov dir real_dir max
    for gov in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor \
               /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
        [[ -e "${gov}" ]] || continue
        dir="$(dirname "${gov}")"
        real_dir="$(readlink -f "${dir}" 2>/dev/null || printf '%s\n' "${dir}")"
        [[ "${seen}" == *" ${real_dir} "* ]] && continue
        seen="${seen}${real_dir} "
        max="$(cat "${dir}/cpuinfo_max_freq" 2>/dev/null || cat "${dir}/scaling_max_freq" 2>/dev/null || true)"
        write_sysfs "${gov}" performance
        [[ -n "${max}" ]] && write_sysfs "${dir}/scaling_max_freq" "${max}"
        [[ -n "${max}" ]] && write_sysfs "${dir}/scaling_min_freq" "${max}"
    done
}

set_devfreq_max() {
    local dir="$1"
    local max governors
    [[ -d "${dir}" ]] || return 0
    governors="$(cat "${dir}/available_governors" 2>/dev/null || true)"
    if [[ "${governors}" == *performance* ]]; then
        write_sysfs "${dir}/governor" performance
    elif [[ "${governors}" == *userspace* ]]; then
        write_sysfs "${dir}/governor" userspace
    fi
    max="$(max_number_from_file "${dir}/available_frequencies")"
    [[ -n "${max}" ]] || max="$(cat "${dir}/max_freq" 2>/dev/null || true)"
    if [[ -n "${max}" ]]; then
        write_sysfs "${dir}/max_freq" "${max}"
        write_sysfs "${dir}/min_freq" "${max}"
        write_sysfs "${dir}/userspace/set_freq" "${max}"
    fi
}

show_status() {
    log "hostname=$(hostname 2>/dev/null || true)"
    local seen=" "
    local dir
    for gov in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
        [[ -e "${gov}" ]] || continue
        dir="$(dirname "${gov}")"
        log "$(basename "$(dirname "${dir}")") governor=$(cat "${gov}" 2>/dev/null) cur=$(cat "${dir}/scaling_cur_freq" 2>/dev/null || true) min=$(cat "${dir}/scaling_min_freq" 2>/dev/null || true) max=$(cat "${dir}/scaling_max_freq" 2>/dev/null || true)"
    done
    for dir in /sys/class/devfreq/fb000000.gpu /sys/class/devfreq/dmc /sys/class/devfreq/*gpu* /sys/class/devfreq/*dmc* /sys/class/devfreq/*ddr* /sys/class/devfreq/*dram*; do
        [[ -d "${dir}" ]] || continue
        [[ "${seen}" == *" ${dir} "* ]] && continue
        seen="${seen}${dir} "
        log "$(basename "${dir}") name=$(cat "${dir}/name" 2>/dev/null || true) governor=$(cat "${dir}/governor" 2>/dev/null || true) cur=$(cat "${dir}/cur_freq" 2>/dev/null || true) min=$(cat "${dir}/min_freq" 2>/dev/null || true) max=$(cat "${dir}/max_freq" 2>/dev/null || true)"
    done
}

if [[ "${SHOW_ONLY}" != "1" ]]; then
    seen_devfreq=" "
    [[ "${DRY_RUN}" == "1" ]] || init_sudo
    set_cpu_max
    for dir in /sys/class/devfreq/fb000000.gpu /sys/class/devfreq/dmc /sys/class/devfreq/*gpu* /sys/class/devfreq/*dmc* /sys/class/devfreq/*ddr* /sys/class/devfreq/*dram*; do
        [[ -d "${dir}" ]] || continue
        real_dir="$(readlink -f "${dir}" 2>/dev/null || printf '%s\n' "${dir}")"
        [[ "${seen_devfreq}" == *" ${real_dir} "* ]] && continue
        seen_devfreq="${seen_devfreq}${real_dir} "
        set_devfreq_max "${dir}"
    done
fi
show_status
