#!/usr/bin/env bash

set -uo pipefail

DRY_RUN="${MNN_PERF_DRY_RUN:-0}"
SHOW_ONLY="0"
ALLOW_SUDO_PROMPT="${MNN_PERF_ALLOW_SUDO_PROMPT:-0}"
JETSON_NVP_MODEL="${JETSON_NVP_MODEL:-0}"
MNN_JETSON_FORCE_MAX_FAN="${MNN_JETSON_FORCE_MAX_FAN:-1}"
SUDO_CMD=()

log() {
    printf '[mnn-max-perf][jetson] %s\n' "$*"
}

warn() {
    printf '[mnn-max-perf][jetson] WARN: %s\n' "$*" >&2
}

usage() {
    cat <<'EOF'
Usage:
  bash set_jetson_max_perf.sh [--dry-run] [--show-only] [--allow-sudo-prompt]

Set Jetson CPU/GPU/EMC clocks to a max-performance state for MNN benchmarks.
Run on the Jetson device. Root or passwordless sudo is required for writes.
By default the script also forces the Jetson fan to max PWM and leaves it there
for benchmark stability.

Environment:
  JETSON_NVP_MODEL=0              nvpmodel mode to request. Default: 0.
  MNN_JETSON_FORCE_MAX_FAN=1      stop nvfancontrol and set pwmfan to 255. Default: 1.
  MNN_PERF_DRY_RUN=1              print writes without applying.
  MNN_PERF_ALLOW_SUDO_PROMPT=1    allow interactive sudo password prompt.
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

set_jetson_clocks() {
    if command -v nvpmodel >/dev/null 2>&1; then
        run_root nvpmodel -m "${JETSON_NVP_MODEL}" || warn "nvpmodel -m ${JETSON_NVP_MODEL} failed"
    else
        warn "nvpmodel not found"
    fi
    if command -v jetson_clocks >/dev/null 2>&1; then
        run_root jetson_clocks || warn "jetson_clocks failed"
    else
        warn "jetson_clocks not found"
    fi
}

find_hwmon_by_name() {
    local target="$1"
    local dir name
    for dir in /sys/class/hwmon/hwmon*; do
        [[ -r "${dir}/name" ]] || continue
        name="$(cat "${dir}/name" 2>/dev/null || true)"
        [[ "${name}" == "${target}" ]] && printf '%s\n' "${dir}" && return 0
    done
    return 1
}

set_fan_max() {
    local fan_dir pwm_path tach_dir rpm_path
    [[ "${MNN_JETSON_FORCE_MAX_FAN}" == "1" ]] || return 0

    if command -v systemctl >/dev/null 2>&1; then
        run_root systemctl stop nvfancontrol.service || warn "failed to stop nvfancontrol.service"
    fi

    fan_dir="$(find_hwmon_by_name pwmfan || true)"
    pwm_path="${fan_dir}/pwm1"
    if [[ -n "${fan_dir}" && -e "${pwm_path}" ]]; then
        write_sysfs "${pwm_path}" 255
    else
        warn "pwmfan/pwm1 not found; cannot force fan to max"
    fi

    sleep 2
    tach_dir="$(find_hwmon_by_name pwm_tach || true)"
    rpm_path="${tach_dir}/rpm"
    if [[ -n "${tach_dir}" && -r "${rpm_path}" ]]; then
        log "fan rpm=$(cat "${rpm_path}" 2>/dev/null || true)"
    fi
}

set_emc_debug_clk() {
    local dir="/sys/kernel/debug/bpmp/debug/clk/emc"
    local max
    [[ -d "${dir}" ]] || return 0
    max="$(cat "${dir}/max_rate" 2>/dev/null || true)"
    [[ -n "${max}" ]] || return 0
    write_sysfs "${dir}/mrq_rate_locked" 1
    write_sysfs "${dir}/rate" "${max}"
}

show_status() {
    log "hostname=$(hostname 2>/dev/null || true)"
    command -v nvpmodel >/dev/null 2>&1 && nvpmodel -q 2>/dev/null | sed 's/^/[mnn-max-perf][jetson] /' || true
    command -v jetson_clocks >/dev/null 2>&1 && jetson_clocks --show 2>/dev/null | sed 's/^/[mnn-max-perf][jetson] /' || true
    for gov in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
        [[ -e "${gov}" ]] || continue
        dir="$(dirname "${gov}")"
        log "$(basename "$(dirname "${dir}")") governor=$(cat "${gov}" 2>/dev/null) cur=$(cat "${dir}/scaling_cur_freq" 2>/dev/null || true) min=$(cat "${dir}/scaling_min_freq" 2>/dev/null || true) max=$(cat "${dir}/scaling_max_freq" 2>/dev/null || true)"
    done
    for dir in /sys/class/devfreq/*; do
        [[ -d "${dir}" ]] || continue
        case "$(basename "${dir}")" in
            *gv11b*|*gpu*|*vic*|*nvenc*|*nvdec*)
                log "$(basename "${dir}") governor=$(cat "${dir}/governor" 2>/dev/null || true) cur=$(cat "${dir}/cur_freq" 2>/dev/null || true) min=$(cat "${dir}/min_freq" 2>/dev/null || true) max=$(cat "${dir}/max_freq" 2>/dev/null || true)"
                ;;
        esac
    done
    for dir in /sys/class/hwmon/hwmon*; do
        [[ -r "${dir}/name" ]] || continue
        case "$(cat "${dir}/name" 2>/dev/null || true)" in
            pwmfan)
                log "fan pwm=$(cat "${dir}/pwm1" 2>/dev/null || true)"
                ;;
            pwm_tach)
                log "fan rpm=$(cat "${dir}/rpm" 2>/dev/null || true)"
                ;;
        esac
    done
}

if [[ "${SHOW_ONLY}" != "1" ]]; then
    [[ "${DRY_RUN}" == "1" ]] || init_sudo
    set_jetson_clocks
    set_cpu_max
    for dir in /sys/class/devfreq/*; do
        [[ -d "${dir}" ]] || continue
        case "$(basename "${dir}")" in
            *gv11b*|*gpu*|*vic*|*nvenc*|*nvdec*) set_devfreq_max "${dir}" ;;
        esac
    done
    set_emc_debug_clk
    set_fan_max
fi
show_status
