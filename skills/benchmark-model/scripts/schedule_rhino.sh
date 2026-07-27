#!/usr/bin/env bash
# Pin CPU/GPU frequencies on Rhino Pi-X1 (Qualcomm Kalama, Android 13) to
# eliminate DVFS noise in benchmark measurements. Idempotent.
#
# Usage on device (as root):
#   schedule_rhino.sh              # lock all clusters + GPU to max freq
#   schedule_rhino.sh restore      # restore defaults (schedutil + msm-adreno-tz)
#   schedule_rhino.sh status       # print current freq / governor (no changes)
#
# Notes:
#   - CPU: 3 clusters (policy0/3/5). Kalama has no scaling_available_frequencies,
#     use cpuinfo_max_freq. Governors: walt/conservative/powersave/performance/schedutil.
#   - GPU: Adreno via kgsl-3d0. devfreq governor is read-only (SELinux), so lock
#     via min/max/default_pwrlevel = 0 (pwrlevel 0 = max freq, 680 MHz).
#   - DDR: Kalama Android exposes no writable DDR devfreq node to root; CPU+GPU
#     lock is sufficient for benchmark stability.

set -euo pipefail
ACTION="${1:-set}"

CPU_POLICIES=(/sys/devices/system/cpu/cpufreq/policy0
              /sys/devices/system/cpu/cpufreq/policy3
              /sys/devices/system/cpu/cpufreq/policy5)
KGSL=/sys/class/kgsl/kgsl-3d0

cpu_max() { cat "$1/cpuinfo_max_freq" 2>/dev/null || echo ""; }

cpu_policy() {
    local p="$1" action="$2"
    [ -d "$p" ] || return 0
    local maxf; maxf=$(cpu_max "$p")
    case "$action" in
        set)
            echo performance > "$p/scaling_governor" 2>/dev/null || true
            ;;
        restore)
            echo schedutil > "$p/scaling_governor" 2>/dev/null || true
            ;;
        status)
            local cur gov
            cur=$(cat "$p/scaling_cur_freq" 2>/dev/null || echo "?")
            gov=$(cat "$p/scaling_governor" 2>/dev/null || echo "?")
            printf '  %-12s cur=%-12s max=%-12s gov=%s\n' \
                "$(basename "$p")" "$cur" "$maxf" "$gov"
            ;;
    esac
}

gpu() {
    local action="$1"
    [ -d "$KGSL" ] || { echo "  GPU: kgsl not found"; return; }
    local np
    np=$(cat "$KGSL/num_pwrlevels" 2>/dev/null || echo 9)
    np=$((np - 1))  # last pwrlevel = lowest freq
    case "$action" in
        set)
            echo 0 > "$KGSL/min_pwrlevel" 2>/dev/null || true
            echo 0 > "$KGSL/max_pwrlevel" 2>/dev/null || true
            echo 0 > "$KGSL/default_pwrlevel" 2>/dev/null || true
            echo performance > "$KGSL/devfreq/governor" 2>/dev/null || true
            ;;
        restore)
            echo 0 > "$KGSL/min_pwrlevel" 2>/dev/null || true
            echo "$np" > "$KGSL/default_pwrlevel" 2>/dev/null || true
            echo "$np" > "$KGSL/max_pwrlevel" 2>/dev/null || true
            echo msm-adreno-tz > "$KGSL/devfreq/governor" 2>/dev/null || true
            ;;
        status)
            local pwr cur max gov
            pwr=$(cat "$KGSL/default_pwrlevel" 2>/dev/null || echo "?")
            cur=$(cat "$KGSL/gpuclk" 2>/dev/null || echo "?")
            max=$(cat "$KGSL/max_gpuclk" 2>/dev/null || echo "?")
            gov=$(cat "$KGSL/devfreq/governor" 2>/dev/null || echo "?")
            printf '  GPU(kgsl)    pwrlevel=%s gpuclk=%s max=%s devfreq_gov=%s\n' \
                "$pwr" "$cur" "$max" "$gov"
            ;;
    esac
}

main() {
    [ "$(id -u)" -eq 0 ] || { echo "must run as root" >&2; exit 1; }
    case "$ACTION" in set|restore|status) ;; *) echo "usage: $0 [set|restore|status]"; exit 1;; esac

    echo "host: $(hostname)  SoC: Kalama (Rhino Pi-X1)"
    echo "=== CPU ==="
    for p in "${CPU_POLICIES[@]}"; do cpu_policy "$p" "$ACTION"; done
    echo "=== GPU ==="
    gpu "$ACTION"
    echo "=== DDR ==="
    echo "  (no writable DDR node on Kalama Android; CPU+GPU lock sufficient)"

    # Stop thermal-engine to avoid throttling during bench (Android only)
    if command -v stop >/dev/null 2>&1; then
        case "$ACTION" in
            set)      stop thermal-engine 2>/dev/null || true ;;
            restore) start thermal-engine 2>/dev/null || true ;;
        esac
    fi
}
main "$@"
