#!/usr/bin/env bash
# Pin CPU/GPU/DDR frequencies on Orange Pi 5 Plus (Rockchip RK3588) to
# eliminate DVFS noise in benchmark measurements. Idempotent.
#
# Usage on device (as root):
#   schedule_orangepi.sh             # lock all clusters + GPU + DDR to max
#   schedule_orangepi.sh restore      # restore defaults (schedutil + simple_ondemand)
#   schedule_orangepi.sh status       # print current freq / governor (no changes)
#
# Notes:
#   - CPU: 3 clusters. policy0 (A55, 0-3, max 1.8GHz), policy4 (A76, 4-5, max 2.3GHz),
#     policy6 (A76, 6-7, max 2.3GHz). Governors include userspace/performance/schedutil.
#   - GPU: Mali via /sys/class/devfreq/fb000000.gpu, max 1GHz. Use performance governor.
#   - DDR: DMC via /sys/class/devfreq/dmc. Use performance governor.

set -euo pipefail
ACTION="${1:-set}"

CPU_POLICIES=(/sys/devices/system/cpu/cpufreq/policy0
              /sys/devices/system/cpu/cpufreq/policy4
              /sys/devices/system/cpu/cpufreq/policy6)
GPU_NODE=/sys/class/devfreq/fb000000.gpu
DMC_NODE=/sys/class/devfreq/dmc

cpu_max() { cat "$1/cpuinfo_max_freq" 2>/dev/null || echo ""; }

cpu_policy() {
    local p="$1" action="$2"
    [ -d "$p" ] || return 0
    local maxf; maxf=$(cpu_max "$p")
    case "$action" in
        set)
            # RK3588 supports userspace governor; use it to lock freq exactly
            if echo userspace 2>/dev/null > "$p/scaling_governor" 2>/dev/null; then
                [ -n "$maxf" ] && echo "$maxf" > "$p/scaling_setspeed" 2>/dev/null || true
            else
                echo performance > "$p/scaling_governor" 2>/dev/null || true
            fi
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

devfreq_node() {
    local node="$1" label="$2" action="$3"
    [ -d "$node" ] || { echo "  $label: not found"; return; }
    local maxf
    maxf=$(cat "$node/max_freq" 2>/dev/null || echo "")
    case "$action" in
        set)
            echo performance > "$node/governor" 2>/dev/null || true
            [ -n "$maxf" ] && echo "$maxf" > "$node/min_freq" 2>/dev/null || true
            ;;
        restore)
            echo simple_ondemand > "$node/governor" 2>/dev/null || true
            echo 0 > "$node/min_freq" 2>/dev/null || true
            ;;
        status)
            local cur gov
            cur=$(cat "$node/cur_freq" 2>/dev/null || echo "?")
            gov=$(cat "$node/governor" 2>/dev/null || echo "?")
            printf '  %-12s cur=%-12s max=%-12s gov=%s\n' "$label" "$cur" "$maxf" "$gov"
            ;;
    esac
}

main() {
    [ "$(id -u)" -eq 0 ] || { echo "must run as root" >&2; exit 1; }
    case "$ACTION" in set|restore|status) ;; *) echo "usage: $0 [set|restore|status]"; exit 1;; esac

    echo "host: $(hostname)  SoC: RK3588 (Orange Pi 5 Plus)"
    echo "=== CPU ==="
    for p in "${CPU_POLICIES[@]}"; do cpu_policy "$p" "$ACTION"; done
    echo "=== GPU ==="
    devfreq_node "$GPU_NODE" "GPU(Mali)" "$ACTION"
    echo "=== DDR ==="
    devfreq_node "$DMC_NODE" "DDR(DMC)" "$ACTION"
}
main "$@"
