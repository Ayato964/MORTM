#!/usr/bin/env bash
# ハードクラッシュ緩和設定。再起動のたびに既定値へ戻るため systemd から毎回適用する。
#
# 背景:
#   - CPU: i5-14500 (Raptor Lake 14th gen)。RAPL PL1/PL2 が 4095W = 無制限で動いていた。
#     Intel 仕様は PL1=65W / PL2=154W。無制限運用は Vmin shift 劣化の促進要因として
#     Intel が 2024 年に名指ししたもの。BIOS 1501 (2023-10-06) は該当ガイダンス以前の版。
#   - ブースト上限を 5.0GHz から下げ、劣化した silicon の不安定点を回避する。
#   - GPU: 従来どおり 260W に制限。
set -u

CPU_PERF_PCT="${CPU_PERF_PCT:-88}"      # 100=5.0GHz, 88≒4.4GHz
PL1_UW="${PL1_UW:-65000000}"            # 65W
PL2_UW="${PL2_UW:-154000000}"           # 154W
GPU_PL="${GPU_PL:-260}"                 # W

log() { echo "[hw_mitigation] $*"; }

# --- CPU: ブースト上限 ---
if [ -w /sys/devices/system/cpu/intel_pstate/max_perf_pct ]; then
    echo "$CPU_PERF_PCT" > /sys/devices/system/cpu/intel_pstate/max_perf_pct
    log "max_perf_pct = $(cat /sys/devices/system/cpu/intel_pstate/max_perf_pct)%"
else
    log "WARN: max_perf_pct に書き込めない"
fi

# --- CPU: RAPL 電力上限 ---
RAPL=/sys/class/powercap/intel-rapl:0
for pair in "constraint_0_power_limit_uw:$PL1_UW:PL1" "constraint_1_power_limit_uw:$PL2_UW:PL2"; do
    f="${pair%%:*}"; rest="${pair#*:}"; val="${rest%%:*}"; name="${rest##*:}"
    if [ -f "$RAPL/$f" ]; then
        if echo "$val" > "$RAPL/$f" 2>/dev/null; then
            log "$name = $(( $(cat $RAPL/$f) / 1000000 ))W"
        else
            log "WARN: $name の書き込み失敗 (BIOSでロックされている可能性)"
        fi
    fi
done

# --- GPU: 電力上限 ---
if command -v nvidia-smi >/dev/null 2>&1; then
    for i in 0 1; do
        if nvidia-smi -i "$i" -pl "$GPU_PL" >/dev/null 2>&1; then
            log "GPU$i power limit = ${GPU_PL}W"
        else
            log "WARN: GPU$i の電力上限設定に失敗 (下限値未満の指定か、GPU不在)"
        fi
    done
fi

log "適用完了"
