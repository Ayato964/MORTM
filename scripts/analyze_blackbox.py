#!/usr/bin/env python3
"""
MORTM Blackbox Crash Analyzer
クラッシュ後に実行すると、最新のブラックボックスログの末尾（落ちた直前の数秒間）を抽出し、
何が限界値に達して落ちたのかを自動判定・レポートします。
"""

import os
import sys
import glob
import csv

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "blackbox")

def analyze_latest_log():
    csv_files = sorted(glob.glob(os.path.join(LOG_DIR, "blackbox_*.csv")))
    if not csv_files:
        print(f"[Error] No blackbox logs found in {LOG_DIR}")
        sys.exit(1)
        
    latest_file = csv_files[-1]
    print(f"\n{'='*70}")
    print(f"  MORTM Crash Blackbox Analyzer")
    print(f"  Target Log: {latest_file}")
    print(f"{'='*70}\n")
    
    rows = []
    with open(latest_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    if not rows:
        print("[Warning] Log file is empty.")
        return
        
    tail_count = min(15, len(rows))
    recent = rows[-tail_count:]
    
    print(f"--- 【クラッシュ直前 {tail_count} 秒間のテレメトリ推移】 ---")
    header_fmt = "{:<19} | {:<9} | {:<7} | {:<7} | {:<8} | {:<8} | {:<7} | {:<8}"
    row_fmt    = "{:<19} | {:<9} | {:<7} | {:<7} | {:<8} | {:<8} | {:<7} | {:<8}"
    print(header_fmt.format("Time", "Total Pwr", "GPU0 Pwr", "GPU1 Pwr", "GPU0 Tmp", "GPU1 Tmp", "CPU Tmp", "RAM Util"))
    print("-" * 90)
    
    max_pwr = 0.0
    max_g0_pwr = 0.0
    max_g1_pwr = 0.0
    max_g0_temp = 0.0
    max_g1_temp = 0.0
    max_cpu_temp = 0.0
    max_ram_pct = 0.0
    
    for r in recent:
        tot_p = float(r.get("total_gpu_power_w", 0))
        g0_p = float(r.get("gpu0_power_w", 0))
        g1_p = float(r.get("gpu1_power_w", 0))
        g0_t = float(r.get("gpu0_temp_c", 0))
        g1_t = float(r.get("gpu1_temp_c", 0))
        cpu_t = float(r.get("cpu_temp_c", 0))
        ram_p = float(r.get("ram_used_pct", 0))
        
        max_pwr = max(max_pwr, tot_p)
        max_g0_pwr = max(max_g0_pwr, g0_p)
        max_g1_pwr = max(max_g1_pwr, g1_p)
        max_g0_temp = max(max_g0_temp, g0_t)
        max_g1_temp = max(max_g1_temp, g1_t)
        max_cpu_temp = max(max_cpu_temp, cpu_t)
        max_ram_pct = max(max_ram_pct, ram_p)
        
        print(row_fmt.format(
            r["timestamp"],
            f"{tot_p:.1f} W",
            f"{g0_p:.1f} W",
            f"{g1_p:.1f} W",
            f"{g0_t:.0f} C",
            f"{g1_t:.0f} C",
            f"{cpu_t:.1f} C",
            f"{ram_p:.1f} %"
        ))
        
    last = rows[-1]
    print("-" * 90)
    print(f"★ 最終記録時刻（電源断の瞬間）: {last['timestamp']} (Uptime: {last['uptime_sec']}s)")
    print(f"   - GPU0 (5070 Ti)     : {last['gpu0_power_w']} W (上限: {last['gpu0_limit_w']} W) | 温度: {last['gpu0_temp_c']} C | VRAM: {last['gpu0_mem_used_mb']} MB")
    print(f"   - GPU1 (4080 SUPER)  : {last['gpu1_power_w']} W (上限: {last['gpu1_limit_w']} W) | 温度: {last['gpu1_temp_c']} C | VRAM: {last['gpu1_mem_used_mb']} MB")
    print(f"   - GPU合計消費電力    : {last['total_gpu_power_w']} W")
    print(f"   - CPU温度            : {last['cpu_temp_c']} C")
    print(f"   - システムRAM使用量  : {last['ram_used_gb']} GB / 62 GB ({last['ram_used_pct']} %)")
    print(f"   - スワップ使用量     : {last['swap_used_mb']} MB")
    print(f"   - GPU0 スロットル    : {last['gpu0_throttle']}")
    print(f"   - GPU1 スロットル    : {last['gpu1_throttle']}")
    
    print("\n" + "="*70)
    print("  【クラッシュ要因の自動判定】")
    print("="*70)
    
    suspicions = []
    
    # 判定1: 電力
    g0_lim = float(last.get("gpu0_limit_w", 300))
    g1_lim = float(last.get("gpu1_limit_w", 320))
    g0_p = float(last.get("gpu0_power_w", 0))
    g1_p = float(last.get("gpu1_power_w", 0))
    if g0_p >= g0_lim * 0.95 or g1_p >= g1_lim * 0.95 or float(last.get("total_gpu_power_w", 0)) >= 500:
        suspicions.append("【高確率】GPU消費電力の急峻なスパイクによる電源ユニット過電流保護(OCP)のトリップ")
        
    # 判定2: 熱
    if float(last.get("gpu0_temp_c", 0)) >= 85 or float(last.get("gpu1_temp_c", 0)) >= 85:
        suspicions.append("【高確率】GPU過熱（85℃以上）によるサーマルシャットダウン")
    elif float(last.get("cpu_temp_c", 0)) >= 90:
        suspicions.append("【高確率】CPU過熱（90℃以上）によるTHERMTRIPサーマルシャットダウン")
        
    # 判定3: メモリ枯渇
    if float(last.get("ram_used_pct", 0)) >= 95 or float(last.get("swap_used_mb", 0)) >= 4000:
        suspicions.append("【高確率】システム物理メモリ枯渇（OOM / スワップ逼迫）によるハードハング")
        
    # 判定4: スロットリング
    if "SW_POWER_CAP" in last.get("gpu0_throttle", "") or "SW_POWER_CAP" in last.get("gpu1_throttle", ""):
        suspicions.append("【注記】GPU内部で電力上限によるクロック抑制（SW Power Cap）が発動していました")
    if "HW_SLOWDOWN" in last.get("gpu0_throttle", "") or "HW_THERMAL" in last.get("gpu0_throttle", ""):
        suspicions.append("【重大】GPU0でハードウェア過熱保護（HW Thermal Slowdown）がトリップしていました")
    if "HW_SLOWDOWN" in last.get("gpu1_throttle", "") or "HW_THERMAL" in last.get("gpu1_throttle", ""):
        suspicions.append("【重大】GPU1でハードウェア過熱保護（HW Thermal Slowdown）がトリップしていました")
        
    if not suspicions:
        suspicions.append("直前のテレメトリ数値は全て安全圏内でした。電源コネクタ接触不良・瞬断、または電源内部の経年劣化（コンデンサ抜け）による電圧ドロップの可能性が高いです。")
        
    for s in suspicions:
        print(f" - {s}")
    print("="*70 + "\n")

if __name__ == "__main__":
    analyze_latest_log()
