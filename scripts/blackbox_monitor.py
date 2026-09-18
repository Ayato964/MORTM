#!/usr/bin/env python3
"""
MORTM Blackbox Flight Recorder (Crash Diagnostic Logger)
1秒ごとに GPU、CPU、RAM、電力、温度、スロットリング警告を記録し、
毎秒 fsync() で SSD に強制フラッシュするため、急な瞬断でも直前1秒のデータが確実に残ります。
"""

import os
import sys
import time
import subprocess
import glob
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "blackbox")
os.makedirs(LOG_DIR, exist_ok=True)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"blackbox_{timestamp_str}.csv")

CSV_HEADER = [
    "timestamp",
    "uptime_sec",
    "gpu0_power_w", "gpu0_limit_w", "gpu0_temp_c", "gpu0_util_pct", "gpu0_mem_used_mb", "gpu0_sm_clock_mhz", "gpu0_throttle",
    "gpu1_power_w", "gpu1_limit_w", "gpu1_temp_c", "gpu1_util_pct", "gpu1_mem_used_mb", "gpu1_sm_clock_mhz", "gpu1_throttle",
    "total_gpu_power_w",
    "cpu_temp_c",
    "ram_used_gb", "ram_avail_gb", "ram_used_pct",
    "swap_used_mb",
]

def get_cpu_temp():
    temps = []
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(path, "r") as f:
                val = float(f.read().strip()) / 1000.0
                temps.append(val)
        except Exception:
            pass
    return max(temps) if temps else -1.0

def get_ram_info():
    mem_total, mem_avail, swap_total, swap_free = 0, 0, 0, 0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                key = parts[0].strip()
                val = int(parts[1].strip().split()[0]) # in kB
                if key == "MemTotal":
                    mem_total = val
                elif key == "MemAvailable":
                    mem_avail = val
                elif key == "SwapTotal":
                    swap_total = val
                elif key == "SwapFree":
                    swap_free = val
    except Exception:
        pass
    
    used_gb = (mem_total - mem_avail) / (1024 * 1024)
    avail_gb = mem_avail / (1024 * 1024)
    pct = ((mem_total - mem_avail) / mem_total * 100) if mem_total else 0
    swap_used_mb = (swap_total - swap_free) / 1024
    return used_gb, avail_gb, pct, swap_used_mb

def query_nvsmi():
    query = (
        "index,power.draw,power.limit,temperature.gpu,utilization.gpu,"
        "memory.used,clocks.current.graphics,clocks_event_reasons.active"
    )
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=1.5)
        if res.returncode != 0:
            return None
        lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
        gpus = {}
        for l in lines:
            parts = [p.strip() for p in l.split(",")]
            if len(parts) >= 8:
                idx = int(parts[0])
                gpus[idx] = {
                    "power": float(parts[1]),
                    "limit": float(parts[2]),
                    "temp": float(parts[3]),
                    "util": float(parts[4]),
                    "mem": float(parts[5]),
                    "clock": float(parts[6]),
                    "throttle": parts[7]
                }
        return gpus
    except Exception:
        return None

def main():
    print(f"[Blackbox Recorder] Logging started: {log_file}")
    print(f"[Blackbox Recorder] Sampling interval: 1.0s (Direct fsync enabled)")
    start_time = time.time()
    
    with open(log_file, "w", encoding="utf-8", buffering=1) as f:
        f.write(",".join(CSV_HEADER) + "\n")
        f.flush()
        os.fsync(f.fileno())
        
        while True:
            t0 = time.time()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            uptime_sec = int(t0 - start_time)
            
            gpus = query_nvsmi()
            cpu_temp = get_cpu_temp()
            ram_used, ram_avail, ram_pct, swap_used = get_ram_info()
            
            g0 = gpus.get(0, {"power": 0, "limit": 0, "temp": 0, "util": 0, "mem": 0, "clock": 0, "throttle": "N/A"}) if gpus else {}
            g1 = gpus.get(1, {"power": 0, "limit": 0, "temp": 0, "util": 0, "mem": 0, "clock": 0, "throttle": "N/A"}) if gpus else {}
            
            total_power = g0.get("power", 0) + g1.get("power", 0)
            
            row = [
                now_str,
                str(uptime_sec),
                f"{g0.get('power', 0):.1f}", f"{g0.get('limit', 0):.1f}", f"{g0.get('temp', 0):.0f}",
                f"{g0.get('util', 0):.0f}", f"{g0.get('mem', 0):.0f}", f"{g0.get('clock', 0):.0f}", f'"{g0.get("throttle", "")}"',
                f"{g1.get('power', 0):.1f}", f"{g1.get('limit', 0):.1f}", f"{g1.get('temp', 0):.0f}",
                f"{g1.get('util', 0):.0f}", f"{g1.get('mem', 0):.0f}", f"{g1.get('clock', 0):.0f}", f'"{g1.get("throttle", "")}"',
                f"{total_power:.1f}",
                f"{cpu_temp:.1f}",
                f"{ram_used:.2f}", f"{ram_avail:.2f}", f"{ram_pct:.1f}",
                f"{swap_used:.0f}",
            ]
            
            f.write(",".join(row) + "\n")
            f.flush()
            os.fsync(f.fileno())  # 強制的にSSDへ物理コミット
            
            elapsed = time.time() - t0
            sleep_time = max(0.1, 1.0 - elapsed)
            time.sleep(sleep_time)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Blackbox Recorder] Stopped by user.")
