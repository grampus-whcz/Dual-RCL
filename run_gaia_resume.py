#!/usr/bin/env python3
"""Resume GAIA 0704 from where it stopped, appending to new log files."""
import os, sys, re, subprocess, pickle, datetime
from pathlib import Path

PYTHON = "/root/shared-nvme/.conda/envs/LocaleXpert_env/bin/python"
PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GT_PKL = PROJ / "Datasets/GAIA/fault_injection_tracerank/fault_injection_list_2021-07-04.pkl"

METHOD_ARGS = {
    "localexpert": ["--skip-multivariate"],
    "dualchannel": ["--anomaly-method", "tranad", "--anomaly-epochs", "3"],
    "tvdig": ["--anomaly-method", "tranad", "--anomaly-epochs", "3",
              "--rca-method", "tvdig", "--tvdig-model", "./tvdig_checkpoint"],
    "hybrid": ["--anomaly-method", "tranad", "--anomaly-epochs", "3",
               "--rca-method", "hybrid", "--tvdig-model", "./tvdig_checkpoint"],
}


def get_completed_times(old_log):
    """Extract completed event times from old log."""
    if not os.path.exists(old_log):
        return set()
    times = set()
    with open(old_log, 'r', errors='replace') as f:
        for line in f:
            m = re.search(r'\[OK\].*?_0704_(\d{4})', line)
            if m:
                times.add(m.group(1))
    return times


def load_events():
    with open(GT_PKL, "rb") as f:
        data = pickle.load(f)
    seen = set(); events = []
    for fi in data:
        if not isinstance(fi, dict) or "time" not in fi or "service" not in fi:
            continue
        t = fi["time"].strftime("%H:%M") if hasattr(fi["time"], "strftime") else str(fi["time"])
        if t in seen: continue
        seen.add(t)
        events.append({"time": t, "service": fi.get("service", "")})
    events.sort(key=lambda e: e["time"])
    return events


def main():
    os.chdir(PROJ)
    events = load_events()
    print(f"Total GAIA 0704 events: {len(events)}")

    for method in ["localexpert", "dualchannel", "tvdig", "hybrid"]:
        old_log = f"logs/experiments_gaia_{method}_0704.log"
        done_times = get_completed_times(old_log)
        remaining = [e for e in events if e["time"].replace(":", "") not in done_times]
        print(f"\n  {method}: {len(done_times)} done, {len(remaining)} remaining")

        if not remaining:
            print(f"  → already complete, skipping")
            continue

        new_log = f"logs/experiments_gaia_{method}_0704_resume.log"
        extra = METHOD_ARGS[method]
        ok = fail = 0
        for i, ev in enumerate(remaining):
            time_str = ev["time"]
            case_name = f"gaia_{method}_0704_{time_str.replace(':','')}"
            task = (f"At 2021/07/04 {time_str} have exceptions in the "
                    f"microservices system. What are these exceptions? "
                    f"Please output an exception analysis.")
            cmd = [PYTHON, "-u", "run.py", "--task", task, "--name", case_name,
                   "--model", "ollama-qwen3-14b",
                   "--ollama-url", "http://localhost:11434/v1",
                   "--report-dir", f"reports/Report_gaia/{method}_0704"] + extra

            print(f"  [{i+1}/{len(remaining)}] {time_str} | GT={ev['service']}")
            try:
                r = subprocess.run(cmd, capture_output=False, timeout=1800)
                if r.returncode == 0:
                    ok += 1; print(f"    [OK]")
                else:
                    fail += 1; print(f"    [FAIL] rc={r.returncode}")
            except subprocess.TimeoutExpired:
                fail += 1; print(f"    [TIMEOUT]")
            except Exception as e:
                fail += 1; print(f"    [ERROR] {e}")

        print(f"\n  {method} resume: {ok} ok, {fail} fail")

    print("\n=== GAIA 0704 RESUME COMPLETE ===")


if __name__ == "__main__":
    main()
