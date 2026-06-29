#!/usr/bin/env python3
"""
Run GAIA experiments (TVDiag-v3 standalone / Hybrid-v2) on a subset of events.
Mirrors run_ccf_aiops_batch.py but for GAIA dataset.

Usage:
  python run_gaia_batch.py --method tvdig --date 2021-07-01 --max-events 30
  python run_gaia_batch.py --method hybrid --date 2021-07-01 --max-events 30
"""
import argparse, datetime, pickle, os, subprocess

PYTHON = "/root/shared-nvme/.conda/envs/LocaleXpert_env/bin/python"
GT_PKL_DIR = "Datasets/GAIA/fault_injection_tracerank"

METHOD_ARGS = {
    "localexpert": ["--skip-multivariate"],
    "dualchannel": ["--anomaly-method", "tranad", "--anomaly-epochs", "3"],
    # TVDiag-v3 standalone (uses GAIA-trained model with fixed inference)
    "tvdig": ["--anomaly-method", "tranad", "--anomaly-epochs", "3",
              "--rca-method", "tvdig", "--tvdig-model", "./tvdig_checkpoint"],
    # Hybrid-v2 (dual-channel + TVDiag confidence-vote fusion)
    "hybrid": ["--anomaly-method", "tranad", "--anomaly-epochs", "3",
               "--rca-method", "hybrid", "--tvdig-model", "./tvdig_checkpoint"],
}


def load_gaia_events(date_str, max_events=None):
    pkl = os.path.join(GT_PKL_DIR, f"fault_injection_list_{date_str}.pkl")
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    seen = set()
    events = []
    for fi in data:
        if not isinstance(fi, dict) or "time" not in fi or "service" not in fi:
            continue
        t = fi["time"].strftime("%H:%M") if hasattr(fi["time"], "strftime") else str(fi["time"])
        if t in seen:
            continue
        seen.add(t)
        events.append({"time": t, "service": fi.get("service", ""), "cmdb": fi.get("cmdb", fi.get("service", ""))})
    events.sort(key=lambda e: e["time"])
    if max_events:
        # Evenly sample across the day for representativeness
        step = max(1, len(events) // max_events)
        events = events[::step][:max_events]
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["localexpert", "dualchannel", "tvdig", "hybrid"])
    ap.add_argument("--date", default="2021-07-01")
    ap.add_argument("--max-events", type=int, default=30)
    ap.add_argument("--all-events", action="store_true", help="Run ALL events (no sampling)")
    ap.add_argument("--model", default="ollama-qwen3-14b")
    ap.add_argument("--report-dir", default=None)
    args = ap.parse_args()

    mmdd = args.date.replace("2021-", "").replace("-", "")
    if args.report_dir is None:
        args.report_dir = f"reports/Report_gaia/{args.method}_{mmdd}"
    os.makedirs(args.report_dir, exist_ok=True)

    events = load_gaia_events(args.date, None if args.all_events else args.max_events)
    print(f"GAIA {args.date} ({mmdd}): {len(events)} events, method={args.method}, all_events={args.all_events}")

    extra = METHOD_ARGS[args.method]
    success = fail = 0
    for i, ev in enumerate(events):
        time_str = ev["time"]
        case_name = f"gaia_{args.method}_{mmdd}_{time_str.replace(':','')}"
        task = (f"At 2021/{mmdd[:2]}/{mmdd[2:]} {time_str} have exceptions in the "
                f"microservices system. What are these exceptions? Please output an exception analysis.")
        cmd = [PYTHON, "-u", "run.py", "--task", task, "--name", case_name,
               "--model", args.model, "--ollama-url", "http://localhost:11434/v1",
               "--report-dir", args.report_dir] + extra
        print(f"\n[{i+1}/{len(events)}] {time_str} | GT={ev['service']}")
        try:
            r = subprocess.run(cmd, capture_output=False, timeout=1800)
            if r.returncode == 0:
                success += 1; print(f"  [OK] {case_name}")
            else:
                fail += 1; print(f"  [FAIL] {case_name} rc={r.returncode}")
        except subprocess.TimeoutExpired:
            fail += 1; print(f"  [TIMEOUT] {case_name}")
        except Exception as e:
            fail += 1; print(f"  [ERROR] {e}")
    print(f"\n{'='*60}\n  GAIA {args.method} {mmdd}: {len(events)} total, {success} ok, {fail} fail\n{'='*60}")


if __name__ == "__main__":
    main()
