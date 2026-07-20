#!/usr/bin/env python3
"""Evaluate GAIA 0704+0705+0706 aggregated (temporal: train 0701-0703, test 0704-0706)."""
import subprocess, re, os, pickle
from datetime import datetime
from pathlib import Path
import numpy as np

PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GT_DIR = PROJ / "Datasets/GAIA/fault_injection_tracerank"
TEST_DATES = ["2021-07-04", "2021-07-05", "2021-07-06"]


def load_gt(date_str):
    with open(GT_DIR / f"fault_injection_list_{date_str}.pkl", "rb") as f:
        data = pickle.load(f)
    gt = {}
    for fi in data:
        if isinstance(fi, dict) and "time" in fi and "service" in fi:
            t = fi["time"].strftime("%H%M") if hasattr(fi["time"], "strftime") else str(fi["time"])
            gt[(date_str, t)] = fi["service"]
    return gt


def grep_extract(logfile):
    cmd = ["grep", "-aE", r"task_parsed|MEPFL top-5 root services|TVDiag root services|Fused root services", logfile]
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    except: return ""


def parse_preds(logfile):
    out = grep_extract(logfile)
    results = {}; cur_key=None; cur_pred=None
    for line in out.split("\n"):
        m = re.search(r'"time": "(\d{2})-(\d{2})"', line)
        if m:
            if cur_key: results[cur_key] = cur_pred or []
            cur_key = f"{m.group(1)}{m.group(2)}"; cur_pred = None
            continue
        if not cur_key: continue
        for pat in [r"MEPFL top-5 root services:\s*(.*)",
                    r"TVDiag root services:\s*(\[.*?\])",
                    r"Fused root services.*?:\s*(\[.*?\])"]:
            mm = re.search(pat, line)
            if mm:
                raw = mm.group(1)
                if raw.startswith("["):
                    try: cur_pred = [str(p) for p in eval(raw)[:5]]
                    except: pass
                else:
                    cur_pred = [p.strip() for p in re.findall(r"\(\d+\)([^,(]+)", raw) if p.strip()][:5]
                break
    if cur_key: results[cur_key] = cur_pred or []
    return results


def evaluate_multi(logfiles, gt_all):
    """Aggregate predictions across multiple dates."""
    all_preds = {}
    for lf in logfiles:
        all_preds.update(parse_preds(lf))
    n_pred=a1=a3=a5=total=0
    for (date_str, hhmm), svc in gt_all.items():
        total += 1
        p = all_preds.get(hhmm)
        if p:
            n_pred+=1
            if any(svc in x for x in p[:1]): a1+=1
            if any(svc in x for x in p[:3]): a3+=1
            if any(svc in x for x in p[:5]): a5+=1
    n=max(n_pred,1)
    return n_pred, total, a1/n, a3/n, a5/n


def main():
    gt_all = {}
    for d in TEST_DATES:
        gt_all.update(load_gt(d))

    configs = [
        ("LocaleXpert", ["logs/experiments_gaia_localexpert_0704.log",
                          "logs/experiments_gaia_localexpert_0705.log",
                          "logs/experiments_gaia_localexpert_0706.log"]),
        ("DualChannel", ["logs/experiments_gaia_dualchannel_0704.log",
                          "logs/experiments_gaia_dualchannel_0705.log",
                          "logs/experiments_gaia_dualchannel_0706.log"]),
        ("TVDiag", ["logs/experiments_gaia_tvdig_0704.log",
                     "logs/experiments_gaia_tvdig_0705.log",
                     "logs/experiments_gaia_tvdig_0706.log"]),
        ("Hybrid", ["logs/experiments_gaia_hybrid_0704.log",
                     "logs/experiments_gaia_hybrid_0705.log",
                     "logs/experiments_gaia_hybrid_0706.log"]),
    ]
    print(f"=== GAIA 0704+0705+0706 AGGREGATED (temporal split) ===")
    print(f"{'Method':<14} {'preds':>10} {'A@1':>7} {'A@3':>7} {'A@5':>7}")
    print("-"*48)
    for name, logfiles in configs:
        paths = [str(PROJ / lf) for lf in logfiles]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"{name:<14}  missing: {missing}")
            continue
        np_, nt, a1, a3, a5 = evaluate_multi(paths, gt_all)
        print(f"{name:<14} {np_}/{nt:<4} {a1:>6.1%} {a3:>6.1%} {a5:>6.1%}")

    # Also per-date breakdown
    print(f"\n=== Per-date breakdown ===")
    for d in TEST_DATES:
        gt_d = load_gt(d)
        suffix = d[5:7] + d[8:10]
        print(f"\n--- {d} ({suffix}) ---")
        print(f"{'Method':<14} {'preds':>10} {'A@1':>7} {'A@3':>7} {'A@5':>7}")
        for name, logfiles in configs:
            # Find the logfile for this date
            lf = [lf for lf in logfiles if suffix in lf]
            if not lf: continue
            path = str(PROJ / lf[0])
            if not os.path.exists(path): continue
            np_, nt, a1, a3, a5 = evaluate_multi([path], gt_d)
            print(f"{name:<14} {np_}/{nt:<4} {a1:>6.1%} {a3:>6.1%} {a5:>6.1%}")


if __name__ == "__main__":
    main()
