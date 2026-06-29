#!/usr/bin/env python3
"""Evaluate GAIA 0704 (temporal test set: train on 0701-0703, test on 0704)."""
import subprocess, re, os, pickle
from pathlib import Path
import numpy as np

PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GT_PKL = PROJ / "Datasets/GAIA/fault_injection_tracerank/fault_injection_list_2021-07-04.pkl"


def load_gt():
    with open(GT_PKL, "rb") as f:
        data = pickle.load(f)
    gt = {}
    for fi in data:
        if isinstance(fi, dict) and "time" in fi and "service" in fi:
            t = fi["time"].strftime("%H%M") if hasattr(fi["time"], "strftime") else str(fi["time"])
            gt[t] = fi["service"]
    return gt


def grep_extract(logfile):
    cmd = ["grep", "-aE", r"task_parsed|MEPFL top-5 root services|TVDiag root services|Fused root services", logfile]
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
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


def evaluate(logfile, gt):
    preds = parse_preds(logfile)
    n_pred=a1=a3=a5=0
    for hhmm, svc in gt.items():
        p = preds.get(hhmm)
        if p:
            n_pred+=1
            if any(svc in x for x in p[:1]): a1+=1
            if any(svc in x for x in p[:3]): a3+=1
            if any(svc in x for x in p[:5]): a5+=1
    n=max(n_pred,1)
    return n_pred, len(gt), a1/n, a3/n, a5/n


def main():
    gt = load_gt()
    configs = [
        ("LocaleXpert", "experiments_gaia_localexpert_0704.log"),
        ("DualChannel", "experiments_gaia_dualchannel_0704.log"),
        ("TVDiag", "experiments_gaia_tvdig_0704.log"),
        ("Hybrid", "experiments_gaia_hybrid_0704.log"),
    ]
    print(f"=== GAIA 0704 (temporal: train=0701-0703, test=0704) ===")
    print(f"{'Method':<14} {'preds':>10} {'A@1':>7} {'A@3':>7} {'A@5':>7}")
    print("-"*48)
    for name, logfile in configs:
        path = str(PROJ / logfile)
        if not os.path.exists(path):
            print(f"{name:<14}  (not found)")
            continue
        np_, nt, a1, a3, a5 = evaluate(path, gt)
        print(f"{name:<14} {np_}/{nt:<4} {a1:>6.1%} {a3:>6.1%} {a5:>6.1%}")


if __name__ == "__main__":
    main()
