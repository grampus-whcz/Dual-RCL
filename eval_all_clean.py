#!/usr/bin/env python3
"""Final clean evaluation: GAIA 0701 (test) + CCF 0501-0509 (test).
All models trained on non-overlapping data — zero leakage."""
import subprocess, re, os, json, pickle
from datetime import datetime
from pathlib import Path
import numpy as np

PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GAIA_GT = PROJ / "Datasets/GAIA/fault_injection_tracerank"
CCF_GT = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"
CCF_DATES = ["2022-05-01","2022-05-03","2022-05-05","2022-05-07","2022-05-09"]


def gt_cands(cmdb, level):
    c = {cmdb}
    if level == "pod": c.add(re.sub(r"-\d+$","",cmdb))
    elif level == "service": c.add(re.sub(r"\d+$","",cmdb).rstrip("-"))
    return c


def grep_extract(logfile, patterns):
    cmd = ["grep","-aE","|".join(patterns),logfile]
    try: return subprocess.run(cmd,capture_output=True,text=True,timeout=180).stdout
    except: return ""


def parse_preds(logfile):
    """Extract per-incident predictions. Returns {HHMM: [top5 services]}."""
    out = grep_extract(logfile, [r'task_parsed', r'MEPFL top-5 root services',
                                  r'TVDiag root services', r'Fused root services'])
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
                    cur_pred = [p.strip() for p in re.findall(r"\(\d+\)([^,(]+)",raw) if p.strip()][:5]
                break
    if cur_key: results[cur_key] = cur_pred or []
    return results


def eval_gaia(logfile):
    """Evaluate on GAIA 0701 test set."""
    preds = parse_preds(logfile)
    with open(GAIA_GT/"fault_injection_list_2021-07-01.pkl","rb") as f:
        data = pickle.load(f)
    gt = {}
    for fi in data:
        if isinstance(fi,dict) and "time" in fi and "service" in fi:
            t = fi["time"].strftime("%H%M") if hasattr(fi["time"],"strftime") else str(fi["time"])
            gt[t] = fi["service"]
    n_pred=a1=a3=a5=0
    for hhmm, svc in gt.items():
        p = preds.get(hhmm)
        if p:
            n_pred+=1
            if any(svc in x for x in p[:1]): a1+=1
            if any(svc in x for x in p[:3]): a3+=1
            if any(svc in x for x in p[:5]): a5+=1
    n = max(n_pred,1)
    return n_pred, len(gt), a1/n, a3/n, a5/n


def eval_ccf(logfile):
    """Evaluate on CCF 5 test dates."""
    preds = parse_preds(logfile)
    n_pred=a1=a3=a5=total=0
    for d in CCF_DATES:
        with open(os.path.join(CCF_GT,f"groundtruth-{d}.json")) as f:
            g = json.load(f)
        for i in range(len(g["timestamp"])):
            hhmm = datetime.fromtimestamp(int(g["timestamp"][i])).strftime("%H%M")
            cmdb, level = g["cmdb_id"][i], g["level"][i]
            total += 1
            p = preds.get(hhmm)
            if p:
                n_pred+=1
                cands = gt_cands(cmdb, level)
                if any(any(c in x for c in cands) for x in p[:1]): a1+=1
                if any(any(c in x for c in cands) for x in p[:3]): a3+=1
                if any(any(c in x for c in cands) for x in p[:5]): a5+=1
    n = max(n_pred,1)
    return n_pred, total, a1/n, a3/n, a5/n


def main():
    configs = [
        ("LocaleXpert", "experiments_gaia_localexpert_0701_clean.log", "gaia"),
        ("TVDiag", "experiments_gaia_tvdig_0701.log", "gaia"),
        ("Hybrid", "experiments_gaia_hybrid_0701_clean.log", "gaia"),
        ("LocaleXpert", "experiments_ccf_localexpert_testall.log", "ccf"),
        ("DualChannel", "experiments_ccf_dualchannel_testall.log", "ccf"),
        ("TVDiag", "experiments_ccf_tvdig_v3_testall.log", "ccf"),
        ("Hybrid", "experiments_ccf_hybrid_testall.log", "ccf"),
    ]
    print(f"{'Method':<14} {'Dataset':<8} {'preds':>10} {'A@1':>7} {'A@3':>7} {'A@5':>7}")
    print("-"*56)
    for name, logfile, ds in configs:
        path = str(PROJ / logfile)
        if not os.path.exists(path):
            print(f"{name:<14} {ds:<8}  (not found)")
            continue
        try:
            if ds == "gaia":
                np_, nt, a1, a3, a5 = eval_gaia(path)
            else:
                np_, nt, a1, a3, a5 = eval_ccf(path)
            print(f"{name:<14} {ds:<8} {np_}/{nt:<4} {a1:>6.1%} {a3:>6.1%} {a5:>6.1%}")
        except Exception as e:
            print(f"{name:<14} {ds:<8}  ERROR: {e}")


if __name__ == "__main__":
    main()
