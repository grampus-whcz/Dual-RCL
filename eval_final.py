#!/usr/bin/env python3
"""Final evaluation: GAIA 0704-0706 aggregated + CCF 5 dates aggregated."""
import subprocess, re, os, json, pickle
from datetime import datetime
from pathlib import Path
import numpy as np

PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GAIA_GT = PROJ / "Datasets/GAIA/fault_injection_tracerank"
CCF_GT = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"
GAIA_DATES = ["2021-07-04", "2021-07-05", "2021-07-06"]
CCF_DATES = ["2022-05-01", "2022-05-03", "2022-05-05", "2022-05-07", "2022-05-09"]
CCF_PREFIXES = ["0501t", "0503t", "0505t", "0507t", "0509t"]


def gt_cands(cmdb, level):
    c = {cmdb}
    if level == "pod": c.add(re.sub(r"-\d+$","",cmdb))
    elif level == "service": c.add(re.sub(r"\d+$","",cmdb).rstrip("-"))
    return c


def grep_extract(logfile):
    cmd = ["grep","-aE",r"task_parsed|MEPFL top-5 root services|TVDiag root services|Fused root services",logfile]
    try: return subprocess.run(cmd,capture_output=True,text=True,timeout=300).stdout
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
                    cur_pred = [p.strip() for p in re.findall(r"\(\d+\)([^,(]+)",raw) if p.strip()][:5]
                break
    if cur_key: results[cur_key] = cur_pred or []
    return results


def eval_gaia(method):
    """Aggregate GAIA 0704+0705+0706."""
    gt_all = {}
    for d in GAIA_DATES:
        with open(GAIA_GT/f"fault_injection_list_{d}.pkl","rb") as f:
            for fi in pickle.load(f):
                if isinstance(fi,dict) and "time" in fi and "service" in fi:
                    t = fi["time"].strftime("%H%M") if hasattr(fi["time"],"strftime") else str(fi["time"])
                    gt_all[t] = fi["service"]
    suffixes = ["0704","0705","0706"]
    all_preds = {}
    for s in suffixes:
        lf = str(PROJ/f"logs/experiments_gaia_{method}_{s}.log")
        if os.path.exists(lf): all_preds.update(parse_preds(lf))
    n_pred=a1=a3=a5=0
    for hhmm, svc in gt_all.items():
        p = all_preds.get(hhmm)
        if p:
            n_pred+=1
            if any(svc in x for x in p[:1]): a1+=1
            if any(svc in x for x in p[:3]): a3+=1
            if any(svc in x for x in p[:5]): a5+=1
    n=max(n_pred,1)
    return n_pred, len(gt_all), a1/n, a3/n, a5/n


def eval_ccf(method):
    """Aggregate CCF 5 dates."""
    gt_all = {}
    for d in CCF_DATES:
        with open(os.path.join(CCF_GT,f"groundtruth-{d}.json")) as f:
            g = json.load(f)
        for i in range(len(g["timestamp"])):
            hhmm = datetime.fromtimestamp(int(g["timestamp"][i])).strftime("%H%M")
            gt_all[hhmm] = (g["cmdb_id"][i], g["level"][i])
    all_preds = {}
    for p in CCF_PREFIXES:
        lf = str(PROJ/f"logs/experiments_ccf_{method}_{p}.log")
        if os.path.exists(lf): all_preds.update(parse_preds(lf))
    n_pred=a1=a3=a5=0
    for hhmm, (cmdb, level) in gt_all.items():
        p = all_preds.get(hhmm)
        if p:
            n_pred+=1
            cands = gt_cands(cmdb, level)
            if any(any(c in x for c in cands) for x in p[:1]): a1+=1
            if any(any(c in x for c in cands) for x in p[:3]): a3+=1
            if any(any(c in x for c in cands) for x in p[:5]): a5+=1
    n=max(n_pred,1)
    return n_pred, len(gt_all), a1/n, a3/n, a5/n


def main():
    methods = ["localexpert", "dualchannel", "tvdig", "hybrid"]
    labels = {"localexpert":"Single-channel","dualchannel":"DualChannel","tvdig":"Multimodal","hybrid":"Hybrid"}
    print(f"{'Method':<20} {'GAIA preds':>10} {'GAIA A@1':>8} {'GAIA A@3':>8} {'GAIA A@5':>8} | {'CCF preds':>10} {'CCF A@1':>8} {'CCF A@3':>8} {'CCF A@5':>8}")
    print("-"*95)
    results = {}
    for m in methods:
        gp, gt, ga1, ga3, ga5 = eval_gaia(m)
        cp, ct, ca1, ca3, ca5 = eval_ccf(m)
        results[m] = (ga1,ga3,ga5,ca1,ca3,ca5)
        print(f"{labels[m]:<20} {gp}/{gt:<4} {ga1:>7.1%} {ga3:>7.1%} {ga5:>7.1%} | {cp}/{ct:<4} {ca1:>7.1%} {ca3:>7.1%} {ca5:>7.1%}")
    return results


if __name__ == "__main__":
    main()
