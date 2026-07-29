#!/usr/bin/env python3
"""Final evaluation: GAIA 0704+0705+0706 (with resume logs) + CCF 5 dates."""
import subprocess, re, os, json, pickle
from datetime import datetime
from pathlib import Path
import numpy as np

PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GT_DIR = PROJ / "Datasets/GAIA/fault_injection_tracerank"
CCF_GT = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"

def grep_lines(logfile, pattern):
    try:
        r = subprocess.run(["grep","-a",pattern,logfile], capture_output=True, text=True, timeout=300)
        return r.stdout.strip().split("\n") if r.stdout.strip() else []
    except: return []

def parse_preds(logfiles):
    all_lines = []
    for lf in logfiles:
        if not os.path.exists(lf): continue
        for pat in ['"type": "task_parsed"', 'root services']:
            all_lines.extend(grep_lines(lf, pat))
    results = {}; cur_key = None; cur_pred = None
    for line in all_lines:
        m = re.search(r'"time": "(\d{2})-(\d{2})"', line)
        if m:
            if cur_key: results[cur_key] = cur_pred or []
            cur_key = f"{m.group(1)}{m.group(2)}"; cur_pred = None; continue
        if not cur_key: continue
        for pat in [r"Fused root services.*?:\s*(\[.*?\])",
                    r"TVDiag root services:\s*(\[.*?\])",
                    r"MEPFL top-5 root services:\s*(.*)"]:
            mm = re.search(pat, line)
            if mm:
                raw = mm.group(1)
                try:
                    cur_pred = [str(p) for p in eval(raw)[:5]] if raw.startswith("[") else \
                               [p.strip() for p in re.findall(r"\(\d+\)([^,(]+)", raw) if p.strip()][:5]
                except: pass
                break
    if cur_key: results[cur_key] = cur_pred or []
    return results

def load_gaia_gt():
    gt = {}
    for d in ["2021-07-04","2021-07-05","2021-07-06"]:
        with open(GT_DIR/f"fault_injection_list_{d}.pkl","rb") as f:
            for fi in pickle.load(f):
                if isinstance(fi,dict) and "time" in fi and "service" in fi:
                    t=fi["time"].strftime("%H%M") if hasattr(fi["time"],"strftime") else str(fi["time"])
                    gt[t]=fi["service"]
    return gt

def load_ccf_gt():
    gt={}
    for d in ["2022-05-01","2022-05-03","2022-05-05","2022-05-07","2022-05-09"]:
        with open(f"{CCF_GT}/groundtruth-{d}.json") as f: g=json.load(f)
        for i in range(len(g["timestamp"])):
            hhmm=datetime.fromtimestamp(int(g["timestamp"][i])).strftime("%H%M")
            cmdb=g["cmdb_id"][i];level=g["level"][i]
            cands={cmdb}
            if level=="pod": cands.add(re.sub(r"-\d+$","",cmdb))
            elif level=="service": cands.add(re.sub(r"\d+$","",cmdb).rstrip("-"))
            gt[hhmm]=cands
    return gt

def evaluate(preds,gt,is_gaia=True):
    n_pred=a1=a3=a5=total=0
    for hhmm,gv in gt.items():
        total+=1; p=preds.get(hhmm)
        if p:
            n_pred+=1
            if is_gaia:
                if any(gv in x for x in p[:1]): a1+=1
                if any(gv in x for x in p[:3]): a3+=1
                if any(gv in x for x in p[:5]): a5+=1
            else:
                if any(any(c in x for c in gv) for x in p[:1]): a1+=1
                if any(any(c in x for c in gv) for x in p[:3]): a3+=1
                if any(any(c in x for c in gv) for x in p[:5]): a5+=1
    n=max(n_pred,1)
    return n_pred,total,a1/n,a3/n,a5/n

def main():
    os.chdir(PROJ)
    gaia_gt=load_gaia_gt(); ccf_gt=load_ccf_gt()
    gaia_cfg=[
        ("LocaleXpert",["logs/experiments_gaia_localexpert_0704.log","logs/experiments_gaia_localexpert_0704_resume.log","logs/experiments_gaia_localexpert_0705.log","logs/experiments_gaia_localexpert_0706.log"]),
        ("DualChannel",["logs/experiments_gaia_dualchannel_0704.log","logs/experiments_gaia_dualchannel_0704_resume.log","logs/experiments_gaia_dualchannel_0705.log","logs/experiments_gaia_dualchannel_0706.log"]),
        ("TVDiag",["logs/experiments_gaia_tvdig_0704.log","logs/experiments_gaia_tvdig_0704_resume.log","logs/experiments_gaia_tvdig_0705.log","logs/experiments_gaia_tvdig_0706.log"]),
        ("Hybrid",["logs/experiments_gaia_hybrid_0704.log","logs/experiments_gaia_hybrid_0704_resume.log","logs/experiments_gaia_hybrid_0705.log","logs/experiments_gaia_hybrid_0706.log"]),
    ]
    ccf_cfg=[
        ("LocaleXpert",[f"logs/experiments_ccf_localexpert_{p}.log" for p in "0501t 0503t 0505t 0507t 0509t".split()]),
        ("DualChannel",[f"logs/experiments_ccf_dualchannel_{p}.log" for p in "0501t 0503t 0505t 0507t 0509t".split()]),
        ("TVDiag",[f"logs/experiments_ccf_tvdig_{p}.log" for p in "0501t 0503t 0505t 0507t 0509t".split()]),
        ("Hybrid",[f"logs/experiments_ccf_hybrid_{p}.log" for p in "0501t 0503t 0505t 0507t 0509t".split()]),
    ]
    print("="*60)
    print("  FINAL EVALUATION")
    print("="*60)
    print(f"\n{'Method':<14} {'Data':<6} {'preds':>10} {'AC@1':>7} {'AC@3':>7} {'AC@5':>7}")
    print("-"*56)
    for name,lfs in gaia_cfg:
        preds=parse_preds([str(PROJ/x) for x in lfs if os.path.exists(PROJ/x)])
        np_,nt,a1,a3,a5=evaluate(preds,gaia_gt,True)
        print(f"{name:<14} {'GAIA':<6} {np_}/{nt:<4} {a1:>6.1%} {a3:>6.1%} {a5:>6.1%}")
    for name,lfs in ccf_cfg:
        preds=parse_preds([str(PROJ/x) for x in lfs if os.path.exists(PROJ/x)])
        np_,nt,a1,a3,a5=evaluate(preds,ccf_gt,False)
        print(f"{name:<14} {'CCF':<6} {np_}/{nt:<4} {a1:>6.1%} {a3:>6.1%} {a5:>6.1%}")

if __name__=="__main__": main()
