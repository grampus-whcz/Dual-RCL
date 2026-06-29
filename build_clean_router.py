#!/usr/bin/env python3
"""Build clean router from VALIDATION data only (no test leakage).

GAIA router training: 0703 Hybrid predictions (validation set)
CCF router training: 0320b + 0320c Hybrid predictions (validation set)
Test: 0701 (GAIA) + 0501-0509 (CCF) — NEVER used for router training.
"""
import os, re, json, pickle, subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GT_TRAIN = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/training_data_with_faults/groundtruth"
GAIA_GT = PROJ / "Datasets/GAIA/fault_injection_tracerank"

FEATURE_NAMES = ['nodes', 'edges', 'density', 'q_tvdig', 'q_causal',
                 'tvdig_unique_top5', 'tvdig_top1_in_fused', 'tvdig_top1_pos']


def gt_cands(cmdb, level):
    c = {cmdb}
    if level == "pod": c.add(re.sub(r"-\d+$", "", cmdb))
    elif level == "service": c.add(re.sub(r"\d+$", "", cmdb).rstrip("-"))
    return c


def grep_extract(logfile, patterns):
    cmd = ["grep", "-aE", "|".join(patterns), logfile]
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
    except: return ""


def parse_log(logfile):
    out = grep_extract(logfile, [r'task_parsed', r'TVDiag root services',
                                 r'Fused root services', r'Adaptive anchor',
                                 r'graph: [0-9]+ nodes'])
    results = {}
    cur_key = None; cur = {}
    for line in out.split("\n"):
        m = re.search(r'"time": "(\d{2})-(\d{2})"', line)
        if m:
            if cur_key: results[cur_key] = cur
            cur_key = f"{m.group(1)}{m.group(2)}"; cur = {'tvdig':None,'fused':None,'q_tvdig':None,'q_causal':None,'nodes':None,'edges':None}
            continue
        if not cur_key: continue
        m = re.search(r"TVDiag root services:\s*(\[.*?\])", line)
        if m:
            try: cur['tvdig'] = [str(p) for p in eval(m.group(1))[:5]]
            except: pass
        m = re.search(r"Fused root services.*?:\s*(\[.*?\])", line)
        if m:
            try: cur['fused'] = [str(p) for p in eval(m.group(1))[:5]]
            except: pass
        m = re.search(r"Adaptive anchor.*q_tvdig=([0-9.]+).*q_causal=([0-9.]+)", line)
        if m: cur['q_tvdig'] = float(m.group(1)); cur['q_causal'] = float(m.group(2))
        m = re.search(r"graph: ([0-9]+) nodes, ([0-9]+) edges", line)
        if m: cur['nodes'] = int(m.group(1)); cur['edges'] = int(m.group(2))
    if cur_key: results[cur_key] = cur
    return results


def build_features(rec):
    f = []
    f.append(rec.get('nodes') or 0)
    f.append(rec.get('edges') or 0)
    f.append((rec.get('edges') or 0) / max(rec.get('nodes') or 1, 1))
    f.append(rec.get('q_tvdig') if rec.get('q_tvdig') is not None else 0.5)
    f.append(rec.get('q_causal') if rec.get('q_causal') is not None else 0.5)
    tv = rec.get('tvdig') or []
    f.append(len(set(tv)))
    fu = rec.get('fused') or []
    f.append(1.0 if (tv and fu and tv[0] in fu) else 0.0)
    f.append(fu.index(tv[0])+1 if (tv and fu and tv[0] in fu) else 6)
    return f


def label_incident(rec, cands):
    tv = rec.get('tvdig') or []; fu = rec.get('fused') or []
    def gt_rank(preds):
        for i, p in enumerate(preds):
            if any(any(c in p for c in cands) for _ in [0]):
                return i + 1
        return 99
    # Correct gt_rank
    def gt_rank2(preds):
        for i, p in enumerate(preds):
            if any(c in p for c in cands):
                return i + 1
        return 99
    tv_r = gt_rank2(tv); fu_r = gt_rank2(fu)
    if fu_r < tv_r: return 1  # causal-anchor (fused) better
    if tv_r < fu_r: return 0  # tvdig better
    return None


def main():
    dataset = []

    # === CCF validation: 0320b + 0320c ===
    print("=== CCF validation (0320b + 0320c) ===")
    for prefix, gt_file in [('0320b', 'groundtruth-k8s-2-2022-03-20.csv'),
                             ('0320c', 'groundtruth-k8s-3-2022-03-20.csv')]:
        log = str(PROJ / f"experiments_ccf_hybrid_{prefix}.log")
        recs = parse_log(log)
        import pandas as pd
        df = pd.read_csv(os.path.join(GT_TRAIN, gt_file))
        gt = {}
        for _, row in df.iterrows():
            ts = datetime.fromtimestamp(int(row['timestamp']))
            gt[ts.strftime('%H%M')] = (row['cmdb_id'], row['level'])
        for hhmm, rec in recs.items():
            if hhmm not in gt: continue
            cmdb, level = gt[hhmm]
            cands = gt_cands(cmdb, level)
            lab = label_incident(rec, cands)
            if lab is None: continue
            dataset.append({'features': build_features(rec), 'label': lab, 'meta': f'ccf_{prefix}_{hhmm}'})
        print(f"  {prefix}: +{len([d for d in dataset if d['meta'].startswith(f'ccf_{prefix}')])} incidents")

    # === GAIA validation: 0703 ===
    print("=== GAIA validation (0703) ===")
    log = str(PROJ / "experiments_gaia_hybrid_0703_router.log")
    recs = parse_log(log)
    with open(GAIA_GT / "fault_injection_list_2021-07-03.pkl", "rb") as f:
        gdata = pickle.load(f)
    gaia_gt = {}
    for fi in gdata:
        if isinstance(fi, dict) and "time" in fi and "service" in fi:
            t = fi["time"].strftime("%H%M") if hasattr(fi["time"], "strftime") else str(fi["time"])
            gaia_gt[t] = fi["service"]
    n_before = len(dataset)
    for hhmm, rec in recs.items():
        if hhmm not in gaia_gt: continue
        svc = gaia_gt[hhmm]
        lab = label_incident(rec, {svc})
        if lab is None: continue
        dataset.append({'features': build_features(rec), 'label': lab, 'meta': f'gaia_0703_{hhmm}'})
    print(f"  0703: +{len(dataset) - n_before} incidents")

    print(f"\n=== Total validation incidents: {len(dataset)} ===")
    labs = [d['label'] for d in dataset]
    print(f"  causal-better (1): {labs.count(1)} ({labs.count(1)/max(len(labs),1):.1%})")
    print(f"  tvdig-better  (0): {labs.count(0)} ({labs.count(0)/max(len(labs),1):.1%})")

    if len(dataset) < 10:
        print("Not enough data for router training.")
        return

    # Train router on validation data
    X = np.array([d['features'] for d in dataset])
    y = np.array([d['label'] for d in dataset])
    scaler = StandardScaler().fit(X)
    clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    clf.fit(scaler.transform(X), y)
    train_acc = accuracy_score(y, clf.predict(scaler.transform(X)))
    print(f"\nRouter training accuracy: {train_acc:.1%} (majority baseline: {max(labs.count(0),labs.count(1))/len(labs):.1%})")

    # Save
    with open(PROJ / "router_model_clean.pkl", "wb") as f:
        pickle.dump({'clf': clf, 'scaler': scaler, 'feature_names': FEATURE_NAMES}, f)
    print(f"Saved clean router to router_model_clean.pkl (trained on validation ONLY)")


if __name__ == "__main__":
    main()
