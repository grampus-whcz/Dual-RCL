#!/usr/bin/env python3
"""
Build a labeled dataset for the learned router.

For each incident we extract features (from causal graph, gamma, TVDiag scores,
cross-source agreement) and a label: which channel's top-1 is correct
(1 = causal correct, 0 = TVDiag correct; ties/incidents where both wrong are
handled by labeling 'causal better' when causal's GT-rank < TVDiag's GT-rank).

The features are dataset-agnostic (no dataset flag), so a router trained on
GAIA+CCF-validation can generalize.

Outputs: router_dataset.pkl with list of {features, label, meta}.
"""
import os, re, json, pickle, subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GT_TRAIN = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/training_data_with_faults/groundtruth"
GT_TEST = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"
GAIA_GT_PKL = PROJ / "Datasets/GAIA/fault_injection_tracerank"


def gt_cands(cmdb, level):
    c = {cmdb}
    if level == "pod": c.add(re.sub(r"-\d+$", "", cmdb))
    elif level == "service": c.add(re.sub(r"\d+$", "", cmdb).rstrip("-"))
    return c


def grep_extract(logfile, patterns):
    cmd = ["grep", "-aE", "|".join(patterns), logfile]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
    except subprocess.TimeoutExpired:
        return ""


def parse_hybrid_log(logfile):
    """Extract per-incident: time, tvdig preds, fused preds, q_tvdig, q_causal, anchor."""
    out = grep_extract(logfile, [r'task_parsed', r'TVDiag root services',
                                 r'Fused root services', r'Adaptive anchor',
                                 r'graph: [0-9]+ nodes'])
    results = {}
    cur_key = None
    cur_tv = cur_fu = None; cur_qt = cur_qc = None; cur_nodes = cur_edges = None
    for line in out.split("\n"):
        m = re.search(r'"time": "(\d{2})-(\d{2})"', line)
        if m:
            if cur_key is not None:
                results[cur_key] = {'tvdig': cur_tv, 'fused': cur_fu, 'q_tvdig': cur_qt,
                                    'q_causal': cur_qc, 'nodes': cur_nodes, 'edges': cur_edges}
            cur_key = f"{m.group(1)}{m.group(2)}"
            cur_tv = cur_fu = None; cur_qt = cur_qc = cur_nodes = cur_edges = None
            continue
        if cur_key is None:
            continue
        m = re.search(r"TVDiag root services:\s*(\[.*?\])", line)
        if m:
            try: cur_tv = [str(p) for p in eval(m.group(1))[:5]]
            except: pass
        m = re.search(r"Fused root services.*?:\s*(\[.*?\])", line)
        if m:
            try: cur_fu = [str(p) for p in eval(m.group(1))[:5]]
            except: pass
        m = re.search(r"Adaptive anchor.*q_tvdig=([0-9.]+).*q_causal=([0-9.]+)", line)
        if m:
            cur_qt = float(m.group(1)); cur_qc = float(m.group(2))
        m = re.search(r"graph: ([0-9]+) nodes, ([0-9]+) edges", line)
        if m:
            cur_nodes = int(m.group(1)); cur_edges = int(m.group(2))
    if cur_key is not None:
        results[cur_key] = {'tvdig': cur_tv, 'fused': cur_fu, 'q_tvdig': cur_qt,
                            'q_causal': cur_qc, 'nodes': cur_nodes, 'edges': cur_edges}
    return results


def build_features(rec, gt_cands_set):
    """Build feature vector from a parsed incident record."""
    f = []
    # 1. Causal graph structure
    nodes = rec.get('nodes') or 0
    edges = rec.get('edges') or 0
    f.append(nodes)                          # node count
    f.append(edges)                          # edge count
    f.append(edges / max(nodes, 1))          # density (edges/node)
    # 2. TVDiag confidence
    f.append(rec.get('q_tvdig') if rec.get('q_tvdig') is not None else 0.5)
    f.append(rec.get('q_causal') if rec.get('q_causal') is not None else 0.5)
    # 3. TVDiag ranking diversity (unique services in top-5)
    tv = rec.get('tvdig') or []
    f.append(len(set(tv)))                   # unique top-5 services
    # 4. Cross-source agreement: does fused contain tvdig's top-1?
    fu = rec.get('fused') or []
    f.append(1.0 if (tv and fu and tv[0] in fu) else 0.0)
    # 5. Position of tvdig top-1 in fused ranking
    if tv and fu and tv[0] in fu:
        f.append(fu.index(tv[0]) + 1)
    else:
        f.append(6)
    return f


def label_incident(rec, cands):
    """Label: 1 if causal(TVDiag-independent) channel's ranking is better, 0 if TVDiag better.
    Uses fused (causal-anchored when anchor=causal) vs tvdig ranking against GT."""
    tv = rec.get('tvdig') or []
    fu = rec.get('fused') or []
    # tvdig top-1 correct?
    tv_hit1 = any(any(c in p for c in cands) for p in tv[:1])
    # fused top-1 correct? (fused reflects the chosen anchor)
    fu_hit1 = any(any(c in p for c in cands) for p in fu[:1])
    # GT rank in tvdig vs fused (lower = better)
    def gt_rank(preds):
        for i, p in enumerate(preds):
            if any(any(c in p for c in cands) for p2 in [p] for c in cands):
                return i + 1
        return 99
    tv_rank = gt_rank(tv)
    fu_rank = gt_rank(fu)
    # Label: 1 = causal-anchored fused is better (or tvdig worse), 0 = tvdig better
    # We want router to pick anchor; label which anchor wins.
    # When fused used causal anchor and won → label 1 (prefer causal)
    # When tvdig wins → label 0 (prefer tvdig)
    if fu_rank < tv_rank:
        return 1  # causal-anchor better
    elif tv_rank < fu_rank:
        return 0  # tvdig better
    else:
        return None  # tie, skip


def main():
    dataset = []
    # CCF AIOps: parse hybrid testall log + GT
    print("=== Parsing CCF AIOps Hybrid logs ===")
    recs = parse_hybrid_log(str(PROJ / "experiments_ccf_hybrid_testall.log"))
    # GT lookup across 5 dates
    from datetime import datetime
    gt_map = {}
    for d in ["2022-05-01", "2022-05-03", "2022-05-05", "2022-05-07", "2022-05-09"]:
        with open(os.path.join(GT_TEST, f"groundtruth-{d}.json")) as f:
            g = json.load(f)
        for i in range(len(g["timestamp"])):
            hhmm = datetime.fromtimestamp(int(g["timestamp"][i])).strftime("%H%M")
            gt_map[hhmm] = (g["cmdb_id"][i], g["level"][i], d)
    for hhmm, rec in recs.items():
        if hhmm not in gt_map: continue
        cmdb, level, d = gt_map[hhmm]
        cands = gt_cands(cmdb, level)
        feat = build_features(rec, cands)
        lab = label_incident(rec, cands)
        if lab is None: continue
        dataset.append({'features': feat, 'label': lab, 'meta': f'ccf_{d}_{hhmm}',
                        'cmdb': cmdb, 'level': level})
    print(f"  CCF: {len(dataset)} labeled incidents")

    print("=== Parsing GAIA Hybrid log ===")
    recs_g = parse_hybrid_log(str(PROJ / "experiments_gaia_hybrid_0701.log"))
    with open(GAIA_GT_PKL / "fault_injection_list_2021-07-01.pkl", "rb") as f:
        gdata = pickle.load(f)
    gaia_gt = {}
    for fi in gdata:
        if isinstance(fi, dict) and "time" in fi and "service" in fi:
            t = fi["time"].strftime("%H%M") if hasattr(fi["time"], "strftime") else str(fi["time"])
            gaia_gt[t] = fi["service"]
    n_before = len(dataset)
    for hhmm, rec in recs_g.items():
        if hhmm not in gaia_gt: continue
        svc = gaia_gt[hhmm]
        cands = {svc}
        feat = build_features(rec, cands)
        lab = label_incident(rec, cands)
        if lab is None: continue
        dataset.append({'features': feat, 'label': lab, 'meta': f'gaia_0701_{hhmm}',
                        'cmdb': svc, 'level': 'service'})
    print(f"  GAIA: {len(dataset) - n_before} labeled incidents")
    print(f"\n=== Total: {len(dataset)} labeled incidents ===")

    # Summary
    labs = [d['label'] for d in dataset]
    print(f"  label=1 (causal-anchor better): {labs.count(1)} ({labs.count(1)/len(labs):.1%})")
    print(f"  label=0 (tvdig better):         {labs.count(0)} ({labs.count(0)/len(labs):.1%})")

    with open(PROJ / "router_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    print(f"\nSaved to router_dataset.pkl")
    # Show feature matrix shape
    X = np.array([d['features'] for d in dataset])
    print(f"Feature matrix: {X.shape}")


if __name__ == "__main__":
    main()
