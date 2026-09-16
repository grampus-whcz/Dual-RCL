#!/usr/bin/env python3
"""Evaluate the anchor-protection rerun on the CCF AIOps validation split
(cloudbed-2 + cloudbed-3, 2022-03-20, 64 incidents) and collect the
instrumentation needed for margin-gated anchor protection.

Compares against the paper's validation row (Table ccf_val):
    Single-channel/DualChannel  3.7/22.2/37.0
    Multimodal (TVDiag)         25.8/25.8/45.2
    Hybrid (pre-patch)          19.4/19.4/38.7
"""
import glob
import os
import re
from datetime import datetime

import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_DIR = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/training_data_with_faults/groundtruth"
RUNS = {  # prefix -> cloudbed number
    "0320b": "2",
    "0320c": "3",
}
DATE = "2022-03-20"


def load_gt(cb):
    df = pd.read_csv(f"{GT_DIR}/groundtruth-k8s-{cb}-{DATE}.csv")
    gt = {}
    for _, row in df.iterrows():
        hhmm = datetime.fromtimestamp(row["timestamp"]).strftime("%H%M")
        cmdb, level = row["cmdb_id"], row["level"]
        cands = {cmdb}
        if level == "pod":
            cands.add(re.sub(r"-\d+$", "", cmdb))
        elif level == "service":
            cands.add(re.sub(r"\d+$", "", cmdb).rstrip("-"))
        gt[hhmm] = cands
    return gt


def parse_log(path):
    preds, prot = {}, []
    cur = None
    for line in open(path, errors="ignore"):
        line = line.strip()
        m = re.search(r'"type": "task_parsed", "date": "(\w+)", "time": "(\d{2})-(\d{2})"', line)
        if m:
            cur = m.group(2) + m.group(3)
            preds.setdefault(cur, None)
            continue
        if cur is None:
            continue
        mm = re.search(r"Fused root services(?:\s*\(confidence-vote\))?:\s*(\[.*?\])", line)
        if mm:
            try:
                preds[cur] = eval(mm.group(1))
            except Exception:
                pass
        mm = re.search(r"Anchor protection fired: pre-protection order (\[.*?\])", line)
        if mm:
            try:
                prot.append({"hhmm": cur, "pre": eval(mm.group(1))})
            except Exception:
                pass
    return preds, prot


def hit(p, cands, k):
    if not p:
        return False
    return any(any(c in x for c in cands) for x in p[:k])


def main():
    os.chdir(PROJ)
    tot = {1: 0, 3: 0, 5: 0}
    n_all = 0
    for pref, cb in RUNS.items():
        log = f"logs/experiments_ccf_hybrid_{pref}_anchorfix.log"
        if not os.path.exists(log):
            print(f"[skip] {log} 不存在")
            continue
        gt = load_gt(cb)
        preds, prot = parse_log(log)
        n_ok = sum(1 for hh in gt if preds.get(hh))
        counts = {k: sum(hit(preds.get(hh), gt[hh], k) for hh in gt) for k in (1, 3, 5)}
        for k in (1, 3, 5):
            tot[k] += counts[k]
        n_all += len(gt)
        print(f"{pref} (cloudbed-{cb}): N={len(gt)}, 有预测={n_ok}, "
              f"AC@1={counts[1]/len(gt):.1%} AC@3={counts[3]/len(gt):.1%} AC@5={counts[5]/len(gt):.1%}, "
              f"保护触发={len(prot)}")
        for p in prot[:6]:
            print(f"    {p['hhmm']}: pre={p['pre'][:3]}")
    if n_all:
        print(f"\n合并 (N={n_all}): AC@1={tot[1]/n_all:.1%} AC@3={tot[3]/n_all:.1%} AC@5={tot[5]/n_all:.1%}")
        print("论文验证集参考: Hybrid 19.4/19.4/38.7 | TVDiag 25.8/25.8/45.2 | 单通道 3.7/22.2/37.0")


if __name__ == "__main__":
    main()
