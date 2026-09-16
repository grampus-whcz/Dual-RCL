#!/usr/bin/env python3
"""Bootstrap CI for the Scenario-1 anchor-protection patch on CCF AIOps test set.

Replays the paper-accounting (HHMM-keyed, last-write-wins) per-case outcomes for
Hybrid (pre-patch logs), applies the patch rule (restore TVDiag top-1 under
multi_anom_single_normal), and bootstraps the paired per-minute AC@k deltas.

Usage:
    python evaluation/patch_bootstrap.py [--n-boot 10000] [--seed 0]
"""
import argparse
import glob
import json
import re
import os
from datetime import datetime

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CCF_GT = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"
PREFS = ["0501t", "0503t", "0505t", "0507t", "0509t"]


def load_gt():
    gt = {}
    for d in ["2022-05-01", "2022-05-03", "2022-05-05", "2022-05-07", "2022-05-09"]:
        g = json.load(open(f"{CCF_GT}/groundtruth-{d}.json"))
        for i in range(len(g["timestamp"])):
            hhmm = datetime.fromtimestamp(int(g["timestamp"][i])).strftime("%H%M")
            cmdb, level = g["cmdb_id"][i], g["level"][i]
            cands = {cmdb}
            if level == "pod":
                cands.add(re.sub(r"-\d+$", "", cmdb))
            elif level == "service":
                cands.add(re.sub(r"\d+$", "", cmdb).rstrip("-"))
            gt[hhmm] = cands
    return gt


def parse_hybrid():
    preds, meta = {}, {}
    for pref in PREFS:
        for p in sorted(glob.glob(f"{PROJ}/logs/experiments_ccf_hybrid_{pref}*.log")):
            if "anchorfix" in p:
                continue
            cur = None
            for line in open(p, errors="ignore"):
                line = line.strip()
                m = re.search(r'"type": "task_parsed", "date": "(\w+)", "time": "(\d{2})-(\d{2})"', line)
                if m:
                    cur = m.group(2) + m.group(3)
                    preds.setdefault(cur, None)
                    meta.setdefault(cur, {})
                    continue
                if cur is None:
                    continue
                mm = re.search(r"TVDiag root services:\s*(\[.*?\])", line)
                if mm:
                    try:
                        meta[cur]["tv"] = eval(mm.group(1))
                    except Exception:
                        pass
                mm = re.search(r"Fused root services(?:\s*\(confidence-vote\))?:\s*(\[.*?\])", line)
                if mm:
                    try:
                        preds[cur] = eval(mm.group(1))
                    except Exception:
                        pass
                mm = re.search(r"Conflict scenario:\s*(\w+)", line)
                if mm:
                    meta[cur]["sc"] = mm.group(1)
    return preds, meta


def hit(preds, cands, k):
    if not preds:
        return False
    return any(any(c in x for c in cands) for x in preds[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.chdir(PROJ)
    gt = load_gt()
    preds, meta = parse_hybrid()

    # Paired per-minute outcomes: orig vs patched, per k
    records = []  # (hhmm, k, orig_hit, patched_hit)
    for hh, cands in gt.items():
        p = preds.get(hh)
        if not p:
            continue
        m = meta.get(hh, {})
        tv, sc = m.get("tv"), m.get("sc")
        patched = p
        if sc == "multi_anom_single_normal" and tv and p[0] != tv[0]:
            patched = [tv[0]] + [x for x in p if x != tv[0]]
        for k in (1, 3, 5):
            records.append((hh, k, hit(p, cands, k), hit(patched, cands, k)))

    rng = np.random.default_rng(args.seed)
    n = len(gt)
    print(f"N GT minutes = {n}")
    for k in (1, 3, 5):
        rec = [(hh, o, q) for hh, kk, o, q in records if kk == k]
        orig = np.array([o for _, o, _ in rec], dtype=float)
        patch = np.array([q for _, _, q in rec], dtype=float)
        delta = patch - orig
        n_improved = int((delta > 0).sum())
        n_regressed = int((delta < 0).sum())
        boots = np.empty(args.n_boot)
        idx = np.arange(len(delta))
        for b in range(args.n_boot):
            s = rng.choice(idx, size=len(idx), replace=True)
            boots[b] = delta[s].mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p_exact = (boots <= 0).mean()  # one-sided P(no improvement)
        print(
            f"AC@{k}: orig {orig.mean():.1%} -> patched {patch.mean():.1%} "
            f"(+{delta.mean() * 100:.1f}pt, +{n_improved}/-{n_regressed} cases)  "
            f"95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]pt  "
            f"P(delta<=0)={p_exact:.3f}"
        )


if __name__ == "__main__":
    main()
