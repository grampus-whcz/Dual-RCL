#!/usr/bin/env python3
"""Compare the anchor-protection rerun (0501t) against the pre-patch log and
against the offline replay prediction.

Parses:
  - logs/experiments_ccf_hybrid_0501t.log           (pre-patch, original run)
  - logs/experiments_ccf_hybrid_0501t_anchorfix.log (post-patch rerun)
and reports per-case rank-1 flips plus AC@k under both the per-case accounting
and the paper's HHMM accounting.
"""
import glob
import json
import re
import os
from datetime import datetime

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_JSON = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth/groundtruth-2022-05-01.json"


def load_gt():
    g = json.load(open(GT_JSON))
    gt = {}
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


def parse_log(path, want_tv=False):
    preds, meta = {}, {}
    cur = None
    for line in open(path, errors="ignore"):
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
    if not want_tv:
        return preds
    return preds, meta


def hit(p, cands, k):
    if not p:
        return False
    return any(any(c in x for c in cands) for x in p[:k])


def main():
    os.chdir(PROJ)
    gt = load_gt()
    old_p, old_m = parse_log("logs/experiments_ccf_hybrid_0501t.log", want_tv=True)
    new_p = parse_log("logs/experiments_ccf_hybrid_0501t_anchorfix.log")

    # Replay prediction from the OLD log: which cases should flip
    predicted_flip = {}
    for hh, p in old_p.items():
        m = old_m.get(hh, {})
        tv, sc = m.get("tv"), m.get("sc")
        if sc == "multi_anom_single_normal" and p and tv and p[0] != tv[0]:
            predicted_flip[hh] = tv[0]

    print(f"重放预测应改写: {len(predicted_flip)} 例")
    flipped, matched_pred, unexpected = [], [], []
    n = o1 = n1 = o3 = n3 = o5 = n5 = 0
    for hh in sorted(set(gt) & set(new_p)):
        old, new = old_p.get(hh), new_p.get(hh)
        if not old or not new:
            continue
        n += 1
        cands = gt[hh]
        oh1, nh1 = hit(old, cands, 1), hit(new, cands, 1)
        o1 += oh1
        n1 += nh1
        o3 += hit(old, cands, 3)
        n3 += hit(new, cands, 3)
        o5 += hit(old, cands, 5)
        n5 += hit(new, cands, 5)
        if old[0] != new[0]:
            pred0 = predicted_flip.get(hh)
            ok = "✓符合预测" if pred0 == new[0] else "✗偏离预测"
            flipped.append((hh, old[0], oh1, new[0], nh1, ok))
            if pred0 == new[0]:
                matched_pred.append(hh)
            else:
                unexpected.append(hh)
    print(f"\n实测改写: {len(flipped)} 例 (符合预测 {len(matched_pred)}, 偏离 {len(unexpected)})")
    print(f"{'case':<8} {'旧fused@1':<22} {'命中':<5} {'新fused@1':<22} {'命中':<5} 判定")
    for f in flipped:
        print(f"{f[0]:<8} {f[1]:<22} {str(f[2]):<5} {f[3]:<22} {str(f[4]):<5} {f[5]}")
    print(f"\n0501t 单日 AC (N={n}):")
    print(f"  AC@1: {o1}/{n} = {o1/n:.1%}  ->  {n1}/{n} = {n1/n:.1%}")
    print(f"  AC@3: {o3}/{n} = {o3/n:.1%}  ->  {n3}/{n} = {n3/n:.1%}")
    print(f"  AC@5: {o5}/{n} = {o5/n:.1%}  ->  {n5}/{n} = {n5/n:.1%}")


if __name__ == "__main__":
    main()
