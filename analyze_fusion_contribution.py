#!/usr/bin/env python3
"""Honest decomposition: does OUR fusion method help, or is TVDiag doing the work?

Compares, on incidents where fusion ACTUALLY ran (non-fallback):
  - TVDiag-standalone prediction
  - Hybrid-fused prediction
isolating the fusion's marginal contribution.
"""
import subprocess, re, os, json
from datetime import datetime
from pathlib import Path
import numpy as np

GT_TEST_DIR = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"
PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
TEST_DATES = ["2022-05-01", "2022-05-03", "2022-05-05", "2022-05-07", "2022-05-09"]


def load_gt(date_str):
    with open(os.path.join(GT_TEST_DIR, f"groundtruth-{date_str}.json")) as f:
        gt = json.load(f)
    return [(datetime.fromtimestamp(int(gt["timestamp"][i])).strftime("%H%M"),
             gt["cmdb_id"][i], gt["level"][i]) for i in range(len(gt["timestamp"]))]


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


def parse_tvdig(logfile):
    """time -> (preds, was_fallback). For Hybrid log, mark fallback incidents."""
    out = grep_extract(logfile, [r'task_parsed', r'TVDiag root services', r'Fused root services', r'Causal walk failed'])
    results = {}
    cur = None; cur_tvdig = None; cur_fused = None; was_fb = False
    for line in out.split("\n"):
        m = re.search(r'"time": "(\d{2})-(\d{2})"', line)
        if m:
            if cur: results[cur] = (cur_tvdig, cur_fused, was_fb)
            cur = f"{m.group(1)}{m.group(2)}"; cur_tvdig = cur_fused = None; was_fb = False
            continue
        if "Causal walk failed" in line or "TVDiag-only fallback" in line:
            was_fb = True
        m = re.search(r"TVDiag root services:\s*(\[.*?\])", line)
        if m and cur:
            try: cur_tvdig = [str(p) for p in eval(m.group(1))[:5]]
            except: pass
        m = re.search(r"Fused root services.*?:\s*(\[.*?\])", line)
        if m and cur:
            try: cur_fused = [str(p) for p in eval(m.group(1))[:5]]
            except: pass
    if cur: results[cur] = (cur_tvdig, cur_fused, was_fb)
    return results


def main():
    hybrid = parse_tvdig(str(PROJ / "experiments_ccf_hybrid_testall.log"))
    # Build GT lookup
    gt = {}
    for d in TEST_DATES:
        for hhmm, cmdb, level in load_gt(d):
            gt[hhmm] = (cmdb, level)

    # Decompose
    fusion_incidents = []  # fusion ran (not fallback)
    fallback_incidents = []
    for hhmm, (tvdig, fused, was_fb) in hybrid.items():
        if hhmm not in gt: continue
        cmdb, level = gt[hhmm]
        cands = gt_cands(cmdb, level)
        if was_fb or fused is None:
            # fallback: Hybrid uses TVDiag-only
            fallback_incidents.append((tvdig, cands))
        else:
            fusion_incidents.append((tvdig, fused, cands))

    def acc(preds_list, k):
        return sum(1 for p, c in preds_list if p and any(any(cc in pp for cc in c) for pp in p[:k])) / max(len(preds_list), 1)

    print(f"=== 分解：fallback 事件 vs 融合事件 ===")
    print(f"  fallback（崩溃，Hybrid=纯TVDiag）: {len(fallback_incidents)} 事件")
    print(f"  fusion （双通道成功，融合运行）  : {len(fusion_incidents)} 事件")
    print()

    print(f"=== 关键：仅看融合运行的 {len(fusion_incidents)} 事件 ===")
    print(f"{'':30} {'A@1':>7} {'A@3':>7} {'A@5':>7}")
    tv_only = [(t, c) for t, f, c in fusion_incidents]
    fu_only = [(f, c) for t, f, c in fusion_incidents]
    print(f"{'TVDiag-standalone (同事件)':30} {acc(tv_only,1):>6.1%} {acc(tv_only,3):>6.1%} {acc(tv_only,5):>6.1%}")
    print(f"{'Hybrid-fused (我们的融合)':30} {acc(fu_only,1):>6.1%} {acc(fu_only,3):>6.1%} {acc(fu_only,5):>6.1%}")
    print()
    # Net effect of fusion
    d1 = acc(fu_only,1) - acc(tv_only,1)
    d3 = acc(fu_only,3) - acc(tv_only,3)
    d5 = acc(fu_only,5) - acc(tv_only,5)
    print(f"=== 融合的净效应（Hybrid - TVDiag，同事件对比）===")
    print(f"  A@1: {d1:+.1%}  {'(融合有害!)' if d1<0 else '(融合有益)'}")
    print(f"  A@3: {d3:+.1%}  {'(融合有益)' if d3>0 else '(融合有害)'}")
    print(f"  A@5: {d5:+.1%}  {'(融合有益)' if d5>0 else '(融合有害)'}")


if __name__ == "__main__":
    main()
