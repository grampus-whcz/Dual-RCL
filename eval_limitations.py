#!/usr/bin/env python3
"""
Experiments 1-2: Conflict-scenario isolation analysis.
Extracts conflict scenarios from logs and computes per-scenario A@k.
Zero API cost — purely log-based.
"""
import os, re, json, subprocess, pickle
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import numpy as np

PROJ = Path("/root/shared-nvme/work/code/RCA/2026/SoC-RCA")
GT_DIR = PROJ / "Datasets/GAIA/fault_injection_tracerank"
CCF_GT = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/test_data/groundtruth"


def grep_extract(logfile, patterns):
    cmd = ["grep", "-aE", "|".join(patterns), logfile]
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=300).stdout
    except: return ""


def parse_full(logfile):
    """Extract per-event: time, predictions (MEPFL/TVDiag/Fused), conflict scenario."""
    out = grep_extract(logfile, [
        r'task_parsed', r'MEPFL top-5 root services',
        r'TVDiag root services', r'Fused root services',
        r'Conflict scenario', r'e2e_latency', r'Phase 9.*completed in'
    ])
    results = {}; cur_key = None; cur = {}
    for line in out.split("\n"):
        m = re.search(r'"time": "(\d{2})-(\d{2})"', line)
        if m:
            if cur_key: results[cur_key] = cur
            cur_key = f"{m.group(1)}{m.group(2)}"; cur = {'preds': None, 'scenario': 'unknown', 'latency': None}
            continue
        if not cur_key: continue
        # Predictions (last one wins: Fused > TVDiag > MEPFL)
        for pat, parser in [
            (r"MEPFL top-5 root services:\s*(.*)", lambda r: [p.strip() for p in re.findall(r"\(\d+\)([^,(]+)", r) if p.strip()][:5]),
            (r"TVDiag root services:\s*(\[.*?\])", lambda r: [str(p) for p in eval(r)[:5]] if r.startswith("[") else []),
            (r"Fused root services.*?:\s*(\[.*?\])", lambda r: [str(p) for p in eval(r)[:5]] if r.startswith("[") else []),
        ]:
            mm = re.search(pat, line)
            if mm:
                try: cur['preds'] = parser(mm.group(1))
                except: pass
        m = re.search(r'Conflict scenario:\s*(\w+)', line)
        if m and cur.get('scenario') == 'unknown':
            cur['scenario'] = m.group(1)
        m = re.search(r'"duration_s":\s*([\d.]+)', line)
        if m: cur['latency'] = float(m.group(1))
        m = re.search(r'\[Phase 9\].*completed in\s+([\d.]+)s', line)
        if m and cur['latency'] is None: cur['latency'] = float(m.group(1))
    if cur_key: results[cur_key] = cur
    return results


def load_gaia_gt():
    gt = {}
    for d in ["2021-07-04", "2021-07-05", "2021-07-06"]:
        with open(GT_DIR / f"fault_injection_list_{d}.pkl", "rb") as f:
            data = pickle.load(f)
        for fi in data:
            if isinstance(fi, dict) and "time" in fi and "service" in fi:
                t = fi["time"].strftime("%H%M") if hasattr(fi["time"], "strftime") else str(fi["time"])
                gt[t] = fi["service"]
    return gt


def load_ccf_gt():
    gt = {}
    for d in ["2022-05-01","2022-05-03","2022-05-05","2022-05-07","2022-05-09"]:
        with open(os.path.join(CCF_GT, f"groundtruth-{d}.json")) as f:
            g = json.load(f)
        for i in range(len(g["timestamp"])):
            hhmm = datetime.fromtimestamp(int(g["timestamp"][i])).strftime("%H%M")
            cmdb = g["cmdb_id"][i]
            level = g["level"][i]
            cands = {cmdb}
            if level == "pod": cands.add(re.sub(r"-\d+$", "", cmdb))
            elif level == "service": cands.add(re.sub(r"\d+$", "", cmdb).rstrip("-"))
            gt[hhmm] = (cmdb, cands)
    return gt


def compute_ak(preds, gt_val, is_gaia=True):
    """Compute A@1/3/5 for one event."""
    if is_gaia:
        svc = gt_val
        hit = lambda k: any(svc in p for p in preds[:k]) if preds else False
    else:
        cmdb, cands = gt_val
        hit = lambda k: any(any(c in p for c in cands) for p in preds[:k]) if preds else False
    return hit(1), hit(3), hit(5)


def main():
    os.chdir(PROJ)

    # ============ Experiment 1: GAIA conflict-scenario isolation ============
    print("=" * 70)
    print("  Experiment 1: GAIA Conflict-Scenario Isolation")
    print("=" * 70)

    gt = load_gaia_gt()

    # Parse DualChannel logs (has conflict scenario data)
    dc_data = {}
    for lf in ["logs/experiments_gaia_dualchannel_0704.log",
               "logs/experiments_gaia_dualchannel_0705.log",
               "logs/experiments_gaia_dualchannel_0706.log"]:
        if os.path.exists(lf):
            dc_data.update(parse_full(lf))

    # Parse LocaleXpert logs (same pipeline, no conflict — all "consistent")
    le_data = {}
    for lf in ["logs/experiments_gaia_localexpert_0704.log",
               "logs/experiments_gaia_localexpert_0705.log",
               "logs/experiments_gaia_localexpert_0706.log"]:
        if os.path.exists(lf):
            le_data.update(parse_full(lf))

    # Group events by conflict scenario
    scenario_events = defaultdict(list)
    for hhmm, rec in dc_data.items():
        if hhmm not in gt: continue
        scenario = rec.get('scenario', 'unknown')
        scenario_events[scenario].append(hhmm)

    print(f"\n  Conflict scenario distribution:")
    for sc, evs in sorted(scenario_events.items(), key=lambda x: -len(x[1])):
        print(f"    {sc}: {len(evs)} events")

    # Compute per-scenario A@k for LocaleXpert vs DualChannel
    print(f"\n  Per-scenario A@k (LocaleXpert vs DualChannel):")
    print(f"  {'Scenario':<30} {'N':>4} {'LE A@1':>7} {'LE A@3':>7} {'LE A@5':>7} | {'DC A@1':>7} {'DC A@3':>7} {'DC A@5':>7}")
    print("  " + "-" * 95)

    for scenario in ['consistent', 'both_anom_different_root', 'multi_anom_single_normal']:
        evs = scenario_events.get(scenario, [])
        if not evs: continue
        le_hits = {'a1':0,'a3':0,'a5':0}
        dc_hits = {'a1':0,'a3':0,'a5':0}
        n_le = n_dc = 0
        for hhmm in evs:
            svc = gt[hhmm]
            # LocaleXpert
            le_preds = le_data.get(hhmm, {}).get('preds')
            if le_preds:
                n_le += 1
                h1,h3,h5 = compute_ak(le_preds, svc)
                le_hits['a1'] += h1; le_hits['a3'] += h3; le_hits['a5'] += h5
            # DualChannel
            dc_preds = dc_data.get(hhmm, {}).get('preds')
            if dc_preds:
                n_dc += 1
                h1,h3,h5 = compute_ak(dc_preds, svc)
                dc_hits['a1'] += h1; dc_hits['a3'] += h3; dc_hits['a5'] += h5

        n = max(len(evs), 1)
        le_n = max(n_le, 1)
        dc_n = max(n_dc, 1)
        print(f"  {scenario:<30} {len(evs):>4} "
              f"{le_hits['a1']/le_n:>6.1%} {le_hits['a3']/le_n:>6.1%} {le_hits['a5']/le_n:>6.1%} | "
              f"{dc_hits['a1']/dc_n:>6.1%} {dc_hits['a3']/dc_n:>6.1%} {dc_hits['a5']/dc_n:>6.1%}")

    # ============ Experiment 2: CCF conflict-scenario isolation ============
    print(f"\n{'=' * 70}")
    print("  Experiment 2: CCF AIOps Conflict-Scenario Isolation")
    print("=" * 70)

    ccf_gt = load_ccf_gt()

    # Parse CCF DualChannel + TVDiag + LocaleXpert logs
    ccf_dc = {}
    ccf_tv = {}
    ccf_le = {}
    for p in ["0501t","0503t","0505t","0507t","0509t"]:
        for method, store in [("dualchannel", ccf_dc), ("tvdig", ccf_tv), ("localexpert", ccf_le)]:
            lf = f"logs/experiments_ccf_{method}_{p}.log"
            if os.path.exists(lf):
                store.update(parse_full(lf))

    # CCF conflict scenarios
    ccf_scenarios = defaultdict(list)
    for hhmm, rec in ccf_dc.items():
        if hhmm not in ccf_gt: continue
        scenario = rec.get('scenario', 'unknown')
        ccf_scenarios[scenario].append(hhmm)

    print(f"\n  CCF conflict scenario distribution:")
    for sc, evs in sorted(ccf_scenarios.items(), key=lambda x: -len(x[1])):
        print(f"    {sc}: {len(evs)} events")

    # Compute per-scenario: TVDiag vs LocaleXpert (the two extreme methods)
    print(f"\n  Per-scenario A@k (LocaleXpert vs TVDiag on CCF):")
    print(f"  {'Scenario':<30} {'N':>4} {'LE A@1':>7} {'LE A@3':>7} {'LE A@5':>7} | {'TV A@1':>7} {'TV A@3':>7} {'TV A@5':>7}")
    print("  " + "-" * 95)

    for scenario in sorted(ccf_scenarios.keys(), key=lambda s: -len(ccf_scenarios[s])):
        evs = ccf_scenarios[scenario]
        le_hits = {'a1':0,'a3':0,'a5':0}
        tv_hits = {'a1':0,'a3':0,'a5':0}
        n_le = n_tv = 0
        for hhmm in evs:
            gt_val = ccf_gt[hhmm]
            le_preds = ccf_le.get(hhmm, {}).get('preds')
            if le_preds:
                n_le += 1
                h1,h3,h5 = compute_ak(le_preds, gt_val, is_gaia=False)
                le_hits['a1'] += h1; le_hits['a3'] += h3; le_hits['a5'] += h5
            tv_preds = ccf_tv.get(hhmm, {}).get('preds')
            if tv_preds:
                n_tv += 1
                h1,h3,h5 = compute_ak(tv_preds, gt_val, is_gaia=False)
                tv_hits['a1'] += h1; tv_hits['a3'] += h3; tv_hits['a5'] += h5

        le_n = max(n_le, 1)
        tv_n = max(n_tv, 1)
        print(f"  {scenario:<30} {len(evs):>4} "
              f"{le_hits['a1']/le_n:>6.1%} {le_hits['a3']/le_n:>6.1%} {le_hits['a5']/le_n:>6.1%} | "
              f"{tv_hits['a1']/tv_n:>6.1%} {tv_hits['a3']/tv_n:>6.1%} {tv_hits['a5']/tv_n:>6.1%}")

    # ============ Experiment 3: Latency breakdown ============
    print(f"\n{'=' * 70}")
    print("  Experiment 3: Latency Breakdown")
    print("=" * 70)

    for method in ["localexpert", "dualchannel", "tvdig", "hybrid"]:
        lats = []
        for d in ["0704", "0705", "0706"]:
            lf = f"logs/experiments_gaia_{method}_{d}.log"
            if os.path.exists(lf):
                data = parse_full(lf)
                lats.extend([v['latency'] for v in data.values() if v.get('latency')])
        if lats:
            arr = np.array(lats)
            print(f"  {method:<14}: mean={np.mean(arr):.0f}s, median={np.median(arr):.0f}s, "
                  f"p95={np.percentile(arr,95):.0f}s, n={len(arr)}")

    print(f"\n{'=' * 70}")
    print("  ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
