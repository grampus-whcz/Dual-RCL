"""
v4 feasibility probe: measure how many runtime event tokens (from the SoC-RCA
online extractors) exist in the upstream TVDiag encoder vocabulary.

Gate (from the C3 plan): proceed with content-aware inference only if
trace >= 60% and metric >= 40% of tokens are in-vocabulary after normalization.
Log is known-dead (hash ids) and excluded from v4 scope.
"""

import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJ = os.path.dirname(os.path.abspath(__file__))
TVDIG_DATA = "/root/shared-nvme/work/code/RCA/2026/TVDiag/data/gaia_v1_idf"

N_CASES = 10


def main():
    from failure_localization import create_localizer as create_rca_localizer
    from failure_localization.tvdig_data import (
        extract_metric_events_from_univariate,
        extract_trace_events_from_soCRCA,
        extract_log_events_from_soCRCA,
    )
    from util_funcs.loaddata import load
    import run as run_mod

    encoders = {}
    for mod in ("metric", "trace", "log"):
        with open(os.path.join(TVDIG_DATA, "tmp", f"{mod}-encoder.pkl"), "rb") as f:
            encoders[mod] = pickle.load(f)

    # collect the first N_CASES logged cases
    cases = {}
    for path in [f"logs/experiments_gaia_hybrid_{d}.log" for d in ("0704", "0705", "0706")]:
        p = os.path.join(PROJ, path)
        if not os.path.exists(p):
            continue
        cur = None
        for line in open(p, errors="ignore"):
            m = __import__("re").search(
                r'"type": "task_parsed", "date": "(\w+)", "time": "(\d{2})-(\d{2})"', line.strip())
            if m:
                cur = f"{m.group(1)}|{m.group(2)}{m.group(3)}"
                cases.setdefault(cur, True)
                if len(cases) >= N_CASES:
                    break
        if len(cases) >= N_CASES:
            break

    stats = {mod: {"hits": 0, "total": 0, "docs": 0, "empty_docs": 0} for mod in ("metric", "trace", "log")}
    node_names = None

    for ci, key in enumerate(sorted(cases)):
        date, hhmm = key.split("|")
        time_result = f"{hhmm[:2]}-{hhmm[2:]}"
        try:
            mc_file = run_mod.find_closest_file(os.listdir(f"{date}_microcause"), time_result)
            dataa, data_head = load(f"{date}_microcause/{mc_file}", normalize=False,
                                    zero_fill_method="prevlatter", aggre_delta=1, verbose=False)
            tr_file = run_mod.find_closest_file(os.listdir(f"{date}_tracerca"), time_result)
            log_file = run_mod.find_closest_file(os.listdir(f"{date}_log_fault"), time_result)
            n_init = int(0.5 * len(dataa))
            node_names = None  # extractor derives from data_head/nodes
            # mirror the localizer: NODE_NAMES from the GAIA config
            from failure_localization.tvdig_config import TVDiagConfig
            node_names = TVDiagConfig.NODE_NAMES

            m_events = extract_metric_events_from_univariate(dataa, data_head, n_init, node_names)
            t_events = extract_trace_events_from_soCRCA(f"./{date}_tracerca/{tr_file}", node_names)
            l_events = extract_log_events_from_soCRCA(f"{date}_log_fault/{log_file}", [], node_names)

            # upstream-style docs per node (gaia config: trace_op=True,
            # trace_ab_type=True, metric_direction=True)
            for mod, events, sep in (("metric", m_events, True), ("trace", t_events, True), ("log", l_events, True)):
                ev = encoders[mod]["event_dic"]
                for node_doc_events in events:
                    doc = ["&".join(e) for e in node_doc_events]
                    tokens = set(" ".join(doc).split(" ")) if doc else set()
                    if not tokens:
                        stats[mod]["empty_docs"] += 1
                        continue
                    stats[mod]["docs"] += 1
                    for tok in tokens:
                        stats[mod]["total"] += 1
                        if tok in ev:
                            stats[mod]["hits"] += 1
        except Exception as e:
            print(f"[probe] {key} failed: {type(e).__name__}: {e}")

    print(f"\n== v4 词表命中率探针 ({N_CASES} 例) ==")
    for mod in ("metric", "trace", "log"):
        s = stats[mod]
        rate = s["hits"] / max(s["total"], 1)
        print(f"{mod:<8} docs={s['docs']:<5} tokens={s['total']:<8} in-vocab={rate:.1%}  "
              f"空文档数={s['empty_docs']}")
    verdict_trace = stats["trace"]["hits"] / max(stats["trace"]["total"], 1) >= 0.60
    verdict_metric = stats["metric"]["hits"] / max(stats["metric"]["total"], 1) >= 0.40
    print(f"\n判定: trace {'PASS' if verdict_trace else 'FAIL'} (≥60%), "
          f"metric {'PASS' if verdict_metric else 'FAIL'} (≥40%) → "
          f"{'v4 可行' if (verdict_trace and verdict_metric) else 'v4 不可行(按计划放弃)'}")


if __name__ == "__main__":
    main()
