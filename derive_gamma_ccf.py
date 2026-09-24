"""
Re-derive the REAL univariate gamma vector per logged CCF case (R2 path A).

Mirrors run.py's Phase 4-5 causal chain exactly:
  load (CCF normalization) -> n_init -> run_SPOT(q=1e-3, d=18) -> get_eta
  -> run_pcmci(pc_alpha=0.05) -> get_links(alpha_level=0.05)
  -> get_Q_matrix_part_corr(rho=0.2) -> randomwalk_metric(1000, frontend[0],
  teleportation_prob=0, walk_step=15) -> get_gamma(lambda_param=0.5)
Fallback (dual-channel failure): gamma = |eta|  (mirrors run.py line 1178).

Output: replay_ccf_gamma.pkl  {inner_key: {"gamma": [...], "mode": "walk"|"fallback"}}
Resume-safe.
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJ = os.path.dirname(os.path.abspath(__file__))
GAMMA_PATH = os.path.join(PROJ, "replay_ccf_gamma.pkl")

from sweep_fusion_weights_ccf import load_cases  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N cases")
    args = ap.parse_args()

    os.chdir(PROJ)
    sys.path.insert(0, PROJ)
    from micro import (run_SPOT, get_eta, run_pcmci, get_Q_matrix_part_corr,
                       randomwalk_metric, get_gamma)
    from util_funcs.loaddata import load
    import contextlib
    import io
    import run as run_mod

    cases = load_cases()
    keys = sorted(k for k, c in cases.items() if c.get("tvdig"))
    if args.limit:
        keys = keys[: args.limit]

    cache = {}
    if os.path.exists(GAMMA_PATH):
        with open(GAMMA_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"[gamma] resume: {len(cache)} cached")

    with open("replay_ccf_data_heads.pkl", "rb") as f:
        heads = pickle.load(f)

    todo = [k for k in keys if cache.get(k) is None]
    print(f"[gamma] {len(todo)} to derive / {len(keys)} total")

    import time
    t_all = time.time()
    for i, key in enumerate(todo):
        pref, inner = key.split("|", 1)
        dir_pref = pref.replace("_anchorfix", "")
        hhmm = inner[-4:]
        try:
            mc_file = run_mod.find_closest_file(os.listdir(f"{dir_pref}_microcause"),
                                                f"{hhmm[:2]}-{hhmm[2:]}")
            with contextlib.redirect_stdout(io.StringIO()):
                dataa, data_head = load(f"{dir_pref}_microcause/{mc_file}", normalize=False,
                                        zero_fill_method="prevlatter", aggre_delta=1, verbose=False)
            means = np.mean(dataa, axis=0, keepdims=True)
            stds = np.std(dataa, axis=0, keepdims=True)
            keep = (stds > 0).flatten()
            data_head = [data_head[j] for j in range(len(data_head)) if keep[j]]
            dataa = dataa[:, keep]
            dataa = (dataa - means[:, keep]) / stds[:, keep]
            dataa = np.nan_to_num(dataa, nan=0.0, posinf=0.0, neginf=0.0)
            dataa = np.clip(dataa, -10.0, 10.0)
            data_head = list(heads.get(key) or data_head)

            n_init = int(0.5 * len(dataa))
            frontend = [1]
            t0 = time.time()
            SPOT_res = run_SPOT(dataa, data_head, q=1e-3, d=18)
            eta, _ = get_eta(dataa, data_head, SPOT_res, n_init)
            pcmci, pcmci_res = run_pcmci(dataa, pc_alpha=0.05, verbosity=0)
            causal_graph = run_mod.get_links(data_head, pcmci, pcmci_res, alpha_level=0.05)
            Q = get_Q_matrix_part_corr(dataa, data_head, frontend, causal_graph, rho=0.2)
            vis_list = randomwalk_metric(Q, 1000, frontend[0], teleportation_prob=0, walk_step=15)
            gamma_uni = get_gamma(data_head, vis_list, eta, lambda_param=0.5)
            cache[key] = {"gamma": np.asarray(gamma_uni).astype(float).tolist(),
                          "mode": "walk", "secs": round(time.time() - t0, 1)}
        except Exception as e:
            # mirror run.py fallback: anomaly-score ranking without the causal walk
            try:
                SPOT_res = run_SPOT(dataa, data_head, q=1e-3, d=18)
                eta, _ = get_eta(dataa, data_head, SPOT_res, n_init)
                cache[key] = {"gamma": np.abs(eta).astype(float).tolist(), "mode": "fallback"}
            except Exception as e2:
                cache[key] = {"gamma": None, "mode": "failed",
                              "err": f"{type(e).__name__}: {e}; fallback: {type(e2).__name__}: {e2}"}
        if (i + 1) % 20 == 0:
            with open(GAMMA_PATH, "wb") as f:
                pickle.dump(cache, f)
            el = time.time() - t_all
            print(f"[gamma] {i+1}/{len(todo)} done ({el/60:.1f} min, "
                  f"{el/(i+1):.1f} s/case ETA {el/(i+1)*(len(todo)-i-1)/60:.0f} min)")
    with open(GAMMA_PATH, "wb") as f:
        pickle.dump(cache, f)
    n_ok = sum(1 for v in cache.values() if v and v.get("gamma"))
    print(f"[gamma] finished: {n_ok} valid / {len(cache)}")


if __name__ == "__main__":
    main()
