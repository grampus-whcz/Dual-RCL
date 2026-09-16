"""
Dump per-incident RCL logits and node embeddings from a trained TVDiag adapter.

Produces the calibration/audit input for the margin-gated anchor-protection
experiments: for a held-out tail of the *train* split (temperature fitting) and
for the full test split (signal audit), records per incident:
    {split, idx, st_time, root_idx, root_logits, es}

Usage:
    python -m failure_localization.dump_root_logits \
        --checkpoint-dir models/tvdig_checkpoint_repro \
        --data-dir /root/shared-nvme/work/code/RCA/2026/TVDiag/data/gaia \
        --output models/tvdig_checkpoint_repro/root_logits_dump.pkl
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from failure_localization.tvdig_config import TVDiagConfig
from failure_localization.tvdig_data import build_training_dataset
from failure_localization.tvdig_model import MainModel


def infer_dim(state_dict):
    import re
    for key, tensor in state_dict.items():
        if re.search(r"encoders\.\w+\.layers\.0\.fc\.weight$", key):
            return int(tensor.shape[1]) // 2
    return None


def main():
    parser = argparse.ArgumentParser(description="Dump per-incident TVDiag logits/embeddings")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default=None, help="Output pkl (default <ckpt>/root_logits_dump.pkl)")
    parser.add_argument("--train-tail", type=int, default=40,
                        help="Held-out tail of the train split (by datetime) for temperature fitting")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else (args.device if args.device != "auto" else "cpu"))
    output = args.output or os.path.join(args.checkpoint_dir, "root_logits_dump.pkl")

    # Config: adopt the checkpoint's embedding dim from meta.json when present
    dim = 128
    meta_path = os.path.join(args.checkpoint_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        dim = int(meta.get("alert_embedding_dim", dim))
    config = TVDiagConfig(data_dir=args.data_dir, alert_embedding_dim=dim)

    state = torch.load(os.path.join(args.checkpoint_dir, "tvdig.pt"),
                       map_location=device, weights_only=False)
    inferred = infer_dim(state["model"])
    if inferred and inferred != dim:
        print(f"[dump] meta dim {dim} != checkpoint dim {inferred}; using {inferred}")
        config.alert_embedding_dim = inferred

    print(f"[dump] building dataset from {args.data_dir} (dim={config.alert_embedding_dim}) ...")
    train_data, _, test_data, _ = build_training_dataset(config)

    # Mirror train_tvdig.py: ft_num is auto-set from the data, and the config must
    # match the checkpoint before constructing the model
    all_ids = [g["failure_type_id"] for g in train_data] + [g["failure_type_id"] for g in test_data]
    config.ft_num = max(all_ids) + 1

    model = MainModel(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # Metadata aligned with the split lists: same row order, same skip rule
    labels_df = pd.read_csv(os.path.join(args.data_dir, "label.csv"))
    labels_df["index"] = labels_df["index"].astype(str)
    with open(os.path.join(args.data_dir, "raw", "nodes.json")) as f:
        nodes_json = json.load(f)
    all_nodes = sorted({n for sublist in nodes_json.values() for n in sublist})
    node2idx = {n: i for i, n in enumerate(all_nodes)}

    meta_rows = {"train": [], "test": []}
    for _, row in labels_df.iterrows():
        if row["instance"] not in node2idx:
            continue
        split = "train" if row["data_type"] == "train" else "test"
        try:
            st = pd.to_datetime(row["datetime"])
            st_time = st.strftime("%H%M")
        except Exception:
            st_time = str(row["datetime"])
        meta_rows[split].append({"idx": str(row["index"]), "st_time": st_time,
                                 "st_dt": str(row.get("datetime", ""))})

    # Temperature-fitting slice: temporal tail of train
    train_sorted = sorted(meta_rows["train"], key=lambda r: r["st_dt"])
    tail_idx = {r["idx"] for r in train_sorted[-args.train_tail:]}

    records = []
    for split, data in (("train", train_data), ("test", test_data)):
        metas = meta_rows[split]
        assert len(metas) == len(data), f"{split}: metadata {len(metas)} != graphs {len(data)}"
        for g, m in zip(data, metas):
            if split == "train" and m["idx"] not in tail_idx:
                continue
            edge_index = g["edge_index"].to(device)
            features = {mod: t.to(device) for mod, t in g["features"].items()}
            with torch.no_grad():
                _, es, rl, tl = model(edge_index, [g["num_nodes"]], features)
            records.append({
                "split": split,
                "idx": m["idx"],
                "st_time": m["st_time"],
                "st_dt": m["st_dt"],
                "root_idx": int(np.argmax(g["root"].tolist())),
                "root_logits": rl.flatten().float().cpu().numpy(),
                "type_logits": tl.flatten().float().cpu().numpy(),
                "es": {mod: e.float().cpu().numpy() for mod, e in es.items()},
            })

    with open(output, "wb") as f:
        import pickle
        pickle.dump(records, f)
    n_train = sum(1 for r in records if r["split"] == "train")
    print(f"[dump] wrote {len(records)} records ({n_train} train-tail, {len(records)-n_train} test) -> {output}")


if __name__ == "__main__":
    main()
