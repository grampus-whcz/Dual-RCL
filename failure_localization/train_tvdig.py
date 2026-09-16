"""
Standalone training script for TVDiag on GAIA data.

Usage:
    python -m failure_localization.train_tvdig \
        --data-dir /root/shared-nvme/work/code/RCA/2026/TVDiag/data/gaia \
        --output-dir ./tvdig_checkpoint \
        --epochs 500
"""

import argparse
import datetime
import json
import logging
import os
import random
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from failure_localization.tvdig_config import TVDiagConfig
from failure_localization.tvdig_data import build_training_dataset, save_embedding_cache
from failure_localization.tvdig_model import TVDiagTrainer


def set_seed(seed: int):
    """Seed every RNG that influences training (mirrors upstream TVDiag main.py)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train TVDiag on GAIA data")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to TVDiag GAIA data directory")
    parser.add_argument("--output-dir", type=str, default="./tvdig_checkpoint", help="Output directory")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="auto")
    # Regularization (anti-overfit for small datasets like CCF AIOps)
    parser.add_argument("--feat-drop", type=float, default=0.3, help="Feature dropout (default 0.3)")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="L2 weight decay")
    parser.add_argument("--aug-times", type=int, default=20, help="Data augmentation multiplier")
    # Reproducibility / embedding-upgrade experiment support
    parser.add_argument("--seed", type=int, default=2, help="RNG seed (default 2)")
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Node feature dimension produced by the upstream encoder (default 128)")
    parser.add_argument("--cache-train-only", type=int, default=1, choices=[0, 1],
                        help="Average the inference embedding cache over train incidents only (1, strict) "
                             "or over all incidents (0, flagged transductive ablation)")
    parser.add_argument("--variant", type=str, default="", help="Variant tag recorded in meta.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    set_seed(args.seed)

    config = TVDiagConfig(
        data_dir=args.data_dir,
        model_checkpoint_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        feat_drop=args.feat_drop,
        aug_times=args.aug_times,
        alert_embedding_dim=args.embedding_dim,
        cache_train_only=bool(args.cache_train_only),
    )

    # Build dataset
    logger = logging.getLogger("train_tvdig")
    logger.info("Building dataset...")
    train_data, aug_data, test_data, embedding_cache = build_training_dataset(config)

    # Auto-set ft_num to match the number of failure types in the data
    # (GAIA default is 5, but CCF AIOps has 15+ fault types)
    if train_data:
        max_ft_id = max(g["failure_type_id"] for g in train_data)
        if test_data:
            max_ft_id = max(max_ft_id, max(g["failure_type_id"] for g in test_data))
        config.ft_num = max_ft_id + 1
        logger.info(f"Auto-set ft_num={config.ft_num} (max failure_type_id={max_ft_id})")

    # Save embedding cache for inference
    os.makedirs(args.output_dir, exist_ok=True)
    save_embedding_cache(embedding_cache, os.path.join(args.output_dir, "embedding_cache.pkl"))

    # Train
    trainer = TVDiagTrainer(config, log_dir=args.output_dir, device=args.device)
    model = trainer.train(train_data, aug_data)

    # Evaluate on test set
    logger.info("Evaluating on test data...")
    results = trainer.evaluate(test_data, model=model)

    logger.info("=" * 60)
    logger.info("Final Results:")
    for k, v in results["rcl"].items():
        logger.info(f"  RCL {k}: {v:.3%}")
    for k, v in results["fti"].items():
        logger.info(f"  FTI {k}: {v:.3%}")
    logger.info("=" * 60)

    # Persist run metadata next to the checkpoint (variant lineage, seed, results)
    meta = {
        "alert_embedding_dim": config.alert_embedding_dim,
        "data_dir": args.data_dir,
        "variant": args.variant,
        "seed": args.seed,
        "cache_train_only": bool(args.cache_train_only),
        "ft_num": config.ft_num,
        "epochs": config.epochs,
        "feat_drop": config.feat_drop,
        "weight_decay": config.weight_decay,
        "aug_times": config.aug_times,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "results": {
            "rcl": {k: float(v) for k, v in results["rcl"].items()},
            "fti": {k: float(v) for k, v in results["fti"].items()},
        },
    }
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Meta written to {meta_path}")


if __name__ == "__main__":
    main()
