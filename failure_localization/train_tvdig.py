"""
Standalone training script for TVDiag on GAIA data.

Usage:
    python -m failure_localization.train_tvdig \
        --data-dir /root/shared-nvme/work/code/RCA/2026/TVDiag/data/gaia \
        --output-dir ./tvdig_checkpoint \
        --epochs 500
"""

import argparse
import logging
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from failure_localization.tvdig_config import TVDiagConfig
from failure_localization.tvdig_data import build_training_dataset, save_embedding_cache
from failure_localization.tvdig_model import TVDiagTrainer


def main():
    parser = argparse.ArgumentParser(description="Train TVDiag on GAIA data")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to TVDiag GAIA data directory")
    parser.add_argument("--output-dir", type=str, default="./tvdig_checkpoint", help="Output directory")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    config = TVDiagConfig(
        data_dir=args.data_dir,
        model_checkpoint_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Build dataset
    logger = logging.getLogger("train_tvdig")
    logger.info("Building dataset...")
    train_data, aug_data, test_data, embedding_cache = build_training_dataset(config)

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


if __name__ == "__main__":
    main()
