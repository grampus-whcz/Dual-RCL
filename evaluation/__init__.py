"""
Evaluation module for LocaleXpert (Paper [171]) metrics.

Implements the evaluation framework described in Section V of:
  "LLM-Enhanced Failure Localization in Microservices:
   Integrating Multi-Modal Data and Expert Interpretation"

Metrics:
  - Failure Localization: Top-k Accuracy (A@1, A@3, A@5)
  - Reasoning Quality:   BLEU-4, ROUGE-L, G-sim, W-rate
  - Performance:         Per-phase and end-to-end latency

Usage:
  # Parse a run.py log and compute metrics:
  from evaluation import Evaluator
  ev = Evaluator(log_path="run_case1.log", ground_truth_service="dbservice1")
  report = ev.evaluate()
  print(report.summary())
"""

from evaluation.evaluator import Evaluator
from evaluation.metrics import (
    topk_accuracy,
    bleu4_score,
    rouge_l_score,
    gpt_similarity,
    mean_latency,
)

__all__ = [
    "Evaluator",
    "topk_accuracy",
    "bleu4_score",
    "rouge_l_score",
    "gpt_similarity",
    "mean_latency",
]
