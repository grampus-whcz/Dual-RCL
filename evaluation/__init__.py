"""
Evaluation module for LocaleXpert (Paper [171]) metrics.

Implements the evaluation framework described in Section V of:
  "LLM-Enhanced Failure Localization in Microservices:
   Integrating Multi-Modal Data and Expert Interpretation"

Metrics:
  - Failure Localization: Top-k Accuracy (A@1, A@3, A@5)
  - Reasoning Quality:   BLEU-4, ROUGE-L, G-sim, W-rate
  - Performance:         Per-phase and end-to-end latency

LLM-based evaluation (replacing human engineers):
  - Reference text generation via multi-stage LLM pipeline
  - Multi-judge G-sim for semantic similarity
  - LLM-based W-rate voting system

Usage:
  from evaluation import Evaluator
  ev = Evaluator(log_path="run_case1.log", ground_truth_service="dbservice1")
  report = ev.evaluate()
  print(report.summary())
"""

from evaluation.evaluator import Evaluator, EvaluationReport, CaseResult
from evaluation.metrics import (
    topk_accuracy,
    bleu4_score,
    rouge_l_score,
    gpt_similarity,
    gpt_similarity_multi_judge,
    win_rate,
    mean_latency,
)
from evaluation.llm_client import LLMClient, LLMConfig, create_client
from evaluation.reference_generator import ReferenceGenerator, CaseData
from evaluation.llm_voter import LLMVoter, VoteResult

__all__ = [
    # Core evaluator
    "Evaluator",
    "EvaluationReport",
    "CaseResult",
    # Metrics
    "topk_accuracy",
    "bleu4_score",
    "rouge_l_score",
    "gpt_similarity",
    "gpt_similarity_multi_judge",
    "win_rate",
    "mean_latency",
    # LLM client
    "LLMClient",
    "LLMConfig",
    "create_client",
    # Reference generation
    "ReferenceGenerator",
    "CaseData",
    # LLM voting
    "LLMVoter",
    "VoteResult",
]
