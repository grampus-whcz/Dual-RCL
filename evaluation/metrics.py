"""
Core metric computation functions for LocaleXpert evaluation.

All metrics follow the definitions in Paper [171] Section V-A4:
  - A@k      : Top-k Accuracy (k=1,3,5)
  - BLEU-4   : 4-gram precision
  - ROUGE-L  : Longest Common Subsequence based recall/F1
  - G-sim    : LLM-as-judge semantic similarity (0-1)
  - W-rate   : Human evaluation win rate
  - Latency  : Average per-case / per-phase time in seconds
"""

from __future__ import annotations

import re
import os
from collections import Counter
from typing import List, Optional

import numpy as np


# =====================================================================
# 1. Failure Localization: Top-k Accuracy (A@k)
# =====================================================================

def topk_accuracy(
    predictions: List[str],
    ground_truth: str | List[str],
    k: int = 5,
) -> float:
    """Compute Top-k Accuracy (A@k) as defined in Paper [171].

    A@k = 1 if the ground truth is among the top-k predictions, else 0.

    Args:
        predictions: Ordered list of predicted root cause candidates
                     (service names or metric names), ranked by confidence.
        ground_truth: Ground truth root cause (string or list of strings).
                      If a list, the metric is 1 if ANY ground truth item
                      appears in the top-k.
        k: Number of top predictions to consider.

    Returns:
        1.0 if hit, 0.0 otherwise.
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    top_k = predictions[:k]
    for gt in ground_truth:
        # Normalize: strip whitespace, lowercase for comparison
        gt_norm = gt.strip().lower()
        for pred in top_k:
            if gt_norm in pred.strip().lower() or pred.strip().lower() in gt_norm:
                return 1.0
    return 0.0


def topk_accuracy_batch(
    cases: List[dict],
    k_values: List[int] = (1, 3, 5),
) -> dict:
    """Compute A@k over a batch of cases.

    Args:
        cases: List of dicts, each with:
            - 'predictions': list of ranked predictions
            - 'ground_truth': string or list of strings
        k_values: Which k values to compute.

    Returns:
        dict mapping k -> accuracy (mean over batch).
    """
    results = {}
    for k in k_values:
        scores = [
            topk_accuracy(c['predictions'], c['ground_truth'], k=k)
            for c in cases
        ]
        results[f'A@{k}'] = float(np.mean(scores))
    return results


# =====================================================================
# 2. Reasoning Quality: BLEU-4
# =====================================================================

def _ngrams(tokens: List[str], n: int) -> Counter:
    """Return Counter of n-grams from token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu4_score(
    hypothesis: str,
    reference: str,
) -> float:
    """Compute BLEU-4 score between hypothesis and reference text.

    Follows the standard BLEU-4 definition [40]:
      BLEU-4 = BP * exp(sum(log(p_n)) / 4)
    where p_n is the modified n-gram precision for n=1..4,
    BP is the brevity penalty.

    Args:
        hypothesis: Generated text.
        reference: Reference (ground truth) text.

    Returns:
        BLEU-4 score in [0, 1].
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    # Modified n-gram precision for n=1..4
    log_precisions = []
    for n in range(1, 5):
        hyp_ngrams = _ngrams(hyp_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)

        if len(hyp_ngrams) == 0:
            log_precisions.append(-np.inf)
            continue

        clipped = 0
        total = 0
        for ng, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ng, 0))
            total += count

        precision = clipped / total if total > 0 else 0.0
        log_precisions.append(np.log(precision) if precision > 0 else -np.inf)

    avg_log = np.mean(log_precisions)
    if np.isinf(avg_log):
        return 0.0

    return float(bp * np.exp(avg_log))


def bleu4_batch(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Average BLEU-4 over a batch."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have same length")
    scores = [bleu4_score(h, r) for h, r in zip(hypotheses, references)]
    return float(np.mean(scores))


# =====================================================================
# 3. Reasoning Quality: ROUGE-L
# =====================================================================

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l_score(
    hypothesis: str,
    reference: str,
) -> float:
    """Compute ROUGE-L F1 score between hypothesis and reference.

    Follows the standard ROUGE-L definition [41]:
      R_lcs = LCS(hyp, ref) / |ref|
      P_lcs = LCS(hyp, ref) / |hyp|
      F1 = (1 + beta^2) * R * P / (R + beta^2 * P)  with beta=1.2

    Args:
        hypothesis: Generated text.
        reference: Reference text.

    Returns:
        ROUGE-L F1 score in [0, 1].
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    lcs_len = _lcs_length(hyp_tokens, ref_tokens)

    recall = lcs_len / len(ref_tokens)
    precision = lcs_len / len(hyp_tokens)

    beta = 1.2
    if recall + precision == 0:
        return 0.0

    f1 = (1 + beta**2) * recall * precision / (recall + beta**2 * precision)
    return float(f1)


def rouge_l_batch(
    hypotheses: List[str],
    references: List[str],
) -> float:
    """Average ROUGE-L over a batch."""
    scores = [rouge_l_score(h, r) for h, r in zip(hypotheses, references)]
    return float(np.mean(scores))


# =====================================================================
# 4. Reasoning Quality: G-sim (LLM-as-Judge Similarity)
# =====================================================================

# Shared prompt template for G-sim evaluation
_GSIM_PROMPT_TEMPLATE = """You are an expert evaluator for microservice failure localization reasoning.

Compare the GENERATED reasoning with the REFERENCE reasoning and evaluate their semantic similarity on a scale from 0.0 to 1.0.

Focus on:
1. Whether the same root cause service/metric is identified
2. Whether the reasoning chain is logically consistent
3. Whether key observations are captured
4. Clarity and coherence of the explanation

GENERATED:
{hypothesis}

REFERENCE:
{reference}

Output ONLY a single float number between 0.0 and 1.0 representing the similarity score. Do not output anything else."""


def gpt_similarity(
    hypothesis: str,
    reference: str,
    model_name: str = "glm-4.7",
    api_key: str | None = None,
    base_url: str | None = None,
) -> float:
    """Compute G-sim: LLM-as-judge semantic similarity score (0-1).

    As described in Paper [171], GPT-4 was used to evaluate the clarity
    and coherence of the reasoning outputs. This implementation uses the
    unified LLM client, supporting any configured model (default: glm-4.7).

    The LLM is asked to rate the semantic similarity between the
    hypothesis (generated reasoning) and reference (expert reasoning)
    on a scale from 0 to 1.

    Args:
        hypothesis: Generated reasoning text.
        reference: Reference reasoning text.
        model_name: LLM model to use for judging.
        api_key: API key (auto-detected from defaults if None).
        base_url: API base URL (auto-detected from defaults if None).

    Returns:
        Similarity score in [0, 1].
    """
    from evaluation.llm_client import create_client

    kwargs = {}
    if api_key:
        kwargs['api_key'] = api_key
    if base_url:
        kwargs['base_url'] = base_url

    client = create_client(model_name, **kwargs)

    prompt = _GSIM_PROMPT_TEMPLATE.format(
        hypothesis=hypothesis, reference=reference
    )

    try:
        score_text = client.call(
            system_prompt="You are an expert evaluator. Output only a number.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=50,
        )
        if score_text is None:
            return 0.0

        # Extract the first float from the response
        match = re.search(r'[0-9]*\.?[0-9]+', score_text)
        if match:
            score = float(match.group())
            return min(1.0, max(0.0, score))
        return 0.0
    except Exception as e:
        print(f"[G-sim] Error calling LLM: {e}")
        return 0.0


def gpt_similarity_batch(
    hypotheses: List[str],
    references: List[str],
    model_name: str = "glm-4.7",
    api_key: str | None = None,
    base_url: str | None = None,
) -> float:
    """Average G-sim over a batch."""
    scores = []
    for h, r in zip(hypotheses, references):
        s = gpt_similarity(h, r, model_name, api_key, base_url)
        scores.append(s)
    return float(np.mean(scores))


def gpt_similarity_multi_judge(
    hypothesis: str,
    reference: str,
    judge_models: List[str] = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    """Compute G-sim using multiple LLM judges and return per-judge scores.

    Each judge independently scores the similarity. This provides a more
    robust evaluation, similar to having multiple human annotators.

    Args:
        hypothesis: Generated reasoning text.
        reference: Reference reasoning text.
        judge_models: List of model names to use as judges.
        api_key: API key (auto-detected if None).
        base_url: API base URL (auto-detected if None).

    Returns:
        dict with:
          - 'scores': {model_name: score}
          - 'mean': average score across judges
    """
    if judge_models is None:
        judge_models = ["glm-4.7"]

    scores = {}
    for model in judge_models:
        score = gpt_similarity(hypothesis, reference, model, api_key, base_url)
        scores[model] = score

    mean_score = float(np.mean(list(scores.values()))) if scores else 0.0
    return {'scores': scores, 'mean': mean_score}


def gpt_similarity_multi_judge_batch(
    hypotheses: List[str],
    references: List[str],
    judge_models: List[str] = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    """Average multi-judge G-sim over a batch.

    Returns:
        dict with:
          - 'per_model_mean': {model_name: mean_score}
          - 'overall_mean': average across all judges and cases
    """
    if judge_models is None:
        judge_models = ["glm-4.7"]

    per_model_scores = {m: [] for m in judge_models}

    for h, r in zip(hypotheses, references):
        result = gpt_similarity_multi_judge(h, r, judge_models, api_key, base_url)
        for model, score in result['scores'].items():
            per_model_scores[model].append(score)

    per_model_mean = {
        m: float(np.mean(scores)) if scores else 0.0
        for m, scores in per_model_scores.items()
    }

    all_scores = [s for scores in per_model_scores.values() for s in scores]
    overall_mean = float(np.mean(all_scores)) if all_scores else 0.0

    return {
        'per_model_mean': per_model_mean,
        'overall_mean': overall_mean,
    }


# =====================================================================
# 5. Win Rate (W-rate): Human Evaluation
# =====================================================================

def win_rate(
    votes: List[List[str]],
    method_name: str,
) -> float:
    """Compute Win Rate (W-rate) from human evaluation votes.

    As described in Paper [171]: 3 engineers independently vote for the
    best method for each test case. W-rate = proportion of cases where
    the method received majority vote (>= 2 out of 3).

    Args:
        votes: List of cases, each case is a list of voted method names.
               e.g., [['LocaleXpert', 'LocaleXpert', 'ReAct'], ...]
               means for case 1, 2 engineers voted LocaleXpert, 1 voted ReAct.
        method_name: The method to compute win rate for.

    Returns:
        Win rate in [0, 1].
    """
    if not votes:
        return 0.0

    wins = 0
    for case_votes in votes:
        count = sum(1 for v in case_votes if v == method_name)
        if count >= (len(case_votes) + 1) // 2:  # majority
            wins += 1

    return wins / len(votes)


# =====================================================================
# 6. Latency
# =====================================================================

def mean_latency(latencies: List[float]) -> dict:
    """Compute latency statistics.

    Args:
        latencies: List of per-case or per-phase latencies in seconds.

    Returns:
        dict with mean, median, std, min, max.
    """
    arr = np.array(latencies)
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'count': len(arr),
    }
