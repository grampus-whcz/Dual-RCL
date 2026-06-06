"""
Reference text generation pipeline for evaluation.

Replaces the paper's "three experienced operations engineers" with a
three-stage LLM pipeline:
  Stage 1: Three independent LLM calls (different temperatures) generate
           candidate reference texts for each expert type.
  Stage 2: A cross-validation call synthesises the best-of.
  Stage 3: A verification call checks correctness against ground truth.

Generated reference texts are cached to disk as JSON files to avoid
repeated API calls.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from evaluation.llm_client import create_client

logger = logging.getLogger(__name__)


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class CaseData:
    """Input data for a single test case, used to generate references.

    Populated from LogParser metadata + [EVAL] records + ground truth pkl.
    """
    case_id: str = ''
    date: str = ''
    time: str = ''
    ground_truth_service: str = ''

    # Multi-modal data extracted from logs
    trace_anomaly_text: str = ''
    root_services: List[str] = field(default_factory=list)
    metric_anomaly_text: str = ''
    root_metrics_text: str = ''
    log_anomaly_text: str = ''
    conflict_resolution: str = ''

    # Additional context
    causal_graph_info: str = ''
    univariate_anomaly_count: int = 0
    multivariate_anomaly: bool = False


# =====================================================================
# Prompt Templates
# =====================================================================

SYSTEM_PROMPT_GENERATE = """You are an experienced operations engineer with \
10+ years of expertise in microservice failure diagnosis. You are tasked with \
writing a reference reasoning text that demonstrates expert-level root cause \
analysis for a microservice system failure.

Your output will be used as the gold-standard reference for evaluating \
automated diagnosis systems. Be thorough, specific, and technically precise."""

SYSTEM_PROMPT_REVIEW = """You are a senior operations architect reviewing \
reference reasoning texts written by three independent engineers. Your job \
is to synthesize the best elements from each into a single, high-quality \
reference text."""

SYSTEM_PROMPT_VERIFY = """You are a quality assurance reviewer for operations \
documentation. Verify that the provided reference reasoning text is correct \
and complete."""


def _build_generation_prompt(expert_type: str, case_data: CaseData) -> str:
    """Build the user prompt for Stage 1 (independent generation)."""

    # Select relevant data for each expert type
    if expert_type == 'TraceExpert':
        data_section = f"""## Trace Data
- MEPFL top-5 predicted root services: {', '.join(case_data.root_services)}
- Trace anomaly summary: {case_data.trace_anomaly_text or 'See system context above.'}"""
    elif expert_type == 'MetricExpert':
        data_section = f"""## Metric Data
- Root cause metrics (univariate): {case_data.root_metrics_text or 'Not available.'}
- Metric anomaly summary: {case_data.metric_anomaly_text or 'See system context above.'}
- Univariate anomaly count: {case_data.univariate_anomaly_count}
- Multivariate anomaly detected: {case_data.multivariate_anomaly}
- Causal graph: {case_data.causal_graph_info or 'Not available.'}"""
    elif expert_type == 'LogExpert':
        data_section = f"""## Log Data
- Log anomaly summary: {case_data.log_anomaly_text or 'See system context above.'}"""
    else:
        data_section = f"""## All Available Data
- Trace anomalies: {case_data.trace_anomaly_text or 'N/A'}
- Metric anomalies: {case_data.metric_anomaly_text or 'N/A'}
- Log anomalies: {case_data.log_anomaly_text or 'N/A'}
- Root metrics: {case_data.root_metrics_text or 'N/A'}"""

    return f"""Given the following multi-modal observability data from a \
production microservice system (GAIA platform), write a reference reasoning \
text for the **{expert_type}**.

## System Context
- Fault injection time: {case_data.date} {case_data.time}
- Ground truth root cause service: **{case_data.ground_truth_service}**
- Predicted root services (ranked): {', '.join(case_data.root_services[:5]) if case_data.root_services else 'N/A'}
- Conflict resolution: {case_data.conflict_resolution or 'N/A'}

{data_section}

## Requirements
1. Structure your output exactly as:
   **Observation:** [List the key anomalies and abnormal signals you observe]
   **Reasoning:** [Provide a logical chain explaining why the root cause is the identified service]
   **Final Answer:** [State the root cause service name clearly]

2. Be specific — reference service names, metric names, error patterns, timestamps where applicable.
3. The reasoning must correctly identify **{case_data.ground_truth_service}** as the root cause.
4. Use professional, technical language appropriate for a senior operations engineer.
5. Keep the text concise but thorough (200-500 words).
"""


def _build_review_prompt(
    expert_type: str,
    texts: List[str],
    ground_truth: str,
) -> str:
    """Build the user prompt for Stage 2 (cross-validation)."""
    numbered = '\n\n'.join(
        f'### Text {i+1} (from Engineer {i+1}):\n{text}'
        for i, text in enumerate(texts)
    )

    return f"""Review the following three independently written reference \
reasoning texts for the **{expert_type}**, and synthesize them into a single, \
high-quality reference text.

{numbered}

## Synthesis Guidelines
1. Keep the most specific and accurate observations from each text.
2. Merge overlapping reasoning chains, keeping the most rigorous argument.
3. Ensure the final answer correctly identifies **{ground_truth}** as the root cause.
4. Remove any hallucinated or speculative data not grounded in the source material.
5. Maintain the structure: **Observation:** ... **Reasoning:** ... **Final Answer:** ...

Output ONLY the synthesized reference text, nothing else."""


def _build_verify_prompt(
    expert_type: str,
    text: str,
    ground_truth: str,
) -> str:
    """Build the user prompt for Stage 3 (verification)."""
    return f"""Verify the following reference reasoning text for correctness.

## Expert Type: {expert_type}
## Ground Truth Root Cause: {ground_truth}

## Reference Text to Verify:
{text}

## Verification Checklist
1. Does the Final Answer correctly identify **{ground_truth}** as the root cause?
2. Are the observations factual and not hallucinated?
3. Is the reasoning chain logically sound?
4. Is the text well-structured and clearly written?

Output a JSON object with the following format:
```json
{{
    "correct": true/false,
    "issues": ["list of any issues found"],
    "corrected_text": "the corrected text if issues exist, otherwise the original text"
}}
```"""


# =====================================================================
# Reference Generator
# =====================================================================

class ReferenceGenerator:
    """Three-stage pipeline for generating reference reasoning texts.

    Stage 1: Three independent LLM calls with different temperatures
             simulate three engineers.
    Stage 2: Cross-validation synthesises the three drafts.
    Stage 3: Verification ensures correctness against ground truth.

    Usage:
        gen = ReferenceGenerator(model_name="glm-4.5")
        refs = gen.generate_for_case(case_data)
        gen.generate_batch(cases, output_dir="evaluation/reference_texts/")
    """

    # Temperature settings simulating three independent engineers
    ENGINEER_TEMPERATURES = [0.1, 0.3, 0.5]

    def __init__(
        self,
        model_name: str = "glm-4.5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        self.model_name = model_name
        self.client = create_client(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )

    # -- Stage 1: Independent Generation ------------------------------------

    def _generate_single(
        self,
        expert_type: str,
        case_data: CaseData,
        temperature: float,
    ) -> str:
        """Generate one candidate reference text."""
        prompt = _build_generation_prompt(expert_type, case_data)
        result = self.client.call(
            SYSTEM_PROMPT_GENERATE, prompt,
            temperature=temperature,
        )
        return result or ''

    # -- Stage 2: Cross-Validation ------------------------------------------

    def _review_texts(
        self,
        expert_type: str,
        texts: List[str],
        ground_truth: str,
    ) -> str:
        """Synthesize multiple drafts into one reference."""
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            return ''
        if len(valid_texts) == 1:
            return valid_texts[0]

        prompt = _build_review_prompt(expert_type, valid_texts, ground_truth)
        result = self.client.call(SYSTEM_PROMPT_REVIEW, prompt)
        return result or valid_texts[0]  # fallback to first

    # -- Stage 3: Verification ----------------------------------------------

    def _verify_text(
        self,
        expert_type: str,
        text: str,
        ground_truth: str,
    ) -> str:
        """Verify and optionally correct the reference text."""
        if not text.strip():
            return text

        prompt = _build_verify_prompt(expert_type, text, ground_truth)
        result = self.client.call_json(SYSTEM_PROMPT_VERIFY, prompt)

        if result is None:
            logger.warning("[RefGen] Verification JSON parse failed; "
                           "using unverified text.")
            return text

        if result.get('correct', False):
            return text
        else:
            corrected = result.get('corrected_text', text)
            issues = result.get('issues', [])
            if issues:
                logger.info(f"[RefGen] Verification issues for {expert_type}: "
                            f"{issues}")
            return corrected if corrected.strip() else text

    # -- Public API ---------------------------------------------------------

    def generate_for_case(self, case_data: CaseData) -> dict:
        """Run the full 3-stage pipeline for one test case.

        Returns:
            dict with 'references' (per-expert) and 'raw_generations'.
        """
        expert_types = ['TraceExpert', 'MetricExpert', 'LogExpert']
        references = {}
        raw_generations = {}

        for expert in expert_types:
            logger.info(f"[RefGen] Generating reference for {expert} "
                        f"(case={case_data.case_id})")

            # Stage 1: Three independent generations
            drafts = []
            for i, temp in enumerate(self.ENGINEER_TEMPERATURES):
                draft = self._generate_single(expert, case_data, temp)
                drafts.append(draft)
                raw_generations[f'{expert}_engineer{i+1}'] = draft

            # Stage 2: Cross-validation
            synthesized = self._review_texts(
                expert, drafts, case_data.ground_truth_service
            )

            # Stage 3: Verification
            final_text = self._verify_text(
                expert, synthesized, case_data.ground_truth_service
            )

            references[expert] = final_text

        return {
            'case_id': case_data.case_id,
            'date': case_data.date,
            'time': case_data.time,
            'ground_truth_service': case_data.ground_truth_service,
            'generated_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
            'generation_model': self.model_name,
            'references': references,
            'raw_generations': raw_generations,
        }

    def generate_batch(
        self,
        cases: List[CaseData],
        output_dir: str,
    ) -> Dict[str, dict]:
        """Generate reference texts for a batch of cases.

        Saves each case's references as a separate JSON file and returns
        all results as a dict keyed by case_id.

        Args:
            cases: List of CaseData to generate references for.
            output_dir: Directory to save JSON files.

        Returns:
            Dict mapping case_id -> generated references dict.
        """
        os.makedirs(output_dir, exist_ok=True)
        all_refs = {}

        for i, case in enumerate(cases):
            logger.info(f"[RefGen] Processing case {i+1}/{len(cases)}: "
                        f"{case.case_id}")
            refs = self.generate_for_case(case)

            # Save to disk
            filename = f"{case.case_id}_reference.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(refs, f, indent=2, ensure_ascii=False)

            all_refs[case.case_id] = refs
            logger.info(f"[RefGen] Saved: {filepath}")

        return all_refs

    @staticmethod
    def load_references(ref_dir: str) -> Dict[str, dict]:
        """Load previously generated reference texts from disk.

        Args:
            ref_dir: Directory containing *_reference.json files.

        Returns:
            Dict mapping case_id -> references dict.
        """
        refs = {}
        if not os.path.isdir(ref_dir):
            logger.warning(f"[RefGen] Reference dir not found: {ref_dir}")
            return refs

        for fname in sorted(os.listdir(ref_dir)):
            if not fname.endswith('_reference.json'):
                continue
            filepath = os.path.join(ref_dir, fname)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                case_id = data.get('case_id', fname.replace('_reference.json', ''))
                refs[case_id] = data
            except Exception as e:
                logger.warning(f"[RefGen] Failed to load {filepath}: {e}")

        return refs
