"""
Anomaly Conflict Resolver – LLM-based arbitration between dual-channel
(univariate + multivariate) root cause analysis results.

Design principle:
  Both univariate and multivariate detection run their own COMPLETE RCA
  pipeline (SPOT eta → random walk, and multi eta → random walk).  The
  resolver compares the two resulting ROOT-CAUSE RANKINGS (not just the
  anomaly detection outputs) and applies the following decision framework:

  Scenario 1  Multi=anomaly, Single=normal
              → Trust multivariate RCA (correlation disruption)
              → Root cause from multivariate channel

  Scenario 2  Single=anomaly, Multi=normal
              → Trust multivariate (system is overall normal)
              → Univariate alerts as supplementary evidence

  Scenario 3  Both=anomaly, different root
              → Neither trusted outright
              → MERGE candidate sets, re-run random walk
              → Causal reasoning (PCMCI) is the FINAL arbiter

In all cases the final arbitration is ultimately handed to causal reasoning
(PCMCI + random walk), which considers fault propagation paths and temporal
ordering — the anomaly signal only determines which metrics get attention.
"""

import json
import logging
import os
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger("anomaly_conflict_resolver")


# ---------------------------------------------------------------------------
#  Conflict Scenario Classification
# ---------------------------------------------------------------------------

class ConflictScenario(Enum):
    """Three conflict scenarios from the design specification."""
    CONSISTENT = "consistent"                        # both agree
    MULTI_ANOM_SINGLE_NORMAL = "multi_anom_single_normal"  # Scenario 1
    SINGLE_ANOM_MULTI_NORMAL = "single_anom_multi_normal"  # Scenario 2
    BOTH_ANOM_DIFFERENT_ROOT = "both_anom_different_root"  # Scenario 3


# ---------------------------------------------------------------------------
#  Resolver
# ---------------------------------------------------------------------------

class AnomalyConflictResolver:
    """Use an LLM to compare multivariate vs univariate detection results and
    produce a unified anomaly report following the conflict-resolution strategy.

    Parameters:
        model_name: LLM model identifier (e.g. 'deepseek-r1-0528').
        api_key:    OpenAI-compatible API key.
        base_url:   OpenAI-compatible base URL.
    """

    def __init__(
        self,
        model_name: str = "deepseek-r1-0528",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self._client: Optional[OpenAI] = None  # lazy init

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def resolve(
        self,
        multivariate_results: dict,
        multivariate_root_metrics: Optional[str],
        univariate_anomaly_descriptions: List[str],
        univariate_root_metrics: str,
        data_head: List[str],
    ) -> dict:
        """Compare dual-channel RCA results and produce a unified report.

        Each channel produces its own root-cause ranking via independent
        random walks on the same causal graph, driven by different eta
        vectors (SPOT eta for univariate, multi eta for multivariate).

        Args:
            multivariate_results: dict from ``run_multivariate_detection()``.
            multivariate_root_metrics: root-cause ranking string from
                multivariate RCA channel (multi eta + random walk), e.g.
                "Top 5 root cause metrics is:(1)...(2)...".  May be None
                if multivariate detection was skipped.
            univariate_anomaly_descriptions: list of anomaly description strings
                from CNN-based pattern detection.
            univariate_root_metrics: root-cause ranking string from
                univariate RCA channel (SPOT eta + random walk), e.g.
                "Top 5 root cause metrics is:(1)...(2)..."
            data_head: list of metric names.

        Returns:
            dict with:
                scenario          : ConflictScenario
                is_consistent     : bool
                unified_description : str  (for ChatChain knowledge injection)
                metric_knowledge  : str  (for MetricAnalysis phase)
                root_cause_knowledge : str  (for RootCauseAnalysis phase)
                llm_analysis      : str  (raw LLM response)
        """
        multi_anomalous = multivariate_results.get("is_anomalous", False)
        uni_anomalous = len(univariate_anomaly_descriptions) > 0

        multi_top5 = multivariate_results.get("top5_metrics", [])
        uni_top5 = self._parse_root_metrics(univariate_root_metrics)

        # Also parse the multivariate RCA ranking (if available)
        multi_rca_top5 = (
            self._parse_root_metrics(multivariate_root_metrics)
            if multivariate_root_metrics
            else []
        )

        # --- Classify scenario using BOTH detection AND RCA rankings ---
        scenario = self._classify_scenario(
            multi_anomalous, uni_anomalous, multi_top5, uni_top5,
            multi_rca_top5=multi_rca_top5,
        )

        logger.info(f"Conflict scenario: {scenario.value}")

        # --- Build prompt for LLM arbitration ---
        prompt = self._build_prompt(
            scenario=scenario,
            multi_results=multivariate_results,
            multi_root_metrics=multivariate_root_metrics,
            uni_descriptions=univariate_anomaly_descriptions,
            uni_root_metrics=univariate_root_metrics,
            data_head=data_head,
        )

        # --- Call LLM ---
        llm_response = self._call_llm(prompt)

        # --- Build unified report based on scenario ---
        unified = self._build_unified_report(
            scenario=scenario,
            multi_results=multivariate_results,
            multi_root_metrics=multivariate_root_metrics,
            uni_descriptions=univariate_anomaly_descriptions,
            uni_root_metrics=univariate_root_metrics,
            llm_analysis=llm_response,
        )

        unified["scenario"] = scenario
        unified["is_consistent"] = (scenario == ConflictScenario.CONSISTENT)
        unified["llm_analysis"] = llm_response

        return unified

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_root_metrics(root_str: str) -> List[str]:
        """Extract metric names from strings like 'Top 5 root cause metrics is:(1)metric_a,(2)metric_b.'."""
        import re
        if not root_str:
            return []
        # Split on (N) markers, e.g. (1), (2), ...
        parts = re.split(r'\(\d+\)', root_str)
        # First part is the prefix text, rest are metric names
        metrics = []
        for p in parts[1:]:
            p = p.strip(" ,.")
            if p:
                metrics.append(p)
        return metrics

    @staticmethod
    def _classify_scenario(
        multi_anomalous: bool,
        uni_anomalous: bool,
        multi_top5: List[str],
        uni_top5: List[str],
        multi_rca_top5: Optional[List[str]] = None,
    ) -> ConflictScenario:
        """Classify which conflict scenario we are in.

        Uses BOTH the detection-level results AND the RCA-level root-cause
        rankings.  The RCA rankings are more informative because they
        incorporate causal graph structure via random walk.

        Priority: if both detection methods flag anomalies, we compare
        the RCA rankings (not just detection top-5) to determine if
        they agree on root cause.
        """
        if multi_anomalous and not uni_anomalous:
            return ConflictScenario.MULTI_ANOM_SINGLE_NORMAL
        if uni_anomalous and not multi_anomalous:
            return ConflictScenario.SINGLE_ANOM_MULTI_NORMAL
        if multi_anomalous and uni_anomalous:
            # Both detected anomalies — compare RCA rankings
            # Prefer RCA-level comparison (if available) over detection top-5
            if multi_rca_top5:
                rca_overlap = set(multi_rca_top5) & set(uni_top5)
                if len(rca_overlap) >= 2:
                    return ConflictScenario.CONSISTENT
            else:
                # Fallback to detection-level comparison
                overlap = set(multi_top5) & set(uni_top5)
                if len(overlap) >= 2:
                    return ConflictScenario.CONSISTENT
            return ConflictScenario.BOTH_ANOM_DIFFERENT_ROOT
        # Neither detected anomaly
        return ConflictScenario.CONSISTENT

    def _build_prompt(
        self,
        scenario: ConflictScenario,
        multi_results: dict,
        multi_root_metrics: Optional[str],
        uni_descriptions: List[str],
        uni_root_metrics: str,
        data_head: List[str],
    ) -> str:
        """Construct the LLM prompt for conflict analysis.

        Now presents TWO complete root-cause rankings (from dual-channel RCA)
        rather than just anomaly detection results.
        """

        multi_desc = multi_results.get("description", "No multivariate results.")
        multi_top5 = ", ".join(multi_results.get("top5_metrics", [])[:5])
        uni_desc = "\n".join(
            f"  - {d}" for d in uni_descriptions[:10]
        ) if uni_descriptions else "  (No univariate anomalies detected)"

        multi_rca_text = (
            multi_root_metrics
            if multi_root_metrics
            else "(Multivariate RCA not available)"
        )

        scenario_instruction = {
            ConflictScenario.CONSISTENT:
                "Both multivariate and univariate RCA channels AGREE on root cause. "
                "Please synthesize a unified anomaly summary combining both perspectives. "
                "The root-cause ranking is reliable — both channels converge.",

            ConflictScenario.MULTI_ANOM_SINGLE_NORMAL:
                "CONFLICT (Scenario 1): Multivariate RCA detected anomalies but "
                "univariate RCA did NOT. This likely indicates a 'correlation disruption' "
                "(variable inter-dependencies broken while individual metrics stay in range). "
                "Per our strategy: TRUST the multivariate RCA ranking for root cause. "
                "The multivariate channel captures hidden dependency breaks that "
                "per-metric thresholds cannot detect. "
                "Investigate which correlation relationships were disrupted.",

            ConflictScenario.SINGLE_ANOM_MULTI_NORMAL:
                "CONFLICT (Scenario 2): Univariate RCA detected anomalies but "
                "multivariate RCA did NOT. This likely indicates isolated noise or a "
                "single-point spike that does not affect the system's joint state. "
                "Per our strategy: TRUST the multivariate result (system is overall normal), "
                "but CHECK whether the flagged single-variable metrics are CRITICAL "
                "(e.g. error rate spikes, availability drops). If critical, flag them as "
                "supplementary evidence that warrants manual investigation.",

            ConflictScenario.BOTH_ANOM_DIFFERENT_ROOT:
                "CONFLICT (Scenario 3): Both RCA channels detected anomalies but point to "
                "DIFFERENT root causes. Per our strategy: do NOT simply pick one. "
                "Instead, MERGE the candidate sets from both channels and note that the "
                "final root-cause determination will be delegated to CAUSAL REASONING "
                "(PCMCI + random walk on merged candidates). "
                "The causal graph considers fault propagation paths and temporal ordering "
                "to determine which metric is the TRUE root cause vs. a downstream symptom. "
                "Please provide your analysis of which candidates are more likely to be "
                "the true root cause based on causal reasoning principles.",
        }.get(scenario, "")

        prompt = f"""You are an expert AIOps anomaly analyst. Your task is to compare and reconcile
results from TWO INDEPENDENT root-cause analysis channels for a microservice system.

## Channel A: Univariate RCA (CNN Pattern Detection + SPOT + Causal Random Walk)
This channel analyses EACH metric independently for anomaly detection, then feeds
per-metric anomaly scores (SPOT eta) into a causal random walk on the PCMCI causal graph.

Anomaly descriptions:
{uni_desc}

Root-cause ranking (univariate RCA):
{uni_root_metrics}

## Channel B: Multivariate RCA (TranAD + Causal Random Walk)
This channel analyses ALL metrics JOINTLY to detect system-level anomalies including
correlation disruptions, then feeds joint anomaly attribution scores (multivariate eta)
into the SAME causal random walk on the PCMCI causal graph.

Detection results:
{multi_desc}

Top-5 root-cause metrics (multivariate detection attribution): {multi_top5}

Root-cause ranking (multivariate RCA):
{multi_rca_text}

## All Monitored Metrics
{', '.join(data_head[:20])}{'...' if len(data_head) > 20 else ''}

## Conflict Decision Framework
Priority rules:
  - Multivariate detection is the BASE for "is the system anomalous?"
    (captures correlation disruptions that univariate misses)
  - Univariate detection SUPPLEMENTS with "which specific metric?"
    (provides interpretable per-metric anomaly signals for extreme single-point faults)
  - Causal reasoning (PCMCI causal graph + random walk) is the FINAL ARBITER
    (uses fault propagation paths and temporal ordering)

## Your Task
{scenario_instruction}

Please provide your analysis in the following JSON format:
{{
  "consistency_assessment": "consistent|conflict_scenario_1|conflict_scenario_2|conflict_scenario_3",
  "trusted_channel": "multivariate_rca|univariate_rca|both_merged",
  "anomaly_summary": "A concise natural-language summary of the anomaly state",
  "root_cause_candidates": ["metric1", "metric2", ...],
  "affected_services": ["service1", "service2", ...],
  "reasoning": "Your step-by-step reasoning considering both channels and causal principles",
  "supplementary_evidence": ["Any additional observations from the non-trusted channel"],
  "correlation_disruption_analysis": "Analysis of whether correlation disruption is likely, referencing specific metric pairs"
}}
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the response text."""
        try:
            logger.info(f"[LLM] Calling model={self.model_name}, "
                        f"prompt length={len(prompt)} chars...")
            t_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert AIOps anomaly analyst specializing in "
                            "microservice root-cause analysis. You are adept at reconciling "
                            "conflicting evidence from multiple anomaly detection methods. "
                            "Always respond with valid JSON as requested."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            elapsed = time.time() - t_start
            content = response.choices[0].message.content.strip()
            usage = response.usage
            logger.info(f"[LLM] Response received in {elapsed:.1f}s, "
                        f"response length={len(content)} chars, "
                        f"tokens: prompt={usage.prompt_tokens if usage else '?'}, "
                        f"completion={usage.completion_tokens if usage else '?'}")
            return content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return json.dumps({
                "consistency_assessment": "error",
                "trusted_channel": "multivariate_rca",
                "anomaly_summary": f"LLM analysis failed: {e}. Defaulting to multivariate result.",
                "root_cause_candidates": [],
                "affected_services": [],
                "reasoning": "Fallback due to LLM error",
                "supplementary_evidence": [],
                "correlation_disruption_analysis": "N/A (LLM error)",
            })

    def _build_unified_report(
        self,
        scenario: ConflictScenario,
        multi_results: dict,
        multi_root_metrics: Optional[str],
        uni_descriptions: List[str],
        uni_root_metrics: str,
        llm_analysis: str,
    ) -> dict:
        """Build the unified knowledge strings for downstream ChatChain phases.

        Now considers both RCA channel rankings in the conflict resolution note.
        """

        multi_desc = multi_results.get("description", "")
        multi_top5 = multi_results.get("top5_metrics", [])
        multi_top5_svc = multi_results.get("top5_services", [])

        # Parse LLM JSON response (best-effort)
        llm_json = self._safe_parse_json(llm_analysis)

        # Parse multivariate RCA ranking (if available)
        multi_rca_metrics = (
            self._parse_root_metrics(multi_root_metrics)
            if multi_root_metrics
            else []
        )

        # --- Determine trusted root-cause candidates ---
        if scenario == ConflictScenario.CONSISTENT:
            # Both RCA channels agree: merge descriptions
            trusted_metrics = multi_top5
            trusted_root = uni_root_metrics

        elif scenario == ConflictScenario.MULTI_ANOM_SINGLE_NORMAL:
            # Trust multivariate RCA (correlation disruption)
            # Use multivariate RCA ranking as primary
            if multi_rca_metrics:
                trusted_metrics = multi_rca_metrics[:5]
            else:
                trusted_metrics = multi_top5
            trusted_root = "Top 5 root cause metrics is:" + ",".join(
                f"({i + 1}){m}" for i, m in enumerate(trusted_metrics[:5])
            ) + "."
            trusted_root += (
                " [Conflict Resolution: Univariate RCA did not flag anomalies, but "
                "multivariate RCA detected correlation disruption. Trusting multivariate "
                "root-cause ranking per decision framework.]"
            )

        elif scenario == ConflictScenario.SINGLE_ANOM_MULTI_NORMAL:
            # Trust multivariate (system normal), flag univariate as supplementary
            if multi_rca_metrics:
                trusted_metrics = multi_rca_metrics[:5]
            else:
                trusted_metrics = multi_top5 if multi_top5 else ["(none)"]
            trusted_root = "Top 5 root cause metrics is:" + ",".join(
                f"({i + 1}){m}" for i, m in enumerate(trusted_metrics[:5])
            ) + "."
            trusted_root += (
                " [Conflict Resolution: Multivariate RCA indicates system is normal. "
                "Univariate flags may be noise or isolated spikes. "
                "Retained as supplementary evidence for final LLM reasoning.]"
            )

        elif scenario == ConflictScenario.BOTH_ANOM_DIFFERENT_ROOT:
            # Merge both RCA candidate sets — causal reasoning will arbitrate
            merged = list(multi_rca_metrics) if multi_rca_metrics else list(multi_top5)
            uni_metrics = self._parse_root_metrics(uni_root_metrics)
            for m in uni_metrics:
                if m not in merged:
                    merged.append(m)
            trusted_metrics = merged[:10]  # up to 10 candidates
            trusted_root = "Top root cause candidates (merged from BOTH RCA channels):" + ",".join(
                f"({i + 1}){m}" for i, m in enumerate(trusted_metrics[:10])
            ) + "."
            trusted_root += (
                " [Conflict Resolution: Multivariate RCA and univariate RCA disagree on "
                "root cause. Both candidate sets MERGED. Final determination delegated to "
                "CAUSAL REASONING (PCMCI + random walk on merged candidates with boosted eta).]"
            )
        else:
            trusted_metrics = multi_top5
            trusted_root = uni_root_metrics

        # --- Build unified anomaly description ---
        anomaly_parts = []
        anomaly_parts.append("[Channel B: Multivariate RCA Detection]")
        anomaly_parts.append(multi_desc)
        if multi_root_metrics:
            anomaly_parts.append(f"Multivariate RCA ranking: {multi_root_metrics}")

        if uni_descriptions:
            anomaly_parts.append("\n[Channel A: Univariate RCA Detection]")
            for d in uni_descriptions[:10]:
                anomaly_parts.append(f"  {d}")
        anomaly_parts.append(f"Univariate RCA ranking: {uni_root_metrics}")

        # Add conflict resolution note
        if scenario != ConflictScenario.CONSISTENT:
            conflict_note = {
                ConflictScenario.MULTI_ANOM_SINGLE_NORMAL:
                    "\n[Conflict Resolution Decision]: "
                    "Scenario 1 — Multivariate RCA detected anomaly, univariate RCA did not. "
                    "This indicates correlation disruption (hidden dependency break). "
                    "TRUSTING multivariate RCA ranking. "
                    "Root cause from Channel B (multivariate random walk).",
                ConflictScenario.SINGLE_ANOM_MULTI_NORMAL:
                    "\n[Conflict Resolution Decision]: "
                    "Scenario 2 — Univariate RCA detected anomaly, multivariate RCA did not. "
                    "System joint state is normal; likely isolated noise. "
                    "TRUSTING multivariate assessment. "
                    "Univariate alerts retained as supplementary evidence for downstream reasoning.",
                ConflictScenario.BOTH_ANOM_DIFFERENT_ROOT:
                    "\n[Conflict Resolution Decision]: "
                    "Scenario 3 — Both RCA channels detected anomalies but DISAGREE on root cause. "
                    "Neither is trusted outright. Candidate sets MERGED. "
                    "Causal reasoning (PCMCI + random walk with boosted merged eta) is the "
                    "FINAL ARBITER — it considers fault propagation paths and temporal ordering "
                    "to determine the TRUE root cause vs. downstream symptoms.",
            }.get(scenario, "")
            anomaly_parts.append(conflict_note)

        # Add LLM reasoning if available
        llm_reasoning = llm_json.get("reasoning", "")
        if llm_reasoning:
            anomaly_parts.append(f"\n[LLM Analysis]: {llm_reasoning}")

        # Add correlation disruption analysis if available
        corr_analysis = llm_json.get("correlation_disruption_analysis", "")
        if corr_analysis:
            anomaly_parts.append(f"\n[Correlation Disruption Analysis]: {corr_analysis}")

        unified_description = "\n".join(anomaly_parts)

        return {
            "unified_description": unified_description,
            "metric_knowledge": f"Knowledge: \nAnomaly description:{unified_description}\n{trusted_root}",
            "root_cause_knowledge": trusted_root,
            "trusted_metrics": trusted_metrics,
            "trusted_services": multi_top5_svc,
            "llm_json": llm_json,
        }

    @staticmethod
    def _safe_parse_json(text: str) -> dict:
        """Best-effort JSON extraction from LLM response."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block from markdown
        import re
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding outermost { }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse LLM JSON response, using empty dict")
        return {}
