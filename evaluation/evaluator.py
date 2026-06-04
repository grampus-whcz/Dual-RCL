"""
Main Evaluator class that orchestrates metric computation.

Produces evaluation reports matching Paper [171] Tables II-VI:
  - Table II:  Failure Localization Performance (A@1, A@3, A@5)
  - Table III: Trace Expert Reasoning Performance (BLEU-4, ROUGE-L, G-sim, W-rate)
  - Table IV:  Metric Expert Reasoning Performance
  - Table V:   Log Expert Reasoning Performance
  - Table VI:  Average Single-Case Latency and Aggregated Performance
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.metrics import (
    topk_accuracy,
    topk_accuracy_batch,
    bleu4_score,
    bleu4_batch,
    rouge_l_score,
    rouge_l_batch,
    gpt_similarity,
    gpt_similarity_batch,
    win_rate,
    mean_latency,
)
from evaluation.log_parser import LogParser, parse_batch_logs, load_ground_truth_services, lookup_ground_truth


@dataclass
class CaseResult:
    """Evaluation result for a single test case."""
    case_id: str = ''
    date: str = ''
    time: str = ''

    # Failure localization
    predictions_services: List[str] = field(default_factory=list)
    predictions_metrics: str = ''
    ground_truth_service: str = ''
    a_at_1: float = 0.0
    a_at_3: float = 0.0
    a_at_5: float = 0.0

    # Agent reasoning outputs
    trace_expert_output: str = ''
    metric_expert_output: str = ''
    log_expert_output: str = ''
    root_cause_expert_output: str = ''

    # Phase latencies
    phase_latencies: Dict[str, float] = field(default_factory=dict)
    e2e_latency: float = 0.0

    # Metadata
    conflict_scenario: str = ''
    anomaly_method: str = ''
    rca_method: str = ''
    model_name: str = ''


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across all test cases.

    Matches the tables in Paper [171].
    """

    # Table II: Failure Localization
    localization: Dict[str, float] = field(default_factory=dict)

    # Tables III-V: Per-expert reasoning quality
    trace_expert: Dict[str, float] = field(default_factory=dict)
    metric_expert: Dict[str, float] = field(default_factory=dict)
    log_expert: Dict[str, float] = field(default_factory=dict)

    # Table VI: Latency
    latency: Dict[str, Any] = field(default_factory=dict)

    # Per-case details
    cases: List[CaseResult] = field(default_factory=list)

    # Metadata
    dataset: str = ''
    model: str = ''
    method: str = ''
    n_cases: int = 0
    timestamp: str = ''

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = []
        lines.append("=" * 70)
        lines.append("  LocaleXpert Evaluation Report (Paper [171] Metrics)")
        lines.append("=" * 70)
        lines.append(f"  Dataset:  {self.dataset}")
        lines.append(f"  Model:    {self.model}")
        lines.append(f"  Method:   {self.method}")
        lines.append(f"  Cases:    {self.n_cases}")
        lines.append(f"  Time:     {self.timestamp}")
        lines.append("")

        # Table II
        lines.append("-" * 70)
        lines.append("  Table II: Failure Localization Performance")
        lines.append("-" * 70)
        loc = self.localization
        lines.append(f"  {'Metric':<10} {'Value':>10}")
        lines.append(f"  {'------':<10} {'-----':>10}")
        for k in ['A@1', 'A@3', 'A@5']:
            lines.append(f"  {k:<10} {loc.get(k, 0.0):>10.4f}")
        lines.append("")

        # Tables III-V
        for expert_name, expert_data in [
            ("Table III: Trace Expert", self.trace_expert),
            ("Table IV: Metric Expert", self.log_expert),
            ("Table V: Log Expert", self.log_expert),
        ]:
            lines.append("-" * 70)
            lines.append(f"  {expert_name} Reasoning Performance")
            lines.append("-" * 70)
            lines.append(f"  {'Metric':<10} {'Value':>10}")
            lines.append(f"  {'------':<10} {'-----':>10}")
            for metric_name in ['BLEU-4', 'ROUGE-L', 'G-sim', 'W-rate']:
                lines.append(f"  {metric_name:<10} {expert_data.get(metric_name, 0.0):>10.4f}")
            lines.append("")

        # Table VI
        lines.append("-" * 70)
        lines.append("  Table VI: Latency")
        lines.append("-" * 70)
        lat = self.latency
        lines.append(f"  Mean latency:   {lat.get('mean', 0.0):>8.1f}s")
        lines.append(f"  Median latency: {lat.get('median', 0.0):>8.1f}s")
        lines.append(f"  Std latency:    {lat.get('std', 0.0):>8.1f}s")
        lines.append(f"  Min latency:    {lat.get('min', 0.0):>8.1f}s")
        lines.append(f"  Max latency:    {lat.get('max', 0.0):>8.1f}s")
        lines.append("")

        # Per-phase breakdown
        if self.cases:
            phases = self.cases[0].phase_latencies.keys()
            lines.append("  Per-phase average latency:")
            for phase in phases:
                vals = [c.phase_latencies.get(phase, 0.0) for c in self.cases
                        if phase in c.phase_latencies]
                if vals:
                    lines.append(f"    {phase:<20} {np.mean(vals):>8.1f}s")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'dataset': self.dataset,
            'model': self.model,
            'method': self.method,
            'n_cases': self.n_cases,
            'timestamp': self.timestamp,
            'localization': self.localization,
            'trace_expert': self.trace_expert,
            'metric_expert': self.metric_expert,
            'log_expert': self.log_expert,
            'latency': self.latency,
        }

    def save_json(self, path: str):
        """Save report as JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_markdown(self, path: str):
        """Save report as Markdown."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.summary())


class Evaluator:
    """Orchestrates evaluation of LocaleXpert pipeline runs.

    Usage:
        # Single log evaluation
        ev = Evaluator(log_path="run_case1.log", ground_truth_service="dbservice1")
        report = ev.evaluate()

        # Batch evaluation
        ev = Evaluator()
        report = ev.evaluate_batch(log_dir="logs/", ground_truth_pkl_dir="Datasets/GAIA/fault_injection_tracerank/")
    """

    def __init__(
        self,
        log_path: Optional[str] = None,
        ground_truth_service: Optional[str] = None,
        reference_texts: Optional[Dict[str, str]] = None,
        model_name: str = 'deepseek-r1-0528',
    ):
        """
        Args:
            log_path: Path to a single run.py log file.
            ground_truth_service: Ground truth root cause service for this case.
            reference_texts: Optional reference reasoning texts for each agent.
                             e.g., {'TraceExpert': '...', 'MetricExpert': '...', ...}
            model_name: LLM model name (for G-sim computation).
        """
        self.log_path = log_path
        self.ground_truth_service = ground_truth_service
        self.reference_texts = reference_texts or {}
        self.model_name = model_name

    def evaluate_single(self, log_path: Optional[str] = None) -> CaseResult:
        """Evaluate a single case from its log file.

        Args:
            log_path: Path to log file (overrides constructor value).

        Returns:
            CaseResult with computed metrics.
        """
        path = log_path or self.log_path
        if not path:
            raise ValueError("No log path provided")

        parser = LogParser(path).parse()
        meta = parser.metadata

        result = CaseResult()
        result.date = meta.get('date', '')
        result.time = meta.get('time', '')
        result.predictions_services = meta.get('root_services', [])
        result.predictions_metrics = meta.get('root_metrics_uni', '')
        result.ground_truth_service = self.ground_truth_service or ''
        result.phase_latencies = dict(parser.phase_latencies)
        result.e2e_latency = parser.get_e2e_latency() or 0.0
        result.conflict_scenario = meta.get('conflict_scenario', '')

        # Extract agent outputs from [EVAL] records
        agent_outputs = parser.get_agent_outputs()
        result.trace_expert_output = agent_outputs.get('TraceExpert', '')
        result.metric_expert_output = agent_outputs.get('MetricExpert', '')
        result.log_expert_output = agent_outputs.get('LogExpert', '')
        result.root_cause_expert_output = agent_outputs.get('RootCauseExpert', '')

        # Compute A@k
        if result.predictions_services and result.ground_truth_service:
            result.a_at_1 = topk_accuracy(
                result.predictions_services, result.ground_truth_service, k=1
            )
            result.a_at_3 = topk_accuracy(
                result.predictions_services, result.ground_truth_service, k=3
            )
            result.a_at_5 = topk_accuracy(
                result.predictions_services, result.ground_truth_service, k=5
            )

        return result

    def evaluate_batch(
        self,
        log_dir: str,
        log_pattern: str = '*.log',
        ground_truth_pkl_dir: Optional[str] = None,
        compute_gsim: bool = False,
        gsim_model: str = 'deepseek-r1-0528',
    ) -> EvaluationReport:
        """Evaluate a batch of cases from a directory of log files.

        Args:
            log_dir: Directory containing run log files.
            log_pattern: Glob pattern for log files.
            ground_truth_pkl_dir: Directory with fault_injection_list_*.pkl
                                  containing ground truth services.
            compute_gsim: Whether to compute G-sim (requires LLM API calls).
            gsim_model: Model to use for G-sim.

        Returns:
            EvaluationReport with aggregated metrics.
        """
        parsers = parse_batch_logs(log_dir, log_pattern)

        if not parsers:
            print("[Warning] No log files found or parsed successfully")
            return EvaluationReport()

        # Load ground truth
        gt_map = {}
        if ground_truth_pkl_dir:
            gt_map = load_ground_truth_services(ground_truth_pkl_dir)

        # Build CaseResult for each parsed log
        cases: List[CaseResult] = []
        for parser in parsers:
            result = CaseResult()
            result.date = parser.metadata.get('date', '')
            result.time = parser.metadata.get('time', '')
            result.predictions_services = parser.metadata.get('root_services', [])
            result.predictions_metrics = parser.metadata.get('root_metrics_uni', '')
            result.phase_latencies = dict(parser.phase_latencies)
            result.e2e_latency = parser.get_e2e_latency() or 0.0
            result.conflict_scenario = parser.metadata.get('conflict_scenario', '')

            # Agent outputs
            agent_outputs = parser.get_agent_outputs()
            result.trace_expert_output = agent_outputs.get('TraceExpert', '')
            result.metric_expert_output = agent_outputs.get('MetricExpert', '')
            result.log_expert_output = agent_outputs.get('LogExpert', '')
            result.root_cause_expert_output = agent_outputs.get('RootCauseExpert', '')

            # Ground truth from pkl map (with fuzzy time matching)
            mmdd = result.date
            hhmm = result.time
            gt_svc = lookup_ground_truth(gt_map, mmdd, hhmm)
            if gt_svc:
                result.ground_truth_service = gt_svc

            # Compute A@k
            if result.predictions_services and result.ground_truth_service:
                result.a_at_1 = topk_accuracy(
                    result.predictions_services, result.ground_truth_service, k=1
                )
                result.a_at_3 = topk_accuracy(
                    result.predictions_services, result.ground_truth_service, k=3
                )
                result.a_at_5 = topk_accuracy(
                    result.predictions_services, result.ground_truth_service, k=5
                )

            cases.append(result)

        # Build aggregated report
        report = EvaluationReport(
            cases=cases,
            n_cases=len(cases),
            dataset='GAIA',
            model=self.model_name,
            method='LocaleXpert',
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        )

        # Table II: Aggregated A@k
        valid_cases = [c for c in cases if c.ground_truth_service]
        if valid_cases:
            report.localization = {
                'A@1': float(np.mean([c.a_at_1 for c in valid_cases])),
                'A@3': float(np.mean([c.a_at_3 for c in valid_cases])),
                'A@5': float(np.mean([c.a_at_5 for c in valid_cases])),
            }

        # Tables III-V: Reasoning quality (requires reference texts)
        # If no reference texts provided, report empty metrics
        for expert_name, output_key in [
            ('trace_expert', 'trace_expert_output'),
            ('metric_expert', 'metric_expert_output'),
            ('log_expert', 'log_expert_output'),
        ]:
            ref_key = {
                'trace_expert': 'TraceExpert',
                'metric_expert': 'MetricExpert',
                'log_expert': 'LogExpert',
            }[expert_name]

            hypotheses = [getattr(c, output_key) for c in cases]
            hypotheses = [h for h in hypotheses if h]  # filter empty

            if hypotheses and ref_key in self.reference_texts:
                ref = self.reference_texts[ref_key]
                expert_report = {
                    'BLEU-4': bleu4_batch(hypotheses, [ref] * len(hypotheses)),
                    'ROUGE-L': rouge_l_batch(hypotheses, [ref] * len(hypotheses)),
                }
                if compute_gsim:
                    expert_report['G-sim'] = gpt_similarity_batch(
                        hypotheses, [ref] * len(hypotheses), model_name=gsim_model
                    )
                setattr(report, expert_name, expert_report)

        # Table VI: Latency
        latencies = [c.e2e_latency for c in cases if c.e2e_latency > 0]
        if latencies:
            report.latency = mean_latency(latencies)

        return report

    def evaluate(self) -> EvaluationReport:
        """Convenience: evaluate from constructor arguments.

        If log_path was given, evaluates a single case.
        """
        if self.log_path:
            case = self.evaluate_single()
            report = EvaluationReport(
                cases=[case],
                n_cases=1,
                dataset='GAIA',
                model=self.model_name,
                method='LocaleXpert',
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
            report.localization = {
                'A@1': case.a_at_1,
                'A@3': case.a_at_3,
                'A@5': case.a_at_5,
            }
            if case.e2e_latency > 0:
                report.latency = mean_latency([case.e2e_latency])
            return report
        else:
            raise ValueError("No log_path set. Use evaluate_batch() instead.")
