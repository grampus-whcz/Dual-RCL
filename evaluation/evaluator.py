"""
Main Evaluator class that orchestrates metric computation.

Produces evaluation reports matching Paper [171] Tables II-VI:
  - Table II:  Failure Localization Performance (A@1, A@3, A@5)
  - Table III: Trace Expert Reasoning Performance (BLEU-4, ROUGE-L, G-sim, W-rate)
  - Table IV:  Metric Expert Reasoning Performance
  - Table V:   Log Expert Reasoning Performance
  - Table VI:  Average Single-Case Latency and Aggregated Performance

Enhanced with LLM-based evaluation replacing human engineers:
  - Reference text generation (replaces expert-written references)
  - Multi-judge G-sim (replaces single-model G-sim)
  - LLM-based W-rate voting (replaces human engineer voting)
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
    gpt_similarity_multi_judge,
    gpt_similarity_multi_judge_batch,
    win_rate,
    mean_latency,
)
from evaluation.log_parser import (
    LogParser, parse_batch_logs, parse_multi_case_log,
    load_ground_truth_services, lookup_ground_truth,
)
from evaluation.reference_generator import ReferenceGenerator, CaseData
from evaluation.llm_voter import LLMVoter


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

    Matches the tables in Paper [171], enhanced with LLM-based metrics.
    """

    # Table II: Failure Localization
    localization: Dict[str, float] = field(default_factory=dict)

    # Tables III-V: Per-expert reasoning quality
    trace_expert: Dict[str, float] = field(default_factory=dict)
    metric_expert: Dict[str, float] = field(default_factory=dict)
    log_expert: Dict[str, float] = field(default_factory=dict)

    # Table VI: Latency
    latency: Dict[str, Any] = field(default_factory=dict)

    # LLM-based W-rate results
    win_rates: Dict[str, float] = field(default_factory=dict)
    vote_details: List[dict] = field(default_factory=list)
    tie_rate: float = 0.0

    # Reference text metadata
    reference_metadata: Dict[str, Any] = field(default_factory=dict)

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
        lines.append("  Evaluation Report (Paper [171] Metrics)")
        lines.append("=" * 70)
        lines.append(f"  Dataset:  {self.dataset}")
        lines.append(f"  Model:    {self.model}")
        lines.append(f"  Method:   {self.method}")
        lines.append(f"  Cases:    {self.n_cases}")
        lines.append(f"  Time:     {self.timestamp}")
        if self.reference_metadata:
            lines.append(f"  Ref Gen:  {self.reference_metadata.get('model', 'N/A')}")
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
            ("Table IV: Metric Expert", self.metric_expert),
            ("Table V: Log Expert", self.log_expert),
        ]:
            lines.append("-" * 70)
            lines.append(f"  {expert_name} Reasoning Performance")
            lines.append("-" * 70)
            lines.append(f"  {'Metric':<10} {'Value':>10}")
            lines.append(f"  {'------':<10} {'-----':>10}")
            for metric_name in ['BLEU-4', 'ROUGE-L', 'G-sim', 'W-rate']:
                val = expert_data.get(metric_name, 0.0)
                lines.append(f"  {metric_name:<10} {val:>10.4f}")
            lines.append("")

        # W-rate table (if computed)
        if self.win_rates:
            lines.append("-" * 70)
            lines.append("  LLM-based Win Rate (W-rate)")
            lines.append("-" * 70)
            lines.append(f"  {'Method':<25} {'W-rate':>10} {'Wins':>6} {'Ties':>6}")
            lines.append(f"  {'------':<25} {'------':>10} {'----':>6} {'----':>6}")
            n_cases = self.n_cases or 1
            for method, rate in sorted(self.win_rates.items(), key=lambda x: -x[1]):
                wins = int(rate * n_cases)
                ties = int(self.tie_rate * n_cases)
                lines.append(f"  {method:<25} {rate:>10.4f} {wins:>6} {ties:>6}")
            lines.append(f"  Tie rate: {self.tie_rate:.2%}")
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
        d = {
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
            'win_rates': self.win_rates,
            'tie_rate': self.tie_rate,
            'reference_metadata': self.reference_metadata,
        }
        return d

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

    Supports three evaluation modes:
      1. Standard metrics (A@k, BLEU-4, ROUGE-L) — no LLM calls needed
      2. With G-sim — LLM-as-Judge similarity scoring
      3. Full pipeline — includes reference generation + LLM W-rate voting

    Usage:
        # Standard evaluation
        ev = Evaluator(log_path="run_case1.log", ground_truth_service="dbservice1")
        report = ev.evaluate()

        # Batch with reference generation and G-sim
        ev = Evaluator()
        report = ev.evaluate_batch(
            log_dir="logs/",
            ground_truth_pkl_dir="Datasets/GAIA/fault_injection_tracerank/",
            generate_references=True,
            compute_gsim=True,
        )
    """

    # Mapping from actual agent names in [EVAL] records to paper's expert names.
    _AGENT_NAME_MAP = {
        'TraceAnalysis':  'TraceExpert',
        'MetricAnalysis': 'MetricExpert',
        'LogAnalysis':    'LogExpert',
        'TradeAnalysis':  'TradeExpert',
        'CMDBAnalysis':   'CMDBExpert',
        'RootCauseAnalysis': 'RootCauseExpert',
        'TraceExpert':    'TraceExpert',
        'MetricExpert':   'MetricExpert',
        'LogExpert':      'LogExpert',
    }

    def __init__(
        self,
        log_path: Optional[str] = None,
        ground_truth_service: Optional[str] = None,
        reference_texts: Optional[Dict[str, str]] = None,
        model_name: str = 'glm-4.7',
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

    def _build_case_result(
        self,
        parser: LogParser,
        gt_service: Optional[str] = None,
    ) -> CaseResult:
        """Build a CaseResult from a parsed LogParser.

        Centralises the extraction of predictions, agent outputs, and
        A@k computation so that both single-case and batch paths share
        the same logic.

        Args:
            parser: A parsed LogParser instance.
            gt_service: Ground truth root cause service (overrides any
                        value stored inside the parser).

        Returns:
            Populated CaseResult.
        """
        meta = parser.metadata
        result = CaseResult()
        result.date = meta.get('date', '')
        result.time = meta.get('time', '')
        result.predictions_services = parser.get_predictions()
        result.predictions_metrics = parser.get_root_metrics()
        result.ground_truth_service = gt_service or ''
        result.phase_latencies = dict(parser.phase_latencies)
        result.e2e_latency = parser.get_e2e_latency() or 0.0
        result.conflict_scenario = meta.get('conflict_scenario', '')

        # Agent outputs — map actual names to paper's canonical names
        raw_outputs = parser.get_agent_outputs()
        mapped: Dict[str, str] = {}
        for agent_name, text in raw_outputs.items():
            canonical = self._AGENT_NAME_MAP.get(agent_name, agent_name)
            mapped[canonical] = text

        result.trace_expert_output = mapped.get('TraceExpert', '')
        result.metric_expert_output = mapped.get('MetricExpert', '')
        result.log_expert_output = mapped.get('LogExpert', '')
        result.root_cause_expert_output = mapped.get('RootCauseExpert', '')

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

    @staticmethod
    def _build_case_data(parser: LogParser, gt_service: str) -> CaseData:
        """Build a CaseData from a parsed LogParser for reference generation."""
        meta = parser.metadata
        return CaseData(
            case_id=f"{meta.get('date', '')}_{meta.get('time', '')}",
            date=meta.get('date', ''),
            time=meta.get('time', ''),
            ground_truth_service=gt_service,
            trace_anomaly_text=str(meta.get('trace_anomaly_count', '')),
            root_services=meta.get('root_services', []),
            metric_anomaly_text='',
            root_metrics_text=meta.get('root_metrics_uni', ''),
            log_anomaly_text='',
            conflict_resolution=meta.get('conflict_scenario', ''),
            causal_graph_info=(
                f"{meta.get('causal_graph_nodes', '?')} nodes, "
                f"{meta.get('causal_graph_edges', '?')} edges"
                if 'causal_graph_nodes' in meta else ''
            ),
            univariate_anomaly_count=meta.get('univariate_anomaly_count', 0),
            multivariate_anomaly=meta.get('multivariate_anomaly', False),
        )

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
        return self._build_case_result(parser, gt_service=self.ground_truth_service)

    def evaluate_batch(
        self,
        log_dir: Optional[str] = None,
        log_pattern: str = '*.log',
        parsers: Optional[List[LogParser]] = None,
        ground_truth_pkl_dir: Optional[str] = None,
        compute_gsim: bool = False,
        gsim_model: str = 'glm-4.7',
        gsim_judges: Optional[List[str]] = None,
        generate_references: bool = False,
        ref_model: str = 'glm-4.7',
        ref_output_dir: str = 'evaluation/reference_texts',
        ref_input_dir: Optional[str] = None,
        compute_wrate: bool = False,
        methods_logs: Optional[Dict[str, str]] = None,
        voter_model: str = 'glm-4.7',
    ) -> EvaluationReport:
        """Evaluate a batch of cases.

        Accepts either a directory of per-case log files (``log_dir``) or a
        pre-split list of ``LogParser`` objects (``parsers``).

        Args:
            log_dir: Directory containing per-case run log files.
            log_pattern: Glob pattern for log files.
            parsers: Pre-parsed LogParser list (overrides log_dir).
            ground_truth_pkl_dir: Directory with fault_injection_list_*.pkl.
            compute_gsim: Whether to compute G-sim.
            gsim_model: Single model for G-sim (used if gsim_judges is None).
            gsim_judges: List of judge models for multi-judge G-sim.
            generate_references: Generate reference texts via LLM.
            ref_model: Model for reference generation.
            ref_output_dir: Where to save generated references.
            ref_input_dir: Load pre-generated references from this dir.
            compute_wrate: Compute W-rate via LLM voting.
            methods_logs: {method_name: log_path} for W-rate comparison.
            voter_model: Model for LLM voters.

        Returns:
            EvaluationReport with aggregated metrics.
        """
        if parsers is None:
            if not log_dir:
                raise ValueError("Either log_dir or parsers must be provided")
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
            mmdd = parser.metadata.get('date', '')
            hhmm = parser.metadata.get('time', '')
            gt_svc = lookup_ground_truth(gt_map, mmdd, hhmm)
            result = self._build_case_result(parser, gt_service=gt_svc)
            cases.append(result)

        # ---- Reference text handling ----
        ref_texts: Dict[str, Dict[str, str]] = {}  # {case_id: {expert: text}}
        ref_metadata: Dict[str, Any] = {}

        if ref_input_dir:
            # Load pre-generated references
            loaded = ReferenceGenerator.load_references(ref_input_dir)
            for case_id, data in loaded.items():
                ref_texts[case_id] = data.get('references', {})
            ref_metadata = {
                'source': 'pre-generated',
                'dir': ref_input_dir,
                'n_cases': len(ref_texts),
            }

        elif generate_references:
            # Generate references via LLM pipeline
            gen = ReferenceGenerator(model_name=ref_model)
            cases_data = []
            for parser in parsers:
                mmdd = parser.metadata.get('date', '')
                hhmm = parser.metadata.get('time', '')
                gt_svc = lookup_ground_truth(gt_map, mmdd, hhmm)
                if gt_svc:
                    cd = self._build_case_data(parser, gt_svc)
                    cases_data.append(cd)

            if cases_data:
                all_refs = gen.generate_batch(cases_data, ref_output_dir)
                for case_id, data in all_refs.items():
                    ref_texts[case_id] = data.get('references', {})
                ref_metadata = {
                    'source': 'generated',
                    'model': ref_model,
                    'dir': ref_output_dir,
                    'n_cases': len(ref_texts),
                }
        elif self.reference_texts:
            # Use reference texts provided at init (single set for all cases)
            ref_metadata = {'source': 'provided'}

        # Build aggregated report
        report = EvaluationReport(
            cases=cases,
            n_cases=len(cases),
            dataset='GAIA',
            model=self.model_name,
            method='LocaleXpert',
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            reference_metadata=ref_metadata,
        )

        # Table II: Aggregated A@k
        valid_cases = [c for c in cases if c.ground_truth_service]
        if valid_cases:
            report.localization = {
                'A@1': float(np.mean([c.a_at_1 for c in valid_cases])),
                'A@3': float(np.mean([c.a_at_3 for c in valid_cases])),
                'A@5': float(np.mean([c.a_at_5 for c in valid_cases])),
            }

        # Tables III-V: Reasoning quality
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
            hypotheses = [h for h in hypotheses if h]

            # Determine reference text for this expert
            refs_for_expert: List[str] = []

            if ref_texts:
                # Per-case references
                for c in cases:
                    cid = f"{c.date}_{c.time}"
                    if cid in ref_texts and ref_key in ref_texts[cid]:
                        refs_for_expert.append(ref_texts[cid][ref_key])
            elif self.reference_texts and ref_key in self.reference_texts:
                # Single reference text for all cases
                refs_for_expert = [self.reference_texts[ref_key]] * len(hypotheses)

            if hypotheses and refs_for_expert and len(hypotheses) == len(refs_for_expert):
                expert_report = {
                    'BLEU-4': bleu4_batch(hypotheses, refs_for_expert),
                    'ROUGE-L': rouge_l_batch(hypotheses, refs_for_expert),
                }
                if compute_gsim:
                    if gsim_judges and len(gsim_judges) > 1:
                        multi_result = gpt_similarity_multi_judge_batch(
                            hypotheses, refs_for_expert,
                            judge_models=gsim_judges,
                        )
                        expert_report['G-sim'] = multi_result['overall_mean']
                        for m, s in multi_result['per_model_mean'].items():
                            expert_report[f'G-sim({m})'] = s
                    else:
                        expert_report['G-sim'] = gpt_similarity_batch(
                            hypotheses, refs_for_expert, model_name=gsim_model
                        )
                setattr(report, expert_name, expert_report)

        # W-rate via LLM voting
        if compute_wrate and methods_logs:
            report = self._compute_wrate(
                report, methods_logs, gt_map, voter_model
            )

        # Table VI: Latency
        latencies = [c.e2e_latency for c in cases if c.e2e_latency > 0]
        if latencies:
            report.latency = mean_latency(latencies)

        return report

    def _compute_wrate(
        self,
        report: EvaluationReport,
        methods_logs: Dict[str, str],
        gt_map: Dict[str, Dict[str, str]],
        voter_model: str,
    ) -> EvaluationReport:
        """Compute W-rate by comparing multiple methods via LLM voting.

        Args:
            report: Partially built EvaluationReport to augment.
            methods_logs: {method_name: log_path} for comparison.
            gt_map: Ground truth service map.
            voter_model: Model name for LLM voters.

        Returns:
            Augmented EvaluationReport with win_rates and vote_details.
        """
        # Parse each method's log
        method_parsers: Dict[str, List[LogParser]] = {}
        for method_name, log_path in methods_logs.items():
            if os.path.isfile(log_path):
                method_parsers[method_name] = parse_multi_case_log(log_path)
            else:
                print(f"[W-rate] Warning: log not found for {method_name}: {log_path}")

        if len(method_parsers) < 2:
            print("[W-rate] Warning: Need at least 2 methods to compare.")
            return report

        # Build voting cases — align by date+time across methods
        # Use the first method's cases as the baseline
        baseline_method = list(method_parsers.keys())[0]
        baseline_parsers = method_parsers[baseline_method]

        voter = LLMVoter(model_name=voter_model)
        vote_cases = []

        for bp in baseline_parsers:
            mmdd = bp.metadata.get('date', '')
            hhmm = bp.metadata.get('time', '')
            gt_svc = lookup_ground_truth(gt_map, mmdd, hhmm)

            if not gt_svc:
                continue

            # Collect this case's output from each method
            methods_output = {}
            for method_name, parsers_list in method_parsers.items():
                # Find the parser for this date+time
                for p in parsers_list:
                    p_date = p.metadata.get('date', '')
                    p_time = p.metadata.get('time', '')
                    if p_date == mmdd and p_time == hhmm:
                        raw_outputs = p.get_agent_outputs()
                        mapped = {}
                        for agent, text in raw_outputs.items():
                            canonical = self._AGENT_NAME_MAP.get(agent, agent)
                            mapped[canonical] = text
                        if mapped:
                            methods_output[method_name] = mapped
                        break

            if len(methods_output) >= 2:
                vote_cases.append({
                    'case_id': f"{mmdd}_{hhmm}",
                    'methods_output': methods_output,
                    'ground_truth': gt_svc,
                    'case_context': f"date={mmdd}, time={hhmm}",
                })

        if not vote_cases:
            print("[W-rate] Warning: No aligned cases found across methods.")
            return report

        # Run voting
        print(f"[W-rate] Voting on {len(vote_cases)} cases across "
              f"{len(method_parsers)} methods...")
        vote_results = voter.vote_batch(vote_cases)
        method_names = list(method_parsers.keys())
        win_rates = voter.compute_win_rates(vote_results, method_names)
        tie_rate = voter.compute_tie_rate(vote_results)

        report.win_rates = win_rates
        report.tie_rate = tie_rate
        report.vote_details = [
            {
                'case_id': r.case_id,
                'votes': r.votes,
                'winner': r.winner,
                'is_tie': r.is_tie,
                'tie_break_reasoning': r.tie_break_reasoning,
            }
            for r in vote_results
        ]

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
