#!/usr/bin/env python3
"""
CLI entry point for running evaluation on LocaleXpert logs.

Supports three modes:
  1. Standard metrics (A@k, Latency) — no LLM calls
  2. With reasoning quality metrics (BLEU-4, ROUGE-L, G-sim)
  3. Full pipeline with LLM-based W-rate voting

Usage:
    # Standard evaluation (A@k + Latency)
    python -m evaluation.run_evaluation \\
        --log experiments_a.log \\
        --gt-pkl-dir Datasets/GAIA/fault_injection_tracerank/

    # With reference generation and G-sim
    python -m evaluation.run_evaluation \\
        --log experiments_a.log \\
        --gt-pkl-dir Datasets/GAIA/fault_injection_tracerank/ \\
        --generate-references \\
        --gsim

    # Full pipeline with W-rate
    python -m evaluation.run_evaluation \\
        --log experiments_a.log \\
        --gt-pkl-dir Datasets/GAIA/fault_injection_tracerank/ \\
        --ref-input evaluation/reference_texts \\
        --gsim --gsim-judges glm-4.7 \\
        --compute-wrate \\
        --methods-log LocaleXpert=experiments_localexpert.log \\
                      DualChannel=experiments_a.log \\
        --output evaluation_results/
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.log_parser import (
    LogParser, load_ground_truth_services, lookup_ground_truth, parse_multi_case_log,
)
from evaluation.evaluator import Evaluator, EvaluationReport
from evaluation.metrics import mean_latency


def parse_methods_log(methods_log_args: list) -> dict:
    """Parse --methods-log arguments into {method_name: log_path}.

    Accepts format: method_name=log_path
    """
    result = {}
    for item in methods_log_args:
        if '=' in item:
            name, path = item.split('=', 1)
            result[name.strip()] = path.strip()
        else:
            print(f"[Warning] Ignoring malformed --methods-log argument: {item}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LocaleXpert runs using Paper [171] metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard evaluation
  %(prog)s --log experiments_a.log --gt-pkl-dir Datasets/GAIA/fault_injection_tracerank/

  # Generate references + G-sim
  %(prog)s --log experiments_a.log --generate-references --gsim

  # Full pipeline with W-rate
  %(prog)s --compute-wrate --methods-log MethodA=a.log MethodB=b.log
        """,
    )

    # ---- Log source (mutually exclusive) ----
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--log', type=str,
        help='Path to a single run.py log file (may contain multiple cases)'
    )
    group.add_argument(
        '--log-dir', type=str,
        help='Directory containing multiple run log files'
    )

    # ---- Ground truth ----
    parser.add_argument(
        '--gt', type=str, default=None,
        help='Manually specify ground truth root cause service name'
    )
    parser.add_argument(
        '--gt-pkl-dir', type=str,
        default='Datasets/GAIA/fault_injection_tracerank',
        help='Directory with fault_injection_list_*.pkl (default: Datasets/GAIA/fault_injection_tracerank)'
    )

    # ---- Reference text options ----
    ref_group = parser.add_argument_group('Reference Text Generation')
    ref_group.add_argument(
        '--generate-references', action='store_true', default=False,
        help='Generate reference texts using LLM pipeline (replaces human experts)'
    )
    ref_group.add_argument(
        '--ref-model', type=str, default='glm-4.7',
        help='Model for reference generation (default: glm-4.7)'
    )
    ref_group.add_argument(
        '--ref-output', type=str, default='evaluation/reference_texts',
        help='Directory to save generated reference texts (default: evaluation/reference_texts)'
    )
    ref_group.add_argument(
        '--ref-input', type=str, default=None,
        help='Load pre-generated reference texts from this directory (skip generation)'
    )
    ref_group.add_argument(
        '--reference-file', type=str, default=None,
        help='(Legacy) JSON file with reference reasoning texts per agent'
    )

    # ---- G-sim options ----
    gsim_group = parser.add_argument_group('G-sim (LLM-as-Judge)')
    gsim_group.add_argument(
        '--gsim', action='store_true', default=False,
        help='Compute G-sim (requires LLM API access)'
    )
    gsim_group.add_argument(
        '--gsim-model', type=str, default='glm-4.7',
        help='Single model for G-sim (default: glm-4.7)'
    )
    gsim_group.add_argument(
        '--gsim-judges', nargs='+', default=None,
        help='Multiple judge models for G-sim (e.g., glm-4.7 gpt-4o)'
    )

    # ---- W-rate options ----
    wrate_group = parser.add_argument_group('W-rate (LLM-based Voting)')
    wrate_group.add_argument(
        '--compute-wrate', action='store_true', default=False,
        help='Compute W-rate using LLM voters (replaces human voters)'
    )
    wrate_group.add_argument(
        '--methods-log', nargs='+', type=str, default=None,
        help='Log files for each method (format: method_name=log_path)'
    )
    wrate_group.add_argument(
        '--voter-model', type=str, default='glm-4.7',
        help='Model for LLM voters (default: glm-4.7)'
    )

    # ---- Output ----
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory for saving results (JSON + Markdown)'
    )
    parser.add_argument(
        '--model-name', type=str, default='glm-4.7',
        help='LLM model name used in the evaluated runs (default: glm-4.7)'
    )

    args = parser.parse_args()

    # ---- Load reference texts (legacy --reference-file) ----
    reference_texts = {}
    if args.reference_file and os.path.exists(args.reference_file):
        import json
        with open(args.reference_file, 'r') as f:
            reference_texts = json.load(f)

    # ---- Determine ref_input vs generate ----
    # If --ref-input is specified, use it; otherwise use --generate-references
    ref_input_dir = args.ref_input
    generate_refs = args.generate_references and not ref_input_dir

    # ---- Parse W-rate methods ----
    methods_logs = None
    if args.compute_wrate and args.methods_log:
        methods_logs = parse_methods_log(args.methods_log)
        if not methods_logs:
            print("[Error] --methods-log requires at least one method_name=log_path")
            sys.exit(1)

    # ---- Run evaluation ----
    if args.log:
        # ---- Single / multi-case log evaluation ----
        parsers = parse_multi_case_log(args.log)

        if len(parsers) == 1:
            # ---- Single case ----
            parsed = parsers[0]
            date_mmdd = parsed.metadata.get('date', '')
            time_hhmm = parsed.metadata.get('time', '')

            # Determine ground truth
            gt_service = args.gt
            if not gt_service and args.gt_pkl_dir:
                gt_map = load_ground_truth_services(args.gt_pkl_dir)
                gt_service = lookup_ground_truth(gt_map, date_mmdd, time_hhmm)
                if gt_service:
                    print(f"  [GT] Auto-loaded: date={date_mmdd}, "
                          f"time={time_hhmm} -> {gt_service}")
                else:
                    print(f"  [GT] WARNING: No ground truth found for "
                          f"date={date_mmdd}, time={time_hhmm}")

            ev = Evaluator(
                ground_truth_service=gt_service,
                reference_texts=reference_texts,
                model_name=args.model_name,
            )
            case = ev.evaluate_single(args.log)

            report = EvaluationReport(
                cases=[case],
                n_cases=1,
                dataset='GAIA',
                model=args.model_name,
                method='LocaleXpert',
            )
            report.localization = {
                'A@1': case.a_at_1,
                'A@3': case.a_at_3,
                'A@5': case.a_at_5,
            }
            if case.e2e_latency > 0:
                report.latency = mean_latency([case.e2e_latency])

        else:
            # ---- Multi-case batch from single log ----
            print(f"  [Batch] Detected {len(parsers)} cases in {args.log}")
            ev = Evaluator(
                reference_texts=reference_texts,
                model_name=args.model_name,
            )
            report = ev.evaluate_batch(
                parsers=parsers,
                ground_truth_pkl_dir=args.gt_pkl_dir,
                compute_gsim=args.gsim,
                gsim_model=args.gsim_model,
                gsim_judges=args.gsim_judges,
                generate_references=generate_refs,
                ref_model=args.ref_model,
                ref_output_dir=args.ref_output,
                ref_input_dir=ref_input_dir,
                compute_wrate=args.compute_wrate,
                methods_logs=methods_logs,
                voter_model=args.voter_model,
            )

    else:
        # ---- Batch from directory of logs ----
        ev = Evaluator(
            reference_texts=reference_texts,
            model_name=args.model_name,
        )
        report = ev.evaluate_batch(
            log_dir=args.log_dir,
            ground_truth_pkl_dir=args.gt_pkl_dir,
            compute_gsim=args.gsim,
            gsim_model=args.gsim_model,
            gsim_judges=args.gsim_judges,
            generate_references=generate_refs,
            ref_model=args.ref_model,
            ref_output_dir=args.ref_output,
            ref_input_dir=ref_input_dir,
            compute_wrate=args.compute_wrate,
            methods_logs=methods_logs,
            voter_model=args.voter_model,
        )

    # ---- Print summary ----
    print(report.summary())

    # ---- Save results ----
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        json_path = os.path.join(args.output, 'evaluation_report.json')
        md_path = os.path.join(args.output, 'evaluation_report.md')
        report.save_json(json_path)
        report.save_markdown(md_path)
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")


if __name__ == '__main__':
    main()
