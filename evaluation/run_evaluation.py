#!/usr/bin/env python3
"""
CLI entry point for running evaluation on LocaleXpert logs.

Usage:
    # Evaluate a single log file (auto-load ground truth from pkl)
    python -m evaluation.run_evaluation \
        --log experiments_a.log \
        --gt-pkl-dir Datasets/GAIA/fault_injection_tracerank/

    # Evaluate a batch of logs
    python -m evaluation.run_evaluation \
        --log-dir Report/ \
        --gt-pkl-dir Datasets/GAIA/fault_injection_tracerank/

    # With G-sim computation (requires LLM API)
    python -m evaluation.run_evaluation \
        --log-dir Report/ \
        --gt-pkl-dir Datasets/GAIA/fault_injection_tracerank/ \
        --gsim

    # Save results to file
    python -m evaluation.run_evaluation \
        --log-dir Report/ \
        --gt-pkl-dir Datasets/GAIA/fault_injection_tracerank/ \
        --output evaluation_results/
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.log_parser import LogParser, load_ground_truth_services, lookup_ground_truth
from evaluation.evaluator import Evaluator, EvaluationReport
from evaluation.metrics import mean_latency


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LocaleXpert runs using Paper [171] metrics'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--log', type=str,
        help='Path to a single run.py log file'
    )
    group.add_argument(
        '--log-dir', type=str,
        help='Directory containing multiple run log files'
    )

    # Ground truth options
    parser.add_argument(
        '--gt', type=str, default=None,
        help='(Optional) Manually specify ground truth root cause service name. '
             'If omitted with --log, automatically loaded from --gt-pkl-dir by matching date+time.'
    )
    parser.add_argument(
        '--gt-pkl-dir', type=str,
        default='Datasets/GAIA/fault_injection_tracerank',
        help='Directory with fault_injection_list_*.pkl for loading ground truth. '
             'Default: Datasets/GAIA/fault_injection_tracerank'
    )

    # G-sim options
    parser.add_argument(
        '--gsim', action='store_true', default=False,
        help='Compute G-sim (requires LLM API access)'
    )
    parser.add_argument(
        '--gsim-model', type=str, default='deepseek-r1-0528',
        help='Model for G-sim computation'
    )

    # Reference texts for reasoning evaluation
    parser.add_argument(
        '--reference-file', type=str, default=None,
        help='JSON file with reference reasoning texts per agent'
    )

    # Output
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory for saving results (JSON + Markdown)'
    )
    parser.add_argument(
        '--model-name', type=str, default='deepseek-r1-0528',
        help='LLM model name used in the evaluated runs'
    )

    args = parser.parse_args()

    # Load reference texts if provided
    reference_texts = {}
    if args.reference_file and os.path.exists(args.reference_file):
        import json
        with open(args.reference_file, 'r') as f:
            reference_texts = json.load(f)

    if args.log:
        # ---- Single log evaluation ----

        # Parse the log to extract date+time
        parsed = LogParser(args.log).parse()
        date_mmdd = parsed.metadata.get('date', '')
        time_hhmm = parsed.metadata.get('time', '')

        # Determine ground truth service
        gt_service = args.gt  # manual override
        if not gt_service and args.gt_pkl_dir:
            # Auto-load from pkl by matching date+time (with fuzzy matching)
            gt_map = load_ground_truth_services(args.gt_pkl_dir)
            gt_service = lookup_ground_truth(gt_map, date_mmdd, time_hhmm)
            if gt_service:
                print(f"  [GT] Auto-loaded from pkl: "
                      f"date={date_mmdd}, time={time_hhmm} -> {gt_service}")
            else:
                print(f"  [GT] WARNING: No ground truth found in pkl for "
                      f"date={date_mmdd}, time={time_hhmm}")
                print(f"  [GT] Available dates in pkl: {sorted(gt_map.keys())[:5]}...")
                if date_mmdd in gt_map:
                    print(f"  [GT] Available times for {date_mmdd}: "
                          f"{sorted(gt_map[date_mmdd].keys())[:5]}...")

        ev = Evaluator(
            ground_truth_service=gt_service,
            reference_texts=reference_texts,
            model_name=args.model_name,
        )
        case = ev.evaluate_single(args.log)

        # Wrap into EvaluationReport
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
        # ---- Batch evaluation ----
        ev = Evaluator(
            reference_texts=reference_texts,
            model_name=args.model_name,
        )
        report = ev.evaluate_batch(
            log_dir=args.log_dir,
            ground_truth_pkl_dir=args.gt_pkl_dir,
            compute_gsim=args.gsim,
            gsim_model=args.gsim_model,
        )

    # Print summary
    print(report.summary())

    # Save if output directory specified
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
