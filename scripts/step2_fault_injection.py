#!/usr/bin/env python3
"""
Step 2: Generate fault injection pkl files and normal trace data.

Reads the merged trace-by-day CSVs and the run_table, and produces:
    - Datasets/GAIA/fault_injection_tracerank/fault_injection_list_{date}.pkl
    - Datasets/GAIA/trace_by_day/normal.csv

This script adapts the logic from data/parse_fault_injection.py, with a streaming
approach: it generates each day's trace CSV on demand (via step1), processes it,
then optionally deletes the day file to save disk space.

Usage:
    python scripts/step2_fault_injection.py
    python scripts/step2_fault_injection.py --cleanup false
"""

import argparse
import copy
import csv
import os
import pickle
import sys
from datetime import datetime
from tqdm import tqdm
from loguru import logger

# Add project root to path for cross-script imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import step1's merge function
from scripts.step1_trace_merge import merge_traces_for_date, setup_directories, copy_run_tables


def parse_run_table(run_csv):
    """Parse run_table and extract fault injection events.

    Filters out rows with 'INFO' or 'ERROR' in the message (system logs).
    Extracts fault injection start time from the message field.

    Args:
        run_csv: Path to run_table CSV file

    Returns:
        dict: {date_str: [list of fault injection dicts]}
              Each dict has keys: date, service, time, trace_list
    """
    fault_injection_dict = {}

    with open(run_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            msg = row.get('message', '')
            # Skip system log entries (INFO/ERROR are not fault injection triggers)
            if 'INFO' in msg or 'ERROR' in msg:
                continue

            date_str = row.get('datetime', '').strip()
            service = row.get('service', '').strip()

            if not date_str or not service:
                continue

            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                continue

            # Extract fault injection start time from message
            # Message format: "2021-07-01 11:44:26,xxx | WARNING | ... | [fault_type] ..."
            # The first part before comma is the timestamp
            try:
                time_str = msg.split(',')[0].strip()
                time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            except (ValueError, IndexError):
                continue

            if date_str not in fault_injection_dict:
                fault_injection_dict[date_str] = []

            fault_injection_dict[date_str].append({
                'date': date,
                'service': service,
                'time': time,
                'trace_list': []  # Will be filled during processing
            })

    logger.info(f"Parsed {sum(len(v) for v in fault_injection_dict.values())} fault injections "
                f"across {len(fault_injection_dict)} dates")
    return fault_injection_dict


def process_date(date_str, fault_injection_list, trace_dir, datasets_root, cleanup=True):
    """Process a single date: generate day CSV, extract normal/fault traces, save pkl.

    Steps:
    1. Call step1's merge_traces_for_date() to generate the day CSV
    2. Build seconds_to_row mapping (second offset -> list of CSV rows)
    3. Extract normal traces (remove rows within 300s of each fault injection)
    4. Extract fault traces (collect rows within 180s of each fault injection)
    5. Save fault_injection_list_{date}.pkl
    6. Optionally delete the day CSV

    Args:
        date_str: Date string like '2021-07-01'
        fault_injection_list: List of fault injection dicts for this date
        trace_dir: GAIA trace directory (source for merging)
        datasets_root: Datasets/GAIA root directory
        cleanup: Whether to delete the day CSV after processing
    """
    trace_by_day_dir = os.path.join(datasets_root, 'trace_by_day')
    pkl_output_dir = os.path.join(datasets_root, 'fault_injection_tracerank')

    # Step 1: Generate day CSV
    day_csv_path = merge_traces_for_date(date_str, trace_dir, trace_by_day_dir)
    if day_csv_path is None:
        logger.warning(f"No trace data for date {date_str}, skipping")
        return

    # Step 2: Read day CSV and build seconds_to_row
    logger.info(f"Reading day CSV: {day_csv_path}")
    seconds_to_row = {}  # second offset (0-86399) -> list of row dicts
    day_start = datetime.strptime(f'{date_str} 00:00:00', '%Y-%m-%d %H:%M:%S')

    with open(day_csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                seconds = int((ts - day_start).total_seconds())
                if 0 <= seconds < 86400:
                    if seconds not in seconds_to_row:
                        seconds_to_row[seconds] = []
                    seconds_to_row[seconds].append(row)
            except (ValueError, KeyError):
                continue

    logger.info(f"Loaded {sum(len(v) for v in seconds_to_row.values())} trace rows for {date_str}")

    # Step 3: Extract normal traces (remove fault injection windows)
    normal_seconds_to_row = copy.deepcopy(seconds_to_row)
    for fault_injection in fault_injection_list:
        fault_seconds = int((fault_injection['time'] - day_start).total_seconds())
        # Remove rows within 300 seconds of fault injection time
        for s in range(max(0, fault_seconds - 300), min(86399, fault_seconds + 300)):
            normal_seconds_to_row[s] = []

    normal_rows = []
    for rows_list in normal_seconds_to_row.values():
        normal_rows.extend(rows_list)

    # Append normal traces to normal.csv
    normal_csv_path = os.path.join(trace_by_day_dir, 'normal.csv')
    normal_header = not os.path.exists(normal_csv_path)
    with open(normal_csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'host_ip', 'service_name', 'trace_id', 'span_id',
            'parent_id', 'start_time', 'end_time', 'url', 'status_code', 'message'
        ])
        if normal_header:
            writer.writeheader()
        writer.writerows(normal_rows)

    logger.info(f"Appended {len(normal_rows)} normal trace rows for {date_str}")

    # Step 4: Extract fault traces for each fault injection
    fault_injection_list_copy = copy.deepcopy(fault_injection_list)
    for fault_injection in tqdm(fault_injection_list_copy, desc=f'Fault traces {date_str}'):
        fault_seconds = int((fault_injection['time'] - day_start).total_seconds())
        # Collect trace rows within 180 seconds of fault injection time
        for s in range(max(0, fault_seconds - 180), min(86399, fault_seconds + 180)):
            if s in seconds_to_row:
                fault_injection['trace_list'].extend(seconds_to_row[s])

    # Step 5: Save pkl
    pkl_path = os.path.join(pkl_output_dir, f'fault_injection_list_{date_str}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(fault_injection_list_copy, f)
    logger.info(f"Saved {len(fault_injection_list_copy)} fault injections to {pkl_path}")

    # Step 6: Optionally cleanup day CSV to save disk space
    if cleanup and os.path.exists(day_csv_path):
        os.remove(day_csv_path)
        logger.info(f"Cleaned up: {day_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 2: Generate fault injection pkl files and normal trace data'
    )
    parser.add_argument(
        '--datasets-root', type=str, default='./Datasets/GAIA',
        help='Root directory for Datasets/GAIA'
    )
    parser.add_argument(
        '--gaia-trace-root', type=str,
        default='/root/shared-nvme/data_set/gaia/gaia/MicroSS/trace',
        help='Path to GAIA MicroSS trace directory'
    )
    parser.add_argument(
        '--cleanup', type=str, default='true',
        help='Delete day CSV files after processing to save disk (true/false)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    do_cleanup = args.cleanup.lower() in ('true', '1', 'yes')

    # Ensure directory structure exists
    setup_directories(args.datasets_root)

    # Find run_table
    run_csv = os.path.join(args.datasets_root, 'run', 'run_table_2021-07.csv')
    if not os.path.exists(run_csv):
        gaia_run = os.path.join(os.path.dirname(args.gaia_trace_root), 'run', 'run_table_2021-07.csv')
        if os.path.exists(gaia_run):
            copy_run_tables(
                os.path.join(os.path.dirname(args.gaia_trace_root), 'run'),
                os.path.join(args.datasets_root, 'run')
            )
        run_csv = os.path.join(args.datasets_root, 'run', 'run_table_2021-07.csv')

    if not os.path.exists(run_csv):
        logger.error(f"run_table not found: {run_csv}")
        sys.exit(1)

    # Parse run table
    fault_injection_dict = parse_run_table(run_csv)

    # Process each date
    for date_str in sorted(fault_injection_dict.keys()):
        logger.info(f"Processing date: {date_str}")
        process_date(
            date_str=date_str,
            fault_injection_list=fault_injection_dict[date_str],
            trace_dir=args.gaia_trace_root,
            datasets_root=args.datasets_root,
            cleanup=do_cleanup
        )

    logger.info("Step 2 complete.")


if __name__ == '__main__':
    main()
