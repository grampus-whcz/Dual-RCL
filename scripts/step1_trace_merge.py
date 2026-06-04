#!/usr/bin/env python3
"""
Step 1: Merge GAIA trace files from per-service format to per-day format.

GAIA raw trace data is organized as trace_table_{service}_2021-07.csv (one file per service per month).
This script merges all service trace files for each day into a single CSV file, producing:
    Datasets/GAIA/trace_by_day/{YYYY-MM-DD}.csv

Usage:
    python scripts/step1_trace_merge.py --gaia-root /path/to/gaia/MicroSS --output-root ./Datasets/GAIA
    python scripts/step1_trace_merge.py --dates 2021-07-01 2021-07-02
"""

import argparse
import csv
import glob
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from heapq import heappush, heappop

import pandas as pd
from loguru import logger

# GAIA trace CSV columns
TRACE_COLUMNS = [
    'timestamp', 'host_ip', 'service_name', 'trace_id', 'span_id',
    'parent_id', 'start_time', 'end_time', 'url', 'status_code', 'message'
]


def setup_directories(output_root):
    """Create the Datasets/GAIA directory structure and copy run tables."""
    dirs = [
        os.path.join(output_root, 'trace_by_day'),
        os.path.join(output_root, 'run'),
        os.path.join(output_root, 'fault_injection_tracerank'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Directory ready: {d}")


def copy_run_tables(gaia_run_dir, output_run_dir):
    """Copy or symlink run_table CSV files from GAIA to Datasets/GAIA/run/."""
    for f in glob.glob(os.path.join(gaia_run_dir, 'run_table_*.csv')):
        dst = os.path.join(output_run_dir, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)
            logger.info(f"Copied {f} -> {dst}")
        else:
            logger.info(f"Already exists: {dst}")


def get_fault_dates(run_csv):
    """Extract dates that have fault injection events from run_table.

    Returns:
        list of date strings like ['2021-07-01', '2021-07-02', ...]
    """
    dates = set()
    with open(run_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            msg = row.get('message', '')
            # Skip INFO and ERROR log entries (these are system logs, not fault injection triggers)
            if 'INFO' in msg or 'ERROR' in msg:
                continue
            dt = row.get('datetime', '').strip()
            if dt:
                dates.add(dt)
    sorted_dates = sorted(dates)
    logger.info(f"Found {len(sorted_dates)} dates with fault injection events")
    return sorted_dates


def merge_traces_for_date(date_str, trace_dir, output_dir):
    """Merge all service trace CSVs for a single date into one file.

    Uses a heap-merge strategy: opens all 10 service trace CSVs simultaneously,
    reads rows whose timestamp matches date_str, and writes them sorted by timestamp.

    Args:
        date_str: Date string like '2021-07-01'
        trace_dir: Directory containing trace_table_*.csv files
        output_dir: Directory to write the merged file

    Returns:
        Path to the output file
    """
    output_path = os.path.join(output_dir, f'{date_str}.csv')

    if os.path.exists(output_path):
        logger.info(f"Already exists, skipping: {output_path}")
        return output_path

    trace_files = sorted(glob.glob(os.path.join(trace_dir, 'trace_table_*.csv')))
    if not trace_files:
        logger.warning(f"No trace files found in {trace_dir}")
        return None

    logger.info(f"Merging {len(trace_files)} trace files for date {date_str}")

    # Collect all rows for this date from all service files
    day_rows = []

    for tf in trace_files:
        service_name = os.path.basename(tf).replace('trace_table_', '').replace('_2021-07.csv', '')
        logger.debug(f"  Reading {service_name}...")
        try:
            with open(tf, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                header = next(reader)  # skip header
                for row in reader:
                    if not row:
                        continue
                    # row[0] is timestamp like '2021-07-01 10:54:23'
                    if row[0].startswith(date_str):
                        day_rows.append(row)
        except Exception as e:
            logger.warning(f"Error reading {tf}: {e}")
            continue

    if not day_rows:
        logger.warning(f"No trace data found for date {date_str}")
        return None

    # Sort by timestamp
    day_rows.sort(key=lambda r: r[0])

    # Write merged file
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(TRACE_COLUMNS)
        writer.writerows(day_rows)

    logger.info(f"Written {len(day_rows)} rows to {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 1: Merge GAIA trace files from per-service to per-day format'
    )
    parser.add_argument(
        '--gaia-root', type=str,
        default='/root/shared-nvme/data_set/gaia/gaia/MicroSS',
        help='Path to GAIA MicroSS directory'
    )
    parser.add_argument(
        '--output-root', type=str,
        default='./Datasets/GAIA',
        help='Output directory for Datasets/GAIA'
    )
    parser.add_argument(
        '--dates', nargs='*', default=None,
        help='Specific dates to process (e.g., 2021-07-01). If not specified, process all fault dates.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    trace_dir = os.path.join(args.gaia_root, 'trace')
    run_dir = os.path.join(args.gaia_root, 'run')

    if not os.path.isdir(trace_dir):
        logger.error(f"Trace directory not found: {trace_dir}")
        sys.exit(1)

    # Setup directory structure
    setup_directories(args.output_root)

    # Copy run tables
    output_run_dir = os.path.join(args.output_root, 'run')
    copy_run_tables(run_dir, output_run_dir)

    # Determine which dates to process
    if args.dates:
        dates = args.dates
    else:
        run_csv = os.path.join(output_run_dir, 'run_table_2021-07.csv')
        if not os.path.exists(run_csv):
            # Try the original GAIA location
            run_csv = os.path.join(run_dir, 'run_table_2021-07.csv')
        if os.path.exists(run_csv):
            dates = get_fault_dates(run_csv)
        else:
            logger.error("No run_table found. Please specify --dates manually.")
            sys.exit(1)

    output_trace_dir = os.path.join(args.output_root, 'trace_by_day')

    for date_str in dates:
        logger.info(f"Processing date: {date_str}")
        result = merge_traces_for_date(date_str, trace_dir, output_trace_dir)
        if result:
            logger.info(f"Done: {result}")
        else:
            logger.warning(f"Skipped date: {date_str}")

    logger.info("Step 1 complete.")


if __name__ == '__main__':
    main()
