#!/usr/bin/env python3
"""
Step 4: Prepare runtime data directories for run.py.

For each fault injection event, creates the 5 directories that run.py expects:
    - {MMDD}_tracerca/          -> Trace/Span pkl files for MEPFL
    - {MMDD}_trace_ano/         -> call_path_dict, normal_datasets, trace CSVs
    - {MMDD}_metric_fault/      -> Per-service metric CSVs with anomaly windows
    - {MMDD}_microcause/        -> Excel files with metric time series matrices
    - {MMDD}_log_fault/         -> Per-service log CSVs

Usage:
    python scripts/step4_prepare_runtime_data.py
    python scripts/step4_prepare_runtime_data.py --dates 2021-07-01
"""

import argparse
import copy
import csv
import glob
import math
import os
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from openpyxl import Workbook

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.data_models import Span, Trace
from scripts.step3_train_mepfl import build_traces_from_rows

# Reuse trace_anomaly functions for call_path construction
# These are imported dynamically to avoid module-level dependencies


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def date_to_mmdd(date_str):
    """Convert '2021-07-01' to '0701'."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt.strftime('%m%d')


def time_to_hhmm(dt):
    """Convert datetime to 'HH-MM' format."""
    return dt.strftime('%H-%M')


# ---------------------------------------------------------------------------
# 4a: tracerca
# ---------------------------------------------------------------------------

def prepare_tracerca_dir(date_str, fault_injection_list, datasets_root):
    """Create {MMDD}_tracerca/ with pkl files grouped by HH-MM.

    Each pkl file contains a list of fault injection dicts where trace_list
    holds Trace/Span objects (not raw CSV rows), matching what mepfl.py expects.

    Args:
        date_str: Date string like '2021-07-01'
        fault_injection_list: List of fault injection dicts from step2 pkl
        datasets_root: Datasets/GAIA root path
    """
    mmdd = date_to_mmdd(date_str)
    output_dir = f'{mmdd}_tracerca'
    os.makedirs(output_dir, exist_ok=True)

    # Group fault injections by HH-MM
    grouped = defaultdict(list)
    for fi in fault_injection_list:
        hhmm = time_to_hhmm(fi['time'])
        # Convert raw trace rows to Trace/Span objects
        fi_copy = copy.deepcopy(fi)
        fi_copy['trace_list'] = build_traces_from_rows(fi['trace_list'])
        grouped[hhmm].append(fi_copy)

    for hhmm, fi_list in grouped.items():
        pkl_path = os.path.join(output_dir, f'{hhmm}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(fi_list, f)
        logger.debug(f"Saved {len(fi_list)} fault injections to {pkl_path}")

    logger.info(f"[tracerca] Created {output_dir}/ with {len(grouped)} pkl files")


# ---------------------------------------------------------------------------
# 4b: trace_ano
# ---------------------------------------------------------------------------

def prepare_trace_ano_dir(date_str, normal_csv_path, fault_injection_list,
                          gaia_trace_dir, datasets_root):
    """Create {MMDD}_trace_ano/ with call_path_dict, normal_datasets, and data CSVs.

    Three outputs:
    1. {MMDD}.pkl: call_path_dict mapping call path strings to indices
    2. normal_datasets: text file with normal trace STV representations
    3. data/*.csv: trace CSV snippets around each fault injection time

    Args:
        date_str: Date string
        normal_csv_path: Path to normal.csv from step2
        fault_injection_list: Fault injection dicts for this date
        gaia_trace_dir: GAIA MicroSS/trace directory
        datasets_root: Datasets/GAIA root
    """
    mmdd = date_to_mmdd(date_str)
    output_dir = f'{mmdd}_trace_ano'
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # --- Build call_path_dict from normal traces ---
    call_path_dict = {}
    normal_stv_lines = []
    call_path_idx = 0

    if os.path.exists(normal_csv_path):
        # Import trace_anomaly functions
        from trace_anomaly import get_trace_from_gaia, walk_call_path, trace_to_STV

        logger.info(f"Building call_path_dict from {normal_csv_path}...")
        chunk_size = 100000
        for chunk in pd.read_csv(normal_csv_path, chunksize=chunk_size):
            csv_data = chunk.values.tolist()
            traces = get_trace_from_gaia(csv_data)
            temp_stv = []
            while True:
                try:
                    trace_id, trace = next(traces)
                    # Extract call paths
                    call_paths, _ = walk_call_path(trace)
                    if isinstance(call_paths, list):
                        for cp in call_paths:
                            if cp not in call_path_dict:
                                call_path_dict[cp] = call_path_idx
                                call_path_idx += 1
                    # Compute STV
                    stv = trace_to_STV(trace, call_path_dict, call_path_idx + 1)
                    if isinstance(stv, list):
                        temp_stv.append(f'{trace_id}:{",".join(str(v) for v in stv)}')
                except StopIteration:
                    break
            normal_stv_lines.extend(temp_stv)

    # Save call_path_dict pkl
    pkl_path = os.path.join(output_dir, f'{mmdd}.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(call_path_dict, f)
    logger.info(f"[trace_ano] Saved call_path_dict ({len(call_path_dict)} paths) to {pkl_path}")

    # Save normal_datasets
    normal_ds_path = os.path.join(output_dir, 'normal_datasets')
    with open(normal_ds_path, 'w') as f:
        f.write('\n'.join(normal_stv_lines) + '\n')
    logger.info(f"[trace_ano] Saved {len(normal_stv_lines)} normal STV lines to {normal_ds_path}")

    # --- Create data CSVs for each fault injection time ---
    # Read all service trace files for this date and filter
    date_prefix = date_str  # e.g., '2021-07-01'

    # Group fault injections by HH-MM
    grouped = defaultdict(list)
    for fi in fault_injection_list:
        hhmm = time_to_hhmm(fi['time'])
        grouped[hhmm].append(fi)

    for hhmm, fi_group in grouped.items():
        # Use the first fault injection's time window to extract traces
        fault_time = fi_group[0]['time']
        # Define time window: ±30 minutes around fault
        time_start = fault_time.strftime('%H:%M')
        csv_filename = f'{date_str}_{hhmm}.csv'
        csv_path = os.path.join(data_dir, csv_filename)

        # Stream through trace files and extract rows in time window
        trace_files = glob.glob(os.path.join(gaia_trace_dir, 'trace_table_*.csv'))
        rows_written = 0
        with open(csv_path, 'w', encoding='utf-8', newline='') as out_f:
            writer = csv.writer(out_f)
            writer.writerow([
                'timestamp', 'host_ip', 'service_name', 'trace_id', 'span_id',
                'parent_id', 'start_time', 'end_time', 'url', 'status_code', 'message'
            ])

            for tf in trace_files:
                with open(tf, 'r', encoding='utf-8', errors='replace') as in_f:
                    reader = csv.reader(in_f)
                    next(reader)  # skip header
                    for row in reader:
                        if not row:
                            continue
                        if row[0].startswith(date_prefix):
                            # Filter by time proximity (within 30 min)
                            try:
                                row_time = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                                diff = abs((row_time - fault_time).total_seconds())
                                if diff <= 1800:
                                    writer.writerow(row)
                                    rows_written += 1
                            except ValueError:
                                continue

        logger.info(f"[trace_ano] Saved {rows_written} rows to {csv_path}")

    logger.info(f"[trace_ano] Created {output_dir}/")


# ---------------------------------------------------------------------------
# 4c: metric_fault
# ---------------------------------------------------------------------------

def build_metric_index(metric_dir):
    """Build an index mapping (service, kpi) -> list of file paths.

    Parses metric file names like:
        dbservice1_0.0.0.4_docker_cpu_core_0_norm_pct_2021-07-01_2021-07-15.csv

    Returns:
        dict: {(service, kpi_name): [file_paths sorted by date range]}
    """
    index = defaultdict(list)
    for f in os.listdir(metric_dir):
        if not f.endswith('.csv'):
            continue
        # Parse filename: {service}_{ip}_{metric_parts}_{start}_{end}.csv
        parts = f[:-4].rsplit('_', 2)  # split from right: [..., start_date, end_date]
        if len(parts) < 3:
            continue
        remainder = parts[0]  # service_ip_metric...
        end_date = parts[2]

        # Extract service name (first segment)
        segments = remainder.split('_')
        service = segments[0]
        # KPI = everything after service and IP (segments[1] is IP like '0.0.0.4')
        kpi = '_'.join(segments[2:]) if len(segments) > 2 else 'unknown'

        filepath = os.path.join(metric_dir, f)
        index[(service, kpi)].append((end_date, filepath))

    # Sort each list by date
    for key in index:
        index[key].sort(key=lambda x: x[0])

    logger.info(f"Built metric index: {len(index)} (service, kpi) pairs")
    return index


def get_metric_window(metric_files, fault_time_ts_ms, window_size=30, sampling_sec=30):
    """Extract a window of metric data around a fault time.

    Args:
        metric_files: List of (date_str, filepath) tuples sorted by date
        fault_time_ts_ms: Fault time as epoch milliseconds
        window_size: Number of data points to extract
        sampling_sec: Sampling interval in seconds (GAIA default: 30s)

    Returns:
        DataFrame with columns ['timestamp', 'value'] and exactly window_size rows,
        or None if insufficient data.
    """
    # Read all relevant metric files and concatenate
    dfs = []
    for _, fpath in metric_files:
        try:
            df = pd.read_csv(fpath)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    all_data = pd.concat(dfs, ignore_index=True)
    all_data['timestamp'] = all_data['timestamp'].astype(np.int64)
    all_data = all_data.sort_values('timestamp').drop_duplicates(subset=['timestamp'])

    # Find the index closest to fault_time
    fault_ts_s = fault_time_ts_ms / 1000.0
    mid_idx = all_data['timestamp'].sub(fault_time_ts_ms).abs().idxmin()

    # Extract window_size points centered on fault time
    half = window_size // 2
    start_idx = max(0, mid_idx - half)
    end_idx = start_idx + window_size
    if end_idx > len(all_data):
        end_idx = len(all_data)
        start_idx = max(0, end_idx - window_size)

    window = all_data.iloc[start_idx:end_idx].reset_index(drop=True)

    if len(window) != window_size:
        return None

    return window


def compute_normal_baseline(metric_files, fault_time_ts_ms, baseline_duration_sec=600):
    """Compute mean and std from a normal period before the fault.

    Args:
        metric_files: List of (date_str, filepath) tuples
        fault_time_ts_ms: Fault time as epoch milliseconds
        baseline_duration_sec: Duration of normal period (seconds)

    Returns:
        (mean, std) tuple
    """
    dfs = []
    for _, fpath in metric_files:
        try:
            df = pd.read_csv(fpath)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return (0.0, 1.0)

    all_data = pd.concat(dfs, ignore_index=True)
    all_data['timestamp'] = all_data['timestamp'].astype(np.int64)
    all_data = all_data.sort_values('timestamp')

    # Use period [fault - 2*duration, fault - duration] as baseline
    baseline_end_ms = fault_time_ts_ms - baseline_duration_sec * 1000
    baseline_start_ms = baseline_end_ms - baseline_duration_sec * 1000

    baseline = all_data[
        (all_data['timestamp'] >= baseline_start_ms) &
        (all_data['timestamp'] <= baseline_end_ms)
    ]

    if len(baseline) < 5:
        return (0.0, 1.0)

    mean = float(baseline['value'].mean())
    std = float(baseline['value'].std())
    if std == 0:
        std = 1e-6

    return (mean, std)


def prepare_metric_fault_dir(date_str, fault_injection_list, gaia_metric_dir, datasets_root):
    """Create {MMDD}_metric_fault/ with per-service metric CSV files.

    Directory structure:
        {MMDD}_metric_fault/{HH-MM}/{service_name}/{kpi}_{mean}-{std}.csv

    Each CSV has 30 rows with columns: timestamp, value

    Args:
        date_str: Date string
        fault_injection_list: Fault injection dicts for this date
        gaia_metric_dir: GAIA MicroSS/metric directory
        datasets_root: Datasets/GAIA root
    """
    mmdd = date_to_mmdd(date_str)

    # Build metric file index
    logger.info("Building metric file index...")
    metric_index = build_metric_index(gaia_metric_dir)

    # Get all unique services from metric files
    services = set()
    for (svc, _) in metric_index.keys():
        services.add(svc)

    # Group fault injections by HH-MM
    grouped = defaultdict(list)
    for fi in fault_injection_list:
        hhmm = time_to_hhmm(fi['time'])
        grouped[hhmm].append(fi)

    for hhmm, fi_group in tqdm(grouped.items(), desc=f'[metric_fault] {mmdd}'):
        fault_time = fi_group[0]['time']
        fault_ts_ms = int(fault_time.timestamp() * 1000)

        # Focus on top-5 root cause services from the fault injections
        fault_services = list(set(fi['service'] for fi in fi_group))

        output_dir = os.path.join(f'{mmdd}_metric_fault', hhmm)
        os.makedirs(output_dir, exist_ok=True)

        for service in fault_services:
            svc_dir = os.path.join(output_dir, service)
            os.makedirs(svc_dir, exist_ok=True)

            # Find all KPIs for this service
            svc_kpis = {kpi: files for (svc, kpi), files in metric_index.items() if svc == service}

            for kpi, files in svc_kpis.items():
                # Compute normal baseline
                mean, std = compute_normal_baseline(files, fault_ts_ms)

                # Extract metric window
                window = get_metric_window(files, fault_ts_ms, window_size=30)
                if window is None:
                    continue

                # Write CSV
                csv_name = f'{kpi}_{mean:.6f}-{std:.6f}.csv'
                csv_path = os.path.join(svc_dir, csv_name)
                window.to_csv(csv_path, index=False)

    logger.info(f"[metric_fault] Created {mmdd}_metric_fault/")


# ---------------------------------------------------------------------------
# 4d: microcause
# ---------------------------------------------------------------------------

def prepare_microcause_dir(date_str, fault_injection_list, gaia_metric_dir, datasets_root):
    """Create {MMDD}_microcause/ with Excel files for MicroCause analysis.

    Excel format (Sheet1):
        Row 1: header_row = ['servicename_kpi', val1, val2, ..., val30]
        Each subsequent row: [variable_name, val1, val2, ...]

    Args:
        date_str: Date string
        fault_injection_list: Fault injection dicts
        gaia_metric_dir: GAIA MicroSS/metric directory
        datasets_root: Datasets/GAIA root
    """
    mmdd = date_to_mmdd(date_str)
    output_dir = f'{mmdd}_microcause'
    os.makedirs(output_dir, exist_ok=True)

    # Build metric file index
    metric_index = build_metric_index(gaia_metric_dir)

    # Group fault injections by HH-MM
    grouped = defaultdict(list)
    for fi in fault_injection_list:
        hhmm = time_to_hhmm(fi['time'])
        grouped[hhmm].append(fi)

    # Select representative KPIs (avoid per-core metrics to reduce dimensionality)
    selected_kpi_keywords = ['cpu_total_pct', 'memory_usage_pct', 'memory_usage_total',
                             'network_in_bytes', 'network_out_bytes']

    for hhmm, fi_group in grouped.items():
        fault_time = fi_group[0]['time']
        fault_ts_ms = int(fault_time.timestamp() * 1000)

        wb = Workbook()
        ws = wb.active
        ws.title = 'Sheet1'

        # Header row
        header = ['metric_name'] + [f't{i}' for i in range(1, 31)]
        ws.append(header)

        row_count = 0
        for (service, kpi), files in metric_index.items():
            # Filter to representative KPIs
            if not any(kw in kpi for kw in selected_kpi_keywords):
                continue

            window = get_metric_window(files, fault_ts_ms, window_size=30)
            if window is None:
                continue

            var_name = f'{service}_{kpi}'
            row_data = [var_name] + window['value'].tolist()
            ws.append(row_data)
            row_count += 1

        if row_count > 0:
            xlsx_path = os.path.join(output_dir, f'microcause_{date_str}_{hhmm}.xlsx')
            wb.save(xlsx_path)
            logger.debug(f"[microcause] Saved {row_count} variables to {xlsx_path}")

    logger.info(f"[microcause] Created {output_dir}/")


# ---------------------------------------------------------------------------
# 4e: log_fault
# ---------------------------------------------------------------------------

def prepare_log_fault_dir(date_str, fault_injection_list, gaia_business_dir,
                           gaia_run_csv, datasets_root):
    """Create {MMDD}_log_fault/ with per-service log CSV files.

    Directory structure:
        {MMDD}_log_fault/{HH-MM}/{service_name}.csv

    CSV format: at least 3 columns, row[2] contains log message text.

    For July 2021, business_table only has webservice1 data.
    For other services, extract log entries from run_table WARNING/ERROR messages.

    Args:
        date_str: Date string
        fault_injection_list: Fault injection dicts
        gaia_business_dir: GAIA MicroSS/business directory
        gaia_run_csv: Path to run_table CSV
        datasets_root: Datasets/GAIA root
    """
    mmdd = date_to_mmdd(date_str)
    output_dir = f'{mmdd}_log_fault'
    os.makedirs(output_dir, exist_ok=True)

    # Group fault injections by HH-MM
    grouped = defaultdict(list)
    for fi in fault_injection_list:
        hhmm = time_to_hhmm(fi['time'])
        grouped[hhmm].append(fi)

    # Load run table entries for this date (for non-webservice1 services)
    run_entries = defaultdict(list)
    if os.path.exists(gaia_run_csv):
        with open(gaia_run_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('datetime', '').strip() == date_str:
                    run_entries[row.get('service', '')].append(row)

    # Load business log for this date (if available)
    business_entries = defaultdict(list)
    business_files = glob.glob(os.path.join(gaia_business_dir, 'business_table_*.csv'))
    for bf in business_files:
        if '2021-07' not in bf and '2021-08' not in bf:
            continue
        try:
            for chunk in pd.read_csv(bf, chunksize=50000):
                mask = chunk['datetime'].astype(str).str.startswith(date_str)
                for _, row in chunk[mask].iterrows():
                    svc = row.get('service', '')
                    business_entries[svc].append(row)
        except Exception as e:
            logger.warning(f"Error reading business file {bf}: {e}")

    for hhmm, fi_group in grouped.items():
        fault_time = fi_group[0]['time']
        time_window_sec = 1800  # ±30 minutes

        hhmm_dir = os.path.join(output_dir, hhmm)
        os.makedirs(hhmm_dir, exist_ok=True)

        # Get fault services
        fault_services = list(set(fi['service'] for fi in fi_group))

        for service in fault_services:
            csv_path = os.path.join(hhmm_dir, f'{service}.csv')

            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['datetime', 'service', 'message'])

                # Try business log entries first
                entries = business_entries.get(service, [])
                for entry in entries:
                    try:
                        msg = str(entry.get('message', ''))
                        entry_dt = datetime.strptime(msg.split('|')[0].strip(), '%Y-%m-%d %H:%M:%S')
                        if abs((entry_dt - fault_time).total_seconds()) <= time_window_sec:
                            writer.writerow([
                                entry.get('datetime', ''),
                                entry.get('service', ''),
                                msg
                            ])
                    except (ValueError, IndexError):
                        # Write anyway if it contains ERROR
                        if 'ERROR' in str(entry.get('message', '')):
                            writer.writerow([
                                entry.get('datetime', ''),
                                entry.get('service', ''),
                                str(entry.get('message', ''))
                            ])

                # Fall back to run table entries
                run_rows = run_entries.get(service, [])
                for row in run_rows:
                    msg = row.get('message', '')
                    if 'ERROR' in msg or 'WARNING' in msg:
                        writer.writerow([
                            row.get('datetime', ''),
                            row.get('service', ''),
                            msg
                        ])

    logger.info(f"[log_fault] Created {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 4: Prepare runtime data directories for run.py'
    )
    parser.add_argument(
        '--datasets-root', type=str, default='./Datasets/GAIA',
        help='Path to Datasets/GAIA'
    )
    parser.add_argument(
        '--gaia-root', type=str,
        default='/root/shared-nvme/data_set/gaia/gaia/MicroSS',
        help='Path to GAIA MicroSS directory'
    )
    parser.add_argument(
        '--dates', nargs='*', default=None,
        help='Specific dates to process (e.g., 2021-07-01)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    gaia_trace_dir = os.path.join(args.gaia_root, 'trace')
    gaia_metric_dir = os.path.join(args.gaia_root, 'metric')
    gaia_business_dir = os.path.join(args.gaia_root, 'business')
    gaia_run_csv = os.path.join(args.gaia_root, 'run', 'run_table_2021-07.csv')
    normal_csv_path = os.path.join(args.datasets_root, 'trace_by_day', 'normal.csv')
    pkl_dir = os.path.join(args.datasets_root, 'fault_injection_tracerank')

    # Load fault injection data
    if not os.path.isdir(pkl_dir):
        logger.error(f"Directory not found: {pkl_dir}. Run step2 first.")
        sys.exit(1)

    pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith('.pkl')])
    if not pkl_files:
        logger.error(f"No pkl files found in {pkl_dir}. Run step2 first.")
        sys.exit(1)

    # Determine dates to process
    if args.dates:
        dates = args.dates
    else:
        dates = []
        for pf in pkl_files:
            # Extract date from filename: fault_injection_list_2021-07-01.pkl
            match = re.search(r'(\d{4}-\d{2}-\d{2})', pf)
            if match:
                dates.append(match.group(1))

    logger.info(f"Processing {len(dates)} dates")

    for date_str in dates:
        logger.info(f"=== Processing date: {date_str} ===")

        # Load fault injection list for this date
        pkl_path = os.path.join(pkl_dir, f'fault_injection_list_{date_str}.pkl')
        if not os.path.exists(pkl_path):
            logger.warning(f"pkl not found: {pkl_path}")
            continue

        fault_injection_list = pickle.load(open(pkl_path, 'rb'))
        logger.info(f"Loaded {len(fault_injection_list)} fault injections for {date_str}")

        # 4a: tracerca
        prepare_tracerca_dir(date_str, fault_injection_list, args.datasets_root)

        # 4b: trace_ano
        prepare_trace_ano_dir(date_str, normal_csv_path, fault_injection_list,
                              gaia_trace_dir, args.datasets_root)

        # 4c: metric_fault
        prepare_metric_fault_dir(date_str, fault_injection_list, gaia_metric_dir, args.datasets_root)

        # 4d: microcause
        prepare_microcause_dir(date_str, fault_injection_list, gaia_metric_dir, args.datasets_root)

        # 4e: log_fault
        prepare_log_fault_dir(date_str, fault_injection_list, gaia_business_dir,
                              gaia_run_csv, args.datasets_root)

    logger.info("Step 4 complete.")


if __name__ == '__main__':
    main()
