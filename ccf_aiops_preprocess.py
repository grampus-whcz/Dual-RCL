#!/usr/bin/env python3
"""
CCF AIOps Challenge 2022 Dataset Preprocessor
==============================================

Converts raw CCF AIOps 2022 data to the SoC-RCA pipeline's expected
intermediate format (the same layout used by the GAIA dataset).

Directory structure produced::

    {output_dir}/
      {date_str}_trace_ano/           # e.g. 0320_trace_ano/
        {date_str}.pkl                # call_path_dict
        normal_datasets               # normal STV text file
        data/
          {time_label}.csv            # trace CSV per fault window
      {date_str}_tracerca/            # e.g. 0320_tracerca/
        {time_label}.pkl              # fault injection list
      {date_str}_metric_fault/        # e.g. 0320_metric_fault/
        {time_label}/
          {service}/
            {kpi}_{mean}-{std}.csv
      {date_str}_microcause/          # e.g. 0320_microcause/
        microcause_{date}_{time_label}.xlsx
      {date_str}_log_fault/           # e.g. 0320_log_fault/
        {time_label}/
          {service}.csv

Usage::

    python ccf_aiops_preprocess.py \\
        --raw-dir  .../training_data_with_faults \\
        --gt-dir   .../training_data_with_faults/groundtruth \\
        --output-dir .../SoC-RCA \\
        --date 2022-03-20 \\
        --cloudbed cloudbed-1
"""

import os
import sys
import csv
import pickle
import argparse
import logging
import warnings
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

# ---------------------------------------------------------------------------
#  Add project root to sys.path so we can import data_models, etc.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.data_models import Span, Trace

# Beijing timezone (CCF AIOps timestamps are CST)
_CST = timezone(timedelta(hours=8))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-5s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ccf_aiops_preprocess")


# ======================================================================
#  Utility helpers
# ======================================================================

def _ts_to_label(ts_seconds: int) -> str:
    """Convert Unix timestamp (seconds) to HH-MM label (CST)."""
    dt = datetime.fromtimestamp(ts_seconds, tz=_CST)
    return dt.strftime("%H-%M")


def _ts_to_full_label(ts_seconds: int) -> str:
    """Convert Unix timestamp (seconds) to YYYY-MM-DD_HH-MM (CST)."""
    dt = datetime.fromtimestamp(ts_seconds, tz=_CST)
    return dt.strftime("%Y-%m-%d_%H-%M")


def _date_to_dirname(date_str: str) -> str:
    """'2022-03-20' -> '0320'."""
    parts = date_str.split("-")
    return parts[1] + parts[2]


def _cloudbed_to_k8s(cloudbed: str) -> str:
    """'cloudbed-1' -> 'k8s-1'."""
    return cloudbed.replace("cloudbed", "k8s")


# ======================================================================
#  Key metrics to include in MicroCause xlsx
#  (avoid overwhelming the causal analysis with hundreds of KPIs)
# ======================================================================
_CONTAINER_KPIS = [
    "container_cpu_usage_seconds",
    "container_cpu_cfs_throttled_periods",
    "container_cpu_cfs_throttled_seconds",
    "container_memory_usage_MB",
    "container_memory_working_set_MB",
    "container_memory_rss",
    "container_memory_cache",
    "container_memory_failcnt",
    "container_network_receive_MB",
    "container_network_transmit_MB",
    "container_network_receive_errors",
    "container_network_transmit_errors",
    "container_network_receive_packets_dropped",
    "container_network_transmit_packets_dropped",
    "container_fs_usage_MB",
    "container_fs_reads_MB",
    "container_fs_writes_MB",
    "container_fs_io_current",
]

_SERVICE_METRIC_COLS = ["rr", "sr", "mrt", "count"]

# Maximum metrics to include in MicroCause xlsx (to keep PCMCI tractable).
# GAIA uses ~80 metrics per event. We keep a similar count by selecting
# the most relevant metrics: container KPIs for the fault entity's pods,
# all service-level KPIs, and node KPIs for affected nodes.
_MICROCAUSE_MAX_METRICS = 120

# ======================================================================
#  Main Preprocessor Class
# ======================================================================


class CCFAIOpsPreprocessor:
    """Convert CCF AIOps 2022 raw data to SoC-RCA pipeline format.

    Parameters
    ----------
    raw_dir : str
        Path to ``training_data_with_faults/`` (contains date directories).
    gt_dir : str
        Path to ``groundtruth/`` directory.
    output_dir : str
        Base output directory (typically the SoC-RCA project root).
    window_minutes : int
        Half-window size in minutes around each fault event (default 30).
    """

    def __init__(
        self,
        raw_dir: str,
        gt_dir: str,
        output_dir: str,
        window_minutes: int = 30,
        skip_existing: bool = False,
    ):
        self.raw_dir = raw_dir
        self.gt_dir = gt_dir
        self.output_dir = output_dir
        self.window_minutes = window_minutes
        self.skip_existing = skip_existing
        self._log_dir = None  # lazy log file path cache

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def preprocess(self, date: str, cloudbed: str):
        """Run full preprocessing for *date* + *cloudbed*.

        Parameters
        ----------
        date : str
            Date string, e.g. ``'2022-03-20'``.
        cloudbed : str
            Cloudbed name, e.g. ``'cloudbed-1'``.
        """
        # Use a unique prefix per cloudbed to avoid overwriting
        # e.g., cloudbed-1 → "0320", cloudbed-2 → "0320b", cloudbed-3 → "0320c"
        #       cloudbed (test) → "0501t" (date + 't' suffix)
        base = _date_to_dirname(date)  # "0320", "0501"
        is_test = "-" not in cloudbed  # "cloudbed" (test) vs "cloudbed-1" (train)
        if is_test:
            date_str = base + "t"  # "0501t"
        else:
            bed_num = cloudbed.split("-")[-1]  # "1", "2", "3"
            if bed_num == "1":
                date_str = base
            else:
                date_str = base + chr(ord('a') + int(bed_num) - 1)  # "0320b", "0320c"
        cloudbed_dir = os.path.join(self.raw_dir, date, cloudbed)
        k8s_id = _cloudbed_to_k8s(cloudbed)  # e.g. "k8s-1", or "k8s" for test

        logger.info("=" * 60)
        logger.info(f"  CCF AIOps Preprocessor")
        logger.info(f"  Date: {date}  Cloudbed: {cloudbed}  ->  dir prefix: {date_str}")
        logger.info(f"  Raw dir: {cloudbed_dir}")
        logger.info("=" * 60)

        # 1. Load ground truth events
        gt_events = self._load_ground_truth(date, k8s_id)
        logger.info(f"  Ground truth events: {len(gt_events)}")

        if not gt_events:
            logger.warning("  No ground truth events found. Exiting.")
            return

        # 2. Load raw data (logs are lazy-loaded per window)
        logger.info("  Loading raw data...")
        trace_df = self._load_traces(cloudbed_dir)
        container_metric_files = self._load_container_metric_files(cloudbed_dir)
        service_metric_df = self._load_service_metrics(cloudbed_dir)
        node_metric_df = self._load_node_metrics(cloudbed_dir)
        self._log_dir = os.path.join(cloudbed_dir, "log", "all")  # lazy load

        logger.info(
            f"  Traces: {len(trace_df)} rows, "
            f"Container metric files: {len(container_metric_files)}, "
            f"Service metrics: {len(service_metric_df)} rows, "
            f"Node metrics: {len(node_metric_df)} rows, "
            f"Logs: lazy-loaded per window"
        )

        # 3. Build shared trace artifacts (call_path_dict, normal STV)
        logger.info("  Building trace artifacts (call_path_dict + normal STV)...")
        call_path_dict, normal_stv_lines = self._build_trace_artifacts(
            trace_df, gt_events
        )
        logger.info(
            f"  Call paths: {len(call_path_dict)}, "
            f"Normal STV traces: {len(normal_stv_lines)}"
        )

        # 4. Write shared trace artifacts
        trace_ano_dir = os.path.join(self.output_dir, f"{date_str}_trace_ano")
        os.makedirs(os.path.join(trace_ano_dir, "data"), exist_ok=True)

        pkl_path = os.path.join(trace_ano_dir, f"{date_str}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(call_path_dict, f)
        logger.info(f"  Written: {pkl_path}")

        normal_path = os.path.join(trace_ano_dir, "normal_datasets")
        with open(normal_path, "w") as f:
            f.write("\n".join(normal_stv_lines))
        logger.info(f"  Written: {normal_path}")

        # 5. Create output directories
        tracerca_dir = os.path.join(self.output_dir, f"{date_str}_tracerca")
        metric_fault_base = os.path.join(self.output_dir, f"{date_str}_metric_fault")
        microcause_dir = os.path.join(self.output_dir, f"{date_str}_microcause")
        log_fault_base = os.path.join(self.output_dir, f"{date_str}_log_fault")
        for d in [tracerca_dir, microcause_dir]:
            os.makedirs(d, exist_ok=True)

        # 6. Process each fault event
        for i, event in enumerate(gt_events):
            fault_ts = event["timestamp"]
            time_label = _ts_to_label(fault_ts)
            full_label = _ts_to_full_label(fault_ts)
            entity = event["cmdb_id"]
            fault_type = event["failure_type"]
            level = event["level"]

            # Skip if all outputs already exist
            if self.skip_existing:
                xlsx_path = os.path.join(
                    microcause_dir, f"microcause_{date}_{time_label}.xlsx"
                )
                if os.path.exists(xlsx_path):
                    logger.info(
                        f"  [{i + 1}/{len(gt_events)}] SKIP {time_label} (already exists)"
                    )
                    continue

            logger.info(
                f"  [{i + 1}/{len(gt_events)}] Event: {full_label} "
                f"entity={entity} level={level} type={fault_type}"
            )

            window_start = fault_ts - self.window_minutes * 60
            window_end = fault_ts + self.window_minutes * 60

            # --- Trace CSV for this window ---
            self._write_trace_csv(
                trace_ano_dir, time_label, trace_df, window_start, window_end
            )

            # --- Tracerca pkl ---
            self._write_tracerca_pkl(
                tracerca_dir, time_label, trace_df,
                window_start, window_end, call_path_dict,
            )

            # --- Metric fault dirs ---
            self._write_metric_fault_dirs(
                metric_fault_base, time_label,
                container_metric_files, service_metric_df,
                window_start, window_end, entity,
            )

            # --- Microcause xlsx ---
            self._write_microcause_xlsx(
                microcause_dir, date, time_label,
                container_metric_files, service_metric_df, node_metric_df,
                window_start, window_end,
            )

            # --- Log fault dirs (lazy-loaded from raw CSVs) ---
            self._write_log_fault_dirs_lazy(
                log_fault_base, time_label,
                window_start, window_end,
            )

        logger.info("=" * 60)
        logger.info(f"  Preprocessing complete: {len(gt_events)} fault events")
        logger.info(f"  Output base: {self.output_dir}")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    #  Ground Truth
    # ------------------------------------------------------------------

    def _load_ground_truth(self, date: str, k8s_id: str) -> list:
        """Load ground truth from CSV or JSON.

        Supports both training (CSV: groundtruth-k8s-{id}-{date}.csv)
        and test (JSON: groundtruth-{date}.json) formats.

        Returns list of dicts with keys:
        ``timestamp`` (int, seconds), ``level``, ``cmdb_id``, ``failure_type``.
        """
        # Try CSV format first (training data)
        gt_file = os.path.join(self.gt_dir, f"groundtruth-{k8s_id}-{date}.csv")
        if os.path.exists(gt_file):
            df = pd.read_csv(gt_file)
            events = []
            for _, row in df.iterrows():
                events.append(
                    {
                        "timestamp": int(row["timestamp"]),
                        "level": row["level"],
                        "cmdb_id": row["cmdb_id"],
                        "failure_type": row["failure_type"],
                    }
                )
            events.sort(key=lambda e: e["timestamp"])
            return events

        # Try JSON format (test data)
        gt_file = os.path.join(self.gt_dir, f"groundtruth-{date}.json")
        if os.path.exists(gt_file):
            import json
            with open(gt_file, 'r') as f:
                gt = json.load(f)
            events = []
            for i in range(len(gt["timestamp"])):
                events.append(
                    {
                        "timestamp": int(gt["timestamp"][i]),
                        "level": gt["level"][i],
                        "cmdb_id": gt["cmdb_id"][i],
                        "failure_type": gt["failure_type"][i],
                    }
                )
            events.sort(key=lambda e: e["timestamp"])
            return events

        logger.warning(f"Ground truth file not found for {date} / {k8s_id}")
        return []

    # ------------------------------------------------------------------
    #  Raw Data Loading
    # ------------------------------------------------------------------

    def _load_traces(self, cloudbed_dir: str) -> pd.DataFrame:
        """Load trace CSV (Jaeger span format).

        Some large trace CSVs (e.g., 0507, 9M rows / 1.3GB) trigger a native
        segmentation fault in the C CSV parser when read in one shot. A SIGSEGV
        cannot be caught by try/except (it kills the process), so we check the
        file size first and use chunked reading with the python engine for any
        file above 500 MB.
        """
        trace_file = os.path.join(cloudbed_dir, "trace", "all", "trace_jaeger-span.csv")
        if not os.path.exists(trace_file):
            logger.warning(f"Trace file not found: {trace_file}")
            return pd.DataFrame()
        size_mb = os.path.getsize(trace_file) / (1024 * 1024)
        if size_mb > 500:
            logger.info(f"    Large trace file ({size_mb:.0f} MB); using chunked python engine")
            chunks = []
            for chunk in pd.read_csv(trace_file, chunksize=500000,
                                     on_bad_lines='warn', engine='python'):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(trace_file)
        logger.info(f"    Traces loaded: {len(df)} spans")
        return df

    def _load_container_metric_files(self, cloudbed_dir: str) -> dict:
        """Load all container metric CSVs.

        Returns dict: kpi_name -> DataFrame(timestamp, cmdb_id, value).
        Only loads KPIs in _CONTAINER_KPIS to save memory and time.
        """
        container_dir = os.path.join(cloudbed_dir, "metric", "container")
        if not os.path.exists(container_dir):
            logger.warning(f"Container metric dir not found: {container_dir}")
            return {}

        result = {}
        all_files = os.listdir(container_dir)
        for fname in all_files:
            if not fname.endswith(".csv"):
                continue
            # Extract kpi_name from filename: kpi_{kpi_name}.csv
            kpi_name = fname[4:-4]  # strip "kpi_" prefix and ".csv" suffix
            if kpi_name not in _CONTAINER_KPIS:
                continue
            fpath = os.path.join(container_dir, fname)
            try:
                df = pd.read_csv(fpath)
                result[kpi_name] = df
            except Exception as e:
                logger.warning(f"    Failed to load {fname}: {e}")
        logger.info(f"    Container KPIs loaded: {list(result.keys())}")
        return result

    def _load_service_metrics(self, cloudbed_dir: str) -> pd.DataFrame:
        """Load service-level metrics CSV."""
        svc_file = os.path.join(cloudbed_dir, "metric", "service", "metric_service.csv")
        if not os.path.exists(svc_file):
            logger.warning(f"Service metric file not found: {svc_file}")
            return pd.DataFrame()
        return pd.read_csv(svc_file)

    def _load_node_metrics(self, cloudbed_dir: str) -> pd.DataFrame:
        """Load node-level metrics CSV."""
        node_dir = os.path.join(cloudbed_dir, "metric", "node")
        if not os.path.exists(node_dir):
            return pd.DataFrame()
        # There should be one CSV file
        csv_files = [f for f in os.listdir(node_dir) if f.endswith(".csv")]
        if not csv_files:
            return pd.DataFrame()
        return pd.read_csv(os.path.join(node_dir, csv_files[0]))

    def _write_log_fault_dirs_lazy(
        self, log_fault_base, time_label,
        window_start, window_end,
    ):
        """Write per-service log CSVs for a time window (lazy: reads raw logs on demand)."""
        out_dir = os.path.join(log_fault_base, time_label)
        os.makedirs(out_dir, exist_ok=True)

        if not self._log_dir or not os.path.isdir(self._log_dir):
            logger.warning(f"    No log directory for window {time_label}")
            return

        # Read log CSVs with chunked filtering to avoid loading all 12M+ rows
        svc_data = defaultdict(list)
        for fname in os.listdir(self._log_dir):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(self._log_dir, fname)
            try:
                # Use chunked reading to filter by time window efficiently
                chunks = pd.read_csv(fpath, chunksize=200_000)
                for chunk in chunks:
                    if "timestamp" not in chunk.columns:
                        continue
                    mask = (
                        (chunk["timestamp"] >= window_start)
                        & (chunk["timestamp"] <= window_end)
                    )
                    window_chunk = chunk.loc[mask]
                    if window_chunk.empty:
                        continue
                    for _, row in window_chunk.iterrows():
                        cmdb_id = row.get("cmdb_id", "unknown")
                        service = str(cmdb_id).split(".")[0] if "." in str(cmdb_id) else str(cmdb_id)
                        ts = row.get("timestamp", "")
                        try:
                            dt = datetime.fromtimestamp(int(ts), tz=_CST)
                            dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except (ValueError, TypeError):
                            dt_str = str(ts)
                        msg = row.get("value", "")
                        svc_data[service].append((dt_str, msg))
            except Exception as e:
                logger.warning(f"    Error reading log file {fname}: {e}")

        # Write per-service CSVs
        for service, entries in svc_data.items():
            fpath = os.path.join(out_dir, f"{service}.csv")
            with open(fpath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["datetime", "service", "message"])
                for dt_str, msg in entries:
                    writer.writerow([dt_str, service, msg])

        logger.info(f"    Written: {out_dir}/ ({len(svc_data)} service log files)")

    def _load_logs(self, cloudbed_dir: str) -> pd.DataFrame:
        """Load log CSVs (service + envoy) — only used if not lazy-loading."""
        log_dir = os.path.join(cloudbed_dir, "log", "all")
        if not os.path.exists(log_dir):
            return pd.DataFrame()
        dfs = []
        for fname in os.listdir(log_dir):
            if not fname.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(os.path.join(log_dir, fname))
                dfs.append(df)
            except Exception:
                pass
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    #  Trace Artifacts
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_trace_spans(trace_df: pd.DataFrame):
        """Parse trace DataFrame into per-trace span dicts.

        Yields (trace_id, span_dict) where span_dict maps
        span_id -> {response_time, operation, start_time, parent}.
        """
        if trace_df.empty:
            return

        current_tid = None
        trace = {}
        for row in trace_df.itertuples(index=False):
            tid = str(row.trace_id)
            if current_tid is not None and tid != current_tid:
                yield current_tid, trace
                trace = {}
            current_tid = tid
            sid = str(row.span_id)
            parent = str(row.parent_span) if pd.notna(row.parent_span) else None
            if parent == "" or parent == "nan":
                parent = None
            trace[sid] = {
                "response_time": int(row.duration),
                "operation": str(row.cmdb_id),
                "start_time": int(row.timestamp),
                "parent": parent,
            }
        if current_tid is not None:
            yield current_tid, trace

    @staticmethod
    def _walk_call_path(trace: dict, span_id: str, call_path: str = "") -> str:
        """Build call path string from span_id by following parent links."""
        span = trace[span_id]
        call_path = "#".join([span["operation"], call_path])
        parent = span["parent"]
        if parent is None:
            return "#".join(["start", call_path[:-1]])
        elif parent not in trace:
            return "#".join(["unclear_start", call_path[:-1]])
        else:
            return CCFAIOpsPreprocessor._walk_call_path(
                trace, parent, call_path
            )

    def _build_trace_artifacts(
        self, trace_df: pd.DataFrame, gt_events: list
    ) -> tuple:
        """Build call_path_dict and normal STV data.

        Returns (call_path_dict, normal_stv_lines).
        """
        call_path_set = set()
        all_traces = []  # (trace_id, trace_dict, is_normal)

        # Determine fault time ranges (to exclude from normal data)
        fault_windows = set()
        for ev in gt_events:
            for t in range(
                ev["timestamp"] - self.window_minutes * 60,
                ev["timestamp"] + self.window_minutes * 60,
                60,
            ):
                fault_windows.add(t)

        # Parse all traces and collect call paths
        for trace_id, trace in self._parse_trace_spans(trace_df):
            # Determine if this trace is in a fault window
            # Use the root span's start time
            is_normal = True
            for sid, span in trace.items():
                if span["parent"] is None:
                    ts_sec = span["start_time"] // 1000
                    if ts_sec in fault_windows:
                        is_normal = False
                    break

            # Collect call paths
            for sid, span in trace.items():
                try:
                    cp = self._walk_call_path(trace, sid)
                    call_path_set.add(cp)
                except Exception:
                    pass

            all_traces.append((trace_id, trace, is_normal))

        # Build call_path_dict (sorted for determinism)
        sorted_paths = sorted(call_path_set)
        call_path_dict = {cp: idx for idx, cp in enumerate(sorted_paths)}

        # Build normal STV data
        normal_stv_lines = []
        STV_length = len(call_path_dict)
        max_normal = 500  # limit normal traces to avoid huge files
        for trace_id, trace, is_normal in all_traces:
            if not is_normal:
                continue
            stv = [0] * STV_length
            valid = True
            for sid, span in trace.items():
                try:
                    cp = self._walk_call_path(trace, sid)
                except Exception:
                    valid = False
                    break
                if cp not in call_path_dict:
                    valid = False
                    break
                idx = call_path_dict[cp]
                rt = span["response_time"]
                if stv[idx] < rt:
                    stv[idx] = rt
            if valid:
                normal_stv_lines.append(
                    f"{trace_id}:{','.join(str(v) for v in stv)}"
                )
                if len(normal_stv_lines) >= max_normal:
                    break

        return call_path_dict, normal_stv_lines

    # ------------------------------------------------------------------
    #  Per-Event Output Writers
    # ------------------------------------------------------------------

    def _write_trace_csv(
        self, trace_ano_dir, time_label, trace_df,
        window_start, window_end,
    ):
        """Write trace CSV for a single time window."""
        data_dir = os.path.join(trace_ano_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        out_path = os.path.join(data_dir, f"{time_label}.csv")

        # Filter traces in the time window
        # trace timestamps are in milliseconds
        mask = (
            (trace_df["timestamp"] >= window_start * 1000)
            & (trace_df["timestamp"] <= window_end * 1000)
        )
        window_df = trace_df.loc[mask]

        # Write with AIOps 2022 columns + 'abnormal' column
        # The trace_anomaly_22.py expects 10 columns:
        # timestamp,cmdb_id,span_id,trace_id,duration,type,
        # status_code,operation_name,parent_span,abnormal
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "cmdb_id", "span_id", "trace_id",
                "duration", "type", "status_code", "operation_name",
                "parent_span", "abnormal",
            ])
            for _, row in window_df.iterrows():
                # Determine abnormal: status_code != 0 -> abnormal
                abnormal = 0
                try:
                    if int(row.get("status_code", 0)) != 0:
                        abnormal = 1
                except (ValueError, TypeError):
                    pass
                writer.writerow([
                    row["timestamp"],
                    row["cmdb_id"],
                    row["span_id"],
                    row["trace_id"],
                    row["duration"],
                    row.get("type", ""),
                    row.get("status_code", 0),
                    row.get("operation_name", ""),
                    row.get("parent_span", ""),
                    abnormal,
                ])
        logger.info(f"    Written: {out_path} ({len(window_df)} spans)")

    def _write_tracerca_pkl(
        self, tracerca_dir, time_label, trace_df,
        window_start, window_end, call_path_dict,
    ):
        """Write tracerca pkl for MEPFL trace root cause localization.

        The pkl contains a list of fault injection dicts, each with a
        ``trace_list`` key holding a list of Trace objects built from
        the spans in the time window.
        """
        out_path = os.path.join(tracerca_dir, f"{time_label}.pkl")
        mask = (
            (trace_df["timestamp"] >= window_start * 1000)
            & (trace_df["timestamp"] <= window_end * 1000)
        )
        window_df = trace_df.loc[mask]

        # Build Trace objects from the window spans
        trace_objs = self._build_trace_objects(window_df)
        fault_injection = {"trace_list": trace_objs}
        fault_injection_list = [fault_injection]

        with open(out_path, "wb") as f:
            pickle.dump(fault_injection_list, f)
        logger.info(f"    Written: {out_path} ({len(trace_objs)} traces)")

    @staticmethod
    def _build_trace_objects(trace_df: pd.DataFrame) -> list:
        """Build Trace/Span objects from a trace DataFrame.

        This follows the data model used by mepfl_22.py.
        """
        if trace_df.empty:
            return []

        # Group spans by trace_id
        trace_groups = {}
        for row in trace_df.itertuples(index=False):
            tid = str(row.trace_id)
            if tid not in trace_groups:
                trace_groups[tid] = []
            trace_groups[tid].append(row)

        trace_list = []
        for tid, spans in trace_groups.items():
            # Build Span objects
            span_dict = {}
            for s in spans:
                sid = str(s.span_id)
                parent_sid = str(s.parent_span) if pd.notna(s.parent_span) else None
                if parent_sid == "" or parent_sid == "nan":
                    parent_sid = None

                status_str = str(s.status_code) if pd.notna(s.status_code) else "0"
                try:
                    status_str = str(int(float(status_str)))
                except (ValueError, TypeError):
                    status_str = "0"

                span_obj = Span(
                    trace_id=tid,
                    span_id=sid,
                    parent_span_id=parent_sid,
                    children_span_list=[],
                    start_time=s.timestamp,
                    duration=float(s.duration),
                    service_name=str(s.cmdb_id),
                    anomaly=0,
                    status_code=status_str,
                    operation_name=str(s.operation_name) if pd.notna(s.operation_name) else "",
                )
                span_dict[sid] = span_obj

            # Link children
            root_span = None
            for sid, span in span_dict.items():
                if span.parent_span_id is None:
                    root_span = span
                elif span.parent_span_id in span_dict:
                    span_dict[span.parent_span_id].children_span_list.append(span)

            if root_span is None:
                # Fallback: use the first span as root
                if span_dict:
                    root_span = next(iter(span_dict.values()))
                else:
                    continue

            trace_obj = Trace(
                trace_id=tid,
                root_span=root_span,
                span_count=len(span_dict),
                source="AIOps2022",
            )
            trace_list.append(trace_obj)

        return trace_list

    def _write_metric_fault_dirs(
        self, metric_fault_base, time_label,
        container_metric_files, service_metric_df,
        window_start, window_end, fault_entity,
    ):
        """Write per-service metric CSV directories.

        Each service gets a directory containing metric CSVs with format:
        ``{kpi}_{mean}-{std}.csv`` containing ``timestamp,value``.
        """
        out_dir = os.path.join(metric_fault_base, time_label)

        # Collect services relevant to the fault entity
        # Extract service names from container metric cmdb_ids
        services_data = defaultdict(lambda: defaultdict(list))

        # Process container metrics
        for kpi_name, df in container_metric_files.items():
            mask = (
                (df["timestamp"] >= window_start)
                & (df["timestamp"] <= window_end)
            )
            window_df = df.loc[mask]

            for _, row in window_df.iterrows():
                cmdb_id = row["cmdb_id"]
                # Extract service: "node-6.adservice-0" -> "adservice-0"
                parts = cmdb_id.split(".")
                if len(parts) >= 2:
                    service = parts[1]
                else:
                    service = cmdb_id

                services_data[service][kpi_name].append(
                    (int(row["timestamp"]), float(row["value"]))
                )

        # Process service metrics
        if not service_metric_df.empty:
            mask = (
                (service_metric_df["timestamp"] >= window_start)
                & (service_metric_df["timestamp"] <= window_end)
            )
            svc_window = service_metric_df.loc[mask]
            for _, row in svc_window.iterrows():
                service_full = str(row["service"])  # e.g., "adservice-grpc"
                service_base = service_full.split("-")[0]  # e.g., "adservice"
                for col in _SERVICE_METRIC_COLS:
                    kpi_name = f"service_{col}"
                    services_data[service_base][kpi_name].append(
                        (int(row["timestamp"]), float(row[col]))
                    )

        # Write CSV files
        n_written = 0
        for service, kpis in services_data.items():
            svc_dir = os.path.join(out_dir, service)
            os.makedirs(svc_dir, exist_ok=True)

            for kpi_name, ts_vals in kpis.items():
                if len(ts_vals) < 5:
                    continue  # skip very short series

                # Sort by timestamp
                ts_vals.sort(key=lambda x: x[0])
                values = [v for _, v in ts_vals]

                # Compute mean and std for filename
                mean_val = np.mean(values)
                std_val = np.std(values)
                if std_val == 0:
                    std_val = 1e-6

                fname = f"{kpi_name}_{mean_val:.6f}-{std_val:.6f}.csv"
                fpath = os.path.join(svc_dir, fname)

                with open(fpath, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "value"])
                    for ts, val in ts_vals:
                        writer.writerow([ts, val])
                n_written += 1

        logger.info(f"    Written: {out_dir}/ ({n_written} metric CSVs)")

    def _write_microcause_xlsx(
        self, microcause_dir, date, time_label,
        container_metric_files, service_metric_df, node_metric_df,
        window_start, window_end,
    ):
        """Write MicroCause input xlsx (wide-format metric time series).

        Each row = one metric variable (first cell = name, rest = values).

        To keep PCMCI tractable (GAIA uses ~80 metrics), we select a
        representative subset: key container KPIs for all pods, all
        service-level KPIs, and a few node KPIs.  Total target: ~100-120.
        """
        out_path = os.path.join(
            microcause_dir, f"microcause_{date}_{time_label}.xlsx"
        )

        # Collect all metric time series in the window
        metric_series = {}  # metric_name -> [(timestamp, value), ...]

        # Key container KPIs (reduced subset for MicroCause tractability)
        _KEY_CONTAINER_KPIS = [
            "container_cpu_usage_seconds",
            "container_memory_usage_MB",
            "container_memory_working_set_MB",
            "container_memory_rss",
            "container_network_receive_MB",
            "container_network_transmit_MB",
            "container_network_receive_errors",
            "container_network_transmit_errors",
            "container_fs_usage_MB",
            "container_fs_io_current",
        ]

        # Container metrics: {pod}_{kpi_name}
        for kpi_name, df in container_metric_files.items():
            # Only include key KPIs to keep metric count manageable
            if kpi_name not in _KEY_CONTAINER_KPIS:
                continue
            mask = (
                (df["timestamp"] >= window_start)
                & (df["timestamp"] <= window_end)
            )
            window_df = df.loc[mask]
            for cmdb_id, group in window_df.groupby("cmdb_id"):
                parts = cmdb_id.split(".")
                entity = parts[1] if len(parts) >= 2 else cmdb_id
                col_name = f"{entity}_{kpi_name}"
                sorted_group = group.sort_values("timestamp")
                metric_series[col_name] = list(
                    zip(sorted_group["timestamp"].astype(int), sorted_group["value"].astype(float))
                )

        # Service metrics: {service}_{col} — include all (40 services × 4 cols ≈ 160)
        if not service_metric_df.empty:
            mask = (
                (service_metric_df["timestamp"] >= window_start)
                & (service_metric_df["timestamp"] <= window_end)
            )
            svc_window = service_metric_df.loc[mask]
            for service, group in svc_window.groupby("service"):
                service_base = service.split("-")[0]
                sorted_group = group.sort_values("timestamp")
                for col in _SERVICE_METRIC_COLS:
                    col_name = f"{service_base}_service_{col}"
                    metric_series[col_name] = list(
                        zip(
                            sorted_group["timestamp"].astype(int),
                            sorted_group[col].astype(float),
                        )
                    )

        # Node metrics: {node}_{kpi_name} — only key node KPIs
        _KEY_NODE_KPIS = {
            "system.cpu.iowait", "system.cpu.system", "system.cpu.user",
            "system.cpu.idle", "system.mem.used.percent",
            "system.mem.total", "system.mem.used",
            "system.net.bytes_in", "system.net.bytes_out",
        }
        if not node_metric_df.empty:
            mask = (
                (node_metric_df["timestamp"] >= window_start)
                & (node_metric_df["timestamp"] <= window_end)
            )
            node_window = node_metric_df.loc[mask]
            for cmdb_id, group in node_window.groupby("cmdb_id"):
                for kpi_name, kpi_group in group.groupby("kpi_name"):
                    if kpi_name not in _KEY_NODE_KPIS:
                        continue
                    col_name = f"{cmdb_id}_{kpi_name}"
                    sorted_group = kpi_group.sort_values("timestamp")
                    metric_series[col_name] = list(
                        zip(
                            sorted_group["timestamp"].astype(int),
                            sorted_group["value"].astype(float),
                        )
                    )

        if not metric_series:
            logger.warning(f"    No metric data for window {time_label}, skipping xlsx")
            return

        # If still too many, prioritize service metrics + prune container metrics
        if len(metric_series) > _MICROCAUSE_MAX_METRICS:
            # Keep all service metrics, prune container/node to fit
            svc_keys = [k for k in metric_series if "_service_" in k]
            other_keys = [k for k in metric_series if "_service_" not in k]
            remaining = _MICROCAUSE_MAX_METRICS - len(svc_keys)
            if remaining > 0:
                selected = svc_keys + other_keys[:remaining]
            else:
                selected = svc_keys[:_MICROCAUSE_MAX_METRICS]
            metric_series = {k: metric_series[k] for k in selected}

        # Align all series to a common time grid
        all_timestamps = set()
        for ts_vals in metric_series.values():
            all_timestamps.update(ts for ts, _ in ts_vals)
        all_timestamps = sorted(all_timestamps)
        ts_to_idx = {ts: idx for idx, ts in enumerate(all_timestamps)}
        n_times = len(all_timestamps)

        # Build xlsx
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"

        header = ["metric"] + [str(ts) for ts in all_timestamps]
        ws.append(header)

        for metric_name, ts_vals in sorted(metric_series.items()):
            row = [metric_name] + [None] * n_times
            val_dict = dict(ts_vals)
            for ts, val in ts_vals:
                if ts in ts_to_idx:
                    row[ts_to_idx[ts] + 1] = val
            # Forward-fill then backward-fill None values
            prev = None
            for i in range(1, len(row)):
                if row[i] is not None:
                    prev = row[i]
                elif prev is not None:
                    row[i] = prev
            nxt = None
            for i in range(len(row) - 1, 0, -1):
                if row[i] is not None:
                    nxt = row[i]
                elif nxt is not None:
                    row[i] = nxt
            row = [0 if v is None else v for v in row]
            ws.append(row)

        wb.save(out_path)
        logger.info(
            f"    Written: {out_path} "
            f"({len(metric_series)} metrics x {n_times} time points)"
        )

    def _write_log_fault_dirs(
        self, log_fault_base, time_label, log_df,
        window_start, window_end,
    ):
        """Write per-service log CSVs for a time window."""
        out_dir = os.path.join(log_fault_base, time_label)
        os.makedirs(out_dir, exist_ok=True)

        if log_df.empty:
            logger.warning(f"    No log data for window {time_label}")
            return

        # Filter logs in window
        if "timestamp" in log_df.columns:
            mask = (
                (log_df["timestamp"] >= window_start)
                & (log_df["timestamp"] <= window_end)
            )
            window_df = log_df.loc[mask]
        else:
            window_df = log_df

        # Group by cmdb_id (service/pod)
        for cmdb_id, group in window_df.groupby("cmdb_id"):
            # Extract base service name
            service = cmdb_id.split(".")[0] if "." in str(cmdb_id) else str(cmdb_id)
            fpath = os.path.join(out_dir, f"{service}.csv")

            with open(fpath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["datetime", "service", "message"])
                for _, row in group.iterrows():
                    ts = row.get("timestamp", "")
                    try:
                        dt = datetime.fromtimestamp(int(ts), tz=_CST)
                        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        dt_str = str(ts)
                    msg = row.get("value", "")
                    writer.writerow([dt_str, service, msg])

        n_files = len(window_df["cmdb_id"].unique()) if not window_df.empty else 0
        logger.info(f"    Written: {out_dir}/ ({n_files} service log files)")


# ======================================================================
#  CLI
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CCF AIOps 2022 data for SoC-RCA pipeline"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/training_data_with_faults",
        help="Path to training_data_with_faults/",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/training_data_with_faults/groundtruth",
        help="Path to groundtruth/ directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/shared-nvme/work/code/RCA/2026/SoC-RCA",
        help="Output directory (SoC-RCA project root)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2022-03-20",
        help="Date to process (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--cloudbed",
        type=str,
        default="cloudbed-1",
        help="Cloudbed to process (e.g., cloudbed-1)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Half-window size in minutes around each fault (default: 30)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip events whose output files already exist",
    )
    args = parser.parse_args()

    proc = CCFAIOpsPreprocessor(
        raw_dir=args.raw_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        window_minutes=args.window,
        skip_existing=args.skip_existing,
    )
    proc.preprocess(date=args.date, cloudbed=args.cloudbed)


if __name__ == "__main__":
    main()
