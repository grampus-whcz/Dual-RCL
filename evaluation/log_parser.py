"""
Parse run.py runtime logs to extract structured evaluation data.

run.py (after instrumentation) prints lines tagged with [EVAL] containing
JSON-serialised evaluation data. This module extracts them into structured
records for metric computation.

Log line format:
    [EVAL] {"type": "localization", "predictions": [...], "ground_truth": "...", ...}
    [EVAL] {"type": "agent_output", "agent": "TraceExpert", "text": "...", ...}
    [EVAL] {"type": "phase_latency", "phase": "Phase 1", "duration_s": 12.3}
    [EVAL] {"type": "e2e_latency", "duration_s": 120.5}
"""

from __future__ import annotations

import json
import re
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


# Regex to match [EVAL] lines
EVAL_LINE_RE = re.compile(r'\[EVAL\]\s*(\{.*\})\s*$')

# Regex to match run.py phase banner lines
PHASE_BANNER_RE = re.compile(
    r'\[.*?\].*?(Phase \d+|TVDiag|Phase \d+:\s*.*)\s+completed in\s+([\d.]+)s',
    re.IGNORECASE,
)

# Regex to match MEPFL top-5 root services
MEPFL_RE = re.compile(r'MEPFL top-5 root services:\s*(.*)')

# Regex to match root metrics
ROOT_METRIC_UNI_RE = re.compile(r'\[5A\] Univariate root metrics:\s*(.*)')
ROOT_METRIC_MULTI_RE = re.compile(r'\[5B\] Multivariate root metrics:\s*(.*)')

# Regex to match conflict scenario
CONFLICT_RE = re.compile(r'Conflict scenario:\s*(\w+)')

# Regex to match task parsed date/time
TASK_PARSED_RE = re.compile(r'Task parsed:\s*date=(\d+),\s*time=([\d-]+)')

# Regex to match trace anomalies count
TRACE_ANOMALY_RE = re.compile(r'Trace anomalies found:\s*(\d+)')

# Regex to match univariate anomaly count
UNI_ANOMALY_RE = re.compile(r'\[4A\] Univariate anomalies found:\s*(\d+)')

# Regex to match SPOT eta
SPOT_ETA_RE = re.compile(r'SPOT eta \(non-zero\):\s*(\d+)/(\d+)')

# Regex to match multivariate detection
MULTI_ANOMALY_RE = re.compile(r'Multivariate anomaly detected:\s*(\w+)')

# Regex to match causal graph
CAUSAL_GRAPH_RE = re.compile(r'Causal graph:\s*(\d+) nodes,\s*(\d+) edges')


class LogParser:
    """Parse run.py stdout/stderr logs into structured evaluation records."""

    def __init__(self, log_path: str = None, lines: List[str] = None):
        """
        Args:
            log_path: Path to the run.py log file (or stdout capture).
            lines:    Pre-loaded log lines (alternative to file path).
                      When provided, the file is not re-read.
        """
        self.log_path = log_path or '<memory>'
        self._init_lines: Optional[List[str]] = lines
        self.raw_lines: List[str] = []
        self.eval_records: List[dict] = []
        self.phase_latencies: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}

    def parse(self) -> 'LogParser':
        """Parse the log file and extract all evaluation data.

        Returns:
            self (for chaining).
        """
        if self._init_lines is not None:
            self.raw_lines = self._init_lines
        else:
            if not os.path.exists(self.log_path):
                raise FileNotFoundError(f"Log file not found: {self.log_path}")
            with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
                self.raw_lines = f.readlines()

        self._extract_eval_records()
        self._extract_phase_latencies()
        self._extract_metadata()

        return self

    # -----------------------------------------------------------------
    # Extract [EVAL] JSON records
    # -----------------------------------------------------------------

    def _extract_eval_records(self):
        """Extract structured [EVAL] JSON records."""
        for line in self.raw_lines:
            m = EVAL_LINE_RE.search(line.strip())
            if m:
                try:
                    record = json.loads(m.group(1))
                    self.eval_records.append(record)
                except json.JSONDecodeError:
                    continue

    # -----------------------------------------------------------------
    # Extract phase latencies from standard log lines
    # -----------------------------------------------------------------

    def _extract_phase_latencies(self):
        """Extract per-phase timing from log lines.

        Matches patterns like:
          [Phase 1] completed in 12.3s
          [Phase 4] completed in 45.6s
        """
        for line in self.raw_lines:
            # Try the _phase_done format
            m = re.search(
                r'\[(Phase \d+|TVDiag)\]\s+completed in\s+([\d.]+)s',
                line,
            )
            if m:
                phase = m.group(1)
                duration = float(m.group(2))
                self.phase_latencies[phase] = duration

    # -----------------------------------------------------------------
    # Extract metadata from standard log lines
    # -----------------------------------------------------------------

    def _extract_metadata(self):
        """Extract case metadata (date, time, predictions, etc.) from logs."""
        for line in self.raw_lines:
            line = line.strip()

            # Task date/time
            m = TASK_PARSED_RE.search(line)
            if m:
                self.metadata['date'] = m.group(1)
                self.metadata['time'] = m.group(2)

            # MEPFL top-5 root services
            m = MEPFL_RE.search(line)
            if m:
                raw = m.group(1)
                # Parse "(1)webservice1,(2)webservice2,..." format
                services = re.findall(r'\(\d+\)(\w+)', raw)
                self.metadata['root_services'] = services

            # Univariate root metrics
            m = ROOT_METRIC_UNI_RE.search(line)
            if m:
                self.metadata['root_metrics_uni'] = m.group(1)

            # Multivariate root metrics
            m = ROOT_METRIC_MULTI_RE.search(line)
            if m:
                self.metadata['root_metrics_multi'] = m.group(1)

            # Conflict scenario
            m = CONFLICT_RE.search(line)
            if m:
                self.metadata['conflict_scenario'] = m.group(1)

            # Trace anomaly count
            m = TRACE_ANOMALY_RE.search(line)
            if m:
                self.metadata['trace_anomaly_count'] = int(m.group(1))

            # Univariate anomaly count
            m = UNI_ANOMALY_RE.search(line)
            if m:
                self.metadata['univariate_anomaly_count'] = int(m.group(1))

            # SPOT eta
            m = SPOT_ETA_RE.search(line)
            if m:
                self.metadata['spot_eta_nonzero'] = int(m.group(1))
                self.metadata['spot_eta_total'] = int(m.group(2))

            # Multivariate anomaly
            m = MULTI_ANOMALY_RE.search(line)
            if m:
                self.metadata['multivariate_anomaly'] = m.group(1) == 'True'

            # Causal graph
            m = CAUSAL_GRAPH_RE.search(line)
            if m:
                self.metadata['causal_graph_nodes'] = int(m.group(1))
                self.metadata['causal_graph_edges'] = int(m.group(2))

    # -----------------------------------------------------------------
    # Convenience accessors
    # -----------------------------------------------------------------

    def get_predictions(self) -> List[str]:
        """Get ranked root cause service predictions (from MEPFL)."""
        return self.metadata.get('root_services', [])

    def get_root_metrics(self) -> str:
        """Get root cause metrics text."""
        return self.metadata.get('root_metrics_uni', '')

    def get_agent_outputs(self) -> Dict[str, str]:
        """Get agent outputs from [EVAL] records.

        Returns:
            dict mapping agent name -> output text.
        """
        outputs = {}
        for rec in self.eval_records:
            if rec.get('type') == 'agent_output':
                agent = rec.get('agent', 'unknown')
                outputs[agent] = rec.get('text', '')
        return outputs

    def get_e2e_latency(self) -> Optional[float]:
        """Get end-to-end latency from [EVAL] records."""
        for rec in self.eval_records:
            if rec.get('type') == 'e2e_latency':
                return rec.get('duration_s')
        # Fallback: sum of all phase latencies
        if self.phase_latencies:
            return sum(self.phase_latencies.values())
        return None

    def get_ground_truth(self) -> Optional[str]:
        """Get ground truth from [EVAL] records or metadata."""
        for rec in self.eval_records:
            if rec.get('type') == 'localization':
                return rec.get('ground_truth')
        return self.metadata.get('ground_truth_service')


def parse_batch_logs(
    log_dir: str,
    pattern: str = '*.log',
) -> List[LogParser]:
    """Parse multiple log files in a directory.

    Args:
        log_dir: Directory containing run log files.
        pattern: Glob pattern for log files.

    Returns:
        List of parsed LogParser objects.
    """
    import glob
    parsers = []
    for path in sorted(glob.glob(os.path.join(log_dir, pattern))):
        try:
            p = LogParser(path).parse()
            parsers.append(p)
        except Exception as e:
            print(f"[Warning] Failed to parse {path}: {e}")
    return parsers


def load_ground_truth_services(
    pkl_dir: str,
) -> Dict[str, Dict[str, str]]:
    """Load ground truth services from fault injection pkl files.

    Args:
        pkl_dir: Path to fault_injection_tracerank/ directory.

    Returns:
        dict mapping date (MMDD) -> {HH-MM: ground_truth_service}.
    """
    gt = {}
    if not os.path.isdir(pkl_dir):
        return gt

    for fname in sorted(os.listdir(pkl_dir)):
        if not fname.endswith('.pkl'):
            continue
        # Extract date from filename: fault_injection_list_2021-07-01.pkl
        m = re.search(r'(\d{4})-(\d{2})-(\d{2})', fname)
        if not m:
            continue
        mmdd = m.group(2) + m.group(3)

        pkl_path = os.path.join(pkl_dir, fname)
        try:
            with open(pkl_path, 'rb') as f:
                fi_list = pickle.load(f)
        except Exception:
            continue

        for fi in fi_list:
            if isinstance(fi, dict) and 'time' in fi and 'service' in fi:
                t = fi['time']
                if hasattr(t, 'strftime'):
                    hhmm = t.strftime('%H-%M')
                else:
                    hhmm = str(t)
                if mmdd not in gt:
                    gt[mmdd] = {}
                gt[mmdd][hhmm] = fi['service']

    return gt


def lookup_ground_truth(
    gt_map: Dict[str, Dict[str, str]],
    date_mmdd: str,
    time_hhmm: str,
    tolerance_minutes: int = 10,
) -> Optional[str]:
    """Look up ground truth service for a given date+time with fuzzy matching.

    The task prompt may specify a time (e.g. 11:50) that does not exactly
    match any fault injection time in the pkl (e.g. 11:54).  This function
    finds the closest fault injection within *tolerance_minutes*.

    Args:
        gt_map: Output of load_ground_truth_services().
        date_mmdd: Date string like '0701'.
        time_hhmm: Time string like '11-50' or '11:50'.
        tolerance_minutes: Maximum allowed difference in minutes.

    Returns:
        Ground truth service name, or None if no match found.
    """
    # Normalise separator
    time_hhmm = time_hhmm.replace(':', '-')

    # Try exact match first
    if date_mmdd in gt_map and time_hhmm in gt_map[date_mmdd]:
        return gt_map[date_mmdd][time_hhmm]

    # Fuzzy: find closest time within tolerance
    if date_mmdd not in gt_map:
        return None

    try:
        parts = time_hhmm.split('-')
        target_min = int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return None

    best_time = None
    best_diff = float('inf')
    for t_key in gt_map[date_mmdd]:
        try:
            h, m = t_key.split('-')
            diff = abs(int(h) * 60 + int(m) - target_min)
            if diff < best_diff:
                best_diff = diff
                best_time = t_key
        except (ValueError, IndexError):
            continue

    if best_time is not None and best_diff <= tolerance_minutes:
        return gt_map[date_mmdd][best_time]

    return None


def parse_multi_case_log(log_path: str) -> List[LogParser]:
    """Parse a single log file containing multiple concatenated cases.

    Experiments scripts redirect stdout from multiple ``run.py`` invocations
    into one file (e.g. ``experiments_localexpert.log``).  Each case starts
    with an ``[EVAL] {"type": "task_parsed", ...}`` record.  This function
    splits the file at those boundaries and returns one ``LogParser`` per case.

    If the file contains no ``task_parsed`` records (single-case or plain
    ChatChain log), a single-element list is returned.

    Args:
        log_path: Path to the multi-case experiments log file.

    Returns:
        List of parsed LogParser objects, one per case.
    """
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        all_lines = f.readlines()

    # Find the line index of each [EVAL] task_parsed record
    boundaries: List[int] = []
    for i, line in enumerate(all_lines):
        m = EVAL_LINE_RE.search(line.strip())
        if m:
            try:
                rec = json.loads(m.group(1))
                if rec.get('type') == 'task_parsed':
                    boundaries.append(i)
            except json.JSONDecodeError:
                pass

    if not boundaries:
        # Single case — delegate to normal LogParser
        return [LogParser(log_path=log_path).parse()]

    # Split into per-case segments
    parsers: List[LogParser] = []
    for j, start in enumerate(boundaries):
        end = boundaries[j + 1] if j + 1 < len(boundaries) else len(all_lines)
        segment = all_lines[start:end]
        p = LogParser(log_path=log_path, lines=segment)
        p.parse()
        parsers.append(p)

    return parsers
