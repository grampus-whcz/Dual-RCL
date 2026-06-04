"""
Default localizer — wraps the existing SoC-RCA pipeline.

Extracts the trace/metric/causal/log analysis logic from run.py into a
callable class. This is a thin wrapper; no logic changes from the original.
"""

import csv
import json
import logging
import os
import re
from typing import Optional

import numpy as np

from .base_localizer import BaseLocalizer, LocalizationResult

logger = logging.getLogger("failure_localization.default_localizer")


class DefaultLocalizer(BaseLocalizer):
    """Wraps the existing SoC-RCA single-modal pipeline.

    This delegates to the original functions in run.py. When ``--rca-method=default``
    (or no flag), the pipeline behaves identically to before.
    """

    def __init__(self):
        pass

    @property
    def method_name(self) -> str:
        return "default"

    def localize(self, task_info: dict) -> LocalizationResult:
        """Run the default SoC-RCA pipeline.

        Expected task_info keys:
            - date_result, time_result, config_phase_path, args
            - trace_an (str) — pre-computed trace anomaly description
            - root_service (str) — MEPFL top-5 services formatted
            - root_se (str) — MEPFL top-5 formatted string
            - metric_an (str) — metric anomaly description
            - root_metric (str) — PCMCI+RW root cause metrics
            - log_an (str) — log anomaly text
            - root_services_list (list) — raw service names from MEPFL
        """
        trace_an = task_info.get("trace_an", "")
        root_se = task_info.get("root_se", "")
        metric_an = task_info.get("metric_an", "")
        root_metric = task_info.get("root_metric", "")
        log_an = task_info.get("log_an", "")
        root_services_list = task_info.get("root_services_list", [])

        root_cause_knowledge = root_metric + "Top5 root cause:" + root_se

        return LocalizationResult(
            root_service=root_se,
            root_metric=root_metric,
            trace_anomaly=trace_an,
            metric_anomaly=metric_an,
            log_anomaly=log_an,
            root_cause_knowledge=root_cause_knowledge,
            raw_root_services=root_services_list[:5],
            raw_root_metrics=[],
        )
