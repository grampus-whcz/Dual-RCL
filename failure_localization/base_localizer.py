"""
Abstract base class for root cause localization strategies.

All localizers return a ``LocalizationResult`` whose string fields can be
directly injected into ``PhaseConfig.json`` for downstream LLM agents.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger("failure_localization")


@dataclass
class LocalizationResult:
    """Unified result from any root cause localization method.

    String fields match the format expected by PhaseConfig.json knowledge
    injection in the SoC-RCA ChatChain pipeline.
    """

    # Top-5 root cause services as formatted string
    root_service: str  # "(1)webservice1,(2)redisservice2,..."

    # Root cause metrics as formatted string
    root_metric: str  # "Top 5 root cause metrics is:(1)...,(2)..."

    # Anomaly description for TraceAnalysis phase prompt
    trace_anomaly: str

    # Anomaly description for MetricAnalysis phase prompt
    metric_anomaly: str

    # Anomaly description for LogAnalysis phase prompt
    log_anomaly: str

    # Combined root cause knowledge for RootCauseAnalysis phase prompt
    root_cause_knowledge: str

    # Ordered list of root cause service names (raw, for programmatic use)
    raw_root_services: List[str] = field(default_factory=list)

    # Ordered list of root cause metric names (raw, for programmatic use)
    raw_root_metrics: List[str] = field(default_factory=list)

    # Raw per-node root-cause scores from the localizer (for confidence estimation)
    raw_root_scores: np.ndarray = field(default=None)


class BaseLocalizer(ABC):
    """Abstract base class for root cause localization strategies."""

    @abstractmethod
    def localize(self, task_info: dict) -> LocalizationResult:
        """Run root cause localization.

        Parameters
        ----------
        task_info : dict
            Must contain:
              - date_result : str  (e.g. "0704")
              - time_result : str  (e.g. "00-50")
              - config_phase_path : pathlib.Path
              - args : argparse.Namespace
            May contain additional data depending on the localizer.

        Returns
        -------
        LocalizationResult
        """
        ...

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return identifier string for this localizer."""
        ...
