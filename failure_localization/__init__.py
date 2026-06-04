"""
failure_localization — Configurable root cause localization for SoC-RCA.

Provides a unified interface to multiple root cause localization strategies.

Available methods:
  - ``default`` — existing PCMCI+RandomWalk + MEPFL pipeline
  - ``tvdig``   — TVDiag multimodal GNN (metric+trace+log joint analysis)

Quick start:
>>> from failure_localization import create_localizer, AVAILABLE_METHODS
>>> localizer = create_localizer("tvdig", model_dir="./tvdig_checkpoint")
>>> result = localizer.localize(task_info)
"""

from .base_localizer import BaseLocalizer, LocalizationResult
from .localizer_factory import create_localizer, AVAILABLE_METHODS

__all__ = [
    "BaseLocalizer",
    "LocalizationResult",
    "create_localizer",
    "AVAILABLE_METHODS",
]
