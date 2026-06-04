"""
Factory function for root cause localization strategies.
"""

import logging
from typing import Optional

from .base_localizer import BaseLocalizer
from .default_localizer import DefaultLocalizer

logger = logging.getLogger("failure_localization.factory")

AVAILABLE_METHODS = ["default", "tvdig"]


def create_localizer(
    method: str = "default",
    model_dir: Optional[str] = None,
    device: str = "auto",
    **kwargs,
) -> BaseLocalizer:
    """Create a root cause localizer by name.

    Args:
        method: ``"default"`` (PCMCI+RW + MEPFL) or ``"tvdig"`` (TVDiag GNN).
        model_dir: Path to TVDiag model checkpoint (required for ``tvdig``).
        device: ``"auto"`` | ``"cpu"`` | ``"cuda"``.
        **kwargs: Additional parameters.

    Returns:
        BaseLocalizer instance.
    """
    method = method.lower().strip()

    if method == "default":
        from .default_localizer import DefaultLocalizer
        return DefaultLocalizer()

    elif method == "tvdig":
        from .tvdig_localizer import TVDiagLocalizer
        from .tvdig_config import TVDiagConfig

        config = kwargs.get("config", TVDiagConfig())
        if model_dir is None:
            raise ValueError("--tvdig-model is required when --rca-method=tvdig")

        return TVDiagLocalizer(
            model_dir=model_dir,
            config=config,
            device=device,
        )

    else:
        raise ValueError(
            f"Unknown RCA method '{method}'. Available: {AVAILABLE_METHODS}"
        )
