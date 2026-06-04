"""
Detector Factory — Registry and factory function for multivariate anomaly
detection methods.

Usage::

    from anomaly_detection import create_detector

    detector = create_detector("usad", n_features=33)
    detector.train(train_data, epochs=5)
    results = detector.detect(test_data)
"""

import logging
from typing import Dict, Optional, Type

from .base_detector import BaseMultivariateDetector

logger = logging.getLogger("anomaly_detection.factory")

# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------

_AVAILABLE: Dict[str, str] = {
    "tranad":       "anomaly_detection.tranad_detector.TranADDetectorAdapter",
    "usad":         "anomaly_detection.usad_detector.USADDetector",
    "omnianomaly":  "anomaly_detection.omnianomaly_detector.OmniAnomalyDetector",
    "mad_gan":      "anomaly_detection.mad_gan_detector.MADGANModel",  # placeholder, fixed below
    "mscred":       "anomaly_detection.mscred_detector.MSCREDDetector",
    "gdn":          "anomaly_detection.gdn_detector.GDNDetector",
    "mtad_gat":     "anomaly_detection.mtad_gat_detector.MTADGATDetector",
}

AVAILABLE_METHODS = sorted(_AVAILABLE.keys())

# ---------------------------------------------------------------------------
#  Lazy import helper
# ---------------------------------------------------------------------------

_CLASS_CACHE: Dict[str, Type[BaseMultivariateDetector]] = {}


def _import_class(dotted_path: str) -> Type[BaseMultivariateDetector]:
    """Import a class from its dotted module path (lazy import)."""
    if dotted_path in _CLASS_CACHE:
        return _CLASS_CACHE[dotted_path]

    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    _CLASS_CACHE[dotted_path] = cls
    return cls


# Fix the MAD_GAN entry
_AVAILABLE["mad_gan"] = "anomaly_detection.mad_gan_detector.MADDetector"


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

def create_detector(
    method: str,
    n_features: int,
    window_size: Optional[int] = None,
    batch_size: int = 128,
    lr: Optional[float] = None,
    device: str = "auto",
    **kwargs,
) -> BaseMultivariateDetector:
    """Create a multivariate anomaly detector by name.

    Parameters
    ----------
    method : str
        One of: ``tranad``, ``usad``, ``omnianomaly``, ``mad_gan``,
        ``mscred``, ``gdn``, ``mtad_gat``.
    n_features : int
        Number of input features (metrics).
    window_size : int, optional
        Sliding window length.  If ``None``, the model default is used.
    batch_size : int
        Mini-batch size for training.
    lr : float, optional
        Learning rate.  If ``None``, the model default is used.
    device : str
        ``"auto"`` | ``"cpu"`` | ``"cuda"``.
    **kwargs
        Additional model-specific parameters (e.g. ``beta`` for OmniAnomaly).

    Returns
    -------
    BaseMultivariateDetector
        An initialised (but not yet trained) detector.

    Raises
    ------
    ValueError
        If ``method`` is not in the registry.
    """
    method = method.lower().strip()
    if method not in _AVAILABLE:
        raise ValueError(
            f"Unknown anomaly detection method '{method}'. "
            f"Available: {AVAILABLE_METHODS}"
        )

    dotted_path = _AVAILABLE[method]
    cls = _import_class(dotted_path)

    # Build kwargs, filtering out None values so the class defaults kick in
    ctor_kwargs: dict = {"n_features": n_features, "device": device}
    if window_size is not None:
        ctor_kwargs["window_size"] = window_size
    if batch_size is not None:
        ctor_kwargs["batch_size"] = batch_size
    if lr is not None:
        ctor_kwargs["lr"] = lr
    ctor_kwargs.update(kwargs)

    logger.info(f"Creating detector: method={method}, class={cls.__name__}")
    return cls(**ctor_kwargs)
