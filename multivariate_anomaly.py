"""
Multivariate Time Series Anomaly Detection for SoC-RCA.

Integrates with the SoC-RCA pipeline for microservice fault diagnosis.
Provides joint detection across multiple metric dimensions to capture
correlation disruptions that univariate methods cannot detect.

Supports multiple detection methods via the ``anomaly_detection`` package:
  TranAD, USAD, OmniAnomaly, MAD_GAN, MSCRED, GDN, MTAD_GAT

Backward compatible — calling without ``method`` defaults to TranAD.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

# Import the factory-based detection system
from anomaly_detection import create_detector, AVAILABLE_METHODS
from anomaly_detection.shared import metrics_to_services, build_description

logger = logging.getLogger("multivariate_anomaly")


# ---------------------------------------------------------------------------
#  Keep the original TranAD-specific classes for backward compatibility
#  (code that imports TranADDetector / TranADModel directly still works)
# ---------------------------------------------------------------------------

from anomaly_detection.tranad_detector import TranADDetectorAdapter as TranADDetector
from anomaly_detection.tranad_detector import _TranADModel as TranADModel


# ---------------------------------------------------------------------------
#  High-level helper: run on metric data loaded from microcause Excel
# ---------------------------------------------------------------------------

def run_multivariate_detection(
    data: np.ndarray,
    data_head: List[str],
    n_init: Optional[int] = None,
    epochs: int = 5,
    lr: float = None,
    method: str = "tranad",
    window_size: Optional[int] = None,
) -> dict:
    """Run multivariate anomaly detection on metric time-series.

    Splits *data* into a training (normal) half and a test (fault) half,
    trains the selected detector on the normal half, and runs detection
    on the fault half.

    Args:
        data:        (T, N) numpy array from ``load()``.
        data_head:   list of N metric names.
        n_init:      index at which to split train / test (default: 50%).
        epochs:      training epochs.
        lr:          learning rate. ``None`` uses the model's default.
        method:      detection method name (default: ``"tranad"``).
                     One of: ``tranad``, ``usad``, ``omnianomaly``,
                     ``mad_gan``, ``mscred``, ``gdn``, ``mtad_gat``.
        window_size: sliding window size. ``None`` uses model default.

    Returns:
        dict with detection results plus human-readable descriptions.
    """
    method = method.lower().strip()
    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {AVAILABLE_METHODS}"
        )

    if n_init is None:
        n_init = int(0.5 * len(data))

    n_features = data.shape[1]

    # Normalise (z-score) so that all metrics are on the same scale
    mu = data[:n_init].mean(axis=0, keepdims=True)
    sigma = data[:n_init].std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1e-6
    data_norm = (data - mu) / sigma

    train_data = data_norm[:n_init]
    test_data = data_norm[n_init:]

    # Create detector via factory
    factory_kwargs = {}
    if lr is not None:
        factory_kwargs["lr"] = lr
    if window_size is not None:
        factory_kwargs["window_size"] = window_size

    detector = create_detector(
        method=method,
        n_features=n_features,
        **factory_kwargs,
    )

    # Train & detect
    detector.train(train_data, epochs=epochs)
    results = detector.detect(test_data)

    # Enrich with metadata
    results["data_head"] = data_head
    results["n_init"] = n_init
    results["train_size"] = n_init
    results["test_size"] = len(data) - n_init
    results["method"] = method
    results["feature_ranking_named"] = [
        (data_head[idx], float(score)) for idx, score in results["feature_ranking"]
    ]
    results["top5_metrics"] = [
        data_head[idx] for idx, _ in results["feature_ranking"][:5]
    ]
    results["top5_services"] = metrics_to_services(results["top5_metrics"])

    # Natural-language description
    results["description"] = build_description(results, data_head)

    return results


# ---------------------------------------------------------------------------
#  Helpers (kept for backward compatibility with direct imports)
# ---------------------------------------------------------------------------

def _metrics_to_services(metrics: List[str]) -> List[str]:
    """Backward-compatible wrapper."""
    return metrics_to_services(metrics)


def _build_description(results: dict, data_head: List[str]) -> str:
    """Backward-compatible wrapper."""
    return build_description(results, data_head)
