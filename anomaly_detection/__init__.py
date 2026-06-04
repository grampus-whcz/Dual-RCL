"""
anomaly_detection — Configurable multivariate anomaly detection for SoC-RCA.

Provides a unified interface to multiple deep-learning anomaly detection
methods.  All detectors share the same ``train()`` / ``detect()`` API.

Available methods
-----------------
- ``tranad``       — TranAD (VLDB 2022, Transformer)
- ``usad``         — USAD (KDD 2020, dual autoencoder)
- ``omnianomaly``  — OmniAnomaly (KDD 2019, GRU + VAE)
- ``mad_gan``      — MAD_GAN (ICANN 2019, GAN)
- ``mscred``       — MSCRED (AAAI 2019, ConvLSTM)
- ``gdn``          — GDN (AAAI 2021, graph attention)
- ``mtad_gat``     — MTAD_GAT (ICDM 2020, graph attention + GRU)

Quick start
-----------
>>> from anomaly_detection import create_detector, AVAILABLE_METHODS
>>> detector = create_detector("usad", n_features=33)
>>> detector.train(train_data, epochs=5)
>>> results = detector.detect(test_data)
>>> print(results["is_anomalous"])
"""

from .base_detector import BaseMultivariateDetector
from .detector_factory import create_detector, AVAILABLE_METHODS

__all__ = [
    "BaseMultivariateDetector",
    "create_detector",
    "AVAILABLE_METHODS",
]
