"""
Abstract base class for multivariate anomaly detectors.

All concrete detectors must implement ``train()`` and ``detect()`` with the
same interface so they can be used interchangeably in the SoC-RCA pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger("anomaly_detection")


class BaseMultivariateDetector(ABC):
    """Base class for multivariate time-series anomaly detectors.

    Parameters
    ----------
    n_features : int
        Number of features (metrics) in the input data.
    window_size : int
        Sliding window length used by the model.
    batch_size : int
        Mini-batch size for training.
    lr : float
        Learning rate.
    device : str
        ``"auto"`` | ``"cpu"`` | ``"cuda"``.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 5,
        batch_size: int = 128,
        lr: float = 0.001,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.n_features = n_features
        self.window_size = window_size
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.train_threshold: Optional[float] = None

    # ------------------------------------------------------------------
    #  Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        """Construct and return the PyTorch model."""
        ...

    @abstractmethod
    def train(self, train_data: np.ndarray, epochs: int = 5) -> None:
        """Train the detector on normal-period data.

        Parameters
        ----------
        train_data : np.ndarray of shape (T, N)
        epochs : int
        """
        ...

    @abstractmethod
    def detect(self, test_data: np.ndarray) -> dict:
        """Run anomaly detection on test data.

        Parameters
        ----------
        test_data : np.ndarray of shape (T, N)

        Returns
        -------
        dict with keys:
            anomaly_scores   : (T,) per-timestep mean reconstruction error
            feature_scores   : (T, N) per-timestep per-feature error
            anomalies        : (T,) bool
            anomaly_features : list[list[int]]
            feature_ranking  : list[(idx, score)]
            threshold        : float
            is_anomalous     : bool
        """
        ...

    # ------------------------------------------------------------------
    #  Shared helpers
    # ------------------------------------------------------------------

    def _to_windows(self, data: torch.Tensor) -> torch.Tensor:
        """Convert (T, N) tensor to (T, W, N) sliding windows."""
        w_size = self.window_size
        windows = []
        for i in range(len(data)):
            if i >= w_size:
                w = data[i - w_size : i]
            else:
                w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
            windows.append(w)
        return torch.stack(windows).to(self.device)

    def _to_windows_flat(self, data: torch.Tensor) -> torch.Tensor:
        """Convert (T, N) tensor to (T, W*N) flattened sliding windows."""
        windows = self._to_windows(data)  # (T, W, N)
        return windows.view(len(data), -1)  # (T, W*N)

    def _calibrate_threshold(self, feature_scores: np.ndarray) -> None:
        """Set training threshold at 99th percentile of mean per-step score."""
        self.train_threshold = float(np.percentile(feature_scores.mean(axis=1), 99))
        logger.info(f"  Training threshold (p99) = {self.train_threshold:.6f}")

    def _build_result(self, feature_scores: np.ndarray) -> dict:
        """Build the standard result dict from per-feature scores."""
        anomaly_scores = feature_scores.mean(axis=1)  # (T,)
        anomalies = anomaly_scores > self.train_threshold

        # Per-feature thresholds (p97 for adaptive attribution)
        feat_thresh = np.percentile(feature_scores, 97, axis=0)
        anomaly_features: List[List[int]] = []
        for t in range(feature_scores.shape[0]):
            if anomalies[t]:
                af = np.where(feature_scores[t] > feat_thresh)[0].tolist()
                anomaly_features.append(af)
            else:
                anomaly_features.append([])

        # Feature ranking by total contribution
        total_contribution = feature_scores.sum(axis=0)
        feature_ranking = sorted(
            enumerate(total_contribution), key=lambda x: x[1], reverse=True
        )

        return {
            "anomaly_scores": anomaly_scores,
            "feature_scores": feature_scores,
            "anomalies": anomalies,
            "anomaly_features": anomaly_features,
            "feature_ranking": feature_ranking,
            "threshold": self.train_threshold,
            "is_anomalous": bool(anomalies.any()),
        }
