"""
USAD — Unsupervised Anomaly Detection via Adversarial Training (KDD 2020).

Dual-autoencoder architecture: encoder + decoder1 + decoder2.
Training uses a two-phase loss that amplifies anomaly scores.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_detector import BaseMultivariateDetector

logger = logging.getLogger("anomaly_detection.usad")


# ---------------------------------------------------------------------------
#  USAD Model
# ---------------------------------------------------------------------------

class USADModel(nn.Module):
    """USAD: dual autoencoder with adversarial training."""

    def __init__(self, feats: int, lr: float = 0.0001, window_size: int = 5):
        super().__init__()
        self.name = "USAD"
        self.lr = lr
        self.n_feats = feats
        self.n_window = window_size
        self.n = feats * window_size
        n_hidden = 16
        n_latent = 5

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, n_hidden), nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(True),
            nn.Linear(n_hidden, n_latent), nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(n_latent, n_hidden), nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(True),
            nn.Linear(n_hidden, self.n), nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(n_latent, n_hidden), nn.ReLU(True),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(True),
            nn.Linear(n_hidden, self.n), nn.Sigmoid(),
        )

    def forward(self, g):
        z = self.encoder(g.view(1, -1))
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)


# ---------------------------------------------------------------------------
#  USAD Detector
# ---------------------------------------------------------------------------

class USADDetector(BaseMultivariateDetector):
    """USAD-based multivariate anomaly detector."""

    def __init__(
        self,
        n_features: int,
        window_size: int = 5,
        batch_size: int = 128,
        lr: float = 0.0001,
        device: str = "auto",
    ):
        super().__init__(n_features, window_size, batch_size, lr, device)

    def _build_model(self) -> nn.Module:
        return USADModel(
            feats=self.n_features, lr=self.lr, window_size=self.window_size
        ).double()

    def _compute_feature_scores(self, data: torch.Tensor) -> np.ndarray:
        """Compute per-timestep per-feature reconstruction error.

        Uses weighted combination: 0.1 * |ae1 - x| + 0.9 * |ae2ae1 - x|,
        matching the original USAD test-time scoring.
        """
        feats = self.n_features
        windows_flat = self._to_windows_flat(data)  # (T, W*N)

        all_scores = []
        with torch.no_grad():
            for d in windows_flat:
                d = d.to(self.device)
                ae1, ae2, ae2ae1 = self.model(d)
                # Last window_size * n_feats elements
                target = d
                loss = 0.1 * (ae1 - target) ** 2 + 0.9 * (ae2ae1 - target) ** 2
                # Reshape to (W, N) and take last row
                loss_matrix = loss.view(self.window_size, feats)
                all_scores.append(loss_matrix[-1].detach().cpu().numpy())

        return np.array(all_scores)  # (T, N)

    def train(self, train_data: np.ndarray, epochs: int = 5) -> None:
        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        criterion = nn.MSELoss(reduction="none")

        data = torch.tensor(train_data, dtype=torch.float64).to(self.device)
        windows_flat = self._to_windows_flat(data)

        dataset = TensorDataset(windows_flat, windows_flat)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        logger.info(
            f"Training USAD: feats={self.n_features}, window={self.window_size}, "
            f"epochs={epochs}, samples={len(train_data)}, device={self.device}"
        )

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            n = epoch + 1
            for d, _ in dataloader:
                d = d.to(self.device)
                l1s_batch, l2s_batch = [], []
                for sample in d:
                    ae1, ae2, ae2ae1 = self.model(sample)
                    l1 = (1 / n) * criterion(ae1, sample) + (1 - 1 / n) * criterion(ae2ae1, sample)
                    l2 = (1 / n) * criterion(ae2, sample) - (1 - 1 / n) * criterion(ae2ae1, sample)
                    l1s_batch.append(torch.mean(l1))
                    l2s_batch.append(torch.mean(l2))
                loss = torch.stack(l1s_batch).mean() + torch.stack(l2s_batch).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            avg_loss = np.mean(epoch_losses)
            logger.info(f"  Epoch {epoch + 1}/{epochs}  loss={avg_loss:.6f}")

        self.model.eval()

        # Calibrate threshold
        train_scores = self._compute_feature_scores(data)
        self._calibrate_threshold(train_scores)

    def detect(self, test_data: np.ndarray) -> dict:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        data = torch.tensor(test_data, dtype=torch.float64).to(self.device)
        feature_scores = self._compute_feature_scores(data)
        result = self._build_result(feature_scores)
        result["method"] = "USAD"
        return result
