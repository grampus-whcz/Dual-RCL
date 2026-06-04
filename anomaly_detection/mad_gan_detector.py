"""
MAD_GAN — Multivariate Anomaly Detection with GAN (ICANN 2019).

Generator + Discriminator. Loss = BCE (discriminator) + MSE (generator).
Reconstruction error from the generator output is used as anomaly score.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_detector import BaseMultivariateDetector

logger = logging.getLogger("anomaly_detection.mad_gan")


# ---------------------------------------------------------------------------
#  MAD_GAN Model
# ---------------------------------------------------------------------------

class MADGANModel(nn.Module):
    """MAD_GAN: Generator-Discriminator for multivariate anomaly detection."""

    def __init__(self, feats: int, lr: float = 0.0001, window_size: int = 5):
        super().__init__()
        self.name = "MAD_GAN"
        self.lr = lr
        self.n_feats = feats
        self.n_window = window_size
        self.n = feats * window_size
        n_hidden = 16

        self.generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, self.n), nn.Sigmoid(),
        )
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        z = self.generator(g.view(1, -1))
        real_score = self.discriminator(g.view(1, -1))
        fake_score = self.discriminator(z.view(1, -1))
        return z.view(-1), real_score.view(-1), fake_score.view(-1)


# ---------------------------------------------------------------------------
#  MAD_GAN Detector
# ---------------------------------------------------------------------------

class MADDetector(BaseMultivariateDetector):
    """MAD_GAN-based multivariate anomaly detector."""

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
        return MADGANModel(
            feats=self.n_features, lr=self.lr, window_size=self.window_size
        ).double()

    def _compute_feature_scores(self, data: torch.Tensor) -> np.ndarray:
        """Use generator reconstruction error as anomaly score."""
        feats = self.n_features
        windows_flat = self._to_windows_flat(data)

        all_scores = []
        with torch.no_grad():
            for d in windows_flat:
                d = d.to(self.device)
                z, _, _ = self.model(d)
                loss = (z - d) ** 2
                loss_matrix = loss.view(self.window_size, feats)
                all_scores.append(loss_matrix[-1].detach().cpu().numpy())

        return np.array(all_scores)  # (T, N)

    def train(self, train_data: np.ndarray, epochs: int = 5) -> None:
        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

        mse_loss = nn.MSELoss(reduction="none")
        bce_loss = nn.BCELoss(reduction="mean")
        mse_mean = nn.MSELoss(reduction="mean")

        # Label smoothing
        real_label = torch.tensor([0.9], dtype=torch.float64, device=self.device)
        fake_label = torch.tensor([0.1], dtype=torch.float64, device=self.device)

        data = torch.tensor(train_data, dtype=torch.float64).to(self.device)
        windows_flat = self._to_windows_flat(data)

        dataset = TensorDataset(windows_flat, windows_flat)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        logger.info(
            f"Training MAD_GAN: feats={self.n_features}, window={self.window_size}, "
            f"epochs={epochs}, samples={len(train_data)}, device={self.device}"
        )

        for epoch in range(epochs):
            self.model.train()
            mses, gls, dls = [], [], []
            for d, _ in dataloader:
                for sample in d:
                    sample = sample.to(self.device)

                    # Train discriminator
                    self.model.discriminator.zero_grad()
                    _, real, fake = self.model(sample)
                    dl = bce_loss(real, real_label) + bce_loss(fake, fake_label)
                    dl.backward()
                    self.model.generator.zero_grad()
                    optimizer.step()

                    # Train generator
                    z, _, fake = self.model(sample)
                    mse = mse_mean(z, sample)
                    gl = bce_loss(fake, real_label)
                    tl = gl + mse
                    tl.backward()
                    self.model.discriminator.zero_grad()
                    optimizer.step()

                    mses.append(mse.item())
                    gls.append(gl.item())
                    dls.append(dl.item())

            scheduler.step()
            logger.info(
                f"  Epoch {epoch + 1}/{epochs}  "
                f"MSE={np.mean(mses):.6f}  G={np.mean(gls):.6f}  D={np.mean(dls):.6f}"
            )

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
        result["method"] = "MAD_GAN"
        return result
