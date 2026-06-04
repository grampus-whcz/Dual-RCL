"""
OmniAnomaly — Stochastic Recurrent Neural Network (KDD 2019).

GRU + VAE with reparameterization trick. Processes data sequentially
with hidden state. Loss = MSE + β · KLD.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .base_detector import BaseMultivariateDetector

logger = logging.getLogger("anomaly_detection.omnianomaly")


# ---------------------------------------------------------------------------
#  OmniAnomaly Model
# ---------------------------------------------------------------------------

class OmniAnomalyModel(nn.Module):
    """OmniAnomaly: GRU + VAE for multivariate anomaly detection."""

    def __init__(self, feats: int, lr: float = 0.002, beta: float = 0.01):
        super().__init__()
        self.name = "OmniAnomaly"
        self.lr = lr
        self.beta = beta
        self.n_feats = feats
        n_hidden = 32
        self.n_hidden = n_hidden
        n_latent = 8
        self.n_latent = n_latent

        self.lstm = nn.GRU(feats, n_hidden, 2)
        self.encoder = nn.Sequential(
            nn.Linear(n_hidden, n_hidden), nn.PReLU(),
            nn.Linear(n_hidden, n_hidden), nn.PReLU(),
            nn.Flatten(),
            nn.Linear(n_hidden, 2 * n_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_hidden), nn.PReLU(),
            nn.Linear(n_hidden, n_hidden), nn.PReLU(),
            nn.Linear(n_hidden, feats), nn.Sigmoid(),
        )

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.rand(
                2, 1, self.n_hidden, dtype=torch.float64, device=x.device
            )
        out, hidden = self.lstm(x.view(1, 1, -1), hidden)
        # Encode
        enc = self.encoder(out)
        mu, logvar = torch.split(enc, [self.n_latent, self.n_latent], dim=-1)
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Decode
        recon = self.decoder(z)
        return recon.view(-1), mu.view(-1), logvar.view(-1), hidden


# ---------------------------------------------------------------------------
#  OmniAnomaly Detector
# ---------------------------------------------------------------------------

class OmniAnomalyDetector(BaseMultivariateDetector):
    """OmniAnomaly-based multivariate anomaly detector.

    Note: OmniAnomaly processes data sequentially (no windowing).
    window_size is set to 1 internally.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 1,  # OmniAnomaly uses sequential processing
        batch_size: int = 128,
        lr: float = 0.002,
        beta: float = 0.01,
        device: str = "auto",
    ):
        super().__init__(n_features, window_size=1, batch_size=batch_size, lr=lr, device=device)
        self.beta = beta

    def _build_model(self) -> nn.Module:
        return OmniAnomalyModel(
            feats=self.n_features, lr=self.lr, beta=self.beta
        ).double()

    def _compute_feature_scores(self, data: torch.Tensor) -> np.ndarray:
        """Sequential forward pass, return per-timestep per-feature MSE."""
        criterion = nn.MSELoss(reduction="none")
        all_scores = []
        with torch.no_grad():
            hidden = None
            for i in range(len(data)):
                d = data[i].to(self.device)
                y_pred, _, _, hidden = self.model(d, hidden)
                loss = criterion(y_pred, d)  # (N,)
                all_scores.append(loss.detach().cpu().numpy())
        return np.array(all_scores)  # (T, N)

    def train(self, train_data: np.ndarray, epochs: int = 5) -> None:
        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        criterion = nn.MSELoss(reduction="mean")

        data = torch.tensor(train_data, dtype=torch.float64).to(self.device)

        logger.info(
            f"Training OmniAnomaly: feats={self.n_features}, "
            f"epochs={epochs}, samples={len(train_data)}, device={self.device}"
        )

        for epoch in range(epochs):
            self.model.train()
            mses, klds = [], []
            hidden = None
            for i in range(len(data)):
                d = data[i]
                # Detach hidden state to prevent backward through previous steps
                if hidden is not None:
                    hidden = hidden.detach()
                y_pred, mu, logvar, hidden = self.model(d, hidden)
                MSE = criterion(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = MSE + self.beta * KLD
                mses.append(MSE.item())
                klds.append((self.beta * KLD).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            logger.info(
                f"  Epoch {epoch + 1}/{epochs}  "
                f"MSE={np.mean(mses):.6f}  KLD={np.mean(klds):.6f}"
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
        result["method"] = "OmniAnomaly"
        return result
