"""
MTAD_GAT — Multivariate Time Series Anomaly Detection with
Graph Attention Network (ICDM 2020).

Pure PyTorch reimplementation (no DGL dependency).
Uses dual graph attention (feature + time) + GRU for temporal modelling.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_detector import BaseMultivariateDetector

logger = logging.getLogger("anomaly_detection.mtad_gat")


# ---------------------------------------------------------------------------
#  Pure-PyTorch Graph Attention (reused pattern from gdn_detector.py)
# ---------------------------------------------------------------------------

class SimpleGAT(nn.Module):
    """Single-head graph attention layer (complete graph)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(2 * out_dim, 1))
        nn.init.xavier_uniform_(self.a)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h):
        """
        h: (N, in_dim)
        returns: (N, out_dim)
        """
        Wh = self.W(h)  # (N, out_dim)
        N = Wh.size(0)
        h_cat = torch.cat([
            Wh.unsqueeze(1).expand(-1, N, -1),
            Wh.unsqueeze(0).expand(N, -1, -1),
        ], dim=-1)
        e = self.leaky_relu(torch.matmul(h_cat, self.a).squeeze(-1))
        alpha = torch.softmax(e, dim=1)
        out = torch.matmul(alpha, Wh)
        return out


# ---------------------------------------------------------------------------
#  MTAD_GAT Model
# ---------------------------------------------------------------------------

class MTADGATModel(nn.Module):
    """MTAD_GAT: dual graph attention (feature + time) + GRU.

    Reimplemented with pure PyTorch (no DGL).
    """

    def __init__(self, feats: int, lr: float = 0.0001, window_size: int = None):
        super().__init__()
        self.name = "MTAD_GAT"
        self.lr = lr
        self.n_feats = feats
        self.n_window = window_size if window_size is not None else feats
        self.n_hidden = feats * feats

        # Feature graph attention: operates on feature dimension
        self.feature_gat = SimpleGAT(feats, 1)
        # Time graph attention: operates on time dimension
        self.time_gat = SimpleGAT(feats, 1)
        # GRU for temporal modelling
        gru_input = (feats + 1) * feats * 3
        self.gru = nn.GRU(gru_input, feats * feats, 1)

    def forward(self, data, hidden=None):
        """
        data: (W*N,) flattened window
        hidden: GRU hidden state
        returns: (reconstruction, hidden)
        """
        wn = data.view(self.n_window, self.n_feats)  # (W, N)

        # Feature GAT: add a zero row, apply attention
        data_r = torch.cat([torch.zeros(1, self.n_feats, device=data.device), wn], dim=0)  # (W+1, N)
        feat_r = self.feature_gat(data_r)  # (W+1, 1)

        # Time GAT: transpose, add zero row, apply attention
        data_t = torch.cat([torch.zeros(1, self.n_feats, device=data.device), wn.t()], dim=0)  # (N+1, W)
        time_r = self.time_gat(data_t)  # (N+1, 1)

        # Prepare for GRU
        data_exp = torch.cat([torch.zeros(1, self.n_feats, device=data.device), wn], dim=0)
        data_exp = data_exp.view(self.n_window + 1, self.n_feats, 1)

        # Broadcast and concatenate
        feat_r_exp = feat_r[:self.n_window + 1].view(self.n_window + 1, 1, 1).expand(-1, self.n_feats, -1)
        time_r_exp = time_r[:self.n_window + 1].view(self.n_window + 1, 1, 1).expand(-1, self.n_feats, -1)

        x = torch.cat([data_exp, feat_r_exp, time_r_exp], dim=2)  # (W+1, N, 3)
        x = x.view(1, 1, -1)  # (1, 1, (W+1)*N*3)

        if hidden is None:
            hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64, device=data.device)
        x, h = self.gru(x, hidden)
        return x.view(-1), h


# ---------------------------------------------------------------------------
#  MTAD_GAT Detector
# ---------------------------------------------------------------------------

class MTADGATDetector(BaseMultivariateDetector):
    """MTAD_GAT-based multivariate anomaly detector."""

    def __init__(
        self,
        n_features: int,
        window_size: int = None,
        batch_size: int = 128,
        lr: float = 0.0001,
        device: str = "auto",
    ):
        ws = window_size if window_size is not None else n_features
        super().__init__(n_features, window_size=ws, batch_size=batch_size, lr=lr, device=device)

    def _build_model(self) -> nn.Module:
        return MTADGATModel(
            feats=self.n_features, lr=self.lr, window_size=self.window_size
        ).double()

    def _compute_feature_scores(self, data: torch.Tensor) -> np.ndarray:
        """Per-timestep per-feature reconstruction error."""
        feats = self.n_features
        windows_flat = self._to_windows_flat(data)

        all_scores = []
        with torch.no_grad():
            for d in windows_flat:
                d = d.to(self.device)
                x, _ = self.model(d, None)
                # x shape: (n_hidden,) = (feats*feats,)
                # We take the first feats elements as per-feature scores
                target = d  # (W*N,)
                loss = (x[:feats] - target[-feats:]) ** 2
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
        windows_flat = self._to_windows_flat(data)

        logger.info(
            f"Training MTAD_GAT: feats={self.n_features}, window={self.window_size}, "
            f"epochs={epochs}, samples={len(train_data)}, device={self.device}"
        )

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            h = None
            for i, d in enumerate(windows_flat):
                d = d.to(self.device)
                # Detach hidden state to prevent backward through previous steps
                if h is not None:
                    h = h.detach()
                x, h = self.model(d, h)
                feats = self.n_features
                target = d[-feats:]
                pred = x[:feats]
                loss = criterion(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            avg_loss = np.mean(epoch_losses)
            logger.info(f"  Epoch {epoch + 1}/{epochs}  MSE={avg_loss:.6f}")

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
        result["method"] = "MTAD_GAT"
        return result
