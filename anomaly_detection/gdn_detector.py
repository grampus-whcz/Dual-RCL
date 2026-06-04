"""
GDN — Graph Deep Network for Anomaly Detection (AAAI 2021).

Pure PyTorch reimplementation (no DGL dependency).
Uses multi-head attention on a complete graph over features,
with Bahdanau-style temporal attention weighting.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_detector import BaseMultivariateDetector

logger = logging.getLogger("anomaly_detection.gdn")


# ---------------------------------------------------------------------------
#  Pure-PyTorch Multi-Head Graph Attention (replaces DGL GATConv)
# ---------------------------------------------------------------------------

class GraphAttentionHead(nn.Module):
    """Single attention head for graph attention."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h):
        """
        h: (N, in_features) — node features
        returns: (N, out_features)
        """
        Wh = self.W(h)  # (N, out_features)
        N = Wh.size(0)
        h_cat = torch.cat([
            Wh.unsqueeze(1).expand(-1, N, -1),
            Wh.unsqueeze(0).expand(N, -1, -1),
        ], dim=-1)  # (N, N, 2*out_features)
        e = self.leaky_relu(torch.matmul(h_cat, self.a).squeeze(-1))  # (N, N)
        alpha = torch.softmax(e, dim=1)
        return torch.matmul(alpha, Wh)  # (N, out_features)


class MultiHeadGraphAttention(nn.Module):
    """Multi-head graph attention over a complete graph.

    Concatenates outputs from multiple heads, matching DGL GATConv behaviour.
    """

    def __init__(self, in_features: int, out_per_head: int, num_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([
            GraphAttentionHead(in_features, out_per_head)
            for _ in range(num_heads)
        ])

    def forward(self, h):
        """
        h: (N, in_features)
        returns: (N, out_per_head * num_heads)
        """
        return torch.cat([head(h) for head in self.heads], dim=-1)


# ---------------------------------------------------------------------------
#  GDN Model
# ---------------------------------------------------------------------------

class GDNModel(nn.Module):
    """GDN: Graph Deep Network for multivariate anomaly detection.

    Reimplemented with pure PyTorch (no DGL).

    Architecture:
      1. Temporal attention (Bahdanau-style) aggregates time steps -> (N, 1)
      2. Multi-head graph attention (feats heads) -> (N, feats)
      3. FCN: (N, feats) -> (N, W) per-node reconstruction
    """

    def __init__(self, feats: int, lr: float = 0.0001, window_size: int = 5):
        super().__init__()
        self.name = "GDN"
        self.lr = lr
        self.n_feats = feats
        self.n_window = window_size
        n_hidden = 16
        self.n_hidden = n_hidden

        # Temporal attention (Bahdanau-style): maps (W*N,) -> (W,) weights
        self.attention = nn.Sequential(
            nn.Linear(window_size * feats, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, window_size), nn.Softmax(dim=0),
        )

        # Multi-head graph attention: (N, 1) -> (N, feats)
        self.graph_attn = MultiHeadGraphAttention(
            in_features=1, out_per_head=1, num_heads=feats
        )

        # Output FCN: each node's (feats,) representation -> (W,) prediction
        self.fcn = nn.Sequential(
            nn.Linear(feats, n_hidden), nn.LeakyReLU(True),
            nn.Linear(n_hidden, window_size), nn.Sigmoid(),
        )

    def forward(self, data):
        """
        data: (W*N,) flattened window data
        returns: (W*N,) reconstruction
        """
        wn = data.view(self.n_window, self.n_feats)  # (W, N)

        # Temporal attention: (W,) weights
        att_score = self.attention(data)  # (W,)
        # Weighted aggregation over time: (N, 1)
        feat_r = torch.matmul(wn.permute(1, 0), att_score.view(-1, 1))  # (N, 1)

        # Multi-head graph attention: (N, 1) -> (N, feats)
        feat_r = self.graph_attn(feat_r)  # (N, feats)

        # FCN per node: (N, feats) -> (N, W)
        x = self.fcn(feat_r)  # (N, W)
        return x.view(-1)  # (N*W,)


# ---------------------------------------------------------------------------
#  GDN Detector
# ---------------------------------------------------------------------------

class GDNDetector(BaseMultivariateDetector):
    """GDN-based multivariate anomaly detector."""

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
        return GDNModel(
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
                recon = self.model(d)  # (N*W,)
                target = d
                loss = (recon - target) ** 2
                # Reshape to (W, N) and take last row for per-feature score
                loss_matrix = loss.view(feats, self.window_size)  # (N, W)
                # Average across window for per-feature score
                feat_scores = loss_matrix.mean(dim=1)  # (N,)
                all_scores.append(feat_scores.detach().cpu().numpy())

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
            f"Training GDN: feats={self.n_features}, window={self.window_size}, "
            f"epochs={epochs}, samples={len(train_data)}, device={self.device}"
        )

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            for d in windows_flat:
                d = d.to(self.device)
                recon = self.model(d)
                target = d
                loss = criterion(recon, target)
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
        result["method"] = "GDN"
        return result
