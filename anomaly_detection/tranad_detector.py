"""
TranAD — Deep Transformer Networks for Anomaly Detection (VLDB 2022).

Thin wrapper that reuses the existing ``TranADDetector`` from
``multivariate_anomaly.py`` but exposes it through the
``BaseMultivariateDetector`` interface for the factory system.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_detector import BaseMultivariateDetector

logger = logging.getLogger("anomaly_detection.tranad")


# ---------------------------------------------------------------------------
#  Model components (self-contained, no dependency on external TranAD repo)
# ---------------------------------------------------------------------------

import math


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class _DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, **kwargs):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class _TranADModel(nn.Module):
    def __init__(self, feats, lr=0.001, window_size=10):
        super().__init__()
        self.name = "TranAD"
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = window_size
        self.n = self.n_feats * self.n_window

        d_model = 2 * feats
        self.pos_encoder = _PositionalEncoding(d_model, 0.1, window_size)
        encoder_layers = _EncoderLayer(
            d_model=d_model, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 1)

        decoder_layers1 = _DecoderLayer(
            d_model=d_model, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layers1, 1)

        decoder_layers2 = _DecoderLayer(
            d_model=d_model, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layers2, 1)

        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def _encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self._encode(src, c, tgt)))
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self._encode(src, c, tgt)))
        return x1, x2


# ---------------------------------------------------------------------------
#  TranAD Detector
# ---------------------------------------------------------------------------

class TranADDetectorAdapter(BaseMultivariateDetector):
    """TranAD-based multivariate anomaly detector.

    This is the same implementation as ``multivariate_anomaly.TranADDetector``
    but adapted to the ``BaseMultivariateDetector`` interface.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = 10,
        batch_size: int = 128,
        lr: float = 0.001,
        device: str = "auto",
    ):
        super().__init__(n_features, window_size, batch_size, lr, device)

    def _build_model(self) -> nn.Module:
        return _TranADModel(
            feats=self.n_features, lr=self.lr, window_size=self.window_size
        ).double()

    def _compute_feature_scores(self, data: torch.Tensor) -> np.ndarray:
        """Per-timestep per-feature reconstruction error via two-phase TranAD."""
        criterion = nn.MSELoss(reduction="none")
        feats = self.n_features

        windows = self._to_windows(data)  # (T, W, N)
        bs = len(data)

        dataset = TensorDataset(windows, windows)
        dataloader = DataLoader(dataset, batch_size=bs)

        all_scores = []
        with torch.no_grad():
            for d, _ in dataloader:
                d = d.to(self.device)           # (B, W, N)
                window = d.permute(1, 0, 2)      # (W, B, N)
                elem = window[-1, :, :].view(1, bs, feats)
                _, x2 = self.model(window, elem)
                loss = criterion(x2, elem)[0]    # (B, N)
                all_scores.append(loss.detach().cpu().numpy())

        return np.concatenate(all_scores, axis=0)  # (T, N)

    def train(self, train_data: np.ndarray, epochs: int = 5) -> None:
        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        criterion = nn.MSELoss(reduction="none")
        feats = self.n_features

        data = torch.tensor(train_data, dtype=torch.float64).to(self.device)
        windows = self._to_windows(data)

        dataset = TensorDataset(windows, windows)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        logger.info(
            f"Training TranAD: feats={feats}, epochs={epochs}, "
            f"samples={len(train_data)}, device={self.device}"
        )

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            n = epoch + 1
            for d, _ in dataloader:
                d = d.to(self.device)
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                x1, x2 = self.model(window, elem)

                l1 = (1 / n) * criterion(x1, elem) + (1 - 1 / n) * criterion(x2, elem)
                loss = torch.mean(l1)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
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
        result["method"] = "TranAD"
        return result
