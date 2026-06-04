"""
MSCRED — Multi-Scale Convolutional Recurrent Encoder-Decoder (AAAI 2019).

Uses ConvLSTM encoder layers + ConvTranspose2d decoder.
Input is reshaped as a 2D matrix (feats × window).
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_detector import BaseMultivariateDetector

logger = logging.getLogger("anomaly_detection.mscred")


# ---------------------------------------------------------------------------
#  ConvLSTM cells (self-contained, from TranAD/src/dlutils.py)
# ---------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super().__init__()
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * num_layers
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim] * num_layers

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(cur_input_dim, self.hidden_dim[i], self.kernel_size[i], bias)
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(b, (h, w))

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(input_tensor.size(1)):
                h, c = self.cell_list[layer_idx](
                    cur_layer_input[:, t, :, :, :], [h, c]
                )
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        return [self.cell_list[i].init_hidden(batch_size, image_size) for i in range(self.num_layers)]


# ---------------------------------------------------------------------------
#  MSCRED Model
# ---------------------------------------------------------------------------

class MSCREDModel(nn.Module):
    """MSCRED: ConvLSTM encoder + ConvTranspose decoder."""

    def __init__(self, feats: int, lr: float = 0.0001):
        super().__init__()
        self.name = "MSCRED"
        self.lr = lr
        self.n_feats = feats
        self.n_window = feats  # MSCRED uses feats as window dimension

        self.encoder = nn.ModuleList([
            ConvLSTM(1, 32, (3, 3), 1, True, True, False),
            ConvLSTM(32, 64, (3, 3), 1, True, True, False),
            ConvLSTM(64, 128, (3, 3), 1, True, True, False),
        ])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        z = g.view(1, 1, self.n_feats, self.n_window)
        for cell in self.encoder:
            _, z = cell(z.view(1, *z.shape))
            z = z[0][0]
        x = self.decoder(z)
        return x.view(-1)


# ---------------------------------------------------------------------------
#  MSCRED Detector
# ---------------------------------------------------------------------------

class MSCREDDetector(BaseMultivariateDetector):
    """MSCRED-based multivariate anomaly detector.

    Note: MSCRED uses n_features as both the spatial dimension and the
    window dimension. The window_size parameter is overridden to equal
    n_features internally.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int = None,
        batch_size: int = 128,
        lr: float = 0.0001,
        device: str = "auto",
    ):
        # MSCRED uses feats × feats 2D input
        super().__init__(n_features, window_size=n_features, batch_size=batch_size, lr=lr, device=device)

    def _build_model(self) -> nn.Module:
        return MSCREDModel(feats=self.n_features, lr=self.lr).double()

    def _compute_feature_scores(self, data: torch.Tensor) -> np.ndarray:
        """Per-timestep per-feature reconstruction error via MSCRED."""
        feats = self.n_features
        windows = self._to_windows(data)  # (T, W, N) — W==N here

        all_scores = []
        with torch.no_grad():
            for d in windows:
                d = d.to(self.device)  # (W, N) = (feats, feats)
                recon = self.model(d)  # (feats*feats,)
                target = d.view(-1)
                loss = (recon - target) ** 2
                # Reshape to (feats, feats) and take diagonal or mean per feat
                loss_matrix = loss.view(feats, feats)
                # Use mean across window dimension as per-feature score
                feat_scores = loss_matrix.mean(dim=1)
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
        windows = self._to_windows(data)

        logger.info(
            f"Training MSCRED: feats={self.n_features}, window={self.window_size}, "
            f"epochs={epochs}, samples={len(train_data)}, device={self.device}"
        )

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            for d in windows:
                d = d.to(self.device)
                recon = self.model(d)
                target = d.view(-1)
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
        result["method"] = "MSCRED"
        return result
