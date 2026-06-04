"""
Pure PyTorch reimplementation of TVDiag model components.

Replaces all DGL dependencies (SAGEConv, graph batching, MaxPooling)
with native PyTorch operations. Faithful to the original TVDiag architecture
from: TVDiag — A Task-oriented and View-invariant Failure Diagnosis
Framework for Microservice-based Systems with Multimodal Data.

Components:
  - SAGEConv: GraphSAGE mean-aggregator convolution
  - SAGEEncoder: Multi-layer SAGE + graph readout
  - Encoder: Per-modality SAGE encoder wrapper
  - Voter: Root cause localization head (per-node scoring)
  - Classifier: Failure type identification head (graph-level)
  - MainModel: Full multimodal TVDiag model
  - Loss functions: SupConLoss, UspConLoss, AutomaticWeightedLoss
  - TVDiagTrainer: Training and evaluation loop
"""

import copy
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tvdig_config import TVDiagConfig

logger = logging.getLogger("failure_localization.tvdig_model")


# =====================================================================
#  Graph utilities (replacing DGL)
# =====================================================================

def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Add self-loop edges for nodes with zero in-degree."""
    src, dst = edge_index[0], edge_index[1]
    in_deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    in_deg.scatter_add_(0, dst, torch.ones_like(dst))
    zero_nodes = torch.where(in_deg == 0)[0]
    if zero_nodes.numel() > 0:
        loops = torch.stack([zero_nodes, zero_nodes], dim=0)
        edge_index = torch.cat([edge_index, loops], dim=1)
    return edge_index


def batch_graphs(graph_list: List[dict]) -> dict:
    """Batch multiple graph dicts into one (replaces dgl.batch).

    Each graph dict has:
        edge_index: (2, E)
        num_nodes: int
        features: dict[str, (N, D)]
        root: (N,) LongTensor
    """
    features_batched: Dict[str, list] = {}
    edge_index_list = []
    num_nodes_list = []
    root_list = []
    offset = 0

    for g in graph_list:
        n = g["num_nodes"]
        num_nodes_list.append(n)
        edge_index_list.append(g["edge_index"] + offset)
        root_list.append(g["root"])
        for mod, feat in g["features"].items():
            features_batched.setdefault(mod, []).append(feat)
        offset += n

    return {
        "edge_index": torch.cat(edge_index_list, dim=1),
        "num_nodes_list": num_nodes_list,
        "features": {mod: torch.cat(feats, dim=0) for mod, feats in features_batched.items()},
        "root": torch.cat(root_list, dim=0),
    }


def unbatch_graphs(batched: dict) -> List[dict]:
    """Split a batched graph back into individual graphs."""
    num_nodes_list = batched["num_nodes_list"]
    graphs = []
    offset = 0
    for n in num_nodes_list:
        # Find edges belonging to this graph
        ei = batched["edge_index"]
        mask = (ei[0] >= offset) & (ei[0] < offset + n) & (ei[1] >= offset) & (ei[1] < offset + n)
        sub_ei = ei[:, mask] - offset
        sub_features = {mod: feat[offset:offset + n] for mod, feat in batched["features"].items()}
        sub_root = batched["root"][offset:offset + n]
        graphs.append({
            "edge_index": sub_ei,
            "num_nodes": n,
            "features": sub_features,
            "root": sub_root,
        })
        offset += n
    return graphs


def aug_drop_node(graph: dict, root_idx: int, drop_percent: float = 0.2) -> dict:
    """Augment by dropping non-root nodes (replaces aug.aug_drop_node)."""
    num_nodes = graph["num_nodes"]
    drop_num = int(num_nodes * drop_percent)
    all_nodes = [i for i in range(num_nodes) if i != root_idx]
    drop_num = min(drop_num, len(all_nodes))
    drop_nodes = set(np.random.choice(all_nodes, drop_num, replace=False).tolist())

    # Build keep mask
    keep = [i for i in range(num_nodes) if i not in drop_nodes]
    keep_set = set(keep)
    old_to_new = {old: new for new, old in enumerate(keep)}

    # Remap edge_index
    ei = graph["edge_index"]
    mask = keep_set.intersection(ei[0].tolist()) & keep_set.intersection(ei[1].tolist())
    # Keep edges where both src and dst are in keep set
    src_in = torch.tensor([s.item() in keep_set for s in ei[0]], dtype=torch.bool)
    dst_in = torch.tensor([d.item() in keep_set for d in ei[1]], dtype=torch.bool)
    valid = src_in & dst_in
    new_ei = ei[:, valid]
    new_ei = torch.tensor(
        [[old_to_new[s.item()] for s in new_ei[0]],
         [old_to_new[d.item()] for d in new_ei[1]]],
        dtype=torch.long,
    )

    new_features = {mod: feat[keep] for mod, feat in graph["features"].items()}
    new_root = graph["root"][keep]

    # Add self-loops for zero in-degree nodes
    new_ei = add_self_loops(new_ei, len(keep))

    return {
        "edge_index": new_ei,
        "num_nodes": len(keep),
        "features": new_features,
        "root": new_root,
    }


# =====================================================================
#  SAGEConv — Pure PyTorch GraphSAGE mean aggregator
# =====================================================================

class SAGEConv(nn.Module):
    """GraphSAGE convolution with mean aggregation (pure PyTorch).

    Replaces dgl.nn.SAGEConv with aggregator_type='mean'.

    Computes: h'_v = W * concat(h_v, mean({h_u : u in N(v)}))
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        aggregator_type: str = "mean",
        feat_drop: float = 0.0,
        activation=None,
        bias: bool = False,
    ):
        super().__init__()
        self.aggregator_type = aggregator_type
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        if aggregator_type == "mean":
            self.fc = nn.Linear(in_feats * 2, out_feats, bias=bias)
        else:
            raise NotImplementedError(f"Aggregator '{aggregator_type}' not implemented")

    def forward(self, edge_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_index: (2, E) — [src, dst] pairs
            x: (N, in_feats) — node features
        Returns:
            (N, out_feats) — updated node features
        """
        x = self.feat_drop(x)
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        # Aggregate: scatter mean of neighbor features
        agg = torch.zeros_like(x)
        count = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
        agg.index_add_(0, dst, x[src])
        count.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device, dtype=x.dtype))
        count = count.clamp(min=1)
        agg = agg / count

        # Concat self + aggregated, then linear transform
        h = self.fc(torch.cat([x, agg], dim=1))

        if self.activation is not None:
            h = self.activation(h)
        return h


# =====================================================================
#  SAGEEncoder — Multi-layer SAGE + graph readout
# =====================================================================

class SAGEEncoder(nn.Module):
    """Multi-layer GraphSAGE encoder producing node and graph embeddings.

    Replaces core/model/backbone/sage.py.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        aggregator_type: str = "mean",
        feat_drop: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(SAGEConv(in_dim, out_dim, aggregator_type, feat_drop, activation=None))
        else:
            self.layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type, feat_drop, activation=F.relu))
            for _ in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type, feat_drop, activation=F.relu))
            self.layers.append(SAGEConv(hidden_dim, out_dim, aggregator_type, feat_drop, activation=None))

    def forward(self, edge_index: torch.Tensor, x: torch.Tensor,
                num_nodes_list: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            edge_index: (2, E)
            x: (N, in_dim)
            num_nodes_list: list of node counts per graph in the batch.
                If None, treats as a single graph.

        Returns:
            f: (num_graphs, out_dim) — graph-level embeddings (max-pool per graph)
            e: (N, out_dim) — node-level embeddings
        """
        for l in range(self.num_layers - 1):
            x = self.layers[l](edge_index, x)
        e = self.layers[-1](edge_index, x)

        # Graph-level readout: max pooling per graph
        if num_nodes_list is None or len(num_nodes_list) == 0:
            f = e.max(dim=0)[0].unsqueeze(0)  # (1, out_dim)
        else:
            graph_embs = []
            offset = 0
            for n in num_nodes_list:
                sub_e = e[offset:offset + n]
                graph_embs.append(sub_e.max(dim=0)[0])
                offset += n
            f = torch.stack(graph_embs, dim=0)  # (num_graphs, out_dim)
        return f, e


# =====================================================================
#  Encoder, Voter, Classifier, FullyConnected
# =====================================================================

class Encoder(nn.Module):
    """Per-modality encoder wrapping SAGEEncoder."""

    def __init__(self, alert_embedding_dim, graph_hidden_dim, graph_out_dim,
                 num_layers=2, aggregator="mean", feat_drop=0.0):
        super().__init__()
        self.graph_encoder = SAGEEncoder(
            in_dim=alert_embedding_dim,
            out_dim=graph_out_dim,
            hidden_dim=graph_hidden_dim,
            num_layers=num_layers,
            aggregator_type=aggregator,
            feat_drop=feat_drop,
        )

    def forward(self, edge_index, x, num_nodes_list=None):
        return self.graph_encoder(edge_index, x, num_nodes_list)


class FullyConnected(nn.Module):
    """Simple MLP (copied from TVDiag core/model/backbone/FC.py)."""

    def __init__(self, in_dim, out_dim, linear_sizes=None):
        super().__init__()
        linear_sizes = linear_sizes or [64]
        layers = []
        for i, hidden in enumerate(linear_sizes):
            input_size = in_dim if i == 0 else linear_sizes[i - 1]
            layers += [nn.Linear(input_size, hidden), nn.ReLU()]
        layers += [nn.Linear(linear_sizes[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Voter(nn.Module):
    """Root cause localization head — scores each node."""

    def __init__(self, in_dim, out_dim=1, hiddens=None):
        super().__init__()
        hiddens = hiddens or [64]
        self.net = FullyConnected(in_dim, out_dim, hiddens)

    def forward(self, h):
        return self.net(h)


class Classifier(nn.Module):
    """Failure type identification head — graph-level classification."""

    def __init__(self, in_dim, out_dim, hiddens=None):
        super().__init__()
        hiddens = hiddens or [64]
        self.net = FullyConnected(in_dim, out_dim, hiddens)

    def forward(self, h):
        return self.net(h)


# =====================================================================
#  MainModel — Full multimodal TVDiag
# =====================================================================

class MainModel(nn.Module):
    """TVDiag MainModel (pure PyTorch, no DGL).

    Separate GraphSAGE encoders per modality, feature fusion,
    Voter for RCL, and Classifier for FTI.
    """

    def __init__(self, config: TVDiagConfig):
        super().__init__()
        self.encoders = nn.ModuleDict()
        for modality in config.modalities:
            self.encoders[modality] = Encoder(
                alert_embedding_dim=config.alert_embedding_dim,
                graph_hidden_dim=config.graph_hidden_dim,
                graph_out_dim=config.graph_out,
                num_layers=config.graph_layers,
                aggregator=config.aggregator,
                feat_drop=config.feat_drop,
            )

        fti_fuse_dim = len(config.modalities) * config.graph_out
        rcl_fuse_dim = len(config.modalities) * config.graph_out

        self.locator = Voter(rcl_fuse_dim, out_dim=1, hiddens=config.linear_hidden)
        self.typeClassifier = Classifier(in_dim=fti_fuse_dim, out_dim=config.ft_num, hiddens=config.linear_hidden)

    def forward(self, edge_index: torch.Tensor, num_nodes_list,
                features: Dict[str, torch.Tensor]):
        """
        Args:
            edge_index: (2, E) — graph edges
            num_nodes_list: list of node counts per graph in the batch
            features: dict mapping modality -> (N, alert_embedding_dim) tensor

        Returns:
            fs: dict[str, (num_graphs, graph_out)] — per-modality graph embeddings
            es: dict[str, (N, graph_out)] — per-modality node embeddings
            root_logit: (N, 1) — per-node root cause score
            type_logit: (num_graphs, ft_num) — failure type logits
        """
        fs, es = {}, {}
        for modality, encoder in self.encoders.items():
            x_d = features[modality]
            f_d, e_d = encoder(edge_index, x_d, num_nodes_list)
            fs[modality] = f_d
            es[modality] = e_d

        # Graph-level fusion for FTI
        f = torch.cat(tuple(fs.values()), dim=1)
        type_logit = self.typeClassifier(f)

        # Node-level fusion for RCL
        e = torch.cat(list(es.values()), dim=1)
        root_logit = self.locator(e)

        return fs, es, root_logit, type_logit


# =====================================================================
#  Loss Functions (copied from TVDiag, no DGL dependency)
# =====================================================================

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""

    def __init__(self, temperature=0.5, device="cpu"):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        self.device = device

    def forward(self, embeddings, labels):
        n = labels.shape[0]
        embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)

        mask = torch.ones_like(similarity_matrix) * (labels.expand(n, n).eq(labels.expand(n, n).t()))
        mask_no_sim = torch.ones_like(mask) - mask

        mask_dj = torch.ones(n, n) - torch.eye(n, n)
        mask_dj = mask_dj.to(self.device)
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)
        similarity_matrix = similarity_matrix * mask_dj

        sim = mask * similarity_matrix
        no_sim = similarity_matrix - sim

        no_sim_sum = torch.sum(no_sim, dim=1)
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum = sim + no_sim_sum_expend

        loss_partial = -torch.log(mask_no_sim + torch.div(sim, sim_sum) + torch.eye(n, n).to(self.device))

        nonzero_count = len(torch.nonzero(loss_partial))
        if nonzero_count == 0:
            nonzero_count = 1

        loss = torch.sum(torch.sum(loss_partial, dim=1)) / nonzero_count
        return loss


class UspConLoss(nn.Module):
    """Unsupervised Contrastive Loss (SimCLR-style)."""

    def __init__(self, temperature=0.5, device="cpu"):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))
        self.device = device

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        logits = z_i @ z_j.T
        logits /= self.temperature
        n = z_j.shape[0]
        labels = torch.arange(0, n, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels)
        return loss


class AutomaticWeightedLoss(nn.Module):
    """Automatically weighted multi-task loss."""

    def __init__(self, num=2):
        super().__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


# =====================================================================
#  Evaluation helpers
# =====================================================================

def rca_eval(root_logit, num_nodes_list, roots):
    """Evaluate Root Cause Localization (HR@k, MRR@3)."""
    res = {"HR@1": [], "HR@2": [], "HR@3": [], "HR@4": [], "HR@5": [], "MRR@3": []}
    start_idx = 0
    for idx, num_nodes in enumerate(num_nodes_list):
        end_idx = start_idx + num_nodes
        node_logits = root_logit[start_idx:end_idx].reshape(1, -1)
        root = roots[start_idx:end_idx].tolist().index(1)
        _, sorted_indices = torch.sort(node_logits, descending=True)
        for j in range(1, 6):
            res[f"HR@{j}"].append(1 if root in sorted_indices.flatten()[:j].tolist() else 0)
        rank = (sorted_indices == root).nonzero(as_tuple=True)
        if len(rank[1]) > 0:
            rank_val = rank[1][0].item() + 1
        else:
            rank_val = num_nodes
        res["MRR@3"].append(1 / rank_val if rank_val <= 3 else 0)
        start_idx += num_nodes

    for k in range(1, 6):
        res[f"HR@{k}"] = np.sum(res[f"HR@{k}"]) / len(num_nodes_list)
    res["MRR@3"] = np.sum(res["MRR@3"]) / len(num_nodes_list)
    return res


def fti_eval(output, target):
    """Evaluate Failure Type Identification (precision, recall, f1)."""
    from sklearn.metrics import precision_score, recall_score
    _, pred = output.topk(1, 1, True, True)
    y_pred = pred.cpu().detach().numpy()
    y_true = target.cpu().detach().numpy().reshape(-1, 1)
    pre = precision_score(y_true, y_pred[:, 0], average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred[:, 0], average="weighted", zero_division=0)
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    return {"pre": pre, "rec": rec, "f1": f1}


# =====================================================================
#  Early Stopping
# =====================================================================

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def should_stop(self, loss, epoch):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# =====================================================================
#  TVDiagTrainer — Training and evaluation loop
# =====================================================================

class TVDiagTrainer:
    """Pure PyTorch TVDiag training and evaluation.

    Replaces core/TVDiag.py, removing all DGL dependencies.
    """

    def __init__(self, config: TVDiagConfig, log_dir: str, device: str = "auto"):
        self.config = config
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        config.print_configs(logger)

    def train(self, train_data: list, aug_data: list):
        """Train the TVDiag model.

        Args:
            train_data: list of graph dicts (from tvdig_data.build_training_dataset)
            aug_data: list of (augmented_graph_dict, (global_root_id, failure_type_id))
        """
        model = MainModel(self.config).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        awl = AutomaticWeightedLoss(4).to(self.device)
        sup_con = SupConLoss(self.config.temperature, self.device).to(self.device)
        usp_con = UspConLoss(self.config.temperature, self.device).to(self.device)

        logger.info(f"Start training for {self.config.epochs} epochs, device={self.device}")
        early_stop = EarlyStopping(self.config.patience)

        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0
            n_iter = 0

            # Sample random order
            indices = list(range(len(train_data)))
            np.random.shuffle(indices)

            for batch_start in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[batch_start:batch_start + self.config.batch_size]
                batch_graphs_raw = [train_data[i] for i in batch_indices]

                # Add augmented samples
                if self.config.aug_times > 0 and aug_data:
                    aug_samples = [aug_data[i] for i in np.random.choice(len(aug_data), len(batch_indices), replace=True)]
                    batch_graphs_raw += [g for g, _ in aug_samples]
                    instance_labels = torch.tensor(
                        [train_data[i]["global_root_id"] for i in batch_indices] +
                        [labels[0] for _, labels in aug_samples]
                    )
                    type_labels = torch.tensor(
                        [train_data[i]["failure_type_id"] for i in batch_indices] +
                        [labels[1] for _, labels in aug_samples]
                    )
                else:
                    instance_labels = torch.tensor([g["global_root_id"] for g in batch_graphs_raw])
                    type_labels = torch.tensor([g["failure_type_id"] for g in batch_graphs_raw])

                batched = batch_graphs(batch_graphs_raw)
                edge_index = batched["edge_index"].to(self.device)
                features = {mod: feat.to(self.device) for mod, feat in batched["features"].items()}
                instance_labels = instance_labels.to(self.device)
                type_labels = type_labels.to(self.device)

                opt.zero_grad()
                num_nodes_list = batched["num_nodes_list"]
                fs, es, root_logit, type_logit = model(edge_index, num_nodes_list, features)

                # Task-oriented contrastive loss
                l_to = torch.tensor(0.0, device=self.device)
                l_cm = torch.tensor(0.0, device=self.device)
                modalities = self.config.modalities

                if self.config.TO:
                    if "metric" in modalities:
                        l_to = l_to + sup_con(fs["metric"], instance_labels)
                    if "log" in modalities:
                        l_to = l_to + sup_con(fs["log"], type_labels)
                    if "trace" in modalities:
                        l_to = l_to + sup_con(fs["trace"], instance_labels)

                # Cross-modal association
                if self.config.CM and len(modalities) >= 2 and "metric" in modalities:
                    for mod in modalities:
                        if mod != "metric":
                            l_cm = l_cm + usp_con(fs["metric"], fs[mod])

                sigma = self.config.contrastive_loss_scale
                l_con = sigma * (l_to + l_cm)

                # RCL loss
                l_rcl = self._cal_rcl_loss(root_logit, batched["num_nodes_list"], batched["root"].to(self.device))
                # FTI loss
                l_fti = F.cross_entropy(type_logit, type_labels)

                if self.config.dynamic_weight:
                    total_loss = awl(l_rcl, l_fti, sigma * l_to, sigma * l_cm)
                else:
                    total_loss = l_con + l_rcl + l_fti

                total_loss.backward()
                opt.step()

                epoch_loss += total_loss.detach().item()
                n_iter += 1

            avg_loss = epoch_loss / max(n_iter, 1)

            # Evaluate on training data
            if epoch % 20 == 0 or epoch == self.config.epochs - 1:
                rcl_res, fti_res = self._quick_eval(model, train_data)
                logger.info(
                    f"Epoch {epoch}: loss={avg_loss:.4f}, "
                    f"HR@1={rcl_res['HR@1']:.3%}, HR@3={rcl_res['HR@3']:.3%}, "
                    f"FTI_f1={fti_res['f1']:.3%}"
                )

            if early_stop.should_stop(avg_loss, epoch):
                logger.info(f"Early stop at epoch {epoch}")
                break

        # Save model
        state = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict()}
        save_path = os.path.join(self.log_dir, "tvdig.pt")
        torch.save(state, save_path)
        logger.info(f"Model saved to {save_path}")
        return model

    def evaluate(self, test_data: list, model=None):
        """Evaluate trained model on test data.

        Returns dict with RCL (HR@k, MRR@3) and FTI (precision, recall, f1).
        """
        if model is None:
            ckpt_path = os.path.join(self.log_dir, "tvdig.pt")
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model = MainModel(self.config).to(self.device)
            model.load_state_dict(state["model"])

        model.eval()
        root_logits, type_logits = [], []
        roots, types = [], []
        num_nodes_list = []

        for g in test_data:
            edge_index = g["edge_index"].to(self.device)
            features = {mod: feat.to(self.device) for mod, feat in g["features"].items()}

            with torch.no_grad():
                _, _, rl, tl = model(edge_index, [g["num_nodes"]], features)

            root_logits.append(rl.flatten().cpu())
            type_logits.append(tl.flatten().cpu())
            roots.append(g["root"])
            types.append(g["failure_type_id"])
            num_nodes_list.append(g["num_nodes"])

        root_logits = torch.hstack(root_logits)
        type_logits = torch.vstack(type_logits)
        roots = torch.hstack(roots)
        types = torch.tensor(types)

        rcl_res = rca_eval(root_logits, num_nodes_list, roots)
        fti_res = fti_eval(type_logits, types)

        logger.info(f"[RCL] HR@1={rcl_res['HR@1']:.3%}, HR@3={rcl_res['HR@3']:.3%}, HR@5={rcl_res['HR@5']:.3%}, MRR@3={rcl_res['MRR@3']:.3%}")
        logger.info(f"[FTI] pre={fti_res['pre']:.3%}, rec={fti_res['rec']:.3%}, f1={fti_res['f1']:.3%}")
        return {"rcl": rcl_res, "fti": fti_res}

    def _cal_rcl_loss(self, root_logit, num_nodes_list, roots):
        """Root Cause Localization loss (cross-entropy per graph, averaged)."""
        total_loss = None
        start_idx = 0
        for num_nodes in num_nodes_list:
            end_idx = start_idx + num_nodes
            node_logits = root_logit[start_idx:end_idx].reshape(1, -1)
            root = roots[start_idx:end_idx].tolist().index(1)
            loss = F.cross_entropy(node_logits, torch.LongTensor([root]).view(1).to(self.device))
            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss
            start_idx += num_nodes
        return total_loss / len(num_nodes_list)

    def _quick_eval(self, model, data_list):
        """Quick evaluation on a sample of data for training monitoring."""
        model.eval()
        sample = data_list[:min(50, len(data_list))]
        root_logits, roots = [], []
        num_nodes_list = []
        type_logits, types = [], []

        with torch.no_grad():
            for g in sample:
                ei = g["edge_index"].to(self.device)
                feats = {mod: f.to(self.device) for mod, f in g["features"].items()}
                _, _, rl, tl = model(ei, [g["num_nodes"]], feats)
                root_logits.append(rl.flatten().cpu())
                type_logits.append(tl.flatten().cpu())
                roots.append(g["root"])
                types.append(g["failure_type_id"])
                num_nodes_list.append(g["num_nodes"])

        root_logits = torch.hstack(root_logits)
        type_logits = torch.vstack(type_logits)
        roots = torch.hstack(roots)
        types = torch.tensor(types)

        rcl = rca_eval(root_logits, num_nodes_list, roots)
        fti = fti_eval(type_logits, types)
        model.train()
        return rcl, fti
