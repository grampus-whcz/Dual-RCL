"""
Data adapter for TVDiag integration.

Handles two paths:
  1. **Offline training**: Reads TVDiag's precomputed GAIA embeddings (pkl)
     and constructs pure-tensor graph datasets.
  2. **Online inference**: Converts SoC-RCA's runtime data into TVDiag-compatible
     graph features using a precomputed embedding cache.
"""

import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .tvdig_config import TVDiagConfig
from .tvdig_model import add_self_loops, aug_drop_node

logger = logging.getLogger("failure_localization.tvdig_data")


# =====================================================================
#  Graph data container
# =====================================================================

def make_graph(
    edge_index: torch.Tensor,
    features: Dict[str, torch.Tensor],
    root: torch.Tensor,
    num_nodes: int,
    global_root_id: int = -1,
    failure_type_id: int = -1,
) -> dict:
    """Create a graph dict (replaces DGL graph + MultiModalDataSet entry)."""
    edge_index = add_self_loops(edge_index, num_nodes)
    return {
        "edge_index": edge_index,
        "num_nodes": num_nodes,
        "features": features,
        "root": root,
        "global_root_id": global_root_id,
        "failure_type_id": failure_type_id,
    }


# =====================================================================
#  Offline training data loading (from TVDiag precomputed pkls)
# =====================================================================

def build_training_dataset(config: TVDiagConfig):
    """Build training and test datasets from TVDiag's precomputed GAIA data.

    Reads the pkl files produced by TVDiag's EventProcess pipeline:
        data_dir/tmp/metric.pkl, trace.pkl, log.pkl
        data_dir/raw/nodes.json, edges.json
        data_dir/label.csv

    Returns:
        (train_data, aug_data, test_data, embedding_cache)
    """
    data_dir = config.data_dir
    tmp_dir = os.path.join(data_dir, "tmp")

    # Load precomputed embeddings
    metric_embs = _load_pkl(os.path.join(tmp_dir, "metric.pkl"))
    trace_embs = _load_pkl(os.path.join(tmp_dir, "trace.pkl"))
    log_embs = _load_pkl(os.path.join(tmp_dir, "log.pkl"))

    # Load topology
    with open(os.path.join(data_dir, "raw", "nodes.json"), "r") as f:
        nodes_json = json.load(f)
    with open(os.path.join(data_dir, "raw", "edges.json"), "r") as f:
        edges_json = json.load(f)

    # Load labels
    label_path = os.path.join(data_dir, "label.csv")
    labels_df = pd.read_csv(label_path)
    labels_df["index"] = labels_df["index"].astype(str)

    # Build label mappings
    all_nodes = sorted(list({item for sublist in nodes_json.values() for item in sublist}))
    node2idx = {node: idx for idx, node in enumerate(all_nodes)}

    types_list = ["normal"] + labels_df["anomaly_type"].unique().tolist()
    ft2idx = {label: idx for idx, label in enumerate(types_list)}

    # Build embedding cache for inference
    embedding_cache = _build_embedding_cache_from_pkls(
        metric_embs, trace_embs, log_embs, nodes_json, labels_df,
        train_only=getattr(config, "cache_train_only", True),
    )

    # Build graph datasets
    train_data, test_data = [], []
    for _, row in labels_df.iterrows():
        idx = str(row["index"])
        nodes = nodes_json[idx]
        edges = edges_json[idx]
        num_nodes = len(nodes)

        # Edge index as tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edge_index.dim() == 1:
            edge_index = edge_index.unsqueeze(0)

        # Features
        m_feats = torch.FloatTensor(metric_embs[idx])
        t_feats = torch.FloatTensor(trace_embs[idx])
        l_feats = torch.FloatTensor(log_embs[idx])

        # Root label
        local_root = row["instance"]
        # Skip events whose root service is not in the global node dictionary
        # (e.g., node-level faults like "node-6" which aren't service pods)
        if local_root not in node2idx:
            continue
        root_list = [0] * num_nodes
        if local_root in nodes:
            root_list[nodes.index(local_root)] = 1
        root = torch.LongTensor(root_list)

        features = {"metric": m_feats, "trace": t_feats, "log": l_feats}
        g = make_graph(
            edge_index=edge_index,
            features=features,
            root=root,
            num_nodes=num_nodes,
            global_root_id=node2idx[local_root],
            failure_type_id=ft2idx[row["anomaly_type"]],
        )

        if row["data_type"] == "train":
            train_data.append(g)
        else:
            test_data.append(g)

    # Augmentation
    aug_data = []
    if config.aug_times > 0:
        for _ in range(config.aug_times):
            for g in train_data:
                root_idx = g["root"].tolist().index(1)
                aug_g = aug_drop_node(g, root_idx, config.aug_percent)
                aug_data.append((aug_g, (g["global_root_id"], g["failure_type_id"])))

    logger.info(f"Dataset: train={len(train_data)}, test={len(test_data)}, aug={len(aug_data)}")
    return train_data, aug_data, test_data, embedding_cache


# =====================================================================
#  Online inference data adapter
# =====================================================================

def build_inference_graph(
    config: TVDiagConfig,
    metric_anomaly_events: List[List[str]],
    trace_events: List[List[str]],
    log_events: List[List[str]],
    node_names: Optional[List[str]] = None,
) -> dict:
    """Build an inference graph from SoC-RCA's runtime anomaly events.

    Args:
        config: TVDiagConfig with GAIA topology.
        metric_anomaly_events: per-node lists of metric event strings
            e.g. [["dbservice1&0.0.0.4&docker_cpu_pct&up", ...], ...]
        trace_events: per-node lists of trace event strings
            e.g. [["dbservice1&redisservice1&http://...&ERROR", ...], ...]
        log_events: per-node lists of log event strings
            e.g. [["dbservice1&3", "dbservice1&5", ...], ...]
        node_names: service node names (default: config.NODE_NAMES)

    Returns:
        Graph dict ready for model inference.
    """
    if node_names is None:
        node_names = config.NODE_NAMES

    num_nodes = len(node_names)
    edge_index = torch.tensor(config.GAIA_EDGES, dtype=torch.long).t().contiguous()

    # Build features from event embeddings (zero vectors for nodes with no events)
    embedding_dim = config.alert_embedding_dim
    features = {}
    for mod_idx, (mod_name, events_per_node) in enumerate([
        ("metric", metric_anomaly_events),
        ("trace", trace_events),
        ("log", log_events),
    ]):
        node_feats = torch.zeros(num_nodes, embedding_dim)
        for node_idx, events in enumerate(events_per_node):
            if events:
                # Average of random embeddings (since we don't have FastText at inference)
                # In practice, use precomputed embedding_cache for best results
                node_feats[node_idx] = _events_to_embedding(events, embedding_dim)
        features[mod_name] = node_feats

    # No root label at inference time (that's what we're predicting)
    root = torch.zeros(num_nodes, dtype=torch.long)

    return make_graph(
        edge_index=edge_index,
        features=features,
        root=root,
        num_nodes=num_nodes,
    )


def build_inference_graph_from_cache(
    config: TVDiagConfig,
    embedding_cache: dict,
    metric_anomaly_events: List[List[str]],
    trace_events: List[List[str]],
    log_events: List[List[str]],
) -> dict:
    """Build inference graph using precomputed embedding cache.

    Fixed: cache is keyed by SERVICE NODE NAME (not event string).
    Nodes with anomaly events get their cached service embedding (scaled by
    event count); nodes without anomalies get zero features. This creates
    per-event variation so the model can rank anomalous nodes.

    Args:
        config: TVDiagConfig.
        embedding_cache: dict from modality -> node_name -> numpy array.
        metric_anomaly_events, trace_events, log_events: per-node event lists.

    Returns:
        Graph dict for model inference.
    """
    node_names = config.NODE_NAMES
    num_nodes = len(node_names)
    edge_index = torch.tensor(config.GAIA_EDGES, dtype=torch.long).t().contiguous()
    embedding_dim = config.alert_embedding_dim

    features = {}
    for mod_name, events_per_node in [
        ("metric", metric_anomaly_events),
        ("trace", trace_events),
        ("log", log_events),
    ]:
        node_feats = torch.zeros(num_nodes, embedding_dim)
        cache = embedding_cache.get(mod_name, {})
        for node_idx, events in enumerate(events_per_node):
            node_name = node_names[node_idx]
            # Look up by node NAME in cache; scale by event count (anomaly intensity)
            if events and node_name in cache:
                base_emb = cache[node_name]
                # Scale embedding by log(1 + event_count) to reflect anomaly intensity
                intensity = np.log1p(len(events))
                node_feats[node_idx] = torch.FloatTensor(base_emb * min(intensity, 3.0))
        features[mod_name] = node_feats

    root = torch.zeros(num_nodes, dtype=torch.long)
    return make_graph(edge_index=edge_index, features=features, root=root, num_nodes=num_nodes)


# =====================================================================
#  Embedding cache
# =====================================================================

def _build_embedding_cache_from_pkls(metric_embs, trace_embs, log_embs, nodes_json, labels_df,
                                     train_only: bool = True):
    """Build a mapping from event strings to embedding vectors.

    Uses TVDiag's precomputed pkl embeddings. For each fault instance,
    maps the event strings (from raw JSON) to the corresponding
    node-level embedding vectors from the pkl.

    Args:
        train_only: if True (strict), average node embeddings over *train*
            incidents only (label.csv data_type == 'train'). The historical
            default averaged over all incidents, which leaks test-set event
            documents into the inference cache; keep ``train_only=False``
            only as a flagged transductive ablation.
    """
    cache = {"metric": {}, "trace": {}, "log": {}}

    if train_only:
        train_idx = set(labels_df.loc[labels_df["data_type"] == "train", "index"].astype(str))
        logger.info(f"Embedding cache (train_only=True): averaging over {len(train_idx)} train incidents")
    else:
        train_idx = None
        logger.info("Embedding cache (train_only=False): averaging over ALL incidents (transductive)")

    # The cache is built by averaging all embeddings for a given event pattern
    # across fault instances. This provides a generalizable lookup.
    # For simplicity, we store the average embedding per service node per modality.
    all_nodes = sorted(list({item for sublist in nodes_json.values() for item in sublist}))

    def _collect(mod_embs, mod):
        for idx_str, node_embs in mod_embs.items():
            if train_idx is not None and idx_str not in train_idx:
                continue
            nodes = nodes_json.get(idx_str, [])
            for node_idx, node_name in enumerate(nodes):
                if node_idx < len(node_embs):
                    cache[mod].setdefault(node_name, []).append(node_embs[node_idx])

    _collect(metric_embs, "metric")
    _collect(trace_embs, "trace")
    _collect(log_embs, "log")

    # Average the collected embeddings
    for mod in cache:
        for key in cache[mod]:
            arr = np.array(cache[mod][key])
            cache[mod][key] = np.mean(arr, axis=0)

    return cache


def save_embedding_cache(cache: dict, path: str):
    """Save embedding cache to pkl."""
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    logger.info(f"Embedding cache saved to {path}")


def load_embedding_cache(path: str) -> dict:
    """Load embedding cache from pkl."""
    with open(path, "rb") as f:
        cache = pickle.load(f)
    logger.info(f"Embedding cache loaded from {path} ({sum(len(v) for v in cache.values())} entries)")
    return cache


# =====================================================================
#  SoC-RCA runtime data extraction helpers
# =====================================================================

def extract_metric_events_from_univariate(
    metric_data: np.ndarray,
    data_head: List[str],
    n_init: int,
    node_names: List[str],
) -> List[List[str]]:
    """Extract metric anomaly events from SoC-RCA's univariate detection results.

    For each service node, create event strings in TVDiag format:
        "service_name&metric_name&direction"

    Args:
        metric_data: (T, N) time series array.
        data_head: list of N metric names.
        n_init: index of train/test split.
        node_names: list of service node names.

    Returns:
        List of length len(node_names), each a list of event strings.
    """
    test_data = metric_data[n_init:]
    train_data = metric_data[:n_init]

    # Compute per-metric deviation
    train_mean = train_data.mean(axis=0)
    train_std = train_data.std(axis=0)
    train_std[train_std == 0] = 1e-6

    # Z-score of test data
    z_scores = np.abs((test_data - train_mean) / train_std)

    # Detect anomalous metrics (z > 3)
    mean_z = z_scores.mean(axis=0)
    anomalous = mean_z > 3.0

    # Group by service
    events_per_node = [[] for _ in node_names]
    for metric_idx, metric_name in enumerate(data_head):
        if not anomalous[metric_idx]:
            continue
        # Determine service and direction
        direction = "up" if test_data[:, metric_idx].mean() > train_mean[metric_idx] else "down"
        service = _extract_service(metric_name, node_names)
        if service and service in node_names:
            node_idx = node_names.index(service)
            event_str = f"{service}&0.0.0.0&{metric_name}&{direction}"
            events_per_node[node_idx].append(event_str)

    return events_per_node


def extract_trace_events_from_soCRCA(
    trace_data_path: str,
    node_names: List[str],
) -> List[List[str]]:
    """Extract trace anomaly events from SoC-RCA's trace data.

    Reads trace data files and creates event strings in TVDiag format:
        "src&dst&operation&error_type"

    Args:
        trace_data_path: path to trace data directory/file.
        node_names: list of service node names.

    Returns:
        List of length len(node_names), each a list of event strings.
    """
    events_per_node = [[] for _ in node_names]

    if not os.path.exists(trace_data_path):
        return events_per_node

    try:
        import pickle as pkl
        with open(trace_data_path, "rb") as f:
            trace_data = pkl.load(f)
        # Trace data is typically a list of span records
        # Extract src, dst, operation, error_type
        if isinstance(trace_data, list):
            for span in trace_data:
                if isinstance(span, (list, tuple)) and len(span) >= 4:
                    src, dst, op, error = str(span[0]), str(span[1]), str(span[2]), str(span[3])
                    event = f"{src}&{dst}&{op}&{error}"
                    for node_idx, node_name in enumerate(node_names):
                        if node_name in src or node_name in dst:
                            events_per_node[node_idx].append(event)
        elif isinstance(trace_data, dict):
            for idx, spans in trace_data.items():
                if isinstance(spans, list):
                    for span in spans:
                        if isinstance(span, (list, tuple)) and len(span) >= 4:
                            src, dst, op, error = str(span[0]), str(span[1]), str(span[2]), str(span[3])
                            event = f"{src}&{dst}&{op}&{error}"
                            for node_idx, node_name in enumerate(node_names):
                                if node_name in src or node_name in dst:
                                    events_per_node[node_idx].append(event)
    except Exception as e:
        logger.warning(f"Could not extract trace events: {e}")

    return events_per_node


def extract_log_events_from_soCRCA(
    log_dir: str,
    root_services: List[str],
    node_names: List[str],
) -> List[List[str]]:
    """Extract log anomaly events from SoC-RCA's log CSV files.

    Creates event strings in TVDiag format: "service&event_template"

    Args:
        log_dir: path to the log fault directory.
        root_services: list of root cause candidate services (to filter logs).
        node_names: list of all service node names.

    Returns:
        List of length len(node_names), each a list of event strings.
    """
    import csv as csv_mod

    events_per_node = [[] for _ in node_names]

    if not os.path.isdir(log_dir):
        return events_per_node

    log_files = os.listdir(log_dir)
    for fi in log_files:
        # Determine which service this log file belongs to
        matched_node = None
        for node_name in node_names:
            if node_name in fi:
                matched_node = node_name
                break
        if matched_node is None:
            continue

        # Also check if it's a root cause candidate
        is_root_candidate = any(root in fi for root in root_services)
        if not is_root_candidate:
            continue

        node_idx = node_names.index(matched_node)
        filepath = os.path.join(log_dir, fi)
        try:
            with open(filepath, "r", newline="", encoding="utf-8") as f:
                reader = csv_mod.reader(f)
                for row in reader:
                    if len(row) >= 3 and "ERROR" in row[2]:
                        # Extract a simple event template
                        msg = row[2][:200]
                        # Use a simple hash as event id
                        event_id = str(hash(msg) % 10000)
                        events_per_node[node_idx].append(f"{matched_node}&{event_id}")
        except Exception as e:
            logger.warning(f"Could not read log file {filepath}: {e}")

    return events_per_node


# =====================================================================
#  Internal helpers
# =====================================================================

def _load_pkl(path):
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _events_to_embedding(events: List[str], dim: int) -> torch.Tensor:
    """Convert event strings to a fixed-dim embedding.

    Uses a deterministic hash-based projection when no pretrained
    FastText embeddings are available. This is a fallback — for
    best results use the precomputed embedding_cache.
    """
    if not events:
        return torch.zeros(dim)

    # Hash-based embedding: each event contributes a deterministic vector
    total = torch.zeros(dim)
    for ev in events:
        torch.manual_seed(hash(ev) % (2**31))
        total += torch.randn(dim)
    torch.manual_seed(42)  # Reset seed

    return total / len(events)


def _extract_service(metric_name: str, node_names: List[str]) -> Optional[str]:
    """Extract service name from a metric name like 'webservice1_docker_cpu'."""
    for svc in node_names:
        if metric_name.startswith(svc):
            return svc
    # Fallback: take first part
    parts = metric_name.split("_")
    if parts:
        return parts[0]
    return None
