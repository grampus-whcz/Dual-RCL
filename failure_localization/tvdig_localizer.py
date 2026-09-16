"""
TVDiag-based multimodal root cause localizer for SoC-RCA.

Uses the trained TVDiag GNN model to jointly analyze metric, trace, and log
data for root cause service localization and failure type identification.
"""

import logging
import os
from typing import Optional

import numpy as np
import torch

from .base_localizer import BaseLocalizer, LocalizationResult
from .tvdig_config import TVDiagConfig
from .tvdig_model import MainModel
from .tvdig_data import (
    build_inference_graph,
    build_inference_graph_from_cache,
    extract_metric_events_from_univariate,
    extract_trace_events_from_soCRCA,
    extract_log_events_from_soCRCA,
    load_embedding_cache,
)

logger = logging.getLogger("failure_localization.tvdig_localizer")


class TVDiagLocalizer(BaseLocalizer):
    """TVDiag multimodal root cause localization strategy.

    Requires a pre-trained model checkpoint and embedding cache.
    """

    def __init__(
        self,
        model_dir: str,
        config: Optional[TVDiagConfig] = None,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.config = config or TVDiagConfig()
        self.model_dir = model_dir

        # Detect CCF AIOps checkpoint: if dir name contains 'ccf', override config
        self._is_ccf_aiops = 'ccf' in os.path.basename(model_dir.rstrip('/')).lower()
        if self._is_ccf_aiops:
            # Override node names and topology for CCF AIOps (40 pods)
            self.config.NODE_NAMES = [
                'frontend-0','frontend-1','frontend-2','frontend2-0',
                'recommendationservice-0','recommendationservice-1','recommendationservice-2','recommendationservice2-0',
                'checkoutservice-0','checkoutservice-1','checkoutservice-2','checkoutservice2-0',
                'paymentservice-0','paymentservice-1','paymentservice-2','paymentservice2-0',
                'currencyservice-0','currencyservice-1','currencyservice-2','currencyservice2-0',
                'emailservice-0','emailservice-1','emailservice-2','emailservice2-0',
                'cartservice-0','cartservice-1','cartservice-2','cartservice2-0',
                'productcatalogservice-0','productcatalogservice-1','productcatalogservice-2','productcatalogservice2-0',
                'shippingservice-0','shippingservice-1','shippingservice-2','shippingservice2-0',
                'adservice-0','adservice-1','adservice-2','adservice2-0',
            ]
            # Use fully-connected fallback edges (per-service topology varies per event)
            n = len(self.config.NODE_NAMES)
            self.config.GAIA_EDGES = [[i, j] for i in range(n) for j in range(n) if i != j][:50]
            # ft_num must match the checkpoint; CCF AIOps trained with 10 (pod-level faults only)
            self.config.ft_num = 10
            logger.info("  [CCF AIOps] Using 40-node CCF AIOps topology for TVDiag inference")

        self._load_model(model_dir)

    @staticmethod
    def _infer_ft_num(state_dict: dict):
        """Infer ft_num from the FTI head output layer (typeClassifier final Linear)."""
        last = None
        for key, tensor in state_dict.items():
            if key.startswith("typeClassifier.") and key.endswith(".weight"):
                last = tensor
        if last is not None:
            return int(last.shape[0])
        return None

    @staticmethod
    def _infer_embedding_dim(state_dict: dict):
        """Infer alert_embedding_dim from the first SAGEConv fc weight.

        SAGEConv.fc = Linear(in_dim*2, out_dim); the first layer of each
        encoder is ``encoders.<modality>.layers.0.fc.weight`` with shape
        (out, in_dim*2).
        """
        import re as _re
        for key, tensor in state_dict.items():
            if _re.search(r"encoders\.\w+\.layers\.0\.fc\.weight$", key):
                return int(tensor.shape[1]) // 2
        return None

    def _load_model(self, model_dir: str):
        """Load tvdig.pt, inferring embedding dim and ft_num from the checkpoint."""
        ckpt_path = os.path.join(model_dir, "tvdig.pt")
        state = None
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            inferred_dim = self._infer_embedding_dim(state["model"])
            if inferred_dim and inferred_dim != self.config.alert_embedding_dim:
                logger.info(f"  [TVDiag] Checkpoint embedding dim {inferred_dim} != default "
                            f"{self.config.alert_embedding_dim}; adopting checkpoint dim")
                self.config.alert_embedding_dim = inferred_dim
            if not self._is_ccf_aiops:
                inferred_ft = self._infer_ft_num(state["model"])
                if inferred_ft and inferred_ft != self.config.ft_num:
                    self.config.ft_num = inferred_ft

        self.model = MainModel(self.config).to(self.device)
        if state is not None:
            self.model.load_state_dict(state["model"])
            self.model.eval()
            logger.info(f"TVDiag model loaded from {ckpt_path}")
        else:
            logger.warning(f"No checkpoint found at {ckpt_path}, model is untrained")

        # Load embedding cache
        cache_path = os.path.join(model_dir, "embedding_cache.pkl")
        self.embedding_cache = None
        if os.path.exists(cache_path):
            self.embedding_cache = load_embedding_cache(cache_path)

    @property
    def method_name(self) -> str:
        return "tvdig"

    def localize(self, task_info: dict) -> LocalizationResult:
        """Run TVDiag multimodal root cause localization.

        Expected task_info keys:
            - metric_data: np.ndarray (T, N) metric time series
            - data_head: List[str] metric names
            - n_init: int train/test split index
            - trace_data_path: str path to trace data
            - log_dir: str path to log fault directory
            - root_services: List[str] top root cause services from MEPFL
        """
        metric_data = task_info["metric_data"]
        data_head = task_info["data_head"]
        n_init = task_info.get("n_init", int(0.5 * len(metric_data)))
        trace_data_path = task_info.get("trace_data_path", "")
        log_dir = task_info.get("log_dir", "")
        root_services = task_info.get("root_services", [])

        node_names = self.config.NODE_NAMES

        # --- Step 1: Extract anomaly events from each modality ---
        metric_events = extract_metric_events_from_univariate(
            metric_data, data_head, n_init, node_names
        )
        trace_events = extract_trace_events_from_soCRCA(
            trace_data_path, node_names
        )
        log_events = extract_log_events_from_soCRCA(
            log_dir, root_services, node_names
        )

        logger.info(f"Events extracted: metric={sum(len(e) for e in metric_events)}, "
                     f"trace={sum(len(e) for e in trace_events)}, "
                     f"log={sum(len(e) for e in log_events)}")

        # --- Step 2: Build inference graph ---
        if self.embedding_cache is not None:
            graph = build_inference_graph_from_cache(
                self.config, self.embedding_cache,
                metric_events, trace_events, log_events,
            )
        else:
            graph = build_inference_graph(
                self.config, metric_events, trace_events, log_events,
            )

        # --- Step 3: Run model inference ---
        edge_index = graph["edge_index"].to(self.device)
        features = {mod: feat.to(self.device) for mod, feat in graph["features"].items()}

        with torch.no_grad():
            _, _, root_logit, type_logit = self.model(
                edge_index, [graph["num_nodes"]], features
            )

        # --- Step 4: Rank nodes by root cause score ---
        scores = root_logit.flatten().cpu().numpy()
        ranked_indices = np.argsort(scores)[::-1]
        ranked_services = [node_names[i] for i in ranked_indices]

        # Top-5 root cause services
        top5_services = ranked_services[:5]

        # Map service scores to metrics
        top5_metrics = self._services_to_metrics(top5_services, data_head, scores, node_names)

        # Failure type prediction
        type_pred = torch.argmax(type_logit, dim=1).item()
        if self._is_ccf_aiops:
            # ft_num=10 matches checkpoint (pod/service-level faults only)
            type_names = [
                "normal", "k8s容器读io负载", "k8s容器内存负载", "k8s容器网络资源包损坏",
                "k8s容器cpu负载", "k8s容器网络丢包", "k8s容器网络资源包重复发送",
                "k8s容器网络延迟", "k8s容器进程中止", "k8s容器写io负载",
            ]
        else:
            type_names = ["normal", "login failure", "memory anomalies",
                          "file moving program", "cpu anomalies"]
        failure_type = type_names[type_pred] if type_pred < len(type_names) else "unknown"

        # --- Step 5: Format results ---
        root_service_str = ",".join(f"({i+1}){s}" for i, s in enumerate(top5_services))
        root_metric_str = "Top 5 root cause metrics is:" + ",".join(
            f"({i+1}){m}" for i, m in enumerate(top5_metrics[:5])
        ) + "."

        # Build descriptions
        trace_anomaly = self._build_trace_description(top5_services, trace_events, node_names)
        metric_anomaly = self._build_metric_description(top5_services, metric_events, node_names, scores)
        log_anomaly = self._build_log_description(top5_services, log_events, node_names)

        root_cause_knowledge = (
            f"[TVDiag Multimodal RCA] Root cause analysis using joint metric+trace+log GNN.\n"
            f"Predicted failure type: {failure_type}\n"
            f"{root_metric_str}\n"
            f"Top 5 root cause services: {root_service_str}"
        )

        return LocalizationResult(
            root_service=root_service_str,
            root_metric=root_metric_str,
            trace_anomaly=trace_anomaly,
            metric_anomaly=metric_anomaly,
            log_anomaly=log_anomaly,
            root_cause_knowledge=root_cause_knowledge,
            raw_root_services=top5_services,
            raw_root_metrics=top5_metrics[:5],
            raw_root_scores=scores,
        )

    # ----- Internal helpers -----

    def _services_to_metrics(self, services, data_head, scores, node_names):
        """Map root cause services to their most anomalous metrics."""
        metric_scores = {}
        for metric_idx, metric_name in enumerate(data_head):
            for svc in services:
                if metric_name.startswith(svc):
                    node_idx = node_names.index(svc) if svc in node_names else 0
                    metric_scores[metric_name] = scores[node_idx] if node_idx < len(scores) else 0
                    break

        ranked = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in ranked[:5]]

    def _build_trace_description(self, top5_services, trace_events, node_names):
        parts = ["[TVDiag Trace Analysis]"]
        has_events = False
        for svc in top5_services:
            if svc in node_names:
                idx = node_names.index(svc)
                events = trace_events[idx]
                if events:
                    has_events = True
                    parts.append(f"  {svc}: {len(events)} anomalous trace spans")
                    for ev in events[:3]:
                        parts.append(f"    - {ev}")
        if not has_events:
            parts.append("  No significant trace anomalies detected in top root cause services.")
        return "\n".join(parts)

    def _build_metric_description(self, top5_services, metric_events, node_names, scores):
        parts = ["[TVDiag Metric Analysis] Multimodal root cause scores:"]
        for i, svc in enumerate(top5_services):
            idx = node_names.index(svc) if svc in node_names else 0
            score = scores[idx] if idx < len(scores) else 0
            events = metric_events[idx] if idx < len(metric_events) else []
            parts.append(f"  ({i+1}) {svc}: score={score:.4f}, anomalous_metrics={len(events)}")
        return "\n".join(parts)

    def _build_log_description(self, top5_services, log_events, node_names):
        parts = ["[TVDiag Log Analysis]"]
        has_errors = False
        for svc in top5_services:
            if svc in node_names:
                idx = node_names.index(svc)
                events = log_events[idx]
                if events:
                    has_errors = True
                    parts.append(f"  {svc}: {len(events)} error log events")
        if not has_errors:
            parts.append("  No error logs found in top root cause services.")
        return "\n".join(parts)
