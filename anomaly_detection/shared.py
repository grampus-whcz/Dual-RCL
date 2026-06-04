"""
Shared utilities for anomaly detection modules.

Provides sliding-window conversion, metric-to-service mapping, and
natural-language description generation — reused by all detectors.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger("anomaly_detection")


# ---------------------------------------------------------------------------
#  Metric-to-service mapping
# ---------------------------------------------------------------------------

_KNOWN_SERVICES = [
    "webservice1", "webservice2",
    "redisservice1", "redisservice2",
    "mobservice1", "mobservice2",
    "logservice1", "logservice2",
    "dbservice1", "dbservice2",
]


def metrics_to_services(metrics: List[str]) -> List[str]:
    """Extract service names from metric names like 'webservice1_docker_cpu'."""
    services = []
    for m in metrics:
        svc = m.split("_")[0] if m else m
        for known in _KNOWN_SERVICES:
            if m.startswith(known):
                svc = known
                break
        if svc not in services:
            services.append(svc)
    return services


# ---------------------------------------------------------------------------
#  Description builder
# ---------------------------------------------------------------------------

def build_description(results: dict, data_head: List[str]) -> str:
    """Generate natural-language summary for downstream LLM consumption."""
    parts: List[str] = []

    n_anom = int(results["anomalies"].sum())
    n_total = len(results["anomalies"])
    method_name = results.get("method", "Multivariate")

    if n_anom > 0:
        parts.append(
            f"[{method_name} Detection] Detected system-level anomalies at "
            f"{n_anom}/{n_total} timesteps (threshold={results['threshold']:.4f}). "
            f"This indicates the joint state of multiple metrics deviates from normal."
        )
    else:
        parts.append(
            f"[{method_name} Detection] No system-level anomaly detected. "
            "The joint behaviour of all metrics remains within normal bounds."
        )

    # Top-5 anomalous metrics
    parts.append("Top-5 anomalous metrics ranked by multivariate reconstruction error:")
    for i, (idx, score) in enumerate(results["feature_ranking"][:5]):
        parts.append(f"  ({i + 1}) {data_head[idx]}  (score={score:.4f})")

    # Per-feature anomaly frequency
    if n_anom > 0:
        anomaly_ts = np.where(results["anomalies"])[0]
        freq: Dict[str, int] = {}
        for t in anomaly_ts:
            for fidx in results["anomaly_features"][t]:
                name = data_head[fidx]
                freq[name] = freq.get(name, 0) + 1
        if freq:
            parts.append("Anomalous feature frequency during detected windows:")
            for name, cnt in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]:
                parts.append(f"  {name}: anomalous in {cnt}/{n_anom} windows")

    return "\n".join(parts)
