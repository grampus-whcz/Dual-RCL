#!/usr/bin/env python3
"""
Build TVDiag-format training data for CCF AIOps 2022 (v2 — fixed embeddings).

Fixes vs v1:
  1. Metric feature extraction: correct LONG-format xlsx parsing (rows=metrics, cols=timestamps)
  2. Embedding normalization: z-score standardize per modality (eliminates scale 10^9 issues)
  3. Richer features: more statistical descriptors
  4. Supports multiple training dates (03-20, 03-21, 03-24)
"""
import argparse, datetime, json, os, pickle, sys
from collections import defaultdict
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SERVICE_LIST = [
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
SERVICE_IDX = {s: i for i, s in enumerate(SERVICE_LIST)}
EMB_DIM = 128
GT_TRAIN_DIR = "/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/training_data_with_faults/groundtruth"


def map_cmdb_to_service(cmdb_id, level):
    if level == 'pod':
        return cmdb_id if cmdb_id in SERVICE_IDX else None
    elif level == 'service':
        for pod in SERVICE_LIST:
            if pod.startswith(cmdb_id + '-'):
                return pod
        return None
    return None


def process_tracerca(tracerca_pkl, max_traces=1500):
    """Load pkl ONCE, extract topology + per-service trace stats."""
    with open(tracerca_pkl, 'rb') as f:
        data = pickle.load(f)
    edges_set = set()
    services_in_event = set()
    svc_stats = defaultdict(lambda: {'count':0,'anomaly':0,'durs':[],'pts':[],'children':[]})
    for fi in data:
        traces = fi['trace_list']
        for trace in traces[:max_traces]:
            queue = [trace.root_span]
            while queue:
                span = queue.pop(0)
                sn = span.service_name
                if sn in SERVICE_IDX:
                    services_in_event.add(sn)
                    s = svc_stats[sn]
                    s['count'] += 1
                    try:
                        if int(float(getattr(span, 'status_code', '0'))) != 0:
                            s['anomaly'] += 1
                    except (ValueError, TypeError):
                        pass
                    s['durs'].append(float(span.duration))
                    pt = float(span.duration) - max(
                        [float(c.duration) for c in span.children_span_list], default=0.0)
                    s['pts'].append(pt)
                    s['children'].append(len(span.children_span_list))
                for child in span.children_span_list:
                    if span.service_name in SERVICE_IDX and child.service_name in SERVICE_IDX:
                        edges_set.add((span.service_name, child.service_name))
                    queue.append(child)
        break
    nodes = sorted(services_in_event)
    node_to_local = {n: i for i, n in enumerate(nodes)}
    edges = [[node_to_local[s], node_to_local[d]] for s, d in edges_set
             if s in node_to_local and d in node_to_local]
    return nodes, edges, svc_stats


def trace_stats_to_feat(svc_stats, nodes):
    """Richer trace features (12 dims per node)."""
    feats = {}
    for node in nodes:
        s = svc_stats.get(node, {'count':0,'anomaly':0,'durs':[],'pts':[],'children':[]})
        cnt = max(s['count'], 1)
        durs = np.array(s['durs']) if s['durs'] else np.array([0.0])
        pts = np.array(s['pts']) if s['pts'] else np.array([0.0])
        ch = np.array(s['children']) if s['children'] else np.array([0.0])
        feats[node] = np.array([
            np.log1p(float(s['count'])),              # log call count
            float(s['anomaly'])/cnt,                   # anomaly rate
            np.log1p(float(s['anomaly'])),             # log anomaly count
            np.log1p(float(np.mean(durs))),            # log mean duration
            float(np.std(durs)),                        # duration std
            np.log1p(float(np.max(durs))),             # log max duration
            np.log1p(float(np.mean(pts))),             # log mean process time
            float(np.std(pts)),                         # process time std
            np.log1p(float(np.max(pts))),              # log max process time
            float(np.median(durs)),                     # median duration
            float(np.mean(ch)),                         # avg fanout
            float(np.max(ch)),                          # max fanout
        ])
    return feats


def extract_metric_features(microcause_xlsx, nodes):
    """FIXED: correct LONG-format xlsx parsing.

    xlsx format: rows = metrics (e.g. 'frontend-1_container_cpu_usage_seconds'),
                 columns = 'metric' + 60 timestamps.
    """
    df = pd.read_excel(microcause_xlsx)
    ts_cols = [c for c in df.columns if c != 'metric']
    feats = {}
    for node in nodes:
        # Find rows where metric name starts with "{node}_"
        node_rows = df[df['metric'].astype(str).str.startswith(node + '_')]
        node_feats = []
        for _, row in node_rows.iterrows():
            series = np.nan_to_num(row[ts_cols].values.astype(float),
                                   nan=0.0, posinf=0.0, neginf=0.0)
            if len(series) == 0 or np.std(series) < 1e-8:
                stats = [0.0]*8
            else:
                # Relative change features (scale-invariant)
                rel = series / (np.abs(series).mean() + 1e-8)
                stats = [
                    float(np.mean(rel)), float(np.std(rel)),
                    float(np.max(rel)), float(np.min(rel)),
                    float(np.percentile(rel, 75) - np.percentile(rel, 25)),  # IQR
                    float(np.mean(np.abs(np.diff(rel))) if len(rel) > 1 else 0.0),  # volatility
                    float((series[-1] - series[0]) / (np.abs(series).mean() + 1e-8)),  # trend
                    float(np.sum(np.abs(np.diff(np.sign(np.diff(rel)))) > 0)),  # oscillation count
                ]
            node_feats.append(stats)
        feats[node] = np.array(node_feats).flatten() if node_feats else np.zeros(8)
    return feats


def extract_log_features(log_fault_dir, nodes):
    """Richer log features (8 dims per node)."""
    feats = {}
    for node in nodes:
        log_file = os.path.join(log_fault_dir, f"{node}.csv")
        n_lines, n_error, n_warn, n_info = 0, 0, 0, 0
        msg_lens = []
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file)
                n_lines = len(df)
                msg_col = 'message' if 'message' in df.columns else df.columns[-1]
                msgs = df[msg_col].astype(str)
                n_error = int(msgs.str.contains('error|exception|fail|fatal', case=False).sum())
                n_warn = int(msgs.str.contains('warn|caution', case=False).sum())
                n_info = n_lines - n_error - n_warn
                msg_lens = msgs.str.len().tolist()
            except Exception:
                pass
        ml = np.array(msg_lens) if msg_lens else np.array([0.0])
        feats[node] = np.array([
            np.log1p(float(n_lines)),          # log total
            np.log1p(float(n_error)),          # log errors
            np.log1p(float(n_warn)),           # log warnings
            float(n_error)/max(n_lines,1),     # error rate
            float(n_warn)/max(n_lines,1),      # warn rate
            float(n_lines > 0),                # has logs
            np.log1p(float(np.mean(ml))),      # log mean msg len
            float(np.std(ml)),                 # msg len std
        ])
    return feats


def project_and_normalize(feats_by_node, nodes, dim, pca=None, scaler=None, fit=False):
    """Project features to dim via PCA, then z-score normalize.
    Returns (embeddings_dict, pca_model, scaler_model)."""
    matrix, valid = [], []
    for node in nodes:
        v = feats_by_node.get(node)
        if v is None:
            continue
        matrix.append(v)
        valid.append(node)
    if not matrix:
        return {n: np.zeros(dim, dtype=np.float32) for n in nodes}, pca, scaler
    matrix = np.array(matrix, dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if fit:
        nc = min(dim, matrix.shape[0], matrix.shape[1])
        nc = max(nc, 1)
        try:
            pca = PCA(n_components=nc)
            reduced = pca.fit_transform(matrix)
        except Exception:
            reduced = matrix[:, :dim]
            pca = None
        # Fit scaler on reduced
        scaler = StandardScaler()
        try:
            reduced = scaler.fit_transform(reduced)
        except Exception:
            pass
    else:
        if pca is not None:
            try:
                reduced = pca.transform(matrix)
            except Exception:
                reduced = matrix[:, :dim]
        else:
            reduced = matrix[:, :dim]
        if scaler is not None:
            try:
                reduced = scaler.transform(reduced)
            except Exception:
                pass

    out = np.zeros((len(valid), dim), dtype=np.float32)
    uc = min(reduced.shape[1], dim)
    out[:, :uc] = reduced[:, :uc]
    result = {node: out[i] for i, node in enumerate(valid)}
    for node in nodes:
        if node not in result:
            result[node] = np.zeros(dim, dtype=np.float32)
    return result, pca, scaler


def load_gt(gt_file):
    events = []
    df = pd.read_csv(gt_file)
    for _, row in df.iterrows():
        ts = datetime.datetime.fromtimestamp(int(row['timestamp']))
        events.append((ts.strftime('%H-%M'), row['cmdb_id'], row['level'], row['failure_type']))
    return events


def process_cloudbed(prefix, cloudbed, date_str, gt_file, data_type, pca_cache, scaler_cache, max_traces):
    """Process one cloudbed, return list of event dicts."""
    if not os.path.exists(f"{prefix}_tracerca"):
        logger.warning(f"[SKIP] {prefix}")
        return []
    events = load_gt(gt_file)
    logger.info(f"  {prefix} ({date_str} {cloudbed}): {len(events)} events, split={data_type}")
    results = []
    for ei, (time_label, cmdb, level, fault) in enumerate(events):
        tracerca_pkl = f"{prefix}_tracerca/{time_label}.pkl"
        microcause = f"{prefix}_microcause/microcause_{date_str}_{time_label}.xlsx"
        log_dir = f"{prefix}_log_fault/{time_label}"
        if not (os.path.exists(tracerca_pkl) and os.path.exists(microcause)):
            continue
        try:
            nodes, edges, svc_stats = process_tracerca(tracerca_pkl, max_traces)
        except Exception as e:
            continue
        if len(nodes) < 2:
            continue
        root_service = map_cmdb_to_service(cmdb, level)
        if data_type == "train" and (root_service is None or root_service not in nodes):
            continue
        try:
            t_feats = trace_stats_to_feat(svc_stats, nodes)
            m_feats = extract_metric_features(microcause, nodes)
            l_feats = extract_log_features(log_dir, nodes)
        except Exception as e:
            continue
        fit = (data_type == "train")
        m_emb, pca_cache['metric'], scaler_cache['metric'] = project_and_normalize(
            m_feats, nodes, EMB_DIM, pca_cache.get('metric'), scaler_cache.get('metric'), fit)
        t_emb, pca_cache['trace'], scaler_cache['trace'] = project_and_normalize(
            t_feats, nodes, EMB_DIM, pca_cache.get('trace'), scaler_cache.get('trace'), fit)
        l_emb, pca_cache['log'], scaler_cache['log'] = project_and_normalize(
            l_feats, nodes, EMB_DIM, pca_cache.get('log'), scaler_cache.get('log'), fit)
        results.append({
            'nodes': nodes, 'edges': edges,
            'metric': [m_emb[n] for n in nodes],
            'trace': [t_emb[n] for n in nodes],
            'log': [l_emb[n] for n in nodes],
            'cmdb': cmdb, 'instance': root_service or cmdb,
            'level': level, 'fault': fault, 'datetime': f"{date_str} {time_label}",
            'data_type': data_type,
        })
        if (ei+1) % 10 == 0:
            logger.info(f"    [{ei+1}/{len(events)}]")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tvdig_data_ccf_v2")
    parser.add_argument("--max-traces", type=int, default=1500)
    args = parser.parse_args()

    os.makedirs(f"{args.output_dir}/tmp", exist_ok=True)
    os.makedirs(f"{args.output_dir}/raw", exist_ok=True)

    # Training data: 03-20 c1 + 03-21 c1/c2/c3 + 03-24 c3
    train_configs = [
        ('0320', 'cloudbed-1', '2022-03-20', 'groundtruth-k8s-1-2022-03-20.csv'),
        ('0321', 'cloudbed-1', '2022-03-21', 'groundtruth-k8s-1-2022-03-21.csv'),
        ('0321b', 'cloudbed-2', '2022-03-21', 'groundtruth-k8s-2-2022-03-21.csv'),
        ('0321c', 'cloudbed-3', '2022-03-21', 'groundtruth-k8s-3-2022-03-21.csv'),
        ('0324c', 'cloudbed-3', '2022-03-24', 'groundtruth-k8s-3-2022-03-24.csv'),
    ]
    # Test data: 03-20 c2/c3 (validation) — kept separate from training
    test_configs = [
        ('0320b', 'cloudbed-2', '2022-03-20', 'groundtruth-k8s-2-2022-03-20.csv'),
        ('0320c', 'cloudbed-3', '2022-03-20', 'groundtruth-k8s-3-2022-03-20.csv'),
    ]

    pca_cache, scaler_cache = {}, {}
    all_events = []
    eid = 0

    logger.info("=== Processing TRAINING data ===")
    for prefix, cloudbed, date_str, gt_name in train_configs:
        gt_file = os.path.join(GT_TRAIN_DIR, gt_name)
        if not os.path.exists(f"{prefix}_tracerca"):
            logger.warning(f"  [SKIP] {prefix} not preprocessed yet")
            continue
        evs = process_cloudbed(prefix, cloudbed, date_str, gt_file, "train",
                               pca_cache, scaler_cache, args.max_traces)
        for ev in evs:
            ev['eid'] = eid
            all_events.append(ev)
            eid += 1
        logger.info(f"  {prefix}: +{len(evs)} events (total {eid})")

    logger.info("=== Processing TEST data ===")
    for prefix, cloudbed, date_str, gt_name in test_configs:
        gt_file = os.path.join(GT_TRAIN_DIR, gt_name)
        if not os.path.exists(f"{prefix}_tracerca"):
            continue
        evs = process_cloudbed(prefix, cloudbed, date_str, gt_file, "test",
                               pca_cache, scaler_cache, args.max_traces)
        for ev in evs:
            ev['eid'] = eid
            all_events.append(ev)
            eid += 1
        logger.info(f"  {prefix}: +{len(evs)} events (total {eid})")

    # Save in TVDiag format
    metric_embs, trace_embs, log_embs = {}, {}, {}
    nodes_json, edges_json = {}, {}
    label_rows = []
    for ev in all_events:
        key = str(ev['eid'])
        metric_embs[key] = ev['metric']
        trace_embs[key] = ev['trace']
        log_embs[key] = ev['log']
        nodes_json[key] = ev['nodes']
        edges_json[key] = ev['edges']
        label_rows.append({
            "index": ev['eid'], "datetime": ev['datetime'],
            "service": ev['cmdb'], "instance": ev['instance'],
            "message": ev['fault'], "level": ev['level'],
            "anomaly_type": ev['fault'], "data_type": ev['data_type'],
        })

    with open(f"{args.output_dir}/tmp/metric.pkl",'wb') as f: pickle.dump(metric_embs, f)
    with open(f"{args.output_dir}/tmp/trace.pkl",'wb') as f: pickle.dump(trace_embs, f)
    with open(f"{args.output_dir}/tmp/log.pkl",'wb') as f: pickle.dump(log_embs, f)
    with open(f"{args.output_dir}/raw/nodes.json",'w') as f: json.dump(nodes_json, f)
    with open(f"{args.output_dir}/raw/edges.json",'w') as f: json.dump(edges_json, f)
    pd.DataFrame(label_rows).to_csv(f"{args.output_dir}/label.csv", index=False)

    n_tr = sum(1 for r in label_rows if r['data_type']=='train')
    n_te = sum(1 for r in label_rows if r['data_type']=='test')
    logger.info(f"=== DONE: {eid} events (train={n_tr}, test={n_te}) ===")

    # Verify embedding quality
    if metric_embs:
        arr = np.array(metric_embs['0'])
        logger.info(f"  metric emb: shape={arr.shape}, var={np.var(arr):.6f}, nonzero={ (arr!=0).mean():.2%}")


if __name__ == "__main__":
    main()
