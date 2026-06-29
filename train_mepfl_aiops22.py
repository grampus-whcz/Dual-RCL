#!/usr/bin/env python3
"""
Train MEPFL (Micro-service Effective Potential Fault Localization) models
for the CCF AIOps 2022 dataset — GPU-accelerated version.

Produces:
  - mepfl_model_aiops22/rf_model.pkl   (RandomForest anomaly classifier, CPU)
  - mepfl_model_aiops22/mlp_model.pt    (PyTorch MLP root-service localizer, GPU)
  - mepfl_model_aiops22/meta.json       (Model metadata: input_size, output_size)

Usage:
  python train_mepfl_aiops22.py [--cloudbeds 1 2 3] [--output-dir mepfl_model_aiops22] [--epochs 200]
"""
import os
import sys
import csv
import json
import pickle
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import shared utilities
from data.read_data import get_trace_list_from_rows, Trace

# ---- Device ----
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- CCF AIOps 2022 service list (40 pods) ----
total_service_list = [
    'frontend-0', 'frontend-1', 'frontend-2', 'frontend2-0',
    'recommendationservice-0', 'recommendationservice-1', 'recommendationservice-2', 'recommendationservice2-0',
    'checkoutservice-0', 'checkoutservice-1', 'checkoutservice-2', 'checkoutservice2-0',
    'paymentservice-0', 'paymentservice-1', 'paymentservice-2', 'paymentservice2-0',
    'currencyservice-0', 'currencyservice-1', 'currencyservice-2', 'currencyservice2-0',
    'emailservice-0', 'emailservice-1', 'emailservice-2', 'emailservice2-0',
    'cartservice-0', 'cartservice-1', 'cartservice-2', 'cartservice2-0',
    'productcatalogservice-0', 'productcatalogservice-1', 'productcatalogservice-2', 'productcatalogservice2-0',
    'shippingservice-0', 'shippingservice-1', 'shippingservice-2', 'shippingservice2-0',
    'adservice-0', 'adservice-1', 'adservice-2', 'adservice2-0',
]

service_index_dict = {service: index for index, service in enumerate(total_service_list)}


class MLPWithSoftmax(nn.Module):
    """PyTorch MLP with Softmax output for root-service localization."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


def get_trace_service_list(trace):
    service_name_set = set()
    queue = [trace.root_span]
    while len(queue):
        span = queue.pop(0)
        queue.extend(span.children_span_list)
        service_name_set.add(span.service_name)
    return service_name_set


def get_trace_list_vector(trace_list):
    for trace in trace_list:
        trace.vector = np.zeros((2 + len(total_service_list) * 3))
        trace.vector[0] = len(total_service_list)
        trace.vector[1] = len(get_trace_service_list(trace))

        queue = [trace.root_span]
        while len(queue):
            span = queue.pop(0)
            temp_list = sorted(span.children_span_list, key=lambda x: x.start_time, reverse=False)
            queue.extend(temp_list)

            duration = span.duration
            try:
                status = int(float(span.status_code))
            except (ValueError, TypeError):
                status = 0
            process_time = span.duration - max(
                [child.duration for child in span.children_span_list], default=0.0
            )

            if span.service_name in service_index_dict:
                idx = service_index_dict[span.service_name]
                trace.vector[-1 + (idx + 1) * 3] = duration
                trace.vector[0 + (idx + 1) * 3] = status
                trace.vector[1 + (idx + 1) * 3] = process_time


def load_ground_truth(gt_path):
    import pandas as pd
    events = []
    df = pd.read_csv(gt_path)
    for _, row in df.iterrows():
        events.append({
            'timestamp': row['timestamp'],
            'level': row['level'],
            'cmdb_id': row['cmdb_id'],
            'failure_type': row['failure_type'],
        })
    return events


def map_cmdb_to_pod(cmdb_id, level):
    if level == 'pod':
        return cmdb_id if cmdb_id in service_index_dict else None
    elif level == 'service':
        for pod in total_service_list:
            if pod.startswith(cmdb_id + '-'):
                return pod
        return None
    else:
        return None


def train_mlp_gpu(X_train, y_train, input_size, output_size, epochs=200, batch_size=4096, lr=0.001):
    """Train MLP on GPU using PyTorch."""
    logger.info(f'  MLP GPU training: device={DEVICE}, epochs={epochs}, batch_size={batch_size}')

    model = MLPWithSoftmax(input_size, 256, output_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_tensor = torch.LongTensor(y_train).to(DEVICE)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            acc = correct / total
            avg_loss = total_loss / total
            logger.info(f'    Epoch {epoch+1}/{epochs} — loss={avg_loss:.4f}, acc={acc:.4f}')

    # Final accuracy
    model.eval()
    with torch.no_grad():
        all_out = model(X_tensor)
        _, preds = torch.max(all_out, 1)
        final_acc = (preds == y_tensor).float().mean().item()
    logger.info(f'  MLP final accuracy: {final_acc:.4f}')

    return model


def main():
    parser = argparse.ArgumentParser(description='Train MEPFL models for CCF AIOps 2022 (GPU)')
    parser.add_argument('--cloudbeds', nargs='+', type=int, default=[1, 2, 3],
                        help='Cloudbed numbers to use for training')
    parser.add_argument('--output-dir', type=str, default='mepfl_model_aiops22',
                        help='Output directory for trained models')
    parser.add_argument('--project-dir', type=str, default='.',
                        help='SoC-RCA project root directory')
    parser.add_argument('--epochs', type=int, default=200, help='MLP training epochs')
    parser.add_argument('--batch-size', type=int, default=4096, help='MLP batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='MLP learning rate')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.project_dir)

    logger.info(f'Device: {DEVICE}')
    if DEVICE.type == 'cuda':
        logger.info(f'  GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

    gt_dir = '/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/training_data_with_faults/groundtruth'
    prefixes = {1: '0320', 2: '0320b', 3: '0320c'}

    classification_traces = []
    localization_traces = []

    for cb_num in args.cloudbeds:
        prefix = prefixes[cb_num]
        tracerca_dir = f'{prefix}_tracerca'
        gt_file = os.path.join(gt_dir, f'groundtruth-k8s-{cb_num}-2022-03-20.csv')

        if not os.path.isdir(tracerca_dir):
            logger.warning(f'  [SKIP] {tracerca_dir} not found')
            continue
        if not os.path.exists(gt_file):
            logger.warning(f'  [SKIP] {gt_file} not found')
            continue

        gt_events = load_ground_truth(gt_file)
        gt_map = {}
        for ev in gt_events:
            ts_hour = datetime.datetime.fromtimestamp(ev['timestamp']).strftime('%H-%M')
            pod = map_cmdb_to_pod(ev['cmdb_id'], ev['level'])
            gt_map[ts_hour] = (pod, ev['cmdb_id'], ev['level'])

        pkl_files = sorted([f for f in os.listdir(tracerca_dir) if f.endswith('.pkl')])
        logger.info(f'  Cloudbed-{cb_num}: {len(pkl_files)} events, {len(gt_events)} GT events')

        for pkl_file in pkl_files:
            time_label = pkl_file.replace('.pkl', '')
            pkl_path = os.path.join(tracerca_dir, pkl_file)

            with open(pkl_path, 'rb') as f:
                fault_injections = pickle.load(f)

            gt_pod, gt_cmdb, gt_level = gt_map.get(time_label, (None, None, None))

            for fi in fault_injections:
                rows = fi['trace_list']
                trace_list = get_trace_list_from_rows(rows)

                if len(trace_list) <= 10:
                    continue

                for trace in trace_list:
                    trace_services = get_trace_service_list(trace)
                    if gt_pod and gt_pod in trace_services:
                        classification_traces.append(trace)
                        localization_traces.append(trace)
                        trace.anomaly_type = 1
                        trace.fault_service_index = service_index_dict[gt_pod]
                    else:
                        classification_traces.append(trace)
                        trace.anomaly_type = 0

    logger.info(f'Total traces: classification={len(classification_traces)}, '
                f'localization={len(localization_traces)}')

    if len(localization_traces) < 10:
        logger.error('Not enough training data. Check that tracerca pkl files exist.')
        sys.exit(1)

    # Compute feature vectors
    logger.info('Computing trace feature vectors...')
    get_trace_list_vector(classification_traces)
    get_trace_list_vector(localization_traces)

    # ---- Train RF anomaly classifier (CPU, fast with n_jobs=-1) ----
    cls_x = np.array([t.vector for t in classification_traces])
    cls_y = np.array([t.anomaly_type for t in classification_traces])
    logger.info(f'RF training: {len(cls_x)} samples, positive={sum(cls_y)}')
    rf_model = RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1)
    rf_model.fit(cls_x, cls_y)
    rf_acc = accuracy_score(cls_y[:min(10000, len(cls_y))],
                            rf_model.predict(cls_x[:min(10000, len(cls_x))]))
    logger.info(f'RF accuracy: {rf_acc:.4f}')

    # Free memory
    del classification_traces

    # ---- Train MLP root-service localizer on GPU ----
    loc_x = np.array([t.vector for t in localization_traces], dtype=np.float32)
    loc_y_raw = np.array([t.fault_service_index for t in localization_traces], dtype=np.int64)

    # Remap labels to contiguous range [0, n_classes) — required by CrossEntropyLoss
    unique_labels = sorted(set(loc_y_raw.tolist()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    loc_y = np.array([label_map[y] for y in loc_y_raw], dtype=np.int64)
    n_classes = len(unique_labels)
    input_size = loc_x.shape[1]
    logger.info(f'MLP training: {len(loc_x)} samples, input_size={input_size}, '
                f'classes={n_classes}, unique_faults={unique_labels}')

    mlp_model = train_mlp_gpu(loc_x, loc_y, input_size, n_classes,
                               epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # ---- Save models ----
    rf_path = os.path.join(args.output_dir, 'rf_model.pkl')
    mlp_path = os.path.join(args.output_dir, 'mlp_model.pt')
    meta_path = os.path.join(args.output_dir, 'meta.json')

    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    torch.save(mlp_model.state_dict(), mlp_path)
    with open(meta_path, 'w') as f:
        json.dump({
            'input_size': input_size,
            'hidden_size': 256,
            'output_size': n_classes,
            'service_list': total_service_list,
            'label_map': label_map,
            'reverse_label_map': {v: k for k, v in label_map.items()},
            'device': str(DEVICE),
            'rf_accuracy': float(rf_acc),
            'framework': 'pytorch',
        }, f, indent=2)

    logger.info(f'Models saved to {args.output_dir}/')
    logger.info(f'  rf_model.pkl:  {os.path.getsize(rf_path)/1024/1024:.1f} MB')
    logger.info(f'  mlp_model.pt:  {os.path.getsize(mlp_path)/1024:.1f} KB')
    logger.info(f'  meta.json:     {os.path.getsize(meta_path)} B')
    logger.info('Done!')


if __name__ == '__main__':
    main()
