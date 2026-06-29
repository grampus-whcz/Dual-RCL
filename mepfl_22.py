import numpy as np
import math
import json
import os
import sys
import datetime
import time
import datetime
from dateutil.parser import parse
import json
import csv
import codecs
from loguru import logger
from tqdm import tqdm

from data.read_data import *
#from microrank.config import ExpConfig

from sklearn.preprocessing import StandardScaler
from spectrum import *
from randomwalk_trace import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


total_service_list = [
        'frontend-0','frontend-1','frontend-2','frontend2-0',
        'recommendationservice-0','recommendationservice-1','recommendationservice-2','recommendationservice2-0',
        'checkoutservice-0','checkoutservice-1','checkoutservice-2','checkoutservice2-0',
        'paymentservice-0','paymentservice-1','paymentservice-2','paymentservice2-0',
        'currencyservice-0','currencyservice-1','currencyservice-2','currencyservice2-0',
        'emailservice-0','emailservice-1','emailservice-2','emailservice2-0',
        'cartservice-0','cartservice-1','cartservice-2','cartservice2-0',
        'productcatalogservice-0','productcatalogservice-1','productcatalogservice-2','productcatalogservice2-0',
        'shippingservice-0','shippingservice-1','shippingservice-2','shippingservice2-0',
        'adservice-0' ,'adservice-1','adservice-2','adservice2-0'
]

service_index_dict = {
    service: index
    for index, service in enumerate(total_service_list)
}


def get_trace_service_list(trace: Trace):
    service_name_set = set()

    queue = [trace.root_span]
    while len(queue):
        span = queue.pop(0)
        queue.extend(span.children_span_list)
        service_name_set.add(span.service_name)
    return service_name_set


def get_trace_list_vector(trace_list: List[Trace]):
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
            if span.status_code=='Ok' or span.status_code=='OK':
                status=0
            else:
                status = int(span.status_code)
            process_time = span.duration - max([child.duration for child in span.children_span_list], default=0.0)

            trace.vector[-1 + (service_index_dict[span.service_name] + 1) * 3] = duration
            trace.vector[0 + (service_index_dict[span.service_name] + 1) * 3] = status
            trace.vector[1 + (service_index_dict[span.service_name] + 1) * 3] = process_time



class MLPWithSoftmax(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPWithSoftmax, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class MLPWithSoftmaxV2(nn.Module):
    """3-layer MLP used by GPU-trained models (train_mepfl_aiops22.py)."""
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


_mepfl_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_mlp_model(model_dir='./mepfl_model_aiops22'):
    """Load MLP model, supporting both sklearn (.pkl) and PyTorch (.pt) formats."""
    meta_path = os.path.join(model_dir, 'meta.json')
    pt_path = os.path.join(model_dir, 'mlp_model.pt')
    pkl_path = os.path.join(model_dir, 'mlp_model.pkl')

    # Prefer PyTorch model if meta.json exists
    if os.path.exists(meta_path) and os.path.exists(pt_path):
        import json
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        hidden = meta.get('hidden_size', 256)
        model = MLPWithSoftmaxV2(meta['input_size'], hidden, meta['output_size'])
        model.load_state_dict(torch.load(pt_path, map_location=_mepfl_device, weights_only=True))
        model.to(_mepfl_device)
        model.eval()
        logger.info(f'  Loaded PyTorch MLP from {pt_path} (device={_mepfl_device})')
        # Store reverse_label_map for converting predicted indices back to service indices
        reverse_map = meta.get('reverse_label_map', {})
        return (model, reverse_map), 'pytorch'

    # Fall back to sklearn
    if os.path.exists(pkl_path):
        mlp = pickle.load(open(pkl_path, 'rb'))
        logger.info(f'  Loaded sklearn MLP from {pkl_path}')
        return mlp, 'sklearn'

    raise FileNotFoundError(f'No MLP model found in {model_dir}/')


def mepfl_22(path, model_dir='./mepfl_model_aiops22'):


    logger.info('Loading models')
    rf_model = pickle.load(
        open(os.path.join(model_dir, 'rf_model.pkl'), 'rb')
    )
    mlp_model, mlp_type = _load_mlp_model(model_dir)

    total_fault_count = 0
    fault_injection_list = pickle.load(
        open(path, 'rb')
    )
    logger.info(f'length: {len(fault_injection_list)}')

    pred_index_list = []
    total_fault_count += len(fault_injection_list)

    # Default ranking (full service list) in case all fault injections are skipped
    sorted_service_list = list(total_service_list)

    for fault_injection in fault_injection_list:
        
        rows = fault_injection['trace_list']
        trace_list = get_trace_list_from_rows(rows)

        if len(trace_list) <= 10:
            total_fault_count -= 1
            continue

        get_trace_list_vector(trace_list)
        trace_vector_list = [_.vector for _ in trace_list]
        trace_anomaly_list = rf_model.predict(trace_vector_list)
        anomaly_trace_list = []
        for index in range(0, len(trace_list)):
            if trace_anomaly_list[index] == 1:
                anomaly_trace_list.append(trace_list[index])

        loc_trace_vector_list = [_.vector for _ in anomaly_trace_list]
        if len(loc_trace_vector_list) == 0:
            continue

        if mlp_type == 'pytorch':
            # PyTorch GPU inference — mlp_model is (model, reverse_label_map)
            pt_model, reverse_map = mlp_model
            with torch.no_grad():
                x_tensor = torch.FloatTensor(np.array(loc_trace_vector_list)).to(_mepfl_device)
                probs = pt_model(x_tensor).cpu().numpy()
            # probs shape: (n_traces, n_mapped_classes)
            # Remap back to full service list (40 entries)
            sum_proba = np.zeros((len(total_service_list)))
            for prob in probs:
                for mapped_idx, p in enumerate(prob):
                    orig_idx = reverse_map.get(str(mapped_idx), mapped_idx)
                    if 0 <= orig_idx < len(total_service_list):
                        sum_proba[orig_idx] += p
        else:
            probs = mlp_model.predict_proba(loc_trace_vector_list)
            logger.info(f"{probs.shape}")
            sum_proba = np.zeros((len(total_service_list)))
            for prob in probs:
                sum_proba += prob
        service_score_list = []
        for index in range(0, len(total_service_list)):
            service_score_list.append((total_service_list[index], sum_proba[index]))
        service_score_list.sort(key=lambda x:x[1], reverse=True)
        sorted_service_list = [_[0] for _ in service_score_list]
        
    return sorted_service_list
        
