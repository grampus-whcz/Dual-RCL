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
        'webservice1', 'webservice2', 'redisservice2', 'redisservice1', 'mobservice1', 'logservice1', 'mobservice2', 'logservice2', 'dbservice2', 'dbservice1' 
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
    
def mepfl(path):
    
    
    logger.info('Loading models')
    rf_model = pickle.load(
        open('./mepfl_model_gaia/rf_model.pkl', 'rb')
    )
    mlp_model = pickle.load(
        open('./mepfl_model_gaia/mlp_model.pkl', 'rb')
    )

    total_fault_count = 0
    fault_injection_list = pickle.load(
        open(path, 'rb')
    )
    logger.info(f'length: {len(fault_injection_list)}')
    
    pred_index_list = []
    total_fault_count += len(fault_injection_list)
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
        probs = mlp_model.predict_proba(loc_trace_vector_list)
        logger.info(f"{probs.shape}")
        sum_proba = np.zeros((len(total_service_list)))

        for prob in probs:
            # print(prob)
            sum_proba += prob
        
        service_score_list = []
        for index in range(0, len(total_service_list)):
            service_score_list.append((total_service_list[index], sum_proba[index]))
        service_score_list.sort(key=lambda x:x[1], reverse=True)
        sorted_service_list = [_[0] for _ in service_score_list]
        
    return sorted_service_list
        

        
    
