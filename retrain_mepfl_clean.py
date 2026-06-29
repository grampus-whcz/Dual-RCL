import numpy as np
import sys; sys.path.insert(0, "mepfl_model"); sys.path.insert(0, ".")
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

from sklearn.preprocessing import StandardScaler
from spectrum import *
from randomwalk import *

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
            status = int(span.status)
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
        self.softmax = nn.Softmax(dim=1)  # 在维度1上进行 softmax 操作

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    

if __name__ == '__main__':
    
    file_list = os.listdir('/'.join(['/root/shared-nvme/work/code/RCA/2026/SoC-RCA/Datasets/GAIA', 'fault_injection_tracerank']))
    # train_set_index = int(len(file_list) * 0.2)
    train_set_index = 5
    # train_set_index = 1

    if (not os.path.exists('/root/shared-nvme/work/code/RCA/2026/SoC-RCA/mepfl_model_gaia_clean/rf_model.pkl')) and (not os.path.exists('./mepfl_model_gaia_clean/mlp_model.pkl') and (not os.path.exists('./mepfl_model_gaia_clean/temp_fault_injection.pkl'))):
        logger.info(f'building train set')
        classification_train_trace_list = []
        localization_trace_trace_list = []

        temp_test_fault_injection_list = []
        valid_fault_injection_count = 0
        stop_flag = 2429

        TRAIN_FILES_CLEAN = ['fault_injection_list_2021-07-02.pkl','fault_injection_list_2021-07-03.pkl','fault_injection_list_2021-07-04.pkl','fault_injection_list_2021-07-05.pkl','fault_injection_list_2021-07-06.pkl']
        for file in TRAIN_FILES_CLEAN:
            fault_injection_list = pickle.load(
            open('/'.join(['/root/shared-nvme/work/code/RCA/2026/SoC-RCA/Datasets/GAIA', 'fault_injection_tracerank', f'{file}']), 'rb')
            )
            logger.info(f'{file}, length: {len(fault_injection_list)}')
            for fault_injection in fault_injection_list:
                if valid_fault_injection_count >= stop_flag:
                    temp_test_fault_injection_list.append(fault_injection)
                    continue

                rows = fault_injection['trace_list']
                trace_list = get_trace_list_from_rows(rows)

                if len(trace_list) <= 10:
                    continue

                valid_fault_injection_count += 1

                for trace in trace_list:
                    if (trace.root_span.start_time - fault_injection['time']).total_seconds() <= 60:
                        if fault_injection['service'] in get_trace_service_list(trace):
                            classification_train_trace_list.append(trace)
                            localization_trace_trace_list.append(trace)
                            trace.anomaly_type = 1
                            trace.fault_service_index = service_index_dict[fault_injection['service']]
                    else:
                        classification_train_trace_list.append(trace)
                        trace.anomaly_type = 0

        logger.info(f"trace trace list length: {len(classification_train_trace_list), len(localization_trace_trace_list)}")
        logger.info('get train trace list vector')
        get_trace_list_vector(classification_train_trace_list)
        get_trace_list_vector(localization_trace_trace_list)

        cls_train_x_list = [_.vector for _ in classification_train_trace_list]
        cls_train_y_list = [_.anomaly_type for _ in classification_train_trace_list]
        rf_model = RandomForestClassifier(random_state=0)
        rf_model.fit(cls_train_x_list, cls_train_y_list)
        temp_y_true = cls_train_y_list[0:10000]
        temp_y_pred = rf_model.predict(cls_train_x_list[0:10000])
        accuracy = accuracy_score(temp_y_true, temp_y_pred)
        logger.info(f'RF model predict accuracy: {accuracy}')


        loc_train_x_list = [_.vector for _ in localization_trace_trace_list]
        loc_train_y_list = [_.fault_service_index for _ in localization_trace_trace_list]
        mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        mlp_model.fit(loc_train_x_list, loc_train_y_list)
        temp_y_pred = mlp_model.predict(loc_train_x_list[0:10000])
        temp_y_true = loc_train_y_list[0:10000]
        accuracy = accuracy_score(temp_y_true, temp_y_pred)
        logger.info(f'MLP model predict accuracy: {accuracy}')

        pickle.dump(rf_model, open('/root/shared-nvme/work/code/RCA/2026/SoC-RCA/mepfl_model_gaia_clean/rf_model.pkl', 'wb'))
        pickle.dump(mlp_model, open('/root/shared-nvme/work/code/RCA/2026/SoC-RCA/mepfl_model_gaia_clean/mlp_model.pkl', 'wb'))
        pickle.dump(temp_test_fault_injection_list, open('/root/shared-nvme/work/code/RCA/2026/SoC-RCA/mepfl_model_gaia_clean/temp_fault_injection.pkl', 'wb'))
    
    else:
        logger.info('Loading models')
        rf_model = pickle.load(
            open('/root/shared-nvme/work/code/RCA/2026/SoC-RCA/mepfl_model_gaia_clean/rf_model.pkl', 'rb')
        )
        mlp_model = pickle.load(
            open('/root/shared-nvme/work/code/RCA/2026/SoC-RCA/mepfl_model_gaia_clean/mlp_model.pkl', 'rb')
        )
        temp_test_fault_injection_list = pickle.load(
            open('/root/shared-nvme/work/code/RCA/2026/SoC-RCA/mepfl_model_gaia_clean/temp_fault_injection.pkl', 'rb')
        )


    total_pred_index_list = []
    total_fault_count = 0
    begin = True
    for file in file_list[train_set_index: ]:
        fault_injection_list = pickle.load(
        open('/'.join(['/root/shared-nvme/work/code/RCA/2026/SoC-RCA/Datasets/GAIA', 'fault_injection_tracerank', f'{file}']), 'rb')
        )
        if begin:
            fault_injection_list.extend(temp_test_fault_injection_list)
            begin = False

        logger.info(f'{file}, length: {len(fault_injection_list)}')
        
        pred_index_list = []
        total_fault_count += len(fault_injection_list)

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
                

            if fault_injection['service'] in sorted_service_list:
                logger.info(f"Root cause service: {fault_injection['service']}, index: {sorted_service_list.index(fault_injection['service'])}")
                pred_index_list.append(sorted_service_list.index(fault_injection['service']))
            else:
                logger.info(f"Root cause service: {fault_injection['service']}, index: -1")
                pred_index_list.append(-1)

            
        try:
            ptopK_1 = (pred_index_list.count(0)) / len(pred_index_list)
            ptopK_3 = (pred_index_list.count(0) + pred_index_list.count(1) + pred_index_list.count(2)) / len(pred_index_list)
            ptopK_5 = (pred_index_list.count(0) + pred_index_list.count(1) + pred_index_list.count(2) + pred_index_list.count(3) + pred_index_list.count(4)) / len(pred_index_list)
            
            logger.info(f'{file}: {len(pred_index_list)} TopK_1: {ptopK_1}, TopK_3: {ptopK_3}, TopK_5: {ptopK_5}')
            with open('/root/shared-nvme/work/code/RCA/2026/SoC-RCA/mepfl_model_gaia_clean/result.txt', 'a', encoding='utf-8') as fout:
                fout.write(f'{file}: {len(pred_index_list)} TopK_1: {ptopK_1}, TopK_3: {ptopK_3}, TopK_5: {ptopK_5} \n')
        
        except:
            continue

        total_pred_index_list.extend(pred_index_list)
    

    
    topK_1 = (total_pred_index_list.count(0)) / len(total_pred_index_list)
    topK_3 = (total_pred_index_list.count(0) + total_pred_index_list.count(1) + total_pred_index_list.count(2)) / len(total_pred_index_list)
    topK_5 = (total_pred_index_list.count(0) + total_pred_index_list.count(1) + total_pred_index_list.count(2) + total_pred_index_list.count(3) + total_pred_index_list.count(4)) / len(total_pred_index_list)

    logger.info(f'Summary TopK_1: {topK_1}, TopK_3: {topK_3}, TopK_5: {topK_5}')
    
    with open('/root/shared-nvme/work/code/RCA/2026/SoC-RCA/mepfl_model_gaia_clean/result.txt', 'a', encoding='utf-8') as fout:
        fout.write(f'Summary TopK_1: {topK_1}, TopK_3: {topK_3}, TopK_5: {topK_5} \n')
        fout.write(f'total fault injection number: {total_fault_count}')
    
    # with open('./tracerank/result.pkl', 'wb') as fout:
    #     pickle.dump(total_pred_index_list, fout)

    if total_fault_count != len(total_pred_index_list):
        logger.error('WARNING')