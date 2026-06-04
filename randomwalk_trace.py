from typing import *
from data.data_models import *
from scipy.stats import pearsonr

import numpy as np

rho = 0.5

def randomwalk(trace_list: List[Trace], service_index_dict: dict, service_list: List[str]) -> dict:
    service_process_time_vector_dict = {
        service: np.zeros((len(trace_list), ))
        for service in service_index_dict
    }
    
    adjancy_matrix = [[0] * len(service_index_dict)] * len(service_index_dict)

    for index, trace in enumerate(trace_list):
        root_span = trace.root_span
        queue = [root_span]
        processing_time = 999999

        while len(queue):
            span = queue.pop(0)
            children = span.children_span_list
            queue.extend(children)
            for child in children:
                processing_time = min(processing_time, span.duration - child.duration)
                adjancy_matrix[service_index_dict[span.service_name]][service_index_dict[child.service_name]] = 1

            service_process_time_vector_dict[span.service_name][index] = processing_time

    front = 'frontend-0'
    correlation_list = []
    for service in service_list:
        if service == front:
            correlation_list.append(-1)
        else:
            if len(trace_list) > 1:
                correlation, _ = pearsonr(
                    service_process_time_vector_dict[front],
                    service_process_time_vector_dict[service]
                )
            else:
                correlation = 1.0
            correlation_list.append(correlation)

    
    A = [[0] * len(service_index_dict)] * len(service_index_dict)

    for i in range(0, len(service_list)):
        children_correlation_score_list = []
        for j in range(0, len(service_list)):
            if adjancy_matrix[i][j] > 0:
                children_correlation_score_list.append(correlation_list[j])
                A[i][j] = correlation_list[j]
                if i != 0:
                    A[j][i] = rho * correlation_list[i]
        if i != 0:
            A[i][i] = max(0, correlation_list[i] - max(children_correlation_score_list, default=0))
    
    for row in range(0, len(A)):
        row_sum = np.sum(A[row])
        for col in range(0, len(A[row])):
            A[row][col] = A[row][col] / row_sum
        
    v = np.zeros((len(service_list), ))
    for i in range(1, len(service_list)):
        v[i] = correlation_list[i]
    
    x = np.zeros((len(service_list), ))

    for i in range(0, 50):
        x = 0.5 * (np.dot(A, x)) + 0.5 * v
    
    return_dict = {}
    for i in range(0, len(service_list)):
        return_dict[service_list[i]] = x[i]
    
    return return_dict


