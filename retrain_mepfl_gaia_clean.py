#!/usr/bin/env python3
"""Retrain MEPFL GAIA model using 0702+0703 tracerca data (EXCLUDING 0701).

0702_tracerca and 0703_tracerca contain Trace objects with service/time labels,
same format as the fault_injection_tracerank pkls. Training on these gives a
model that has never seen 0701 data → no leakage when evaluating on 0701.
"""
import os, sys, pickle
sys.path.insert(0, 'mepfl_model')
sys.path.insert(0, '.')

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from data.read_data import get_trace_list_from_rows, Trace

SERVICE_LIST = ['webservice1','webservice2','redisservice2','redisservice1',
                'mobservice1','logservice1','mobservice2','logservice2','dbservice2','dbservice1']
service_index_dict = {s: i for i, s in enumerate(SERVICE_LIST)}

OUT_DIR = 'mepfl_model_gaia_clean'
os.makedirs(OUT_DIR, exist_ok=True)


def get_trace_service_list(trace):
    sset = set()
    queue = [trace.root_span]
    while queue:
        span = queue.pop(0)
        queue.extend(span.children_span_list)
        sset.add(span.service_name)
    return sset


def get_trace_list_vector(trace_list):
    for trace in trace_list:
        trace.vector = np.zeros((2 + len(SERVICE_LIST) * 3))
        trace.vector[0] = len(SERVICE_LIST)
        trace.vector[1] = len(get_trace_service_list(trace))
        queue = [trace.root_span]
        while queue:
            span = queue.pop(0)
            temp = sorted(span.children_span_list, key=lambda x: x.start_time, reverse=False)
            queue.extend(temp)
            duration = span.duration
            try:
                status = int(float(span.status_code))
            except (ValueError, TypeError):
                status = 0
            process_time = span.duration - max([c.duration for c in span.children_span_list], default=0.0)
            if span.service_name in service_index_dict:
                idx = service_index_dict[span.service_name]
                trace.vector[-1 + (idx + 1) * 3] = duration
                trace.vector[0 + (idx + 1) * 3] = status
                trace.vector[1 + (idx + 1) * 3] = process_time


def main():
    classification_traces = []
    localization_traces = []
    valid_count = 0
    stop_flag = 2429  # same as original

    for date_dir in ['0702_tracerca']:  # ONLY 0702 for training; 0703 reserved for router
        files = sorted([f for f in os.listdir(date_dir) if f.endswith('.pkl')])
        logger.info(f'{date_dir}: {len(files)} events')
        for fname in files:
            if valid_count >= stop_flag:
                break
            with open(os.path.join(date_dir, fname), 'rb') as f:
                fi_list = pickle.load(f)
            for fi in fi_list:
                if valid_count >= stop_flag:
                    break
                rows = fi['trace_list']
                trace_list = get_trace_list_from_rows(rows)
                if len(trace_list) <= 10:
                    continue
                valid_count += 1
                for trace in trace_list:
                    if (trace.root_span.start_time - fi['time']).total_seconds() <= 60:
                        if fi['service'] in get_trace_service_list(trace):
                            classification_traces.append(trace)
                            localization_traces.append(trace)
                            trace.anomaly_type = 1
                            trace.fault_service_index = service_index_dict[fi['service']]
                    else:
                        classification_traces.append(trace)
                        trace.anomaly_type = 0

    logger.info(f'Traces: cls={len(classification_traces)}, loc={len(localization_traces)}')

    get_trace_list_vector(classification_traces)
    get_trace_list_vector(localization_traces)

    # Train RF
    cls_x = [t.vector for t in classification_traces]
    cls_y = [t.anomaly_type for t in classification_traces]
    logger.info(f'RF training: {len(cls_x)} samples, positive={sum(cls_y)}')
    rf = RandomForestClassifier(random_state=0)
    rf.fit(cls_x, cls_y)
    logger.info(f'RF acc: {accuracy_score(cls_y[:10000], rf.predict(cls_x[:10000])):.4f}')

    # Train MLP
    loc_x = [t.vector for t in localization_traces]
    loc_y = [t.fault_service_index for t in localization_traces]
    logger.info(f'MLP training: {len(loc_x)} samples, classes={len(set(loc_y))}')
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    mlp.fit(loc_x, loc_y)
    logger.info(f'MLP acc: {accuracy_score(loc_y[:10000], mlp.predict(loc_x[:10000])):.4f}')

    # Save
    pickle.dump(rf, open(f'{OUT_DIR}/rf_model.pkl', 'wb'))
    pickle.dump(mlp, open(f'{OUT_DIR}/mlp_model.pkl', 'wb'))
    logger.info(f'Saved clean MEPFL to {OUT_DIR}/ (trained on 0702+0703, 0701 EXCLUDED)')


if __name__ == '__main__':
    main()
