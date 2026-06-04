import os
import pickle
import pandas as pd
import numpy as np

from typing import *
from data.data_models import *


def get_normal_span_list_from_csv(dataset_path: str) -> List[pd.Series]:
    df = pickle.load(
        open('/'.join([dataset_path, 'normal.pkl']), 'rb')
    )
    span_list = []

    for row in df:
        span_list.append(row)
    rows=span_list

    trace_list: List[Trace] = list()

    trace_id_span_list_dict = {}
    for row in rows:
        trace_id = str(row.trace_id)
        temp_list = trace_id_span_list_dict.get(trace_id, list())
        start_time = row.start_time

        temp_list.append(Span(
            trace_id=trace_id,
            span_id=row.span_id,
            parent_span_id=row.parent_span_id,
            children_span_list=row.children_span_list,
            service_name=row.service_name,
            status_code=row.status_code,

            start_time=start_time,
            duration=row.duration,

            anomaly=row.anomaly,

            operation_name=row.operation_name,
            root_cause=row.root_cause,
            latency=row.latency,
            structure=row.structure,
            extra=row.extra
        ))
        trace_id_span_list_dict[trace_id] = temp_list

    for trace_id, span_list in trace_id_span_list_dict.items():
        span_id_dict = {}
        root_span: Span = None
        for span in span_list:
            span_id_dict[span.span_id] = span
        for span in span_list:
            if span.parent_span_id is None:
                root_span = span

        trace = Trace(
            trace_id=trace_id,
            root_span=root_span,
            span_count=len(span_list)
        )
        trace_list.append(trace)

    return trace_list


def get_trace_list_from_rows(rows) -> List[Trace]:

    return rows


def get_span_processing_time(span: Span) -> float:
    return max(
        0,
        span.duration - max([child.duration for child in span.children_span_list], default=0.0)
    )


def get_service_duration_mean_std(trace_list: List[Trace]) -> Dict:
    service_duration_dict = {}

    for trace in trace_list:
        root_span = trace.root_span
        queue = [root_span]

        while len(queue):
            span = queue.pop(0)
            service = span.service_name
            processing_time = get_span_processing_time(span)

            temp_list: List[float]
            temp_list = service_duration_dict.get(service, list())
            temp_list.append(processing_time)
            service_duration_dict[service] = temp_list

            queue.extend(span.children_span_list)

    service_mean_std_dict = {
        service: [
            np.mean(service_duration_dict[service]),
            np.std(service_duration_dict[service])
        ]
        for service in list(service_duration_dict.keys())
    }

    return service_mean_std_dict


def get_service_duration_mean_from_anomalous_trace(trace_list: List[Trace]) -> dict:
    service_duration_dict = {}

    for trace in trace_list:
        if trace.anomaly_type != 1:
            continue

        root_span = trace.root_span
        queue = [root_span]

        while len(queue):
            span = queue.pop(0)
            service = span.service_name
            duration = span.duration

            temp_list = service_duration_dict.get(service, list())
            temp_list.append(duration)
            service_duration_dict[service] = temp_list

            queue.extend(span.children_span_list)

    service_mean_dict = {
        service: [
            np.mean(service_duration_dict[service])
        ]
        for service in list(service_duration_dict.keys())
    }

    return service_mean_dict


def check_train_test_split():
    fault_injection_length_name_list = []

    file_list = os.listdir('/'.join(['./Datasets/GAIA', 'fault_injection_tracerank']))
    for file in file_list:

        fault_injection_list = pickle.load(
            open('/'.join(['./Datasets/GAIA', 'fault_injection_tracerank', f'{file}']), 'rb')
        )

        usable_fault_injection_count = 0
        for index, fault_injection in enumerate(fault_injection_list):
            rows = fault_injection['trace_list']
            trace_list = get_trace_list_from_rows(rows)

            if len(trace_list) <= 10:
                continue

            else:
                usable_fault_injection_count += 1
        print((file, usable_fault_injection_count))
        fault_injection_length_name_list.append((file, usable_fault_injection_count))

    total_usable_count = sum([_[1] for _ in fault_injection_length_name_list])
    print(total_usable_count * 0.2)

    temp = 0
    for file, count in fault_injection_length_name_list:
        if temp + count >= total_usable_count * 0.2:
            print(file)
            break
        else:
            temp += count


if __name__ == '__main__':
    check_train_test_split()
    pass
