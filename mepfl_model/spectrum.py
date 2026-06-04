from typing import *
from data.data_models import *


def get_visited_service_set(trace: Trace) -> set:
    root_span = trace.root_span
    queue = [root_span]
    service_set = set()
    while len(queue):
        span = queue.pop(0)
        queue.extend(span.children_span_list)

        service_set.add(span.service_name)
    
    return service_set


def spectrum(trace_list: List[Trace], service_index_dict: dict) -> dict:

    service_statistic_dict ={
        service: 0.0
        for service in service_index_dict
    }

    for service in service_statistic_dict:
        ef = 0
        ep = 0
        nf = 0

        for trace in trace_list:
            service_set = get_visited_service_set(trace)

            if trace.anomaly_type == 1:
                if service in service_set:
                    ef += 1
                else:
                    nf += 1
            else:
                if service in service_set:
                    ep += 1
        
        ef = ef if ef > 0 else 0.0001
        ep = ep if ep > 0 else 0.0001
        nf = nf if nf > 0 else 0.0001

        statistic = ef / (((ef + ep) * (ef + nf)) ** 0.5)

        service_statistic_dict[service] = statistic
    
    return service_statistic_dict