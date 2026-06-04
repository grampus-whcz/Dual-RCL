import copy
import os
import pickle
import pickle as pkl
import shutil
from datetime import timedelta

import pandas as pd
import numpy as np

from typing import *
from data.data_models import *



# 将span类转化为字典方便后续dataframe使用
def span_to_dict(span):
    return {
        'trace_id': span.trace_id,
        'span_id': span.span_id,
        'parent_span_id': span.parent_span_id,
        'start_time': span.start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': span.duration,
        'service_name': span.service_name,
        'anomaly': span.anomaly,
        'status_code': span.status_code,
        'operation_name': span.operation_name,
        'root_cause': span.root_cause,
        'latency': span.latency,
        'structure': span.structure,
        'extra': span.extra
    }


def normal_pkl_to_pkl(path):
    with open(os.path.join(path, 'test_normal.pkl'), "rb") as f:
        traces = pkl.load(f)
    # 创建一个空列表
    data_list = []
    if os.path.exists('../modelcoder/Datasets/GAIA/'):
        shutil.rmtree('../modelcoder/Datasets/GAIA/')

    # 创建文件夹
    os.makedirs('../modelcoder/Datasets/GAIA/')

    # 定义文件储存位置
    pkl_file_name = f'normal.pkl'
    pkl_file_path = f'../modelcoder/Datasets/GAIA/{pkl_file_name}'
    # 遍历每个 Trace 对象
    for trace in traces:

        # 添加每一个Trace的根Span
        data_list.append(trace.root_span)
        # 添加Trace中其他的Span
        for span in trace.root_span.children_span_list:
            data_list.append(span)
    with open(pkl_file_path, 'wb') as file:
        pickle.dump(data_list, file)


def fault_injection_to_pkl(path):

    with open(os.path.join(path, 'abnormal.pkl'), "rb") as f:
        traces = pkl.load(f)

    if os.path.exists('../modelcoder/Datasets/GAIA/fault_injection_tracerank'):
        shutil.rmtree('../modelcoder/Datasets/GAIA/fault_injection_tracerank')

    # 创建文件夹
    os.makedirs('../modelcoder/Datasets/GAIA/fault_injection_tracerank')
    fault_injection_dict = {
        datetime.strptime(f'2021-07-{day}', '%Y-%m-%d').strftime('%Y-%m-%d'): []
        for day in range(1, 32)
    }

    csv = pd.read_csv('../../newest/run/run_table_2021-07.csv')

    for _, row in csv.iterrows():
        if 'INFO' in row['message'] or 'ERROR' in row['message']:
            continue

        date = datetime.strptime(row['datetime'], '%Y-%m-%d')
        service = row['service']
        try:
            time = datetime.strptime(row['message'].split(',')[0], '%Y-%m-%d %H:%M:%S')
        except:
            continue

        if row['datetime'] not in fault_injection_dict:
            continue

        fault_injection_dict[row['datetime']].append({
            'date': date,
            'service': service,
            'time': time,
            'trace_list': []
        })

    for date, temp_list in fault_injection_dict.items():

        fault_injection_list = copy.deepcopy(temp_list)

        # if date in ['2021-07-29', '2021-07-30', '2021-07-31']:
        #     continue

        seconds_to_row = {
            i: []
            for i in range(-2*60 * 60 * 24, 2*60 * 60 * 24)
        }
        for trace in traces:
            seconds = int(
                (trace.root_span.start_time - datetime.strptime(f'{date} 00:00:00','%Y-%m-%d %H:%M:%S')).total_seconds())
            if seconds in range(-2*60 * 60 * 24, 2*60 * 60 * 24):
                seconds_to_row[seconds].append(trace)
            else:
                continue
        for index in range(len(fault_injection_list)):
            fault_injection = fault_injection_list[index]

            seconds = (fault_injection['time'] - datetime.strptime(f'{date} 00:00:00','%Y-%m-%d %H:%M:%S')).total_seconds()

            for j in range(int(max(-2*60 * 60 * 24, seconds - 60 * 60 * 24)), int(min(2*60 * 60 * 24, seconds + 60 * 60 * 24))):
                fault_injection['trace_list'].extend(seconds_to_row[j])

            fault_injection_list[index] = fault_injection

        pickle.dump(
            fault_injection_list,
            open(f'../modelcoder/Datasets/GAIA/fault_injection_tracerank/fault_injection_list_{date}.pkl', 'wb')
        )




if __name__ == '__main__':
    fault_injection_to_pkl('../../newest')
