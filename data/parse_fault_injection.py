import pandas as pd
import random
import pickle
import copy
import os

from typing import *
from datetime import datetime
from dataclasses import dataclass
from tqdm import tqdm
from loguru import logger

fault_injection_dict = {
    datetime.strptime(f'2021-07-{day}', '%Y-%m-%d').strftime('%Y-%m-%d'): []
    for day in range(1, 32)
}

csv = pd.read_csv('./Datasets/GAIA/run/run_table_2021-07.csv')

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
        for i in range(0, 60 * 60 * 24)
    }
    csv = pd.read_csv(
        '/'.join(['./Datasets/GAIA/trace_by_day', date + '.csv'])
    )
    for _, row in csv.iterrows():
        seconds = int((datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S') - datetime.strptime(f'{date} 00:00:00',
                                                                                                    '%Y-%m-%d %H:%M:%S')).total_seconds())
        seconds_to_row[seconds].append(row)

    if date == '2021-07-01' or True:
        normal_seconds_to_row = copy.deepcopy(seconds_to_row)

        for index in tqdm(range(0, len(fault_injection_list)), desc='get normal trace'):
            fault_injection = fault_injection_list[index]

            seconds = (fault_injection['time'] - datetime.strptime(f'{date} 00:00:00',
                                                                   '%Y-%m-%d %H:%M:%S')).total_seconds()

            for j in range(int(max(0, seconds - 300)), int(min(86399, seconds + 300))):
                normal_seconds_to_row[j] = []

        normal_trace_list = []
        for trace_list in normal_seconds_to_row.values():
            normal_trace_list.extend(trace_list)

        logger.info(f'{date} Normal trace number', len(normal_trace_list))
        normal_trace_list = pd.DataFrame(normal_trace_list)[1:]
        if not os.path.exists('/'.join(['./Datasets/GAIA/trace_by_day', 'normal.csv'])):
            normal_trace_list.to_csv(
                '/'.join(['./Datasets/GAIA/trace_by_day', 'normal.csv']),
                sep=',',
                header=['timestamp', 'host_ip', 'service_name', 'trace_id', 'span_id', 'parent_id', 'start_time',
                        'end_time', 'url', 'status_code', 'message'],
                mode='a',
                index=False
            )
        else:
            normal_trace_list.to_csv(
                '/'.join(['./Datasets/GAIA/trace_by_day', 'normal.csv']),
                sep=',',
                header=False,
                mode='a',
                index=False
            )

    for index in tqdm(range(0, len(fault_injection_list)), desc='get abnormal trace'):
        fault_injection = fault_injection_list[index]

        seconds = (fault_injection['time'] - datetime.strptime(f'{date} 00:00:00', '%Y-%m-%d %H:%M:%S')).total_seconds()

        for j in range(int(max(0, seconds - 180)), int(min(86399, seconds + 180))):
            fault_injection['trace_list'].extend(seconds_to_row[j])

        fault_injection_list[index] = fault_injection
        logger.info(f"Trace list length {len(fault_injection['trace_list'])}")

    logger.info(f'{date} finish')

    pickle.dump(
        fault_injection_list,
        open(f'./Datasets/GAIA/fault_injection_tracerank/fault_injection_list_{date}.pkl', 'wb')
    )
    logger.info(f'dump finish, {date} length: {len(fault_injection_list)}')

    logger.info(f'{date} fault_injection_dict dump')
