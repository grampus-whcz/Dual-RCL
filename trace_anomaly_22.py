import csv
import random
import shutil
import time
from queue import Queue

import numpy as np
import pickle
import os
import json
import sys
import re
import datetime
import functools
import pandas as pd
import yaml
import pickle

def get_trace_from_aiops2022(csv_data):
    count = 0
    traceID = None
    trace = dict()
    # 'timestamp', 'cmdb_id', 'span_id', 'trace_id', 'duration', 'type', 'status_code',
    # 'operation_name', 'parent_span', 'abnormal'
    # for _, row in df.iterrows():
    for row in csv_data:
        if traceID is None:
            traceID = row[3]
        if traceID != row[3]:
            count += 1
            if count % 1000000 == 0:
                print('scan count: {}'.format(count))
            yield traceID, trace
            traceID = row[3]
            trace = dict()
        trace[row[2]] = {'response_time': int(row[4]), 'operation': row[1], 'start_time': int(row[0])}
        parent = None
        if row[-2] != '':
            parent = row[-2]
        trace[row[2]]['parent'] = parent

def get_trace_from_gaia(csv_data):
    count = 0
    traceID = None
    trace = dict()
    # 'timestamp', 'host_ip', 'service_name', 'trace_id', 'span_id', 'parent_id', 'start_time', 'end_time', 
    # 'url', 'status_code', 'message', 'abnormal'
    for row in csv_data:
        if traceID is None:
            traceID = row[3]
        if traceID != row[3]:
            count += 1
            if count % 1000000 == 0:
                print('scan count: {}'.format(count))
            yield traceID, trace
            traceID = row[3]
            trace = dict()

        try:
            start= datetime.datetime.strptime(row[6], '%Y-%m-%d %H:%M:%S.%f').timestamp()
        except ValueError:
            try:
                start= datetime.datetime.strptime(row[6], '%Y-%m-%d %H:%M:%S').timestamp()
            except ValueError:
                print("Does not match either format")
        try:
            end= datetime.datetime.strptime(row[7], '%Y-%m-%d %H:%M:%S.%f').timestamp()
        except ValueError:
            try:
                end= datetime.datetime.strptime(row[7], '%Y-%m-%d %H:%M:%S').timestamp()
            except ValueError:
                print("Does not match either format")
        #start= datetime.datetime.strptime(row[6], '%Y-%m-%d %H:%M:%S.%f').timestamp()
        #end= datetime.datetime.strptime(row[7], '%Y-%m-%d %H:%M:%S.%f').timestamp()
        duration=(end-start)*1000
        trace[row[4]] = {'response_time': int(duration), 'operation': row[2], 'start_time': int(start)}
        parent = None
        if row[5] != '':
            parent = row[5]
        trace[row[4]]['parent'] = parent

def walk_call_path(trace, span=None, call_path=''):
    '''
    将单个trace转换为call path列表
    '''
    if span == None:
        call_path_list = []
        response_time = []
        for _, span_temp in trace.items():
            call_path_list.append(walk_call_path(trace, span=span_temp))
            response_time.append(span_temp['response_time'])
        return call_path_list, response_time
    # 如果有目标span 则处理该span
    call_path = '#'.join([span['operation'], call_path])
    parent = span['parent']
    # 如果没有父节点 则返回该条callpath 返回时去掉末尾 '#'
    if parent == None:
        return '#'.join(['start', call_path[:-1]])
    elif parent not in trace:
        return '#'.join(['unclear_start', call_path[:-1]])
    # 否则递归处理父节点span
    else:
        return walk_call_path(trace, span=trace[parent], call_path=call_path)
    
def walk_call_path_anomaly(trace, span=None, call_path=''):
    '''
    将单个trace转换为call path列表
    '''
    # 如果是刚开始 则递归遍历trace中span
    if span == None:
        call_path_list = []
        response_time = []
        for _, span_temp in trace.items():
            call_path_list.append(walk_call_path(trace, span=span_temp))
            response_time.append(span_temp['response_time'])
        return call_path_list, response_time
    # 如果有目标span 则处理该span
    call_path = '#'.join([span['operation'], call_path])
    parent = span['parent']
    # 如果没有父节点 则返回该条callpath 返回时去掉末尾 '#'
    if parent == None:
        return '#'.join(['start', call_path[:-1]])
    elif parent not in trace:
        return '#'.join(['unclear_start', call_path[:-1]])
    # 否则递归处理父节点span
    else:
        return walk_call_path(trace, span=trace[parent], call_path=call_path)
    
def trace_to_STV(trace, call_path_dict, STV_length):
    '''
    # 获得trace的STV
    '''
    STV = [0] * STV_length
    call_path, response_time = walk_call_path(trace)
    for cp, rt in list(zip(call_path, response_time)):
        if cp not in call_path_dict:
            print('Invalid call path: {}  time:{}'.format(cp, rt))
            return 'Invalid call path: {}  time:{}'.format(cp, rt)
        else:
            if STV[call_path_dict[cp]] < rt:
                STV[call_path_dict[cp]] = rt
    return STV

def adjust_size(tempstv):
    stv = []
    trace_id = []
    for tem in tempstv:
        trace_id.append(tem.strip().split(':')[0])
        tmp = tem.strip().split(':')[1].split(',')
        stv.append(tmp)
    max_len = 0
    for sub in stv:
        if len(sub) > max_len:
            max_len = len(sub)
    for item in stv:
        if len(item) < max_len:
            for i in range(len(item), max_len):
                item.append(0)
    for sub in stv:
        if len(sub) != max_len:
            print(1)
    for i in range(len(stv)):
        tmp = ",".join(str(item) for item in stv[i])
        stv[i] = tmp
    final=[]
    for i in range(len(trace_id)):
        final.append('{}:{}'.format(trace_id[i], stv[i]))
    return  final

def anomaly_detect_and_generate_describe(idx_path,normal_path,file_path):
    with open(idx_path, 'rb') as f:
        call_path_dict = pickle.load(f)
    call_path = list(call_path_dict.keys())
    STV_length = len(call_path_dict)
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        csv_data = [row for row in reader]
    dic={}
    for d in csv_data:
        if len(d[8])<1:
            starttt=datetime.datetime.fromtimestamp(int(d[0])/1000).strftime('%Y-%m-%d %H:%M:%S')
            dic[d[3]]=[starttt]
    #traces = get_trace_from_gaia(csv_data) 
    traces = get_trace_from_aiops2022(csv_data) 
    tempstv=[]
    while True:
        try:
            trace_ID, trace = next(traces)
            tempstv.append('{}:{}'.format(trace_ID, ','.join(
                    [str(t) for t in trace_to_STV(trace, call_path_dict, STV_length)])))
        except StopIteration:
            break
    abnormalstv=adjust_size(tempstv)

    with open(normal_path, 'r') as fin:
        normalstv = fin.read().strip().split('\n')
    
    flows_ab = list()
    vectors_ab = list()
    for line in abnormalstv:
        if line.strip() == "":
            continue
        if 'I,n,v,a,l,i,d' in line:
            continue
        flows_ab.append(line.split(':')[0])
        vectors_ab.append([float(x) for x in line.split(':')[1].split(',')])
    
    flows_tr = list()
    vectors_tr = list()
    for line in normalstv:
        if line.strip() == "":
            continue
        flows_tr.append(line.split(':')[0])
        vectors_tr.append([float(x) for x in line.split(':')[1].split(',')])
    anomalys=[]
    # 前查看是否存在异常维或调用链不存在
    for i in range(len(vectors_ab)):
        if len(anomalys)>30:
            break
        if flows_ab[i] not in dic:
            print(flows_ab[i])
            continue
        index_ab=[]
        value_ab=[]
        for index, value in enumerate(vectors_ab[i]):
            if value!=0:
                index_ab.append(index)
                value_ab.append(value)
        num_ab=len(index_ab)
        anomaly_list = [[0] for _ in range(num_ab)]
        for j in range(len(vectors_tr)):
            index_tr=[]
            value_tr=[]
            for index, value in enumerate(vectors_tr[j]):
                if value!=0:
                    index_tr.append(index)
                    value_tr.append(value)
            num_tr=len(index_tr)
            if num_tr!=num_ab:
                continue
            f=1
            #print(index_tr)
            for k in range(num_ab):
                if index_tr[k]!=index_ab[k]:
                    f=0
                    break
            if f==0:
                continue
            for k in range(num_ab):
                anomaly_list[k].append(value_tr[k])
            if len(anomaly_list[0])>6:
                break
        anomaly_arr = np.array(anomaly_list)
        if len(anomaly_arr[0])==1:
            # 调用链异常
            anomaly='The summary of anomaly is that there is a structural anomaly in the trace,or a new trace appears,trace_id is '+flows_ab[i]+' ,the start time of the trace call is '+dic[flows_ab[i]][0]+'.'
            #print(anomaly)
            anomalys.append(anomaly)
            continue
        selected_columns = anomaly_arr[:, 1:]
        # 计算每行的平均值和标准差
        means = np.mean(selected_columns, axis=1)
        std_devs = np.std(selected_columns, axis=1, ddof=1)
    
        anomaly_service="The services with anomalies include:"
        f=0
        score=0
        max_call=''
        for k in range(num_ab):
            splitt = call_path[index_ab[k]].split("#")
            cp = "->".join(splitt[1:])
            if len(call_path[index_ab[k]])>len(max_call):
                max_call=cp
            if value_ab[k]>(means[k]+3*std_devs[k]) or value_ab[k]<(means[k]-3*std_devs[k]):
                if std_devs[k]==0:
                    continue
                score_tem=((value_ab[k]-means[k])/std_devs[k])/3
                if score_tem>score:
                    score=score_tem
                segments = call_path[index_ab[k]].split('#')
                tem=""
                if value_ab[k]>(means[k]+3*std_devs[k]):
                    tem+=" the call time for %d s exceeding the normal upper limit by %d ms"%(value_ab[k],(value_ab[k]-(means[k]+3*std_devs[k])))
                else:
                    tem+=" the call time for %d s below the normal lower limit by %d ms"%(value_ab[k],((means[k]-3*std_devs[k])-value_ab[k]))
                anomaly_service +=" "+segments[-1] + tem +" ,"
                f=1
                    
        if f==0:
            #print("无异常")
            continue
        anomaly="The summary of anomaly is that the trace call timeout,trace_id is %s ,the start time of the trace call is %s ,"% (flows_ab[i],dic[flows_ab[i]][0])
        anomaly += f'the call path is {max_call},'+anomaly_service
        anomaly=anomaly[:-1]+'.'
        if score>2:
            #print(score)
            #print(anomaly)
            anomalys.append(anomaly)
            
    return anomalys
