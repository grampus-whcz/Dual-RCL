import logging
import pathlib
import argparse

from camel.typing import ModelType
from chatops.chat_chain import ChatChain
import os

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
import json

import time
import pandas as pd
import re
import time
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI

from util_funcs.loaddata import load
from micro import run_SPOT,get_eta,run_pcmci,get_Q_matrix,get_Q_matrix_part_corr,randomwalk_metric,get_gamma,evaluate,root_kpi,get_links

import numpy as np

from metric_anomaly_22 import *
from trace_anomaly_22 import *

from mepfl_22 import *
import csv


os.environ['OPENAI_API_KEY'] = 'nn'
os.environ['BASE_URL'] = "http://localhost:8000/v1"



# Converted to Timestamps
def swap(qurry):
    pattern1 = r"\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{1,2}"
    pattern2 = r"\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{1,2}"

    matches1 = re.findall(pattern1, qurry)
    matches2 = re.findall(pattern2, qurry)
    if matches1:
        pat="%Y/%m/%d %H:%M"
        for match in matches1:
            Array = time.strptime(match, pat)
            stamp = int(time.mktime(Array))
            qurry=qurry.replace(match, str(stamp))
    elif matches2:
        pat="%Y-%m-%d %H:%M"
        for match in matches2:
            Array = time.strptime(match, pat)
            stamp = int(time.mktime(Array))
            qurry=qurry.replace(match, str(stamp))
    return qurry


def get_config(config: str) -> tuple[pathlib.Path, ...]:
    """
    Get config path
    Args:
        config: Name of config, which is used to load configuration under CompanyConfig/

    Returns:
        config_path: Path of ChatChainConfig.json
        config_phase_path: Path of PhaseConfig.json
        config_role_path: Path of RoleConfig.json
    """
    root = pathlib.Path(__file__).parent
    config_dir = root / 'CompanyConfig' / config
    default_config_dir = root / 'CompanyConfig' / 'Default'
    config_files = [
        'ChatChainConfig.json',
        'PhaseConfig.json',
        'RoleConfig.json',
    ]
    config_paths = []
    for config_file in config_files:
        company_config_path = config_dir / config_file
        default_config_path = default_config_dir / config_file
        if company_config_path.exists():
            config_paths.append(company_config_path)
        else:
            config_paths.append(default_config_path)
    return tuple(config_paths)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='ChatOps', description='ChatOps: A Chatbot AIOps Framework')
    parser.add_argument(
        '--config', type=str, default='SelfIntroduction',
        help='Name of config, which is used to load configuration under CompanyConfig/',
    )
    parser.add_argument(
        '--namespace', type=str, default='DefaultNameSpace',
        help='Namespace of the AIOps case, your report will be generated in Report/name_namespace_timestamp',
    )
    parser.add_argument(
        '--task', type=str, default='At 2022/05/03 06:56 have exceptions in the microservices system. What are these exceptions? Please output an exception analysis.',
        help='Task prompt, which is used to generate the first message of the chatbot',
    )
    parser.add_argument(
        '--name', type=str, default='DefaultName',
        help='Name of the AIOps case, your report will be generated in Report/name_namespace_timestamp',
    )
    parser.add_argument(
        '--model', type=str, default='LLAMA_3_8B',
        help='Large language model, choose from {"GPT_3_5_TURBO", "GPT_4", "GPT_4_32K", "GPT_4_TURBO", "ERNIE_BOT_4", "LLAMA_3_8B"}',
    )
    parser.add_argument(
        '--path', type=str, default='',
        help='Your file directory, ChatOps will generate reports based upon existing documents in the Incremental mode',
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    # Start
    config_path, config_phase_path, config_role_path = get_config(args.config)


    datetime_pattern = r'(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2})'
    match = re.search(datetime_pattern, args.task)

    if match:
        year = match.group(1)  
        month = match.group(2)  
        day = match.group(3)  
        hour = match.group(4)  
        minute = match.group(5)  
        
        date_result = month + day
        time_result = f"{hour}-{minute}"
        
        print(f"Data: {date_result}")
        print(f"Time: {time_result}")
    else:
        print("Not Found.")
    

    # ----------------------------------------
    #          Trace
    # ----------------------------------------
    # Get all files and subdirectories in a directory
    trace_dirs = os.listdir(f'{date_result}_trace_ano/data')

    files_trace = [file for file in trace_dirs if time_result in file]
    print(files_trace[0])
    trace_ans = anomaly_detect_and_generate_describe(f'{date_result}_trace_ano/{date_result}.pkl',f'{date_result}_trace_ano/normal_datasets',f'{date_result}_trace_ano/data/{files_trace[0]}  ')
    trace_an=''
    for index, value in enumerate(trace_ans):
        trace_an+=f'({index+1})'+value
        if len(trace_an)>2500:#14000
            break
    

    # ----------------------------------------
    #          mepfl
    # ----------------------------------------
    # Get all files and subdirectories in a directory
    tracerca_dirs = os.listdir(f'{date_result}_tracerca')

    files_tracerace = [file for file in tracerca_dirs if time_result in file]
    print(files_tracerace[0])
    root_service=mepfl_22(f'./{date_result}_tracerca/{files_tracerace[0]}')
    root_se=''
    for i in range(5):
        root_se+="("+str(i+1)+")"+root_service[i]
        if i==4:
            root_se+='.'
        else:
            root_se+=','
    # write knowledge
    with open(config_phase_path) as f:
        dataconfig = json.load(f)

    
    dataconfig['TraceAnalysis']['phase_prompt'][0]="Top root cause and their anomaly description:\n Anomaly description:"+trace_an+'\nTop5 root cause:'+root_se
    
    with open(config_phase_path, 'w') as file:
        json.dump(dataconfig, file)


    # ----------------------------------------
    #          metric
    # ----------------------------------------
    clf = CNNClassifier(class_num=11)
    clf.load_model("patterncla.pt")
    # Get all files and subdirectories in a directory
    metric_dirs = os.listdir(f'{date_result}_metric_fault')

    files_metric = [file for file in metric_dirs if time_result in file]
    print(files_metric[0])

    metric_ans=generate_metric_describe(clf,f'{date_result}_metric_fault/{files_metric[0]}',root_service[:5])
    metric_an=''
    for index, value in enumerate(metric_ans):
        metric_an+=f'({index+1})'+value
        if len(metric_an)>3000:
            break
    

    # ----------------------------------------
    #          cause analysis
    # ----------------------------------------

    # Get all files and subdirectories in a directory
    microcause_dirs = os.listdir(f'{date_result}_microcause')

    files_microcause = [file for file in microcause_dirs if time_result in file]
    print(files_microcause[0])


    frontend=[1] 
    dataa, data_head = load(
        f'{date_result}_microcause/{files_microcause[0]}',
        normalize=False,
        zero_fill_method='prevlatter',
        aggre_delta=1,
        verbose=True,
    )

    SPOT_res = run_SPOT(dataa,data_head, q=1e-3, d=10)
    eta, ab_timepoint = get_eta(dataa,data_head,SPOT_res, int(0.5 * len(dataa)))
    pcmci, pcmci_res = run_pcmci(dataa,pc_alpha = 0.05, verbosity=1)
    g = get_links(data_head,pcmci, pcmci_res, alpha_level = 0.05)
    Q = get_Q_matrix_part_corr(dataa,data_head,frontend,g, rho=0.2)
    vis_list = randomwalk_metric(Q, 1000, frontend[0], teleportation_prob=0, walk_step=15)
    gamma = get_gamma(data_head,vis_list, eta, lambda_param=0.5)
    root_metric=root_kpi(data_head,gamma)

    # write knowledge
    with open(config_phase_path) as f:
        dataconfig = json.load(f)

    dataconfig['MetricAnalysis']['phase_prompt'][0]="Top root cause metrics and their anomaly description: \nAnomaly description:"+metric_an+'\n'+root_metric

    with open(config_phase_path, 'w') as file:
        json.dump(dataconfig, file)


    # ----------------------------------------
    #          log
    # ----------------------------------------
    # Get all files and subdirectories in a directory
    log_dirs = os.listdir(f'{date_result}_log_fault')

    files_log = [file for file in log_dirs if time_result in file]
    print(files_log[0])
    log_list = os.listdir(f'{date_result}_log_fault/{files_log[0]}')
    log_an=''
    print(root_service[:5])
    for fi in log_list:
        f=0
        for root in root_service[:5]:
            if root in fi:
                f=1
        if f==0:
            continue
        print(fi)
        with open(os.path.join(f'{date_result}_log_fault/{files_log[0]}',fi), mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            tem_log=''
            for row in reader:
                if 'trace_id' in row[4]:
                    continue
                if 'traci_id' in row[4]:
                    continue
                if(len(tem_log))>1000:#3000
                    break
                #if 'ERROR' in row[2]:
                tem_log+=row[4]
            log_an+=tem_log
        if(len(log_an))>3000:#10000
            break

    # write knowledge
    with open(config_phase_path) as f:
        dataconfig = json.load(f)

    dataconfig['LogAnalysis']['phase_prompt'][0]="Knowledge: "+log_an

    with open(config_phase_path, 'w') as file:
        json.dump(dataconfig, file)
    
    
    # ----------------------------------------
    #          rootcause
    # ----------------------------------------
    with open(config_phase_path) as f:
        dataconfig = json.load(f)
    dataconfig['RootCauseAnalysis']['phase_prompt'][0]="Knowledge: Please generate root service results based on TraceAgent's Trace root cause analysis."
    with open(config_phase_path, 'w') as file:
        json.dump(dataconfig, file)

    
    # ----------------------------------------
    #          Init ChatChain
    # ----------------------------------------
    args2type = {
        'GPT_3_5_TURBO': ModelType.GPT_3_5_TURBO,
        'GPT_4': ModelType.GPT_4,
        'GPT_4_32K': ModelType.GPT_4_32K,
        'GPT_4_TURBO': ModelType.GPT_4_TURBO,
        'GPT_4_TURBO_V': ModelType.GPT_4_TURBO_V,
        'ERNIE_BOT_4': ModelType.ERNIE_BOT_4,
        'MISTRAL_7B': ModelType.MISTRAL_7B,
        'LLAMA_3_8B':ModelType.LLAMA_3_8B,
    }

    chat_chain = ChatChain(
        config_path=config_path,
        config_phase_path=config_phase_path,
        config_role_path=config_role_path,
        task_prompt=args.task,
        case_name=args.name,
        namespace=args.namespace,
        model_type=args2type[args.model],
        docs_path=args.path,
    )

    # ----------------------------------------
    #          Init Log
    # ----------------------------------------
    logging.basicConfig(
        filename=chat_chain.log_path,
        level=logging.INFO,
        format='[%(asctime)s %(levelname)s] %(message)s',
        datefmt='%Y-%d-%m %H:%M:%S',
        encoding='utf-8',
    )

    # ----------------------------------------
    #          Pre Processing
    # ----------------------------------------

    chat_chain.pre_processing()

    # ----------------------------------------
    #          Personnel Recruitment
    # ----------------------------------------

    chat_chain.make_recruitment()

    # ----------------------------------------
    #          Chat Chain
    # ----------------------------------------

    chat_chain.execute_chain()

    # ----------------------------------------
    #          Post Processing
    # ----------------------------------------

    chat_chain.post_processing()


if __name__ == '__main__':
    main(get_args())
