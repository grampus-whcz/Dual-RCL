from collections import defaultdict
import datetime
import threading
import os
import time
import pickle
from concurrent.futures import ThreadPoolExecutor
import pprint

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from tqdm import tqdm
from SPOT.spot import dSPOT
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
#from tigramite.independence_tests import ParCorr
from tigramite.independence_tests.parcorr import ParCorr
import pandas as pd
from pingouin import partial_corr

from util_funcs.loaddata import load
# from utils.draw_graph import draw_weighted_graph
from util_funcs.evaluation_function import prCal, my_acc, pr_stat, print_prk_acc
from util_funcs.format_ouput import format_to_excel
from util_funcs.excel_utils import saveToExcel, readExl

def run_SPOT(data,data_head, q=1e-3, d=300, n_init=None):
    result_dict = {}
    if n_init is None:
        n_init = int(0.5 * len(data))
    # depth must be strictly less than n_init so SPOT has enough calibration data
    d = min(d, n_init - 1)
    for svc_id in range(len(data_head)):
        init_data = data[:n_init, svc_id] 	# initial batch
        _data = data[n_init:, svc_id]  		# stream
        s = dSPOT(q,d)     	# DSPOT object
        s.fit(init_data,_data) 	# data import
        s.initialize() 	  		# initialization step
        results = s.run()    	# run
        result_dict[svc_id] = results
    return result_dict

def get_eta(data,data_head,SPOT_res, n_init):
    eta = np.zeros([len(data_head)])
    ab_timepoint = [0 for i in range(len(data_head))]
    for svc_id in range(len(data_head)):
        mask = data[n_init:, svc_id] > np.array(SPOT_res[svc_id]['thresholds'])
        ratio = np.abs(data[n_init:, svc_id] - np.array(SPOT_res[svc_id]['thresholds'])) / np.array(SPOT_res[svc_id]['thresholds'])
        if mask.nonzero()[0].shape[0] > 0:
            eta[svc_id] = np.max(ratio[mask.nonzero()[0]])
            ab_timepoint[svc_id] = np.min(mask.nonzero()[0])
        else:
            eta[svc_id] = 0
    return eta, ab_timepoint

def run_pcmci(data,pc_alpha = 0.1, verbosity=0):
    dataframe = pp.DataFrame(data)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=verbosity)
    pcmci_res = pcmci.run_pcmci(tau_max=10, pc_alpha=pc_alpha)
    return pcmci, pcmci_res

def get_Q_matrix(g, rho=0.2):
    corr = np.corrcoef(np.array(data).T)
    for i in range(corr.shape[0]):
        corr[i, i] = 0.0
    corr = np.abs(corr)
    
    Q = np.zeros([len(data_head), len(data_head)])
    for e in g.edges():
        Q[e[0], e[1]] = corr[frontend[0]-1, e[1]]
        backward_e = (e[1], e[0])
        if backward_e not in g.edges():
            Q[e[1], e[0]] = rho * corr[frontend[0]-1, e[0]]
            
    adj = nx.adjacency_matrix(g).todense()
    for i in range(len(data_head)):
        P_pc_max = None
        res_l = np.array([corr[frontend[0]-1, k] for k in adj[:, i]])
        if corr[frontend[0]-1, i] > np.max(res_l):
            Q[i, i] = corr[frontend[0]-1, i] - np.max(res_l)
        else:
            Q[i, i] = 0
    l = []
    for i in np.sum(Q, axis=1):
        if i > 0:
            l.append(1.0/i)
        else:
            l.append(0.0)
    l = np.diag(l)
    Q = np.dot(l, Q)
    return Q

def get_Q_matrix_part_corr(data,data_head,frontend,g, rho=0.2):
    df = pd.DataFrame(data, columns=data_head)
    def get_part_corr(x, y):
        cond = get_confounders(y)
        if x in cond:
            cond.discard(x)
        if y in cond:
            cond.discard(y)
        # Convert to list for pandas indexing (sets are not supported)
        cond = list(cond)
        # Limit the number of covariates to avoid rank-deficient covariance
        # matrices.  When the confounder set is large relative to the sample
        # size, partial correlation becomes unreliable (multicollinearity).
        # Keep at most min(n_samples//3, 5) covariates — the most informative
        # ones as measured by absolute Pearson correlation with y.
        max_covariates = max(1, min(len(df) // 3, 5))
        if len(cond) > max_covariates:
            corr_vals = df.iloc[:, cond].corrwith(df.iloc[:, y]).abs()
            top_indices = corr_vals.nlargest(max_covariates).index.tolist()
            cond = [df.columns.get_loc(c) for c in top_indices]
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*covariance matrix.*")
            ret = partial_corr(data=df,
                               x=df.columns[x], y=df.columns[y],
                               covar=[df.columns[_] for _ in cond],
                               method='pearson')
        # For a valid transition probability, use absolute correlation values.
        # Use .iloc[0] to extract scalar from Series (avoids FutureWarning).
        r_val = ret['r'].iloc[0]
        if np.isnan(r_val):
            return 0.0
        return abs(float(r_val))
    
    # Calculate the parent nodes set.
    pa_set = {}
    for e in g.edges():
        # Skip self links.
        if e[0] == e[1]:
            continue
        if e[1] not in pa_set:
            pa_set[e[1]] = set([e[0]])
        else:
            pa_set[e[1]].add(e[0])
    # Set an empty set for the nodes without parent nodes.
    for n in g.nodes():
        if n not in pa_set:
            pa_set[n] = set([])
            
    def get_confounders(j: int):
        ret = pa_set[frontend[0]-1].difference([j])
        ret = ret.union(pa_set[j])
        return ret
    
    Q = np.zeros([len(data_head), len(data_head)])
    
    for e in g.edges():
        # Do not add self links.
        if e[0] == e[1]:
            continue
        # e[0] --> e[1]: cause --> result
        # Forward step. 
        # Note for partial correlation, the two variables cannot be the same.
        if frontend[0]-1 != e[0]:
            Q[e[1], e[0]] = get_part_corr(frontend[0]-1, e[0])
        # Backward step
        backward_e = (e[1], e[0])
        # Note for partial correlation, the two variables cannot be the same.
        if backward_e not in g.edges() and frontend[0]-1 != e[1]:
            Q[e[0], e[1]] = rho * get_part_corr(frontend[0]-1, e[1])

    adj = nx.adjacency_matrix(g).todense()
    for i in range(len(data_head)):
        # Calculate P_pc^max
        P_pc_max = []
        # (k, i) in edges.
        for k in adj[:, i].nonzero()[0]:
            # Note for partial correlation, the two variables cannot be the same.
            if frontend[0]-1 != k:
                P_pc_max.append(get_part_corr(frontend[0]-1, k))
        if len(P_pc_max) > 0:
            P_pc_max = np.max(P_pc_max)
        else:
            P_pc_max = 0

        # Note for partial correlation, the two variables cannot be the same.
        if frontend[0]-1 != i:
            q_ii = get_part_corr(frontend[0]-1, i)
            if q_ii > P_pc_max:
                Q[i, i] = q_ii - P_pc_max
            else:
                Q[i, i] = 0

    l = []
    for i in np.sum(Q, axis=1):
        if i > 0:
            l.append(1.0/i)
        else:
            l.append(0.0)
    l = np.diag(l)
    Q = np.dot(l, Q)
    return Q

def randomwalk_metric(
    P,
    epochs,
    start_node,
    teleportation_prob,
    walk_step=50,
    print_trace=False,
):
    n = P.shape[0]
    score = np.zeros([n])
    current = start_node - 1
    for epoch in range(epochs):
        current = start_node - 1
        if print_trace:
            pass
            ##print("\n{:2d}".format(current + 1), end="->")
        for step in range(walk_step):
            if np.sum(P[current]) == 0:
                break
            else:
                next_node = np.random.choice(range(n), p=P[current])
                if print_trace:
                    pass
                    ##print("{:2d}".format(current + 1), end="->")
                score[next_node] += 1
                current = next_node
    label = [i for i in range(n)]
    score_list = list(zip(label, score))
    score_list.sort(key=lambda x: x[1], reverse=True)
    return score_list

def get_gamma(data_head,score_list, eta, lambda_param=0.8):
    gamma = [0 for _ in range(len(data_head))]
    max_vis_time = np.max([i[1] for i in score_list])
    max_eta = np.max(eta)
    for n,vis in score_list:
        gamma[n] = lambda_param * vis / max_vis_time + (1-lambda_param) * eta[n] / max_eta
    return gamma

def evaluate(gamma):
    score_list = sorted(zip([(i+1) for i in range(len(data_head))], gamma), key=lambda x:x[1], reverse=True)
    acc = my_acc(score_list, true_root_cause, n=len(data_head))
    prks = pr_stat(score_list, true_root_cause, k=5)
    print(score_list)
    print(prks)
    print_prk_acc(prks, acc)
    return prks, acc

def root_kpi(data_head,gamma):
    score_list = sorted(zip([(i+1) for i in range(len(data_head))], gamma), key=lambda x:x[1], reverse=True)
    node_rank = [_[0] for _ in score_list]
    result="Top 5 root cause metrics is:"
    for i in range(5):
        result+="("+str(i+1)+")"+data_head[node_rank[i]-1]
        if i==4:
            result+='.'
        else:
            result+=','
    return result

def get_links(data_head,pcmci, results, alpha_level = 0.01):

    sig_links = (results['p_matrix'] <= alpha_level)
    sig_links[:, :, 0] = False
    dic={}
    for j in range(33):
        links = [[p[0], -p[1]] for p in zip(*np.where(sig_links[:, j, :]))]
        dic[str(j)]=links
    g = nx.DiGraph()
    for i in range(len(data_head)):
        g.add_node(i,label=data_head[i])
    for n, links in dic.items():
        for l in links:
            if int(l[0])==int(n):
                continue
            #g.add_edge(int(l[0]), int(n))
            g.add_edge(int(n), int(l[0]))
    return g
