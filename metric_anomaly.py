import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time
import sys

import logging
import sys
import os
import pandas as pd

import warnings
import math
import numpy as np
import pytz
import csv
from datetime import datetime

def get_logger(name):
    is_debug = True if sys.gettrace() else False
    logger=logging.getLogger(name)


    if is_debug:
        log_level=logging.DEBUG
        # logging.basicConfig(level=logging.DEBUG)
    else:
        log_level=logging.INFO
        # logging.basicConfig(level=logging.INFO)

    logger.propagate = False

    if not logger.hasHandlers():
        console = logging.StreamHandler()
        # formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        formatter=logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")
        console.setFormatter(formatter)
        console.setLevel(log_level)
        logger.addHandler(console)
        logger.setLevel(log_level)
    return logger

BATCH_SIZE = 32
INIT_LR = 1e-4
logger = get_logger("classification")

def min_max_normalization(x: np.ndarray):
    if not isinstance(x, np.ndarray):
        logger.error("min_max_normalization: input must be np.ndarray")
    if x.ndim == 1:
        return (x-x.min())/(x.max()-x.min())
    else:
        x = x.T
        x = (x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))
        return x.T

class DataSetCNN(Dataset):
    def __init__(self, data, label, length=30, eval=False):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if not eval:
            assert(data.shape[0] == label.shape[0])
            self.datas = data
            self.label = label
        else:
            self.datas = data
        self.length = length
        self.eval = eval

    def __len__(self):
        return self.datas.shape[0]

    def __getitem__(self, i):
        if self.eval:
            return torch.tensor(self.datas[i].reshape(-1, self.length), dtype=torch.float)
        else:
            return torch.tensor(self.datas[i].reshape(-1, self.length), dtype=torch.float), torch.tensor(self.label[i], dtype=torch.long)
        
class CNN(nn.Module):
    def __init__(self, classes=50):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, 5, 1, 1),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            # nn.AdaptiveAvgPool1d(1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, 1, 1, bias=False),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, 5, 1, 1, bias=False),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.flatten = nn.Sequential(
            nn.Linear(512, 64), nn.BatchNorm1d(64), nn.ReLU())
        # self.out = nn.Linear(256, 64)
        self.out = nn.Linear(64, classes)
        self.dropout = nn.Dropout(0.5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # print(x)
        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.out(x)
        x = nn.LogSoftmax(dim=-1)(x)
        return x


class CNNClassifier(object):
    def __init__(self, model=None, class_num=13, length=30):
        """
        Parameters:
            model: 如果是训练好的模型，这里直接赋值，否则不传参
            class_num: 类别
        """
        if model is None:
            self.model = CNN(class_num)
        else:
            self.model = model
        self.class_num = class_num
        self.length = length

    def fit(self, X: np.ndarray, y: np.array, max_epoch=100, stop_epoch=10):
        """
        训练模型
        Parameters:
            X: 数据
            y: label
        """
        logger.info("start training")
        # ------init--------
        initial_lr = INIT_LR
        batch_size = BATCH_SIZE
        criterion = nn.CrossEntropyLoss()
        count = 0
        train_X, valid_X, train_y, valid_y = train_test_split(
            X, y, test_size=0.3)
        train_data, valid_data = DataSetCNN(train_X, train_y, length=self.length), DataSetCNN(
            valid_X, valid_y, length=self.length)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr)
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(
            valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
        is_gpu = torch.cuda.is_available()
        if is_gpu:
            self.model = self.model.cuda()
            criterion = criterion.cuda()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.8)
        max_f, best_p, best_r = 0, 0, 0
        best_model = None

        # ------train------
        for epoch in range(max_epoch):
            y_true, y_pred = np.array(
                [], dtype=np.int32), np.array([], dtype=np.int32)
            self.model.train()
            # for b in tqdm(train_loader):
            for _, b in enumerate(train_loader):
                optimizer.zero_grad()
                if is_gpu:
                    b[0] = b[0].cuda()
                    b[1] = b[1].cuda()
                out = self.model(b[0])
                batch_pred = out.data.max(1)[1]
                loss = criterion(out, b[1])
                loss.backward()
                optimizer.step()
                if is_gpu:
                    b[1] = b[1].cpu()
                    batch_pred = batch_pred.cpu()
                y_true = np.append(y_true, b[1].numpy())
                y_pred = np.append(y_pred, batch_pred.numpy())
            p1, r1, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=np.unique(y_pred), average="macro")
            scheduler.step()

        # ----------validation-------------
            with torch.no_grad():
                self.model.eval()
                y_true, y_pred = np.array(
                    [], dtype=np.int32), np.array([], dtype=np.int32)
                for _, b in enumerate(valid_loader):
                    if is_gpu:
                        b[0] = b[0].cuda()
                        b[1] = b[1].cuda()
                    out = self.model(b[0])
                    batch_pred = out.data.max(1)[1]
                    if is_gpu:
                        b[1] = b[1].cpu()
                        batch_pred = batch_pred.cpu()
                    y_true = np.append(y_true, b[1].numpy())
                    y_pred = np.append(y_pred, batch_pred.numpy())
                p2, r2, f2, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=np.unique(y_pred), average="macro")
                if f2 > max_f:
                    best_p = p2
                    best_r = r2
                    max_f = f2
                    best_model = self.model
                    count = 0
                else:
                    count += 1
                    if count >= stop_epoch:
                        break
            logger.info("Epoch:{}, Loss:{:.5f}\ntrain_p:{:.5f}, train_r:{:.5f}, train_f:{:.5f},valid_p:{:.5f}, valid_r:{:.5f}, valid_f:{:.5f},best_p:{:.5f}, best_r:{:.5f}, best_f:{:.5f}".format(
                epoch+1, loss.item(), p1, r1, f1, p2, r2, f2, best_p, best_r, max_f))
        self.model = best_model

    def predict(self, X: np.ndarray) -> np.array:
        """
        Parameters:
            X:待预测的数据
        Returns:
            pred:预测结果
        """
        if self.model is None:
            raise ValueError("please load or fit model first")

        test_data = DataSetCNN(min_max_normalization(
            X), None, eval=True, length=self.length)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
        self.model.eval()
        pred = np.array([], dtype=np.int32)
        for _, b in enumerate(test_loader):
            out = self.model(b)
            batch_pred = out.data.max(1)[1]
            pred = np.append(pred, batch_pred.numpy())
        # if pred.shape[0] == 1:
        #     pred = pred[0]
        return pred

    def save_model(self, path):
        """
        Parameters:
            path:模型存储地址，强制加.pt后缀
        """
        if ".pt" not in path:
            path += ".pt"
        torch.save(self.model.state_dict(), path)
        logger.info("save model successfully")

    def load_model(self, path):
        """
        Parameters:
            path:模型存储地址，强制加.pt后缀
        """
        if ".pt" not in path:
            path += ".pt"
        self.model.load_state_dict(torch.load(path))
        logger.info("load model successfully")


def generate_metric_describe(clf,path,services):
    an_type=['Level shift up','Level shift down','Steady increase','Steady decrease','Single spike','Single dip','Transient level shift up','Transient level shift down','Multiple spikes','Multiple dips','Fluctuations']
    anomaly1=[]
    listt = os.listdir(path)
    for name in listt:
        if '.ipynb_' in name:
            continue
        f=0
        for root in services[:5]:
            if root in name:
                f=1
        if f==0:
            continue
        ps = os.path.join(path,name)
        files = os.listdir(ps)
        for f in files:
            if '.ipynb_' in f:
                continue
            try:
                df =  pd.read_csv(os.path.join(ps,f))
                parts = f[:-4].split('_')
                ms = parts[-1].split('-')
                ymean = float(ms[0])
                ystd = float(ms[1])
                i = [int(df.loc[0, 'timestamp']),int(df.loc[29, 'timestamp'])]
                kpi = '_'.join(parts[:-1])
                an='The '+kpi+' metric for the service '+name+' is abnormal,'
                #an="Service "+name+"'s "+kpi+" anomaly,"
                
                tem = df['value'].tolist()
                with warnings.catch_warnings(record=True) as w:
                  warnings.simplefilter("always")
                  if len(tem)!=30:
                    continue
                  p=clf.predict(np.array(tem))
                  if p[0]%2==0:
                    trend='increase'
                    value=max(tem)
                  else:
                    trend='decrease'
                    value=min(tem)
                  if ystd==0:
                      ystd=0.000001
                  score = round(abs((value-ymean)/ystd)/2,2)
    
                  if math.isnan(score):
                    continue
    
                  if score<=1:
                    continue
        
        
                  if not w:
                    start_utc = datetime.fromtimestamp(int(i[0]/1000), tz=pytz.UTC)
                    start_local = start_utc.astimezone(pytz.timezone('Asia/Shanghai'))
                    start = start_local.strftime('%Y-%m-%d %H:%M:%S')
        
                    end_utc = datetime.fromtimestamp(int(i[1]/1000), tz=pytz.UTC)
                    end_local = end_utc.astimezone(pytz.timezone('Asia/Shanghai'))
                    end = end_local.strftime('%Y-%m-%d %H:%M:%S')
        
                    an+='with anomaly pattern of '+an_type[p[0]]+',started at '+start+',ended at '+end+',reach '+str(value)+\
                    ','+trend +' from the previous '+str(tem[0])+',anomaly score is '+str(score)+'.'
                    #print(an)
                    anomaly1.append(an)
            except Exception as e:
                continue
    
                
    return anomaly1


# =====================================================================
# Enhanced Metric Description Generator
# Implements Paper [171] Algorithm 1: statistical feature extraction
# + 11 pattern-specific templates + severity classification
# =====================================================================

# 11 pattern types matching the CNN classifier output
ANOMALY_TYPES = [
    'Level shift up', 'Level shift down', 'Steady increase', 'Steady decrease',
    'Single spike', 'Single dip', 'Transient level shift up',
    'Transient level shift down', 'Multiple spikes', 'Multiple dips', 'Fluctuations',
]


def _classify_severity(score: float) -> str:
    """Classify anomaly severity based on deviation score.

    Args:
        score: Anomaly deviation score (value-mean)/std.

    Returns:
        Severity level string.
    """
    if score >= 5.0:
        return 'critical'
    elif score >= 3.0:
        return 'severe'
    elif score >= 2.0:
        return 'moderate'
    else:
        return 'mild'


def _count_peaks(values: list) -> int:
    """Count the number of local peaks in a value series."""
    peaks = 0
    for i in range(1, len(values) - 1):
        if values[i] > values[i-1] and values[i] > values[i+1]:
            peaks += 1
    return peaks


def _count_dips(values: list) -> int:
    """Count the number of local dips in a value series."""
    dips = 0
    for i in range(1, len(values) - 1):
        if values[i] < values[i-1] and values[i] < values[i+1]:
            dips += 1
    return dips


def _generate_pattern_description(
    pattern_type: str,
    kpi: str,
    service: str,
    values: list,
    ymean: float,
    ystd: float,
    start: str,
    end: str,
    score: float,
) -> str:
    """Generate a pattern-specific description using rich templates.

    Each template emphasises the characteristic of that pattern type:
    - Spikes/dips → suddenness, peak value
    - Level shifts → sustained change, before/after comparison
    - Trends → direction, magnitude of change over time
    - Fluctuations → volatility, range
    """
    arr = np.array(values)
    val_max = float(np.max(arr))
    val_min = float(np.min(arr))
    val_mean = float(np.mean(arr))
    val_std = float(np.std(arr))
    duration_sec = 0
    try:
        duration_sec = int(end.replace(':', '').replace('-', '').replace(' ', '')[-6:]) - \
                       int(start.replace(':', '').replace('-', '').replace(' ', '')[-6:])
        if duration_sec < 0:
            duration_sec += 240000  # cross-midnight
    except Exception:
        duration_sec = 0
    duration_min = abs(duration_sec) // 10000 * 60 + (abs(duration_sec) % 10000) // 100

    severity = _classify_severity(score)
    baseline = values[0]
    n_sigma = score * 2  # approximate sigma deviation

    # Change percentage from baseline
    if abs(baseline) > 1e-10:
        change_pct = abs((val_max - baseline) / baseline) * 100
    else:
        change_pct = abs(val_max - baseline) * 100

    # Service/metric extraction
    metric_name = kpi.split('_')[-1] if '_' in kpi else kpi

    # Pattern-specific descriptions
    if pattern_type == 'Single spike':
        desc = (
            f"The {kpi} metric for service {service} exhibited a sudden spike "
            f"reaching {val_max:.4f}, which is approximately {n_sigma:.1f}σ above the "
            f"historical mean of {ymean:.4f}. The spike occurred around {start}, "
            f"rising sharply from {baseline:.4f} and returning to normal levels "
            f"within approximately {duration_min} minutes."
        )

    elif pattern_type == 'Single dip':
        desc = (
            f"The {kpi} metric for service {service} exhibited a sudden dip "
            f"to {val_min:.4f}, which is approximately {n_sigma:.1f}σ below the "
            f"historical mean of {ymean:.4f}. The dip occurred around {start}, "
            f"dropping from {baseline:.4f} and recovering within approximately "
            f"{duration_min} minutes."
        )

    elif pattern_type == 'Multiple spikes':
        n_peaks = _count_peaks(values)
        desc = (
            f"The {kpi} metric for service {service} displayed {n_peaks} repeated "
            f"spikes over a {duration_min}-minute period starting at {start}. "
            f"Peak values reached {val_max:.4f} (mean: {ymean:.4f}, σ: {ystd:.4f}), "
            f"suggesting intermittent load bursts or resource contention. "
            f"The overall deviation score is {score:.2f}."
        )

    elif pattern_type == 'Multiple dips':
        n_dips = _count_dips(values)
        desc = (
            f"The {kpi} metric for service {service} displayed {n_dips} repeated "
            f"dips over a {duration_min}-minute period starting at {start}. "
            f"The lowest value reached {val_min:.4f} (mean: {ymean:.4f}, σ: {ystd:.4f}), "
            f"suggesting intermittent service degradation or resource starvation. "
            f"The overall deviation score is {score:.2f}."
        )

    elif pattern_type == 'Level shift up':
        # Compute pre-shift and post-shift means
        mid = len(values) // 2
        pre_mean = float(np.mean(arr[:mid]))
        post_mean = float(np.mean(arr[mid:]))
        if abs(pre_mean) > 1e-10:
            shift_pct = (post_mean - pre_mean) / abs(pre_mean) * 100
        else:
            shift_pct = (post_mean - pre_mean) * 100
        desc = (
            f"The {kpi} metric for service {service} shifted upward by "
            f"approximately {shift_pct:.1f}%, from a pre-shift average of "
            f"{pre_mean:.4f} to a post-shift average of {post_mean:.4f}. "
            f"This sustained level shift started around {start} and persisted "
            f"for at least {duration_min} minutes. The historical mean was "
            f"{ymean:.4f} (σ: {ystd:.4f}), deviation score: {score:.2f}."
        )

    elif pattern_type == 'Level shift down':
        mid = len(values) // 2
        pre_mean = float(np.mean(arr[:mid]))
        post_mean = float(np.mean(arr[mid:]))
        if abs(pre_mean) > 1e-10:
            shift_pct = (pre_mean - post_mean) / abs(pre_mean) * 100
        else:
            shift_pct = (pre_mean - post_mean) * 100
        desc = (
            f"The {kpi} metric for service {service} shifted downward by "
            f"approximately {shift_pct:.1f}%, from a pre-shift average of "
            f"{pre_mean:.4f} to a post-shift average of {post_mean:.4f}. "
            f"This sustained level drop started around {start} and persisted "
            f"for at least {duration_min} minutes. The historical mean was "
            f"{ymean:.4f} (σ: {ystd:.4f}), deviation score: {score:.2f}."
        )

    elif pattern_type == 'Steady increase':
        first_val = values[0]
        last_val = values[-1]
        if abs(first_val) > 1e-10:
            inc_pct = (last_val - first_val) / abs(first_val) * 100
        else:
            inc_pct = (last_val - first_val) * 100
        desc = (
            f"The {kpi} metric for service {service} showed a steady upward trend, "
            f"increasing by {inc_pct:.1f}% from {first_val:.4f} to {last_val:.4f} "
            f"over a {duration_min}-minute period starting at {start}. "
            f"The peak value was {val_max:.4f}, reaching approximately {n_sigma:.1f}σ "
            f"above the historical mean ({ymean:.4f}). Deviation score: {score:.2f}."
        )

    elif pattern_type == 'Steady decrease':
        first_val = values[0]
        last_val = values[-1]
        if abs(first_val) > 1e-10:
            dec_pct = (first_val - last_val) / abs(first_val) * 100
        else:
            dec_pct = (first_val - last_val) * 100
        desc = (
            f"The {kpi} metric for service {service} showed a steady downward trend, "
            f"decreasing by {dec_pct:.1f}% from {first_val:.4f} to {last_val:.4f} "
            f"over a {duration_min}-minute period starting at {start}. "
            f"The lowest value was {val_min:.4f}, approximately {n_sigma:.1f}σ "
            f"below the historical mean ({ymean:.4f}). Deviation score: {score:.2f}."
        )

    elif pattern_type == 'Transient level shift up':
        desc = (
            f"The {kpi} metric for service {service} exhibited a transient upward "
            f"level shift, temporarily rising from {baseline:.4f} to a peak of "
            f"{val_max:.4f} around {start}, before partially recovering. "
            f"The peak was approximately {n_sigma:.1f}σ above the historical mean "
            f"({ymean:.4f}, σ: {ystd:.4f}). Duration: approximately {duration_min} minutes. "
            f"Deviation score: {score:.2f}."
        )

    elif pattern_type == 'Transient level shift down':
        desc = (
            f"The {kpi} metric for service {service} exhibited a transient downward "
            f"level shift, temporarily dropping from {baseline:.4f} to {val_min:.4f} "
            f"around {start}, before partially recovering. "
            f"The dip was approximately {n_sigma:.1f}σ below the historical mean "
            f"({ymean:.4f}, σ: {ystd:.4f}). Duration: approximately {duration_min} minutes. "
            f"Deviation score: {score:.2f}."
        )

    elif pattern_type == 'Fluctuations':
        val_range = val_max - val_min
        cv = val_std / abs(val_mean) if abs(val_mean) > 1e-10 else val_std
        desc = (
            f"The {kpi} metric for service {service} displayed irregular fluctuations "
            f"over a {duration_min}-minute period starting at {start}. "
            f"The metric oscillated between {val_min:.4f} and {val_max:.4f} "
            f"(range: {val_range:.4f}, coefficient of variation: {cv:.2f}). "
            f"Historical mean: {ymean:.4f}, σ: {ystd:.4f}. "
            f"Deviation score: {score:.2f}. This volatile behavior may indicate "
            f"unstable resource allocation or oscillating load."
        )

    else:
        # Fallback: generic description
        desc = (
            f"The {kpi} metric for service {service} is abnormal with pattern "
            f"'{pattern_type}', from {start} to {end}. "
            f"Value range: [{val_min:.4f}, {val_max:.4f}], historical mean: {ymean:.4f}. "
            f"Deviation score: {score:.2f}."
        )

    # Append severity tag
    severity_map = {
        'mild': '[MILD]',
        'moderate': '[MODERATE]',
        'severe': '[SEVERE]',
        'critical': '[CRITICAL]',
    }
    desc += f" Severity: {severity_map.get(severity, '[UNKNOWN]')}."

    return desc


def enhance_metric_describe(clf, path, services, data_head=None):
    """Enhanced metric anomaly description generator.

    Implements Paper [171] Algorithm 1 with:
      - Statistical feature extraction (mean, std, peak, change %, duration)
      - 11 pattern-specific description templates
      - Anomaly severity classification (mild/moderate/severe/critical)
      - Cross-metric correlation hints for same-service anomalies

    Args:
        clf: Trained CNNClassifier instance.
        path: Path to the metric fault directory (containing service subdirs).
        services: List of root cause service names to filter metrics.
        data_head: Optional list of all metric names (for cross-reference).

    Returns:
        List of enhanced anomaly description strings.
    """
    anomaly_descriptions = []
    # Track per-service anomaly count for cross-metric correlation hints
    service_anomaly_count = {}

    listt = os.listdir(path)
    for name in listt:
        if '.ipynb_' in name:
            continue
        # Filter: only process metrics for top-5 root services
        f = 0
        for root in services[:5]:
            if root in name:
                f = 1
        if f == 0:
            continue

        ps = os.path.join(path, name)
        files = os.listdir(ps)
        for fi in files:
            if '.ipynb_' in fi:
                continue
            try:
                df = pd.read_csv(os.path.join(ps, fi))
                parts = fi[:-4].split('_')
                ms = parts[-1].split('-')
                ymean = float(ms[0])
                ystd = float(ms[1])
                timestamps = [int(df.loc[0, 'timestamp']), int(df.loc[29, 'timestamp'])]
                kpi = '_'.join(parts[:-1])

                tem = df['value'].tolist()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    if len(tem) != 30:
                        continue

                    # CNN pattern prediction
                    p = clf.predict(np.array(tem))
                    pattern_type = ANOMALY_TYPES[p[0]] if p[0] < len(ANOMALY_TYPES) else 'Fluctuations'

                    if p[0] % 2 == 0:
                        trend = 'increase'
                        value = max(tem)
                    else:
                        trend = 'decrease'
                        value = min(tem)

                    if ystd == 0:
                        ystd = 0.000001
                    score = round(abs((value - ymean) / ystd) / 2, 2)

                    if math.isnan(score):
                        continue
                    if score <= 1:
                        continue

                    if not w:
                        # Format timestamps
                        start_utc = datetime.fromtimestamp(int(timestamps[0] / 1000), tz=pytz.UTC)
                        start_local = start_utc.astimezone(pytz.timezone('Asia/Shanghai'))
                        start = start_local.strftime('%Y-%m-%d %H:%M:%S')

                        end_utc = datetime.fromtimestamp(int(timestamps[1] / 1000), tz=pytz.UTC)
                        end_local = end_utc.astimezone(pytz.timezone('Asia/Shanghai'))
                        end = end_local.strftime('%Y-%m-%d %H:%M:%S')

                        # Generate pattern-specific description
                        desc = _generate_pattern_description(
                            pattern_type=pattern_type,
                            kpi=kpi,
                            service=name,
                            values=tem,
                            ymean=ymean,
                            ystd=ystd,
                            start=start,
                            end=end,
                            score=score,
                        )
                        anomaly_descriptions.append(desc)

                        # Track per-service count
                        service_anomaly_count[name] = service_anomaly_count.get(name, 0) + 1
            except Exception as e:
                continue

    # Append cross-metric correlation hints
    for svc, count in service_anomaly_count.items():
        if count >= 2:
            anomaly_descriptions.append(
                f"[Cross-Metric Alert] Service {svc} has {count} anomalous metrics "
                f"simultaneously, suggesting a systemic issue affecting multiple "
                f"resource dimensions of this service."
            )

    return anomaly_descriptions
        
