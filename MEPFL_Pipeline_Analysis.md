# MEPFL 方法流程分析

> 本文档对 LocaleXpert 工程中基于 MEPFL (Microservice Endpoint Failure Localization) 的 Trace 根因分析流程进行完整解释，涵盖数据模型、训练阶段、推理阶段和上下游衔接。

---

## 一、MEPFL 在 LocaleXpert 中的角色

LocaleXpert 对微服务故障定位采用多模态数据（Trace、Metric、Log）+ LLM 推理的架构。**MEPFL** 负责 Trace 侧的根因分析——从异常 Trace 数据中定位导致故障的根因微服务（Root Cause Services）。

在 [run.py:188-213](run.py#L188-L213) 中，MEPFL 被调用并输出 Top-5 根因服务排名，该结果注入到 LLM Agent（Trace Analysis Expert）的 prompt 中。

整体调用链路：

```
原始 Trace 数据 (pkl)
    │
    ▼
[前置] Trace 异常检测 (trace_anomaly.py)
    │   生成异常描述文本 → trace_an
    ▼
[核心] MEPFL Trace 根因定位 (mepfl.py)
    │   RF 异常分类 → MLP 服务定位 → Top-5 根因服务
    ▼
[后置] 结果注入 PhaseConfig.json
    │   Knowledge = trace_an + Top-5 root cause
    ▼
[下游] LLM Trace Analysis Expert 推理
```

---

## 二、前置阶段：Trace 异常检测与描述生成

在调用 MEPFL 之前，[run.py:170-185](run.py#L170-L185) 先执行了 Trace 异常检测，生成自然语言异常描述：

```python
trace_ans = anomaly_detect_and_generate_describe(
    f'{date_result}_trace_ano/{date_result}.pkl',     # 调用路径字典
    f'{date_result}_trace_ano/normal_datasets',        # 正常基线数据
    f'{date_result}_trace_ano/data/{files_trace}'      # 待检测的 trace CSV
)
```

**代码位置**: [trace_anomaly.py:170-300](trace_anomaly.py#L170-L300)

### 2.1 数据模型

Trace 数据在系统中以树状结构组织，定义在 [data/data_models.py](data/data_models.py)：

```
Span (跨度/调用段)
├── trace_id       : 所属追踪 ID
├── span_id        : 当前跨度 ID
├── parent_span_id : 父跨度 ID (根节点为 None)
├── children_span_list : 子跨度列表
├── start_time     : 开始时间
├── duration       : 持续时间 (毫秒)
├── service_name   : 所属微服务
├── status_code    : 状态码
└── operation_name : 操作名称

Trace (追踪)
├── trace_id       : 追踪 ID
├── root_span      : 根跨度 (入口调用)
├── span_count     : 跨度数量
└── anomaly_type   : 异常类型 (0=正常, 1=延迟异常, 2=结构异常, 3=两者兼有)
```

### 2.2 异常检测流程

`anomaly_detect_and_generate_describe()` 的核心逻辑：

1. **加载调用路径字典**: 从 pkl 文件中加载 `call_path_dict`（所有已知调用路径的映射，如 `start#webservice1#logservice1#dbservice1`）

2. **解析异常 Trace**: 读取 CSV 格式的 trace 数据，按 `trace_id` 分组，构建每个 trace 的调用树，转换为 **Span Time Vector (STV)** —— 一个固定长度的向量，每个维度对应一条调用路径的响应时间

3. **与正常基线对比**:
   - 加载正常 trace 的 STV 数据作为基线
   - 对每条异常 trace，找到调用路径结构完全相同的正常 trace 子集
   - 计算正常基线的均值 `μ` 和标准差 `σ`

4. **3-sigma 异常判定**:
   ```python
   if value_ab[k] > (means[k] + 3*std_devs[k])  # 超过上界
   or value_ab[k] < (means[k] - 3*std_devs[k]):  # 低于下界
   ```
   异常得分 = `(实际值 - 均值) / (3 × 标准差)`

5. **生成异常描述文本**:
   - **结构异常**: 调用路径在正常基线中不存在（如新服务出现）
   - **延迟异常**: 某个调用段的响应时间显著偏高或偏低

   输出示例：
   ```
   The summary of anomaly is that the trace call timeout, trace_id is 4f958a1d...,
   the call path is webservice1->logservice1->dbservice1->redisservice1,
   The services with anomalies include: dbservice1 the call time for 1336 ms
   exceeding the normal upper limit by 971 ms, redisservice1 the call time for
   1790 ms exceeding the normal upper limit by 831 ms.
   ```

---

## 三、MEPFL 核心流程

### 3.1 总体架构

MEPFL 采用 **两阶段机器学习** 方法：

```
                      输入: Fault Injection Trace 数据 (.pkl)
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Step 1: 数据加载         │
                    │  从 pkl 中读取 trace_list │
                    └─────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │  Step 2: 特征提取         │
                    │  Trace → 特征向量 (32维)  │
                    └─────────┬───────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
          ┌──────────────────┐  ┌──────────────────┐
          │ Stage A: 异常分类  │  │ Stage B: 服务定位  │
          │ (RandomForest)    │  │ (MLP + Softmax)   │
          │                   │  │                    │
          │ Trace → 正常/异常  │  │ 异常Trace → 故障服务 │
          └────────┬─────────┘  └────────┬──────────┘
                   │                     │
                   └──────────┬──────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │  Step 3: 服务排序         │
                    │  聚合概率 → Top-5 排名    │
                    └─────────────────────────┘
```

### 3.2 调用入口

**代码位置**: [run.py:188-213](run.py#L188-L213)

```python
# 1. 定位数据文件
tracerca_dirs = os.listdir(f'{date_result}_tracerca')
files_tracerace = find_closest_file(tracerca_dirs, time_result)

# 2. 调用 MEPFL 核心函数
root_service = mepfl(f'./{date_result}_tracerca/{files_tracerace}')

# 3. 格式化 Top-5 结果
root_se = ''
for i in range(5):
    root_se += "(" + str(i+1) + ")" + root_service[i]
    ...

# 4. 注入 PhaseConfig 供 LLM 使用
dataconfig['TraceAnalysis']['phase_prompt'][0] = \
    "Knowledge:\n Anomaly description:" + trace_an + '\nTop5 root cause:' + root_se
```

**输入数据格式**: `{date}_tracerca/{time}.pkl` — pickle 序列化的故障注入列表，每个条目包含：
```python
{
    'date': datetime,          # 故障注入日期
    'service': str,            # 被注入故障的服务名
    'time': datetime,          # 故障注入时间
    'trace_list': [Span, ...]  # 该故障时段内的 trace 数据（已解析为 Span/Trace 对象树）
}
```

---

### Step 1: 数据加载

**代码位置**: [mepfl.py:94-109](mepfl.py#L94-L109)

```python
def mepfl(path):
    # 加载预训练模型
    rf_model  = pickle.load(open('./mepfl_model_gaia/rf_model.pkl', 'rb'))   # RandomForest
    mlp_model = pickle.load(open('./mepfl_model_gaia/mlp_model.pkl', 'rb'))  # MLP

    # 加载故障注入数据
    fault_injection_list = pickle.load(open(path, 'rb'))
```

模型文件说明：
| 文件 | 类型 | 用途 |
|---|---|---|
| `rf_model.pkl` | `sklearn.ensemble.RandomForestClassifier` | Trace 异常分类（正常 vs 异常） |
| `mlp_model.pkl` | `sklearn.neural_network.MLPClassifier` | 异常 Trace 的故障服务定位 |

### Step 2: 特征提取

**代码位置**: [mepfl.py:55-76](mepfl.py#L55-L76)

对每条 Trace 提取一个 **32 维特征向量**：

```python
def get_trace_list_vector(trace_list):
    for trace in trace_list:
        # 32 = 2 + 10个服务 × 3个特征
        trace.vector = np.zeros((2 + len(total_service_list) * 3))  # [0..31]

        # 全局特征
        trace.vector[0] = len(total_service_list)      # 系统总服务数 (10)
        trace.vector[1] = len(get_trace_service_list(trace))  # 该trace涉及的服务数

        # BFS 遍历所有 Span
        queue = [trace.root_span]
        while queue:
            span = queue.pop(0)
            queue.extend(span.children_span_list)

            duration = span.duration
            status = 0 if span.status_code in ('Ok','OK') else int(span.status_code)
            process_time = span.duration - max([child.duration for child in span.children_span_list], default=0.0)

            # 每个服务占 3 个维度: [duration, status, process_time]
            idx = service_index_dict[span.service_name]
            trace.vector[-1 + (idx + 1) * 3] = duration       # 该服务的调用持续时间
            trace.vector[0 + (idx + 1) * 3] = status          # 该服务的状态码
            trace.vector[1 + (idx + 1) * 3] = process_time    # 该服务的自身处理时间
```

**特征向量结构**（以 10 个服务为例）：

| 维度 | 0 | 1 | 2-4 | 5-7 | 8-10 | ... | 29-31 |
|---|---|---|---|---|---|---|---|
| 含义 | 系统总服务数 | 涉及服务数 | webservice1 | webservice2 | redisservice2 | ... | dbservice1 |
| | | | dur / status / proc | dur / status / proc | dur / status / proc | | dur / status / proc |

**关键设计**：
- `duration`（持续时间）: Span 从开始到结束的总时长，包含子调用的时间
- `process_time`（处理时间）: `duration - max(children.duration)`，即该服务自身消耗的时间（扣除等待下游服务的时间）
- 处理时间能更精准地反映服务本身的异常情况

### Step 3: Stage A — 异常分类 (RandomForest)

**代码位置**: [mepfl.py:126-130](mepfl.py#L126-L130)

```python
trace_vector_list = [_.vector for _ in trace_list]
trace_anomaly_list = rf_model.predict(trace_vector_list)   # 二分类: 0=正常, 1=异常

# 筛选出异常 trace
anomaly_trace_list = []
for index in range(len(trace_list)):
    if trace_anomaly_list[index] == 1:
        anomaly_trace_list.append(trace_list[index])
```

**作用**: 使用预训练的 RandomForest 分类器，从所有 trace 中筛选出异常 trace。

**训练方式** ([scripts/step3_train_mepfl.py:297-307](scripts/step3_train_mepfl.py#L297-L307)):

训练标签的生成规则：
- 若 trace 在故障注入时间 60 秒内且包含故障服务 → `anomaly_type = 1`（异常）
- 否则 → `anomaly_type = 0`（正常）

```python
for trace in trace_list:
    time_diff = (trace.root_span.start_time - fault_time).total_seconds()
    if time_diff <= 60 and fault_service in get_trace_service_list(trace):
        trace.anomaly_type = 1   # 异常
    else:
        trace.anomaly_type = 0   # 正常

rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(cls_x, cls_y)       # cls_x: 特征向量, cls_y: 0/1 标签
```

### Step 4: Stage B — 服务定位 (MLP)

**代码位置**: [mepfl.py:132-145](mepfl.py#L132-L145)

```python
loc_trace_vector_list = [_.vector for _ in anomaly_trace_list]
probs = mlp_model.predict_proba(loc_trace_vector_list)   # 输出每个服务的故障概率

# 聚合所有异常 trace 的概率
sum_proba = np.zeros((len(total_service_list)))            # [10] 维
for prob in probs:
    sum_proba += prob                                       # 累加概率

# 按概率排序
service_score_list = [
    (total_service_list[index], sum_proba[index])
    for index in range(len(total_service_list))
]
service_score_list.sort(key=lambda x: x[1], reverse=True)
sorted_service_list = [_[0] for _ in service_score_list]
```

**作用**: 对筛选出的异常 trace，使用 MLP 分类器预测每个服务是根因的概率，然后将所有异常 trace 的预测概率累加，得到最终的服务排名。

**MLP 模型结构** ([mepfl.py:80-92](mepfl.py#L80-L92)):

```python
class MLPWithSoftmax(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size)   # 输入层 → 隐藏层
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)  # 隐藏层 → 输出层
        self.softmax = Softmax(dim=1)                # 转为概率分布

# 实际训练使用的是 sklearn 的 MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
```

> **注意**: 代码中定义了 PyTorch 版 `MLPWithSoftmax`（[mepfl.py:80-92](mepfl.py#L80-L92)），但实际推理使用的是 sklearn 的 `MLPClassifier`，其 `predict_proba()` 方法直接输出各服务的概率分布。

**训练方式** ([scripts/step3_train_mepfl.py:315-326](scripts/step3_train_mepfl.py#L315-L326)):

```python
# 只用异常 trace 训练
loc_x = np.array([t.vector for t in loc_traces])
loc_y = np.array([t.fault_service_index for t in loc_traces])   # 10分类: 0-9 对应 10 个服务

mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model.fit(loc_x, loc_y)
```

训练标签是**故障注入的目标服务索引**（0-9），这是一个 10 分类问题。

### Step 5: 输出 Top-5 根因服务

**代码位置**: [mepfl.py:145-147](mepfl.py#L145-L147)

```python
sorted_service_list = [_[0] for _ in service_score_list]  # 按概率降序排列的服务名列表
return sorted_service_list
```

**输出格式** (在 [run.py:199-205](run.py#L199-L205) 中格式化):

```python
root_se = ''
for i in range(5):
    root_se += "(" + str(i+1) + ")" + root_service[i]
    if i == 4: root_se += '.'
    else: root_se += ','
```

输出示例：
```
(1)webservice1,(2)webservice2,(3)redisservice2,(4)redisservice1,(5)mobservice1.
```

---

## 四、结果注入 LLM Agent

**代码位置**: [run.py:207-213](run.py#L207-L213)

```python
dataconfig['TraceAnalysis']['phase_prompt'][0] = \
    "Knowledge:\n Anomaly description:" + trace_an + '\nTop5 root cause:' + root_se
```

注入到 `PhaseConfig.json` 的 `TraceAnalysis` 阶段，供 **Trace Analysis Expert** Agent 使用。LLM Agent 会结合：
- `trace_an`: 异常 Trace 的自然语言描述（来自 `trace_anomaly.py`）
- `root_se`: MEPFL 输出的 Top-5 根因服务排名

进行 Observation → Reasoning → Final Answer 推理。

---

## 五、辅助模块说明

### 5.1 Spectrum (频谱分析)

**代码位置**: [spectrum.py](spectrum.py)

Spectrum-based Fault Localization (SBFL) 是一种经典的软件故障定位技术：

```python
def spectrum(trace_list, service_index_dict):
    for service in service_statistic_dict:
        ef = 0  # 异常 trace 中包含该服务的数量 (Error Fail)
        ep = 0  # 正常 trace 中包含该服务的数量 (Error Pass)
        nf = 0  # 异常 trace 中不包含该服务的数量 (Not Fail)

        for trace in trace_list:
            if trace.anomaly_type == 1:        # 异常 trace
                if service in service_set:      # 包含该服务
                    ef += 1
                else:                           # 不包含
                    nf += 1
            else:                               # 正常 trace
                if service in service_set:      # 包含该服务
                    ep += 1

        # Ochiai 系数: ef / sqrt((ef+ep) * (ef+nf))
        statistic = ef / (((ef + ep) * (ef + nf)) ** 0.5)
```

> **注意**: Spectrum 模块在当前 MEPFL 推理流程中**未被直接调用**，但它是 MEPFL 框架的可选组件，可通过集成提升定位精度。

### 5.2 Random Walk on Trace (Trace 随机游走)

**代码位置**: [randomwalk_trace.py](randomwalk_trace.py)

这是 Trace 侧的随机游打分方法（与 MicroCause 中 Metric 侧的随机游走类似）：

```python
def randomwalk(trace_list, service_index_dict, service_list):
    # 1. 构建服务处理时间向量
    # 2. 计算 frontend 与每个服务的 Pearson 相关系数
    # 3. 构建转移矩阵 A:
    #    - 正向边: A[i][j] = correlation[j]
    #    - 反向边: A[j][i] = rho * correlation[i]  (rho=0.5 折扣)
    #    - 自环: A[i][i] = max(0, correlation[i] - max(children_correlations))
    # 4. 迭代求解: x = 0.5 * A @ x + 0.5 * v  (50 次迭代)
    # 5. 返回每个服务的得分
```

> **注意**: 与 Spectrum 类似，`randomwalk_trace.py` 在当前 MEPFL 推理路径中**未被直接调用**。当前 MEPFL 使用的是 RF + MLP 两阶段分类器方案。这些模块可作为备选或集成方法。

---

## 六、MEPFL 训练流程

MEPFL 模型的训练由 [scripts/step3_train_mepfl.py](scripts/step3_train_mepfl.py) 完成。

### 6.1 训练数据准备

训练数据来自 GAIA 数据集的故障注入实验，预处理流程：

1. **故障注入记录解析** (`parse_fault_injection.py`):
   - 读取 `run_table_2021-07.csv` 获取故障注入时间和目标服务
   - 对每个故障注入，收集故障时间前后 180 秒内的 trace 数据
   - 同时收集排除故障时段的正常 trace 数据
   - 保存为 `fault_injection_tracerank/fault_injection_list_{date}.pkl`

2. **训练集构建** (`step3_train_mepfl.py:185-241`):
   - 读取 pkl 文件，将原始 CSV 行解析为 Trace/Span 对象树
   - 标签规则：在故障时间 60s 内且包含故障服务的 trace → 异常（标签 1）；其余 → 正常（标签 0）
   - 异常 trace 的故障服务索引作为 MLP 的分类标签

### 6.2 模型训练

```python
# Stage A: RandomForest 异常分类器
rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(cls_x, cls_y)    # X: [N_traces, 32], y: [N_traces] (0 or 1)

# Stage B: MLP 服务定位分类器
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model.fit(loc_x, loc_y)   # X: [N_anomaly_traces, 32], y: [N_anomaly_traces] (0-9)
```

### 6.3 输出模型文件

| 文件 | 内容 | 维度 |
|---|---|---|
| `rf_model.pkl` | RandomForest 二分类模型 | 输入: 32 维 → 输出: 2 类 (正常/异常) |
| `mlp_model.pkl` | MLP 10 分类模型 | 输入: 32 维 → 输出: 10 类 (10 个服务) |
| `temp_fault_injection.pkl` | 测试用故障注入数据 | 用于离线评估 |

---

## 七、服务列表

GAIA 数据集包含 **10 个微服务**，在 MEPFL 中定义如下 ([mepfl.py:35-37](mepfl.py#L35-L37))：

```python
total_service_list = [
    'webservice1',    # 0 - Web 服务 1 (前端入口)
    'webservice2',    # 1 - Web 服务 2
    'redisservice2',  # 2 - Redis 缓存服务 2
    'redisservice1',  # 3 - Redis 缓存服务 1
    'mobservice1',    # 4 - 移动端服务 1
    'logservice1',    # 5 - 日志服务 1
    'mobservice2',    # 6 - 移动端服务 2
    'logservice2',    # 7 - 日志服务 2
    'dbservice2',     # 8 - 数据库服务 2
    'dbservice1'      # 9 - 数据库服务 1
]
```

---

## 八、完整流水线时序图

```
┌──────────────────────────────────────────────────────────────┐
│                       run.py 主流程                           │
│                                                              │
│  1. 解析任务描述 → 提取 date_result, time_result              │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ Trace 异常检测 (trace_anomaly.py)                    │     │
│  │                                                      │     │
│  │  加载 call_path_dict (.pkl)                          │     │
│  │       ↓                                              │     │
│  │  解析异常 trace CSV → 构建 STV 向量                   │     │
│  │       ↓                                              │     │
│  │  与正常基线对比 → 3-sigma 异常判定                     │     │
│  │       ↓                                              │     │
│  │  输出: trace_an (自然语言异常描述)                     │     │
│  └─────────────────────────────────────────────────────┘     │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ MEPFL 根因定位 (mepfl.py)                            │     │
│  │                                                      │     │
│  │  加载 rf_model.pkl + mlp_model.pkl                   │     │
│  │       ↓                                              │     │
│  │  加载故障注入 trace 数据 (.pkl)                       │     │
│  │       ↓                                              │     │
│  │  特征提取: Trace → 32 维向量                          │     │
│  │       ↓                                              │     │
│  │  Stage A: RF 异常分类 → 筛选异常 trace                 │     │
│  │       ↓                                              │     │
│  │  Stage B: MLP 服务定位 → 各服务故障概率                │     │
│  │       ↓                                              │     │
│  │  概率聚合 → Top-5 根因服务排名                        │     │
│  │       ↓                                              │     │
│  │  输出: root_service (服务名列表)                      │     │
│  └─────────────────────────────────────────────────────┘     │
│                          ↓                                   │
│  格式化: root_se = "(1)svc1,(2)svc2,...,(5)svc5."             │
│                          ↓                                   │
│  注入 PhaseConfig.json:                                       │
│    TraceAnalysis.phase_prompt[0] =                            │
│      "Knowledge:\n Anomaly description:" + trace_an           │
│      + "\nTop5 root cause:" + root_se                         │
│                          ↓                                   │
│  LLM Trace Analysis Expert 推理                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 九、关键参数汇总

| 参数 | 值 | 作用 |
|---|---|---|
| 特征维度 | 32 | 2 (全局) + 10 服务 × 3 (duration/status/process_time) |
| 服务数量 | 10 | GAIA 数据集的微服务总数 |
| RF 异常分类 | `RandomForestClassifier` | 判断 trace 是否异常 |
| MLP 服务定位 | `MLPClassifier(100,)` | 预测故障服务 (10 分类) |
| 异常 trace 筛选阈值 | `len(trace_list) > 10` | Trace 数量过少的故障注入被跳过 |
| 训练标签时间窗口 | 60 秒 | 故障注入前后 60s 内的 trace 视为异常 |
| 故障 trace 收集窗口 | 180 秒 | 故障注入前后 180s 的 trace 被收集 |

---

## 十、与 MicroCause 的对比

| 特性 | MEPFL (Trace 侧) | MicroCause (Metric 侧) |
|---|---|---|
| 数据类型 | Trace (调用链) | Metric (时间序列) |
| 异常检测 | RF 二分类器 | dSPOT 极值理论 |
| 因果分析 | 无显式因果图 | PCMCI 因果发现 |
| 根因定位 | MLP 多分类 + 概率聚合 | 偏相关 + 随机游走 |
| 输出 | Top-5 根因服务 | Top-5 根因指标 |
| 模型 | 预训练 (RF + MLP) | 无需训练 (纯算法) |
| 依赖库 | sklearn | tigramite, pingouin, networkx |

两者结果最终在 `RootCauseAnalysis` 阶段汇合：
```python
dataconfig['RootCauseAnalysis']['phase_prompt'][0] = \
    "Knowledge: " + root_metric + "Top5 root cause:" + root_se
```

由 **Fault Diagnosis Expert** Agent 综合两方面的分析结果，输出最终的根因结论。
