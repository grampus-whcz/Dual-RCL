# MicroCause 方法流程分析

> 本文档对 LocaleXpert 工程中基于 MicroCause 的 Metric 根因分析流程进行完整解释，涵盖论文原理与源码实现的逐行对照。

---

## 一、背景概述

### 1.1 MicroCause 在 LocaleXpert 中的角色

LocaleXpert 是一个基于 LLM 多智能体协作的微服务故障定位框架。其整体架构包含多模态数据分析（Trace、Metric、Log）和 LLM 推理两部分。**MicroCause 方法**作为 Metric 根因分析的核心算法，负责从多维度指标时间序列中定位导致异常的根因指标（Root Cause Metrics）。

在 [run.py:236-274](run.py#L236-L274) 中，MicroCause 流程被完整调用，其输出（Top-5 根因指标）会注入到 LLM Agent（Metric Analysis Expert）的 prompt 中，作为领域知识辅助 LLM 进行推理分析。

### 1.2 整体流水线概览

MicroCause 方法的完整流水线包含以下 7 个步骤：

```
原始指标数据 (Excel)
    │
    ▼
[Step 1] 数据加载与预处理 (load)
    │   loaddata.py → load()
    │   输出: data[T, N], data_head[N]
    ▼
[Step 2] 异常检测 — dSPOT (run_SPOT)
    │   micro.py → run_SPOT()
    │   基于 SPOT/dSPOT 算法，为每个指标生成动态异常阈值
    ▼
[Step 3] 异常偏离度计算 (get_eta)
    │   micro.py → get_eta()
    │   输出: 每个指标的异常严重程度 η
    ▼
[Step 4] 因果发现 — PCMCI (run_pcmci)
    │   micro.py → run_pcmci()
    │   基于 PCMCI 算法发现指标间的因果图结构
    ▼
[Step 5] 因果图提取 (get_links)
    │   micro.py → get_links()
    │   从 PCMCI 结果中提取显著的因果关系，构建有向因果图
    ▼
[Step 6] 转移概率矩阵构建 + 随机游走 (get_Q_matrix_part_corr + randomwalk_metric)
    │   micro.py → get_Q_matrix_part_corr(), randomwalk_metric()
    │   基于偏相关系数构建转移概率矩阵 Q，执行随机游打分
    ▼
[Step 7] 根因排序 (get_gamma + root_kpi)
    │   micro.py → get_gamma(), root_kpi()
    │   综合 η 和随机游走访问频次，输出 Top-5 根因指标
    ▼
最终结果 → 注入 PhaseConfig.json 供 LLM Agent 使用
```

---

## 二、流水线逐步详解

### Step 1: 数据加载与预处理

**代码位置**: [util_funcs/loaddata.py:17-132](util_funcs/loaddata.py#L17-L132)

**调用方式** ([run.py:250-256](run.py#L250-L256)):
```python
dataa, data_head = load(
    f'{date_result}_microcause/{files_microcause}',
    normalize=False,
    zero_fill_method='prevlatter',
    aggre_delta=1,
    verbose=True,
)
```

**输入数据格式**: Excel (.xlsx) 文件，每一行对应一个指标变量，第一列为指标名称，后续列为时间序列观测值。例如：
```
| metric_name               | t1  | t2  | t3  | ... | tN  |
|---------------------------|-----|-----|-----|-----|-----|
| webservice1_cpu_pct       | 0.5 | 0.6 | 0.7 | ... | 0.8 |
| dbservice2_memory_usage   | 1.2 | 1.1 | 1.0 | ... | 1.3 |
| ...                       | ... | ... | ... | ... | ... |
```

**处理流程**:

1. **读取 Excel**: 使用 `openpyxl` 读取 `.xlsx` 文件的 `Sheet1`，提取指标名称列表 (`data_head`) 和数据矩阵
2. **数据聚合** (`aggre_delta`): 当 `aggre_delta > 1` 时，对时间序列做窗口为 `aggre_delta` 的累加聚合（本流程中 `aggre_delta=1`，跳过聚合）
3. **转置**: 原始数据为 `[N_vars, T]`（每行一个变量），转置为 `[T, N_vars]`（每列一个变量），符合时序分析惯例
4. **零值填充** (`zero_fill_method='prevlatter'`):
   - 第一轮：正向遍历，将零值替换为前一个有效值（Previous）
   - 第二轮：反向遍历，将剩余零值替换为后一个有效值（Latter）
   - 这确保了所有零值（可能是缺失数据）都被合理填充
5. **标准化** (`normalize=False`): 本流程中不进行 Z-score 标准化

**输出**:
- `dataa`: numpy 数组，形状 `[T, N]`，N 为指标数量，T 为时间步数
- `data_head`: 指标名称列表，长度 N

---

### Step 2: 异常检测 — dSPOT

**代码位置**: [micro.py:28-42](micro.py#L28-L42)

**调用方式** ([run.py:258](run.py#L258)):
```python
SPOT_res = run_SPOT(dataa, data_head, q=1e-3, d=18)
```

**算法原理**:

dSPOT (drifting SPOT) 是一种**流式异常检测算法**，基于极值理论（Extreme Value Theory, EVT）。它的核心思想是：
- 用初始批次数据校准一个基准阈值
- 对后续流式数据，计算偏离滑动窗口均值的残差
- 使用广义 Pareto 分布 (GPD) 对超过基准阈值的峰值进行建模
- 基于 GPD 计算极端分位数作为动态异常阈值

**参数说明**:
| 参数 | 值 | 含义 |
|---|---|---|
| `q` | `1e-3` | 误报率（风险概率），越小阈值越高，检测越严格 |
| `d` | `18` | 滑动窗口深度，用于计算局部均值以适应数据漂移 |
| `n_init` | `int(0.5 * len(data))` | 初始校准批次大小，取数据前半部分 |

**处理流程**:

```python
def run_SPOT(data, data_head, q=1e-3, d=300, n_init=None):
    result_dict = {}
    if n_init is None:
        n_init = int(0.5 * len(data))       # 默认用前 50% 数据校准
    d = min(d, n_init - 1)                   # 确保 depth < n_init
    for svc_id in range(len(data_head)):     # 对每个指标独立检测
        init_data = data[:n_init, svc_id]    # 校准数据
        _data = data[n_init:, svc_id]        # 待检测流式数据
        s = dSPOT(q, d)                      # 创建 dSPOT 对象
        s.fit(init_data, _data)              # 导入数据
        s.initialize()                       # 校准：拟合 GPD 参数
        results = s.run()                    # 流式检测，返回动态阈值
        result_dict[svc_id] = results
    return result_dict
```

**dSPOT 内部机制** ([SPOT/spot.py:1069-1557](SPOT/spot.py#L1069-L1557)):

1. **校准阶段** (`initialize`):
   - 计算初始数据的滑动均值 `backMean(init_data, depth=18)`
   - 用 `init_data[depth:] - M[:-1]` 得到去漂移后的残差序列
   - 取残差序列的 80 分位数作为初始阈值 `init_threshold`
   - 收集超过初始阈值的峰值（peaks）
   - 使用 **Grimshaw 方法** 拟合 GPD 的参数 (γ, σ)
   - 基于 GPD 计算极端分位数作为异常检测阈值

2. **检测阶段** (`run`):
   - 维护一个长度为 `depth` 的滑动窗口 `W`
   - 对每个新数据点，计算 `x_i - mean(W)` 得到去漂移残差
   - 如果残差超过极端分位数 → 标记为异常（alarm）
   - 如果残差超过初始阈值但未超极端分位数 → 更新 GPD 模型
   - 滑动窗口向前滑动

**输出**: `SPOT_res` 是一个字典，`{svc_id: {'thresholds': [...], 'alarms': [...]}}`，其中 `thresholds` 是每个时间点的动态异常阈值。

---

### Step 3: 异常偏离度计算 (η)

**代码位置**: [micro.py:44-55](micro.py#L44-L55)

**调用方式** ([run.py:259](run.py#L259)):
```python
eta, ab_timepoint = get_eta(dataa, data_head, SPOT_res, int(0.5 * len(dataa)))
```

**算法原理**:

计算每个指标在异常时段**偏离动态阈值的最大相对幅度**，作为该指标的异常严重程度评分 η (eta)。

**处理流程**:

```python
def get_eta(data, data_head, SPOT_res, n_init):
    eta = np.zeros([len(data_head)])           # 每个指标的异常偏离度
    ab_timepoint = [0] * len(data_head)        # 每个指标最早异常时间点
    for svc_id in range(len(data_head)):
        # 找到超过动态阈值的时间点
        mask = data[n_init:, svc_id] > np.array(SPOT_res[svc_id]['thresholds'])
        # 计算相对偏离比例
        ratio = |data[n_init:, svc_id] - thresholds| / thresholds
        if 存在异常点:
            eta[svc_id] = max(ratio[mask])     # 最大相对偏离
            ab_timepoint[svc_id] = min(mask位置) # 最早异常时间
        else:
            eta[svc_id] = 0                    # 无异常
    return eta, ab_timepoint
```

**关键公式**:
```
η_i = max(|x_i(t) - threshold(t)| / threshold(t))，对所有异常时刻 t
```

**输出**:
- `eta`: 数组，每个指标的异常严重程度评分
- `ab_timepoint`: 列表，每个指标最早发生异常的时间索引

---

### Step 4: 因果发现 — PCMCI

**代码位置**: [micro.py:57-62](micro.py#L57-L62)

**调用方式** ([run.py:260](run.py#L260)):
```python
pcmci, pcmci_res = run_pcmci(dataa, pc_alpha=0.05, verbosity=1)
```

**算法原理**:

PCMCI (PC algorithm with Momentary Conditional Independence test) 是一种**基于条件独立性检验的因果发现算法**，来自 `tigramite` 库。它能从多变量时间序列中推断出变量间的因果关系（方向性和时延），特别适合微服务系统中的指标因果分析。

PCMCI 分为两个阶段：
1. **PC 阶段** (Peter-Clark): 通过条件独立性检验逐步筛选可能的因果父节点集合
2. **MCI 阶段** (Momentary Conditional Independence): 对 PC 阶段的结果做更严格的条件独立性检验，去除虚假因果

**处理流程**:

```python
def run_pcmci(data, pc_alpha=0.1, verbosity=0):
    dataframe = pp.DataFrame(data)                    # tigramite 数据格式
    cond_ind_test = ParCorr()                         # 偏相关作为独立性检验
    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=cond_ind_test,
                  verbosity=verbosity)
    pcmci_res = pcmci.run_pcmci(tau_max=10,           # 最大因果时延 10 个时间步
                                pc_alpha=pc_alpha)    # 显著性水平 0.05
    return pcmci, pcmci_res
```

**参数说明**:
| 参数 | 值 | 含义 |
|---|---|---|
| `pc_alpha` | `0.05` | 条件独立性检验的显著性水平 |
| `tau_max` | `10` | 最大因果时延（允许 X 在过去 10 个时间步内影响 Y） |
| `cond_ind_test` | `ParCorr` | 使用偏相关系数作为条件独立性检验统计量 |

**输出**:
- `pcmci`: PCMCI 对象（包含内部状态）
- `pcmci_res`: 字典，包含 `p_matrix`（p 值矩阵）、`val_matrix`（因果强度矩阵）等

---

### Step 5: 因果图提取

**代码位置**: [micro.py:239-256](micro.py#L239-L256)

**调用方式** ([run.py:261](run.py#L261)):
```python
g = get_links(data_head, pcmci, pcmci_res, alpha_level=0.05)
```

**算法原理**:

从 PCMCI 结果中提取**统计显著的因果关系**，构建有向因果图（Directed Causal Graph）。图中节点为指标变量，边表示因果关系。

**处理流程**:

```python
def get_links(data_head, pcmci, results, alpha_level=0.01):
    # 基于 p 值筛选显著因果链接
    sig_links = (results['p_matrix'] <= alpha_level)
    sig_links[:, :, 0] = False               # 排除零时延（瞬时因果）

    dic = {}
    for j in range(33):                       # 对每个目标变量 j
        # 找到所有显著影响 j 的 (source, lag) 对
        links = [[p[0], -p[1]] for p in zip(*np.where(sig_links[:, j, :]))]
        dic[str(j)] = links

    # 构建有向图
    g = nx.DiGraph()
    for i in range(len(data_head)):
        g.add_node(i, label=data_head[i])
    for n, links in dic.items():
        for l in links:
            if int(l[0]) == int(n):
                continue                      # 跳过自环
            g.add_edge(int(n), int(l[0]))     # n ← l[0] → n 被 l[0] 影响
    return g
```

**注意**: `sig_links` 是三维布尔数组 `sig_links[source, target, lag]`，`p_matrix[source, target, lag]` 表示 "source 在 lag 时间步前是否显著影响 target"。代码排除了 `lag=0` 的情况（瞬时因果在微服务场景下不太有意义）。

**输出**: `g` — 一个 `networkx.DiGraph` 有向图，节点为指标索引，边表示因果关系。

---

### Step 6: 转移概率矩阵构建 + 随机游走

这是 MicroCause 的**核心创新步骤**，分为两个子步骤。

#### Step 6a: 构建转移概率矩阵 Q

**代码位置**: [micro.py:95-177](micro.py#L95-L177)

**调用方式** ([run.py:262](run.py#L262)):
```python
Q = get_Q_matrix_part_corr(dataa, data_head, frontend, g, rho=0.2)
```

**算法原理**:

本方法使用**偏相关系数 (Partial Correlation)** 而非简单 Pearson 相关系数来构建转移概率矩阵。偏相关能控制混淆变量的影响，更准确地反映两个指标之间的直接关联。

**关键设计思想**:
- **前端优先**: 以 `frontend` 节点（通常是入口服务指标）为锚点，计算所有其他指标与它的偏相关性
- **因果图引导**: 只在因果图中存在边的节点对之间计算转移概率
- **方向性**: 正向边（因果方向）和反向边（非因果方向）赋予不同权重
- **自环**: 节点的自环概率反映其与前端指标的独立偏相关程度

**处理流程详解**:

1. **计算父节点集合** (`pa_set`): 对因果图中每个节点，收集其所有父节点

2. **确定混淆变量** (`get_confounders`): 对每对变量 (i, j)，混淆变量集为前端节点的父节点集（排除 j）与 j 的父节点集的并集

3. **计算偏相关系数**: 使用 `pingouin.partial_corr()` 库，在控制混淆变量的条件下，计算每个指标与前端指标的偏相关系数

4. **构建 Q 矩阵**:

   ```
   对因果图中的每条边 e[0] → e[1]:
       正向: Q[e[1], e[0]] = |partial_corr(frontend, e[0])|   # 因果方向，全权重
       反向: Q[e[0], e[1]] = ρ × |partial_corr(frontend, e[1])|  # 反向，折扣 ρ=0.2

   对每个节点 i:
       P_pc_max = max(所有指向 i 的邻居与 frontend 的偏相关)
       Q[i, i] = max(0, |partial_corr(frontend, i)| - P_pc_max)  # 自环
   ```

5. **行归一化**: 将每行除以其行和，使每行概率之和为 1

**参数说明**:
| 参数 | 值 | 含义 |
|---|---|---|
| `frontend` | `[1]` | 前端服务（入口）的索引，作为偏相关计算的锚点 |
| `rho` | `0.2` | 反向边的折扣系数，防止非因果方向的传播过强 |

#### Step 6b: 随机游走打分

**代码位置**: [micro.py:179-208](micro.py#L179-L208)

**调用方式** ([run.py:263](run.py#L263)):
```python
vis_list = randomwalk_metric(Q, 1000, frontend[0], teleportation_prob=0, walk_step=15)
```

**算法原理**:

在因果图上进行**蒙特卡洛随机游走**，统计每个节点被访问的频次。被访问频次越高的节点，越可能是传播异常的关键节点。

**处理流程**:

```python
def randomwalk_metric(P, epochs, start_node, teleportation_prob, walk_step=50):
    n = P.shape[0]
    score = np.zeros([n])                  # 访问频次计数器
    current = start_node - 1               # 从前端节点开始
    for epoch in range(epochs):            # 重复 1000 次随机游走
        current = start_node - 1           # 每次都从入口节点开始
        for step in range(walk_step):      # 每次走 15 步
            if P[current] 全为 0:          # 死胡同，停止
                break
            next_node = np.random.choice(  # 按转移概率随机选择下一节点
                range(n), p=P[current])
            score[next_node] += 1          # 记录被访问
            current = next_node
    # 按访问频次降序排列
    score_list = list(zip(range(n), score))
    score_list.sort(key=lambda x: x[1], reverse=True)
    return score_list
```

**参数说明**:
| 参数 | 值 | 含义 |
|---|---|---|
| `epochs` | `1000` | 随机游走重复次数（蒙特卡洛模拟次数） |
| `start_node` | `frontend[0]` | 每次游走的起始节点（前端入口） |
| `teleportation_prob` | `0` | 传送概率（设为 0 表示纯因果图游走） |
| `walk_step` | `15` | 每次游走的步数 |

**输出**: `vis_list` — `[(node_id, visit_count), ...]` 列表，按访问频次降序排列

---

### Step 7: 根因排序

**代码位置**: [micro.py:210-237](micro.py#L210-L237)

**调用方式** ([run.py:264-265](run.py#L264-L265)):
```python
gamma = get_gamma(data_head, vis_list, eta, lambda_param=0.5)
root_metric = root_kpi(data_head, gamma)
```

#### Step 7a: 综合评分 (γ)

```python
def get_gamma(data_head, score_list, eta, lambda_param=0.8):
    gamma = [0] * len(data_head)
    max_vis_time = max([i[1] for i in score_list])
    max_eta = max(eta)
    for n, vis in score_list:
        gamma[n] = λ × (vis / max_vis_time) + (1-λ) × (eta[n] / max_eta)
    return gamma
```

**核心公式**:
```
γ_i = λ × (visit_count_i / max_visit) + (1 - λ) × (η_i / max_η)
```

这是一个**双因素加权评分**：
- **随机游走访问频次** (`visit_count`): 反映指标在因果传播路径上的重要性
- **异常偏离度** (`η`): 反映指标自身的异常严重程度
- `λ = 0.5`（调用时传入 `lambda_param=0.5`）: 两个因素的等权加权

> **注意**: 函数定义中默认 `lambda_param=0.8`（偏重随机游走），但实际调用时传入 `0.5`（等权）。

#### Step 7b: 输出 Top-5 根因指标

```python
def root_kpi(data_head, gamma):
    score_list = sorted(zip(range(1, len(data_head)+1), gamma),
                        key=lambda x: x[1], reverse=True)
    node_rank = [_[0] for _ in score_list]
    result = "Top 5 root cause metrics is:"
    for i in range(5):
        result += f"({i+1}){data_head[node_rank[i]-1]}"
        if i == 4: result += '.'
        else: result += ','
    return result
```

**输出格式示例**:
```
Top 5 root cause metrics is:(1)webservice1_docker_cpu_total_pct,(2)mobservice2_docker_cpu_total_pct,(3)dbservice2_docker_cpu_total_pct,(4)dbservice2_docker_network_in_bytes,(5)dbservice2_docker_network_out_bytes.
```

---

## 三、结果注入 LLM Agent

**代码位置**: [run.py:267-274](run.py#L267-L274)

```python
with open(config_phase_path) as f:
    dataconfig = json.load(f)
dataconfig['MetricAnalysis']['phase_prompt'][0] = \
    "Knowledge: \nAnomaly description:" + metric_an + '\n' + root_metric
with open(config_phase_path, 'w') as file:
    json.dump(dataconfig, file)
```

MicroCause 的输出 (`root_metric`) 会和前面的 Metric 异常描述 (`metric_an`，来自 PatternMatcher CNN 分类器) 一起，作为 **Knowledge** 注入到 `PhaseConfig.json` 的 `MetricAnalysis` 阶段 prompt 中。

LLM Agent（Metric Analysis Expert）随后会基于这些知识进行推理，结合异常模式描述和因果排序结果，输出最终的 Observation → Reasoning → Final Answer。

---

## 四、方法创新点总结

MicroCause 方法相比传统根因定位方法的创新在于：

| 特性 | 传统方法 | MicroCause |
|---|---|---|
| 异常检测 | 固定阈值或 3-sigma | **dSPOT 动态阈值**（基于极值理论，自适应数据漂移） |
| 因果发现 | Pearson 相关或 Granger | **PCMCI 因果发现**（条件独立性检验，控制混淆变量） |
| 传播建模 | 简单相关矩阵 | **偏相关 + 因果图引导的转移概率矩阵** |
| 打分方式 | 单一指标 | **双因素加权**（随机游走访问频次 + 异常偏离度） |

---

## 五、依赖关系图

```
run.py
├── micro.py
│   ├── run_SPOT()           ← SPOT/spot.py (dSPOT 类)
│   ├── get_eta()
│   ├── run_pcmci()          ← tigramite (PCMCI, ParCorr)
│   ├── get_links()          ← networkx (DiGraph)
│   ├── get_Q_matrix_part_corr() ← pingouin (partial_corr)
│   ├── randomwalk_metric()
│   ├── get_gamma()
│   └── root_kpi()
├── util_funcs/loaddata.py
│   └── load()               ← openpyxl (Excel 读取)
└── metric_anomaly.py
    └── CNNClassifier         ← torch (PatternMatcher CNN)
```

---

## 六、关键参数汇总

| 参数 | 默认值 | 调用值 | 作用 |
|---|---|---|---|
| `q` (dSPOT 误报率) | `1e-4` | `1e-3` | 异常检测灵敏度 |
| `d` (dSPOT 窗口深度) | `300` | `18` | 局部均值计算窗口 |
| `pc_alpha` (PCMCI 显著性) | `0.1` | `0.05` | 因果关系显著性水平 |
| `tau_max` (最大因果时延) | — | `10` | 允许的最大因果滞后步数 |
| `alpha_level` (因果图筛选) | `0.01` | `0.05` | 因果链接 p 值阈值 |
| `rho` (反向边折扣) | `0.2` | `0.2` | 非因果方向的权重衰减 |
| `epochs` (随机游走次数) | — | `1000` | 蒙特卡洛模拟次数 |
| `walk_step` (游走步数) | `50` | `15` | 每次游走的步数 |
| `lambda_param` (评分权重) | `0.8` | `0.5` | 随机游走 vs 异常偏离度的权衡 |
| `frontend` (入口节点) | — | `[1]` | 前端服务指标索引 |

---

## 七、论文对照

本工程实现对应论文中以下关键描述：

1. **"We employ dSPOT for metric anomaly detection"** → [micro.py:28-42](micro.py#L28-L42) 中的 `run_SPOT()` 使用 dSPOT 类
2. **"PCMCI is used for causal discovery among metrics"** → [micro.py:57-62](micro.py#L57-L62) 使用 tigramite 库的 PCMCI 算法
3. **"Random walk on the causal graph to propagate anomaly scores"** → [micro.py:179-208](micro.py#L179-L208) 的蒙特卡洛随机游走
4. **"Combining anomaly deviation and random walk visit frequency"** → [micro.py:210-216](micro.py#L210-L216) 的 `get_gamma()` 双因素加权

论文的 MicroCause 方法参考了 Ikram et al. 的 "Root Cause Analysis of Failures in Microservices through Causal Discovery"（即原始 MicroCause 论文），本工程是该方法的 GAIA 数据集适配实现。
