# LocaleXpert 与 GAIA 数据集不匹配问题分析报告

## 一、LocaleXpert 核心方法

论文提出 **LocaleXpert** —— 一个基于 LLM 多智能体协作的微服务故障定位框架。核心思路是：

1. **多模态数据转自然语言**：将 metrics、traces、logs 三种遥测数据通过管道转换为 LLM 可理解的自然语言描述
2. **专用专家智能体**：Trace Expert、Metric Expert、Log Expert 分别分析各数据类型
3. **外部故障定位模块集成**：
   - Trace → **MEPFL**（微服务追踪故障定位，基于 RF + MLP 模型）
   - Metric → **MicroCause**（基于 PCMCI 因果推断 + Random Walk）
   - Metric 异常模式分类 → **PatternMatcher**（CNN 分类器，11 类异常模式）
4. **解释模块**：Failure Localization Expert 汇总三位专家分析结果，输出可解释的根因结论

## 二、`run.py` 运行时期望的数据目录结构

`run.py` 解析任务描述中的日期（如 `2022/05/03 00:50`），提取 `date_result="0503"`, `time_result="00-50"`，然后期望以下**预处理后的目录**：

```
工作目录/
├── {MMDD}_trace_ano/          # 如 0503_trace_ano/
│   ├── {MMDD}.pkl             # 调用路径字典 (call_path_dict)
│   ├── normal_datasets/       # 正常 trace 的 STV 数据
│   └── data/                  # 按时间窗口筛选的 trace CSV
│       └── {time_result}*.csv
├── {MMDD}_tracerca/           # 如 0503_tracerca/
│   └── {time_result}*.pkl     # fault_injection_list pkl（含 trace_list）
├── {MMDD}_metric_fault/       # 如 0503_metric_fault/
│   └── {service_name}/        # 每个服务的目录
│       └── {metric}_{mean}-{std}.csv  # 指标 CSV 文件
├── {MMDD}_microcause/         # 如 0503_microcause/
│   └── {time_result}*.xlsx    # MicroCause 输入数据 (Excel)
├── {MMDD}_log_fault/          # 如 0503_log_fault/
│   └── {time_result}/         # 按时间的子目录
│       └── {service}_*.csv    # 日志 CSV 文件
├── mepfl_model_gaia/          # MEPFL 预训练模型
│   ├── rf_model.pkl
│   └── mlp_model.pkl
├── patterncla.pt              # PatternMatcher CNN 模型
└── CompanyConfig/             # Agent 配置
```

### 相关代码引用

| 数据类型 | 代码位置 | 期望路径 |
|---|---|---|
| Trace 异常检测 | `run.py:150-154` | `{date}_trace_ano/data/`, `{date}_trace_ano/{date}.pkl`, `{date}_trace_ano/normal_datasets` |
| Trace 故障定位 | `run.py:166-170` | `{date}_tracerca/{time}*.pkl` |
| Metric 异常描述 | `run.py:193-198` | `{date}_metric_fault/{time}*/{service}/` |
| MicroCause 定位 | `run.py:211-224` | `{date}_microcause/{time}*.xlsx` |
| Log 故障分析 | `run.py:248-264` | `{date}_log_fault/{time}*/{service}*.csv` |
| MEPFL 模型 | `mepfl.py:99-103` | `./mepfl_model_gaia/rf_model.pkl`, `./mepfl_model_gaia/mlp_model.pkl` |
| PatternMatcher | `run.py:190-191` | `patterncla.pt` |

## 三、GAIA 原始数据集目录结构

GAIA 数据集（下载自 https://github.com/CloudWise-OpenSource/GAIA-DataSet）的**原始**结构：

```
gaia/
├── MicroSS/
│   ├── metric/     → CSV: {service}_{ip}_{metric}_{daterange}.csv （按服务×指标×时间段）
│   ├── trace/      → CSV: trace_table_{service}_2021-07.csv （按服务×月份的追踪数据）
│   ├── business/   → CSV: business_table_*.csv （业务日志）
│   └── run/        → CSV: run_table_2021-07.csv, run_table_2021-08.csv （故障注入记录）
└── Companion_Data/
    ├── log/                    → 日志解析/异常检测/NER 数据
    ├── metric_detection/       → 指标异常检测标注数据
    └── metric_forecast/        → 指标预测数据
```

### GAIA Trace 数据字段

| 字段 | 含义 |
|---|---|
| timestamp | 时间记录 `YYYY-MM-DD hh:mm:ss` |
| host_ip | 运行服务的主机 IP |
| service_name | 服务名称 |
| trace_id | 业务追踪 UUID |
| span_id | 当前追踪中的节点 UUID |
| parent_id | 父节点 UUID |
| start_time | 调用创建时间 |
| end_time | 调用关闭时间 |
| url | RPC 调用地址 |
| status_code | 200 为正常，其他为异常 |
| message | 带外消息 |

## 四、不匹配的根本原因 —— 不是代码有 Bug，是缺少完整的数据预处理管道

**核心结论：代码没有 Bug，但项目发布不完整。** 存在以下关键断层：

### 4.1 原始数据 ≠ 运行时数据：中间缺少完整的预处理管道

`run.py` 期望的数据是经过**深度预处理**后的中间产物，而非 GAIA 原始数据。虽然 `data/` 目录提供了部分预处理代码，但**不完整**：

| 预处理步骤 | 是否有代码 | 代码位置 |
|---|---|---|
| Trace 按天拆分（`trace_by_day/{date}.csv`） | ❌ 缺失 | — |
| Trace 异常检测数据（`{MMDD}_trace_ano/`） | ❌ 缺失 | — |
| Trace 故障定位数据（`{MMDD}_tracerca/`） | ⚠️ 部分 | `data/parse_fault_injection.py` |
| Metric 故障数据（`{MMDD}_metric_fault/`） | ❌ 缺失 | — |
| MicroCause 输入数据（`{MMDD}_microcause/`） | ❌ 缺失 | — |
| Log 故障数据（`{MMDD}_log_fault/`） | ❌ 缺失 | — |
| MEPFL 模型（`mepfl_model_gaia/`） | ❌ 缺失 | — |
| PatternMatcher 模型（`patterncla.pt`） | ❌ 缺失 | — |

### 4.2 Trace 数据组织方式不同

- **GAIA 原始**: `trace_table_{service}_2021-07.csv`（**按服务分**，一个月一个文件）
- **代码期望**: `trace_by_day/{date}.csv`（**按日期分**，一天一个文件，合并所有服务）

`data/parse_fault_injection.py` 第 53 行明确引用了 `./Datasets/GAIA/trace_by_day/{date}.csv`，但 GAIA 原始数据没有这个 `trace_by_day/` 目录——需要先做一个按天合并所有服务的预处理步骤。

### 4.3 日期范围差异

- **GAIA 数据**: 2021-07 和 2021-08
- **run.py 默认任务**: `2022/05/03`（仅作示例，实际可通过 `--task` 参数传入 GAIA 的日期）

### 4.4 `data/parse_data.py` 引用了不存在的上游路径

`parse_data.py` 读取 `test_normal.pkl` 和 `abnormal.pkl`，这些文件本身也需要从原始 GAIA trace 数据预处理而来，且输出路径是 `../modelcoder/Datasets/GAIA/`——这似乎是作者开发时使用的另一个项目目录结构。

### 4.5 缺失模型文件

`run.py` 依赖的预训练模型也不在仓库中：
- `patterncla.pt`（PatternMatcher CNN 模型）
- `mepfl_model_gaia/rf_model.pkl` 和 `mepfl_model_gaia/mlp_model.pkl`

## 五、总结

这不是代码 Bug，而是**开源项目发布不完整**：

1. **`data/` 目录只包含预处理管道的一部分**（主要是 trace 相关），缺少 metric、log、microcause 的预处理代码
2. **缺少从 GAIA 原始格式到代码期望格式的转换脚本**（如 `trace_by_service` → `trace_by_day`）
3. **缺少预训练模型文件**（`patterncla.pt`、`mepfl_model_gaia/*.pkl`）
4. **缺少已预处理好的中间数据目录**（`{MMDD}_trace_ano/`、`{MMDD}_tracerca/`、`{MMDD}_metric_fault/`、`{MMDD}_microcause/`、`{MMDD}_log_fault/`）

这实际上是学术界开源项目的常见情况——论文关注的是方法框架而非完整的数据工程管道，因此很多预处理步骤和模型文件没有包含在开源仓库中。

## 六、Metric 相关代码检查结果

### 6.1 ✅ 异常模式分类模型：PatternMatcher（CNN）

**存在，代码完整。** 位于 `metric_anomaly.py`：

| 组件 | 文件:行号 | 说明 |
|---|---|---|
| `class CNN(nn.Module)` | `metric_anomaly.py:83` | 3层1D-CNN + 全连接 + LogSoftmax |
| `class CNNClassifier` | `metric_anomaly.py:128` | 分类器封装（训练/预测/保存/加载） |
| 11种异常模式定义 | `metric_anomaly.py:275` | `an_type=['Level shift up', 'Level shift down', 'Steady increase', ...]` |
| `generate_metric_describe()` | `metric_anomaly.py:274` | 生成指标异常的自然语言描述 |
| 调用入口 | `run.py:190-198` | 加载模型 → 按服务筛选 → 生成描述 |

**⚠️ 缺失**：预训练模型文件 `patterncla.pt`（`run.py:191` 引用但文件不存在）

### 6.2 ✅ MicroCause 故障定位模块

**存在，代码完整。** 位于 `micro.py`：

| 函数 | 文件:行号 | 对应论文描述 |
|---|---|---|
| `run_SPOT()` | `micro.py:28` | SPOT/dSPOT 异常检测算法 |
| `get_eta()` | `micro.py:42` | 计算异常偏离度 η |
| `run_pcmci()` | `micro.py:55` | PCMCI 因果发现（PCTS 方法） |
| `get_links()` | `micro.py:237` | 从 PCMCI 结果提取因果图 |
| `get_Q_matrix_part_corr()` | `micro.py:93` | 基于偏相关构建转移概率矩阵 Q |
| `randomwalk_metric()` | `micro.py:177` | 时序因果随机游走 |
| `get_gamma()` | `micro.py:208` | 综合 vis_time 和 η 计算得分 γ |
| `root_kpi()` | `micro.py:225` | 输出 Top-5 根因指标 |
| 调用入口 | `run.py:226-233` | 完整的 MicroCause 流水线调用 |

依赖库 `SPOT/spot.py` 包含完整的 dSPOT 算法实现。

**⚠️ 缺失**：输入数据文件 `{MMDD}_microcause/{time}*.xlsx`（Excel 格式，由 `util_funcs/loaddata.py` 的 `load()` 函数读取）

### 6.3 ⚠️ 3-sigma 异常检测

**存在，但仅用于 trace 而非 metric。** 位于 `trace_anomaly.py:263-285`：

```python
std_devs = np.std(selected_columns, axis=1, ddof=1)
if value_ab[k]>(means[k]+3*std_devs[k]) or value_ab[k]<(means[k]-3*std_devs[k]):
```

这是 trace 异常检测中的 3-sigma 逻辑，用于判断 trace 调用耗时是否异常。论文中提到的 **metric 3-sigma 异常检测** 在代码中**并未独立实现为一个函数**——它的功能实际上被 MicroCause 的 SPOT 算法（`micro.py:28`）和 PatternMatcher 的 CNN 分类器（`metric_anomaly.py:83`）覆盖了。

具体而言：
- `run_SPOT()` 执行了 metric 级别的异常检测（基于统计过程控制，比 3-sigma 更精细）
- `generate_metric_describe()` 中的 `score = abs((value-ymean)/ystd)/2`（`metric_anomaly.py:317`）本质上就是一个 sigma-score 计算

### 6.4 ✅ Metric Expert LLM Agent 配置

**存在，配置完整。** 位于 CompanyConfig：

| 配置文件 | 内容 |
|---|---|
| `PhaseConfig.json` | `MetricAnalysis` 阶段定义，含 Observation/Reasoning/Final Answer 格式 |
| `RoleConfig.json` | `Metric Analysis Expert` 角色定义，含角色 prompt 和示例 |
| `ChatChainConfig.json` | `MetricAnalysis` 在 ChatChain 中的位置 |
| 动态写入 | `run.py:236-242` | 将异常描述和根因指标注入到 PhaseConfig 的 prompt 中 |

### 6.5 总结

| 论文描述的模块 | 代码是否存在 | 文件位置 | 缺失项 |
|---|---|---|---|
| 3-sigma 异常检测 | ⚠️ 部分 | 被 SPOT + CNN 覆盖 | — |
| PatternMatcher (CNN) | ✅ 完整 | `metric_anomaly.py` | `patterncla.pt` 模型文件 |
| MicroCause (PCMCI+RandomWalk) | ✅ 完整 | `micro.py` + `SPOT/spot.py` | 输入 `.xlsx` 数据文件 |
| Metric Expert Agent | ✅ 完整 | `CompanyConfig/` 配置 | — |

**结论**：metric 处理流水线的**所有算法代码都已存在且完整**，缺失的仅是：
1. 预训练模型文件 `patterncla.pt`
2. 预处理后的输入数据文件（`.xlsx` 格式的 MicroCause 输入、`_metric_fault/` 目录下的指标 CSV）

## 七、PatternMatcher 与 `patterncla.pt` 来源分析

### 7.1 CNN 架构与 AlexNet 的关系

代码中的 CNN 架构（`metric_anomaly.py:83-125`）与 AlexNet（Krizhevsky et al., 2012 "ImageNet Classification with Deep Convolutional Neural Networks"）逐层对照：

**代码中的 CNN**：
```
输入: [batch, 1, 30]   ← 1通道, 长度30的一维时间序列

Conv1: Conv1d(1→64, kernel=5, stride=1, padding=1) → ReLU → MaxPool(2)   → [batch, 64, 15]
Conv2: Conv1d(64→128, kernel=5, stride=1, padding=1) → ReLU → MaxPool(2)  → [batch, 128, 7]
Conv3: Conv1d(128→256, kernel=5, stride=1, padding=1) → ReLU → MaxPool(2) → [batch, 256, 3]

Flatten → Dropout(0.5)
FC1: Linear(512→64) → BatchNorm → ReLU
FC2: Linear(64→classes) → LogSoftmax
```

**原始 AlexNet（2012, Image Classification）**：
```
输入: [batch, 3, 224×224]   ← 3通道 RGB 图像

Conv1: Conv2d(3→96, 11×11, stride=4) → ReLU → MaxPool(3×3, stride=2)
Conv2: Conv2d(96→256, 5×5) → ReLU → MaxPool(3×3, stride=2)
Conv3: Conv2d(256→384, 3×3) → ReLU
Conv4: Conv2d(384→384, 3×3) → ReLU
Conv5: Conv2d(384→256, 3×3) → ReLU → MaxPool(3×3, stride=2)

Flatten → Dropout(0.5)
FC1: 4096 → ReLU → Dropout
FC2: 4096 → ReLU → Dropout
FC3: N_classes → Softmax
```

**对照结论**：这是 AlexNet 的 **1-D 简化改编版**，借鉴了 AlexNet 的核心设计范式：

| 设计特征 | AlexNet（2D 图像） | 本代码（1D 时间序列） |
|---|---|---|
| 卷积→激活→池化堆叠 | ✅ 5层 Conv | ✅ 3层 Conv（简化） |
| 通道数递增模式 | 96→256→384→384→256 | 64→128→256 |
| 全连接层 + Dropout | FC-FC-FC + Dropout(0.5) | FC-FC + Dropout(0.5) |
| 最终 Softmax 输出 | ✅ | ✅ LogSoftmax |
| 输入维度 | 3通道 224×224 图像 | 1通道 长度30 时间序列 |

但本质区别在于：这是**从零训练**的，不是迁移学习——AlexNet 的 ImageNet 预训练权重与 1-D 时间序列任务完全不兼容（维度、通道数、语义完全不同）。

### 7.2 `patterncla.pt` 的生成过程

代码中给出了完整的答案。`CNNClassifier` 类包含完整的训练和保存流程：

| 方法 | 文件:行号 | 功能 |
|---|---|---|
| `CNNClassifier.fit(X, y)` | `metric_anomaly.py:142` | 训练模型 |
| `CNNClassifier.save_model(path)` | `metric_anomaly.py:253` | 保存为 `.pt` 文件 |
| `CNNClassifier.load_model(path)` | `metric_anomaly.py:263` | 加载 `.pt` 文件 |

训练过程的详细参数（`metric_anomaly.py:142-228`）：

| 参数 | 值 | 说明 |
|---|---|---|
| 输入 X | `np.ndarray` | 多条长度为 30 的时间序列片段 |
| 输入 y | `np.array` | 对应的 11 类异常模式标签 |
| 数据划分 | 70% / 30% | 训练 / 验证 |
| 优化器 | Adam (lr=1e-4) | `BATCH_SIZE = 32` |
| 损失函数 | CrossEntropyLoss | 多分类交叉熵 |
| 学习率调度 | StepLR (step=50, gamma=0.8) | 每 50 epoch 衰减 0.8 |
| 早停策略 | 连续 10 epoch 验证 F1 无提升 | `stop_epoch=10` |
| 模型选择 | 验证集 macro F1 最高者 | `best_model` |

`patterncla.pt` 就是**用这个 `fit()` 方法在标注好的指标异常模式数据集上训练后，通过 `save_model("patterncla")` 保存出来的**。

### 7.3 模型文件缺失的原因

训练链路的缺失环节：

```
标注好的训练数据 (X: 时间序列片段, y: 11类标签)
    ↓  ❌ 缺失: 没有标注数据集
CNNClassifier.fit(X, y)
    ↓  ✅ 代码存在: metric_anomaly.py:142
best_model (验证集最优)
    ↓
CNNClassifier.save_model("patterncla")
    ↓  ❌ 缺失: .pt 文件未包含在仓库中
patterncla.pt
    ↓
run.py → CNNClassifier(class_num=11).load_model("patterncla.pt")
```

论文中引用了 PatternMatcher 的原始出处（参考文献 [36]），说明这个 CNN 架构和训练方法来自前人工作。`patterncla.pt` 是作者在自己的 GAIA 指标数据上训练得到的，但由于**标注数据未公开** + **模型文件未上传**，所以仓库中缺少这个 `.pt` 文件。

### 7.4 复现 `patterncla.pt` 的方法

如果要重新生成 `patterncla.pt`，需要：

1. **准备标注数据**：从 GAIA 指标数据中截取 30 点长度的时间序列片段，人工标注为以下 11 类异常模式之一：
   ```
   ['Level shift up', 'Level shift down', 'Steady increase', 'Steady decrease',
    'Single spike', 'Single dip', 'Transient level shift up', 'Transient level shift down',
    'Multiple spikes', 'Multiple dips', 'Fluctuations']
   ```
2. **调用训练代码**：
   ```python
   from metric_anomaly import CNNClassifier
   clf = CNNClassifier(class_num=11)
   clf.fit(X_train, y_train)      # X: shape [N, 30], y: shape [N] (0-10)
   clf.save_model("patterncla")   # 生成 patterncla.pt
   ```

**总结**：`patterncla.pt` 是作者在自有标注数据上用代码中已有的 `CNNClassifier.fit()` 从零训练出来的，不是来自任何公开预训练模型。缺失原因是标注训练数据和模型文件都未包含在开源发布中。

## 八、后续建议

由于数据从原始 GAIA 到 `run.py` 期望的格式之间有巨大的格式转换鸿沟（需要按天合并 trace、按故障注入时间窗口切片、生成 pkl/xlsx 中间文件等），简单的软连接**无法解决问题**。真正需要的是补全缺失的预处理管道代码。

如果要使项目完整可运行，需要：

1. **编写 trace 按天合并脚本**：将 10 个 `trace_table_{service}_*.csv` 合并为按天的 CSV（生成 `trace_by_day/{date}.csv`）
2. **补全预处理代码**：为 `{MMDD}_trace_ano/`、`{MMDD}_metric_fault/`、`{MMDD}_microcause/`、`{MMDD}_log_fault/` 编写数据转换脚本
3. **训练或获取模型**：MEPFL 模型（RF + MLP）和 PatternMatcher 模型（CNN）
4. **或者联系作者**：获取已预处理好的数据和模型文件
