# 评估模块使用说明

> 对应论文：*LLM-Enhanced Failure Localization in Microservices: Integrating Multi-Modal Data and Expert Interpretation* (Paper [171]) Section V

## 一、论文评估指标体系

论文在 Section V-A4 定义了以下三类评估指标：

### 1. Failure Localization（根因定位准确率）

| 指标 | 定义 | 论文位置 |
|------|------|----------|
| **A@k** (Top-k Accuracy) | 根因是否出现在 top-k 预测结果中。报告 k=1, 3, 5 | Table II |

### 2. Reasoning Quality（推理质量）

| 指标 | 定义 | 说明 | 论文位置 |
|------|------|------|----------|
| **BLEU-4** | 生成文本与参考文本的 4-gram 精确率 | 侧重精确率（precision） | Tables III-V |
| **ROUGE-L** | 最长公共子序列召回率 | 侧重召回率（recall） | Tables III-V |
| **G-sim** | LLM-as-Judge 语义相似度（0-1） | 使用 GPT-4（本项目默认用 DeepSeek）评估生成推理的清晰度和一致性 | Tables III-V |
| **W-rate** | 人类工程师投票胜率 | 3 位工程师独立投票，胜率 = 获多数票案例比例 | Tables III-V |

BLEU-4 和 ROUGE-L 是词法层面的文本相似度指标，主要反映生成文本与参考文本的字面匹配程度。由于词法指标无法捕捉语义差异，论文额外引入了 G-sim（LLM 评估）和 W-rate（人工评估）来衡量语义层面的推理质量。

### 3. Performance（性能）

| 指标 | 定义 | 论文位置 |
|------|------|----------|
| **Latency** | 平均每案例端到端时延（秒） | Table VI |

---

## 二、工程中原有实现

项目中已存在的评估代码：

| 文件 | 内容 | 状态 |
|------|------|------|
| `util_funcs/evaluation_function.py` | `prCal()`, `pr_stat()`, `print_prk_acc()`, `my_acc()` — PR@k 和自定义 Accuracy | ✅ 存在，但未在 `run.py` 中调用 |
| `micro.py` | `evaluate(gamma)`, `root_kpi()` — MicroCause 内部评估 | ✅ 存在 |
| `metric_anomaly.py` | CNN 训练过程中的 Precision / Recall / F1 | ✅ 仅训练时使用 |

**缺少的指标**：BLEU-4、ROUGE-L、G-sim、W-rate、端到端 Latency 统计、A@k 的批量评估与汇总。

---

## 三、新建评估模块

在 `evaluation/` 目录下实现了论文中的全部评估指标：

```
evaluation/
├── __init__.py           # 包入口，导出核心类
├── metrics.py            # 6 个指标计算函数
├── log_parser.py         # 解析 run.py 日志，提取结构化评估数据
├── evaluator.py          # Evaluator 类：单案例/批量评估，生成 EvaluationReport
├── run_evaluation.py     # CLI 入口脚本
└── README.md             # 本文档
```

### 各文件功能

| 文件 | 核心函数/类 | 说明 |
|------|-------------|------|
| `metrics.py` | `topk_accuracy()`, `topk_accuracy_batch()` | A@1, A@3, A@5 |
| | `bleu4_score()`, `bleu4_batch()` | BLEU-4（4-gram precision） |
| | `rouge_l_score()`, `rouge_l_batch()` | ROUGE-L（LCS-based F1） |
| | `gpt_similarity()`, `gpt_similarity_batch()` | G-sim（LLM-as-Judge, 调用 OpenAI 兼容 API） |
| | `win_rate()` | W-rate（人工投票胜率） |
| | `mean_latency()` | Latency 统计（mean/median/std/min/max） |
| `log_parser.py` | `LogParser` | 解析日志文件，提取 `[EVAL]` JSON 记录 + 标准日志行中的预测结果、phase 时延等 |
| | `load_ground_truth_services()` | 从 `fault_injection_list_*.pkl` 加载 ground truth 服务名 |
| | `parse_batch_logs()` | 批量解析目录下多个日志文件 |
| `evaluator.py` | `Evaluator` | 编排评估流程，支持单案例和批量模式 |
| | `EvaluationReport` | 聚合报告，包含 Table II-VI 格式的汇总，支持 `summary()` 文本输出、JSON/Markdown 保存 |
| `run_evaluation.py` | CLI | 命令行入口，支持 `--log`、`--log-dir`、`--gsim` 等参数 |

---

## 四、运行时埋点

为了从运行时日志中提取评估数据，在以下文件中添加了 `[EVAL]` 结构化日志输出。

### 4.1 `run.py` 中的埋点

添加了 `_eval_log()` 辅助函数，用于输出 `[EVAL]` 开头的 JSON 行：

| 位置 | 输出内容 |
|------|----------|
| 任务解析后 | `{"type": "task_parsed", "date": "0701", "time": "11-50"}` |
| MEPFL 预测后（Phase 2） | `{"type": "localization", "predictions": ["svc1", ...]}` |
| 单变量根因指标后（Phase 5A） | `{"type": "root_metrics", "channel": "univariate", "metrics": "..."}` |
| 多变量根因指标后（Phase 5B） | `{"type": "root_metrics", "channel": "multivariate", "metrics": "..."}` |
| ChatChain 完成后（Phase 9） | `{"type": "e2e_latency", "duration_s": 277.1, ...}` |

### 4.2 `chatops/phase.py` 中的埋点

在每个 Phase 执行完成后（`execute()` 方法末尾）添加：

```
[EVAL] {"type": "agent_output", "agent": "TraceAnalysis", "text": "The trace analysis reveals..."}
```

这会自动捕获 TraceExpert、MetricExpert、LogExpert、RootCauseExpert 四个 agent 的推理输出文本。

### 4.3 日志行格式

`[EVAL]` 行使用严格格式：

```
[EVAL] {"type": "...", "key": "value", ...}
```

`log_parser.py` 通过正则 `r'\[EVAL\]\s*(\{.*\})\s*$'` 精确匹配，不会与其他日志内容混淆。

---

## 五、日志生成方式

### 5.1 生成方法

日志通过 `nohup` 重定向生成，与 `experiments_a.sh` 的运行方式一致：

```bash
nohup bash experiments_a.sh > experiments_a.log 2>&1 &
```

`experiments_a.sh` 内部调用 `python run.py --task '...' --name test_0701 ...`，`run.py` 的所有 `print()` 和 `logger.info()` 输出（包括 `[EVAL]` 行）都会被重定向到日志文件。

### 5.2 评估数据的两种来源

| 来源 | 包含的数据 | 是否需要重跑 |
|------|-----------|-------------|
| **标准日志行** | date/time、root_services、phase_latencies、conflict_scenario、trace/metric 异常数、causal graph 规模 | ❌ 现有日志可直接解析 |
| **`[EVAL]` 结构化行** | agent_output（各专家推理文本）、e2e_latency、ground_truth | ✅ 需要用新版代码重跑 |

### 5.3 已有日志验证

使用现有的 `experiments_a.log`（重定向日志，尚未包含 `[EVAL]` 行）测试解析器：

```
Date: 0701
Time: 11-50
Root services: ['webservice1', 'webservice2', 'redisservice2', 'redisservice1', 'mobservice1']
Phase latencies: {'Phase 1': 9.6, 'Phase 2': 49.1, 'Phase 3': 0.4, 'Phase 4': 95.3, ...}
E2E latency: 277.1s
Conflict scenario: multi_anom_single_normal
Causal graph: 80 nodes, 1074 edges
```

解析器能够从标准日志行中提取根因预测、phase 时延等关键数据，无需 `[EVAL]` 行也可工作。`[EVAL]` 行提供了额外的 agent 推理文本等数据，用于 BLEU-4、ROUGE-L、G-sim 等推理质量评估。

---

## 六、使用方式

### 6.1 单案例评估

```bash
# 仅 A@k + Latency（无需重跑，使用现有日志）
python -m evaluation.run_evaluation \
    --log experiments_a.log \
    --gt mobservice1

# 指定 ground truth 服务名
python -m evaluation.run_evaluation \
    --log experiments_a.log \
    --gt dbservice1
```

### 6.2 批量评估

```bash
# 从 pkl 文件加载 ground truth，评估 Report/ 下所有日志
python -m evaluation.run_evaluation \
    --log-dir Report/ \
    --gt-pkl-dir Datasets/GAIA/fault_injection_tracerank/ \
    --output evaluation_results/

# 带 G-sim 计算（需要 LLM API）
python -m evaluation.run_evaluation \
    --log-dir Report/ \
    --gt-pkl-dir Datasets/GAIA/fault_injection_tracerank/ \
    --gsim --gsim-model deepseek-r1-0528 \
    --output evaluation_results/
```

### 6.3 Python API 调用

```python
from evaluation import Evaluator

# 单案例
ev = Evaluator(log_path="experiments_a.log", ground_truth_service="dbservice1")
report = ev.evaluate()
print(report.summary())

# 批量
ev = Evaluator()
report = ev.evaluate_batch(
    log_dir="Report/",
    ground_truth_pkl_dir="Datasets/GAIA/fault_injection_tracerank/",
    compute_gsim=True,
)
report.save_json("evaluation_results/report.json")
report.save_markdown("evaluation_results/report.md")
```

### 6.4 输出格式

评估报告包含与论文 Table II-VI 对齐的汇总：

```
======================================================================
  LocaleXpert Evaluation Report (Paper [171] Metrics)
======================================================================
  Dataset:  GAIA
  Model:    deepseek-r1-0528
  Method:   LocaleXpert
  Cases:    10

----------------------------------------------------------------------
  Table II: Failure Localization Performance
----------------------------------------------------------------------
  Metric         Value
  ------         -----
  A@1           0.4330
  A@3           0.9377
  A@5           0.9643

----------------------------------------------------------------------
  Table III: Trace Expert Reasoning Performance
----------------------------------------------------------------------
  Metric         Value
  ------         -----
  BLEU-4        0.0448
  ROUGE-L       0.3479
  G-sim         0.8125
  W-rate        0.3848

----------------------------------------------------------------------
  Table VI: Latency
----------------------------------------------------------------------
  Mean latency:     45.2s
  ...
```

---

## 七、指标计算细节

### A@k（Top-k Accuracy）

- 将 MEPFL 预测的 top-k 服务列表与 ground truth 比较
- Ground truth 来源：`fault_injection_list_*.pkl` 中的 `service` 字段
- 比较时做模糊匹配（大小写无关、子串包含）

### BLEU-4

- 标准实现：BP × exp(mean(log(p_n))) for n=1..4
- 比较 agent 推理文本 vs 人工撰写的参考推理文本
- 需要提供 `--reference-file`（JSON 格式，key 为 agent 名称）

### ROUGE-L

- 基于 LCS（最长公共子序列）的 F1 度量
- β=1.2，更重视召回率

### G-sim

- 调用 LLM API，让模型对生成推理 vs 参考推理打分（0-1）
- 默认使用项目已配置的 DeepSeek API
- 可通过 `--gsim-model` 切换模型

### W-rate

- 需要人工评估数据（3 位工程师投票）
- 通过 `metrics.win_rate(votes, method_name)` 计算
- 本模块仅提供计算框架，人工数据需单独收集
