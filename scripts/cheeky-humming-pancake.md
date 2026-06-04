# Plan: GAIA Data Preprocessing Pipeline for LocaleXpert

## Context

LocaleXpert 的 `run.py` 期望读取经过深度预处理的数据目录，但 GAIA 原始数据集的格式与之不匹配。需要编写 5 个脚本补全整个数据预处理管道。**目标是把 Python 脚本写好、逻辑正确即可，不需要实际跑通数据处理。**

## 目录

所有新脚本放置在 `scripts/` 下：
- `scripts/__init__.py`
- `scripts/step1_trace_merge.py`
- `scripts/step2_fault_injection.py`
- `scripts/step3_train_mepfl.py`
- `scripts/step4_prepare_runtime_data.py`
- `scripts/step5_train_patternmatcher.py`

## 执行顺序

step1 → step2 → (step3 + step4 可并行) ‖ step5（独立）

---

## Step 1: `scripts/step1_trace_merge.py`

**目的**: 合并 10 个按服务分的 trace CSV 为按天的 CSV + 建立 `Datasets/GAIA/` 目录结构。

**CLI**: `--gaia-root`（GAIA MicroSS 路径）, `--output-root`（默认 `./Datasets/GAIA`）, `--dates`（可选，指定处理哪些天，如 `2021-07-01`）

**核心函数**:
- `setup_directories(output_root)` — 创建 `trace_by_day/`、`run/`、`fault_injection_tracerank/` 目录，复制/软链 run_table
- `merge_traces_for_date(date_str, trace_dir, output_dir)` — 流式读取 10 个 trace CSV，按 timestamp 过滤出当天行，排序后写出 `{YYYY-MM-DD}.csv`
- `get_fault_dates(run_csv)` — 从 run_table 提取有故障注入的日期列表
- `main()` — CLI 入口

**输出**:
- `Datasets/GAIA/trace_by_day/{YYYY-MM-DD}.csv`（columns 与 GAIA trace 一致）
- `Datasets/GAIA/run/run_table_2021-07.csv`

**复用**: 无外部依赖，直接读 GAIA 原始格式。

---

## Step 2: `scripts/step2_fault_injection.py`

**目的**: 改编 `data/parse_fault_injection.py`，生成故障注入 pkl + 正常 trace。

**CLI**: `--datasets-root`, `--gaia-trace-root`, `--cleanup`（默认 True，处理完当天 CSV 后删除）

**核心函数**:
- `parse_run_table(run_csv)` — 解析 run_table，提取故障注入事件（过滤 INFO/ERROR 行），返回 `fault_injection_dict`（按日期分组）
- `process_date(date_str, fault_injection_list, trace_dir, datasets_root, cleanup)` — 对单天：调用 step1 生成当天 CSV → 构建 `seconds_to_row` → 提取正常 trace（去掉故障前后 300s）→ 提取故障 trace（故障前后 180s）→ 保存 pkl → 可选删除当天 CSV
- `main()` — CLI 入口，遍历所有故障日期

**pkl 结构**: `list[dict]`，每个 dict 含 `{date: datetime, service: str, time: datetime, trace_list: list[dict]}`（raw CSV 行 dicts）

**输出**:
- `Datasets/GAIA/fault_injection_tracerank/fault_injection_list_{date}.pkl`
- `Datasets/GAIA/trace_by_day/normal.csv`

**复用**: 改编 `data/parse_fault_injection.py` 的解析逻辑。

---

## Step 3: `scripts/step3_train_mepfl.py`

**目的**: 训练 MEPFL 的 RandomForest + MLPClassifier 模型。

**CLI**: `--datasets-root`, `--output-dir`（默认 `./mepfl_model_gaia`）, `--train-days`（默认 5）

**核心函数**:
- `build_traces_from_rows(rows)` — **关键**：将 raw CSV 行 dicts 转换为 `data_models.Trace`/`Span` 对象树
  - 按 trace_id 分组
  - 构建 Span（span_id, parent_span_id, start_time, duration=(end-start)*1000, service_name, status_code）
  - 通过 parent_span_id 建立 children_span_list
  - 构建 Trace（root_span 为 parent 为 None 的 span）
- `get_trace_list_vector(trace_list)` — 复用 `mepfl.py` 的逻辑，提取 32 维特征向量
- `main()` — 加载 pkl → 转换 Trace/Span → 标注 anomaly_type → 训练 RF + MLP → 保存

**训练参数**（与 `mepfl_model/main.py` 一致）:
- RF: `RandomForestClassifier(random_state=0)`
- MLP: `MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)`
- 标注: 故障注入 60s 内且包含故障服务的 trace → anomaly_type=1

**输出**: `mepfl_model_gaia/rf_model.pkl`, `mepfl_model_gaia/mlp_model.pkl`

**复用**: `data/data_models.py`（Span/Trace）, `mepfl.py`（`total_service_list`, `get_trace_list_vector`）

---

## Step 4: `scripts/step4_prepare_runtime_data.py`

**目的**: 为每个故障注入事件创建 run.py 期望的 5 类目录。这是最复杂的脚本。

**CLI**: `--datasets-root`, `--gaia-root`, `--dates`

### 4a: `{MMDD}_tracerca/`
- `prepare_tracerca_dir()` — 按 HH-MM 分组故障注入，将 trace_list 转换为 Trace/Span 对象（复用 step3 的 `build_traces_from_rows`），保存为 `{MMDD}_tracerca/{HH-MM}.pkl`

### 4b: `{MMDD}_trace_ano/`
- `prepare_trace_ano_dir()` — 三部分输出:
  - `{MMDD}.pkl`: 从正常 trace 构建 `call_path_dict`（调用 `trace_anomaly.py` 的 `get_trace_from_gaia()` + `walk_call_path()`）
  - `normal_datasets`: 正常 trace STV 文本（调用 `trace_to_STV()`），每行 `traceID:v1,v2,...,vN`
  - `data/*.csv`: 每个故障时间点的 trace CSV 片段

### 4c: `{MMDD}_metric_fault/`
- `prepare_metric_fault_dir()` — 从 GAIA metric CSV 中提取故障时间窗前后 30 个采样点
  - 启动时构建指标文件索引（service→kpi→文件路径映射）
  - 计算 normal baseline（mean/std）嵌入文件名
  - 输出: `{MMDD}_metric_fault/{HH-MM}/{service}/{kpi}_{mean}-{std}.csv`（columns: timestamp,value，30 行）

### 4d: `{MMDD}_microcause/`
- `prepare_microcause_dir()` — 从 GAIA metric 构建 30 步多服务指标矩阵，保存为 Excel
  - Sheet1: 每行一个变量（`servicename_kpiname`），第一列名称，其余列时间序列值
  - 输出: `{MMDD}_microcause/microcause_{date}_{HH-MM}.xlsx`

### 4e: `{MMDD}_log_fault/`
- `prepare_log_fault_dir()` — 从 GAIA business 表（webservice1 7月）和 run_table（其他服务 WARNING/ERROR 日志）提取故障时间窗日志
  - 输出: `{MMDD}_log_fault/{HH-MM}/{service}.csv`（至少 3 列，row[2] 含日志文本）

**复用**: `trace_anomaly.py`（call_path 相关函数）, `data/data_models.py`, `openpyxl`

---

## Step 5: `scripts/step5_train_patternmatcher.py`

**目的**: 用合成数据训练 11 类异常模式 CNN 分类器。

**CLI**: `--output-path`（默认 `./patterncla.pt`）, `--samples-per-class`（默认 500）, `--epochs`（默认 100）

**核心函数**:
- `generate_synthetic_data(samples_per_class, length=30)` — 为 11 种模式各生成 N 条合成时间序列（基线+模式信号+随机噪声）
- `main()` — 生成数据 → `CNNClassifier(class_num=11).fit(X, y)` → `save_model()`

**11 种模式的合成公式**:
- Level shift up/down: `baseline + step_at_t * direction`
- Steady increase/decrease: `baseline + slope * t`
- Single spike/dip: `baseline + peak * delta(t0)`
- Transient level shift up/down: `baseline + shift * window(t0, t1)`
- Multiple spikes/dips: `baseline + sum(peaks)`
- Fluctuations: `baseline + A * sin(freq * t + phase)`

**复用**: 直接 import `metric_anomaly.py` 的 `CNNClassifier`

---

## 关键复用文件清单

| 现有文件 | 被哪些脚本复用 | 复用内容 |
|---|---|---|
| `data/data_models.py` | step3, step4 | Span、Trace 数据类 |
| `data/read_data.py` | step3 | `get_trace_list_from_rows()` |
| `data/parse_fault_injection.py` | step2 | 故障注入解析逻辑（改编） |
| `mepfl.py` | step3 | `get_trace_list_vector()`, `total_service_list` |
| `mepfl_model/main.py` | step3 | 训练循环结构（改编） |
| `trace_anomaly.py` | step4 | `get_trace_from_gaia()`, `walk_call_path()`, `trace_to_STV()` |
| `metric_anomaly.py` | step5 | `CNNClassifier` |

## 验证方式

脚本写好后，通过以下方式验证正确性:
1. 每个脚本的 `--help` 能正常输出
2. step5 可直接运行（合成数据不依赖 GAIA），验证 `patterncla.pt` 生成
3. step1 对单天 `--dates 2021-07-01` 可运行（流式处理不耗太多磁盘）
4. 代码逻辑审查：pkl 结构、CSV 格式、目录命名与 `run.py` 的读取逻辑一一对应
