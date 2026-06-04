### LocaleXpert + Dual-Channel RCA + Multimodal Root Cause Localization

基于 LocaleXpert（文献 [171]）框架，集成了 **双通道根因分析（Dual-Channel RCA）**、**可配置多变量时间序列异常检测**、LLM 冲突解决模块、以及 **可配置多模态根因定位**。

**异常检测**支持 7 种方法：**TranAD、USAD、OmniAnomaly、MAD_GAN、MSCRED、GDN、MTAD_GAT**，通过 `--anomaly-method` 切换。

**根因定位**支持 2 种策略：**default**（原有 PCMCI+RandomWalk + MEPFL 单模态流水线）和 **tvdig**（TVDiag 多模态 GNN，文献 [167]），通过 `--rca-method` 切换。

### The file structure is as follows:

```
├── /CompanyConfig/
│  └──/SelfIntroduction/
├── /mepfl_model/
├── /data/
├── /anomaly_detection/               # [新增] 可配置多变量异常检测包
│   ├── __init__.py                   #   包入口，导出 create_detector / AVAILABLE_METHODS
│   ├── base_detector.py              #   抽象基类 BaseMultivariateDetector
│   ├── shared.py                     #   共享工具（滑动窗口、指标映射、描述生成）
│   ├── detector_factory.py           #   工厂函数 + 懒加载注册表
│   ├── tranad_detector.py            #   TranAD (VLDB 2022, Transformer)
│   ├── usad_detector.py              #   USAD (KDD 2020, 双自编码器对抗训练)
│   ├── omnianomaly_detector.py       #   OmniAnomaly (KDD 2019, GRU + VAE)
│   ├── mad_gan_detector.py           #   MAD_GAN (ICANN 2019, GAN)
│   ├── mscred_detector.py            #   MSCRED (AAAI 2019, ConvLSTM)
│   ├── gdn_detector.py              #   GDN (AAAI 2021, 图注意力, 纯 PyTorch)
│   └── mtad_gat_detector.py          #   MTAD_GAT (ICDM 2020, 图注意力 + GRU)
├── /failure_localization/            # [新增] 可配置根因定位包
│   ├── __init__.py                   #   包入口，导出 create_localizer / AVAILABLE_METHODS
│   ├── base_localizer.py             #   LocalizationResult 数据类 + BaseLocalizer 抽象基类
│   ├── default_localizer.py          #   Default 策略（封装原有 PCMCI+RW + MEPFL 流水线）
│   ├── tvdig_localizer.py            #   TVDiag 策略（多模态 GNN 推理封装）
│   ├── tvdig_config.py               #   TVDiag 配置（模型维度、训练参数、GAIA 拓扑）
│   ├── tvdig_model.py                #   纯 PyTorch TVDiag 模型（SAGEConv、MainModel、损失函数、训练器）
│   ├── tvdig_data.py                 #   数据适配器（离线训练 + 在线推理特征构建 + 嵌入缓存）
│   ├── localizer_factory.py          #   工厂函数（按名称创建定位器）
│   └── train_tvdig.py                #   TVDiag 独立训练脚本
├── metric_anomaly.py                 # 单变量异常检测 (CNN 模式分类 + SPOT)
├── multivariate_anomaly.py           # 多变量异常检测入口（委托工厂调度）
├── anomaly_conflict_resolver.py      # LLM 冲突仲裁模块
├── trace_anomaly.py
├── mepfl.py
├── micro.py                          # 因果分析 (PCMCI + Random Walk)
├── run.py                            # 主入口 (含可配置异常检测 + 根因定位 + 冲突解决)
├── run_22.py
├── tvdig_checkpoint/                 # [新增] TVDiag 训练产出
│   ├── tvdig.pt                      #   模型权重
│   └── embedding_cache.pkl           #   事件→嵌入向量缓存
├── Multivariate_Anomaly_Integration_Report.md  # TranAD 集成方法与测试报告
```

- `SelfIntroduction`: Prompt for each Agent and solution paths.
- `mepfl_model`: Code for training the trace failure localization model.
  - `mepfl_model\main.py` : Main train program
  - `mepfl_model\data` : Data process code
- `data`: Code for data process.
- `metric_anomaly.py`: Code for metrics description generate, along with the code for training the metric classification model.
- `anomaly_detection/`: **[新增]** 可配置多变量异常检测包。所有检测器遵循统一的 `train()` / `detect()` 接口，通过工厂函数 `create_detector(method, n_features)` 按名称创建。全部为纯 PyTorch 实现，无外部依赖（GDN/MTAD_GAT 用纯 PyTorch 重写了图注意力，无需 DGL）。
- `failure_localization/`: **[新增]** 可配置多模态根因定位包。所有定位器遵循统一的 `localize()` → `LocalizationResult` 接口。支持：
  - `default`：封装原有单模态流水线（PCMCI+RandomWalk + MEPFL），行为完全不变。
  - `tvdig`：基于 TVDiag（文献 [167]）的多模态 GNN 方法，在服务依赖图上联合分析 metric+trace+log 三种模态数据，进行端到端的根因服务定位和故障类型分类。纯 PyTorch 实现（重写了 GraphSAGE 卷积），无需 DGL。
- `multivariate_anomaly.py`: 多变量异常检测入口。通过 `method` 参数选择检测方法（默认 `tranad`），委托 `anomaly_detection` 包执行。**向后兼容**：不传 `method` 时行为与之前完全一致。
- `anomaly_conflict_resolver.py`: LLM 冲突仲裁模块。基于双通道 RCA 排名，当多变量和单变量根因分析结果不一致时，自动分类冲突场景（4种），调用 LLM 进行对比分析，生成统一的异常报告。
- `trace_anomaly.py`: Code for traces description generate.
- `mepfl.py`: Code for online trace failure localization.
- `micro.py`: 因果分析模块 (PCMCI 因果推断 + Random Walk)，作为根因定位的最终裁决器。
- `run.py`: LocaleXpert in GAIA Dataset (含双通道RCA + 可配置异常检测 + 可配置根因定位 + 冲突解决)
- `run_22.py`: LocaleXpert in AIOps Challenge Dataset
- `tvdig_checkpoint/`: TVDiag 模型训练产出，包含模型权重和事件嵌入缓存。

### Install

1. **Set Up Python Environment:** Use the existing conda environment:

   ```
   conda activate /root/shared-nvme/.conda/envs/LocaleXpert_env
   ```

2. **Install Dependencies:** Install the necessary dependencies by running:

   ```
   pip install -r requirements.txt
   ```

3. **Run (default pipeline):** Replace [description_of_task] with the task description and [project_name] with the AIOps case name:

   ```
   python run.py --task "[description_of_task]" --name "[project_name]"
   ```

4. **Run (specify anomaly detection method):**

   ```
   python run.py --task "[description_of_task]" --name "[project_name]" --anomaly-method usad
   python run.py --task "[description_of_task]" --name "[project_name]" --anomaly-method gdn --anomaly-epochs 10
   ```

5. **Run (TVDiag multimodal root cause localization):**

   ```
   python run.py --task "[description_of_task]" --name "[project_name]" \
       --rca-method tvdig --tvdig-model ./tvdig_checkpoint
   ```

6. **Run (skip multivariate detection, original pipeline):**

   ```
   python run.py --task "[description_of_task]" --name "[project_name]" --skip-multivariate
   ```

7. **Train TVDiag model (offline, on GAIA historical data):**

   ```
   python -m failure_localization.train_tvdig \
       --data-dir /root/shared-nvme/work/code/RCA/2026/TVDiag/data/gaia \
       --output-dir ./tvdig_checkpoint --epochs 500
   ```

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | (required) | Task description with datetime |
| `--name` | `DefaultName` | Case name for report output |
| `--model` | `deepseek-r1-0528` | LLM model name. API models: `deepseek-r1-0528`, `GPT_4` 等。本地 Ollama: `ollama-qwen3-14b`, `ollama-qwen3-8b` |
| `--ollama-url` | `http://localhost:11434/v1` | Ollama API 地址（仅当 `--model` 以 `ollama-` 开头时生效） |
| `--anomaly-method` | `tranad` | 多变量异常检测方法，可选: `tranad`, `usad`, `omnianomaly`, `mad_gan`, `mscred`, `gdn`, `mtad_gat` |
| `--anomaly-epochs` | `5` | 多变量异常检测训练轮数 |
| `--anomaly-lr` | (model-specific) | 学习率，不指定则使用各模型默认值 |
| `--anomaly-window` | (model-specific) | 滑动窗口大小，不指定则使用各模型默认值 |
| `--rca-method` | `default` | 根因定位方法: `default`（PCMCI+RW + MEPFL 单模态）或 `tvdig`（TVDiag 多模态 GNN） |
| `--tvdig-model` | `None` | TVDiag 模型检查点目录（`--rca-method=tvdig` 时必需） |
| `--skip-multivariate` | `False` | Skip multivariate detection (original univariate-only pipeline) |

### Supported Anomaly Detection Methods

| Method | 来源 | 核心架构 | 默认窗口 | 默认学习率 |
|--------|------|----------|----------|------------|
| `tranad` | VLDB 2022 | 双阶段自条件 Transformer | 10 | 0.001 |
| `usad` | KDD 2020 | 双自编码器 + 对抗训练 | 5 | 0.0001 |
| `omnianomaly` | KDD 2019 | GRU + VAE (重参数化) | 1 | 0.002 |
| `mad_gan` | ICANN 2019 | GAN (生成器 + 判别器) | 5 | 0.0001 |
| `mscred` | AAAI 2019 | ConvLSTM 编码器 + 反卷积解码器 | (auto) | 0.0001 |
| `gdn` | AAAI 2021 | 多头图注意力网络 | 5 | 0.0001 |
| `mtad_gat` | ICDM 2020 | 双图注意力 (特征+时间) + GRU | (auto) | 0.0001 |

### Supported Root Cause Localization Methods

| Method | 来源 | 核心架构 | 输入模态 | 说明 |
|--------|------|----------|----------|------|
| `default` | LocaleXpert [171] | PCMCI 因果推断 + Random Walk (指标) / RF+MLP (追踪) | 单模态分别处理 | 原有流水线，行为不变 |
| `tvdig` | TVDiag [167] | 多模态 GraphSAGE + 监督对比学习 + 跨模态关联 | Metric + Trace + Log 联合分析 | 在服务依赖图上融合三模态，端到端根因定位 + 故障分类 |

### TVDiag Training Results (GAIA Dataset)

TVDiag 模型在 GAIA 数据集上的离线训练结果（100 epochs，纯 PyTorch 实现，无 DGL 依赖）：

| 指标 | 值 |
|------|-----|
| **RCL HR@1** | 72.4% |
| **RCL HR@3** | 89.4% |
| **RCL HR@5** | 93.8% |
| **RCL MRR@3** | 80.1% |
| **FTI Precision** | 89.4% |
| **FTI Recall** | 90.6% |
| **FTI F1** | 90.0% |

### Key Design Decisions

1. **纯 PyTorch 实现**：TVDiag 的 GraphSAGE 卷积使用 scatter mean + Linear 重写，无 DGL 依赖（DGL 与当前 torch 2.12 不兼容）。
2. **策略模式 (Strategy Pattern)**：`BaseLocalizer` 抽象基类定义 `localize() → LocalizationResult` 接口，由 `DefaultLocalizer` 和 `TVDiagLocalizer` 分别实现。
3. **嵌入缓存 (Embedding Cache)**：离线训练时构建事件→向量映射表，保存在检查点旁，用于在线推理时快速查找。
4. **向后兼容**：`--rca-method default`（或不传该参数）运行原有流水线，行为完全不变。

---

### 双通道 RCA 决策框架（Dual-Channel RCA）

#### 背景：为什么需要双通道？

根据文献分析与讨论，面对单变量和多变量异常检测及根因分析结果冲突时，不应绝对化地只信其中一个。优先级上应以"多变量联合检测"的结果为基座，以"单变量检测"为补充解释，并最终交由"因果推理"进行裁决。

**一、为什么优先相信"多变量联合检测"？**

| 理由 | 说明 |
|------|------|
| **避免"相关性破坏"导致的漏报** | 如文献 [174] 所述，多变量时间序列的核心特征是变量间存在相互依赖。很多微服务故障表现为"相关性破坏"——即单个指标看都在正常阈值内，但它们之间的联动关系打破了常规。此时单变量检测会认为一切正常，而多变量检测能捕捉到这种隐秘的异常 |
| **契合微服务故障的级联传播特性** | 约 60% 以上的微服务故障是跨服务、跨指标的级联影响。文献 [173] 强调，单维方法"易受噪声和隐式依赖影响，限制了根因定位的准确性和鲁棒性" |
| **抗噪声能力更强** | 单变量检测极易受局部毛刺干扰而误报；多变量联合检测综合了多个维度的信息，能够过滤掉仅存在于单一指标上的随机噪声 |

**二、单变量检测的价值何在？**

| 价值 | 说明 |
|------|------|
| **极端显式异常的快速定位** | 当故障是单点故障（如某服务 OOM），单变量检测能最直接地锁定"刺眼"的异常指标，而多变量模型可能因降维或平滑作用削弱了该信号的显著性 |
| **提供根因归因的"候选集"** | 文献 [57] 提到多变量检测到异常后，需要"由低重构概率单变量解释"。即：多变量负责"定性（是否异常）"，单变量负责"归因（哪个指标异常）" |

#### 架构对比：原流程 vs 双通道流程

**原流程（旧版）：**
```
Phase 4:  4A(单变量检测) → 4C(PCMCI+SPOT eta) → 4B(多变量检测)
Phase 5:  单次随机游走（仅用 SPOT eta）→ baseline 排名
Phase 6:  LLM 比较检测结果 → 可能用不同 eta 重跑随机游走
```

**改进后（双通道 RCA）：**
```
Phase 4:  4A(单变量检测+SPOT eta) → 4B(多变量检测+multi eta) → 4C(PCMCI因果图)
Phase 5:  双通道 RCA:
            5A: SPOT eta → 随机游走 → 单变量根因排名
            5B: multi eta → 随机游走 → 多变量根因排名
Phase 6:  LLM 比较两个完整根因排名 → 冲突场景判定 → 因果推理裁决
```

核心区别：原流程只有**一次**随机游走，多变量检测结果仅作为"是否需要调整 eta"的参考。改进后，两种检测方法各自驱动一次**完整的**随机游走，产出独立的根因排名，然后通过 LLM 比较两个排名进行裁决。

#### 完整流水线架构

```
                     Phase 4: 异常检测
                 ┌─────────────────────────┐
                 │  4A: 单变量检测 (CNN+SPOT) │──── SPOT eta
                 │  4B: 多变量检测 (TranAD等) │──── multi eta
                 │  4C: 因果发现 (PCMCI)      │──── 因果图 + Q矩阵
                 └─────────────────────────┘
                              │
                     Phase 5: 双通道 RCA
                 ┌─────────────────────────┐
                 │  5A: SPOT eta + 随机游走  │──→ 单变量根因排名
                 │  5B: multi eta + 随机游走 │──→ 多变量根因排名
                 └─────────────────────────┘
                              │
              Phase 6: LLM 冲突解决 + 因果裁决
                 ┌─────────────────────────┐
                 │  LLM 比较两个根因排名      │
                 │  ↓ 分类冲突场景            │
                 │  一致 → 直接使用            │
                 │  场景1 → 信任多变量RCA     │
                 │  场景2 → 信任多变量+补充   │
                 │  场景3 → 合并+因果裁决     │
                 └─────────────────────────┘
                              │
                     Phase 7-9: 下游推理
              ┌──────────────────────────────┐
              │  Phase 7: 知识注入 PhaseConfig  │
              │  Phase 8: 日志分析              │
              │  Phase 9: ChatChain LLM 推理    │
              └──────────────────────────────┘
```

#### Phase 6 冲突场景决策表

| 场景 | 表现 | 应该信任谁 | 后续动作 |
|------|------|-----------|---------|
| **一致** | 两通道根因排名吻合 | 合并使用 | 排名可靠，直接进入下游 |
| **场景1** | 多变量异常，单变量正常 | 信任多变量 RCA | 典型"相关性破坏"（变量间依赖关系打破，但单个指标都在阈值内），使用多变量 RCA 排名 |
| **场景2** | 单变量异常，多变量正常 | 信任多变量 | 单指标异常大概率是噪声，系统整体正常。但结合业务逻辑判断是否为关键致命指标（如错误率飙升），若是则注入为先验知识 |
| **场景3** | 均报异常，但指向不同根因 | 都不尽信，交给因果推理 | 合并两通道候选集，用平均 eta × boost 向量重跑随机游走。PCMCI 因果图根据故障传播路径和时间滞后顺序倒推源头——**因果推理是最终裁决者** |

#### 终极原则：让"因果推理"做最终裁决

如文献 [112] 和 [173] 所指出的：**异常检测只是第一步，根因定位必须依靠因果推理**。当单变量和多变量检测给出不同结论时：

1. **构建统一输入**：如文献 [141] 和 [171] 所述，将单变量异常分数和多变量异常特征统一为多模态事件表示
2. **因果图筛选**：利用 PCMCI 的时序因果随机游走，根据故障的"传播路径"和"时间滞后顺序"来倒推源头
   - 如果单变量检测到的 A 指标是因，多变量归因的 B 指标是果，RCA 会沿着因果链把 A 定位为根因
   - 如果单变量检测到的 A 只是表象，而多变量检测捕捉到的 B 才是触发 A 的隐式源头，RCA 也会通过拓扑纠偏定位到 B

这也印证了文献 [171] LocaleXpert 的核心思想：**不盲信单一模态，而是让多模态专家在推理层面交叉验证**。
