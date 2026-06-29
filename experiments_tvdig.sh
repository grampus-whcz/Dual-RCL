#!/bin/bash
# TVDiag 多模态 GNN 根因定位 + 双通道 RCA + LLM 推理批量运行脚本
#
# TVDiag vs default 的核心区别：
#
#   | 阶段      | default (PCMCI+RW+MEPFL)              | TVDiag (多模态 GNN)                    |
#   |-----------|---------------------------------------|----------------------------------------|
#   | 异常检测  | 单变量 CNN+SPOT + 多变量 TranAD 等     | 相同（Phase 4 不受 rca-method 影响）   |
#   | 根因定位  | PCMCI 因果图 → Random Walk + MEPFL    | 多模态 GraphSAGE 联合分析              |
#   |           | （指标和 Trace 分别独立定位）           | （metric+trace+log 三模态融合）        |
#   | 输出      | Top-N 根因指标 + Top-N 根因服务        | Top-N 根因服务 + Top-N 根因指标 + 故障类型 |
#   | 知识注入  | 异常描述 + 根因排名 → PhaseConfig      | TVDiag 三模态分析结果 → PhaseConfig    |
#   | LLM 推理  | ChatChain 多 Agent 协作（Phase 7-9）   | 相同                                   |
#
# 关键参数：
#   --rca-method tvdig         使用 TVDiag 多模态 GNN 替代 PCMCI+RW+MEPFL
#   --tvdig-model ./tvdig_checkpoint   TVDiag 模型检查点目录（含 tvdig.pt + embedding_cache.pkl）
#   --anomaly-method tranad    多变量异常检测（与双通道配置相同）
#
# 用法:
#   nohup bash experiments_tvdig.sh > experiments_tvdig.log 2>&1 &
#
# 运行后评估 (自动执行):
#   python -m evaluation.run_evaluation --log experiments_tvdig.log \
#       --gt-pkl-dir Datasets/GAIA/fault_injection_tracerank/ \
#       --generate-references --gsim

set -e

PYTHON=/root/shared-nvme/.conda/envs/LocaleXpert_env/bin/python

# === 配置 ===

# LLM 模型选择:
#   API 模型:  glm-4.5, glm-4.7, deepseek-r1-0528, GPT_4
#   本地模型:  ollama-qwen3-14b, ollama-qwen3-8b (需要 Ollama 服务运行)
MODEL="ollama-qwen3-14b"

# Ollama 地址 (仅本地模型需要)
OLLAMA_URL="http://localhost:11434/v1"

# 多变量异常检测方法 (可选: tranad, usad, omnianomaly, mad_gan, mscred, gdn, mtad_gat)
ANOMALY_METHOD="tranad"

# 多变量检测训练轮数
ANOMALY_EPOCHS=3

# TVDiag 模型检查点目录
TVDIG_MODEL="./tvdig_checkpoint"

# === 可用日期 (只有 0701-0703 有预处理数据) ===
# 如需更多日期，先运行: python scripts/step4_prepare_runtime_data.py --dates 2021-07-04
DATES=(
    "2021-07-01"
)

# === 预检查 ===
echo "============================================================"
echo " TVDiag Multimodal GNN + Dual-Channel RCA"
echo " Model:          ${MODEL}"
echo " Anomaly method: ${ANOMALY_METHOD}"
echo " Anomaly epochs: ${ANOMALY_EPOCHS}"
echo " RCA method:     tvdig"
echo " TVDiag model:   ${TVDIG_MODEL}"
echo " Dates:          ${DATES[*]}"
echo "============================================================"

# 检查 TVDiag 模型文件
if [ ! -f "${TVDIG_MODEL}/tvdig.pt" ]; then
    echo "[ERROR] TVDiag model not found: ${TVDIG_MODEL}/tvdig.pt"
    echo "  Run training first: python -m failure_localization.train_tvdig --output-dir ${TVDIG_MODEL}"
    exit 1
fi

if [ ! -f "${TVDIG_MODEL}/embedding_cache.pkl" ]; then
    echo "[WARN] Embedding cache not found: ${TVDIG_MODEL}/embedding_cache.pkl"
    echo "  TVDiag will build embeddings on-the-fly (slower first run)"
fi

# 检查预处理数据目录
for DATE in "${DATES[@]}"; do
    MMDD=$(echo "$DATE" | sed 's/2021-//;s/-//')
    MISSING=0
    for DIR in "${MMDD}_metric_fault" "${MMDD}_microcause" "${MMDD}_log_fault" "${MMDD}_tracerca"; do
        if [ ! -d "$DIR" ]; then
            echo "[WARN] Missing directory: $DIR"
            MISSING=1
        fi
    done
    if [ "$MISSING" -eq 1 ]; then
        echo "[WARN] Some data directories for ${DATE} are missing. Runs may fail."
    fi
done

# === 生成运行命令 ===

SUCCESS=0
FAIL=0

for DATE in "${DATES[@]}"; do
    # 提取 MMDD
    MMDD=$(echo "$DATE" | sed 's/2021-//;s/-//')

    # 从 pkl 中提取该日期的所有故障注入时刻（全量运行）
    TIMES=$(${PYTHON} -c "
import pickle
with open('Datasets/GAIA/fault_injection_tracerank/fault_injection_list_${DATE}.pkl', 'rb') as f:
    data = pickle.load(f)
times = sorted(set(fi['time'].strftime('%H:%M') for fi in data if isinstance(fi, dict) and 'time' in fi))
print(' '.join(times))
" 2>/dev/null)

    if [ -z "$TIMES" ]; then
        echo "  [SKIP] No fault events for ${DATE}"
        continue
    fi

    echo ""
    echo ">>> Date: ${DATE} | ${TIMES} fault events <<<"

    for TIME in $TIMES; do
        CASE_NAME="tvdig_${MMDD}"
        TASK_DESC="At 2021/${MMDD:0:2}/${MMDD:2:2} ${TIME} have exceptions in the microservices system. What are these exceptions? Please output an exception analysis."

        echo ""
        echo "--- [$(date '+%Y-%m-%d %H:%M:%S')] ${DATE} ${TIME} ---"

        if [[ "$MODEL" == ollama-* ]]; then
            # 本地 Ollama 模型
            if $PYTHON run.py \
                --task "$TASK_DESC" \
                --name "${CASE_NAME}" \
                --model "$MODEL" \
                --ollama-url "$OLLAMA_URL" \
                --anomaly-method "$ANOMALY_METHOD" \
                --anomaly-epochs "$ANOMALY_EPOCHS" \
                --rca-method tvdig \
                --tvdig-model "$TVDIG_MODEL" \
                --report-dir Report_tvdig; then
                SUCCESS=$((SUCCESS + 1))
            else
                echo "[FAIL] ${DATE} ${TIME} exited with error"
                FAIL=$((FAIL + 1))
            fi
        else
            # API 模型 (glm-4.5, glm-4.7, deepseek-r1-0528, GPT_4 等)
            if $PYTHON run.py \
                --task "$TASK_DESC" \
                --name "${CASE_NAME}" \
                --model "$MODEL" \
                --anomaly-method "$ANOMALY_METHOD" \
                --anomaly-epochs "$ANOMALY_EPOCHS" \
                --rca-method tvdig \
                --tvdig-model "$TVDIG_MODEL" \
                --report-dir Report_tvdig; then
                SUCCESS=$((SUCCESS + 1))
            else
                echo "[FAIL] ${DATE} ${TIME} exited with error"
                FAIL=$((FAIL + 1))
            fi
        fi
    done
done

echo ""
echo "============================================================"
echo " Runs complete: ${SUCCESS} success, ${FAIL} fail"
echo " Starting evaluation..."
echo "============================================================"

# ================================================================
#  自动评估：生成参考文本 + G-sim + W-rate（如其他方法日志存在）
# ================================================================

EVAL_MODEL="glm-4.5"
GT_PKL_DIR="Datasets/GAIA/fault_injection_tracerank/"
REF_OUTPUT="evaluation/reference_texts_tvdig"

echo ""
echo "--- Evaluation: TVDiag ---"
echo "  Log: experiments_tvdig.log"
echo "  GT:  ${GT_PKL_DIR}"
echo "  Ref: ${REF_OUTPUT}"
echo ""

# 基本评估 + 参考文本生成 + G-sim
$PYTHON -m evaluation.run_evaluation \
    --log experiments_tvdig.log \
    --gt-pkl-dir "$GT_PKL_DIR" \
    --generate-references \
    --ref-model "$EVAL_MODEL" \
    --ref-output "$REF_OUTPUT" \
    --gsim \
    --gsim-model "$EVAL_MODEL" \
    --model-name "$EVAL_MODEL"

# 如果 DualChannel 日志也存在，额外运行 W-rate 对比评估
DC_LOG="experiments_dualchannel.log"
if [ -f "$DC_LOG" ]; then
    echo ""
    echo "--- Evaluation: W-rate (TVDiag vs DualChannel) ---"
    $PYTHON -m evaluation.run_evaluation \
        --log experiments_tvdig.log \
        --gt-pkl-dir "$GT_PKL_DIR" \
        --ref-input "$REF_OUTPUT" \
        --gsim \
        --gsim-model "$EVAL_MODEL" \
        --compute-wrate \
        --methods-log "TVDiag=experiments_tvdig.log" \
                      "DualChannel=${DC_LOG}" \
        --voter-model "$EVAL_MODEL" \
        --model-name "$EVAL_MODEL"
fi

# 如果 LocaleXpert 日志也存在，额外运行三方对比
LX_LOG="experiments_localexpert.log"
if [ -f "$LX_LOG" ]; then
    echo ""
    echo "--- Evaluation: W-rate (TVDiag vs LocaleXpert) ---"
    $PYTHON -m evaluation.run_evaluation \
        --log experiments_tvdig.log \
        --gt-pkl-dir "$GT_PKL_DIR" \
        --ref-input "$REF_OUTPUT" \
        --gsim \
        --gsim-model "$EVAL_MODEL" \
        --compute-wrate \
        --methods-log "TVDiag=experiments_tvdig.log" \
                      "LocaleXpert=${LX_LOG}" \
        --voter-model "$EVAL_MODEL" \
        --model-name "$EVAL_MODEL"
fi

echo ""
echo "============================================================"
echo " All evaluation complete."
echo "============================================================"
