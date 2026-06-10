#!/bin/bash
# 论文 [171] 原始方法 (LocaleXpert) 批量运行脚本
#
# 论文原始方法 vs 当前增强版的关键区别：
#
#   | 方面             | 论文原始方法 (LocaleXpert)           | 当前增强版             |
#   |------------------|--------------------------------------|-----------------------|
#   | 异常检测         | 3-sigma (单变量) + TraceAnomaly      | + 多变量检测 (TranAD等) |
#   | 根因定位         | PCMCI+RandomWalk + MEPFL             | + 双通道RCA + 冲突裁决  |
#   | LLM推理          | ChatChain 多Agent协作                | 相同                   |
#   | 模型             | Qwen 3 8B Instruct                  | 可配置                 |
#
# 因此，运行论文原始方法只需加上 --skip-multivariate 跳过多变量检测，
# 即可还原论文的单通道（仅单变量）RCA流水线。
#
# 用法:
#   nohup bash experiments_localexpert.sh > experiments_localexpert.log 2>&1 &
#
# 运行后评估 (自动执行):
#   python -m evaluation.run_evaluation --log experiments_localexpert.log \
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

# === 可用日期 (只有 0701-0703 有预处理数据) ===
# 如需更多日期，先运行: python scripts/step4_prepare_runtime_data.py --dates 2021-07-04
DATES=(
    "2021-07-01"
)

# === 生成运行命令 ===
# 每个日期取 3 个代表性故障注入时刻（早、中、晚各一个）
# 如需全量运行，取消下方 "全量运行" 的注释

echo "============================================================"
echo " LocaleXpert (Paper [171]) - Original Method"
echo " Model:  ${MODEL}"
echo " Dates:  ${DATES[*]}"
echo " Mode:   --skip-multivariate (original univariate-only pipeline)"
echo "============================================================"

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

    for TIME in $TIMES; do
        CASE_NAME="localexpert_${MMDD}"

        echo ""
        echo ">>> ${DATE} ${TIME} <<<"

        if [[ "$MODEL" == ollama-* ]]; then
            # 本地 Ollama 模型
            $PYTHON run.py \
                --task "At 2021/${MMDD:0:2}/${MMDD:2:2} ${TIME} have exceptions in the microservices system. What are these exceptions? Please output an exception analysis." \
                --name "${CASE_NAME}" \
                --model "$MODEL" \
                --ollama-url "$OLLAMA_URL" \
                --skip-multivariate \
                --report-dir Report_localexpert
        else
            # API 模型 (glm-4.5, glm-4.7, deepseek-r1-0528, GPT_4 等)
            $PYTHON run.py \
                --task "At 2021/${MMDD:0:2}/${MMDD:2:2} ${TIME} have exceptions in the microservices system. What are these exceptions? Please output an exception analysis." \
                --name "${CASE_NAME}" \
                --model "$MODEL" \
                --skip-multivariate \
                --report-dir Report_localexpert
        fi
    done
done

echo ""
echo "============================================================"
echo " All runs complete. Starting evaluation..."
echo "============================================================"

# ================================================================
#  自动评估：生成参考文本 + G-sim + W-rate（如两个方法日志都存在）
# ================================================================

EVAL_MODEL="glm-4.5"
GT_PKL_DIR="Datasets/GAIA/fault_injection_tracerank/"
REF_OUTPUT="evaluation/reference_texts_localexpert"

echo ""
echo "--- Evaluation: LocaleXpert ---"
echo "  Log: experiments_localexpert.log"
echo "  GT:  ${GT_PKL_DIR}"
echo "  Ref: ${REF_OUTPUT}"
echo ""

# 基本评估 + 参考文本生成 + G-sim
$PYTHON -m evaluation.run_evaluation \
    --log experiments_localexpert.log \
    --gt-pkl-dir "$GT_PKL_DIR" \
    --generate-references \
    --ref-model "$EVAL_MODEL" \
    --ref-output "$REF_OUTPUT" \
    --gsim \
    --gsim-model "$EVAL_MODEL" \
    --model-name "$EVAL_MODEL"

# 如果 dualchannel 日志也存在，额外运行 W-rate 对比评估
DUALCH_LOG="experiments_dualchannel.log"
if [ -f "$DUALCH_LOG" ]; then
    echo ""
    echo "--- Evaluation: W-rate (LocaleXpert vs DualChannel) ---"
    $PYTHON -m evaluation.run_evaluation \
        --log experiments_localexpert.log \
        --gt-pkl-dir "$GT_PKL_DIR" \
        --ref-input "$REF_OUTPUT" \
        --gsim \
        --gsim-model "$EVAL_MODEL" \
        --compute-wrate \
        --methods-log "LocaleXpert=experiments_localexpert.log" \
                      "DualChannel=${DUALCH_LOG}" \
        --voter-model "$EVAL_MODEL" \
        --model-name "$EVAL_MODEL"
fi

echo ""
echo "============================================================"
echo " All evaluation complete."
echo "============================================================"
