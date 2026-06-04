#!/bin/bash
# 最终改进版 (Dual-Channel RCA + 多变量检测 + LLM冲突裁决) 批量运行脚本
#
# 改进版 vs 论文原始方法的核心区别：
#
#   Phase 4:  单变量检测(CNN+SPOT) + 多变量检测(TranAD/USAD等) + PCMCI因果发现
#   Phase 5:  双通道 RCA:
#               5A: SPOT eta  → 随机游走 → 单变量根因排名
#               5B: Multi eta → 随机游走 → 多变量根因排名
#   Phase 6:  LLM 冲突仲裁:
#               比较两个根因排名 → 分类冲突场景(4种) → 因果推理最终裁决
#   Phase 7-9: 知识注入 + 日志分析 + ChatChain LLM 推理
#
# 用法:
#   nohup bash experiments_dualchannel.sh > experiments_dualchannel.log 2>&1 &
#
# 运行后评估:
#   python -m evaluation.run_evaluation --log experiments_dualchannel.log
#
# 批量评估 Report/ 下所有日志:
#   python -m evaluation.run_evaluation --log-dir Report/

set -e

PYTHON=/root/shared-nvme/.conda/envs/LocaleXpert_env/bin/python

# === 配置 ===

# LLM 模型选择:
#   API 模型:  deepseek-r1-0528, GPT_4
#   本地模型:  ollama-qwen3-14b, ollama-qwen3-8b (需要 Ollama 服务运行)
MODEL="ollama-qwen3-14b"
# MODEL="deepseek-r1-0528"

# Ollama 地址 (仅本地模型需要)
OLLAMA_URL="http://localhost:11434/v1"

# 多变量异常检测方法 (可选: tranad, usad, omnianomaly, mad_gan, mscred, gdn, mtad_gat)
ANOMALY_METHOD="usad"

# 多变量检测训练轮数
ANOMALY_EPOCHS=5

# === 可用日期 (只有 0701-0703 有预处理数据) ===
DATES=(
    "2021-07-01"
    "2021-07-02"
    "2021-07-03"
)

# === 生成运行命令 ===
# 每个日期取 3 个代表性故障注入时刻（早、中、晚各一个）
# 如需全量运行，取消下方 "全量运行" 的注释

echo "============================================================"
echo " Dual-Channel RCA (Final Improved Version)"
echo " Model:          ${MODEL}"
echo " Anomaly method: ${ANOMALY_METHOD}"
echo " Anomaly epochs: ${ANOMALY_EPOCHS}"
echo " Dates:          ${DATES[*]}"
echo " Mode:           双通道 RCA + LLM 冲突裁决"
echo "============================================================"

for DATE in "${DATES[@]}"; do
    # 提取 MMDD
    MMDD=$(echo "$DATE" | sed 's/2021-//;s/-//')

    # 从 pkl 中提取该日期的前 3 个故障注入时刻
    TIMES=$(${PYTHON} -c "
import pickle, re
with open('Datasets/GAIA/fault_injection_tracerank/fault_injection_list_${DATE}.pkl', 'rb') as f:
    data = pickle.load(f)
times = sorted(set(fi['time'].strftime('%H:%M') for fi in data if isinstance(fi, dict) and 'time' in fi))
# 取早、中、晚各一个
if len(times) >= 3:
    pick = [times[0], times[len(times)//2], times[-1]]
elif len(times) >= 1:
    pick = times[:1]
else:
    pick = []
print(' '.join(pick))
" 2>/dev/null)

    if [ -z "$TIMES" ]; then
        echo "  [SKIP] No fault events for ${DATE}"
        continue
    fi

    for TIME in $TIMES; do
        CASE_NAME="dualch_${MMDD}"

        echo ""
        echo ">>> ${DATE} ${TIME} <<<"

        if [[ "$MODEL" == ollama-* ]]; then
            # 本地 Ollama 模型
            $PYTHON run.py \
                --task "At 2021/${MMDD:0:2}/${MMDD:2:2} ${TIME} have exceptions in the microservices system. What are these exceptions? Please output an exception analysis." \
                --name "${CASE_NAME}" \
                --model "$MODEL" \
                --ollama-url "$OLLAMA_URL" \
                --anomaly-method "$ANOMALY_METHOD" \
                --anomaly-epochs "$ANOMALY_EPOCHS"
        else
            # API 模型
            $PYTHON run.py \
                --task "At 2021/${MMDD:0:2}/${MMDD:2:2} ${TIME} have exceptions in the microservices system. What are these exceptions? Please output an exception analysis." \
                --name "${CASE_NAME}" \
                --model "$MODEL" \
                --anomaly-method "$ANOMALY_METHOD" \
                --anomaly-epochs "$ANOMALY_EPOCHS"
        fi
    done
done

# === 全量运行（取消注释以运行所有故障注入时刻）===
# for DATE in "${DATES[@]}"; do
#     MMDD=$(echo "$DATE" | sed 's/2021-//;s/-//')
#     $PYTHON -c "
# import pickle
# with open('Datasets/GAIA/fault_injection_tracerank/fault_injection_list_${DATE}.pkl', 'rb') as f:
#     data = pickle.load(f)
# times = sorted(set(fi['time'].strftime('%H:%M') for fi in data if isinstance(fi, dict)))
# for t in times:
#     print(t)
# " | while read TIME; do
#         CASE_NAME="dualch_${MMDD}"
#         echo ">>> ${DATE} ${TIME} <<<"
#         if [[ "$MODEL" == ollama-* ]]; then
#             $PYTHON run.py \
#                 --task "At 2021/${MMDD:0:2}/${MMDD:2:2} ${TIME} have exceptions in the microservices system. What are these exceptions? Please output an exception analysis." \
#                 --name "${CASE_NAME}" \
#                 --model "$MODEL" \
#                 --ollama-url "$OLLAMA_URL" \
#                 --anomaly-method "$ANOMALY_METHOD" \
#                 --anomaly-epochs "$ANOMALY_EPOCHS"
#         else
#             $PYTHON run.py \
#                 --task "At 2021/${MMDD:0:2}/${MMDD:2:2} ${TIME} have exceptions in the microservices system. What are these exceptions? Please output an exception analysis." \
#                 --name "${CASE_NAME}" \
#                 --model "$MODEL" \
#                 --anomaly-method "$ANOMALY_METHOD" \
#                 --anomaly-epochs "$ANOMALY_EPOCHS"
#         fi
#     done
# done

echo ""
echo "============================================================"
echo " All runs complete."
echo " Evaluate with:"
echo "   python -m evaluation.run_evaluation --log-dir Report/"
echo "============================================================"
