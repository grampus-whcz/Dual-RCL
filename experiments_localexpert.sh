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
# 运行后评估:
#   python -m evaluation.run_evaluation --log experiments_localexpert.log
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

# === 可用日期 (只有 0701-0703 有预处理数据) ===
# 如需更多日期，先运行: python scripts/step4_prepare_runtime_data.py --dates 2021-07-04
DATES=(
    "2021-07-01"
    "2021-07-02"
    "2021-07-03"
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
        # 将 HH:MM 转为论文格式的 task 描述
        HH=$(echo "$TIME" | cut -d: -f1)
        MM=$(echo "$TIME" | cut -d: -f2)

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
                --skip-multivariate
        else
            # API 模型
            $PYTHON run.py \
                --task "At 2021/${MMDD:0:2}/${MMDD:2:2} ${TIME} have exceptions in the microservices system. What are these exceptions? Please output an exception analysis." \
                --name "${CASE_NAME}" \
                --model "$MODEL" \
                --skip-multivariate
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
#         HH=$(echo "$TIME" | cut -d: -f1)
#         MM=$(echo "$TIME" | cut -d: -f2)
#         CASE_NAME="localexpert_${MMDD}"
#         echo ">>> ${DATE} ${TIME} <<<"
#         $PYTHON run.py \
#             --task "At 2021/${MMDD:0:2}/${MMDD:2:2} ${TIME} have exceptions in the microservices system. What are these exceptions? Please output an exception analysis." \
#             --name "${CASE_NAME}" \
#             --model "$MODEL" \
#             --skip-multivariate
#     done
# done

echo ""
echo "============================================================"
echo " All runs complete."
echo " Evaluate with:"
echo "   python -m evaluation.run_evaluation --log-dir Report/"
echo "============================================================"
