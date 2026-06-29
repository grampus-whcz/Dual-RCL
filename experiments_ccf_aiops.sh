#!/bin/bash
# CCF AIOps 2022 批量实验脚本
# 三种方法：LocaleXpert / DualChannel / DualChannel+TVDiag
#
# 用法：
#   nohup bash experiments_ccf_aiops.sh localexpert    > experiments_ccf_aiops_localexpert.log 2>&1 &
#   nohup bash experiments_ccf_aiops.sh dualchannel    > experiments_ccf_aiops_dualchannel.log 2>&1 &
#   nohup bash experiments_ccf_aiops.sh tvdig          > experiments_ccf_aiops_tvdig.log 2>&1 &
#
# 数据要求：需要先运行 ccf_aiops_preprocess.py 生成 0320*/0320b*/0320c* 目录

set -e

PYTHON=/root/shared-nvme/.conda/envs/LocaleXpert_env/bin/python

# === 实验方法选择 ===
METHOD="${1:-localexpert}"

# === LLM 模型配置 ===
MODEL="ollama-qwen3-14b"
OLLAMA_URL="http://localhost:11434/v1"

# === 数据集配置 ===
DATE="2022-03-20"
GT_DIR="/root/shared-nvme/data_set/2022_CCF_AIOps_challenge/training_data_with_faults/groundtruth"

# === 根据方法设置参数 ===
case "$METHOD" in
    localexpert)
        EXTRA_ARGS="--skip-multivariate --dataset ccf_aiops"
        REPORT_DIR="Report_ccf_aiops/localexpert"
        ;;
    dualchannel)
        EXTRA_ARGS="--anomaly-method tranad --anomaly-epochs 3 --dataset ccf_aiops"
        REPORT_DIR="Report_ccf_aiops/dualchannel"
        ;;
    tvdig)
        EXTRA_ARGS="--anomaly-method tranad --anomaly-epochs 3 --rca-method tvdig --tvdig-model ./tvdig_checkpoint --dataset ccf_aiops"
        REPORT_DIR="Report_ccf_aiops/tvdig"
        ;;
    *)
        echo "Unknown method: $METHOD (use: localexpert, dualchannel, tvdig)"
        exit 1
        ;;
esac

mkdir -p "$REPORT_DIR"

echo "============================================================"
echo " CCF AIOps 2022 Experiment"
echo " Method:      ${METHOD}"
echo " Model:       ${MODEL}"
echo " Date:        ${DATE}"
echo " Extra args:  ${EXTRA_ARGS}"
echo " Report dir:  ${REPORT_DIR}"
echo "============================================================"

# === 遍历所有 cloudbed ===
CLOUDBEDS=("cloudbed-1" "cloudbed-2" "cloudbed-3")
DIR_PREFIXES=("0320" "0320b" "0320c")
GT_FILES=("groundtruth-k8s-1-2022-03-20.csv" "groundtruth-k8s-2-2022-03-20.csv" "groundtruth-k8s-3-2022-03-20.csv")

TOTAL=0
SUCCESS=0
FAIL=0

for i in "${!CLOUDBEDS[@]}"; do
    CLOUDBED="${CLOUDBEDS[$i]}"
    PREFIX="${DIR_PREFIXES[$i]}"
    GT_FILE="${GT_FILES[$i]}"

    echo ""
    echo ">>> Cloudbed: ${CLOUDBED}, dir prefix: ${PREFIX} <<<"

    # 检查预处理数据是否存在
    if [ ! -d "${PREFIX}_trace_ano" ]; then
        echo "  [SKIP] Preprocessed data not found (${PREFIX}_trace_ano/)"
        continue
    fi

    # 从 groundtruth CSV 中提取事件时间
    EVENTS=$(${PYTHON} -c "
import pandas as pd
from datetime import datetime
gt = pd.read_csv('${GT_DIR}/${GT_FILE}')
for _, row in gt.iterrows():
    ts = datetime.fromtimestamp(row['timestamp'])
    print(f'{ts.strftime(\"%H:%M\")}|{row[\"level\"]}|{row[\"cmdb_id\"]}|{row[\"failure_type\"]}')
" 2>/dev/null)

    if [ -z "$EVENTS" ]; then
        echo "  [SKIP] No events found in ${GT_FILE}"
        continue
    fi

    while IFS= read -r line; do
        IFS='|' read -r TIME LEVEL CMDB FAULT <<< "$line"
        TOTAL=$((TOTAL + 1))

        echo ""
        echo "  [${TOTAL}] ${CLOUDBED} ${TIME} — ${LEVEL}/${CMDB}: ${FAULT}"

        # 构造 task 描述
        TASK="At ${DATE} ${TIME} have exceptions in the microservices system. What are these exceptions? Please output an exception analysis."
        CASE_NAME="ccf_${METHOD}_${PREFIX}_${TIME//:/-}"

        set +e
        if [[ "$MODEL" == ollama-* ]]; then
            $PYTHON run.py \
                --task "$TASK" \
                --name "${CASE_NAME}" \
                --model "$MODEL" \
                --ollama-url "$OLLAMA_URL" \
                $EXTRA_ARGS \
                --report-dir "$REPORT_DIR"
        else
            $PYTHON run.py \
                --task "$TASK" \
                --name "${CASE_NAME}" \
                --model "$MODEL" \
                $EXTRA_ARGS \
                --report-dir "$REPORT_DIR"
        fi
        RC=$?
        set -e

        if [ $RC -eq 0 ]; then
            SUCCESS=$((SUCCESS + 1))
            echo "  [OK] ${CASE_NAME}"
        else
            FAIL=$((FAIL + 1))
            echo "  [FAIL] ${CASE_NAME} (exit code: ${RC})"
        fi

    done <<< "$EVENTS"
done

echo ""
echo "============================================================"
echo " Experiment Complete: ${METHOD}"
echo " Total: ${TOTAL}  Success: ${SUCCESS}  Failed: ${FAIL}"
echo "============================================================"
