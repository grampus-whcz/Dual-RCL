#！/bin/bash
# nohup bash run_all.sh > run_all.log 2>&1 &
#
# LocaleXpert GAIA 数据预处理一键运行脚本
#
# 执行顺序: step1 -> step2 -> step3 + step4（可并行）+ step5（独立）
#
# 用法:
#   bash scripts/run_all.sh                    # 全量运行
#   bash scripts/run_all.sh 2021-07-01         # 仅处理指定日期
#   bash scripts/run_all.sh --skip-train       # 跳过模型训练（step3+step5）
#
# 前置条件:
#   - GAIA 数据集已下载到 /root/shared-nvme/data_set/gaia/gaia/
#   - 虚拟环境 /root/shared-nvme/.conda/envs/LocaleXpert_env/ 已就绪
#

set -euo pipefail

# ===== 配置 =====
PYTHON=/root/shared-nvme/.conda/envs/LocaleXpert_env/bin/python
GAIA_ROOT=/root/shared-nvme/data_set/gaia/gaia/MicroSS
DATASETS_ROOT=./Datasets/GAIA
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

# 解析参数
DATE_ARG=""
SKIP_TRAIN=false
for arg in "$@"; do
    case $arg in
        --skip-train)
            SKIP_TRAIN=true
            ;;
        *)
            DATE_ARG="$arg"
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo "============================================================"
echo " LocaleXpert GAIA Data Preprocessing Pipeline"
echo "============================================================"
echo " Project root : $PROJECT_ROOT"
echo " Python       : $PYTHON"
echo " GAIA root    : $GAIA_ROOT"
echo " Datasets     : $DATASETS_ROOT"
echo " Date filter  : ${DATE_ARG:-all dates}"
echo " Skip training: $SKIP_TRAIN"
echo "============================================================"
echo ""

# 构建 --dates 参数
DATES_OPT=""
if [ -n "$DATE_ARG" ]; then
    DATES_OPT="--dates $DATE_ARG"
fi

# ===== Step 1: 合并 trace =====
echo ">>>>> Step 1: Merging trace files (per-service -> per-day) <<<<<"
$PYTHON scripts/step1_trace_merge.py \
    --gaia-root "$GAIA_ROOT" \
    --output-root "$DATASETS_ROOT" \
    $DATES_OPT
echo "Step 1 done."
echo ""

# ===== Step 2: 故障注入 pkl =====
echo ">>>>> Step 2: Generating fault injection pkl files <<<<<"
$PYTHON scripts/step2_fault_injection.py \
    --datasets-root "$DATASETS_ROOT" \
    --gaia-trace-root "$GAIA_ROOT/trace" \
    --cleanup true
echo "Step 2 done."
echo ""

if [ "$SKIP_TRAIN" = false ]; then

    # ===== Step 3 + Step 5: 训练模型（可并行）=====
    echo ">>>>> Step 3 & Step 5: Training models (parallel) <<<<<"

    # Step 3: 训练 MEPFL 模型
    $PYTHON scripts/step3_train_mepfl.py \
        --datasets-root "$DATASETS_ROOT" \
        --output-dir ./mepfl_model_gaia \
        --train-days 5 \
        --max-faults 2429 &
    PID_STEP3=$!

    # Step 5: 训练 PatternMatcher 模型
    $PYTHON scripts/step5_train_patternmatcher.py \
        --output-path ./patterncla.pt \
        --samples-per-class 500 \
        --epochs 100 \
        --seed 42 &
    PID_STEP5=$!

    # 等待两个训练任务完成
    echo "Waiting for step3 (PID $PID_STEP3) and step5 (PID $PID_STEP5)..."
    wait $PID_STEP3
    echo "Step 3 (MEPFL training) done."
    wait $PID_STEP5
    echo "Step 5 (PatternMatcher training) done."
    echo ""

fi

# ===== Step 4: 生成运行时数据目录 =====
echo ">>>>> Step 4: Preparing runtime data directories <<<<<"
$PYTHON scripts/step4_prepare_runtime_data.py \
    --datasets-root "$DATASETS_ROOT" \
    --gaia-root "$GAIA_ROOT" \
    $DATES_OPT
echo "Step 4 done."
echo ""

echo "============================================================"
echo " All steps completed!"
echo ""
echo " Output files:"
echo "   $DATASETS_ROOT/trace_by_day/*.csv"
echo "   $DATASETS_ROOT/fault_injection_tracerank/*.pkl"
echo "   ./mepfl_model_gaia/rf_model.pkl"
echo "   ./mepfl_model_gaia/mlp_model.pkl"
echo "   ./patterncla.pt"
echo "   ./{MMDD}_tracerca/"
echo "   ./{MMDD}_trace_ano/"
echo "   ./{MMDD}_metric_fault/"
echo "   ./{MMDD}_microcause/"
echo "   ./{MMDD}_log_fault/"
echo ""
echo " To run LocaleXpert:"
echo "   $PYTHON run.py --task 'At 2021/07/01 11:50 have exceptions...' --name test_0701"
echo "============================================================"
