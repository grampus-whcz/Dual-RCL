#！/bin/bash
# nohup bash experiments_a.sh > experiments_a.log 2>&1 &

# === 运行模式说明 ===
#
# 模式1: 双通道 RCA（完整流水线，推荐）
#   单变量+多变量各跑独立RCA → LLM冲突裁决 → 因果推理最终裁决
#   python run.py --task '...' --name test_0701 --anomaly-method usad
#
# 模式2: TVDiag 多模态 GNN（跳过双通道RCA，直接GNN推理）
#   python run.py --task '...' --name test_0701 \
#       --rca-method tvdig --tvdig-model ./tvdig_checkpoint
#
# 模式3: 仅单变量（跳过多变量检测）
#   python run.py --task '...' --name test_0701 --skip-multivariate
#

# --- 模式1: 双通道 RCA (默认 anomaly-method=tranad) ---
python run.py \
  --task 'At 2021/07/01 11:50 have exceptions in the microservices system. What are these exceptions? Please output an exception analysis.' \
  --name test_0701 \
  --anomaly-method usad \
  --model ollama-qwen3-14b

# --- 模式1b: 双通道 RCA + 本地 Ollama (无需 API) ---
# python run.py \
#   --task 'At 2021/07/01 11:50 have exceptions in the microservices system. What are these exceptions? Please output an exception analysis.' \
#   --name test_0701_ollama \
#   --anomaly-method usad \
#   --model ollama-qwen3-14b \
#   --ollama-url http://localhost:11434/v1

# --- 模式2: TVDiag 多模态 GNN ---
# python run.py \
#   --task 'At 2021/07/01 11:50 have exceptions in the microservices system. What are these exceptions? Please output an exception analysis.' \
#   --name test_0701 \
#   --rca-method tvdig --tvdig-model ./tvdig_checkpoint

# # 用现有日志评估（A@k + Latency）
# python -m evaluation.run_evaluation \
#     --log experiments_a.log \
#     --gt mobservice1

# # 或批量评估 Report/ 目录下的所有日志
# python -m evaluation.run_evaluation \
#     --log-dir Report/ \
#     --gt-pkl-dir Datasets/GAIA/fault_injection_tracerank/