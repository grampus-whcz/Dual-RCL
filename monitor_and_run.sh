#！/bin/bash
# nohup bash monitor_and_run.sh > monitor_and_run.log 2>&1 &

# 定义要监控的进程的关键字
# TARGET_PROCESS_KEYWORD="python -m rca.run_agent_standard_multi_candidate --dataset Bank"
TARGET_PROCESS_KEYWORD="/root/shared-nvme/.conda/envs/LocaleXpert_env/bin/python run.py --task At"

# 定义进程结束后要执行的命令
COMMAND_TO_RUN="nohup bash experiments_dualchannel.sh > experiments_dualchannel.log 2>&1 &"

# 定义检查间隔，单位为秒 (5分钟 = 300秒)
CHECK_INTERVAL=300

echo "开始监控进程: $TARGET_PROCESS_KEYWORD"
echo "检查间隔: $CHECK_INTERVAL 秒"

# 无限循环，直到目标进程结束并执行新命令
while true; do
    # 使用 ps 和 grep 查找进程
    # grep -F 用于字面匹配，避免特殊字符问题
    # grep -v grep 用于排除 grep 命令本身
    if ps aux | grep -F "$TARGET_PROCESS_KEYWORD" | grep -v grep > /dev/null; then
        # 进程存在，打印信息并等待
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 目标进程仍在运行，将在 $CHECK_INTERVAL 秒后再次检查。"
        sleep $CHECK_INTERVAL
    else
        # 进程不存在，执行新命令并退出循环
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 目标进程已结束。正在执行新命令..."
        eval "$COMMAND_TO_RUN"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 新命令已启动。监控脚本即将退出。"
        break
    fi
done

exit 0