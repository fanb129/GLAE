#!/bin/bash

# 定义基本参数
DATASET_NAME="seed4"
LOG_DIR="lstm_dann/shell_logs/${DATASET_NAME}"
SCRIPT_PATH="lstm_dann/main_glae.py"

# 创建日志目录（如果不存在）
mkdir -p $LOG_DIR

# 循环从 0 到 14，为每个受试者启动一个任务
for i in $(seq 0 14)
do
    echo "Starting training for subject $i..."

    # 构建日志文件名
    LOG_FILE="${LOG_DIR}/subject_${i}.log"

    # 使用 nohup 和 & 在后台启动每个任务，并将输出重定向到单独的日志文件
    # setsid 是可选的，但可以确保即使父进程（终端）关闭，子进程也不会被挂断信号（SIGHUP）杀死
    # -u 参数确保 Python 的 stdout 和 stderr 不被缓冲
    setsid nohup python -u ${SCRIPT_PATH} \
        --dataset_name="${DATASET_NAME}" \
        --subject=${i} \
        --cuda="2" \
        > "${LOG_FILE}" 2>&1 &

    # 打印进程ID
    echo "Process started with PID: $!"
done

echo "All 15 training processes have been launched."