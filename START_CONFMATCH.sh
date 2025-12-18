#!/bin/bash
# ConfMatch 训练启动脚本（不覆盖旧日志/输出）

set -euo pipefail

# 兼容：有的外部组件要求 CRS_OAI_KEY
if [[ -z "${CRS_OAI_KEY:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
  export CRS_OAI_KEY="${OPENAI_API_KEY}"
fi

MODE="${1:-gpu}" # gpu | cpu
if [[ "${MODE}" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  export MUSA_VISIBLE_DEVICES=""
fi

OUT_DIR="${TEDG_OUTPUT_DIR:-results_confmatch}"
LOG_FILE="${2:-train_confmatch.log}"

echo "=================================="
echo "TEDG-RL ConfMatch 训练"
echo "=================================="
echo "模式: ${MODE}"
echo "输出目录: ${OUT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo ""

nohup python train_confmatch.py > "${LOG_FILE}" 2>&1 &
PID=$!
echo "训练进程ID: $PID"
echo ""
echo "监控命令:"
echo "  tail -f ${LOG_FILE}"
echo "  ps aux | grep train_confmatch"
echo "  kill ${PID}"
echo ""
echo "训练已在后台启动！"

