#!/bin/bash
# 新版本完整训练 - 5000 episodes
# 包含所有4个机制：Manager约束、熵最小化、时间一致性、重叠惩罚

set -e

echo "=========================================="
echo "新版本V3完整训练 - 5000 Episodes"
echo "=========================================="
echo ""
echo "包含机制："
echo "  ✓ Manager内层约束（超图-路由对齐）"
echo "  ✓ 熵最小化（Fine-tune阶段符号反转）"
echo "  ✓ 时间一致性（减少意图震荡）"
echo "  ✓ 专家重叠惩罚（强制正交）"
echo ""
echo "预期效果："
echo "  - Alpha熵: 1.38 → 0.2-0.3 (下降80%+)"
echo "  - 平均分数: 提升100%+"
echo "  - 专家极致专业化"
echo ""
echo "预计时间: 20-30小时（CPU）"
echo "=========================================="
echo ""

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tedg-rl-demo

# 设置实验名称（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="v3_new_version_5000ep_${TIMESTAMP}"

echo "实验名称: ${EXP_NAME}"
echo "开始时间: $(date)"
echo ""
echo "⚠️  这将运行20-30小时，建议使用nohup或screen"
echo ""

# 运行训练
nohup python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name "${EXP_NAME}" \
    --episodes 5000 \
    --max-steps 2000 \
    > "ablation_v3/results/${EXP_NAME}/training_output.log" 2>&1 &

PID=$!
echo "训练已在后台启动，PID: ${PID}"
echo ""
echo "监控训练："
echo "  tail -f ablation_v3/results/${EXP_NAME}/training_output.log"
echo ""
echo "停止训练："
echo "  kill ${PID}"
echo ""
echo "PID已保存到: ablation_v3/results/${EXP_NAME}/train.pid"
echo "${PID}" > "ablation_v3/results/${EXP_NAME}/train.pid"
echo ""
