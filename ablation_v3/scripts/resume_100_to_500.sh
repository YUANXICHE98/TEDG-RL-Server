#!/bin/bash
# Resume训练脚本 - 从100 episodes继续训练到500
# 使用新版本代码（包含所有4个机制）

set -e

echo "=========================================="
echo "Resume训练: 100 → 500 episodes"
echo "=========================================="
echo ""

# 配置
CHECKPOINT="ablation_v3/results/quick_manager_20260111_230845/checkpoints/model_00100.pth"
EXP_NAME="resume_500_from_100"
TOTAL_EPISODES=500
MAX_STEPS=2000

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint不存在: $CHECKPOINT"
    exit 1
fi

echo "✓ Checkpoint: $CHECKPOINT"
echo "✓ 实验名称: $EXP_NAME"
echo "✓ 目标Episodes: $TOTAL_EPISODES"
echo "✓ 每Episode最大步数: $MAX_STEPS"
echo ""

# 激活conda环境
echo "[激活conda环境]"
eval "$(conda shell.bash hook)"
conda activate tedg-rl-demo
echo "✓ 环境: $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""

# 启动训练
echo "[启动训练]"
echo "命令: python -u ablation_v3/train/train_v3_gat_moe.py \\"
echo "  --exp-name $EXP_NAME \\"
echo "  --episodes $TOTAL_EPISODES \\"
echo "  --max-steps $MAX_STEPS \\"
echo "  --resume $CHECKPOINT"
echo ""

python -u ablation_v3/train/train_v3_gat_moe.py \
    --exp-name "$EXP_NAME" \
    --episodes $TOTAL_EPISODES \
    --max-steps $MAX_STEPS \
    --resume "$CHECKPOINT" \
    2>&1 | tee "ablation_v3/results/${EXP_NAME}/training.log"

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "结果目录: ablation_v3/results/${EXP_NAME}/"
echo "日志文件: ablation_v3/results/${EXP_NAME}/training.log"
