#!/bin/bash
# V4 Transition阶段训练 (1000-3000 episodes)
# Cross-Attention Guided Hierarchical MoE

set -e

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate tedg-rl-demo

echo "=========================================="
echo "V4 Transition阶段训练 (1000-3000 episodes)"
echo "=========================================="

# 检查Warmup checkpoint
WARMUP_CHECKPOINT="ablation_v4/results/warmup_1000/checkpoints/model_01000.pth"
if [ ! -f "$WARMUP_CHECKPOINT" ]; then
    echo "❌ 错误: 找不到Warmup checkpoint: $WARMUP_CHECKPOINT"
    echo "   请先运行: bash ablation_v4/scripts/run_warmup_1000.sh"
    exit 1
fi

python ablation_v4/train/train_v4_cross_attention.py \
    --exp-name transition_3000 \
    --episodes 3000 \
    --max-steps 2000 \
    --num-experts 4 \
    --resume "$WARMUP_CHECKPOINT" \
    2>&1 | tee ablation_v4/results/transition_3000.log

echo ""
echo "✓ Transition阶段训练完成"
echo "  结果保存在: ablation_v4/results/transition_3000/"
echo ""
