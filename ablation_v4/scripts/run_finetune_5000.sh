#!/bin/bash
# V4 Fine-tune阶段训练 (3000-5000 episodes)
# Cross-Attention Guided Hierarchical MoE

set -e

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate tedg-rl-demo

echo "=========================================="
echo "V4 Fine-tune阶段训练 (3000-5000 episodes)"
echo "=========================================="

# 检查Transition checkpoint
TRANSITION_CHECKPOINT="ablation_v4/results/transition_3000/checkpoints/model_03000.pth"
if [ ! -f "$TRANSITION_CHECKPOINT" ]; then
    echo "❌ 错误: 找不到Transition checkpoint: $TRANSITION_CHECKPOINT"
    echo "   请先运行: bash ablation_v4/scripts/run_transition_3000.sh"
    exit 1
fi

python ablation_v4/train/train_v4_cross_attention.py \
    --exp-name finetune_5000 \
    --episodes 5000 \
    --max-steps 2000 \
    --num-experts 4 \
    --resume "$TRANSITION_CHECKPOINT" \
    2>&1 | tee ablation_v4/results/finetune_5000.log

echo ""
echo "✓ Fine-tune阶段训练完成"
echo "  结果保存在: ablation_v4/results/finetune_5000/"
echo ""
