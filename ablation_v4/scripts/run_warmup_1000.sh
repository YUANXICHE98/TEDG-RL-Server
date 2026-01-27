#!/bin/bash
# V4 Warmup阶段训练 (0-1000 episodes)
# Cross-Attention Guided Hierarchical MoE

set -e

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate tedg-rl-demo

echo "=========================================="
echo "V4 Warmup阶段训练 (0-1000 episodes)"
echo "=========================================="

python ablation_v4/train/train_v4_cross_attention.py \
    --exp-name warmup_1000 \
    --episodes 1000 \
    --max-steps 2000 \
    --num-experts 4 \
    2>&1 | tee ablation_v4/results/warmup_1000.log

echo ""
echo "✓ Warmup阶段训练完成"
echo "  结果保存在: ablation_v4/results/warmup_1000/"
echo ""
