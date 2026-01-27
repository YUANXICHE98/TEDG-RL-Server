#!/bin/bash
# V4 Warmup阶段训练 - 从100ep checkpoint继续到1000ep
# Cross-Attention Guided Hierarchical MoE

set -e

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate tedg-rl-demo

echo "=========================================="
echo "V4 Warmup阶段训练 (从100ep继续到1000ep)"
echo "=========================================="

# 检查100ep checkpoint
CHECKPOINT_100="ablation_v4/results/test_100ep/checkpoints/model_00100.pth"
if [ ! -f "$CHECKPOINT_100" ]; then
    echo "❌ 错误: 找不到100ep checkpoint: $CHECKPOINT_100"
    exit 1
fi

echo "✓ 找到100ep checkpoint，继续训练..."

python ablation_v4/train/train_v4_cross_attention.py \
    --exp-name warmup_1000 \
    --episodes 1000 \
    --max-steps 2000 \
    --num-experts 4 \
    --resume "$CHECKPOINT_100" \
    2>&1 | tee ablation_v4/results/warmup_1000.log

echo ""
echo "✓ Warmup阶段训练完成 (100 → 1000 episodes)"
echo "  结果保存在: ablation_v4/results/warmup_1000/"
echo ""
