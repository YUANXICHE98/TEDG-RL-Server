#!/bin/bash

# Transition阶段训练 (1000→3000 episodes)
# 使用Manager约束

set -e

echo "========================================="
echo "Transition阶段训练 (1000→3000 episodes)"
echo "========================================="

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tedg-rl-demo

# 检查checkpoint是否存在
CHECKPOINT="ablation_v3/results/warmup_1000_with_manager/checkpoints/model_final.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint不存在: $CHECKPOINT"
    echo "请先完成Warmup阶段训练"
    exit 1
fi

echo "✅ 找到checkpoint: $CHECKPOINT"
echo ""

# 训练参数
EXP_NAME="transition_3000_with_manager"
EPISODES=3000
MAX_STEPS=2000
PHASE="transition"

echo "训练配置:"
echo "  实验名称: $EXP_NAME"
echo "  目标Episodes: $EPISODES"
echo "  最大Steps: $MAX_STEPS"
echo "  阶段: $PHASE"
echo "  Resume from: $CHECKPOINT"
echo ""

# 开始训练
echo "开始训练..."
echo "预计时间: 8-10小时"
echo ""

python -u ablation_v3/train/train_v3_gat_moe.py \
    --exp-name "$EXP_NAME" \
    --episodes $EPISODES \
    --max-steps $MAX_STEPS \
    --phase "$PHASE" \
    --resume "$CHECKPOINT" \
    2>&1 | tee "ablation_v3/results/${EXP_NAME}/training_output.log"

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Transition阶段训练完成！"
    echo "========================================="
    echo ""
    echo "结果位置: ablation_v3/results/$EXP_NAME/"
    echo ""
    echo "下一步:"
    echo "1. 分析结果: python tools/analyze_transition_results.py --result_dir ablation_v3/results/$EXP_NAME"
    echo "2. 对比分析: python tools/compare_with_without_manager.py --phase transition"
    echo "3. 启动Fine-tune: bash ablation_v3/scripts/run_finetune_5000.sh"
else
    echo ""
    echo "❌ 训练失败，请检查日志"
    exit 1
fi
