#!/bin/bash

# Fine-tune阶段训练 (3000→5000 episodes)
# 使用Manager约束

set -e

echo "========================================="
echo "Fine-tune阶段训练 (3000→5000 episodes)"
echo "========================================="

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tedg-rl-demo

# 检查checkpoint是否存在
CHECKPOINT="ablation_v3/results/transition_3000_with_manager/checkpoints/model_final.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint不存在: $CHECKPOINT"
    echo "请先完成Transition阶段训练"
    exit 1
fi

echo "✅ 找到checkpoint: $CHECKPOINT"
echo ""

# 训练参数
EXP_NAME="finetune_5000_with_manager"
EPISODES=5000
MAX_STEPS=2000
PHASE="finetune"

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
    echo "✅ Fine-tune阶段训练完成！"
    echo "========================================="
    echo ""
    echo "结果位置: ablation_v3/results/$EXP_NAME/"
    echo ""
    echo "下一步:"
    echo "1. 分析结果: python tools/analyze_finetune_results.py --result_dir ablation_v3/results/$EXP_NAME"
    echo "2. 完整对比: python tools/compare_full_training.py"
    echo "3. 生成报告: python tools/generate_final_report.py"
else
    echo ""
    echo "❌ 训练失败，请检查日志"
    exit 1
fi
