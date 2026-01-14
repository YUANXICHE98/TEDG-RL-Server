#!/bin/bash

# 快速启动完整训练流程
# 从500ep继续到5000ep，包含所有三个阶段

set -e

echo "========================================="
echo "V3完整训练 - 从500ep到5000ep"
echo "========================================="
echo ""
echo "训练计划:"
echo "  ✅ 已完成: 0-500 episodes (Warmup部分)"
echo "  ⏳ Phase 1: 500→1000 episodes (完成Warmup) - 2-3小时"
echo "  ⏳ Phase 2: 1000→3000 episodes (Transition) - 8-10小时"
echo "  ⏳ Phase 3: 3000→5000 episodes (Fine-tune) - 8-10小时"
echo ""
echo "总预计时间: 18-23小时"
echo ""

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tedg-rl-demo

# 记录开始时间
START_TIME=$(date +%s)
echo "开始时间: $(date)"
echo ""

# ========================================
# Phase 1: 完成Warmup (500→1000)
# ========================================
echo "========================================="
echo "Phase 1: 完成Warmup阶段 (500→1000)"
echo "========================================="
echo ""

CHECKPOINT="ablation_v3/results/resume_500_from_100/checkpoints/model_00500.pth"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint不存在: $CHECKPOINT"
    exit 1
fi

echo "✅ 找到checkpoint: $CHECKPOINT"
echo "开始训练..."
echo ""

python -u ablation_v3/train/train_v3_gat_moe.py \
    --exp-name warmup_1000_with_manager \
    --episodes 1000 \
    --max-steps 2000 \
    --resume "$CHECKPOINT" \
    2>&1 | tee ablation_v3/results/warmup_1000_with_manager/training_output.log

if [ $? -ne 0 ]; then
    echo "❌ Warmup阶段失败"
    exit 1
fi

echo ""
echo "✅ Warmup阶段完成！"
echo ""

# 快速分析
echo "快速分析Warmup结果..."
python tools/analyze_500ep_results.py || true

# ========================================
# Phase 2: Transition (1000→3000)
# ========================================
echo ""
echo "========================================="
echo "Phase 2: Transition阶段 (1000→3000)"
echo "========================================="
echo ""

CHECKPOINT2="ablation_v3/results/warmup_1000_with_manager/checkpoints/model_final.pth"

if [ ! -f "$CHECKPOINT2" ]; then
    echo "❌ Checkpoint不存在: $CHECKPOINT2"
    exit 1
fi

echo "✅ 找到checkpoint: $CHECKPOINT2"
echo "开始训练..."
echo ""

python -u ablation_v3/train/train_v3_gat_moe.py \
    --exp-name transition_3000_with_manager \
    --episodes 3000 \
    --max-steps 2000 \
    --phase transition \
    --resume "$CHECKPOINT2" \
    2>&1 | tee ablation_v3/results/transition_3000_with_manager/training_output.log

if [ $? -ne 0 ]; then
    echo "❌ Transition阶段失败"
    exit 1
fi

echo ""
echo "✅ Transition阶段完成！"
echo ""

# ========================================
# Phase 3: Fine-tune (3000→5000)
# ========================================
echo ""
echo "========================================="
echo "Phase 3: Fine-tune阶段 (3000→5000)"
echo "========================================="
echo ""

CHECKPOINT3="ablation_v3/results/transition_3000_with_manager/checkpoints/model_final.pth"

if [ ! -f "$CHECKPOINT3" ]; then
    echo "❌ Checkpoint不存在: $CHECKPOINT3"
    exit 1
fi

echo "✅ 找到checkpoint: $CHECKPOINT3"
echo "开始训练..."
echo ""

python -u ablation_v3/train/train_v3_gat_moe.py \
    --exp-name finetune_5000_with_manager \
    --episodes 5000 \
    --max-steps 2000 \
    --phase finetune \
    --resume "$CHECKPOINT3" \
    2>&1 | tee ablation_v3/results/finetune_5000_with_manager/training_output.log

if [ $? -ne 0 ]; then
    echo "❌ Fine-tune阶段失败"
    exit 1
fi

echo ""
echo "✅ Fine-tune阶段完成！"
echo ""

# ========================================
# 完成
# ========================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "========================================="
echo "🎉 完整训练流程完成！"
echo "========================================="
echo ""
echo "总用时: ${HOURS}小时${MINUTES}分钟"
echo ""
echo "结果位置:"
echo "  - Warmup: ablation_v3/results/warmup_1000_with_manager/"
echo "  - Transition: ablation_v3/results/transition_3000_with_manager/"
echo "  - Fine-tune: ablation_v3/results/finetune_5000_with_manager/"
echo ""
echo "下一步: 运行对比分析"
echo "  python tools/compare_with_without_manager.py --phase warmup"
echo "  python tools/compare_with_without_manager.py --phase transition"
echo "  python tools/compare_with_without_manager.py --phase finetune"
echo ""
