#!/bin/bash
# 运行消融实验 - 对比三个版本
# 每个版本都跑相同的episodes数

set -e

# 配置
EPISODES=500
MAX_STEPS=2000

echo "=========================================="
echo "消融实验 - 对比三个版本"
echo "=========================================="
echo "Episodes: $EPISODES (每个版本)"
echo "Max Steps: $MAX_STEPS"
echo ""
echo "版本:"
echo "  1. Baseline - 无Manager约束，无熵最小化"
echo "  2. +Manager - 有Manager约束，无熵最小化"
echo "  3. +Manager+Entropy - 有Manager约束，有熵最小化"
echo ""
echo "预计时间: 约6-9小时（3个版本 × 2-3小时）"
echo ""
read -p "确认开始？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tedg-rl-demo

# 1. Baseline
echo ""
echo "=========================================="
echo "1/3 运行 Baseline"
echo "=========================================="
python ablation_v3/train/train_v3_ablation.py \
    --mode baseline \
    --exp-name ablation_baseline_${EPISODES} \
    --episodes $EPISODES \
    --max-steps $MAX_STEPS

echo "✓ Baseline完成"

# 2. +Manager
echo ""
echo "=========================================="
echo "2/3 运行 +Manager"
echo "=========================================="
python ablation_v3/train/train_v3_ablation.py \
    --mode manager \
    --exp-name ablation_manager_${EPISODES} \
    --episodes $EPISODES \
    --max-steps $MAX_STEPS

echo "✓ +Manager完成"

# 3. +Manager+Entropy
echo ""
echo "=========================================="
echo "3/3 运行 +Manager+Entropy"
echo "=========================================="
python ablation_v3/train/train_v3_ablation.py \
    --mode full \
    --exp-name ablation_full_${EPISODES} \
    --episodes $EPISODES \
    --max-steps $MAX_STEPS

echo "✓ +Manager+Entropy完成"

# 4. 对比分析
echo ""
echo "=========================================="
echo "对比分析"
echo "=========================================="
python tools/compare_ablation_results.py \
    --baseline ablation_v3/results/ablation_baseline_${EPISODES} \
    --manager ablation_v3/results/ablation_manager_${EPISODES} \
    --full ablation_v3/results/ablation_full_${EPISODES} \
    --output ablation_v3/visualizations/ablation_study_${EPISODES}ep

echo ""
echo "=========================================="
echo "消融实验完成！"
echo "=========================================="
echo "结果保存在: ablation_v3/visualizations/ablation_study_${EPISODES}ep"
echo ""
echo "查看对比图:"
echo "  open ablation_v3/visualizations/ablation_study_${EPISODES}ep/ablation_comparison.png"
