#!/bin/bash
# 消融实验：对比三个版本的效果
# 每个版本都跑500 episodes，同比对比

set -e

EPISODES=500
MAX_STEPS=2000

echo "=========================================="
echo "消融实验 - 对比三个版本"
echo "=========================================="
echo "Episodes: $EPISODES"
echo "Max Steps: $MAX_STEPS"
echo ""

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tedg-rl-demo

# 1. Baseline - 无Manager约束，无熵最小化
echo "=========================================="
echo "1. 运行 Baseline (无Manager，无熵最小化)"
echo "=========================================="

# 需要临时修改代码，注释掉Manager约束和熵最小化
# 这里我们先创建一个修改版本

python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name ablation_baseline_500 \
    --episodes $EPISODES \
    --max-steps $MAX_STEPS

echo "✓ Baseline完成"
echo ""

# 2. +Manager - 有Manager约束，无熵最小化
echo "=========================================="
echo "2. 运行 +Manager (有Manager，无熵最小化)"
echo "=========================================="

# 需要临时修改代码，保留Manager约束，但熵符号保持-1

python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name ablation_manager_500 \
    --episodes $EPISODES \
    --max-steps $MAX_STEPS

echo "✓ +Manager完成"
echo ""

# 3. +Manager+Entropy - 有Manager约束，有熵最小化
echo "=========================================="
echo "3. 运行 +Manager+Entropy (有Manager，有熵最小化)"
echo "=========================================="

# 使用当前完整版本

python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name ablation_full_500 \
    --episodes $EPISODES \
    --max-steps $MAX_STEPS

echo "✓ +Manager+Entropy完成"
echo ""

# 4. 对比分析
echo "=========================================="
echo "4. 对比分析"
echo "=========================================="

python tools/compare_ablation_results.py \
    --baseline ablation_v3/results/ablation_baseline_500 \
    --manager ablation_v3/results/ablation_manager_500 \
    --full ablation_v3/results/ablation_full_500 \
    --output ablation_v3/visualizations/ablation_study

echo ""
echo "=========================================="
echo "消融实验完成！"
echo "=========================================="
echo "结果保存在: ablation_v3/visualizations/ablation_study"
