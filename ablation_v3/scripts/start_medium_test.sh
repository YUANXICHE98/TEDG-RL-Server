#!/bin/bash
# V3中期测试脚本（500 episodes）
# 用于快速验证Manager约束和高级机制的效果

set -e

echo "========================================================================"
echo "V3中期测试 - 500 Episodes"
echo "========================================================================"
echo ""

# 激活环境
eval "$(conda shell.bash hook)"
conda activate tedg-rl-demo

echo "配置:"
echo "  - Episodes: 500"
echo "  - Max steps: 2000"
echo "  - 预期时间: ~2-3小时"
echo ""
echo "目标:"
echo "  - 完成Warmup阶段"
echo "  - 进入Transition阶段早期"
echo "  - 观察高级机制开始发挥作用"
echo ""

read -p "按Enter开始测试，或Ctrl+C取消..."

EXP_NAME="v3_medium_test_$(date +%Y%m%d_%H%M%S)"

python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name "$EXP_NAME" \
    --episodes 500 \
    --max-steps 2000

echo ""
echo "========================================================================"
echo "测试完成！"
echo "========================================================================"
echo ""
echo "结果: ablation_v3/results/${EXP_NAME}/"
echo ""
echo "检查要点:"
echo "  1. Alpha熵是否开始下降（从1.38降到1.0-1.1）"
echo "  2. Manager约束loss是否正常"
echo "  3. 高级机制是否开始工作（Transition阶段）"
echo ""
