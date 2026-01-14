#!/bin/bash
# V3完整训练启动脚本
# 包含Manager约束和所有高级机制

set -e

echo "========================================================================"
echo "V3完整训练 - Manager约束 + 高级机制"
echo "========================================================================"
echo ""

# 检查环境
if ! conda env list | grep -q "tedg-rl-demo"; then
    echo "❌ 错误: conda环境 'tedg-rl-demo' 不存在"
    echo "请先创建环境: conda create -n tedg-rl-demo python=3.9"
    exit 1
fi

echo "✓ 环境检查通过"
echo ""

# 激活环境
echo "激活conda环境..."
eval "$(conda shell.bash hook)"
conda activate tedg-rl-demo

# 检查代码
echo "检查代码实现..."
python ablation_v3/diagnose_manager_constraints.py

echo ""
echo "========================================================================"
echo "开始训练"
echo "========================================================================"
echo ""
echo "配置:"
echo "  - Episodes: 5000"
echo "  - Max steps: 2000"
echo "  - 实验名称: v3_full_mechanisms_$(date +%Y%m%d_%H%M%S)"
echo ""
echo "预期时间: ~20-30小时（CPU）"
echo ""
echo "阶段:"
echo "  - Warmup (0-1000): 探索，学习基础策略"
echo "  - Transition (1000-3000): 平滑过渡，开始专业化"
echo "  - Fine-tune (3000-5000): 极致专业化"
echo ""
echo "预期效果:"
echo "  - Alpha熵: 1.38 → 0.2-0.3"
echo "  - 平均分数: 12 → 20-25"
echo ""

read -p "按Enter开始训练，或Ctrl+C取消..."

# 开始训练
EXP_NAME="v3_full_mechanisms_$(date +%Y%m%d_%H%M%S)"

python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name "$EXP_NAME" \
    --episodes 5000 \
    --max-steps 2000 \
    2>&1 | tee "ablation_v3/results/${EXP_NAME}/training_full.log"

echo ""
echo "========================================================================"
echo "训练完成！"
echo "========================================================================"
echo ""
echo "结果保存在: ablation_v3/results/${EXP_NAME}/"
echo ""
echo "下一步:"
echo "  1. 查看训练日志: cat ablation_v3/results/${EXP_NAME}/training.log"
echo "  2. 可视化结果: python tools/visualize_three_phases.py --exp-name ${EXP_NAME}"
echo "  3. 分析专家激活: python tools/analyze_expert_activation.py --exp-name ${EXP_NAME}"
echo ""
