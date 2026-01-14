#!/bin/bash
# 新版本中期测试 - 500 episodes
# 包含所有4个机制：Manager约束、熵最小化、时间一致性、重叠惩罚

set -e

echo "=========================================="
echo "新版本V3训练 - 500 Episodes中期测试"
echo "=========================================="
echo ""
echo "包含机制："
echo "  ✓ Manager内层约束（超图-路由对齐）"
echo "  ✓ 熵最小化（Fine-tune阶段符号反转）"
echo "  ✓ 时间一致性（减少意图震荡）"
echo "  ✓ 专家重叠惩罚（强制正交）"
echo ""
echo "预期效果："
echo "  - Alpha熵: 1.38 → 1.0-1.1 (下降20-30%)"
echo "  - 平均分数: 提升30-50%"
echo ""
echo "预计时间: 2-3小时（CPU）"
echo "=========================================="
echo ""

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tedg-rl-demo

# 设置实验名称（带时间戳）
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_NAME="v3_new_version_500ep_${TIMESTAMP}"

echo "实验名称: ${EXP_NAME}"
echo "开始时间: $(date)"
echo ""

# 运行训练
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name "${EXP_NAME}" \
    --episodes 500 \
    --max-steps 2000 \
    2>&1 | tee "ablation_v3/results/${EXP_NAME}/training_output.log"

echo ""
echo "=========================================="
echo "训练完成！"
echo "结束时间: $(date)"
echo "=========================================="
echo ""
echo "结果保存在: ablation_v3/results/${EXP_NAME}/"
echo ""
echo "查看结果："
echo "  tail -100 ablation_v3/results/${EXP_NAME}/training_output.log"
echo ""
echo "下一步："
echo "  1. 检查Alpha熵是否下降"
echo "  2. 检查Manager约束loss"
echo "  3. 如果效果好，运行完整5000 episodes"
echo ""
