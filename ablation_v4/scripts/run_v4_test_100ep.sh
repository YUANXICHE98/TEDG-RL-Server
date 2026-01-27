#!/bin/bash
# V4 独立测试 (100 episodes)
# 快速验证V4实现是否正常工作

set -e

echo "=========================================="
echo "V4 独立测试 (100 episodes)"
echo "=========================================="
echo ""

# 创建结果目录
mkdir -p ablation_v4/results/test_100ep

# ==========================================
# 运行V4训练 (100 episodes)
# ==========================================

echo "开始V4训练..."
echo ""

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate tedg-rl-demo

python ablation_v4/train/train_v4_cross_attention.py \
    --exp-name test_100ep \
    --episodes 100 \
    --max-steps 500 \
    --num-experts 4 \
    2>&1 | tee ablation_v4/results/test_100ep.log

echo ""
echo "=========================================="
echo "✅ V4训练完成！"
echo "=========================================="
echo ""
echo "结果保存在:"
echo "  - 训练日志: ablation_v4/results/test_100ep/logs/training_log.json"
echo "  - 模型检查点: ablation_v4/results/test_100ep/checkpoints/"
echo "  - 完整日志: ablation_v4/results/test_100ep.log"
echo ""
echo "查看训练日志:"
echo "  cat ablation_v4/results/test_100ep.log"
echo ""
echo "查看最终统计:"
echo "  tail -50 ablation_v4/results/test_100ep.log"
echo ""
