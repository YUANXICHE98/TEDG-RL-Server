#!/bin/bash
# V3 vs V4 小规模对比测试 (100 episodes)
# 用于快速验证V4的改进效果
# 注意: V3结果已经存在，只运行V4训练

set -e

echo "=========================================="
echo "V3 vs V4 小规模对比测试 (100 episodes)"
echo "=========================================="
echo ""

# 创建结果目录
mkdir -p ablation_v4/results/test_100ep

# ==========================================
# 1. 检查V3结果 (使用已有数据)
# ==========================================

echo "=========================================="
echo "1. 检查V3结果 (使用已有数据)"
echo "=========================================="
echo ""

# 使用V3的warmup_1000前100个episode作为对比基准
V3_LOG="ablation_v3/results/warmup_1000/logs/training_log.json"

if [ -f "$V3_LOG" ]; then
    echo "✓ 找到V3训练日志: $V3_LOG"
    echo "  将使用前100个episode作为对比基准"
else
    echo "❌ 未找到V3训练日志: $V3_LOG"
    echo "   请先运行V3训练或检查路径"
    exit 1
fi

echo ""

# ==========================================
# 2. 运行V4训练 (100 episodes)
# ==========================================

echo "=========================================="
echo "2. 运行V4训练 (100 episodes)"
echo "=========================================="
echo ""

python ablation_v4/train/train_v4_cross_attention.py \
    --exp-name test_100ep \
    --episodes 100 \
    --max-steps 500 \
    --num-experts 4 \
    2>&1 | tee ablation_v4/results/test_100ep.log

echo ""
echo "✓ V4训练完成"
echo ""

# ==========================================
# 3. 对比分析
# ==========================================

echo "=========================================="
echo "3. 对比分析"
echo "=========================================="
echo ""

python ablation_v4/scripts/compare_v3_v4_100ep.py \
    --v3-log "$V3_LOG" \
    --v4-log ablation_v4/results/test_100ep/logs/training_log.json \
    --output ablation_v4/results/v3_v4_comparison_100ep.png

echo ""
echo "=========================================="
echo "✅ 对比测试完成！"
echo "=========================================="
echo ""
echo "结果保存在:"
echo "  - V3 (参考): $V3_LOG (前100 episodes)"
echo "  - V4 (新): ablation_v4/results/test_100ep/"
echo "  - 对比图: ablation_v4/results/v3_v4_comparison_100ep.png"
echo ""
