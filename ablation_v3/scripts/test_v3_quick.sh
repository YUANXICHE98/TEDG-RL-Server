#!/bin/bash
# V3快速测试脚本 - CPU版本，验证代码可行性

echo "=========================================="
echo "V3 快速测试 (CPU版本)"
echo "=========================================="
echo ""
echo "配置:"
echo "  - 设备: CPU"
echo "  - Episodes: 10"
echo "  - Max Steps: 100"
echo "  - 目的: 验证代码可行性和收敛趋势"
echo ""
echo "=========================================="

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate tedg-rl-demo

# 强制使用CPU
export TEDG_FORCE_CPU=1
export PYTHONPATH=.

# 运行快速测试
python ablation_v3/train/train_v3_gat_moe.py \
  --exp-name v3_quick_test \
  --episodes 10 \
  --max-steps 100

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "检查结果:"
echo "  - 查看日志: ablation_v3/results/v3_quick_test/logs/training_log.json"
echo "  - 查看checkpoint: ablation_v3/results/v3_quick_test/checkpoints/"
echo ""

