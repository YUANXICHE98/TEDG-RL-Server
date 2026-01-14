#!/bin/bash
# V3 GPU大规模训练脚本

echo "=========================================="
echo "V3 GPU大规模训练"
echo "=========================================="
echo ""
echo "配置:"
echo "  - 设备: GPU (自动检测)"
echo "  - Episodes: 10000"
echo "  - Max Steps: 2000"
echo "  - 三阶段训练:"
echo "    * Warmup (0-1000): Softmax"
echo "    * Transition (1000-3000): 温度退火"
echo "    * Fine-tune (3000-10000): Sparsemax"
echo ""
echo "=========================================="

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate tedg-rl-demo

# 设置PYTHONPATH
export PYTHONPATH=.

# 运行完整训练
python ablation_v3/train/train_v3_gat_moe.py \
  --exp-name v3_full_training \
  --episodes 10000 \
  --max-steps 2000

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "结果位置: ablation_v3/results/v3_full_training/"
echo ""

