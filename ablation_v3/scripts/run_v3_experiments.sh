#!/bin/bash
# V3实验启动脚本

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tedg-rl-demo

# 设置PYTHONPATH
export PYTHONPATH=.

# 创建结果目录
mkdir -p ablation_v3/results

echo "=========================================="
echo "V3 实验启动"
echo "=========================================="

# 实验1: 完整V3 (Sparsemax + GAT + 4专家)
echo ""
echo "[实验1] V3 Full - Sparsemax + GAT + 4专家"
python ablation_v3/train/train_v3_gat_moe.py \
  --exp-name v3_full \
  --episodes 10000 \
  --max-steps 2000 \
  --num-experts 4

# 实验2: 固定GAT (验证GAT贡献)
echo ""
echo "[实验2] V3 Fixed GAT - 冻结GAT参数"
python ablation_v3/train/train_v3_gat_moe.py \
  --exp-name v3_fixed_gat \
  --episodes 10000 \
  --max-steps 2000 \
  --num-experts 4 \
  --freeze-gat

# 实验3: 2个专家 (验证专家数量)
echo ""
echo "[实验3] V3 2 Experts - 减少专家数量"
python ablation_v3/train/train_v3_gat_moe.py \
  --exp-name v3_2experts \
  --episodes 10000 \
  --max-steps 2000 \
  --num-experts 2

# 实验4: 无动作掩码 (验证掩码贡献)
echo ""
echo "[实验4] V3 No Mask - 禁用动作掩码"
python ablation_v3/train/train_v3_gat_moe.py \
  --exp-name v3_no_mask \
  --episodes 10000 \
  --max-steps 2000 \
  --num-experts 4 \
  --no-mask

echo ""
echo "=========================================="
echo "所有实验已启动"
echo "=========================================="

