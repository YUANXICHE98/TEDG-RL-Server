#!/bin/bash

# TEDG-RL V2 消融实验并行运行脚本
# 支持 Gumbel-Softmax 和 Sparse MoE 优化

echo "=============================================================================="
echo "                                                                            TEDG-RL V2 消融实验启动"
echo "=============================================================================="
echo "
实验配置：
  基线: embedding + 2000步 + 从零训练
  优化: Gumbel-Softmax硬路由 / Sparse MoE / Cross-Attention / Transformer-Block

准备启动 V2 消融实验...

"

# 创建输出目录
mkdir -p ablation_v2/results/{embedding2000,gumbel,sparse_moe,gumbel_sparse}

# 实验1: 基线 (embedding + 2000步)
echo "----------------------------------------
实验1: embedding2000 - 嵌入匹配 + 2000步 (新基线)
  Episodes: 3000
  最大步数: 2000
  输出目录: ablation_v2/results/embedding2000"

nohup python -u ablation_v2/train/train_v2.py \
    --exp-name embedding2000 \
    --episodes 3000 \
    --max-steps 2000 \
    --use-embedding \
    > ablation_v2/results/embedding2000/training.log 2>&1 &

PID1=$!
echo "✓ PID: $PID1"

# 实验2: Gumbel-Softmax 硬路由
echo "----------------------------------------
实验2: gumbel - Gumbel-Softmax硬路由
  Episodes: 3000
  最大步数: 2000
  Gumbel-Tau: 1.0
  输出目录: ablation_v2/results/gumbel"

nohup python -u ablation_v2/train/train_v2.py \
    --exp-name gumbel \
    --episodes 3000 \
    --max-steps 2000 \
    --use-embedding \
    --use-gumbel \
    --gumbel-tau 1.0 \
    > ablation_v2/results/gumbel/training.log 2>&1 &

PID2=$!
echo "✓ PID: $PID2"

# 实验3: Sparse MoE (Top-2)
echo "----------------------------------------
实验3: sparse_moe - Top-2稀疏专家
  Episodes: 3000
  最大步数: 2000
  Top-K: 2
  输出目录: ablation_v2/results/sparse_moe"

nohup python -u ablation_v2/train/train_v2.py \
    --exp-name sparse_moe \
    --episodes 3000 \
    --max-steps 2000 \
    --use-embedding \
    --use-gumbel \
    --sparse-topk 2 \
    > ablation_v2/results/sparse_moe/training.log 2>&1 &

PID3=$!
echo "✓ PID: $PID3"

# 实验4: Gumbel + Sparse MoE (Top-1)
echo "----------------------------------------
实验4: gumbel_sparse - Gumbel硬路由 + Top-1专家
  Episodes: 3000
  最大步数: 2000
  Top-K: 1
  输出目录: ablation_v2/results/gumbel_sparse"

nohup python -u ablation_v2/train/train_v2.py \
    --exp-name gumbel_sparse \
    --episodes 3000 \
    --max-steps 2000 \
    --use-embedding \
    --use-gumbel \
    --gumbel-tau 0.5 \
    --sparse-topk 1 \
    > ablation_v2/results/gumbel_sparse/training.log 2>&1 &

PID4=$!
echo "✓ PID: $PID4"

echo ""
echo "=============================================================================="
echo "                                                                            V2消融实验已启动！
=============================================================================="
echo "
监控命令：
  查看所有实验进度:
    tail -n 5 ablation_v2/results/*/training.log
    
  查看特定实验:
    tail -f ablation_v2/results/embedding2000/training.log
    
  停止所有实验:
    kill $PID1 $PID2 $PID3 $PID4

输出目录：
  ablation_v2/results/ - 所有V2实验结果
    embedding2000/ - 基线实验 (embedding + 2000步)
    gumbel/ - Gumbel-Softmax硬路由实验
    sparse_moe/ - Top-2稀疏专家实验
    gumbel_sparse/ - 组合优化实验
    hram_doc/ - H-RAM文档方案 (4Actor + 检索)
    hram_e2e/ - H-RAM端到端方案

=============================================================================="
echo ""

# ============================================================================
# H-RAM 实验组 (可选，需要更多显存)
# ============================================================================

echo ""
echo "是否启动 H-RAM 实验组? (需要额外显存)"
echo "手动启动命令:"
echo "  bash ablation_v2/scripts/run_hram_experiments.sh"
