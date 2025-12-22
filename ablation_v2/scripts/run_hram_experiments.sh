#!/bin/bash

# TEDG-RL V3 H-RAM 实验运行脚本
# 两种方案对比：文档方案 vs 端到端方案

echo "=============================================================================="
echo "                    TEDG-RL H-RAM 架构对比实验"
echo "=============================================================================="
echo "
两种 H-RAM 实现方案对比：

1. 文档方案 (hram_doc):
   - 保留4个Actor专家
   - 检索上下文压缩后分发给各专家
   - Gumbel-Softmax硬路由

2. 端到端方案 (hram_e2e):
   - 完全端到端，无4个Actor
   - 直接从检索上下文生成动作
   - 更简洁的架构

"

# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建输出目录
mkdir -p ablation_v2/results/{hram_doc,hram_e2e}

# 实验1: H-RAM 文档方案 (4 Actors + 检索上下文)
echo "----------------------------------------
实验1: hram_doc - 文档方案 (4 Actors + 检索)
  Episodes: 3000
  最大步数: 2000
  架构: StateEncoder → Memory → Compress → 4 Actors → Gumbel → Action
  输出目录: ablation_v2/results/hram_doc"

nohup python -u ablation_v2/train/train_hram_doc.py \
    --exp-name hram_doc \
    --episodes 3000 \
    --max-steps 2000 \
    > ablation_v2/results/hram_doc/training.log 2>&1 &

PID1=$!
echo "✓ PID: $PID1"

# 实验2: H-RAM 端到端方案
echo "----------------------------------------
实验2: hram_e2e - 端到端方案
  Episodes: 3000
  最大步数: 2000
  架构: StateEncoder → Memory → CrossAttention → Action
  输出目录: ablation_v2/results/hram_e2e"

nohup python -u ablation_v2/train/train_hram.py \
    --exp-name hram_e2e \
    --episodes 3000 \
    --max-steps 2000 \
    --embed-dim 3072 \
    > ablation_v2/results/hram_e2e/training.log 2>&1 &

PID2=$!
echo "✓ PID: $PID2"

echo ""
echo "=============================================================================="
echo "                    H-RAM 对比实验已启动！"
echo "=============================================================================="
echo "
监控命令：
  查看所有实验进度:
    tail -n 5 ablation_v2/results/hram_*/training.log
    
  查看特定实验:
    tail -f ablation_v2/results/hram_doc/training.log
    tail -f ablation_v2/results/hram_e2e/training.log
    
  停止所有实验:
    kill $PID1 $PID2

输出目录：
  ablation_v2/results/hram_doc/ - 文档方案 (4 Actors)
  ablation_v2/results/hram_e2e/ - 端到端方案

架构对比：
  hram_doc: State → Encode → Retrieve → Compress → 4 Actors → Gumbel → Action
  hram_e2e: State → Encode → Retrieve → CrossAttn → Action

=============================================================================="
echo ""
