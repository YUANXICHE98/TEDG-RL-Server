#!/bin/bash

# TEDG-RL V2 全量消融实验启动脚本
# 6组实验并行运行（共享GPU）

echo "=============================================================================="
echo "              TEDG-RL V2 消融实验 - 全量启动"
echo "=============================================================================="
echo "
实验配置 (7组并行):
  第一批 (MultiChannelPolicyNet - 5组):
    1. baseline      - 基线对照组 (Embedding + Mask)
    2. no_mask       - 无掩码对照组 (必须打败它!)
    3. gumbel        - Gumbel-Softmax硬路由
    4. sparse_moe    - Top-2稀疏专家
    5. gumbel_sparse - Gumbel + Top-1组合

  第二批 (H-RAM - 2组):
    6. hram_doc      - 文档方案 (4 Actors + 检索)
    7. hram_e2e      - 端到端方案

训练参数 (修复数学陷阱):
  Episodes: 50000 (最多5万，宁多勿少)
  Max Steps: 2000 (统一考试时间)
  Min Episodes: 10000 (约2000万步，充分训练)
  Patience: 5000 (连续5000轮无提升才停止)
  预计总步数: 2000万-1亿步/实验

"

# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 创建所有输出目录
echo "[准备] 创建输出目录..."
mkdir -p ablation_v2/results/{baseline,no_mask,gumbel,sparse_moe,gumbel_sparse,hram_doc,hram_e2e}/{checkpoints,logs}

echo "[准备] 目录创建完成"
echo ""

# ============================================================================
# 第一批: MultiChannelPolicyNet 实验
# ============================================================================

echo "=========================================="
echo "启动第一批实验 (MultiChannelPolicyNet)"
echo "=========================================="

# 实验1: baseline 基线对照组 (Embedding + Mask)
echo "[1/7] baseline - 启动中..."
nohup python -u ablation_v2/train/train_v2.py \
    --exp-name baseline \
    --episodes 50000 \
    --max-steps 2000 \
    --min-episodes 10000 \
    --patience 5000 \
    --use-embedding \
    > ablation_v2/results/baseline/training.log 2>&1 &
PID1=$!
echo "  PID: $PID1"

# 实验2: no_mask 无掩码对照组 (必须打败它!)
echo "[2/7] no_mask - 启动中..."
nohup python -u ablation_v2/train/train_v2.py \
    --exp-name no_mask \
    --episodes 50000 \
    --max-steps 2000 \
    --min-episodes 10000 \
    --patience 5000 \
    --use-embedding \
    --no-mask \
    > ablation_v2/results/no_mask/training.log 2>&1 &
PID2=$!
echo "  PID: $PID2"

# 实验3: gumbel
echo "[3/7] gumbel - 启动中..."
nohup python -u ablation_v2/train/train_v2.py \
    --exp-name gumbel \
    --episodes 50000 \
    --max-steps 2000 \
    --min-episodes 10000 \
    --patience 5000 \
    --use-embedding \
    --use-gumbel \
    --gumbel-tau 1.0 \
    > ablation_v2/results/gumbel/training.log 2>&1 &
PID3=$!
echo "  PID: $PID3"

# 实验4: sparse_moe
echo "[4/7] sparse_moe - 启动中..."
nohup python -u ablation_v2/train/train_v2.py \
    --exp-name sparse_moe \
    --episodes 50000 \
    --max-steps 2000 \
    --min-episodes 10000 \
    --patience 5000 \
    --use-embedding \
    --use-gumbel \
    --sparse-topk 2 \
    > ablation_v2/results/sparse_moe/training.log 2>&1 &
PID4=$!
echo "  PID: $PID4"

# 实验5: gumbel_sparse
echo "[5/7] gumbel_sparse - 启动中..."
nohup python -u ablation_v2/train/train_v2.py \
    --exp-name gumbel_sparse \
    --episodes 50000 \
    --max-steps 2000 \
    --min-episodes 10000 \
    --patience 5000 \
    --use-embedding \
    --use-gumbel \
    --gumbel-tau 0.5 \
    --sparse-topk 1 \
    > ablation_v2/results/gumbel_sparse/training.log 2>&1 &
PID5=$!
echo "  PID: $PID5"

# ============================================================================
# 第二批: H-RAM 实验
# ============================================================================

echo ""
echo "=========================================="
echo "启动第二批实验 (H-RAM)"
echo "=========================================="

# 实验6: hram_doc
echo "[6/7] hram_doc - 启动中..."
nohup python -u ablation_v2/train/train_hram_doc.py \
    --exp-name hram_doc \
    --episodes 50000 \
    --max-steps 2000 \
    --min-episodes 10000 \
    --patience 5000 \
    > ablation_v2/results/hram_doc/training.log 2>&1 &
PID6=$!
echo "  PID: $PID6"

# 实验7: hram_e2e
echo "[7/7] hram_e2e - 启动中..."
nohup python -u ablation_v2/train/train_hram.py \
    --exp-name hram_e2e \
    --episodes 50000 \
    --max-steps 2000 \
    --min-episodes 10000 \
    --patience 5000 \
    --embed-dim 3072 \
    > ablation_v2/results/hram_e2e/training.log 2>&1 &
PID7=$!
echo "  PID: $PID7"

# ============================================================================
# 保存 PID 供后续管理
# ============================================================================

echo ""
echo "$PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7" > ablation_v2/results/experiment_pids.txt

echo "=============================================================================="
echo "                    7组实验已全部启动！"
echo "=============================================================================="
echo "
实验进程 PID:
  baseline:       $PID1  (基线对照组)
  no_mask:        $PID2  (无掩码 - 必须打败它!)
  gumbel:         $PID3
  sparse_moe:     $PID4
  gumbel_sparse:  $PID5
  hram_doc:       $PID6
  hram_e2e:       $PID7

训练参数 (修复数学陷阱):
  Episodes: 50000 (最多5万)
  Max Steps: 2000 (统一考试时间)
  Min Episodes: 10000 (约2000万步)
  Patience: 5000 (连续5000轮无提升才停止)

监控命令:
  # 查看所有实验最新日志
  tail -n 3 ablation_v2/results/*/training.log

  # 实时跟踪某个实验
  tail -f ablation_v2/results/gumbel/training.log

  # 查看GPU使用
  nvidia-smi

  # 停止所有实验
  kill $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7

  # 训练完成后生成可视化
  python ablation_v2/visualize_v2_results.py

输出结构:
  ablation_v2/results/
  ├── baseline/         # 基线对照组
  ├── no_mask/          # 无掩码 (必须打败!)
  ├── gumbel/
  ├── sparse_moe/
  ├── gumbel_sparse/
  ├── hram_doc/
  └── hram_e2e/

预计运行时间: 8-24小时 (取决于GPU)

=============================================================================="
