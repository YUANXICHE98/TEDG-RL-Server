#!/bin/bash

# TEDG-RL 扩展步数实验
# 基于 embedding 实验的 checkpoint，测试 MAX_STEPS=2000 的效果

echo "=============================================================================="
echo "                                                                            TEDG-RL 扩展步数实验启动"
echo "=============================================================================="
echo "                                                                            
实验配置：
  基础模型: results_embedding/checkpoints/best_model.pth
  最大步数: 2000 (原500步的4倍)
  Episodes: 1000 (快速验证)
  匹配方式: Embedding (余弦相似度)
  动态路由: 是
  动作掩码: 是
  4通道: 是

准备启动扩展实验...

"

# 设置环境变量
export TEDG_OUTPUT_DIR="results_extended_steps"
export TEDG_NUM_EPISODES=1000
export TEDG_MAX_STEPS=2000
export TEDG_VERBOSE_INTERVAL=50
export TEDG_USE_EMBEDDING=1
export TEDG_USE_MASK=1
export TEDG_DYNAMIC_TH=1
export TEDG_SINGLE_CHANNEL=0

# 创建输出目录
mkdir -p $TEDG_OUTPUT_DIR/{checkpoints,logs}

echo "----------------------------------------
实验: extended_steps - 扩展步数(2000)
  基础模型: results_embedding/checkpoints/best_model.pth
  最大步数: 2000
  Episodes: 1000
  嵌入匹配: 是
  输出目录: $TEDG_OUTPUT_DIR"

# 启动训练
echo "✓ 启动训练..."

nohup python -u train_confmatch.py \
    --resume results_embedding/checkpoints/best_model.pth \
    > $TEDG_OUTPUT_DIR/training.log 2>&1 &

PID=$!
echo "✓ PID: $PID"

echo ""
echo "=============================================================================="
echo "                                                                            扩展实验已启动！
=============================================================================="
echo "                                                                            
监控命令：
  查看实时日志:
    tail -f $TEDG_OUTPUT_DIR/training.log

  停止实验:
    kill $PID

输出目录：
  $TEDG_OUTPUT_DIR/ - 扩展步数实验结果

=============================================================================="
echo ""
