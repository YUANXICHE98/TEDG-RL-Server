#!/bin/bash

# TEDG-RL V2 训练启动脚本（解决导入路径问题）

# 设置 PYTHONPATH 以包含项目根目录
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练脚本
python ablation_v2/train/train_v2.py "$@"
