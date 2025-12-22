#!/bin/bash
# TEDG-RL NetHack训练启动脚本

echo "=================================="
echo "TEDG-RL NetHack训练"
echo "=================================="
echo ""
echo "启动训练..."
nohup python train_nethack.py > train_nethack.log 2>&1 &
PID=$!
echo "训练进程ID: $PID"
echo "日志文件: train_nethack.log"
echo ""
echo "监控命令:"
echo "  tail -f train_nethack.log          # 实时查看日志"
echo "  watch -n 5 'tail -20 train_nethack.log'  # 每5秒刷新"
echo "  ps aux | grep train_nethack        # 查看进程状态"
echo "  kill $PID                          # 停止训练"
echo ""
echo "训练已在后台启动！"
