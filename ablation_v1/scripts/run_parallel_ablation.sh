#!/bin/bash
################################################################################
# 并行消融实验启动脚本
# 
# 功能：同时启动多个 train_confmatch.py 实例，每个使用不同的输出目录
# 
# 使用方法：
#   bash run_parallel_ablation.sh
#
# 注意：
#   - 每个实验会在后台运行，输出到独立的日志文件
#   - 可以通过 tail -f 查看实时日志
#   - 使用 ps aux | grep train_confmatch 查看运行状态
#   - 使用 pkill -f train_confmatch 停止所有实验
################################################################################

set -e

echo "================================================================================"
echo "TEDG-RL 并行消融实验启动"
echo "================================================================================"

# 配置
NUM_EPISODES=${TEDG_NUM_EPISODES:-10000}
MAX_STEPS=${TEDG_MAX_STEPS:-1000}
VERBOSE_INTERVAL=${TEDG_VERBOSE_INTERVAL:-10}

echo ""
echo "训练配置："
echo "  总 Episodes: $NUM_EPISODES"
echo "  每轮最大步数: $MAX_STEPS"
echo "  详细日志间隔: 每 $VERBOSE_INTERVAL 个 episode"
echo ""

# 实验列表 (名称:输出目录:USE_EMBEDDING:USE_MASK:DYNAMIC_TH:SINGLE_CH:描述)
experiments=(
    "full:results_full:0:1:1:0:完整版（4通道+掩码+动态阈值）"
    "no_mask:results_no_mask:0:0:1:0:无动作掩码"
    "fixed_th:results_fixed_th:0:1:0:0:固定阈值（0.3/0.7）"
    "single_ch:results_single_ch:0:1:1:1:单通道（只用pre）"
    "embedding:results_embedding:1:1:1:0:嵌入匹配（余弦相似度）"
)

echo "准备启动 ${#experiments[@]} 个并行实验："
echo ""

# 启动每个实验
for exp in "${experiments[@]}"; do
    IFS=':' read -r name output_dir use_embedding use_mask dynamic_th single_ch description <<< "$exp"
    
    echo "----------------------------------------"
    echo "实验: $name - $description"
    echo "  4通道: $( [ "$single_ch" == "0" ] && echo "是" || echo "否(只用pre)" )"
    echo "  动态路由: $( [ "$dynamic_th" == "1" ] && echo "是" || echo "否" )"
    echo "  动作掩码: $( [ "$use_mask" == "1" ] && echo "是" || echo "否" )"
    echo "  嵌入匹配: $( [ "$use_embedding" == "1" ] && echo "是" || echo "否" )"
    echo "  输出目录: $output_dir"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 启动训练（后台运行）
    TEDG_OUTPUT_DIR="$output_dir" \
    TEDG_NUM_EPISODES="$NUM_EPISODES" \
    TEDG_MAX_STEPS="$MAX_STEPS" \
    TEDG_VERBOSE_INTERVAL="$VERBOSE_INTERVAL" \
    TEDG_USE_EMBEDDING="$use_embedding" \
    TEDG_USE_MASK="$use_mask" \
    TEDG_DYNAMIC_TH="$dynamic_th" \
    TEDG_SINGLE_CHANNEL="$single_ch" \
    nohup python -u train_confmatch.py > "${output_dir}/training.log" 2>&1 &
    
    pid=$!
    echo "✓ PID: $pid"
    echo ""
    
    sleep 2
done

echo "================================================================================"
echo "所有实验已启动！"
echo "================================================================================"
echo ""
echo "监控命令："
echo "  查看所有运行的实验:"
echo "    ps aux | grep 'train_.*\.py' | grep -v grep"
echo ""
echo "  查看实时日志:"
echo "    tail -f results_full/training.log"
echo "    tail -f results_no_mask/training.log"
echo "    tail -f results_fixed_th/training.log"
echo "    tail -f results_single_ch/training.log"
echo "    tail -f results_embedding/training.log"
echo ""
echo "  停止所有实验:"
echo "    pkill -f 'python train_confmatch.py'"
echo ""
echo "  查看GPU使用情况:"
echo "    nvidia-smi"
echo "    watch -n 1 nvidia-smi"
echo ""
echo "输出目录："
for exp in "${experiments[@]}"; do
    IFS=':' read -r name output_dir use_embedding use_mask dynamic_th single_ch description <<< "$exp"
    echo "  $output_dir/ - $description"
done
echo ""
echo "================================================================================"
