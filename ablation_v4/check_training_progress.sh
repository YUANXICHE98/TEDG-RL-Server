#!/bin/bash
# 检查V4训练进度

echo "=========================================="
echo "V4训练进度监控"
echo "=========================================="
echo ""

# 检查运行中的进程
echo "📊 运行中的训练进程:"
ps aux | grep "train_v4_cross_attention" | grep -v grep | awk '{print "  PID:", $2, "| CPU:", $3"%", "| 内存:", $4"%", "| 运行时间:", $10}'
echo ""

# 检查Warmup 1000结果
if [ -d "ablation_v4/results/warmup_1000" ]; then
    echo "📁 Warmup 1000 训练状态:"
    
    # 检查checkpoints
    if [ -d "ablation_v4/results/warmup_1000/checkpoints" ]; then
        CHECKPOINT_COUNT=$(ls ablation_v4/results/warmup_1000/checkpoints/*.pth 2>/dev/null | wc -l)
        echo "  ✓ Checkpoints: $CHECKPOINT_COUNT 个"
        ls -lh ablation_v4/results/warmup_1000/checkpoints/ 2>/dev/null | tail -5
    fi
    
    # 检查训练日志
    if [ -f "ablation_v4/results/warmup_1000/logs/training_log.json" ]; then
        echo ""
        echo "  ✓ 训练统计:"
        python3 -c "
import json
try:
    with open('ablation_v4/results/warmup_1000/logs/training_log.json', 'r') as f:
        log = json.load(f)
    episodes = len(log['episode_rewards'])
    best_score = log.get('best_score', 0)
    best_reward = log.get('best_reward', 0)
    avg_score = sum(log['episode_scores']) / len(log['episode_scores']) if log['episode_scores'] else 0
    print(f'    - 已完成Episodes: {episodes}/1000')
    print(f'    - 最佳分数: {best_score}')
    print(f'    - 最佳奖励: {best_reward:.2f}')
    print(f'    - 平均分数: {avg_score:.1f}')
except Exception as e:
    print(f'    ⚠️ 无法读取日志: {e}')
"
    fi
    echo ""
else
    echo "⚠️ Warmup 1000 训练尚未开始"
    echo ""
fi

# 检查日志文件
if [ -f "ablation_v4/results/warmup_1000.log" ]; then
    echo "📝 最新训练日志 (最后20行):"
    tail -20 ablation_v4/results/warmup_1000.log
fi

echo ""
echo "=========================================="
