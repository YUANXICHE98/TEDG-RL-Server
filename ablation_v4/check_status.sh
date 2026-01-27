#!/bin/bash
# V4 训练状态检查脚本

echo "=========================================="
echo "V4 训练状态检查"
echo "=========================================="
echo ""

# 检查是否有训练进程在运行
echo "1. 检查运行中的训练进程..."
RUNNING=$(ps aux | grep -E "python.*train_v4" | grep -v grep)
if [ -z "$RUNNING" ]; then
    echo "   ❌ 没有V4训练进程在运行"
else
    echo "   ✓ 发现运行中的V4训练:"
    echo "$RUNNING"
fi
echo ""

# 检查结果目录
echo "2. 检查训练结果..."
if [ -d "ablation_v4/results" ]; then
    echo "   ✓ 结果目录存在"
    echo ""
    echo "   已完成的训练:"
    for dir in ablation_v4/results/*/; do
        if [ -d "$dir" ]; then
            exp_name=$(basename "$dir")
            if [ -f "${dir}logs/training_log.json" ]; then
                episodes=$(python3 -c "import json; data=json.load(open('${dir}logs/training_log.json')); print(len(data.get('episode_rewards', [])))" 2>/dev/null || echo "?")
                echo "     - $exp_name: $episodes episodes"
            else
                echo "     - $exp_name: (训练中或未完成)"
            fi
        fi
    done
else
    echo "   ❌ 结果目录不存在 (还没有运行过训练)"
fi
echo ""

# 检查最近的日志
echo "3. 检查最近的训练日志..."
if [ -f "ablation_v4/results/test_100ep.log" ]; then
    echo "   ✓ 找到 test_100ep.log"
    echo ""
    echo "   最后10行:"
    tail -10 ablation_v4/results/test_100ep.log
elif ls ablation_v4/results/*.log 2>/dev/null | head -1 > /dev/null; then
    latest_log=$(ls -t ablation_v4/results/*.log 2>/dev/null | head -1)
    echo "   ✓ 找到最新日志: $latest_log"
    echo ""
    echo "   最后10行:"
    tail -10 "$latest_log"
else
    echo "   ❌ 没有找到训练日志"
fi
echo ""

# 检查V4实现文件
echo "4. 检查V4实现文件..."
files=(
    "src/core/networks_v4_cross_attention.py"
    "ablation_v4/train/train_v4_cross_attention.py"
    "ablation_v4/test_v4_smoke.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ❌ $file (缺失)"
    fi
done
echo ""

# 给出建议
echo "=========================================="
echo "建议的下一步操作"
echo "=========================================="
echo ""

if [ -z "$RUNNING" ]; then
    if [ ! -d "ablation_v4/results/test_100ep" ]; then
        echo "✨ 还没有运行过V4训练"
        echo ""
        echo "建议操作:"
        echo "  1. 先运行烟雾测试:"
        echo "     python ablation_v4/test_v4_smoke.py"
        echo ""
        echo "  2. 然后运行小规模测试 (100 episodes):"
        echo "     bash ablation_v4/scripts/run_v4_test_100ep.sh"
    else
        echo "✅ V4训练已完成"
        echo ""
        echo "查看结果:"
        echo "  cat ablation_v4/results/test_100ep.log"
        echo ""
        echo "继续训练:"
        echo "  bash ablation_v4/scripts/run_warmup_1000.sh"
    fi
else
    echo "⏳ V4训练正在运行中..."
    echo ""
    echo "监控训练:"
    echo "  tail -f ablation_v4/results/test_100ep.log"
fi
echo ""
