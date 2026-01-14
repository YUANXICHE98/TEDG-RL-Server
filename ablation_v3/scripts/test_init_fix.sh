#!/bin/bash
# 测试初始化修正效果
# 快速训练50 episodes验证Alpha是否有变化

set -e

echo "=========================================="
echo "V3 初始化修正验证测试"
echo "=========================================="
echo ""

# 配置
EPISODES=50
MAX_STEPS=500
DEVICE="cpu"
OUTPUT_DIR="ablation_v3/results/init_fix_test"

# 回到项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "项目根目录: $PROJECT_ROOT"
echo ""

# 激活conda环境
echo "激活conda环境: tedg-rl-demo"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate tedg-rl-demo

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行训练
echo "开始训练 (Episodes: $EPISODES, Max Steps: $MAX_STEPS)..."
echo ""

python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name init_fix_test \
    --episodes $EPISODES \
    --max-steps $MAX_STEPS \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "关键检查点:"
echo "1. 查看 training.log 中的 DEBUG 输出，确认 Warmup 阶段使用 Softmax"
echo "2. 查看 Alpha 变化，是否从初始值有明显变化"
echo "3. 查看 Reward 曲线，是否比之前更快增长"
echo "4. 查看动作 logits，是否从 ~0.02 增长到 >0.1"
echo ""
echo "可视化命令:"
echo "  python tools/visualize_v3_episode.py --checkpoint $OUTPUT_DIR/checkpoint_ep50.pt"
echo ""
