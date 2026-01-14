#!/bin/bash
# V3收敛测试 - CPU版本，小数据量验证收敛趋势

echo "=========================================="
echo "V3 收敛测试 (CPU - 小数据量)"
echo "=========================================="
echo ""
echo "配置:"
echo "  - 设备: CPU"
echo "  - Episodes: 50"
echo "  - Max Steps: 200"
echo "  - 目的: 验证收敛趋势"
echo ""
echo "预期:"
echo "  - Episode 1-10: 探索阶段，分数波动"
echo "  - Episode 10-30: 开始学习，分数上升"
echo "  - Episode 30-50: 稳定提升，显示收敛趋势"
echo ""
echo "=========================================="

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate tedg-rl-demo

# 强制使用CPU
export TEDG_FORCE_CPU=1
export PYTHONPATH=.

# 运行收敛测试
python ablation_v3/train/train_v3_gat_moe.py \
  --exp-name v3_convergence_cpu \
  --episodes 50 \
  --max-steps 200

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "分析结果:"
echo "  1. 查看训练曲线:"
echo "     python -c \"import json; data=json.load(open('ablation_v3/results/v3_convergence_cpu/logs/training_log.json')); print('Scores:', data['episode_scores'])\""
echo ""
echo "  2. 检查收敛趋势:"
echo "     - 前10个: 应该较低且波动"
echo "     - 中间20个: 应该逐渐上升"
echo "     - 最后20个: 应该稳定在较高水平"
echo ""
echo "  3. 检查α熵:"
echo "     python -c \"import json; data=json.load(open('ablation_v3/results/v3_convergence_cpu/logs/training_log.json')); print('Alpha entropy:', [round(x,3) for x in data['monitor_metrics']['alpha_entropy'][-10:]])\""
echo ""

