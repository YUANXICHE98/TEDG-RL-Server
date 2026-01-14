#!/bin/bash
# 快速对比测试：验证Manager约束是否有效
# 运行100 episodes对比baseline vs with_manager_constraints

set -e

echo "========================================================================"
echo "快速对比测试：Baseline vs Manager约束"
echo "========================================================================"
echo ""
echo "测试配置："
echo "  - Episodes: 100"
echo "  - Max steps: 2000"
echo "  - 对比指标: Alpha熵、平均分数、专家使用方差"
echo ""

# 激活环境
source ~/miniconda3/bin/activate tedg-rl-demo

# 创建结果目录
RESULT_DIR="ablation_v3/results/quick_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

echo "结果将保存到: $RESULT_DIR"
echo ""

# ========================================================================
# 实验1：Baseline（无Manager约束）
# ========================================================================

echo "========================================================================"
echo "实验1：Baseline（无Manager约束）"
echo "========================================================================"
echo ""

# 临时修改训练脚本，禁用Manager约束
# 创建一个临时的baseline版本
cp ablation_v3/train/train_v3_gat_moe.py ablation_v3/train/train_v3_gat_moe_backup.py

# 使用sed禁用Manager约束（将系数设为0）
sed -i.bak "s/'alignment_coef': 0.1/'alignment_coef': 0.0/g" ablation_v3/train/train_v3_gat_moe.py
sed -i.bak "s/'semantic_coef': 0.05/'semantic_coef': 0.0/g" ablation_v3/train/train_v3_gat_moe.py

echo "运行Baseline训练（100 episodes）..."
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name quick_baseline \
    --episodes 100 \
    --max-steps 2000 \
    2>&1 | tee "$RESULT_DIR/baseline_log.txt"

# 恢复原始文件
mv ablation_v3/train/train_v3_gat_moe_backup.py ablation_v3/train/train_v3_gat_moe.py
rm -f ablation_v3/train/train_v3_gat_moe.py.bak

echo ""
echo "✓ Baseline训练完成"
echo ""

# ========================================================================
# 实验2：With Manager约束
# ========================================================================

echo "========================================================================"
echo "实验2：With Manager约束"
echo "========================================================================"
echo ""

echo "运行Manager约束训练（100 episodes）..."
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name quick_manager \
    --episodes 100 \
    --max-steps 2000 \
    2>&1 | tee "$RESULT_DIR/manager_log.txt"

echo ""
echo "✓ Manager约束训练完成"
echo ""

# ========================================================================
# 分析结果
# ========================================================================

echo "========================================================================"
echo "分析结果"
echo "========================================================================"
echo ""

# 提取关键指标
echo "提取Baseline指标..."
BASELINE_FINAL_ALPHA=$(grep "α_entropy" "$RESULT_DIR/baseline_log.txt" | tail -1 | grep -oP 'α_entropy: \K[0-9.]+' || echo "N/A")
BASELINE_FINAL_SCORE=$(grep "Score:" "$RESULT_DIR/baseline_log.txt" | tail -10 | grep -oP 'Score: \K[0-9]+' | awk '{sum+=$1} END {print sum/NR}' || echo "N/A")

echo "提取Manager指标..."
MANAGER_FINAL_ALPHA=$(grep "α_entropy" "$RESULT_DIR/manager_log.txt" | tail -1 | grep -oP 'α_entropy: \K[0-9.]+' || echo "N/A")
MANAGER_FINAL_SCORE=$(grep "Score:" "$RESULT_DIR/manager_log.txt" | tail -10 | grep -oP 'Score: \K[0-9]+' | awk '{sum+=$1} END {print sum/NR}' || echo "N/A")
MANAGER_ALIGNMENT=$(grep "Alignment=" "$RESULT_DIR/manager_log.txt" | tail -1 | grep -oP 'Alignment=\K[0-9.]+' || echo "N/A")

echo ""
echo "========================================================================"
echo "对比结果"
echo "========================================================================"
echo ""
echo "指标                  | Baseline      | With Manager  | 改进"
echo "---------------------+---------------+---------------+---------------"
echo "Alpha熵（终态）      | $BASELINE_FINAL_ALPHA         | $MANAGER_FINAL_ALPHA         | $(python3 -c "try: print(f'{(float('$BASELINE_FINAL_ALPHA') - float('$MANAGER_FINAL_ALPHA')) / float('$BASELINE_FINAL_ALPHA') * 100:.1f}%')
except: print('N/A')" 2>/dev/null || echo 'N/A')"
echo "平均分数（最后10ep） | $BASELINE_FINAL_SCORE         | $MANAGER_FINAL_SCORE         | $(python3 -c "try: print(f'{(float('$MANAGER_FINAL_SCORE') - float('$BASELINE_FINAL_SCORE')) / float('$BASELINE_FINAL_SCORE') * 100:.1f}%')
except: print('N/A')" 2>/dev/null || echo 'N/A')"
echo "Alignment Loss       | N/A           | $MANAGER_ALIGNMENT         | -"
echo ""

# 保存结果摘要
cat > "$RESULT_DIR/summary.txt" << EOF
快速对比测试结果
================

测试时间: $(date)
Episodes: 100
Max Steps: 2000

结果对比
--------

Baseline（无Manager约束）:
  - 最终Alpha熵: $BASELINE_FINAL_ALPHA
  - 平均分数（最后10ep）: $BASELINE_FINAL_SCORE

With Manager约束:
  - 最终Alpha熵: $MANAGER_FINAL_ALPHA
  - 平均分数（最后10ep）: $MANAGER_FINAL_SCORE
  - Alignment Loss: $MANAGER_ALIGNMENT

结论
----

$(python3 -c "
try:
    baseline_alpha = float('$BASELINE_FINAL_ALPHA')
    manager_alpha = float('$MANAGER_FINAL_ALPHA')
    baseline_score = float('$BASELINE_FINAL_SCORE')
    manager_score = float('$MANAGER_FINAL_SCORE')
    
    alpha_improve = (baseline_alpha - manager_alpha) / baseline_alpha * 100
    score_improve = (manager_score - baseline_score) / baseline_score * 100
    
    if alpha_improve > 10 or score_improve > 10:
        print('✓ Manager约束显著改善了训练效果！')
        print(f'  - Alpha熵降低: {alpha_improve:.1f}%')
        print(f'  - 分数提升: {score_improve:.1f}%')
        print('  建议：继续进行完整训练（1000-5000 episodes）')
    elif alpha_improve > 0 or score_improve > 0:
        print('⚠ Manager约束有轻微改善，但不明显')
        print(f'  - Alpha熵降低: {alpha_improve:.1f}%')
        print(f'  - 分数提升: {score_improve:.1f}%')
        print('  建议：调整系数或增加训练episodes')
    else:
        print('✗ Manager约束未显示改善')
        print('  建议：检查实现或调整超参数')
except:
    print('无法计算改进（数据不完整）')
" 2>/dev/null || echo "无法计算改进（数据不完整）")

详细日志
--------
Baseline: $RESULT_DIR/baseline_log.txt
Manager:  $RESULT_DIR/manager_log.txt
EOF

cat "$RESULT_DIR/summary.txt"

echo ""
echo "========================================================================"
echo "测试完成！"
echo "========================================================================"
echo ""
echo "结果已保存到: $RESULT_DIR/summary.txt"
echo ""
echo "下一步："
echo "  1. 如果Manager约束有效 → 运行完整训练（1000-5000 episodes）"
echo "  2. 如果效果不明显 → 调整系数或检查实现"
echo ""
