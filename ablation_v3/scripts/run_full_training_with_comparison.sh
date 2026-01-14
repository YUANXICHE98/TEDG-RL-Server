#!/bin/bash

# 完整训练流程：从500ep到5000ep，包含所有阶段和对比分析
# 使用Manager约束

set -e

echo "========================================="
echo "V3完整训练流程 - 带Manager约束对比"
echo "========================================="
echo ""
echo "训练计划:"
echo "  Phase 1: Warmup (500→1000 episodes) - 2-3小时"
echo "  Phase 2: Transition (1000→3000 episodes) - 8-10小时"
echo "  Phase 3: Fine-tune (3000→5000 episodes) - 8-10小时"
echo "  总预计时间: 18-23小时"
echo ""

# 询问用户确认
read -p "是否开始完整训练流程？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消训练"
    exit 0
fi

# 记录开始时间
START_TIME=$(date +%s)
echo "开始时间: $(date)"
echo ""

# ========================================
# Phase 1: 完成Warmup阶段
# ========================================
echo "========================================="
echo "Phase 1: Warmup阶段 (500→1000 episodes)"
echo "========================================="
echo ""

bash ablation_v3/scripts/complete_warmup_1000.sh

if [ $? -ne 0 ]; then
    echo "❌ Warmup阶段失败"
    exit 1
fi

echo ""
echo "✅ Warmup阶段完成"
echo ""

# 分析Warmup结果
echo "分析Warmup结果..."
python tools/analyze_warmup_results.py \
    --result_dir ablation_v3/results/warmup_1000_with_manager \
    --output ablation_v3/visualizations/warmup_analysis/

# 对比Warmup阶段
echo "对比Warmup阶段（有/无Manager约束）..."
python tools/compare_with_without_manager.py \
    --baseline ablation_v3/results/warmup_1000 \
    --with_manager ablation_v3/results/warmup_1000_with_manager \
    --phase warmup \
    --output ablation_v3/visualizations/warmup_comparison/

echo ""
echo "Warmup阶段分析完成"
echo "按Enter继续到Transition阶段..."
read

# ========================================
# Phase 2: Transition阶段
# ========================================
echo "========================================="
echo "Phase 2: Transition阶段 (1000→3000 episodes)"
echo "========================================="
echo ""

bash ablation_v3/scripts/run_transition_3000.sh

if [ $? -ne 0 ]; then
    echo "❌ Transition阶段失败"
    exit 1
fi

echo ""
echo "✅ Transition阶段完成"
echo ""

# 分析Transition结果
echo "分析Transition结果..."
python tools/analyze_transition_results.py \
    --result_dir ablation_v3/results/transition_3000_with_manager \
    --output ablation_v3/visualizations/transition_analysis/

# 对比Transition阶段
echo "对比Transition阶段（有/无Manager约束）..."
python tools/compare_with_without_manager.py \
    --baseline ablation_v3/results/transition_3000 \
    --with_manager ablation_v3/results/transition_3000_with_manager \
    --phase transition \
    --output ablation_v3/visualizations/transition_comparison/

echo ""
echo "Transition阶段分析完成"
echo "按Enter继续到Fine-tune阶段..."
read

# ========================================
# Phase 3: Fine-tune阶段
# ========================================
echo "========================================="
echo "Phase 3: Fine-tune阶段 (3000→5000 episodes)"
echo "========================================="
echo ""

bash ablation_v3/scripts/run_finetune_5000.sh

if [ $? -ne 0 ]; then
    echo "❌ Fine-tune阶段失败"
    exit 1
fi

echo ""
echo "✅ Fine-tune阶段完成"
echo ""

# 分析Fine-tune结果
echo "分析Fine-tune结果..."
python tools/analyze_finetune_results.py \
    --result_dir ablation_v3/results/finetune_5000_with_manager \
    --output ablation_v3/visualizations/finetune_analysis/

# 对比Fine-tune阶段
echo "对比Fine-tune阶段（有/无Manager约束）..."
python tools/compare_with_without_manager.py \
    --baseline ablation_v3/results/finetune_5000 \
    --with_manager ablation_v3/results/finetune_5000_with_manager \
    --phase finetune \
    --output ablation_v3/visualizations/finetune_comparison/

# ========================================
# 完整对比分析
# ========================================
echo ""
echo "========================================="
echo "生成完整对比分析"
echo "========================================="
echo ""

python tools/compare_full_training.py \
    --baseline_warmup ablation_v3/results/warmup_1000 \
    --baseline_transition ablation_v3/results/transition_3000 \
    --baseline_finetune ablation_v3/results/finetune_5000 \
    --manager_warmup ablation_v3/results/warmup_1000_with_manager \
    --manager_transition ablation_v3/results/transition_3000_with_manager \
    --manager_finetune ablation_v3/results/finetune_5000_with_manager \
    --output ablation_v3/visualizations/full_comparison/

# 生成最终报告
echo "生成最终报告..."
python tools/generate_final_report.py \
    --output ablation_v3/FULL_TRAINING_RESULTS.md

# ========================================
# 完成
# ========================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "========================================="
echo "🎉 完整训练流程完成！"
echo "========================================="
echo ""
echo "总用时: ${HOURS}小时${MINUTES}分钟"
echo ""
echo "结果位置:"
echo "  - Warmup: ablation_v3/results/warmup_1000_with_manager/"
echo "  - Transition: ablation_v3/results/transition_3000_with_manager/"
echo "  - Fine-tune: ablation_v3/results/finetune_5000_with_manager/"
echo ""
echo "可视化位置:"
echo "  - 完整对比: ablation_v3/visualizations/full_comparison/"
echo "  - 各阶段分析: ablation_v3/visualizations/{warmup,transition,finetune}_analysis/"
echo ""
echo "报告位置:"
echo "  - 完整报告: ablation_v3/FULL_TRAINING_RESULTS.md"
echo ""
echo "查看报告: cat ablation_v3/FULL_TRAINING_RESULTS.md"
echo ""
