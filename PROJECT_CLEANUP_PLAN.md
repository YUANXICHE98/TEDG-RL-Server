# 项目清理计划 - Project Cleanup Plan

## 清理目标
1. 删除过时的、重复的文档
2. 删除临时测试文件和脚本
3. 整理results目录，只保留关键训练结果
4. 保留核心代码和最终可视化结果

---

## 📁 ablation_v3/ 目录清理

### ✅ 保留的核心文档（10个）

#### 主要说明文档
1. **README.md** - 项目主文档
2. **PAPER_FIGURES_READY.md** - 论文图表说明（最终版）
3. **500EP_COMPARISON_RESULTS.md** - 训练对比结果
4. **EXPERT_ORTHOGONALITY_FIGURES_README.md** - 专家正交性图表说明

#### 训练相关
5. **TRAINING_PHASES_EXPLAINED.md** - 训练阶段说明
6. **MANAGER_CONSTRAINT_QUICK_REF.md** - Manager约束快速参考

#### 可视化相关
7. **VISUALIZATION_GUIDE.md** - 可视化指南
8. **RESULTS_INTERPRETATION_GUIDE.md** - 结果解读指南

#### 中文文档（保留2个最重要的）
9. **论文图表已完成.md** - 论文图表完成总结
10. **专家正交性可视化说明.md** - 专家正交性说明

### ❌ 删除的文档（40+个）

#### 过时的实现文档
- [ ] ADVANCED_MECHANISMS_IMPLEMENTATION_PLAN.md（已完成，有COMPLETE版本）
- [ ] ADVANCED_MECHANISMS_IMPLEMENTATION_COMPLETE.md（内容已整合）
- [ ] IMPLEMENTATION_COMPLETE_REPORT.md（重复）
- [ ] FINAL_IMPLEMENTATION_README.md（重复）
- [ ] V3_COMPLETE_IMPLEMENTATION_SUMMARY.md（重复）
- [ ] MANAGER_CONSTRAINT_IMPLEMENTATION.md（有COMPLETE版本）
- [ ] MANAGER_CONSTRAINT_IMPLEMENTATION_COMPLETE.md（内容已整合）

#### 过时的分析文档
- [ ] CONVERGENCE_TEST_ANALYSIS.md（早期测试）
- [ ] INIT_FIX_SUMMARY.md（早期修复）
- [ ] INIT_FIX_TEST_RESULTS.md（早期测试）
- [ ] QUICK_TEST_RESULTS.md（临时测试）
- [ ] STATE_ENHANCEMENT_RESULTS.md（早期测试）
- [ ] ROOT_CAUSE_ANALYSIS.md（早期诊断）
- [ ] V3_ROUTING_ISSUES_DIAGNOSIS.md（早期问题）
- [ ] SPARSEMAX_COMPETITION_ANALYSIS.md（早期分析）

#### 重复的Manager约束文档
- [ ] MANAGER_CONSTRAINT_ANALYSIS.md（有QUICK_REF）
- [ ] MANAGER_CONSTRAINT_DIAGRAM.md（内容已整合）
- [ ] MANAGER_CONSTRAINT_EFFECT_ANALYSIS.md（有SUMMARY）
- [ ] MANAGER_CONSTRAINT_EFFECT_SUMMARY.md（保留QUICK_REF即可）
- [ ] MANAGER_CONSTRAINT_SUMMARY.md（重复）
- [ ] MANAGER_CONSTRAINT_TEST_RESULTS.md（早期测试）

#### 重复的训练文档
- [ ] TRAINING_1000EP_README.md（内容已整合到PHASES）
- [ ] TRAINING_COMPLETE_ANALYSIS.md（重复）
- [ ] TRAINING_HEALTH_CHECK.md（临时检查）
- [ ] TRAINING_STATUS.md（过时状态）
- [ ] WARMUP_1000_RESULTS.md（具体结果，保留500EP即可）
- [ ] WARMUP_1000_SUMMARY.md（重复）
- [ ] TRANSITION_3000_RESULTS.md（中间结果）
- [ ] TRANSITION_TRAINING_STATUS.md（过时状态）
- [ ] FINETUNE_5000_RESULTS.md（中间结果）
- [ ] RESUME_TRAINING_PLAN.md（已完成）
- [ ] FULL_TRAINING_COMPARISON_PLAN.md（已完成）

#### 重复的可视化文档
- [ ] VISUALIZATION_ANALYSIS.md（重复）
- [ ] VISUALIZATION_EXPLANATION.md（重复）
- [ ] VISUALIZATION_TOOLS_SUMMARY.md（重复）
- [ ] EPISODE_VISUALIZATION_SUMMARY.md（临时）
- [ ] EXPERT_ACTIVATION_ANALYSIS.md（已整合）
- [ ] EXPERT_ORTHOGONALITY_VISUALIZATION_GUIDE.md（有README）

#### 重复的中文文档
- [ ] 除了加上内部奖励之外的修改部分.md（过时）
- [ ] 可视化对比完成.md（临时）
- [ ] 可视化工具使用说明.md（重复）
- [ ] 内部奖励效果分析总结.md（有最新版）
- [ ] 内部奖励效果总结_最新.md（已整合到英文文档）
- [ ] 下一步行动指南.md（过时）
- [ ] 专家行为分析说明.md（已整合）

#### 其他过时文档
- [ ] END_TO_END_VS_CURRICULUM.md（早期讨论）
- [ ] IMPROVEMENT_PLAN.md（已完成）
- [ ] V3_DATA_FLOW_DIAGRAM.md（已整合）

### 🧪 测试文件清理

#### 保留
- ✅ test_v3_smoke.py（基础烟雾测试）

#### 删除
- [ ] test_manager_constraints.py（临时测试）
- [ ] diagnose_manager_constraints.py（临时诊断）
- [ ] quick_comparison.py（临时对比）

---

## 📁 ablation_v3/results/ 清理

### ✅ 保留的训练结果（3个）
1. **warmup_1000/** - Baseline训练结果（1000 episodes）
2. **resume_500_from_100/** - With Manager训练结果（500 episodes）
3. **finetune_5000/** - 完整训练结果（如果有用）

### ❌ 删除的临时结果（15+个）
- [ ] quick_baseline_20260111_* （3个临时测试）
- [ ] quick_comparison_20260111_* （3个临时测试）
- [ ] quick_manager_20260111_* （3个临时测试）
- [ ] init_fix_test/（早期测试）
- [ ] test_all_mechanisms/（临时测试）
- [ ] test_manager_logging/（临时测试）
- [ ] v3_convergence_cpu/（早期测试）
- [ ] v3_enhanced_state_50ep/（早期测试）
- [ ] v3_enhanced_state_test/（早期测试）
- [ ] v3_quick_test/（临时测试）
- [ ] verify_manager/（临时验证）
- [ ] transition_3000/（中间结果，如果不需要）

---

## 📁 ablation_v3/scripts/ 清理

### ✅ 保留的脚本（5个）
1. **test_convergence_cpu.sh** - CPU收敛测试
2. **run_full_training_with_comparison.sh** - 完整训练对比
3. **run_finetune_5000.sh** - Fine-tune脚本
4. **run_transition_3000.sh** - Transition脚本
5. **test_init_fix.sh** - 初始化测试

### ❌ 删除的脚本（10+个）
- [ ] ablation_study.sh（重复）
- [ ] complete_warmup_1000.sh（已完成）
- [ ] quick_comparison_test.sh（临时）
- [ ] resume_100_to_500.sh（已完成）
- [ ] run_ablation_study.sh（重复）
- [ ] run_gpu_training.sh（重复）
- [ ] run_v3_experiments.sh（重复）
- [ ] start_full_training.sh（重复）
- [ ] start_medium_test.sh（临时）
- [ ] start_new_version_5000ep.sh（重复）
- [ ] start_new_version_500ep.sh（重复）
- [ ] test_v3_quick.sh（临时）
- [ ] ablation_v3/（空目录）

---

## 📁 ablation_v3/visualizations/ 清理

### ✅ 保留的可视化（3个目录）
1. **comprehensive_comparison/** - 训练对比图（论文用）
2. **expert_orthogonality_with_manager/** - 带Manager的专家正交性
3. **expert_orthogonality_baseline/** - Baseline的专家正交性

### ❌ 删除的可视化（8+个目录）
- [ ] 1000ep/（中间结果）
- [ ] 500ep_analysis/（临时分析）
- [ ] 500ep_comparison/（重复）
- [ ] episode/（临时）
- [ ] expert_data_with_manager/（中间数据）
- [ ] expert_orthogonality/（旧版本）
- [ ] expert_orthogonality_demo/（演示版本）
- [ ] manager_effect_100ep/（早期测试）
- [ ] expert_specialization_analysis.png（单独文件，已整合）
- [ ] three_phases_comparison.png（中间结果）
- [ ] v3_inference_demo.png（演示）

---

## 📁 tools/ 目录清理

### ✅ 保留的工具（核心工具）
1. **visualize_comprehensive_comparison.py** - 训练对比可视化
2. **visualize_expert_orthogonality_simple.py** - 专家正交性可视化
3. **analyze_1000ep_results.py** - 结果分析
4. **monitor_training.sh** - 训练监控

### ❌ 删除的工具（临时/重复）
- [ ] visualize_expert_orthogonality.py（旧版本）
- [ ] visualize_expert_orthogonality_real.py（未完成版本）
- [ ] extract_real_expert_data.py（未完成）
- [ ] compare_baseline_vs_manager_500ep.py（重复）
- [ ] compare_with_without_manager.py（重复）
- [ ] compare_500ep_results.py（重复）
- [ ] visualize_manager_effect_comparison.py（重复）
- [ ] analyze_manager_constraint_effect.py（临时）
- [ ] visualize_three_phases.py（中间结果）
- [ ] visualize_expert_specialization.py（已整合）
- [ ] analyze_expert_activation.py（临时）
- [ ] visualize_1000ep_training.py（中间结果）
- [ ] test_v3_routing_dynamic.py（临时测试）
- [ ] debug_v3_routing.py（临时调试）
- [ ] visualize_v3_episode.py（临时）
- [ ] visualize_v3_inference.py（演示）
- [ ] auto_visualize_after_training.sh（自动化脚本，可选）

---

## 📁 根目录清理

### ✅ 保留
- README.md（如果有）
- requirements.txt
- config.yaml
- .gitignore

### ❌ 删除
- [ ] run_v2.sh（V2版本）
- [ ] test_blstats.py（临时测试）
- [ ] test_train_smoke.py（临时测试）
- [ ] .DS_Store（Mac系统文件）

---

## 📁 其他目录

### ablation_v1/ 和 ablation_v2/
- **建议**：如果不再需要，可以整体删除或归档
- **保留条件**：如果需要对比V1/V2/V3的演进

### docsV1&V2/ 和 docsV3/
- **docsV1&V2/**：可以删除或归档
- **docsV3/**：保留，但需要清理重复文档

---

## 执行顺序

### Phase 1: 删除明显的临时文件（安全）
```bash
# 删除系统文件
find . -name ".DS_Store" -delete

# 删除Python缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### Phase 2: 删除临时测试结果
```bash
cd ablation_v3/results
rm -rf quick_*
rm -rf init_fix_test test_all_mechanisms test_manager_logging
rm -rf v3_convergence_cpu v3_enhanced_state_* v3_quick_test verify_manager
```

### Phase 3: 删除过时文档（需要确认）
```bash
cd ablation_v3
# 逐个删除或批量删除
```

### Phase 4: 删除临时可视化
```bash
cd ablation_v3/visualizations
rm -rf 1000ep 500ep_* episode expert_data_with_manager
rm -rf expert_orthogonality expert_orthogonality_demo manager_effect_100ep
rm *.png  # 删除单独的图片文件
```

### Phase 5: 删除临时工具
```bash
cd tools
# 逐个删除不需要的工具脚本
```

---

## 清理后的目录结构

```
ablation_v3/
├── README.md
├── PAPER_FIGURES_READY.md
├── 500EP_COMPARISON_RESULTS.md
├── EXPERT_ORTHOGONALITY_FIGURES_README.md
├── TRAINING_PHASES_EXPLAINED.md
├── MANAGER_CONSTRAINT_QUICK_REF.md
├── VISUALIZATION_GUIDE.md
├── RESULTS_INTERPRETATION_GUIDE.md
├── 论文图表已完成.md
├── 专家正交性可视化说明.md
├── test_v3_smoke.py
├── results/
│   ├── warmup_1000/          # Baseline
│   ├── resume_500_from_100/  # With Manager
│   └── finetune_5000/        # Full training (optional)
├── scripts/
│   ├── test_convergence_cpu.sh
│   ├── run_full_training_with_comparison.sh
│   ├── run_finetune_5000.sh
│   ├── run_transition_3000.sh
│   └── test_init_fix.sh
├── train/
│   └── train_v3_gat_moe.py
└── visualizations/
    ├── comprehensive_comparison/
    ├── expert_orthogonality_with_manager/
    └── expert_orthogonality_baseline/

tools/
├── visualize_comprehensive_comparison.py
├── visualize_expert_orthogonality_simple.py
├── analyze_1000ep_results.py
└── monitor_training.sh
```

---

## 预期效果

### 清理前
- **文档数量**：~60个
- **结果目录**：~20个
- **可视化目录**：~10个
- **工具脚本**：~20个

### 清理后
- **文档数量**：~10个（减少83%）
- **结果目录**：~3个（减少85%）
- **可视化目录**：~3个（减少70%）
- **工具脚本**：~4个（减少80%）

---

## 注意事项

1. **备份**：清理前先备份整个项目
2. **确认**：逐步删除，每次删除前确认
3. **Git**：如果使用Git，可以先commit当前状态
4. **恢复**：如果误删，可以从Git历史恢复

---

## 下一步

是否开始执行清理？建议顺序：
1. 先删除系统文件和缓存（最安全）
2. 再删除临时测试结果
3. 最后删除文档（需要逐个确认）
