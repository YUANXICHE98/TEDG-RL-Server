# 完整训练与对比计划

## 目标

完成V3的全部训练（5000 episodes），并在训练过程中对比Manager约束实施前后的效果。

## 当前状态

### 已完成的训练

1. **旧版本（无Manager约束）**：
   - Warmup (1000ep): 分数7.53, Alpha熵1.3853
   - Transition (3000ep): 分数10.35, Alpha熵0.6929
   - Fine-tune (5000ep): 分数15.46, Alpha熵0.6928

2. **新版本快速测试（有Manager约束）**：
   - 100 episodes: 分数11.12, Alpha熵1.3832
   - 500 episodes: 已完成（需要分析）

### 需要完成的训练

新版本（有Manager约束）的完整训练：
- ✅ Warmup (0-1000): 部分完成（已有500ep）
- ⏳ Transition (1000-3000): 待训练
- ⏳ Fine-tune (3000-5000): 待训练

## 训练计划

### Phase 1: 完成Warmup阶段 (500→1000 episodes)

**起点**: `resume_500_from_100/checkpoints/model_00500.pth`  
**终点**: 1000 episodes  
**预计时间**: 2-3小时

```bash
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name warmup_1000_with_manager \
    --episodes 1000 \
    --max-steps 2000 \
    --resume ablation_v3/results/resume_500_from_100/checkpoints/model_00500.pth
```

**预期指标**:
- 平均分数: 12-14
- Alpha熵: 1.2-1.3（保持较高熵）
- 对齐损失: 逐渐下降

### Phase 2: Transition阶段 (1000→3000 episodes)

**起点**: Warmup完成的checkpoint  
**终点**: 3000 episodes  
**预计时间**: 8-10小时

```bash
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name transition_3000_with_manager \
    --episodes 3000 \
    --max-steps 2000 \
    --phase transition \
    --resume ablation_v3/results/warmup_1000_with_manager/checkpoints/model_final.pth
```

**预期指标**:
- 平均分数: 15-18（vs 旧版10.35）
- Alpha熵: 0.5-0.7（开始专业化）
- 对齐损失: 持续下降

### Phase 3: Fine-tune阶段 (3000→5000 episodes)

**起点**: Transition完成的checkpoint  
**终点**: 5000 episodes  
**预计时间**: 8-10小时

```bash
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name finetune_5000_with_manager \
    --episodes 5000 \
    --max-steps 2000 \
    --phase finetune \
    --resume ablation_v3/results/transition_3000_with_manager/checkpoints/model_final.pth
```

**预期指标**:
- 平均分数: 20-25（vs 旧版15.46）
- Alpha熵: 0.3-0.4（高度专业化）
- 对齐损失: 稳定在低值

## 对比分析计划

### 1. 实时监控对比

在训练过程中，每个阶段完成后立即进行对比：

```bash
# 对比工具
python tools/compare_with_without_manager.py \
    --baseline ablation_v3/results/warmup_1000 \
    --with_manager ablation_v3/results/warmup_1000_with_manager \
    --phase warmup
```

### 2. 关键指标对比

| 阶段 | 指标 | 旧版本（无约束） | 新版本（有约束） | 改进 |
|------|------|----------------|----------------|------|
| **Warmup@1000** | 平均分数 | 7.53 | ? | ? |
| | Alpha熵 | 1.3853 | ? | ? |
| | 对齐损失 | N/A | ? | N/A |
| **Transition@3000** | 平均分数 | 10.35 | ? | ? |
| | Alpha熵 | 0.6929 | ? | ? |
| | 对齐损失 | N/A | ? | N/A |
| **Fine-tune@5000** | 平均分数 | 15.46 | ? | ? |
| | Alpha熵 | 0.6928 | ? | ? |
| | 对齐损失 | N/A | ? | N/A |

### 3. 可视化对比

每个阶段完成后生成对比可视化：

```bash
# 生成对比图表
python tools/visualize_manager_comparison.py \
    --baseline_warmup ablation_v3/results/warmup_1000 \
    --baseline_transition ablation_v3/results/transition_3000 \
    --baseline_finetune ablation_v3/results/finetune_5000 \
    --manager_warmup ablation_v3/results/warmup_1000_with_manager \
    --manager_transition ablation_v3/results/transition_3000_with_manager \
    --manager_finetune ablation_v3/results/finetune_5000_with_manager \
    --output ablation_v3/visualizations/full_comparison/
```

**生成的图表**:
1. 分数对比曲线（三个阶段）
2. Alpha熵对比曲线
3. 专家使用率对比
4. 对齐损失变化（仅新版本）
5. 阶段性改进百分比

### 4. 专家行为分析

对比专家的专业化程度：

```bash
# 分析专家行为
python tools/analyze_expert_specialization.py \
    --baseline ablation_v3/results/finetune_5000 \
    --with_manager ablation_v3/results/finetune_5000_with_manager \
    --output ablation_v3/visualizations/expert_comparison/
```

**分析内容**:
- 专家激活模式
- 专家-场景对应关系
- 专家决策可解释性

## 执行时间表

### 总预计时间: 18-23小时

| 阶段 | 时间 | 累计时间 |
|------|------|---------|
| Warmup (500→1000) | 2-3h | 2-3h |
| 可视化分析 | 0.5h | 2.5-3.5h |
| Transition (1000→3000) | 8-10h | 10.5-13.5h |
| 可视化分析 | 0.5h | 11-14h |
| Fine-tune (3000→5000) | 8-10h | 19-24h |
| 最终分析 | 1h | 20-25h |

### 建议执行方式

**方式1: 连续执行（推荐）**
```bash
# 创建自动化脚本
bash ablation_v3/scripts/run_full_training_with_comparison.sh
```

**方式2: 分阶段执行**
- Day 1: Warmup阶段 + 分析
- Day 2: Transition阶段 + 分析
- Day 3: Fine-tune阶段 + 最终分析

## 成功标准

### 定量标准

1. **分数提升**: 新版本最终分数 > 旧版本 + 30%
2. **专业化提升**: 新版本Alpha熵 < 0.4（vs 旧版本0.69）
3. **稳定性**: 无NaN/Inf，训练完整完成
4. **对齐效果**: 对齐损失在Fine-tune阶段 < 0.1

### 定性标准

1. **可解释性**: 能清晰解释每个专家的行为
2. **一致性**: GAT推理和Router选择高度一致
3. **鲁棒性**: 在不同场景下表现稳定

## 风险与应对

### 风险1: 训练时间过长

**应对**: 
- 使用GPU加速（如果可用）
- 降低max_steps到1000（快速验证）
- 分阶段执行，每阶段独立分析

### 风险2: 效果不如预期

**应对**:
- 检查对齐损失是否下降
- 调整alignment_coef（0.05-0.2）
- 分析专家使用率是否均衡

### 风险3: 训练不稳定

**应对**:
- 降低学习率
- 增加梯度裁剪
- 检查GAT注意力方差

## 输出文件

### 训练结果
```
ablation_v3/results/
├── warmup_1000_with_manager/
│   ├── checkpoints/
│   ├── logs/
│   └── training.log
├── transition_3000_with_manager/
│   ├── checkpoints/
│   ├── logs/
│   └── training.log
└── finetune_5000_with_manager/
    ├── checkpoints/
    ├── logs/
    └── training.log
```

### 可视化结果
```
ablation_v3/visualizations/
├── full_comparison/
│   ├── score_comparison.png
│   ├── alpha_entropy_comparison.png
│   ├── expert_usage_comparison.png
│   └── improvement_summary.png
└── expert_comparison/
    ├── expert_activation_patterns.png
    ├── expert_scene_correspondence.png
    └── interpretability_analysis.png
```

### 分析报告
```
ablation_v3/
├── FULL_TRAINING_RESULTS.md
├── MANAGER_CONSTRAINT_EFFECT_FINAL.md
└── EXPERT_SPECIALIZATION_ANALYSIS.md
```

## 下一步行动

### 立即执行

1. **检查500ep结果**:
```bash
python tools/analyze_500ep_results.py \
    --result_dir ablation_v3/results/resume_500_from_100
```

2. **启动Warmup完成训练**:
```bash
bash ablation_v3/scripts/complete_warmup_1000.sh
```

3. **设置监控**:
```bash
bash tools/monitor_training.sh ablation_v3/results/warmup_1000_with_manager
```

### 后续步骤

- [ ] 完成Warmup阶段（500→1000）
- [ ] 分析Warmup结果并对比
- [ ] 启动Transition阶段（1000→3000）
- [ ] 分析Transition结果并对比
- [ ] 启动Fine-tune阶段（3000→5000）
- [ ] 完成最终分析和对比
- [ ] 撰写完整报告

---

**创建时间**: 2026-01-13  
**预计完成**: 2026-01-14 或 2026-01-15  
**负责人**: Kiro AI Assistant  
**状态**: 📋 计划制定完成，等待执行
