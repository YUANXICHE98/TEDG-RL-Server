# V3 vs V4 对比测试准备完成

## ✅ 已完成的工作

### 1. 对比分析脚本

**文件**: `ablation_v4/scripts/compare_v3_v4_100ep.py`

**功能**:
- 加载V3和V4的训练日志
- 支持只提取前N个episode进行对比
- 对比关键指标:
  - Episode Rewards/Scores
  - Alpha Entropy (Router专业化程度)
  - Expert Usage Variance (专家差异化)
  - Training Stability
- 生成4个对比图表:
  - Episode Rewards曲线
  - Episode Scores曲线
  - Alpha Entropy曲线
  - Performance Bar Chart
- 输出详细的分析报告

### 2. 对比测试脚本

**文件**: `ablation_v4/scripts/run_v3_v4_comparison_100ep.sh`

**流程**:
1. ✅ 检查V3结果 (使用已有的warmup_1000前100 episodes)
2. ✅ 运行V4训练 (100 episodes)
3. ✅ 调用对比分析脚本
4. ✅ 生成对比可视化

**优化**:
- 不重新运行V3训练（使用已有结果）
- 只运行V4训练，节省时间
- 自动生成对比报告

### 3. 文档更新

**文件**:
- `ablation_v4/README.md` - 添加小规模对比测试说明
- `ablation_v4/QUICK_START.md` - 快速开始指南
- `ablation_v4/COMPARISON_TEST_READY.md` - 本文件

### 4. 权限设置

```bash
chmod +x ablation_v4/scripts/run_v3_v4_comparison_100ep.sh
chmod +x ablation_v4/scripts/compare_v3_v4_100ep.py
```

---

## 🚀 如何运行

### 一键运行对比测试

```bash
bash ablation_v4/scripts/run_v3_v4_comparison_100ep.sh
```

### 查看结果

```bash
# 对比图
open ablation_v4/results/v3_v4_comparison_100ep.png

# V4训练日志
cat ablation_v4/results/test_100ep/logs/training_log.json

# V3参考日志（前100 episodes）
cat ablation_v3/results/warmup_1000/logs/training_log.json
```

---

## 📊 对比指标

### 1. Performance Metrics

| 指标 | 说明 | 预期 |
|------|------|------|
| Avg Reward | 平均奖励 | V4 > V3 |
| Avg Score | 平均分数 | V4 > V3 |
| Best Reward | 最佳奖励 | V4 > V3 |
| Best Score | 最佳分数 | V4 > V3 |

### 2. Alpha Entropy

| 指标 | V3 | V4 (预期) | 说明 |
|------|----|-----------| -----|
| Avg Entropy | ~0.693 | 0.3-0.5 | 越低越好（更专业化） |
| Near ln(2) % | >70% | <30% | 卡在ln(2)的比例 |

### 3. Expert Usage

| 指标 | V3 | V4 (预期) | 说明 |
|------|----|-----------| -----|
| Usage Variance | 低 | 高 | 专家差异化程度 |
| Active Experts | 2/4 | 4/4 | 实际使用的专家数 |

---

## 🎯 成功标准

### ✅ V4显著优于V3

如果满足以下条件，认为V4成功：

1. **Performance**: 至少3/4的性能指标有改进
2. **Alpha Entropy**: 平均值 < 0.6 (V3约0.693)
3. **Expert Usage**: 方差明显高于V3
4. **Stability**: 训练曲线更平滑

→ **继续完整3阶段训练**

### ➡️ V4中等改进

如果满足以下条件：

1. **Performance**: 1-2个指标有改进
2. **Alpha Entropy**: 略有下降但仍>0.6
3. **Expert Usage**: 方差略有提升

→ **调整超参数后再测试**

### ⚠️ V4无明显改进

如果：

1. **Performance**: 无改进或下降
2. **Alpha Entropy**: 无变化或上升
3. **Expert Usage**: 无变化

→ **重新审视Cross-Attention设计**

---

## 🔧 可调参数

如果效果不理想，可以调整：

### 1. Sparse Gate阈值

```python
# networks_v4_cross_attention.py
sparse_topk = 0.3  # 当前值
# 可以尝试: 0.5, 0.7 (更宽松)
```

### 2. Attention Head数量

```python
num_heads = 4  # 当前值
# 可以尝试: 8 (更多表达能力)
```

### 3. Context Vector维度

```python
context_dim = 256  # 当前值
# 可以尝试: 384, 512 (更大容量)
```

### 4. Manager约束强度

```python
alignment_coef = 0.1  # 当前值
semantic_coef = 0.05  # 当前值
# 可以尝试: 0.15, 0.1 (更强约束)
```

---

## 📝 测试清单

- [x] 创建对比分析脚本
- [x] 创建对比测试脚本
- [x] 更新文档
- [x] 设置执行权限
- [ ] 运行烟雾测试
- [ ] 运行小规模对比测试
- [ ] 分析结果
- [ ] 决定下一步行动

---

## 🎓 技术细节

### V3 数据来源

使用 `ablation_v3/results/warmup_1000/logs/training_log.json` 的前100个episode作为V3基准。

**原因**:
- V3的warmup阶段已经训练完成
- 前100个episode足够代表V3的初期性能
- 避免重新运行V3训练，节省时间

### V4 训练配置

```python
--exp-name test_100ep
--episodes 100
--max-steps 500
--num-experts 4
```

**配置说明**:
- 100 episodes: 快速验证
- 500 max-steps: 与V3一致
- 4 experts: 与V3一致

### 对比分析方法

1. **加载数据**: 从JSON日志提取指标
2. **截取数据**: V3只取前100 episodes
3. **统计分析**: 计算均值、方差、最大值
4. **可视化**: 生成4个对比图表
5. **报告生成**: 输出详细分析报告

---

**准备完成时间**: 2026-01-22  
**预计测试时间**: 30分钟  
**状态**: ✅ 准备就绪，可以开始测试
