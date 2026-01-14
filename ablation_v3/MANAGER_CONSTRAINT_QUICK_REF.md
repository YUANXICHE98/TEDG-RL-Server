# Manager内层约束 - 快速参考

## 🎯 一句话总结

**Manager内层约束 = 让GAT的超图推理直接指导Router选择专家，通过KL散度强制对齐**

---

## 📊 核心公式

```python
# 1. GAT推理 → Operator分数
operator_scores = GAT(hypergraph)  # (batch, 76)

# 2. 聚合 → Expert目标分布
expert_scores = aggregate(operator_scores)  # (batch, 4)
target_alpha = softmax(expert_scores / T)

# 3. Router输出
alpha = Router(h_logic)  # (batch, 4)

# 4. 对齐损失
L_alignment = KL(target_alpha || alpha)
```

---

## 🔑 关键文件

| 文件 | 作用 |
|------|------|
| `src/core/operator_expert_mapping.py` | 76个Operators → 4个Experts映射 |
| `ablation_v3/train/train_v3_gat_moe.py` | 主训练脚本（包含所有实现） |
| `ablation_v3/MANAGER_CONSTRAINT_EFFECT_ANALYSIS.md` | 完整效果分析 |

---

## 📈 预期效果（5000 episodes）

| 指标 | Baseline | +Manager | 改进 |
|------|----------|----------|------|
| **Alpha熵** | 0.69 | 0.2-0.3 | **-57% to -65%** |
| **平均分数** | 12.23 | 20-25 | **+63% to +104%** |
| **对齐度** | 中 | 极高 | **+80%** |
| **专家切换** | 高 | 低 | **-70%** |

---

## ⚙️ 关键参数

```python
# 在get_training_config()中
config = {
    'alignment_coef': 0.1,           # 对齐损失系数（推荐0.1）
    'alignment_temperature': 1.0,    # 对齐温度（推荐1.0）
    'semantic_coef': 0.05,           # 语义正交系数（推荐0.05）
}
```

---

## 🚀 快速开始

### 1. 运行训练

```bash
conda activate tedg-rl-demo

# 中期测试（500 episodes，2-3小时）
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name v3_manager_500 \
    --episodes 500 \
    --max-steps 2000

# 完整训练（5000 episodes，20-30小时）
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name v3_manager_full \
    --episodes 5000 \
    --max-steps 2000
```

### 2. 查看效果

```bash
# 查看Manager约束loss
grep "Manager Constraints" ablation_v3/results/v3_manager_500/training.log

# 查看Alpha熵
grep "α_entropy" ablation_v3/results/v3_manager_500/training.log | tail -50
```

### 3. 对比分析

```bash
# 对比有无Manager约束
python tools/analyze_manager_constraint_effect.py \
    --baseline ablation_v3/results/baseline_exp \
    --manager ablation_v3/results/v3_manager_500 \
    --output ablation_v3/visualizations
```

---

## 🔍 如何验证有效？

### 1. 查看训练日志

**期望看到**:
```
Episode 100: Manager Constraints: Alignment=0.3521, Semantic=0.1234
Episode 500: Manager Constraints: Alignment=0.1823, Semantic=0.0789
```

**解读**:
- Alignment loss下降 → Router越来越听从GAT ✓
- Semantic loss下降 → 专家越来越正交 ✓

### 2. 观察Alpha熵

**期望曲线**:
```
1.4 |     Warmup
1.2 | ___________
1.0 |            \
0.8 |             \  Transition
0.6 |              \
0.4 |               \___
0.2 |                   \_____ Fine-tune
    +---------------------------------> Episodes
    0   1000        3000         5000
```

**关键点**:
- 0-1000: 熵~1.38（正常，防塌缩）
- 1000-3000: 熵下降到0.5-0.7
- 3000-5000: 熵快速降到0.2-0.3 ← **Manager约束发力**

### 3. 计算对齐度

```bash
# 运行对齐度计算
python tools/analyze_manager_constraint_effect.py \
    --baseline ablation_v3/results/baseline \
    --manager ablation_v3/results/v3_manager_500 \
    --compute-alignment \
    --alignment-episodes 10
```

**期望**:
- Warmup: 0.3-0.5（低）
- Transition: 0.5-0.7（中）
- Fine-tune: 0.8-0.95（高）← **接近完全对齐**

---

## ⚠️ 常见问题

### Q1: 100 episodes看不到效果？

**A**: 正常！原因：
1. Warmup阶段（0-1000）在最大化熵，与Manager约束方向相反
2. 神经网络需要更多样本学习对齐关系
3. **建议**: 至少运行500+ episodes

### Q2: Alignment loss为0？

**A**: 检查：
1. `operator_names`是否正确提取？
2. `hypergraph_gat_structure.json`是否存在？
3. 运行诊断：`python ablation_v3/diagnose_manager_constraints.py`

### Q3: Alpha熵不下降？

**A**: 可能原因：
1. 训练时间太短（<1000 episodes）
2. `alignment_coef`太小（<0.05）
3. 还在Warmup阶段（熵最大化）

### Q4: 如何调整参数？

**A**: 建议顺序：
1. 先用默认参数（`alignment_coef=0.1`）
2. 如果效果不明显，增大到0.15-0.2
3. 如果过拟合GAT，减小到0.05-0.08

---

## 📚 深入阅读

1. **MANAGER_CONSTRAINT_EFFECT_ANALYSIS.md** - 完整效果分析（推荐）
2. **MANAGER_CONSTRAINT_SUMMARY.md** - 理论总结
3. **FINAL_IMPLEMENTATION_README.md** - 使用指南
4. **除了加上内部奖励之外的修改部分.md** - 理论分析

---

## 🎓 核心洞察

### 为什么需要Manager约束？

**问题**: Alpha熵停滞在0.69 = Router在2个专家之间"和稀泥"

**根本原因**: GAT和Router松耦合
- GAT推理出"应该战斗"
- 但Router可以忽略，选择"逃跑"
- GAT的76个Operator分数被浪费

**解决方案**: Manager约束 = 强耦合
- 通过KL散度强制Router听从GAT
- 提供密集监督信号
- 加速收敛，提高性能

### 与其他机制的关系

```
完整的专家专业化体系:
├── Manager约束（本机制）← 密集监督
├── 熵最小化 ← 强制专业化
├── 时间一致性 ← 引入记忆
└── 重叠惩罚 ← 真正竞争
```

**它们共同作用，缺一不可！**

---

## ✅ 检查清单

训练前:
- [ ] 确认`operator_expert_mapping.py`存在
- [ ] 确认`hypergraph_gat_structure.json`存在
- [ ] 运行测试：`python ablation_v3/test_manager_constraints.py`

训练中:
- [ ] 每100 episodes检查Manager约束loss
- [ ] 观察Alpha熵是否下降
- [ ] 监控平均分数是否提升

训练后:
- [ ] 对比有无Manager约束的效果
- [ ] 计算对齐度
- [ ] 可视化专家选择

---

## 🚀 下一步

1. ✅ 代码已实现
2. ✅ 测试已通过
3. ⏳ 运行500 episodes中期测试
4. ⏳ 分析效果
5. ⏳ 运行5000 episodes完整训练

**准备就绪，开始训练！** 🎉

---

**创建时间**: 2026-01-12  
**状态**: ✅ 完整实现  
**准备就绪**: 可以开始训练
