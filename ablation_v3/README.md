# TEDG-RL V3: GAT-Guided Hierarchical MoE

> **版本**: V3.0  
> **状态**: ✅ 已验证可行，准备GPU训练  
> **创建日期**: 2025-01-05

---

## 🎯 核心创新

V3引入了**GAT引导的层级混合专家系统**，相比V1/V2有以下创新：

1. **超图GAT推理**: 动态激活节点，多跳因果推理
2. **Sparsemax路由**: 软中带硬，避免专家塌缩
3. **语义对齐专家**: Survival/Combat/Exploration/General
4. **三阶段训练**: Warmup → Transition → Fine-tune
5. **多重稳定性措施**: 负载均衡、多样性、NaN检测

---

## 📊 验证结果

### CPU收敛测试 (50 episodes)

| 指标 | 结果 |
|------|------|
| **收敛趋势** | ✅ 明确显示 |
| **分数提升** | 0.6 → 5.7 (9.5x) |
| **最佳分数** | 50 |
| **α熵** | 1.342 (稳定) |
| **稳定性** | ✅ 无崩溃 |

**结论**: V3显示明确的学习和收敛趋势，可以进行GPU大规模训练。

详细分析见: `CONVERGENCE_TEST_ANALYSIS.md`

---

## 🚀 快速开始

### 1. CPU快速测试 (验证代码)

```bash
# 10 episodes, 验证代码可行性
bash ablation_v3/scripts/test_v3_quick.sh
```

### 2. CPU收敛测试 (验证收敛)

```bash
# 50 episodes, 验证收敛趋势
bash ablation_v3/scripts/test_convergence_cpu.sh
```

### 3. GPU大规模训练 (正式训练)

```bash
# 10000 episodes, 完整三阶段训练
bash ablation_v3/scripts/run_gpu_training.sh
```

---

## 📁 目录结构

```
ablation_v3/
├── train/
│   └── train_v3_gat_moe.py          # 主训练脚本
├── scripts/
│   ├── test_v3_quick.sh             # 快速测试 (10 episodes)
│   ├── test_convergence_cpu.sh      # 收敛测试 (50 episodes)
│   ├── run_gpu_training.sh          # GPU训练 (10000 episodes)
│   └── run_v3_experiments.sh        # 消融实验
├── results/
│   ├── v3_quick_test/               # 快速测试结果
│   ├── v3_convergence_cpu/          # 收敛测试结果
│   └── v3_full_training/            # GPU训练结果 (待生成)
├── README.md                         # 本文档
├── QUICK_TEST_RESULTS.md            # 快速测试报告
└── CONVERGENCE_TEST_ANALYSIS.md     # 收敛测试分析
```

---

## 🔧 训练配置

### 网络架构

| 组件 | 配置 |
|------|------|
| **GAT层** | 2层, 4头, 256维 |
| **专家数量** | 4 (Survival/Combat/Exploration/General) |
| **路由方式** | Sparsemax (Warmup时Softmax) |
| **总参数** | 1,211,041 |

### 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **learning_rate** | 1e-4 | V2是3e-4，V3更小 |
| **clip_ratio** | 0.15 | V2是0.2，V3更保守 |
| **batch_size** | 256 | V2是128，V3更大 |
| **ppo_epochs** | 4 | V2是3，V3更充分 |
| **gamma** | 0.995 | V2是0.99，V3更长视野 |
| **max_grad_norm** | 1.0 | V2是0.5，V3更宽松 |

### 三阶段训练

| 阶段 | Episodes | 路由 | 学习率 | 负载均衡系数 |
|------|----------|------|--------|--------------|
| **Warmup** | 0-1000 | Softmax | 1e-4 | 0.02 |
| **Transition** | 1000-3000 | 温度退火 | 5e-5 | 0.01 |
| **Fine-tune** | 3000+ | Sparsemax | 1e-5 | 0.005 |

---

## 📈 预期结果

### 训练曲线

| 阶段 | 预期分数 | α熵 | 专家使用 |
|------|----------|-----|----------|
| **Warmup** | 50-100 | 1.2-1.4 | 均衡 (20-30% each) |
| **Transition** | 100-300 | 0.8-1.0 | 开始分化 |
| **Fine-tune** | 300-600+ | 0.5-0.7 | 明确分工 |

### 与V1/V2对比

| 指标 | V1 | V2 | V3 (预期) |
|------|-----|-----|-----------|
| **best_score** | 500-600 | 600-700 | **800+** |
| **sample_efficiency** | 1.0x | 1.2x | **1.5x** |
| **training_stability** | 中 | 低 | **高** |
| **interpretability** | 低 | 中 | **高** |

---

## 🔍 监控指标

### 必须监控

1. **episode_score**: 应持续上升
2. **alpha_entropy**: 
   - Warmup: 1.2-1.4
   - Transition: 0.8-1.0
   - Fine-tune: 0.5-0.7
3. **expert_usage**: 避免某个>80%
4. **gradient_norm**: 应<5.0

### 异常检测

| 异常 | 阈值 | 处理 |
|------|------|------|
| **专家塌缩** | α熵<0.3 | 增加负载均衡系数 |
| **梯度爆炸** | 梯度范数>10.0 | 降低学习率 |
| **GAT过平滑** | 注意力方差<0.05 | 增加Dropout |
| **奖励不收敛** | 长期不变 | 调整奖励塑形 |

---

## 🧪 消融实验

### 实验组

```bash
# 1. 完整V3 (基线)
python ablation_v3/train/train_v3_gat_moe.py --exp-name v3_full

# 2. 固定GAT (验证GAT贡献)
python ablation_v3/train/train_v3_gat_moe.py --exp-name v3_fixed_gat --freeze-gat

# 3. 2个专家 (验证专家数量)
python ablation_v3/train/train_v3_gat_moe.py --exp-name v3_2experts --num-experts 2

# 4. 无动作掩码 (验证掩码贡献)
python ablation_v3/train/train_v3_gat_moe.py --exp-name v3_no_mask --no-mask
```

---

## 📚 相关文档

### 设计文档

- `docsV3/V3_ARCHITECTURE_DESIGN.md` - 完整架构设计
- `docsV3/V1_V2_V3_COMPARISON.md` - 版本对比
- `docsV3/语义正交MOE.md` - 设计动机

### 稳定性文档

- `docsV3/V3_TRAINING_STABILITY_CHECKLIST.md` - 完整稳定性清单
- `docsV3/TRAINING_STABILITY_SUMMARY.md` - 快速参考
- `docsV3/DEBUGGING_QUICK_REFERENCE.md` - 调试指南

### 实现文档

- `docsV3/IMPLEMENTATION_STATUS.md` - 实现状态
- `docsV3/STABILITY_IMPLEMENTATION_STATUS.md` - 稳定性实现状态

---

## 🐛 故障排除

### 常见问题

1. **专家塌缩** (α熵<0.3)
   - 增加负载均衡系数: 0.01 → 0.05
   - 延长Warmup: 1000 → 2000 episodes

2. **梯度爆炸** (梯度范数>10.0)
   - 降低学习率: 1e-4 → 5e-5
   - 增加梯度裁剪: 1.0 → 0.5

3. **奖励不收敛**
   - 调整奖励塑形权重
   - 增加熵正则化: 0.01 → 0.05

详细调试指南见: `docsV3/DEBUGGING_QUICK_REFERENCE.md`

---

## ✅ 检查清单

训练前确认：

- [ ] 超图已转换 (`data/hypergraph/hypergraph_gat_structure.json`)
- [ ] PyTorch Geometric已安装
- [ ] conda环境已激活 (`tedg-rl-demo`)
- [ ] GPU可用 (或强制CPU: `export TEDG_FORCE_CPU=1`)
- [ ] 磁盘空间充足 (>10GB for checkpoints)

---

## 📞 支持

如遇问题：

1. 查看 `DEBUGGING_QUICK_REFERENCE.md`
2. 检查 `CONVERGENCE_TEST_ANALYSIS.md`
3. 参考 `V3_TRAINING_STABILITY_CHECKLIST.md`

---

**V3已验证可行，准备GPU训练！** 🚀

