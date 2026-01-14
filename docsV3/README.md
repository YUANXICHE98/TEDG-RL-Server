# V3 文档索引

## 📖 快速导航

### 🎯 从这里开始
- **[V3_COMPLETE_JOURNEY.md](V3_COMPLETE_JOURNEY.md)** - 完整开发历程（推荐首先阅读）

### 📐 架构设计
- **[V3_ARCHITECTURE_DESIGN.md](V3_ARCHITECTURE_DESIGN.md)** - V3架构设计
- **[语义正交MOE.md](语义正交MOE.md)** - 混合专家系统设计
- **[超图修改.md](超图修改.md)** - 超图结构修改
- **[V1_V2_V3_COMPARISON.md](V1_V2_V3_COMPARISON.md)** - 三版本对比

### 🛡️ 稳定性措施
- **[TRAINING_STABILITY_SUMMARY.md](TRAINING_STABILITY_SUMMARY.md)** - 稳定性总结
- **[TRAINING_STABILITY_CHECKLIST.md](TRAINING_STABILITY_CHECKLIST.md)** - 检查清单
- **[STABILITY_MEASURES_DIAGRAM.md](STABILITY_MEASURES_DIAGRAM.md)** - 措施图解
- **[STABILITY_IMPLEMENTATION_STATUS.md](STABILITY_IMPLEMENTATION_STATUS.md)** - 实施状态
- **[DEBUGGING_QUICK_REFERENCE.md](DEBUGGING_QUICK_REFERENCE.md)** - 调试参考
- **[README_STABILITY_DOCS.md](README_STABILITY_DOCS.md)** - 稳定性文档索引

### 🔧 问题诊断与修正
- **[修改建议.md](修改建议.md)** - 深度分析和修正方案（重要）
- **[修改完成状态.md](修改完成状态.md)** - 修改清单

### 📊 实验结果
详见 `../ablation_v3/` 目录：
- **[README.md](../ablation_v3/README.md)** - V3实验说明
- **[INIT_FIX_SUMMARY.md](../ablation_v3/INIT_FIX_SUMMARY.md)** - 初始化修正总结（重要）
- **[VISUALIZATION_ANALYSIS.md](../ablation_v3/VISUALIZATION_ANALYSIS.md)** - 可视化分析
- 其他测试结果文档...

## 📅 开发时间线

```
第一阶段 (2026-01-05)
├── 架构设计与实现
└── 稳定性措施设计

第二阶段 (2026-01-06 早期)
├── 初步测试
└── 发现：Alpha不变

第三阶段 (2026-01-06 中期)
├── 路由问题诊断
└── 找到：根本原因

第四阶段 (2026-01-06 中期)
├── 状态增强尝试
└── 结论：有改善但不够

第五阶段 (2026-01-06 晚期)
├── 初始化修正
└── 成功：Alpha变化240x

第六阶段 (2026-01-07)
├── 可视化分析
└── 验证：效果确认
```

## 🎯 核心问题与解决方案

### 问题：Alpha不变
- **现象**：Alpha几乎固定，每步变化<0.001
- **根因**：Expert初始化增益0.01太小
- **方案**：gain 0.01 → 0.5
- **结果**：✅ Alpha变化240x

### 问题：专家学不动
- **现象**：动作logits接近0（~0.02）
- **根因**：初始化太小 + Sparsemax过早
- **方案**：增大初始化 + Warmup用Softmax
- **结果**：✅ Exploration被激活到43%

### 问题：训练不稳定
- **现象**：担心梯度爆炸/NaN
- **方案**：五大稳定性措施
- **结果**：✅ 训练稳定，Alpha熵1.38

## 📈 当前状态

### ✅ 已解决
- Alpha不变问题
- 专家学习问题
- 场景感知问题
- Softmax生效问题

### ⚠️ 待观察
- 专家分工（Survival过于主导）
- Combat激活（需要战斗场景）
- 训练时长（50 episodes不足）

### 🎯 下一步
1. 继续训练到1000 episodes
2. 观察专家分工演化
3. 进入Transition阶段

## 🔗 相关链接

### 代码
- `src/core/networks_v3_gat_moe.py` - 主网络
- `src/core/hypergraph_gat.py` - GAT实现
- `ablation_v3/train/train_v3_gat_moe.py` - 训练脚本

### 工具
- `tools/debug_v3_routing.py` - 路由调试
- `tools/test_v3_routing_dynamic.py` - 动态测试
- `tools/visualize_v3_episode.py` - Episode可视化

### 测试脚本
- `ablation_v3/scripts/test_init_fix.sh` - 初始化修正测试
- `ablation_v3/scripts/test_convergence_cpu.sh` - 收敛测试

---

**最后更新**: 2026-01-07
**维护者**: V3开发团队
