# Visualization Tools Summary

## 已创建的可视化工具

### 1. 综合训练对比 (Comprehensive Training Comparison)

**工具**: `tools/visualize_comprehensive_comparison.py`

**功能**: 对比Baseline vs With Manager的训练效果

**生成图表**: 12个子图的综合对比
- 分数/奖励对比
- 改进率变化
- 分布分析（CDF, 直方图, 箱线图）
- 统计表格

**输出**: `ablation_v3/visualizations/comprehensive_comparison/comprehensive_comparison.png`

**运行**:
```bash
python3 tools/visualize_comprehensive_comparison.py
```

**状态**: ✅ 已完成并生成

---

### 2. 专家正交性可视化 (Expert Orthogonality)

**工具**: 
- `tools/extract_real_expert_data.py` - 数据提取
- `tools/visualize_expert_orthogonality_real.py` - 可视化生成

**功能**: 证明专家的时间和参数空间正交性

**生成图表**:
1. **Expert Activation Heatmap** - 时间正交性
   - 横轴：时间步
   - 纵轴：4个专家
   - 颜色：激活强度
   - 预期：块状分布

2. **t-SNE Visualization** - 参数空间正交性
   - 2D散点图
   - 4个专家的权重cluster
   - 预期：清晰分离的cluster

3. **Combined Figure** - 组合图（用于论文）

**输出**: `ablation_v3/visualizations/expert_orthogonality/`

**运行**:
```bash
# Step 1: 提取数据（需要PyTorch环境）
python3 tools/extract_real_expert_data.py

# Step 2: 生成可视化
python3 tools/visualize_expert_orthogonality_real.py
```

**状态**: 📋 工具已创建，需要在PyTorch环境中运行

**详细说明**: 
- `ablation_v3/EXPERT_ORTHOGONALITY_VISUALIZATION_GUIDE.md` (英文)
- `ablation_v3/专家正交性可视化说明.md` (中文)

---

## 文件结构

```
TEDG-RL-Server/
├── tools/
│   ├── visualize_comprehensive_comparison.py      # 综合对比
│   ├── extract_real_expert_data.py                # 提取专家数据
│   └── visualize_expert_orthogonality_real.py     # 专家正交性可视化
│
├── ablation_v3/
│   ├── visualizations/
│   │   ├── comprehensive_comparison/              # 综合对比图
│   │   │   └── comprehensive_comparison.png       ✅ 已生成
│   │   ├── expert_data/                           # 提取的专家数据
│   │   │   ├── alpha_history.npy                  📋 待生成
│   │   │   ├── expert_weights.npy                 📋 待生成
│   │   │   └── episodes_analysis.json             📋 待生成
│   │   └── expert_orthogonality/                  # 专家正交性图
│   │       ├── expert_activation_heatmap_real.png 📋 待生成
│   │       ├── expert_weights_tsne_real.png       📋 待生成
│   │       └── expert_orthogonality_combined_real.png 📋 待生成
│   │
│   ├── 500EP_COMPARISON_RESULTS.md                # 对比结果文档
│   ├── VISUALIZATION_EXPLANATION.md               # 可视化说明
│   ├── RESULTS_INTERPRETATION_GUIDE.md            # 结果解读
│   ├── EXPERT_ORTHOGONALITY_VISUALIZATION_GUIDE.md # 专家可视化指南(英文)
│   └── 专家正交性可视化说明.md                     # 专家可视化指南(中文)
```

---

## 使用场景

### 场景1：展示Manager约束的效果

**使用**: `visualize_comprehensive_comparison.py`

**目的**: 证明Manager约束带来显著提升

**关键发现**:
- 分数提升: +58.1%
- 奖励提升: +186.0%
- 效果随训练时间加速增长

**论文位置**: Experiments章节 - Ablation Study

---

### 场景2：证明专家正交性

**使用**: `extract_real_expert_data.py` + `visualize_expert_orthogonality_real.py`

**目的**: 证明专家学习到了不同的表示

**关键证据**:
1. **时间正交性**: 不同时刻激活不同专家
2. **参数空间正交性**: 专家权重形成分离的cluster

**论文位置**: Experiments章节 - Expert Specialization Analysis

---

## 论文中的使用

### Figure 1: Training Comparison

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.9\textwidth]{comprehensive_comparison.png}
\caption{Comprehensive training comparison between baseline and 
with Manager constraints over 400 episodes. The Manager constraints 
lead to +58.1\% score improvement and +186.0\% reward improvement.}
\label{fig:training_comparison}
\end{figure*}
```

### Figure 2: Expert Orthogonality

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.9\textwidth]{expert_orthogonality_combined_real.png}
\caption{Expert Orthogonality Analysis. (a) Temporal orthogonality: 
different experts activate at different times. (b) Parameter space 
orthogonality: experts form well-separated clusters.}
\label{fig:expert_orthogonality}
\end{figure*}
```

---

## 下一步行动

### 立即可做

✅ **综合对比图已完成**
- 图表已生成
- 可以直接用于论文

### 需要PyTorch环境

📋 **专家正交性图待生成**

**选项A: 在训练服务器上运行**
```bash
ssh training-server
conda activate rl-env
cd /path/to/TEDG-RL-Server
python3 tools/extract_real_expert_data.py
python3 tools/visualize_expert_orthogonality_real.py
```

**选项B: 本地安装依赖**
```bash
pip install torch numpy matplotlib seaborn scikit-learn nle
python3 tools/extract_real_expert_data.py
python3 tools/visualize_expert_orthogonality_real.py
```

---

## 关键指标

### 综合对比 (已完成)

| 指标 | Baseline | With Manager | 改进 |
|------|----------|--------------|------|
| 平均分数 | 8.05 | 12.73 | **+58.1%** |
| 平均奖励 | 6.17 | 17.64 | **+186.0%** |

### 专家正交性 (待测量)

| 指标 | 目标 | 说明 |
|------|------|------|
| Expert Usage Balance | 均衡分布 | 每个专家使用率20-30% |
| Separation Ratio | > 2.0 | 参数空间分离程度 |
| Expert Switches | > 10 | 专家切换次数 |

---

## 故障排除

### 问题1: PyTorch not found

**解决**: 在有PyTorch的环境中运行，或安装依赖

### 问题2: Checkpoint not found

**解决**: 
```bash
# 检查可用checkpoints
ls ablation_v3/results/resume_500_from_100/checkpoints/

# 修改脚本中的路径
```

### 问题3: Out of memory

**解决**: 减少inference episodes数量或max_steps

---

## 相关文档

1. **综合对比**:
   - `ablation_v3/500EP_COMPARISON_RESULTS.md` - 详细结果
   - `ablation_v3/VISUALIZATION_EXPLANATION.md` - 图表说明
   - `ablation_v3/RESULTS_INTERPRETATION_GUIDE.md` - 结果解读

2. **专家正交性**:
   - `ablation_v3/EXPERT_ORTHOGONALITY_VISUALIZATION_GUIDE.md` - 完整指南(英文)
   - `ablation_v3/专家正交性可视化说明.md` - 快速指南(中文)

3. **专家行为**:
   - `ablation_v3/专家行为分析说明.md` - 专家行为分析方案
   - `ablation_v3/EXPERT_ACTIVATION_ANALYSIS.md` - 专家激活分析

---

**创建时间**: 2026-01-13  
**状态**: 
- ✅ 综合对比工具完成并已生成图表
- 📋 专家正交性工具已创建，等待在PyTorch环境中运行
