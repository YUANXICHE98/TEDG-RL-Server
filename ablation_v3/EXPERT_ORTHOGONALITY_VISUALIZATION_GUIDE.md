# Expert Orthogonality Visualization Guide

## 目标

生成两个关键图表来证明专家的正交性：

1. **Expert Activation Heatmap** - 展示时间上的正交性（不同时刻激活不同专家）
2. **t-SNE Visualization of Expert Weights** - 展示参数空间的正交性（专家权重在参数空间中分离）

## 步骤

### Step 1: 提取真实训练数据

运行数据提取脚本来从训练好的模型中收集专家激活数据和权重：

```bash
# 确保在正确的Python环境中（需要torch, nle等依赖）
python3 tools/extract_real_expert_data.py
```

**这个脚本会做什么：**
- 加载训练好的checkpoint (`ablation_v3/results/resume_500_from_100/checkpoints/checkpoint_500.pt`)
- 运行5个inference episodes
- 在每个step记录alpha值（4个专家的激活强度）
- 提取每个专家网络的权重参数
- 分析专家使用模式和切换频率

**输出文件：**
- `ablation_v3/visualizations/expert_data/alpha_history.npy` - 所有step的alpha值
- `ablation_v3/visualizations/expert_data/expert_weights.npy` - 4个专家的权重
- `ablation_v3/visualizations/expert_data/episodes_analysis.json` - 详细分析

### Step 2: 生成可视化

使用提取的真实数据生成publication-quality的图表：

```bash
python3 tools/visualize_expert_orthogonality_real.py
```

**这个脚本会生成：**

1. **Expert Activation Heatmap** (`expert_activation_heatmap_real.png`)
   - 横轴：时间步 (Time Step)
   - 纵轴：4个专家 (Expert 0-3)
   - 颜色：Alpha值 (0-1)
   - **预期现象**：块状分布，不同时间段由不同专家主导

2. **t-SNE Visualization** (`expert_weights_tsne_real.png`)
   - 将高维专家权重降维到2D
   - 每个专家形成一个cluster
   - **预期现象**：4个离得很远的簇，证明参数空间正交性

3. **Combined Figure** (`expert_orthogonality_combined_real.png`)
   - 两个图并排显示
   - 适合放在论文中

**输出目录：**
```
ablation_v3/visualizations/expert_orthogonality/
├── expert_activation_heatmap_real.png
├── expert_weights_tsne_real.png
├── expert_orthogonality_combined_real.png
└── orthogonality_summary.json
```

## 如果没有PyTorch环境

如果你的当前环境没有安装PyTorch，你需要：

### 选项1：在训练环境中运行

```bash
# SSH到训练服务器
ssh your-training-server

# 激活训练环境
conda activate your-training-env  # 或者 source venv/bin/activate

# 运行脚本
cd /path/to/TEDG-RL-Server
python3 tools/extract_real_expert_data.py
python3 tools/visualize_expert_orthogonality_real.py

# 下载生成的图片
scp your-server:path/to/ablation_v3/visualizations/expert_orthogonality/*.png ./
```

### 选项2：安装依赖

```bash
# 安装必要的包
pip install torch numpy matplotlib seaborn scikit-learn nle

# 然后运行脚本
python3 tools/extract_real_expert_data.py
python3 tools/visualize_expert_orthogonality_real.py
```

## 预期结果

### 1. Expert Activation Heatmap

**好的结果应该显示：**
- ✅ 清晰的块状分布（block patterns）
- ✅ 不同时间段由不同专家主导
- ✅ 专家之间有明显的切换点
- ✅ 每个专家的使用率相对均衡（不是一个专家主导所有）

**示例解释：**
```
Steps 0-150:   Expert 0 主导 (红色) - 可能在战斗
Steps 150-250: Expert 1 主导 (蓝色) - 可能在探索
Steps 250-330: Expert 2 主导 (绿色) - 可能在管理物品
Steps 330-400: Expert 3 主导 (橙色) - 可能在治疗/恢复
```

这证明了**时间上的正交性** - 不同专家在不同时刻被激活。

### 2. t-SNE Visualization

**好的结果应该显示：**
- ✅ 4个清晰分离的cluster
- ✅ Cluster之间距离远
- ✅ Cluster内部紧密
- ✅ Separation Ratio > 2.0

**关键指标：**
- **Avg Inter-Cluster Distance**: 越大越好（>10）
- **Separation Ratio**: 越大越好（>2.0表示强正交性）

这证明了**参数空间的正交性** - 专家的权重在参数空间中是分离的。

## 论文中如何使用

### Figure Caption 示例

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{expert_orthogonality_combined_real.png}
\caption{Expert Orthogonality Analysis. 
(a) Expert Activation Heatmap shows temporal orthogonality: different experts 
are activated at different time steps, with clear block patterns indicating 
specialized behavior for different game scenarios. 
(b) t-SNE visualization of expert weights shows parameter space orthogonality: 
the four experts form well-separated clusters (separation ratio = X.XX), 
indicating that each expert has learned distinct representations.}
\label{fig:expert_orthogonality}
\end{figure}
```

### 文字描述示例

```
To validate the orthogonality of our expert networks, we conduct two analyses:

**Temporal Orthogonality**: Figure X(a) shows the expert activation heatmap 
over 1000 time steps. We observe clear block patterns where different experts 
dominate at different times. For example, Expert 0 is primarily activated 
during combat scenarios (steps 0-150), while Expert 3 handles healing and 
recovery (steps 330-400). This demonstrates that experts specialize in 
different temporal contexts.

**Parameter Space Orthogonality**: Figure X(b) shows t-SNE visualization of 
expert weights. The four experts form well-separated clusters with an average 
inter-cluster distance of X.XX and a separation ratio of X.XX, indicating 
strong orthogonality in parameter space. This confirms that each expert has 
learned distinct representations rather than redundant features.
```

## 故障排除

### 问题1：Checkpoint not found

```bash
# 检查可用的checkpoints
ls -la ablation_v3/results/resume_500_from_100/checkpoints/

# 修改脚本中的checkpoint路径
# 编辑 tools/extract_real_expert_data.py 第35行
```

### 问题2：Model architecture mismatch

这可能是因为checkpoint是用旧版本的模型保存的。需要：
1. 检查checkpoint中的模型结构
2. 确保当前代码的模型结构匹配

### 问题3：Out of memory

```bash
# 减少inference episodes数量
# 编辑 tools/extract_real_expert_data.py 第155行
num_episodes = 3  # 从5改为3

# 或减少max_steps
max_steps = 1000  # 从2000改为1000
```

## 文件清单

**创建的工具：**
1. `tools/extract_real_expert_data.py` - 提取真实训练数据
2. `tools/visualize_expert_orthogonality_real.py` - 生成可视化

**生成的数据：**
1. `ablation_v3/visualizations/expert_data/alpha_history.npy`
2. `ablation_v3/visualizations/expert_data/expert_weights.npy`
3. `ablation_v3/visualizations/expert_data/episodes_analysis.json`

**生成的图表：**
1. `ablation_v3/visualizations/expert_orthogonality/expert_activation_heatmap_real.png`
2. `ablation_v3/visualizations/expert_orthogonality/expert_weights_tsne_real.png`
3. `ablation_v3/visualizations/expert_orthogonality/expert_orthogonality_combined_real.png`

## 下一步

1. 运行数据提取脚本
2. 运行可视化脚本
3. 检查生成的图表
4. 将combined figure放入论文
5. 根据实际数据调整figure caption

---

**生成时间**: 2026-01-13  
**状态**: 📋 工具已创建，等待运行  
**优先级**: 高（论文需要）
