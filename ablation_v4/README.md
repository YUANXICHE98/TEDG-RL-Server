# Ablation V4: Causal-Gated Cross-Attention MoE

**版本**: V4  
**基于**: V3 (GAT-Guided Hierarchical MoE)  
**核心改动**: 将Concat融合替换为Causal-Gated Cross-Attention

---

## 🎯 核心改进

### V3 → V4的关键变化

| 组件 | V3 | V4 |
|------|----|----|
| **融合方式** | Simple Concat | **Causal-Gated Cross-Attention** |
| **信息流** | `z = concat(h_vis, h_logic)` | `c = CrossAttn(Q=h_logic, K=h_vis, V=h_vis)` |
| **注意力机制** | 无 | **Sparse Attention Gate** |
| **Router输入** | `z` (512维) | `c` (256维 Context Vector) |
| **Expert输入** | `z` (512维) | `c` (256维 Context Vector) |

### 为什么需要V4？

**V3的问题**（来自docsV4文档）：
1. **模态主导问题**: 简单concat可能导致某个模态（视觉或符号）主导决策
2. **信息冗余**: 512维的concat包含大量冗余信息
3. **缺乏交互**: 两个模态之间没有显式的交互机制

**V4的解决方案**：
1. **Cross-Attention**: 让符号信息（Query）主动查询视觉信息（Key/Value）
2. **Sparse Gate**: 只关注相关的视觉特征，过滤噪声
3. **Context Vector**: 生成紧凑的256维上下文表示

---

## 🏗️ 架构详解

### 3.2 The Semantic Prism: Causal-Gated Cross-Attention

```
输入:
  - h_logic (256维): GAT输出的高层意图
  - h_vis (256维): CNN输出的视觉特征

步骤:
  1. 投影:
     Q = W_Q @ h_logic  (256 → 256)
     K = W_K @ h_vis    (256 → 256)
     V = W_V @ h_vis    (256 → 256)
  
  2. 稀疏注意力门控:
     Attention = Softmax(Q @ K^T / √d_k)  (稀疏化)
     
  3. 语义过滤与聚合:
     c = Σ(Attention ⊙ V)  (256维 Context Vector)

输出:
  - c (256维): 融合后的上下文向量
```

### 数据流

```
Environment
    ↓
State (115维)
    ↓
┌─────────────────────────────────────┐
│ 3.1 Problem Formulation            │
│  ┌──────────┐      ┌──────────┐    │
│  │   GAT    │      │   CNN    │    │
│  │ h_logic  │      │  h_vis   │    │
│  │  (256)   │      │  (256)   │    │
│  └────┬─────┘      └────┬─────┘    │
└───────┼─────────────────┼──────────┘
        │                 │
        │    ┌────────────┘
        │    │
        ↓    ↓
┌─────────────────────────────────────┐
│ 3.2 Semantic Prism (NEW!)          │
│  ┌─────────────────────────────┐   │
│  │  Cross-Attention Fusion     │   │
│  │  Q ← h_logic                │   │
│  │  K,V ← h_vis                │   │
│  │  c = Attn(Q,K,V)            │   │
│  └──────────┬──────────────────┘   │
└─────────────┼──────────────────────┘
              │
              ↓ c (256维)
┌─────────────────────────────────────┐
│ 3.3 Entropy-Regularized Routing    │
│  ┌──────────┐    ┌──────────────┐  │
│  │ Router   │    │  Experts     │  │
│  │  α ← c   │    │  E_i ← c     │  │
│  └────┬─────┘    └──────┬───────┘  │
└───────┼──────────────────┼──────────┘
        │                  │
        └────────┬─────────┘
                 ↓
           Final Policy π
```

---

## 📁 文件结构

```
ablation_v4/
├── README.md                          # 本文件
├── train/
│   └── train_v4_cross_attention.py   # V4训练脚本
├── scripts/
│   ├── run_warmup_1000.sh            # Warmup阶段
│   ├── run_transition_3000.sh        # Transition阶段
│   └── run_finetune_5000.sh          # Fine-tune阶段
└── results/                           # 训练结果（自动生成）
```

---

## 🚀 快速开始

### 1. 烟雾测试

验证V4实现正确性：

```bash
python ablation_v4/test_v4_smoke.py
```

### 2. 小规模对比测试 (100 episodes)

快速验证V4相对V3的改进：

```bash
bash ablation_v4/scripts/run_v3_v4_comparison_100ep.sh
```

**这个脚本会**:
- 使用V3已有的warmup_1000结果（前100 episodes）作为基准
- 运行V4训练100 episodes
- 生成对比分析报告和可视化

**查看结果**:
```bash
# 查看对比图
open ablation_v4/results/v3_v4_comparison_100ep.png

# 查看详细日志
cat ablation_v4/results/test_100ep/logs/training_log.json
```

### 3. 完整三阶段训练

```bash
conda activate tedg-rl-demo

# Warmup阶段 (0-1000 episodes)
python ablation_v4/train/train_v4_cross_attention.py \
    --exp-name v4_warmup_1000 \
    --episodes 1000 \
    --max-steps 2000

# Transition阶段 (1001-3000 episodes)
python ablation_v4/train/train_v4_cross_attention.py \
    --exp-name v4_transition_3000 \
    --resume-from ablation_v4/results/v4_warmup_1000/checkpoints/model_final.pth \
    --episodes 3000 \
    --max-steps 2000

# Fine-tune阶段 (3001-5000 episodes)
python ablation_v4/train/train_v4_cross_attention.py \
    --exp-name v4_finetune_5000 \
    --resume-from ablation_v4/results/v4_transition_3000/checkpoints/model_final.pth \
    --episodes 5000 \
    --max-steps 2000
```

### 2. 对比V3 vs V4

```bash
# 分析V4结果
python tools/analyze_complete_5000ep_training.py \
    --v4-results ablation_v4/results/

# 对比V3和V4
python tools/compare_v3_v4.py \
    --v3-dir ablation_v3/results/ \
    --v4-dir ablation_v4/results/
```

---

## 🔬 实验假设

### 预期改进

| 指标 | V3 | V4 (预期) | 改进 |
|------|----|-----------| -----|
| **Alpha熵** | 0.693 (ln2) | 0.3-0.4 | -43% to -57% |
| **平均分数** | 10.68 | 20-25 | +87% to +134% |
| **模态平衡** | 不平衡 | 平衡 | ✓ |
| **注意力稀疏度** | N/A | 0.6-0.8 | ✓ |

### 关键假设

1. **Cross-Attention能缓解模态主导**
   - V3的concat可能让视觉特征主导
   - V4的Query-Key机制让符号信息主动查询

2. **Sparse Gate能过滤噪声**
   - 只关注相关的视觉特征
   - 减少冗余信息

3. **Context Vector更紧凑**
   - 256维 vs V3的512维
   - 更高效的信息表示

---

## 📊 与V3的对比

### 架构对比

| 组件 | V3 | V4 |
|------|----|----|
| **GAT输出** | h_logic (256) | h_logic (256) |
| **CNN输出** | h_vis (256) | h_vis (256) |
| **融合层** | Concat | **Cross-Attention** |
| **融合输出** | z (512) | **c (256)** |
| **Router输入维度** | 512 | **256** |
| **Expert输入维度** | 512 | **256** |
| **参数量** | ~2.5M | ~2.6M (+4%) |

### 训练配置（沿用V3）

- **三阶段训练**: Warmup → Transition → Fine-tune
- **Sparsemax路由**: 软中带硬
- **Manager约束**: Alignment + Semantic Orthogonality
- **负载均衡**: Load Balance Loss
- **专家数量**: 4 (Survival, Combat, Exploration, General)

---

## 🎓 理论基础

### Cross-Attention的优势

1. **显式交互**: Query-Key机制让两个模态显式交互
2. **选择性关注**: 只关注相关的视觉特征
3. **信息压缩**: 生成紧凑的上下文表示

### Sparse Attention Gate

```python
# 稀疏化注意力权重
Attention = Softmax(Q @ K^T / √d_k)
Attention = TopK(Attention, k=0.3)  # 只保留top 30%
```

**作用**:
- 过滤不相关的视觉特征
- 减少计算量
- 提高可解释性

---

## 📚 参考文献

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Transformer的Cross-Attention机制

2. **Sparse Attention** (Child et al., 2019)
   - 稀疏注意力的理论基础

3. **Mixture-of-Experts** (Shazeer et al., 2017)
   - MoE架构的原始论文

---

## ✅ 实现清单

- [x] 创建V4目录结构
- [x] 实现Cross-Attention融合层 (`src/core/networks_v4_cross_attention.py`)
- [x] 修改网络架构 (networks_v4_cross_attention.py)
- [x] 修改训练脚本 (train_v4_cross_attention.py) - **完整实现**
- [x] 创建训练脚本 (run_*.sh)
- [x] 创建烟雾测试 (test_v4_smoke.py)
- [ ] 运行烟雾测试验证
- [ ] 运行小规模验证 (100 episodes)
- [ ] 运行Warmup训练 (1000 episodes)
- [ ] 运行Transition训练 (3000 episodes)
- [ ] 运行Fine-tune训练 (5000 episodes)
- [ ] 实现对比分析工具 (compare_v3_v4.py)
- [ ] 分析结果并对比V3

---

## 🎯 下一步行动

### 1. 运行烟雾测试

```bash
python ablation_v4/test_v4_smoke.py
```

**测试内容**:
- ✓ 网络创建
- ✓ 前向传播
- ✓ Cross-Attention机制
- ✓ 专家路由
- ✓ 动作采样
- ✓ 梯度流
- ✓ V3 vs V4对比

### 2. 小规模验证 (100 episodes)

```bash
python ablation_v4/train/train_v4_cross_attention.py \
    --exp-name v4_test_100 \
    --episodes 100 \
    --max-steps 500 \
    --num-experts 4
```

**检查点**:
- 训练是否正常运行
- 损失是否收敛
- 专家是否被使用
- 无NaN/Inf

### 3. 完整三阶段训练

```bash
# 阶段1: Warmup (0-1000)
bash ablation_v4/scripts/run_warmup_1000.sh

# 阶段2: Transition (1000-3000)
bash ablation_v4/scripts/run_transition_3000.sh

# 阶段3: Fine-tune (3000-5000)
bash ablation_v4/scripts/run_finetune_5000.sh
```

---

## 📝 实现说明

### 完成的工作

1. **Cross-Attention融合层** (`CausalGatedCrossAttention`)
   - 4-head Multi-Head Attention
   - Sparse Attention Gate (top-30%)
   - 残差连接 + LayerNorm
   - 输出256维Context Vector

2. **V4网络架构** (`CrossAttentionMoEPolicy`)
   - 双流编码: Visual (CNN) + Logic (GAT)
   - Cross-Attention融合
   - 因果路由器 (输入Context Vector)
   - 4个语义专家 (输入Context Vector)
   - Critic网络

3. **完整训练脚本** (`train_v4_cross_attention.py`)
   - **完全沿用V3的训练逻辑**
   - 所有辅助损失函数 (Load Balance, Diversity, Alignment, Semantic, Temporal, Overlap)
   - 三阶段训练配置 (Warmup → Transition → Fine-tune)
   - NaN检测和回滚机制
   - 训练监控和诊断

4. **训练脚本**
   - `run_warmup_1000.sh`: Warmup阶段
   - `run_transition_3000.sh`: Transition阶段
   - `run_finetune_5000.sh`: Fine-tune阶段

5. **烟雾测试** (`test_v4_smoke.py`)
   - 7个测试用例
   - V3 vs V4对比
   - 参数量和速度对比

### 关键设计决策

1. **完全沿用V3的训练逻辑**
   - 只替换网络类为`CrossAttentionMoEPolicy`
   - 所有辅助损失函数保持不变
   - 训练配置保持不变
   - 确保公平对比

2. **Context Vector维度: 256**
   - 相比V3的512维concat减少50%
   - 更紧凑的表示
   - 减少参数量和计算量

3. **Sparse Attention Gate: top-30%**
   - 过滤不相关的视觉特征
   - 提高可解释性
   - 可调参数: `sparse_topk`

4. **4-head Multi-Head Attention**
   - 平衡表达能力和计算效率
   - 每个head维度: 256/4 = 64

---

**创建时间**: 2026-01-22  
**状态**: 🚧 开发中  
**基于**: V3 (GAT-Guided Hierarchical MoE)
