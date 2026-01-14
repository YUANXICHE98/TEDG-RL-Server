# TEDG-RL V3 架构设计文档

> **版本**: V3.0 - GAT-Guided Hierarchical MoE  
> **创建日期**: 2025-01-05  
> **状态**: 设计阶段  
> **核心创新**: 因果图引导的层级混合专家 (Causal-Graph Guided Hierarchical MoE)

---

## 一、科学问题与研究动机

### 1.1 宏观科学问题

**如何在长程、稀疏奖励且规则复杂的环境中，实现样本高效且可解释的策略学习？**

NetHack 是这个问题的完美载体：
- 纯 RL 样本效率低，探索浪费在不可行动作上
- 纯规则系统无法泛化到未见场景
- 需要结合结构化先验知识与自适应学习

### 1.2 微观科学问题

**结构化意图解耦 (Structured Intent Disentanglement)**

**痛点**: 传统 RL Agent 的隐状态是"一团浆糊"的黑盒，在同一时刻既想打架又想逃跑，导致动作震荡。

**假设**: 如果能将 Agent 的隐状态强制解耦为正交的意图（如"生存意图" vs "探索意图"），并利用超图先验作为因果引导，就能大幅降低策略搜索空间。

### 1.3 V1/V2 的局限性

| 版本 | 核心方法 | 主要问题 |
|------|----------|----------|
| **V1** | 手工特征 + 软融合 | • 115维state依赖手工特征工程<br>• q_pre/q_scene等是启发式压缩<br>• 专家分工不明确（softmax平均） |
| **V2** | Gumbel路由 + 稀疏MoE | • 仍依赖手工特征<br>• 硬路由容易塌缩<br>• 专家语义不明确（仍是pre/scene/effect/rule） |
| **HRAM** | 检索增强 + 交叉注意力 | • 端到端学习不稳定<br>• 缺乏结构化约束<br>• 可解释性不足 |

### 1.4 V3 的核心创新

**用图神经网络将超图知识转化为可微的因果推理，指导语义化专家的动态选择**

1. **超图拓扑结构**: 节点共享，支持消息传递（已完成转换）
2. **GAT推理层**: 动态激活节点，传导因果信号
3. **语义对齐专家**: 明确定义 Survival/Combat/Exploration/General
4. **Sparsemax路由**: 软中带硬，避免平均主义和塌缩
5. **可解释性**: GAT注意力热图 + 专家选择可视化

---

## 二、V3 整体架构

### 2.1 端到端数据流

```
obs(dict) → NetHack观测
  ↓
[状态解析模块]
  ├─ blstats → 数值特征
  ├─ glyphs → 视觉特征 (可选CNN)
  └─ atoms → 激活的超图节点
  ↓
[双流编码器]
  ├─ Visual Stream: CNN(glyphs) → h_vis (256)
  └─ Logic Stream: GAT(超图, active_nodes) → h_logic (256)
  ↓
[因果路由器]
  z = Concat(h_vis, h_logic) → Sparsemax → α(4)
  ↓
[语义专家层]
  ├─ Survival Expert (α₁)
  ├─ Combat Expert (α₂)
  ├─ Exploration Expert (α₃)
  └─ General Expert (α₄)
  ↓
[动作融合]
  fused_logits = Σ αᵢ · Expert_i(h_vis)
  ↓
[Critic网络]
  value = Critic(z)
  ↓
action ~ Categorical(fused_logits)
```

### 2.2 关键设计决策

| 设计点 | V1/V2 | V3 | 理由 |
|--------|-------|-----|------|
| **状态表示** | 手工115维 | GAT图嵌入 | 端到端学习因果关系 |
| **专家定义** | pre/scene/effect/rule | Survival/Combat/Exploration/General | 语义化，可解释 |
| **路由机制** | Softmax/Gumbel | Sparsemax | 软中带硬，稳定训练 |
| **知识利用** | 覆盖率/检索 | GAT消息传递 | 动态激活，多跳推理 |
| **可解释性** | α权重 | GAT热图 + α权重 | 双层可视化 |

---

## 三、核心模块设计

### 3.1 超图GAT推理层

**文件**: `src/core/hypergraph_gat.py`

#### 3.1.1 输入输出

**输入**:
- `edge_index`: (2, num_edges) - 超图边索引
- `edge_attr`: (num_edges,) - 边类型 (satisfies/context_of/leads_to)
- `node_types`: (num_nodes,) - 节点类型 (condition/operator/effect)
- `active_mask`: (num_nodes,) - 当前激活的节点 (0/1)

**输出**:
- `operator_embeddings`: (num_operators, hidden_dim) - 操作符节点嵌入
- `attention_scores`: (num_operators,) - 操作符激活分数
- `intent_vector`: (hidden_dim,) - 全局意图向量

#### 3.1.2 网络结构

```python
class HypergraphGAT(nn.Module):
    """
    超图GAT推理层
    
    设计思想:
    1. 节点初始化: 根据类型和激活状态初始化节点特征
    2. 消息传递: 2层GAT，信息从Condition流向Operator流向Effect
    3. 操作符聚合: 提取Operator节点的嵌入和注意力分数
    4. 意图提取: 全局Readout得到Intent Vector
    """
    
    def __init__(self, 
                 num_nodes: int,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        # 节点类型嵌入 (condition=0, operator=1, effect=2)
        self.node_type_embedding = nn.Embedding(3, hidden_dim)
        
        # 2层GAT
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, 
                           dropout=dropout, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=num_heads,
                           dropout=dropout, concat=False)
        
        # 全局Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
```

#### 3.1.3 动态激活机制

```python
def activate_nodes(self, atoms: Dict[str, List[str]]) -> torch.Tensor:
    """
    根据当前游戏状态激活超图节点
    
    Args:
        atoms: {"pre_nodes": [...], "scene_atoms": [...], ...}
    
    Returns:
        active_mask: (num_nodes,) 0/1向量
    """
    active_mask = torch.zeros(self.num_nodes)
    
    # 激活Condition节点
    for atom in atoms.get("pre_nodes", []) + atoms.get("scene_atoms", []):
        node = self.loader.get_node_by_label(atom)
        if node and node['type'] == 'condition':
            active_mask[node['id']] = 1.0
    
    return active_mask
```

### 3.2 因果路由器 (Causal Router)

**文件**: `src/core/networks_v3_gat_moe.py`

#### 3.2.1 Sparsemax vs Softmax vs Gumbel

| 机制 | 特点 | 优点 | 缺点 |
|------|------|------|------|
| **Softmax** | 所有专家都激活 | 平滑，稳定 | 平均主义，分工不明 |
| **Gumbel** | 训练时one-hot | 强制分工 | 容易塌缩 |
| **Sparsemax** | 自动稀疏化 | 软中带硬，不相关的权重为0 | 需要调节温度 |

**V3选择Sparsemax的理由**:
- 比Softmax更果断（能把不相关专家权重压为0）
- 比Gumbel更平滑（避免训练不稳定）
- 结合GAT的因果偏置，能学到"场景→专家"的明确映射

```python
class CausalRouter(nn.Module):
    """
    因果路由器 - 使用Sparsemax实现软中带硬的路由
    """
    
    def __init__(self, input_dim: int = 512, num_experts: int = 4):
        super().__init__()
        
        self.router = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, 512) - Concat(h_vis, h_logic)
        
        Returns:
            alpha: (batch, 4) - Sparsemax权重
        """
        logits = self.router(z)
        alpha = sparsemax(logits, dim=-1)  # 自动稀疏化
        return alpha
```

### 3.3 语义对齐专家 (Semantic Experts)

**文件**: `src/core/networks_v3_gat_moe.py`

#### 3.3.1 专家定义

| 专家 | 关注场景 | 典型动作 | 训练信号来源 |
|------|----------|----------|--------------|
| **Survival** | hp_low, hunger, poison | eat, pray, quaff_potion | 生存相关奖励 |
| **Combat** | adjacent_to_monster, has_weapon | attack, move_toward, kick | 战斗相关奖励 |
| **Exploration** | unexplored, has_key, near_door | search, move, open_door | 探索相关奖励 |
| **General** | 其他/兜底 | wait, look, inventory | 通用奖励 |

#### 3.3.2 专家网络结构

```python
class SemanticExpert(nn.Module):
    """
    语义专家 - 每个专家是一个小型MLP
    
    设计思想:
    - 输入: h_vis (视觉特征，所有专家共享)
    - 输出: logits(23) 动作分布
    - 每个专家独立训练，梯度不互相干扰
    """
    
    def __init__(self, input_dim: int = 256, action_dim: int = 23):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # 小增益初始化，防止初始logits过大
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.constant_(self.network[-1].bias, 0)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.network(h)
```

### 3.4 V3完整策略网络

**文件**: `src/core/networks_v3_gat_moe.py`

```python
class GATGuidedMoEPolicy(nn.Module):
    """
    V3: GAT引导的混合专家策略网络
    
    架构流程:
    1. 双流编码: Visual + Logic (GAT)
    2. 因果路由: Sparsemax选择专家
    3. 专家融合: 加权组合
    4. 价值估计: Critic网络
    """
    
    def __init__(self,
                 hypergraph_path: str,
                 num_nodes: int = 527,
                 hidden_dim: int = 256,
                 action_dim: int = 23):
        super().__init__()
        
        # 1. 超图GAT
        self.gat = HypergraphGAT(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim
        )
        
        # 2. 视觉编码器 (可选，当前用blstats)
        self.visual_encoder = nn.Sequential(
            nn.Linear(115, 256),  # blstats等数值特征
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        
        # 3. 因果路由器
        self.router = CausalRouter(
            input_dim=hidden_dim * 2,  # h_vis + h_logic
            num_experts=4
        )
        
        # 4. 四个语义专家
        self.experts = nn.ModuleList([
            SemanticExpert(hidden_dim, action_dim)  # Survival
            for _ in range(4)
        ])
        
        # 5. Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state: torch.Tensor, atoms: Dict) -> Tuple:
        """
        前向传播
        
        Args:
            state: (batch, 115) 数值状态
            atoms: 当前激活的atoms字典
        
        Returns:
            logits: (batch, 23)
            alpha: (batch, 4) 专家权重
            value: (batch, 1)
            gat_attention: GAT注意力分数 (可视化用)
        """
        # 1. 双流编码
        h_vis = self.visual_encoder(state)  # (batch, 256)
        h_logic, gat_attention = self.gat(atoms)  # (batch, 256), attention
        
        # 2. 拼接
        z = torch.cat([h_vis, h_logic], dim=-1)  # (batch, 512)
        
        # 3. 路由
        alpha = self.router(z)  # (batch, 4)
        
        # 4. 专家输出
        expert_logits = torch.stack([
            expert(h_vis) for expert in self.experts
        ], dim=1)  # (batch, 4, 23)
        
        # 5. 融合
        alpha_expanded = alpha.unsqueeze(2)  # (batch, 4, 1)
        fused_logits = (alpha_expanded * expert_logits).sum(dim=1)
        
        # 6. 价值
        value = self.critic(z)
        
        return fused_logits, alpha, value, gat_attention
```

---

## 四、训练流程设计

### 4.1 训练脚本

**文件**: `ablation_v3/train/train_v3_gat_moe.py`

### 4.2 超参数设置

| 参数 | V1/V2 | V3 | 理由 |
|------|-------|-----|------|
| **learning_rate** | 3e-4 | 1e-4 | GAT需要更小学习率 |
| **gat_layers** | - | 2 | 2层足够，避免过平滑 |
| **gat_heads** | - | 4 | 多头注意力 |
| **hidden_dim** | 128 | 256 | GAT需要更大容量 |
| **entropy_coef** | 0.05 | 0.01 | Sparsemax已稀疏，降低熵正则 |
| **alpha_entropy_coef** | 0.1 | 0.05 | 同上 |

### 4.3 训练技巧

1. **Warmup阶段** (前1000 episodes)
   - 使用Softmax路由（不用Sparsemax）
   - 让专家先学到基础策略
   - 避免过早塌缩

2. **温度退火**
   - Sparsemax温度从1.0逐渐降到0.5
   - 让路由逐渐变硬

3. **辅助损失**
   - Next-Intent Prediction: 预测下一步哪个Operator会激活
   - 强迫GAT学习因果关系

### 4.4 评估指标

| 指标 | 含义 | 目标 |
|------|------|------|
| **episode_score** | NetHack分数 | 越高越好 |
| **alpha_entropy** | 路由熵 | 适中（0.5-1.0） |
| **expert_usage** | 每个专家被选中次数 | 均衡（避免塌缩） |
| **gat_attention_variance** | GAT注意力方差 | 高（说明动态激活） |
| **operator_activation_rate** | Operator节点激活率 | 适中（10-30%） |

---

## 五、可解释性与可视化

### 5.1 双层可视化

**层1: GAT注意力热图**
- 显示哪些Condition节点激活
- 哪些Operator节点被点亮
- 消息传递路径

**层2: 专家选择热图**
- 不同场景下α的分布
- 专家-场景对应关系
- 是否学到语义分工

### 5.2 可视化工具

**文件**: `tools/visualize_v3_gat_attention.py`

```python
def visualize_gat_attention(episode_data):
    """
    可视化GAT注意力和专家选择
    
    输出:
    1. GAT子图: 当前激活的节点和边
    2. 专家权重条形图: α分布
    3. 时序热图: 整个episode的α变化
    """
    pass
```

---

## 六、实验设计

### 6.1 消融实验

| 实验组 | 配置 | 目的 |
|--------|------|------|
| **v3_full** | GAT + Sparsemax + 4专家 | 完整V3 |
| **v3_no_gat** | 手工特征 + Sparsemax + 4专家 | 验证GAT贡献 |
| **v3_softmax** | GAT + Softmax + 4专家 | 验证Sparsemax贡献 |
| **v3_2experts** | GAT + Sparsemax + 2专家 | 验证专家数量 |

### 6.2 对比基线

- **V1 baseline**: 手工特征 + Softmax
- **V2 gumbel**: 手工特征 + Gumbel
- **V2 hram_e2e**: 检索增强端到端
- **V3 full**: GAT + Sparsemax + 语义专家

### 6.3 预期结果

| 指标 | V1 | V2 | V3 (预期) |
|------|-----|-----|-----------|
| **best_score** | 500-600 | 600-700 | **800+** |
| **sample_efficiency** | 基线 | 1.2x | **1.5x** |
| **alpha_interpretability** | 低 | 中 | **高** |
| **training_stability** | 中 | 低 (塌缩) | **高** |

---

## 七、实现路线图

### Phase 1: 核心模块实现 (1-2天)
- [ ] `src/core/hypergraph_gat.py` - GAT推理层
- [ ] `src/core/networks_v3_gat_moe.py` - V3策略网络
- [ ] 单元测试: 确保形状正确，无NaN

### Phase 2: 训练脚本 (1天)
- [ ] `ablation_v3/train/train_v3_gat_moe.py`
- [ ] 小规模测试 (100 episodes)
- [ ] 确保训练稳定

### Phase 3: 可视化工具 (1天)
- [ ] `tools/visualize_v3_gat_attention.py`
- [ ] GAT热图
- [ ] 专家选择分析

### Phase 4: 全面实验 (3-5天)
- [ ] 运行4组消融实验
- [ ] 对比V1/V2基线
- [ ] 生成论文图表

### Phase 5: 论文撰写 (持续)
- [ ] 方法论章节
- [ ] 实验结果章节
- [ ] 可视化案例分析

---

## 八、风险与缓解策略

### 8.1 潜在风险

| 风险 | 可能性 | 影响 | 缓解策略 |
|------|--------|------|----------|
| **GAT过平滑** | 中 | 高 | 限制2层，使用残差连接 |
| **专家塌缩** | 中 | 高 | Warmup + 温度退火 + 熵正则 |
| **训练不稳定** | 中 | 中 | 小学习率 + LayerNorm + 梯度裁剪 |
| **显存不足** | 低 | 中 | 减小batch_size或hidden_dim |

### 8.2 降级方案

如果V3训练失败，可以：
1. 先用固定GAT（不训练），只训练路由和专家
2. 用预训练的GAT嵌入作为特征
3. 回退到V2 + GAT特征的混合方案

---

## 九、论文贡献点

### 9.1 核心贡献

1. **方法创新**: 首次将超图GAT用于RL的因果推理和专家路由
2. **架构创新**: Sparsemax路由 + 语义对齐专家
3. **可解释性**: 双层可视化（GAT + 专家选择）
4. **实验验证**: 在NetHack上验证有效性

### 9.2 论文标题候选

- "Causal Graph-Guided Hierarchical Mixture-of-Experts for Sample-Efficient Reinforcement Learning"
- "Structured Intent Disentanglement via Hypergraph Attention for Complex Decision Making"
- "GAT-MoE: Graph Attention Guided Expert Routing for Interpretable Reinforcement Learning"

---

## 十、参考文献

### 10.1 相关工作

- **Graph Neural Networks**: GAT (Veličković et al., 2018)
- **Mixture of Experts**: Switch Transformer (Fedus et al., 2021)
- **Neuro-Symbolic RL**: Graph-based RL (Jiang et al., 2019)
- **NetHack**: NLE (Küttler et al., 2020), AutoAscend

### 10.2 技术栈

- PyTorch 2.0+
- PyTorch Geometric 2.3+
- NetHack Learning Environment (NLE)
- Weights & Biases (可选，用于实验追踪)

---

## 附录A: 代码文件清单

```
src/core/
├── hypergraph_gat.py              # GAT推理层
├── networks_v3_gat_moe.py         # V3策略网络
├── hypergraph_gat_loader.py       # 超图数据加载器 (已完成)
└── ppo_trainer.py                 # PPO训练器 (复用V2)

ablation_v3/
├── train/
│   └── train_v3_gat_moe.py        # V3训练脚本
├── scripts/
│   └── run_v3_experiments.sh      # 实验启动脚本
└── results/
    └── <exp_name>/                # 实验结果

tools/
├── visualize_v3_gat_attention.py  # GAT可视化
└── compare_v1_v2_v3.py            # 版本对比

docsV3/
├── V3_ARCHITECTURE_DESIGN.md      # 本文档
├── 超图修改.md                     # 超图转换说明 (已完成)
└── 语义正交MOE.md                  # 原始设计思路
```

---

## 附录B: 快速开始

```bash
# 1. 确保超图已转换
ls data/hypergraph/hypergraph_gat_structure.json

# 2. 测试GAT加载
conda activate tedg-rl-demo
python -c "from src.core.hypergraph_gat_loader import HypergraphGATLoader; \
           loader = HypergraphGATLoader('data/hypergraph/hypergraph_gat_structure.json'); \
           print('✓ GAT loader works')"

# 3. 实现核心模块 (下一步)
# 创建 src/core/hypergraph_gat.py
# 创建 src/core/networks_v3_gat_moe.py

# 4. 运行小规模测试
python ablation_v3/train/train_v3_gat_moe.py --episodes 100 --exp-name v3_test

# 5. 全面实验
bash ablation_v3/scripts/run_v3_experiments.sh
```

---

**文档状态**: ✅ 设计完成，待实现  
**下一步**: 实现 `src/core/hypergraph_gat.py`
