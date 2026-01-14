# V3 架构数据流向详解

**当前训练状态**: Episode ~1300/3000, Transition阶段, Sparsemax路由已启动

---

## 完整数据流程图

```
NetHack环境观察
    ↓
[1] 状态提取 (State Constructor)
    ↓
[2] 超图匹配 (Hypergraph Matcher)
    ↓
[3] GAT推理层 (HypergraphGAT) ← 可训练
    ↓
[4] 场景嵌入 (Scene Embedding)
    ↓
[5] 路由网络 (Router) ← 可训练
    ↓
[6] 专家权重 (Alpha) ← Sparsemax/Softmax
    ↓
[7] 4个语义专家 (Experts) ← 可训练
    ↓
[8] 加权融合 (Weighted Sum)
    ↓
[9] 动作概率分布
    ↓
采样动作 → NetHack环境
```

---

## 详细流程分解

### [1] NetHack环境观察 → 状态提取

**输入**: NetHack原始观察
```python
obs = {
    'glyphs': (21, 79),      # 地图符号
    'blstats': (25,),        # 基础统计
    'message': str,          # 游戏消息
    'inv_*': ...,           # 物品栏
}
```

**处理**: StateConstructor
```python
# src/core/state_constructor.py
state = [
    # 基础状态 (25维)
    HP, MaxHP, Hunger, AC, Level, ...
    
    # 环境感知 (90维)
    nearby_monsters, nearby_walls, nearby_items, ...
]
```

**输出**: 状态向量
- **维度**: 115维
- **可训练**: ❌ 否（特征工程）

---

### [2] 状态 → 超图匹配

**输入**: 状态向量 (115维)

**处理**: HypergraphMatcher
```python
# src/core/hypergraph_matcher.py
# 匹配当前状态到超图中的场景原子

matched_atoms = matcher.match(state)
# 例如: ['low_hp', 'monster_nearby', 'corridor']
```

**超图数据**:
```
data/hypergraph/hypergraph.json
├── 450条超边 (规则)
├── 65个前置节点 (条件)
├── 82个场景原子 (scene_atoms)
└── 101个效果节点 (结果)
```

**输出**: 激活的场景原子列表
- **维度**: 可变长度列表
- **可训练**: ❌ 否（规则匹配）

---

### [3] 场景原子 → GAT推理层

**输入**: 激活的场景原子

**处理**: HypergraphGAT (核心可训练组件)
```python
# src/core/hypergraph_gat.py
class HypergraphGAT(nn.Module):
    def __init__(self):
        # 图结构
        self.num_nodes = 527      # 总节点数
        self.num_edges = 3016     # 总边数
        self.operator_nodes = 279 # 操作符节点
        
        # 可训练层
        self.node_embedding = nn.Embedding(527, 256)  # ✅ 可训练
        self.gat_layers = nn.ModuleList([
            GATConv(256, 256, heads=4),  # Layer 1 ✅ 可训练
            GATConv(256*4, 256, heads=4), # Layer 2 ✅ 可训练
        ])
```

**GAT层数**: 2层
- **Layer 1**: 256维 → 256维 × 4头 = 1024维
- **Layer 2**: 1024维 → 256维 × 4头 = 1024维
- **输出**: 256维场景嵌入

**注意力机制**:
```python
# 每层GAT计算注意力权重
attention = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))

# 4个注意力头并行计算
h' = concat(head_1, head_2, head_3, head_4)
```

**输出**: 场景嵌入向量
- **维度**: 256维
- **可训练**: ✅ 是（2层GAT + Embedding）
- **参数量**: ~500K

---

### [4] 场景嵌入 → 路由网络

**输入**: 场景嵌入 (256维)

**处理**: Router网络
```python
# src/core/networks_v3_gat_moe.py
class Router(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(256, 128)  # ✅ 可训练
        self.fc2 = nn.Linear(128, 4)    # ✅ 可训练
        # 初始化: gain=0.1
    
    def forward(self, scene_emb):
        x = F.relu(self.fc1(scene_emb))
        logits = self.fc2(x)  # (batch, 4)
        return logits
```

**输出**: 路由logits
- **维度**: 4维（对应4个专家）
- **可训练**: ✅ 是
- **参数量**: ~33K

---

### [5] 路由logits → 专家权重 (Alpha)

**输入**: 路由logits (4维)

**处理**: Sparsemax/Softmax
```python
# 当前阶段: Transition (Episode 1300)
# 使用: Sparsemax ✅

if self.use_sparsemax:
    alpha = sparsemax(logits / temperature)
else:
    alpha = softmax(logits)

# 温度退火 (Transition阶段)
progress = (episode - 1000) / 2000  # 当前: 0.15
temperature = 1.0 - 0.5 * progress  # 当前: 0.925
```

**Sparsemax特性**:
- 输出稀疏（部分专家权重为0）
- 鼓励专家专业化
- 软中带硬

**输出**: 专家权重 Alpha
- **维度**: 4维，和为1
- **范围**: [0, 1]
- **稀疏性**: 部分为0（Sparsemax）
- **可训练**: ❌ 否（激活函数）

**当前状态** (Episode 1300):
- **路由方式**: Sparsemax ✅
- **温度**: 0.925
- **预期Alpha熵**: ~1.2-1.3（从1.385下降中）

---

### [6] Alpha + 状态 → 4个语义专家

**输入**: 
- 状态向量 (115维)
- Alpha权重 (4维)

**处理**: 4个并行的语义专家
```python
# src/core/networks_v3_gat_moe.py
class SemanticExpert(nn.Module):
    def __init__(self, name):
        self.name = name  # Survival/Combat/Exploration/General
        self.fc1 = nn.Linear(115, 256)  # ✅ 可训练
        self.fc2 = nn.Linear(256, 128)  # ✅ 可训练
        self.fc3 = nn.Linear(128, 23)   # ✅ 可训练
        # 初始化: gain=0.5 (修复后)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)  # (batch, 23)
        return logits
```

**4个专家**:
1. **Survival Expert** (生存专家)
   - 专注: 治疗、逃跑、休息
   - 触发场景: 低血量、饥饿、中毒

2. **Combat Expert** (战斗专家)
   - 专注: 攻击、防御、战术
   - 触发场景: 怪物附近、战斗中

3. **Exploration Expert** (探索专家)
   - 专注: 移动、搜索、开门
   - 触发场景: 新区域、走廊、房间

4. **General Expert** (通用专家)
   - 专注: 物品管理、装备、其他
   - 触发场景: 安全时、物品栏操作

**输出**: 4个专家的动作logits
- **维度**: (4, 23) - 4个专家，每个23维
- **可训练**: ✅ 是
- **参数量**: ~120K × 4 = ~480K

---

### [7] 专家logits + Alpha → 加权融合

**输入**:
- 专家logits: (4, 23)
- Alpha权重: (4,)

**处理**: 加权求和
```python
# 广播乘法 + 求和
action_logits = (alpha.unsqueeze(-1) * expert_logits).sum(dim=1)

# 等价于:
action_logits = sum(alpha[i] * expert_logits[i] for i in range(4))
```

**示例** (假设当前状态):
```python
# 假设Alpha权重 (Sparsemax输出)
alpha = [0.6, 0.3, 0.1, 0.0]  # Survival主导，Combat辅助

# 专家logits (简化)
survival_logits  = [0.5, 0.2, ...]  # 23维
combat_logits    = [0.1, 0.8, ...]  # 23维
explore_logits   = [0.3, 0.1, ...]  # 23维
general_logits   = [0.2, 0.3, ...]  # 23维

# 加权融合
action_logits = 0.6*survival + 0.3*combat + 0.1*explore + 0.0*general
              = [0.36, 0.36, ...]  # 23维
```

**输出**: 最终动作logits
- **维度**: 23维（NetHack动作空间）
- **可训练**: ❌ 否（加权求和）

---

### [8] 动作logits → 动作概率分布

**输入**: 动作logits (23维)

**处理**: Softmax
```python
action_probs = F.softmax(action_logits, dim=-1)
```

**输出**: 动作概率分布
- **维度**: 23维
- **范围**: [0, 1]，和为1
- **可训练**: ❌ 否（激活函数）

---

### [9] 动作概率 → 采样动作

**输入**: 动作概率 (23维)

**处理**: 
```python
# 训练时: 随机采样
action = torch.multinomial(action_probs, 1)

# 推理时: 贪心选择
action = torch.argmax(action_probs)
```

**输出**: 动作索引
- **维度**: 标量 (0-22)
- **映射**: NetHack动作

**NetHack动作空间** (23个):
```
0: 向北移动
1: 向东移动
2: 向南移动
3: 向西移动
4: 向东北移动
5: 向东南移动
6: 向西南移动
7: 向西北移动
8: 搜索
9: 休息
10: 拾取物品
11: 使用物品
12: 攻击
...
22: 其他
```

---

## 可训练参数总结

| 组件 | 参数量 | 是否训练 | 初始化 |
|------|--------|----------|--------|
| **GAT Embedding** | ~135K | ✅ 是 | Xavier |
| **GAT Layer 1** | ~265K | ✅ 是 | Xavier |
| **GAT Layer 2** | ~265K | ✅ 是 | Xavier |
| **Router** | ~33K | ✅ 是 | gain=0.1 |
| **Survival Expert** | ~120K | ✅ 是 | gain=0.5 |
| **Combat Expert** | ~120K | ✅ 是 | gain=0.5 |
| **Exploration Expert** | ~120K | ✅ 是 | gain=0.5 |
| **General Expert** | ~120K | ✅ 是 | gain=0.5 |
| **总计** | **~1.21M** | ✅ 全部 | - |

---

## 当前训练状态检查

### Episode 1300 状态

**训练阶段**: Transition
- ✅ Sparsemax路由已启动
- ✅ 学习率降低到5e-5
- ✅ 温度退火中 (0.925)

**预期变化**:
```
Alpha熵: 1.385 (Warmup) → ~1.2-1.3 (当前) → 0.7 (目标)
专家使用: 均匀分布 → 开始分化 → 明确专业化
```

### 专家是否启动？

**判断标准**:
1. ✅ **Sparsemax已启动** - Episode 1000自动切换
2. ⏳ **Alpha熵下降中** - 需要检查日志确认
3. ⏳ **专家权重分化** - 需要可视化确认

**如何验证**:
```bash
# 加载最新checkpoint
checkpoint = torch.load('ablation_v3/results/transition_3000/checkpoints/model_01300.pth')

# 检查Alpha分布
# 如果专家启动，应该看到:
# - Alpha不再是[0.25, 0.25, 0.25, 0.25]
# - 某些专家权重明显更高
# - Alpha熵 < 1.385
```

---

## 数据维度变化追踪

```
NetHack观察 (原始)
    ↓
状态向量 (115维)
    ↓
场景原子 (可变长度列表)
    ↓
GAT Layer 1 (256维 → 1024维)
    ↓
GAT Layer 2 (1024维 → 256维)
    ↓
场景嵌入 (256维)
    ↓
Router FC1 (256维 → 128维)
    ↓
Router FC2 (128维 → 4维)
    ↓
路由logits (4维)
    ↓
Alpha权重 (4维, Sparsemax)
    ↓
专家并行处理:
  - Expert FC1 (115维 → 256维)
  - Expert FC2 (256维 → 128维)
  - Expert FC3 (128维 → 23维)
    ↓
专家logits (4 × 23维)
    ↓
加权融合 (23维)
    ↓
动作概率 (23维, Softmax)
    ↓
动作 (标量)
```

---

## 关键设计决策

### 1. 为什么用GAT而不是普通GNN？

**GAT优势**:
- ✅ 注意力机制：自动学习节点重要性
- ✅ 多头注意力：捕获不同类型关系
- ✅ 动态权重：根据场景调整

### 2. 为什么用Sparsemax而不是Softmax？

**Sparsemax优势**:
- ✅ 稀疏输出：部分专家权重为0
- ✅ 鼓励专业化：避免所有专家都参与
- ✅ 软中带硬：比Gumbel-Softmax更稳定

### 3. 为什么要4个语义专家？

**设计理由**:
- ✅ 语义正交：Survival/Combat/Exploration/General互不重叠
- ✅ 覆盖全面：涵盖NetHack主要场景
- ✅ 可解释性：每个专家有明确职责

### 4. 为什么要分阶段训练？

**Curriculum Learning**:
- ✅ Warmup: 让所有专家学到基础知识
- ✅ Transition: 逐渐引入稀疏性
- ✅ Fine-tune: 完全专业化

---

## 梯度反向传播路径

```
Loss (PPO)
    ↓
动作logits (23维)
    ↓
加权融合 ← Alpha权重
    ↓           ↓
专家logits    Router
    ↓           ↓
专家网络    场景嵌入
    ↓           ↓
状态向量    GAT层
              ↓
          节点嵌入
```

**所有参数都接收梯度** ✅

---

## 推理时的简化

训练完成后，推理时可以简化：

```python
# 推理时固定use_sparsemax=True
def inference(state):
    # 1. GAT推理
    scene_emb = gat(state)
    
    # 2. 路由
    alpha = sparsemax(router(scene_emb))
    
    # 3. 专家
    expert_logits = [expert(state) for expert in experts]
    
    # 4. 融合
    action_logits = (alpha @ expert_logits)
    
    # 5. 贪心选择
    action = argmax(action_logits)
    
    return action
```

**无需if语句，完全端到端** ✅

---

## 总结

### 数据流向
```
NetHack → 状态(115维) → 超图匹配 → GAT(2层,256维) → 
场景嵌入(256维) → Router(4维) → Sparsemax → Alpha(4维) → 
4个专家(并行) → 加权融合(23维) → 动作
```

### 可训练组件
- ✅ GAT (2层, ~665K参数)
- ✅ Router (~33K参数)
- ✅ 4个专家 (~480K参数)
- ✅ 总计: ~1.21M参数

### 当前状态 (Episode 1300)
- ✅ Sparsemax路由已启动
- ✅ 温度退火中 (0.925)
- ⏳ Alpha熵下降中
- ⏳ 专家开始专业化

---

**文档生成时间**: 2026-01-08 19:00  
**训练状态**: Episode ~1300/3000, Transition阶段  
**下一步**: 等待训练完成，分析专家专业化程度
