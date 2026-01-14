# V3高级机制实现计划 - 超越Manager约束

## 背景

Manager内层约束（超图-路由对齐）已经实现，但Alpha熵仍可能停滞在0.69。这个数字意味着Router在2个专家之间"和稀泥"，而不是真正专业化。

本文档基于`除了加上内部奖励之外的修改部分.md`的深度分析，提出三个核心机制的实现方案。

---

## 核心问题分析

### 问题1: 缺乏时间一致性（记忆缺失）

**现象**: 
- Agent在相邻帧之间频繁切换专家
- 上一帧Exploration，下一帧Combat，再下一帧又回到Exploration

**本质**: 
- Router是**无状态的（Stateless）**
- 只看当前帧状态，没有意图惯性

**数学后果**: 
- 时间平均上表现为对多个专家的平均使用
- Alpha熵降不下来

### 问题2: 缺乏惩罚冗余的竞争机制

**现象**:
- Survival专家会"移动"，Exploration专家也会"移动"
- 多个专家功能重叠

**本质**:
- 没有惩罚"多头下注"
- Router同时激活多个功能相似的专家是最安全策略

**数学后果**:
- PPO的Advantage信号说"选A也行，选B也行"
- Router收敛到α = (0.5, 0.5, 0, 0)

### 问题3: 熵正则化的反噬

**现象**:
- Fine-tune阶段仍在最大化Alpha熵
- `-alpha_entropy_coef * alpha_entropy`在loss中

**本质**:
- Warmup阶段的良药（防塌缩）
- Fine-tune阶段的毒药（阻止专业化）

**数学后果**:
- Router想要专一，但loss函数逼着它"不要把话说太死"

---

## 解决方案：三个核心机制

### 机制1: 熵最小化（Entropy Minimization）

**目标**: 在Fine-tune阶段逼迫Router做决定

**数学形式**:
```
旧: L_entropy = -alpha_entropy_coef * H(α)  # 最大化熵
新: L_entropy = +alpha_entropy_coef * H(α)  # 最小化熵
```

**物理含义**:
- 逼迫α向量向One-hot分布坍缩
- Router必须选边站，不能骑墙

**实现策略**:
- Warmup (0-1000): 最大化熵（防塌缩）
- Transition (1000-3000): 逐渐减小熵系数
- Fine-tune (3000+): 最小化熵（强制专业化）

### 机制2: 时间一致性损失（Temporal Consistency Loss）

**目标**: 引入伪记忆，减少意图震荡

**数学形式**:
```
L_temporal = KL(α_t || α_{t-1})
或简化为: L_temporal = MSE(α_t, α_{t-1})
```

**物理含义**:
- 惩罚Router在相邻时间步做出剧烈改变
- 如果环境没有剧变，意图就不该变
- 给无状态Router强加"惯性约束"

**实现细节**:
- 需要在Buffer中存储`last_alpha`
- 只在同一episode内计算（episode开始时不计算）
- 系数建议: 0.01-0.05

### 机制3: 专家重叠惩罚（Expert Overlap Penalty）

**目标**: 真正的竞争，禁止功能重叠

**数学形式**:
```
L_overlap = Σ_{i≠j} α_i * α_j * CosSim(expert_i, expert_j)
```

**物理含义**:
- 如果专家i和j同时被激活（α_i, α_j都大）
- 并且它们的输出很像（CosSim高）
- 那就重罚

**逻辑**:
- 逼迫Router: 要么只激活一个专家
- 要么激活两个输出完全不同的专家（正交）
- 禁止"两个专家做同样的事"

---

## 实现计划

### Phase 1: 熵最小化（优先级P0）

**文件**: `ablation_v3/train/train_v3_gat_moe.py`

**修改点1**: 修改`get_training_config()`

```python
def get_training_config(episode: int) -> Dict:
    if episode < 1000:
        # Warmup: 最大化熵（防塌缩）
        return {
            'alpha_entropy_coef': 0.1,  # 正数，在loss中减去
            'alpha_entropy_sign': -1,   # 新增：符号控制
            ...
        }
    elif episode < 3000:
        # Transition: 逐渐减小熵正则
        progress = (episode - 1000) / 2000
        coef = 0.1 * (1 - progress)  # 0.1 → 0
        return {
            'alpha_entropy_coef': coef,
            'alpha_entropy_sign': -1,
            ...
        }
    else:
        # Fine-tune: 最小化熵（强制专业化）
        return {
            'alpha_entropy_coef': 0.05,  # 正数
            'alpha_entropy_sign': +1,    # 新增：反转符号！
            ...
        }
```

**修改点2**: 修改loss计算

```python
# 计算α熵
alpha_entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean()

# 总损失
total_loss = (
    actor_loss +
    0.5 * critic_loss -
    config['entropy_coef'] * entropy +
    config['alpha_entropy_sign'] * config['alpha_entropy_coef'] * alpha_entropy +  # 修改！
    ...
)
```

### Phase 2: 时间一致性损失（优先级P1）

**文件**: `ablation_v3/train/train_v3_gat_moe.py`

**修改点1**: 在episode循环中追踪last_alpha

```python
# Episode循环开始
last_alpha = None
episode_temporal_losses = []

while not (done or truncated) and steps < args.max_steps:
    # ... 前向传播 ...
    
    # 记录当前alpha
    current_alpha = alpha.detach()
    
    # ... 存储经验 ...
    
    # 更新网络
    if len(trainer.buffer) >= trainer.batch_size:
        # ... PPO更新 ...
        
        # 计算时间一致性损失
        temporal_loss = torch.tensor(0.0, device=device)
        if last_alpha is not None:
            # KL散度或MSE
            temporal_loss = F.mse_loss(alpha, last_alpha)
            episode_temporal_losses.append(temporal_loss.item())
        
        # 总损失
        total_loss = (
            ... +
            config.get('temporal_coef', 0.02) * temporal_loss  # 新增！
        )
        
        # 更新last_alpha
        last_alpha = alpha.detach()
```

**修改点2**: 添加配置参数

```python
def get_training_config(episode: int) -> Dict:
    if episode < 1000:
        return {
            ...
            'temporal_coef': 0.0,  # Warmup: 不使用
        }
    elif episode < 3000:
        return {
            ...
            'temporal_coef': 0.01,  # Transition: 轻微约束
        }
    else:
        return {
            ...
            'temporal_coef': 0.02,  # Fine-tune: 强约束
        }
```

### Phase 3: 专家重叠惩罚（优先级P2）

**文件**: `ablation_v3/train/train_v3_gat_moe.py`

**新增函数**:

```python
def expert_overlap_penalty(
    alpha: torch.Tensor,
    expert_logits: torch.Tensor
) -> torch.Tensor:
    """
    专家重叠惩罚
    
    惩罚同时激活多个功能相似的专家
    
    Args:
        alpha: (batch, num_experts) 专家权重
        expert_logits: (batch, num_experts, action_dim) 专家输出
    
    Returns:
        loss: 标量损失
    """
    batch_size, num_experts, action_dim = expert_logits.shape
    
    if num_experts < 2:
        return torch.tensor(0.0, device=alpha.device)
    
    # L2归一化
    expert_norm = F.normalize(expert_logits, p=2, dim=2)
    
    # 计算余弦相似度矩阵: (batch, num_experts, num_experts)
    similarity = torch.bmm(expert_norm, expert_norm.transpose(1, 2))
    
    # 计算权重乘积矩阵: (batch, num_experts, num_experts)
    alpha_product = torch.bmm(
        alpha.unsqueeze(2),  # (batch, num_experts, 1)
        alpha.unsqueeze(1)   # (batch, 1, num_experts)
    )  # (batch, num_experts, num_experts)
    
    # 只惩罚非对角线元素（不同专家之间）
    mask = 1 - torch.eye(num_experts, device=similarity.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # 重叠惩罚 = 权重乘积 * 相似度
    overlap = (alpha_product * similarity * mask).sum(dim=(1, 2)).mean()
    
    return overlap
```

**修改loss计算**:

```python
# 计算专家重叠惩罚
overlap_loss = torch.tensor(0.0, device=device)
if aux_info['expert_logits'] is not None:
    overlap_loss = expert_overlap_penalty(alpha, aux_info['expert_logits'])

# 总损失
total_loss = (
    ... +
    config.get('overlap_coef', 0.05) * overlap_loss  # 新增！
)
```

---

## 训练配置总结

### 完整的三阶段配置

```python
def get_training_config(episode: int) -> Dict:
    if episode < 1000:
        # Warmup: 探索为主
        return {
            'phase': 'warmup',
            'use_sparsemax': False,
            'learning_rate': 1e-4,
            'entropy_coef': 0.05,
            'alpha_entropy_coef': 0.1,
            'alpha_entropy_sign': -1,      # 最大化熵
            'load_balance_coef': 0.02,
            'diversity_coef': 0.01,
            # Manager约束
            'alignment_coef': 0.1,
            'semantic_coef': 0.05,
            # 高级机制
            'temporal_coef': 0.0,          # 不使用
            'overlap_coef': 0.0,           # 不使用
        }
    elif episode < 3000:
        # Transition: 平滑过渡
        progress = (episode - 1000) / 2000
        alpha_entropy_coef = 0.1 * (1 - progress)  # 0.1 → 0
        
        return {
            'phase': 'transition',
            'use_sparsemax': True,
            'learning_rate': 5e-5,
            'entropy_coef': 0.02,
            'alpha_entropy_coef': alpha_entropy_coef,
            'alpha_entropy_sign': -1,      # 仍然最大化
            'load_balance_coef': 0.01,
            'diversity_coef': 0.01,
            # Manager约束
            'alignment_coef': 0.1,
            'semantic_coef': 0.05,
            # 高级机制
            'temporal_coef': 0.01,         # 开始使用
            'overlap_coef': 0.03,          # 开始使用
        }
    else:
        # Fine-tune: 专业化为主
        return {
            'phase': 'fine-tune',
            'use_sparsemax': True,
            'learning_rate': 1e-5,
            'entropy_coef': 0.01,
            'alpha_entropy_coef': 0.05,
            'alpha_entropy_sign': +1,      # 最小化熵！
            'load_balance_coef': 0.005,
            'diversity_coef': 0.005,
            # Manager约束
            'alignment_coef': 0.1,
            'semantic_coef': 0.05,
            # 高级机制
            'temporal_coef': 0.02,         # 强约束
            'overlap_coef': 0.05,          # 强约束
        }
```

---

## 预期效果

### 定量指标

| 指标 | 当前（仅Manager约束） | 预期（全部机制） | 改进 |
|------|---------------------|----------------|------|
| Alpha熵（终态） | 0.5-0.6 | 0.2-0.3 | -40% to -50% |
| 专家专业化 | 中度 | 极高 | 质变 |
| 意图稳定性 | 中等 | 高 | +50% |
| 平均分数 | 15-18 | 20-25 | +25% to +40% |

### 定性改进

1. **极致专业化**:
   - Alpha熵降到0.2-0.3（接近One-hot）
   - 每个时刻只有1个专家主导

2. **意图连贯性**:
   - 专家切换频率大幅降低
   - 行为更像人类玩家（有计划性）

3. **功能正交性**:
   - 专家之间功能完全不重叠
   - 每个专家有独特的行为模式

---

## 实施顺序

### 阶段1: 熵最小化（1-2小时）

**优先级**: P0（最重要）

**原因**: 
- 实现最简单
- 效果最直接
- 是其他机制的基础

**验证**: 
- 运行500 episodes
- 观察Fine-tune阶段Alpha熵是否快速下降

### 阶段2: 时间一致性（2-3小时）

**优先级**: P1（重要）

**原因**:
- 解决意图震荡问题
- 提升行为连贯性

**验证**:
- 可视化专家切换频率
- 应该看到切换次数减少50%+

### 阶段3: 专家重叠惩罚（2-3小时）

**优先级**: P2（有益）

**原因**:
- 进一步强化专家正交性
- 锦上添花

**验证**:
- 计算专家输出的余弦相似度
- 应该看到相似度降低

---

## 风险和缓解

### 风险1: 过度专业化导致性能下降

**现象**: Alpha熵降到0.1以下，但分数反而下降

**原因**: Router过早坍缩到次优专家

**缓解**: 
- 调低`alpha_entropy_sign`的系数
- 延后熵最小化的启动时间（3000 → 4000）

### 风险2: 时间一致性过强导致僵化

**现象**: Agent陷入单一行为模式，无法应对环境变化

**原因**: `temporal_coef`太大

**缓解**:
- 降低系数（0.02 → 0.01）
- 只在环境变化不大时计算（检测状态差异）

### 风险3: 训练不稳定

**现象**: Loss震荡，梯度爆炸

**原因**: 多个约束项相互冲突

**缓解**:
- 逐个添加机制，不要一次全加
- 仔细调整各项系数的平衡

---

## 总结

这三个机制是对Manager约束的**补充和强化**，而不是替代：

1. **Manager约束**: 告诉Router"应该选哪个专家"（密集监督）
2. **熵最小化**: 逼迫Router"必须做决定"（强制专业化）
3. **时间一致性**: 让Router"保持意图稳定"（引入记忆）
4. **重叠惩罚**: 让专家"功能正交"（真正竞争）

**它们共同构成了一个完整的专家专业化体系。**

---

**文档创建时间**: 2026-01-12 00:15  
**状态**: 待实现  
**预计实现时间**: 6-8小时
