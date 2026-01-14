# V3 训练稳定性检查清单

> **版本**: V3.0 - GAT-Guided Hierarchical MoE  
> **创建日期**: 2025-01-05  
> **目的**: 确保V3训练稳定、收敛、无塌缩  
> **状态**: 设计阶段 → 实现前必读

---

## 一、概述

V3引入了GAT和Sparsemax路由，相比V1/V2有更多潜在的训练不稳定因素：
- **GAT过平滑**: 多层GAT可能导致节点特征趋同
- **专家塌缩**: 路由器可能只选择1-2个专家，其他专家退化
- **梯度爆炸**: GAT + MoE的深层网络容易梯度不稳定
- **奖励稀疏**: NetHack的长程稀疏奖励问题
- **数值不稳定**: NaN/Inf导致训练崩溃

本文档提供**完整的稳定性措施清单**，涵盖：
1. 网络架构设计
2. 训练超参数
3. 辅助损失和正则化
4. 数值稳定性技巧
5. 监控和诊断
6. 降级方案

---

## 二、网络架构稳定性措施

### 2.1 GAT层设计

#### ✅ 已实现
- **2层GAT**: 限制层数，避免过平滑
- **残差连接**: `x2 = x1 + GAT(x1)` 保持梯度流
- **LayerNorm**: 每层后归一化，稳定激活值
- **多头注意力**: 4个头，增加表达能力
- **Dropout**: 0.1，防止过拟合

#### 🔧 待实现
- **注意力温度**: 添加可学习温度参数，防止注意力过于尖锐
- **边Dropout**: 训练时随机丢弃部分边，增强鲁棒性
- **节点特征裁剪**: 限制节点嵌入的L2范数，防止爆炸


```python
# 示例: 添加注意力温度
class GATConvWithTemp(GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = nn.Parameter(torch.ones(1))  # 可学习温度
    
    def forward(self, x, edge_index):
        alpha = self.attention(x, edge_index) / self.temp  # 温度缩放
        return super().forward(x, edge_index, alpha=alpha)
```

### 2.2 路由器设计

#### ✅ 已实现
- **Sparsemax激活**: 自动稀疏化，避免平均主义
- **3层MLP**: 512→128→64→4，逐步降维
- **LayerNorm**: 每层后归一化
- **小增益初始化**: 最后一层权重用0.01增益

#### 🔧 待实现
- **温度退火**: Sparsemax温度从1.0逐渐降到0.5
- **Warmup阶段**: 前1000 episodes使用Softmax，避免过早塌缩
- **负载均衡损失**: 鼓励专家使用均衡

```python
# 示例: 温度退火
def get_sparsemax_temp(episode, warmup=1000, max_episodes=10000):
    if episode < warmup:
        return 1.0  # Warmup阶段: Softmax
    else:
        # 线性退火: 1.0 → 0.5
        progress = (episode - warmup) / (max_episodes - warmup)
        return 1.0 - 0.5 * progress
```


### 2.3 专家网络设计

#### ✅ 已实现
- **独立MLP**: 每个专家是独立的2层MLP
- **LayerNorm**: 稳定激活值
- **小增益初始化**: 输出层0.01增益，防止初始logits过大

#### 🔧 待实现
- **专家正则化**: L2正则化，防止过拟合
- **专家多样性损失**: 鼓励专家学到不同策略

```python
# 示例: 专家多样性损失
def expert_diversity_loss(expert_logits):
    """
    鼓励专家输出不同的动作分布
    
    Args:
        expert_logits: (batch, num_experts, action_dim)
    
    Returns:
        diversity_loss: 标量
    """
    # 计算专家间的余弦相似度
    num_experts = expert_logits.size(1)
    diversity_loss = 0.0
    
    for i in range(num_experts):
        for j in range(i+1, num_experts):
            cos_sim = F.cosine_similarity(
                expert_logits[:, i, :], 
                expert_logits[:, j, :], 
                dim=-1
            ).mean()
            diversity_loss += cos_sim
    
    # 归一化
    diversity_loss /= (num_experts * (num_experts - 1) / 2)
    
    return diversity_loss
```


### 2.4 Critic网络设计

#### ✅ 已实现
- **双流输入**: h_vis + h_logic (512维)
- **3层MLP**: 512→256→128→1
- **LayerNorm**: 稳定训练

#### 🔧 待实现
- **价值裁剪**: 限制价值估计范围，防止爆炸
- **双Critic**: 使用两个Critic取最小值，减少过估计

```python
# 示例: 价值裁剪
def forward_critic(self, z):
    value = self.critic(z)
    value = torch.clamp(value, -1000, 1000)  # 裁剪到合理范围
    return value
```

---

## 三、训练超参数设置

### 3.1 学习率

| 参数 | V1/V2 | V3 (推荐) | 理由 |
|------|-------|-----------|------|
| **learning_rate** | 3e-4 | **1e-4** | GAT需要更小学习率 |
| **lr_scheduler** | 无 | **CosineAnnealing** | 后期精细调整 |
| **warmup_steps** | 0 | **1000** | 避免初期梯度爆炸 |

```python
# 示例: 学习率Warmup + CosineAnnealing
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

def get_lr_scheduler(optimizer, warmup_steps=1000, max_steps=100000):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # 线性warmup
        else:
            return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_steps-warmup_steps)
    
    return warmup_scheduler, cosine_scheduler
```


### 3.2 PPO超参数

| 参数 | V1/V2 | V3 (推荐) | 理由 |
|------|-------|-----------|------|
| **clip_ratio** | 0.2 | **0.15** | 更保守的策略更新 |
| **ppo_epochs** | 3 | **4** | 更充分的策略优化 |
| **batch_size** | 128 | **256** | 更稳定的梯度估计 |
| **gamma** | 0.99 | **0.995** | NetHack需要更长视野 |
| **gae_lambda** | 0.95 | **0.97** | 更平滑的优势估计 |

### 3.3 正则化系数

| 参数 | V1/V2 | V3 (推荐) | 理由 |
|------|-------|-----------|------|
| **entropy_coef** | 0.05 | **0.01** | Sparsemax已稀疏，降低熵正则 |
| **alpha_entropy_coef** | 0.1 | **0.05** | 鼓励专家均衡使用 |
| **diversity_coef** | 0 | **0.01** | 鼓励专家多样性 |
| **load_balance_coef** | 0 | **0.01** | 防止专家塌缩 |

---

## 四、辅助损失和正则化

### 4.1 专家负载均衡损失

**目的**: 防止路由器只选择1-2个专家，其他专家退化

```python
def load_balance_loss(alpha, num_experts=4):
    """
    负载均衡损失: 鼓励每个专家被均匀使用
    
    Args:
        alpha: (batch, num_experts) 专家权重
    
    Returns:
        loss: 标量
    """
    # 计算每个专家的平均使用率
    expert_usage = alpha.mean(dim=0)  # (num_experts,)
    
    # 理想情况: 每个专家使用率 = 1/num_experts
    target_usage = torch.ones_like(expert_usage) / num_experts
    
    # L2损失
    loss = F.mse_loss(expert_usage, target_usage)
    
    return loss
```


### 4.2 专家多样性损失

**目的**: 鼓励不同专家学到不同的策略

```python
def expert_diversity_loss(expert_logits):
    """
    专家多样性损失: 最小化专家间的相似度
    
    Args:
        expert_logits: (batch, num_experts, action_dim)
    
    Returns:
        loss: 标量
    """
    num_experts = expert_logits.size(1)
    
    # 计算专家输出的协方差矩阵
    # 展平: (batch*num_experts, action_dim)
    flat_logits = expert_logits.view(-1, expert_logits.size(-1))
    
    # 中心化
    mean_logits = flat_logits.mean(dim=0, keepdim=True)
    centered = flat_logits - mean_logits
    
    # 协方差矩阵
    cov = torch.mm(centered.t(), centered) / centered.size(0)
    
    # 对角线外的元素 (专家间相似度)
    off_diag = cov - torch.diag(torch.diag(cov))
    
    # 最小化相似度
    loss = off_diag.abs().mean()
    
    return loss
```

### 4.3 GAT注意力正则化

**目的**: 防止GAT注意力过于集中或过于分散

```python
def attention_regularization(attention_weights, target_sparsity=0.3):
    """
    注意力正则化: 鼓励适度稀疏的注意力
    
    Args:
        attention_weights: (num_edges,) GAT注意力权重
        target_sparsity: 目标稀疏度 (0-1)
    
    Returns:
        loss: 标量
    """
    # 计算实际稀疏度 (Gini系数)
    sorted_weights, _ = torch.sort(attention_weights)
    n = len(sorted_weights)
    index = torch.arange(1, n+1, device=sorted_weights.device)
    gini = (2 * (index * sorted_weights).sum()) / (n * sorted_weights.sum()) - (n+1)/n
    
    # L2损失
    loss = (gini - target_sparsity) ** 2
    
    return loss
```


### 4.4 Next-Intent Prediction (辅助任务)

**目的**: 强迫GAT学习因果关系，预测下一步哪个Operator会激活

```python
def next_intent_prediction_loss(h_logic, next_operator_mask):
    """
    下一意图预测损失: 预测下一步激活的Operator
    
    Args:
        h_logic: (batch, hidden_dim) Intent Vector
        next_operator_mask: (batch, num_operators) 下一步激活的Operator (0/1)
    
    Returns:
        loss: 标量
    """
    # 预测头
    predictor = nn.Linear(hidden_dim, num_operators)
    
    # 预测logits
    pred_logits = predictor(h_logic)  # (batch, num_operators)
    
    # 二分类交叉熵
    loss = F.binary_cross_entropy_with_logits(
        pred_logits, 
        next_operator_mask.float()
    )
    
    return loss
```

**使用方法**:
1. 在训练循环中，记录当前和下一步的atoms
2. 从atoms构造next_operator_mask
3. 将此损失加入总损失: `total_loss += 0.1 * next_intent_loss`

---

## 五、梯度和数值稳定性

### 5.1 梯度裁剪

#### ✅ 已实现 (PPO Trainer)
```python
nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
```

#### 🔧 推荐调整
- **V3推荐**: `max_norm=1.0` (更保守)
- **监控**: 记录梯度范数，检测爆炸

```python
# 示例: 监控梯度范数
def log_gradient_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    print(f"Gradient norm: {total_norm:.4f}")
    
    if total_norm > 10.0:
        print(f"⚠️ 梯度爆炸警告: {total_norm:.4f}")
```


### 5.2 NaN/Inf处理

#### ✅ 已实现
```python
# 在 networks_v3_gat_moe.py
logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
value = torch.nan_to_num(value, nan=0.0, posinf=1e3, neginf=-1e3)
```

#### 🔧 额外措施
- **检查点**: 每次前向传播后检查NaN
- **自动回滚**: 检测到NaN时回滚到上一个checkpoint

```python
# 示例: NaN检测和回滚
class NaNDetector:
    def __init__(self, model):
        self.model = model
        self.last_good_state = None
    
    def save_checkpoint(self):
        self.last_good_state = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }
    
    def check_and_rollback(self, loss):
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ NaN/Inf detected! Rolling back...")
            if self.last_good_state:
                self.model.load_state_dict(self.last_good_state)
            return True
        return False
```

### 5.3 权重初始化

#### ✅ 已实现
- **专家输出层**: Orthogonal初始化，gain=0.01
- **嵌入层**: 默认初始化

#### 🔧 推荐
- **GAT层**: Xavier初始化
- **路由器**: He初始化

```python
# 示例: 自定义初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, GATConv):
        nn.init.xavier_uniform_(m.lin.weight)

policy_net.apply(init_weights)
```


---

## 六、奖励塑形和稀疏奖励处理

### 6.1 奖励归一化

**问题**: NetHack奖励范围大 (-1000到+1000)，导致价值估计不稳定

```python
class RewardNormalizer:
    def __init__(self, clip_range=10.0):
        self.mean = 0.0
        self.std = 1.0
        self.clip_range = clip_range
        self.count = 0
    
    def update(self, reward):
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.std = np.sqrt((self.std**2 * (self.count-1) + delta**2) / self.count)
    
    def normalize(self, reward):
        normalized = (reward - self.mean) / (self.std + 1e-8)
        return np.clip(normalized, -self.clip_range, self.clip_range)
```

### 6.2 奖励塑形

**V2已有**: 5分量奖励 (progress, safety, efficiency, feasibility, exploration)

**V3增强**: 添加GAT相关奖励

```python
def compute_v3_reward(env_reward, gat_info, expert_info):
    """
    V3奖励塑形
    
    Args:
        env_reward: 环境原始奖励
        gat_info: GAT相关信息 (operator激活率等)
        expert_info: 专家相关信息 (α权重等)
    
    Returns:
        shaped_reward: 塑形后的奖励
    """
    # 基础奖励
    r_base = env_reward / 1000.0
    
    # GAT奖励: 鼓励激活有意义的Operator
    r_gat = 0.01 * gat_info['operator_activation_rate']
    
    # 专家奖励: 鼓励明确的专家选择 (高α熵惩罚)
    alpha_entropy = -(expert_info['alpha'] * np.log(expert_info['alpha'] + 1e-8)).sum()
    r_expert = -0.01 * alpha_entropy  # 熵越低越好
    
    return r_base + r_gat + r_expert
```


### 6.3 奖励裁剪

**目的**: 防止极端奖励破坏训练

```python
def clip_reward(reward, clip_range=10.0):
    return np.clip(reward, -clip_range, clip_range)
```

---

## 七、训练流程和Warmup机制

### 7.1 三阶段训练

| 阶段 | Episodes | 特点 | 目的 |
|------|----------|------|------|
| **Warmup** | 0-1000 | Softmax路由, 高学习率 | 让专家学到基础策略 |
| **Transition** | 1000-3000 | 温度退火, 逐渐稀疏化 | 平滑过渡到Sparsemax |
| **Fine-tune** | 3000+ | Sparsemax路由, 低学习率 | 精细调整专家分工 |

```python
def get_training_config(episode):
    """根据训练阶段返回配置"""
    if episode < 1000:
        # Warmup阶段
        return {
            'use_sparsemax': False,  # 使用Softmax
            'learning_rate': 1e-4,
            'entropy_coef': 0.05,
            'load_balance_coef': 0.02,  # 强制均衡
        }
    elif episode < 3000:
        # Transition阶段
        temp = 1.0 - 0.5 * (episode - 1000) / 2000  # 1.0 → 0.5
        return {
            'use_sparsemax': True,
            'sparsemax_temp': temp,
            'learning_rate': 5e-5,
            'entropy_coef': 0.02,
            'load_balance_coef': 0.01,
        }
    else:
        # Fine-tune阶段
        return {
            'use_sparsemax': True,
            'sparsemax_temp': 0.5,
            'learning_rate': 1e-5,
            'entropy_coef': 0.01,
            'load_balance_coef': 0.005,
        }
```


### 7.2 Checkpoint和恢复

**策略**:
1. 每100 episodes保存checkpoint
2. 保存最佳模型 (best_reward, best_score)
3. 检测到NaN时自动回滚

```python
def save_checkpoint(episode, policy_net, optimizer, stats, path):
    torch.save({
        'episode': episode,
        'policy_net': policy_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'stats': stats,
        'timestamp': time.time(),
    }, path)

def load_checkpoint(path, policy_net, optimizer):
    checkpoint = torch.load(path)
    policy_net.load_state_dict(checkpoint['policy_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['episode'], checkpoint['stats']
```

---

## 八、监控和诊断指标

### 8.1 必须监控的指标

| 指标 | 正常范围 | 异常信号 | 处理方法 |
|------|----------|----------|----------|
| **episode_score** | 逐渐上升 | 长期不变或下降 | 检查奖励塑形 |
| **alpha_entropy** | 0.5-1.0 | <0.3 (塌缩) 或 >1.2 (混乱) | 调整load_balance_coef |
| **expert_usage** | 每个专家10-40% | 某个专家>80% | 增加load_balance_coef |
| **gat_attention_variance** | >0.1 | <0.05 (过平滑) | 减少GAT层数或增加dropout |
| **operator_activation_rate** | 10-30% | <5% 或 >50% | 检查atoms提取逻辑 |
| **gradient_norm** | <5.0 | >10.0 | 降低学习率或增加裁剪 |
| **actor_loss** | 逐渐下降 | 震荡或爆炸 | 降低clip_ratio |
| **critic_loss** | 逐渐下降 | 不收敛 | 增加batch_size |


### 8.2 实时监控代码

```python
class TrainingMonitor:
    def __init__(self, log_interval=50):
        self.log_interval = log_interval
        self.metrics = defaultdict(list)
    
    def log(self, episode, metrics):
        """记录指标"""
        for k, v in metrics.items():
            self.metrics[k].append(v)
        
        if episode % self.log_interval == 0:
            self.print_summary(episode)
            self.check_anomalies(episode)
    
    def print_summary(self, episode):
        """打印摘要"""
        print(f"\n=== Episode {episode} Summary ===")
        for k, v in self.metrics.items():
            recent = v[-self.log_interval:]
            print(f"  {k}: {np.mean(recent):.4f} ± {np.std(recent):.4f}")
    
    def check_anomalies(self, episode):
        """检查异常"""
        # 检查专家塌缩
        if 'alpha_entropy' in self.metrics:
            recent_entropy = np.mean(self.metrics['alpha_entropy'][-50:])
            if recent_entropy < 0.3:
                print(f"⚠️ 专家塌缩警告: α熵={recent_entropy:.4f}")
        
        # 检查梯度爆炸
        if 'gradient_norm' in self.metrics:
            recent_grad = np.mean(self.metrics['gradient_norm'][-10:])
            if recent_grad > 10.0:
                print(f"⚠️ 梯度爆炸警告: 梯度范数={recent_grad:.4f}")
        
        # 检查GAT过平滑
        if 'gat_attention_variance' in self.metrics:
            recent_var = np.mean(self.metrics['gat_attention_variance'][-50:])
            if recent_var < 0.05:
                print(f"⚠️ GAT过平滑警告: 注意力方差={recent_var:.4f}")
```


### 8.3 可视化监控

**推荐工具**: TensorBoard 或 Weights & Biases

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=f"runs/{exp_name}")

# 记录标量
writer.add_scalar('Train/episode_score', score, episode)
writer.add_scalar('Train/alpha_entropy', alpha_entropy, episode)
writer.add_scalar('Train/gradient_norm', grad_norm, episode)

# 记录直方图
writer.add_histogram('Train/alpha_distribution', alpha, episode)
writer.add_histogram('Train/operator_scores', operator_scores, episode)

# 记录图像 (GAT注意力热图)
writer.add_image('GAT/attention_heatmap', attention_heatmap, episode)
```

---

## 九、常见问题和解决方案

### 9.1 专家塌缩

**症状**:
- α熵 < 0.3
- 某个专家权重 > 0.8
- 其他专家梯度接近0

**原因**:
- Sparsemax过早收敛
- 负载均衡损失不足
- 专家初始化不当

**解决方案**:
1. 增加Warmup阶段长度 (1000 → 2000 episodes)
2. 增加load_balance_coef (0.01 → 0.05)
3. 添加专家多样性损失
4. 使用更大的Sparsemax温度


### 9.2 GAT过平滑

**症状**:
- GAT注意力方差 < 0.05
- 所有节点嵌入趋同
- Intent Vector变化很小

**原因**:
- GAT层数过多
- 学习率过大
- 缺乏正则化

**解决方案**:
1. 减少GAT层数 (2 → 1)
2. 增加Dropout (0.1 → 0.2)
3. 添加边Dropout
4. 使用残差连接 (已实现)

### 9.3 梯度爆炸

**症状**:
- 梯度范数 > 10.0
- 损失突然变为NaN
- 参数更新过大

**原因**:
- 学习率过大
- 梯度裁剪不足
- 网络初始化不当

**解决方案**:
1. 降低学习率 (1e-4 → 5e-5)
2. 增加梯度裁剪 (0.5 → 1.0)
3. 使用Xavier初始化
4. 添加LayerNorm (已实现)

### 9.4 奖励不收敛

**症状**:
- episode_score长期不变
- 策略震荡
- α权重混乱

**原因**:
- 奖励塑形不当
- 探索不足
- 价值估计偏差

**解决方案**:
1. 调整奖励权重
2. 增加熵正则化
3. 使用双Critic
4. 增加batch_size


---

## 十、降级方案

如果V3训练失败，按以下顺序尝试降级方案：

### 方案1: 固定GAT (不训练)

```python
# 冻结GAT参数
for param in policy_net.gat.parameters():
    param.requires_grad = False

# 只训练路由器和专家
optimizer = optim.Adam([
    {'params': policy_net.router.parameters()},
    {'params': policy_net.experts.parameters()},
    {'params': policy_net.critic.parameters()},
], lr=1e-4)
```

### 方案2: 使用Softmax路由

```python
# 禁用Sparsemax，使用Softmax
policy_net = GATGuidedMoEPolicy(
    use_sparsemax=False  # 使用Softmax
)
```

### 方案3: 减少专家数量

```python
# 从4个专家减少到2个
policy_net = GATGuidedMoEPolicy(
    num_experts=2  # Survival + General
)
```

### 方案4: 回退到V2 + GAT特征

```python
# 使用GAT提取特征，但用V2的路由方式
h_logic, _, _ = gat(atoms=atoms)
state_with_gat = np.concatenate([state, h_logic.cpu().numpy()])

# 使用V2网络
policy_net_v2 = MultiChannelPolicyNet(
    state_dim=115 + 256,  # 原始state + GAT特征
    use_gumbel=True
)
```


---

## 十一、完整训练脚本模板

```python
def train_v3(args):
    """V3训练主循环 - 包含所有稳定性措施"""
    
    # 1. 初始化
    device = get_device()
    policy_net = GATGuidedMoEPolicy(...).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    lr_scheduler = get_lr_scheduler(optimizer)
    
    # 2. 监控器
    monitor = TrainingMonitor(log_interval=50)
    nan_detector = NaNDetector(policy_net)
    reward_normalizer = RewardNormalizer()
    
    # 3. 训练循环
    for episode in range(args.episodes):
        # 3.1 获取当前阶段配置
        config = get_training_config(episode)
        policy_net.use_sparsemax = config['use_sparsemax']
        
        # 3.2 Episode循环
        obs, info = env.reset()
        done = False
        episode_metrics = defaultdict(list)
        
        while not done:
            # 选择动作
            state, atoms = extract_state_and_atoms(obs)
            logits, alpha, value, aux_info = policy_net(state, atoms)
            
            # 记录指标
            episode_metrics['alpha_entropy'].append(
                -(alpha * torch.log(alpha + 1e-8)).sum().item()
            )
            episode_metrics['operator_activation_rate'].append(
                (aux_info['operator_scores'] > 0.5).float().mean().item()
            )
            
            # 执行动作
            action = Categorical(logits=logits).sample()
            obs, reward, done, truncated, info = env.step(action.item())
            
            # 奖励塑形
            shaped_reward = compute_v3_reward(reward, aux_info, {'alpha': alpha})
            normalized_reward = reward_normalizer.normalize(shaped_reward)
            
            # 存储经验
            trainer.buffer.add(state, action, normalized_reward, ...)
        
        # 3.3 更新网络
        if len(trainer.buffer) >= trainer.batch_size:
            # 保存checkpoint (用于NaN回滚)
            nan_detector.save_checkpoint()
            
            # 计算损失
            actor_loss, critic_loss = trainer.compute_losses()
            
            # 辅助损失
            lb_loss = load_balance_loss(alpha_history)
            div_loss = expert_diversity_loss(expert_logits_history)
            
            total_loss = (
                actor_loss + 
                0.5 * critic_loss + 
                config['load_balance_coef'] * lb_loss +
                0.01 * div_loss
            )
            
            # 检查NaN
            if nan_detector.check_and_rollback(total_loss):
                continue
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 记录梯度范数
            grad_norm = log_gradient_norms(policy_net)
            episode_metrics['gradient_norm'].append(grad_norm)
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            
            # 更新
            optimizer.step()
            lr_scheduler.step()
        
        # 3.4 监控
        monitor.log(episode, {
            'episode_score': info['score'],
            'alpha_entropy': np.mean(episode_metrics['alpha_entropy']),
            'operator_activation_rate': np.mean(episode_metrics['operator_activation_rate']),
            'gradient_norm': np.mean(episode_metrics['gradient_norm']),
        })
        
        # 3.5 保存checkpoint
        if episode % 100 == 0:
            save_checkpoint(episode, policy_net, optimizer, ...)
```


---

## 十二、实施检查清单

在开始训练前，确保以下所有项目都已完成：

### 网络架构 ✅/❌

- [ ] GAT使用2层，带残差连接
- [ ] 每层后有LayerNorm
- [ ] 路由器使用Sparsemax (带温度退火)
- [ ] 专家输出层小增益初始化 (0.01)
- [ ] Critic有价值裁剪

### 训练超参数 ✅/❌

- [ ] 学习率设为1e-4 (比V2更小)
- [ ] 使用学习率Warmup (1000 steps)
- [ ] PPO clip_ratio设为0.15
- [ ] batch_size >= 256
- [ ] 梯度裁剪max_norm=1.0

### 辅助损失 ✅/❌

- [ ] 实现load_balance_loss
- [ ] 实现expert_diversity_loss
- [ ] 设置合适的损失权重
- [ ] (可选) 实现next_intent_prediction_loss

### 数值稳定性 ✅/❌

- [ ] 所有logits做nan_to_num和clamp
- [ ] 实现NaN检测和回滚
- [ ] 奖励归一化
- [ ] 权重初始化检查

### 监控和诊断 ✅/❌

- [ ] 实现TrainingMonitor类
- [ ] 记录所有关键指标
- [ ] 设置异常检测阈值
- [ ] (可选) 集成TensorBoard

### Warmup机制 ✅/❌

- [ ] 前1000 episodes使用Softmax
- [ ] 1000-3000 episodes温度退火
- [ ] 3000+ episodes使用Sparsemax (temp=0.5)
- [ ] 学习率随阶段调整

### Checkpoint和恢复 ✅/❌

- [ ] 每100 episodes保存checkpoint
- [ ] 保存最佳模型
- [ ] 实现checkpoint加载逻辑
- [ ] NaN时自动回滚


---

## 十三、预期训练曲线

### 正常训练曲线

```
Episode Score:
  0-1000:    50-100   (Warmup, 探索)
  1000-3000: 100-300  (Transition, 专家分工形成)
  3000-5000: 300-600  (Fine-tune, 稳定提升)
  5000+:     600-800+ (收敛)

Alpha Entropy:
  0-1000:    1.2-1.4  (Softmax, 高熵)
  1000-3000: 1.0-0.6  (退火, 逐渐稀疏)
  3000+:     0.5-0.8  (Sparsemax, 稳定)

Expert Usage (理想):
  Survival:    20-30%
  Combat:      25-35%
  Exploration: 25-35%
  General:     10-20%

Gradient Norm:
  全程:       1.0-3.0  (稳定)
  异常:       >10.0    (需要干预)
```

### 异常训练曲线

```
专家塌缩:
  Alpha Entropy < 0.3
  某个专家 > 80%
  → 增加load_balance_coef

GAT过平滑:
  Attention Variance < 0.05
  Operator Activation Rate < 5%
  → 减少GAT层数或增加Dropout

梯度爆炸:
  Gradient Norm > 10.0
  Loss突然变NaN
  → 降低学习率，增加裁剪

奖励不收敛:
  Score长期不变
  Alpha权重混乱
  → 调整奖励塑形，增加探索
```


---

## 十四、参考文献

### 相关技术

1. **GAT稳定性**:
   - "Graph Attention Networks" (Veličković et al., 2018)
   - "How to Train Your Graph Neural Network" (Dwivedi et al., 2020)

2. **MoE训练**:
   - "Switch Transformers" (Fedus et al., 2021) - 负载均衡
   - "GShard" (Lepikhin et al., 2020) - 专家并行

3. **Sparsemax**:
   - "From Softmax to Sparsemax" (Martins & Astudillo, 2016)

4. **PPO稳定性**:
   - "Implementation Matters in Deep RL" (Engstrom et al., 2020)
   - "What Matters in On-Policy RL" (Andrychowicz et al., 2021)

### V1/V2经验

- V1: Softmax路由稳定但分工不明确
- V2: Gumbel路由容易塌缩，需要强正则化
- V3: Sparsemax是折中方案，需要Warmup

---

## 十五、总结

### 核心原则

1. **渐进式训练**: Warmup → Transition → Fine-tune
2. **多重正则化**: 负载均衡 + 多样性 + 熵正则
3. **严格监控**: 实时检测异常，及时干预
4. **数值稳定**: NaN检测、梯度裁剪、奖励归一化
5. **降级准备**: 多个备选方案，避免全盘失败

### 成功标准

- **短期** (1周): 训练稳定，无NaN，专家无塌缩
- **中期** (2周): Score > 600, α熵 < 1.0
- **长期** (1月): Score > 800, 样本效率 > 1.5x V1

### 下一步

1. 实现训练脚本 `ablation_v3/train/train_v3_gat_moe.py`
2. 集成所有稳定性措施
3. 小规模测试 (100 episodes)
4. 全面训练 (10000 episodes)

---

**文档状态**: ✅ 完成  
**准备度**: ✅ 可以开始实现训练脚本  
**信心**: 🔥🔥🔥🔥🔥 (5/5)

