# V3 稳定性措施实现状态

> **更新时间**: 2025-01-05  
> **目的**: 对比checklist中的措施与当前代码实现

---

## ✅ 已实现的稳定性措施

### 1. 网络架构层

#### GAT层 (src/core/hypergraph_gat.py)
- ✅ **2层GAT**: 限制层数，避免过平滑
- ✅ **残差连接**: `x2 = x1 + GAT(x1)` 保持梯度流
- ✅ **LayerNorm**: 每层后归一化
- ✅ **多头注意力**: 4个头
- ✅ **Dropout**: 0.1

#### 路由器 (src/core/networks_v3_gat_moe.py - CausalRouter)
- ✅ **Sparsemax激活**: 自动稀疏化
- ✅ **3层MLP**: 512→128→64→4
- ✅ **LayerNorm**: 每层后归一化
- ✅ **Warmup支持**: `use_sparsemax`参数可切换Softmax/Sparsemax
- ✅ **数值稳定**: logits做nan_to_num和clamp

#### 专家网络 (src/core/networks_v3_gat_moe.py - SemanticExpert)
- ✅ **独立MLP**: 每个专家2层MLP
- ✅ **LayerNorm**: 稳定激活值
- ✅ **小增益初始化**: 输出层gain=0.01

#### Critic网络 (src/core/networks_v3_gat_moe.py)
- ✅ **双流输入**: h_vis + h_logic (512维)
- ✅ **3层MLP**: 512→256→128→1
- ✅ **LayerNorm**: 稳定训练

### 2. 数值稳定性

#### NaN/Inf处理
- ✅ **路由器logits**: `torch.nan_to_num(...).clamp(-20.0, 20.0)`
- ✅ **融合logits**: `torch.nan_to_num(...).clamp(-20.0, 20.0)`
- ✅ **价值估计**: `torch.nan_to_num(...)`

#### 梯度裁剪 (src/core/ppo_trainer.py)
- ✅ **已实现**: `nn.utils.clip_grad_norm_(parameters, 0.5)`
- ⚠️ **需调整**: V3推荐max_norm=1.0

### 3. 辅助功能

#### 专家使用统计
- ✅ **get_expert_usage_stats**: 分析α权重分布
- ✅ **dominant_counts**: 统计主导专家

#### 动作分布
- ✅ **get_action_distribution**: 用于PPO采样

---

## ❌ 待实现的稳定性措施

### 1. 辅助损失函数 (需要在训练脚本中实现)

#### 负载均衡损失 (防止专家塌缩)
```python
def load_balance_loss(alpha, num_experts=4):
    """鼓励每个专家被均匀使用"""
    expert_usage = alpha.mean(dim=0)
    target_usage = torch.ones_like(expert_usage) / num_experts
    return F.mse_loss(expert_usage, target_usage)
```
**状态**: ❌ 未实现  
**优先级**: 🔥🔥🔥 极高 (防止塌缩的关键)

#### 专家多样性损失 (鼓励差异化)
```python
def expert_diversity_loss(expert_logits):
    """最小化专家间余弦相似度"""
    num_experts = expert_logits.size(1)
    diversity = 0.0
    for i in range(num_experts):
        for j in range(i+1, num_experts):
            cos_sim = F.cosine_similarity(
                expert_logits[:, i, :], 
                expert_logits[:, j, :], 
                dim=-1
            ).mean()
            diversity += cos_sim
    return diversity / (num_experts * (num_experts - 1) / 2)
```
**状态**: ❌ 未实现  
**优先级**: 🔥🔥 高

#### GAT注意力正则化
```python
def attention_regularization(attention_weights, target_sparsity=0.3):
    """鼓励适度稀疏的注意力"""
    # 计算Gini系数
    sorted_weights, _ = torch.sort(attention_weights)
    n = len(sorted_weights)
    index = torch.arange(1, n+1, device=sorted_weights.device)
    gini = (2 * (index * sorted_weights).sum()) / (n * sorted_weights.sum()) - (n+1)/n
    return (gini - target_sparsity) ** 2
```
**状态**: ❌ 未实现  
**优先级**: 🔥 中


### 2. 训练流程机制 (需要在训练脚本中实现)

#### 三阶段训练配置
```python
def get_training_config(episode):
    """根据训练阶段返回配置"""
    if episode < 1000:
        # Warmup阶段
        return {
            'use_sparsemax': False,
            'learning_rate': 1e-4,
            'load_balance_coef': 0.02,
        }
    elif episode < 3000:
        # Transition阶段
        temp = 1.0 - 0.5 * (episode - 1000) / 2000
        return {
            'use_sparsemax': True,
            'sparsemax_temp': temp,
            'learning_rate': 5e-5,
            'load_balance_coef': 0.01,
        }
    else:
        # Fine-tune阶段
        return {
            'use_sparsemax': True,
            'sparsemax_temp': 0.5,
            'learning_rate': 1e-5,
            'load_balance_coef': 0.005,
        }
```
**状态**: ❌ 未实现  
**优先级**: 🔥🔥🔥 极高 (核心训练策略)

#### 学习率Warmup和退火
```python
def get_lr_scheduler(optimizer, warmup_steps=1000, max_steps=100000):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max_steps-warmup_steps)
    
    return warmup_scheduler, cosine_scheduler
```
**状态**: ❌ 未实现  
**优先级**: 🔥🔥 高

#### NaN检测和自动回滚
```python
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
**状态**: ❌ 未实现  
**优先级**: 🔥🔥 高

### 3. 监控和诊断 (需要在训练脚本中实现)

#### TrainingMonitor类
```python
class TrainingMonitor:
    def __init__(self, log_interval=50):
        self.log_interval = log_interval
        self.metrics = defaultdict(list)
    
    def log(self, episode, metrics):
        for k, v in metrics.items():
            self.metrics[k].append(v)
        
        if episode % self.log_interval == 0:
            self.print_summary(episode)
            self.check_anomalies(episode)
    
    def check_anomalies(self, episode):
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
```
**状态**: ❌ 未实现  
**优先级**: 🔥🔥 高

#### 梯度范数监控
```python
def log_gradient_norms(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    if total_norm > 10.0:
        print(f"⚠️ 梯度爆炸警告: {total_norm:.4f}")
    
    return total_norm
```
**状态**: ❌ 未实现  
**优先级**: 🔥🔥 高

### 4. 奖励处理 (需要在训练脚本中实现)

#### 奖励归一化
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
**状态**: ❌ 未实现  
**优先级**: 🔥 中

#### V3奖励塑形
```python
def compute_v3_reward(env_reward, gat_info, expert_info):
    """V3增强的奖励塑形"""
    r_base = env_reward / 1000.0
    
    # GAT奖励: 鼓励激活有意义的Operator
    r_gat = 0.01 * gat_info['operator_activation_rate']
    
    # 专家奖励: 鼓励明确的专家选择
    alpha_entropy = -(expert_info['alpha'] * np.log(expert_info['alpha'] + 1e-8)).sum()
    r_expert = -0.01 * alpha_entropy
    
    return r_base + r_gat + r_expert
```
**状态**: ❌ 未实现  
**优先级**: 🔥 中

---

## 🔧 需要调整的现有实现

### 1. PPO Trainer (src/core/ppo_trainer.py)

#### 梯度裁剪
```python
# 当前: max_norm=0.5
nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)

# V3推荐: max_norm=1.0
nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
```
**状态**: ⚠️ 需调整  
**优先级**: 🔥 中

#### 学习率
```python
# 当前: 3e-4
self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)

# V3推荐: 1e-4
self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
```
**状态**: ⚠️ 需调整  
**优先级**: 🔥🔥 高

#### PPO超参数
```python
# 当前值 → V3推荐值
clip_ratio: 0.2 → 0.15
batch_size: 64 → 256
ppo_epochs: 3 → 4
gamma: 0.99 → 0.995
gae_lambda: 0.95 → 0.97
entropy_coef: 0.05 → 0.01
alpha_entropy_coef: 0.1 → 0.05
```
**状态**: ⚠️ 需调整  
**优先级**: 🔥🔥 高

### 2. Sparsemax实现 (src/core/networks_v3_gat_moe.py)

#### 当前实现
```python
def sparsemax(logits, dim=-1):
    # 简化实现: top-k + softmax
    k = max(2, logits.size(dim) // 2)
    topk_values, topk_indices = torch.topk(logits, k, dim=dim)
    topk_probs = F.softmax(topk_values, dim=dim)
    output = torch.zeros_like(logits)
    output.scatter_(dim, topk_indices, topk_probs)
    return output
```
**状态**: ⚠️ 简化版，可用但不完美  
**优先级**: 🔥 低 (短期可接受)

#### 完整实现 (可选)
```python
def sparsemax(logits, dim=-1):
    """完整的Sparsemax实现"""
    # 排序
    sorted_logits, _ = torch.sort(logits, dim=dim, descending=True)
    
    # 计算阈值
    cumsum = torch.cumsum(sorted_logits, dim=dim)
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device)
    support = (1 + k * sorted_logits) > cumsum
    k_z = support.sum(dim=dim, keepdim=True)
    tau = (cumsum.gather(dim, k_z - 1) - 1) / k_z
    
    # 应用阈值
    output = torch.clamp(logits - tau, min=0)
    return output
```
**状态**: ❌ 未实现 (可选优化)  
**优先级**: 🔥 低

---

## 📊 实现优先级总结

### 🔥🔥🔥 极高优先级 (必须实现)

1. **负载均衡损失** - 防止专家塌缩的关键
2. **三阶段训练配置** - Warmup → Transition → Fine-tune
3. **调整PPO超参数** - 学习率、clip_ratio、batch_size等

### 🔥🔥 高优先级 (强烈推荐)

4. **专家多样性损失** - 鼓励专家差异化
5. **学习率Warmup和退火** - 稳定训练
6. **NaN检测和回滚** - 防止崩溃
7. **TrainingMonitor类** - 实时监控
8. **梯度范数监控** - 检测爆炸

### 🔥 中优先级 (建议实现)

9. **GAT注意力正则化** - 防止过平滑
10. **奖励归一化** - 稳定价值估计
11. **V3奖励塑形** - 利用GAT和专家信息
12. **调整梯度裁剪** - max_norm=1.0

### 低优先级 (可选优化)

13. **完整Sparsemax实现** - 当前简化版可用
14. **双Critic** - 减少过估计
15. **边Dropout** - GAT正则化

---

## 📝 实现建议

### 立即行动 (开始实现训练脚本时)

1. 创建 `ablation_v3/train/train_v3_gat_moe.py`
2. 实现负载均衡损失和专家多样性损失
3. 实现三阶段训练配置
4. 调整PPO超参数
5. 实现TrainingMonitor类
6. 实现NaN检测和回滚

### 代码结构建议

```python
# ablation_v3/train/train_v3_gat_moe.py

# 1. 辅助损失函数
def load_balance_loss(alpha): ...
def expert_diversity_loss(expert_logits): ...

# 2. 训练配置
def get_training_config(episode): ...

# 3. 监控类
class TrainingMonitor: ...
class NaNDetector: ...
class RewardNormalizer: ...

# 4. 主训练循环
def train_v3(args):
    # 初始化
    policy_net = GATGuidedMoEPolicy(...)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    monitor = TrainingMonitor()
    nan_detector = NaNDetector(policy_net)
    
    # 训练循环
    for episode in range(args.episodes):
        config = get_training_config(episode)
        policy_net.use_sparsemax = config['use_sparsemax']
        
        # Episode循环
        ...
        
        # 更新网络
        actor_loss, critic_loss = ...
        lb_loss = load_balance_loss(alpha_history)
        div_loss = expert_diversity_loss(expert_logits_history)
        
        total_loss = (
            actor_loss + 
            0.5 * critic_loss + 
            config['load_balance_coef'] * lb_loss +
            0.01 * div_loss
        )
        
        # NaN检测
        if nan_detector.check_and_rollback(total_loss):
            continue
        
        # 反向传播
        ...
        
        # 监控
        monitor.log(episode, metrics)
```

---

## ✅ 检查清单

在开始训练前，确保：

- [ ] 负载均衡损失已实现
- [ ] 专家多样性损失已实现
- [ ] 三阶段训练配置已实现
- [ ] PPO超参数已调整 (lr=1e-4, clip=0.15, batch=256)
- [ ] TrainingMonitor已实现
- [ ] NaN检测和回滚已实现
- [ ] 梯度范数监控已实现
- [ ] 奖励归一化已实现
- [ ] 学习率Warmup已实现
- [ ] Checkpoint保存逻辑已实现

---

**总结**: 核心网络架构已实现，但训练流程中的稳定性措施（辅助损失、监控、Warmup等）需要在训练脚本中实现。

