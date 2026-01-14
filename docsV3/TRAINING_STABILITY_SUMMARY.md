# V3 训练稳定性措施 - 快速参考

> **完整文档**: `V3_TRAINING_STABILITY_CHECKLIST.md`  
> **目的**: 快速查阅关键稳定性措施

---

## 🎯 核心问题

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| **专家塌缩** | α熵<0.3, 某专家>80% | Warmup + 负载均衡损失 + 温度退火 |
| **GAT过平滑** | 注意力方差<0.05 | 限制2层 + 残差连接 + Dropout |
| **梯度爆炸** | 梯度范数>10.0 | 小学习率 + 梯度裁剪 + LayerNorm |
| **奖励不收敛** | Score长期不变 | 奖励塑形 + 熵正则 + 增加探索 |

---

## 📊 关键超参数

```python
# 学习率
learning_rate = 1e-4  # V2是3e-4，V3更小
warmup_steps = 1000
lr_scheduler = CosineAnnealingLR

# PPO
clip_ratio = 0.15     # V2是0.2，V3更保守
batch_size = 256      # V2是128，V3更大
ppo_epochs = 4        # V2是3，V3更充分

# 正则化
entropy_coef = 0.01           # 动作熵
alpha_entropy_coef = 0.05     # 专家熵
load_balance_coef = 0.01      # 负载均衡
diversity_coef = 0.01         # 专家多样性

# 梯度
max_grad_norm = 1.0   # V2是0.5，V3更宽松
```

---

## 🔄 三阶段训练

| 阶段 | Episodes | 路由方式 | 学习率 | 目的 |
|------|----------|----------|--------|------|
| **Warmup** | 0-1000 | Softmax | 1e-4 | 让专家学基础策略 |
| **Transition** | 1000-3000 | 温度退火 | 5e-5 | 平滑过渡到稀疏 |
| **Fine-tune** | 3000+ | Sparsemax | 1e-5 | 精细调整分工 |

```python
def get_training_config(episode):
    if episode < 1000:
        return {'use_sparsemax': False, 'lr': 1e-4}
    elif episode < 3000:
        temp = 1.0 - 0.5 * (episode - 1000) / 2000
        return {'use_sparsemax': True, 'temp': temp, 'lr': 5e-5}
    else:
        return {'use_sparsemax': True, 'temp': 0.5, 'lr': 1e-5}
```

---

## 🛡️ 必须实现的辅助损失

### 1. 负载均衡损失 (防止塌缩)

```python
def load_balance_loss(alpha):
    expert_usage = alpha.mean(dim=0)
    target = torch.ones_like(expert_usage) / num_experts
    return F.mse_loss(expert_usage, target)
```

### 2. 专家多样性损失 (鼓励差异化)

```python
def expert_diversity_loss(expert_logits):
    # 最小化专家间余弦相似度
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

### 3. 总损失组合

```python
total_loss = (
    actor_loss + 
    0.5 * critic_loss + 
    0.01 * load_balance_loss(alpha) +
    0.01 * expert_diversity_loss(expert_logits) -
    0.01 * entropy -
    0.05 * alpha_entropy
)
```

---

## 📈 监控指标

### 必须监控

| 指标 | 正常范围 | 异常阈值 |
|------|----------|----------|
| **alpha_entropy** | 0.5-1.0 | <0.3 或 >1.2 |
| **expert_usage** | 每个10-40% | 某个>80% |
| **gradient_norm** | <5.0 | >10.0 |
| **gat_attention_variance** | >0.1 | <0.05 |

### 监控代码

```python
class TrainingMonitor:
    def check_anomalies(self, metrics):
        if metrics['alpha_entropy'] < 0.3:
            print("⚠️ 专家塌缩!")
        if metrics['gradient_norm'] > 10.0:
            print("⚠️ 梯度爆炸!")
        if metrics['gat_attention_variance'] < 0.05:
            print("⚠️ GAT过平滑!")
```

---

## 🔧 数值稳定性

### NaN/Inf处理

```python
# 在forward中
logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
logits = logits.clamp(-20.0, 20.0)

# 在update中
if torch.isnan(loss) or torch.isinf(loss):
    print("⚠️ NaN detected! Skipping batch...")
    continue
```

### 梯度裁剪

```python
nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
```

### 奖励归一化

```python
class RewardNormalizer:
    def normalize(self, reward):
        normalized = (reward - self.mean) / (self.std + 1e-8)
        return np.clip(normalized, -10.0, 10.0)
```

---

## 🚨 降级方案

如果训练失败，按顺序尝试：

1. **固定GAT**: 冻结GAT参数，只训练路由和专家
2. **使用Softmax**: 禁用Sparsemax，全程使用Softmax
3. **减少专家**: 从4个减少到2个 (Survival + General)
4. **回退V2+GAT**: 用GAT提取特征，但用V2路由

---

## ✅ 实施前检查清单

- [ ] 学习率设为1e-4 (比V2小)
- [ ] 实现负载均衡损失
- [ ] 实现专家多样性损失
- [ ] 实现Warmup机制 (Softmax → Sparsemax)
- [ ] 实现温度退火
- [ ] 实现NaN检测和回滚
- [ ] 实现TrainingMonitor
- [ ] 梯度裁剪max_norm=1.0
- [ ] 所有logits做nan_to_num和clamp
- [ ] 每100 episodes保存checkpoint

---

## 📚 完整文档

详细说明请参考: `docsV3/V3_TRAINING_STABILITY_CHECKLIST.md`

