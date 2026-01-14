# V3 训练调试快速参考卡

> **紧急情况**: 训练出问题时快速查阅

---

## 🚨 问题1: 专家塌缩

### 症状
```
Alpha distribution: [0.95, 0.02, 0.02, 0.01]
Alpha entropy: 0.15
Expert usage: Survival=95%, Combat=2%, Exploration=2%, General=1%
```

### 诊断
- α熵 < 0.3 ✗
- 某个专家 > 80% ✗
- 其他专家梯度接近0 ✗

### 立即行动
```python
# 1. 增加负载均衡系数
load_balance_coef = 0.05  # 从0.01增加到0.05

# 2. 延长Warmup
warmup_episodes = 2000  # 从1000增加到2000

# 3. 增加Sparsemax温度
sparsemax_temp = 1.0  # 从0.5增加到1.0

# 4. 强制均匀初始化
def init_router_uniform(router):
    nn.init.constant_(router.router[-1].weight, 0)
    nn.init.constant_(router.router[-1].bias, 0)
```

### 预防措施
- Warmup阶段使用Softmax
- 监控α熵，低于0.5时警告
- 定期检查专家使用率

---

## 🚨 问题2: GAT过平滑

### 症状
```
GAT attention variance: 0.02
Operator activation rate: 3%
All node embeddings similar: cosine_sim > 0.95
```

### 诊断
- 注意力方差 < 0.05 ✗
- 节点嵌入趋同 ✗
- Intent Vector变化小 ✗

### 立即行动
```python
# 1. 减少GAT层数
num_gat_layers = 1  # 从2减少到1

# 2. 增加Dropout
gat_dropout = 0.3  # 从0.1增加到0.3

# 3. 添加边Dropout
edge_dropout = 0.2

# 4. 降低学习率
learning_rate = 5e-5  # 从1e-4降低到5e-5
```

### 预防措施
- 限制GAT层数 ≤ 2
- 使用残差连接
- 监控注意力方差

---

## 🚨 问题3: 梯度爆炸

### 症状
```
Gradient norm: 45.7
Loss: NaN
Parameters contain Inf
```

### 诊断
- 梯度范数 > 10.0 ✗
- 损失变为NaN ✗
- 参数爆炸 ✗

### 立即行动
```python
# 1. 立即回滚到上一个checkpoint
policy_net.load_state_dict(last_good_checkpoint)

# 2. 降低学习率
learning_rate = 1e-5  # 从1e-4降低到1e-5

# 3. 增加梯度裁剪
max_grad_norm = 0.5  # 从1.0降低到0.5

# 4. 检查数值稳定性
logits = torch.clamp(logits, -20, 20)
value = torch.clamp(value, -1000, 1000)
```

### 预防措施
- 每次更新前保存checkpoint
- 实时监控梯度范数
- 使用LayerNorm

---

## 🚨 问题4: 奖励不收敛

### 症状
```
Episode 5000: score=120
Episode 5100: score=115
Episode 5200: score=125
No improvement for 2000 episodes
```

### 诊断
- Score长期震荡 ✗
- 无明显上升趋势 ✗
- α权重混乱 ✗

### 立即行动
```python
# 1. 调整奖励塑形
r_progress_weight = 0.5  # 增加进展奖励权重
r_safety_weight = 0.3    # 降低安全惩罚

# 2. 增加探索
entropy_coef = 0.05  # 从0.01增加到0.05

# 3. 检查价值估计
# 使用双Critic
critic1 = Critic(...)
critic2 = Critic(...)
value = torch.min(critic1(z), critic2(z))

# 4. 增加batch_size
batch_size = 512  # 从256增加到512
```

### 预防措施
- 奖励归一化
- 监控价值估计偏差
- 定期调整奖励权重

---

## 🚨 问题5: NaN/Inf崩溃

### 症状
```
RuntimeError: Function 'CategoricalBackward' returned nan values
Loss: tensor(nan)
Logits contain inf
```

### 诊断
- 前向传播产生NaN ✗
- 损失计算出现Inf ✗
- 梯度包含NaN ✗

### 立即行动
```python
# 1. 添加NaN检测
def check_nan(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"⚠️ {name} contains NaN/Inf!")
        return True
    return False

# 2. 在关键位置检查
logits = policy_net(state)
if check_nan(logits, "logits"):
    # 回滚或跳过
    continue

# 3. 强制数值稳定
logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
logits = logits.clamp(-20.0, 20.0)

# 4. 检查输入数据
if check_nan(state, "state"):
    # 数据问题，跳过
    continue
```

### 预防措施
- 所有输出做nan_to_num
- 所有logits做clamp
- 实现自动回滚机制

---

## 📊 正常训练参考值

```
Episode 1000:
  score: 100-150
  alpha_entropy: 1.0-1.2
  gradient_norm: 1.5-3.0
  expert_usage: [0.25, 0.25, 0.25, 0.25]

Episode 3000:
  score: 300-400
  alpha_entropy: 0.6-0.8
  gradient_norm: 1.0-2.5
  expert_usage: [0.20, 0.35, 0.30, 0.15]

Episode 5000:
  score: 500-700
  alpha_entropy: 0.5-0.7
  gradient_norm: 0.8-2.0
  expert_usage: [0.25, 0.30, 0.30, 0.15]
```

---

## 🔧 紧急修复命令

### 重启训练 (从checkpoint)
```bash
python ablation_v3/train/train_v3_gat_moe.py \
  --exp-name v3_full \
  --resume ablation_v3/results/v3_full/checkpoints/model_05000.pth \
  --learning-rate 5e-5
```

### 降级到Softmax
```bash
python ablation_v3/train/train_v3_gat_moe.py \
  --exp-name v3_softmax \
  --no-sparsemax
```

### 固定GAT
```bash
python ablation_v3/train/train_v3_gat_moe.py \
  --exp-name v3_fixed_gat \
  --freeze-gat
```

### 减少专家
```bash
python ablation_v3/train/train_v3_gat_moe.py \
  --exp-name v3_2experts \
  --num-experts 2
```

---

## 📞 调试检查清单

训练出问题时，按顺序检查：

- [ ] 1. 查看最近50个episodes的指标
- [ ] 2. 检查α熵是否正常 (0.5-1.0)
- [ ] 3. 检查梯度范数是否正常 (<5.0)
- [ ] 4. 检查是否有NaN/Inf
- [ ] 5. 查看专家使用率是否均衡
- [ ] 6. 检查GAT注意力方差 (>0.1)
- [ ] 7. 查看奖励是否归一化
- [ ] 8. 检查学习率是否合适
- [ ] 9. 查看是否在正确的训练阶段
- [ ] 10. 考虑是否需要降级方案

---

## 💡 经验法则

1. **专家塌缩**: 增加负载均衡，延长Warmup
2. **GAT过平滑**: 减少层数，增加Dropout
3. **梯度爆炸**: 降低学习率，增加裁剪
4. **奖励不收敛**: 调整奖励塑形，增加探索
5. **NaN崩溃**: 检查数值稳定性，实现回滚

**记住**: 稳定性 > 性能。先保证训练不崩溃，再优化性能。

