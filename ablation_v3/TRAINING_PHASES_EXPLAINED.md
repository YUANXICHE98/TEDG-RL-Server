# V3 训练阶段详解

## 是的，V3使用分阶段训练！

V3架构设计了**三阶段渐进式训练策略**，这是代码中已经实现的功能。

---

## 三阶段训练流程

### 📊 阶段概览

```
Episode 0 ────────────> 1000 ────────────> 3000 ────────────> ∞
         Warmup              Transition           Fine-tune
         
路由:    Softmax             Sparsemax           Sparsemax
温度:    1.0                 1.0 → 0.5           0.5
学习率:  1e-4                5e-5                1e-5
目标:    广泛探索            开始专业化          精细调整
```

---

## 阶段1: Warmup (0-1000 episodes)

### 目标
让所有专家从各种情况中学习，建立基础知识

### 配置
```python
{
    'phase': 'warmup',
    'use_sparsemax': False,        # 使用Softmax路由
    'sparsemax_temp': 1.0,
    'learning_rate': 1e-4,         # 高学习率
    'entropy_coef': 0.05,
    'alpha_entropy_coef': 0.1,
    'load_balance_coef': 0.02,     # 强制负载均衡
    'diversity_coef': 0.01,
}
```

### 特点
- ✅ **Softmax路由**: 所有专家都参与决策
- ✅ **高Alpha熵**: ~1.385 (最大值，所有专家平等)
- ✅ **强负载均衡**: 防止某个专家主导
- ✅ **广泛探索**: 高方差，低平均分

### 预期表现
- 平均分数: 低 (5-15)
- 方差: 高
- Alpha熵: ~1.385 (稳定)
- 专家使用: 均匀分布

### 当前状态
✅ **已完成** (1000 episodes)
- 平均分数: 8.5
- 最高分数: 207
- Alpha熵: 1.3849 ± 0.0010
- 训练稳定，无崩溃

---

## 阶段2: Transition (1000-3000 episodes)

### 目标
平滑过渡到稀疏路由，让专家开始专业化

### 配置
```python
{
    'phase': 'transition',
    'use_sparsemax': True,         # 切换到Sparsemax
    'sparsemax_temp': 1.0 → 0.5,   # 温度退火
    'learning_rate': 5e-5,         # 中等学习率
    'entropy_coef': 0.02,
    'alpha_entropy_coef': 0.05,
    'load_balance_coef': 0.01,     # 中等负载均衡
    'diversity_coef': 0.01,
}
```

### 特点
- 🔄 **Sparsemax路由**: 开始稀疏化
- 📉 **温度退火**: 1.0 → 0.5 (逐渐增强稀疏性)
- 📉 **Alpha熵下降**: 1.385 → ~0.7
- 🎯 **专家专业化**: 开始出现分工

### 预期表现
- 平均分数: 提升 (15-30)
- 方差: 降低 (更稳定)
- Alpha熵: 逐渐下降
- 专家使用: 开始分化

### 当前状态
⏳ **待开始**

---

## 阶段3: Fine-tune (3000+ episodes)

### 目标
精细调整专家分工，优化性能

### 配置
```python
{
    'phase': 'fine-tune',
    'use_sparsemax': True,
    'sparsemax_temp': 0.5,         # 固定温度
    'learning_rate': 1e-5,         # 低学习率
    'entropy_coef': 0.01,
    'alpha_entropy_coef': 0.05,
    'load_balance_coef': 0.005,    # 弱负载均衡
    'diversity_coef': 0.005,
}
```

### 特点
- ✨ **完全Sparsemax**: 稀疏路由
- 🎯 **专家专业化**: 明确分工
- 📊 **低Alpha熵**: ~0.3-0.5
- 🔧 **精细调整**: 小学习率

### 预期表现
- 平均分数: 最高 (30-50+)
- 方差: 最低 (稳定)
- Alpha熵: 低 (~0.3-0.5)
- 专家使用: 明确专业化

### 当前状态
⏳ **待开始**

---

## 代码实现位置

### 阶段判断函数
```python
# ablation_v3/train/train_v3_gat_moe.py, line 118-172

def get_training_config(episode: int) -> Dict:
    """根据训练阶段返回配置"""
    if episode < 1000:
        return {...}  # Warmup
    elif episode < 3000:
        return {...}  # Transition
    else:
        return {...}  # Fine-tune
```

### 训练循环调用
```python
# ablation_v3/train/train_v3_gat_moe.py, line 736-757

for episode in range(start_episode, args.episodes):
    # 获取当前阶段配置
    config = get_training_config(episode)
    
    # 更新网络配置
    policy_net.use_sparsemax = config['use_sparsemax']
    
    # 更新学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = config['learning_rate']
    
    # 打印阶段信息
    if episode in [0, 1000, 3000]:
        print(f"进入 {config['phase'].upper()} 阶段")
```

---

## 为什么需要分阶段训练？

### 问题: 直接使用Sparsemax会怎样？

如果从一开始就用Sparsemax:
- ❌ **专家塌缩**: 某个专家主导，其他专家不学习
- ❌ **局部最优**: 陷入次优解
- ❌ **不稳定**: 训练容易崩溃

### 解决方案: 渐进式训练

```
Warmup:      让所有专家都学到基础知识
             ↓
Transition:  逐渐引入稀疏性，开始分工
             ↓
Fine-tune:   完全稀疏化，精细调整
```

这类似于**课程学习 (Curriculum Learning)**:
1. 先学简单的（Softmax，所有专家参与）
2. 再学复杂的（Sparsemax，专家分工）
3. 最后精通（Fine-tune，优化细节）

---

## 当前训练进度

### ✅ 已完成
- **Warmup阶段** (0-1000 episodes)
  - 状态: 完成
  - 结果: 平均分8.5，最高207
  - Alpha熵: 1.3849 (符合预期)

### ⏳ 待进行
- **Transition阶段** (1000-3000 episodes)
  - 状态: 未开始
  - 预期: Alpha熵下降，专家开始专业化
  
- **Fine-tune阶段** (3000+ episodes)
  - 状态: 未开始
  - 预期: 最佳性能

---

## 如何继续训练？

### 方法1: 自动继续 (推荐)

训练脚本会自动判断阶段，只需指定总episode数：

```bash
# 训练到3000轮 (会自动经历Warmup → Transition)
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name full_training \
    --episodes 3000 \
    --max-steps 500 \
    --resume ablation_v3/results/warmup_1000/checkpoints/model_final.pth
```

### 方法2: 分段训练

也可以分段训练，每个阶段单独运行：

```bash
# 阶段1: Warmup (0-1000) - 已完成
# ✅ Done

# 阶段2: Transition (1000-3000)
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name transition_3000 \
    --episodes 3000 \
    --max-steps 500 \
    --resume ablation_v3/results/warmup_1000/checkpoints/model_final.pth

# 阶段3: Fine-tune (3000+)
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name finetune_5000 \
    --episodes 5000 \
    --max-steps 500 \
    --resume ablation_v3/results/transition_3000/checkpoints/model_final.pth
```

---

## 关键指标监控

### Warmup → Transition 转换点 (Episode 1000)

监控指标:
- ✅ **路由方式**: Softmax → Sparsemax
- ✅ **Alpha熵**: 开始下降 (1.385 → ~1.0)
- ✅ **学习率**: 1e-4 → 5e-5
- ✅ **负载均衡**: 0.02 → 0.01

### Transition → Fine-tune 转换点 (Episode 3000)

监控指标:
- ✅ **温度**: 固定在0.5
- ✅ **Alpha熵**: 继续下降 (~0.7 → ~0.5)
- ✅ **学习率**: 5e-5 → 1e-5
- ✅ **性能**: 应该明显提升

---

## 常见问题

### Q1: 为什么Warmup阶段表现这么差？
**A**: 这是预期的！Warmup阶段的目标是探索，不是优化性能。所有专家都在学习基础知识，还没有专业化。

### Q2: 可以跳过Warmup直接用Sparsemax吗？
**A**: 不推荐。直接用Sparsemax容易导致专家塌缩（某个专家主导，其他不学习）。

### Q3: 可以调整阶段边界吗？
**A**: 可以！修改 `get_training_config()` 函数中的边界值：
```python
if episode < 1000:      # 改成 500 或 1500
    # Warmup
elif episode < 3000:    # 改成 2000 或 4000
    # Transition
```

### Q4: 如何知道当前在哪个阶段？
**A**: 训练日志会打印：
```
DEBUG: Episode 1000, Routing: Sparsemax, Phase: transition
```

### Q5: Transition阶段需要多久？
**A**: 2000 episodes (1000-3000)，预计2-3小时（CPU）。

---

## 总结

### ✅ 分阶段训练的优势

1. **避免专家塌缩**: Warmup让所有专家都学到知识
2. **平滑过渡**: Transition逐渐引入稀疏性
3. **更好性能**: Fine-tune精细调整
4. **训练稳定**: 渐进式比一步到位更稳定

### 🎯 当前建议

**继续训练到3000轮，完成Transition阶段**

这样可以看到：
- Alpha熵下降
- 专家开始专业化
- 性能提升
- 方差降低

---

**文档生成时间**: 2026-01-08  
**代码位置**: `ablation_v3/train/train_v3_gat_moe.py`  
**相关文档**: `ablation_v3/WARMUP_1000_RESULTS.md`
