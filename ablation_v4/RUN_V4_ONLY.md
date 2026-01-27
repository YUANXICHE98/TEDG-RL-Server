# V4 独立运行指南

## 🎯 目标

只运行V4训练，验证Cross-Attention机制的实现。

---

## 🚀 快速开始

### 1. 烟雾测试 (1分钟)

验证V4实现正确性：

```bash
python ablation_v4/test_v4_smoke.py
```

**预期输出**: 7个测试全部通过 ✓

---

### 2. 小规模测试 (100 episodes, 约30分钟)

```bash
bash ablation_v4/scripts/run_v4_test_100ep.sh
```

**这个脚本会**:
- 运行V4训练100 episodes
- 保存训练日志和模型检查点
- 输出训练统计信息

**查看结果**:
```bash
# 查看完整日志
cat ablation_v4/results/test_100ep.log

# 查看最后50行（包含最终统计）
tail -50 ablation_v4/results/test_100ep.log

# 查看训练日志JSON
cat ablation_v4/results/test_100ep/logs/training_log.json
```

---

### 3. 完整训练 (如果小规模测试正常)

#### Warmup阶段 (1000 episodes)
```bash
bash ablation_v4/scripts/run_warmup_1000.sh
```

#### Transition阶段 (3000 episodes)
```bash
bash ablation_v4/scripts/run_transition_3000.sh
```

#### Fine-tune阶段 (5000 episodes)
```bash
bash ablation_v4/scripts/run_finetune_5000.sh
```

---

## 📊 关键指标

### 训练过程中关注

1. **Episode Rewards/Scores**
   - 是否在增长
   - 是否稳定

2. **Alpha Entropy**
   - 目标: 0.3-0.5
   - 避免: 卡在0.693 (ln2)

3. **Expert Usage**
   - 4个专家是否都被使用
   - 使用是否均衡

4. **Loss收敛**
   - Policy Loss
   - Value Loss
   - Auxiliary Losses

### 训练日志中的关键信息

```
Episode 100/100:
  Reward: xxx
  Score: xxx
  Length: xxx
  Alpha Entropy: xxx  ← 关注这个
  Expert Usage Variance: xxx  ← 关注这个
```

---

## 🔍 检查训练是否正常

### ✅ 正常的标志

1. **无错误**: 没有NaN/Inf错误
2. **Loss下降**: Policy Loss和Value Loss逐渐下降
3. **Reward增长**: Episode Rewards有上升趋势
4. **专家使用**: 4个专家都有被使用

### ⚠️ 需要注意的问题

1. **NaN/Inf**: 学习率过大或梯度爆炸
2. **Loss不降**: 学习率过小或网络初始化问题
3. **Reward不增**: 探索不足或奖励设计问题
4. **Alpha Entropy卡在ln(2)**: Manager约束太弱

---

## 🛠️ 故障排查

### 训练失败

```bash
# 检查环境
conda activate tedg-rl-demo
python -c "import torch; print(torch.__version__)"
python -c "import nle; print('NLE OK')"

# 检查数据
ls data/hypergraph/hypergraph_gat_structure.json
```

### NaN/Inf错误

查看日志中的NaN检测信息：
```bash
grep "NaN" ablation_v4/results/test_100ep.log
```

如果出现NaN：
1. 检查学习率是否过大
2. 检查梯度裁剪是否生效
3. 尝试降低学习率: `--lr 5e-5`

### 训练太慢

```bash
# 检查是否使用GPU
python -c "import torch; print(torch.cuda.is_available())"

# 如果有GPU但没用上，在训练脚本中添加:
# --device cuda
```

---

## 📈 分析训练结果

### 提取关键指标

```bash
# 使用Python分析训练日志
python -c "
import json
with open('ablation_v4/results/test_100ep/logs/training_log.json') as f:
    data = json.load(f)
    
import numpy as np
print(f'Avg Reward: {np.mean(data[\"episode_rewards\"]):.2f}')
print(f'Avg Score: {np.mean(data[\"episode_scores\"]):.2f}')
print(f'Best Score: {np.max(data[\"episode_scores\"]):.0f}')

monitor = data.get('monitor_metrics', {})
if 'alpha_entropy' in monitor:
    print(f'Avg Alpha Entropy: {np.mean(monitor[\"alpha_entropy\"]):.4f}')
"
```

### 可视化训练曲线

```bash
# 使用matplotlib绘制曲线
python -c "
import json
import matplotlib.pyplot as plt
import numpy as np

with open('ablation_v4/results/test_100ep/logs/training_log.json') as f:
    data = json.load(f)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Rewards
axes[0].plot(data['episode_rewards'], alpha=0.3)
window = 10
if len(data['episode_rewards']) > window:
    smoothed = np.convolve(data['episode_rewards'], np.ones(window)/window, mode='valid')
    axes[0].plot(range(window-1, len(data['episode_rewards'])), smoothed, linewidth=2)
axes[0].set_title('Episode Rewards')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')
axes[0].grid(True, alpha=0.3)

# Alpha Entropy
monitor = data.get('monitor_metrics', {})
if 'alpha_entropy' in monitor:
    axes[1].plot(monitor['alpha_entropy'], alpha=0.5)
    axes[1].axhline(0.693, color='red', linestyle='--', label='ln(2)')
    axes[1].axhline(0.5, color='green', linestyle='--', label='Target')
    axes[1].set_title('Alpha Entropy')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Entropy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ablation_v4/results/test_100ep_curves.png', dpi=150)
print('Saved to: ablation_v4/results/test_100ep_curves.png')
"
```

---

## 🎯 下一步

### 如果100 episodes测试正常

✅ **继续完整训练**
```bash
# 依次运行三个阶段
bash ablation_v4/scripts/run_warmup_1000.sh
bash ablation_v4/scripts/run_transition_3000.sh
bash ablation_v4/scripts/run_finetune_5000.sh
```

### 如果出现问题

⚠️ **调试和调整**
1. 检查错误日志
2. 调整超参数
3. 重新运行小规模测试

---

## 📝 训练配置

### 当前配置

```python
--exp-name test_100ep
--episodes 100
--max-steps 500
--num-experts 4
```

### 可调参数

如果需要调整：

```bash
# 更长的episode
python ablation_v4/train/train_v4_cross_attention.py \
    --exp-name test_100ep \
    --episodes 100 \
    --max-steps 1000 \  # 增加到1000步
    --num-experts 4

# 更小的学习率
python ablation_v4/train/train_v4_cross_attention.py \
    --exp-name test_100ep \
    --episodes 100 \
    --max-steps 500 \
    --num-experts 4 \
    --lr 5e-5  # 降低学习率
```

---

**创建时间**: 2026-01-22  
**预计时间**: 烟雾测试1分钟 + 小规模测试30分钟 = 31分钟  
**状态**: ✅ 准备就绪
