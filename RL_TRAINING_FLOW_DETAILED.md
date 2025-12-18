# TEDG-RL 训练全流程详解

## 概览

本文档详细描述 `train_confmatch.py` 的完整训练流程，包括每一步的输入输出、网络结构、超图匹配逻辑、PPO更新机制等。

---

## 一、初始化阶段

### 1.1 环境初始化
```python
env = gym.make("NetHackScore-v0")
obs, info = env.reset()
```

**NetHack 观测空间：**
- `obs["blstats"]` (27维): 游戏状态统计
  - 索引10: HP (当前生命值)
  - 索引11: HPMAX (最大生命值)
  - 索引12: DEPTH (地牢深度)
  - 索引13: GOLD (金币数量)
  - 索引9: SCORE (分数)
  - 索引21: HUNGER (饥饿度)
  - 索引16: AC (护甲等级)
  - 索引19: EXP (经验等级)
  - 索引0/1: X/Y (位置坐标)
  - 其他: 属性、状态标志等

- `obs["glyphs"]` (21×79): 地图符号矩阵
- `obs["message"]`: 游戏文本消息

### 1.2 超图匹配器初始化
```python
matcher = HypergraphMatcher(
    hypergraph_path="data/hypergraph/hypergraph_complete_real.json",
    weights=(0.35, 0.35, 0.2, 0.1),  # pre, scene, effect, rule 权重
    tau=200.0  # 时间衰减常数
)
```

**超图结构：**
- 每条超边包含4个通道：
  - `pre_nodes`: 前置条件（如 hp_full, has_gold, strong）
  - `scene_atoms`: 场景原子（如 dlvl_1, near_edge, ac_good）
  - `effect_metadata`: 效果/风险
  - `conditional_effects`: 规则/条件效果
- 每条超边对应一个 NetHack 动作（通过 operator 字段映射）

### 1.3 状态构造器初始化
```python
state_constructor = StateConstructor(
    belief_dim=50,      # 从 blstats 提取的游戏状态
    q_pre_dim=15,       # 前置条件嵌入
    q_scene_dim=15,     # 场景原子嵌入
    q_effect_dim=8,     # 效果嵌入
    q_rule_dim=10,      # 规则嵌入
    confidence_dim=1,   # 置信度
    goal_dim=16         # 目标嵌入
)
# 总维度: 50+15+15+8+10+1+16 = 115
```

### 1.4 网络初始化
```python
policy_net = MultiChannelPolicyNet(
    state_dim=115,
    action_dim=23,      # NetHack 动作空间
    hidden_dim=128
)
```

**网络架构：**
```
输入: state (115维)
  ↓
┌─────────────────────────────────────────────────────────┐
│ 4个独立的 Actor 网络（每个都是 MLP）                      │
├─────────────────────────────────────────────────────────┤
│ Actor_pre:    [115 → 128 → 128 → 23]                   │
│ Actor_scene:  [115 → 128 → 128 → 23]                   │
│ Actor_effect: [115 → 128 → 128 → 23]                   │
│ Actor_rule:   [115 → 128 → 128 → 23]                   │
└─────────────────────────────────────────────────────────┘
  ↓
每个 Actor 输出 23维 logits (未归一化的动作分数)
  ↓
┌─────────────────────────────────────────────────────────┐
│ AttentionWeightNet (注意力权重网络)                       │
├─────────────────────────────────────────────────────────┤
│ [115 → 64 → 64 → 4]                                     │
│ 输出: α = [α_pre, α_scene, α_effect, α_rule]            │
│ 经过 softmax 归一化，和为1                                │
└─────────────────────────────────────────────────────────┘
  ↓
融合: final_logits = α_pre * logits_pre 
                   + α_scene * logits_scene
                   + α_effect * logits_effect
                   + α_rule * logits_rule
  ↓
softmax(final_logits) → 动作概率分布 (23维)
  ↓
采样得到动作 a

同时：
┌─────────────────────────────────────────────────────────┐
│ Critic 网络（价值估计）                                   │
├─────────────────────────────────────────────────────────┤
│ [115 → 256 → 256 → 1]                                   │
│ 输出: V(s) - 状态价值估计                                │
└─────────────────────────────────────────────────────────┘
```

### 1.5 PPO 训练器初始化
```python
ppo_trainer = PPOTrainer(
    policy_net=policy_net,
    lr=3e-4,
    clip_ratio=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    ppo_epochs=3,
    batch_size=128
)
```

---

## 二、单个 Episode 训练流程

### 2.1 环境重置
```python
obs, info = env.reset()
t_now = 0  # 当前步数
episode_reward = 0
```

### 2.2 状态提取（每一步都执行）

#### Step 1: 从 blstats 提取 belief (50维)
```python
blstats = obs["blstats"]

# 核心特征
hp = blstats[nh.NLE_BL_HP]
hpmax = blstats[nh.NLE_BL_HPMAX]
hp_ratio = hp / max(1, hpmax)

depth = blstats[nh.NLE_BL_DEPTH]
gold = blstats[nh.NLE_BL_GOLD]
hunger = blstats[nh.NLE_HUNGER]
score = blstats[nh.NLE_BL_SCORE]
ac = blstats[nh.NLE_BL_AC]
exp_level = blstats[nh.NLE_BL_EXP]
x = blstats[nh.NLE_BL_X]
y = blstats[nh.NLE_BL_Y]

# 属性（通过 blstats[0:8] 推断，具体索引见代码）
# STR, DEX, CON, INT, WIS, CHA

# 构造 belief 向量 (50维)
belief = [
    hp_ratio, depth/50.0, gold/1000.0, hunger/5.0,
    str/25.0, dex/25.0, con/25.0, int/25.0, wis/25.0, cha/25.0,
    x/79.0, y/21.0, score/1000.0, ac/20.0, exp_level/30.0,
    # ... 其他归一化特征和状态标志
]
```

#### Step 2: 推断 pre_nodes（前置条件）
```python
pre_nodes = []

# 基于 HP
if hp_ratio >= 0.9:
    pre_nodes.append("hp_full")
elif hp_ratio <= 0.3:
    pre_nodes.append("hp_low")

# 基于饥饿度
if hunger >= 3:
    pre_nodes.append("hungry")

# 基于金币
if gold > 0:
    pre_nodes.append("has_gold")

# 基于属性
if str >= 16:
    pre_nodes.append("strong")

# 基于护甲
if ac <= 5:
    pre_nodes.append("well_armored")

# ... 更多条件
```

#### Step 3: 推断 scene_atoms（场景原子）
```python
scene_atoms = []

# 地牢深度
scene_atoms.append(f"dlvl_{depth}")

# 位置
if x <= 5 or x >= 74 or y <= 2 or y >= 19:
    scene_atoms.append("near_edge")

# 护甲等级
if ac <= 5:
    scene_atoms.append("ac_good")
elif ac >= 10:
    scene_atoms.append("ac_poor")

# 经验等级
if exp_level <= 3:
    scene_atoms.append("exp_low")
elif exp_level >= 10:
    scene_atoms.append("exp_high")

# ... 更多场景特征
```

#### Step 4: 超图匹配
```python
plot_atoms = {
    "pre": pre_nodes,
    "scene": scene_atoms,
    "effect": [],  # 当前未使用
    "rule": []     # 当前未使用
}

# 调用超图匹配器
topk_matches = matcher.match(
    plot_atoms=plot_atoms,
    t_now=float(t_now),
    t_i=float(t_now),  # ⚠️ 当前未启用时间衰减（t_i=t_now导致decay=1.0）
    top_k=8
)

# topk_matches 是一个列表，每个元素是 MatchResult:
# MatchResult(
#     edge_id=...,
#     score=...,      # 匹配分数（4通道覆盖度加权和 * 时间衰减）
#     coverages=...,  # (pre_cov, scene_cov, effect_cov, rule_cov)
#     operator=...    # 对应的 NetHack 动作
# )
```

**匹配分数计算：**
```python
# 对于每条超边 e：
pre_coverage = len(set(plot_atoms["pre"]) & set(e.pre_nodes)) / max(1, len(e.pre_nodes))
scene_coverage = len(set(plot_atoms["scene"]) & set(e.scene_atoms)) / max(1, len(e.scene_atoms))
effect_coverage = len(set(plot_atoms["effect"]) & set(e.effect_metadata)) / max(1, len(e.effect_metadata))
rule_coverage = len(set(plot_atoms["rule"]) & set(e.conditional_effects)) / max(1, len(e.conditional_effects))

# 加权融合
global_score = (0.35 * pre_coverage 
              + 0.35 * scene_coverage 
              + 0.2 * effect_coverage 
              + 0.1 * rule_coverage)

# 时间衰减（当前未启用）
decay = exp(-(t_now - t_i) / tau)  # 当前 t_i=t_now，所以 decay=1.0

# 最终分数
score = global_score * decay
```

#### Step 5: 通道内选边
```python
# 从 Top-K 中为每个通道选择最佳边
selected_edges = HypergraphMatcher.select_channel_edges(topk_matches)

# selected_edges 是一个字典:
# {
#     "pre": MatchResult(...),      # pre 通道覆盖度最高的边
#     "scene": MatchResult(...),    # scene 通道覆盖度最高的边
#     "effect": MatchResult(...),   # effect 通道覆盖度最高的边
#     "rule": MatchResult(...)      # rule 通道覆盖度最高的边
# }
```

#### Step 6: 计算置信度
```python
if not topk_matches:
    confidence = 0.0
else:
    confidence = max(r.score for r in topk_matches)
    # 置信度 = Top-K 中的最高匹配分数
```

#### Step 7: 构造 115维 state
```python
state = state_constructor.construct_state(
    belief=belief,                                    # 50维
    pre_nodes=selected_edges["pre"].edge.pre_nodes,   # → 15维嵌入
    scene_atoms=selected_edges["scene"].edge.scene_atoms,  # → 15维嵌入
    effect_metadata=selected_edges["effect"].edge.effect_metadata,  # → 8维嵌入
    conditional_effects=selected_edges["rule"].edge.conditional_effects,  # → 10维嵌入
    confidence=confidence,                            # 1维
    goal_embedding=np.zeros(16)                       # 16维（当前未使用）
)
# state.shape = (115,)
```

### 2.3 动作选择
```python
# 将 state 转为 tensor
state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

# 前向传播
with torch.no_grad():
    action_probs, alpha_weights, value = policy_net(state_tensor)
    # action_probs: (1, 23) - 动作概率分布
    # alpha_weights: (1, 4) - [α_pre, α_scene, α_effect, α_rule]
    # value: (1, 1) - V(s)

# 采样动作
action_dist = torch.distributions.Categorical(action_probs)
action = action_dist.sample().item()  # 0-22 之间的整数

# 计算 log_prob（用于 PPO 更新）
log_prob = action_dist.log_prob(torch.tensor([action])).item()
```

### 2.4 环境交互
```python
obs_next, reward, terminated, truncated, info = env.step(action)

# reward: NetHack 环境给的奖励（主要是分数增量）
# terminated: 游戏是否结束（死亡）
# truncated: 是否达到最大步数
```

### 2.5 存储经验
```python
ppo_trainer.store_transition(
    state=state,
    action=action,
    reward=reward,
    value=value.item(),
    log_prob=log_prob,
    done=terminated or truncated
)

episode_reward += reward
t_now += 1
```

### 2.6 循环直到 episode 结束
```python
while not (terminated or truncated) and t_now < max_steps:
    # 重复 2.2 - 2.5
    ...
```

---

## 三、PPO 更新阶段（每个 episode 结束后）

### 3.1 计算优势函数（GAE）
```python
# 对于每个时间步 t：
delta_t = reward_t + gamma * V(s_{t+1}) - V(s_t)

# GAE 优势估计：
A_t = delta_t + (gamma * lambda) * delta_{t+1} + (gamma * lambda)^2 * delta_{t+2} + ...

# 目标价值：
V_target_t = A_t + V(s_t)
```

### 3.2 PPO 更新（重复 ppo_epochs=3 次）
```python
for epoch in range(ppo_epochs):
    # 打乱数据
    indices = shuffle(range(len(buffer)))
    
    for batch_start in range(0, len(buffer), batch_size):
        batch_indices = indices[batch_start:batch_start+batch_size]
        
        # 提取 batch 数据
        states_batch = buffer.states[batch_indices]
        actions_batch = buffer.actions[batch_indices]
        old_log_probs_batch = buffer.log_probs[batch_indices]
        advantages_batch = buffer.advantages[batch_indices]
        returns_batch = buffer.returns[batch_indices]
        
        # 前向传播
        action_probs, alpha_weights, values = policy_net(states_batch)
        
        # 计算新的 log_prob
        action_dist = Categorical(action_probs)
        new_log_probs = action_dist.log_prob(actions_batch)
        
        # PPO clip 损失
        ratio = torch.exp(new_log_probs - old_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * advantages_batch
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic 损失
        critic_loss = F.mse_loss(values.squeeze(), returns_batch)
        
        # 熵正则化（鼓励探索）
        entropy = action_dist.entropy().mean()
        
        # 总损失
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
        optimizer.step()
```

---

## 四、日志输出（每 10 个 episode）

### 4.1 Episode 统计
```
Episode 10/10000:
  总奖励: 123.45
  步数: 456
  最终分数: 789
  平均α权重: [0.28, 0.31, 0.22, 0.19]
```

### 4.2 详细状态（verbose 模式）
```
[观测解析]
  HP: 14/14 (100%)
  深度: 1层
  金币: 0
  分数: 0
  饥饿度: 1
  护甲: 4
  经验: 0

[推断结果]
  前置条件: ['hp_full']
  场景原子: ['dlvl_1', 'ac_good', 'exp_low']

[超图匹配]
  Top-1: edge_42, score=0.85, operator=search
    覆盖度: pre=1.00, scene=0.80, effect=0.50, rule=0.60
  Top-2: edge_17, score=0.78, operator=move_north
    覆盖度: pre=0.90, scene=0.75, effect=0.45, rule=0.55
  ...

[通道选边]
  pre 通道: edge_42 (覆盖度=1.00)
  scene 通道: edge_42 (覆盖度=0.80)
  effect 通道: edge_17 (覆盖度=0.50)
  rule 通道: edge_42 (覆盖度=0.60)

[置信度]
  confidence = 0.85 (最高匹配分数)

[网络决策]
  α权重: [0.30, 0.28, 0.20, 0.22]
  选择动作: 5 (search)
  动作概率: 0.42
  V值: 12.34
```

---

## 五、检查点保存

### 5.1 保存时机
- 每 100 个 episode
- 当前最佳模型（最高平均奖励）
- 训练结束时

### 5.2 保存内容
```python
checkpoint = {
    'episode': episode_idx,
    'policy_net_state_dict': policy_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_reward': best_reward,
    'training_log': training_log
}
torch.save(checkpoint, f'results_confmatch/checkpoints/model_{episode_idx:05d}.pth')
```

---

## 六、当前实现的限制与待改进点

### 6.1 未启用的特性
1. **时间衰减**：`t_i=t_now` 导致 `decay=1.0`
   - 需要维护每条超边的历史时间戳
   
2. **置信度阈值分支**：未实现高/中/低置信度的不同策略
   - 高置信度：信任超图，严格掩码
   - 中置信度：放宽掩码
   - 低置信度：调用 LLM 或允许所有动作

3. **Query 动作**：未区分执行动作（A类）和查询动作（Q类）
   - Q类动作不改变环境状态，仅更新信念

4. **LLM 接入**：未实现低置信度时的 LLM fallback

5. **动作掩码应用**：虽然初始化了 ActionMasker，但未在动作选择时使用

### 6.2 已实现的核心特性
✅ 多通道超图匹配（4通道覆盖度计算）
✅ Top-K 选边 + 通道内最佳边选择
✅ 置信度计算（基于匹配分数）
✅ 多通道 Actor 网络 + α 权重融合
✅ PPO 训练循环
✅ blstats 索引修复（使用 NLE 官方常量）

---

## 七、训练超参数总结

| 参数 | 值 | 说明 |
|------|-----|------|
| num_episodes | 10000 | 总训练轮数 |
| max_steps | 1000 | 每轮最大步数 |
| learning_rate | 3e-4 | 学习率 |
| clip_ratio | 0.2 | PPO 裁剪比例 |
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE λ |
| ppo_epochs | 3 | 每次更新的 epoch 数 |
| batch_size | 128 | 批次大小 |
| tau | 200.0 | 时间衰减常数（未启用） |
| top_k | 8 | 超图匹配返回的边数 |
| weights | (0.35, 0.35, 0.2, 0.1) | 4通道权重 |

---

## 八、输入输出总结

### 输入
- NetHack 环境观测（blstats, glyphs, message）
- 超图数据（hypergraph_complete_real.json）

### 中间表示
- belief (50维): 游戏状态特征
- pre_nodes: 前置条件列表
- scene_atoms: 场景原子列表
- topk_matches: Top-K 匹配结果
- confidence: 置信度分数
- state (115维): 完整状态向量

### 输出
- action (0-22): NetHack 动作
- α_weights (4维): 通道注意力权重
- V(s): 状态价值估计

### 训练目标
- 最大化累积奖励（NetHack 分数）
- 学习合理的 α 权重分配

---

## 九、与论文/方法论的对齐

### 已对齐
✅ 多通道超图匹配
✅ 4通道覆盖度计算
✅ 置信度信号
✅ 多通道 Actor 融合

### 未对齐（待实现）
⏳ 时间衰减机制
⏳ 置信度阈值分支
⏳ Query 动作集
⏳ LLM grounding
⏳ 动作掩码约束

---

## 十、消融实验计划

| 实验名 | 时间衰减 | 置信度分支 | Query动作 | LLM | 说明 |
|--------|---------|-----------|----------|-----|------|
| full | ✅ | ✅ | ✅ | ✅ | 完整方法 |
| no_decay | ❌ | ✅ | ✅ | ✅ | 关闭时间衰减 |
| no_branch | ✅ | ❌ | ✅ | ✅ | 关闭置信度分支 |
| no_query | ✅ | ✅ | ❌ | ✅ | 关闭Query动作 |
| no_llm | ✅ | ✅ | ✅ | ❌ | 关闭LLM |
| baseline | ❌ | ❌ | ❌ | ❌ | 仅多通道匹配 |

**当前可运行的实验：**
- baseline（当前 train_confmatch.py 的实际状态）
- 对照组：train_verbose.py（HP伪置信度 + 随机选边）
