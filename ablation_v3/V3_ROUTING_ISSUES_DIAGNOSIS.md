# V3 路由问题诊断报告

> **日期**: 2025-01-06  
> **问题**: 专家不切换，动作选择接近随机  
> **状态**: 🔴 需要修复

---

## 🔍 问题现象

### 1. 专家路由固定
```
Step 0-50: Alpha = [0.6775, 0.3225, 0.0, 0.0]
```
- ❌ 50步中Alpha完全相同
- ❌ 只有Survival和Combat被激活
- ❌ Exploration和General权重为0

### 2. 动作选择接近随机
```
动作概率分布:
  teleport: 0.0455
  SE: 0.0446
  wait: 0.0443
  close: 0.0442
  look: 0.0441
```
- ❌ 所有动作概率接近 (0.043-0.046)
- ❌ 几乎是均匀分布 (1/23 ≈ 0.0435)
- ❌ 没有明确的策略

### 3. Expert Logits太小
```
Survival: mean=0.0121, std=0.0286, max=0.0815
Combat:   mean=0.0001, std=0.0336, max=0.0824
```
- ❌ Logits范围 [-0.06, 0.09]
- ❌ 融合后 [-0.02, 0.05]
- ❌ Softmax后接近均匀

---

## 🔬 根本原因分析

### 原因1: 状态向量不变 ⚠️

**观察**:
```python
Step 0: Atoms = ['player_alive', 'hp_full', 'dlvl_1']
Step 1: Atoms = ['player_alive', 'hp_full', 'dlvl_1']
Step 2: Atoms = ['player_alive', 'hp_full', 'dlvl_1']
...
```

**问题**:
- 状态提取只依赖blstats (HP, hunger, gold等)
- 没有利用glyphs (地图信息)
- 没有利用message (游戏事件)
- 导致状态向量几乎不变

**影响**:
- 路由器输入相同 → Alpha相同
- 无法根据环境变化调整专家

### 原因2: Expert初始化太保守 ⚠️

**代码**:
```python
# src/core/networks_v3_gat_moe.py, line 67
nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
```

**问题**:
- gain=0.01导致初始logits太小
- 训练50 episodes不足以放大logits
- 动作选择接近随机

**对比**:
- V1/V2使用gain=0.1或1.0
- V3使用0.01是为了稳定性，但过于保守

### 原因3: Sparsemax在Warmup阶段使用 ⚠️

**训练配置**:
```python
# 训练时: Warmup阶段应该用Softmax
# 实际: 一直用Sparsemax
use_sparsemax=True  # 固定为True
```

**问题**:
- Warmup阶段应该用Softmax探索
- Sparsemax会过早稀疏化
- 导致某些专家从未被训练

---

## 📊 数据验证

### 训练数据分析

从`CONVERGENCE_TEST_ANALYSIS.md`:
- ✅ 分数有提升: 0.6 → 5.7 (9.5x)
- ✅ 奖励转正: -0.78 → +4.48
- ✅ α熵稳定: 1.342 (Warmup正常)
- ❌ 但推理时专家固定

**结论**: 训练过程正常，但推理时状态不变导致路由固定

---

## 🔧 修复方案

### 方案A: 增强状态表示 (推荐) ⭐

**目标**: 让状态向量随环境变化

**修改**:
1. 添加glyphs特征 (周围8格的地形/怪物)
2. 添加message特征 (最近的游戏事件)
3. 添加时间特征 (step count, 上次动作等)

**实现**:
```python
# 在extract_state_from_obs中添加:
# 1. 提取glyphs周围8格
glyphs = obs.get("glyphs", np.zeros((21, 79)))
player_y, player_x = blstats[nh.NLE_BL_Y], blstats[nh.NLE_BL_X]
surrounding = glyphs[max(0, player_y-1):player_y+2, 
                     max(0, player_x-1):player_x+2]

# 2. 统计周围的怪物/物品/墙壁
monster_count = (surrounding == ord('d')).sum()  # 狗
wall_count = (surrounding == ord('-')).sum()
...

# 3. 添加到belief向量
belief[10] = monster_count / 9.0
belief[11] = wall_count / 9.0
...
```

**优点**:
- ✅ 状态向量会随环境变化
- ✅ 路由器能根据情况选择专家
- ✅ 不需要重新训练

**缺点**:
- ⚠️ 需要修改状态提取代码
- ⚠️ 可能需要重新训练以适应新特征

### 方案B: 增大Expert初始化增益

**目标**: 让logits更有区分度

**修改**:
```python
# src/core/networks_v3_gat_moe.py, line 67
nn.init.orthogonal_(self.network[-1].weight, gain=0.1)  # 0.01 → 0.1
```

**优点**:
- ✅ 简单，只改一行
- ✅ 动作选择更有区分度

**缺点**:
- ❌ 需要重新训练
- ⚠️ 可能影响训练稳定性

### 方案C: 三阶段路由策略 (最佳) ⭐⭐⭐

**目标**: 训练和推理使用正确的路由方式

**修改**:
```python
# 训练脚本中:
# Phase 1 (Warmup): use_sparsemax=False
# Phase 2 (Transition): 逐渐从Softmax过渡到Sparsemax
# Phase 3 (Fine-tune): use_sparsemax=True

# 推理时: 根据训练阶段选择
if episode < 1000:
    use_sparsemax = False
elif episode < 3000:
    use_sparsemax = True  # 或温度退火
else:
    use_sparsemax = True
```

**优点**:
- ✅ 符合设计理念
- ✅ Warmup阶段充分探索
- ✅ Fine-tune阶段稀疏化

**缺点**:
- ❌ 需要重新训练
- ⚠️ 需要修改训练脚本

---

## 🎯 推荐行动方案

### 立即执行 (不需要重新训练)

1. **方案A**: 增强状态表示
   - 添加glyphs特征
   - 添加时间特征
   - 测试路由是否变化

### 短期 (需要重新训练)

2. **方案C**: 实现三阶段路由
   - 修改训练脚本
   - Warmup用Softmax
   - 重新训练50 episodes验证

3. **方案B**: 增大初始化增益
   - gain: 0.01 → 0.1
   - 重新训练验证

### 长期 (完整训练)

4. **GPU大规模训练**
   - 10000 episodes
   - 完整三阶段
   - 对比V1/V2

---

## 📝 验证清单

修复后需要验证:

- [ ] Alpha随状态变化 (不同obs → 不同alpha)
- [ ] 专家切换 (50步中至少5次切换)
- [ ] 动作有区分度 (top-1概率 > 0.1)
- [ ] 分数提升 (50步内score > 0)
- [ ] 不同专家的top动作不同

---

## 🔍 深层问题思考

### 为什么训练时分数提升，但推理时失效？

**假设1**: 训练时的分数提升是偶然的
- 50 episodes样本量小
- Episode 17的50分可能是运气

**假设2**: 训练时状态有变化，但测试时没有
- 训练时可能遇到了不同的场景
- 测试时一直在同一个房间

**假设3**: 模型学到了某种策略，但不是通过专家路由
- 可能是Visual Stream学到的
- Logic Stream (GAT)贡献不大

**验证方法**:
- 查看训练日志中的alpha分布
- 对比不同episode的alpha变化
- 消融实验: 固定alpha vs 动态alpha

---

## 结论

**当前状态**: 🔴 V3方法有效性未验证

**核心问题**:
1. 状态向量不随环境变化
2. Expert logits太小
3. 路由策略不匹配训练阶段

**下一步**:
1. 先执行方案A (增强状态)，快速验证
2. 如果有效，执行方案C (三阶段路由)
3. 重新训练并验证收敛性

---

**报告人**: Kiro  
**日期**: 2025-01-06
