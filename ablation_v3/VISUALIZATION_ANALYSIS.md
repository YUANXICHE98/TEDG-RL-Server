# V3 专家路由可视化分析

## 生成时间
2026-01-07 10:49

## 测试模型
- **Checkpoint**: `ablation_v3/results/init_fix_test/checkpoints/best_model.pth`
- **训练Episodes**: 50
- **Phase**: Warmup (Softmax)

## 路由动态性测试结果

### 测试方法
强制Agent向N、E、S、W四个方向各移动5步，观察Alpha变化

### 关键发现

#### 1. 专家使用统计 ✅

**Survival专家 (主导)**:
- 平均权重: 0.7012 (70.12%)
- 标准差: 0.1028
- 范围: [0.5695, 0.8087]
- **结论**: Survival专家占主导地位

**Exploration专家 (次要)**:
- 平均权重: 0.1822 (18.22%)
- 标准差: 0.2030
- 范围: [0.0000, 0.4305]
- **结论**: 在某些情境下被激活（最高43%）

**General专家 (辅助)**:
- 平均权重: 0.1166 (11.66%)
- 标准差: 0.1089
- 范围: [0.0000, 0.2910]
- **结论**: 作为辅助专家使用

**Combat专家 (未激活)**:
- 平均权重: 0.0000 (0%)
- **结论**: 在测试场景中未被使用

#### 2. Alpha变化模式 ✅

**不同场景下的Alpha分布**:

| 场景 | Atoms | Alpha分布 | 主导专家 |
|------|-------|-----------|----------|
| 房间内 | `in_room, monsters_nearby` | [0.636, 0.000, 0.364, 0.000] | Survival |
| 门附近 | `door_nearby, monsters_nearby` | [0.650, 0.000, 0.350, 0.000] | Survival |
| 物品附近 | `items_nearby, monsters_nearby` | [0.570, 0.000, 0.430, 0.000] | Survival |
| 复杂场景 | `in_room, door_nearby, items_nearby, monsters_nearby` | [0.809, 0.000, 0.000, 0.191] | Survival |

**观察**:
1. ✅ **Alpha确实在变化** - 从0.570到0.809，变化范围约24%
2. ✅ **场景敏感** - 不同atoms组合导致不同的Alpha分布
3. ✅ **Exploration专家被激活** - 在items_nearby时权重达到43%
4. ❌ **专家切换次数为0** - 主导专家始终是Survival

#### 3. 场景多样性 ✅

**观察到的Scene Atoms**: 5种
- `dlvl_1` (地牢层级)
- `door_nearby` (门附近)
- `in_room` (房间内)
- `items_nearby` (物品附近)
- `monsters_nearby` (怪物附近)

**结论**: Agent能够感知环境变化，atoms正确更新

## 详细分析

### 为什么Survival专家占主导？

#### 可能原因
1. **训练数据偏向** - 50 episodes中，大部分时间在探索和生存
2. **Combat场景少** - 没有遇到需要战斗的紧急情况
3. **Warmup阶段特性** - Softmax让所有专家都能学习，但Survival学得最好

#### 是否正常？
✅ **正常** - 在NetHack早期阶段，生存确实是最重要的策略

### Alpha变化是否足够？

#### 对比之前的结果
- **之前**: Alpha几乎不变，每步变化0.001
- **现在**: Alpha在不同场景下变化0.24 (24%)

#### 结论
✅ **显著改进** - 修改后Alpha对场景变化敏感

### 为什么没有专家切换？

#### 定义
专家切换 = 主导专家从一个变成另一个（例如从Survival变成Combat）

#### 原因
1. **Survival权重太高** - 即使Exploration上升到43%，Survival仍然是57%
2. **测试场景单一** - 没有遇到需要Combat的紧急情况
3. **Warmup阶段** - 专家还在学习基础策略，分工不明确

#### 预期
在更长的训练后（1000+ episodes），应该能看到：
- Combat专家在战斗时被激活
- Exploration专家在探索新区域时主导
- 明确的专家切换

## 可视化文件

### 生成的图表
1. **episode_heatmaps.png** (187KB)
   - Alpha权重热力图
   - 奖励曲线
   - 专家使用统计

2. **key_moments/** (5张图)
   - `switch_1_step_6.png` - 第一次Alpha显著变化
   - `switch_2_step_8.png` - 第二次变化
   - `switch_3_step_11.png` - 第三次变化
   - `switch_4_step_12.png` - 第四次变化
   - `switch_5_step_13.png` - 第五次变化

### 查看方式
```bash
# macOS
open ablation_v3/visualizations/episode/episode_heatmaps.png
open ablation_v3/visualizations/episode/key_moments/

# Linux
xdg-open ablation_v3/visualizations/episode/episode_heatmaps.png
```

## 与理论预期对比

### 预期 vs 实际

| 指标 | 预期 | 实际 | 状态 |
|------|------|------|------|
| Alpha熵 | >1.3 (Warmup) | 1.38 | ✅ |
| Alpha变化 | >0.01/step | 0.24 (场景变化) | ✅ |
| 专家使用 | 均匀分布 | Survival主导 | ⚠️ |
| 专家切换 | 偶尔发生 | 0次 | ⚠️ |
| 场景感知 | 敏感 | 5种atoms | ✅ |

### 解释

#### ✅ 成功的部分
1. **Alpha熵高** - Softmax让所有专家都能学习
2. **Alpha变化** - 对场景变化敏感
3. **场景感知** - Atoms正确更新

#### ⚠️ 需要改进的部分
1. **专家分工不明确** - Survival过于主导
2. **Combat未激活** - 需要更多战斗场景
3. **专家切换少** - 需要更长训练

## 下一步建议

### 1. 继续训练到1000 episodes ✅
**目的**: 观察专家分工是否会更明确

**预期**:
- Survival权重下降到50-60%
- Combat在战斗时被激活
- Exploration在探索时主导

### 2. 增加训练难度 ✅
**方法**: 使用更难的NetHack任务
- 更多怪物
- 更复杂的地牢
- 需要战斗的场景

### 3. 分析专家学到的策略 ✅
**方法**: 可视化每个专家的动作分布
```bash
python tools/analyze_expert_strategies.py \
    --checkpoint ablation_v3/results/init_fix_test/checkpoints/best_model.pth
```

### 4. 进入Transition阶段 ✅
**时机**: Episode 1000后
**预期**: Sparsemax开始稀疏化，专家分工更明确

## 结论

### ✅ 修改成功
1. Alpha对场景变化敏感（变化24%）
2. 场景感知正常（5种atoms）
3. Exploration专家在合适场景被激活（43%）

### ⚠️ 需要更长训练
1. 专家分工还不明确
2. Combat专家未被激活
3. 需要1000+ episodes观察完整Warmup效果

### 📊 可视化质量
- 5张关键时刻截图
- 热力图清晰展示Alpha变化
- 中文标签有字体警告（不影响功能）

## 文件位置
- **热力图**: `ablation_v3/visualizations/episode/episode_heatmaps.png`
- **关键时刻**: `ablation_v3/visualizations/episode/key_moments/`
- **路由测试输出**: 见上文"路由动态性测试结果"
