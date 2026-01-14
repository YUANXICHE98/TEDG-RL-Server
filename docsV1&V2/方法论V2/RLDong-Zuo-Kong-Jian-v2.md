# RL 动作空间完整版 v2.0（超图投影版）

> **版本升级背景**：从静态KG到静态超图框架下，RL动作空间需要适配新的"置信度+子图投影"机制，确保动作集动态生成但精简高效。

## 核心设计原则

在TEDG-RL v2.0超图框架中，RL动作空间设计遵循：

1. **超图驱动的动作过滤**：不是固定的全动作集，而是根据当前投影子图 $G_t^{\text{proj}}$ 和信念状态 $b_t$ 动态生成可行动作。
2. **置信度感知的分层**：低置信度（<0.78）时优先查询动作，高置信度时优先执行动作。
3. **层次化减轻负担**：分原子执行、宏查询、宏序列三层，避免细动作爆炸，每步总数20-55个。
4. **掩码约束**：RL只能在action_mask=1的动作上采样（由可行性检查器生成）。

---

## 一、动作空间整体架构

```
动作空间 = 执行动作(A类) + 查询动作(Q类) + 层次动作(H类)

可行动作 O_t = {o ∈ O : ∀pre ∈ PRE(o), pre ∈ b_t, π_t(pre) ≥ 0.5}
                ∪ {q ∈ Q : uncertainty(b_t) > unc_th}
                ∪ {h ∈ H : ∃o ∈ O_t, h 为 o 的宏包装}

动作掩码 mask_t ∈ {0,1}^{|O|+|Q|+|H|} (由FeasibilityChecker + RL投影模块生成)
```

---

## 二、执行动作（A类）：NetHack完整版

**特点**：原子操作，直接改变环境状态。来自任务超图 $G_T$ 的原始动作库。

### 2.1 A类动作表（按超图中的超边分类）

| 超边ID | 动作名称 | 描述 | NetHack示例 | 前提条件(超边pre集合) | 状态转移(超边eff) | 典型规模/步 | 掩码生成规则 |
|--------|----------|------|------------|---------------------|------------------|-----------|-----------|
| HE_move_* | move(direction) | 单步/跑步移动 | move(north) (h/H) | ¬blocked(direction), within_bounds | player.pos ← next_pos | 25-35 | ∀dir, ¬wall_at(dir) → mask=1 |
| HE_fight_* | fight(monster) | 近战攻击怪物 | fight(goblin) (F) | adjacent(monster), ¬swallowed | monster.hp ↓, player.exp ↑ | 25-35 | ∃mon adjacent → mask=1 |
| HE_throw_* | throw(item, target) | 投掷物品 | throw(dagger) (t) | holding(item), ¬blind | target.hp ↓ or item.location=floor | 25-35 | holding(item) ∧ ∃target → mask=1 |
| HE_fire_* | fire(projectile) | 射击弹药 | fire(arrow) (f/Q) | ∃quiver, quiver.count>0 | projectile.in_flight=true | 25-35 | quiver.count>0 ∧ ∃target → mask=1 |
| HE_zap_* | zap(wand, direction) | 挥魔杖 | zap(goblin) (z) | holding(wand), wand.charge>0 | wand.charge ↓, target affected | 25-35 | holding(wand) ∧ wand.charge>0 → mask=1 |
| HE_cast_* | cast(spell) | 施法 | cast(fireball) (Z) | know_spell(spell), player.mana≥cost | spell.effect executed | 25-35 | know_spell ∧ mana≥cost → mask=1 |
| HE_open_* | open(door/container) | 开门/容器 | open(door) (o) | adjacent(door), ¬locked | door.state=open | 25-35 | adjacent(door) ∧ ¬locked → mask=1 |
| HE_close_* | close(door) | 关门 | close(door) (c) | adjacent(door), door.state=open | door.state=closed | 25-35 | adjacent(door) ∧ open → mask=1 |
| HE_kick_* | kick(door/monster) | 踢物体/怪物 | kick(door) (^D/k) | adjacent(target) | target moved/damaged | 25-35 | adjacent(target) → mask=1 |
| HE_wield_* | wield(weapon) | 装备武器 | wield(sword) (w) | holding(weapon) | weapon.equipped=true | 25-35 | holding(weapon) → mask=1 |
| HE_wear_* | wear(armor) | 穿甲 | wear(armor) (W) | holding(armor) | armor.equipped=true | 25-35 | holding(armor) → mask=1 |
| HE_takeoff_* | takeoff(armor) | 脱甲 | takeoff(armor) (T/A) | equipped(armor) | armor.equipped=false | 25-35 | equipped(armor) → mask=1 |
| HE_drop_* | drop(item) | 丢弃物品 | drop(potion) (d/D) | holding(item) | item.location=floor | 25-35 | holding(item) → mask=1 |
| HE_pickup_* | pickup(item) | 捡起物品 | pickup(coin) (,) | on_floor(item), ¬inventory_full | holding(item)=true | 25-35 | on_floor(item) ∧ ¬full → mask=1 |
| HE_eat_* | eat(food) | 食用食物 | eat(corpse) (e) | holding(food), ¬swallowed | player.hunger ↓, potential effect | 25-35 | holding(food) → mask=1 |
| HE_quaff_* | quaff(potion) | 喝药水 | quaff(healing) (q) | holding(potion) | effect applied (heal/cure/etc) | 25-35 | holding(potion) → mask=1 |
| HE_read_* | read(scroll) | 阅读卷轴 | read(identify) (r) | holding(scroll), ¬blind | effect executed | 25-35 | holding(scroll) ∧ ¬blind → mask=1 |
| HE_apply_* | apply(tool) | 使用工具 | apply(key) (a) | holding(tool) | tool effect applied | 25-35 | holding(tool) → mask=1 |
| HE_engrave_* | engrave(text) | 刻字地板 | engrave(Elbereth) (E) | ¬blind (optional) | altar/floor marked | 25-35 | true → mask=1 |
| HE_search_* | search(area) | 搜索隐藏 | search(trap) (s) | in(area) | hidden items/traps revealed | 25-35 | true → mask=1 |
| HE_wait_* | wait(turns) | 等待/休息 | wait(1) (.) | true | time advances | 25-35 | true → mask=1 |
| HE_upstairs_* | go_upstairs | 上楼梯 | upstairs (<) | on(stairs), stairs.type=up | level.current ↑ | 25-35 | on(stairs) → mask=1 |
| HE_downstairs_* | go_downstairs | 下楼梯 | downstairs (>) | on(stairs), stairs.type=down | level.current ↓ | 25-35 | on(stairs) → mask=1 |
| HE_swap_* | swap_weapons | 交换武器 | swap_weapons (x) | dual_wield_capable | weapon.primary ↔ weapon.off | 25-35 | dual_wield → mask=1 |
| HE_twoweapon_* | twoweapon(toggle) | 双持战斗 | twoweapon (X) | dual_wield_capable | dual_mode toggled | 25-35 | dual_wield → mask=1 |
| HE_pay_* | pay_bill | 付账 | pay_bill (p) | in(shop), ∃bill | balance updated | 25-35 | in(shop) ∧ bill → mask=1 |

**设计说明**：
- 每个A类动作对应超图中的一条或多条超边变体（e.g., move有8个方向对应多条HE_move_*）
- 前提条件直接来自超边的 $\{\text{PRE}\}_i$，状态转移来自 $\{\text{EFF}\}_{i,j}^{\text{cond}}$
- 掩码生成规则：FeasibilityChecker根据 $b_t$ 检查前提，若满足则mask=1
- 每步可用数：10-30个（动态过滤，仅保留mask=1的）

---

## 三、查询动作（Q类）：宏查询简化版

**特点**：信息获取动作，不改变环境状态，仅更新信念层 $G_E^{(t)}$。与信息增益相关。

### 3.1 Q类动作表（按查询类别）

| 类别 | 动作名称 | 描述 | 示例 (NetHack) | 查询目标 | 返回信息格式 | 掩码生成规则 | 成本 | 置信度<0.78时优先度 |
|------|----------|------|---------------|---------|------------|-----------|------|-------------------|
| 属性查询 | query_inventory | 查看库存 | i/I | 当前holding | item list + types | true | 0.05 | P1: 库存不确定时 |
| 属性查询 | query_look | 观察位置 | :/; | 当前区域 | objects/monsters/features | true | 0.05 | P1: 位置不确定时 |
| 属性查询 | query_whatis | 符号含义 | / | 任意符号 | 对象类型 | true | 0.03 | P2: 未知符号 |
| 属性查询 | query_attributes | 角色属性 | ^X/#attributes | 玩家 | STR/DEX/etc | true | 0.05 | P1: HP/能力不确定 |
| 属性查询 | query_seetrap | 查陷阱 | ^/#seetrap | 相邻陷阱 | trap位置/类型 | adjacent(trap) | 0.05 | P1: 陷阱存在时 |
| 属性查询 | query_overview | 地牢概览 | ^O/#overview | 全关卡 | 地图 | true | 0.03 | P3: 长期规划 |
| 属性查询 | query_known | 已发现物品 | `/#known | 发现列表 | identified item list | true | 0.03 | P2: 物品辨识 |
| 属性查询 | query_equipment | 装备状态 | */**/(等 | 当前装备 | 武器/甲/环/项等 | true | 0.05 | P1: 装备状态不确定 |
| 属性查询 | query_spells | 已知咒语 | +/#seespells | 已学咒语 | spell list + cost/dmg | true | 0.03 | P2: 法术规划 |
| 属性查询 | query_help | 命令帮助 | ?/&/# | 键绑定 | command help text | true | 0.02 | P3: 命令帮助 |
| 属性查询 | query_conduct | 行为挑战 | #conduct | 素食/无伤等 | challenge status | true | 0.02 | P3: 长期跟踪 |
| 属性查询 | query_history | 游戏历史 | V/#history | 版本/历史 | version + history log | true | 0.01 | P3: 背景信息 |
| 预条件探测 | probe_pre(operator) | 检查单个前提 | (synthetic) | operator o的pre | which pre satisfied/not | ∀o with pre → mask=1 | 0.10 | P1: 前提不确定 |
| 预条件探测 | probe_all_pre(operator) | 探测所有前提 | (synthetic) | operator o的所有pre | all pre status + gaps | ∃o ∧ ¬all_pre_sat | 0.20 | P1: 前提集合不确定 |
| 状态关系 | query_relation_between | 查询实体关系 | (synthetic) | entity pair (e.g., player-monster) | distance/adjacency | ∃pair → mask=1 | 0.08 | P1: 相对位置不确定 |
| 状态关系 | query_failure_reason | 查询失败原因 | (synthetic) | 最后失败动作 | pre/eff failure reason | ∃recent_failure | 0.15 | P1: 失败调试 |
| 信息融合 | resolve_conflict | 冲突消解 | (synthetic) | 冲突原子对 | merged fact + confidence | conflict_detected(b_t) | 0.12 | P1: 信念冲突时 |
| 信息融合 | fusion_summary | 融合总结 | (synthetic) | 当前b_t中的原子 | fused belief snapshot + w_i | true | 0.10 | P2: 周期性同步 |

**设计说明**：
- 前3-4组（属性查询）对应NetHack原生命令（i, :, /, ^X等），易转换
- 中间组（预条件探测）是合成查询，直接返回超边pre的满足度
- 后2组（状态关系、冲突消解）是新增，支持信念融合
- 掩码生成规则：低置信度(<0.78)时自动激活相关Q类（e.g., probe_pre在pre_unsatisfied_count>0时mask=1）
- 成本：0.02-0.20 tokens等价，标准化为-λ_qry·成本的奖励惩罚
- 优先度：P1 > P2 > P3，在三层降级策略中按顺序尝试

---

## 四、层次动作（H类）：宏序列简化版

**特点**：高层复合动作，是多步原子链的宏包装。减轻RL的细粒度探索负担。

### 4.1 H类动作表（按场景）

| 宏序列ID | 宏序列名称 | 描述 | 伪代码(原子链) | 前提条件 | 隐含效果 | 掩码生成规则 | 成本 | 推荐置信度 |
|---------|-----------|------|---------------|---------|---------|-----------|------|-----------|
| H_combat_1 | combat_sequence | 战斗链 | wield(w); fight(m); throw(p) | monster.adjacent, ¬weapon_equipped | monster defeated or damaged | monster.adjacent | 3 | high |
| H_explore_1 | explore_room | 房间探索 | search(area); open(door); loot(container) | ¬explored(area) | new entities discovered | ¬explored(area) | 4 | high |
| H_loot_1 | inventory_manage | 库存整理 | query_inventory; drop(unwanted); wield(best) | inventory.count≈full | inventory optimized | inventory≈full | 3 | high |
| H_container_1 | loot_container | 容器掠夺 | open(container); pickup(all); query_look | ¬opened(container) | items in inventory | ¬opened(container) | 2 | medium |
| H_recover_1 | recover_hp | 恢复生命 | eat(food) OR quaff(potion) OR pray() | player.hp<threshold | hp restored | player.hp<threshold | 2 | high |
| H_stairs_1 | travel_stairs | 楼梯旅行 | query_look; go_upstairs OR downstairs | on(stairs) | level changed, belief reset | on(stairs) | 1 | high |
| H_trap_1 | untrap_path | 拆陷阱 | query_seetrap; search(trap); jump/cast | trap_suspected | trap triggered/avoided | trap_suspected | 3 | medium |
| H_equip_1 | equip_fight | 装备战斗 | wear(armor); wield(weapon); twoweapon(on) | ¬combat_ready | combat mode enabled | ¬combat_ready | 3 | high |
| H_sacrifice_1 | sacrifice_ritual | 祭献祭坛 | pickup(corpse); apply(altar); query_attributes | ¬used(altar), holding(corpse) | divine effect | corpse_available | 2 | medium |
| H_shop_1 | shop_trade | 商店交易 | query_inventory; query_look; pay_bill; pickup | in(shop) | trade completed | in(shop) | 3 | medium |
| H_spell_1 | spell_combo | 法术连击 | query_spells; cast(offensive); cast(heal) | know_spell, mana≥cost | spell effects | know_spell | 2 | high |
| H_door_1 | door_passage | 门道通过 | apply(key) OR unlock_spell; open(door); move(through) | locked(door) | door opened, room entered | locked(door) | 3 | high |

**设计说明**：
- 每个H类动作是2-5个原子A类的有序组合，形成常见"脚本"
- 前提条件由原子链中最严格的AND组成
- 隐含效果：自动执行链中所有原子的综合eff
- 掩码生成规则：超图投影后，若投影子图中存在对应原子链的所有节点，则mask=1
- 成本：原子链长度的加权和，一般2-4 steps等价
- 推荐置信度：high时优先选H类（降低步数），low时退回A类（更灵活）

---

## 五、动作掩码生成与RL约束

### 5.1 掩码生成算法

```python
def generate_action_mask(b_t, G_t_proj, confidence, unc_th=0.5):
    """
    根据信念、投影子图、置信度生成掩码
    """
    mask = [0] * (len(A_class) + len(Q_class) + len(H_class))
    
    # A类：基于信念可行性
    for i, action_a in enumerate(A_class):
        pre_satisfied = all(
            p in b_t and b_t[p].confidence >= 0.5
            for p in action_a.preconditions
        )
        if pre_satisfied:
            mask[i] = 1
    
    # Q类：基于置信度 & 不确定性
    if confidence < confidence_threshold:  # 0.78
        for i, action_q in enumerate(Q_class):
            # 低置信度时激活查询
            if action_q.targets_uncertainty_type() in get_uncertain_atoms(b_t):
                mask[len(A_class) + i] = 1
    
    # H类：基于原子链可行 & 投影子图
    for i, action_h in enumerate(H_class):
        chain_satisfied = all(
            action_a in [mask[j]==1 for j in range(len(A_class))]
            for action_a in action_h.atomic_chain
        )
        if chain_satisfied:
            mask[len(A_class) + len(Q_class) + i] = 1
    
    return mask

# RL采样时应用掩码
def rl_sample_action(pi_theta, psi_t, mask):
    q_values = pi_theta(psi_t)
    q_values[mask == 0] = -inf  # 禁止mask=0的动作
    action_idx = argmax(q_values)
    return action_list[action_idx]
```

### 5.2 置信度与动作类别的关系

```
置信度 confidence = scene_sim × precondition_completeness

confidence ≥ 0.78 (HIGH):
  ├─ 优先A类(执行) → 快速推进任务
  ├─ Q类掩码关闭 → 减少token消耗
  └─ H类激活 → 加速长链

confidence < 0.78 (LOW):
  ├─ A类部分禁用 → 避免无效执行
  ├─ Q类优先激活 → 查询关键信息
  ├─ 三层降级策略
  │  ├─ L1: probe_pre (查前提, 最高优先)
  │  ├─ L2: query_attributes/relation (查状态, 中优先)
  │  └─ L3: resolve_conflict (融合冲突, 低优先)
  └─ H类禁用 → 原子粒度控制
```

---

## 六、动作空间规模总结

| 类别 | 数量/步 | 掩码约束 | 特点 |
|-----|--------|--------|------|
| A类(执行) | 10-30 | ∀pre ∈ PRE(a), π_t(pre)≥0.5 | 动态过滤,confidence敏感 |
| Q类(查询) | 5-15 | confidence<0.78或uncertain_count>0 | 信息导向,三层优先度 |
| H类(宏序列) | 5-10 | ∃atomic_chain ⊆ A_satisfied | 长度1-5steps,efficiency导向 |
| **总计** | **20-55** | **动态掩码** | **高效探索,稀疏可行域** |

---

## 七、RL与动作空间的交互

### 7.1 状态表示 ψ_t 中的信息

```python
psi_t = {
    'embedding': emb(g),           # 任务目标嵌入 (1024-dim)
    'belief_emb': belief_emb(b_t), # 信念融合嵌入 (256-dim)
    'proj_feat': {
        'node_count': len(G_t_proj.nodes),
        'edge_count': len(G_t_proj.edges),
        'unsatisfied_pre': count_unsatisfied_in_b_t(),
        'shortest_path': compute_path_length(g, b_t),
        'fuzzy_match_score': avg(cos(emb(b_atoms), emb(g))),
        'confidence_score': confidence,  # <-关键: 影响掩码
        'query_cost': lambda_qry
    }
}
```

### 7.2 RL的学习过程

```
初始化: π_θ 从专家轨迹学 (behavioral cloning)
    ↓
每步循环:
  1. 观测o_t → LLM grounding → atoms → b_t
  2. FeasibilityChecker(b_t) → confidence, G_t_proj
  3. generate_action_mask(b_t, G_t_proj, confidence)
  4. RL采样: a_t ~ π_θ(· | ψ_t) [受mask约束]
  5. 执行/查询 → 环境反馈o_{t+1}
  6. 计算奖励: r_t = r_progress + r_efficiency - λ_qry·I{a_t∈Q}
  7. 存储轨迹 → 离线缓冲
    ↓
每100步:
  批量优化 π_θ:
    min_θ E[(π_θ(a|ψ_t) - π_expert(a|ψ_t))^2] + entropy_bonus
    其中 π_expert 来自早期HER重写轨迹
    ↓
  更新 λ_qry (自适应):
    if query_count_recent > threshold:
      λ_qry ← λ_qry + Δ  (上调惩罚)
    else:
      λ_qry ← λ_qry - Δ  (下调惩罚)
```

---

## 八、实现检查清单

- [ ] A类掩码生成：验证pre满足逻辑
- [ ] Q类掩码生成：验证置信度阈值触发
- [ ] H类掩码生成：验证原子链可行性检查
- [ ] 动作采样：确保只采样mask=1的动作
- [ ] 奖励计算：包含层次奖励(r_sub)、查询惩罚、冲突奖励
- [ ] 自适应λ_qry：监控查询频率，动态调整
- [ ] 预训练初始化：用专家轨迹初始化π_θ
- [ ] 多任务训练：在NetHack+ScienceWorld联合训练
- [ ] 性能基准：验证avg_query_count<20%、success_rate>60%@18k步

---

这个设计完全适配超图框架下的"静态本体+动态信念+约束RL"三层解耦！
