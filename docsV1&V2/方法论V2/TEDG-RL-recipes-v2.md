# TEDG-RL NetHack 菜谱式流程 v2.0（超图版）

> **核心变化**：从"静态KG导入"→ 升级为 "静态超图 + 动态信念图"的双层架构

**一句话总结**：离线构建一次不变的**任务超图本体**（所有合法动作变体），在线动态维护**情节证据图**（当前置信度），RL只在投影子图上做决策。

---

## 🏗️ 阶段 0：系统架构总览（先理解全貌）

```
┌─────────────────────────────────────────────────────────────────┐
│                        TEDG-RL 完整闭环                          │
└─────────────────────────────────────────────────────────────────┘

【离线 - 一次性】
┌───────────────────────────────────┐
│ 1. 从 NetHack 源码抽取            │
│    → CSV: 78行 (pre/eff/fail)    │
│                                   │
│ 2. 构建静态任务超图 G_T            │
│    节点: action, state_var, const │
│    超边: (a, {pre}, {eff}, cost)  │
│    存储: .pkl / .json (不再改变)  │
└───────────────────────────────────┘

【在线 - 每轮循环】
     ↓
┌──────────────────────────────────────────────────────┐
│  obs(t) ──→ LLM GroundingParser                       │
│            拆解为 atoms + sceneatoms                  │
│                                                      │
│              ↓                                        │
│  FeasibilityChecker 查询 G_T                          │
│  • 计算 scenematchconfidence                          │
│  • 如果 ≥ 0.78 → 直接命中 hyperedge                  │
│    否则 → 触发 query mode (explore/ask LLM)           │
│                                                      │
│              ↓                                        │
│  【情节证据图更新】G_E^(t)                             │
│  • 注册 obs(t) 节点                                   │
│  • 更新 π_t(v) 置信度                                 │
│  • 衰减旧观测 τ_t                                     │
│                                                      │
│              ↓                                        │
│  【RL 决策】                                          │
│  • 子图投影: G_proj ← G_T × 当前belief × mask          │
│  • DQN: state = [belief, mask], action = π_RL        │
│  • 动作执行 / query 执行                              │
│                                                      │
│              ↓                                        │
│  collect_reward(t) ──→ offline_buffer                 │
│  定期批量训练 (每 100 步)                              │
└──────────────────────────────────────────────────────┘
```

---

## 📋 阶段 1：离线构建静态超图（一次性准备）

### 1.1 数据源：NetHack 源码 CSV

**输入**：从你现有的 `nethack_extracted_core.csv` (78行)

```csv
operator, item_type, category, condition_id, predicate, args, 
logic_type, probability, consequence, source_file, source_function, 
source_line, notes, probability_source
───────────────────────────────────────────────────────────────
unlock_door, precondition, property, PRE_1, nohands, player, ...
unlock_door, precondition, property, PRE_2, door.locked, door, ...
unlock_door, effect, state_changes, EFF_1, door.unlocked=true, door, ...
...
```

**关键字段**（相比 v1.0 的新增）：
- `probability_source`：标注这个概率来自哪里
  - `rn2_code`: 直接从源码 rn2() 计算
  - `heuristic`: 从错误消息启发式推断
  - `default`: 经验值
  - `cursed_blessed_modifier`: cursed/blessed 修正链

**示例补充**：
```
unlock_door, failure_mode, no_hands, FAIL_1, nohands, player, ATOMIC, 0.10, 
"hold %s -- you have no hands!", lock.c, pick_lock, 406, 
nohands(gy.youmonst.data), default

unlock_door, precondition, property, PRE_5, random_check, rn2(100), 
ATOMIC, 0.70, success attempt, lock.c, picklock, 98, 
rn2(100) >= xlock.chance, rn2_code
```

### 1.2 构建任务超图节点集合 $V_T$

#### **1.2.1 Action 节点** (6个)

```json
{
  "actions": [
    {
      "id": "unlock_door",
      "name": "unlock_door",
      "type": "manipulation",
      "cost": 1,
      "precondition_sets": ["PRE_1", "PRE_2", ...],
      "effect_sets": ["EFF_1", "EFF_2", ...],
      "failure_modes": ["FAIL_1", "FAIL_2", ...]
    },
    {
      "id": "move",
      "name": "move",
      "type": "locomotion",
      "cost": 1,
      "precondition_sets": [...],
      ...
    },
    ...
  ]
}
```

#### **1.2.2 State Variable 节点** (~40-60个)

**玩家状态**:
```
player.nohands        # bool
player.blind          # bool
player.inventory_size # int
player.hp             # int (实际不需要 encode，只用 obs)
player.position       # (x, y) - 由 sceneatom 隐含
player.role           # enum: warrior|rogue|wizard|...
player.poly_form      # enum: human|bat|...
player.cursed_items   # set
player.blessed_items  # set
```

**对象/门/怪物**:
```
door.locked           # bool
door.broken           # bool
door.trapped          # bool
item.cursed           # bool
item.blessed          # bool
monster.type          # enum
monster.threat_level  # enum: low|medium|high
```

**场景/关系**:
```
scene.location        # enum: kitchen|dungeon|shop|...
relation.adjacent_to  # bool (player, obj)
relation.contains     # bool (container, item)
relation.blocks       # bool (obj, path)
```

#### **1.2.3 Constant 节点** (~30-50个)

```
CREDIT_CARD, LOCK_PICK, SKELETON_KEY  # 工具类型
KITCHEN, DUNGEON, SHOP, ...            # 场景类型
CURSED, BLESSED, UNCURSED             # 物品状态
ROGUE, WARRIOR, WIZARD, ...           # 角色类型
OPEN, LOCKED, BROKEN                  # 门状态
```

**合计**：$|V_T| \approx 120-150$ 节点

### 1.3 构建超边集合 $E_T^{cond}$

**核心定义**：每个 operator 的一个"变体"对应一条超边

#### **示例 1：unlock_door 的 3 条变体超边**

```json
{
  "hyperedges_conditional": [
    {
      "id": "HE_unlock_door_v1",
      "operator": "unlock_door",
      "type": "atomic_action",
      "precondition_group": {
        "spatial": ["adjacent_to(player, door)"],
        "property": ["door.locked == true", "not door.broken"],
        "possession": ["has_tool(player, LOCK_PICK)"],
        "state": ["player.nohands == false"]
      },
      "condition_logic": "AND",  // 必须全部满足
      "effect_branch": [
        {
          "name": "success",
          "probability": 0.70,  // 来自 rn2(100) >= 30
          "effects": [
            {"type": "state_change", "target": "door.locked", "value": false},
            {"type": "state_change", "target": "player.skillexp", "delta": +5},
            {"type": "enabled_operator", "operator": "open_door"},
            {"type": "event", "message": "You hear a loud click."}
          ]
        },
        {
          "name": "failure_timeout",
          "probability": 0.20,
          "effects": [
            {"type": "event", "message": "Your pick breaks."},
            {"type": "state_change", "target": "has_tool(player, LOCK_PICK)", "value": false}
          ]
        },
        {
          "name": "failure_jammed",
          "probability": 0.10,
          "effects": [
            {"type": "event", "message": "The lock resists your efforts."}
          ]
        }
      ],
      "cost": 1,
      "source": "lock.c:pick_lock"
    },

    {
      "id": "HE_unlock_door_v2",
      "operator": "unlock_door",
      "type": "atomic_action",
      "precondition_group": {
        "spatial": ["adjacent_to(player, door)"],
        "property": ["door.locked == true", "not door.broken"],
        "possession": ["has_tool(player, SKELETON_KEY)"],
        "state": ["player.nohands == false"]
      },
      "condition_logic": "AND",
      "effect_branch": [
        {
          "name": "success",
          "probability": 0.95,  // 骨架钥匙更可靠
          "effects": [
            {"type": "state_change", "target": "door.locked", "value": false},
            {"type": "enabled_operator", "operator": "open_door"}
          ]
        },
        {
          "name": "failure_wrong_key",
          "probability": 0.05,
          "effects": [
            {"type": "event", "message": "The key doesn't fit."}
          ]
        }
      ],
      "cost": 1,
      "source": "lock.c:pick_lock"
    },

    {
      "id": "HE_unlock_door_v3",
      "operator": "unlock_door",
      "type": "atomic_action",
      "precondition_group": {
        "spatial": ["adjacent_to(player, door)"],
        "property": ["door.locked == true", "not door.broken"],
        "possession": ["has_tool(player, CREDIT_CARD)"],
        "state": ["player.nohands == false"]
      },
      "condition_logic": "AND",
      "effect_branch": [
        {
          "name": "success",
          "probability": 0.30,  // 信用卡最不靠谱
          "effects": [
            {"type": "state_change", "target": "door.locked", "value": false}
          ]
        },
        {
          "name": "failure_broken",
          "probability": 0.50,
          "effects": [
            {"type": "event", "message": "The card shatters."},
            {"type": "state_change", "target": "has_tool(player, CREDIT_CARD)", "value": false}
          ]
        },
        {
          "name": "failure_generic",
          "probability": 0.20,
          "effects": []
        }
      ],
      "cost": 1,
      "source": "lock.c:pick_lock"
    }
  ]
}
```

#### **关键设计**

- ✅ **同一个 operator 多条超边**：每条对应一个"前置条件组合"（比如工具不同）
- ✅ **超边连接多个 pre 节点**：表示 AND 关系（必须都满足）
- ✅ **effect_branch 内置概率**：来自 `probability_source` 字段
- ✅ **enabled_operator**：编码"完成 unlock_door 后，open_door 变成可行"这个因果链

### 1.4 构建序列依赖超边 $E_T^{seq}$

```json
{
  "hyperedges_sequential": [
    {
      "id": "HE_seq_door_workflow",
      "type": "task_sequence",
      "nodes": ["unlock_door", "open_door"],
      "constraint": "unlock_door 必须先于 open_door",
      "weight": 1.0
    },
    {
      "id": "HE_seq_combat_workflow",
      "type": "task_sequence",
      "nodes": ["identify_monster", "cast_spell", "attack", "pickup_loot"],
      "constraint": "sequential dependency",
      "weight": 2.0
    }
  ]
}
```

### 1.5 保存静态超图

```python
# 伪代码
G_T = {
    "version": "v2.0",
    "created_at": "2025-12-02",
    "frozen": True,  # ← 关键标记：这个图永远不变
    "nodes": {
        "actions": [...],      # 6 个
        "state_vars": [...],   # ~50 个
        "constants": [...]     # ~40 个
    },
    "hyperedges": {
        "conditional": [...],  # ~25-30 条（每个 operator 变体）
        "sequential": [...]    # ~10-15 条
    },
    "embeddings": {
        # 预计算好所有节点的向量，加速后续查询
        "action_emb": {...},
        "state_var_emb": {...},
        ...
    }
}

# 保存
import pickle
with open('G_T_static.pkl', 'wb') as f:
    pickle.dump(G_T, f)
```

---

## 🕐 阶段 2：在线运行 - 每轮决策循环

### 2.1 观测解析 → Grounding Atoms

```python
# 伪代码
obs_t = game_state_observation()  
# 典型内容：
# "You are in a dark room. There is a locked door to the east. You have a lock pick."

atoms, scene_atoms = llm_grounding_parser(obs_t)
# atoms = [
#   Atom("player_at", (10, 15)),
#   Atom("has_item", "lock_pick"),
#   Atom("adjacent", "door", "player"),
#   Atom("door_locked", "door_1"),
# ]
# scene_atoms = [
#   Atom("location", "dark_room"),
#   Atom("room_type", "dungeon_room"),
# ]
```

**预期 atoms 数量**：10-20 个原子

### 2.2 可行性检查 → 信心匹配

**FeasibilityChecker** 查询静态超图 $G_T$：

```python
def feasibility_check(atoms, scene_atoms, belief_state):
    """
    返回：(matched_hyperedges, confidence_score)
    """
    
    candidates = []
    
    # 遍历所有超边（只有 ~25-30 条，非常快）
    for hyperedge in G_T["hyperedges"]["conditional"]:
        
        # 1. 匹配前置条件节点
        pre_match = all(
            atom in atoms 
            for atom in hyperedge["precondition_group"].values()
        )
        
        if not pre_match:
            continue
        
        # 2. 匹配场景节点
        scene_emb_current = embed(scene_atoms)
        scene_emb_hyperedge = G_T["embeddings"]["scene_" + hyperedge["id"]]
        scene_sim = cosine_similarity(scene_emb_current, scene_emb_hyperedge)
        
        # 3. 计算置信度
        completeness = sum(
            belief_state.get(pre, 0.0) 
            for pre in hyperedge["precondition_group"].values()
        ) / len(hyperedge["precondition_group"])
        
        confidence = scene_sim * completeness
        
        candidates.append({
            "hyperedge_id": hyperedge["id"],
            "operator": hyperedge["operator"],
            "confidence": confidence
        })
    
    # 排序
    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    
    return candidates, candidates[0]["confidence"] if candidates else 0.0
```

### 2.3 信心检验 - 分支决策

```python
confidence_threshold = 0.78  # ← Station 实测最优阈值

if confidence >= confidence_threshold:
    # 【HIGH CONFIDENCE 分支】
    #  ✅ 直接使用RL子图决策
    print(f"✅ 高置信度命中超边: {matched_hyperedges[0]['hyperedge_id']}")
    
    subgraph = project_subgraph(
        G_T, 
        matched_hyperedges[:4],  # 通常 1-4 条变体
        belief_state
    )
    
    decision_mode = "RL_FAST"  # 接下来直接RL决策
    
else:
    # 【LOW CONFIDENCE 分支】
    # ❌ 场景模糊，触发主动查询模式
    print(f"❌ 低置信度 ({confidence:.2f}), 触发查询模式")
    
    # 三层降级策略
    if can_query_property():
        # 优先级 1: 查询未知物体属性
        query_action = query(unknown_object.property)
        decision_mode = "QUERY_MODE"
        
    elif can_safe_exploration():
        # 优先级 2: 安全试探 (cast detect_monster / read scroll)
        query_action = cast_spell("detect_monster")
        decision_mode = "EXPLORE_MODE"
        
    else:
        # 优先级 3: 回到反思链
        llm_reflection = llm_step_by_step_solver(obs_t, history[-3:])
        query_action = llm_reflection
        decision_mode = "LLM_MODE"
```

### 2.4 更新情节证据图 $G_E^{(t)}$

```python
def update_evidential_hypergraph(G_E_prev, obs_t, atoms, belief_state):
    """
    动态维护信念与观测的对齐
    """
    
    G_E_t = deepcopy(G_E_prev)
    current_time = t
    
    # 1. 注册新观测节点
    for atom in atoms:
        if atom not in G_E_t["nodes"]:
            G_E_t["nodes"][atom] = {
                "type": "observed",
                "first_seen": current_time,
                "confidence": 1.0,  # 新观测完全确定
                "timestamp": current_time
            }
        else:
            # 已存在的原子，刷新置信度
            G_E_t["nodes"][atom]["timestamp"] = current_time
            G_E_t["nodes"][atom]["confidence"] = 1.0
    
    # 2. 时间衰减 - 旧观测逐步淡出
    decay_factor = 0.95
    for node_id, node in G_E_t["nodes"].items():
        age = current_time - node["timestamp"]
        node["confidence"] *= (decay_factor ** age)
        
        # 如果太旧了（置信度 < 0.01），标记为过期
        if node["confidence"] < 0.01:
            node["type"] = "stale"
    
    # 3. 多源一致性加权
    for inferred_node in G_E_t["nodes"]:
        if inferred_node["type"] == "inferred":
            # 推理节点的置信度 = ∑ (支持证据的权重)
            supporting_evidence = [
                obs_node for obs_node in atoms 
                if causally_related(obs_node, inferred_node)
            ]
            inferred_node["confidence"] = (
                sum(G_E_t["nodes"][obs]["confidence"] 
                    for obs in supporting_evidence)
                / max(len(supporting_evidence), 1)
            )
    
    # 4. 返回当前信念状态（简化为原子→置信度映射）
    belief_state = {
        node_id: node["confidence"]
        for node_id, node in G_E_t["nodes"].items()
        if node["confidence"] > 0.5 and node["type"] != "stale"
    }
    
    return G_E_t, belief_state
```

**关键数据结构**：
```python
belief_state = {
    "player_at(10, 15)": 1.0,           # 刚观测到
    "adjacent(door, player)": 1.0,
    "has_item(lock_pick)": 1.0,
    "door_locked(door_1)": 0.95,        # 1 步前观测，衰减
    "monster_nearby": 0.3,              # 3 步前推理，快淡出
}
```

### 2.5 子图投影 → 动作掩码

```python
def project_subgraph(G_T, active_hyperedges, belief_state):
    """
    从静态大超图裁剪出当前可行的子超图
    
    返回：
    - subgraph: 包含 1-4 条超边的投影子图
    - action_mask: 长度 |A|=6 的 01 向量，表示哪些动作可行
    """
    
    subgraph = {"hyperedges": [], "nodes": set()}
    action_mask = [0] * len(G_T["actions"])  # [0,0,0,0,0,0]
    
    for hyperedge in active_hyperedges:
        # 1. 检查前置条件是否都满足（belief_state ≥ 某个阈值）
        pre_satisfied = all(
            belief_state.get(pre, 0.0) >= 0.5
            for pre in hyperedge["precondition_group"].values()
        )
        
        if not pre_satisfied:
            continue  # 跳过不可行超边
        
        # 2. 添加到子图
        subgraph["hyperedges"].append(hyperedge)
        
        # 3. 设置动作掩码
        action_idx = G_T["actions"].index(
            a for a in G_T["actions"] 
            if a["id"] == hyperedge["operator"]
        )
        action_mask[action_idx] = 1
    
    return subgraph, action_mask
```

### 2.6 RL 决策 (DQN / PPO 小模型)

```python
def rl_decision(state, action_mask, model, decision_mode):
    """
    小型神经网络决策（<1M 参数）
    
    输入：
    - state: [belief_vector(50dim), subgraph_hash(10dim), goal_embedding(16dim)]
    - action_mask: [0,0,1,1,0,1] ← 只有mask=1的位置可以选
    
    输出：
    - action_id: 0-5（对应6个算子）
    """
    
    if decision_mode == "RL_FAST":
        # 标准RL前向传递
        q_values = model(state)  # 形状 (6,)
        
        # 应用掩码：不可行动作的Q值设为-∞
        q_values_masked = q_values.clone()
        q_values_masked[action_mask == 0] = -1e9
        
        action_id = q_values_masked.argmax()
        
    elif decision_mode == "QUERY_MODE":
        # 如果处于查询模式，返回特殊动作
        action_id = None
        action_token = query_action  # str类型
        
    else:
        # EXPLORE_MODE / LLM_MODE 已在上层处理
        action_id = None
        action_token = query_action
    
    return action_id, action_token
```

### 2.7 执行 & 奖励计算

```python
def execute_and_reward(action_id, obs_t, goal):
    """
    执行动作，计算奖励
    """
    
    if action_id is not None:
        # 标准动作执行
        result = game.execute_action(action_id)
        obs_next = result["observation"]
        
    else:
        # 查询 / 探索 / LLM 动作
        result = game.execute_special_action(action_token)
        obs_next = result["observation"]
    
    # 奖励函数（5个分量）
    reward = (
        w_progress * progress_reward(obs_t, obs_next, goal) +
        w_efficiency * efficiency_reward(action_id) +
        w_feasibility * feasibility_reward(was_executable) +
        w_exploration * exploration_reward(seen_before) +
        w_safety * safety_reward(no_damage_taken)
    )
    
    # 添加到离线缓冲区
    offline_buffer.append({
        "state": state_t,
        "action": action_id,
        "reward": reward,
        "next_state": state_next,
        "done": is_episode_done,
        "action_mask": action_mask
    })
    
    return obs_next, reward
```

### 2.8 批量离线训练 (每 100 步一次)

```python
def offline_dqn_update(offline_buffer, model, batch_size=32):
    """
    标准 DQN 更新（使用掩码约束）
    """
    
    batch = offline_buffer.sample(batch_size)
    
    for sample in batch:
        state, action, reward, next_state, done, action_mask = sample
        
        # DQN 损失
        q_target = model(next_state).max(dim=1)[0]
        q_target = reward + gamma * q_target * (1 - done)
        
        q_pred = model(state)[action]
        loss = (q_pred - q_target) ** 2
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 📊 完整流程时间线

```
时刻 t=0   →  观测"dark room"
              ↓
t=1         →  LLM Grounding: [player_at(10,15), adjacent(door), door_locked]
              ↓
t=2         →  查询 G_T: 匹配 HE_unlock_door_v1, confidence=0.92
              ↓
t=3         →  ✅ 高置信度! 子图投影 + RL前向 → action=unlock_door
              ↓
t=4         →  执行 unlock_door → 成功 (70% 概率) → reward=+2.0
              ↓
t=5         →  观测更新: door_locked=false, door_open_available=true
              ↓
t=6         →  G_E^(t) 融合: belief_state更新
              ↓
t=7         →  RL 看到 door 现在可打开 → 下一个动作=open_door
              ...
              
[每 100 步] →  离线批量训练一次
```

---

## 🎯 核心工作流三部曲

### 三层降级策略（当 confidence < 0.78 时）

```python
priority_order = [
    ("query_property", 0.3),      # 询问未知物体属性
    ("safe_exploration", 0.5),    # 安全试探
    ("llm_reflection", 1.0),      # 回到LLM反思链
]

for action_type, token_cost in priority_order:
    if can_perform(action_type):
        execute(action_type)
        break
```

### 预期效果

| 指标 | 值 | 说明 |
|-----|-----|------|
| 单步决策延迟 | <40ms | 子图查询 + RL 前向无 LLM 调用 |
| Token 消耗/步 | ~0-10 | 仅在低置信度时调用 LLM |
| 长序列成功率 (18k 步) | >60% | 因为没有在线学习风险 |
| 样本效率 | ~1000 轮 | 相比纯 RL 快 100 倍 |
| 超参数敏感性 | 低 | 只需调 confidence_threshold |

---

## 📝 Method 一句话（论文版）

> We propose TEDG-RL, a neuro-symbolic approach that decouples action models, evidential beliefs, and policies by maintaining a **static Task Hypergraph** extracted offline from NetHack source code (78 conditional rules with probability calibration), an **Episodic Evidential Graph** updated online via Bayesian fusion and temporal decay, and an **RL policy constrained to feasible subgraphs**, enabling long-horizon decision-making (18k+ steps) with <40ms latency and zero online learning risk compared to prior dynamic-model methods.

---

## 📁 完整文件清单

```
离线阶段：
├─ nethack_extracted_core.csv           (数据源，78行)
├─ G_T_static.pkl                       (静态超图序列化)
├─ G_T_schema.json                      (超图结构定义)
└─ embeddings_precomputed.pkl           (节点向量缓存)

在线阶段：
├─ llm_grounding_parser.py              (观测 → atoms)
├─ feasibility_checker.py               (查询 G_T)
├─ evidential_hypergraph.py             (G_E^(t) 维护)
├─ subgraph_projector.py                (投影 & 掩码生成)
├─ dqn_small_model.pth                  (RL模型 <1M 参数)
├─ offline_buffer.py                    (轨迹缓冲)
└─ nethack_interface.py                 (游戏交互)

```

---

这就是完整的菜谱 v2.0！相比 v1.0 的核心升级：
✅ 从"KG导入"到"超图查询"（在线灵活）
✅ 从"静态信念"到"动态融合"（观测可衰减）
✅ 从"RL全空间"到"RL掩码子图"（约束可行）
✅ 从"一路LLM"到"三层降级"（智能回退）
