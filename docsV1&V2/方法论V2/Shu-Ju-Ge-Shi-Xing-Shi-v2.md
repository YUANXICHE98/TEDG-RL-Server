# TEDG-RL v2.0 数据格式与流转规范

> **版本升级背景**：从纯KG知识图形式升级到**超图本体 + 信念图 + 投影融合**的多层表示。每一层都有明确的数据格式定义，确保模块间信息一致性和可追溯性。

---

## 第一层：任务超图 $G_T$（离线静态，不变）

### 定义

任务超图由源代码规则静态提取，包含所有可执行动作及其条件分支的显式编码。

### 格式规范

```yaml
# task_hypergraph.yaml

hypergraph:
  name: "NetHack 3.6.7 Action Ontology"
  version: "v2.0"
  extraction_date: "2025-12-02"
  source: "NetHack source code + Guidebook"
  
  nodes:
    count: 127
    predicates:  # 所有可能的一阶谓词
      - move(entity, direction)         # 移动
      - adjacent(entity1, entity2)      # 相邻
      - holding(entity, object)         # 持有
      - on_floor(object, location)      # 地面
      - blocked(direction)              # 阻挡
      - ison(device)                    # 设备开启
      - equipped(object)                # 装备
      - monster.hp                      # 数值属性
      - player.hunger                   # 数值属性
      # ... ~120个谓词
  
  hyperedges:
    count: 28  # 6个核心算子 × 多个条件变体
    
    # 示例 1: move动作 (8个变体 = 8个方向)
    HE_move_north:
      operator: "move"
      variant: "direction=north"
      direction_value: "north"
      preconditions:
        - blocked(north) == False          # 必须
        - within_bounds(north) == True     # 必须
      effects:
        - player.position ← next_pos(north)  # 确定性
      cost: 1
      failure_modes: []  # move总是成功（如果前提满足）
    
    HE_move_south:
      operator: "move"
      variant: "direction=south"
      direction_value: "south"
      preconditions:
        - blocked(south) == False
        - within_bounds(south) == True
      effects:
        - player.position ← next_pos(south)
      cost: 1
      failure_modes: []
    
    # ... (HE_move_east, HE_move_west, HE_move_up, HE_move_down, HE_move_ne, HE_move_nw, HE_move_se, HE_move_sw)
    
    # 示例 2: heat动作 (2个变体 = 成功/失败分支)
    HE_heat_success:
      operator: "heat"
      variant: "outcome=success"
      preconditions:
        - holding(item) == True
        - ison(device) == True
        - item.temperature < 50
      effects:
        - item.temperature ← 100       # 确定性效果
        - player.mana ↓ 5              # 可选副作用
      cost: 2
      failure_modes: []
    
    HE_heat_device_off:
      operator: "heat"
      variant: "outcome=device_off"
      preconditions:
        - holding(item) == True
        - ison(device) == False        # 与上一变体互斥
      effects:
        - None  # 无效动作，浪费一回合
      cost: 1
      failure_modes:
        - pre_violation: "ison(device)==False"
        - recovery_action: "turnon(device)"
    
    # 示例 3: throw动作 (多变体覆盖目标种类)
    HE_throw_monster:
      operator: "throw"
      variant: "target_type=monster"
      preconditions:
        - holding(projectile) == True
        - adjacent(monster) == True
      effects:
        - monster.hp ↓ damage(projectile)
        - projectile.location ← floor
      cost: 1
      failure_modes: []
    
    HE_throw_trash:
      operator: "throw"
      variant: "target_type=trash"
      preconditions:
        - holding(object) == True
        - adjacent(trash_bin) == True
      effects:
        - object.location ← trash_bin
        - inventory.count ↓ 1
      cost: 1
      failure_modes: []
    
    HE_throw_altar:
      operator: "throw"
      variant: "target_type=altar"
      preconditions:
        - holding(object) == True
        - adjacent(altar) == True
      effects:
        - object ← offered_to(altar)
        - divine_effect ← triggered()
      cost: 1
      failure_modes:
        - pre_violation: "nearby_god_status!=neutral"
    
    # ... (共28条超边)

# 注：
# - 每条超边对应一个明确的可执行"脚本"，有确定的pre/eff/fail
# - 多个变体对应**条件分支**而非选择分支：RL在知道precondition后才决定用哪个变体
# - 此超图在游戏开始时一次性加载，之后只读
```

### 关键设计

1. **分支编码**：每个operator可有多个HE_*变体（通过variant字段区分），对应不同precondition或target
2. **失败处理**：显式列出failure_modes及recovery_action，为三层降级提供线索
3. **成本编码**：用cost表示action执行的token等价消耗（便于λ_qry计算）
4. **静态性**：一旦构建，不再改变（除非游戏更新），避免在线学习的复杂性

---

## 第二层：情节证据图 $G_E^{(t)}$（在线动态，时序演化）

### 定义

在游戏运行时，每次观测都产生新的原子事实，写入证据图。同时旧事实随时间衰减。融合后得到当前信念状态。

### 格式规范

```yaml
# episodic_evidence_graph.yaml (运行时生成，每步更新)

episodic_graph:
  timestamp: 2025120200120345  # epoch milliseconds
  episode_id: "ep_001_nethack_dlvl_1"
  max_age: 500  # 超过500步的原子自动清理
  
  atoms:
    count: 187  # 当前活跃原子数
    
    # 原子格式: (entity, relation, value) + 元数据
    
    # 类型1：直接观测（来自LLM Grounding）
    - id: "atom_001"
      timestamp: 2025120200120345
      age: 0
      entity: "apple"
      relation: "location"
      value: "counter"
      confidence: 1.0          # 直接观测，完全信任
      source: "visual"         # 来源类型
      llm_grounding: {
        raw_text: "You see an apple on the counter.",
        extraction_confidence: 0.95
      }
      embedding: [0.234, -0.156, ..., 0.089]  # 1024-dim嵌入，用于模糊匹配
    
    - id: "atom_002"
      timestamp: 2025120200120300  # 上一步观测
      age: 45
      entity: "microwave"
      relation: "ison"
      value: False
      confidence: 1.0 * (0.95 ** 45) = 0.105  # 已衰减
      source: "visual"
      embedding: [0.145, 0.267, ..., -0.078]
    
    # 类型2：推理原子（来自eff融合或查询）
    - id: "atom_003"
      timestamp: 2025120200120345
      age: 0
      entity: "apple"
      relation: "held_by"
      value: "player"
      confidence: 1.0           # 来自前一步的执行反馈
      source: "effect"          # 来自action effect
      supporting_atoms: [atom_001]  # 此原子依赖的观测
      reasoning_rule: "pickup(apple) ⟹ held_by(player)"
      embedding: [0.189, -0.234, ..., 0.156]
    
    # 类型3：推理原子（多源融合）
    - id: "atom_004"
      timestamp: 2025120200120200  # 最后更新时间
      age: 145
      entity: "player"
      relation: "hunger_level"
      value: "starving"  # 融合多个观测的最终值
      confidence: 0.82   # avg(supporting atoms' confidence after decay)
      source: "fusion"
      supporting_atoms: [atom_XXX, atom_YYY]  # 多个证据
      fusion_rule: "avg_confidence([obs1, obs2, ...])"
      embedding: [0.267, 0.089, ..., -0.145]
    
    # 类型4：冲突原子（待消解）
    - id: "atom_005_conflict_marker"
      timestamp: 2025120200120340
      age: 5
      entity: "apple"
      relation: "temperature"
      value: CONFLICT  # 特殊标记
      conflict_values: ["hot", "cold", "room_temp"]  # 多个候选
      conflict_sources: ["llm_query", "prev_effect", "player_input"]
      confidence: 0.0  # 冲突时置0，待resolve_conflict查询
      embedding: None  # 冲突原子不计算嵌入
      # 待消解：下一步应执行 query_whatis(apple) 或 query_attributes(apple)
  
  hyperedges:
    # 原子间的支撑关系（用于逆向推理）
    count: 89
    
    - source: "atom_003"          # apple held_by player
      target: "atom_001"          # apple location counter
      relation: "supports"        # 直接推理关系
      confidence: 0.95
      rule_id: "rule_pickup_implies_held"
    
    - source: ["atom_XXX", "atom_YYY"]  # 多源融合
      target: "atom_004"
      relation: "converges_to"
      confidence: 0.82
      fusion_method: "average_confidence"

  # 时间衰减与清理策略
  decay_config:
    factor: 0.95
    halflife_steps: 14  # ~14步后，confidence降到50%
    cleanup_age_threshold: 500
    cleanup_confidence_threshold: 0.01

# 说明：
# - 原子持续演化，新观测立即加入，旧观测按指数衰减
# - 冲突原子需要显式resolve_conflict查询
# - 推理原子的confidence = avg(supporting_atoms)，会继承衰减
# - 嵌入向量用于后续模糊匹配（resolve冲突/fusion加权）
```

### 融合逻辑

```python
def update_episodic_graph(G_E_t, new_atoms, current_step):
    """
    将新观测加入证据图，衰减旧观测
    """
    # 1. 衰减所有旧原子
    for atom in G_E_t.atoms:
        age_delta = current_step - atom.timestamp
        atom.confidence *= 0.95 ** age_delta
        atom.age = age_delta
    
    # 2. 添加新原子
    for new_atom in new_atoms:
        existing = G_E_t.find_by_predicate(new_atom.entity, new_atom.relation)
        if existing:
            # 冲突检测
            if existing.value != new_atom.value:
                # 创建冲突标记
                conflict_atom = create_conflict_marker(
                    predicate=(new_atom.entity, new_atom.relation),
                    old_value=existing.value,
                    new_value=new_atom.value,
                    sources=[existing.source, new_atom.source]
                )
                G_E_t.add_atom(conflict_atom)
                # 旧原子标记为待验证
                existing.status = "pending_resolution"
            else:
                # 相同值，更新置信度（融合）
                existing.confidence = max(existing.confidence, 1.0)
                existing.timestamp = current_step
        else:
            # 新谓词，直接加入
            G_E_t.add_atom(new_atom)
    
    # 3. 清理过期原子
    G_E_t.cleanup_expired_atoms(age_threshold=500, conf_threshold=0.01)
    
    return G_E_t
```

---

## 第三层：信念状态 $b_t$（投影融合，实时更新）

### 定义

从$G_E^{(t)}$融合得到的"最佳估计"，用于FeasibilityChecker和RL决策。是一个简化的字典，每个谓词对应最高置信度的值。

### 格式规范

```yaml
# belief_state.yaml (实时，用于决策)

belief_state:
  timestamp: 2025120200120345
  source_graph: "G_E_t (atom_count=187)"
  fusion_method: "max_confidence_with_decay"
  
  # 谓词→最佳值的映射
  facts:
    # 实体位置（最常查询）
    apple.location: 
      value: "counter"
      confidence: 1.0
      source: "visual"
      age: 0
      supporting_atoms: [atom_001]
    
    player.position:
      value: [42, 15]  # grid coordinate
      confidence: 0.98
      source: "fusion"  # 多个传感器融合
      age: 2
      supporting_atoms: [atom_XXX, atom_YYY, atom_ZZZ]
    
    microwave.ison:
      value: False
      confidence: 0.105  # 衰减了45步
      source: "visual"
      age: 45
      supporting_atoms: [atom_002]
      recommendation: "QUERY_RECOMMENDED"  # 置信度<0.5时提示查询
    
    apple.temperature:
      value: CONFLICT
      confidence: 0.0
      source: "conflict"
      age: 5
      conflicting_values: ["hot", "cold"]
      conflicting_sources: ["effect", "visual"]
      resolution_action: "query_attributes(apple)"  # 建议的解决方案
    
    player.hunger_level:
      value: "starving"
      confidence: 0.82
      source: "fusion"
      age: 0
      supporting_atoms: [multiple sources]
    
    # ... (~120+个谓词)
  
  # 摘要统计（便于RL特征提取）
  summary:
    total_predicates: 127
    known_predicates: 89         # confidence >= 0.5
    uncertain_predicates: 23     # 0.1 <= confidence < 0.5
    conflicted_predicates: 4     # CONFLICT值
    unknown_predicates: 11       # confidence < 0.1
    
    average_confidence: 0.74
    oldest_fact_age: 145
    newest_fact_age: 0
    
    fuzzy_match_score: 0.91      # 与任务目标的语义相似度
    
    # 关键指标：置信度分布
    confidence_percentiles:
      p10: 0.32
      p50: 0.85
      p90: 0.99
      p99: 1.00

# 说明：
# - 此结构是$G_E^{(t)}$的**完全投影**，仅保留最高置信的值
# - 用于FeasibilityChecker检查前提是否满足
# - 用于RL编码状态特征（belief_emb）
# - conflict和uncertain谓词会提示RL优先查询
```

### 投影规则

```python
def project_to_belief_state(G_E_t, threshold=0.5):
    """
    从证据图投影到简洁的信念状态
    """
    belief = {}
    
    for predicate in G_E_t.all_predicates():
        # 同一谓词可能有多个原子值（冲突）
        atoms = G_E_t.find_atoms_by_predicate(predicate)
        
        # 按confidence排序
        atoms.sort(key=lambda a: a.confidence, reverse=True)
        
        if atoms:
            best_atom = atoms[0]
            
            if len(atoms) > 1 and atoms[0].confidence - atoms[1].confidence < 0.1:
                # 前两个置信度接近 → 冲突
                belief[predicate] = {
                    'value': CONFLICT,
                    'confidence': 0.0,
                    'conflicting_values': [a.value for a in atoms[:3]]
                }
            else:
                # 明确的值
                belief[predicate] = {
                    'value': best_atom.value,
                    'confidence': best_atom.confidence,
                    'source': best_atom.source
                }
        else:
            # 无信息
            belief[predicate] = {
                'value': UNKNOWN,
                'confidence': 0.0
            }
    
    return belief
```

---

## 第四层：投影子图 $G_t^{\text{proj}}$（实时）

### 定义

从$G_T$中选择"当前可行"的超边子集，形成投影子图。用于RL决策的上下文限制。

### 格式规范

```yaml
# projected_subgraph.yaml

projected_subgraph:
  timestamp: 2025120200120345
  source: "G_T"
  belief_state: "b_t"
  
  # 可行的超边（通过前提检查）
  feasible_hyperedges:
    count: 12  # 总28条中只有12条前提满足
    
    edges:
      - hyperedge_id: HE_move_north
        confidence: 0.98  # 前提满足度（soft score）
        unsatisfied_preconditions: []
        constraint_strength: "hard"  # 前提全满足
      
      - hyperedge_id: HE_move_south
        confidence: 0.98
        unsatisfied_preconditions: []
      
      # HE_move_east-west: similarly feasible
      
      - hyperedge_id: HE_heat_success
        confidence: 0.52  # 软分数：部分前提不确定
        unsatisfied_preconditions: 
          - predicate: "ison(microwave)"
            actual_confidence: 0.105  # 因为太旧了
        constraint_strength: "soft"  # 风险：可能失败
        recommendation: "QUERY_FIRST"
      
      # HE_heat_device_off: 前提互斥，不可行
      
      - hyperedge_id: HE_throw_trash
        confidence: 0.91
        unsatisfied_preconditions: []
        constraint_strength: "hard"
  
  # 不可行的超边（前提不满足）
  infeasible_hyperedges:
    count: 16
    
    edges:
      - hyperedge_id: HE_heat_device_off
        reason: "pre_contradiction"
        conflicting_pre: "ison(device)==False"
        actual_value: "unknown (conflict)"
        confidence: 0.0
      
      - hyperedge_id: HE_sacrifice_ritual
        reason: "unsatisfied_precondition"
        missing_pre: ["adjacent(altar)"]
        confidence: 0.0

  # 摘要特征（用于RL投影特征提取）
  features:
    total_hyperedges: 28
    feasible_count: 12
    soft_feasible_count: 3  # confidence < 0.5但>0
    infeasible_count: 13
    
    avg_confidence: 0.78          # 可行超边的平均置信度
    median_unsatisfied_pre_count: 0
    max_unsatisfied_pre_count: 2
    
    critical_missing_predicates:
      - "ison(microwave)"  # 多个超边缺这个
      - "adjacent(monster)"
    
    # 关键指标：动作掩码
    action_mask: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, ..., 0]
    # mask[i] = 1 if hyperedge_i is feasible, 0 else
```

---

## 第五层：RL状态编码 $\psi_t$（实时）

### 定义

将上述多层信息融合为RL的输入向量，便于神经网络决策。

### 格式规范

```python
# rl_state_encoding.py

class RLStateEncoding:
    """
    混合状态表示：符号特征 + 嵌入向量
    """
    
    def __init__(self, goal, belief_state, projected_subgraph):
        # ======== 嵌入层 ========
        self.goal_embedding = embed_nlp(goal)          # 1024-dim
        self.belief_embedding = self.encode_belief(belief_state)  # 256-dim
        
        # ======== 统计特征 ========
        self.subgraph_features = self.extract_subgraph_features(projected_subgraph)
        # 包含:
        #   - node_count: int
        #   - edge_count: int
        #   - unsatisfied_pre_count: int
        #   - shortest_path_to_goal: int
        #   - feasible_ratio: float (0-1)
        
        # ======== 信念质量指标 ========
        self.confidence_score = belief_state['summary']['average_confidence']  # 0-1
        self.uncertainty_mask = self.get_uncertain_predicates(belief_state)   # bool array
        self.conflict_count = belief_state['summary']['conflicted_predicates']
        
        # ======== 模糊匹配得分 ========
        self.fuzzy_match_score = belief_state['summary']['fuzzy_match_score']  # 0-1
        
        # ======== 动作掩码 ========
        self.action_mask = projected_subgraph['features']['action_mask']  # [0/1]×28
    
    @property
    def as_tensor(self):
        """转为RL输入张量"""
        return torch.cat([
            self.goal_embedding,                    # 1024
            self.belief_embedding,                  # 256
            torch.tensor(self.subgraph_features),   # ~10
            torch.tensor([
                self.confidence_score,              # 1
                self.fuzzy_match_score,             # 1
                self.conflict_count / 127,          # 1 (normalized)
            ]),                                     # 3
            torch.tensor(self.action_mask, dtype=torch.float32)  # 28
        ])  # Total: 1024 + 256 + 10 + 3 + 28 = 1321 dim
    
    def encode_belief(self, belief_state):
        """
        将信念状态压缩为256-dim嵌入
        """
        # 策略：对所有known predicates的嵌入求加权平均
        embeddings = []
        weights = []
        
        for pred, fact in belief_state['facts'].items():
            if fact['confidence'] >= 0.1:  # 忽略极低置信度
                emb = self.embed_fact(pred, fact)  # 256-dim
                embeddings.append(emb)
                weights.append(fact['confidence'])
        
        # 加权平均
        belief_emb = sum(e * w for e, w in zip(embeddings, weights)) / sum(weights)
        return belief_emb
    
    @staticmethod
    def extract_subgraph_features(proj_subgraph):
        """提取投影子图的统计特征"""
        return {
            'node_count': len(proj_subgraph.nodes),
            'edge_count': len(proj_subgraph.feasible_edges),
            'unsatisfied_pre_count': sum(
                len(e['unsatisfied_preconditions']) 
                for e in proj_subgraph.feasible_edges
            ),
            'shortest_path': compute_shortest_path(proj_subgraph),
            'feasible_ratio': len(proj_subgraph.feasible_edges) / len(proj_subgraph.all_edges)
        }

# 使用示例：
goal = "Heat apple and throw to trash"
belief = project_to_belief_state(G_E_t)
subgraph = project_subgraph(G_T, belief)

psi_t = RLStateEncoding(goal, belief, subgraph)
q_values = dqn_model(psi_t.as_tensor)  # DQN前向推理
```

---

## 数据流转全景图

```
源代码 (NetHack source)
    ↓
[Rule Extraction] (离线一次)
    ↓
G_T: Task Hypergraph (28条超边, 127个节点, 静态)
    ↓
    ├─→ [每步]
    │   ├─ obs_t (raw text) 
    │   │   ↓
    │   │ [LLM Grounding]
    │   │   ↓
    │   │ atoms_t, sceneatoms_t
    │   │   ↓
    │   │ [FeasibilityChecker] (查询G_T + b_t)
    │   │   ↓
    │   │ candidates, confidence
    │   │   ├─ 高置信 (≥0.78)
    │   │   │   ↓
    │   │   │ [Subgraph Projection] (快速路线)
    │   │   │   ↓
    │   │   │ G_t^proj, action_mask
    │   │   │
    │   │   └─ 低置信 (<0.78)
    │   │       ↓
    │   │       [Three-Tier Fallback]
    │   │       (query/explore/llm)
    │   │
    │   ├─→ [Evidential Fusion]
    │   │   G_E^(t) ← update with atoms_t
    │   │   (衰减旧原子, 融合新原子)
    │   │
    │   ├─→ [Belief Projection]
    │   │   b_t ← project(G_E^(t))
    │   │
    │   ├─→ [RL Encoding]
    │   │   ψ_t ← encode(goal, b_t, G_t^proj)
    │   │
    │   ├─→ [RL Decision]
    │   │   a_t ~ π_θ(ψ_t) [受action_mask约束]
    │   │
    │   └─→ [Execute & Store]
    │       experience ← (ψ_t, a_t, r_t, ψ_{t+1})
    │       离线回放缓冲: 每100步训练一次
    │
    └─ [End of Episode]
        评估: success_rate, token_count, query_count

每步时间分解:
  - LLM Grounding: ~200ms (大部分时间)
  - FeasibilityChecker: ~10ms
  - 高置信快速路线: ~40ms (projection + RL)
  - 低置信降级: ~300ms (额外查询)
```

---

## 检查清单与一致性保证

### 数据流转检查

- [ ] G_T加载时验证：所有超边的pre都在谓词集合P中
- [ ] 原子写入时验证：entity, relation都在谓词定义中
- [ ] 信念投影时验证：无遗漏谓词，conflict标记准确
- [ ] 子图投影时验证：可行动作集为全集的子集
- [ ] RL编码时验证：action_mask与可行动作一一对应

### 衰减与融合一致性

- [ ] 衰减因子统一：所有地方都用0.95^age
- [ ] 融合方法统一：avg(confidence)用于推理原子
- [ ] conflict消解：标记为CONFLICT且confidence=0.0
- [ ] 置信度范围：总在[0, 1]，conflict为特殊情况

### 格式兼容性

- [ ] YAML能序列化所有数据结构
- [ ] 嵌入向量维度一致（1024/256）
- [ ] action_mask维度与超边数匹配（28）
- [ ] 时间戳格式统一（epoch ms）

---

这个多层数据格式设计支撑了TEDG-RL v2.0的三层解耦架构，确保了**模块间信息一致性**和**长序列状态跟踪的可靠性**。

