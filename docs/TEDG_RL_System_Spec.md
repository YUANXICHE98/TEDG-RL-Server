# TEDG-RL-System Implementation Specification

## Project Overview
A hierarchical game AI system combining:
- **Static Task Hypergraph (G_T)**: 450 hyperedges encoding NetHack rules
- **Multi-Channel Embedding**: 4 parallel information pathways (pre-conditions, scene atoms, effects, rules)
- **RL Decision Layer**: PPO-based policy learning with action masking
- **LLM Grounding**: BERT/LLM for observation → symbolic atom translation

---

## Part 1: RL Algorithm & Training Specification

### 1.1 Algorithm Selection
**Primary Algorithm: PPO (Proximal Policy Optimization)**
- Action masking support for hypergraph constraints
- Sample efficiency suitable for game environments
- Stable convergence with monotonic improvement guarantee
- Multi-channel architecture compatibility

### 1.2 Hyperparameter Configuration
| Parameter | Value | Note |
|-----------|-------|------|
| Learning Rate | 3e-4 | Adjust if loss diverges |
| Clip Range | 0.2 | PPO epsilon clipping |
| GAE Lambda | 0.95 | Generalized Advantage Estimation |
| PPO Epochs | 3-5 | Updates per batch |
| Batch Size | 128-256 | Per mini-batch |
| Discount γ | 0.99 | Future reward weight |
| Entropy Bonus | 0.01 | Exploration regularization |
| Gradient Clip | 0.5 | Per-layer gradient norm clipping |

### 1.3 Network Architecture
```
4 Independent Actor Networks:
├─ actor_pre:   Input [q_pre(15) + belief_context(20)] → Output logits[6-10 actions]
├─ actor_scene: Input [q_scene(15) + location_context(20)] → Output logits[6-10 actions]
├─ actor_effect: Input [q_effect(8) + hp_context(10)] → Output logits[6-10 actions]
└─ actor_rule:  Input [q_rule(10) + inventory_context(15)] → Output logits[6-10 actions]

Attention Fusion Network:
└─ AttentionWeightNet: Input state(115 dim) → Output α weights [4 dim] via softmax

Critic Network (Shared):
└─ ValueNet: Input state(115 dim) → Output value scalar

Fusion Operation:
└─ fused_logits = sum(α_i * logits_i for i in {pre,scene,effect,rule})
```

### 1.4 Training Stages

#### Stage 1: Initialization
1. Initialize 4 actor networks with Xavier initialization
2. Initialize α weights uniformly: α = [0.25, 0.25, 0.25, 0.25]
3. Initialize critic to predict 0 (unbiased)

#### Stage 2: Data Collection
1. Run game episodes, collect (state_t, action_t, reward_t, state_{t+1}, done_t)
2. State structure: [belief(50), q_pre(15), q_scene(15), q_effect(8), q_rule(10), conf(1), goal(16)] = 115 dim
3. Record episode returns G_t and compute GAE advantages A_t = G_t - V(s_t)

#### Stage 3: Model Update
```
Forward Pass:
  - Compute logits from 4 independent actors
  - Compute α weights from AttentionWeightNet
  - Fuse: fused_logits = sum(α_i * logits_i)
  - Compute value V(s_t)

Backward Pass:
  - Actor loss: L_actor = -E[log π(a|s) * A_t]
  - Critic loss: L_critic = E[(V_pred - G_t)^2]
  - PPO clip: importance_ratio = π_new / π_old
  - Clipped loss: L_ppo = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
  - Total loss: L = L_actor + 0.5*L_critic + entropy_bonus
  - Backprop to all networks
```

#### Stage 4: Evaluation
Every N episodes (recommended N=100):
- Compute mean episode return
- Compute mean episode length
- Track query action frequency (target: < 30%)
- Monitor α weight distribution (check for pathological concentration)

#### Stage 5: Hyperparameter Tuning
- Loss diverging? → Reduce critic learning rate
- Policy changing too rapidly? → Increase clip range to 0.3
- Insufficient exploration? → Increase entropy bonus
- Gradient explosion in one channel? → Apply per-network gradient clipping

---

## Part 2: RL Learning Triple (State, Action, Reward)

### 2.1 State Representation

**Dimension: 115 = [50 + 15 + 15 + 8 + 10 + 1 + 16]**

```
state = [
  belief_vector(50),      # From evidence graph G_E^(t): hp levels, hunger, monsters, dlvl, room type
  q_pre(15),              # Multi-channel embedding: pre-conditions pathway
  q_scene(15),            # Multi-channel embedding: scene atoms pathway
  q_effect(8),            # Multi-channel embedding: effects/risk pathway
  q_rule(10),             # Multi-channel embedding: rule patterns pathway
  confidence(1),          # Hypergraph match confidence [0, 1]
  goal_embedding(16)      # LLM-identified mid-term goal embedding
]
```

**Information Preservation: 95% (vs 70% in single-channel design)**

### 2.2 Action Space

#### High Confidence Branch (confidence ≥ 0.78)
- **Action Count**: 6-10 game actions
- **Source**: Hypergraph applicable_operators(state)
- **Examples**: move, eat, zap, wait, unlock_door, open_door
- **RL Task**: Learn priority ordering among masked actions
- **Token Cost**: 0 (pure RL, microsecond inference)

#### Low Confidence Branch (confidence < 0.78)
| Action | Description | Token Cost | Source |
|--------|-------------|-----------|--------|
| query_property | Ask LLM: "What is this object?" | 100-500 | LLM |
| safe_exploration | Execute hypergraph-marked safe detection | 0 | Game + Hypergraph |
| llm_reflection | Send full scenario to LLM for reasoning chain | 1000+ | LLM |
| wait | Pause, let environment provide information | 1 step | Game |

**Action Masking**: Hypergraph permanently masks physically infeasible actions with -inf logits

### 2.3 Reward Design

**Formula**:
```
r(t) = w_p * r_progress(t) 
     + w_s * r_safety(t) 
     + w_e * r_efficiency(t) 
     + w_f * r_feasibility(t) 
     + w_x * r_exploration(t)
```

#### Reward Components

| Component | Weight | Range | Formula |
|-----------|--------|-------|---------|
| r_progress | 0.3 | [-1, +1] | dlvl_change + task_completion_bonus |
| r_safety | 0.3 | [-1000, +0.1] | -1000*death - 100*hp_critical - 0.1*(1-safety_score) |
| r_efficiency | 0.2 | [-0.1, +0.1] | -steps/1000 + speed_bonus |
| r_feasibility | 0.1 | [-0.3, +0.05] | -0.2*precond_violation - 0.3*failure_mode_trigger + 0.05*normal |
| r_exploration | 0.1 | [0, +0.15] | 0.15*new_conditional_effect + 0.1*new_room |

#### Gradient Flow to Channels
- r_progress → All channels (global)
- r_safety → Mainly q_effect, also q_pre
- r_efficiency → All channels (global)
- r_feasibility → Mainly q_pre and q_rule
- r_exploration → Mainly q_scene

---

## Part 3: Hypergraph Embedding Specification

### 3.1 Four Embedding Pathways

#### Pathway 1: Pre-Conditions (q_pre, 15 dim)
```
Input Nodes (example):
  - has_gold, hunger_normal, hp_full, power_empty, confused, blind, stunned
  - Total: ~30 distinct pre-condition nodes

Encoding: HGNN Two-Stage Aggregation
  Stage 1 (Node→Hyperedge): Aggregate pre-condition compatibility
    - Learn: which conditions co-occur in feasible operations
    - Example: has_gold + hp_full more meaningful than has_gold alone
  Stage 2 (Hyperedge→Feature): Feedback to feature vector
    - Output: q_pre ∈ R^15, semantic vector
    - Captures: sufficiency and compatibility of current pre-conditions

RL Usage: actor_pre reads q_pre → "Are preconditions satisfied?"
```

#### Pathway 2: Scene Atoms (q_scene, 15 dim)
```
Input Atoms (example):
  - dlvl_1 to dlvl_36 (depth level), in_shop, in_altar, near_gold_vault
  - monsters_present, ac_poor/good, in_room
  - Total: ~40 distinct scene atoms

Encoding: HGNN Two-Stage Aggregation
  Stage 1: Aggregate scene atom relationships
    - Learn: which atoms are compatible (e.g., in_shop + monsters_present = anomaly)
  Stage 2: Feedback to feature vector
    - Output: q_scene ∈ R^15, semantic vector
    - Captures: environmental characteristics and anomalies

RL Usage: actor_scene reads q_scene → "Is environment safe for this action?"
```

#### Pathway 3: Effects & Risk (q_effect, 8 dim)
```
Input Data:
  - eff_nodes: [ate_food, hunger_satisfied, hit, combat_success, ...]
  - success_probability: 0.0-1.0 (aggregated from 450 hyperedges)
  - safety_score: 0.0-1.0 (computed from failure_modes)
  - failure_modes: {precond_violation: count, bad_aim: count, ...}

Encoding: MLP + Dense Layers
  Step 1: Embed eff_nodes to vectors
  Step 2: MLP processes [eff_embedding, success_prob, safety_score, failure_counts]
    - Hidden: ReLU(64 dim)
    - Output: q_effect ∈ R^8
    - Captures: success-safety trade-off in single vector

RL Usage: actor_effect reads q_effect → "Is risk-reward balance acceptable?"
```

#### Pathway 4: Rule Patterns (q_rule, 10 dim)
```
Input Data:
  - conditional_effects: [if item.blessed then got_blessed, ...]
  - failure_mode type classifications
  - rule_patterns: 50 encoded NetHack rules

Encoding: RuleEncoder + Embedding
  Step 1: Symbolic encoding of if-then relationships
    - Build: condition-node + effect-node + directed edge graph
  Step 2: RuleEncoder processes graph (permutation-invariant)
    - Input: conditional_effect dependency graph
    - Output: q_rule ∈ R^10
    - Captures: hidden mechanisms and pitfalls

RL Usage: actor_rule reads q_rule → "Are there hidden traps or effects?"
```

### 3.2 Fusion Layer (Multi-Head Attention)
```
Input: 4 pathway embeddings [q_pre(15), q_scene(15), q_effect(8), q_rule(10)]

Processing:
  1. Concatenate: [q_pre, q_scene, q_effect, q_rule] → 48 dim
  2. AttentionWeightNet: 48+67 (other state) → 4 dim logits
     - Hidden: ReLU(64 dim)
     - Output: logits_α [4 dim]
  3. Softmax: α = softmax(logits_α) ∈ [0,1]^4, sum(α)=1
  4. Weighted Fusion: q_fused = Σ α_i * normalize(q_i)
  5. Output: q_fused ∈ R^20

Semantic Meaning of α:
  - α_pre high: "Current preconditions are critical to decide"
  - α_scene high: "Environmental understanding is key"
  - α_effect high: "Risk-reward assessment is uncertain"
  - α_rule high: "Hidden effects/traps are the concern"
```

### 3.3 Complete Embedding Pipeline
```
Input from LLM Grounding + Game Observation:
  - pre_nodes: boolean/one-hot indicators
  - scene_atoms: boolean/one-hot indicators
  - eff_metadata: {success_prob, safety_score, failure_modes, ...}
  - conditional_effects: if-then rule list

Processing (Parallel Pathways):
  pre_nodes ──→ HGNN ──→ q_pre(15)
  scene_atoms ─→ HGNN ──→ q_scene(15)
  eff_metadata → MLP ──→ q_effect(8)
  conditional_effects → RuleEncoder → q_rule(10)

                    ↓ (All pathways feed into)

             AttentionWeightNet
                  ↓
             α = softmax(...)
                  ↓
             Weighted Fusion
                  ↓
             q_fused(20)

Final State Construction:
  state = [belief_vector(50), q_pre(15), q_scene(15), q_effect(8), q_rule(10), confidence(1), goal_embedding(16)]
  Total: 115 dim
```

### 3.4 Computational Complexity
- **Time**: O(HGNN on 450 edges) + O(attention fusion) ≈ millisecond per step
- **Space**: O(4 pathways params) ≈ moderate, significantly smaller than monolithic network

### 3.5 Data Sources
**Static Data** (offline, from NetHack):
- 450 hyperedge definitions (preconditions, effects, failure modes)
- 33 operator rules
- 204 variants with statistical aggregates
- 50 network rules from source code

**Dynamic Data** (runtime per step):
- pre_nodes: computed from belief_vector
- scene_atoms: output from LLM grounding
- confidence: hypergraph matching score
- goal_embedding: LLM identified target vector

---

## Part 4: Integration Points

### 4.1 LLM ↔ RL Interface
- Input: Raw game observation (ASCII frame, game state)
- LLM processing: Grounding → scene_atoms extraction
- Output to RL: scene_atoms, updated belief, confidence score, goal_embedding

### 4.2 Hypergraph ↔ RL Interface
- Input: Current state, pre_nodes, scene_atoms
- Hypergraph processing: Embedding via 4 pathways, confidence computation
- Output to RL: [q_pre, q_scene, q_effect, q_rule, confidence, α weights]
- Action masking: Permanent -inf for physically infeasible actions

### 4.3 RL ↔ Execution Interface
- Output: Selected action
- If action in {query_property, safe_exploration, llm_reflection, wait}:
  - Route to appropriate handler (LLM or game environment)
  - Await response, update belief
  - Next step: recompute state and iterate
- If action in game_actions:
  - Execute in environment
  - Observe reward, next_state
  - Continue episode

---

## Part 5: Code Agent Tasks

### Implementation Checklist
- [ ] 1. Implement 4 independent actor networks
- [ ] 2. Implement AttentionWeightNet for α computation
- [ ] 3. Implement shared critic network
- [ ] 4. Implement HGNN encoder for pre-conditions pathway
- [ ] 5. Implement HGNN encoder for scene atoms pathway
- [ ] 6. Implement MLP encoder for effects pathway
- [ ] 7. Implement RuleEncoder for rule patterns pathway
- [ ] 8. Implement fusion layer (weighted combination of pathways)
- [ ] 9. Implement PPO loss computation (actor + critic + entropy)
- [ ] 10. Implement action masking mechanism
- [ ] 11. Implement reward shaping function (5-component weighted sum)
- [ ] 12. Implement training loop (data collection → batch update)
- [ ] 13. Integrate with hypergraph embedding (state construction)
- [ ] 14. Integrate with LLM grounding (observation → atoms)
- [ ] 15. Add evaluation metrics (return, episode length, query frequency, α distribution)

