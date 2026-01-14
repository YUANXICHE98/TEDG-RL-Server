# Manager Constraint Implementation Guide

## Overview

This document provides a step-by-step implementation guide for adding Manager's inner constraints (hypergraph planning constraints) to the V3 training pipeline.

## 1. Operator-to-Expert Mapping

Based on the 76 unique operators in the hypergraph, we map them to 4 experts:

### Expert 0: Survival (生存专家)
**Focus**: Health, hunger, safety, resources
**Operators** (18):
- `eat`, `drink`, `quaff`, `pray`
- `flee`, `wait`, `teleport`
- `wear`, `put_on`, `remove`, `takeoff`, `remove_ring`
- `drop`, `drop_multi`, `pickup`
- `count_gold`, `save_game`, `sit`

### Expert 1: Combat (战斗专家)
**Focus**: Fighting, attacking, weapons
**Operators** (15):
- `attack`, `attack_melee`, `attack_ranged`, `melee_attack`, `ranged_attack`
- `fire`, `throw`, `kick`
- `wield`, `swap_weapon`, `enhance`
- `elbereth`, `turn_undead`
- `medusa`, `vlad`

### Expert 2: Exploration (探索专家)
**Focus**: Movement, searching, information gathering
**Operators** (28):
- `move`, `move_no_pickup`, `move_no_fight`, `move_until_near`
- `search`, `look`, `whatis`, `identify`, `identify_trap`
- `open`, `open_door`, `close`, `close_door`, `unlock_door`, `force_lock`
- `go_up`, `go_down`, `climb`, `jump`
- `inventory`, `known`, `call_name`, `name`
- `loot`, `untrap`
- `minetown`, `oracle`, `sokoban`

### Expert 3: General (通用专家)
**Focus**: Utility, special actions, fallback
**Operators** (15):
- `apply`, `invoke`, `zap`, `rub`, `read`, `engrave`, `wipe`
- `dip`, `put_in`
- `altar`, `offer`, `sacrifice`, `pay`
- `extended_command`, `toggle_autopickup`



## 2. Implementation Steps

### Step 1: Create Operator-Expert Mapping File

Create `src/core/operator_expert_mapping.py`:

```python
"""Operator to Expert mapping for hypergraph-guided MoE."""

# Mapping from operator name to expert index
OPERATOR_TO_EXPERT = {
    # Expert 0: Survival
    'eat': 0, 'drink': 0, 'quaff': 0, 'pray': 0,
    'flee': 0, 'wait': 0, 'teleport': 0,
    'wear': 0, 'put_on': 0, 'remove': 0, 'takeoff': 0, 'remove_ring': 0,
    'drop': 0, 'drop_multi': 0, 'pickup': 0,
    'count_gold': 0, 'save_game': 0, 'sit': 0,
    
    # Expert 1: Combat
    'attack': 1, 'attack_melee': 1, 'attack_ranged': 1,
    'melee_attack': 1, 'ranged_attack': 1,
    'fire': 1, 'throw': 1, 'kick': 1,
    'wield': 1, 'swap_weapon': 1, 'enhance': 1,
    'elbereth': 1, 'turn_undead': 1,
    'medusa': 1, 'vlad': 1,
    
    # Expert 2: Exploration
    'move': 2, 'move_no_pickup': 2, 'move_no_fight': 2, 'move_until_near': 2,
    'search': 2, 'look': 2, 'whatis': 2, 'identify': 2, 'identify_trap': 2,
    'open': 2, 'open_door': 2, 'close': 2, 'close_door': 2,
    'unlock_door': 2, 'force_lock': 2,
    'go_up': 2, 'go_down': 2, 'climb': 2, 'jump': 2,
    'inventory': 2, 'known': 2, 'call_name': 2, 'name': 2,
    'loot': 2, 'untrap': 2,
    'minetown': 2, 'oracle': 2, 'sokoban': 2,
    
    # Expert 3: General
    'apply': 3, 'invoke': 3, 'zap': 3, 'rub': 3, 'read': 3,
    'engrave': 3, 'wipe': 3,
    'dip': 3, 'put_in': 3,
    'altar': 3, 'offer': 3, 'sacrifice': 3, 'pay': 3,
    'extended_command': 3, 'toggle_autopickup': 3,
}

EXPERT_NAMES = ['Survival', 'Combat', 'Exploration', 'General']

def get_expert_for_operator(operator_name: str) -> int:
    """Get expert index for an operator."""
    return OPERATOR_TO_EXPERT.get(operator_name, 3)  # Default to General
```

### Step 2: Add Loss Functions

Add to `ablation_v3/train/train_v3_gat_moe.py`:


```python
import torch
import torch.nn.functional as F
from src.core.operator_expert_mapping import OPERATOR_TO_EXPERT

def aggregate_operators_to_experts(operator_scores, operator_names, num_experts=4):
    """
    Aggregate operator scores to expert scores.
    
    Args:
        operator_scores: (batch, num_operators) GAT output scores
        operator_names: List of operator names (length = num_operators)
        num_experts: Number of experts
    
    Returns:
        expert_scores: (batch, num_experts)
    """
    batch_size = operator_scores.size(0)
    num_operators = operator_scores.size(1)
    
    # Create mapping matrix: (num_operators, num_experts)
    mapping = torch.zeros(num_operators, num_experts, device=operator_scores.device)
    for i, op_name in enumerate(operator_names):
        expert_idx = OPERATOR_TO_EXPERT.get(op_name, 3)
        mapping[i, expert_idx] = 1.0
    
    # Normalize by number of operators per expert
    expert_counts = mapping.sum(dim=0, keepdim=True).clamp(min=1.0)
    mapping = mapping / expert_counts
    
    # Aggregate: (batch, num_operators) @ (num_operators, num_experts)
    expert_scores = torch.matmul(operator_scores, mapping)
    
    return expert_scores


def hypergraph_alignment_loss(operator_scores, alpha, operator_names, temperature=1.0):
    """
    Hypergraph-Router alignment loss.
    Forces router to follow GAT's reasoning.
    
    Args:
        operator_scores: (batch, num_operators) GAT output
        alpha: (batch, num_experts) Router output
        operator_names: List of operator names
        temperature: Temperature for softmax
    
    Returns:
        loss: Scalar tensor
    """
    # Aggregate operator scores to expert scores
    expert_scores = aggregate_operators_to_experts(operator_scores, operator_names)
    
    # Create target distribution from GAT
    target_alpha = F.softmax(expert_scores / temperature, dim=-1)
    
    # KL divergence: KL(target || current)
    loss = F.kl_div(
        F.log_softmax(alpha, dim=-1),
        target_alpha,
        reduction='batchmean'
    )
    
    return loss


def enhanced_semantic_orthogonality_loss(expert_logits):
    """
    Enhanced semantic orthogonality loss.
    Forces different experts to have different policies.
    
    Args:
        expert_logits: (batch, num_experts, action_dim)
    
    Returns:
        loss: Scalar tensor
    """
    batch_size, num_experts, action_dim = expert_logits.shape
    
    # Flatten to (batch, num_experts, action_dim)
    expert_flat = expert_logits.view(batch_size, num_experts, -1)
    
    # L2 normalize
    expert_norm = F.normalize(expert_flat, p=2, dim=2)
    
    # Compute pairwise cosine similarity
    # (batch, num_experts, action_dim) @ (batch, action_dim, num_experts)
    similarity = torch.bmm(expert_norm, expert_norm.transpose(1, 2))
    
    # Penalize off-diagonal elements (experts should be different)
    mask = 1 - torch.eye(num_experts, device=similarity.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    loss = (similarity * mask).abs().mean()
    
    return loss
```

### Step 3: Modify Training Loop

In `ablation_v3/train/train_v3_gat_moe.py`, find the loss calculation section and modify:


```python
# OLD CODE (around line 904):
total_loss = (
    actor_loss +
    0.5 * critic_loss -
    config['entropy_coef'] * entropy -
    config['alpha_entropy_coef'] * alpha_entropy +
    config['load_balance_coef'] * lb_loss +
    config['diversity_coef'] * div_loss
)

# NEW CODE:
# Calculate Manager inner constraints
alignment_loss = torch.tensor(0.0, device=device)
semantic_loss = torch.tensor(0.0, device=device)

if aux_info['operator_scores'] is not None and aux_info['expert_logits'] is not None:
    # Get operator names from hypergraph
    operator_names = [node['label'] for node in hypergraph_data['nodes'] 
                     if node['type'] == 'operator']
    
    # Hypergraph-Router alignment
    alignment_loss = hypergraph_alignment_loss(
        aux_info['operator_scores'],
        alpha,
        operator_names,
        temperature=config.get('alignment_temperature', 1.0)
    )
    
    # Enhanced semantic orthogonality
    semantic_loss = enhanced_semantic_orthogonality_loss(
        aux_info['expert_logits']
    )

# Total loss with Manager constraints
total_loss = (
    actor_loss +
    0.5 * critic_loss -
    config['entropy_coef'] * entropy -
    config['alpha_entropy_coef'] * alpha_entropy +
    config['load_balance_coef'] * lb_loss +
    config['diversity_coef'] * div_loss +
    config.get('alignment_coef', 0.1) * alignment_loss +
    config.get('semantic_coef', 0.05) * semantic_loss
)
```

### Step 4: Update Config

Add to `config.yaml`:

```yaml
# Manager Inner Constraints
alignment_coef: 0.1          # Hypergraph-Router alignment
alignment_temperature: 1.0   # Temperature for target distribution
semantic_coef: 0.05          # Semantic orthogonality
```

### Step 5: Update Logging

Add to the logging section (around line 950):

```python
# Log Manager constraint losses
if episode % 10 == 0:
    print(f"  Alignment Loss: {alignment_loss.item():.4f}")
    print(f"  Semantic Loss: {semantic_loss.item():.4f}")
```

## 3. Testing Plan

### Phase 1: Smoke Test (100 episodes)
- Verify no crashes
- Check loss values are reasonable
- Monitor Alpha entropy trend

### Phase 2: Short Run (1000 episodes)
- Compare with baseline (no constraints)
- Expected: Alpha entropy drops faster
- Expected: Expert specialization emerges earlier

### Phase 3: Full Run (5000 episodes)
- Full 3-phase training
- Expected: Alpha entropy reaches 0.3-0.4
- Expected: Average score improves to 15-20

## 4. Expected Results

### Baseline (Current)
- Alpha entropy: 1.38 → 0.69 (停滞)
- Average score: 8.50 → 12.23
- Expert specialization: 中度

### With Manager Constraints
- Alpha entropy: 1.38 → 0.3-0.4 (高度专业化)
- Average score: 8.50 → 15-20 (+23-63%)
- Expert specialization: 高度（清晰的专家-场景对应）

## 5. Troubleshooting

### Issue 1: Alignment loss too high
**Symptom**: alignment_loss > 1.0
**Solution**: Increase `alignment_temperature` to 2.0 or 3.0

### Issue 2: Training unstable
**Symptom**: NaN or divergence
**Solution**: Reduce `alignment_coef` to 0.05 or 0.01

### Issue 3: No improvement
**Symptom**: Alpha entropy still stuck
**Solution**: 
- Check operator_names are correctly extracted
- Verify OPERATOR_TO_EXPERT mapping is loaded
- Increase `alignment_coef` to 0.2

## 6. Next Steps

1. Implement the code changes
2. Run smoke test (100 episodes)
3. If successful, run short test (1000 episodes)
4. Compare results with baseline
5. If promising, run full training (5000 episodes)
