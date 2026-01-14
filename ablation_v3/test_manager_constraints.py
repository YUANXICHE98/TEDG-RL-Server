#!/usr/bin/env python3
"""
测试Manager内层约束的实现

验证：
1. Operator到Expert的映射是否正确
2. aggregate_operators_to_experts函数是否工作
3. hypergraph_alignment_loss是否能正常计算
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import json
from src.core.operator_expert_mapping import (
    OPERATOR_TO_EXPERT, 
    EXPERT_NAMES,
    get_expert_for_operator,
    get_operators_for_expert,
    print_mapping_stats
)

# 导入loss函数（需要从训练脚本导入）
import torch.nn.functional as F


def aggregate_operators_to_experts(
    operator_scores: torch.Tensor,
    operator_names: list,
    num_experts: int = 4
) -> torch.Tensor:
    """复制自训练脚本的函数"""
    batch_size = operator_scores.size(0)
    num_operators = operator_scores.size(1)
    device = operator_scores.device
    
    mapping = torch.zeros(num_operators, num_experts, device=device)
    
    for i, op_name in enumerate(operator_names):
        base_op_name = op_name.split('_')[0] if '_' in op_name else op_name
        expert_idx = OPERATOR_TO_EXPERT.get(base_op_name, 3)
        mapping[i, expert_idx] = 1.0
    
    expert_counts = mapping.sum(dim=0, keepdim=True).clamp(min=1.0)
    mapping = mapping / expert_counts
    
    expert_scores = torch.matmul(operator_scores, mapping)
    
    return expert_scores


def hypergraph_alignment_loss(
    operator_scores: torch.Tensor,
    alpha: torch.Tensor,
    operator_names: list,
    temperature: float = 1.0
) -> torch.Tensor:
    """复制自训练脚本的函数"""
    expert_scores = aggregate_operators_to_experts(
        operator_scores, 
        operator_names, 
        num_experts=alpha.size(1)
    )
    
    target_alpha = F.softmax(expert_scores / temperature, dim=-1)
    
    loss = F.kl_div(
        F.log_softmax(alpha, dim=-1),
        target_alpha,
        reduction='batchmean'
    )
    
    return loss


def test_operator_mapping():
    """测试1：Operator到Expert的映射"""
    print("\n" + "="*70)
    print("测试1：Operator到Expert的映射")
    print("="*70)
    
    print_mapping_stats()
    
    # 测试几个例子
    test_cases = [
        ('move', 2, 'Exploration'),
        ('attack', 1, 'Combat'),
        ('eat', 0, 'Survival'),
        ('apply', 3, 'General'),
        ('unknown_operator', 3, 'General'),  # 默认
    ]
    
    print("\n测试映射:")
    all_passed = True
    for op_name, expected_idx, expected_name in test_cases:
        actual_idx = get_expert_for_operator(op_name)
        actual_name = EXPERT_NAMES[actual_idx]
        status = "✓" if actual_idx == expected_idx else "✗"
        print(f"  {status} {op_name:20s} -> Expert {actual_idx} ({actual_name})")
        if actual_idx != expected_idx:
            all_passed = False
            print(f"     期望: Expert {expected_idx} ({expected_name})")
    
    return all_passed


def test_aggregation():
    """测试2：Operator分数聚合到Expert"""
    print("\n" + "="*70)
    print("测试2：Operator分数聚合到Expert")
    print("="*70)
    
    # 加载超图结构
    with open("data/hypergraph/hypergraph_gat_structure.json", 'r') as f:
        hypergraph_structure = json.load(f)
    
    operator_names = [node['label'] for node in hypergraph_structure['nodes'] 
                     if node['type'] == 'operator']
    
    print(f"\n加载了 {len(operator_names)} 个Operator节点")
    print(f"示例: {operator_names[:5]}")
    
    # 创建模拟的operator_scores
    batch_size = 4
    num_operators = len(operator_names)
    
    # 场景1：所有Combat相关的operators得分高
    operator_scores = torch.rand(batch_size, num_operators) * 0.1
    for i, op_name in enumerate(operator_names):
        base_name = op_name.split('_')[0]
        if base_name in ['attack', 'fire', 'throw', 'kick', 'wield']:
            operator_scores[:, i] = torch.rand(batch_size) * 0.9 + 0.5  # 高分
    
    print(f"\n创建模拟数据: operator_scores shape = {operator_scores.shape}")
    
    # 聚合
    expert_scores = aggregate_operators_to_experts(operator_scores, operator_names)
    
    print(f"聚合后: expert_scores shape = {expert_scores.shape}")
    print(f"\nExpert分数 (batch 0):")
    for i, name in enumerate(EXPERT_NAMES):
        print(f"  {name:15s}: {expert_scores[0, i].item():.4f}")
    
    # 验证：Combat Expert应该得分最高
    combat_idx = 1
    max_idx = expert_scores[0].argmax().item()
    
    if max_idx == combat_idx:
        print(f"\n✓ 测试通过：Combat Expert得分最高（符合预期）")
        return True
    else:
        print(f"\n✗ 测试失败：Expert {max_idx} ({EXPERT_NAMES[max_idx]}) 得分最高")
        print(f"   期望：Expert {combat_idx} (Combat)")
        return False


def test_alignment_loss():
    """测试3：超图-路由对齐损失"""
    print("\n" + "="*70)
    print("测试3：超图-路由对齐损失")
    print("="*70)
    
    # 加载operator_names
    with open("data/hypergraph/hypergraph_gat_structure.json", 'r') as f:
        hypergraph_structure = json.load(f)
    operator_names = [node['label'] for node in hypergraph_structure['nodes'] 
                     if node['type'] == 'operator']
    
    batch_size = 4
    num_operators = len(operator_names)
    num_experts = 4
    
    # 场景1：GAT建议Combat，Router也选Combat（应该loss低）
    operator_scores = torch.rand(batch_size, num_operators) * 0.1
    for i, op_name in enumerate(operator_names):
        base_name = op_name.split('_')[0]
        if base_name in ['attack', 'fire', 'throw']:
            operator_scores[:, i] = 0.9
    
    alpha_aligned = torch.tensor([
        [0.1, 0.7, 0.1, 0.1],  # Combat主导
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
    ])
    
    loss_aligned = hypergraph_alignment_loss(
        operator_scores, alpha_aligned, operator_names, temperature=1.0
    )
    
    # 场景2：GAT建议Combat，Router选Exploration（应该loss高）
    alpha_misaligned = torch.tensor([
        [0.1, 0.1, 0.7, 0.1],  # Exploration主导
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.1, 0.7, 0.1],
    ])
    
    loss_misaligned = hypergraph_alignment_loss(
        operator_scores, alpha_misaligned, operator_names, temperature=1.0
    )
    
    print(f"\n对齐情况下的loss: {loss_aligned.item():.4f}")
    print(f"不对齐情况下的loss: {loss_misaligned.item():.4f}")
    
    if loss_misaligned > loss_aligned:
        print(f"\n✓ 测试通过：不对齐的loss更高（符合预期）")
        return True
    else:
        print(f"\n✗ 测试失败：loss关系不符合预期")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("Manager内层约束 - 功能测试")
    print("="*70)
    
    results = []
    
    # 测试1
    results.append(("Operator映射", test_operator_mapping()))
    
    # 测试2
    results.append(("Operator聚合", test_aggregation()))
    
    # 测试3
    results.append(("对齐损失", test_alignment_loss()))
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！Manager内层约束实现正确。")
    else:
        print("\n⚠️  部分测试失败，请检查实现。")
    
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
