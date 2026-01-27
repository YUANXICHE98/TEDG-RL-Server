#!/usr/bin/env python3
"""
V4 Smoke Test - 验证Cross-Attention机制
快速测试V4网络的基本功能
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import torch
import numpy as np

from src.core.networks_v4_cross_attention import CrossAttentionMoEPolicy


def test_network_creation():
    """测试1: 网络创建"""
    print("\n" + "="*60)
    print("测试1: 网络创建")
    print("="*60)
    
    try:
        policy = CrossAttentionMoEPolicy(
            hypergraph_path="data/hypergraph/hypergraph_gat_structure.json",
            state_dim=115,
            hidden_dim=256,
            action_dim=23,
            num_experts=4,
            use_sparsemax=True,
            cross_attn_heads=4,
            sparse_topk=0.3
        )
        
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"✓ 网络创建成功")
        print(f"  总参数: {total_params:,}")
        print(f"  Cross-Attention heads: 4")
        print(f"  Sparse topk: 0.3")
        return policy
    except Exception as e:
        print(f"✗ 网络创建失败: {e}")
        raise


def test_forward_pass(policy):
    """测试2: 前向传播"""
    print("\n" + "="*60)
    print("测试2: 前向传播")
    print("="*60)
    
    try:
        # 创建假数据
        batch_size = 4
        state = torch.randn(batch_size, 115)
        atoms = {
            'pre_nodes': [['player_alive', 'hp_full'] for _ in range(batch_size)],
            'scene_atoms': [['dlvl_1', 'in_room'] for _ in range(batch_size)]
        }
        
        # 前向传播
        with torch.no_grad():
            outputs = policy.forward(
                state, 
                atoms,
                return_attention=True,
                return_expert_logits=True
            )
        
        print(f"✓ 前向传播成功")
        print(f"\n输出维度:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {tuple(value.shape)}")
        
        return outputs
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_cross_attention(outputs):
    """测试3: Cross-Attention机制"""
    print("\n" + "="*60)
    print("测试3: Cross-Attention机制")
    print("="*60)
    
    try:
        # 检查Context Vector
        c = outputs['context_vector']
        print(f"✓ Context Vector: {tuple(c.shape)}")
        print(f"  维度: {c.shape[1]} (期望256)")
        assert c.shape[1] == 256, "Context Vector维度应该是256"
        
        # 检查注意力权重
        if 'attention_weights' in outputs and outputs['attention_weights'] is not None:
            attn = outputs['attention_weights']
            print(f"✓ Attention Weights: {tuple(attn.shape)}")
            print(f"  注意力头数: {attn.shape[1]} (期望4)")
            assert attn.shape[1] == 4, "注意力头数应该是4"
            
            # 检查稀疏性
            attn_flat = attn.squeeze(-1).squeeze(-1)  # (batch, num_heads)
            non_zero_ratio = (attn_flat > 0.01).float().mean().item()
            print(f"  非零比例: {non_zero_ratio:.2%}")
            print(f"  稀疏度: {1-non_zero_ratio:.2%}")
        
        # 对比V3和V4的输入维度
        print(f"\n✓ V3 vs V4对比:")
        print(f"  V3 Router输入: concat(h_vis, h_logic) = 512维")
        print(f"  V4 Router输入: Context Vector = 256维")
        print(f"  参数减少: {(512-256)/512*100:.1f}%")
        
    except Exception as e:
        print(f"✗ Cross-Attention测试失败: {e}")
        raise


def test_expert_routing(outputs):
    """测试4: 专家路由"""
    print("\n" + "="*60)
    print("测试4: 专家路由")
    print("="*60)
    
    try:
        alpha = outputs['alpha']
        print(f"✓ 专家权重 α: {tuple(alpha.shape)}")
        
        # 检查归一化
        alpha_sum = alpha.sum(dim=-1)
        print(f"  权重和: {alpha_sum.mean().item():.4f} (期望1.0)")
        assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5), "权重应该归一化"
        
        # 检查稀疏性
        alpha_mean = alpha.mean(dim=0)
        print(f"  平均专家使用率:")
        expert_names = ['Survival', 'Combat', 'Exploration', 'General']
        for i, name in enumerate(expert_names):
            print(f"    {name}: {alpha_mean[i].item():.4f}")
        
        # 计算熵
        entropy = -(alpha * torch.log(alpha + 1e-10)).sum(dim=-1).mean()
        print(f"  平均熵: {entropy.item():.4f}")
        print(f"    ln(4)=1.386 (均匀), ln(2)=0.693 (2专家), 0.0 (单专家)")
        
    except Exception as e:
        print(f"✗ 专家路由测试失败: {e}")
        raise


def test_action_sampling(policy):
    """测试5: 动作采样"""
    print("\n" + "="*60)
    print("测试5: 动作采样")
    print("="*60)
    
    try:
        # 创建假数据
        state = torch.randn(1, 115)
        atoms = {
            'pre_nodes': [['player_alive', 'hp_full']],
            'scene_atoms': [['dlvl_1', 'in_room']]
        }
        
        # 采样动作
        with torch.no_grad():
            action, log_prob, entropy, value = policy.get_action_and_value(
                state, atoms, deterministic=False
            )
        
        print(f"✓ 动作采样成功")
        print(f"  动作: {action.item()}")
        print(f"  Log概率: {log_prob.item():.4f}")
        print(f"  熵: {entropy.item():.4f}")
        print(f"  价值: {value.item():.4f}")
        
        # 测试确定性采样
        with torch.no_grad():
            action_det, _, _, _ = policy.get_action_and_value(
                state, atoms, deterministic=True
            )
        print(f"  确定性动作: {action_det.item()}")
        
    except Exception as e:
        print(f"✗ 动作采样失败: {e}")
        raise


def test_gradient_flow(policy):
    """测试6: 梯度流"""
    print("\n" + "="*60)
    print("测试6: 梯度流")
    print("="*60)
    
    try:
        # 创建假数据
        state = torch.randn(2, 115)
        atoms = {
            'pre_nodes': [['player_alive'] for _ in range(2)],
            'scene_atoms': [['dlvl_1'] for _ in range(2)]
        }
        
        # 前向传播
        outputs = policy.forward(state, atoms, return_expert_logits=True)
        
        # 计算假损失
        policy_logits = outputs['policy_logits']
        value = outputs['value']
        alpha = outputs['alpha']
        
        loss = policy_logits.mean() + value.mean() + alpha.mean()
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        has_grad = 0
        total_params = 0
        for name, param in policy.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    has_grad += 1
        
        print(f"✓ 梯度流测试成功")
        print(f"  有梯度的参数: {has_grad}/{total_params}")
        
        # 检查Cross-Attention的梯度
        ca_params = [name for name, p in policy.named_parameters() if 'cross_attention' in name and p.grad is not None]
        print(f"  Cross-Attention参数有梯度: {len(ca_params)}")
        
        if len(ca_params) > 0:
            print(f"  示例: {ca_params[0]}")
        
    except Exception as e:
        print(f"✗ 梯度流测试失败: {e}")
        raise


def compare_with_v3():
    """测试7: 与V3对比"""
    print("\n" + "="*60)
    print("测试7: V3 vs V4 对比")
    print("="*60)
    
    try:
        from src.core.networks_v3_gat_moe import GATGuidedMoEPolicy as V3Policy
        
        # 创建V3网络
        v3_policy = V3Policy(
            hypergraph_path="data/hypergraph/hypergraph_gat_structure.json",
            state_dim=115,
            hidden_dim=256,
            action_dim=23,
            num_experts=4,
            use_sparsemax=True
        )
        
        # 创建V4网络
        v4_policy = CrossAttentionMoEPolicy(
            hypergraph_path="data/hypergraph/hypergraph_gat_structure.json",
            state_dim=115,
            hidden_dim=256,
            action_dim=23,
            num_experts=4,
            use_sparsemax=True,
            cross_attn_heads=4,
            sparse_topk=0.3
        )
        
        v3_params = sum(p.numel() for p in v3_policy.parameters())
        v4_params = sum(p.numel() for p in v4_policy.parameters())
        
        print(f"✓ 参数对比:")
        print(f"  V3总参数: {v3_params:,}")
        print(f"  V4总参数: {v4_params:,}")
        print(f"  差异: {v4_params - v3_params:+,} ({(v4_params-v3_params)/v3_params*100:+.1f}%)")
        
        # 前向传播速度对比
        import time
        state = torch.randn(8, 115)
        atoms = {
            'pre_nodes': [['player_alive'] for _ in range(8)],
            'scene_atoms': [['dlvl_1'] for _ in range(8)]
        }
        
        # V3速度
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = v3_policy.forward(state, atoms)
        v3_time = time.time() - start
        
        # V4速度
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = v4_policy.forward(state, atoms)
        v4_time = time.time() - start
        
        print(f"\n✓ 速度对比 (100次前向传播):")
        print(f"  V3: {v3_time:.3f}s")
        print(f"  V4: {v4_time:.3f}s")
        print(f"  差异: {(v4_time-v3_time)/v3_time*100:+.1f}%")
        
    except Exception as e:
        print(f"✗ V3 vs V4对比失败: {e}")
        print(f"  (可能V3网络不可用，跳过)")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("V4 Cross-Attention MoE - Smoke Test")
    print("="*60)
    
    try:
        # 测试1: 网络创建
        policy = test_network_creation()
        
        # 测试2: 前向传播
        outputs = test_forward_pass(policy)
        
        # 测试3: Cross-Attention机制
        test_cross_attention(outputs)
        
        # 测试4: 专家路由
        test_expert_routing(outputs)
        
        # 测试5: 动作采样
        test_action_sampling(policy)
        
        # 测试6: 梯度流
        test_gradient_flow(policy)
        
        # 测试7: 与V3对比
        compare_with_v3()
        
        # 总结
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        print("\nV4 Cross-Attention机制工作正常，可以开始训练。")
        print("\n下一步:")
        print("  python ablation_v4/train/train_v4_cross_attention.py \\")
        print("      --exp-name v4_test_100 \\")
        print("      --episodes 100 \\")
        print("      --max-steps 500")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ 测试失败")
        print("="*60)
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
