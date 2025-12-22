#!/usr/bin/env python3
"""
Smoke Test: V2 消融实验全面测试
测试所有6种实验配置：
  1. embedding2000 - MultiChannelPolicyNet + Embedding
  2. gumbel - MultiChannelPolicyNet + Gumbel
  3. sparse_moe - MultiChannelPolicyNet + Sparse MoE
  4. gumbel_sparse - MultiChannelPolicyNet + Gumbel + Sparse
  5. hram_doc - HRAMPolicyNetDoc (4 Actors + 检索)
  6. hram_e2e - HRAMPolicyNet (端到端)
"""

import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import gymnasium as gym
import nle.env
import nle.nethack as nh

# 测试配置
TEST_EPISODES = 2
TEST_STEPS = 10

def get_device():
    """自动检测可用设备"""
    try:
        import torch_musa
        if torch.musa.is_available():
            return torch.device('musa:0')
    except:
        pass
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def extract_state(obs):
    """从观测中提取115维状态"""
    blstats = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
    state = np.zeros(115, dtype=np.float32)
    
    hp = blstats[nh.NLE_BL_HP]
    hp_max = blstats[nh.NLE_BL_HPMAX]
    state[0] = 1.0 if hp > 0 else 0.0
    state[1] = hp / max(hp_max, 1)
    state[2] = blstats[nh.NLE_BL_DEPTH] / 20.0
    state[3] = np.log1p(blstats[nh.NLE_BL_GOLD]) / 10.0
    state[4] = blstats[nh.NLE_BL_AC] / 20.0
    state[5] = blstats[nh.NLE_BL_XP] / 30.0  # 经验等级
    state[6] = blstats[nh.NLE_BL_HUNGER] / 20.0
    
    return state


def test_multichannel(name, use_gumbel=False, sparse_topk=None):
    """测试 MultiChannelPolicyNet 变体"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"  use_gumbel={use_gumbel}, sparse_topk={sparse_topk}")
    print('='*60)
    
    from src.core.networks_correct import MultiChannelPolicyNet
    
    device = get_device()
    
    # 创建网络
    policy_net = MultiChannelPolicyNet(
        state_dim=115,
        action_dim=23,
        use_gumbel=use_gumbel,
        gumbel_tau=1.0,
        sparse_topk=sparse_topk
    ).to(device)
    
    params = sum(p.numel() for p in policy_net.parameters())
    print(f"  ✓ 网络参数: {params:,}")
    
    # 创建环境
    env = gym.make("NetHackScore-v0")
    
    total_reward = 0
    for ep in range(TEST_EPISODES):
        obs, _ = env.reset()
        for step in range(TEST_STEPS):
            state = extract_state(obs)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, value, alpha = policy_net(state_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            
            obs, reward, done, truncated, _ = env.step(action.item())
            total_reward += reward
            if done or truncated:
                break
    
    env.close()
    print(f"  ✓ {TEST_EPISODES} episodes × {TEST_STEPS} steps 完成")
    print(f"  ✓ 累计奖励: {total_reward:.1f}")
    return True


def test_hram_e2e():
    """测试 HRAMPolicyNet (端到端)"""
    print(f"\n{'='*60}")
    print(f"测试: hram_e2e (端到端方案)")
    print('='*60)
    
    from src.core.networks_hram import HRAMPolicyNet
    
    device = get_device()
    
    # 创建网络
    policy_net = HRAMPolicyNet(
        state_dim=115,
        embed_dim=3072,
        action_dim=23
    ).to(device)
    
    params = sum(p.numel() for p in policy_net.parameters())
    print(f"  ✓ 网络参数: {params:,}")
    
    # 创建环境
    env = gym.make("NetHackScore-v0")
    
    total_reward = 0
    for ep in range(TEST_EPISODES):
        obs, _ = env.reset()
        for step in range(TEST_STEPS):
            state = extract_state(obs)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, value, _ = policy_net(state_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            
            obs, reward, done, truncated, _ = env.step(action.item())
            total_reward += reward
            if done or truncated:
                break
    
    env.close()
    print(f"  ✓ {TEST_EPISODES} episodes × {TEST_STEPS} steps 完成")
    print(f"  ✓ 累计奖励: {total_reward:.1f}")
    return True


def test_hram_doc():
    """测试 HRAMPolicyNetDoc (文档方案)"""
    print(f"\n{'='*60}")
    print(f"测试: hram_doc (文档方案: 4 Actors + 检索)")
    print('='*60)
    
    from src.core.networks_hram import HRAMPolicyNetDoc
    
    device = get_device()
    
    # 创建网络
    policy_net = HRAMPolicyNetDoc(
        state_dim=115,
        embed_dim=3072,
        context_dim=128,
        action_dim=23,
        gumbel_tau=1.0
    ).to(device)
    
    params = sum(p.numel() for p in policy_net.parameters())
    print(f"  ✓ 网络参数: {params:,}")
    
    # 创建环境
    env = gym.make("NetHackScore-v0")
    
    total_reward = 0
    for ep in range(TEST_EPISODES):
        obs, _ = env.reset()
        for step in range(TEST_STEPS):
            state = extract_state(obs)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits, alpha, value = policy_net(state_t, use_gumbel=True)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            
            obs, reward, done, truncated, _ = env.step(action.item())
            total_reward += reward
            if done or truncated:
                break
    
    env.close()
    print(f"  ✓ {TEST_EPISODES} episodes × {TEST_STEPS} steps 完成")
    print(f"  ✓ 累计奖励: {total_reward:.1f}")
    return True


def test_output_dirs():
    """测试输出目录结构"""
    print(f"\n{'='*60}")
    print("测试: 输出目录结构")
    print('='*60)
    
    dirs = [
        "ablation_v2/results/baseline/checkpoints",
        "ablation_v2/results/baseline/logs",
        "ablation_v2/results/no_mask/checkpoints",
        "ablation_v2/results/no_mask/logs",
        "ablation_v2/results/gumbel/checkpoints",
        "ablation_v2/results/gumbel/logs",
        "ablation_v2/results/sparse_moe/checkpoints",
        "ablation_v2/results/sparse_moe/logs",
        "ablation_v2/results/gumbel_sparse/checkpoints",
        "ablation_v2/results/gumbel_sparse/logs",
        "ablation_v2/results/hram_doc/checkpoints",
        "ablation_v2/results/hram_doc/logs",
        "ablation_v2/results/hram_e2e/checkpoints",
        "ablation_v2/results/hram_e2e/logs",
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")
    
    return True


def main():
    print("=" * 70)
    print("       TEDG-RL V2 消融实验 Smoke Test")
    print("=" * 70)
    print(f"测试配置: {TEST_EPISODES} episodes × {TEST_STEPS} steps")
    
    device = get_device()
    print(f"计算设备: {device}")
    
    results = {}
    
    # 测试目录
    try:
        results["output_dirs"] = test_output_dirs()
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        results["output_dirs"] = False
    
    # 测试 MultiChannelPolicyNet 变体
    tests = [
        ("baseline", False, None),
        ("no_mask", False, None),
        ("gumbel", True, None),
        ("sparse_moe", True, 2),
        ("gumbel_sparse", True, 1),
    ]
    
    for name, use_gumbel, sparse_topk in tests:
        try:
            results[name] = test_multichannel(name, use_gumbel, sparse_topk)
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            traceback.print_exc()
            results[name] = False
    
    # 测试 H-RAM
    try:
        results["hram_e2e"] = test_hram_e2e()
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        traceback.print_exc()
        results["hram_e2e"] = False
    
    try:
        results["hram_doc"] = test_hram_doc()
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        traceback.print_exc()
        results["hram_doc"] = False
    
    # 汇总
    print("\n" + "=" * 70)
    print("                    测试结果汇总")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name:20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("✅ 所有测试通过！可以启动正式实验")
        print("\n启动命令:")
        print("  bash ablation_v2/scripts/run_all_experiments.sh")
    else:
        print("❌ 部分测试失败，请先修复问题")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
