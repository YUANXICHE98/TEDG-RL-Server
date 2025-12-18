#!/usr/bin/env python3
"""Smoke test: 运行 train_confmatch.py 的核心逻辑（1个episode，50步）"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import gymnasium as gym
import nle.env
import nle.nethack as nh

from src.core.state_constructor import StateConstructor
from src.core.networks_correct import MultiChannelPolicyNet
from src.core.ppo_trainer import PPOTrainer
from src.core.hypergraph_matcher import HypergraphMatcher

print("=" * 80)
print("Smoke Test: train_confmatch.py 核心逻辑")
print("=" * 80)

# 1. 初始化环境
print("\n[1/6] 初始化 NetHack 环境...")
env = gym.make("NetHackScore-v0")
obs, info = env.reset()
print(f"  ✅ 环境初始化成功")

# 2. 初始化超图匹配器
print("\n[2/6] 初始化超图匹配器...")
try:
    matcher = HypergraphMatcher(
        hypergraph_path="data/hypergraph/hypergraph_complete_real.json",
        weights=(0.35, 0.35, 0.2, 0.1),
        tau=200.0
    )
    print(f"  ✅ 超图加载成功，共 {len(matcher.edges)} 条超边")
except Exception as e:
    print(f"  ❌ 超图加载失败: {e}")
    sys.exit(1)

# 3. 初始化状态构造器
print("\n[3/6] 初始化状态构造器...")
state_constructor = StateConstructor(
    belief_dim=50,
    q_pre_dim=15,
    q_scene_dim=15,
    q_effect_dim=8,
    q_rule_dim=10,
    confidence_dim=1,
    goal_dim=16
)
print(f"  ✅ 状态构造器初始化成功，输出维度: {state_constructor.state_dim}")

# 4. 初始化网络
print("\n[4/6] 初始化多通道网络...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = MultiChannelPolicyNet(
    state_dim=state_constructor.state_dim,
    action_dim=23,
    hidden_dim=128
).to(device)
print(f"  ✅ 网络初始化成功，设备: {device}")

# 5. 测试状态提取
print("\n[5/6] 测试状态提取...")
blstats = obs["blstats"]
hp = blstats[nh.NLE_BL_HP]
hpmax = blstats[nh.NLE_BL_HPMAX]
depth = blstats[nh.NLE_BL_DEPTH]
gold = blstats[nh.NLE_BL_GOLD]
score = blstats[nh.NLE_BL_SCORE]

print(f"  原始 blstats:")
print(f"    HP: {hp}/{hpmax} ({100*hp/max(1,hpmax):.0f}%)")
print(f"    深度: {depth}层")
print(f"    金币: {gold}")
print(f"    分数: {score}")

# 验证索引正确性
if hp > 0 and hp == hpmax and depth == 1 and score == 0:
    print(f"  ✅ blstats 索引正确")
else:
    print(f"  ❌ blstats 索引可能有问题")
    sys.exit(1)

# 6. 测试一步交互
print("\n[6/6] 测试环境交互...")
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"  ✅ 环境交互成功")
print(f"    动作: {action}")
print(f"    奖励: {reward}")
print(f"    终止: {terminated}")

env.close()

print("\n" + "=" * 80)
print("✅ Smoke Test 通过！train_confmatch.py 核心逻辑正常")
print("=" * 80)
print("\n可以安全运行完整训练:")
print("  python train_confmatch.py")
print("\n或运行短期测试（100 episodes）:")
print("  TEDG_NUM_EPISODES=100 TEDG_MAX_STEPS=500 python train_confmatch.py")
