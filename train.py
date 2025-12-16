#!/usr/bin/env python3
"""TEDG-RL 训练脚本"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import numpy as np
import gymnasium as gym
from nle import nethack

from src.core.hypergraph_loader import HypergraphLoader
from src.core.mixed_embedding import MixedEmbedding
from src.core.rl_agent import TEDGRLAgent


def load_config(config_path="config.yaml"):
    """加载配置"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_state_from_observation(obs):
    """从 NLE 观测中提取状态向量"""
    # 简化：使用 glyphs 作为状态的一部分
    # 实际应该提取更丰富的特征
    glyphs = obs['glyphs']
    
    # 将 glyphs 展平并归一化
    state = glyphs.flatten().astype(np.float32) / 6000.0
    
    # 截断或填充到固定维度
    if len(state) > 115:
        state = state[:115]
    elif len(state) < 115:
        state = np.pad(state, (0, 115 - len(state)))
    
    return state


def main():
    """主训练循环"""
    print("初始化 TEDG-RL...")
    
    # 加载配置
    config = load_config()
    
    # 初始化组件
    print("加载超图...")
    hypergraph = HypergraphLoader(config["hypergraph"])
    
    print("初始化混合嵌入...")
    embedding = MixedEmbedding(config["embedding"])
    
    # 如果没有缓存的嵌入，构建新的
    if embedding.embedding_index is None:
        print("构建混合嵌入索引...")
        embedding.build_mixed_channel_embedding_index(
            hypergraph.hypergraph,
            min_support=5,
            channel_weights=config["embedding"].get("channel_weights", [0.3, 0.4, 0.2, 0.1])
        )
    
    print("初始化 RL Agent...")
    agent = TEDGRLAgent(config["rl"])
    
    # 创建环境
    print("创建 NetHack 环境...")
    env = gym.make("NetHackScore-v0", character="mon-hum-neu-mal")
    
    # 训练统计
    episode_rewards = []
    best_reward = float('-inf')
    
    print("\n开始训练...")
    for episode in range(config["rl"]["num_episodes"]):
        obs, info = env.reset()
        state = extract_state_from_observation(obs)
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < config["rl"]["max_steps"]:
            # 获取可行动作
            applicable_actions = hypergraph.get_applicable_actions(state)
            
            # 选择动作
            action, log_prob = agent.select_action(state, applicable_actions)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = extract_state_from_observation(next_obs)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done, log_prob)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # 更新模型
        if len(agent.buffer) >= agent.buffer.batch_size:
            agent.update()
        
        episode_rewards.append(total_reward)
        
        # 记录最佳模型
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save("results/checkpoints/best_model.pth")
        
        # 打印进度
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
            print(f"Episode {episode:4d} | Reward: {total_reward:6.1f} | Avg(10): {avg_reward:6.1f} | Steps: {steps:3d}")
        
        # 定期保存检查点
        if episode % 100 == 0 and episode > 0:
            agent.save(f"results/checkpoints/model_{episode:04d}.pth")
    
    # 保存最终模型
    agent.save("results/checkpoints/model_final.pth")
    
    print(f"\n训练完成！最佳奖励: {best_reward:.1f}")
    
    # 保存训练曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig("results/visualizations/training_curve.png")
    print("训练曲线已保存到 results/visualizations/training_curve.png")


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)
    
    main()
