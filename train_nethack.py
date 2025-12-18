#!/usr/bin/env python3
"""TEDG-RL NetHack真实环境训练"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import time
import gymnasium as gym
import nle.env  # 导入NLE环境以注册NetHack
import nle.nethack as nh

from src.core.state_constructor import StateConstructor
from src.core.networks_correct import MultiChannelPolicyNet
from src.core.ppo_trainer import PPOTrainer
from src.core.action_masking import ActionMasker


def get_device():
    """自动检测可用设备：MUSA > CUDA > CPU"""
    try:
        import torch_musa
        if torch.musa.is_available():
            device = torch.device('musa:0')
            print(f"✓ 使用MUSA设备: {torch.musa.get_device_name(0)}")
            print(f"  显存: {torch.musa.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return device
    except:
        pass
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ 使用CUDA设备: {torch.cuda.get_device_name(0)}")
        return device
    
    print("⚠ 使用CPU")
    return torch.device('cpu')


def extract_state_from_nethack_obs(obs: dict, state_constructor: StateConstructor) -> np.ndarray:
    """
    从NetHack观测中提取115维state
    
    NetHack观测包含:
    - glyphs: (21, 79) 地图符号
    - blstats: (25,) 玩家状态 [x, y, strength, dex, con, int, wis, cha, score, hp, maxhp, depth, gold, ...]
    - message: 游戏消息
    """
    blstats = obs.get('blstats', np.zeros(nh.NLE_BLSTATS_SIZE))
    
    # 构造belief_vector (50维) - 从blstats提取
    belief = np.zeros(50, dtype=np.float32)
    
    # 前10维: 基础属性 (归一化)
    belief[0] = blstats[nh.NLE_BL_HP] / max(blstats[nh.NLE_BL_HPMAX], 1)  # hp_ratio
    belief[1] = blstats[nh.NLE_BL_DEPTH] / 50.0  # depth (dlvl)
    belief[2] = min(blstats[nh.NLE_BL_GOLD] / 1000.0, 1.0)  # gold
    belief[3] = blstats[nh.NLE_BL_HUNGER] / 1000.0  # hunger
    belief[4] = blstats[nh.NLE_BL_STR25] / 25.0  # strength
    belief[5] = blstats[nh.NLE_BL_DEX] / 25.0  # dexterity
    belief[6] = blstats[nh.NLE_BL_CON] / 25.0  # constitution
    belief[7] = blstats[nh.NLE_BL_INT] / 25.0  # intelligence
    belief[8] = blstats[nh.NLE_BL_WIS] / 25.0  # wisdom
    belief[9] = blstats[nh.NLE_BL_CHA] / 25.0  # charisma
    
    # 中间20维: 位置和环境信息
    belief[10] = blstats[nh.NLE_BL_X] / 79.0  # x position
    belief[11] = blstats[nh.NLE_BL_Y] / 21.0  # y position
    belief[12] = blstats[nh.NLE_BL_SCORE] / 10000.0  # score
    
    # 后20维: 状态标志 (简化处理)
    belief[30] = 1.0 if blstats[nh.NLE_BL_HP] < blstats[nh.NLE_BL_HPMAX] * 0.3 else 0.0  # low_hp
    belief[31] = 1.0 if blstats[nh.NLE_BL_HUNGER] > 800 else 0.0  # hungry
    
    # 从超图中选择匹配的超边 (简化: 随机选择)
    edges = state_constructor.hypergraph['hyperedges']
    edge = np.random.choice(edges)
    
    # 根据当前状态推断pre_nodes和scene_atoms
    pre_nodes = []
    if blstats[nh.NLE_BL_GOLD] > 0:
        pre_nodes.append('has_gold')
    if blstats[nh.NLE_BL_HUNGER] < 500:
        pre_nodes.append('hunger_normal')
    if blstats[nh.NLE_BL_HP] >= blstats[nh.NLE_BL_HPMAX] * 0.8:
        pre_nodes.append('hp_full')
    
    scene_atoms = [f'dlvl_{int(blstats[nh.NLE_BL_DEPTH])}']
    
    eff_metadata = edge.get('eff_metadata', {})
    conditional_effects = eff_metadata.get('conditional_effects', [])
    
    # 置信度: 基于状态完整性
    confidence = 0.5 + 0.3 * (blstats[nh.NLE_BL_HP] / max(blstats[nh.NLE_BL_HPMAX], 1))
    
    # goal embedding (简化: 固定目标向量)
    goal = np.zeros(16, dtype=np.float32)
    goal[0] = 1.0  # 目标: 向下探索
    
    state = state_constructor.construct_state(
        belief_vector=belief,
        pre_nodes=pre_nodes,
        scene_atoms=scene_atoms,
        eff_metadata=eff_metadata,
        conditional_effects=conditional_effects,
        confidence=confidence,
        goal_embedding=goal,
    )
    
    return state


def main():
    """主训练循环"""
    print("=" * 80)
    print("TEDG-RL NetHack真实环境训练")
    print("=" * 80)
    
    # 检测设备
    print("\n[设备检测]")
    device = get_device()
    
    # 创建输出目录
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # 初始化
    print("\n[初始化] 加载超图...")
    state_constructor = StateConstructor("data/hypergraph/hypergraph_complete_real.json")
    
    print("[初始化] 初始化动作掩蔽器...")
    action_masker = ActionMasker(state_constructor.hypergraph, num_actions=23)  # NetHack有23个动作
    
    print("[初始化] 创建NetHack环境...")
    try:
        env = gym.make('NetHackScore-v0')
        print(f"  ✓ 环境创建成功: NetHackScore-v0")
        print(f"  - 动作空间: {env.action_space.n}")
    except Exception as e:
        print(f"  尝试备用环境...")
        env = gym.make('NetHack-v0')
        print(f"  ✓ 环境创建成功: NetHack-v0")
        print(f"  - 动作空间: {env.action_space.n}")
    
    print("[初始化] 初始化策略网络...")
    policy_net = MultiChannelPolicyNet(
        state_dim=115,
        action_dim=23,  # NetHack动作数
        actor_hidden_dim=128,
        attention_hidden_dim=64,
    )
    
    total_params = sum(p.numel() for p in policy_net.parameters())
    print(f"  - 网络参数: {total_params:,}")
    
    print("[初始化] 初始化PPO训练器...")
    trainer = PPOTrainer(
        policy_net=policy_net,
        learning_rate=3e-4,
        clip_ratio=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=128,
        device=device,
    )
    
    # 训练参数
    num_episodes = 10000
    max_steps = 1000  # NetHack每个episode更长
    eval_interval = 100
    checkpoint_interval = 500
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    episode_scores = []  # NetHack分数
    alpha_history = []
    best_reward = float('-inf')
    best_score = 0
    
    start_time = time.time()
    
    print(f"\n" + "=" * 80)
    print(f"开始训练: {num_episodes} episodes")
    print(f"  - 真实NetHack环境")
    print(f"  - Batch size: 128")
    print(f"  - 设备: {device}")
    print(f"  - 目标: 学习α权重 + 最大化NetHack分数")
    print(f"=" * 80 + "\n")
    
    # 主训练循环
    pbar = tqdm(range(num_episodes), desc="Training")
    
    for episode in pbar:
        # 重置环境
        obs, info = env.reset()
        state = extract_state_from_nethack_obs(obs, state_constructor)
        
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        episode_alphas = []
        
        # 单个episode循环
        while not (done or truncated) and steps < max_steps:
            # 选择动作
            action, log_prob = trainer.select_action(state)
            
            # 记录α权重
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                alpha = trainer.policy_net.get_alpha_weights(state_tensor)
                episode_alphas.append(alpha.cpu().numpy())
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            
            # 提取下一个状态
            next_state = extract_state_from_nethack_obs(obs, state_constructor)
            
            # 存储经验
            trainer.store_transition(state, action, reward, next_state, done or truncated, log_prob)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # 更新策略
        update_stats = trainer.update()
        
        # 记录统计
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        final_score = (
            obs.get('blstats', [0] * nh.NLE_BLSTATS_SIZE)[nh.NLE_BL_SCORE] if isinstance(obs, dict) else 0
        )
        episode_scores.append(final_score)
        
        if episode_alphas:
            avg_alpha = np.mean(episode_alphas, axis=0)
            alpha_history.append(avg_alpha)
        
        if total_reward > best_reward:
            best_reward = total_reward
            trainer.save_checkpoint(str(output_dir / "checkpoints" / "best_model.pth"))
        
        if final_score > best_score:
            best_score = final_score
        
        # 定期保存检查点
        if (episode + 1) % checkpoint_interval == 0:
            trainer.save_checkpoint(str(output_dir / "checkpoints" / f"model_{episode+1:05d}.pth"))
        
        # 更新进度条
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_length = np.mean(episode_lengths[-eval_interval:])
            avg_score = np.mean(episode_scores[-eval_interval:])
            
            if len(alpha_history) >= eval_interval:
                recent_alphas = np.array(alpha_history[-eval_interval:])
                avg_alpha = recent_alphas.mean(axis=0)
                
                pbar.set_postfix({
                    'reward': f'{avg_reward:.1f}',
                    'score': f'{avg_score:.0f}',
                    'len': f'{avg_length:.0f}',
                    'α_pre': f'{avg_alpha[0]:.2f}',
                    'α_sce': f'{avg_alpha[1]:.2f}',
                    'α_eff': f'{avg_alpha[2]:.2f}',
                    'α_rul': f'{avg_alpha[3]:.2f}',
                })
                
                # 详细日志
                if (episode + 1) % (eval_interval * 5) == 0:
                    elapsed = time.time() - start_time
                    eps_per_sec = (episode + 1) / elapsed
                    print(f"\n[Episode {episode+1:5d}]")
                    print(f"  奖励: avg={avg_reward:.2f}, best={best_reward:.2f}")
                    print(f"  分数: avg={avg_score:.0f}, best={best_score:.0f}")
                    print(f"  长度: {avg_length:.0f}")
                    print(f"  α权重: pre={avg_alpha[0]:.3f}, scene={avg_alpha[1]:.3f}, "
                          f"effect={avg_alpha[2]:.3f}, rule={avg_alpha[3]:.3f}")
                    print(f"  速度: {eps_per_sec:.2f} eps/s")
                    if update_stats:
                        print(f"  Loss: actor={update_stats.get('actor_loss', 0):.4f}, "
                              f"critic={update_stats.get('critic_loss', 0):.4f}")
    
    env.close()
    
    # 保存最终模型
    trainer.save_checkpoint(str(output_dir / "checkpoints" / "model_final.pth"))
    
    # 保存训练日志
    log_data = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_scores': episode_scores,
        'alpha_history': [a.tolist() for a in alpha_history],
        'best_reward': float(best_reward),
        'best_score': float(best_score),
        'total_episodes': num_episodes,
        'total_time_seconds': time.time() - start_time,
        'device': str(device),
        'environment': 'NetHackScore-v0',
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / "logs" / "training_log.json", 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # 分析结果
    if alpha_history:
        alpha_array = np.array(alpha_history)
        alpha_mean = alpha_array.mean(axis=0)
        alpha_std = alpha_array.std(axis=0)
        
        print(f"\n" + "=" * 80)
        print(f"训练完成!")
        print(f"  总时间: {(time.time() - start_time)/60:.1f} 分钟")
        print(f"  最佳奖励: {best_reward:.2f}")
        print(f"  最佳分数: {best_score:.0f}")
        print(f"  平均奖励: {np.mean(episode_rewards):.2f}")
        print(f"  平均分数: {np.mean(episode_scores):.0f}")
        print(f"\nα权重分布分析:")
        print(f"  α_pre:    {alpha_mean[0]:.3f} ± {alpha_std[0]:.3f}")
        print(f"  α_scene:  {alpha_mean[1]:.3f} ± {alpha_std[1]:.3f}")
        print(f"  α_effect: {alpha_mean[2]:.3f} ± {alpha_std[2]:.3f}")
        print(f"  α_rule:   {alpha_mean[3]:.3f} ± {alpha_std[3]:.3f}")
        print(f"\n检查点: results/checkpoints/")
        print(f"日志: results/logs/training_log.json")
        print(f"=" * 80)


if __name__ == "__main__":
    main()
