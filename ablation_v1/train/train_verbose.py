#!/usr/bin/env python3
"""TEDG-RL NetHack训练 - 详细监控版本"""

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
import nle.env
import nle.nethack as nh

from src.core.state_constructor import StateConstructor
from src.core.networks_correct import MultiChannelPolicyNet
from src.core.ppo_trainer import PPOTrainer
from src.core.action_masking import ActionMasker


def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_step(step, content):
    """打印步骤"""
    print(f"[步骤{step}] {content}")


def get_device():
    """检测设备"""
    try:
        import torch_musa
        if torch.musa.is_available():
            device = torch.device('musa:0')
            print(f"✓ MUSA GPU: {torch.musa.get_device_name(0)}")
            print(f"  显存: {torch.musa.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return device
    except:
        pass
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    
    print("⚠ 使用CPU")
    return torch.device('cpu')


def extract_state_from_nethack_obs(obs: dict, state_constructor: StateConstructor, verbose=False) -> np.ndarray:
    """从NetHack观测提取state"""
    blstats = obs.get('blstats', np.zeros(nh.NLE_BLSTATS_SIZE))
    
    if verbose:
        print(f"\n  [观测解析]")
        print(
            f"    HP: {int(blstats[nh.NLE_BL_HP])}/{int(blstats[nh.NLE_BL_HPMAX])} "
            f"({blstats[nh.NLE_BL_HP]/max(blstats[nh.NLE_BL_HPMAX],1)*100:.0f}%)"
        )
        print(f"    深度: {int(blstats[nh.NLE_BL_DEPTH])}层")
        print(f"    金币: {int(blstats[nh.NLE_BL_GOLD])}")
        print(f"    饥饿: {int(blstats[nh.NLE_BL_HUNGER])}")
        print(f"    位置: ({int(blstats[nh.NLE_BL_X])}, {int(blstats[nh.NLE_BL_Y])})")
        print(f"    分数: {int(blstats[nh.NLE_BL_SCORE])}")
    
    # 构造belief (50维)
    belief = np.zeros(50, dtype=np.float32)
    belief[0] = blstats[nh.NLE_BL_HP] / max(blstats[nh.NLE_BL_HPMAX], 1)  # hp_ratio
    belief[1] = blstats[nh.NLE_BL_DEPTH] / 50.0  # depth
    belief[2] = min(blstats[nh.NLE_BL_GOLD] / 1000.0, 1.0)  # gold
    belief[3] = blstats[nh.NLE_BL_HUNGER] / 1000.0  # hunger
    belief[4] = blstats[nh.NLE_BL_STR25] / 25.0  # strength
    belief[5] = blstats[nh.NLE_BL_DEX] / 25.0  # dex
    belief[6] = blstats[nh.NLE_BL_CON] / 25.0  # con
    belief[7] = blstats[nh.NLE_BL_INT] / 25.0  # int
    belief[8] = blstats[nh.NLE_BL_WIS] / 25.0  # wis
    belief[9] = blstats[nh.NLE_BL_CHA] / 25.0  # cha
    belief[10] = blstats[nh.NLE_BL_X] / 79.0  # x
    belief[11] = blstats[nh.NLE_BL_Y] / 21.0  # y
    belief[12] = blstats[nh.NLE_BL_SCORE] / 10000.0  # score
    belief[30] = 1.0 if blstats[nh.NLE_BL_HP] < blstats[nh.NLE_BL_HPMAX] * 0.3 else 0.0  # low_hp
    belief[31] = 1.0 if blstats[nh.NLE_BL_HUNGER] > 800 else 0.0  # hungry
    
    # 从超图选择匹配的边
    edges = state_constructor.hypergraph['hyperedges']
    edge = np.random.choice(edges)
    
    # 推断pre_nodes (增强版 - 提取更多信息)
    pre_nodes = []
    
    # HP状态
    hp_ratio = blstats[nh.NLE_BL_HP] / max(blstats[nh.NLE_BL_HPMAX], 1)
    if hp_ratio >= 0.8:
        pre_nodes.append('hp_full')
    elif hp_ratio < 0.3:
        pre_nodes.append('hp_low')
    else:
        pre_nodes.append('hp_medium')
    
    # 饥饿状态
    if blstats[nh.NLE_BL_HUNGER] < 500:
        pre_nodes.append('hunger_normal')
    elif blstats[nh.NLE_BL_HUNGER] > 800:
        pre_nodes.append('hungry')
    
    # 金币
    if blstats[nh.NLE_BL_GOLD] > 0:
        pre_nodes.append('has_gold')
    else:
        pre_nodes.append('no_gold')
    
    # 力量
    if blstats[nh.NLE_BL_STR25] > 18:
        pre_nodes.append('strong')
    
    # 装备状态
    if blstats[nh.NLE_BL_AC] < 0:  # AC (armor class)
        pre_nodes.append('well_armored')
    elif blstats[nh.NLE_BL_AC] > 5:
        pre_nodes.append('poorly_armored')
    
    # scene_atoms (增强版 - 提取更多场景信息)
    scene_atoms = []
    
    # 1. 深度信息
    scene_atoms.append(f'dlvl_{int(blstats[nh.NLE_BL_DEPTH])}')
    
    # 2. 位置信息
    x, y = int(blstats[nh.NLE_BL_X]), int(blstats[nh.NLE_BL_Y])
    if x < 20:
        scene_atoms.append('near_left_edge')
    elif x > 60:
        scene_atoms.append('near_right_edge')
    
    if y < 5:
        scene_atoms.append('near_top')
    elif y > 16:
        scene_atoms.append('near_bottom')
    
    # 3. AC状态 (防御)
    ac = blstats[nh.NLE_BL_AC]
    if ac < 0:
        scene_atoms.append('ac_good')
    elif ac > 5:
        scene_atoms.append('ac_poor')
    
    # 4. 经验等级
    exp_level = int(blstats[nh.NLE_BL_EXP])
    if exp_level <= 3:
        scene_atoms.append('exp_low')
    elif exp_level >= 10:
        scene_atoms.append('exp_high')
    else:
        scene_atoms.append(f'exp_{exp_level}')
    
    # 5. 怪物检测 (简化版 - 检查glyphs)
    glyphs = obs.get('glyphs', np.zeros((21, 79)))
    nearby_glyphs = glyphs[max(0,y-2):min(21,y+3), max(0,x-2):min(79,x+3)]
    # NetHack怪物的glyph范围大约是381-638
    if np.any((nearby_glyphs >= 381) & (nearby_glyphs <= 638)):
        scene_atoms.append('monsters_present')
    
    eff_metadata = edge.get('eff_metadata', {})
    conditional_effects = eff_metadata.get('conditional_effects', [])
    
    confidence = 0.5 + 0.3 * (blstats[nh.NLE_BL_HP] / max(blstats[nh.NLE_BL_HPMAX], 1))
    
    if verbose:
        print(f"\n  [超图匹配]")
        print(f"    前置条件: {pre_nodes[:3]}")
        print(f"    场景原子: {scene_atoms}")
        print(f"    置信度: {confidence:.2f}")
    
    goal = np.zeros(16, dtype=np.float32)
    goal[0] = 1.0
    
    state = state_constructor.construct_state(
        belief_vector=belief,
        pre_nodes=pre_nodes,
        scene_atoms=scene_atoms,
        eff_metadata=eff_metadata,
        conditional_effects=conditional_effects,
        confidence=confidence,
        goal_embedding=goal,
    )
    
    if verbose:
        print(f"\n  [状态构造]")
        print(f"    state维度: {state.shape}")
        print(f"    belief: {belief[:5]} ...")
        print(f"    q_pre: {state[50:55]} ...")
        print(f"    q_scene: {state[65:70]} ...")
        print(f"    q_effect: {state[80:85]} ...")
        print(f"    q_rule: {state[88:93]} ...")
    
    return state


def main():
    """主训练循环"""
    print_section("TEDG-RL NetHack训练 - 详细监控版")
    
    # 设备检测
    print("\n[初始化]")
    print_step(1, "检测计算设备")
    device = get_device()
    
    # 创建输出目录
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # 加载超图
    print_step(2, "加载超图数据")
    state_constructor = StateConstructor("data/hypergraph/hypergraph_complete_real.json")
    print(f"  ✓ 加载完成")
    
    # 初始化动作掩蔽
    print_step(3, "初始化动作掩蔽器")
    action_masker = ActionMasker(state_constructor.hypergraph, num_actions=23)
    print(f"  ✓ 初始化完成")
    
    # 创建NetHack环境
    print_step(4, "创建NetHack环境")
    try:
        env = gym.make('NetHackScore-v0')
        print(f"  ✓ NetHackScore-v0")
    except:
        env = gym.make('NetHack-v0')
        print(f"  ✓ NetHack-v0")
    print(f"  动作空间: {env.action_space.n}个动作")
    
    # 创建网络
    print_step(5, "初始化多通道策略网络")
    policy_net = MultiChannelPolicyNet(
        state_dim=115,
        action_dim=23,
        actor_hidden_dim=128,
        attention_hidden_dim=64,
    )
    total_params = sum(p.numel() for p in policy_net.parameters())
    print(f"  ✓ 网络参数: {total_params:,}")
    print(f"  - 4个独立Actor")
    print(f"  - 1个AttentionWeightNet")
    print(f"  - 1个共享Critic")
    
    # 创建训练器
    print_step(6, "初始化PPO训练器")
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
    print(f"  ✓ 训练器就绪")
    print(f"  - 学习率: 3e-4")
    print(f"  - Batch size: 128")
    print(f"  - PPO epochs: 3")
    
    # 训练参数
    num_episodes = 10000
    max_steps = 1000
    eval_interval = 50  # 更频繁的评估
    checkpoint_interval = 500
    verbose_interval = 10  # 每10个episode详细打印一次
    
    # 统计
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    alpha_history = []
    best_reward = float('-inf')
    best_score = 0
    
    start_time = time.time()
    
    print_section("开始训练")
    print(f"总Episodes: {num_episodes}")
    print(f"每Episode最大步数: {max_steps}")
    print(f"设备: {device}")
    print(f"目标: 学习α权重动态分配 + 最大化NetHack分数")
    
    # 主训练循环
    for episode in range(num_episodes):
        verbose = (episode % verbose_interval == 0)  # 每10个episode详细打印
        
        if verbose:
            print(f"\n{'─'*80}")
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"{'─'*80}")
        
        # 重置环境
        if verbose:
            print(f"\n[1. 重置NetHack环境]")
        obs, info = env.reset()
        state = extract_state_from_nethack_obs(obs, state_constructor, verbose=verbose)
        
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        episode_alphas = []
        
        # Episode循环
        while not (done or truncated) and steps < max_steps:
            # 选择动作
            action, log_prob = trainer.select_action(state)
            
            # 获取α权重
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                alpha = trainer.policy_net.get_alpha_weights(state_tensor)
                episode_alphas.append(alpha.cpu().numpy())
            
            if verbose and steps == 0:
                print(f"\n[2. 网络决策 - 第1步]")
                print(f"  4个Actor投票:")
                print(f"    α权重: pre={alpha[0]:.3f}, scene={alpha[1]:.3f}, effect={alpha[2]:.3f}, rule={alpha[3]:.3f}")
                print(f"  选择动作: {action}")
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            
            if verbose and steps == 0:
                print(f"\n[3. 执行动作]")
                print(f"  动作ID: {action}")
                print(f"  奖励: {reward:.3f}")
                print(f"  完成: {done or truncated}")
            
            # 提取下一状态
            next_state = extract_state_from_nethack_obs(obs, state_constructor, verbose=False)
            
            # 存储经验
            trainer.store_transition(state, action, reward, next_state, done or truncated, log_prob)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # 更新策略
        if verbose:
            print(f"\n[4. 学习更新]")
            print(f"  收集经验: {len(trainer.buffer)}条")
        
        update_stats = trainer.update()
        
        if verbose and update_stats:
            print(f"  Actor Loss: {update_stats.get('actor_loss', 0):.4f}")
            print(f"  Critic Loss: {update_stats.get('critic_loss', 0):.4f}")
            print(f"  平均优势: {update_stats.get('avg_advantage', 0):.4f}")
        
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
            if verbose:
                print(f"\n  🎉 新最佳奖励: {best_reward:.2f}")
        
        if final_score > best_score:
            best_score = final_score
            if verbose:
                print(f"  🎉 新最佳分数: {best_score:.0f}")
        
        # 定期保存
        if (episode + 1) % checkpoint_interval == 0:
            trainer.save_checkpoint(str(output_dir / "checkpoints" / f"model_{episode+1:05d}.pth"))
            print(f"\n💾 保存检查点: model_{episode+1:05d}.pth")
        
        # 评估统计
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(episode_rewards[-eval_interval:])
            avg_length = np.mean(episode_lengths[-eval_interval:])
            avg_score = np.mean(episode_scores[-eval_interval:])
            
            if len(alpha_history) >= eval_interval:
                recent_alphas = np.array(alpha_history[-eval_interval:])
                avg_alpha = recent_alphas.mean(axis=0)
                
                print(f"\n{'─'*80}")
                print(f"📊 评估统计 [Episode {episode+1}]")
                print(f"{'─'*80}")
                print(f"  奖励: 平均={avg_reward:.2f}, 最佳={best_reward:.2f}")
                print(f"  分数: 平均={avg_score:.0f}, 最佳={best_score:.0f}")
                print(f"  长度: {avg_length:.0f}步")
                print(f"  α权重: pre={avg_alpha[0]:.3f}, scene={avg_alpha[1]:.3f}, effect={avg_alpha[2]:.3f}, rule={avg_alpha[3]:.3f}")
                
                elapsed = time.time() - start_time
                eps_per_sec = (episode + 1) / elapsed
                print(f"  速度: {eps_per_sec:.2f} eps/s")
                print(f"  已用时间: {elapsed/60:.1f}分钟")
    
    env.close()
    
    # 保存最终模型
    trainer.save_checkpoint(str(output_dir / "checkpoints" / "model_final.pth"))
    
    # 保存日志
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
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / "logs" / "training_log.json", 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # 最终报告
    if alpha_history:
        alpha_array = np.array(alpha_history)
        alpha_mean = alpha_array.mean(axis=0)
        alpha_std = alpha_array.std(axis=0)
        
        print_section("训练完成")
        print(f"总时间: {(time.time() - start_time)/60:.1f}分钟")
        print(f"最佳奖励: {best_reward:.2f}")
        print(f"最佳分数: {best_score:.0f}")
        print(f"平均奖励: {np.mean(episode_rewards):.2f}")
        print(f"平均分数: {np.mean(episode_scores):.0f}")
        print(f"\nα权重分布:")
        print(f"  α_pre:    {alpha_mean[0]:.3f} ± {alpha_std[0]:.3f}")
        print(f"  α_scene:  {alpha_mean[1]:.3f} ± {alpha_std[1]:.3f}")
        print(f"  α_effect: {alpha_mean[2]:.3f} ± {alpha_std[2]:.3f}")
        print(f"  α_rule:   {alpha_mean[3]:.3f} ± {alpha_std[3]:.3f}")
        print(f"\n检查点: results/checkpoints/")
        print(f"日志: results/logs/training_log.json")


if __name__ == "__main__":
    main()
