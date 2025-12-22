#!/usr/bin/env python3
"""
TEDG-RL NetHack训练 - V3 H-RAM 版本
基于检索增强的混合专家模型，完全端到端学习
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import nle.env
import nle.nethack as nh

# 导入核心模块
from src.core.state_constructor import StateConstructor
from src.core.networks_hram import HRAMPolicyNet
from src.core.ppo_trainer import PPOTrainer
from src.core.action_masking import ActionMasker


def get_device():
    """自动检测可用设备：MUSA > CUDA > CPU"""
    try:
        import torch_musa
        if torch.musa.is_available():
            device = torch.device('musa:0')
            print(f"✓ 使用MUSA设备: {torch.musa.get_device_name(0)}")
            return device
    except:
        pass
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ 使用CUDA设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ 使用CPU设备")
    
    return device


def extract_state_from_nethack_obs(obs, state_constructor, verbose=False):
    """
    V3: 简化的状态提取 - 只需原始向量，让网络学习特征
    
    V1/V2: 手工提取 pre_nodes, scene_atoms 等
    V3: 直接使用 blstats 和原始观测，端到端学习
    """
    # 解析blstats
    blstats = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
    
    # 提取原始特征（不做过多处理）
    hp = blstats[nh.NLE_BL_HP]
    hp_max = blstats[nh.NLE_BL_HPMAX]
    depth = blstats[nh.NLE_BL_DEPTH]
    gold = blstats[nh.NLE_BL_GOLD]
    ac = blstats[nh.NLE_BL_AC]
    exp_level = blstats[nh.NLE_BL_XP]
    hunger = blstats[nh.NLE_BL_HUNGER]
    strength = blstats[nh.NLE_BL_STR25]
    dexterity = blstats[nh.NLE_BL_DEX]
    constitution = blstats[nh.NLE_BL_CON]
    intelligence = blstats[nh.NLE_BL_INT]
    wisdom = blstats[nh.NLE_BL_WIS]
    charisma = blstats[nh.NLE_BL_CHA]
    score = blstats[nh.NLE_BL_SCORE]
    
    # V3: 构建原始状态向量（让网络学习如何编码）
    # 包含所有可能有用的信息，不做过多预处理
    state = np.zeros(115, dtype=np.float32)
    
    # 基础状态 (0-20)
    state[0] = 1.0 if hp > 0 else 0.0  # player_alive
    state[1] = hp / max(hp_max, 1)  # hp_ratio
    state[2] = depth / 20.0  # depth_normalized
    state[3] = np.log1p(gold) / 10.0  # log_gold_normalized
    state[4] = ac / 20.0  # ac_normalized
    state[5] = exp_level / 30.0  # exp_normalized
    state[6] = hunger / 20.0  # hunger_normalized
    
    # 属性值 (7-17)
    state[7] = strength / 25.0
    state[8] = dexterity / 25.0
    state[9] = constitution / 25.0
    state[10] = intelligence / 25.0
    state[11] = wisdom / 25.0
    state[12] = charisma / 25.0
    
    # 状态位 (18-30) - 简化的状态编码
    state[18] = 1.0 if hunger == 0 else 0.0  # satiated
    state[19] = 1.0 if hunger > 15 else 0.0  # hungry
    state[20] = 1.0 if hunger > 18 else 0.0  # weak
    # ... 其他状态位
    
    # 位置信息 (31-40) - 如果有坐标信息
    x = blstats[nh.NLE_BL_X] if nh.NLE_BL_X < len(blstats) else 0
    y = blstats[nh.NLE_BL_Y] if nh.NLE_BL_Y < len(blstats) else 0
    state[31] = x / 80.0  # x_normalized
    state[32] = y / 80.0  # y_normalized
    
    # V1/V2 对比（注释）：
    # V1: 手工构建 belief_vector, q_pre, q_scene 等特征
    # V2: 在 V1 基础上添加 Gumbel-Softmax 等优化
    # V3: 直接提供原始信息，让网络学习特征表示
    
    if verbose:
        print(f"\n[V3 状态提取]")
        print(f"  原始状态维度: {state.shape}")
        print(f"  HP: {hp}/{hp_max}, 深度: {depth}, 金币: {gold}")
        print(f"  状态向量示例: {state[:10]} ...")
    
    return state


def main():
    """主训练循环"""
    parser = argparse.ArgumentParser(description="TEDG-RL V3 H-RAM 训练脚本")
    parser.add_argument("--exp-name", type=str, default="hram_baseline", help="实验名称")
    parser.add_argument("--episodes", type=int, default=50000, help="训练episodes数（最多5万）")
    parser.add_argument("--max-steps", type=int, default=2000, help="每episode最大步数")
    parser.add_argument("--patience", type=int, default=5000, help="收敛检测：连续N个episode无提升则停止")
    parser.add_argument("--min-episodes", type=int, default=10000, help="最少训练episodes数（约2000万步）")
    parser.add_argument("--embed-dim", type=int, default=3072, help="嵌入维度")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()
    
    print(f"=== TEDG-RL V3 H-RAM 训练启动 ===")
    print(f"实验名称: {args.exp_name}")
    print(f"训练配置: {args.episodes} episodes, {args.max_steps} steps/episode")
    print(f"架构: H-RAM (检索增强 + 端到端学习)")
    
    # 设备检测
    device = get_device()
    
    # 创建输出目录
    output_dir = Path(f"ablation_v2/results/{args.exp_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # 创建环境
    print("\n[创建NetHack环境]")
    try:
        env = gym.make("NetHackScore-v0")
    except:
        env = gym.make("NetHack-v0")
    print(f"✓ 动作空间: {env.action_space.n}个动作")
    
    # 创建 H-RAM 网络
    print(f"\n[创建 H-RAM 网络]")
    policy_net = HRAMPolicyNet(
        state_dim=115,
        embed_dim=args.embed_dim,
        action_dim=23,
        hypergraph_path="data/hypergraph/hypergraph_complete_real.json",
        embedding_index_path="data/cache/hypergraph_embedding_index_minsup5.pkl"
    )
    
    total_params = sum(p.numel() for p in policy_net.parameters())
    print(f"✓ 网络参数: {total_params:,}")
    print(f"  - StateEncoder: 状态编码器")
    print(f"  - HypergraphMemory: {len(policy_net.memory.keys)} 条超边")
    print(f"  - CrossAttentionFusion: 交叉注意力融合")
    
    # 创建训练器
    # V3.2 E2E模式: alpha_entropy_coef=0.0，因为Attention自带Softmax
    trainer = PPOTrainer(
        policy_net=policy_net,
        learning_rate=1e-4,  # E2E大模型用更低学习率，防止后期震荡
        clip_ratio=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=128,
        device=device,
        alpha_entropy_coef=0.0,  # E2E模式禁用α熵正则化
    )
    
    # 恢复checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"\n[恢复checkpoint: {args.resume}]")
        checkpoint = torch.load(args.resume, map_location=device)
        if "policy_net" in checkpoint:
            policy_net.load_state_dict(checkpoint["policy_net"])
        if "optimizer" in checkpoint and hasattr(trainer, 'optimizer'):
            trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        print("✓ 模型参数已恢复")
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    best_reward = float("-inf")
    best_score = 0
    
    # 场景-Actor-动作对应记录 (用于可视化)
    scene_actor_samples = []
    
    # 训练日志
    training_log = {
        "episode_rewards": [],
        "episode_lengths": [],
        "episode_scores": [],
        "config": vars(args),
        "architecture": "H-RAM"
    }
    
    print(f"\n=== 开始训练 ===")
    start_time = time.time()
    
    for episode in range(args.episodes):
        print(f"\n=== Episode {episode+1}/{args.episodes} 开始 ===")
        print(f"[DEBUG] 开始重置环境...")
        # 重置环境
        obs, info = env.reset()
        print(f"[DEBUG] 环境重置完成，obs类型: {type(obs)}")
        print(f"[DEBUG] 开始提取状态...")
        state = extract_state_from_nethack_obs(obs, None, verbose=args.verbose and episode % 100 == 0)
        print(f"[DEBUG] 状态提取完成，state长度: {len(state) if state is not None else 0}")
        
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        # 打印初始状态
        if obs and 'blstats' in obs:
            blstats = obs['blstats']
            print(f"[DEBUG] 初始状态: HP={blstats[nh.NLE_BL_HP]}/{blstats[nh.NLE_BL_HPMAX]}, Gold={blstats[nh.NLE_BL_GOLD]}, Depth={blstats[nh.NLE_BL_DEPTH]}")
        else:
            print(f"[DEBUG] 警告: obs为空或无blstats")
        
        # Episode循环
        print(f"[DEBUG] 开始Episode循环，max_steps={args.max_steps}")
        while not (done or truncated) and steps < args.max_steps:
            if steps % 100 == 0:
                print(f"[DEBUG] Step {steps}/{args.max_steps}, Reward: {total_reward:.2f}")
                sys.stdout.flush()
            # 获取动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                logits, attn_weights, value = policy_net(state_tensor)
                
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action.item())
            
            # 记录场景-Actor-动作对应 (每100步采样一次)
            # 注意: V3.2的attn_weights是(batch,1,K)的注意力权重，表示对K条知识的关注程度
            if steps % 100 == 0 and len(scene_actor_samples) < 10000:
                # V3.2: attn_weights是(batch, 1, K)的注意力权重
                attn_list = attn_weights.squeeze().cpu().tolist() if attn_weights is not None else [1.0]
                blstats_now = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
                hp_ratio = blstats_now[nh.NLE_BL_HP] / max(blstats_now[nh.NLE_BL_HPMAX], 1)
                depth_now = blstats_now[nh.NLE_BL_DEPTH]
                gold_now = blstats_now[nh.NLE_BL_GOLD]
                
                scene_actor_samples.append({
                    "episode": episode,
                    "step": steps,
                    "hp_ratio": float(hp_ratio),
                    "depth": int(depth_now),
                    "gold": int(gold_now),
                    "action": int(action.item()),
                    "alpha": attn_list if isinstance(attn_list, list) else [attn_list],  # V3.2: Top-K注意力权重
                    "dominant": 0,  # 端到端方案没有专家选择
                    "reward": float(reward)
                })
            total_reward += reward
            steps += 1
            
            # 每50步打印一次详细状态
            if steps % 50 == 0:
                blstats = obs.get('blstats', [0]*27)
                print(f"[DEBUG] 状态更新: HP={blstats[nh.NLE_BL_HP]}/{blstats[nh.NLE_BL_HPMAX]}, Gold={blstats[nh.NLE_BL_GOLD]}, Done={done}, Truncated={truncated}, Reward={reward:.2f}")
                sys.stdout.flush()
            
            # 存储经验
            next_state = extract_state_from_nethack_obs(obs, None)
            
            trainer.buffer.add(
                state=state,
                action=action.item(),
                reward=reward,
                next_state=next_state,
                done=done or truncated,
                log_prob=dist.log_prob(action).item()
            )
            
            state = next_state
            
            # 更新网络
            if len(trainer.buffer) >= trainer.batch_size:
                update_stats = trainer.update()
        
        # Episode结束统计
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        final_score = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)[nh.NLE_BL_SCORE]
        episode_scores.append(final_score)
        
        # 更新最佳记录
        if total_reward > best_reward:
            best_reward = total_reward
            # 保存最佳模型
            torch.save({
                "policy_net": policy_net.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "episode": episode,
                "best_reward": best_reward,
                "best_score": final_score,
                "config": vars(args),
                "alpha_example": attn_weights.squeeze().cpu().numpy().tolist() if attn_weights is not None else []
            }, output_dir / "checkpoints" / "best_model.pth")
        
        if final_score > best_score:
            best_score = final_score
        
        # 定期保存checkpoint
        if (episode + 1) % 500 == 0:
            torch.save({
                "policy_net": policy_net.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "episode": episode,
                "reward": total_reward,
                "score": final_score,
                "config": vars(args)
            }, output_dir / "checkpoints" / f"model_{episode+1:05d}.pth")
        
        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_score = np.mean(episode_scores[-50:])
            print(f"\nEpisode {episode+1}/{args.episodes}")
            print(f"  平均奖励: {avg_reward:.2f}, 最佳奖励: {best_reward:.2f}")
            print(f"  平均分数: {avg_score:.1f}, 最佳分数: {best_score}")
            
            # V3.2 特有：显示注意力权重
            if args.verbose and attn_weights is not None:
                print(f"  注意力权重: {attn_weights.squeeze().cpu().numpy()}")
    
    # 保存最终结果
    torch.save({
        "policy_net": policy_net.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "episode": args.episodes,
        "best_reward": best_reward,
        "best_score": best_score,
        "config": vars(args)
    }, output_dir / "checkpoints" / "model_final.pth")
    
    # 更新训练日志 (转换numpy类型)
    training_log["episode_rewards"] = [float(r) for r in episode_rewards]
    training_log["episode_lengths"] = [int(l) for l in episode_lengths]
    training_log["episode_scores"] = [int(s) for s in episode_scores]
    training_log["scene_actor_samples"] = scene_actor_samples  # 场景-权重对应记录
    
    with open(output_dir / "logs" / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    
    # 打印最终统计
    elapsed_time = time.time() - start_time
    print(f"\n=== 训练完成 ===")
    print(f"总耗时: {elapsed_time/3600:.2f} 小时")
    print(f"最佳奖励: {best_reward:.2f}")
    print(f"最佳分数: {best_score}")
    print(f"平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"平均分数: {np.mean(episode_scores):.1f}")
    print(f"结果保存在: {output_dir}")
    
    # V3 特有输出
    print(f"\n[V3 架构特点]")
    print(f"  - 端到端学习，无需手工特征")
    print(f"  - 检索增强，利用超图知识")
    print(f"  - 交叉注意力，动态融合信息")


if __name__ == "__main__":
    main()
