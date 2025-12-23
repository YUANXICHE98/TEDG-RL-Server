#!/usr/bin/env python3
"""
TEDG-RL NetHack训练 - H-RAM 文档方案
保留4个Acto·r专家 + 检索上下文 + Gumbel硬路由
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import nle.env
import nle.nethack as nh

# 允许直接 `python ablation_v2/train/train_hram_doc.py` 运行（不依赖外部PYTHONPATH脚本）
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 导入核心模块
from src.core.state_constructor import StateConstructor
from src.core.networks_hram import HRAMPolicyNetDoc
from src.core.ppo_trainer import PPOTrainer


def get_device():
    """自动检测可用设备"""
    if os.getenv("TEDG_DEVICE", "").lower() == "cpu" or os.getenv("TEDG_FORCE_CPU", "0") == "1":
        print("✓ 强制使用CPU设备 (TEDG_DEVICE)")
        return torch.device('cpu')
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


def extract_state_from_nethack_obs(obs):
    """从NetHack观测中提取状态向量"""
    blstats = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
    
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
    
    state = np.zeros(115, dtype=np.float32)
    
    # 基础状态
    state[0] = 1.0 if hp > 0 else 0.0
    state[1] = hp / max(hp_max, 1)
    state[2] = depth / 20.0
    state[3] = np.log1p(gold) / 10.0
    state[4] = ac / 20.0
    state[5] = exp_level / 30.0
    state[6] = hunger / 20.0
    
    # 属性值
    state[7] = strength / 25.0
    state[8] = dexterity / 25.0
    state[9] = constitution / 25.0
    state[10] = intelligence / 25.0
    state[11] = wisdom / 25.0
    state[12] = charisma / 25.0
    
    # 状态位
    state[18] = 1.0 if hunger == 0 else 0.0
    state[19] = 1.0 if hunger > 15 else 0.0
    state[20] = 1.0 if hunger > 18 else 0.0
    
    # 位置信息
    x = blstats[nh.NLE_BL_X] if nh.NLE_BL_X < len(blstats) else 0
    y = blstats[nh.NLE_BL_Y] if nh.NLE_BL_Y < len(blstats) else 0
    state[31] = x / 80.0
    state[32] = y / 80.0
    
    return state


def main():
    """主训练循环"""
    parser = argparse.ArgumentParser(description="TEDG-RL H-RAM 文档方案训练")
    parser.add_argument("--exp-name", type=str, default="hram_doc", help="实验名称")
    parser.add_argument("--episodes", type=int, default=50000, help="训练episodes数（最多5万）")
    parser.add_argument("--max-steps", type=int, default=2000, help="每episode最大步数")
    parser.add_argument("--patience", type=int, default=5000, help="收敛检测：连续N个episode无提升则停止")
    parser.add_argument("--min-episodes", type=int, default=10000, help="最少训练episodes数（约2000万步）")
    parser.add_argument("--gumbel-tau", type=float, default=1.0, help="Gumbel温度")
    parser.add_argument(
        "--alpha-entropy-coef",
        type=float,
        default=0.0,
        help="α(专家路由)熵正则系数；HRAMDoc 想要专家分工时建议设为 0（>0 会鼓励均匀=25%/25%/25%/25%）",
    )
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint")
    args = parser.parse_args()
    
    print(f"=== TEDG-RL H-RAM 文档方案训练 ===")
    print(f"实验名称: {args.exp_name}")
    print(f"训练配置: {args.episodes} episodes, {args.max_steps} steps/episode")
    print(f"架构: 4 Actors + 检索上下文 + Gumbel硬路由")
    print(f"路由正则: alpha_entropy_coef={args.alpha_entropy_coef}")
    
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
    
    # 创建 H-RAM 文档方案网络
    print(f"\n[创建 H-RAM 文档方案网络]")
    policy_net = HRAMPolicyNetDoc(
        state_dim=115,
        embed_dim=3072,
        context_dim=128,
        action_dim=23,
        gumbel_tau=args.gumbel_tau
    ).to(device)
    
    total_params = sum(p.numel() for p in policy_net.parameters())
    print(f"✓ 网络参数: {total_params:,}")
    print(f"  - StateEncoder: 状态编码")
    print(f"  - HypergraphMemory: 超图记忆")
    print(f"  - ContextCompressor: 上下文压缩")
    print(f"  - 4个HRAM Actor专家")
    print(f"  - GumbelRouter: 硬路由")
    
    # 创建训练器
    trainer = PPOTrainer(
        policy_net=policy_net,
        learning_rate=1e-4,  # H-RAM参数量大，降低学习率避免logits数值爆炸
        clip_ratio=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=3,
        batch_size=128,
        device=device,
        alpha_entropy_coef=args.alpha_entropy_coef,
    )
    
    # 恢复checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"\n[恢复checkpoint: {args.resume}]")
        checkpoint = torch.load(args.resume, map_location=device)
        if "policy_net" in checkpoint:
            # strict=False忽略不匹配的key（网络结构可能有变化）
            policy_net.load_state_dict(checkpoint["policy_net"], strict=False)
        print("✓ 模型参数已恢复")
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    best_reward = float("-inf")
    best_score = 0
    
    # 路由统计
    route_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # pre, scene, effect, rule
    
    # 场景-Actor-动作对应记录 (用于可视化)
    scene_actor_samples = []
    
    training_log = {
        "episode_rewards": [],
        "episode_lengths": [],
        "episode_scores": [],
        "config": vars(args),
        "architecture": "H-RAM_Doc"
    }
    
    print(f"\n=== 开始训练 ===")
    start_time = time.time()
    
    for episode in range(args.episodes):
        print(f"\n=== Episode {episode+1}/{args.episodes} 开始 ===")
        print(f"[DEBUG] 开始重置环境...")
        obs, info = env.reset()
        print(f"[DEBUG] 环境重置完成，obs类型: {type(obs)}")
        print(f"[DEBUG] 开始提取状态...")
        state = extract_state_from_nethack_obs(obs)
        print(f"[DEBUG] 状态提取完成，state长度: {len(state) if state is not None else 0}")
        
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        episode_alphas = []
        
        # 打印初始状态
        if obs and 'blstats' in obs:
            blstats = obs['blstats']
            print(f"[DEBUG] 初始状态: HP={blstats[nh.NLE_BL_HP]}/{blstats[nh.NLE_BL_HPMAX]}, Gold={blstats[nh.NLE_BL_GOLD]}, Depth={blstats[nh.NLE_BL_DEPTH]}")
        else:
            print(f"[DEBUG] 警告: obs为空或无blstats")
        
        print(f"[DEBUG] 开始Episode循环，max_steps={args.max_steps}")
        while not (done or truncated) and steps < args.max_steps:
            if steps % 100 == 0:
                print(f"[DEBUG] Step {steps}/{args.max_steps}, Reward: {total_reward:.2f}")
                sys.stdout.flush()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                logits, alpha, value = policy_net(state_tensor)
                
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            
            # 记录路由选择
            alpha_np = alpha.cpu().numpy().squeeze()
            selected_expert = int(np.argmax(alpha_np))
            route_counts[selected_expert] += 1
            episode_alphas.append(alpha_np)
            
            obs, reward, done, truncated, info = env.step(action.item())
            
            # 记录场景-Actor-动作对应 (每100步采样一次)
            if steps % 100 == 0 and len(scene_actor_samples) < 10000:
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
                    "alpha": alpha_np.tolist(),
                    "dominant": selected_expert,
                    "reward": float(reward)
                })
            total_reward += reward
            steps += 1
            
            next_state = extract_state_from_nethack_obs(obs)
            
            trainer.buffer.add(
                state=state,
                action=action.item(),
                reward=reward,
                next_state=next_state,
                done=done or truncated,
                log_prob=dist.log_prob(action).item()
            )
            
            state = next_state
            
            if len(trainer.buffer) >= trainer.batch_size:
                trainer.update()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        final_score = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)[nh.NLE_BL_SCORE]
        episode_scores.append(final_score)
        
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({
                "policy_net": policy_net.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "episode": episode,
                "best_reward": best_reward,
                "best_score": final_score,
                "config": vars(args)
            }, output_dir / "checkpoints" / "best_model.pth")
        
        if final_score > best_score:
            best_score = final_score
        
        if (episode + 1) % 500 == 0:
            torch.save({
                "policy_net": policy_net.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "episode": episode,
                "config": vars(args)
            }, output_dir / "checkpoints" / f"model_{episode+1:05d}.pth")
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_score = np.mean(episode_scores[-50:])
            total_routes = sum(route_counts.values())
            
            print(f"\nEpisode {episode+1}/{args.episodes}")
            print(f"  平均奖励: {avg_reward:.2f}, 最佳奖励: {best_reward:.2f}")
            print(f"  平均分数: {avg_score:.1f}, 最佳分数: {best_score}")
            print(f"  路由分布: Pre={route_counts[0]/total_routes:.1%}, "
                  f"Scene={route_counts[1]/total_routes:.1%}, "
                  f"Effect={route_counts[2]/total_routes:.1%}, "
                  f"Rule={route_counts[3]/total_routes:.1%}")
    
    # 保存最终结果
    torch.save({
        "policy_net": policy_net.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "episode": args.episodes,
        "best_reward": best_reward,
        "best_score": best_score,
        "route_counts": route_counts,
        "config": vars(args)
    }, output_dir / "checkpoints" / "model_final.pth")
    
    training_log["episode_rewards"] = [float(r) for r in episode_rewards]
    training_log["episode_lengths"] = [int(l) for l in episode_lengths]
    training_log["episode_scores"] = [int(s) for s in episode_scores]
    training_log["route_counts"] = {str(k): int(v) for k, v in route_counts.items()}
    training_log["scene_actor_samples"] = scene_actor_samples  # 场景-权重对应记录
    
    with open(output_dir / "logs" / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    
    elapsed_time = time.time() - start_time
    print(f"\n=== 训练完成 ===")
    print(f"总耗时: {elapsed_time/3600:.2f} 小时")
    print(f"最佳奖励: {best_reward:.2f}")
    print(f"最佳分数: {best_score}")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
