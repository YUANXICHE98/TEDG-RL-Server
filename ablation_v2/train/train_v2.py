#!/usr/bin/env python3
"""
TEDG-RL NetHack训练 - V2版本
支持 Gumbel-Softmax 硬路由和 Sparse MoE
"""

import os
import sys
import json
import time
import random
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

# 允许直接 `python ablation_v2/train/train_v2.py` 运行（不依赖外部PYTHONPATH脚本）
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 导入核心模块
from src.core.state_constructor import StateConstructor
from src.core.networks_correct import MultiChannelPolicyNet
from src.core.ppo_trainer import PPOTrainer
from src.core.action_masking import ActionMasker
from src.core.hypergraph_matcher import HypergraphMatcher
from src.core.hypergraph_loader import EmbeddingMatcher


def get_device():
    """自动检测可用设备：MUSA > CUDA > CPU"""
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


def extract_state_from_nethack_obs(obs, state_constructor, matcher, t_now=0, verbose=False, embedding_matcher=None, return_atoms=False):
    """从NetHack观测中提取状态向量
    
    Args:
        return_atoms: 如果为True，返回(state, pre_nodes, scene_atoms, confidence)用于MASK
    """
    # 解析blstats
    blstats = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
    
    # 提取基础信息
    hp = blstats[nh.NLE_BL_HP]
    hp_max = blstats[nh.NLE_BL_HPMAX]
    depth = blstats[nh.NLE_BL_DEPTH]
    gold = blstats[nh.NLE_BL_GOLD]
    ac = blstats[nh.NLE_BL_AC]
    exp_level = blstats[nh.NLE_BL_XP]
    hunger = blstats[nh.NLE_BL_HUNGER]
    
    # 构建belief向量
    belief = np.zeros(50, dtype=np.float32)
    belief[0] = 1.0  # player_alive
    belief[1] = 1.0 if hp > 0 else 0.0  # game_active
    belief[2] = 1.0  # any_hp
    belief[3] = hp / max(hp_max, 1)  # hp_ratio
    belief[4] = 1.0 if hunger == 0 else 0.0  # hunger_satiated
    belief[5] = 1.0 if gold == 0 else 0.0  # no_gold
    belief[6] = 1.0  # power_full
    belief[7] = 1.0  # not_blind
    belief[8] = 1.0  # hands_free
    belief[9] = 1.0 if ac > 0 else 0.0  # has_armor
    # ... 其余belief维度
    
    # 提取原子信息（简化版）
    pre_nodes = []
    scene_atoms = []
    effect_atoms = []
    rule_atoms = []
    
    # 根据游戏状态添加原子
    if hp > 0:
        pre_nodes.append("player_alive")
    if hp_max > 0 and hp / hp_max < 0.5:
        pre_nodes.append("low_hp")
    if depth > 0:
        scene_atoms.append(f"dlvl_{depth}")
    if gold > 0:
        pre_nodes.append("has_gold")
    
    # 超图匹配
    if embedding_matcher:
        # V2: 使用嵌入匹配
        confidence, matched_edges = embedding_matcher.match(
            pre_nodes, scene_atoms, effect_atoms, rule_atoms, top_k=8
        )
    else:
        # V1: 使用覆盖率匹配
        confidence, matched_edges = matcher.match(
            pre_nodes, scene_atoms, effect_atoms, rule_atoms
        )
    
    # 构建状态向量
    goal = np.zeros(16, dtype=np.float32)
    goal[0] = 1.0
    
    state = state_constructor.construct_state(
        belief_vector=belief,
        pre_nodes=pre_nodes,
        scene_atoms=scene_atoms,
        eff_metadata=[],
        conditional_effects=[],
        confidence=confidence,
        goal_embedding=goal,
    )
    
    if return_atoms:
        return state, pre_nodes, scene_atoms, confidence
    return state


class ConfidenceRouter:
    """动态置信度路由器"""
    
    def __init__(self, window_size: int = 500, warmup_steps: int = 100):
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.history: list[float] = []
        self.high_threshold = 0.7
        self.low_threshold = 0.3
    
    def update(self, confidence: float):
        self.history.append(confidence)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        if len(self.history) >= self.warmup_steps:
            sorted_conf = sorted(self.history)
            n = len(sorted_conf)
            self.high_threshold = sorted_conf[int(n * 0.75)]
            self.low_threshold = sorted_conf[int(n * 0.25)]
    
    def route(self, confidence: float) -> str:
        if confidence >= self.high_threshold:
            return "high"
        elif confidence >= self.low_threshold:
            return "mid"
        else:
            return "low"
    
    def get_stats(self) -> Dict:
        if not self.history:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "low_th": self.low_threshold, "high_th": self.high_threshold}
        
        return {
            "mean": np.mean(self.history),
            "std": np.std(self.history),
            "min": np.min(self.history),
            "max": np.max(self.history),
            "low_th": self.low_threshold,
            "high_th": self.high_threshold
        }


def main():
    """主训练循环"""
    parser = argparse.ArgumentParser(description="TEDG-RL V2 训练脚本")
    parser.add_argument("--exp-name", type=str, default="baseline_2000", help="实验名称")
    parser.add_argument("--episodes", type=int, default=50000, help="训练episodes数（最多5万，确保足够训练量）")
    parser.add_argument("--max-steps", type=int, default=2000, help="每episode最大步数")
    parser.add_argument("--patience", type=int, default=5000, help="收敛检测：连续N个episode无提升则停止")
    parser.add_argument("--min-episodes", type=int, default=10000, help="最少训练episodes数（约2000万步）")
    parser.add_argument("--no-mask", action="store_true", help="禁用动作掩码（No Mask对照组）")
    parser.add_argument("--use-gumbel", action="store_true", help="使用Gumbel-Softmax硬路由")
    parser.add_argument("--gumbel-tau", type=float, default=1.0, help="Gumbel温度参数")
    parser.add_argument("--sparse-topk", type=int, default=None, help="Sparse MoE的top-k值")
    parser.add_argument("--use-embedding", action="store_true", help="使用嵌入匹配")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径")
    args = parser.parse_args()
    
    print(f"=== TEDG-RL V2 训练启动 ===")
    print(f"实验名称: {args.exp_name}")
    print(f"训练配置: {args.episodes} episodes, {args.max_steps} steps/episode")
    print(f"网络配置: Gumbel={args.use_gumbel}, Sparse Top-K={args.sparse_topk}, Embedding={args.use_embedding}")
    
    # 设备检测
    device = get_device()
    
    # 创建输出目录
    output_dir = Path(f"ablation_v2/results/{args.exp_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # 加载超图
    print("\n[加载超图数据]")
    state_constructor = StateConstructor("data/hypergraph/hypergraph_complete_real.json")
    
    # 初始化匹配器
    if args.use_embedding:
        embedding_matcher = EmbeddingMatcher(min_support=5)
        matcher = None
        print(f"✓ 使用嵌入匹配: {len(embedding_matcher.atom_cache)} 个 atom 缓存")
    else:
        embedding_matcher = None
        matcher = HypergraphMatcher(state_constructor.hypergraph, weights=(0.35, 0.35, 0.2, 0.1), tau=200.0)
        print(f"✓ 使用覆盖率匹配: {len(matcher.edges)} 条超边")
    
    # 创建环境
    print("\n[创建NetHack环境]")
    try:
        env = gym.make("NetHackScore-v0")
    except:
        env = gym.make("NetHack-v0")
    print(f"✓ 动作空间: {env.action_space.n}个动作")
    
    # 创建网络 - V2 支持新参数
    policy_net = MultiChannelPolicyNet(
        state_dim=115,
        action_dim=23,
        actor_hidden_dim=128,
        attention_hidden_dim=64,
        use_gumbel=args.use_gumbel,
        gumbel_tau=args.gumbel_tau,
        sparse_topk=args.sparse_topk,
    )
    
    total_params = sum(p.numel() for p in policy_net.parameters())
    print(f"✓ 网络参数: {total_params:,}")
    
    # 创建训练器
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
    
    # 训练统计 (先初始化，后面可能被恢复)
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    alpha_history = []  # 记录 Actor 权重
    best_reward = float("-inf")
    best_score = 0
    best_avg_score = 0  # 用于收敛检测
    no_improve_count = 0  # 无提升计数
    start_episode = 0  # 起始 episode
    
    # 恢复checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"\n[恢复checkpoint: {args.resume}]")
        checkpoint = torch.load(args.resume, map_location=device)
        if "policy_net" in checkpoint:
            policy_net.load_state_dict(checkpoint["policy_net"])
        if "optimizer" in checkpoint and hasattr(trainer, 'optimizer'):
            trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        # 恢复训练进度
        if "episode" in checkpoint:
            start_episode = checkpoint["episode"] + 1
        if "best_reward" in checkpoint:
            best_reward = checkpoint["best_reward"]
        if "best_score" in checkpoint:
            best_score = checkpoint["best_score"]
        print(f"✓ 模型参数已恢复, 从 Episode {start_episode} 继续, Best Reward: {best_reward:.2f}")
    
    # 初始化其他组件
    action_masker = ActionMasker(state_constructor.hypergraph, num_actions=23)
    conf_router = ConfidenceRouter(window_size=500, warmup_steps=100)
    
    # V2新增: 场景-Actor-动作对应记录 (用于可视化)
    scene_actor_samples = []  # 采样记录: {scene, hp_ratio, action, actor_weights, dominant_actor}
    
    # 训练日志
    training_log = {
        "episode_rewards": [],
        "episode_lengths": [],
        "episode_scores": [],
        "alpha_history": [],
        "config": vars(args),
        "converged": False,
        "stopped_at": None
    }
    
    print(f"\n=== 开始训练 ===")
    start_time = time.time()
    
    for episode in range(start_episode, args.episodes):
        print(f"\n=== Episode {episode+1}/{args.episodes} 开始 ===")
        print(f"[DEBUG] 开始重置环境...")
        
        # V2新增: 重置时间衰减记录 (每个Episode独立)
        if embedding_matcher:
            embedding_matcher.reset_episode()
        
        # 重置环境
        obs, info = env.reset()
        print(f"[DEBUG] 环境重置完成，obs类型: {type(obs)}")
        print(f"[DEBUG] 开始提取状态...")
        # 提取状态和原子信息（用于MASK）
        state, current_pre_nodes, current_scene_atoms, current_confidence = extract_state_from_nethack_obs(
            obs, state_constructor, matcher, 
            embedding_matcher=embedding_matcher,
            return_atoms=True
        )
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
        
        # Episode循环
        print(f"[DEBUG] 开始Episode循环，max_steps={args.max_steps}")
        while not (done or truncated) and steps < args.max_steps:
            if steps % 100 == 0:
                print(f"[DEBUG] Step {steps}/{args.max_steps}, Reward: {total_reward:.2f}")
                sys.stdout.flush()  # 强制刷新输出
            # 获取动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                logits, alpha, value = policy_net(state_tensor)
                
                # 应用动作掩码（如果未禁用）
                action_mask = None
                if not args.no_mask:
                    action_mask = action_masker.get_action_mask(
                        current_pre_nodes,
                        current_scene_atoms,
                        current_confidence,
                    )
                    mask_t = torch.as_tensor(action_mask, device=logits.device, dtype=torch.bool)
                    # 用 -inf mask（并做兜底，避免整行全-inf导致Categorical NaN）
                    masked_logits = logits.squeeze(0).masked_fill(~mask_t, float("-inf"))
                    if not torch.isfinite(masked_logits).any():
                        masked_logits = logits.squeeze(0)
                        # 兜底：保持“有mask但全放开”，保证PPO更新时分布一致
                        action_mask = np.ones(23, dtype=bool)
                    logits = masked_logits.unsqueeze(0)
                
                # 数值稳定性：防止分布因NaN/Inf崩溃
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action.item())
            total_reward += reward
            steps += 1
            
            # 每50步打印一次详细状态
            if steps % 50 == 0:
                blstats = obs.get('blstats', [0]*27)
                print(f"[DEBUG] 状态更新: HP={blstats[nh.NLE_BL_HP]}/{blstats[nh.NLE_BL_HPMAX]}, Gold={blstats[nh.NLE_BL_GOLD]}, Done={done}, Truncated={truncated}, Reward={reward:.2f}")
                sys.stdout.flush()
            
                        
            # 记录alpha权重
            alpha_np = alpha.cpu().numpy().squeeze()
            episode_alphas.append(alpha_np)
            
            # V2新增: 采样记录场景-Actor-动作对应 (每100步采样一次)
            if steps % 100 == 0 and len(scene_actor_samples) < 10000:
                blstats_now = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
                hp_ratio = blstats_now[nh.NLE_BL_HP] / max(blstats_now[nh.NLE_BL_HPMAX], 1)
                depth_now = blstats_now[nh.NLE_BL_DEPTH]
                dominant_actor = int(np.argmax(alpha_np))  # 0=Pre, 1=Scene, 2=Effect, 3=Rule
                
                scene_actor_samples.append({
                    "episode": episode,
                    "step": steps,
                    "hp_ratio": float(hp_ratio),
                    "depth": int(depth_now),
                    "action": int(action.item()),
                    "alpha": alpha_np.tolist(),
                    "dominant": dominant_actor,
                    "reward": float(reward)
                })
            
            # V2新增: 更新时间步 (用于时间衰减计算)
            if embedding_matcher:
                embedding_matcher.set_step(steps)
            
            # 存储经验（同时更新原子信息用于下一步MASK）
            next_state, current_pre_nodes, current_scene_atoms, current_confidence = extract_state_from_nethack_obs(
                obs, state_constructor, matcher,
                embedding_matcher=embedding_matcher,
                return_atoms=True
            )
            
            trainer.buffer.add(
                state=state,
                action=action.item(),
                reward=reward,
                next_state=next_state,
                done=done or truncated,
                log_prob=dist.log_prob(action).item(),
                action_mask=action_mask,
            )
            
            state = next_state
            
            # 更新网络
            if len(trainer.buffer) >= trainer.batch_size:
                update_stats = trainer.update()
        
        # Episode结束统计
        print(f"[DEBUG] Episode结束，统计中...")
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        final_score = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)[nh.NLE_BL_SCORE]
        episode_scores.append(final_score)
        print(f"[DEBUG] 统计完成: reward={total_reward:.2f}, score={final_score}, steps={steps}")
        sys.stdout.flush()
        
        # 记录平均 alpha 权重
        if episode_alphas:
            avg_alpha = np.mean(episode_alphas, axis=0).tolist()
            alpha_history.append(avg_alpha)
        
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
                "config": vars(args)
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
        
        # 收敛检测（每100个episode检查一次）
        if (episode + 1) % 100 == 0 and episode >= args.min_episodes:
            recent_avg = np.mean(episode_scores[-500:]) if len(episode_scores) >= 500 else np.mean(episode_scores)
            if recent_avg > best_avg_score * 1.02:  # 提升超过2%
                best_avg_score = recent_avg
                no_improve_count = 0
            else:
                no_improve_count += 100
            
            if no_improve_count >= args.patience:
                print(f"\n=== 收敛检测：连续 {no_improve_count} episodes 无显著提升，停止训练 ===")
                training_log["converged"] = True
                training_log["stopped_at"] = episode + 1
                break
        
        # 打印进度 - 每轮都打印
        if True:  # 临时改为每轮都打印
            avg_reward = np.mean(episode_rewards[-1:]) if episode_rewards else 0
            avg_score = np.mean(episode_scores[-1:]) if episode_scores else 0
            print(f"\n[DEBUG] Episode {episode+1}/{args.episodes}")
            print(f"[DEBUG] 奖励: {total_reward:.2f}, 步数: {steps}, 分数: {final_score}")
            sys.stdout.flush()
            
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_score = np.mean(episode_scores[-50:])
            print(f"\nEpisode {episode+1}/{args.episodes} (patience: {args.patience - no_improve_count})")
            print(f"  平均奖励: {avg_reward:.2f}, 最佳奖励: {best_reward:.2f}")
            print(f"  平均分数: {avg_score:.1f}, 最佳分数: {best_score}, 最佳平均: {best_avg_score:.1f}")
            
            if episode_alphas:
                avg_alpha = np.mean(episode_alphas, axis=0)
                max_alpha = np.max(episode_alphas, axis=0)
                min_alpha = np.min(episode_alphas, axis=0)
                alpha_std = np.std(np.array(episode_alphas), axis=0)
                print(f"  平均α权重: pre={avg_alpha[0]:.3f}, scene={avg_alpha[1]:.3f}, "
                      f"effect={avg_alpha[2]:.3f}, rule={avg_alpha[3]:.3f}")
                print(f"  权重极值: max={max_alpha.max():.3f}, min={min_alpha.min():.3f}, "
                      f"std={alpha_std.mean():.3f} {'✓硬路由' if max_alpha.max() > 0.5 else '⚠软融合'}")
    
    # 保存最终结果
    torch.save({
        "policy_net": policy_net.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "episode": args.episodes,
        "best_reward": best_reward,
        "best_score": best_score,
        "config": vars(args)
    }, output_dir / "checkpoints" / "model_final.pth")
    
    # 更新训练日志 (转换numpy类型为Python原生类型)
    training_log["episode_rewards"] = [float(r) for r in episode_rewards]
    training_log["episode_lengths"] = [int(l) for l in episode_lengths]
    training_log["episode_scores"] = [int(s) for s in episode_scores]
    training_log["alpha_history"] = [[float(a) for a in alpha] for alpha in alpha_history]
    training_log["best_avg_score"] = float(best_avg_score)
    training_log["scene_actor_samples"] = scene_actor_samples  # V2新增
    
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


if __name__ == "__main__":
    main()
