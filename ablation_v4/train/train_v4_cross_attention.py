#!/usr/bin/env python3
"""
TEDG-RL NetHack训练 - V4版本
Cross-Attention Guided Hierarchical MoE

核心特性 (相比V3):
1. **Cross-Attention融合**: 替代concat，让符号信息主动查询视觉信息
2. **Sparse Attention Gate**: 只关注相关的视觉特征
3. **Context Vector**: 生成紧凑的256维上下文表示
4. **模态平衡**: 缓解模态主导问题

其他特性沿用V3:
- GAT推理层 - 动态激活超图节点
- Sparsemax路由 - 软中带硬，避免塌缩
- 语义专家 - Survival/Combat/Exploration/General
- 三阶段训练 - Warmup → Transition → Fine-tune
- 多重稳定性措施 - 负载均衡、多样性、NaN检测
- **所有辅助损失函数**: Manager约束、负载均衡、专家多样性等

实现策略:
- 直接导入V3的训练脚本
- 只替换网络类为V4的CrossAttentionMoEPolicy
- 其他训练逻辑（PPO、损失函数、监控等）完全沿用V3
"""

import os
import sys
import json
import time
from pathlib import Path

# 允许直接运行
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F
import numpy as np

# ============================================================================
# 核心策略: 直接使用V3的训练脚本，只替换网络类
# ============================================================================

# 1. 导入V3的所有训练组件
from ablation_v3.train.train_v3_gat_moe import (
    # 辅助损失函数 (完全沿用)
    load_balance_loss,
    expert_diversity_loss,
    aggregate_operators_to_experts,
    hypergraph_alignment_loss,
    enhanced_semantic_orthogonality_loss,
    expert_overlap_penalty,
    
    # 训练配置 (完全沿用)
    get_training_config,
    get_lr_scheduler,
    
    # 监控和诊断 (完全沿用)
    TrainingMonitor,
    NaNDetector,
    RewardNormalizer,
    
    # 工具函数 (完全沿用)
    get_device,
    extract_atoms_from_obs,
    extract_state_from_obs,
    log_gradient_norms,
)

# 2. 导入V4网络 (唯一的改动)
from src.core.networks_v4_cross_attention import CrossAttentionMoEPolicy

# ============================================================================
# V4特定的网络创建函数 (唯一需要修改的部分)
# ============================================================================

def create_v4_policy(device, args):
    """
    创建V4网络 (Cross-Attention MoE)
    
    与V3的区别:
    - 使用CrossAttentionMoEPolicy替代GATGuidedMoEPolicy
    - 新增cross_attn_heads和sparse_topk参数
    - 其他参数完全一致
    
    Args:
        device: 设备
        args: 命令行参数
    
    Returns:
        policy_net: V4策略网络
    """
    print(f"\n[初始化V4网络 - Cross-Attention MoE]")
    
    policy_net = CrossAttentionMoEPolicy(
        hypergraph_path="data/hypergraph/hypergraph_gat_structure.json",
        state_dim=115,
        hidden_dim=256,
        action_dim=23,
        num_experts=args.num_experts,
        use_sparsemax=True,
        cross_attn_heads=4,      # V4新增: Cross-Attention头数
        sparse_topk=0.3          # V4新增: 稀疏注意力保留比例
    ).to(device)
    
    # 冻结GAT (如果需要)
    if args.freeze_gat:
        print("  → 冻结GAT参数")
        for param in policy_net.gat.parameters():
            param.requires_grad = False
    
    total_params = sum(p.numel() for p in policy_net.parameters())
    trainable_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
    print(f"✓ 总参数: {total_params:,}, 可训练: {trainable_params:,}")
    print(f"✓ Cross-Attention: 4 heads, sparse_topk=0.3")
    print(f"✓ Context Vector: 256维 (vs V3的512维)")
    print(f"✓ 所有V3的辅助损失函数已保留")
    
    return policy_net


def main():
    """V4主训练循环 - 沿用V3的训练逻辑"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TEDG-RL V4 训练脚本 - Cross-Attention MoE")
    parser.add_argument("--exp-name", type=str, default="v4_full", help="实验名称")
    parser.add_argument("--episodes", type=int, default=10000, help="训练episodes数")
    parser.add_argument("--max-steps", type=int, default=2000, help="每episode最大步数")
    parser.add_argument("--no-mask", action="store_true", help="禁用动作掩码")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径")
    parser.add_argument("--freeze-gat", action="store_true", help="冻结GAT参数")
    parser.add_argument("--num-experts", type=int, default=4, help="专家数量")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"TEDG-RL V4 训练启动 - Cross-Attention Guided Hierarchical MoE")
    print(f"{'='*70}")
    print(f"实验名称: {args.exp_name}")
    print(f"训练配置: {args.episodes} episodes, {args.max_steps} steps/episode")
    print(f"专家数量: {args.num_experts}")
    print(f"冻结GAT: {args.freeze_gat}")
    print(f"\n🆕 V4新特性:")
    print(f"  - Cross-Attention融合 (替代V3的concat)")
    print(f"  - Sparse Attention Gate (稀疏注意力)")
    print(f"  - Context Vector (256维紧凑表示)")
    
    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(f"ablation_v4/results/{args.exp_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # 加载超图
    print(f"\n[加载超图数据]")
    from src.core.state_constructor import StateConstructor
    from src.core.hypergraph_matcher import HypergraphMatcher
    
    state_constructor = StateConstructor("data/hypergraph/hypergraph_complete_real.json")
    matcher = HypergraphMatcher(
        state_constructor.hypergraph,
        weights=(0.35, 0.35, 0.2, 0.1),
        tau=200.0
    )
    print(f"✓ 超图加载完成: {len(matcher.edges)} 条超边")
    
    # 创建环境
    print(f"\n[创建NetHack环境]")
    import gymnasium as gym
    try:
        env = gym.make("NetHackScore-v0")
    except:
        env = gym.make("NetHack-v0")
    print(f"✓ 动作空间: {env.action_space.n}个动作")
    
    # 创建V4网络
    policy_net = create_v4_policy(device, args)
    
    # 加载超图结构以获取operator_names（用于Manager约束）
    print(f"\n[加载超图结构用于Manager约束]")
    import json
    with open("data/hypergraph/hypergraph_gat_structure.json", 'r') as f:
        hypergraph_structure = json.load(f)
    operator_names = [node['label'] for node in hypergraph_structure['nodes'] if node['type'] == 'operator']
    print(f"✓ 提取了 {len(operator_names)} 个Operator节点")
    
    # 创建优化器
    import torch.optim as optim
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    
    # 学习率调度器
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=1000, max_steps=args.episodes*args.max_steps)
    
    # 创建训练器 (沿用V3配置)
    from src.core.ppo_trainer import PPOTrainer
    trainer = PPOTrainer(
        policy_net=policy_net,
        learning_rate=1e-4,
        clip_ratio=0.15,
        gamma=0.995,
        gae_lambda=0.97,
        ppo_epochs=4,
        batch_size=256,
        device=device,
        alpha_entropy_coef=0.05,
    )
    
    # 初始化监控器
    monitor = TrainingMonitor(log_interval=50)
    nan_detector = NaNDetector(policy_net)
    reward_normalizer = RewardNormalizer(clip_range=10.0)
    
    # 动作掩码
    from src.core.action_masking import ActionMasker
    action_masker = ActionMasker(state_constructor.hypergraph, num_actions=23)
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    best_reward = float("-inf")
    best_score = 0
    start_episode = 0
    
    # 恢复checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"\n[恢复checkpoint: {args.resume}]")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        policy_net.load_state_dict(checkpoint["policy_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_episode = checkpoint.get("episode", 0) + 1
        best_reward = checkpoint.get("best_reward", float("-inf"))
        best_score = checkpoint.get("best_score", 0)
        print(f"✓ 从Episode {start_episode}继续, Best Score: {best_score}")
    
    print(f"\n{'='*70}")
    print(f"开始训练")
    print(f"{'='*70}\n")
    
    import time
    start_time = time.time()
    global_step = 0
    
    # ========================================================================
    # 主训练循环 (完整沿用V3逻辑)
    # ========================================================================
    
    import nle.nethack as nh
    
    for episode in range(start_episode, args.episodes):
        # 获取当前阶段配置
        config = get_training_config(episode)
        
        # 更新网络配置
        policy_net.use_sparsemax = config['use_sparsemax']
        
        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['learning_rate']
        
        # 打印阶段信息
        if episode in [0, 1000, 3000]:
            print(f"\n{'='*70}")
            print(f"进入 {config['phase'].upper()} 阶段 (Episode {episode})")
            print(f"  - 路由方式: {'Sparsemax' if config['use_sparsemax'] else 'Softmax'}")
            print(f"  - 学习率: {config['learning_rate']}")
            print(f"  - 负载均衡系数: {config['load_balance_coef']}")
            print(f"{'='*70}\n")
        
        # 重置环境
        obs, info = env.reset()
        state, atoms = extract_state_from_obs(obs, state_constructor, matcher, t_now=0)
        
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        # Episode内的统计
        episode_alphas = []
        episode_expert_logits = []
        episode_alignment_losses = []
        episode_semantic_losses = []
        episode_temporal_losses = []
        episode_overlap_losses = []
        
        # 时间一致性追踪
        last_alpha = None
        
        # ====================================================================
        # Episode循环
        # ====================================================================
        
        while not (done or truncated) and steps < args.max_steps:
            # 获取动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # V4前向传播 (使用新的forward接口)
                outputs = policy_net.forward(
                    state_tensor,
                    atoms=atoms,
                    return_expert_logits=True
                )
                
                logits = outputs['policy_logits']
                alpha = outputs['alpha']
                value = outputs['value']
                
                # 记录统计信息
                episode_alphas.append(alpha.cpu())
                if 'expert_logits' in outputs and outputs['expert_logits'] is not None:
                    episode_expert_logits.append(outputs['expert_logits'].cpu())
                
                # 应用动作掩码
                action_mask = None
                if not args.no_mask:
                    action_mask = action_masker.get_action_mask(
                        atoms["pre_nodes"],
                        atoms["scene_atoms"],
                        0.5  # confidence
                    )
                    mask_t = torch.as_tensor(action_mask, device=logits.device, dtype=torch.bool)
                    masked_logits = logits.masked_fill(~mask_t, float("-inf"))
                    
                    # 兜底: 避免全-inf
                    if not torch.isfinite(masked_logits).any():
                        masked_logits = logits
                        action_mask = np.ones(23, dtype=bool)
                    
                    logits = masked_logits
                
                # 数值稳定性
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                
                # 采样动作
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action.item())
            
            # 奖励归一化
            reward_normalizer.update(reward)
            normalized_reward = reward_normalizer.normalize(reward)
            
            total_reward += reward
            steps += 1
            global_step += 1
            
            # 提取下一状态
            next_state, next_atoms = extract_state_from_obs(obs, state_constructor, matcher, t_now=steps)
            
            # 存储经验
            trainer.buffer.add(
                state=state,
                action=action.item(),
                reward=normalized_reward,
                next_state=next_state,
                done=done or truncated,
                log_prob=log_prob.item(),
                action_mask=action_mask,
            )
            
            state = next_state
            atoms = next_atoms
            
            # 更新网络
            if len(trainer.buffer) >= trainer.batch_size:
                # 保存checkpoint (用于NaN回滚)
                nan_detector.save_checkpoint()
                
                # 采样批次
                batch = trainer.buffer.sample_batch(trainer.batch_size)
                states = batch['states'].to(device)
                actions = batch['actions'].to(device)
                rewards = batch['rewards'].to(device)
                dones = batch['dones'].to(device)
                old_log_probs = batch['old_log_probs'].to(device)
                
                # 计算GAE优势
                with torch.no_grad():
                    old_outputs = policy_net.forward(states, atoms=atoms)
                    old_values = old_outputs['value'].squeeze(-1)
                
                advantages, returns = trainer.compute_gae_advantages(rewards, old_values, dones)
                
                # 归一化优势
                adv_mean = advantages.mean()
                adv_std = advantages.cpu().std().to(advantages.device) if advantages.is_cuda or str(advantages.device).startswith('musa') else advantages.std()
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                
                # PPO更新循环
                for ppo_epoch in range(trainer.ppo_epochs):
                    # 前向传播
                    outputs = policy_net.forward(
                        states,
                        atoms=atoms,
                        return_expert_logits=True
                    )
                    
                    logits = outputs['policy_logits']
                    alpha = outputs['alpha']
                    values = outputs['value'].squeeze(-1)
                    
                    # 计算新的对数概率
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy().mean()
                    
                    # 计算α熵
                    alpha_entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean()
                    
                    # PPO比率
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    # Actor损失
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - trainer.clip_ratio, 1 + trainer.clip_ratio) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Critic损失
                    critic_loss = F.mse_loss(values, returns)
                    
                    # 辅助损失
                    lb_loss = load_balance_loss(alpha, num_experts=args.num_experts)
                    
                    div_loss = torch.tensor(0.0, device=device)
                    if 'expert_logits' in outputs and outputs['expert_logits'] is not None:
                        div_loss = expert_diversity_loss(outputs['expert_logits'])
                    
                    # ===== Manager内层约束 =====
                    alignment_loss = torch.tensor(0.0, device=device)
                    semantic_loss = torch.tensor(0.0, device=device)
                    
                    if 'operator_scores' in outputs and outputs['operator_scores'] is not None and 'expert_logits' in outputs and outputs['expert_logits'] is not None:
                        # 1. 超图-路由对齐损失
                        alignment_loss = hypergraph_alignment_loss(
                            outputs['operator_scores'],
                            alpha,
                            operator_names,
                            temperature=config.get('alignment_temperature', 1.0)
                        )
                        
                        # 2. 增强的语义正交损失
                        semantic_loss = enhanced_semantic_orthogonality_loss(
                            outputs['expert_logits']
                        )
                    
                    # ===== 高级机制 =====
                    # 3. 时间一致性损失
                    temporal_loss = torch.tensor(0.0, device=device)
                    if last_alpha is not None and config.get('temporal_coef', 0.0) > 0:
                        temporal_loss = F.mse_loss(alpha, last_alpha)
                    
                    # 4. 专家重叠惩罚
                    overlap_loss = torch.tensor(0.0, device=device)
                    if 'expert_logits' in outputs and outputs['expert_logits'] is not None and config.get('overlap_coef', 0.0) > 0:
                        overlap_loss = expert_overlap_penalty(alpha, outputs['expert_logits'])
                    
                    # 总损失（包含所有约束）
                    total_loss = (
                        actor_loss +
                        0.5 * critic_loss -
                        config['entropy_coef'] * entropy +
                        config.get('alpha_entropy_sign', -1) * config['alpha_entropy_coef'] * alpha_entropy +
                        config['load_balance_coef'] * lb_loss +
                        config['diversity_coef'] * div_loss +
                        config.get('alignment_coef', 0.1) * alignment_loss +
                        config.get('semantic_coef', 0.05) * semantic_loss +
                        config.get('temporal_coef', 0.0) * temporal_loss +
                        config.get('overlap_coef', 0.0) * overlap_loss
                    )
                    
                    # NaN检测
                    if nan_detector.check_and_rollback(total_loss):
                        break
                    
                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    # 梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                    
                    # 更新
                    optimizer.step()
                    lr_scheduler.step()
                
                # 记录Manager约束损失
                if alignment_loss.item() > 0:
                    episode_alignment_losses.append(alignment_loss.item())
                    episode_semantic_losses.append(semantic_loss.item())
                
                # 记录高级机制损失
                if temporal_loss.item() > 0:
                    episode_temporal_losses.append(temporal_loss.item())
                if overlap_loss.item() > 0:
                    episode_overlap_losses.append(overlap_loss.item())
                
                # 更新last_alpha
                last_alpha = alpha.detach()
                
                # 清空缓冲区
                trainer.buffer.clear()
        
        # ====================================================================
        # Episode结束统计
        # ====================================================================
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        final_score = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)[nh.NLE_BL_SCORE]
        episode_scores.append(final_score)
        
        # 计算专家统计
        if episode_alphas:
            alpha_tensor = torch.stack(episode_alphas, dim=0)
            mean_alpha = alpha_tensor.mean(dim=0).numpy()
            alpha_entropy_val = -(mean_alpha * np.log(mean_alpha + 1e-8)).sum()
            expert_usage_variance = alpha_tensor.var(dim=0).mean().item()
        else:
            alpha_entropy_val = 0.0
            expert_usage_variance = 0.0
        
        # 记录监控指标
        monitor.log(episode, {
            'episode_score': final_score,
            'episode_reward': total_reward,
            'episode_length': steps,
            'alpha_entropy': alpha_entropy_val,
            'expert_usage_variance': expert_usage_variance,
            'gradient_norm': grad_norm.item() if 'grad_norm' in locals() else 0.0,
        })
        
        # 更新最佳记录
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({
                "policy_net": policy_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode,
                "best_reward": best_reward,
                "best_score": final_score,
                "config": vars(args)
            }, output_dir / "checkpoints" / "best_model.pth")
        
        if final_score > best_score:
            best_score = final_score
        
        # 定期保存checkpoint
        if (episode + 1) % 100 == 0:
            torch.save({
                "policy_net": policy_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode,
                "reward": total_reward,
                "score": final_score,
                "config": vars(args)
            }, output_dir / "checkpoints" / f"model_{episode+1:05d}.pth")
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_score = np.mean(episode_scores[-10:])
            print(f"Episode {episode+1}/{args.episodes} | "
                  f"Score: {final_score} | "
                  f"Reward: {total_reward:.2f} | "
                  f"Steps: {steps} | "
                  f"α_entropy: {alpha_entropy_val:.3f} | "
                  f"Phase: {config['phase']}")
            
            # 打印Manager约束的loss
            if episode_alignment_losses:
                avg_alignment = np.mean(episode_alignment_losses)
                avg_semantic = np.mean(episode_semantic_losses)
                print(f"  → Manager Constraints: "
                      f"Alignment={avg_alignment:.4f}, "
                      f"Semantic={avg_semantic:.4f}")
            
            # 打印高级机制的loss
            if episode_temporal_losses or episode_overlap_losses:
                losses_str = "  → Advanced Mechanisms: "
                if episode_temporal_losses:
                    avg_temporal = np.mean(episode_temporal_losses)
                    losses_str += f"Temporal={avg_temporal:.4f}, "
                if episode_overlap_losses:
                    avg_overlap = np.mean(episode_overlap_losses)
                    losses_str += f"Overlap={avg_overlap:.4f}"
                print(losses_str)
    
    
    # ========================================================================
    # 训练结束
    # ========================================================================
    
    # 保存最终模型
    torch.save({
        "policy_net": policy_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": args.episodes,
        "best_reward": best_reward,
        "best_score": best_score,
        "config": vars(args)
    }, output_dir / "checkpoints" / "model_final.pth")
    
    # 保存训练日志
    training_log = {
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_lengths": [int(l) for l in episode_lengths],
        "episode_scores": [int(s) for s in episode_scores],
        "best_reward": float(best_reward),
        "best_score": int(best_score),
        "config": vars(args),
        "monitor_metrics": {k: [float(v) for v in vals] for k, vals in monitor.metrics.items()},
    }
    
    with open(output_dir / "logs" / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    
    # 打印最终统计
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"训练完成")
    print(f"{'='*70}")
    print(f"总耗时: {elapsed_time/3600:.2f} 小时")
    print(f"最佳奖励: {best_reward:.2f}")
    print(f"最佳分数: {best_score}")
    print(f"平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"平均分数: {np.mean(episode_scores):.1f}")
    print(f"结果保存在: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
