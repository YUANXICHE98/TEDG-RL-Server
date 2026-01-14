#!/usr/bin/env python3
"""
分析Manager内层约束的效果

功能:
1. 对比有无Manager约束的训练曲线
2. 计算对齐度（Router与GAT的一致性）
3. 可视化Alpha熵演化
4. 分析专家专业化程度
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

# 添加项目根目录到路径
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.networks_v3_gat_moe import GATGuidedMoEPolicy
from src.core.state_constructor import StateConstructor
from src.core.hypergraph_matcher import HypergraphMatcher
from src.core.operator_expert_mapping import OPERATOR_TO_EXPERT, EXPERT_NAMES


def load_training_log(exp_dir: Path) -> Dict:
    """加载训练日志"""
    log_file = exp_dir / "logs" / "training_log.json"
    if not log_file.exists():
        raise FileNotFoundError(f"训练日志不存在: {log_file}")
    
    with open(log_file, 'r') as f:
        return json.load(f)


def extract_alpha_entropy(log: Dict) -> List[float]:
    """提取Alpha熵序列"""
    if 'monitor_metrics' in log and 'alpha_entropy' in log['monitor_metrics']:
        return log['monitor_metrics']['alpha_entropy']
    return []


def extract_scores(log: Dict) -> List[int]:
    """提取分数序列"""
    return log.get('episode_scores', [])


def compute_moving_average(data: List[float], window: int = 50) -> List[float]:
    """计算移动平均"""
    if len(data) < window:
        return data
    
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(np.mean(data[start:i+1]))
    
    return result


def plot_comparison(
    baseline_log: Dict,
    manager_log: Dict,
    output_dir: Path
):
    """绘制对比图"""
    
    # 提取数据
    baseline_entropy = extract_alpha_entropy(baseline_log)
    manager_entropy = extract_alpha_entropy(manager_log)
    
    baseline_scores = extract_scores(baseline_log)
    manager_scores = extract_scores(manager_log)
    
    # 计算移动平均
    baseline_entropy_ma = compute_moving_average(baseline_entropy, window=50)
    manager_entropy_ma = compute_moving_average(manager_entropy, window=50)
    
    baseline_scores_ma = compute_moving_average(baseline_scores, window=50)
    manager_scores_ma = compute_moving_average(manager_scores, window=50)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Alpha熵对比
    ax = axes[0, 0]
    if baseline_entropy:
        ax.plot(baseline_entropy_ma, label='Baseline', alpha=0.7, linewidth=2)
    if manager_entropy:
        ax.plot(manager_entropy_ma, label='With Manager Constraints', alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Alpha Entropy', fontsize=12)
    ax.set_title('Alpha Entropy Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加阶段分隔线
    if manager_entropy:
        max_episodes = len(manager_entropy)
        if max_episodes >= 1000:
            ax.axvline(x=1000, color='red', linestyle='--', alpha=0.5, label='Transition Start')
        if max_episodes >= 3000:
            ax.axvline(x=3000, color='green', linestyle='--', alpha=0.5, label='Fine-tune Start')
    
    # 2. 分数对比
    ax = axes[0, 1]
    if baseline_scores:
        ax.plot(baseline_scores_ma, label='Baseline', alpha=0.7, linewidth=2)
    if manager_scores:
        ax.plot(manager_scores_ma, label='With Manager Constraints', alpha=0.7, linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Score Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 3. Alpha熵改进率
    ax = axes[1, 0]
    if baseline_entropy and manager_entropy:
        min_len = min(len(baseline_entropy_ma), len(manager_entropy_ma))
        improvement = [
            (baseline_entropy_ma[i] - manager_entropy_ma[i]) / baseline_entropy_ma[i] * 100
            for i in range(min_len)
            if baseline_entropy_ma[i] > 0
        ]
        ax.plot(improvement, color='purple', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title('Alpha Entropy Improvement', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 显示平均改进
        avg_improvement = np.mean(improvement[-100:]) if len(improvement) >= 100 else np.mean(improvement)
        ax.text(0.02, 0.98, f'Avg Improvement (last 100): {avg_improvement:.1f}%',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. 分数改进率
    ax = axes[1, 1]
    if baseline_scores and manager_scores:
        min_len = min(len(baseline_scores_ma), len(manager_scores_ma))
        improvement = [
            (manager_scores_ma[i] - baseline_scores_ma[i]) / max(baseline_scores_ma[i], 1) * 100
            for i in range(min_len)
        ]
        ax.plot(improvement, color='green', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Improvement (%)', fontsize=12)
        ax.set_title('Score Improvement', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 显示平均改进
        avg_improvement = np.mean(improvement[-100:]) if len(improvement) >= 100 else np.mean(improvement)
        ax.text(0.02, 0.98, f'Avg Improvement (last 100): {avg_improvement:.1f}%',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存
    output_file = output_dir / "manager_constraint_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 对比图已保存: {output_file}")
    
    plt.close()


def compute_alignment_score(
    checkpoint_path: Path,
    num_episodes: int = 10,
    max_steps: int = 500
) -> Dict:
    """
    计算对齐度（Router输出与GAT建议的一致性）
    
    Returns:
        {
            'mean_alignment': float,  # 平均对齐度
            'std_alignment': float,   # 标准差
            'alignment_per_episode': List[float]
        }
    """
    import gymnasium as gym
    import nle.env
    import nle.nethack as nh
    
    # 加载模型
    device = torch.device('cpu')
    policy_net = GATGuidedMoEPolicy(
        hypergraph_path="data/hypergraph/hypergraph_gat_structure.json",
        state_dim=115,
        hidden_dim=256,
        action_dim=23,
        num_experts=4,
        use_sparsemax=True
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy_net.load_state_dict(checkpoint['policy_net'])
    policy_net.eval()
    
    # 加载超图
    state_constructor = StateConstructor("data/hypergraph/hypergraph_complete_real.json")
    matcher = HypergraphMatcher(
        state_constructor.hypergraph,
        weights=(0.35, 0.35, 0.2, 0.1),
        tau=200.0
    )
    
    # 加载operator_names
    with open("data/hypergraph/hypergraph_gat_structure.json", 'r') as f:
        hypergraph_structure = json.load(f)
    operator_names = [node['label'] for node in hypergraph_structure['nodes'] if node['type'] == 'operator']
    
    # 创建环境
    try:
        env = gym.make("NetHackScore-v0")
    except:
        env = gym.make("NetHack-v0")
    
    alignment_per_episode = []
    
    print(f"\n计算对齐度 ({num_episodes} episodes)...")
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0
        
        episode_alignments = []
        
        while not (done or truncated) and steps < max_steps:
            # 提取状态
            from ablation_v3.train.train_v3_gat_moe import extract_state_from_obs
            state, atoms = extract_state_from_obs(obs, state_constructor, matcher, t_now=steps)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # 前向传播
                logits, alpha, value, aux_info = policy_net(state_tensor, atoms=atoms)
                
                # 计算target_alpha（GAT建议）
                if aux_info['operator_scores'] is not None:
                    from ablation_v3.train.train_v3_gat_moe import aggregate_operators_to_experts
                    expert_scores = aggregate_operators_to_experts(
                        aux_info['operator_scores'],
                        operator_names,
                        num_experts=4
                    )
                    target_alpha = torch.softmax(expert_scores, dim=-1)
                    
                    # 计算余弦相似度
                    alignment = torch.nn.functional.cosine_similarity(
                        alpha, target_alpha, dim=-1
                    ).item()
                    
                    episode_alignments.append(alignment)
            
            # 执行动作
            action = torch.distributions.Categorical(logits=logits).sample()
            obs, reward, done, truncated, info = env.step(action.item())
            steps += 1
        
        if episode_alignments:
            mean_alignment = np.mean(episode_alignments)
            alignment_per_episode.append(mean_alignment)
            print(f"  Episode {ep+1}/{num_episodes}: Alignment = {mean_alignment:.4f}")
    
    env.close()
    
    # 统计
    result = {
        'mean_alignment': np.mean(alignment_per_episode),
        'std_alignment': np.std(alignment_per_episode),
        'alignment_per_episode': alignment_per_episode
    }
    
    print(f"\n对齐度统计:")
    print(f"  平均: {result['mean_alignment']:.4f}")
    print(f"  标准差: {result['std_alignment']:.4f}")
    print(f"  最小: {np.min(alignment_per_episode):.4f}")
    print(f"  最大: {np.max(alignment_per_episode):.4f}")
    
    return result


def print_summary(baseline_log: Dict, manager_log: Dict):
    """打印统计摘要"""
    
    print(f"\n{'='*70}")
    print("Manager内层约束效果摘要")
    print(f"{'='*70}\n")
    
    # Alpha熵
    baseline_entropy = extract_alpha_entropy(baseline_log)
    manager_entropy = extract_alpha_entropy(manager_log)
    
    if baseline_entropy and manager_entropy:
        baseline_final = np.mean(baseline_entropy[-100:])
        manager_final = np.mean(manager_entropy[-100:])
        improvement = (baseline_final - manager_final) / baseline_final * 100
        
        print(f"Alpha熵 (最后100 episodes):")
        print(f"  Baseline:        {baseline_final:.4f}")
        print(f"  With Manager:    {manager_final:.4f}")
        print(f"  Improvement:     {improvement:.1f}%")
        print()
    
    # 分数
    baseline_scores = extract_scores(baseline_log)
    manager_scores = extract_scores(manager_log)
    
    if baseline_scores and manager_scores:
        baseline_final = np.mean(baseline_scores[-100:])
        manager_final = np.mean(manager_scores[-100:])
        improvement = (manager_final - baseline_final) / baseline_final * 100
        
        print(f"平均分数 (最后100 episodes):")
        print(f"  Baseline:        {baseline_final:.2f}")
        print(f"  With Manager:    {manager_final:.2f}")
        print(f"  Improvement:     {improvement:.1f}%")
        print()
    
    # 训练episodes
    baseline_episodes = len(baseline_scores)
    manager_episodes = len(manager_scores)
    
    print(f"训练Episodes:")
    print(f"  Baseline:        {baseline_episodes}")
    print(f"  With Manager:    {manager_episodes}")
    print()
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="分析Manager内层约束的效果")
    parser.add_argument("--baseline", type=str, required=True,
                        help="Baseline实验目录（无Manager约束）")
    parser.add_argument("--manager", type=str, required=True,
                        help="Manager实验目录（有Manager约束）")
    parser.add_argument("--output", type=str, default="ablation_v3/visualizations",
                        help="输出目录")
    parser.add_argument("--compute-alignment", action="store_true",
                        help="计算对齐度（需要运行环境）")
    parser.add_argument("--alignment-episodes", type=int, default=10,
                        help="计算对齐度的episodes数")
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载训练日志
    print("加载训练日志...")
    baseline_log = load_training_log(Path(args.baseline))
    manager_log = load_training_log(Path(args.manager))
    print("✓ 日志加载完成")
    
    # 绘制对比图
    print("\n绘制对比图...")
    plot_comparison(baseline_log, manager_log, output_dir)
    
    # 打印摘要
    print_summary(baseline_log, manager_log)
    
    # 计算对齐度（可选）
    if args.compute_alignment:
        manager_checkpoint = Path(args.manager) / "checkpoints" / "model_final.pth"
        if manager_checkpoint.exists():
            print("\n计算对齐度...")
            alignment_result = compute_alignment_score(
                manager_checkpoint,
                num_episodes=args.alignment_episodes
            )
            
            # 保存结果
            result_file = output_dir / "alignment_score.json"
            with open(result_file, 'w') as f:
                json.dump(alignment_result, f, indent=2)
            print(f"✓ 对齐度结果已保存: {result_file}")
        else:
            print(f"⚠️ Checkpoint不存在: {manager_checkpoint}")
    
    print(f"\n✓ 分析完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
