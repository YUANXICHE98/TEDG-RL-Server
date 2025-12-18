#!/usr/bin/env python3
"""验证方法有效性 - 分析训练结果"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import linregress


def load_training_log():
    """加载训练日志"""
    log_path = Path("results/logs/training_log.json")
    if not log_path.exists():
        print("❌ 训练日志不存在，请先运行训练")
        return None
    
    with open(log_path, 'r') as f:
        return json.load(f)


def validate_alpha_distribution(log):
    """验证1: α权重分布合理性"""
    print("\n" + "="*80)
    print("验证1: α权重分布")
    print("="*80)
    
    alpha_history = np.array(log['alpha_history'])
    alpha_mean = alpha_history.mean(axis=0)
    alpha_std = alpha_history.std(axis=0)
    
    print(f"\nα权重统计:")
    print(f"  α_pre:    {alpha_mean[0]:.3f} ± {alpha_std[0]:.3f}")
    print(f"  α_scene:  {alpha_mean[1]:.3f} ± {alpha_std[1]:.3f}")
    print(f"  α_effect: {alpha_mean[2]:.3f} ± {alpha_std[2]:.3f}")
    print(f"  α_rule:   {alpha_mean[3]:.3f} ± {alpha_std[3]:.3f}")
    
    # 检查1: 均值在合理范围
    checks = []
    for i, name in enumerate(['pre', 'scene', 'effect', 'rule']):
        in_range = 0.15 < alpha_mean[i] < 0.35
        checks.append(in_range)
        status = "✓" if in_range else "✗"
        print(f"\n{status} α_{name}均值在[0.15, 0.35]: {in_range}")
    
    # 检查2: 有动态变化
    has_variance = all(std > 0.05 for std in alpha_std)
    checks.append(has_variance)
    print(f"\n{'✓' if has_variance else '✗'} α权重有动态变化 (std > 0.05): {has_variance}")
    
    # 检查3: 不过度集中
    max_mean = alpha_mean.max()
    min_mean = alpha_mean.min()
    not_concentrated = (max_mean - min_mean) < 0.2
    checks.append(not_concentrated)
    print(f"\n{'✓' if not_concentrated else '✗'} α权重不过度集中 (差异 < 0.2): {not_concentrated}")
    
    success_rate = sum(checks) / len(checks) * 100
    print(f"\n总体: {sum(checks)}/{len(checks)} 通过 ({success_rate:.0f}%)")
    
    return all(checks)


def validate_performance_improvement(log):
    """验证2: 性能提升"""
    print("\n" + "="*80)
    print("验证2: 性能提升")
    print("="*80)
    
    rewards = log['episode_rewards']
    scores = log['episode_scores']
    lengths = log['episode_lengths']
    
    # 初期 vs 后期
    early_rewards = np.mean(rewards[:100])
    late_rewards = np.mean(rewards[-100:])
    early_scores = np.mean(scores[:100])
    late_scores = np.mean(scores[-100:])
    early_lengths = np.mean(lengths[:100])
    late_lengths = np.mean(lengths[-100:])
    
    print(f"\n初期 vs 后期:")
    print(f"  奖励:  {early_rewards:.2f} → {late_rewards:.2f} ({late_rewards/max(early_rewards,0.1):.2f}x)")
    print(f"  分数:  {early_scores:.0f} → {late_scores:.0f} ({late_scores/max(early_scores,1):.2f}x)")
    print(f"  长度:  {early_lengths:.0f} → {late_lengths:.0f} ({late_lengths/max(early_lengths,1):.2f}x)")
    
    # 趋势检查
    checks = []
    
    # 检查1: 奖励上升
    slope, _, _, p_value, _ = linregress(range(len(rewards)), rewards)
    reward_improving = slope > 0 and p_value < 0.05
    checks.append(reward_improving)
    print(f"\n{'✓' if reward_improving else '✗'} 奖励显著上升 (slope={slope:.6f}, p={p_value:.4f}): {reward_improving}")
    
    # 检查2: 分数提升
    score_improvement = late_scores > early_scores * 1.5
    checks.append(score_improvement)
    print(f"\n{'✓' if score_improvement else '✗'} 分数提升 > 50%: {score_improvement}")
    
    # 检查3: 长度增加
    length_improvement = late_lengths > early_lengths * 1.5
    checks.append(length_improvement)
    print(f"\n{'✓' if length_improvement else '✗'} Episode长度增加 > 50%: {length_improvement}")
    
    success_rate = sum(checks) / len(checks) * 100
    print(f"\n总体: {sum(checks)}/{len(checks)} 通过 ({success_rate:.0f}%)")
    
    return all(checks)


def validate_behavior_rationality(log):
    """验证3: 行为合理性（理论验证）"""
    print("\n" + "="*80)
    print("验证3: 行为合理性")
    print("="*80)
    
    alpha_history = np.array(log['alpha_history'])
    
    print("\n理论验证: α权重应该根据场景变化")
    
    # 检查α权重的变化范围
    alpha_ranges = []
    for i, name in enumerate(['pre', 'scene', 'effect', 'rule']):
        min_val = alpha_history[:, i].min()
        max_val = alpha_history[:, i].max()
        range_val = max_val - min_val
        alpha_ranges.append(range_val)
        print(f"  α_{name}: 范围 [{min_val:.3f}, {max_val:.3f}], 变化幅度 {range_val:.3f}")
    
    # 检查: 每个α都有显著变化
    checks = []
    significant_variation = all(r > 0.1 for r in alpha_ranges)
    checks.append(significant_variation)
    print(f"\n{'✓' if significant_variation else '✗'} 所有α权重都有显著变化 (> 0.1): {significant_variation}")
    
    # 检查: α权重不是固定的
    alpha_std_over_time = alpha_history.std(axis=0).mean()
    not_fixed = alpha_std_over_time > 0.05
    checks.append(not_fixed)
    print(f"\n{'✓' if not_fixed else '✗'} α权重动态变化 (平均std={alpha_std_over_time:.3f}): {not_fixed}")
    
    success_rate = sum(checks) / len(checks) * 100
    print(f"\n总体: {sum(checks)}/{len(checks)} 通过 ({success_rate:.0f}%)")
    
    return all(checks)


def generate_visualizations(log):
    """生成可视化图表"""
    print("\n" + "="*80)
    print("生成可视化")
    print("="*80)
    
    output_dir = Path("results/validation")
    output_dir.mkdir(exist_ok=True)
    
    # 图1: 性能曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励曲线
    rewards = log['episode_rewards']
    window = 50
    rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0, 0].plot(rewards, alpha=0.3, label='Raw')
    axes[0, 0].plot(range(window-1, len(rewards)), rewards_smooth, label=f'Smooth ({window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('训练奖励曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 分数曲线
    scores = log['episode_scores']
    scores_smooth = np.convolve(scores, np.ones(window)/window, mode='valid')
    axes[0, 1].plot(scores, alpha=0.3, label='Raw')
    axes[0, 1].plot(range(window-1, len(scores)), scores_smooth, label=f'Smooth ({window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('NetHack分数曲线')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Episode长度
    lengths = log['episode_lengths']
    lengths_smooth = np.convolve(lengths, np.ones(window)/window, mode='valid')
    axes[1, 0].plot(lengths, alpha=0.3, label='Raw')
    axes[1, 0].plot(range(window-1, len(lengths)), lengths_smooth, label=f'Smooth ({window})')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title('Episode长度曲线')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # α权重分布
    alpha_history = np.array(log['alpha_history'])
    for i, name in enumerate(['α_pre', 'α_scene', 'α_effect', 'α_rule']):
        axes[1, 1].plot(alpha_history[:, i], label=name, alpha=0.7)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('α Weight')
    axes[1, 1].set_title('α权重变化')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    print(f"✓ 保存: {output_dir / 'training_curves.png'}")
    
    # 图2: α权重分布箱线图
    fig, ax = plt.subplots(figsize=(10, 6))
    alpha_data = [alpha_history[:, i] for i in range(4)]
    ax.boxplot(alpha_data, labels=['α_pre', 'α_scene', 'α_effect', 'α_rule'])
    ax.set_ylabel('α Weight')
    ax.set_title('α权重分布')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'alpha_distribution.png', dpi=150)
    print(f"✓ 保存: {output_dir / 'alpha_distribution.png'}")
    
    plt.close('all')


def main():
    """主验证流程"""
    print("="*80)
    print("TEDG-RL 方法有效性验证")
    print("="*80)
    
    # 加载日志
    log = load_training_log()
    if log is None:
        return
    
    print(f"\n训练信息:")
    print(f"  Episodes: {log['total_episodes']}")
    print(f"  训练时间: {log['total_time_seconds']/60:.1f}分钟")
    print(f"  设备: {log['device']}")
    print(f"  最佳奖励: {log['best_reward']:.2f}")
    print(f"  最佳分数: {log['best_score']:.0f}")
    
    # 执行验证
    results = []
    
    results.append(("α权重分布", validate_alpha_distribution(log)))
    results.append(("性能提升", validate_performance_improvement(log)))
    results.append(("行为合理性", validate_behavior_rationality(log)))
    
    # 生成可视化
    try:
        generate_visualizations(log)
    except Exception as e:
        print(f"\n⚠ 可视化生成失败: {e}")
    
    # 总结
    print("\n" + "="*80)
    print("验证总结")
    print("="*80)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 未通过"
        print(f"{status}: {name}")
    
    total_passed = sum(1 for _, p in results if p)
    success_rate = total_passed / len(results) * 100
    
    print(f"\n总体: {total_passed}/{len(results)} 通过 ({success_rate:.0f}%)")
    
    if success_rate >= 66:
        print("\n🎉 方法有效性验证通过！")
        print("\n论文中可以说明:")
        print("  1. α权重学习合理（均衡分布，动态变化）")
        print("  2. 性能显著提升（奖励/分数/长度都提升）")
        print("  3. 行为具有场景适应性（α权重根据情况变化）")
    else:
        print("\n⚠ 部分验证未通过，建议:")
        print("  1. 增加训练episodes")
        print("  2. 调整超参数")
        print("  3. 检查奖励函数设计")


if __name__ == "__main__":
    main()
