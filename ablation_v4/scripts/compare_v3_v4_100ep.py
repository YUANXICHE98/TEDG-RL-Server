#!/usr/bin/env python3
"""
V3 vs V4 小规模对比分析 (100 episodes)
Compare V3 (concat fusion) vs V4 (cross-attention fusion)

用于快速验证V4的Cross-Attention机制是否优于V3的Concat融合
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


def load_training_log(log_path, max_episodes=None):
    """加载训练日志
    
    Args:
        log_path: 日志文件路径
        max_episodes: 最多加载多少个episode (None表示全部)
    """
    log_path = Path(log_path)
    
    if not log_path.exists():
        print(f"⚠️  Warning: {log_path} not found")
        return None
    
    with open(log_path, 'r') as f:
        data = json.load(f)
    
    # 提取数据
    rewards = data.get('episode_rewards', [])
    scores = data.get('episode_scores', [])
    lengths = data.get('episode_lengths', [])
    monitor = data.get('monitor_metrics', {})
    
    # 如果指定了max_episodes，只取前N个
    if max_episodes is not None:
        rewards = rewards[:max_episodes]
        scores = scores[:max_episodes]
        lengths = lengths[:max_episodes]
        
        # 也截取monitor数据
        if monitor:
            for key in monitor:
                if isinstance(monitor[key], list):
                    monitor[key] = monitor[key][:max_episodes]
    
    return {
        'rewards': rewards,
        'scores': scores,
        'lengths': lengths,
        'monitor': monitor,
    }


def analyze_comparison(v3_data, v4_data):
    """对比分析V3和V4"""
    print("\n" + "="*80)
    print("📊 V3 vs V4 COMPARISON ANALYSIS (100 Episodes)")
    print("="*80)
    
    # 基本统计
    print("\n📈 Performance Metrics:")
    print(f"\n{'Metric':<25} {'V3 (Concat)':<20} {'V4 (CrossAttn)':<20} {'Improvement':<15}")
    print("-" * 80)
    
    metrics = [
        ('Avg Reward', 'rewards'),
        ('Avg Score', 'scores'),
        ('Best Reward', 'rewards'),
        ('Best Score', 'scores'),
        ('Avg Episode Length', 'lengths'),
    ]
    
    improvements = {}
    
    for metric_name, key in metrics:
        v3_values = v3_data[key]
        v4_values = v4_data[key]
        
        if 'Best' in metric_name:
            v3_val = np.max(v3_values)
            v4_val = np.max(v4_values)
        else:
            v3_val = np.mean(v3_values)
            v4_val = np.mean(v4_values)
        
        improvement = v4_val - v3_val
        pct_improvement = 100 * improvement / abs(v3_val) if v3_val != 0 else 0
        
        improvements[metric_name] = pct_improvement
        
        print(f"{metric_name:<25} {v3_val:>18.2f}  {v4_val:>18.2f}  {improvement:>+7.2f} ({pct_improvement:>+6.1f}%)")
    
    # Alpha Entropy分析
    print("\n🔍 Alpha Entropy Analysis:")
    v3_entropy = v3_data['monitor'].get('alpha_entropy', [])
    v4_entropy = v4_data['monitor'].get('alpha_entropy', [])
    
    if v3_entropy and v4_entropy:
        v3_avg_entropy = np.mean(v3_entropy)
        v4_avg_entropy = np.mean(v4_entropy)
        
        ln2 = 0.693
        ln4 = 1.386
        
        print(f"  V3 Avg Entropy: {v3_avg_entropy:.4f}")
        print(f"  V4 Avg Entropy: {v4_avg_entropy:.4f}")
        print(f"  Difference: {v4_avg_entropy - v3_avg_entropy:+.4f}")
        
        # 检查是否卡在ln(2)
        v3_near_ln2 = sum(abs(e - ln2) < 0.01 for e in v3_entropy)
        v4_near_ln2 = sum(abs(e - ln2) < 0.01 for e in v4_entropy)
        
        v3_pct_ln2 = 100 * v3_near_ln2 / len(v3_entropy)
        v4_pct_ln2 = 100 * v4_near_ln2 / len(v4_entropy)
        
        print(f"\n  Episodes near ln(2)=0.693:")
        print(f"    V3: {v3_near_ln2}/{len(v3_entropy)} ({v3_pct_ln2:.1f}%)")
        print(f"    V4: {v4_near_ln2}/{len(v4_entropy)} ({v4_pct_ln2:.1f}%)")
        
        if v4_pct_ln2 < v3_pct_ln2:
            print(f"    ✅ V4 reduced ln(2) stuck rate by {v3_pct_ln2 - v4_pct_ln2:.1f}%")
        elif v4_pct_ln2 > v3_pct_ln2:
            print(f"    ⚠️  V4 increased ln(2) stuck rate by {v4_pct_ln2 - v3_pct_ln2:.1f}%")
        else:
            print(f"    ➡️  No change in ln(2) stuck rate")
    
    # Expert Usage分析
    print("\n🎯 Expert Usage Variance:")
    v3_variance = v3_data['monitor'].get('expert_usage_variance', [])
    v4_variance = v4_data['monitor'].get('expert_usage_variance', [])
    
    if v3_variance and v4_variance:
        v3_avg_var = np.mean(v3_variance)
        v4_avg_var = np.mean(v4_variance)
        
        print(f"  V3 Avg Variance: {v3_avg_var:.4f}")
        print(f"  V4 Avg Variance: {v4_avg_var:.4f}")
        print(f"  Difference: {v4_avg_var - v3_avg_var:+.4f}")
        
        if v4_avg_var > v3_avg_var:
            print(f"    ✅ V4 has higher variance (better expert differentiation)")
        else:
            print(f"    ⚠️  V4 has lower variance (worse expert differentiation)")
    
    # 总结
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    
    positive_improvements = sum(1 for v in improvements.values() if v > 0)
    total_metrics = len(improvements)
    
    print(f"\nPositive Improvements: {positive_improvements}/{total_metrics}")
    
    if positive_improvements >= total_metrics * 0.7:
        print("\n✅ V4 shows SIGNIFICANT improvement over V3!")
        print("   Cross-Attention mechanism is working better than Concat fusion.")
    elif positive_improvements >= total_metrics * 0.5:
        print("\n➡️  V4 shows MODERATE improvement over V3.")
        print("   Cross-Attention has some benefits but not dramatic.")
    else:
        print("\n⚠️  V4 does NOT show clear improvement over V3.")
        print("   Cross-Attention may need further tuning.")
    
    return improvements


def create_comparison_plots(v3_data, v4_data, output_path):
    """创建对比可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('V3 (Concat) vs V4 (Cross-Attention) Comparison - 100 Episodes', 
                 fontsize=16, fontweight='bold')
    
    v3_rewards = v3_data['rewards']
    v4_rewards = v4_data['rewards']
    v3_scores = v3_data['scores']
    v4_scores = v4_data['scores']
    
    episodes = np.arange(len(v3_rewards))
    
    # Plot 1: Episode Rewards
    ax = axes[0, 0]
    ax.plot(episodes, v3_rewards, alpha=0.3, color='blue', label='V3 (Concat)')
    ax.plot(episodes, v4_rewards, alpha=0.3, color='red', label='V4 (CrossAttn)')
    
    # Smoothed
    window = 10
    if len(v3_rewards) > window:
        v3_smooth = np.convolve(v3_rewards, np.ones(window)/window, mode='valid')
        v4_smooth = np.convolve(v4_rewards, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(v3_rewards)), v3_smooth, 
                color='blue', linewidth=2, label=f'V3 Smoothed ({window}ep)')
        ax.plot(np.arange(window-1, len(v4_rewards)), v4_smooth, 
                color='red', linewidth=2, label=f'V4 Smoothed ({window}ep)')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Episode Rewards', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Episode Scores
    ax = axes[0, 1]
    ax.plot(episodes, v3_scores, alpha=0.3, color='blue', label='V3 (Concat)')
    ax.plot(episodes, v4_scores, alpha=0.3, color='red', label='V4 (CrossAttn)')
    
    if len(v3_scores) > window:
        v3_smooth = np.convolve(v3_scores, np.ones(window)/window, mode='valid')
        v4_smooth = np.convolve(v4_scores, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(v3_scores)), v3_smooth, 
                color='blue', linewidth=2, label=f'V3 Smoothed ({window}ep)')
        ax.plot(np.arange(window-1, len(v4_scores)), v4_smooth, 
                color='red', linewidth=2, label=f'V4 Smoothed ({window}ep)')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Episode Scores', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Alpha Entropy
    ax = axes[1, 0]
    v3_entropy = v3_data['monitor'].get('alpha_entropy', [])
    v4_entropy = v4_data['monitor'].get('alpha_entropy', [])
    
    if v3_entropy and v4_entropy:
        entropy_episodes = np.arange(len(v3_entropy))
        ax.plot(entropy_episodes, v3_entropy, alpha=0.5, color='blue', 
                linewidth=1, label='V3 (Concat)')
        ax.plot(entropy_episodes, v4_entropy, alpha=0.5, color='red', 
                linewidth=1, label='V4 (CrossAttn)')
        
        # Reference lines
        ax.axhline(0.693, color='orange', linestyle='--', alpha=0.7, 
                   linewidth=2, label='ln(2) = 0.693 (2 experts)')
        ax.axhline(1.386, color='gray', linestyle='--', alpha=0.5, 
                   linewidth=1, label='ln(4) = 1.386 (uniform)')
        ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, 
                   linewidth=1, label='Target: 0.3-0.5')
        ax.axhline(0.3, color='green', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Entropy', fontsize=12)
        ax.set_title('Alpha Entropy (Router Specialization)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.5])
    
    # Plot 4: Performance Comparison Bar Chart
    ax = axes[1, 1]
    metrics = ['Avg\nReward', 'Avg\nScore', 'Best\nReward', 'Best\nScore']
    v3_values = [
        np.mean(v3_rewards),
        np.mean(v3_scores),
        np.max(v3_rewards),
        np.max(v3_scores)
    ]
    v4_values = [
        np.mean(v4_rewards),
        np.mean(v4_scores),
        np.max(v4_rewards),
        np.max(v4_scores)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, v3_values, width, label='V3 (Concat)', 
                   color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, v4_values, width, label='V4 (CrossAttn)', 
                   color='red', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Performance Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plots saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare V3 and V4 training results')
    parser.add_argument('--v3-log', type=str, required=True,
                       help='Path to V3 training log JSON')
    parser.add_argument('--v4-log', type=str, required=True,
                       help='Path to V4 training log JSON')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for comparison plot')
    parser.add_argument('--max-episodes', type=int, default=100,
                       help='Maximum episodes to compare (default: 100)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔬 V3 vs V4 COMPARISON ANALYSIS")
    print("="*80)
    
    # Load data
    print(f"\n📂 Loading V3 data from: {args.v3_log}")
    print(f"   (using first {args.max_episodes} episodes)")
    v3_data = load_training_log(args.v3_log, max_episodes=args.max_episodes)
    
    print(f"\n📂 Loading V4 data from: {args.v4_log}")
    v4_data = load_training_log(args.v4_log, max_episodes=args.max_episodes)
    
    if v3_data is None or v4_data is None:
        print("\n❌ Failed to load data!")
        return
    
    print(f"\n✓ V3: {len(v3_data['rewards'])} episodes")
    print(f"✓ V4: {len(v4_data['rewards'])} episodes")
    
    # Analyze
    improvements = analyze_comparison(v3_data, v4_data)
    
    # Create plots
    create_comparison_plots(v3_data, v4_data, args.output)
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
