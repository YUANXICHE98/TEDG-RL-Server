#!/usr/bin/env python3
"""
对比有/无Manager约束的训练结果
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_training_log(result_dir):
    """加载训练日志"""
    log_file = Path(result_dir) / "logs" / "training_log.json"
    
    if not log_file.exists():
        raise FileNotFoundError(f"训练日志不存在: {log_file}")
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    return data

def extract_metrics(data):
    """提取关键指标"""
    episodes = [entry['episode'] for entry in data]
    scores = [entry['score'] for entry in data]
    rewards = [entry['reward'] for entry in data]
    alpha_entropies = [entry.get('alpha_entropy', 0) for entry in data]
    
    # Manager约束相关指标（如果存在）
    alignment_losses = [entry.get('alignment_loss', None) for entry in data]
    semantic_losses = [entry.get('semantic_loss', None) for entry in data]
    
    return {
        'episodes': episodes,
        'scores': scores,
        'rewards': rewards,
        'alpha_entropies': alpha_entropies,
        'alignment_losses': alignment_losses,
        'semantic_losses': semantic_losses
    }

def compute_statistics(values):
    """计算统计指标"""
    values = [v for v in values if v is not None]
    if not values:
        return None
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'final': values[-1] if values else None
    }

def moving_average(values, window=50):
    """计算移动平均"""
    if len(values) < window:
        return values
    
    ma = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        ma.append(np.mean(values[start:i+1]))
    return ma

def plot_comparison(baseline_metrics, manager_metrics, phase, output_dir):
    """绘制对比图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Training Comparison: With vs Without Manager Constraints ({phase.capitalize()} Phase)', 
                 fontsize=16, fontweight='bold')
    
    # 1. 分数对比
    ax = axes[0, 0]
    ax.plot(baseline_metrics['episodes'], 
            moving_average(baseline_metrics['scores']), 
            label='Baseline (No Manager)', color='blue', alpha=0.7)
    ax.plot(manager_metrics['episodes'], 
            moving_average(manager_metrics['scores']), 
            label='With Manager Constraints', color='red', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Score Comparison (50-episode MA)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Alpha熵对比
    ax = axes[0, 1]
    ax.plot(baseline_metrics['episodes'], 
            baseline_metrics['alpha_entropies'], 
            label='Baseline (No Manager)', color='blue', alpha=0.7)
    ax.plot(manager_metrics['episodes'], 
            manager_metrics['alpha_entropies'], 
            label='With Manager Constraints', color='red', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Alpha Entropy')
    ax.set_title('Alpha Entropy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.386, color='gray', linestyle='--', alpha=0.5, label='Theoretical Max')
    
    # 3. 奖励对比
    ax = axes[1, 0]
    ax.plot(baseline_metrics['episodes'], 
            moving_average(baseline_metrics['rewards']), 
            label='Baseline (No Manager)', color='blue', alpha=0.7)
    ax.plot(manager_metrics['episodes'], 
            moving_average(manager_metrics['rewards']), 
            label='With Manager Constraints', color='red', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Comparison (50-episode MA)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Manager约束损失（仅新版本）
    ax = axes[1, 1]
    if any(v is not None for v in manager_metrics['alignment_losses']):
        alignment = [v for v in manager_metrics['alignment_losses'] if v is not None]
        episodes_align = [e for e, v in zip(manager_metrics['episodes'], 
                                            manager_metrics['alignment_losses']) if v is not None]
        ax.plot(episodes_align, alignment, label='Alignment Loss', color='green', alpha=0.7)
    
    if any(v is not None for v in manager_metrics['semantic_losses']):
        semantic = [v for v in manager_metrics['semantic_losses'] if v is not None]
        episodes_sem = [e for e, v in zip(manager_metrics['episodes'], 
                                          manager_metrics['semantic_losses']) if v is not None]
        ax.plot(episodes_sem, semantic, label='Semantic Loss', color='orange', alpha=0.7)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Manager Constraint Losses (With Manager Only)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{phase}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ 保存对比图: {output_dir / f'{phase}_comparison.png'}")
    plt.close()

def print_comparison_table(baseline_stats, manager_stats, phase):
    """打印对比表格"""
    print(f"\n{'='*80}")
    print(f"{phase.upper()} PHASE COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Metric':<25} {'Baseline':<20} {'With Manager':<20} {'Improvement':<15}")
    print(f"{'-'*80}")
    
    # 分数对比
    if baseline_stats['scores'] and manager_stats['scores']:
        baseline_score = baseline_stats['scores']['mean']
        manager_score = manager_stats['scores']['mean']
        improvement = ((manager_score - baseline_score) / baseline_score) * 100
        print(f"{'Average Score':<25} {baseline_score:<20.2f} {manager_score:<20.2f} {improvement:>+14.1f}%")
        
        baseline_final = baseline_stats['scores']['final']
        manager_final = manager_stats['scores']['final']
        improvement_final = ((manager_final - baseline_final) / baseline_final) * 100
        print(f"{'Final Score':<25} {baseline_final:<20.2f} {manager_final:<20.2f} {improvement_final:>+14.1f}%")
    
    # Alpha熵对比
    if baseline_stats['alpha_entropies'] and manager_stats['alpha_entropies']:
        baseline_entropy = baseline_stats['alpha_entropies']['mean']
        manager_entropy = manager_stats['alpha_entropies']['mean']
        change = ((manager_entropy - baseline_entropy) / baseline_entropy) * 100
        print(f"{'Average Alpha Entropy':<25} {baseline_entropy:<20.4f} {manager_entropy:<20.4f} {change:>+14.1f}%")
        
        baseline_final_ent = baseline_stats['alpha_entropies']['final']
        manager_final_ent = manager_stats['alpha_entropies']['final']
        change_final = ((manager_final_ent - baseline_final_ent) / baseline_final_ent) * 100
        print(f"{'Final Alpha Entropy':<25} {baseline_final_ent:<20.4f} {manager_final_ent:<20.4f} {change_final:>+14.1f}%")
    
    # 奖励对比
    if baseline_stats['rewards'] and manager_stats['rewards']:
        baseline_reward = baseline_stats['rewards']['mean']
        manager_reward = manager_stats['rewards']['mean']
        improvement_rew = ((manager_reward - baseline_reward) / baseline_reward) * 100
        print(f"{'Average Reward':<25} {baseline_reward:<20.2f} {manager_reward:<20.2f} {improvement_rew:>+14.1f}%")
    
    print(f"{'-'*80}\n")
    
    # Manager约束损失（仅新版本）
    if manager_stats['alignment_losses']:
        print(f"Manager Constraint Losses (With Manager Only):")
        print(f"  Alignment Loss: {manager_stats['alignment_losses']['mean']:.4f} ± {manager_stats['alignment_losses']['std']:.4f}")
        print(f"  Final Alignment Loss: {manager_stats['alignment_losses']['final']:.4f}")
    
    if manager_stats['semantic_losses']:
        print(f"  Semantic Loss: {manager_stats['semantic_losses']['mean']:.4f} ± {manager_stats['semantic_losses']['std']:.4f}")
        print(f"  Final Semantic Loss: {manager_stats['semantic_losses']['final']:.4f}")
    
    print()

def main():
    parser = argparse.ArgumentParser(description='对比有/无Manager约束的训练结果')
    parser.add_argument('--baseline', required=True, help='Baseline结果目录（无Manager约束）')
    parser.add_argument('--with_manager', required=True, help='新版本结果目录（有Manager约束）')
    parser.add_argument('--phase', required=True, choices=['warmup', 'transition', 'finetune'], 
                       help='训练阶段')
    parser.add_argument('--output', default='ablation_v3/visualizations/comparison/', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    print(f"\n加载训练数据...")
    print(f"  Baseline: {args.baseline}")
    print(f"  With Manager: {args.with_manager}")
    
    # 加载数据
    baseline_data = load_training_log(args.baseline)
    manager_data = load_training_log(args.with_manager)
    
    # 提取指标
    baseline_metrics = extract_metrics(baseline_data)
    manager_metrics = extract_metrics(manager_data)
    
    # 计算统计
    baseline_stats = {
        'scores': compute_statistics(baseline_metrics['scores']),
        'rewards': compute_statistics(baseline_metrics['rewards']),
        'alpha_entropies': compute_statistics(baseline_metrics['alpha_entropies']),
        'alignment_losses': None,
        'semantic_losses': None
    }
    
    manager_stats = {
        'scores': compute_statistics(manager_metrics['scores']),
        'rewards': compute_statistics(manager_metrics['rewards']),
        'alpha_entropies': compute_statistics(manager_metrics['alpha_entropies']),
        'alignment_losses': compute_statistics(manager_metrics['alignment_losses']),
        'semantic_losses': compute_statistics(manager_metrics['semantic_losses'])
    }
    
    # 打印对比表格
    print_comparison_table(baseline_stats, manager_stats, args.phase)
    
    # 绘制对比图
    print(f"生成对比可视化...")
    plot_comparison(baseline_metrics, manager_metrics, args.phase, args.output)
    
    print(f"\n✅ 对比分析完成！")
    print(f"结果保存在: {args.output}")

if __name__ == '__main__':
    main()
