#!/usr/bin/env python3
"""
对比前500 episodes: Baseline vs With Manager Constraints
两组数据画在同一个图上
"""

import json
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

def extract_metrics(data, max_episodes=500):
    """提取前N个episodes的指标"""
    # 处理不同的日志格式
    if isinstance(data, dict):
        # 新格式：字典包含列表
        rewards = data.get('episode_rewards', [])
        scores = data.get('episode_scores', rewards)  # 如果没有scores就用rewards
        alpha_entropies = data.get('alpha_entropies', [0] * len(rewards))
        alignment_losses = data.get('alignment_losses', [None] * len(rewards))
        semantic_losses = data.get('semantic_losses', [None] * len(rewards))
        
        # 截取前max_episodes
        episodes = list(range(min(len(rewards), max_episodes)))
        scores = scores[:max_episodes]
        rewards = rewards[:max_episodes]
        alpha_entropies = alpha_entropies[:max_episodes]
        alignment_losses = alignment_losses[:max_episodes]
        semantic_losses = semantic_losses[:max_episodes]
    else:
        # 旧格式：列表包含字典
        data = data[:max_episodes]
        episodes = [entry['episode'] for entry in data]
        scores = [entry.get('score', entry.get('reward', 0)) for entry in data]
        rewards = [entry['reward'] for entry in data]
        alpha_entropies = [entry.get('alpha_entropy', 0) for entry in data]
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

def moving_average(values, window=50):
    """计算移动平均"""
    if len(values) < window:
        window = max(1, len(values) // 10)
    
    ma = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        ma.append(np.mean(values[start:i+1]))
    return ma

def plot_full_comparison(baseline_metrics, manager_metrics, output_dir):
    """绘制完整对比图（两组数据在同一图上）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Training Comparison: Baseline vs With Manager Constraints (First 500 Episodes)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 颜色方案
    color_baseline = '#1f77b4'  # 蓝色
    color_manager = '#d62728'   # 红色
    
    # ========================================
    # 1. 分数对比（原始 + 移动平均）
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 原始数据（半透明）
    ax1.plot(baseline_metrics['episodes'], baseline_metrics['scores'], 
             alpha=0.2, color=color_baseline, linewidth=0.5)
    ax1.plot(manager_metrics['episodes'], manager_metrics['scores'], 
             alpha=0.2, color=color_manager, linewidth=0.5)
    
    # 移动平均（实线）
    ma_baseline = moving_average(baseline_metrics['scores'], window=50)
    ma_manager = moving_average(manager_metrics['scores'], window=50)
    
    ax1.plot(baseline_metrics['episodes'], ma_baseline, 
             label='Baseline (No Manager)', color=color_baseline, linewidth=2.5)
    ax1.plot(manager_metrics['episodes'], ma_manager, 
             label='With Manager Constraints', color=color_manager, linewidth=2.5)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Score Comparison (50-episode Moving Average)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    baseline_mean = np.mean(baseline_metrics['scores'])
    manager_mean = np.mean(manager_metrics['scores'])
    improvement = ((manager_mean - baseline_mean) / baseline_mean) * 100
    
    ax1.text(0.98, 0.02, 
             f'Baseline Avg: {baseline_mean:.2f}\n'
             f'Manager Avg: {manager_mean:.2f}\n'
             f'Improvement: {improvement:+.1f}%',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========================================
    # 2. 奖励对比（原始 + 移动平均）
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 原始数据（半透明）
    ax2.plot(baseline_metrics['episodes'], baseline_metrics['rewards'], 
             alpha=0.2, color=color_baseline, linewidth=0.5)
    ax2.plot(manager_metrics['episodes'], manager_metrics['rewards'], 
             alpha=0.2, color=color_manager, linewidth=0.5)
    
    # 移动平均（实线）
    ma_baseline_rew = moving_average(baseline_metrics['rewards'], window=50)
    ma_manager_rew = moving_average(manager_metrics['rewards'], window=50)
    
    ax2.plot(baseline_metrics['episodes'], ma_baseline_rew, 
             label='Baseline (No Manager)', color=color_baseline, linewidth=2.5)
    ax2.plot(manager_metrics['episodes'], ma_manager_rew, 
             label='With Manager Constraints', color=color_manager, linewidth=2.5)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Reward Comparison (50-episode Moving Average)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    baseline_rew_mean = np.mean(baseline_metrics['rewards'])
    manager_rew_mean = np.mean(manager_metrics['rewards'])
    rew_improvement = ((manager_rew_mean - baseline_rew_mean) / baseline_rew_mean) * 100
    
    ax2.text(0.98, 0.02, 
             f'Baseline Avg: {baseline_rew_mean:.2f}\n'
             f'Manager Avg: {manager_rew_mean:.2f}\n'
             f'Improvement: {rew_improvement:+.1f}%',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========================================
    # 3. Alpha熵对比
    # ========================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.plot(baseline_metrics['episodes'], baseline_metrics['alpha_entropies'], 
             label='Baseline (No Manager)', color=color_baseline, linewidth=2, alpha=0.8)
    ax3.plot(manager_metrics['episodes'], manager_metrics['alpha_entropies'], 
             label='With Manager Constraints', color=color_manager, linewidth=2, alpha=0.8)
    
    # 理论最大值参考线
    ax3.axhline(y=1.386, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Theoretical Max (ln(4))')
    
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Alpha Entropy', fontsize=12)
    ax3.set_title('Alpha Entropy Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 添加统计信息
    baseline_ent_mean = np.mean([e for e in baseline_metrics['alpha_entropies'] if e > 0])
    manager_ent_mean = np.mean([e for e in manager_metrics['alpha_entropies'] if e > 0])
    
    if baseline_ent_mean > 0 and manager_ent_mean > 0:
        ax3.text(0.98, 0.98, 
                 f'Baseline Avg: {baseline_ent_mean:.4f}\n'
                 f'Manager Avg: {manager_ent_mean:.4f}',
                 transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # ========================================
    # 4. Manager约束损失（仅With Manager）
    # ========================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    has_alignment = any(v is not None and v > 0 for v in manager_metrics['alignment_losses'])
    has_semantic = any(v is not None and v > 0 for v in manager_metrics['semantic_losses'])
    
    if has_alignment:
        alignment = [v if v is not None else 0 for v in manager_metrics['alignment_losses']]
        ax4.plot(manager_metrics['episodes'], alignment, 
                label='Alignment Loss', color='green', linewidth=2, alpha=0.8)
    
    if has_semantic:
        semantic = [v if v is not None else 0 for v in manager_metrics['semantic_losses']]
        ax4.plot(manager_metrics['episodes'], semantic, 
                label='Semantic Loss', color='orange', linewidth=2, alpha=0.8)
    
    if has_alignment or has_semantic:
        ax4.set_xlabel('Episode', fontsize=12)
        ax4.set_ylabel('Loss', fontsize=12)
        ax4.set_title('Manager Constraint Losses (With Manager Only)', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11, loc='best')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Manager Constraint Data Available', 
                transform=ax4.transAxes, fontsize=14, ha='center', va='center')
        ax4.set_title('Manager Constraint Losses', fontsize=14, fontweight='bold')
    
    # ========================================
    # 5. 分数分布对比（直方图）
    # ========================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    bins = np.linspace(0, max(max(baseline_metrics['scores']), max(manager_metrics['scores'])), 30)
    
    ax5.hist(baseline_metrics['scores'], bins=bins, alpha=0.6, 
             color=color_baseline, label='Baseline', edgecolor='black', linewidth=0.5)
    ax5.hist(manager_metrics['scores'], bins=bins, alpha=0.6, 
             color=color_manager, label='With Manager', edgecolor='black', linewidth=0.5)
    
    ax5.set_xlabel('Score', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.set_title('Score Distribution Comparison', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ========================================
    # 6. 统计对比表格
    # ========================================
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # 计算统计数据
    stats_data = [
        ['Metric', 'Baseline', 'With Manager', 'Improvement'],
        ['─' * 20, '─' * 15, '─' * 15, '─' * 15],
        ['Avg Score', f'{baseline_mean:.2f}', f'{manager_mean:.2f}', f'{improvement:+.1f}%'],
        ['Max Score', f'{max(baseline_metrics["scores"]):.2f}', 
         f'{max(manager_metrics["scores"]):.2f}', ''],
        ['Final Score', f'{baseline_metrics["scores"][-1]:.2f}', 
         f'{manager_metrics["scores"][-1]:.2f}', ''],
        ['', '', '', ''],
        ['Avg Reward', f'{baseline_rew_mean:.2f}', f'{manager_rew_mean:.2f}', 
         f'{rew_improvement:+.1f}%'],
        ['Max Reward', f'{max(baseline_metrics["rewards"]):.2f}', 
         f'{max(manager_metrics["rewards"]):.2f}', ''],
        ['', '', '', ''],
    ]
    
    if baseline_ent_mean > 0 and manager_ent_mean > 0:
        ent_change = ((manager_ent_mean - baseline_ent_mean) / baseline_ent_mean) * 100
        stats_data.extend([
            ['Avg Alpha Entropy', f'{baseline_ent_mean:.4f}', f'{manager_ent_mean:.4f}', 
             f'{ent_change:+.1f}%'],
        ])
    
    # 绘制表格
    table = ax6.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.35, 0.2, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 设置分隔行样式
    for i in range(4):
        table[(1, i)].set_facecolor('#f0f0f0')
    
    ax6.set_title('Statistical Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    # 保存图片
    plt.savefig(output_dir / 'baseline_vs_manager_500ep_full_comparison.png', 
                dpi=150, bbox_inches='tight')
    print(f"\n✅ 完整对比图已保存: {output_dir / 'baseline_vs_manager_500ep_full_comparison.png'}")
    
    plt.close()

def print_summary(baseline_metrics, manager_metrics):
    """打印对比摘要"""
    print(f"\n{'='*80}")
    print(f"BASELINE vs WITH MANAGER - 前500 EPISODES对比")
    print(f"{'='*80}\n")
    
    baseline_mean = np.mean(baseline_metrics['scores'])
    manager_mean = np.mean(manager_metrics['scores'])
    improvement = ((manager_mean - baseline_mean) / baseline_mean) * 100
    
    print(f"分数对比:")
    print(f"  Baseline平均分数: {baseline_mean:.2f}")
    print(f"  With Manager平均分数: {manager_mean:.2f}")
    print(f"  改进: {improvement:+.1f}%")
    print()
    
    baseline_rew = np.mean(baseline_metrics['rewards'])
    manager_rew = np.mean(manager_metrics['rewards'])
    rew_improvement = ((manager_rew - baseline_rew) / baseline_rew) * 100
    
    print(f"奖励对比:")
    print(f"  Baseline平均奖励: {baseline_rew:.2f}")
    print(f"  With Manager平均奖励: {manager_rew:.2f}")
    print(f"  改进: {rew_improvement:+.1f}%")
    print()
    
    baseline_ent = [e for e in baseline_metrics['alpha_entropies'] if e > 0]
    manager_ent = [e for e in manager_metrics['alpha_entropies'] if e > 0]
    
    if baseline_ent and manager_ent:
        baseline_ent_mean = np.mean(baseline_ent)
        manager_ent_mean = np.mean(manager_ent)
        
        print(f"Alpha熵对比:")
        print(f"  Baseline平均熵: {baseline_ent_mean:.4f}")
        print(f"  With Manager平均熵: {manager_ent_mean:.4f}")
        print()
    
    print(f"{'='*80}\n")

def main():
    # 数据路径
    baseline_dir = "ablation_v3/results/warmup_1000"  # 1000 episodes的baseline
    manager_dir = "ablation_v3/results/resume_500_from_100"  # 500 episodes with manager
    output_dir = "ablation_v3/visualizations/500ep_comparison"
    
    print(f"\n加载训练数据...")
    print(f"  Baseline (无Manager约束): {baseline_dir}")
    print(f"  With Manager (有Manager约束): {manager_dir}")
    
    # 加载数据
    baseline_data = load_training_log(baseline_dir)
    manager_data = load_training_log(manager_dir)
    
    # 提取前500个episodes的指标
    baseline_metrics = extract_metrics(baseline_data, max_episodes=500)
    manager_metrics = extract_metrics(manager_data, max_episodes=500)
    
    print(f"\n数据加载完成:")
    print(f"  Baseline: {len(baseline_metrics['episodes'])} episodes")
    print(f"  With Manager: {len(manager_metrics['episodes'])} episodes")
    
    # 打印对比摘要
    print_summary(baseline_metrics, manager_metrics)
    
    # 绘制完整对比图
    print(f"生成完整对比可视化...")
    plot_full_comparison(baseline_metrics, manager_metrics, output_dir)
    
    print(f"\n✅ 对比分析完成！")
    print(f"结果保存在: {output_dir}")
    print(f"\n查看图片:")
    print(f"  open {output_dir}/baseline_vs_manager_500ep_full_comparison.png")

if __name__ == '__main__':
    main()
