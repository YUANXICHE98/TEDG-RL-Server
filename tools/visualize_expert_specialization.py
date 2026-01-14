#!/usr/bin/env python3
"""
可视化第二阶段专家专业化过程
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_expert_specialization():
    """可视化专家专业化过程"""
    
    # 读取Warmup和Transition数据
    warmup_path = Path('ablation_v3/results/warmup_1000/logs/training_log.json')
    transition_path = Path('ablation_v3/results/transition_3000/logs/training_log.json')
    
    with open(warmup_path, 'r') as f:
        warmup_data = json.load(f)
    
    with open(transition_path, 'r') as f:
        transition_data = json.load(f)
    
    # 提取数据
    warmup_alpha = warmup_data['monitor_metrics']['alpha_entropy']
    warmup_scores = warmup_data['episode_scores']
    
    transition_alpha = transition_data['monitor_metrics']['alpha_entropy']
    transition_scores = transition_data['episode_scores']
    
    # 合并数据
    all_alpha = warmup_alpha + transition_alpha
    all_scores = warmup_scores + transition_scores
    episodes = list(range(len(all_alpha)))
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # ============================================================
    # 图1: Alpha熵变化（专家专业化指标）
    # ============================================================
    ax1 = axes[0]
    
    # 绘制Alpha熵
    ax1.plot(episodes[:1000], all_alpha[:1000], 
             color='#3498db', linewidth=1.5, alpha=0.7, label='Warmup Phase')
    ax1.plot(episodes[1000:], all_alpha[1000:], 
             color='#e74c3c', linewidth=1.5, alpha=0.7, label='Transition Phase')
    
    # 添加平均线
    warmup_avg = np.mean(warmup_alpha)
    transition_avg = np.mean(transition_alpha)
    
    ax1.axhline(y=warmup_avg, xmin=0, xmax=1000/len(episodes), 
                color='#3498db', linestyle='--', linewidth=2, 
                label=f'Warmup Avg: {warmup_avg:.4f}')
    ax1.axhline(y=transition_avg, xmin=1000/len(episodes), xmax=1, 
                color='#e74c3c', linestyle='--', linewidth=2, 
                label=f'Transition Avg: {transition_avg:.4f}')
    
    # 添加理论值参考线
    ax1.axhline(y=1.386, color='gray', linestyle=':', linewidth=1, 
                label='Theoretical Max (log(4)=1.386)')
    ax1.axhline(y=0.7, color='green', linestyle=':', linewidth=1, 
                label='Target (~0.7)')
    
    # 添加阶段分隔线
    ax1.axvline(x=1000, color='black', linestyle='-', linewidth=2, alpha=0.3)
    ax1.text(1000, 1.3, 'Sparsemax\nActivated', 
             ha='center', va='top', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Alpha Entropy', fontsize=12)
    ax1.set_title('Expert Specialization Progress (Alpha Entropy)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.5])
    
    # 添加注释
    ax1.annotate('Experts uniformly distributed\n(No specialization)', 
                 xy=(500, warmup_avg), xytext=(500, 1.2),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                 fontsize=10, ha='center')
    
    ax1.annotate('Experts begin to specialize\n(50% entropy reduction)', 
                 xy=(2000, transition_avg), xytext=(2000, 0.4),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 fontsize=10, ha='center')
    
    # ============================================================
    # 图2: 分数变化（性能指标）
    # ============================================================
    ax2 = axes[1]
    
    # 计算移动平均
    window = 50
    warmup_scores_ma = np.convolve(warmup_scores, np.ones(window)/window, mode='valid')
    transition_scores_ma = np.convolve(transition_scores, np.ones(window)/window, mode='valid')
    
    # 绘制原始分数（半透明）
    ax2.scatter(episodes[:1000], all_scores[:1000], 
                color='#3498db', s=5, alpha=0.2)
    ax2.scatter(episodes[1000:], all_scores[1000:], 
                color='#e74c3c', s=5, alpha=0.2)
    
    # 绘制移动平均
    ax2.plot(range(window-1, 1000), warmup_scores_ma, 
             color='#3498db', linewidth=2.5, label='Warmup Phase (MA-50)')
    ax2.plot(range(1000+window-1, len(episodes)), transition_scores_ma, 
             color='#e74c3c', linewidth=2.5, label='Transition Phase (MA-50)')
    
    # 添加平均线
    warmup_score_avg = np.mean(warmup_scores)
    transition_score_avg = np.mean(transition_scores)
    
    ax2.axhline(y=warmup_score_avg, xmin=0, xmax=1000/len(episodes), 
                color='#3498db', linestyle='--', linewidth=2, 
                label=f'Warmup Avg: {warmup_score_avg:.2f}')
    ax2.axhline(y=transition_score_avg, xmin=1000/len(episodes), xmax=1, 
                color='#e74c3c', linestyle='--', linewidth=2, 
                label=f'Transition Avg: {transition_score_avg:.2f}')
    
    # 添加阶段分隔线
    ax2.axvline(x=1000, color='black', linestyle='-', linewidth=2, alpha=0.3)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Episode Score', fontsize=12)
    ax2.set_title('Performance Improvement (Episode Score)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-10, 250])
    
    # 添加注释
    improvement = (transition_score_avg - warmup_score_avg) / warmup_score_avg * 100
    ax2.text(1500, 200, 
             f'Performance Improvement:\n+{improvement:.1f}%\n({warmup_score_avg:.2f} → {transition_score_avg:.2f})',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11, ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'ablation_v3/visualizations/expert_specialization_analysis.png'
    Path('ablation_v3/visualizations').mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存到: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    visualize_expert_specialization()
