#!/usr/bin/env python3
"""
可视化三个训练阶段的对比
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_three_phases():
    """可视化三个训练阶段"""
    
    # 读取三个阶段的数据
    warmup_path = Path('ablation_v3/results/warmup_1000/logs/training_log.json')
    transition_path = Path('ablation_v3/results/transition_3000/logs/training_log.json')
    finetune_path = Path('ablation_v3/results/finetune_5000/logs/training_log.json')
    
    with open(warmup_path, 'r') as f:
        warmup_data = json.load(f)
    
    with open(transition_path, 'r') as f:
        transition_data = json.load(f)
    
    with open(finetune_path, 'r') as f:
        finetune_data = json.load(f)
    
    # 提取数据
    warmup_alpha = warmup_data['monitor_metrics']['alpha_entropy']
    warmup_scores = warmup_data['episode_scores']
    
    transition_alpha = transition_data['monitor_metrics']['alpha_entropy']
    transition_scores = transition_data['episode_scores']
    
    finetune_alpha = finetune_data['monitor_metrics']['alpha_entropy']
    finetune_scores = finetune_data['episode_scores']
    
    # 合并数据
    all_alpha = warmup_alpha + transition_alpha + finetune_alpha
    all_scores = warmup_scores + transition_scores + finetune_scores
    episodes = list(range(len(all_alpha)))
    
    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    # ============================================================
    # 图1: Alpha熵变化（专家专业化）
    # ============================================================
    ax1 = axes[0]
    
    # 绘制Alpha熵
    ax1.plot(episodes[:1000], all_alpha[:1000], 
             color='#3498db', linewidth=1.5, alpha=0.7, label='Warmup Phase (0-1000)')
    ax1.plot(episodes[1000:3000], all_alpha[1000:3000], 
             color='#e74c3c', linewidth=1.5, alpha=0.7, label='Transition Phase (1000-3000)')
    ax1.plot(episodes[3000:], all_alpha[3000:], 
             color='#2ecc71', linewidth=1.5, alpha=0.7, label='Fine-tune Phase (3000-5000)')
    
    # 添加平均线
    warmup_avg = np.mean(warmup_alpha)
    transition_avg = np.mean(transition_alpha)
    finetune_avg = np.mean(finetune_alpha)
    
    ax1.axhline(y=warmup_avg, xmin=0, xmax=1000/len(episodes), 
                color='#3498db', linestyle='--', linewidth=2.5, 
                label=f'Warmup Avg: {warmup_avg:.4f}')
    ax1.axhline(y=transition_avg, xmin=1000/len(episodes), xmax=3000/len(episodes), 
                color='#e74c3c', linestyle='--', linewidth=2.5, 
                label=f'Transition Avg: {transition_avg:.4f}')
    ax1.axhline(y=finetune_avg, xmin=3000/len(episodes), xmax=1, 
                color='#2ecc71', linestyle='--', linewidth=2.5, 
                label=f'Fine-tune Avg: {finetune_avg:.4f}')
    
    # 添加理论值参考线
    ax1.axhline(y=1.386, color='gray', linestyle=':', linewidth=1.5, 
                label='Theoretical Max (log(4)=1.386)', alpha=0.5)
    ax1.axhline(y=0.7, color='orange', linestyle=':', linewidth=1.5, 
                label='Target (~0.7)', alpha=0.5)
    ax1.axhline(y=0.5, color='purple', linestyle=':', linewidth=1.5, 
                label='High Specialization (~0.5)', alpha=0.5)
    
    # 添加阶段分隔线
    ax1.axvline(x=1000, color='black', linestyle='-', linewidth=2.5, alpha=0.4)
    ax1.axvline(x=3000, color='black', linestyle='-', linewidth=2.5, alpha=0.4)
    
    ax1.text(1000, 1.35, 'Sparsemax\nActivated', 
             ha='center', va='top', fontsize=11, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.text(3000, 1.35, 'Fine-tune\nStarts', 
             ha='center', va='top', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax1.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Alpha Entropy', fontsize=13, fontweight='bold')
    ax1.set_title('Expert Specialization Progress (Alpha Entropy)', 
                  fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.5])
    
    # ============================================================
    # 图2: 分数变化（性能）
    # ============================================================
    ax2 = axes[1]
    
    # 计算移动平均
    window = 50
    warmup_scores_ma = np.convolve(warmup_scores, np.ones(window)/window, mode='valid')
    transition_scores_ma = np.convolve(transition_scores, np.ones(window)/window, mode='valid')
    finetune_scores_ma = np.convolve(finetune_scores, np.ones(window)/window, mode='valid')
    
    # 绘制原始分数（半透明）
    ax2.scatter(episodes[:1000], all_scores[:1000], 
                color='#3498db', s=3, alpha=0.15)
    ax2.scatter(episodes[1000:3000], all_scores[1000:3000], 
                color='#e74c3c', s=3, alpha=0.15)
    ax2.scatter(episodes[3000:], all_scores[3000:], 
                color='#2ecc71', s=3, alpha=0.15)
    
    # 绘制移动平均
    ax2.plot(range(window-1, window-1+len(warmup_scores_ma)), warmup_scores_ma, 
             color='#3498db', linewidth=3, label='Warmup Phase (MA-50)')
    ax2.plot(range(1000+window-1, 1000+window-1+len(transition_scores_ma)), transition_scores_ma, 
             color='#e74c3c', linewidth=3, label='Transition Phase (MA-50)')
    ax2.plot(range(3000+window-1, 3000+window-1+len(finetune_scores_ma)), finetune_scores_ma, 
             color='#2ecc71', linewidth=3, label='Fine-tune Phase (MA-50)')
    
    # 添加平均线
    warmup_score_avg = np.mean(warmup_scores)
    transition_score_avg = np.mean(transition_scores)
    finetune_score_avg = np.mean(finetune_scores)
    
    ax2.axhline(y=warmup_score_avg, xmin=0, xmax=1000/len(episodes), 
                color='#3498db', linestyle='--', linewidth=2.5, 
                label=f'Warmup Avg: {warmup_score_avg:.2f}')
    ax2.axhline(y=transition_score_avg, xmin=1000/len(episodes), xmax=3000/len(episodes), 
                color='#e74c3c', linestyle='--', linewidth=2.5, 
                label=f'Transition Avg: {transition_score_avg:.2f}')
    ax2.axhline(y=finetune_score_avg, xmin=3000/len(episodes), xmax=1, 
                color='#2ecc71', linestyle='--', linewidth=2.5, 
                label=f'Fine-tune Avg: {finetune_score_avg:.2f}')
    
    # 添加阶段分隔线
    ax2.axvline(x=1000, color='black', linestyle='-', linewidth=2.5, alpha=0.4)
    ax2.axvline(x=3000, color='black', linestyle='-', linewidth=2.5, alpha=0.4)
    
    ax2.set_xlabel('Episode', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Episode Score', fontsize=13, fontweight='bold')
    ax2.set_title('Performance Improvement (Episode Score)', 
                  fontsize=15, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-10, 500])
    
    # 添加最高分标注
    max_score = max(all_scores)
    max_idx = all_scores.index(max_score)
    ax2.scatter([max_idx], [max_score], color='red', s=200, marker='*', 
                zorder=5, label=f'Best Score: {max_score}')
    ax2.annotate(f'Best: {max_score}', 
                 xy=(max_idx, max_score), xytext=(max_idx-500, max_score+50),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2),
                 fontsize=12, ha='center', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # ============================================================
    # 图3: 阶段对比柱状图
    # ============================================================
    ax3 = axes[2]
    
    phases = ['Warmup\n(0-1000)', 'Transition\n(1000-3000)', 'Fine-tune\n(3000-5000)']
    scores = [warmup_score_avg, transition_score_avg, finetune_score_avg]
    alphas = [warmup_avg, transition_avg, finetune_avg]
    
    x = np.arange(len(phases))
    width = 0.35
    
    # 绘制柱状图
    bars1 = ax3.bar(x - width/2, scores, width, label='Avg Score', 
                    color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8)
    
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x + width/2, alphas, width, label='Alpha Entropy', 
                         color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.4)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars1, scores)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for i, (bar, alpha) in enumerate(zip(bars2, alphas)):
        height = bar.get_height()
        ax3_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{alpha:.4f}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_xlabel('Training Phase', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Average Score', fontsize=13, fontweight='bold', color='#2c3e50')
    ax3_twin.set_ylabel('Alpha Entropy', fontsize=13, fontweight='bold', color='#7f8c8d')
    ax3.set_title('Phase Comparison: Score vs Alpha Entropy', 
                  fontsize=15, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(phases, fontsize=11)
    ax3.legend(loc='upper left', fontsize=11)
    ax3_twin.legend(loc='upper right', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, max(scores) * 1.3])
    ax3_twin.set_ylim([0, max(alphas) * 1.3])
    
    # 添加改善百分比
    improvement_1 = (transition_score_avg - warmup_score_avg) / warmup_score_avg * 100
    improvement_2 = (finetune_score_avg - transition_score_avg) / transition_score_avg * 100
    improvement_total = (finetune_score_avg - warmup_score_avg) / warmup_score_avg * 100
    
    ax3.text(0.5, max(scores) * 1.15, f'+{improvement_1:.1f}%', 
             ha='center', fontsize=11, fontweight='bold', color='green')
    ax3.text(1.5, max(scores) * 1.15, f'+{improvement_2:.1f}%', 
             ha='center', fontsize=11, fontweight='bold', color='green')
    ax3.text(1, max(scores) * 1.25, f'Total: +{improvement_total:.1f}%', 
             ha='center', fontsize=12, fontweight='bold', color='darkgreen',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'ablation_v3/visualizations/three_phases_comparison.png'
    Path('ablation_v3/visualizations').mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存到: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    visualize_three_phases()
