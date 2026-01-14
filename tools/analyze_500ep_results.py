#!/usr/bin/env python3
"""
分析500 episodes训练结果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_training_log(result_dir):
    """加载训练日志"""
    log_file = Path(result_dir) / "logs" / "training_log.json"
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    return data

def analyze_results(data):
    """分析训练结果"""
    # 处理不同的日志格式
    if isinstance(data, dict):
        # 新格式：字典包含列表
        episodes = list(range(len(data.get('episode_rewards', []))))
        scores = data.get('episode_scores', data.get('episode_rewards', []))
        rewards = data.get('episode_rewards', [])
        alpha_entropies = data.get('alpha_entropies', [0] * len(rewards))
        alignment_losses = data.get('alignment_losses', [None] * len(rewards))
        semantic_losses = data.get('semantic_losses', [None] * len(rewards))
    else:
        # 旧格式：列表包含字典
        episodes = [entry['episode'] for entry in data]
        scores = [entry['score'] for entry in data]
        rewards = [entry['reward'] for entry in data]
        alpha_entropies = [entry.get('alpha_entropy', 0) for entry in data]
        alignment_losses = [entry.get('alignment_loss', None) for entry in data]
        semantic_losses = [entry.get('semantic_loss', None) for entry in data]
    
    print(f"\n{'='*80}")
    print(f"500 EPISODES TRAINING RESULTS")
    print(f"{'='*80}\n")
    
    print(f"训练Episodes: {len(episodes)}")
    print(f"Episode范围: {min(episodes)} - {max(episodes)}")
    print()
    
    # 分数统计
    print(f"分数统计:")
    print(f"  平均分数: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"  最高分数: {np.max(scores):.2f}")
    print(f"  最低分数: {np.min(scores):.2f}")
    print(f"  最终分数: {scores[-1]:.2f}")
    print()
    
    # 奖励统计
    print(f"奖励统计:")
    print(f"  平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  最高奖励: {np.max(rewards):.2f}")
    print(f"  最低奖励: {np.min(rewards):.2f}")
    print(f"  最终奖励: {rewards[-1]:.2f}")
    print()
    
    # Alpha熵统计
    print(f"Alpha熵统计:")
    print(f"  平均熵: {np.mean(alpha_entropies):.4f} ± {np.std(alpha_entropies):.4f}")
    print(f"  最终熵: {alpha_entropies[-1]:.4f}")
    print(f"  理论最大值: 1.3863 (ln(4))")
    print()
    
    # Manager约束统计
    if any(v is not None for v in alignment_losses):
        alignment = [v for v in alignment_losses if v is not None]
        print(f"Manager约束统计:")
        print(f"  对齐损失: {np.mean(alignment):.4f} ± {np.std(alignment):.4f}")
        print(f"  最终对齐损失: {alignment[-1]:.4f}")
        
        if any(v is not None for v in semantic_losses):
            semantic = [v for v in semantic_losses if v is not None]
            print(f"  语义损失: {np.mean(semantic):.4f} ± {np.std(semantic):.4f}")
            print(f"  最终语义损失: {semantic[-1]:.4f}")
        print()
    
    # 趋势分析
    print(f"趋势分析:")
    first_100 = scores[:100]
    last_100 = scores[-100:]
    improvement = ((np.mean(last_100) - np.mean(first_100)) / np.mean(first_100)) * 100
    print(f"  前100ep平均分数: {np.mean(first_100):.2f}")
    print(f"  后100ep平均分数: {np.mean(last_100):.2f}")
    print(f"  改进: {improvement:+.1f}%")
    print()
    
    return {
        'episodes': episodes,
        'scores': scores,
        'rewards': rewards,
        'alpha_entropies': alpha_entropies,
        'alignment_losses': alignment_losses,
        'semantic_losses': semantic_losses
    }

def plot_results(metrics, output_dir):
    """绘制结果图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('500 Episodes Training Results (With Manager Constraints)', 
                 fontsize=16, fontweight='bold')
    
    # 1. 分数曲线
    ax = axes[0, 0]
    ax.plot(metrics['episodes'], metrics['scores'], alpha=0.3, color='blue')
    # 移动平均
    window = 50
    ma_scores = []
    for i in range(len(metrics['scores'])):
        start = max(0, i - window + 1)
        ma_scores.append(np.mean(metrics['scores'][start:i+1]))
    ax.plot(metrics['episodes'], ma_scores, color='blue', linewidth=2, label='50-ep MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Score over Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Alpha熵
    ax = axes[0, 1]
    ax.plot(metrics['episodes'], metrics['alpha_entropies'], color='green', alpha=0.7)
    ax.axhline(y=1.386, color='red', linestyle='--', alpha=0.5, label='Theoretical Max')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Alpha Entropy')
    ax.set_title('Alpha Entropy over Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 奖励曲线
    ax = axes[1, 0]
    ax.plot(metrics['episodes'], metrics['rewards'], alpha=0.3, color='orange')
    # 移动平均
    ma_rewards = []
    for i in range(len(metrics['rewards'])):
        start = max(0, i - window + 1)
        ma_rewards.append(np.mean(metrics['rewards'][start:i+1]))
    ax.plot(metrics['episodes'], ma_rewards, color='orange', linewidth=2, label='50-ep MA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward over Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Manager约束损失
    ax = axes[1, 1]
    if any(v is not None for v in metrics['alignment_losses']):
        alignment = [v for v in metrics['alignment_losses'] if v is not None]
        episodes_align = [e for e, v in zip(metrics['episodes'], 
                                            metrics['alignment_losses']) if v is not None]
        ax.plot(episodes_align, alignment, label='Alignment Loss', color='purple', alpha=0.7)
    
    if any(v is not None for v in metrics['semantic_losses']):
        semantic = [v for v in metrics['semantic_losses'] if v is not None]
        episodes_sem = [e for e, v in zip(metrics['episodes'], 
                                          metrics['semantic_losses']) if v is not None]
        ax.plot(episodes_sem, semantic, label='Semantic Loss', color='brown', alpha=0.7)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Manager Constraint Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '500ep_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存分析图: {output_dir / '500ep_analysis.png'}")
    plt.close()

def main():
    result_dir = "ablation_v3/results/resume_500_from_100"
    output_dir = "ablation_v3/visualizations/500ep_analysis"
    
    print(f"加载训练数据: {result_dir}")
    data = load_training_log(result_dir)
    
    metrics = analyze_results(data)
    
    print(f"生成可视化...")
    plot_results(metrics, output_dir)
    
    print(f"\n✅ 分析完成！")
    print(f"可视化保存在: {output_dir}")

if __name__ == '__main__':
    main()
