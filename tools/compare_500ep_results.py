#!/usr/bin/env python3
"""
对比500 episodes训练结果
- 旧版本 (quick_manager_20260111_230845): Episode 0-100
- 新版本 (resume_500_from_100): Episode 100-500
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 数据路径
OLD_LOG = "ablation_v3/results/quick_manager_20260111_230845/logs/training_log.json"
NEW_LOG = "ablation_v3/results/resume_500_from_100.log"

def load_old_data():
    """加载旧版本数据 (JSON格式)"""
    with open(OLD_LOG, 'r') as f:
        data = json.load(f)
    
    episodes = []
    scores = []
    entropies = []
    
    for entry in data:
        episodes.append(entry['episode'])
        scores.append(entry['score'])
        entropies.append(entry['alpha_entropy'])
    
    return episodes, scores, entropies

def load_new_data():
    """加载新版本数据 (日志格式)"""
    import re
    
    episodes = []
    scores = []
    entropies = []
    
    with open(NEW_LOG, 'r') as f:
        for line in f:
            match = re.search(r'Episode (\d+)/500.*Score: ([\d.]+).*α_entropy: ([\d.]+)', line)
            if match:
                ep = int(match.group(1))
                score = float(match.group(2))
                entropy = float(match.group(4))
                episodes.append(ep)
                scores.append(score)
                entropies.append(entropy)
    
    return episodes, scores, entropies

def smooth(data, window=10):
    """滑动平均平滑"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def main():
    # 加载数据
    old_ep, old_scores, old_entropy = load_old_data()
    new_ep, new_scores, new_entropy = load_new_data()
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('500 Episodes训练对比：旧版本(0-100) vs 新版本(100-500)', fontsize=16, fontweight='bold')
    
    # 1. 分数对比
    ax = axes[0, 0]
    ax.plot(old_ep, old_scores, 'o-', alpha=0.3, color='blue', label='旧版本 (0-100ep)')
    ax.plot(new_ep, new_scores, 'o-', alpha=0.3, color='red', label='新版本 (100-500ep)')
    
    # 滑动平均
    if len(old_scores) >= 10:
        ax.plot(old_ep[9:], smooth(old_scores, 10), '-', linewidth=2, color='blue', label='旧版本 (平滑)')
    if len(new_scores) >= 10:
        ax.plot(new_ep[9:], smooth(new_scores, 10), '-', linewidth=2, color='red', label='新版本 (平滑)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('分数对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Alpha熵对比
    ax = axes[0, 1]
    ax.plot(old_ep, old_entropy, 'o-', alpha=0.5, color='blue', label='旧版本')
    ax.plot(new_ep, new_entropy, 'o-', alpha=0.5, color='red', label='新版本')
    ax.axhline(y=1.386, color='green', linestyle='--', label='理想熵 (ln4≈1.386)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Alpha Entropy')
    ax.set_title('专家路由熵对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 分段统计
    ax = axes[1, 0]
    
    # 旧版本分段
    old_segments = []
    old_labels = []
    for start in range(0, 100, 20):
        end = start + 20
        segment = [s for e, s in zip(old_ep, old_scores) if start <= e < end]
        if segment:
            old_segments.append(np.mean(segment))
            old_labels.append(f'{start}-{end}')
    
    # 新版本分段
    new_segments = []
    new_labels = []
    for start in range(100, 500, 50):
        end = start + 50
        segment = [s for e, s in zip(new_ep, new_scores) if start <= e < end]
        if segment:
            new_segments.append(np.mean(segment))
            new_labels.append(f'{start}-{end}')
    
    x_old = np.arange(len(old_segments))
    x_new = np.arange(len(new_segments)) + len(old_segments) + 0.5
    
    ax.bar(x_old, old_segments, width=0.8, color='blue', alpha=0.6, label='旧版本')
    ax.bar(x_new, new_segments, width=0.8, color='red', alpha=0.6, label='新版本')
    
    ax.set_xticks(list(x_old) + list(x_new))
    ax.set_xticklabels(old_labels + new_labels, rotation=45, ha='right')
    ax.set_ylabel('平均分数')
    ax.set_title('分段平均分数')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 统计摘要
    ax = axes[1, 1]
    ax.axis('off')
    
    # 计算统计
    old_mean = np.mean(old_scores)
    old_max = np.max(old_scores)
    old_entropy_mean = np.mean(old_entropy)
    
    new_mean = np.mean(new_scores)
    new_max = np.max(new_scores)
    new_entropy_mean = np.mean(new_entropy)
    
    improvement = ((new_mean - old_mean) / old_mean * 100) if old_mean > 0 else 0
    
    summary_text = f"""
统计摘要

旧版本 (Episode 0-100):
  • 平均分数: {old_mean:.2f}
  • 最高分数: {old_max:.1f}
  • 平均Alpha熵: {old_entropy_mean:.4f}

新版本 (Episode 100-500):
  • 平均分数: {new_mean:.2f}
  • 最高分数: {new_max:.1f}
  • 平均Alpha熵: {new_entropy_mean:.4f}

改进:
  • 分数提升: {improvement:+.1f}%
  • 熵变化: {new_entropy_mean - old_entropy_mean:+.4f}

关键发现:
  • 新版本包含4个机制:
    1. Manager内层约束
    2. 熵最小化 (Fine-tune阶段)
    3. 时间一致性
    4. 专家重叠惩罚
  
  • 训练阶段: Warmup (0-1000ep)
  • 路由方式: Softmax
  • Alpha熵稳定在1.385左右
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # 保存
    output_dir = Path("ablation_v3/visualizations")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "compare_500ep_old_vs_new.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化已保存: {output_path}")
    
    # 打印统计
    print(f"\n{'='*60}")
    print("500 Episodes训练对比")
    print(f"{'='*60}")
    print(f"\n旧版本 (Episode 0-100):")
    print(f"  平均分数: {old_mean:.2f}")
    print(f"  最高分数: {old_max:.1f}")
    print(f"  平均Alpha熵: {old_entropy_mean:.4f}")
    
    print(f"\n新版本 (Episode 100-500):")
    print(f"  平均分数: {new_mean:.2f}")
    print(f"  最高分数: {new_max:.1f}")
    print(f"  平均Alpha熵: {new_entropy_mean:.4f}")
    
    print(f"\n改进:")
    print(f"  分数提升: {improvement:+.1f}%")
    print(f"  熵变化: {new_entropy_mean - old_entropy_mean:+.4f}")
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
