#!/usr/bin/env python3
"""
消融实验结果对比分析

对比三个版本:
1. Baseline - 无Manager约束，无熵最小化
2. +Manager - 有Manager约束，无熵最小化
3. +Manager+Entropy - 有Manager约束，有熵最小化
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_log(exp_dir: Path):
    """加载训练日志"""
    log_file = exp_dir / "logs" / "training_log.json"
    with open(log_file, 'r') as f:
        return json.load(f)


def compute_moving_average(data, window=50):
    """计算移动平均"""
    if len(data) < window:
        return data
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(np.mean(data[start:i+1]))
    return result


def plot_comparison(baseline_log, manager_log, full_log, output_dir):
    """绘制对比图"""
    
    # 提取数据
    baseline_scores = baseline_log.get('episode_scores', [])
    manager_scores = manager_log.get('episode_scores', [])
    full_scores = full_log.get('episode_scores', [])
    
    baseline_entropy = baseline_log.get('monitor_metrics', {}).get('alpha_entropy', [])
    manager_entropy = manager_log.get('monitor_metrics', {}).get('alpha_entropy', [])
    full_entropy = full_log.get('monitor_metrics', {}).get('alpha_entropy', [])
    
    # 移动平均
    baseline_scores_ma = compute_moving_average(baseline_scores, 50)
    manager_scores_ma = compute_moving_average(manager_scores, 50)
    full_scores_ma = compute_moving_average(full_scores, 50)
    
    baseline_entropy_ma = compute_moving_average(baseline_entropy, 50)
    manager_entropy_ma = compute_moving_average(manager_entropy, 50)
    full_entropy_ma = compute_moving_average(full_entropy, 50)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 分数对比
    ax = axes[0, 0]
    ax.plot(baseline_scores_ma, label='Baseline', linewidth=2, alpha=0.8)
    ax.plot(manager_scores_ma, label='+Manager', linewidth=2, alpha=0.8)
    ax.plot(full_scores_ma, label='+Manager+Entropy', linewidth=2, alpha=0.8)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Score Evolution (Moving Avg)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. Alpha熵对比
    ax = axes[0, 1]
    if baseline_entropy:
        ax.plot(baseline_entropy_ma, label='Baseline', linewidth=2, alpha=0.8)
    if manager_entropy:
        ax.plot(manager_entropy_ma, label='+Manager', linewidth=2, alpha=0.8)
    if full_entropy:
        ax.plot(full_entropy_ma, label='+Manager+Entropy', linewidth=2, alpha=0.8)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Alpha Entropy', fontsize=12)
    ax.set_title('Alpha Entropy Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 3. 分数改进（相对Baseline）
    ax = axes[1, 0]
    min_len = min(len(baseline_scores_ma), len(manager_scores_ma), len(full_scores_ma))
    
    manager_improvement = [
        (manager_scores_ma[i] - baseline_scores_ma[i]) / max(baseline_scores_ma[i], 1) * 100
        for i in range(min_len)
    ]
    full_improvement = [
        (full_scores_ma[i] - baseline_scores_ma[i]) / max(baseline_scores_ma[i], 1) * 100
        for i in range(min_len)
    ]
    
    ax.plot(manager_improvement, label='+Manager vs Baseline', linewidth=2, alpha=0.8)
    ax.plot(full_improvement, label='+Manager+Entropy vs Baseline', linewidth=2, alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Improvement (%)', fontsize=12)
    ax.set_title('Score Improvement vs Baseline', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 显示最终改进
    if len(manager_improvement) >= 50:
        manager_final = np.mean(manager_improvement[-50:])
        full_final = np.mean(full_improvement[-50:])
        ax.text(0.02, 0.98, 
                f'+Manager: {manager_final:+.1f}%\n+Manager+Entropy: {full_final:+.1f}%',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Alpha熵改进（相对Baseline）
    ax = axes[1, 1]
    if baseline_entropy and manager_entropy and full_entropy:
        min_len = min(len(baseline_entropy_ma), len(manager_entropy_ma), len(full_entropy_ma))
        
        manager_entropy_change = [
            (baseline_entropy_ma[i] - manager_entropy_ma[i]) / baseline_entropy_ma[i] * 100
            for i in range(min_len)
            if baseline_entropy_ma[i] > 0
        ]
        full_entropy_change = [
            (baseline_entropy_ma[i] - full_entropy_ma[i]) / baseline_entropy_ma[i] * 100
            for i in range(min_len)
            if baseline_entropy_ma[i] > 0
        ]
        
        ax.plot(manager_entropy_change, label='+Manager vs Baseline', linewidth=2, alpha=0.8)
        ax.plot(full_entropy_change, label='+Manager+Entropy vs Baseline', linewidth=2, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Entropy Reduction (%)', fontsize=12)
        ax.set_title('Alpha Entropy Reduction vs Baseline', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 显示最终改进
        if len(manager_entropy_change) >= 50:
            manager_final = np.mean(manager_entropy_change[-50:])
            full_final = np.mean(full_entropy_change[-50:])
            ax.text(0.02, 0.98,
                    f'+Manager: {manager_final:+.1f}%\n+Manager+Entropy: {full_final:+.1f}%',
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存
    output_file = output_dir / "ablation_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 对比图已保存: {output_file}")
    
    plt.close()


def print_summary(baseline_log, manager_log, full_log):
    """打印统计摘要"""
    
    print(f"\n{'='*70}")
    print("消融实验结果摘要")
    print(f"{'='*70}\n")
    
    # 提取数据
    baseline_scores = baseline_log.get('episode_scores', [])
    manager_scores = manager_log.get('episode_scores', [])
    full_scores = full_log.get('episode_scores', [])
    
    baseline_entropy = baseline_log.get('monitor_metrics', {}).get('alpha_entropy', [])
    manager_entropy = manager_log.get('monitor_metrics', {}).get('alpha_entropy', [])
    full_entropy = full_log.get('monitor_metrics', {}).get('alpha_entropy', [])
    
    # 计算最后50 episodes的统计
    window = min(50, len(baseline_scores), len(manager_scores), len(full_scores))
    
    baseline_final_score = np.mean(baseline_scores[-window:])
    manager_final_score = np.mean(manager_scores[-window:])
    full_final_score = np.mean(full_scores[-window:])
    
    print(f"平均分数 (最后{window} episodes):")
    print(f"  Baseline:              {baseline_final_score:.2f}")
    print(f"  +Manager:              {manager_final_score:.2f}  ({(manager_final_score-baseline_final_score)/baseline_final_score*100:+.1f}%)")
    print(f"  +Manager+Entropy:      {full_final_score:.2f}  ({(full_final_score-baseline_final_score)/baseline_final_score*100:+.1f}%)")
    print()
    
    if baseline_entropy and manager_entropy and full_entropy:
        baseline_final_entropy = np.mean(baseline_entropy[-window:])
        manager_final_entropy = np.mean(manager_entropy[-window:])
        full_final_entropy = np.mean(full_entropy[-window:])
        
        print(f"Alpha熵 (最后{window} episodes):")
        print(f"  Baseline:              {baseline_final_entropy:.4f}")
        print(f"  +Manager:              {manager_final_entropy:.4f}  ({(baseline_final_entropy-manager_final_entropy)/baseline_final_entropy*100:+.1f}%)")
        print(f"  +Manager+Entropy:      {full_final_entropy:.4f}  ({(baseline_final_entropy-full_final_entropy)/baseline_final_entropy*100:+.1f}%)")
        print()
    
    # 增量贡献
    print("增量贡献分析:")
    manager_contribution = manager_final_score - baseline_final_score
    full_contribution = full_final_score - manager_final_score
    
    print(f"  Manager约束贡献:       {manager_contribution:+.2f} ({manager_contribution/baseline_final_score*100:+.1f}%)")
    print(f"  熵最小化贡献:          {full_contribution:+.2f} ({full_contribution/baseline_final_score*100:+.1f}%)")
    print(f"  总贡献:                {full_final_score-baseline_final_score:+.2f} ({(full_final_score-baseline_final_score)/baseline_final_score*100:+.1f}%)")
    print()
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="消融实验结果对比")
    parser.add_argument("--baseline", type=str, required=True,
                        help="Baseline实验目录")
    parser.add_argument("--manager", type=str, required=True,
                        help="+Manager实验目录")
    parser.add_argument("--full", type=str, required=True,
                        help="+Manager+Entropy实验目录")
    parser.add_argument("--output", type=str, default="ablation_v3/visualizations/ablation_study",
                        help="输出目录")
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载日志
    print("加载训练日志...")
    baseline_log = load_log(Path(args.baseline))
    manager_log = load_log(Path(args.manager))
    full_log = load_log(Path(args.full))
    print("✓ 日志加载完成")
    
    # 绘制对比图
    print("\n绘制对比图...")
    plot_comparison(baseline_log, manager_log, full_log, output_dir)
    
    # 打印摘要
    print_summary(baseline_log, manager_log, full_log)
    
    print(f"✓ 分析完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
