#!/usr/bin/env python3
"""
V1 消融实验结果可视化
绘制所有实验的对比图表
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 实验目录
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = Path(__file__).parent / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# 实验配置 (所有6个实验)
EXPERIMENTS = {
    "results_embedding": {"name": "Embedding", "color": "#2ecc71", "marker": "o"},
    "results_fixed_th": {"name": "Fixed Threshold", "color": "#e74c3c", "marker": "^"},
    "results_full": {"name": "Full Pipeline", "color": "#9b59b6", "marker": "d"},
    "results_no_mask": {"name": "No Mask", "color": "#f39c12", "marker": "s"},
    "results_single_ch": {"name": "Single Channel", "color": "#1abc9c", "marker": "v"},
    "results_extended_steps": {"name": "Extended Steps", "color": "#3498db", "marker": "p"},
}


def load_training_log(exp_dir: Path) -> Dict:
    """加载训练日志"""
    log_path = exp_dir / "logs" / "training_log.json"
    if log_path.exists():
        with open(log_path, 'r') as f:
            return json.load(f)
    return None


def smooth_curve(data: List[float], window: int = 50) -> np.ndarray:
    """平滑曲线"""
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_reward_comparison(all_data: Dict, output_path: Path):
    """绘制奖励对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：原始奖励曲线
    ax1 = axes[0]
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            rewards = all_data[exp_name].get("episode_rewards", [])
            if rewards:
                episodes = range(len(rewards))
                ax1.plot(episodes, rewards, alpha=0.3, color=config["color"])
                smoothed = smooth_curve(rewards)
                ax1.plot(range(len(smoothed)), smoothed, 
                        label=config["name"], color=config["color"], linewidth=2)
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Training Reward Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：平滑后的奖励
    ax2 = axes[1]
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            rewards = all_data[exp_name].get("episode_rewards", [])
            if rewards:
                smoothed = smooth_curve(rewards, window=100)
                ax2.plot(range(len(smoothed)), smoothed,
                        label=config["name"], color=config["color"], 
                        linewidth=2, marker=config["marker"], markevery=len(smoothed)//10)
    
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Smoothed Reward (window=100)")
    ax2.set_title("Smoothed Training Reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {output_path}")


def plot_score_comparison(all_data: Dict, output_path: Path):
    """绘制分数对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：分数曲线
    ax1 = axes[0]
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            scores = all_data[exp_name].get("episode_scores", [])
            if scores:
                smoothed = smooth_curve(scores, window=50)
                ax1.plot(range(len(smoothed)), smoothed,
                        label=config["name"], color=config["color"], linewidth=2)
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Game Score")
    ax1.set_title("Game Score Over Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：最终分数分布（箱线图）
    ax2 = axes[1]
    box_data = []
    labels = []
    colors = []
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            scores = all_data[exp_name].get("episode_scores", [])
            if scores:
                # 取最后500个episode的分数
                final_scores = scores[-500:] if len(scores) >= 500 else scores
                box_data.append(final_scores)
                labels.append(config["name"])
                colors.append(config["color"])
    
    if box_data:
        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    ax2.set_ylabel("Final Score Distribution")
    ax2.set_title("Score Distribution (Last 500 Episodes)")
    ax2.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {output_path}")


def plot_episode_length(all_data: Dict, output_path: Path):
    """绘制Episode长度对比图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            lengths = all_data[exp_name].get("episode_lengths", [])
            if lengths:
                smoothed = smooth_curve(lengths, window=50)
                ax.plot(range(len(smoothed)), smoothed,
                       label=config["name"], color=config["color"], linewidth=2)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length (Steps)")
    ax.set_title("Episode Length Over Training (Longer = Agent Survives Longer)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {output_path}")


def plot_final_stats(all_data: Dict, output_path: Path):
    """绘制最终统计对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    exp_names = []
    best_rewards = []
    best_scores = []
    avg_lengths = []
    colors = []
    
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            data = all_data[exp_name]
            exp_names.append(config["name"])
            colors.append(config["color"])
            
            rewards = data.get("episode_rewards", [0])
            scores = data.get("episode_scores", [0])
            lengths = data.get("episode_lengths", [0])
            
            best_rewards.append(max(rewards) if rewards else 0)
            best_scores.append(max(scores) if scores else 0)
            avg_lengths.append(np.mean(lengths[-500:]) if len(lengths) >= 500 else np.mean(lengths))
    
    # 最佳奖励
    ax1 = axes[0]
    bars1 = ax1.bar(exp_names, best_rewards, color=colors, alpha=0.7)
    ax1.set_ylabel("Best Reward")
    ax1.set_title("Best Episode Reward")
    ax1.bar_label(bars1, fmt='%.1f')
    plt.sca(ax1)
    plt.xticks(rotation=15)
    
    # 最佳分数
    ax2 = axes[1]
    bars2 = ax2.bar(exp_names, best_scores, color=colors, alpha=0.7)
    ax2.set_ylabel("Best Score")
    ax2.set_title("Best Game Score")
    ax2.bar_label(bars2, fmt='%.0f')
    plt.sca(ax2)
    plt.xticks(rotation=15)
    
    # 平均Episode长度
    ax3 = axes[2]
    bars3 = ax3.bar(exp_names, avg_lengths, color=colors, alpha=0.7)
    ax3.set_ylabel("Avg Length")
    ax3.set_title("Avg Episode Length (Last 500)")
    ax3.bar_label(bars3, fmt='%.0f')
    plt.sca(ax3)
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {output_path}")


def plot_learning_efficiency(all_data: Dict, output_path: Path):
    """绘制学习效率对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：累计奖励
    ax1 = axes[0]
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            rewards = all_data[exp_name].get("episode_rewards", [])
            if rewards:
                cumsum = np.cumsum(rewards)
                ax1.plot(range(len(cumsum)), cumsum,
                        label=config["name"], color=config["color"], linewidth=2)
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("Cumulative Reward (Learning Efficiency)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：每100 episode的平均奖励
    ax2 = axes[1]
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            rewards = all_data[exp_name].get("episode_rewards", [])
            if rewards and len(rewards) >= 100:
                chunks = [rewards[i:i+100] for i in range(0, len(rewards)-100, 100)]
                means = [np.mean(chunk) for chunk in chunks]
                ax2.plot(range(len(means)), means,
                        label=config["name"], color=config["color"], 
                        linewidth=2, marker=config["marker"])
    
    ax2.set_xlabel("Episode Block (x100)")
    ax2.set_ylabel("Mean Reward per 100 Episodes")
    ax2.set_title("Learning Progress (100-Episode Blocks)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {output_path}")


def plot_actor_weights(all_data: Dict, output_path: Path):
    """绘制 Actor 权重分布图"""
    # 只选择有 alpha_history 的实验
    valid_exps = []
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            alpha = all_data[exp_name].get("alpha_history", [])
            if alpha and len(alpha) > 0:
                valid_exps.append(exp_name)
    
    if not valid_exps:
        print("  ⚠ 没有 alpha_history 数据，跳过 Actor 权重图")
        return
    
    n_exps = len(valid_exps)
    fig, axes = plt.subplots(2, (n_exps + 1) // 2, figsize=(5 * ((n_exps + 1) // 2), 10))
    axes = axes.flatten() if n_exps > 1 else [axes]
    
    actor_names = ["Pre", "Scene", "Effect", "Rule"]
    actor_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    
    for idx, exp_name in enumerate(valid_exps):
        ax = axes[idx]
        config = EXPERIMENTS[exp_name]
        alpha_history = all_data[exp_name].get("alpha_history", [])
        
        if alpha_history:
            alpha_array = np.array(alpha_history)
            # 计算每个 Actor 的平均权重随时间变化
            window = 100
            if len(alpha_array) >= window:
                for i, (name, color) in enumerate(zip(actor_names, actor_colors)):
                    if alpha_array.ndim == 2 and alpha_array.shape[1] > i:
                        weights = alpha_array[:, i]
                        smoothed = smooth_curve(weights, window=window)
                        ax.plot(range(len(smoothed)), smoothed, 
                               label=name, color=color, linewidth=2)
            
            ax.set_xlabel("Episode")
            ax.set_ylabel("Weight")
            ax.set_title(f"{config['name']}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
    
    # 隐藏多余的子图
    for idx in range(len(valid_exps), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Actor Weight Distribution Over Training", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {output_path}")


def plot_actor_comparison(all_data: Dict, output_path: Path):
    """绘制各实验的 Actor 最终权重对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    actor_names = ["Pre", "Scene", "Effect", "Rule"]
    actor_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    
    # 左图：各实验的平均 Actor 权重
    ax1 = axes[0]
    exp_labels = []
    weight_data = {name: [] for name in actor_names}
    
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            alpha = all_data[exp_name].get("alpha_history", [])
            if alpha and len(alpha) > 100:
                alpha_array = np.array(alpha)
                if alpha_array.ndim == 2 and alpha_array.shape[1] >= 4:
                    exp_labels.append(config["name"])
                    # 取最后500个episode的平均
                    final_alpha = alpha_array[-500:] if len(alpha_array) >= 500 else alpha_array
                    mean_weights = np.mean(final_alpha, axis=0)
                    for i, name in enumerate(actor_names):
                        weight_data[name].append(mean_weights[i])
    
    if exp_labels:
        x = np.arange(len(exp_labels))
        width = 0.2
        for i, (name, color) in enumerate(zip(actor_names, actor_colors)):
            ax1.bar(x + i * width, weight_data[name], width, label=name, color=color, alpha=0.8)
        
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(exp_labels, rotation=15)
        ax1.set_ylabel("Average Weight")
        ax1.set_title("Final Actor Weights by Experiment")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：饼图 - Embedding 实验的 Actor 分布
    ax2 = axes[1]
    if "results_embedding" in all_data and all_data["results_embedding"]:
        alpha = all_data["results_embedding"].get("alpha_history", [])
        if alpha and len(alpha) > 100:
            alpha_array = np.array(alpha)
            if alpha_array.ndim == 2 and alpha_array.shape[1] >= 4:
                final_alpha = alpha_array[-500:] if len(alpha_array) >= 500 else alpha_array
                mean_weights = np.mean(final_alpha, axis=0)
                ax2.pie(mean_weights, labels=actor_names, colors=actor_colors,
                       autopct='%1.1f%%', startangle=90)
                ax2.set_title("Embedding: Actor Weight Distribution")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {output_path}")


def plot_summary_dashboard(all_data: Dict, output_path: Path):
    """绘制总结仪表板"""
    fig = plt.figure(figsize=(16, 12))
    
    # 2x2 布局
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # 1. 奖励曲线
    ax1 = fig.add_subplot(gs[0, 0])
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            rewards = all_data[exp_name].get("episode_rewards", [])
            if rewards:
                smoothed = smooth_curve(rewards, window=100)
                ax1.plot(range(len(smoothed)), smoothed,
                        label=config["name"], color=config["color"], linewidth=2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("(A) Training Reward")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. 分数曲线
    ax2 = fig.add_subplot(gs[0, 1])
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            scores = all_data[exp_name].get("episode_scores", [])
            if scores:
                smoothed = smooth_curve(scores, window=100)
                ax2.plot(range(len(smoothed)), smoothed,
                        label=config["name"], color=config["color"], linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Score")
    ax2.set_title("(B) Game Score")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. 最终统计柱状图
    ax3 = fig.add_subplot(gs[1, 0])
    exp_labels = []
    best_scores = []
    colors = []
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            scores = all_data[exp_name].get("episode_scores", [0])
            exp_labels.append(config["name"].replace(" ", "\n"))
            best_scores.append(max(scores) if scores else 0)
            colors.append(config["color"])
    
    if exp_labels:
        bars = ax3.bar(exp_labels, best_scores, color=colors, alpha=0.7)
        ax3.bar_label(bars, fmt='%.0f', fontsize=10)
    ax3.set_ylabel("Best Score")
    ax3.set_title("(C) Best Score Comparison")
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Episode长度
    ax4 = fig.add_subplot(gs[1, 1])
    for exp_name, config in EXPERIMENTS.items():
        if exp_name in all_data and all_data[exp_name]:
            lengths = all_data[exp_name].get("episode_lengths", [])
            if lengths:
                smoothed = smooth_curve(lengths, window=100)
                ax4.plot(range(len(smoothed)), smoothed,
                        label=config["name"], color=config["color"], linewidth=2)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Steps")
    ax4.set_title("(D) Episode Length (Survival)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle("TEDG-RL V1 Ablation Study Results", fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {output_path}")


def main():
    print("=" * 60)
    print("V1 消融实验结果可视化")
    print("=" * 60)
    
    # 加载所有实验数据
    all_data = {}
    for exp_name in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp_name
        if exp_dir.exists():
            data = load_training_log(exp_dir)
            if data:
                all_data[exp_name] = data
                rewards = data.get("episode_rewards", [])
                scores = data.get("episode_scores", [])
                print(f"  ✓ {exp_name}: {len(rewards)} episodes, best score={max(scores) if scores else 0}")
            else:
                print(f"  ✗ {exp_name}: 无训练日志")
        else:
            print(f"  ✗ {exp_name}: 目录不存在")
    
    if not all_data:
        print("\n❌ 没有找到任何实验数据")
        return
    
    print(f"\n生成可视化图表...")
    
    # 生成各种图表
    plot_reward_comparison(all_data, OUTPUT_DIR / "01_reward_comparison.png")
    plot_score_comparison(all_data, OUTPUT_DIR / "02_score_comparison.png")
    plot_episode_length(all_data, OUTPUT_DIR / "03_episode_length.png")
    plot_final_stats(all_data, OUTPUT_DIR / "04_final_stats.png")
    plot_learning_efficiency(all_data, OUTPUT_DIR / "05_learning_efficiency.png")
    plot_actor_weights(all_data, OUTPUT_DIR / "06_actor_weights.png")
    plot_actor_comparison(all_data, OUTPUT_DIR / "07_actor_comparison.png")
    plot_summary_dashboard(all_data, OUTPUT_DIR / "08_summary_dashboard.png")
    
    print(f"\n✅ 所有图表保存在: {OUTPUT_DIR}")
    print("\n生成的图表:")
    print("  01_reward_comparison.png  - 奖励曲线对比")
    print("  02_score_comparison.png   - 分数曲线与分布")
    print("  03_episode_length.png     - Episode长度（生存能力）")
    print("  04_final_stats.png        - 最终统计柱状图")
    print("  05_learning_efficiency.png - 学习效率对比")
    print("  06_actor_weights.png      - Actor权重随时间变化")
    print("  07_actor_comparison.png   - Actor权重对比（柱状图+饼图）")
    print("  08_summary_dashboard.png  - 总结仪表板")


if __name__ == "__main__":
    main()
