#!/usr/bin/env python3
"""
V1 消融实验 - 论文级别可视化
特点：大平滑窗口、置信区间、专业配色
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats

# 论文风格设置
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
})

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = Path(__file__).parent / "visualizations_paper"
OUTPUT_DIR.mkdir(exist_ok=True)

# 实验配置 - 按性能排序
EXPERIMENTS = {
    "results_no_mask": {"name": "No Mask", "color": "#e74c3c"},
    "results_embedding": {"name": "Embedding", "color": "#2ecc71"},
    "results_fixed_th": {"name": "Fixed Threshold", "color": "#3498db"},
    "results_full": {"name": "Full Pipeline", "color": "#9b59b6"},
    "results_single_ch": {"name": "Single Channel", "color": "#f39c12"},
    "results_extended_steps": {"name": "Extended (2000)", "color": "#1abc9c"},
}


def load_training_log(exp_dir: Path) -> Dict:
    log_path = exp_dir / "logs" / "training_log.json"
    if log_path.exists():
        with open(log_path, 'r') as f:
            return json.load(f)
    return None


def compute_rolling_stats(data: List[float], window: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算滚动均值和标准差"""
    data = np.array(data)
    n = len(data)
    if n < window:
        return np.array([np.mean(data)]), np.array([np.std(data)]), np.array([0])
    
    means = []
    stds = []
    episodes = []
    
    for i in range(0, n - window + 1, window // 4):  # 25% 步进
        chunk = data[i:i+window]
        means.append(np.mean(chunk))
        stds.append(np.std(chunk))
        episodes.append(i + window // 2)
    
    return np.array(means), np.array(stds), np.array(episodes)


def plot_learning_curves_with_ci(all_data: Dict, output_path: Path):
    """绘制学习曲线（带置信区间）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for exp_name, config in EXPERIMENTS.items():
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        
        rewards = all_data[exp_name].get("episode_rewards", [])
        if not rewards or len(rewards) < 100:
            continue
        
        means, stds, episodes = compute_rolling_stats(rewards, window=200)
        
        # 绘制均值线
        ax.plot(episodes, means, label=config["name"], color=config["color"], linewidth=2.5)
        
        # 绘制置信区间 (mean ± std)
        ax.fill_between(episodes, means - stds, means + stds, 
                       color=config["color"], alpha=0.15)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves (Mean ± Std, window=200)")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()
    print(f"✓ {output_path.name}")


def plot_score_curves_with_ci(all_data: Dict, output_path: Path):
    """绘制分数曲线（带置信区间）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for exp_name, config in EXPERIMENTS.items():
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        
        scores = all_data[exp_name].get("episode_scores", [])
        if not scores or len(scores) < 100:
            continue
        
        means, stds, episodes = compute_rolling_stats(scores, window=200)
        
        ax.plot(episodes, means, label=config["name"], color=config["color"], linewidth=2.5)
        ax.fill_between(episodes, means - stds, means + stds,
                       color=config["color"], alpha=0.15)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Game Score")
    ax.set_title("Game Score Over Training (Mean ± Std, window=200)")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()
    print(f"✓ {output_path.name}")


def plot_final_performance(all_data: Dict, output_path: Path):
    """绘制最终性能柱状图（带误差条）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    exp_names = []
    mean_scores = []
    std_scores = []
    best_scores = []
    colors = []
    
    for exp_name, config in EXPERIMENTS.items():
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        
        scores = all_data[exp_name].get("episode_scores", [])
        if not scores:
            continue
        
        # 取最后 500 episodes
        final_scores = scores[-500:] if len(scores) >= 500 else scores
        
        exp_names.append(config["name"])
        mean_scores.append(np.mean(final_scores))
        std_scores.append(np.std(final_scores))
        best_scores.append(max(scores))
        colors.append(config["color"])
    
    # 按均值排序
    sorted_idx = np.argsort(mean_scores)[::-1]
    exp_names = [exp_names[i] for i in sorted_idx]
    mean_scores = [mean_scores[i] for i in sorted_idx]
    std_scores = [std_scores[i] for i in sorted_idx]
    best_scores = [best_scores[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    
    # 左图：平均分数 + 误差条
    ax1 = axes[0]
    x = np.arange(len(exp_names))
    bars1 = ax1.bar(x, mean_scores, yerr=std_scores, color=colors, alpha=0.8,
                   capsize=5, error_kw={'linewidth': 1.5})
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_names, rotation=20, ha='right')
    ax1.set_ylabel("Mean Score (Last 500 Episodes)")
    ax1.set_title("Average Performance")
    
    # 添加数值标签
    for bar, val in zip(bars1, mean_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 右图：最佳分数
    ax2 = axes[1]
    bars2 = ax2.bar(x, best_scores, color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_names, rotation=20, ha='right')
    ax2.set_ylabel("Best Score")
    ax2.set_title("Peak Performance")
    
    for bar, val in zip(bars2, best_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ {output_path.name}")


def plot_sample_efficiency(all_data: Dict, output_path: Path):
    """绘制样本效率图（x轴为总步数）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for exp_name, config in EXPERIMENTS.items():
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        
        scores = all_data[exp_name].get("episode_scores", [])
        lengths = all_data[exp_name].get("episode_lengths", [])
        
        if not scores or not lengths or len(scores) < 100:
            continue
        
        # 计算累计步数
        cumsum_steps = np.cumsum(lengths)
        
        # 对分数进行平滑
        window = 200
        if len(scores) >= window:
            smoothed_scores = []
            step_points = []
            for i in range(0, len(scores) - window + 1, window // 4):
                smoothed_scores.append(np.mean(scores[i:i+window]))
                step_points.append(cumsum_steps[i + window // 2])
            
            ax.plot(np.array(step_points) / 1e6, smoothed_scores, 
                   label=config["name"], color=config["color"], linewidth=2.5)
    
    ax.set_xlabel("Total Steps (Million)")
    ax.set_ylabel("Game Score")
    ax.set_title("Sample Efficiency (Score vs Total Steps)")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()
    print(f"✓ {output_path.name}")


def plot_actor_weight_evolution(all_data: Dict, output_path: Path):
    """绘制 Actor 权重演化图"""
    # 只用 Embedding 实验
    if "results_embedding" not in all_data or not all_data["results_embedding"]:
        return
    
    alpha = all_data["results_embedding"].get("alpha_history", [])
    if not alpha or len(alpha) < 100:
        return
    
    alpha_array = np.array(alpha)
    if alpha_array.ndim != 2 or alpha_array.shape[1] < 4:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    actor_names = ["Pre", "Scene", "Effect", "Rule"]
    actor_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    
    # 左图：权重随时间变化（堆叠面积图）
    ax1 = axes[0]
    window = 100
    smoothed = []
    for i in range(4):
        weights = alpha_array[:, i]
        if len(weights) >= window:
            kernel = np.ones(window) / window
            smoothed.append(np.convolve(weights, kernel, mode='valid'))
    
    if smoothed:
        smoothed = np.array(smoothed)
        episodes = range(len(smoothed[0]))
        ax1.stackplot(episodes, smoothed, labels=actor_names, colors=actor_colors, alpha=0.8)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Weight")
        ax1.set_title("Actor Weight Evolution (Embedding)")
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, 1)
    
    # 右图：最终分布饼图
    ax2 = axes[1]
    final_alpha = alpha_array[-500:] if len(alpha_array) >= 500 else alpha_array
    mean_weights = np.mean(final_alpha, axis=0)
    
    wedges, texts, autotexts = ax2.pie(mean_weights, labels=actor_names, colors=actor_colors,
                                        autopct='%1.1f%%', startangle=90,
                                        wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    ax2.set_title("Final Actor Weight Distribution")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ {output_path.name}")


def plot_ablation_summary(all_data: Dict, output_path: Path):
    """绘制消融研究总结图（论文主图）"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)
    
    # (A) 学习曲线
    ax1 = fig.add_subplot(gs[0, 0])
    for exp_name, config in EXPERIMENTS.items():
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        scores = all_data[exp_name].get("episode_scores", [])
        if scores and len(scores) >= 100:
            means, stds, episodes = compute_rolling_stats(scores, window=200)
            ax1.plot(episodes, means, label=config["name"], color=config["color"], linewidth=2)
            ax1.fill_between(episodes, means - stds, means + stds, color=config["color"], alpha=0.1)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.set_title("(A) Learning Curves")
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # (B) 最终性能
    ax2 = fig.add_subplot(gs[0, 1])
    exp_names, mean_scores, colors = [], [], []
    for exp_name, config in EXPERIMENTS.items():
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        scores = all_data[exp_name].get("episode_scores", [])
        if scores:
            final = scores[-500:] if len(scores) >= 500 else scores
            exp_names.append(config["name"])
            mean_scores.append(np.mean(final))
            colors.append(config["color"])
    
    sorted_idx = np.argsort(mean_scores)[::-1]
    exp_names = [exp_names[i] for i in sorted_idx]
    mean_scores = [mean_scores[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    
    bars = ax2.barh(exp_names[::-1], mean_scores[::-1], color=colors[::-1], alpha=0.8)
    ax2.set_xlabel("Mean Score (Last 500 Episodes)")
    ax2.set_title("(B) Final Performance Comparison")
    for bar, val in zip(bars, mean_scores[::-1]):
        ax2.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.0f}', 
                va='center', fontsize=10)
    
    # (C) 样本效率
    ax3 = fig.add_subplot(gs[1, 0])
    for exp_name, config in EXPERIMENTS.items():
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        scores = all_data[exp_name].get("episode_scores", [])
        lengths = all_data[exp_name].get("episode_lengths", [])
        if scores and lengths and len(scores) >= 100:
            cumsum_steps = np.cumsum(lengths)
            means, _, episodes = compute_rolling_stats(scores, window=200)
            step_points = [cumsum_steps[min(e, len(cumsum_steps)-1)] for e in episodes.astype(int)]
            ax3.plot(np.array(step_points)/1e6, means, label=config["name"], 
                    color=config["color"], linewidth=2)
    ax3.set_xlabel("Total Steps (Million)")
    ax3.set_ylabel("Score")
    ax3.set_title("(C) Sample Efficiency")
    ax3.legend(fontsize=9, loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # (D) Actor 权重（仅 Embedding）
    ax4 = fig.add_subplot(gs[1, 1])
    if "results_embedding" in all_data and all_data["results_embedding"]:
        alpha = all_data["results_embedding"].get("alpha_history", [])
        if alpha and len(alpha) >= 100:
            alpha_array = np.array(alpha)
            if alpha_array.ndim == 2 and alpha_array.shape[1] >= 4:
                actor_names = ["Pre", "Scene", "Effect", "Rule"]
                actor_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
                window = 100
                for i, (name, color) in enumerate(zip(actor_names, actor_colors)):
                    weights = alpha_array[:, i]
                    kernel = np.ones(window) / window
                    smoothed = np.convolve(weights, kernel, mode='valid')
                    ax4.plot(range(len(smoothed)), smoothed, label=name, color=color, linewidth=2)
                ax4.set_xlabel("Episode")
                ax4.set_ylabel("Attention Weight")
                ax4.set_title("(D) Actor Attention Weights (Embedding)")
                ax4.legend(fontsize=9)
                ax4.set_ylim(0, 0.5)
                ax4.grid(True, alpha=0.3)
    
    plt.suptitle("TEDG-RL V1 Ablation Study", fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(output_path)
    plt.close()
    print(f"✓ {output_path.name}")


def main():
    print("=" * 60)
    print("V1 消融实验 - 论文级别可视化")
    print("=" * 60)
    
    all_data = {}
    for exp_name in EXPERIMENTS:
        exp_dir = RESULTS_DIR / exp_name
        if exp_dir.exists():
            data = load_training_log(exp_dir)
            if data:
                all_data[exp_name] = data
                scores = data.get("episode_scores", [])
                print(f"  ✓ {exp_name}: {len(scores)} eps, best={max(scores) if scores else 0}")
    
    if not all_data:
        print("❌ 没有数据")
        return
    
    print(f"\n生成论文级别图表...")
    
    plot_learning_curves_with_ci(all_data, OUTPUT_DIR / "fig1_learning_curves.png")
    plot_score_curves_with_ci(all_data, OUTPUT_DIR / "fig2_score_curves.png")
    plot_final_performance(all_data, OUTPUT_DIR / "fig3_final_performance.png")
    plot_sample_efficiency(all_data, OUTPUT_DIR / "fig4_sample_efficiency.png")
    plot_actor_weight_evolution(all_data, OUTPUT_DIR / "fig5_actor_weights.png")
    plot_ablation_summary(all_data, OUTPUT_DIR / "fig6_ablation_summary.png")
    
    print(f"\n✅ 论文图表保存在: {OUTPUT_DIR}")
    print("\n推荐用于论文的图:")
    print("  fig6_ablation_summary.png - 2x2 综合图（主图）")
    print("  fig3_final_performance.png - 性能对比柱状图")


if __name__ == "__main__":
    main()
