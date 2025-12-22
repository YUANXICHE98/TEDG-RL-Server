#!/usr/bin/env python3
"""
V2 消融实验结果可视化
包含：学习曲线、Actor权重演化、场景-Actor对应分析
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
})

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = Path(__file__).parent / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

EXPERIMENTS = {
    "baseline": {"name": "Baseline", "color": "#2ecc71"},
    "no_mask": {"name": "No Mask (Beat This!)", "color": "#7f8c8d"},
    "gumbel": {"name": "Gumbel-Softmax", "color": "#e74c3c"},
    "sparse_moe": {"name": "Sparse MoE (Top-2)", "color": "#3498db"},
    "gumbel_sparse": {"name": "Gumbel + Sparse", "color": "#9b59b6"},
    "hram_doc": {"name": "H-RAM (Doc)", "color": "#f39c12"},
    "hram_e2e": {"name": "H-RAM (E2E)", "color": "#1abc9c"},
}

ACTOR_NAMES = ["Pre", "Scene", "Effect", "Rule"]
ACTOR_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]


def load_log(exp_name: str) -> Dict:
    log_path = RESULTS_DIR / exp_name / "logs" / "training_log.json"
    if log_path.exists():
        with open(log_path, 'r') as f:
            return json.load(f)
    return None


def smooth(data: List[float], window: int = 100) -> np.ndarray:
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def compute_ci(data: List[float], window: int = 200):
    """计算滚动均值和标准差"""
    data = np.array(data)
    n = len(data)
    if n < window:
        return np.array([np.mean(data)]), np.array([np.std(data)]), np.array([0])
    
    means, stds, episodes = [], [], []
    for i in range(0, n - window + 1, window // 4):
        chunk = data[i:i+window]
        means.append(np.mean(chunk))
        stds.append(np.std(chunk))
        episodes.append(i + window // 2)
    
    return np.array(means), np.array(stds), np.array(episodes)


def plot_learning_curves(all_data: Dict, output_path: Path):
    """学习曲线（带置信区间）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：Score
    ax1 = axes[0]
    for exp_name, config in EXPERIMENTS.items():
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        scores = all_data[exp_name].get("episode_scores", [])
        if scores and len(scores) >= 100:
            means, stds, eps = compute_ci(scores, window=200)
            ax1.plot(eps, means, label=config["name"], color=config["color"], linewidth=2)
            ax1.fill_between(eps, means-stds, means+stds, color=config["color"], alpha=0.15)
    
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Game Score")
    ax1.set_title("(A) Game Score (Mean ± Std)")
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 右图：Episode Length
    ax2 = axes[1]
    for exp_name, config in EXPERIMENTS.items():
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        lengths = all_data[exp_name].get("episode_lengths", [])
        if lengths and len(lengths) >= 100:
            means, stds, eps = compute_ci(lengths, window=200)
            ax2.plot(eps, means, label=config["name"], color=config["color"], linewidth=2)
            ax2.fill_between(eps, means-stds, means+stds, color=config["color"], alpha=0.15)
    
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("(B) Episode Length (Survival)")
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ {output_path.name}")


def plot_actor_weight_evolution(all_data: Dict, output_path: Path):
    """Actor权重随时间演化"""
    valid_exps = [e for e in EXPERIMENTS if e in all_data and all_data[e] 
                  and all_data[e].get("alpha_history")]
    
    if not valid_exps:
        print("  ⚠ 无 alpha_history 数据")
        return
    
    n = len(valid_exps)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    
    for idx, exp_name in enumerate(valid_exps):
        ax = axes[idx]
        config = EXPERIMENTS[exp_name]
        alpha = np.array(all_data[exp_name]["alpha_history"])
        
        if alpha.ndim == 2 and alpha.shape[1] >= 4:
            window = min(100, len(alpha) // 10)
            for i, (name, color) in enumerate(zip(ACTOR_NAMES, ACTOR_COLORS)):
                smoothed = smooth(alpha[:, i], window=window)
                ax.plot(range(len(smoothed)), smoothed, label=name, color=color, linewidth=2)
            
            ax.set_xlabel("Episode")
            ax.set_ylabel("Weight")
            ax.set_title(config["name"])
            ax.legend(fontsize=8)
            ax.set_ylim(0, 0.6)
            ax.grid(True, alpha=0.3)
    
    for idx in range(len(valid_exps), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Actor Weight Evolution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ {output_path.name}")


def plot_scene_actor_heatmap(all_data: Dict, output_path: Path):
    """场景-Actor 热力图：HP vs Dominant Actor"""
    valid_exps = [e for e in EXPERIMENTS if e in all_data and all_data[e]
                  and all_data[e].get("scene_actor_samples")]
    
    if not valid_exps:
        print("  ⚠ 无 scene_actor_samples 数据")
        return
    
    n = len(valid_exps)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    
    for idx, exp_name in enumerate(valid_exps):
        ax = axes[idx]
        config = EXPERIMENTS[exp_name]
        samples = all_data[exp_name]["scene_actor_samples"]
        
        if not samples:
            continue
        
        # 按 HP 区间统计 dominant actor 分布
        hp_bins = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        hp_labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
        
        actor_counts = np.zeros((4, 4))  # HP bins x Actors
        for s in samples:
            hp = s["hp_ratio"]
            dom = s["dominant"]
            for i, (lo, hi) in enumerate(hp_bins):
                if lo <= hp < hi or (hi == 1.0 and hp == 1.0):
                    actor_counts[i, dom] += 1
                    break
        
        # 归一化为百分比
        row_sums = actor_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        actor_pct = actor_counts / row_sums * 100
        
        im = ax.imshow(actor_pct, cmap='YlOrRd', aspect='auto', vmin=0, vmax=50)
        ax.set_xticks(range(4))
        ax.set_xticklabels(ACTOR_NAMES)
        ax.set_yticks(range(4))
        ax.set_yticklabels(hp_labels)
        ax.set_xlabel("Dominant Actor")
        ax.set_ylabel("HP Range")
        ax.set_title(config["name"])
        
        # 添加数值标签
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{actor_pct[i,j]:.0f}%', ha='center', va='center', fontsize=9)
    
    for idx in range(len(valid_exps), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Scene-Actor Correspondence: HP vs Dominant Actor", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ {output_path.name}")


def plot_actor_by_depth(all_data: Dict, output_path: Path):
    """Actor 权重按深度分布"""
    valid_exps = [e for e in EXPERIMENTS if e in all_data and all_data[e]
                  and all_data[e].get("scene_actor_samples")]
    
    if not valid_exps:
        return
    
    n = len(valid_exps)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    
    for idx, exp_name in enumerate(valid_exps):
        ax = axes[idx]
        config = EXPERIMENTS[exp_name]
        samples = all_data[exp_name]["scene_actor_samples"]
        
        if not samples:
            continue
        
        # 按深度统计平均 alpha
        depth_alphas = defaultdict(list)
        for s in samples:
            depth = min(s["depth"], 10)  # 截断到10层
            depth_alphas[depth].append(s["alpha"])
        
        depths = sorted(depth_alphas.keys())
        mean_alphas = np.array([np.mean(depth_alphas[d], axis=0) for d in depths])
        
        x = np.arange(len(depths))
        width = 0.2
        for i, (name, color) in enumerate(zip(ACTOR_NAMES, ACTOR_COLORS)):
            ax.bar(x + i*width, mean_alphas[:, i], width, label=name, color=color, alpha=0.8)
        
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels([f'D{d}' for d in depths])
        ax.set_xlabel("Dungeon Level")
        ax.set_ylabel("Mean Weight")
        ax.set_title(config["name"])
        ax.legend(fontsize=8)
        ax.set_ylim(0, 0.5)
    
    for idx in range(len(valid_exps), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Actor Weights by Dungeon Depth", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ {output_path.name}")


def plot_final_comparison(all_data: Dict, output_path: Path):
    """最终性能对比柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    exp_names, mean_scores, std_scores, best_scores, colors = [], [], [], [], []
    
    for exp_name, config in EXPERIMENTS.items():
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        scores = all_data[exp_name].get("episode_scores", [])
        if not scores:
            continue
        
        final = scores[-500:] if len(scores) >= 500 else scores
        exp_names.append(config["name"])
        mean_scores.append(np.mean(final))
        std_scores.append(np.std(final))
        best_scores.append(max(scores))
        colors.append(config["color"])
    
    if not exp_names:
        print("  ⚠ 无数据")
        return
    
    # 排序
    sorted_idx = np.argsort(mean_scores)[::-1]
    exp_names = [exp_names[i] for i in sorted_idx]
    mean_scores = [mean_scores[i] for i in sorted_idx]
    std_scores = [std_scores[i] for i in sorted_idx]
    best_scores = [best_scores[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    
    # 左图：平均分
    ax1 = axes[0]
    x = np.arange(len(exp_names))
    bars1 = ax1.bar(x, mean_scores, yerr=std_scores, color=colors, alpha=0.8, capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_names, rotation=20, ha='right')
    ax1.set_ylabel("Mean Score (Last 500 Episodes)")
    ax1.set_title("(A) Average Performance")
    for bar, val in zip(bars1, mean_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    # 右图：最佳分
    ax2 = axes[1]
    bars2 = ax2.bar(x, best_scores, color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_names, rotation=20, ha='right')
    ax2.set_ylabel("Best Score")
    ax2.set_title("(B) Peak Performance")
    for bar, val in zip(bars2, best_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ {output_path.name}")


def plot_summary_dashboard(all_data: Dict, output_path: Path):
    """总结仪表板"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)
    
    # (A) 学习曲线
    ax1 = fig.add_subplot(gs[0, 0])
    for exp_name, config in EXPERIMENTS.items():
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        scores = all_data[exp_name].get("episode_scores", [])
        if scores and len(scores) >= 100:
            means, stds, eps = compute_ci(scores, window=200)
            ax1.plot(eps, means, label=config["name"], color=config["color"], linewidth=2)
            ax1.fill_between(eps, means-stds, means+stds, color=config["color"], alpha=0.1)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.set_title("(A) Learning Curves")
    ax1.legend(fontsize=8, loc='lower right')
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
            exp_names.append(config["name"].replace(" ", "\n"))
            mean_scores.append(np.mean(final))
            colors.append(config["color"])
    
    if exp_names:
        sorted_idx = np.argsort(mean_scores)[::-1]
        bars = ax2.barh([exp_names[i] for i in sorted_idx][::-1], 
                       [mean_scores[i] for i in sorted_idx][::-1],
                       color=[colors[i] for i in sorted_idx][::-1], alpha=0.8)
        ax2.set_xlabel("Mean Score")
        ax2.set_title("(B) Final Performance")
    
    # (C) Episode Length
    ax3 = fig.add_subplot(gs[1, 0])
    for exp_name, config in EXPERIMENTS.items():
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        lengths = all_data[exp_name].get("episode_lengths", [])
        if lengths and len(lengths) >= 100:
            means, stds, eps = compute_ci(lengths, window=200)
            ax3.plot(eps, means, label=config["name"], color=config["color"], linewidth=2)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Steps")
    ax3.set_title("(C) Episode Length (Survival)")
    ax3.legend(fontsize=8, loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # (D) Actor 权重 (Gumbel vs Baseline)
    ax4 = fig.add_subplot(gs[1, 1])
    for exp_name in ["embedding2000", "gumbel"]:
        if exp_name not in all_data or not all_data[exp_name]:
            continue
        alpha = all_data[exp_name].get("alpha_history", [])
        if alpha and len(alpha) >= 100:
            alpha = np.array(alpha)
            if alpha.ndim == 2 and alpha.shape[1] >= 4:
                # 计算权重方差
                stds = np.std(alpha, axis=1)
                smoothed = smooth(stds, window=100)
                config = EXPERIMENTS[exp_name]
                ax4.plot(range(len(smoothed)), smoothed, label=config["name"], 
                        color=config["color"], linewidth=2)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Weight Std (Differentiation)")
    ax4.set_title("(D) Actor Weight Differentiation")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle("TEDG-RL V2 Ablation Study Results", fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(output_path)
    plt.close()
    print(f"✓ {output_path.name}")


def main():
    print("=" * 60)
    print("V2 消融实验结果可视化")
    print("=" * 60)
    
    all_data = {}
    for exp_name in EXPERIMENTS:
        data = load_log(exp_name)
        if data:
            all_data[exp_name] = data
            scores = data.get("episode_scores", [])
            samples = len(data.get("scene_actor_samples", []))
            converged = data.get("converged", False)
            stopped = data.get("stopped_at", len(scores))
            print(f"  ✓ {exp_name}: {len(scores)} eps, best={max(scores) if scores else 0}, "
                  f"samples={samples}, converged={converged}")
        else:
            print(f"  ✗ {exp_name}: 未找到")
    
    if not all_data:
        print("\n❌ 没有数据，请先运行实验")
        return
    
    print(f"\n生成图表...")
    
    plot_learning_curves(all_data, OUTPUT_DIR / "fig1_learning_curves.png")
    plot_actor_weight_evolution(all_data, OUTPUT_DIR / "fig2_actor_evolution.png")
    plot_scene_actor_heatmap(all_data, OUTPUT_DIR / "fig3_scene_actor_heatmap.png")
    plot_actor_by_depth(all_data, OUTPUT_DIR / "fig4_actor_by_depth.png")
    plot_final_comparison(all_data, OUTPUT_DIR / "fig5_final_comparison.png")
    plot_summary_dashboard(all_data, OUTPUT_DIR / "fig6_summary_dashboard.png")
    
    print(f"\n✅ 图表保存在: {OUTPUT_DIR}")
    print("\n关键图表说明:")
    print("  fig3_scene_actor_heatmap.png - 场景(HP)-Actor对应热力图 ⭐")
    print("  fig4_actor_by_depth.png - Actor权重按深度分布")
    print("  fig6_summary_dashboard.png - 论文主图")


if __name__ == "__main__":
    main()
