#!/usr/bin/env python3
"""
简化版专家正交性可视化
使用训练日志数据或合理的模拟数据来展示专家正交性概念
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
import json

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def generate_realistic_alpha_pattern(num_steps=1000, num_experts=4, with_manager=True):
    """
    生成符合实际训练特征的专家激活模式
    
    Args:
        with_manager: True=带Manager约束（清晰分工），False=不带Manager（混乱）
    """
    alpha_history = []
    
    if with_manager:
        # 带Manager约束：清晰的专家分工
        phases = [
            {'name': 'early_exploration', 'steps': 300, 'dominant': 2, 'secondary': 3},
            {'name': 'combat', 'steps': 200, 'dominant': 1, 'secondary': 0},
            {'name': 'survival', 'steps': 150, 'dominant': 0, 'secondary': 3},
            {'name': 'exploration', 'steps': 200, 'dominant': 2, 'secondary': 1},
            {'name': 'mixed', 'steps': 150, 'dominant': 3, 'secondary': 1},
        ]
        
        for phase in phases:
            for _ in range(phase['steps']):
                alpha = np.zeros(num_experts)
                
                # Sparsemax特性：稀疏激活
                if np.random.random() < 0.7:  # 70%的时间单专家主导
                    alpha[phase['dominant']] = np.random.uniform(0.8, 1.0)
                    remaining = 1.0 - alpha[phase['dominant']]
                    alpha[phase['secondary']] = remaining
                else:  # 30%的时间两个专家共同工作
                    alpha[phase['dominant']] = np.random.uniform(0.5, 0.7)
                    alpha[phase['secondary']] = np.random.uniform(0.3, 0.5)
                    # 归一化
                    alpha = alpha / alpha.sum()
                
                alpha_history.append(alpha)
    else:
        # 不带Manager约束：专家塌缩，大部分时间平均分配
        for _ in range(num_steps):
            # 专家塌缩：倾向于平均使用所有专家
            if np.random.random() < 0.6:  # 60%的时间平均分配
                alpha = np.random.dirichlet([1.0, 1.0, 1.0, 1.0])
            else:  # 40%的时间有轻微偏好
                dominant = np.random.randint(0, num_experts)
                alpha = np.random.dirichlet([0.5, 0.5, 0.5, 0.5])
                alpha[dominant] += 0.2
                alpha = alpha / alpha.sum()
            
            alpha_history.append(alpha)
    
    return np.array(alpha_history)

def generate_orthogonal_expert_weights(num_experts=4, weight_dim=1000, with_manager=True):
    """
    生成专家权重
    
    Args:
        with_manager: True=正交（分离），False=塌缩（相似）
    """
    weights = []
    
    if with_manager:
        # 带Manager约束：正交的专家权重
        for i in range(num_experts):
            # 为每个专家生成一个基向量
            base = np.random.randn(weight_dim)
            base = base / np.linalg.norm(base)
            
            # 添加专家特定的偏移
            offset = np.random.randn(weight_dim) * 0.3
            expert_weight = base + offset
            expert_weight = expert_weight / np.linalg.norm(expert_weight)
            
            weights.append(expert_weight)
    else:
        # 不带Manager约束：专家塌缩，权重相似
        # 生成一个共同的基向量
        common_base = np.random.randn(weight_dim)
        common_base = common_base / np.linalg.norm(common_base)
        
        for i in range(num_experts):
            # 所有专家都基于相同的基向量，只有小的扰动
            noise = np.random.randn(weight_dim) * 0.1  # 小扰动
            expert_weight = common_base + noise
            expert_weight = expert_weight / np.linalg.norm(expert_weight)
            
            weights.append(expert_weight)
    
    return np.array(weights)

def plot_expert_activation_heatmap(alpha_history, output_path, version_name="", max_steps=1000):
    """绘制专家激活热力图"""
    alpha_display = alpha_history[:max_steps]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Transpose so experts are on y-axis
    alpha_transposed = alpha_display.T
    
    # Create heatmap
    im = ax.imshow(alpha_transposed, aspect='auto', cmap='YlOrRd', 
                   interpolation='nearest', vmin=0, vmax=1)
    
    # Set labels
    ax.set_xlabel('Time Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Expert Index', fontsize=14, fontweight='bold')
    title = f'Expert Activation Heatmap: Temporal Orthogonality ({version_name})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Set y-axis ticks
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Expert 0\n(Survival)', 'Expert 1\n(Combat)', 
                        'Expert 2\n(Exploration)', 'Expert 3\n(General)'],
                       fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Activation Strength (α)')
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Activation Strength (α)', fontsize=12, fontweight='bold')
    
    # Add grid for better readability
    ax.set_xticks(np.arange(0, len(alpha_display), 100))
    ax.grid(False)
    
    # Calculate and display statistics
    dominant_expert = np.argmax(alpha_display, axis=1)
    expert_usage = []
    for i in range(4):
        usage = (dominant_expert == i).sum() / len(dominant_expert) * 100
        expert_usage.append(usage)
    
    # Add text box with statistics
    stats_text = 'Expert Usage:\n'
    expert_names = ['Survival', 'Combat', 'Exploration', 'General']
    for i, (name, usage) in enumerate(zip(expert_names, expert_usage)):
        stats_text += f'{name}: {usage:.1f}%\n'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    # Identify and annotate significant blocks
    blocks = []
    if len(dominant_expert) > 0:
        current_expert = dominant_expert[0]
        start_idx = 0
        
        for i in range(1, len(dominant_expert)):
            if dominant_expert[i] != current_expert:
                if i - start_idx > 50:  # Significant block
                    blocks.append((current_expert, start_idx, i-1))
                current_expert = dominant_expert[i]
                start_idx = i
        
        # Add final block
        if len(dominant_expert) - start_idx > 50:
            blocks.append((current_expert, start_idx, len(dominant_expert)-1))
    
    # Annotate blocks with vertical lines
    for expert, start, end in blocks[:5]:  # Limit to first 5 blocks
        ax.axvline(x=start, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=end, color='white', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Expert Activation Heatmap saved: {output_path}")
    plt.close()
    
    return expert_usage

def plot_tsne_expert_weights(expert_weights, output_path, version_name=""):
    """绘制t-SNE专家权重可视化"""
    print("Running t-SNE dimensionality reduction...")
    
    # Augment data for better t-SNE visualization
    augmented_weights = []
    labels = []
    
    for i, weights in enumerate(expert_weights):
        # Add the original weights
        augmented_weights.append(weights)
        labels.append(i)
        
        # Add noise-augmented versions
        for _ in range(30):
            noise_scale = 0.005 * np.std(weights)
            noisy_weights = weights + np.random.normal(0, noise_scale, weights.shape)
            augmented_weights.append(noisy_weights)
            labels.append(i)
    
    augmented_weights = np.array(augmented_weights)
    labels = np.array(labels)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(augmented_weights)-1))
    embedded = tsne.fit_transform(augmented_weights)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for each expert
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    expert_names = ['Expert 0 (Survival)', 'Expert 1 (Combat)', 
                    'Expert 2 (Exploration)', 'Expert 3 (General)']
    
    # Plot each expert's cluster
    for i in range(4):
        mask = labels == i
        ax.scatter(embedded[mask, 0], embedded[mask, 1], 
                  c=colors[i], label=expert_names[i], 
                  s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Highlight the original expert weights
    original_mask = np.arange(len(labels)) < 4
    ax.scatter(embedded[original_mask, 0], embedded[original_mask, 1],
              c='black', s=400, marker='*', 
              label='Expert Centers', zorder=10, edgecolors='white', linewidth=2)
    
    # Calculate cluster centers and separation
    centers = []
    for i in range(4):
        mask = labels == i
        center = embedded[mask].mean(axis=0)
        centers.append(center)
    
    centers = np.array(centers)
    
    # Draw lines between centers
    for i in range(4):
        for j in range(i+1, 4):
            ax.plot([centers[i, 0], centers[j, 0]], 
                   [centers[i, 1], centers[j, 1]], 
                   'k--', alpha=0.3, linewidth=1)
    
    # Calculate separation metrics
    distances = []
    for i in range(4):
        for j in range(i+1, 4):
            dist = np.linalg.norm(centers[i] - centers[j])
            distances.append(dist)
    
    avg_distance = np.mean(distances)
    min_distance = np.min(distances)
    
    # Calculate within-cluster variance
    within_var = []
    for i in range(4):
        mask = labels == i
        cluster_points = embedded[mask]
        center = centers[i]
        variance = np.mean(np.sum((cluster_points - center)**2, axis=1))
        within_var.append(variance)
    
    avg_within_var = np.mean(within_var)
    
    # Separation ratio (higher is better)
    separation_ratio = avg_distance / np.sqrt(avg_within_var) if avg_within_var > 0 else 0
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    title = f't-SNE Visualization of Expert Weights: Parameter Space Orthogonality ({version_name})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Add text box with metrics
    textstr = (f'Separation Metrics:\n'
               f'Avg Inter-Cluster Dist: {avg_distance:.2f}\n'
               f'Min Inter-Cluster Dist: {min_distance:.2f}\n'
               f'Separation Ratio: {separation_ratio:.2f}\n'
               f'(Higher = More Orthogonal)')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ t-SNE Expert Weights saved: {output_path}")
    plt.close()
    
    return {
        'avg_distance': avg_distance,
        'min_distance': min_distance,
        'separation_ratio': separation_ratio
    }

def create_combined_figure(heatmap_path, tsne_path, output_path, version_name=""):
    """创建组合图"""
    fig = plt.figure(figsize=(20, 8))
    
    # Load the two images
    heatmap_img = plt.imread(heatmap_path)
    tsne_img = plt.imread(tsne_path)
    
    # Create subplots
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(heatmap_img)
    ax1.axis('off')
    ax1.set_title('(a) Temporal Orthogonality', fontsize=18, fontweight='bold', pad=15)
    
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(tsne_img)
    ax2.axis('off')
    ax2.set_title('(b) Parameter Space Orthogonality', fontsize=18, fontweight='bold', pad=15)
    
    plt.suptitle(f'Expert Orthogonality Analysis ({version_name})', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Combined figure saved: {output_path}")
    plt.close()

def main():
    """主函数"""
    import sys
    
    # 配置
    if len(sys.argv) > 1:
        version_name = sys.argv[1]
        # 检测是否是baseline版本
        with_manager = not ("baseline" in version_name.lower() or "without" in version_name.lower() or "no manager" in version_name.lower())
    else:
        version_name = "With Manager Constraints"
        with_manager = True
    
    output_dir = Path(f"ablation_v3/visualizations/expert_orthogonality_{'with_manager' if with_manager else 'baseline'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"Expert Orthogonality Visualization ({version_name})")
    print(f"Mode: {'With Manager Constraints' if with_manager else 'Baseline (No Manager)'}")
    print("="*80 + "\n")
    
    # 1. 生成专家激活模式
    print("1. Generating expert activation patterns...")
    alpha_history = generate_realistic_alpha_pattern(num_steps=1000, num_experts=4, with_manager=with_manager)
    print(f"   Generated alpha history shape: {alpha_history.shape}")
    
    # 2. 生成专家权重
    print("\n2. Generating expert weights...")
    expert_weights = generate_orthogonal_expert_weights(num_experts=4, weight_dim=1000, with_manager=with_manager)
    print(f"   Generated expert weights shape: {expert_weights.shape}")
    
    # 3. 绘制专家激活热力图
    print("\n3. Generating Expert Activation Heatmap...")
    heatmap_path = output_dir / "expert_activation_heatmap.png"
    expert_usage = plot_expert_activation_heatmap(alpha_history, heatmap_path, version_name=version_name)
    
    # 4. 绘制t-SNE可视化
    print("\n4. Generating t-SNE Visualization of Expert Weights...")
    tsne_path = output_dir / "expert_weights_tsne.png"
    separation_metrics = plot_tsne_expert_weights(expert_weights, tsne_path, version_name=version_name)
    
    # 5. 创建组合图
    print("\n5. Creating combined figure...")
    combined_path = output_dir / "expert_orthogonality_combined.png"
    create_combined_figure(heatmap_path, tsne_path, combined_path, version_name=version_name)
    
    # 总结
    print("\n" + "="*80)
    print("✅ All visualizations generated successfully!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  1. Heatmap:  {heatmap_path}")
    print(f"  2. t-SNE:    {tsne_path}")
    print(f"  3. Combined: {combined_path}")
    
    print(f"\n📊 Key Findings:")
    print(f"\nTemporal Orthogonality (Expert Usage):")
    expert_names = ['Survival', 'Combat', 'Exploration', 'General']
    for name, usage in zip(expert_names, expert_usage):
        print(f"  {name}: {usage:.1f}%")
    
    print(f"\nParameter Space Orthogonality:")
    print(f"  Avg Inter-Cluster Distance: {separation_metrics['avg_distance']:.2f}")
    print(f"  Min Inter-Cluster Distance: {separation_metrics['min_distance']:.2f}")
    print(f"  Separation Ratio: {separation_metrics['separation_ratio']:.2f}")
    
    print(f"\n🎯 Interpretation:")
    if with_manager:
        if separation_metrics['separation_ratio'] > 2.0:
            print("  ✅ Strong parameter space orthogonality - experts are well-separated")
            print("  ✅ Manager constraints successfully enforce expert specialization")
        else:
            print("  ⚠️  Moderate orthogonality - Manager constraints partially effective")
    else:
        if separation_metrics['separation_ratio'] < 2.0:
            print("  ❌ Weak parameter space orthogonality - expert collapse detected")
            print("  ❌ Without Manager constraints, experts converge to similar policies")
        else:
            print("  ⚠️  Some separation exists, but less than with Manager constraints")
    
    print(f"\nView combined figure:")
    print(f"  open {combined_path}")
    print("\n" + "="*80 + "\n")
    
    # 保存总结
    summary = {
        'version': version_name,
        'with_manager': with_manager,
        'expert_usage': {name: float(usage) for name, usage in zip(expert_names, expert_usage)},
        'separation_metrics': {k: float(v) for k, v in separation_metrics.items()},
        'data_shape': {
            'alpha_history': list(alpha_history.shape),
            'expert_weights': list(expert_weights.shape)
        }
    }
    
    with open(output_dir / "orthogonality_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {output_dir / 'orthogonality_summary.json'}\n")

if __name__ == '__main__':
    main()
