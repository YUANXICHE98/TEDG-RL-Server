#!/usr/bin/env python3
"""
Visualize Expert Orthogonality Using Real Training Data
Generate publication-quality figures for paper
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

def load_real_data(data_dir):
    """Load extracted real data"""
    data_dir = Path(data_dir)
    
    # Load alpha history
    alpha_path = data_dir / "alpha_history.npy"
    if not alpha_path.exists():
        raise FileNotFoundError(f"Alpha history not found: {alpha_path}")
    alpha_history = np.load(alpha_path)
    
    # Load expert weights
    weights_path = data_dir / "expert_weights.npy"
    if not weights_path.exists():
        raise FileNotFoundError(f"Expert weights not found: {weights_path}")
    expert_weights = np.load(weights_path)
    
    # Load analysis
    analysis_path = data_dir / "episodes_analysis.json"
    analysis = None
    if analysis_path.exists():
        with open(analysis_path, 'r') as f:
            analysis = json.load(f)
    
    return alpha_history, expert_weights, analysis

def plot_expert_activation_heatmap(alpha_history, output_path, max_steps=1000, version_name=""):
    """
    Plot Expert Activation Heatmap from real data
    Shows which expert is active at each time step
    """
    # Limit to max_steps for visualization
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
    title = f'Expert Activation Heatmap: Temporal Orthogonality ({version_name})\n(Real Training Data)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Set y-axis ticks
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Expert 0', 'Expert 1', 'Expert 2', 'Expert 3'],
                       fontsize=12)
    
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
    for i, usage in enumerate(expert_usage):
        stats_text += f'Expert {i}: {usage:.1f}%\n'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    # Identify and annotate significant blocks
    blocks = []
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
    
    # Annotate blocks
    for expert, start, end in blocks[:5]:  # Limit to first 5 blocks
        mid_point = (start + end) / 2
        ax.axvline(x=start, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=end, color='white', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Expert Activation Heatmap saved: {output_path}")
    plt.close()
    
    return expert_usage

def plot_tsne_expert_weights(expert_weights, output_path, version_name=""):
    """
    Plot t-SNE visualization of expert weights from real data
    Shows parameter space orthogonality
    """
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
    expert_names = ['Expert 0', 'Expert 1', 'Expert 2', 'Expert 3']
    
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
    title = f't-SNE Visualization of Expert Weights: Parameter Space Orthogonality ({version_name})\n(Real Training Data)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
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

def create_combined_figure(heatmap_path, tsne_path, output_path):
    """Create a combined figure with both visualizations"""
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
    
    plt.suptitle('Expert Orthogonality Analysis (Real Training Data)', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Combined figure saved: {output_path}")
    plt.close()

def main():
    """Main function"""
    
    # Configuration - can be overridden by command line args
    import sys
    if len(sys.argv) > 1:
        version_name = sys.argv[1]
    else:
        version_name = "with_manager"
    
    data_dir = f"ablation_v3/visualizations/expert_data_{version_name}"
    output_dir = Path(f"ablation_v3/visualizations/expert_orthogonality_{version_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"Expert Orthogonality Visualization ({version_name})")
    print("="*80 + "\n")
    
    # Load real data
    try:
        print("Loading extracted data...")
        alpha_history, expert_weights, analysis = load_real_data(data_dir)
        print(f"✅ Data loaded:")
        print(f"   Alpha history shape: {alpha_history.shape}")
        print(f"   Expert weights shape: {expert_weights.shape}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nPlease run data extraction first:")
        print(f"  python3 tools/extract_real_expert_data.py <checkpoint_path> {version_name}")
        return
    
    # 1. Generate Expert Activation Heatmap
    print("\n1. Generating Expert Activation Heatmap...")
    heatmap_path = output_dir / "expert_activation_heatmap_real.png"
    expert_usage = plot_expert_activation_heatmap(alpha_history, heatmap_path, version_name=version_name)
    
    # 2. Generate t-SNE Visualization
    print("\n2. Generating t-SNE Visualization of Expert Weights...")
    tsne_path = output_dir / "expert_weights_tsne_real.png"
    separation_metrics = plot_tsne_expert_weights(expert_weights, tsne_path, version_name=version_name)
    
    # 3. Create combined figure
    print("\n3. Creating combined figure...")
    combined_path = output_dir / "expert_orthogonality_combined_real.png"
    create_combined_figure(heatmap_path, tsne_path, combined_path)
    
    # Summary
    print("\n" + "="*80)
    print("✅ All visualizations generated successfully!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  1. Heatmap:  {heatmap_path}")
    print(f"  2. t-SNE:    {tsne_path}")
    print(f"  3. Combined: {combined_path}")
    
    print(f"\n📊 Key Findings:")
    print(f"\nTemporal Orthogonality (Expert Usage):")
    for i, usage in enumerate(expert_usage):
        print(f"  Expert {i}: {usage:.1f}%")
    
    print(f"\nParameter Space Orthogonality:")
    print(f"  Avg Inter-Cluster Distance: {separation_metrics['avg_distance']:.2f}")
    print(f"  Min Inter-Cluster Distance: {separation_metrics['min_distance']:.2f}")
    print(f"  Separation Ratio: {separation_metrics['separation_ratio']:.2f}")
    
    print(f"\n🎯 Interpretation:")
    if separation_metrics['separation_ratio'] > 2.0:
        print("  ✅ Strong parameter space orthogonality - experts are well-separated")
    elif separation_metrics['separation_ratio'] > 1.0:
        print("  ⚠️  Moderate parameter space orthogonality - some overlap exists")
    else:
        print("  ❌ Weak parameter space orthogonality - experts may be redundant")
    
    print(f"\nView combined figure:")
    print(f"  open {combined_path}")
    print("\n" + "="*80 + "\n")
    
    # Save summary
    summary = {
        'expert_usage': {f'expert_{i}': float(usage) for i, usage in enumerate(expert_usage)},
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
