#!/usr/bin/env python3
"""
Visualize 1000 Episodes Training Results
All labels and titles in English
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Set English font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_training_log(log_path):
    """Load training log JSON"""
    with open(log_path, 'r') as f:
        data = json.load(f)
    return data


def plot_training_curves(data, output_dir):
    """Plot comprehensive training curves"""
    
    episodes = list(range(len(data['episode_rewards'])))
    rewards = data['episode_rewards']
    scores = data['episode_scores']
    alpha_entropies = data.get('alpha_entropies', [])
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('V3 Training Progress (1000 Episodes)', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw Reward')
    # Moving average
    window = 50
    if len(rewards) >= window:
        ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), ma_rewards, 
                color='red', linewidth=2, label=f'{window}-Episode MA')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Scores
    ax2 = axes[0, 1]
    ax2.plot(episodes, scores, alpha=0.3, color='green', label='Raw Score')
    if len(scores) >= window:
        ma_scores = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(scores)), ma_scores,
                color='darkgreen', linewidth=2, label=f'{window}-Episode MA')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Episode Scores Over Time', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Alpha Entropy
    ax3 = axes[1, 0]
    if alpha_entropies:
        ax3.plot(episodes[:len(alpha_entropies)], alpha_entropies, 
                color='purple', linewidth=1.5, label='Alpha Entropy')
        ax3.axhline(y=1.386, color='red', linestyle='--', 
                   label='Theoretical Max (log(4))', alpha=0.7)
        ax3.axhline(y=0.5, color='orange', linestyle='--',
                   label='Target (Fine-tune)', alpha=0.7)
        # Mark phase transitions
        ax3.axvline(x=1000, color='gray', linestyle=':', alpha=0.5, label='Transition Phase')
        ax3.set_xlabel('Episode', fontsize=12)
        ax3.set_ylabel('Entropy', fontsize=12)
        ax3.set_title('Expert Routing Entropy (α)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Alpha Entropy Data', 
                ha='center', va='center', fontsize=14)
        ax3.set_title('Expert Routing Entropy', fontsize=14, fontweight='bold')
    
    # 4. Performance Distribution
    ax4 = axes[1, 1]
    # Split into phases
    phase1_scores = scores[:min(1000, len(scores))]  # Warmup
    phase2_scores = scores[1000:] if len(scores) > 1000 else []  # Transition+
    
    data_to_plot = []
    labels = []
    if phase1_scores:
        data_to_plot.append(phase1_scores)
        labels.append(f'Warmup\n(0-1000)')
    if phase2_scores:
        data_to_plot.append(phase2_scores)
        labels.append(f'Transition+\n(1000+)')
    
    if data_to_plot:
        bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Score Distribution by Training Phase', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    save_path = output_dir / 'training_curves_1000ep.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_expert_analysis(data, output_dir):
    """Plot expert usage analysis"""
    
    # This requires per-episode expert data
    # For now, create placeholder
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Expert System Analysis', fontsize=16, fontweight='bold')
    
    ax1 = axes[0]
    ax1.text(0.5, 0.5, 'Expert Usage Over Time\n(Requires detailed logs)', 
            ha='center', va='center', fontsize=12)
    ax1.set_title('Expert Selection Frequency', fontsize=14, fontweight='bold')
    
    ax2 = axes[1]
    ax2.text(0.5, 0.5, 'Expert Performance Comparison\n(Requires detailed logs)',
            ha='center', va='center', fontsize=12)
    ax2.set_title('Expert-wise Score Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_dir / 'expert_analysis_1000ep.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_phase_comparison(data, output_dir):
    """Compare different training phases"""
    
    rewards = data['episode_rewards']
    scores = data['episode_scores']
    alpha_entropies = data.get('alpha_entropies', [])
    
    # Define phases
    phases = {
        'Warmup (0-1000)': (0, min(1000, len(rewards))),
    }
    
    if len(rewards) > 1000:
        phases['Transition (1000+)'] = (1000, len(rewards))
    
    # Create comparison table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    table_data = [['Phase', 'Episodes', 'Avg Reward', 'Avg Score', 'Max Score', 'Avg α Entropy']]
    
    for phase_name, (start, end) in phases.items():
        phase_rewards = rewards[start:end]
        phase_scores = scores[start:end]
        phase_entropies = alpha_entropies[start:end] if alpha_entropies else []
        
        avg_reward = np.mean(phase_rewards) if phase_rewards else 0
        avg_score = np.mean(phase_scores) if phase_scores else 0
        max_score = np.max(phase_scores) if phase_scores else 0
        avg_entropy = np.mean(phase_entropies) if phase_entropies else 0
        
        table_data.append([
            phase_name,
            f'{start}-{end}',
            f'{avg_reward:.2f}',
            f'{avg_score:.2f}',
            f'{max_score:.0f}',
            f'{avg_entropy:.3f}' if avg_entropy > 0 else 'N/A'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Training Phase Comparison', fontsize=16, fontweight='bold', pad=20)
    
    save_path = output_dir / 'phase_comparison_1000ep.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_summary_report(data, output_path):
    """Generate text summary report"""
    
    rewards = data['episode_rewards']
    scores = data['episode_scores']
    alpha_entropies = data.get('alpha_entropies', [])
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("V3 Training Summary (1000 Episodes)\n")
        f.write("="*60 + "\n\n")
        
        f.write("Overall Statistics:\n")
        f.write(f"  Total Episodes: {len(rewards)}\n")
        f.write(f"  Average Reward: {np.mean(rewards):.2f}\n")
        f.write(f"  Average Score: {np.mean(scores):.2f}\n")
        f.write(f"  Max Score: {np.max(scores):.0f}\n")
        f.write(f"  Min Score: {np.min(scores):.0f}\n")
        if alpha_entropies:
            f.write(f"  Average α Entropy: {np.mean(alpha_entropies):.3f}\n")
        f.write("\n")
        
        # Phase breakdown
        f.write("Phase Breakdown:\n")
        
        # Warmup
        warmup_end = min(1000, len(rewards))
        warmup_rewards = rewards[:warmup_end]
        warmup_scores = scores[:warmup_end]
        f.write(f"\n  Warmup Phase (0-{warmup_end}):\n")
        f.write(f"    Avg Reward: {np.mean(warmup_rewards):.2f}\n")
        f.write(f"    Avg Score: {np.mean(warmup_scores):.2f}\n")
        f.write(f"    Max Score: {np.max(warmup_scores):.0f}\n")
        
        if len(rewards) > 1000:
            trans_rewards = rewards[1000:]
            trans_scores = scores[1000:]
            f.write(f"\n  Transition+ Phase (1000-{len(rewards)}):\n")
            f.write(f"    Avg Reward: {np.mean(trans_rewards):.2f}\n")
            f.write(f"    Avg Score: {np.mean(trans_scores):.2f}\n")
            f.write(f"    Max Score: {np.max(trans_scores):.0f}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize 1000 Episodes Training')
    parser.add_argument('--log-dir', type=str, 
                       default='ablation_v3/results/warmup_1000/logs',
                       help='Training log directory')
    parser.add_argument('--output-dir', type=str,
                       default='ablation_v3/visualizations/1000ep',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Setup paths
    log_path = Path(args.log_dir) / 'training_log.json'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("V3 Training Visualization (1000 Episodes)")
    print("="*60)
    print()
    
    # Load data
    print(f"Loading training log from: {log_path}")
    data = load_training_log(log_path)
    print(f"✓ Loaded {len(data['episode_rewards'])} episodes")
    print()
    
    # Generate plots
    print("Generating visualizations...")
    plot_training_curves(data, output_dir)
    plot_expert_analysis(data, output_dir)
    plot_phase_comparison(data, output_dir)
    
    # Generate summary
    summary_path = output_dir / 'training_summary.txt'
    generate_summary_report(data, summary_path)
    
    print()
    print("="*60)
    print("✓ Visualization Complete!")
    print(f"  Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
