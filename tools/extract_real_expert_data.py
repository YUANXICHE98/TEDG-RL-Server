#!/usr/bin/env python3
"""
Extract Real Expert Activation Data from Trained Model
Run inference and collect actual alpha values and expert weights
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import numpy as np
import json
from pathlib import Path
import nle
from nle import nethack

# Import your model architecture
from src.core.networks_v3_gat_moe import GATGuidedMoEPolicy

def load_trained_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model with same architecture as training
    model = GATGuidedMoEPolicy(
        hypergraph_path="data/hypergraph/hypergraph_gat_structure.json",
        state_dim=115,
        hidden_dim=256,
        action_dim=23,
        num_experts=4,
        use_sparsemax=True
    ).to(device)
    
    # Load state dict - try different keys
    if 'policy_net' in checkpoint:
        model.load_state_dict(checkpoint['policy_net'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'policy_net_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['policy_net_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Assume the checkpoint is the state dict itself
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print("✅ Model loaded successfully")
    return model

def run_inference_episode(model, env, max_steps=2000, device='cpu'):
    """
    Run one inference episode and collect alpha values at each step
    """
    obs = env.reset()
    done = False
    step = 0
    
    alpha_history = []
    actions_history = []
    rewards_history = []
    total_reward = 0
    
    with torch.no_grad():
        while not done and step < max_steps:
            # Convert observation to tensor
            # For blstats, we need to extract the relevant features
            blstats = obs['blstats']
            
            # Create state tensor (115 dimensions as in training)
            # This includes blstats and other features
            state_features = []
            
            # Add blstats features (first 26 dimensions)
            state_features.extend(blstats[:26])
            
            # Add additional features to reach 115 dimensions
            # (In actual training, this would include more game state info)
            # For now, pad with zeros
            while len(state_features) < 115:
                state_features.append(0.0)
            
            state_tensor = torch.FloatTensor(state_features).to(device)
            
            # Get action and alpha values
            # Note: We don't have atoms during inference, so GAT will use zero vectors
            logits, alpha, value, aux_info = model.forward(state_tensor, atoms=None, active_mask=None)
            
            # Sample action
            action_probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            
            # Store alpha values
            alpha_np = alpha.cpu().numpy()  # Shape: (num_experts,)
            alpha_history.append(alpha_np)
            actions_history.append(action)
            
            # Take step
            obs, reward, done, info = env.step(action)
            rewards_history.append(reward)
            total_reward += reward
            step += 1
    
    return {
        'alpha_history': np.array(alpha_history),
        'actions': actions_history,
        'rewards': rewards_history,
        'steps': step,
        'score': total_reward  # Use accumulated reward as score
    }

def extract_expert_weights(model):
    """Extract expert network weights from model"""
    expert_weights = []
    
    # Get expert networks from the model (not router)
    for i in range(4):
        expert = model.experts[i]
        
        # Collect all weights from the expert network
        weights = []
        for param in expert.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        
        # Concatenate all weights
        expert_weight_vector = np.concatenate(weights)
        expert_weights.append(expert_weight_vector)
    
    return np.array(expert_weights)

def analyze_alpha_patterns(alpha_history, actions, window=50):
    """Analyze patterns in alpha values"""
    num_steps, num_experts = alpha_history.shape
    
    # Find dominant expert at each step
    dominant_expert = np.argmax(alpha_history, axis=1)
    
    # Calculate expert usage statistics
    expert_usage = {}
    for i in range(num_experts):
        usage_pct = (dominant_expert == i).sum() / num_steps * 100
        expert_usage[f'expert_{i}'] = usage_pct
    
    # Find continuous blocks where same expert is dominant
    blocks = []
    if num_steps > 0:
        current_expert = dominant_expert[0]
        start_idx = 0
        
        for i in range(1, num_steps):
            if dominant_expert[i] != current_expert:
                blocks.append({
                    'expert': int(current_expert),
                    'start': start_idx,
                    'end': i-1,
                    'duration': i - start_idx
                })
                current_expert = dominant_expert[i]
                start_idx = i
        
        # Add final block
        blocks.append({
            'expert': int(current_expert),
            'start': start_idx,
            'end': num_steps-1,
            'duration': num_steps - start_idx
        })
    
    # Filter significant blocks (duration > 10)
    significant_blocks = [b for b in blocks if b['duration'] > 10]
    
    return {
        'expert_usage': expert_usage,
        'blocks': significant_blocks,
        'num_switches': len(blocks) - 1
    }

def main():
    """Main function"""
    
    # Configuration - can be overridden by command line args
    import sys
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        version_name = sys.argv[2] if len(sys.argv) > 2 else "default"
    else:
        checkpoint_path = "ablation_v3/results/resume_500_from_100/checkpoints/model_00500.pth"
        version_name = "with_manager"
    
    output_dir = Path(f"ablation_v3/visualizations/expert_data_{version_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_episodes = 5  # Run multiple episodes to get diverse data
    max_steps = 2000
    
    print("\n" + "="*80)
    print(f"Extracting Real Expert Data from Trained Model ({version_name})")
    print("="*80 + "\n")
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = Path(checkpoint_path).parent
        if checkpoint_dir.exists():
            for ckpt in sorted(checkpoint_dir.glob("*.pth")):
                print(f"  - {ckpt}")
        return
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    try:
        model = load_trained_model(checkpoint_path, device=device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nThis might be due to model architecture mismatch.")
        print("Please check that the checkpoint matches the current model architecture.")
        return
    
    # Extract expert weights
    print("\n1. Extracting Expert Weights...")
    expert_weights = extract_expert_weights(model)
    print(f"   Expert weights shape: {expert_weights.shape}")
    print(f"   Each expert has {expert_weights.shape[1]} parameters")
    
    # Save expert weights
    np.save(output_dir / "expert_weights.npy", expert_weights)
    print(f"   ✅ Saved to: {output_dir / 'expert_weights.npy'}")
    
    # Create environment
    print("\n2. Running Inference Episodes...")
    env = nle.env.NLE(
        observation_keys=("glyphs", "blstats"),
        actions=nethack.ACTIONS,
    )
    
    all_episodes_data = []
    all_alpha_history = []
    
    for ep in range(num_episodes):
        print(f"\n   Episode {ep+1}/{num_episodes}:")
        
        try:
            episode_data = run_inference_episode(model, env, max_steps=max_steps, device=device)
            
            print(f"     Steps: {episode_data['steps']}")
            print(f"     Score: {episode_data['score']}")
            print(f"     Alpha shape: {episode_data['alpha_history'].shape}")
            
            # Analyze patterns
            analysis = analyze_alpha_patterns(
                episode_data['alpha_history'],
                episode_data['actions']
            )
            
            print(f"     Expert usage:")
            for expert, pct in analysis['expert_usage'].items():
                print(f"       {expert}: {pct:.1f}%")
            print(f"     Expert switches: {analysis['num_switches']}")
            print(f"     Significant blocks: {len(analysis['significant_blocks'])}") 
            
            episode_data['analysis'] = analysis
            all_episodes_data.append(episode_data)
            all_alpha_history.append(episode_data['alpha_history'])
            
        except Exception as e:
            print(f"     ❌ Error in episode {ep+1}: {e}")
            continue
    
    env.close()
    
    if not all_episodes_data:
        print("\n❌ No episodes completed successfully")
        return
    
    # Concatenate all alpha histories
    combined_alpha = np.vstack(all_alpha_history)
    print(f"\n3. Combined Data:")
    print(f"   Total steps: {combined_alpha.shape[0]}")
    print(f"   Alpha shape: {combined_alpha.shape}")
    
    # Save alpha history
    np.save(output_dir / "alpha_history.npy", combined_alpha)
    print(f"   ✅ Saved alpha history: {output_dir / 'alpha_history.npy'}")
    
    # Save detailed episode data
    # Convert numpy arrays to lists for JSON serialization
    episodes_json = []
    for ep_data in all_episodes_data:
        ep_json = {
            'steps': ep_data['steps'],
            'score': ep_data['score'],
            'analysis': ep_data['analysis'],
            'alpha_stats': {
                'mean': ep_data['alpha_history'].mean(axis=0).tolist(),
                'std': ep_data['alpha_history'].std(axis=0).tolist(),
                'min': ep_data['alpha_history'].min(axis=0).tolist(),
                'max': ep_data['alpha_history'].max(axis=0).tolist(),
            }
        }
        episodes_json.append(ep_json)
    
    with open(output_dir / "episodes_analysis.json", 'w') as f:
        json.dump(episodes_json, f, indent=2)
    print(f"   ✅ Saved analysis: {output_dir / 'episodes_analysis.json'}")
    
    # Summary
    print("\n" + "="*80)
    print(f"✅ Data Extraction Complete ({version_name})!")
    print("="*80)
    print(f"\nExtracted data:")
    print(f"  1. Expert weights: {output_dir / 'expert_weights.npy'}")
    print(f"  2. Alpha history: {output_dir / 'alpha_history.npy'}")
    print(f"  3. Episode analysis: {output_dir / 'episodes_analysis.json'}")
    print(f"\nNext step: Run visualization script")
    print(f"  python3 tools/visualize_expert_orthogonality_real.py {version_name}")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
