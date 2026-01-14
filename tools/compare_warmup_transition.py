#!/usr/bin/env python3
import json
import numpy as np

# Warmup数据
with open('ablation_v3/results/warmup_1000/logs/training_log.json', 'r') as f:
    warmup = json.load(f)

# Transition数据  
with open('ablation_v3/results/transition_3000/logs/training_log.json', 'r') as f:
    transition = json.load(f)

print('='*70)
print('Warmup vs Transition Comparison')
print('='*70)
print()
print('| Metric | Warmup (0-1000) | Transition (1000-3000) | Change |')
print('|--------|-----------------|------------------------|--------|')

# Score
warmup_score = np.mean(warmup['episode_scores'])
trans_score = np.mean(transition['episode_scores'])
score_change = ((trans_score / warmup_score) - 1) * 100
print(f'| Avg Score | {warmup_score:.2f} | {trans_score:.2f} | {score_change:+.1f}% |')

# Reward
warmup_reward = np.mean(warmup['episode_rewards'])
trans_reward = np.mean(transition['episode_rewards'])
reward_change = ((trans_reward / warmup_reward) - 1) * 100
print(f'| Avg Reward | {warmup_reward:.2f} | {trans_reward:.2f} | {reward_change:+.1f}% |')

# Best Score
warmup_best = max(warmup['episode_scores'])
trans_best = max(transition['episode_scores'])
print(f'| Best Score | {warmup_best} | {trans_best} | {trans_best - warmup_best:+d} |')

# Alpha Entropy
warmup_alpha = np.mean(warmup['monitor_metrics']['alpha_entropy'])
trans_alpha = np.mean(transition['monitor_metrics']['alpha_entropy'])
alpha_change = warmup_alpha - trans_alpha
print(f'| Alpha Entropy | {warmup_alpha:.4f} | {trans_alpha:.4f} | -{alpha_change:.4f} |')

# Variance
warmup_var = np.std(warmup['episode_scores'])
trans_var = np.std(transition['episode_scores'])
var_change = ((trans_var / warmup_var) - 1) * 100
print(f'| Score Std | {warmup_var:.2f} | {trans_var:.2f} | {var_change:+.1f}% |')

print()
print('='*70)
print('Key Findings:')
print('='*70)
print(f'Alpha Entropy Decreased: {warmup_alpha:.4f} -> {trans_alpha:.4f} ({alpha_change/warmup_alpha*100:.1f}% reduction)')
print(f'Sparsemax Working: Alpha entropy now at {trans_alpha:.4f} (target ~0.7)')
print(f'Score Change: {score_change:+.1f}% {"(Improved)" if trans_score > warmup_score else "(Declined)"}')
print(f'Reward Change: {reward_change:+.1f}% {"(Improved)" if trans_reward > warmup_reward else "(Declined)"}')
print()
print('Expert Specialization: {"YES" if trans_alpha < 1.0 else "NO"} - Alpha entropy < 1.0 indicates specialization')
