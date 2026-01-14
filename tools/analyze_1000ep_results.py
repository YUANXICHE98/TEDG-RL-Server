#!/usr/bin/env python3
"""
Analyze 1000 episode training results
"""
import json
import numpy as np

# 读取训练日志
with open('ablation_v3/results/warmup_1000/logs/training_log.json', 'r') as f:
    data = json.load(f)

# 提取数据
rewards = data['episode_rewards']
scores = data['episode_scores']
monitor = data['monitor_metrics']

# 检查monitor格式
print(f"Monitor type: {type(monitor)}")
if isinstance(monitor, dict):
    print(f"Monitor keys: {list(monitor.keys())}")
    # 如果是字典，提取alpha_entropy
    if 'alpha_entropy' in monitor:
        alpha_entropies = monitor['alpha_entropy']
    else:
        print("Available monitor keys:", list(monitor.keys())[:5])
        alpha_entropies = None
else:
    print(f"Monitor length: {len(monitor)}")
    print(f"First monitor item: {monitor[0] if len(monitor) > 0 else 'empty'}")
    alpha_entropies = None

# 分段分析
def analyze_segment(start, end, name):
    seg_rewards = rewards[start:end]
    seg_scores = scores[start:end]
    
    print(f'\n{name} (Episodes {start}-{end}):')
    print(f'  Reward: {np.mean(seg_rewards):.2f} ± {np.std(seg_rewards):.2f}')
    print(f'  Score: {np.mean(seg_scores):.2f} ± {np.std(seg_scores):.2f}')
    print(f'  Max Score: {max(seg_scores)}')
    print(f'  Min Score: {min(seg_scores)}')

# 分段分析
print("="*60)
print("分段分析")
print("="*60)
analyze_segment(0, 100, 'First 100')
analyze_segment(100, 300, 'Episodes 100-300')
analyze_segment(300, 500, 'Episodes 300-500')
analyze_segment(500, 700, 'Episodes 500-700')
analyze_segment(700, 900, 'Episodes 700-900')
analyze_segment(900, 1000, 'Last 100')

# 整体统计
print(f'\n{"="*60}')
print(f'整体统计 (1000 episodes)')
print(f'{"="*60}')
print(f'  平均Reward: {np.mean(rewards):.2f}')
print(f'  平均Score: {np.mean(scores):.2f}')
print(f'  最高Score: {max(scores)}')
print(f'  最低Score: {min(scores)}')
print(f'  Reward标准差: {np.std(rewards):.2f}')
print(f'  Score标准差: {np.std(scores):.2f}')

# 检查趋势
first_half_score = np.mean(scores[:500])
second_half_score = np.mean(scores[500:])
first_half_reward = np.mean(rewards[:500])
second_half_reward = np.mean(rewards[500:])

print(f'\n{"="*60}')
print(f'趋势分析')
print(f'{"="*60}')
print(f'  前500轮平均Score: {first_half_score:.2f}')
print(f'  后500轮平均Score: {second_half_score:.2f}')
print(f'  Score改善: {second_half_score - first_half_score:.2f} ({(second_half_score/first_half_score - 1)*100:.1f}%)')
print(f'\n  前500轮平均Reward: {first_half_reward:.2f}')
print(f'  后500轮平均Reward: {second_half_reward:.2f}')
print(f'  Reward改善: {second_half_reward - first_half_reward:.2f} ({(second_half_reward/first_half_reward - 1)*100:.1f}%)')

# 找出最佳episodes
best_indices = np.argsort(scores)[-10:][::-1]
print(f'\n{"="*60}')
print(f'Top 10 Episodes')
print(f'{"="*60}')
for i, idx in enumerate(best_indices, 1):
    print(f'  {i}. Episode {idx}: Score={scores[idx]}, Reward={rewards[idx]:.2f}')

# 统计score分布
score_bins = [0, 10, 20, 50, 100, 300]
print(f'\n{"="*60}')
print(f'Score分布')
print(f'{"="*60}')
for i in range(len(score_bins)-1):
    count = sum(1 for s in scores if score_bins[i] <= s < score_bins[i+1])
    print(f'  {score_bins[i]}-{score_bins[i+1]}: {count} episodes ({count/10:.1f}%)')
count_above = sum(1 for s in scores if s >= score_bins[-1])
print(f'  >={score_bins[-1]}: {count_above} episodes ({count_above/10:.1f}%)')
