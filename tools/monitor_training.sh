#!/bin/bash
# Monitor training progress

LOG_DIR="ablation_v3/results/warmup_1000/logs"
LOG_FILE="$LOG_DIR/training_log.json"

echo "=========================================="
echo "V3 Training Monitor"
echo "=========================================="
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Training log not found: $LOG_FILE"
    echo "Training may not have started yet."
    exit 1
fi

# Parse JSON and show progress
python3 << 'PYEOF'
import json
import sys
from pathlib import Path

log_file = Path("ablation_v3/results/warmup_1000/logs/training_log.json")

if not log_file.exists():
    print("❌ Log file not found")
    sys.exit(1)

with open(log_file, 'r') as f:
    data = json.load(f)

total_eps = len(data.get('episode_rewards', []))
rewards = data.get('episode_rewards', [])
scores = data.get('episode_scores', [])
entropies = data.get('alpha_entropies', [])

print(f"📊 Training Progress: {total_eps}/1000 episodes")
print(f"   Progress: {'█' * (total_eps // 20)}{'░' * (50 - total_eps // 20)} {total_eps/10:.1f}%")
print()

if total_eps > 0:
    import numpy as np
    
    # Recent performance (last 10 episodes)
    recent_rewards = rewards[-10:] if len(rewards) >= 10 else rewards
    recent_scores = scores[-10:] if len(scores) >= 10 else scores
    recent_entropies = entropies[-10:] if len(entropies) >= 10 else entropies
    
    print("📈 Recent Performance (last 10 episodes):")
    print(f"   Avg Reward: {np.mean(recent_rewards):.2f}")
    print(f"   Avg Score: {np.mean(recent_scores):.2f}")
    print(f"   Max Score: {np.max(recent_scores):.0f}")
    if recent_entropies:
        print(f"   Avg α Entropy: {np.mean(recent_entropies):.3f}")
    print()
    
    # Overall stats
    print("📊 Overall Statistics:")
    print(f"   Avg Reward: {np.mean(rewards):.2f}")
    print(f"   Avg Score: {np.mean(scores):.2f}")
    print(f"   Max Score: {np.max(scores):.0f}")
    if entropies:
        print(f"   Avg α Entropy: {np.mean(entropies):.3f}")
    print()
    
    # Phase info
    if total_eps < 1000:
        print(f"🔄 Current Phase: Warmup (Softmax routing)")
        print(f"   Episodes until Transition: {1000 - total_eps}")
    else:
        print(f"🔄 Current Phase: Transition (Sparsemax routing)")

PYEOF

echo ""
echo "=========================================="
echo "Use: watch -n 10 bash tools/monitor_training.sh"
echo "=========================================="
