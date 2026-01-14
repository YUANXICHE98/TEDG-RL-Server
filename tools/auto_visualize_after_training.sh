#!/bin/bash
# Auto-visualize after training completes

LOG_FILE="ablation_v3/results/warmup_1000/logs/training_log.json"
TARGET_EPISODES=1000

echo "=========================================="
echo "Waiting for training to complete..."
echo "Target: $TARGET_EPISODES episodes"
echo "=========================================="
echo ""

# Wait for log file to exist
while [ ! -f "$LOG_FILE" ]; do
    echo "⏳ Waiting for training to start..."
    sleep 10
done

echo "✓ Training started!"
echo ""

# Monitor progress
while true; do
    if [ -f "$LOG_FILE" ]; then
        CURRENT=$(python3 -c "import json; data=json.load(open('$LOG_FILE')); print(len(data.get('episode_rewards', [])))")
        
        if [ "$CURRENT" -ge "$TARGET_EPISODES" ]; then
            echo ""
            echo "=========================================="
            echo "✓ Training completed! ($CURRENT episodes)"
            echo "=========================================="
            echo ""
            break
        fi
        
        PROGRESS=$((CURRENT * 100 / TARGET_EPISODES))
        echo -ne "\r⏳ Progress: $CURRENT/$TARGET_EPISODES episodes ($PROGRESS%)   "
    fi
    
    sleep 30
done

# Run visualization
echo "Starting visualization..."
echo ""

conda activate tedg-rl-demo
python tools/visualize_1000ep_training.py

echo ""
echo "=========================================="
echo "✓ Visualization complete!"
echo "=========================================="
echo ""
echo "View results:"
echo "  open ablation_v3/visualizations/1000ep/training_curves_1000ep.png"
echo ""
