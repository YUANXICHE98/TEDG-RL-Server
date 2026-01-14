# V3 Visualization Guide

## Overview

This guide explains how to visualize V3 training results with **all labels and titles in English**.

## Available Visualization Tools

### 1. Training Progress (1000 Episodes)
**Tool**: `tools/visualize_1000ep_training.py`

**Generates**:
- Training curves (rewards, scores, alpha entropy)
- Phase comparison
- Performance distribution
- Summary report

**Usage**:
```bash
conda activate tedg-rl-demo
python tools/visualize_1000ep_training.py
```

**Output**:
- `ablation_v3/visualizations/1000ep/training_curves_1000ep.png`
- `ablation_v3/visualizations/1000ep/expert_analysis_1000ep.png`
- `ablation_v3/visualizations/1000ep/phase_comparison_1000ep.png`
- `ablation_v3/visualizations/1000ep/training_summary.txt`

### 2. Episode Visualization
**Tool**: `tools/visualize_v3_episode.py`

**Generates**:
- Episode heatmaps
- Key moments screenshots
- Alpha evolution

**Usage**:
```bash
python tools/visualize_v3_episode.py \
    --checkpoint ablation_v3/results/warmup_1000/checkpoints/best_model.pth \
    --steps 100
```

### 3. Routing Dynamic Test
**Tool**: `tools/test_v3_routing_dynamic.py`

**Tests**:
- Alpha changes across different scenarios
- Expert switching behavior
- Scene sensitivity

**Usage**:
```bash
python tools/test_v3_routing_dynamic.py \
    --checkpoint ablation_v3/results/warmup_1000/checkpoints/best_model.pth
```

## Plot Descriptions

### Training Curves (4 subplots)

#### 1. Episode Rewards Over Time
- **X-axis**: Episode
- **Y-axis**: Reward
- **Lines**: 
  - Blue (transparent): Raw reward per episode
  - Red (solid): 50-episode moving average
  - Gray dashed: Zero line

#### 2. Episode Scores Over Time
- **X-axis**: Episode
- **Y-axis**: Score
- **Lines**:
  - Green (transparent): Raw score per episode
  - Dark green (solid): 50-episode moving average

#### 3. Expert Routing Entropy (α)
- **X-axis**: Episode
- **Y-axis**: Entropy
- **Lines**:
  - Purple: Alpha entropy over time
  - Red dashed: Theoretical max (log(4) = 1.386)
  - Orange dashed: Target for fine-tune (0.5)
  - Gray dotted: Phase transition markers

#### 4. Score Distribution by Training Phase
- **Box plots** showing score distribution
- **Phases**:
  - Warmup (0-1000): Light blue
  - Transition+ (1000+): Light green

### Phase Comparison Table

**Columns**:
- Phase: Training phase name
- Episodes: Episode range
- Avg Reward: Average reward in phase
- Avg Score: Average score in phase
- Max Score: Maximum score achieved
- Avg α Entropy: Average alpha entropy

**Color coding**:
- Header: Green background
- Rows: Alternating white/gray

## Interpreting Results

### Warmup Phase (0-1000)

**Expected**:
- ✅ Alpha entropy ~1.38 (close to 1.386)
- ✅ Gradual score improvement
- ✅ All experts being used (balanced)
- ✅ Softmax routing

**Good signs**:
- Entropy stays high and stable
- Scores show upward trend
- No NaN/Inf values

**Bad signs**:
- Entropy drops below 1.0 (premature sparsification)
- Scores don't improve
- Training crashes

### Transition Phase (1000-3000)

**Expected**:
- ⬇️ Alpha entropy gradually decreases
- ⬆️ Scores continue to improve
- 🎯 Expert specialization begins
- 🔄 Sparsemax routing starts

**Good signs**:
- Entropy decreases smoothly (1.38 → 0.8)
- Scores improve faster
- Expert switching becomes more purposeful

### Fine-tune Phase (3000+)

**Expected**:
- ⬇️ Alpha entropy low (~0.5-0.7)
- ⬆️ Scores stabilize at high level
- 🎯 Clear expert specialization
- 🔒 Sparsemax routing (sparse)

## Comparison with Previous Results

### vs 50 Episodes (init_fix_test)

**Metrics to compare**:
1. **Alpha entropy stability**
   - 50ep: 1.38 (stable)
   - 1000ep: Should remain ~1.38 in Warmup

2. **Score improvement**
   - 50ep: Best score 54
   - 1000ep: Should exceed 54

3. **Expert usage**
   - 50ep: Survival 70%, Exploration 18%
   - 1000ep: Should become more balanced

4. **Alpha sensitivity**
   - 50ep: 24% change across scenarios
   - 1000ep: Should maintain or improve

## Customization

### Change Output Directory
```bash
python tools/visualize_1000ep_training.py \
    --output-dir my_custom_dir
```

### Change Log Directory
```bash
python tools/visualize_1000ep_training.py \
    --log-dir path/to/logs
```

### Modify Plot Style

Edit `tools/visualize_1000ep_training.py`:
```python
# Change figure size
fig, axes = plt.subplots(2, 2, figsize=(20, 15))  # Larger

# Change colors
ax.plot(..., color='red')  # Different color

# Change font size
plt.rcParams['font.size'] = 14  # Larger font
```

## Troubleshooting

### "No module named matplotlib"
```bash
conda activate tedg-rl-demo
pip install matplotlib numpy
```

### "File not found"
```bash
# Check if training log exists
ls -lh ablation_v3/results/warmup_1000/logs/training_log.json

# If not, training may not have completed
# Check training process status
```

### "Chinese characters not displaying"
This is expected and intentional. All labels are in English.
The warnings about missing Chinese glyphs can be ignored.

### Plots look wrong
```bash
# Regenerate with verbose output
python tools/visualize_1000ep_training.py --verbose

# Check data format
python -c "import json; print(json.load(open('ablation_v3/results/warmup_1000/logs/training_log.json')))"
```

## Best Practices

1. **Wait for training to complete** before visualizing
2. **Check log file exists** before running visualization
3. **Compare with previous results** to verify improvement
4. **Save plots** with descriptive names
5. **Document findings** in markdown files

## Example Workflow

```bash
# 1. Start training
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name warmup_1000 \
    --episodes 1000 \
    --max-steps 500

# 2. Monitor progress (in another terminal)
watch -n 10 bash tools/monitor_training.sh

# 3. After training completes, visualize
python tools/visualize_1000ep_training.py

# 4. View results
open ablation_v3/visualizations/1000ep/training_curves_1000ep.png

# 5. Test routing dynamics
python tools/test_v3_routing_dynamic.py \
    --checkpoint ablation_v3/results/warmup_1000/checkpoints/best_model.pth

# 6. Document findings
# Edit ablation_v3/WARMUP_1000_RESULTS.md
```

## References

- Training script: `ablation_v3/train/train_v3_gat_moe.py`
- Visualization tool: `tools/visualize_1000ep_training.py`
- Monitoring tool: `tools/monitor_training.sh`
- Results guide: `ablation_v3/TRAINING_1000EP_README.md`
