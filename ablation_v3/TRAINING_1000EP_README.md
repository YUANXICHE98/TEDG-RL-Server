# V3 1000 Episodes Training

## Training Status

**Started**: 2026-01-07
**Configuration**:
- Episodes: 1000
- Max Steps: 500/episode
- Device: CPU
- Experiment Name: warmup_1000

## Monitor Training Progress

### Real-time Monitoring
```bash
# Check current progress
bash tools/monitor_training.sh

# Auto-refresh every 10 seconds
watch -n 10 bash tools/monitor_training.sh
```

### Check Process Status
```bash
# View recent output
tail -50 <process_output>

# Or check if training is still running
ps aux | grep train_v3_gat_moe
```

## Visualization (After Training Completes)

### Generate All Plots
```bash
conda activate tedg-rl-demo
python tools/visualize_1000ep_training.py
```

This will generate:
1. **training_curves_1000ep.png** - Comprehensive training curves
   - Episode rewards over time
   - Episode scores over time
   - Alpha entropy evolution
   - Score distribution by phase

2. **expert_analysis_1000ep.png** - Expert system analysis
   - Expert selection frequency
   - Expert performance comparison

3. **phase_comparison_1000ep.png** - Phase comparison table
   - Warmup phase (0-1000)
   - Transition phase (1000+) if applicable

4. **training_summary.txt** - Text summary report

### Custom Visualization
```bash
# Specify custom paths
python tools/visualize_1000ep_training.py \
    --log-dir ablation_v3/results/warmup_1000/logs \
    --output-dir ablation_v3/visualizations/1000ep
```

## Expected Results

### Warmup Phase (0-1000 episodes)
- **Alpha Entropy**: Should stay high (~1.38, close to log(4)=1.386)
- **Routing**: Softmax (all experts get gradients)
- **Score**: Gradual improvement
- **Expert Usage**: Relatively balanced

### Key Metrics to Watch
1. **Alpha Entropy Trend**
   - Should remain stable in Warmup
   - Indicates all experts are learning

2. **Score Improvement**
   - Should show upward trend
   - Moving average should increase

3. **Training Stability**
   - No NaN/Inf values
   - Gradients within reasonable range

## Files Generated

### During Training
```
ablation_v3/results/warmup_1000/
├── checkpoints/
│   ├── best_model.pth
│   ├── checkpoint_ep100.pth
│   ├── checkpoint_ep200.pth
│   └── ...
├── logs/
│   └── training_log.json
└── training.log
```

### After Visualization
```
ablation_v3/visualizations/1000ep/
├── training_curves_1000ep.png
├── expert_analysis_1000ep.png
├── phase_comparison_1000ep.png
└── training_summary.txt
```

## Troubleshooting

### Training Stopped Unexpectedly
```bash
# Check if process is still running
ps aux | grep train_v3_gat_moe

# Check last few lines of output
tail -100 <process_output>

# Resume from checkpoint (if available)
python ablation_v3/train/train_v3_gat_moe.py \
    --exp-name warmup_1000 \
    --episodes 1000 \
    --max-steps 500 \
    --resume ablation_v3/results/warmup_1000/checkpoints/checkpoint_latest.pth
```

### Visualization Fails
```bash
# Check if training log exists
ls -lh ablation_v3/results/warmup_1000/logs/training_log.json

# Check log format
head -20 ablation_v3/results/warmup_1000/logs/training_log.json

# Install missing dependencies
pip install matplotlib numpy
```

## Next Steps After 1000 Episodes

1. **Analyze Results**
   - Review all generated plots
   - Check if Alpha entropy is stable
   - Verify score improvement

2. **Compare with 50 Episodes**
   - Compare with `ablation_v3/results/init_fix_test/`
   - Check if longer training helps

3. **Continue to Transition Phase**
   - If results are good, continue to 3000 episodes
   - Sparsemax will start to sparsify routing

4. **Document Findings**
   - Update `ablation_v3/INIT_FIX_SUMMARY.md`
   - Create comparison plots

## Estimated Time

- **CPU Training**: ~1-2 hours for 1000 episodes
- **Visualization**: ~10 seconds
- **Total**: ~1-2 hours

## Notes

- All plot labels and titles are in English
- Training uses Softmax routing (Warmup phase)
- Alpha entropy should stay high (~1.38)
- Expert usage should be relatively balanced
