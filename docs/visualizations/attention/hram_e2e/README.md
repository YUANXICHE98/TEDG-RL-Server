# Attention/Action heatmaps: `hram_e2e`

- checkpoint: `ablation_v2/results/hram_e2e/checkpoints/best_model.pth`
- model: `hram_e2e`
- attn_kind: `knowledge_attn`
- episodes: `3`, max_steps: `600`
- scenario_mode: `simple`
- masking: `off`
- matcher: `coverage`

## Outputs
- `docs/visualizations/attention/hram_e2e/alpha_heatmap.png`
- `docs/visualizations/attention/hram_e2e/action_freq_heatmap.png`
- `docs/visualizations/attention/hram_e2e/action_prob_heatmap.png`
- `docs/visualizations/attention/hram_e2e/action_freq_heatmap_T.png` (axis switched)
- `docs/visualizations/attention/hram_e2e/action_prob_heatmap_T.png` (axis switched)
- `docs/visualizations/attention/hram_e2e/summary.json`

## Notes
- `alpha_heatmap` shows which channel the policy trusts more in each scenario.
- `action_freq_heatmap` uses sampled actions; use `action_prob_heatmap` for the underlying distribution.
