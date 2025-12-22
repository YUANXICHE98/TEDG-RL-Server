# Attention/Action heatmaps: `sparse_moe`

- checkpoint: `ablation_v2/results/sparse_moe/checkpoints/best_model.pth`
- model: `multichannel`
- attn_kind: `channel_alpha`
- episodes: `5`, max_steps: `600`
- scenario_mode: `atoms`
- masking: `on`
- matcher: `embedding`

## Outputs
- `docs/visualizations/attention/sparse_moe/alpha_heatmap.png`
- `docs/visualizations/attention/sparse_moe/action_freq_heatmap.png`
- `docs/visualizations/attention/sparse_moe/action_prob_heatmap.png`
- `docs/visualizations/attention/sparse_moe/action_freq_heatmap_T.png` (axis switched)
- `docs/visualizations/attention/sparse_moe/action_prob_heatmap_T.png` (axis switched)
- `docs/visualizations/attention/sparse_moe/summary.json`

## Notes
- `alpha_heatmap` shows which channel the policy trusts more in each scenario.
- `action_freq_heatmap` uses sampled actions; use `action_prob_heatmap` for the underlying distribution.
