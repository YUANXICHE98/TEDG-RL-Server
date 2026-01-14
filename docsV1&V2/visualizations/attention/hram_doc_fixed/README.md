# Attention/Action heatmaps: `hram_doc_fixed`

- checkpoint: `ablation_v2/results/hram_doc_fixed/checkpoints/best_model.pth`
- model: `hram_doc`
- attn_kind: `channel_alpha`
- episodes: `3`, max_steps: `600`
- scenario_mode: `simple`
- masking: `off`
- matcher: `coverage`

## Outputs
- `docs/visualizations/attention/hram_doc_fixed/alpha_heatmap.png`
- `docs/visualizations/attention/hram_doc_fixed/action_freq_heatmap.png`
- `docs/visualizations/attention/hram_doc_fixed/action_prob_heatmap.png`
- `docs/visualizations/attention/hram_doc_fixed/action_freq_heatmap_T.png` (axis switched)
- `docs/visualizations/attention/hram_doc_fixed/action_prob_heatmap_T.png` (axis switched)
- `docs/visualizations/attention/hram_doc_fixed/summary.json`

## Notes
- `alpha_heatmap` shows which channel the policy trusts more in each scenario.
- `action_freq_heatmap` uses sampled actions; use `action_prob_heatmap` for the underlying distribution.
