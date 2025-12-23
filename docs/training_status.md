# Training Status (Auto)

- Generated: `2025-12-23 12:55:02`
- Scope: `ablation_v2/results`

## Summary Table

| exp | running | pid | ni | cpu% | flags | cfg | device | lastEvalEp | lastAvgS | bestS | lastAvgR | bestR | epCnt | epAvgS(200) | epNZ%(200) | epBestS(200) | epAvgR(200) | tb | nan | recommendation | log |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline | no | - | - | - | - | - | ✓ 使用MUSA设备: MTT S4000 | 4150 | 0.90 | 162.00 | 19.55 | 1183.34 | 4160 | 0.35 | 1.00 | 43.00 | 18.30 | no | no | continue+tune: rare high score, avgS low (needs stability) | `ablation_v2/results/baseline/training.log` |
| gumbel | no | - | - | - | - | - | ✓ 使用MUSA设备: MTT S4000 | 3650 | 0.00 | 93.00 | 14.22 | 905.75 | 3657 | 0.69 | 2.50 | 39.00 | 14.80 | yes | yes | continue (insufficient evidence) | `ablation_v2/results/gumbel/training.log` |
| gumbel_fixed | no | - | - | - | - | - | ✓ 使用CPU设备 | 4000 | 0.00 | 71.00 | 10.28 | 391.18 | 1006 | 0.04 | 0.50 | 7.00 | 10.88 | no | no | stop/repurpose (CPU): avgS stuck low, bestS low | `ablation_v2/results/gumbel_fixed/training.log` |
| gumbel_sparse | no | - | - | - | - | - | ✓ 使用MUSA设备: MTT S4000 | 4050 | 0.00 | 48.00 | 25.86 | 516.97 | 3589 | 0.34 | 1.50 | 48.00 | 17.32 | no | no | stop/repurpose: avgS stuck low, bestS low | `ablation_v2/results/gumbel_sparse/training.log` |
| hram_doc | no | - | - | - | - | - | ✓ 使用MUSA设备: MTT S4000 | 1000 | 1.60 | 149.00 | 19.48 | 223.07 | 0 | - | - | - | - | no | yes | continue (insufficient evidence) | `ablation_v2/results/hram_doc/training.log` |
| hram_doc_fixed | yes | 544491 | 19 | 167.00 | tau=1.2,resume | ok | ✓ 使用MUSA设备: MTT S4000 | 450 | 0.10 | 76.00 | 33.31 | 734.44 | 0 | - | - | - | - | no | no | stop/repurpose: avgS stuck low, bestS low | `ablation_v2/results/hram_doc_fixed/training.log` |
| hram_e2e | yes | 96922 | - | 71.50 | - | ok | ✓ 使用MUSA设备: MTT S4000 | 5950 | 4.20 | 465.00 | 23.87 | 779.43 | 0 | - | - | - | - | no | no | continue: avgS ok | `ablation_v2/results/hram_e2e/training.log` |
| no_mask | no | - | - | - | - | - | ✓ 使用MUSA设备: MTT S4000 | 5350 | 0.70 | 175.00 | 14.65 | 814.47 | 5373 | 4.08 | 9.00 | 175.00 | 21.30 | no | no | continue (insufficient evidence) | `ablation_v2/results/no_mask/training.log` |
| sparse_moe | no | - | - | - | - | - | ✓ 使用MUSA设备: MTT S4000 | 4350 | 3.30 | 156.00 | 26.56 | 824.05 | 4367 | 4.10 | 10.50 | 156.00 | 21.98 | no | no | continue: avgS ok | `ablation_v2/results/sparse_moe/training.log` |

