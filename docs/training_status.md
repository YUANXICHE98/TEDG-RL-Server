# Training Status (Auto)

- Generated: `2025-12-22 20:06:56`
- Scope: `ablation_v2/results`

## Summary Table

| exp | running | pid | ni | cpu% | flags | cfg | device | lastEvalEp | lastAvgS | bestS | lastAvgR | bestR | epCnt | epAvgS(200) | epNZ%(200) | epBestS(200) | epAvgR(200) | tb | nan | recommendation | log |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline | yes | 836543 | - | 299.00 | emb | ok | ✓ 使用MUSA设备: MTT S4000 | 4000 | 0.00 | 162.00 | 19.95 | 1183.34 | 4029 | 1.68 | 4.00 | 82.00 | 19.29 | no | no | continue (insufficient evidence) | `ablation_v2/results/baseline/training.log` |
| gumbel | no | - | - | - | - | - | ✓ 使用MUSA设备: MTT S4000 | 3650 | 0.00 | 93.00 | 14.22 | 905.75 | 3657 | 0.69 | 2.50 | 39.00 | 14.80 | yes | yes | continue (insufficient evidence) | `ablation_v2/results/gumbel/training.log` |
| gumbel_fixed | yes | 349256 | 19 | 315.00 | emb,gumbel,tau=1.5,resume | ok | ✓ 使用CPU设备 | 3450 | 1.00 | 34.00 | 20.05 | 391.18 | 482 | 0.28 | 1.50 | 34.00 | 15.64 | no | no | stop/repurpose (CPU): avgS stuck low, bestS low | `ablation_v2/results/gumbel_fixed/training.log` |
| gumbel_sparse | yes | 387587 | - | 299.00 | no_mask,emb,gumbel,topk=1,resume | unexpected_no_mask | ✓ 使用MUSA设备: MTT S4000 | 3900 | 0.00 | 45.00 | 15.08 | 516.97 | 3441 | 0.24 | 0.50 | 48.00 | 17.83 | no | no | stop/repurpose: avgS stuck low, bestS low | `ablation_v2/results/gumbel_sparse/training.log` |
| hram_doc | no | - | - | - | - | - | ✓ 使用MUSA设备: MTT S4000 | 1000 | 1.60 | 149.00 | 19.48 | 223.07 | 0 | - | - | - | - | no | yes | continue (insufficient evidence) | `ablation_v2/results/hram_doc/training.log` |
| hram_doc_fixed | yes | 935889 | 19 | 230.00 | tau=1.2,resume | ok | ✓ 使用CPU设备 | 1100 | 4.40 | 139.00 | 29.52 | 464.36 | 0 | - | - | - | - | no | no | continue (CPU): avgR improving | `ablation_v2/results/hram_doc_fixed/training.log` |
| hram_e2e | yes | 96922 | - | 37.10 | - | ok | ✓ 使用MUSA设备: MTT S4000 | 1550 | 2.70 | 465.00 | 17.88 | 468.05 | 0 | - | - | - | - | no | no | continue (insufficient evidence) | `ablation_v2/results/hram_e2e/training.log` |
| no_mask | yes | 194325 | - | 300.00 | no_mask,emb | ok | ✓ 使用MUSA设备: MTT S4000 | 5200 | 3.80 | 130.00 | 27.28 | 814.47 | 5231 | 2.58 | 6.50 | 111.00 | 23.77 | no | no | continue: avgS ok | `ablation_v2/results/no_mask/training.log` |
| sparse_moe | yes | 687567 | - | 298.00 | emb,topk=2,resume | ok | ✓ 使用MUSA设备: MTT S4000 | 4250 | 0.60 | 123.00 | 16.69 | 824.05 | 4255 | 4.57 | 13.00 | 102.00 | 19.77 | no | no | continue: avgS ok | `ablation_v2/results/sparse_moe/training.log` |

