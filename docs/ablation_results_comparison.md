# TEDG-RL V1 vs V2 实验对比（自动汇总）

- 生成时间: `2025-12-22 17:28:04`
- 汇总脚本: `tools/summarize_ablation_results.py`

## 当前仍在跑的实验（来自 `experiment_pids.txt`）

```
PID     ELAPSED CMD
194325  3-16:28:29 python -u ablation_v2/train/train_v2.py --exp-name no_mask --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding --no-mask
```

## 当前仍在跑的训练进程（实时 `ps` 扫描）

```
root      96922 37.9  0.1 915483648 2010244 ?   Rl   Dec21 696:32 python -u ablation_v2/train/train_hram.py --exp-name hram_e2e --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000
root     194325  302  0.1 914675416 1973780 ?   Rl   Dec19 16083:22 python -u ablation_v2/train/train_v2.py --exp-name no_mask --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding --no-mask
root     349256  273  0.0 5032524 379440 ?      RNl  17:26   3:38 python -u ablation_v2/train/train_v2.py --exp-name gumbel_fixed --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding --use-gumbel --gumbel-tau 1.5 --resume ablation_v2/results/gumbel/checkpoints/model_03000.pth
root     387587  303  0.2 914813960 2132436 ?   Rl   Dec20 9723:50 python -u ablation_v2/train/train_v2.py --exp-name gumbel_sparse --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding --use-gumbel --sparse-topk 1 --no-mask --resume ablation_v2/results/gumbel_sparse/checkpoints/model_00500.pth
root     687567  300  0.1 914793552 2100800 ?   Rl   Dec19 13825:45 python -u ablation_v2/train/train_v2.py --exp-name sparse_moe --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding --sparse-topk 2 --resume ablation_v2/results/sparse_moe/checkpoints/best_model.pth
root     836543  301  0.1 914648172 1992268 ?   Rl   Dec19 13782:40 python -u ablation_v2/train/train_v2.py --exp-name baseline --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding
root     935889  249  0.0 5629936 554604 ?      SNl  13:06 651:37 python -u ablation_v2/train/train_hram_doc.py --exp-name hram_doc_fixed --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --gumbel-tau 1.2 --resume ablation_v2/results/hram_doc/checkpoints/best_model.pth
```

## 可比性说明（非常重要）

- V1 默认 `500 steps/episode`（另有 `results_extended_steps=2000 steps/episode`）。
- V2 默认 `2000 steps/episode`，回报/分数分布与 V1 不在同一量纲；建议只比较：稳定性、是否学到“持续得分/生存”、以及同设置下的对照组。

## V1 结果总表（来自 `training_log.json`）

| Exp | Matcher | Steps/ep | Episodes | BestR | BestS | AvgR | AvgS | Last500R | Last500S | α(last500 mean) | 日志 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| results_embedding | embedding | 500 | 5000 | 642.48 | 620.00 | 9.29 | 9.47 | 11.07 | 11.21 | pre=0.250, scene=0.251, eff=0.250, rule=0.249 | `ablation_v1/results/results_embedding/logs/training_log.json` |
| results_extended_steps | embedding | 2000 | 1000 | 226.02 | 176.00 | 26.95 | 9.75 | 27.55 | 13.09 | pre=0.246, scene=0.254, eff=0.248, rule=0.252 | `ablation_v1/results/results_extended_steps/logs/training_log.json` |
| results_fixed_th | coverage | 500 | 5000 | 661.35 | 503.00 | 10.11 | 10.14 | 11.43 | 10.96 | pre=0.245, scene=0.257, eff=0.248, rule=0.250 | `ablation_v1/results/results_fixed_th/logs/training_log.json` |
| results_full | coverage | 500 | 5000 | 651.27 | 503.00 | 9.84 | 10.08 | 11.56 | 11.31 | pre=0.249, scene=0.254, eff=0.250, rule=0.247 | `ablation_v1/results/results_full/logs/training_log.json` |
| results_no_mask | coverage | 500 | 5000 | 792.24 | 623.00 | 10.65 | 10.59 | 9.60 | 10.59 | pre=0.249, scene=0.249, eff=0.253, rule=0.250 | `ablation_v1/results/results_no_mask/logs/training_log.json` |
| results_single_ch | coverage | 500 | 5000 | 1255.95 | 300.00 | 9.62 | 9.68 | 11.89 | 11.11 | pre=1.000, scene=0.000, eff=0.000, rule=0.000 | `ablation_v1/results/results_single_ch/logs/training_log.json` |

## V2 结果总表（来自 `training.log` 的周期性评估块）

| Exp | Steps/ep | SeenEp | LastEvalEp | LastAvgR | BestR | LastAvgS | BestS | α(last) | Route(last) | 稳定性 | 最新ckpt | 日志 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| baseline | 2000 | 3901 | 3900 | 25.60 | 1183.34 | 2.20 | 162.00 | pre=0.258, scene=0.247, eff=0.248, rule=0.247 | - | ok | `ablation_v2/results/baseline/checkpoints/model_03500.pth` | `ablation_v2/results/baseline/training.log` |
| no_mask | 2000 | 5117 | 5100 | 15.55 | 814.47 | 2.70 | 130.00 | pre=0.246, scene=0.251, eff=0.251, rule=0.252 | - | ok | `ablation_v2/results/no_mask/checkpoints/model_05000.pth` | `ablation_v2/results/no_mask/training.log` |
| gumbel | 2000 | 3658 | 3650 | 14.22 | 905.75 | 0.00 | 93.00 | pre=0.251, scene=0.253, eff=0.249, rule=0.247 | - | traceback | `ablation_v2/results/gumbel/checkpoints/model_03500.pth` | `ablation_v2/results/gumbel/training.log` |
| sparse_moe | 2000 | 4133 | 4100 | 27.55 | 824.05 | 6.30 | 123.00 | pre=0.248, scene=0.254, eff=0.251, rule=0.246 | - | ok | `ablation_v2/results/sparse_moe/checkpoints/model_04000.pth` | `ablation_v2/results/sparse_moe/training.log` |
| gumbel_sparse | 2000 | 3813 | 3800 | 21.90 | 516.97 | 0.00 | 45.00 | pre=0.242, scene=0.253, eff=0.263, rule=0.242 | - | ok | `ablation_v2/results/gumbel_sparse/checkpoints/model_03500.pth` | `ablation_v2/results/gumbel_sparse/training.log` |
| hram_doc | 2000 | 1025 | 1000 | 19.48 | 223.07 | 1.60 | 149.00 | - | Pre=25.0%, Scene=25.0%, Eff=25.0%, Rule=25.1% | nan_warn | `ablation_v2/results/hram_doc/checkpoints/model_01000.pth` | `ablation_v2/results/hram_doc/training.log` |
| hram_e2e | 2000 | 1469 | 1450 | 23.15 | 468.05 | 1.60 | 465.00 | - | - | ok | `ablation_v2/results/hram_e2e/checkpoints/model_01000.pth` | `ablation_v2/results/hram_e2e/training.log` |

### V2 其他目录（debug/smoke 等）

- 这些不纳入主对比，但保留在CSV里。

## V2 组内排名（截至当前日志）

### 按 `BestR` 排名

| Rank | Exp | BestR | BestS | 稳定性 |
|---:|---|---:|---:|---|
| 1 | baseline | 1183.34 | 162.00 | ok |
| 2 | gumbel | 905.75 | 93.00 | traceback |
| 3 | sparse_moe | 824.05 | 123.00 | ok |
| 4 | no_mask | 814.47 | 130.00 | ok |
| 5 | gumbel_sparse | 516.97 | 45.00 | ok |
| 6 | hram_e2e | 468.05 | 465.00 | ok |
| 7 | hram_doc | 223.07 | 149.00 | nan_warn |

### 按 `BestS` 排名

| Rank | Exp | BestS | BestR | 稳定性 |
|---:|---|---:|---:|---|
| 1 | hram_e2e | 465.00 | 468.05 | ok |
| 2 | baseline | 162.00 | 1183.34 | ok |
| 3 | hram_doc | 149.00 | 223.07 | nan_warn |
| 4 | no_mask | 130.00 | 814.47 | ok |
| 5 | sparse_moe | 123.00 | 824.05 | ok |
| 6 | gumbel | 93.00 | 905.75 | traceback |
| 7 | gumbel_sparse | 45.00 | 516.97 | ok |

## 现在“停谁/让谁继续跑”的建议（基于当前日志+稳定性）

- **立刻暂停/不要继续**: `gumbel`（已发生 `logits=NaN` 导致 Traceback），`hram_doc`（出现 NaN/Inf 警告，虽然未必立刻崩，但训练质量不可信）。
- **可以继续跑到 `min_episodes=10000` 再下结论**: `baseline`、`sparse_moe`、`hram_e2e`（目前无崩溃迹象，best_score 也不差）。
- **作为对照组，建议“跑够就停”**: `no_mask`（用于验证mask必要性；如果跑到 10k 仍然 `LastAvgS≈0`，就可以停掉，把算力让给mask版本/HRAM）。
- **当前正在跑**: `no_mask`。建议让它至少跑到下一次 `model_05000.pth`（或直接跑到 10k），然后停止并启动 `baseline`/`hram_e2e` 的续跑。

## 推荐的续跑命令（尽量复用已有checkpoint）

- `baseline`（稳定基线，建议先跑到 10k）:
  - `python -u ablation_v2/train/train_v2.py --exp-name baseline --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding --resume ablation_v2/results/baseline/checkpoints/model_03500.pth`
- `sparse_moe`（如果要做稀疏专家对比，建议续跑到 10k）:
  - `python -u ablation_v2/train/train_v2.py --exp-name sparse_moe --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding --use-gumbel --sparse-topk 2 --resume ablation_v2/results/sparse_moe/checkpoints/model_04000.pth`
- `hram_e2e`（端到端检索，建议单独跑、学习率更稳）:
  - `python -u ablation_v2/train/train_hram.py --exp-name hram_e2e --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --resume ablation_v2/results/hram_e2e/checkpoints/model_01000.pth`

## 我建议你下一步先做的 2 件事

- 把 **对照组**跑全：让 `baseline`（mask+soft融合）和 `no_mask`（无mask）都跑到 10k，然后用 `best_score` + `last_avg_score` 判断 mask 的必要性。
- 再做 **创新组稳定化**：修好 `gumbel/hram_doc` 的 NaN 后再继续跑，不然 best_reward 峰值没有论文价值（不可复现/不稳定）。

