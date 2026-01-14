# TEDG-RL V1 vs V2 实验对比（自动汇总）

- 生成时间: `2025-12-24 11:57:32`
- 汇总脚本: `tools/summarize_ablation_results.py`

## 当前仍在跑的实验（来自 `experiment_pids.txt`）

- 未检测到 `experiment_pids.txt` 对应的存活进程（或 `ps` 不可用）。

## 当前仍在跑的训练进程（实时 `ps` 扫描）

- 未检测到运行中的 `ablation_v2/train/*.py` 进程。

## 可比性说明（非常重要）

- V1 默认 `500 steps/episode`（另有 `results_extended_steps=2000 steps/episode`）。
- V2 默认 `2000 steps/episode`，回报/分数分布与 V1 不在同一量纲；建议只比较：稳定性、是否学到“持续得分/生存”、以及同设置下的对照组。

## V1 结果总表（来自 `training_log.json`）

| Exp | Matcher | Steps/ep | Episodes | BestR | BestS | AvgR | AvgS | Last500R | Last500S | α(last500 mean) | 日志 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|

## V2 结果总表（来自 `training.log` 的周期性评估块）

| Exp | Steps/ep | SeenEp | LastEvalEp | LastAvgR | BestR | LastAvgS | BestS | α(last) | Route(last) | 稳定性 | 最新ckpt | 日志 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|

## V2 组内排名（截至当前日志）

### 按 `BestR` 排名

| Rank | Exp | BestR | BestS | 稳定性 |
|---:|---|---:|---:|---|

### 按 `BestS` 排名

| Rank | Exp | BestS | BestR | 稳定性 |
|---:|---|---:|---:|---|

## 现在“停谁/让谁继续跑”的建议（基于当前日志+稳定性）

- **立刻暂停/不要继续**: `gumbel`（已发生 `logits=NaN` 导致 Traceback），`hram_doc`（出现 NaN/Inf 警告，虽然未必立刻崩，但训练质量不可信）。
- **可以继续跑到 `min_episodes=10000` 再下结论**: `baseline`、`sparse_moe`、`hram_e2e`（目前无崩溃迹象，best_score 也不差）。
- **作为对照组，建议“跑够就停”**: `no_mask`（用于验证mask必要性；如果跑到 10k 仍然 `LastAvgS≈0`，就可以停掉，把算力让给mask版本/HRAM）。

## 推荐的续跑命令（尽量复用已有checkpoint）

- `baseline`（稳定基线，建议先跑到 10k）:
  - `python -u ablation_v2/train/train_v2.py --exp-name baseline --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding`
- `sparse_moe`（如果要做稀疏专家对比，建议续跑到 10k）:
  - `python -u ablation_v2/train/train_v2.py --exp-name sparse_moe --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding --use-gumbel --sparse-topk 2`
- `hram_e2e`（端到端检索，建议单独跑、学习率更稳）:
  - `python -u ablation_v2/train/train_hram.py --exp-name hram_e2e --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000`

## 我建议你下一步先做的 2 件事

- 把 **对照组**跑全：让 `baseline`（mask+soft融合）和 `no_mask`（无mask）都跑到 10k，然后用 `best_score` + `last_avg_score` 判断 mask 的必要性。
- 再做 **创新组稳定化**：修好 `gumbel/hram_doc` 的 NaN 后再继续跑，不然 best_reward 峰值没有论文价值（不可复现/不稳定）。

