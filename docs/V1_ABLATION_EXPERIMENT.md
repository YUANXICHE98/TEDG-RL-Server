# TEDG-RL V1 消融实验文档（详细版）

> 最后更新：2025-12-22  
> 适用范围：`ablation_v1/`（V1 ConfMatch 系列实验）  
> 参考模板：`docs/V2_ABLATION_EXPERIMENT.md`（但本文件更细，补齐模块级设计与实现细节）

---

## 一、宏观概述

### 1.1 项目背景（V1 在做什么）

TEDG-RL 的目标是在 NetHack（NLE / NetHackScore-v0）中，让 RL Agent **在超图知识约束/引导下学习策略**：
- 超图提供“情境 → 可做什么/建议做什么”的结构化先验（超边包含 `pre_nodes / scene_atoms / eff_nodes / conditional_effects` 等字段）。
- RL 学“该做什么”：在 23 个离散动作上学习策略分布，并学习如何在 **4 个知识通道**（Pre/Scene/Effect/Rule）之间分配权重 `α`。

V1 的核心实现是 `train_confmatch.py`：把观测解析成 atoms → 在超图中做 Top‑K 匹配（ConfMatch）→ 构造 115 维 state → 用 PPO 训练 `MultiChannelPolicyNet`。

### 1.2 V1 消融实验目标

V1 主要想回答（对应“能不能学到 + 学到哪些结构”）：
1. **超图匹配方式**：覆盖率匹配（ConfMatch） vs 嵌入相似度匹配（EmbeddingMatcher）是否影响性能？
2. **多通道是否必要**：4 通道融合 vs 单通道（只用 pre）是否退化？
3. **置信度与路由**：动态阈值（rolling quantile） vs 固定阈值是否影响训练行为？
4. **动作掩码是否有效**：开启/关闭动作掩码的对比（但注意 V1 的掩码接线实现有已知限制，见 3.6）。
5. **单局最大步数**：从 `max_steps=1000` 扩到 `2000`（extended_steps）是否提升分数上限？

### 1.3 实验组设计（V1）

V1 的并行消融由 `ablation_v1/scripts/run_parallel_ablation.sh` 定义（通过环境变量开关功能）：

| 实验名 | 输出目录 | 关键开关 | 目的 |
|---|---|---|---|
| full | `ablation_v1/results/results_full` | `USE_EMBEDDING=0, USE_MASK=1, DYNAMIC_TH=1, SINGLE_CH=0` | 4通道 + 置信度路由 +（名义）掩码 |
| no_mask | `ablation_v1/results/results_no_mask` | `USE_MASK=0` | 验证动作掩码贡献（见 3.6 注意事项） |
| fixed_th | `ablation_v1/results/results_fixed_th` | `DYNAMIC_TH=0` | 动态阈值 vs 固定阈值 |
| single_ch | `ablation_v1/results/results_single_ch` | `SINGLE_CH=1` | 4通道是否必要 |
| embedding | `ablation_v1/results/results_embedding` | `USE_EMBEDDING=1` | 嵌入匹配 vs 覆盖率匹配 |
| extended_steps | `ablation_v1/results/results_extended_steps` | `MAX_STEPS=2000` + resume embedding | 延长探索时间（从 embedding best checkpoint 续训） |

> extended_steps 启动脚本：`ablation_v1/scripts/run_extended_experiment.sh`

---

## 二、主要文件结构（V1）

```
ablation_v1/
├── train/
│   ├── train_confmatch.py         # ✅ V1 主训练脚本（消融都基于它）
│   ├── train_verbose.py           # 对照/调试：随机选边 + HP伪置信度（不用于消融主结论）
│   └── train_nethack.py           # 更早期简化版（不用于消融主结论）
├── scripts/
│   ├── run_parallel_ablation.sh   # 5组并行消融（full/no_mask/fixed_th/single_ch/embedding）
│   └── run_extended_experiment.sh # extended_steps（从 embedding checkpoint 续训）
├── results/                       # V1 实验输出
│   ├── results_full/
│   ├── results_no_mask/
│   ├── results_fixed_th/
│   ├── results_single_ch/
│   ├── results_embedding/
│   └── results_extended_steps/
├── visualize_v1_results.py        # V1 全实验可视化（曲线/统计）
└── visualizations*/               # 已生成图表（含论文图）
```

复用的核心模块（与 V2 共用，位于 `src/core/`）：
- `src/core/state_constructor.py`：构造 115 维 state（拆分见第 5 章）
- `src/core/networks_correct.py`：`MultiChannelPolicyNet`（4 Actor + 注意力融合）
- `src/core/ppo_trainer.py`：PPO 训练器（含熵正则/α 熵正则）
- `src/core/hypergraph_matcher.py`：覆盖率 ConfMatch（4 通道覆盖率 + Top‑K）
- `src/core/hypergraph_loader.py`：`EmbeddingMatcher`（语义相似度匹配 + atom cache）
- `src/core/action_masking.py`：动作掩码（operator→action_id 映射）

---

## 三、V1 方法与架构（模块级细化）

### 3.1 端到端数据流（从观测到动作）

以 `ablation_v1/train/train_confmatch.py` 为准，单步流程如下：

```
obs(dict)
  ├─ blstats(27) + inv_* + glyphs/chars + condition bits
  ↓
Atoms 解析
  ├─ pre_nodes (≈65词表)
  ├─ scene_atoms (≈82词表)
  ├─ effect_atoms (eff_nodes词表子集)
  └─ rule_atoms (conditional_effects/规则词表子集)
  ↓
超图匹配（2选1）
  A) ConfMatch(coverage): HypergraphMatcher.match(plot_atoms, top_k=8)
  B) EmbeddingMatcher.match(all_atoms, top_k=8)
  ↓
选边 + 置信度
  - coverage: 每通道选一条边（pre/scene/effect/rule），confidence=max(score)
  - embedding: 取 top-1 边，confidence=top1 fused_score
  ↓
StateConstructor.construct_state(...)
  => state(115)=[belief(50), q_pre(15), q_scene(15), q_effect(8), q_rule(10), confidence(1), goal(16)]
  ↓
MultiChannelPolicyNet(state)
  - 4个Actor输出 logits_pre/scene/effect/rule (23)
  - AttentionWeightNet 输出 α(4)（softmax）
  - 融合：fused_logits = Σ α_i * logits_i
  ↓
（可选）ConfidenceRouter 路由 + ActionMasker 掩码
  ↓
Categorical(fused_logits) 采样 action
  ↓
env.step(action) → reward/next_obs
  ↓
PPOTrainer.store_transition(...) → PPOTrainer.update()
```

### 3.2 Atoms 解析（V1 的“场景理解输入”）

V1 在 `extract_state_from_nethack_obs()` 中做了相对完整的 atoms 解析（相比 V2 的简化 atoms）：

1) **数值状态**（来自 `blstats`）：
- `hp/hpmax → hp_ratio`、`depth`、`gold`、`hunger`、`ac`、`exp_level`、`power/power_max`、`x/y` 等。

2) **物品栏解析**（`_parse_inventory(obs)`）：
- 使用 `inv_oclasses / inv_strs` 识别 `has_weapon/has_food/has_wand/has_key_or_lockpick/...` 等。

3) **地图邻域解析**（`_analyze_glyphs(obs, x, y)`）：
- 在玩家周围 5×5 扫描 `glyphs/chars`，构造 `adjacent_to_monster/adjacent_to_door/on_stairs/near_altar/...` 等。

4) **状态异常（condition bits）**：
- `blind/confused/stunned/hallucinating` 等。

最终得到四类 atoms：
- `pre_nodes`：偏“状态是否满足/资源是否具备”（如 `hp_low/has_weapon/not_blind/...`）
- `scene_atoms`：偏“局部场景与策略标签”（如 `dlvl_2_5/adjacent_to_monster/strategy_attack/...`）
- `effect_atoms`：偏“期望效果/风险”（如 `hp_restored/combat_success/door_opened/...`）
- `rule_atoms`：偏“规则与条件效果”（如 `avoid_contact/if item.blessed == True/...`）

> 实务建议：V1 的 atoms 数量较多，利于匹配超图；但也引入噪声与稀疏性，后续版本（V2/HRAM）才引入更系统的“表示学习/检索融合”。

### 3.3 覆盖率 ConfMatch（HypergraphMatcher）

文件：`src/core/hypergraph_matcher.py`  
输入：`plot_atoms={"pre":..,"scene":..,"effect":..,"rule":..}`  
输出：Top‑K `MatchResult`（每条边带 4 通道覆盖率 + 全局分数）

**(1) 覆盖率定义**

对某条超边 e 的某通道覆盖率：

```
cov = |query ∩ edge| / max(|query|, 1)
```

四通道覆盖向量：
`match_vec = (cov_pre, cov_scene, cov_effect, cov_rule)`

**(2) 加权全局分数**

默认权重（V1/V2 复用）：
`w = (0.35, 0.35, 0.2, 0.1)`

```
global_score = w · match_vec
```

**(3) 时间衰减（V1 实现现状）**

matcher 支持指数衰减 `exp(-(t_now-t_i)/tau)`，但在 V1 `train_confmatch.py` 中调用为 `t_i=t_now`，因此 `decay≈1`，时间衰减在 V1 实验中基本未生效（这是实现细节，不是理论限制）。

**(4) 通道内选边**

在 Top‑K 内，V1 选择每个通道覆盖率最大的边（ties 用总分打破）：
- `pre_edge / scene_edge / effect_edge / rule_edge`

并将其用于构造 state 的 `pre_for_state / scene_for_state / eff_metadata / conditional_effects`。

### 3.4 嵌入匹配（EmbeddingMatcher）

文件：`src/core/hypergraph_loader.py` → `EmbeddingMatcher`

核心逻辑：
1) 超边文本化：对每条超边拼接 `operator + pre_nodes + scene_atoms + eff_nodes`，调用 embedding API 得到向量（缓存到 `data/cache/hypergraph_embedding_index_minsup5.pkl`）。
2) atom embedding：对每个 atom 单独 embedding 并缓存（`data/cache/atom_embedding_cache.pkl`）。
3) 查询向量：把当前 step 的所有 atoms 的向量做均值池化 + L2 normalize。
4) 相似度：对所有超边做余弦相似度（点积，因为都已归一化），取 Top‑K。
5) 置信度：默认取 top1 的 `fused_score`（语义相似度 + 可选时间衰减加成）。

V1 训练中（`TEDG_USE_EMBEDDING=1`）：
- 用 embedding Top‑1 边来构造 `pre_for_state / scene_for_state`（与覆盖率模式不同，覆盖率是“每通道选一条边”）。
- `confidence` 直接来自 embedding 相似度。

### 3.5 115 维 State 的构造（StateConstructor）

文件：`src/core/state_constructor.py`

**结构：**

```
state(115) = [
  belief(50),
  q_pre(15),
  q_scene(15),
  q_effect(8),
  q_rule(10),
  confidence(1),
  goal(16)
]
```

各部分的来源与编码策略（要点）：
- `belief(50)`：来自 `blstats` 的归一化数值 + 少量状态位（V1 只填充部分维度，其余为 0）。
- `q_pre/q_scene`：对 `pre_nodes/scene_atoms` 做“哈希到前 10 维 + 后 5 维统计特征”的轻量嵌入（可看作 V1 的手工特征压缩器）。
- `q_effect(8)`：从 `eff_metadata` 抽取 `success_probability/safety_score/applicability_confidence/avg_score_gain/...` 等数值（缺失则回退 0）。
- `q_rule(10)`：从 `conditional_effects` 做类型统计（blessed/cursed/poison/...）+ 风险特征。
- `confidence(1)`：匹配置信度（coverage 或 embedding）。
- `goal(16)`：V1 里是固定目标（如 `goal[0]=1`），用于让网络有“任务方向”输入。

### 3.6 置信度路由与动作掩码（V1 的实现注意事项）

V1 有两个“与超图约束相关”的控制模块：

1) **ConfidenceRouter（动态阈值）**
- 文件：`ablation_v1/train/train_confmatch.py`
- 逻辑：维护 rolling window 的置信度历史，warmup 后用分位数更新 `low/high threshold`。
- 开关：`TEDG_DYNAMIC_TH=1/0`

2) **ActionMasker（动作掩码）**
- 文件：`src/core/action_masking.py`
- 思路：从当前 `pre_nodes/scene_atoms/confidence` 推断可应用超边集合，映射到允许动作集合，mask 掉不可行动作 logits。

**重要：V1 训练脚本的接线限制**

在 `train_confmatch.py` 的动作选择阶段，掩码调用为：
`action_masker.get_action_mask([], [], confidence)`

由于 `pre_nodes/scene_atoms` 传入为空，掩码模块无法做“基于 atoms 的 operator 过滤”，实际效果会显著弱化（多数情况下近似“不过滤”）。  
因此：
- `no_mask` vs `full` 在 V1 中**可能不代表真实的掩码贡献**，解释时需谨慎。
- 如果要严谨复现实验，建议后续把 `[]` 改为当前 step 的 `pre_for_state/scene_for_state`，并把 mask 存入 PPO buffer（更新阶段一致应用）。

---

## 四、训练参数（V1）

V1 主要通过环境变量控制（参见 `ablation_v1/scripts/run_parallel_ablation.sh`）：

| 变量 | 默认 | 含义 |
|---|---:|---|
| `TEDG_NUM_EPISODES` | 10000 | 训练 episode 数 |
| `TEDG_MAX_STEPS` | 1000 | 每个 episode 最大步数 |
| `TEDG_EVAL_INTERVAL` | 50 | 打印评估统计间隔 |
| `TEDG_CKPT_INTERVAL` | 500 | checkpoint 保存间隔 |
| `TEDG_VERBOSE_INTERVAL` | 10 | verbose 打印间隔 |
| `TEDG_OUTPUT_DIR` | results_confmatch | 输出目录 |
| `TEDG_USE_EMBEDDING` | 0/1 | 0=coverage ConfMatch，1=EmbeddingMatcher |
| `TEDG_DYNAMIC_TH` | 1/0 | 动态阈值开关 |
| `TEDG_USE_MASK` | 1/0 | 掩码开关（见 3.6 接线限制） |
| `TEDG_SINGLE_CHANNEL` | 0/1 | 单通道开关（只用 pre Actor） |

PPO 超参来自 `train_confmatch.py` 里初始化 `PPOTrainer(...)`：
- `learning_rate=3e-4`
- `gamma=0.99`
- `gae_lambda=0.95`
- `clip_ratio=0.2`
- `ppo_epochs=3`
- `batch_size=128`

PPO 内部默认（见 `src/core/ppo_trainer.py`）：
- `entropy_coef=0.05`（鼓励探索）
- `alpha_entropy_coef=0.1`（鼓励 α 更均衡；对 E2E/HRAM 可设为 0，但 V1 默认开启）

---

## 五、模型架构（V1 MultiChannelPolicyNet，模块级说明）

文件：`src/core/networks_correct.py`

### 5.1 输入 / 输出

- 输入：`state (115,)` 或 `(batch,115)`
- 输出：
  - `fused_logits (23,)`：融合后的动作 logits
  - `alpha (4,)`：4 通道权重
  - `value (1,)`：Critic 价值估计

### 5.2 四个 Actor（专家）

| Actor | 输入（来自 state） | 拼接维度 | 输出 |
|---|---|---:|---|
| ActorPre | `q_pre(15)` + `belief_context(20)` | 35 | logits(23) |
| ActorScene | `q_scene(15)` + `location_context(20)` | 35 | logits(23) |
| ActorEffect | `q_effect(8)` + `hp_context(10)` | 18 | logits(23) |
| ActorRule | `q_rule(10)` + `inventory_context(15)` | 25 | logits(23) |

> `inventory_context` 在实现中由 `belief[45:50] + goal[:10]` 组成（见 `extract_contexts()`）。

### 5.3 AttentionWeightNet（α 权重网络）

- 输入：完整 `state(115)`
- MLP：`115 → 64 → 64 → 4`
- 输出：`alpha = softmax(logits)`（V1 为软融合；V2 才扩展到 gumbel/sparse）

### 5.4 融合方式

将 4 个 Actor 的 logits 堆叠成 `(4, 23)`，用 `alpha` 做加权和：

```
fused_logits[a] = Σ_i alpha[i] * logits_i[a]
```

### 5.5 单通道消融（single_ch）

当设置 `TEDG_SINGLE_CHANNEL=1` 时：
- `fused_logits = logits_pre`
- `alpha = [1,0,0,0]`

用于验证“多通道分工”是否必要。

---

## 六、动作空间（23 个动作）

本仓库的 23 动作默认以 `src/core/action_masking.py` 的映射为准（也与热力图工具一致）：

| id | 动作 |
|---:|---|
| 0 | MORE（确认/继续） |
| 1-8 | 8方向移动（N/E/S/W/NE/SE/SW/NW） |
| 9-16 | RUN（大写方向） |
| 17 | UP（上楼 `<`） |
| 18 | DOWN（下楼 `>`） |
| 19 | WAIT（等待 `.`） |
| 20 | KICK（踢 `k`/开门相关映射） |
| 21 | EAT（吃 `e`） |
| 22 | SEARCH（搜索 `s`/look 等映射） |

---

## 七、超图知识结构（V1 使用的数据）

V1 读取的超图文件：
- `data/hypergraph/hypergraph_complete_real.json`

典型超边字段（简化示例）：
```json
{
  "id": "he_001",
  "operator": "attack",
  "pre_nodes": ["player_alive", "has_weapon"],
  "scene_atoms": ["adjacent_to_monster", "combat_situation"],
  "eff_nodes": ["monster_killed", "xp_gained"],
  "eff_metadata": {
    "success_probability": 0.7,
    "safety_score": 0.4,
    "conditional_effects": [{"condition": "...", "effect": "..."}]
  }
}
```

V1 侧重使用：
- `pre_nodes/scene_atoms/eff_nodes`：用于 ConfMatch/Embedding 匹配
- `eff_metadata/conditional_effects`：用于构造 `q_effect/q_rule`

---

## 八、输出文件格式（V1）

每个实验目录结构：
```
ablation_v1/results/<exp>/
├── training.log                 # 控制台日志（含 eval 打印）
├── logs/training_log.json       # 结构化指标
└── checkpoints/
    ├── best_model.pth
    └── model_XXXXX.pth
```

`logs/training_log.json`（V1 实际字段）：
```json
{
  "episode_rewards": [...],
  "episode_lengths": [...],
  "episode_scores": [...],
  "alpha_history": [...],
  "best_reward": 0.0,
  "best_score": 0,
  "total_episodes": 5000,
  "total_time_seconds": 12345.6,
  "device": "musa:0",
  "timestamp": "..."
}
```

---

## 九、监控与复现实验

### 9.1 启动并行消融（V1）

```bash
cd ablation_v1
bash scripts/run_parallel_ablation.sh
```

### 9.2 启动 extended_steps（V1）

```bash
cd ablation_v1
bash scripts/run_extended_experiment.sh
```

### 9.3 常用监控命令

```bash
# 查看训练进程
ps aux | grep train_confmatch.py | grep -v grep

# 看单个实验日志
tail -f ablation_v1/results/results_full/training.log

# 生成 V1 可视化
python ablation_v1/visualize_v1_results.py
```

---

## 十、当前 V1 结果摘要（score 为主）

以下取自已落盘的 V1 训练日志（见 `ablation_v1/results/*/logs/training_log.json`）：

| 实验 | best_score | best_reward | episodes |
|---|---:|---:|---:|
| results_no_mask | 623 | 792.24 | 5000 |
| results_embedding | 620 | 642.48 | 5000 |
| results_fixed_th | 503 | 661.35 | 5000 |
| results_full | 503 | 651.27 | 5000 |
| results_single_ch | 300 | 1255.95 | 5000 |
| results_extended_steps | 176 | 226.02 | 1000 |

配套图表（已生成）：
- `ablation_v1/visualizations/`
- `ablation_v1/visualizations_paper/`

---

## 附录：V1 vs V2 的关键差异（便于论文叙述）

- V1 主要创新在 **atoms→超图匹配→115维state→多通道策略学习** 的闭环；多处仍是“轻量手工编码”（`StateConstructor`）。
- V2 在 V1 的基础上系统化了：动作掩码一致性、gumbel 硬路由、稀疏专家、HRAM 检索增强/端到端等。
- V1 的某些开关（如掩码/动态阈值）在当前实现中存在接线限制（见 3.6），对消融结论应标注“实现版本假设”。

