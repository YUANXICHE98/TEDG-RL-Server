# TEDG-RL V2 消融实验文档（详细版）

> 最后更新：2025-12-22  
> 适用范围：`ablation_v2/`（V2：MultiChannel + Gumbel/Sparse + HRAM）  
> 参考模板：旧版 `docs/V2_ABLATION_EXPERIMENT.md`（本文件补齐模块级设计与科研思路）  
> 当前训练状态以 `docs/training_status.md` 为准

---

## 一、宏观概述（科研思路）

### 1.1 项目背景：为什么要用“超图 + RL”

NetHack 是高噪声、长时序、稀疏奖励的复杂环境。纯 RL 常见问题是样本效率低、探索浪费在“根本不可行”的动作上。  
TEDG-RL 的核心是把游戏知识抽象为 **任务超图**，并把它用在两层：

1) **约束层（hard constraint）**：动作掩码把“不可行动作”直接禁掉，减少无意义探索。  
2) **引导层（soft guidance）**：把知识匹配/检索结果编码为 state 的一部分，让策略学“在不同知识源之间如何自适应融合”。

### 1.2 V2 的核心假设

V2 主要验证三个科学假设（对应实验设计）：

- **H1：Action Masking 有效**  
  在知识约束下训练的策略应更稳定、更高样本效率。

- **H2：路由（Routing）比平均融合更能学到“结构化决策”**  
  用 gumbel/稀疏激活迫使专家分工，可能比 softmax 融合更清晰、更泛化。

- **H3：检索增强策略（HRAM）能更好利用“多样化超图嵌入”**  
  让策略从 Top‑K 知识中“挑选并融合”，比把知识压扁成手工特征更强。

### 1.3 实验组设计（7 组）

| # | 实验名 | 架构 | 关键特性 | 科研对比点 |
|---|--------|------|----------|------------|
| 1 | `baseline` | MultiChannelPolicyNet | EmbeddingMatcher + soft α + mask | 基线：语义检索 + 软融合 + 约束 |
| 2 | `no_mask` | MultiChannelPolicyNet | EmbeddingMatcher + 无 mask | 验证 hard constraint 的价值 |
| 3 | `gumbel` | MultiChannelPolicyNet | gumbel‑softmax α | “明确分工”是否优于平均融合 |
| 4 | `sparse_moe` | MultiChannelPolicyNet | top‑2 稀疏 α | 稀疏激活是否更高效/更稳 |
| 5 | `gumbel_sparse` | MultiChannelPolicyNet | gumbel + top‑1 | 极致稀疏是否塌缩 |
| 6 | `hram_doc` | HRAMPolicyNetDoc | 检索上下文 + 4专家 + gumbel router | “保留专家结构”的检索增强 |
| 7 | `hram_e2e` | HRAMPolicyNet (V3.2) | Top‑K 检索 + Cross‑Attention | 检索‑融合端到端作为策略组件 |

---

## 二、主要文件结构（V2）

### 2.1 训练脚本

```
ablation_v2/
├── train/
│   ├── train_v2.py          # MultiChannelPolicyNet 训练 (baseline/no_mask/gumbel/sparse_moe/gumbel_sparse)
│   ├── train_hram.py        # HRAMPolicyNet (V3.2) 端到端训练 (hram_e2e)
│   └── train_hram_doc.py    # HRAMPolicyNetDoc 文档方案训练 (hram_doc*)
├── scripts/
│   ├── run_all_experiments.sh
│   ├── run_ablation_v2.sh
│   └── run_hram_experiments.sh
└── results/
    └── <exp_name>/
        ├── training.log
        ├── checkpoints/
        └── logs/
```

### 2.1.1 各实验组的真实启动命令（与脚本一致）

以 `ablation_v2/scripts/run_all_experiments.sh` 为准（训练参数/开关直接体现在命令行）：

- `baseline`：`python -u ablation_v2/train/train_v2.py --exp-name baseline --use-embedding`（默认启用 mask）
- `no_mask`：`python -u ablation_v2/train/train_v2.py --exp-name no_mask --use-embedding --no-mask`
- `gumbel`：`python -u ablation_v2/train/train_v2.py --exp-name gumbel --use-embedding --use-gumbel --gumbel-tau 1.0`
- `sparse_moe`：`python -u ablation_v2/train/train_v2.py --exp-name sparse_moe --use-embedding --use-gumbel --sparse-topk 2`
- `gumbel_sparse`：`python -u ablation_v2/train/train_v2.py --exp-name gumbel_sparse --use-embedding --use-gumbel --gumbel-tau 0.5 --sparse-topk 1`
- `hram_doc`：`python -u ablation_v2/train/train_hram_doc.py --exp-name hram_doc`
- `hram_e2e`：`python -u ablation_v2/train/train_hram.py --exp-name hram_e2e --embed-dim 3072`

> 注：上面的脚本还会统一设置 `episodes/max-steps/min-episodes/patience`；你做论文写作时，可以直接引用脚本里的“统一预算”作为实验 protocol。

### 2.2 核心模块（建议阅读顺序）

1) `src/core/state_constructor.py`：115 维 state 的分解与编码  
2) `src/core/networks_correct.py`：V2 多通道策略网络（专家 + α）  
3) `src/core/action_masking.py`：mask 规则与 operator→action 映射  
4) `src/core/ppo_trainer.py`：PPO 更新逻辑（特别是 mask 一致性）  
5) `src/core/hypergraph_loader.py`：EmbeddingMatcher（语义检索 + 缓存 + 离线）  
6) `src/core/networks_hram.py`：HRAMPolicyNet / HRAMPolicyNetDoc（检索增强策略）

### 2.3 数据与缓存（实际路径）

```
data/
├── hypergraph/
│   └── hypergraph_complete_real.json          # 超图结构（约450条超边）
└── cache/
    ├── hypergraph_embedding_index_minsup5.pkl # 超边嵌入索引（min_support=5）
    └── atom_embedding_cache.pkl               # atom 嵌入缓存
```

---

## 三、V2 总体流程（从观测到动作）

### 3.1 MultiChannel（实验 1‑5）的决策闭环

```
obs(dict)
  ↓
atoms 解析（pre_nodes/scene_atoms/...） + confidence（来自匹配/检索）
  ↓
StateConstructor → state(115)
  ↓
MultiChannelPolicyNet(state)
  ├─ 4个Actor 输出 logits_pre/scene/effect/rule (23)
  ├─ AttentionWeightNet 输出 α(4)
  └─ fused_logits = Σ α_i * logits_i
  ↓
（可选）ActionMasker → action_mask(23) → masked_logits
  ↓
Categorical(masked_logits) 采样 action
  ↓
env.step(action) → reward/next_obs
  ↓
PPO：store_transition + update（更新阶段同样应用 action_mask）
```

### 3.2 HRAM（实验 6‑7）的差异

- `hram_doc`：保留“4 个专家 + α 路由”，但专家输入来自检索上下文（不是手工 q_*）。  
- `hram_e2e`：把“Top‑K 检索 + 从 K 条知识中挑选（cross-attn）”做成网络内部模块，attention 本身就是可解释信号。

### 3.3 重要说明：MultiChannel 与 HRAM 的 state 语义并不完全一致（当前实现现状）

为了科研结论更“干净”，这里把当前代码真实情况讲清楚（方便你后续写论文时解释“可比性/控制变量”）：

- MultiChannel（`train_v2.py`）：使用 `StateConstructor` 构造的 **结构化 115 维 state**：`belief(50)+q_pre(15)+q_scene(15)+q_effect(8)+q_rule(10)+confidence(1)+goal(16)`。
- HRAM（`train_hram.py` / `train_hram_doc.py`）：目前直接用 `blstats` 等构造 **“原始/简化”115 维 state**（不包含 q_pre/q_scene/confidence 的那套结构化切片）。

这意味着：`baseline/no_mask/gumbel/...` 与 `hram_*` 的对比，除了“网络结构差异”之外，还混入了“输入特征差异”的变量。  
如果你后续要把 HRAM 的收益写得更扎实，建议你把 HRAM 的输入也统一到 `StateConstructor` 的结构化 state（或者反过来统一成原始 state），这样消融更强。

---

## 四、MultiChannelPolicyNet（实验 1‑5）——模块级说明与设计理由

文件：`src/core/networks_correct.py`

### 4.1 输入 / 输出定义

- 输入：`state ∈ R^{115}`
- 输出：
  - `fused_logits ∈ R^{23}`：动作 logits
  - `alpha ∈ R^{4}`：四通道权重（`[pre, scene, effect, rule]`，和为 1）
  - `value ∈ R`：Critic 价值估计

科研上 `alpha` 是一个非常重要的“可解释路由变量”：  
你可以用它研究“在什么场景更信任规则/更信任效果预期/更信任场景证据”等。

### 4.1.1 α 的“软/硬”实现差异（对应你的科研叙述）

同样叫“attention/α”，但在不同实验里它的“离散性”不同：

- `baseline/no_mask`：`alpha = softmax(router_logits)` → 典型 **软路由/软融合**
- `gumbel`：`alpha = gumbel_softmax(..., hard=training)` → 训练期更像 **硬路由**（one‑hot 倾向），评估期变回软路由
- `sparse_moe`：先 soft/gumbel 得到 α，再只保留 top‑2 归一化 → **稀疏路由**
- `gumbel_sparse`：只保留 top‑1 → **极稀疏路由（最容易塌缩）**

写论文时建议用一句话把它抽象成：  
“我们用 (soft / gumbel / sparse) 三类路由机制控制专家激活的稀疏度，以检验‘明确分工’是否提升知识调用的可解释性与样本效率。”

### 4.2 四个 Actor（四个专家）

| 专家 | 关注的信息 | 输入（从 state 切片） | 输出 |
|---|---|---|---|
| ActorPre | 可行性/前置条件 | `q_pre(15) + belief[:20]` | logits(23) |
| ActorScene | 局部场景 | `q_scene(15) + belief[20:40]` | logits(23) |
| ActorEffect | 效果/风险 | `q_effect(8) + belief[40:50]` | logits(23) |
| ActorRule | 规则/物品约束 | `q_rule(10) + inventory_context(15)` | logits(23) |

为什么要“拆成专家”而不是一个网络？
- **结构化归因**：更容易解释（论文里可视化 α/专家分工）。  
- **减少梯度干扰**：不同信号（场景/规则/风险）在一个网络里容易互相干扰。  
- **方便做消融**：no_mask/gumbel/sparse 等都是围绕这个结构验证的。

### 4.3 AttentionWeightNet（α 的生成器）

α 来自一个小 MLP：`state(115) → 4`。V2 的消融基本都发生在“α 怎么变得稀疏/更确定”上：

- `baseline`：`alpha = softmax(router_logits)`（软融合）
- `gumbel`：`alpha = gumbel_softmax(router_logits, tau, hard=training)`（硬路由倾向）
- `sparse_moe`：保留 top‑2 通道并归一化（稀疏激活）
- `gumbel_sparse`：gumbel + top‑1（极致稀疏，风险是塌缩）

为什么 gumbel/稀疏可能更好？
- soft 融合容易“平均主义”：α 接近均匀，专家分工不明显  
- gumbel/稀疏能逼迫形成“场景‑专家”对应关系（更像结构化知识调用）  
- 但过强稀疏会让探索不足或训练不稳 → 需要通过消融验证

### 4.4 融合方式

```
fused_logits[a] = Σ_i alpha[i] * logits_i[a]
```

这一步相当于“动态加权委员会”：α 表示“当前状态下该听哪个专家更重要”。

---

## 五、动作掩码（ActionMasking）——约束的设计理由与实现细节

文件：`src/core/action_masking.py`

### 5.1 科研动机

动作掩码把超图的可行性先验变成 hard constraint：  
减少不可行动作带来的噪声探索，提高样本效率和稳定性。

### 5.2 实现方式（高层）

1) 对当前 atoms，筛出“可应用超边”（简化阈值：pre/scene 匹配度 ≥ 0.5 且 confidence ≥ 0.3）。  
2) 收集这些超边的 `operator` 集合。  
3) `operator → action_id`（23 动作 id）映射为允许动作集合。  
4) 生成 `mask(23)`，对 logits 做 `-inf` 屏蔽。  
5) 兜底：如果 mask 误杀导致整行 `-inf`，回退原 logits（避免 NaN）。

> 代码细节（`get_applicable_edges`）：匹配度计算是 `|query ∩ edge| / |edge|`（以超边节点数量为分母），因此 mask 的约束是“偏保守”的：只有当你的 atoms 覆盖到某条超边的大部分 pre/scene 节点时，才会激活该 operator。

### 5.3 训练一致性（非常关键）

**采样时使用了 mask，PPO 更新时必须使用相同 mask 重算分布**。否则：
- old_logp/new_logp 不在同一分布上
- PPO ratio 会被系统性污染
- 很容易出现 NaN 或“学不动”

对应实现：`src/core/ppo_trainer.py` 会把 `action_mask` 存入 buffer，更新阶段一致应用。

---

## 六、115 维 state（V2）

文件：`src/core/state_constructor.py`

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

设计理由（科研叙述角度）：
- `belief`：连续数值底座（血量/深度/位置/资源…）  
- `q_*`：把离散 atoms 压缩成低维输入（便于训练）  
- `confidence`：把“知识匹配可靠度”显式喂给策略，允许策略学“何时更保守/何时更依赖知识”  
- `goal`：目标条件化接口（便于扩展多任务）

> 注：V2 的 `train_v2.py` atoms 解析相对简化；如果你想更强的场景表达，可以向 V1 的“完整 atoms 解析”靠拢，或直接走 HRAM 的检索增强路线。

---

## 七、EmbeddingMatcher（语义检索）——为何比纯覆盖率更鲁棒

文件：`src/core/hypergraph_loader.py`

### 7.1 为什么要引入语义检索

覆盖率匹配需要字符串完全重合，难以处理：
- 同义表达（`low_hp` vs `hp_low`）
- atoms 缺失（观测不完整）
- 场景组合泛化（新组合未见过）

EmbeddingMatcher 用向量相似度替代完全匹配，让“语义相近”的知识也能被检索到。

### 7.2 可复现与工程稳定（缓存 + 离线降级）

EmbeddingMatcher 有两个缓存：
- 超边嵌入索引：`data/cache/hypergraph_embedding_index_minsup5.pkl`
- atom 嵌入缓存：`data/cache/atom_embedding_cache.pkl`

并支持离线降级（避免网络/DNS 问题阻塞训练）：
- `TEDG_OFFLINE_EMBEDDINGS=1` 强制不访问外部 API  
- 离线伪嵌入使用 `sha256(atom)` 作为随机种子生成单位向量（可复现，维度一致）

---

## 八、HRAM（实验 6‑7）——把检索与融合纳入策略网络

文件：`src/core/networks_hram.py`

### 8.1 HRAMPolicyNet（`hram_e2e`，V3.2）：Top‑K 检索 + Cross‑Attention 融合

模块与职责：

1) **Retrieval encoder**：`state → query(3072)`  
2) **HypergraphMemory**：`query → topK embeddings(3072)`（cosine 检索，数值更稳）  
3) **KnowledgeAdapter**：`3072 → 256`（把知识翻译到 RL 空间）  
4) **Decision encoder**：`state → agent_state(256)`（决策表征与检索表征分离）  
5) **CrossAttentionFusion**：从 K 条知识里学会“挑/聚合”，输出 `logits(23)` + `attn_weights(1,K)`  
6) **Critic**：`state → value`（价值估计不依赖知识，减噪）

科研意义：
- `attn_weights` 可解释：不同场景关注哪些知识条目（可做热力图/案例分析）
- 这是“多样化超图嵌入 + RL 自定义匹配”的直接实现

#### 8.1.1 可微性/学习信号（写论文时常被问）

- `HypergraphMemory` 的 keys 是 **固定 buffer**（预先计算的超边嵌入），不会被 RL 直接更新；学习发生在 `retrieval_encoder / adapter / fusion`。  
- Top‑K 检索本身是离散索引（不可微），但这不妨碍端到端训练：梯度会通过“被选中的 K 条知识向量”回传到 `retrieval_encoder`（让 query 越来越能检索到对任务有用的知识）。

### 8.2 HRAMPolicyNetDoc（`hram_doc*`）：保留 4 专家结构，引入检索上下文

折中思路：
- 保留 V2 的“4 专家 + α”结构（便于解释与消融）
- 用“检索上下文压缩向量”替代手工 `q_pre/q_scene/...`

模块：
1) `StateEncoder → query(3072)`  
2) `HypergraphMemory → topK → mean pool`  
3) `ContextCompressor(3072→128)` 得到 shared knowledge  
4) 4 个专家分别看：`shared_knowledge + state slice`  
5) `GumbelRouter` 输出 α，融合 logits

科研意义：
- 如果 `hram_doc` > `baseline`：说明“检索上下文作为专家输入”比手工 q_* 更有效  
- 如果 `hram_e2e` 更强：说明“交叉注意力挑知识”比“平均池化 + 专家”更强

#### 8.2.1 当前实现的一个关键点：`hram_doc` 默认是“真正 one‑hot”的硬路由

`src/core/networks_hram.py` 的 `GumbelRouter` 在 `use_gumbel=True` 时是 `hard=True`，因此 `alpha` 通常是 one‑hot。  
这有两个科研含义：

- 优点：专家分工会更“可解释”（路由更离散）。  
- 风险：更容易出现“专家塌缩”（长期只激活某一个专家），需要通过 `route_counts`/热力图观察，并通过温度退火、熵正则、或 warmup 软路由来缓解。

---

## 九、动作空间（23 动作，与代码一致）

动作 id 约定与 `src/core/action_masking.py` 一致：

| id | 动作 |
|---:|---|
| 0 | MORE（确认/继续） |
| 1-8 | 8方向移动（N/E/S/W/NE/SE/SW/NW） |
| 9-16 | RUN（大写方向） |
| 17 | UP（上楼 `<`） |
| 18 | DOWN（下楼 `>`） |
| 19 | WAIT（等待 `.`） |
| 20 | KICK（踢/开门相关映射） |
| 21 | EAT（吃） |
| 22 | SEARCH（搜索/查看等映射） |

---

## 十、输出与可视化（V2）

### 10.1 结果目录

```
ablation_v2/results/<exp>/
├── training.log
├── checkpoints/
│   ├── best_model.pth
│   └── model_XXXXX.pth
└── logs/
    ├── training_log.json
    └── *.log
```

### 10.1.1 `logs/training_log.json`：科研复盘最重要的结构化记录

三类训练脚本都会在结束时输出 `ablation_v2/results/<exp>/logs/training_log.json`，其中包含（按脚本略有差异）：

- `episode_scores / episode_rewards / episode_lengths`：画曲线的原始序列
- `config`：命令行参数（写论文/复现实验必须引用）
- MultiChannel (`train_v2.py`) 额外含：
  - `alpha_history`：每局平均 α（用于“路由是否成形”）
  - `scene_actor_samples`：每 100 step 抽样的场景指标 + 动作 + α（用于热力图）
- HRAMDoc (`train_hram_doc.py`) 额外含：
  - `route_counts`：四专家被选中的计数（快速判断是否塌缩）
- HRAM E2E (`train_hram.py`) 额外含：
  - `scene_actor_samples`：同样用于 attention/action 的可视化（E2E 情况下会更偏“knowledge attention”）

### 10.2 状态表/对比表（自动生成）

- `docs/training_status.md`：实时进度（PID/设备/bestS/bestR/epAvgS200 等）  
- `docs/ablation_results_comparison.md`：V1/V2 汇总对比表  

### 10.3 可视化工具（你已经在用的）

- 训练曲线：`tools/plot_all_curves.py`（all/V1/V2）  
- 状态条形图：`tools/visualize_current_results.py`  
- attention/action 热力图：`tools/analyze_attention_heatmaps.py`

> 读“路由/注意力”的入门建议：先看 `docs/方法论V2/attention_analysis.md`（里面解释了 MA vs eval、如何判断“学到分工”）。

---

## 附录：如何做“科研式”读图与定位问题

1) **性能主指标优先看 score**：reward 会受塑形项影响，score 更贴近 NetHack 目标。  
2) **看 non‑zero score rate**：从 0 分占比下降开始，往往代表学到基础生存/探索。  
3) **看 α/attention 是否随场景变化**：  
   - 一直均匀：分工没形成（或信号不够）  
   - 过强 one‑hot 且不稳定：可能硬路由过强/塌缩  
4) **mask 的收益通常体现在稳定性与样本效率**：  
   - 若 no_mask 更好：可能 mask 过严误杀动作，或 mask/更新不一致导致噪声（当前实现已做一致性修复）
