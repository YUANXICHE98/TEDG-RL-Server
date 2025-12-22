# TEDG-RL V2 消融实验文档

> 最后更新: 2024-12-19
> 实验状态: 7组实验并行运行中

---

## 一、宏观概述

### 1.1 项目背景

TEDG-RL 是一个基于**超图知识引导**的强化学习框架，用于 NetHack 游戏。核心思想是利用游戏知识（以超图形式表示）来指导 Agent 的决策。

### 1.2 V2 消融实验目标

验证不同架构设计对性能的影响，回答以下问题：
1. **动作掩码 (Action Masking)** 是否有效？
2. **Gumbel-Softmax 硬路由**是否优于软融合？
3. **稀疏专家 (Sparse MoE)** 是否能提升效率？
4. **H-RAM 端到端学习**是否优于显式多通道设计？

### 1.3 实验组设计 (7组)

| # | 实验名 | 架构 | 关键特性 | 对比点 |
|---|--------|------|----------|--------|
| 1 | **baseline** | MultiChannelPolicyNet | Embedding + Softmax融合 + Mask | 基线 |
| 2 | **no_mask** | MultiChannelPolicyNet | Embedding + 无动作掩码 | 验证 Mask 有效性 |
| 3 | **gumbel** | MultiChannelPolicyNet | Gumbel-Softmax 硬路由 | 硬路由 vs 软融合 |
| 4 | **sparse_moe** | MultiChannelPolicyNet | Top-2 稀疏专家 | 稀疏激活 |
| 5 | **gumbel_sparse** | MultiChannelPolicyNet | Gumbel + Top-1 | 极致稀疏 |
| 6 | **hram_doc** | HRAMPolicyNetDoc | 4 Actors + 检索上下文 + Gumbel | 文档方案 |
| 7 | **hram_e2e** | HRAMPolicyNet | 端到端 + 交叉注意力 | 端到端学习 |

---

## 二、主要文件结构

### 2.1 训练脚本

```
ablation_v2/
├── train/
│   ├── train_v2.py          # MultiChannelPolicyNet 训练 (实验1-5)
│   ├── train_hram.py        # HRAMPolicyNet 端到端训练 (实验7)
│   └── train_hram_doc.py    # HRAMPolicyNetDoc 文档方案训练 (实验6)
├── scripts/
│   └── run_all_experiments.sh  # 7组实验并行启动脚本
├── results/                 # 实验结果输出目录
│   ├── baseline/
│   ├── no_mask/
│   ├── gumbel/
│   ├── sparse_moe/
│   ├── gumbel_sparse/
│   ├── hram_doc/
│   └── hram_e2e/
└── visualize_v2_results.py  # 结果可视化脚本
```

### 2.2 核心网络文件

```
src/core/
├── networks_correct.py      # MultiChannelPolicyNet (4 Actor + 注意力融合)
├── networks_hram.py         # HRAMPolicyNet / HRAMPolicyNetDoc
├── ppo_trainer.py           # PPO 训练器
├── action_masking.py        # 动作掩码模块
├── state_constructor.py     # 状态构建器 (超图 → 状态向量)
├── hypergraph_matcher.py    # 超图匹配器
└── embedding_matcher.py     # 嵌入匹配器 (语义相似度)
```

### 2.3 数据文件

```
data/
├── hypergraph_minsup5.json           # 超图数据 (450条超边)
├── hypergraph_embedding_index_minsup5.pkl  # 超边嵌入索引 (268条)
└── atom_embeddings_cache.pkl         # Atom 嵌入缓存 (88个)
```

---

## 三、架构设计对比

### 3.1 MultiChannelPolicyNet (实验1-5)

```
┌─────────────────────────────────────────────────────────────┐
│                    MultiChannelPolicyNet                     │
├─────────────────────────────────────────────────────────────┤
│  State (115维)                                               │
│      ↓                                                       │
│  ┌─────────┬─────────┬─────────┬─────────┐                  │
│  │ActorPre │ActorScene│ActorEffect│ActorRule│  ← 4个独立Actor │
│  └────┬────┴────┬────┴────┬────┴────┬────┘                  │
│       ↓         ↓         ↓         ↓                        │
│  ┌─────────────────────────────────────────┐                │
│  │      AttentionWeightNet (α权重)          │                │
│  │  Softmax / Gumbel-Softmax / Sparse MoE  │                │
│  └─────────────────────────────────────────┘                │
│       ↓                                                       │
│  Fused Logits → Action                                       │
└─────────────────────────────────────────────────────────────┘
```

**变体差异:**

| 变体 | α计算方式 | 融合方式 |
|------|----------|----------|
| baseline | Softmax | 加权求和 (4个全参与) |
| gumbel | Gumbel-Softmax (τ=1.0) | 硬路由 (接近 one-hot) |
| sparse_moe | Gumbel + Top-2 | 只激活 2 个专家 |
| gumbel_sparse | Gumbel (τ=0.5) + Top-1 | 只激活 1 个专家 |

### 3.2 HRAMPolicyNet (实验7: hram_e2e)

```
┌─────────────────────────────────────────────────────────────┐
│                      HRAMPolicyNet                           │
├─────────────────────────────────────────────────────────────┤
│  State (115维)                                               │
│      ↓                                                       │
│  StateEncoder → Query (3072维)                               │
│      ↓                                                       │
│  HypergraphMemory (268条超边嵌入)                            │
│      ↓ 检索                                                  │
│  CrossAttentionFusion ← Retrieved Context                   │
│      ↓                                                       │
│  Action Logits                                               │
└─────────────────────────────────────────────────────────────┘
```

**特点:** 端到端学习，无需显式的4通道划分

### 3.3 HRAMPolicyNetDoc (实验6: hram_doc)

```
┌─────────────────────────────────────────────────────────────┐
│                    HRAMPolicyNetDoc                          │
├─────────────────────────────────────────────────────────────┤
│  State (115维)                                               │
│      ↓                                                       │
│  StateEncoder → Query                                        │
│      ↓                                                       │
│  HypergraphMemory → Retrieved Context                       │
│      ↓                                                       │
│  ContextCompressor → Compressed Context (128维)             │
│      ↓                                                       │
│  ┌─────────┬─────────┬─────────┬─────────┐                  │
│  │ActorPre │ActorScene│ActorEffect│ActorRule│  ← 4个HRAM Actor│
│  └────┬────┴────┬────┴────┬────┴────┬────┘                  │
│       ↓                                                       │
│  GumbelRouter → 硬路由选择                                   │
│       ↓                                                       │
│  Action Logits                                               │
└─────────────────────────────────────────────────────────────┘
```

**特点:** 保留4个Actor专家 + 检索增强 + Gumbel硬路由

---

## 四、训练参数 (统一配置)

### 4.1 核心训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--episodes` | 50000 | 最大训练轮数 |
| `--max-steps` | 2000 | 每轮最大步数 (统一考试时间) |
| `--min-episodes` | 10000 | 最少训练轮数 (≈2000万步) |
| `--patience` | 5000 | 连续无提升轮数阈值 |
| `--lr` | 3e-4 | 学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--gae-lambda` | 0.95 | GAE λ 参数 |
| `--clip-eps` | 0.2 | PPO clip 范围 |
| `--batch-size` | 64 | 批大小 |

### 4.2 Gumbel-Softmax 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--gumbel-tau` | 1.0 (gumbel) / 0.5 (gumbel_sparse) | 温度参数 |
| `--sparse-topk` | None / 2 / 1 | Top-K 专家数 |

### 4.3 H-RAM 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--embed-dim` | 3072 | 嵌入维度 (与超边嵌入对齐) |
| `--context-dim` | 128 | 压缩后上下文维度 |
| `--top-k-retrieval` | 5 | 检索 Top-K 超边数 |

---

## 五、状态向量结构 (115维)

### 5.1 完整结构

```
State Vector (115维) = [
    belief (50维),      # 信念状态
    q_pre (15维),       # 前置条件特征
    q_scene (15维),     # 场景特征
    q_effect (8维),     # 效果特征
    q_rule (10维),      # 规则特征
    confidence (1维),   # 匹配置信度
    goal (16维)         # 目标特征
]
```

### 5.2 Belief 向量 (50维) 详细

| 索引 | 名称 | 来源 | 说明 |
|------|------|------|------|
| 0 | player_alive | HP > 0 | 玩家存活 |
| 1 | game_active | HP > 0 | 游戏进行中 |
| 2 | any_hp | HP > 0 | 有生命值 |
| 3 | hp_ratio | HP / HP_MAX | 血量比例 |
| 4 | hunger_satiated | hunger == 0 | 饱食状态 |
| 5 | no_gold | gold == 0 | 无金币 |
| 6 | low_hp | HP < 5 | 低血量 |
| 7 | critical_hp | HP < 3 | 危险血量 |
| 8 | wounded | HP / HP_MAX < 0.5 | 受伤状态 |
| 9 | depth_normalized | depth / 20 | 地下城深度 |
| 10 | gold_normalized | log(gold+1) / 10 | 金币 (对数) |
| 11 | ac_normalized | (20 - AC) / 40 | 护甲等级 |
| 12 | exp_normalized | exp_level / 30 | 经验等级 |
| 13-49 | reserved | 0 | 预留位 |

### 5.3 NetHack blstats 索引 (NLE 常量)

```python
import nle.nethack as nh

# 关键字段索引
nh.NLE_BL_X        = 0   # X 坐标
nh.NLE_BL_Y        = 1   # Y 坐标
nh.NLE_BL_STR25    = 2   # 力量 (0-25)
nh.NLE_BL_STR125   = 3   # 力量 (0-125)
nh.NLE_BL_DEX      = 4   # 敏捷
nh.NLE_BL_CON      = 5   # 体质
nh.NLE_BL_INT      = 6   # 智力
nh.NLE_BL_WIS      = 7   # 智慧
nh.NLE_BL_CHA      = 8   # 魅力
nh.NLE_BL_SCORE    = 9   # 游戏分数
nh.NLE_BL_HP       = 10  # 当前 HP
nh.NLE_BL_HPMAX    = 11  # 最大 HP
nh.NLE_BL_DEPTH    = 12  # 地下城深度
nh.NLE_BL_GOLD     = 13  # 金币
nh.NLE_BL_ENE      = 14  # 当前魔力
nh.NLE_BL_ENEMAX   = 15  # 最大魔力
nh.NLE_BL_AC       = 16  # 护甲等级 (越低越好)
nh.NLE_BL_HD       = 17  # Hit Dice
nh.NLE_BL_XP       = 18  # 经验等级
nh.NLE_BL_EXP      = 19  # 经验值
nh.NLE_BL_TIME     = 20  # 游戏时间
nh.NLE_BL_HUNGER   = 21  # 饥饿状态 (0=饱, 1=正常, ...)
nh.NLE_BL_CAP      = 22  # 负重状态
nh.NLE_BL_DTEFN    = 23  # 困难等级
nh.NLE_BL_DLEVEL   = 24  # Dungeon Level
nh.NLE_BL_CONDITION = 25 # 状态条件
nh.NLE_BL_ALIGN    = 26  # 阵营

# 总长度
nh.NLE_BLSTATS_SIZE = 27
```

---

## 六、动作空间 (23个动作)

| 索引 | 动作 | 说明 |
|------|------|------|
| 0-7 | 移动 | 8个方向移动 |
| 8 | 等待 | 原地等待 |
| 9 | 攻击 | 攻击相邻敌人 |
| 10 | 拾取 | 拾取物品 |
| 11 | 丢弃 | 丢弃物品 |
| 12 | 使用 | 使用物品 |
| 13 | 穿戴 | 穿戴装备 |
| 14 | 脱下 | 脱下装备 |
| 15 | 阅读 | 阅读卷轴 |
| 16 | 喝 | 喝药水 |
| 17 | 投掷 | 投掷物品 |
| 18 | 施法 | 施放法术 |
| 19 | 祈祷 | 祈祷 |
| 20 | 搜索 | 搜索隐藏门/陷阱 |
| 21 | 踢 | 踢门/物品 |
| 22 | 开门 | 打开门 |

---

## 七、超图知识结构

### 7.1 超边格式

```json
{
  "pre_nodes": ["player_alive", "has_weapon"],
  "scene_atoms": ["monster_adjacent", "low_hp"],
  "operator": "attack",
  "eff_nodes": ["monster_killed", "exp_gained"],
  "rule_nodes": ["combat_rule"]
}
```

### 7.2 四通道对应关系

| 通道 | Actor | 输入特征 | 关注点 |
|------|-------|----------|--------|
| Pre | ActorPre | q_pre (15维) + belief | 前置条件是否满足 |
| Scene | ActorScene | q_scene (15维) + location | 当前场景特征 |
| Effect | ActorEffect | q_effect (8维) + hp_context | 预期效果/风险 |
| Rule | ActorRule | q_rule (10维) + inventory | 游戏规则约束 |

---

## 八、输出文件格式

### 8.1 训练日志 (`logs/training_log.json`)

```json
{
  "experiment_name": "gumbel",
  "start_time": "2024-12-19T00:00:00",
  "episode_rewards": [0.1, 0.2, ...],
  "episode_lengths": [100, 150, ...],
  "episode_scores": [0, 10, ...],
  "alpha_history": [[0.25, 0.25, 0.25, 0.25], ...],
  "best_avg_score": 50.0,
  "scene_actor_samples": [...],
  "converged": false,
  "stopped_at": 10000
}
```

### 8.2 模型检查点 (`checkpoints/`)

- `best_model.pth` - 最佳模型
- `model_final.pth` - 最终模型
- `model_00500.pth` - 定期保存

---

## 九、监控命令

```bash
# 查看所有实验进度
tail -n 10 ablation_v2/results/*/training.log

# 实时跟踪单个实验
tail -f ablation_v2/results/gumbel/training.log

# 查看进程状态
ps aux | grep "train_v2\|train_hram" | grep -v grep

# 查看 GPU 使用
nvidia-smi

# 停止所有实验
pkill -f "train_v2\|train_hram"

# 生成可视化
python ablation_v2/visualize_v2_results.py
```

---

## 十、预期结果分析

### 10.1 关键对比

| 对比组 | 预期结论 |
|--------|----------|
| baseline vs no_mask | Mask 应该提升性能 (避免无效动作) |
| baseline vs gumbel | 硬路由可能有更清晰的专家分工 |
| gumbel vs gumbel_sparse | Top-1 极致稀疏是否过于激进 |
| baseline vs hram_e2e | 端到端学习是否优于显式通道设计 |
| hram_doc vs hram_e2e | 保留 Actor 结构是否比纯端到端更好 |

### 10.2 评估指标

1. **Episode Score** - 游戏分数 (主要指标)
2. **Episode Length** - 存活步数
3. **Episode Reward** - 累计奖励
4. **Alpha Distribution** - 专家权重分布 (专业化程度)
5. **收敛速度** - 达到稳定性能的轮数

---

## 附录：快速参考

### A. 启动实验

```bash
cd /root/autodl-tmp/TEDG-RL-Server
bash ablation_v2/scripts/run_all_experiments.sh
```

### B. 单独运行某个实验

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# baseline
python ablation_v2/train/train_v2.py --exp-name baseline --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding

# gumbel
python ablation_v2/train/train_v2.py --exp-name gumbel --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000 --use-embedding --use-gumbel

# hram_e2e
python ablation_v2/train/train_hram.py --exp-name hram_e2e --episodes 50000 --max-steps 2000 --min-episodes 10000 --patience 5000
```

### C. 查看实验结果

```bash
# 查看训练曲线
python ablation_v2/visualize_v2_results.py

# 结果保存在
ls ablation_v2/results/*/logs/training_log.json
```
