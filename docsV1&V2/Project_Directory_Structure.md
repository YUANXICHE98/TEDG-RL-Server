# TEDG-RL-System 项目目录结构

## 完整项目树

```
TEDG-RL-System/
├── README.md                          # 项目文档
├── requirements.txt                   # Python依赖
├── setup.py                          # 安装配置
├── config.yaml                       # 全局配置
├── Makefile                          # 快捷命令
├── .gitignore
├── LICENSE
│
├── src/                              # 核心源代码
│   ├── core/                         # 核心算法实现
│   │   ├── networks/                 # 神经网络模块
│   │   │   ├── __init__.py
│   │   │   ├── actors.py            # 4个独立actor (pre, scene, effect, rule)
│   │   │   ├── attention.py         # AttentionWeightNet (α权重计算)
│   │   │   ├── critic.py            # 共享critic价值网络
│   │   │   ├── fusion.py            # FusionLayer (加权融合)
│   │   │   └── base.py              # 基础网络类
│   │   │
│   │   ├── encoders/                # 信息编码器（四通路）
│   │   │   ├── __init__.py
│   │   │   ├── hgnn.py              # HGNN编码器 (q_pre, q_scene)
│   │   │   ├── effect_encoder.py    # 效果编码器 (q_effect)
│   │   │   ├── rule_encoder.py      # 规则编码器 (q_rule)
│   │   │   └── base_encoder.py      # 编码器基类
│   │   │
│   │   ├── rl/                      # 强化学习算法
│   │   │   ├── __init__.py
│   │   │   ├── ppo_trainer.py       # PPO训练器 (核心训练循环)
│   │   │   ├── buffer.py            # 经验回放缓冲区 (轨迹+GAE)
│   │   │   ├── loss.py              # PPO损失函数
│   │   │   ├── action_masking.py    # 动作掩蔽 (超图约束)
│   │   │   ├── reward_computer.py   # 多分量奖励计算
│   │   │   └── base_agent.py        # agent基类
│   │   │
│   │   └── state/                   # 状态表示与构造
│   │       ├── __init__.py
│   │       ├── state_constructor.py # 115维state构造
│   │       ├── belief_tracker.py    # 证据图与belief向量
│   │       └── confidence_scorer.py # 置信度计算
│   │
│   ├── hypergraph/                  # 超图相关模块
│   │   ├── __init__.py
│   │   ├── hypergraph.py            # 超图数据结构 (450超边)
│   │   ├── hypergraph_embedding.py  # 超图嵌入管理器 (4通路并行)
│   │   ├── action_applicator.py     # 可行动作子图提取
│   │   ├── effect_analyzer.py       # 效果与风险分析
│   │   ├── rule_matcher.py          # 规则与条件效果匹配
│   │   └── failure_modes.py         # 失败模式定义与分类
│   │
│   ├── llm/                         # LLM相关接口
│   │   ├── __init__.py
│   │   ├── grounder.py              # LLM Grounding (观测→atoms)
│   │   ├── query_handler.py         # 查询处理 (query_property, reflection)
│   │   ├── llm_client.py            # LLM API客户端 (BERT/GPT)
│   │   └── prompts.py               # 提示词模板
│   │
│   ├── environment/                 # 游戏环境交互
│   │   ├── __init__.py
│   │   ├── nethack_env.py           # NetHack环境包装器 (NLE/NAO)
│   │   ├── observation_parser.py    # 游戏观测解析 (ASCII→atoms)
│   │   ├── action_executor.py       # 动作执行器 (game commands)
│   │   └── reward_observer.py       # 奖励观测 (death, progress)
│   │
│   └── utils/                       # 工具函数
│       ├── __init__.py
│       ├── logger.py                # 日志记录
│       ├── metrics.py               # 评估指标计算
│       ├── visualization.py         # α权重与轨迹可视化
│       ├── data_loader.py           # 离线数据加载 (NLD/NAO)
│       └── config_parser.py         # 配置解析
│
├── data/                            # 数据文件
│   ├── hypergraph/                  # 超图数据
│   │   ├── preconditions.json       # ~30个前置条件节点
│   │   ├── scene_atoms.json         # ~40个场景原子
│   │   ├── operators.json           # 33个操作符规则
│   │   ├── effect_edges.json        # 450个超边定义
│   │   ├── conditional_effects.json # 条件效果列表
│   │   ├── failure_modes.json       # 失败模式分类
│   │   └── rule_patterns.json       # 50个NetHack规则
│   │
│   ├── offline_trajectories/        # 离线轨迹数据
│   │   ├── nld_trajectories.pkl     # NLD(NetHack Leaderboard Dataset)
│   │   ├── nao_trajectories.pkl     # NAO(NetHack Autosaved Observations)
│   │   └── trajectory_metadata.json # 轨迹元数据
│   │
│   └── embeddings/                  # 预训练嵌入
│       ├── pretrained_hgnn.pth      # 预训练HGNN编码器
│       └── pretrained_rule_encoder.pth # 预训练RuleEncoder
│
├── configs/                         # 配置文件
│   ├── default.yaml                 # 默认配置
│   ├── training_default.yaml        # 训练超参数
│   ├── network_default.yaml         # 网络架构配置
│   ├── hypergraph_default.yaml      # 超图参数
│   └── reward_default.yaml          # 奖励权重配置
│
├── scripts/                         # 执行脚本
│   ├── train.py                     # 主训练脚本
│   ├── evaluate.py                  # 评估脚本
│   ├── preprocess_hypergraph.py     # 超图预处理
│   ├── preprocess_offline_data.py   # 离线数据预处理
│   ├── visualize_results.py         # 结果可视化
│   └── export_model.py              # 模型导出
│
├── notebooks/                       # Jupyter分析笔记本
│   ├── 01_hypergraph_exploration.ipynb      # 超图数据探索
│   ├── 02_embedding_visualization.ipynb     # 嵌入向量可视化
│   ├── 03_training_dynamics.ipynb           # 训练过程分析
│   ├── 04_alpha_weights_analysis.ipynb      # α权重变化分析
│   └── 05_trajectory_analysis.ipynb         # 学习轨迹分析
│
├── tests/                           # 单元测试与集成测试
│   ├── __init__.py
│   ├── test_networks.py             # 网络模块测试
│   ├── test_encoders.py             # 编码器测试
│   ├── test_rl_trainer.py           # PPO训练器测试
│   ├── test_hypergraph.py           # 超图模块测试
│   ├── test_state_construction.py   # 状态构造测试
│   └── test_integration.py          # 集成测试
│
└── results/                         # 实验结果输出
    ├── checkpoints/                 # 模型检查点
    │   ├── model_0000.pth
    │   ├── model_0100.pth
    │   └── model_latest.pth
    ├── logs/                        # 训练日志
    │   ├── train_log.txt
    │   └── metrics.json
    ├── metrics/                     # 性能指标
    │   ├── return_curves.png
    │   ├── loss_curves.png
    │   └── metrics_summary.json
    ├── visualizations/              # 可视化图表
    │   ├── alpha_weights_evolution.png
    │   ├── action_distribution.png
    │   └── trajectory_samples.png
    └── trajectories/                # 学习轨迹
        └── episode_traces.pkl
```

---

## 核心模块快速导航

### 网络模块 (src/core/networks/)
| 文件 | 类 | 功能 |
|------|----|----|
| actors.py | ActorPre<br/>ActorScene<br/>ActorEffect<br/>ActorRule | 4个独立actor<br/>各产生logits |
| attention.py | AttentionWeightNet | state(115) → α[4] |
| fusion.py | FusionLayer | 加权融合logits |
| critic.py | CriticNet | state(115) → value |

### 编码器模块 (src/core/encoders/)
| 通路 | 类 | 输入 | 输出 | 编码方式 |
|------|----|----|------|---------|
| q_pre | HGNNEncoder | 前置条件节点 | 15维 | HGNN 2阶段 |
| q_scene | HGNNEncoder | 场景原子 | 15维 | HGNN 2阶段 |
| q_effect | EffectEncoder | success/safety | 8维 | MLP |
| q_rule | RuleEncoder | 条件效果 | 10维 | Symbol Graph |

### RL模块 (src/core/rl/)
| 文件 | 类/函数 | 功能 |
|------|---------|------|
| ppo_trainer.py | PPOTrainer | 完整训练循环 |
| buffer.py | ReplayBuffer | 轨迹存储 + GAE计算 |
| loss.py | actor_loss()<br/>critic_loss()<br/>ppo_clipped_loss() | 损失函数 |
| reward_computer.py | RewardComputer | 5分量奖励 |
| action_masking.py | ActionMasker | -inf mask掉非法动作 |

### 超图模块 (src/hypergraph/)
| 文件 | 类 | 功能 |
|------|----|----|
| hypergraph.py | Hypergraph | 450超边数据结构 |
| hypergraph_embedding.py | HypergraphEmbedding | 4通路并行处理 |
| action_applicator.py | ActionApplicator | 可行动作子图提取 |
| effect_analyzer.py | EffectAnalyzer | success/safety评分 |
| rule_matcher.py | RuleMatcher | 条件效果匹配 |

---

## 执行脚本快速参考

### 训练
```bash
# 基础训练
python scripts/train.py --config configs/training_default.yaml

# 自定义超参数
python scripts/train.py \
  --config configs/training_default.yaml \
  --learning-rate 5e-4 \
  --batch-size 256 \
  --num-epochs 100

# 从checkpoint继续训练
python scripts/train.py \
  --config configs/training_default.yaml \
  --checkpoint results/checkpoints/model_0050.pth
```

### 评估
```bash
# 评估指定模型
python scripts/evaluate.py \
  --checkpoint results/checkpoints/model_latest.pth \
  --num-episodes 100

# 生成详细报告
python scripts/evaluate.py \
  --checkpoint results/checkpoints/model_latest.pth \
  --num-episodes 100 \
  --verbose
```

### 可视化
```bash
# 从日志目录生成图表
python scripts/visualize_results.py \
  --log-dir results/logs/ \
  --output-dir results/visualizations/

# 分析α权重
python scripts/visualize_results.py \
  --log-dir results/logs/ \
  --plot-alpha-weights \
  --output-dir results/visualizations/
```

### 预处理
```bash
# 构建超图数据结构
python scripts/preprocess_hypergraph.py \
  --output-dir data/hypergraph/

# 加载并转换离线轨迹
python scripts/preprocess_offline_data.py \
  --input-dir data/raw_trajectories/ \
  --output-dir data/offline_trajectories/
```

---

## 数据文件规范

### 超图数据 (data/hypergraph/)
```json
# preconditions.json
{
  "nodes": [
    {
      "id": "pre_0",
      "name": "has_gold",
      "type": "inventory",
      "compatible_with": ["pre_1", "pre_5", ...]
    },
    ...
  ]
}

# effect_edges.json
{
  "edges": [
    {
      "edge_id": 0,
      "precond_nodes": ["pre_0", "pre_1"],
      "effect_nodes": ["eff_5", "eff_10"],
      "success_probability": 0.95,
      "failure_modes": {
        "precond_violation": 10,
        "bad_aim": 5,
        ...
      }
    },
    ...
  ]
}
```

### 离线轨迹 (data/offline_trajectories/)
```python
# 格式：List[Dict]
[
  {
    "episode_id": 0,
    "steps": [
      {
        "state": {...},  # 游戏状态
        "action": "move_north",
        "reward": 0.1,
        "next_state": {...},
        "done": False
      },
      ...
    ],
    "return": 250.5,
    "length": 123,
    "death_cause": None
  },
  ...
]
```

---

## 配置文件示例

### training_default.yaml
```yaml
# 算法
algorithm: ppo
learning_rate: 3e-4
clip_range: 0.2
gae_lambda: 0.95
ppo_epochs: 3

# 训练
batch_size: 128
num_episodes: 10000
eval_interval: 100
checkpoint_interval: 50

# 网络
actor_hidden_dim: 128
critic_hidden_dim: 128
attention_hidden_dim: 64

# 奖励权重
reward_weights:
  progress: 0.3
  safety: 0.3
  efficiency: 0.2
  feasibility: 0.1
  exploration: 0.1
```

---

## 代码Agent实现顺序

### Phase 1: 基础设施 (Week 1)
1. ✅ 实现config_parser.py (配置加载)
2. ✅ 实现logger.py (日志系统)
3. ✅ 实现data_loader.py (数据加载)
4. ✅ 实现超图数据结构 (hypergraph.py)

### Phase 2: 编码器与网络 (Week 2-3)
5. ✅ 实现HGNN编码器
6. ✅ 实现EffectEncoder
7. ✅ 实现RuleEncoder
8. ✅ 实现4个actor网络
9. ✅ 实现AttentionWeightNet
10. ✅ 实现Critic网络

### Phase 3: RL算法 (Week 3-4)
11. ✅ 实现ReplayBuffer
12. ✅ 实现损失函数
13. ✅ 实现PPOTrainer
14. ✅ 实现奖励计算
15. ✅ 实现动作掩蔽

### Phase 4: 集成与优化 (Week 4-5)
16. ✅ 实现状态构造器
17. ✅ 实现超图嵌入管理器
18. ✅ 实现训练脚本
19. ✅ 实现评估脚本
20. ✅ 添加单元测试与集成测试

---

## 快速启动命令

```bash
# 设置开发环境
git clone https://github.com/your_repo/TEDG-RL-System
cd TEDG-RL-System
pip install -r requirements.txt

# 预处理数据
python scripts/preprocess_hypergraph.py
python scripts/preprocess_offline_data.py

# 开始训练
python scripts/train.py --config configs/training_default.yaml

# 评估模型
python scripts/evaluate.py --checkpoint results/checkpoints/model_latest.pth

# 生成报告
python scripts/visualize_results.py --log-dir results/logs/
```

