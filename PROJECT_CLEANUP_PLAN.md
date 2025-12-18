# 项目清理计划

## 当前使用的核心文件（保留）

### 训练脚本
- ✅ `train_confmatch.py` - 主实验脚本（多通道超图匹配）
- ✅ `train_verbose.py` - 对照脚本（HP伪置信度）
- ✅ `train_nethack.py` - 简化版脚本

### src/core/ 核心模块（train_confmatch.py 使用）
- ✅ `state_constructor.py` - 状态构造器
- ✅ `networks_correct.py` - 多通道网络架构
- ✅ `ppo_trainer.py` - PPO训练器
- ✅ `action_masking.py` - 动作掩码
- ✅ `hypergraph_matcher.py` - 超图匹配器
- ✅ `hypergraph_loader.py` - 超图加载器（被 matcher 使用）
- ✅ `__init__.py` - 包初始化

### 配置和数据
- ✅ `config.yaml` - 配置文件
- ✅ `requirements.txt` - 依赖列表
- ✅ `data/` - 数据目录

---

## 需要删除的文件

### 过时的 Markdown 文档（临时分析/改动说明）
- ❌ `FIXES_APPLIED.md` - 修复说明（已过时）
- ❌ `PROBLEM_ANALYSIS.md` - 问题分析（已过时）
- ❌ `STAGE2_CLARIFICATION.md` - 阶段2澄清（已过时）
- ❌ `UNDERSTANDING_RL_QUERY.md` - RL查询理解（已过时）
- ❌ `RL_LEARNING_QUERY_EXPLAINED.md` - 重复内容
- ❌ `THEORETICAL_QUESTIONS.md` - 理论问答（已过时）
- ❌ `RESEARCH_ROADMAP.md` - 科研路线图（已过时）
- ❌ `CORE_INSIGHT.md` - 核心洞察（已过时）
- ❌ `FLOW_EXPLAINED.md` - 流程解释（已过时）
- ❌ `MDP_EXPLAINED.md` - MDP解释（已过时）
- ❌ `HYPERGRAPH_VALUE.md` - 超图价值（已过时）
- ❌ `LLM_GROUNDING_STATUS.md` - LLM状态（已过时）
- ❌ `PROJECT_STATUS.md` - 项目状态（已过时）
- ❌ `DEPLOYMENT.md` - 部署说明（已过时）
- ❌ `Debugging RL Training.md` - 调试说明（已过时）
- ❌ `README_FINAL.md` - 重复的README

### 保留的有用文档
- ✅ `README.md` - 项目主README
- ✅ `TRAIN_SCRIPTS_COMPARISON.md` - 脚本对比（有用）
- ✅ `QUICK_START_GUIDE.md` - 快速启动指南（有用）

### src/core/ 不用的Python文件
- ❌ `networks.py` - 旧版网络（已被 networks_correct.py 替代）
- ❌ `rl_agent.py` - 未使用的RL代理
- ❌ `mixed_embedding.py` - 未使用的混合嵌入
- ❌ `offline_data_loader.py` - 离线数据加载器（当前在线RL不需要）

---

## 清理操作

### 删除过时的Markdown文档（15个）
```bash
rm FIXES_APPLIED.md
rm PROBLEM_ANALYSIS.md
rm STAGE2_CLARIFICATION.md
rm UNDERSTANDING_RL_QUERY.md
rm RL_LEARNING_QUERY_EXPLAINED.md
rm THEORETICAL_QUESTIONS.md
rm RESEARCH_ROADMAP.md
rm CORE_INSIGHT.md
rm FLOW_EXPLAINED.md
rm MDP_EXPLAINED.md
rm HYPERGRAPH_VALUE.md
rm LLM_GROUNDING_STATUS.md
rm PROJECT_STATUS.md
rm DEPLOYMENT.md
rm "Debugging RL Training.md"
rm README_FINAL.md
```

### 删除不用的Python文件（4个）
```bash
rm src/core/networks.py
rm src/core/rl_agent.py
rm src/core/mixed_embedding.py
rm src/core/offline_data_loader.py
```

---

## 清理后的项目结构

```
TEDG-RL-Server/
├── README.md                          # 项目主README
├── TRAIN_SCRIPTS_COMPARISON.md        # 脚本对比说明
├── QUICK_START_GUIDE.md               # 快速启动指南
├── config.yaml                        # 配置文件
├── requirements.txt                   # 依赖列表
├── train_confmatch.py                 # 主实验脚本
├── train_verbose.py                   # 对照脚本
├── train_nethack.py                   # 简化版脚本
├── src/
│   └── core/
│       ├── __init__.py
│       ├── state_constructor.py       # 状态构造
│       ├── networks_correct.py        # 多通道网络
│       ├── ppo_trainer.py             # PPO训练
│       ├── action_masking.py          # 动作掩码
│       ├── hypergraph_matcher.py      # 超图匹配
│       └── hypergraph_loader.py       # 超图加载
├── data/
│   └── hypergraph/
│       └── hypergraph_complete_real.json
└── docs/
    └── 方法论V2/
        ├── TEDG-RL-Hypergraph-Recipe.md
        └── RLDong-Zuo-Kong-Jian-v2.md
```

---

## 保留原因说明

### 保留的文档
1. **README.md** - 项目基础说明
2. **TRAIN_SCRIPTS_COMPARISON.md** - 详细对比三个训练脚本的差异，帮助选择使用哪个
3. **QUICK_START_GUIDE.md** - 快速启动和测试指南

### 保留的代码
1. **state_constructor.py** - 构造115维状态向量
2. **networks_correct.py** - 多通道Actor + AttentionWeightNet + Critic
3. **ppo_trainer.py** - PPO算法实现
4. **action_masking.py** - 基于置信度的动作掩码
5. **hypergraph_matcher.py** - 4通道覆盖度匹配 + 时间衰减
6. **hypergraph_loader.py** - 加载超图和嵌入索引

### 删除原因
- **过时的Markdown** - 都是临时的问题分析和改动说明，已经完成修复
- **networks.py** - 被 networks_correct.py 替代
- **rl_agent.py, mixed_embedding.py** - 未被任何脚本使用
- **offline_data_loader.py** - 当前在线RL训练不需要
