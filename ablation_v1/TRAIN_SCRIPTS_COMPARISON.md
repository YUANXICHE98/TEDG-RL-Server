# TEDG-RL 训练脚本对比与选择指南

## 概览

| 脚本 | 用途 | 置信度来源 | 超图匹配 | 输出目录 | 状态 |
|------|------|-----------|---------|---------|------|
| `train_confmatch.py` | **主实验脚本** | Top-K超图匹配 + 时间衰减 | HypergraphMatcher (4通道覆盖度) | `results_confmatch/` | ✅ **推荐使用** |
| `train_verbose.py` | 调试/对照脚本 | HP伪置信度 (0.5+0.3*hp_ratio) | 随机选边 | `results/` | ⚠️ 仅作对照 |
| `train_nethack.py` | 简化版脚本 | HP伪置信度 | 随机选边 | `results/` | ⚠️ 早期版本 |

---

## 详细对比

### 1. `train_confmatch.py` ✅ **主实验脚本（推荐）**

**核心特性：**
- ✅ 使用 `HypergraphMatcher` 进行 **Top-K 超边匹配**（4通道覆盖度计算）
- ✅ 支持 **指数时间衰减**（`tau=200.0`，虽然当前 `t_i=t_now` 实际未启用）
- ✅ **多通道选边**：从 Top-K 中为 pre/scene/effect/rule 各选最佳边
- ✅ **置信度 = max(topk_scores)**，反映超图匹配质量
- ✅ 输出到独立目录 `results_confmatch/`，不影响旧实验
- ✅ 已修复 `blstats` 索引（使用 `nh.NLE_BL_*` 常量）

**状态构造流程：**
```
NetHack obs → blstats (27维)
  ↓
belief (50维): hp_ratio, depth, gold, hunger, 6属性, x, y, score, ...
  ↓
推断 pre_nodes / scene_atoms（基于 HP/饥饿/金币/AC/经验等阈值）
  ↓
HypergraphMatcher.match(plot_atoms, t_now, t_i, top_k=8)
  → Top-K 匹配结果（每条边有 4通道覆盖度 + 时间衰减）
  ↓
select_channel_edges(topk) → 为每通道选最佳边
  ↓
confidence = max(topk.score)
  ↓
StateConstructor.construct_state(
  belief, pre_nodes, scene_atoms, eff_metadata, conditional_effects, confidence, goal
) → 115维 state
```

**PPO 训练循环：**
- **输入**：115维 state = [belief(50) + q_pre(15) + q_scene(15) + q_effect(8) + q_rule(10) + confidence(1) + goal(16)]
- **网络**：
  - 4个独立 Actor（pre/scene/effect/rule）
  - 1个 AttentionWeightNet（输出 α 权重，4维）
  - 1个共享 Critic（输出 V 值）
- **输出**：
  - 动作分布：4个 Actor 输出加权融合 → 23维动作概率
  - α 权重：[α_pre, α_scene, α_effect, α_rule]
  - V 值：状态价值估计
- **奖励**：NetHack 环境原生奖励（主要是分数增量）
- **超参数**：
  - lr=3e-4, clip_ratio=0.2, gamma=0.99, gae_lambda=0.95
  - ppo_epochs=3, batch_size=128
  - max_steps=1000, num_episodes=10000

**当前问题与待改进点：**
1. ❌ **时间衰减未真正启用**：`t_i=t_now` 导致 `decay=1.0`（需要维护历史 `t_i`）
2. ❌ **无置信度阈值分支**：未实现高/中/低置信度的不同策略
3. ❌ **无 Query 动作**：未区分执行动作（A类）和查询动作（Q类）
4. ❌ **无 LLM 接入**：未实现低置信度时的 LLM fallback
5. ❌ **无动作掩码应用**：虽然初始化了 `ActionMasker`，但 `select_action` 未使用

**适用场景：**
- ✅ **主实验**：验证 TEDG-RL 多通道超图匹配 + α 权重学习
- ✅ **消融实验基线**：作为"完整方法"的起点，逐步关闭特性做消融

---

### 2. `train_verbose.py` ⚠️ **调试/对照脚本**

**核心特性：**
- ⚠️ **HP伪置信度**：`confidence = 0.5 + 0.3 * hp_ratio`（不反映超图匹配质量）
- ⚠️ **随机选边**：`edge = np.random.choice(edges)`（无超图匹配逻辑）
- ✅ 详细的 verbose 日志（每10个 episode 打印状态/匹配/网络决策）
- ✅ 已修复 `blstats` 索引

**用途：**
- 对照实验：验证"真实超图匹配"相比"随机选边+伪置信度"的提升
- 调试工具：通过 verbose 输出排查状态提取/网络决策问题

**不推荐作为主实验**：置信度和超边选择都不符合方法论设计

---

### 3. `train_nethack.py` ⚠️ **简化版脚本（早期版本）**

**核心特性：**
- ⚠️ 与 `train_verbose.py` 类似，使用 HP伪置信度 + 随机选边
- ⚠️ 更少的 verbose 输出（使用 tqdm 进度条）
- ⚠️ pre_nodes / scene_atoms 提取更简化

**用途：**
- 早期快速验证训练循环是否跑通
- 不推荐用于正式实验

---

## 推荐使用方案

### 方案 A：单脚本消融（推荐）
**使用 `train_confmatch.py` + 环境变量控制特性开关**

```bash
# 完整版（所有特性开启）
TEDG_USE_TIME_DECAY=1 TEDG_USE_CONF_BRANCH=1 TEDG_USE_QUERY=1 TEDG_USE_LLM=1 \
TEDG_OUTPUT_DIR=results_full python train_confmatch.py

# 消融1：关闭时间衰减
TEDG_USE_TIME_DECAY=0 TEDG_USE_CONF_BRANCH=1 TEDG_USE_QUERY=1 TEDG_USE_LLM=1 \
TEDG_OUTPUT_DIR=results_no_decay python train_confmatch.py

# 消融2：关闭置信度阈值分支
TEDG_USE_TIME_DECAY=1 TEDG_USE_CONF_BRANCH=0 TEDG_USE_QUERY=1 TEDG_USE_LLM=1 \
TEDG_OUTPUT_DIR=results_no_branch python train_confmatch.py

# 消融3：关闭 Query 动作
TEDG_USE_TIME_DECAY=1 TEDG_USE_CONF_BRANCH=1 TEDG_USE_QUERY=0 TEDG_USE_LLM=1 \
TEDG_OUTPUT_DIR=results_no_query python train_confmatch.py

# 消融4：关闭 LLM
TEDG_USE_TIME_DECAY=1 TEDG_USE_CONF_BRANCH=1 TEDG_USE_QUERY=1 TEDG_USE_LLM=0 \
TEDG_OUTPUT_DIR=results_no_llm python train_confmatch.py

# 基线：仅多通道匹配（所有高级特性关闭）
TEDG_USE_TIME_DECAY=0 TEDG_USE_CONF_BRANCH=0 TEDG_USE_QUERY=0 TEDG_USE_LLM=0 \
TEDG_OUTPUT_DIR=results_baseline python train_confmatch.py
```

### 方案 B：对照实验
**使用不同脚本对比方法有效性**

```bash
# 主方法（train_confmatch.py）
python train_confmatch.py

# 对照1：随机选边 + HP伪置信度（train_verbose.py）
python train_verbose.py

# 对照2：无超图（纯 PPO baseline，需单独实现）
# ...
```

---

## 当前任务清单

### 立即执行（高优先级）
1. ✅ **修复 `blstats` 索引**（已完成）
2. 🔄 **在 `train_confmatch.py` 中添加详细中文注释**（进行中）
3. ⏳ **实现特性开关**（时间衰减/置信度分支/Query动作/LLM）
4. ⏳ **创建并行消融启动脚本**

### 后续优化（中优先级）
5. ⏳ 修复时间衰减：维护 `t_i` 历史记录
6. ⏳ 实现置信度阈值分支逻辑
7. ⏳ 实现 Query 动作集和奖励调整
8. ⏳ 实现 LLM 调用封装

### 验证测试
9. ⏳ Smoke test：运行 1 episode 验证可运行性
10. ⏳ 短期训练：运行 100 episodes 验证收敛趋势

---

## 结论

**当前应该使用的脚本：`train_confmatch.py`**

**原因：**
1. 唯一实现了真实超图匹配逻辑的脚本
2. 置信度来源符合方法论（超图匹配分数）
3. 已修复 `blstats` 索引错误
4. 输出到独立目录，不影响旧实验
5. 可通过环境变量扩展为支持消融实验

**下一步：**
- 在该脚本中添加详细中文注释（实验逻辑/输入输出/超参数）
- 实现特性开关（时间衰减/置信度分支/Query/LLM）
- 准备并行消融实验启动脚本
